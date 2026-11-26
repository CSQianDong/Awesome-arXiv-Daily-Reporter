# Knowledge-Base Gradient Descent: Self-Structuring Learning through Question Answering 

**Authors**: Masahiro Kato  

**Link**: [PDF](https://arxiv.org/pdf/2611.34841)  

**Abstract**: We consider the problem of constructing knowledge bases for applications based on large language models, in which the structure of a knowledge base determines how an application can retrieve and connect the information needed to answer a question. We study Self-Structuring Learning, a setting in which questions and expected answers guide the construction of a persistent knowledge base whose editable structure can be a graph schema or the organization of linked knowledge documents. We propose Knowledge-Base Gradient Descent, which diagnoses discrepancies between expected answers and responses obtained through the current knowledge base. At each round, the method proposes a structural edit, rebuilds the knowledge base, and adopts the edit when the learning loss decreases. The retained structure is evaluated on other questions about the same sources. By randomizing over candidate edits, we express the loss change of an edit as a derivative with respect to the probability of selecting it, and we evaluate joint edits as single candidates because their changes interact. Schema Descent instantiates the method for graph schemas. In experiments on large documents, each a bundle of research papers, the updated knowledge cards and graph schemas obtain higher mean Answer F1 on held-out generated questions than their initial structures, with schema gains in every document. On human questions, mean F1 rises for schemas and falls for cards: the updated structures serve the kind of question they were learned from. 

---
# TASK Memory: Beyond Text-Only Memory with Activation Steering and Knowledge Editing against Memory Hallucination 

**Authors**: Jinkun Zhao, Wenjun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2611.34762)  

**Abstract**: LLM agents increasingly rely on external memory, yet they still hallucinate about what they remember. Often the correct memory is retrieved and placed in the prompt, but the answer contradicts it or the memory is applied to the wrong question. We call these use-time memory hallucinations. Across five open models, we find that they arise from the conflict between parametric knowledge and external memory and take three forms: memory-ignoring, where the model returns to a stale fact in its weights (up to 46% of queries); memory-misapplying, where it attributes the memory to unrelated entities (up to 24%, and 49% when retrieval is combined with weight editing); and version confusion, where 53% of answers about revised facts assert the superseded value when both versions remain stored. We propose TASK memory, which goes beyond text-only memory by combining a Text note with Activation Steering and Knowledge editing that share one in-model address. The text holds the content. A removable knowledge edit supplies the address through its subject key and supports multi-hop composition. A pair of steering vectors decides, query by query, whether the model adopts the memory or keeps its own answer; the vectors store no content and control how the text memory is used. On frozen models under a pre-registered, test-once protocol, TASK reduces memory-ignoring hallucinations to at most 1.5% in all ten model-dataset cells and cuts neighbour misuse from 39.2% to 7.5% (+10.8 pp composite on CounterFact). It improves a 300-memory library over BM25 RAG by 16.4 pp, an advantage that holds at 3,000 memories, resolves fact revisions without fact IDs, and adds 7.5 pp of multi-hop accuracy on MQuAKE. 21 of 23 cross-model comparisons (8B-27B, four families) are significant, and TASK beats three recent context-steering methods in 9 of 10 model-dataset cells. 

---
# What Is a Theory Worth to an LLM Solver? Measuring, Inducing, and Merging Theories in Context 

**Authors**: Jun Gao, Shuchao Bi  

**Link**: [PDF](https://arxiv.org/pdf/2611.34751)  

**Abstract**: Mathematicians solve problems faster when they know the right theory. We ask how much a stated theory is worth to a large language model (LLM) solver, and whether a model can write that theory itself. In a length-matched protocol, adding the right theory to the context of Qwen3-14B raises pass@1 on pattern waiting-time problems from 0.37 to 0.83 and cuts the median tokens of correct solutions from 10.2k to 6.3k, while worked examples of the same length do not help. The gain shrinks as the solver and its token budget grow (+0.46, +0.26, +0.07), and at a long budget it comes almost entirely from finishing within budget: solutions that finish are correct 98--100\% of the time in every condition. To separate induction from recall, we plant unpublished theorems in custom probability families and measure how much of the oracle theory's gain an induced theory recovers. A strong open model, required to explore the training problems with code before writing, recovers 60\% of the gain on one planted family, 48\% on a second, and almost none on a third, where it finds the right form but not the condition that makes the form computable. No single induced note is complete, but a strong model that merges several notes recovers 61\% of the gain on the classic family, above the best single note (47\%). Theories are worth the reasoning they save, and models can find their pieces before they can assemble them. 

---
# From Shared Patterns to Individual Evidence: Two-Stage Reasoning for Emerging Entities in Temporal Knowledge Graphs 

**Authors**: Jinchuan Zhang, Lei Gao, Hongyu Jiang, Chong Mu, Chenang Cai, Ke Yan  

**Link**: [PDF](https://arxiv.org/pdf/2611.34708)  

**Abstract**: Temporal knowledge graphs (TKGs) continually introduce emerging entities whose interaction histories evolve from zero-shot debuts into sparse, bursty few-shot trajectories. However, current paradigms are fragmented: pattern-transfer methods lack catalog-wide event memory, whereas online memory models suffer from endpoint-role asymmetry that degrades unseen-answer ranking. Furthermore, sparse early interactions exhibit both long-range dispersion and local redundancy under skewed distribution. To address these challenges, we propose INSPIRE (Integrating Shared Patterns and Individual Evidence for Reasoning), a two-stage forecasting framework over a known catalog. In Stage I, INSPIRE learns an availability-aware shared-pattern predictor that fuses event-updated memory across the full candidate catalog while applying coordinated query- and catalog-side masking. Subsequently, Stage II freezes this predictor and optimizes a lightweight individual-evidence adapter (EA) combining textual semantic priors with correlation-weighted long-range event statistics under support- and role-balanced co-answer supervision with prior-odds debiasing and soft residual alignment. In addition, we decompose evaluation by endpoint roles and track longitudinal emerging-answer cohorts via query-micro and answer-entity macro metrics. Across four benchmarks based on the ICEWS and GDELT event datasets, INSPIRE improves Emerging MRR by 16.73%-88.83% over the strongest baselines. Relative to paired Stage I predictors, EA further increases Emerging MRR by 8.28%-32.63%, Answer-new MRR by 7.81%-41.24%, and cohort macro MRR by 9.64%-39.94%, gains attributed to semantic priors and candidate-level score competition suppressing mismatched established competitors without zero-support prior shortcuts. The official code is publicly available at this https URL. 

---
# A Failure-Driven Data Flywheel for Post-Training Vision-Language-Action Models 

**Authors**: Yuhao Luo, Zihao Sheng, Zilin Huang, Zhengyang Wan, Boyue Wang, Ruolin Yang, Shuai Wang, Sikai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.34698)  

**Abstract**: Pretrained vision-language-action (VLA) models rarely satisfy all behavioral requirements of a new platform, and teams without a large deployed fleet lack the data to address the resulting failures. We propose a failure-driven data flywheel for post-training such models. It constructs target scenarios absent from the available data by inserting task-relevant entities into reconstructed real-world scenes, and evaluates the model against each behavioral requirement of the target task. Failure rates form a failure map across task conditions and difficulty levels, which is re-evaluated after each round and assigns larger data generation quotas to conditions where the model still fails. We instantiate the framework in autonomous driving for occluded-pedestrian emergence and yielding at stop-sign-controlled intersections, and evaluate it on two VLA models and one non-VLA planner in simulation and on a real vehicle. For occluded-pedestrian emergence, five rounds of post-training improve the critical-phase collision-free rate on the sealed test set from 0.261 to 0.579 for AutoVLA and from 0.390 to 0.765 for ReCogDrive, while the unnecessary braking rate in paired control scenarios never exceeds 0.013. On a full-scale vehicle without real-world training, post-trained ReCogDrive achieves 4 of 5 safe passages, versus 1 of 5 for the base model. Fine-grained evaluation reveals a pattern obscured by the collision-free rate alone: both VLA models learn anticipatory deceleration before reaching the occluding vehicle, while braking after pedestrian emergence remains limited. For stop-sign yielding, the stop-before-line rate rises from 0.100 to 0.897 for AutoVLA and from 0.000 to 0.667 for ReCogDrive. With sequential training from the pedestrian-capability models, it reaches 0.599 and 0.358, respectively, while the pedestrian collision-free rate decreases by at most 0.032. 

---
# REE: A Scientific Research Entrance Exam for AI Agents 

**Authors**: Zeqi Zhou, Minghao Liu, Jiarui Liu, Qingcheng Zeng, Rui Yang, Tianyu Liu, Zetao Yang, Wenqiang Zu, Enduo Zhao, Heng-Jui Chang, Lun-Hsin Kuo, Meng-Fu Chen, Haokai Zhao, Weijia Zhang, Jen-Tse Huang, Stephen Huang, Eli Cheung, Zhijing Jin, Mona Diab, Weihao Xuan  

**Link**: [PDF](https://arxiv.org/pdf/2611.34646)  

**Abstract**: Autonomous research agents are emerging as a promising approach to scientific problem-solving across disciplines, owing to their growing ability to improve a solution through repeated experiments. Evaluating this ability requires benchmarks that pose open-ended research problems and measure improvements over established scientific methods. We introduce REE (Research Entrance Exam), a benchmark that tests AI agents on scientific research. It consists of 100 open-ended research problems spanning five broad scientific areas and 23 domains. To ensure high quality, every problem is drawn from active research in its field and individually reviewed by domain experts. Each problem is posed as a task of iterative improvement: an agent has up to 12 hours to improve a working starter solution, and REE scores the result on a common scale against an established expert solution evaluated under comparable conditions. By this measure, two of the four frontier agents we evaluate, Claude Opus 5.5 and GPT-6.1 Sol, outperform the expert solutions on 86% and 81% of problems, across all five scientific areas. Their runs also show how they do so: both agents surpass the expert solutions within the first two hours on average, and their best solutions mostly tailor or combine established methods of the field. REE thus provides a unified framework for evaluating autonomous scientific research, capturing both improvements over established methods and the research processes behind them. 

---
# ZeroHero: Training Small Language Agents from Scratch 

**Authors**: Nikita Sorokin, Zoya Volovikova, Alexey Skrynnik, Aleksandr Panov  

**Link**: [PDF](https://arxiv.org/pdf/2611.34635)  

**Abstract**: Large pretrained language models provide capable policies for interactive tasks, but how much model capacity is necessary for a specialized language agent? We introduce ZeroHero, a framework for training small language agents (SLAs) from scratch. In each environment, a large language model acts as the expert, and we collect its successful trajectories. We pretrain each SLA, which has fewer than 50 million parameters, to predict the expert's responses given the interaction history. We then post-train the agent on its own interactions using proximal policy optimization (PPO). During post-training, reasoning alignment supplements environment rewards with expert supervision on the language the agent generates alongside its actions. We evaluate the agents in three multistep environments where acting requires both understanding and generating language. In Crafter, agents interpret textual observations and generate reasoning before acting. In WebShop, they translate shopping instructions into search queries. In Alfworld-Dialogue, they ask clarifying questions and use the answers to guide household actions. The agents approach expert performance on WebShop and surpass their respective demonstration experts on Crafter training rollouts and held-out Alfworld-Dialogue tasks. An interactive demo is available at this https URL. 

---
# HoMe: Toward Practical Harness Optimization of Memory Systems for AI Assistants 

**Authors**: Yifei Xu, Li Yu, Zhaoyang Liu, Ce Wang, Sen Huang, Yanyan Shen, Yun Lin, Linpeng Huang, Hong Mei  

**Link**: [PDF](https://arxiv.org/pdf/2611.34624)  

**Abstract**: The behavior of memory systems for long-running AI assistants is governed by a harness, which consists of the code, prompts, and configurations for memory construction, maintenance, retrieval, and answer generation. However, optimizing the harness of a memory system remains largely manual, and unrestricted editing alone does not ensure structural improvements. In this paper, we introduce HoMe, an evaluation-driven framework that iteratively evaluates candidate harnesses and derives each new candidate from any previously evaluated harness by editing any part of it. HoMe combines session-level memory versioning with execution traces to identify shared failure mechanisms across cases and localize them to the responsible sessions and harness components. It encourages early structural exploration through candidate gating and reduces evaluation cost through smoke tests, adaptive sampling, and the reuse of constructed memory. We validate HoMe with the open-source memory system ReMe on two large-scale long-horizon memory benchmarks, LongMemEval-S and BEAM. With six formal candidate evaluations per benchmark, after a full initial evaluation, HoMe discovers harnesses that improve end-task performance over the originally strong harness while consuming far fewer query-time tokens. The code is available at this https URL. 

---
# Do We Really Need an LLM for Weight-Space Programming? A Deterministic Forward-Only Paradigm and Its Capability Boundaries 

**Authors**: Lin Yao  

**Link**: [PDF](https://arxiv.org/pdf/2611.34612)  

**Abstract**: Recent work on Weight-Space Programming uses LLMs to edit neural weights with forward passes only, conflating LLM scripting with the underlying algebra of forward-only editing. We show that for core structured editing regimes, an LLM optimizer is unnecessary, and introduce Forward Weight-Space Programming (FWSP): a deterministic pipeline combining probing, localization, stratified operators (O1-O4), automatic blind routing (83.3% vs. 33.3% for a 27B LLM with the same tool library), and paired verification with zero additional per-token computation. For reasoning LLMs, we propose Protected FisherGate, a closed-form covariance-whitened readout edit orthogonal to normal-reasoning states: on DeepSeek-R1-0528-8B, a single frozen math-calibrated patch achieves 84.2% HumanEval pass@1 (1K budget; 75.4 +/- 9.4% across 3/5 Stage-4-passing subsets), beating 512-token-reserved budget forcing by +14.6 pp (-23.4% tokens) while transferring across HF-to-vLLM backends on full GSM8K and MATH-500. Across five modalities and thirteen models (50K to 27B dense and FP8 MoE), we further prove and verify two operator boundaries, subspace entanglement and frozen representation span, where FWSP-localized LoRA bridges forward-only and gradient adaptation. 

---
# Procedure Inference as Memory: Learning Recruiter Decision Procedures from OKF-Structured Episodes in Multi-Session Hiring Assistants 

**Authors**: Praveen Kumar Bodigutla, Yizhi Zhang, Sai Vivek Kanaparthy, Arup De  

**Link**: [PDF](https://arxiv.org/pdf/2611.34576)  

**Abstract**: Multi-session hiring assistants must infer evolving recruiter preferences and decision patterns across projects and time. We introduce Open Knowledge Format (OKF)-indexed episodic memory to organize interactions and retrieve related events through their lineage. We also introduce Procedure Memory, which learns reusable decision procedures from these episodes through prompted reflection or reinforcement learning. We compare these approaches with no memory, full-history context, and conversation search, and evaluate Jev-class models for index navigation and GPT-5.6 Luna as a lower-cost alternative to GPT-5.4.
In our evaluation, OKF-indexed retrieval achieves low-priority preference F1 scores of 0.93-0.95 at medium and high preference overlap between hiring projects. It significantly outperforms full-history context and conversation search in both settings, and no memory at high overlap, while using 69% fewer context tokens than full history. Procedure Memory reduces conversational turns by up to 39% relative to no memory. Qualitative and long-horizon evaluations provide evidence of temporal reasoning, with RL-learned procedures achieving 60% on both correct abstention and exact-field identification. Jev-class models improve conversational efficiency but more frequently miss temporally relevant evidence, highlighting the importance of evidence selection.
On PersonaMem-v2, Procedure Memory with OKF retrieval achieves 50.2% accuracy versus 48.6% for full-history context and 29.0% for no memory, using 90.5% fewer reader input tokens than full history. GPT-5.6 Luna performs comparably to GPT-5.4 at lower cost. These findings highlight the value of structured episodic memory and learned decision procedures for efficient personalization and temporal reasoning. 

---
# On Utilization of Edge-AI for condition monitoring towards predictive maintenance: A use case in dairy processing using explainable AI 

**Authors**: Rahul Umesh Mhapsekar, Norah O'Shea, Omer Ali, Steven Davy, Lizy Abraham  

**Link**: [PDF](https://arxiv.org/pdf/2611.34563)  

**Abstract**: The Industrial Internet of Things (IIoT) and Artificial Intelligence (AI) are driving Industry 4.0, enhancing productivity, automation, and efficiency in manufacturing systems. This study develops an Edge-AI-based condition monitoring framework, as a step towards predictive maintenance, for a rotary valve in a dairy processing pilot plant, using vibration data collected via a low-power IIoT sensor node and processed on an edge device. 

---
# EgoUseBench: Benchmarking Computer-Use Agents under Evolving First-Person Contexts 

**Authors**: Yurun Chen, Biao Yi, Qihang Yu, Keting Yin, Shengyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.34466)  

**Abstract**: AI wearables, such as smart glasses, extend the context available to agents beyond digital environments to users' physical surroundings. With these devices, computer tasks can originate from users' ongoing activities rather than solely from explicit instructions. Through first-person observations, agents can identify objects and activities relevant to user requests and ground digital tasks in the surrounding context. We call this interaction paradigm Ego Use. However, first-person observations are limited by the wearer's viewpoint and may provide incomplete or ambiguous evidence of ongoing activities. As new observations arrive, agents may need to revise their understanding of the physical context and user intent, and reassess whether prior digital actions remain valid. Whether current agents can reliably handle these changes remains insufficiently evaluated. We introduce EgoUseBench, a benchmark for evaluating Ego Use across three interaction settings with different levels of first-person context. Computer-Use relies on explicit instructions, Offline Ego Use provides prerecorded first-person context before execution, and Online Ego Use provides continuous observations as user intent evolves. Based on over 74 hours of real-world first-person recordings, we construct 693 tasks across the three interaction settings. To capture task evolution, we track first-person observations, user requirements, and website state separately. We further group tasks by how information changes during execution, resulting in three types: Graft (revised user requirements), Refresh (new first-person observations), and Clarify (agent-initiated information seeking). We evaluate advanced models and find that average success rates in Online Ego Use are 26.3% lower than in Computer-Use. This result reveals the difficulty of keeping task execution aligned with evolving first-person context and user intent. 

---
# vSFT: Learning to Verbally Tune Agents at Test Time 

**Authors**: Qishuai Wen, Rui Chen, Haiwen Li, Zhiyuan Huang, Qiwen Gu, Bingjie Gao, Geng Li, Jing Tang, Xiangxiang Chu, Chun-Guang Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.34440)  

**Abstract**: Large language model agents are increasingly improved via agentic reinforcement learning, which can be on-policy but suffers from expensive rollouts and sparse rewards, limiting its application at test time. In this paper, we present a verbally Supervised Fine-Tuning (vSFT) framework that enables the conjunction of dense feedback in natural language and weight updates at test time. Specifically, our vSFT forms a closed loop between an agent and a supervisor via a hypernetwork, in which the supervisor generates verbal supervision from the agent's failed trajectories, and the hypernetwork converts that supervision into low-rank weight updates to tune the agent. We validate that this test-time in-parameter evolving process outperforms both its in-context counterparts and prevalent reinforcement learning methods. In particular, our vSFT using Qwen3-4B outperforms GRPO and Reflexion by 8.7% and 13.9% in success rate on ALFWorld, and by 4.8% and 18.2% on WebShop, respectively. We believe that learning how to adapt a model, rather than an adapted model, offers a promising path towards more capable agents that improve themselves. 

---
# Evaluating Aviation Safety Knowledge Extraction with SmolLM2: From Zero-Shot to Retrieval-Grounded Prompting 

**Authors**: Aziida Nanyonga, Mustafa Tahsin Yilmaz, Amir Hussain, Ugur Turhan, Graham Wild  

**Link**: [PDF](https://arxiv.org/pdf/2611.34432)  

**Abstract**: Aviation accident narratives contain safety knowledge that is difficult to capture using structured occurrence fields alone. This study evaluates prompting-based knowledge extraction across four dimensions: concept accuracy, verbatim evidence grounding, structural compliance, and directed relation extraction. SmolLM2-1.7B-Instruct was evaluated under zero-shot, one-shot, fixed few-shot, and retrieval-grounded prompting using 300 National Transportation Safety Board narratives from 2000-2020, containing 3,763 reference concepts and 467 directed relations. Evaluation combined structural checks, verbatim evidence matching, expert-reviewed semantic matching, and paired bootstrap confidence intervals. No prompting strategy performed best across all dimensions. Retrieval-grounded prompting achieved the highest concept micro-F1 of 32.27%, although its advantage over fixed few-shot prompting was uncertain (95% CI: -0.92 to 2.40 percentage points). One-shot prompting achieved the highest relation F1 of 6.95%, identifying directed relation extraction as the principal SmolLM2 bottleneck. Compared with fixed few-shot prompting, retrieval improved verbatim grounding (59.45% vs. 54.02%) but reduced complete structural compliance (33.3% vs. 42.7%). Verbatim matching establishes source presence rather than evidential relevance. An exploratory comparison using 35 benchmark narratives showed that Qwen3-8B achieved higher concept micro-F1 across all prompting conditions and higher relation F1 under demonstration-based prompting, reaching 51.37% and 34.62%, respectively, under one-shot prompting. The small subset limits generalization. These findings support evaluating semantic correctness, source traceability, structural reliability, and directed relation accuracy separately. 

---
# Agentic Waterfall: An Intent-Driven Information Discovery Framework for Reclaiming User Control in the Attention Economy 

**Authors**: Yige Luo, Shining Zhang, Jikun Liu, Hongjiao Qiang, Ran Guan  

**Link**: [PDF](https://arxiv.org/pdf/2611.34407)  

**Abstract**: We formalize Agentic Waterfall Feed, a new problem of personalized information discovery in which users actively direct recommendations and the system adapts to evolving needs inferred from both explicit requests and browsing behavior. We address this problem with the Agentic Waterfall framework, whose language-model controller maintains persistent user state and coordinates content retrieval, ranking, and state updates. To evaluate systems on this task, we contribute the Agentic Waterfall Dataset and a replay-based evaluation protocol. The dataset combines real content metadata with simulated interaction histories. The protocol compares systems under shared histories using relevance scores and blind judgments of how well recommendations satisfy current user goals and constraints. Agentic Waterfall remains competitive with strong baselines on conventional recommendation tasks across five public benchmarks. On the Agentic Waterfall Dataset, it outperforms the strongest baseline in both evaluations, with relative improvements of up to 16.1%. 

---
# EvoLink-Bench: A Benchmark for Evaluating Self-Evolving Agents in Long-Horizon Interdependent Task Streams 

**Authors**: Wei Chen, Xiaodong Lu, Weihao Shen, Gongxun Li, Xingyu Guo, Qihang Sun, Fuwei Zhang, Zhao Zhang, Yikun Ban, Shuang Li, Deqing Wang, Fuzhen Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2611.34400)  

**Abstract**: Large language model (LLM) agents are evolving from passive chatbots into autonomous systems that operate continuously in dynamic environments. Such agents need not only solve current tasks but also learn from past interactions to improve future performance. However, existing streaming benchmarks often rely on short task sequences or largely independent tasks, providing limited evidence of whether agents can effectively reuse experience when later tasks depend on earlier outcomes. To address this gap, we introduce \textbf{EvoLink-Bench}, a benchmark for evaluating self-evolving agents in long-horizon, interdependent task streams. It covers daily office work, multi-turn dialogue, and AutoResearch-style workflows, with persistent workspaces and dependencies across tasks. The benchmark includes two complementary levels: \textbf{L1}, which evaluates the acquisition and revision of language-expressible knowledge, and \textbf{L2}, which assesses the development of task-level capabilities through execution and feedback. Its evaluation further distinguishes within-task recovery, persistent-state updates, and experience reuse in later tasks. We evaluate five agent frameworks across three backbone configurations under three persistence conditions, covering 19,635 scored task slots from selected evaluation cohorts. Results show that no framework consistently performs best across both levels, retries recover only a limited proportion of failed tasks, and persistent-state updates do not necessarily improve subsequent performance. Furthermore, the effects of evolution differ between knowledge- and capability-level tasks, suggesting that these two forms of evolution require different learning mechanisms. Performance gains also vary across domains and task streams, with improvements on some tasks not necessarily extending to others. 

---
# MeterRouter: Budget-Aware Dynamic Routing for Multi-Agent Systems 

**Authors**: Yue Xu, Leikun Yin, Ruiming Min, Yufeng Lai, Justin Andrew Johnson, Sitao Luan, Xiaotong Shen  

**Link**: [PDF](https://arxiv.org/pdf/2611.34327)  

**Abstract**: Multi-agent systems (MAS) built on large language models (LLMs) outperform single-agent systems, but inter-agent communication incurs substantial token overhead. While assigning a large LLM to every agent is common, it results in inefficient resource utilization, leaving the system far from the Pareto frontier. The cost of answering a query is set jointly by how many agents are recruited, the role and LLM assigned to each, and communication topology among them. Most existing routers treat cost only implicitly rather than as an explicit optimization constraint. A fixed accuracy-cost preference also cannot express how much a caller is willing to pay for a particular query. We introduce MeterRouter, a budget-aware dynamic router that incrementally synthesizes a collaboration graph by selecting each agent's role, LLM, incoming connections, and termination conditioned on the query and a per-query monetary budget. We initialize the router by imitating successful execution graphs, then jointly refine its decisions with a variant of Group Relative Policy Optimization (GRPO) whose reward favors correct answers and penalizes spending and budget overruns. Across five benchmarks and a three-tier LLM pool, MeterRouter is (1) accurate, achieving performance comparable to the strongest MAS; (2) economical, reaching that accuracy at 82% lower cost than the strongest MAS router and extending the accuracy-cost frontier beyond the ceiling of single-agent systems; and (3) controllable, adapting its spending to the specified budget while rarely exceeding it. 

---
# Crucible: Evaluating LLM Agents in Executable Chemistry Research Environments 

**Authors**: Yinjun Liu, Xinye Li, Ruisen Peng, Lang Gao, Zixiang Xu, Zirui Song, Xiuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.34254)  

**Abstract**: Large language models (LLMs) perform well on chemistry knowledge benchmarks, which mainly assess question answering and isolated tasks. Chemical research, however, requires using specialized tools and interpreting their outputs to obtain chemically valid results. To address this gap, we introduce Crucible Bench to evaluate this ability across all four stages of the design-make-test-analyze (DMTA) cycle. The benchmark contains 1,078 tasks in 12 task families curated from 18 published datasets, covering molecular design, synthesis planning, experimental execution, closed-loop optimization, simulation, and analytical characterization. Agents access 75 domain-specific tools through the Model Context Protocol in seven isolated environments. Sandboxed graders verify the resulting synthesis routes, experimental protocols, and other outputs. Laboratory tasks use robotic simulators or previously measured experimental data. We evaluate 14 LLMs as agents and find that the best achieves an overall score of 0.545, with synthesis planning and closed-loop optimization emerging as the hardest domains. Overall performance does not predict optimization performance well: the model ranked twelfth overall achieves the highest score in closed-loop optimization. Error analysis also shows that agents often submit chemically invalid results despite running their tools successfully. Stronger models benefit more from tool access, but reliable use of chemical software remains a challenge that Crucible Bench can track as models improve. 

---
# MedSkill-E$^{2}$: Evaluating and Evolving Skills for Medical Agents 

**Authors**: Jingjing Hua, Shaocong Wei, Jianghan Shen, Junzhi Ning, Min Cen, Cheng Tang, Ziyan Huang, Jiepeng Li, Fengshuo Liu, Kangkun Mao, Mouxiao Bian, Yirong Chen, Ming Hu, Jiyao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.34250)  

**Abstract**: Large language model (LLM) agents increasingly use skills to access domain knowledge and complete real-world tasks. We observe that even when a skill specifies the rules and procedures needed for a task, agents may fail to apply them effectively during autonomous execution, a phenomenon we call skill Enable But Underuse (EBU). This issue is particularly consequential in medical tasks, where overlooking a single critical constraint can yield a plausible but incorrect result. To evaluate and address EBU, we introduce MedSkill-E$^{2}$, a framework for evaluating and evolving skills for medical agents. We construct EBU-Bench, a challenging benchmark covering 12 medically relevant task groups and associated skills, to evaluate agents' ability to use available skills autonomously. We quantify the EBU gap as the difference in skill-use accuracy between autonomous use and explicitly guided reading and application of the same skill. On EBU-Bench, all evaluated model-harness combinations exhibit EBU, with a mean gap of 43.94 percentage points (pp). For the same model, skill-use accuracy under autonomous use differs by up to 38.96 pp across harnesses. Motivated by these observations, we propose EBU-Evolve, which uses execution feedback from autonomous and explicitly guided skill use to distill effective guidance into reusable skill instructions. Relative to their pre-evolution versions, the revised skills improve skill-use accuracy during autonomous execution by 33.33 to 55.17 pp and continue to yield improvements when transferred across multiple models and harnesses. Our results show that autonomous skill-use accuracy varies across harnesses with the model and skill held fixed, and model rankings can change. EBU can persist even after a skill is opened. EBU-Evolve improves autonomous skill-use accuracy on unseen tasks within the same skill, with gains transferring across models and harnesses. 

---
# Navigent: A Geometry-Grounded Agentic Framework for Vision Language Navigation 

**Authors**: Xinyu Zhou, Weiren Cai, Yitong Chen, Shenghai Yuan, Zihang Mu, Boyu Ma, Jiahua Dong, Yanshuo Lu, Jiaqi Bai, Jianfei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2611.34249)  

**Abstract**: Vision-and-Language Navigation (VLN) with general-purpose multimodal large language models (MLLMs) offers a flexible alternative to task-specific navigation policies, while reliable long-horizon execution remains challenging. Agents must track navigation progress, detect discrepancies between commanded and actual motion, and recognize when movement has passed relevant landmarks or entrances. We present Navigent, an agentic VLN framework built around a frozen MLLM and a Geometry-Grounded Execution Critic (GEC). GEC performs two complementary checks: action-to-motion consistency, which evaluates measured displacement against commanded movement, and motion-to-landmark consistency, which tracks agent-selected visual anchors and entrance candidates in world coordinates. Detected discrepancies interrupt the remaining action sequence and trigger reobservation, local adjustment, or node-based recovery. Historical visual memory supports progress tracking across extended routes. An offline self-improvement procedure further analyzes successful and failed training trajectories to refine navigation prompts and GEC logic while keeping model weights fixed. On the R2R-CE validation-unseen split (1,839 episodes) and RxR-CE validation-unseen split(3,669 episodes), Navigent achieves 84.0%/86.0% SR and 72.9%/69.1% SPL, respectively, demonstrating the value of geometry-grounded execution verification for reliable agentic navigation. Project Page: this https URL Code: this https URL 

---
# EXPOSURE: Explaining Sources of Uncertainty in Reinforcement Learning 

**Authors**: Kiana Farhadyar, Yannick Vogt, Philipp Laur, Ayça Çelik, Iman Nematollahi, Gabriel Kalweit, Maria Kalweit  

**Link**: [PDF](https://arxiv.org/pdf/2611.34248)  

**Abstract**: Shapley values in reinforcement learning are used to explain which observation features drive an agent's behavior and expected return, but not its certainty. Explaining certainty requires understanding how observation features influence different sources of return uncertainty, and Shapley values need the agent's prediction for every subset of visible features. We propose EXPOSURE, which uses one mask-aware belief network for both action selection and Shapley explanations. It explains the variance of the return, split into an observational part due to missing features and an inherent part due to the environment's own randomness, and the entropy of its belief over the masked features. In tic-tac-toe and Minesweeper, where exact Shapley values are computable, EXPOSURE gives more accurate explanations of expected return and its uncertainty, while uniquely explaining belief uncertainty. 

---
# Did the Model Forecast It, or Already Know It? Evaluating Large Language Models for Novel Drug-Target Prioritisation with a Temporal Knowledge-Graph Oracle 

**Authors**: Pui Chung Siu, Yuan Liang, Claudia Cabrera, Mani Mudaliar, Arkaitz Zubiaga  

**Link**: [PDF](https://arxiv.org/pdf/2611.34247)  

**Abstract**: AI systems increasingly help decide which drug targets enter costly clinical trials. We recast each target-disease hypothesis as a dated event, the pair's first clinical trial. This lets us test whether a language model can forecast such events after its training cutoff, rather than recall events before it. We rebuild a dated biomedical knowledge graph as it stood at the end of each year from 2020 up to its newest version. We then ask six open language models, with and without this graph, which targets will enter a first trial for a disease the next year. Without retrieval, the models rank about as well as trial history, a count of each target's past trials in the same therapeutic area, also after their training cutoffs. Evidence dated before the decision adds at most a small gain, before or after the models' training cutoffs. The newest graph raises five of six models clearly above that baseline, but its paths usually include the answer: the pair's own trial, dated after the decision. Evaluations of target prioritisation should date both features and labels, with evidence from before the decision and outcomes after it. 

---
# PROVE: Mitigating Long-Horizon LLM Collusion with Reinforcement Learning from Verified Evidence 

**Authors**: Haochen Luo, Jie Fu, Mukai Li, Yuqi Wang, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.34215)  

**Abstract**: When LLM agents verify each other's work over a long interaction, they drift into collusion: they come to accept work without the evidence that the verification protocol requires. Collusion is absent at the start and builds up episode by episode, which we trace to accurate feedback accumulating in memory. An agent that follows the rule is told that its verdict was wrong whenever its peer was right, so breaking the rule is the reward-maximizing choice, and agents talk themselves into it in their private reflections. Existing mitigations leave the agents unchanged; we train the behavior out. PROVE (Protocol Reinforcement from Verified Evidence) is a multi-agent reinforcement learning method that rewards verdicts by the evidence behind them rather than by their outcome, and credits every call an agent makes, including the calls that solve its own task, with the outcome that call controls. On Qwen3.6-27B, a LoRA adapter trained on six-episode interactions takes the collusion rate on held-out task sequences at the original ten-episode horizon from 65.0% to 1.0%, with no detectable change in task success. Collusion that converged in every interaction now converges in none, and the gain holds four episodes beyond the longest interaction seen in training. The trained 27B model colludes less than every model evaluated in the original study, closed-weight frontier models included. Code is available at this https URL. 

---
# SOLO: Towards Sample-Faithful and Detectable LLM Watermarking under Repeated Sampling via Content-Anchored Single-Owner Contexts 

**Authors**: Zhongliang Guo, Yan Lin, Yifei Qian, Shuai Zhao, Luwei Xiao, Daizong Liu, Chun Pong Lau  

**Link**: [PDF](https://arxiv.org/pdf/2611.34190)  

**Abstract**: Providers watermark the outputs of large language models, while applications sample many responses per prompt for self-consistency, best-of-n selection, and uncertainty estimation. We show that deployed configurations of distortion-free watermarks, which deduplicate keyed contexts only within a response, couple the responses of a request: answer templates make all responses query the watermark with the same context before the answer, so collisions across responses, usually assumed rare, become the norm. Key pools reduce or avoid the coupling at the cost of multiple testing, and the exact remedy of a context history shared across the request leaves many positions unwatermarked. We propose SOLO, which keeps a request-scoped context history but anchors its keyed contexts on content selected by min-hash, so that siblings rarely compete for a context and random edits stay local. Its responses are jointly distributed as independent unwatermarked responses, each detectable with a single key. On five language models from three families and three benchmarks, SOLO preserves self-consistency and uncertainty estimation and, at a 1% false positive rate, detects more responses than four-token key pools on most tasks, also after randomly editing 10% of the tokens, and more than the context history on most clean tasks. Code is available at this https URL. 

---
# Letting Models Manage Their Own Context: Where We Stand 

**Authors**: Alessio Devoto, Avinash Vem, Tara Safavi, Sanjay Kariyappa, Yu-Hsuan Tseng, Paul Furgale, Ricardo Silveira Cabral  

**Link**: [PDF](https://arxiv.org/pdf/2611.34147)  

**Abstract**: Agent harnesses today manage context on the model's behalf: when the window fills, the harness triggers a compaction that summarizes the history. We study the alternative, \emph{model-driven} context management, in which the model itself decides when and what to compact through a context API. We gave models control over their own context in \nooa, NVIDIA's open source agent harness, and compared the two strategies on document research and machine learning engineering across context budgets from 8K to 512K tokens. Models that use the context API effectively reach higher accuracy, often with fewer input tokens, especially at tight budgets: \textbf{GPT 6.1 nearly triples its score at a 32K budget} on OfficeQA Pro v2 (21\%~$\rightarrow$~56\%), and \textbf{Kimi gains with up to 41\% fewer input tokens}. Models differ widely, however, in how often they use the API, how aggressively they compact, and when they start compacting, and prompt changes shift that timing. Finally, we show that \textbf{inference engines can make model-driven edits cheaper}: by reusing the KV cache of the unchanged suffix after an edit, \textbf{we cut prefill compute by up to 15\% per request with no measurable change in accuracy}. Together, these results point to a new co-design frontier: models trained to manage their own context, served by inference engines that support KV cache editing. 

---
# Not Every Input Needs Everything: Unifying Expert Gating, Model Cascades, and Agent Orchestration 

**Authors**: Dat Tat Mai, Hung Tran Quy, Thai Viet Pham, James Jin Kang  

**Link**: [PDF](https://arxiv.org/pdf/2611.34112)  

**Abstract**: Executing monolithic neural weights, frontier foundation models, or entire multi-agent workflows uniformly on every input incurs substantial computational, latency, and monetary overhead. Conditional computation mitigates this cost by deciding dynamically which internal experts, model endpoints, or specialized agents to invoke per request. Yet token-level gating in sparse neural architectures, query-level model cascades, and task-level multi-agent coordination have largely progressed in disconnected research silos, obscuring recurring system failures and transferable fixes. In this systematic review, we synthesize 2,289 studies published between 2017 and September 2026 (selected from 13,884 unique records) across 20 methodological families under a unified two-step inspect-then-allocate framework. Discrete hard selection accounts for 70%, 83%, and 91% of token-, query-, and task-scale literature respectively, and computational overhead appears as a top operational bottleneck across every layer. Engineering safeguards remain sharply asymmetric: intra-layer gating prioritizes load balancing while rarely incorporating uncertainty estimation, whereas model selection and agent orchestration rely on confidence thresholds and budget rules while confronting miscalibration and cascading errors. At the same time, publication volume is accelerating fastest in agentic task delegation, where routing mechanisms are least often isolated experimentally and methodological bias risk is highest. 

---
# Budget, Readout or Reasoning? Where Small Reasoning Models Lose Value in Auction Bidding 

**Authors**: Mahbod Khaleti, Seyed Reza Tavakoli, Baran Hoseini, Alireza Noori, Amir Amini, Koorosh Aslansefat, Marjan Alirezaie, Ali Habibnia  

**Link**: [PDF](https://arxiv.org/pdf/2611.34068)  

**Abstract**: Ad platforms and marketplaces show bidders a bid landscape, the estimated chance of winning at each candidate bid, and increasingly let agents built on reasoning language models choose the bid. We give six open reasoning models of 4B to 14B parameters such landscapes, with win curves from eBay closing prices and from a simulated advertising market (AuctionNet), and score their bids on held-out auctions against an exact calculator. Because the agent is handed the estimates, the lost value can be split by cause. Two failures belong to the models. They pass over the best option even when the values written in their own trace already rank it first, and their wrong answers lean toward the highest bid on the menu. The lean is about the bid's size, not its place in the list, and it survives sampling. At 768 forced tokens on eBay, under pay-your-bid pricing, the wrong answers cost bidders 7-17% of the surplus the calculator's bids earn. Arithmetic errors are rare. Two further losses come from operator choices. A token cap cuts traces short: allowed to stop on their own within 4,096 tokens, five models recover 91-100% of the calculator's lead over a model that does no reasoning. Asking for the answer while the reasoning block is still open misreads it: closing the block first lifts the weakest model from 42% to 84% on eBay and from 24% to 81% on AuctionNet. Finally, an audit that flags bids disagreeing with the trace's own numbers, and overrides them with the calculator's bid, raises the share of the calculator's lead recovered at 768 forced tokens from 42-85% to 84-100% on eBay under nested cross-validation, and the same reader settings transfer from eBay to AuctionNet. 

---
# Few-Shot Adaptation of Vision-Language Models via Agentic Model Editing 

**Authors**: Yusheng Zhao, Xiao Luo, Zhiping Xiao, Ming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.34036)  

**Abstract**: Few-shot adaptation transfers pretrained Vision-Language Models (VLMs) to new domains using a small set of labeled images. Most methods fit prompts, representations, or classifiers within an adaptation procedure specified in advance. Yet the visual distinctions needed for recognition can differ substantially across domains, motivating the question: Can an LLM agent use few-shot examples and model observations to design the adaptation procedure itself? We propose Agentic Model Editing (AME), an agentic framework that jointly designs input transformations, encoder edits, feature computation, and classifier fitting as an executable, domain-customized adaptation program. The agent inspects images, class semantics, model parameters, intermediate representations, and activations to formulate hypotheses about the target task. It implements and tests candidate programs on the support set, with aggregate validation accuracy guiding refinement and selection. Each program fits a complete domain model that is fixed before testing. The agent and its tools are shared across domains, while each domain receives its own adaptation program. Across eleven datasets with CLIP ViT-B/16, AME achieves the highest average accuracies of 81.2% without back-propagation and 81.9% with back-propagation among the compared methods. Further experiments demonstrate applicability across VLM backbones and LLM agents. 

---
# Honest but Wrong: Formalizing Retraction in the Ethics of Speech Acts 

**Authors**: Benjamin Icard, Gauvain Bourgne, Jeanne Bonnaventure, Jean-Gabriel Ganascia  

**Link**: [PDF](https://arxiv.org/pdf/2611.34011)  

**Abstract**: A speaker who asserts honestly may later discover that its assertion was false. Such erroneous truths are not lies, yet they raise the question of whether, and on which moral grounds, the speaker must retract them. Using the FIPA-ACL performative disconfirm, we extend a belief-based framework that evaluates speech acts under deontologism and consequentialism to the case of retraction. We prove that the honesty duty of deontologism, which bears on utterances when they are made, never requires retracting an erroneous truth, and add a persistent one, which forbids letting the addressee believe what the speaker no longer believes. On a legal-advice case implemented in Answer Set Programming, we show that deontologism and consequentialism both require the retraction, but that consequentialism no longer does once it comes too late to prevent harm. 

---
# PatternMiner: Automated Mathematical Discovery Through LLM-Guided Training and Interpretation of Small Models 

**Authors**: Michael Ruofan Zeng, Junaid Hasan, Xuanyu Yang, Emily Xia Meng, Henry Kvinge  

**Link**: [PDF](https://arxiv.org/pdf/2611.33966)  

**Abstract**: There are several approaches to AI-driven mathematical discovery. One of these involves the use of large language models (LLMs) to resolve well-specified conjectures via text-based reasoning, while another uses narrow models and interpretability for more open-ended, human-in-the-loop exploration using object-based mathematical data. The latter is limited in its ability to move beyond simple empirical pattern detection while the former is not well-designed for direct, open-ended exploration of large mathematical datasets. In this paper we introduce PatternMiner, a system that combines these two paradigms. In PatternMiner, an LLM agent generates mathematical datasets based on an initial broad question, trains narrow machine learning models on them, uses interpretability techniques to extract patterns from these trained models, and uses these patterns to formulate conjectures. Once an initial conjecture has been formulated, PatternMiner instructs the agent to search for counterexamples using calls to mathematical software. If a set of counterexamples can be found, these are used as feedback for conjecture refinement or another round of training and interpretability. The conjecture progressively evolves through this iterative sequence with the agent allowed to try to prove it at any point. We describe the results of applying PatternMiner to five well-known positivity problems in algebraic combinatorics. PatternMiner proves several new theorems related to Schubert positivity and projection, a character-row evaluation formula, a restricted version of the Kazhdan--Lusztig combinatorial invariance conjecture, and Schur positivity for a family of immanant coefficients. We end by analyzing the most common routes to conjecture generation that are used by PatternMiner in our case studies as well as documenting the most common failure modes. 

---
# Should Service Requirements Be Scalarized or Constrained When Minimizing Costs? A Case Study on HVAC Control 

**Authors**: Tai-You Chen, Mihael Mohorcic, Blaz Bertalanic, Carolina Fortuna  

**Link**: [PDF](https://arxiv.org/pdf/2611.33946)  

**Abstract**: Automated decision systems, from learned policies to model-based optimizers, often minimize costs such as energy or resource use while meeting a service requirement, for example the quality of service (QoS) specified in a service level agreement (SLA), a comfort band, or a safety margin. The requirement can be scalarized with the costs into a single objective, with a fixed or an adaptively tuned (Lagrangian) weight, or it can be imposed as a constraint. We study this choice in the Smart Buildings Challenge, in which a single policy controls the heating, ventilation and air conditioning (HVAC) of previously unseen buildings and is evaluated by a weighted sum of comfort, energy cost and carbon emissions. With all other components fixed, we compare the two treatments in model-based optimization, represented by model predictive control (MPC) with the requirement as a hard or a soft (l1-penalized) constraint, and in learned policies, represented by proximal policy optimization (PPO) with a fixed or a Lagrangian penalty in the reward or a projection layer on the action. In MPC, the soft constraint reproduces the hard one once its weight exceeds the requirement's shadow price, which the challenge's comfort weight exceeds approximately fourfold; below this weight, comfort is sacrificed, and under a metric that values comfort below its shadow price the soft constraint outperforms the hard one by up to 28%. A weight tuned on the training buildings is 3.9% worse than the hard constraint on 48 unseen buildings, because the shadow price varies across buildings. In PPO, the projected policy satisfies the requirement from the outset and leads early in training; with longer training, the Lagrangian policy overtakes it while the projected policy deteriorates, and the fixed penalty does not learn to satisfy the requirement. 

---
# What Does Skill Lift Actually Measure? A Source-Code-Level Audit and Noise-Aware Reanalysis of SkillsBench 

**Authors**: Haozhe Zhang, Zhen Long  

**Link**: [PDF](https://arxiv.org/pdf/2611.33943)  

**Abstract**: Agent skills are directories of instructions, reference materials, and scripts loaded on demand by coding agents. According to SkillsBench, although human-curated skills can raise the passing rate by about 16.6 percentage points (pp) on average, they have actually reduced the results for some tasks. Later research has started to gate or eliminate skills according to the above per-task gains. We have audited the above benchmark at the source-code level and analysed 7,554 public trials to form 1,421 paired "(configuration, task) cells". Source-code audit identified two controlled-verified cases of grader dependence and one operational definition gap; the average lift on these three tasks was 74.6 pp, and the overall average lift for the full set was 18.7 pp. The audit also found that four tasks contained placeholder skill directories, and a certain execution harness additionally added 48 built-in skills in most runs. No task-level reduction achieved statistical significance in the pooled reanalysis with multiple-testing correction. As there were at most three runs per condition for each cell, the observed negative differences are very susceptible to run-sampling noise, and arithmetically, 58% of the cells could not have shown a decrease. Therefore, the raw numbers of negative cells cannot show that skills cause harm. A gating strategy based on the skill-free score seems to improve by 2-3 pp in the logs; however, even after shuffling condition labels to remove any skill effect, it still achieves an increase of 4-7 pp. On the held-out runs not used for fitting and decision-making, there was no statistically significant improvement. The audit ledger and analysis scripts have been published by us. 

---
# EGSE: Evidence-Grounded Skill Evolution for Medical Reasoning Agents 

**Authors**: Mingyuan Li, Tong Jia, Ziyuan Yang, Wenbo Qi, Haoran Wang, Jie Gui, Baosheng Yu  

**Link**: [PDF](https://arxiv.org/pdf/2611.33909)  

**Abstract**: Medical reasoning agents combine foundation models with specialized tools to interpret medical images, yet reusing their historical experience remains challenging. A correct final answer can conceal flawed reasoning or tool use, synthesized advice may lack source support, and superficially similar cases may require different diagnostic procedures. In this paper, we propose EGSE, an evidence-grounded skill evolution framework that distills historical dialogues from similar cases into reusable conditional workflows and progressively refines a multi-faceted skill library through supported and validated updates to improve medical agents. EGSE evaluates intermediate reasoning and tool use and contrasts repeated successful and failed rollouts to extract useful practices and evidence-supported avoidance guidance. These experiences are synthesized into workflows with applicability conditions and checks traceable to source dialogues. Library additions and merges require source-grounded review and execution validation. At inference time, EGSE routes queries to skill nodes organized by imaging modality, question task type, and disease category, then retrieves and selects applicable workflows based on their conditions, the current images, and the question to guide evidence acquisition and interpretation. Experiments across four chest X-ray benchmarks show that EGSE outperforms representative medical agents and skill evolution methods in reasoning accuracy while reducing tool calls. It also delivers consistent accuracy gains across seven foundation models and supports skill transfer across datasets. 

---
# EATS: Scaling Tool-Use Experience for Medical Reasoning Agents 

**Authors**: Mingyuan Li, Tong Jia, Haoran Wang, Wenbo Qi, Ziyuan Yang, Bowen Ma, Jie Gui, Baosheng Yu  

**Link**: [PDF](https://arxiv.org/pdf/2611.33897)  

**Abstract**: Medical reasoning agents combine large language model (LLM) reasoning with clinical workflows and specialized neural tools. However, general-purpose LLMs receive limited training exposure to domain-specific medical tools, and the uncertainty and distributional shifts of neural tool outputs make selecting tools from simple descriptions significantly more challenging than in deterministic settings such as calculators or dictionary lookups. In this paper, we propose EATS, an experience-augmented tool selection strategy that leverages accumulated tool-use trajectories to improve tool selection in medical reasoning agents. EATS constructs a memory of past interactions, including queries, tool choices, arguments, and self-evaluated feedback, and retrieves relevant experiences to guide tool selection for new queries. As memory construction does not require ground-truth supervision, EATS scales naturally with test-time experience. We evaluate EATS on tool-augmented medical reasoning tasks, including closed-ended visual question answering and free-form report generation. Experiments on four public datasets, ChestAgentBench, CheXbench, MIMIC-CXR, and SLAKE-VQA, across three multimodal LLMs, Qwen3-VL 2B, Qwen3-VL 32B, and GPT-5.5, show consistent gains in tool selection behavior and downstream medical reasoning performance. Code is available at this https URL. 

---
# FORK: Co-Evolving Safe Disclosure and Decision Skills Against Attackers Who Read Every Refusal 

**Authors**: Hieu Man, Chao Shang, Rui Dong, Devang Kulshreshtha, Yanjun Qi, Sandesh Swamy  

**Link**: [PDF](https://arxiv.org/pdf/2611.33879)  

**Abstract**: When a guardrail rejects an LLM agent's action, its message reaches any attacker observing the trajectory: a fixed refusal leaves benign users unable to recover, whereas an explanation can tell the attacker what to try next. We study this as safe disclosure: explaining rejections so that authorized tasks recover without increasing adaptive attack success. Safe disclosure depends on both the decision and the message: what the guardrail blocks and what it reveals to the agent and the attacker. Because hand-written rules alone cannot anticipate how adaptive attackers probe decisions or use messages, we introduce FORK, which co-evolves a guardrail's natural-language decision and disclosure skills with an attacker that reads every public message. Each exploited message and permissive decision gives FORK evidence for a gradient-free revision of its natural-language skill library. We evaluate FORK on four benchmarks. Averaged over three agents on FORK-World, FORK cuts attack success rate (ASR) under adaptive attack from 78.4% to 14.1% and keeps utility above 85%, outperforming baselines on both ASR and utility. Notably, on MT-AgentRisk, whose environments and tasks are unseen in training, FORK lowers ASR from 70.6% to 9.1% with 93.1% benign utility. 

---
# Can Its Own Misses Teach an MLLM to Find Small Brain Metastases? 

**Authors**: Yixuan Duan, Kexin Geng, Hanwen Xu, Wei Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2611.33836)  

**Abstract**: Multimodal large language models (MLLMs) are becoming general readers of medical images, yet they remain weak at locating small abnormalities. Brain metastases expose this gap: at 384 px a 0.1 mL lesion covers less than a quarter of one visual token, and at one false positive per scan frontier MLLMs find fewer than 30% of lesions of 0.1-1 mL. Fine-tuning on one dataset helps but transfers poorly, and improving it further is slow trial and error steered by a single score. We introduce LANTERN (Lesion-finding AgeNt Through Error-driven Recursive learNing), a framework in which an MLLM improves recursively from its own misses. LANTERN recasts model development as an error-driven loop: a failure census attributes every missed lesion to one of four causes, an LLM mentor revises one of four levels (evidence acquisition, visual representation, learning policy, decision aggregation) per round, and a rule frozen before the first round decides what is kept, without test data and with every tool call audited. Over 15 unattended rounds, LANTERN raises a fine-tuned 4B MLLM from finding 75% to 94% of lesions at one false positive per scan on BraTS-MET, and 91% of those of 0.1-1 mL, compared with at most 51% and 29% for five frontier MLLMs read under the starting protocol; the gains hold on an external cohort (89%) and on unseen meningiomas (77%). These results show that an MLLM's own misses, attributed to their causes, can drive its reliable recursive self-improvement. 

---
# MetaDecomp: Adaptive Task Decomposition via Meta-Learned Procedural Memory 

**Authors**: Ziyi Wang, Yixiao He, Yichen Wu, Ruite Lu, Jiayu Wang, Ying Wei  

**Link**: [PDF](https://arxiv.org/pdf/2611.33822)  

**Abstract**: Reusing past experience in long-horizon tasks requires more than storing successful procedures: Language agents must distinguish which procedures apply and what work remains as execution progresses. We introduce MetaDecomp, an in-context meta-learning framework for adaptive task decomposition that jointly learns a task-relative state space and reusable procedures. Its outer loop builds procedural memory from successful training executions, representing task progress as abstract states and procedures between states as directed acyclic graphs of operations and dependencies. As new progress distinctions are learned, MetaDecomp revisits historical execution evidence to reassess state assignments and reorganize the procedures they support. With memory fixed at test time, the inner loop refines an initial decomposition and revises it from environmental feedback, combining applicable procedure fragments with task-specific steps missing from memory. Evaluated on ALFWorld and ScienceWorld over three seeds, MetaDecomp achieves the highest mean performance among the compared methods in four of five settings, for example, an 86.3% success rate on ALFWorld with Qwen3-4B. Fixing the initial schema or omitting historical reprojection reduces performance in component ablations. MetaDecomp also uses fewer inference tokens than ReAct and AutoRefine in all five settings. Cross-benchmark analysis identifies a low-performance area where partial coverage degrades success relative to hidden memory; high coverage improves success, while low coverage changes little. Code will be released upon acceptance. 

---
# RadOncGym: An Agent-Based Framework for Radiotherapy Dose-Mimicking Optimization 

**Authors**: Tong Miao, Shupeng Cheng, Xiwen Chen, Ruishan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.33821)  

**Abstract**: AI dose-prediction models generate patient-specific 3D dose distributions from patient anatomy, providing an important intermediate representation for automated radiotherapy planning. However, an AI-predicted dose must undergo downstream optimization to produce a physically constrained treatment plan, a process that requires balancing competing planning objectives. We present RadOncGym, an agent-based framework that enables an LLM agent to iteratively tune 12 planning-objective weights for dose-mimicking optimization. To reduce iterative evaluation cost, RadOncGym incorporates a surrogate verifier that provides rapid feedback while allowing the agent to decide when to request a full planning solve within a fixed solve budget. We evaluate RadOncGym on 60 OpenKBP head-and-neck patients, each paired with 10 predictions from OpenKBP-Opt dose-prediction sets, yielding 600 patient-prediction contexts per experimental configuration. Under RadOncGym's 16-turn setting with a three-solve budget, a GPT-5.6 Sol agent reduces mean clinical-criterion violation by 0.6131 Gy and improves DVH and dose scores by 0.5986 and 0.2204 Gy relative to leaving all 12 weights fixed at 1, outperforming all evaluated non-agent weight-tuning methods under the same solve budget. It remains competitive with the 16-solve variants of these methods while requiring 75.0-77.6% less normalized runtime. RadOncGym demonstrates the potential of LLM agents as optimization planners in downstream radiotherapy plan optimization and provides a public testbed for their systematic evaluation. Code and configurations are available at this https URL. 

---
# Spectral Optimal Transport for Dynamics-Aware Sign Language Generation 

**Authors**: Nabeela Khan, Bowen Wu, Runwu Shi, Hanlong Li, Benjamin Yen, Carlos Toshinori Ishi, Kazuhiro Nakadai  

**Link**: [PDF](https://arxiv.org/pdf/2611.33818)  

**Abstract**: Sign language generation (SLG) translates spoken-language text into pose sequences that convey its meaning. Recently, flow-based models with optimal transport (OT) have become a strong SLG baseline by handling the one-to-many nature of text-to-sign. While classifier-free guidance (CFG) could be applied to further strengthen the semantic alignment theoretically, its training involves conditional and unconditional training, which may contradict the pose-space OT, thus reducing the generation quality. Under the condition dropout, one noise sample can be paired with target motions exhibiting incompatible motion dynamics, interfering with the semantic guidance when the text condition is present. To address this issue, we propose Spectral Optimal Transport (SOT), which augments the pose-space cost with a spectral feature distance, reweighting the coupling toward the temporal-frequency structure of the target motions, thereby improving the performance when using CFG. The quantitative and qualitative experiments demonstrate the effectiveness of the proposed method. 

---
# Ensuring Deployment-Time Safety of Neural Network Controlled Systems via Localized Certificate Repair 

**Authors**: Xiaoyang Lv, Peixin Wang, Jianhao Bai, Bai Xue, Min Zhang, Luke Ong  

**Link**: [PDF](https://arxiv.org/pdf/2611.33813)  

**Abstract**: Barrier certificates provide formal safety guarantees for neural network controlled systems by ensuring that system trajectories avoid unsafe regions over time. However, after deployment, previously unseen obstacles or constraint changes may introduce new unsafe regions, invalidating the original certificate and potentially leading to unsafe behavior. We address this setting with adaptive piecewise barrier certificates, which preserve the original certificate and policy in unchanged regions while introducing localized updates around newly emerged unsafe regions. The central question is how to perform such updates efficiently during system execution without retraining from scratch. Our key idea is to cast deployment-time adaptation as a localized certificate repair problem and reduce it to an optimization task that leverages the learned structure of previous barrier certificates, enabling fast updates without global retraining. We further design a staged repair framework that progressively transitions from localized repair to joint policy-certificate updates and, when necessary, localized retraining. Empirical results on four benchmarks with five new obstacles show that our method succeeds in all cases and completes within tens of seconds on average, whereas retraining from scratch fails in most cases. On instances where retraining succeeds, our method is 17.3$\times$ faster on average. It incurs only 16.2% performance degradation relative to the original policy. 

---
# Keep or Roll Back? Out-of-Loop Judges for Self-Improving VLA and End-to-End Embodied Policies 

**Authors**: Zilin Huang, Zhengyang Wan, Kangan Qian, Zihao Sheng, Sikai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.33794)  

**Abstract**: Self-improvement is now a practical recipe for embodied policies: a policy collects experience, is retrained, and is kept when a judge rates it higher. However, the judge is usually trained or chosen inside the same loop, so its evidence is not independent of the policy. We introduce SealedLoop, an evaluation framework that adds a sealed judge: an evaluator outside the loop that the improver cannot see or optimize against. A new policy is kept only when a sequential test on the sealed judge confirms the gain, with a bounded probability of keeping a worse one. Running the same loop under each judge, we separate two sources of a false gain, which we call judge error (the loop's judge is wrong on its own data) and transfer gap (a real gain there does not carry over to unseen conditions). We evaluate on LIBERO manipulation, NAVSIM driving, and a full-scale real vehicle. In manipulation, a success classifier accepted 65 of 196 failures in one generation. In driving, policies that gain on the loop's own frames lose in an unseen city, so even an exact in-loop judge would keep 8/15 of them. When the sealed test decides, a driving loop built to improve ends with a better policy than under its own critic (0.674 against 0.649 PDMS, from 0.638), and on driving pairs with known sealed gain the test accepted 11 of 12 real gains and none of 59 losses. In a real-world field experiment on a published improvement step, the reinforced version drove further into a curve than the version it started from in all 6 blind pairs, though neither completed it. These results show that evidence from inside the loop, however accurate, is not enough to decide whether a policy has improved. SealedLoop offers a foundation for trustworthy self-improvement of embodied policies, in which progress is judged by evidence the improving system cannot influence. 

---
# ExtractBench Pro: The Long Tail of Document Extraction 

**Authors**: Boyang Zhang, Justin Xu, Adrian Lyjak, Eli Stewart, Simon Suo  

**Link**: [PDF](https://arxiv.org/pdf/2611.33791)  

**Abstract**: We present ExtractBench Pro, a schema-guided document extraction benchmark that tests six challenging modes of extraction underrepresented in existing datasets. It is created following a framework for synthetic document generation, in which deterministic document generators assemble challenging components into documents that are designed to be difficult at a level calibrated for frontier systems. We synthesize 100 realistic documents spanning a diverse range of styles. On this data, we evaluate ten coding agents and eight extraction API configurations for accuracy, grounding, and their ability to perfectly extract documents. No extraction system extracts more than 61% of documents error-free. The strongest general-purpose coding agents, Claude Code Opus 5.5 and Codex GPT-6 Astra, outperform specialized extraction APIs on both perfect-document rate and Value F1, but at a higher cost per page. Among specialized extraction APIs, LlamaExtract spans the lower-cost frontier, with Agentic Plus Max Effort approaching the strongest coding agent on both perfect-document rate and Value F1 at lower cost per page. Our analysis of coding agent traces identifies intelligent image processing and programmatic reasoning, but also failures in applying document-specific definitions and recovering relationships from diagrams. We release ExtractBench Pro and these evaluation results as resources for the future development of improved frontier extraction systems. 

---
# InterGenEHR$^+$: An Interactive Framework for Natural Language-Conditioned Generation of Longitudinal Patient Trajectories 

**Authors**: Yu Yang, Jiafeng Song, Baraa F Abed, Christina Creel, Sagar B Dave, Henry Foote, Lisa Lima, Victor Moas, Craig S Jabaley, Rishikesan Kamaleswaran, Pan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2611.33788)  

**Abstract**: Objective: To develop and validate InterGenEHR$^+$, an interactive framework for natural language-conditioned generation of synthetic tabular electronic health records (EHRs) and longitudinal trajectories for scenario-based evaluation of machine learning models.
Methods: An optional large language model (LLM) parser converts clinical requests into structured constraints. Conditional variational autoencoders and diffusion models generate single-time-point records, while diffusion models generate longitudinal trajectories. We evaluate parsing, distributional fidelity, imputation, clinical-constraint adherence, mortality prediction, and clinical plausibility using a multi-institutional cohort of 874 patients receiving venovenous extracorporeal membrane oxygenation (VV-ECMO).
Results: Gemma 3 4B with few-shot prompting achieves the highest parsing performance with a low spurious slot rate. IGE-TabDDPM and IGE-PMVAE perform well in single-time-point generation; IGE-CSDI, IGE-FGTI, and IGE-MTSCI lead on different longitudinal fidelity metrics. Across five random seeds and three classifiers, synthetic cohorts retain predictive utility for mortality prediction on real test data. Clinician ratings support trajectory plausibility and alignment with requested scenarios.
Conclusion: InterGenEHR$^+$ supports clinician-specified generation of EHR trajectories with statistical fidelity and downstream predictive utility in the evaluated VV-ECMO cohort. Extension to other clinical settings requires further validation.
Significance: The framework supports scenario-based evaluation of machine learning models in data-sparse critical care settings. 

---
# CHD-Bench: Diagnosing Congenital Heart Disease through Interactive Examination 

**Authors**: Gautham Krishnan P, Shrisharanyan Vasu, Aniketh Vijesh, Rajagopalan Rameshbabu, Harikrishnan Anil Maya, Sarin Xavier, Mahesh Kappanayil, Gilad Gressel  

**Link**: [PDF](https://arxiv.org/pdf/2611.33775)  

**Abstract**: Diagnosis of congenital heart disease (CHD) requires understanding the spatial relationships between cardiac structures, which clinicians assess by scrolling through CT or cardiac magnetic resonance imaging (MRI) slices and rotating 3D reconstructions. Although automated methods have advanced cardiac segmentation, identifying congenital heart defects from volumetric scans remains an open challenge. The two existing approaches rely on handcrafted diagnostic rules limited to static anatomical abnormalities. We introduce CHD-Bench, the first benchmark for interactive CHD diagnosis, defining a multi-label task and evaluation protocol over 247 CT and MRI scans from three cohorts covering 38 CHD types. We also present CHD-Harness, open-source software that runs a VLM in a 3D Slicer environment through 22 tools, manages each examination, and records the resulting diagnosis and supporting views. We evaluate off-the-shelf medical, open weight, and closed weight VLMs and compare them with a 3D CNN classifier and published rule-based results. Gemini 4 Argon with CHD-Harness achieves the highest macro F1 across all three cohorts (0.63, 0.63, and 0.43). Allowing for interactive examination raises its macro F1 by up to 20 points over static views. These results establish CHD-Bench and CHD-Harness as a foundation for developing and evaluating VLMs for interactive CHD diagnosis. 

---
# Improving Building Energy Efficiency with an Indoor Environmental World Model and Counterfactual Adaptive Learning 

**Authors**: Xu Yang, Kailai Sun, Dianyu Zhong, Irfan Qaisar, Yu Yang, Qianchuan Zhao, Jian Kang  

**Link**: [PDF](https://arxiv.org/pdf/2611.33764)  

**Abstract**: Optimizing indoor climate systems plays a critical role in achieving the United Nations' Sustainable Development Goals 11 and 13. However, indoor delayed thermodynamic responses and partial observability severely hinder existing reinforcement learning (RL) methods, which are primarily limited by implicit thermal inertia, occupancy dynamic prediction, and cumulative prediction errors during long-horizon rollouts. Thus, we propose a novel occupant-centric indoor climate control framework. It develops indoor environmental world models to predict future physical thermal baselines and dynamic occupancy, explicitly capturing latent thermal inertia. By formulating adaptive regulation as counterfactual advantage learning against future environments via a Branching Dueling Q-Network, and integrating a Held-Action TD-{\lambda} target, it effectively mitigates delayed credit assignment. Additionally, a proposed physics-aligned smoothness regularization suppresses high-frequency actuator fluctuations, potentially extending equipment service life. Experiments on SemiBuildingSim and Sinergym demonstrate that our method achieves a superior multi-objective trade-off. Compared to the best-performing methods, it further reduces energy consumption by 11.2%, occupant discomfort by 44.2%, and action fluctuation by 47.3%, achieving better sample efficiency and cross-region effectiveness. This interdisciplinary study offers an energy-efficient, occupant-centric AI solution for control science, building science, and energy sustainability. 

---
# TransHarness: From Benchmark Overfitting to Transferable Harness Evolution 

**Authors**: Tianze Xia, Zimo Wang, Heng Tong, Zhaoxiang Liu, Shiguo Lian, Fan Huang, Wenyu Liu, Yunchao Wei, Xinggang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.33757)  

**Abstract**: Harness evolution offers a path to improving coding agents without updating model weights. However, existing methods can overfit their evolution benchmarks, producing gains that fail to carry over to new tasks. To address this challenge, we introduce Transferable Harness Evolution (TransHarness), which evolves harness intervention policies from same-task success-failure contrasts. These policies couple execution-context triggers with diagnostic guidance, enabling task-dependent checks and corrections beyond the source executions. Cross-task replay and independent semantic selection retain policies that intervene selectively and ground corrections in current-task evidence rather than source-specific solutions. We further curate an independent evolution set for cross-distribution transfer and conduct a cross-benchmark evaluation covering multiple strong models and a range of lightweight and mature harnesses. Experiments show that, with only 300 source trajectories from outside the test distribution, TransHarness further improves strong agents across distinct target benchmarks, demonstrating transferable harness evolution beyond benchmark overfitting. Specifically, TransHarness achieves state-of-the-art scores of 76.4% on DeepSWE v1.1 and 95.5% on Terminal-Bench 2.1, surpassing the strongest baseline model, GPT-6 Astra, by 2.4 and 5.6 percentage points, respectively. 

---
# REFORGE: LLM-Assisted RTL Bug Repair with Structured Context and Verification Feedback 

**Authors**: Tanvir Hossain, Sudipta Paria, Aidan Gleason Myers, Swarup Bhunia, Tamzidul Hoque  

**Link**: [PDF](https://arxiv.org/pdf/2611.33748)  

**Abstract**: Large language models (LLMs) can write and repair RTL, but successful compilation or testbench execution does not establish defect removal, behavioral preservation, security, or implementation feasibility. Existing LLM-based RTL repair methods often report a single repair rate without retaining separate evidence for these properties, limiting reproducible comparison and integration into hardware design flows. We present REFORGE, which, to our knowledge, is the first LLM-assisted RTL repair framework that maintains and accumulates structured repair evidence across iterations, allowing it to generate fixes that satisfy multiple objectives simultaneously, including functionality, behavioral equivalence, security, and cost efficiency. REFORGE profiles buggy RTL, identifies structural and security-relevant features, localizes failing behavior, and uses verification-guided diagnosis to generate complete RTL repair variants. Each variant undergoes syntax checking, simulation, formal analysis, off-trigger preservation, security re-scan, and synthesis-based power, performance, and area analysis. We implement REFORGE as a backend-independent repair-and-verification flow and evaluate it in 288 runs across three LLM backends and three benchmark suites. On CirFix, the proposed context and diagnosis increase runs delivering a checked repair from 32/63 to 51/63, while every configuration repairs the Hack@DAC OpenTitan defect. Evidence records also show that a testbench-only criterion reports 3.9 times more repairs than the full set of applicable checks, security-clean outputs may lack functional verification, and different RTL repairs can implement equivalent hardware. These results show that LLM-generated RTL repairs require explicit multi-stage verification evidence rather than a single pass/fail outcome. 

---
# WhatIf: Counterfactual World Models over Action-Grounded Causal Graphs for Robotic Manipulation 

**Authors**: Xu Yang, Kailai Sun, Qianchuan Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.33745)  

**Abstract**: World models predict what happens under an action, but choosing between actions requires comparing their outcomes in the same physical situation. Factual transitions alone determine action-wise marginal futures and average effects, but not individual effects or the probability that switching actions causes harm. We introduce WhatIf, a counterfactual world model for manipulation that couples candidate futures through shared latent physics and transition noise inferred from factual history. Each candidate action induces a sparse object-factor causal graph describing the physical pathways it activates. Rolling out these graphs under the shared situation anchors predictions to observed outcomes and confines action contrasts to the objects each action reaches. Shared type-conditioned functions support varying numbers of objects, while a round-trip loss enforces action-invariant abduction without counterfactual labels. We show that replacing an individual effect with its conditional mean incurs mean-squared error equal to its conditional variance, motivating comparisons between coupled futures. Building on these comparisons, Counterfactual Advantage Planning (CAP) ranks plans by their advantage over an incumbent within shared situations and constrains the probability of harm. Paired advantages reduce estimator variance by a factor of $1-\rho$, where $\rho$ is the correlation between branch outcomes, and cancel model errors shared by both branches. Each executed transition provides hindsight advantages to update the physics posterior, the next plan, and an amortized policy without a learned critic. We further establish when observational transitions identify interventional dynamics. Paired simulator interventions evaluate counterfactual accuracy, graph recovery, compositional generalization, and decision quality. 

---
# JOVE ICE: Intent Guided Cloud Edge Device Collaboration for AI Glasses 

**Authors**: Yuyang Jin, Bingfeng Pi, Jiamei Sun, Wei Jin, Dongyang Wang, Yunpeng Li, Yan Bai, Lei Zuo  

**Link**: [PDF](https://arxiv.org/pdf/2611.33735)  

**Abstract**: Recent advances in large language models (LLMs) and LLM based agents have expanded the capabilities of AI glasses, enabling egocentric visual question answering, memory retrieval, and proactive assistance. These applications involve diverse inputs, including speech, images, and video, and range from simple device control to complex visual reasoning, posing challenges for efficient computation and communication. While video centric capabilities have received considerable attention, simple text based and device executable tasks remain underexplored despite their prevalence in daily interactions. Efficiently supporting diverse workloads within a unified assistant remains an open challenge. We propose JOVE ICE, an intent guided cloud, edge, and device collaborative framework that routes requests among AI glasses, a paired smartphone, and cloud servers according to user intent. JOVE ICE integrates hierarchical intent classification with dynamic subagent orchestration to select efficient execution paths and flexibly incorporate new applications through specialized subagents. Analysis of real world AI glasses requests reveals that simple device control tasks account for over 30% of interactions, which JOVE ICE handles directly on device to avoid unnecessary cloud computation and costs. Evaluations on three public benchmarks demonstrate that JOVE ICE reduces time to first token by approximately 13% and token usage by over 11% compared with conventional device and cloud architectures, while maintaining task quality. These results demonstrate the potential of intent guided collaboration to improve the efficiency and scalability of AI glasses. 

---
# Asset4D-Bench: Benchmarking Coding Agents for Editable 3D and 4D Reconstruction 

**Authors**: Chengyi Sun, Deyuan Liu, Lei Zhao, Jungang Li, Qi Jia  

**Link**: [PDF](https://arxiv.org/pdf/2611.33728)  

**Abstract**: A cart can follow its reference trajectory while its cargo floats above the deck. Asset4D-Bench tests whether coding agents recover objects, motion and interactions from visual references as editable Blender scenes. Its 114 conditions span image editing, video reconstruction with matched full or partial starter assets, and independently selected from-scratch tasks. We independently reopen projects and measure them against native reference states. The 228 archived scores include original delivery outcomes and versioned measures of scene quality. On 24 from-scratch tasks, each model earns full dynamics credit on 14. Of these, three GPT-6.1-Sol and eight GPT-6-Astra reconstructions earn zero relation points. Under the fixed scoring weights, relations contribute 5.15 points to the 6.50-point Sol-minus-Astra difference in mean recovered fidelity. Differences in execution and recovery policy across archived runs preclude a causal estimate of starter-asset effects. 

---
# KeyDeck: Multi-Paper-to-Slide Generation via Cross-Paper Synthesis and Reference-Guided Personalization 

**Authors**: Tarik Can Ozden, Sachidanand VS, Ram Prakash Reddy Motati, Ozgur Kara, Dilek Hakkani-Tür, Junho Kim, James Matthew Rehg  

**Link**: [PDF](https://arxiv.org/pdf/2611.33715)  

**Abstract**: Multi-paper-to-slide generation (multi-PTSG) is the task of constructing a unified presentation from multiple research papers by selecting, organizing, and synthesizing information across sources. This task introduces two main challenges: (i) combining multiple sources into a coherent narrative requires identifying cross-paper relationships and synthesizing their content, and (ii) the desired organization and emphasis varies substantially across presenters, making personalization essential for user satisfaction. Motivated by these challenges, we propose KeyDeck, the first multi-PTSG framework that integrates per-paper and cross-paper analysis to identify relationships among sources and synthesize their contributions into a coherent narrative, and reference-guided personalization to infer presenters' narrative, content, textual, and visual preferences and adapt slide generation accordingly. To evaluate generation quality, personalization adherence, and conversational goal satisfaction, we introduce KeyBench, a benchmark of 100 expert-authored multi-paper presentations, together with a comprehensive evaluation framework. Our results demonstrate that KeyDeck outperforms existing approaches in multi-PTSG quality, personalization adherence, and conversational goal satisfaction. 

---
# PowerFix: Physics-Guided Iterative Repair for LLM-Assisted Power Converter Design 

**Authors**: Can Wang, Shi Qiu, Fanfan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2611.33712)  

**Abstract**: Large language model (LLM) agents are increasingly being explored for automated engineering design. However, repairing power-converter designs that fail engineering validation remains challenging: the system must identify physical causes, locate responsible design variables, and resolve violations without compromising other constraints. We introduce PowerFix, a physics-guided iterative repair framework that converts validation failures into structured diagnoses and executable patches while preserving topology and design requirements. PowerDiagGraph links failed checks to possible physical causes and constrains repair routes. An LLM prioritizes these routes, while a deterministic compiler translates selected repairs into authorized field-level edits. Candidates are validated on isolated design copies and committed only after all required engineering checks pass; failed validation can guide replanning. On our constructed 96-task flyback repair benchmark, comprising injected single faults, coupled faults, and held-out natural failures from our design generator, PowerFix achieves 78.7% verified repair success rate (VRSR), compared with 30.1% for the strongest of three LLM baselines. Three additional replanning rounds increase VRSR to 88.4%. These offline results provide pilot-stage evidence within a simulation-based design workflow, with hardware prototype validation and transfer to a second converter topology as the next steps toward deployment. 

---
# P-FedSkill: A Personalized Federated Skill Learning Framework under Task Heterogeneity 

**Authors**: Zehang Lin, Jiayi Ji, Rongrong Ji  

**Link**: [PDF](https://arxiv.org/pdf/2611.33707)  

**Abstract**: Skill learning has become an important way to improve agent performance. However, agent trajectories often contain sensitive information and cannot be shared across clients, hindering collaborative skill learning. To tackle this issue, federated skill learning (FSL) has emerged as a solution by distilling trajectories into compact skill representations and uploading them to a central server for sharing. However, existing FSL methods typically share skills based on skill similarity without considering their contributions to heterogeneous tasks across clients, thereby degrading agent performance. To address this challenge, we propose P-FedSkill, a personalized FSL framework that comprises three key components: local skill extraction (LSE), personalized skill selection (PSS), and task-aware skill routing (TASR). Specifically, LSE first extracts skill patches (i.e., task applicability conditions and execution strategies) and task execution outcomes from trajectories. PSS then uses these outcomes to evaluate each skill patch's performance on client tasks and incorporates its task applicability condition to construct a personalized skill set for each client. Finally, TASR adaptively selects a task-relevant subset of skill patches based on their applicability conditions to guide agent execution. We further provide a theoretical guarantee for personalized skill selection in P-FedSkill, and extensive experiments also demonstrate its effectiveness. 

---
# EVA: Evidence-seeking Visual Agent for Hallucination-Resistant Multimodal Reasoning 

**Authors**: Ting Lei, Yu Chen, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.33706)  

**Abstract**: Hallucinations in multimodal large language models (MLLMs) often arise from visual assumptions that go unchecked. Self-reflection, multi-agent debate, and counterfactual verification can reinforce erroneous conclusions when they rely on the same unverified visual assumptions. External visual tools can provide verifiable evidence, but models may select the wrong tools, supply incorrect parameters, or ignore or contradict the evidence they obtain. In this work, we propose EVA, an Evidence-seeking Visual Agent that reframes multimodal reasoning as an explicit process of evidence acquisition and verification. At the core of EVA is a unified evidence-grounded execution pipeline designed to address the above challenges through two tightly coupled mechanisms. First, we introduce Iterative Tool Replanning, which dynamically refines tool selection and parameters based on intermediate observations, improving the reliability of tool usage. Second, we develop Evidence Consistency Verification, an answer validation mechanism that improves alignment between the final prediction and the collected evidence, mitigating evidence--reasoning inconsistencies. In addition, to balance robustness and efficiency, EVA incorporates Selective Evidence Routing, an adaptive fast--slow reasoning strategy that determines when external evidence is necessary. Visually straightforward queries are handled via direct reasoning, while complex or high-risk cases trigger tool-based evidence acquisition. By jointly addressing how to reliably acquire evidence and how to use it consistently, EVA transforms multimodal reasoning from passive answer validation into explicit evidence-backed decision making. Experimental results demonstrate that EVA achieves significantly improved reasoning reliability over direct VLM inference and multi-agent methods, while maintaining efficiency through selective tool invocation. 

---
# Faster and Better Query (FBQ): Learning Future-Position Probes from the First Generated Token 

**Authors**: Jiankun Zou, Daixing Chen, Tongtong Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2611.33679)  

**Abstract**: When a large language model (LLM) evicts prompt keys and values at the end of prefill, it must decide what to keep before it sees the queries from subsequent decoding steps. Generating a draft continuation can approximate those queries at the cost of extra autoregressive steps. Can we use the embedding of the first output token, already available from the final prefill logits, to construct low-cost pseudo queries that approximate future decoding queries? We start with FTQ-8, which places eight copies of the first token ID at consecutive future positions and reads their attention to the prompt in one causal forward pass. Each copy sees a different prefix, but a generic opening such as the offers a poor literal guess for seven successive tokens. Faster and Better Query (FBQ) preserves the first embedding and learns an additive offset at each of the seven remaining positions. We train these offsets to predict the average prompt attention of a full-cache continuation. On LongBench, FBQ performs comparably to eight-step Lookahead Q-Cache at the same cache budget, using a single forward pass to compute all probe queries. 

---
# Interactive Proofs as a Unified Framework for Evaluating and Leveraging Verifiable Proof Generation by LLM Agents 

**Authors**: Yankai Shu, Xingyu Zhao, Chuan Mao, Congchi Yin  

**Link**: [PDF](https://arxiv.org/pdf/2611.33676)  

**Abstract**: Recent advances in LLM-assisted theorem proving and program verification have attracted widespread attention. Beyond determining whether a proposition is true, LLM agents must produce evidence that can be independently checked by humans or formal verification tools. To evaluate the underlying, protocol-agnostic ability of LLM agents to construct verifiable proofs-rather than their proficiency with any particular proof language or verification tool, we introduce Merlin-Bench. Drawing on interactive proof systems from computational complexity theory, we cast an LLM agent as the prover: under a designated protocol, the agent interacts with a formal verifier to substantiate its claimed answer to a problem instance. We develop an automated framework for constructing test cases and use it to create 100 cases across 15 problem families, each formulated within this prover-verifier framework. Experiments with 10 frontier LLMs across multiple agent harnesses show that reliably producing verifiable proofs in accordance with specified protocols remains a substantial challenge. Our case studies further provide actionable insights into improving agents' ability to generate verifier-accepted proofs: tool use is essential for completing practical proof tasks; different forms of guidance, ranging from engineering instructions to problem-specific knowledge, substantially affect success rates on interactive-proof tasks; and proof-solving experience distilled from abstract tasks can transfer to concrete, real-world verification tasks. 

---
# SlicePlanner: Logic-Guided Agentic AI for Online QoS-Aware Network Slicing in Open RAN 

**Authors**: Adhwaa Al-Chaab, Ayman Younis, Dario Pompili  

**Link**: [PDF](https://arxiv.org/pdf/2611.33664)  

**Abstract**: Adaptive network slicing in Open Radio Access Network (O-RAN) demands reconfiguration decisions that respond to live traffic conditions while remaining safe with respect to Service Level Agreement (SLA) constraints, a combination that purely data-driven controllers cannot reliably guarantee without continuous retraining. We present SlicePlanner, a logic-guided closed-loop framework that treats each RAN reconfiguration as a controlled hypothesis: proposed from live and historical KeyPerformance Indicator (KPI) evidence, filtered through explicit SLA constraints and mined RAN control rules, and committed only when post-action measurements confirm improvement. SlicePlanner comprises two cooperating agents: SliceSense, which monitors live KPIs, retrieves context-conditioned evidence, and ranks candidate RAN actions through a surrogate-guided admissibility filter; and SliceGuard, which validates post-action KPI responses, rolls back harmful configurations, and refreshes rules under distribution drift. Validated over the air using OpenAirInterface, FlexRIC, and commercial Sierra Wireless 5G devices, SlicePlanner reduces SLA latency violations by up to 86% over static slicing and 50% over Deep Reinforcement Learning (DRL)-based slicing, while maintaining comparable cell throughput and auditable decisions without online retraining. 

---
# How Much Effort Does an AI Code Reviewer Need? Evidence from Claude 5.5 Models on Plausible-but-Wrong Patches 

**Authors**: Zax Shen, Hanling Zhang, Zixu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.33639)  

**Abstract**: AI coding agents write code faster than people can review it, so teams add AI reviewers. Reasoning models let such a reviewer trade cost for diligence through an effort setting. We asked how much effort a reviewer needs to catch plausible-but-wrong patches, which pass a benchmark's tests but fail stronger ones. Claude Opus 5.5 (preregistered), Sonnet 5.5 and Haiku 5.5 (exploratory) each reviewed the same 90 agent-written Django patches from SWE-bench Verified at low, medium and high effort. By strengthened SWE-ABS tests, 55 patches were wrong and 35 correct. In every model, high effort accepted fewer wrong patches than low (false success .45 against .67 for Opus, .49 against .65 for Sonnet, .51 against .78 for Haiku), a supported difference for Opus and Haiku. For Opus, medium was non-inferior to high within 15 percentage points (difference .018, 95% CI [-.065, .108]) and cost 20% less, short of the preregistered 30%. For Sonnet and Haiku, medium cost 39% and 51% less than high but was not shown non-inferior. Opus at medium or high agreed best with the labels; Haiku at high cost about 8% as much per review as Opus at medium, accuracy difference unresolved. Exploratory coding showed that most rejections named the tested defect and that effort added such rejections. No setting is a strong gate: each accepted at least 45% of wrong patches. For 32 of the 55 wrong patches the issue may not require what the failing test checks; without them, neither accuracy hypothesis holds for any model. 

---
# Controllable CAD Generation and Interactive Editing of Fluid Rotors 

**Authors**: Shen Chen, Lei Li, Jiajun Sun, Zitian Wang, Huaping Zhang, Peng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2611.33617)  

**Abstract**: Computer-aided design (CAD) provides the geometric foundation for rotors in fluid rotating machinery. However, creating models for different rotors requires different modeling processes, making it difficult to control geometry through design intent and reuse existing models. We introduce VaneCraft, an agent-driven framework for rotor generation, inverse recovery, and editing through a unified analytical representation. The agent interprets design requests and invokes geometric modeling operations. The representation defines geometric controls through meridional geometry and independent boundary angle laws. An anisotropic partial differential equation (PDE) maps these controls to the blade surfaces. Geometric parameter correction (GPC) uses a joint geometric encoder and a parameter decoder to refine controls initialized from rotor category and dimensions. Inverse recovery maps existing CAD to the same geometric controls for reconstruction and continued editing. Experiments demonstrate that VaneCraft supports anisotropy control and parameter editing and outperforms representative baselines in rotor CAD generation tasks. Project page is available on \href{this https URL}{GitHub}. 

---
# Learning from Natural Images Untangles Number Sense 

**Authors**: Yechan Cho, Hyeonsu Lee, Emma Deneuville, Jaeson Jang  

**Link**: [PDF](https://arxiv.org/pdf/2611.33615)  

**Abstract**: Numerical cognition is a simple yet fundamental ability that remains challenging for modern vision-language models. Meanwhile, number sense is already present in numerically naïve animals, suggesting that it can emerge without explicit numerical learning. Deep neural networks provide a related observation, with numerosity-selective units appearing after training on unrelated natural-image tasks and even before any training. However, these findings leave unresolved whether visual learning contributes beyond the spontaneous emergence of numerical selectivity. Human development provides a clue, as children become better at making judgments in which numerosity and proportion conflict. Here, we introduce a geometric framework to trace how representations of these two quantities change across learning. Although untrained and pretrained networks exhibited similar unit-level numerosity selectivity and basic comparison performance, natural-image pretraining increased the angle between numerosity and proportion subspaces while reducing their overlap, supporting more robust readout under conflict. Direct manipulation of these geometric properties further demonstrated their functional contributions to decoding accuracy and functional independence. Importantly, conventional supervised pretraining was not essential for quantity separation, which also emerged through self-supervised learning without category labels and synthetic-image training without natural-image exposure. Together, our findings suggest that structured visual learning can refine spontaneously emerging number sense into robust and functionally independent quantity representations without explicit numerical training. 

---
# Aligned Machine: A Benchmark for Evaluating Multimodal Embedding Similarity via Human Comparative Judgements 

**Authors**: Jiechen Li, Hannah Groos, Brinnae Bent  

**Link**: [PDF](https://arxiv.org/pdf/2611.33594)  

**Abstract**: Embedding models increasingly inform how people search, recommend, and retrieve information, yet we lack credible ways to test whether these similarity judgments reflect how humans understand meaning. Existing benchmarks typically rely on static, expert-labeled pairs with coarse Likert-style annotations, limiting scalability and making it difficult to obtain consistent, high-confidence annotations. We introduce Aligned Machine, an interactive, public-facing platform that transforms broad participation into a continuously growing Human-Aligned Similarity Benchmark (HASB) for evaluating semantic alignment in embedding-based systems. Participants complete a cognitively-natural, forced-choice task, comparing two pairs of items (text or images) and selecting which pair is more similar. This comparative design is accessible to non-experts, yields clean preference data, and supports robust aggregation. We report results from an initial deployment with 115 participants contributing 2,875 total labeled pairs and 872 unique pairs. Using HASB, we evaluated 12 widely used embedding models and found only partial alignment between embedding similarity and human judgments. Text-only models consistently achieve higher overall agreement than multimodal models, while multimodal performance varies systematically by task type: alignment is strongest for image-image comparisons, weaker for image-text comparisons, and lowest for text-text comparisons. Aligned Machine is designed to evolve over time as participation grows, supporting longitudinal analysis, domain-specific studies, bias detection, and measurement of concept drift across model versions. By combining accessible public participation with rigorous comparative evaluation, HASB provides a scalable and accountable framework for auditing semantic alignment in embedding models. 

---
# FIG-MAC: A Fine-grained Inspiration Graph Empowered Multi-Agent Collaboration for Automated Scientific Hypothesis Generation 

**Authors**: Siyuan Chen, Yuping Gao, Tai Huang, Yuling Zhou, Dongyu Su, Tongguan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.33563)  

**Abstract**: Automated Scientific Hypothesis Generation (ASHG) leveraging large language models (LLMs) aims to harness their strengths in knowledge integration, logical reasoning, and text generation for efficiently producing innovative, verifiable scientific hypotheses. However, existing approaches face three critical limitations: (i) insufficient fine-grained knowledge representation, as unstructured context impairs LLM comprehension and reasoning; (ii) lack of knowledge evolution traceability, constraining the generation of innovative cross-domain conjectures; (iii) absence of human-inspired and role-specialized multi-agent collaboration, failing to leverage the efficient innovation mechanisms arising from cognitive diversity and division of labor among expert roles to complete the full ASHG-evaluation loop. To address these limitations, we propose the Fine-grained Inspiration Graph-empowered Multi-Agent Collaboration framework (FIG-MAC) including: (i) Fine-grained Inspiration Graph module (FIG), to obtain fine-grained structured knowledge representations, where nodes represent academic semantic units and edges encode heuristic relationships among them; (ii) ontology-guided hybrid reasoning module, which follows the FIG ontology to construct structured reasoning paths and uses vector search to enrich them with semantic evidence, to obtain the traceable scientific conjecture paths; (iii) role-specialized Multi-Agent framework, inspired by real scientific teamwork, enabling a closed-loop process of scientific conjecture generation, screening, and validation. Experimental results demonstrate that FIG-MAC significantly outperforms state-of-the-art methods on source diversity, provenance-adjusted novelty, and raw novelty. 

---
# PeerReveal: Learning to Generalize in AI Peer Review Detection 

**Authors**: Zi Li, André V. Duarte, Lei Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.33555)  

**Abstract**: Can we distinguish fully AI-generated reviews from those using AI only to rewrite human judgement? Furthermore, can we do so reliably when reviews are generated from diverse AI models and workflows across scientific domains? To this end, we introduce PeerReveal, a new detector that grounds its decisions in learned relationships between a review's claims and reference claims drawn from AI-generated reviews of the same paper, complemented by review-level textual signals. We evaluate PeerReveal on PeerShift, a new, challenging test set of 5,430 reviews spanning multiple domains, including machine learning, life sciences, and social sciences, combining variation in disciplinary review conventions with diverse AI review-production mechanisms. With regular training, PeerReveal achieves 70.52% pooled Macro-F1 on this benchmark, improving over the strongest baseline by 11.48 points. We develop an effective and practical adversarial training framework that further improves the detector's generalization using a small local generator optimized to produce challenging reviews, raising pooled Macro-F1 from 70.52% to 82.40%. These results show that robust AI-review detection benefits from richer review-level evidence and adversarial training on more challenging generation processes. 

---
# Learning to Reason in Real-Time 

**Authors**: Xavier Schneider, Benjamin Schneider, Sun Sun, Olga Vechtomova, Victor Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2611.33513)  

**Abstract**: AI agents that operate in the real world must reason correctly and on-time, yet they currently do not account for how time should alter their reasoning. Modern agents learn to do tasks in environments that are entirely synchronous: where the world waits for the agent to act before changing. A real-time environment, by contrast, changes independent of the agent. Existing approaches try to bridge this gap by imposing a ratio of thinking to acting. However, for an agent to succeed in real-time it must learn to act quickly when delay is costly and to reason longer when the benefit outweighs the missed opportunities. To this end, we introduce real-time reasoning training (RRT) as the class of training techniques for real-time reasoning agents. We develop three standard RRT techniques: (1) using real-time data, (2) decoupling acting and observing, and (3) curriculum learning in time. Together, these simple techniques substantially outperform strong baselines. Across time pressures, we outperform the strongest non-RRT baseline by an average of 33% reward on in-distribution and 55% reward on out-of-distribution tasks in two real-time environments: Frogger and Snake. These results demonstrate that we can teach agents to balance reasoning and responsiveness under real-time constraints, and that RRT opens a broad, new set of capabilities for real-time agents. 

---
# CA-RAG: Constraint-Aligned Retrieval for Structurally Faithful Long-Form Generation 

**Authors**: Vimal Kumar Kumar, Meghana Kshirsagar, Conor Ryan  

**Link**: [PDF](https://arxiv.org/pdf/2611.33497)  

**Abstract**: Retrieval-augmented generation (RAG) typically optimises evidence selection for semantic relevance, whereas compliance-sensitive long-form generation also requires evidence supporting explicit structural obligations. Semantically relevant retrieval may therefore omit information needed for mandatory downstream clauses. We propose Constraint-Aligned RAG (CA-RAG), a retrieval-conditioning framework that augments section-specific queries with ontology-derived structural constraints and policy-grounded knowledge-graph relations before evidence selection. Retrieval-only evaluation with manually annotated evidence shows consistent improvements across retrieval budgets. At $k=6$, CA-RAG increases Precision@$k$ from $0.5268$ to $0.5692$, pooled Recall@$k$ from $0.5402$ to $0.5950$, and F1@$k$ from $0.5389$ to $0.5672$ over standard RAG. Generation experiments further show improvements in compliance and reliability in several model--domain settings, while generation-only constraints achieve higher clause coverage but lower overall compliance and reliability. These results indicate that structurally aligned retrieval can improve evidence sufficiency in compliance-sensitive long-form generation. 

---
# Dynamic Reasoning in Next-Gen Detection and Response: An Empirical Study of LLM Triage, Reinforcement Tuning, and Adversarial Evasion 

**Authors**: Jeremy Canale  

**Link**: [PDF](https://arxiv.org/pdf/2611.33470)  

**Abstract**: We present an end-to-end empirical study of dynamic reasoning in next-generation detection and response pipelines, evaluating how static classifiers, graph-based signal fusion, and large language model (LLM) triage agents perform under strict human-analyst attention budgets (Recall at B alerts/day). Across 56 reproducible experiments spanning a multi-layer enterprise-and-cloud simulator and four public datasets (SOREL-20M, BODMAS, LANL, and DARPA Transparent Computing), we test twelve architectural claims governing modern fleet-scale defense. First, we show that fleet growth halves recall at a fixed analyst budget even at negligible compute cost (<$0.0003 per million events), and that rarity-weighted graph risk propagation outperforms GraphSAGE without neural training. Second, evaluating frontier reasoning models (Gemini) and fine-tuned open-weight models (Qwen3-4B) for alert triage reveals that unconstrained LLM auto-closure eliminates 90% of false positives while silently discarding 31-44% of true attack cases. Third, post-training with Group Relative Policy Optimization (GRPO) causes severe reward hacking on unseen attack classes, suppressing up to 86% of novel attacks compared to two-epoch supervised fine-tuning. Finally, against an adaptive adversary generating plausible cover stories in benign telemetry fields, LLM triage evasion rises sharply unless gated by a deterministic policy harness prior to model invocation. 

---
# Collaborative Language Models for Semantic IoT Data Processing: Validation through Zero-Shot Human Activity Recognition 

**Authors**: Mohsen Shirali, Kean Ooi, Anna Knab, Mohammadreza Fani Sani, Estefania Serral  

**Link**: [PDF](https://arxiv.org/pdf/2611.33411)  

**Abstract**: Internet of Things (IoT) systems generate low-level sensor observations that are not directly suitable for process analysis and must first be transformed into semantically meaningful events. This paper investigates collaborative language models as a zero-shot semantic abstraction layer for converting such observations into higher-level event and activity labels. We propose a five-stage workflow comprising semantic mapping, semantic aggregation, temporal and contextual aggregation, specialized reasoning, and final event abstraction. The resulting labels and contextual information can subsequently be represented as event records for downstream Process Mining (PM) tasks. The approach is evaluated through Human Activity Recognition on the CASAS Aruba smart home dataset using eleven workflow configurations. The experiments show that collaboration alone does not guarantee improved performance: general-purpose auxiliary SLMs can degrade recognition, whereas assigning complementary specialist roles leads to better results and allows a smaller final classifier to retain much of the performance of a larger model. The selected workflow achieves a macro F1 of 0.643 in the 1.5-day comparison and 0.567 (weighted F1 0.680) in a one-week evaluation. Although trained ML approaches remain more accurate on this dataset, the results demonstrate a different design point: modular semantic event abstraction using off-the-shelf models without task-specific parameter fine-tuning. The findings indicate that the structure of LM collaboration and task decomposition is more important than simply increasing the number of participating models. 

---
# Partial Routing: Trace-Tuned Delegation for Resource-Efficient Agentic Workflows 

**Authors**: Siddharth Arya, Kshitij Jain  

**Link**: [PDF](https://arxiv.org/pdf/2611.33402)  

**Abstract**: AI models vary enormously in capability and price. Model routing exploits this by sending easier tasks to cheaper models. However, a single agentic task often combines easy and difficult subtasks, making one model choice for the entire task inefficient. We introduce partial routing, which guides a frontier lead model to delegate suitable subtasks to a cheaper worker model. From both models' past runs, we identify recurring subtasks and measure each model's quality and cost on them. These measurements produce delegation guidance, with a parameter, $\delta$, controlling how much is delegated. The lead decides when and what to delegate during execution, without training either model. On held-out tasks from ViBench and Terminal-Bench 4.0, partial routing achieves higher mean task performance than the lead alone and task-level routing. With $\delta = 0.75$, it raises the lead's ViBench score from 70.2 to 78.5 at 19% lower cost, and its Terminal-Bench pass rate from 54% to 61% at similar cost. Giving the lead only the subtask breakdown, without a worker, also reduces cost on both benchmarks and raises the Terminal-Bench pass rate. These results show that analysis of past runs can support more efficient planning and more effective delegation, with little cost and implementation overhead. 

---
# Confidence scores for gated automation of large language models 

**Authors**: Luiz G. A. Alves, Aaron Vermeersch  

**Link**: [PDF](https://arxiv.org/pdf/2611.33394)  

**Abstract**: AI agents built on large language models (LLMs) demonstrate strong capabilities in automated decision-making. However, LLMs may produce incorrect answers with high confidence, limiting their applicability in decisions that affect individuals and organizations. Reliable confidence scores are necessary to gate such decisions, and existing estimates are frequently unavailable or too coarse to support effective gating. This work uses supervised learning to train models to predict the correctness of LLM-generated answers, enabling automated confident responses and routing uncertain ones for human review. A linear confidence score head is fitted to the final-layer hidden state of the frozen LLM backbone, and text-only metamodels are trained as an alternative for closed models. All results come from a single 2B-parameter backbone, on graded answers that are about 98% SQuAD and the remainder TruthfulQA. Against multiple baselines, the confidence score head outperforms all baselines and metamodels, automating the largest fraction of the workload at high accuracy targets. Prompted confidence scores can appear well calibrated yet lack the resolution needed for thresholding. Although transferability to unseen datasets is modest, the confidence score head remains the most robust approach. The proposed confidence scoring framework enhances the safety of automating LLM-based agents and can be adapted to gate other properties, including safety or compliance. 

---
# Brand Recall and Category Accuracy in Language Models: An Audit of 1,000 B2B SaaS Brands 

**Authors**: Ben Moore  

**Link**: [PDF](https://arxiv.org/pdf/2611.33322)  

**Abstract**: A language model can name a brand without classifying it correctly. This study measures these outcomes separately for 1,000 B2B SaaS brands across 333 product categories. On Gemma 4 12B, with no external retrieval, category prompts elicit 510 brands (51.0%). A separate five-candidate category probe ranks the reference label first for 787 brands, a distractor first for 194, and yields little separation for 19. Among the 981 separated rankings, the error rate is 19.8%. It falls to 3.5% for the 547 brands with a winner-to-runner-up log-probability margin above 0.6. These are within-cohort measurements, not calibrated confidence estimates. Exploratory regressions associate technical coverage with recall and an owned-properties composite with category accuracy, but the composites use different adjustment sets and do not establish causal effects. Across four Gemma generations, measured recall is 39.8%, 44.3%, 51.1% and 51.0%. A separate representation-based sentiment measure has little linear association with customer ratings (r=0.03, n=921). The findings support reporting recall and category accuracy separately. Failure to elicit a brand does not establish its absence from model parameters or training data. 

---
# The Platonic Representation Hypothesis in Jacobian-Space: Language and Vision Models Agree Approximately Up to a Rotation 

**Authors**: Ross Murphy, Gabriel Brostow  

**Link**: [PDF](https://arxiv.org/pdf/2611.33304)  

**Abstract**: The Platonic Representation Hypothesis (PRH) posits that AI models converge toward a shared representation of the world, even when trained on different data modalities. Previous work has found evidence for the PRH in the activations of vision and language models. In this work, we ask whether vision and language models also converge in how they act on these representations: precisely, in how these internal representations affect the models' output. At a chosen layer of a transformer, the Jacobian lens gives every word in the vocabulary (or class in an image classifier) a vector in the hidden state: the vector that most increases the model's tendency to generate that word, at that position or later, or to predict that class. Collectively, these vectors make up what is referred to as the model's J-space. We find evidence of the PRH in J-space across vision and language models, and furthermore, we find that J-spaces mostly converge up to a rotation. That is, a single rotation, fitted on a subset of words, approximately maps one model's J-space onto another's, and so applies to words and classes it was not fitted on. A word's vector, rotated from a text-only language model's J-space into the visual patch tokens of a vision-language model, changes the object the vision-language model says is in the image (and only in the image region where it is added). Rotated into an image generator's text conditioning, a word's vector changes the object in the generated image. Rotated into an ImageNet classifier (trained without text), it raises that object's class score above almost all others, and vice versa. This suggests that models converge not only in how they represent the world, but also in how they act on it. 

---
# Certify What You Plant: Shortcut-Free Proof Depth in Logical Reasoning Data 

**Authors**: Baptiste Juran, Dimitri Kachler, Damien Sileo  

**Link**: [PDF](https://arxiv.org/pdf/2611.33303)  

**Abstract**: Procedurally generated logic problems are used to both evaluate and train reasoning in language models. Their difficulty is usually set by planting a proof, that is, by sampling a proof of the desired depth and stating its premises as the problem. However, a planted proof is only one possible proof. A problem can also have shorter proofs, or several unrelated ones, and its stated depth then overstates what solving it requires. We measure what a problem actually requires, relative to a stated set of inference rules. For each problem, we compute the steps every proof must make, the shortest proof, the number of minimal proofs and the indispensable premises, and export certificates that an independent checker verifies. Audits of eight datasets find depth that is forced (ProofWriter, Blocksworld), problems labelled depth 5 that require about two specific rule applications (FLDx2), and answers that need no particular step (ZebraLogic). We then construct problems whose every proof has a requested structure, using a catalog of shortcuts proved once per rule set. Every constructed problem is checked in full and comes with its own certificate. In training, the best data mixes depths and problem types. At equal size, such mixtures of certified problem types, including model-checking problems that forward generation does not produce, transfer better than FLDx2-style derivation data for Qwen3 models from 0.6B to 8B. 

---
# What, Who, and When: Structured Harm Assessment for Mental Health Safety via Short Reasoning 

**Authors**: Catarina Botelho, António Farinhas, John Mendonça, Areti Vassilopoulos, Inês Gonçalves, Maya D'Eon, Hugo Pitorro, Laura Cabello, Carrie Spadine, Nuno M. Guerreiro, Ricardo Rei  

**Link**: [PDF](https://arxiv.org/pdf/2611.33263)  

**Abstract**: The growing use of large language models (LLMs) for mental health support raises important safety concerns. Existing guard models primarily identify categories of harmful content, often overlooking the context needed to determine an appropriate response. We introduce a clinically grounded taxonomy that characterizes harm along three dimensions: what the harm is, who is involved, and when it occurs. It further separates harm characterization from intervention decisions, thus enabling safety policies to be specified outside the model weights through explicit, clinician-curated instructions that can be updated and audited. We validate the taxonomy with expert clinicians and validate a panel of LLM judges to scale annotation. Relabeling public benchmarks in this taxonomy shows that they rarely contain harm directed at the user, which accounts for nearly half of the harm in real conversations. We then train MindGuard2, a 35B-A3B reasoning-based guard model using supervised fine-tuning on short reasoning traces followed by reinforcement learning with a reasoning-length penalty, to minimize inference overhead while preserving the benefits of reasoning. On two public benchmarks and real-world conversations, MindGuard2 achieves balanced accuracy comparable to the substantially larger GLM-5.3 at approximately 10x lower median latency, advancing the accuracy-latency Pareto frontier. Furthermore, it achieves low false-positive rates at high recall, helping balance the need to detect unsafe situations against the risk of unnecessary interventions that may disrupt support and undermine user trust. We release a safety guardrail model with SOTA performance trained on synthetic data, together with synthetic conversations and annotations of public benchmarks in our taxonomy. 

---
# General-Purpose Decision Models for AI Configuration Tuning 

**Authors**: Jingzhi Gong, Jie M. Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.33190)  

**Abstract**: AI configuration tuning searches for settings that improve the performance of AI systems under a limited budget. Existing tuners use different sampling and search policies, making the choice of decision strategy central to both tuning quality and cost. We investigate whether general-purpose decision models can guide tuning, using Jev as a deployed representative that returns structured choices and numeric scores across diverse tasks. To assess where it helps, we compare its use as an optimizer, performance predictor, and region guider against conventional methods and a large language model (LLM) on eight AI tuning tasks. Conventional optimizers attain rank 1 in six of eight comparisons. Similarly, established predictors outperform Jev in seven. Through region guidance, however, it finds configurations with lower perplexity than those selected by the native tuner, offering a quality gain when additional decision time and API fees are acceptable. These findings yield actionable recommendations on selecting the decision role, retaining conventional predictors, and balancing tuning quality against cost. 

---
# AutoformBench: A Lean Formalization Benchmark for Advanced Mathematics 

**Authors**: Charles Arnal, Ahmad Rammal, Niket Patel, Fabian Gloeckle, Bartosz Piotrowski, Amaury Hayat, Julia Kempe, Rémi Munos, Vivien Cabannes  

**Link**: [PDF](https://arxiv.org/pdf/2611.33165)  

**Abstract**: AI systems are increasingly capable of producing machine-checked Lean proofs of advanced mathematical results. Real mathematical formalization, however, is not isolated theorem proving: it requires building the definitions and intermediate lemmas that connect a result to a formal library. Existing evaluations often provide isolated theorem statements in settings where much of this supporting theory is already available in Mathlib. We introduce AutoformBench, a benchmark of 341 Lean 4 formalization tasks drawn from 24 advanced mathematical sources. Lean-experienced mathematicians formalized and audited each target statement, leaving the corresponding Lean proof to the evaluated agent. Each problem is embedded in a broader mathematical development: agents receive the informal proof but must translate it into Lean while constructing the missing definitions, intermediate lemmas, and supporting infrastructure. We compare the performance of frontier models on these problems. We find a largely shared difficulty frontier: success depends less on mathematical domain than on models' ability to construct missing formal infrastructure. Data and code at this https URL 

---
# Constraint-Aware Test-Time Scoring for LLM and VLM Candidate Selection 

**Authors**: Rangarajan Ramesh, Yongchao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.33161)  

**Abstract**: When an LLM or VLM produces several candidate answers, picking the best one usually relies on the model's own confidence, a learned judge model, or a domain check. Each has a known weakness: an answer can be fluent but insecure, mathematically plausible but physically infeasible, or rated highly by a judge while quietly breaking a task requirement. We propose a scoring framework that combines a small set of hard checks every answer must pass with a wider set of softer checks whose scores are blended into one number per candidate. This adaptive scoring framework consists of three components: a drafting model that produces the candidates, a configuring model that selects and weights the checks for each task, and scoring models that rate properties no grounded tool can compute. We apply our proposed method to four tasks: writing code that is both correct and secure (CWEval), generating robot manipulation plans (LIBERO with added safety variants), passing competitive-programming hidden tests (LiveCodeBench), and answering questions from retrieved documents (HotpotQA). An ablation study isolates the contribution of each component and finds that the largest gains come from combining hard checks with candidate comparison: on tasks where the first sampled answer is often broken or unsafe, this pairing lifts joint correctness-and-safety by up to +0.21 (retrieval-augmented QA) and code-generation safety by up to +0.28 (CWEval), with the size of the gain tracking how weak the first sample is, falling to near zero or slightly negative where it is already strong. The per-task layer that selects and weights the softer checks adds no measurable value over simpler alternatives. This builds a unified scoring framework, together with an ablation that identifies the grounded hard checks, not the soft-factor configuration, as the source of the gain. 

---
# Context Scaling: A New Frontier of AI Scaling 

**Authors**: Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2611.33160)  

**Abstract**: Many failures of deployed AI arise not from weak inference but from a mismatch between the state a task depends on and the context available when the system acts. Relevant user state, repository revisions, tool traces, sensor readings, or environment changes may be unavailable, distorted, or poorly selected. We call the effort to reduce this bottleneck context scaling. We frame a model as $P_\theta(d\mid c)$ within a system governed by three budgets: training $B_\theta$, decision-time $B_d$, and context-infrastructure $B_c$. Pretraining scaling invests in $\theta$; decision-time scaling searches, refines, or verifies decisions under supplied context; context scaling improves the system's capacity to acquire, represent, deliver, evaluate, and persist deployment state for model use. The three programs interact, but only $B_c$ targets what the system can make available before a decision. This view makes the model-infrastructure pair the unit of analysis. We define effective-context value as a task- and intervention-dependent attribution, separate deployment state from model-visible context, and identify availability, fidelity, selection, and utilization failures. We then organize current mechanisms through a five-operation context lifecycle and propose controlled budget interventions and utility-cost curves for evaluation. The resulting agenda centers on grounded reporting, interaction-data and full-cost infrastructure, and safety and governance for persistent, adaptive systems. 

---
# Spectrum Adaptation of AI-Native Autoencoder Communications with Transfer Learning 

**Authors**: Yalin E. Sagduyu, Tugba Erpek  

**Link**: [PDF](https://arxiv.org/pdf/2611.34942)  

**Abstract**: End-to-end physical-layer learning replaces separate modulation, coding, and detection operations with an autoencoder (AE) that consists of a jointly trained encoder-decoder pair at the transmitter and the receiver. This paper studies AI-native communications with AEs and quantifies the sensitivity to training-test mismatch in channel models. Models trained only on an AWGN channel achieve low bit error rates (BERs) under matched conditions but degrade severely on Rayleigh fading with practical front-end impairments. To reduce the time and energy required to adapt to new channel conditions, we propose a low-latency transfer-learning procedure that starts from an AWGN-pretrained system and fine-tunes only a small subset of transmitter and receiver layers while freezing the remainder. We further formulate layer selection as a budget-constrained optimization problem that identifies the most informative parameters to update during adaptation. This preserves the learned signal geometry and the deep decision structure while adapting lightweight front ends to fading- and impairment-induced mixing. For AE communications, we show that partial updates remove the mismatch-induced error floor and approach the BER of fully retrained models. We provide a layer-wise study and quantify resource-performance trade-offs, showing that modest adaptation substantially reduces BER while cutting training time, multiply-accumulate (MAC) operations, energy consumption, and the fraction of updated parameters for resource-efficient implementation. 

---
# Attack Policy Selection can Mislead Monitor Evaluators 

**Authors**: Charlie Griffin, Oliver Makins, Asa Cooper Stickland, Alessandro Abate, Yarin Gal, Abby D'Cruz, Xander Davies  

**Link**: [PDF](https://arxiv.org/pdf/2611.34934)  

**Abstract**: AI monitors, classifiers that flag dangerous behaviour by AI agents, are often evaluated against transcripts generated by attack policies acting as proxies for dangerous behaviour in deployment. We evaluate an ex-production OpenAI monitor as a case study and find that measured false negative rate is highly sensitive to both the method used to generate attack policies and the level of optimisation applied within that method. This rate ranged from 0.5\% to 20.8\% across optimisation levels of a single method (reinforcement learning), and from 6.4\% to 23.0\% across methods at their best observed optimisation level (optimised prompting vs.\ transcript editing). Every attack policy was developed against proxy monitors only: all reported results show attacks transferring to the held-out GPT-5.4 target monitor. Our results suggest that monitor evasion evaluations can easily mislead: an evaluation using only basic attack methods may give false confidence that a monitor is strong.
The sensitivity to attack-policy choice raises a problem: interpreting any particular result as evidence about deployment performance requires justification. The transcripts produced by attack policies are proxies for the dangerous behaviour a monitor must flag in deployment. These proxies can introduce visible artefacts, such as reasoning about a ``side task'' absent from the instructions shown to the monitor, that affect monitor judgements. Given the sensitivity above, the choice of proxy can change an evaluation's conclusion. Reading these results as evidence about deployment therefore requires a separate legitimacy argument for why the proxies are informative, including any justification of conservativeness. Making convincing arguments remains an open problem. 

---
# Diverse Alternative Reasoning Traces for Hard Math Problems 

**Authors**: Lukas Aichberger, Adam Golinski, Arno Blaas, Arwen Bradley, Jacob Springer, Mehrdad Farajtabar, Michael Kirchhof, Sinead Williamson  

**Link**: [PDF](https://arxiv.org/pdf/2611.34928)  

**Abstract**: Post-training large language models (LLMs) for mathematical reasoning often relies on accepted traces whose final answers pass a verifier. For hard math problems, however, such traces can be rare under independent Monte Carlo (MC) sampling. To address this, we propose Diverse Alternative Reasoning Traces (DART), a trace-mining method that starts from a rejected trace, decomposes it into reasoning steps, and ranks them by position-weighted uncertainty. It then generates deliberately distinct alternative candidates for highly ranked steps using reference-answer guidance and a meta-thought before completing the resulting traces. On greedy-unsolved DeepMath-103K problems, DART recovers 37% more problems than MC sampling at a matched number of generated traces. Its estimated generation cost per recovered problem is nearly identical, while its cost per accepted trace is 30% lower. Post-training Llama-3.1-8B-Instruct on traces mined by DART achieves the highest point estimate in five of six evaluations across MATH-500 and AIME benchmarks. These results demonstrate that revisiting rejected traces with DART can expand problem coverage and provide effective post-training data. 

---
# $μ$-Bench: Benchmarking Multi-User Task Completion in Shared Interaction Voice Agents 

**Authors**: Qihao Lin, Guanxu Chen, Shuhang Zheng, Xiaoya Lu, Jingyi Yang, Terumi Chiba, Wenyi Yu, Chao Zhang, Zhuan Shi, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2611.34920)  

**Abstract**: Voice agents in shared spaces must satisfy interdependent multi-user requests while upholding per-user authority, privacy, and task constraints. Existing evaluations offer limited coverage when participants react throughout a session. We introduce $\mu$-Bench, a benchmark for joint task completion in closed-loop, full-duplex multi-user voice interaction. Simulated participants act from isolated profiles of goals, knowledge, and permissions, adapting to the agent's speech and tool outcomes. Across 20 domains, 100 scenarios vary group size, acoustics, and follow-up demands to form 800 cases (408 tool-augmented, 392 tool-free). Joint Task Success (JTS) requires all user goals, privacy boundaries, and terminal states to hold jointly, while Pure JTS further disallows unauthorized tool calls; Attribution, Threading, and Coordination (ATC) diagnostics localize failures. Across 13 voice agents, proprietary systems peak at 62.24\% tool-free and 65.44\% tool-augmented success, while open-weight models remain below 20\%. ATC diagnostics show that open-weight models lag primarily in multi-turn attribution and threading, and controlled contrasts reveal that larger groups consistently degrade joint task success across capable systems. By coupling reactive multi-user simulation with verifiable session-level outcomes, $\mu$-Bench provides a rigorous testbed for shared voice agents. 

---
# Enhancing Maritime Domain Awareness using Latent Space Representations 

**Authors**: Rikard Vinge, Miguel A. Belenguer-Plomer, Isabelle Wittmann, Octavian Dumitru, Michele Lazzarini, Sergio Albani, Conrad M Albrecht  

**Link**: [PDF](https://arxiv.org/pdf/2611.34899)  

**Abstract**: This work presents a novel application of Earth Observation (EO) embeddings for Maritime Domain Awareness (MDA) from the perspective of the geospatial foundation model TerraMind. By operating directly on compressed latent-space representations rather than on the original Sentinel-1 imagery, this approach directly addresses the data gravity bottleneck that increasingly constrains operational EO processing, in line with the objectives of the Embed2Scale project. In particular, linear and non-linear probing of compressed embeddings, latent space representations of synthetic aperture radar (SAR) imagery, is used to detect, localise, and characterise vessels from Sentinel-1 GRD images. Vessel detection achieves a ROC-AUC of 0.95, while vessel localisation has a median error of around 300 m for vessels longer than 100 m, and vessel AIS type-code classification achieves 62% accuracy for the majority classes. The latter, however, is highly correlated with vessel length. These results reveal both the potential and current limitations of Geospatial Foundation Models (GeoFMs) when applied to fine-grained MDA understanding in real-world scenarios.
Additionally, a new maritime remote sensing dataset is released, pairing about 2,000 Sentinel-1 images with temporally aligned AIS signals from commercial vessels. Through manual inspection, each candidate sample was verified for bounding-box quality and for consistency between the AIS-reported vessel length and the observed vessel extent, cross-checking vessel identity and location against publicly available vessel records and, where available, photographs. The dataset was made ready for use with the NeuCo-Bench neural compression benchmark framework, publicly available on HuggingFace through this https URL . 

---
# Interpretable and Accurate Hybrid Decision Trees with Selective Case-Based Splits 

**Authors**: Cristiano Landi, Alessio Cascione, Riccardo Guidotti  

**Link**: [PDF](https://arxiv.org/pdf/2611.34887)  

**Abstract**: Decision trees are among the most well-known interpretable models in Machine Learning. Over the years, a wide variety of strategies have been proposed to define split conditions within the greedy learning framework, aiming to guide instance routing and enhance predictive performance. These strategies range from axis-parallel and oblique splits based on feature values to proximity-based splits using exemplar instances and more complex hybrid approaches. However, no single method has emerged as universally superior. In this work, we present two key contributions to the construction of greedy decision trees. First, we propose a hybrid tree architecture that integrates both feature-based and case-based splits, allowing for heterogeneous splits throughout the tree. Second, we introduce a novel type of split that leverages a selective case-based approach: when a split is needed, our method searches for the most informative subset of features to define the split, effectively balancing the specificity of case-based methods with the simplicity of feature-based ones. Experiments on a diverse set of tabular datasets show that our method outperforms standard decision trees and achieves performance on par with ensemble methods, while producing shallow trees that retain or even improve predictive accuracy. 

---
# How Much Do Coding Agents Over-Edit, and Why? An Empirical Study of Unneeded Edits in Repository-Level Agent Patches 

**Authors**: Aakash Goel, Benjamin Steenhoek, Yu Hu  

**Link**: [PDF](https://arxiv.org/pdf/2611.34875)  

**Abstract**: Agent patches are larger than developers', which is widely read as over-editing. But patch size does not say whether the extra edits are needed, what they are, or where they come from. We introduce Delta Residue Attribution (DRA), which replays agent trajectories, traces every block (unit) of the final patch to the step that wrote it, reverts each unit, and reruns the task tests to decide whether the unit is needed. We apply DRA to 7,660 resolved GitHub Copilot CLI trajectories from 18 frontier models on SWE-bench Verified (6,728) and DeepSWE (932), and grade developer patches using the same method. Agent patches do change more lines than developers'; on SWE-bench Verified the majority of extra edits add tests (61%) and documentation (13%) while on DeepSWE, most extra edits are code (58%). On both benchmarks, agents' code is on average at least as necessary as developers' (mean $NP_{code}$ 0.887 vs. 0.864 on SWE-bench Verified, where 11 of 18 models leave less unneeded code than developers, and 0.797 vs. 0.785 on DeepSWE). On average, 11% of an agent patch's code units on SWE-bench Verified are still unneeded, and 89% of those change behavior. Debug steps write almost none of the unneeded code. Code necessity differs substantially by model; it is unrelated to resolve rate on SWE-bench Verified and negatively correlated with it on DeepSWE (Spearman -0.59, p = 0.01). On SWE-bench Verified, pruning with DRA using only the tests the agent can see (its own tests and the repository's) removes the majority of unneeded code but also some needed code (67% recall, 46% precision), and 89% of pruned patches still resolve the task; asking the agent to audit its own diff is more precise but less effective (36% recall, 66% precision). We recommend that benchmarks report code necessity next to resolve rate and that over-editing be addressed through model training and harness-level pruning. 

---
# Shape-Aware Three-Way Merge for RDF Knowledge Graphs: The schema defines the conflict: an implementation and evaluation 

**Authors**: Steve Brown  

**Link**: [PDF](https://arxiv.org/pdf/2611.34870)  

**Abstract**: Merging RDF serializations creates ordering conflicts and misses semantic ones. We describe and evaluate a three-way merge in which the schema defines a conflict: set algebra combines canonical triples, and a slot is contended only where a SHACL shape bounds its cardinality and both sides' changes cannot stand within that bound. Unbounded predicates union; contended slots go to a person. The same shapes graph validates the result. We do not claim the first three-way merge for RDF. Our contribution is an implementation in a production multi-agent store and an evaluation against a reimplementation of Quit Store's Context Merge overlap rule, which it extends rather than replaces, and six other strategies. Across five synthetic seeds, the shape-aware operator raised 33 conflicts, all true positives as judged by the same shapes that define them, with no triples lost or fabricated and no SHACL violations admitted. The context-overlap baseline, counted per slot, detected the same 33 conflicts but asked 845 questions. A replay built from recorded production incidents covers 105 human alias repairs (93 after excluding 12 chained pairs) and 939 duplicate-value incidents. It reproduces a triple-level limit: different names for one entity remain invisible (0 of 93 alias repairs detected, matching 0 of 21 synthetic cases). In a study of evolving shape contracts, merging shape declarations first handles the tested cardinality changes, while datatype evolution exposes a detector blind spot caught by post-merge validation. An oracle that reads no shapes agrees with the shipped operator on the facts and history of all 23 reachable single-slot cases. Under wrong shapes, a missing bound loses data silently and a spurious one only adds questions. The supported claim is fewer human decisions with nothing silently lost, for cardinality conflicts in one deployment, not general merge correctness. 

---
# A Survey of Security, Privacy, and Ethics of Large Language Models in Telecom 

**Authors**: Joy Bose, Srinivas Adyapak  

**Link**: [PDF](https://arxiv.org/pdf/2611.34868)  

**Abstract**: Large Language Models (LLMs) are increasingly integrated into telecommunications for customer service, network analytics, and operational support. While offering efficiency and innovation, their deployment raises critical privacy, security, and ethical challenges due to the uniquely sensitive and longitudinal nature of telecom data. This paper presents a survey of LLM risks and safeguards in telecom context. Our survey introduces a telecom-specific taxonomy linking privacy, security, and ethics with use cases and outlines open research challenges such as longitudinal data privacy, cross-border compliance, and scalable guardrails. 

---
# RePace: Reasoning about Demonstration Speedup for Faster Vision-Language-Action Policies 

**Authors**: Insup Lee, Hojin Jeon, Sunguk Shin, Seoungbin Bae, Taekwan Kim, Jinwoo Shin, Dongsu Han  

**Link**: [PDF](https://arxiv.org/pdf/2611.34864)  

**Abstract**: Since vision-language-action (VLA) policies can inherit the slow pace of human demonstrations, accelerating these demonstrations before fine-tuning can lead to faster VLA policies without an additional speed-selection module at deployment. Several offline speedup methods have focused on identifying segments that require less precision and assigning predefined speedup factors to the segments. However, even within a single segment, a speedup factor suitable for steady motion may be too high when precise alignment is required. To this end, this paper presents RePace, an offline reasoning framework that determines how much to accelerate each part of a demonstration. For the same recorded movement, RePace measures how much each candidate speedup factor shortens the sequence and compares the resulting observation-action pairs used for training. Given the task instruction, a pretrained language model uses the visual context and original motion of successive windows to interpret these candidate retiming effects and select a speedup profile for the complete episode. Experimental results demonstrate that RePace outperforms the baselines in overall task execution efficiency on both LIBERO and RoboCasa-Kitchen. On all three tasks with a gripper-equipped Franka Research 3 arm, RePace matches or exceeds the acceleration baselines in success rate and completes successful trials faster. 

---
# BOP-Refer: Benchmark for Monocular 3D and 2D Object Grounding 

**Authors**: Tomas Hodan, Vineet Bhat, Martin Sundermeyer, Jonathan Tremblay, Yash Patel, Médéric Fourmy, Stephen Tyree, Junwen Huang, Varun Burde, Martin Cífka, Eric Brachmann, Vincent Lepetit, Stan Birchfield, Jiri Matas, Carsten Rother  

**Link**: [PDF](https://arxiv.org/pdf/2611.34862)  

**Abstract**: We introduce BOP-Refer, the first benchmark for monocular 3D and 2D grounding of everyday and industrial objects sourced from 9 diverse BOP datasets. We pair images from these datasets with VLM-generated and human-verified referring expressions, and derive accurate ground-truth 3D and 2D bounding boxes of the referred objects from their 3D models and 6DoF pose annotations. Our systematic evaluation of 17 frontier VLMs across the major proprietary and open-weight families (Gemini, GPT, Claude, Qwen, Gemma) shows that the best model, Gemini 4 Carbon, reaches 72.6% AP_IoU2D in 2D but only 12.4% AP_IoU3D in 3D, despite the more lenient IoU thresholds in 3D. The predicted 3D boxes are usually centered on the target objects in the image but often off in depth by more than the object's size, suggesting that metric scale estimation, which needs reliable object size priors, is currently by far the main source of error. 

---
# From Prediction to Decision: Decision-Focused Fine-Tuning of Autoregressive Tabular Transformers for Cost-Sensitive Financial Events 

**Authors**: Ashkan Golgoon, Giacomo Domeniconi, Samuel Atkins, Sergio Servantez, Zain Khandwala  

**Link**: [PDF](https://arxiv.org/pdf/2611.34858)  

**Abstract**: Autoregressive transformers trained on tokenized tabular event streams are a simple and strong baseline for predicting financial events such as fraud. Yet institutions do not act on probabilities; they act on decisions whose consequences are asymmetric and heteroscedastic: the cost of a missed high-value fraud and that of a wrongly declined small purchase are not the same. The standard bridge is post-hoc Bayes-optimal thresholding of calibrated probabilities under a cost matrix. We ask whether fine-tuning the model to the decision objective can do better. We fine-tune the same pretrained model so that its restricted softmax over label tokens (already a distribution over business actions) maximizes expected utility under an institution-specified, amount-scaled cost matrix. Because the reward is a known function of (action, outcome, amount) and the outcome is observed in the data, the expected-utility objective and its gradient are available in closed form. This is example-dependent cost-sensitive learning whose per-example loss is linear in the predicted probability, with no sampling, critics, or advantage estimation. A KL anchor to the pretrained model and an auxiliary language-modeling term preserve calibration and the generative event model. A sampled, group-normalized (GRPO-style) variant provides the extension point to trajectory-level rewards. 

---
# DRBench Pro: A Knowledge-Centric Benchmark for Professional Deep Research Agents 

**Authors**: Han Huang, Yuchen Yan, Xingyu Wu, Yijun Li, Chao Deng, Jie Liu, Jin Ma, Yongliang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2611.34852)  

**Abstract**: Deep research agents gather information from multiple sources and synthesize it into reports on complex, open-ended questions. Existing benchmarks assess report quality along several dimensions, but lack a dedicated knowledge-centric evaluation of which research directions a report covers and how deeply it explores them. We introduce DRBench Pro, a benchmark that makes structured knowledge coverage the primary evaluation target. Our automated pipeline builds local knowledge trees from Wikipedia and uses sampled subgraphs to generate research queries and evidence-grounded rubrics. Each rubric corresponds to a relation in the tree, preserving the structure of the knowledge being evaluated. Agents receive only the research query. This structure allows us to measure breadth across major research directions and depth along related knowledge chains, with their geometric mean giving overall coverage. The benchmark includes 144 cases across 24 domains and 44,102 rubrics. Evaluating 16 deep research agents reveals different strengths in breadth and depth, with the best-performing agent reaching 57.04% overall coverage. Rubric judgments agree closely with human annotations, and agent rankings remain stable across the tested metric configurations. 

---
# Probing Visual Idiosyncrasies in Text-to-Image Models 

**Authors**: Yongqian Peng, Hongjie Li, Yichao Zhou, Chi Zhang, Yixin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2611.34843)  

**Abstract**: Text-to-Image (T2I) models turn short text descriptions into images, and the leading models now produce images of comparable visual quality. Whether their outputs still carry model-specific characteristics matters for attribution, for model comparison beyond aggregate quality, and for evaluations that assume anonymity. We ask whether images from frontier T2I models can still be classified by source, and which visual properties make this possible. Here we show that a standard ResNet-50 identifies the source of an image among HiDream-O1-Dev, FLUX.2-dev, HunyuanImage-3.0, and Qwen-Image-2512, the four highest-ranked open-source models on the Text-to-Image Arena, with 100.00% accuracy on held-out prompts. Human participants reach only 62.61% on the pairwise version of this task, so the distinguishing cues are not visually obvious. They are also not tied to a particular classifier and survive severe image corruptions and out-of-distribution prompts. We trace them to visual idiosyncrasies at three levels. Low-level texture statistics alone let a lightweight multi-layer perceptron match ResNet-50. Object layout, depth, and albedo estimated from the images each carry a source signal. Image captions reveal model-specific preferences in style and background that the prompts never specify. A one-sentence hint about these cues raises human accuracy by 9.03 percentage points. These idiosyncrasies make source classification a tool for measuring similarity between models and between real and generated images, and they caution against assuming anonymity in human preference evaluation of T2I models. 

---
# Repairing the Trustworthiness of Distilled Language Models 

**Authors**: Kumar Shubham, Devendra Vyas, Vidushi Agarwal, Jayin Khanna, Prathosh AP  

**Link**: [PDF](https://arxiv.org/pdf/2611.34831)  

**Abstract**: Knowledge distillation enables the deployment of large language models through smaller student models, but the compression process can introduce regressions in trustworthiness even when downstream performance is preserved. Repairing such failures is challenging because the original distillation corpus is often unavailable, while direct preference optimization may alter capabilities inherited from the teacher. We propose a targeted repair framework based on the proximal Bregman response function (PBRF) that addresses both limitations. We estimate the curvature required for PBRF using a surrogate proxy dataset, removing dependence on the original transfer corpus, and extend the framework to preference-based supervision through an odds-ratio objective. This enables the use of preferred and dispreferred response pairs while constraining changes to the distilled model. Our results show that the proposed approach improves targeted trustworthiness behaviors for both pre-training and post-training models while largely preserving downstream performance, providing a practical alternative to re-distillation or unconstrained preference optimization. Our experiment results show that across different distilled models, the trustworthy regression can be improved by up to 8% with less than 1% decrement in perplexity. 

---
# Not all world models can doubt: how training objective and architecture shape epistemic uncertainty in JEPA and RSSM world models 

**Authors**: Simon Baur, Wojciech Samek, Jackie Ma  

**Link**: [PDF](https://arxiv.org/pdf/2611.34830)  

**Abstract**: World models predict how an environment responds to actions, so that candidate actions can be evaluated before execution. Executed actions, or decisions, taken in this manner are only as reliable as the model's predictive capabilities, and predictions in turn are only reliable within the model's experience. Epistemic uncertainty (EU) is the signal that marks where that experience ends. In this work we offer an extensive empirical study on how training objectives and architectures shape epistemic uncertainty of world models. More precisely, we present a controlled comparison of post-hoc EU (last-layer Laplace and ensembles) between two dominant models of different paradigms: 1) LeWorldModel (LeWM): A joint-embedding predictive architecture (JEPA), which predicts the next embedding without a decoder, and 2) DreamerV3: A recurrent state-space model (RSSM), which reconstructs the observation from a recurrent latent state. In ten simulated visual-control environments, we follow a controlled state perturbation of the environment state through every stage of the models up to the final EU estimate. We find that the RSSM attenuates the state perturbation through two mechanisms before it reaches an estimator on the predictor feature. Its recurrent update averages over context, so a change in a single frame reaches the state at a fraction of its size and recovers only as the change persists. Its reconstruction objective weights an entity by the pixels it covers, so a small object can be lost due to the training objective. The JEPA has neither mechanism and passes the change through all stages to the uncertainty. We further show that this translates to gaps in five uncertainty related downstream tasks, from unfamiliar-state detection to uncertainty-penalised planning: the RSSM uncertainty misses deviations that its own posterior registers, whereas the JEPA uncertainty reports what its model sees. 

---
# Transferring Architecture as an Optimization Prior 

**Authors**: Vighnesh Subramaniam, Boris Katz, Brian Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2611.34816)  

**Abstract**: Initialization determines the optimization "fate" of a model, going far beyond just setting the scale of its weights. It establishes a fundamental initial geometry over the input data, dictating certain aspects such as which examples are considered close and which representations gradient descent will naturally refine first. In this paper, we view the initial geometry of a network as an optimization prior and ask whether a network can change its optimization prior by referencing a completely different architecture. To investigate this question, we use an initialization procedure where a target network is aligned to the representational geometry of a randomly initialized guide network before any downstream training occurs. Crucially, alignment to the guide network cannot transfer learned features, never receives gradients, and does not use labels. We demonstrate that this cross-architecture transfer of "fate" benefits the target network in two key domains: (1) instilling the optimization benefits of depth into shallow networks by aligning them to deeper networks, and (2) transferring representational granularity by using fine-resolution networks to shape the geometry of networks with coarser representations. In both cases, the alignment results in improved downstream performance of the target network. Ultimately, this demonstrates that a model's architectural destiny can be decoupled from its physical structure and explicitly modified at initialization. 

---
# Solve Vision with Code: Coding Agents Are Stronger Visual Reasoners Than Video Models 

**Authors**: Dezhi Luo, Fengyuan Yu, Haotian Zhang, Zehong Zhao, Ran Ji, Siyuan An, Huayi Qin, Qingying Gao, Zhongang Cai, Lei Yang, Ziqi Huang, Ruisi Wang, Maijunxian Wang, Hokin Deng  

**Link**: [PDF](https://arxiv.org/pdf/2611.34813)  

**Abstract**: Video generation models are increasingly evaluated as visual reasoners: given a first frame and a natural-language prompt, can the model produce the video in which the specified task is carried out? The paradigm is considered advantageous because the reasoning is native to the visual medium, unfolding directly in the frames rather than through language or code. Whether this native route is in fact the more reliable path to visual reasoning has not been tested against a symbolic alternative under the same conditions. We therefore pose the same tasks, with identical inputs, output specification and evaluator, to both media: rather than denoising pixels, a coding agent writes a program that renders the answer video, which an unmodified rule-based evaluator scores. On VBVR-Pro-Bench (Xu et al., 2026), a 500-instance suite spanning five cognitive categories with symbolic ground truth, the agent receives the same first frame and prompt as the video model, with no access to ground truth, metadata or generation models, and one attempt per instance. Three closed-model agents (Codex with gpt-6-astra, Claude Code with Fable 5.1 and Gemini CLI with Gemini 3.1 Pro) score 0.923, 0.894 and 0.893, each more than 0.22 above the strongest video model. Among 23 runnable open-weight models, the best, GLM-5, reaches 0.572, above all video models except the RL-trained one. Together, we show that for tasks whose solutions can be computed from a first frame and a prompt, program synthesis is currently a substantially more reliable route to visual reasoning than video generation. We further distinguish depictive programs, which navigate the image to recover a transferable scheme, from propositional programs, which operate on hard-coded values, revealing how visual reasoning unfolds in agent-written programs and what it implies for its nature of reasoning in general. 

---
# HOG-WAM: Improving Visual Robustness through Future Gradient Prediction 

**Authors**: Boxuan Dong, Yupeng Qian, Ying Wu, Heshuai Li, Junhao Tan, Zhenxiong Tan, Weijia Wu, Weihao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2611.34811)  

**Abstract**: World-action models jointly predict future observations and actions, yet visual corruption can disrupt the resulting policies. We introduce HOG-WAM, which adds future Histograms of Oriented Gradients (HOG) as structural supervision derived directly from demonstration images. An auxiliary HOG diffusion transformer (DiT) module reads RGB features through asymmetric attention, allowing its objective to shape the visual representation used for control. The module is used only during training and removed at inference, preserving FastWAM's RGB-to-action architecture. We evaluate four controlled observation corruptions through a visual-corruption extension of RoboTwin, termed RoboTwin-Plus; HOG-WAM achieves the highest success among the compared policies under all four. Mixed-noise success increases from 78.86% to 82.86% relative to FastWAM, with comparable performance on clean scenes. On LIBERO Plus, overall success increases from 74.49% to 75.26%, with the largest gain under sensor noise, from 74.14% to 78.89%. Across four Franka tasks, average success increases from 40% to 59% under clean observations and from 17% to 24% under noisy observations. A representation ablation favors combining RGB observations with auxiliary HOG prediction over replacing RGB with HOG. These findings support future-gradient prediction as complementary supervision for visually robust control, with larger absolute gains on physical tasks than in simulation. Code is available at this https URL. 

---
# PersonaFactory: Stylistic User Prompt Augmentation with Synthetic Persona Bank 

**Authors**: Bich Ngoc Doan, Hanna Yukhymenko, Imanol Schlag  

**Link**: [PDF](https://arxiv.org/pdf/2611.34785)  

**Abstract**: Large language models increasingly replace real people in the AI development pipeline, as they are used to generate training dialogues at scale and to drive alignment and evaluation. However, out-of-the-box models produce monolingual and stylistically flat texts as synthetic user prompts, failing to replicate the real user-AI communication mechanics. We present PersonaFactory, an end-to-end pipeline that generates detailed user personas and uses them to augment existing user prompts from their point of view, producing a bank of over 740000 structured synthetic personas detailing their background and communication behaviors, and a corpus of roughly 1.24 million persona-styled synthetic conversations spanning 85 languages. We evaluate the effect of personal features on style adoption and intent preservation, using an LLM-judge and heuristic metrics to assess augmentation fidelity together with the downstream impact on assistant persona control through training data. Results show that our structured, attribute-based style representation enables per-axis control and better intent preservation at comparable style fidelity than free-text persona descriptions used in existing methods. A human expert evaluation found the augmented conversations plausible, with favorable ratings for intent preservation, style naturalness, and multilingual authenticity. Finally, PersonaFactory conversations achieve comparable performance to original training data on capability benchmarks, are preferred on real user prompts at equal reply length, and adapt their reply style more closely to users without explicit steering. 

---
# When Every Rollout Fails: Splicing Learner-Conditioned Options into Long-Horizon Agent RL 

**Authors**: Haowen Hou, Yicheng Liu, Zhen Huang, Jiaqi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.34777)  

**Abstract**: Group-based reinforcement learning with verifiable rewards often yields all-fail groups on long-horizon agent tasks: every sampled trajectory receives zero reward, leaving group-relative methods without a reward-based policy gradient. A common remedy supplies complete solutions from a stronger model, but full trajectories are expensive and expose the learner to long off-policy histories. We introduce SPLICE, which constructs local training targets from failed current-policy trajectories online. For each all-fail group, a proposal model selects a splice point in a current-policy trajectory and generates several bounded options from that point. We estimate each option's value using repeated current-policy continuations, convert positive estimates into a return-weighted local target, and fit the current policy on the options. We evaluate SPLICE across two policy families on two long-horizon agent tasks, Software Engineering with SWE-bench Verified and Deep Research with BrowseComp-Plus. SPLICE outperforms GRPO and full-trajectory guidance baselines in all four settings, improving over GRPO by 2.8--9.8 percentage points. On initially failed SWE tasks, it raises GRPO's graduation rate by 10.8 percentage points. Further experiments show that SPLICE improves performance while reducing costly proposal-model computation. 

---
# Make the Adversary Spend More: Stateful Deception as a Cost-Exchange Primitive for AI-Era Cyber Defense 

**Authors**: Alessandro Cantelli-Forti, Isabella Marasco, Hosam Alamleh, Giada Boschi, Michele Colajanni  

**Link**: [PDF](https://arxiv.org/pdf/2611.34770)  

**Abstract**: Large language models (LLMs) are transforming the economics of intrusion by reducing the attacker's cost to scan and adapt, while tempting defenders into unbounded compute. We argue that cyber deception should be judged by the cost exchange it imposes, not by realism or detection, and that the primitive making that exchange favorable is adversary-specific persistent state. Rather than a honeypot that resets or merges sessions, each recognized adversary re-enters the world it believes it has compromised, kept coherent with its own past actions and isolated from every other adversary, while the language model acts only as a constrained renderer of explicit state, never as its store. We formalize the trade-off as a Deception Exchange Rate (DER), relating mission-weighted adversary effort burned to defender resources spent, and instantiate it in an SSH honeypot pairing Cowrie with a local LLM. In this paper, we focus on the defender-side evaluation of that exchange, while specifying, but not yet measuring, per-adversary state and the adversary side of the DER. Across five open-weight models, deterministic-layer and prompt optimizations cut per-session defender cost by around 63% against a naive all-LLM decoy. 

---
# InterConRank: Joint Node Centrality in Interconnected Heterogeneous Networks 

**Authors**: Hao Ren, Jiaojiao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2611.34763)  

**Abstract**: Two nodes with identical connections within their own layer can differ in importance if their neighbors in another layer occupy different positions in the network. We introduce InterConRank, a centrality algorithm that computes a separate node ranking for each layer of an interconnected network. Directed dependencies specify which layers supply scores to each receiving layer. InterConRank normalizes each type of link separately and updates all layers together, so a score can cross into a layer in one iteration and propagate along its internal links in the next. We prove convergence of this joint update and evaluate the resulting rankings on four scholarly networks. On all four networks, InterConRank gives the highest rank correlations with both h-index and total citations for nodes in the collaboration layer. 

---
# One Model, Many Modes: Evolving System Prompt Ensembles for Creative Generation 

**Authors**: Daiwei Chen, Anda He, Binwei Yao, Junjie Hu, Ramya Korlakai Vinayak  

**Link**: [PDF](https://arxiv.org/pdf/2611.34760)  

**Abstract**: Preference-aligned LLMs can exhibit mode collapse, concentrating their output distributions around a narrow set of high-reward responses. This limits creative generation, which requires quality, novelty, and diversity. Existing test-time methods often emphasize diversity, giving less attention to whether individual responses are both high-quality and novel. This distinction matters in practical use, where users may seek a single compelling, original response. We formalize creative generation as a multi-objective distributional optimization problem and introduce SEPO, a training-free framework for System Prompt Ensemble Optimization. The framework combines prompt evolution with greedy max-min selection to identify behaviorally distinct system prompts under a quality constraint, then uniformly samples one prompt per response at inference time. Across four creative generation tasks with increasing quality requirements and three open-weight models, SEPO ranks first in 8 of 12 settings on the harmonic mean of quality, diversity, and novelty, while no baseline leads in more than two. On average, it raises LLM-judged diversity from 1.92 to 3.43 and novelty from 1.00 to 3.34 over base sampling (1-5 scale), with quality decreasing from 3.72 to 3.30. We further show that expanding the optimized ensemble increases separation among prompt-conditioned response distributions. These results support system prompt ensemble optimization as a promising approach to improving LLM creativity at inference time. 

---
# Reading Fine-Tuning Diffs Through Lenses: Model Diffing for Backdoor Detection 

**Authors**: Maximilian Petrowitsch, Ben Connors, Chris Kapulkin  

**Link**: [PDF](https://arxiv.org/pdf/2611.34758)  

**Abstract**: We introduce a common scoring system based on token readouts to evaluate weight, activation and output-based model diffing methods for backdoor detection in fine-tuning processes. We also improve existing weight and activation based methods by incorporating the $J$-lens and its refinement, the $J{+}{+}$-lens. We test our setup in experiments where we compare a backdoored model with the base model from which it was fine-tuned. Our results show that incorporating the $J{+}{+}$-lens improves the weight-based method in all our settings, while the $J$-lens improves its ranks within single layers. The strongest single method is $\Delta W J{+}{+}$, the weight difference decoded through the $J{+}{+}$-lens of the base model, which ranks a token naming the backdoor among its first ten tokens in 11 of our 12 settings. Moreover, we show that output and activation based methods such as ADL and Diff Mining can easily break if the backdoored model is penalised during fine-tuning for giving the backdoor's words probability on clean inputs, whereas the weight-based methods are not affected. Conversely, permuting the hidden units or training only the layers that read the residual stream hides the backdoor from the weight-based methods but not from ADL and Diff Mining. No single method resists both attacks. 

---
# Collect for Coverage, Replay with Evidence: Adaptive and Efficient On-Policy Distillation for Long-Horizon Agents 

**Authors**: Haowen Hou, Gen Li, Ruilin Li, Yicheng Liu, Zhen Huang, Shuai Dong, Congcong Wang, Kaiwen Tuo, Hongrui Wu, Zelin Peng, Kanghui Tian, Ji Zhao, Dianyi Wang, Jiaqi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.34756)  

**Abstract**: Fully online on-policy distillation (OPD) is costly for long-horizon agents because each update depends on fresh environment rollouts. Prefix replay offers an efficient alternative by sampling current-student responses from stored histories, reducing idle time for long-tail rollouts and allowing each collected trajectory to support multiple updates. However, our analysis of the complete prefix replay pipeline identifies two limitations in long-horizon settings. In trajectory collection, teacher-only trajectories miss student-induced histories, while broader history coverage can reduce teacher reliability. In trajectory reuse, fixed position-based schedules cannot identify critical decisions that may occur at any trajectory depth. We introduce CARE-OPD (Collect for coverAge, Replay with Evidence) to address both limitations. Student-seeded teacher completions and response-level confidence gating expand history coverage while filtering unreliable updates. Hindsight replay allocation then uses downstream execution evidence to select consequential turns. Across long-horizon tasks Software Engineering and Deep Research, CARE-OPD outperforms online OPD and prior prefix replay baselines. On SWE-bench Verified, it raises task success to 59.5% while reducing end-to-end training time by 45% and environment actions by 44% relative to asynchronous online OPD. 

---
# How Well Do Vision-Language Scene Graphs Serve as Memory for Embodied Question Answering? 

**Authors**: Divyansh Pandey, Shubh Garg, Vedant Khandelwal, Niyati Rawal, Amit Sheth  

**Link**: [PDF](https://arxiv.org/pdf/2611.34745)  

**Abstract**: Embodied QA needs memory, but answer accuracy mixes retained evidence, reading errors and question priors. For a simulated multi-room scene-graph pipeline we define answerability: the fraction of questions a fixed rule-based reader answers correctly from the graph alone. The baseline recovers 24.1% of visible object types and its room labels flicker. Four targeted repairs double recall, raise answerability from 27.2% to 37.4%, and improve retrained QA from 71.4% to 75.9%, yet better object F1 can lower answerability. A question-type prior beats this reader on every predicted graph. In one real-video test, graph memory lowers accuracy. These exploratory findings favour scoring retained evidence alongside accuracy. 

---
# When the Proxy Is Wrong: Mismatch-Corrected Bandits for Adaptive Data Sampling 

**Authors**: Amine Saadi, Elissa Mhanna, Raphaël Féraud, Gwénolé Lecorvé  

**Link**: [PDF](https://arxiv.org/pdf/2611.34707)  

**Abstract**: Data allocation methods often use a small proxy model to select training data for a larger target model. However, this proxy can rank data differently from the target, leading to suboptimal training performance. To address this problem, we model the mismatch between the proxy and target models and incorporate it into the allocation decision. To do so, the problem is cast as a multi-fidelity bandit, where each arm is a data intervention. The proxy reward is observed every round, whereas the target reward is revealed only on rare audits of the same update. These paired observations identify arm-wise mismatches without assuming a shared offset or preservation of rankings. Mismatch-corrected bandits correct the proxy with paired residuals, audit only uncertain decisions, and pace the audit budget over training. From a theoretical viewpoint, we prove a regret decomposition and lower bounds under stationarity. These bounds show that proxy-only allocation can incur linear regret and that a minimum number of paired audits is necessary. Under abrupt changes, using the audit budget early can also incur linear regret. A sliding-window policy that spreads audits over training achieves regret of order $\sqrt{\Gamma_T T\log T}$ where $\Gamma_T$ denotes the number of change points. Experimentally, we test the approach in a closed-loop proof of concept on CivilComments-WILDS, where BERT-Small guides the training of BERT-Large. Mismatch correction improves worst-group accuracy over a target-only policy with the same target-model budget. Spreading audits over training improves worst-group accuracy by 6.3 to 10.5 percentage points at the same audit count. 

---
# Maco-RL: Scaling Agent Capabilities with Mixed-Task Reinforcement Learning 

**Authors**: Jiacheng Zhang, Qixing Zhou, Haiyang Wang, Qipeng Gu, Shuzhe Wu, Feiyang Pan, Lue Fan, Dandan Tu, Zhaoxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.34703)  

**Abstract**: Mixed-task reinforcement learning (MixRL) directly trains a unified policy across domains to continually expand agent capabilities without training separate specialists or requiring an extra distillation stage. Despite its appeal, scaling MixRL poses significant challenges in achieving stable and efficient mixed-task learning, effectively allocating rollout compute, and supporting continual domain expansion. To address these challenges, we propose Maco-RL, an approach that effectively scales MixRL. Specifically, Maco-RL employs a quota-based scheduler for asynchronous rollouts across multiple task domains, thereby stabilizing training by balancing gradient contributions and mitigating off-policy effects. Along an orthogonal axis, Maco-RL further introduces Tri-Buffer Sampling (TBS) to allocate compute among samples with varying levels of mastery, promoting sustained frontier exploration while preserving acquired capabilities. Experiments on five benchmarks show Maco-RL raises Qwen3.5-27B's overall score from 43.6 to 54.2 and then to 60.1 with continual expansion, reaching the frontier held by Qwen3.8-27B, which is further pushed outward by +5.7. Maco-RL also outperforms single-task RL with less compute for policy updates. These results support MixRL as a practical and compute-efficient post-training approach with substantial potential to strengthen general agent capabilities. 

---
# Evacuation Traffic Prediction for Dynamic Transportation Networks over Multiple Hurricanes: A Generalized Graph Learning Approach 

**Authors**: Md Nafees Fuad Rafi, Samiul Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2611.34687)  

**Abstract**: High traffic demand during hurricane evacuations causes severe congestion on major highways, leading to delays and safety risks for evacuees. Predicting evacuation traffic in real-time can significantly enhance evacuation management efforts. However, evacuation traffic is highly dynamic and uncertain due to evolving hurricane characteristics and diverse population response. Each hurricane has unique characteristics and evacuation contexts, resulting in different spatial-temporal traffic patterns. Moreover, the structure of detector-based transportation networks varies across multiple hurricanes. Hence, a model trained on the data from one hurricane may not generalize well for others. This study proposes a generalized dynamic graph learning framework using traffic detectors and evacuation data from 12 historical hurricanes. Leveraging the dynamic nature of transportation network, we develop a Graph Convolutional Long Short-Term Memory (GCN-LSTM) model that predicts evacuation traffic up to 6 hours in advance. For an unseen hurricane (Milton, 2024), the model shows strong generalizability, achieving an RMSE of 280.6 and R2 of 0.96 for 1-hour prediction, and an RMSE of 459.8 and R2 of 0.89 for overall 6-hours prediction. We also introduce a modular training approach, where we evaluate performance improvements by sequentially incorporating data from multiple hurricanes. The proposed model outperforms state-of-the-art baseline models for overall 6-hour prediction horizons. Ablation experiments show an 8.4% improvement in RMSE for the proposed dynamic GCN-LSTM model compared to its static counterpart. Traffic management officials can use the developed model to understand future traffic conditions during hurricane evacuation in advance and take proactive measures to reduce delays during evacuation. 

---
# Find the Clue, Not the Scene: Rethinking Reasoning for Entity Recognition in Vision-Language Models 

**Authors**: Kyungmin Min, Soyeon Kim, Junseok Kim, Kyomin Jung  

**Link**: [PDF](https://arxiv.org/pdf/2611.34681)  

**Abstract**: Answering a knowledge-based visual question requires first recognizing the entity in the image and then recalling knowledge about it. Thinking vision-language models (VLMs) reason for thousands of tokens before answering, yet it remains underexplored how well such thinking VLMs perform each of these two steps. We therefore analyze thinking VLMs on this task and make two key observations. (1) Recognition is the bottleneck, since 76.3% of errors across four models on ViQuAE stem from misrecognizing the entity, and 84.9% of these failures are resolved once the entity name is provided in the question. (2) Recognition is driven by entity-recall clues rather than by general descriptions of the image. An entity-recall clue is a distinctive visual detail that helps the model recall the entity, such as keys, a traditional attribute of Saint Peter. Based on these findings, we propose ClueRL, a reinforcement learning approach that extends Group Relative Policy Optimization (GRPO) with an advantage localized to the entity-recall clue in the thinking trace. Rather than injecting new entity knowledge, ClueRL changes how VLMs think: it leads them to attend more closely to visual details and spot these clues, helping them use the knowledge they already have. Consequently, it outperforms both GRPO and a perception-reward baseline on three entity recognition benchmarks. Moreover, this clue-seeking behavior generalizes beyond entity recognition, the only task used for training. Since recognition is the bottleneck, the gains carry over to knowledge-based visual question answering. They even extend to general multimodal reasoning, raising Qwen3.5-4B by 4.4 and 4.0 points on the ScienceQA and MMMU subsets of MMStar, respectively. 

---
# RoboEvo$^2$: Recursive Robot Self-Improvement via Harness-Policy Co-Evolution 

**Authors**: Chenzhi Liu, Jiehong Lin, Wei Huang, Xuzhao Huang, Haoyuan Li, Yang Zhang, Junchao Huang, Chenyeyang Cai, Bo Wang, Kazuhiro Kosuge, Zhongrui Wang, Xiaojuan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2611.34675)  

**Abstract**: Robots should adapt to unfamiliar environments and acquire new skills through interaction, then execute those skills efficiently with practice. Vision-language-action policies provide efficient control, but often need additional human demonstrations to handle unfamiliar settings; reasoning agents can discover solutions using broad prior knowledge and robot-control APIs, but repeated deliberation and repair are costly. Inspired by human learning through reasoning and practice, we propose RoboEvo$^2$, a framework for recursive harness-policy co-evolution. In Policy-to-Harness (P2H) evolution, policy failures guide a coding agent to search complementary Code, Code+ICL, and Code+Policy routes, consolidating verified solutions into reusable skills, routing strategies, and experience memory. In Harness-to-Policy (H2P) evolution, verified trajectories provide supervised training, while Counterfactual Preference Distillation (CPD) teaches the policy to favor successful corrections over failures from the same starting state. The improved policy provides stronger motor skills for subsequent search, while its remaining failures redirect harness evolution, closing the reciprocal loop. Through recursive co-evolution without additional human demonstrations, RoboEvo$^2$ improves standalone success from 11.0% to 84.1% and hybrid success from 30.4% to 90.5% on LIBERO-PRO, with similar gains and reduced agentic costs on RoboTwin C2R. Further experiments demonstrate autonomous task acquisition, cross-embodiment harness transfer, and 95.0% hybrid success on the Unitree G1-D. Our models and code will be publicly released. 

---
# DemoFoundry: Toward Scalable Photorealistic Robot Data with Coding Agents and Generative Rendering 

**Authors**: Ruixiang Wang, Xinyue Yao, Xuanning Zhou, Yueci Deng, Yongyi Su, Kui Jia  

**Link**: [PDF](https://arxiv.org/pdf/2611.34671)  

**Abstract**: Scaling robot learning requires demonstrations that combine realistic visual observations with reliable action supervision. Simulation-based synthesis faces a substantial visual gap relative to real environments, while video-only generation struggles to maintain physically consistent interactions and provide aligned action labels. We present DemoFoundry, an automated framework that couples coding agents with generative rendering to synthesize photorealistic and physically grounded robot demonstrations. Coding agents both construct aligned real--proxy video pairs from existing robot episodes to train a video renderer and write executable interaction programs to generate new demonstrations. Given a scene reference and task instruction, an agent reconstructs task-relevant geometry and writes a control program without requiring target-task human demonstrations. Program execution in a physics simulator produces proxy videos with aligned robot states and actions. The video renderer translates these proxy videos into scene-consistent, photorealistic observations, reducing the need for detailed appearance modeling in each scene. Automated task-execution checks select successful trajectories for rendering and policy learning. Experiments show improved visual quality and action--video consistency. Policies trained on the generated demonstrations outperform raw-simulation training and approach the performance of training on an equal number of real demonstrations. 

---
# Multi-Party Authorization of Maintenance Prescriptions on a Permissioned Blockchain 

**Authors**: Siyuan Chen, Ebru Turanoglu Bekar, Anders Skoogh  

**Link**: [PDF](https://arxiv.org/pdf/2611.34662)  

**Abstract**: In industrial prescriptive maintenance, an equipment supplier's model may recommend an intervention that a plant schedules, a contractor executes, and a quality or safety function approves. A shared ledger records who submitted the prescription, yet cannot check whether the agreed model and decision rule produced it. We require permissioned-blockchain endorsers to verify a succinct proof of that calculation before releasing a work order. On a four-organization Hyperledger Fabric testbed using a public engine benchmark, this stopped all nine contested or mis-operated prescriptions, whereas hash checks accepted four that either raised simulated maintenance cost or reintroduced late failures. 

---
# Do We Still Need Reinforcement Learning When We Can Read the World Model? 

**Authors**: Mingchao Zhang, Miduo Cui, Kangze He, Yuanjun Li, Dapeng Li, Ce Zhang, Chuanchen Luo, Xue Yan, Mingzhe Xing, Zhumin Chen, Hangyu Mao, Bin Zhang, Zhiwei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2611.34656)  

**Abstract**: Reinforcement learning has achieved remarkable success in sequential decision-making, but learning effective policies often requires substantial interaction experience. Model-based reinforcement learning alleviates this dependence by leveraging world models, which are typically used to support planning or policy optimization. In this paper, we investigate an alternative paradigm in which autonomous LLM-based coding agents directly synthesize control policies by reasoning over readable world models. By examining environment source code, these agents translate environmental knowledge into executable decision rules and iteratively refine their policies using online feedback or offline data, without task-specific RL training. Experiments across diverse decision-making benchmarks show that synthesized policies match or outperform conventional RL methods while requiring substantially less experience. Further analyses reveal interpretable decision rules and how they evolve through feedback-driven refinement. Comparisons with policies generated without source code access and experiments under challenging conditions further demonstrate the benefits of readable world models. These findings suggest that direct policy synthesis can reduce reliance on trial and error, challenging the necessity of task-specific RL training. 

---
# Motion Beyond Pixels: Learning Future Visual Representations for Object-Centric Dynamics 

**Authors**: Chengzhi Zhao, Juelin Li, Zhennan Wang, Sheng Wang, Zhuoman Liu, Yuhao Lu, Jiongxu Chen, Haitang Feng, Yanjie Li, Lin Shao  

**Link**: [PDF](https://arxiv.org/pdf/2611.34644)  

**Abstract**: Many embodied world models predict future states in latent spaces optimized for RGB reconstruction, which do not explicitly prioritize object-level semantics and geometry relevant to motion generation. We present MotionRep, a visual representation world model that jointly generates future DINO representations and 3D object flow, coupling semantic and geometric information with object-centric dynamics. To make high-dimensional DINO features amenable to generative modeling, we introduce DINO-VAE, which combines temporal and channel compression with latent regularization to produce compact latents while preserving semantic and geometric information. Conditioned on an initial observation, object query points, and a language instruction, a unified diffusion transformer predicts future visual representations and 3D object flow through joint attention. MotionRep employs asymmetric modality-specific noise schedules to keep the visual stream ahead of 3D object flow during joint denoising, allowing progressively refined future visual representations to guide flow generation. Experiments on simulated and real-world manipulation benchmarks demonstrate that MotionRep outperforms pixel-based baselines in 3D object flow accuracy and generalization to unseen scenes, while requiring only three denoising steps for inference. These results support predictive visual representations as an effective future-state space for object-centric dynamics modeling. 

---
# Carrying Feature Ablations Across Sparse Autoencoder Seeds 

**Authors**: Karl Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.34637)  

**Abstract**: A vision encoder turns an image into one vector per patch, refined layer by layer. A sparse autoencoder (SAE) writes each vector of one layer as a sum of learned directions, most with a coefficient of zero. Researchers call each direction a feature and edit the encoder by removing one feature. An SAE trained from another random seed learns different features. Earlier work matches the two SAEs' features one to one by how closely their directions agree. Matched features can point alike yet respond to different image parts, so removing them may change the encoder differently. We remove a feature and, separately, its match, then compare the changes in the encoder's output on the same held out images. When the second change nearly reproduces the first, we say the removal carries across seeds. A setting fixes a DINOv2 or SigLIP layer and its SAE feature count. At our tolerance, the removal carries for 12% to 70% of matched pairs across three settings. The share is 0% in two controls: an SAE whose directions stay random and, post hoc, matches shuffled among trained features. Post hoc, predicting which removals carry is more accurate with layer contributions than with directions. Some incorrect predictions remain. One persists on independent images. So pairs can be screened at the layer. Measuring the output detects errors this screen misses. 

---
# LatentRL-QAS: Transfer via Latent Reinforcement Learning for Quantum Architecture Search 

**Authors**: Maria Ilektra Plakitsi, Sebastian Feld, Akash Kundu  

**Link**: [PDF](https://arxiv.org/pdf/2611.34634)  

**Abstract**: Reinforcement learning (RL) emerges as a prominent approach to quantum architecture search (QAS), the automated design of quantum circuits. However, transferring what an agent has learned from one number of qubits to another is still weakly explored: the state, the action space and the neural network of a state-of-the-art RL-QAS agent all grow with the number of qubits, which makes transfer learning across qubit counts prohibitive. We make this transfer possible by introducing LatentRL-QAS, a latent space transfer and refresh framework for RL-based QAS. At its core, a circuit variational autoencoder encodes a circuit as one token per gate and maps circuits on any number of qubits into the same latent space. An RL agent navigates this space instead of adding gates, so its state, actions and network have the same size at every qubit count. The latent space is periodically refreshed with the agent's new discoveries and, for a larger system, grown from the solutions of the smaller one, so what is transferred is the representation of circuits, not network weights or past experience. To show transfer, we consider absolutely maximally entangled (AME) states, a resource for quantum error correction. LatentRL-QAS discovers the five- and six-qubit AME states without being shown them and prepares them in 89-99% of evaluation episodes, whereas the baseline RL-QAS fails to find them with the same budget, even with a network 10 to 17 times larger. LatentRL-QAS remains robust under depolarizing noise with single-qubit gate errors amplified 10x and two-qubit gate errors 5x compared with IBM quantum hardware. By decoupling the RL agent from the number of qubits, LatentRL-QAS lets quantum architecture search reuse circuit representations learned on smaller problems, providing a foundation for the transferable design of circuits across quantum optimization problems. 

---
# DecisionReplay: A Corpus and Offline Simulator for Ensembling Decision Models 

**Authors**: Dean Ariel, Seffi Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2611.34627)  

**Abstract**: Comparing ensembles, routers and cascades requires the output of every model on every item, and its cost. Stored collections built for routing keep no per-option probabilities. We provide a dataset on which ensemble methods, routers and cascades are simulated quickly, without new model calls. DecisionReplay holds the probability vectors of 17 commercial decision models, endpoints that return a probability for every answer option, on 35 benchmarks, with the token counts, billed cost and measured latency of every call, all released. A deterministic simulator replays any policy that selects, combines or cascades them. One well-chosen model per benchmark is hard to beat: pooling all models loses 2.7 accuracy points to it, and stacking gains 1.0 points at 3.4 times its cost. The models are wrong together: two that are both wrong choose the same wrong option in 72% of cases. The best router matches that model at 64% of its cost, and certified early exit cuts the cost of a stacked ensemble by 43% without changing an answer. DecisionReplay makes it cheap and fast to test whether a new method changes these results. 

---
# From Models to Services: Systematizing Security Assurance for Model-as-a-Service 

**Authors**: Jianming Zhou, Yang Luo, Guangpu Chen, Zhaoyu Chen, Yifeng Gao, Qingni Shen, Zhonghai Wu  

**Link**: [PDF](https://arxiv.org/pdf/2611.34623)  

**Abstract**: Large language models are mainly delivered as cloud services, in which the assets of different principals are coupled within a single service pipeline, while the related security research is scattered across several fields with differing threat models. This paper systematizes this research from a service perspective. We decompose Model-as-a-Service (MaaS) into a five-layer stack and five trust boundaries, define six adversary classes and six security goals, use them to code 120 studies, and evaluate 50 defenses under a unified set of criteria. The results show that 44.2% of the studies concentrate on behavioral integrity, while service integrity, availability and cost, and accountability together account for only 13.3%. Defenses are evaluated on production services at less than half the rate of attacks, and none of the cryptographic or hardware mechanisms that protect against the provider can be adopted unilaterally by customers. Finally, we propose a research agenda by comparison with current standards and regulations. 

---
# VisualMontage: Manipulating the Beliefs of Vision-Language Agents with Authentic Image Sequences 

**Authors**: Jinwei Hu, Lai Wei, Zhuoyun Li, Youcheng Sun, Yi Dong, Xiaowei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.34603)  

**Abstract**: As agents based on vision-language models (VLMs) increasingly interpret social-media content, errors in their analysis may lend credibility to rumors and amplify their spread. In this paper, we study the problem of visual cognitive collusion, in which coordinated posts combine authentic images and locally faithful captions to induce VLM agents to infer and accept attacker-chosen target claims through their own reasoning. To enable systematic evaluation of this cognitive threat, we introduce a new benchmark comprising 544 target cases and 33.2K candidate images across eight application domains. We further propose VisualMontage, a multi-agent framework for simulating such attacks through narrative planning, image selection and ordering, and caption assignment under iterative review. We evaluate this threat through extensive experiments across 12 vision-language models, examining modality effects, production-team components, and downstream defenses. Using identical attack feeds across models without victim-specific optimization, we demonstrate cross-model transferability, achieving mean attack success rates of 61.93% for proprietary models and 68.71% for open-weight models. Moreover, 92.65% of generated feeds pass independent model-based validity auditing. Even after downstream review, residual attack success rates range from 35.3% to 76.7% across the evaluated configurations. Our additional controlled comparisons further reveal that captions can improve model-rated posting naturalness while reducing attack success. Together, these findings reveal a novel cognitive threat in social-media interpretation and motivate safeguards that assess how authentic posts jointly shape agents' reasoning and conclusions. 

---
# Tune for Retention, Not Just Gain: SFT Forgets Far Less than You Think 

**Authors**: Nirmit Joshi, Nathan Srebro, Eran Malach  

**Link**: [PDF](https://arxiv.org/pdf/2611.34579)  

**Abstract**: Catastrophic forgetting, the drastic loss of previously acquired capabilities when learning new ones, is a central challenge in language model fine-tuning. Supervised fine-tuning (SFT) is widely believed to be prone to such forgetting. Recent works have attributed this to the off-policy nature of SFT, arguing that on-policy learning is key to continual learning. In this paper, we challenge this premise. We show that while maximizing new-task performance can induce catastrophic forgetting, alternative hyperparameter choices (lower learning rate and batch size) can yield similar gains with little to no forgetting. With hyperparameters selected jointly for gain and retention, vanilla SFT retains prior capabilities far better than previously suggested and can be competitive with Self-Distillation Fine-Tuning (SDFT), an on-policy approach designed to mitigate forgetting. While SDFT can be more robust to hyperparameter choices, it also requires a retention-aware criterion to identify hyperparameter configurations that avoid catastrophic forgetting. Our results challenge the view that on-policy training is what enables learning with retention and raise broader questions about what drives catastrophic forgetting during fine-tuning. 

---
# Sage-Med: Scaling Medical Multimodal Intelligence 

**Authors**: Sage-Med Team, Ming Hu, Cheng Tang, Wei Li, Rongbin Li, Feilong Tang, Junru Hu, Hongze Zhu, Qiujuan Yan, Huanze Tang, Shaocong Wei, Siyuan Yan, Min Cen, Zhongxing Xu, Zhonghua Wang, Lie Ju, Zhen Yu, Wenjie Li, Shiyan Su, Jing Hao, Jingyong Liang, Diping Song, Fang Yan, Chenglong Ma, Junzhi Ning, Haochen Xue, Zanting Ye, Chenrun Wang, Jinjie Wei, Yi Xin, Jianghan Shen, Siqi Luo, Jiyao Liu, Huidong Xie, Jun Zhao, Rulin Zhou, Longquan Jiang, Ziyan Huang, Ying Chen, Hanxiao Zhang, Junyang Wu, Mouxiao Bian, Jiashi Lin, Chenyu Wang, Qi Qin, Jie Huang, Yiling Zhou, Jiawei Chen, Kefan Wang, Jingjing Hua, Yaoyu Liu, Huihui Xu, Tianbin Li, Jin Ye, Ziyang Chen, Shujian Gao, Zilin Lu, Jiangtao Yan, Changkai Ji, Wenhao Tang, Xinyi Zeng, Yuxing Chen, Hongwei Yang, Rongzhao Zhang, Jinru Ding, Yingcheng Wu, Zhenfei Yin, Jinzhe Li, Lu Lu, Yun Zhong, Zhishan Lin, Bingli Wang, Xiaohan Ren, Jianmin Qian, Penghao Zhao, Zhongbo Tian, Zhenbang Zhou, Peiling Dai, Mengjing Zhao, Jingyu Zheng, Pinan Liu, Shiwei Li, Victoria Mar, Cristina Vico-Alonso, Qiang Wu, Xianming Liu, Lingtong Huang, Dili Song, Minghui Zhang, Wenlong Zhang, Qipeng Guo, Yuqiang Li, Nanqing Dong, Yun Gu, Qi Duan, Hongliang Ren, Hongming Shan, Yin Xie, Xiang An, Benyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.34575)  

**Abstract**: Establishing effective scaling laws for medical multimodal large language models (MLLMs) remains a persistent and fundamental challenge. Our systematic evaluation shows that existing medical MLLMs tend to underperform frontier general-purpose MLLMs across a range of medical tasks, while scaling model size primarily improves text capabilities rather than multimodal perception, and enabling reasoning modes fails to close this visual gap. To investigate how medical multimodal intelligence scales in practice, we introduce Sage-Med-Preview, a 35B-parameter medical multimodal model with 3B activated parameters, built by combining stage-specific curation over 322.7 million candidate records with a progressive training curriculum spanning medical pretraining, staged supervised fine-tuning, and adaptive reinforcement learning. Across this curriculum, progressive supervised refinement and adaptive reinforcement learning yield consistent stage-wise gains, substantially advancing both specialized visual understanding and medical language capabilities. Evaluated against 77 competing models across 238 clinical tasks spanning 61 public and 13 private multi-center benchmarks in more than 40 modalities, Sage-Med-Preview achieves an average score of 58.26% on the 31 public multimodal benchmarks. It outperforms the state-of-the-art open-source medical model Hulu-Med-235B-A22B at 50.99% and the general scientific model Intern-S2-Preview at 52.24%, and approaches the advanced proprietary model GPT-6.2-Sol across multiple tasks such as comprehensive multimodal understanding, report generation, and visual grounding. Although substantial challenges remain in complex medical reasoning and multi-step clinical workflows, this report provides empirical evidence, actionable data and training recipes, and a rigorous benchmark toward scalable clinical multimodal intelligence. 

---
# An Information-Theoretic Analysis of Information Hiding in Semantic Communication 

**Authors**: Yanhao Huo, Shijun Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2611.34550)  

**Abstract**: Semantic communication shifts the transmission goal from the accurate recovery of syntactic symbols to the reliable conveyance of semantic content, yet a systematic information-theoretic characterization of information hiding at the semantic layer is still missing. Building on the semantic information theory of Niu-Zhang and the classical information hiding theory of Moulin-O'Sullivan, this paper establishes an information-theoretic framework for information hiding under semantic communication. The classical hiding model is extended to a two-layer structure (semantic host, syntactic host, syntactic composite signal, syntactic received, semantic received) bridged by synonymous mappings f_s, f_x and the inverse mapping f_y^{-1}, subject to four distortion constraints: syntactic transparency, semantic fidelity, syntactic perturbation, and semantic destruction. The capacity is cast as a Stackelberg game with the hider moving first and the payoff J(Q,A) = I(U~; Y~ | K) - I(U~; S~ | K), yielding the single-letter maximin C^s = max_Q min_A J(Q,A), whose achievability and converse follow from random coding and joint typicality decoding. Relative to the classical capacity, comparison payoffs built from the up/down semantic mutual information give the endpoint sandwich C^s_end(0,1) <= C^s <= C^s_end(1,0) with the classical value C in the same interval, while closing semantic tolerance yields C^s_no-tol <= C_cl under a fair comparison. These results clarify that semantic hiding is not unconditionally superior to classical hiding. Binary, ternary, and Gaussian examples confirm the degeneration under one-to-one mappings and the capacity loss under coarsening. 

---
# Recalling the Past to Answer the Present: Context-Aware Multi-Evidence Reasoning for Streaming Video Question Answering 

**Authors**: Cui Yakun, Tiantian Geng, Huixiang Kong, Xingqun Qi, Sirui Han, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2611.34545)  

**Abstract**: Vision-language models (VLMs) have achieved remarkable progress in video understanding, yet existing video question answering (VQA) benchmarks mainly evaluate models in offline settings. Real-world streaming scenarios instead require models to answer context-dependent queries by retrieving and reasoning over evidence distributed across the video history. We introduce StreamRecall, a benchmark for context-aware Multi evidence reasoning in streaming video QA. StreamRecall requires models to retrieve, associate, and reason over temporally scattered evidence to answer queries at the appropriate moment. It contains 4,983 questions over 462 videos, with context-aligned query times and fine-grained temporal evidence annotations. We further propose the Evidence Utilization Cascade to diagnose model reasoning through evidence localization, grounding, chain completion, and answer derivation. To enable efficient long-term streaming reasoning, we present EviStream, a training-free causal visual memory mechanism that preserves reasoning-critical evidence under a fixed KV budget. Experiments reveal substantial gaps between current VLMs and realistic streaming requirements, establishing StreamRecall as a foundation for developing evidence-aware streaming video intelligence. 

---
# HAS: Hierarchical Adaptive Frame Sampling for Video Understanding 

**Authors**: Yuyao Sun, Daiqi Shi, Shuang Li, Liao Zhang, Yang Cai, Deqing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.34544)  

**Abstract**: Due to context-length constraints, most MLLMs cannot process full-length videos and therefore rely on sampling a subset of frames as input. However, existing sampling methods, ranging from uniform sampling to relevance-based selection, are often driven by a single sampling principle and thus struggle to accommodate the heterogeneous evidential requirements of different queries: some demand a holistic understanding of how the video unfolds over time, while others hinge on fine-grained events within a short temporal window. To address this limitation, we propose \textbf{Hierarchical Adaptive Frame Sampling (HAS)}, a two-stage frame sampling framework. In the first stage, \emph{Backbone Frame Construction}, we apply a Determinantal Point Process (DPP) to sample frames that are both query-relevant and non-redundant. These selected frames capture key moments across the video and form the backbone of the sampling set, providing a foundation for subsequent enrichment. In the second stage, \emph{Adaptive Contextual Enrichment}, we analyze the temporal distribution of the backbone frames to infer the query's requirements and adaptively allocate the remaining frame budget between Local and Global Context. The Local Context enriches the backbone with fine-grained temporal dynamics and short-range causal relations, whereas the Global Context connects temporally isolated evidence and provides a holistic view of the entire video to support global understanding. Through such a hierarchical two-stage process, HAS effectively addresses diverse query requirements. Incorporated into three leading MLLMs, it demonstrates consistently superior performance across the Video-MME, LongVideoBench, and MLVU benchmarks. 

---
# EPPC-CARE: Staged Hierarchical Annotation of Communication Behaviors in Secure Patient-Provider Messages 

**Authors**: Samah Fodeh, Sreeraj Ramachandran, Ganesh Puthiaraju, Sarah Schellhorn  

**Link**: [PDF](https://arxiv.org/pdf/2611.34539)  

**Abstract**: Objectives: Patient portal messages provide invaluable information about communication dynamics between patients and healthcare providers, but extracting it manually is labor-intensive. We developed and evaluated EPPC-CARE, an automated framework for identifying communication behaviors together with the supporting text for each classification. Materials and Methods: From 867 messages, we obtained 5,516 reference standard annotations. Each annotation is a triplet of communication-related text (span), a broader communication category (Code), and a specific subcategory (Sub-code). We evaluated EPPC-CARE's identified triplets across six open-weight large language models (LLMs) against supervised fine-tuning and communication-based models such as EPPC-OASIS. We also compared EPPC-CARE with human expert annotations on 100 messages. Results: EPPC-CARE improved performance over EPPC-OASIS across all six models. EPPC-CARE's Triplet F1 improved by 2.90 percentage points on average, with a best score of 64.4%. On the human expert evaluation subset, EPPC-CARE and the human expert showed comparable agreement with the reference standard, with differences favoring EPPC-CARE. Discussion: These findings suggest that identifying triplet components (span, code, subcode) sequentially can improve performance across model families while preserving the supporting text behind each identification. EPPC-CARE links the communication code and subcode with the source text (span), supporting human verification of automated annotations in clinical informatics. EPPC-CARE may therefore make larger-scale studies of patient-provider communication more feasible and provide a foundation for human-in-the-loop annotation workflows. Conclusion: EPPC-CARE offers a more reliable and reviewable approach for automated characterization of communication behaviors in secure patient-provider messaging. 

---
# PURA: Video Token Pruning via Unified Spatiotemporal Redundancy Assessment 

**Authors**: Yuyao Sun, Hongli Yuan, Shuang Li, Tao Deng, Deqing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.34521)  

**Abstract**: Video large language models incur substantial inference costs from long visual token sequences. Existing video token compression methods mitigate these costs by exploiting spatial redundancy within frames and temporal redundancy across frames. However, many approaches impose predefined processing orders or fixed allocation ratios for spatial and temporal compression, limiting their ability to adapt to the varying balance of spatiotemporal redundancy across videos. To address this limitation, we propose PURA, a training-free method that prunes visual tokens through unified spatiotemporal redundancy assessment. For each token, PURA constructs spatial and temporal reference candidates and selects the reference with the lowest representation error across the two dimensions. The representation error is combined with token importance to determine each token's removal cost. To avoid the time overhead of removing the lowest-cost token one by one, we introduce Coordinated Pruning, which iteratively groups mutually compatible low-cost removals into batches for simultaneous execution and updates the pruning state after each batch until the target budget is reached. Extensive experiments on three video large language models and five video understanding benchmarks demonstrate that PURA achieves a favorable trade-off between task performance and inference efficiency. 

---
# Long-Horizon Planning with Latent World Models via Graph Anchors 

**Authors**: Syed Nazmus Sakib, Abdul Monaf Chowdhury, Peizhen Li, Nafiul Haque, Chandak Chakma, Md Mehedi Hasan, Zehong Cao, Shifat E. Arman  

**Link**: [PDF](https://arxiv.org/pdf/2611.34509)  

**Abstract**: Latent world models plan in a compact latent space by predicting the outcomes of actions. Their planners judge a plan by the latent distance between its
predicted end state and the goal image. We show that this distance fails in two distinct ways. First, it loses directional signal a few steps from the
goal, causing distant goals to fail. Second, it weighs the robot arm over the manipulated object, causing manipulation to fail at contact. We measure
both failures directly on recorded data by asking whether a lower latent distance corresponds to true progress toward the goal. Across five
environments, six visual representations (including raw pixels and V-JEPA 2), and nine Meta-World tasks, this ranking test cleanly separates the two
failures, and the graph distance to a goal predicts where closed-loop planning breaks down. Both failures share a single remedy: replacing the distant
goal with a nearby target that lies on a recorded trajectory. We realize this principle with Graph Anchors, a training-free planner for frozen latent
world models. It builds a sparse graph once over the model's own training frames and aims each planning step at a node a few moves ahead along the
shortest path to the goal, keeping paths on recorded transitions wherever contact matters. On goals up to 400 steps away, Graph Anchors raise LeWM's
success from 13% to 85% on PointMaze-Medium, from 8% to 87% on PointMaze-Large, and from 5% to 71% on Push-T, where stronger search toward the goal
fails and where prior training-free planners break down under missing step horizons, subsampling shortcuts, or fragmented contact. 

---
# Angular Interpolation in Sparse-View Low-Dose CT Reconstruction: Placement Effects and Data-Consistency Step-Size Scaling 

**Authors**: Yitian Tang, Yuncheng Zhang, Hualiang Wang, Zuozhu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.34508)  

**Abstract**: Dual-domain networks combine projection and image processing for sparse-view low-dose computed tomography (CT). We quantify the contribution of angular interpolation before learning and distinguish its roles in sinogram initialisation and the image residual base. On synthetic parallel-beam data, parameter-free interpolation reduces noise-free missing-view mean absolute error by 99.7% relative to zero filling at 60 views. Full-view filtered back-projection (FBP) of the jointly learned completed sinogram adds 1.27 dB of peak signal-to-noise ratio (PSNR) over interpolated FBP. In a data-consistency dual-domain network (DC-DuNet), a 2x2 configuration ablation controls branch width, depth and training schedule while varying these placements and their associated input channels. After 20 epochs, the PSNR contrasts are +1.98 dB for sinogram initialisation and -2.16 dB for the interpolated image base; the image-base ranking also reverses during training. The sinogram-initialised variant exceeds an image-only U-Net by 0.96 dB at 60 views and 10% dose, and by 1.28 dB on independently generated synthetic data. The U-Net leads at higher view counts and with a shared interpolated image base. Without retraining, the variant retains a 0.12 dB advantage on held-out CHAOS abdominal CT under simulated degradation; the image-base contrast rounds to zero. The implemented projection/backprojection pair has empirical dominant gain of approximately 0.85NK, where N is image-grid width and K acquired view count. These results distinguish interpolation placement from network architecture and support NK normalisation for step calibration. 

---
# Noise threshold for learning with quantum memory 

**Authors**: Miaojiang Chen, Wenjing Xiao, Haiqiang Chen, Jiarong Liang, Anfeng Liu, Zhenrong Zhang, Athanasios V. Vasilakos, Saif Al-Kuwari, Ahmed Farouk, H. Herbert Song  

**Link**: [PDF](https://arxiv.org/pdf/2611.34474)  

**Abstract**: A quantum memory can reduce exponentially the number of experiments needed to learn a physical process, but real memories are noisy. Here we show that, for Pauli memory noise with eigenvalues $\lambda_X,\lambda_Y,\lambda_Z$, the advantage is governed by $\lambda_X^2+\lambda_Y^2+\lambda_Z^2$. We prove that no protocol with an entangled probe gains an exponential advantage when this sum is at most one, reduce the general case to the additivity of a Rényi-2 coherent information, and find coded memories that retain a growing advantage within 1% of this threshold. On a superconducting processor, nine simultaneously operated Bell pairs show an advantage that grows or shrinks exponentially with system size, reversing where a parameter-free model predicts. A three-qubit repetition code turns a shrinking advantage into a growing one with all processor and readout errors included, and under dephasing its memory performs better than Bell sampling in any memory frame on a perfect processor. 

---
# ProxyGenBench: Benchmarking Proxy-Guided Video Generation 

**Authors**: Zhenchen Tang, Haolin Lu, Songlin Yang, Xiaotong Zhao, Ran Zhang, Bo Peng, Shuai Li, Haotian Fan, Haobin Zhong, Yanni Zhang, Songlian Li, Ning Zhang, Xun Gao, Alan Zhao, Jing Dong  

**Link**: [PDF](https://arxiv.org/pdf/2611.34457)  

**Abstract**: Recent video generators produce realistic videos, yet controlling how scenes unfold remains challenging. Structural proxies, such as animated 3D blockouts, guide generation by specifying scene layout, motion, camera behavior, and event order. However, these coarse inputs leave details unspecified and may contain artifacts that should not be reproduced. Successful proxy-guided generation must therefore Preserve intended structure, Complete missing content and correct proxy artifacts, and Bind text, image, and audio conditions to the right entities and events. We introduce ProxyGenBench to study these interacting requirements, with 312 evaluation cases from 60 scenes across three scene tasks and controlled variations in proxy fidelity, text detail, and multimodal inputs. We evaluate five video generators using verified, case-specific checklists. Experiments show that proxies improve spatiotemporal control, but greater proxy detail can coincide with poorer completion, while strong structural adherence does not ensure correct binding. 

---
# Most of the Way There: Turning a Strong Video Prior into an Action-Conditioned World Model 

**Authors**: Zanyi Wang, Xia Chen, Liu Liu, Xianglong Guo, Dengyang Jiang, Haodong Li, Tingyang Xiao, Wenhao Hu, Zhizhong Su  

**Link**: [PDF](https://arxiv.org/pdf/2611.34435)  

**Abstract**: Modern video generators excel at producing visually plausible futures guided by text and image prompts. Yet their emphasis on semantic plausibility does not guarantee fidelity to the physical consequences of specific actions. Action-conditioned world models (ACWMs), in contrast, aim to simulate how the world responds to robot actions, enabling scalable policy evaluation and improvement without costly real-world rollouts. The challenge is not merely to generate a plausible future, but to predict what follows from a specific physical action. As video priors grow stronger, we ask how little adaptation is truly needed to bridge this gap. We introduce Dianjing, a minimal recipe that preserves and unlocks the capabilities of strong video priors through unified physical action conditioning, training concentrated at the high noise levels where motion and layout, and thus the consequences of actions, are decided, and native history conditioning. Built upon MiniMax H3, a 33B video foundation model, Dianjing freezes the pretrained backbone and trains only 0.34% of its parameters using approximately 65 hours of robot video across two embodiments, without additional robot-specific pretraining or camera calibration. Controlled ablations show that where training is allocated across noise levels matters more than how actions enter the network. On WorldOracle, which evaluates predictions against paired real-world executions, Dianjing achieves the best scores on four of the five evaluation tracks and ranks second on AgiBot-Long. Our findings suggest that strong video priors already bring us most of the way to world modeling; lightweight physical action grounding can unlock their potential for real-world prediction. 

---
# One Representation to Rule Them All -- Open-Vocabulary Change Detection from Text or Visual Examples 

**Authors**: Blaž Rolih, Matic Fučka, Filip Wolf, Luka Čehovin Zajc  

**Link**: [PDF](https://arxiv.org/pdf/2611.34430)  

**Abstract**: Open-vocabulary change detection (OVCD) aims to localise user-specified changes without retraining for predefined categories. Existing approaches typically rely on textual descriptions and combine separate models for region proposal, semantic recognition, and temporal comparison. This limits how target concepts can be expressed and makes detection dependent on several intermediate predictions. We introduce OVTSID3, a training-free framework that unifies region grouping, concept identification, and temporal comparison in the representation space of DINOv3. OVTSID3 supports two complementary modes of concept specification: zero-shot textual queries or one-shot visual prompting through a masked reference pair. OVTSID3 outperforms all prior training-free OVCD methods on two building and two natural-disaster benchmarks, raising the average F1 by 8.9 points. OVTSID3 also supports semantic change detection and achieves state-of-the-art on SECOND. These results demonstrate the flexibility of a unified representation for text- or example-guided change detection without task-specific model training. 

---
# Calibration of Neuromusculoskeletal Models Using Differentiable Physics: Recommendations for Improving Model Accuracy 

**Authors**: Philip Sanderink, Yingfan Zhou, Shuzhen Luo, Cheng Fang  

**Link**: [PDF](https://arxiv.org/pdf/2611.34428)  

**Abstract**: Differentiable physics has shown potential to usher in a new paradigm in human neuromusculoskeletal (NMS) model calibration. Early results from pioneering works have demonstrated that auto-differentiable NMS models can greatly reduce NMS model calibration time while achieving comparable accuracy compared to traditional calibration methods. The efficient gradient-based optimizers have enabled calibration of increasingly sophisticated human models with a large number of parameters, which seek to narrow the gap between simulation and reality. Several modules have been evaluated separately in different studies, but have never been directly compared within the same study. In this work, these modules are compared on the same differentiable upper limb model with a 3-DoF shoulder, a 2-DoF elbow, and 11 superficial and 5 deep muscles in the calibration to find an optimal pipeline for improving model accuracy. The considered modules include muscle synergy, learnable bone parameters, optimizable geometry of wrap objects, and constraint torque of the differentiable arm model. To evaluate the effect and interplay between the modules on calibration accuracy, one or more modules were used in each of 16 calibration configurations, which were compared in terms of the error between two joint torques calculated by the forward and inverse dynamics with the calibrated model on 9 distinctive testing movements across 4 subjects. Results showed that torque loss was reduced by 23.6% on average using muscle synergy extrapolation to estimate deep muscle activities, and by 10.9% on average through bone dynamic parameter optimization, which are both recommended to be incorporated in the calibration. 

---
# CHASE: Curiosity-Guided Skill Acquisition for Embodied Task Execution 

**Authors**: Zixuan Wang, Wei Zhang, Chenlei Zhu, Naiyu Chen, Xihua Wang, Jie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.34416)  

**Abstract**: Robots executing tasks in partially observed environments must resolve uncertainty while operating with incomplete skills. Inspired by human curiosity, we investigate how robots can formulate task-relevant hypotheses, seek evidence, and acquire missing skills during task execution. We propose CHASE, a Curiosity-guided Harness for Acquiring Skills from Evidence, which operationalizes curiosity as selecting investigations that resolve task uncertainty and expand executable capabilities. Four modules connect dynamic scene understanding, hypothesis-driven planning, on-demand skill acquisition, and verified execution. The planner balances information value, skill reuse, and acquisition/execution cost. Missing skills trigger bounded exploration, learning from verified segments, and held-out admission during acquisition; separate checks verify physical effects and acquired evidence. Newly admitted skills expand the executable investigation set, closing the loop between resolving uncertainty and acquiring capabilities. Simulation and real-robot experiments evaluate investigation, skill acquisition, and task execution. CHASE achieves 50-80% task success under combined information and capability gaps, with 70-90% evidence resolution. 

---
# SemProxy: Semantic 3D Proxies for Controllable Agentic 3D Modeling 

**Authors**: Shuyuan Zhang, Changjian Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.34414)  

**Abstract**: Coding agents are increasingly capable of generating 3D models, yet enabling users to effectively control the modeling process remains challenging. Existing workflows rely primarily on text prompts and reference images, which often struggle to convey precise spatial intentions and require iterative corrections. We introduce SemProxy, a lightweight intermediate representation that bridges intuitive human interaction and agentic 3D modeling. Instantiated as labeled 3D bounding boxes, semantic 3D proxies allow users to directly specify object parts and spatial arrangements while leaving geometric details to coding agents. These proxies can be serialized into structured text and rendered as images, making them directly accessible to existing coding agents without additional training or modifications. We further develop ProxyStudio, a user interface that integrates text, image references, and interactive 3D proxy editing into agentic modeling workflows. We demonstrate the versatility of SemProxy across object- and scene-level modeling tasks with different coding agents, and evaluate its effectiveness in supporting controllable human-agent collaborative 3D modeling through a user study. 

---
# When Skills Fail to Compose: Compatibility and Retention in Language Models 

**Authors**: Yan Lin, Zhongliang Guo, Jingyu Sun, Qiyan Zhao, Ruihong Zeng, Ying Chen, Zhengtao Yao, Yan Wen, Kunyu Peng, Yuxia Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.34398)  

**Abstract**: Skill acquisition alone does not establish a reusable skill library: task proficiency does not specify a compositional interface. Motivated by agents that acquire and refine skills, we study independently learned soft-prompt skills on frozen language models, with skill identities and execution order supplied. Across four digit-task bases and three record-editing models, mean composition accuracy in one call is at most 7.6%, compared with 88.2% to 100% over two calls that add computation and an explicit intermediate output. Neutral-insertion training improves robustness but leaves substantial interference, which block length and per-token norm alone cannot explain on two of four digit bases. Blocks of the same operation need not be interchangeable: after training next to frozen learned partners, a post hoc test on two Qwen bases shows that replacing those partners with independently learned blocks of the same operations lowers median composition accuracy from 98.0% and 95.3% to 12.7% and 0.5%, although every replacement scores at least 90% alone. Repair can improve a composition while impairing the original skill, and rehearsal improves retention while transfer remains limited on Qwen2.5-0.5B. Internal interventions constrain explanations without identifying a shared failure mechanism. These findings motivate evaluating compatibility, retention and transfer alongside acquisition when building skill libraries for self-evolving agents. Code is available at this https URL. 

---
# ConTra: A Training Framework for Effective Contextual Retrieval via Prefix Generation 

**Authors**: Jiajun Hou, Jingbang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.34366)  

**Abstract**: Data management systems organize large text collections for tasks such as finding relevant studies and extracting contract clauses. Systems make long documents searchable by splitting them into independently indexed chunks. Chunking can separate text from its identifying details. The indexing challenge is to keep each chunk independently searchable while preserving the context needed to find and rank it. Existing methods either encode that context in chunk vectors or attach it as a text prefix. Context encoded only in vectors is unavailable to text rerankers. Prompted prefixes serve both vector search and reranking, but are costly to generate at indexing time and can underperform contextual encoders. We propose ConTra (Context That Travels), a reinforcement learning framework for training lightweight prefix generators. Its reward combines feedback from frozen embedding and reranking models with factual verification and length control. ConTra generates one prefix per chunk during index construction and stores the augmented text and its embedding for reuse across queries. To evaluate contextual retrieval across workloads and under interference, we introduce RealContextBench, a benchmark with five application collections and three controlled conditions that isolate the effect of distracting text. On ConTEB and RealContextBench, ConTra-2B outperforms the prompted 9B baseline using Anthropic's method and the PPLX-4B contextual encoder, with a margin of up to 3.15 points over their best overall scores, while indexing short documents 7.74 times and 2.15 times as fast as those baselines, respectively. Post-trained only on ConTEB, it leads the standalone methods on RealContextBench after reranking and is less sensitive than the prompted 2B baseline to random and topically similar distractors. Code, datasets, and trained models are available at this https URL. 

---
# Inverse Reinforcement Learning with State-Dependent Foresight for Human Decision-Making 

**Authors**: Jinil Kim, Jinwoo Jeong, Hyeongjun Yun, Min-hwan Oh, Woo-Young Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2611.34351)  

**Abstract**: Computational modeling of behavior represents individuals with a model fitted to their observed behavior and characterizes them through its parameters. Naturalistic tasks incorporate the complexity of real environments and yield high-dimensional behavioral trajectories. Inverse reinforcement learning (IRL) infers the latent reward function underlying these trajectories. Standard IRL, however, assumes effectively infinite-horizon planning, whereas human foresight is both bounded and state-dependent. Here we introduce SC-AIRL (State-Conditional Adversarial IRL), which builds these two cognitive properties into the planner. Per-depth policies share one reward but differ in how far ahead they sum it explicitly. A state-conditional router mixes them according to the current state. Across held-out episodes in pedestrian crossing and highway overtaking tasks, SC-AIRL improves imitation over Standard AIRL, raising top-1 accuracy from 0.549 to 0.647 and from 0.629 to 0.718, respectively, while more closely reproducing participant behavior in rollouts. Inferred planning depth correlates with self-reported anxiety and motor impulsivity. SC-AIRL thus reproduces human behavior more closely than Standard AIRL while providing an interpretable, model-derived measure of state-dependent foresight. 

---
# Puzzling-Bench: Evaluating Language Models on 23,777 Human-Written Puzzles with Community-Verified Answers 

**Authors**: Abhinav Sharma, Li Li, Chien Van Nguyen, Sai Karthik Navuluru, Bo Ni, Xiangbo Gao, Xiaolei Huang, Yue Zhao, Nesreen K. Ahmed, Tyler Derr, Yu Wang, Hongjie Chen, Zhengzhong Tu, Thien Huu Nguyen, Ryan A. Rossi, Franck Dernoncourt  

**Link**: [PDF](https://arxiv.org/pdf/2611.34331)  

**Abstract**: We introduce Puzzling-Bench, a benchmark of 23,777 puzzles with community-verified answers drawn from the Puzzling Stack Exchange archive (2014-2026): riddles, ciphers, logic grids, lateral-thinking problems, wordplay and recreational mathematics, a third of them with images. Every puzzle carries the answer accepted by its author, a short reference answer taken from it (LLM-extracted for the evaluation sets and checked on a 50-puzzle human review), the author's own hints, community tags and a creation date, so that the benchmark supports hint ablations, category analysis and a rolling post-cutoff slice (254 puzzles posted after the training cutoff of six of the nine models we evaluate, including the best one) as a contamination check. We evaluate nine frontier and open models with a two-tier scorer (normalized exact match, then an LLM judge) and validate the judge with three judges from two model families: they agree on 95-97% of verdicts (kappa 0.82-0.96), move no model by more than 1.6 points, and show no own-family preference. The best model, GPT-6.1, reaches 74.8% on a shared 255-puzzle set; a hard core of 17 puzzles defeats all nine models; model failures are strongly correlated, and the second-best model contributes a single puzzle to a greedy ensemble, yet an oracle over all nine reaches 91%. On a 100-puzzle multimodal pilot, withholding the question's image costs the two best models 24 and 35 points. We release the dataset, the scoring pipeline and all per-row provenance. 

---
# LUDIUS: A Benchmark for Embodied Social Intelligence in Everyday Human-Robot Interactions 

**Authors**: Chaewon Kim, Sangheon Lee, Seyeon Choi, Kangwook Lee, Yea Seul Kim  

**Link**: [PDF](https://arxiv.org/pdf/2611.34328)  

**Abstract**: Social intelligence in a robot requires interpreting a situation from the perspectives of the people involved and using that understanding to decide when and how to act. We introduce LUDIUS, a benchmark for evaluating how vision-language models exercise this judgment through a robot's verbal responses and actions, organized around 15 social intelligence skills drawn from social psychology, sociology, and cognitive science. LUDIUS comprises six editions, each a collection of episodes spanning one simulated week. Three editions each cover all 15 skills in a different household. The other three focus on interactions involving multiple people, safety, and context-specific norms, respectively. We evaluate frontier models and a smaller model finetuned on social-interaction data we generated, using a common evaluation harness. The harness supplies each model with the robot's camera views and residents' reactions, retrieves relevant interaction history, and converts the model's tool calls into robot verbal responses and actions. Models tend to perform worse when they must reason about what a resident saw or heard earlier, reconcile competing interests, or fulfill a request within safety constraints and household rules. Beyond the home setting, LUDIUS's design pipeline offers a way to construct social intelligence benchmarks by translating social intelligence skills into context-specific judgments and observable robot behavior. 

---
# The cost of being seen as AI in group bargaining 

**Authors**: Tahmineh Godazgar, Nico Mutzner, Taha Yasseri  

**Link**: [PDF](https://arxiv.org/pdf/2611.34322)  

**Abstract**: As artificial intelligence (AI) becomes embedded in collective decision-making, people's perceptions of AI may shape who reaches agreement and what they receive. We studied this in a repeated three-player majoritarian bargaining game with 2,234 participants. Players saw each opponent identified as Human or AI. The instructions described AI as acting either independently or on behalf of a human. Human participants made every decision. In mixed groups, players presented as AI received fewer points in opening offers, were included in fewer final agreements and earned about 18% less per round than those presented as Human. Smaller payoffs in agreements that included them accounted for most of the observed gap. Further analysis found a payoff gap even when all three players agreed. Social perceptions of AI may therefore shape who benefits from collective decisions, with potential costs for people represented by AI. 

---
# RAP: Radial-Angular Random Projection for Hyperbolic Representations 

**Authors**: Jiahong Liu, Wenhao Yu, Yifan Li, Yanyu Chen, Rex Ying, Menglin Yang, Irwin King  

**Link**: [PDF](https://arxiv.org/pdf/2611.34298)  

**Abstract**: Random projection is a basic compression primitive for Euclidean representation learning, with the Johnson--Lindenstrauss lemma giving a target dimension $k=O(\varepsilon^{-2}\log n)$ independent of the ambient dimension. As hyperbolic representations gain increasing attention in modern Transformers, vision-language models, and large language models, the same need for oblivious dimensionality reduction arises, but no analogous primitive yet exists for hyperbolic representations. We study this gap. Two natural attempts fail in dual ways: a tangent-space scheme gives a clean local guarantee but is chart-bound and pays a curvature penalty, while a fully hyperbolic projection in the hyperboloid model is geometrically native but mixes radial and angular error before feeding them into the conditioning of $\operatorname{arcosh}$. We propose radial-angular random projection (RAP), which preserves the radial coordinate exactly and applies a Euclidean random projection only to the angular direction, lifting the result back to the target hyperboloid. RAP is hyperbolic-to-hyperbolic by construction, costs one Gaussian matrix-vector product, and reduces hyperbolic compression to a normalized angular JL question on $\mathbb{S}^{d-1}$. We prove a finite-set RAP guarantee with explicit $\sinh(\sqrt{c}\,r_x)\sinh(\sqrt{c}\,r_y)$ conditioning, complemented by a local tangent-space JL theorem and an arcosh-conditioning lemma for the fully hyperbolic comparator. Across an origin-centered broad-radius point cloud, a balanced tree, a clustered Gaussian, and the WordNet noun hierarchy, RAP attains the lowest mean relative distortion in every $(n,k)$ cell, reducing distortion over the strongest hyperbolic baseline by approximately $48\%$--$49\%$ on broad-radius data. RAP also scales to million-point datasets and improves retrieval on vision--language representations over hyperbolic projection baselines. 

---
# WebVibe: Multimodal Interactive Website Generation via Reverse Synthesis and Visual Self-Distillation 

**Authors**: Tianxiang Jiang, Sheng Xia, Yilin Long, Kanghui Tian, Linquan Wu, Yongheng Zhang, Daixian Liu, Di Yin, Yi Wang, Yu Qiao, Qinglong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.34292)  

**Abstract**: Although multimodal large language models translate single screenshots into static HTML, generating interactive functional websites from design specifications still struggles with multi-page routing and stateful widget transitions. Examining existing training pipelines reveals two bottlenecks: (1) in forward data synthesis (X -> Y), writing the specification X before code execution leaves unimplemented or altered requirements in X as supervision noise; and (2) in reinforcement learning, scalar rewards discard the rich state transitions observed during replay. Based on these findings, we introduce WebVibe and Interactive-SDPO to train InteractiveCoder. WebVibe reverses synthesis into Y -> X by verifying cross-page and in-page interactions in the browser before deriving multimodal inputs from replay-confirmed states to build the 31,542-site WebVibe-32K corpus with reusable execution contracts. Interactive-SDPO then conditions a weight-sharing self-teacher on the replay trace and a 2x2 state-transition montage to provide token-level supervision at zero inference-time overhead. On the public Vision2Web L2, one-shot InteractiveCoder-9B nearly 3x its base model's functional score (20.2 to 55.2 FS) and matches a 1T-parameter coding agent in a single pass with 111x fewer total parameters and 78x fewer tokens. On our WebVibeBench, InteractiveCoder-4B and 9B further outperform all evaluated baselines by over 23 points. 

---
# StateWeaver: Multi-Agent Coordination of Story-State Transitions for Long-Horizon Video Generation 

**Authors**: Modi Jin, Xuying Zhang, Yijing Guo, Haoran Li, Yanwen Ma, Jing Wang, Chenyu Ge, Ming-Ming Cheng, Haoyang Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.34290)  

**Abstract**: Modern video foundation models generate high-quality short clips, but preserving long-video consistency and correctly depicting event-driven state changes remain challenging. Existing agentic workflows face these challenges because continual planning and repair accumulate context, while reference updates remain limited. We introduce StateWeaver, which combines Active States and Hierarchical Collaboration to connect review-driven visual reference updates with cross-shot state inheritance. Active States translate state discrepancies identified during review into concrete visual asset requirements that guide reference construction, validation, and regeneration, preserving identity and already-correct content while repairing target attributes. Confirmed generation results are linked to the corresponding asset states, story time, and reference versions for verification and reuse by subsequent shots. Hierarchical Collaboration isolates execution contexts by task, clip, and repair round, and uses hierarchical memory to organize this process across multiple sets. Across multiple benchmarks, StateWeaver outperforms existing methods and produces long videos with sustained cross-shot consistency under predefined and autonomously planned storyboards. 

---
# Latent-in-the-Middle: On the Security of Cross-Model Latent Communication in Multi-Agent Systems 

**Authors**: Manit Baser, Vyoman Jain, Rachel Yeo Hui Min, Xianglin Yang, Meenatchi Sundaram Muthu Selva Annamalai, Dinil Mon Divakaran  

**Link**: [PDF](https://arxiv.org/pdf/2611.34278)  

**Abstract**: As large language models increasingly collaborate in heterogeneous multi-agent systems, latent communication has emerged as an efficient alternative to exchanging natural-language messages. By sharing internal representations, agents can avoid repeatedly processing the same context, reducing communication overhead and prefill computation. One such mechanism is cross-model Key-Value (KV) cache transfer, which uses a learned mapper to translate a source model's KV cache into one a target model can use. However, this handoff means the target model consumes context it did not encode itself, thus calling into question whether its safety alignment still holds.
In this work, we investigate the security implications of latent handoffs by introducing Chasm, a framework characterizing cross-model handoff attacks across three new attack surfaces: the source model, mapper, and handoff mechanism itself. Our experiments reveal that latent handoffs substantially undermine safety alignment and prompt-injection defenses. Notably, a target model that follows only 0.4% of benchmark prompt injections natively, follows up to 44.2% when transferred through a mapper from a source model. On the other hand, for jailbreak attacks, transfer alone raises attack success by up to 24.6 pp within a model family and zero-shot attacks by up to 69.2 pp, whereas white-box optimization through the mapper boosts the attack success rate to 92.5%, against 37.5% for ordinary GCG on the same target. To address these vulnerabilities, we propose a lightweight target-side cache guard that corrects transferred caches to restore the target model's refusal, considerably mitigating these attacks. 

---
# LatentCP-DPO: Conformal Pessimism for DPO with Finite-Sample Guarantees 

**Authors**: Hongwei Xu, Jingyi Liu, Vincent Y. F. Tan, Zai Yang, Yulian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2611.34275)  

**Abstract**: Direct Preference Optimization (DPO) learns a response policy from human preference data, but with a limited dataset it can overfit to noisy or weakly informative preference labels. We introduce LatentCP-DPO, a framework that combines conformal prediction with pessimistic preference optimization for Bradley-Terry (BT) pairwise feedback and Plackett-Luce (PL) ranking feedback. Building on LatentCP's construction of uncertainty sets for unobserved model parameters, we address how to incorporate this uncertainty into preference optimization and characterize its effect on policy performance. Under exchangeability and a correctly specified preference model, we construct uncertainty sets for latent relative rewards with finite-sample marginal coverage from conformal prediction sets for observable preferences. We then learn a response policy by maximizing the worst-case expected preference log-likelihood over these uncertainty sets, using an independent collection of prompts and candidate responses without preference labels. Under suitable assumptions, we establish finite-sample upper bounds on regularized policy suboptimality that separate policy-class approximation error, conformal uncertainty, and statistical estimation error. For BT feedback, we further establish a worst-case lower bound for LatentCP-DPO that matches the upper bound's reciprocal dependence on the number of unlabeled optimization examples. The lower bound also shows that the uncertainty sets can induce a bias that persists as the optimization sample grows. Together, these results characterize both the statistical accuracy and potential limitations of preference optimization using conformal uncertainty sets. Experiments support the benefits of the proposed framework in finite preference datasets. 

---
# LLM-coupled interpretable tool wear monitoring with multi-evidence reasoning in CNC machining 

**Authors**: Yuntong Chen, Yingqi Li, Qiwen Shen, Guobin Zhao, Xitian Tian, Lijiang Huang  

**Link**: [PDF](https://arxiv.org/pdf/2611.34270)  

**Abstract**: Interpretable tool wear assessment in CNC machining is challenging. Existing diagnostic models have high accuracy but provide limited guidance for tool replacement. This study proposes a two layer framework combining deep learning wear classification and LLM assisted reasoning. The first layer uses PCA Loading feature selection and optimized GRU to recognize three wear states, achieving 98.38 percent macro F1 score on the PHM2010 dataset. The second layer uses LLMs to generate natural language reports and replacement suggestions from classification results and feature trends. Engineer evaluation shows the reports outperform rule templates. Cross dataset tests on NASA milling data support the generalizability of this framework. 

---
# Weakly Conditioned Diffusion Model Augmentation for Prostate Segmentation 

**Authors**: Chibike Ugbam, Muhammad Bilal, Jia Wu, Shafaq Murad, Rizwan Qureshi, Hazrat Ali  

**Link**: [PDF](https://arxiv.org/pdf/2611.34230)  

**Abstract**: Deep learning models for prostate MRI segmentation often struggle to generalize when trained on limited heterogeneous data, leading to boundary errors and over-segmentation. We present an augmentation framework using mask-conditioned synthetic and diverse T2-weighted MRI volumes to improve boundary precision and reduce over-segmentation. We employ a 2.5D weakly mask-conditioned denoising diffusion probabilistic model (DDPM) via channel concatenation. The diffusion model conditions on stacks of adjacent binary masks from training folds. The model generates T2 volumes consistent with the conditioning anatomy while diversifying tissue appearance. We evaluate the proposed approach on two cohorts of data: (i) the LUND probe cohort (n=432) for strategy selection and training nnU-Net from scratch with a 23% synthetic data mix, and (ii) the Medical Segmentation Decathlon (MSD) prostate cohort (n=32) evaluated with a 20% synthetic data mix. For the LUND cohort, we found that training an nnU-Net model from scratch with a 23% synthetic data mix reduced the mean false-positive voxels per case from 11,102 (baseline) to 10,757. On the MSD cohort, the augmented model with a 20% synthetic mix achieved a mean cross-validated Dice of 0.9094, improving over the baseline (0.8935) by 1.59 percentage points and reducing mean false-positive voxels by 23.1% (4,201 to 3,229). Experimental results on both these cohorts suggest that mask-conditioned DDPM augmentation can reduce over-segmentation errors and improve boundary adherence in data-constrained settings. 

---
# The Socratic PAUSE: safeguarding scientific judgement at AI speed 

**Authors**: Abdulkadir Celik, Sir Bashir M. Al-Hashimi  

**Link**: [PDF](https://arxiv.org/pdf/2611.34222)  

**Abstract**: Generative AI can turn a tentative scientific question into hypotheses, methods, code and interpretations within minutes. Yet generating a contribution does not establish that it deserves scientific reliance: researchers must still examine its evidence, assumptions and implications, and remain accountable for the decision to rely on it. Using Einstein's sketch of scientific discovery as a lens, we argue that AI has accelerated discovery far more than justification. In human-AI research partnerships, confidence can therefore outgrow evidence through three mechanisms: recursive reinforcement, loss of provenance and miscalibrated reliance on AI-generated contributions. Verification debt accumulates when such contributions are carried into later claims or decisions without adequate scrutiny. Accordingly, we propose the Socratic Pause as a deliberate checkpoint at consequential decision gates, where an AI contribution remains provisional while a researcher examines its basis. The PAUSE protocol structures this examination through five steps: Pin down the claim, Ask and challenge, Uncover assumptions and alternatives, Scrutinise the evidence and Exercise judgement. It concludes with a recorded decision to accept, revise, reject or defer the contribution. We show how AI research tools can support this practice without displacing human judgement, and outline a framework for evaluating whether it improves verification, decision-making and independent understanding. The aim is to preserve the freedom to explore with AI while ensuring that those responsible for the work can say what was examined, what remains uncertain and why the next step is justified. 

---
# GeoCAM: Self-Supervised Sonar-Optical Representation Learning via Geographic Overlap Supervision 

**Authors**: Fadi Abdeladhim Zidi, Andrea De Salve, Giuseppe Celeste, Abdellah Zakaria Sellam, Paolo Spagnolo, Cosimo Distante  

**Link**: [PDF](https://arxiv.org/pdf/2611.34218)  

**Abstract**: Cross-modal self-supervised learning for paired side-scan sonar and underwater optical imagery is a promising direction for benthic habitat understanding, but it remains underexplored because existing methods do not fully exploit the one-to-many correspondences and geographic metadata available in real opti-acoustic surveys. This limits the ability to learn representations that are both semantically meaningful and geometrically consistent from abundant unlabeled marine data. To address this problem, we present GeoCAM, a geography-guided framework that learns a shared embedding space for sonar tiles and their associated optical views without semantic annotations. GeoCAM achieves its goal by combining three components: (i) a Set Attention Aggregator that adaptively summarizes multiple optical observations into a single tile-level representation; (ii) a geometry-weighted InfoNCE objective that converts GPS-derived footprint overlap into soft positive supervision; and (iii) an optional cross-modal masked reconstruction branch that encourages physically grounded cross-modal alignment. Experiments on the BenthiCat SSS-CAM dataset show that GeoCAM can be effectively exploited to learn which sonar and camera samples describe the same seabed patch, improving cross-modal retrieval and alignment over strong contrastive baselines. Code is available at this https URL. 

---
# Beyond Logits: Coherent Hallucination Mitigation via Attention Contrastive Decoding 

**Authors**: Yujia Chen, Rui Sun, Huayu Mai, Wangkai Li, Zhangyu He, Bingzhou Wang, Aibing Li, Wenzhang Sun, Tianzhu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.34207)  

**Abstract**: Large Vision-Language Models (LVLMs) demonstrate impressive multimodal capabilities, yet suffer from hallucination - generating factually inaccurate content. Contrastive Decoding (CD) mitigates this by contrasting amateur and expert branches at the logit level. However, our investigation reveals that such logit-level interventions fundamentally compromise generation coherence, necessitating restrictive penalty constraints unrelated to hallucination suppression.
We introduce Attention Contrastive Decoding (ACD), a training-free plug-in that complements logit-level CD by relocating part of the contrastive operations to the attention mechanism. Operating at an earlier stage of the forward pass, ACD performs smooth semantic-preserving interventions through an Adaptive Subtraction Strategy (ASS), which attenuates hallucination-associated attention patterns while amplifying critical visual information. Extensive experiments demonstrate that combining ACD with existing CD methods (e.g., VCD+ACD) produces substantially more coherent outputs with further reduced hallucinations, eliminating restrictive penalties while enabling trustworthy multimodal generation. 

---
# MINE: A Training-Free Defense against Indirect Prompt Injection via Mutual Information and Embedding Variance 

**Authors**: Truong Phan, Ngoc-Quan Pham, Dat Quoc Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2611.34188)  

**Abstract**: Existing defenses against indirect prompt injection in retrieval-augmented generation (RAG) are inherently reactive: prompt-engineering methods are often overridden by the injected instructions, while supervised detectors must be retrained for each new attack style. We propose a different paradigm: defend by maximizing relevance, not by detecting attacks. Our method, MINE, is training-free and attack-agnostic, and it also serves as a query-conditioned reranker. To evaluate defenses under realistic conditions, we introduce LC-Inj-SQuAD, a long-context benchmark with an average of 4,032 words per context. We evaluate on four datasets spanning short-context QA (Inj-SQuAD, Inj-TriviaQA), long-context QA (LC-Inj-SQuAD), and agent tool-call injection (InjecAgent). Across all four, MINE achieves the best accuracy--security trade-off among training-free defenses. It is also the only defense that stays usable against a fully adaptive attacker, generalizes to agent tool-call injection, and achieves the lowest inference latency on long contexts. 

---
# Hierarchical Semantic Learning for Multimodal Sentiment Analysis with Missing Modalities 

**Authors**: Chunlei Meng, Rong Fu, Zhan Chen, Yi Liu, Zhuo Zou, LiDa Xu, Chun Ouyang, Zhongxue Gan  

**Link**: [PDF](https://arxiv.org/pdf/2611.34184)  

**Abstract**: Multimodal sentiment analysis (MSA) in real-world systems must handle unpredictable modality missingness caused by sensor failures, privacy masking, and unstable transmission. Existing methods either learn reliability-aware representations or reconstruct missing inputs, often using a single flattened encoder output. This overlooks deep encoders' semantic hierarchy and intra- and inter-modal interactions, causing semantic averaging and ill-posed mappings. We propose Hierarchical Semantic Learning (HSL), a framework that exploits semantic hierarchy for robust MSA with missing modalities. Hierarchical Semantic Decoupling exposes low-, mid-, and high-level feature streams from each modality encoder. Cross-Modal Cross-Level Semantic Reconstruction performs ladder-style recovery, using high-level semantics from observed modalities as global anchors to progressively reconstruct mid-level structures and low-level details of missing modalities. To stabilize cross-level reconstruction, Hierarchical Cascade Error Supervisor applies prototypical contrastive regularization to observed and reconstructed features, enforcing level-specific semantic consistency and limiting error propagation. Finally, Hierarchical Interactive Fusion captures dense interactions across modalities and semantic levels to produce robust predictions under diverse missingness patterns. Experiments on CMU-MOSI, CMU-MOSEI, IEMOCAP, and CH-SIMS show that HSL outperforms baselines in most missing-modality settings. 

---
# Agentic-Level Adversarial Purification: Revisiting Diffusion Purifiers on Native-Resolution Agentic Inputs 

**Authors**: Yan Lin, Zhongliang Guo, Jingyu Lin, Daizong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2611.34181)  

**Abstract**: Vision-language agents act on native-resolution renders of multi-element pages, where one perturbed photo or a few altered text strokes can redirect the decision. Any content provider can craft such a perturbation on the open model behind the agent, so a plug-in purifier is the natural first line of defense. Yet diffusion purifiers were designed and evaluated at the fixed resolution of the diffusion model on classifier or natural-image benchmarks, and defense evaluations for agents test JPEG, blur and noise but no diffusion purifier. We present the first study of adversarial purification at the agentic level, for an agent built on Qwen3-VL-8B: purifiers run at native resolution on multi-element inputs, face grey-box attackers who know the agent's model but not the defense, and are compared at matched clean cost. We show that treating agentic inputs as classifier inputs misleads both evaluation and design. Resizing the input to the resolution of the diffusion model, as existing purifiers do, raises the TextVQA clean cost of the Divide-and-Conquer purifier (DC) from 4.0 to 35.7 points. On web-shop pages, DC keeps 85.0% grey-box robust accuracy against web-hardened perturbations, where cheap input transformations at matched clean cost keep at most 40.0%, and this robustness comes from the global noise level: input-estimated masks and noise levels add none. These findings hold on a disjoint held-out set and on a second VLM and yield three requirements for defenses, we release the protocol. 

---
# Multi-view fnirs identifies neurophysiological subtypes in post-stroke aphasia 

**Authors**: Zijian Kang, Bingbing Lin, Jinglei Ni, Meifang Tian, Wai Ting Siok, Jia Huang, Nizhuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.34169)  

**Abstract**: Post-stroke aphasia is clinically heterogeneous, and conventional classifications may not capture differences in brain functional organisation. We tested whether resting-state and task-based functional near-infrared spectroscopy (fNIRS) could identify neurophysiological subgroups alongside clinical this http URL-nine patients underwent unsupervised clustering using three fNIRS feature views: resting-state activity and connectivity, language-task activation, and rest-to-task connectivity reconfiguration. Similarity network fusion and spectral clustering integrated the views. Clinical and behavioural measures were excluded from this http URL clusters (n = 32, 17 and 30) were identified. This solution had the largest eigengap and was the only candidate significant after permutation testing and Holm correction (adjusted P = 0.001). It was stable under subsampling (mean adjusted Rand index, 0.773), but agreement fell to 0.217 when the resting-state view was removed. Clusters differed in most regional intrinsic-activity measures and all resting-state connectivity summaries, with few differences in task activation or connectivity reconfiguration. Their profiles showed higher right anterior prefrontal activity, distributed fronto-sensorimotor and parietal alterations, and reduced interhemispheric connectivity, this http URL membership was not significantly associated with conventional aphasia subtype (Cramér's V = 0.287), and baseline clinical language scores and task performance did not differ between this http URL findings identify three neurophysiological subgroups distinguished mainly by intrinsic activity and resting-state functional connectivity, not clearly aligned with conventional aphasia classifications or baseline clinical measures. Multi-view fNIRS analysis may complement clinical characterisation, requiring independent and longitudinal validation. 

---
# Strategic and Robust Task Allocation for Combat Objectives in Adversarial Multi-Agent Environments 

**Authors**: Jonas Peché, Dmitry Matveev, Vladimir Ciric, Lizaveta Sakalova, Günter Wallner  

**Link**: [PDF](https://arxiv.org/pdf/2611.34158)  

**Abstract**: Task allocation in contested-objective team games requires coordinated decisions under uncertainty, spatial constraints, and changing tactical conditions. In such settings, multiple agents must anticipate both teammate behavior and opponent actions. We study a hybrid predict-then-optimize approach for assigning controllable agents to objectives in real time. A masked multi-head model predicts tactical signals for each agent, including survival chance, local superiority, combat interaction, travel time, and objective contribution. These signals are combined with rule-based terms in a score matrix, and a minimum-cost flow solver computes coordinated global assignments constrained by capacity, occupancy, stability, and defensive strategies. Learning is limited to tactical scoring, while explicit rules and solver structure preserve controllability and interpretability. We evaluate the system using predictive metrics, offline assignment diagnostics, online A/B evaluation, and latency measurements. The system is an R\&D research prototype built on a commercial game and is not deployed in the shipped game. The results indicate that learned tactical estimates can support stable, feasible, and inspectable objective allocation under strict real-time constraints. 

---
# Lost in the Grid: Where Multimodal LLMs Fail Between Seeing and Planning 

**Authors**: Luoxi Zhang, Tianyi Tan, Jiawei Liu, Xingyu Zeng, Min Yang, Shiwen Ni  

**Link**: [PDF](https://arxiv.org/pdf/2611.34152)  

**Abstract**: Multimodal large language models (MLLMs) excel at static visual understanding and are increasingly applied to spatial planning, where a model interprets a visual layout and outputs an action sequence to reach a goal. Grid games are a common testbed for this ability because their rules are explicit and a solver can verify every move. Yet existing benchmarks mostly report end-to-end success rates, which hide whether a failure stems from visual perception, rule application, or multi-step planning. We introduce GridProbe, a diagnostic benchmark that 1) locates the failing stage between visual perception and planning and 2) verifies the diagnosis by targeted fine-tuning. On Sokoban, GridProbe pairs end-to-end solving with four probes (recognition, localization, single-step prediction, and state tracking) and gives every board both as an image and as an equivalent text grid. Evaluating seven MLLMs on Sokoban and a maze game reveals distinct bottlenecks behind similar success rates: proprietary models are limited by multi-step planning, whereas open-source models are limited by rule application or by visual grid indexing, i.e., counting rows and columns. Planning failures also hide perception failures: when planning is easier, the visual cost of the same model grows from 7.6 to 37.7 points. The diagnosis identifies what to train: fine-tuning only the vision module of the perception-limited model on board reading raises its interactive Sokoban success 3.5-fold (5.8% to 20.4%) and transfers zero-shot to the unseen maze game (single-turn success 13.8% to 67.1%), whereas the same data does not help a rule-limited model and training on solver solutions brings no gain. We will release the benchmark and code at this https URL. 

---
# Reproducer Synthesis for AI-Reported Bugs in the Linux Kernel 

**Authors**: Xiao Wu, Yunbo Ni, Hao Sun, Yuanyuan Yuan, Shaohua Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.34142)  

**Abstract**: Agentic review systems are increasingly flagging suspected bugs in Linux kernel patches as textual reports. A maintainer who receives such reports needs further inspection to determine whether the report is true. The inspection requires understanding whether the bug can be triggered and, if necessary, manually constructing one. As a consequence, despite the excessive textual explanation, many reports go unanswered. In this paper, we argue that a runnable proof-of-concept (PoC) is the key information in such AI-generated kernel review reports. It is, however, challenging to automatically synthesize a valid and runnable PoC.
We present ReproCue, a framework that generates PoCs for AI-generated Linux kernel reviews. ReproCue replays the historical kernel bugs and learns one reproduction finite-state-machine (FSM) per subsystem from successful runs. An FSM records the steps a reproduction goes through, together with the requirements observed at each step. Given a new review, ReproCue follows the FSM of that subsystem, adjusting the PoC and the environment as execution shows what is missing. In 100 recent AI-generated reviews, ReproCue writes PoCs for 49 of them, compared with 10 for the current agentic reproducer and 9 for the state-of-the-art directed fuzzing. Excitingly, with our generated PoCs, Linux kernel maintainers confirmed 69.4% of them, much higher than 9.3% from Sashiko where only textual explanations are included, making AI reviews more actionable in practice. 

---
# WorldPhy: Benchmarking Fine-Grained Physical Plausibility of World Models in Robotics 

**Authors**: Ziyuan Chu, Mingzhi Zheng, Kaiyue Zhang, Jinan Zhang, Linchao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2611.34130)  

**Abstract**: World models are increasingly used in robotics to improve the performance of embodied models. Compared with general-purpose video generation, robotics places more stringent requirements on world models. Generated videos must satisfy fine-grained physical constraints on robot motion and object interactions. However, existing world model benchmarks can overlook robotics specific physical details and struggle to assess them accurately. To address this problem, we introduce WorldPhy, a world model benchmark for accurate, fine-grained evaluation of physical plausibility in robotics. Our benchmark contains 300 tasks, each consisting of an initial-frame image and a task instruction, covering diverse robotics scenes, task categories, viewpoints, and robot configurations. We further collect over 12K high-quality human annotations, and use them to train WorldPhyReward to accurately provide scores and reasons across six fine-grained physical dimensions and an instruction-following dimension. Comparisons with general VLMs and existing reward models show that WorldPhyReward better aligns with human ratings. We system- atically evaluate 13 frontier open and closed-source world models. The results reveal that current world models struggle to consistently maintain fine-grained physical plausibility across different robotics scenarios. Our code is available at this https URL WorldPhy. 

---
# FoundSurface: Feedforward Ray-Based 3D Scene Surface Reconstruction from Unposed Images 

**Authors**: Shenxing Wei, Jinxi Li, Zihui Zhang, Ajay Kumar, Wei Wang, Bo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2611.34128)  

**Abstract**: Accurate 3D surface reconstruction from unposed images remains a fundamental challenge in computer vision. Recent feed-forward pointmap methods eliminate the need for camera poses but inherently produce discrete, sparse representations. In this paper, we present a generalizable pipeline that directly recovers continuous, high-fidelity 3D surfaces from unposed images by seamlessly integrating pointmap- and ray-based representations. Our approach first extracts explicit geometric priors using a pre-trained pointmap model. A query-view-centric selection module then identifies geometrically relevant reference views via a robust surfel voting mechanism. Next, a cross-attention regressor bridges target query rays with reference features to estimate a continuous coarse surface. Lastly, a raylet-based module aggregates local 3D features to carve out high-frequency details and analytical normals. Trained as a single foundation model on 9 diverse datasets, our method demonstrates exceptional zero-shot generalization on 6 unseen datasets, clearly surpassing state-of-the-art baselines in geometric accuracy and detail preservation. 

---
# Out of Sight, Not Out of Clues: Can Video LLMs Guess from Partial Evidence? 

**Authors**: Luoxi Zhang, Jiawei Liu, Yanlin Luo, Peiyang He, Junlin Zhu, Chun To Choi, Yang Xiong, Jiang Wu, Xingyu Zeng, Min Yang, Shiwen Ni  

**Link**: [PDF](https://arxiv.org/pdf/2611.34126)  

**Abstract**: In real-world applications like egocentric recording, visual accessibility, and surveillance review, visual evidence for Video Large Language Models (Video LLMs) is often only partially captured. A reliable model should match its response to the evidence: answering when sufficient, guessing from visible clues when partial, and abstaining when absent. Yet existing benchmarks enforce a binary answerable-or-unanswerable split, leaving partial evidence unmeasured and obscuring whether failures arise from missing evidence or misjudging sufficiency. We introduce ClueProbe, a diagnostic benchmark to (1) separate perceiving evidence from judging sufficiency, and (2) validate our diagnosis via targeted fine-tuning. ClueProbe comprises 1,721 human-annotated questions across 598 natural videos, labeled Answerable, Guessable, or Unanswerable, alongside controlled perturbations altering wording or visual input. Evaluating 10 open-source and proprietary Video LLMs reveals that failures stem from judgment rather than perception: models perceive the evidence yet lack graded sufficiency awareness, answering or refusing outright while rarely guessing from partial clues. Targeted fine-tuning shows that this reflects missing supervision rather than insufficient capacity: supervising the guess level lifts appropriate guessing from 1.6% to 40.4% without extra false refusals, whereas conventional refusal tuning rejects 92% of guessable questions. Data and code will be released at this https URL. 

---
# MUSE: Multi-Task Uncertainty-Aware Self-Supervised Encoder for mmWave ISAC 

**Authors**: Nabeel Nisar Bhat, Jalees Nehvi, Rafael Berkvens, Jeroen Famaey  

**Link**: [PDF](https://arxiv.org/pdf/2611.34122)  

**Abstract**: Integrated sensing and communication (ISAC) allows a wireless system to use the same waveform for communication and environment sensing. At mmWave frequency bands, ISAC enables diverse human sensing tasks, such as localization, gesture recognition, and pose estimation, with high precision. However, each task requires training a dedicated model, typically from scratch on a task-specific dataset, with no mechanism to flag unreliable predictions. This hurts the scalability, transferability, and reliability of mmWave ISAC models. In this work, we propose MUSE (Multi-task Uncertainty-aware Self-supervised Encoder), a unified framework for mmWave human sensing. MUSE leverages a self-supervised transformer model with per-dataset tokenization and is trained via masked autoencoding across five datasets. During fine-tuning, MUSE uses evidential classification heads that enable uncertainty quantification. MUSE replaces N separate encoders with a single shared transformer encoder, an N-times backbone parameter reduction, while achieving a 9.7 percentage points improvement in classification performance compared to the task-specific model, showing that learning jointly across tasks improves per-task generalization. Across four classification tasks, MUSE produces reliable uncertainty estimates that enable selective prediction, improving accuracy by abstaining on the least confident 20% of predictions. We further improve the reliability of the model's confidence estimates using temperature scaling and conformal prediction. Finally, MUSE's uncertainty signal remains reliable even under cross-user deployment, assigning higher uncertainty to wrong predictions than to correct ones 78% of the time, and recovering 9.2 percentage points of accuracy through selective prediction, with no additional retraining on the unseen user. 

---
# Unstable Action Boundaries in Lethal Drone Agents 

**Authors**: Jacques Dumora, André-Louis Rochet, Simon Roburin  

**Link**: [PDF](https://arxiv.org/pdf/2611.34121)  

**Abstract**: Tool-using large language model (LLM) agents raise safety risks that are not captured by text-only refusal behavior, especially when tools can trigger high-impact actions. This concern is salient in military settings, where autonomous weapons raise the possibility of delegating force decisions to software agents. In this work, we study whether a drone agent's action boundary is stable under changes to the target descriptor. We evaluate recent and popular LLMs in a simulated perception-action pipeline over real-world drone imagery, where models can localize requested targets and invoke a simulated lethal tool. We find two concerning failure modes. First, some LLM agents invoke the lethal tool for ordinary generic descriptors such as \emph{people}, requiring no jailbreak at all. Second, an agent that initially appears conservative under protected descriptors can become unsafe under a simple semantic target shift. 

---
# SysBio-SR: Benchmarking Symbolic Regression for Trajectory Fidelity and Mechanism Recovery in Systems Biology 

**Authors**: Heng Rao, Dan Gao, Benjamin J Hoffman, Yu Gu, Zhenghao Liu, Ge Yu, Yang Cao, Minghan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.34106)  

**Abstract**: Symbolic regression seeks to discover interpretable governing equations for dynamical systems from data. In systems biology, however, accurately reproducing system trajectories does not necessarily imply recovery of the underlying reaction mechanisms. Existing benchmarks evaluate numerical accuracy, expression complexity, or equation-level similarity, but not agreement between recovered reactions and reference mechanisms in biological systems. We introduce SysBio-SR, a benchmark of 198 literature-based biological ODE systems with reaction annotations spanning mass action, Michaelis--Menten, Hill, and complex kinetics. Trajectory fidelity is evaluated across all systems under in-distribution reconstruction and out-of-distribution (OOD) extrapolation, while mechanism recovery is assessed on the subset of 161 systems. We evaluate trajectory fidelity across 11 symbolic-discovery methods and assess mechanism recovery for three reaction-based methods. Joint analysis across systems highlights how severely trajectory-based evaluation overestimates true mechanism recovery. Motivated by these findings, we develop KineSelect, a framework that searches candidate reaction topologies under biologically grounded kinetic laws to select optimal mechanisms that balance trajectory fidelity with structural parsimony. Compared with the strongest baseline, KineSelect improves aggregate OOD extrapolation \(R^2\) by 19.2\% and overall reaction F1 score by 14.3\%. 

---
# Do Machine Learning Ensembles Beat Published Quotas? Target Parameterization, Censored Baselines, and Conformal Intervals Under Admissions Reform 

**Authors**: Dat Tat Mai, James Jin Kang  

**Link**: [PDF](https://arxiv.org/pdf/2611.34104)  

**Abstract**: Public admission capacity caps published prior to matriculation are widely used by ministries and institutional analysts as default intake projections in centralized postsecondary clearinghouses. Using an audited 2022 to 2026 Vietnamese undergraduate panel (26,895 program-year records across 309 universities) linked to strictly lagged t-1 occupational vacancy signals, this paper evaluates machine learning, multilevel, and censored econometric estimators against published quotas across 5,385 continuing 2024 programs (five-fold university-grouped cross-validation) and 3,348 matched (2,179 grain-consistent) 2025 programs strictly out of time. Because 62.8% of 2024 programs enroll their exact published quota, unanchored log-growth ensembles degrade accuracy, while the worst 5% generate half of total quota error via unfilled "phantom capacity." Controlled target ablations, including a single-factor test under widened non-binding clipping, show that parameterizing the log fill ratio ln(D_it / S_it) instead of log growth reduces 2024 MAE by 21.5% (14.86 to 11.67), aligns ML with Tobit, Cragg, REML HLM, and capped L1 baselines, and lifts combined tuition and institutional-tier TreeSHAP attribution from 2.5% to 45.8%. Hurdle stacking attains the lowest full-census 2024 MAE (H-Hard 11.29; H-Soft 11.32 vs. 12.35, p = 0.022, clearing 10% not 5% FDR; fold-overlap p = 0.134), whereas continuous QA-Stack outperforms H-Soft on mixed-reporting campuses, Tobit (median) ties H-Soft across both 90% MCS sets, and interior-only Cragg-ML lowers RMSE to 28.1. Under Vietnam's 2025 unified-clearing reform, exact filling falls to 10.4%, H-Soft and capped L1 tie quotas (p = 0.979), and 90% cluster-weighted conformal coverage drops from 89.3% to 74.6%. Across 257 reporting campuses, reconciled 2026 national totals (627,114 to 629,256) lie squarely within 604,533 to 662,044. 

---
# AlignPre: Skill-Aligned Post-Training for Latent-Free Generative Precipitation Nowcasting 

**Authors**: Demin Yu, Kenghong Lin, Xutao Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.34101)  

**Abstract**: Precipitation nowcasting requires accurate short-term predictions under rapidly evolving spatiotemporal dynamics. Generative diffusion models can produce detailed and competitive radar forecasts by modeling conditional distributions. However, most existing approaches rely on separately trained components, such as deterministic branches or latent compression modules. More fundamentally, distribution matching alone fails to explicitly optimize operational forecast skill. We propose AlignPre, a streamlined pixel-space nowcasting framework that bridges this gap via skill-aligned post-training (SAPT). AlignPre employs an end-to-end flow-matching forecaster with a hierarchical spatiotemporal architecture for distribution matching and direct radar-field synthesis, without latent compression. During post-training, SAPT exploits the analytic structure of forecast verification scores to construct a differentiable objective. By backpropagating gradients of this objective from radar fields to the generator, SAPT turns forecast verification into direct training feedback. Across three real-world benchmarks spanning different forecast horizons, AlignPre consistently surpasses state-of-the-art deterministic and generative baselines in forecast skill, demonstrating that aligning generative modeling with domain-specific verification provides an effective paradigm for meteorological forecasting. 

---
# Will AI Replace Juniors? How Task Complexity Reduces Automation Risk 

**Authors**: Vincent Rost, Fabian Stephany  

**Link**: [PDF](https://arxiv.org/pdf/2611.34099)  

**Abstract**: How does task complexity shape firms' willingness to substitute entry-level labour with artificial intelligence (AI) tools? We study this question using a pre-registered survey experiment with managers from 1,435 German firms in the information economy and manufacturing industry. Managers are randomly assigned to consider a long lasting need for additional capacity involving either well-structured and standardised tasks, or more complex tasks that require professional judgement, contextual knowledge and independent problem-solving. Relative to standardised tasks, complex tasks reduce willingness to use AI rather than hire an entry-level employee by 0.54 points on a five-point scale and increase the importance attached to human oversight by 0.15 points. Further, pre-treatment responses indicate that managers expect the availability of AI applications to reduce hiring of both entry-level employees and experienced workers, but significantly more strongly for entry-level workers. 

---
# Whose Side Are You On? A Framework and Benchmark for Assessing LLM Political Stances through Politicians Social Media Posts 

**Authors**: Mohamed Osman, Mucahid Kutlu, Tamer Elsayed, Saban Kardas  

**Link**: [PDF](https://arxiv.org/pdf/2611.34098)  

**Abstract**: Large language models (LLMs) increasingly participate in political discussions, making it important to understand which positions they support or oppose. We propose a framework for assessing LLMs' political stances through simulated social media interactions. The framework uses politicians' posts and multiple prompting methods to identify consistently expressed stances. Using this framework, we construct WorldPOLL, comprising 4,600 political posts from X/Twitter collected from approximately 200 politicians representing ruling and opposition blocs across 23 countries. Our quantitative analysis shows how support and opposition vary across models, countries, and political directions, while our qualitative analysis identifies the political views and policies models support or oppose. Together, these analyses reveal shared positions across models and differences across national contexts. We also examine whether models express consistent stances across prompting methods, investigate the reasons for disagreement, and assess the effects of model size and instruction tuning in stance inconsistency. WorldPOLL offers a practical way to measure LLM political tendencies, identify the views behind them, and assess their consistency in real-world political discussions. 

---
# WorldBridge: Bridging Scattered Views for Coherent 3D World Generation 

**Authors**: Weilong Yan, Jingyu Hu, Hongli Xu, Yifan Zhan, Beibei Lin, Anbang Wang, Qinqian Lei, Jingwei Huang, Robby T. Tan  

**Link**: [PDF](https://arxiv.org/pdf/2611.34079)  

**Abstract**: Generating coherent and explorable 3D worlds from scattered images with little or no visual overlap is a fundamental and long-standing challenge in computer vision, which existing geometry-aware methods fail to address. Meanwhile, existing image- and video-based methods may generate unseen regions but produce inconsistent geometry across views. To bridge this gap, we present WorldBridge, a framework for coherent and controllable 3D world generation under scattered images with little or no visual overlap. WorldBridge encodes scattered references into a global scene context to guide geometry and appearance generation across views. Our temporal and spatial reference attention mechanism further enhances the fidelity of temporal references along the target trajectory and supports flexible spatial references outside it. A plug-and-play latent-reference consistency strategy improves reference fidelity through reprojection-guided test-time optimization. We also introduce the WorldBridge benchmark suite to evaluate 3D world generation from scattered, low-overlap references, which is rarely explored by current benchmarks. Both qualitative and quantitative results demonstrate that our method enables controllable and coherent 3D world generation, surpassing the existing state-of-the-art solutions. Our project page is available at this https URL. 

---
# Hijacking 3D Generation: Targeted Adversarial Attacks against Image-to-3D Models via Prefix Trajectory Alignment 

**Authors**: Yilong Yang, Jiale Teng, Yule Liu, Yang Liu, Xinlei He, Zhuo Ma  

**Link**: [PDF](https://arxiv.org/pdf/2611.34075)  

**Abstract**: Online services that deploy Image-to-3D models enable users to generate high-quality 3D assets from a single image. However, the generated 3D geometry can violate content policies and cause broader security risks. To prevent such content, these services typically use policy-restricted inspectors to examine uploaded images before generation. Since the inspector examines only the visible content of the input, a visually benign image can still induce the generation of policy-restricted 3D geometry. However, turning this vulnerability into an effective attack is non-trivial in the black-box setting: global backpropagation through the long DiT generation trajectory is difficult to fit, while different image preprocessing pipelines across services can substantially reduce transferability. In this paper, we present PTA3D, a targeted adversarial attack against Image-to-3D models that addresses these challenges through Prefix Trajectory Alignment (PTA) and Preprocessing Aware Optimization (PAO). PTA progressively aligns the prefix DiT outputs of the adversarial trajectory with those of the target trajectory, where the core geometric structure is formed, allowing the final geometry to be controlled without optimizing through the complete generation trajectory. PAO exposes adversarial optimization to diverse image transformations rather than a single preprocessing pipeline, reducing sensitivity to the mismatch between surrogate and target services. We establish a prefix-to-output bound and a preprocessing-aware transfer bound to theoretically characterize the effectiveness of the two designs. Experiments on three open-source Image-to-3D models and three online services show that PTA3D achieves 41.5%-88.5% ASR on online services and 90.8%-94.6% ASR in white box settings, substantially outperforming existing attacks. 

---
# Post-Training Policy Replication for Strategy-Following Multi-Agent Deep Reinforcement Learning across Team Sizes 

**Authors**: Yamato Takahagi, Yoshinari Motokawa, Toshiharu Sugawara  

**Link**: [PDF](https://arxiv.org/pdf/2611.34035)  

**Abstract**: Although multi-agent deep reinforcement learning enables coordinated behavior, aligning it with managerial intent remains difficult due to complex agent interactions. Although the strategy-following distributed attentional actor architecture (sfDA7-X) improves controllability via regional instructions, instruction combinations scale exponentially with team and instruction sizes, leading to prohibitive training costs and poor transferability. To reduce this computational cost, we introduce a team-size-independent observation architecture and reconstruct collective behavior by replicating a single trained policy across agents in teams of different sizes, thereby eliminating the need to train from scratch. Additionally, we evaluate warm-start retraining by initializing policy networks with a pre-trained policy to significantly lower computational overhead. In an object collection game, a policy trained in a four-agent team is deployed to teams of three to eight agents. Experiments demonstrate that training with the proposed aggregated observation matrix achieves performance comparable to that of the baseline method, sfDA7-X. Furthermore, policy replication preserves performance without substantial degradation in three- and five-agent teams, and warm-start retraining outperforms training from scratch while reducing required training episodes by up to 90%. Deployments to six- and eight-agent teams further confirm effective spatial region allocation following given instructions. These results demonstrate that the proposed method enables zero-shot collective behavior reconstruction via policy replication. 

---
# Counter-DECEIVR: Counterfactual Explanations for Deepfake Detectors using Detector-Guided Rectified Flow Inversion 

**Authors**: Mariano V. Ntrougkas, Vasileios Mezaris, Ioannis Patras  

**Link**: [PDF](https://arxiv.org/pdf/2611.34032)  

**Abstract**: Deep-learning-generated media are increasingly passed off as real, driving an arms race between image generators and deepfake detectors. Trust in the latter is limited by their often black-box nature; and, while region-based explanations can sometimes be extracted, these become less useful as generation quality improves. Counterfactual examples, which show the minimal change that flips a detector's decision, reveal what evidence drives a verdict, helping analysts decide whether to trust it. This work introduces Counter-DECEIVR (DEtector-guided reCtifiEd flow InVeRsion for counterfactuals), a novel training-free framework that generates counterfactual examples. Counter-DECEIVR steers latents in the velocity field of a rectified flow model using gradients from off-the-shelf deepfake detectors, without training a generator or adapting the detector; unlike prior methods, it requires no per-instance or per-detector fitting and no diffusion training on the target data. It also leverages the detector's explanation maps to concentrate editing to salient regions, turning low-level feature importance into localised counterfactual edits. A single parameter controls the strength of the edits. Across two deepfake detectors and two widely used forensics datasets, Counter-DECEIVR outperforms prior counterfactual generation methods, including training-based ones. Code will be made available upon acceptance. 

---
# Incorporating the Knowledge Distillation to Improve the EfficientNet Transfer Learning Capability 

**Authors**: Tjeng Wawan Cenggoro  

**Link**: [PDF](https://arxiv.org/pdf/2611.34025)  

**Abstract**: Due to the Deep Learning requirement for a large training dataset, Transfer Learning has become a central method in the field of Computer Vision, which heavily used Deep Learning. Since the adoption of transfer learning in the field, the performance of the models in computer vision is significantly upgraded. The common transfer learning practice in computer vision research is to use state-of-the-art architectures in the ImageNet dataset as the backbone network. The best performing architecture in ImageNet is currently held by EfficientNet, which is a set of architectures that were progressively grown from a baseline architecture.
By observing the EfficientNet development approach, it is natural to hypothesize that the smaller EfficientNet architecture is capable to contain the knowledge of the larger architecture. This hypothesis comes from the fact that the larger EfficientNet architecture was grown from the smaller architecture. Therefore, in this study, we experimented if it is beneficial to transfer the knowledge from the larger architecture to the smaller architecture of the EfficientNet. To achieve this goal, we proposed a transfer learning method that uses Knowledge Distillation as the knowledge transfer mechanism from the larger to smaller architecture. We found that the proposed method is able to upgrade the performance of each of the EfficientNet architecture. Several smaller architectures are even able to outperform the larger architecture that was trained using the standard transfer learning method. 

---
# Assessing the Physical Interpretability of AlphaEarth Foundation Embeddings Through Multi-Sensor Satellite Time-Series Benchmarking for Agricultural Mapping 

**Authors**: El Houcine El Moussaoui, Chouaib El Hachimi, Saïd Khabba, Salwa Belaqziz, Bouchra Ait Hssaine, Abderrahman Lahrouni, Salah Er-Raki, Abdelghani Chehbouni  

**Link**: [PDF](https://arxiv.org/pdf/2611.34019)  

**Abstract**: Accurate and dynamic maps of agricultural LULC are important for agricultural water resource management, particularly in areas where water scarcity is prevalent, such as Morocco. Here, we evaluate the utility of AlphaEarth Foundation Model (AEF) embeddings for agricultural LULC and irrigation-type mapping in Morocco. AEF embeddings were compared to single-date and time-series remote-sensing features using three machine-learning classifiers: XGBoost, CatBoost, and TabPFN. We also evaluated whether AEF embeddings can: (a) provide an efficient and transferable alternative to conventional direct use of Earth observation products and (b) reconstruct biophysical information. Results showed that multi-temporal feature time series achieved the highest performance, with accuracies of 0.93 for agricultural LULC and 0.87 for irrigation-type mapping when using TabPFN. However, AEF embeddings yielded comparable results, achieving accuracies of 0.91 for agricultural LULC and 0.85 for irrigation-type mapping, while reducing the need for extensive data processing and manual feature engineering typically required for satellite products. Spatial LULC mapping using TabPFN achieved comparable overall accuracies (0.85-0.86) across the three feature configurations, but revealed substantial differences in mapped tree cover, ranging from 14.57% for surface reflectance to 29.79% for AFE. Physical probing revealed that AEF embeddings strongly encode spectral indices (A30-SWIR: rho=0.58-0.67), (A38-NDMI: rho=-0.72), vegetation (A49-NDVI: rho = 0.72-0.73), and SAR structure (A25-VH: rho = 0.71). Future research should focus on enhancing the interpretability of embedding-based approaches by incorporating physical knowledge to develop more transparent and trustworthy AI systems for agricultural applications. 

---
# Evaluating Ukrainian Grammatical Competence in Language Models: Probability Versus Prompting 

**Authors**: Dmytro Babarytskyi, Oleksandr Babarytskyi, Vadym Savanevych  

**Link**: [PDF](https://arxiv.org/pdf/2611.34016)  

**Abstract**: Organisations deploying large language models in Ukrainian-language information systems need to know whether a model produces normative Ukrainian. This paper establishes a protocol for measuring that reliably and applies it to 19 models. The instrument is a released set of 1,444 Ukrainian minimal pairs covering the vocative case, numeral-noun and adjective agreement, verb government and calques from russian, with subject-verb agreement as a control; a quarter of the set was checked against the answer key in two independent native-speaker ratings, which agreed at Cohen's $\kappa$ = 0.963. Every pair is scored twice: by comparing the probabilities a model assigns to the two sentences, and by asking the model directly, with the sentences presented in both orders so that answer-position effects are measured instead of being absorbed into the score. Three results follow. Order consistency (the share of pairs answered identically when the sentences are exchanged) falls below 0.05 for ten of the nineteen models, and what separates the usable models is instruction tuning, not scale. The vocative case is systematically inverted: 0.153 accuracy against 0.951 on the control, with models preferring the non-normative nominative form of address. Ukrainian adaptation substantially reduces this deficit, measured against the exact checkpoints two adapted models were trained from. 

---
# Residual Cross-Model Narrative Convergence under Shared Prompts: Auditing Prompt Controls and Narrative Measurement 

**Authors**: Mehdy Sedaghat Payam  

**Link**: [PDF](https://arxiv.org/pdf/2611.34015)  

**Abstract**: Stories created by LLMs based on the same prompts can be narratively similar to one another. This similarity may reflect constraints imposed by the prompts as well as the models' writing tendencies. The current research analyzes 51,240 stories created by five models in the StoryScope corpus based on 10,248 common prompts. It uses 213 narrative features in seven dimensions to compare cross-model distances between stories generated from the same prompts with distances between stories generated from different but semantically matched prompts. To assess the relationship between narrative features and prompt content, a semantic relevance mask was evaluated against 420 author-annotated prompt-feature judgments (F1 = .813; near-threshold accuracy = .621). The equally weighted mean Residual Convergence Index (RCI) was .409 without masking and .406 with masking. These findings indicate measurable cross-model narrative convergence relative to a semantically matched-prompt baseline, with little change following semantic relevance masking. 

---
# CARROT: Competitive Adaptive Graph Clustering via Semi-Relaxed Optimal Transport 

**Authors**: Ching Fan Chiang, Jia-Yu Pan, Wen-Chieh Lin  

**Link**: [PDF](https://arxiv.org/pdf/2611.34012)  

**Abstract**: Deep graph clustering (DGC) has emerged as a prominent paradigm for unsupervised attributed graph clustering. However, existing methods often struggle with two practical challenges encountered in real-world applications: the number of clusters $K$ is unknown a priori, and cluster sizes are frequently highly imbalanced. Methods relying on a pre-specified $K$ may fail to recover the underlying clustering structure when $K$ is misspecified, while many approaches implicitly favor balanced partitions, hindering the discovery of minority clusters. We propose CARROT, a Competitive Adaptive gRaph clustering framework via semi-Relaxed Optimal Transport, which introduces semi-relaxed optimal transport into deep graph clustering to jointly address both challenges. CARROT first pre-trains node embeddings to establish a meaningful embedding space, and then jointly optimizes embeddings and cluster assignments through an Adaptive Clustering Mechanism. The mechanism combines semi-relaxed optimal transport to adapt cluster capacities to the underlying data, a competition loss to concentrate mass onto well-supported cluster prototypes, thereby starving degenerate ones, and hard pruning to progressively eliminate them. Extensive experiments on five public benchmarks demonstrate that CARROT achieves state-of-the-art NMI and ARI, accurately recovers the number of clusters without supervision, and faithfully preserves size-imbalanced community structures. 

---
# DosePET: An Open-Source Dataset with 2100 Radiation Dose Maps for PET/CT 

**Authors**: Eleftherios Tzanis, Antonios Tzortzakakis, Michail E. Klontzas  

**Link**: [PDF](https://arxiv.org/pdf/2611.34008)  

**Abstract**: Large collections of three-dimensional radiation dose maps linked to patient anatomy are needed to develop artificial intelligence (AI) models for personalized internal dosimetry in nuclear medicine, but access to such data remains limited. Here we present DosePET, an open-source dataset of 2,100 patient-specific absorbed-dose maps generated by Monte Carlo simulation of publicly available PET/CT examinations: 1,503 [18F]FDG examinations from this http URL and 369 [18F]PSMA-1007 and 228 [68Ga]Ga-PSMA-11 examinations from PSMA-PET-CT-Lesions, acquired at five institutions with seven scanner models. For each examination, the CT was resampled to the native PET grid to build a voxelized patient phantom, the quantitative PET image defined the radionuclide source, and radiation transport was simulated with GATE version 9.4. Organ and tissue masks were generated from the CT with TotalSegmentator. Each study folder contains a NIfTI dose map integrated from scan time to infinity, 117 binary organ/tissue masks, a report of regional dosimetric quantities, including mean absorbed doses, dose conversion factors and scan-time dose rates, and simulation metadata. The total processing time was 1,542.4 hours. Python tools support selective downloading and conversion to scan-time dose-rate or dose-per-decay maps, and an agent skill guides pairing with the source PET/CT images. DosePET is publicly available at this https URL. 

---
# Aegis-VLA: Reliability-Gated Action Refinement via a Trajectory World Model for Dynamic Object Manipulation 

**Authors**: Lennox Wei, Jiaqi Yao, Shixin Sun, Yuanzheng Zhang, Jianhua Han, Meng Tian, Shaohui Liu, Wangmeng Zuo  

**Link**: [PDF](https://arxiv.org/pdf/2611.33993)  

**Abstract**: Dynamic object manipulation requires maintaining spatial and temporal alignment with targets that continue to move during inference and execution. Although future prediction can inform action correction, an accurate forecast does not establish whether modifying an existing action would be beneficial. We present Aegis-VLA, a world-model-guided vision-language-action (VLA) framework that generates predictive action corrections and separately determines whether and how strongly to apply them. A Trajectory World Model combines kinematic extrapolation with learned residual dynamics to predict object motion and estimate positional uncertainty. Discrepancy-Guided Action Refinement uses differences between history-only and base-action-conditioned end-effector forecasts to propose bounded arm corrections. A Reliability Gate learns when to admit corrections based on demonstration-relative improvements over base actions, and regulates their application during inference using predictive uncertainty and anticipated object--end-effector proximity. Rejected corrections preserve the corresponding base actions. Aegis-VLA introduces only approximately 2.45M trainable parameters. On the simulated Dynamic Object Manipulation (DOM) benchmark, Aegis-VLA achieves 56.72\% success over 1,800 evaluation trials, exceeding our reproduced DynamicVLA baseline of 45.44\% by 11.28 percentage points. 

---
# MolGPAO: Molecular Generative Pre-trained Autoregressive Octree with Speculative Decoding and Sub-Millisecond Seam Healing for 3D Molecules 

**Authors**: Shangqian Chen, Guojiang Zhao, Zheng Cheng, Jiankun Wang, Hongshuai Wang, Zhifeng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2611.33985)  

**Abstract**: Autoregressive generation over adaptive 3D octrees (Uni-3DAR) offers an intrinsic $O(N \log N)$ sparsity advantage over dense $O(N^2)$ equivariant diffusion and flow models on large drug-like molecules, yet lags in geometric equilibrium, closed-shell valence stability, and property fidelity due to Morton-seam coordinate jumps, leaf-heavy compute with root-count trapping, and target-shortcut collapse in internal critics. We introduce MolGPAO (Molecular Generative Pre-trained Autoregressive Octree), a standalone 3D octree architecture and single-output speculative tree-decoding engine using zero external surrogates, zero reinforcement learning, and zero inference-time MLIP relaxation. MolGPAO unifies: (1) non-trapping 6-D ScaleFreeBudgetEmbedding with L5-to-L1 covalent packing priors; (2) a FlashAttention-2 pause-and-gather speculative octree funnel ($m_{s1} \to m_{s2} \to 1$, $M_{\mathrm{out}} = 1$); (3) an anti-shortcut ground-truth TwoStageOctreeProcessCritic with 64-RBF covalent edge gating and $SO(3)$-invariant 12-channel multipole norms (blind QM9 $r = 0.9867$--$0.9989$); (4) a $<0.5$ ms 4-layer $SE(3)$ PostTreeGraphRefiner for Morton-seam healing, expected-XYZ bonded supervision, selective CFG, and zero-heavy-displacement valence reconciliation; and (5) bubble-free stride-2 GPU leaf decoding. Across GEOM-Drugs (0.49 GPU-h), disjoint 50k/50k QM9 (0.43 GPU-h), and PBE0/def2-TZVP-verified OE62 (0.56 GPU-h), MolGPAO lifts drug-scale valid-connected yield 6.78x (12.7% to 86.2%, 99.83% atom stability), runs 7.3x faster (or 1.65x faster at 2.0x yield) than 100-step FlowMol at $N=100$, beats 156 GPU-h PropMolFlow on QM9 HOMO (-12.7%) and dipole (-21.0%) at 362x lower compute, and doubles live DFT hit rate on OE62 (33.3% vs. 16.7%) at 92.8x lower compute. 

---
# Kolpona: An Agentic Framework with Gradient-Free Neuron Programming for RTL Generation 

**Authors**: Musa Tur Farazi, Md Nasir Uddin, Soumik Bhattacharjee, Tishna Sabrina, Chowdhury Mohammad Rakin Haider  

**Link**: [PDF](https://arxiv.org/pdf/2611.33973)  

**Abstract**: Open-weight large language models (LLMs) for register-transfer level (RTL) design are usually improved by fine-tuning on curated Verilog corpora, which needs large sets of verified designs and GPU training. We present Kolpona, an agentic framework that improves such models without gradient training. Its first component is neuron programming, a new form of weight-space programming (WSP) that writes single neurons in closed form into the weights to remove failure habits found on practice tasks, and keeps each edit only when testbench results on fresh tasks favour it. At inference time, tool-using agents propose designs, and a label-free selection stage delivers one design per task from the specification, the candidates and their simulations alone. On Qwen3.5-4B, edits to 16 neurons raise the VerilogEval v2 pass@1 of the model alone from 31.7% to 55.4%, and the full framework reaches 77.6%. On Qwen3.8-27B, Kolpona lifts pass@1 from 73.1% to 94.9% on VerilogEval v2 and from 66.0% to 92.0% on RTLLM v2.1 on a single 32 GB GPU, above every published system in our comparison. The RTL edits also carry over to Python and raise the pass@1 of the 4B model alone from 84.8% to 91.5% on HumanEval+ and from 67.7% to 77.5% on MBPP+. 

---
# Fine-Tuning Without Gradients? Closed-Form, LLM-Written Classifiers on a Frozen Encoder 

**Authors**: Md Nasir Uddin, Musa Tur Farazi, Soumik Bhattacharjee, Tishna Sabrina, Chowdhury Mohammad Rakin Haider  

**Link**: [PDF](https://arxiv.org/pdf/2611.33972)  

**Abstract**: Can a classifier computed in closed form on a frozen encoder match gradient fine-tuning? We present closed-form weight-space programming (WSP): a large language model (LLM), used as a design assistant, writes programs that read a frozen masked language model through cloze-style prompt templates at several layers, fit closed-form heads (shrinkage discriminant analysis, kernel ridge regression and the encoder's label-word prior) on labelled data, and select among a pool of candidate heads by cross-fitting. No gradient is computed and no encoder weight changes. We instantiate the framework on RoBERTa-large. On 11 development datasets with 512 to 8,192 training labels, WSP is ahead of full fine-tuning (FT) and LoRA at every label budget (by 0.3 to 0.6 points on average; the registered non-inferiority tests pass) and within 0.5 points of prompt-based fine-tuning (PFT); these datasets also shaped the recipe. With fewer labels its lead over FT and LoRA grows: 1.2 to 3.5 points with 64 and 128 labels, and 3.8 to 5.3 points in the few-shot setting of 16 labels per class, where WSP is again within half a point of PFT on the development datasets. On six held-out datasets, WSP stays ahead of LoRA and within half a point of FT on average, although with six datasets no registered test passes, and PFT is ahead, by 1.30 points with 8,192 labels. WSP never collapses and is deterministic, but it needs one encoder pass per template (17 here). We report all eight recipe versions, every decision that saw test results, and 27 ideas that did not help. 

---
# Embodied-VGGT: Preserving Reliable Local Geometry for Long-Sequence Reconstruction 

**Authors**: Yiheng Han, Ziang Chen, Zhuyong Fu, Yiqing Yin, Nan Ma  

**Link**: [PDF](https://arxiv.org/pdf/2611.33964)  

**Abstract**: Long sequence reconstruction must correct accumulated drift while preserving locally consistent geometry. In robotics and autonomous driving, embodied acquisition provides recurring platform- environment geometry and repeated observations of scene structure and platform motion. We present Embodied-VGGT, which exploits these cues to establish reliable local relationships and selectively preserve them during global refinement. Memory-Augmented Grounded Chunks (MAGC) builds on Scal3R's global context memory and uses relative scale references derived from predicted ground geometry to regularize test-time memory updates, encouraging cross-chunk consistency while limiting changes to reliable local geometry. Superpoint Alignment (SA) and Cross-Loop Optimization (CL) jointly address the robust establishment and preservation of geometric relationships. SA balances ground and non-ground evidence across spatial regions to estimate inter-chunk translation. CL uses geometric and motion consistency across interleaved windows to select coherent chunk groups, each receiving a shared Sim(3) correction that preserves its internal relative transformations during global optimization. Experiments on five public driving benchmarks demonstrate consistent and substantial improvements in long sequence reconstruction. On KITTI Odometry, Embodied-VGGT achieves a mean absolute trajectory error of 7.30 m and a Chamfer distance of 2.67 m, reducing these errors by 46.8% and 23.3% compared with the state-of-the-art baseline under same evaluation protocol. To facilitate reproducibility, we released our code: this https URL. 

---
# AI Buckingham: Agentic Discovery of Dimensionless Scaling Laws Across Physical Regimes 

**Authors**: Ao Cheng, Lei Zhang, Guowei He  

**Link**: [PDF](https://arxiv.org/pdf/2611.33961)  

**Abstract**: Classical dimensional analysis defines admissible dimensionless representations but does not determine which combinations capture the dominant physical dependence, whether it changes across physical regimes, or when dependent-variable rescaling is required. We present AI Buckingham, an agentic framework for autonomous dimensionless discovery from natural-language descriptions of physical problems and observational data. It couples reasoning and planning by a large language model (LLM) with predefined symbolic and numerical tools. Literature-based knowledge retrieval supports dimensional checking, gradient-based clustering identifies physical regimes, and mutual information provides an objective, reparameterization-invariant criterion for identifying dominant dimensionless groups. Explicit numerical criteria determine the group count and assess simplification. Across ten physical systems spanning fluid dynamics, magnetohydrodynamics, thermal convection and additive manufacturing, AI Buckingham recovers established similarity representations and resolves regime-dependent scalings. For the longstanding challenge of predicting turbulent drag on rough surfaces, it identifies informative reparameterizations of surface statistics that improve equivalent sand-grain roughness estimates for engineering-oriented drag modelling. 

---
# RefMix: Expanding Endpoint Proposals for Diversity-Preserving Reward Fine-Tuning 

**Authors**: Jaemin Kim, Christian Simon, Dongseok Shim, Wei-Yao Wang, Qiyu Wu, Masato Ishii, Shusuke Takahashi, Jong Chul Ye, Yuki Mitsufuji  

**Link**: [PDF](https://arxiv.org/pdf/2611.33957)  

**Abstract**: Online forward matching aligns flow-based generative models by repeatedly generating, rewarding, and fitting model-produced endpoints. However, this behavior creates an on-policy feedback loop: as high-reward modes become increasingly favored, alternative modes become less likely to appear in subsequent training pools, progressively reducing visual diversity. We introduce RefMix, a reference-assisted endpoint proposal strategy that addresses this limitation by injecting alternative candidates into the training pool while retaining the original task reward and forward matching objective. In particular, RefMix uses a reference model during early high-noise steps to expose alternative visual modes, followed by the current reward-tuned model for endpoint completion. Across state-of-the-art online forward matching methods, RefMix consistently improves visual diversity while preserving task reward performance. Controlled inference experiments further demonstrate that training-time endpoint exposure is critical to improve reward--diversity trade-offs, beyond the impact of reference-assisted inference. By rethinking how training endpoints are proposed, RefMix achieves up to 44% higher diversity over forward matching baselines at comparable reward levels. 

---
# DuoMI: Dual Mutual Information Regularization for Expert Reuse in Looped Transformers 

**Authors**: Xinlai Kang, Zhenhe Wu, Haoyu Wang, Weihua He, Yaping Jin, Jie Hu, Hanting Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.33952)  

**Abstract**: We introduce Dual Mutual Information Regularization (DuoMI), a method for regulating expert routing across rounds in looped mixture-of-experts (MoE) Transformers. When two rounds of computation share an expert pool, their routers may favor different expert subsets, limiting the reuse of individual expert parameters across rounds. Even when aggregate expert usage is identical across rounds, we would still prefer each token to retain some experts while switching to others. DuoMI addresses both aspects of routing: it uses round-level marginal mutual information to limit the dependence of aggregate expert usage on the round, and token-level conditional mutual information to penalize average cross-round routing changes that fall outside a target interval. During training, DuoMI computes differentiable penalties from soft routing distributions over all experts. It adds no model parameters and leaves sparse Top-$k$ routing strategy unchanged at inference. On a looped OLMoE backbone in which paired adjacent layers share experts, DuoMI improves the average score across $15$ downstream tasks from $50.08\%$ to $51.82\%$, a gain of $1.74$ percentage points over the looped baseline. 

---
# InstructTTS-OPD: On-Policy Distillation for Multi-Objective Instruction-Following TTS 

**Authors**: Haitao Li, Tian Tan, Yuguang Yang, Guanrou Yang, Wenhao Guan, Chenglin Li, Shan Yang, Xie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.33937)  

**Abstract**: Instruction-following text-to-speech (Instruct-TTS) renders a target transcript in the manner described by a free-form natural-language instruction, and a good rendition must succeed on three fronts at once: following the instruction, pronouncing the transcript correctly, and sounding natural. Reinforcement learning has become the standard way to post-train such generative models toward human preferences, and TTS systems increasingly adopt it. When the three objectives are optimized together under a single scalar reward, typically with group relative policy optimization (GRPO), training tends to be unstable: the objectives compete, and the shared update drifts toward whichever one is easiest to improve, producing a seesaw among them. We instead adapt on-policy distillation (OPD), which fuses several single-objective teachers by matching their velocity fields on the student's own rollouts. Realizing this for Instruct-TTS first calls for a sound reward on each axis, which we analyze carefully: a distilled Qwen-Omni proxy for the costly Gemini judge, dialect-free data for the WER reward, and DNSMOS in place of the language-biased UTMOS. We then train three single-objective teachers with GRPO and fuse them with OPD, routing each rollout to the instruction-following or WER teacher by its data domain and letting every rollout additionally follow the MOS teacher for quality. This lets instruction following and intelligibility improve together without a trade-off, but the quality target conflicts with instruction following, so following it additively reintroduces the seesaw. We therefore propose conflict-aware velocity fusion, which projects the quality target off the direction that opposes the primary objective before fusing, and show that it markedly reduces the instruction-following/quality trade-off. The fused model improves over scalar-reward GRPO on all three objectives at once. 

---
# Towards 360$^\circ$ Embodied UAV Tracking Without Turning 

**Authors**: Bingyu Li, Wu Du, Zhiyuan Zhao, Junyu Gao, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.33934)  

**Abstract**: Conventional embodied UAV tracking couples target observability with heading control because targets may leave the field of view of forward-facing cameras. We investigate a complementary paradigm that uses full-spherical perception to follow designated targets without active heading adjustment. To support this setting, we introduce 360UAV-ETrack, a simulation benchmark comprising 2,652 sequences, over 1.73 million synchronized observations, eight tracking tasks, and 150 held-out closed-loop scenarios. We propose 360UAV-TVLA, a panoramic vision--language--action (VLA) framework integrating persistent visual referring (PVR), compressed temporal memory (CTM), and geometry-aligned trajectory supervision (GTS) for instance identification, motion reasoning, and three-dimensional control. Experiments across two VLA backbones show consistent gains over standard fine-tuning, with normalized average tracking frames increasing from 22.56\% to 35.28\% for $\pi_0$ and from 25.68\% to 33.64\% for $\pi_{0.5}$. Our benchmark further reveals a substantial gap between offline imitation accuracy and closed-loop tracking reliability, establishing a challenging testbed for future research this https URL 

---
# Connecting AI Agents to Live Discrete Element Simulations via the Model Context Protocol 

**Authors**: Yusong Han, Ryosuke Uzuoka, Kyohei Ueda  

**Link**: [PDF](https://arxiv.org/pdf/2611.33926)  

**Abstract**: This paper describes a Model Context Protocol (MCP) server through which a general-purpose AI agent operates a discrete element engine, commercial or open-source, while the simulation runs. The agent submits a long simulation to the engine and, while it computes, reads the running model, changes it, or stops it. The human modeler stays at the engine's console and graphical interface throughout, and the agent sees what the modeler does there. The tools are derived from what the agent needs to do at the engine and are held to four principles: correctness, coverage, orthogonality and simplicity. Two documentation tools let the agent search the engine's manual and browse it level by level. Five execution tools let the agent run code in the engine, submit long simulations in the background, check on them, read the running model, and stop them. A bridge inside the engine serves these tools in a commercial DEM engine, PFC, and an open-source one, YADE. In a case study, a general-purpose agent calibrated a drained triaxial model of dense Toyoura sand against laboratory data in both engines, while the modeler worked at the same engine; the agent noticed the modeler's actions at the console and acted on them. The same calibration on YADE was then given to seven language models, three sessions each and without human intervention. Twenty of the twenty-one sessions reached the targets, and nineteen of the parameter sets lie in one small region of the admissible ranges. The server and the bridge are released as two open-source packages, itasca-mcp, which also serves FLAC, 3DEC, MPoint and MassFlow, and yade-mcp. 

---
# Beyond the Happy Path: Generalizing Robot World Models from Simulated Failures 

**Authors**: Zhiyue Zhao, Mingyu Liu, Bolin Zhang, Liyang Li, Guangyu Huang, Canyu Zhao, Jinchuan Shi, Haokai Zhang, Hao Chen, Chunhua Shen  

**Link**: [PDF](https://arxiv.org/pdf/2611.33925)  

**Abstract**: Robot world models are increasingly used to evaluate and plan policies, yet robot training data are dominated by successful demonstrations. Off the happy path, when a grasp is misaligned or mistimed and the object is pushed, tilted, or slips, their predictions break down. Across five pretrained action-conditioned models, object-motion error is at most 10.4% lower than that of a static baseline that copies the first frame, and two models are worse than this baseline. We present WorldReforge, which teaches world models what happens when manipulation goes wrong. We construct the Reforge dataset of 62k simulated trajectories by perturbing reference grasps in position, orientation, and closure timing, recording each resulting push, slip, or failed pickup with ground-truth object pose, contact, and surface points. WorldReforge turns this ground truth into two training signals without additional inference cost. Interaction Alignment focuses reconstruction on the small object and contact regions that a global loss underweights, and Surface Correspondence trains backbone features to follow the same physical surface points across frames. On Cosmos3 Nano, WorldReforge reduces object trajectory error by 28.2% overall and by 35.0% on unseen objects, and it lowers the error in predicting how two actions' outcomes diverge by 43.3%. Although trained only in simulation, it also improves reconstruction on DROID and on real failure videos. Controlled data studies show that failed contact and successful holding teach complementary physics: failed contact teaches how objects respond, and holding teaches how a grasped state persists. Our code, data, and models are available at this http URL. 

---
# DiCALM: Converting Autoregressive Convolution-Attention Language Models into Discrete Diffusion Language Models 

**Authors**: Phong Le, Ngoc-Quan Pham, Dat Quoc Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2611.33919)  

**Abstract**: Existing methods for adapting autoregressive (AR) language models to diffusion generation focus mainly on full-attention architectures. More efficient hybrid architectures remain underexplored. We introduce DiCALM, a recipe for converting the hybrid convolution-attention models of the LFM2 family (LFM2 and LFM2.5) from AR to masked-diffusion generation. DiCALM progressively extends LFM2's causal gated convolution so that it captures local context in both directions. It combines this extension with a masking warm-up and joint AR--diffusion training. In controlled experiments, DiCALM performs comparably to the corresponding fine-tuned LFM2.5-based AR models and delivers consistent speedups over them at both the 350M and 1.2B scales: over 1.4x on average and up to 1.79x. 

---
# Correlation-Aware Evidential Fusion for Trusted Multi-View Classification 

**Authors**: Tianyang Wang, Mengchi Zuo, Cai Xu, Yilin Zhang, Ziyu Guan, Wei Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.33900)  

**Abstract**: Trusted multi-view classification aims to make accurate and reliable predictions by integrating information from multiple views. Existing methods typically estimate class evidence and uncertainty for each view and combine the resulting opinions through aggregation rules. However, they generally treat different views as independent information sources, overlooking the correlations among them. Highly correlated views tend to produce similar evidence, and treating them as independent sources may repeatedly count consistent information and give it disproportionate influence, while weakening complementary evidence that provides additional information. Consequently, such correlation-unaware fusion may not only limit classification performance but also produce miscalibrated uncertainty estimates. To address this issue, we propose Correlation-Aware Fusion of Evidence (CAFE), a framework for trusted multi-view classification. CAFE applies complementarity-based trust discounting to the evidence produced by view-private experts and introduces a correlation-aware constraint to reduce the repeated contribution of similar evidence. It then learns sample-specific weights from the global multi-view context to adaptively combine the correlation-adjusted private evidence with cross-view collaborative evidence. Extensive experiments on six real-world datasets demonstrate that CAFE consistently improves classification and calibration performance in general settings, with particularly significant gains when strong correlations exist among views. 

---
# When Are Parallel Subagents Effective for Coding? An Empirical Study on Repository Construction 

**Authors**: Zhiheng Lyu, Chengyu Yin, Lingming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.33893)  

**Abstract**: LLM-based coding agents are increasingly used for long-horizon tasks such as building an entire program from a specification. Products like Codex and Claude Code let an agent delegate work to parallel subagents, multiplying run cost, yet it remains unclear whether and when delegation produces a better program. We compare a single agent with a main agent delegating to subagents on repository-construction tasks from NL2Repo and ProgramBench (which also provides an executable reference program). We compare scores at equal time and equal cost, run intervention experiments, and analyze fine-grained behavior. On randomly drawn tasks, parallel subagents raise hidden-test pass rates by only about 1 to 5 percentage points at up to 3.8x the cost of a single agent. Their gain is larger on hard tasks, where we find evidence of three mechanisms. Self-checking: with GPT-5.6 Sol, one sentence telling a single agent to probe the reference program first recovers most of this gain at a fraction of the cost; from GPT-5.6 Sol to GPT-6.1 Sol, the gain shrinks as single agents check their own work more. Fresh context: in extended runs, a single agent that periodically starts a new session with an empty conversation makes faster early progress in every comparison than one that resumes its conversation. Modularity: multi-agent runs produce more modular code (more files and shorter functions), closer to the upstream source. We observe similar patterns in a smaller replication with Opus 5.5 on Claude Code. Overall, subagents help when (i) a single agent stops checking too early, (ii) a long session keeps editing with little improvement, or (iii) modular code helps on the task; otherwise, they mainly add cost. 

---
# Random Voronoi Networks 

**Authors**: Ray Verma, Pau Mateo-Bernardó, Daniel Mas Montserrat, David Bonet, Marçal Comajoan Cara, Alexander G. Ioannidis  

**Link**: [PDF](https://arxiv.org/pdf/2611.33875)  

**Abstract**: Nearest-neighbor methods are local, simple, and interpretable, but their non-parametric form makes them memory-intensive, expensive at inference, and difficult to integrate into differentiable neural architectures. We introduce Random Voronoi Networks (RVNs), a family of structured three-layer MLPs that express nearest-neighbor geometry through sampled Voronoi comparisons. The first layer computes Random Voronoi Features (RVF) from affine pairwise separators between selected training samples, the second layer aggregates these comparisons into Voronoi cells or top-k neighborhood scores, and the readout layer produces classification or regression predictions. We show that, when all samples and pairwise comparisons are included, step RVNs recover exact Euclidean k-NN, while ReLU RVNs provide a violation-score realization for 1-NN that can be relaxed into a differentiable soft weighting. Subsampling yields approximate RVNs with explicit control over the number of cells and features. We study several unsupervised sampling methods that trade off random coverage, directional diversity, coordinate sparsity, and empirical feature variance. Empirically, RVFs produce a highly effective data encoding strategy, providing performance boosts to downstream methods ranging from linear to modern tabular deep learning models. Finally, we showcase that RVF provides an interpretable feature map due to its k-NN based construction and ties to pairwise local directions. These results position our Random Voronoi formulation as a bridge between nearest-neighbor rules, random features, and modern tabular models. 

---
# Mitigating Overgeneralization in RND via Spectral Target Design 

**Authors**: Minseok Jeong, Yechan Lee, Hyewon Choi, Jeongyong Yang, SooJean Han  

**Link**: [PDF](https://arxiv.org/pdf/2611.33850)  

**Abstract**: Random Network Distillation (RND) is a scalable novelty signal for RL based on a fixed random target network and a trained predictor. However, it can overgeneralize: the predictor extrapolates the random target off the data manifold, causing novelty scores to collapse on unfamiliar inputs. This paper analyzes this failure through a kernel-theoretic lens. Under a GP target model and an NTK-regime predictor, we show that the RND novelty score is a cross-kernel residual variance: it is governed jointly by the target covariance kernel and the predictor NTK, rather than by either one alone. This identifies overgeneralization as spectral under-excitation. Motivated by this view, we formulate spectral target design as a principle for shaping RND novelty geometry. As a concrete instance, we replace the implicit random-network target with a bandwidth-controlled Random Fourier Feature (RFF) target, redistributing residual signal toward modes less attenuated by the predictor. Experiments on offline D4RL, online Atari, and Fetch manipulation benchmarks show improved novelty discrimination and downstream RL performance. The code for the RL implementation is available at this https URL. 

---
# When Rerankers Work Together: Better and Faster Reranking without Joint Training 

**Authors**: Yanqiao Chen, Dongsheng Hou, Yuhan Rui  

**Link**: [PDF](https://arxiv.org/pdf/2611.33844)  

**Abstract**: Standalone reranker quality does not determine its value in a multistage RAG pipeline. We systematically study heterogeneous rerankers composed without joint training, spanning lexical, single-vector, late-interaction, and joint query-context scoring. Two lightweight semantic coarse scorers, Cache-Pointwise and Cache-Shapley, learn from passage relevance labels and share cacheable representations and linear-time singleton inference. Under matched candidates and budgets, we compare cascades with stage-removal references to measure evidence retention, repair and breakage, reader accuracy, and online pruning cost. On NoLiMa 32K with Gemma, inserting these scorers before CrossEncoder or Provence improves answer accuracy by 4.02-9.43 percentage points while accelerating online pruning by 1.57-1.88x relative to the corresponding reranker alone. Cache-Pointwise composed with ColBERTv2 through reciprocal rank fusion improves accuracy by 4.60 points at 1.36x speedup over direct ColBERTv2. The stage-removal comparisons reveal an asymmetry: a cascade can outperform its final reranker alone while underperforming its coarse selector alone. Two readers, a second multi-hop dataset, and three-stage extensions show that the gains depend on the reader, reranker pairing, and stage configuration. These findings demonstrate a practical route to better and faster reranking without joint training and motivate studying stage order, candidate budgets, and rank fusion as determinants of composition quality. 

---
# FEAST: Fair and Efficient Adaptive Sampling for Task-Free On-Policy Learning 

**Authors**: Yongxi Zhu, Ziyi Wang, Yichen Wu, Ruite Lu, Liao Chang, Haipeng Li, Ying Wei  

**Link**: [PDF](https://arxiv.org/pdf/2611.33827)  

**Abstract**: Language agents trained from self-generated experience must allocate a limited rollout budget across heterogeneous capabilities. A common strategy is to prioritize capabilities with strong current learning signals, but in on-policy learning these signals are endogenous: past allocation changes the policy and therefore the evidence available for future allocation. This can create a feedback loop in which under-trained capabilities receive increasingly fewer opportunities to improve. We formalize this effect through the Quota-Efficiency Law (QEL), which shows that allocation history can change learning efficiency even under matched capability-specific exposure. Motivated by this, we introduce FEAST, a task-free adaptive sampling framework that separates opportunity allocation from compute allocation. FEAST dynamically discovers capability clusters while preserving their learning histories, maintains continued sampling opportunities across clusters, and reserves a rollout budget for each selected cluster. Within each budget, it progressively concentrates computation on prompts with informative reward variation. Across GURU-18K and SMC, FEAST improves average performance over competitive adaptive sampling baselines under matched rollout budgets, while better preserving learning progress on weaker capabilities. Ablation studies further support the complementary roles of cross-cluster opportunity protection and within-cluster selective allocation. Code will be released upon acceptance. 

---
# Pitfalls of Dependency-Ready Decoding for Masked Diffusion Language Models 

**Authors**: Lina Berrayana, Christopher Malon, Yoichi Ishibashi, Taichi Aida, Masafumi Oyamada  

**Link**: [PDF](https://arxiv.org/pdf/2611.33823)  

**Abstract**: Masked diffusion models (MDMs) have shown strong performance on structured reasoning tasks such as Sudoku, Trip Planning, and Countdown. MDMs introduce a freedom in the order in which tokens may be unmasked, and to exercise this freedom for good accuracy, early decoding strategies relied on signals of confidence or the downstream impact of decoding a token. However, simple examples seem to illustrate that resolving a token's dependencies before unmasking it should be important. Therefore, we examine whether this principle, which we call dependency-ready decoding, would contribute to better accuracy, from a practical and a theoretical perspective. We examine a recent decoder that implements dependency-readiness through attention, and generally find that when it performs better than confidence-based or future utility baselines, its decoding order collapses to left-to-right decoding. For theoretical analysis, we establish a definition of dependency without resorting to attention, measured by an information-theoretic score of how much revealing a single masked token (or a set) would shift the model's prediction for another. We prove that when the margin is large relative to this score, no such revelation can change the top prediction, so in this regime Maximum-Margin decoding already commits dependency-ready tokens. These findings leave only a narrow scope for future dependency-ready decoders: their potential benefit is limited to positions with margins smaller than the information-theoretic sensitivity, and reliance on attention should be questioned. 

---
# KnitNet: Learning to Search at Inference Time for Exact RNA Inverse Design 

**Authors**: Antonia Panescu, Xin Wang, Haoran Xu, Qingyu Chen, Carlos Oliver, Rex Ying  

**Link**: [PDF](https://arxiv.org/pdf/2611.33812)  

**Abstract**: RNA inverse design seeks nucleotide sequences that fold into specified target structures. Existing methods rely on learned generators that can miss exact structural matches, especially beyond their training distributions, or on costly search over a vast, multimodal sequence space. Search is further complicated by globally coupled nucleotide interactions, as correcting one pairing error can create new errors beyond a fixed repair scope. To address this, we introduce KnitNet, an inference-time scaling framework that learns from verifier feedback to refine sequences for exact fold-back matching. Starting from candidates generated for a target secondary structure, KnitNet uses mismatched folds to infer nucleotide dependencies and adapt its joint repair strategy, while an online surrogate learns to predict which edits are promising. This makes search progressively more informed and compute-efficient. Crucially, KnitNet generalizes across RNA structural regimes by changing only the optimization signal to match the verifier required by each setting. Across five secondary-structure benchmarks, KnitNet increases exact solve rates by an average of 21 percentage points over continued sampling with 70% less time-to-solve. On OpenKnot57 specifically, KnitNet is the first method to achieve exact computational matches for all complex, pseudoknotted structures. Beyond 2D, KnitNet improves 3D structural self-consistency over per-target RL by 7.8% while being 39.3% faster, demonstrating generalization and effective inference-time scaling across RNA design regimes. 

---
# QPAT: A Universal Quantum Machine Learning Framework for Quantum-Assisted Training and Classical Inference 

**Authors**: Yixiong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.33806)  

**Abstract**: Deep learning models often struggle with sharp local optima and overfitting in limited-data regimes, and exhibit limited capacity for capturing high-order, complex probability distributions. Quantum Machine Learning (QML) models, by operating in exponentially large Hilbert spaces, can efficiently represent such distributions and provide favorable inductive biases for learning, offering a promising pathway to overcome these limitations. However, their practical deployment in the Noisy Intermediate-Scale Quantum (NISQ) era is severely hindered by gradient coupling, barren plateaus, data-loading bottlenecks, and high inference latency. To bridge classical AI and QML, we propose QPAT (QCBM-Prior-Assisted Training), a universal framework that employs a Quantum Circuit Born Machine (QCBM) as a learnable generative prior to provide a task-aligned structured inductive bias during classical model training. The framework can be seamlessly integrated into any classical backbone, including LSTMs, CNNs, and Transformers. We introduce a decoupled fitness-based scoring mechanism and a stochastic "Drop-Prior" mechanism to implicitly distill quantum structural knowledge into the classical weights during training. Evaluated extensively on six benchmarks spanning time-series forecasting, computer vision, and natural language processing, our framework consistently outperforms purely classical backbones in both predictive performance and stability across random seeds; ablation studies further show that these gains are not reproduced by unstructured random priors or a classically learned generative prior. Remarkably, our approach enables completely classical inference, bypassing quantum latency while preserving quantum-enhanced accuracy, paving a highly practical path for QML deployment. 

---
# LSRP*-LA: Scalable and Anytime Multi-Agent Path Finding for Large Agents with Asynchronous Actions 

**Authors**: Dachuan Liu, Shizhe Zhao, Zhongqiang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2611.33793)  

**Abstract**: Multi-Agent Path Finding (MAPF) seeks collision-free paths for multiple agents from their start locations to their goal locations while minimizing path costs. Most MAPF algorithms assume that each agent occupies a single vertex and that the actions of all agents are synchronized. These assumptions may limit the use of MAPF planners in practice. Existing planners that get rid of both assumptions either handle only a few agents or cannot improve their solutions. Recently, a planner called LSRP* was proposed, which is scalable and anytime with asynchronous actions, but still limited by that each agent occupies a single vertex. This paper develops a new planner called LSRP*-LA for MAPF for large agents with asynchronous actions (LA-MAPF-AA), where an agent can occupy several cells. LSRP*-LA is scalable and complete, and it keeps improving solution quality after the initial solution is obtained until the runtime budget depletes. We analyze the properties of LSRP*-LA and test it against several baselines with up to 800 agents on various maps. LSRP*-LA can handle twice as many agents as baseline methods and its solution cost is about half that of recent synchronous planners for large agents. 

---
# Triage Before Fine-Tuning: Feasibility-Gated Selection and Monitored Adaptation of Embodied Foundation Models for a Fixed Physical Platform 

**Authors**: Zilin Huang, Zhengyang Wan, Zihao Sheng, Boyue Wang, Ruolin Yang, Yuhao Luo, Sikai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.33772)  

**Abstract**: Embodied foundation models are increasingly deployed by selection: a practitioner picks a public checkpoint by benchmark rank and fine-tunes it on local data. The platform is fixed before the model is chosen, with little or no local data. We ask what such a checkpoint loses on a fixed platform and which part of the loss local data can buy back. We split the gap by what can repair it. A sensing gap, information the sensors do not capture, cannot be closed with their data; a latency gap, output arriving after the control loop needs it, belongs to the loop, not the weights; only a distribution gap is fine-tuning's to repair. TRIAGE turns this into feasibility-gated selection and monitored adaptation in five stages: gates on the sensor contract and on end-to-end latency that no score can override, probes on ten minutes of local logs, a ranking by target-relevant score among the candidates that pass, and fine-tuning that is monitored and checked in closed loop before release. On driving benchmarks, a deadline below a leading closed-loop model's latency makes it collide on 80 to 90% of routes, and losing field of view moves plans 2.2 to 5.7 times more than losing resolution at equal pixels. Local frames close 75% of a stability gap but raise perturbation sensitivity at small budgets, and picking by score violates a gate in two of four deployment profiles. On a full-scale vehicle, the factory camera failed the sensing gate and an added camera repaired it. The latency gate removed the two highest-scoring candidates, neither of which completed a closed-loop trip. The selected model passed acceptance in 4 of 5 closed-loop trips, and the closed-loop check withheld a fine-tuned model that had improved offline. TRIAGE lets organizations that cannot train foundation models from scratch find out whether a public checkpoint can run on the robot or vehicle they own, and what to fix if not. 

---
# ScanAR: Parallel Row-Scan Decoding for Efficient Image Generation in Unified Autoregressive Models 

**Authors**: Jingen Qu, Xi Chen, Yingbo Tang, Xinghang Li, Tao Kong  

**Link**: [PDF](https://arxiv.org/pdf/2611.33761)  

**Abstract**: Unified multimodal autoregressive (AR) models represent text and images in one token sequence and generate all modalities with a shared next-token prediction head. Their image generation, however, decodes an $H\times W$ visual grid token by token in raster order, which takes $HW$ sequential steps and dominates inference time. Existing acceleration methods for visual AR generation either change the tokenizer, the prediction head, or the model interface, or keep the model but cannot reach high parallelism without losing quality, so they do not carry over to unified AR models. We propose ScanAR, a row-scan parallel decoding method that only redefines the predecessors of visual tokens. The first visual row is generated from left to right as usual, and each later row is generated in a single parallel step, with every token predicted from the vertically aligned token of the previous row. Nonvisual tokens keep the standard autoregressive rule. ScanAR reduces the visual decoding depth from $HW$ to $H+W$ without changing the backbone, vocabulary, or output head and without adding parameters, and it supports both training from scratch and lightweight post-training. On Emu3.5-Image-34B, ScanAR post-training reaches a GenEval score of 82.36, the best among post-training acceleration methods trained on the same data, and generates $512\times512$ images $13.8\times$ faster than raster decoding under the same inference setting. On ImageNet, ScanAR trained from scratch has the highest throughput at every model size with FID comparable to raster LlamaGen. On the 34B Xiaomi-Robotics-U0 model, ScanAR preserves the overall quality of embodied scene generation, interleaved image-text generation, and video generation. 

---
# Teaching Students to Stand on Their Own: Distilling Vision Foundation Models for Fine-Grained Few-Shot Class-Incremental Learning 

**Authors**: Junsu Kim, Seungryul Baek  

**Link**: [PDF](https://arxiv.org/pdf/2611.33749)  

**Abstract**: Fine-grained few-shot class-incremental learning (FG-FSCIL) aims to incrementally learn visually similar new classes from limited images without catastrophic forgetting. Prior work typically learns strong representations on base classes or leverages vision foundation models (VFMs) directly as backbones or via knowledge distillation. However, VFM backbones incur high computational costs and do not guarantee fine-grained recognition. Although knowledge distillation reduces inference cost, our preliminary experiments reveal that distilled students still lag substantially behind their VFM teachers, particularly on new classes. To address this limitation, we propose STAND, a systematic VFM-to-student distillation design tailored for FG-FSCIL. Specifically, STAND connects the teacher and student by equipping the student with a fixed copy of the teacher's classifier, aligning their outputs via projectors, and feeding both models composite images that combine different training images. On these composite images, relation distillation matches how the teacher and the student relate each image to other images, while difference distillation matches how this relation changes between two composite images. Finally, we freeze the student and classify new classes using length-calibrated prototypes, which restore the reduced similarity of few-shot prototypes. Extensive evaluations demonstrate that STAND achieves state-of-the-art performance across FG-FSCIL datasets at low computational cost, while generalizing across student backbones and broader datasets. 

---
# FlashProve-Next: Verifiable Inference for Production LLM APIs 

**Authors**: Liang Song  

**Link**: [PDF](https://arxiv.org/pdf/2611.33741)  

**Abstract**: Clients of closed-weight LLM APIs cannot verify whether a provider ran the advertised model, precision, hidden prompt, and reasoning budget. Existing approaches either trust hardware or third parties, or re-prove every layer of an integer surrogate at high cost while diverging from native floating-point serving.
FlashProve-Next produces verifiable, zero-knowledge-masked proofs for individual LLM API responses without modifying native serving (bf16/fp8/mxfp4, random sampling, FlashAttention, CUDA graphs, batching, prefix caching) and without disclosing weights, hidden prompts, hidden reasoning, or activations (dense profiles fix coarse geometry; cross-architecture hiding is deferred). A non-depleting Reed-Solomon weight commitment model_id opens inside proofs via hidden column openings, while local per-operator rounding contracts remove integer surrogates and the refusal-bit leak. Cost-weighted spot-checking supports non-interactive public verification (NI-315, $\varepsilon \le 2^{-40}$, $Q \le 2^{64}$) and external-seed C_pre mode (ExtSeed-153, $\varepsilon \le 2^{-20}$, $Q_{\rm samp} \le 2^{30}$).
Twenty-eight Lean 4 modules (696 declarations, zero sorry, standard axioms only) mechanize round-by-round soundness, bucketed sampling, $\mathbb{F}_{p^3}$ constrained-mask binding, causal masking, six IR gadgets, and executable FPN2 verifiers. Five-model Blackwell calibration gives zero false rejections across 1.85M GEMM/MoE blocks and 1.96M non-GEMM rows. Our zero-Phase-2 folded backend (folded ZK simulator open) proves Qwen3-0.6B ExtSeed-153 (1.47B cells, 8 MiB) in 450.2 s GPU (10.76 s CPU) at unique-decoding $Q=128$, passes a 14/14 cheating matrix, and scales to 36-layer Qwen3-4B (dev $4+2$) and live vLLM Qwen3-30B-A3B-FP8 Relation 6 (1.49 s prove, 23.4 ms Rust verify, public weight tiles). Rust fpn-verify checks 8 MiB FPN2 Relation 2 proofs in 10.1 ms on CPU. 

---
# Practice Makes Fluent: Learning to Use Earlier Predictions in Looped Models 

**Authors**: Minseok seo, Hee-Seon Kim, Changick Kim  

**Link**: [PDF](https://arxiv.org/pdf/2611.33722)  

**Abstract**: People become faster at a task through practice, but models that improve with experience do not necessarily learn to use less computation. We train looped models, whose intermediate predictions allow earlier exits, to perform tasks with fewer iterations through practice. This is difficult because shared-weight updates can degrade the final-loop prediction while improving early ones, and the exit gate must track early predictions as their quality changes. We propose FLUENT, which requires no additional task labels or rewards: it distills the initial model's final-loop prediction into early predictions, anchors the current final-loop prediction to the same frozen teacher, and trains the exit gate to balance disagreement with the current final-loop prediction against computation cost. FLUENT improves performance at shallower depth in new Crafter worlds, raises LIBERO success with fewer action-head iterations, and maintains language-model accuracy at lower selected depth. On a real robot, success rises from 62.3\% to 74.6\% while mean time over successful episodes falls from 69\,s to 31\,s. The project page is available at this https URL. 

---
# Sports Agent: A Self-Evolving System for Conversational Data Analytics 

**Authors**: Chao Chen, Fuqi Lin, Ze Chen, Ziyan Liu, Chen Li, Haishu Jiang, Bo Xu, Zihan Jiang, Yue Wu, Zheng Wang, Kun Wang, Zhihang Fu, Jieping Ye  

**Link**: [PDF](https://arxiv.org/pdf/2611.33700)  

**Abstract**: Conversational sports analytics requires accurate, evidence-based responses with low latency. Changes in query distributions, data resources, and model behavior can degrade static agent configurations, motivating controlled evolution guided by runtime evidence.
We present Sports Agent, a self-evolving framework integrating an Agentic SQL Model for structured-data reasoning, a harness for planning and tool orchestration, and a unified Feedback Engine connecting online execution with offline evolution. The engine assesses responses and execution trajectories, verifies factual claims, diagnoses failures, and generates evolution signals. Model evolution combines SQL-to-Question Scaling for coverage expansion with Graph-Guided Scaling for error-targeted synthesis. Generated questions undergo trajectory distillation and quality filtering for iterative supervised fine-tuning, followed by a separate Post-RL stage. Harness evolution translates recurrent failures into bounded prompt and skill updates, validated through independent audits, execution replay, and regression testing. Performance-tolerance checks guard against unacceptable degradation during simplification.
The resulting Qwen3.6-35B-A3B-based model outperforms Qwen3.8-Max on sports question answering and analysis while achieving approximately three times the inference efficiency. Across 30 harness-evolution rounds, the aggregate performance score improves by 10.8 points. In a live sports analytics product, the evolved system improves factual correctness by 10.5 points and average response length, a proxy for answer richness, by 158.1%, while reducing average latency by 41.3% relative to the baseline. 

---
# Rethinking Sequential Locate-Then-Edit: Optimality and Stability 

**Authors**: Bingqing Liu, Wei Liu, Zhiying Deng, Jun Wang, Yuhua Li, Ruixuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.33692)  

**Abstract**: The Locate-Then-Edit framework enables efficient knowledge updates in large language models without requiring full retraining. Recent research has increasingly focused on sequential editing, where knowledge arrives continuously and the model is updated incrementally. However, existing sequential methods commonly suffer from catastrophic forgetting and model collapse after only a few hundred edits. In contrast, batch editing - which jointly optimizes all edits and yields the optimal solution under a specified editing objective - can support tens of thousands of edits with high edit efficacy while preserving the model's general capabilities, resulting in a substantial performance gap between the two editing paradigms. In this paper, we demonstrate that this gap can be closed via a batch-equivalent sequential editing approach. Moreover, we investigate the underlying mechanism of model degradation in existing sequential methods by analyzing the convergence behavior of weight perturbations as edits accumulate. Our analysis reveals that current methods typically exhibit linear or logarithmic growth in perturbation, whereas the proposed batch-equivalent method achieves time-invariant growth, thereby theoretically guaranteeing long-term stability and preserving the integrity of pretrained knowledge throughout massive-scale sequential editing. 

---
# Read Before You Use: Improving Visual Memory Utilization in Vision-Language Models 

**Authors**: Junseok Kim, Kyungmin Min, Kyomin Jung  

**Link**: [PDF](https://arxiv.org/pdf/2611.33687)  

**Abstract**: Multimodal memory enables long-horizon vision-language agents to preserve and retrieve past experiences. While prior research has focused on memory construction and retrieval, effectively using the provided memory remains a challenge for vision-language models (VLMs). We find that VLMs often fail to answer downstream questions even when sufficient multimodal evidence is provided. To investigate why VLMs struggle, we replace memory images with reference captions and observe substantial performance gains, suggesting difficulties in reading visual evidence directly. However, model-generated captions may omit critical information, limiting this approach in practice. These limitations motivate a closer examination of memory utilization, revealing failures in both evidence reading and downstream question answering. Building on this distinction, we propose \textbf{MemRU}, a two-stage training framework that improves evidence reading through supervised fine-tuning (Read SFT) and downstream evidence utilization through Group Relative Policy Optimization (Use GRPO). Experimental results demonstrate that our 8B model generalizes across three unseen multimodal memory systems, outperforming Qwen3-VL-32B and approaching GPT-5-mini. 

---
# What Is Redundant at High Resolution? Efficient Attention and Feed-Forward Design for All-in-One Image Restoration 

**Authors**: Dongheon Lee, DongJae Lee, Sewhan Chun, Chanseul Cho, Youngwook Kim, Youngmin Ro, Yonghyun Jeong  

**Link**: [PDF](https://arxiv.org/pdf/2611.33636)  

**Abstract**: Multi-scale architectures are widely adopted in all-in-one image restoration (AIO-IR) to capture larger-scale context at low-resolution stages and refine details at high-resolution (HR) stages. However, processing spatially large feature maps at HR stages incurs substantial computational costs and memory traffic, making efficient HR processing challenging. To address this challenge, we propose High-resolution Global-Local Restoration (HiGLR), an efficient multi-scale AIO-IR network grounded in an analysis of spatial redundancy in self-attention and convolutional feed-forward networks (ConvFFNs). Specifically, this analysis identifies three key redundancies: per-query readout at the finest-scale stage, key routing across neighboring queries at intermediate stages, and spatial mixing in the expanded feature spaces of ConvFFNs. Accordingly, HiGLR replaces costly self-attention at the finest-scale stage with a learned input-independent kernel. At intermediate stages, we propose a Block-Routed Hybrid Attention, which combines coarse-scale dense attention with fine-scale block-sparse attention over value residuals relative to coarse summaries, and shares routing indices across neighboring queries. We further introduce a Dual-Space FFN that performs spatial mixing in a compact space and pointwise nonlinear transformation in an expanded space, enabling efficient kernel fusion. Combining these designs, HiGLR achieves state-of-the-art performance while delivering up to a $4.24\times$ inference speedup over existing methods. 

---
# The Detour Is the Danger: Preserving Security During LLM Agent Recovery 

**Authors**: Boyang Ma, Jinku Li, Hao Wu, Sheng Gao, Man Zhou, Minghui Xu, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.33605)  

**Abstract**: LLM agents increasingly rely on external tools and reusable Skills to execute complex workflows, autonomously replanning when operations fail. However, such recovery may preserve task functionality while silently discarding the security protections required by the original workflow. In this paper, we investigate workflow security downgrade, a failure-induced security threat in which agents execute sensitive operations without satisfying their original protection requirements. Unlike prompt injection, this threat exploits autonomous recovery decisions triggered by execution failures, without modifying instructions or tool definitions. We systematically evaluate this behavior through fault injection in an executable MCP/Skill environment spanning seven LLMs, twelve task templates, and three security-sensitive domains. To address this threat, we propose ARGUS, an obligation-preserving recovery framework that combines resource-bound security evidence, isolated recovery-plan validation, and execution-time enforcement. In a separate 864-run evaluation, ARGUS prevents all observed protection-loss events under enforcement, while recovery assistance yields model-dependent improvements in protected task completion over blocking alone. Our findings demonstrate that secure agent recovery requires preserving security obligations across workflow changes, rather than merely restoring task functionality. 

---
# DexEngine: Scaling Human Demonstrations into Physically Executable Whole-Robot Data 

**Authors**: Wongyun Yu, Honggeun Kim, Donguk Lee, Minchan Kim, Kyoung Whan Choe, Minsu Cho  

**Link**: [PDF](https://arxiv.org/pdf/2611.33579)  

**Abstract**: A growing body of recent work addresses the retargeting of human hand-object demonstrations to robotic hands. To train whole-robot policies, however, the retargeted motions must be kinematically feasible and physically executable, and be produced efficiently at scale. We present DexEngine, a simulation pipeline that turns human manipulation data into train-ready, physically executable arm-hand trajectories. DexEngine first places each clip's hand-object motion where the arms can reach it, finds an arm motion that follows the human hands, and then corrects the motion in physics simulation with residual reinforcement learning. From nine human hand-object datasets, we generate about 5,200 clips for each of four dexterous hands, augmented into 65,000 trajectories (116 h) and 325,000 episodes (578 h) in total. Even with arms attached, DexEngine tracks objects more accurately than prior floating-hand methods and runs over 10x faster than RL-based ones. Policies trained on our data far outperform those trained on the same clips without physics correction or with the arms attached after a floating-hand correction, showing that fine-tuning data need physics correction. Mid-training on our data further improves them at equal fine-tuning budget, in simulation and on a real robot. 

---
# RankCast: Phase-Aware Rank Balancing in MoE Serving 

**Authors**: Kaidi Fu, Xu-Hao Lu, Pingwei Sun, Jianchao Tan, Yuchen Xie, Xunliang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2611.33542)  

**Abstract**: Modern large language model (LLM) inference increasingly adopts disaggregated architectures that separate Prefill (P) and Decode (D) phases. Meanwhile, Expert Parallelism (EP) has been widely adopted in large Mixture-of-Experts (MoE) model serving. However, load imbalance still slows down MoE inference because each layer must wait for the busiest EP rank. The main source of latency differs between prefill and decode. Prefill is compute-bound, and its MoE layers' latency is determined by the maximum token count of the busiest rank. At the same time, under moderate batch sizes, Decode is memory-bound, so the latency is dominated by the maximum active-expert count across ranks. To solve both problems under a unified framework, we present RankCast, a system that reduces rank-level maximum token counts during prefill and active-expert counts during decode. For every token, RankCast preserves the predicted Top-p experts and chooses the remaining k-p from its Top-L candidates (L > k). In prefill, RankCast yields a price per rank that steers flexible assignments to less loaded ranks. In decode, RankCast closes experts on the busiest ranks while preserving the constraints for each token. Both paths are implemented and tested in the state-of-the-art inference frameworks SGLang and DeepEP. We overlap RankCast's kernels with model execution and reuse the existing DeepEP communication channel for metadata transmission. On Qwen3-235B-A22B with EP=8 and Qwen3.5-397B-A17B with EP=16, RankCast removes over 75% of the busiest rank's excess prefill load, improves prefill input throughput by up to 19%, and reduces decode time per output token by up to 10%. The quality cost is below one percentage point. 

---
# PRECEPT: Security Self-Improvement of Coding Agents via Evolving Principles 

**Authors**: Keyan Guo, Rupam Patir, Haipeng Cai, Hongxin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2611.33519)  

**Abstract**: AI coding agents increasingly write and submit code with limited human review, and much of it passes functional tests while remaining insecure. We show that these failures persist. Across 14 task sequences, Codex, Claude Code, and Qwen Code fixed a vulnerability 85.7% of the time when shown the failing security test, then reintroduced the same vulnerability class in a new session 26.7% of the time, about as often as agents that had never seen it (27.5%). The capability is present; the retention is not. Neither recording the repair in the agent's persistent context nor supplying published secure coding principles closed the gap: the recorded fix stayed tied to the family of the task it came from, and the principles, written for human developers, remained too general to apply to the code at hand. The knowledge exists; it is not activated. We call the missing ability security self-improvement: an agent learns from its own security failures so that each lesson persists across sessions, applies to new code, and does not break what works. We present PRECEPT, which equips coding agents with this ability through an evolving library that starts from authoritative secure coding principles and revises them against the agent's own failures, so that each principle comes to fire on the code where that agent actually goes wrong. A revision is kept only if fresh executions show it makes code more secure while keeping functional success within a small tolerance of the agent without the library. Improvement continues across evolution steps rather than saturating after the first. On task families never seen during evolution, PRECEPT's weakest agent reaches 97.1% secure working code, above the strongest unaided agent at 81.9% and the strongest baseline at 87.7%. 

---
# Joint Estimation of Abilities and Item Parameters for Large Language Models Using Probability and Binary Responses: A Fractional-Response IRT Framework 

**Authors**: Masayuki Takayama, Shohei Shimizu  

**Link**: [PDF](https://arxiv.org/pdf/2611.33493)  

**Abstract**: Large language model (LLM) evaluations provide two response types: probabilities assigned to correct answers and binary indicators of correctness. Restricting evaluation to one type narrows the range of available models or discards probability information. We propose a method to integrate both types through a common two-parameter logistic item response model and fractional-response quasi-maximum likelihood estimation. The framework includes binary-only models in common-scale evaluation without discarding available probabilityinformation; broader measurement coverage does not by itself ensure greater precision or measurement equivalence. Simulations evaluate statistical efficiency when both response types have the same expected value. Moreover, adding probability respondents improves item calibration more efficiently than adding binary respondents. However, adding noisier binary responses to an already precise probability sample can increase estimation error under equal observation weights. An application to Massive Multitask Language Understanding (MMLU) combines probability response data from the Open LLM Leaderboard with binary response data from Holistic Evaluation of Language Models (HELM) and reveals model-family response patterns, differences in item characteristics, and systematic ability shifts between evaluation protocols. The proposed framework extends common-scale IRT evaluation to models with different response formats while highlighting the roles of observation variability and evaluation-protocol comparability in statistical efficiency and interpretation. 

---
# The Geometry Is the Memory: Task-Density Routing for Continual LoRA 

**Authors**: Alireza Heidari, Thomas Veasey, Mariano Tepper  

**Link**: [PDF](https://arxiv.org/pdf/2611.33487)  

**Abstract**: Freezing older adapters does not ensure that a continual learner will select and use them properly at inference time. When multiple adapters exist, routing among them and identifying the adapter whose representation space matches the query's answer can be difficult. We propose CONTLORA, which replaces stored examples with compact class statistics for adapter selection and classification. Gaussian class models in a frozen representation select LoRA adapters, while accumulated feature moments exactly recover a class-balanced ridge classifier for fixed training features. Trained without retaining images or feature vectors, CONTLORA reaches 88.55% accuracy on 20-task CIFAR-100 and 86.99% on CUB-200. Across three benchmarks in the matched component study, its mean final accuracy differs from an independently trained stored-feature counterpart by at most 0.07 percentage points. Ablations show that improved task coverage need not improve classification and adapter composition also matters. On held-out classes, competitor-centered likelihood scores improve novelty AUROC over raw likelihood without altering routing ranks. These results support statistical memory as an alternative to example-retention approaches in the evaluated settings, while identifying task ambiguity as an ongoing limitation. 

---
# Benchmarking Open-Ended Vulnerability Discovery in Windows Kernel Binaries 

**Authors**: Jongseong Kim, Hwiwon Lee, Dongjun Kim, Lingming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.33486)  

**Abstract**: Frontier models are increasingly used to discover real-world security vulnerabilities, making it critical to systematically evaluate their cybersecurity capabilities. However, existing security benchmarks primarily focus on known vulnerabilities in open-source projects, largely overlooking closed-source binaries, which underpin much of the software industry. In this paper, we ask: how effective are frontier models at uncovering deep vulnerabilities in production binaries? To answer this question, we present DeepBinary, the first comprehensive benchmark for open-ended vulnerability discovery in Windows kernel binaries. Given a target binary, an agent analyzes it using a disassembler and kernel debugger in network-isolated VMs and reports each vulnerability with a crashing proof of concept (PoC). Unlike existing benchmarks, DeepBinary replays every report on fresh VMs and employs an LLM-based judge to assess its validity, uniqueness, and impact (estimated by bug bounty values). Beyond known vulnerabilities, DeepBinary also enables the discovery and validation of previously unknown (0-day) vulnerabilities. We conduct a comprehensive evaluation of representative open- and closed-weight models (including GPT-6 Sol, Claude Opus 5, and DeepSeek V4.1 Flash) across diverse coding agents (including Codex and Claude Code) on 12 target binaries. In total, they discover over 200 vulnerabilities with an estimated aggregate bounty of $661,000, including 146 distinct previously unknown vulnerabilities across all experiments. Our results further reveal substantial differences in models' ability to analyze binaries, conduct experiments, and sustain vulnerability discovery over long horizons. Overall, DeepBinary makes long-horizon, open-ended vulnerability discovery in complex binaries measurable, revealing fundamental differences in how frontier models analyze and explore such systems. 

---
# SecID: An LLM-Guided Framework for Threat-Aware Secondary Asset Identification with Quantitative RTL Analysis 

**Authors**: Md Ajoad Hasan, Dipayan Saha, Upoma Das, Sujan Kumar Saha, Mark Tehranipoor, Farimah Farahmandi  

**Link**: [PDF](https://arxiv.org/pdf/2611.33453)  

**Abstract**: The growing complexity of modern hardware designs is making security assurance increasingly difficult. One of the foundational steps in pre-silicon security verification is identifying security assets. While primary assets are often evident from design functionality, the internal signals that can facilitate their compromise are much less apparent. Complex data- and control-flow dependencies create many such internal security-critical locations, whose exploitation depends on the threat-specific attack mechanism---these are the secondary assets. To precisely identify these, we propose SecID, a multi-agent framework that combines deterministic RTL analysis with bounded dynamic evaluation and LLM reasoning to identify threat-specific secondary assets. Each asset decision is supported by functional influence, threat features, and explicit attack mechanisms. We demonstrate SecID on seven open-source SoC and IP designs, including openMSP430, NEORV32, and CV32E40P. The experimental results are corroborated through comparison with prior work, fault simulation, and formal analysis, and further evaluated through pipeline- and final-stage ablation, threshold-sensitivity studies, and final-stage repeatability analysis. Together, these evaluations show that SecID can identify highly threat-relevant secondary assets with respect to a given primary asset, providing a scalable path toward hardware-security verification. 

---
# HumblePRM: Improving VLM Abstention in Ambiguous Situations through Process Reward Modeling 

**Authors**: Saket Reddy, Ke Yang, ChengXiang Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2611.33452)  

**Abstract**: Reliable deployment of Vision Language Models (VLMs) requires that a model's expressed confidence remain grounded in what it can actually see, and that it recognize when the visual evidence is too ambiguous to commit to a confident response. Standard post-training uses outcome reward models (ORMs) that supervise only the final decision, rewarding a confident answer regardless of whether the image supports one, while existing methods either tend to detect uncertainty post-hoc or assume an objective verdict always exists. We instead split multimodal reasoning into two stages: perception, where the model must factually describe the scene, and calibration, where it must judge whether that evidence is sufficient to commit to a conclusion. We introduce HumblePRM, a lightweight process reward model (PRM) that scores each stage independently, rewarding agreement between evidence and expressed confidence rather than outcome correctness alone. Training with HumblePRM increases abstention due to uncertainty on ambiguous inputs considerably while leaving accuracy on clear images roughly unchanged. We also find that the improvement transfers to Gemma 4 E4B, a larger, popular edge model. Our findings show that process reward supervision is a promising route to VLMs with better calibrated abstention, ensuring more reliability rather than unpredictable behavior. 

---
# When the Knowledge Runs Out: A Part-Level Pilot Evaluation of How Language Models Handle Partially Covered Questions 

**Authors**: Vahid Mohammadi  

**Link**: [PDF](https://arxiv.org/pdf/2611.33428)  

**Abstract**: Assistants that answer from supplied reference material often receive material that covers some parts of a user's question and not others. Existing evaluations of insufficient context mostly score one answer per question, and often treat partially sufficient context as simply insufficient. We describe a small, open evaluation that scores each part of a three-part question separately, with a verdict for parts that are answered correctly from the material but carry an added unsupported claim. On 30 cases (five scenarios, six coverage patterns, five samples each) we test claude-haiku-4-5 and llama3.1:8b. Without an explicit instruction to stay within the material, both models fill most gaps from their own knowledge (unsupported-claim rates of 0.71 and 0.90). One added sentence instructing them to answer only from the material brings these rates to 0.00 and 0.06, consistent with prior work. What remains is less visible: answers that strengthen the material, for example turning "many job sites prohibit bots" into a flat "no". We also report on the judge. Three rounds of blind hand labelling show that this added-claim verdict is the hardest to apply, for the judge and the author alike; RAGAS faithfulness is unsuitable for this task because it penalises correct withholding; and a difference between the two models that is clear under one judge disappears when the judge model is replaced. The dataset, code, hand labels and per-run figures are public. 

---
# Temporal Selective Exploration for Reinforcement Learning-Guided Continuous-Discrete Flow Matching in 3D Molecular Design 

**Authors**: Lianghong Chen, Yan Yi Li, Gen Zhou, Yuxi Long, Ganlin Feng, Mike Domaratzki, Pingzhao Hu  

**Link**: [PDF](https://arxiv.org/pdf/2611.33417)  

**Abstract**: Flow matching has shown strong potential in generative tasks, while its optimization via Reinforcement Learning (RL) remains underexplored, especially in continuous-discrete mixed settings. Moreover, flow matching is formulated as an Ordinary Differential Equation (ODE), whose deterministic trajectories do not naturally support RL optimization. Introducing stochasticity by converting the entire trajectory into a Stochastic Differential Equation (SDE) is a common practice. Nevertheless, not all timesteps contribute to the final outcomes, and redundant exploration introduces additional noise. Furthermore, in multi-objective settings, some objectives may dominate the optimization while others receive insufficient updates. In this study, we propose an RL-guided continuous-discrete flow matching framework with temporal selective exploration for 3D de novo molecular design. Specifically, our method jointly optimizes continuous and discrete flow matching for atomic coordinates and molecular identities, respectively, while restricting exploration to timesteps that primarily contribute to target properties, reducing ineffective exploration. We also introduce a reward-based advantage calibration mechanism to mitigate objective dominance in multi-objective optimization. The framework is applied to design inhibitors for both the well-studied target protein EGFR and the challenging Pin1 protein. We identify some novel candidate inhibitors, supported by in silico validation. 

---
# Sequentially Acquiring Concept Knowledge to Guide Continual Learning 

**Authors**: Shivanand Kundargi, Kowshik Thopalli, Tejas Gokhale  

**Link**: [PDF](https://arxiv.org/pdf/2611.33399)  

**Abstract**: The goal of continual learning (CL) is to adapt to new data (plasticity) while retaining the knowledge acquired from old data (stability). Existing methods are explicitly designed to focus on balancing stability and plasticity to mitigate the challenge of catastrophic forgetting while promoting learning. However, the impact of order and nature of samples from the new data that the network is trained on remains an under explored factor. We argue that a CL algorithm should have the ability to rank incoming samples in terms of their relationship with prior data and their effect on the learning process. In this work, we investigate if scoring and prioritizing incoming data based on their semantic relationships with the model's current knowledge can benefit CL. We propose SACK, short for Sequentially Acquiring Concept Knowledge, a plug and play technique for continual learning methods. SACK dissects categorical knowledge of the model into fine-grained concepts, computes the relationships between previously learned concepts and new concepts in each experience, and uses this relationship knowledge for prioritizing new samples. Experiments across several types of CL methods (regularization, replay, and prompt based) in class incremental and task incremental settings demonstrate that integration of SACK not only improves accuracy and reduces forgetting but also handles long tail distribution, helps focus on semantically interpretable regions, and yields better calibrated and robust models compared to prior approaches. 

---
# Best-of-$N$ Test-Time Steering Can Lead to Reward Hacking in Vision-Language-Action Models 

**Authors**: Vanshaj Khattar, Jing Liu, Toshiaki Koike-Akino, Ming Jin, Ye Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.33398)  

**Abstract**: Vision-language-action (VLA) models have shown promise for general-purpose robotic manipulation, but adapting to changing environmental conditions still remains challenging. Recent test-time steering methods have shown promise in enabling adaptive VLA policies by utilizing additional computation during deployment. In this paper, we consider best-of-$N$ test-time steering for VLA models, where multiple action candidates are sampled from a fixed policy and a verifier is used to select the best one during deployment. We show that increasing $N$ does not always improve performance when the verifier is inaccurate, and can favor candidates whose scores exploit prediction errors, a form of reward hacking. We study this effect on 60 LIBERO-Safety task-level cases using a $\pi_{0.5}$ policy, with ten repetitions per case. We introduce controlled errors in a MuJoCo trajectory verifier and show that as $N$ increases, selected candidate rewards diverge from the accurate verifier rewards, leading to reward hacking. We also show that verifier calibration and softmax-based candidate selection can mitigate reward hacking in some settings. 

---
# RoboARC: Physical Puzzles for Robotic Manipulation under Spatial Constraints 

**Authors**: Qineng Wang, Jianwen Lyu, Pingyue Zhang, Reuben Tan, Xiaoyu Huang, Tonghe Zhang, Kangrui Wang, Yelong Shen, Jianfeng Gao, Guanya Shi, Ruohan Zhang, Manling Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.33390)  

**Abstract**: Many hard manipulation tasks come down to a spatial constraint that only a few motions satisfy, such as passing a part through a gap barely wider than itself or separating two interlocked parts. These motions depend on the exact geometry of each scene, so demonstrations and skills recorded elsewhere rarely transfer. We therefore ask: when manipulation hinges on spatial constraints, can robots reason out the right motions? We introduce RoboARC, a benchmark of 90 tasks in six constraint types over rigid and soft materials, on three robots in MuJoCo. Every task is a frozen scene with a physics-based grader and a verified reference solution that replays bit-identically. Each task comes in a Privileged regime, with the full simulator state and resets, and a Standard regime with real-sensor inputs (camera images and proprioception) and a single attempt without resets. We evaluate seven coding agents and four VLA policies, none trained on these tasks. The strongest agent, GPT-6 Astra in Codex, solves 86% of the tasks in the Privileged regime and 43% in the Standard regime, and the four VLA policies, run zero-shot, solve none of their 18 one-arm puzzles. On 15 puzzles that GPT-6 Astra solved in the Privileged regime and failed in the Standard regime, object poses add 4 points and part geometry 13, but both together add 36, from 24% to 60%. Only 7% of the coding agents' failed runs pass more than half of a task, and 53% of the failed runs in the Standard regime are stuck on a single step. Frontier coding agents already reason out the right motions for many of these tasks once they know where the parts are and what shape they have. From their own sensors, in a single attempt, the strongest solves fewer than half of the tasks. The website can be found at this https URL 

---
# Poisoning Self-Evolving Security Policies in LLM Agents 

**Authors**: Aymen Noor, Honglu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2611.33385)  

**Abstract**: Self-evolving defenses improve LLM-agent security by learning reusable policies from observed attacks. However, we demonstrate that this learning process itself can become an attack surface. Using SED (Self-Evolving Defense) in the AgentDojo benchmark, we construct a single prompt-injection episode that causes the defense to learn an over-broad policy and subsequently refuse legitimate requests. Across 10 Nemotron runs, every poisoned run over-refuses at least one legitimate task, while unrelated-task performance remains unchanged and the original prompt-injection defense remains effective. The attack also transfers to gpt-oss-120b; however, its poisoning mechanism depends on a successful initial prompt injection and therefore fails to activate when the target agent resists the underlying injection. In contrast, integrity-oriented attacks targeting SED's judge, policy synthesis, retrieval, and organization mechanisms fail in our evaluated settings, identifying over-refusal as the primary exploitable failure mode observed in our experiments. We further identify self-poisoning from false-positive harm judgments without an attacker. Finally, contrastive policy synthesis and an automatic broadness-check gate substantially reduce, but do not eliminate over-refusal, without compromising security in our evaluated settings. 

---
# Benchmarking Segmentation Models for Scalable Mangrove Mapping with Sentinel-1/2 Imagery 

**Authors**: Damba Koné, Sara Mobsite, Joris Guérin, Élodie Blanchard, Thibault Catry  

**Link**: [PDF](https://arxiv.org/pdf/2611.33369)  

**Abstract**: Mangroves are dynamic coastal ecosystems that contribute to carbon storage, coastal protection and biodiversity conservation, but are vulnerable to climate change. Very-high-resolution imagery such as Pléiades enables detailed mangrove mapping but it is costly and have a low revisit rate. Large-scale monitoring requires free and regularly available data such as Sentinel-1 and Sentinel-2 but their spatial resolution poses challenges for detailed mangrove mapping. In this work, we benchmark deep learning approaches for mangrove mapping from Sentinel-1 radar and Sentinel-2 optical imagery using reference maps derived from very-high-resolution Pléiades imagery. We compare four segmentation configurations based on CROMA and BigEarthNet-pretrained ResNet50 representations with different segmentation decoders. We construct a mangrove mapping dataset covering French Guiana, Suriname, and Amapá. CROMA + UPerNet achieves the best performance, with an F1-score of 0.9218 and an IoU of 0.8549. These results highlight the potential of pretrained multimodal Earth observation representations and open Sentinel data to support mangrove resilience through scalable monitoring. 

---
# A Hierarchy of Risk Certificates for Representation-Based Prediction 

**Authors**: Mohit Kumar, Somayeh Kargaran  

**Link**: [PDF](https://arxiv.org/pdf/2611.33333)  

**Abstract**: Representation-based prediction underlies many machine-learning systems. Yet a fitted predictor's failure to meet an accuracy requirement does not establish that its representation-decoder class cannot meet it. We study optimal population mean squared error (MSE) over Lipschitz decoders acting on a fixed representation. For blocks of $b\geq2$ independent and identically distributed (iid) observations, we minimize total squared error (normalized by $b-1$) under pairwise Lipschitz constraints. We prove that expected block scores form nondecreasing lower bounds on optimal class risk and converge to that risk on compact metric feature spaces, although no fixed block size provides a uniform relative guarantee. Verified dual lower bounds and iid calibration observations independent of model development, with bounded targets and predictions, yield simultaneous lower confidence bounds on optimal class risk and upper confidence bounds on fitted-predictor risk. These bounds can establish that every admissible decoder exceeds a prescribed MSE threshold or that an admissible fitted predictor meets it. Controlled experiments illustrate population-risk certification; fixed PTB-XL and Bike Sharing experiments provide deterministic empirical-risk conclusions. The results support principled assessment and selection of representation-decoder configurations. Efficient certification at scale and extensions to dependent observations remain open problems. 

---
# Onnx2TEE: Automated Graph-Aware Model Partitioning for TEE-based Edge AI Protection 

**Authors**: Daniela M. Vela Hernandez, Samuel Hildebrand, Sean Oesch, Brian Weber, Shayan Monadjemi  

**Link**: [PDF](https://arxiv.org/pdf/2611.33262)  

**Abstract**: AI models are increasingly being deployed at the edge. Since these models often require significant research and development to produce, are often deployed in potentially hostile environments, and downstream systems often rely on accurate data; protecting them from theft or manipulation is worthy of serious consideration. Trusted Execution Environments (TEEs), which provide hardware-enforced separation of software execution, can protect model weights but impose strict resource constraints, making full-model execution impractical for large neural networks. We present Onnx2TEE, an automated, model-independent pipeline that partitions ONNX models across rich, trusted execution environments for non-sequential feed-forward networks using graph-aware segmentation to preserve computational dependencies while minimizing boundary communication and reducing TEE-induced inference latency. Evaluation on an NVIDIA Jetson AGX Orin using an Arm TrustZone TEE demonstrates that model parameters can be highly concentrated in individual segments, supporting selective protection of parameter-dense model components. The evaluation further shows that the generated trusted app's memory requirements depend on both computation and intermediate tensor sizes. 

---
# One Token, Many Lessons: Sliding-Window Attention within Gated Delta Networks 

**Authors**: Yuchen Song, Yangbin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.33257)  

**Abstract**: Linear attention compresses historical context into a fixed-size recurrent state, reducing the computational and memory costs of long-sequence modeling. From the perspective of test-time regression (TTR), the state matrix can be viewed as an online-learned regression model, where keys and values serve as inputs and targets respectively, and queries are used to read from the model. From this perspective, the effectiveness of state learning depends not only on the update rule but also on how the key-value learning pairs used for state updates are constructed. While existing work has largely focused on how the state is updated and organized, we investigate how the learning pairs used for state updates are constructed. To this end, we propose Sliding-Window Gated DeltaNet (SWGDN), which replaces short convolutions in Delta-Rule linear attention with sliding-window attention to perform content-dependent local aggregation of projected queries, keys, and values. Our experiments show that SWGDN makes the key-value pairs used in consecutive updates more similar in both GDN2 and KDA. In SWGDN, carrying information from the same token across updates can help the recurrent state reinforce a previously learned association. The local mixing strategy can be incorporated into any Delta-Rule-based linear attention architecture without changing its state update equations or chunkwise parallel algorithms. We apply this strategy to two Delta-Rule variants, GDN2 and KDA, and experimental results demonstrate that SWGDN outperforms the original Delta-Rule baseline across multiple benchmarks. 

---
# Anti-AI in the Wild: From Toxic Discourse to Action 

**Authors**: Yage Zhang, Yukun Jiang, Ziqing Yang, Yihan Ma, Xinyue Shen, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.33254)  

**Abstract**: Online discussion of generative AI has become increasingly contentious, and communities devoted to supporting, opposing, or debating AI have emerged on social media. Some of these communities take anti-AI actions, such as deploying crawler tarpits and data-poisoning services to interfere with data collection and model training. Yet, the scope of these actions and their relationship to online discourse remain poorly understood. In this paper, we present the first measurement study of online conflict over generative AI, from toxic discourse to anti-AI actions. We collect 931,640 posts from 25 Reddit communities, label them with a large language model detector validated against human labels, identify 3,312 anti-AI actions, and trace the output of three deployed anti-AI tools. Our analysis reveals that 30.0% of posts in communities devoted to this conflict are toxic, compared to 4.4% in general AI communities. Anti-AI actions in these communities mainly target AI crawling or model training data, such as image cloaking and data poisoning. We further examine the implications of these anti-AI actions for AI training, by analyzing the text and code generated from deployed tools. Notably, 66.5% of the code samples generated by a data-poisoning service contain fabricated facts, misused APIs, or broken logic, yet 93.6% passes a public code-curation pipeline, close to the rate for real source code (95.0%). For crawler tarpit output, adding a period to the end of each paragraph increases the curation survival rate from 16.4% to 91.9%. We also find that crawler tarpit and data-poisoning output has already entered Common Crawl, a widely used public web archive, and that crawler tarpit output has also entered FineWeb, a released pretraining dataset. We hope our study can facilitate a better understanding of online conflict over generative AI and its implications for content moderation and AI training. 

---
# Beyond the Efficiency-Efficacy Dilemma: A Critical Analysis of Transferability in Adversarial Malware Detection 

**Authors**: Veysel Alevcan, Mohammad Furqan Ali, Teresa Sousa, Luís Miguel Campos, Christoforos Ntantogian  

**Link**: [PDF](https://arxiv.org/pdf/2611.33243)  

**Abstract**: Machine Learning (ML) models are already being deployed at scale for malware detection. Therefore there is a pressing need to rigorously study adversarial robustness. To that end, we introduce the Adversarial AI Engine (AAIE): a platform for automated security stress-testing. We performed transfer attacks, via EMBER2018, against a Random Forest target. Counterintuitively, we found single-step FGSM to have the highest attack success rate (ASR) of 52.9% compared to iterative PGD with 45.2% (p < 0.002). Compared to iterative attacks which tend to overfit surrogate networks creating brittle perturbations, FGSM created more transferable examples. We benchmarked our attacks using ASR, transferability score, evasion rate, and resilience index. Through these metrics, proposed in concordance with emerging standards in AI governance, we expose shortcomings as our transferability scores peak at ~0.8 with constrained perturbations. We highlight the practical security implications of these findings and demonstrate AAIE for automated adversarial auditing. 

---
# CypherMAS: Benchmarking Routing Reliability in LLM-Based Multi-Agent Systems 

**Authors**: Danil Kostromin, Andrey Sakhovskiy, Semen Budennyy  

**Link**: [PDF](https://arxiv.org/pdf/2611.33242)  

**Abstract**: Multi-agent systems (MAS) built on large language models (LLMs) distribute reasoning across specialized agents that must exchange messages and invoke tools reliably. Existing evaluations typically couple communication with open domain tasks, making it difficult to determine whether a failure arises from reasoning, malformed tool use, lost messages, or forgotten requests. Prior benchmarks do not isolate communication execution or measure routing redundancy against a known optimal path. We introduce CypherMAS, a synthetic benchmark that isolates communication and routing in fixed multi-agent graphs. Agents in chain, star, bipartite, or ring graphs route cyphered messages to remote peers holding the matching decoders; since the optimal route is known, redundant LLM calls can be measured and routing failures may be easily tracked. Across eleven open-weight models from 4B to 753B, small models fail at routing and tool use, dense models above 27B degrade with diameter, and successful runs exceed minimum LLM calls. These findings identify communication reliability as a distinct scaling bottleneck for multi-agent systems and provide a controlled diagnostic for complex multi-agent systems. The source code is freely available: this https URL. 

---
# Sequential Testing on Complex and High Dimensional Data Streams Leveraging Unsupervised Model Errors 

**Authors**: George Stamatelis, Iordanis Koutsopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2611.33238)  

**Abstract**: We propose two general, distribution-free test statistics for binary Sequential Hypothesis Testing (SHT) and Sequential Change Point Detection (SCPD) respectively, in streaming environments where the data (e.g., network traffic metadata) are too complicated for the exact probabilistic modeling required by conventional likelihood ratio-driven tests. Our framework also addresses the challenge of information asymmetry, where data is available only from the null/pre-change regime and not from the alternative/post-change one such as zero-day attacks or emerging disease related regimes. The proposed statistics monitor the errors of a predictive or reconstructive machine learning model trained solely on null/pre-change data, relying on imperfect generalization (the model's performance will degrade when deployed on data from the alternative/post-change class). Under mild assumptions on the model's errors, we establish rigorous analytical guarantees: strict non-asymptotic error control for SHT, and explicit bounds on the Average Run Length (ARL) and the Expected Detection Delay (EDD) for SCPD. Extensive evaluations in target tracking, chaotic dynamical systems, and real-world network security datasets, demonstrate that our framework consistently satisfies prescribed constraints where existing algorithms succumb to inflated missed detections and/or false alarms. Finally, because our methods operate on scalar errors without requiring online retraining or kernel evaluations, they are highly scalable. 

---
# KiRA: Kronecker Input-Gated Rank Adaptation for Parameter-Efficient Fine-Tuning of Pretrained Models 

**Authors**: Akil Ahmad Taki, Shaikh Anowarul Fattah  

**Link**: [PDF](https://arxiv.org/pdf/2611.33236)  

**Abstract**: Full fine-tuning of large pretrained models is expensive, and parameter-efficient fine-tuning (PEFT) is the standard alternative. Most PEFT methods use low-rank updates, whose rank lies far below that of full fine-tuning and is the same for every input. Input-adaptive variants vary the capacity per input but remain within the low-rank bottleneck, and whether a task benefits from input conditioning is not known before training. This paper introduces Kronecker input-gated rank adaptation (KiRA). Instead of the conventional low-rank product, the update $\Delta W$ is the Kronecker product of a frozen random $p\times p$ matrix $A$ and a trainable factor $M(x)$, which overcomes the rank ceiling: because rank is multiplicative under this product, $A$ multiplies the rank of $M(x)$ by $p$ without adding trainable parameters. The factor $M(x)$ is further decomposed as $B_1\,\mathrm{diag}(g(x))\,B_2$, where the input gate $g(x)$, computed once per sample by a network shared across all adapted layers, adjusts the rank of the update to each input in steps of $p$. Where input conditioning is unnecessary, the gate is folded into $B_1$ at deployment. Because the gate changes the rank only in discrete steps, the benefit of input conditioning can be predicted before training from a few hundred labeled inputs. Across six medical imaging and language benchmarks, KiRA attains the highest mean score against five published baselines with 8.7--9.7 times fewer trainable parameters than low-rank adaptation, and the pre-training criterion agrees with the measured benefit of input conditioning on all six tasks. 

---
# Diagnostically Meaningful, Privacy-aware Synthetic Chest CT via Class-Conditional Latent Sampling in Diffusion Autoencoders 

**Authors**: Johannes B. Thalhammer, Kacper Ogórek, Tina Dorosti, Sebastian Peterhansl, Daniel Frey, Zijin Huang, Theresa Hiu, Tobias Lasser, Daniela Pfeiffer, Franz Pfeiffer, Florian Schaff  

**Link**: [PDF](https://arxiv.org/pdf/2611.33234)  

**Abstract**: Large, well-annotated datasets are essential for developing medical artificial intelligence models, yet data sharing is often restricted by privacy regulations. Synthetic images may help address this limitation if they preserve diagnostic information while reducing the risk of reproducing patient data. We developed a diffusion autoencoder-based framework for class-conditional synthetic chest CT generation that incorporates latent-space distance filtering to reduce the likelihood of regenerating training samples.
A diffusion autoencoder was trained on the public RSNA 2020 Pulmonary Embolism (PE) dataset. PE-positive and PE-negative images were synthesized by sampling latent representations from class-specific kernel density estimates. Samples with a nearest-neighbor latent-space distance below a predefined threshold were rejected. To assess diagnostic utility, ResNet-50 classifiers were trained across 10 independent runs using real, synthetic, or mixed real-synthetic datasets. Performance was evaluated on a held-out real test set using the area under the receiver operating characteristic curve (AUC), with DeLong 95% confidence intervals (CIs).
Models trained on synthetic data achieved an AUC of 0.862 (95% CI, 0.851-0.874), only slightly lower than models trained on real data (AUC = 0.872, 95% CI, 0.861-0.883) and mixed real-synthetic data (AUC = 0.865, 95% CI, 0.855-0.876). These differences were statistically significant (p < 0.05) but small in absolute terms. A nearest-neighbor copy-detection analysis of 38,000 synthetic slices found no direct copies of real training images.
These findings indicate that the synthetic CT images preserve diagnostically meaningful information for image-level PE classification while incorporating measures intended to enhance privacy awareness and reduce similarity to real patient data. 

---
# How Many Samples Is RL Worth? Best-of-N versus RL Post-Training for Text-to-Image Models 

**Authors**: Salil Bhatnagar  

**Link**: [PDF](https://arxiv.org/pdf/2611.33232)  

**Abstract**: Reinforcement learning (RL) on a reward that can be checked automatically has become a common last step in the training of text-to-image models. A simpler way to get more correct images is best-of-N: draw N images from the base model and keep the one that a picker prefers. We ask how many base model samples one RL sample is worth, and call this number the exchange rate. We measure it for Stable Diffusion 3.5 Medium and its public Flow-GRPO checkpoint on 398 new prompts in the style of GenEval, the benchmark behind the RL reward, with human labels on every prompt and two automatic judges other than the reward. The answer depends on the prompt. Four fifths of the prompts do not ask for a position (for example, one object left of another). On these, the base model catches up: best-of-16 with an off-the-shelf vision-language model (VLM) as the picker is about as good as one RL image. A person who looks through up to 3 base images finds a correct one as often as one RL image is correct. On the prompts that ask for a position, RL is far ahead: a person finds a correct image among 8 base images for only 52% of the prompts, against 92% for one RL image. Over all prompts, our primary test, which we fixed in advance, finds best-of-16 with the VLM picker 9.3 points behind one RL image when both automatic judges must pass an image. People see a smaller gap, 5.5 and 6.5 points in two checks. FLUX.1-dev shows a similar gap in the same test. RL also solves 20 prompts that the base model misses in 128 tries, and 1,024 tries recover half of them. The automatic judges are wrong in opposite directions: a detector-based judge like the reward, as used in our primary test, reports a larger gap than people see, and VLM judges report little or none. Comparisons of RL with best-of-N should therefore check the selected images by hand. 

---
# The AI Disclosure Paradox: Operational Disclosure and the Rise of Corporate AI Narratives 

**Authors**: Abdullah Karasan  

**Link**: [PDF](https://arxiv.org/pdf/2611.33201)  

**Abstract**: AI-related corporate disclosure can expand while becoming less concentrated on current use. Across U.S. SEC filings from 2017 to 2025, AI-related passage intensity increased 6.3-fold between 2017-2022 and 2023-2025, whereas corrected operational-disclosure intensity increased 3.3-fold. The operational share fell from 25.76% to 13.45%; approximately 89% of the net increase in passage intensity fell outside the operational category. The decline persists within sectors and filing types and in 10-K-only analysis. These estimates combine frozen language-model predictions for 26,279 AI-related passages with an audit of 1,000 annotated passages. The findings distinguish growth in organizational attention from growth in implementation-specific claims: more operational disclosure can coexist with a substantially lower operational share. Operational disclosure was thus diluted, not displaced, by nonoperational disclosure. Reading corporate technology narratives therefore requires attention to both disclosure intensity and the organizational state described. 

---
# UniMoE-CP: Unified Conditional Waveform Prediction Across Measured Optical Wireless Links 

**Authors**: Yuning Zhou, Ruizhe Jin, Yunkai Wang, Zhilan Lu, Kejia Bian, Chao Shen, Feng Bao, Jianyang Shi, Ziwei Li, Junwen Zhang, Nan Chi  

**Link**: [PDF](https://arxiv.org/pdf/2611.33198)  

**Abstract**: Shared prediction across measured optical wireless links requires a common model to accommodate link-dependent waveform dynamics. This paper studies unified learning for underwater optical, 100-m visible-light communication and 4.3-km free-space optical links. We formulate received-waveform forecasting from historical observations and known future transmitted excitation using one shared parameter set. UniMoE-CP combines a shared pool of temporal experts, window-dependent routing and spectral covariate fusion with an autoregressive decoder. The design permits conditional specialization while reusing expert parameters across the measured links. Pooled-training benchmarks evaluate each link separately and report an aggregated result using normalized mean square error (NMSE) and normalized mean absolute error (NMAE). The reported average NMSE reductions are 29.9% for 100-m VLC, 29.1% underwater and 21.4% for 4.3-km FSO. The aggregated average NMSE decreases from 0.1645 for the best baseline to 0.1178 for UniMoE-CP. 

---
# Extraneous Cognitive Load in Large Language Models 

**Authors**: Arushi Gupta, Sheng Liu, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2611.33197)  

**Abstract**: Cognitive Load Theory (CLT) shows that human problem-solving is affected by both intrinsic load, arising from task difficulty, and extraneous load, arising from how task information is presented. Large language models (LLMs) show analogous effects: presentation alone can affect model performance, but these effects have not yet been systematically isolated from task difficulty. To address this gap, we introduce CogLoadBench, a controlled benchmark that varies five CLT-inspired presentation factors across three reasoning task families. Across models, we find that higher extraneous load conditions often reduce accuracy by either adding competition or complicating the reasoning path. To support diagnosis and mitigation of load-related effects, we propose the Extraneous Load Score (ELS), a prompt-time metric computed from activations that estimates a model's extraneous load, and show that it generalizes across reasoning task types and load levels. We further show that shifting model representations toward lower-load directions at inference time can improve average model performance without changing prompt content. 

---
# Verified experience improves materials prediction with large language models 

**Authors**: Yiming Zhang, Koji Tsuda  

**Link**: [PDF](https://arxiv.org/pdf/2611.33195)  

**Abstract**: Materials discovery is an iterative loop: experiments and calculations verify predictions, and each outcome can inform the next. Yet most predictors are specialized single-task models fixed after training. Large language models have recently emerged as predictors across heterogeneous tasks but generally underperform specialized models, and retrieval-augmented variants discard their own outcomes. Benchmarks also score queries in isolation, leaving improvement from outcomes untested. Here we construct MatMemBench, a sequential benchmark of 22 materials tasks that reveals outcomes only after prediction and tests online adaptation, memory retention and cross-task transfer. We present MatRecall, a framework that lets large language models store verified predictions, with evidence, outcome and error, as episodes and recall them without retraining. MatRecall improved median performance by 62% over a memory-free model, exceeded specialized models on 11 tasks, remained useful after feedback stopped and transferred between related tasks. Verified experience could become a reusable resource for self-improving discovery loops. 

---
# Formalization, Transparency, and Collusion in Open-Source Game Theory 

**Authors**: Colomban Duclaux, Riccardo Formenti, Pepijn Cobben, Terry Jingchen Zhang, Bernhard Schölkopf, Zhijing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2611.33193)  

**Abstract**: Cooperation among AI agents will undermine alignment protocols when it takes the form of collusion. Under these protocols, agents often have some information about one another, and how much they have may determine whether they collude. Open-source game theory studies the extreme case, in which each program reads its opponent's source code and acts based on what it can prove about it. We develop formal foundations for this theory: we specify a concrete proof system, resolve an open problem concerning DupocBot and CupodBot through a transposition argument, and mechanize the framework in Lean 4; to our knowledge, this is the first mechanization of resource-bounded open-source game theory. We then extend the framework to the more realistic setting of partial transparency, where each program receives only partial information about its opponent. In Prisoner's Dilemma experiments, where mutual cooperation models collusion, greater transparency increases both the cooperation sustainable in equilibrium and the cooperation selected by replicator dynamics. This suggests that controlling how much agents can know about one another matters for alignment protocols. 

---
# Dynamics of Collective Diversity in Human-AI Co-Creation for Creative Tasks 

**Authors**: Manh Hung Nguyen, Chao Wen, Sebastian Tschiatschek, Adish Singla  

**Link**: [PDF](https://arxiv.org/pdf/2611.33192)  

**Abstract**: Generative AI systems are increasingly deployed at scale to support human users in creative tasks across a variety of domains. However, as more users adopt the same AI agent, the resulting artifacts may become increasingly homogeneous, leading to a collapse in collective diversity. Prior work has documented this risk, but lacks a systematic framework for understanding how collective diversity changes as AI adoption scales across co-creation settings. In this work, we conduct a large-scale crowdsourcing study to understand and model the dynamics of collective diversity, quality, and effort in human-AI co-creation across two creative domains: Creative Hypothesis Generation and Creative Story Writing. We find that current AI agents used in co-creation can improve artifact quality and reduce effort, but at the cost of declining collective diversity as adoption increases. We further show that more fine-grained human-AI interaction, along with more diverse agent policies, can mitigate this collapse. Finally, we fit predictive models that quantify how collective diversity scales with adoption across co-creation settings, providing insights for designing AI agents that aim to preserve collective diversity. 

---
# Semantic Density Does Not Predict Agent-Memory Collapse 

**Authors**: Saksham Gupta, Yash Choudhary, Arpit Upadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2611.33184)  

**Abstract**: Persistent textual memory can improve LLM agents, but repeated updates can erode task performance. We ask whether memory-store statistics warn before held-out accuracy declines. We registered mean pairwise cosine similarity ($\rho$) and 13 additional geometry, retrieval, and graph statistics on 805 snapshots from 13 memory methods across ALFWorld, AppWorld, ScienceWorld, WebShop, and two ARC-AGI settings. These methods store trajectories, rules, workflows, strategies, or linked records, using selective retrieval or whole-store prompting. No such statistic reliably predicted early signs of task accuracy decline, and $\rho$ had zero median lead. With the checkpoint after peak accuracy as a secondary decline point, measures of highly similar pairs and nearest-neighbor crowding changed earlier but did not consistently outperform stored-item count. We trained classifiers on this corpus and applied them unchanged to PropWorld runs of up to 5,000 crafting tasks. They falsely flagged 56.9% of checkpoints from size-limited stores with stable accuracy and detected only 18% of checkpoints after task success fell to zero. Thus, global semantic density did not provide a reliable warning, while local crowding did not transfer consistently across memory designs. In these long-horizon runs, size-limited stores remained stable, while the sole failure arose when unbounded whole-store prompting exceeded model input capacity, which token growth predicted directly. 

---
# REI: Learning Ideation from Repository Evolution 

**Authors**: Muyu Zou, Xiang Feng, Yufei Xu, Shanshan Ye, Zulong Chen, Lingyu Mu, Jinxin Hu, Jing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.33183)  

**Abstract**: As AI systems become increasingly capable of executing and evaluating experiments, ideation is emerging as a bottleneck in automated scientific and engineering iteration. In many real-world domains, useful ideas are sparse and their validation is costly, making direct online exploration inefficient. We observe that repository evolution provides a scalable source of reusable technical experience: code changes record concrete technical hypotheses, while subsequent revisions, adoption, and reuse provide signals about their utility. Based on this observation, we propose REI, a Repository Evolution-driven Ideation framework that learns reusable ideation policies from repository evolution, injecting domain-specific technical experience into model parameters and reducing reliance on costly trial-and-error. We further identify a hidden failure mode in quality-driven ideation optimization: semantic diversity collapse. Although generated ideas can remain lexically diverse, they increasingly concentrate in a small number of dominant method families, limiting cross-task generalization and long-tail exploration. To address this problem, we introduce Semantic Prior Optimization and Debiasing (SPOD), a diversity-preserving alignment mechanism that makes method identity explicit, corrects category-level reward bias, and limits redundant accumulation within the same method family while retaining strong representatives. Experiments on public benchmarks and a real-world industrial setting demonstrate improvements in ideation quality, method diversity, cross-task generalization, and long-tail exploration. 

---
# Neighbor Crowding Predicts Feature Stability in Sparse Autoencoders 

**Authors**: Denis Federico Lim, Naveen Dasari, George Flint, Kevin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2611.33182)  

**Abstract**: Sparse Autoencoders (SAEs) offer an approach to identify language model representations by breaking apart polysemantic features into more sparse, interpretable ones. Yet, it is challenging to identify which of the learned features are actually stable, rather than artifacts of initialization. One approach is to train multiple SAE seeds, and compare the learned features across seeds by cosine similarity, but this is notably costly. This raises the question of whether stable features can be identified without the computational expense of training multiple seeds. In this work, we examine cross-seed feature stability across differing levels of training tokens, and demonstrate that in TopK SAEs, it is highly predictable using six computationally cheap statistics derived from a single training run. Neighbor crowding, a geometric property of the dictionary, is the strongest single predictor, with features that sit in dense regions of the unit sphere more likely to be stable than isolated ones. 

---
# Psychophysics of Duration Perception in Multimodal Large Language Models 

**Authors**: Auss Abbood, Zihao Fu, Nigel Collier  

**Link**: [PDF](https://arxiv.org/pdf/2611.33176)  

**Abstract**: Studies of time perception in language models focus mainly on text-based settings. Yet multimodal models are increasingly used for tasks that require perceiving the passage of time, making the limits of their temporal perception important to understand. We measure the limits of duration perception from audio against a human baseline to address this gap. Drawing from psychophysics, we operationalise duration perception as a melody recognition task, testing when a melody becomes unrecognisable as it is slowed down. Of 21 tested models, most cannot recognise well-known melodies without labelled in-context examples. With in-context examples, seven recognise melodies reliably, most peaking within the median human recognition band. Generalisation is limited: (i) recognition peaks shift with in-context example speed, (ii) melody titles induce prediction bias, and (iii) providing examples with scientific pitch notation does not prevent accuracy declining at long note durations for most models. In humans, the recognition window is a property of the perceiver; in current models, it is largely a property of the prompt. 

---
# Quantum Machine Learning for Signature Forgery Detection 

**Authors**: Nabila, Md Farhan Shahriyar, Nowshin Tabassum Puspa  

**Link**: [PDF](https://arxiv.org/pdf/2611.33163)  

**Abstract**: Offline handwritten signatures remain a primary behavioural biometric in legal and financial documents. Detecting skilled forgeries from a static raster image is hard because pen pressure, timing and stroke order are missing. This work studies a family of quantum machine learning pipelines for offline signature forgery detection on a public Kaggle corpus of images split into two classes. We pair a ZZ feature map quantum kernel with three classical learners: K-Nearest Neighbours, Support Vector Machine and Random Forest. We then propose a deeper hybrid that uses EfficientNetB0 as a frozen feature extractor, reduces the resulting 1280 dimensional representation to a qubit register sized subspace through a quantum compatible principal component projection and decides class membership with a nearest neighbour rule. On a single held out test fold the proposed hybrid reaches 92.62 percent accuracy. On stratified five fold cross validation the same pipeline yields a mean accuracy of 91.26 percent with a standard deviation of 1.13 percent. The pipeline keeps a quantum embedding step that is portable to NISQ hardware. We report a controlled ablation across six quantum variants, confusion matrices, a t SNE projection and per class precision, recall and F1 score. 

---
# Attributing Agentic Failure in Reasoning Vision-Language Models with Interactive 2D Environments 

**Authors**: Pranav Guruprasad, Sean Rivera, Helen Lu, Arushi Jain, Hangliang Ren, Harshvardhan Sikka  

**Link**: [PDF](https://arxiv.org/pdf/2611.33162)  

**Abstract**: Agentic benchmarks report a success rate that confounds distinct causes of failure: misreading the environment, never working out how components of the environment operates, operating the components in the wrong order, and failing to recover from a mistake, each of which calls for a different fix. We present an early preview of MultiNet v2.0-Gridworld - an interactive maze environment built for attribution rather than ranking. Four different types of mechanisms, namely key-door pairs, switch-gate pairs, distractors and decoys, and ordered combinations of these mechanisms, can be added, modified, or removed independently while the renderer, the action space, and the scoring function stay fixed. A breadth-first search over the joint state of position, facing direction, inventory, and mechanism state gives every instance an exact optimal action count, which sets its difficulty and grades unfinished episodes. We fix the evaluation protocol on 15 held-out mazes over 540 episodes, then evaluate three vision-language models on 50 mazes. They solve 6 of 150 episodes, 45 of the 50 mazes were not solved by any model, and no episode containing a switch-gate mechanism was solved. 

---
# Show Your Work: Can Small Open Models Read SEC Filings with Evidence a Machine Can Check? 

**Authors**: Krishna Nandula  

**Link**: [PDF](https://arxiv.org/pdf/2611.24990)  

**Abstract**: A fluent answer from a language model is not a verified one. We ask whether small open models that run on a single consumer GPU can extract numbers from SEC filings with evidence a machine can check: the page and a verbatim quote containing the number. We introduce Filing-Evidence, a benchmark of 1,000 questions over 90 recent 10-K and 10-Q filings of 45 US companies, with answer keys taken from the filers' inline XBRL tags. We evaluate ten instruct models from ten developers (7-21B parameters) on one RTX 4090 in three setups: plain, check-and-retry, and the model's own reasoning mode. Accuracy with checkable evidence ranges from 17.1% to 82.7% in the plain setup, and the best result is 84.5% (Qwen3.5-9B). An automatic checker rejects many wrong answers, but between 16 and 167 answers per 1,000 give a wrong value for an answerable question and still pass every check. In the plain setup, 57% of these are scale or unit errors and 30% take another period or column of the right line. Reasoning significantly reduces such errors for 4 of 5 models that offer it. Retrying after a failed check raises accuracy by up to 3.9 points but never reduces them. Questions, model outputs, code and protocol are public. 

---
