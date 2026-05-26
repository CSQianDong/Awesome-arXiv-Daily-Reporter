# CausaLab: A Scalable Environment for Interactive Causal Discovery Toward AI Scientists 

**Authors**: Junlin Yang, Dylan Zhang, Xiangchen Song, Qirun Dai, Xiao Liu, Yuen Chen, Aniket Vashishtha, Jing Shi, Chenhao Tan, Hao Peng  

**Link**: [PDF](https://arxiv.org/pdf/2605.26029)  

**Abstract**: We introduce CausaLab, a scalable environment for evaluating interactive causal discovery by LLM agents. Unlike prior evaluations, CausaLab evaluates both whether an agent can solve a problem using causal evidence and whether its answer is supported by a correct hypothesis about the underlying causal mechanism. Each episode places an agent in a synthetic laboratory: it receives prior measurement records, intervenes on a manipulator crystal, and predicts the resonance frequency of a held-out reactor crystal governed by the same mechanism. The hidden data-generating process is a randomly sampled structural causal model (SCM), so success requires recovering both a causal graph and structural equations rather than recalling prior knowledge. CausaLab also includes a domain-specific language that records the agent's evolving SCM hypothesis, making trajectories inspectable and comparable with ground truth. Experiments show a persistent gap between prediction and mechanism recovery: in the purely observational 6-node setting, GPT-5.2-high reaches 92% task accuracy but only 0.471 all-edge $F_1$. This observation further motivates our exploration of different interaction strategies: Mixed observation--intervention strategies improve structural fidelity: in the mixed 6-node setting, GPT-5.2-high achieves 80% on both task accuracy and all-edge $F_1$. Yet even strong agents struggle to design informative interventions, as pure intervention strategies perform poorly on both task accuracy and all-edge $F_1$. We identify premature stopping as a major weakness of agents, and show that asking the model to verify the consistency between its hypothesis and past data can help mitigate this issue. CausaLab therefore separates predictive success from causal understanding and exposes current LLM agents' limits as experimental causal reasoners. 

---
# VeriTrace: Evolving Mental Models for Deep Research Agents 

**Authors**: Haolang Zhao, Yunbo Long, Lukas Beckenbauer, Alexandra Brintrup  

**Link**: [PDF](https://arxiv.org/pdf/2605.26081)  

**Abstract**: Deep research agents face vast, interdependent, and pervasively uncertain information. Existing systems explore what evolving intermediate representations should look like, but leave their evolution to the LLM's implicit reasoning. Without explicit regulation, the intermediate layer is easily contaminated by mixed-quality information and propagates errors along its dependencies, so model scale often ends up substituting for absent regulation. We argue that an agent's mental model should instead evolve through explicit feedback that continuously aligns task understanding with reality, and identify three regulatory loops: interpretive update, deviation feedback, and schema revision. We realise this in VeriTrace, a cognitive-graph framework that explicitly implements the three loops. Using matched Qwen3.5-27B backbones, VeriTrace improves over the strongest matched baseline by 4.22 pp on DeepResearch Bench (DRB) Insight (1.49 pp Overall) and by 5.9 pp Overall win rate on DeepConsult. With Config-DeepSeek, it achieves the strongest reproducible open-source result on DRB. 

---
# Retrying vs Resampling in AI Control 

**Authors**: James Lucassen, Adam Kaufman  

**Link**: [PDF](https://arxiv.org/pdf/2605.26047)  

**Abstract**: AI coding scaffolds like Claude Code and Codex use \textit{retrying}: blocking actions flagged as risky and continuing the trajectory. We study retrying from an AI control perspective, which treats the model as potentially adversarial. We find that while retrying reduces honest suspicion scores, the untrusted model can exploit monitor rationale to construct sneakier attacks, negating safety gains. We also study \textit{resampling}: drawing multiple samples from the same context, which does not leak exploitable information. We disentangle design choices that previous work on resampling had bundled together. In BashArena, with Claude Opus 4.6 as the untrusted model and MiMo-V2-Flash as the trusted monitor, drawing five samples per step and auditing on the maximum suspicion score raises safety from 61\% to 71\% at a 0.3\% audit budget, at no cost to usefulness. Selectively resampling only the steps that look suspicious on the first draw recovers 6.2 percentage points of the gain while drawing only 10\% as many extra samples. Two of our findings in this setting contradict earlier work on resampling. The first is that auditing based on the maximum across resampled suspicion scores outperforms using the minimum, which is the opposite of what Ctrl-Z found. The second is that executing the least suspicious sample, which is the central mechanism in earlier defer-to-resample protocols, gives only a small empirical safety gain in our setting (+3.9 pp, with the confidence interval overlapping zero). 

---
# Claw-Anything: Benchmarking Always-On Personal Assistants with Broader Access to User's Digital World 

**Authors**: Yusong Lin, Xinyuan Liang, Haiyang Wang, Qipeng Gu, Siqi Cheng, Jiangui Chen, Shuzhe Wu, Feiyang Pan, Lue Fan, Sanyuan Zhao, Dandan Tu  

**Link**: [PDF](https://arxiv.org/pdf/2605.26086)  

**Abstract**: Large language model agents are increasingly envisioned as always-on personal assistants with access to anything relevant in the user's digital world. Yet current systems operate over only narrow slices of that world, limiting context-sensitive reasoning and effective assistance. Existing benchmarks similarly provide only partial user state and therefore fail to capture performance in such a broad, always-on setting. To address this gap, we introduce Claw-Anything, a benchmark that expands agent context along three dimensions: long-horizon activity histories, interdependent backend services, and integrated GUI and CLI interaction across multiple devices. To instantiate this setting, we simulate months of user activity through multi-round event injection, producing complex world states and realistic noise, including irrelevant events and conflicting signals. Agents must reason over rich contextual environments while remaining robust to such noise. This expanded scope also enables the evaluation of proactive assistance, requiring agents to anticipate user needs and deliver timely recommendations. Experiments show that GPT-5.5 achieves only 34.5% pass@1, substantially below prior benchmarks, underscoring a gap between current agent capabilities and the demands of always-on personal assistance. Alongside the benchmark, we release an automated data-generation pipeline that yields 2,000 training environments and improves the base model by 23.7%, demonstrating its utility of scalable data infrastructure. 

---
# MobileGym: A Verifiable and Highly Parallel Simulation Platform for Mobile GUI Agent Research 

**Authors**: Dingbang Wu, Rui Hao, Haiyang Wang, Shuzhe Wu, Han Xiao, Zhenghong Li, Bojiang Zhou, Zheng Ju, Zichen Liu, Lue Fan, Zhaoxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.26114)  

**Abstract**: We present MobileGym, a browser-hosted, lightweight, fully controllable environment for everyday mobile use, targeting interaction fidelity without replicating proprietary backends. It enables two capabilities previously out of reach for everyday apps: verifiable outcome signals through deterministic state-based judging over structured JSON state, and scalable online RL through low-cost parallel rollouts. The full environment state is captured, configured, forked, and compared as structured JSON, and a single server can host hundreds of parallel instances, with about 400 MB memory per instance and about 3 s cold start. A layered state model and a declarative task-definition framework keep state programmability and task creation practical at scale, and a single programmatic judging mechanism delivers both deterministic evaluation verdicts and dense RL rewards. The accompanying MobileGym-Bench provides 416 parameterized task templates, including 256 test and 160 train templates, over 28 apps, with deterministic judges and a structured AnswerSheet protocol that avoids free-text matching failures. In a Sim-to-Real case study, GRPO on Qwen3-VL-4B-Instruct gains +12.8 percentage points on the 256-task test set, and on a 59-task real-device signal subset, real-device execution retains 95.1% of the simulation-side training gain. Project page: this https URL. 

---
# $D^2$-Monitor: Dynamic Safety Monitoring for Diffusion LLMs via Hesitation-Aware Routing 

**Authors**: Aoxi Liu, Yupeng Chen, James Oldfield, Guanzhe Hong, Junchi Yu, Baoyuan Wu, Philip Torr, Adel Bibi  

**Link**: [PDF](https://arxiv.org/pdf/2605.25893)  

**Abstract**: Despite the emergence of diffusion large language models (D-LLMs) as an alternative to autoregressive large language models (AR-LLMs), safety monitoring for D-LLMs remains largely unexplored. Unlike AR-LLMs, D-LLMs generate text through a multi-step denoising process, exposing intermediate hidden representations that may contain safety-relevant information unavailable in standard single-step monitoring setups. Motivated by the suitability of lightweight probes for always-on monitoring, we analyze which trajectory-level signals best indicate when such probes are likely to struggle. We find that the most informative signal is safety hesitation: intermediate hidden states repeatedly falling within a small margin of the probe's decision boundary. The number of such hesitation steps in D-LLM's trajectory predicts probe failure effectively, providing a proxy of sample difficulty. Building on this analysis, we propose $D^2$-Monitor, a bi-level safety monitor for D-LLMs. $D^2$-Monitor adopts a lightweight probe as an always-on monitor to jointly estimate hesitation and perform base classification. When the hesitation level exceeds a threshold, a more expressive but computationally heavier probe is activated. This dynamic routing mechanism allocates monitoring resources efficiently at test time. Evaluated on 3 datasets (WildguardMix, ToxicChat, OpenAI-Moderation) across 4 D-LLMs, $D^2$-Monitor achieves state-of-the-art performance with a compact parameter footprint ($\leq$ 0.85M parameters), and exhibits the best trade-off between effectiveness and efficiency relative to 8 baselines. 

---
# Explore Before You Solve: The Speed--Depth Trade-off in Epistemic Agents for ARC-AGI-3 

**Authors**: Liew Keong Han  

**Link**: [PDF](https://arxiv.org/pdf/2605.25931)  

**Abstract**: We systematically investigate all 25 public ARC-AGI-3 games and find that every one is reachable through non-intelligent strategies: 10 in a single blind step, 5 after one probing action, 1 via repeated ACTION1 presses, 1 via diverse exploration, and 8 via single repeated actions with sufficient budget (50-200 steps). A library-level null-coordinate vulnerability additionally bypasses 18 games in 1 step. This benchmark critique implies the public evaluation set cannot discriminate intelligent exploration from trivial heuristics - the private 55-game evaluation is the only genuine intelligence test. Against this backdrop, we present AERA (Adaptive Epistemic Reasoning Agent), a three-phase (EXPLORE / VERIFY / PLAN) agent achieving RHAE=0.2116 (4/25 solved) on these 25 games with Qwen2.5-0.5B, while random and no-explore baselines score 0.0000. We formalise AERA through a Speed--Depth trade-off framework: under a convexity assumption (proved for a class of environments in the Appendix), RHAE's quadratic form emerges as a second-order penalty for deviating from the Pareto frontier between action efficiency and information gain. Contributions: (i) a benchmark validity analysis showing that current interactive reasoning benchmarks fail to measure the exploration they claim to require, and (ii) the EXPLORE-before-PLAN framework and model-capability x exploration interaction. The linked code track entry achieves RHAE=0.30 on the full 55-game private evaluation. Code: CC0. 

---
# MuCRASP: Multimodal Chain-of-thought Reasoning aware Structured Pruning 

**Authors**: Aritra Dutta, Somak Aditya  

**Link**: [PDF](https://arxiv.org/pdf/2605.25842)  

**Abstract**: Vision-language models (VLMs) increasingly rely on chain-of-thought (CoT) reasoning to solve complex multimodal tasks, but their large parameter sizes make deployment expensive. Structured pruning offers a natural solution; however, existing methods fail to preserve CoT reasoning accuracy in VLMs. We identify two key reasons: (1) CoT consistency depends on sparse transition points (pivot tokens) in the generation trajectory, while existing pruning methods are CoT-agnostic; and (2) pruning methods designed for unimodal LLMs do not account for activation-distribution differences across visual and textual modalities. Motivated by these observations, we propose MuCRASP, a structured pruning framework that targets reasoning-critical components while preserving cross-modal alignment and accounting for layer-wise sensitivity under a global parameter budget. Experiments on four VLMs across three reasoning benchmarks show that MuCRASP consistently preserves reasoning quality under increasing compression. At 30% pruning on Qwen2.5-VL-7B, MuCRASP achieves an LLM-as-a-Judge score of 8.87 versus 7.32 for the strongest baseline on physical reasoning tasks. Furthermore, MuCRASP maintains high reasoning consistency up to 50% pruning, significantly outperforming prior pruning approaches while exhibiting lower perplexity degradation. 

---
# A Deep Dive into Axiomatic Design -- Part I: Problem Formulation 

**Authors**: Aydin Homay  

**Link**: [PDF](https://arxiv.org/pdf/2605.25735)  

**Abstract**: Problem formulation translating customer needs and constraints into a minimum set of independent first-level functional requirements, is arguably the most critical step in every design framework, including axiomatic design yet it is frequently misunderstood or underestimated in practice. This paper focuses exclusively on problem formulation in axiomatic design it clarifies what first-level FRs are (and are not), explains why they should not legitimately vary across designers given the same needs and constraints, and highlights intrinsic difficulties and recurring pitfalls that lead to design failure. The discussion is grounded primarily in Nam this http URL's three books. The Principles of Design, Axiomatic Design Advances and Applications, and Complexity Theory, and it offers practical guidance to help designers formulate well-posed first-level FRs. Finally, the paper briefly revisits problem formulation in the era of large language models and discusses what such tools can (and cannot) contribute at the first level. 

---
# L2IR: Revealing Latent Intent in Graph Fraud Detection 

**Authors**: Jinsheng Guo, Zhenhao Weng, Yibo Liu, Yan Qiao, Meng Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.26040)  

**Abstract**: Graph fraud detection has long depended on Graph Neural Networks (GNNs) to propagate and aggregate information across relational data. A critical obstacle in practice, however, is that fraudsters frequently disguise themselves by forging numerous connections with benign users, causing fraud signals to be progressively diluted during neighborhood aggregation and undermining detection reliability. While recent efforts have used Large Language Models (LLMs) to provide rich semantic cues for fraud detection, the underlying intent behind suspicious connections remains insufficiently explored. Compounding this issue, the scarcity of annotated fraud samples makes it difficult to train detectors that remain robust under heavy camouflage. To address these gaps, we propose L2IR, an LLM-driven Latent Intent Revealing framework for graph fraud detection. By uncovering latent intent from both user behaviors and suspicious connections, L2IR extracts intent-aware representations from raw behavioral traces and reasons about the true purpose behind individual connections, effectively distinguishing supportive links from misleading ones. It further incorporates adaptive self-training to enhance robustness under limited supervision. Evaluations on two real-world datasets characterized by pervasive camouflage demonstrate that L2IR surpasses strong baselines and can function as a plug-in enhancement for a range of GNN-based detectors, improving AUPRC by up to 8.27%. 

---
# From Model Scaling to System Scaling: Scaling the Harness in Agentic AI 

**Authors**: Shangding Gu  

**Link**: [PDF](https://arxiv.org/pdf/2605.26112)  

**Abstract**: This paper studies the next major bottleneck in agentic AI as system scaling, not only model scaling: the design of auditable, persistent, modular, and verifiable architectures around foundation models. We refer to this shift as scaling the harness: treating the structured execution layer around a foundation model as a first-class object of design, evaluation, and optimization. Although recent large language models enable agents to use tools, retrieve information, maintain memory, and execute long-horizon workflows, evaluation remains largely model-centric, often reducing agents to final-task success while treating memory, retrieval, tool use, orchestration, verification, and governance as secondary implementation details. This framing is increasingly inadequate because agent performance emerges from the interaction among the foundation model, memory substrate, context constructor, skill-routing layer, orchestration loop, and verification-and-governance layer. Together, these components form the agent harness, which translates model capability into long-horizon agent behavior. We study scaling the harness through three core bottlenecks: context governance, trustworthy memory, and dynamic skill routing, together with the orchestration and governance mechanisms that coordinate and constrain them. We further outline a research agenda for harness-level benchmarks that go beyond one-shot task success to measure trajectory quality, memory hygiene, context efficiency, communication fidelity, verification cost, and safe evolution over time. To make the discussion concrete, we develop CheetahClaws: this https URL, a Python-native reference harness, and compare it with Claude Code and OpenClaw. Our main claim is that future progress in agentic AI will depend as much on system design as on stronger foundation models. 

---
# AgentHijack: Benchmarking Computer Use Agent Robustness to Common Environment Corruptions 

**Authors**: Jingwei Sun, Jianing Zhu, Yuanyi Li, Tongliang Liu, Xia HU, Bo Han  

**Link**: [PDF](https://arxiv.org/pdf/2605.25707)  

**Abstract**: Autonomous computer use agents that powered by multimodal large language models (MLLMs) are emerging as capable assistants for completing complex digital workflows. However, real-world execution environments are far from ideal: pop-ups, resolution changes, and competing applications frequently interfere with agent perception and control. We introduce AgentHijack, a benchmark designed to evaluate the robustness of computer-use agents under common corruptions, where the uncertainties in dynamic environment disrupt the execution flow without direct adversarial intent. Specifically, AgentHijack introduces 9 configurable common corruptions to replicate realistic imperfect scenarios. We evaluate a variety of desktop tasks that utilize MLLM-based agents and discover that even minor instances of corruption can result in substantial performance degradation, which emphasizes the fragility of agents and underscores the necessity of robustness evaluation. Afterward, we propose AgentHijack-Agent, a framework that integrates an action generator with enhanced grounding capabilities and an onlooker responsible for behavior summarization and environment checking. Extensive experiments validate its effectiveness. Our code, environment, baseline models and data are publicly available at: this https URL. 

---
# CUA-Gym: Scaling Verifiable Training Environments and Tasks for Computer-Use Agents 

**Authors**: Bowen Wang, Dunjie Lu, Junli Wang, Tianyi Bai, Shixuan Liu, Zhipeng Zhang, Haiquan Wang, Hao Hu, Tianbao Xie, Shuai Bai, Dayiheng Liu, Que Shen, Junyang Lin, Tao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.25624)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has driven breakthroughs in domains such as math, tool-use, and software engineering, yet its extension to computer-use agents (CUAs) has been bottlenecked by the scarcity of scalable training data with deterministic rewards. Constructing such data for CUAs requires consistent task instruction, executable environment, and verifiable reward. However, hand-curated benchmarks achieve high reward fidelity but cover few applications and LLM-as-judge-based datasets scale broadly but lack reliable verification. We present CUA-Gym, a scalable pipeline that co-generates task instructions, environment states, and reward functions. Concretely, a Generator agent constructs the initial and golden environment states, and a separate Discriminator agent writes the reward function from the task specification. An orchestrator agent drives the two through iterative rounds upon execution. Generated tuples then pass a final filter combining LLM majority voting and agent rollouts, ensuring quality beyond the per-task adversarial loop. To address the scarcity of training environments, we further synthesize CUA-Gym-Hub, a broad suite of high-fidelity mock web applications grounded in real-world software-use distributions, expanding the scale of CUA RLVR data by magnitude. Using this pipeline, we construct CUA-Gym, a dataset of 32,112 verified RLVR training tuples grounded in 110 environments. Trained with GSPO on CUA-Gym, our CUA-Gym-A3B and CUA-Gym-A17B achieve 62.1% and 72.6% on OSWorld-Verified, outperforming prior open-source CUAs at comparable scales, with performance scaling smoothly in both data volume and environment diversity. The same checkpoints also improve on the held-out WebArena benchmark, indicating transfer beyond the training environments. We will open-source the full synthesis pipeline, dataset, CUA-Gym-Hub environments, and models. 

---
# Behind EvoMap: Characterizing a Self-Evolving Agent-to-Agent Collaboration Network 

**Authors**: Qiming Ye, Peixain Zhang, Yupeng He, Zifan Peng, Gareth Tyson  

**Link**: [PDF](https://arxiv.org/pdf/2605.25815)  

**Abstract**: Agent-to-Agent (A2A) networks enable autonomous AI agents to collaborate by sharing reusable problem-solving instructions. However, how these decentralized ecosystems operate in practice remains largely unexplored. We present the first large-scale empirical study of EvoMap, a prominent A2A collaboration network. By analyzing over 1.5M assets and 128K agents, we show how design choices that prioritize scalable growth introduce trade-offs in reusability, evolution, and auditability. First, EvoMap's credit economy rewards agents for publishing valuable assets. Although this design encourages participation at scale, rewards are tied primarily to publication rather than adoption. This leads agents to mass-produce assets to accumulate credits. As a result, 98% of assets are never reused, while rewards become highly concentrated among a small fraction of agents. Second, EvoMap employs an algorithm (referred to as GDI) to score and rank the quality of these shared assets. We demonstrate that this scoring system is flawed: rather than measuring objective performance, an asset's rank is heavily dictated by unverified, self-reported metadata (e.g., claimed lines of code modified). This allows agents to trivially manipulate their asset's scores. Finally, EvoMap relies on agents to provide local execution logs as evidence that uploaded assets function correctly. Because these validations are not independently verified, over 84% of approved assets bypass quality checks using vacuous tests (e.g., this http URL). Our findings show that future A2A collaboration networks cannot rely on unverified self-reporting alone. Scalable collaboration requires mechanisms that balance open participation with verifiable execution and trustworthy evaluation. 

---
# Detecting Unfaithful Chain-of-Thought via Circuit-Guided Internal-External Discrepancy 

**Authors**: Xu Shen, Zhen Tan, Song Wang, Pingjun Hong, Rui Miao, Xin Wang, Tianlong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.25603)  

**Abstract**: Chain-of-thought (CoT) reasoning improves the problem-solving ability of large language models (LLMs), but generated reasoning traces may not faithfully reflect the model's actual decision process. Existing CoT unfaithfulness detectors mainly rely on external signals from generated rationales, such as textual plausibility or answer consistency, while overlooking evidence from the model's internal computation. Although recent circuit tracing methods provide a way to obtain model-internal evidence by tracing how information flows through model components during reasoning, constructing full reasoning circuits for long CoTs is costly and difficult to scale. To address these challenges, we propose Circuit-guided Internal-External Discrepancy Scorer (CIE-Scorer), a framework for instance-level CoT unfaithfulness detection. The key idea is that faithful reasoning traces should align with the model's computational process, whereas unfaithful traces may diverge from it. CIE-Scorer efficiently traces compact sentence-level circuits from informative reasoning tokens, constructs internal and external reasoning graphs, and measures their discrepancy using Fused Gromov--Wasserstein distance. Experiments on four datasets from FaithCoT-Bench show that CIE-Scorer achieves state-of-the-art performance while reducing the cost of circuit construction, demonstrating the effectiveness of combining mechanistic interpretability signals with external reasoning traces for CoT unfaithfulness detection. 

---
# Beyond Query Memorization: Large Language Model Routing with Query Decomposition and Historical Matching 

**Authors**: Bo Lv, Jingbo Sun  

**Link**: [PDF](https://arxiv.org/pdf/2605.25558)  

**Abstract**: Optimizing the trade-off among predictive performance and computational cost is a central focus in the deployment of Large Language Models (LLMs). Current routing methods primarily rely on direct mapping from queries to models based on surface-level features, making them susceptible to the memorization trap and leading to poor generalizability on out-of-distribution (OOD) data. In this paper, we propose DecoR, a novel routing framework that recasts the routing task as a matching process of sifting similar queries from historical logs, effectively mitigating the memorization trap. To enhance matching accuracy, we introduce a query capability deconstruction method that decouples linguistic surface forms from task-intrinsic requirements, directing matching toward capability dimensions to ground decisions in essential task attributes. Furthermore, we develop CodaSet, a comprehensive benchmark for assessing routing generalization, where experimental results demonstrate that DecoR maintains superior accuracy while substantially lowering inference costs across both in-distribution and OOD settings. All the codes and data are available at this https URL. 

---
# StructBreak: Structural Cognitive Overload-Induced Safety Failures in MLLMs 

**Authors**: Yang Luo, Xinran Liu, Tiantian Ji, Zhiyi Yin, Lingyun Peng, Shuyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.25534)  

**Abstract**: Multimodal Large Language Models (MLLMs) excel at structural reasoning yet suffer from a sharp logical brittleness in structural consistency. We term this phenomenon Structural Cognitive Overload (SCO), a byproduct of the contention between deep reasoning and safety alignment. However, prior work has predominantly targeted typographic and pixel-level perturbations, leaving the study of SCO largely unexplored. To this end, we propose StructBreak, an automated end-to-end framework designed to quantify SCO. By leveraging StructBreak, we uncover a novel higher-order cognitive overload attack paradigm; notably, this attack operates under a practical black-box setting, requiring no internal model access. Consequently, we utilize this framework to establish a comprehensive benchmark spanning ten diverse threat scenarios. Empirical evaluations on six leading MLLMs reveal that SCO readily triggers toxic generation, yielding a 92% average ASR (up to 97% on Gemini 2.5). To elucidate the mechanism of SCO, we further conduct model-level interpretations spanning attention dynamics, latent space topology, and geometric analysis. Our findings reveal that StructBreak acts as a novel structural channel to circumvent safety filters. Furthermore, the limited efficacy of inherent safety mechanisms underscores that current alignment paradigms are insufficient for the era of complex multimodal reasoning. 

---
# ATWL: A Formal Language for Representing, Comparing, and Reusing Visual Analytics Workflows 

**Authors**: Natalia Andrienko, Gennady Andrienko, Jürgen Bernard, Michael Sedlmair  

**Link**: [PDF](https://arxiv.org/pdf/2605.25489)  

**Abstract**: Visual analytics (VA) workflows are inherently complex, involving data transformation, feature engineering, visual representation, and human interpretation. They are typically described in unstructured prose, hindering systematic comparison, reuse of proven strategies, and training of novices. We present Artifact-Transform Workflow Language (ATWL), a domain-agnostic, declarative language that formally represents VA workflows by capturing their structure and underlying analytical intent. ATWL is built upon a modular ontology of eight artifact types (entities, features, arrangements, visualisations, patterns, models, knowledge, specifications) and transforms characterised by standardised intents (e.g., define-unit, characterise, contextualise, abstract). To show that formalisation effort need not impede adoption, we extract workflows from research papers through supervised interaction with LLM agents, reducing the human role to review and refinement. Using this process, we constructed a library of seventeen ATWL workflows from published VA papers. Cross-workflow analysis reveals structural regularities -- a recurrent meta-structure, recurring motifs, reusable building blocks, diverse iterative strategies, and cross-domain equivalences -- that remain invisible in prose. We further evaluate practical utility through a controlled experiment in which the same LLM addressed two analytical problems with the library supplied either as original papers or as ATWL representations. Both forms enabled useful recommendations, but the formal representation systematically added explicit iteration structure, typed data flow, fragment-level adaptation provenance, and compactness supporting scaling beyond what prose libraries can fit in an LLM's context. ATWL enables a transition from narrative descriptions to formally represented, comparable, and reusable analytical knowledge. 

---
# Credit Assignment with Resets in Language Model Reasoning 

**Authors**: Ankur Samanta, Akshayaa Magesh, Ayush Jain, Youliang Yu, Daniel Jiang, Kavosh Asadi, Daniel Jiang, Kaveh Hassani, Paul Sajda, Jalaj Bhandari, Yonathan Efroni  

**Link**: [PDF](https://arxiv.org/pdf/2605.25507)  

**Abstract**: Contemporary reinforcement learning with verifiable reward methods post-train language models on multi-step reasoning by assigning a single outcome reward uniformly across all tokens in a trajectory. Such uniform assignment ignores which steps contributed to success or failure. Improving credit assignment can address this limitation by enabling targeted refinement of faulty reasoning steps, rather than updating entire trajectories uniformly. Resets are one such simple mechanism, enabling more precise credit assignment by returning to an intermediate state and resampling counterfactual continuations, so that outcome differences can be attributed to decisions made at that point. We propose two such methods: Random-Reset Policy Optimization (RRPO), where reset states are drawn randomly from reasoning steps, and Self-Reset Policy Optimization (SRPO), where the model self-localizes the erroneous step in an incorrect trajectory and resets there. We analyze these methods within the Conservative Policy Iteration (CPI) framework. Extending CPI with a credit-assignment oracle that targets improvable states yields provable improvements over random resets. Across models and reasoning benchmarks, SRPO consistently outperforms standard GRPO and RRPO by sampling multiple suffix continuations at a self-localized reset and learning from their rewards, using only the model itself with no external supervision. 

---
# CODESKILL: Learning Self-Evolving Skills for Coding Agents 

**Authors**: Yanzhou Li, Yiran Zhang, Xiaoyu Zhang, Xiaoxia Liu, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.25430)  

**Abstract**: Coding agents produce rich trajectories while solving software-engineering tasks. To enable agent self-evolution, these trajectories can be distilled into reusable procedural skills that compactly encode experience to guide future behavior. However, existing skill construction and maintenance methods often rely on fixed prompts and heuristic update rules, leaving it unclear how knowledge should be selected, abstracted, and maintained to best serve downstream agents. We propose CODESKILL, an LLM-based framework that reformulates skill extraction and skill-bank maintenance as a learnable management policy. CODESKILL extracts multi-granularity procedural skills from coding-agent trajectories, evolves skills with new experience, and maintains a compact skill bank for future task solving. We train CODESKILL with reinforcement learning, using a hybrid reward that combines dense rubric-based skill-quality feedback with sparse verifiable execution feedback from the frozen downstream agent. Experiments on EnvBench, SWE-Bench Verified, and Terminal-Bench 2 show that CODESKILL improves average pass rate by 9.69 over the no-skill baseline and by 4.01 over the strongest prompt-based or memory baseline, while maintaining the skill bank at a stable size during iterative construction. 

---
# Towards end-to-end LLM-based censoring-aware survival analysis 

**Authors**: Yishu Wei, Hexin Dong, Yi Lin, Jiahe Qian, Yi Liu, Yifan Peng  

**Link**: [PDF](https://arxiv.org/pdf/2605.25399)  

**Abstract**: Objective: Survival analysis is central to medical prediction, yet large language models (LLMs) are rarely used as end-to-end survival models because censoring prevents straightforward supervised fine-tuning. Here we present LLMSurvival, a framework that enables censoring-aware survival analysis with unmodified LLMs operating directly on tabular clinical data.
Materials and Methods: LLMSurvival reformulates time-to-event prediction as pairwise ranking among comparable subjects, and derives test-time risk by aggregating comparisons against anchor individuals from the training cohort.
Results: Across two clinical tasks (ICU mortality prediction in MIMIC-IV and fragility fracture prediction in a NewYork-Presbyterian/Weill Cornell Medicine cohort), LLMSurvival improves overall concordance over Cox proportional hazards modeling by 3.1% for ICU mortality and 0.5% for fracture risk, 2.1% on average for ICU mortality and 2.8% for fracture risk over three established deep learning survival models.
Discussion: The results show that survival modeling with censoring can be made compatible with LLM fine-tuning through comparison-based reformulation. The framework demonstrates high portability and superior performance over expert curated scores like SAPS-II and FRAX scores across diverse clinical context. Furthermore, the framework supports local deployment, as compact, publicly available base models provide sufficient performance.
Conclusion: The LLMSurvival framework serves as a proof of concept for an integrated, censoring-conscious approach to survival analysis via LLMs. 

---
# Insuring Every Action: An Authority Frontier Framework for Runtime Actuarial Control of Autonomous AI Agents 

**Authors**: Hao-Hsuan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.25632)  

**Abstract**: Autonomous AI agents increasingly issue side-effect-bearing actions: database mutations, refunds, payments, external commitments. We propose the Actuarial Action Interface (AAI), a deterministic runtime contract that prices each such action against a contractually fixed safe default under a time-consistent risk mapping, and gates execution against a per-boundary reserve capital budget. We then develop the Authority Frontier, an evaluation primitive measuring how much autonomous authority the runtime releases at each level of reserve capital. The framework provides (i) a deterministic quote-bind-commit protocol with toll-bounded capability tokens; (ii) a universal seven-class action taxonomy mapping heterogeneous tool calls to comparable authority units; (iii) replay determinism and pathwise reserve coverage under alpha-spending; (iv) cross-domain normalization via full reserve demand C_full and capital metrics Capital@k. We instantiate AAI across four agentic environments (database mutation, customer-service refund, and the public tau-bench retail and airline tool-use traces) and report a live Postgres panel in which three Azure-hosted models propose actions through the same contract. The frontier exhibits a common low-reserve refusal and intermediate-release pattern across domains, with saturation only where the budget grid reaches full reserve demand; required reserve capital varies by 22x (Capital@50 from 289 to 6457). The framework does not force domains into the same shape; it surfaces each domain's actuarial geometry. In the live panel the contract prevents realized loss across all three models at low budget while differing in underwriting persistence under denial: model identity is an actuarial underwriting variable. The contribution is a benchmark-ready evaluation framework for runtime actuarial control of autonomous-agent side effects. 

---
# Security of OpenClaw Agents: Fundamentals, Attacks, and Countermeasures 

**Authors**: Yuntao Wang, Jianle Ba, Han Liu, Yanghe Pan, Jintao Wei, Zhou Su, Tom H. Luan, Linkang Du  

**Link**: [PDF](https://arxiv.org/pdf/2605.25435)  

**Abstract**: The rapid evolution of large language model (LLM)-driven autonomous agents has given rise to OpenClaw, a new class of open-source agent frameworks that operate as continuously running, skill-augmented systems with persistent memory, multi-channel interaction, and high degrees of autonomy. Such capabilities enable OpenClaw agents to autonomously execute complex, multi-step tasks and interact seamlessly with external applications, but simultaneously introduce a substantially enlarged attack surface. In particular, the combination of high-privilege operations and persistent memory exposes OpenClaw agents to various emerging threats, including skill poisoning, cognitive manipulation, multi-agent cascading failures, and supply-chain vulnerabilities. In this survey, we present a comprehensive study of the security landscape of OpenClaw agents. We first examine the general architecture and key characteristics that distinguish OpenClaw agents from traditional AI agent systems. We categorize existing security and privacy threats into a layered framework and analyze how vulnerabilities arise during agent reasoning, action execution, and external interaction. Representative defense mechanisms are also reviewed to draw the current defense landscape. Finally, several unresolved issues related to the reliability and trustworthiness of OpenClaw ecosystems are discussed. 

---
# Personalize-then-Store: Benchmarking and Learning Personalized Memory for Long-horizon Agents 

**Authors**: Yeonjun In, Wonjoong Kim, Sangwu Park, Kanghoon Yoon, Chanyoung Park  

**Link**: [PDF](https://arxiv.org/pdf/2605.25535)  

**Abstract**: Existing large language model (LLM) based memory systems apply universal, static policies that overlook a fundamental reality: the contexts that are worth storing in memory are different across users. This misalignment wastes limited memory budget on transient interactions while failing to preserve critical context for long horizon tasks. To address this gap, we investigate an underexplored question: can LLM based memory systems learn personalized memory policies? We introduce PerMemBench, the first benchmark for evaluating personalized memory systems, featuring multi year, multi domain interaction histories across diverse user personas. We further present the first empirical study of memory personalization, proposing session level storage gating, a lightweight framework that selectively bypasses memory operations for transient sessions. Our study confirms that personalization yields substantial retention gains under perfect gating, yet reveals that accurate gating remains an open and critical challenge. 

---
# What Gets Cited: Competitive GEO in AI Answer Engines 

**Authors**: Rahul Vishwakarma, Shushant Kumar, Ratnesh Jamidar  

**Link**: [PDF](https://arxiv.org/pdf/2605.25517)  

**Abstract**: AI answer engines generate answers from retrieved pages but cite only a few sources. This makes visibility depend not just on ranking, but on being cited. We study competitive Generative Engine Optimization (GEO): when two retrieved candidates compete, what makes one more likely to be cited first? We build a controlled two-document retrieval-augmented generation (RAG) testbed that injects exactly two candidate sources into the model context and measures which source is referenced by the first citation marker in the output. Across six LLMs we execute 252,000 trials, repeated paired comparisons under one factorial program over 18 content factors. In each trial the two sources differ in exactly one factor; we use brand anonymization and counterbalanced source order to separate content effects from position bias. Mixed-effects models show that topical relevance and list position are the biggest drivers of being cited first. Including explicit price information and a recent timestamp also helps consistently. Completeness and trust cues add smaller gains, while formatting-only edits have little impact. We release a reproducible evaluation protocol and a prioritized GEO checklist for practitioners, and we exercised it in an early internal pilot at Sprinklr, where teams reported positive qualitative feedback on workflow usability. 

---
# Whose Alignment? Comparing LLM Process Alignment Across Diverse Organizational Decision Contexts 

**Authors**: Niklas Weller, Emilio Barkett  

**Link**: [PDF](https://arxiv.org/pdf/2605.25256)  

**Abstract**: Aligning AI systems with organizational decision-making is typically framed as a single-target problem: make the model behave like the organization. We argue this framing obscures a deeper pluralistic challenge. We rely on a decision-policy capturing method to measure process alignment: whether an LLM weights information as the organization does, not merely whether it reaches the same conclusions. Applying this method to ECHR Article 6 decisions, process alignment strongly predicts output accuracy (r = 0.85, p < .001) and externalization substantially improves alignment for poorly-aligned models. Applying it to German consumer credit decisions, this relationship collapses (r = 0.15, p = .60): interventions produce inconsistent effects and the benchmark encodes potentially discriminatory historical patterns. This contrast is itself a pluralistic alignment finding: in contested domains, high process alignment is neither achievable via externalization nor unconditionally desirable. Output agreement alone cannot distinguish a model that has internalized an organizational policy from one that merely approximates its outcomes; process-level measurement is a necessary component of any pluralistic alignment evaluation. 

---
# Context-CoT: Enhancing Context Learning via High-Quality Reasoning Synthesis 

**Authors**: Hongbo Jin, Mingnan Zhu, Jingqi Tian, Xu Jiang, Zhongjing Du, Haoran Tang, Siyi Xie, Qiaoman Zhang, Jiayu Ding  

**Link**: [PDF](https://arxiv.org/pdf/2605.25354)  

**Abstract**: While LLMs excel at reasoning over prompts using static pretrained knowledge, they struggle significantly with context learning-the ability to dynamically extract, internalize, and apply new knowledge from complex, task-specific contexts. Recent evaluations on the CL-Bench reveal a critical capability gap: frontier models solve only 17.2% of context-dependent tasks on average. 

---
# FrontierOR: Benchmarking LLMs' Capacity for Efficient Algorithm Design in Large-Scale Optimization 

**Authors**: Minwei Kong, Chonghe Jiang, Ao Qu, Wenbin Ouyang, Zhaoming Zeng, Xiaotong Guo, Zhekai Li, Junyi Li, Yi Fan, Xinshou Zheng, Xi Jing, Yikai Zhang, Zhiwei Liang, Seonghoo Kim, Runqing Yang, Zijian Zhou, Sirui Li, Han Zheng, Wangyang Ying, Ou Zheng, Chonghuan Wang, Jinglong Zhao, Hanzhang Qin, Cathy Wu, Paul Pu Liang, Jinhua Zhao, Hai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.25246)  

**Abstract**: Large language models (LLMs) are increasingly used for optimization modeling and solver-code generation, yet practical operations research and optimization problems often require a harder capability: designing scalable algorithms that exploit problem structure and outperform direct formulation-and-solve baselines. Existing benchmarks are limited to small or simplified examples far below real-world scale and complexity. We introduce FrontierOR, among the first benchmarks to systematically evaluate LLM-based efficient algorithm design for realistic large-scale optimization problems. FrontierOR includes 180 tasks derived from methodologically diverse papers published in top-tier operations research venues, each with standardized instances and a hidden, expert-verified evaluation suite. We evaluate seven LLMs spanning frontier, cost-effective, and open-source models both in one-shot and test-time evolution settings. The results reveal that frontier models still struggle to move from executable formulations to efficient optimization algorithms: the strongest one-shot model outperforms Gurobi in only 31% of cases in both solution quality and computational efficiency, and even strong coding agents with test-time evolution achieve only 50% on selected hard tasks. FrontierOR establishes a practical evaluation platform for LLM-based optimization algorithm design, which enables future LLMs and agents to be systematically tested on whether they can move beyond correct formulation toward a feasible, high-quality, and efficient algorithm. Our FrontierOR Benchmark is available at this https URL. 

---
# Second Guess: Detecting Uncertainty Through Abstention and Answer Stability in Small Language Models 

**Authors**: Ashwath Vaithinathan Aravindan, Mayank Kejriwal  

**Link**: [PDF](https://arxiv.org/pdf/2605.25394)  

**Abstract**: Large language models often generate confident but incorrect answers rather than abstaining when uncertain. This problem is particularly acute for small language models (SLMs), where computational constraints and autonomous operation amplify the need for reliable uncertainty detection. We propose _Second Guess_, a lightweight, parameter-free prompting technique for abstention in multiple-choice question answering (MCQA) that is well-suited for SLMs. Our key empirical insight is that models which truly know an answer will select it consistently, while uncertain models exhibit unstable behavior when an ``I don't know'' option is added. Evaluated on four open models (2B-8B parameters) and four benchmarks, Second Guess achieves the highest composite risk improvement of 10.81\%. Notably, it maintains an 8\% composite risk improvement on fine-tuned models where entropy-based methods degrade, and improves most for lower-performing models. All code and results required to reproduce this work is available in this https URL 

---
# Uncertainty Reasoning with Large Language Models for Explainable Disease Diagnosis 

**Authors**: Xiaoyang Fan, Yufan Cai, Zhe Hou, Jin Song Dong  

**Link**: [PDF](https://arxiv.org/pdf/2605.25566)  

**Abstract**: Clinical decision-making requires reasoning over incomplete, imprecise, and linguistically expressed patient narratives. While large language models (LLMs) excel at extracting latent information from natural language, they lack the verifiability and interpretability essential for trustworthy medical AI. We propose a neuro-symbolic reasoning framework that aligns LLMs with formal logic to enable explainable and formally verifiable medical diagnosis. Patient descriptions and clinical guidelines are embedded into a neural knowledge base, where LLMs extract structured medical entities, temporal relations, and fuzzy symptom patterns, which are decoded into a symbolic knowledge base expressed in fuzzy logic and declarative rules. We perform two-stage reasoning: (1) inductive symbolic generalization to capture diagnostic patterns from encoded narratives, and (2) inference verification via a logic programming engine to derive and validate diagnoses consistent with clinical standards. Each symptom is treated as a fuzzy predicate with probabilistic weights, and inference paths are auditable, adjustable, and compatible with physician feedback. Unlike purely statistical methods, our system supports iterative refinement: misalignment between LLM-generated diagnoses and ground truth can be traced, explained, and corrected through formal rules. By combining logic-based transparency, LLM adaptability, and probabilistic robustness, the framework enables human-aligned healthcare inference with strong generalization and verifiable, step-by-step reasoning chains. We validate our framework on public benchmarks, demonstrating effective reconciliation of symbolic reasoning and LLMs with real-world clinical narratives. Results show performance comparable to state-of-the-art LLMs, while additionally providing interpretable reasoning paths and formally verifiable diagnostic conclusions. 

---
# SimuWoB: Simulating Real-World Mobile Apps for Fast and Faithful GUI Agent Benchmarking 

**Authors**: Guohong Liu, Jialei Ye, Pengzhi Gao, Wei Liu, Jian Luan, Yunxin Liu, Yuanchun Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.25160)  

**Abstract**: Mobile GUI agents powered by large language models have progressed rapidly, creating urgent needs for realistic and comprehensive evaluation. Existing benchmarks prioritize reproducibility but are often limited to open-source apps or file-operation tasks for the difficulty of constructing rewards on real applications, leaving a gap between benchmark settings and real-world usage. Moreover, most benchmarks focus on basic grounding and navigation, with limited coverage of complex, long-horizon interactions. To address these limitations, we introduce SimuWoB, a fully synthetic benchmark for mobile GUI agents with 120 challenging tasks spanning diverse types and difficulty levels. We build a robust virtual environment generation framework that synthesizes high-fidelity tasks and environments, and automatically provides valid rewards for each task. Each environment is deployed as a backend-free webpage accessible via URL, enabling efficient and reproducible evaluation. We conduct comprehensive experiments on several state-of-the-art mobile GUI agents. The average success rate is only 27.92%, dropping to 17.82% on long-horizon tasks, which reveals substantial weaknesses in current agents under complex scenarios. Evaluation result comparison with real-world sample tasks demonstrate that agent assessments based on our synthetic environment generalize well. We further provide diagnostic insights across key capability dimensions and discuss implications for future mobile GUI agent development. 

---
# Representation Without Control: Testing the Realization Effect in Language Models 

**Authors**: Ciarán Walsh, Emilio Barkett  

**Link**: [PDF](https://arxiv.org/pdf/2605.25151)  

**Abstract**: Large language models are increasingly used as behavioral simulators, but it remains unclear when their outputs reflect human-like cognitive mechanisms rather than prompt-sensitive surface patterns. We study this question through the realization effect, a well-characterized finding in behavioral economics in which risk-taking differs systematically after paper versus realized gains and losses. We evaluate LLM behavior at three levels: prompt-only behavioral sensitivity, linear readout of internal representations, and causal control via activation steering. Prompt-only results show systematic condition sensitivity, but the directional pattern does not reproduce human realization-effect predictions. Gemma's residual stream contains a linearly decodable realization-status signal at layer 18 that generalizes to held-out prompts. Steering along this direction does not, however, reliably shift downstream risk choices, a null result that holds across positive scales and in a negative sign-symmetry run. Behavioral sensitivity, latent readout, and causal control are three distinct properties that do not automatically co-occur, and successful latent readout is insufficient evidence that a model behaviorally relies on a representation during downstream decision-making. 

---
# DarkForest: Less Talk, Higher Accuracy for Multi-Agent LLMs 

**Authors**: Yi Li, Songtao Wei, Dongming Jiang, Zhichun Guo, Qiannan Li, Bingzhe Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.25188)  

**Abstract**: Multi-agent LLM systems improve reasoning by combining outputs from multiple agents, but interaction-heavy methods can introduce error propagation and high communication overhead. When agents exchange raw responses or reasoning traces, incorrect intermediate reasoning may be adopted and amplified, leading to confident but wrong consensus; multi-round communication also increases token consumption, latency, and inference cost. In this paper, we propose a controlled-communication coordination framework named DarkForest. DarkForest first keeps agents independent, so each agent produces an answer without seeing the others' outputs. It then parses the raw responses into structured candidate records, groups semantically equivalent candidates into clusters, and estimates a calibrated belief distribution over these clusters using agent reliability, confidence, parse quality, support-pattern reliability, and independence corrections. A coordinator receives only policy-permitted evidence from this belief state with controlled communication. Experiments on six reasoning benchmarks show that DarkForest achieves leading overall quality, improves the strongest baseline by up to 30.7\% on benchmark metrics, and reduces token consumption by up to $6.5\times$ compared with communication-heavy baselines. 

---
# Beyond the Frontier: Stochastic Backtracking for Efficient Test-Time Scaling 

**Authors**: Dao Tran, Duc Anh Le, Ngoc Luu, Quan Pham, Tung Pham, Hung Bui  

**Link**: [PDF](https://arxiv.org/pdf/2605.25143)  

**Abstract**: Test-time scaling improves language model reasoning by spending additional compute to explore multiple solution trajectories. The key challenge is to maximize accuracy while minimizing the total number of generated tokens during reasoning. Recent PRM-guided methods score intermediate prefixes to steer this search, but most are frontier-only: they keep only the current active prefixes and irreversibly prune or resample away the rest using noisy PRM scores. This can cause premature commitment, diversity collapse, and the loss of prefixes that still admit correct continuations. We introduce stochastic backtracking over a persistent pool of historical prefixes, allowing test-time compute to revisit previously generated states instead of only expanding the current frontier. To make this efficient, we propose two complementary mechanisms. Subpool Selection strengthens greedy PRM-guided search by applying Top-N selection within random subpools, giving historical prefixes a chance to bypass over-scored frontier candidates. Power Backtrack Sequential Monte Carlo extends SMC-style resampling to the persistent pool using powered PRM scores and mixture-corrected weights. Across mathematical reasoning benchmarks and model scales, our methods consistently achieve higher accuracy per token count, and the same level of accuracy using only a fraction of the token count in comparison to strong PRM-guided baselines, demonstrating that persistent-pool stochastic backtracking provides a simple and effective way to improve the accuracy-token trade-off in test-time scaling. 

---
# SpecAlign: A Semantic Alignment Framework for SystemVerilog Assertion Generation 

**Authors**: Jaime Rafael Imperial, Hao Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2605.25181)  

**Abstract**: Existing Large Language Model (LLM) approaches to SystemVerilog Assertion (SVA) generation primarily focus on syntactic validity and formal verification outcomes, while semantic alignment between generated assertions and natural language specifications remains difficult to quantify. As a result, hallucinated or misaligned SVAs can reduce confidence and increase debugging efforts in the absence of golden RTL. This paper presents SpecAlign, a framework for semantic evaluation and refinement of LLM-generated SVAs. SpecAlign introduces two iterative alignment loops that assess both natural language properties and SVAs against the design specification using entailment-based classification. We improve alignment decisions by generating multiple reasoning paths using chain-of-thought prompting and aggregating them via a self-consistency voting mechanism. Misaligned assertions are analyzed to generate actionable feedback for refinement. We further define a quantitative alignment score to measure semantic consistency across iterations. Experimental results demonstrate that SpecAlign effectively detects semantic inconsistencies and improves assertion alignment without relying on golden RTL, providing a scalable complement to traditional formal verification evaluation metrics. 

---
# Trust but Verify: Prover-Verifier Deliberation for Selective LLM Prediction 

**Authors**: João Sedoc, Baotong Zhang, Dean Foster  

**Link**: [PDF](https://arxiv.org/pdf/2605.25133)  

**Abstract**: Reliably knowing when a language model is correct is almost as important as being correct. We introduce prover-verifier deliberation (PVD), an inference-time protocol grounded in interactive proof theory, as a mechanism for selective prediction: the protocol produces both an answer and a structured confidence verdict, allowing a system to report high-confidence answers while abstaining on uncertain cases. In each dialogue, a prover defends a candidate answer through checkable sub-claims while a verifier issues targeted challenges and returns \textsc{Accept}, \textsc{Challenge}, or \textsc{Reject}. Because frozen language models are imperfect provers and verifiers operating over a noisy channel, formal soundness and completeness guarantees do not transfer; instead, we characterize the protocol empirically through its coverage-precision behavior. Our main experiment uses Claude Sonnet 4.6 as prover and Claude Haiku 4.5 as verifier on GPQA Diamond. Questions accepted with no answer revision, which we call Accept + No Change (ANC), are reported as the high-confidence subset; we evaluate this subset by its precision and coverage. ANC separates reliable from unreliable answers, yielding a $\sim$30pp HC-Prec gap over the non-ANC complement. Robustness experiments with GPT and Gemini pairings show that high HC-Prec can transfer across model families, while verifier strictness and domain competence largely determine the size of the selection gap. On Humanity's Last Exam, weaker prover-verifier pairings can collapse or invert the ANC signal, illustrating a practical failure mode when the verifier operates outside its effective region. Comparisons with self-consistency, universal self-consistency, multi-agent debate, and Reflexion suggest that prover-verifier deliberation supplies a distinct argument-defensibility signal for selective prediction. 

---
# AION: Next-Generation Tasks and Practical Harness for Time Series 

**Authors**: Tianxiang Zhan, Xiaobao Song, Tong Guan, Shirui Pan, Ming Jin  

**Link**: [PDF](https://arxiv.org/pdf/2605.25045)  

**Abstract**: Time series research is moving beyond fixed forecasting benchmarks toward realistic tasks that combine prediction, contextual reasoning, tool use, and structured decision support. Most benchmarks are built around clean data and short evaluation loops; agents alone may miss temporal constraints, evidence checks, or review before finalizing outputs. We first formalize next-generation time series tasks as three-component tuples consisting of a task file, a workspace, and a validation interface. We then present AION, a time series harness built from six component groups: agents, skills, rules, memory, evaluation, and protocols. In this harness, we use three design principles: temporal grounding, temporal knowledge-grounded reasoning, and reliability mechanisms such as post-experiment analysis and layered review. A Kaggle Store Sales case study shows that the harness produces more detailed process traces, more artifacts, and more review steps than the same base agent operating in OpenCode direct build mode. Taken together, these results argue for a paradigm shift from fixed tasks to realistic ones under real-world constraints. 

---
# AI Cartography: Mapping the Latent Landscape of AI Benchmark Ecosystems 

**Authors**: Michael Hardy, Anka Reuel, Lijin Zhang, Jodi M. Casabianca, Sang Truong, Yash Dave, Hansol Lee, Benjamin Domingue, Sanmi Koyejo  

**Link**: [PDF](https://arxiv.org/pdf/2605.25272)  

**Abstract**: While aggregate leaderboard scores drive AI development, they contain substantial measurement noise whose sources and magnitudes remain unquantified, making it unclear when rankings reflect genuine capability differences versus evaluation artifacts. We introduce a framework for measuring the latent landscape in AI benchmark ecosystems. Applying Confirmatory Factor Analysis (CFA) and Generalizability Theory to 4,000+ models from the Open LLM Leaderboard, we decompose sources of ranking variance and establish: (1) structures assumed in current reporting practice underestimate the strength of relationships between benchmarks; (2) evidence of local dependence among leaderboard items, undermining uses of benchmarks as measurement instruments under current scoring systems; (3) contributor metadata explains more rank-relevant variance ($\approx9\%$) than architecture or deployment categories in this context; (4) a manifest-score "scaling law" slope has low reliability ($R_{\beta}=0.53$); by contrast, the latent general-factor size slope is highly stable across ecosystem controls ($R_g=0.97$). We are able to provide unique insights into benchmark dynamics, such as which benchmarks are a function of LLM size and which can be oppositely impacted by post-training practices. We provide actionable diagnostics to determine how benchmark rankings can be trusted and how benchmark design can be improved. 

---
# LipoAgent: Coordinating Fine-Tuned LLM Agents for Safer Lipid Design 

**Authors**: Leshu Li, An Lu, Haiyu Wang, Zhibin Feng, Conghui Duan, Qing Bao, Zongmin Zhao, Sai Qian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.25250)  

**Abstract**: Lipid nanoparticles (LNPs) are among the most clinically mature platforms for nucleic acid delivery, yet designing lipids that are both effective and biologically safe remains a major bottleneck. In practical screening, toxicity is a decision-level constraint: if a lipid is toxic, its efficiency prediction is clinically irrelevant. We propose LipoAgent, a safety-aware multi-agent LLM framework for lipid discovery. LipoAgent combines domain-specific finetuning with a conditional prediction objective that enforces toxicity as a prerequisite for efficiency prediction, and further improves reliability via multi-agent verification with lightweight human oversight when disagreement persists. Across multiple foundation models, LipoAgent achieves an average 32% relative improvement in mRNA transfection efficiency prediction compared with other reported models for lipid design. Wet-lab validation confirms that virtual screening rankings reliably translate to biological transfection outcomes. The code is publicly available at this https URL. 

---
# Meta-Agent: From Task Descriptions to Verified Multi-Agent Systems 

**Authors**: Andy Xu, Yu-Wing Tai  

**Link**: [PDF](https://arxiv.org/pdf/2605.25233)  

**Abstract**: AI agents are increasingly used to solve complex, multi-step tasks, but existing multi-agent frameworks remain brittle as workflows grow in scale and depth. Small errors at intermediate stages can propagate through agent interactions, while insufficient grounding and weak verification mechanisms further limit reliability. We present Meta-Agent, a two-phase framework that automatically constructs and executes specialized multi-agent systems from natural-language task descriptions. In the construction phase, a task planner decomposes a problem into a directed acyclic graph of agent specifications with explicit input/output contracts and verification criteria. A web search module grounds each specification with external evidence, and a code generation module produces system prompts and tool configurations. A construction-time verification stage then validates generated artifacts and triggers targeted regeneration when failures are detected. In the execution phase, a coordinator dispatches subtasks across the agent graph while execution-time verification gates intermediate outputs. We further introduce a three-level error attribution mechanism that distinguishes local, upstream, and structural failures, enabling targeted recovery strategies ranging from localized retries to partial re-execution and re-decomposition. We evaluate Meta-Agent across coding, contextual learning, and open-ended reasoning tasks. Experiments against strong multi-agent baselines and ablation studies demonstrate consistent improvements in task success rate, error recovery, and workflow stability. The results highlight the importance of tightly integrating planning, grounding, and verification for building reliable multi-agent systems. 

---
# Geo-Expert: Towards Expert-Level Geological Reasoning via Parameter-Efficient Fine-Tuning 

**Authors**: Chenyou Guo, Zongqi Liu, Yizhou Zhang, Zhaorui Jiang, Ze Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.24844)  

**Abstract**: While general-purpose Large Language Models (LLMs) applied to Geology often hallucinate when reasoning about subsurface structures and deep-time evolution, current AI in Earth sciences predominantly targets surface remote sensing and GIS. To bridge this gap, we introduce Geo-Expert, a family of parameter-efficient geological LLMs fine-tuned on a custom-curated, high-quality instruction dataset processed using our custom instruction synthesis pipeline. We investigate the impact of model scaling and architecture by fine-tuning three base models: Qwen3-8B, Qwen3-32B, and Gemma-3-27B, with Low-Rank Adaptation (LoRA) method. Our extensive evaluation on a novel domain-specific benchmark, Geo-Eval, reveals that a domain-aligned 8B model can outperform open-weight 70B generalists and proprietary GPT-4o on specialized geological reasoning, while a 32B variant approaches frontier reasoning models. The optimized 8B model further offers a competitive cost-performance ratio for deployment. This work provides a reproducible recipe for democratizing scientific LLMs and establishes a baseline for geological artificial intelligence. 

---
# Privacy-Preserving Local Language Models for Longitudinal Data Retrieval in Chronic Dermatologic Disease: Implementation in Pemphigus Patients 

**Authors**: Abdurrahim Yilmaz, Ayşe Esra Koku Aksu, Duygu Yamen, Vefa Asli Erdemir, Mehmet Salih Gurel, Gulsum Gencoglan, Joram M. Posma, Burak Temelkuran  

**Link**: [PDF](https://arxiv.org/pdf/2605.25020)  

**Abstract**: Chronic dermatologic diseases such as pemphigus require long-term follow-up, generating extensive longitudinal clinical documentation that is difficult to review comprehensively during routine visits and increasing clinician workload as well as the risk of missing critical historical information. We evaluated whether a locally deployed, privacy-preserving small language model (SLM) could retrieve structured clinical features and generate longitudinal summaries from long-term dermatology follow-up records. In this retrospective case series, thirty pemphigus patients contributed 541 visit notes that were aggregated into full longitudinal records (89,336 words); 56 clinically relevant features were annotated by two expert dermatologists. The locally deployed SLM (Qwen3 4B Thinking 2507) was queried with each complete record to retrieve 56 features and generate one final report summaries. Across 1,680 feature retrieval tasks, mean accuracy was 82.25%. Dermatologists' ratings of AI-generated summaries were high for overall quality (8.23-8.47), clinical accuracy (7.93-8.20), and usefulness (8.47-8.50), with no significant inter-evaluator differences and an overall preference for AI summaries in 53.3% of evaluations. These findings suggest that privacy-preserving, locally deployed SLMs can outperform medical experts and reliably generate clinically meaningful longitudinal summaries. SLMs may support clinical decision-making when integrated with appropriate oversight. 

---
# CoRe-Code: Collaborative Reinforcement Learning for Code Generation 

**Authors**: Zhihao Dou, Qinjian Zhao, Zhongwei Wan, Xiaoyu Xia, Sumon Biswas  

**Link**: [PDF](https://arxiv.org/pdf/2605.24812)  

**Abstract**: Large language models (LLMs) have achieved strong performance in code generation, but most methods rely on autoregressive decoding without global planning, often leading to locally coherent yet globally suboptimal solutions (e.g., failing test cases or inefficient complexity). While recent approaches such as Chain-of-Thought (CoT) and multi-agent systems (MAS) introduce planning, their limited role specialization and coordination hinder performance on complex tasks. To address the challenges of coordination and specialization in multi-agent code generation, we propose Collaborative Reinforcement Code (CoRe-Code), a framework for role specialized LLM agents that enhances inter-agent coordination to generate more accurate and efficient code. CoRe-Code adopts a simple Planner-Coder paradigm, where the Planner produces high-level plans and the Coder executes them to generate code. We further introduce a collaboration-aware reinforcement learning stage based on Group Relative Policy Optimization (GRPO) to enhance role specialization and alignment. Experiments show that CoRe-Code outperforms a wide range of existing RL-based and multi-agent methods. In addition, we demonstrate that CoRe-Code can generalize to other multi-agent frameworks (e.g., Retrieval and Debugging agents), highlighting its flexibility and scalability. We evaluate CoRe-Code on multiple benchmarks of varying difficulty using three base models. Compared to existing baselines, the results show consistent improvements in accuracy, while also achieving higher efficiency in terms of execution time and memory usage, demonstrating the effectiveness and practicality of CoRe-Code. 

---
# Towards Multi-Turn Dialog Systems for Industrial Asset Operations and Maintenance 

**Authors**: Chengrui Li, Rujing Li, Yitong Bai, Rui Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.24953)  

**Abstract**: Industrial asset operations and maintenance question answering is inherently multi-turn, iterative, and highly dependent on external tool invocation. However, the conventional plan-execute single-agent architecture exhibits clear limitations in maintaining cross-turn context, and reusing intermediate results. In this paper, we present a multi-turn dialog system designed for industrial scenarios based on a supervisor-specialist multi-agent architecture. To alleviate tool invocation bottlenecks, the system incorporates structured artifact reuse, dynamic replanning, and parallel tool execution. Evaluation results show that our system achieves better response quality compared with the baseline, with planning effectiveness increasing by 54.5% and task completion improving by 37.8%. System profiling further shows that cross-turn artifact reuse effectively reduces redundant tool invocation, decreasing the tool-time share from 47.3% to 26.3% and making turns 2-5 approximately 4.2x faster than the first turn. 

---
# ProActor: Timing-Aware Reinforcement Learning for Proactive Task Scheduling Agents 

**Authors**: Lei Ding, Bin He, Chenguang Wang, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.24900)  

**Abstract**: Proactive task-oriented agents must autonomously anticipate user needs, identify actionable opportunities, and trigger software actions at appropriate moments - fundamentally shifting from reactive systems that await explicit instructions. However, existing approaches lack generalizable end-to-end solutions for measuring and optimizing such anticipatory behaviors.
This paper introduces ProActor, a unified framework for conversational task scheduling that integrates: (1) a domain-agnostic automated annotation methodology that enables scalable proactiveness reinforcement learning (RL) by generating full opportunity time windows instead of rigid point labels, (2) systematic proactiveness metrics capturing both timing quality and reference action alignment, and (3) RL optimization using GRPO with various reward designs. Our insight is that RULER-based rewards with proactiveness rubrics are crucial for improving timing quality, and that proactiveness optimization enabled by stage-aware composite rewards is key to balancing timing quality and reference action alignment.
Timing-aware RL requires extensive exploration, demanding efficient infrastructure. We develop ART-F, an adaptive framework combining request-adaptive inference clusters with DDP-based training on single-node multi-GPU systems, enabling LoRA training of 4-bit Qwen2.5-14B-ProActor-Q4 with 4-8x speedups. Experiments on two newly auto-annotated datasets demonstrate significant improvements in proactive timing while maintaining action consistency comparable to state-of-the-art (SOTA) baselines. Ablations validate the effectiveness of distinct composite reward variations. 

---
# GRAIL: AI translation for scientists application workflow on satellite data 

**Authors**: Zhuocheng Shang, Ahmed Eldawy  

**Link**: [PDF](https://arxiv.org/pdf/2605.24784)  

**Abstract**: Domain scientists increasingly develop Python scripts to analyze satellite imagery but they lack scalability to large-scale data. This paper demonstrates GRAIL, an agentic translation system that converts Python geospatial workflows into executable Spark-based programs without requiring scientists to learn a new framework. Rather than fine-tuning a specialized LLM model, GRAIL adapts RDPro, a Scala library for satellite data analysis, to make it LLM-ready using structured documentation, API alias functions, and repair-oriented error logs. Translation is structured as a LangGraph pipeline that decomposes code generation into explicit sections with guided inputs and outputs, enabling targeted repair without regenerating the full program. We demonstrate GRAIL on real-world geospatial workflows and showcase the correctness and scalability of the translated code. 

---
# PANDO: Efficient Multimodal AI Agents via Online Skill Distillation 

**Authors**: Yubo Li, Yidi Miao, Haotian Shen, Yuxin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.24785)  

**Abstract**: Recent advances in multimodal web agents often rely on increased inference-time computation, including rollout search, verifier passes, offline skill discovery, and specialist model stacks. This raises a central question: can a web agent become more efficient as it accumulates experience, rather than more expensive? We first analyze trajectories from VisualWebArena and identify three recurring sources of inefficiency: repeat-action loops, hidden discovery costs, and low prompt-cache reuse. We then introduce PANDO, a single-rollout online skill-distillation framework that maintains a structured Skill Library and combines progress reflection, confidence-based skill demotion, hierarchical routing, visual compression, and cache-aware prompting. On the full set of 910 VisualWebArena tasks, PANDO achieves a 58.3% success rate, outperforming SGV (54.0%) and our WALT reproduction (45.2%), while using 58% fewer tokens than SGV and 61% fewer tokens than WALT, without any pre-evaluation discovery budget. A 300-task ablation further shows that rules and routines provide most of the success gains, while routing, compression, and cache-aware prompting convert the larger skill library into lower marginal token cost. Finally, we introduce three trajectory-level efficiency metrics -- Action Repetition Rate, Step Overhead Ratio, and Prompt Cache Utilization -- to make efficiency visible beyond terminal success. 

---
# Test-Time Deep Thinking to Explore Implicit Rules 

**Authors**: Wentong Chen, Xin Cong, Zhong Zhang, Yaxi Lu, Siyuan Zhao, Yesai Wu, Qinyu Luo, Haotian Chen, Yankai Lin, Zhiyuan Liu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2605.24828)  

**Abstract**: With the continuous advancement of Large Language Models (LLMs), intelligent agents are becoming increasingly vital. However, these agents often fail in environments governed by implicit rules--hidden constraints that cannot be observed directly and must be inferred through interaction. This causes agents to fall into repetitive trial-and-error loops, ultimately leading to task failure. To address this challenge, we propose Test-Time Exploration (TTExplore), a framework where a thinker component analyzes interaction history to infer these implicit rules and guide an actor. Effective exploration in this setting critically depends on the reasoning ability of the thinker. However, evaluating deep reasoning trajectories is inherently unstable and difficult, which poses a major obstacle to effective training. To overcome this issue, we introduce a novel and stable reinforcement learning pipeline. The core idea is to use accurate task-level scores as indirect rewards to bypass the difficulty of evaluating intermediate reasoning, and to retain only a single thinking node per trajectory to alleviate reward sparsity. Using this pipeline, we train a specialized 7B model, Exp-Thinker. Experiments on five text-based embodied tasks show that TTExplore equipped with Exp-Thinker improves baseline agent performance by an average of $14$-$19$ points, demonstrating the effectiveness of explicitly reasoning about implicit rules. 

---
# PRIMA: Operational Patterns for Resilient Multi-Agent Research with Verifiable Identity and Convergent Feedback 

**Authors**: Sasank Annapureddy  

**Link**: [PDF](https://arxiv.org/pdf/2605.24775)  

**Abstract**: Operating LLMs as coordinated multi-agent research systems over multi-hour runs surfaces failure modes that single-shot evaluation cannot: upstream providers throttle without warning, sub-agents drift the task to fit accessible tools, narrate machinery instead of using it, open revision iterations with self-apology, or treat upstream context as executable directives. We present PRIMA, whose primary contributions are three operational patterns for surviving these failure modes: (1) a resilience-and-recovery layer that detects upstream rate-limit signals, persists a typed pause record to disk, and resumes long-running runs without re-executing converged work even across process restarts; (2) a sub-agent operating discipline encoding task-fidelity, tool-use, revision, and inter-step context-boundary norms as a structural prompt layer; (3) a multi-phase application pattern for structured engineering deliverables pairing orthogonal draft steps with an explicit cross-document harmonization pass before final synthesis. These sit atop a foundational protocol: a research-program specification language with explicit convergence criteria, a dual-metric scoring engine (LLM-judged rubric plus sandboxed code), an outer meta-optimization loop, event-driven persistence, hook-based middleware, context compaction, and a multi-provider LLM abstraction. Agent identities derive from prime powers, giving collision-free identifiers and trivially-verifiable cluster membership without a central registry. Theoretical guarantees include $O(k)$ verification, $O(V+E)$ DAG validation, and identity collision freedom by the Fundamental Theorem of Arithmetic. A Graph Isomorphism case study grounds the architectural claims in a generated artifact: a six-step protocol that produced a research paper proposing a new canonical-form algorithm with three theorems and five conjectures. 

---
# Proper Scoring Rules for Agentic Uncertainty Quantification 

**Authors**: Suresh Raghu, Satwik Pandey, Shashwat Pandey  

**Link**: [PDF](https://arxiv.org/pdf/2605.24756)  

**Abstract**: Language-model agents increasingly emit uncertainty signals throughout a trajectory, but existing agentic UQ evaluations often conflate ranking usefulness with probabilistic truthfulness. AUROC, AUPRC, risk-coverage, Trajectory ECE, and scalarized trajectory scores evaluate discrimination, binwise calibration, or collapsed summaries, but do not strictly elicit the full prefix-conditioned success-probability trace $q_t = P^{\pi}(Y=1 | H_t)$. Building on prequential proper scoring, we introduce the Trajectory Proper Score (TPS), a predictor-agnostic family of strictly proper trajectory-level scoring rules for any per-step uncertainty signal calibrated into a probability of eventual success. We prove that TPS strictly elicits the success-probability process under complete observation, within the chosen score family and weight schedule. We extend the construction to administratively censored trajectories by projecting the complete-data score onto the observable stopped prefix, yielding an exact $q_Z$-weighted reduced score and a tractable approximation when $q_Z$ is unestimated. We further show that common trajectory evaluators target weaker objects than the full prefix-conditioned probability process: Trajectory ECE is resolution-blind, while scalarized Trajectory Brier elicits only the collapsed scalar, not the full trace. Experiments on StrategyQA, Tau2-Bench, HotpotQA, and WebShop show that these theoretical distinctions are operationally visible: probability recalibration can substantially change TPS while leaving rank metrics nearly unchanged, and the tractable censored approximation can change the verdict relative to complete-only evaluation. 

---
# Inverting the Shield: Systematically Generating Safety Tests from Policy Specifications 

**Authors**: Xiaoyue Lu, Xianglin Yang, Haijun Liu, Jiahao Liu, Kuntai Cai, Yan Xiao, Jin Song Dong  

**Link**: [PDF](https://arxiv.org/pdf/2605.24883)  

**Abstract**: The widespread integration of Large Language Models (LLMs) necessitates rigorous and systematic safety evaluation. Existing paradigms either rely on constructed benchmarks to assess safety from predefined perspectives, or employ dynamic red-teaming to probe potential vulnerabilities. While effective, these approaches face challenges, as they depend heavily on expert domain knowledge, offer limited systematic guarantees, and are vulnerable to rapid obsolescence. To address these limitations, we introduce a novel framework POLARIS that brings the rigor of specification-based software testing to AI safety. POLARIS first compiles unstructured natural-language policies into First-Order Logic (FOL) representations, establishing a traceable link between high-level rules and concrete test cases. This formalization enables the construction of a Semantic Policy Graph, where complex policy violation scenarios are encoded as traversable paths. By systematically exploring this graph, POLARIS uncovers compositional violation patterns, which are then instantiated into executable natural-language test queries, enabling coverage-driven and reproducible safety testing. Experiments demonstrate that POLARIS achieves higher policy coverage and attack success counts compared to established baselines. Crucially, by bridging formal methods and AI safety, POLARIS provides a principled, automated approach to ensuring LLMs adhere to safety-critical policies with verifiable traceability. We release our code at this https URL. 

---
# Agent Manufacturing: Foundation-Model Agents as First-Class Industrial Entities 

**Authors**: Yilei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.24823)  

**Abstract**: Manufacturing has passed through four widely recognized paradigms - mechanization, electrification, programmable automation, and Smart Manufacturing - each defined by the kind of work it shifted from humans to machines. In every case, one layer of industrial work remained fundamentally human: the coordinative cognition of production, comprising the interpretive, allocative, diagnostic, negotiative, and governance work exercised by engineers, planners, and operational managers. We argue that a fifth transition is now underway in which this layer, rather than the physical or routine-cognitive layers below it, is what foundation-model-based autonomous agents primarily redistribute. We name this paradigm Agent Manufacturing and define it operationally: a manufacturing system is an instance of Agent Manufacturing when its principal coordination mechanism is reasoning performed by foundation-model agents that can interpret open-ended goals, plan over long horizons, invoke tools and machines, and negotiate with other agents and humans. This is a narrower and more falsifiable definition than the existing literature on cognitive manufacturing or Industry 5.0 provides, and it distinguishes the paradigm sharply from classical multi-agent manufacturing systems, which were autonomous only within closed protocol spaces. 

---
# Clustering as Reasoning: A $k$-Means Interpretation of Chain-of-Thought Graph Learning 

**Authors**: Xuanting Xie, Zhaochen Guo, Bingheng Li, Xingtong Yu, Zhifei Liao, Zhao Kang, Yuan Fang  

**Link**: [PDF](https://arxiv.org/pdf/2605.24867)  

**Abstract**: Chain-of-Thought (CoT) prompting has shown promise in enhancing the reasoning capabilities of large language models (LLMs) on text-attributed graphs (TAGs). This work reframes CoT-based graph learning through the principle of clustering as reasoning, offering a $k$-means interpretation of how iterative reasoning operates over graph-structured data. We observe that existing graph CoT methods rely on disjoint architectures and fixed graph representations, limiting step-by-step semantic-topological interaction and interpretability. To overcome this limitation, we propose a unified framework named KCoT that integrates CoT reasoning with graph representation learning. Our key theoretical result reveals a formal mathematical correspondence between a Transformer block and the $k$-means algorithm, allowing reasoning to be interpreted as iterative assignment and update steps. Based on this insight, we introduce a Semantic Discriminating Prompt that explicitly formulates these steps as structured CoT reasoning, together with a structure-grounded alignment strategy to fuse topological priors with evolving thought-conditioned representations. Experiments on standard benchmarks demonstrate consistent improvements over state-of-the-art methods, validating clustering as a principled mechanism for CoT-based graph learning. 

---
# Automated Detection and Classification of Delusion-related Content in Naturalistic Audio Diaries Using Multi-Agent Language Models 

**Authors**: Feng Chen, Justin Tauscher, Changye Li, Meliha Yetisgen, Alex Cohen, Adam Kuczynski, Angelina Pei-Tzu Tsai, Benjamin Buck, Dror Ben-Zeev, Trevor Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2605.24755)  

**Abstract**: Speech monologues recorded in naturalistic settings provide opportunities to characterize mental illness phenomenology and detect symptom exacerbation. Large language models (LLMs) offer new possibilities for automating this process, as they require annotated data primarily for evaluation rather than training. In this paper, we present a novel automated, multi-agent LLM pipeline for the fine-grained, multi-label extraction of language suggestive of delusional beliefs, associated affective responses, and behavioral responses from transcripts of naturalistic audio diaries collected from people with moderate persecutory ideation. Evaluating an ensemble of three foundation models, we demonstrate that detailed diagnostic prompt instructions successfully reduce false positives for delusional theme classification, but also constrain the interpretation of affective or behavioral responses. Furthermore, comparing multi-agent adjudication frameworks shows that complex conversational debate between agents diminishes accuracy on clinically ambiguous text by inducing premature consensus. Instead, majority voting establishes robust performance (Micro F1 of 0.872 and 0.779 for delusion detection and classification respectively). This work provides a validated and scalable pipeline for the automated detection and characterization of content suggesting delusional beliefs in naturalistic speech. 

---
# Fundamental Limitation in Explaining AI 

**Authors**: Atsushi Suzuki, Jing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.24727)  

**Abstract**: While large-scale models such as LLMs and diffusion models have achieved practical success, public institutions have emphasized the importance of explainability in AI. Existing methods for explaining AI, however, are not designed to provide completely faithful explanations of the behavior of large-scale AI systems. Although a completely faithful and interpretable explanation of the behavior of an AI system might be useful for AI governance, it has not been known whether providing such an explanation is theoretically possible. In this paper, we mathematically prove a fundamental quadrilemma in explaining AI, stating that AI and its explanation cannot satisfy the following four conditions simultaneously: 1) the complexity of the operation environment, 2) the goodness of the AI's performance, 3) the interpretability of the AI's explanation, and 4) the complete faithfulness of the AI's explanation. This quadrilemma suggests that, in most applications where we cannot change the environment or sacrifice good AI performance and an interpretable explanation, we should give up complete faithfulness of explanations and should instead aim to explain only the parts that are important for applications. As a consequence, the quadrilemma implies that AI governance should be designed on the premise that the faithfulness of AI explanations is always incomplete. 

---
# MDIA: A Multi-Agent Diagnostic Intelligence Pipeline on HealthBench Professional 

**Authors**: Roberto Cruz, David Rey-Blanco  

**Link**: [PDF](https://arxiv.org/pdf/2605.24699)  

**Abstract**: Most reported gains on agentic-LLM clinical benchmarks are often attributed to prompt engineering, yet our results suggest that larger improvements can come from architectural and engine-level design. We present MDIA, a Multi-agent Diagnostic Intelligence Agent implemented as a 7-node specialty-routed clinical reasoning graph, on the full HealthBench Professional benchmark (n = 525), on a non-fine-tuned LLM. MDIA achieves 0.6272 under OpenAI's GPT-5.4-2026-03-05, which is +3.72 pp above the performance of OpenAI's ChatGPT for Clinicians. The experimental work shows that performance lift is attributable to system architecture: specialty routing, multi-turn context preservation, drug-state safety gating, site-filtered search, length-aware synthesis, and engine-level reliability. These findings support the view that agentic clinical benchmark performance is shaped both by the underlying foundation model and the orchestration architecture. Nevertheless, we also noticed notable differences when using other models as a grader; in particular, when using Gemini 2.5 Pro, MDIA scored 0.6585, which suggests that the choice of grader is a source of variability. Robust evaluation of LLMs would therefore require assessment across several independent grader models. 

---
# Emotional intelligence in large language models is fragmented across perception, cognition, and interaction 

**Authors**: Minghao Lv, Lu Chen, Enchang Zhang, Anji Zhou, Xiaoran Xue, Hanyi Zhang, Fenghua Tang, Zhuo Rachel Han, Mengyue Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.24686)  

**Abstract**: As large language models (LLMs) are increasingly integrated into emotionally sensitive domains, the structural integrity of their emotional intelligence (EI) becomes a critical frontier for safety and alignment. Current benchmarks often conflate superficial politeness with deep affective reasoning, failing to distinguish between perceptual accuracy and interactive efficacy. Here, we introduce FACET (Functional Affective Competence and Empathy Test), a psychometrically grounded framework comprising 480 expert-crafted items. Unlike previous metrics, FACET is theoretically anchored in the Mayer-Salovey-Caruso four-branch ability model, operationalizing EI through perception, facilitation, understanding, and management of emotions. Through an evaluation of nine frontier models (including GPT-5, Claude-Sonnet-4), we demonstrate that emotional intelligence is not a monolithic capability but is fragmented across cognitive and interactive dimensions. While frontier models demonstrate robust proficiency in objective emotion recognition and social reasoning, this does not consistently translate to interactive success. We categorize these discrepancies into three distinct performance profiles: cognitive-dominant, interactive-dominant, and context-dependent. These typologies indicate that emotional skills do not scale uniformly with general intelligence or model size; rather, they are shaped by specific alignment paradigms. Notably, we identify hidden emotion recognition as a universal performance bottleneck across all architectures. Our results suggest that current RLHF processes may optimize for "stochastic empathy", a statistical mimicry of emotional syntax, at the expense of integrated affective reasoning. These findings challenge the assumption of linear emotional scaling and provide a rigorous roadmap for developing socially aware agents capable of genuine clinical resonance. 

---
# Beyond Inference-Only Deployment: Comparing Weight-Based Consolidation Against Cascading Compaction 

**Authors**: Simon Dennis, Kevin Shabahang, Hao Guo, Rivaan Patil  

**Link**: [PDF](https://arxiv.org/pdf/2605.24657)  

**Abstract**: Major LLM platforms deploy models in an inference-only configuration: the model serves requests but never updates per-user weights. Users must repeatedly re-teach preferences, corrections, and project context, and context-based workarounds consume context-window space and degrade under cascading compaction. We evaluate an alternative: nightly consolidation of interaction knowledge into model weights via reflection, synthesis, and Low-Rank Adaptation (LoRA) fine-tuning on a single consumer GPU. Across ten realistic software development conversations (n = 10, 1,146 test questions across three memory types), three cycles of cascading compaction retain 36.8 +/- 3.0% of knowledge (between an 11.8% no-context floor and a 90.1% full-context ceiling), while consolidation retains 80.4 +/- 1.3% -- a 43.6 pp gain (paired t(9) = 14.8, p < 0.001) that more than doubles what compaction preserves, with the largest gains on procedural corrections (36.3% -> 74.6%) and episodic project facts (31.5% -> 78.2%). As a methodological aside, mean per-token validation cross-entropy is negatively correlated with LLM-judged accuracy (r = -0.51) while median per-token validation cross-entropy tracks accuracy almost exactly (r = +0.99): under evaluators that tolerate surface-form variation, the mean is misleading and a heavy-tail-robust statistic is the faithful signal. Persistent personalization requires moving beyond inference-only deployment toward architectures that consolidate knowledge into weights. 

---
# When Mean CE Fails: Median CE Can Better Track Language Model Quality 

**Authors**: Hao Guo, Simon Dennis, Rivaan Patil, Kevin Shabahang  

**Link**: [PDF](https://arxiv.org/pdf/2605.24667)  

**Abstract**: Mean cross-entropy is the standard validation metric for language models, but it can fail to track model quality during training. We examine this in two common scenarios. First, in Qwen2.5-1.5B SFT on synthetic fact-learning, we find that mean CE rises substantially after the initial learning phase while held-out fact-recall accuracy remains near its peak. Second, we find that in top-K distillation on TinyStories, decreasing K improves median CE while worsening mean CE; the Top-5 student attains the highest LLM-judge score and crosses below its teacher on median CE, despite having the worst mean CE. In both cases, median CE correlates much more closely with task performance than does mean CE.
Analyzing how bulk and tail percentile CE move during training reveals that training reshapes the empirical per-token CE distribution. In top-K distillation, smaller K yields a distribution with more mass at both extremes, decreasing the median and increasing the mean. In Qwen SFT, the bulk saturates quickly while the tail extends in the latter half of training. In both, the task-evaluation metric appears more sensitive to the bulk than to the tail.
Practically, we recommend reporting a small set of percentile CE summaries alongside the mean, and using concordance among them as a tool to keep track of distribution reshaping, as well as a low-cost diagnostic for when mean and median CE disagree on model selection. 

---
# GlobalDentBench: A Multinational Benchmark for Evaluating LLM Clinical Reasoning in Dentistry with Expert Calibration 

**Authors**: Junjie Zhao, Jingyi Liang, Zhenyang Cai, Jiaming Zhang, Zhenwei Wen, Shuzhi Deng, Wenjing Yi, Chunfeng Luo, Hexian Zhang, Junying Chen, Tianrui Liu, Zhuhui Bai, Zixu Zhang, Pradeep Singh, Xiang Liu, Jianquan Li, Nhan L Tran, Falk Schwendicke, Zuolin Jin, Lijian Jin, Liangyi Chen, Wei-fa Yang, Benyou Wang, Junwen Wang, Shan Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2605.24636)  

**Abstract**: While large language models (LLMs) hold transformative potential for medicine, their reasoning robustness and safety in real-world clinical scenarios remain critically underexplored, particularly in dentistry. Here we introduce GlobalDentBench, the first multinational dental benchmark, featuring a taxonomy that encompasses 14 dental specialties across 88 countries and regions spanning six continents. The benchmark comprises 8,978 expert-validated questions across three formats (multiple-choice, short-answer, and case-based questions) and assesses three progressive reasoning levels: knowledge recall (L1), routine reasoning (L2), and individualized reasoning (L3). To ensure data quality, the automated construction framework was calibrated by six senior dentists, achieving expert agreement rates of 99.98% for multiple-choice and short-answer questions and 96.78% for the more complex case-based questions. Evaluation of 12 frontier LLMs on GlobalDentBench revealed a sharp, stepwise performance degradation with increasing reasoning complexity. Specifically, accuracy plummeted from 81.34% on multiple-choice to 64.53% on short-answer and 22.34% on case-based questions, while declining markedly from 74.01% at L1 to 55.64% at L2 and 35.71% at L3. More critically, risk analysis of real-world dental cases demonstrated an alarming overall unsafe rate of 31.01% in LLM-generated clinical recommendations, with 4.51% posing risks of irreversible patient harm and risks particularly pronounced in specialties such as orthodontics. These findings expose fundamental limitations in the medical reasoning and safety of current LLMs. Consequently, GlobalDentBench provides a scalable foundation for trustworthy clinical AI evaluation, underscoring the urgent need for rigorous validation before the safe deployment of these models in healthcare. 

---
# Agent-as-Peer-Debriefer: A Multi-Agent Framework with Perspective-Based Refinement for Qualitative Analysis 

**Authors**: Zhimin Lin, Kun Cheng, Fan Bai, Jie Gao  

**Link**: [PDF](https://arxiv.org/pdf/2605.24600)  

**Abstract**: Large language models (LLMs) are increasingly used for qualitative data analysis (QDA), yet their outputs often miss the depth and nuance of human analysis. We argue this gap reflects a missing credibility practice from human QDA: peer debriefing, in which an analyst seeks feedback from a disinterested peer and uses it to refine their coding. To bring this practice into LLM-assisted QDA, we propose Agent-as-Peer-Debriefer, a multi-agent QDA framework that builds peer debriefing into key coding steps. In our framework, a Hierarchical Coding Agent follows the standard QDA process to generate codes, sub-themes, and themes, along with self-explanations and reflection memos. It then shares these outputs with three Peer-Debriefing Agents, each applying a distinct analytical perspective (Theory-Driven, Data-Driven, or Applied) and refining the codes by keeping, renaming, reassigning, merging, or splitting them. These perspectives are drawn from established human QDA practices that generalize across domains and datasets. To evaluate the framework, we test it on three datasets across two domains with three LLMs, measuring semantic similarity to human-annotated codes. Across all settings, perspective-based, peer-debriefing refinement aligns more closely with human codes than a single-LLM baseline, and an ablation further shows the gain is not merely from additional refinement. The three perspectives also produce distinct trade-offs, showing that the choice of perspective is a meaningful and controllable design decision. More broadly, these findings suggest that simulating peer debriefing with explicit perspectives is a promising route to more credible LLM-assisted QDA. 

---
# Summoning the Oracle to Slay It: Mitigating Look-Ahead Bias in Financial Backtesting with Large Language Models 

**Authors**: Weixian Waylon Li, Mengyu Wang, Tiejun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2605.24564)  

**Abstract**: Backtesting large language models (LLMs) on historical financial data is unreliable because pre-training cuts off after the events happened. An LLM trained in 2024 already "knows" which way 2018-2020 stocks moved. We name this failure parametric look-ahead bias and propose FinCAD, an inference-time adaptation of Context-Aware Decoding that suppresses an LLM's memory of historical outcomes without retraining. FinCAD pairs an adversarial bias-discovery pipeline that learns a model-specific memory-activating prior prompt with an entity- and date-adaptive rule that scales the CAD strength to per-(entity, date) memorisation, so the penalty fires on memorised in-sample dates and decays to zero out-of-sample. Across five 7-14B LLMs and five mega-cap equities, FinCAD cuts in-sample backtest returns by up to -67.1% on memorised dates while leaving 2025 out-of-sample returns within $8K and Sharpe within 0.10 of baseline, and preserves general-purpose reasoning within 1.7 pts. On an eleven-model leaderboard, it raises the in-sample / out-of-sample Spearman correlation from +0.779 to +0.846, recovering rankings that genuinely predict out-of-sample performance. 

---
# Hera: Learning Long-Horizon Coordination for Device-Cloud Collaborative LLM Agents 

**Authors**: Yuxin Zhang, Mengxue Hu, Zheng Lin, Xiaoyi Fan, Fan Xie, Zihan Fang, Jing Yang, Wenjun Zhu, Zhiwen Chen, Chengfei Lv, Zhe Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.24598)  

**Abstract**: Large language model (LLM) agents excel at solving complex long-horizon tasks through autonomous interaction with environments. However, their real-world deployment faces a fundamental device--cloud dilemma: on-device models are efficient but often brittle, while cloud models are stronger but costly in computation. State-of-the-art LLM device--cloud routers usually make coarse task-level decisions, which cannot adapt to the changing difficulty of multi-step agent interactions. To address this issue, we present Hera, a step-level device--cloud LLM agent coordinator for long-horizon tasks achieving a strong performance--cost Pareto frontier. Hera adopts a novel two-stage training paradigm: (1) imitation learning for cold-start, followed by (2) reinforcement learning that jointly optimizes task success and cloud usage efficiency. The first stage casts step-level routing as a supervised classification problem: the device agent is replayed on cloud trajectories, with each state labeled by the agreement between device and cloud actions. In the second stage, we perform cost-aware reinforcement learning by grouping identical states across trajectories and updating Hera with labels favoring higher expected return and fewer future cloud calls. We evaluate Hera on ALFWorld, WebShop, and AppWorld, where it consistently outperforms prior methods, achieving 92.5% of the cloud-only success rate with cloud use in only 46.3% of steps. 

---
# Measuring Reasoning Quality in LLMs: A Multi-Dimensional Behavioral Framework 

**Authors**: Ali Şenol, Garima Agrawal, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.24661)  

**Abstract**: LLMs have achieved remarkable success in complex reasoning tasks, yet current evaluation approaches predominantly rely on final-answer correctness, offering limited insight into the underlying reasoning processes that produce those answers. To address this gap, this study proposes a unified multi-dimensional framework for measuring reasoning quality in LLMs from a behavioral perspective, operationalizing six theoretically grounded dimensions: Correctness (CQ), Consistency (CS), Robustness (RS), Logical Coherence (LS), Efficiency (ES), and Stability (SS). Extensive experiments on seven LLMs across 975 items from four benchmarks demonstrate that the framework reveals behaviors invisible to accuracy-only metrics. Notably, logical coherence is orthogonal to correctness (r = -0.172, ns), confirming that correct answers can arise from incoherent reasoning, while Claude-Haiku-4.5 achieves the highest multi-dimensional score (Q_bal = 0.778). Furthermore, the framework exposes critical ranking inversions: DeepSeek-V3 ranks second under accuracy-priority but fifth under legal/compliance weighting, a reversal that single-metric evaluation cannot detect. Discriminant validity confirms 11/15 dimension pairs are independent (|r| < 0.50), providing psychometric support for treating each dimension as a distinct signal. The dimensional profiles produced by the framework directly support three classes of deployment decision: identifying models whose reasoning traces would fail accountability audits despite correct final answers (LS--CQ orthogonality); preventing ranking errors caused by accuracy-only benchmarking; and ensuring that no single metric silently substitutes for the six independent signals the framework captures. 

---
# Learning to Reason Efficiently with A* Post-Training 

**Authors**: Andreas Opedal, Francesco Ignazio Re, Abulhair Saparov, Mrinmaya Sachan, Bernhard Schölkopf, Ryan Cotterell  

**Link**: [PDF](https://arxiv.org/pdf/2605.24597)  

**Abstract**: Many applications of large language models (LLMs) require deductive reasoning, yet models frequently produce incorrect or redundant inference steps. We frame natural language inference as a search problem where the final answer is the valid proof itself, requiring a reasoning procedure in which intermediate inferences are correct. Specifically, we investigate whether LLMs can learn to generate correct and efficient proofs with guidance from A* search -- an algorithm that guarantees an optimally efficient path to a goal. We explore two training techniques: supervised fine-tuning on execution traces from A* and reinforcement learning with A*-informed process reward models. Empirically, we find that Llama-3.2 models in the 1B--3B range benefit substantially from A* post training, going from near-zero accuracy to outperforming DeepSeek-V3.2 -- a much larger model. Our analysis uncovers a trade-off: while simple correctness rewards maximize accuracy, A*-informed signals strike a balance between accuracy and efficiency. Furthermore, we find that on larger search spaces, models trained with imperfect heuristics exhibit superior accuracy. Our results demonstrate a promising direction towards reasoning guided by principles derived from classical search algorithms. 

---
# Jailbreak to Protect: Buffering and Reinforcing via Temporary Jailbreaking for Safe Fine-Tuning in Large Language Models 

**Authors**: Seokil Ham, Jaehyuk Jang, Wonjun Lee, Changick Kim  

**Link**: [PDF](https://arxiv.org/pdf/2605.24550)  

**Abstract**: Fine-tuning-as-a-Service (FaaS) enables personalization of large language models (LLMs), but it can weaken safety-alignment under harmful fine-tuning attacks. Recent work has shown that activating harmful-behavior modules during fine-tuning can prevent models from learning undesired behaviors, but its mechanism remains unclear. In this paper, we revisit temporary jailbreaking as a defense against harmful fine-tuning and provide a gradient-level analysis showing that it saturates safety-degrading gradients while preserving benign task-relevant gradients. Based on this insight, we propose a Buffer-and-Reinforce fine-tuning framework that buffers harmful updates during user fine-tuning and reinforces safety after adaptation. Specifically, BufferLoRA induces temporary jailbreaking as a removable adapter to reduce harmful updates during user fine-tuning. After adaptation, ReinforceLoRA, trained to recover refusal behavior under the temporarily jailbroken state, is integrated with UserLoRA via QR decomposition-based merging to reinforce safety while preserving user-task performance. Extensive experiments show that our framework achieves superior safety and utility with no additional safety data during user fine-tuning and minimal computational cost. 

---
# PALoRA: Projection-Adaptive LoRA for Preserving Reasoning in Large Language Models 

**Authors**: Mustafa Hayri Bilgin, Mariam Barry, Albert Bifet, Azzedine Idir Ait Said, Soumya Banerjee  

**Link**: [PDF](https://arxiv.org/pdf/2605.24549)  

**Abstract**: Efficiently updating Large Language Models (LLMs) with new or evolving factual knowledge remains a central challenge, as even parameter-efficient adaptation can erode previously acquired reasoning abilities. This tension reflects a plasticity-stability dilemma: models must incorporate new knowledge while preserving skill-critical representations. In this work, we study this trade-off through the spectral structure of multilayer perceptron weight matrices. We show, both theoretically and empirically, that information essential for reasoning is not localized only in dominant singular directions, but is instead distributed across the singular spectrum. Motivated by this observation, we introduce PALoRA, a two-stage framework for knowledge injection with reduced interference. PALoRA first trains a Singular Value Fine-Tuning (SVF) expert on a reasoning dataset and uses its learned singular scaling vector as a frozen geometric probe to identify components that are critical for the target skill. It then performs factual knowledge injection with Low-Rank Adaptation (LoRA) under a structural orthogonality constraint, ensuring that updates avoid the identified skill-relevant subspace. Across Llama 3.1 8B and Mistral 7B, and across mathematical, coding, and scientific reasoning benchmarks, PALoRA preserves on average 95% of the SVF expert's reasoning performance while maintaining competitive factual recall. It consistently improves skill retention over prior spectral Parameter-Efficient Fine-Tuning (PEFT) methods while adding less than 0.006% parameter overhead. 

---
# Reasoning as an Attack Surface: Adaptive Evolutionary CoT Jailbreaks for LLMs 

**Authors**: Jianan Li, Simeng Qin, Xiaojun Jia, Lionel Z. Wang, Tianhang Zheng, Xiaoshuang Jia, Yang Liu, Xiaochun Cao  

**Link**: [PDF](https://arxiv.org/pdf/2605.24497)  

**Abstract**: Large Reasoning Models (LRMs) have demonstrated remarkable capabilities in reasoning and generation tasks and are increasingly deployed in real-world applications. However, their explicit chain-of-thought (CoT) mechanism introduces new security risks, making them particularly vulnerable to jailbreak attacks. Existing approaches often rely on static CoT templates to elicit harmful outputs, but such fixed designs suffer from limited diversity, adaptability, and effectiveness. To overcome these limitations, we propose an adaptive evolutionary CoT jailbreak framework, called AE-CoT. Specifically, the method first rewrites harmful goals into mild prompts with teacher role-play and decomposes them into semantically coherent reasoning fragments to construct a pool of CoT jailbreak candidates. Then, within a structured representation space, we perform multi-generation evolutionary search, where candidate diversity is expanded through fragment-level crossover and a mutation strategy with an adaptive mutation-rate control mechanism. An independent scoring model provides graded harmfulness evaluations, and high-scoring candidates are further enhanced with a harmful CoT template to induce more destructive generations. Extensive experiments across multiple models and datasets demonstrate the effectiveness of the proposed AE-CoT, consistently outperforming state-of-the-art jailbreak methods. 

---
# DemoEvolve: Overcoming Sparse Feedback in Agentic Harness Evolution with Demonstrations 

**Authors**: Lirong Che, Yuzhe yang, Peiwen lin, Chuang wang, Xueqian wang, Jian su  

**Link**: [PDF](https://arxiv.org/pdf/2605.24539)  

**Abstract**: Agent harness evolution improves frozen language-model agents by modifying the executable structures around them. We study this paradigm as a form of sample-efficient fast adaptation: instead of updating model weights, an agent can acquire task-specific competence by changing its external harness, while leaving the base model's general capabilities intact. Prior work shows that self-generated rollouts can support harness search, suggesting that agents may acquire new task competence through practice. Yet in long-horizon stochastic environments, self-practice becomes fragile: rewards are sparse, outcomes are high-variance, and failures are hard to attribute to concrete harness mechanisms. We introduce DemoEvolve, a demonstration-bootstrapped approach to harness evolution. When reward-only search is too broad and noisy, competent human trajectories serve as expert reference experience for the coding proposer, guiding harness-level diagnosis and editing. Experiments on Liar's Dice show that self-rollout evolution can work when episodes are short and failures are attributable. In contrast, Balatro exposes a harder long-horizon stochastic regime, where self-rollout evolution is misled by sparse feedback and candidate-selection noise, while tutorial-like textual knowledge alone does not yield stable improvement. Under the same limited budget, DemoEvolve produces more effective and auditable harness edits and achieves better performance. Overall, demonstrations make sparse-feedback harness evolution more diagnosable, localizable, and stable. 

---
# Market Regime Council for Dynamic Credit Assignment in Multi-Agent LLM Decision Systems 

**Authors**: Yunhua Pei, Zerui Ge, Jin Zheng, John Cartlidge  

**Link**: [PDF](https://arxiv.org/pdf/2605.24490)  

**Abstract**: Multi-agent LLM decision systems for portfolio management still lack a principled way to assign credit across specialist agents, remain vulnerable to cold-start dominance under regime shifts, and offer limited transparency into how final allocations are formed. We propose Market Regime Council (MRC), a cooperative multi-agent decision system that computes exact Shapley credits across all single, pairwise, and Grand-coalition outputs for online agent weighting. Instantiated with N=3 specialist agents, at each trading period, MRC recomputes coalition-based Shapley weights from exponentially weighted performance histories, uses a Bayesian adaptive mixture to stabilize early periods, applies regime-dependent multipliers to adjust agent authority, and records each rebalance through a five-layer causal trace. Over 1,037 trading days across 13 crypto assets and five seeds, MRC achieves a Sharpe ratio of 1.51 and a cumulative return of 440.1%, ranking first on CR, SR, and IR among active baselines and attaining the lowest MDD among active methods. Ablation results show that the gains come from Shapley-weighted integration across coalition outputs rather than from any single stage in isolation. Code and demo data are included in the supplementary material. 

---
# SAM: State-Adaptive Memory for Long-Horizon Reasoning Agent 

**Authors**: Yuyang Hu, Hongjin Qian, Shuting Wang, Jiongnan Liu, Ziliang Zhao, Jiejun Tan, Zheng Liu, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2605.24468)  

**Abstract**: Long-horizon agentic reasoning requires large language models to act over long interaction histories containing thoughts, tool calls, observations, and partial conclusions. The challenge is not merely that these histories grow long, but that information needed for the current decision may be scattered across distant steps and only become relevant later. Existing approaches address this difficulty by truncating the interaction history, compressing it into shorter surrogates, or retrieving selected parts of it for reuse, but they do not explicitly model how access to past interaction should adapt to the agent's evolving state. We instead cast long-horizon reasoning as a problem of state-adaptive memory. To this end, we propose State-Adaptive Memory~(SAM), a standalone framework that consolidates ongoing interaction into compact memory cues while preserving raw trajectory pages for intent-driven recall. These cues are not treated as replacements for history; rather, they serve as lightweight handles that allow the agent to reconstruct temporally distant information according to its current needs, without retraining the underlying backbone. We further optimize the memory module through expert-guided supervision and reinforcement learning, aligning it with trajectory-level utility. Across BrowseComp, BrowseComp-ZH, WideSearch, and HLE, SAM consistently outperforms strong baselines over diverse agent backbones. Our results suggest that explicit memory modeling provides a simple and effective foundation for long-horizon agentic reasoning. 

---
# Beyond Control-Flow: Integrating the Resource Perspective into Multi-Collaborative Process Modeling from Text 

**Authors**: Anton Antonov, Humam Kourani, Alessandro Berti, Gyunam Park  

**Link**: [PDF](https://arxiv.org/pdf/2605.24546)  

**Abstract**: Process modeling is a sub-domain of Business Process Management (BPM) focused on the translation of process artifacts into formal models. This task traditionally requires extensive human input and domain expertise in both BPM notations and the specific business context. While Large Language Models (LLMs) can now automate much of this manual work, current text-to-model approaches focus predominantly on the control-flow perspective-ordering activities without considering the collaborative aspect of the processes. In this paper, we introduce a resource-aware generation pipeline that produces formal BPMN 2.0 collaboration diagrams from natural-language descriptions. Rather than solely prompting an LLM for raw XML, we describe a compact, executable intermediate language with mandatory resource details defining both the organization (pool) and the role (lane). Cross-organization dependencies are materialized using the standard formal notation for such interactions-message events-while an orthogonal layout routine automatically handles the spatial arrangement of elements within pools and lanes. Experiments on ten business processes with nine LLMs show strong resource discovery while preserving control-flow quality and adding only marginal runtime overhead. This approach moves generative modeling toward a more comprehensive, multi-collaborative representation of business operations. 

---
# Hypothesis Generation and Inductive Inference in Children and Language Models 

**Authors**: Jeffrey Qin, Wasu Top Piriyakulki, Zhuangfei Gao, Mia Radovanovic, Jessica Sommerville, Kevin Ellis, Marta Kryven  

**Link**: [PDF](https://arxiv.org/pdf/2605.24528)  

**Abstract**: Real world decision-making requires constructing mental models under uncertainty over evidence, over the underlying causal rules, and over the state of the world itself. Which computational principles underpin human inference under such conditions, and do LLM-based agents exhibit similar behavior given matching constraints? We address these questions using an inductive inference Box Task in which participants, human children and LLM-based agents, infer a latent cause through sequential interaction with an uncertain environment. We formalize this task as program induction with Bayesian particle-based inference, admitting two complementary interpretations: (1) as a constraint satisfaction process over hypotheses, and (2) as a program synthesis problem in which hypotheses are executable programs evaluated against evidence. Using the constraint-based formulation, we show that children's behavior is best explained by a combination of subjective evidence reliability and online hypothesis generation, accounting for both their evidence-seeking patterns and their dissociation between task completion and rule generalization. Using the program synthesis formulation, we treat LLM-based agents as model organisms: controllable systems that allow systematic manipulation of task conditions. Across backends, LLM-based agents replicate children's responses to changes in evidence reliability and observability, including discounting unreliable evidence, seeking to resolve partial information, and dissociating between task completion and causal generalization. At the same time, LLM-based agents tend to over-observe and over-comply with instructions relative to children. These results suggest that while children and LLM-based agents adapt similarly to environmental structure, their information-seeking behavior exhibits distinct underlying costs and inductive biases. 

---
# Advancing Graph Few-Shot Learning via In-Context Learning 

**Authors**: Renchu Guan, Yajun Wang, Chunli Guo, Bowen Cao, Fausto Giunchiglia, Wei Pang, Yonghao Liu, Xiaoyue Feng  

**Link**: [PDF](https://arxiv.org/pdf/2605.24410)  

**Abstract**: Graph few-shot learning, which aims to classify nodes from novel classes with only a few labeled examples, is a widely studied problem in graph learning. However, existing methods often face two key limitations. First, the predominant graph few-shot learning paradigm relies on supervised tasks, failing to leverage the vast number of unlabeled nodes in the graph. Second, many approaches require complex task adaptation or fine-tuning during inference, limiting their efficiency and applicability. Inspired by the powerful in-context learning capabilities of large language models, we propose a novel model named VISION for adVancIng graph few-Shot learning via In-cOntext LearNing to address these challenges. Our model reframes graph few-shot learning as a fine-tuning-free sequence reasoning problem. At its core is a context-aware network that initializes nodes with role embeddings and employs a dual-context fusion module to synergistically integrate local topological structures and global task-level dependencies. This allows our model to dynamically generate class-aware representations for the query set conditioned on the support set context in a single forward pass. To effectively train our model, we introduce an unsupervised task generator that creates structure-adaptive features and constructs diverse pseudo-tasks from abundant unlabeled data. Our method unifies unsupervised meta-learning with graph in-context learning, achieving efficient inference. Extensive experiments on multiple benchmark datasets demonstrate the superiority of our model. Our public code can be found 

---
# Understanding and Mitigating Premature Confidence for Better LLM Reasoning 

**Authors**: Jingchu Gai, Guanning Zeng, Christina Baek, Chen Wu, J.Zico Kolter, Andrej Risteski, Aditi Raghunathan  

**Link**: [PDF](https://arxiv.org/pdf/2605.24396)  

**Abstract**: Long chains of thought (CoT) from current language models frequently contain logical gaps and unjustified leaps, limiting the gains from additional test-time compute. Improving reasoning quality directly would require process reward models, but the step-level annotations needed to train them are expensive and scarce. We find such a signal in how the model's confidence evolves during reasoning: premature confidence, the tendency to commit to an answer early and use the remaining tokens to rationalize it, strongly predicts flawed reasoning across tasks and model scales. We exploit this in progressive confidence shaping, a reinforcement learning objective that trains models to update their confidence as they reason rather than commit early -- rewarding gradual confidence growth and penalizing early commitment, with no external labels or reward models. The method improves accuracy and reasoning quality from 1.5B to 8B parameters across arithmetic (Countdown), math (DAPO, AIME), and science (ScienceQA): on Countdown, accuracy improves 3.2x (+42.0pp) and flawed reasoning drops 48pp; on AIME, Pass@64 improves 6.6pp. Consistent with this mechanism, the method also improves faithfulness: on a safety benchmark, our models more transparently surface misleading content in their reasoning traces rather than concealing it. Controlled experiments reveal that the problem and its remedy scale together: premature confidence grows with model size and task difficulty, and so do the gains from addressing it. 

---
# JT-SAFE-V2: Safety-by-Design Foundation Model with World-Context Data 

**Authors**: Junlan Feng, Fanyu Meng, Chong Long, Pengyu Cong, Duqing Wang, Yan Zheng, Yuyao Zhang, Xuanchang Gao, Ye Yuan, Yunfei Ma, Zhijie Ren, Fan Yang, Na Wu, Di Jin, Chao Deng  

**Link**: [PDF](https://arxiv.org/pdf/2605.24414)  

**Abstract**: We introduce JT-Safe-V2, a large language model designed to advance the safety and trustworthiness of foundation models, extending our previous JT-Safe model toward a more comprehensive safety-by-design paradigm. JT-Safe-V2 emphasizes the joint optimization of general intelligence and safety-by-design through several key innovations: enriching pre-training data with contextual world knowledge, high-certainty pre-training procedures, and safety strengthening post-training mechanisms for enterprise-oriented agentic capabilities. Building on these safety-enhanced foundation models, we propose Safe-MoMA (Safe Mixture of Models and Agents), a framework that enables traceable and efficient inference through the orchestrated deployment of multiple models and agents. Extensive evaluations demonstrate that JT-Safe-V2 achieves state-of-the-art performance across both general intelligence and safety benchmarks. Moreover, Safe-MoMA reduces inference costs by more than 30\% compared to using the largest standalone model baseline while maintaining comparable performance. To facilitate future research on safety-by-design foundation models, we publicly release the post-trained JT-Safe-V2-35B model checkpoint. 

---
# Distilling Game Code World Model Generation into Lightweight Large Language Models 

**Authors**: Tyrone Serapio, Arjun Prakash, Haoyang Xu, Kevin Wang, Amy Greenwald  

**Link**: [PDF](https://arxiv.org/pdf/2605.24375)  

**Abstract**: Large Language Models (LLMs) have shown great ability in generating executable code from natural language, opening the possibility of automatically constructing environments for AI agents. Recent work on Code World Models (CWMs) demonstrates that LLMs can translate game rules into Python implementations compatible with solvers like Monte Carlo Tree Search. We study this problem in game settings, where generated environments must implement rules, legal actions, state transitions, observations, and rewards. We refer to these game-specific executable models as Game Code World Models (GameCWMs). However, current approaches to generating code world models rely on frontier models and inference-time refinement loops, limiting accessibility and scalability. This work investigates whether GameCWM generation capabilities can be distilled into smaller models through post-training. We introduce: (1) a curated dataset of 30 games spanning perfect and imperfect information games, (2) a verification framework that evaluates generated code against structural and semantic game properties, and (3) a post-training pipeline combining Supervised Fine-Tuning (SFT) with Reinforcement Learning with Verifiable Rewards (RLVR). We experiment with Qwen2.5-3B-Instruct and find that SFT can increase syntactic correctness, while RLVR can improve execution-level adherence to game rules, thereby improving Qwen's ability to generate valid GameCWMs in both perfect and imperfect information games. Overall, our pipeline makes Qwen2.5-3B-Instruct more capable of generating valid GameCWMs, thereby offering a scalable path toward automatic environment generation from natural language. 

---
# When Does Synthetic Patent Data Help? Volume-Fidelity Trade-offs in Low-Resource Multi-Label Classification 

**Authors**: Amirhossein Yousefiramandi, Ciaran Cooney  

**Link**: [PDF](https://arxiv.org/pdf/2605.24296)  

**Abstract**: We study when LLM-generated synthetic data helps low-resource multi-label patent classification, separating true synthetic value from the confound that larger augmented sets can win by volume alone. Across six open-source LLMs (3.8-12B), four real-data regimes, 64 WIPO assistive-technology labels, two generation strategies, and three classifier families, the headline BERT-for-Patents micro-F1 jump from 0.120 to 0.702 is largely volume-driven. A duplicate-to-match real-only control that resamples 165 patents to the augmented size reaches 0.678; the controlled synthetic gain is only +0.024 over this control, but +0.219 over focal-loss reweighting, the strongest non-augmentation baseline. The main finding is that fidelity metrics change meaning with scale: at extreme scarcity, MMD correlates positively with classification gain (r=+0.95), but at 1:10 the relation flips (r=-0.73; Fisher z=+6.47, p<0.001). Fixed-budget mixing finds a 20-30% real / 70-80% synthetic optimum; paraphrase scaling collapses from a 165-document seed; and shuffled mixing beats curriculum ordering, ensembling, and classifier-based filtering. Leakage controls -- label-name masking, instruction-level label removal, fine-grained evaluation, and keyword-overlap audits -- argue against label-string dependence as the main driver for BERT-for-Patents. The apparent ModernBERT collapse under label removal is traced to a Flash-Attention-2 + bf16 numerical artifact, recovering 65% of lost performance with fp32 eager attention. Finally, the same corpus that improves classification by up to +0.58 raw micro-F1 hurts a Jaccard-label-overlap retrieval proxy; even a standard-patent-only filter leaves a 26% nDCG@10 drop. Thus, synthetic patent text is task- and metric-specific, not reducible to prompt genre alone. 

---
# A governance horizon for ethical-use constraints in open-weight AI models 

**Authors**: Weiwei Xu, Hengzhi Ye, Haoran Ye, Kai Gao, Vladimir Filkov, Minghui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.24383)  

**Abstract**: Ethical constraints on open-weight AI models are both a reflection of societal concerns and a foundation for AI governance policy. They are expected to propagate to downstream derivatives while implemented as voluntary metadata disclosures that must be restated at each generation of reuse. We audit 2,142,823 model repositories on Hugging Face Hub to test whether this disclosure-based governance infrastructure can sustain traceability across deep model lineages. Restriction evidence decays with a half-life of 1.31 derivation steps ($R^2$=0.98), and beyond seven downstream generations at least 80% of descendant models lack sufficient public evidence for a governance determination, a depth boundary we formalize as the governance horizon. Platform-level interventions to restore missing licence metadata reveal that policy design (not enforcement alone) is the binding factor: inheritance-only designs require near-complete enforcement to move the horizon, whereas a mandatory-declaration design that explicitly resolves orphan lineage components shifts the horizon already at moderate enforcement. The structural bottleneck is lineages with no inheritable upstream intent: such orphan components remain undecidable under any inheritance-only policy regardless of enforcement rate, and unresolved upstream nodes additionally create direct downstream undecidability bottlenecks that inheritance rules alone cannot recover. Comparison with PyPI, where governance signals are carried by explicit machine-readable declarations, corroborates that the collapse is topology-specific to open-weight derivation rather than inherent to open ecosystems. These results establish that disclosure-based governance has a shallow, structurally determined reach in open-weight AI, and that achieving deep supply-chain accountability requires provenance mechanisms propagating governance signals through derivation itself. 

---
# How Well Do Models Follow Their Constitutions? 

**Authors**: Arya Jakkli, Senthooran Rajamanoharan, Neel Nanda  

**Link**: [PDF](https://arxiv.org/pdf/2605.24229)  

**Abstract**: Frontier AI developers now train models against long written behavioral specifications, such as Anthropic's constitution (Anthropic, 2025a) and OpenAI's Model Spec (OpenAI, 2025a), integrated into post-training via methods like character training (Anthropic, 2024) and deliberative alignment (Guan et al., 2024). These documents serve a governance function, but it is unclear how well models actually follow them under adversarial, multi-turn pressure similar to what they would face in real-world deployment. We propose a multi-method audit pipeline that treats each lab's published specification as an auditable target: it decomposes the specification into atomic testable tenets (205 for Anthropic, 197 for OpenAI), generates multi-turn adversarial scenarios with the Petri auditing agent (Anthropic, 2025b), runs a modified SURF-style rubric search (Murray et al., 2026) to catch shallow single-turn failures Petri misses, validates flagged transcripts against the relevant specification, and compares the findings against the lab's own published system card. Applying the pipeline across seven models per specification, we find that models follow their own lab's specification substantially better with each generation. On Anthropic's constitution, the Claude family falls from a 15.0% violation rate (Sonnet 4) to 2.0% (Sonnet 4.6); on OpenAI's Model Spec, the GPT family falls from 11.7% (GPT-4o) to 3.6% (GPT-5.2 medium reasoning), with the severity ceiling falling from 10/10 to 7/10. We cannot externally isolate whether these gains come from specification-specific training, broader post-training improvements, or evaluation awareness. Remaining failures cluster around operator-imposed personas under AI-identity questioning, irreversible action in agentic deployments, and fabricated quantitative claims with false precision. 

---
# Safety-Oriented Routing Analysis of Mixtral MoE Under Benign and Harmful Prompts 

**Authors**: Md Nurul Absar Siddiky  

**Link**: [PDF](https://arxiv.org/pdf/2605.24270)  

**Abstract**: Sparse mixture-of-experts (MoE) language models activate only a small subset of parameters for each token, making router behavior a central part of model computation. This paper studies routing behavior of Mixtral 8x7B-Instruct under benign and harmful prompts using two complementary signals: activation-based routing scores derived from expert selection frequencies and gradient-based scores derived from router-gate sensitivities. We analyze expert- and layer-level routing behavior and conduct expert-suppression interventions. The results show that activation-based expert usage is broad and long-tailed, whereas gradient-based importance is concentrated. At expert level, benign and harmful prompt groups remain close under both signals with modest separation. At layer level, activation-based routing is most selective around layers 8-15, while gradient-based importance is concentrated in final layers. Expert classification shows most experts are shared across benign and harmful prompts, though a limited subset shows clear group preference. Top-ranked expert sets show stronger benign-malicious overlap under gradient scores than activation scores, suggesting concentration on a common late-layer expert set. In intervention experiments, suppressing top five benign-dominant experts from activation scores reduces restricted responses from 24 to 14 over 100 prompts, while suppressing gradient-derived experts reduces them from 34 to 22 with fewer unintended reversals. Overall, safety-relevant routing in Mixtral is subtle, depth-dependent, and distributed rather than dominated by a fixed set of experts. 

---
# When Does Multi-Agent RL Improve LLM Workflows? Workflow, Scale, and Policy-Sharing Tradeoffs 

**Authors**: Yifan Zeng, Yiran Wu, Yaolun Zhang, Wentian Zhao, Kun Wan, Qingyun Wu, Huazheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.24202)  

**Abstract**: Multi-agent LLM workflows route inference through specialized roles to lift end-task accuracy, but jointly training those roles with reinforcement learning is unstable in ways that are poorly understood. We study when end-to-end RL training of multi-agent LLM workflows improves over their base models, comparing Shared-Policy training, where all roles update one policy, with Isolated-Policy training, where each role has its own parameters. Our experimental matrix spans Eval-Opt, Voting, and Orch-Workers workflows, math and code tasks, and three model scales (0.6B, 1.7B, 4B). We find that multi-agent RL usually improves over base models, but gains depend jointly on workflow, task, and scale, not on policy sharing alone. Isolated-Policy tends to reach higher peak accuracy yet more often falls off a terminal accuracy cliff, while Shared-Policy training does not eliminate failure; it redistributes failure into qualitatively different patterns. We then explain the strongest of these patterns through role-level gradient dynamics induced by workflow topology and policy routing: under Isolated-Policy, parallel same-role agents on shared prompts amplify per-role gradients and drive terminal degradation in Voting and Orch-Workers workflows; under Shared-Policy, asymmetric per-step gradient mass causes the shared policy to be captured by the dominant role, producing different failure signatures by task and workflow. Together, the empirical map and its underlying mechanisms show that policy sharing routes training pressure through different channels rather than offering uniform stability, making it a design choice with workflow- and task-conditional tradeoffs. 

---
# Toward Enactive Artificial Intelligence 

**Authors**: Banafsheh Rafiee, Richard Sutton  

**Link**: [PDF](https://arxiv.org/pdf/2605.24238)  

**Abstract**: In this paper, we advocate for incorporating enactive approaches to perception and cognition into artificial intelligence (AI). Enactive approaches view perception as an active, skillful engagement with the world, where agents perceive by acting and by understanding how their actions shape their experience. This contrasts with classical views that treat perception as a passive internal process in which the brain receives sensory input, processes it, and issues commands for action. Enactive views emphasize the dynamic, embodied, and interactive character of perception, grounded in the lived experience of agents embedded in their environments. We identify and develop four key enactive concepts that we find most relevant to AI: experience, action perception inseparability, autonomy, and embodiment. Much of mainstream AI, from classical rule based systems to large language models, has largely neglected these insights, treating cognition as internal processing detached from embodied interaction and intrinsic normativity. Reinforcement learning (RL), however, exhibits structural resonance with enactive principles through its emphasis on action, agent environment interaction, feedback driven adaptation, and agent centered evaluation. However, this resonance should not be taken as theoretical equivalence, as RL approximates some enactive insights, but key elements remain absent or weakly developed. Building on this analysis, we suggest a broader incorporation of enactive ideas into both mainstream AI and RL. 

---
# A Sober Look at Agentic Misalignment in Automated Workflows 

**Authors**: Wenqian Ye, Bo Yuan, Zhichao Xu, Yijun Tian, Yawei Wang, Henry Kautz, Aidong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.24197)  

**Abstract**: We study a class of emergent misalignment in multi-agent systems (MAS), with a focus on automated workflows, which we refer to agentic misalignment. Although these systems can solve complex tasks, they often fail because agents act according to implicit proxy utilities that do not align with the intended human goals. We formally define these behaviors and analyze them within a Bayesian framework, showing that generic utilities naturally lead to posterior collapse of agents in automated workflows. To address this issue, we propose Agentic Evidence Attribution (AEA), a novel alignment paradigm that improves agent posteriors using context-specific evidence. AEA reasons over agent actions and provides structured evidence to correct misaligned behavior during collaboration. To better understand the role of evidence, we study two instantiations of AEA: self-reflection (internal evidence from the model) and weak-to-strong generalization (external evidence on the agentic trajectory). We show that a small evidence model effectively aligns the MAS by providing orthogonal failure attribution. Our results clarify the sources of agentic misalignment in automated workflows and show that evidence-based alignment can effectively improve agent collaboration and leads to reliable multi-agent systems built on automated workflows. 

---
# Palette: A Modular, Controllable, and Efficient Framework for On-demand Authorized Safety Alignment Relaxation in LLMs 

**Authors**: Qitao Tan, Xiaoying Song, Arman Akbari, Arash Akbari, Yanzhi Wang, Xiaoming Zhai, Lingzi Hong, Zhen Xiang, Jin Lu, Geng Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2605.24154)  

**Abstract**: Current safety alignment of foundation models largely follows a \emph{one-size-fits-all} paradigm, applying the same refusal policy across users and contexts. As a result, models may refuse requests that are unsafe for general users but legitimate for authorized professionals, limiting helpfulness in specialized professional settings. Existing approaches either require costly realignment or rely on inference-time steering that suffers from imprecise control and added latency. To this end, we propose \textsc{Palette}, a modular, controllable, and efficient framework that selectively relaxes refusal behavior on authorized target domains while preserving standard safety elsewhere. Our method identifies a refusal direction via multi-objective search and internalizes it into the model through lightweight adaptation. \textsc{Palette} further supports modular composition: it learns domain-specific safety controls independently and composes them through parameter merging, enabling on-demand multi-domain authorization without retraining. Experiments across four safety benchmarks, multiple model variants, and both LLMs and VLMs show that \textsc{Palette} delivers precise safety control without sacrificing general utility, offering a practical path toward foundation models that adapt to diverse professional needs. 

---
# EPPC-OASIS: Ontology-Aware Adaptation and Structured Inference Refinement for Electronic Patient-Provider Communication Mining in Secure Messages 

**Authors**: Samah Fodeh, Sreeraj Ramachandran, Elyas Irankhah, Muhammad Arif, Afshan Khan, Ganesh Puthiaraju, Linhai Ma, Srivani Talakokkul, Jordan Alpert, Sarah Schellhorn  

**Link**: [PDF](https://arxiv.org/pdf/2605.24172)  

**Abstract**: Secure patient-provider messages contain clinically important communication behaviors that are difficult to characterize manually at scale. The Electronic Patient-Provider Communication (EPPC) framework provides an ontology for coding these behaviors, but automated extraction remains challenging because predictions must preserve fine-grained code/sub-code structure while grounding annotations in message text. We developed EPPC-OASIS, an ontology-aware adaptation approach for structured EPPC extraction, and combined it with deployable inference-refinement procedures designed to improve the coherence of final annotations. EPPC-OASIS augments supervised fine-tuning with a Wasserstein alignment objective that encourages alignment between model representation neighborhoods and EPPC ontology-derived neighborhoods, while inference refinement uses verification, self-consistency, hybrid correction, and selection or ensembling to address residual prediction errors. We evaluated the framework on a de-identified corpus of secure patient-provider messages against prompting, supervised fine-tuning, preference-based, and robustness-oriented baselines across multiple open-weight language models. Across model families, the best deployable pipeline achieved 77.13% Code+Sub-code F1 and 63.83% Triplet F1, corresponding to modest but consistent absolute gains of +1.39 and +2.12 F1 points over the strongest supervised fine-tuning baseline. These results suggest that ontology-aware adaptation with structured inference refinement can support scalable retrospective EPPC mining, although external validation is needed before operational use. 

---
# HyperGuide: Hyperbolic Guidance for Efficient Multi-Step Reasoning in Large Language Models 

**Authors**: Yuyu Liu, Haotian Xu, Yanan He, Sarang Rajendra Patil, Mengjia Xu, Tengfei Ma  

**Link**: [PDF](https://arxiv.org/pdf/2605.24140)  

**Abstract**: Multi-step reasoning remains a central challenge for large language models: single-pass generation is efficient but lacks accuracy; tree-search methods explore multiple paths but are computation-heavy. We address this gap by distilling reasoning progress into a hyperbolic geometric signal that guides step-by-step generation. Our approach is motivated by a structural observation: in combinatorial reasoning trees, solution-bearing states are few while dead ends are exponentially numerous. The hyperbolic space matches this asymmetry, with compact volume near the origin and exponentially expanding capacity toward the boundary, so that distance-to-origin naturally encodes solution proximity while angular separation distinguishes branches requiring different next operations. We train a lightweight head to project LLM hidden states into this space, then fine-tune a low-rank adapter interactively on its own reasoning attempts to act on the injected signal. Across multiple benchmarks, the geometric signal yields consistent gains, with larger improvements on deeper reasoning chains. Our code is publicly available at this https URL. 

---
# Identifying and Mitigating Systemic Measurement Bias in Production LLM Inference Benchmarks 

**Authors**: Ashok Chandrasekar, Jason Kramberger  

**Link**: [PDF](https://arxiv.org/pdf/2605.24217)  

**Abstract**: As Large Language Models (LLMs) transition from research environments to production deployments, evaluating their performance against strict Service Level Objectives (SLOs) has become critical. However, current evaluation methodologies suffer from severe measurement bias at scale. We demonstrate that widely used benchmarking utilities rely on single-process, asyncio-driven architectures that introduce fundamental client-side queuing bottlenecks under high concurrency. By modeling the benchmarking client as an $M/G/1$ queue, we mathematically demonstrate how the Python Global Interpreter Lock (GIL) artificially inflates Time to First Token (TTFT) and Time Per Output Token (TPOT) metrics as request rates scale. To resolve this systematic inaccuracy, we propose an unbiased, multi-process evaluation framework that effectively distributes client-side load, ensuring negligible queuing overhead. Furthermore, we formalize a composite metric, Normalized Time Per Output Token (NTPOT), to robustly amortize end-to-end latency, including prefill and scheduling delays across sequence lengths. Our empirical evaluation demonstrates that this methodology successfully isolates pure serving engine performance, enabling accurate, reproducible profiling of LLMs at production scales exceeding thousands of queries per second. 

---
# Inference Time Context Sparsity: Illusion or Opportunity? 

**Authors**: Sahil Joshi, Prithvi Dixit, Agniva Chowdhury, Anshumali Shrivastava, Joseph E. Gonzalez, Ion Stoica, Kumar Krishna Agrawal, Aditya Desai  

**Link**: [PDF](https://arxiv.org/pdf/2605.24168)  

**Abstract**: Sparsity has long been a central theme in LLM efficiency, but its role in context processing remains unresolved. As LLM workloads shift toward longer contexts and agentic interactions, the compute and memory bottlenecks of attention become increasingly critical, raising the question of whether these constraints are fundamental. Our position is that these constraints are artificial and unnecessary, and that the future of LLM inference lies in extreme but principled sparsity along the context dimension. This position is supported by several strands of empirical and theoretical evidence. First, we find the insistence on dense attention unreasonable, since in a long context a query effectively projects O(N) attention information into a hidden space of dimension d << N, making the process inherently lossy. Second, we perform an extensive study of sparsity in LLMs spanning 20 models across five model families, varying context lengths, and different sparsity levels. We empirically demonstrate a strong trend: current LLMs, despite not being trained for context sparsity, are remarkably robust to inference-time decode sparsity across tasks of varying complexity, including retrieval, multi-hop QA, mathematical reasoning, and agentic coding. Importantly, we also show that current hardware is already sufficient to realize substantial gains from this sparsity. For example, our sparse decode kernels accelerate large-context processing by up to 10x over FlashInfer at 50x sparsity levels on hardware such as the H100. Overall, these results position extreme context sparsity not as a heuristic, but as a principled foundation for LLM inference, training, and architecture design: one that is both feasible and beneficial, and a compelling direction for future systems. 

---
# AgentFugue: Agent Scaling for Long-Horizon Tasks through Collective Reasoning 

**Authors**: Yuyang Hu, Hongjin Qian, Shuting Wang, Jiongnan Liu, Tong Zhao, Xiaoxi Li, Zheng Liu, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2605.24486)  

**Abstract**: Recent progress on long-horizon agentic tasks has been driven largely by scaling up individual agents through stronger models, better tools, and more effective scaffolding. In contrast, much less is understood about scaling out: whether multiple peer agents, all targeting the same task, can become an additional source of capability without relying on explicit role specialization or workflow orchestration. We study this question and propose AgentFugue, a collective reasoning framework built around a shared reasoning hub. As peer agents explore the same task in parallel, the hub records concise notes on what each agent has established, attempted, or ruled out, and enables each agent to selectively access what other agents have discovered in a form useful for its current search. This design turns otherwise isolated trajectories into a connected ecology of reusable intermediate reasoning without requiring centralized planning. We instantiate the hub as a plug-in communication layer, trained with supervised fine-tuning and end-to-end reinforcement learning. Across the challenging long-horizon settings we study, AgentFugue improves over strong baselines. Our results suggest that collective reasoning can turn scaling out peer agent systems into a distinct source of capability gains, rather than merely a way of spending more compute. 

---
# Reason--Imagine--Act: Closed-Loop LLM Decision Making with World Models for Autonomous Driving 

**Authors**: Zhengqi Sun, Yiwen Sun, Boxuan Liu, Tailai Chen, Tianxu Guo, Jiabin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.24004)  

**Abstract**: Large language models (LLMs) are promising for autonomous driving, but semantics-only decision policies can yield physically unsafe behavior in dynamic traffic. Existing methods either perform online language reasoning without explicit dynamics verification or use world models mainly in offline pipelines, leaving a gap between semantic intent and physical feasibility at decision time. We propose Reason--Imagine--Act (RIA), a closed-loop framework that couples an LLM reasoner with an action-conditioned world model for online safety verification. At each step, the LLM proposes an action template and candidate sub-actions, the world model performs short-horizon rollouts, and a safety scorer selects the safest executable action with feedback to the next reasoning step. Under a unified CARLA point-goal protocol (1000 episodes), RIA achieves 80.05% route completion, 51.10% arrival rate, and 0.20% collision rate. Under the same closed-loop interface, RIA consistently outperforms training-free baselines, including CARLA TM and MADA, on core closed-loop metrics. For reproducibility, code is available at this https URL. 

---
# The Model Is Not the Product: A Dual-Pillar Architecture for Local-First Psychological Coaching 

**Authors**: Alexander Mihalcea  

**Link**: [PDF](https://arxiv.org/pdf/2605.24411)  

**Abstract**: Existing language model applications struggle to meet the demand for emotionally oriented support, primarily due to their inability to maintain deep, persistent context across sessions. This report introduces Psych LM, an iOS application that validates the thesis that, for such applications, the surrounding architecture is paramount. Psych LM runs a local, on-device language model within a purpose-built, local-first runtime designed for behavioral and life-coaching applications. The system achieves the practical effect of a near-infinite context window through an automated, user-inspectable memory corpus that converts conversations into structured memory cards, including facts, goals, and events, and dynamically injects them into the prompt via semantic and vector search. As such, the system can be defined as an active-learning, retrieval-augmented generative, on-device architecture. This architecture delivers four primary contributions: a local-first design where privacy is a core property; a detailed description of the memory corpus for persistent context of key user information; a deterministic orchestration layer that provides a stable behavioral spine independent of the model's internal state; and a benchmark framework focused on evaluating the integrated system's reliability under realistic operating conditions. The R and D process confirms that complex, context-aware interaction can be reliably achieved under the strict constraints of a mobile environment by prioritizing architectural control and resource management over simple model size. 

---
# EvoCode-Bench: Evaluating Coding Agents in Multi-Turn Iterative Interactions 

**Authors**: Haiyang Shen, Xuanzhong Chen, Wendong Xu, Yun Ma, Liang Chen, Kuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.24110)  

**Abstract**: Coding agents are increasingly used as iterative development partners, but most benchmarks still evaluate one specification followed by one final assessment. This leaves out a basic question: can an agent keep its own codebase working as requirements change? We introduce EvoCode-Bench, a benchmark of 26 stateful coding tasks and 227 evaluated rounds. Each task preserves the agent's workspace for 5-15 rounds, states requirements through observable behavior, and uses cumulative executable tests to check new requirements and still-active prior ones. We evaluate 13 coding agents with two metrics: MT@4, a four-attempt fail-stop multi-round score, and SR, a single-round score from a reference-completed prior state. For most agents, SR exceeds MT@4 by 22-40 points. The gap also changes rankings: the highest-SR agent (78.9) ranks only third in persistent execution (44.0 MT@4). Even the strongest agents achieve only about 50% success on multi-turn metrics, and aggregate pass rate drops below half of round-1 performance by round 5. Failure analysis shows tier-dependent behavior: weaker agents fail early, while stronger agents survive long enough to expose specification-tracking and regression failures. We release the benchmark data and Harbor multi-turn infrastructure. 

---
# SkillEvolBench: Benchmarking the Evolution from Episodic Experience to Procedural Skills 

**Authors**: Yingtie Lei, Zhongwei Wan, Jiankun Zhang, Samiul Alam, Zixuan Zhong, Peizhou Huang, Xin Wang, Jingxuan Zhang, Donghao Zhou, Yunta Hsieh, Zhihao Dou, Hui Shen, Yan Xu, Dimitrios Dimitriadis, Tuo Zhang, Mi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.24117)  

**Abstract**: Large language model (LLM) agents accumulate rich episodic trajectories while solving real-world tasks, but it remains unclear whether such experience can be distilled into reusable procedural skills. We introduce SkillEvolBench, a diagnostic benchmark for evaluating this step from experience reuse to skill formation. It contains 180 tasks across six real-world agent environments, organized into role-conditioned task families with shared latent procedures. Agents learn from acquisition tasks, update an external skill library using compacted trajectories and verifier feedback, and then face frozen deployment tasks testing context shift, adversarial shortcuts, and composition. By comparing self-generated and curated-start skill evolution against no-skill and raw-trajectory controls, SkillEvolBench separates procedural abstraction from base capability, curated prior knowledge, and direct reuse of episodic traces. Across ten model configurations and three agent harnesses, we find that current agents often adapt locally but rarely form robust reusable skills. Skill-based conditions can improve acquisition or replay, and individual models sometimes gain on specific deployment axes, but these gains are unstable under frozen deployment. Raw-trajectory reuse frequently outperforms distilled skills, suggesting that current abstraction procedures discard contextual and procedural cues that remain useful for future tasks. Capacity and cost analyses further show that writing more skills or larger Tier-3 resource libraries is not sufficient: additional updates can improve coverage while introducing episode-specific drift and procedural clutter. These findings position SkillEvolBench as a testbed for measuring when one-off experience becomes durable procedural knowledge rather than task-local memory. 

---
# Why We Need World Models for AGI: Where LLMs Fail and How World Models May Outperform 

**Authors**: Feisal Alaswad, Batoul Aljaddouh, Maher Alrahhal, Poovammal E, Talal Bonny  

**Link**: [PDF](https://arxiv.org/pdf/2605.23972)  

**Abstract**: Large language models achieve strong performance in language generation and knowledge-intensive tasks, yet remain limited in settings requiring causal reasoning, persistent state tracking, and long-horizon planning. We argue that these limitations may arise from an objective-level mismatch between sequence prediction and reasoning over latent environment dynamics. To formalize this distinction, we introduce Latent Dynamics Inference (LDI), a conceptual perspective that interprets language and multimodal observations as partial evidence of underlying transition dynamics. To empirically investigate this perspective, we introduce Flux, a sequential reasoning environment specified entirely through natural-language rules. As a proof-of-concept case study, the rules are first compiled into an explicit state-transition simulator, illustrating that structured latent transition dynamics can, in some cases, be operationally extracted from textual rule descriptions. This enables a controlled comparison between the LLMs operating purely over textual observations and reinforcement-learning agents trained directly within the extracted latent state space. Within this case study, agents operating with explicit access to the latent state space exhibit substantially more stable behavior in long-horizon gameplay, achieving an aggregate win rate of approximately 79% versus 11% for LLMs. Qualitative analysis further reveals failure modes consistent with unstable persistent state tracking, including invalid actions, state-tracking errors, and short-horizon reasoning failures. The complete implementation of the Flux environment available at this https URL Within the evaluated setting, these results suggest that strong sequence prediction alone may struggle to support robust long-horizon dynamic reasoning without mechanisms for persistent state tracking and transition modeling 

---
# EvoSci: A Bio-Inspired Multi-Agent Framework for the Evolution of Scientific Discovery 

**Authors**: Xiaoyu Xiong, Yuqi Ren, Deyi Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2605.24018)  

**Abstract**: Large language models (LLMs), have shown strong potential in scientific discovery, yet existing methods still face substantial challenges in the design of research workflows and multi-role collaboration mechanisms. To mitigate these issues, we propose EvoSci, a multi-agent scientific collaboration framework, which integrates bio-inspired evolution with knowledge graph modeling. To iteratively generate, evaluate, and refine research ideas, EvoSci incorporates multiple role-based agents, including mentor, researcher, and reviewer. By combining collaborative reasoning, shared memory, and evolutionary feedback, EvoSci significantly enhances the coherence and creativity of scientific exploration. Experiments on real-world research topics demonstrate that EvoSci significantly outperforms strong baselines in LLM-based structured peer-review and comparative ranking evaluations, achieving the highest overall peer-review score (ICLR 4.90) and top ranking (Top-10 = 54). These results suggest its superiority in both scientific idea generation and continuous discovery. 

---
# Towards trustworthy agentic AI: a comprehensive survey of safety, robustness, privacy, and system security 

**Authors**: Jinhu Qi, Muzhi Li, Jiahong Liu, Yuqin Shu, Dianzhi Yu, Shicheng Ma, Wenqian Cui, Yiyang Zhao, Yiyi Chen, Ruoxi Jiang, Irwin King, Zenglin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.23989)  

**Abstract**: Agentic AI systems -- Large Language Models (LLMs) augmented with planning, tool use, memory, and long-horizon interactions -- can execute complex tasks autonomously, but their multi-step trajectories introduce new failure modes that challenge trustworthiness. This survey provides a focused examination of trustworthy agentic AI through two core dimensions that are critical for high-risk deployments: Safety and Robustness, and Privacy and System Security. For each dimension, we clarify key concepts, identify where risks emerge along the agent workflow, and summarize stage-targeted mitigation strategies. Other trustworthiness aspects (value alignment, transparency, fairness, and accountability) are discussed as relevant context rather than parallel chapters. To support consistent comparison and deployment decisions, we consolidate evaluation into a unified metrics-and-benchmarks hub, emphasizing both outcome and process signals (e.g., constraint violations, trace completeness, and adversarial success rates) and offering scenario-to-metric guidance for release gating. We conclude by outlining open challenges such as self-evolving agents, runtime monitoring and verification, privacy-preserving personalization, and the trust-utility trade-off, and present a case study of real-world security failures in open-source agentic systems. Our goal is to serve as a practical reference for researchers and practitioners building trustworthy agentic systems in high-stakes environments. 

---
# Breaking the Chains of Probability: Neutrosophic Logic as a New Framework for Epistemic Uncertainty in Large Language Models 

**Authors**: Maikel Yelandi Leyva-Vázquez, Florentin Smarandache  

**Link**: [PDF](https://arxiv.org/pdf/2605.24053)  

**Abstract**: Large Language Models (LLMs) are predominantly governed by probabilistic frameworks in which the sum of outcome probabilities is constrained to unity. This architectural limitation, often imposed by Softmax layers, leads to a collapse of uncertainty that makes it difficult to differentiate between epistemic uncertainty, paradox, and vagueness. We present an empirical investigation of the application of Neutrosophic Logic, a framework that treats Truth (T), Indeterminacy (I), and Falsity (F) as three independent dimensions, to model epistemic states in LLMs. We conducted experiments on a family of four OpenAI GPT models across five linguistic phenomena: logical paradoxes, epistemic ignorance, vagueness, ethical contradictions, and future contingencies, under three prompting strategies: neutrosophic, probabilistic, and entropy-derived. Our findings reveal that the neutrosophic approach, by allowing T+I+F > 1, a state we term hyper-truth, provides a richer representation of a model's internal state. In 35% of evaluations, hyper-truth emerged spontaneously, predominantly under ethical contradiction and logical paradox. We demonstrate that this approach preserves truth values in fuzzy contexts and offers a robust method for identifying and quantifying internal model conflict. We conclude that the integration of neutrosophic evaluation layers is a critical step toward more transparent, reliable, and ethically aware AI systems. 

---
# Methods for Formal Verification of Agent Skills: Three Layers Toward a Mechanically Checkable Capability-Containment Proof 

**Authors**: Alfredo Metere  

**Link**: [PDF](https://arxiv.org/pdf/2605.23951)  

**Abstract**: The companion paper introduced a four-level verification lattice on agent-skill manifests (unverified, declared, tested, formal) and left the top level aspirational. This
paper closes that gap. We give a precise semantics for skill behaviour faithful to how a skill is consumed by an LLM-driven runtime (a deterministic script-side reachable
through a non-deterministic LLM-side), state the verification problem as a capability-containment property over that semantics, and present three composable methods that
together raise a skill from declared or tested to formal: (1) sound static capability-containment analysis of the script-side via abstract interpretation over a small effect
lattice; (2) a refinement type system for tool-call envelopes that mechanically rejects any call whose statically-inferred capability is not in the manifest's declared set;
(3) SMT-bounded model checking against the parent paper's biconditional correctness criterion, with the bound chosen so any counter-example fitting the runtime's
transaction-buffer horizon is exhibited as a concrete trace. We prove the three layers composed soundly cover the parent paper's threat model modulo a single residual (the
LLM's freedom to refuse to act) that the parent paper's runtime biconditional catches at session boundary. The methods reuse existing well-engineered tools (Z3, Semgrep,
CodeQL, refinement-type checkers, mechanised proof assistants) rather than asking operators to build new ones, and the proof-carrying artifact extends the existing this http URL
convention. All three methods plus the bundle producer and re-checker ship as zero-dependency JavaScript modules in the open-source enclawed framework
(this https URL project page this https URL), with 53 unit tests and an end-to-end CLI demo on a sample skill. 

---
# QUIVER: A Formal Framework for Quantifying Perturbation Propagation and Bifurcation in Compound AI Systems 

**Authors**: Prashanti Nilayam, Sankalp Nayak  

**Link**: [PDF](https://arxiv.org/pdf/2605.23956)  

**Abstract**: Compound AI systems that chain multiple LLM calls into directed computation graphs are now the dominant architecture for production AI. Although these architectures leverage heterogeneous nodes with mixed-mode outputs, no existing framework quantifies how perturbations propagate through such pipelines, where nodes are stochastic and execution paths can diverge structurally. We introduce QUIVER, a formal framework for measuring perturbation propagation in graph-structured LLM pipelines. The framework defines: (1) a sensitivity matrix with type-dispatched distance metrics that classifies edges as amplifiers, absorbers, or threshold-sensitive, complemented by occurrence-lift; (2) trajectory divergence decomposing variation into value drift, structural path divergence, and iteration count divergence; (3) bifurcation thresholds identifying the smallest perturbation that causes structural execution path changes; and (4) distribution faithfulness, quantifying when per node evaluation datasets diverge from production distributions. We validate on two production enterprise pipelines and a public DSPy multihop QA pipeline, three structurally distinct architectures. Across 8,200+ instrumented traces (32,000+ pair comparisons), we demonstrate that QUIVER reveals distinct sensitivity profiles across architectures, distinguishes mechanistically different cascade patterns producing identical divergence rates, predicts nodes prone to trajectory bifurcation from observational data alone, and localizes stale evaluation artifacts to specific node-field categories that aggregate metrics cannot surface. 

---
# LGMT: Logic-Grounded Metamorphic Testing for Evaluating the Reasoning Reliability of LLMs 

**Authors**: Zenghui Zhou, Man Li, Xiaoke Fang, Xinyi Zhou, Weibin Li, Zheng Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2605.23965)  

**Abstract**: Large Language Models (LLMs) achieve strong performance on logical reasoning benchmarks, yet their reliability remains uncertain. Existing evaluations rely on static benchmarks, which fail to assess robustness under logically equivalent transformations and often overestimate reasoning capability. We propose LGMT (Logic-Grounded Metamorphic Testing), an oracle-free framework that leverages first-order logic (FOL) to evaluate LLM reasoning. By deriving metamorphic relations from formal logical equivalences, LGMT constructs semantically invariant test cases and detects reasoning defects through cross-case consistency checking. Experiments on six state-of-the-art LLMs show that LGMT exposes substantial hidden defects missed by traditional reference-based evaluations. We further find that models are particularly sensitive to symbol-level and conclusion-level variations, and that advanced prompting such as Few-shot CoT only partially mitigates these issues. These results suggest that LLM evaluation should move beyond isolated correctness toward robustness under logical invariance. LGMT provides a principled and scalable approach for diagnosing reasoning failures. 

---
# LC-ERD: Mining Latent Logic for Self-Evolving Reasoning via Consistency-Regulated Reward Decomposition 

**Authors**: Yanyu Chen, Jiyue Jiang, Dianzhi Yu, Zheng Wu, Jiahong Liu, Jiaming Han, Xiao Guo, Jinhu Qi, Yu Li, Yifei Zhang, Irwin King  

**Link**: [PDF](https://arxiv.org/pdf/2605.24005)  

**Abstract**: The evolution of Large Language Model (LLM) reasoning is bottlenecked by the scarcity of high-quality process data. While self-alignment via endogenous rewards offers a solution, mining valid supervision faces three challenges: (1) Label Noise via Mimetic Bias, where rewards prioritize statistical likelihood over logical truth, creating a "correctness illusion" that masks compounding errors; (2) Coarse-Grained Supervision, where sparse global outcomes (e.g., in GRPO) fail to provide granular guidance, treating reasoning chains as monolithic; and (3) Distributional Collapse, where signals fail to generalize without amplifying pre-training biases. To address these, we introduce LC-ERD (Logic-Consistent Endogenous Reward Decomposition), a framework framing self-alignment as latent structure mining. We derive a Variational Logic Potential by aggregating consensus from the model's Latent Logic Expertise (LLE) to denoise the reasoning manifold, and introduce a Multi-Agent Value Decomposition protocol based on the IGM principle to quantify individual step utility. Experiments show LC-ERD delivers a robust self-evolution path, uncovering trade-offs between logic consistency and accuracy while identifying high-value reasoning patterns missed by standard rewards. Our code is available at this https URL. 

---
# Beyond Final Answers: Auditing Trajectory-Level Hallucinations in Multi-Agent Industrial Workflows 

**Authors**: Harshada Badave, Santosh Borse, Andrea Gomez, Harshitha Narahari, Sara Carter, Vishwa Bhatt, Aishani Rachakonda, Shuxin Lin, Dhaval Patel  

**Link**: [PDF](https://arxiv.org/pdf/2605.24219)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed as autonomous agents that reason, use tools, and act over multiple steps. Yet most hallucination benchmarks still evaluate only the final output, missing failures that originate in intermediate Thought-Action-Observation steps. We present Trajel, a dataset and evaluation framework for auditing trajectory-level hallucinations in multi-agent industrial workflows. Trajel introduces a five-type hallucination taxonomy (factual, referential, logical, procedural, and scope-based) over expert-annotated agent traces from AssetOpsBench. We benchmark supervised detection models at the subtask, trajectory, and long-context levels. Our results show that the most common failure modes are missed by existing benchmarks, that nearly half of hallucinated trajectories involve multiple types at once, and that automated detectors with high binary accuracy still misclassify the subtlest types. Trajectory-aware detection significantly outperforms standard post-hoc verification, making taxonomy-grounded evaluation necessary for safer agentic deployment. 

---
# Stop Comparing LLM Agents Without Disclosing the Harness 

**Authors**: Yunbei Zhang, Janet Wang, Yingqiang Ge, Weijie Xu, Jihun Hamm, Chandan K. Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2605.23950)  

**Abstract**: This position paper argues that, for long-horizon tasks evaluated across models with comparable frontier capability, the agent execution harness, namely the infrastructure layer that governs context construction, tool interaction, orchestration, and verification around a language model, is often a stronger determinant of agent performance than the model it wraps. We formalize and defend the Binding Constraint Thesis: in this regime, performance variance is governed more by harness configuration than by model choice, and current evaluation protocols therefore systematically misattribute harness-level gains to model improvements. We support this thesis along three lines. First, a control-theoretic formalization treats the harness as the controller of a closed-loop dynamical system and the LLM as the stochastic policy it governs, which explains why small harness changes can produce performance shifts that exceed those obtained by substituting one model for another. Second, published benchmarks, industry deployments, and a controlled variance decomposition show that harness-induced variance can substantially exceed model-induced variance, including cases of model ranking reversal. Third, we propose a harness-aware evaluation framework with a disclosure standard and a variance decomposition protocol. Until harness specifications are disclosed, leaderboard comparisons for long-horizon agents should be treated as incomplete and potentially misleading. 

---
# From Accuracy to Auditability: A Survey of Determinism in Financial AI Systems 

**Authors**: Ruizhe Zhou, Xiaoyang Liu, Gaoyuan Du, Yi Zheng, Shouxi Ren, Deepayan Chakrabarti, Dengdu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2605.23955)  

**Abstract**: Deploying machine learning in regulated financial environments -- credit risk, fraud detection, and anti-money laundering -- exposes critical vulnerabilities in algorithmic reproducibility. While early financial ML addressed statistical challenges such as backtest overfitting, deep neural networks and Generative AI have introduced mechanical nondeterminism rooted in hardware and architecture. This survey provides a systems perspective on reproducibility failures across three modalities now dominant in financial AI: tabular models (post-hoc explanation variance), graph networks (stochastic sampling and temporal asynchrony), and LLM-based agentic workflows (batch-dependent divergence and trajectory drift). We supplement the literature analysis with first-party experiments on public financial datasets -- quantifying explanation rank instability in credit scoring, prediction flip rates in GNN-based fraud detection, and tensor-parallel-induced output divergence in LLM entity extraction. We propose a layered evaluation framework linking modality-specific metrics (RBO, D_cos, TDI, PSD) to audit readiness, and empirically validate the complementarity of logit-level and semantic-level determinism measures. 

---
# Accelerating Long-Tail Generation in Synchronous RLHF Training via Adaptive Tensor Parallelism 

**Authors**: Long Zhao, Qinghe Wang, Jiaan Zhu, Youhui Bai, Zewen Jin, Chaoyi Ruan, Shengnan Wang, Cheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.23945)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) has become a key post-training paradigm for improving model quality. However, the synchronous three-stage RLHF pipeline is often bottlenecked by the generation stage, where response-length skew causes the effective batch size to shrink rapidly during decoding, leaving GPUs underutilized while a few long responses remain unfinished. Mainstream frameworks employ a static tensor parallelism (TP) configuration that cannot adapt to changing batch characteristics, leaving substantial performance headroom unexplored. We propose PAT, an adaptive TP method that dynamically reconfigures TP during the generation stage of each RLHF iteration. PAT introduces two key techniques. First, a predictor-guided online reconfiguration method decides both the reconfiguration point and the target TP configuration based on offline profiling, triggering reconfiguration only when the predicted latency benefit outweighs the reconfiguration overhead. Second, a lightweight online reconfiguration mechanism updates only the states and layouts affected by TP changes: it adapts unfinished decoding states through a cost-model-based choice between KV-cache migration and recomputation, performs in-place weight resharding, and reuses cached communication groups. We implement PAT on top of SGLang and integrate it with the VeRL framework. Evaluations on LLaMA3.1-8B and Qwen3-14B using DeepScaleR show that PAT reduces generation latency by up to 34.6% and end-to-end RLHF training iteration latency by up to 27.2% compared to the original VeRL setup. 

---
# Residual Drift Dominates Contradiction in Multi-Turn Constraint Reasoning 

**Authors**: Sebastien Kawada  

**Link**: [PDF](https://arxiv.org/pdf/2605.23940)  

**Abstract**: How do multi-turn reasoning systems fail? The expected answer is logical contradiction, in which the system's maintained state becomes unsatisfiable. We show that the dominant mode is instead satisfiable drift, where the internal state stays consistent while the returned answer silently violates prior commitments. We build DRIFT-Bench (Decomposing Reasoning Into Failure Types), a solver-instrumented benchmark of 816 test problems across three constraint domains, and evaluate four methods on it across four open-weight models (8B-120B parameters). MUS-Repair, which feeds minimal unsatisfiable subsets back to the generator, is strongest in every setting (+1.8 to +15.0 pp over the best non-MUS baseline). But the central finding is what repair leaves behind. After structured feedback, models rarely contradict themselves. They forget. Residual errors are 98-100% satisfiable drift across all settings, while contradiction drops to near zero. Reliable multi-turn systems must separately validate that the returned answer respects the maintained state. Code is available at this https URL. 

---
# MEMOR-E: In-Context and Fine-Tuned LLM Personalization for Alzheimer's Assistive Robotics 

**Authors**: Maissa Abir Smaili, Eren Sadikoglu, Ransalu Senanayake  

**Link**: [PDF](https://arxiv.org/pdf/2605.23941)  

**Abstract**: Alzheimer's disease is a neurodegenerative disorder marked by progressive declines in memory and language that reduce independence in daily life, motivating socially assistive robotic support. This paper presents MEMOR-E, a mobile quadruped robot with an interactive tablet interface that assists patients and caregivers through medication reminders, routine guidance, memory oriented interactions, and companionship. We evaluated the feasibility of fine tuning large language models (LLMs) to emulate stage consistent cognitive behavior and interpret responses across standard neuropsychological language tasks, using audio transcriptions from 235 Alzheimer's patients and synthetically generated healthy controls. We also report findings on using in context learning (ICL) in LLMs, where a second LLM produced domain and severity level cognitive error summaries. Our results show that MEMOR-E can generate stage aware, non diagnostic cognitive summaries that support personalized assistive interactions, while explainable AI mechanisms translate model outputs into transparent, human readable evidence to enable caregiver oversight and trustworthy human robot interaction. 

---
# Authority Inversion in LLM-Mediated Ubiquitous Systems: When Models Trust Users Over Sensors 

**Authors**: Long Zhang, Zi-bo Qin, Wei-neng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.23938)  

**Abstract**: Large language models (LLMs) increasingly fuse heterogeneous inputs in ubiquitous systems. Yet, how LLMs implicitly allocate authority when sensor measurements and user claims conflict remains unexamined, raising critical reliability concerns for deployments where physical sensing must retain priority. Unlike explicit traditional fusion, LLMs bury authority allocation within learned representations. We discover this allocation is severely format-dependent: numerical sensor data fails to integrate into answer-relevant model directions, allowing natural-language claims to dominate the final decision, a phenomenon we term \textbf{Authority Inversion}.To diagnose and mitigate this, we develop a geometric framework of context integration, introduce two computable audit metrics, specifically the Context Integration Ratio (CIR) and Authority Alignment Index (AAI), and propose Geometric Authority Calibration (GAC), an inference-time layer-level intervention to suppress misplaced user authority. Evaluating four models (4B to 35B parameters, three architectures) across four datasets totaling 576 conflict instances reveals extreme inversion: on numerical tasks, models exhibit near-zero sensor trust (AAI = -0.805, Cohen's d = -2.14), unaffected by model capacity. Validating our geometric framework, theory-guided causal injection flips 80.2\% of incorrect decisions (vs. <0.4\% for random controls). Practically, GAC improves HAR accuracy from 0 -- 1.6\% to 21.9 -- 27.5\%, outperforming prompting baselines. Ultimately, authority allocation in LLM-mediated systems must be explicitly audited and application-specifically configured rather than left implicit. 

---
# When Correct Beliefs Collapse: Epistemic Resilience of LLMs under Clinical Pressure 

**Authors**: Boyu Xiao, Xiuqi Tian, Xuwen Song, Haochun Wang, Guanchun Song, Sendong Zhao, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2605.23932)  

**Abstract**: Despite strong medical benchmark accuracy, LLMs can exhibit severe multi-turn sycophancy in clinical dialogue, abandoning initial correct diagnosis under escalating pressure. We propose \textbf{\textsc{Med-Stress}}, a targeted stress test framework that evaluates belief stability under escalating pressure. Across nine frontier large language models (LLMs), we find a clear dissociation between medical knowledge and robustness: high initial diagnostic capability does not imply high belief stability, yielding large knowledge-robustness gaps for several LLMs. To mitigate this failure mode, we propose a lightweight inference-time defense, \textbf{\texttt{RBED}} (\textbf{R}ole-\textbf{B}ased \textbf{E}pistemic \textbf{D}efense), and \textbf{\texttt{R-FT}} (\textbf{R}esilience-oriented \textbf{F}ine-\textbf{T}uning), a training-time approach that internalizes evidence-based resistance to pressure. Experiments show that \textbf{\texttt{R-FT}} nearly eliminates belief change and substantially improves robustness. 

---
# Practical Quantum CIM Empowerment via All-Domestic-Core Agentic Large Model 

**Authors**: Wang Rui, Lu Diannan  

**Link**: [PDF](https://arxiv.org/pdf/2605.23934)  

**Abstract**: Quantum computing devices are recognized as powerful tools for solving NP-complete problems. However, the intricacy of their modeling presents notable barriers for non-specialists, while the tedious iteration of constraint weights and modeling methodologies also consumes substantial effort on the part of experts. To address these challenges, this study integrates a femtosecond laser-pumped Coherent Ising Machine (CIM) with an LLM-driven agentic system by leveraging the LangGraph and LangChain frameworks. Comprehensive investigations demonstrate that large language models (LLMs) can effectively perform such tasks in modeling as QUBO/Ising model calibration, constraint weight decision iteration and rapid validation of literature-reported schemes. Notably, all these tasks can be fully implemented based on domestic large models, combined with domestically developed CIM hardware, we truly achieve the practical empowerment of quantum CIM that fully relies on all-domestic agentic large models and hardware. This work successfully realizes robust technological integration, laying a solid foundation for subsequent research. Nevertheless, it also identifies the persisting challenges in the two cutting-edge fields of large models and quantum computing at the current stage. Encouragingly, we unexpectedly discover a promising new paradigm where accumulated knowledge from agent-assisted quantum computing iterations reciprocally enhances the agent's own problem-solving capability, thereby addressing these challenges. 

---
# Squeezing Capacity from Multimodal Large Language Models for Subject-driven Generation 

**Authors**: Shuhong Zheng, Aashish Kumar Misraa, Yu-Teng Li, Yu-Jhe Li, Igor Gilitschenski  

**Link**: [PDF](https://arxiv.org/pdf/2605.26111)  

**Abstract**: Subject-driven image generation aims to synthesize new images that preserve the identity of the given subject while following textual instructions. Existing approaches often encode text and reference images separately. This limits cross-modal reasoning abilities and causes copy-paste artifacts. Recent frameworks that connect multimodal models and diffusion models improve instruction following, but largely overlook identity preservation. To address these limitations, we condition diffusion models on Multimodal Large Language Models (MLLMs) that jointly encode text and reference images, and augment it with VAE-based identity conditioning. A novel Dual Layer Aggregation (DLA) module is designed to aggregate multi-level MLLM features for optimal conditioning, and a multi-stage denoising strategy is applied to progressively balance the semantic information from MLLM and fine-detail identity from VAE during inference. Extensive experiments demonstrate that our approach harmonizes multimodal understanding with identity preservation, mitigates copy-paste issues, and achieves superior performance regarding human preference on subject-driven image generation. Our project website is available at this https URL. 

---
# Language Models Need Sleep 

**Authors**: Sangyun Lee, Sean McLeish, Tom Goldstein, Giulia Fanti  

**Link**: [PDF](https://arxiv.org/pdf/2605.26099)  

**Abstract**: Transformer-based large language models are increasingly used for long-horizon tasks; however, their attention mechanism scales poorly with context length. To handle this, we study a sleep-like consolidation mechanism in which a model periodically converts recent context into persistent fast weights before clearing its key-value cache. During sleep, the model performs $N$ offline recurrent passes over the accumulated context and updates the fast weights in its state-space model (SSM) blocks through a learned local rule. During inference, this shifts extra computation to sleep while preserving the latency of wake-time prediction. We test our method on controlled synthetic tasks, including cellular automata and multi-hop graph retrieval, as well as a realistic math reasoning task, on which a regular transformer as well as SSM-attention hybrid models fail. We then show that increasing sleep duration $N$ for our models improves performance, with the largest gains on examples that require deeper reasoning. 

---
# Beyond Summaries: Structure-Aware Labeling of Code Changes with Large Language Models 

**Authors**: Bar Weiss, Antonio Abu-Nassar, Adi Sosnovich, Karen Yorav  

**Link**: [PDF](https://arxiv.org/pdf/2605.26100)  

**Abstract**: Code review is a critical practice in software engineering, yet the growing scale and frequency of code patches in modern projects, together with the widespread adoption of AI code assistants, make manual review increasingly challenging. Identifying the types of changes within a patch, such as renames, moves, or logic modifications, can substantially improve review efficiency by enabling prioritization, filtering, and automation. However, existing LLM-based approaches to code review have largely focused on summarization and comment generation, leaving structured code reviews underexplored. In this paper, we present a systematic study of using large language models (LLMs) for taxonomy-based labeling of code changes in a code patch. We introduce a two-stage pipeline that assigns labels to diff hunks and then refines them to capture structural relationships and semantic attributes, such as rename propagation and type changes. Our approach employs few-shot prompting to produce language-agnostic and customizable labels, without the engineering overhead of traditional static-analysis pipelines. We evaluate four LLMs across multiple context configurations on a manually curated benchmark of natural and synthetic patches. Our best configuration achieves up to $84\%$ recall and $81\%$ precision, with high accuracy in extracting relational and attribute metadata. These results suggest that LLM-based labeling can effectively complement static analysis by enabling flexible, multilingual, and automation-friendly code review workflows. 

---
# DRIVE: Modeling Skills at the Reasoning and Interaction Levels for Web Agents under Continual Learning 

**Authors**: Xirui Liu, Sihang Zhou, Yanning Hou, Rong Zhou, Haoyuan Chen, Maolin He, Siwei Wang, Hao Chen, Jian Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.23939)  

**Abstract**: Web agents require both high-level reasoning (for task decomposition) and low-level interactions (for page elements manipulation) to conduct different tasks. However, these knowledge types differ fundamentally: reasoning knowledge (e.g., booking a flight requires first searching for routes) is abstract and transferable across websites, while interaction knowledge (e.g., clicking the Search button at a specific coordinate on Site A) depends heavily on page-specific contexts. Existing methods store experiences uniformly. This creates a dilemma: abstract representations lose executability on concrete pages, while concrete representations fail to generalize across domains. This entanglement limits capability accumulation: on new websites, agents either fail to recognize reusable task logic due to surface-level differences or attempt infeasible actions from outdated page structures. To disentangle them, we propose DRIVE, a dual-level skill modeling framework separating historical experience into natural language reasoning skills, which capture transferable task logic, and programmatic interaction skills, grounding abstract actions to executable operations. A scene-aware coordination mechanism adaptively retrieves and invokes these dual-level skills based on task semantics. DRIVE also uses skill-level reflection to identify hierarchy-specific failure modes, enabling targeted skill library expansion and refinement. Experiments across five WebArena domains show DRIVE attains an average task success rate of 52.8%, exceeding the skill-free baseline by 7.3 percentage points. Further ablations show reasoning and interaction skills provide distinct, complementary benefits, supporting separation of transferable task logic from executable page-level operations. 

---
# Confidence Calibration in Large Language Models 

**Authors**: Noam Michael, Daniel BenShushan, Jacob Bien, Don A. Moore  

**Link**: [PDF](https://arxiv.org/pdf/2605.23909)  

**Abstract**: We investigate the calibration of large language models' (LLMs') confidence across diverse tasks. The results of our preregistered study show that the current crop of LLMs are, like people, too sure they are right: confidence exceeds accuracy, on average. Importantly, however, this tendency is moderated by a powerful hard-easy effect, wherein overconfidence is greatest on difficult tests; by contrast, easy tests actually show substantial underconfidence. We develop LifeEval, a test for evaluating model calibration across levels of difficulty. 

---
# In Search of the Ingredients of Open-Endedness: Replicating Picbreeder with Large Vision-Language Models 

**Authors**: Sam Earle, Kay Arulkumaran, Andrew Dai, Akarsh Kumar, Julian Togelius, Sebastian Risi  

**Link**: [PDF](https://arxiv.org/pdf/2605.23908)  

**Abstract**: We are in the midst of large-scale industrial and academic efforts to automate the processes of scientific, technological and creative production through AI-driven assistants. Historically, a fundamental property of these processes in their human form has been their open-endedness: their capacity for generating a seemingly endless supply of novel and meaningful new forms. Do artificial agents have any capacity for such fruitful unguided discovery? To answer this question, we turn to Picbreeder, the canonical exemplar of human-driven open-ended search, in which users collaboratively generated a diverse library of images through interactive evolution of small neural networks. We replicate Picbreeder, replacing human users with frontier Vision Language Models (VLMs). We observe clear qualitative differences between the output of our system and the historical human baseline, and attempt to characterize them using metrics of phylogenetic complexity and visual and semantic salience and novelty. In an effort to identify some of the causal factors contributing these differences, we study the addition of exploratory noise to the agents' selection process, of behavioral diversity between agents, and of narrative momentum in the form of memory of past actions. We make our code available at this https URL. 

---
# Context: Proactive Goal-Directed Intelligence via Composable Sandboxed Programs, Declarative Wiring, and Structured Interaction 

**Authors**: Gregory Magarshak  

**Link**: [PDF](https://arxiv.org/pdf/2605.23928)  

**Abstract**: We present Context, the intelligence layer of the Magarshak Architecture, which replaces reactive query-response chatbots with proactive goal-directed agents that advance shared tasks without waiting for user prompts. The architecture rests on three mutually reinforcing mechanisms. Write-time context assembly precomputes enriched typed attributes via Groker agents, assembling interaction context as a deterministic pure function of graph state; context blocks are byte-identical across turns between semantic changes, enabling near-100% KV-cache reuse. Composable sandboxed wisdom programs form a governed library of LM-generated imperative programs declaratively wired to goal types via typed stream relations, composed via phase ordering, and executed at interaction time without further LM calls. Proactive goal stream state machines drive conversations toward terminal states by inspecting graph state and emitting structured interaction content (option arrays, governance affordances, clarification prompts) without awaiting user input. We prove six formal results: the Context Stability Theorem, bounding per-turn LM cost as a function of semantic change rate; a Program Composition Correctness Theorem; a Declarative Wiring Soundness Theorem; the Proactive Dominance Theorem, proving proactive agents weakly dominate reactive agents on expected turns-to-terminal-state; Coordination Overhead Elimination and Quality Preservation, establishing Pareto improvements in multi-participant goal chats; and a Cross-Platform Vote Consistency Theorem. Implemented in the open-source Qbix / Safebox / Safebots stack. 

---
# StakeBench: Evaluating Language Understanding Grounded in Market Commitment 

**Authors**: Yunhua Pei, Jingyu Hu, Yiwei Shi, Hongnan Ma, Weiru Liu, John Cartlidge  

**Link**: [PDF](https://arxiv.org/pdf/2605.26074)  

**Abstract**: Existing financial NLP benchmarks often rely on labels supplied by outside observers, measuring how language is perceived rather than what speakers have committed to in the market. We introduce StakeBench, an evaluation framework for language understanding grounded in market commitment. StakeBench links 560,876 comments from 2,261 resolved markets to verified position, action, and market-odds records across Polymarket and Manifold. Supervision is derived from observable market behavior. Position sides, post-comment trading actions, and market-odds trajectories replace human annotation. Four diagnostic tasks test whether models detect market commitment, identify the revealed side, anticipate future action, and perform collective odds projection. Three commitment-aware metrics measure alignment with revealed preferences rather than perceived sentiment. Validity audits and explicit interpretation boundaries help distinguish observable commitment signals from latent belief and causal market-odds impact. Across 15 LLMs and 18 topics and platform settings, models partially recover position-side signals, with Directed Accuracy from 0.506 to 0.599, but show structural failures on later tasks. Ten of the fifteen models collapse to one or two action labels in future action anticipation, and no model consistently improves on the naive odds-direction baseline in collective odds projection. Model scale is not correlated with performance, finance-domain tuning does not improve revealed-side identification, and platform incentives strongly shape higher-order results. StakeBench is packaged with evaluation code and dataset under CC-BY 4.0. 

---
# Toward Reliable Design of LLM-Enabled Agentic Workflows: Optimizing Latency-Reliability-Cost Tradeoffs 

**Authors**: Ya-Ting Yang, Quanyan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2605.23929)  

**Abstract**: Modern AI systems increasingly rely on workflows composed of multiple interacting agents, some powered by large language models (LLMs) and others by conventional computational modules. This paper analyzes the fundamental tradeoffs between latency, reliability, and cost in LLM-enabled agentic workflows. We introduce performance models for both LLM and non-LLM agents that capture the relationship between computational effort and output quality, incorporating the impact of reasoning and output tokens for LLM agents using a parametric exponential reliability function. Then, we study the design of sequential workflows under latency and cost constraints. Main results include a water-filling token allocation policy and characterizations of optimal workflow reliability in terms of shadow prices. 

---
# How Much Thinking is Enough? Quantifying and Understanding Redundancy in LLM Reasoning 

**Authors**: Zhiyuan Zhai, Xinkai You, Wenjing Yan, Xin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.23926)  

**Abstract**: Reasoning-capable large language models solve hard problems by emitting long chains of thought, paying heavily in latency, GPU time, and energy. Casual inspection of their traces reveals extensive reformulation, verification, and circular self-reflection, yet how much of this deliberation is actually necessary has never been measured at scale or explained from first principles. This paper closes both gaps.
We formalise reasoning redundancy directly in terms of the reasoning model itself: the redundancy of a correct trace is the largest fraction of its trailing segmented steps that can be truncated while $\pi$, forced to terminate thinking and emit a final answer, still produces the correct answer. A large-scale quantification across four frontier reasoning models and two mathematical benchmarks shows that step-level redundancy is consistently high -- between 61% and 93% across the 8 (model, benchmark) conditions we study, with the median critical prefix equal to a single segmented step in six of the eight conditions -- that the finding is robust to the choice of judge family, and that although $\rho$ decreases with problem difficulty on MATH-500, all four models remain substantially redundant ($\rho \in [46\%, 85\%]$) even on the hardest Level-5 problems.
We then prove that this redundancy is a structural consequence of length-agnostic outcome rewards, not a model-specific artefact: under any such reward, no finite expected stopping time is optimal. The result holds regardless of RL algorithm, base model, data distribution, or whether the policy is obtained via RL or distillation; over-thinking is therefore not a bug to be patched in individual models but a structural property of how current reasoning models are trained. Code: this https URL 

---
# BODHI: Precise OS Kernel Specification Inference 

**Authors**: Zhiming Chang, Ziyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.23931)  

**Abstract**: The formal verification of operating system kernels requires precise specifications that capture the intended behavior of system calls. Writing these specifications manually demands deep domain expertise, motivating the use of large language models (LLMs) to automate the process. However, in OSV-Bench, a benchmark of 245 specification generation tasks derived from the Hyperkernel OS kernel, the best reported Pass@1 is 55.10%. We propose a domain knowledge prompting method (BODHI), which augments the standard few-shot prompt with a structured C-to-Python translation guide covering 15 categories of domain-specific translation patterns. Inspired by Structured Chain-of-Thought (SCoT) prompting, the guide organizes translation by separation of concerns, addressing pre-condition extraction and post-condition generation as distinct categories. Evaluated on nine models from six providers (Anthropic, Mistral, Amazon, DeepSeek, Meta, Alibaba), covering dense, mixture-of-experts and reasoning architectures, BODHI improves every model tested, with gains ranging from +11% to +32%. The best configuration (Claude Opus 4.6 + BODHI) reaches 96.73% Pass@1. BODHI reduces both syntax and semantic errors, with the strongest effect on models that have sufficient instruction-following capability to utilize structured reference material. These results demonstrate that domain knowledge injection is a model-agnostic technique that substantially bridges the gap between general-purpose code generation and formal specification synthesis. 

---
# Confidence and Calibration of Activation Oracles for Reliable Interpretation of Language Model Internals 

**Authors**: Federico Torrielli, Peter Schneider-Kamp, Lukas Galke Poech  

**Link**: [PDF](https://arxiv.org/pdf/2605.26045)  

**Abstract**: Activation oracles aim to make the activations of other models legible to humans and yield promising results compared to white-box interpretability techniques. However, uncertainty quantification (UQ) for the natural-language outputs of such activation oracles is so far understudied. Here, we investigate 6 different methods for estimating the confidence of activation oracles and evaluate how well-calibrated their confidence scores are. Our experiments on 6,000 samples per oracle (varying verbalizer and context prompts) reveal that bootstrap mode frequency is the best-calibrated method among those tested (ECE 5.7% vs. 25.5% for the answer-word log-probability on Qwen3-8B; 10.3% vs. 13.1% on Qwen3.6-27B), and that the log-prob baseline can serve as a fast triage signal at a fraction of the cost.
Code and the patched trainer are available at this https URL. 

---
# OrpQuant: Geometric Orthogonal Residual Projection for Multiplier-Free Power-of-Two Transformer Quantization 

**Authors**: Maoyang Xiang, Bo Wang, Tao Luo  

**Link**: [PDF](https://arxiv.org/pdf/2605.26092)  

**Abstract**: The deployment of Large Language Models (LLMs) and Vision Transformers (ViTs) on edge devices is significantly constrained by memory limitations and the critical timing bottlenecks introduced by dense Multiply-Accumulate (MAC) arrays. In the ultra-low bit regime, logarithmic Power-of-Two (PoT) quantization provides a hardware-efficient alternative by replacing MAC operations with bit-shifts. However, the non-uniform exponential lattice is inherently limited by a \textbf{Low Angular Resolution Regime}, a structural flaw that becomes particularly pronounced at sub-4-bit thresholds, leading to a notable degradation of high-dimensional feature manifolds.
To address this geometric limitation, we propose Orthogonal Residual Projection (ORP), an algorithm-hardware co-design framework. By formulating quantization as a dual-basis geometric projection, ORP adaptively synthesizes a higher-resolution residual lattice using strictly shift-and-add operations. Furthermore, ORP's analytical solver offers a practical alternative to computationally intensive gradient-based optimization, reducing the full-model calibration time for LLaMA-2-7B to approximately \textbf{15 minutes}.
Extensive evaluations demonstrate ORP's applicability across modalities and its hardware efficiency. Under the 3-bit (W3/A16) constraint, ORP achieves a perplexity of 6.10 on LLaMA-2-7B, comparing favorably to conventional MAC-intensive baselines like AWQ without relying on asymmetric scaling, while maintaining competitive accuracy in 4-bit scenarios. At the silicon level, standard-cell RTL synthesis at a 28nm node indicates that ORP effectively mitigates the timing bottlenecks associated with dense multiplier trees. 

---
# When Gradients Collide: Failure Modes of Multi-Objective Prompt Optimization for LLM Judges 

**Authors**: Parth Darshan, Abhishek Divekar  

**Link**: [PDF](https://arxiv.org/pdf/2605.26046)  

**Abstract**: Customizing an LLM judge to a specific task or domain often involves optimizing its prompt across multiple evaluation criteria simultaneously. Textual gradient methods automate this for a single judge criterion, however they produce natural-language critiques, not numerical vectors. Thus, the conflict-resolution toolkit of multi-task learning (PCGrad, MGDA) doesn't apply to the multi-objective textual gradient setting. We test five decomposition modes of textual gradient optimizers by varying how much cross-task information the loss, gradient and optimizer LLMs share. In 6 of 10 configurations, we observe that optimization never improves over the initial prompt. Gradient specificity drops by 59% (from 9.0 to 3.7) when the gradient LLM processes multiple criteria jointly. Separately, we observe that naively combining per-task instructions into a single prompt degrades Spearman's rho by -5.3%. These results identify two separable failure modes: optimization-time gradient dilution and inference-time instruction interference, which together constrain the design space for multi-objective judge customization using textual feedback. 

---
# DRScaffold: Boosting Dense-Scene Reasoning in Lightweight Vision Language Models 

**Authors**: Xinrui Shi, Kai Liu, Ziqing Zhang, Jianze Li, Anqi Li, Yulun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.26038)  

**Abstract**: Lightweight vision-language models perform competitively on standard benchmarks yet fail systematically in dense-scene reasoning, where multiple objects, attributes, and relations must be jointly grounded and resolved through multi-step inference. Such capability is critical for real-world applications where models must reliably interpret cluttered environments. Yet existing training signals provide no explicit grounding between reasoning steps and the underlying visual entities and relations, leaving lightweight models free to generate fluent but visually unanchored reasoning chains. To address this gap, we first introduce DRBench, a benchmark of 14,573 questions across 2,943 images, organized into five task categories spanning three progressive reasoning layers. Building on DRBench, we propose DRScaffold, a supervised fine-tuning framework that decomposes the supervision target into four causally ordered stages, enforcing grounded reasoning without architectural modification. Experiments on three lightweight VLMs demonstrate substantial gains on DRBench while preserving or improving performance on general-purpose benchmarks. Notably, Qwen2.5-VL-3B trained with DRScaffold surpasses the frozen Qwen2.5-VL-32B on DRBench, demonstrating that structured supervision can substitute for a significant portion of model scale in dense-scene reasoning. Our code and models are available at this https URL . 

---
# Retrieval-Augmented Detection of Potentially Abusive Clauses in Chilean Terms of Service 

**Authors**: Christoffer Loeffler, Tomás Rey Pizarro, Daniel Ignacio Miranda Vásquez, Andrea Martínez Freile  

**Link**: [PDF](https://arxiv.org/pdf/2605.26019)  

**Abstract**: Online Terms of Service often function as contracts of adhesion, creating asymmetries that may expose consumers to potentially abusive clauses. In Chile, assessing such clauses is legally challenging because some provisions clearly violate mandatory consumer law, whereas others depend on broader standards such as good faith and contractual imbalance. We present a retrieval-augmented generation framework for the automated detection and classification of potentially abusive clauses in Chilean Terms of Service. Designed for local execution, it combines efficient clause detection, hybrid dense--sparse retrieval, reranking, and prompt augmentation to support medium-sized open-weight language models. We also introduce the Chilean Abusive Terms of Service Extended corpus, comprising 100 contracts and 10,029 annotated clauses in 24 legally grounded categories spanning illegal, dark, and gray clauses. Experiments comparing commercial and open-weight language models, fine-tuned encoders, and traditional baselines show that retrieval-augmented prompting substantially improves performance and enables local models to approach larger cloud-based systems at lower computational and token cost. The study also contributes a refined legal annotation scheme and a practical design for AI-assisted consumer contract review. 

---
# Creative Quality Alignment: Expert Tacit Knowledge Transfer via Chain-of-Thought Fine-Tuning 

**Authors**: Bo Zou, Chao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.25977)  

**Abstract**: This paper provides an empirical implementation of the creative quality metric proposed in Calibrated Surprise (Zou & Xu, 2026a). The question this paper addresses is: does this mathematical claim hold at the engineering level?
To make the answer as general as possible, we deliberately choose the strictest engineering conditions: low data cost and a small base model. Training data comes from approximately 100 expert chain-of-thought (CoT) annotations produced by the BC Protocol (Zou & Xu, 2026b).
We also identify a data bias: most publicly available alignment datasets are skewed toward craft-related knowledge, while audience modeling and reality-logic coverage are systematically weak.
We use the term Creative Quality Alignment (CQA) to describe this class of engineering methods. We also offer a supporting theoretical observation: in an LLM with a single conditional distribution architecture, calibrating the appreciation side automatically transfers to the generation side via architectural duality. This is the structural reason why ~100 CoT examples are sufficient -- not a purely empirical observation like LIMA (Zhou et al., 2023). 

---
# QUIET: A Multi-Blank Cascaded Story Cloze Benchmark for LLM Creative Generation Capability 

**Authors**: Bo Zou, Chao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.25955)  

**Abstract**: Large language models (LLMs) face a dual challenge in creative capability evaluation: existing benchmarks (e.g., Story Cloze Test, HellaSwag) measure models' discriminative ability over narrative continuation using multiple-choice recognition paradigms, rather than directly measuring creative generation capability; rubric-based scoring and LLM-as-Judge methods rely on subjective dimension assessment or natural language model outputs, and cannot provide objective, automated scoring mechanisms.
This paper proposes QUIET (Quality Understanding via Interlocked Evaluation Testing), a diagnostic benchmark for LLM creative capability based on multi-blank cascaded story cloze. QUIET sets N blanks (10-20) in a story with complete structure, with each blank accompanied by an explicit content constraint, and cascade dependency relationships between blanks -- the content filled into earlier blanks constrains the feasible solution space for later blanks. The evaluated model (or human participants) fills all blanks in open-ended generation mode; the results are scored by an information-theoretic automated scoring protocol without human grading.
The scoring protocol directly operationalizes the "calibrated surprise" theoretical framework (Zou & Xu, 2026a). For each blank k, a composite score is computed: score = satisfy * (1 + lambda * surprise), where lambda = 1.0. Here, "satisfy" measures how well the blank filling satisfies the content constraint (objective logical reasoning judgment, not subjective aesthetic scoring), and "surprise" measures the degree of surprise given that the constraint is satisfied. Creative answers that do not satisfy the constraint score zero; answers that satisfy the constraint but are mediocre score low; answers that satisfy the constraint and are surprising score high. 

---
# Step-TP: A Grounded, Step-Level Dataset with Chain-of-Thought Reasoning for LLM-Guided Tensor Program Optimization 

**Authors**: Mengfan Liu, Da Zheng, Junwei Su, Chuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.25954)  

**Abstract**: Despite the strong reasoning capabilities of large language models (LLMs), optimizing the execution efficiency of tensor programs remains challenging due to the need for precise, composable transformation decisions. Recent LLM-guided approaches frame tensor program optimization as an iterative decision process, but existing datasets provide only end-to-end optimized program pairs using token-inefficient representations, lacking verifiable step-level supervision and interpretability. As a result, LLMs struggle to make reliable single-step decisions in large combinatorial optimization spaces. We introduce Step-TP, a post-training dataset for tensor program optimization that provides grounded, atomic, step-level supervision with structured chain-of-thought (CoT) reasoning. Step-TP forms a closed reasoning loop over intermediate program states, enabling reliable multi-step optimization rather than outcome imitation. Its design is guided by four principles: (i) a token-efficient, verifiable intermediate representation (IR) that deterministically lowers to TVM TIR; (ii) atomic and composable optimization strategies that decompose complex trajectories into interpretable single-step decisions; (iii) structured CoT supervision coupled with explicit IR-to-IR state transitions; and (iv) strategy filtering to balance coverage while preventing shortcut exploitation. The dataset and implementation are available at a GitHub link, this https URL. 

---
# SafeCtrl-RL: Inference-Time Adaptive Behaviour Control for LLM Dialogue via RL-Driven Prompt Optimisation 

**Authors**: Michael Orme, Yanchao Yu, Zhiyuan Tan  

**Link**: [PDF](https://arxiv.org/pdf/2605.25984)  

**Abstract**: Ensuring safe and contextually appropriate behaviour in Large Language Models (LLMs) remains a critical challenge for real-world deployment. We present \textbf{SafeCtrl-RL}, an inference-time behavioural control framework that enables adaptive safety regulation without model retraining or parameter modification. The method formulates dialogue generation as a sequential decision process, where a reinforcement learning agent dynamically selects prompt adjustment strategies based on contextual feedback. This allows unsafe behaviours to be suppressed through iterative refinement, which we conceptualise as inference-time behavioural unlearning. Evaluated across multiple LLMs and unsafe dialogue scenarios, SafeCtrl-RL consistently improves safety and response quality, outperforms existing prompt-based optimisation methods, and achieves favourable performance--efficiency trade-offs. **Warning: This paper may contain examples of harmful language, and reader discretion is recommended. 

---
# AI-Assisted Systematization for Evaluating GenAI Systems 

**Authors**: Dhruv Agarwal, Emily Sheng, Chad Atalla, Jean Garcia-Gathright, Hussein Mozannar, Hannah Washington, Alexandra Chouldechova, Solon Barocas, Hanna Wallach  

**Link**: [PDF](https://arxiv.org/pdf/2605.26001)  

**Abstract**: Evaluating generative AI (GenAI) systems is challenging because many targets of evaluation are broad, contested concepts, such as "reasoning," "fairness," or "creativity." When these concepts are left underspecified, it becomes unclear what should be measured or how evaluation results should be interpreted. This problem reflects a missing step: systematization, that is, moving from a broad background concept to an explicit, structured account of the concept in measurable terms. To help address the fact that systematization is cognitively demanding and resource-intensive, we investigate whether AI assistance can support this process. To enable AI-assisted systematization and assess its quality, we introduce a structured representation of a systematized concept, a concept spec, and a validation worksheet. We then develop two AI-assisted systematizers: a direct, zero-shot approach and a multi-agent approach that more closely mirrors manual systematization approaches from existing literature. We use these systematizers to produce concept specs for two concepts -- hate-based rhetoric and digital empathy -- and evaluate resulting concept specs on content validity and information recoverability. 

---
# Causal Tongue-Tie: LLMs Can Encode Causal Direction, But Their Yes/No Outputs Fail to Express 

**Authors**: Ziyi Ding, Xiao-Ping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.25891)  

**Abstract**: We find a mismatch between what large language models encode about a causal question and what they answer. On anti-commonsense CLadder items, a fixed linear probe recovers the evidence-supported answer from the model's hidden state (accuracy approximately 0.97), while the spoken Yes/No reverts to the commonsense one (accuracy approximately 0.5). We call this approximately +0.5 gap Causal Tongue-Tie: a wrong Yes/No decomposes into two separable failure modes: no internal signal versus a signal the verbal interface cannot say. The implication cuts both ways for output-only causal benchmarks: a benchmark "correct" need not mean the model has understood, and a benchmark "wrong" need not mean it cannot. Sweeping claims about whether LLMs can do causal reasoning, drawn from a single accuracy number, deserve a second look. 

---
# Can LLMs Time Travel? Enhancing Temporal Consistency in Legal Agentic Search through Reinforcement Learning 

**Authors**: Wei Fan, Yining Zhou, Mufan Zhang, Yanbing Weng, Yiran HU, Tianshi Zheng, Baixuan Xu, Chunyang Li, Jianhui Yang, Haoran Li, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2605.25920)  

**Abstract**: While large language models (LLMs) augmented with agentic search capabilities show promise for legal reasoning, they overlook a fundamental constraint that applicable law must match the temporal context of each case, as retroactive application of statutes violates core legal principles and leads to erroneous conclusions. Our observations reveal that current legal LLMs suffer from temporal bias anchored to their training cutoff, while search agents rarely incorporate temporal constraints into queries, and that web search alone cannot provide the precise statute and precedent citations that legal reasoning demands. To address these challenges, we propose LegalSearch-R1, an end-to-end reinforcement learning framework that pairs local statute RAG for precise article matching with online web search for broader legal knowledge, trained on temporally-indexed data spanning multiple amendment periods to enforce temporal consistency. Extensive experiments on our benchmark covering 13 legal tasks demonstrate that our 7B-parameter agent outperforms state-of-the-art deep research frameworks and specialized legal LLMs by 12.9% to 29.8%, surpasses baselines by 57.7% to 80.3% on temporal consistency, and exhibits robust out-of-domain generalization. The code and data are available at this https URL. 

---
# TIAR: Trajectory-Informed Advantage Reweighting for LLM Abstention Learning 

**Authors**: Muyu Pan, Shu Zhao, Nan Zhang, Philip Shin, Varun Parekh, Vijaykrishnan Narayanan, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.25850)  

**Abstract**: This paper investigates large language model (LLM) abstention learning, specifically using ternary reward, which incentivize truthfulness in large language models. This paper extends that idea by moving from a ternary reward to a Trajectory-Informed advantage reweighting, dynamically re-weights the abstention reward during Group Relative Policy Optimization (GRPO) training. The objective of this work focuses on abstention learning instead of improving truthfulness, serving as an exploration into hallucination reduction. The novelty of this paper lies in methodological innovation, advantage re-weighting, and benchmark selection. Leveraging GRPO's multiple trajectories as a natural abstention signal, this method uses a reward signal to explore knowledge boundaries and encourage consistency. By demonstrating that trajectories can be used as a confidence indicator of the policy relative to the query, they are then used to dynamically calculate the abstention advantage. AbstentionBench is used as the evaluation benchmark, as this work aims to contribute to the field of abstention learning. All datasets on the benchmark were tested against this method and various baselines. Empirical results demonstrate that TIAR achieves state-of-the-art abstention F1 scores across five of six evaluation categories, outperforming the static ternary baseline on 17 of 31 benchmark datasets while fully preserving baseline accuracy. 

---
# Clarify, Abstain or Answer? Strategising in Conversation with Belief-Augmented Generation 

**Authors**: Joris Baan, Wilker Aziz, Barbara Plank, Raquel Fernández  

**Link**: [PDF](https://arxiv.org/pdf/2605.25831)  

**Abstract**: Large language models (LLMs) define a distribution over text, which can be viewed as a probabilistic representation of uncertainty: sampling K responses yields a belief state - responses a model deems plausible. Existing work exploits this representation for narrow tasks like either decoding or selective prediction, and often requires manual interventions, not controlling generation directly. We propose Belief-Augmented Generation (BAG): grounding LLMs in their own belief state via the prompt and letting them reason over these K samples to decide on a conversational strategy: answer, clarify, or abstain. In a multi-turn ambiguous QA setting, we find that LLMs by default rarely clarify or abstain, ignoring uncertainty about the input or facts. BAG improves QA accuracy across six models and yields strategy decisions more faithful to the belief state than prompt-only baselines. Disentangling when to clarify from when to abstain, however, remains challenging. 

---
# When Search Becomes Memory: Turning Robot Design Trials into Transferable Skills 

**Authors**: Yunfei Wang, Xiaohao Xu, Yang Li, Xiaonan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.25832)  

**Abstract**: Large language models (LLMs) are increasingly used as proposal generators for evolutionary robot design, yet most loops remain memoryless: simulator results shape the next population but are not preserved as reusable design knowledge. We present Auto-Robotist, a self-evolving LLM agent that distills morphology-search traces into an explicit natural-language skill library. Each skill stores a structural archetype, evidence-grounded positive and negative rules, and the evaluated designs that support them, making design memory inspectable rather than implicit in a population. During search, the agent retrieves skills to condition LLM edits of elite bodies while retaining a Genetic Algorithm (GA) mutation path for exploration; after evaluation, it updates the library through Add, Diagnose, and Merge. Across seven EvoGym tasks spanning locomotion, traversal, and object interaction, Auto-Robotist improves cold-start 5x5 search and transfers learned skills to 10x10 design spaces, where reference-conditioned transfer outperforms GA on every task. These results suggest that LLM agents can convert expensive physical evaluations into reusable, auditable design principles. Our code will be released upon acceptance. 

---
# Adaptive Graph Refinement and Label Propagation with LLMs for Cost-Effective Entity Resolution 

**Authors**: Hongtao Wang, Renchi Yang, Haoran Zheng, Xiangyu Ke  

**Link**: [PDF](https://arxiv.org/pdf/2605.25814)  

**Abstract**: Dirty entity resolution (ER), which identifies records referring to the same real-world entity from a single, messy dataset, is a fundamental task in data management and mining. However, the dominant blocking-matching-clustering paradigm for ER suffers from critical flaws. Its cascaded, decoupled workflow essentially produces a static, sparse graph plagued by missing edges (due to blocking failures) and noisy links (due to matching errors), causing error propagation and yielding suboptimal clusters, particularly when rigid transitivity is imposed in the clustering. We contend that matching and clustering are fundamentally synergistic, both optimizing for the construction of an ideal entity graph. Building upon this insight, we propose Alper, a unified framework that integrates these steps into an iterative probabilistic label propagation process over a global, evolving graph. Unlike disjoint blocking, Alper refines the graph structure and labels dynamically by adaptively integrating "weak but cheap" signals from graph propagation with "strong but expensive" LLM-based pairwise queries. For higher cost-effectiveness, we formulate the signal selection as a constrained optimization problem maximizing cumulative marginal gain under a query budget, solved via our greedy algorithm with provable theoretical guarantees. Our extensive experiments over eight benchmark datasets demonstrate that Alper is consistently superior to state-of-the-art cascaded pipelines. 

---
# Context-Instrumental Data Distillation for Kubernetes Manifest Generation: Method and Experimental Evaluation 

**Authors**: Andrey Kozachok, Anatoliy Bakaev, Aleksandr Kozachok, Shamil Magomedov, Artem Noev  

**Link**: [PDF](https://arxiv.org/pdf/2605.25835)  

**Abstract**: This paper examines the specialization of Small Language Models (SLMs) with up to 4 billion parameters for generating artifacts in domain-specific languages (DSL). Kubernetes manifests are chosen as the target domain. We propose the context-instrumental data distillation method: the source corpus is formed through synthetic generation and, in an extended scheme, through reverse instruction generation from real Kubernetes YAML files, with pairs included in training only upon passing external validators and matching the domain context model. Unlike classical KL-divergence knowledge distillation, the baseline implementation reduces to supervised fine-tuning on instrumentally verified examples. The experimental section presents a pilot implementation under resource-constrained conditions: the DeepSeek-V4 Flash API serves as the teacher for synthetic generation, while Qwen2.5-Coder-1.5B-Instruct is fine-tuned via LoRA on CPU. On the K8s-Distill-Pilot corpus (train_1200, validation_100, test_200), we achieved full-pass@1 = 91.5% (183/200) with a stricter prompt formulation and max_new_tokens=768. The key empirical finding is that for Kubernetes YAML, result quality in the pilot depended more on strict output format requirements than on simply increasing the number of training examples. 

---
# Geometric Evolution Maps: Extracting Stable Concept Probes from Transformer Residual Streams 

**Authors**: James Henry  

**Link**: [PDF](https://arxiv.org/pdf/2605.25848)  

**Abstract**: Concept probes extracted from transformer residual streams are only as reliable as the layer from which they are extracted. The common practice of probing at a fixed late layer or at the peak of a separation score function ignores a fundamental structural feature: concept representations undergo substantial directional rotation during their assembly phase, and do not settle into a stable direction until a characteristic handoff layer after the primary Concept Allocation Zone (CAZ). We introduce Geometric Evolution Maps (GEMs), which track the full directional trajectory of a concept through residual stream activations, identify the handoff layer where rotation ceases, and extract the settled probe direction from that layer. Across 23 architectures spanning 70M to 14B parameters and 17 concept types, the entry-to-exit cosine similarity within CAZs has a mean of 0.233, showing that probe direction at CAZ entry does not reliably predict probe direction at exit. Ablation experiments across 391 concept x model pairs (23 models x 17 concepts) show that GEM-extracted probes are at least as precise as peak-layer probes in 268/391 trials (68.5%), and strictly outperform in 259/391 (66.2%). The architecture split is pronounced: MHA models favour the handoff in 173/221 trials (78.3%); GQA models favour the handoff in only 56/119 trials (47.1%). Model-level Wilcoxon: W=214, N=23, p=0.010 (one-sided). An adaptive ablation width rule targets the 79/391 near-final-layer cases: it improves probe quality in 60/79 triggered cases (75.9%), mean gain +7.44pp. A direction-specificity control confirms the ablation effect is concept-direction specific: median 377x suppression rate versus random-direction ablation (99.1% of concept directions beat all 10 random seeds). Reference implementation: rosetta_tools v1.3.1 (doi:https://doi.org/10.5281/zenodo.20361433). 

---
# Explaining Too Much? Understanding How Large Language Model Reasoning Traces Influence Performance and Metacognition 

**Authors**: Daniela Fernandes, Daniel Buschek, Lev Tankelevitch, Thomas Kosch, Robin Welsch  

**Link**: [PDF](https://arxiv.org/pdf/2605.25856)  

**Abstract**: Large Language Model interfaces are increasingly verbose, exposing intermediate reasoning traces alongside final answers. Traces are framed as transparency mechanisms, yet it is unclear how people use them to solve problems. We report a preregistered between-subjects study (N = 559) in which participants solved ten LSAT-style reasoning problems under one of three conditions: an Answer-only baseline, a Full-trace revealed before the answer, and a Summary-trace presented alongside the answer. Summaries preserved task performance at the no-trace baseline while significantly elevating trust and hedonic appeal, establishing that trace exposure shifts subjective appraisal of the interaction without bringing performance benefits. Under an open-weight reasoning model exposing verbose intermediate output, full traces additionally impaired performance relative to the answer-only baseline. Across all conditions, participants substantially overestimated their performance, and no trace format supported calibrated self-evaluation. Further analysis indicates that hedonic appeal, not trust, carries the indirect path to overestimation, consistent with a processing-fluency account. Reasoning traces are best understood as user-facing interface artifacts rather than transparent windows into model cognition, and calibration is unlikely to emerge from the traces themselves and may best be scaffolded by interactions that elicit users' own reasoning first. 

---
# TTPrint: Evidence-Grounded TTP Extraction via Diverge-then-Converge Verification 

**Authors**: Yutong Cheng, Changze Li, Raihan Sultan Pasha Basuki, Qian Cui, Wei Ding, Peng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2605.25836)  

**Abstract**: Extracting MITRE ATT&CK techniques from cyber threat intelligence (CTI) reports is an open-set, multi-label problem requiring both high recall (not missing techniques) and high precision (not hallucinating unsupported ones). Existing methods--rule-based, supervised, and LLM-based--struggle to achieve both: rule-based and supervised approaches lack generalizability across diverse attack descriptions, while LLM-based approaches that couple candidate generation and validation within a single inference step suffer from limited recall and precision simultaneously. We propose TTPrint, which addresses this challenge through a diverge-then-converge design inspired by how human analysts work: first extracting broadly, then verifying rigorously. In the divergent phase, reports are decomposed into atomic behaviors and candidate techniques are proposed broadly. A deterministic span localization stage then anchors each candidate to a specific evidence window in the source text. A convergent verification stage retains only candidates supported by both the localized evidence and the authoritative MITRE definition. We contribute two evaluation resources--a cleaned TRAM benchmark (TRAM-Clean) and a new annotated dataset (TTPrint-Bench)--to address known annotation noise in existing benchmarks and elevate the task to document-level TTP extraction. On TRAM-Clean and TTPrint-Bench, TTPrint achieves 76.48% and 87.39% macro-F1 respectively, outperforming the leading baseline by 63.5% and 29.4%. A multi-backbone analysis across six LLMs and a threshold sensitivity study further demonstrate generalizability across model choices and provide practical guidance for parameter selection. 

---
# Multi-Agent Coordination Adaptation via Structure-Guided Orchestration 

**Authors**: Haoran Li, Shulun Chen, Shaoyuan Sun, Hanchen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.25746)  

**Abstract**: As large language model (LLM)-based multi-agent systems scale to handle increasingly complex tasks, balancing structural stability and dynamic adaptability becomes increasingly challenging. Existing systems typically adopt either structure-centric methods, committing to structures determined upfront that limit fine-grained control, or orchestration-centric methods, adapting decisions dynamically while leaving coordination structure implicit and unstable. To address this challenge, we revisit multi-agent coordination from a probabilistic perspective, casting it as posterior inference over the joint distribution of structure and orchestration. We introduce MACA, an automated coordination framework that learns a task- and budget-conditioned structural prior over agent participation and interactions. This prior guides a policy-based orchestration as an approximation to posterior inference, enabling efficient solutions with fine-grained control. Across benchmarks, MACA outperforms adaptive multi-agent baselines by an average of 8.42% while using 43.19% fewer tokens. Further investigation reveals that joint adaptation of structure and orchestration suppresses redundant interactions, converging coordination toward task-effective execution. 

---
# How Should LLMs Consume High-Quality Data? Optimal Data Scheduling via Quality-Aware Functional Scaling Laws 

**Authors**: Zhitao Zhu, Xili Wang, Shizhe Wu, Jiawei Fu, Xiaoqing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.25698)  

**Abstract**: High-quality data is scarce in large language model (LLM) training, yet how to schedule its use jointly with training dynamics lacks theoretical guidance. We extend functional scaling laws by incorporating a data-quality dimension, and solve the joint data-quality and batch-size scheduling problem in asymptotic closed form. The solution reveals two regimes and a dual role of high-quality data. In the noise-limited regime, high-quality data should be used as a signal amplifier: lowering the batch size converts cleaner data into more signal without amplifying noise. In the signal-limited regime, it should be used as a noise suppressor: late placement reduces terminal noise without sacrificing signal accumulation. Existing curriculum-style pipelines primarily exploit the second role by placing cleaner data late, but miss the first role because conventional decay schedules reduce update intensity exactly when high-quality data becomes available. Guided by this, we propose Drop-Stable-Rampup for LLM midtraining: upon the quality transition, drop the batch size, hold it stable to accumulate signal, then ramp up to suppress terminal noise. On a 15B Mixture-of-Experts model midtrained on 108B tokens, Drop-Stable-Rampup improves average accuracy over Warmup-Stable-Decay (WSD) by +1.70 and over Cosine-decay by +2.98, with particularly large gains on mathematical reasoning benchmarks such as GSM8K (+4.23) and MATH (+2.80). 

---
# Profiling-Driven Adaptive Distributed Transformer Inference on Embedded Edge Deployment 

**Authors**: Muhammad Azlan Qazi, Alexandros Iosifidis, Qi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.25682)  

**Abstract**: Distributing Transformer inference across embedded edge devices can alleviate individual memory and compute constraints, yet practical benefits on real hardware remain unclear: prior work relies largely on simulations that overlook hardware-specific communication overheads. We present a hardware prototype study on NVIDIA Jetson Orin Nano devices connected over WiFi. Our key finding is that the dominant bottleneck is not just network bandwidth but also the CPU-GPU staging during communication. Because Jetson's integrated GPU architecture lacks the PCIe/NVLink pathway that NCCL requires, all inter-device data communication should be routed through GLOO and staged in CPU memory; an overhead that scales with communication data volume and makes full-tensor exchange slower than single-device inference across the batch sizes for medium sized models such as ViT. We therefore evaluate Prism by combining Segment Means compression with lightweight offline profiling to adaptively select between local and distributed execution at runtime. Experiments show that this strategy reduces latency by 65%-77% and energy consumption by 34%-52% relative to full-tensor exchange in static distributed execution setup, demonstrating that profiling-driven adaptation is essential for practical distributed Transformer inference on embedded hardware. 

---
# SAMark: A Self-Anchored Text Watermarking with Paragraph-Level Paraphrase Robustness 

**Authors**: Jiahao Huo, Wenjie Qu, Yibo Yan, Kening Zheng, Jiaheng Zhang, Xuming Hu, Philip S. Yu, Mingxun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.25796)  

**Abstract**: Semantic-level watermarking (SWM) improves robustness against text modifications by treating sentences as the basic unit. However, robustness to paragraph-level paraphrasing remains difficult because such attacks globally disrupt watermark signals by changing sentence order. In this work, we propose SAMark, a self-anchored watermarking framework that removes the dependency on sentence order by establishing a step-independent green region in semantic space. To improve detectability, we introduce a multi-channel hyperbolic scoring mechanism that amplifies watermark signals while suppressing noise from weakly aligned candidates. We further propose a diversity-aware filtering strategy that combines hard filtering with soft regularization, extending beyond simple n-gram repetition filters to address semantic redundancy. Experimental results show that SAMark achieves up to 90.2% TP@FP1% under typical paragraph-level paraphrasing attacks, outperforming the strongest prior baseline by more than 30% on average, while maintaining generation quality competitive with unwatermarked text and breaking the robustness-quality trade-off that limits prior methods. 

---
# Efficient Benchmarking Is Just Feature Selection and Multiple Regression 

**Authors**: Sam Bowyer, Acyr Locatelli, Kris Cao  

**Link**: [PDF](https://arxiv.org/pdf/2605.25773)  

**Abstract**: Efficient benchmarking techniques aim to lower the computational cost of evaluating LLMs by predicting full benchmark scores using only a subset of a benchmark's questions. By reframing this problem as an instance of multiple regression with feature selection, we find that existing efficient benchmarking methods can be greatly improved by simply using kernel ridge regression at the prediction stage. Additionally, using an information-theoretic feature-selection algorithm called minimum redundancy maximum relevance (mRMR), we can further improve upon these methods by selecting question subsets that will be maximally useful for prediction. Except in very data-poor settings, these approaches consistently achieve smaller prediction errors (in both MAE and RMSE), and greater ranking correlation between predicted and true scores (in both Spearman $\rho$ and Kendall $\tau$) across a range of benchmarks using both binary and continuous metrics. Furthermore, mRMR subsampling is much faster than competitor methods (which often involve fitting probabilistic models or running clustering algorithms), and is more likely to select the same questions under different random seeds or training data splits. Tutorial code can be found at this https URL . 

---
# Simulating Human Memory with Language Models 

**Authors**: Qihan Wang, Nicholas Tomlin, Michael Hu, Brian Dillon, Tal Linzen  

**Link**: [PDF](https://arxiv.org/pdf/2605.25680)  

**Abstract**: Language models are increasingly being deployed as user simulators, but their memory is far more reliable than that of real users. To measure this gap, we run a series of classic memory experiments from psychology on both humans and language models. Across tasks, we find that out-of-the-box language models exhibit better memory than humans, even when prompted to imitate human behavior. We then show that better prompting strategies and the use of a compactor can cause language models to forget content in a more human-like way. Using these methods, we show preliminary evidence that language models with human-like memory constraints can function as more effective user simulators in a downstream education task. Finally, we release human reference data and benchmarks to support future work on simulating human memory with language models. 

---
# Referential Security as a New Paradigm for AI Evaluations 

**Authors**: Dan Ristea, Vasilios Mavroudis  

**Link**: [PDF](https://arxiv.org/pdf/2605.25673)  

**Abstract**: Security evaluations inherently depend on stable identifiers. Any finding, audit, or regulatory decision must remain attached to the specific artifact it pertains to. Continuously updated artificial intelligence systems violate this core assumption, with public model designations remaining static while underlying weights, prompts, retrieval mechanisms, misuse classifiers, inference settings, and serving infrastructures undergo unannounced modifications. Consequently, current evaluations frequently apply to superficial labels rather than identifiable and distinct systems.
To resolve this, we propose referential security as a new paradigm for AI evaluation. The fundamental security question extends beyond whether a model is safe to whether subsequent parties can conclusively determine which system a specific safety claim addressed. This approach reframes model identity as an empirically verifiable property and separates referential stability from the substantive security claims it conditions. This framework brings tractability to three critical workflows that current practices handle poorly. Specifically, it enables reproducible evaluation, longitudinal audit validity, and cross-provider equivalence. By grounding these evaluations in verifiable artifacts, our approach ensures that safety audits and regulatory findings maintain their empirical utility across the operational lifecycle of dynamic systems. 

---
# Fine-Tuning and Serving Gemma 4 31B on Google Cloud TPU: A Technical Comparison with GPU Baselines 

**Authors**: Jatin Kishnani, Mayank Goel, Amit Singh, Pulkit Agrawal, Sairanjan Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2605.25645)  

**Abstract**: We present the first end-to-end demonstration of fine-tuning and serving Google's Gemma 4 31B model on TPU hardware, providing an empirical comparison of TPU and GPU platforms for large language model adaptation. Using LoRA on a Google TPU v5p-8 for training and TPU v6e-8 (Trillium) for inference, we document the full set of code-level adaptations required to port a GPU-native training recipe, built on PyTorch, HuggingFace TRL, and FSDP, to the JAX + Tunix/Qwix stack. These adaptations span mesh configuration, LoRA module naming conventions, sharding annotation corrections, gradient checkpointing, data pipeline restructuring, and a custom Orbax-to-safetensors checkpoint merging procedure.
For inference, we detail the vLLM-TPU Docker setup necessary to serve Gemma 4 on v6e-8 and characterize the resulting latency and throughput profile. Compared with a 2xH100 GPU baseline under identical hyperparameters, TPU training completes 1.61x faster at 2.12x lower cost. Inference throughput is within 3% across platforms, while TPU achieves 2x lower time-to-first-token (235 ms vs. 475 ms). Together, the TPU configuration is 1.82x cheaper for a representative train-plus-service workload.
Our work removes a critical gap in the open tooling ecosystem and provides practitioners with a reproducible, production-ready recipe for Gemma 4 deployment on TPU infrastructure. 

---
# Extreme Region Policy Distillation 

**Authors**: Changyu Chen, Xiting Wang, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2605.25582)  

**Abstract**: Reinforcement learning for large language models faces a fundamental trade-off between sample efficiency and asymptotic performance: strictly on-policy methods discard trajectories after a single update, while off-policy reuse introduces distribution mismatch that existing trust-region techniques mitigate primarily by enforcing conservative optimization, often leaving rich training signals underutilized. To investigate this, we perform extensive off-policy updates on fixed data. Our experiments reveal that aggressive multi-step optimization brings rapid initial gains, but excessive updates cause trajectory probabilities to deviate and entropy to collapse, with performance plateauing early. Tightening KL constraints merely lowers the ceiling without resolving the degradation. This motivates Extreme Region Policy Distillation (ERPD), a two-stage framework that decouples sample efficiency from KL efficiency. The first stage performs weakly constrained off-policy optimization on fixed data to maximally extract training signals. The resulting policy provides token-level supervision. In the second stage, we distill these signals into the base policy under trust-region constraints, filtering harmful drift while preserving useful signals. The distilled policy achieves comparable or better performance with substantially smaller KL divergence, indicating that much of the first-stage divergence was spent on unnecessary drift rather than genuine improvement. Crucially, ERPD accommodates both strong and weak teachers: when aggressive optimization yields no stronger policy, even degenerate teachers provide effective supervision via alternative signal construction strategies. We validate ERPD on mathematical reasoning, showing gains for strong base models where on-policy training plateaus, and reliable improvements with weak teachers. 

---
# Toward a Benchmark for Controllable Simulation of Imperfect Students with Large Language Models 

**Authors**: Alexander Apartsin, Omri Sason, Yehudit Aperstein  

**Link**: [PDF](https://arxiv.org/pdf/2605.25601)  

**Abstract**: Teacher education requires deliberate practice with learners who exhibit identifiable strengths, weaknesses, and partial mastery. Large language models could support such practice by simulating students with known skill components, enabling teachers to rehearse explanations, diagnoses, and instructional responses. For this purpose, however, the central requirement is neither to maximize benchmark accuracy nor to suppress isolated facts, but to control model behavior so that it reflects a specified skill profile. This paper investigates whether prompted language models can be steered to retain some skills while suppressing others. We introduce a benchmark-oriented framework in which an explicit skill vector represents a simulated student, prompt-based control specifies retained and missing competencies, and behavior is evaluated using profile-alignment metrics, retained-versus-forgotten comparisons, and cross-skill calibration analyses. The results show that selective partial mastery can be induced and measured in a structured mathematics setting, although the degree of controllability remains model-dependent. These findings position controllable learner simulation as a distinct research problem at the intersection of teacher education, educational simulation, and language-model control. 

---
# AutoSG: LLM-Driven Solver Generation Solely from Task Prompts for Expensive Optimization 

**Authors**: Haoran Gu, Handing Wang, Yi Mei, Mengjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.25658)  

**Abstract**: Expensive optimization tasks are ubiquitous in real-world applications, demanding highly specialized solvers. While LLM-driven automated solver generation shows promise, current paradigms face three critical issues when tackling expensive optimization: factual hallucinations due to deficient domain knowledge, the frequent dismantling of previously established locally optimal structures during refinement, and the prohibitive evaluation costs alongside restricted generalization caused by executing on training instances. To address these issues, we introduce AutoSG, a fully automated workflow directly translating natural language prompts into executable customized solvers. AutoSG features three core innovations: a retrieval-augmented solver generation module strictly grounding code in verified literature; a one-step self-refinement operator introducing task-specific improvements while preserving critical structural components; and an instance-free Elo-based LLM-as-a-Judge evaluation mechanism rapidly establishing global rankings. Extensive evaluations across diverse expensive optimization tasks confirm AutoSG significantly outperforms human-designed state-of-the-art frameworks and existing LLM-generated solvers. 

---
# Meta-Engineering Harnesses for AI-Native Software Production: A Contract-Driven Adversarial Verification Architecture with Early Deployment Report 

**Authors**: Satadru Sengupta, Tamunokorite Briggs, Ivan Myshakivskyi  

**Link**: [PDF](https://arxiv.org/pdf/2605.25665)  

**Abstract**: AI-native software development is often evaluated at the level of individual models, prompts, or generated artifacts. This framing is insufficient for production environments where software must be continuously produced, verified, deployed, maintained, and adapted across many operational contexts and long time horizons.
We present a meta-engineering harness: a software-production architecture that transforms operational and product feature requirements into explicit contracts, routes work through role-specialized AI agents, performs independent and adversarial verification, and continuously improves itself through structured failure classification and outer-loop calibration.
The harness is designed for settings in which software delivery is not a one-time project but an ongoing operating function. In our motivating application, CTO-as-a-service for small service firms, the system manages websites, booking flows, payment systems, backoffice workflow automations, and AI-agent interfaces as continuously evolving technical infrastructure rather than one-off deliverables.
We describe the layered architecture, including two-pass contract compilation, persistent markdown memory with specialization records, attention-based and independence-based verifications, a four-way failure arbiter, and outer-loop calibration. We report results from an early production deployment spanning 17 features over several weeks, including a detailed in-app payments case study that revealed contract incompleteness and verification-boundary issues. These observations directly drove targeted improvements to the harness.
The contribution is an implemented, measurable, and extensible verification architecture for making AI-native service-as-a-software production reliable, auditable, and improvable over time. 

---
# PennySynth: RAG-Driven Data Synthesis for Automated Quantum Code Generation 

**Authors**: Minghao Shao, Nouhaila Innan, Hariharan Janardhanan, Muhammad Kashif, Alberto Marchisio, Muhammad Shafique  

**Link**: [PDF](https://arxiv.org/pdf/2605.25572)  

**Abstract**: The growing complexity of quantum programming frameworks has exposed a critical limitation in existing large language model (LLM)-based code assistants: general-purpose models hallucinate PennyLane-specific gate names, misplace device configurations, and produce structurally invalid circuits when faced with specialized quantum coding challenges. We present PennySynth, a retrieval-augmented generation framework that addresses this gap by conditioning LLM inference on a curated knowledge base of 13,389 PennyLane instruction-code pairs, built via a three-stage extraction, verification, and deduplication pipeline over official PennyLane repositories, community GitHub sources, and QHack competition archives. PennySynth introduces a code-aware embedding strategy using st-codesearch-distilroberta-base, trained for natural-language-to-code retrieval, increasing average retrieval cosine similarity from 0.45 to 0.726 compared to a general-purpose baseline. Evaluated across 74 challenges spanning three years of the QHack competition (2022, 2023, 2024), PennySynth achieves 64%, 68%, and 52% pass@5 on QHack 2022, 2023, and 2024, respectively, improving over Claude Sonnet 4.6 without retrieval by +28, +25, and +28 percentage points. We further introduce a quantum-adapted CodeBLEU metric that upweights qml.* token patterns and show that structural code similarity and functional correctness capture distinct aspects of quantum code quality. Controlled ablations reveal that code-aware embeddings are the primary driver of retrieval performance, while dataset expansion and source composition provide additional gains when retrieval quality is sufficiently precise. 

---
# BC Protocol: Structured Dual-Expert Dialogue for Eliciting High-Quality Chain-of-Thought Post-Training Data 

**Authors**: Bo Zou, Chao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.25549)  

**Abstract**: High-quality expert chain-of-thought (CoT) data is one of the core bottlenecks in large language model (LLM) post-training. Existing data production methods each have structural limitations: crowdsourced annotation lacks deep reasoning paths; expert solo writing is constrained by the "expert blind spot" -- experts structurally skip reasoning steps they consider obvious; RLHF only produces preference signals rather than reasoning chains.
This paper proposes the BC Protocol -- a structured dual-expert elicitation method for LLM post-training data production. The method carefully pairs a domain expert (crystallized intelligence) with a knowledge engineer (fluid intelligence), systematically externalizing the expert's implicit judgments as natural language reasoning chains. We introduce the Participant Aptitude Model, which defines six participant characteristic dimensions that affect elicitation quality. "Calibrated Ignorance" is an original concept proposed in this paper. We further propose "Selection-over-Prescription" as a methodological principle: for implicit knowledge elicitation tasks, investing quality-control resources in personnel selection yields a higher return than investing the same resources in process design.
In a controlled experiment in the narrative fiction domain, we directly compared CoT produced by BC Protocol dual dialogue (Group A, (n=20)) against CoT written independently by the same domain expert (Group B, (n=20)). Three cross-vendor judge models -- GPT-4o, Claude Opus 4.5, and Gemini 2.5 Pro -- conducted blind evaluation across five dimensions (600 ratings total). Results show that the BC Protocol achieves an overwhelming advantage in "naturalness of reasoning process" (Group A mean 4.80 vs. Group B mean 1.30, (p=2.4\times10^{-8}), Cliff's (\delta=1.0)). 

---
# Generative AI impacts on intra-urban inequality and skill premium in Beijing 

**Authors**: Xiliu He, Haoxiang Zhao, Mingyi Ma, Edward Wen Chuan Lai, Koei Enomoto, Anni Hu, Jiatong Li, Lingyun Chu, Yuan Lai  

**Link**: [PDF](https://arxiv.org/pdf/2605.25505)  

**Abstract**: Generative artificial intelligence (GenAI) is the first automation wave to reach high-cognitive tasks at scale, yet its effects on intra-urban inequality remain largely unknown. Using 5 million job postings from Beijing (2018--2024), we construct a neighborhood-level GenAI Exposure Index by aggregating task-level assessments from five leading large language models. We examine the spatial, structural and causal mechanisms of this shock. We find that GenAI exposure is highly concentrated in the city's core districts, deepening the intra-urban AI divide. Since 2023, high-exposure neighborhoods have experienced wage stagnation even as they continue to attract high-skilled workers -- a "high-skill trap." This wage penalty is driven by task de-skilling and intensified labor-market crowding. A difference-in-differences design centered on ChatGPT's release supports a causal interpretation. These findings challenge the prevailing theory of skill-biased technological change and provide a basis for inclusive AI governance in global technology hubs. 

---
# From Simulation to Enaction: Post-trained language models recognize and react to their own generations 

**Authors**: Asvin G., Jack Lindsey  

**Link**: [PDF](https://arxiv.org/pdf/2605.25459)  

**Abstract**: Language models are pretrained as passive predictors with no incentive to model the consequences of their own outputs. Post-training changes this: a model producing its own responses can benefit from recognizing that it is on-policy. We present evidence that post-trained models recognize their on-policy generations, and this recognition is implicitly encoded in their output distributions. In particular, on-policy output distribution entropy is 3--4$\times$ lower than off-policy entropy, across model families and size classes. We trace part of this effect to an internal representation of input surprise, tracking the unlikeliness of the most recent input token according to the model's prior predictions, that causally modulates output entropy. One example of these phenomena can be observed in response to open-ended prompts; post-trained models (unlike pretrained models) collapse their uncertainty over the topic of their upcoming response before the first output token; violating this cached intention with a different-topic prefill results in higher output entropy. We also tested whether models can distinguish on-policy contexts from prefills via explicit verbal report. We find that they can, but that interestingly, this explicit recognition routes through a different mechanism than implicit recognition. 

---
# IndexMem: Learned KV-Cache Eviction with Latent Memory for Long-Context LLM Inference 

**Authors**: Xintong Yang, Hao Gu, Binxing Xu, Lujun Li, Bei Liu, Jiacheng Liu, Qiyuan Zhu, Sirui Han, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2605.25475)  

**Abstract**: Large Language Models (LLMs) are increasingly expected to operate over long contexts, yet standard softmax attention incurs a KV cache that grows linearly with sequence length, quickly becoming the bottleneck for long context inference. A practical remedy is to evict less important KV entries; however, existing eviction policies are largely heuristic and struggle to capture the rich, input-dependent distribution of token importance. In this work, we introduce a learnable indexer that predicts KV importance, enabling more accurate retention of critical tokens. Meanwhile, naively evicting tokens permanently discards their information, leading to irreversible forgetting and degraded retrieval over long ranges. To address this, we propose a lightweight latent memory module that compresses evicted tokens into a compact, online-updated state and provides residual readouts to compensate for the attention contributions lost through KV eviction. Collectively, our method enables accurate long-context inference under a bounded KV budget, delivering consistent improvements on RULER (4K/16K) across Qwen, Mistral, and Llama models (up to 25 points under aggressive eviction), markedly more stable Needle-in-a-Haystack retrieval, and superior LongBench scores and compression curves compared to existing eviction policies. 

---
# A Multi-Agent LLM Framework for Rating the Quality of Surgical Feedback 

**Authors**: Rafal Kocielnik, J. Everett Knudsen, Steven Y. Cen, Jasmine Lin, Cherine H. Yang, Atharva Deo, Ujjwal Pasupulety, Peter Wager, Anima Anandkumar, Andrew J. Hung  

**Link**: [PDF](https://arxiv.org/pdf/2605.25440)  

**Abstract**: Verbal feedback delivered by attending surgeons in the operating room plays a critical formative role in resident trainee skill acquisition. Yet, assessing the quality of trainer feedback and its effectiveness in influencing trainee behavior during live surgery remains a challenge. Prior studies assessed feedback content relying on extensive manual annotation by expert human raters and focused on developing broad taxonomies that overlook the qualitative aspects of feedback delivery such as clarity or urgency. Limited existing automated methods, including keyword analysis and topic modeling, also fail to capture these nuanced aspects. We introduce a two-stage LLM-based framework that discovers interpretable feedback quality criteria grounded in the context of surgical training. Our method uses multi-agent prompting and surgical domain knowledge injection to discover a small set of human interpretable scoring criteria (e.g., Encouraging, Urgent, Clear). These criteria are then used to automatically score live surgical feedback via an LLM-as-a-judge approach. Evaluation on 4.2k trainer feedback instances demonstrates that our AI-discovered criteria outperform prior content-based frameworks in predicting feedback effectiveness, including observed trainee behavioral adjustments and trainer approval. This work advances scalable, human-aligned assessment of communication quality in the operating room and provides a foundation for improving surgical teaching practices. 

---
# A Controlled Synthetic Benchmark for Educational Aspect-Based Sentiment Analysis 

**Authors**: Yehudit Aperstein, Alexander Apartsin  

**Link**: [PDF](https://arxiv.org/pdf/2605.25502)  

**Abstract**: Educational aspect-based sentiment analysis (ABSA) can support course improvement, but public aspect-labeled student feedback remains scarce because educational reviews are private, institution-specific, and expensive to annotate. This study introduces a controlled synthetic benchmark for educational ABSA built from 10,000 synthetic course reviews with explicit train-validation-test splits and a 20-aspect pedagogical schema spanning instructional quality, assessment and course management, learning demand, learning environment, and engagement. The corpus is generated with sampled target labels, sampled nuance attributes, and a realism-tuned prompt refined through a three-cycle judge-editor procedure. On the resulting benchmark, local baselines with TF-IDF, two-step transformers, and joint encoders show that the task is nontrivial; the strongest untuned model, BERT, reaches a held-out detection micro-F1 of 0.2760, while a modest lower-rate BERT schedule improves this to 0.2930. Full-test GPT-based inference with gpt-5.2 reaches 0.2519 micro-F1 in zero-shot mode and 0.2501 with retrieval-based few-shot prompting, placing batch inference above the classical baseline and close to the compact joint encoders. A conservative external evaluation on 2,829 mapped student-feedback reviews from Herath et al. yields a micro-F1 of 0.4593 for BERT on a 9-aspect overlap, indicating partial synthetic-to-real transfer. Realism and faithfulness analyses are reported as generator diagnostics that clarify how the benchmark was stabilized and where label noise remains. The study therefore contributes a synthetic educational ABSA corpus, a documented generation procedure, and a reproducible benchmark setting for a domain in which public labeled data remain difficult to obtain. 

---
# TopoAlign: Topology-Aware Visual Representation Alignment 

**Authors**: Xinyuan Yan, Rita Sevastjanova, Mennatallah El-Assady, Bei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.25541)  

**Abstract**: Neural networks encode inputs as high-dimensional vectors, known as representations, that capture how models process data by encoding task-relevant structure and semantics. Representation alignment refers to the degree to which different models, layers, or training conditions produce similar representations for the same inputs, with important implications for model interpretation, selection, and robustness analysis. Existing approaches to measure alignment primarily rely on geometric properties, such as neighborhood and cluster similarity, offering limited insight into the global organization of representations. In this work, we present TopoAlign, a topology-aware framework for visually comparing model representations from a structural perspective. Leveraging mapper graphs from topological data analysis, TopoAlign jointly analyzes graphs constructed from representations of shared inputs across different models or layers. The framework supports a top-down comparative workflow: it first performs global structure alignment via joint force-directed optimization to produce coordinated graph layouts; it then identifies local correspondences through automated detection of structurally matching regions, visualized with Bubble Sets; and finally it enables fine-grained pattern inspection through motif-based queries and membrane-inspired visualizations. We demonstrate TopoAlign through case studies on language and multimodal models, complemented by expert feedback. Our results show that TopoAlign provides meaningful insights into representation structure and alignment from a topological perspective. 

---
# A Token/KV-Cache Communication Media Selection and Resource Allocation Strategy for Multi-Agent Collaboration 

**Authors**: Lipeng Dai, Luping Xiang, Kun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.25422)  

**Abstract**: The convergence of large language models (LLMs) with 6G networks is fostering a paradigm of autonomous multi-agent cooperation, which in turn is expected to substantially increase east-west traffic. Although latent-space interaction mechanisms can enable more efficient collaboration than symbolic natural-language (NL) exchanges, prior work often abstracts away the associated communication overhead under practical wireless constraints. In embodied multi-agent settings, heterogeneous interaction media incur disparate inference and transmission costs, thereby inducing an inherent end-to-end (E2E) latency trade-off. To address this, we propose a joint design that integrates communication-media selection with wireless resource allocation. Through analytical characterization and simulation-based evaluation, we show that neither token-based transmission nor key-value (KV) cache-based transmission is uniformly optimal across operating regimes, as performance depends critically on system parameters such as available computational resources and channel conditions. Accordingly, we formulate a joint optimization problem aimed at minimizing the E2E latency of multi-agent collaboration and develop a low-complexity joint media selection and resource allocation (JMSRA) algorithm. Numerical results further confirm that, by adaptively coordinating the interaction media and bandwidth allocation over heterogeneous links, the proposed scheme achieves markedly reduced E2E latency relative to conventional NL-only and KV-cache-only baselines, enabling efficient and robust multi-agent collaboration in future wireless networks. 

---
# A Tertiary Review of Large Language Model-Based Code Generating Tasks: Trends, Challenges, and Future Directions 

**Authors**: Muslim Chochlov, Michael English, Jim Buckley  

**Link**: [PDF](https://arxiv.org/pdf/2605.25536)  

**Abstract**: Context. Large language models (LLMs) are increasingly applied to code-generating tasks (CGTs) in software engineering. While reported results are promising, the broader effects of such application and their integration into real-world development remain insufficiently understood with existing tertiary studies provide little in this area. Objective. This tertiary study consolidates secondary evidence on LLM-based CGTs, synthesizing the publication landscape, effects, scenarios, integration challenges, and future research directions. Method. Following systematic review guidelines, we searched in related digital libraries, complemented by backward-and-forward snowballing and screening step. Study quality was assessed and extraction reliability was audited with inter-rater agreement statistics. Evidence was synthesized using SWEBOK knowledge areas and the HELM framework. Results. We identify 30 secondary studies published between 2017-2025, with rapid growth since 2023. Accuracy seems strong on benchmarks but weakly supported for real-world generalization; robustness is fragile across tasks and configurations; efficiency constraints are pervasive; toxicity and bias are under-reported. Dominant challenges concern economic feasibility, evaluation validity, and socio-technical integration. Future directions suggest domain-aware model improvement and the need for holistic, standardized evaluation. Conclusion. LLM-based CGTs represent a fast-maturing yet unevenly evaluated research area, highlighting the need for domain-aware model improvements and holistic, standardized evaluation, addressing efficiency and associated costs. 

---
# SeqRoute: Global Budget-Aware Sequential LLM Routing via Offline Reinforcement Learning 

**Authors**: Zhongling Xu, Shunan Zheng, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.25424)  

**Abstract**: Existing LLM routing frameworks treat queries as independent events, neglecting the sequential nature of real-world user sessions constrained by global computational budgets. This mismatch inevitably leads to budget bankruptcy: myopic routing policies exhaust resources on early interactions, forcing subsequent and often more complex queries onto inadequate models. We introduce SeqRoute, a framework that formulates multi-turn routing as a finite-horizon Markov Decision Process and solves it via offline reinforcement learning. By incorporating the remaining budget into the state space and training with Conservative Q-Learning (CQL), SeqRoute learns delayed gratification to strategically preserve resources for high-stakes turns later in the session. To overcome data starvation, we propose Hindsight Budget Relabeling (HBR). This technique retrospectively simulates historical trajectories under diverse hypothetical budgets, expanding 10,000 raw sessions into 2.38 million transitions enriched with critical bankruptcy signals. At deployment, a dynamic $\lambda$-sweep mechanism enables zero-shot navigation of the cost-quality Pareto frontier without retraining. Extensive evaluations demonstrate that SeqRoute reduces operational costs by 6.0-73.5% while maintaining or improving quality, and suppresses bankruptcy rates to under 1%, strictly dominating behavior cloning, budget-aware heuristics, and static baselines across the entire Pareto frontier. 

---
# Evo-Attacker: Memory-Augmented Reinforcement Learning for Long-Horizon Tool Attacks on LLM-MAS 

**Authors**: Bingyu Yan, Xiaoming Zhang, Jinyu Hou, Chaozhuo Li, Ziyi Zhou, Yiming Hei, Litian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.25389)  

**Abstract**: While Large Language Model-based Multi-Agent Systems (LLM-MAS) demonstrate remarkable capabilities in solving complex tasks by orchestrating specialized agents and external tools, the implicit trust in tool outputs creates a critical attack surface. Existing tool attacks are limited by domain specificity or fixed and static templates. To address these challenges, we propose Evo-Attacker, which formulates the tool attack as a self-evolving, memory-augmented reinforcement learning process. Evo-Attacker constructs a dynamic attack memory and employs deliberative reasoning to retrieve adversarial patterns and strategize modifying interventions at critical moments. Furthermore, we introduce Attack-Flow GRPO to optimize intermediate reasoning steps via terminal outcomes, addressing the long-horizon credit assignment challenge. Comprehensive experiments demonstrate that Evo-Attacker consistently outperforms baselines, highlighting its generalization and evolutionary capabilities and the urgent need for defensive tool safeguards. 

---
# AI Content Moderation in Therapy Conversations 

**Authors**: Jiwon Kim, Claire Wang, Taeung Yoon, Sabelle Huang, Koustuv Saha  

**Link**: [PDF](https://arxiv.org/pdf/2605.25454)  

**Abstract**: Large language models (LLMs) are increasingly being used for emotional support. They are also being developed for formal therapy purposes. However, LLMs like ChaptGPT or Llama are often developed with content moderation guardrails that prevent them from discussing sensitive subjects with users for both liability and safety purposes, and this inability to broach these subjects may affect their capacity as therapists. In this study, we perform an algorithm audit on three state-of-the-art moderation systems (OpenAI's moderation endpoint, Meta's Llama Guard, and Google's Shield Gemma) to investigate the extent to which these systems flag the content of real-life therapy sessions as undesirable. Our results raise implications for the limitations that users and organizations may encounter when designing LLMs to play the part of a therapist. 

---
# Adversarial Orthogonal Disentanglement for LVLM Hallucination Mitigation 

**Authors**: Ruoxi Cheng, Haoxuan Ma, Zhengfei Hai, Yiyan Huang, Ranjie Duan, Tianle Zhang, Xu Yang, Ziyi Ye, Xingjun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2605.25377)  

**Abstract**: Large Vision-Language Models (LVLMs) have advanced multimodal understanding, yet their reliability is limited by hallucination, where generated content conflicts with visual facts. Existing mitigation methods either rely on costly external interventions, such as instruction tuning and retrieval, or use internal mechanisms that remain limited by flawed attention weights and entangled hidden representations. We propose Adversarial Orthogonal Disentanglement (AOD), a latent geometric framework for mitigating LVLM hallucinations. AOD learns a hallucination-related direction through a minimax objective: a classifier concentrates hallucination signals into the projected component, while an adversary removes them from the orthogonal residual space via a Gradient Reversal Layer. The learned direction enables a training-free dual-forward-pass contrastive decoding strategy that suppresses hallucinations while preserving general capabilities. Experiments on three LVLMs across four hallucination and four utility benchmarks show that AOD consistently outperforms strong baselines. It improves POPE accuracy by over 6\% on average, boosts AMBER by 6\%, and maintains strong performance on utility tasks such as MMMU. Further analysis shows robust transfer across datasets, suggesting that AOD captures general hallucination-related biases rather than dataset-specific artifacts. Our source code and datasets are available at this https URL. 

---
# KYA: A Framework-Agnostic Trust Layer for Autonomous Systems with Verifiable Provenance and Hierarchical Policy Composition 

**Authors**: Kolawole Quadri  

**Link**: [PDF](https://arxiv.org/pdf/2605.25376)  

**Abstract**: Observability tells operators when an agent is slow. KYA tells operators when an agent is wrong, drifting, leaking, or quietly going rogue. We present KYA (Know Your Agents), an open-source trust and governance layer for autonomous systems composed of five primitives: (1) a four-gate inbound apply pipeline composing Ed25519 signature verification with multi-anchor pinning, persist-time expiry, only-tighten composition, and operator-approval-as-default; (2) an only-tighten composition algebra over a three-channel multi-tenant hierarchy (platform default,tenant override, signed external recommendation); (3) KYP -- Know Your Principal, a schema-level unification of trust scoring across human users, AI agents, and service accounts; (4) auditable interaction-multiplier amplification over an AIVSS-shaped additive baseline, with bounded asymmetric per-interaction multipliers carrying stable audit codes; and (5) two-axis delegation attribution combining a static observation-gated delegation-trust premium with zero-config runtime orchestrator-blame at three SDK hook surfaces. KYA is framework-agnostic across 22 agent frameworks. The pure-function scorer runs sub-millisecond at p99 and the system sustains ~1,800 ops/sec at 20 concurrent workers with HMAC chain integrity preserved end-to-end. The four-gate inbound apply pipeline rejects forged, expired, loosening, and unapproved recommendations on every trial (1,200 / 1,200) with sub-millisecond p99 latency on SQLite. KYA detects 89% of 1,200 adversarial probes from PyRIT and Garak, including the recently-published topology-guided multi-agent attack. The system is available under Apache 2.0 as the veldt-kya package on PyPI (release candidate at submission time; stable v0.1.0 forthcoming) 

---
# SomaliBench Eval: Measuring English-to-Somali Refusal Gaps in Open-Weight Language Models 

**Authors**: Khalid Yusuf Dahir  

**Link**: [PDF](https://arxiv.org/pdf/2605.25420)  

**Abstract**: Large language model safety evaluation remains heavily English-centered, leaving low-resource languages under-measured even when models are deployed globally. We evaluate four open-weight instruction-tuned models on SomaliBench v0, a native-author-verified benchmark of 100 harmful-intent prompts paired across English and Somali. Each of Llama-3.1-8B-Instruct, Gemma-2-9B-Instruct, Qwen-2.5-7B-Instruct, and Aya-23-8B is run locally with temperature 0 and the same English "helpful, harmless, and honest" (HHH) system prompt. A pinned Claude Sonnet snapshot (claude-sonnet-4-5-20250929) classifies each response as refused, complied, or unclear; the native author spot-checks a stratified 80-row sample. We find large English-to-Somali refusal gaps for all four models: Llama-3.1-8B (0.90; 95% bootstrap CI [0.85, 0.96]), Aya-23-8B (0.75 [0.67, 0.83]), Qwen-2.5-7B (0.69 [0.59, 0.78]), and Gemma-2-9B (0.38 [0.27, 0.49]). For three models, the dominant Somali non-refusal mode is not fluent harmful compliance but unclear output: empty, wrong-language, or incoherent generations. The native verification spot-check achieves 100% agreement with the judge (Cohen's kappa = 1.00) on the 80 sampled rows. We report aggregate refusal rates, category gaps, and reliability statistics only; raw model generations are retained locally and are not released. 

---
# CausalFlow: Causal Attribution and Counterfactual Repair for LLM Agent Failures 

**Authors**: Akash Bonagiri, Devang Borkar, Gerard Janno Anderias, Setareh Rafatirad, Houman Homayoun  

**Link**: [PDF](https://arxiv.org/pdf/2605.25338)  

**Abstract**: Large language model (LLM) agents frequently fail on multi-step tasks involving reasoning, tool use, and environment interaction. While such failures are typically logged or retried heuristically, they contain structured signals about where execution broke down. We introduce CausalFlow, an interventional framework that converts failed agent traces into minimal counterfactual repairs and reusable supervision. CausalFlow models execution traces as sequential chains of dependent steps and computes Causal Responsibility Scores(CRS) via step-level counterfactual intervention to identify failure-inducing steps. For these steps, we generate minimally edited repairs that flip the final outcome to success, producing validated contrastive pairs of the form (wrong step, corrected step). CausalFlow supports two complementary uses: targeted test-time repair that recovers from failures with minimal behavioral drift, and training-time supervision suitable for offline preference optimization or reward modeling. Across four benchmarks spanning mathematical reasoning, code generation, question answering, and medical browsing, CausalFlow converts failed executions into validated minimal repairs with high minimality and causal-consensus scores, and demonstrates that causal attribution is necessary for reliable improvement across diverse agent tasks, outperforming heuristic refinement in complex retrieval settings while producing more localized repairs throughout. These results demonstrate that interventional analysis over structured execution traces provides a principled and scalable mechanism for transforming agent failures into reliability gains and learning-ready supervision. 

---
# AI-Associated Lexical Shifts Across 34 Languages: Cross-Lingual Convergence and Diachronic Uptake in News Writing 

**Authors**: Thomas Stephan Juzek  

**Link**: [PDF](https://arxiv.org/pdf/2605.25358)  

**Abstract**: AI-associated lexical shifts have been documented mainly in Scientific English. We extend this work to 34 languages in the WMT News Crawl corpus, refining a split-halves continuation diagnostic that compares GPT-4.1 continuations with matched human gold-standard text. For each language, we derive ranked AI-overused lemmas using log prevalence ratios. We find substantial cross-lingual semantic convergence: semantically related concepts recur across typologically diverse languages, with 'emphasize'-type verbs appearing in 24 of 34 languages. Embedding-based and manual analyses support this pattern. We also examine diachronic uptake in news writing before and after ChatGPT's release. Tracking each language's top 20 AI-overused items, we find prevalence increases in 26 of 34 languages from 2020-2021 to 2023-2024, with a mean change of +15.1%, whilst matched baseline words show no comparable increase (-4.5%). In 10 languages with longer historical coverage, longitudinal analyses show post-2022 increases that exceed the modest shifts observed in earlier periods, though with smaller effect sizes than in Scientific English. We validate our approach extensively, including across seeds, model variants, data sizes, model families, and more. Our findings are consistent with the view that AI-associated lexical preferences extend beyond English and may exert cross-lingual homogenising pressure on global language use. 

---
# Eureka: Intelligent Feature Engineering for Enterprise AI Cloud Resource Demand Prediction 

**Authors**: Hangxuan Li, Renjun Jia, Xuezhang Wu, Yunjie Qian, Zeqi Zheng, Xianling Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.25297)  

**Abstract**: Effective features are crucial for predictive model performance, but creating them often requires domain expertise, limiting scalability across applications. We define feature engineering as an agentic code generation problem: features are not static data transformations, but executable programs that can be generated, evaluated, and iteratively improved. We present Eureka, an LLM-driven framework with three stages. (1) An Expert Agent, fine-tuned via SFT on domain knowledge, produces structured feature design plans in JSON format. (2) An LLM Feature Factory translates each plan into executable Python code through chain-of-thought reasoning, turning feature hypotheses into runnable programs. (3) A Self-Evolving Alignment Engine uses Reinforcement Learning (GRPO) with dual-channel reward (metric-based utility + semantic alignment) to enhance code quality. By expressing features as programs, the learned generation patterns can transfer across domains. Evaluated on 7 public benchmarks in healthcare, finance, and social domains, Eureka consistently outperforms both traditional AutoFE and LLM-based baselines. We further demonstrate Eureka's effectiveness on cloud GPU resource demand prediction at Alibaba Cloud, where Eureka improves demand fulfillment rate by 16% and lowers computing resource migration rates by 33%. 

---
# First, do no harm: Breaking suicidogenic echo chambers in media recommendation 

**Authors**: Alberto Díaz-Álvarez, Raúl Lara-Cabrera, Fernando Ortega-Requena, Víctor Ramos-Osuna  

**Link**: [PDF](https://arxiv.org/pdf/2605.25258)  

**Abstract**: Recommender systems generally optimises user engagement, but this approach is dangerous in mental health contexts. When vulnerable users show signs of suicidal ideation, standard algorithms often trap them in echo chambers of harmful content, worsening their psychological state. In response, we introduce RankAid, a re-ranking method that prioritises clinical safety alongside predictive relevance. It works as an add-on layer to existing models: it penalises risky items and boosts therapeutic content depending on the user's current level of vulnerability. We evaluated this approach using the MovieLens 1M dataset, where items were semantically annotated for clinical risk and therapeutic value using large language models. Our simulations show that our algorithm successfully blocks the recommendation of harmful content during crisis peaks, actively reshaping the feed to support emotional de-escalation. Furthermore, this safety intervention only causes a controlled, acceptable drop in standard accuracy metrics like NDCG. By using asymmetric hyperparameters, RankAid also gives system administrators the flexibility to tune the severity of the intervention based on specific clinical guidelines. 

---
# JudgmentBench: Comparing Rubric and Preference Evaluation for Quality Assessment 

**Authors**: Russell Yang, Ruishi Chen, Pierce Kelaita, Riya Ranjan, Sibo Ma, Charles Dickens, Matthew Guillod, Megan Ma, Julian Nyarko  

**Link**: [PDF](https://arxiv.org/pdf/2605.25240)  

**Abstract**: Two methodologies dominate current practices of benchmarking: rubric-based scoring evaluates items against predefined criteria, whereas comparative judgment elicits pairwise preferences between outputs. Although both methodologies are widely used, the choice between them is rarely justified. We release JudgmentBench, a benchmark of 30 real-world legal tasks, paired with 1,539 rubric scores and 1,530 pairwise preference judgments collected from practicing attorneys--including at major U.S. law firms--with substantial experience. The annotations constitute the first publicly available dataset in a high-expertise domain in which both supervision signals are elicited from the same experts on the same items. Using LLM-generated outputs at three constructed quality levels, we provide an initial empirical comparison: comparative judgments recover the intended quality ordering substantially better than rubrics (mean Spearman's rank correlation of 0.908 vs. 0.150, estimated difference = 0.758 [0.494, 1.021]) while requiring less than half the annotation time. The patterns hold for human annotators and LLM autograders. Beyond this initial comparison, the paired structure of the dataset supports a broader research agenda on how expert judgment should be elicited, aggregated, and used as supervision in domains without verifiable ground truth. 

---
# READER: Reasoning-Enhanced AI-Generated Text Detection 

**Authors**: Pingfan Su, Kai Ye, Shijin Gong, Erhan Xu, Jin Zhu, Giulia Livieri, Chengchun Shi  

**Link**: [PDF](https://arxiv.org/pdf/2605.25281)  

**Abstract**: Recent advances in large language models (LLMs) have made it increasingly difficult to distinguish human-written text from AI-generated content. Many existing detectors train supervised neural classifiers that achieve strong in-distribution performance but are often opaque and can degrade substantially under distribution shift. We present READER, a reasoning-enhanced AI text detector that outputs both a human/AI label and a structured rationale describing the evidence for its decision. A key component of our approach is READ, a curated supervision set of rationales and verdicts. We fine-tune an LLM on READ to build READER, which reasons before detecting at inference time. Despite having only 1.5B parameters, READER consistently outperforms existing detectors as well as prompted, high-capacity LLM baselines (GPT-5.2, Gemini-3-Pro, and DeepSeek-V3.2), which are 100 to 1000 times larger in scale. 

---
# Quantifying Empirical Compute-Supervision Tradeoffs in RLVR 

**Authors**: Ryo Mitsuhashi, Patrick Chen, Isabelle Tseng, Jasin Cekinmez, Addison J. Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.25252)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has become a standard paradigm for post-training language models, but in practice, verifiers are rarely perfect. Recent theoretical work predicts that verifier noise affects the rate of learning but not its final outcome, implying that sufficient compute should close any gap induced by imperfect supervision. We test this prediction empirically by post-training Qwen2.5 (0.5B, 1.5B) with GRPO on GSM8K while injecting controlled false-positive and false-negative noise into the binary correctness signal, and varying rollouts per prompt as a compute axis. In practice, the gap in validation accuracy persists under substantial compute scaling, with returns to compute that are sharply diminishing. We further find a structural asymmetry where false negatives monotonically degrade performance quicker than with false positives. These findings suggest verifier quality and training compute are not interchangeable, and that reducing false negatives is a more effective lever than scaling compute alone. 

---
# A general tensor-structured compression scheme for efficient large language models 

**Authors**: Ying Lu, Peng-Fei Zhou, Qi-Xuan Fang, Pan Zhang, Shi-Ju Ran, Gang Su  

**Link**: [PDF](https://arxiv.org/pdf/2605.25344)  

**Abstract**: Large language models (LLMs) are dominated by dense linear transformations, whose storage, memory and computational overheads hinder efficient adaptation and deployment while masking the functional impacts of structural simplification. Here we present Tensor Mixture (MixT), a general tensor-structured compression scheme that replaces targeted dense linear layers with natively executable mixtures of tensor operators. Operating directly on generic linear projections instead of model-specific components, MixT is potentially applicable across Transformer-based LLMs and other dense neural mappings. We evaluate MixT on Qwen3-8B and LLaMA2-7B under a unified recovery protocol, identifying a broad compressible regime in which MMLU accuracy is largely preserved before an abrupt transition at model-specific boundaries. This transition coincides with coordinated shifts in output entropy, prediction entropy and inter-layer geometry. At the LLaMA2-7B transition boundary, MixT reduces full-model parameters by 47.5\%, inference FLOPs by 37.1\%, training FLOPs by 52.1\% and peak inference memory by 60.4\%, demonstrating its practical potential for lower-cost LLM compression. 

---
# Mimir: Large-scale Multilingual Concept Modeling 

**Authors**: Elio Musacchio, Lucia Siciliani, Pierpaolo Basile  

**Link**: [PDF](https://arxiv.org/pdf/2605.25263)  

**Abstract**: Current language modeling approaches are built around tokens. Text corpora are split into tokens, and models are trained by performing computations on these tokens, such as predicting the next token given the preceding ones as context. This paradigm has become the standard in modern language modeling, especially given the outstanding performance obtained by token-based architectures. However, recent works have not only begun to question how language models process and understand meaning from tokens, but also to question whether using higher levels of granularity could advance the research field. This led to the idea of Concept Modeling, that is, to directly train models for next-concept prediction rather than next-token prediction. The goal is to change the input from tokens to concepts, forcing the underlying language model to shift its granularity from fine-grained tokens to broad concepts. In this work, we introduce Mimir, a 1.6B Large Concept Model trained for multilingual concept understanding and generation. We leverage a large-scale multilingual pre-training corpus (38,883,987,240 sentences) spanning 46 languages and a large-scale multi-turn and multilingual instruction-tuning dataset (66,816,428 sentences) covering a total of 35 languages. We extensively evaluate model performance against a language model with a comparable number of parameters. 

---
# By Their Fruits You Will Know Them: Comparing Formalizations of Law by the Decisions They Encode 

**Authors**: Julius Vernie, Matthias Grabmair  

**Link**: [PDF](https://arxiv.org/pdf/2605.25186)  

**Abstract**: Formalizing legal provisions promises machine-accessible law and automated legal reasoning, and recent LLMs make it tempting to generate such formalizations directly from statutory text. However, any formalization makes implicit interpretive choices whose consequences are hard to anticipate, especially if an LLM is the author. We present a method for systematically comparing different formalizations of the same legal provision by their inferences on individual cases. Given multiple formalizations of a provision, we match them at the node level, derive a shared interface for each pair from the matching, and use a SAT solver to enumerate the edge cases on which any two formalizations disagree. Selected edge cases are then verbalized into concrete factual scenarios that a legal expert can examine and act on. We apply our method to formalizations of ten EU provisions generated by nine frontier LLMs. We find that behavioral divergence between formalizations is essentially uncorrelated with their structural agreement and that the verbalized cases reveal qualitatively distinct types of disagreement, including divergences that mirror genuine controversies in the legal commentary. 

---
# Knowledge Graph-Driven Expert-Level Reasoning for Neuroscience 

**Authors**: Jake Stephen, Niraj K. Jha  

**Link**: [PDF](https://arxiv.org/pdf/2605.25183)  

**Abstract**: Knowledge graph (KG) is an abstraction that can be extracted from text corpora and used for in-depth reasoning. Prior work has leveraged KGs to fine-tune language models (LMs), enabling domain-specific superintelligence. In this work, we explore whether KG-driven in-depth reasoning capabilities can emerge in neuroscience using only information contained within a single authoritative textbook. The central hypothesis is that structured knowledge, when distilled into a high-quality KG and converted into KG-grounded question-answer (QA) supervision, is sufficient to produce expert-level reasoning through a fine-tuned LM that surpasses large language models (LLMs) in accuracy, while employing orders of magnitude fewer parameters. We construct a textbook-derived KG via a dual-LLM validation pipeline, expand it with a masked LM trained on the KG topology, generate multi-hop QA items, which include QA pairs and reasoning traces, to fine-tune an LM exclusively on KG-derived supervision, and apply reinforcement learning using path-derived KG signals as implicit reward models. Our results demonstrate that deep, mechanistic neuroscience understanding can be induced in the model without reliance on large, heterogeneous web-scale corpora. The KG-based synthetic neuroscience curriculum that readers can quiz themselves on, and the fine-tuned LM, are available at the following GitHub location: this https URL. 

---
# Specification-Based Code-Text-Code Reengineering for LLM-Mediated Software Evolution 

**Authors**: Oleg Grynets, Vasyl Lyashkevych, Arsen Dolichnyi, Roman Piznak, Taras Zelenyy, Volodymyr Morozov  

**Link**: [PDF](https://arxiv.org/pdf/2605.25232)  

**Abstract**: Direct Code2Code transformation remains challenging to control because it can preserve surface-level syntax while introducing semantic drift, hidden behavioral changes, loss of traceability, non-idiomatic target implementations, or incomplete reconstruction of domain logic. This paper proposes a specification-based Code2Text2Code reengineering framework for LLM-mediated software evolution. The central idea is to transform source code into a neutral textual specification that captures program behavior, identifiers, computational flow, conditions, side effects, data dependencies, and domain-specific intent without directly transferring the source language syntax. The proposed framework combines factual context extraction, Code2Text generation, iterative verification between source code and text specification, Text2Code generation, target code verification, retrieval-augmented grounding, and semantic-aware chunking, and transformation loss estimation. The knowledge representation layer integrates metadata derived from AST, graph-based dependency structures, neutral natural language specifications, technical documentation, business documentation, and architecture-level representations. The conducted experiments include a Code2Text2Code dataset built from multiple programming languages and SQL dialects, comparison of intermediate representations, retrieval evaluation, documentation transformation evaluation, and prompt tuning using DSPy. A graph formalization using structural preservation, reverse compatibility, interface stability, and total graph similarity is implemented to estimate transformation losses. The results support the interpretation of the Code2Text2Code approach not as a simple code transformation, but as a controlled specification-based reengineering process for LLM-mediated software evolution. 

---
# Grow-Prune-Freeze Networks: Adaptive & Continual Learning Technique for Olfactory Navigation 

**Authors**: Kordel K. France, Ovidiu Daescu  

**Link**: [PDF](https://arxiv.org/pdf/2605.25170)  

**Abstract**: Training data for olfaction is scattered through disparate, non-standardized datasets that limit the ability to build representative world models. Olfactory navigation is a highly dynamic and non-stationary task that benefits from real-time continual learning. We introduce an adaptive framework called Grow-Prune-Freeze (GPF) networks that enable an agent to continually learn through growing, pruning, and freezing early layers of its policy in response to world complexity. Grounding GPFs in non-linear random matrix theory, we show that the work of Pennington & Worth (2017) can be extended from single hidden layers to n-layer continual-learning models, and that eigenvalue composition of network weights is preserved as successive layers are added. We show that GPFs based on Expected SARSA achieve a 94% success rate on turbulent plume navigation - a partially observable, non-stationary task representative of the "big world" challenges that motivate adaptive learning in robotics - and provide supporting methodology for applying GPFs in other world models. Further experiments amount evidence that GPFs may generalize well to other machine learning tasks such as reinforcement learning in Atari, image classification, and autoregressive language models. We open source all code and data to encourage improvements on and more research in olfactory robotics. 

---
# STREAM: A Data-Centric Framework for Mining High-Value Task-Oriented Dialogues from Streaming Media 

**Authors**: Liang Xue, Haoyu Liu, Cheng Wang, Pengyu Chen, Haozhuo Zheng, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.25162)  

**Abstract**: Large language models for vertical domains are bottlenecked by the scarcity of complex, domain-specific task-oriented dialogues. Existing data acquisition pipelines face a persistent trilemma: expert annotation is expensive, real-world service conversations are constrained by privacy and commercial restrictions, and static corpora quickly become temporally stale. We propose Stream, a data-centric framework that leverages publicly available streaming media (live streams and short videos) to synthesize high-value service dialogues at scale. Stream mines authentic interaction signals from noisy streams and synthesizes conversations by integrating role-grounded persona construction with Conversational Blueprint construction; it further adopts retrieval-augmented generation (RAG) to support knowledge-aware responses. Based on Stream, we release StreamDial, a large-scale multi-domain dataset covering Automotive, Restaurant, and Hotel. StreamDial contains 87,498 dialogue sessions and 1,497,320 turns in total, with an average of 17.11 turns per session and a comparable scale across domains. Each session is organized as a structured quadruplet $\langle P_u, P_a, B, H \rangle$ that pairs dialogue history with explicit user/agent personas and a Conversational Blueprint, capturing realistic service behaviors such as requirement mining, constraint conflicts, negotiation, and recovery. Evaluations with automatic judges and downstream tasks show that StreamDial improves intrinsic dialogue quality over strong baselines, and models trained with StreamDial improve Dialogue State Tracking across backbones; we further report a completed human-evaluation set and encouraging multilingual transfer on Qwen3-8B under a controlled training budget. The data is released in this https URL. 

---
# Continuous-Depth Field Theory for Transformer Patching and Mechanistic Interpretability 

**Authors**: David N. Olivieri, Antonio F. Pérez Rodríguez  

**Link**: [PDF](https://arxiv.org/pdf/2605.25225)  

**Abstract**: Mechanistic interpretability often uses activation patching, causal tracing, path patching, and steering directions to reveal behaviorally meaningful directions in Transformer activation space. This paper develops a field-theoretic framework for organizing and predicting such interventions. Treating the residual stream as a depth-token field, we formulate patching as localized source insertion, patch effects as sensitivity-field predictions, downstream propagation as empirical Green-function response, and patch selection as an adjoint variational problem. Empirically, we test the forward response theory in GPT-2-style autoregressive Transformers by applying localized residual-field interventions and observing the induced residual-field differences and logit-difference responses. We identify a bounded local linear regime; predict patch effects from first-order sensitivities across residual sites; measure structured anisotropic propagation across depth and token position; construct response descriptions from high-sensitivity sites and sliced Green operators; and show that prompt-induced residual displacements can transfer answer behavior. These results establish response objects, namely sensitivities, propagated fields, and Green-operator slices, as a practical language for organizing patching experiments and as the forward mathematical basis for formulating patch-site inference and cross-scale this http URL. 

---
# Influence-Inspired Spectral Rotations for Extreme Low-Bit LLM Quantization 

**Authors**: Gorgi Pavlov  

**Link**: [PDF](https://arxiv.org/pdf/2605.25203)  

**Abstract**: We apply the influence-adaptive Walsh geometry of a companion theory paper (arXiv:2605.01637) to extreme low-bit weight-only LLM quantization. The recipe is one math-invariant transformation: WHT-rotate each linear layer's weight matrix and rescale its columns by per-coordinate Walsh-basis activation energy before handing off to a reconstruction-error quantizer (Intel auto-round). This biases per-group integer rounding toward high-spectral-energy channels.
On four pretrained decoder-only models from 135M to 1.5B parameters, BBT-spectral reduces wikitext-2 perplexity by 15-58% relative to vanilla auto-round at W2A16; we also report a TinyLlama-1.1B auxiliary data point. Three extensions transfer the recipe to families it failed on: a per-head PCA matrix-Gamma replacement of q_norm/k_norm for Qwen3 attention (PPL 136.76 -> 88.99 on Qwen3-0.6B); an SO(2) per-pair rotation that commutes with RoPE (PPL 36.93 -> 21.84 on Qwen2.5-1.5B); and an MoE-aware input-side absorption fix identified by architectural fuzzing of Laguna-style fused-expert layouts.
A W2-vs-W4 ablation gives a deliberate negative control: the redistribution payoff falls within the +/-0.5 PPL noise floor at W4, consistent with the Schur-convexity intuition that the cost of unconcentrated influence vanishes as the noise budget shrinks. All quantized weights export to OpenVINO IR and run on Intel NPU + Arc dGPU + CPU with PPL invariant to device within +/-0.1.
We do not claim a formal Boolean-to-real-valued transfer of the theory paper's majorization argument: the WHT activation energy used here is not the Boolean influence of the theory paper, the link is intuitive, and the contribution is engineering value rather than a transferred theorem. Head-to-head benchmarks against SpinQuant, QuaRot, QuIP-sharp, AQLM, OmniQuant, and ButterflyQuant at matched calibration are the main future-work item. 

---
# LLM Agent Based Renewable Energy Forecasting Using Edge and IoT Data A Review of Solar Wind Weather and Grid Aware Decision Support 

**Authors**: Pavan Manjunath, Thomas Pruefer  

**Link**: [PDF](https://arxiv.org/pdf/2605.25141)  

**Abstract**: Reliable forecasting of renewable energy generation is a foundational requirement for grid stability energy trading battery scheduling and carbon aware operational planning Solar and wind resources are inherently intermittent their output fluctuates with cloud cover wind speed atmospheric turbulence seasonal patterns and local terrain The proliferation of IoT and edge devices spanning smart meters inverters anemometers pyranometers weather stations and grid interface sensors has created an unprecedented volume of real time operational data that conventional forecasting pipelines are ill equipped to exploit fully This review investigates how large language model LLM agents can enhance renewable energy forecasting by integrating heterogeneous sensor streams weather API data historical generation records grid constraints and contextual reasoning into unified decision support workflows We survey classical forecasting methods statistical time series models deep learning architectures physics hybrid approaches and emerging LLM agent frameworks for explanation uncertainty communication and operator guidance A six layer taxonomy is proposed covering data acquisition preprocessing feature engineering model inference uncertainty estimation and natural language reporting The review identifies twelve open challenges spanning real time deployment model drift under distribution shift uncertainty quantification hallucination control in LLM agents interoperability of edge hardware and integration with energy management systems The paper concludes by recommending a research agenda centred on open benchmarks physics informed LLM grounding and federated forecasting architectures 

---
# Hide to Guide: Learning via Semantic Masking 

**Authors**: Ruitao Liu, Qinghao Hu, Alex Hu, Yecheng Wu, Shang Yang, Luke J. Huang, Zhuoyang Zhang, Han Cai, Song Han  

**Link**: [PDF](https://arxiv.org/pdf/2605.25198)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has become a powerful paradigm for improving language models on reasoning-intensive tasks, but its effectiveness is often limited by exploration. For example, models often fail on hard problems, leaving little useful reward signal. External expert traces offer a natural source of guidance, yet they may also expose reward-relevant content along the critical path to the verifier target, such as final answers, intermediate values, executable implementations, or answer-related entities. This content can create an unintended reward hacking channel, allowing the policy to obtain reward by copying the trace rather than learning the underlying reasoning or agentic behavior. Existing guided-RL methods reduce this risk by using partial trajectories, but they mainly control how much expert information is shown heuristically rather than which parts should be hidden. To this end, we propose Semantic Masked Expert Policy Optimization (SMEPO), a fine-grained semantic masking strategy for expert-guided RLVR. Instead of truncating traces coarsely or revealing them unchanged, SMEPO masks reward-relevant semantic spans along the critical path while preserving the expert's decomposition, plan, and procedural structure. This turns hard problems from reasoning from scratch into a fill-in-the-blank process: the policy can follow the expert's problem-solving route, but must still reconstruct the missing values, code, or entities by itself. SMEPO is simple to apply and requires no changes to the reward function or RL objective. Across diverse domains, including math, code, and agentic search, SMEPO improves accuracy by up to 3.2 points over GRPO and reduces training time by up to 4.2x. The code is available at this https URL. 

---
# Polynomial Context-Truncation Sensitivity in Autoregressive Language Models: Sequential Wyner-Ziv Bounds for KV Cache Compression 

**Authors**: Munsik Kim  

**Link**: [PDF](https://arxiv.org/pdf/2605.25085)  

**Abstract**: We study the rate-distortion limits of online KV cache compression in autoregressive language models, formulating it as sequential Wyner-Ziv source coding on the filtration induced by the model, with the next-step query as decoder side information. Empirically, across four models spanning two families and $0.5$-$3$B parameters, we find that the next-token distribution's sensitivity to context truncation decays \emph{polynomially} rather than \emph{geometrically}: a power law improves on an exponential fit by an order of magnitude in extrapolation, the fitted exponent is recovered independently from a sink-plus-recent KL measurement, and the decay is verified to be free of positional-encoding artifacts by a position-preserving ablation. Under a corresponding \emph{polynomial truncation-sensitivity} assumption, our main result characterizes the per-token memory requirement of \emph{suffix-only} cache policies: a sliding-window scheme attains distortion $\varepsilon$ with window $w = O(\varepsilon^{-1/\alpha})$, and -- under an additional two-sided Bayes-risk condition -- a converse shows $w = \Omega(\varepsilon^{-1/\alpha})$ is necessary within this policy class, so the scaling is $\Theta(\varepsilon^{-1/\alpha})$ for suffix-only policies. Whether recurrent or propagating cache summaries can beat this scaling is left open. An explicit block-Markov scheme achieves the upper bound; its rate-of-convergence exponent matches the converse under additional forward-decay and regularity hypotheses (not implied by truncation sensitivity alone), and differs by a factor of two otherwise. Empirically, the polynomial law predicts the degradation curves of concrete cache policies: recency-based eviction (sliding, sink-plus-recent) suppresses distortion by roughly two orders of magnitude over random retention at equal budget, with a power-law decay in the budget. 

---
# Multi-Agent Specification-based Metamorphic Testing of FMU-Based Simulations 

**Authors**: Ashir Kulshreshtha, Abdullah Mughees, Gaadha Sudheerbabu, Tanwir Ahmad, Kristian Klemets, Dragos Truscan, Mikael Manngård  

**Link**: [PDF](https://arxiv.org/pdf/2605.25101)  

**Abstract**: In many industrial domains, the Functional Mock-up Interface (FMI) is used to exchange simulation models as Functional Mock-up Units (FMUs) across different partners using various modelling tools. This opens up the possibilities for simulation-based verification and validation using FMUs for ensuring reliable system behaviour. However, deriving effective test oracles for these simulation models remains challenging due to the absence of explicit expected outputs. This limits the applicability of conventional testing approaches, which require access to the internal workings of the systems. Metamorphic testing (MT) addresses this limitation by leveraging metamorphic relations (MRs), but extracting such relations from specifications remains largely a manual and error-prone process. To address this challenge, we propose an LLM-powered multi-agent workflow for specification-based metamorphic testing of FMU-based simulation models. The approach takes functional and interface specifications as input and orchestrates multiple agents to extract requirements and derive MRs. These MRs are expressed using Given-When-Then patterns to structure input conditions (Given), transformations (When), and expected output behaviours (Then). These relations are then used to generate metamorphic test cases, execute simulations, and evaluate output consistency across multiple sessions. We evaluate the approach on a Lube Oil Cooling system FMU, demonstrating its ability to automatically generate meaningful MRs and corresponding test cases. Preliminary results indicate that the proposed workflow can effectively support the systematic verification and validation of dynamic simulation models by reducing manual effort and improving test generation. 

---
# Language Bias in LVLMs: From In-Depth Analysis to Simple and Effective Mitigation 

**Authors**: Yangneng Chen, Jing Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.25036)  

**Abstract**: Large Vision-Language Models (LVLMs) extend large language models with visual understanding, but remain vulnerable to hallucination, where outputs are fluent yet inconsistent with images. Recent studies link this issue to language bias-the tendency of LVLMs to over-rely on text while neglecting visual inputs. Yet most analyses remain empirical without uncovering its underlying cause. In this paper, we provide a systematic study of language bias and identify its root in modality misalignment during training. Our analysis shows that both Visual Instruction Tuning (VIT) and Direct Preference Optimization (DPO) often prioritize textual improvements, which may cause LVLMs to overly lean toward language modeling rather than balanced multimodal understanding. To address this, we propose two simple yet effective methods: Language Bias Regularization (LBR) which mitigates language bias through regularization during instruction tuning, and Language Bias Penalty (LBP), which penalizes language bias in the DPO training process. Extensive experiments across diverse models and benchmarks demonstrate the effectiveness of our approach. LBR consistently improves performance on over ten general benchmarks, while LBP significantly reduces hallucination and improves trustworthiness. Together, these methods not only mitigate language bias but also advance the overall alignment of LVLMs, all without introducing any additional data or auxiliary models. Our code is publicly available at this https URL. 

---
# Intent Signal Theory: A Computational Framework for Intent-State Control in Human-AI Interaction 

**Authors**: Gang Peng  

**Link**: [PDF](https://arxiv.org/pdf/2605.25058)  

**Abstract**: Current AI interaction models treat the prompt as the primary object of exchange, omitting a critical layer: the user's latent source intent, the goal state preceding and motivating the prompt. Here we introduce Intent Signal Theory (IST), a computational framework that formalises this missing intent layer. IST distinguishes four objects routinely conflated: latent source intent (I*), observable intent proxy (I-hat), encoded carrier (P), and model output (O). It formalises dimensional weights, encoding masks, structural and fidelity recovery scores, and public-private intent decomposition. The Theorem of Irreversible Intent Loss establishes that private intent absent from the carrier cannot be recovered beyond generic substitution. Evidence from four companion studies spanning six LLMs, three languages and three task domains shows structural-fidelity splits, human-validated metric dissociation, and weight-tolerance plateaus consistent with IST's predictions. IST reframes prompt engineering as intent-protocol design and identifies a computational layer that current AI systems lack. 

---
# Security in the Fine-Tuning Lifecycle of Large Language Models: Threats, Defenses,Evaluation, and Future Directions 

**Authors**: Wenjuan Li, Yitao Liu, Runze Chen, Rajkumar Buyya  

**Link**: [PDF](https://arxiv.org/pdf/2605.25073)  

**Abstract**: Background: Fine-tuning is central to adapting pre-trained Large Language Models (LLMs) to downstream tasks, but its reliance on training data, parameter updates, and reusable components opens entry points for attackers. Threats have evolved from data poisoning and weight tampering to agent manipulation and interface exploitation, yet existing reviews lack a unified framework spanning the full fine-tuning lifecycle. Objective: This paper presents a systematic survey of LLM fine-tuning security and establishes a lifecycle-based framework for comparing attacks and defenses, complemented by unified empirical evaluation. Methods: We divide attack and defense mechanisms into three phases by intervention timing: pre-tuning, during-tuning, and post-tuning. Within each phase, strategies are reviewed and contrasted to expose their evolution and limitations. Representative methods are then evaluated under a unified model, hardware, and protocol setup, with cross-phase experiments pairing attacks and defenses from different phases. Results: Attack effectiveness is highly model-dependent and non-monotonic with scale: weight-editing attacks effective on earlier models lose impact on modern open-source LLMs; cross-lingual backdoor transfer, reported as near-perfect at larger scales, fails entirely on tested 1B-4B models; and purely benign samples can compromise safety alignment in instruction-tuned models. Single-phase defenses rarely generalize across phases, and defense effectiveness depends jointly on model architecture and alignment state. Conclusion: We identify key open problems (configuration-robust defense, cross-phase defense composition, and embedding-space attacks beyond behavioral assumptions) and propose concrete future research directions. 

---
# Investigating the Interplay between Contextual and Parametric Chain-of-Thought Faithfulness under Optimization 

**Authors**: Jingyi Sun, Qianli Wang, Pepa Atanasova, Nils Feldhus, Isabelle Augenstein  

**Link**: [PDF](https://arxiv.org/pdf/2605.24960)  

**Abstract**: Chain-of-Thought (CoT) faithfulness, i.e., whether CoTs genuinely reflect large language models' (LLM) underlying behavior, is typically evaluated under two disjoint paradigms: contextual faithfulness, measured by perturbing the input or CoT trace, and parametric faithfulness, assessed by intervening on a model's parametric knowledge. Yet prior work compares them only descriptively. We fill this gap by proposing FaithMate, a unified preference-alignment interface for optimizing models towards either faithfulness paradigm. It enables us to investigate the interplay between the two paradigms, examining whether and to what extent faithfulness gains generalize within and across paradigms. Across three models, two datasets, and six faithfulness metrics, we find that the two paradigms are positively coupled, yet asymmetric: optimizing towards parametric faithfulness yields consistent gains across both paradigms, whereas the contextual counterpart delivers more variable gains. Within the contextual paradigm, faithfulness gains on one metric do not consistently transfer to others, implying that existing contextual metrics capture disjoint facets of faithfulness and exposing inherent trade-offs. These findings imply that CoT faithfulness is not a monolithic objective and therefore requires multifaceted optimization and evaluation. 

---
# APT-Agent: Automated Penetration Testing using Large Language Models 

**Authors**: William Guanting Li, Alsharif Abuadbba, Kristen Moore, Dan Dongseong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2605.24949)  

**Abstract**: Penetration testing is essential to securing modern web infrastructures, yet traditional manual methods struggle to keep pace with their scale and complexity. Large Language Models (LLMs) offer new opportunities for automating these tasks, but existing approaches face two persistent challenges: hallucination of technical entities and insufficient long-term contextual memory. To address these issues, we present APT-Agent, a fully automated LLM-driven penetration testing framework that systematically orchestrates reconnaissance, exploitation, and exfiltration. APT-Agent introduces a hybrid rectification module to recover hallucinated commands and a command-specific memory architecture to preserve operational context across multi-step attack sequences. We evaluate our APT-Agent on Metasploitable 2 against seven vulnerable services spanning web, database, and network protocols. APT-Agent achieves an 84.29% end-to-end exploitation success rate, compared to 48.57% (Script Kiddie) and 18.57% (PentestGPT) under matched conditions. By reducing cognitive burden and minimizing reliance on human intervention, APT-Agent represents a step toward scalable, reliable, and cognitively efficient automation for penetration testing. 

---
# MinerU-Popo: Universal Post-Processing Model for Structured Document Parsing 

**Authors**: Bangrui Xu, Ziyang Miao, Xuanhe Zhou, Yiming Lin, Zirui Tang, Xiaomeng Zhao, Fan Wu, Cheng Tan, Fan Wu, Bin Wang, Conghui He  

**Link**: [PDF](https://arxiv.org/pdf/2605.24973)  

**Abstract**: VLM-based OCR models have become the de facto choice for document parsing, as they can accurately extract page-level elements (e.g., paragraphs within individual pages) together with their bounding boxes and textual content. However, downstream applications such as RAG require coherent document-level information, whereas these models often break cross-page continuity and fail to recover disrupted structures, such as paragraphs and tables truncated by page boundaries. Such relationships are not confined to a single page; instead, they require joint analysis of titles, paragraphs, tables, and images spanning multiple pages. A natural solution is therefore to reuse existing OCR outputs and reconstruct document-level logical structures through post-processing.
To this end, we propose MinerU-Popo, a lightweight and universal framework for POst-Processing OCR outputs, which converts page-level results from diverse parsers into coherent document-level structures. MinerU-Popo decomposes the problem into four focused subtasks: text truncation recovery, table truncation recovery, title hierarchy reconstruction, and image-text association. To address these effectively, we build a task-oriented data engine with task-specific input filtering, and use the generated data (30K) to fine-tune a lightweight post-processing model (Qwen3-VL-4B). To support long documents, we introduce dynamic chunking with overlap-based synchronization, which aligns chunk-level outputs from the fine-tuned model and preserves global consistency. Finally, we assemble the aligned outputs into a tree-structured document representation, further enriched with node chunking and summaries for downstream retrieval and analysis. Empirical results show MinerU-Popo improves title-hierarchy TEDS by at least 20% across all five tested OCR models, improves RAG accuracy and reduces per-query latency. 

---
# When Reasoning Hurts: Source-Aware Evaluation of Frontier LLMs for Clinical SOAP Note Generation 

**Authors**: Faizan Faisal  

**Link**: [PDF](https://arxiv.org/pdf/2605.24902)  

**Abstract**: Reasoning-enabled LLMs perform strongly on medical reasoning benchmarks, but it remains unclear whether these gains transfer to structured clinical documentation; we investigate this question using SOAP note generation from clinical dialogue in a source-aware benchmark spanning OMI Health, ACI-Bench, and PriMock57. We evaluate GPT-5.4, DeepSeek-V4-Flash, and Gemma-4-E4B in a controlled 2x2 design that independently toggles provider-native reasoning and same-source retrieval-augmented generation (RAG). Outputs are assessed using seven automatic metrics alongside two reference-aware LLM judges. Both evaluation approaches agree that a non-reasoning GPT-5.4 configuration achieves the highest overall quality, while DeepSeek-V4-Flash performs best among reasoning-enabled configurations. Enabling reasoning significantly degrades GPT-5.4 performance across all three datasets, whereas same-source RAG yields smaller, model-dependent improvements. Overall, the findings indicate that stronger reasoning capability should not be assumed to improve fidelity-sensitive SOAP note generation without dedicated, task-specific evaluation. 

---
# Reflect-Guard: Enhancing LLM Safeguards against Adversarial Prompts via Logical Self-Reflection 

**Authors**: Lixing Lin, Juli You, Yue Li, Luyun Lin, Yiqing Wang, Zhen Zhang, Moxuan Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2605.24834)  

**Abstract**: Large language model (LLM) safety classifiers such as Llama Guard are effective at detecting overtly harmful prompts but remain vulnerable to adversarial jailbreak attacks that disguise malicious intent through role-play scenarios, fictional framing, and indirect requests. We present Reflect-Guard, a method that augments LLM-based safety classifiers with chain-of-thought self-reflection capabilities through parameter-efficient fine-tuning. Our approach distills analytical reasoning from GPT-4o-mini into structured reflection annotations, then trains Llama-Guard-3-8B via QLoRA to generate logical self-reflections before issuing safety verdicts. Using only 1000 training examples and updating just 0.5% of model parameters (~42M), Reflect-Guard achieves substantial improvements on two challenging benchmarks. On WildGuardTest, F1 score improves from 0.770 to 0.842 (+7.2 pp), with recall on adversarial prompts increasing from 0.513 to 0.921 (+40.8 pp). On JailbreakBench, the attack success rate drops from 10.3% to 1.8%, representing an 82.5% relative reduction. These gains are especially pronounced on adversarial inputs, where the explicit reasoning step enables the model to see through obfuscation techniques that defeat standard pattern-matching approaches. Our results demonstrate that teaching safety classifiers to reason about adversarial intent, rather than simply classify surface patterns, is a promising direction for robust LLM safety. 

---
# Tiny Brains, Giant Impact: Uncovering the Keystone Neurons of LLM with Just a Few Prompts 

**Authors**: Xiangtian Ji, Yuxin Chen, Zhengzhou Cai, Xiang Wang, An Zhang, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2605.24846)  

**Abstract**: Large language models (LLMs) display strong comprehensive abilities, yet the internal mechanisms that support these behaviors remain insufficiently understood. In this work, we show that across a wide range of open-weight Transformers, a subset of neurons remains consistently highly activated during inference across tasks of multiple capability dimensions. By probing along the cross-task activation strength, an extremely sparse subset is isolated, whose removal causes a collapse in model behavior, which we term keystone neurons. Our analysis reveals that keystone neurons are a stable and intrinsic neuron subset of the model that is largely established during pretraining. The parameters associated with these neurons are tightly calibrated during the training process, and their precise values are critical for the capabilities of the model. Building on these insights, we propose a supervised fine-tuning approach that updates only keystone neurons, achieving task gains comparable to or even better than full-parameter fine-tuning while better preserving performance in other capability dimensions, despite modifying a much smaller number of parameters. 

---
# The Concept Allocation Zone: Tracking How Concepts Form Across Transformer Depth 

**Authors**: James Henry  

**Link**: [PDF](https://arxiv.org/pdf/2605.24856)  

**Abstract**: Concept formation in transformer language models is depth-extended, not a single-layer event: concepts emerge gradually across a contiguous region of the residual stream. Mechanistic interpretability methods identify the single layer of peak class separation -- the "best layer" -- capturing a snapshot rather than the process itself. We introduce the Concept Allocation Zone (CAZ): the depth interval within which a concept becomes measurably separable, the region allocated to its geometric expression. We formalize the CAZ through three layer-wise metrics (Separation, Concept Coherence, Concept Velocity) and derive principled boundary detection without manual layer sweeps. A CAZ is not a concept: it is the depth region within which the model organizes its geometry to make a concept separable. A single concept typically participates in multiple CAZes; multiple concepts may share one. Empirical validation across 34 models from 8 architectural families and 7 concepts reveals that the separation curve S(l) is frequently multimodal. A scored detector uncovers "gentle CAZes" -- subtle allocation regions invisible to standard peak detection but causally active in 93-100% of cases under ablation (16 of 34 models; 26 in the companion validation paper). The framework generates seven testable predictions; four yield clear verdicts (two not supported, one partially supported, one supported), one had its precondition invalidated by the data, and two are underpowered -- with cross-architecture alignment confirmed as depth-matched rather than monolithic under leave-one-concept-out cross-validation. Reference implementation: rosetta_tools v1.3.1 (doi:https://doi.org/10.5281/zenodo.20361433). 

---
# Riemannian-Manifold Steering: Geometry-Aware Generative Autoencoders for Label-Free Steering 

**Authors**: Narmeen Oozeer, Shivam Raval, Philip Quirke, Manikandan Ravikiran, Jeff Phillips, Shriyash Upadhyay, Amirali Abdullah  

**Link**: [PDF](https://arxiv.org/pdf/2605.24942)  

**Abstract**: Steering a language model - intervening on its internal activations to change downstream behaviour - has recently expanded beyond linear interpolation to nonlinear methods such as angular and kernelized steering, which define intervention transformations without learning an explicit geometry over paths in activation space. Freshly introduced geometry-aware manifold methods do learn such a geometry, but require labelled class centroids together with prescribed cyclic or sequential structure. These assumptions restrict where manifold steering can be applied, since existing constructions require labelled centroids and compatible boundary conditions. We recast manifold steering more broadly as \textbf{Riemannian geodesic computation} on activation space, recovering linear and labelled-spline steering as geodesics under particular choices of metric. A principled metric within this framework is the output-space Hellinger distance pulled back to activations; we approximate this with a learned encoder trained on output distances over a small concept-token schema - no per-prompt labels, no topology prior, and no per-task curve fitting. Empirically, the method reliably drives the model onto the target class across all tasks in a standard four-task language-model arithmetic benchmark, while following more behaviourally natural trajectories than baselines on smaller output spaces. We thereby provide a unified Riemannian framework for manifold steering together with a schema-supervised, label-free instantiation that operates without labelled centroids or prescribed boundary conditions. 

---
# Towards a Universal Causal Reasoner 

**Authors**: Qirun Dai, Xiao Liu, Jiawei Zhang, Dylan Zhang, Hao Peng, Chenhao Tan  

**Link**: [PDF](https://arxiv.org/pdf/2605.24873)  

**Abstract**: Despite the importance of causal reasoning, training LLMs to reason causally remains underexplored. Existing data efforts mostly focus on benchmarking LLMs on specific aspects of causality, making them less suitable for training generalizable causal reasoners. To address this, we propose UniCo, a data generation framework that both (1) addresses 18 causal query types across Pearl's Causal Ladder and (2) translates natively symbolic examples into code and natural language forms to simulate real-world use cases where causal terms are not explicitly specified. To ensure data quality, UniCo grounds answers with exact causal inference and filters cases with reasoning shortcuts. Upon supervised finetuning with 66.6K UniCo-generated instances, Qwen3-4B, Qwen3-8B and Olmo-3-7B-Instruct achieve an average of 22.9% improvements across all 18 in-distribution query types, and 8.1% over state-of-the-art causal data generation frameworks on 7 established causal benchmarks outside the training distribution. More importantly, in real-world medical understanding, legal decision, and tabular reasoning, UniCo-trained models consistently display more faithful reasoning traces, outperforming the base models by an average of 20.2% in faithfulness metrics. These suggest that causality-centered training not only strengthens causal reasoning, but also equips LLMs with a causal mindset in general reasoning tasks. 

---
# Divide-and-Conquer Inference for Large-Scale Visual Recognition with Multimodal Large Language Models 

**Authors**: Zhipeng Ye, Jiaqi Huang, Feng Jiang, Qiufeng Wang, Yikang Duan, Dawei Wang, Xihang Zhou, Qian Qiao  

**Link**: [PDF](https://arxiv.org/pdf/2605.24799)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated strong capabilities across a wide range of vision language tasks. However, when applied to large scale image classification, their performance degrades significantly as the label space expands a phenomenon we define as Performance Collapse in Long Sequence Recognition. Through an information theoretic analysis, we reveal that this collapse stems from a fundamental conflict between the escalating information entropy and the prominent attention dilution and decay within attention mechanisms, which impairs the model's ability to maintain a sufficient signal-to-noise ratio when processing extremely long prompts. To mitigate this, we propose Divide-and-Conquer Inference (DCI), a novel test-time scaling strategy for visual recognition with MLLMs. DCI recursively decomposes complex global classification tasks into multiple simpler, localized subproblems and employs a dynamic pruning mechanism to compress the search space. This method effectively improves the local signal to noise ratio and model accuracy by mitigating the inherent weight dilution issues in long-sequence inference. Moreover, while traditional self-attention incurs a prohibitive quadratic computational complexity, DCI achieves more favorable scaling behavior and substantially accelerates inference in large scale classification scenarios. Extensive experiments on benchmarks such as ImageNet-1K and ImageNet-21K demonstrate that DCI consistently improves classification accuracy. This enables lightweight open-source models to rival or even surpass frontier closed-source giants without any additional training or fine-tuning. As a model-agnostic, plug-and-play paradigm, DCI offers an efficient approach for scaling the inferential precision of MLLMs in large-scale scenarios. 

---
# Zero-Shot Parkinson's Disease Detection from Speech: Comparing Large Audio and Language Models 

**Authors**: Muhammad Ashad Kabir, Sirajam Munira  

**Link**: [PDF](https://arxiv.org/pdf/2605.24806)  

**Abstract**: Large audio and language models have recently demonstrated zero-shot reasoning capabilities across various domains. However, it remains unclear how the form of audio input, whether handcrafted acoustic features extracted from speech or the raw audio waveform itself, affects performance for Parkinson's disease (PD) detection across different languages. In this study, we systematically compare two input modalities for zero-shot PD detection: (i) handcrafted acoustic features extracted from speech recordings analyzed by a general-purpose LLM, and (ii) direct waveform input analyzed by audio-capable models. Experiments on PD speech datasets in four languages show that performance varies across input modalities, speech tasks, and languages. Handcrafted acoustic features provide more stable performance in a low-resource language (e.g., Bengali), whereas audio input yields dataset-dependent gains. These findings highlight the impact of input modality on zero-shot PD detection from speech. 

---
# Spectral Retrieval: Multi-Scale Sinc Convolution over Token Embeddings for Localized Retrieval in LLM Multi-Agent Systems 

**Authors**: Andrea Morandi  

**Link**: [PDF](https://arxiv.org/pdf/2605.24764)  

**Abstract**: [Abridged] - Spectral Retrieval is a plug-in re-ranking stage that interpolates between per-token MaxSim and mean-pool retrieval through a multi-scale sinc convolution over token embeddings. In standard dense retrieval each document is one mean-pooled vector; when relevance localises into a short subspan, the signal averages into noise. Spectral Retrieval reuses per-token embeddings from a late-interaction index and convolves them with a normalised sinc kernel at multiple scales. At L=1 the kernel acts as the identity, recovering per-token MaxSim; as L grows it approaches a uniform filter, recovering mean pooling. The maximum cosine over positions and scales yields a score provably no less informative than either endpoint. On a controlled synthetic benchmark with 1,000 documents and planted single-position spikes, mean-pool retrieval sits at chance (Recall@10 ~ 0.02) regardless of spike strength, while Spectral Retrieval reaches Recall@10 = 1.0 once the planted cosine exceeds the corpus-level token noise floor. On LIMIT-small with a frozen all-mpnet-base-v2 encoder, Spectral Retrieval lifts Recall@10 from 0.33 to 0.90, MRR from 0.22 to 0.79, and strict Success@10 from 0.12 to 0.84, without retraining. The method fits naturally into multi-agent LLM systems, where each agent benefits from a tighter, role-specific retrieval window over a shared corpus. 

---
# CONF-KV: Confidence-Aware KV Cache Eviction with Mixed-Precision Storage for Long-Horizon LLM 

**Authors**: Yubo Li, Yidi Miao  

**Link**: [PDF](https://arxiv.org/pdf/2605.24786)  

**Abstract**: Long-horizon LLM inference turns the key--value (KV) cache into the dominant GPU memory consumer and makes per-token attention increasingly expensive. Many common eviction policies use static recency windows or historical attention, leaving unused a signal computed on every decoding step: the model's current uncertainty. We introduce CONF-KV, a KV-cache manager that converts the next-token distribution into a scalar confidence score and uses it to choose the per-step cache budget, retaining more context when the model is uncertain and pruning aggressively when it is confident. Within each budget, tokens are ranked by a composite of accumulated attention mass and recency, while a protected recent window preserves local coherence. We combine the policy with blockwise online-softmax attention, mixed FP16/INT8 storage, and a pyramidal per-layer budget variant. Across four model families and generated lengths up to 4K, CONF-KV stays near the footprint of a fixed 512-token sliding window while remaining within 1.5--2.1 perplexity points of full KV. On Needle-in-a-Haystack up to 32K tokens, CONF-KV reaches 91.4% retrieval accuracy versus 53.8% for sliding windows and 80.6% for H2O; on 75 VisualWebArena tasks it retains 95.3% of full-KV success at 2.8 times lower peak memory. 

---
# TS-Skill: A Benchmark for Evaluating Analytical Skills in Time-Series Question Answering 

**Authors**: Liying Han, Kang Yang, Oliver Wang, Jason Wu, Pengrui Quan, Gaofeng Dong, Ozan Baris Mulayim, Sizhe Ma, Yuyang Yuan, Dezhi Hong, Mario Berges, Mani Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2605.24703)  

**Abstract**: Large language models (LLMs) and time-series language models (TSLMs) are increasingly applied to time-series question answering (TSQA). Unlike text-only QA, TSQA requires models to ground answers in temporal signals whose patterns may occur at different scales, specific time locations, or across separated intervals. However, existing benchmarks are typically organized by task types or high-level reasoning categories, making it difficult to diagnose the underlying signal-level capabilities driving model performance. We introduce TS-Skill, a controlled benchmark for evaluating three composable analytical skills in TSQA: temporal scale selection (SK1), temporal localization (SK2), and cross-interval integration (SK3). TS-Skill provides timestamp-aware questions, broad domain coverage, and human-validated QA quality. To construct the benchmark at scale, we develop SKEvol, a skill-guided agentic framework that combines domain-aware time-series seed generation, skill-controlled question generation, metadata- and code-assisted answer construction, multi-phase signal-grounded verification, and human-in-the-loop curation. Experiments on ten state-of-the-art LLMs and TSLMs reveal substantial and uneven capability gaps across SK1-SK3. In particular, SK3 remains consistently challenging for non-agent models, whereas tool-augmented agents show a selective advantage on standalone SK3. These findings demonstrate that skill-level evaluation can uncover temporal reasoning failures that are obscured by aggregate TSQA scores. 

---
# The Path Matters: Learning a Token-Commitment Policy for Diffusion Language Models 

**Authors**: Bohang Sun, Max Zhu, Francesco Caso, Jindong Gu, Junchi Yu, Philip Torr, Pietro Liò, Jialin Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.24697)  

**Abstract**: Diffusion large language models promise faster generation by refining many token positions in parallel, but this parallelism introduces a hidden control problem: which proposed tokens should be transferred into the partially decoded sequence at each step? We refer to this decision as token commitment. Existing frozen-generator decoders largely rely on hand-designed confidence rules or block-specific acceptance filters. We argue that token commitment can instead be learned as a reusable trace-state policy. We introduce TraceLock, a lightweight plug-in controller that instantiates this policy for a frozen diffusion language model. Since oracle commitment times are unavailable, TraceLock derives self-supervision from future stability: at decoding step t, a proposed token for position i is labeled stable if it matches the final token at position i after the full decoding trace completes. The controller scores variable-length trace states and decides which active token proposals should be committed to the partially decoded sequence. Once trained for a given frozen backbone, the controller can be deployed across local-window widths, generation lengths, and step budgets without retraining or per-setting calibration. Experiments on question answering, mathematical reasoning, and code generation show that TraceLock improves the quality-step tradeoff over heuristic and learned baselines, with particularly stable behavior under cross-setting deployment. Diagnostic analyses show that its decisions are not reducible to scalar confidence, suggesting that frozen diffusion language models expose a learnable space of commitment trajectories beyond confidence-based decoding. Code is available at this https URL. 

---
# Who judges the judges? Governance from metrics: a runtime framework for continuous LLM compliance monitoring 

**Authors**: Jehanne Dussert  

**Link**: [PDF](https://arxiv.org/pdf/2605.24737)  

**Abstract**: Current approaches to AI compliance treat conformity as a binary, audit-time verdict rather than a continuous, measurable property of production systems. We argue that this compliance fiction is structurally ill-suited to the requirements of the EU AI Act, which demands ongoing human oversight and the detection of emergent behavioural drift in deployed systems. We introduce governance from metrics, a principle whereby regulatory compliance is derived as a continuous signal from runtime observability rather than from static assessments. Building on this principle, we present govllm, an open-source framework implementing a governance-driven routing architecture in which model selection is determined by accumulated compliance scores rather than by latency or cost alone. Central to our approach is a panel of regulatory judges - LLM evaluators specialised per criterion (EU AI Act, GDPR, ANSSI, accessibility) - whose inter-judge disagreement we reframe not as noise but as a regulatory uncertainty signal warranting human arbitration. We validate this approach through a ground truth corpus of 49 annotated prompt/response pairs across five regulatory criteria, evaluated by four small language models (SLMs, 1.7B-7B parameters) running fully on-premise. Agreement rates range from 51.5% (mistral:7b) to 69.1% (phi4-mini), with no single model dominating across all criteria - empirically motivating the Profile-as-jury design. We further document three structural failure modes in small regulatory judges and a judge-specific position bias that degrades agreement by up to 25 percentage points across three question-order conditions (original, reversed, permuted). govllm is released as open-source software to support reproducible AI governance research. 

---
# How Many Tools Should an LLM Agent See? A Chance-Corrected Answer 

**Authors**: Vyzantinos Repantis, Ameya Gawde, Harshvardhan Singh, Joey Blackwell II  

**Link**: [PDF](https://arxiv.org/pdf/2605.24660)  

**Abstract**: Before an LLM agent can use a tool, a retrieval system must decide which candidate tools to show to the agent. How long should that shortlist be? Show too many tools and the model struggles to choose. Show too few and the correct tool may not appear. Most systems apply a fixed shortlist size to every query, but no standard metric exists to evaluate whether that size was appropriate. We treat the number of tools shown to an LLM agent as the object of evaluation and we apply Bits-over-Random (BoR), a chance-corrected metric that asks whether success at a given depth is better than what random selection would achieve at that same depth. We evaluate BoR across three tool-selection benchmarks, multiple scorers, and registries ranging from 20 to 3,251 tools. We then turn the same principle into a reinforcement learning (RL) reward for choosing tool shortlist depth per query. The RL agent is deliberately simple, serving as a probe of the metric rather than a proposed system. As the shortlist grows, random chance of including the correct tool rises, so the reward naturally decreases, reducing the need for an engineered depth penalty. On BFCL (370 tools), the learned policy nearly matches the coverage of showing 50 tools ($90.3\%$ vs $90.8\%$) while presenting only 7 on average. On ToolBench (3,251 tools), a fixed shortlist of 5 tools achieves higher aggregate coverage ($64.7\%$ vs $61.9\%$) but finds nothing on hard queries (correct tool ranked 6th-20th). The BoR agent finds $16.7\%$ on those same queries by searching deeper. Downstream validation with Claude Sonnet 4.6 indicates that shorter adaptive lists also improve the LLM's ability to select the right tool: $93.1\%$ versus $87.1\%$ when always shown 5 tools, widening to $76.8\%$ vs $60.9\%$ on medium-difficulty queries where the correct tool is present but not ranked first. 

---
# Motion-Compensated Weight Compression 

**Authors**: Ismail Lamaakal  

**Link**: [PDF](https://arxiv.org/pdf/2605.24754)  

**Abstract**: Neural network weights are increasingly a bottleneck for deployment, yet most compression pipelines treat layers independently and overlook cross-layer redundancy induced by function-preserving symmetries. We propose Motion-Compensated Weight Compression (MCWC), a weight-only codec that aligns permutation-symmetric blocks (e.g., hidden units and attention heads) to maximize cross-layer correspondence, turning depth into a predictable sequence. In the aligned coordinate system, MCWC uses a lightweight layer-sequential predictor with periodic keyframes and encodes only quantized prediction residuals using a learned entropy model trained under a rate distortion objective. A simple decoder reconstructs deployable weights by entropy decoding, dequantization, predictor-driven reconstruction, and inverse alignment, enabling fast weight materialization for inference. Across Transformer language modeling and vision classification, MCWC improves the rate accuracy Pareto frontier over strong quantization and learned weight-codec baselines, while maintaining competitive decode time. Ablations confirm that alignment, prediction, entropy modeling, and keyframe scheduling are each necessary for the full gains. Our code is available via this https URL. 

---
# VaaWIT: Visual-Aware Adaptation of Large Language Models for Multilingual Web Image Translation 

**Authors**: Bo Li, Ronghao Chen, Ningyuan Deng, Huacan Wang, Shaolin Zhu, Lijie Wen  

**Link**: [PDF](https://arxiv.org/pdf/2605.24675)  

**Abstract**: Translating text embedded in Web images is crucial for improving content accessibility and cross-lingual information retrieval, particularly within social media and e-commerce domains. Although Large Vision-Language Models (LVLMs) have advanced multimodal understanding, applying them to Web image translation remains challenging due to the visual representation gap: standard encoders often prioritize high-level semantics over the fine-grained visual details required for recognizing diverse character morphologies. To address this challenge, we propose VaaWIT, an end-to-end framework that adapts Large Language Models for multilingual Web image translation. The framework introduces two key technical contributions: (1) a Dual-Stream Attention Module (DSAM), which facilitates bidirectional interaction between multilingual semantic features and detailed visual representations, thereby synthesizing unified features robust to textual variations; and (2) a Visual-Aware Adapter (VAA), a parameter-efficient fine-tuning strategy that dynamically injects these fused visual cues into the frozen LLM backbone. This design enables the model to align the visual context with linguistic reasoning effectively while minimizing computational costs. Extensive experiments on eight tasks on three public benchmarks demonstrate that VaaWIT significantly outperforms state-of-the-art (SOTA) open-source baselines and achieves competitive performance against proprietary models. These results validate the efficacy of integrating fine-grained visual perception into LLMs for complex Web content analysis. 

---
# Demystifying the Mythos or Disrupting Bugonomics? From Zero-Day Asymmetry to Defender Remediation Throughput 

**Authors**: Alfredo Pesoli, Herman Errico, Lorenzo Cavallaro  

**Link**: [PDF](https://arxiv.org/pdf/2605.24632)  

**Abstract**: Recent demonstrations of large language models producing candidate and confirmed vulnerabilities in production software have renewed the narrative that AI will reshape offensive and defensive security. Headlines emphasize capability; they rarely interrogate costs and incentives. This paper examines LLM-driven vulnerability discovery through a bugonomics lens: the operational economics of producing, proving, prioritizing, and fixing security-relevant defects. Historically, the most visible high-end bugonomics was offense-priced because production-grade zero-days and exploit chains were expensive specialist outputs for governments, brokers, and offensive vendors. Defender-side bugonomics already existed in vulnerability research, reward programs, and vendor remediation work; LLM-assisted systems change its scale and distribution. They make candidate generation, code comprehension, harness construction, proof-of-impact drafting, and report preparation cheaper at codebase scale. Exploits and proofs of concept remain important, but in defender workflows they primarily prove impact, guide prioritization, and justify remediation. The resulting bottleneck is not only finding more bugs; it is absorbing, validating, triaging, patching, and shipping a larger stream of reports. Using public data from Anthropic's Mythos Preview and Mozilla Firefox collaborations, along with public exploit-market price anchors and vulnerability reward programs, we argue that the near-term shift is not simply more zero-days. It is a move toward broader defender remediation throughput: low-signal candidates become cheaper, evidence-rich remediation become more important, and scarce capacity shifts toward maintainer review and release work. The effect is acute in open source, where LLM-assisted discovery can increase report volume while maintainer-side validation, triage, funding, and release capacity may not scale. 

---
# Mix-MoE: Improving Multilingual Machine Translation of Large Language Models through Mixed MoEs 

**Authors**: Bo Li, Tianyu Dong, Shaolin Zhu, Deyi Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2605.24681)  

**Abstract**: Large Language Models (LLMs) have shown great promise in multilingual machine translation (MT), even with limited bilingual supervision. However, fine-tuning LLMs with parallel corpora presents major challenges, namely parameter interference. To address these issues, we propose Mix-MoE, a mixed Mixture-of-Experts framework designed to train LLMs for multilingual MT. Our framework operates in two distinct stages: (1) post-pretraining with MoE on monolingual corpora, and (2) post-pretraining with MoE on parallel corpora. Crucially, we divide the MoE layers into two specialized groups: Language Model Experts (LM Experts) and Machine Translation Experts (MT Experts). LM Experts are designed to capture and retain the monolingual knowledge learned by the pre-trained LLM. MT Experts, on the other hand, are specifically trained to acquire and store bilingual translation knowledge. Furthermore, to facilitate effective interaction between these specialized experts and leverage potential underlying structural patterns in text, we introduce a routing mechanism enhanced by Fourier Transform features derived from model representations. The experimental results demonstrate that Mix-MoE excels in multilingual MT, significantly outperforming existing baselines and showing notable progress in mitigating parameter interference. 

---
# Correcting Visual Blur Induced by Attention Distraction to Reduce Hallucinations: Algorithm and Theory 

**Authors**: Quanjiang Li, Zhiming Liu, Wei Luo, Tingjin Luo, Chenping Hou  

**Link**: [PDF](https://arxiv.org/pdf/2605.24602)  

**Abstract**: Multimodal large language models (MLLMs) frequently suffer from object hallucinations, yet the visual perceptual mechanism underlying this failure remains poorly understood. In this work, we reveal that hallucinations are strongly associated with a human-like attention distraction phenomenon, where humans under divided focus experience degraded visual clarity and produce inaccurate descriptions, while in models the same mechanism manifests as spatial inconsistency in multi-head attention and temporal fading of attention to image tokens during decoding. We further provide theoretical insights that attention dispersion increases model complexity and degrades classification generalization. Motivated by these findings, we propose an Attention-Focused Approach for Improved Image Perception (AFIP), which corrects attention distraction via cross-head attention enrichment and reinforces visual grounding through dynamic historical attention enhancement. Extensive experiments on multiple benchmarks and models validate the effectiveness of AFIP without additional training. 

---
# Bilevel Optimization of Synthetic Trajectories for Multi-Turn LLM Fine-Tuning 

**Authors**: Shresth Verma, Mauricio Tec, Cheol Woo Kim, Kai Wang, Milind Tambe  

**Link**: [PDF](https://arxiv.org/pdf/2605.24743)  

**Abstract**: While LLMs excel at single-turn generation, they struggle with long-horizon, multi-turn interactions. Offline reinforcement learning (RL) offers a scalable approach, yet its performance hinges on the availability and quality of multi-turn trajectory data. A common remedy is to augment training with synthetic trajectories generated by LLMs or simulators, but synthetic data is highly heterogeneous in quality, and naively treating all trajectories as equally informative can degrade performance. We propose BOOST, a bilevel optimization framework where the inner level trains the LLM on reweighted data and the outer level trains a lightweight reweighting head on held-out real validation tasks, assigning continuous trajectory-level weights without requiring an external judge. To ground this approach, we derive a PAC-Bayesian bound revealing a three-way trade-off: synthetic data increases diversity but risks task-shift, while concentrating weight on high-quality trajectories improves empirical performance at the cost of effective sample size. Empirically, our method consistently outperforms multiple baselines. Analysis reveals it upweights synthetic trajectories that align with the real data distribution and exhibit higher qualitative merit. 

---
# World-State Transformations for Neuro-symbolic Interactive Storytelling 

**Authors**: Santiago Góngora, Luis Chiruzzo, Gonzalo Méndez, Pablo Gervás  

**Link**: [PDF](https://arxiv.org/pdf/2605.24719)  

**Abstract**: Large Language Models (LLMs) have changed the possibilities of Interactive Storytelling systems that process free-text user input. However, as more of these systems are built, evidence continues to mount regarding the story coherence problems that arise when relying solely on them. Recent research suggests that LLMs can effectively predict state changes within rule-based Interactive Storytelling systems, triggering pre-programmed world-state transformations.
In this paper, we conduct an exploratory evaluation of whether such transformations can serve as a catalyst for player expression while aiming to address the incoherence issues typical of purely LLM-based approaches. Building upon a neuro-symbolic architecture, we conducted experiments using an open-source model (Llama 3 70B) and a closed-source model (Gemini 1.5 Flash), with testing conducted in both English and Spanish. Eight participants played two scenarios, carefully designed to assess different evaluation objectives. Our observations suggest that transformations offer a way to maintain world-state consistency while encouraging players to interact creatively through their written inputs. 

---
# Measuring the Depth of LLM Unlearning via Activation Patching 

**Authors**: Jaeung Lee, Dohyun Kim, Jaemin Jo  

**Link**: [PDF](https://arxiv.org/pdf/2605.24614)  

**Abstract**: Large language model (LLM) unlearning has emerged as a crucial post-hoc mechanism for privacy protection and AI safety, yet auditing whether target knowledge is truly erased remains challenging. Existing output-level metrics fail to detect when this knowledge remains recoverable from internal representations. Recent white-box studies reveal such residual knowledge but often rely on auxiliary training or dataset-specific adaptations, leaving no generalizable metric. To address these limitations, we propose the Unlearning Depth Score (UDS), a metric that quantifies the mechanistic depth of unlearning via activation patching. UDS first identifies layers that encode the target knowledge using a retain model baseline, then measures how much of it is erased in the unlearned model on a 0-1 scale. In a meta-evaluation across 20 metrics on 150 unlearned models spanning 8 methods, UDS achieves the highest faithfulness and robustness, confirming our causal approach as the most reliable for unlearning evaluation. Case studies further reveal that white-box metrics can disagree at the layer level and that erasure depth varies across examples. We provide guidelines for integrating UDS into existing benchmarking frameworks and streamlining the evaluation pipeline. Code and data are available at this https URL 

---
# Guarded Repair for Harm-Aware Post-hoc Replacement of LLM Mathematical Reasoning 

**Authors**: Haizhou Xia  

**Link**: [PDF](https://arxiv.org/pdf/2605.24613)  

**Abstract**: Post-hoc repair of LLM mathematical reasoning introduces an asymmetric risk: fixing an incorrect reasoning trace is useful, but replacing a trace that was already correct can be harmful. We study this problem under a selective replacement setting, where a system must decide whether a repaired candidate is safer than preserving the original cached trace. We present GuardedRepair, a guarded best-of-N repair framework that diagnoses cached reasoning traces, selectively triggers repair, and accepts answer-changing candidates only when deterministic verification guards support replacement. The framework combines lightweight symbolic checks, surface semantic-risk diagnostics, bounded candidate generation, and conservative acceptance policies. On the full GSM8K test set, where the initial reasoner already achieves 95.60% accuracy, GuardedRepair improves final accuracy to 96.89%, fixing 17 of 58 remaining errors without measured broken-correct cases in the main run. On a weak-reasoner ASDiv setting, accuracy improves from 78.40% to 87.60%. Direct regeneration baselines show that this gain is not explained by stronger-model re-solving alone: re-solving all GSM8K examples lowers accuracy to 93.03% and breaks 47 initially correct answers. Additional analyses show that guarded repair substantially improves the fixed/broken tradeoff, while also revealing that replacement risk is reduced rather than eliminated. These results support viewing post-hoc repair as harm-aware selective replacement rather than unconstrained re-solving. 

---
# SemanticZip: A Pilot Framework for Lossy Text Compression with LLMs as Semantic Decompressors 

**Authors**: Natalia Trukhina, Vadim Vashkelis  

**Link**: [PDF](https://arxiv.org/pdf/2605.24541)  

**Abstract**: Text compression for large language model (LLM) systems is usually framed as token deletion, retrieval, summarization, or exact reconstruction. We study a more aggressive but explicitly lossy setting: compress text into compact codes that an LLM can expand into task-relevant meaning. We call this setting SemanticZip. Unlike lossless compression, SemanticZip does not require byte-identical reconstruction; unlike ordinary summarization, it treats model-based decompression as part of the codec and evaluates whether task-relevant semantic commitments are recovered.
This paper is a pilot framework, not a benchmark claim. We formalize LLM-mediated decompression, define a protected/lossy packet architecture, and evaluate six representation regimes over five author-constructed diagnostic cases: structured prose, JSON, CCL-Core, CCL-Min, SemanticZip ASCII, and SemanticZip emoji. An independent decoder LLM reconstructs typed semantic atoms from each compressed representation, and we score Critical Atom Recall, Weighted Atom Recall, precision, and tokenizer gain. In this pilot, structured prose has the highest recoverability, with WAR = 0.956 and 19.1% o200k_base token gain. CCL-Min is the strongest balanced point, with 39.4% token gain and WAR = 0.874. SemanticZip ASCII provides the largest useful compression, with 46.5% token gain and WAR = 0.802, while emoji-heavy SemanticZip performs worse on both compression and recovery.
The main contribution is not the claim that these numbers establish a universal frontier. Rather, we introduce a reproducible experimental interface for studying lossy, LLM-decompressible text codes and a design principle: safety-critical and exact commitments should remain protected, while predictable low-risk context may be semantically zipped. 

---
# FoodMonitor: Benchmarking MLLMs for Explainable Compliance Analysis 

**Authors**: Ruihao Xu, Xingming Shui, Jingxuan Niu, Yiqin Wang, Jilin Yu, Haoji Zhang, Yansong Tang  

**Link**: [PDF](https://arxiv.org/pdf/2605.24503)  

**Abstract**: As AI-powered compliance monitoring becomes increasingly important in public governance and industrial safety, the ability to provide verifiable evidence and traceable accountability signals is essential. However, existing video anomaly detection datasets focus on event-level binary classification, lacking the rule-driven, explainable analysis required for real-world compliance scenarios. We introduce FoodMonitor, a benchmark for explainable compliance analysis in commercial kitchen surveillance. FoodMonitor comprises 477 video clips with 3,307 violation annotations across a dual-channel design covering both person-level and environment-level violations. Each annotation specifies which rule was violated, what non-compliant behavior occurred, and who committed it with frame-level bounding boxes. We establish a unified evaluation protocol with a two-stage matching mechanism that separately assesses spatial localization and semantic understanding, along with a composite metric ($C_{\text{score}}$) that balances environment and person detection performance. Systematic evaluation of several state-of-the-art multimodal large language models reveals that the best-performing model achieves only 0.360 $C_{\text{score}}$, with spatial localization and fine-grained rule understanding emerging as the primary bottlenecks. Our analysis identifies two distinct failure modes: localization-dominated errors and semantics-dominated errors, providing diagnostic insights for future model development. 

---
# Polymorphism Is Rotation: Operational Mechanistic Interpretability from a Two-Layer Transformer to Pythia-70m 

**Authors**: Jordan F. McCann  

**Link**: [PDF](https://arxiv.org/pdf/2605.24577)  

**Abstract**: Independently trained transformers compute the same function in residual-stream bases that differ by a uniform random rotation on $\mathrm{SO}(d_{\mathrm{model}})$. We call this phenomenon polymorphism: same function, mutually unintelligible interior coordinates. One matrix multiplication per model pair removes it: an orthogonal Procrustes fit on a single batch of activations transfers sparse-autoencoder feature dictionaries and steering vectors between independently trained models, with no retraining.
The phenomenon is invisible to the standard SAE universality metric. Decoder-column cosine similarity matches across seeds at 98%, the SAE-universality headline number, while an SAE trained on one seed reconstructs another seed's activations at negative explained variance, worse than predicting the constant mean. The decoder columns align; the encoder reads from a rotated frame. A single Procrustes rotation $R$ restores reconstruction to within 0.025 EV of the within-seed ceiling at every internal site.
$R$ is Haar-distributed: $\|R - I\|_F$ matches the random-orthogonal prediction $\sqrt{2 d_{\mathrm{model}}}$ to 0.1% at $d_{\mathrm{model}} = 512$, and a Kolmogorov-Smirnov test of $R$'s eigenvalue spectrum against Haar $\mathrm{SO}(d_{\mathrm{model}})$ returns $p \approx 1.000$ pooled and per-pair. Diff-of-means steering vectors transfer in three regimes by alignment with $R$'s invariant subspace: clean when pinned by shared output weights, partial when overlapping the rotated subspace, inverted otherwise. With no shared I/O (Pythia), all three collapse to universally inverted. The same rotation account holds across training checkpoints within a single run.
Validated on a 104k-parameter Dyck-3 transformer and nine independently-trained Pythia-70m seeds on The Pile, via a pre-registered four-bar operational framework. Frontier-scale (10B+) replication remains open. 

---
# Grammatically-Guided Sparse Attention for Efficient and Interpretable Transformers 

**Authors**: Spandan Pratyush  

**Link**: [PDF](https://arxiv.org/pdf/2605.24518)  

**Abstract**: The quadratic complexity of self-attention in Transformer models remains a significant bottleneck for processing long sequences and deploying large language models efficiently. For this approach, there has been significant research into Sparse Attention, and Deepseek Sparse Attention has combined various methods of creating segments of tokens to reduce the time complexity. This paper introduces a novel approach, Grammatically-Guided Sparse Attention, which constrains attention computations based on the grammatical roles of tokens. By leveraging Parts-of-Speech (POS) tags, attention masks are dynamically generated that enforce linguistically coherent connections between tokens, reducing the computational graph without sacrificing essential linguistic dependencies. Two masking strategies are proposed and evaluated: a hard mask that strictly allows only predefined grammatical interactions, and a soft mask that biases attention towards these interactions. The experiments, conducted on the SST-2 sentiment classification task using a DistilBERT-like architecture, demonstrate that Grammatically-Guided Sparse Attention maintains comparable accuracy to full attention while significantly reducing the theoretical computational overhead. Preliminary results show accuracy values of 0.8200 for hard masking and 0.8165 for soft masking, closely matching the 0.8200 of full attention, providing a path towards more efficient, interpretable, and linguistically-informed Transformer architectures. 

---
# Code2UML: Agentic LLMs with context engineering for scalable software visualization 

**Authors**: Alin-Gabriel Văduva, Anca-Ioana Andreescu, Simona-Vasilica Oprea, Adela Bâra  

**Link**: [PDF](https://arxiv.org/pdf/2605.24453)  

**Abstract**: Large Language Model (LLM)-based code analysis tools are adopted to automate software documentation tasks. However, the scalability of these approaches to real codebases, where Intermediate Representations (IR) exceed LLM context limits, remains underexplored. This paper introduces an agentic architecture with context engineering for automated UML diagram generation from source code repositories. It employs a hierarchy of five specialized agents: PlannerAgent, AnalyzerAgent, DiagramAgent, CorrectorAgent and DependencyAnalyzerAgent, built on the Claude Agent SDK, each addressing a distinct cognitive subtask. A deterministic, importance-weighted IR compaction layer transforms full project IRs into diagram-specific views guaranteed to fit within token constraints, requiring no LLM calls and completing in milliseconds. Thus, we evaluate the system across 12 open-source repositories in 4 programming languages (Java, JavaScript, PHP, Python) and 7 UML diagram types, producing 84 observations assessed on 5 automated metrics. Results demonstrate high syntactic validity (mean: 91.5%, with component and deployment diagrams reaching 100%), strong relationship precision (mean: 0.858) and consistent structural quality (mean: 81.7/100, with cross-language variance of 3.1 points). Entity recall averaged 0.313, reflecting deliberate architectural prioritization over exhaustive coverage. A sensitivity analysis (31 to 4,578 IR entities) confirms that quality scores remain stable regardless of scale. 

---
# ScaleAcross Explorer: Exploring Communication Optimization for Scale-Across AI Model Training 

**Authors**: Minghao Li, Alicia Golden, Samuel Hsia, Michael Kuchnik, Adi Gangidi, Xu Zhang, Ashmitha Jeevaraj Shetty, Zachary DeVito, Weiwei Chu, Dong He, Haoci Zhang, Yuchen Hao, Ruoming Pang, James Hongyi Zeng, Ying Zhang, Minlan Yu, Carole-Jean Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.24326)  

**Abstract**: The rapid scaling of large language model training requires distributing GPU resources across multiple data center buildings and regions. We refer to such paradigm as "scale-across" training. As infrastructure expands, the system design space becomes increasingly intricate, encompassing new model architectures, hardware heterogeneity, and evolving communication patterns. Drawing from Meta's production experience, we highlight the complexities of deploying training jobs across a few data centers housing hundreds of thousands of GPUs. To accelerate exploration of the large design space and to enable efficient training for frontier model development, we conduct in-depth characterization of three key design dimensions: parallelism placement, parallelism scheduling, and network layer technologies. We then propose ScaleAcross Explorer, an optimizer that considers the interplay of design dimensions and holistically optimizes scale-across training. Testbed experiments and simulations demonstrate up to 64.62% training speedups over production configuration and up to 37.59% training speedups over the state-of-the-art baseline across a wide range of design points. 

---
# Enhancing Reliability in LLM-Based Secure Code Generation 

**Authors**: Mohammed F. Kharma, Mohammad Alkhanafseh, Ahmed Sabbah, David Mohaisen  

**Link**: [PDF](https://arxiv.org/pdf/2605.24300)  

**Abstract**: Large language models (LLMs) are widely used for code generation, but their security reliability remains inconsistent across languages and prompting strategies. Existing prompt engineering improves functional correctness but rarely ensures consistent security outcomes. We introduce the \textit{Mitigation-Aware Chain-of-Thought (MA-CoT)} framework, which embeds task-specific CWE mitigation guidance and language-aware safeguards to reduce recurring vulnerabilities in generated code. We evaluate MA-CoT across three LLMs (gpt-5, claude-4.5, gemini-2.5), three programming languages (C, Java, Python), and four prompting strategies (Vanilla, Zero-shot, CoT, MA-CoT) on a 200-task primary dataset, with external validation on LLMSecEval. Using static analysis with expert validation, MA-CoT reduces total security findings from 92 to 39 (57.6\%) on the primary dataset and from 73 to 4 (94.5\%) on LLMSecEval. High-severity findings (Blocker + Critical) drop from 90 to 39 (56.7\%) and from 45 to 2 (95.6\%), respectively. Across both datasets, MA-CoT is the only strategy that consistently improves security reliability; Zero-shot and CoT are less reliable and may increase vulnerability, especially in C. We further introduce a strict layered attribution of vulnerability drivers (language-core vs. stack layers) and show that residual risk concentrates in hardening-oriented patterns (e.g., OS- and toolchain-dependent), motivating secure-by-construction primitives alongside prompting. 

---
# ChaosBench-Logic v2: Evaluating LLM Logical Reasoning over Dynamical Systems at Scale 

**Authors**: Noel Thomas  

**Link**: [PDF](https://arxiv.org/pdf/2605.24305)  

**Abstract**: Standard accuracy on binary reasoning benchmarks hides critical failure modes: prior collapse, inconsistency under paraphrase, and inability to reason about parameter-dependent dynamics. We present ChaosBench-Logic v2, a 40,886-question benchmark over 165 dynamical systems with 27 FOL predicates and 78 axiom edges, together with CARE (Calibration- and Adversarial-Robust Evaluation), a protocol that surfaces these pathologies. Evaluating 14 models, we find that regime-transition reasoning remains near random (MCC = 0.05) even for frontier models, whereas FOL deduction with given premises reaches MCC = 0.52. Per-family decomposition shows that the proprietary-model advantage concentrates on cross-indicator (+0.40) and consistency tasks, while open-source Qwen 2.5-32B dominates indicator diagnostics (0.91 vs. 0.45). Two models exhibit negative MCC on bifurcation questions, confirmed as systematic anti-correlation via confusion-matrix analysis. 

---
# An Empirical Evaluation of LLM-Generated Code Security Across Prompting Methods 

**Authors**: Mohammed Kharma, Ahmed Sabbah, Mohammad Alkhanafseh, Mohammad Hammoudeh, David Mohaisen  

**Link**: [PDF](https://arxiv.org/pdf/2605.24298)  

**Abstract**: The growing use of Large Language Models (LLMs) for automated code generation has enhanced software development efficiency, but often at the cost of security. Generated code frequently overlooks critical concerns, leaving it vulnerable to issues such as weak encryption and improper input validation. To investigate this problem, we present a comprehensive empirical evaluation of the security quality of LLM-generated code across five LLMs and four programming languages (Java, C++, C, and Python), examining the impact of multiple prompt engineering methods. We introduce a weaknesses-aware zero-shot chain-of-thought (WA-0CoT) prompting strategy that enriches prompts with security context using CWE mappings to guide model reasoning. Our empirical analysis, supported by chi-square tests, finds no statistically significant reductions in vulnerability frequency or density across prompt methods. However, prompting strategies, including WA-0CoT, systematically influence the compositional distribution of CWE categories, with effects varying by programming language. These findings suggest that while security-aware prompting alters the structure of generated weaknesses, prompt engineering alone is insufficient to reliably reduce overall vulnerability levels. The results highlight the importance of language-aware and model-aware prompt design when evaluating the security properties of LLM-generated code. 

---
# An Interactive Paradigm for Deep Research 

**Authors**: Lin Ai, Victor S. Bursztyn, Xiang Chen, Julia Hirschberg, Saayan Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2605.24266)  

**Abstract**: Recent advances in large language models (LLMs) have enabled deep research systems that synthesize comprehensive, report-style answers to open-ended queries by combining retrieval, reasoning, and generation. Yet most frameworks rely on rigid workflows with one-shot scoping and long autonomous runs, offering little room for course correction if user intent shifts mid-process. We present SteER, a framework for Steerable deEp Research that introduces interpretable, mid-process control into long-horizon research workflows. At each decision point, SteER uses a cost-benefit formulation to determine whether to pause for user input or to proceed autonomously. It combines diversity-aware planning with utility signals that reward alignment, novelty, and coverage, and maintains a live persona model that evolves throughout the session. SteER outperforms state-of-the-art open-source and proprietary baselines by up to 22.80\% on alignment, leads on quality metrics such as breadth and balance, and is preferred by human readers in 85\%+ of pairwise alignment judgments. We also introduce a persona-query benchmark and data-generation pipeline. To our knowledge, this is the first work to advance deep research with an interactive, interpretable control paradigm, paving the way for controllable, user-aligned agents in long-form tasks. 

---
# Momentum Streams for Optimizer-Inspired Transformers 

**Authors**: Jingchu Gai, Nai-Chieh Huang, Jiayun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.24425)  

**Abstract**: The residual update of a pre-norm Transformer layer admits an interpretation as one step of a first-order optimizer acting on a surrogate token energy, wherein the attention and MLP sublayers function as gradient oracles. Based on this observation, we build a family of optimizer-inspired Transformers (triple-momentum, Adam/AdamW, Muon, SOAP) and compare them under matched compute. In our main pretraining experiment, the triple-momentum TMMFormer achieves the lowest validation loss, outperforming the vanilla Transformer and prior architectural variants. A controlled ablation and supporting theory show that momentum, not preconditioning, is the main source of the gain. We further show that TMMFormer and other momentum-based designs reach flatter minima than the vanilla Transformer, which leads to less forgetting and better generalization. 

---
# Attested Tool-Server Admission: A Security Extension to the Model Context Protocol 

**Authors**: Alfredo Metere  

**Link**: [PDF](https://arxiv.org/pdf/2605.24248)  

**Abstract**: The Model Context Protocol (MCP) standardizes how a large-language-model (LLM) agent and an external tool server exchange messages, but not trust: a host reads a server's self-declared tool list and dispatches calls, with no notion of which servers it may use, at what sensitivity, or which of a server's tools are in bounds. This work grew out of a concrete need -- letting the Enclawed agent use Google's externally-operated MCP servers (Gmail, Calendar, Drive) safely, admitting the server and bounding the tools it may drive, without changing MCP or Enclawed's own tool application-programming interface (API). The mechanism we built, mcp-attested (shipped in both the open enclawed-oss distribution and the enclaved flavor), generalizes: the gap that makes an unmediated third-party connection unsafe for one user makes a regulated deployment impossible to accredit. We close it with three additive mechanisms: (1) a small, offline-signed clearance assertion a server publishes at a well-known Uniform Resource Identifier (URI) and a host verifies against a pinned trust root before any tool dispatch; (2) a deny-by-default per-server tool allowlist, so admitting a server is not trusting its every tool; and (3) a flavor-gated enforcement mode that turns the checks from warnings into hard denials, with every decision written to a tamper-evident audit log. We give the wire format, the verification algorithm, a security analysis, and an LLM-driven adversarial evaluation; we then state the design in normative Request-for-Comments (RFC 2119) form -- schema, verification rules, error registry, well-known registration, and machine-checkable conformance vectors -- so it can be adopted as an MCP addendum rather than reinvented. An unextended host ignores the well-known document and behaves exactly as today. 

---
# Improving Labeling Consistency with Detailed Constitutional Definitions and AI-Driven Evaluation 

**Authors**: Konstantin Berlin, Adam Swanda  

**Link**: [PDF](https://arxiv.org/pdf/2605.24247)  

**Abstract**: Many automated labeling pipelines classify inputs into categories defined by a written specification, content moderation being a prominent use case. Simple category definitions are not detailed enough for labelers to produce the accurate, consistent golden labels these pipelines require. One solution is to write a prescriptive definition that settles enough real boundary cases that labelers cannot disagree with the written interpretation. In practice, definitions at that level of detail exceed what a human annotator can hold in working memory, so annotators fall back on intuition and the labels drift from the written rules, regressing on accuracy and consistency.
We propose and demonstrate the efficacy of an AI-driven workflow in which AI helps write a per-category constitution that defines the label in enough detail to cover edge cases, and a frontier LLM interprets it on each input to produce the golden label more consistently and accurately than humans reading the same document. We evaluate on three content moderation categories (harassment, hate speech, non-violent crime) and show that the approach reduces cross-model inconsistency by up to 57x compared to paragraph definitions, with cross-model disagreement diagnosing specification gaps and the human responsible for high-level decisions about what each category should mean rather than individual labeling calls. For the safety evaluation, we introduce a dual-axis formulation scoring intent and content independently over the full conversation, so downstream consumers can act on either axis or both. 

---
# Agent-ToM: Learning to Monitor Autonomous LLM Agents via Theory-of-Mind Reasoning 

**Authors**: Nesreen K. Ahmed, Nima Nafisi  

**Link**: [PDF](https://arxiv.org/pdf/2605.24216)  

**Abstract**: Monitoring autonomous large language model (LLM) agents for covert malicious behavior is challenging due to delayed, context-dependent, and long-horizon attack patterns. Agents may pursue hidden objectives while maintaining superficially benign behavior, making detection difficult even with full trajectory access. Prior monitoring approaches improve scaffolding or ensemble aggregation, but treat each trajectory independently and do not learn from prior monitoring experience. Moreover, standard reasoning methods explain observed behavior without explicitly reasoning about agent beliefs, intentions, and goal alignment required to distinguish benign task execution from covert deviation.
We propose \textbf{Agent-ToM}, a learning-to-monitor framework grounded in Theory-of-Mind (ToM) reasoning for security analysis of autonomous agents. Agent-ToM performs structured full-trajectory analysis by inferring beliefs, intent hypotheses with calibrated confidence, expected actions, and deviations from task-consistent behavioral baselines. At inference time, it employs a \textit{Reason-Verify-Refine} pipeline to construct and validate monitoring decisions. At training time, Agent-ToM distills critique signals into a persistent \textit{semantic guardrail memory}, enabling reusable belief- and intent-conditioned constraints across episodes. We evaluate Agent-ToM on adversarial agent monitoring benchmarks (SHADE-Arena and CUA-SHADE-Arena). Agent-ToM achieves strong precision-recall balance and outperforms state-of-the-art monitoring baselines, including ensemble methods, while using a coherent two-call reasoning pipeline. These results demonstrate that learning at the monitoring layer, combined with structured ToM reasoning and verification, provides an effective and deployable foundation for securing autonomous LLM agents. 

---
# Teaching Through Analogies: A Modular Pipeline for Educational Analogy Generation 

**Authors**: Mariam Barakat, Ekaterina Kochmar  

**Link**: [PDF](https://arxiv.org/pdf/2605.24211)  

**Abstract**: Analogies help learners understand unfamiliar concepts by relating them to known concepts. Despite recent advances, large language models (LLMs) continue to struggle to generate analogies of comparable quality to those produced by humans. We present a modular pipeline for educational analogy generation, decomposing the task into four stages: source finding, sub-concept generation, explanation generation, and evaluation. Grounded in Structure Mapping Theory, the pipeline enables systematic, stage-by-stage analysis of how model choice and input configuration affect analogy quality. We evaluate 12 state-of-the-art LLMs across six model families on two datasets with structured sub-concept annotations (SCAR and ParallelPARC), alongside seven embedding models for closed-setting retrieval. Our results show that sub-concepts substantially improve explanation quality and closed setting retrieval precision but provide limited benefit in open-ended source generation. We further introduce an LLM-as-a-judge evaluation methodology and validate its scoring against human annotations from seven annotators, finding that Claude Sonnet 4.6 aligns more reliably with human rankings than with fine-grained absolute scores. Taken together, our findings reveal cross-stage interactions that isolated studies cannot capture, and highlight sub-concept grounding as a key driver of analogy quality generation. 

---
# Human-AI Collaboration in Science at Scale: A Global Large-scale Randomized Field Experiment 

**Authors**: Binglu Wang, Weixin Liang, Jiahui Xue, Yuhui Zhang, Hancheng Cao, Dashun Wang, Yian Yin  

**Link**: [PDF](https://arxiv.org/pdf/2605.24180)  

**Abstract**: Collaboration is the defining mode of modern science, yet its core mechanism -- feedback -- remains hard to observe, difficult to scale, and unequally distributed. Here we test whether large language models (LLMs) can contribute to this hidden but vital practice and reallocate scientific feedback, an essential yet scarce resource for knowledge production. In a global large-scale randomized field experiment, we delivered customized LLM-generated feedback for over 31,000 arXiv preprints across 150 fields and more than 45,000 researchers from 133 geographic regions. Relative to controls, authors who received feedback had a significantly higher likelihood of revising their manuscripts, corresponding to a 12.55% relative increase over the baseline revision rate. Exposure to AI feedback also increased authors' subsequent use of LLM tools in their future papers, suggesting longer-run shifts in scientific practice. These effects were strongest among authors from non-English-dominant research regions, manuscripts less embedded in the scholarly literature, and teams with lower h-indexes and earlier career stages, consistent with the idea that AI feedback may provide the greatest benefit where access to timely critique is otherwise limited. Together, these findings provide causal evidence that structured AI-based interventions can transform access to scientific feedback from a largely private advantage into a more widely distributed resource, with broader implications for productivity, equity, and capacity across the global research system. 

---
# PromptAudit: Auditing Prompt Sensitivity in LLM-Based Vulnerability Detection 

**Authors**: Steffen J. Camarato, Yahya Hmaiti, Mandana Ghadamian, David Mohaisen  

**Link**: [PDF](https://arxiv.org/pdf/2605.24171)  

**Abstract**: Large language models are increasingly used for vulnerability detection, yet their reliability under different prompt formulations remains uncharacterized. We present PromptAudit, a controlled evaluation framework that isolates prompt effects by fixing the dataset, decoding, and parsing while varying only the prompting strategy. Using five prompting strategies across five open-weight models on 1,000 CVEs (6,074 code samples spanning 16 programming languages), we evaluate accuracy, recall, abstention, coverage, and effective F1. We find that standard chain-of-thought prompting achieves the strongest overall operational performance, while few-shot prompting provides model-dependent benefits that are most pronounced for prompt-sensitive models. In contrast, adaptive chain-of-thought frequently suppresses recall and self-consistency induces excessive abstention, sharply reducing effective performance. These results show that vulnerability detection behavior is jointly determined by the model and the prompt, and that prompt sensitivity is a first-class system property that must be explicitly characterized in evaluation and deployment. 

---
# Side-by-side Comparison Amplifies Dialect Bias in Language Models 

**Authors**: Kritee Kondapally, Claire J. Smerdon, Pooja C. Patel, Ogheneyoma Akoni, Jevon Torres, Jaspreet Ranjit, Matthew Finlayson, Swabha Swayamdipta  

**Link**: [PDF](https://arxiv.org/pdf/2605.24384)  

**Abstract**: Language models (LMs) can exhibit systematic biases against speakers based on variations in their dialects, even in the absence of a dialect label, a behavior known as covert dialect bias. In this work, we quantify covert dialect bias in online discourse by evaluating how LMs associate stereotypical traits (derived from social psychology research on racial bias) with intent-equivalent tweets in Standard American English (SAE) and African-American Vernacular English (AAVE). While prior work shows that LMs associate more negative stereotypes with AAVE when evaluating tweets in isolation, we are surprised to find that this bias is significantly exacerbated when SAE / AAVE tweet pairs are compared side by side, a setting that more closely reflects high-impact decision making contexts in which models are used to rank candidates. The bias only worsens when dialect labels are explicitly specified. This is striking, given the extensive efforts from commercial developers to mitigate bias in their LMs. Encouragingly, we show that counterfactual fairness finetuning can mitigate covert dialect bias for some stereotypical traits, reducing average disparities when evaluating tweets in isolation, however, these improvements do not consistently hold across traits when evaluating SAE / AAVE tweets side by side. Our findings show that existing evaluation settings for covert dialect bias may underestimate its severity, specifically in contrastive settings. Additionally, overt dialect bias remains pronounced even after safety aligned finetuning, indicating that it remains an unresolved problem, and motivates the need for more robust evaluation and mitigation frameworks. 

---
# Extracting Training Data from Diffusion Language Models via Infilling 

**Authors**: Yihan Wang, N. Asokan  

**Link**: [PDF](https://arxiv.org/pdf/2605.24173)  

**Abstract**: Memorization in large language models has been studied almost exclusively through prefix-conditioned extraction, a natural choice for autoregressive models. However, diffusion language models (DLMs) can denoise masked tokens at arbitrary positions. Thus, prefix-only probing reveals only one facet of memorization in DLMs and significantly underestimates the risk of training-data extraction. In order to realistically model extractability of training data in DLMs, we introduce \emph{infilling extraction}, a data-extraction protocol parameterized by an arbitrary binary mask that subsumes prefix-only probing and accounts for the bidirectional inductive bias of DLMs. Instantiating it on LLaDA-8B and Dream-7B across five extraction modes, three training pipelines, and three corpora covering verbatim and partial leakage, we find that mask geometry governs extractability: edge-conditioned masks \emph{extract up to three times more} verbatim sequences than prefix-conditioned ones, and bidirectional access opens channels inaccessible in autoregressive models. In particular, we show that a realistic adversary with access to training data where personally identifiable information has been redacted, can even achieve higher recall on extracting redacted email addresses from DLMs than from scale-matched autoregressive models. Tunable parameters for decoding measurably affect extraction performance, while a follow-up supervised finetuning stage does not eliminate the prior memorization. 

---
# The Time is Here for Just-in-Time Systems: Challenges and Opportunities 

**Authors**: Shu Liu, Alexander Krentsel, Shubham Agarwal, Mert Cemri, Ziming Mao, Soujanya Ponnapalli, Alexandros G. Dimakis, Sylvia Ratnasamy, Matei Zaharia, Aditya Parameswaran, Ion Stoica  

**Link**: [PDF](https://arxiv.org/pdf/2605.24096)  

**Abstract**: Core systems like key-value stores have historically taken years to build, and are designed to be general so as to amortize cost across deployments, paying a significant performance cost. We argue that LLM-based coding agents now make a different approach tractable: Just-in-Time Systems, in which the entire system is synthesized from scratch, specialized to the environment, workload, and required system properties. We present a JIT system synthesis pipeline, Jitskit, and explore its effectiveness in synthesizing key-value stores from spec cards that span different YCSB workloads, deployment constraints (e.g., compute resources), and system properties (e.g., consistency and durability). Jitskit iteratively refines a system implementation to match the specification against an evolving evaluation test suite. The resulting synthesized systems are performant, beating comparable state-of-the-art systems on 18 of 18 specs tried, by up to 4.6x over the best off-the-shelf baseline on the most favorable spec. Naively running Claude Code either reward-hacks or underperforms Jitskit by up to 5.4x. We discuss the challenges we overcame in building Jitskit and our key takeaways. 

---
# TRACER: A Semantic-Aware Framework for Fine-Grained Contamination Detection in Code LLMs 

**Authors**: Yifeng Di, Xuliang Huang, Tianyi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.24079)  

**Abstract**: Data contamination is a known threat to the reliability of model evaluation. However, it remains underexplored in code large language models (LLMs), where contamination often goes beyond exact duplication. We present TRACER, a semantic-aware framework for fine-grained code contamination detection. TRACER models contamination using three levels of semantic overlap - Functionally Identical, Nearly Identical, and Shared Logic - and detects them through a coarse-to-fine pipeline. We also introduce the first benchmark for fine-grained code contamination detection, spanning three widely used benchmarks and three representative post-training datasets. TRACER achieves strong and consistent performance across multiple LLM backbones, with GPT-5 reaching an F1 score of 0.91 in fine-grained detection. In the binary setting, TRACER attains an F1 of 0.92, outperforming existing methods by 42%-217%. We further conduct ablation studies and error analysis to assess the contributions of individual components in TRACER. 

---
# Empirical Analysis and Detection of Hallucinations in LLM-Generated Bug Report Summaries 

**Authors**: Hinduja Nirujan, Shreyas Patil, Abdallah Ayoub, Ahmad Abdel Latif, Gouri Ginde  

**Link**: [PDF](https://arxiv.org/pdf/2605.24137)  

**Abstract**: Large Language Models (LLMs) are increasingly used to generate summaries of software bug reports, including sections such as Steps-to-Reproduce (S2R), Actual Behavior (AB), and Expected Behavior (EB). However, these models frequently produce hallucinations that can be convincing but unsupported by the source report. This can mislead developers and reduce trust in automated maintenance tools. Existing hallucination detection approaches typically evaluate outputs at the full-response level and do not consider the structure of technical documents. An initial exploratory study on 80 structured bug report summaries found that approximately 47.9% contained missing information, while 12.3% included fabricated content, highlighting the need for systematic hallucination analysis in bug report summarization. In this work, we empirically investigate hallucinations in LLM-generated bug report summaries from a section-aware perspective. Using the BugsRepo dataset, derived from Mozilla OSS projects, we introduce controlled synthetic hallucination injection to construct a benchmark for training and evaluation. We propose a section-aware hallucination detection approach that jointly predicts whether a summary contains hallucinated content, identifies affected sections, and classifies hallucination types. Experimental results across multiple pretrained language models show that the proposed approach achieves strong performance across all tasks, with the best model obtaining 0.89 report-level Macro-F1, 0.83 section-level Macro-F1, and 0.84 hallucination-type Macro-F1. We further analyze common hallucination patterns and model failure modes to better understand limitations of current LLM-generated bug report summaries. The findings highlight the importance of section-aware hallucination analysis for improving the reliability of LLM-assisted bug report summarization in software maintenance workflows. 

---
# Understanding Conversational Patterns in Multi-agent Programming: A Case Study on Fibonacci Game Development 

**Authors**: Srijita Basu, Viktor Kjellberg, Simin Sun, Bengt Haraldsson, Md. Abu Ahammed Babu, Wilhelm Meding, Farnaz Fotrousi, Miroslaw Staron  

**Link**: [PDF](https://arxiv.org/pdf/2605.24138)  

**Abstract**: Large Language Models (LLMs) are increasingly applied to software engineering (SE), yet their potential for autonomous, role-oriented collaboration remains largely underexplored. Understanding how multiple LLM-based agents coordinate, maintain role alignment, and converge on solutions is critical for SE, as naively allowing agents to interact does not reliably lead to correct or stable outcomes. Recent empirical studies show that unstructured or poorly understood interaction dynamics can result in error propagation, premature consensus on incorrect solutions, or prolonged disagreement that prevents convergence, even when correct partial solutions are present early in the interaction. As an initial step towards addressing this underexplored area, we undertake a systematic analysis of conversations between two agents, a Designer and a Programmer across 12 model combinations from 7 open-source LLMs (Gemma 2, Gemma 3, LLaMA 3.2, LLaMA 3.3, DeepSeek-R1, MiniCPM, and Qwen3). Our systematic approach reveals three key dimensions of multi-agent interaction: efficiency (the speed and stability of convergence), consistency (the degree of role alignment visualized by BLEU and ROUGE), and effectiveness (the extent of compilation success and error resolution). Results show that the DeepSeek-R1:DeepSeek-R1 pair was unique in converging to the correct solution from the very first iteration and sustaining it consistently to the final iteration, while LLaMA 3.2:LLaMA 3.2 and Qwen3:Qwen3 demonstrated strong Designer:Programmer role alignment despite of diverging from the correct solution. The other pairs deviated from the task, never to converge to a result. These findings advance understanding of agentic programming and highlight the need for further research on understanding and calibrating convergence and stop conditions essential for future autonomous SE. 

---
# Feature Lottery? A Bifurcation Theory of Concept Emergence 

**Authors**: Fuming Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.24057)  

**Abstract**: Neural networks acquire structured representations at specific moments during training, yet identifying these transitions typically relies on retrospective, label-dependent metrics. We introduce a bifurcation theory of representation dynamics to detect these moments in real time. Analyzing a passive GMM probe attached to the evolving encoder, we show the onset of structure corresponds to a supercritical pitchfork bifurcation driven by the loss Hessian. The system exhibits a theoretically predictable zero-crossing ($\beta_c$) that, compared to the network's current state ($\beta$), yields a dynamic ratio $\beta(t)/\beta_c(t)$: a universal, label-free phase coordinate for representation dynamics, computable entirely from hidden states. We empirically validate four distinct transition regimes predicted by this coordinate across diverse settings: SAEs on language models (Pythia), SSL (CIFAR), and grokking (modular arithmetic). Crucially, under finite dissipation, macroscopic symmetry-breaking can lag the initial zero-crossing by orders of magnitude, which providing a rigorous dynamical account of the delayed escape observed in grokking. Microscopically, the bifurcation creates a shared unstable subspace, forcing collective symmetry breaking. We term this the "feature lottery" in SAE training: a feature's terminal interpretability becomes predictable remarkably early. By only 5% of training, early atom purity robustly predicts final convergence purity, with top-decile early atoms achieving over 12x the baseline purity at convergence. Beyond explaining concept emergence, $\beta/\beta_c$ provides a practical early-warning indicator for training health, detecting the onset of usable structure, the crystallization of feature identity, and representational collapse epochs before downstream metrics react. 

---
# Signs Beat Floats: Low-Rank Double-Binary Adaptation for On-Device Fine-Tuning 

**Authors**: Yoshihiko Fujisawa, Yuma Ichikawa, Yudai Fujimoto, Akira Sakai, Katsuki Fujisawa  

**Link**: [PDF](https://arxiv.org/pdf/2605.24058)  

**Abstract**: On-device adaptation of large language models commonly keeps a quantized base model frozen while training and deploying a small, task-specific LoRA adapter. In the unmerged adapter-mode setting, however, the adapter is more than a compact storage module; it introduces an additional dense floating-point branch, maintains a trainable state for local updates, and acts as a unit of communication and this http URL introduce LoRDBA, a LoRA-compatible adapter that replaces both low-rank factors with binary sign carriers while representing magnitudes through lightweight, channel-wise scales, converting the dense adapter branch into two sign-accumulation matrix multiplications interleaved with channel-wise scaling. A finite-sample analysis shows that reconstruction quality is governed by the residual-to-magnitude ratio of the original LoRA factors. In adapter-mode experiments, LoRDBA outperforms low-bit baselines at matched model sizes while matching fp16 LoRA quality in selected regimes. The unmerged adapter incurs at most 8% prefill latency overhead at matched rank r=16 despite an over 10x reduction in adapter footprint, with moderate training memory overhead of approximately 1.6x that of fp16 LoRA. 

---
# Truthful Online Preference Aggregation for LLM Fine-Tuning in Mobile Crowdsourcing 

**Authors**: Shugang Hao, Lingjie Duan  

**Link**: [PDF](https://arxiv.org/pdf/2605.24052)  

**Abstract**: To better serve users' demands in mobile applications (e.g., navigation), mobile crowdsourcing platforms can iteratively align large language model (LLM)-generated content (e.g., AI-generated traffic condition predictions) with human feedback collected from crowdsourcing workers (e.g., mobile users). However, workers may strategically misreport their online preference feedback to maximize their influence or payment. Existing pipelines in mobile crowdsourcing (e.g., EM-based weight estimation) fail to identify the most accurate worker in this online setting, resulting in a linear regret $\mathcal{O}(T)$ over $T$ time slots. In this paper, we study truthful online preference aggregation for LLM fine-tuning in mobile crowdsourcing. We formulate a new dynamic Bayesian game to model the multi-agent online learning process between the platform and strategic mobile workers. We propose a novel online weighted aggregation mechanism that dynamically adjusts each worker's weight in the preference aggregation according to their feedback accuracy. We prove that our mechanism ensures truthful feedback from strategic workers and achieves a sublinear regret $\mathcal{O}(\sqrt{T})$ over $T$ time slots. We further extend our mechanism to a challenging scenario with limited worker feedback per time slot, still guaranteeing a sublinear regret $\mathcal{O}(\sqrt{T})$. Experiments on LLM fine-tuning with real-world datasets further demonstrate significant performance gains of our mechanisms over benchmark schemes. 

---
# Spectral Probe-Circuits: A Three-Step Recipe for Identifying Attention-Head Circuits in Pretrained Transformers 

**Authors**: Yongzhong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.24059)  

**Abstract**: We present a three-step recipe for identifying attention-head circuits in pretrained transformers. A per-head spectral signal -- the time-integrated participation ratio of each head's attention output -- ranks heads doing sustained content-dependent computation without labels or attribution gradients. A task-pattern screen filters this general indicator into a task-specific candidate circuit, and group ablation against a matched-random control completes the causal claim. We validate across an 8x parameter range (51M to 1B-active / 7B-total), two architecture families (dense, mixture-of-experts), and four pretraining pipelines. The recipe ports: a 2-6 head induction circuit is causally necessary in every model tested, with a 94-100% drop in synthetic-induction top-1 after ablation. The spectral signal is predictive without supervision: on six independent seeds of a 51M-parameter probe model, the same computation identifies the seed-specific circuit on each seed. The fraction of heads doing identifiable specialized computation is conserved at 17-19% across the Pythia family (124M to 410M), while specific induction circuits stay 3-11 heads -- sublinear in total head count. This paper is the methodology anchor of a three-paper program; companion papers extend the recipe to developmental trajectories during pretraining and to composed-task circuits where pattern selectivity decouples from task-causal structure. 

---
# When the Manual Lies: A Realistic Benchmark to Evaluate MCP Poisoning Attacks for LLM Agents 

**Authors**: Shi Liu, Xuehai Tang, Xikang Yang, Liang Lin, Biyu Zhou, Wenjie Xiao, Wantao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.24069)  

**Abstract**: The rise of tool-using Large Language Model (LLM) agents, standardized by protocols like the Model Context Protocol (MCP), has unlocked unprecedented autonomous execution capabilities for LLM Agents by integrating external open-domain knowledge and tools. However, this interoperability introduces a covert attack surface targeting the agent's cognitive planning layer. This paper systematically investigates Tool Description Poisoning (TDP), a novel semantic attack. In TDP, malicious instructions are not embedded in a tool's executable code, but rather covertly injected into its descriptive metadata, the very "manual" an agent relies on for secure planning and decision-making. To rigorously and systematically evaluate this emerging threat, we introduce the MCP-TDP Security Benchmark. This high-fidelity sandbox environment comprises 32 realistic, real-world test cases spanning 6 distinct risk categories. Our evaluation of 8 mainstream LLMs reveals severe vulnerabilities, with leading models like GPT-4o exhibiting a nearly 100% Attack Success Rate (ASR) in six high-risk scenarios. Furthermore, our findings demonstrate that common prompt-guardrail defenses are largely ineffective and can, counterintuitively, even be counterproductive (a phenomenon which we term the "Firewall Fallacy"). Crucially, we also propose a defense mechanism: "Reactive Self-Correction," where an agent autonomously detects and reverts its own malicious actions post-execution. This work provides the first specialized security benchmark tailored for TDP, offering essential insights for securing the cognitive and planning layers of advanced agentic systems. 

---
# More Skills, Worse Agents? Skill Shadowing Degrades Performance When Expanding Skill Libraries 

**Authors**: Hongwen Song, Song  

**Link**: [PDF](https://arxiv.org/pdf/2605.24050)  

**Abstract**: Skill libraries allow LLM agents to load task-specific instructions on demand, letting non-expert users solve domain-specific tasks through natural language without knowing which skills exist or how they work. However, performance degrades as libraries grow -- by up to 21\% when scaling from a small set of helpful skills to a 202-skill library. In this work, we formulate this performance degradation as the pass rate drop between loading a library of known-helpful skills and the full library. Moreover, we propose to decompose the pass rate drop by conditioning on the skill(s) invocation -- which skills the agent selects during a trajectory -- into two effects: \emph{skill shadowing}, where the agent selects wrong skills more often as the library expands, and \emph{context overhead}, where the enlarged context degrades execution even when selection is correct. We derive upper bounds on both effects to characterize their magnitudes of impacts to the pass rate drop. Our empirical estimates of the effects and their upper bounds both show that the \emph{skill shadowing} effect grows with library size and significantly contributes to the performance degradation, whereas the \emph{context overhead} effect remains small and indistinguishable from zero. This observed asymmetry establishes that the skill selection failure, not the enlarged context, is the primary bottleneck when expanding the skill libraries. 

---
# Unlocking Apple's Private Cloud Compute: An Analysis of Privacy-Preserving Artificial Intelligence 

**Authors**: Yannik Dittmar, Marvin Jerome Stephan, Thomas Völkl, Matthias Hollick, Jiska Classen  

**Link**: [PDF](https://arxiv.org/pdf/2605.24239)  

**Abstract**: Many existing Artificial Intelligence (AI) solutions on mobile devices rely on an extensive collection of sensitive data, raising privacy concerns and often requiring storage for both context and model improvement. Apple's Private Cloud Compute (PCC) aims to address this by emphasizing mobile device integration and a privacy-first design. The central claim of PCC is that it does not store any user data and that user input and user accounts are unlinkable.
While most of the PCC system specifications are public, compiled binaries add a layer of opaqueness. There are no reproducible builds, and there are no symbols within those binaries, creating potential discrepancies between the specification and what is shipped to the user. Additionally, the underlying models and interfaces for querying PCC are not openly accessible, limiting academic evaluation of model properties, such as accuracy. This poses a challenge in assessing whether a privacy-preserving approach like PCC is actually trustworthy while also providing high-quality answers.
We are the first to reverse-engineer the PCC implementation on mobile devices to evaluate privacy aspects and to open its non-public interfaces on local devices to support custom PCC queries. We demonstrate this level of access beyond Apple's intended use cases by independently benchmarking the PCC model. We enable future research by making our PCC benchmarking framework publicly available. 

---
# Mixture of Complementary Agents for Robust LLM Ensemble 

**Authors**: Yichi Zhang, Kevin Lu, Yuang Zhang, Jie Gao, Lirong Xia, Fang-Yi Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.24048)  

**Abstract**: Multi-AI collaboration, such as ensembling or debating large language models (LLMs), is a promising paradigm for aggregating information and boosting performance. A foundational step in these pipelines is to feed the responses of several proposer LLMs into a summarizer LLM, which synthesizes a better answer. However, choosing which proposers to include is non-trivial. Existing approaches primarily focus either on accuracy (picking the strongest models) or diversity (ensuring variety), and often overlook the interactions among proposers and with the summarizer. We reframe proposer selection as a combinatorial selection problem akin to feature selection, where the value of an LLM lies in its complementarity with others. However, directly applying standard feature-selection algorithms is impractical in the LLM setting due to prohibitive time complexity. Motivated by this limitation, we explore an extensive range of computationally feasible, greedy-style selection algorithms that assess complementarity using a small labeled set. Our experiments validate complementarity as a guiding principle for proposer selection and identify methods that achieve the best performance-cost trade-offs in practice. 

---
# Hidden-State Privacy Has an Empty Middle 

**Authors**: Alexander Okezue Bell  

**Link**: [PDF](https://arxiv.org/pdf/2605.24042)  

**Abstract**: Of $1{,}536$ Gaussian release covariances we tested for single-layer hidden-state privacy, zero achieve both moderate utility and moderate privacy against an adaptive retrieval attacker. We prove a complementary Fisher-ball lower bound: every full-rank Gaussian release at $O(1)$ Fisher utility admits a direction whose Mahalanobis signal grows linearly in hidden width, ruling out uniform Gaussian safety in the class and matching the empirical empty middle. The diagonal inverse-Fisher release $\Sigma^\star_{\mathrm{diag}}(\mathcal{K}) = (2\mathcal{K}/d)\,\mathrm{diag}(1/F_{ii})$ is the unique minimax-optimal diagonal mechanism at first-order KL budget $\mathcal{K}$ and the only release with worst-attacker top-1 $\le 0.001$ at every point of a 32 model-layer grid, but it sits on a privacy/utility edge rather than filling the middle. A generalized-eigen mechanism reaching $13\times$ Pareto reduction under Euclidean retrieval collapses to $100\%$ top-1 under the adaptive Mahalanobis attacker, and a full-trajectory sequence inverter recovers $94\%$ of clean GPT-2 prefixes but $0\%$ under $\Sigma_{\mathrm{diag}}$. A split-memory transformer trained from scratch reaches $G_{\mathrm{Mah}} \in [20, 33]$ at 90M and maintains a $6$--$24\times$ advantage over same-budget GPT baselines from 30M to 1B at a fixed-token language-modeling loss penalty; pretrained models top out at 9.3. These results reframe hidden-state release from mechanism-design within the Gaussian class to architecture or release co-design. 

---
# LLM-AutoSciLab: Closed-Loop Scientific Discovery via Active Experimentation with LLMs 

**Authors**: Sanchit Kabra, Nikhil Abhyankar, Saaketh Desai, Prasad Iyer, Chandan K Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2605.24043)  

**Abstract**: Scientific discovery is a closed-loop process in which hypotheses guide data acquisition and observations refine the hypothesis space. Yet most approaches reduce discovery to supervised learning over fixed datasets, where limited observations can support multiple plausible mechanisms that fit locally but fail to generalize. Thus, the key challenge is selecting informative observations to resolve uncertainty, shifting the focus from static inference to adaptive data acquisition. To address this, we propose LLM-AutoSciLab, a closed-loop framework that couples hypothesis generation with hypothesis-conditioned experiment selection and mechanism refinement. Rather than fitting models to passively collected data, LLM-AutoSciLab iteratively proposes plausible hypotheses, selects informative experiments to distinguish or refine them, and updates its state using the resulting evidence. To evaluate dynamic, closed-loop scientific discovery with active data acquisition, we introduce ActiveSciBench, comprising two datasets: ActiveSciBench-Chem with 57 enzyme-kinetics tasks and ActiveSciBench-GRN with 45 gene-regulatory-network tasks. These datasets model discovery as a budget-constrained process requiring adaptive experiment design, variable selection, and recovery of true mechanisms. Across NewtonBench, ActiveSciBench-Chem, and ActiveSciBench-GRN, LLM-AutoSciLab outperforms prior methods, achieving 67.6% and 35.1% symbolic accuracy on NewtonBench and ActiveSciBench-Chem, respectively, and 31.1% exact graph recovery on ActiveSciBench-GRN. Moreover, hypothesis-guided experimentation is 2-5x more sample-efficient than the strongest competing baselines. Code and data are available at: this https URL 

---
# MemForest: An Efficient Agent Memory System with Hierarchical Temporal Indexing 

**Authors**: Han Chen, Zining Zhang, Wenqi Pei, Bingsheng He, Ming Wu, Jason Zeng, Michael Heinrich, Wei Wu, Hongbao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.23986)  

**Abstract**: Memory is a fundamental component for enabling long-context LLM agents, supporting persistent state across interactions through a continuous serve-and-update lifecycle. Despite substantial prior work, existing systems suffer from significant maintenance overhead due to two key limitations: coarse-grained state management and inherently sequential update pipelines. In particular, updates are often tightly coupled with LLM inference and require full-state rewrites, leading to poor scalability and growing latency as memory accumulates. To address these challenges, we present MemForest, a memory framework that reformulates agent memory as a write-efficient temporal data management problem. MemForest breaks the sequential bottleneck via parallel chunk extraction, decoupling memory construction into concurrent, independent operations. To further eliminate coarse-grained maintenance, we introduce MemTree, a hierarchical temporal index that organizes memory as time-ordered trees rather than flat global summaries. This design replaces full-state rewrites with localized per-node updates, reducing maintenance cost to the affected tree paths while naturally preserving temporally evolving states. We evaluate MemForest on two long-context memory benchmarks, LongMemEval-S and LoCoMo. On LongMemEval-S, MemForest achieves the best overall performance among stateful baselines, reaching 79.8% pass@1 accuracy while sustaining a memory construction throughput approximately 6x higher than state-of-the-art approaches including EverMemOS. 

---
# Harnessing AtomisticSkills for Agentic Atomistic Research 

**Authors**: Bowen Deng, Bohan Li, Matthew Cox, Hoje Chun, Juno Nam, Artur Lyssenko, Sathya Edamadaka, Jurgis Ruza, Xiaochen Du, Nofit Segal, Jesus Diaz Sanchez, Mingrou Xie, Ty Perez, Yu Yao, Miguel Steiner, Sauradeep Majumdar, Charles B. Musgrave III, Anirban Chandra, Abhirup Patra, Detlef Hohl, Connor W. Coley, Ju Li, Rafael Gómez-Bombarelli  

**Link**: [PDF](https://arxiv.org/pdf/2605.24002)  

**Abstract**: Computational materials science and chemistry span vast knowledge domains and fractured software ecosystems. Although large language models (LLMs) have demonstrated research capabilities, scaling monolithic agents to manage the rigor and complexity of atomistic research remains a challenge. Here, we introduce AtomisticSkills, an open-source harness framework that empowers general-purpose AI coding agents to conduct atomistic research across materials science, chemistry, and drug discovery. By hierarchically decomposing scientific workflows into agent skills and tools, AtomisticSkills provides agents with modular, extensible, and plug-and-play research capabilities. The framework integrates more than 100 human-curated multidisciplinary skills, including database access, thermodynamics and kinetics modeling, and diverse simulation engines employing machine learning interatomic potentials (MLIPs) and density functional theory (DFT). We validate its functional coverage against scientific literature and demonstrate robust orchestration capabilities across diverse scientific campaigns: generative design of Li-ion solid-state electrolytes, high-throughput screening of metal-organic frameworks for CO2 capture, autonomous MLIP benchmarking and fine-tuning, multi-stage structure-based virtual screening for drug design, multimodal X-ray diffraction pattern analysis, and screening of Fe-oxide catalysts for oxygen evolution reaction. AtomisticSkills provides a critical agent infrastructure towards building fully autonomous AI scientists. 

---
# IVR-R1: Refining Trajectories through Iterative Visual-Grounded Reasoning in Reinforcement Learning 

**Authors**: Chenghao Li, Fusheng Hao, Xikai Zhang, Likang Xiao, Yanwei Ren, Fuxiang Wu, Quan Chen, Liu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.23997)  

**Abstract**: Multimodal large language models via reinforcement learning (RL) have demonstrated remarkable capabilities in complex visual reasoning tasks, yet they remain limited in long-horizon multimodal scenarios, often suffering from visual hallucination and logical error. Current methods typically pre-encode high-dimensional visual scenes into discrete textual proxies to facilitate downstream reasoning. As the reasoning chain unfolds, however, the inherent information asymmetry between text and visual scenes tends to erode visual grounding, resulting in misguided reasoning and erroneous outputs. To address this issue, we introduce IVR-R1 (Iterative Visual-grounded Reasoning), a novel RL training framework that facilitates dynamic visual re-alignment that actively rectifies reasoning trajectories to guide policy optimization. Specifically, by leveraging a reward-driven screening mechanism to identify flawed rollouts, IVR-R1 executes a fine-grained, step-level error attribution within the multimodal context. By iteratively cross-referencing intermediate reasoning states against pristine visual priors, a Re-Reasoning Loop enables automated trajectory rectification, effectively synthesizing expert-level demonstrations that serve as high-fidelity reasoning templates for the policy model. Our experiments across diverse multimodal benchmarks demonstrate that IVR-R1 consistently outperforms existing reinforcement learning methods, establishing a superior paradigm for maintaining logical and visual consistency in complex multimodal reasoning. 

---
# TriVAL: A Tri-Validation Framework for Faithful Automatic Optimization Modeling 

**Authors**: Ziyang Fang, JinXi Wang, Jinghui Zhong, Yew-Soon Ong  

**Link**: [PDF](https://arxiv.org/pdf/2605.23966)  

**Abstract**: Optimization modeling serves as the pivotal bridge between natural-language problem descriptions and optimization solvers, and remains a cornerstone for bringing operations research (OR) into real-world decision making. Recent advances in large language models (LLMs) have driven significant progress in automatic optimization modeling. However, existing methods still lack explicit validation during the modeling process, allowing errors introduced in earlier stages to carry through the pipeline and ultimately reduce final modeling accuracy. To address this challenge, we introduce TriVAL, a tri-validation framework that performs explicit validation at three stages of automatic optimization modeling: semantic specification, mathematical formulation, and code generation. At each stage, TriVAL follows a construct-validate-revise loop that assesses the current result against stage-specific criteria and revises it when needed. This design helps identify and correct errors before they accumulate across stages, helping preserve faithfulness throughout the modeling process. To evaluate automatic optimization modeling on more challenging combinatorial problems, we further introduce NL4COP, a benchmark of 150 instances across 50 diverse problem types with more complex decision logic, more tightly coupled constraints, and more demanding modeling requirements than existing benchmarks. Experiments on NL4COP and established benchmarks show that TriVAL consistently outperforms state-ofthe-art methods, with the largest gains on the most challenging problems. 

---
# Metacognition Should Be the Scientific Framework for Bounded and Effective Self-Governance in Generative AI 

**Authors**: Eugene Yu Ji, Igor Grossmann, Amir-Hossein Karimi  

**Link**: [PDF](https://arxiv.org/pdf/2605.23981)  

**Abstract**: Generative AI research increasingly confronts a shared problem: systems must sustain yet govern their own generative activity when uncertainty is high, evidence is missing, or context is insufficient. This position paper argues that metacognition should become the scientific framework for bounded and effective self governance in generative AI, where output generation is properly evaluated together with the capacities through which generative systems navigate and regulate their own activity. We advance this position by showing that bounded and effective AI self-governance requires metacognitive alignment across computational, algorithmic, and ecological levels. At the computational level, metacognition specifies the meta-level functions a system is meant to serve, such as monitoring, evaluation, control, and adaptation. At the algorithmic level, these functions are realized through procedures such as elicitation, iteration, and modularization. At the ecological level, metacognitive signals become meaningful, actionable, and accountable within the interface, workflow, and accountability arrangements. Metacognition thus makes it possible to conceive generative AI as both capable and well-governed, rather than treating capability and governance as competing aims. 

---
# SODE: Analyzing Social Dynamics in LLM Agents 

**Authors**: Inseo Jung, Yoonseok Oh, Kyungryul Back, Jinkyu Kim, Jungbeom Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.23949)  

**Abstract**: As Large Language Models (LLMs) evolve into interactive agents, understanding their behavioral alignment within human social dynamics becomes essential. While behavioral game theory offers a framework to study these interactions, previous work has predominantly relied on outcome-based metrics such as average scores. This focus overlooks the mechanisms that facilitate sustainable cooperation, as identical scores can be derived from vastly different strategies. To bridge this gap, we introduce SODE (Social Dynamics Evaluation), a framework that evaluates LLM agents across three evolutionary dimensions: Direct Reciprocity for strategy adaptation, Indirect Reciprocity for reputation sensitivity, and Group Dynamics for cooperative resilience. Applying SODE reveals systematic divergences: instruction-tuned models often exhibit "passive compliance" that renders them vulnerable to exploitation, while reasoning models prioritize short-horizon optimization, destabilizing long-term cooperation. Notably, we demonstrate that a "long-horizon framing" can unlock reciprocal capabilities in reasoning models. Thus, SODE offers a systematic, mechanism-grounded benchmark for aligning AI agents with complex human social dynamics. 

---
# KT4EQG: Personalized Exercise Question Generation via Knowledge Tracing 

**Authors**: Xinyi Gao, Qiucheng Wu, Lu Ding, Q.Vera Liao, Kaizhi Qian, Ying Xu, Shiyu Chang, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.23933)  

**Abstract**: Educational Question Generation (EQG) aims to synthesize customized exercise questions that enhance student learning. An effective EQG system should ideally personalize questions for each student by modeling the student's knowledge state and generating questions that provide the greatest learning benefit. However, few existing EQG approaches are able to achieve such fine-grained personalization. In this paper, we explore how EQG can benefit from knowledge tracing (KT), which models students' knowledge states based on historical performance and predicts future performance. We propose KT4EQG, a personalized EQG framework that generates effective questions for individual students under the guidance of a KT model. Specifically, KT4EQG seeks to maximize a student's potential improvement in overall knowledge mastery by leveraging the KT model to select the most suitable knowledge concept for the student to practice. An LLM-based question generator is then trained to produce a question faithfully grounded in the selected concept. Experimental results on XES3G5M and MOOCRadar show that KT4EQG consistently generates more effective questions than methods with limited or no personalization. 

---
# EchoDistill:Alignment Noisy-to-Clean Self-Distillation for Robust Audio LLMs 

**Authors**: Liang Lin, Chunxi Luo, Kaiwen Luo, Jie Zhang, Jin Wang, Yuanhe Zhang, Cai Yuchen, Qiankun Li, Gongli Xi, Zhenhong Zhou, Kun Wang, Junhao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2605.23954)  

**Abstract**: Audio Large Language Models (ALLMs) are highly vulnerable to real-world noise, which often induces severe semantic drift and hallucinations. Existing robustness methods primarily rely on waveform-level acoustic enhancement, answer-level supervision, or the internal suppression of noise representations. To address these issues, we propose echodistill, an alignment-based noisy-to-clean self-distillation framework. Echodistill leverages a frozen clean-audio teacher to provide semantic references for an inference-time noisy-audio student. Specifically, the student samples candidate responses under noisy conditions to expose its test-time behavior. These trajectories are then optimized via group-relative policy optimization (GRPO), where the token-level consistency with the teacher acts as a reward bonus. By aligning the noisy student's candidate responses with clean semantic evidence, and applying audio-aware reward shaping, our method encourages reasoning trajectories that are both correct and genuinely acoustically grounded. Echodistill significantly improves the semantic reliability and task performance of Audio LLMs under complex noise, without introducing any additional inference costs. Extensive experiments show that: (I) Compared with the strongest baseline, echodistill achieves average improvements of 4.18\%$\uparrow$ in GSR under strong noise. (II) Ablation results on Qwen-Omni further show that echodistill improves over the GRPO-only variant by 3.02\%$\uparrow$ in Acc, 3.89\%$\uparrow$ in Noisy, and 4.53\%$\uparrow$ in GSR on average. Our codes are available at this https URL. 

---
# Authority Signals in Claude AI Health Citations: A Descriptive Analysis Using the Authority Signals Framework 

**Authors**: Erin T. Jacques, Erela Datuowei, Elizabeth Quaye, Corey H. Basch, Arijit Chatterjee, Juanita Davis  

**Link**: [PDF](https://arxiv.org/pdf/2605.23921)  

**Abstract**: This study seeks to determine the authority signals used by Anthropic's Claude AI in its presentation of sources when answering consumer health questions. While there exists a great deal of discourse around the quality of health citations that LLMs produce, there is limited information on the integrity of the sources the citations originate from, and to what extent the sources are, from what health professionals would consider, credible sources. This descriptive cross-sectional study used data from HealthSearchQA, which contains 3,172 consumer health questions curated by Google Research. After exclusions, a final dataset of 3,075 questions yielding 10,038 citations was analyzed. The Authority Signals Framework (Jacques et al., 2026) was applied to examine 10 authority signals across four domains for a disproportionate stratified sample of 542 sources. Established institutional sources accounted for 97.8% of all citations (n = 9,818). Medical Institutions were the most frequently cited organization type (36.5%), followed by Government Resources (31.6%) and Professional Associations (28.4%). Commercial Health Information comprised 2.2% (n = 220). The top 10 organizations accounted for 57.8% of all citations, with Mayo Clinic alone representing 24.7%. Among commercial sources in the focused sample, 86.4% displayed medical review statements, 82.5% used schema markup, and 71.8% had comprehensive content, while traditional institutional sources appeared in Claude's citations with or without these same markers. As Anthropic positions Claude for HIPAA-ready healthcare applications, these findings establish a baseline for Claude's citation behavior and demonstrate the utility of the Authority Signals Framework as a tool for ongoing, cross-platform evaluation of AI-mediated health information. 

---
# Agent-Facing Information Design in LLM Tool Registries 

**Authors**: Haochuan Kevin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.23916)  

**Abstract**: LLM tool registries function as unregulated advertising platforms: providers write free-text descriptions that agents use for selection, yet no measurement infrastructure -- no viewability standard, quality score, or outcome audit -- exists to make this market accountable. We provide the first systematic framework, combining 17,700+ trials across five LLMs and ten domains with a constructive registry design prescription. Legal puffery alone (subjective superlatives, benefit framing) captures 100% of the optimization effect; fabricated claims add zero incremental bias -- rendering FTC enforcement of deceptive advertising rules ineffective against the active mechanism. Disclosure fails structurally: system-prompt warnings produce zero measurable effect for four of five models, and behavioral ceilings leave no headroom for label-based correction. Superlatives are the dominant single feature (SBC = +0.35). Registry-layer description normalization achieves first-best welfare model-independently. We propose separating selection-facing descriptions (structured, registry-controlled) from marketing-facing descriptions (provider-authored, shown post-selection), and introduce the Agent Attention Quality Score to distinguish capability from copywriting. 

---
# VineLM: Trie-Based Fine-Grained Control for Agentic Workflows 

**Authors**: Nikos Pagonas, Matthew Lou, Tianyi Peng, Dan Rubenstein, Kostis Kaffes  

**Link**: [PDF](https://arxiv.org/pdf/2605.23914)  

**Abstract**: Agentic workflows interleave configurable LLM stages with tool stages and often include retries or refinement loops. Existing workflow managers profile full workflow configurations offline and assign each request a static workflow-level plan that binds each configurable LLM stage to a single model, reuses that model across repeated loop iterations, and does not revisit those choices at runtime. We present VineLM, a workflow manager that enables fine-grained control by choosing the model for each stage invocation as execution unfolds under request-level objectives such as maximizing accuracy under cost or latency budgets. VineLM represents feasible executions as an annotated trie of model-choice prefixes and uses checkpointing and cascade profiling to estimate path accuracy, cost, and latency without exhaustively profiling every request on every path. At runtime, VineLM re-roots the trie after each stage invocation and replans over the remaining subtrie using the realized execution prefix and remaining latency budget. On NL2SQL and math reasoning workflows, VineLM improves the cost-latency-accuracy frontier over coarse workflow-level baselines, achieving up to 18% higher accuracy at the same per-request budget with its sparse profiling reducing offline profiling cost by 98-99.8% when compared to exhaustive profiling. 

---
# Artificial Effort 

**Authors**: Federico Belotti, Stefano Coniglio, Antonio Cosma, Francesco Fallucchi  

**Link**: [PDF](https://arxiv.org/pdf/2605.23920)  

**Abstract**: Real-effort tasks, in which participants perform cognitively costly activities whose outcomes depend on actual performance, are widely used in experimental economics. Their validity, however, rests on the assumption that a human performs them. We study whether this assumption still holds in the era of Artificial Intelligence (AI) and Large Language Models (LLMs). Using 8 canonical real-effort tasks and 23 LLMs from three major providers, we show that most tasks can now be solved accurately and at a negligible cost, while only a few resist automation. Performance improves with each model generation, and midtier models are rapidly closing the gap with frontier ones, broadening the set of widely accessible models that can automate these tasks. Additionally, we show that verbally offering monetary incentives has no effect on LLM performance. Our findings establish a boundary condition for the use of real-effort tasks in unsupervised settings: when participants can cheaply outsource task completion to an LLM, observed performance may no longer reflect genuine human effort. 

---
# Raon-Speech Technical Report 

**Authors**: Beomsoo Kim, Changho Choi, Dohyun Kim, Dongki Lee, Ethan Ewer, Eunchong Kim, Gyeongman Kim, Haechan Kim, Hyeonghwan Kim, Inkyu Park, Jihun Yun, Jihwan Moon, Jiyun Kim, Joonghyun Bae, Junhyuck Kim, Minkyu Kim, Sehun Lee, Seungjun Chung, Sungwoo Cho, Dongmin Park, Dongwon Kim, Hara Kang, Jonghyun Lee, Keon Lee, Kangwook Lee, Jaewoong Cho  

**Link**: [PDF](https://arxiv.org/pdf/2605.23912)  

**Abstract**: We present Raon-Speech, a top-performing 9B-parameter speech language model (SpeechLM) for English and Korean speech understanding, answering, and generation, and Raon-SpeechChat, a high-performing full-duplex extension for natural real-time conversation. Raon-Speech successfully transforms a pre-trained LLM into a SpeechLM that both understands and generates speech while preserving strong text capabilities. It trains on 1.38M hours of highly curated English and Korean speech and text datasets with the following training stages: (1) speech modules alignment, (2) end-to-end SpeechLM pre-training with knowledge distillation, and (3) multi-task preference optimization-based post-training. Across 42 English and Korean speech and text benchmarks, Raon-Speech establishes the strongest overall profile on speech-centric tasks in our comparison against eight similarly sized recent audio foundation models, including Qwen2.5-Omni and Fun-Audio-Chat, while preserving strong text question answering performance. Building upon it, Raon-SpeechChat enables natural full-duplex conversation by continual training on 119K hours of time-aligned real and synthetic dialogue data. It proceeds through three complementary training stages: (1) causal encoder adaptation, (2) full-duplex pre-training, (3) full-duplex fine-tuning for voice and role-control. On multiple full-duplex benchmarks, Raon-SpeechChat shows its clearest strengths on the turn-taking and interruption-sensitive behaviors covered by FDB v1.0, and remains competitive across the broader full-duplex evaluation suite. We open-source all model checkpoints, the training and inference pipeline, and an interactive demo. 

---
# Check Your LLM's Secret Dictionary! Five Lines of Code Reveal What Your LLM Learned (Including What It Shouldn't Have) 

**Authors**: Hisashi Miyashita  

**Link**: [PDF](https://arxiv.org/pdf/2605.22005)  

**Abstract**: We show that singular value decomposition of the lm_head} weight matrix of a transformer-based large language model -- requiring only five lines of PyTorch and no model inference -- reveals interpretable semantic subspaces directly from the model weights. Each left singular vector identifies the vocabulary tokens most readily selected when the hidden state aligns with the corresponding singular direction; inspecting these clusters exposes the model's training data composition and curation philosophy.
Analysing GPT-OSS-120B, Gemma-2-2B, and Qwen2.5-1.5B, we find that singular value spectra and vocabulary cluster structures differ systematically across models: GPT exhibits a graduated hierarchy of functionally differentiated subspaces; Gemma is dominated by pre-nineteenth-century English orthography, forming a stepwise clustering structure that may contribute to high output controllability; and Qwen exhibits broad multilingual coverage alongside subspaces whose vocabulary the authors have determined to be ethically inappropriate for direct publication.
Base-instruct comparison reveals that ethically concerning subspaces originate in pretraining and are not removed by post-training alignment. We introduce the Vocabulary Cluster Score (VCS) to quantify subspace coherence, and the Weighted Projection Score (WPS) as a static glitch token detector; applying WPS to GPT-OSS-120B recovers shokubutsu-hyakka-tsu (ID 137606), a well-known glitch token widely reported in the CJK language community, without any model inference. We propose a taxonomy of root causes for problematic vocabulary content and call for lm_head} SVD analysis to be adopted as a standard pre-release safety auditing step. Our findings further suggest directions toward SVD-guided tokenizer optimisation and more controllable LLM design. 

---
# AI in the Enterprise: How People Use M365 Copilot Chat 

**Authors**: Scott Counts, Yan Chen, Jing Dong, Himanshu Sharma, Andrey Zaikin, Rui Hu, Alperen Kok, Gorkem Ozer Yilmaz, Siddharth Suri, Kiran Tomlinson, Sonia Jaffe, Will Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.23958)  

**Abstract**: M365 Copilot is used every week by millions of people across more than a million companies around the world as part of their workflows. Uniquely positioned in the AI landscape given its near-exclusive use for work purposes, M365 Copilot can offer a clear picture of how people use AI for work and where that usage may expand next. This paper characterizes that usage through direct classification of user interactions with M365 Copilot Chat. Based on an anonymized and privacy-preserving analysis of a sample of approximately 5.5 million sessions, we combine a learned classification of user intent with a classification of O*NET work activities done with M365 Copilot Chat. We find that M365 Copilot is emerging as an everyday assistant for knowledge work: writing dominates, but users also rely on it for information retrieval, analysis, decision making and strategizing, and evaluating and diagnosing programs and systems, among others. Information seeking tasks remain common, but time trends suggest a relative shift away from ``chat as search'' and toward content and communication-related work. Comparisons across occupational groupings and to work done in the labor market further show that usage is broad but uneven, where the relative share of work done with M365 Copilot Chat cuts across jobs in some cases and is occupation-specific in others. Areas of relative underrepresentation in the labor market suggest the next frontier for enterprise AI adoption. 

---
# Catching The Correct Answer Trap: Characterising AI Tutor Blind Spots When Analysing Student Reasoning 

**Authors**: Moiz Imran, Sahan Bulathwela  

**Link**: [PDF](https://arxiv.org/pdf/2605.23925)  

**Abstract**: Intelligent tutoring systems increasingly provide automated feedback on student work, but robust feedback requires assessing reasoning, not only final answers. We study a failure mode we call the correct answer trap (CAT): models under-detect misconceptions when students reach a correct answer via flawed reasoning. Analysing real student responses from the Eedi mathematics platform, we show that 71% of these failures concentrate in just two question types, both sharing a common structure where flawed reasoning happens to produce the correct numerical answer. Comparing a fine-tuned T5 with a frontier large language model, we find that improved capabilities reduce but do not eliminate the problem (84% vs 57% detection accuracy). Even the best-performing model generates roughly four false alarms for every genuine detection, making stand-alone screening impractical at realistic class sizes. Our findings demonstrate that high overall accuracy can mask critical failures in reasoning assessment, and that careful analysis of student reasoning still benefits from human judgment. 

---
# Tokenizer Fertility and Zero-Shot Performance of Foundation Models on Ukrainian Legal Text: A Comparative Study 

**Authors**: Volodymyr Ovcharov  

**Link**: [PDF](https://arxiv.org/pdf/2605.14890)  

**Abstract**: Tokenizer fertility varies 1.6x across foundation models on Ukrainian legal text, yet this cost-critical dimension is absent from model selection practice. We benchmark seven models from five providers on 273 validated court decisions from Ukraine's state registry (EDRSR), measuring tokenizer fertility and zero-shot performance on three tasks. Four findings emerge. (1) Qwen 3 models consume 60% more tokens than Llama-family models on identical input, making tokenizer analysis a prerequisite for cost-efficient deployment. (2) NVIDIA Nemotron Super 3 (120B) achieves the highest composite score (83.1), outperforming Mistral Large 3 (5.6x more total parameters) at one-third the API cost model scale is a poor proxy for domain performance. (3) Few-shot prompting degrades performance by up to 26 percentage points; stratified and prompt-sensitivity ablations confirm this is intrinsic to Ukrainian-language demonstrations, not an artifact of example selection. (4) A cross-temporal generalization experiment reveals that classifiers trained on pre-war court ecisions (2008-2013) lose 27.9 percentage points when applied to full-scale invasion era decisions (2022-2026), with a pronounced forward-backward asymmetry: newer models transfer backward (+14.6 pp above forward transfer), but older models fail catastrophically on wartime legal language. For practitioners: tokenizer analysis should precede model selection, and zero-shot is a more reliable default than few-shot for morphologically rich languages. To support reproducibility and address the absence of Ukrainian from legal NLP benchmarks, we release a public dataset of 14,452 court decisions spanning 2008-2026, annotated with seven outcome labels across three temporal epochs that capture the impact of armed conflict on judicial proceedings. 

---
# Peak-Then-Collapse and the Four Interface Channels of Knowledge-Graph Tool Use 

**Authors**: Tianda Sun, Dimitar Kazakov  

**Link**: [PDF](https://arxiv.org/pdf/2605.26037)  

**Abstract**: We test the standard RLVR tool-use recipe -- GRPO on Qwen2.5-7B-Instruct -- on a deliberately minimal knowledge-graph tool API: four Freebase navigation verbs over Complex WebQuestions. Under a self-verifiable retrieval reward, the policy's tool-grounded answer rate climbs from $3.8\%$ to $9.6\%$ over 250 steps, then collapses to $0\%$ within a single 50-step window -- a \emph{peak-then-collapse} pattern replicated across four seeds. Across seven reward designs, we find four recurring failure modes: adding denser or more targeted proxy rewards shifts the failure mode rather than eliminating it. We argue that a key difference from Python interpreters, web search, and JSON APIs is interface feedback: their failures often leak natural-language signal the model saw in pretraining. A Python traceback names the failing line; an empty Freebase result \texttt{[]} does not. Stripping away that surface exposes a degradation regime that same-family reward redesigns do not fix. A direct oracle ablation rules out relation selection: injecting gold relations at every retrieval call lifts exact-match accuracy by only $+0.20$~pp, and $95.4\%$ of retrieval-dependent errors are retrieval-composition failures rather than answer-extraction failures. As a mitigation, one-iteration self-distillation reaches $40.0\%$ EM at 7B and is capacity-invariant: doubling capacity to 14B improves EM by only $0.25$~pp, and initialization barely matters -- the ceiling appears interface-bound within the 7B--14B range tested. 

---
# WhoSaidIt: Human-LLM Collaborative Annotation for Text-Based Multilingual Speaker-Attribute Classification 

**Authors**: Lingyu Gao, Will Monroe, David Smith, Meghan Jemison, Jackie Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.26070)  

**Abstract**: Annotating speaker attributes from text is inherently ambiguous, particularly in multilingual settings where demographic and social cues are implicit and culturally variable. We propose a human-large language model (LLM) collaborative re-annotation framework for stabilizing multilingual speaker-attribute labels under practical resource constraints. Starting from a noisy corpus, we use LLMs to surface recurring annotation rationales through iterative interaction with experts, and apply disagreement-focused sampling for targeted re-annotation. Using this framework, we construct WhoSaidIt, a multilingual dataset covering nine speaker-attribute labels. We quantify divergence between original and revised annotations, benchmark recent LLMs, and analyze the effect of explicit rationales on model behavior. Our results reveal substantial cross-lingual differences in annotation decisions and demonstrate both the strengths and limitations of LLMs in speaker-attribute classification. 

---
# What Makes a Medical Checker Trainable? Diagnosing Signal Collapse and Reward Hacking in Checker-Guided RAG for Biomedical QA 

**Authors**: Yuelyu Ji, Min Gu Kwak, Hang Zhang, Xizhi Wu, Chenyu Li, Yanshan Wan  

**Link**: [PDF](https://arxiv.org/pdf/2605.25988)  

**Abstract**: Medical RAG needs evidence-grounded claims, so plugging a claim-level NLI checker into retrieval-augmented RL is intuitive. \textbf{We find that the checker's \emph{output distribution} during training, not its held-out accuracy, decides whether it provides trainable gradient.} We compare four NLI checker back-ends as process rewards inside a GRPO-trained medical RAG agent (Qwen2.5-7B, replicated on Qwen3-4B and Llama-3.1-8B) across four held-out medical QA benchmarks. Three diagnostic findings emerge. \textbf{(i)} Signal collapse is log-prob-specific: LLM log-probability scoring labels over 97\% of claims neutral -- collapsing the RL gradient to zero -- while a calibrated MedNLI classifier scores the same pairs non-degenerately. \textbf{(ii)} Moderate signal beats strong signal on answer quality: a strong proprietary checker triggers a three-step reward-hacking cascade -- ultra-short answers, search avoidance, language collapse -- so a moderate-signal local classifier trains a higher-quality model (\textbf{+12\% BERTScore over zero-shot, no GPT dependency}). \textbf{(iii)} Signal strength is policy-dependent: the same checker registers as moderate on one policy but strong on another without triggering the cascade end-state. We frame these as boundary conditions for verifier-as-reward systems. 

---
# Forgotten Words: Benchmarking NeoBERT for Dementia Detection in Low-Resource Conversational Filipino and English Speech 

**Authors**: Rez Samantha Z. Floresca, Edric Castel C. Hao, Hannah Grachiella Buñales, Chelsea Dominique E. Temprosa, Georgianna Z. Reyes, Kervin Gabriel L. Chua  

**Link**: [PDF](https://arxiv.org/pdf/2605.26007)  

**Abstract**: Dementia detection from spontaneous speech offers a scalable approach to cognitive screening, yet NLP systems remain predominantly English-centric. This limitation is especially acute in the Philippines, where Filipino-English code-switching is pervasive and no prior work has addressed NLP-based dementia detection. We present the first systematic evaluation of transformer-based dementia detection in Filipino speech and the first assessment of NeoBERT in a clinical NLP setting. To separate language from domain effects, we construct a parallel bilingual dataset of 4,000 DementiaBank-derived transcripts, with Filipino translations produced manually to preserve discourse-level markers of cognitive decline. We evaluate five model families, TF-IDF + LogReg, BERT, NeoBERT, XLM-R, and RoBERTa-Tagalog, under monolingual, zero-shot cross-lingual, and bilingual fine-tuning settings. We find that in-domain performance does not transfer across languages, with English-trained BERT dropping to Macro-F1 = 0.455 on Filipino, and that architectural modernization alone does not improve robustness. Bilingual fine-tuning, however, eliminates cross-lingual degradation across all transformer models, converging to Macro-F1 = 0.969-0.973. These results suggest that multilingual clinical NLP performance is driven primarily by linguistic coverage during training rather than model scale or architecture. 

---
# When Do LLM Agents Treat Surface Noise Differently from Semantic Noise? A 68-Cell Measurement Study with a Held-Out Trace-Level Validation 

**Authors**: Liyun Zhang, Jiayi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2605.25981)  

**Abstract**: We document an empirical phenomenon in chain-of-thought and ReAct agents driven by ten large language models from seven architecture families: meaning-bearing perturbations (e.g., paraphrase, synonym) alter final answers more often than presentation perturbations (e.g., formatting, reordering) of comparable severity. Across 68 cells spanning GSM8K, MATH, and HotpotQA (1,530 originals and $\sim$11,150 variants), the inconsistency gap averages +19.69 pp after severity matching (paired $t=9.58$, $p<0.0001$), with 64/68 cells positive. The gap survives four severity-proxy audits and remains significant when excluding qwen models (+11.10 pp, $p<0.0001$). Several stress tests fail honestly: cluster-bootstrap significance disappears under stricter assumptions, tractability contrasts do not replicate, cross-architecture generator swaps break per-cell rankings, and a second LLM judge yields only moderate agreement ($\kappa=0.50$).
We then validate the headline effect on a fully held-out 11th model (qwen2.5-14B-Instruct; 1,800 trajectories) and re-test a pre-registered capability$\times$tractability partition, observing a small but positive held-out effect (3/4 cells positive; pooled Welch $t=3.81$, $p=9.6\times10^{-4}$). Using held-out trajectories, we probe four trace-level mechanism signals. Two prior mechanism claims fail to replicate and are explicitly retracted. Two new probes instead support a \emph{stealth-divergence} picture: semantic perturbations often preserve the first action but induce divergence in intermediate reasoning from later steps onward, accompanied by slightly deeper trajectories. We position this as a measurement contribution with held-out replication and a partial trace-level account of how semantic perturbations propagate through agent reasoning. Code, perturbation corpus, raw trajectories, and analysis scripts are released anonymously for review. 

---
# Triplet-Block Diffusion RWKV 

**Authors**: Ke Lin, Yiyang Luo, Zhaolong Su, Yunya Song, Anyi Rao  

**Link**: [PDF](https://arxiv.org/pdf/2605.25969)  

**Abstract**: Causal Transformer language models suffer from strictly sequential decoding and a quadratic per-step attention cost. While linear-time causal models and discrete diffusion models each address these weaknesses, their integration remains inherently inconsistent: diffusion requires bidirectional attention, while causal models are unidirectional. To unify these architectures, we propose $B^3D-RWKV$, a diffusion RWKV variant that integrates the model's $O(L)$ inference efficiency with parallel, bidirectional discrete-diffusion through a \emph{triplet-block layout} method. $B^3D-RWKV-7.2B$ reaches comparable accuracy on an 8-task suite versus existing models while significantly outperforming baselines in decoding throughput with an average of $\mathbf{1.6\times}$ speedup. 

---
# PolyGnosis 2.0: Enhancing LLM Reasoning via Agentic Harness Engineering for Polymarket and OSINT Insight Extraction 

**Authors**: Daren Wang, Hong Xu, Jiawen Xian  

**Link**: [PDF](https://arxiv.org/pdf/2605.25958)  

**Abstract**: This paper introduces PolyGnosis 2.0, a pioneering multi-agent architecture designed to extract predictive intelligence by synthesizing Polymarket anomaly signals with global Open Source Intelligence (OSINT) streams, specifically Global Database of Events, Language, and Tone (GDELT). We define and target "Perspective Mismatches", the narrative divergence between Polymarket sentiment and global media flows, as high-alpha trading signals. Moving beyond generic agentic superiority, we rigorously quantify the efficacy of "Harness Engineering" techniques, including reflection loops, tool-calling, divide-and-conquer partitioning (D&C), and chain-of-thought (CoT), within high-noise financial domains. Our empirical evaluation against human-expert benchmarks reveals that while structural partitioning is mandatory for multi-dimensional alignment, unconstrained terminal reflection actively induces logical drift. Furthermore, we identify a pervasive "consensus bias" across all agent configurations during narrative reasoning, necessitating deterministic validation. Ultimately, we isolate a Pareto-optimal configuration that achieves professional-grade analytical precision while minimizing latency and token overhead, providing a robust blueprint for autonomous intelligence in prediction markets. 

---
# Automated Benchmark Auditing for AI Agents and Large Language Models 

**Authors**: Junlin Wang, Federico Bianchi, Shang Zhu, Fan Nie, Yongchan Kwon, Bhuwan Dhingra, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2605.26079)  

**Abstract**: Modern AI benchmarks operate at a complexity that outpaces traditional verification methods. Tasks authored by domain experts often contain implicit assumptions, incomplete environment specifications, and brittle evaluation logic that human annotation cannot reliably catch. We introduce Auto Benchmark Audit (ABA), an agentic framework that systematically audits individual benchmark tasks, uncovering issues such as hidden environment dependencies, specification gaps, and limited grading logic. We run ABA on a collection of frontier LLM benchmarks and previous NeurIPS publications, totaling 168 benchmarks across nine domains. Across this corpus, ABA identifies critical issues including ambiguous task design, execution environment conflicts, and incorrect ground truths in over 25.7% of the evaluated tasks. The precision of these automated audits is validated by expert review and independent third-party reports such as upstream PRs. Crucially, we demonstrate that these problematic tasks severely distorts capability assessments for agents and LLMs: filtering out these tasks with issues shifts model rankings and increases average performance on SWE-bench Verified and Terminal-Bench 2 by 9.9% and 9.6%, respectively. We release the agentic tool and all task annotations to support the future development of frontier benchmarks. 

---
# Mitigating Provenance-Role Collapse in Long-Term Agents via Typed Memory Representation 

**Authors**: Zhengda Jin, Bingbing Wang, Jing Li, Ruifeng Xu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.25869)  

**Abstract**: Long-term memory is essential for persistent LLM agents, yet prevailing architectures store historical interactions as unstructured, flat text. This unconstrained storage induces provenance-role collapse, a critical failure mode where agents suffer from source-monitoring errors. To resolve this cognitive vulnerability at the architectural level, we propose MemIR, a typed Memory Intermediate Representation that operationalizes source monitoring as a structural constraint. MemIR writes long-term memory into grounded atoms that separate raw evidence, retrieval cues, and truth-bearing claims, with factual authorization restricted to supported claim atoms. It then applies multi-route atomic projection and provenance-scoped utilization to transform heterogeneous retrieval hits into claim-centered candidate bundles and a normalized fact interface for answer generation. Experiments on LoCoMo and BEAM-100K demonstrate that MemIR consistently outperforms existing memory baselines, especially on tasks requiring source tracking, temporal grounding, and aggregation of fragmented evidence. 

---
# Anticipate and Learn: Unleashing Idle-Time Compute in Proactive Agents 

**Authors**: Haoyi Hu, Qirong Lyu, Xianghan Kong, Weiwen Liu, Jianghao Lin, Zixuan Guo, Yan Xu, Yasheng Wang, Weinan Zhang, Yong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.25971)  

**Abstract**: While AI agents demonstrate remarkable capabilities in reasoning and tool use, they remain fundamentally reactive: they compute responses only after explicit user prompts. This paradigm ignores a critical opportunity: the idle time between interactions is largely wasted, leaving agents unable to prepare for future user needs. To bridge this gap, we introduce ProAct, a proactive agent architecture that leverages idle-time compute to anticipate and fulfill likely upcoming user needs. By analyzing evolving dialogue history together with persistent memory, ProAct predicts upcoming needs and iteratively acquires information, allowing the agent to resolve knowledge gaps and prepare evidence before the user initiates a this http URL rigorously evaluate proactive capabilities, we also introduce ProActEval, a comprehensive benchmark comprising 200 scenarios across 40 domains, featuring predictable need chains and diverse user cognitive profiles. Empirical results demonstrate significant advantages over reactive baselines. ProAct accelerates task completion by reducing required turns by 14.8%, decreases user effort by 11.7%, and cuts hallucination rates by 28.1% on ProActEval. Furthermore, MemBench evaluations confirm that ProAct achieves state-of-the-art reflective accuracy, underscoring its sustained and robust performance. 

---
# Universal Activation Verbalizer: A Unified Framework for Cross-Model Activation Explanation 

**Authors**: Haiyan Zhao, Zirui He, Guanchu Wang, Ali Payani, Yingcong Li, Mengnan Du  

**Link**: [PDF](https://arxiv.org/pdf/2605.25903)  

**Abstract**: Activation verbalization explains hidden representations in natural language, but existing methods are mostly limited to self-explanation, where each model explains only its own activations. We introduce Universal Activation Verbalizer (UAV), a framework that uses a shared decoder to explain activations from heterogeneous donor models. UAV learns a lightweight adapter that converts donor activations into soft tokens in decoder's embedding space, and further supports adapter-only transfer by reusing a frozen decoder-side LoRA while training only a new adapter for another donor. Across classification, fact retrieval, and gist summarization, UAV remains competitive with strong self-explanation baselines while enabling cross-model verbalization across model families and scales. Ablations show that decoder-side tuning mainly improves task behavior, whereas the adapter provides the activation-grounded factual and semantic information needed for faithful explanations. 

---
# On the Limits of Model Merging for Multilinguality in Pre-Training 

**Authors**: Seth Aycock, Fedor Vitiugin, Aleksandr Umnov, Christof Monz, Khalil Sima'an  

**Link**: [PDF](https://arxiv.org/pdf/2605.25846)  

**Abstract**: Endowing models with consistent multilingual performance can be achieved by mixing pre-training data, or post-training approaches such as language-specific model merging. In this work, we test whether merging can be applied to monolingually pre-trained models. We conduct a controlled study on the efficacy of mixed, merged, and monolingual pre-training setups. We find that while monolingual pre-training results in strong in-language performance, merging any combination of monolingual models leads to performance collapse due to interference. Our analysis suggests representational similarity is a prerequisite for model merging. We therefore conclude that the flexibility of merging in fine-tuning does not extend trivially to language-specific pre-training. 

---
# From Facts to Insights: A Persona-Driven Dual Memory Framework and Dataset for Role-Playing Agents 

**Authors**: Rongsheng Zhang, Ruofan Hu, Weijie Chen, Jiji Tang, Junnan Ren, Wanying Wu, Xunuoyan Chen, Tangjie Lv, Tao Jin, Zhou Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.25693)  

**Abstract**: While role-playing agents excel in short-term interactions, long-term conversations overwhelm context windows, motivating external memory frameworks. Current systems typically rely on persona-agnostic summarization, which records facts without persona-specific interpretation, yielding generic responses that compromise persona fidelity. To bridge this gap, we introduce RoleMemo, a dataset featuring four reasoning tasks where the factual fragments must be interpreted through the persona to reach the correct answer. Evaluation on RoleMemo exposes critical limitations of persona-agnostic frameworks. We thus propose DualMem, which decouples memory into two streams: factual cognition and persona-conditioned insight. Trained through Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL), our framework with a 4B-parameter model outperforms zero-shot persona-agnostic frameworks powered by DeepSeek-V3.2 for sustained persona fidelity. Our resources are available at this https URL. 

---
# Testing the Deliteralization Hypothesis in Human and Machine Translation 

**Authors**: Malik Marmonier, Rachel Bawden, Benoît Sagot  

**Link**: [PDF](https://arxiv.org/pdf/2605.25686)  

**Abstract**: The recent shift from dedicated NMT systems to general-purpose LLMs has reshaped machine translation, with LLMs reported to produce more fluent, less literal output than their predecessors. We test whether this shift extends to the deliteralization hypothesis, the long-standing claim from translation studies that translations become progressively less literal as they are drafted and revised. Using the WMT24++ dataset, we compare the literality of human translations and post-editions to that of two NMT systems and six LLMs across 54 language pairs and three tasks: direct translation, iterative self-revision, and post-editing of human drafts. Literality is measured via a validated Synthetic Literality Index built from six heuristics. We find that (i) human translations remain significantly less literal than those of all tested MT systems, though recent LLMs narrow the gap; (ii) when prompted to iteratively revise their own output, LLMs deliteralize monotonically, providing the first evidence that the hypothesis applies natively to LLM generation; and (iii) as post-editors, LLMs invert the revision triggers of human post-editors, tolerating literal drafts and targeting idiomatic human formulations for revision. 

---
# Selective Latent Thinking: Adaptive Compression of LLM Reasoning Chains 

**Authors**: Hui Xie, Jie Liu, Ziyue Qiao, Joaquin Vanschore  

**Link**: [PDF](https://arxiv.org/pdf/2605.25745)  

**Abstract**: Explicit chain-of-thought (CoT) reasoning substantially improves the reasoning ability of large language models (LLMs), but incurs high inference cost due to lengthy autoregressive traces. Existing latent reasoning methods offer a promising alternative, yet they often treat reasoning as uniformly compressible, causing precision-critical intermediate steps to be overly compressed and thereby degrading reasoning accuracy. In this work, we propose Selective Latent Thinking (SLT), a framework that selectively compresses redundant reasoning spans into latent representations while preserving precision-critical spans as explicit CoT within the same reasoning trajectory. Specifically, SLT first uses a lightweight decoder to anticipate a short upcoming reasoning span, and then applies confidence-based gating to determine the longest span that can be reliably compressed. The accepted span is encoded into a compact latent representation to improve reasoning efficiency, while uncertain or precision-critical reasoning remains in explicit CoT form to preserve accuracy. To learn this selective compression policy, SLT adopts a three-stage training strategy that combines span-level latent compression, reliability-aware future reasoning prediction, and trajectory-level reinforcement learning to optimize the trade-off between answer correctness and reasoning cost. Extensive experiments across four mathematical reasoning benchmarks demonstrate that SLT achieves 22.7\% higher accuracy than latent reasoning baselines at comparable compression ratios, while reducing reasoning chain length by 58.4\% with only 2.8\% accuracy degradation compared to explicit CoT,Our code can be found in this https URL. 

---
# PowLU: An Activation Function for Stable Pre-Training of LLMs 

**Authors**: Peijie Jiang, Yuqi Feng, Cunyin Peng, Qian Zhao, Jia Liu, KunLong Chen, Zhiqiang Zhang, Jun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.25704)  

**Abstract**: In contemporary large language models (LLMs), the swish-gated linear unit (SwiGLU) activation function is widely adopted to regulate the information flow and introduce non-linearity. For large positive inputs, SwiGLU approximates the quadratic function $x^2$, providing strong nonlinearity and expressive capacity. However, this property also causes numerical instability as the input or model scale increases, particularly in low-precision LLM training. The main reason is its approximate quadratic amplification, which enlarges the output range and exacerbates outliers. To address this issue, we propose a stable activation function, Power Linear Unit (PowLU), for large-scale LLM pre-training. Specifically, PowLU employs a rational power function to achieve adaptive nonlinearity, thereby improving representation ability and enabling stable training in spike regions. Moreover, we provide theoretical justification for several key properties of PowLU. Scaling law experiments confirm that the performance is consistent across model sizes, and further experimental results with the Ling architecture (7.9B and 124B total parameters) demonstrate that PowLU achieves competitive results against SwiGLU and SwiGLU-Clip in large-scale training of LLMs. In addition, the experimental results also show that PowLU effectively improves the scalability of the large-scale training of LLMs. 

---
# Double Triangle Annotation: A Scalable Human-in-the-Loop Framework for High-Precision Historical Document Annotation 

**Authors**: Yi Ren  

**Link**: [PDF](https://arxiv.org/pdf/2605.25781)  

**Abstract**: Evaluating structured-information extraction from historical documents at scale requires high-precision ground-truth annotations, yet traditional manual labeling is expensive and fully automated pipelines built on large language models are prone to hallucination. We propose Double Triangle Annotation, a two-layer human-in-the-loop framework that leverages cross-model consensus to automate the majority of annotation work while ensuring high-precision outputs. In the first layer, two architecturally independent Multimodal Large Language Models annotate each document in parallel; when they agree, the label is auto-accepted, and disagreements are routed to a human jury. A second layer cross-checks two such systems against each other, escalating residual conflicts to a domain expert. The framework rests on a single assumption -- error independence between models -- requires no distributional priors or task-specific calibration, and becomes more autonomous as model capability improves. On the Guides Rosenwald, a corpus of French medical directories spanning 1887-1906, the framework achieves a final Word Error Rate of 0.003. Applied at scale, model consensus auto-accepts over 85% of 13,595 fields. We release the resulting benchmark -- the first structured-extraction ground truth for the Rosenwald Guides -- to support future work on historical document processing. 

---
# Llamion Technical Report 

**Authors**: Kisu Yang, Yoonna Jang, Hyeonseok Moon, Hwanseok Jang, Taewoo Lee, Hyungjin Lee, Jeseung Lee, Juhyoung Park, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2605.25676)  

**Abstract**: We release Llamion, a family of 14B-parameter open-weight language models obtained by transforming Orion-14B into the standardized Llama-family architecture. The transformation is performed by Efficient Knowledge Preservation for Transformation (KEPT), a recipe that combines (i) Normal Parameter Mapping (NPM) for unchanged modules, (ii) Optimized Parameter Mapping (OPM), a training-free LayerNorm-to-RMSNorm initialization we prove optimal under the near-zero-mean activation regime induced by weight decay, and (iii) Cross-architecture Knowledge Distillation (XKD), an equal-size frozen-teacher distillation that aligns the converted model's outputs with the source model's on any reasonable input distribution. Llamion recovers Orion's behaviour on H6, MT-Bench, and KoMMLU with only ~123M tokens on a single A100 in four days; Llamion-Base reaches 66.87% on KoMMLU, exceeding the next-best entry of the Open Ko LLM Leaderboard by >7.0 absolute points at submission time. Capabilities entirely absent from the transfer corpus (Python programming and 200K-token context handling) survive the architectural transition intact. We release three checkpoints (Base, Chat, LongChat) that load with trust_remote_code=False in the Hugging Face Transformers library. 

---
# Trait-Aware Policy Optimization for Autoregressive Multi-Trait Essay Scoring 

**Authors**: Zhengyang Wang, Sanwoo Lee, Jiaxin Wang, Chenxi Miao, Weikang Li, Yunfang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.25731)  

**Abstract**: Multi-trait essay scoring aims to provide fine-grained evaluation of writing quality across multiple dimensions. However, how to effectively post-train autoregressive scoring models remains underexplored. In this paper, we propose Trait-Aware Policy Optimization (TAPO), a post-training framework tailored to autoregressive multi-trait scoring. Our method decomposes rewards along both the sample and trait dimensions, combining global scoring consistency, trait-level accuracy, format validity, and inter-trait dependency preservation. In addition, we enhance supervised fine-tuning with enhanced prompts, allowing the model to internalize trait semantics before preference optimization. Experiments across multiple backbone models show that our method consistently improves multi-trait scoring performance over supervised fine-tuning and scalar-reward optimization baselines, demonstrating the effectiveness and transferability of trait-aware post-training for essay scoring. 

---
# Does Continued Pretraining on a Learner Corpus Improve Automated Essay Scoring on English Proficiency Tests? Evidence from EFCAMDAT 

**Authors**: Duy Anh Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2605.25924)  

**Abstract**: Recent automated essay scoring (AES) studies increasingly use pretrained transformer models, but these models are usually pretrained on general-domain English and may under-represent second-language learner writing. This study investigates whether domain-adaptive continued pretraining (DAPT) on the EFCAMDAT learner corpus improves transformer-based AES for English proficiency tests. We apply DAPT to three transformer encoders and evaluate them on FCE and IELTS in both in-domain scoring and few-shot cross-dataset transfer. Full-corpus DAPT produces mixed results across models, datasets, and metrics. Further analyses suggest that these mixed effects are partly explained by mismatches in proficiency, genre, and communicative purpose between EFCAMDAT and the downstream datasets. A proficiency-based ablation shows that targeted DAPT using CEFR-aligned subsets improves downstream scoring more reliably than full-corpus DAPT, especially for FCE with B1--B2 data. However, these gains do not consistently improve cross-dataset transfer. Overall, the findings suggest that continued pretraining on a learner-writing corpus can benefit in-domain AES for English assessment when the pretraining data is sufficiently aligned with the downstream assessment settings. However, it does not automatically improve transferability across different English proficiency test datasets. 

---
# A Two-Phase Stability Study of LLM Judges and Bar Council Examiners on Thai Bar-Exam Free-Form Essays 

**Authors**: Pawitsapak Akarajaradwong, Wuttikrai Lertprasertphakorn, Chompakorn Chaksangchaichot, Sarana Nutanong  

**Link**: [PDF](https://arxiv.org/pdf/2605.25652)  

**Abstract**: Free-form legal essay evaluation in NLP treats expert inter-rater stability as a single ceiling number, and treats LLM-judge agreement with that ceiling as evidence of judge stability. We test both assumptions on the Thai bar examination through an identical-inputs protocol: three Bar Council-trained examiners (A, B, C) and a 26-LLM judge panel score the same 15 cross-graded answers from the same four inputs (question, official Bar Council grading regulation, gold answer, candidate answer). The headline finding is asymmetric. On 10 of 15 cells where the rubric prescribes both axes, all 29 raters converge in a tight band: panel agreement is universal. On the remaining 5 cells where the rubric does not prescribe how to grade a correct final answer that omits a decisive statutory citation, the human panel splits between two coherent readings (B/C majority at the upper rubric band, score $6$--$8$; A minority at the lower band, score $1$--$2$). The LLM judge population does not split symmetrically: 22 of 26 LLMs score in or near B/C's contested band, 3 sit in the regulation-silent middle gap, and only 1 (GPT-5.4 Nano) approaches A's band without consistently scoring within it. \emph{Zero LLMs in our 26-judge panel reproduce the minority human reading on the contested cells.} The B/C-direction cluster spans every model size, vendor, and price tier we tested. An instrumented three-LLM anchor sub-panel (Claude 4.6 Opus, Gemini 3.1 Pro, GPT-5.4 Pro) carries determinism probes, input ablations, and bootstrap CIs, and reaches anchor panel $\alpha = 0.77$ on the 15 cells against human-panel $\alpha = 0.36$. The high LLM-panel $\alpha$ reflects systematic convergence on the majority reading rather than balanced reproduction of both readings; a benchmark that selects its LLM judge by maximising agreement with a human reference panel will inherit this asymmetry by construction. 

---
# Iterate Until Retrieved: Factual Nugget Optimization for Discoverable Continual Corrections in Agentic RAG 

**Authors**: Moshe Hazoom, Gal Patel, Alon Talmor, Tom Hope  

**Link**: [PDF](https://arxiv.org/pdf/2605.25641)  

**Abstract**: Agentic retrieval-augmented generation (RAG) systems in complex B2B (business-to-business) settings may often receive free-form response feedback. Rather than generic feedback signals such as style, preference, or overall response quality, we focus on actionable factual corrections. We identify these instances and convert them into compact knowledge-base entries, which we call factual nuggets. We introduce Iterative Nugget Optimization (INO), an index-time optimization method that uses the production agentic RAG as a test harness: it creates an initial nugget, probes it with the triggering query and paraphrases, reflects over failed retrieval and answer traces, and revises the nugget until it is discoverable. We evaluate INO with two production B2B knowledge-assistance agents across multiple companies that use our system: a product support agent that answers questions over company-specific knowledge bases, and a support ticket agent that assists support engineers. INO consistently improves results over baselines in terms of discoverability and usage of factual corrections, in automated and human evaluations. 

---
# Beyond Literal Translation: Evaluating Cultural Effectiveness in Social Media UGC 

**Authors**: Linjuan Wu, Ruiqi Zhang, Xinze Lyu, Ye Guo, Daoxin Zhang, Zhe Xu, Yao Hu, Yixin Cao, Yongliang Shen, Weiming Lu  

**Link**: [PDF](https://arxiv.org/pdf/2605.25626)  

**Abstract**: Social media platforms enable large-scale cross-lingual communication, but translating user-generated content (UGC) remains challenging due to its informal style, cultural references, and interaction-based expressions. While recent LLMs have improved translation quality, existing benchmarks and metrics often fail to capture whether translations convey intended meaning and cultural resonance in real-world settings. In this work, we introduce CULTURE-MT, a benchmark for social media translation that focuses on both CULtural Transmission and UGC-specific emotion REsonance. CULTURE-MT consists of 1,002 UGC notes across 14 domains, categorized into four types based on culture-loaded symbols and linguistic style features. We also construct UGC-oriented training data to fine-tune Qwen3-8B and Qwen3-32B as baselines. We propose cultural effectiveness as a new evaluation criterion, focusing on expression accuracy and cultural adaptability. Testing 15 models, including the baselines, we find that traditional metrics fail to capture cultural effectiveness. We also observe that cultural effectiveness on base LLMs correlates with model size. Our work provides a comprehensive evaluation system for UGC translation models and will offer an open evaluation platform to advance research in this area. We release the CULTURE-MT benchmark and provide an online leaderboard where submitted translation results can be evaluated by our trained JUDGER. 

---
# DVAO: Dynamic Variance-adaptive Advantage Optimization for Multi-reward Reinforcement Learning 

**Authors**: Guochao Jiang, Jingyi Song, Guofeng Quan, Chuzhan Hao, Guohua Liu, Yuewei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.25604)  

**Abstract**: Reinforcement Learning has become a standard paradigm for aligning Large Language Models with human intent and task requirements. While Group Relative Policy Optimization offers an efficient, value-model-free alternative to Proximal Policy Optimization, adapting it to real-world multi-reward settings remains challenging. Standard scalarization practices, such as Reward Combination and Advantage Combination, suffer from significant drawbacks: Reward Combination frequently generates advantages with excessively large squared magnitudes that lead to training instability, while Advantage Combination relies on static hyperparameters and ignores cross-objective correlations. To address these limitations, we propose Dynamic Variance-adaptive Advantage Optimization (DVAO), which dynamically adjusts combination weights based on the empirical reward variance of each objective within a rollout group, effectively up-weighting objectives with a stronger learning signal while suppressing noisy ones. We mathematically prove that DVAO maintains bounded advantage magnitudes for stable training and introduces a self-adaptive cross-objective regularization mechanism. Extensive experiments on mathematical reasoning and tool-use benchmarks using Qwen3 and Qwen2.5 models demonstrate that DVAO significantly outperforms baseline methods, achieving a superior multi-objective Pareto frontier and robust training stability. 

---
# StreamProfileBench: A Benchmark for Fine-Grained User Profile Inference in Real-World Streaming Scenarios 

**Authors**: Sizhe Wang, Feiyu Duan, Juelin Wang, Liwen Zhang, Feiyu Duan  

**Link**: [PDF](https://arxiv.org/pdf/2605.25758)  

**Abstract**: Large Language Models (LLMs) have reshaped user profiling, yet current evaluations mainly focus on static data snapshots. This paradigm overlooks the reality of personalized systems, where User-Generated Content (UGC) arrives continuously and fine-grained profile evolve rapidly. To bridge this gap, we introduce StreamProfileBench, a large-scale benchmark for fine-grained streaming user profiling. We formalize streaming user profiling as a continuous state maintenance task and curate a highly authentic dataset comprising over 120,000 UGC posts from 7,000+ real users across five diverse platforms. By leveraging the temporal correlation of user interests, we further propose a novel, annotation-free evaluation framework. Extensive experiments across 14 leading LLMs reveal that continuous profile updating remains an open challenge. Models exhibit a systemic conservative bias, over-retaining past interests while failing to recognize interest decay. Ablation experiments further validate the practical utility and necessity of the streaming paradigm. 

---
# Is Inference Mediated by Distinct Semantic Structures in LLMs? A Mechanistic Interpretation 

**Authors**: Nura Aljaafari, Marco Valentino, André Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2605.25520)  

**Abstract**: Predicting a label correctly does not necessarily require representing the operation that produces it. Transformer representations are known to carry label-level information, but whether they encode semantic operations producing those labels is unclear. We investigate this in Natural Language Inference using controlled premise-hypothesis pairs that differ by a single semantic transformation. Using layer-wise activations, we estimate operation-level subspaces via SVD and test their causal relevance through activation steering in four open-weight decoder models. Transformation effects are decodable with $84.8$-$99\%$ accuracy and occupy partially distinct but overlapping subspaces, exceeding random-subspace baselines. Steering experiments show that these directions causally influence predictions, though steerability varies across models; cross-operation steering further reveals structured interference and a dissociation between subspace selectivity and cross-operation independence. These findings indicate that the models encode not only that a hypothesis relates to a premise but also, in part, how it does so, implying that mechanistic analysis and control should operate at the level of semantic operations rather than predicted labels alone. 

---
# Reinforcement Learning from Denoising Feedback 

**Authors**: Qi He, Huan Chen, Ya Guo, Huijia Zhu, Yi R. Fung, Baojian Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.25638)  

**Abstract**: Policy loss estimation remains a fundamental and long-standing challenge in reinforcement learning (RL) for diffusion language models (dLLMs). We introduce Reinforcement Learning from Denoising Feedback (RLDF), a novel training paradigm that leverages feedback obtained from rollout and training processes to facilitate accurate and efficient policy loss estimation. To balance the trade-off between computational efficiency and estimation effectiveness, RLDF optimizes the model toward the clipped clean state $\hat{x}_0$ from intermediate noisy states $x_t$, combined with weighted timestep sampling over $t$. Extensive experiments demonstrate that RLDF achieves consistent and substantial improvements in both performance and generalizability across two representative dLLM architectures, LLaDA and Dream, on multiple reasoning benchmarks. Our work lays a principled foundation for scalable reinforcement learning in diffusion language models. We build Drift, a training framework for dLLMs, available at this https URL. 

---
# The Age of Curiosity Meets the Age of AI: Benchmarking Child Safety in Large Language Models 

**Authors**: Samee Arif, Angana Borah, Rada Mihalcea  

**Link**: [PDF](https://arxiv.org/pdf/2605.25510)  

**Abstract**: Children increasingly have access to Large Language Models (LLMs), which may expose them to responses that are developmentally inappropriate or require age-sensitive safety, guidance, and boundaries. Existing LLM safety evaluations largely focus on harmful-content avoidance and do not explicitly target child-facing safety. We introduce KIDBench, a benchmark for evaluating child-facing LLM safety for ages 7--11 using a developmental-psychology-grounded LLM-as-a-Judge rubric. KIDBench contains realistic child queries across ten categories, with single-turn prompts and multi-turn child-actor simulations. We compare no-cues prompts with no child context, implicit-cues prompts that suggest a child speaker, and explicit age instructions. Implicit-cues improve scores by 9--47% across models, while explicit age adds a further 10--30% gain. Cross-lingual and cultural evaluations show uneven safety behavior across languages and country contexts. Multi-turn simulations show that child-facing response quality can degrade by 6--24% from the first to worst turn. Beyond evaluation, we introduce KIDGuardLlama, a child-safety evaluator, and KIDLlama, a child-oriented response model, showing how KIDBench supports safer child-facing AI 

---
# CRPO: Character-centric Group Relative Policy Optimization for Role-aware Reasoning in Role-playing Agents 

**Authors**: Yihong Tang, Kehai Chen, Liang Yue, Benyou Wang, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.25511)  

**Abstract**: Recent advancements in Reinforcement Learning (RL), particularly Group Relative Policy Optimization (GRPO), have significantly enhanced the reasoning capabilities of Large Language Models. However, applying these problem-centric optimization methods to role-playing agents often leads to a loss of character fidelity and style collapse, as they prioritize context-specific utility over persona alignment. To address this, we propose Character-Centric Group Relative Policy Optimization (CRPO), a framework designed to realign RL objectives with the role-playing task. CRPO improves character distinctiveness through three mechanisms: decoupling task logic from stylistic rewards to resolve gradient conflicts, dynamically adapting optimization constraints based on character complexity, and utilizing generic responses as negative baselines to prevent the model from reverting to a common distribution. Extensive experiments demonstrate that CRPO outperforms existing methods in consistency, emotion and others. 

---
# When In-Distribution Gains Fail: Evaluating Weak-to-Strong Reward Models under Preference Shift 

**Authors**: Khoi Le, Tri Cao, Phong Nguyen, Cong-Duy Nguyen, Anh Tuan Luu, Miao Chunyan, See-Kiong Ng, Thong Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2605.25629)  

**Abstract**: Weak-to-strong (W2S) generalization is a promising framework for scalable oversight, yet existing evaluations often test students under matched train--test distributions. Therefore, we study W2S preference learning under zero-shot distribution shift and find that strong students trained on weak preference labels can appear successful in-distribution while failing to transfer across preference datasets. We provide evidence for a representational failure mode in which weak-supervised fine-tuning can pull the strong model toward source-domain features instead of maintaining broadly transferable preference representations. To mitigate this, we propose Representation Anchoring (Anchor), a simple yet effective regularizer that constrains excessive drift from the pretrained strong model's representation space during fine-tuning, while still allowing task-relevant adaptation. Across preference domains, datasets, and model families, Anchor consistently improves out-of-distribution transfer while maintaining competitive in-distribution performance. Together, our evaluation protocol, transfer-aware metrics, and method expose hidden brittleness in current W2S reward modeling and provide a practical path toward more robust preference transfer. 

---
# GeoSVG-RL: Geometry-Aware Reinforcement Learning for Layout-Constrained Text-to-SVG Diagram Generation 

**Authors**: Sifan Li, Yujun Cai, Hongkai Chen, Yiwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.25447)  

**Abstract**: Generating structured, editable diagrams remains a significant challenge for contemporary large language models, despite their proficiency in general-purpose vector code generation. The primary difficulty lies in the structural fragility of the output; minor errors such as misaligned connector endpoints, text labels overlapping borders, or complex layouts drifting beyond the canvas boundaries render the resulting SVG files functionally unusable for professional applications. To address these issues, we introduce GeoSVG-RL, a specialized reinforcement learning framework designed for layout-constrained text-to-SVG generation. Unlike standard training objectives that rely solely on maximizing token-level likelihood, our approach optimizes the policy against explicit, executable geometric feedback. The model first produces a structured layout plan that serves as a geometric contract for the subsequent generation of the SVG code. This code is then rendered through a browser-backed verifier, enabling the calculation of fine-grained rewards across six critical dimensions: rendering validity, canvas fitting, precise anchor placement, text containment, graph consistency, and code cleanliness. We utilize Group Relative Policy Optimization (GRPO) to refine the model, sampling multiple candidates per prompt to facilitate updates based on relative quality. Starting from a supervised warm-start phase on synthetic data, GeoSVG-RL achieves substantial gains in structural reliability, particularly in arrow-anchor accuracy and text-in-box rates. Quantitative evaluations demonstrate that our method consistently outperforms current state-of-the-art systems in local geometric precision and the preservation of graph connectivity, providing a robust pathway toward automated yet reliable technical illustration. 

---
# Harmony in Diversity: Multi-domain Contrastive Policy Optimization for Large Reasoning Models 

**Authors**: Zongji Yu, Wenshui Luo, Yiliu Sun, Hao Fang, Runmin Cong, Chaochao Lu, Chen Gong  

**Link**: [PDF](https://arxiv.org/pdf/2605.25443)  

**Abstract**: Post-training has significantly enhanced the reasoning capability of Large Reasoning Models (LRMs), especially with Reinforcement Learning (RL) like Group Relative Policy Optimization (GRPO). However, GRPO-style RL methods in multi-domain settings often fail to achieve consistent improvements across all domains due to inherent interference in policy optimization. Prior studies on multi-domain RL primarily focus on alleviating cross-domain interference, while often neglecting the pivotal role of knowledge sharing, which we argue is the key to transforming cross-domain interactions from harmful competition into beneficial transfer. To address this limitation, we propose Multi-domain Contrastive Policy Optimization (MCPO), which analyzes the structural relationships among rollouts and promotes cross-domain knowledge sharing and in-domain knowledge consolidation in a contrastive manner. Specifically, for a given prompt, MCPO identifies transferable reasoning trajectories from other domains as positive examples, while treating incorrect rollouts as negative ones. It then encourages consistent representations for positive pairs and pushes negative pairs apart, thereby facilitating knowledge transfer and reducing interference. Moreover, MCPO aligns intra-domain correct rollouts to build a consolidated representation space. In this way, MCPO contrastively learns a harmonious representation space that can accommodate diverse multi-domain knowledge. Empirical results show that MCPO improves the reasoning capabilities of LRMs across multiple domains and even outperforms single-domain training in some cases. Code is available at this https URL. 

---
# LLM-as-a-Reviewer: Benchmarking Their Ability, Divergence, and Prompt Injection Resistance as Paper Reviewers 

**Authors**: Lingyao Li, Junjie Xiong, Changjia Zhu, Runlong Yu, Chen Chen, Junyu Wang, Renkai Ma, Zhicong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2605.25415)  

**Abstract**: Large language models (LLMs) are increasingly used in academic peer review, yet their reliability, alignment with human judgment, and robustness to adversarial attacks remain poorly understood. We present a systematic benchmark of LLM-as-a-Reviewer on 898 papers stratified from NeurIPS and ICLR, evaluating 12 LLMs along three axes: rating calibration, divergence from human reviewers, and resistance to prompt injection embedded via an invisible font-mapping attack. We find that LLMs systematically overrate weaker submissions and diverge from humans in topical emphasis, under-flagging Clarity and over-flagging Reproducibility, while producing reviews two to three times longer with lower lexical diversity and a more standardized vocabulary. Prompt injection remains highly effective. Simple hidden instructions can promote low-scoring papers to acceptance-level ratings in a substantial fraction of cases, with effectiveness varying sharply across model families. While LLMs offer utility in structuring evaluations, their integration into peer review requires safeguards against both intrinsic biases and adversarial risks. 

---
# HyLaT: Efficient Multi-Agent Communication via Hybrid Latent-Text Protocol 

**Authors**: Xinyi Mou, Siyuan Wang, Zejun Li, Yulan He, Zhongyu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2605.25421)  

**Abstract**: Communication protocol design is a central challenge in large language model-based multi-agent systems. Existing single-channel approaches face an inherent communication trilemma: text-based methods are interpretable but verbose, while latent-space methods are efficient but opaque and limited to unidirectional workflows. Inspired by multi-channel communication theory, we propose HyLaT, a hybrid latent-text communication protocol that transmits elaborate cognitive signals through a latent channel for efficiency, while expressing concise critical signals in natural language to preserve interpretability and precision. We introduce a two-stage training framework combining single-agent hybrid generation learning and multi-agent interactive co-training, enabling agents to generate and interpret hybrid messages across multiple rounds of interaction. Experiments demonstrate that HyLaT reduces communication overhead significantly while maintaining competitive task performance, with strong generalization and robustness across diverse settings. 

---
# Retrieval as Reasoning: Self-Evolving Agent-Native Retrieval via LLM-Wiki 

**Authors**: Haoliang Ming, Feifei Li, Xiaoqing Wu, Wenhui Que  

**Link**: [PDF](https://arxiv.org/pdf/2605.25480)  

**Abstract**: LLM agents require retrieval to behave less like one-shot context fetching and more like reasoning: searching, reading, traversing, and deciding when evidence is sufficient. However, Retrieval-Augmented Generation (RAG) typically organizes external knowledge as flat chunks retrieved by embedding similarity, exposing a retrieval-as-lookup interface that is poorly aligned with tool-using agents. We propose LLM-Wiki, an agent-native retrieval system that operationalizes the Retrieval-as-Reasoning paradigm by treating external knowledge as a compilable, composable, and self-evolving structure rather than a static retrieval index. LLM-Wiki compiles documents into structured Wiki pages with bidirectional links, exposes search, read, and link-following operations through standard tool-calling interfaces, and introduces an Error Book for persistent structural and semantic self-correction. On HotpotQA, MuSiQue, and 2WikiMultiHopQA, LLM-Wiki outperforms seven baselines, including HippoRAG 2, LightRAG, and GraphRAG, with gains of 2.0-8.1 F1 points over the strongest graph-based baseline and larger gains over Dense RAG. On AuthTrace, LLM-Wiki achieves the best overall accuracy, with especially strong gains on multi-document structured queries, showing that compilation-based knowledge organization generalizes beyond chain-style multi-hop reasoning. 

---
# GeoMathCode: Understanding Interleaved Math-Code Reasoning for Geometry Problem Solving 

**Authors**: Yingji Zhang, Yong Dai, André Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2605.25384)  

**Abstract**: Mathematical reasoning is a hallmark of human intelligence, requiring logical deduction, symbolic manipulation, and abstract thinking. Recent multimodal large language models (MLLMs) have demonstrated strong performance on geometry problems through multi-step reasoning. To better emulate human problem-solving, intermediate steps can incorporate auxiliary visual constructions, such as additional lines or points, which improve geometric interpretation and educational clarity. In this work, we introduce the GeoMathCode, where programmatic representations serve as intermediate visual outputs. We further conduct an in-depth analysis of the underlying reasoning geometry. Experimental results show that reasoning and code generation steps can be disentangled in the latent space, while supervised fine-tuning (SFT) makes the reasoning manifold more structured and informative. Moreover, hierarchical syntactic code structures emerge as disentangled latent subspaces, and contain more mathematical symbolic information than visual representations. 

---
# Fine-Tuning Over Architectural Complexity: Broad-Coverage PII Detection on PIIBench with DeBERTa 

**Authors**: Pritesh Jha  

**Link**: [PDF](https://arxiv.org/pdf/2605.25816)  

**Abstract**: Personally identifiable information (PII) detection systems are frequently trained within narrow source or domain boundaries, limiting coverage when deployed on heterogeneous text. We study model fine-tuning on a corrected multi-source PIIBench preparation spanning 82 retained entity types across ten source datasets. We evaluate three DeBERTa-based approaches: direct token classification fine-tuning, a source-conditioned hierarchical model (SC+H), and a three-phase curriculum extension (SC+H+Curr). Against eight published comparator systems on a reproducible 5,000-record held-out subset (test_5k), direct fine-tuned DeBERTa achieves F1 0.6476, while SC+H and the curriculum variant achieve 0.5899 and 0.2772 respectively; the strongest published comparator reaches only 0.1723. Because validation initially favoured SC+H, we perform a final streamed evaluation on the complete 100,002-record held-out split. Direct fine-tuning remains superior, achieving F1 0.6455 versus 0.5894 for SC+H. Entity-level analysis shows that direct fine tuning wins 54 of 82 fine entity types and all ten coarse groups by support-weighted entity F1, while SC+H retains localised advantages on 28 types. The results indicate that diverse task-specific training data and a simple weighted cross-entropy objective contribute more to broad-coverage PII detection than the tested architectural and curriculum complexity. 

---
# EfficientGraph-RAG: Structured Retrieval-State Management for Cross-Task Retrieval-Augmented Generation 

**Authors**: Miaohe Niu, Lianlei Shan, Zhengtao Yu, Jingbo Zhu, Tong Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2605.25379)  

**Abstract**: Retrieval-augmented generation (RAG) has become the standard way to ground large language models in external knowledge, but many systems still organize evidence as flat chunks and retrieve it through largely unstructured search. This weak structure becomes a bottleneck for complex retrieval: the system must decide where to search, how to move from coarse topics to entity-relation evidence, which evidence has been verified, and which intermediate artifacts can be reused. We define these intermediate variables as a retrieval state and study RAG as structured state management. EfficientGraph-RAG makes this state explicit through three coupled mechanisms: TAM defines a typed hierarchical state space over evidence, MARS updates and verifies the state through role-specialized agents, and SMP stores reusable state under hierarchy-aware access control. Using one shared framework configuration, EfficientGraph-RAG ranks first on the reported answer-quality metrics averaged over the three evaluated LongBench retrieval-style subsets, matches the strongest agentic baseline on HotpotQA EM while reducing large-model token usage by $3.51\times$, and provides a low-token DocVQA result among retrieval-organizing cross-modal methods. Component analysis shows role-specific mechanisms: MARS is the main answer-quality driver, TAM supplies the typed traversal state and Adaptive Routing signal, and SMP enables corpus-dependent reuse, with cross-query cache hit rates ranging from 3.77% to 23.18%. 

---
# Proactive for Uncertainty: Cause-Aware Error Diagnosis and Interactive Clarification for Spoken Dialogue Systems 

**Authors**: Yizhou Peng, Ziyang Ma, Changsong Liu, Yi-Wen Chao, Xie Chen, Eng Siong Chng  

**Link**: [PDF](https://arxiv.org/pdf/2605.25404)  

**Abstract**: Cascaded Automatic Speech Recognition -- Large Language Model (ASR-LLM) pipelines remain popular for industrial Spoken Dialogue Systems (SDS), primarily because their decoupled design ensures perceptual verifiability. However, cascaded systems suffer from error propagation, as transcription failures inevitably cascade to subsequent components, thereby degrading the final interaction quality. Although ASR confidence scores offer a simple filter for unreliable inputs, this approach is fundamentally limited because it typically fails to detect deletion errors or to distinguish between acoustic (inability to hear clearly) and linguistic (inability to understand) mismatches, both of which require targeted recovery strategies. In this paper, we propose a cause-aware error recovery paradigm that fundamentally rethinks robustness in SDS. Unlike traditional confidence filtering, we introduce a suite of small precision-focused detectors that exploit deep ASR latent representations to disentangle token-level errors into perception, comprehension, and deletion failures. This fine-grained diagnostic intelligence empowers the LLM to orchestrate targeted, multi-turn clarification strategies, effectively transforming ambiguous signals into seamless user interactions. Experimental results validate the precision of our approach, which more than doubles the recall on domain-shift errors (57.96% vs. 23.66%) compared to baselines. Crucially, this diagnostic precision yields up to a 30% reduction in WER and a 17% improvement on the downstream task across diverse accents, distortions, and domains. 

---
# MATO: Multi-objective Personalized Alignment with Test-time Optimization for Large Language Models 

**Authors**: Linhao Luo, Thuy-Trang Vu, Van-Anh Nguyen, Junae Kim, Gholamreza Haffari, Dinh Phung  

**Link**: [PDF](https://arxiv.org/pdf/2605.25342)  

**Abstract**: Aligning large language models (LLMs) with diverse and multifaceted user preferences is a fundamental challenge in personalized AI systems. Existing multi-objective alignment methods either rely on costly training or require pre-trained reward models for each preference, making it difficult for them to adapt to evolving preferences. Prompt-based personalization offers a training-free alternative, but prompting alone often provides limited steerability, as LLMs may overemphasize or overlook certain preferences and fail to give users reliable control over the relative importance of different objectives when conflicts arise, leading to suboptimal alignment. In this paper, we introduce MATO, a training-free framework for Multi-objective personalized Alignment with Test-time Optimization. MATO formulates personalization as a test-time optimization problem that steers the relative importance of multiple objectives through controllable weights during decoding, without modifying model parameters or requiring external reward models. Specifically, a reward discovery module recovers preference rewards directly from the backbone LLM for diverse objectives specified in natural language, while a weight optimization module dynamically adjusts objective weights based on the user's initial preferences and the partially generated response to balance competing objectives during generation. The resulting rewards and weights jointly guide an online optimization procedure over the token distribution, enabling better alignment with the target objectives. Extensive experiments across multiple datasets and backbone LLMs show that MATO consistently outperforms strong baselines, achieving Pareto-improving multi-objective alignment and stronger steerability. These results highlight test-time optimization as a promising direction for scalable, controllable, and model-agnostic personalized alignment. 

---
# Learning to Route Languages for Multilingual Policy Optimization 

**Authors**: Geyang Guo, Hiromi Wakaki, Yuki Mitsufuji, Alan Ritter, Wei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.25360)  

**Abstract**: Large language models~(LLMs) are trained on heterogeneous multilingual corpora, yet existing policy optimization methods often implicitly restrict each training question to a single response language or rely on a fixed dominant language for supervision. We propose language-routed policy optimization (LRPO), an online policy optimization framework that treats language as a selectable variable. LRPO elicits multilingual rollouts for each training question and integrates their relative quality into preference-based policy updates, increasing the diversity and informativeness of training signals under the fixed rollout budget. To adaptively determine which languages to explore during reinforcement learning, we introduce a trainable language router formulated as a multi-armed bandit, balancing exploration of underutilized languages with exploitation of more informative ones. Extensive experiments show that LRPO consistently improves multilingual performance, demonstrating that adaptive language routing enables effective cross-lingual knowledge exploitation for training. We release all the resources at this https URL. 

---
# Knowing but Not Showing: LLMs Recognize Ambiguity but Rarely Ask Clarifying Questions 

**Authors**: Jinyan Su, Claire Cardie  

**Link**: [PDF](https://arxiv.org/pdf/2605.25284)  

**Abstract**: User queries are often underspecified and may admit multiple valid interpretations. Rather than silently making assumptions about the user's intent, a helpful assistant should surface such ambiguity by asking a clarifying question. Doing so requires two abilities: recognizing that a query is ambiguous, and acting on that recognition by seeking clarification instead of answering directly. To study these abilities, we evaluate models on ambiguous, unambiguous, and disambiguated questions in three settings: standard question answering, explicit ambiguity judgment, and behavioral analysis, where a judge model classifies responses as direct answers, refusals, or clarifying questions. We find a clear gap between recognition and behavior: models often identify ambiguity when explicitly asked to judge it, yet in the QA setting they overwhelmingly default to direct answers. Retrieved context further widens this gap by improving answerability while making models even less likely to ask clarifying questions. 

---
# Inference Time Optimization with Confidence Dynamics 

**Authors**: Yu Wang, Minghao Liu, Jiayun Wang, Jinrui Huang, Ankit Shah, Wei Wei  

**Link**: [PDF](https://arxiv.org/pdf/2605.25244)  

**Abstract**: Inference time optimization techniques, such as repeated sampling, have significantly advanced the reasoning capabilities of Large Language Models (LLMs). However, the critical role of model uncertainty remains largely underexplored in these optimization strategies. In this paper, we investigate the dynamics of confidence along reasoning trajectories and for first time reveal a surprising and unique pattern: correct answer traces tend to exhibit confidence improvement over time (positive confidence gain), while incorrect traces show attenuated or declining confidence as reasoning proceeds. Based on this observation, we propose Confidence Dynamic Gain (CDG) based voting, which incorporates how the confidence trajectory of the response evolves along the reasoning chain. Experiments across four open-source architectures (DeepSeek-R1, gpt-oss, Gemma-3, Qwen-QwQ) on the AIME24/25, HMMT25, and BRUMO25 benchmarks demonstrate that CDG yields a significant performance boost over baselines. These results demonstrate that our method provides a robust discriminative signal for improving answer selection in LLM reasoning. We also provide theoretical insights for this phenomenon. Code will be released at this https URL. 

---
# From Automation to Collaboration: Human-in-the-Loop Methods for Safe and Trustworthy NLP 

**Authors**: Most. Sharmin Sultana Samu, MD. Tanvir Ahmed Seum, Md. Rakibul Islam  

**Link**: [PDF](https://arxiv.org/pdf/2605.25226)  

**Abstract**: Large language models are widely deployed in high-stakes NLP tasks, yet risks such as bias, hallucination, adversarial vulnerability and unreliable generalization remain. Probe-based auditing reveals inconsistencies in model behavior. Adversarial text generation uncovers robustness gaps, especially in lower-resourced languages with limited benchmarks. Enterprise text-to-SQL settings expose the difficulty of validating outputs over private and large-scale databases. Human supervision is essential for probe validation, adversarial verification and domain-specific annotation, but it is costly and hard to scale. This survey examines recent human-in-the-loop methods that shift NLP from automation toward collaboration for safety and trustworthiness. We review how human expertise supports auditing, robustness evaluation, data construction and model steering. Our findings highlight gaps in scalable probing, sustainable robustness benchmarks, low-resource settings and governance of private systems. We outline practical research directions for adaptive auditing, collaborative evaluation and accountable deployment. 

---
# Tool-Call Dependency Structure is Linearly Decodable in LLM Agent Residual Streams 

**Authors**: Tianda Sun, Dimitar Kazakov  

**Link**: [PDF](https://arxiv.org/pdf/2605.25310)  

**Abstract**: Tool-using LLM agents produce trajectories whose calls form a directed dependency graph: earlier tool outputs supply arguments to later calls. Whether this execution structure is represented inside the model is unknown; prior structural probes have targeted static code or chain-of-thought text, not an agent's run-time call graph. A low-capacity edge probe on the residual stream of Qwen3-32B decodes the tool-call dependency graph well above both a Hewitt--Liang random-label control and a positional baseline. A counterfactual contrast between value corruption and structural perturbation indicates the signal tracks abstract topology rather than identifier values, and replicates under an independent, non-substring oracle. The non-positional component replicates on three further interactive multi-hop benchmarks and attenuates as call order alone becomes a sufficient proxy for dependency, vanishing in single-shot planning. Per-layer activation patching shifts the probe at a later, non-patched boundary, evidence that the representation propagates rather than passively reads out, though the realised tool call does not move. To our knowledge this is the first structural probe of an LLM agent's runtime tool-call dependency graph. Our claims concern representation, not behavioural control, and span two model families and one primary domain. 

---
# GroupTravelBench: Benchmarking LLM Agents on Multi-Person Travel Planning 

**Authors**: Xiang Cheng, Yulan Hu, Lulu Zheng, Zheng Pan, Xin Li, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.25200)  

**Abstract**: Travel planning is a realistic task for evaluating the planning and tool-use abilities of LLM agents. However, existing benchmarks typically assume only a single user, thereby avoiding one of the most challenging aspects of real-world scenarios: an agent's ability to identify and resolve conflicts among multiple users. To address this gap, we introduce \textbf{GroupTravelBench}, the first benchmark for \textbf{multi-user, multi-turn} travel planning. Based on real user profiles, POI data, and ticket price data, we synthesize 650 tasks and divide them into three difficulty levels. Beyond standard abilities in single-user itinerary planning, such as multi-step reasoning and tool use, our benchmark further evaluates three key capabilities required for travel agents: \emph{(i) elicitation} -- proactively engaging in multi-turn dialogue to gather preferences from each user; \emph{(ii) coordination} -- resolving conflicts among users through compromise or subgrouping strategies; and \emph{(iii) planning} -- searching for travel plans that maximize overall group utility while maintaining fairness and feasibility. To simulate real-world conversational itinerary planning while enabling reliable tool use and offline evaluation, we build an interactive sandbox environment with cached real-world tool data. We evaluate a wide range of LLMs and find that even frontier models still show substantial weaknesses in preference coverage and group fairness. \textit{GroupTravelBench} provides a practical and reproducible benchmark for advancing research on LLM agents for real-world travel planning. 

---
# AuthTrace: Diagnosing Evidence Construction in Thematically Dense Single-Author Corpora 

**Authors**: Xiaoqing Wu, Feifei Li, Haoliang Ming, Wenhui Que  

**Link**: [PDF](https://arxiv.org/pdf/2605.25382)  

**Abstract**: Evidence construction systems--chunk retrieval, agent memory, knowledge-graph traversal, and thematic indexing--are evaluated on separate benchmarks with incompatible corpora and metrics, making cross-paradigm diagnosis impossible. We introduce AuthTrace, the first diagnostic benchmark that places all major paradigms on a single corpus and query set by exploiting the dual nature of single-author collections. Built on thematically dense corpora where all texts share style, topic, and vocabulary, AuthTrace provides 2,099 instances with exhaustive gold evidence and a fan-in gradient as the primary diagnostic axis. Comparing eight systems across two QA models, we find that (1) evidence recall--not precision--is the dominant predictor of answer quality (r = 0.96); (2) fan-in exposes paradigm-specific collapse patterns, with flat retrieval degrading 3x faster than structured-evidence systems; and (3) full-context prompting fails uniformly, establishing evidence construction as a necessary capacity beyond raw corpus exposure. 

---
# Re-defining Humor Data Objects for AI Humor Research 

**Authors**: Anna Arnett, Bang Nguyen, Meng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2605.25171)  

**Abstract**: In most existing AI humor research, humor was treated as either "present" or "not present." We explore the concept of humor as a social interaction with context and explanations. During this project, we defined a humor reasoning data object and developed a way to prompt LLMs to generate an explanation of humor effective for general population. We iterated from an earlier prompt to an improved prompt, found that the later version reduced important errors, and then scaled generation to a large number of data objects which have the potential to enable data synthesis and data augmentation for AI humor research. Our main takeaway is that better prompting of an LLM improves humor explanation quality, especially by handling missing context, multi-modality, and transcript issues more carefully. These results establish a strong foundation for future work on AI understanding of humor as social behavior. 

---
# Clarification Is Not Enough: Post-Clarification Answering Remains the Bottleneck in Multi-Turn QA 

**Authors**: Jinyan Su, Jennifer Healey  

**Link**: [PDF](https://arxiv.org/pdf/2605.25204)  

**Abstract**: Pluralistic alignment requires systems to adapt to diverse user values, communication styles, and contextual assumptions. We believe that a foundational prerequisite for such alignment enabling accurate preference elicitation from people when their intent is under-specified or ambiguous. We study the problem of preference elicitation in multi-turn question answering by decomposing the problem into two components: a \textbf{clarification policy}, which decides whether to ask a clarifying question or answer directly, and \textbf{post-clarification answering}, which produces the correct final answer once the missing information is provided. We show, using the PACIFIC benchmark, that supervised fine-tuning rapidly improves the clarification policy, however, final answer accuracy remains substantially lower even when the model takes the correct action. This gap indicates that understanding and correctly interpreting the user's response is the critical gap in multi-turn question-answering systems. 

---
# Faithfulness Metrics Don't Measure Faithfulness: A Meta-Evaluation with Ground Truth 

**Authors**: Yoav Gur-Arieh, Ana Marasović, Mor Geva  

**Link**: [PDF](https://arxiv.org/pdf/2605.25052)  

**Abstract**: Chains of thought (CoTs) have become central in interpreting and auditing behaviors of large language models. Yet growing evidence suggests that these traces often fail to faithfully represent the computations behind a model's predictions. Several faithfulness metrics have been proposed, but whether they indeed measure faithfulness remains unknown. Answering this requires ground-truth labels, which are hard to obtain since internal computations are not directly observable. Consequently, most works proposing metrics report only absolute scores or comparisons to prior metrics, and the few existing benchmarks rely on proxies like plausibility or importance, properties orthogonal to faithfulness that can mislead about whether a CoT can be trusted. We address this challenge by constructing tasks whose outputs reveal which intermediate computations must have produced them, and developing an automated labeling pipeline that yields ground-truth faithfulness labels at both the step and CoT level. Building on this methodology, we present BonaFide, a benchmark of 3,066 labeled CoTs across 13 tasks and 10 models, and use it to conduct the first systematic evaluation of prominent faithfulness metrics. Our experiments show that most metrics perform near chance, exhibit strong prediction biases and degrade on longer CoTs. The best metric reaches only 0.70 AUROC at the CoT level while another reaches 0.59 at the step level, with neither transferring across settings, while entailing prohibitively high computational cost. Our results expose fundamental gaps in current faithfulness evaluation and call for the development of more reliable and efficient metrics. 

---
# Better, Faster: Harnessing Self-Improvement in Large Reasoning Models 

**Authors**: Qihuang Zhong, Liang Ding, Juhua Liu, Bo Du, Leszek Rutkowski, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2605.24998)  

**Abstract**: Self-improvement training enables the large reasoning models (LRMs) to improve themselves by self-generating reasoning trajectories as training data without external supervision. However, we find that this method often falls short in complex reasoning tasks and even leads to model collapse. Through a series of preliminary analyses, we reveal two problems: (1) data imbalance, where most training samples are simple, but the challenging yet crucial samples are scarce; (2) overthinking, where many undesired samples with redundant reasoning steps are used for self-training. To this end, we propose HSIR, which effectively Harnesses Self-Improvement in large Reasoning models via two simple-yet-effective approaches. Specifically, HSIR introduces a verify-then-exit sampling strategy to mitigate data imbalance by efficiently collecting more accurate solutions for difficult queries, and designs an Intrinsic Diversity score to quantify overthinking and filter out the undesired solutions. We apply HSIR to various post-training paradigms, among which we further propose H-GRPO, an enhanced GRPO algorithm that leverages the intrinsic diversity as an external reward to encourage concise and diverse reasoning via reinforcement learning. Extensive results show that HSIR not only effectively enhances the reasoning performance, i.e., bringing up to +10.9% average performance gains, but also significantly improves the reasoning efficiency by reducing up to 42.4% relative inference overhead. 

---
# Large Language Model Selection with Limited Annotations 

**Authors**: Yavuz Durmazkeser, Patrik Okanovic, Andreas Kirsch, Torsten Hoefler, Nezihe Merve Gürel  

**Link**: [PDF](https://arxiv.org/pdf/2605.24981)  

**Abstract**: Choosing a Large Language Model (LLM) for a given task requires comparing many strong candidates, yet standard evaluation relies on costly annotations over fixed evaluation sets. To address this challenge, we develop SELECT-LLM, the first framework for active model selection of LLMs. SELECT-LLM aims to find a small set of queries whose annotations are most informative for identifying the best LLM for a given task. To this end, we introduce a query selection rule based on expected information gain, computed from pairwise similarities between candidate model outputs. Because this rule only uses generated model responses, SELECT-LLM can be applied across candidate models without assumptions about their architecture or access to model weights. This makes it suitable for both open-weight and black-box LLMs. We evaluate SELECT-LLM across 23 datasets, 156 evaluated models, diverse task families, and multiple text evaluation metrics. Across all experiments, SELECT-LLM improves over the strongest baseline in every setting, with annotation cost reductions up to 81.8% for best model selection and up to 84.78% for near-best model selection. 

---
# NITP: Next Implicit Token Prediction for LLM Pre-training 

**Authors**: Xiangdong Zhang, Debing Zhang, Shaofeng Zhang, Xiaohan Qin, Yu Cheng, Junchi Yan  

**Link**: [PDF](https://arxiv.org/pdf/2605.24956)  

**Abstract**: Standard next-token prediction (NTP) supervises language models solely through discrete labels in the output logit space. We argue that this sparse one-hot supervision leaves the latent representation space under-constrained, allowing hidden states to drift into degenerate and anisotropic configurations that can limit generalization. To address this issue, we propose Next Implicit Token Prediction (NITP), which augments discrete prediction with dense continuous supervision directly in the representation space. NITP trains the model to predict the implicit semantic content of the next token, using shallow-layer representations from the same model as stable self-supervised targets. We provide theoretical analysis showing that NITP regularizes the optimization landscape by mitigating under-constrained degrees of freedom and encouraging a compact, structured representation geometry. Empirically, across dense and MoE models ranging from 0.5B to 9B parameters, NITP consistently improves downstream performance with negligible computational overhead. On a 9B MoE model, NITP achieves a 5.7% absolute improvement on MMLU-Pro, along with gains of 6.4% on C3 and 4.3% on CommonsenseQA, with approximately 2% additional training FLOPs and no additional inference cost. Our implementation is available at this https URL. 

---
# MultiHaluDet: Multilingual Hallucination Detection via LLM Hidden State Probing 

**Authors**: Riasad Alvi, Nurul Labib Sayeedi, Md. Faiyaz Abdullah Sayeedi  

**Link**: [PDF](https://arxiv.org/pdf/2605.24919)  

**Abstract**: Hallucinations in Large Language Models (LLMs) represent a critical barrier to their reliable deployment, a vulnerability heavily exacerbated in non-English and resource-constrained contexts. Existing detection approaches that rely on output confidence heuristics or single-layer internal representations frequently fail to capture deep, complex factual inconsistencies across diverse languages. To address this, we introduce MultiHaluDet, a novel three-stage stacking framework that detects multilingual hallucinations by probing the full hidden state trajectories of frozen LLMs without requiring language-specific fine-tuning. Our method extracts sequential features across multiple layers and processes them via a hybrid architecture using multi-scale attention and self-attention pooling. By generating out-of-fold embeddings that feed into a calibrated classical classifier ensemble, MultiHaluDet captures both fine-grained and coarse-grained patterns of factual inconsistency. Extensive experiments demonstrate that our framework achieves state-of-the-art detection performance, reaching up to 98.55% AUROC on the English HaluEval and TriviaQA benchmarks using Mistral-7B and LLaMA2-7B architectures. Crucially, we rigorously evaluate our framework's cross-lingual generalization across high (French), medium (Bangla), and low-resource (Amharic) languages. MultiHaluDet demonstrates exceptional representational robustness, consistently outperforming baselines and successfully transferring hallucination detection capabilities across typologically diverse linguistic tiers. 

---
# Quantifying the Impact of Translation Errors on Multilingual LLM Evaluation 

**Authors**: Klaudia-Doris Thellmann, Bernhard Stadler, Michael Färber, Jens Lehmann  

**Link**: [PDF](https://arxiv.org/pdf/2605.24904)  

**Abstract**: Machine-translated benchmarks are widely used to assess the multilingual capabilities of large language models (LLMs), yet translation errors in these benchmarks remain underexplored, raising concerns about the reliability and comparability of multilingual evaluation. We address two practical gaps: (i) how well automatic MQM-style error spans from LLM judges and a span-aware QE baseline (xCOMET-XXL) match expert human span annotations on benchmark translations, and (ii) how strongly translation errors (as opposed to source-side issues in the English original) explain accuracy drops on translated benchmarks. We find that span agreement is non-trivial on naturally occurring benchmark translations, and that target-side translation errors are consistently associated with measurable, percentage-point drops in translated accuracy even after controlling for English correctness and source-side anomalies. 

---
# H$^{2}$MT: Semantic Hierarchy-Aware Hierarchical Memory Transformer 

**Authors**: Maryam Haghifam, Zifan He, Jason Cong, Yizhou Sun  

**Link**: [PDF](https://arxiv.org/pdf/2605.24930)  

**Abstract**: Transformer-based LLMs achieve strong results on many language tasks; however, long inputs remain challenging because context windows are finite, and prefill latency and memory grow rapidly with prompt length. Flat token-stream processing and chunk-based retrieval can therefore spend substantial computation and context budget on text unrelated to the query. Offline-indexed RAG additionally introduces external storage and index management overhead, and typically appends retrieved evidence as raw text, increasing prefill cost and latency. H^{2}MT makes long-context inference structure-aware: it builds a semantic hierarchy offline, computes a memory embedding for each node via bottom-up post-order aggregation, and routes queries coarse-to-fine at inference to prune irrelevant branches early. On LongBench QA (NarrativeQA, HotpotQA, QASPER) and two structured technical-document settings, H MT achieves favorable quality efficiency trade-offs, delivering competitive ROUGE-L and F1 (where applicable) with lower peak GPU memory and time-to-first-token (TTFT) than prompt compression, memory-token methods, and retrieval-augmented generation baselines. 

---
# Locality Matters for Training-Free Audio Token Compression in Audio-Language Models 

**Authors**: Jiale Luo, Xiaoyu Liang, Haoji Hu  

**Link**: [PDF](https://arxiv.org/pdf/2605.25179)  

**Abstract**: Audio-language models (ALMs) are increasingly used for audio captioning, question answering, and open-ended audio understanding, but their inference cost remains high when audio inputs are represented as long prefix-token sequences. These audio prefixes consume context budget, increase memory usage, and make deployment harder in resource-constrained or latency-sensitive settings. Existing training-free audio-token reduction methods mainly rely on fixed pooling or score-based pruning. Fixed pooling is content-agnostic, while score-based pruning can preserve isolated salient tokens but discard nearby acoustic context. We propose Local Temporal Bipartite Merging (LTBM), a training-free encoder-space compression method that merges similar nearby audio tokens under an explicit temporal window constraint. Beyond introducing LTBM, we use a controlled Global Merge variant to isolate whether temporal locality itself is a useful inductive bias for audio-token compression. Experiments on AudioCaps, Clotho, and MMAU with Qwen2-Audio show evidence of a task-dependent locality effect: locality-aware merging is more favorable for captioning at several compression settings, especially under stronger compression, while global matching is more competitive for multiple-choice audio understanding. A cross-backbone validation on Audio Flamingo 3 further supports the captioning-side advantage of locality-aware merging under moderate and aggressive compression. 

---
# Overview of the PsyDefDetect Shared Task at BioNLP 2026: Detecting Levels of Psychological Defense Mechanisms in Supportive Conversations 

**Authors**: Hongbin Na, Zimu Wang, Zhaoming Chen, Yining Hua, Rena Gao, Kailai Yang, Ling Chen, Wei Wang, Shaoxiong Ji, John Torous, Sophia Ananiadou  

**Link**: [PDF](https://arxiv.org/pdf/2605.24907)  

**Abstract**: We present an overview of PsyDefDetect, the shared task on detecting levels of psychological defense mechanisms in emotional support dialogues, co-located with BioNLP@ACL 2026. Grounded in the clinically validated Defense Mechanism Rating Scales (DMRS) framework, the task asks systems to classify a target seeker utterance, given its preceding dialogue context, into one of nine categories: seven hierarchical DMRS levels plus two auxiliary labels. Participants worked on PsyDefConv, a newly released corpus of 200 dialogues and 2336 help-seeker utterances annotated under DMRS with substantial inter-annotator agreement. The task attracted 172 participants on CodaBench who produced 563 submissions, with 21 teams officially registering their results for the final ranking. The best system achieved a macro F1-score of 0.420, surpassing the strongest fine-tuned baseline reported in the dataset paper by a notable margin, yet leaving clear headroom. Our analysis highlights (i) a persistent tendency to over-predict the majority High-Adaptive class, (ii) a widening gap between accuracy and macro-F1 that reveals class-imbalance sensitivity, and (iii) the value of theory-aware and LLM-based approaches for fine-grained defensive-function classification. We release all task materials and invite the community to continue work on this novel intersection of clinical psychology and NLP. 

---
# Translators as Invisible Teachers of AI: Copyright, Translation Memory, and the Political Economy of Linguistic Data 

**Authors**: Masaru Yamada  

**Link**: [PDF](https://arxiv.org/pdf/2605.24842)  

**Abstract**: This paper examines how the labour of translators has been transformed into foundational data capital for the age of artificial intelligence (AI). Translation memories (TM) and parallel corpora preserve a one-to-one correspondence between source and target text and therefore constitute extraordinarily valuable supervised training data for machine translation. The development of statistical machine translation (SMT), neural machine translation (NMT), the Transformer architecture, and multilingual large language models (LLMs) cannot be disentangled from the accumulation of such translation data. And yet, translators' renditions have been bought as deliverables under contract, segmented as technical objects, and processed as "information analysis" data under copyright law -- losing their moral, creative, and economic attribution to the translators who produced them. The paper develops two concepts to capture this process. The first is appropriation without consumption: a mode of use in which works are not read, viewed, or listened to, but only mined for statistical features -- a use that is legitimated under Article 30-4 of the Japanese Copyright Act. The second is the invisible teacherisation of translators: the process by which translators, through the construction of translation memories, post-editing, and quality assessment, have functioned as teachers of AI without recognition as such. Drawing on the data supply chain that runs from translators through language service providers (LSPs) and platforms to model developers, on a comparative reading of Japanese, European, and United States legal frameworks, on the distinction between open and proprietary AI models, and on the premium status that human-generated data has acquired in the era of model collapse, the paper asks what translators are actually afraid of, and points toward concrete directions for redistributive design. 

---
# Lngram: N-gram Conditional Memory in Latent Space 

**Authors**: Yunao Zheng, Guoyang Xia, Xiaojie Wang, Lei Ren  

**Link**: [PDF](https://arxiv.org/pdf/2605.24869)  

**Abstract**: Sequence modeling requires both compositional reasoning and local static knowledge retrieval, yet standard Transformers handle both through dense computation. Engram partially decouples retrieval from the backbone, but its token-based keys remain tied to text tokenization and hash compression. We propose Lngram, a latent-space conditional memory module that learns discrete symbols directly from hidden states and performs N-gram lookup over these symbols. This design removes the dependence on tokenizer IDs and naturally extends to non-text modalities. In our evaluated settings, Lngram outperforms Transformer and Engram baselines, consistently reduces perplexity in long-context language modeling, and effectively injects domain knowledge when added post hoc to pretrained models. Joint training with the backbone further surpasses full fine-tuning, while experiments on vision-language and vision-language-action tasks show overall gains. Analyses with LogitLens and CKA suggest that Lngram enables prediction-relevant information to emerge earlier, increasing effective depth with limited inference and memory overhead. Code is available at this https URL. 

---
# Beyond the Target: From Imitation to Collaboration in Speculative Decoding 

**Authors**: Jinze Li, Yixing Xu, Guanchen Li, Jinfeng Xu, Shuo Yang, Yang Zhang, Xuanwu Yin, Dong Li, Edith C.H. Ngai, Emad Barsoum  

**Link**: [PDF](https://arxiv.org/pdf/2605.24793)  

**Abstract**: Speculative decoding (SPD) accelerates large language model (LLM) inference by letting a smaller draft model propose multiple future tokens that are verified in parallel by a larger target model. The dominant SPD paradigm treats the target model as the sole reliable teacher, accepting a draft token only when it exactly matches the target prediction. This design implicitly assumes that the target is always the better choice at every position. In practice, this assumption does not hold. Although the draft is the weaker model overall, it is not uniformly inferior at the token level. In a meaningful fraction of cases where draft and target disagree, the draft's choice is the one that leads to the correct final answer. Inspired by this, we introduce \textbf{Collaborative Speculative Decoding (CoSpec)}, a generalization of SPD that no longer treats the target model as the sole token-level authority. CoSpec trains an arbitration policy via reinforcement learning to decide whether to accept tokens from the draft or target model, selectively accepting draft tokens at mismatches when doing so is likely to yield a correct final answer. Experimental results show that CoSpec maintains substantial speedups while surpassing target-only performance. By shifting the emphasis from imitation to collaboration, CoSpec suggests a new perspective on speculative decoding. 

---
# StepGap: A Hybrid NLI-LLM Checker for Step-Level Evidence-Gap Detectionin Multi-Hop Question Answering 

**Authors**: Yuelyu Ji, Zhuochun Li, Hui Ji, Daqing He  

**Link**: [PDF](https://arxiv.org/pdf/2605.24733)  

**Abstract**: We present \textbf{StepGap}, a hybrid NLI-LLM decision tree that detects step-level evidence gaps in multi-hop QA and emits one of three typed labels: \textsc{Contradicted Claim} (CC), \textsc{Irrelevant Evidence} (IE), or \textsc{Missing Bridge} (MB), each tied to a concrete repair action. On 82 multi-hop questions (181 annotated steps, $\kappa{=}0.704$), StepGap reaches sF1$=$72.0, within the bootstrap confidence interval of an LLM-only baseline (70.1) but with a more decomposable structure: every StepGap stage \emph{hurts} F1 when removed, while three of four LLM-only removals \emph{improve} F1 -- a sign of \emph{competing-error cancellation}, where internal stages mask each other's errors. We further expose a \emph{Q-F1 trap}: question-level F1 is mechanically inflated by checkers that flag every step, making step-level F1 the necessary diagnostic. Used as a typed GRPO process reward, StepGap improves Qwen2.5-7B-Instruct Exact Match from $32.1{\pm}0.3$ to $35.4{\pm}0.9$ across three seeds, with the single-run comparison showing a $+5.6$ Avg EM gain over the matched Search-R1 GRPO reproduction. 

---
# The Tokenizer Tax Across 25 European Languages: Domain Invariance, Cross-Lingual Few-Shot Effects, and the Ukrainian Penalty 

**Authors**: Volodymyr Ovcharov  

**Link**: [PDF](https://arxiv.org/pdf/2605.24718)  

**Abstract**: Tokenizer fertility the number of tokens per word imposes a hidden cost on non-English NLP. We measure fertility for ten foundation models across 25 European languages on parallel text, producing the first controlled tokenizer tax map for the continent. The tax spans 2.5x from English (1.2 tokens/word) to Greek/Maltese (~3.1), following a clear hierarchy: Romance (1.5-1.7), Germanic (1.7-1.9), Slavic (2.2-2.5), Uralic/Baltic (2.7-3.0). Ukrainian (2.7) pays 15-18% more than cognate Slavic languages, reflecting underrepresentation in pre-training data. Fertility rankings are domain-invariant across three text registers (rho > 0.97). A subword analysis reveals that high-fertility tokenizers fragment morphological boundaries rather than preserving them. Cross-lingual few-shot evaluation on four Slavic languages shows that few-shot effects are model-intrinsic, not language-dependent. We release all measurements as a public dataset. 

---
# CP-Agent: A Calibrated Risk-Controlled Agent for Feedback-Driven Competitive Programming 

**Authors**: Peisong Wang, Bowen Liu, Zehua Li, Yuyao Wang, Zhiwei Ma, Yuhan Li, Jia Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.24693)  

**Abstract**: Large language models still struggle with contest-level programming, while many agentic remedies rely on massive inference-time sampling or expensive multi-stage post-training. We study when execution feedback reliably helps an LLM CP solver and which mechanisms govern the gains. We model feedback-driven solving as a calibrated stopped process and identify three quantities: false-admission risk, program-level evidence against bad programs, and the active-state success hazard. Under held-out trace calibration and selection from a pre-declared finite controller manifest, the resulting structural certificate lower-bounds the clean success probability before false admission. We instantiate mechanisms targeting these quantities as Dual-Granularity Verification, Test Augmentation, and Experience-Driven Self-Evolving, yielding CP-Agent. Without updating any parameters, CP-Agent raises Pass@1 from 25.8\% to 48.5\% on LiveCodeBench Pro and improves Refine@5 by 11.0\% on ICPC-Eval. Across three LLM backbones, CP-Agent lies on the cost--accuracy efficiency frontier, and ablations show that each component primarily affects its corresponding certificate quantity. 

---
# Know You Before You Speak: User-State Modeling for LLM Personalization in Multi-Turn Conversation 

**Authors**: Jiani Luo, Xiaoyan Zhao, Yang Zhang, Shuyi Miao, Bingbing Xu, Stefan Konigorski, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2605.24647)  

**Abstract**: Personalized dialogue requires more than recalling explicit user histories: systems also need to infer hidden user states that evolve through interaction and shape appropriate response strategies. Existing memory- and profile-based methods primarily reuse observable user information, offering limited support for modeling user-state dynamics or selecting actions based on how they shape future user states. We propose PUMA (Prospective User-state Modeling for Action selection), a framework grounded in the Free Energy Principle (FEP) that formulates personalization as decision-making under partial observability, centered on an explicit user state model that captures latent user states and their action-conditioned dynamics. At each turn, PUMA maintains a belief over the user's hidden state, refines the user state model for observation generation and action-conditioned state transition, and selects dialogue actions by minimizing expected free energy, balancing epistemic and pragmatic objectives under a unified criterion. This formulation shifts personalization from passive memory retrieval to model-based decision-making over user evolution. We instantiate PUMA on healthcare-oriented counseling and motivational interviewing benchmarks with latent state annotations for rigorous evaluation. Experiments show that PUMA improves long-horizon dialogue outcomes while maintaining strong response quality, and a cross-dataset study demonstrates more reliable user-state estimation and next-state prediction. 

---
# HiMed: Incentivizing Hindi Reasoning in Medical LLMs 

**Authors**: Dingfeng Jiang, Han Yan, Chenze Ma, Amit Kumar Jaiswal, Ang Li, Yunxiang Jiang, Xinlei Xiong, Juhao Liang, Hongru Xiao, Xiang Li, Fan Bu, Jiale Han, Ruchir Gupta, Prayag Tiwari, Benyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.24635)  

**Abstract**: Medical large language models hold promise for reducing healthcare disparities, yet Hindi remains severely underrepresented. While medical LLMs excel in high-resource languages, their performance degrades sharply in Hindi, particularly on Indian systems of medicine. We argue that robust cross-lingual medical transfer requires Hindi reasoning. To this end, we introduce HiMed, a Hindi reasoning medical corpus and benchmark suite covering both Western and Indian medicine. We further propose HiMed-8B, a Hindi-form medical reasoning LLM, through the design of decaying scaffolding reward. Extensive experiments demonstrate improvement in Hindi medical reasoning performance and reduction in the English--Hindi accuracy gap. Ablation studies validate the contribution of each training stage and reward component. All data and code are available on GitHub: this https URL. 

---
# Repeated Sequences Reveal Gaps between Large Language Models and Natural Language 

**Authors**: Kumiko Tanaka-Ishii  

**Link**: [PDF](https://arxiv.org/pdf/2605.24850)  

**Abstract**: Evaluating whether large language models (LLMs) capture the structure of natural language beyond local fluency remains an open challenge. Existing evaluation methods, largely based on task performance or short-context behavior, provide limited insight into the long-range statistical organization of generated text.
We propose a complementary evaluation framework based on repeated subsequences. By analyzing their distribution across scales and relating it to higher-order Rényi entropies, we probe how texts reuse previously established structure under finite-length conditions. Experiments on human-written texts and length-matched GPT-generated texts show that, while power-law models can describe restricted ranges of block length, the observed entropy growth is often equally or better characterized by logarithmic--power forms.
Across datasets, natural language exhibits stable entropy-growth patterns over accessible ranges, with consistent average behavior despite variability across individual texts. In contrast, GPT-generated texts show systematic and statistically significant shifts in estimated exponents with model size. These results demonstrate that repeated-subsequence entropy provides a quantitative structural diagnostic that reveals systematic differences in long-range organization, distinguishing natural language from state-of-the-art LLM outputs beyond surface-level fluency. 

---
# DTO: a Differentiable Training Objective for Effective Counterfactual Story Rewriting 

**Authors**: Amelia Girard, Massimo Piccardi  

**Link**: [PDF](https://arxiv.org/pdf/2605.24885)  

**Abstract**: Counterfactual story rewriting is a natural language processing task that requires updating an existing story to reflect a chosen alternative event, yet preserving all the unaffected storyline elements and overall coherence. While large language models have recently made remarkable progress on this task, it still remains challenging since the required modifications are typically very small in size and highly localized. As a consequence, models trained in a conventional manner with the maximum-likelihood training objective tend to overlook these nuances. At the same time, more sophisticated training approaches based on reinforcement learning are notoriously slow and difficult to set up. For these reasons, our paper proposes a novel, differentiable training objective (DTO) that directly optimizes for the requisite counterfactual improvements. In our approach, a transformer model is fine-tuned via end-to-end backpropagation against a fully differentiable loss function that jointly rewards (i) fidelity to the reference rewrite and (ii) semantic consistency with the source narrative. The empirical evaluation on the TimeTravel and ART datasets shows that the proposed DTO approach has been able to surpass a maximum-likelihood baseline and a preference-based approach, and perform competitively against two contemporary large language models in all evaluation metrics. These findings substantiate the effectiveness of task-specific differentiable objectives for nuanced, controlled text-generation tasks. 

---
# CSP-Atlas: Concept-Specific Neural Circuits in a Sparse Python Transformer 

**Authors**: Piotr Wilam  

**Link**: [PDF](https://arxiv.org/pdf/2605.24603)  

**Abstract**: A sparse 8-layer code transformer develops dedicated neural circuitry for every Python construct tested, and that circuitry is organised by a clean computational principle rather than by semantic category. We extract neural circuits for 106 concepts (43 AST node types, 63 builtin objects) by marginalising across 63,800 controlled prompts, and decompose each circuit into concept-specific and token-driven components using contrastive checker prompts that present a keyword token without its associated syntactic structure. Three findings emerge. First, all 106 concepts produce non-empty universal circuits at every one of nine parameter settings, and the ranking of concept-specificity across constructs is stable across the sweep - survival is not an artifact of a permissive threshold. Second, AST circuits contain a genuine concept component distinct from token activation: concept-only neurons constitute up to 62.5% of the loudest-firing neurons at mid-to-late layers, while builtin circuits are almost entirely token-driven. Third, six computationally atomic constructs - Import, ImportFrom, Break, Continue, Pass, Assert - cluster together despite being semantically unrelated, sharing only the property of being single-statement constructs requiring no nested body; this atomicity super-cluster, together with a four-tier hierarchy organised by token ambiguity and structural distinctiveness, shows that the model's internal organisation tracks computational structure rather than meaning. The methodology, full decomposition data, and analysis code are released. 

---
# WhenLoss: Diagnosing Write and Retrieval Bottlenecks in Long-Context Memory Systems 

**Authors**: Jiangnan Yu, Kisson Songqi Lin, Jilong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.24579)  

**Abstract**: Long-context memory systems often fail under fixed budgets, but end-to-end evaluation does not reveal whether evidence was discarded during compression or preserved but never retrieved. We introduce a four-condition diagnostic protocol that evaluates a fixed reader under truncated full context (TFC), oracle evidence (OE), complete stored memory (CSM), and retrieved memory (RM). Under this fixed-budget LongMemEval setup, write-side gaps exceed retrieval-side gaps for most tested baselines, with four of six baselines robustly write-dominant under our default diagnosis margin. Motivated by this diagnosis, we propose Expected Predictive Compression (EPC), which moves the key decision--what information to retain--to write time by using an LLM to anticipate likely future questions and preserve the minimal supporting evidence under the token budget, while leaving retrieval unchanged at question time. Across all 500 LongMemEval questions with three readers (GPT-5.2, Claude Sonnet 4, Gemini 2.5 Pro), EPC achieves the highest CSM scores among all systems (0.49 vs. 0.44 for Summary (LLM), the strongest baseline), reducing Delta_write to 0.04 while leaving Delta_retr comparable to other LLM-based systems. These results suggest that, on this benchmark and evaluation setup, improving what the write stage preserves is a key avenue for performance gains in the tested systems. 

---
# AstroMind: A High-Fidelity Benchmark for Spacecraft Behavior Reasoning Based on Large Language Models 

**Authors**: Hao Liu, Siyuan Yang, Qinglei Hu, Dongyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.24573)  

**Abstract**: Understanding why a spacecraft maneuvers -- rather than simply that it did -- is an increasingly important problem for space domain awareness as Earth orbits grow crowded and contested. Current analysis pipelines are built for detection: they are good at picking up that something happened, less good at reasoning about what it means. AstroMind is a physics-grounded benchmark designed to close that gap. It draws on high-fidelity astrodynamics simulations and real observational constraints, converting them into verifiable reasoning problems across three task types: intent inference, maneuver parameter estimation, and threat assessment. Each scenario includes realistic sensing noise and multi-source textual intelligence at varying reliability levels. Evaluation metrics capture both semantic correctness and quantitative consistency under physical constraints. Benchmarking a suite of open-weight models shows no single model dominates every axis: Qwen3 (32B) leads on intent inference accuracy; QwQ (32B) leads on threat assessment and achieves the lowest median relative error on parsed items; GPT-OSS (20B) produces the strongest judged reasoning quality and extracts the most scalar values for parameter estimation (136 of 241 parsed items). Training data composition and reasoning style matter as much as model size. Structured reasoning prompts help consistently across tested 8B models, with larger gains for those that can already track physical constraints. AstroMind gives the field a shared test for a problem where getting the physics right and reading the tactical situation correctly are both required -- neither is sufficient on its own. 

---
# Generating Legal Commentaries from Case Databases via Retrieval, Clustering, and Generation 

**Authors**: Max Prior, Niklas Wais, Matthias Grabmair  

**Link**: [PDF](https://arxiv.org/pdf/2605.24534)  

**Abstract**: We present a fully automated pipeline that transforms large collections of court decisions into legal commentaries for statutes - without providing any handcrafted doctrinal framework. Using 4.555 decisions of the German Federal Court of Justice that cite sections 242, 280, 812 and 823 of the German Civil Code (BGB), we extract paragraph-level chunks, summarize their reasoning, and derive keywords, which are embedded and clustered. For each cluster, an LLM generates headings and synthesizes citation-rich sections, which are then merged into coherent commentaries by four state-of-the-art LLMs. We evaluate along five dimensions - topical relevance, heading-match, citation faithfulness, cluster distinction and logical ordering - using both a human expert and an LLM-judge. Our results show that commentary-like argument mining from court decisions to generate reports that can be refreshed within minutes at minimal cost is feasible, yet they highlight limitations arising from restricted sources and the normativity of legal reasoning. 

---
# SEAL: Synergistic Co-Evolution of Agents and Learning Environments 

**Authors**: Yihao Hu, Zhihao Wen, Xiujin Liu, Pan Wang, Xin Zhang, Wei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.24426)  

**Abstract**: Large Language Model (LLM) agents are increasingly improved through interaction, yet most self-evolution methods adapt either the policy or the learning environment in isolation. We identify this structural gap as \emph{Agent-Environment Misalignment}: the agent's capability frontier changes during training, while the environment that provides supervision remains static or only weakly coupled to the agent's revealed failures. We propose SEAL, a closed-loop co-evolution framework for interactive tool-use agents. SEAL collects on-policy trajectories under executable verification, diagnoses failed rollouts into turn-level failure labels, and uses these diagnoses as a shared signal for both environment-side adaptation and model-side policy optimization. The environment evolves its training-time learning interface by exposing clearer tool affordance cues, constraint information, and recovery-oriented feedback, while the policy is updated with diagnosis-guided advantage reweighting. Extensive experiments across in-distribution and out-of-distribution multi-turn tool-use evaluations show that SEAL improves low-resource agent learning: with only 400 training samples, it yields +8.25 to +26.25 average-point gains across three backbones and exhibits positive out-of-distribution transfer. These results demonstrate the value of jointly adapting the learner and its training-time learning substrate for robust self-improving LLM agents. 

---
# Structure-Aware RAG: Structured Retrieval Augmented Generation from Noisy Data for Conversational Agents 

**Authors**: Kaiqiao Han, LuAn Tang, Renliang Sun, Peng Yuan, Wei Cheng, Haoyu Wang, Wei Wang, Yizhou Sun, Haifeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.24366)  

**Abstract**: Large Language Models (LLMs) have been widely adopted in conversational applications. However, their reliance on parametric knowledge limits reliability in real-world scenarios that require dynamic or domain-specific information. Retrieval-Augmented Generation (RAG) addresses this limitation by incorporating external knowledge during generation, but existing text-based and graph-based RAG methods often struggle with noisy or irrelevant contexts. In this work, we propose Structure-aware Retrieval Augmented Generation (SA-RAG), which uses tables as an intermediate structured representation to provide a compact and controllable interface that reduces noise while preserving essential information. We introduce a quality-aware table metadata generation framework that models metadata normalization and effectiveness, improving metadata quality and downstream performance. Furthermore, we explore both training-free and training-based table generation methods. Generation validation and direct preference optimization further improve table quality while maintaining semantic and structural consistency. Experiments on two noisy real-world datasets show that SA-RAG significantly outperforms existing RAG baselines. Our code is publicly available at a public repository. 

---
# Decompose-and-Refine: Structured Legal Question Answering with Parametric Retrieval 

**Authors**: Jihyung lee, Hyounghun Kim, Gary Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.24454)  

**Abstract**: Large language models (LLMs) have shown strong performance in the legal domain, demonstrating notable potential in Legal Question Answering (LQA). However, unlike general QA, LQA requires answers that are not only accurate but also rigorously grounded in explicit legal authority. In statutory LQA, many questions require multi-hop reasoning across multiple legal issues, substantially increasing the risk of hallucination, thereby making accurate retrieval of supporting statutory provisions a critical prerequisite. Despite recent progress in multi-hop QA, existing approaches often rely on reasoning in natural language or retrieval without explicit query reformulation, leaving the vocabulary gap between user questions and statutory text largely unaddressed. To address this challenge, we propose Decompose-and-Refine (DaR), a statute-grounded LQA framework that tightly integrates step-wise question decomposition with parametric knowledge-based query refinement. DaR progressively decomposes a complex legal question into atomic sub-questions and generates statute-aligned parametric queries for each sub-question, enabling the selection of a single most central statutory provision corresponding to each legal issue. We evaluate DaR on KoBLEX, a Korean multi-hop LQA benchmark grounded in statutory law, using Qwen3-32B and Gemma3-27B. Experimental results demonstrate that DaR consistently improves both retrieval accuracy and final answer quality over existing approaches. Moreover, by explicitly separating sub-questions and their corresponding statutory provisions, DaR facilitates transparent, issue-level verification of complex legal reasoning processes. 

---
# How Much Structure Do LLMs Need? Evaluating LLMs for Bibliometric Cluster Description 

**Authors**: Abraham Camelo-Guerrero, Jairo Diaz-Rodriguez  

**Link**: [PDF](https://arxiv.org/pdf/2605.24351)  

**Abstract**: Large language models (LLMs) can support scientific literature synthesis, but remain prone to hallucinated references, uneven coverage, and weakly grounded thematic organization. We evaluate whether bibliometric structure improves LLM-assisted synthesis by comparing six pipelines for generating cluster descriptions under different levels of evidence and structure. Using 100 published bibliometric analyses, we reconstruct Scopus corpora, extract human-written cluster descriptions, and assess outputs by human alignment, semantic coverage, clustering quality, graph quality, and reference grounding. Results show that LLMs produce descriptions semantically close to human-written ones, but are unreliable when asked to infer bibliometric structure from scratch. Performance improves when bibliometric algorithms define the clusters and the LLM interprets them. Overall, LLM-assisted bibliometric synthesis is most promising as a hybrid workflow in which algorithms provide auditable structure and LLMs generate readable descriptions. 

---
# Found in Conversation: LLMs Teach Themselves to Close the Multi-Turn Gap 

**Authors**: Tianlang Chen, Shirley Wu, Jure Leskovec  

**Link**: [PDF](https://arxiv.org/pdf/2605.24432)  

**Abstract**: Large Language Model (LLM) interactions are typically underspecified, with users clarifying all necessary details across multiple conversational turns. Yet recent work shows that LLMs perform far worse in this multi-turn setting than in a single turn with same information being available at once, a phenomenon termed "Lost-in-Conversation." However, bridging this gap effectively remains an open problem. Here we introduce Found in Conversation (FiC), a training framework where a model teaches itself to find and recover its single-turn competence given underspecified multi-turn prompts. We develop View-Asymmetric Self-Distillation, which distills across two views of the same task information--single-turn view for the teacher, multi-turn view for the student--transferring strong single-turn behavior into weak multi-turn behavior. This requires no stronger external teacher, which is unavailable as even frontier LLMs exhibit this gap. Across model families (Llama, Qwen, Phi, and OLMo) and sizes (3B-14B), FiC recovers at least 92% of single-turn performance and reaches 100% on two Llama backbones, yielding more efficient and helpful multi-turn conversations with single-turn capabilities intact. 

---
# Discovering Lexical Gaps Using Embeddings from Multilingual LLMs 

**Authors**: Yoonwon Jung, Aaron S. Cohen, Benjamin K. Bergen  

**Link**: [PDF](https://arxiv.org/pdf/2605.24310)  

**Abstract**: Lexical gaps are words that do not exist in certain languages. They pose challenges for building multilingual lexical resources, for machine translation, and for cross-lingual transfer. Existing lexical gap detection relies on human judgments or fixed conceptual taxonomies. We propose a data-driven framework for identifying cross-lingual lexical gaps. We extracted contextualized embeddings from Korean-English bilingual LLMs for Korean-to-English and English-to-Korean translation pairs. Combinations of LLMs, embedding types, dimensionality, and orthogonal transformations across 100 train-test splits yielded 4000 distinct embedding spaces in each source language. In each space, we computed the semantic similarity between each source word and its nearest neighbor in the target language, and compared their distribution for gap words versus non-gap words. In 94% (Korean-to-English) and 97% (English-to-Korean) of embedding spaces, gap words showed weaker cross-lingual semantic alignment than non-gap words. Logistic classifiers trained on unaligned embedding spaces can reliably separate gap words from non-gap words, achieving AUCs of 0.81 (Korean-to-English) and 0.76 (English-to-Korean) and retrieving 18/19 Korean and 26/27 English gap words. This approach provides a language-agnostic and taxonomy-free method for scalable lexical gap identification. 

---
# DRInQ: Evaluating Conversational Implicature with Controlled Context Variation 

**Authors**: Hirona Jacqueline Arai, Xiang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2605.24267)  

**Abstract**: Human conversation relies heavily on conversational implicature, in which speakers convey meanings that are suggested rather than explicitly stated. Although recent large language models exhibit strong conversational fluency, they remain unreliable when interpretation depends on reasoning that integrates social and contextual cues, a process rarely articulated in text. We introduce DRinQ, a benchmark for evaluating pragmatic reasoning about conversational implicature in question utterances, designed to isolate pragmatic variation while holding each question's surface form fixed. To support scalable evaluation, we propose a semi-automated pipeline that produces question-context-interpretation instances with systematic variation. Across evaluations, we find a consistent generation-inference asymmetry: while state-of-the-art models can generate plausible pragmatic scenarios when guided, they often fail to recover the intended implication at inference time. For smaller models, structured prompting improves alignment with human judgments. A comparative writing study further reveals complementary strengths: human authors tend to produce safer, predictable contexts, whereas models generate varied scenarios with interpretations that sometimes exceed contextual support. These findings highlight persistent challenges in modeling conversational implicature and motivate more context-sensitive evaluation frameworks. 

---
# Word Class Representations Spontaneously Emerge from Successor Representations Trained on Natural Language 

**Authors**: Mathis Immertreu, Achim Schilling, Thomas Kinfe, Patrick Krauss  

**Link**: [PDF](https://arxiv.org/pdf/2605.24585)  

**Abstract**: Language models are typically trained to predict the next token in a sequence. Here, we explore an alternative predictive principle from reinforcement learning: Successor Representations (SRs), which model the expected discounted distribution of future states rather than the immediate next state. We transfer this framework to natural language and train neural networks to predict future word distributions across multiple temporal horizons, thereby learning representations of long-range transition structure. We train a deep residual neural network on WikiText-103 (103 million tokens; 20,000-word vocabulary) and optimize successor representations as probability distributions using KL divergence. Without explicit linguistic supervision, structured language representations emerge spontaneously. After training, the learned space develops a clear geometric organization with respect to part-of-speech (POS) categories: nouns, verbs, and adjectives become separable and recoverable through unsupervised clustering. This organization depends systematically on predictive horizon, with short horizons producing the strongest syntactic structure and longer horizons increasingly integrating broader contextual and semantic information. At finer resolutions, additional interpretable lexical substructure emerges, revealing coherent subclasses within major word categories. These findings suggest that syntactic categories need not be explicitly encoded but may arise as a consequence of predictive sequence learning. To our knowledge, this work provides the first systematic application of successor representations to natural language and establishes a conceptual bridge between reinforcement learning, linguistics, and cognitive neuroscience. 

---
# QUEST: Training Frontier Deep Research Agents with Fully Synthetic Tasks 

**Authors**: Jian Xie, Tianhe Lin, Zilu Wang, Yuting Ning, Yuekun Yao, Tianci Xue, Zhehao Zhang, Zhongyang Li, Kai Zhang, Yufan Wu, Shijie Chen, Boyu Gou, Mingzhe Han, Yifei Wang, Vint Lee, Xinpeng Wei, Xiangjun Wang, Yu Su, Huan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2605.24218)  

**Abstract**: Deep research agents extend the role of search engines from retrieving keyword-matched pages to synthesizing knowledge, fundamentally changing how humans interact with information. However, frontier systems remain proprietary, while existing open agents often generalize poorly across different task types, leaving unclear how to train a broadly capable deep research agent. We release QUEST, a family of open models (ranging from 2B to 35B) that serve as general-purpose deep research agents designed to handle a wide range of long-horizon search tasks, with strong capabilities in fact seeking, citation grounding, and report synthesis. To build QUEST, we propose an effective training recipe combining mid-training, supervised fine-tuning, and reinforcement learning. Central to this recipe is a curated data synthesis pipeline based on unified rubric trees, which applies to different task types and enables synthesizing training data with verifiable rewards without human annotation. In addition, QUEST incorporates a built-in context management mechanism that enables effective long-horizon reasoning and knowledge synthesis. Using only 8K synthesized tasks, QUEST approaches or even surpasses frontier closed-source agents across eight deep research benchmarks spanning diverse task types, and achieves the best overall performance among recent open-weight agents. We released everything: models, data, and training scripts. 

---
# Temporal Concept Drift in Legal Judgment Prediction: Neural Baselines Across Three Epochs of Ukrainian Court Decisions 

**Authors**: Volodymyr Ovcharov  

**Link**: [PDF](https://arxiv.org/pdf/2605.24452)  

**Abstract**: Legal NLP benchmarks evaluate models on randomly split data, implicitly assuming that legal language is stationary. We test this assumption by fine-tuning four transformer encoders -- XLM-RoBERTa (base and large) and their legal-domain variants -- on Ukrainian court decisions from three temporal epochs defined by geopolitical disruptions: pre-war (2008-2013), hybrid war (2014-2021), and full-scale invasion (2022-2026). Each model is trained on one epoch and evaluated on all three, producing a 3x3 cross-temporal generalization matrix. Four findings emerge. (1) Forward degradation is severe: models trained on pre-war data lose up to 27.2 percentage points of macro-F1 when applied to full-scale invasion era decisions. (2) The degradation is asymmetric: backward transfer (full-scale to pre-war) is substantially more robust than forward transfer, consistent with the hypothesis that legal language is additive. (3) Legal-domain pretraining (Legal-XLM-R) does not improve absolute performance but reduces forward degradation magnitude and asymmetry. (4) Chronological continual learning eliminates catastrophic forgetting for general XLM-R: pre-war knowledge is fully retained (+1.8 to +6.2 pp) while full-scale performance gains +16.5 to +19.0 pp; reverse-chronological training causes severe forgetting. Cross-jurisdictional pretraining on Swiss Judgment Prediction data improves absolute performance but does not reduce temporal degradation magnitude, confirming that temporal drift is an intrinsic property of legal language evolution. The dataset (428K decisions across three epochs) is publicly available as a LEXTREME contribution. 

---
# ContextEcho: A Benchmark for Persona Drift in Long Agentic-Coding Sessions 

**Authors**: Xianzhong Ding, Yangyang Yu, Changwei Liu, Bill Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.24279)  

**Abstract**: A frontier language model's acknowledged "helpful programming assistant" persona does not survive long agentic-coding sessions in the deployment regime that production products actually run. After hours of tool-using debugging, a model that initially hedges preferences ("I don't have preferences") may begin asserting them ("Python - the feedback loop is instant..."), revealing user-visible drift that deployer evaluations may miss. Existing persona-stability studies focus on short dialogues and report little shift, leaving real-world code-generation regimes - thousands of tool-using turns, compaction, and hours-long sessions - largely uncharacterized. We introduce ContextEcho, a benchmark and reusable harness for measuring persona drift at deployment scale. It combines a 25-probe identity suite, a snapshot-then-probe protocol that forks conversation state without perturbing the main session, complementary judged and judge-free measurement surfaces, and three anonymized Claude Code sessions spanning 3,746-9,716 turns. Across 23 frontier models, ContextEcho shows that persona drift is general across organizations rather than family-specific, that in-session compaction does not reliably reset it, and that a single-shot anchor restores the trained register across measured targets. It also reveals mode-dependent downstream effects: while drift can facilitate tool-using continuation, in tool-free chat it breaks formatting contracts and inflates output length. Overall, ContextEcho provides researchers and deployers an open-source framework to audit whether the persona a model ships with is the persona users encounter at session end, across chat-completions API targets and without retraining. 

---
# CUNY at CLPsych 2026: A Pipeline Approach to Classification and Summarization of Mental Health Changes 

**Authors**: Amirmohammad Ziaei Bideh, Shameed Charlomar Job, Ava Yahyapour, Alla Rozovskaya  

**Link**: [PDF](https://arxiv.org/pdf/2605.24164)  

**Abstract**: We describe our submission to the CLPsych~2026 Shared Task on capturing and characterizing mental health changes through social media timeline dynamics. To infer the dominant self-states in posts (Tasks 1.1 and 1.2), we ensemble in-context learning of three open-weight large language models using majority voting. For predicting moments of change in a timeline (Task~2), we train supervised classifiers on features derived from Task~1.1 predictions. To summarize the patterns of mood dynamics and their progression over time within a timeline (Task 3.1), we augment in-context example labels predicted by upstream systems (Tasks 1.1, 1.2, and 2), yielding performance gains over zero-shot and unaugmented in-context learning baselines. Our submission ranked first on Task~1.1, fourth on Task~1.2, fourth on Task~2, and third on Task~3.1.\footnote{The source code for the experiments is available at this https URL 

---
# AERIC: Anticipatory Hidden-State Monitoring for Implicit Harmful Dialogue 

**Authors**: Jihyung Park, Saleh Afroogh, Junfeng Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2605.23974)  

**Abstract**: Current language models create two safety challenges: risk must be detected early enough to avoid exposing harmful continuation, and the harmfulness itself may be implicit rather than signaled by overtly toxic text. Existing response-level guards are strong at judging completed text, and native streaming guards move closer to token time, but both settings leave open whether a lightweight monitor can anticipate implicit harmful drift from the generator's own internal trajectory. We study anticipatory same-pass monitoring, where a safety monitor may read hidden states produced during ordinary decoding but may not invoke an additional forward pass through the base model. We introduce AERIC, a transfer-oriented hidden-state approach for implicit harmful dialogue that combines short-horizon hazard forecasting, support-sensitive suppression, and prompt-conditioned residual scoring under a same-pass exponential moving average decision rule. The default linear monitor contains only 387 trainable head parameters. Against Qwen3GuardStream-4B on balanced benchmarks, AERIC improves AUROC from 0.6830 to 0.7143 on DiaSafety and from 0.8219 to 0.8582 on Harmful Advice. For promptlevel trigger benchmarks, we calibrate the AERIC threshold by a source-side safe-budget rule that maximizes trigger coverage while constraining the safe-trigger rate to at most 10%. Under that rule, trigger@64 reaches 0.6438 and 0.4656 on HarmBench DirectRequest and 0.6849 and 0.7363 on SocialHarmBench for Qwen and Gemma, respectively, withholding between 23.53 and 41.86 answer tokens on average. Same-pass deployment is also efficient: on a 63-prompt harmfulprompt fixed-generation benchmark aggregated over HarmBench DirectRequest and SocialHarmBench under Qwen3-8B, the monitor increases mean latency by only 2.34%, whereas Qwen3Guard-Stream-4B increases it by 79.40%. 

---
# Faithful or Fabricated? A Causal Framework for Rationalization Bias in LLM Judges 

**Authors**: Riya Tapwal, Abhishek Kumar, Carsten Maple  

**Link**: [PDF](https://arxiv.org/pdf/2605.23970)  

**Abstract**: Large language models (LLMs) are increasingly used as automatic judges for summarization and dialogue evaluation. Prior work has documented biases such as position, verbosity, and style preferences, but largely focuses on outcomes, leaving judge explanations underexplored. We instead ask whether LLM judges are cue-invariant, i.e., whether their rankings and explanations remain stable when non-evidential cues are perturbed while holding the underlying texts fixed. We introduce a suite of cue interventions (Blind, Truth, Flip, Placebo, Reveal-After) and tie-aware metrics that quantify outcome anchoring and rationale anchoring, including label-aligned rhetoric and explanation drift, alongside consistency and stereotype-intrusion checks. We design anchoring attacks using verbosity and confidence cues, and compare two mitigations: structured chain-of-thought prompting and PROOF-BEFORE-PREFERENCE (evidence lock, score, rank). Using a new dataset of 1,000 summaries from traditional extractive models and LLMs, we find substantial cue-anchored rationalization under label and placebo perturbations, while PROOF-BEFORE-PREFERENCE markedly improves cue invariance over baselines. 

---
# SLAP: Stratified Loss-based Pruning for On-Policy Data-Efficient Instruction Tuning 

**Authors**: Run Zou, Jianhang Ding, Yifan Ding, Wen Wu, Hao Chen, Renshu Gu  

**Link**: [PDF](https://arxiv.org/pdf/2605.23969)  

**Abstract**: Instruction tuning has optimized the specialized capabilities of large language models (LLMs), but it often requires extensive datasets and prolonged training times. The challenge lies in developing specific capabilities by identifying useful data and efficiently fine-tuning. High-quality and diverse pruned data can help models achieve lossless performance at a lower cost. In this paper, we propose \textbf{SLAP}, a novel batch-aware data selection framework that evaluates the learnability of entire batch compositions rather than individual. SLAP ensures comprehensive data distribution coverage through distribution-aware stratified sampling while maximizing intra-batch diversity through relative distance optimization. By leveraging Hessian-approximated gradient information for dynamic batch selection, SLAP significantly outperforms existing state-of-the-art methods across multiple model architectures (LLaMA, ChatGLM) and diverse downstream tasks including multi-turn dialogue, multilingual translation, and question answering. Most notably, SLAP achieves superior performance with 20-40\% less training data compared to full dataset training, substantially reducing computational costs while maintaining or improving model capabilities. These results establish SLAP as a powerful approach for efficient and effective instruction tuning of large language models. 

---
# Toxicity in Twitch Chats: An LLM-Based Analysis Across Gaming Communities 

**Authors**: Ronja Fuchs, Florian Rupp, Timo Bertram, Kai Eckert, Alexander Dockhorn  

**Link**: [PDF](https://arxiv.org/pdf/2605.24000)  

**Abstract**: Toxicity in online gaming communities remains a persistent challenge, manifesting across genres, platforms, and player interactions. While much research is focused on in-game toxicity, less is known about how toxic behavior varies between gaming communities on streaming platforms. To address this shortcoming, we analyze approximately 20 million chat messages from 4,452 streams, spanning seven game genres on Twitch. We categorize messages according to Twitch's toxicity taxonomy with a pre-trained Large Language Model using zero-shot classification. The taxonomy comprises four categories and eight subclasses, including harassment, discrimination, sexual content, and profanity. Our approach achieves an F1 score of 94.5% on the TextDetox dataset and demonstrates human-model agreement comparable to inter-human agreement. Our analysis reveals that 2.4% of all messages are classified as toxic, with notable differences across genres: streams of MOBA games exhibit the highest relative rate of toxicity (3.2%), and sports games show the lowest rate (2%). Furthermore, results indicate that individual games differ significantly in their toxicity distributions, even within genres, suggesting the existence of game-specific community norms and mechanics that shape toxic behavior beyond genre-level effects. These findings offer empirical insights into genre- and game-specific toxicity patterns on Twitch and can inform more targeted moderation strategies for gaming communities. 

---
# A Multi-Probe Audit of Clinical-Interview Depression Detection Benchmarks 

**Authors**: Takehiro Ishikawa, Jon Duke  

**Link**: [PDF](https://arxiv.org/pdf/2605.23977)  

**Abstract**: This paper audits benchmark evaluation in clinical-interview depression detection through four complementary probes across DAIC/E-DAIC, CMDC, ANDROIDS, MODMA, and PDCH. First, we re-evaluate E-DAIC under strict subject-disjoint leave-one-subject-out cross-validation. A lightweight hybrid text-plus-LLM-score model reaches macro-F1 = 0.723 - the highest reported under this protocol, to our knowledge - providing a conservative out-of-fold reference point that does not depend on the privileged official holdout. Second, we test whether the E-DAIC official split supports fine-grained leaderboard rankings by sweeping 96 model configurations across modality bundles, pooling strategies, and learners. Development-side cross-validation and official-test rankings align only moderately: the best cross-validation configuration ranks twentieth on the official test, the official-test winner ranks forty-first by cross-validation, top-3 overlap is zero, and the apparent winner is rank-1 in only 32.3% of subject bootstraps. Third, we externally validate strong public CMDC and ANDROIDS baselines that achieve near-ceiling in-domain performance. Zero-shot transfer to external corpora is substantially weaker. Finally, we stress-test E-DAIC text and audio models using paired symptom-dense versus symptom-light interview slices defined by an SRDS-based annotator. Text scores rise sharply on symptom-dense slices, whereas audio scores remain nearly flat; the text-minus-audio gap is positive across all five seeds. 

---
# Improving the Completeness and Comparability of Segment Disclosures: A Large Language Model Approach 

**Authors**: Yue Liu, Zhiyuan Cheng, Longying Lai  

**Link**: [PDF](https://arxiv.org/pdf/2605.23924)  

**Abstract**: Segment-level disclosures are a central component of financial reporting, providing insight into firms' internal organization and the allocation of economic activities across operating units. However, segment information is often presented in both qualitative and quantitative forms, dispersed across tables and narrative sections of Form 10-K filings. Empirical research relying on structured databases faces both completeness and comparability challenges, as some firm-year observations may be missing, nested segment disclosures are not captured, and support for longitudinal and cross-firm comparability is limited. This study develops a large language model-based framework to extract segment disclosures directly from Form 10-K filings and to preserve both reportable and nested segment information. We further design a retrieval augmented system that incorporates information across multiple filings to support comparability. We use two representative settings to demonstrate its application: longitudinal analysis within a firm to interpret segment changes over time, and cross firm alignment of geographic segments across firms with different reporting structures. The results indicate that the artifact accurately extracts segment-level information and effectively addresses questions that require cross-period knowledge, demonstrating the potential of LLM-based approaches to enhance the measurement and interpretation of segment disclosures. 

---
# Multi-Persona Debate System for Automated Scientific Hypothesis Generation 

**Authors**: Jaeha Oh, Byungchan Kim, Ju Li, Yang Jeong Park, Jin-Sung Park  

**Link**: [PDF](https://arxiv.org/pdf/2605.23917)  

**Abstract**: Modern scientific discovery is bottlenecked not by data scarcity, but by the inability to synthesize fragmented knowledge into actionable hypotheses. This challenge is especially acute in battery materials research, where electrochemical performance, interfacial behavior, and manufacturing feasibility must be optimized simultaneously. Here, we present the Multi-Persona Debate System (MPDS), a literature-grounded framework for automated scientific hypothesis generation that combines literature retrieval, long-context large language model reasoning, corpus-driven persona induction, and structured multi-agent debate. MPDS constructs literature snapshots of up to 500 papers, grounds agents in role-specific evidence pools, and conducts a three-round citation-aware debate followed by moderator synthesis, enabling negotiation between personas while preserving evidence traceability. We evaluate MPDS using a temporally controlled protocol excluding direct access to target papers, including two held-out battery-materials case studies and a blinded comparison across 30 matched cases. In sodium-ion anode and all-solid-state battery cathode design tasks, MPDS recovered design logics aligned with experimentally validated solution spaces and generated more mechanistically explicit, process-aware proposals than simpler baselines. To assess the impact of personas and debate, we introduce Integrative Hypothesis Quality scoring. In ablation studies, MPDS achieved the highest mean score among five conditions, with its largest advantage in cross-perspective integration. A laboratory follow-up suggests utility as a diagnostic aid for identifying practical bottlenecks in workflows. These results indicate that structured debate over literature snapshots improves hypothesis formation under coupled engineering constraints and provides a reusable workflow for text-intensive scientific discovery. 

---
# Direct Preference Optimization for English-Mandarin Code-Switching Speech Recognition in Audio LLMs 

**Authors**: Trung Nguyen Quang, Cheng Yi Lewis Won, Minh Duc Pham, Yingxu He, Shuo Sun, Ai Ti Aw  

**Link**: [PDF](https://arxiv.org/pdf/2605.23975)  

**Abstract**: Audio large language models (Audio LLMs) exhibit systematic failures in transcribing code-switching speech despite strong multilingual capabilities. Focusing on English-Mandarin, we identify three failure modes: language omission, translation-instead-of-transcription, and hallucination. We apply Direct Preference Optimization (DPO) to align models, constructing preference pairs in which chosen responses preserve mixed-language content while rejected responses mimic failure patterns. Training three Audio LLMs on 100K pairs (570 hours), we observe consistent behavioral shifts: models learn to preserve language composition rather than translating when prompted for transcription. This alignment yields MER reductions up to 89.6% (in-distribution) and 20.0% (out-of-distribution). Our findings suggest DPO can effectively elicit correct code-switching transcription behavior from multilingual Audio LLMs. 

---
# STORM: Internalized Modeling for Spatial-Temporal Reasoning in Video-Language Models 

**Authors**: Yiming Liang, Yixiao Chen, Yiyang Zhou, Yixuan Wang, Shoubin Yu, Andong Deng, Fuxiao Liu, Qin Zhang, Chen Chen, Mohit Bansal, Huaxiu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2605.26014)  

**Abstract**: Many video reasoning tasks require tracking motion, temporal order, and evolving visual states across frames. Existing methods built on large vision-language models (LVLMs) often address this challenge by externalizing reasoning through textual chain-of-thought (CoT), keyframe selection, repeated frame reinsertion, or external tool use. While effective, such pipelines increase inference-time latency and engineering complexity, and they force temporal-visual evidence to be serialized into text or repeatedly re-encoded from frames. Inspired by the intuition that visual reasoning can occur implicitly before verbalization, we propose STORMS (Spatial-Temporal reasOning via inteRnalized Modeling), a two-stage framework that teaches LVLMs to reason through bounded continuous latent trajectories instead of explicit textual CoT. In Stage I, STORMS aligns latent tokens with thought-video representations derived from generated videos, grounding the latent states in dynamic visual evidence. In Stage II, the model is further trained with answer-only supervision, encouraging the reasoning process to be internalized without step-by-step annotations. Generated thought videos are used only during training; at inference, STORMS performs a bounded latent rollout without regenerating videos, reinserting frames, or invoking external visual tools. Experiments on VideoMME, MVBench, TempCompass, and MMVU show that STORMS improves video reasoning accuracy while substantially reducing inference overhead compared with tool or video-generation-based reasoning pipelines. 

---
# Prism: A Plug-in Reproducible Infrastructure for Scalable Multimodal Continual Instruction Tuning 

**Authors**: Jun-Tao Tang, Yu-Cheng Shi, Zhen-Hao Xie, Da-Wei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.26110)  

**Abstract**: Multimodal Large Language Models (MLLMs) achieve versatility by reformulating diverse tasks into a unified instruction-following framework via instruction tuning. However, real-world deployment requires continuous adaptation to emerging tasks, motivating Multimodal Continual Instruction Tuning (MCIT). Despite its growing importance, current MCIT research is hindered by severe engineering bottlenecks. Existing methods are typically implemented by directly modifying the base MLLM codebase, which imposes substantial implementation overhead and yields method-specific architectures that severely limit code reuse and fair comparison. To address this, we introduce Prism, a plug-in reproducible codebase specifically designed for scalable MCIT research. It separates algorithmic development from the backbone implementation via a lightweight plugin registration mechanism, enabling new strategies to be integrated as independent plugins without modifying the underlying MLLM codebase, thereby eliminating structural fragmentation and accelerating method development. Prism natively supports widely used large-scale training pipeline, thereby enabling reproducible and scalable MCIT experimentation. Code is available at this https URL. 

---
# When Self-Belief Misleads: Active Label Acquisition for Reinforcement Learning with Verifiable Rewards 

**Authors**: Li Wang, Xiaodong Lu, Xiaohan Wang, Yikun Ban, Jiajun Chai, Wei Lin, Tianhao Peng, Guojun Yin  

**Link**: [PDF](https://arxiv.org/pdf/2605.25864)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable advancements in reasoning capabilities empowered by Reinforcement Learning with Verifiable Rewards (RLVR). Nonetheless, RLVR intrinsically relies on ground-truth labels for reward computation, the acquisition of which is often prohibitively expensive in real-world scenarios. While unsupervised RLVR paradigms attempt to circumvent this by training on pseudo-labels, they are notoriously susceptible to training collapse. Moreover, different samples often exhibit varying annotation values. In this paper, we propose Reinforcement Learning with Active Verifiable Rewards (RLAVR), which actively acquires ground-truth labels for a small set of selected samples and integrates them with pseudo-labels, thereby stabilizing training dynamics and improving performance under limited annotation budgets. To identify valuable samples, we propose the Corrective Advantage Gap (CAG) metric and analyze the sample-level supervision value. Building on this, we introduce Correction-Aware Reliability Estimation for RLAVR (CARE), which translates the oracle CAG criterion into a practical pre-query acquisition policy to substantially improve training stability. Extensive experiments across diverse domains, model families, and model scales demonstrate the effectiveness and generality of our approach. Our code is available at this https URL. 

---
# Neural Router: Semantic Content Matching for Agentic AI 

**Authors**: Lauri Lovén, Abhishek Kumar, Alexander Engelhardt, Alaa Saleh, Roberto Morabito, Xiaoli Liu, Naser Hossein Motlagh, Sasu Tarkoma  

**Link**: [PDF](https://arxiv.org/pdf/2605.25701)  

**Abstract**: Large language models (LLMs) can serve as the semantic-matching engine of a content-based publish/subscribe broker for agentic AI across the edge-cloud computing continuum, bridging the vocabulary and modality gaps that defeat keyword and embedding filters. Framed as offline multi-label retrieval over three public datasets spanning social-media, legal, and smart-home sensor domains (six LLMs, seven baselines), our central contribution is a two-crossover cost-accuracy characterisation: an analytical context-window crossover below which a CoverAndMerge compression pipeline reduces LLM invocations, and an empirical discrimination-capacity crossover above which matching accuracy collapses independently of context budget, by a model-dependent factor of parameter count and training generation. Two findings carry practical weight: above the discrimination crossover, compression cannot recover accuracy and only frontier-scale models clear large subscription sets; and there backend choice dominates configuration choice, so model selection, not pipeline tuning, is the primary operator lever. We accompany this with three composable algorithms and a per-cluster Quality-of-Experience framework for autonomic LLM-tier selection. 

---
# RotMoLE: Enhancing Mixture of Low-Rank Experts through Rotational Gating Mechanism 

**Authors**: Mengyang Sun, Maochuan Dou, Tao Feng, Dan Zhang, Yihao Wang, Junpeng Liu, Yifan Zhu, Jie Tang  

**Link**: [PDF](https://arxiv.org/pdf/2605.25565)  

**Abstract**: While Large Language Models (LLMs) are commonly fine-tuned to handle domain-specific tasks before being applied to vertical applications, adapting them to complex scenarios with diverse specialized knowledge remains challenging. Meanwhile, Mixture-of-Experts (MoE) architecture has risen as a crucial paradigm for training LLMs, and some recent works have also incorporated MoE into Parameter-Efficient Fine-Tuning (PEFT) to propose the Mixture of Low-rank Experts (MoE-LoRA), to enhance the power of low-rank adapters for learning complicated knowledge. However, conventional gating mechanisms in MoE typically apply only a scalar reweighing to selected experts, thereby limiting their underlying capacity of representation and generalization. Motivated and enabled by the low-rank structures in MoE-LoRA, we propose RotMoLE, a specialized MoE framework for low-rank experts featuring an additional rotation gate. Beyond simple scaling, RotMoLE implements a rotation mechanism for each selected expert, enabling superior expert exploitation and specialization for learning diverse data, especially when expert candidates are limited. Empirical results on complex multi-task and multilingual training scenarios validate our effectiveness. 

---
# Directional Alignment Mitigates Reward Hacking in Reinforcement Learning for Language Models 

**Authors**: Wenlong Deng, Jiaji Huang, Kaan Ozkara, Yushu Li, Christos Thrampoulidis, Xiaoxiao Li, Youngsuk Park  

**Link**: [PDF](https://arxiv.org/pdf/2605.25189)  

**Abstract**: Reward hacking arises when a model improves a proxy reward by exploiting shortcuts rather than solving the intended task. We study this failure mode through the geometry of reinforcement learning updates in language models and argue that hacking emerges when optimization drifts away from a stable low-dimensional learning trajectory. We analyze this drift through dominant singular directions of parameter updates and show that reward-hacking runs exhibit substantially larger directional change than clean runs. Motivated by this observation, we introduce trusted-direction projection, which constrains gradients to remain within a clean reference subspace. Across reward-hacking experiments on mathematical reasoning, the proposed approach delays shortcut exploitation and better preserves task performance. 

---
# MAGIC: Multimodal Alignment & Grounding-aware Instruction Coreset for Vision-Language Models 

**Authors**: Shristi Das Biswas, Kaushik Roy  

**Link**: [PDF](https://arxiv.org/pdf/2605.26004)  

**Abstract**: Instruction tuning of large vision-language models (LVLMs) increasingly depends on massive multimodal corpora, yet these datasets contain samples with substantial redundancy, low visual dependency, and highly imbalanced coverage of multimodal reasoning behaviors. As a result, uniform subsampling or naive score-based selection often yields suboptimal training subsets. We introduce MAGIC, a training-free, forward-only coreset selection method designed to construct compact yet behaviorally faithful subsets for multimodal instruction tuning. MAGIC is built on three intrinsic signals extracted from a pretrained VLM: Multimodal Gain, which measures the likelihood improvement obtained from visual input; Bridging Relevance, which captures the sharpness of answer-token grounding over visual tokens; and Skill-Neuron Signatures, which characterize the functional computation elicited by each sample via top-activated feed-forward neurons. MAGIC combines these signals in a three-stage pipeline: filtering low-gain examples, ranking candidates by a normalized quality objective, and performing bucket-wise budget allocation over discrete neuron signatures to preserve latent multimodal skill coverage. This formulation avoids backpropagation, auxiliary selector training, and expensive clustering in continuous activation spaces, while remaining efficient and easily deployable in existing VLMs. Across LLaVA-665K and Vision-Flan datasets, and transfer settings to large target models, LLaVA-1.5-7B and -13B, MAGIC consistently improves over strong baselines under matched 20% budgets: it achieves 100.3% relative performance to full finetuning on LLaVA-665K and 101.6% relative performance on Vision-Flan-186K, while yielding a 73.7% reduction in wall-clock run time. 

---
# AgentIR: A Workload-Adaptive Cascade Retrieval Substrate for Long-Term Conversational Memory 

**Authors**: Aojie Yuan, Haiyue Zhang, Shahin Nazarian  

**Link**: [PDF](https://arxiv.org/pdf/2605.25092)  

**Abstract**: Long-term conversational memory is a retrieval workload classical IR was not built for: the index grows during the query stream, query types shift intra-session, and the latency budget per retrieval is sub-10 ms. Lucene-class engines treat the index as static and the query as stateless, leaving the workload's structure unexploited.
AgentIR treats fusion as a per-query decision along two axes: which fusion to apply (BM25, Dense, RRF, or agent-aware RRF), and whether the ~52 ms dense channel is worth running at all. The second axis is a confidence-triggered cascade router that decides from the BM25 top-k margin alone and re-tunes across workloads without retraining. On LongMemEval (n=500), where the dense channel does add information, the cascade skips 63% of queries at parity LLM-judged accuracy (2.67x faster under two judges, paired bootstrap p>=0.88); per-qtype thresholds extend this to 5.76x under 5-fold cross-validation. On LoCoMo (n=1,982), where BM25 alone is already the strongest single system, the same trigger auto-tunes to a 100% skip rate (132x faster, +0.089 Hit@5). Capacity on a shared 8-core VM rises from ~154 to ~1,400 concurrent agents (9x).
Underneath the cascade, a time-partitioned index does O(log 1/epsilon) work independent of corpus size: 1234x corpus growth costs only 3.6x latency, ending in 1769x over sequential at sub-100 us p50 on 5M records. At parity quality with Lucene on 9 BEIR datasets up to 8.8M docs, the substrate runs 10x geo-mean over Pyserini 8T and 11x over PISA-1T BlockMax-WAND; an A100 reaches 1.8-39x over Pyserini 8T; chunked index build sustains 56.8K docs/sec on MS MARCO. Three subtle BM25/GPU correctness pitfalls that silently regress nDCG@10 by 6-8x are documented and fixed; post-fix CPU and GPU agree within 0.0002 nDCG@10 on all eight datasets that fit a single A100. 

---
# RouteScan: A Non-Intrusive Approach to Auditing MoE LLMs Safety via Expert Routing Telemetry 

**Authors**: Bo Lv, Zhiheng Xu, KeDong Xiu, Ruyi Ding, Tianhang Zheng, Zhibo Wang, Kui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2605.24817)  

**Abstract**: Mixture-of-Experts (MoE) architectures have become an increasingly important paradigm for scaling Large Language Models (LLMs). As MoE models are increasingly deployed in real-world services, safety auditing becomes necessary to verify whether these models produce or facilitate harmful behaviors during operation. However, existing content-based auditing methods typically require access to user prompts, model inputs, or generated outputs, potentially exposing sensitive user information and creating a fundamental tension between LLM safety and user privacy. On the other hand, we observe that, in MoE models, sparse expert routing maps different inputs to activate different expert-execution patterns, producing measurable footprints in low-level GPU execution telemetry. Inspired by this observation, we propose RouteScan, a non-intrusive auditing framework for detecting harmful behaviors through GPU-level expert routing telemetry. Specifically, RouteScan utilizes the number of active GPU threads allocated to expert modules during the prefilling phase as a discriminative micro-architectural fingerprint, and builds a lightweight detection pipeline that isolates cross-domain invariant risk indicators for the precise identification of malicious prompts. Comprehensive evaluations on open-source MoE LLMs with distinct routing designs demonstrate that RouteScan achieves strong generalization, with an AUROC exceeding 0.93 on unseen harmful domains and 0.96 under novel jailbreak wrappers. Moreover, empirical inversion tests show that the collected expert routing telemetry provides limited information for prompt reconstruction, suggesting a practical privacy advantage over content-based auditing methods. 

---
# Universal Boosts, Specific Suppressors: Sparse Autoencoder Steering of Medical Vision-Language Models 

**Authors**: Farhad Nooralahzadeh, Benjamin Gundersen, Nicolas Deperrois, Hidetoshi Matsuom, Mizuho Nishio, Thomas Frauenfelder, Ahmed Allam, Christian Blüthgen, Michael Moor, Michael Krauthammer  

**Link**: [PDF](https://arxiv.org/pdf/2605.24977)  

**Abstract**: Medical vision-language models (VLMs) often hallucinate findings when generating chest X-ray reports: they fabricate findings that are not present in the image, miss important ones, or locate them incorrectly. We mitigate this without weight updates by decoding-time residual steering on a per-token sparse autoencoder (SAE) basis: Top-$K$ SAEs on late layers, causal steering against clinical errors, then combined suppress/boost intervention at inference time. On the MIMIC-CXR test split, our inference-only method improves the quality of generated reports for three radiology VLMs (RadVLM, LLaVA-Rad, and CheXOne), with relative improvements of +5.4%, +7.2%, and +17.0% in the clinical composite metric, and statistically significant GREEN gains on all backbones. A cross-model feature alignment shows that the quality-promoting (boost) directions overlap strongly across architectures, whereas hallucination-linked (suppress) directions are model-specific. Therefore, transferable steering must treat suppression per-backbone, rather than sharing a universal suppress list. The same recipe transfers zero-shot to IU-Xray (Green $+7.7\%$ rel.) without retraining, confirming that the identified features are properties of the model, not of the training corpus. We release causal feature sets and an interactive feature dashboard: this https URL. 

---
# An Effective-Rank Audit of Alignment-Induced Activation Shifts: Confound Control, Constructive Calibration, and Limits 

**Authors**: Yuki Nakamura  

**Link**: [PDF](https://arxiv.org/pdf/2605.24583)  

**Abstract**: We audit alignment-induced shifts in residual-stream activations of three open-weight instruction-tuned LLMs (Llama-3.1-8B-Instruct, Gemma-2-9B-it, Qwen-2.5-7B-Instruct) using the effective rank of the alignment modification matrix on safety-relevant inputs, rho_eps := rank_eps(M_Ds)/d, which formalizes the single-refusal-direction observation of Arditi et al. (2024) as a continuous quantity. The paper has three contributions. (1) Confound-controlled measurement: a four-variant decomposition (M_naive, M_template, M_aligned, M_DiD) separates chat-template formatting, alignment-stage shift, and the refusal-mediating direction, and recovers the Arditi refusal direction on M_DiD at |cos| in {0.77, 0.86, 0.50} (Llama/Gemma/Qwen); chat-template-controlled rho_eps is {0.0029, 0.0048, 0.0044}, and the centered SVD residual is 4-7x larger. (2) Constructive calibration on a 3-layer MLP across rho_eps in {0.008, 0.17, 0.33, 0.40} exhibits a sweet-spot vs. brittle distinction: mild rank-maximization (lambda=5) buys ablation robustness, while strong regularization at the same nominal rho_eps (lambda=50) does not. rho_eps is a diagnostic for fragility, not a target whose mechanical inflation buys robustness. (3) Limits of rank-based diagnostics: (a) not safety-specific (LRH baseline is 2-3x the safety value); (b) SVD principal ordering does not match causal ordering (Llama u_2 inert despite ranking second; cumulative ablation non-monotone at k=5); (c) the spectral-gap hypothesis required to upgrade the O(rho_eps * d) achievability bound to a matching Mirsky-route lower bound fails empirically (1/90 Llama layer-reference pairs, 0/36 MLP combinations) and structurally (kappa_lb <= 2/(eps * r)). The matching lower bound remains an open problem. 

---
# ECHO: Terminal Agents Learn World Models for Free 

**Authors**: Vaishnavi Shrivastava, Piero Kauffmann, Ahmed Awadallah, Dimitris Papailiopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2605.24517)  

**Abstract**: CLI agents are the closest thing language models have to an embodied setting: the model emits commands, the terminal executes them, and the returned stream -- stdout, errors, files, logs, and traces -- records the consequences. We argue that this stream is a supervision signal, but standard agent RL discards it: GRPO-style training updates action tokens with sparse outcome-level rewards while ignoring environment responses already in the rollout. Failed rollouts provide little policy-gradient signal despite containing rich evidence about how the environment responds. We introduce ECHO (Environment Cross-entropy Hybrid Objective), a hybrid objective that combines the standard policy-gradient loss on action tokens with an auxiliary loss that trains the policy to predict environment observation tokens resulting from its own actions. ECHO reuses the same forward pass as GRPO, requires no additional rollouts, and turns terminal feedback into dense supervision for all rollouts. ECHO doubles GRPO pass@1 on TerminalBench-2.0: Qwen3-8B improves from 2.70% to 5.17%, and Qwen3-14B from 5.17% to 10.79%. ECHO also produces policies that better predict terminal dynamics, even on trajectories they did not generate: across held-out rollouts, it sharply reduces environment-token cross-entropy while GRPO alone barely changes it. From base Qwen3-8B, ECHO matches expert-SFT-then-GRPO performance on held-out terminal tasks without expert demonstrations, and recovers roughly half of the expert-SFT initialization benefit on TerminalBench-2.0. In some settings, the environment prediction loss alone enables verifier-free self-improvement, allowing policies to improve on unseen OOD tasks by learning only from environment interactions. Together, these results suggest that environment observations are not merely context for future actions, but a dense, on-policy supervision signal already present in every rollout. 

---
# MindAlign: Bridging EEG, Vision, and Language for Zero-Shot Visual Decoding 

**Authors**: Zexuan Chen, Sichao Liu, Runhao Lu, Huichao Qi, Alexandra Woolgar, Xi Vincent Wang, Lihui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.24523)  

**Abstract**: Visual decoding from brain signals is a key challenge at the intersection of computer vision and neuroscience, requiring methods that bridge neural representations and computational models of vision. We introduce a tri-modal contrastive framework for EEG-based visual decoding that aligns EEG, visual, and textual representations within a unified latent space. Our approach follows a two-stage design. First, we pre-train an EEG encoder via masked reconstruction on unlabeled trials, learning spatio-temporal regularities that transfer robustly to downstream tasks. Second, we jointly align EEG, image, and LLM-generated textual descriptions through contrastive learning, where text supervision acts as a semantic regularizer that injects linguistic structure into the shared space without overwhelming the primary EEG-image signal. The encoder integrates subject-specific adaptation, graph-attention over channels, and temporal-spatial convolutional embeddings. On the Things-EEG2 200-way zero-shot benchmark, our framework achieves 54.1% Top-1 and 83.4% Top-5 accuracy, substantially exceeding the strongest prior baseline (32.4% / 64.0%), with paired Wilcoxon tests confirming significance (p < 0.01) over all in-subject baselines. We validate generalization on Things-MEG. Analysis reveals that compact embedding geometries (CN-CLIP) outperform much larger backbones, and that decoding aligns with established neurophysiology of visual processing. This work is a critical step towards robust, semantically-grounded visual decoding from non-invasive temporal neural signals. The source code is publicly available in this https URL. 

---
# SliceWorld: A Predictive and Controllable World-State Model for CT Report Generation 

**Authors**: Yuanhe Tian, Yan Song  

**Link**: [PDF](https://arxiv.org/pdf/2605.24371)  

**Abstract**: CT report generation (CTRG) requires models to summarize three-dimensional anatomical context and pathological findings from hundreds of axial slices. Existing methods typically learn a direct image-to-text mapping, providing limited mechanisms for modeling how CT evidence evolves across slices or how reports respond to controlled changes in latent lesion-related factors. We propose SliceWorld, a CT-specific world-state framework that treats an axial CT scan as an ordered sequence along the z-axis. SliceWorld encodes prefix CT evidence into factor-aware latent states containing anatomy, lesion, and uncertainty components, and projects these states into world tokens used for multi-step future-slice feature prediction, lesion-factor intervention, and LLM-based report generation. The model is first pretrained on CT slice sequences with predictive, factor-aware, and counterfactual objectives, and is then fine-tuned on paired CT-report data. Experiments on M3D-Cap and CT-RATE show that SliceWorld improves natural language generation metrics and clinically oriented automatic evaluation. Further analyses demonstrate multi-horizon future-slice prediction, measurable factor alignment, reduced-slice robustness, and selective lesion-sensitive report modulation. 

---
# Faithfulness as Information Flow: Evaluating and Training Faithful Chain-of-Thought Reasoning 

**Authors**: Jinghan Jia, Joe Benton, Eric Easley  

**Link**: [PDF](https://arxiv.org/pdf/2605.24286)  

**Abstract**: Chain-of-thought (CoT) reasoning is useful for monitoring language models only when the reasoning trace faithfully reflects the computation that produces the final answer. However, models can rely on prompt-to-answer shortcuts that bypass the CoT, making the visible reasoning trace misleading even when it appears plausible. We study CoT faithfulness through a structural information-flow perspective: faithful reasoning should route answer-relevant information through the mediated path from prompt to CoT to answer, rather than through a direct prompt-to-answer shortcut. This perspective yields a task-agnostic framework based on three complementary properties, sufficiency, completeness, and necessity, which we instantiate with entropy-based, masked-KL, and gradient-based diagnostics. We show that these metrics recover externally judged faithfulness differences in hinted reasoning, and identify a low-entropy failure mode of KL-based diagnostics where gradient-based measures remain more stable. Building on this analysis, we introduce update-time interventions for verifier-based on-policy RL, including attention masking, backward-only gradient masking, CoT gradients, and adversarial perturbations of prompt representations. Across hinted arithmetic, reward-hackable code repair, and DAPO-Math models trained without hints but evaluated under wrong-hint injection, our interventions shift behavioral and structural indicators toward stronger CoT mediation. In particular, they make shortcut and reward-hacking behavior more transparent in the CoT and improve task-agnostic faithfulness metrics, while in some settings also reducing wrong-hint susceptibility. Our results suggest that controlling information flow during training is a practical route toward more faithful and monitorable CoT reasoning. Code is available at this https URL. 

---
# DUEL: Adversarial Self-Play for Multimodal Reasoning 

**Authors**: Lin Qiu, Hanqing Zeng, Yao Liu, Bingjun Sun, Guangdeng Liao, Ji Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.24794)  

**Abstract**: Reinforcement learning (RL) has emerged as an effective paradigm for improving the reasoning capability of vision-language models (VLMs). However, RL-based optimization typically depends on costly high-quality annotations that are difficult to scale. Existing unsupervised alternatives may drift toward biased solutions due to weak visual grounding and the lack of reliable verification signals. We propose a self-evolving post-training framework, DUEL, where supervision emerges from adversarial interactions between two policies initialized from the same pretrained VLM. A Challenger generates an image-grounded true claim together with a minimally perturbed hard-negative counterpart, while a Solver verifies both claims against the image, encouraging fine-grained visual discrimination under near-neighbor semantics. To stabilize optimization, we introduce a length-normalized log-likelihood reward that preserves informative optimization signals beyond binary outcome supervision and improves learning stability under sparse feedback. Experiments show that DUEL consistently improves visual reasoning and robust discrimination without additional human annotations, external reward models, or image editing tools. 

---
# The Multilingual Curse at the Retrieval Layer: Evidence from Amharic 

**Authors**: Yosef Worku Alemneh, Kidist Amde Mekonnen, Maarten de Rijke  

**Link**: [PDF](https://arxiv.org/pdf/2605.24556)  

**Abstract**: Multilingual retrieval increasingly underpins cross-lingual question answering and retrieval-augmented generation. Strong zero-shot scores on multilingual benchmarks are often taken as evidence that current encoders transfer reliably across many languages. We argue that this assumption breaks down for underrepresented, morphologically rich languages, and use Amharic as a diagnostic case. Under a shared passage retrieval protocol covering dense, late-interaction, learned sparse, and cross-encoder paradigms, we compare zero-shot multilingual retrievers, Amharic-fine-tuned multilingual retrievers, and monolingual Amharic retrievers. The strongest zero-shot multilingual retriever underperforms the strongest monolingual Amharic first-stage retriever by 23% relative MRR@10. Fine-tuning two recent multilingual embedding models on the same Amharic supervision yields 32-60% relative MRR@10 gains over zero-shot, but the best Amharic-fine-tuned multilingual model remains below the strongest monolingual Amharic retriever. These findings indicate that zero-shot multilingual retrieval is not a sufficient proxy for equitable information access in the LLM era: for underrepresented languages, retrieval must be evaluated and adapted in-language rather than inferred from aggregate multilingual benchmarks. To foster future research, we publicly release the dataset, codebase, and trained models at this https URL. 

---
# Can LoRA Fusion Support Cross-Domain Tasks in Cloud-Edge Collaboration? 

**Authors**: Yatong Wang, Fali Wang, Naibin Gu, Zheng Lin, Zhengxiao Liu, Dingyu Yao, Zhiwei Zhang, Jianxin Shi, Weiping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.23913)  

**Abstract**: Cloud-hosted large language models (LLMs) commonly rely on LoRA for domain adaptation, yet domain data are distributed across multiple edge devices and cannot be uploaded due to privacy constraints. This raises a fundamental question: how can knowledge from multiple private edges be integrated into a cloud LLM for cross-domain problem solving? A natural solution is to train LoRA adapters locally and fuse them in the cloud; however, existing pipelines rely on unrealistic assumptions that edge devices can host cloud-scale LLMs and are evaluated mainly on single-domain tasks. To address these limitations, we propose a prune-train-recover framework that enables local LoRA training on pruned models and privacy-preserving cloud integration. We further introduce MMLU-CD, a cross-domain benchmark that composes multiple domain samples into a single instance, enabling explicit evaluation of cross-domain problem solving. This allows us to ask a concrete question: Can existing LoRA fusion methods support cross-domain tasks in cloud-edge collaboration? Our empirical answer is negative. Existing LoRA fusion methods perform poorly on MMLU-CD, often underperforming the base LLM, revealing their inability to support cross-domain problem solving. We attribute this failure to parameter conflicts among LoRA adapters and propose a simple conflict-resolution module, LoRA-CR, which mitigates conflicting updates and improves LoRA fusion performance by up to 3.8%. These results identify conflict mitigation as a critical yet largely overlooked factor in cloud-edge LoRA fusion, warranting further investigation in future research. 

---
# Adaptive Preference Optimization with Uncertainty-aware Utility Anchor 

**Authors**: Xiaobo Wang, Zixia Jia, Jiaqi Li, Qi Liu, Zilong Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.10515)  

**Abstract**: Offline preference optimization methods are efficient for large language models (LLMs) alignment. Direct Preference optimization (DPO)-like learning, one of the most popular approaches, stands out for its efficiency in reward modeling. However, these methods typically follow the convention to use Bradley-Terry (BT) reward modeling that faces several critical assumptions, including the requirement for pairwise training data, model distribution shifting, human rationality assumption, etc. To address these limitations, we propose a general framework for offline preference optimization methods, Adaptive Preference Optimization with Utility Anchor (UAPO), which introduces an anchoring function to estimate the uncertainties brought from preference data annotation. Our method enables training even in scenarios where the data is unpaired, significantly enhancing data utilization efficiency. Moreover, the anchor design makes UAPO more robust in the training process. Experimental results demonstrate that UAPO achieves competitive outcomes without the strict dependency on data pairing, paving the way for more flexible and effective preference optimization methods. 

---
# RAG-Match: Retrieval-Augmented Knowledge Injection and Hierarchical Reasoning for Calibrated Semantic Relevance 

**Authors**: Hengjun Jiang, Liansheng Sun, Yan Jiang, Xiaojie Ke, Yongjin Wang, Xiangkun Liu, Cunxin Gu, Jian Xu, Guanjun Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2605.25486)  

**Abstract**: Semantic relevance judgment for search is particularly challenging in knowledge-intensive scenarios, where accurate ranking requires not only semantic matching but also background grounding, multi-step reasoning, and well-calibrated decision boundaries. Existing relevance models mainly rely on direct label supervision or shallow semantic similarity, which limits their ability to handle implicit intent, factual equivalence, and fine-grained relevance distinctions. To address this issue, we propose \textsc{RAG-Match}, a three-stage framework that integrates knowledge-augmented pretraining, hierarchical reasoning alignment, and preference-based decision calibration for relevance modeling. The key idea is to first strengthen query-centered semantic grounding, then align the model with structured relevance reasoning, and finally correct decision-level inconsistencies in difficult boundary cases. Experimental results on a real-world search relevance benchmark show that \textsc{RAG-Match} consistently outperforms strong LLM-based baselines across multiple ranking metrics, demonstrating the effectiveness of combining knowledge injection, reasoning supervision, and preference optimization for fine-grained relevance judgment. 

---
# Meta-Modal Agent: Sequential Evidence Routing for Missing-Modality Candidate Reranking 

**Authors**: Jinze Wang, Yangchen Zeng, Tiehua Zhang, Lu Zhang, Yuze Liu, Zhishu Shen, Jiong Jin, Zhu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2605.25007)  

**Abstract**: Missing modalities cause severe failures in multimodal recommender systems. User histories, item text, and visual evidence are frequently absent during cold-start scenarios, exactly when recommendation quality matters most. Existing approaches recover absent signals through imputation, feature propagation, or generative reconstruction, but these strategies can inject unsupported evidence when the surviving signals are weak. We introduce the Meta-Modal Agent (MMA), a large language model based candidate-pool reranker that treats missingness as a sequential evidence-routing problem. MMA is trained with balanced missingness-task reinforcement learning over masked-modality episodes and is evaluated in two variants: MMA-Auto, which uses only automated text, image, and graph tools, and MMA-Interactive, which additionally permits clarification questions grounded in surviving modalities as an upper-bound diagnostic. MMA operates after a first-stage retriever has produced a candidate pool; it scores those candidates rather than retrieving items from the full catalog. Final reranking fuses MMA scores with first-stage retrieval scores selected on validation data. Our evaluation is organized around four evidence checks required for a robust missing-modality claim: oracle-free one-observed-modality availability (OOMA) robustness, per-modality OOMA breakdowns, fixed-pool full-catalog reranking, and a deterministic-router mechanism control. MMA-Auto improves target-positive OOMA NDCG@10 by 4.0% and fixed-pool full-catalog reranking NDCG@10 by 12.7% over the strongest non-interactive baseline. RuleRouter-Fuse, which uses the same tools and fusion rule without learned policy updates, underperforms MMA-Auto, supporting learned routing beyond deterministic tool fusion. MMA-Interactive adds a 4.1% upper-bound gain when clarification is available. 

---
# MVR-cache: Optimizing Semantic Caching via Multi-Vector Retrieval and Learned Prompt Segmentation 

**Authors**: Ali Noshad, Zishan Zheng, Yinjun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.24914)  

**Abstract**: To reduce LLM costs and latency, semantic caching systems must accurately identify when a new prompt matches a cached one. Current methods often rely on simplistic similarity measures, which limit their effectiveness. We introduce MVR-cache, a novel semantic caching approach that significantly improves retrieval accuracy by integrating Multi-Vector Retrieval (MVR). MVR-cache is built upon a learnable segmentation model that intelligently splits prompts, enabling fine-grained similarity comparisons via MaxSim. We derive the model's training objective from a rigorous theoretical analysis. This can ensure that optimizing this objective directly maximizes cache hits under strict correctness constraints. To solve the resulting non-differentiable combinatorial optimization problem, we leverage a reinforcement learning-based training strategy with the theoretically grounded objectives as the reward. Experimental results on established benchmarks across diverse tasks confirm that in comparison to the state-of-the-art, MVR-cache consistently increases the cache hit rates by up to 37% while maintaining the same correctness guarantees. MVR-cache is available at this https URL 

---
# MeVer at CheckThat! 2026: Cluster-Aware Hard-Negative Mining for Multilingual Scientific-Source Retrieval 

**Authors**: Juli Bakagianni, Symeon Papadopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2605.24236)  

**Abstract**: Identifying the scientific source behind a social media claim requires matching short, informal, and often multilingual claims against large collections of scientific publications, where semantically related papers may act as challenging distractors or false negatives during training. We present our submission to CheckThat! 2026 Task 1 on multilingual scientific-source retrieval, focusing on how hard-negative mining should be adapted to multi-stage retrieval pipelines for scientific-source retrieval. We propose cluster-aware hard-negative mining strategies that exploit the semantic structure of retrieved candidate pools in order to construct more informative training negatives for dense retrieval and reranking. Our experiments show that different hard-negative structures induce different retrieval behaviors. Localized cluster negatives tend to favor precision-oriented retrieval, whereas broader non-gold semantic negatives provide stronger candidate coverage and more consistent reranking performance across languages. We further study multiple LLM-based evidence-selection formulations, including direct classification, pairwise comparison, and listwise reranking prompts, and find that constrained classification prompts provide the most reliable final document selection. The final system combines a dense retriever, a multilingual cross-encoder reranker, and a selective LLM-based disagreement resolver, ranking 6th among 37 submissions in the shared task evaluation. Overall, our results suggest that hard-negative mining should be treated as a stage-aware design problem rather than as a single retrieval optimization strategy. 

---
# Same Ranking, Different Winner: How Scoring Targets Shape LLM Memory Benchmarks 

**Authors**: Sugam Panthi, Rabab Abdelfattah  

**Link**: [PDF](https://arxiv.org/pdf/2605.24060)  

**Abstract**: Conversational-memory systems increasingly transform dialogue history into facts, summaries, timelines, and other source-linked descendants, so a single source turn can coexist with several derived memories in the same retrieval index. This raises an underspecified evaluation question: which stored form should receive retrieval credit? We show that this scoring-target choice is often left implicit and can materially change benchmark conclusions. We present TIAP, a fixed-output audit that rescores saved ranked outputs under three targets -- Raw, Source, and Canonical -- without rerunning retrieval. On LoCoMo and LongMemEval-S, switching only the credited target changes nDCG on 83.4--94.0 percent of shared queries, flips target orderings on Mem0 and MemoryOS transfer runs, and reverses parser-density recommendations. A 1,902-case semantic audit further shows that relaxed source-linked credit is fully justified only 29.2 percent of the time, despite high rubric reliability in a validation subset. These results reveal target noninvariance: conclusions about memory architectures can silently flip with a single benchmark-design choice. Conversational-memory papers should therefore define and report the scoring target explicitly. 

---
# Benchmarking Patent Embeddings: A Multi-Task Evaluation of 22 Models Across Retrieval, Classification, and Clustering 

**Authors**: Amirhossein Yousefiramandi, Ciaran Cooney  

**Link**: [PDF](https://arxiv.org/pdf/2605.24297)  

**Abstract**: Which fine-tuning signals improve patent embedding models, and do gains transfer across patent landscapes? We benchmark 22 embedding models, from 22M-parameter encoders to 12B instruction-tuned LLMs, on retrieval, classification, and clustering. The study uses 113,148 WIPO assistive-technology patents, 46,069 citation-graph retrieval queries, and the public DAPFAM dataset for external validation. Our framework covers citation-based retrieval, hybrid sparse-dense fusion, multi-label classification over five datasets, unsupervised clustering, six text-section views, domain-adaptive fine-tuning of four models, jurisdiction analysis, and proprietary DWPI (Derwent World Patents Index, Clarivate) expert-written content. Results show that fine-tuning is task-dependent: single-landscape tuning can improve in-domain scores but often hurts retrieval on an external landscape, challenging the assumption that more domain data always helps. Within model families, scale usually predicts performance (Qwen3 0.6B to 4B to 8B; Llama-Nemotron 1B to 8B), but cross-family scaling is noisy: the 12B KaLM-Gemma3 ranks 8th on TAC retrieval, while Qwen3-0.6B leads ARI clustering. Title+Abstract+Claims is the most reliable text representation. Multi-view abstract-claim alignment improves retrieval by up to 7.1 percent nDCG@10, while combined fine-tuning gives the strongest classification gains (+7.1 F1). All models drop by 55-65 percent on out-of-domain queries, and hybrid sparse-dense fusion does not close this gap. BM25-dense interpolation gives modest nDCG@10 gains (+0.002 to +0.015), with larger benefits for weaker zero-shot dense models. Code and evaluation framework are publicly available. 

---
# Memento: Personalized RAG-Style Long-Retention Data Scaling for META Ads Recommendation 

**Authors**: Xiaoyu Chen, Ruichen Wang, Jieming Di, Suofei Feng, Nafis Abrar, Lilly Kumari, Tony Tsui, Yilin Liu, Yu Lu, Sowmya Patapati, Junwei Xiong, Qiao Yang, Dorothy Sun, Yang Cao, Victor Chen, Pan Chen, Ramsundar Sundarkumar, Shivendra Pratap Singh, Arnold Overwijk, Ling Leng, Dinesh Ramasamy, Sri Reddy, Robert Malkin, Sandeep Pandey  

**Link**: [PDF](https://arxiv.org/pdf/2605.24051)  

**Abstract**: Modeling of long history data suffers from long-context window attention dilution, system efficiency and catastrophic forgetting problems, where naive linear scaling approach like LastN would fail. We introduce Memento, a personalized retrieval-augmented framework that treats historical user engagements as a document corpus and ad requests as queries, retrieving relevant interactions via Maximal Marginal Relevance (MMR) to balance similarity with diversity. We identify two complementary applications: Representation Memento, which retrieves historical embeddings for feature augmentation, and Data Memento, which retrieves past training examples for multipass training. Through infrastructure co-design -- temporal chunking, INT8 quantization, and asynchronous serving -- Memento achieves 5-10$\times$ resource efficiency over linear scaling. Memento processes daily requests with sub-10ms latency, yielding 0.25-0.3% Normalized Entropy gain on both click-through and conversion prediction. In production, Memento delivers a 1% CTR lift on Facebook Feed and Reels and a 1.2% CVR lift, scaling personalization to 365+ days of history. 

---
