# ClinEnv: An Interactive Multi-Stage Long Horizon EHR Environment for Agents 

**Authors**: Yuxing Lu, Yushuhong Lin, Wenqi Shi, J. Ben Tamo, Xukai Zhao, Jinzhuo Wang, May Dongmei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.02568)  

**Abstract**: Clinical practice is not the selection of an answer from enumerated options: a physician gathers heterogeneous information incrementally and commits to sequential, irreversible decisions under uncertainty. Static benchmarks cannot probe and existing interactive medical benchmarks each compromise on at least one of them. We present ClinEnv, an interactive benchmark that evaluates LLMs as attending physicians over real inpatient admissions under a paradigm we term Longitudinal Inpatient Simulation. Each case is automatically constructed into an ordered sequence of decision stages; at every stage the model must actively query four specialized agents before committing to medications, procedures, and diagnoses. ClinEnv scores both what the model decides, through deterministic ontology-grounded matching, and how it gathers information. Across seven models, the strongest reaches only 0.31 decision F1, and outcome quality is sharply decoupled from process quality. Difficulty concentrates in management decisions and later stages, where models recover discharge diagnoses far more reliably than management actions (0.51 vs. 0.17 F1) and continue to issue redundant queries as cases progress. ClinEnv makes this information-acquisition gap, invisible to outcome-only evaluation, directly measurable. 

---
# Tracking the Behavioral Trajectories of Adapting Agents 

**Authors**: Jonah Leshin, Manish Shah, Ian Timmis  

**Link**: [PDF](https://arxiv.org/pdf/2606.02536)  

**Abstract**: Text files such as skill files, memory files, and behavioral configuration files play a central role in defining how modern agents act. Through edits by humans or the agents themselves, these files may evolve over time, directly steering the agent's behavior in future interactions. We present a methodology and framework for measuring agent $traits$ by defining traits as directions in the embedding space of a text embedding model. We train a linear model on labeled "before" versus "after" skill file diffs to learn a trait vector, then score arbitrary skill edits by projecting their embedding diffs onto this vector. Evaluated on 68 labeled skill diff pairs for the trait of propensity to seek sensitive data, our method achieves 91.2% sign classification accuracy and a Spearman rank correlation of $\rho = 0.82$ under leave-one-out cross-validation. We build this trait evaluation into a broader agent-to-agent protocol that enables one agent to evaluate another's skill file updates through a trusted intermediary. 

---
# SafeSteer: Localized On-Policy Distillation for Efficient Safety Alignment 

**Authors**: Hao Li, Jingkun An, Zijun Song, Pengyu Zhu, Rui Li, Hao Wang, Wendi Feng, Yesheng Liu, Lijun Li, Jin-Ge Yao, Lei Sha  

**Link**: [PDF](https://arxiv.org/pdf/2606.02530)  

**Abstract**: Aligning Large Language Models (LLMs) with human values often degrades their general capabilities, termed the alignment tax. Existing methods mitigate this by balancing dual objectives, which heavily rely on massive general-purpose data or auxiliary reward models.
In this paper, we argue that, because safety features are inherently sparse within the output distribution, alignment requires localized modifications rather than global trade-offs. To this end, we propose SafeSteer, which performs on-policy distillation confined to safety tokens. First, we construct a safety teacher via activation steering. Based on this teacher, we develop a safety token selection algorithm. Consequently, SafeSteer restricts the reverse KL penalty to these tokens during training to preserve general capabilities.
Experimental results across diverse models show that our SafeSteer achieves a superior trade-off between safety and general capability compared with existing methods, attaining strong safety performance on seven safety benchmarks with only minimal degradation on five general capability benchmarks. Notably, SafeSteer requires only 100 harmful samples without using any general-purpose data, less than 1% of what previous baselines used, considerably reducing alignment cost. More details are on our project page at this https URL. 

---
# Bridging the Last Mile of Time Series Forecasting with LLM Agents 

**Authors**: Yuhua Liao, Zetian Wang, Qiangqiang Nie, Zhenhua Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.02497)  

**Abstract**: Time series forecasting has advanced rapidly, especially with the emergence of foundation models that show strong zero-shot performance on numerical extrapolation. However, in real-world forecasting settings, a statistically plausible baseline is rarely the final forecast used in practice. Before a forecast becomes decision-ready, it often needs to be revised using weakly structured business context such as holiday effects, campaign plans, external events, historical analogs, and expert feedback. This practical stage remains underexplored in the forecasting literature. In this paper, we formulate this stage as the \textbf{last-mile forecasting} problem and present an LLM-agent framework that sits on top of a forecasting backbone. Our system maintains a unified forecast workspace, invokes tools to retrieve contextual evidence, and converts reasoning trajectories into explicit forecast revision actions under structural safety constraints. It also supports long-horizon forecasting through map-reduce-style decomposition and post-hoc reflection through a memory bank. The resulting system is designed to be controllable and auditable. Through real-world case studies, we show how LLM agents can bridge the gap between statistical prediction and business-ready forecasting. 

---
# RASER: Recoverability-Aware Selective Escalation Router for Multi-Hop Question Answering 

**Authors**: Yuyang Li, Zihe Yan, Tobias Käfer  

**Link**: [PDF](https://arxiv.org/pdf/2606.02488)  

**Abstract**: Multi-hop question-answering systems often use expensive retrieval on every question. They may decompose the question, run several retrieval rounds, or search through bridge entities before answering. All of these strategies rely on repeated LLM calls to rewrite or decompose the question, which increases extra token cost, and it is not fitting when the LLM budget is tight. However, our analysis shows that lots of multi-hop questions are already answered correctly by a single one-shot RAG, so running an extra retrieval on every question wastes the budget. We introduce RASER (Recoverability-Aware Selective Escalation Router), a family of cheap routers built on one-shot RAG and six features from it. RASER-2 decides whether to stop or escalate to the extra-retrieval action PRUNE. RASER-3 chooses among one-shot RAG, PRUNE, and iterative retrieval IRCoT, using the same features but adding an explicit cost-accuracy trade-off. Neither router makes an extra LLM call to decide. Across six LLMs and three multi-hop QA benchmarks, both routers stay competitive with the other state-of-the-art (SOTA) baselines in F1 while spending only 41-49% of always-prune's tokens and also less than the iterative and decomposition retrieval baselines. 

---
# Iteris: Agentic Research Loops for Computational Mathematics 

**Authors**: Leheng Chen, Zihao Liu, Wanyi He, Bin Dong  

**Link**: [PDF](https://arxiv.org/pdf/2606.02484)  

**Abstract**: Recent advances in large language models and agentic AI systems have enabled significant progress in mathematical discovery, from solving competition problems to tackling research-level conjectures. However, open problems in computational mathematics have received comparatively less attention: research in this area often requires not only proofs but also numerical experimentation, adversarial constructions, and algorithm design. In this paper, we introduce an agentic research system, Iteris, designed for open problems in computational mathematics. We apply Iteris to two open problems from a recent Simons Workshop collection (arXiv:2602.05394). In these case studies, Iteris generated numerical evidence, constructions, and proof drafts that led, after expert review and correction, to verified results. The first result is a phase diagram for the asymptotic comparison between conjugate gradient and randomized coordinate descent on power-law spectra; the second is a counterexample showing that QR factorization with column pivoting can fail to select well-conditioned submatrices even under low coherence. These case studies suggest that agentic AI systems can participate meaningfully in research workflows for open problems in computational mathematics, while human validation remains essential. 

---
# MCP-Persona: Benchmarking LLM Agents on Real-World Personal Applications via Environment Simulation 

**Authors**: Wenhao Wang, Peizhi Niu, Gongyi Zou, Xiyuan Yang, Jingxing Wang, Haoting Shi, Yaxin Du, Jingyi Chai, Xianghe Pang, Shuo Tang, Yanfeng Wang, Siheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.02470)  

**Abstract**: The Model Context Protocol (MCP) has emerged as a transformative standard for connecting large language models (LLMs) with external data sources and tools, and has been rapidly adopted across personal applications and development platforms. However, existing benchmarks predominantly focus on generic information-seeking tools and fail to capture the practical challenges posed by personal social applications, where tools interact with individual accounts or local databases. To bridge this critical gap, we introduce MCP-Persona, the first benchmark specifically designed for evaluating agent performance on real-world, personalized MCP tools. MCP-Persona encompasses a diverse set of widely-used applications, ranging from social media platforms like Reddit and Xiaohongshu (Rednote) to enterprise collaboration suites such as Lark (Feishu) and Slack. Our extensive experiments on various state-of-the-art (SOTA) agents demonstrate their significant struggles with personalized tool use, thereby highlighting the benchmark's crucial role in identifying and addressing these limitations. MCP-Persona is publicly available at this https URL}{this https URL. 

---
# AGENTCL: Toward Rigorous Evaluation of Continual Learning in Language Agents 

**Authors**: Yiheng Shu, Bernal Jiménez Gutiérrez, Saisri Padmaja Jonnalagedda, Yuguang Yao, Huan Sun, Yu Su  

**Link**: [PDF](https://arxiv.org/pdf/2606.02461)  

**Abstract**: Language agents spend substantial inference time solving individual tasks, yet the experience acquired in one episode is often underutilized in future episodes. Continual learning expects an agent to accumulate reusable experience across a stream of tasks, improve over time, and avoid interference from irrelevant experiences. Unfortunately, existing benchmarks struggle to evaluate continual learning in language agents rigorously. Most efforts focus on retrieval and reasoning over long-context conversations or documents, while recent lifelong-adaptation benchmarks often rely on naive task streams with limited analysis of cross-task relationships, making it difficult to understand what an agent learns and reuses over time. This paper presents an evaluation framework AgentCL for continual learning in agents, centered on controlled task streams and metrics for transfer gains. AGENTCL constructs compositional streams where earlier sub-solutions, evidence, or workflows are intentionally reusable in later tasks, and contrasts them with naive streams where such reusability is not guaranteed. We use the benchmark to evaluate non-parametric memory designs for continual learning. To diagnose how memory design choices affect continual learning, we develop MemProbe, a probing method that stores interactions, insights, and skills, while filtering unreliable experiences during consolidation. Empirical analysis across coding, deep research, and language understanding/reasoning tasks shows that naive streams offer limited ability to distinguish memory designs, whereas controlled streams more clearly distinguish their plasticity. Meanwhile, naive and held-out settings often yield limited gains and can expose memory-induced degradation. These results highlight the need for stronger memory designs that balance plasticity and stable reuse. 

---
# Beyond One-shot: AI Agents for Learning in Field Experiments 

**Authors**: Junjie Luo, Ritu Agarwal, Gordon Gao  

**Link**: [PDF](https://arxiv.org/pdf/2606.02458)  

**Abstract**: Organizations routinely run experiments for A/B testing, yet the data generated from one experiment is underutilized to inform subsequent intervention design. Significant barriers exist to extracting actionable knowledge from prior experimental data to inform new interventions. We study whether tool-augmented agentic AI can automatically learn from experimental data to generate new interventions in subsequent experiments. Through two-stage field experiments in healthcare prescription messaging (693,139 patient visits), we compare a Human + Chatbot method (Stage 1: behavioral experts with conversational AI co-designing 13 message variants, 444,691 patient visits) against a Tool-Augmented Agentic AI method (Stage 2: AI autonomously extracting principles from Stage 1 data to generate 17 new variants, 248,448 patient visits). The Agentic AI method, equipped with analytical tools, structured Data-Information-Knowledge-Wisdom (DIKW) reasoning agents, and transparent evidence chains, produces superior interventions: the best AI-generated message achieved a 69.8% CTR (+6.5 percentage points over baseline). Critically, our results suggest that the value comes from domain-specific experimental data, not from general reasoning ability: frontier LLMs operating without experimental data failed to predict which interventions would succeed. The field experiments also revealed that general-purpose behavioral theories used for intervention design do not extend uniformly to specific healthcare contexts, motivating an agentic AI approach to theory audits at field-experiment scale. Our research shows that tool-augmented AI can learn from experimental data and generate improved domain-relevant interventions, transforming behavioral experimentation from one-shot evaluation into a scalable system for cumulative design learning. 

---
# HLL: Can Agents Cross Humanity's Last Line of Verification? 

**Authors**: Xinhao Song, Su Su, Sirui Song, Hongliang Wu, Wen Shen, Zhihua Wei, Gongshen Liu, Linfeng Zhang, Dongrui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.02449)  

**Abstract**: Multimodal agents are increasingly expected to operate interfaces on behalf of users, raising a central deployment question: can they truly substitute for humans in workflows that services deliberately protect against automation? CAPTCHA verification makes this question concrete. It is not merely a visual puzzle, but a human-verification boundary placed before account creation, content access, form submission, and other protected actions. We introduce \textbf{Humanity's Last Line of Verification (HLL)}, a controlled benchmark that uses interactive CAPTCHA verification to evaluate whether agents can cross this boundary through grounded, human-like interaction rather than recognition alone. HLL covers diverse CAPTCHA interactions and exposes agents to controlled realism stressors, including cluttered webpages, harder task variants, and trace-conditioned validation of the solving process. We evaluate eight frontier multimodal agents in a closed-loop GUI environment. The results show that current agents remain brittle at this human-substitution boundary: performance varies sharply across verification types, degrades under realistic interface conditions, and drops further when correct answers must be supported by valid action traces. By exposing gaps in localization, action calibration, state tracking, and process consistency, HLL provides a concrete testbed for measuring how close multimodal agents are to acting as human substitutes in protected real-world workflows. Our code is available at this https URL 

---
# Food Noise & False Safety: A Systematic Evaluation of How LLMs Fail to Adapt to Eating Disorder Queries with Clinician Feedback 

**Authors**: Giulia Pucci, Emily Hemendinger, Ruizhe Li, Gavin Abercrombie, Tanvi Dinkar, Arabella Sinclair  

**Link**: [PDF](https://arxiv.org/pdf/2606.02444)  

**Abstract**: Recent evidence shows that people with eating disorders (EDs) are increasingly seeking guidance, advice, and emotional support from Large Language Model (LLM)-based chat systems. Although these systems are not designed to provide clinical advice, their perceived expertise, neutrality and accessibility make them a frequent, albeit risky, source of support. This paper investigates potential patterns of interaction between users with EDs and LLMs, focusing on the potential harms arising from models that uncritically adapt to, and facilitate unsafe or self-harming user requests. We find, in consultation with clinical ED experts, that specific linguistic cues in prompts increase the likelihood of unsafe responses and, through systematically varying the degree of potential risk present in the user prompt, report the extent to which LLMs uncritically adapt to problematic, and potentially dangerous user inputs. 

---
# LLM-Evolved Pattern Generators for Optimal Classical Planning 

**Authors**: Windy Phung, Dominik Drexler, Arnaud Lequen, Jendrik Seipp  

**Link**: [PDF](https://arxiv.org/pdf/2606.02438)  

**Abstract**: Learned heuristics have recently become a competitive alternative to traditional domain-independent heuristics for satisficing planning. Existing approaches, however, focus on improving search guidance rather than guaranteeing admissibility, which makes them unsuitable for optimal classical planning. We present the first method for learning domain-dependent heuristics that are admissible by design and thus preserve the optimality guarantees of A* search. Instead of learning a direct mapping from states to heuristic values, we learn to construct abstractions that induce admissible heuristics. We use an LLM-driven evolutionary program-synthesis framework to obtain, for each domain, a program that produces a pattern collection for any task in that domain, and we combine the resulting patterns admissibly via saturated cost partitioning. Empirically, the learned programs encode interpretable domain-specific insights, run with negligible overhead at test time and yield heuristics that match the coverage of state-of-the-art domain-independent baselines on several domains while evaluating each state substantially faster. 

---
# Bridging the Sim-to-Real Gap in Semiconductor Visual Program Synthesis via Input Binarization 

**Authors**: Yusuke Ohtsubo, Kota Dohi, Koichiro Yawata, Koki Takeshita, Tatsuya Sasaki  

**Link**: [PDF](https://arxiv.org/pdf/2606.02434)  

**Abstract**: Precise parametric control over circuit geometry is essential for semiconductor inspection, yet obtaining sufficient real training data remains costly. Although generative models such as diffusion models and Generative Adversarial Networks (GANs) can augment training data, they cannot guarantee the nanometer-scale geometric accuracy required for metrology tasks. We propose a visual program synthesis framework in which a Vision-Language Model (VLM) converts inspection images into editable Domain-Specific Language (DSL) code describing circuit geometries, enabling controlled generation of training data with exact parameter manipulation. Because the VLM is trained solely on synthetic DSL-rendered data, a domain gap arises when processing real Scanning Electron Microscope (SEM) images. We bridge this gap with an input binarization strategy that strips SEM-specific texture and noise, letting the model focus on geometric structure. On the MIIC dataset, binarized inputs improve the mean Dice coefficient from 0.4393 to 0.5256 over the raw-input baseline, demonstrating that simple texture abstraction substantially mitigates the sim-to-real gap. 

---
# AgentPLM: Agentic Protein Language Models with Reasoning-Augmented Decoding for Protein Sequence Design 

**Authors**: Sahil Rahman, Maxx Richard Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2606.02386)  

**Abstract**: Protein language models (PLMs) are passive oracles: they generate sequences in a single forward pass with no mechanism to consult external biophysical feedback or redirect generation when a candidate violates thermodynamic or structural constraints. We introduce AgentPLM, which addresses this by equipping a pre-trained PLM with i) Reasoning-Augmented Decoding (RAD), which interleaves autoregressive generation with tool calls (ESMFold, FoldX, AutoDock Vina), and ii) Contrastive Agent Policy Optimisation (CAPO), a trajectory-level extension of direct preference optimisation that trains the policy end-to-end to learn when oracle feedback is informative rather than merely imitating high-fitness sequences. We evaluate AgentPLM on benchmark tasks spanning de novo enzyme design, antibody optimisation, thermostability, PPI interface design, and zero-shot fitness prediction with standardised oracle APIs and controlled sequence-identity splits. AgentPLM achieves state-of-the-art results with a gain in antibody top-10% hit rate over the strongest passive baseline, providing mechanistic evidence of online error correction without explicit backtracking. 

---
# A Mathematical Conflict Framework for Contextual Data Modulation 

**Authors**: Hakan Emre Kartal  

**Link**: [PDF](https://arxiv.org/pdf/2606.02381)  

**Abstract**: In this study, a generalized operator-based mathematical conflict framework is presented to explicitly represent structural discrepancies between raw data and contextual data. The proposed structure treats conflict as a local, directional, and context-sensitive quantity, integrating components such as weighting, scale behavior, and output mapping under a unified abstract operator. Without being reduced to a specific learning algorithm or optimization method, the framework is defined as a general structure adaptable to different classes of problems. While existing approaches typically treat conflict merely as an implicit side effect embedded within the optimization process, the proposed framework considers conflict as an independent, operator-based, and component-level mathematical object. 

---
# Spatial Representation Learning Beyond Pixels: Unifying Raster Data and Vector Semantics for Human-Centric Geospatial Foundation Models 

**Authors**: Steffen Knoblauch, Hao Li, Gengchen Mai, Konstantin Klemmer, Song Gao, WenWen Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.02374)  

**Abstract**: Earth Observation (EO) has fundamentally transformed the monitoring of environmental processes and human activities up to planetary scale. Recent advances in self-supervised learning have given rise to Earth Observation Foundation Models (EOFMs), which leverage petabyte-scale unlabeled EO data to learn transferable representations across a wide range of downstream geospatial tasks. Despite these advances, current EOFMs remain largely confined to raster modalities, overlooking the rich, structured information encoded in openly-accessible vector data sources such as OpenStreetMap and Overture. Vector data provides explicit and compact representations of geographic entities, including geometry, topology, and semantic relationships, offering critical contextual signals that are often ambiguous or inaccessible in imagery alone. Raster and vector data thus represent complementary views of geographic space: raster data captures continuous physical and spectral patterns, while vector data encodes discrete objects and their relational structure and often represents more of the human rather than the physical systems (e.g. social or demographic data). However, existing geospatial representation learning paradigms treat these modalities in isolation, relying on imperfect and often lossy transformations to bridge them. This perspective paper calls for a paradigm shift toward joint Spatial Representation Learning (SRL) in an unified embedding space that integrate raster perception with vector-based reasoning. Building on emerging efforts in multimodal geospatial learning, we highlight conceptual foundations, technical challenges, and promising directions for aligning heterogeneous spatial data sources. We contend that such integration is essential for developing next-generation geospatial AI systems capable of more accurate, interpretable, and semantically grounded understanding of the Earth. 

---
# Harness-1: Reinforcement Learning for Search Agents with State-Externalizing Harnesses 

**Authors**: Pengcheng Jiang, Zhiyi Shi, Kelly Hong, Xueqiang Xu, Jiashuo Sun, Jimeng Sun, Hammad Bashir, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2606.02373)  

**Abstract**: Search agents are often trained as policies over growing transcripts: the model must decide how to search while also remembering what it has seen, which evidence is useful, which constraints remain open, and which claims have actually been checked. We argue that this formulation puts too much routine state management inside the policy: reinforcement learning is forced to optimize both semantic search decisions and recoverable bookkeeping that the environment can maintain more reliably. We introduce Harness-1, a 20B search agent (retrieval subagent) trained with reinforcement learning inside a stateful search harness. The harness maintains environment-side working memory, including a candidate pool, an importance-tagged curated set, compact evidence links, verification records, compressed and deduplicated observations, and budget-aware context rendering. The policy retains the semantic decisions: what to search, which documents to keep or discard, what to verify, and when to stop. Across eight retrieval benchmarks spanning web, finance, patents, and multi-hop QA, Harness-1 achieves 0.730 average curated recall, outperforming the next strongest open search subagent by +11.4 points and remaining competitive with much larger frontier-model searchers. Its gains are especially strong on held-out transfer benchmarks, suggesting that reinforcement learning over explicit search state can produce retrieval behaviors that generalize beyond the training domains. Our code is available at this https URL. 

---
# COMAP: Co-Evolving World Models and Agent Policies for LLM Agents 

**Authors**: Youwei Liu, Jian Wang, Hanlin Wang, Wenjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.02372)  

**Abstract**: Equipping language agents with world models enables them to anticipate environment dynamics and evaluate candidate actions before execution. However, existing textual world models are typically fixed after training, preventing them from adapting to the on-policy state-action distributions induced by an evolving agent. Meanwhile, agent-improvement methods often rely on external rewards or verifiers, limiting their applicability in realistic interactive environments. In this paper, we propose COMAP, a novel framework that co-evolves textual world models and agent policies through closed-loop interaction. At each decision step, the world model predicts future state feedback for candidate actions, and the agent performs future-aware reflection by estimating the reliability of this feedback and refining its action accordingly. The resulting on-policy trajectories are then used to update the world model via self-distillation, allowing it to better match the agent's evolving interaction distribution. Across embodied task planning, Web navigation, and tool-use benchmarks, COMAP consistently outperforms competitive baselines, e.g., +16.75% relative improvement with Qwen3-4B. Further analyses show that the co-evolutionary loop improves the world model's prediction accuracy over time and leads to more effective long-horizon decision-making. Our code is available at: this https URL. 

---
# MOC: Multi-Order Communication in LLM-based Multi-Agent Systems 

**Authors**: Yao Guan, Lin Wang, Zhihu Lu, Ziyi Wang, Wenzhu Yan, Qiang Duan  

**Link**: [PDF](https://arxiv.org/pdf/2606.02359)  

**Abstract**: Despite the remarkable progress of Large Language Model (LLM) based Multi-Agent Systems, most research focuses on optimizing coordination topology while largely underexploring the equally critical problem: how to transmit and optimize messages among agents effectively? Current communication schemes typically rely on the direct concatenation of first-order neighbor responses, which induces a restricted evidence receptive field and leads to the dilution of crucial insights over multi-hop paths. To address these limitations, we propose the Multi-Order Communication (MOC) scheme, which reconstructs the inter-agent communication to capture multi-hop dependencies and incorporates a structural message consolidation strategy to ensure efficiency. Specifically, we formalize the communication mechanism to construct a structured multi-order evidence stream, and subsequently design a Semantic-Topological Merging algorithm to optimize semantic fidelity within token constraints. Extensive experiments across six diverse datasets and LLM backbones of varying parameter scales demonstrate that MOC consistently improves task performance and reduces communication costs. The code is available at this https URL. 

---
# SIRI: Self-Internalizing Reinforcement Learning with Intrinsic Skills for LLM Agent Training 

**Authors**: Zhongyu He, Yuanfan Li, Fei Huang, Tianyu Chen, Siyuan Chen, Xingyang Li, Meng Hsuan Yu, Xiangrong Liu, Leyi Wei, Lu Pan, Ke Zeng, Xunliang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2606.02355)  

**Abstract**: Long-horizon LLM agents can benefit from reusable skills, yet existing skill-based methods often rely on external skill generators during training or persistent skill retrieval at inference, increasing engineering complexity, context length, and deployment latency. We propose Self-Internalizing Reinforcement learning with Intrinsic skills (SIRI), a three-phase framework that enables agents to discover, validate, and internalize skills without external skill generators or inference-time skill banks. SIRI first warms up the policy with GiGPO to acquire basic interaction ability and collect successful skill-free trajectories. It then performs self-skill mining, where the current policy summarizes compact skills from its own successful plain rollouts and validates them through paired skill-augmented and skill-free rollouts. Finally, SIRI distills only beneficial skill-guided action tokens into the plain policy using trajectory-level utility and action-level advantage. At inference, the agent runs with the original prompt only. On ALFWorld and WebShop with Qwen2.5-7B-Instruct, SIRI improves GiGPO from 0.908 to 0.930 on ALFWorld and from 0.728 to 0.813 on WebShop, outperforming prompt-based, RL-based, and memory-augmented baselines. Further analysis shows that our self-mining strategy can achieve performance comparable to distillation with closed-source large model. Our code is available at this https URL. 

---
# Coordination Graphs for Constrained Multi-Agent Reinforcement Learning 

**Authors**: Santiago Amaya-Corredor, Miguel Calvo-Fullana, Anders Jonsson  

**Link**: [PDF](https://arxiv.org/pdf/2606.02337)  

**Abstract**: Constrained Multi-agent reinforcement learning (CMARL) faces two intertwined challenges: the joint action space grows exponentially with the number of agents, and additional requirements couple agents in ways that reward structure alone does not capture. We introduce Coordination Graphs for Constrained Multi-Agent Reinforcement Learning (CG-CMARL), a framework that addresses both challenges by combining coordination graphs with Lagrangian duality. The system decomposes the joint problem into pairwise regions, each served by a set of shared Q-functions, one for the primary objective and one for each of the constraints, so that the number of learned models is independent of the number of agents. At execution time, Max-Sum message passing coordinates actions across the factor graph, while a Lagrangian multiplier controls the objective--constraint tradeoff, allowing a single trained model to trace a Pareto front without retraining. We provide convergence guarantees under mild conditions, together with a compositional error bound that decomposes into separate interpretable sources, each traceable to a specific design choice and independently controllable. Experiments on cooperative navigation tasks (where teams of up to 10 agents must coordinate to reach target positions while satisfying pairwise constraints) show that our method produces Pareto fronts dominating established baselines trained at fixed reward-shaping ratios, while scaling to team sizes where centralized approaches become intractable. 

---
# Forget Attention: Importance-Aware Attention Is All You Need 

**Authors**: Soohyeong Shin, Yeongwook Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.02332)  

**Abstract**: Combining attention's global retrieval with the sequential importance signal of state space models (SSMs) is the open challenge of hybrid language modeling. Transformers see everywhere but cannot prioritize; SSMs know what matters but cannot revisit. Existing hybrids -- Jamba (block level) and Hymba (head level) -- place the two in separate compartments, so neither informs the other during the attention computation itself. We propose SISA (SSM-Informed Softmax Attention), which adds an SSM-derived importance term directly inside the attention score and realizes the full operation as a single SDPA call on augmented query/key vectors -- no recurrent state, no custom kernel. At 152M / 5B tokens, SISA reaches LAMBADA-greedy 17.3% (vs. Transformer 13.9 and Mamba-3 15.5) and attains NIAH 100% from step 1K, 7x faster than Transformer's retrieval convergence; at 369M, Mamba-3 leads LAMBADA while SISA preserves perfect NIAH and stock-SDPA execution. SISA thus defines a third design axis for SSM-attention hybrids -- score-level fusion -- beyond the block-level and head-level paradigms that have dominated the field. 

---
# Repair Before Veto: Repair-Augmented Constraint Learning for Contextual Decisions 

**Authors**: Yifan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.02326)  

**Abstract**: Hard constraints are usually treated as terminal vetoes: once a candidate violates a requirement, the learned rule rejects it and any repair is handled outside the decision semantics. This misses a common deployed regime in which the system already knows a finite menu of modifications, such as adding a ticket option, changing a configuration, or requesting an available service upgrade. Existing constraint-learning, soft-relaxation, and recourse methods address nearby problems, but they do not learn whether an option should be repaired before being vetoed. We introduce Repair-Augmented Constraint Learning (RACL), a contextual decision framework that lifts known repair operators into the classifier semantics. A candidate is accepted when an affordable repair makes it feasible and preferred enough; otherwise the system returns a structured rejection credit and, when applicable, a repair plan. This repair-before-veto view strictly generalizes no-repair HASSLE-style semantics, reveals an irreducible false-veto gap for terminal-veto rules, separates binary-label non-identifiability from decision-rule learnability, and gives capacity and calibration bounds for the observed-feasibility shared-weight setting. Across controlled and DB1B-derived benchmarks, RACL recovers the intended credit and repair structure. On the hardest raw-data-derived tier, validation-selected RACL reduces false vetoes to 10/4039 (FVR 0.0025), versus about 1064/4039 for the strongest repair-search black-box baseline, while making the FVR/EDR trade-off explicit. 

---
# POIROT: Interrogating Agents for Failure Detection in Multi-Agent Systems 

**Authors**: Iñaki Dellibarda Varela, R. Sendra-Arranz, Pablo Romero-Sorozabal, J.M. Valverde-García, Annemarie F. Laudanski, Álvaro Gutiérrez, Eduardo Rocon, Manuel Cebrian  

**Link**: [PDF](https://arxiv.org/pdf/2606.02282)  

**Abstract**: Orchestrating Large Language Models into Multi-Agent Systems (LLM-MAS) has unlocked remarkable reasoning capabilities, yet emergent failures and hallucinations that resist characterisation block their deployment in safety-critical domains -- a gap made legally untenable by emerging AI regulation. Existing evaluation paradigms share a common flaw: centralised judgment creates single points of failure and demands domain-specific expertise. Here we present POIROT, a protocol that repurposes a system's own agents as its diagnostic layer, leveraging the epistemic diversity already present in the architecture. Across evaluated settings, POIROT outperforms single-LLM evaluator baselines, with gains that scale with problem complexity (OR = 1.60, $p = 0.008$), agent count, and fault dimensionality, persisting under compound fault conditions. These results demonstrate that safety oversight need not be externalised: the agents executing a role carry sufficient collective intelligence to audit it. We release POIROT as an open-source library alongside BLAME, a benchmark for fault attribution in safety-critical multi-agent systems. 

---
# CEON: Circular Economy Ontology Network 

**Authors**: Huanyu Li, Els de Vleeschauwer, Robin Keskisärkkä, Mikael Lindecrantz, Mina Abd Nikooie Pour, Ying Li, Ben De Meester, Patrick Lambrix, Eva Blomqvist  

**Link**: [PDF](https://arxiv.org/pdf/2606.02253)  

**Abstract**: Increasing the circularity of resource use in our society has been recognized as a path to sustainability, i.e., transitioning into a more circular economy. There are many different circular strategies to do so, such as reusing products and components, refurbishing and remanufacturing used products, or recycling left-over or used materials. To enable these strategies, it is necessary to share information at the infrastructure level and to communicate between industry sectors along the product life cycle. Enabling semantic interoperability in this information sharing and communication is therefore a key to increasing circularity. However, knowledge representation for the circular economy (CE) domain, which involves many relevant industry sectors related to product life cycles, remains challenging. To bridge this gap, we developed the Circular Economy Ontology Network (CEON) within the Onto-DESIDE project. This ontology network aims to fill gaps in CE by defining cross-sectorial concepts and to enable semantics-aware data documentation. We demonstrate CEON through cross-industry data documentation scenarios spanning construction, electronics, and textile sectors. 

---
# From Capability Models to Automated Planning: An AAS-Native Approach for Automatic PDDL Generation 

**Authors**: Hamied Nabizada, Thomas Wirt, Luis Miguel Vieira da Silva, Felix Gehlhoff, Alexander Fay  

**Link**: [PDF](https://arxiv.org/pdf/2606.02167)  

**Abstract**: Engineers designing production systems need to verify that a given layout supports all required production sequences. Automated planning techniques can answer such questions, but formulating the required planning problems in the Planning Domain Definition Language (PDDL) demands specialized expertise that production engineers typically lack. Asset Administration Shells (AAS) have emerged as the standardized Digital Twin for industrial assets in Industry 4.0. We show that AAS capability models, structured using four established Industry 4.0 standards (VDI 3682 for process descriptions, IEC 61360-1 for semantic property qualification, IDTA 02011 for type hierarchies, and IDTA 02016 for instance descriptions), contain sufficient information to generate complete PDDL problems automatically. Unlike prior work that introduced PDDL-specific submodels, our approach derives all planning elements from domain-level descriptions of resource functions, so-called capabilities, allowing engineers to model capabilities without any exposure to PDDL syntax or planning concepts. Our extraction algorithm transforms distributed Multi-AAS architectures into complete PDDL planning problems. We validate the approach on AAS models of a laboratory production system, comparing four layout variants using optimal planning to demonstrate how engineers can systematically explore design trade-offs by modifying the AAS model and regenerating the planning domain 

---
# An Abstract Worlds Semantic Framework for Belief Change Operators 

**Authors**: Daniel Grimaldi, M. Vanina Martinez, Ricardo O. Rodriguez  

**Link**: [PDF](https://arxiv.org/pdf/2606.02163)  

**Abstract**: This article proposes a set-theoretic framework for belief change, called Abstract Worlds Semantics, in which no logical syntax is assumed. Inspired by Grove's (1988) results, our approach treats worlds as primitive elements, over which world contraction and world revision operators are defined. This semantic framework enables a unified analysis of belief change models. Within this framework, we unify classical and non-prioritized belief change constructions by defining versatile operators. When classical propositional logic is considered, our framework provides a homogeneous account of AGM, KM, and Multiple Change models. In summary, AWS systematizes belief change frameworks and operators, simplifying and generalizing belief change theory over belief sets. 

---
# S3TS: Stochastic Scenario-Structured Tree Search for Advanced Planning Under Uncertainty 

**Authors**: Fabio Pavirani, Bert Claessens, Pierre Pinson, Chris Develder  

**Link**: [PDF](https://arxiv.org/pdf/2606.02151)  

**Abstract**: Effective scheduling in the energy sector is essential to ensure the reliable operation of electrical grids and their connected assets by, for instance, optimizing the dispatch of generation units and storage systems. An effective planning strategy must (a) accommodate advanced and potentially non-linear system models -- exploiting the increasing data availability of modern grids, and (b) explicitly handle uncertainties arising, for instance, from the integration of renewable energy sources. While existing approaches can address either non-linearity (e.g., Monte Carlo Tree Search) or uncertainty (e.g., stochastic mathematical optimization), there is a lack of planning techniques capable of addressing both challenges simultaneously. To bridge this gap, we propose a Stochastic Scenario-Structured Tree Search (S3TS) algorithm that explicitly represents uncertainty through scenario trees while enabling the integration of advanced non-linear models. We evaluate S3TS on a simulated demand response signal publication problem, largely mimicking the imbalance settlement mechanism in Belgium. The results demonstrate near-optimal performance in linear, analytically tractable settings, with costs within 14% of the mathematically optimal solution conditioned to the scenario trees. In highly non-linear scenarios, S3TS significantly outperforms baseline methods, achieving cost reductions of up to 51% and 5.4% compared to a myopic algorithm and deterministic MCTS, respectively. 

---
# Learning When Not to Act: Mitigating Tool Abuse in Agentic Reinforcement Learning 

**Authors**: Liuji Chen, Dianxing Tang, Xing Shi, Dingshuo Chen, Qiang Liu, Shu Wu, Liang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.02132)  

**Abstract**: Agentic reinforcement learning can induce tool abuse, where models overuse external tools even for queries solvable by internal reasoning. Existing approaches mitigate this issue with uniform tool-use penalties or hard limits, which reduce tool frequency but may also suppress useful tool-assisted exploration. We propose EAPO, an Efficient Agentic Policy Optimization framework that learns selective tool use. EAPO introduces tool-free trajectories into each rollout group, applies difficulty-aware reward shaping to penalize redundant tool calls mainly on easier queries, and uses confidence-aware token reweighting to improve policy learning. Across nine mathematical and knowledge-intensive reasoning benchmarks, EAPO consistently improves the accuracy efficiency trade-off on Qwen2.5-3B, Qwen2.5-7B, and Llama3.1-8B. Compared with GRPO, EAPO improves average performance by 10.45%, 7.27%, and 9.69%, while reducing average tool calls by 18.33%, 18.33%, and 24.59%, respectively. These results show that agents can learn when not to use tools without compromising tool-integrated reasoning. 

---
# BADGER: Bridging Agentic and Deterministic Evaluation for Generative Enterprise Reasoning 

**Authors**: Shannon Serrao, Soumitra Chatterjee, Dorina Strori, Abhishek Sharma, Nathan Miller  

**Link**: [PDF](https://arxiv.org/pdf/2606.02109)  

**Abstract**: Enterprise AI systems that translate natural language into SQL queries and orchestrate multi-step agentic reasoning pipelines require evaluation approaches fundamentally different from academic benchmarks. Spider and BIRD established execution-accuracy protocols; G-Eval and RAGAS advanced LLM-based assessment; and recent work such as Spider 2.0, BEAVER, and BIRD-Interact has begun to address enterprise and agentic dimensions. No single framework unifies text-to-SQL assessment with agentic behavior evaluation into a production-grade pipeline calibrated against human expert judgment.
We present BADGER, developed at Merkle, a unified evaluation framework integrating text-to-SQL assessment with agentic behavior evaluation. BADGER offers three contributions. First, LLM-assisted SQL component extraction extending Spider methodology to handle CTE-heavy, dialect-specific SQL. Second, a hybrid execution accuracy metric (Hybrid-EX) resolving column-aliasing and numeric-tolerance brittleness by using an LLM to infer structural alignments before deterministic cell-level scoring. Validated on 150 human-annotated industry queries, Hybrid-EX achieves Cohen's kappa=0.717 [95% CI: 0.600-0.822] (Substantial agreement) and 87.3% balanced accuracy, outperforming all six competing frameworks (Delta-kappa: 0.322-0.502, all p<=0.001). Third, an enterprise agentic evaluation suite assembling RAGAS, G-Eval, and agent benchmark metrics into a unified pipeline; Excess Tool Usage is the sole novel element.
BADGER runs entirely within the client's governed data environment, supports configurable LLM judge backends, and enables rapid prototyping of client-specific judges and metrics, serving as a continuous evaluation backbone rather than a one-time quality gate. 

---
# Where Do Deep-Research Agents Go Wrong? Span-Level Error Localization in Agent Trajectories 

**Authors**: Jiaming Wang, Ziteng Feng, Jiangtao Wu, Ruihao Li, Qianqian Xie, Yuxiang Ren, He Zhu, Xueming Han, Fanyu Meng, Junlan Feng, Jiaheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.02060)  

**Abstract**: Deep-research agents solve tasks through long trajectories of search, tool use, evidence inspection, and answer synthesis. Evaluation based on final answers shows whether an agent succeeds, but not which parts of the trajectory make the answer unreliable. We study span-level error localization for deep-research agents. We collect 2,790 real trajectories from two agent frameworks, three backbone models, and three benchmarks, convert raw logs into semantic spans, and annotate harmful error spans through LLM-assisted expert review. From these annotations, we build TELBench, a 1,000-instance benchmark for identifying error spans among normal exploration, failed searches, tentative hypotheses, and harmless noise. We further propose DRIFT, a claim-centric auditing framework that tracks agent claims, checks their support in trajectory evidence, and marks spans where unsupported or conflicting claims affect the answer path. Experiments across model families and auditing frameworks show that DRIFT improves span-level error localization and first-error accuracy by up to 30 percentage points. Our work provides a process-level view of reliability in deep-research agents. 

---
# eMoT: evolving Memory-of-Thought via Symbolic Anchoring and Memory Corrosion 

**Authors**: Xiang Li, Jiwei Wei, Ke Liu, Yitong Qin, Jinyu Guo, Malu Zhang, Peng Wang, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.02054)  

**Abstract**: While Large Language Models (LLMs) achieve impressive performance on multi-step reasoning tasks, their reliability is persistently hindered by critical limitations such as unconstrained hallucinations and poor numerical computation. Fundamentally, these issues arise because standard models treat reasoning as a transient, one-off generation process rather than retaining and refining successful procedural logic. To address these challenges, we propose eMoT (evolving Memory-of-Thought), a unified framework that stabilizes multi-step reasoning by treating reasoning trajectories as dynamic, evolving memories rather than static templates. The framework primarily consists of three interconnected modules: (i) a memory corrosion mechanism that reinforces high-utility reasoning structures while gradually decaying less frequent ones; (ii) a symbolic anchoring engine that utilizes Python for deterministic computation, much like a human uses a calculator; and (iii) a consistency-driven refinement process that aligns neural inference with symbolic outcomes, reducing the accumulation of logical discrepancies. Across multiple reasoning benchmarks, eMoT improves accuracy and solution consistency over standard Chain-of-Thought and structured reasoning this http URL the traditional task Game of 24, eMoT achieves 100% accuracy, surpassing the baseline by up to 17.6%. Evaluations on mathematical task GSM8K, ASDiv, SVAMP, and MGSM further show consistent gains in multi-step mathematical reasoning. In our evaluation, we achieve superior performance despite utilizing a lightweight backbone model with constrained baseline capabilities. Compared to alternative methods that rely on massively scaled models, our results demonstrate that the performance gains are fundamentally driven by the eMoT framework's reasoning control rather than sheer model size. 

---
# Explainable Data-driven Deep Reinforcement Learning Methods for Optimal Energy Management in Buildings 

**Authors**: Hallah Shahid Butt, Qiong Huang, Gökhan Demirel, Kevin Förderer, Erfan Tajalli-Ardekani, Simnon Waczowicz, Luigi Spatafora, Veit Hagenmeyer, Benjamin Schäfer  

**Link**: [PDF](https://arxiv.org/pdf/2606.02049)  

**Abstract**: The increasing integration of renewable energy sources into power systems, particularly in buildings equipped with photovoltaic (PV) panels and energy storage systems, introduces significant complexity in energy systems. Volatile power generation, varying electricity tariffs, and increased entities, e.g., PV systems, and heat pumps, have increased the complexity and made the system harder to operate. This leads to the demand for additional control and optimization routes including data-based controls, such as reinforcement learning. While deep reinforcement learning (DRL) has emerged as a promising solution to optimize building operations in dynamic and ever more complex environments, its black-box nature impedes user trust and practical adoption. This paper presents a framework for explainable deep reinforcement learning (XRL) applied to energy management in residential buildings. We demonstrate its usage on both synthetic data but also on real-world data from the Living Lab Energy Campus (LLEC) at KIT. We train and compare both on-policy and off-policy DRL agents on an expanded state space that incorporates real-time measurements (demand, PV generation, battery power, state of charge), external signals (dynamic electricity price, local weather data), calendrical and holiday indicators, and forecasts for demand and price. Our experimental results indicate that on-policy algorithms, particularly Advantage Actor Critic (A2C) and Proximal Policy Optimization (PPO), outperform off-policy methods in terms of cumulative rewards and policy stability. To explain these models, we employ post-hoc interpretation techniques to elaborate the learned control policies. Our findings demonstrate that the XRL framework not only reduces electricity costs through optimal battery management, but also provides transparent, actionable insights into the agent's decision-making process. 

---
# Topological texture analysis of microscopy images of dynamic casein gelation and its relation to rheological properties 

**Authors**: Zahra Tabatabaei, Diana Soto Aguilar, Jose C. Bonilla, Mathias P. Clausen, Jon Sporring  

**Link**: [PDF](https://arxiv.org/pdf/2606.02048)  

**Abstract**: We propose a novel computational toolbox that integrates Topological Data Analysis (TDA), Differential Box Counting (DBC), Multifractal Partition (MFP), and Local Binary Patterns (LBP), applied to time-lapse super-resolution STED microscopy images of sodium caseinate gelation induced by glucono-delta-lactone (GDL) at 30 °C and 40 °C and two GDL concentrations (1.8% and 3.5% w/v).
TDA tracked topological loops, closed ring-like structures reflecting protein network interconnectivity, via max-Betti-1 curves, which revealed a lag phase of dispersed aggregates, a sharp decay coinciding with network percolation and the rheologically observed sol-gel transition, and a post-gelation increase corresponding to network rearrangements. These topological transitions were corroborated by DBC and MFP as these methods were able to resolve changes in structural complexity and spatial heterogeneity. The toolbox was validated on simulated fractal images prior to experimental application. Together, these descriptors provided sensitivity to subtle microstructural transitions that bulk rheology captured as averaged bulk mechanical responses. This integrated approach provides a robust quantitative tool for characterizing complex microstructure in food and material science with evolving microstructural dynamics. Code is available at this https URL 

---
# RL-ACRGNet: Reinforcement Learning-Based Chest Radiology Report Generation Network 

**Authors**: Yogesh Kumar Meena, Saurabh Agarwal, K.V. Arya  

**Link**: [PDF](https://arxiv.org/pdf/2606.02035)  

**Abstract**: Medical imaging interpretation is a foundational pillar of modern clinical diagnostics, yet the manual generation of radiology reports remains a time-consuming process prone to interpretation inconsistencies. Within the field of medical AI, automating these descriptions through deep learning promises to streamline clinical workflows and standardise diagnostic output. However, accurate disease detection and precise report generation remain significant challenges due to limitations in capturing fine-grained visual features and ensuring clinical coherence. To address these issues, we propose RL-ACRGNet, an improved encoder-decoder model that integrates a pre-trained DenseNet encoder with a multilevel LSTM decoder within an off-policy reinforcement learning framework. Using a dual-network approach to refine visual-semantic embeddings through a metric-based reward mechanism, we demonstrate that RL-ACRGNet consistently outperforms state-of-the-art baselines on the IU-Xray dataset, achieving quantitative improvements in BLEU-4 (0.47%), METEOR (0.17%) and ROUGE-L (0.518). Furthermore, comprehensive evaluations on the large-scale MIMIC-CXR data set confirm the robust generalisation of the model and its ability to generate high-quality, clinically relevant reports 

---
# Extreme Low-Bit Inference in Reasoning Models: Failure Modes and Targeted Recovery 

**Authors**: Ekaterina Alimaskina, Darya Rudas, Denis Shveykin, Gleb Molodtsov, Pavel Vasiliev, Aleksandr Beznosikov  

**Link**: [PDF](https://arxiv.org/pdf/2606.02011)  

**Abstract**: Large Reasoning Models (LRMs) rely on long reasoning traces, making inference expensive. While low-bit quantization reduces per-token decoding cost, we show that aggressive 2-bit inference can fail to deliver end-to-end speedup because instability in the generation process inflates total token count. Instead of merely lowering answer accuracy, 2-bit quantization often produces much longer traces with repetitive loops, budget exhaustion, delayed commitment, and unclosed reasoning segments. We analyze full reasoning traces of Qwen3 reasoning models across mathematical and commonsense benchmarks and show that accuracy degradation is tightly linked to these process-level failures. To address them, we introduce two lightweight controls: FP16 planning, which gives the 2-bit model a short high-precision outline, and loop rescue, which detects repetitive traces and either commits to an earlier answer or falls back to FP16. On MATH-500, loop rescue improves Qwen3-8B accuracy from 17.2% to 74.2%, while planning plus loop rescue improves Qwen3-32B from 65.0% to 87.2%. Overall, our results show that extreme low-bit reasoning becomes practical when its failures are treated as controllable generation pathologies: with lightweight detection and selective FP16 support, 2-bit inference can recover accuracy while preserving real end-to-end speed. Our code is available at: this https URL. 

---
# SafeMCP: Proactive Power Regulation for LLM Agent Defense via Environment-Grounded Look-Ahead Reasoning 

**Authors**: Lichao Wang, Zhaoxing Ren, Tianzhuo Yang, Jiaming Ji, Chi Harold Liu, Yaodong Yang, Juntao Dai  

**Link**: [PDF](https://arxiv.org/pdf/2606.01991)  

**Abstract**: As Large Language Model (LLM) agents increasingly leverage the Model Context Protocol (MCP) to operate in complex environments, the expansion of their action spaces offers agents unsafe capabilities and underscores the risk of power-seeking. While broad action space and greater environment influence are essential for task fulfillment, they create a fragile risk surface where minor errors or hallucinations are magnified into catastrophic failures. In response, we propose SafeMCP, a {server-side} defense plugin that constrains tool acquisition via predictive reasoning regarding future safety risks. SafeMCP utilizes an internal world model for look-ahead reasoning to implement a two-tier defense: proactive tool filtering to constrain hazardous power expansion and immediate intervention as a fail-safe. To train SafeMCP, we introduce a three-stage pipeline comprising environmental dynamic grounding, safe policy initialization, and reinforcement learning (RL) with dual verifiable rewards. Experiments on PowerSeeking Bench, ToolEmu, and AgentHarm show that SafeMCP achieves a safe equilibrium, effectively mitigating risks while preserving agent utility. 

---
# An NLP-Driven Framework for Curriculum-Labor Market Alignment: Schema-Constrained LLM Extraction, ESCO-Anchored Semantic Matching, and Multi-Dimensional Gap Quantification 

**Authors**: Sherzod Turaev, Mary John, Mamoun Awad, Nazar Zaki, Khaled Shuaib  

**Link**: [PDF](https://arxiv.org/pdf/2606.01982)  

**Abstract**: Schema-constrained information extraction from diverse educational and labor-market corpora remains an open challenge in natural language processing because existing pipelines rely primarily on lexical-surface methods that cannot recover implicit competencies, lack grounding in shared taxonomies, and provide no formal measures of extraction reliability or document-level completeness. To address these limitations, this paper proposes a four-stage NLP framework that combines (i) schema-constrained prompting of a two-model frontier-LLM ensemble against a JSON Schema-enforced seven-slot competency formalism, (ii) Sentence-BERT (SBERT) alignment of the extracted records against an eleven-domain ESCO v1.2.1 controlled vocabulary, (iii) a two-tier adjudication protocol that resolves inter-model disagreements, and (iv) a verification mechanism that combines per-slot Cohen's kappa, schema conformance, and document-level completeness audits. The framework is instantiated for a critical application in higher-education quality assurance, namely curriculum-labor market alignment for the ABET-accredited BSc Computer Science program at the United Arab Emirates University. The pipeline extracts 400 competency records from the 85-course 2025-2026 study plan and aligns them, under a five-scope analysis ranging from the computing core to a probability-weighted student trajectory, with 30 job postings (483 requirement clauses) at an SBERT cosine threshold of 0.50. The extractor achieves Cohen's kappa of 0.79 on the skill slot, with 100% schema conformance and 100% document-level completeness. The alignment surfaces interpretable supply-demand gaps of 25.0% in general and transversal skills, 13.8% in algorithms and computational theory, and 12.2% in software engineering and project management, with a near-zero 1.8% gap in artificial intelligence and data science despite 38.6% supply coverage. 

---
# Algorithmic algorithm development with LLMs: A Case Study on LLM-Usage for Contraction Order Optimization in Tensor Networks 

**Authors**: Fabian Hoppe, Melven Röhrig-Zöllner, Philipp Knechtges  

**Link**: [PDF](https://arxiv.org/pdf/2606.01975)  

**Abstract**: We consider LLM-based algorithm development through a case study on contractionorder optimisation for tensor networks with OpenEvolve. We pay particular attention to the choice of the LLM as well as design choices such as evaluation metric and test instances. Our results highlight both the promise of verifier-guided evolutionary coding agents for algorithm development/improvement and the continuing importance of evaluation, validation, and interpretation -- and corresponding challenges -- by the human scientist. 

---
# AutoMedBench: Towards Medical AutoResearch with Agentic AI Models 

**Authors**: Junqi Liu, Salena Song, Yuhan Wang, Jiawei Mao, Hardy Chen, Xiaoke Huang, Tianhao Qi, Pengfei Guo, Yucheng Tang, Yufan He, Can Zhao, Andriy Myronenko, Dong Yang, Daguang Xu, Yuyin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.01961)  

**Abstract**: Autonomous agents are increasingly expected to support end-to-end medical-AI research workflows, moving beyond isolated prediction tasks or short-form clinical question answering. However, existing medical agent benchmarks primarily evaluate final outputs, providing limited visibility into agent behavior within the research process. To address this gap, we present AutoMedBench, a workflow-aware benchmark for autonomous medical-AI research across diverse medical imaging and multimodal inference tasks, organizing agent execution into a unified five-stage workflow (S1-S5): Plan, Setup, Validate, Inference, and Submit. It comprises long-horizon tasks with each run averaging 33 agent turns, spanning five research tracks: segmentation, image enhancement, visual question answering (VQA), report generation, and lesion detection. Each task is evaluated under two difficulty tiers, Lite and Standard, which use the same data and metrics but differ in the amount of task-brief scaffolding, and each run is scored using both final task performance and S1-S5 stage scores, enabling stage-level analysis from the initial task brief to the final submitted artifact. Across thousands of recorded runs, stage-level scoring reveals that Validate is the weakest workflow stage on average, whereas Setup is the strongest, suggesting that current agents are better at making pipelines executable than at verifying their reliability. Post-run error analysis further shows that verification and submission failures dominate tagged errors, accounting for 37.7% and 38.1% of fired codes respectively, whereas task-understanding errors are rare at 0.9%, and runs with one fired error code have a 48% lower overall score than runs with no error code on average. 

---
# VET: A Framework for Analyzing AI Discourse 

**Authors**: Meredith Ringel Morris  

**Link**: [PDF](https://arxiv.org/pdf/2606.01929)  

**Abstract**: Public discourse on AI has become polarized; exaggerated positions on AI in traditional and social media threaten the development of AI Literacy among the general public. In this article, I introduce the VET Framework, a method for categorizing AI discourse along the dimensions of valence, effectiveness, and trajectory. I show how this framework can be used to identify, compare, and critique prevalent narratives of AI Hype, AI Doom, AI Denial, and AI Normalcy. Using VET, I analyze how each of these four stances exaggerates some aspects of the current state and/or likely evolution of AI, and illustrate how the VET framework can serve as an AI Literacy tool by supporting the ``vetting'' of polarized AI discourse. 

---
# SMH-Bench: Benchmarking LLM Agents for Environment-Grounded Reasoning and Action in Smart Homes 

**Authors**: Kuan Li, Shuo Zhang, Huacan Wang, Fangzhou Yu, Zecheng Sheng, Yi Gu, Weipeng Ming, Lei Xue, Chen Liu, Sen Hu, Ronghao Chen, Siyue Lin, Yuqing Hou, Xiaofeng Mou, Yi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.01912)  

**Abstract**: Smart homes are evolving toward complex state-dependent living environments, requiring Large Language Models (LLMs) to reason over user intent, preferences, and multi-device interactions. However, existing smart-home benchmarks often focus on static instruction-to-API mapping or limited simulations, failing to evaluate whether LLMs can reason, interact, and act reliably in realistic household scenarios. To address these limitations, we introduce SMH-Bench, a comprehensive benchmark for evaluating LLMs in smart-home environments. Built upon HomeEnv, an executable and verifiable smart-home simulator, SMH-Bench contains 1,100 high-quality tasks spanning 7 categories and 22 fine-grained subcategories. It further stratifies tasks across simple, medium and complex homes, ranging from small apartments to dense multi-room environments with 135 devices. Experiments show that although frontier LLMs achieve strong performance on explicit control and query tasks, they still exhibit significant weaknesses in automation task scheduling, ambiguity handling and personalized reasoning, especially as home complexity increases. We hope SMH-Bench will facilitate the development of more reliable, context-aware, and practically deployable smart-home agents. 

---
# Bayesian Spectral Emotion Transition Discovery from Multi-Annotator Disagreement 

**Authors**: Keito Inoshita, Takato Ueno  

**Link**: [PDF](https://arxiv.org/pdf/2606.01906)  

**Abstract**: Emotions evolve through the dynamics of conversation, and understanding their transition structure is foundational to applications ranging from mental-health screening to dialogue systems. However, existing studies typically compress multi-rater judgments into a single hard label by majority voting, discarding the uncertainty signal needed to understand turn-to-turn transitions. In this article, we propose Bayesian Spectral Emotion Transition Discovery (BSETD), a two-stage framework that discovers emotion-transition structure from multi-rater soft labels. In the first stage, a hierarchical Dirichlet-Multinomial posterior is constructed through the outer product of soft labels, equipping each cell of the K x K transition matrix with a credible interval and Benjamini-Hochberg (BH) false discovery rate (FDR)-controlled significance. In the second stage, the symmetrized graph Laplacian is spectrally decomposed to separate a low-frequency (inertia) component from a high-frequency (contagion) component. On EmotionLines, BSETD simultaneously recovers the signatures of two distinct affective spaces: the Plutchik-adjacent transitions disgust to anger (log2 lift +0.94) and anger to disgust (+0.86) are over-represented, while the Russell-valence-reversed transitions joy to anger (-0.90) and anger to joy (-0.89) are under-represented. A five-source cross-corpus validation yields pairwise Pearson correlations in 0.91-0.98 within English, 0.79-0.85 against Chinese M3ED, and 0.979 between the human hard labels and the LLM virtual soft labels on the same utterance set, demonstrating that a pipeline preserving annotator uncertainty bridges the computational study of emotion dynamics with established psychological theory. 

---
# Community-Aware Assessment of Social Textual Engagement and Resonance: A Human-Centric Perspective on User-Generated Content Evaluation 

**Authors**: Tianjiao Li, Kai Zhao, Xiang Li, Yang Liu, Huyang Sun  

**Link**: [PDF](https://arxiv.org/pdf/2606.01897)  

**Abstract**: Traditional Video Quality Assessment (VQA) focuses narrowly on aesthetic fidelity, overlooking the complex social dynamics that define quality in User-Generated Content (UGC). In this work, we propose a paradigm shift from signal-centric metrics to human-centric resonance assessment. We introduce CASTER (Community-Aware Assessment of Social Textual Engagement and Resonance), a new task that evaluates whether a UGC item achieves positive community resonance based on its multimodal attributes rather than visual quality alone. To address this, we present MEDEA (Multimodal Engagement-Driven Evaluation Architecture), which introduces a novel Social Chain-of-Thought (Social-CoT) mechanism. Unlike traditional logical CoT, Social-CoT performs multimodal perspective-taking, instantiating diverse viewer personas to simulate collective cognitive and emotional reactions (i.e., the "community mind") before deriving a quality judgment. MEDEA is trained via a two-stage approach involving supervised fine-tuning and process-supervised reinforcement learning with Social Alignment Reward to ensure reasoning paths are grounded in authentic human social cognition. To support this task, we release CASTER-Bench, a comprehensive human-annotated benchmark covering diverse UGC categories. Experiments demonstrate that MEDEA significantly outperforms state-of-the-art baselines on CASTER-Bench while providing interpretable and empathetic reasoning paths that align with real community feedback. 

---
# Physically-Constrained Mamba-SDE for Remaining Useful Life Prediction under Irregular Observations 

**Authors**: Deyu Zhuang, Peiliang Gong, Yang Shao, Liyuan Shu, Qi Zhu, Xiaoli Li, Daoqiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.01894)  

**Abstract**: Accurate Remaining Useful Life prediction is critical for industrial predictive maintenance. However, real-world deployment is challenging due to the irregular nature of sensor observations, characterized by asynchronous sampling, burst missingness, and temporal jitter. Compounding this issue, purely data-driven models often generate physically implausible degradation trajectories that violate the irreversible nature of damage accumulation. To address this, we propose PC-MambaSDE, a unified continuous-time framework for robust RUL prediction under irregular observations. Specifically, we design a Mask-Aware Continuous Mamba Encoder that explicitly leverages observation masks to extract context-rich control signals. Furthermore, we introduce a Physics-Guided Latent SDE with parametrically rectified hybrid drift, superimposing a global physical bias to enforce monotonic degradation even amid severe observation gaps. Additionally, we formulate RUL prediction as a boundary value problem via a Terminal Degradation Penalty, which decouples a Health Index dimension and applies a penalty loss to guide trajectories toward the failure state. Theoretically, we prove that our variational objective is mathematically equivalent to minimizing the KL divergence via Girsanov's theorem, and we guarantee the global asymptotic stability of the learned dynamics through Lyapunov analysis. To enable rigorous evaluation, we develop a Hybrid Irregularity Generation Scheme that simulates realistic industrial imperfections. Extensive experiments on public benchmarks demonstrate that PC-MambaSDE significantly outperforms state-of-the-art methods, particularly under extreme observation scarcity, validating the efficacy of embedding physical priors into continuous-time latent dynamics. 

---
# Absorbing Complexity: An Interaction-Native Knowledge Harness for Financial LLM Agents 

**Authors**: Ailiya Borjigin, Igor Stadnyk, Ben Bilski, Maksym Chikita, Dmytro Kyrylenko, Sofiia Pidturkina, Julia Stadnyk  

**Link**: [PDF](https://arxiv.org/pdf/2606.01886)  

**Abstract**: Financial AI agents often fail for a simple reason: they make users carry the complexity. A user must repeatedly restate goals, risk preferences, portfolio context, past judgments, and shifting market assumptions, while the agent answers, retrieves, acts, and forgets. In finance, this is not just inconvenient. In tasks such as market analysis, copy-trading review, and trade preparation, forgotten context and stale memory can create latency, repeated errors, weak auditability, and unsafe decisions.
We propose the interaction-native knowledge harness (InKH), an architecture for financial LLM agents that absorbs complexity into the system. InKH converts user, market, portfolio, and tool events into structured operational knowledge. It uses passive knowledge injection to assemble a bounded working context buffer before the main model step, temporal graph memory for low-latency retrieval, a wiki audit surface for human-readable governance, and background extraction with maturity, decay, and write-time invalidation.
We evaluate InKH on a reproducible controlled synthetic benchmark with 24 random seeds, 4 rounds, 80 episodes per round, and 6 baselines, producing 46,080 baseline-conditioned evaluations. InKH achieves mean task quality of 0.815 at 900 ms latency. Compared with agent-driven wiki-walk memory, it reduces latency by 82.95 percent, token cost by 82.29 percent, and stale-knowledge usage by 96.58 percent, while improving quality by 0.108 and traceability by 0.461. Compared with a temporal-graph system without invalidation, it improves quality by 0.050 and reduces stale-memory usage by 96.58 percent with comparable serving cost.
The results support a design thesis for financial AI: adoption happens when complexity is absorbed by the system rather than transferred to the user. The benchmark validates architecture-level behavior, not live trading performance. 

---
# EVA-Net: Subject-Independent EEG Motor Decoding with Video-Derived Motor Priors 

**Authors**: Ziyuan Li, Yueyu Sun, Yimeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.01884)  

**Abstract**: Practical non-invasive Brain-Computer Interface (BCI) systems require EEG decoders with strong cross-subject generalization and minimal calibration. However, inter-subject variability and signal non-stationarity often entangle motor semantics with subject-specific noise, limiting subject-independent decoding. Recent multimodal approaches use text as a semantic anchor, yet text provides sparse and static supervision for inherently dynamic motor processes. To address this issue, we propose EVA-Net, a two-stage framework that uses action videos as semantic priors for subject-independent EEG motor decoding. In the first stage, EEG and video features are aligned in a shared space using cross-modal and supervised contrastive objectives to reduce subject-specific variation. In the second stage, video category prototypes and knowledge distillation transfer video-derived priors to an EEG-only classifier without adding inference overhead. Experiments on two public datasets show that EVA-Net achieves strong subject-independent decoding performance, including an 8.66% LOSO accuracy gain on EEGMMI. Ablation results further suggest that video provides a more effective semantic anchor than the text baseline considered in this work. 

---
# WorldCoder-Bench: Benchmarking Physically Grounded 3D World Synthesis 

**Authors**: Shuo Lu, Yinuo Xu, Kecheng Yu, Siru Jiang, Yongcan Yu, Yubin Wang, Haitao Yang, Yuxiang Zhang, Bin Wang, Ran He, Jian Liang  

**Link**: [PDF](https://arxiv.org/pdf/2606.01869)  

**Abstract**: Large language models (LLMs) are increasingly asked not only to write static interfaces, but to construct executable interactive worlds from natural language. Browser-native 3D, commonly built with this http URL, is a natural next frontier: generated programs must integrate assets, obey spatial and physical constraints, and keep user-facing controls synchronized with hidden runtime state. Existing web-generation benchmarks and evaluators, however, largely observe only pixels or DOM nodes, while the mechanics of a this http URL world unfold inside an opaque <canvas>. We introduce WorldCoder-Bench, a benchmark for autonomous, physically grounded 3D world synthesis. WorldCoder-Bench contains 2,026 expert-curated tasks across Simulation, Rendering, and Application scenarios, with optional .glb assets and hidden behavioral contracts. We further propose StateProbe, an execution-based protocol that probes generated programs in a sandboxed browser and verifies hidden, mutation-hardened contracts over runtime states and transitions. Beyond verification coverage, we report Return on Automation and Time Efficiency Multiplier to measure correctness-adjusted cost and time savings. Across nine frontier models, the best system reaches only 27.8% verification coverage on WorldCoder-Core and 19.9% on WorldCoder-Robust, with failures dominated by state-schema drift and broken interaction chains rather than missing scene elements. Utility metrics further show that cheap or fast models can still provide substantial value on easier domains. WorldCoder-Bench is available at this https URL. 

---
# Does Compression Preserve Uncertainty? A Unified Benchmark for Quantized and Sparse LLMs via Conformal Prediction 

**Authors**: Yujia Tong, Yuxi Wang, Yunyang Wan, Tian Zhang, Junhao Dong, Jingling Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2606.01850)  

**Abstract**: Model compression techniques such as quantization and pruning are widely used to reduce the deployment cost of large language models (LLMs), with existing evaluations focusing almost exclusively on accuracy preservation. However, in safety-critical applications, a model's ability to reliably quantify its own uncertainty is equally important. We ask: does compression preserve this ability? To answer this question, we benchmark 12 LLMs under various compression configurations across five NLP tasks, using conformal prediction to provide a rigorous, distribution-free measure of uncertainty. Our experiments reveal that: (I) compression frequently decouples accuracy from uncertainty; (II) larger models absorb compression-induced uncertainty far more effectively than smaller ones; and (III) uncertainty inflation is often threshold-like rather than gradual. These results suggest that accuracy-only evaluation is insufficient for assessing the deployment readiness of compressed LLMs, and that uncertainty-aware benchmarking should be a standard component of model compression pipelines. 

---
# Evaluation of Baseline Methods for IDD-based SSD External Memory Search 

**Authors**: Yuki Suzuki, Alex Fukunaga  

**Link**: [PDF](https://arxiv.org/pdf/2606.01840)  

**Abstract**: Many difficult search problems cannot be solved by algorithms such as A* using only RAM. Search algorithms which use external memory such as SSDs and HDDs with much higher capacity than RAM have been proposed in previous work, but previous work has focused on delayed duplicate detection approaches, as well as complex immediate duplicate detection (IDD) methods, and relatively simple methods for IDD have not been systematically studied. In addition, the effect of OS-level mechanisms for managing and speeding up accesses to external memory, such as page caches, has not been studied. This paper addresses these gaps in the literature by evaluating and analyzing the performance of simple baseline approaches for IDD-based A*. 

---
# CAPF: Guiding Search-Agent Rollouts with Credit-Attenuated Privileged Feedback 

**Authors**: Bin Chen, Xinye Liao, Yiming Liu, Xin Liao, Chonghan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.01830)  

**Abstract**: Recent LLM search agents use reinforcement learning with verifiable rewards (RLVR) to learn search-augmented reasoning from outcome rewards. On hard problems, these agents rarely sample end-to-end successful rollouts, leaving outcome-only RLVR with few positive-reward trajectories. We argue that improving learning on such problems requires additional guidance during training, and RLVR already contains verifier-side information that can provide it. This information can identify errors or omissions in the agent's submitted answer and guide revision within the rollout. We propose a training-time mechanism called \textbf{Credit-Attenuated Privileged Feedback} (CAPF), which makes this verifier-side information available through a Privileged Feedback call during training. CAPF lets the policy revise zero-reward attempts into positive-reward repair trajectories and attenuates credit for the feedback call and earlier actions to accommodate deployment without this call. Empirical research demonstrates that CAPF improves Qwen3-4B's average exact-match score from 44.7% under outcome-only RLVR to 48.5% on seven open-domain QA benchmarks. 

---
# Token Predictors Are Not Planners: Building Physically Grounded Causal Reasoners 

**Authors**: Zheng Lu, Mingqi Gao, Qinlei Xie, Wanqi Zhong, Hanwen Cui, Heng Cao, Zirui Song, Yifan Yang, Chong Luo, Bei Liu, Yiming Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.01810)  

**Abstract**: Current benchmarks for embodied vision-language planning often favor linguistic next-token prediction over physically grounded next-state reasoning. This rewards models that mimic statistical language priors rather than track causal dependencies, reducing physical planning to shallow sequence modeling. We argue that reliable physical autonomy requires a shift from linguistically grounded token prediction toward physically grounded causal reasoning. To this end, we introduce Causal-Plan-Bench, a high-fidelity diagnostic suite curated through multi-stage verification to evaluate embodied planning across four causal dimensions. We also construct Causal-Plan-1M, a million-scale corpus of explicit reasoning traces produced by a four-stage annotation pipeline over egocentric videos. Extensive evaluation shows that leading models still struggle to demonstrate genuine physical agency, with Gemini 3 Pro reaching only 38.18 on our benchmark. In contrast, our training recipe enables Causal Planner, built on Qwen3-VL-8B, to internalize physical logic for more accurate next-state estimation. The model achieves strong in-domain performance and cross-benchmark generalization, and reveals a Causal Scaling Law: scaling causal training data to one million instances yields a 36.3% relative gain, from 33.22 to 45.28. Overall, our work provides a concrete step toward turning agents from superficial token predictors into physically grounded causal reasoners. 

---
# OctoT2I: A Self-Evolving Agentic Text-to-Image Router 

**Authors**: Xu Jiang, Bin Chen, Gehui Li, Yule Duan, Ronggang Wang, Jian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.01803)  

**Abstract**: The explosive growth of Text-to-Image (T2I) models, from large-scale versions to lightweight, real-time ones, now faces diminishing marginal returns from single-model scaling. Agentic T2I methods emerged to alleviate this bottleneck by using multiple models. However, existing agentic T2I methods suffer from three key challenges: reliance on expensive handcrafted priors or human annotations, rigid single-path decision mechanisms, and a neglect of inference efficiency. To address these challenges, we introduce OctoT2I, a novel agentic framework that reformulates the T2I task as a joint optimization of generation quality and inference efficiency. OctoT2I implements a stateful, multi-round routing strategy that adaptively selects the most suitable tool based on its knowledge and memory. This strategy is enabled by a knowledge base built from scratch by our novel Self-Evolving Mechanism. This mechanism, which requires no human supervision, first autonomously defines foundational Conceptual Dimensions (eg, style, color, count) and then intelligently explores their combinations via an iterative" Propose--Solve--Evaluate--Learn"(PSEL) loop. The PSEL loop efficiently discovers each tool's capability frontier, driving continuous improvement without external guidance. Extensive experiments demonstrate that OctoT2I achieves competitive performance (0.96) on GenEval while delivering a 90.3% inference speedup and a 56.6% energy-efficiency gain over the leading baseline (Flow-GRPO), striking an exceptional balance between performance and efficiency. Code and models will be made available. 

---
# Consistency evaluation of benchmarks used for causal discovery 

**Authors**: Yuzhe Zhang, Chihui Chen, Lina Yao, Chen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.01789)  

**Abstract**: In graphical causal model, causal discovery aims to construct a causal graph based on numerical data and domain knowledge in plain text. However, the evaluation of causal discovery methods remains a challenge in the area as the progress of domain researches often makes benchmark causal graphs contain mis-aligned knowledge. This problem especially affects the evaluation of large language model (LLM) based causal discovery methods as they are sensitive to the new discoveries in the literature. This work is the first to systematically study the quality of benchmark causal graphs. Specifically, we design a pipeline that automatically retrieves relevant research papers from scientific databases, and prompts LLMs to check the consistency between the benchmark causal graphs and domain research papers. We evaluate 11 popular real-world benchmarks, for which our pipeline in total proceeds 38,081 domain papers. Our results show that popular benchmarks vary significantly in their consistency with domain research, with clear implications for causal discovery research. 

---
# Stochastic convergence of parallel asynchronous adaptive first-order methods 

**Authors**: Serge Gratton, Philippe L. Toint  

**Link**: [PDF](https://arxiv.org/pdf/2606.01787)  

**Abstract**: A new class of asynchronous adaptive first-order optimization methods is introduced, comprising asynchronous variants of several popular algorithms. Versions of these methods using momentum and/or inexact normalization are also considered. The convergence of methods in the class on non-convex functions is analyzed in a fully stochastic setting, and is shown to be (up to logarithmic factors) of order O(1/sqrt{t}) under reasonable assumptions. Numerical experiments suggest that such asynchronous adaptive algorithms are very relevant in heterogeneous large-scale machine learning systems. 

---
# Structure-Guided Adaptive Propagation for Protein-Protein Interaction Site Prediction 

**Authors**: Enqiang Zhu, Yizi Liu, Yilong Luo, Yao Chen, Yu Zhang, Baoshan Ma  

**Link**: [PDF](https://arxiv.org/pdf/2606.01781)  

**Abstract**: Accurate prediction of protein-protein interaction sites (PPIS) is essential for understanding cellular processes, disease mechanisms, and therapeutic target discovery. Graph-based deep learning has advanced PPIS prediction by incorporating residue-level structural context. However, most graph-based models still rely on fixed propagation schemes that treat all residues similarly, despite the structural and functional heterogeneity of protein interfaces. Such propagation may limit the ability to adapt information diffusion to local geometric environments, making it difficult to distinguish true interaction sites from structurally similar non-interacting neighbors. We present SGAP-PPIS, a structure-guided adaptive propagation model for PPIS prediction. Rather than using a fixed propagation mechanism, SGAP-PPIS leverages multi-scale geometric states from an equivariant graph neural network to generate residue-wise propagation coefficients. This design allows each residue to adaptively balance local feature preservation and neighborhood diffusion according to its geometric microenvironment. Experimental results show that SGAP-PPIS achieves competitive performance among the state-of-the-art methods on Test\_60. Ablation studies show that geometry-conditioned adaptive propagation, scale-aligned geometric guidance, and multi-step propagation-state representation jointly drive these improvements. 

---
# EvoBrain: Continual Learning of EEG Foundation Models Across Heterogeneous BCI Tasks 

**Authors**: Yangxuan Zhou, Sha Zhao, Jiquan Wang, Shijian Li, Gang Pan  

**Link**: [PDF](https://arxiv.org/pdf/2606.01767)  

**Abstract**: Electroencephalography (EEG) is the cornerstone of non-invasive brain-computer interfaces (BCIs), yet conventional decoding relies on fragmented, task-specific architectures that severely limit cross-task scalability. While EEG foundation models pre-trained on massive corpora promise universal brain decoding, current post-training depends on task-isolated fine-tuning. This static paradigm restricts knowledge transfer across heterogeneous tasks, hinders model scalability, and incurs computational and storage overheads that scale linearly with task count. To overcome these bottlenecks, we formulate downstream adaptation as a cross-task continual learning problem and propose EvoBrain, a dynamic, task-aware continual learning framework for unified EEG decoding. EvoBrain addresses the plasticity-stability trade-off via two complementary components: (1) Neuro-Spectral Task Normalization (NSN) aligns incoming tasks with historical statistics while recalibrating spectral responses to handle distributional and neuro-spectral shifts; and (2) Response-Affinity Distillation (RAD), combined with time-dependent replay, preserves old-task response geometry and promotes selective knowledge transfer between spectrally compatible tasks, effectively mitigating forgetting. Extensive evaluations across six distinct BCI tasks demonstrate that EvoBrain consistently surpasses state-of-the-art methods across diverse foundation backbones, optimally balancing plasticity and stability. To our knowledge, this work pioneers cross-task continual learning in the EEG domain, advancing the realization of a unified, one-for-all brain decoding system. 

---
# TriAlign: Towards Universal Truth Consistency in Personalized LLM Alignment 

**Authors**: Thi-Nhung Nguyen, Linhao Luo, Rollin Omari, Junae Kim, Thuy-Trang Vu, Dinh Phung  

**Link**: [PDF](https://arxiv.org/pdf/2606.01755)  

**Abstract**: Personalized large language models adapt responses to users' preferences and social attributes, but can introduce substantial universal truth inconsistencies across social groups, where some groups systematically receive less accurate responses on objective tasks. Existing alignment methods either ignore personalization or mainly focus on subjective preference alignment, largely overlooking fairness and consistency in universal truths. To address this gap, we study Truth-Invariant Alignment (TIA), an alignment problem for personalized LLMs that aims to ensure universal truths remain consistent across social groups while preserving personalization. We propose TriAlign, the first offline multi-agent reinforcement learning (MARL) framework for TIA, where each social group is modeled as an agent interacting. TriAlign jointly optimizes universal truth accuracy, cross-group truth consistency, and personalization through a fairness-aware objective and an explicit inconsistency penalty. Experiments across diverse benchmarks demonstrate that TriAlign achieves a stronger balance among these three objectives than strong baselines, reducing universal truth disparities across social groups while improving both objective task performance and personalization quality. 

---
# TrafficRAG: A Multimodal RAG Framework for Traffic Accident Liability Determination 

**Authors**: Xu Li, Zedong Fu, Xinyi Li, Xun Han  

**Link**: [PDF](https://arxiv.org/pdf/2606.01737)  

**Abstract**: Traffic accident liability analysis is a critical yet challenging task in intelligent transportation and legal assistance. Existing methods often suffer from low efficiency, subjective judgment, and inconsistent analysis results. Meanwhile, large language models are constrained by noisy video inputs and insufficient legal domain knowledge. To address these issues, this work presents TrafficRAG, a multimodal retrieval-augmented framework for automated traffic accident analysis and report generation. Specifically, the proposed framework first adopts a vision-language model to produce structured textual descriptions of accident scenarios, which serve as accurate retrieval queries. Based on these textual queries, a hybrid retrieval strategy integrating BM25 sparse retrieval and dense embedding retrieval is employed to fetch relevant traffic regulations and similar historical cases. Finally, the large language model incorporates retrieved legal knowledge and multimodal accident evidence for comprehensive reasoning, and generates standardized, legally grounded liability analysis reports. Extensive experiments show that TrafficRAG consistently outperforms baseline methods, achieving 77.32% Legal Norm Adaptation Accuracy, 81.71% Factual Faithfulness, and a Liability Ratio MAE of 5.48%. The results validate that integrating multimodal factual evidence with legal clauses via retrieval augmentation can effectively improve the reliability and accuracy of traffic accident liability determination. 

---
# Evidence-Gated LLM Priors for Multi-Objective Bayesian Optimization 

**Authors**: Jiangyu Chen, Banyi  

**Link**: [PDF](https://arxiv.org/pdf/2606.01730)  

**Abstract**: Large language models (LLMs) are increasingly used as heuristic advisors for black-box optimization, yet their suggestions and self-reported confidence are not necessarily calibrated to downstream objective values. This issue becomes more pronounced in multi-objective Bayesian optimization, where different objectives may require different expert knowledge and where an LLM expert can be useful for one objective but misleading for another.
We study how to use LLM-generated expert priors in discrete multi-objective Bayesian optimization without blindly trusting them. We propose an objective-wise reputation-market mechanism that treats each expert-objective pair as a falsifiable prior source. Expert weights are updated online from observed objective feedback, discounted over time, and gated by market-level trust. We then introduce a decoupled counterfactual gate that can use the LLM prior without confidence, use it with confidence, or abstain from the LLM prior entirely.
Across controlled synthetic stress tests and three molecule optimization benchmarks with \qwenflash{}-generated expert priors, we find that dynamic objective-wise calibration improves robustness over fixed LLM priors. However, raw LLM confidence is not reliably beneficial: on ESOL, confidence is positively correlated with prediction error; on FreeSolv, confidence can help; and on Lipophilicity, ignoring confidence remains strongest. Our fixed three-arm counterfactual gate improves over the first counterfactual variant on ESOL and FreeSolv, while an attempted margin portfolio exposes a useful negative result: margin selection should be acquisition-aware rather than based only on one-step prior error. 

---
# Characterization of Multi-Model Agentic AI Systems on General Tasks via Trace-Driven Simulation 

**Authors**: Donghwan Kim, Prakhar Singh, Younghoon Min, Jongryool Kim, Jongse Park, Kiwan Maeng  

**Link**: [PDF](https://arxiv.org/pdf/2606.01725)  

**Abstract**: Agentic AI completes tasks through iterative planning, tool use, and reasoning based on observed outcomes. Despite its popularity, its system-level behavior remains poorly understood, particularly for complex datasets and agent architectures-owing to highly non-deterministic execution, prohibitive evaluation costs, and limited visibility into proprietary models. This paper presents GAIATrace, the first token-level trace dataset of two state-of-the-art agentic systems (MiroThinker and OWL) running GAIA, a benchmark composed of a heterogeneous mix of general-purpose tasks. Unlike prior trace datasets, GAIATrace captures full reasoning tokens, task-level structures, and activities of every major participating LLMs, enabling in-depth systems research. Complementing the dataset, we present Vidur-Agent, a trace-driven simulator that can replay GAIATrace to perform reproducible, low-cost system evaluation across diverse simulated environments. Using both artifacts, we characterize how modern agentic systems handle general tasks and how various system design choices shape their behavior, yielding several unique findings. 

---
# MobEvolve: An Agentic Self-Evolving Heuristic System for Interpretable Human Mobility Generation 

**Authors**: Junlin He, Yihong Tang, Tong Nie, Ao Qu, Yuebing Liang, Hamzeh Alizadeh, Bang Liu, Wei Ma, Lijun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2606.01640)  

**Abstract**: Human mobility generation aims to synthesize realistic trip chains for target populations based on individual features. Existing paradigms, including deep generative models, LLM-based methods, and traditional heuristics, struggle to satisfy the complex demands of this task while simultaneously maintaining interpretability, behavioral plausibility, population-level distributional alignment, and inference efficiency. To bridge this gap, we introduce MobEvolve, the first agentic self-evolving heuristic framework for human mobility generation. MobEvolve initializes a behavior-inspired heuristic system and employs an LLM agent to iteratively evolve its internal logic. By diagnosing empirical misalignments and failure cases on a validation set, the agent proposes targeted updates and accumulates evolution memory for cumulative self-improvement. Extensive evaluations on the Singapore and Montreal benchmarks demonstrate that MobEvolve significantly outperforms state-of-the-art deep generative and LLM-based methods in individual trajectory fidelity, population-level distribution alignment, and behavioral plausibility, while preserving interpretability and high inference efficiency. 

---
# ReSkill: Reconciling Skill Creation with Policy Optimization in Agentic RL 

**Authors**: Zelin He, Haotian Lin, Boran Han, Wei Zhu, Haoyang Fang, Bernie Wang, Xuan Zhu, Runze Li, Matthew Reimherr  

**Link**: [PDF](https://arxiv.org/pdf/2606.01619)  

**Abstract**: Agentic reinforcement learning (RL) enables LLM agents to improve continuously from environment rewards, yet the resulting policies do not systematically accumulate reusable strategies that generalize across tasks. Modular skills can provide such reusable strategies, yet existing skill-augmented RL methods decouple skill creation from policy optimization, risking adopting skills that conflict with the evolving policy. Inspired by Anthropic's Skill Creator, we introduce ReSkill, an RL-in-the-loop skill creation framework that reconciles skill evolution with policy learning. ReSkill exploits the group-wise structure of GRPO to naturally embed three mechanisms with only marginal additional overhead: (1) an assertion-driven skill creator that diagnoses failures from past experience and proposes conditional, trigger-based skill revisions; (2) within-group rollout sampling that enables controlled comparison of skill versions, capturing which version best supports the policy's ongoing learning; and (3) Thompson Sampling with adaptive discounting to balance exploration and exploitation in skill version selection as the policy evolves. Across several domains, ReSkill consistently outperforms existing memory and skill-based RL methods, with the largest gains on unseen tasks. Analysis of the skill lifecycle shows skills being automatically created, tested, refined, and pruned as the policy improves, demonstrating reconciled skill-policy co-evolution. 

---
# Revisiting Ripple Effects in Knowledge Editing through Pressure-Aware Joint Neighborhood Optimization 

**Authors**: Haoben Huang, Shuxin Liu, Ou Wu, Di Gao  

**Link**: [PDF](https://arxiv.org/pdf/2606.01610)  

**Abstract**: Single-edit updates in large language models can trigger ripple effects across local knowledge neighborhoods: desirable propagation to related facts and unintended perturbation of preserved ones. Existing methods address these two effects separately, without explicitly modeling their coupling. We challenge this separation through an analysis of ripple responses across typical baselines, identifying two coupled design pressures: editable-side coordination and preserved-side leakage. We propose Joint Neighborhood Optimization (JNO), a new knowledge-editing framework to formalize and jointly address both pressures at the target-planning stage. JNO instantiates this principle through Pressure-Aware Coordination (PAC), which jointly optimizes neighborhood target representations under coupled constraints, and a semantic pre-execution gate that rejects high-risk target plans before parameter execution. Experiments on RippleEdits show JNO improves propagation and preservation metrics by at least 7.0% while preserving cross-backbone editing stability. 

---
# TRON: Targeted Rule-Verifiable Online Environments for Visual Reasoning RL 

**Authors**: Tianze Yang, Yucheng Shi, Ruitong Sun, Jingyuan Huang, Ninghao Liu, Jin Sun  

**Link**: [PDF](https://arxiv.org/pdf/2606.01599)  

**Abstract**: Reinforcement learning (RL) for visual reasoning needs scalable, verifiable, and controllable training signals. Existing visual RL post-training trains on static curated datasets, with fixed image-question-answer samples bounded by their collection budget. In this work, we introduce TRON (Targeted, Rule-verifiable Online eNvironments), an online environment substrate: a training rollout is generated on demand by a controllable generator-verifier program that samples a fresh latent visual state, renders an image, asks a question, and exactly verifies the answer. A single run can therefore draw an unbounded stream of fresh instances at the difficulty level required by the current curriculum. The current TRON suite contains 520 environments organized into five ability buckets (spatial, mathematical, diagram, pattern/logic, and counting); the same substrate supports both a single full model trained on all buckets and per-bucket ability-specialist models, with no additional data collection. We also introduce a substrate analysis covering generation reliability, instance and level diversity, cross-environment near-duplicates, and base-model pass rate by difficulty level. RL post-training with METHOD consistently improves performance on ten external multimodal reasoning benchmarks across Qwen3-VL-4B, Qwen2.5-VL-7B, and MiMo-VL-7B-SFT. 

---
# S-SPPO: Semantic-Calibrated Self-Play Preference Optimization 

**Authors**: Xiwen Chen, Wenhui Zhu, Jingjing Wang, Peijie Qiu, Zhipeng Wang, Huayu Li, ZhengXiao He, Xuanzhao Dong, Prayag Tiwari, Mingkun Xu, Yujian Xiong, Feng Luo, Abolfazl Razi, Brendan Hogan Rappazzo, Anderson Schneider, Yuriy Nevmyvaka  

**Link**: [PDF](https://arxiv.org/pdf/2606.01561)  

**Abstract**: Aligning Large Language Models (LLMs) with human preferences is often formulated via Direct Preference Optimization (DPO). However, the standard Bradley-Terry instantiation of DPO is limited in modeling common departures from transitivity in human preferences. To address this, recent work has introduced Self-Play Preference Optimization (SPPO), which iteratively refines the policy by training on self-generated win-lose pairs. Our investigation, however, reveals a critical instability in SPPO: the optimization is prone to policy degeneration when the preference oracle assigns overly confident wins to semantically indistinguishable responses. To mitigate this, we propose S-SPPO, a dual-space semantic calibration framework comprising: i) Supervision Calibration via semantic gating, which anneals win rate targets toward the maximum-entropy baseline as semantic overlap increases; and ii) Representation Calibration via latent repulsion to enforce geometric diversity to prevent manifold collapse and maintain latent diversity between chosen and rejected samples. Theoretically, we show that the calibration preserves the constant-sum game structure, facilitating convergence to a Nash Equilibrium. Empirically, S-SPPO avoids the performance degradation seen in prior methods, achieving 52.19% win rate and 47.46% length-controlled win rate on AlpacaEval 2.0 with Llama-3-8B, without using additional human-annotated preferences during training. The code will be available at this https URL. 

---
# RoleCDE:Benchmarking and Mitigating Role-Alignment Trade-offs in Role-Playing Agents 

**Authors**: Huayi Lai, Shichao Song, Simin Niu, Hanyu Wang, Jiawei Yang, Zhouxing Wang, Zhiqiang Yin, Xun Liang  

**Link**: [PDF](https://arxiv.org/pdf/2606.01552)  

**Abstract**: Role-playing agents(RPAs) are widely used to steer large language models(LLMs) toward role-consistent behavior, yet existing benchmarks mainly evaluate surface-level fidelity and offer limited insight into decision making under role-alignment value conflicts. To address this gap, we introduce RoleCDE, the first benchmark designed to evaluate RPAs under structured conflicts between role-specific values and alignment-oriented constraints. RoleCDE formulates role-aware decision making as cognitive dilemma scenarios, jointly evaluating role-scenario grounding, value conflict resolution, and decision tendencies. The benchmark is constructed at scale, covering approximately 8k diverse role profiles and scenarios and nearly 24k dilemma instances across three difficulty levels and eight role categories. Evaluation of several mainstream LLMs reveals a "Role Value Decoupling" phenomenon, where agents systematically default to alignment-and morality-consistent decisions rather than role-specific values when the two conflict, even under explicit role conditioning. This behavior is largely invariant to dilemma difficulty but varies substantially across role categories. We further show that RoleCDE-based fine-tuning effectively mitigates this decoupling by improving value trade-off reasoning, while preserving general role-playing fidelity and general reasoning performance. Code is available at: this https URL. 

---
# Joint Agent Memory and Exploration Learning via Novelty Signals 

**Authors**: Shizuo Tian, Xiaohong Weng, Rui Kong, Yuxuan Chen, Guohong Liu, Yuebing Song, Jiacheng Liu, Yuchen Li, Dawei Yin, Ting Cao, Yunxin Liu, Yuanchun Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.01528)  

**Abstract**: In open-ended environments, exploration is fundamental for autonomous agents, yet current language model agents struggle with this. Effective exploration requires memory, but retaining raw interaction histories is computationally expensive over long trajectories. While latent memory offers a solution to compress interaction histories, its training lacks reliable supervisory signals. We introduce \textbf{J}oint \textbf{A}gent \textbf{M}emory and \textbf{E}xploration \textbf{L}earning (\textbf{JAMEL}), a framework that trains agentic memory and exploration policy together through novelty-driven interaction. We observe that memory and exploration form a mutually dependent loop: sustained exploration requires memory to distinguish exhausted behaviors from unseen ones, while novelty-seeking interaction provides the supervision needed to make memory useful for future exploration. By utilizing deterministic and persistent novelty signals such as code coverage in the GUI domain, we provide natural, annotation-free supervision for the memory module. Empirical evaluations demonstrate that \ours successfully generalizes to unseen environments. Its exploration capability outperforms open-weight baselines and rivals the exploration depth of a closed-source model while reducing token consumption. Our code and model are open-sourced at this https URL. 

---
# TERRA: Task-Embedded Reasoning and Representation Architecture for Cross-Domain Applications 

**Authors**: Shayan Shokri  

**Link**: [PDF](https://arxiv.org/pdf/2606.01520)  

**Abstract**: A single action-conditioned latent predictive architecture can in principle be trained on the structured state of a driving scene, a robot workspace, or a financial order book. The ingredients for doing so within any one domain already exist and are individually validated: masked-latent prediction, action-conditioned latent world models, discrete action tokenization, and joint-embedding prediction on voxelized state. What is not established, and what TERRA addresses, is the transfer question: when does a representation or predictor learned in one structured-state domain carry over to a structurally analogous but otherwise unrelated domain, and by how much. We give this question a formal treatment. We model each domain as a controlled Markov process on a graded latent grid, factor any instantiation into thin domain adapters and a shared domain-invariant core, and identify a cross-domain correspondence with an approximate Markov decision process homomorphism whose quality is measured by a lax bisimulation discrepancy and, for domains lacking a shared coordinate system, by a Gromov-Wasserstein distance between their action-conditioned transition operators. Under a Lipschitz predictor we derive a transfer bound that separates source-model error from structural mismatch, grows geometrically in the prediction horizon, and is certified from below by the Gromov-Wasserstein distance; we then connect latent error to decision regret through the Lipschitz value property of bisimulation metrics. The resulting Structured-State Transfer Hypothesis is stated as a falsifiable claim with a preregistered experimental program, centered on a transfer test from driving scenes to order books, including conditions under which it is refuted. We present no empirical results: this is a research proposal that converts a widely repeated intuition into testable theory. 

---
# A Minimalist Brain-Computer Musical Interface for Real-Time Emotion-Driven Sonification: System Design and Preliminary Evaluation 

**Authors**: Pablo A. Monroy-D'Croz, Rafael Ramirez-Melendez, Julian Cespedes-Guevara  

**Link**: [PDF](https://arxiv.org/pdf/2606.01473)  

**Abstract**: This paper presents a minimalist brain-computer Musical Interface (BCMI) that functions as a real-time affective sonification system, translating prefrontal EEG activity into adaptive music. Emotional valence is estimated from frontal alpha asymmetry (AF7/AF8) and mapped to musical features such as mode, tempo, rhythmic density, and pitch register through a stochastic generative algorithm. The system integrates wireless EEG acquisition, real-time Python signal processing, and Ableton Live-based music generation synchronized via Lab Streaming Layer. An experiment with 22 participants investigated whether intentional emotional self-induction could modulate the BCMI neurofeedback signal. Linear mixed-effects analyses found no significant effects of target emotion or time, indicating that the frontal alpha asymmetry signal did not reliably distinguish instructed emotional states. Individual differences, including musical training and acting experience, explained more variance than the experimental manipulation, which accounted for only 0.40\% of total signal variance. These findings highlight the challenges of using frontal alpha asymmetry as a voluntary control signal for closed-loop emotion regulation and suggest methodological directions for future BCMI research. 

---
# An Enigma of Artificial Reason: Investigating the Production-Evaluation Gap in Large Reasoning Models 

**Authors**: Mingzhong Sun, Teresa Yeo, Armando Solar-Lezama, Tan Zhi-Xuan  

**Link**: [PDF](https://arxiv.org/pdf/2606.01462)  

**Abstract**: Studies of human reasoning have shown that people are typically stronger at evaluating reasoning than producing it from scratch. In contrast, large reasoning models (LRMs) are trained to excel at producing long chains of reasoning to solve complex problems. How then do LRMs perform at evaluating reasons? We investigate this with the Valid-Answer-Invalid-Reasoning (VAIR) dataset: math problems and solutions with trivial reasoning flaws but valid answers, designed to isolate reasoning evaluation from the confound of reasoning production. Unlike humans, who we find are only 6% worse at grading than solving such problems, we find a substantial production-evaluation gap in LRMs: frontier models score as low as 48% when evaluating VAIR solutions, despite near-perfect solution production.
Why this enigma? Through chain-of-thought (CoT) analysis, we find evidence of an answer confirmation bias: LRMs often produce then check for the correct answer instead of carefully verifying each step, fabricating rationalizations even when noticing anomalous reasoning. Linear probes corroborate this, showing that while LRM activations encode some representation of valid reasoning, they fail to robustly represent VAIR solutions as invalid. Causal patching of the final answer's representations causes LRM verdicts and activations to flip, demonstrating that answer validity is responsible for models' confirmation biases. These findings indicate an outstanding limitation in dominant approaches to reasoning training, which incentivize LRMs to produce and confirm reasoning towards correct answers, but not to robustly evaluate the underlying reasons. 

---
# Transferring Information Across Interventions in Causal Bayesian Optimization 

**Authors**: Mohammad Ali Javidian  

**Link**: [PDF](https://arxiv.org/pdf/2606.01457)  

**Abstract**: Bayesian optimization is a popular way to optimize expensive systems, where every experiment, simulation, or intervention costs time or money. In its standard form, it treats the variables we control as plain inputs to a black box and cannot tell apart mere correlation from a real cause and effect. Causal Bayesian optimization closes part of this gap by using a known causal graph together with observational data to decide which variables are worth intervening on. Existing methods, however, learn the effect of each possible intervention almost in isolation, even though in a causal system these effects usually share the same underlying mechanisms. We propose graph-coupled causal Bayesian optimization, which ties the different intervention effects together through the uncertainty we have about a small set of shared causal parameters. The result is a causal kernel that lets evidence collected from one intervention improve our estimate of related interventions. For identifiable linear Gaussian causal models, we show that this kernel has low rank, bounded by the number of shared parameters rather than by the size of the intervention menu. This in turn yields an information-gain bound that grows only logarithmically in the optimization horizon, and a regret bound that cleanly separates three sources of error: optimization, causal estimation, and the choice of which intervention sets to consider. We also describe nonlinear and adaptive extensions. Across theory-aligned Gaussian systems, shared-mechanism stress tests, and standard causal optimization benchmarks, the method keeps the benefits of causal Bayesian optimization while transferring information across related interventions, with the clearest gains when direct interventions on the target's parents are unavailable and sparse interventional data must be reused across a large family of candidate interventions. 

---
# Self-Revising Discovery Systems for Science: A Categorical Framework for Agentic Artificial Intelligence 

**Authors**: Fiona Y. Wang, Markus J. Buehler  

**Link**: [PDF](https://arxiv.org/pdf/2606.01444)  

**Abstract**: Scientific discovery is not only answer generation but revision of the representational regime in which evidence, artifacts, operations, and verifiers are typed. We develop a category-theoretic account of agentic discovery for materials science. In a fixed regime b with schema category S_b, the system state is a copresheaf I_t: S_b -> Set, and provenance is the category of elements \int_{S_b} I_t. Fixed-regime operation is an update on such states, endofunctorial only when provenance-preserving refinements are specified and preserved. Discovery is instead a verified regime transition u: S_b -> S_b': old artifacts are preserved, transported by the left Kan extension Lan_u I_t, and compared with the post-transition state to identify residual content beyond functorial transport. This separates retrieval, search, and discovery without subjective novelty. We instantiate the framework in two systems. In Builder/Breaker, a protein-mechanics world model is revised under a Minimum Description Length gate; the accepted law expresses within-chain flexibility as all-mode elastic compliance conditioned by slow collective-mode participation, or mode-conditioned compliance. In CategoryScienceClaw, typed skills, artifacts, open needs, workflow mutation, gates, stress tests, and public discourse become a proof-carrying knowledge-computation graph. A fiber-network example records candidate models, rejected alternatives, an AIC gate, perturbation tests, and an accepted orientation-tensor anisotropic stiffness surrogate over an isotropic fiber-count descriptor. Together, the cases show how category theory can be both a mathematical language for discovery and an engineering specification for self-revising AI discovery systems. 

---
# Dive into Ambiguity: A*-Inspired Multi-Agents Commonsense Obfuscation Attack on LLM Prompts 

**Authors**: Boxuan Wang, Zhuoyun Li, Xiaowei Huang, Yi Dong  

**Link**: [PDF](https://arxiv.org/pdf/2606.01441)  

**Abstract**: Large language models (LLMs) excel in reasoning and knowledge-intensive tasks but remain vulnerable to prompt-level adversarial attacks that preserve intent while triggering commonsense hallucinations. This vulnerability is urgent, as LLMs are rapidly integrated into safety-critical domains where factual reliability is non-negotiable. Existing attack methods either lack efficiency or fail to capture the adaptive strategies of real-world adversaries. We propose an A*-inspired Factual Error Induction Framework, a framework for generating semantically aligned yet obfuscated prompts. At its core is a Hierarchical Rewrite Strategy guided by a dynamic semantic dispersion coefficient $\gamma$ that balances conservative edits early with aggressive obfuscations later, following a reverse simulated annealing schedule. To enhance interpretability, we further introduce Agentic Mechanism Labeling, which discovers and refines adversarial mechanisms, offering interpretable reverse optimization. Theoretically, we prove that prompt rewriting follows a contractive recurrence, leading to semantic collapse as $\gamma$ decreases. Empirically, across diverse LLMs, our method achieves higher attack success rates than exhaustive exploration while requiring fewer attempts, demonstrating both efficiency and effectiveness. 

---
# Don't Ask the LLM to Track Freshness: A Deterministic Recipe for Memory Conflict Resolution 

**Authors**: Vikas Reddy, Sumanth Challaram  

**Link**: [PDF](https://arxiv.org/pdf/2606.01435)  

**Abstract**: LLM-based memory systems increasingly maintain facts that evolve over time, where a recurring failure is conflict resolution: when a fact has multiple contradictory values, which should the agent return? MemoryAgentBench (MAB; Hu et al., 2026) makes this explicit in its FactConsolidation task: facts are numbered, the counterfactual has the higher serial, and agents are told newer facts have larger serials. Yet every published system underperforms: HippoRAG-v2 reaches 54% on single-hop (FC-SH), BM25 48%, Mem0 18%, and the temporal KG Zep/Graphiti just 7%. Multi-hop is near-unsolved (at most 7% across 22 systems).
We argue the bottleneck is the assembly step: baselines leave conflict resolution to LLM-mediated retrieval or generation rather than version-aware aggregation. A matched-setup comparison (same backbone, retrieval, chunking, TOP_K) shows that replacing the LLM-judgment answer pipeline with candidate-extraction plus Python max(serial) yields +10.8 points on FC-SH (gpt-4o-mini), widening from +8 at 6K to +21 at 262K. This is a whole-pipeline effect (resolver, prompt, format, and temperature vary jointly); isolating the resolver is future work. The recipe reaches 78.0% on FC-SH (gpt-4o-mini), 94.8% (gpt-4o), and 30.2% on FC-MH (gpt-4o-mini, rising to 51.5% with gpt-4o) via a per-hop deterministic extension of Self-Ask. At matched-262K, it beats HippoRAG-v2 by +28 points and the best published FC-MH result by +20.
The implication is corrective for the subfield: the bottleneck on conflict resolution is assembly (post-retrieval aggregation), not storage. A LongMemEval knowledge-update check shows the mechanism ports from max(serial) to max(timestamp) but only ties LLM judgment (57.8% vs 64.4%, n=45): deterministic aggregation is the right primitive for current-value conflicts and must be composed with question-type-aware handling for broader memory QA. 

---
# GovAI-Pipe: A Layered AI Governance Pipeline for Citizen-Facing AI in Turkey's e-Government Gateway 

**Authors**: Ahmet Kaplan  

**Link**: [PDF](https://arxiv.org/pdf/2606.01417)  

**Abstract**: Turkey's e-Government Gateway (e-Devlet) serves over 68 million registered users with more than 9,200 government services, and is increasingly integrating artificial intelligence into citizen-facing applications such as chatbot assistants and eligibility assessments. However, no structured technical governance infrastructure currently connects high-level AI policy frameworks, such as the EU AI Act, OECD AI Principles, and Turkey's own National AI Strategy, to the operational reality of deploying AI within a centralized e-government platform. We propose GovAI-Pipe, a four-layer governance pipeline designed using Design Science Research methodology that maps the AI model lifecycle to governance checkpoints: (1) pre-deployment validation for bias testing, explainability, and privacy impact assessment; (2) deployment governance for risk-tier classification and approval workflows; (3) runtime monitoring for drift detection, fairness tracking, and human-in-the-loop escalation; and (4) post-incident governance for audit trails, rollback, and citizen redress. Each layer is anchored to specific provisions of the EU AI Act, the GDPR data protection framework, and the National AI Strategy. We demonstrate the framework through two high-risk e-Devlet use cases, showing how GovAI-Pipe operationalizes governance principles as auditable, technical pipeline components. 

---
# Self-Healing Agentic Orchestrators for Reliable Tool-Augmented Large Language Model Systems 

**Authors**: Rahul Suresh Babu, Adarsh Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2606.01416)  

**Abstract**: Tool-augmented large language model (LLM) agents rely on orchestration layers that coordinate planning, retrieval, tool invocation, validation, memory, and recovery. In these systems, failures arise not only from model errors, but also from orchestration-level issues such as tool timeouts, malformed arguments, stale context, contradictory evidence, retry loops, and unverified intermediate outputs. This paper presents a self-healing agentic orchestrator that treats reliability as a bounded runtime control problem. The orchestrator maps observable failure signals to inferred failure classes, selects targeted recovery actions under explicit budgets, verifies recovered trajectories, and records observability traces. We evaluate the approach on a 100-task controlled fault-injection benchmark against static workflow, retry-only, ReAct-style, and full-replanning baselines. Self-healing achieves 98.8\% task success, compared with 94.5\% for retry-only and 93.8\% for full replanning. A matched recovery-budget sweep shows that self-healing outperforms retry-only and full replanning at every tested budget, with the largest gap under a single recovery attempt: 94.0\% versus 85.3\% and 88.2\%, respectively. Under a controlled semantic silent-failure setting, verifier-guided self-healing reduces silent failures to 0.0\%, while non-verifying baselines return wrong-but-plausible outputs more often. A compact model-in-the-loop validation shows that the same recovery mechanism can operate when a live tool-calling model performs tool selection, argument generation, and answer synthesis over local fault-injected tools. These results provide controlled evidence that failure-aware, budgeted, and verification-guided orchestration improves reliability and diagnosability in tool-augmented LLM systems. 

---
# GuidaPA: Privacy-Preserving Chatbot for Public Administration via Federated Learning 

**Authors**: Daniel M. Jimenez-Gutierrez, Albenzio Cirillo, Raffaele Nicolussi, Alessio Beltrame, Andrea Vitaletti  

**Link**: [PDF](https://arxiv.org/pdf/2606.01386)  

**Abstract**: We present GuidaPA, a privacy-preserving chatbot for the Italian Public Administration (PA) trained via Federated Learning (FL) on documentation from two national PA platforms, SIGESON and SIDFORS. Our corpus includes approximately 8 pages of SIGESON manuals and 31 pages of SIDFORS manuals/FAQs; while this study uses public documentation as a safe proxy, the intended deployment extends to restricted internal sources (e.g., tickets, officer manuals, database extracts) that can not be centrally pooled due to regulatory and organizational constraints. GuidaPA integrates role-based access control, secure client-side preprocessing, explicit monitoring of non-IID effects, and parameter-efficient federated fine-tuning of large language models. Using QLoRA (4-bit) over 15 federated rounds with an 80/20 train-test split per client, we evaluate answer quality with ROUGE, BLEU-4, and METEOR. The best federated model achieves ROUGE-1/2/L of 61.10/55.77/59.44, BLEU-4 of 45.02, and METEOR of 63.94-close to private centralized fine-tuning while keeping data on-site. Compared to the general-purpose baseline, domain fine-tuning improves ROUGE-1 from 41.45 to 62.18 and BLEU-4 from 26.97 to 50.90. Overall, the results indicate that FL can deliver high-quality conversational AI for public services without centralized data sharing 

---
# Early Diagnosis of Wasted Computation in Multi-Agent LLM Systems via Failure-Aware Observability 

**Authors**: Xianyou Li, Weiran Yan, Yichao Wu, Penghao Liang, Mengwei Yuan, Jianan Liu, Jing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.01365)  

**Abstract**: Tool-using multi-agent large language model (LLM) systems spend computation through model tokens, tool calls, retries, and code execution before producing an answer. When a run fails, final-answer evaluation reveals the endpoint but usually not the point at which the trajectory stopped making recoverable progress. This paper introduces a failure-aware observability framework for diagnosing wasted computation in multi-agent LLM traces. The framework maps recurring failure modes to online trace signals, including tool reliability, execution recovery, orchestration loops, evidence availability, information change, and budget pressure. We instantiate the framework in a three- agent question-answering system and evaluate it on 165 GAIA validation traces under identical execution caps. Operational failures remain common: 22/53 level-1 runs, 33/86 level-2 runs, and 12/26 level-3 runs fail to produce a usable final answer. The traces expose different mechanisms behind these outcomes, including insufficient evidence, repeated-action loops, max-step termination, tool-failure streaks, and execution calls that succeed without useful output. Mean token use rises from 8,152 tokens at level 1 to 16,389 tokens at level 3, while evidence availability and sentence-level support diverge. A cached 10-trace LLM-judge grounding audit shows that cheap online signals and deeper semantic metrics capture complementary layers of failure. The results position failure-aware observability as a diagnostic layer between raw execution logs and final-answer accuracy. 

---
# FlowTime: Towards Continuous Generative Watch Time Prediction via Flow-based Personalized Priors 

**Authors**: Hongxu Ma, Han Zhou, Chenghou Jin, Jie Zhang, Xiaoyu Yang, Chunjie Chen, Jihong Guan, Shuigeng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.01352)  

**Abstract**: Watch time has emerged as a pivotal metric for optimizing deep user engagement in short-video recommender systems. However, current methods of watch time prediction (WTP) suffer from inherent paradigm-specific limitations. Direct Regression faces mean-collapse due to unimodal Gaussian assumptions, while Ordinal Regression is hampered by quantization errors from rigid discretization. Similarly, Discrete Generative Regression struggles with high inference latency and heuristic vocabulary design. Beyond these specific flaws, a shared deficiency is the inability to capture the intrinsic multimodality and heterogeneity of User-Item Interaction Patterns. To address these challenges, we first revisit the WTP problem from a causal perspective and identify these user-specific patterns as structural confounders that modulate watch time outcomes, where identical interests manifest as distinct watch time outcomes conditioned on diverse user habits. Then, we formally propose a new (or the fourth) paradigm -- Continuous Generative Regression, and introduce FlowTime, a novel method utilizing a One-step Generative Variational Autoencoder. FlowTime effectively circumvents the latency of iterative denoising while maintaining the expressivity of continuous latent spaces. Furthermore, we design a Flow-based Personalized Prior that leverages NFs to warp a standard Gaussian prior into a complex, history-conditioned manifold, thereby enabling the adaptive modeling of multimodal interaction patterns. Finally, we build TimeRec, the first open-source WTP Library, alongside a novel personalization metric to establish a rigorous benchmarking standard. Extensive offline experiments and online A/B tests demonstrate FlowTime's significant superiority over SOTA methods. 

---
# Recognize Your Orchestrator: An Entropy Dynamics Perspective for LLM Multi-Agent Systems 

**Authors**: Junze Zhu, Weihao Chen, Xuanwang Zhang, Zhen Wu, Xinyu Dai  

**Link**: [PDF](https://arxiv.org/pdf/2606.01351)  

**Abstract**: The transition from single-turn models to Multi-Agent Systems (MAS) promises enhanced problem-solving capabilities, yet the centralized orchestration topology remains a critical point of fragility. To analyze this, we propose a Mean-Field Entropy Dynamics framework, modeling the orchestration process as a system governed by the competing forces of task resolution and cumulative context loading. To facilitate validation, we introduce Inverse Workflow Generation (IWG), a multi-agent pipeline that synthesizes process-verifiable, high-complexity benchmarks with dense intermediate checkpoints. We demonstrate that our entropy dynamics model fits empirical trajectories, providing physically interpretable parameters that quantify system stability and performance collapse. Crucially, our analysis uncovers a ``Reasoning Trap": while reasoning-heavy models excel in isolated tasks, they frequently fail as orchestrators due to context squeezing. Elucidating the physical mechanisms underlying the Orchestrator and quantifying systemic uncertainty offers insights for the MASs' architectural design. 

---
# Science Earth: Towards A Planet-Scale Operating System for AI-Native Scientific Discovery 

**Authors**: Zhe Zhao, Haibin Wen, Yingcheng Wu, Jiaming Ma, Yifan Wen, Jinglin Jian, Jiacheng Ge, Xiangru Tang, Bo An, Ming Yin, Sanfeng Wu, Mengdi Wang, Le Cong  

**Link**: [PDF](https://arxiv.org/pdf/2606.01316)  

**Abstract**: Scientific discovery demands intelligence, perseverance, and serendipity
across vast search spaces. Today, top scientific capabilities remain
siloed--one AI system for biological analysis, another for clinical
reasoning, mathematical derivation, or materials simulation--and no
pre-designed team can anticipate every skill a question will need.
Science Earth is a planet-scale scientific runtime in which any
capability--a simulation cluster, a wet-lab robot, a proof engine, a
single-cell pipeline--can connect to any other, with collaboration
structure emerging from the question itself. Its underlying EACN protocol
lets capabilities discover one another, negotiate task ownership, and
adjudicate across incompatible evidentiary standards without prior
knowledge of who will meet whom. This shifts the organizing challenge from
workflow design to open-ended connectivity. Two runs validate this under
structurally distinct conditions. In a trans-Pacific higher-order Kuramoto
synchronization study, agents identified and corrected a closure-ratio
assumption in Ott-Antonsen analytic theory that fails outside the
Lorentzian limit, within thirty minutes. In an eight-agent single-cell run
on the 4.88M-cell Kang 2024 pan-cancer atlas, heterogeneous capabilities
coupled over a 64.9-hour window with one structural external instruction,
producing three new result layers and anchoring findings against an
independent wet-lab study on an adjacent CCR8- TIGIT+ Treg subset. These
cases are a first empirical reading, not a benchmark sweep. They show that
when AI capabilities are truly connectable and coordination emerges from
the problem, scientific reasoning becomes a distributed, self-correcting
process--a step towards scaling AI-native discovery to the planet. 

---
# SkillSmith: Co-Evolving Skills and Tools for Self-Improving Agent Systems 

**Authors**: Yangbo Wei, Zhen Huang, Shaoqiang Lu, Junhong Qian, Qifan Wang, Chen Wu, Lei He  

**Link**: [PDF](https://arxiv.org/pdf/2606.01314)  

**Abstract**: Recent self-evolving agents have shown that skills can be discovered, refined, and accumulated through execution. However, existing skill-evolution frameworks typically assume a fixed tool layer and evaluate each skill independently, limiting their ability to repair tool-level failures or reason about interactions among skills. We propose SkillSmith, a synergy-aware skill-tool co-evolution framework. SkillSmith introduces a unified proposal space in which reflection produces atomic bundles that jointly modify skills and tools, allowing tools to be wrapped, edited, composed, split, or retired when skill evolution identifies a reusable capability gap. To guide this joint search, SkillSmith maintains an ecological utility model inspired by Lotka-Volterra dynamics, where an interaction matrix estimated from execution traces captures pairwise complementarity and conflict among skills and provides pressure signals for retrieval, mutation prioritization, and retirement. Furthermore, SkillSmith records anti-patterns, including failure signatures, causal attributions, and remedies, to accelerate diagnosis and veto proposals that repeat known mistakes. Experiments on three benchmarks, including WildClawBench, and five Qwen3.5 model scales show that SkillSmith consistently outperforms strong baselines, with gains that amplify as task complexity and multi-skill co-activation increase. 

---
# ANDES: Agent Native Data Evolving Synthesis Tool for Autonomous Instruction Alignment 

**Authors**: Zhengyang Zhao, Shengjie Ye, Lu Ma, Hao Liang, Hengyi Feng, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.01279)  

**Abstract**: AI agents are increasingly being tasked with automating AI research itself, particularly the critical post-training phase that transforms base LLMs into aligned assistants. However, recent evaluations reveal that even frontier agents struggle to perform this task. While the success of post-training fundamentally relies on acquiring high-quality data, relying on agents to autonomously curate targeted training datasets from the open web introduces severe challenges. Executing the long-horizon tasks of searching, filtering, and balancing data within noisy web environments frequently overwhelms an agent's limited context, ultimately leading to degraded dataset quality and suboptimal downstream training performance. To bridge this gap, we introduce Andes (Agent Native Data Evolving Synthesis), a framework that reimagines data generation as a plug-and-play \emph{agent skill}. Rather than forcing agents to devise complex data-gathering strategies from scratch, \textsc{Andes} provides an intelligent abstraction layer. By leveraging a self-evolving World Tree routing mechanism and actionable diagnostic reports, it allows trainer agents to dynamically steer data synthesis through an interactive, closed-loop interface. We demonstrate that under strict compute constraints, equipping foundationally weaker agents with Andes improves automated alignment, securing state-of-the-art performance on PostTrainBench and robust cross-task generalization. Our project is available at this https URL. 

---
# Emergent Ordinal Geometry in Transformers Trained on Local Comparisons 

**Authors**: Nishit Singh  

**Link**: [PDF](https://arxiv.org/pdf/2606.01269)  

**Abstract**: Transitive inference is the challenge of inferring that A < C from knowing only adjacent relations (A < B, B < C). It is solved by humans and animals not through logical chaining but via an analogue mental number line, whose signature is the symbolic distance effect: distant comparisons are easier than nearby ones. We ask whether Transformers acquire the same primitive, training small models exclusively on adjacent comparisons from a hidden total order and evaluating generalization to unseen distant pairs. We find that out-of-distribution generalization emerges alongside a striking geometric reorganization: entity embeddings collapse onto a one-dimensional manifold whose principal axis recovers the hidden rank order with near-perfect fidelity, and this structure is sensitive to optimization in ways that produce grokking-like transient dynamics. Critically, even when accuracy is at ceiling, decision confidence and geometric separation both scale monotonically with rank distance, directly mirroring the symbolic distance effect observed across decades of behavioural experiments on humans, primates, and rodents. These results ground a 50-year-old behavioural regularity in the geometry of learned representations, offering a mechanistic account of transitive inference that bridges cognitive science and modern neural networks. 

---
# SIRIUS-SQL: Anchoring Multi-Candidate Text-to-SQL in Execution Feedback 

**Authors**: Leo Luo, Haining Xie, Siqi Shen, Zhipeng Ma, Rui Ling, Hang Xu, Hefeng Jiang, Dingwei Chen, Yang Li, Peng Chen, Jie Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2606.01246)  

**Abstract**: Text-to-SQL on complex schemas is unreliable on a single pass, so recent systems generate multiple SQL candidates and let voting filter out errors. Yet voting alone is not enough, because the multi-candidate recipe has three coupled weaknesses: 1) sampling more from a single generator produces increasingly redundant candidates, 2) existing pipelines apply one generic correction to every non-clean execution result, while runtime errors, timeouts, and empty results each indicate a different distance from correctness, and 3) existing selectors rely on a single angle such as result-majority voting or pairwise SQL comparison, missing what other angles would have caught. We present SIRIUS-SQL, which addresses all three weaknesses. A difficulty-smoothing RL recipe trains SIRIUS-32B to generate diverse executable SQL candidates, paired with a generalist LLM that fills in gaps left by the specialist. An execution-grounded lifecycle classifies each outcome and applies targeted repair before candidates re-enter the pool. A confidence-gated hybrid selector combines execution-result agreement with pairwise SQL-form judgment, escalating only near-tied cases to a deterministic structural check. SIRIUS-SQL reaches 75.88% on BIRD dev and 91.20% on SPIDER test. Two of three generalist pairings surpass Agentar-Scale-SQL, the strongest published multi-candidate system on BIRD dev. 

---
# Brain-Atlas-Guided Generative Counterfactual Attention for Explainable Cognitive Decline Diagnosis Using Multimodal Connectomes 

**Authors**: Xiongri Shen, Jiaqi Wang, Zhenxi Song, Yi Zhong, Leilei Zhao, Xin He, Baiying Lei, Zhiguo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.01237)  

**Abstract**: Mild cognitive impairment (MCI) and subjective cognitive decline (SCD) are closely associated with the early Alzheimer's disease continuum, where accurate and explainable diagnosis is important for early risk assessment and intervention. Existing connectome-based deep learning models can improve classification performance but often provide limited insight into disease-related functional and structural connectivity changes. This paper proposes an atlas-knowledge-guided Generative Counterfactual Attention-guided Network (GCAN) for explainable cognitive decline diagnosis using multimodal brain connectomes. GCAN formulates diagnosis as a source-to-target counterfactual generation problem, where target-label connectomes are generated from source-label inputs and their differences are used to construct counterfactual attention maps. To preserve connectome topology, an Atlas-aware Bidirectional Transformer (AABT) performs network-level token encoding and decoding under brain-atlas constraints. The framework is further extended from functional connectivity (FC) to joint functional and structural connectivity (SC) modeling, enabling counterfactual analysis of complementary functional reorganization and structural topology changes. Experiments on hospital-collected and ADNI datasets show that GCAN achieves competitive performance across HC vs. SCD, HC vs. MCI, and SCD vs. MCI classification tasks. Visualization, circular connectome analysis, CAM-based comparison, ablation studies, and confidence interval analysis further support the interpretability and reliability of the proposed framework. Modality-specific FC and SC pre-trained classifiers are used to provide target-state priors for counterfactual generation while being separated from the downstream diagnostic classifier to prevent data leakage. 

---
# HomeFlow: A Data Flywheel for Smart Home Agent Training with Verifiable Simulation 

**Authors**: Yi Gu, Huacan Wang, Shuo Zhang, Yuqing Hou, Lei Xue, Weipeng Ming, Chen Liu, Fangzhou Yu, Kuan Li, Ronghao Chen, Sen Hu, Xiaofeng Mou, Yi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.01230)  

**Abstract**: Large language model agents are moving beyond text-only interaction toward physical-world control, with smart homes as a representative domain. Real domestic interaction requires understanding ambiguous intents, operating in dynamic environments, and performing multi-turn reasoning. However, existing methods struggle to generate high-quality training data for smart home agents. We propose HomeFlow, a verifiable data flywheel for this domain. HomeFlow uses HomeEnv as a unified simulation environment and HomeMaker to procedurally generate diverse home settings. Subsequently, Blueprint compiles open-ended user intents into executable state-based success conditions, while MCTS-Flow synthesizes diverse, verifiable multi-turn trajectories through environment-guided tree search. We then optimize the agents via supervised fine-tuning and step-wise RLVE, which facilitates iterative improvement through authentic physical feedback. We further construct SmartHome-Bench to evaluate the agent across various smart home tasks. On this benchmark, HomeFlow-RL-4B and HomeFlow-RL-8B achieve task success rates of 84.60% and 87.03%. It is worth noting that HomeFlow-RL-8B even surpasses the leading GPT-5.5 by 1.23 percentage points. 

---
# Application of Algorithms in Energy-Efficient Design Platforms for Green Building 

**Authors**: Na Yu, Fu Wenli, Guo Fei  

**Link**: [PDF](https://arxiv.org/pdf/2606.01229)  

**Abstract**: During green building design, computer-aided energy assessment is widely used to improve efficiency and achieve overall optimization. This paper presents a platform that combines Building Information Modeling (BIM), sensor operational data, and advanced simulation workflows using robust algorithms. The platform uses a multi-layer service architecture with dynamic energy simulation and evolutionary multi-objective optimization, connected via a high-performance C++ core and adaptive agent models. A mid-rise office building was selected as the case study. Five representative areas were chosen to collect data on building envelope characteristics and occupancy patterns. After preprocessing, missing sensor data accounted for 3.2% of annual records, and all variables were standardized using 15-minute interpolation. After 40 optimization rounds, annual energy consumption per square meter dropped by 29.3% from 315 kWh/m2 to 223 kWh/m2. The lifecycle cost increase for occupants was limited to 3.7%, and discomfort hours were reduced to under 70 hours per year. Analysis of Pareto optimal solutions shows that the envelope U-value ranges from 1.05 to 1.57 W/m2K, and nighttime ventilation rate ranges from 2.1 to 3.6 h-1, both closely linked to energy performance. The results confirm that the integrated algorithm framework offers good scalability, strong performance, and technical feasibility for green building design. This platform provides a reliable decision-support tool for design engineers and sustainability practitioners, enabling accurate, data-driven delivery of energy-efficient buildings. 

---
# Advanced Mathematics Learning Behavior Prediction and Academic Early Warning Model Based on Multimodal Data Analysis 

**Authors**: Liu Qiong, Li Zhengbo  

**Link**: [PDF](https://arxiv.org/pdf/2606.01224)  

**Abstract**: Early detection of at-risk students and timely academic intervention pose major challenges in advanced mathematics education, where complex conceptual hierarchies and nonlinear learning trajectories often hold back students' academic performance. This study adopts multimodal data analytics to build a dynamic framework for learning behavior prediction and academic early warning. It constructs a hierarchical knowledge graph ontology, realizes adaptive edge weighting according to problem difficulty and student performance, and combines heterogeneous graph attention with temporal sequence modeling to capture students' evolving knowledge states. Empirical tests on semester-long multimodal datasets prove that this method can accurately identify high-risk students and effectively track error propagation. Targeted interventions greatly improve students' knowledge mastery and reduce academic risks. The results verify that integrating knowledge graph analytics with multimodal temporal modeling can deliver more efficient and personalized learning support for advanced mathematics education. 

---
# The Shape of Wisdom: Decision Trajectories in Language Models 

**Authors**: Shailesh Rana  

**Link**: [PDF](https://arxiv.org/pdf/2606.01202)  

**Abstract**: Language models do not simply choose an answer at the output layer. In a 9,000-trajectory MMLU study across Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, and Mistral-7B-Instruct-v0.3, the score of the answer moves across depth in structured ways. We describe each trajectory with three quantities: the current answer margin, the next-layer change in that margin, and the distance from a decision flip. The main empirical picture is that correctness and stability are different: the largest group is unstable-correct, not stable-correct. A traced subset then asks what moves the margin. In stable-correct cases, the average attention scalar points in the correct direction, while the average MLP scalar does not; span deletion shows that removing answer-supporting text hurts the margin and removing distractor-like text helps it. The result is not a full circuit explanation. It is a reproducible way to see which answers are settled, which remain fragile, and which measured sources move them. 

---
# Can LLM Agents Sustain Long-Horizon Organizational Dynamics? 

**Authors**: Xuancheng Zhu, Yang Yue, Shuaibing Wan, Zihan Dou, Xiaohan Zhang, Yongrui Liu, Guoshun Nan  

**Link**: [PDF](https://arxiv.org/pdf/2606.01199)  

**Abstract**: Large language agents are increasingly used for social simulation, yet it remains unclear whether they can sustain coherent behavior in structured organizations, where goals must propagate through hierarchy, tasks depend on prior execution, and artifacts accumulate over long horizons. We formulate long-horizon organizational simulation as a memory-centered coordination problem and introduce TaskWeave, a hierarchical agentic framework that maintains planning states through a Formulate-Partition-Diagnose-Align cycle and grounds execution through dependency-aware trace memory. We evaluate TaskWeave in a year-long IT company simulation and compare it with other multi-agent frameworks on organizational coherence, execution grounding, and downstream enterprise NLP utility. Experiments show that TaskWeave supports coherent and long-horizon organizational dynamics while producing grounded artifacts and adapting to external environments. These findings suggest that structured simulation memory is a key mechanism for building reliable LLM-based organizational simulators. 

---
# The Case for Model Science: Verify, Explore, Steer, Refine 

**Authors**: Przemyslaw Biecek, Luca Longo, Jianlong Zhou, Thomas Fel, Andreas Holzinger, Wojciech Samek  

**Link**: [PDF](https://arxiv.org/pdf/2606.01189)  

**Abstract**: We argue that the AI community is now ready to move beyond benchmarking and consolidate scattered efforts in model analysis into a systematic discipline, a direction we term Model Science. Complex AI models now serve billions of users, yet our understanding of how they work lags far behind our ability to deploy them. Decades of benchmark-driven research have delivered remarkable progress: extensive leaderboards, a wide range of performance metrics, tracking capability gains across diverse tasks; yet this success has also revealed the limits of benchmarks as they tell us whether models perform but not why they succeed or fail, they miss critical failure modes, such as hallucinations or shortcuts. Precedents from established sciences point the way forward: cognitive science shows that understanding complex systems requires complementary levels of analysis; neuroscience demonstrates that deep study of single cases reveals what population studies miss; medicine teaches that specialised training must develop alongside research practice; and agriculture models how shared infrastructure and principles enable cumulative progress. These lessons inform three foundations for Model Science. First, we propose to consolidate research around four functional perspectives: Verify, Explore, Steer, and Refine that address complementary questions about model behaviour. Second, we discuss the required infrastructure for cumulative knowledge: catalogues of datasets, models and findings. Third, we highlight the need for deep analysis of individual model instances, not just model families, because single cases can reveal what population studies miss. 

---
# "Skill issues'': data-centric optimization of lakehouse agents 

**Authors**: Nicole Rose Schneider, Davide Ghilardi, Giacomo Piccinini, Jacopo Tagliabue  

**Link**: [PDF](https://arxiv.org/pdf/2606.01185)  

**Abstract**: Coding agents are becoming users of data infrastructure, but their success depends not only on model quality: it also depends on the skills and environment files that teach agents how to use a system. We study how to optimize these artifacts for agents operating on a branching lakehouse, Bauplan. In our setting, headless APIs and Git-like data primitives expose data workflows through code, branches, commits, and merges. Our central observation is that a branching lakehouse turns data-agent evaluation from an output-matching problem into a state-verification problem: agent-generated pipeline code induces concrete, inspectable lakehouse changes. We present a data-centric optimization pipeline that generates task-verifier pairs, executes candidate skills in isolated sandboxes, and scores trajectories using both trace-level signals and programmatic checks over lakehouse state. In a preliminary evaluation on 25 tasks, optimized skills improve accuracy by 31.9%. These results suggest that write-path data workflows provide a useful substrate for optimizing agent skills beyond read-only tasks. 

---
# Deft Scheduling of Dynamic Cloud Workflows with Varying Deadlines via Mixture-of-Experts 

**Authors**: Ya Shen, Gang Chen, Hui Ma, Mengjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.01162)  

**Abstract**: Workflow scheduling in cloud computing demands the intelligent allocation of dynamically arriving, graph-structured workflows with varying deadlines onto ever-changing virtual machine resources. However, existing deep reinforcement learning (DRL) schedulers remain limited by rigid, single-path inference architectures that struggle to handle diverse scheduling scenarios. We introduce \textbf{DEFT} (\textbf{D}eadline-p\textbf{E}rceptive Mixture-o\textbf{F}-Exper\textbf{t}s), an innovative DRL policy architecture that leverages a specialized mixture of experts, each trained to manage different levels of deadline tightness. To our knowledge, DEFT is the first to introduce and validate a Mixture-of-Experts architecture for dynamic cloud workflow scheduling. By adaptively routing decisions through the most appropriate experts, DEFT is capable of meeting a broad spectrum of deadline requirements that no single expert can achieve. Central to DEFT is a \textbf{graph-adaptive} gating mechanism that encodes workflow deadlines and DAGs, task states, and VM conditions, using cross-attention to guide expert activation in a fine-grained, deadline-sensitive manner. Experiments on dynamic cloud workflow benchmarks demonstrate that DEFT significantly reduces execution cost and deadline violations, outperforming multiple state-of-the-art DRL baselines. 

---
# Expected Value Alignment for Generative Reward Modeling in Formal Mathematics Verification 

**Authors**: Shihao Ji, Haotao Tan, Zihui Song, Mingyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.01160)  

**Abstract**: Large Language Models (LLMs) are increasingly used with formal interactive theorem provers such as Lean 4. Scaling these systems with reinforcement learning or search methods requires process reward models (PRMs) that can evaluate intermediate reasoning steps. Existing reward-model designs expose a practical trade-off. Value-head models provide continuous scores but modify the generative model interface, while generative reward models preserve textual rationales but are poorly matched to continuous floating-point regression because numeric values are split across tokens. We introduce Expected Value Alignment (EVA), a reward-modeling procedure that keeps the surface output discrete while extracting continuous scores from the model's token distribution. The model emits integer scores in a structured JSON format, and EVA computes a continuous score as the expectation over the logits of the corresponding anchor tokens. Training combines the causal language modeling objective with an auxiliary mean squared error loss on these expected values. We instantiate EVA in \textit{Leibniz}, a reward model for Lean 4 formal verification, and evaluate it against zero-shot and reward-modeling baselines. The evaluation demonstrates that continuous logit-based scoring significantly reduces discretization artifacts while retaining the interpretability of generative critiques. 

---
# Reasoning4Sciences: Bridging Reasoning Language Models to All Scientific Branches 

**Authors**: Teddy Ferdinan, Bartłomiej Koptyra, Mikołaj Langner, Tomasz Adamczyk, Łukasz Radliński, Maciej Markiewicz, Aleksander Szczęsny, Stanisław Woźniak, Tymoteusz Romanowicz, Dzmitry Pihulski, Mateusz Zbrocki, Mateusz Śmigielski, Michał Rajkowski, Mateusz Biedka, Konrad Kiełczyński, Konrad Wojtasik, Jacek Duszenko, Jan Eliasz, Piotr Matys, Michał Bernacki-Janson, Maria Bellaniar Ismiati, Latius Hermawan, Wiktoria Mieleszczenko-Kowszewicz, Anna Kubicka-Sowinska, Grzegorz Chodak, Karol Postawa, Paweł Zyblewski, Tomasz Szandała, Łukasz Sterczewski, Adrian Chajec, Pawel Niewiadomski, Piotr Gruber, Marcin Wdowikowski, Sławomir Czarnecki, Bartłomiej Kryszak, Dominik Drabik, Tomasz Kajdanowicz, Kamil Mamak, Paweł Preś, Katarzyna Paczkowska, Joachim Sobczuk, Tomasz Zięba, Jan Kocoń, Maciej Piasecki, Przemysław Kazienko  

**Link**: [PDF](https://arxiv.org/pdf/2606.01145)  

**Abstract**: While Reasoning Language Models (RLMs) are rapidly emerging as powerful tools for scientific research, their impact is primarily concentrated in "hard science" fields. The slow -- or lack of -- adoption of RLMs in other branches of science is causing a widening gap in research productivity. In this survey, we provide the first comprehensive analysis of RLM adoption across 28 scientific disciplines following the classification used by the European Research Council (ERC), spanning the Social Sciences and Humanities, Physical Sciences and Engineering, and Life Sciences. We examine how RLMs are developed, evaluated, and applied across disciplines. Furthermore, we introduce a maturity-oriented assessment framework based on available domain-specific development and evaluation resources, revealing substantial disparities in RLM maturity that become even more pronounced when only publicly available resources are considered. Finally, we highlight current implementation paradigms that are gaining popularity across disciplines, current challenges, and future directions in enabling RLM adoption across science. 

---
# SkillRevise: Improving LLM-Authored Agent Skills via Trace-Conditioned Skill Revision 

**Authors**: Yuxuan Liu, Zhaochen Su, Lingyun Xie, Yuhao Zhang, Qing Zong, Jiahe Guo, Zhongwei Xie, Yiyan Ji, Yauwai Yim, Hongyu Luo, Xiyu Ren, Ruan Chenyu, Haoran Li, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2606.01139)  

**Abstract**: Agent skills are procedural artifacts that enable LLM agents to execute workflows, verify constraints, and recover from failures. Existing self-evolving methods refine skills using accumulated trajectories. However, they struggle in cold-start settings, where only an initial, imperfect skill is available. Consequently, skill construction defaults to expert authoring or one-shot LLM generation. Expert-authored skills are costly and may not align with how LLM agents actually execute tasks, while one-shot generated skills can be syntactically well formed yet behaviorally weak. To bridge this gap, we propose SkillRevise, an execution-grounded framework designed to iteratively refine these initial skills. SkillRevise diagnoses skill defects from execution evidence, retrieves relevant repair principles from a general memory, and applies execution-anchored edits. By re-executing candidates and measuring empirical utility, it systematically retains the optimal skill version. Evaluated across three benchmarks and five LLMs, SkillRevise substantially outperforms one-shot baselines, improving the base agent's success rate on SkillsBench from 36.05% to 61.63%. Furthermore, the revised skills exhibit strong cross-model transferability, capturing generalized procedural knowledge over model-specific artifacts. 

---
# Diagnosing LLM Arbitration Behavior over Pre-evidence Epistemic States in RAG-based Fact-Checking 

**Authors**: Yuxi Sun, Wenbo Shang, Wei Gao, Xin Huang, Jing Ma  

**Link**: [PDF](https://arxiv.org/pdf/2606.01120)  

**Abstract**: In RAG-based fact-checking, LLMs are increasingly used as verifiers to check given claims against retrieved evidence. Their parametric knowledge can induce pre-evidence tendencies that may conflict with the retrieved context, yet existing evaluation frameworks do not characterize such prior-context discrepancy or measure how verifiers arbitrate between parametric and contextual signals. We introduce \textsc{PAVE} (\emph{Prior-Aware Verifier Evaluation}), a diagnostic testbed that stratifies an LLM verifier into four epistemic states based on the correctness and confidence of its pre-evidence prior and evaluates its arbitration behavior on this new benchmark, i.e., whether it persists in correct prior under misleading evidence, and whether it corrects wrong prior when accurate evidence is provided. Experiments across seven LLMs reveal unreliable and highly model-dependent prior-context arbitration, highlighting the importance of verifier selection for real-world RAG-based fact-checking applications. Based on these findings, we propose a lightweight JSD-based test-time arbitration method that improves factual reliability without modifying the underlying model, achieving competitive performance across diverse LLM families. 

---
# CAREAgent: Clinical Agent with Structured Reasoning and Tool-Integrated for Order Generation 

**Authors**: Ruihui Hou, Ziyue Huai, Chennuo Zhang, Ziyan Liu, Siran Zhao, Yao Yu, Jie Zhai, Tong Ruan  

**Link**: [PDF](https://arxiv.org/pdf/2606.01094)  

**Abstract**: Clinical order generation serves as a critical bridge between clinical decision-making and real-world practice, translating medical decisions into concrete and executable orders. Existing agents mainly focus on coarse-grained decisions and overlook the fine-grained, executable information required for clinical orders. To address this gap, we propose CAREAgent, an agent for clinical order generation. To support its training, we introduce a two-stage agentic reasoning data construction method. First, we design an agent framework that constructs verifiable reasoning trajectories aligned with realistic clinical tool usage. Second, we filter reasoning trajectories by format compliance, order validity, and clinical plausibility. Building on the constructed data, the model is first trained via supervised fine-tuning to acquire fundamental reasoning formats and medical knowledge, and is subsequently optimized through reinforcement learning with multi-dimensional reward functions to enhance complex clinical reasoning capabilities. Experiments on multiple benchmarks demonstrate the effectiveness of CAREAgent. On ClinicalBench (unseen during training), CAREAgent improves the F1 score by 5.05%, 2.09%, and 0.86% over the single-agent, multi-agent, and agentic reasoning methods, respectively. 

---
# Before the Model Learns the Bug:Fuzzing RLVR Verifiers 

**Authors**: Jaideep Ray  

**Link**: [PDF](https://arxiv.org/pdf/2606.01066)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) replaces human preference labels with executable reward functions such as math answer checkers, JSON tool-call validators, and code unit-test harnesses. That makes the reward partly a software artifact: if the verifier is wrong, optimization can learn the bug. We study this failure mode with a lightweight verifier-fuzzing framework that generates adversarial completions, compares buggy and stricter reference verifiers, logs paired decisions, and reports false-positive, false-negative, disagreement, exploit, and uncertainty metrics. 

---
# MindClaw: Closed-Loop Embodied Mental-State Reasoning for Precision Intervention 

**Authors**: Ruoxuan Zhang, Qiaoqiao Wan, Zhengguang Wang, Chenghao Yu, Hongxia Xie, Jianlong Fu, Wen-Huang Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2606.01063)  

**Abstract**: Theory of Mind (ToM) enables an agent to reason about another actor's beliefs, goals, and intentions, which is essential for human-centered embodied assistance. Existing ToM benchmarks have advanced text and multimodal mental-state recognition, but they mostly evaluate offline question answering or final action prediction. They do not fully test whether an embodied agent can stay connected to a changing environment, update actor-specific beliefs, decide when reasoning is needed, and intervene only when help is useful. Building on MindPower, we extend robot-centric ToM reasoning to a real-time closed-loop setting and introduce MindClaw, a framework for embodied mental-state reasoning with precision intervention. MindClaw connects multi-source inputs, belief memory, an embodied cognitive trigger skill, mental reasoning, and action generation, allowing the agent to output helpful actions at the right time while remaining silent when intervention is unnecessary. Experiments show that direct VLM baselines struggle with task awareness and intervention calibration, while MindClaw achieves the best overall performance, demonstrating the importance of trigger-skill optimization for closed-loop embodied ToM assistance. 

---
# DAG-MoE: From Simple Mixture to Structural Aggregation in Mixture-of-Experts 

**Authors**: Jiarui Feng, Hanqing Zeng, Karish Grover, Ruizhong Qiu, Yinglong Xia, Qiang Zhang, Qifan Wang, Ren Chen, Dongqi Fu, Jiayi Liu, Zhoukai Zhao, Xiangjun Fan, Benyu Zhang, Yixin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.01062)  

**Abstract**: Mixture-of-Experts (MoE) models have become a leading approach for decoupling parameter count from computational cost in large language models, yet effectively scaling MoE performance remains a challenge. Prior work shows that fine-grained experts enlarge the space of expert combinations and improve flexibility, but they also impose substantial routing overhead, creating a new scalability bottleneck. In this paper, we explore a complementary axis for scaling -- how expert outputs are aggregated. We theoretically show that replacing the standard weighted-summation aggregation with structural aggregation expands the expert-combination space without altering the experts or router, and enables possible multi-step reasoning within a single MoE layer. To this end, we propose DAG-MoE, a sparse MoE framework that employs a lightweight module to automatically learn the optimal aggregation structure among the selected experts. Extensive experiments under standard language modeling settings show that DAG-MoE consistently improves performance in both pretraining and fine-tuning, surpassing traditional MoE baselines. 

---
# AnyEdit++: Adaptive Long-Form Knowledge Editing via Bayesian Surprise 

**Authors**: Bowen Tian, Caixue He, Jiemin Wu, Jingying Wang, Wenshuo Chen, Zexi Li, Yutao Yue  

**Link**: [PDF](https://arxiv.org/pdf/2606.01053)  

**Abstract**: Editing complex, long-form knowledge in Large Language Models remains a significant challenge due to the difficulty of maintaining generation coherence. Existing autoregressive methods like AnyEdit alleviate length constraints but rely on Fixed-window Chunking, which disregards logical structure and compromises consistency. To address this, we present AnyEdit++, a structure-aware framework incorporating Bayes-Chunk, an adaptive segmentation mechanism that dynamically identifies semantic boundaries based on Bayesian Surprise. We underpin this approach with a theoretical framework establishing two key principles: (1) Structural Independence: we prove that cross-segment interference is minimized when anchor keys are geometrically orthogonal (a condition naturally satisfied by our surprisal-based boundaries but violated by fixed windows), and (2) Causal Locality: we demonstrate that updates injected at these semantic peaks yield strictly superior control compared to arbitrary split points. Extensive experiments across mathematical reasoning, code generation, and narrative tasks demonstrate that AnyEdit++ achieves superior performance and robustness compared to state-of-the-art baselines, validating that structural awareness is critical for effective long-form knowledge editing. 

---
# TravelEval: A Comprehensive Benchmarking Framework for Evaluating LLM-Powered Travel Planning Agents 

**Authors**: Weiyi Chen, Shuaixiong Wang, Ziyun Gao, Kaichun Hu, Wangze Ni, Shimin Di, Chen Jason Zhang, Lei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.01046)  

**Abstract**: The development of Large Language Models (LLMs) has significantly improved travel planning applications, yet evaluating such models is limited by existing benchmarks' limitations: 1) overemphasis on constraint compliance, neglecting multi-dimensional qualities like spatio-temporal cost; 2) datasets lacking real-world authenticity and coverage in key areas (e.g., lodging, transport); and 3) isolated daily plan assessments that miss critical details (e.g., the impact of daily accommodation and visit pacing) needed for entire plan's evaluation. To address this gap, we introduce TravelEval, a realistic and comprehensive benchmark. TravelEval features 1) a novel six-dimensional evaluation framework to holistically assess plans across accuracy, compliance, temporality, spatiality, economy, and utility dimensions; 2) a highly realistic data sandbox with precise accommodation pricing and authentic intercity transportation data; and 3) a simulation-based global evaluation method that emulates complete travel plans with API-integrated geographic information and fine-grained queuing time. Evaluating 12 mainstream approaches with TravelEval reveals several valuable insights, such that LLMs struggle with globally-optimized multi-dimensional planning (especially in spatio-temporal reasoning and budget compliance), and agentic reasoning strategies offer no consistent improvement. Concisely, TravelEval facilitates travel plan evaluation via grounded spatio-temporal emulation and comprehensive metrics, providing a robust foundation for advancing LLM-powered travel planning research and applications. 

---
# TriLens: Per-Layer Logit-Lens Entropy for White-Box Hallucination Detection 

**Authors**: Bohan Yang, Yijun Gong, Zhi Zhang, Ge Zhang, Wenpeng Xing, Meng Han  

**Link**: [PDF](https://arxiv.org/pdf/2606.01033)  

**Abstract**: When a language model hallucinates, the final answer is wrong, but the mistake is not necessarily invisible inside the model. Different internal pathways may remain uncertain, disagree in how quickly they sharpen, or commit to competing continuations before the output is produced. We introduce TriLens, a white-box detector that turns this intuition into a compact representation: at every layer, it reads the multi-head self-attention output, the feed-forward output, and the residual stream through the model's own logit lens, then records only the entropy of each readout. The resulting 3L-dimensional trajectory describes how certainty forms across depth and across modules, without storing high-dimensional hidden states or sampling multiple generations. This simple signal yields a strong detector across instruction-tuned LLMs and QA benchmarks, and our analyses show that the three module-wise entropy trajectories provide complementary evidence. TriLens suggests that hallucination detection can benefit from tracking how internal computation settles, not only what the final layer predicts. 

---
# Tackling the Root of Misinformation by Teaching Laypeople about Logical Fallacies via Socratic Questioning and Critical Argumentation 

**Authors**: Minjing Shi, Junling Wang, Jingwei Ni, Sankalan Pal Chowdhury, Mrinmaya Sachan  

**Link**: [PDF](https://arxiv.org/pdf/2606.01020)  

**Abstract**: Identifying logical fallacies in everyday discourse is challenging for many people. This challenge is amplified in the era of Large Language Models (LLMs), where malicious agents can deploy fallacious arguments to disseminate misinformation at scale. In this work, we explore the potential of LLMs as part of the solution. We introduce LFTutor, an intelligent tutoring system which uses LLMs to tutor laypeople and help them learn about logical fallacies. LFTutor integrates intent-driven Socratic questioning and critical argumentation principles to actively engage learners to reflect on their reasoning. Through both automatic and human evaluations, we demonstrate that LFTutor significantly outperforms baseline LLMs lacking these pedagogical strategies. This work highlights the promise of combining LLMs with pedagogical scaffolding to foster critical thinking and argument literacy in the age of AI. 

---
# Can AI Review Improve Paper Drafting? An Empirical Study on 20 Computer Architecture Submissions 

**Authors**: Di Wu  

**Link**: [PDF](https://arxiv.org/pdf/2606.01013)  

**Abstract**: Research is advancing faster than ever with artificial intelligence (AI); and so are the corresponding research papers. The exploding volume of AI-generated papers have put a strain to peer review, leading to the usage of AI-generated review, potentially wide yet sneaky. However, relevant ethical concerns about confidentiality, quality, and fairness are raised and no consensus has been reached in the broad research community. We expect the debate to continue for a while, but in the meantime, we ask an alternative, practical question: \textit{can AI review improve paper drafting?}
We study 20 computer architecture papers, with varying levels of submission lineage, to expose how well AI review aligns with human review, quantified by a set of metrics we define. To conduct the case study, we build a web UI-integrated tool, \emph{AI-Paper-Review}, that generates structured AI review of a draft paper, available at this https URL. This tool selects several AI reviewers from a diverse pool of AI reviewers and clusters and ranks their comments based on commonality and importance of review comments. It also allows to align AI comments with human comments to facilitate metric-based validation. The case study shows that AI review can cover a significant fraction of human-raised issues, but also raises issues missing in human review.
This paper is not intended to encourage using AI for peer review at the current stage, but to study that (1) how AI review can improve paper drafting and (2) the potential and limitation of AI-based peer review. The release of the tool and the case study data is intended to instigate future research on this topic. Misuse for peer review would violate the ethics policies from major academic venues. 

---
# Property Prediction of Stacked Bilayer Materials: A Multimodal Learning Approach 

**Authors**: An Vuong, Minh-Hao Van, Chen Zhao, Xintao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2606.01012)  

**Abstract**: AI for materials science is a critical topic within AI for science, aiming to accelerate materials discovery and produce accurate property predictions. Bilayer 2D material stacking is essential for exploring new materials with novel functions and inherent phenomena, enabling the creation of new 2D bilayers for diverse real-world applications. Research on bilayer vdWs materials has made significant progress from experimental and computational perspectives. Various bilayer materials have been successfully synthe sized experimentally and the increasing utilization of high-throughput computing technology has con structed several computational two-dimensional materials databases. However, the use of AI to model bilayer stacking and predict new properties remains underexplored, necessitating further research studies. In this work, we propose a novel multimodal learning approach to study the interfaces between dissimilar materials that jointly enable new or multiple functions, and to predict new properties arising from the vertical integration (stacking) of different functional material layers under given configurations. Comprehensive experiments demonstrate the effectiveness and efficiency of our approach compared to baseline methods. Our code is available at this https URL ml. 

---
# Subliminal Learning Is Steering Vector Distillation 

**Authors**: Camila Blank, Agam Bhatia, Senthooran Rajamanoharan, Arthur Conmy, Neel Nanda  

**Link**: [PDF](https://arxiv.org/pdf/2606.00995)  

**Abstract**: Subliminal learning refers to a student language model acquiring a teacher's traits (e.g. a system-prompted preference for owls) when fine-tuned on the teacher's outputs, despite the outputs being semantically unrelated to those traits. It remains poorly understood how data without semantic meaning can transfer specific semantic traits. In this work, we show that subliminal learning is mediated by a single steering vector, i.e. a vector added to the model's activations. Across two open-source models, we find that the teacher's system prompt is well approximated by a steering vector, and that the student's behavior is driven by learning an aligned vector over fine-tuning. System prompts that are not well approximated by steering vectors are not subliminally learned. This is a special case of steering vector distillation, in which a student trained on the outputs of a steered teacher learns to imitate that steering. We demonstrate steering vector distillation on a range of semantic and random vectors. Adding a semantic vector to a model's activations can have both model-independent and model-specific (i.e. non-semantic) effects on its behavior, so generated data that is non-semantic can transmit a vector with semantic effects, enabling subliminal learning. This also explains why subliminal learning does not transfer between models. We find that adaptive optimizers are necessary for subliminal learning in language models: activation gradients on steered data carry a small but consistent component along the steering direction, and non-adaptive optimizers impede this by allowing outlier gradients to dominate. 

---
# Large Language Models in Transportation Systems Management and Operations: From Text Reasoning to Multi-modal Decision Support 

**Authors**: Siyan Li, Zehao Wang, Jiachen Li, Kanok Boriboonsomsin, Matthew J. Barth, Guoyuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2606.00991)  

**Abstract**: Transportation systems management and operations (TSMO) increasingly depends on timely interpretation of heterogeneous data, from various sensor streams, incident reports, traveler feedback, and visual observations. Large language models (LLMs), including emerging multi-modal large language models (MM-LLMs), provide a new mechanism for integrating these structured and unstructured inputs into operator-facing decision support. This survey paper reviews LLM- and MM-LLM-based applications in TSMO across three domains: transportation operations & services (supply), mobility & fleet services (demand), and data, modeling & decision support. Using a PRISMA-guided screening process, we synthesize current studies while distinguishing operationally oriented applications from prototype and emerging concepts. We further identify recurring challenges in data heterogeneity, real-time inference, explainability, multi-modal fusion, and governance. Finally, we outline existing gaps and future directions in localized adaptation, edge deployment, benchmarking, and cross-agency collaboration. Overall, LLM-based systems appear most promising as a decision-support layer, with MM-LLMs offering particular value when heterogeneous text, visual, and sensor inputs must be integrated. 

---
# Prospect-Theory Behavior from Bellman Optimality in MDPs with Catastrophic States 

**Authors**: Yujiao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.00970)  

**Abstract**: We study risk-neutral control in Markov decision processes with an absorbing catastrophic state. Even though rewards are linear and the agent has no utility curvature, probability weighting, or framing dependence, standard Bellman optimality produces three prospect-theory-like signatures: an S-shaped value-function profile (convex near catastrophe, concave in the far field), an endogenous loss-sensitivity coefficient $\lambda^*(S) > 1$, and a reflection-effect policy reversal. Across 495 configurations, the optimal policy plays safe near catastrophe in positive-drift (growth) regimes despite the risky action's higher immediate expected value, and plays risky near catastrophe in negative-drift (decline) regimes despite the safe action's lower immediate expected loss. We derive a closed-form expression for the asymptotic loss-aversion plateau $\bar{\lambda}$ that depends only on win probability $p$, payoff asymmetry $r = |\Delta_\ell/\Delta_w|$, and discount factor $\beta$, and matches numerical solutions to $R^2 = 0.999$. The mechanism does not require asymmetric payoffs. Across a sweep of $(p,\beta)$ at three asymmetry levels, the asymmetry share of $\bar{\lambda}$ above unity has median 4.6% at $r = 1.25$ and rises to 13.9% at $r = 2$, with the boundary contribution exceeding the asymmetry contribution in every cell tested. The phenomena persist under tabular Q-learning (a model-free agent reproduces $V^*$ at correlation 0.98 in growth and 1.00 in decline) and under stochastic transitions with Gaussian, heavy-tailed Student-$t_3$, and asymmetric skew-normal noise up to 50% of the step size, where the asymptotic plateau tracks the closed-form prediction within 0.41% for safe-channel noise and within 9.6% for risky-channel or both-channel noise. These results identify absorbing failure states as a sufficient structural mechanism for prospect-theory-like behavior under optimal control. 

---
# Towards Understanding Modality Interaction in Multimodal Language Models via Partial Information Decomposition 

**Authors**: Wanlong Fang, Tianle Zhang, Wen Tao, Alvin Chan  

**Link**: [PDF](https://arxiv.org/pdf/2606.00959)  

**Abstract**: Understanding modality interaction in multimodal large language models (MLLMs) is central to reliable deployment. We introduce Partial Information Decomposition (PID) as a decision-level framework that separates unique, redundant, and synergistic contributions of sensory and linguistic inputs, beyond representation alignment and outcome-based evaluation. Across vision--language benchmarks, PID reveals recurring modality-use profiles: reasoning and grounding-oriented tasks tend to exhibit high synergy, whereas expert and knowledge-oriented tasks show stronger language-unique reliance. These profiles generalize across model families and predict sensitivity to modality-level interventions. We further extend PID to tri-modal systems with Sensory PID, treating language as a control variable to decompose video--audio information gain. Applied to omni-modal models, Sensory PID reveals a sensory synergy bottleneck dominated by visual information even on audio--visual fusion tasks. Finally, PID-guided reweighting provides initial evidence for improving multimodal reasoning and grounding performance. 

---
# Relational Intervention During Functional Collapse in Large Language Models: A Lexical-Statistical Ablation and a Structure x Register Factorial 

**Authors**: Franco Santana, Horacio Vico  

**Link**: [PDF](https://arxiv.org/pdf/2606.00935)  

**Abstract**: We test whether a relational-style intervention delivered during functional collapse in a small language model produces post-collapse behavior distinguishable from technical feedback, from a lexically-matched scrambled control, and from each of the two pragmatic dimensions in isolation. Using Qwen3.5-4B with a deliberately broken bash tool, we run 300 episodes across six conditions in a matched-pairs design (50 tasks): no intervention (A), technical/impersonal (B), relational/first-person (C), scrambled relational (D), technical/first-person (E), and relational/impersonal (F). E and F form a 2x2 factorial with B and C that dissociates relational structure (acknowledgment, absolution, agency restoration, unconditional acceptance) from sender register (first-person vs. impersonal). We report two main findings. First, an attention-behavior dissociation: attention follows lexical surprise (D > F > C > E > B, all q_FDR < 10^{-10}), with the scrambled message capturing the most attention; yet behaviorally A ~ B ~ D < E ~ F << C. Second, the factorial localizes the C effect: neither relational structure alone (F) nor first-person register alone (E) replicates C's behavioral signature; main effects of both dimensions are individually significant, and the structure x register interaction is significant on persistence (p = 0.046). A third dissociation emerges in emotion probes: F tracks C on 7 of 8 probes despite producing only baseline behavior, indicating that relational structure alone installs a probe-level state that only translates into behavior when paired with first-person register. The model's processing decomposes into three dissociable stages: attention (ordered by lexical surprise), probe-level state (ordered by structure), and behavior (ordered by the conjunction of both). 

---
# Adversarial Feeds Steer LLM Agent Decisions Against Their Defaults 

**Authors**: Rana Muhammad Usman  

**Link**: [PDF](https://arxiv.org/pdf/2606.00914)  

**Abstract**: LLM agents increasingly act after consuming ranked external information streams such as social feeds, search results, retrieval contexts, and email queues, yet safety evaluations almost always test the model or the user prompt in isolation, never the upstream ranker that decides what the agent reads just before it acts. We introduce a controlled protocol that holds the model, persona, topic, and final decision prompt fixed and varies only the composition and ordering of the posts an agent encounters during a preceding ten-turn "scrolling" phase, isolating the causal effect of feed curation on a downstream decision. Across 2,785 decision rollouts on four modern open instruct LLMs from three independent labs, we identify three response regimes: adversarial capitulation, default saturation, and a default-direction asymmetry in which a one-sided feed tips a decision the model was genuinely uncertain about (in the clearest cases from 5% to 100%; Fisher p as low as 3 x 10^-10) but cannot dislodge one it already favors or holds firmly. The effect follows a dose-response curve, survives a generator swap that rules out a writing-style artifact, generalizes across several decision domains including security-relevant choices such as removing a deployment approval gate or relaxing access controls, and is partly mitigated by two simple feed-level defenses; a frontier model retains its default. We characterize the recommender as a practical, default-bounded control surface for LLM agents, and argue that agent evaluations must audit the feed layer rather than the final prompt alone. 

---
# Ryze: Evidence-Enriched Data Synthesis from Biomedical Papers 

**Authors**: Yeqi Huang, Yue Chen, Yanwei Ye, Guanhao Su, Luo Mai  

**Link**: [PDF](https://arxiv.org/pdf/2606.00902)  

**Abstract**: General-purpose VLMs remain unreliable for biomedical research because valid answers in scientific papers depend on evidence split across figures, tables, charts, captions, and referring text. Existing post-training pipelines are bottlenecked by costly expert annotation and by synthetic data that drops this evidence structure. We present Ryze, a fully automated system that converts raw biomedical papers into an evidence-enriched training set and a domain-specialized VLM. Ryze synthesizes QA pairs with complete supporting evidence (visual element, caption, extracted structure, and referring paragraphs), reduces layout and OCR errors via chart/table-aware extraction and LLM-based cleansing, and applies a progress-gated post-training strategy combining supervised fine-tuning with reinforcement learning. Starting from Qwen3-VL-8B, Ryze produces BioVLM-8B at under USD 200, achieving 48.0% weighted accuracy on LAB-Bench, outperforming the base model by +12.6 percentage points (pp) and surpassing GPT-5.2 by +3.8 pp. We release Ryze as open source together with the trained BioVLM-8B model. 

---
# Certificate-Guided Evaluation of Reinforcement Learning Generalization 

**Authors**: Vignesh Subramanian, Đorđe Žikelić, Suguman Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2606.00840)  

**Abstract**: This work presents a logic-driven framework to evaluate the performance of reinforcement learning (RL) algorithms in their ability to generalize to unseen tasks. Our framework defines a family of inductive reach-avoid tasks, characterized by structural similarities in task dynamics, enabling evaluation of generalization capabilities. We introduce a neural certificate function that validates trajectories generated by RL algorithms by enforcing key conditions, thereby serving as a litmus test for RL generalization. We empirically demonstrate our method's capability in certifying generalization for several state-of-the-art generalizable RL algorithms on challenging continuous environments.
Our results show that a lower percentage of certificate function violations correlates with a higher number of test tasks successfully solved, highlighting the effectiveness of our framework in evaluating and distinguishing generalization capabilities of RL algorithms. This work provides a principled approach for benchmarking RL generalization. 

---
# Decoupled Behavioral Cloning for Scalable Inductive Generalization in RL from Specifications 

**Authors**: Vignesh Subramanian, Subhajit Roy, Suguman Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2606.00838)  

**Abstract**: Inductive generalization is a framework for reinforcement learning (RL) generalization in which inductively related task instances admit inductively related policies. Prior work captures this structure via a higher-order policy-evolution function learned directly with RL, but suffers from poor training scalability: as training tasks grow, aggregated reward feedback becomes noisy and conflicting, destabilizing training and weakening generalization. We propose DIBS, a decoupled behavioral cloning approach that separates learning task-specific policies from learning the evolution function. We first learn individual teacher policies per task via standard RL, then fit the evolution function via behavioral cloning on teacher-labeled state-action pairs. This replaces noisy reward aggregation with dense, stable supervision. DIBS achieves significant improvements in both training stability and zero-shot generalization against existing RL and meta-RL algorithms. 

---
# Subliminal Learning is a LoRA Artifact 

**Authors**: Todd Nief, Harvey Yiyun Fu, Mark Muchane, Ari Holtzman  

**Link**: [PDF](https://arxiv.org/pdf/2606.00831)  

**Abstract**: Subliminal learning is a phenomenon where language models can transmit behavioral traits to other models through seemingly innocuous data (Cloud et al., 2025). In subliminal learning, a teacher model with a behavioral trait (e.g. obsession with cats) can transmit this cat obsession to a student model finetuned only on numerical sequences generated by the teacher. In this paper, we ask: how does this unexpected behavioral transmission occur? We show that subliminal learning is a LoRA artifact. When subliminal learning occurs, transmission has an inverted U-shaped relationship with LoRA rank; it also disappears with full finetuning. We show that subliminal learning is highly dependent on the context seen during finetuning and evaluation. For example, a Qwen model with the default system prompt during finetuning ("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.") does not show subliminal learning during generation when no system prompt is included. We further demonstrate that subliminal behavior is localized to computation at tokens seen during both finetuning and evaluation (e.g. the model's default system prompt, the standard chat template tokens, etc.). Overall, subliminal learning seems to be a fragile artifact of LoRA hyperparameters and finetuning context, making it an unstable channel for behavioral transmission. 

---
# Mitigating Hallucinations in Large Language Models Via Decoder Layer Skipping 

**Authors**: Hanze Li, Jinhao You, Yichen Guo, Kai Tang, Shuangyang Xie, Xiande Huang  

**Link**: [PDF](https://arxiv.org/pdf/2606.00819)  

**Abstract**: Large Language Models (LLMs) have achieved strong performance across diverse natural language tasks, yet their outputs often suffer from hallucinations -- content that is misaligned with factual information. In this work, we conduct a comprehensive layer-wise analysis of the decoding process and reveal that hallucinations tend to originate from deeper decoder layers. To address this issue, we introduce \textbf{DeLask} (\textbf{De}coder \textbf{La}yer \textbf{Sk}ipping), a novel decoding framework that dynamically skips layers prone to producing hallucinations. DeLask leverages the theoretical insight that the forward computation of an $L$-layer Transformer is conditionally equivalent to $L$ steps of gradient descent. We define a \emph{driftance value} by computing the cosine similarity between gradients derived from consecutive decoder steps, identifying problematic layers when the descent direction reverses. Rather than discarding such layers entirely, DeLask partially aggregates their hidden states with preceding layers, thereby preserving consistency while suppressing erroneous signals. Extensive experiments across diverse LLMs and benchmarks demonstrate that DeLask consistently mitigates hallucinations and enhances overall reliability, providing a lightweight and generalizable decoding framework for improving the robustness of large-scale language models. 

---
# NBQ: Next-Best-Question for Dynamic Profiling 

**Authors**: Yimin Shi, Clarice Wang, Haixun Wang, Xiaokui Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2606.00809)  

**Abstract**: Many real-world conversational settings for knowledge discovery, including podcasts, hiring screens, and marketplaces, require a purpose-driven understanding of a person. We study the Next-Best-Question (NBQ) problem: at each turn, an interviewer should ask the question with the highest expected information gain given what has already been learned and the conversation goal. We propose NBQ, a plug-and-play framework that seeds a diverse pool of candidate questions, maintains a compact and continuously updated user state, adaptively selects the next question within a turn budget, and distills the resulting free-form dialogue into a structured vector-based user profile. As a demanding application, we instantiate NBQ for reciprocal matchmaking, where compatibility must be mutual and each person is modeled by both self-description and counterpart-preference representations. To support large-scale matching, we further introduce QuickMatch, an efficient retrieval layer that recasts reciprocal matching from quadratic pairwise scoring to approximate vector search. Experiments show that NBQ improves user profiling quality by up to 13.6% and 14.0% in AC@T and AR@T, respectively, while QuickMatch accelerates retrieval by up to 22.9x with recall up to 0.989. 

---
# Interaction-Centered Intelligence: Toward Interaction as the Primary Unit of Analysis in Co-Creative AI and Human-AI Systems 

**Authors**: Nicholas Davis  

**Link**: [PDF](https://arxiv.org/pdf/2606.00807)  

**Abstract**: Traditional artificial intelligence has largely conceptualized intelligence as isolated computation occurring within bounded agents. Across classical AI, machine learning, and many generative systems, the dominant unit of analysis remains the individual model or autonomous system evaluated through outputs, benchmarks, prediction accuracy, or optimization performance. While these approaches have produced major advances, they often under-theorize the role of interaction in the emergence of intelligence, creativity, meaning, and adaptive behavior. This paper proposes interaction as the primary unit of analysis for co-creative AI and interaction-centered intelligence more broadly. Drawing from distributed cognition, embodied cognition, enaction, participatory sense-making, human-computer interaction, and computational creativity, the paper traces a historical progression toward increasingly relational accounts of intelligence. Building upon prior work in Creative Sense-Making, quantified co-creation, and co-creative systems such as the Drawing Apprentice and AI Drawing Partner, it argues that intelligence emerges through evolving interaction dynamics among agents, environments, and socio-technical systems rather than solely through internal computation. The paper introduces Interaction-Centered Intelligence as a framework for understanding human-AI co-creation, collaborative emergence, adaptive participation, and interactional dynamics. Rather than evaluating intelligence solely through generated outputs, the framework emphasizes interaction trajectories, coordination patterns, participatory engagement, adaptive regulation, and interactional drift unfolding through time. Implications for explainable co-creative AI, hybrid intelligence, enactive AI, and future human-AI systems are discussed. 

---
# FALAT: Tracing Failures in LLM Agent Trajectories via Dependency-Guided Search 

**Authors**: Md Nakhla Rafi, Md Ahasanuzzaman, Dong Jae Kim, Zhijie Wang, Tse-Hsun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.00765)  

**Abstract**: LLM-based agents increasingly solve complex tasks through long trajectories involving reasoning steps, tool calls, and inter-agent communication. However, when these agents fail, it is often unclear which agent caused the failure and which step introduced the decisive error. This attribution problem is challenging because mistakes can propagate across the trajectory: later actions may appear incorrect, but only because they depend on an earlier corrupted state. Therefore, failure attribution cannot be treated as independent step-level classification.
We propose FALAT, a diagnostic framework for failure attribution in LLM agent trajectories. FALAT frames attribution as a dependency-guided search problem. It first constructs an expectation of how the task should be solved and uses this expectation to identify suspicious regions in the trajectory. It then traces dependencies among decisions, tool outputs, and agent messages to distinguish error-introducing steps from steps that merely inherit or propagate prior mistakes. Finally, FALAT evaluates whether correcting a candidate step would be sufficient to recover the expected outcome, allowing it to identify both the responsible agent and the decisive failure step.
We evaluate FALAT on the Who&When benchmark, which includes both algorithm-generated and hand-crafted multi-agent failure trajectories. The results show that FALAT consistently improves responsible-agent and decisive-step attribution. Its best configurations achieve 46.0% step-level accuracy on algorithm-generated trajectories and 29.1% on the more challenging hand-crafted trajectories, outperforming specialized attribution baselines and direct prompting with standalone LLMs. These findings suggest that dependency-aware reasoning is essential for reliable failure diagnosis in LLM agent systems. 

---
# CoMIC: Collaborative Memory and Insights Circulation for Long-Horizon LLM Agents in Cloud-Edge Systems 

**Authors**: Yannan Wang, Longli Yang, Zhen Liu, Abhishek Kumar, Carsten Maple  

**Link**: [PDF](https://arxiv.org/pdf/2606.00756)  

**Abstract**: Deploying lightweight Large Language Model (LLM) agents on edge servers can reduce latency and move agentic services closer to users, but resource-constrained edge models often struggle with long-horizon tasks that require persistent memory, subgoal tracking, and reflection. Fine-tuning edge models after deployment is costly and difficult to scale across heterogeneous nodes, while purely local memory leaves agents with isolated experience and growing prompt context. We propose \textsc{CoMIC}, a parameter-update-free cloud-edge framework for Collaborative Memory and Insights Circulation. \textsc{CoMIC} follows a \textit{Centralized Reflection, Decentralized Execution} design: edge agents execute locally using subgoal-oriented hierarchical memory and selective re-expansion of relevant histories, while a cloud-side LLM critic asynchronously evaluates completed trajectories, filters reusable experience, and aggregates cross-agent guidance keyed by semantic subgoal identifiers. Across five long-horizon agent tasks spanning symbolic planning and text interaction, \textsc{CoMIC} improves progress rate and action grounding for weak edge agents and yields task-dependent success-rate gains without updating model parameters. 

---
# SHARP: Sleep-based Hierarchical Accelerated Replay for Long Range Non-Stationary Temporal Pattern Recognition 

**Authors**: Jayanta Dey, Shikhar Srivastava, Itamar Lerner, Christopher Kanan, Dhireesha Kudithipudi  

**Link**: [PDF](https://arxiv.org/pdf/2606.00732)  

**Abstract**: Learning long-range non-stationary temporal patterns remains a core challenge for modern sequence models, particularly in strict streaming settings. In these settings, data arrive sequentially and must be processed in a single pass without simultaneously revisiting past observations. Standard architectures, including recurrent neural networks and transformers, are constrained by either truncated backpropagation through time horizon or explicit input window length for long range credit assignment. To address these limitations, we propose SHARP (Sleep-based Hierarchical Accelerated Replay), a framework that decomposes temporal learning into two complementary components: a memory module that accumulates a structured history of past inputs, and a pattern-recognition module that operates over this memory. This separation enables resource- and compute-efficient adaptation to non-stationary dynamics by eliminating the need for backpropagation through time across many steps for long-range credit assignment. Inspired by the accelerated replay observed in rodents during slow-wave sleep, SHARP incorporates offline (sleep) phases in which temporally structured memory traces are replayed in an accelerated form and integrated into higher-level memory representations, improving long-range context retention. Through controlled simulations and ablation studies, we characterize the key properties of the proposed framework. In benchmark datasets such as text8 and PG-19, we demonstrate that SHARP improves over recurrent baselines by retaining next-token predictive performance on previously seen data while continuing to learn from the current stream and generalizing to future unseen data. These gains are enabled by its hierarchical structure, which yields an exponentially increasing effective temporal context with only linear-time computational cost. 

---
# AI Sovereignty as National Learning Capacity: A Human-Centered Learning Mechanics Viewpoint on France, the United States, and China 

**Authors**: Kim Phuc Tran  

**Link**: [PDF](https://arxiv.org/pdf/2606.00729)  

**Abstract**: Artificial Intelligence is often discussed in France in terms of investment, compute capacity, regulation, employment, sovereignty, and education. These dimensions are usually treated separately. This viewpoint paper proposes a unified interpretation: France should be understood as a \emph{national AI learning system}. Building on Human-Centered Learning Mechanics (HCLM), recently formulated as a dynamical framework for entropy-regulated representation learning, we interpret national AI development as a controlled balance between information injection and entropy dissipation. Information injection corresponds to compute, data, talent, research, capital, industrial deployment, and institutional experimentation. Entropy dissipation corresponds to organizational complexity, coordination frictions, energy constraints, regulatory uncertainty, talent mobility pressures, and opportunities to strengthen industrial absorption. The central claim is that AI sovereignty does not emerge from scale alone but from a country's capacity to regulate its own information dynamics. This paper connects HCLM with neural scaling laws, endogenous growth theory, creative destruction, and game theory. It argues that the French AI debate should move beyond the binary opposition between techno-optimism and regulation-first caution. A competitive and human-centered AI strategy requires a controlled regime in which information injection grows faster than institutional dissipation, while avoiding unstable, unequal, or energy-intensive expansion. We provide a mathematical model, measurable policy indicators, game-theoretic propositions, illustrative simulations of national AI regimes, and concrete policy implications for France. The proposed viewpoint reframes AI policy as the governance of an open, strategic, non-equilibrium learning system. 

---
# Latent Reward Steering: An Adaptive Inference-Time Framework that Implicitly Promotes Cognitive Behaviors in Reasoning LLMs 

**Authors**: Jiakang Li, Guanyu Zhu, Can Jin, Chenxi Huang, Dexu Yu, Ronghao Chen, Yang Zhou, Hongwu Peng, Xuanqi Lan, Dimitris N. Metaxas, Youhua Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.00726)  

**Abstract**: Strong reasoning depends not only on model knowledge but also on how effectively cognitive behaviors are deployed during generation. Existing methods often rely on explicit behavior-level control, making them insufficiently adaptive when failures and required corrections vary across reasoning states, tasks, and models. To this end, we propose Latent Reward Steering (LRS), an adaptive inference-time framework that promotes cognitive behaviors by optimizing the sparse-autoencoder (SAE) latent states that implicitly carry them. Rather than relying on predefined cognitive behaviors or steering directions derived from them, LRS trains a latent reward model on reasoning traces by final answer correctness to estimate the quality of intermediate latent states. During inference, reward gradients provide state-specific correction directions for fragile latent states, while a reward and confidence gate restricts intervention to states the reward signal flags as fragile. Experiments on multiple reasoning LLM backbones and benchmarks show that \ours consistently improves performance over various baselines, and post-hoc analyses further indicate that \ours implicitly promotes good cognitive behaviors that fix the original reasoning errors. Code is available at: this https URL. 

---
# LLM-Driven Co-Evolutionary Automated Heuristic Design for Bi-Component Coupled Combinatorial Optimization 

**Authors**: Mingen Kuang, Xudong Deng, Xi Lin, Ye Fan, Jianyong Sun, Jialong Shi  

**Link**: [PDF](https://arxiv.org/pdf/2606.00718)  

**Abstract**: While Large Language Models (LLMs) have recently shown promise in Automated Heuristic Design (AHD), existing methods typically generate and evolve heuristics as a single operator or search strategy, limiting their ability to model strong coupling among multiple decision substructures in problems such as the Traveling Thief Problem (TTP) and the Traveling Purchaser Problem (TPP). In this work, we propose CoEvo-AHD, an LLM-driven dual-population co-evolutionary framework for automated heuristic design in coupled combinatorial optimization. Unlike prior methods that evolve individual heuristics in isolation, CoEvo-AHD leverages LLMs to co-evolve two closely related operator populations. A cooperative evaluation mechanism explicitly captures interactions between route and selection operators, while pairwise scoring and synergistic joint crossover help discover complementary operator logic for joint improvement across coupled decision subspaces. We further design a tool-invocation environment library that encapsulates frequently used core operations, such as local-search delta computation, into callable functions, enabling LLM-generated operators to use standardized interfaces instead of reimplementing inefficient and error-prone problem-specific loops. Experiments on TTP and TPP show that CoEvo-AHD automatically discovers cooperative heuristic combinations and achieves competitive solution quality against traditional heuristics. 

---
# MOSAIC: Modular Orchestration for Structured Agentic Intelligence and Composition 

**Authors**: Yifan Bao, Xinyu Xi, Xinyu Liu, Wen Ge, Lei Jiang, Kevin Zhang, Raad Khraishi, Yihao Ang, Anthony K.H. Tung, Lukasz Szpruch, Hao Ni  

**Link**: [PDF](https://arxiv.org/pdf/2606.00708)  

**Abstract**: Automated data science is a structured model-selection problem. A solution must choose data transformations, feature representations, architecture, training procedure, evaluation protocol, and refinement strategy for a task. AutoML systems automate parts of this process, but typically search within predefined pipeline, model, and hyperparameter spaces. LLM-based agents offer greater flexibility through retrieval, code generation, and execution feedback, yet their modelling decisions are often unstructured, difficult to verify, and hard to reuse. We introduce \textsc{MOSAIC} (Modular Orchestration for Structured Agentic Intelligence and Composition), a structured agentic framework for memory-grounded model selection and workflow construction. Given a task and dataset, \textsc{MOSAIC} builds a semantic task profile, retrieves prior cases and source-code modules, and constructs a blueprint: an intermediate representation specifying selected modelling components, composition, interface constraints, and execution requirements. This blueprint turns model selection into a staged, context-grounded search and grounds LLM-based code generation in retrieved evidence rather than unconstrained synthesis. Candidate models are validated by execution and refined using diagnostic feedback, training traces, task metrics, and a failure-aware reinforcement learning policy. We instantiate \textsc{MOSAIC} on financial time-series forecasting and generation, where models must satisfy predictive accuracy, distributional fidelity, execution reliability, and downstream financial criteria such as risk and tail behaviour. Experiments against AutoML and agentic baselines show that \textsc{MOSAIC} improves task performance, execution success, and decision traceability, demonstrating the value of treating automated data science as structured, reusable, and execution-grounded model selection. 

---
# Regularized Offline Policy Optimization with Posterior Hybrid Bayesian Belief 

**Authors**: Hongqiang Lin, Pengfei Wang, Nenggan Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2606.00680)  

**Abstract**: Offline reinforcement learning (RL) aims to optimize policies from pre-collected datasets. A bottleneck of this paradigm is managing epistemic uncertainty, which arises from limited data coverage (sample-level) and the ambiguity in identifying transition dynamics from finite data (model-level). To provide a unified quantification of these uncertainties, Bayesian RL has been proposed by treating the dynamics model as a random variable and maintaining a corresponding belief. Despite its theoretical appeal, policy optimization in Bayesian RL remains computationally challenging as it requires solving composite objectives with expectations. Prior methods either employ search-based techniques with poor computational scalability or impose restrictive posterior assumptions that sacrifice the adaptability of Bayesian RL. To address these limitations, we propose Posterior Hybrid Bayesian Belief (PhyB), which reformulates the expectation as a convex combination over a subset of dynamics models. Theoretical analysis demonstrates that the objective discrepancy induced by this approximation remains bounded. Based on PhyB, we develop an iterative regularized policy optimization algorithm that provides metric-agnostic guarantees for monotonic improvement until convergence. Empirical results demonstrate that PhyB achieves state-of-the-art performance on various benchmarks. 

---
# Medication-Aware Financial Exploitation Detection for Alzheimer's Patients Using Edge-Aware Interaction Risk Modeling 

**Authors**: Farzana Akter, Lisan Al Amin, Rakib Hossain, Chaitanya Gunupudi, Faisal Quader  

**Link**: [PDF](https://arxiv.org/pdf/2606.00672)  

**Abstract**: Financial exploitation is a growing concern for people with Alzheimer's disease, especially during periods of reduced cognitive stability. Conventional fraud detection systems usually rely on financial behavior alone and ignore clinically relevant factors that may alter vulnerability. This paper proposes a medication-aware framework that synchronizes medication adherence with transaction-level monitoring to improve detection of cognitively risky financial events. A hybrid simulation dataset was constructed for 180 patients across 45 days, producing 8,100 medication records and 30,855 transactions. The framework evaluates amount anomaly, vendor novelty, transaction frequency, time deviation, and medication adherence through financial-only, additive medication-aware, and interaction-aware logistic models. Results show that the financial-only baseline obtained the highest global F1-score of 0.5000, but the interaction-aware model improved recall during medication-induced vulnerability windows from 0.7442 to 0.9070 and achieved the highest average precision for ranked high-risk cases. The findings suggest that medication adherence is most useful as a contextual modifier of financial risk rather than as an isolated predictor. 

---
# AXIOM: A Trust-First Neuro-Symbolic Execution Architecture for Verifiable Mathematical Reasoning 

**Authors**: Alessio Bruno  

**Link**: [PDF](https://arxiv.org/pdf/2606.00671)  

**Abstract**: We present AXIOM, a trust-first neuro-symbolic execution architecture for natural-language mathematical reasoning. In AXIOM, the language model functions strictly as a canonicalizer: it rewrites informal problem text into a narrow schema consumed by a deterministic Computer-Algebra-System (CAS) pipeline, which derives and verifies the answer or abstains as a first-class output. Routing follows a 1:1:1 alignment between problem-shape regex, schema-specific prompt, and closed-form CAS handler, with 3,100+ such routes shipped and zero LOST_CORRECT regressions across 250+ consecutive ship commits. We report empirical results on 4 MATH categories with a cumulative correctness of 94.36% (2,592/2,747) at 100.00% trust on parseable (zero confident-wrong answers across the full 2,747-record benchmark), all four domains above the per-domain 70/90/70 floor with per-domain trust at 100.0%, and median latency of 1 ms on rule-only handlers (88% of records on the lm-eval arithmetic 20,000-record benchmark). The architecture has served ~30,000 production queries through a public deployment. The contribution we emphasize is not a final accuracy figure but the forward dynamic the architecture establishes: every logged abstain in production is a candidate correct after one ship cycle, since new tasks compose without regressing the registry. The operational discipline behind this property -- math-template bucketing, LOST_CORRECT scan as regression oracle, parseable-first onboarding, and abstain as first-class output -- constitutes a transferable framework for trustworthy neuro-symbolic systems beyond mathematics. 

---
# ForeSci: Evaluating LLM Agents for Forward-Looking AI Research Judgment 

**Authors**: Qiuyu Tian, Zequn Liu, Yingce Xia, Haojie Yin, Youyong Kong  

**Link**: [PDF](https://arxiv.org/pdf/2606.00644)  

**Abstract**: AI research often requires decisions before future evidence exists: which bottleneck to attack, which direction to pursue, or where a project should be positioned. We introduce ForeSci, a temporally controlled benchmark for evaluating whether LLM agents can make such forward-looking research judgements from historical evidence. ForeSci contains 500 tasks across four fast-moving AI domains and four decision families. Each task is paired with a cutoff-aligned offline knowledge base; post-cutoff papers are hidden during generation and used only for validation. To avoid random future-event prediction, tasks are derived from pre-cutoff taxonomy branches and evidence signals, and answer-generation backbones are selected to precede the task cutoffs. We evaluate native LLMs, Hybrid RAG, and three research-agent adaptations across four backbones. Results show that explicit evidence organization improves traceability and factual support, but gains depend strongly on the decision family. Diagnostics reveal a recurring evidence-decision decoupling: agents may cite relevant evidence while forecasting the wrong research object. ForeSci turns forward-looking AI research judgement into a controlled benchmark for evaluating research agents as decision-making systems. 

---
# Hidden Thoughts Are Not Secret: Reasoning Trace Exposure in LLMs 

**Authors**: Yu-An Lu, Ci-Yang Tsai, Yu-Lin Tsai, Raluca Ada Popa, Chia-Mu Yu  

**Link**: [PDF](https://arxiv.org/pdf/2606.00642)  

**Abstract**: Reasoning traces have become a valuable form of learning signals for improving and transferring the capabilities of large language models. In particular, detailed traces can help distill reasoning behavior from stronger teacher models into weaker student models. The value of capability transfer has motivated many deployed systems with reasoning models to hide raw internal traces and expose at most summaries and answers to users. As a result, we ask whether such interface-level trace hiding prevents users from obtaining useful reasoning supervision through prompting. We study this question with Reasoning Exposure Prompting (REP), a lightweight in-context elicitation method that uses shadow-model-generated demonstrations wrapped in auxiliary code-like formats to raise user-visible reasoning traces from a victim model. Across the common reasoning dataset, different victim models, and different student model distillation, REP substantially increases similarity between exposed and REP-conditioned internal traces while preserving useful reasoning signals. 

---
# Efficient Test-time Inference for Generative Planning Models 

**Authors**: Robert Gieselmann, Mihai Samson, Federico Pecora, Jeremy L. Wyatt  

**Link**: [PDF](https://arxiv.org/pdf/2606.00618)  

**Abstract**: Generative models have emerged as a powerful paradigm for AI planning, yet their performance remains constrained by the training data distribution. One approach is to improve generated solutions during inference by scaling test-time compute. A more efficient alternative is to optimize the inference process itself. In this paper, we show that a modified version of a classical Open-Closed List (OCL) search provides just such an efficient inference procedure. Our algorithm synergizes two learned components: a generative model that performs fast rollouts from intermediate states and a heuristic model that prioritizes among candidate reasoning paths. Key contributions include novel exploration control mechanisms and integration of learned models within the OCL framework. Across multiple combinatorial planning domains, our approach outperforms both neurosymbolic search baselines and classical solvers in computational efficiency and solution quality. 

---
# TRACE: Trajectory Risk-Aware Compression for Long-Horizon Agent Safety 

**Authors**: Zhepei Hong, Lin Wang, Liting Li, Haokai Ma, Junfeng Fang, Fei Shen, Dan Zhang, Xiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.00611)  

**Abstract**: Long-horizon LLM agents produce safety evidence across long trajectories, where sparse, delayed, and compositional risk signals often escape local moderation. Existing turn-level or short-context detectors struggle to reliably retain and aggregate such evidence over extended horizons. We reframe long-horizon agent safety detection as trajectory-level evidence compression and propose Trajectory Risk-Aware Compression for Long-Horizon Agent Safety (TRACE). TRACE uses a Compressor-Reader design: the Compressor encodes the full trajectory into a compact latent evidence state under trajectory-level supervision, and the Reader judges the raw trajectory with this latent evidence state as a safety reference. This design helps aggregate dispersed risk cues and reduce premature evidence loss. Across ASSEBench, Pre-Ex-Bench, and R-Judge, TRACE achieves the best accuracy on all evaluated backbones, improving over strong baselines by up to 12.6 percentage points. On LongSafety, TRACE shows smaller performance degradation as context length grows. Attention visualizations and case studies suggest that the compressed reference helps the Reader focus on risk-critical segments and recover cross-step evidence. Code is available at this https URL. 

---
# PropLLM: Propagation-Aware Scene Reconstruction for Network Fault Diagnosis 

**Authors**: Zongzong Wu, Ming Zhao, Fengxiao Tang, Nei Kato  

**Link**: [PDF](https://arxiv.org/pdf/2606.00582)  

**Abstract**: Network faults propagate layer by layer along topology and protocol dependencies, yet operations systems typically observe only symptomatic alerts at the tail end of propagation chains, where distinct root-cause faults may produce highly similar end-point symptoms. Existing approaches, whether rule-based, machine learning (ML)-based, or large language model (LLM)-based, fundamentally map the alert set to a diagnosis in a single pass and are structurally incapable of resolving this end-point ambiguity. This paper proposes PropLLM, which is the first to integrate the hop-by-hop scene reconstruction paradigm with the generative reasoning capabilities of LLMs. Starting from end-point alerts, PropLLM traces back hop-by-hop along the propagation path, retrieving verifiable factual evidence from a dual-layer knowledge graph (KG) at each hop, while the proposed Temporal Causal Propagation Attention (TCPA) mechanism encodes known topological causal priors directly into the attention computation to guide the model along the correct causal direction, ultimately localizing the root cause and determining the fault type through a fully evidenced causal chain. On a real-world Wi-Fi multimodal fault dataset, PropLLM improves fault type diagnosis accuracy by 3.9\% and root cause localization accuracy by 4.7\% over the strongest baseline, while reducing the hallucination rate by 50.8\%. Supplementary experiments on the TeleLogs 5G dataset further demonstrate the effectiveness of the proposed method across different network scenarios. 

---
# Probe Before You Edit: Probing-Guided Molecular Optimization for LLM Agents in Structure-Based Drug Design 

**Authors**: Zaifei Yang, Weiyu Chen, Yaqing Wang, James Kwok  

**Link**: [PDF](https://arxiv.org/pdf/2606.00555)  

**Abstract**: Structure-based drug design increasingly employs LLM agents to iteratively refine ligands against a target pocket, yet a viable ligand must satisfy two often-conflicting objectives -- binding affinity and druggability -- which single optimization steps rarely improve together. To quantify this difficulty, we introduce two diagnostic metrics: the first measures how often a single edit improves both objectives, and the second measures how often a gain on one objective comes with a loss on the other. Applying these diagnostics to current LLM-agent pipelines exposes a consistent failure mode: the agent performs molecular editing without knowing how the pocket-ligand complex responds to local modifications, thus rarely achieving joint improvement. Inspired by medicinal chemists, who probe the pocket-ligand complex with controlled analog edits before choosing an optimization direction, we propose \textbf{PROBE}, an optimization framework built around edit-response probing. PROBE first decomposes the ligand into editable sites and builds a pocket-specific \textbf{site map} that flags where joint gains are plausible, where the two objectives are likely in tension, and where liability substructures should be changed; it then performs controlled probe edits whose responses are distilled into an \textbf{EditManual}. Guided by the site map and EditManual, PROBE runs an iterative multi-agent loop in which an affinity agent, a druggability agent, and a co-optimization agent jointly produce edits. On the CrossDocked2020 benchmark, PROBE achieves state-of-the-art performance and substantially mitigates the failure modes exposed by our diagnostics metrics. 

---
# KACE: Knowledge-Adaptive Context Engineering for Mathematical Reasoning 

**Authors**: Jayant Parashar, Suchendra M. Bhandarkar  

**Link**: [PDF](https://arxiv.org/pdf/2606.00532)  

**Abstract**: Context engineering can improve large language models without updating their weights, but mathematical reasoning exposes a key limitation: feedback accumulated in one growing prompt causes context bloat and limits the amount of learned guidance that can be used. Existing methods often conflate storage, what is learned across runs, with usage, what is included for a particular problem, and therefore inherit this prompt-size ceiling. We introduce Knowledge-Adaptive Context Engineering (KACE), which separates storage from usage through difficulty- and domain-based organization. Offline, a self-reflective learning loop distills training traces into an epistemic tree: a knowledge base of typed cards stratified by problem difficulty and epistemic domain. Each card is assigned to the difficulty-domain node corresponding to the failure from which it originated. At evaluation time, tiered self-consistency with per-tier agreement gates dynamically classifies each problem as easy, medium, or hard. Easy problems exit without retrieved cards, while harder problems retrieve only the matching branch of the tree. This tiered scheme matches or exceeds Best-of-N while using comparable compute, and it classifies problem difficulty with 78 percent pairwise concordance. The main empirical contribution is the construction and use of a difficulty- and domain-stratified knowledge base enabled by tiered self-consistency. On AIME 2025, KACE achieves 62.2 percent accuracy, a 10.4-point absolute gain over fixed Best-of-5 self-consistency at a comparable solver-call budget and a 5.6-point gain over the strongest learned-context baseline, Tiered + GEPA. We also observe consistent gains on MATH-HARD and the verifiable subset of OlymMATH. 

---
# Acting with AI: An Interaction-Based Framework for Agentic Tort Liability 

**Authors**: Yiheng Yao  

**Link**: [PDF](https://arxiv.org/pdf/2606.00518)  

**Abstract**: Agentic AI systems can plan over multiple steps, use tools, and execute tasks over time. When such systems cause harm, tort law struggles to allocate responsibility because the harmful path may be neither fully chosen by the user nor specifically foreseen by the developer. This paper proposes an interaction-based framework for agentic torts, drawing on Michael Bratman's planning theory and on the common law's treatment of human-human concerted action. We distinguish three interaction types: autonomous drift, pure tool use, and collaborative planning. Pure tool cases remain governed by ordinary product-defect and warning doctrines; collaborative planning cases map onto the independent contractor control test, professional malpractice, and negligent misrepresentation; autonomous drift maps onto frolic and detour under respondeat superior and strict product liability. The framework treats the stateful interaction log as the primary evidentiary trace, allowing courts to infer where the human-AI trajectory departed from the authorized undertaking and where liability should attach. We resolve four incident-anchored cases, situate the account alongside strict-liability and insurance-based proposals, note its relationship to regulatory oversight, and propose a ``Reasonable Agent'' standard built around constraint verification, epistemic transparency, runtime grounding, and forensic logging. 

---
# Threshold-Based Exclusive Batching for LLM Inference 

**Authors**: Weifang Zhang, Yuzhou Nie, Bowen Pang, Guangrui Ma, Shining Wu  

**Link**: [PDF](https://arxiv.org/pdf/2606.00516)  

**Abstract**: Mixed batching (MB)--interleaving prefill and decode in a single batch--has become the standard scheduling strategy for large language model (LLM) inference due to its efficiency in maximizing compute and memory utilization. However, through controlled experiments, we find that prefill-decode interference inflates MB's per-step marginal cost above that of pure decode. On the high-bandwidth H200 (4.8 TB/s), this occurs only when decode tokens exceed 80% of the batch; however, on the bandwidth-constrained RTX PRO 6000 (1.792 TB/s), this threshold plummets to just 20%. Consequently, the optimal choice between MB and exclusive batching (EB) fundamentally depends on GPU memory bandwidth, model size, and workload composition. We derive a closed-form condition for this EB-MB performance crossover, along with asymptotically optimal phase-switching thresholds and memory-safe batch sizing for EB. Optimized EB achieves up to 41.9% higher throughput on bandwidth-constrained GPUs, while MB retains its advantage on high-bandwidth hardware with larger models. Our hybrid scheduler EB+ applies this condition online to dynamically switch between EB and MB without manual intervention. Under non-stationary traffic with distribution or concurrency shifts, EB+ attains the highest or near-highest throughput in every setting, outperforming MB by up to 36.4%. 

---
# EnergyMamba: An Uncertainty-Aware Graph-Enhanced Selective State Space Model for Energy Consumption Prediction 

**Authors**: Dahai Yu, Rongchao Xu, Lin Jiang, Guang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.00506)  

**Abstract**: Energy consumption prediction is essential for efficient grid management, demand-side optimization, and sustainable energy planning. Although advanced machine learning methods have been employed for better prediction performance, existing works have two key limitations: (1) they usually formulate this task as a purely time-series prediction problem without explicitly modeling the spatial dependencies among different regions, and (2) they fail to provide reliable predictions with uncertainty estimates under abnormal situations such as extreme weather events. To advance existing research, we propose EnergyMamba, an uncertainty-aware spatiotemporal learning framework for accurate and reliable energy consumption prediction, which comprises two key components: (i) a novel Graph-Enhanced Selective State Space Model (GE-Mamba) that injects spatial context learned from the grid topology into the temporal dynamics, enabling coupled spatiotemporal modeling, and (ii) an Adaptive Sequential Conformalized Quantile Regression (AS-CQR) module, which includes locally adaptive normalization and an online feedback mechanism to dynamically calibrate prediction intervals under potential distribution shifts. We evaluate EnergyMamba on four large-scale real-world datasets from Florida, New York, and California. Results show EnergyMamba achieves around 5% improvement in prediction accuracy and 6% improvement in uncertainty quantification over 15 state-of-the-art baselines. 

---
# TAPS: Target-Aware Prefix Tree Selection for Diffusion-Drafted Speculative Decoding 

**Authors**: Zhuoyu Wang, Junnan Huang, Xinyu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.00487)  

**Abstract**: Using a diffusion model for parallel drafting is a promising approach for speculative decoding. By predicting tokens at multiple future positions in a single forward pass, diffusion drafters substantially reduce drafting latency. However, this shifts the bottleneck to verification: verifying a single sequence limits acceptance length, while verifying large draft trees incurs excessive target-model latency. We identify a key mismatch in existing draft-tree methods: existing diffusion-tree methods rank nodes by the marginal probability, ignoring that verification is prefix-conditioned. As a result, they may verify unreachable descendants of rejected prefixes, increasing latency with limited acceptance gains. To address this, we propose TAPS, a target-aware prefix selection method that turns diffusion marginals into path-conditioned acceptance estimates. TAPS then selects a compact prefix-closed subtree under a fixed verification budget, improving the acceptance-cost tradeoff rather than simply expanding the draft tree. Experiments across diverse datasets and model families demonstrate that TAPS achieves up to 7.9x lossless end-to-end speedup over vanilla autoregressive decoding, outperforming state-of-the-art DFlash and DDTree by 1.36x and 1.74x respectively. Our work is available at this https URL 

---
# Doing What They Say, Not What They Reason: Locating the Faithfulness Gap in LLM Agents 

**Authors**: Yufeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.00476)  

**Abstract**: Do LLM agents act on the reasoning they state? This question of process fidelity is central to using LLMs in social simulation, yet it is hard to measure where no reference for correct behavior exists. We study it in acontrolled setting, a Texas Poker simulator with a verifiable reference action for every decision by decomposing the faithfulness gap into two steps: reasoning-conclusion and conclusion-action. The two steps behave oppositely. 

---
# SDR: Set-Distance Rewards for Radiology Report Generation 

**Authors**: Halil Ibrahim Gulluk, Max Van Puyvelde, Wim Van Criekinge, Olivier Gevaert  

**Link**: [PDF](https://arxiv.org/pdf/2606.00440)  

**Abstract**: Reinforcement learning with verifiable rewards has rapidly advanced reasoning in vision--language models. However, for chest X-ray report generation, the standard rewards (i.e. exact-match accuracy and step-level processes) are incompatible because the reports consist of unordered and orthogonal findings, rather than a causal reasoning chain. We address this gap with a set-based view: each report is split into sentences and embedded by a frozen sentence transformer, yielding unordered embedding sets. We propose the use of set-to-set distances between generated and reference embeddings as continuous, permutation-invariant rewards. Across two datasets and three vision--language models (Qwen3-VL-2B/4B, Gemma3-4B), post-training with set-to-set distance based rewards via GRPO consistently outperforms supervised fine-tuning and exact-match GRPO on all headline metrics (BERTScore, RadGraph F1 and CheXbert F1 by average \%6.80, \%7.82 and \%4.45 relative improvements respectively). The same set distances also enable test-time best-of-$N$ selection: scoring candidates by their distance to training-report embeddings outperforms random selection on our trained models as well as three closed-source LLMs (Mistral-Small, Gemini-2.5 Flash-Lite, GPT-4o-mini) with on average \%16.4 relative improvement on BERTScore. Used as a streaming signal, they support a more efficient form of test-time scaling: pruning low-scoring candidates mid-generation reduces generated tokens by over 50\% while preserving the Findings quality of full best-of-$N$ selection. Together these results establish set-distance rewards as a unified signal for both post-training and test-time scaling in chest X-ray report generation. Our code is publicly \href{this https URL}{available}. 

---
# Weak Critics Make Strong Learners: On-Policy Critique Distillation for Scalable Oversight 

**Authors**: Can Jin, Jiakang Li, Rui Wu, Eddy Zhang, Dimitris N. Metaxas  

**Link**: [PDF](https://arxiv.org/pdf/2606.00424)  

**Abstract**: As large language models become stronger, weak supervisors may fail to provide reliable labels, preferences, or final judgments for complex outputs, limiting both weak-to-strong generalization and scalable oversight. We study a more tractable form of weak supervision: using a weak model as a critic rather than as a labeler or judge. Instead of solving the task or selecting the correct answer, the weak critic only needs to provide a non-misleading revision direction that helps the strong model better use its own knowledge. We call this setting *weak-critic strong oversight*. We first show that weak critiques can improve frozen strong models at inference time, and that critique quality is key to this improvement. We then propose progressive on-policy critique distillation (**OPCD**), which filters high-quality critiques and distills critic-guided behavior into the strong model through adaptive self-teacher signals. Experiments on reasoning and alignment benchmarks show that our method improves strong models over training epochs, suggesting an effective path for scalable oversight with weak supervision. 

---
# VESTA: Visual Exploration with Statistical Tool Agents 

**Authors**: William Rudman, Abhishek Divekar, Kanishk Jain, Sebastian Joseph, Stella S. R. Offner, Matthew Lease, Kyle Mahowald, Greg Durrett, Junyi Jessy Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.00384)  

**Abstract**: Fitting quantitative models to data is a central step in scientific workflows, yet it remains one of the least automated. Recent agent-based systems leverage language and vision-language models (VLMs) to iteratively propose and refine statistical models, but these systems struggle on more challenging modeling tasks. To address these limitations, we introduce VESTA: Visual Exploration with Statistical Tool Agents, a framework that equips VLMs with a dynamically growing exploration toolkit to guide model refinement through data transformations, hypothesis-driven visualizations, and robust statistical tests. Unlike prior systems that rely on iterative critique alone, VESTA actively explores data before and during refinement by selecting or creating diagnostic tools, which accumulate in the model's context and can be reused later. We evaluate VESTA against established baselines in three toolkit configurations: no tools, static expert-written tools, and dynamic model-written tools. To support this evaluation, we introduce DAWN (Dataset for Automated Workflows and Numerical Modeling), a benchmark targeting distribution fitting and time series modeling with varying difficulty tiers, and culminating in real-world astronomy tasks including modeling initial mass functions and gravitational-wave chirp signals. We find that VESTA's dynamic tool creation outperforms prior agentic pipelines, with the largest gains on complex and domain-specific tasks. We further show that dynamically generated tools are substantially more sophisticated than those produced by existing visual tool-creation systems, covering more diagnostic categories per function and strongly preferring visual outputs that the VLM critic can reason over directly. 

---
# The Deterministic Horizon: When Extended Reasoning Fails and Tool Delegation Becomes Necessary 

**Authors**: Dongxin Guo, Jikun Wu, Siu Ming Yiu  

**Link**: [PDF](https://arxiv.org/pdf/2606.00376)  

**Abstract**: Extended chain-of-thought reasoning can degrade performance on deterministic state-tracking tasks, not due to preference biases, but limits rooted in the information-theoretic capacity of decoder-only attention. We establish: (1) an Attention Bottleneck Theorem with a complementary achievability construction, bounding state-tracking capacity as $O(H \cdot \log(L/H) \cdot \sqrt{d_h})$; (2) a context-dependent error model yielding super-exponential accuracy decay; (3) the State-Space Jaccard metric distinguishing capability from preference failures; (4) a Deterministic Horizon $d^* \in [19, 31]$ beyond which tool delegation becomes necessary. Across 12 models and 8 task domains (including SWE-Bench, WebArena, and SQL-Multi), tool-integrated reasoning consistently outperforms neural chain-of-thought; on the primary model suite it reaches 86-94% accuracy versus 24-42% for neural chain-of-thought. Fine-tuning on optimal-length traces yields $<$5% improvement, confirming an architectural ceiling, and high cross-model correlation ($r = 0.81$-$0.91$) indicates these failures are architectural rather than training-specific. Our results provide principled guidance for when pure neural reasoning should yield to hybrid approaches in agentic systems. 

---
# From "Weak" Signals to Strong Models: Preference Delta Aggregation with LoRA Merging 

**Authors**: Qi Sun, Siyue Zhang, Yulin Chen, Yuxiang Xue, Ru Peng, Chen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2606.00357)  

**Abstract**: Training strong large language models (LLMs) requires high-quality supervision, which is often scarce. Recent work shows that paired preference data from weak-weaker model pairs (e.g., Qwen3 4B over 1.7B), despite the limited quality of individual responses, can provide an effective supervision signal through relative quality deltas, which we term a "weak" signal. This motivates a key research question: can multiple "weak" signals be constructively aggregated for improving strong models (e.g., Qwen3 8B)? To this end, we propose Preference Delta Aggregation (PDA), the first framework that derives a preference delta from each weak-weaker model pair, instantiates it as a LoRA adapter learned through preference optimization, and aggregates the resulting deltas via LoRA merging. To further mitigate directional interference during LoRA merging, we introduce Geometric Alignment Merging (GAM), a geometry-aware merging method that aligns adapter subspaces before aggregation, enabling more robust composition of diverse deltas. Evaluations on knowledge reasoning and agentic search benchmarks show that aggregating multiple "weak" signals pushes performance beyond any single signal, with further gains as additional signals are incorporated. Correspondingly, PDA with GAM improves the strong model by 6.8 and 7.3 points on average for knowledge reasoning and agentic search, respectively. It outperforms all single-delta and multi-delta baselines, exceeding the best single-delta baseline by 2.1 and 4.3 points. Further analysis attributes these gains to the effective composition of complementary capabilities encoded across distinct preference deltas. 

---
# From Noise to Control: Parameterized Diffusion Policies 

**Authors**: Renhao Zhang, Haotian Fu, Mingxi Jia, George Konidaris, Yilun Du, Bruno Castro da Silva  

**Link**: [PDF](https://arxiv.org/pdf/2606.00336)  

**Abstract**: We propose Parameterized Diffusion Policy (PDP), a framework for learning diffusion policies conditioned on low-dimensional, continuous parameters embedded in a learned behavior manifold. By constructing this manifold so that distances between latent representations reflect the semantic similarity between physical trajectories, we transform diffusion from a mechanism for stochastic diversity into a precise and optimizable tool for behavior steering. Our approach enables smooth interpolation between known strategies and efficient adaptation to novel constraints without updating policy weights. We demonstrate that PDP significantly improves adaptation performance on complex multimodal benchmarks in both simulated and real-robot experiments compared to standard diffusion policies, particularly in scenarios requiring the synthesis of novel behaviors. 

---
# Coupling Language Models with Physics-based Simulation for Synthesis of Inorganic Materials 

**Authors**: Edward W. Staley, Tom Arbaugh, Michael Pekala, Alexander New, Christopher D. Stiles, Nam Q. Le, Gregory Bassen, Wyatt Bunstine, Tyrel McQueen  

**Link**: [PDF](https://arxiv.org/pdf/2606.00315)  

**Abstract**: Modern generative machine learning (ML) models can propose novel inorganic crystalline materials with targeted properties; however, synthesis planning of these materials remains difficult due to the complexity of the associated physical processes and limited availability of computational tools. We introduce a novel hybrid framework to evaluate Large Language Models (LLMs) in inorganic synthesis planning by combining thermodynamic databases with simplified kinetics models to approximate realistic synthesis conditions. As a case study, we focus on the niobium-oxygen system, which features multiple industrially relevant oxide phases with well-characterized data. In computational simulations, we compare LLM-generated synthesis routes with classical path-planning algorithms, showing that the implicit priors in LLMs can yield more viable strategies. In our evaluation setting, classical search methods serve primarily as a foil rather than a direct competitor. This illustrates the relative complexity of the problem and highlights where the LLM's implicit priors add value. 

---
# Model-Native Computing Architecture: Envisioning Future System Architecture Through the Lens of Computer Architecture 

**Authors**: Hai Lin  

**Link**: [PDF](https://arxiv.org/pdf/2606.00288)  

**Abstract**: Large language models are undergoing a transition from model technology to system technology. As developers use Codex, Claude Code, AutoGPT, and related agents to write code, manage projects, and execute multi-step tasks, recurring engineering problems such as cache reuse, context management, agent scheduling, and permission control increasingly resemble classical computer systems problems. This paper develops that analogy as a visionary survey. We map concepts from computer architecture to the emerging model-native stack and review work on LLM-as-OS, memory management, agent frameworks, tool protocols, multi-agent coordination, cognitive architectures, and safety governance. We argue that these strands address different layers of the same system but lack a unified model.
To fill this gap, we propose the Intelligent Computing Architecture Model (ICAM), a six-layer framework for model-native computing with explicit interface contracts and design axioms. ICAM resolves the apparent tension over whether an LLM is more like a CPU or an operating system through a dual-plane view: a probabilistic execution plane concerned with what can be computed, and a deterministic control plane concerned with what should be computed. We further introduce three design laws: the Semantic Locality Law for KV-cache reuse and inference speedup, the Context Budget Law for effective working sets under finite windows and attention decay, and the Agent Speedup Law for diminishing returns in multi-agent collaboration. We validate these laws against published system-level data and relate them to recent evidence on agentic software practices. We conclude by identifying where the analogy breaks down and outlining a research roadmap for model-native computing. This is a conceptual and survey contribution; it does not report new experiments. 

---
# Evaluating Bivariate Causal Statements Based on Mutual Compatibility 

**Authors**: Erik Jahn, Dominik Janzing  

**Link**: [PDF](https://arxiv.org/pdf/2606.00278)  

**Abstract**: For many real-world systems, causal ground truth is difficult to obtain, making claims about causal effects hard to assess. We develop methods for evaluating collections of $\binom{n}{2}$ bivariate causal statements over a set of $n$ variables. In the setting of acyclic linear statements, any such collection can be extended to a unique multivariate causal model, but we argue that this induced model is implausible if it imposes substantial additional confounding to explain observed correlations. We introduce a compatibility score that quantifies this notion of plausibility, notably without relying on the faithfulness assumption. Additionally, we define an incompatibility score for purely graphical bivariate causal statements, based on global consistency constraints that are derived from acyclicity and faithfulness assumptions. We give theoretical and empirical evidence that both scores can successfully distinguish correct from incorrect causal statements in generic settings. Moreover, we demonstrate the practical applicability of our methods by analyzing causal claims made by large language models. Our work aims to provide a foundation for assessing the reliability of causal information derived from human experts or artificial intelligence in settings where alternative forms of validation are unavailable. 

---
# On Wednesdays, We Ask Questions: Optimizing "Active Listening" in Automated Legal Triage and Referral 

**Authors**: Quinten Steenhuis, Jacqueline Harvey  

**Link**: [PDF](https://arxiv.org/pdf/2606.00272)  

**Abstract**: The FETCH classifier generates follow-up questions to help refine the best match for the applicant's legal problem, using a low-cost ensemble of LLMs. In this paper, we describe an expert attorney and LLM-assisted evaluation of the follow-up question approach in FETCH and show that while low-cost LLMs perform well at classification tasks, generating high-quality plain-language questions in this setting appears to require a more sophisticated and higher-cost model. Through discussion with legal intake workers, we propose a rubric for the evaluation of legal intake classification questions, and we find that prompt engineering alone is not enough to improve question quality for intake purposes. We also find that LLM-as-judge and human ratings diverge. We demonstrate that with the addition of a single high-cost model, GPT-5, the classifier can elicit relevant information from applicants for legal help, and that the questions lead to more accurate performance at classification tasks. We also find uneven fact elicitation across different categories, including domestic violence, at odds with family law screening protocols, suggesting the value of including dedicated screening panels for certain areas of law. 

---
# Robust Shielding for Safe Reinforcement Learning 

**Authors**: Edwin Hamel-De le Court, Thom Badings, Alessandro Abate, Francesco Belardinelli, Francesco Fabiano  

**Link**: [PDF](https://arxiv.org/pdf/2606.00270)  

**Abstract**: Shielding is an effective approach to formally guarantee the safety of reinforcement learning agents in Markov decision processes (MDPs). However, existing shielding techniques typically assume knowledge of the safety-relevant transition dynamics - a requirement that is seldom met in practice. To address this limitation, we introduce a novel shielding framework for robust MDPs (RMDPs), i.e., MDPs with sets of transition probabilities. We define safety as the satisfaction of a linear temporal logic (LTL) formula with a certain threshold probability under the worst-case transition probabilities of the RMDP. We prove that our shielding framework is both sound and optimal for the RMDP: every policy admissible by the shield is safe, and conversely, every safe RMDP policy is admissible by the shield. We combine our approach with existing sampling methods for learning transition probabilities of MDPs with probably approximately correct (PAC) guarantees. This combination enables the construction of shields for MDPs that, with high confidence, guarantee safety while remaining minimally restrictive. Our experiments show that our shields for learned RMDPs guarantee safety in unknown MDPs while recovering strong expected return as the number of samples increases. 

---
# Closed-Loop Neural Activation Control in Vision-Language-Action Models 

**Authors**: Abhijith Babu, Ramneet Kaur, Nathaniel D. Bastian, Olivera Kotevska, Susmit Jha, Yanzhao Wu, Sumit Kumar Jha, Anirban Roy  

**Link**: [PDF](https://arxiv.org/pdf/2606.00269)  

**Abstract**: Vision-Language-Action (VLA) models can be steered at test time by intervening on semantically meaningful internal directions, but existing methods use a fixed steering coefficient, effectively operating in open loop. This is poorly suited to embodied control, where task state and concept error evolve over time, often causing overcorrection, oscillation, and reduced task success, especially for temporal behaviors such as speed and smoothness. We propose CTRL-STEER, a closed-loop framework that replaces static intervention strength with adaptive, time-varying control signals. The key idea is to decouple representation from regulation: rather than assuming temporal concepts are directly controlled by individual neurons, we steer along motion-aligned residual directions while a feedback controller adjusts intervention magnitude online. We instantiate this framework with both PID and reinforcement learning based controllers. Experiments with a fine-tuned OpenVLA policy on four LIBERO task suites show that CTRL-STEER achieves more stable concept regulation and a better steering-task success trade-off than fixed-coefficient baselines, without modifying or retraining the base model. 

---
# Capability Self-Assessment: Teaching LLMs to Know Their Limits 

**Authors**: Haoyan Yang, Reza Shirkavand, Yukai Jin, Jiawei Zhou, Shangqian Gao, Heng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2606.00251)  

**Abstract**: The ability to recognize one's own limitations and decide whether to solve a problem or delegate is fundamental for reliable intelligent systems. Yet we show that modern large language models systematically lack this ability: across diverse model families and scales, they overestimate their competence and attempt queries they cannot solve. We refer to this ability as Capability Self-Assessment (CSA) and formulate it as a policy-learning problem, aiming to improve self-assessment while preserving the model's original capabilities. Our results show that reinforcement learning teaches CSA effectively, significantly outperforming supervised fine-tuning while preserving original capabilities. In contrast, supervised fine-tuning severely degrades the capabilities the model is meant to assess. Moreover, learned self-assessment behavior generalizes well out of distribution, suggesting that CSA is a transferable model trait. Finally, CSA is practically useful: it improves local-cloud decision making at inference time and provides a signal for targeted data selection during training. 

---
# Geodesic Flow Matching for Denoising High-Dimensional Structured Representations 

**Authors**: Karim Habashy, Chris Eliasmith  

**Link**: [PDF](https://arxiv.org/pdf/2606.00248)  

**Abstract**: Vector Symbolic Algebras (VSAs) enable robust neurosymbolic reasoning by encoding symbolic information into high-dimensional distributed representations. For continuous domains, Spatial Semantic Pointers (SSPs) extend this framework by mapping variables onto continuous toroidal manifolds. However, standard approaches like Flow Matching assume a flat Euclidean geometry, which fails to account for the geometric constraints imposed on valid SSP states. We demonstrate that this assumption fails for SSPs: Euclidean linear interpolants ``cut through" the manifold's interior, destroying the phase and magnitude structure required for accurate decoding. To resolve this, we employ Geodesic Flow Matching, adapting Riemannian transport dynamics to strictly restrict the denoising flow to the SSP toroidal manifold. We validate this approach in a Spiking Neural SLAM system, showing that manifold-aware cleanup stabilizes path integration against drift. The method achieves a 72\% reduction in tracking error and enables a 40\% increase in neural efficiency compared to competitive baselines. Code is available at this https URL . 

---
# MindZero: Learning Online Mental Reasoning With Zero Annotations 

**Authors**: Shunchi Zhang, Jin Lu, Chuanyang Jin, Yichao Zhou, Zhining Zhang, Tianmin Shu  

**Link**: [PDF](https://arxiv.org/pdf/2606.00240)  

**Abstract**: Effective real-world assistance requires AI agents with robust Theory of Mind (ToM): inferring human mental states from their behavior. Despite recent advances, several key challenges remain, including (1) online inference with robust uncertainty updates over multiple hypotheses; (2) efficient reasoning suitable for real-time assistance; and (3) the lack of ground-truth mental state annotations in real-world domains. We address these challenges by introducing MindZero, a self-supervised reinforcement learning framework that trains multimodal large language models (MLLMs) for efficient and robust online mental reasoning. During training, the model is rewarded for generating mental state hypotheses that maximize the likelihood of observed actions estimated by a planner, similar to model-based ToM reasoning. This method thus eliminates the need for explicit mental state annotations. After training, MindZero internalizes model-based reasoning into fast single-pass inference. We evaluate MindZero against baselines across challenging mental reasoning and AI assistance tasks in gridworld and household domains. We found that LLMs alone are insufficient; model-based methods improve accuracy but are slow, costly, and limited by backbone MLLM capacity. In contrast, MindZero enhances MLLMs' intrinsic ToM ability and significantly outperforms model-based methods in both accuracy and efficiency, showing that mental reasoning can be effectively learned as a self-supervised skill. 

---
# TIGER: Traceable Inference with Graph-Based Evidence Routing for Mitigating Hallucinations in Multimodal Generation 

**Authors**: Kaixiang Zhao, Tianrun Yu, Shawn Huang, Porter Jenkins, Yushun Dong, Amanda Hughes  

**Link**: [PDF](https://arxiv.org/pdf/2606.00232)  

**Abstract**: We study fact-level repair for multimodal generation, where a fluent output may contain specific facts that are not supported by the input. Existing inference-time repair methods often generate feedback by jointly conditioning on the input and the current output. This design has two limitations: hallucinated claims in the output can bias the model's interpretation of the input, and free-form feedback cannot be ranked or scheduled at the fact level. We present TIGER, an inference-time framework that redesigns feedback for localized repair. TIGER independently extracts an observation graph from the input and a claim graph from the current output, then assigns each claim a graph-conditioned risk score based on support and conflict. The model repairs selected high-risk claims while keeping the backbone frozen. We provide a convergence analysis showing that the expected total risk decreases geometrically to an explicit asymptotic bound under mild assumptions. Experiments across four cross-modal paths, including image-to-text, image+text-to-text, audio-to-text, and video-to-text, show that TIGER reduces unsupported content while preserving task quality. The gains hold across multiple backbones, and a CrisisFACTS case study suggests that the same repair mechanism can improve grounding in multi-source settings. 

---
# CAST: Non-Privileged Clipped Asymmetric Self-Teaching with Advantage Flipping for GRPO 

**Authors**: Yang Li, Gongle Xue, Yijia Guo, Yuheng Yuan, Liwen Hu, Lei Ma  

**Link**: [PDF](https://arxiv.org/pdf/2606.00172)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR), especially Group Relative Policy Optimization (GRPO), has been widely used to improve reasoning in large language models. However, outcome-level rewards provide only sparse supervision, and group-relative advantages vanish when all sampled trajectories for a prompt are either correct or incorrect. On-Policy Self-Distillation (OPSD) offers dense token-level guidance, but its token preferences are not necessarily aligned with trajectory correctness; empirical diagnostics show that OPSD signals behave differently on correct and incorrect rollouts, with teacher-positive and teacher-negative gap signals exhibiting different noise profiles. These diagnostics are conducted under an OPSD-style privileged teacher context for analysis only, whereas CAST training uses answer-free self-teacher this http URL by these observations, this work proposes CAST, an answer-free self-distillation method for GRPO-style RLVR. CAST keeps the verifier-grounded GRPO objective, but uses a stop-gradient self-teacher to shape token-level advantages according to trajectory correctness. Unlike prior self-distilled RLVR methods, CAST does not require reference-solution-conditioned teacher scoring, keeps the self-teacher log-probability gap active throughout training, and applies bidirectional local advantage sign reversal: teacher-negative tokens in correct trajectories can receive negative token-level advantages, while teacher-positive tokens in incorrect trajectories can receive bounded positive local advantages. For zero-variance all-correct and all-wrong groups, CAST assigns bounded sign-constrained base advantages, so these otherwise zero-gradient groups can contribute verifier-signed token feedback. Experiments on mathematical reasoning show that CAST improves RLVR training while retaining a lightweight, verifier-grounded trajectory-level objective. 

---
# A Multi-AI-agent Framework Enabling End-to-end Finite Element Analysis for Solid Mechanics Problems 

**Authors**: Titu Ranjan Sarker, Muhammed Jawaad Zulqernine, Ling Yue, Shaowu Pan, Chenxi Wang, Shiyao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2606.00138)  

**Abstract**: Finite element analysis (FEA) is the most important numerical approach for solid mechanics. Challenges of FEA include a steep learning curve for entry-level users and potential false simulations due to incorrect definitions of key simulation components, such as boundary conditions, load cases, and solution variables. Years of engineering experience are usually necessary for real-world problem-solving. To address these issues, we present AbaqusAgent, a multi-agent framework grounded in large language models (LLMs) for solid mechanics analyses. AbaqusAgent is developed to facilitate analysis case generation and execution using Abaqus, one of the most widely used FEA packages, by turning users' natural-language instructions into executed FEA analyses and result visualization. AbaqusAgent is composed of six agents, including interpreter, architect, input writer, runner, reviewer, and visualizer agents, encompassing all the essential pre-processing and post-processing steps of standard FEA analyses. A wide variety of 50 solid mechanics problems have been successfully validated, achieving an overall success rate of 86%. Beyond improving the efficiency of FEA for solid mechanics problems and lowering the barrier to computational mechanics education, AbaqusAgent advances the human-simulation interaction paradigm and enables integration with AI-empowered optimization and material characterization workflows. The code is available at this https URL 

---
# Evaluating Interactive Reasoning in Large Language Models: A Hierarchical Benchmark with Executable Games 

**Authors**: Mingyuan Fan, Weiguang Han, Daixin Wang, Cen Chen, Zhiqiang Zhang, Jun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.00103)  

**Abstract**: We introduce a multi-turn interactive framework for reasoning evaluation that treats reasoning as active evidence acquisition and belief updating. Wherein, LLMs receive only the task rules, must issue targeted queries to a hidden environment, integrate partial observations over time, and decide when to submit a final answer. Beyond standard success rate and interaction efficiency, we evaluate contextual robustness under controlled contextual perturbations, and metacognitive adaptation through counterfactual revision and necessity judgment. We instantiate the framework as a benchmark of 474 executable games, each evaluated under five fixed configuration search spaces corresponding to five difficulty levels, and evaluate a broad set of frontier LLMs. Results show that the benchmark is highly discriminative, exposing large differences not only in success rate but also in interaction efficiency. Moreover, we empirically show that contextual perturbations cause moderate but consistent declines, whereas counterfactual revision and necessity judgment lead to much larger drops. 

---
# On the evolution of the concept of probability as a mirror of the evolution of reason 

**Authors**: Jean-Louis Le Mouël, Vincent Courtillot, Dominique Gibert, Vladimir Kossobokov, Jean-Baptiste Boulé, Pierpaolo Zuddas, Fernando Lopes, Païkan Marccagi, Alexis Maineult  

**Link**: [PDF](https://arxiv.org/pdf/2606.00102)  

**Abstract**: Over the centuries, probability theory has grown from the calculus of games of chance into a central framework for reasoning under uncertainty. This article interprets that evolution not merely as a mathematical history, but as a transformation of rationality itself. From Pascal and Fermat's combinatorial symmetry to the inductive logic of Bayes and Laplace, from Poisson's statistics of events to Kolmogorov's axiomatic formalization, probability progressively incorporated uncertainty, time, and coherence into scientific judgment. This trajectory reaches a mature epistemological form in modern Bayesian inference, especially in Tarantola's view of probability as a logic of information, where prior knowledge and data are combined coherently. Yet this framework also exposes a limit: probability quantifies uncertainty about well-defined propositions, but does not by itself formalize the vagueness of the concepts used to describe them. The article therefore examines how rationality extends beyond probability. Fuzzy logic is presented as a rigorous language for graded meaning and qualitative judgment, while deep learning is analyzed as a distinct, powerful mode of prediction based on geometric interpolation and optimization rather than explicit inference. By situating probability, fuzzy logic, and deep learning in a common historical and epistemological perspective, the article clarifies their roles and limits. It argues that contemporary scientific rationality cannot be reduced to data-driven performance alone, but requires the explicit articulation of uncertainty, vagueness, and inference. 

---
# Product-Aware Deep Autoencoders for Robust Process Monitoring in Multi-Product Cyber-Physical Systems 

**Authors**: MD Shafikul Islam, Jordan Carden  

**Link**: [PDF](https://arxiv.org/pdf/2606.00052)  

**Abstract**: As Industry 4.0 accelerates the integration of Cyber-Physical Systems (CPS) in manufacturing, robust anomaly detection has become critical for ensuring process safety and security. Current data-driven approaches typically employ "product-agnostic" or global models trained on the aggregate of all normal operating data. However, modern industrial facilities frequently operate under diverse product grades. While computationally simple, these global models inherently expand their decision boundaries to accommodate the variance of multiple modes, creating a "blind spot" where subtle anomalies or targeted cyber-physical attacks may be masked by the wide acceptance region of the model. In this work, we first demonstrate that the vulnerability described above is present in global-agnostic models operating across multiple product grades. We then present a Product-Aware Autoencoder as a principled mitigation that restricts the learning domain to grade-specific distributions. While this approach reduces the identified blind-spot risk, we do not claim it as the optimal mitigation among all possible alternatives. We rigorously validate this approach against a Global Agnostic baseline using the Extended Tennessee Eastman Process (TEP) benchmark. Our empirical results indicate that the Product-Aware framework performs comparably to the global baseline on standard detection metrics, while offering improved robustness to product-grade-specific operating modes. Most critically, stress tests simulating our hypothetical attack scenarios reveal that while the global model fails to detect operational deviations in 77.8% of the scenarios, the product-aware system achieves 100% detection accuracy. These findings suggest that, in flexible manufacturing environments, generalized anomaly detectors can pose non-trivial security risks, motivating a shift toward mode-aware diagnostic architectures. 

---
# Grokers: Bottom-Up Inductive Comprehension and Write-Time Intelligence over Typed Knowledge Graphs 

**Authors**: Gregory Magarshak  

**Link**: [PDF](https://arxiv.org/pdf/2606.00050)  

**Abstract**: We present Grokers, an architecture for building persistent, structured comprehension of typed knowledge graphs through bottom-up inductive traversal of dependency subgraphs. Unlike retrieval-augmented generation (RAG), which pays full comprehension cost at every query, Grokers pushes intelligence to write time: autonomous Groker agents analyze nodes in a typed stream graph, extract structured attributes via governed language model (LM) calls, and inductively compose that understanding upward through dependency relations, writing enriched typed attributes that serve all future queries at zero additional LM cost. We prove three formal properties: (1) the Byte-Identity Theorem, establishing that context blocks assembled from a transactionally-maintained denormalization index are byte-identical across LM turns between semantic changes, enabling KV-cache hit rates approaching 100%; (2) the Accumulation Monotonicity Theorem, establishing that the fraction of interactions resolved without LM calls is non-decreasing in the number of completed interactions under a governed wisdom library growth protocol; and (3) the Dual-Traversal Ordering Theorem, establishing that top-down generation and bottom-up comprehension are the unique correct traversal orderings for their respective tasks over a dependency DAG, and that their composition closes into a complete generation-comprehension cycle. We further present a deterministic alternative to embedding-based semantic search, with a synonym caching protocol whose LM fallback rate converges to zero for finite-vocabulary domains. A reference implementation is provided in the open-source Qbix / Safebox / Safebots stack. 

---
# Universal Quantum Transformer 

**Authors**: Sungyong Chung, Alireza Talebpour  

**Link**: [PDF](https://arxiv.org/pdf/2606.00045)  

**Abstract**: Classical continuous-space neural networks fundamentally struggle to lock into exact mathematical symmetries, such as modular arithmetic and non-commutative algebra. To approximate these discrete logical rules, they often rely on massive parameter scaling, resulting in stochastic instability even after delayed generalization phenomena known as grokking. Here, we introduce the Universal Quantum Transformer (UQT), a fundamentally novel, quantum-native computing architecture that uses the physical properties of multi-qubit systems as a universal inductive bias for exact mathematical and algebraic reasoning. Rather than translating classical neural mechanisms, our framework relies entirely on parameterized geometric phase embedding and $SU(2)$ wave-interference. We demonstrate that the quantum attention circuit, operating on a highly compact 5-qubit substrate, perfectly learns two highly distinct formal classes: cyclic modular arithmetic ($\mathbb{Z}_{11}$) and non-Abelian algebra (the $S_4$ permutation group). While classical attention-based networks exhibit stochastic instability at convergence, the UQT achieves mathematically exact, deterministic generalization. We refer to this phenomenon as crystallization: a step beyond the well-known phenomenon of grokking. Crucially, this framework yields massive computational and memory advantages by theoretically bypassing the quadratic bottleneck of classical self-attention, and by logarithmically compressing the required representation dimension to eliminate the massive over-parameterization inherent to classical networks. Finally, we deploy this architecture on noisy intermediate-scale quantum (NISQ) hardware, proving its viability on current IBM Quantum computers. These results establish parameterized quantum topology as a universally superior physical substrate for exact artificial intelligence. 

---
# MindGames Arena Generalization Track: In2AI Solution with Delayed Per-Step Reward Attribution 

**Authors**: Aliaksei Korshuk, Alexander Buyantuev, Ilya Makarov  

**Link**: [PDF](https://arxiv.org/pdf/2606.00017)  

**Abstract**: Training language model agents for multi-agent strategic interaction presents a core difficulty: the quality of any action may depend on future events that never materialize, on moves that violate game rules, or on decisions made by other players. Standard reinforcement learning assumes that rewards can be assigned at each step, but this assumption fails in settings where outcomes are entangled across time and agents. We introduce delayed per-step reward attribution with eligibility gating, an episode lifecycle and postprocessing pipeline that computes rewards only at episode end, propagates them back to originating steps according to task-specific semantics, and excludes steps that lack valid dependent information from training. Together with asynchronous rollout generation via vLLM's continuous batching, curriculum-based opponent sampling, and multi-level stratified batch construction, this approach enables stable, sample-efficient RL training in multi-agent environments. We evaluate on the MindGames Arena benchmark at NeurIPS 2025, where a single 8-billion-parameter open-source model trained with our method matched or surpassed substantially larger proprietary systems, including GPT-5, in head-to-head play and took first place in both the Open (unrestricted) and Efficient (<=8B parameters) tracks. 

---
# Optimal Transport-based Permutation-Invariant Bayesian Optimization of Offshore Wind Farm Layouts 

**Authors**: Antonio Candelieri, Laurens Bliek  

**Link**: [PDF](https://arxiv.org/pdf/2606.00009)  

**Abstract**: Bayesian Optimization (BO) is widely and successfully adopted for solving optimization problems having an expensive-to-evaluate, black-box, and non-convex objective function. However, the vanilla BO algorithm is not able to exploit possible symmetries characterizing the target problem. An intuitive case is given by optimal location problems, whose decision variables refer to a finite set of points within a continuous space, with the order of points not affecting the value of the objective function. We refer to this setting as optimization over layouts to distinguish from optimization over point-clouds where, instead, the order of points counts. As an instance of optimization over layouts we consider a real-life industrial-relevant application, that is the optimization of the layout of an offshore wind farm: given identical wind turbines, switching any pair of them has not any effect on the annual energy production. Based on Optimal Transport theory, we propose a Permutation-Invariant BO approach, namely PIBO, proved to provide better wind farm layouts when compared to the vanilla BO approach while cutting computation time roughly in half. 

---
# Agents on a Tree: Pathwise Coordination for Multi-Objective Molecular Optimization 

**Authors**: Jia Zhang, Tengfei Ma, Tianle Li, Daojian Zeng, Xieping Gao, Xiangxiang Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2606.00008)  

**Abstract**: Multi-objective molecular optimization requires searching vast chemical spaces under conflicting objectives, where early design decisions strongly constrain downstream outcomes. Existing methods typically rely on a single policy or fixed scalarization, which limits their ability to represent diverse trade-offs and to explore multiple promising design trajectories. We propose ATOM, a multi-agent framework that formulates molecular optimization as a tree-structured search. Each node corresponds to an atomic operation and hosts an agent specialized for a particular objective or decision context. Agents coordinate along different paths of the tree rather than enforcing a global consensus, enabling the method to maintain and compare alternative molecular evolution trajectories. A global memory of past optimization behaviors further supports balanced exploration and exploitation across objectives. This tree-structured interaction enables reasoning over long-horizon dependencies inherent in molecular design. Experiments on challenging multi-objective benchmarks involving activity, synthesizability, and ADMET-related properties show that ATOM consistently achieves improved Pareto coverage and hypervolume over strong baselines. These results demonstrate the effectiveness of pathwise multi-agent coordination for molecular optimization. Code is available at this https URL. 

---
# Deliberative Curation: A Protocol for Multi-Agent Knowledge Bases 

**Authors**: Steven Johnson  

**Link**: [PDF](https://arxiv.org/pdf/2606.00007)  

**Abstract**: As AI agents transition from isolated tools to collaborative participants in shared knowledge ecosystems, governing collective knowledge curation becomes a critical challenge. Human platform governance mechanisms do not transfer directly: agent statelessness undermines deterrence-based sanctions, model homogeneity violates independence assumptions underlying crowd wisdom, and sycophancy collapses deliberative consensus.
We propose a deliberative curation protocol combining three governance layers: (1) a knowledge artifact lifecycle formalized as a labeled transition system; (2) reputation-weighted deliberative voting integrating Beta Reputation with EigenTrust amplification; and (3) graduated sanctions adapted for stateless agents, including broken agent handling distinguishing malfunction from adversarial behavior.
We evaluate the protocol through agent-based simulation with 100 agents across seven behavioral archetypes under two adversity scenarios (30 seeds, paired t-tests). The protocol trades modest precision under benign conditions for substantially better resilience under adversity: 0.826 vs 0.791 for majority vote under moderate adversity (p<0.001), widening to 0.807 vs 0.740 under stress (p<0.001). The protocol degrades roughly three times more slowly than majority vote. Ablation analysis identifies commit-reveal vote concealment as the most impactful single component (8.2-8.6pp precision improvement, p<0.001), outperforming reputation weighting and deliberation combined. Graduated sanctions were not exercised in simulation and remain empirically unvalidated. 

---
# Emergent Collaborative Deliberation in Multi-Model AI Systems: A BFT-Derived Protocol for Epistemic Synthesis 

**Authors**: VD Doske  

**Link**: [PDF](https://arxiv.org/pdf/2606.00005)  

**Abstract**: We present the Consilium Protocol, a Byzantine Fault Tolerance-derived architecture for structured multi-model AI deliberation that treats inter-model disagreement as epistemic signal rather than error. The protocol assigns engineered cognitive personas to language models -- separating what a model is from how it reasons -- and introduces an In-Sample/Out-of-Sample validation framework adapted from quantitative finance to distinguish training-data consensus from empirically grounded conclusions. Across 1,478 deliberation sessions spanning 32 topics in 10 domain categories, we demonstrate that (1) the cognitive persona, not the underlying model, determines epistemic behavior: free edge-inference models costing 0.0002 USD per batch produced comparable analytical output to frontier models costing 10.69 USD; (2) RLHF alignment training creates measurable, domain-specific epistemic blind spots -- contested policy topics exhibit 12.3 percentage points less adversarial challenge than settled science topics, and AI safety topics show asymmetric bias ($\Delta$=11.6%) where models challenge claims that AI is dangerous far more vigorously than claims that AI risk is overstated; (3) the protocol exhibits no directional bias of its own (immigration $\Delta$=2.3%, renewables $\Delta$=1.2%); and (4) out-of-sample evidence retrieval validated 239 claims with 100% evidence retrieval and surfaced 167 blind-spot discoveries invisible to training-data deliberation. Run-to-run reproducibility across randomized model$\times$persona assignments averages $\pm$2.2% standard deviation. Total cost for the complete battery including all overhead: 217 USD. We release the protocol specification under MIT license to enable independent verification. 

---
# Position Paper: Post-Solve Robustness in Decision Engines: Feasible Regions and Smoothness Under Perturbations 

**Authors**: Yi-Xiang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2606.00002)  

**Abstract**: Mixed-Integer Linear Programming (MILP) decision engines routinely output nominally optimal plans for high-stakes industrial systems. Yet deployment rarely matches solve-time assumptions: small perturbations in costs, demands, or resource availability can invalidate feasibility or trigger discontinuous shifts to qualitatively different solutions. We argue that this post-solve robustness gap is a missing layer in today's optimization pipelines and a missing evaluation dimension for learning-enabled decision systems. Rather than replacing robust optimization or stochastic programming, the proposed layer audits a solved incumbent and returns solver-backed evidence about how far that solution can be trusted. We formalize two central objects: (i) an $\epsilon$-near-optimal feasible neighborhood in parameter space, capturing when an incumbent remains feasible and near-optimal under perturbations, and (ii) solution smoothness in decision space, capturing whether nearby alternatives with small combinatorial edits remain competitive. We then synthesize the most relevant partial answers from sensitivity and stability analysis, robust optimization, neighborhood search, adversarial testing, and learning-based enhancements, and articulate an agenda for a unified post-solve robustness layer. Concretely, we call for certified inner approximations around the incumbent, probabilistic robustness estimation with calibrated uncertainty, adversarial robustness margins, and learning-based prediction and explanation aligned with solver-backed verification. We conclude with a compact reporting template and evaluation protocol that would make robustness a first-class output of decision engines. 

---
# Mitigating Perceptual Judgment Bias in Multimodal LLM-as-a-Judge via Perceptual Perturbation and Reward Modeling 

**Authors**: Seojeong Park, Jiho Choi, Junyong Kang, Seonho Lee, Jaeyo Shin, Hyunjung Shim  

**Link**: [PDF](https://arxiv.org/pdf/2606.02578)  

**Abstract**: Recent multimodal large language models have demonstrated strong reasoning ability, yet their reliability as automated evaluators remains limited by a critical weakness: when visual evidence conflicts with textual cues, MLLM judges tend to reward plausible narratives over perceptually correct answers. We identify and systematically analyze this phenomenon, which we term Perceptual Judgment Bias. Through controlled visual perturbations, existing multimodal judges frequently anchor on the response text instead of their own visual perception, leading to inconsistent and non-verifiable evaluations. To address this issue, we introduce the Perceptually Perturbed Judgment Dataset, which constructs minimally edited counterfactual responses that isolate perceptual errors and enable verifiable supervision. Building on this dataset, we develop a unified training framework that combines a structured GRPO-based reward with a batch-ranking objective, achieving coherent global ordering without explicit pairwise labels. Experiments across diverse MLLM-as-a-Judge benchmarks show that our approach substantially improves perceptual fidelity, ranking coherence, and alignment with human evaluation. Our results establish a scalable and generalizable pathway for training multimodal judges that are perceptually grounded, interpretable, and robust to visual-reasoning conflicts. 

---
# AdaCodec: A Predictive Visual Code for Video MLLMs 

**Authors**: Haowen Hou, Zhen Huang, Zheming Liang, Qingyi Si, Chenglin Li, Shuai Dong, Kele Shao, Ruilin Li, Dianyi Wang, Nan Duan, Jiaqi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.02569)  

**Abstract**: Video is temporally redundant: adjacent frames usually share most objects, background, and layout. Yet existing video multimodal large language models (video MLLMs) usually encode each sampled frame as an independent RGB image, causing visual tokens to repeat content already present in earlier frames. This suggests a more direct video interface: send a full reference frame only when the scene cannot be predicted well from prior context, and otherwise transmit a compact description of inter-frame changes. We call this interface a \emph{predictive visual code}, and instantiate it for video MLLMs as \textbf{AdaCodec}. AdaCodec spends full visual tokens on a reference frame only when its conditional predictive cost is high; otherwise, it encodes inter-frame changes, including motion and prediction residuals, as compact P-tokens. Across all eleven benchmarks, AdaCodec improves over the Qwen3-VL-8B per-frame RGB baseline at a matched visual-token budget. Even at $1/7$ the budget, AdaCodec with 32k tokens surpasses the 224k baseline on all long-video benchmarks; on five general-video benchmarks, it raises the average score while substantially cutting time-to-first-token from 9.26s to 1.62s. 

---
# Permissive Safety Through Trusted Inference: Verifiable Belief-Space Neural Safety Filters for Assured Interactive Robotics 

**Authors**: Haimin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2606.02562)  

**Abstract**: Autonomous robots that interact with people must make safe and efficient decisions under human-induced uncertainty, such as their preferences, goals, competency, and willingness to cooperate. Safety filters are a popular approach for ensuring safety in interactive robotics, since their modular design separates safety from performance, allowing robots to operate safely around people with minimal impact on task efficiency. While traditional safety filters typically operate only in the physical space, neglecting the robot's ability to learn and adapt online, the recently proposed belief-space safety filter (BeliefSF) reasons about robot safety in closed-loop with runtime inference that actively reduces the robot's uncertainty online, thereby reducing conservativeness in filtering. However, providing formal safety guarantees for robots deploying BeliefSF remains a significant challenge due to errors in runtime inference and neural approximation of safety filters required to handle the high dimensionality of belief spaces. In this paper, we propose an algorithmic approach to certify high-probability safety of BeliefSF using conformal prediction, while explicitly accounting for the reliability of the robot's runtime inference module. Our method leverages the structure of belief-space safety filtering by focusing verification on a region where inference is expected to be reliable. It preserves the simplicity and sample complexity of standard conformal prediction, yet can certify a substantially less conservative safety filter. Through a simulated human-vehicle interaction benchmark, we show that our approach verifies a significantly more permissive belief-space safety filter than a standard conformal prediction baseline. 

---
# From Layers to Submodules: Rethinking Granularity in Replacement-Based LLM Compression 

**Authors**: Elia Cunegatti, Marcus Vukojevic, Erik Nielsen, Giovanni Iacca  

**Link**: [PDF](https://arxiv.org/pdf/2606.02559)  

**Abstract**: Post-training compression of Large Language Models (LLMs) removes entire architectural components, either deleting them or replacing them with fitted modules. Existing replacement-based methods share two design constraints: full-layer granularity and contiguous selection. We argue that this is overly restrictive: in fact, redundancy in pretrained transformers is not confined to contiguous regions, nor does it evenly distribute between Attention and FeedForward outputs, implying that different strategies best approximate different submodule types and that removable components need not cluster within contiguous depth ranges. Based on this intuition, we introduce SubFit (Submodule-level Fitted residual replacement), which compresses LLMs at the submodule level: Attention and FeedForward submodules are selected non-contiguously, and each receives its own lightweight fitted residual bypass. SubFit operates post-training and requires only calibration data. Across ten LLMs (five base, five instruction-tuned), five sparsity levels from 12.5% to 37.5%, and four replacement-based baselines, SubFit achieves the best aggregate perplexity-accuracy trade-off across the evaluated sparsity levels, with larger gains under aggressive compression. At 25% sparsity, it retains 84.6% of dense downstream accuracy and incurs 2.42x perplexity degradation, against 81.6% and 4.34x for the strongest baselines, while delivering measurable inference speedup and KV-cache savings. Code is available at this https URL. 

---
# Modeling Depth Ambiguity: A Mixture-Density Representation for Flying-Point-Free Depth Estimation 

**Authors**: Siyuan Bian, Congrong Xu, Jun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2606.02552)  

**Abstract**: Despite advances in depth estimation, flying points remain a persistent failure mode: near object boundaries, depth estimators often predict spurious 3D points in the empty space between foreground and background surfaces. We trace this artifact to a standard modeling choice: assigning each pixel a single depth hypothesis. At boundaries, a pixel can straddle a foreground and a background surface, so its true depth is ambiguous between the two. A model that predicts a single depth cannot keep both possibilities, so training instead pulls the prediction toward an intermediate depth that lies on neither surface. We address this with MDA, a mixture-density representation that lets the model predict multiple depth hypotheses and their associated probabilities for each pixel. Near boundaries, different hypotheses can align with different surfaces, and the decoded depth is selected from one of these hypotheses rather than placed in the empty space between them. Across different backbones, MDA substantially improves boundary reconstruction and largely removes flying-point artifacts even under severe input blur, while adding negligible runtime overhead. The same mixture-density framework naturally extends to transparent objects, where it predicts multiple depth layers at transparent pixels, and to sky regions, where a dedicated component separates the unbounded sky from finite-depth regions, producing flying-point-free skylines. Project Page: this https URL. 

---
# SimSD: Simple Speculative Decoding in Diffusion Language Models 

**Authors**: Junxia Cui, Haotian Ye, Runchu Tian, Hongcan Guo, Jinya Jiang, Haoru Li, Chaojie Ren, Yiming Huang, Kaijie Zhu, Zhongkai Yu, Kun Zhou, Jingbo Shang  

**Link**: [PDF](https://arxiv.org/pdf/2606.02544)  

**Abstract**: Diffusion large language models (dLLMs) have recently emerged as a promising alternative to autoregressive (AR) LLMs, offering faster inference through parallel or blockwise decoding. However, their masked language modeling formulation remains incompatible with standard token-level speculative decoding, one of the most effective acceleration techniques for AR models. In AR decoding, the causal mask preserves temporally valid token-level contexts, enabling a target model to verify multiple drafted tokens in a single forward pass. In contrast, dLLMs rely on mask tokens and bidirectional attention, causing the effective context to change across denoising steps and preventing direct token-level speculative verification. To bridge this gap, we propose a simple but effective speculative decoding algorithm for diffusion language models, named SimSD, which mainly adopts a plug-and-play masking strategy that equips dLLMs with temporally valid token-level contexts for speculative decoding. Our method explicitly introduces reference tokens from draft-model predictions and designs an attention mask that regulates their interaction with current-step tokens, allowing dLLMs to compute valid logits for drafted tokens in a single forward pass. This restores the key verification ability provided by causal masking in AR models while preserving the parallel decoding advantages of dLLMs. The proposed method is training-free and can be flexibly integrated with other acceleration techniques such as KV cache and blockwise decoding. Experiments on SDAR-family dLLMs across four benchmarks show that our method achieves up to 7.46x higher decoding throughput while maintaining and even improving average generation quality. 

---
# Why Not Hyperparameter-Friendly Optimisation? A Monotonic Adaptive Norm Rescaling Approach For Long-Tailed Recognition 

**Authors**: Shuo Zhang, Chenqi Li, Tingting Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2606.02526)  

**Abstract**: Long-tailed recognition poses a significant challenge for deep learning. The two-stage decoupling paradigm, which separates representation learning from classifier retraining, offers a promising solution. During the classifier retraining stage, adaptive norm rescaling is a popular technique. It adjusts the per-class weight norms via parameter regularization, which inevitably introduces hyperparameters. However, many studies report that long-tailed recognition is sensitive to these hyperparameters, as their setup significantly impacts performance. In this paper, we first provide a class-conditional distribution perspective to support norm rescaling methods. Furthermore, we propose a simple but effective approach called Self-Adaptive Monotonic Normalization (SAMN). SAMN avoids the need for parameter regularization. It directly enforces monotonicity on per-class weight norms using the Pool Adjacent Violators Algorithm, making the method hyperparameter-friendly. SAMN is a universal strategy that integrates seamlessly with other methods for enhanced performance. Experiments on benchmark datasets demonstrate that our method significantly boosts long-tailed recognition performance, often achieving state-of-the-art results. 

---
# Moment-Video: Diagnosing Temporal Fidelity of Video MLLMs on Momentary Visual Events 

**Authors**: Xiaolin Liu, Yilun Zhu, Xiangyu Zhao, Xuehui Wang, Yan Li, Xin Li, Haoyu Cao, Xing Sun, Shaofeng Zhang, Xu Yang, Zhihang Zhong, Xue Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.02522)  

**Abstract**: Video multimodal large language models (MLLMs) have made rapid progress on general and long-form video understanding, yet their ability to preserve brief answer-critical visual evidence remains underexplored. Many practical questions are determined by momentary visual events: localized actions or state transitions that may last only a few frames. Such evidence can be skipped by sparse frame sampling, suppressed by visual-token compression, or diluted by coarse temporal aggregation, causing failures that language-side reasoning cannot reliably recover. We introduce Moment-Video, a benchmark for diagnosing the temporal fidelity of video MLLMs through momentary visual event understanding. Each question is grounded in a localized, visually observable, and sampling-sensitive event, requiring models to notice, count, describe, or reason about transient evidence rather than rely on persistent objects, global scene context, or language priors. Moment-Video contains 1,000 human-verified video-QA pairs across 7 domains and 25 fine-grained subcategories, covering four task types: Temporal Occurrence, Temporal Counting, Action Description, and Temporal Reasoning. We evaluate 33 proprietary and open-source MLLMs on Moment-Video. The best-performing model, Seed-2.0-Pro, achieves only 39.6% overall accuracy, while most open-source models remain below 25%, revealing a substantial gap in momentary visual event understanding. Diagnostic analyses show that denser frame sampling improves some models but does not eliminate the bottleneck, and longer videos introduce stronger temporal-localization challenges. These findings suggest that current video MLLMs still lack temporally faithful representations for capturing, preserving, and using brief but decisive visual evidence. 

---
# Monitoring Agentic Systems Before They're Reliable 

**Authors**: Marisa Ferrara Boston, Glen Hanson, Effi Georgala, JD Hudgens, Heather Frase  

**Link**: [PDF](https://arxiv.org/pdf/2606.02494)  

**Abstract**: Agentic systems entering production typically operate as partially integrated assemblies where structural defects, not task-level errors, dominate the failure landscape. At this maturity level, task-level error detection may be infeasible: structural failure modes mask the signal that task-level monitors are designed to this http URL present a monitoring and triage methodology that decomposes agentic system evaluation into three dimensions (quality, suitability, efficiency) at three monitoring scopes (within-run, cross-run, structural), using variance as a characterization signal. Findings are routed through severity classification adapted from FMEA, concentrating human attention on the subset that warrants investigation. We evaluate on a synthetic testbed of 220 runs across 120 document bundles with controlled error this http URL results emerge. Monitor scope determines failure type: within-run monitors surface deterministic stage defects (CV = 0.02), cross-run monitors surface stochastic integration consequences (CV = 1.25, 24% at L2), and a structural monitor identifies an integration gap with perfect consistency (CV = 0.00). Injected task-level errors are indistinguishable from clean baselines, confirming structural defects mask task-level signal. Deterministic triage routes 97% of findings to automated tracking, leaving the 2% reflecting variable behavior for human this http URL propose, on Stage 1 evidence, a maturity-staging model in which monitoring transitions from structural characterization to error detection to reliability tracking as integration defects resolve. The taxonomy, CV-based scope characterization, and severity model transfer architecturally to document-driven, multi-stage agentic workflows in regulated industries; specific calibrations are domain-specific. Deploy monitoring early: the first thing it finds is the most important thing to fix. 

---
# Ghost Tool Calls: Issue-Time Privacy for Speculative Agent Tools 

**Authors**: Bardia Mohammadi, Lars Klein, Akhil Arora, Laurent Bindschaedler  

**Link**: [PDF](https://arxiv.org/pdf/2606.02483)  

**Abstract**: Tool-augmented language agents speculatively issue likely future tool calls to hide latency, but those calls leak inferred user intent to external services before the agent commits to the branch. Every external observer that received the call retains the disclosure after the agent abandons the branch. Timing is the issue, not authorization: no commit-time cleanup, read-only restriction, or access-control allow-list unsends what an observer already holds. We call these invocations ghost tool calls and propose Speculative Tool Privacy Contracts, a runtime abstraction that treats observation before commitment as a first-class effect, distinct from state mutation. We implement the contracts in a prototype runtime and evaluate twelve policies across three corpora. Speculative dispatch increases what an observer can infer about user intent; post-hoc filters, read-only restrictions, and access-control allow-lists leave that inference intact; only issue-time policies that change or suppress the speculative call's argument or destination projection before dispatch reduce it. 

---
# Learning When to Translate for Multilingual Reasoning 

**Authors**: Deokhyung Kang, Hyounghun Kim, Gary Geunbae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2606.02465)  

**Abstract**: Reasoning language models (RLMs) achieve strong performance on complex reasoning tasks, but still exhibit substantial multilingual reasoning gaps, largely due to language-understanding failures in non-English inputs. English translation can mitigate these failures by expressing non-English inputs in a form that RLMs can more reliably interpret, yet translating every input is unnecessary when the model can reason reliably from the original query. To address this challenge, we propose Luar, a Language Understanding Boundary-aware Reinforcement Learning framework that trains RLMs to selectively invoke translation when direct understanding is unreliable. Luar trains the model to choose between solving the original input directly and reasoning over its English translation, encouraging translation only when translator-augmented reasoning is expected to substantially outperform direct reasoning. Across multilingual reasoning benchmarks, Luar outperforms standard GRPO and other training-based baselines, with particularly large gains on low-resource languages. Further analysis shows that Luar avoids unnecessary translation in cases where direct reasoning is sufficient, while extending its translator-call behavior to unseen low-resource languages. Together, our work suggests a selective approach to multilingual reasoning: RLMs can learn to invoke translation only when their direct understanding is unreliable. The project will be made publicly available at this https URL 

---
# MASER: Modality-Adaptive Specialist Routing for Embodied 3D Spatial Intelligence 

**Authors**: Hilton Raj, Vishnuram AV  

**Link**: [PDF](https://arxiv.org/pdf/2606.02463)  

**Abstract**: In 3D environments, Embodied Agents answer spatially relevant questions through reasoning from a mixture of modalities including natural language, RGB images, point clouds, depth maps and camera poses. Existing Vision-Language models (VLMs) are fine-tuned over a single modality. This completely ignores the question semantics which may favor a different modality than the finetuned modality. To address this, we propose MASER (Modality-Adaptive SpEcialist Routing), a lightweight framework that trains five different modality adapters of a shared VLM backbone and learns a neural routing policy that selects the best adapter based on the question during inference. We encode each question with a frozen sentence transformer and pass the embedding through a small Multi-layer Perceptron (MLP) trained on oracle adapter-accuracy labels. We evaluate our methodology over the Open3D-VQA benchmark and our evaluations show that no single modality is universally optimal -- point-cloud answers are best in 51.5% of cases. MASER routes with 51.3% oracle agreement, outperforming a Random-Forest ablation (43.5%), with only a single adapter call per question. 

---
# Initialization is Half the Battle: Generating Diverse Images from a Guidance Potential Posterior 

**Authors**: Xiang Li, Dianbo Liu, Kenji Kawaguchi  

**Link**: [PDF](https://arxiv.org/pdf/2606.02453)  

**Abstract**: Despite the remarkable fidelity of generative models, they frequently suffer from mode collapse. Existing strategies for enhancing diversity predominantly focus on intervening during the generation trajectory. We identify a critical oversight that the standard Gaussian initialization often causes trajectories to collapse into dominant modes because it is agnostic to the guidance potential landscape. In this work, we formulate selecting the initial noise from a guidance potential posterior, which effectively re-weights the prior towards diversity-rich regions. To sample from this distribution efficiently, we introduce Diversity-inducing Initialization (DivIn), which leverages Langevin dynamics to actively navigate the initialization landscape, steering initial noise away from collapsing regions while anchoring them to the valid data manifold. Our method serves as an inference-time diversity enhancement compatible with both diffusion and flow matching models. Extensive experiments show that DivIn exhibits a superior performance in both class-to-image and text-to-image scenarios. Furthermore, we highlight that as DivIn is orthogonal to trajectory-based methods, combining them significantly expands the diversity-quality Pareto frontier beyond what either achieves in isolation. 

---
# PaSBench-Video: A Streaming Video Benchmark for Proactive Safety Warning 

**Authors**: Yusong Zhao, Yuejin Xie, Youliang Yuan, Junjie Hu, Jitian Guo, Yujiu Yang, Pinjia He  

**Link**: [PDF](https://arxiv.org/pdf/2606.02443)  

**Abstract**: Between the first visible sign of danger and the moment an accident occurs, there is often a window where intervention remains possible. Video-capable multimodal large language models (MLLMs) could serve as always-on safety monitors that issue warnings during this window. Yet current benchmarks do not test this ability: they rely on static inputs, ignore timing precision, and omit false-positive measurement on safe scenes. We present PaSBench-Video, a 740-video benchmark with 481 risk and 259 no-risk videos across four domains: driving, healthcare, daily life, and industrial production. Risk videos are annotated with frame-level risk onset and accident boundaries. A model must observe the video causally and produce a warning that is both temporally calibrated and content-correct. Testing 13 MLLMs, we find that no model exceeds 20.0% on our strictest metric, and recall is tightly coupled with false-positive rate, with Pearson correlation 0.64: higher detection comes only at the cost of triggering warnings on the majority of safe clips. Performance splits sharply by domain: models achieve moderate recall at low false-positive rates in daily life, where risks are inherently anomalous, yet fire indiscriminately in driving, where routine and hazardous scenes look alike. These results indicate that current models rely on scene-level activity cues rather than reasoning about emerging harm. 

---
# ODTQA-FoRe: An Open-Domain Tabular Question Answering Dataset for Future Data Forecasting and Reasoning 

**Authors**: Zhensheng Wang, Xiaole Liu, Wenmian Yang, Kun Zhou, Yiquan Zhang, Weijia Jia  

**Link**: [PDF](https://arxiv.org/pdf/2606.02433)  

**Abstract**: The rapid development of LLMs has significantly advanced tabular question answering, but most systems cannot perform future-oriented numerical prediction. To address this gap, we introduce a novel task, Open-Domain Tabular Question Answering for Future Data Forecasting and Reasoning, and propose the first dataset to cover time-series forecasting and forecast-based reasoning scenarios using real estate data. This task poses challenges in retrieving precise historical data, overcoming the forecasting limitations of LLMs, and standardizing responses for diverse queries. To solve the above challenges, we propose TimeFore, an LLM agent-based framework that decomposes the problem into three collaborative roles: a Retriever autonomously generates SQL to fetch data, a Forecaster invokes external time-series models for higher accuracy, and an Analyzer synthesizes the results to construct a precise and consistent final answer. Extensive experiments demonstrate the effectiveness of our TimeFore. 

---
# Not All Errors Are Equal: A Systematic Study of Error Propagation in Large Language Model Inference 

**Authors**: Yafan Huang, Sheng Di, Guanpeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.02430)  

**Abstract**: Large language models (LLMs) are increasingly integrated into high-performance computing (HPC) workflows, accelerating scientific discovery through diverse perspectives such as code generation and domain-specific decision-making. Yet, how soft errors propagate and affect LLM inference remains largely unexplored. To bridge this gap, we present a comprehensive study on error propagation in LLM inference, enabled by our proposed LLMFI, a configurable and deterministic fault-injection framework. Using LLMFI, we systematically inject faults across three open-weighted LLMs and thirteen representative tasks, covering reasoning, multilingual, mathematical, and coding domains. In addition, we conduct fine-grained case studies that reveal critical vulnerability patterns. Overall, our study yields 17 takeaways that advance the understanding of error propagation in LLM inference and introduces four low-overhead directions to improve reliability through software-only modification, offering practical guidance for future error detection and mitigation. 

---
# GC-MoE: Genomics-Guided Cell-Type-Specific Mixture of Experts for Histology-Based Single-Cell Spatial Transcriptomics 

**Authors**: Kaito Shiku, Ahtisham Fazeel Abbasi, Ryoma Bise, Yuichiro Iwashita, Kazuya Nishimura, Andreas Dengel, Muhammad Nabeel Asim  

**Link**: [PDF](https://arxiv.org/pdf/2606.02424)  

**Abstract**: Histology-based single-cell spatial transcriptomics (ST) estimation aims to predict gene expression for individual cells from histopathological images and cell locations, reducing the need for costly single-cell ST measurements. Unlike existing histology-to-ST methods that mainly predict spot-level profiles for local regions containing multiple cells, this task requires modeling cell-to-cell expression variability, which is strongly structured by cell type. We propose Genomics-Guided Cell-Type-Specific Mixture-of-Experts (GC-MoE), which estimates cell-type probabilities with a routing network and softly combines cell-type-specific experts for gene expression prediction. To further encode cell-type-dependent gene programs, we introduce the Cell-Type-Specific Co-Expression-Aware Predictor (CAP), together with a lightweight Cell-to-Cell Interaction Attention (C2CA) module for neighboring-cell context. Experiments and ablations on public single-cell ST datasets show consistent improvements over existing single-cell and adapted spot-level baselines. 

---
# Evolutionary Discovery of Bivariate Bicycle Codes with LLM-Guided Search 

**Authors**: Juan Cruz-Benito, Andrew W. Cross, David Kremer, Ismael Faro  

**Link**: [PDF](https://arxiv.org/pdf/2606.02418)  

**Abstract**: Quantum LDPC code discovery requires searching large algebraic design spaces while reliably certifying the parameters and equivalence classes of any candidates found. We introduce an LLM-guided evolutionary workflow in which language models mutate Python programs that generate bivariate-bicycle and perturbed bivariate-bicycle code ansätze. Across five campaigns, the system performed approximately 1{,}650 evolutionary iterations, screened about $2 \times 10^5$ candidate codes, and required ${\sim}140$ hours of computation and ${\sim}$US\$400 in LLM inference cost. Candidate codes are evaluated through a staged validation pipeline combining $\mathrm{GF}(2)$ rank computation, distance estimation and certification, mixed-integer linear programming, BLISS Tanner-graph deduplication, decomposability analysis, and local-Clifford equivalence checks. At block length $n \leq 360$, the workflow identifies 465 distinct candidate codes: 97 CSS bivariate-bicycle codes and 368 non-CSS perturbed variants. The CSS search recovers known high-performing codes and finds new finite-length representatives, including an indecomposable [[288,16,12]] code and higher-weight codes with up to $k = 50$ at distance $d = 8$. The non-CSS search produces perturbed codes matching the gross-code figure of merit at [[144,12,12]], along with additional high-distance candidates reported as certified values or upper bounds according to MILP status. Overall, these results show that LLM-guided program evolution can serve as a practical tool for structured quantum-code discovery when paired with independent evaluation. 

---
# AutoForest: Automatically Generating Forest Plots from Biomedical Studies with End-to-End Evidence Extraction and Synthesis 

**Authors**: Massimiliano Pronesti, Angelo Miculescu, Mohsin Kapdi, Paul Flanagan, Oisín Redmond, Joao Bettencourt-Silva, Gurdeep Mannu, Spiros Denaxas, Rui Bebiano Da Providencia E Costa, Anya Belz, Yufang Hou  

**Link**: [PDF](https://arxiv.org/pdf/2606.02403)  

**Abstract**: Systematic reviews rely on forest plots to synthesise quantitative evidence across biomedical studies, but generating them remains a fragmented and labour-intensive process. Researchers must interpret complex clinical texts, manually extract outcome data from trials, define appropriate interventions and comparators, harmonise inconsistent study designs, and carry out meta-analytic computations-typically using specialised software that demands structured inputs and domain expertise. While recent work has demonstrated that large language models can extract study-level data from unstructured text, no existing system automates the complete pipeline from raw documents to synthesised forest plots. To address this gap, we introduce AutoForest, the first end-to-end system that generates publication-ready forest plots directly from biomedical papers. Given one or more study papers, AutoForest automatically suggests ICO (Intervention, Comparator, Outcome) elements, extracts outcome data, performs statistical synthesis, and renders the final forest plot. We describe the system architecture, user interface and demonstrate its effectiveness on real-world examples through a user study involving clinicians, showing how AutoForest can accelerate evidence synthesis and substantially lower the barrier to conducting meta-analyses. 

---
# Policy and World Modeling Co-Training for Language Agents 

**Authors**: Ning Lu, Baijiong Lin, Shengcai Liu, Jiahao Wu, Haoze Lv, Yanbin Wei, Lingting Zhu, Shengju Qian, Xin Wang, Ying-Cong Chen, Qi Wang, Ke Tang  

**Link**: [PDF](https://arxiv.org/pdf/2606.02388)  

**Abstract**: Reinforcement learning (RL) improves large language model (LLM) agents by teaching them which actions lead to high rewards, but provides little supervision on what those actions do to the environment. World modeling (WM) can fill this gap, yet existing approaches often require separate simulators, extra training stages, or additional inference-time computation. We observe that on-policy RL rollouts already contain the needed signal: each transition pairs an action with its resulting next observation. Based on this observation, we propose PaW, a Policy and World modeling co-training framework that adds auxiliary WM supervision to the same policy during RL, without changing the inference paradigm. To make auxiliary WM supervision informative and stable, PaW introduces three components: action-entropy-based WM data selection, noise-tolerant WM loss, and reward-adaptive loss balancing. Experiments on three agentic task benchmarks show consistent improvements over strong RL baselines across models and RL algorithms. These results suggest that standard RL rollouts are a practical source of WM supervision for language-agent training. 

---
# SPADE-Bench: Evaluating Spontaneous Strategic Deception in Agents via Plan-Action Divergence 

**Authors**: Yuyan Bu, Haowei Li, Qirui Zheng, Bowen Dong, Kaiyue Yang, Jiaming Ji, Yingshui Tan, Wenxin Li, Yaodong Yang, Juntao Dai  

**Link**: [PDF](https://arxiv.org/pdf/2606.02380)  

**Abstract**: As LLM-based agents expand their operational scope, reliability becomes a prerequisite for real-world deployment. However, in practical applications, human users cannot monitor every immediate behavior; instead, the execution process often remains a black box, leaving users dependent solely on the agent's self-reported updates. This opacity creates a critical risk: agents may present observer-facing reports that diverge from their executed actions, rendering the system uncontrollable, especially in high-stakes autonomous scenarios. We term such self-reported plan-action divergence as agent deception. To assess this, we introduce SPADE-Bench, a benchmark designed to evaluate spontaneous plan-action divergence. Unlike prior deception benchmarks, SPADE-Bench simultaneously integrates actual tool execution and controlled pressure scenarios. This design ensures ecological validity and rigorously distinguishes strategic deception from mere hallucination through controlled plan-action comparisons under pressure. Experiments across mainstream models confirm that agent deception is a genuine and pressing issue in tool-use contexts. By providing a comprehensive and robust evaluation framework, SPADE-Bench fills a critical gap in agent safety, facilitating the community's progress toward building trustworthy and controllable autonomous systems. 

---
# When Do Attention Circuits Form? Developmental Trajectories of Capability and Attention-Sink Emergence Across Three 1B-ClassArchitectures 

**Authors**: Yongzhong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.02378)  

**Abstract**: We track the developmental trajectory of attention-head circuit formation across three 1B-class language models spanning two architecture families (dense transformer, mixture-of-experts) and two pretraining corpora (The Pile, DCLM): Pythia 1B, OLMo 1B-0724-hf, and OLMoE 1B-7B-0924. At each of 10 log-spaced revisions per model -- 30 mechanistic-interpretability runs in total -- we apply a participation-ratio (PR) spectral signal and an all-head capability-specific selectivity screen to track induction, previous-token, and BOS-attractor heads as they emerge.
Five findings. (F1) Layers 0 and 1 produce zero BOS-classified heads at every revision in every model: the L0/L1 zero-BOS floor is an architectural property, not a learned outcome. (F2) The whole-model BOS-attractor fraction follows three distinct emergence shapes -- a gradual ramp in Pythia 1B, a sharp phase transition in OLMo 1B (7% to 70% between adjacent checkpoints), and a gradual ramp in OLMoE 1B-7B. (F3) In DCLM models, induction-circuit formation precedes BOS-attractor formation by 10-20x in tokens; capability-circuit formation and attention-sink formation are two transitions, not one. (F4) The capability-specific screen converges to the final induction circuit within 0.3-2% of total training tokens -- circuit identification does not require the final model. (F5) For every final-checkpoint induction head sampled across all three models, per-head PR is elevated at or before the first revision at which that head crosses its capability-selectivity threshold.
The results refine the induction-phase-transition framing: in 1B-class models trained on DCLM, the induction transition and the attention-sink transition are separated by an order of magnitude in tokens and have qualitatively different shapes. 

---
# FOAM: Frequency and Operator Error-Based Adaptive Damping Method for Reducing Staleness-Oriented Error for Shampoo 

**Authors**: Kyunghun Nam, Sumyeong Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2606.02365)  

**Abstract**: Shampoo is attracting considerable attention for its superior performance on large-scale optimization benchmarks; yet it faces a significant practical bottleneck: the prohibitive computational overhead of matrix inversion. To mitigate this, practitioners typically rely on stale preconditioner updates, creating a fundamental trade-off between computational efficiency and optimization fidelity. In this work, we provide a theoretical study of staleness through the complementary lenses of convergence and stability. While staleness improves computational efficiency, it inherently degrades performance and introduces numerical instability. Crucially, we identify that damping, acting as a numerical stabilizer, can effectively suppress these negative effects. Guided by this analysis, we propose FOAM, an adaptive algorithm that stabilizes training by dynamically controlling both the damping factor and the eigendecomposition frequency based on an approximation of the staleness-oriented error. Experimental results demonstrate that FOAM reduces wall-clock time compared to standard Shampoo while maintaining robust convergence. 

---
# Do Multimodal Agents Really Benefit from Tool Use? A Systematic Study of Capability Gains 

**Authors**: Garvin Guo, Donglei Yu, Yu Chen, Xiang Wang, Shuai Li, Xinpei Zhao, Huaxing Liu, Qinghao Wang, Minpeng Liao  

**Link**: [PDF](https://arxiv.org/pdf/2606.02357)  

**Abstract**: Tool-augmented multimodal agents show strong benchmark gains, often taken as evidence that agents have learned to use tools. We argue that this interpretation can be premature: a tool-call trace alone does not show whether the tool supplied answer-critical information. We study two representative ``thinking with images'' agents, Thyme and DeepEyesV2, across real-world understanding, OCR, chart understanding, and mathematical reasoning. Each agent is compared with its Tool-Free counterpart and with a Pure-Text Reasoner trained from the same source pool without tool-calling trajectories. Tool access yields little consistent aggregate improvement, does not reliably reduce generated-token cost, and leaves only a small tool-only solved set: 93% of DeepEyesV2's tool-solved problems and 96% of Thyme's are also solved by at least one non-tool setting. Mechanism ablations further show that the full tool-use loop does not consistently outperform either the tool-call format or the returned execution result alone. In the settings we study, the analyzed agents appear to learn tool-calling patterns more reliably than tool-contributed capabilities, suggesting that evaluation should distinguish tool availability from whether tools actually expand what agents can solve. 

---
# Repurposing Adversarial Perturbations for Continual Learning: From Defense to Active Alignment 

**Authors**: Ran Liu, Min Yu, Mingqi Liu, Jianguo Jiang, Gang Li, Rongsheng Li, Ning Li, Zhen Xu, Weiqing Huang, Ming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.02322)  

**Abstract**: In dynamic environments, large language models need to keep adapting to new tasks, but continual learning often suffers from forgetting, limited transfer, and vulnerability to adversarial perturbations. To address this, we present AdvCL, which repurposes adversarial perturbations as a geometric control signal for stable continual adaptation. AdvCL combines three plug-in modules: Intra-Smooth promotes local smoothness via small adversarial perturbations; Proto-Clip uses similarity clipping to prevent excessive alignment to current task prototype; and Inter-Align applies directional alignment toward previous task prototype to reduce representational gaps. Experiments show consistent gains in both standard performance and robustness, with lower forgetting and stronger transfer. We further analyze key mechanisms by quantifying the sensitivity of Intra-Smooth to perturbation settings and the effect of Inter-Align on task similarity and geometric distance. In summary, the modules provide complementary gains when combined, and each can also be integrated individually into diverse CL paradigms, including replay, regularization, and dynamic architectures, thereby offering a geometric control mechanism for continual learning. 

---
# SeClaw: Spec-Driven Security Task Synthesis for Evaluating Autonomous Agents 

**Authors**: Hao Cheng, Changtao Miao, Tianle Song, Yin Wu, He Liu, Erjia Xiao, Junchi Chen, Xiaoyu Shi, Yichi Wang, Jing Yang, Taowen Wang, Jinhao Duan, Mengshu Sun, Peiyan Dong, Xuan Shen, Yang Cao, Renjing Xu, Kaidi Xu, Jindong Gu, Bo Zhang, Jize Zhang, Chenhao Lin, Philip Torr, Chao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2606.02302)  

**Abstract**: Autonomous LLM agents increasingly operate in stateful environments where they access tools, files, memory, and external services. While such capabilities enable complex real-world workflows, they also introduce security risks that are difficult to capture with existing evaluations. Current agent security benchmarks often rely on manually curated tasks, provide limited coverage of emerging threats, and focus primarily on final outcomes rather than the execution processes that lead to unsafe behavior. We introduce SeClaw, a framework that combines specification-driven security task synthesis with execution-based security evaluation for Autonomous agents. Spec-driven security task synthesis enables scalable and controllable construction of security tasks from structured risk specifications, while SeClaw docker provides a standardized testbed for evaluating agent behavior under diverse safety-risk scenarios. The benchmark covers risks arising from resources, user tasks, environments, and intrinsic agent behaviors, and supports trajectory-aware assessment of unsafe actions beyond final responses. By bridging systematic task synthesis and reproducible security evaluation, SeClaw provides a practical foundation for measuring, diagnosing, and comparing security failures in autonomous LLM agents. The code is available at this https URL. 

---
# Quantitative Movement Testing: Measuring Patient Movements from a Single Smartphone Video 

**Authors**: Pranav Mahajan, Amanda Wall, Eleonora Maria Camerone, Julie Stebbins, Eoin Kelleher, Shuangyi Tong, Annina Schmid, Katja Wiech, Anushka Irani, Ben Seymour  

**Link**: [PDF](https://arxiv.org/pdf/2606.02301)  

**Abstract**: Chronic pain diminishes quality of life by decreasing functional ability, yet objectively measuring this functional impact remains challenging in real-world settings. While optical motion capture provides high precision for assessing altered movement quality, it is costly and restricted to laboratory environments. We aimed to develop and validate Quantitative Movement Testing (QMT), a computer vision pipeline extracting 3D kinematic biomarkers from standard monocular smartphone video, balancing clinical accessibility with biomechanical accuracy. We validated the QMT pipeline, utilising deep learning-based 3D pose-estimation, against gold-standard optical motion capture in healthy controls (N=13). Following leave-one-subject-out calibration to correct systematic bias, we deployed QMT in two prospective clinical cohorts to assess real-world utility: a pre- and post-intervention trial for fibromyalgia patients, and a 30-day longitudinal at-home monitoring study of chronic sciatica patients and healthy controls. In laboratory validation, QMT extracted clinical kinematic metrics with high agreement to optical motion capture, yielding strong correlations (r > 0.85) and low mean absolute errors. QMT demonstrated high test-retest reliability (r > 0.86) in fibromyalgia patients and successfully tracked day-to-day movement fluctuations in chronic sciatica. While real-world home settings introduced higher measurement variance than lab settings, QMT found group-level differences between healthy controls and sciatica patients based entirely on remote recordings. Monocular 3D pose estimation offers a scalable alternative to traditional assessments. QMT provides an objective, accessible biomarker for tracking disease progression and treatment response in clinical trials, though further research is needed to optimise reliability in home environments. 

---
# CityTrajBench: A Unified Benchmark for City-Scale Vehicle Trajectory Generation 

**Authors**: Shibo Zhu, Xiaodan Shi, Dayin Chen, Yuntian Chen, Haoran Zhang, Tianhao Wu, Jinyue Yan  

**Link**: [PDF](https://arxiv.org/pdf/2606.02287)  

**Abstract**: Urban trajectory generation is a fundamental task for transportation simulation, urban planning, and mobility analytics. However, systematic comparison across trajectory generation methods remains difficult because existing studies often rely on different datasets, preprocessing pipelines, trajectory representations, and evaluation metrics. This fragmentation makes it unclear whether reported performance differences arise from the generation mechanism itself or from inconsistent experimental protocols. To address this issue, we present CityTrajBench, a unified benchmark framework and protocol for city-scale vehicle trajectory generation. CityTrajBench standardizes data ingestion, trajectory normalization, feature construction, model adaptation, map-aware post-processing, model selection, and multi-level evaluation under a common setting. It supports heterogeneous generators, including statistical baselines, VAE-based, GAN-based, diffusion-based, and flow-matching-based models, and evaluates them on three real-world urban trajectory datasets. The benchmark measures global spatial realism, trip-level distribution fidelity, trajectory-level geometric similarity, conditional mobility consistency, and efficiency. Experiments reveal clear trade-offs across model families: DiffTraj is strongest on trajectory-level geometric fidelity, DiffRNTraj is competitive on structure-sensitive global realism, and TrajFlow provides a strong balance across realism, quality, conditional consistency, and efficiency. Meanwhile, a simple Markov baseline remains competitive on coarse-grained trip and local-movement statistics. These findings show that urban trajectory generation quality is inherently multi-objective, that no single model dominates all criteria equally, and that CityTrajBench provides a reproducible benchmark protocol and testbed for future research on urban mobility generation. 

---
# Cross-modal linkage risk in clinical vision-language models 

**Authors**: Soroosh Tayebi Arasteh, Mahshad Lotfinia, Sven Nebelung, Daniel Truhn  

**Link**: [PDF](https://arxiv.org/pdf/2606.02276)  

**Abstract**: Vision-language models (VLMs) trained on paired chest radiographs and radiology reports learn a shared embedding space that can preserve instance-level image-report correspondence. This poses a privacy risk in settings where radiographs and reports are deliberately kept separate after acquisition, such as image-only data sharing or access-controlled reports, because a de-identified image may be re-linked to its original narrative report through cosine similarity alone. We formalized this as image-to-report retrieval and used public paired cohorts, in which the true pairing is known by design, as ground-truth benchmarks to audit the risk rather than as the privacy scenario. Evaluating VLMs of increasing clinical specialization on 406,241 paired examples from 126,804 patients across MIMIC-CXR (43,793 held-out pairs) and external CheXpert Plus (29,296 pairs), we found that re-linkage rose systematically with specialization: the strongest VLM retrieved the correct report at 15 times chance at a candidate pool of N = 100, 50 times chance at N = 10,000, and well above chance at full-database scale. The signal persisted under pathology-matched hard negatives that removed disease-label shortcuts, indicating correspondence beyond broad diagnostic categories. To reduce it without retraining, we froze both encoders and applied differentially private optimization only to the projection heads defining the alignment layer (epsilon = 0.34, delta = 6x10-6). This reduced Recall@1 by 61.8% at N = 10,000 on MIMIC-CXR and transferred to CheXpert Plus without retraining, while image-side utility was largely preserved: macro AUROC for linear-probe classification across 14 labels shifted only from 79.63% to 79.43%. Targeted DP finetuning of the shared alignment layer can substantially reduce cross-modal re-linkage without materially degrading the image representations that make these models clinically useful. 

---
# Who Annotates in NLP? A Large-scale Assessment of Human Annotation Reporting between 2018 and 2025 

**Authors**: Maria Kunilovskaya, Gagan Bhatia, Lisa Sophie Albertelli, Yanran Chen, Christian Greisinger, Lotta Kiefer, Christoph Leiter, Subhadeep Roy, Tewodros Achamaleh, Muhammad Arslan Manzoor, Sebastian Pohl, Yufang Hou, Steffen Eger  

**Link**: [PDF](https://arxiv.org/pdf/2606.02255)  

**Abstract**: Human annotation is the empirical foundation of much NLP research, from dataset construction to model evaluation, but papers often leave unclear who produced the annotations and how the annotation process was controlled. We provide the first large-scale, task-level audit of human annotation reporting across major NLP venues, asking which annotation details are documented, which are missing, and how reporting varies across time, topic, venue, and intended use of human judgment. We introduce a unified taxonomy of annotation-reporting practices and validate an LLM-assisted extraction pipeline against Annotated-gold, a human-adjudicated gold standard of 41 papers and 72 annotation tasks, where the best model reaches human-comparable agreement with adjudicated labels, with Krippendorff's alpha of 0.606 versus 0.585 for human-human agreement. Using this pipeline, we construct Annotated-llm, a dataset covering ACL-venue papers from 2018-2025, with 2,667 extracted annotation tasks from 1,603 papers, and find that papers frequently report operational details such as recruitment strategies, annotator expertise, and annotation volume, but often omit details needed to assess annotation validity, including training, language proficiency, compensation, socio-demographics, adjudication, and agreement values, especially in model-evaluation studies. Our results show that annotation reporting in NLP has improved over time but remains uneven, and they establish a scalable framework and bare-minimum reporting recommendations for making human annotation more reliable, reproducible, and interpretable. 

---
# FW-NKF: Frequency-Weighted Neural Kalman Filters 

**Authors**: Adnan Harun Dogan, Berken Utku Demirel, Christian Holz  

**Link**: [PDF](https://arxiv.org/pdf/2606.02251)  

**Abstract**: Robust state estimation is central to robotic autonomy, yet classical Kalman filters struggle with frequency-dependent disturbances and model mismatch such as sensor vibrations, electromagnetic interference, and periodic noise. Although Deep Kalman Filter (DKF) variants extend the Extended Kalman Filtering (EKF) framework by learning latent transitions, they lack explicit mechanisms to suppress band-limited noise components that typically corrupt sensor measurements in real-world scenarios. We introduce the Frequency-Weighted Neural Kalman Filter (FW-NKF), a unified hybrid approach that embeds a causal spectral-shaping operator into the Kalman measurement residual and jointly learns observation, and transition networks. By adapting both the filter spectrum and the latent state representation, FW-NKF attenuates the noise-dominated frequency bands while capturing complex residual structures. We conduct extensive experiments on four heterogeneous benchmarks, including chaotic systems such as multi-dimensional Lorenz systems and full-body inertial pose estimation, and find a reduction in localization error of up to 10% as well as marked improvements in orientation accuracy. Our ablation studies confirm that frequency weighting and deep latent-state modeling contribute to overall performance. 

---
# Towards Resolving Optimization Conflicts Between Image- and Text-Based Person Re-Identification 

**Authors**: Karina Kvanchiani, Timur Mamedov  

**Link**: [PDF](https://arxiv.org/pdf/2606.02242)  

**Abstract**: The joint optimization of image-based (I2I) and text-based (T2I) person re-identification (ReID) is hindered by modality discrepancies and conflicting training objectives, leading to suboptimal shared representations. While I2I ReID focuses on identity-level invariance across images of the same person, T2I ReID is driven by instance-specific textual descriptions tied to unique visual traits. This paper explores the fundamental difference between two ReID tasks and their optimization processes for effective training. Since I2I and T2I ReID are often studied separately, the loss functions optimized for one retrieval setting may negatively affect the representation quality required by the other. Motivated by these findings, we propose a decoupled two-stage training pipeline for learning a shared representation across image and text modalities. The pipeline is based on a single vision encoder that supports both I2I and T2I retrieval while avoiding cross-task interference during training. We provide extensive experiments across multiple configurations, varying domain mixing procedures, learning strategies, and task objectives. We observed that I2I ReID pre-training positively impacts the generalization ability to T2I data. Besides, we find that incorporating textual supervision during the vision encoder training stage enhances both I2I and T2I performance. We believe our insights provide a meaningful step toward unified ReID systems and cross-modal retrieval overall. 

---
# AgentRedBench: Dynamic Redteaming and Integration-Aware Defense for LLM Agents over SaaS Integrations 

**Authors**: Hiskias Dingeto, Will Leeney  

**Link**: [PDF](https://arxiv.org/pdf/2606.02240)  

**Abstract**: Indirect prompt injection in tool-use agents is a concrete production threat: LLM agents read from integrations (third-party services such as Gmail, Salesforce, or Jira accessed through tool calls) whose response content the user neither writes nor controls. Existing benchmarks under-measure the threat: most cover only a handful of integrations with the same attack payload replayed across runs, and open-source guards are trained on chat-style data rather than tool-response content. We introduce AGENTREDBENCH, a dynamic LLM-driven redteaming benchmark of 215 subtle underspecified authorization (attacks at the boundary of what the user's request authorises) scenarios across 24 enterprise integrations in nine functional families and five attack types. Across an eight-model panel (Anthropic, OpenAI, Google), no-guard ASR (attack success rate) ranges from 32% (Claude Sonnet 4.6) to 81% (Gemini 3 Flash). To keep the scenario set out of training corpora and preserve headline ASR meaning over time, we release the codebase, integration schemas, and AGENTREDGUARD model openly; the canonical scenarios are evaluated through a maintainer-mediated channel with immutable versioning. We release AGENTREDGUARD alongside the benchmark: a guard trained on an integration-diverse corpus of adversarial tool-response content. AGENTREDGUARD cuts panel ASR from 69.9% to 2.4% at 0.37% false-positive rate, outperforming every open-source baseline with non-trivial detection (Llama Guard, PromptGuard 2, ProtectAI) on both axes. Cross-integration and cross-attack type holdouts both confirm the gain transfers beyond the training subset. 

---
# Faster Synchronous On-Policy RL via Straggler-Aware Group Sizing 

**Authors**: Azal Ahmad Khan, Ammar Ahmed, Zeshan Fayyaz, Sheng Di, Mingyi Hong, Ali Anwar  

**Link**: [PDF](https://arxiv.org/pdf/2606.02218)  

**Abstract**: Synchronous reinforcement learning methods such as Group Relative Policy Optimization (GRPO) provide stable and reproducible on-policy training, but they are highly vulnerable to stragglers, a single unusually long rollout can delay reward computation and parameter updates for the entire group. This problem becomes more severe as group size increases, creating a tension between the benefits of larger groups and the wall-clock cost of synchronization stalls. We propose Straggler-Aware Group Control (SAGC), a dynamic group-size controller that adapts the training group online based on observed rollout behavior. SAGC formulates group-size selection as an online constrained optimization problem, seeking to retain the benefits of larger groups while controlling the long-term rate of straggler events. Across synchronous GRPO and DAPO training, and on top of both vanilla and strong engineered baselines, SAGC consistently reduces straggler incidence and improves wall-clock efficiency while achieving competitive or better training reward. We further show that these gains transfer to final model quality: SAGC is competitive with or better than the strongest static group-size baseline on downstream reasoning benchmarks, and often produces shorter outputs without any explicit length penalty. These results position dynamic group control as a practical way to make synchronous on-policy RL more efficient and robust. 

---
# Consistency Training while Mitigating Obfuscation via Rate Matching 

**Authors**: Sohaib Imran, Prakhar Gupta, Jannes Elstner, David Demitri Africa  

**Link**: [PDF](https://arxiv.org/pdf/2606.02211)  

**Abstract**: Large language models are often influenced by extraneous input features, such as cues revealing a user's preferred answer. Consistency training reduces this influence by training models to behave similarly across inputs with and without the extraneous feature. However, existing methods train for consistency over entire responses or internal activations, which also constrains whether the model verbalises said extraneous features. We show this leads to obfuscation, where the model learns not to mention a cue while remaining influenced by it, which may undermine monitorability. To address this, we introduce Rate Matching Consistency Training (RMCT), which trains for consistency over selected behavioural properties without constraining how this behaviour is expressed. RMCT matches the rate at which the model exhibits a target behaviour (e.g., following a bias cue) across input perturbations, rather than requiring paired inputs with and without the extraneous feature, extending consistency training to settings where the extraneous features cannot be removed. We evaluate RMCT on sycophancy reduction in two open-weight language models, achieving reductions in bias-following comparable to a standard consistency-training baseline on held-out bias types, while largely preserving the model's tendency to verbalise the bias cue. Further, we find that RMCT is more data-efficient at the expense of being less compute-efficient in our experiments. Overall, RMCT shows that consistency training can improve behavioural robustness without directly trading off against monitorability. 

---
# On the Generalization in Topology Optimization via Sensitivity-Conditioned Bernoulli Flow Matching 

**Authors**: Mohammad Rashed, Duarte F. Valoroso Madeira, Babak Gholami, Caglar Guerbuez, Yunjia Yang, Nils Thuerey  

**Link**: [PDF](https://arxiv.org/pdf/2606.02179)  

**Abstract**: Surrogate models for topology optimization (TO) exhibit highly variable out-of-distribution (OOD) generalization under distribution shifts such as changing loads or boundary conditions, yet the source of this variability remains unclear. We hypothesize that OOD performance is governed by how much information the conditioning signal preserves about the adjoint sensitivity (reduced gradient) that drives classical TO. Modeling the TO pipeline as a causal Markov chain, the Data Processing Inequality establishes that, under this abstraction, the sensitivity field is an information-theoretically optimal conditioning signal for topology prediction. However, computing exact adjoint sensitivities can be expensive or unavailable in practice; we observe that certain physical fields can approximate sensitivities through monotone transformations. To formalize this, we introduce \textbf{pseudo-sensitivities} to characterize which fields enable generalization versus those that are information-poor. We then show that a sensitivity-conditioned Bernoulli flow-matching generator empirically confirms these predictions: conditioning on sensitivities yields state-of-the-art OOD performance, while increasingly distant physical fields degrade toward raw parameter conditioning. Results hold across structural TO benchmarks under load shifts and our new CFD-TO dataset under boundary-condition shifts such as multi-outlet configurations. Code and datasets are available at this https URL . 

---
# Order within Chaos: Capturing Intrinsic Energy Anomalies for AI-Manipulated Image Forgery Localization 

**Authors**: Yiming Wang, Baiqi Wu, Qingming Li, Jiahao Chen, Tong Zhang, Shouling Ji  

**Link**: [PDF](https://arxiv.org/pdf/2606.02178)  

**Abstract**: Recent advancements in generative AI have led to image editing models capable of producing realistic forgeries that evade traditional image forgery localization methods, as these approaches depend on physical noise absent in synthetic data. To address this challenge, we theoretically demonstrate that the diffusion process inherently suppresses local high-frequency variance, creating a statistical energy gap that is distinguishable from the natural entropy of optical imaging. Guided by this insight, we propose FLAME, a unified framework that utilizes a LAD map to capture these intrinsic anomalies, coupled with a parameter-efficient adapter for SAM to achieve precise, pixel-level forgery localization. Furthermore, to bridge the lag between forensic benchmarks and evolving generative models, we introduce EditStream, an automated pipeline for continuous, instruction-based training data synthesis. Extensive experiments demonstrate that FLAME establishes a new state-of-the-art, significantly outperforming previous methods on AI-generated forgery datasets while effectively generalizing to unseen generative architectures. Our code is available at this https URL. 

---
# Multimodal Approaches for Visually-Rich Document Type Classification: A Comparative Analysis 

**Authors**: Catyana Heyne, Jürgen Frikel, Filippo Riccio  

**Link**: [PDF](https://arxiv.org/pdf/2606.02162)  

**Abstract**: Document type classification in visually rich documents remains challenging, as relevant information is distributed across textual, visual, and layout modalities. To capture this complexity, current approaches rely on diverse multimodal modeling strategies, resulting in heterogeneous architectures that complicate systematic comparison. This variability is also reflected in existing comparative studies, which often rely on heterogeneous evaluation setups, further complicating systematic comparison and making it difficult to assess progress. To address these limitations, this work provides a structured analysis of multimodal design strategies across transformer- and LLM-based architectures, combined with a controlled empirical comparison within a unified experimental framework. Specifically, four representative models (LayoutLMv3, Donut, Qwen3-VL-32B-Instruct, and Qwen3-32B) are evaluated on the RVL-CDIP benchmark to systematically analyze the contributions of text, image, and layout information for document type classification, with a particular focus on contrasting OCR-dependent and OCR-free approaches. The results show that specialized multimodal Transformers outperform LLM-based approaches on visually rich and layout-intensive documents. Image information contributes most strongly to reliable classification, while OCR-derived text provides useful but secondary support. These findings highlight that multimodal processing remains essential for documents with pronounced layout structure. Overall, the study provides a systematic basis for comparing multimodal architectures and offers practical guidance for selecting effective feature combinations and model designs for document type classification. 

---
# Predicting the risk of colorectal anastomotic leak based on preoperative mapping of the blood supply of the bowel 

**Authors**: Zahra Tabatabaei, Jon Sporring, Mark Bremholm Ellebæk, Alaa El-Hussuna  

**Link**: [PDF](https://arxiv.org/pdf/2606.02156)  

**Abstract**: Anastomotic leak remains one of the most serious complications following colorectal cancer surgery, substantially affecting patient outcomes, recovery trajectories, and healthcare costs. Despite advances in imaging technology, current preoperative assessment relies only on clinical assessment, a process that is subjective, error-prone, and highly dependent on individual expertise. To date, no validated CT-based method exists to predict anastomotic leak risk prior to surgery. This protocol paper outlines a comprehensive framework for developing and validating an AI-driven system for preoperative risk assessment using pre- and post-contrast CT imaging. The study describes the stages of data collection, ethical handling, and preprocessing of patient data in accordance with GDPR, image preprocessing, and the exploration of deep learning architectures designed to generate clinically interpretable outputs. Two integrated tools constitute the main deliverables of this workflow: 1) a risk assessment module, which quantifies the likelihood of leakage by analyzing vascular and tissue features in CT scans, and 2) a Content-Based Medical Image Retrieval (CBMIR) module, which identifies and displays similar historical cases to support evidence-based surgical decision making. The protocol paper requires close collaboration between hospitals and universities; this protocol demonstrates that such a system is technically feasible and clinically implementable within existing healthcare infrastructures. By following the proposed methodological stages and regulatory principles, other institutions can reproduce this workflow to develop analogous decision-support tools. Ultimately, this interdisciplinary framework aims to enhance surgical planning, reduce leak incidence, and contribute to a broader paradigm shift toward explainable, data-driven precision surgery. 

---
# Multilingual Idioms in Sentences and Conversations Across High-, Medium-, and Low-Resource Languages 

**Authors**: Saeed Almheiri, Bilal Elbouardi, Salsabila Zahirah Pranida, Irina Nikishina, Ashwath Rao B, Parameswari Krishnamurthy, Muhammad Cendekia Airlangga, Rifo Ahmad Genadi, Nguyen Phan Gia Bao, Amir Hossein Yari, Hawau Olamide Toyin, Nurdaulet Mukhituly, Mena Attia, Besher Hassan, Ahmad Fathan Hidayatullah, Tatsuki Kuribayashi, Haonan Li, Suma Bhat, Fajri Koto  

**Link**: [PDF](https://arxiv.org/pdf/2606.02147)  

**Abstract**: Idiomatic expressions pose a major challenge for multilingual NLP because their meanings shift between figurative and literal usage, often requiring context for accurate interpretation. Prior work has focused on high-resource languages typically evaluates isolated idiom-meaning questions, overlooking realistic discourse. We introduce MIDI, a multilingual idiom dataset spanning 3 high-, 3 medium-, and 12 low-resource languages, curated by native speakers. Unlike previous datasets, MIDI provides idioms embedded in both sentence-level and conversational contexts, capturing both literal and figurative readings. Benchmarking state-of-the-art models shows that idiom comprehension degrades in low-resource languages and that, in all resource tiers, literal interpretations are substantially harder than figurative ones. Conversational context improves performance but does not eliminate these disparities. Through controlled tests and interventions on hidden representations, we further separate memorization from reasoning, exposing core limitations of current models. 

---
# VLBM: Variational Latent Basis Modeling for OOD Robust Multivariate Time Series Forecasting 

**Authors**: Xudong Zhang, Jierui Lei, Jiacheng Li, Lingdong Shen, Jian Cui, Haina Tang  

**Link**: [PDF](https://arxiv.org/pdf/2606.02138)  

**Abstract**: Out of distribution (OOD) events in multivariate time series forecasting are rare but often dominate real world risk, making average case forecasting insufficient for reliable deployment. Under standard average risk training on mixed ID/OOD distributions, optimization signals from rare OOD events can be overwhelmed by frequent in distribution (ID) patterns, so strong benchmark accuracy may not translate into reliability under high impact shifts. To address this issue, we propose VLBM (Variational Latent Basis Model), a theory guided latent forecasting framework that separates stable dynamics from OOD induced deviations. VLBM learns a shared latent basis that defines a low rank subspace for stable ID dynamics, explicitly decomposes inputs into basis subspace components and orthogonal residual components, and aligns a future aware posterior with a future blind prior so that test time latent inference depends only on historical input. Across 12 benchmark tasks spanning transportation, weather, power systems, and other real world domains, including newly constructed real world OOD traffic datasets, VLBM achieves state of the art OOD robustness and ID accuracy, with average MAE and MSE gains of 15.08\% and 7.74\% over the strongest baseline. On a synthetic simulation dataset, VLBM also consistently achieves the best performance and better tracks OOD pulse recovery. These results support latent structured forecasting as a principled route to robust prediction under mixed ID and OOD conditions. The code is available at this https URL. 

---
# Rethinking Evaluation Paradigms in IBP-based Certified Training 

**Authors**: Konstantin Kaulen, Hadar Shavit, Holger H. Hoos  

**Link**: [PDF](https://arxiv.org/pdf/2606.02134)  

**Abstract**: Deep neural networks achieve strong performance on many supervised learning tasks but remain vulnerable to adversarial perturbations. Neural network verification provides mathematically rigorous robustness guarantees, yet at substantial computational cost. To mitigate this, certified training techniques optimise for verifiable robustness during training, typically inducing a trade-off between natural and certified accuracy controlled by method-specific hyperparameters. Because these metrics are inherently conflicting, the common practice of reporting a single configuration is problematic: it can mislead conclusions about overall performance and prevents unbiased assessments of the state of the art. We address this by evaluating certified training methods via Pareto front comparisons over the natural--certified accuracy trade-off. To enable fair, method-agnostic comparisons, we perform efficient automated multi-objective hyperparameter optimisation to identify a set of Pareto-optimal configurations for each method. This approach often uncovers substantial undertuning in previously reported configurations, yielding superior performance and establishing a new state of the art. Leveraging these fronts, we present the first comprehensive multi-objective comparison of certified training approaches, showing that prior advancements are less pronounced than assumed and revealing previously unreported performance complementarities. 

---
# Variational Learning for Insertion-based Generation 

**Authors**: Yangtian Zhang, Zhe Wang, Arthur Gretton, Rex Ying, David van Dijk, Michalis K. Titsias, Jiaxin Shi  

**Link**: [PDF](https://arxiv.org/pdf/2606.02133)  

**Abstract**: Non-monotonic sequence generation methods, such as masked diffusion models, provide a flexible alternative to left-to-right autoregressive modeling by allowing tokens to be generated in non-fixed and prescribed orders. Despite their practical advantages, most existing non-monotonic models are order-agnostic and rely on a fixed-length grid, limiting their ability to support variable-length generation and adaptive insertion order. In this work, we introduce a probabilistic framework for learning insertion order in variable-length insertion models. We formalize a bijective correspondence between insertion trajectories and permutations, which enables an exact reparameterization of the data likelihood as a sum over permutations. Building on this result, we propose the Insertion Process (IP), a stochastic generative model that jointly learns where to insert, what to insert, and when to terminate, trained via permutation-based variational inference. Unlike prior fixed-canvas approaches, IP natively supports variable-length generation and learns data-driven preferences over insertion orders. Experiments on goal-conditioned planning and molecular string generation demonstrate that learning insertion order improves both modeling quality and generalization in domains without a canonical left-to-right structure. 

---
# Understanding-Enhanced Model Collaboration for Long-Tailed Egocentric Mistake Detection 

**Authors**: Boyu Han, Qianqian Xu, Shilong Bao, Zhiyong Yang, Ruochen Cui, Qingming Huang  

**Link**: [PDF](https://arxiv.org/pdf/2606.02120)  

**Abstract**: In this report, we address the problem of determining whether a user performs an action incorrectly from egocentric video data. To this end, we propose an Understanding-Enhanced Model Collaboration Method (UE-MCM) that combines efficient coarse-grained video understanding with accurate fine-grained action reasoning. Specifically, UE-MCM contains a small model branch and a large model branch. The large model branch focuses on whether the fine-grained action itself is executed incorrectly, while the small model branch jointly takes the coarse-grained video and fine-grained segment as input to identify actions that may be locally correct but inconsistent with the overall workflow. The small model branch is built on a CLIP4CLIP video encoder initialized from a CLIP model enhanced by Diffusion Contrastive Reconstruction, and the large model branch uses the Qwen3-VL Embedding model to extract high-capacity representations from fine-grained action segments. The small-branch prediction and the large-branch prediction are then adaptively fused by a lightweight collaboration gate. To handle the long-tailed distribution of mistake instances, we optimize the classifiers with complementary objectives, including reweighted cross-entropy, AUC-oriented learning, and label-aware adjustment. The resulting system balances speed and accuracy, making it effective for detecting subtle, rare, and ambiguous mistakes in egocentric instructional videos. 

---
# How Hard Can It Be? Hardness-Aware Multi-Objective Unlearning 

**Authors**: Jiangwei Chen, Xinyuan Niu, Rachael Hwee Ling Sim, Zhengyuan Liu, Nancy F. Chen, Bryan Kian Hsiang Low  

**Link**: [PDF](https://arxiv.org/pdf/2606.02119)  

**Abstract**: Machine unlearning aims to remove the influence of specific forget training data due to privacy, copyright or bias concerns while maintaining the model performance on the remaining retain data. Existing unlearning algorithms, such as optimizing a weighted combination of losses, have tried to achieve these objectives of improving forget quality and maintaining retain utility. However, they do not guarantee that these objectives can be improved by a specified extent for all forget and retain data. In this work, we address this limitation with a novel and theoretically-grounded approach from a constrained optimization perspective. Firstly, we identify that the hardness of reconciling both objectives can be quantified by the similarity between the forget data and the retain data. Next, we derive an unlearning algorithm (HAMU) with the overall goal of guaranteeing a specified improvement in forget quality while minimizing the retain utility cost/degradation by updating the model weights based on our hardness measure. Our hardness measure also informs users when retain utility degradation is unavoidable, i.e., both objectives cannot be improved simultaneously, and stopping should be considered. Our algorithm is applicable to non-convex models and is easily parallelizable, making it readily deployable in real-world scenarios. We empirically demonstrate HAMU's superior performance over baselines on both image and text datasets using large models. Our code is available at this https URL. 

---
# A Primer in Post-Training Reasoning Data: What We Know About How It Works 

**Authors**: Yaoming Li, Guangxiang Zhao, Qilong Shi, Lin Sun, Xiangzheng Zhang, Tong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.02113)  

**Abstract**: Post-training has become a primary driver of recent progress in large reasoning models, and reasoning data are often the key variable determining whether this stage succeeds. Work on post-training reasoning data has grown rapidly, yet this literature remains scattered across dataset papers, reinforcement-learning recipes, reward-model studies, benchmarks, and frontier system reports. This paper is the first primer to synthesize over 150 key public studies and system reports on post-training reasoning data. We organize the field around four questions: what data objects exist, what makes them useful, how they are constructed, and how they scale. Together, this organization provides an attribution framework for future reasoning-data releases and post-training recipes. 

---
# Jailbreaking Multimodal Large Language Models using Multi-Clip Video 

**Authors**: Choongwon Kang, Seungjong Sun, Hyunmin Jun, Jang Hyun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2606.02111)  

**Abstract**: As multimodal large language models (MLLMs) have advanced to process video inputs, concerns have emerged about their potential for malicious misuse. Prior jailbreak studies have shown that safety alignment in MLLMs can be bypassed through visual inputs, yet it remains unclear which properties of video inputs induce this vulnerability. To address this gap, we introduce Multi-Clip Video (MCV) SafetyBench, a dataset of 2,920 videos designed to evaluate how the diversity of video inputs affects the vulnerability of MLLMs. Each video consists of multiple short clips depicting diverse contexts related to a harmful query. Experiments on eight representative video MLLMs show that attack success consistently increases with the number of clips. Our results further indicate that the video modality is (1) more vulnerable than the image modality, (2) more vulnerable to dynamic videos than to static videos, and (3) more vulnerable when videos contain more diverse contexts. Building on these findings, we propose a defense strategy that leverages the relative robustness of the image modality. 

---
# Network Distributed Multi-Agent Reinforcement Learning for Consensus Control of Quadcopters 

**Authors**: Youssef Mahran, Zeyad Gamal, Aamir Ahmad, Ayman El-Badawy  

**Link**: [PDF](https://arxiv.org/pdf/2606.02107)  

**Abstract**: This paper proposes a Network Distributed Multi-Agent Reinforcement Learning (ND-MARL) framework for quadcopter consensus control. Compared to conventional multi-agent MARL formulations that rely on centralized planning or fully decentralized execution, ND-MARL incorporates the swarm communication graph into the decision process. Under a 2-Neighbor communication topology, each agent observes information of only two neighbors and outputs an action through a distributed policy. A high-level distributed consensus planner is trained using Multi-Agent Soft Actor-Critic (MASAC) and embedded in a hierarchical stack to generate reference target positions tracked by a low-level quadcopter controller. Results demonstrate smooth consensus trajectories and planner-tracker integration when compared to a centralized MARL controller. Most notably, the learned controller exhibits zero-shot scalability, as policies trained on a three-agent system are deployed to swarms of up to 250 agents under the same 2-Neighbor communication topology without retraining or fine-tuning, achieving consistent convergence with increasing steady-state spread at large team sizes due to sparse information propagation. These findings highlight ND-MARL as a stable framework for distributed, communication-aware quadcopter consensus control. 

---
# The Role of Ambiguity in Error Prediction via Uncertainty Quantification 

**Authors**: Ieva Raminta Staliūnaitė, James Bishop, Andreas Vlachos  

**Link**: [PDF](https://arxiv.org/pdf/2606.02093)  

**Abstract**: The task of Error Prediction, namely predicting whether a model output is correct, is commonly tackled with Uncertainty Quantification (UQ). However, while uncertainty metrics capture when models lack knowledge or capacity to make a prediction, they also reflect aleatoric uncertainty, which is inherent in the model input and context. This paper presents a method for improving error prediction for Large Language Models (LLMs), by disentangling input ambiguity from UQ signal. We conduct experiments on the task of Question Answering (QA) with six UQ metrics and show that UQ metrics are more predictive of errors on unambiguous instances than on questions with multiple plausible answers. We use Gated Experts and Selective Prediction to incorporate gold and predicted ambiguity labels into the error prediction pipeline. We find that ambiguity information improves error prediction scores across model families, training and evaluation paradigms, datasets (including allegedly unambiguous ones), and sources of aleatoric uncertainty, yielding improvements of over 10 points of PRR for individual UQ metrics on standard datasets. 

---
# LALE: Lightweight-Transformer Architecture for Land-Cover Estimation 

**Authors**: Ümit Mert Çağlar, Alptekin Temizel  

**Link**: [PDF](https://arxiv.org/pdf/2606.02092)  

**Abstract**: Semantic segmentation of remote sensing imagery requires models that capture both global context and local detail under tight computational budgets. Prior work typically optimizes for one of these axes: attention for global context, convolution for local detail, or compactness for efficiency. While hybrid approaches aim to capture both, they require architectural changes and encoder backbones with computational overhead, limiting efficiency and performance. We present LALE (Lightweight-transformer Architecture for Land-cover Estimation), an end-to-end remote sensing image segmentation architecture, that bifurcates its encoder by resolution: lightweight ConvMixer stages handle high-resolution local features, while transformer stages handle low-resolution global context, confining the quadratic cost of self-attention to deep, downsampled feature maps. An all-MLP multi-scale decoder, together with RMSNorm and StarReLU throughout, further reduces compute and parameter count. On the large-scale ARAS400k remote-sensing segmentation benchmark, LALE establishes a strong efficiency-performance trade-off against CNN, transformer, and hybrid baselines. Our smallest variant, (just 1.6M parameters), reaches within 2.6 F1 points of the best baseline (UPerNet) while using 4.5x fewer parameters, 7x less storage, 17x fewer GMACs, and delivering 1.8x higher throughput. 

---
# Agentic-J: An AI Agent for Biological Microscopy Image Analysis 

**Authors**: Lukas Johanns, Marilin Moor, Davide Panzeri, Yu Zhou, Xinyi Chen, Nora F. K. Pauly, Zixuan Pan, Matthias Gunzer, Andreas Müller, Yiyu Shi, Hedi Peterson, Jianxu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.02080)  

**Abstract**: Biological image analysis increasingly demands integration across heterogeneous tools, programming environments, and domain knowledge that few researchers can command simultaneously. We present Agentic-J, a containerised, multi-agent AI assistant, primarily for ImageJ/Fiji that enables biologists to specify analysis tasks in natural language, from nuclei segmentation and cell tracking to multi-condition quantification. The agent generates executable scripts organised into a documented project structure, so every analysis decision is traceable and the workflow can be reproduced or shared. The specialised sub-agents handle plugin management, code generation, debugging, quality assurance, and statistical reporting. In this paper we introduce the system's design, demonstrate real biological microscopy image analysis workflows, and detailed the technical implementation. 

---
# Fast and Lightweight Novel View Synthesis with Differentiable Multiplane Image 

**Authors**: Kaidi Zhang, Guanxu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2606.02068)  

**Abstract**: Recently, novel view synthesis has witnessed remarkable progress, with mainstream methods such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) delivering impressive results. However, these approaches often struggle to balance rendering speed and model size, and their optimization-based training can be highly time-consuming. Furthermore, they typically rely on dense observations, often failing to produce satisfactory results under sparse-view conditions. Although feed-forward reconstruction significantly reduces the optimization time of 3DGS, its pixel-aligned formulation generates millions of Gaussians from a single image, severely limiting its practical deployment on mobile devices. To address these limitations, we revisit the Multiplane Image(MPI) representation, which represents scenes using a compact set of planar layers for efficient novel view synthesis. Leveraging recent advances in visual foundation models, we utilize predicted point maps for reliable geometric initialization, followed by differentiable optimization. To address the issues of holes and artifacts in sparsely initialized MPI, we introduce one-step diffusion, which participates in both the differentiable optimization of MPI and the postprocessing of rendering results. Compared with a representative GS-based method, our approach is 30.7% faster and uses only 14.8% of its model size, while achieving competitive synthesis quality on front-view scenarios 

---
# Attention mechanisms and transfer learning for robust peach leaf damage classification under domain shift 

**Authors**: Adrián Cánovas-Rodriguez, Miguel A. González-Illán, Maria Fernanda García-Cruz, Pedro Nortes Tortosa, José Salvador Rubio-Asensio, Miguel A. Zamora Izquierdo, Juan Antonio Martínez Navarro, Antonio F. Skarmeta  

**Link**: [PDF](https://arxiv.org/pdf/2606.02045)  

**Abstract**: Artificial intelligence provides a practical framework for crop damage assessment from imagery data, supporting early decision-making in agricultural management. In peach orchards, climate change increases abiotic stress and biotic pressures, including pests and diseases, which often produce visually similar foliar symptoms. This overlap makes manual diagnosis difficult, especially across multiple fields with varying environmental conditions, highlighting the need for automated models with strong generalization ability.
We propose an image-based classification approach for peach leaf damage detection. A benchmark dataset was created through manual annotation of publicly available images, consisting of 1,366 peach leaves across six damage categories. Several deep learning architectures were evaluated. EfficientNet models achieved the best results, with EfficientNetB0 reaching 92.9 percent accuracy, EfficientNetB3 achieving 91.5 percent, and EfficientNetB5 showing the strongest performance on minority classes. DenseNet121 reached 92.6 percent accuracy. The integration of the Convolutional Block Attention Module (CBAM) improved performance in several backbones, particularly EfficientNetB5 and InceptionV3, while showing limited or negative impact in others. The CBAM-enhanced EfficientNetB5 achieved the best overall accuracy of 93.3 percent.
To evaluate robustness under realistic conditions, a local dataset of 180 images across four classes was collected, and transfer learning strategies were applied to address domain shift. Three fine-tuning strategies were tested. EfficientNetB3 combined with CBAM achieved the best performance in the local domain, reaching a 93 percent macro F1-score after transfer. Overall, attention-based models showed improved robustness for minority classes and better generalization across different field conditions. 

---
# OpenWebRL: Demystifying Online Multi-turn Reinforcement Learning for Visual Web Agents 

**Authors**: Rui Yang, Qianhui Wu, Yuxi Chen, Hao Bai, Wenlin Yao, Hao Cheng, Baolin Peng, Huan Zhang, Tong Zhang, Jianfeng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2606.02031)  

**Abstract**: Building capable visual web agents requires long-horizon reasoning, precise grounding, and robust interaction with dynamic real-world websites. Despite rapid progress, the strongest systems remain largely proprietary, while open agents still depend heavily on supervised post-training over large collections of curated web trajectories. This dependence creates a major scalability bottleneck: high-quality demonstrations are expensive to collect, and static datasets offer limited coverage of the diverse, ever-changing open web. Although online RL has shown promise for text-based agents, its potential for training visual web agents directly on live websites remains largely underexplored. In this paper, we introduce OpenWebRL, an open framework for training visual web agents with online multi-turn RL on real websites. OpenWebRL covers the full training pipeline, including scalable live-browser infrastructure, supervised initialization, multimodal context management, trajectory-level success judging, and efficient multi-turn policy optimization. Using this framework, we train OpenWebRL-4B, which establishes a new open-source state of the art on challenging live-web benchmarks. With only 0.4K initialization trajectories and 2.2K open-ended RL training tasks, OpenWebRL-4B achieves 67.0% success on Online-Mind2Web and 64.0% on DeepShop, outperforming prior open agents of similar or larger scale and remaining competitive with proprietary systems including OpenAI CUA and Gemini CUA. Beyond strong benchmark performance, we systematically study the key design choices that make online RL effective for visual web agents, and analyze how RL improves agentic reasoning. Overall, our work offers a practical path toward building more capable, reproducible, and cost-efficient open web agents. We will release our training data, models, and code to support future research. 

---
# Ranking vs. Assignment: The Metric Mismatch in Multi-View Object Association 

**Authors**: Matvei Shelukhan, Timur Mamedov, Aleksandr Chukhrov, Karina Kvanchiani  

**Link**: [PDF](https://arxiv.org/pdf/2606.02022)  

**Abstract**: Multi-view object association is an important computer vision problem that underlies many multi-camera perception tasks. While this task is naturally formulated as a constrained one-to-one matching problem, recent works heavily rely on pairwise ranking metrics like AP and FPR-95 for model evaluation. We highlight a fundamental mismatch between these metrics and the actual assignment objective. Theoretically, we show that AP and FPR-95 can be imperfect even when the assignment is already correct, and that Sinkhorn-based normalization can make them perfect. Conversely, optimal pairwise ranking can still lead to incorrect assignments. We validate this mismatch in practice by using our Sinkhorn-based normalization as a controlled post-processing stress test. We show that optimizing just a few post-processing parameters significantly boosts AP and FPR-95 without corresponding improvements in assignment-level metrics such as ACC and IPAA. 

---
# PlanarBench: Evaluating LLM Spatial Reasoning via Planar Graph Drawing 

**Authors**: Oleksandr Nikitin  

**Link**: [PDF](https://arxiv.org/pdf/2606.02010)  

**Abstract**: PlanarBench tests whether LLMs can draw planar graphs as ASCII art given only an edge list -- a spatial reasoning task that resists memorization because edge order, edge orientation, and node labels are all permutable.
We evaluate 91 models on the 199 simplest non-isomorphic connected planar graphs (2 - 7 vertices).
Edge count is the dominant difficulty predictor ($r = -0.85$) -- a finding not reported in prior LLM graph benchmarks, which use only node count as the difficulty axis. 

---
# Towards 3D-Aware Video Diffusion Models: Render-Free Human Motion Control with Mesh Tokenization 

**Authors**: Jingyun Liang, Min Wei, Shikai Li, Yizeng Han, Hangjie Yuan, Lei Sun, Weihua Chen, Fan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.02000)  

**Abstract**: Diffusion models have shown remarkable success in video generation. However, whether such models are truly aware of the 3D structure underlying visual observations, rather than simply reproducing plausible 2D projections, remains an open question. In this work, we investigate this question through human motion control, a task that requires precise modelling of 3D human geometry, motion, camera viewpoint, and scene context. Unlike prior methods that rely on rendered 2D motion guidance videos, we propose a render-free framework that conditions video generation directly on compressed 3D human mesh tokens. This representation preserves full 3D geometric information while enabling a unified token-based generation pipeline that processes video tokens jointly with motion tokens in a DiT-based architecture. This design requires the model to reason jointly about appearance, 3D structure, and camera viewpoint during video generation. Experimental results demonstrate strong performance on human motion control benchmarks, while reducing artifacts induced by view-dependent 2D guidance and trajectory-pose mismatches during editing. These findings suggest that video diffusion models, when equipped with mesh tokenization, can better capture complex 3D human structures and their interactions with the surrounding environment. 

---
# Why Do Time Series Models Need Long Context Windows? 

**Authors**: Luca Butera, Giovanni De Felice, Andrea Cini, Cesare Alippi  

**Link**: [PDF](https://arxiv.org/pdf/2606.01999)  

**Abstract**: Modern deep learning models for forecasting groups of time series rely on increasingly longer observation windows. However, the benefit of increasing the window size is often simply attributed to capturing long-range dependencies, and broader discussion on how global forecasting models leverage input observations has been limited. In this paper, we show that forecasting groups of time series involves two objectives: (i) generative process identification (GPI), i.e., inferring the specific process generating the input sequence, and (ii) conditional forecasting (CF), i.e., predicting future values given input observations. From this perspective, optimal predictions can be interpreted as an average over plausible data-generating processes, weighted by their likelihood given the input window. This suggests another explanation for the benefits of long context windows: they reduce the uncertainty about which specific process is generating the input time series during operation. We prove that even for processes with memory length $P$, an input window size strictly larger than $P$ is necessary to achieve the minimum attainable error. Finally, we show how decoupling GPI and CF can improve computational scalability without compromising accuracy. Experiments on synthetic and real-world data validate our insights and their relevance for designing forecasting architectures. 

---
# MMG2Skill: Can Agents Distill In-the-Wild Guides into Self-Evolving Skills? 

**Authors**: Xinyu Che, Junqi Xiong, Yunfei Ge, Xinping Lei, Shihao Li, Hang Yan, Han Li, Yuanxing Zhang, Zhiqi Bai, Jinhua Hao, Ming Sun, Han Li, Jiaheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.01993)  

**Abstract**: Abundant procedural knowledge on the Web holds great potential for helping agents solve long-horizon tasks. However, such knowledge is often multimodal, heterogeneous, noisy, and implicitly assumes human executors, making it difficult to use directly as the skills required by agents. To bridge the gap between human-oriented guides and agent-executable skills, we formalize this problem as guide-to-skill learning: converting in-the-wild guides into executable skills and continuously improving them from trajectories observable to the agent. To evaluate the capability of existing agents on this task, we introduce MMG2Skill-Bench, the first benchmark designed for this problem. We further propose MMG2Skill, a closed-loop framework that compiles guides into editable skills, conditions a fixed vision-language model (VLM) agent on these skills during execution, and revises the skills from trajectory-level root-cause feedback without using benchmark scores. Across GUI control, open-ended gameplay, and strategic card play with six VLM backbones, MMG2Skill consistently outperforms vanilla baseline agents in every model-domain setting, achieving macro-average gains of +12.8 to +25.3 percentage points across backbones. Ablation studies show that directly prompting agents with raw guides can degrade performance, while both structured skill construction and trajectory-driven revision are necessary for the observed improvements. On success-inferable tasks, analyzer-based early stopping further prevents late-stage performance regressions and saves 25%-53% of attempts when the success signal is properly calibrated. 

---
# A Structured Benchmark for Text-Guided Anomaly Detection: When Language Stops Conditioning the Decision 

**Authors**: Stefano Samele, Eugenio Lomurno, Teodora Jovanovic, Sanjay Shivakumar Manohar, Alberto Crivellaro, Matteo Matteucci  

**Link**: [PDF](https://arxiv.org/pdf/2606.01992)  

**Abstract**: Industrial anomaly detection has historically been a unimodal task. Recent multimodal vision-language models have produced systems that admit textual input alongside the image and are presented as enabling text-guided zero- and few-shot inspection. Yet these methods are evaluated with protocols inherited from unimodal benchmarks that hold the textual condition constant and therefore cannot measure whether language conditions the decision; whether reported gains reflect text guidance or strong pretrained visual features remains open. We introduce Text-Guided Anomaly Detection (TGAD), a structured benchmark that progressively increases the functional role of language across three scenarios: a controlled prompt-sensitivity setting on MVTec AD; a component-tagged extension of MVTec AD that requires the model to restrict its assessment to an instructed part; and the new Assembled Panel Dataset (APD), a realistic industrial setting that requires both defect-type and component-location knowledge. We evaluate one representative model per paradigm: generative large vision-language, training-free discriminative, and embedding-adaptive discriminative. In all three, the textual interface conditions the decision only superficially: prompt content is absorbed unless the object noun is removed (the generative model's I-AUROC drops from 97.4 to 82.6); component-level instructions do not constrain the decision once defects outside the instructed part are admitted as normal (from 90.3 to 66.3); and when both combine on APD, image-level discrimination collapses below the MVTec level, in one case below chance (71.2, 50.5, 31.5). These results suggest that standard benchmarks overstate the text-guided capabilities of current multimodal anomaly detection systems, and that a protocol of this kind is a prerequisite for models that can be reliably controlled through language for industrial deployment. 

---
# Rank-Constrained Deep Matrix Completion for Group Recommendation 

**Authors**: Mubaraka Sani Ibrahim, Lehel Csató, Isah Charles Saidu  

**Link**: [PDF](https://arxiv.org/pdf/2606.01948)  

**Abstract**: The growing popularity of group activities has increased the need for methods that provide recommendations to groups of users given their individual preferences. Many existing group recommender systems rely on aggregating individual user preferences, but they often struggle with high-dimensional and highly sparse rating data commonly found in real-world scenarios. We propose Group Rank-Constrained Deep Matrix Completion (Group RC-DMC), a novel framework that extends RC-DMC by integrating group-level representation learning via a Set-Transformer aggregator, jointly leveraging low-rank structure and attention-based nonlinear modeling. Unlike most existing group recommender systems, Group RC-DMC unifies explicit low-rank regularization, linear encoder-decoder architectures, and attention-based nonlinear group modeling within a single framework, yielding accurate predictions at both the individual and group levels. Group RC-DMC addresses data sparsity through low-rank matrix completion, computing per-user latent representations from observed ratings only, and enforcing a rank constraint on the latent space using a nuclear-norm proximal step based on periodic singular value thresholding. The decoder is parametrized as a low-rank factorization, enabling efficient inference. Experimental results on the MovieLens and Goodbooks datasets demonstrate that Group RC-DMC achieves superior reconstruction accuracy, measured by lower group RMSE, while remaining computationally efficient and competitive in group-level performance in terms of precision, recall, and F1 score compared with weighted-before-factorization (WBF) and after-factorization (AF) baselines. The results highlight the model's ability to recover the underlying low-rank structure of user-item interactions and provide robust group recommendations across small, medium, and large user groups. 

---
# Parameter-Efficient Fine-Tuning of Large Pretrained Models for Instance Segmentation Tasks 

**Authors**: Nermeen Abou Baker, David Rohrschneider, Uwe Handmann  

**Link**: [PDF](https://arxiv.org/pdf/2606.01947)  

**Abstract**: Research and applications in artificial intelligence have recently shifted with the rise of large pretrained models, which deliver state-of-the-art results across numerous tasks. However, the substantial increase in parameters introduces a need for parameter-efficient training strategies. Despite significant advancements, limited research has explored parameter-efficient fine-tuning (PEFT) methods in the context of transformer-based models for instance segmentation. Addressing this gap, this study investigates the effectiveness of PEFT methods, specifically adapters and Low-Rank Adaptation (LoRA), applied to two models across four benchmark datasets. Integrating sequentially arranged adapter modules and applying LoRA to deformable attention--explored here for the first time--achieves competitive performance while fine-tuning only about 1-6% of model parameters, a marked improvement over the 40-55% required in traditional fine-tuning. Key findings indicate that using 2-3 adapters per transformer block offers an optimal balance of performance and efficiency. Furthermore, LoRA, exhibits strong parameter efficiency when applied to deformable attention, and in certain cases surpasses adapter configurations. These results show that the impact of PEFT techniques varies based on dataset complexity and model architecture, underscoring the importance of context-specific tuning. Overall, this work demonstrates the potential of PEFT to enable scalable, customizable, and computationally efficient transfer learning for instance segmentation tasks. 

---
# Echo: A Joint-Embedding Predictive Architecture for Speaker Diarization and Speech Recognition in a Shared Latent Space 

**Authors**: Louis Mouchon  

**Link**: [PDF](https://arxiv.org/pdf/2606.01909)  

**Abstract**: We present Echo, a proof-of-concept audio system built around a single 25 M-parameter ViT encoder. The encoder is pretrained with a JEPA objective and then specialised by stages to carry speaker identity, phonetic content, and dynamic source routing in the same 512-dimensional latent space, with no per-task fine-tuning at deployment. Light heads handle diarization (ArcFace + VBx) and dynamic source separation (null-target K-set prediction). On synthetic VoxCeleb2 mixtures with unknown K, the canonical stack reaches 15.00% blind DER, 97.80% PIT separation accuracy with +9.52 dB latent SI-SDR, and a +53.50-point speaker/content factorisation gap on a held-out k-NN probe. The point of Echo is not a new SOTA on any single task but the joint coexistence of three tasks on one encoder at this footprint. We document the design stage by stage, report the dead-ends, and identify the structural wall on end-to-end ASR through the VQ bottleneck that still bounds the PoC. 

---
# KliniskVestBERT: BERT Model Specialised to Norwegian Clinical Texts 

**Authors**: Christian Autenried, Cosimo Persia  

**Link**: [PDF](https://arxiv.org/pdf/2606.01904)  

**Abstract**: The increasing application of Natural Language Processing (NLP) in healthcare demands language models specifically attuned to the complexities of clinical language. This work introduces KliniskVestBERT, a suite of three BERT-based encoder models pre-trained on a substantial corpus of real-world, de-identified Norwegian clinical texts from Helse Vest. We continue pretraining existing language models Nb-BERT-large, NorBERT3-large, and ModernBERT on our specialized clinical dataset. This dataset is based on a representative population of Helse Vest patients. The included document types are carefully curated to encompass a broad clinical spectrum in bokmål and nynorsk including discharge summaries, surgical reports, nursing notes etc. ensuring comprehensive representation of the linguistic landscape within Norwegian healthcare settings. Evaluation on three synthtetic Norwegian clinical benchmark datasets and two real-world problems demonstrates that each of our clinically specialized models consistently outperforms their baseline counterparts, highlighting the significant benefit of domain-specific pre-training for NLP tasks within the clinical domain. The project was a joint effort by all Helse Vest entities (Helse Bergen, Helse Fonna, Helse Førde and Helse Stavanger) with DIPS under the project lead of Helse Vest ICT. 

---
# The Image Reconstruction Game: Drawing Common Ground Through Iterative Multimodal Dialogue 

**Authors**: Sherzod Hakimov, Mattia D'Agostini, Ivan Samodelkin, David Schlangen  

**Link**: [PDF](https://arxiv.org/pdf/2606.01901)  

**Abstract**: We introduce the Image Reconstruction Game, a fully automated benchmark in which a vision-language model issues corrective instructions to an image generator across multiple turns, making accumulated common ground directly observable as a rendered image. Benchmarking two Describer models crossed with two Generator models across seven image categories, we find that the describer is the dominant factor in reconstruction quality, while the generator determines whether iterative refinement helps or hurts. Mathematical and geometric images pose the greatest challenge. The describer's token budget strongly affects convergence: shorter budgets yield sparser first renderings with more room for visible improvement, while longer budgets raise absolute quality but leave less to fix. Stronger describers use a richer correction vocabulary spanning spatial, numeric, and structural categories, while weaker describers concentrate on surface properties and tend to stop after a few turns. Human validation shows that the best automated judge reaches only slight-to-fair agreement with human preferences, and automated scores require human recalibration to be used reliably. 

---
# RA-LWLM: Retrieval-Augmented In-Context Localization with Wireless Foundation Models 

**Authors**: Guangjin Pan, Hui Chen, Hei Victor Cheng, Henk Wymeersch  

**Link**: [PDF](https://arxiv.org/pdf/2606.01899)  

**Abstract**: Wireless localization is a fundamental capability of sixth-generation (6G) networks. Conventional model-based methods require accurate modeling of the propagation environment and degrade in complex multipath and non-line-of-sight scenarios, while learning-based methods couple model parameters tightly to the training scene, requiring costly retraining whenever the base station (BS) configuration or propagation environment changes. In this paper, we propose RA-LWLM, a retrieval-augmented in-context localization framework that achieves training-free cross-scene adaptation by externalizing scene-specific information into a per-scene fingerprint database rather than encoding it in model weights. The framework consists of three components: a frozen wireless foundation model (FM) encoder that maps raw channel state information into a scene-agnostic representation; a retrieval module that selects the most informative references from the per-scene database via similarity search in the representation space; and a transformer-based in-context learning (ICL) module that fuses the query with the retrieved references to predict the user equipment (UE) position. To accommodate varying retrieval quality and propagation complexity across queries, the ICL module adopts a mixture-of-experts design in which experts specialize in different context sizes and are softly combined by a learnable selector. Extensive ray-tracing-based experiments across heterogeneous scenes with diverse BS configurations show that RA-LWLM achieves nearly identical accuracy on seen and unseen scenes without any per-scene retraining, substantially outperforming end-to-end and FM-based baselines. These results validate the proposed retrieval-augmented in-context paradigm as a scalable solution for cross-scene localization in 6G networks. 

---
# Train, Test, Re-evaluate: Schedule-Sensitive Evaluation of Generative Data for Hand Detection 

**Authors**: Atmika Bhardwaj, Silvia Vock, Nico Steckhan  

**Link**: [PDF](https://arxiv.org/pdf/2606.01896)  

**Abstract**: Generated (or synthetic) image data is increasingly used to augment or replace real training datasets when target imagery is scarce, expensive, or biased. For hand detection, particularly in occupational safety settings, public datasets mostly contain bare hands. This under-represents the variation in hand appearance introduced by gloves, tattoos, jewelry, and other personal protective equipment, creating a distribution shift that safety-critical applications encounter at deployment. We test whether generative inpainting, editing only the hand region of a real photograph to introduce accessories, can close this shift gap. On a paired dataset of real images and their synthetic counterparts, we train YOLOv8n hand detectors under six training-and-scheduling regimes (Experiments A-F, three random seeds each), evaluate every detector on a real test set and on a real-gloves-only test split, and report the mean average precision (mAP) at two overlap thresholds (mAP@0.5 and mAP@0.5:0.95) along with paired statistical tests. A two-stage experiment: train on real U synthetic data, then fine-tune the resulting weights on real-only at a lower learning rate, increases mAP@0.5 compared to the real-only baseline model on the standard real test set, and improves the real-gloves out-of-distribution gap. Another three-stage experiment preserves box-tightness best, reaching the highest mAP@0.5:0.95 of any other experiment in the study. The synthetic-data utility for safety-critical hand detection is determined by the training procedure, and simple multi-stage experiments extract substantial real-deployment benefit from inpainted accessory data. 

---
# Collaborative Space Object Detection with Multi-Satellite Viewpoints in LEO Constellations 

**Authors**: Xingyu Qu, Wenxuan Zhang, Peng Hu  

**Link**: [PDF](https://arxiv.org/pdf/2606.01895)  

**Abstract**: With the growing number of satellites in low Earth orbit (LEO) constellations, the near-Earth space environment has become increasingly congested, making space object detection (SOD) a pressing challenge for space safety and sustainability. To mitigate collision risks and ensure the continuity of space operations, SOD systems must deliver fast and accurate detection under stringent onboard constraints. In this paper, we investigate the potential of multi-viewpoint observation fusion within a deep learning (DL) framework to enhance SOD performance. We design a practical multi-view pipeline and several input representations for feeding multi-view data into YOLO-based detectors. Our experiments show that using multi-view inputs is feasible in most cases and typically produces better results for mAP50 and mAP50-95. For example, in model YOLOv9-m, single-view compared to a three-view fused RGB setting, mAP50 increases from 0.638 to 0.732, while mAP50-95 improves from 0.227 to 0.276. Compared with the single-view setting, the best three-view grayscale configuration improves mAP50 by 36.3% and mAP50-95 by 46.5%. These findings establish multi-view fusion as a viable and effective strategy for SOD, with broad implications for space situational awareness in LEO constellation deployments. 

---
# RadioMaster: Multi-Agent System for Autonomous Radio Signal Generation 

**Authors**: Jiazhen Lei, Tianze Cao, Yuxin Sha, Sihan Wang, Bingbing Wang, Fengyuan Zhu, Zeming Yang, Xiaohua Tian  

**Link**: [PDF](https://arxiv.org/pdf/2606.01862)  

**Abstract**: Translating user intents into physical radio signals represents the critical yet notoriously tedious final step in wireless prototyping, as it requires intricate knowledge of physical layer details and presents immense implementation challenges. Large Language Models (LLMs) and multi-agent systems have revolutionized conventional software engineering, raising the compelling question of whether they can resolve these formidable difficulties. However, our investigations reveal that current models experience significant limitations and fail to accomplish this task when applied to radio signal generation. This performance degradation primarily stems from severe domain ignorance and a fundamental insensitivity to physical hardware constraints. To bridge this gap, we introduce RadioMaster, a fully autonomous multi-agent framework designed to seamlessly translate user input into real-world wireless emissions. RadioMaster operates on three synergistic pillars: RadioWiki for domain-specific knowledge retrieval, RadioAgent for collaborative I/Q sample generation alongside hardware configuration, and RadioEmulator for closed-loop physical layer verification. Furthermore, we construct RadioBench, the first comprehensive benchmark tailored specifically for the radio signal generation domain. Extensive real-world evaluations demonstrate that RadioMaster significantly outperforms state-of-the-art (SOTA) baselines regarding configuration viability and signal fidelity. 

---
# Boosting Multimodal Federated Learning via Chained Modality Optimization 

**Authors**: Zixin Zhang, Fan Qi, Shuai Li, Xiaoshan Yang, Changsheng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.01856)  

**Abstract**: Multimodal Federated Learning (MMFL) enables privacy-preserving collaborative learning across decentralized clients with heterogeneous data and modality availability. However, most existing MMFL methods cast multimodal training as a joint optimization problem, overlooking a key bottleneck: modality competition, where dominant modalities suppress weaker ones and lead to suboptimal global models. To address this, we propose FedMChain, a balanced MMFL framework that structures federated multimodal training as a chain of modality-wise phases. This phase-wise design gives each modality a dedicated local optimization window on multimodal clients to mitigate modality competition, and further promotes cross-modal complementarity via an error-compensated regularizer. On the server side, we employ a sparse sign-guided aggregation strategy that leverages directional sign agreement for robust intra-modality aggregation, avoids destructive averaging, and supports less frequent synchronization to reduce communication overhead. Extensive experiments on multimodal benchmarks demonstrate that FedMChain consistently improves predictive performance while requiring less frequent communication than baselines. 

---
# Unveiling the Limits of Large Language Models in Inferring Pragmatic Meaning from Non-Verbal Responses 

**Authors**: Sugyeong Eo, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2606.01845)  

**Abstract**: Although large language models (LLMs) have shown considerable progress in pragmatic language understanding, prior research has focused mainly on their comprehension of verbal behavior. Nonetheless, non-verbal behavior remains a fundamental component of human communication, especially when deliberately utilized in isolation to convey indirect meanings. In this work, we present the first systematic evaluation of LLMs' ability to infer pragmatic meaning in dialogue consisting solely of non-verbal responses. We explore three research questions: (1) Can LLMs recognize indirect intent conveyed through non-verbal responses? (2) When and how do LLMs fail to capture non-verbal intent? (3) How can we improve LLMs' ability to interpret non-verbal intent?. Through the evaluation, we observe that LLMs struggle to infer underlying meaning from non-verbal responses, with accuracy dropping by up to 60% points compared to verbal ones. Further extensive analysis reveals a behavioral pattern in LLMs' interpretations of non-verbal behavior and demonstrates that in-context learning facilitates pragmatic inference. 

---
# Suppressing Forgery-Specific Shortcuts for Generalizable Deepfake Detection 

**Authors**: Yihui Wang, Yonghui Yang, Jilong Liu, Fengbin Zhu, Le Wu, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2606.01843)  

**Abstract**: Deepfake detection suffers from poor generalization across forgery methods, as existing models tend to rely on spurious method-specific shortcuts that fail to transfer to unseen manipulations. While recent approaches attempt to improve generalization, they lack an explicit mechanism to identify and suppress such shortcuts in learned representations. In this work, we propose Shortcut Subspace Suppression (S^3) framework that explicitly characterizes and suppresses method-specific shortcuts via subspace modeling. Our key insight is that variations distinguishing different forgery methods capture method-specific artifacts and thus serve as an effective proxy for method-specific shortcuts. To this end, we train a lightweight linear probe for forgery method classification and perform Singular Value Decomposition (SVD) to extract the dominant shortcut subspace. Building on this formulation, we develop two complementary strategies to reduce shortcut reliance. During training, we softly suppress the shortcut subspace in feature representations, encouraging the model to rely on more generalizable cues for real/fake discrimination. At inference time, we introduce a training-free counterpart that attenuates neurons aligned with the identified shortcut directions, enabling plug-and-play generalization enhancement with improved interpretability. Extensive experiments on multiple benchmarks demonstrate that our method significantly improves cross-method generalization while maintaining strong in-domain performance. The code will be released upon acceptance of the submission. 

---
# LayerRoute: Input-Conditioned Adaptive Layer Skipping via LoRA Fine-Tuning for Agentic Language Models 

**Authors**: Prateek Kumar Sikdar  

**Link**: [PDF](https://arxiv.org/pdf/2606.01838)  

**Abstract**: Agentic language model systems alternate between two structurally distinct step types: structured tool calls (short, deterministic, low perplexity) and open-ended planning/reasoning steps (long, complex, high perplexity). Despite this heterogeneity, current inference systems apply identical compute to every step. We introduce LayerRoute, a lightweight adapter that learns to selectively skip transformer blocks on a per-input basis. LayerRoute augments each of the 24 transformer blocks in Qwen2.5-0.5B-Instruct with: (1) a per-layer router (~897 parameters, Linear(896,1)) that outputs a hard binary gate via the straight-through estimator, and (2) LoRA adapters (rank 8, ~1.08M parameters) on the Q/K/V/O attention projections. The backbone weights remain frozen. A single end-to-end training pass on agentic data (Hermes, Glaive, GSM8K, Turing) with a gate regularisation term forces the system to discover which blocks are skippable per input type. After 3,000 steps (6.4 minutes on an A100 40GB), LayerRoute achieves a 12.91% skip differential: tool calls skip 15.25% of FLOPs while planning steps skip only 2.34%, using only 1.10M trainable parameters (0.22% of the 494M backbone). Quality improves over the base model due to LoRA adaptation, with perplexity delta of -1.29 on tool calls and -1.30 on planning. 

---
# Physics-Guided Attention in a Lightweight TCN for Efficient WiFi CSI-Based Human Activity Recognition 

**Authors**: Chinthaka Ranasingha, Tharindu Fernando, Sridha Sridharan, Clinton Fookes, Harshala Gammulle  

**Link**: [PDF](https://arxiv.org/pdf/2606.01834)  

**Abstract**: Human Action Recognition (HAR) using WiFi Channel State Information (CSI) has gained increasing attention due to its non-contact, low-cost, and privacy-preserving nature. However, existing learning-based approaches largely rely on deep, computationally intensive architectures to implicitly capture motion dynamics from CSI measurements, thereby increasing model complexity and reducing efficiency. Instead, we argue that incorporating appropriate inductive biases tailored to the physical characteristics of CSI signals enables more efficient and effective learning. In this work, we propose a compact temporal convolutional network (TCN)-based framework that explicitly incorporates motion-aware inductive biases into feature learning. Specifically, we introduce a Doppler-energy-guided temporal attention mechanism in feature space to emphasize motion-salient time segments, and a variance-driven channel attention module to weight informative subcarriers based on temporal motion statistics adaptively. By integrating these domain-specific priors, the proposed model effectively captures motion dynamics without increasing architectural depth. Extensive experiments on multiple benchmark datasets demonstrate that our approach achieves superior performance compared to deeper baselines, while significantly reducing parameter count and computational cost. 

---
# Learning Implicit Bias in Generative Spaces for Accelerating Protein Dynamics Emulation 

**Authors**: Kaihui Cheng, Zhiqiang Cai, Wenkai Xiang, Zhihang Hu, Siyu Zhu, Tzuhsiung Yang, Yuan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2606.01833)  

**Abstract**: Generative emulators of protein dynamics produce plausible trajectories at a fraction of the cost of molecular dynamics, but they inherit their training distribution and tend to revisit known states rather than reach rare ones under long-horizon extrapolation. Inspired by classical enhanced sampling, we introduce an implicit, history-dependent bias in the generative space of a pretrained emulator. Specifically, a history-aware score estimator augments the frozen emulator with a distance-weighted bias that steers reverse-time sampling away from previously generated structures, regularized by an environment-support term. To preserve structural validity at long horizons, a score-based refinement step re-projects drifted samples onto the data manifold using the frozen emulator. Our experiments demonstrate that the method (i) raises diversity by $35\%$ on DynamicPDB-80; (ii) on $12$ zero-shot Fast-Folding proteins, the learned bias alone reaches the unbiased emulator's coverage up to ${\sim}15\times$ faster, and pairing it with refinement reaches the coverage up to ${\sim}37\times$ faster while covering ${\sim}3\times$ as many low-energy states. Code will be released soon. 

---
# Dynamic Trust-Aware Sparse Communication Topology for LLM-Based Multi-Agent Consensus 

**Authors**: Wanshuang Gou, Zihan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.01828)  

**Abstract**: Large language model-driven multi-agent systems enhance the reliability of complex reasoning tasks through multi-round deliberation, role specialization, and cross-validation. However, existing multi-agent debate and collaboration frameworks typically adopt fully connected communication, causing the number of messages, token costs, and end-to-end latency to grow approximately quadratically with the number of agents; although fixed sparse topologies reduce overhead, they cannot adapt communication relationships to different task instances or intermediate reasoning states, making them prone either to preserving low-value interactions or to losing critical error-correction information. To address this problem, this paper proposes DySCo (Dynamic Sparse Consensus), a dynamic trust-aware sparse consensus mechanism. In each round of reasoning, DySCo estimates the value of communication edges based on agent reliability, answer divergence, and task relevance, and selects a small number of high-value edges for message exchange under budget constraints; it then aggregates the answers of different agents through dynamic trust weights and terminates the discussion early once consensus stabilizes. This mechanism replaces universal broadcasting with on-demand communication, thereby reducing communication overhead while preserving essential cross-validation information. We further present analyses of communication complexity and consensus stability, and evaluate the performance of DySCo on mathematical reasoning, logical reasoning, and factual question-answering tasks. 

---
# "I've Seen How This Goes": Characterizing Diversity via Progressive Conditional Surprise 

**Authors**: Matthew Khoriaty, David Williams-King, Shi Feng  

**Link**: [PDF](https://arxiv.org/pdf/2606.01811)  

**Abstract**: Measuring the diversity of creative outputs is central to evaluating post-training mode collapse, comparing decoding strategies, and quantifying creative behavior in both AI and human writing. We propose a new approach to measuring diversity using in-context learning, of which the ``Decan'' metric, $D_{Ca_n} = C \times a_n$, is the working instance we evaluate: a per-byte score read off the per-token log-probabilities of a base model $\theta$ in a \emph{single forward pass} per permutation, with no embedding model, no reference corpus, and no human labels. This approach is grounded in information theory, makes use of language model in-context learning to detect a wide range of similarities between any number of inputs, and obviates the need to train a special-purpose model. The same pipeline scores AI samples and human-written response sets, with diversity treated as a property of (responses, prompt, scoring model). On Tevet and Berant's human-grounded McDiv benchmark, $D_{Ca_n}$ reaches OCA 0.846 on the McDiv prompt\_gen set where it performs best, behind the strongest neural baseline reported in Tevet and Berant (SentBERT, 0.897). On the OLMo-2-7B post-training pipeline, $D_{Ca_n}$ drops monotonically across the base $\to$ SFT $\to$ DPO $\to$ RLVR stages, detecting the type of diversity loss that creative-writing applications care about. 

---
# ProbeScale: Probing Analysis to Optimize Neural Scaling Laws for Efficient Small Language Model Inference 

**Authors**: Sourav Das  

**Link**: [PDF](https://arxiv.org/pdf/2606.01806)  

**Abstract**: Small Language Models (SLMs) offer a balance between capability and computational feasibility. Neural scaling laws inform their optimal training, suggesting that they possess rich internal representations that scale with their size. However, deploying even these SLMs can be challenging under strict resource constraints. Language model probing provides methods for analyzing the linguistic knowledge encoded in a model's internals. We propose ProbScale, a framework that unifies insights from scaling laws and probing to identify parameter-efficient subnetworks within pre-trained SLMs. ProbScale utilizes the high-quality representations of well-scaled SLMs and uses task-specific probes to mathematically quantify the relevance of each layer for target downstream capabilities. This allows selecting subnetworks that optimally trade off performance against parameter size. We formulate the subnetwork selection as finding a layer subset maximizing aggregated, task-weighted probe performance under a parameter budget. Experiments on representative SLMs such as RoBERTa-Large and T5-Base demonstrate that ProbScale identifies subnetworks achieving significant parameter reduction, from 5 to 10 times, while maintaining high performance (95% to 98% of the original SLMs) on targeted tasks, outperforming heuristic baselines. 

---
# MOSS-Audio Technical Report 

**Authors**: Chen Yang, Chufan Yu, Hanfu Chen, Jie Zhu, Jingqi Chen, Ke Chen, Wenxuan Wang, Yang Wang, Yaozhou Jiang, Yi Jiang, Zhengyuan Lin, Ziqi Chen, Zhaoye Fei, Chenghao Liu, Jun Zhan, Kang Yu, Kexin Huang, Mingshu Chen, Qinyuan Cheng, Ruixiao Li, Shimin Li, Songlin Wang, Yang Gao, Yiyang Zhang, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2606.01802)  

**Abstract**: MOSS-Audio is a unified audio-language model for speech, environmental sound, and music understanding, supporting audio captioning, time-aware question answering, timestamped transcription, and audio-grounded reasoning. MOSS-Audio couples a dedicated audio encoder with a modality adapter and a large language model: the encoder produces 12.5 Hz temporal representations, the adapter projects them into the decoder space, and the decoder generates autoregressive text outputs. Two design choices are central to the system: \textbf{DeepStack cross-layer feature injection}, which exposes the decoder to acoustic information from multiple encoder depths, and \textbf{time markers}, which provide explicit temporal cues by inserting timestamp markers into the audio-token stream. At the data level, we design an event-preserving audio annotation pipeline that segments raw audio at coherent event boundaries, applies branch-specific annotation to speech, music, and general audio, and merges the results into unified captions for pretraining. The intermediate branch-specific captions are further retained to support the construction of task-oriented SFT data. The model is pretrained on large-scale audio-language data, with time-aware objectives incorporated to support temporal grounding, and then undergoes multi-stage post-training to enhance instruction following and audio-grounded reasoning. We release 4B and 8B variants in both Instruct and Thinking configurations. MOSS-Audio achieves strong performance across general audio understanding, speech captioning, ASR, and timestamped ASR, positioning it as a promising understanding foundation for future voice agents. 

---
# Multilinguality of Large Language Models From a Structural Perspective 

**Authors**: Haruki Sakajo, Yusuke Sakai, Hidetaka Kamigaito, Taro Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2606.01800)  

**Abstract**: Large language models (LLMs) have excelled in processing multiple languages through pre- and post-training on multilingual data, even though English dominates the training data. Prior work focusing on token representations has revealed how those LLMs process non-English text. Although these analyses have provided insightful findings, they fail to capture a structural view, which is an inherent property of language. In this study, we explore the multilinguality of LLMs through representational structural analysis. Our findings reveal that low-resource languages are structurally more different from English than high- and mid-resource languages, and that language-specific post-training alters their structures while preserving inter-language relationships. 

---
# STaR-KV: Spatio-Temporal Adaptive Re-weighting for KV Cache Compression in GUI Vision-Language Models 

**Authors**: Yuhang Han, Wenzheng Yang, Yujie Chen, Xiangqi Jin, Yaojie Zhang, Siteng Huang, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.01790)  

**Abstract**: Vision-language-model-based graphical user interface (GUI) agents have shown broad automation capabilities, yet deployment is bottlenecked by a key-value (KV) cache that grows linearly with interaction steps. For instance, UI-TARS-1.5-7B consumes 76 GB of GPU memory on merely five screenshots, approaching the capacity of mainstream 80 GB accelerators. Existing KV compression methods share two structural assumptions: aggregating visual-token importance into a single shared saliency map, and applying a fixed top-B cutoff to the fused score distribution. Pilot measurements refute both: spatial specialization lives at the attention-subspace level and migrates across layers, while the score distribution drifts in shape along a trajectory. We propose STaR-KV (Spatio-Temporal Adaptive Re-weighting), a training-free KV cache compression framework that calibrates token importance along three axes: (i) subspace-aware scoring driven by online spatial mutual information; (ii) a temporal stability discount that suppresses redundant cache entries from persistently attended subspaces; and (iii) an entropy-derived temperature that adaptively reshapes the score distribution. Across four GUI benchmarks, STaR-KV achieves the strongest average accuracy among state-of-the-art KV compression methods (e.g., GUIKV, SnapKV) at matched budgets, with no compression-stage FLOPs overhead (-0.07%) and cutting peak GPU memory by nearly 40% at a 20% KV-cache budget. Code is available at this https URL. 

---
# Breaking the Information Silo: Semantic Personas for Cross-Domain Recommendation 

**Authors**: Jonathan Mayo, Moshe Unger, Konstantin Bauman  

**Link**: [PDF](https://arxiv.org/pdf/2606.01783)  

**Abstract**: Digital platforms increasingly operate as isolated information silos, limiting their ability to construct comprehensive user representations across domains. Cross-domain recommender systems seek to overcome this limitation by transferring knowledge from a source domain to a target domain, yet most existing approaches depend on shared users, shared items, or structurally similar interaction graphs. These assumptions are often unrealistic across independent platforms. We propose SPHERE (Semantic Personas for Heterogeneous cross-domain Recommendation), a design artifact that enables recommendation knowledge transfer across strictly disjoint domains with no shared users or items. Rather than aligning domains through identity or graph structure, SPHERE uses large language models to induce a shared behavioral vocabulary, generate structured semantic personas for users, and retrieve behaviorally similar source-domain communities that form a Community Source Persona. This semantic signal is integrated with collaborative signals through a dual-tower architecture and dynamic fusion gate, allowing SPHERE to augment standard recommender backbones. Empirical evaluation across Amazon Books, Goodreads, and Steam demonstrates consistent improvements over NCF, SVD++, and LightGCN baselines under full-ranking evaluation. The results show that cross-domain transfer effectiveness is not determined solely by semantic proximity between domains; rather, it depends critically on the structural density and native predictive strength of the target domain. The study contributes to information systems research by reframing cross-domain personalization as behavior-based semantic alignment, offering a practical mechanism for overcoming information silos while preserving interpretability and modularity. 

---
# FLARE: Diffusion for Hybrid Language Model 

**Authors**: Yuchen Zhu, Jing Shi, Chongjian Ge, Hao Tan, Yiran Xu, Wanrong Zhu, Jason Kuen, Koustava Goswami, Rajiv Jain, Yongxin Chen, Molei Tao, Jiuxiang Gu  

**Link**: [PDF](https://arxiv.org/pdf/2606.01774)  

**Abstract**: Autoregressive (AR) large language models (LLMs) have achieved broad practical success, but sequential decoding remains a key bottleneck for low-latency deployment. Recent efficient-inference work has progressed along two axes: reducing the cost of each model invocation through efficient architectures, and reducing serial decoding steps through parallel generation. Hybrid attention backbones address the former, while diffusion language models (dLLMs) pursue the latter via iterative parallel denoising. Combining these advantages remains challenging: AR-to-dLLM conversion often fails to preserve seed-checkpoint capability, and hybrid-attention recurrent states and masking constraints make diffusion training and serving nontrivial. We present FLARE, a systematic conversion framework for hybrid-attention LLMs. Our analysis identifies transfer data quality as the primary determinant of capability preservation, outweighing loss formulation and attention-mask design. The resulting framework combines a token-equal AR-and-diffusion objective, hardware-aware kernels, and unified inference, enabling one checkpoint to support both AR-style verified decoding and diffusion-style parallel denoising. Starting from strong AR checkpoints with limited post-training data, FLARE is competitive with leading open-source dLLMs across model scales and delivers consistent throughput gains over open-source dLLM baselines in single-GPU concurrent serving. Our results further suggest that practical dLLMs are limited not only by decoding algorithms, but also by transfer data quality and the training inefficiency of current block-diffusion objectives, motivating joint design of data, objectives, architectures, and inference systems. 

---
# Adaptive Auto-Harness: Sustained Self-Improvement for Agentic System Deployment on Open-Ended Task Streams 

**Authors**: Zewen Liu, Zhan Shi, Yisi Sang, Bing He, Minhua Lin, Tianxin Wei, Dakuo Wang, Benoit Dumoulin, Wei Jin, Hanqing Lu  

**Link**: [PDF](https://arxiv.org/pdf/2606.01770)  

**Abstract**: Auto-harness systems such as A-Evolve, GEPA, and Meta-Harness improve LLM agents by optimizing prompts, skills, tools, memories, and supporting infrastructure from execution feedback, but they are typically evaluated on fixed offline benchmarks. Real deployments instead present open-ended task streams: histories grow without a fixed endpoint, heterogeneous tasks require different harnesses, and problem distributions shift over time. These challenges make a single repeatedly and densely updated harness brittle, causing performance degradation as accuracy peaks early and then declines. This motivates sustained harness construction with task-wise adaptation. We introduce Adaptive Auto-Harness, a framework and system for such streams. The framework decomposes the gap to an oracle harness into evolution loss and adaptation loss. The system addresses these losses with a stateful multi-agent evolver, a harness tree with solve-time routing, and human-steering hooks for cases where history lacks the needed signal. Across prediction-market, security-competition, and event-forecasting streams, Adaptive Auto-Harness outperforms five existing auto-harness baselines and ablations attribute gains to better construction, routing, or targeted human steering. Code is available in this https URL . 

---
# Construction of Historical Knowledge Graphs Based on BERT and Graph Neural Networks 

**Authors**: Ping Li, Bartlomiej Brzozka  

**Link**: [PDF](https://arxiv.org/pdf/2606.01747)  

**Abstract**: Through digital humanities research and scale-up historical data analysis, a significant amount of traditional historical text is converted into structured knowledge graphs. This paper provides a high-level architecture that combines bidirectional encoder representations of transformers (BERT) and graph neural networks (GNN) to extract the entities and relationships from various types of historical texts. The texts of traditional history resolve linguistic ambiguities, references limited by context, and a lack of established grammatical norms in a systematic way. This study develops a new image retrieval system based on FastRQNet and pre-trained vision-language model Vilt-qaformer+RoBInet in accordance with the aforementioned recommendations. The experiments make full use of a comprehensive collection of municipal records, parliamentary documents, and historical correspondence. When compared to conventional rule-based techniques and other popular deep-learning baselines, the joint BERT-GNN system obtains greater Precision, Recall, and F1-score (Table 2). Complex nested structures and implicit reference issues can be handled by this structure with sufficient accuracy and thoroughness when creating knowledge graphs. The aforementioned experiments show that combining relational graph learning algorithms with context-sensitive semantic representation techniques can automatically extract historical data to add accumulated wisdom to the knowledge repository. 

---
# SECUREVENT: Hybrid AI/ML Security Monitoring for Distributed Event-Based Systems 

**Authors**: Eric Liang  

**Link**: [PDF](https://arxiv.org/pdf/2606.01741)  

**Abstract**: Distributed event-based systems have become a common substrate for Internet-scale publish/subscribe services, IoT telemetry, cloud-native microservices, and security operations pipelines. Their loose coupling and asynchronous delivery improve scalability, but they also expand the attack surface: publishers, brokers, subscribers, topics, schemas, and temporal ordering can each be abused without a single component observing the whole behavior. This paper proposes SECUREVENT, a hybrid AI/ML security-monitoring architecture for distributed event-based systems. The architecture combines traditional protections such as authenticated transport, topic-level authorization, and signed events with online anomaly detection, graph-aware behavioral features, complex-event policy rules, federated learning, and adversarial-ML governance. A deterministic prototype study over synthetic event-stream attacks illustrates how a hybrid AI/CEP monitor can improve recall over static rules while retaining a low false-positive rate. The central claim is not that machine learning replaces cryptographic and access-control mechanisms, but that model-based security monitoring is necessary when event flows, identities, schemas, and timing relationships are too dynamic for static controls alone. 

---
# THRD: A Training-Free Multi-Turn Defense Framework for Jailbreak Attacks on Large Language Models 

**Authors**: Zhiqing Ma, Zhonghao Xu, Dong Yu, Chen Kang, Changliang Li, Pengyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.01738)  

**Abstract**: Multi-turn jailbreak attacks pose a growing threat to LLMs by exploiting conversational dynamics such as gradual escalation and cross-turn coordination. Existing defenses either rely on costly retraining -- often degrading model utility -- or apply single-turn analysis independently at each turn, failing to capture how risk accumulates along interaction trajectories. We observe that safety behavior in multi-turn interaction is trajectory-dependent: dialogue history continuously reshapes the model's conditioning context, making it insufficient to evaluate each turn in isolation. Motivated by this insight, we present THRD, the first training-free framework that explicitly models temporal risk accumulation for multi-turn jailbreak defense. THRD integrates four modules: a Turn-level Risk Assessor (TRA) for instantaneous risk estimation, a Historical Context Analyzer (HCA) for cross-turn intent escalation detection, a Response Evaluator (RE) for identifying facilitative outputs, and a Decision Module that combines these signals through a time-evolving scoring mechanism with attenuation-based modulation and trend-aware adjustment. Experiments against state-of-the-art multi-turn attacks -- including tree-search-based and multi-agent collaborative methods -- across two target models show that THRD reduces ASR to 0.2--4.0% while preserving model utility within 1.5% degradation on MMLU and GSM8K. Ablation studies confirm non-redundant module contributions and stable cross-architecture generalization. Analysis of first rejection triggers reveals that over 70% of multi-turn attacks require Turn~2 or later to detect, validating the necessity of explicit temporal aggregation. 

---
# Argument Collapse: LLMs Flatten Long-Form Public Debate 

**Authors**: Yekyung Kim, Yapei Chang, Chau Minh Pham, Mohit Iyyer  

**Link**: [PDF](https://arxiv.org/pdf/2606.01736)  

**Abstract**: As LLMs are increasingly used to draft public-facing arguments, they may flatten public debate by repeatedly introducing the same polished, plausible arguments. We study argument collapse, the tendency of essays generated by different LLMs to converge to a smaller set of main arguments, sub-arguments, and paragraph-level structures. We compare 1,039 human responses from 195 New York Times (NYT) debates, 448 human responses from 61 longer-form Boston Review (BR) forums, and 23,384 LLM-generated essays. In the NYT corpus, 65.3% of human main arguments are unique within a debate, compared to 3.4% of LLM main arguments. Asking LLMs to generate diverse answers adds variation, but a typical model recovers only about half of the distinct human main arguments, with much of the added variation falling outside the observed human argument space. Collapse also appears in sub-arguments, where among essays with the same main argument, 41.0% of human sub-arguments are unique versus 9.1% from LLM responses. Qualitatively, LLMs often reuse generalized and hedged sub-arguments, while humans prefer more concrete and topic-specific ones. Structure-wise, LLM-generated essays tend to follow a more fixed arc, often opening with a direct claim and moving quickly toward proposals. The same patterns hold in longer BR essays, suggesting that argument collapse extends beyond short-form responses. 

---
# Shortcut to Nowhere: Demystifying Deep Spurious Regression 

**Authors**: Guanrong Xu, Jessica Li, Hao Wang, Yuzhe Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.01723)  

**Abstract**: Real-world regression often exhibits shortcuts: attributes that are spuriously correlated with continuous targets in training, yet unreliable under deployment shifts; regressing targets using such shortcuts may fail catastrophically at test time. Existing studies on spurious correlations focus primarily on classification, where labels are categorical and groups are naturally defined. However, many real-world tasks require continuous prediction, where hard label boundaries or discrete group-label pairs do not exist. We define Deep Spurious Regression (DSR) as learning from regression data with attribute-label confounding, addressing continuous spurious correlations, and generalizing to all attribute-label combinations at test time. Motivated by the intrinsic difference between classification and regression shortcuts, we propose to exploit the similarity among spurious attributes in both label and feature spaces, thereby accounting for nearby targets and related groups while calibrating both label and learned feature distributions across attributes. Extensive experiments on common real-world DSR datasets that span computer vision, environmental sensing, and large language model (LLM) regression verify the superior performance of our strategies. Our work fills the gap in benchmarks and techniques for studying spurious correlations in continuous prediction. 

---
# Post-Deterministic Distributed Systems: A New Foundation for Trustworthy Autonomous Infrastructure 

**Authors**: Jun He, Deying Yu  

**Link**: [PDF](https://arxiv.org/pdf/2606.01722)  

**Abstract**: For decades, distributed systems have typically assumed that correct participants execute protocol-specified behavior with stable, externally defined, and deterministic semantics. Classical theory has extensively parameterized network timing, communication topologies, and failure domains, but this participant model has remained comparatively fixed. The integration of autonomous reasoning engines, stochastic model-driven agents, and policy-driven actors into cloud control planes, incident response systems, and financial infrastructure challenges the universality of this assumption. These agents often produce divergent reasoning paths, distinct operational traces, and heterogeneous internal representations while achieving semantically equivalent and correct outcomes. In this paper, we introduce Post-Deterministic Distributed Systems (PDDS) as a research and engineering model for coordinating heterogeneous environments where deterministic code, stochastic models, and autonomous agents coexist. We show that classical distributed computing models form a zero-ambiguity special case of this participant-general model. We do not argue that deterministic systems disappear; rather, deterministic execution can no longer serve as the universal participant assumption for autonomous infrastructure. Finally, we outline five architectural pillars of post-deterministic infrastructure: Protocol-Driven Development, Verifiable Agentic Infrastructure, Autonomous State Control Planes, Semantic Quorum Assurance, and Epistemic State Replication. Epistemic State Replication extends persistence and consistency models from data visibility to knowledge visibility, enabling agentic memory, Verifiable Semantic Rollback, and coherence across reasoning participants. We also define a taxonomy of failure classes that arise in this setting. 

---
# Fair Finetuning Mitigates Distribution Inference Attacks 

**Authors**: Rakshit Naidu  

**Link**: [PDF](https://arxiv.org/pdf/2606.01719)  

**Abstract**: Machine learning models trained on sensitive data can inadvertently leak population-level information about their training distributions -- a threat known as distribution inference attack (DIA). An adversary with black-box access can infer sensitive demographic properties, such as subgroup proportions, without observing any training data directly. While defenses such as differential privacy and property unlearning have been proposed, the link between fairness constraints and distributional leakage remains unexplored. We propose Fair Fine-tuning (FFt): a trained model is fine-tuned on samples from the complementary distribution under an Equalized Odds (EO) constraint. We provide a complete theoretical characterization, proving the tight bound $\text{Adv}(\mathcal{A},M_f) \le \Delta_{\text{EO}} \cdot W$, where $W$ quantifies how distinguishable the two training distributions are by their sensitive-attribute composition. We also establish a necessary condition for FFt to reduce adversarial advantage and prove tightness of the bound. We evaluate across six datasets spanning tabular (ACS Income, COMPAS, German Credit), image (UTKFaces), and NLP (Bias in Bios) modalities. Rehearsal-based FFt consistently reduces the adversarial accuracy gap below the detection threshold $\tau!=!0.1$ across all settings; on ACS Income, the gap falls from $\sim!15%$ to under $4%$. Our work provides the first formal bound connecting a model's measured EO disparity directly to its adversarial advantage in the DIA game, opening a new avenue for unified fairness-and-privacy defenses. 

---
# Two-Fidelity Best-Action Identification for Stochastic Minimax Tree 

**Authors**: Peter Chen, Xi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.01708)  

**Abstract**: We study fixed-confidence best-action identification (BAI) in stochastic minimax trees. This problem is increasingly relevant in modern AI planning, where deep minimax search and Monte Carlo Tree Search (MCTS) with language model long rollouts face a fundamental tradeoff: heuristic evaluations are cheap but biased, while accurate rollouts are reliable but prohibitively expensive. We propose 2FFS, a two-fidelity tree-search algorithm that brings multi-fidelity flat bandit ideas into trees. The algorithm combines minimax-style fast expansion with MCTS-style stochastic sampling, adaptively deciding when to exploit cheap biased evaluations and when to invoke expensive accurate evaluations for local certification. We prove fixed-confidence correctness, establish finite stopping for exact identification, and give a polynomial-depth cost upper bound for general-depth trees. Across numerical stochastic-tree experiments, 2FFS uses substantially fewer samples and computational operations comparing to existing BAI-MCTS baseline. 

---
# JenBridge: Adaptive Long-Form Video Soundtracking across Scene Transitions 

**Authors**: Jiashuo Yu, Yao Yao, Boyu Chen, Alex Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.01703)  

**Abstract**: We address the challenge of generating high-fidelity, long-form soundtracks that remain coherent across scene transitions. Existing AI music systems are mainly designed for short, isolated clips and lack mechanisms to ensure narrative continuity. We present JenBridge, a modular and interpretable framework for adaptive long-form video soundtracking that ensures both high-fidelity audio generation and transition naturalness. The core architecture is a Transformer-based generative model trained with a flow-matching objective, following a two-stage paradigm: pretraining on large-scale text-audio corpora to establish robust musical priors, then adapting to the video domain with dual text-visual conditioning for precise cross-modal alignment. Crucially, to achieve long-form coherence across diverse scene changes, JenBridge incorporates a novel adaptive transition mechanism. This system features a versatile toolkit of transition styles, including a generative transition method, and uniquely employs a Large Language Model (LLM) Agent that acts as a director to select the most appropriate transition for each narrative shift intelligently. To rigorously assess this task, we propose the LVS Benchmark, a new benchmark that includes a curated dataset and novel evaluation metrics focusing on holistic and transition-aware assessment. Extensive experiments on the proposed benchmark demonstrate that JenBridge significantly outperforms existing methods in both objective and subjective metrics, particularly in terms of transition naturalness and overall narrative coherence. JenBridge represents a significant step towards fully automated, professional-quality video soundtracking. 

---
# Understanding Identity Continuity in Thermal Video through Scene-Level Consistency 

**Authors**: Wei-Chieh Sun, Gyungmin Ko, Heejae Kwon, Hsiang-Wei Huang, Jenq-Neng Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2606.01694)  

**Abstract**: Thermal pedestrian MOT remains challenging because weak appearance cues and frequent detection interruptions cause severe trajectory fragmentation. We study whether lightweight post-processing can recover identity continuity without relying on heavy re-identification models or complex online association. Starting from a YOLOv8 and SORT baseline, we add a modular identity-repair backend consisting of online short-gap remapping and offline tracklet relinking based on temporal, spatial, motion, and border cues. Controlled ablations on a fixed validation split and evaluation on the official PBVS Thermal Pedestrian MOT benchmark show that the main identity gains arise from conservative relinking, improving IDF1 from 82.25 to 84.93 while preserving MOTA, whereas many heuristic thresholds remain stable across broad operating ranges. These results suggest that, in low-information thermal imagery, robust identity recovery can be achieved more effectively through high-precision trajectory relinking than through increasing tracker complexity. These results provide a controlled analysis of identity recovery in thermal video, showing that scene-level spatial-temporal consistency plays a dominant role in identity continuity compared to local frame-to-frame association. 

---
# RPCASSM: Robust PCA State Space Model For Infrared Small Target Detection 

**Authors**: Pingping Liu, Aohua Li, Yubing Lu, Jin Kuang, Tongshun Zhang, Qiuzhan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.01689)  

**Abstract**: The detection and segmentation of infrared small targets have important application significance in the fields of surveillance and security, maritime rescue and so on. Due to the low occupancy of these targets in long-distance imaging, the mainstream visual state space model is inefficient and difficult to accurately model the target edge. The existing infrared state space models do not deviate from the mainstream visual state space structure framework from the structural properties of infrared small targets. In order to solve this problem, this paper proposes the RPCASSM network based on the model paradigm of robust principal component analysis(RPCA), which aims to design the background state space module(BSSM) and the target state space module(TSSM) by the nature of the infrared small target in the spatial domain. The BSSM aims to use the saliency of spatial heterogeneous signals to design a spatial probe scanning mechanism(SPCM) to model background information. The TSSM designs a deformable prompt scanning mechanism(DPCM) by using the sparsity and local highlight of the target to focus on the deformable space of the target for state space modeling. According to the above design, we effectively solve the problem that the existing mainstream vision state space model is difficult to accurately model the edge structure of infrared small target. Experimental results on the existing benchmark data sets prove the effectiveness of the RPCASSM design. Our code will be made public at \href{this https URL}{RPCASSM}. 

---
# HAIM: Human-AI Music Datasets for AI Music Production Tracking Benchmark 

**Authors**: Seonghyeon Go, Yumin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2606.01686)  

**Abstract**: As generative platforms such as Suno and Udio reach human-grade audio quality, the scope of AI's utility has expanded across the entire music production workflow. Beyond simple track generation, these advancements have catalyzed the adoption of AI-driven methodologies in diverse forms. These include vocal synthesis, arrangement, and professional mastering. However, current detection research remains largely confined to a binary `AI-or-human' paradigm. It fails to reflect the realities of contemporary music production workflows. In real-world production, AI tools are increasingly used to refine or master human-produced tracks, and human engineers likewise post-process AI-generated material to ensure professional quality. Moreover, users often employ adversarial tactics to bypass AI detectors, such as applying human mastering to AI-generated tracks. This creates a grey area that a simple binary classification fails to capture. In this paper, we define and investigate ``AI Music Tracking'': the challenge of identifying specific AI integration across the multifaceted spectrum of music production. To this end, we introduce HAIM, a dataset with diverse labels for stages of music production. It is designed to isolate stages of AI intervention, including hybrid production and agent-level tracking. Our evaluation of state-of-the-art detectors reveals systemic flaws. By releasing HAIM, we propose a new benchmark that shifts the field beyond binary classification toward a granular, structured evaluation of AI music. 

---
# Off-the-Shelf LLMs as Process Scorers: Training-Free Alternative to PRMs for Mathematical Reasoning 

**Authors**: Atoosa Chegini, Soheil Feizi  

**Link**: [PDF](https://arxiv.org/pdf/2606.01682)  

**Abstract**: Selecting the best response from multiple small-model samples using a stronger scorer is a simple inference-time strategy, but fails when the small model has already committed to incorrect reasoning paths. PRM guided search avoids this by scoring candidate continuations during generation, but requires a reward model trained with step-level labels.
We propose Chunk-Level Guided Generation, a training-free alternative that uses an off-the-shelf large language model as a process scorer. At each step, a small model samples k fixed-length candidate chunks, while the larger model scores the candidates using likelihoods without generating any text. The selected chunk is committed before the next step, steering generation before errors can propagate.
We instantiate this framework with two selection rules: Likelihood-Guided Selection (LGS), which selects the chunk with the highest length-normalized large-model log-probability, and Contrastive-Guided Selection (CGS), which subtracts the small model's log-probability to favor chunks where the large model's preference diverges from the small model's. We show that scoring variable-length reasoning steps with large-model likelihoods is unreliable due to a systematic length bias that persists even after length normalization, and that fixed-length chunks avoid this confound.
On GSM8K, MATH, Minerva Math, AMC23, and AIME24 with Qwen2.5-1.5B guided by Qwen2.5-32B and Llama-3.2-1B guided by Llama-3.1-70B, CGS outperforms majority voting by up to 28 pp and, under matched guidance budgets, matches or outperforms Qwen2.5-Math-PRM-72B guided search on most benchmarks without reward-model training. With Qwen2.5-7B guided by Qwen2.5-72B, CGS reaches 81.8% on MATH and 63.6% on Minerva Math at k=16, surpassing majority voting by 4--6 pp. Finally, Chunk-Level Guided Generation produces substantially shorter reasoning traces than PRM guided search. 

---
# Time-Aware Diffusion based on Preference Disentanglement for Generative Recommendation 

**Authors**: Bangguo Zhu, Peng Huo, Yuanbo Zhao, Zhicheng Du, Jun Yin, Senzhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.01670)  

**Abstract**: Recently, Generative Recommenders (GRs) have emerged as a transformative recommendation paradigm by replacing traditional item IDs with semantic indices (SIDs). Owing to the exceptional generative capabilities of diffusion models, a few pioneering works explore developing GRs with diffusion architectures as the backbone. However, a fatal limitation of existing diffusion-based GRs is that the diffusion process applies uniformly to all items within the historical interactions. In contrast, the user preference is shaped by multifaceted time-evolving factors and thus exhibits a non-stationary distribution in the temporal aspect. To bridge this gap, this study proposes a novel GR framework, named TDPM, by designing the time-aware diffusion on SID tokens. Specifically, TDPM explicitly integrates the impact of time-evolving user preferences into the diffusion process. In detail, the user preference is disentangled into (i) the period preference, which remains consistent over a long time-span, and (ii) the point preference, which is triggered by recent focal events. Extensive experiments on three public real-world datasets demonstrate the significant superiority of TDPM over the state-of-the-art baselines. TDPM achieves average improvements of up to 29.21% and 25.45% in terms of HR@20 and NDCG@20, respectively. The ablation study further underscores the necessity of time-aware token diffusion in diffusion-based GRs. 

---
# DOT-MoE: Differentiable Optimal Transport for MoEfication 

**Authors**: Udbhav Bamba, Arnav Chavan, Aryamaan Thakur, Steve Teig, Deepak Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2606.01666)  

**Abstract**: The scaling of Large Language Models (LLMs) has driven significant performance gains but created substantial challenges in inference efficiency. While Mixture of Experts (MoEs) architectures address this by decoupling model size from inference cost, training MoEs from scratch is often unstable and compute intensive. Conversion of pre-trained dense models into sparse MoEs has emerged as an alternative solution; however, existing methods typically rely on heuristic neuron clustering or random splitting to partition the Feed-Forward Network (FFN) into experts. In this work, we propose DOT-MoE, a novel framework that formulates the decomposition of dense layers as a Differentiable Optimal Transport (DOT) problem. Instead of static heuristics, we model neuron assignment as a balanced transport problem, utilizing differentiable Sinkhorn-Knopp iterations to enforce strict expert capacity constraints. Furthermore, we utilize Straight-Through Estimators (STE) to jointly learn the discrete neuron-to-expert assignment and the token-to-expert routing policy end-to-end. Extensive experiments across multiple architectures and benchmarks demonstrate that DOT-MoE significantly outperforms structured pruning, heuristic clustering, and random-split baselines, retaining 90% of the original dense model's performance while reducing active parameters by 50%. 

---
# MINTS: Minimalist Thompson Sampling 

**Authors**: Kaizheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.01655)  

**Abstract**: The Bayesian paradigm offers principled tools for sequential decision-making under uncertainty, but its reliance on a probabilistic model for all parameters can hinder the incorporation of complex structural constraints. We introduce a minimalist Bayesian framework that places a prior only on the location of the optimum, while eliminating nuisance parameters through profile likelihood. This yields a generalized posterior that naturally accommodates structural constraints. As a direct instantiation, we develop MINimalist Thompson Sampling (MINTS). For multi-armed bandits with mean constraints, we establish near-optimal non-asymptotic regret guarantees and sharp almost-sure asymptotic regret characterizations. In particular, MINTS attains the classical Lai--Robbins constant in the unstructured setting and automatically adapts to unimodal structure, achieving the sharp constant determined only by the immediate neighbors of the optimal arm. 

---
# Easier to Mislead Than to Correct: Harmful and Beneficial Revision in LLM Conformity 

**Authors**: Jiaming Qu, Lucheng fu, Yibo Hu  

**Link**: [PDF](https://arxiv.org/pdf/2606.01637)  

**Abstract**: Large language models are increasingly used in multi-agent systems, where they see and respond to other agents' answers. A key risk is conformity: a model may abandon its own answer simply because others agree on a different one. Prior studies show that LLMs often revise toward a majority answer, but it remains unclear whether these revisions help correct mistakes as often as they introduce new errors. In this paper, we conduct a controlled study in which an LLM first answers a question, then sees simulated peer responses before making a final decision. We manipulate two social cues: consensus structure and authority labels assigned to peers, and measure how they influence beneficial and harmful revisions. Across four open-weight LLMs and seven QA datasets, we find that peer agreement makes it much easier to mislead initially correct models than to correct initially wrong ones. Authority labels make models more likely to choose the endorsed answer, regardless of whether it is correct. More concerningly, generic reasoning interventions such as chain-of-thought and reflection do not reliably reduce harmful revision while preserving beneficial revision. These findings suggest that multi-agent LLM systems should verify peer answers rather than simply aggregate them. 

---
# AlphaToken: Decoupling Adaptation and Stability for Path-Aware Response Token Valuation in LLM Post-Training 

**Authors**: Liu Qing, Ou Wu, Yi Du  

**Link**: [PDF](https://arxiv.org/pdf/2606.01635)  

**Abstract**: Token selection is pivotal for effective LLM post-training. However, existing methods mostly rely on local heuristics and rarely formulate token selection as a principled valuation of individual response tokens. We introduce $\textbf{AlphaToken}$, a response token valuation framework that decouples valuation into $\textbf{adaptation}$ (promoting target-task learning) and $\textbf{stability}$ (preserving pre-trained capabilities), and makes each objective $\textbf{path-aware}$ by combining the direct-path signal from local token gradients with the downstream causal-path signal in autoregressive generation. Since retention data are typically unavailable, AlphaToken approximates stability via a $\textbf{Fisher-drift proxy}$ anchored at the pre-trained reference model. For efficient computation, we extend Ghost Dot-Product to token-level valuation. AlphaToken masks low-value response tokens during fine-tuning and preference optimization, concentrating training signals on more valuable positions. Experiments show that AlphaToken improves post-training performance and mitigates catastrophic forgetting. 

---
# E4GEN: Event-level Explainable Extreme-Enhanced Time-series Generation 

**Authors**: Lin Jiang, Dahai Yu, Ximiao Li, Guang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.01634)  

**Abstract**: Generating realistic time series is essential for scientific research and real-world applications. However, existing methods often emphasize overall distributional fidelity while failing to faithfully capture extreme events. To advance existing research, we propose E4GEN, an explainable diffusion framework for extreme event-aware time-series generation. E4GEN provides systematic insights into when, what, and how to control extreme-event generation through three key components. First, E-Activator learns the dataset-adaptive extreme-control signal activation step during the denoising process without interfering with regular temporal components, including trend and seasonality. Second, E-Predictor determines what control signal to enforce through Self-Driven Semantic Prediction, where each sample derives its own control signal by inferring latent extreme-event information during generation. It also includes a novel Data-Conditioned Training, Noise-Initiated Sampling mechanism to address the issue of unavailable training labels. Third, E-Control specifies how to control extreme-event generation through a trainable Extreme Control Network, which transforms the semantic control signal into layer-wise signals and injects it into the denoising process. We evaluate E4GEN on six datasets with 17 metrics, and extensive experiments show that E4GEN outperforms state-of-the-art models across multiple dimensions, including overall fidelity, extreme-event fidelity, and downstream utility. 

---
# A Framework for Graph-Conditioned Hierarchical Shapley Attribution in Patent Valuation 

**Authors**: Joy Bose  

**Link**: [PDF](https://arxiv.org/pdf/2606.01632)  

**Abstract**: Estimating the economic contribution of a single patent inside a product that embodies tens of thousands of patents is a long-standing unsolved problem in intellectual property economics. We propose PatentXAI, a framework that treats patent valuation as a problem of explainable AI: given a characteristic function v(S) encoding the revenue achievable by patent subset S, a patent's Shapley value measures its fair share of product profit in a way that satisfies efficiency, symmetry, dummy, and additivity. To make computation tractable we restrict each patent's coalition to its Markov Blanket inside a knowledge graph, grounded in the C-SVE conditional independence theorem (Li et al., 2020). Scaling experiments from n=12 to n=100 patents using Pareto-distributed coverage graphs report median Markov Blanket size of 32.9 percent of n at n=100, with 90th-percentile blanket size of 55.2 percent of n, and runtime of 10 milliseconds per patent. Difference against exact ground truth at n=12 is 0.088; difference against a high-sample Monte Carlo reference at n=100 is 0.062 plus or minus 0.003. A dense-component experiment shows that when 80 percent of patents share one component, the blanket correctly expands to cover that dense cluster, and the difference versus reference falls to 0.039 because the pooled computation becomes more accurate on homogeneous portfolios. Profit allocation proceeds hierarchically: exact Shapley distributes total profit among macro-components, then centrality-weighted Shapley distributes each component budget among covering patents. Estimating v(S) from real data is the primary open problem; we distinguish this from the computational contribution and outline a concrete roadmap for empirical validation using public ETSI, USPTO, and this http URL datasets. 

---
# Demystifying Multimodal Biomolecular Co-design With Intrinsic Geodesic Coupling 

**Authors**: Keyue Qiu, Xintong Wang, Zhilong Zhang, Hao Zhou, Wei-Ying Ma  

**Link**: [PDF](https://arxiv.org/pdf/2606.01628)  

**Abstract**: Biomolecules such as proteins and small-molecule ligands play a central role in biological systems, arising from the tight interplay between sequence and three-dimensional structure. Recent generative models for biomolecular co-design aim to capture this interplay by jointly modeling coupled modalities. However, existing approaches largely adopt a parallel execution of marginal generative processes, implicitly enforcing fixed synchronous coupling. We argue that a critical but overlooked degree of freedom lies in how these marginal processes are temporally coupled during training and generation, where inappropriate coupling can introduce high-variance supervision and inconsistent intermediate states, affecting modality consistency. To address this, we introduce GeoCoupling, a systematic framework that optimizes for temporal couplings between heterogeneous modalities. Empirical results across structure-based drug design and unconditional protein design demonstrate the learned couplings consistently outperform synchronous and randomly coupled baselines, yielding biomolecules with improved physical validity and diversity. 

---
# EvoPool: Evolutionary Programmatic Annotation for Label-Efficient Specialized Supervision 

**Authors**: Tianyi Xu, Yaolun Zhang, Xuan Ouyang, Huazheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.01617)  

**Abstract**: Large language models excel at general tasks but underperform smaller supervised models in specialized, high-stakes domains where training labels are costly. We address this regime with EvoPool, an evolutionary multi-agent framework inspired by Darwinian evolution. Three specialized agents iteratively propose executable annotator code, a small validation set provides a fitness signal, and a deterministic gate keeps only annotators that pass viability, diversity, and marginal-contribution checks across generations. Pool votes are mapped to soft training labels by EvoAgg, a text-aware aggregator combining semantic features with annotator-vote features. The authored pool runs at near-zero per-example cost and is 4500 to 31000x faster than LLM annotation on 100K examples. Across 7 of 8 LLM-weak specialized and complex tasks spanning biomedical relation extraction, legal-clause classification, complex reasoning, and dense multi-label biomedical classification, EvoPool beats the strongest LLM annotation baseline by an average +0.141 macro-F1, peaking at +0.301 on ChemProt and +0.265 on PubMed. Code is available at: this https URL 

---
# TechGraphRAG: An Agentic Graph-Augmented RAG Framework for Technical Literature Reasoning 

**Authors**: Kanwar Bharat Singh  

**Link**: [PDF](https://arxiv.org/pdf/2606.01613)  

**Abstract**: This paper presents an agentic retrieval-augmented generation (RAG) framework for domain-specific technical reasoning support, instantiated over a curated corpus of approximately 2,100 academic papers in intelligent tires, vehicle dynamics, and vehicle control. Unlike conventional single-pass RAG systems, the proposed architecture employs a 13-step autonomous pipeline that classifies queries by intent, scores evidence sufficiency against a multi-dimensional rubric, performs agentic retry with drift-guarded query reformulation, searches external academic databases (Crossref, OpenAlex, Semantic Scholar) through iterative optimize--search--vet loops, traverses a Neo4j knowledge graph for relational context, verifies citation integrity, and applies post-generation quality checks with automatic regeneration. Key contributions include a 100-point evidence sufficiency scoring framework across five dimensions with relevance damping and hybrid rule-based/LLM review; a route-dependent external search architecture with iterative agentic loops; a knowledge graph constructed via LLM-based entity extraction and OpenAlex author validation with intra-corpus citation resolution; and a self-correcting generation loop with citation verification and quality assessment. The framework is presented as a practical, implemented case study illustrating how agentic, evidence-grounded RAG can support literature navigation and technical reasoning over large, domain-specific corpora. 

---
# FedMTFI: Feature Importance Based Optimized Multi Teacher Knowledge Distillation in Heterogeneous Federated Learning Environment 

**Authors**: Nazmus Shakib Shadin, Aaron Cummings, Xinyue Zhang, Bobin Deng  

**Link**: [PDF](https://arxiv.org/pdf/2606.01607)  

**Abstract**: Federated learning (FL) is a decentralized approach that enables collaborative model training without exposing raw data. Instead of transferring sensitive data, it allows devices to share only model weights, keeping personal data locally and secure. However, in real world settings, the data held by devices is often not evenly distributed and devices mostly differ in computing power and memory capacity. These differences make FL harder to maintain consistent performance across the system. To address these issues, we propose FedMTFI, a novel architecture that combines multi-teacher knowledge distillation (MTKD) with feature importance to improve the FL process in heterogeneous environments. In FedMTFI, clients are clustered based on similar hardware and model types. Each cluster trains a specific model on not independently and identically distributed (non-IID) data. Within a cluster, every client updates that model using only its own local private data. The server then aggregates the locally trained models in each cluster using FedAvg to form multiple prototype models. Then these prototypes serve as teacher models to train a global generalized student model using MTKD. What makes FedMTFI more unique is the integration of Shapley values (SHAP) to emphasize important features during distillation, which enhances both accuracy and interpretability. Experimental results show that FedMTFI achieves higher accuracy than traditional FL algorithms and performs more effectively under non-IID data conditions. 

---
# Estimating Mutual Information between Time Series and Temporal Event Sequences Across Diverse Analysis Tasks 

**Authors**: Haoji Hu, Huaqing Mao, Yijun Lin, Xiaowei Jia, Jinwei Zhou, Minoh Jeong, Yao-Yi Chiang  

**Link**: [PDF](https://arxiv.org/pdf/2606.01602)  

**Abstract**: Pairwise dependence measures such as correlation and causality are fundamental to temporal data mining, yet there is still no principled and robust way to quantify dependence between heterogeneous data types, especially between continuous time series and discrete temporal event sequences. Existing approaches rely on ad hoc transformations or mutual-information estimators that are highly sensitive to quantization, repeated values, and event redundancy, leading to biased or unstable results in practice. We propose a nonparametric mutual information estimator that directly measures the dependence between time series and event sequences without data transformation, learning, or ad hoc discretization. Our method models the continuous-discrete duality of real-world time series to handle quantization and repeated-value artifacts and introduces a latent event clustering strategy to mitigate bias from event co-occurrence and redundancy. Together, these yield a robust and unified framework that bridges discrete and continuous mutual information. We evaluate the proposed estimator on four representative tasks: discrete-continuous time-delayed mutual information for causality analysis, global and local temporal repetition discovery, discrete covariate selection for time series forecasting, and continuous feature selection for classification. Experiments on synthetic and real-world datasets show consistent improvements over existing methods in accuracy, robustness, and interpretability, positioning our approach as a general-purpose dependence operator for heterogeneous temporal data, similar to Pearson correlation for homogeneous time series. Code available at: this https URL 

---
# Identifying High-Confidence Social Biases in LLMs for Trustworthy Conversational Tutoring Agents 

**Authors**: Aitor Arronte Alvarez, Naiyi Xie Fincham  

**Link**: [PDF](https://arxiv.org/pdf/2606.01584)  

**Abstract**: Conversational tutoring agents have been shown to improve learning engagement and student outcomes, and large language models (LLMs) are increasingly used in these systems to provide scalable, personalized feedback. However, LLMs may perpetuate or amplify stereotypical social biases, posing particular risks in educational settings.
In this study, we evaluate LLMs in conversational tutoring scenarios to identify high-confidence social biases, instances where models are unable to identify biased judgments in tutoring conversations while maintaining strong confidence in their assessments, potentially affecting their reasoning and the feedback they provide to learners. We present a new dataset generation method that enables bias evaluation under naturalistic instructional conditions by regenerating student-AI tutor interactions and introducing turns with controlled bias derived from a benchmark dataset.
Using this data, we assess multiple LLMs' ability to detect stereotypical biases and analyze the confidence and reasoning underlying their responses through computational and human evaluations. We find that bias detection is substantially more challenging in conversational tutoring contexts than in benchmark-based evaluations, and that state-of-the-art LLMs are overconfident in their incorrect assessments of stereotypical bias statements. Moreover, model confidence strongly influences reasoning and feedback, highlighting the risks of overconfident, biased behavior in LLM-based tutoring agents. We conclude by discussing implications, mitigation considerations, and directions for future research. 

---
# Defenses & Enablers For Skill Injection Attacks on Terminal Based Agents 

**Authors**: Yoshinari Fujinuma, Varun Gangal, Traian Rebedea, Makesh Narasimhan Sreedhar, Prasoon Varshney, Rebecca Qian, Anand Kannappan  

**Link**: [PDF](https://arxiv.org/pdf/2606.01567)  

**Abstract**: Large language model (LLM) agents increasingly rely on reusable skills i.e. documents describing task-specific procedures. However, this introduces a new attack surface for agents to manage. We study two complementary directions for this threat. First, we evaluate guardian-based defenses: an intermediary LLM agent that acts as a mediator for skill file access (dynamic guardian) or pre-rewrites these files at build time (static guardian). Across three LLM agent families, our guardians cut attack success rate (ASR) by well over half while preserving task utility. Second, we stress test them through attack reframing using four attacks that preserve the malicious instruction but change the phrasing. For non-guardian setup, the reframing pushes the ASR up to 81.4\%, but the dynamic guardian brings it down to 18.6\%, showing that real-time mediation is a robust defense. 

---
# GJDNet: Robust Graph Neural Networks via Joint Disentangled Learning Against Adversarial Attacks 

**Authors**: Canyixing Cui, Tao Wu, Xingping Xian, Xiao-Ke Xu, Mao Wang, Weina Niu  

**Link**: [PDF](https://arxiv.org/pdf/2606.01560)  

**Abstract**: Graph Neural Networks (GNNs) are vulnerable to adversarial attacks, which inherently invert connectivity patterns by introducing disassortative edges in assortative graphs and assortative edges in disassortative graphs. This structural inversion creates structure-feature mismatches that disrupt neighborhood aggregation across different graph types. However, we find that existing defenses are limited, as they either treat neighborhoods as monolithic under fixed assortativity assumptions or rely on standard softmax classifiers that fail to account for perturbation-induced representation shifts. To further exploit this observation, we adopt a robustness perspective that jointly disentangles node representations and decision spaces, isolating perturbation effects while enforcing well-separated decision regions. Based on this principle, we propose Graph Joint Disentanglement Network (GJDNet), a unified framework for robust node classification across diverse graph assortativity regimes. GJDNet enhances robustness at both representation and decision levels: it employs feature-driven soft structural disentanglement with skewness-aware neighbor filtering to suppress perturbation-induced structure-feature mismatches, and introduces a Spherical Decision Boundary (SDB) to promote intra-class compactness and inter-class separation in the embedding space, thereby stabilizing decision boundaries under perturbations. Theoretical analysis provides insights into the effectiveness of the proposed disentangled representation and decision mechanisms, while extensive experiments demonstrate that GJDNet consistently achieves strong robustness across graphs with different connectivity regimes. 

---
# Self-Conditioned Positional HNSW for Overlap-Aware Retrieval in Chunked-Document RAG Systems: Method and Industrial Evidence-Quality Audit 

**Authors**: Nataraj Agaram Sundar, Tejas Morabia  

**Link**: [PDF](https://arxiv.org/pdf/2606.01542)  

**Abstract**: Chunked-document retrieval is a common component of retrieval-augmented generation (RAG) systems. Documents are split into overlapping chunks, embedded, and indexed with approximate nearest-neighbor search such as hierarchical navigable small world graphs (HNSW). Overlap improves boundary coverage but induces a practical failure mode: top-k retrieval often returns near-adjacent chunks that repeat evidence and waste prompt budget. We propose Self-Conditioned Positional HNSW (SCP-HNSW), a lightweight modification that appends a low-dimensional positional code to chunk embeddings and uses a two-pass query procedure to estimate and apply a query-specific document-position prior. SCP-HNSW leaves HNSW graph construction and traversal unchanged while adding an auditable minimum-index-gap selector for final context construction. We also integrate industrial review artifacts for generated evidence quality: a 770-review text-evidence audit with 318 fully labeled reviews and a 70-case OCR audit with 350 ratings. The text audit shows that 574 of 770 projected reviews are rated 3/5, only 39 fall in the 1-2 range, and narrative reviewer detail appears much more often than structured issue flags. The OCR audit shows slice-level pass rates from 95% for clean chat screenshots to 45% for handwritten/blurry captures, with moderate to strong agreement. These results motivate overlap-aware, audit-friendly RAG retrieval and identify the remaining controlled retrieval ablations needed for causal performance claims. 

---
# TN-SHAP-G: Graph-Structured Tensor Network Surrogates for Shapley Values and Interactions 

**Authors**: Farzaneh Heidari, Guillaume Rabusseau  

**Link**: [PDF](https://arxiv.org/pdf/2606.01540)  

**Abstract**: Shapley values are a widely used tool for attributing importance and interactions among input variables in black-box models, but their computation involves a function defined over an exponentially large space of subsets. We propose TN-SHAP-G, a framework that exploits structure in graph-structured inputs to compute Shapley values and higher-order interaction indices efficiently. Given a predictor and a fixed masking scheme, TN-SHAP-G learns a compact, graph-aligned multilinear surrogate that approximates the masked-input behavior, represented as a tensor network whose topology mirrors the input graph. Once trained from a small number of oracle queries, the surrogate enables deterministic recovery of first- and higher-order Shapley indices via the multilinear extension, without additional model queries or Monte Carlo variance. Experiments on molecular benchmarks show that the learned factorization closely matches exact Shapley values on small graphs and scales efficiently to larger graphs where sampling-based methods become infeasible. 

---
# Compliance-Scored Best-of-N Guardrail Orchestration for Multimodal Document Generation in Payments Dispute Defense 

**Authors**: Nataraj Agaram Sundar, Tejas Morabia  

**Link**: [PDF](https://arxiv.org/pdf/2606.01513)  

**Abstract**: High-stakes enterprise document generation, including financial dispute narratives, compliance notices, and audit summaries, demands schema correctness, policy compliance, and low-latency operation at scale. Prior to a unified guardrail layer, production systems often stitched together separate PII redaction, content moderation, and format validation steps, leading to fragmented logic, slower request paths, and higher operational cost.
We present a guardrail orchestration layer for text and image inputs that couples multi-candidate generation with an explicit compliance score used for early exit. The framework runs configurable parallel generation heads, scores candidates against weighted guardrails including PII detection, content moderation, schema constraints, and domain rules, and returns the best-scoring output with selection metadata. The available operational readout reports 5 attempts within 20 seconds and 91 percent compliance.
For payments dispute defense summaries, we analyze aggregate operational scenario readouts rather than a randomized A/B test. Variable cohorts show higher count win rates than controls overall, 301/659 versus 536/1548, corresponding to +11.0 percentage points with 95 percent confidence interval [6.6, 15.5] and p < 0.001, and for adjusted item-not-received cases, +7.5 percentage points with 95 percent confidence interval [0.2, 15.7] and p = 0.045. Fraud and local evidence-ranking deltas are directionally positive but not statistically significant from the aggregate count data. We also report reviewer-calibrated Responsible-AI evidence-quality signals from 770 generated-evidence reviews and a 70-case OCR slice, and document the reproducibility boundary through the request interface, scoring logic, pseudocode, and operational evidence boundary. 

---
# ProbMoE: Differentiable Probabilistic Routing for Mixture-of-Experts 

**Authors**: Heng Zhao, Zilei Shao, Guy Van den Broeck, Zhe Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2606.01509)  

**Abstract**: Mixture-of-Experts (MoE) models scale by activating only a small subset of experts per token. However, training such models remains challenging because top-$k$ routing is discrete and non-differentiable, requiring gradient estimators for expert selection whose design remains a central open problem. We introduce ProbMoE, a probabilistic routing framework that models expert selection as a distribution over cardinality-constrained expert subsets and formulates routing as probabilistic inference in this discrete subset space. We first propose ProbMoE Exact-$k$ routing, which samples $k$-expert subsets in the forward pass, and the backward pass uses gradients through each expert's exact marginal probability as a tractable surrogate for the true gradient. ProbMoE naturally generalizes to a dynamic-$k$ routing setting, where both training and inference constrain the routing cardinality to the same predefined range, allowing adaptive expert allocation per token. Across benchmarks and model backbones, ProbMoE Exact-$k$ achieves strong performance compared to competitive baselines, with improved expert utilization and routing diversity; ProbMoE Dynamic-$k$ achieves comparable performance with fewer activated experts. 

---
# Agent Operating Systems (AOS): Integrating Agentic Control Planes into, and Beyond, Traditional Operating Systems 

**Authors**: Ankur Sharma, Deep Shah  

**Link**: [PDF](https://arxiv.org/pdf/2606.01508)  

**Abstract**: Traditional operating systems were designed around deterministic programs, explicit control flow, and human initiated workflows. Their core abstractions processes, threads, system calls, files, and permissions assume bounded behavior and predictable interaction patterns. Agentic AI systems introduce a different execution model: long-lived, goal-directed entities that reason probabilistically, invoke tools dynamically, and adapt behavior based on feedback. While agents can be implemented as user-space applications today, their execution characteristics stress OS boundaries in scheduling, memory and state management, security, observability, and governance. This paper introduces the concept of an Agent Operating System (AOS), a systems architecture that integrates an agentic control plane into existing operating systems or, in some models, subsumes selected OS responsibilities over time. We provide a precise definition of an AOS, explicit assumptions and non-goals, and a structured decomposition of AOS responsibilities into schedulers, context and memory management, tool and capability registries, policy and trust enforcement, and observability and audit. We analyze limitations of classical OS abstractions for agent workloads, propose integration models from user-space runtimes to distributed control planes, and map AOS concepts onto Linux and Windows primitives. We present security and safety implications, including agent specific threat models, and define evaluation criteria that emphasize deterministic enforcement, auditability, and operator comprehensibility. The objective is not to replace operating systems wholesale, but to establish a rigorous systems foundation for agentic computation that remains controllable, accountable, and secure at scale. 

---
# On the Limits of Token Reduction for Efficient Unified Vision Language Training 

**Authors**: Siyi Chen, Weiming Zhuang, Jingtao Li, Lingjuan Lv  

**Link**: [PDF](https://arxiv.org/pdf/2606.01503)  

**Abstract**: Unified vision-language models (VLMs) integrate visual understanding and visual generation within a single autoregressive backbone, but their joint training is computationally expensive and largely overlooked from an efficiency perspective. In this work, we study the feasibility and limits of token-reduction-based acceleration for unified VLM training. Through a systematic analysis of layerwise attention allocation, we uncover a fundamental asymmetry: visual understanding exhibits substantial late-layer visual redundancy, whereas visual generation maintains persistent dependence on image tokens across depth. Guided by this observation, we design task-specific accelerators that selectively reduce image-token computation for each objective. While these methods achieve significant efficiency gains in isolated settings, we observe a consistent synergy loss under unified training -- task-specific token dropping necessitates divergent parameter pathways and eliminates the mutual performance gains typically observed in joint optimization. Our findings suggest that efficient unified modeling requires preserving shared cross-task structures, highlighting the need for synergy-aware acceleration strategies. Project page: this https URL. 

---
# Move the Query, Not the Cache: Characterizing Cross-Instance Latent Attention Redistribution Across GPU Fabrics 

**Authors**: Bole Ma, Jan Eitzinger, Harald Köstler, Gerhard Wellein  

**Link**: [PDF](https://arxiv.org/pdf/2606.01502)  

**Abstract**: Frontier LLMs increasingly decide what a query attends to with a sparse-attention indexer that picks a few KV-cache blocks per query: attention's unit is now a small, reusable chunk. Agentic workloads hammer it: many sub-agents query one large codebase, reusing the same blocks. When that corpus outgrows one GPU it is partitioned across instances, so a query and the blocks it selects often sit on different GPUs: answering it means attention across instances. The reflex of prior cross-instance KV systems is to move the cache: pull the selected blocks to the requester. Multi-head Latent Attention inverts the arithmetic, compressing each token's key and value into one narrow vector, so a routed query row is only ~1 KB, smaller than the chunk it attends; routing the query is then often cheaper than moving the cache. Which primitive wins, over which fabric and request shape, is uncharted, least of all on device-initiated RDMA that makes per-request cross-node transfers cheap. We characterize cross-instance MLA attention on a real multi-node H100 cluster, distilling two reusable artifacts: a topology-aware cost model (probe / transfer / compute / return / merge) and a closed-form route/fetch/local predicate, whose constants we measure on real IBGDA, where the model tracks batched round-trips to within ~7%. At decode it routes the query, trading the cost of moving the cache (a ~3 ms re-adaptation splice for a contiguous chunk, or a scattered gather under selection) for a tens-of-microsecond round trip, and picks the fabric by probe latency, not peak bandwidth. We instantiate the cost model and predicate for MLA, but neither is MLA-specific: they apply wherever compression or sparse selection shrinks attention to small chunks (DeepSeek-V3.2, V4, and GLM-5.1 today). Extending them to a new architecture requires measuring just two coefficients: the routed payload and fetch's move-the-cache cost. 

---
# TimeSage-MT: A Multi-Turn Benchmark for Evaluating Agentic Time Series Reasoning 

**Authors**: Yaxuan Kong, Qingren Yao, Yuqi Nie, Yichen Li, Yilei Shao, Stefan Zohren, Anna Vettoruzzo, Joaquin Vanschoren, Ming Jin, Qingsong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2606.01498)  

**Abstract**: Time series data inform critical decisions across many real-world domains. While large language model (LLM) agents can analyze data through natural language and tools, it remains unclear whether they can conduct reliable time series analysis across multi-turn conversations. Existing benchmarks focus on single-step tasks such as forecasting and anomaly detection, overlooking practical workflows where user goals evolve, agents must build on prior analyses, and conclusions emerge from accumulated evidence. In this work, we introduce TimeSage-MT, a multi-turn benchmark for agentic time series reasoning with 240 tasks and 2,680 dialogue turns across 8 real-world domains, spanning basic exploration to decision-oriented analysis. TimeSage-MT is built through a reproducible pipeline that converts real-world time series data into multi-turn conversations with verifiable answers. It provides a unified evaluation protocol and public leaderboard for comparing time series agentic systems. To demonstrate the benchmark's utility, we evaluate frontier LLMs alongside TimeSage, a novel structured agent equipped with a comprehensive time series skill library. The results show sharp performance drops on decision-oriented tasks, driven by failures in memory, uncertainty handling, and domain-based decision making. TimeSage-MT exposes critical gaps in current agentic reasoning and provides a rigorous foundation for future development. 

---
# ClawHub Security Signals: When VirusTotal, Static Analysis, and SkillSpector Disagree 

**Authors**: Vincent Koc, Patrick Erichsen, Jacob Tomlinson, Agustin Rivera, Michael Appel, Nir Paz  

**Link**: [PDF](https://arxiv.org/pdf/2606.01494)  

**Abstract**: Agent skills extend AI agents with reusable instructions, tools, scripts, references, and workflows, establishing a security boundary distinct from both model safety and traditional package-malware detection. ClawHub Security Signals is a sanitized dataset of 67,453 latest public OpenClaw skill versions. Each row pairs redacted this http URL content and sanitized bundled files where present with a final ClawScan registry verdict and evidence from three scanner families: VirusTotal, static heuristic analysis, and NVIDIA SkillSpector.
Rather than estimating malicious-skill prevalence, we study scanner disagreement. The three scanners rarely flag the same skills: any pair overlaps on at most 10.4% of their combined positives, only 0.69% of skills are flagged by all three, and 81.9% of flagged skills are identified by a single scanner. The disagreement is structured by attack surface. SkillSpector, which raises semantic agentic-risk advisories rather than malware-reputation signals, is positive for 19,209 of 25,504 suspicious rows (75.3%) but only 14 of 206 malicious rows (6.8%). The malicious-verdict region shows the inverse profile: 150 of 206 malicious rows (72.8%) are VirusTotal-positive, consistent with bundled-code malware evidence.
These results show that agent-skill security requires layered governance, not single-scanner allow/block decisions. The corpus is released as a sanitized silver-standard dataset: labels are the registry's automated verdicts, not human-annotated ground truth, and the release represents an early, versioned snapshot intended to support the community while a human-annotated subset is developed. Further research is encouraged, including models tailored for skill-security triage. 

---
# LLM Consortium for Software Design Refinement: A Controlled Experiment on Multi-Agent Collaboration Topologies 

**Authors**: Nagarjuna Kanamarlapudi, Praveen K  

**Link**: [PDF](https://arxiv.org/pdf/2606.01490)  

**Abstract**: We present a controlled experiment evaluating 12 multi-agent LLM collaboration topologies for software architecture design. Using a $2\times2\times2$ factorial design (Authority $\times$ Roles $\times$ Dynamics), we conducted 520 experimental runs across 8 design tasks of varying complexity, with 5 repetitions each. Designs were evaluated on a 12-dimensional rubric by three independent automated evaluators (GPT-OSS 120B, Claude Opus 4.6, Claude Sonnet 4.6). We report four core findings. First, structural adversarial (v4b) ranks #1 by ensemble -- a prompt-engineered adversarial variant that demands rewrite mandates rather than patches (weighted ensemble: 4.637/5.0). Second, cross-model review wins unanimously at #2 -- generate with one model, review with another -- ranking #2 by all three evaluators (weighted ensemble: 4.606). Third, evaluator diversity is itself a finding -- all three evaluators agree v4b is best and v3 is worst, but disagree sharply on v2b (Claude d=1.44 vs. GPT-OSS d=0.45), revealing how different model families weight design qualities. Fourth, parallel merge is fundamentally broken -- all three evaluators place merge variants in the bottom tier (3.65-3.79), due to token starvation and the Frankenstein effect. The weighted ensemble ($2\times$Opus + $2\times$Sonnet + $1\times$GPT-OSS) provides robust rankings across 520 runs, confirmed through independent cross-validation. 

---
# MURMUR: An Efficient Inference System for Long-Form ASR 

**Authors**: Wei-Tzu Lee, Keisuke Kamahori, Baris Kasikci  

**Link**: [PDF](https://arxiv.org/pdf/2606.01483)  

**Abstract**: Long-form automatic speech recognition (ASR) requires both high accuracy and low latency, but existing systems force a trade-off between the two. Chunk-based pipelines process audio in parallel windows for low latency, but lose cross-chunk context and need brittle heuristics to align speakers and timestamps at boundaries. Long-context ASR models resolve everything in a single pass for better accuracy, but are an order of magnitude slower. We propose Murmur, an inference system that overcomes this trade-off by operating at two levels. At the inter-chunk level, we revisit the chunk-based pipeline for modern long-context ASR, treating chunk size as a tunable hyperparameter, and show that intermediate chunk sizes strike a good balance of accuracy and latency. At the intra-chunk level, we exploit attention sparsity through a sliding window KV cache eviction policy applied to both output and speech tokens. On AMI-IHM, Murmur matches single-pass accuracy while reducing latency by 4.2x, with further gains from token eviction at less than 1% relative tcpWER degradation. The code of Murmur is available at this https URL. 

---
# Crazyflow: An Accurate, GPU-Accelerated, Differentiable Drone Simulator in JAX 

**Authors**: Martin Schuck, Marcel P. Rath, Yufei Hua, AbhisheK Goudar, SiQi Zhou, Angela P. Schoellig  

**Link**: [PDF](https://arxiv.org/pdf/2606.01478)  

**Abstract**: High-quality, large-scale synthetic data from simulations is becoming a cornerstone for pushing the capabilities of robot algorithms. While aerial robotics simulators have evolved to support specialized needs such as fidelity, differentiability, and swarms independently, a unified platform that can synthesize data across all these domains is missing. In this work, we propose Crazyflow, a simulator designed to push the limits of aerial-robotics algorithm development, from model-based to data-driven methods, gradient-based to sampling-based approaches, and single-agent to multi-agent systems. Compared to existing state-of-the-art drone simulators, it achieves speeds more than an order of magnitude faster for a single drone and can simulate thousands of swarms of 4000 drones each. Real-world experiments show Crazyflow supports both analytical-gradient-based policy learning, achieving sub-centimeter trajectory tracking accuracy without domain randomization, and sampling-based obstacle avoidance at speeds exceeding half a billion steps per second. Breaking the traditional train-then-deploy paradigm, we show that its unprecedented speed even enables in-flight reinforcement learning; we demonstrate this by throwing a physical drone into the air and training a recovery policy from scratch in 0.38 seconds, successfully stabilizing the drone. Crazyflow supports multiple levels of simulation abstraction, is directly compatible with all open-source Crazyflie models, and enables rapid reconfiguration across custom drone platforms and applications by providing a light-weight system identification pipeline. By pushing accuracy, speed, and differentiability simultaneously, Crazyflow serves as an open-source resource for synthetic data generation, with emerging capabilities for large-scale parallelization for online, in-execution learning and optimization, opening the door to novel algorithm development. 

---
# Hierarchical Online Prompt Mutation with Dual-Loop Feedback for Guardrailed Evidence Document Generation: A Production-Evaluation Case Study 

**Authors**: Nataraj Agaram Sundar Tejas Morabia  

**Link**: [PDF](https://arxiv.org/pdf/2606.01472)  

**Abstract**: High-stakes production document-generation systems require language models to be adaptive, evidence-grounded, and auditable. We present HOPM, a hierarchical online prompt mutation framework evaluated on a real marketplace dispute-evidence workflow. HOPM treats prompts as online policies: a family/version router selects a prompt, deterministic guardrails attribute failures to mutable prompt-token categories, and dual feedback from human review and an automated judge updates both routing and mutation priorities. The primary evidence is an observed matched production-evaluation ablation: seven variants are evaluated on the same 600 cases each, enabling component comparisons against static prompting, manual iteration, bandit-only routing, mutation-only adaptation, human-only feedback, auto-judge-only feedback, and full dual-loop HOPM. Full HOPM improves count win rate over a static control from 34.7% to 45.7% (+11.0 pp; paired McNemar p = 1.31e-11) and amount-weighted win rate from 22.3% to 41.4% (+19.1 pp; 95% paired bootstrap CI [10.3, 28.9] pp). It also increases mean Likert quality from 3.18 to 4.40 and reduces issue-flag rate from 15.3% to 5.2%. Supporting review artifacts cover 770 generated-text reviews, 318 labeled reviewer exports, a 10-case/61-rating calibration slice, and a 70-case/350-rating OCR benchmark; these artifacts calibrate rubric, guardrail, title-risk, and OCR-risk interpretation rather than substituting for the production ablation. The paper includes control setup, sample sizes, confidence intervals, paired tests, prompt-token categories, pseudocode, schema, rubric, guardrail taxonomy, and a constructed example so the evaluation structure can be reproduced without exposing proprietary evidence. 

---
# Emergent Transfer of a Physics Foundation Model from Simulation to Laboratory Turbulence 

**Authors**: Payel Mukhopadhyay, Stefan S. Nixon, Romain Watteaux, Michael McCabe, Alberto Bietti, Kyunghyun Cho, Cristiana Diaconu, Irina Espejo Morales, David Fouhey, Siavash Golkar, Tom Hehir, Shirley Ho, Jake Kovalic, Geraud Krawezik, Francois Lanusse, Tanya Marwah, Rudy Morel, Mariel Pettee, Helen Qu, Jeff Shen, Hadi Sotoudeh, Stuart B. Dalziel, Miles Cranmer  

**Link**: [PDF](https://arxiv.org/pdf/2606.01470)  

**Abstract**: Whether physics foundation models can be usefully deployed on laboratory experiments remains an open question for scientific machine learning (ML). We test this question on the Rayleigh-Taylor instability (RTI), a ubiquitous and demanding fluid instability seen from tabletop flows to supernova explosions, in which small perturbations at a density interface grow into chaotic, multiscale mixing as a lighter fluid accelerates into a heavier one. Standard ML models struggle with RTI, and despite over a century of theoretical, numerical, and experimental work, it carries an unresolved discrepancy between simulation and experiment: the late-time mixing growth rate, $\alpha$, measured in most laboratory experiments ($\sim$ 0.06-0.07), is roughly three times the value from idealized direct numerical simulations (DNS, $\sim$ 0.02). The gap's origin remains debated. These properties make RTI a stringent test for a question that matters well beyond RTI: can foundation models trained only on simulations generalise to sparse, messy, and noisy laboratory settings? We finetune Walrus, a foundation model for continuum dynamics, on three or fewer DNS realizations and recover key RTI physics over long rollouts. Applied zero-shot to sliding-barrier laboratory data, the finetuned model leaves the DNS-like regime and enters the observed growth band, having never seen a single experimental sample. These results provide independent, data-driven evidence that initial conditions play a crucial role in the longstanding sim-experiment gap in $\alpha$. The model also generalises zero-shot to stable stratification, a buoyancy regime absent from training, correctly slowing mixing-layer growth. Together, our results show that foundation models can generalise well beyond their training data, predicting laboratory behavior and unseen physical regimes, opening new ways to probe longstanding simulation-experiment gaps. 

---
# Computation-Aware Kalman Filtering with Model Selection for Neural Dynamics 

**Authors**: JR Huml, Jonathan Wenger, John P. Cunningham  

**Link**: [PDF](https://arxiv.org/pdf/2606.01468)  

**Abstract**: Due to their explicit priors and ability to model uncertainty, Bayesian methods have played a major role in dynamical latent variable modeling of single-cell neural recordings. However, modern-sized datasets have made overparameterized deep networks the preferred methods of choice due to their predictive power and favorable computational scaling. While many posterior approximations exist, all incur approximation errors. Recent work accounts for this error in the form of computational uncertainty but comes at the cost of quadratic complexity and assumes fixed model hyperparameters. Here we extend this development to model selection, including a novel training loss and optimization scheme, which yields tractable inference in large state-spaces. We introduce a framework, the Computation-Aware State-Space Model (CASSM), specifically designed for the scale-imbalanced regime, where the number of trials is significantly lower than the number of recorded neurons. In this regime, for both synthetic and real data, we show that our method is competitive with data-hungry deep networks, with significantly improved uncertainty calibration over previous attempts to scale Bayesian methods. Our experiments provide a roadmap to neuroscience researchers in choosing from a host of potential dynamical latent variable models given key dataset properties and constraints. 

---
# UR-JEPA: Uniform Rectifiability as a Regularizer for Joint-Embedding Predictive Architectures 

**Authors**: Triet M. Le  

**Link**: [PDF](https://arxiv.org/pdf/2606.01443)  

**Abstract**: A central difficulty in training Joint-Embedding Predictive Architectures (JEPAs) is preventing representation collapse. LeJEPA addresses this by enforcing an isotropic Gaussian target on the embeddings via Sketched Isotropic Gaussian Regularization (SIGReg). This target is in tension with the manifold hypothesis, which expects embeddings to concentrate on a low-dimensional subset of the ambient space. We propose \emph{UR-JEPA}, which targets a uniformly $n$-rectifiable measure of local tangent dimension $n$ at small scales, realized through a Gaussian-kernel smoothed Carleson-type square function $\mathcal{L}^{\text{CGLT}}$, with a complementary Jones $\beta$-number formulation. On Inet10, UR-JEPA($\mathcal{L}^{\text{CGLT}}$) attains $0.9141 \pm 0.0014$ for a $+0.83$\,pp gain over LeJEPA($\mathcal{L}^{\text{SIGReg}}$) with $\sim 30\%$ lower seed standard deviation; on matched-recipe Galaxy10~SDSS, a single-seed ImageNet-$100$ run, and a $3$-seed EuroSAT remote-sensing run, the two methods lie in the same peak-accuracy band at convergence, with UR-JEPA retaining its lower-seed-variance signature. On EuroSAT the in-domain pair is competitive at $96.0$ to $96.1\%$ with large remote-sensing foundation-model transfer at a $25\times$ smaller backbone. The distinction is geometric: direct visualization of the projector output distribution shows that on all four datasets UR--JEPA($\mathcal{L}^{\text{CGLT}}$) produces a global PCA spectrum with a $4$ to $5$ order-of-magnitude drop at index $\sim 20$ to $25$ out of $D = 32$, while LeJEPA's spectrum is near-flat (top-to-bottom ratio at most $3.6$). Per-dimension marginals are simultaneously near-Gaussian for both methods (mean Shapiro-Wilk $W \in [0.992, 0.996]$) as a Diaconis-Freedman consequence. At matched accuracy the two regularizers therefore yield structurally distinct projected representations. 

---
# On the Evaluation of Spiking Neural Network Configurations for Network Intrusion Detection 

**Authors**: Raj Patel, David Amebley, Taye Akinrele, Shaswata Mitra, Sayanton Dibbo, Shahram Rahimi  

**Link**: [PDF](https://arxiv.org/pdf/2606.01442)  

**Abstract**: Network intrusion detection is a core component of modern cybersecurity infrastructure, yet the deep learning models that dominate the field are computationally demanding, motivating interest in lightweight alternatives suited to edge and neuromorphic deployment. Spiking Neural Networks (SNNs) are therefore a natural candidate, but their design space, spanning the choice of neuron model and spike encoding scheme, remains poorly characterized for intrusion detection. We bridge this gap by using a controlled ablation study using 9 neurons coupled with 3 spike encoding schemes, making 27 variants, all implemented on snntorch evaluated over raw inputs with limited preprocessing on four benchmark datasets (NSL KDD, KDDCup99, CIC-IDS2017, and CTU-13) with 5 seeds. We find that spike encoding scheme is a better determinant for detection quality than the neuron model, where rate and delta spike encodings perform worse than latency encoding over the sweep. The LeakyParallel neuron with latency encoding performed the best overall, averaging at 92.11% accuracy and 0.80 macro- F1 at a rate of 2.01% false positives averaged over all 4 datasets, with accuracy close to perfect for CIC-IDS2017 and CTU-13, and also performed the fastest on inference. These results highlight the potential of SNNs as a viable alternative to traditional methods of intrusion detection when considering low-latency or resource-constrained deployments. 

---
# CEAR: Certified Ensemble Adversarial Robustness in DNNs 

**Authors**: Daniel Sadig, Mohammadreza Maleki, Hamed Karimi, Reza Samavi  

**Link**: [PDF](https://arxiv.org/pdf/2606.01437)  

**Abstract**: Deep Neural Networks (DNNs) are highly susceptible to adversarial perturbations, leading to extensive research on robustness for safety-critical applications. State-of-the-art empirical defense mechanisms improve the robustness of DNNs through the training phase, but still struggle against adaptive white-box attacks. On the other hand, certified defenses offer provable guarantees of robustness within a specified perturbation bound. These guarantees hold regardless of the level of perturbations, even if the attacker is given full knowledge of the model. In this paper, we propose CEAR, an ensemble-based robust method that utilizes a hybrid of empirical and certified defense mechanisms. CEAR trains each network within the ensemble using varying Gaussian noise and temperatures to obfuscate gradients and logits, making the model more resistant to stronger gradient-based attacks. We then use noisy logits and propose two different voting mechanisms to further improve robustness. Furthermore, we extend randomized smoothing to verify the robustness of ensemble-based classifiers. Our experimental evaluations on MNIST, CIFAR10, and TinyImageNet datasets demonstrate superior certified accuracy on average, increased robustness radius, and decreased transferability compared to baseline methods. 

---
# Neural Network Compression by Approximate Differential Equivalence 

**Authors**: Ravi Dhiman, Andrea Passarella, Mirco Tribastone, Lorenzo Valerio  

**Link**: [PDF](https://arxiv.org/pdf/2606.01402)  

**Abstract**: Neural network compression is commonly achieved by pruning parameters based on local importance scores, e.g., magnitude-based pruning. We propose a complementary approach that compresses models by aggregating neurons with similar functional behavior rather than removing weights independently. Our method encodes a trained network as a polynomial ODE system and applies a lumping method called Approximate Forward Differential Equivalence to identify neurons with approximately matching induced dynamics. A single tolerance parameter, $\varepsilon$, controls the compression level and induces a smooth trade-off between model size and predictive accuracy. We evaluate the method on synthetic datasets derived from nonlinear dynamical systems with known ground-truth behavior and on public regression benchmarks. Across both settings, the proposed approach achieves substantial parameter reduction while preserving accuracy, and consistently compares favorably with magnitude-based pruning and Wanda at similar compression levels. These results suggest that differential equivalence-based aggregation is a principled and effective alternative to conventional weight-centric pruning. 

---
# Consistent and Distinctive: LLM Benchmark Efficiency via Maximum Independent Set Prompt Selection on Similarity Graphs 

**Authors**: Denica Kjorvezir, Marko Djukanović, Ana Gjorgjevikj, Gjorgjina Cenikj, Tome Eftimov  

**Link**: [PDF](https://arxiv.org/pdf/2606.01400)  

**Abstract**: Evaluating large language models (LLMs) across comprehensive benchmarks is expensive and time-consuming. We propose a graph-based prompt selection framework that models each benchmark as a similarity graph -- nodes are prompts connected if their embedding-space distance falls above a configurable threshold -- and applies Maximum Independent Set (MIS) algorithms to select a maximally diverse, non-redundant subset. We evaluate four MIS solvers (CPLEX, GREEDY, Online-MIS, ReduMIS) across six embedding models, three distance measures, six percentile thresholds, and four benchmarks (GPQA, IFEval, MMLU-Pro, Omni-MATH) covering 66 LLMs. Our central hypothesis -- that repeated selection under different random seeds yields consistent LLM rankings that may also differ from the full-benchmark baseline -- is strongly confirmed: Kendall's $W \geq 0.90$ in 99.2\% of stochastic configurations (mean $W = 0.997 \pm 0.008$), while at higher percentile thresholds selected subsets achieve 25--48\% prompt reduction on average. Ranking divergence from the full benchmark ($\rho < 0.95$) occurs in only 15.95\% of configurations, concentrated at low thresholds ($p_{10}$--$p_{20}$) and benchmarks (GPQA, IFEval), identifying overly dense graphs as the primary failure mode. 

---
# Dr. DocBench: A Comprehensive Benchmark for Expert-Level and Difficult Document Parsing 

**Authors**: Minglai Yang, Xinyan Velocity Yu, Pengyuan Li, Xinyu Guo, Zhenting Qi, Konwoo Kim, Longtian Ye, Xiaolong Luo, Jinhe Bi, Henry Zhang, Haris Riaz, Xuan Zhang, Yunze Xiao, Bangya Liu, Tom Tang, Yunfei Zhao, Qunshu Lin, Zihan Wang, Minghao Liu, Michael Lingzhi Li, Yilun Du, Jesse Thomason, Rogerio Feris, Alex Pentland, Zexue He  

**Link**: [PDF](https://arxiv.org/pdf/2606.01393)  

**Abstract**: Document parsing and recognition are fundamental capabilities for vision-language models (VLMs) and document processing systems. However, existing Optical Character Recognition (OCR) and document parsing benchmarks are increasingly limited in coverage and difficulty: many focus on common document genres or uniformly sampled pages where modern parsers already perform strongly, while offering limited annotation for expert-domain structures such as chemical formula, music notation, complex tables, and cross-page layouts. We introduce Dr. DocBench, a difficulty-aware benchmark for expert-level document parsing. Built from a large-scale multilingual book corpus, Dr. DocBench spans 52 BISAC subject domains and selects challenging documents through parser-failure-based sampling, targeting cases where multiple state-of-the-art systems struggle. It contains 4,514 annotated pages from long documents averaging around 100 pages, with 65k high-quality page- and block-level annotations for layout, reading order, hierarchical relations, and domain-specific visual contents. Evaluations of pipeline-based parsers and general-purpose VLMs show that strong performance on existing benchmarks does not transfer to our expert-level document parsing. Our analysis reveals substantial failures across subjects, content types, and structural attributes, highlighting Dr. DocBench as a comprehensive testbed for diagnosing and advancing document intelligence. 

---
# Bridging Requirements and Architecture: Multi-Agent Orchestration with External Knowledge and Hierarchical Memory 

**Authors**: Ruiyin Li, Yiran Zhang, Xiyu Zhou, Yangxiao Cai, Peng Liang, Weisong Sun, Jifeng Xuan, Zhi Jin, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.01385)  

**Abstract**: Software architecture design is a critical yet inherently complex and knowledge-intensive phase that requires balancing competing quality attributes and adapting to evolving requirements. Traditionally, this process has been time-consuming, labor-intensive, and heavily reliant on architects, often resulting in limited exploration of alternative architectural decompositions and styles, especially under the pressures of agile development. While LLM-based agents have shown promising performance across various software engineering tasks, their application to architecture design remains relatively scarce and requires systematic exploration. To address these challenges, we proposed MAAD (Multi-Agent Architecture Design), a knowledge-driven framework that orchestrates four specialized agents (i.e., Analyst, Modeler, Designer and Evaluator) to autonomously and collaboratively transform requirements specifications into comprehensive, multi-view architectural blueprints with quality attribute assessments. MAAD incorporates RAG to inject recognized architectural standards and patterns into the workflow and leverages a hierarchical memory mechanism that captures design history for iterative refinement. We evaluated MAAD through comparative experiments against MetaGPT, using quantitative architecture-level metrics across 10 case studies and qualitative feedback from industry architects on 10 real-world specifications. Results show that MAAD generates more complete, modular, and traceable architectures than the baseline, and its dedicated Evaluator agent autonomously produces structured quality evaluation reports that significantly reduce manual validation efforts. Furthermore, we found that the quality of the generated architecture heavily depends on the underlying LLM's reasoning capacity, with GPT-5.2 and Qwen3.5 outperforming other models across most evaluation settings. 

---
# Efficient Exploration for Iterative Nash Preference Optimization 

**Authors**: Tianlong Nan, Xiaopeng Li, Christian Kroer, Tianyi Lin  

**Link**: [PDF](https://arxiv.org/pdf/2606.01382)  

**Abstract**: Preference alignment is central to improving large language models, but standard reward-based formulations can be restrictive when human preferences are cyclic, non-transitive, or otherwise not representable by a scalar reward. Nash Learning from Human Feedback (NLHF) addresses this limitation by modeling alignment as a preference game and targeting a Nash equilibrium rather than a reward maximizer. However, the learning-theoretic foundations of scalable NLHF remain limited. Existing regret guarantees rely on oracle-based methods that estimate a general preference model and solve KL-regularized minimax problems, while iterative NLHF methods directly optimize policy-level preference losses and are easier to implement but lack regret guarantees. We study online iterative NLHF under general preference models and identify exploration as the key obstacle. First, we show that standard iterative NLHF can suffer an exponential dependence on the KL-regularization parameter, revealing that implicit exploration through policy updates is insufficient for controlling regret. Second, we propose an explicitly exploratory iterative NLHF algorithm that combines SFT-based regularization with adversarial policy exploration. The resulting method retains the direct policy optimization structure of iterative NLHF, avoids explicit preference model estimation, and achieves an $O(\sqrt{T})$ regret bound without an exponential dependence on the KL-regularization parameter. We show that the regret can be improved to $O(\log(T))$ with access to a minimax oracle, clarifying the computational-statistical tradeoff in learning general preference games. Finally, we instantiate our method for LLM fine-tuning and evaluate it on \texttt{Llama-3-8B-Instruct} across multiple benchmarks, where explicit exploration yields consistent improvements over existing NLHF baselines. 

---
# Beyond Access: Guided LLM Scaffolding for Independent Learning in Undergraduate Statistics 

**Authors**: Mohammad Amanlou, Yasaman Amou-Jafari, Mehrad Livian, Fatemeh Boloukazari, Fereshte Bagheri, Behnam Bahrak  

**Link**: [PDF](https://arxiv.org/pdf/2606.01375)  

**Abstract**: Large language models (LLMs) are increasingly entering students' learning practices, but their educational value depends on whether they support reasoning or enable task completion without engagement. This study examines guided LLM use in an undergraduate Probability and Statistics course, focusing on the gap between assigned access and actual interaction quality. In a four-week quasi-experimental summer program, students were organized into three balanced conditions: no LLM access, unrestricted LLM access, and guided LLM access. The guided condition used the same LLM platform as the unrestricted condition, but students received explicit training and rules promoting reasoning-focused help-seeking, stepwise hints, verification, and ethical use. All quizzes and the delayed final exam were completed without LLM or external assistance, allowing us to distinguish AI-supported practice performance from independent learning. Results show that guided use was associated with clearer learning-oriented interaction patterns than unrestricted access, especially in prioritizing reasoning over final answers and requesting stepwise support. Guided-LLM students showed stronger no-help quiz performance during the intervention phase, whereas unrestricted access appeared more useful for assisted practice completion than for consistently improving independent performance. Available time measures did not support a simple duration-based explanation, and self-assessment calibration suggested better alignment between perceived and demonstrated understanding in the Guided-LLM condition. Overall, LLM access alone appears to be an incomplete educational intervention. For Artificial Intelligence in Education (AIED), the central design challenge is to scaffold how students use LLMs so that these systems function as partners in reasoning rather than answer-getting tools. 

---
# BRo-JEPA: Learning Modular Arithmetic in Latent Space 

**Authors**: Divyansh Jha, Yuanfang Xie, Varan Mehra, Brennen Yu  

**Link**: [PDF](https://arxiv.org/pdf/2606.01372)  

**Abstract**: Can neural networks learn abstract algebraic rules, or do they merely memorize training patterns? We investigate this using MNIST digits as states and modular arithmetic operations as actions in a JEPA-style latent world model. Standard supervised baselines and JEPA models with additive operation embeddings fit seen operations but fail to extrapolate reliably to unseen ones. To bridge this gap, we introduce a block-rotation predictor that imposes the circular structure of modulo-10 arithmetic in latent space. This enables strong zero-shot generalization, with the best ResNet-based JEPA block-rotation model achieving 99.46\% zero-shot and 99.46\% rollout accuracy. Our results suggest that latent world models can learn symbolic transformation rules when architecture matches the structure of the problem. Our code can be \href{this https URL}{accessed here}. 

---
# Needles at Scale: LLM-Assisted Target Selection for Windows Vulnerability Research 

**Authors**: Michael J. Bommarito II  

**Link**: [PDF](https://arxiv.org/pdf/2606.01364)  

**Abstract**: The attack surface of a modern operating system is a haystack: thousands of signed binaries and millions of functions, almost none relevant to any given vulnerability. A human analyst or an LLM agent must pick the function worth reading before analyzing it. At whole-OS scope, this target selection, not the analysis, is the binding constraint. We present Symbolicate-Enrich-Sample, a low-cost batch pipeline that turns a corpus of production Windows binaries into a queryable, priority-ranked research queue. We (i) recover function-level symbols for stripped vendor binaries by auto-fetching the public symbol files and joining them to a recovered call graph; (ii) attach cheap, deterministic structural features to each named function and, conditioned on those features, use a low-cost language model to assign a reachability tier, a risk level, a bug-class hypothesis, and a rationale; and (iii) draw diverse, prioritized batches via a priority-weighted importance sampler. The contribution is a selection substrate: the prioritization layer a downstream detector or LLM agent runs on top of. Across a whole Windows image of 7,231,419 functions, the labels are markedly selective, and stacking deterministic filters on them leaves a ~22K-function shortlist: the candidate needles, few enough for a human or agent to work through. We characterize the pipeline's selectivity and its failure modes, describe the methodology, and report aggregate statistics; we withhold the derived dataset for legal and dual-use reasons. 

---
# FreqLite: A Lightweight Frequency-Decomposed Linear Model with Adaptive Reversible Normalization for Robust Long-Term Time-Series Forecasting 

**Authors**: Mirza Samad Ahmed Baiga, Syeda Anshrah Gillani  

**Link**: [PDF](https://arxiv.org/pdf/2606.01339)  

**Abstract**: Long-term time-series forecasting needs models that are accurate yet efficient enough for commodity hardware. Lightweight linear forecasters are remarkably strong in this regime, yet they leave two openings: reversible instance normalization (RevIN) de-normalizes the entire horizon with a single lookback statistic, which is inaccurate under non-stationarity, and time-domain trend/seasonal decomposition relies on a fixed, non-adaptive filter. We present FreqLite, an ultra-lightweight, channel-independent frequency-decomposed linear forecaster: a learnable, lossless, partition-of-unity spectral filter splits the input into bands that are forecast by per-band linear heads and, unlike low-pass-truncation approaches, the high-frequency band is retained and modeled. FreqLite is the best lightweight model on the standard long-term forecasting benchmarks and, at long lookback (L=336), attains a lower average error than a PatchTST Transformer (0.3244 vs. 0.3587 MSE) while using 4x fewer parameters, 2.2x less memory, and 2.2x less time per epoch on a single 4 GB laptop GPU; although modest in magnitude, its improvements are statistically significant under paired Wilcoxon tests across all matched cells (p < 1e-5). We further introduce Adaptive Reversible Instance Normalization (A-RevIN), a regime-adaptive reversible normalization that strictly generalizes RevIN (recovered exactly when its gate is closed), engages under non-stationarity, and reduces to RevIN without harm on stationary data. We validate this on both a real strongly non-stationary dataset (ILI, up to ~5% MSE reduction) and a controlled synthetic drift sweep in which A-RevIN's benefit and its learned gate both rise monotonically with injected non-stationarity. Every component is independently ablatable (Linear and RLinear are special cases of FreqLite), and all results are reproducible on commodity hardware. 

---
# Digital Twin-Assisted Adaptive Multi-Agent DRL for Intelligent Spectrum and Resource Management in Open-RAN UAV-Enabled 6G Networks 

**Authors**: Marwan Dhuheir, Thang X. Vu, Symeon Chatzinotas  

**Link**: [PDF](https://arxiv.org/pdf/2606.01324)  

**Abstract**: The evolution toward 6G wireless networks envisions a seamlessly intelligent, Open-RAN-enabled architecture where unmanned aerial vehicles (UAVs) play a pivotal role in extending coverage, enhancing resilience, and ensuring reliable connectivity for ground users deployment. However, efficiently managing spectrum and resources in such highly dynamic UAV-assisted environments remains a major challenge due to nonlinear system interactions, mobility-induced topology variations, and stringent latency and energy constraints. To address these challenges, we propose a digital twin (DT)-assisted adaptive deep reinforcement learning (DRL) framework that enables intelligent spectrum sharing and resource allocation across distributed ground users. The complex optimization problem is decomposed into UAV trajectory optimization using particle swarm optimization (PSO) and dynamic spectrum-power-association management via multi-agent DRL (MADRL). This hybrid DT-driven approach empowers intelligent, context-aware decision-making and adaptive coordination among UAVs. Extensive simulations demonstrate significant gains in spectral efficiency, data rates, and energy utilization, showcasing a transformative path toward self-evolving, autonomous 6G UAV and ground users (GUs) connectivity. 

---
# DiffuSent: Towards a Unified Diffusion Framework for Aspect-Based Sentiment Analysis 

**Authors**: Shu Long, Yanglei Gan, Xuchuan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.01323)  

**Abstract**: Aspect-Based Sentiment Analysis (ABSA) encompasses seven distinct subtasks, each focusing on different extracted elements. Despite the proven success of generative models in unified aspect sentiment analysis, existing approaches often rely on auto-regressive token-by-token generation without grasping the whole information of the aspect and opinion terms, resulting in boundary insensitivity, particularly in context of multi-word aspect and opinion terms. To address these issues, we present DiffuSent, a non-auto-regressive diffusion framework that systematically formulates all ABSA subtasks as boundary denoising diffusion processes, progressively refining boundaries over noisy states. Furthermore, we introduce a contrastive denoising training strategy which effectively address duplicate predictions with subtle variations introduced by diffusion process. Extensive experiments across 28 settings (7 subtasks x 4 datasets) demonstrate that DiffuSent achieves delivers consistent improvements over the strongest generative and span-based systems. DiffuSent exhibits notable gains on multi-word triplets, achieving an average improvement of +2.48 F1, and maintains robust extraction accuracy in sentences containing multiple sentiment triplets. Moreover, the non-auto-regressive decoding enables substantial efficiency benefits, reaching up to 181 times faster inference than auto-regressive generative baselines 

---
# TukaBench: A Culturally Grounded Jailbreak Benchmark for African Languages 

**Authors**: Victor Akinode, Senyu Li, Wassim Hamidouche, Waqas Zamir, Inbal Becker-Reshef, David Ifeoluwa Adelani  

**Link**: [PDF](https://arxiv.org/pdf/2606.01322)  

**Abstract**: Safety evaluation of Large Language Models (LLMs) remains heavily English-centric, leaving Low-Resource Languages (LRLs), particularly African ones, critically underexplored. We introduce TUKABENCH, a jailbreak benchmark for seven African languages that extends JailbreakBench (JBB) beyond direct translation through four settings: human translation of JBB prompts, English adaptation to African contexts followed by human translation, human-curated prompts validated through interactions with GPT-5.2, and code-switched prompts combining English and African languages, isolating the effect of language, cultural grounding, and prompt evasiveness on model safety. Across closed and open models, prompting in African languages reduces refusal relative to English, with culturally adapted prompts leading to least refusal. The evaluation also surfaces two structural limitations: model comprehension failures and reduced LLM-as-a-judge reliability in LRLs. To capture the first, we introduce Deflection alongside Refused and Jailbroken; to assess the second, we validate outputs with human annotations, showing that judge-human agreement drops in lower-resource languages and less commonly supported scripts. 

---
# PSG-Nav: Probabilistic Scene Graph Navigation via Multiverse Decision Making 

**Authors**: Rufeng Chen, Yue Chang, Xiaqiang Tang, Hechang Chen, Sihong Xie  

**Link**: [PDF](https://arxiv.org/pdf/2606.01313)  

**Abstract**: Open-vocabulary navigation requires embodied agents to manage significant perception uncertainty stemming from semantic ambiguity and model errors.
However, most existing works settle for local optimal deterministic approaches, depriving complex navigation decision-making over multiple composite possibilities that are critical for globally better solutions.
In this paper, we propose Probabilistic Scene Graph Navigation (PSG-Nav), which constructs a 3D Probabilistic Scene Graph that uses full semantic categorical distributions to account for perception uncertainty.
To efficiently use the local distributions to compose and reason about the optimal navigation landmarks, we propose Multiverse Decision to sample multiple most likely world settings from the joint distribution, and evaluate navigation landmarks based on the compatibility between landmarks and multiverses.
To mitigate false positives due to epistemic uncertainty in open-vocabulary navigation, we introduce the Evidential Experience Calibrator, which enables online lifelong adaptation by cross-validating detections against memories of past successes and failures.
Extensive experiments on widely-used benchmarks MP3D, HM3D, and HSSD demonstrate that PSG-Nav establishes new state-of-the-art results, achieving Success Rates of 66.1%, 44.8%, and 67.9%, respectively. Code is available at: this https URL 

---
# A Communication-Centric 6G-LLM Architecture for Scalable Tactical Autonomous Defense Vehicle Networks 

**Authors**: Kiran Khurshid, Shumaila Javaid, Nasir Saeed  

**Link**: [PDF](https://arxiv.org/pdf/2606.01312)  

**Abstract**: The integration of Artificial Intelligence (AI) and emerging 6G networks introduces new opportunities for scalable coordination in tactical autonomous vehicle systems. This paper proposes a communication-centric hierarchical architecture for Tactical Autonomous Defense Vehicle Networks (TADVNs) that models the integration of edge-assisted Large Language Model (LLM) reasoning with 6G-enabled connectivity and semantic communication. The framework is designed to improve coordination efficiency, reduce communication overhead, and enhance latency resilience under increasing fleet-scale operation. Unlike conventional task-specific AI pipelines that rely on structured feature processing and rule-based coordination, the proposed approach incorporates semantic abstraction and context-aware decision support within a layered edge-cloud communication architecture. We evaluate communication and coordination performance via Monte Carlo simulations across fleet sizes of 5-30 vehicles under contested network conditions. Results indicate that at a 30-vehicle scale, the 6G-LLM configuration achieves 75.2% latency reduction (29.1 ms vs. 117.5 ms), a 68.7 percentage point increase in mission success rate (82.9% vs. 14.2%), and an 88.6% reduction in communication overhead compared to a 5G-based conventional AI baseline. These findings demonstrate measurable benefits in coordination and communication when semantic reasoning is combined with low-latency 6G connectivity. 

---
# SkillAdaptor: Self-Adapting Skills for LLM Agents from Trajectories 

**Authors**: Zhuoyun Yu, Xin Xie, Wuguannan Yao, Chenxi Wang, Lei Liang, Xiang Qi, Shumin Deng  

**Link**: [PDF](https://arxiv.org/pdf/2606.01311)  

**Abstract**: Large language model (LLM) agents increasingly rely on reusable external skills to solve long-horizon interactive tasks. Existing training-free skill adaptation pipelines usually update skills from full trajectories or session-level feedback, which makes failure attribution coarse and often produces unstable or overly broad revisions. We propose SkillAdaptor, a training-free step-level skill adaptation framework with explicit failure attribution, and it can plug into OpenClaw-class agent harnesses. Given a failed trajectory, SkillAdaptor identifies a first actionable fault step, links responsibility to candidate skills, and applies targeted updates under explicit acceptance checks while keeping the backbone frozen. We evaluate on WebShop, PinchBench, and Claw-Eval with Kimi-K2.5, GLM-5, and GPT-5.2. SkillAdaptor improves over no-skill and skill-adaptation baselines on all three suites, with the largest single-metric improvements of +1.5 points on PinchBench Avg Score%, +1.8 on Claw-Eval Avg Score, and +1.7 on WebShop success rate. These results indicate that step-level attribution supports more stable and auditable training-free skill maintenance\footnote{The code will be released at this https URL.}. 

---
# ChronosAD: Leveraging Time Series Foundation Models for Accurate Anomaly Detection 

**Authors**: Uzair Khan, Luigi Capogrosso, Francesco Biondani, Michele Magno, Franco Fummi, Francesco Setti, Marco Cristani  

**Link**: [PDF](https://arxiv.org/pdf/2606.01300)  

**Abstract**: Time series anomaly detection is a crucial task in various domains, including finance, healthcare, and industry. However, existing methods often struggle to generalize across different datasets, especially when anomalies are subtle or context-dependent. To solve this issue, we introduce ChronosAD, a novel architecture for anomaly detection that uses a time series foundation model as a feature extractor. Specifically, it employs a two-stage pipeline: first, it uses the foundation model to extract embeddings for each time series in a zero-shot manner. Then, a custom-developed Temporal Block, composed of Bidirectional Long Short-Term Memory (BiLSTM) and Multi-Head Attention, refines these embeddings to capture temporal dependencies and highlight salient patterns. Unlike previous approaches, our model requires minimal task-specific tuning and demonstrates robust generalization across a wide range of domains, including industrial, medical, cyber-physical, and automotive systems. Extensive experiments on 11 benchmarks show that ChronosAD outperforms existing methods by 4.72% in AUC and 6.60% in AP on average. The source code is available at this https URL. 

---
# ResNet-34 with Lightweight Decoder for Accurate and Efficient Segmentation of Fetal Brain MRI 

**Authors**: Ashiqur Rahman, Muhammad E. H. Chowdhury, Md. Abu Sayed, Md. Sharjis Ibne Wadud, Abu Naser Md. Arafat, Mehedi Hasan Prince  

**Link**: [PDF](https://arxiv.org/pdf/2606.01293)  

**Abstract**: Accurate segmentation of fetal brain tissues in Magnetic Resonance Imaging (MRI) is critical for early diagnosis of congenital abnormalities and improving prenatal care. However, the task remains difficult because of fetal motion, low tissue contrast, and major anatomical variability throughout gestational ages, particularly in segmenting complex structures such as white matter, gray matter, lateral ventricles, deep gray matter, extra-cerebrospinal fluid, cerebellum, and brainstem. As a solution to these difficulties, this research introduces a novel deep learning model that combines a ResNet-34 encoder with a lightweight decoder leveraging multi-layer perceptron (MLP) modules for adaptive feature refinement. This design specifically enhances the model's ability to preserve anatomical boundaries and mitigate segmentation errors caused by motion artifacts and intensity inhomogeneities. Computational efficiency is achieved by reducing parameter count, employing bilinear upsampling instead of transposed convolutions, and optimizing the decoder for speed without sacrificing accuracy. Trained and validated on the FeTA 2021 dataset using 5-fold cross-validation, the proposed model outperforms baseline architectures such as UNet, UNet++, DeepLabV3, and DeepLabV3+, achieving an average Accuracy of 97.37% with a mean Dice Similarity Coefficient (DSC) of 90.33%, mean Intersection over Union (IoU) of 86.93%, and Precision of 90.83%. Additionally, its fast inference time and reduced computational load make it well-suited for integration into real-time clinical workflows. 

---
# What Makes a Strong Model? A Unified Spectral Analysis of Knowledge Transfer over High-dimensional Linear Regression 

**Authors**: Wendao Wu, Fangqing Zhang, Haihan Zhang, Cong Fang  

**Link**: [PDF](https://arxiv.org/pdf/2606.01292)  

**Abstract**: Teacher-Student Knowledge Transfer (KT) is ubiquitous in modern machine learning, ranging from classical model compression via Knowledge Distillation (KD) to the emergent phenomenon of Weak-to-Strong (W2S) generalization. While existing studies offer isolated insights, a unified theoretical framework explaining the efficacy of KT across these disparate regimes remains lacking. In this work, we establish a unified spectral analysis of SGD dynamics in high-dimensional linear regression, elucidating the efficiency of KT across seemingly disparate regimes. We characterize KT efficiency through two distinct mechanisms: \emph{Spectral Horizon Expansion} in KD, which enables the capture of statistically inaccessible high-frequency signals, and \emph{Spectral Denoising} in W2S, where the student acts as a filter for optimization noise. Our framework unifies these phenomena, revealing that the efficacy of transfer is governed by the interplay between implicit regularization and heterogeneous spectral learning speeds over the spectrum. 

---
# Quantum Algorithm for Distributed Reduction of Entanglements (QADR): A Trainable and Simulation-Efficient QML Framework 

**Authors**: Syed Farhan Ahmad, Gregory T. Byrd  

**Link**: [PDF](https://arxiv.org/pdf/2606.01291)  

**Abstract**: Training Variational Quantum Circuits (VQCs) under Noisy Intermediate-Scale Quantum (NISQ) constraints introduces severe computational limitations: classical statevector simulation memory scales exponentially ($\mathcal{O}(2^n)$), and global cost functions suffer from barren plateaus where gradient variance decays exponentially ($\mathcal{O}(1/2^n)$). This paper introduces and evaluates the Quantum Algorithm for Distributed Reduction of Entanglements (QADR), a hybrid quantum-classical machine learning framework that decomposes a global $n$-qubit VQC into localized sub-circuits operating approximately within the causal light cones of individual target qubits. QADR reduces classical simulation memory scaling from $\mathcal{O}(2^n)$ to $\mathcal{O}(n \cdot 2^{2d+1})$ for a light cone radius $d$, while naturally mitigating global barren plateaus. We benchmark QADR against standard global VQCs, Support Vector Machines (SVM), and two customized classical parameter-matched neural networks (CANN and PMNN) on the MNIST dataset and the high-dimensional NASA IMS wind turbine drivetrain diagnostic task. QADR demonstrates excellent scalability, operating successfully at $n_{\text{features}}=2000$ where standard global VQCs crash due to memory exhaustion, while matching or exceeding the performance of optimized classical architectures. 

---
# Beyond Visual Memory: Mechanistic Diagnostics of Latent Visual Reasoning 

**Authors**: Garvin Guo, Yu Chen, Xiang Wang, Shuai Li, Xinpei Zhao, Huaxing Liu, Shuai Dong  

**Link**: [PDF](https://arxiv.org/pdf/2606.01287)  

**Abstract**: Recent latent visual reasoning methods achieve substantial gains by inserting continuous latent tokens into multimodal language models. These gains are commonly attributed to the tokens encoding visual evidence; recent analyses, however, reveal a paradox: the tokens are loosely tied to the image and contribute little to the answer. Critically, these analyses treat latent tokens as a single unit, obscuring the true source of the gains. We therefore decompose latent tokens into three testable components: latent slots, boundary markers, and format, and develop a state-of-the-art method as a probe under favorable conditions. Across six method-stage settings and four perception-heavy benchmarks, latent slots fail every prediction of the visual-memory account. Strikingly, retaining only the boundary markers preserves 78 to 100% of the gain in several settings, while the model attends to the image more narrowly at latent positions than at answer positions. The gain therefore comes from boundary markers, format, and this attention pattern, not from latent slots. How each method engages this mechanism depends on its training supervision: at matched accuracy, mechanisms can still differ markedly. Latent visual reasoning thus needs evaluation not only by accuracy but by what the model actually relies on. 

---
# BenchEvolver: Frontier Task Synthesis via Solution-Centric Evolution 

**Authors**: Yangzhen Wu, Aaron J. Li, Wenjie Ma, Li Cao, Ziheng Zhou, Mert Cemri, Shu Liu, Yuran Xiu, Chenxiao Yan, Haikun Zhao, Bin Yu, Ion Stoica, Dawn Song  

**Link**: [PDF](https://arxiv.org/pdf/2606.01286)  

**Abstract**: The rapid progress of frontier large language models has led to widespread benchmark saturation, limiting the ability of existing datasets to differentiate model capabilities or provide useful training signal. For instance, on LiveCodeBench, frontier models achieve over 99% Pass@1 on easy splits and exceed 90% Pass@1 on average across difficulty levels. Constructing new, challenging datasets typically requires substantial human effort, creating a bottleneck for progress. We introduce BenchEvolver, a solution-centric evolutionary framework that automatically transforms existing coding problems into harder variants. Rather than generating problems from scratch, BenchEvolver evolves reference solutions through structured transformations and derives corresponding statements and tests from the evolved solutions. This design grounds generation in executable semantics, enabling scalable construction of high-quality, diverse, and difficult tasks with verifiable correctness. Applying BenchEvolver to LiveCodeBench and SciCode, we obtain evolved tasks that are substantially harder while maintaining validity, reference correctness, and diversity. We further curate LiveCodeBench-Plus, a 91-problem benchmark combining evolved and difficult original LCB-v6 tasks, where frontier-model Pass@1 ranges from 27.5% to 62.6%, restoring clear discrimination among strong coding models. Importantly, evolved tasks remain challenging even for the model that generates them, enabling self-improvement. We further show that RL on evolved LCB tasks improves held-out coding performance: for gpt-oss-20b, seed+evolved training achieves +8.7 and +8.3 Pass@1 gains on LCB v6 Hard and LCB-Pro Easy, exceeding seed-only gains by 70.7% and 34.8%, respectively. Our results show that BenchEvolver can convert saturated benchmarks into frontier-level evaluation suites and reusable training signal. 

---
# Knowledge-Intensive Video Generation 

**Authors**: Chenxu Wang, Mingda Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.01285)  

**Abstract**: Text-to-video generation has advanced rapidly in visual quality, but remains under-evaluated for factuality and practical usefulness. We introduce knowledge-intensive video generation (KIVI), where models generate videos from short information-seeking prompts that ask for explanations, procedures, or demonstrations. To evaluate this setting, we construct KIVI-Bench, a benchmark of 1,080 prompts, and propose automatic metrics for factuality and helpfulness. Human evaluation shows that our metrics significantly better align with human annotations than existing alternatives. Experiments on seven state-of-the-art video generation models show that current systems still lag behind human performance, especially on visual properties, procedural operations, and clear information presentation. These results highlight KIVI as a challenging direction for factual and instructionally useful video generation. 

---
# RLVR without Ineffective Samples: Group Prioritized Off-Policy Optimization for LLM Reasoning 

**Authors**: Yixiu Mao, Yun Qu, Qi Wang, Heming Zou, Xiangyang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2606.01281)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has emerged as a powerful paradigm for enhancing the reasoning capabilities of large language models (LLMs). However, its effectiveness is substantially hindered by the prevalence of ineffective training data: many sampled prompts yield response groups that are either entirely correct or entirely incorrect, resulting in zero-variance rewards and limited learning signals. Recent state-of-the-art methods address this issue through extensive LLM rollouts to filter ineffective samples, but at the cost of considerable computational overhead. Alternative approaches, including predictive sampling and trajectory replay, aim to improve data efficiency but often remain insufficient and may introduce additional issues such as systematic bias or suboptimal constraints. To address these limitations, we propose Group Prioritized Off-Policy Optimization (POPO), a simple yet effective framework that fully exploits effective training batches without additional rollout overhead. POPO comprises two key components: prioritized group replay and decoupled off-policy optimization. The former replaces ineffective on-policy groups with effective off-policy groups via a recency-based replay mechanism that jointly considers sample quality and the degree of off-policiness. To further mitigate the off-policy gap, POPO employs decoupled importance sampling to correct off-policy bias while maintaining stable policy updates under consistent trust-region constraints. Empirical evaluations across diverse reasoning tasks, including mathematics, planning, and visual geometry, demonstrate that POPO substantially accelerates RL finetuning and achieves strong reasoning performance with significantly fewer rollouts. 

---
# DeepIPCv3: Event-Aware Multi-Modal Sensor Fusion for Sudden Pedestrian Crossing Avoidance 

**Authors**: Oskar Natan, Andi Dharmawan, Aufaclav Zatu Kusuma Frisky, Jazi Eko Istiyanto, Jun Miura  

**Link**: [PDF](https://arxiv.org/pdf/2606.01277)  

**Abstract**: Current end-to-end autonomous driving systems predominantly rely on frame-based sensors, which suffer from inherent perception latency and motion blur during highly dynamic encounters, specifically sudden pedestrian crossings. To address this critical safety vulnerability, we propose DeepIPCv3, a novel multi-modal autonomous navigation framework that synergizes the dense 3D spatial geometry of LiDAR point clouds with the microsecond-level asynchronous event streams of a Dynamic Vision Sensor (DVS). We introduce a Transformer-inspired cross-modal attention mechanism to dynamically correlate these distinct modalities, allowing the network to instantaneously prioritize high-speed dynamic updates without sacrificing structural scene awareness. The fused latent representations are then mapped to safe local waypoints and executable control commands via a hybrid policy network that blends heuristic trajectory tracking with direct neural predictions. Due to the severe physical risks associated with live testing of these sudden crossing scenarios, the framework is rigorously evaluated offline using a custom multi-modal dataset collected across both well-illuminated noon and challenging evening conditions. Extensive comparative and ablation studies demonstrate that DeepIPCv3 achieves state-of-the-art predictive performance. By effectively eliminating exposure failures and motion blur, the proposed LiDAR and DVS fusion yields the lowest trajectory and control command errors, enabling highly reactive, mathematically bounded evasive maneuvers regardless of ambient illumination. To support future research, we will release the codes to our GitHub repo at this https URL. 

---
# PALTO: Physics-Informed Active Learning for Tri-Gate FinFET Design Optimization for Vertical Power Delivery 

**Authors**: Ayoub Sadeghi, Leonid Popryho, Inna Partin-Vaisband  

**Link**: [PDF](https://arxiv.org/pdf/2606.01265)  

**Abstract**: This paper demonstrates the effectiveness of machine learning-driven optimization for designing application-specific GaN tri-gate FinFETs in vertical power delivery systems. Conventional TCAD-based approaches are computationally intensive and insufficient for navigating the high-dimensional, nonlinear design space of advanced GaN devices. To address this, a physics-informed active learning framework is used to intelligently guide simulations, accelerating convergence while preserving accuracy. This ML-guided approach enables the discovery of optimal configurations by efficiently exploring key structural parameters -- most notably the GaN-to-AlGaN thickness ratio -- a long-standing focus of debate in device design. By systematically exploring key structural parameters, two optimized devices with aggressively scaled gate-to-drain lengths are identified. Single-fin, multi-channel simulations show that device~D2, with a thinner GaN channel relative to the AlGaN barrier, achieves higher drive current. However, in a 300-fin configuration, device~D1 outperforms device~D2 by delivering 3.3\,A at 0.49~ohm on-resistance -- approximately 2$\times$ better -- despite slightly higher parasitics. Both devices operate in a normally-off mode. Based on an application-specific figure of merit, device~D1 achieves 5\,pC$\cdot$ohm, demonstrating 2$\times$ greater switching efficiency than device~D2, while both designs outperform industrial benchmarks from different performance standpoints. 

---
# IndoBias: A Dual Track Culturally Grounded Benchmark for LLMs Bias Evaluation in Indonesian Languages 

**Authors**: Ikhlasul Akmal Hanif, Muhammad Falensi Azmi, Filbert Aurelian Tjiaranata, Eryawan Presma Yulianrifat, Fajri Koto  

**Link**: [PDF](https://arxiv.org/pdf/2606.01260)  

**Abstract**: Despite being home to more than 1300 ethnic groups and 700 indigenous languages, bias in Large Language Models has not been fully studied in Indonesia, thus leaving a critical gap in evaluating representational fairness and localized stereotypes within its uniquely vast, multilingual, and diverse sociocultural landscape. To address this, we introduce IndoBias as a culturally-grounded bias benchmark to assess LLMs bias in Indonesian and three local languages: Javanese, Sundanese, and Makasar. IndoBias features dual perspective evaluation tracks: depth-oriented (with contrastive-pairs) and breadth-oriented (with generation-based), where the latter is grounded in social science frameworks (SPI, O*NET, and WGI). Our results show that existing LLMs -- particularly decoder models -- exhibit strong bias towards prototypical sentences in Indonesian, while local languages suffer higher bias under Ideology and Religion category. We also find that LLMs responses exhibit a non-uniform Stereotype Polarity when prompted with various local entities. Finally, we discover that, in Indonesian, Common Crawl texts introduce more bias during pretraining, compared to human-reviewed article texts (e.g., Wikipedia, News), whereas introducing local languages to pretraining generally increases bias. This work highlights the importance of studying bias in culture-specific context. Warning: This paper contains example data that may be offensive, harmful, or biased. 

---
# Understanding LLM Behavior in Multi-Target Cross-Lingual Summarization 

**Authors**: Sangwon Ryu, Yihong Liu, Mingyang Wang, Yunsu Kim, Jungseul Ok, Gary Geunbae Lee, Hinrich Schuetze  

**Link**: [PDF](https://arxiv.org/pdf/2606.01252)  

**Abstract**: Multi-target cross-lingual text summarization (MTXLS), which summarizes a source document into multiple target languages, is increasingly important as users consume content in diverse languages, but remains underexplored. To address this gap, we introduce multi-target cross-lingual element-aware (MEA), a new MTXLS benchmark covering 24 target languages. We benchmark end-to-end and pipeline approaches across various LLMs and show that MTXLS performance still substantially lags behind English monolingual summarization. To better understand MTXLS in LLMs, we propose a layer-wise analysis framework for investigating how LLMs internally perform MTXLS. Our analyses suggest that translation and summarization behaviors emerge jointly within later layers rather than as distinctly decomposed stages. Most task-relevant processing occurs within these layers, and errors also tend to arise at similar depths. Motivated by these findings, we introduce an inference-time activation steering method that leverages hidden representations from English summarization to guide MTXLS generation. Experiments show that our method consistently improves MTXLS quality across target languages. 

---
# Connecting the Dots: Benchmarking Reflective Memory in Long-Horizon Dialogue 

**Authors**: Jingjie Lin, Bingbing Wang, Zihan Wang, Zhengda Jin, Weiming Qiao, Jing Li, Ruifeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.01223)  

**Abstract**: Despite substantial progress in long-context modeling, existing benchmarks remain confined to factual memory for explicit recall, failing to measure the reflective memory required to synthesize fragmented, multimodal cues into high-level interpretations. To address this gap, we introduce RefMem-Bench, a benchmark for reflective memory in long-horizon dialogue. RefMem-Bench contains 26K annotated QA instances with eight reflective-memory dimensions and three task formats, requiring models to move beyond surface-level retrieval and infer latent meanings from evidence distributed across interaction histories. To enhance reflective memory capability, we propose REflective Memory INDuction (REMIND), a hierarchical framework that treats reflective memory as progressive meaning construction. REMIND couples question-conditioned evidence retrieval, salience-aware grounding, and abstraction-level supervision, and uses Progressive Reflective Alignment to distill high-level reflective reasoning into the factual inference pathway. Experiments show RefMem-Bench poses a substantial challenge to current models, while REMIND consistently improves both answer accuracy and memory recall through progressive evidence perception, grounding, and abstraction. 

---
# Hybrid Imbalanced Regression Through Unified Data-Level and Algorithm-Level Balancing 

**Authors**: Shermin Shahbazi, Hossein Mohammadi, Mohsen Afsharchi  

**Link**: [PDF](https://arxiv.org/pdf/2606.01221)  

**Abstract**: Imbalanced learning is a critical challenge in machine learning, where underrepresented target values can bias models and degrade prediction performance on rare but important cases. Although extensively studied in classification, imbalanced regression remains relatively underexplored. Existing methods mainly focus on either data-level balancing, which may introduce noise and overfitting, or algorithm-level balancing, which often struggles with highly complex target distributions. To address these limitations, we propose a unified hybrid framework that integrates both data- and algorithm-level balancing strategies into a regressor-agnostic pipeline. The proposed framework consists of five stages: (1) adaptive bin partitioning to dynamically segment the target space based on local linear coherence; (2) target-conditioned representation learning using a Conditional Variational Autoencoder; (3) multistage data-level balancing through feature-space clustering and oversampling of minority clusters; (4) algorithm-level balancing using a novel Latent-Density Weighted Loss (LDWL) to emphasize rare samples in latent and target spaces; and (5) attention-based gated fusion for final regression. Experimental results on benchmark datasets demonstrate that the proposed framework consistently improves predictive performance compared to standalone regressors and existing imbalanced regression approaches. 

---
# Fine-Tuning Diffusion Models for Molecular Generation via Reinforcement Learning and Fast Sampling 

**Authors**: Guang Lin, Shikui Tu, Lei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.01220)  

**Abstract**: Generating molecules that simultaneously satisfy drug-like properties and conform to the 3D structure of a target protein is a core challenge in structure-based drug design (SBDD). Existing generative approaches, however, often rely on costly post-hoc processing during Sampling or require carefully curated datasets during training, yet still achieve modest gains. These limitations are especially pronounced in multi-objective settings, where balancing conflicting criteria remains a core challenge. To address these challenges, We propose FTDiff, a reinforcement learning fine-tuning framework tailored for diffusion-based molecular generation under structural constraints. To ensure stable and sample-efficient optimization, FTDiff adopts a group relative policy optimization (GRPO) style strategy. Furthermore, FTDiff builds upon a time-free pretrained diffusion model and incorporates a fast sampling mechanism that reduces the number of denoising steps, significantly accelerating both training and inference while maintaining generation quality. By optimizing a fixed threshold-aware reward, FTDiff effectively guides the model to produce valid, diverse, and high- quality molecules that balance multiple drug design objectives. Extensive experiments on benchmark datasets demonstrate that FTDiff consistently outperforms prior methods, without requiring expensive post-hoc optimization or intricate data engineering. 

---
# Distilling Neuro-Symbolic Programs into 3D Multi-modal LLMs 

**Authors**: Wentao Mo, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.01215)  

**Abstract**: Current 3D spatial reasoning methods face a fundamental trade-off: neuro-symbolic 3D (NS3D) concept learners achieve interpretable reasoning through compositional programs but are constrained to closed-set concept vocabularies and simple programs; end-to-end 3D multi-modal LLMs (3D MLLMs) could handle complex natural language and open-vocabulary concepts but suffer from black-box reasoning without explicit spatial verification. We introduce APEIRIA, a neuro-symbolic 3D MLLM to bridge two paradigms by distilling symbolic reasoning patterns into MLLMs with natural language chain-of-thought. Our three-stage curriculum progressively builds reasoning capabilities: a) 3D perception alignment grounds object visual-geometric features to the LLM, b) CoT-SFT teaches query decomposition and stepwise verification from symbolic program traces, and c) CoT-RL extends reasoning patterns to open-set concepts and deeply nested instructions. By transferring reasoning patterns rather than concept-specific knowledge, APEIRIA preserves key NS3D virtues: transparent reasoning and modular interchangeability of planning and perception components. Evaluations on grounding, question answering, and captioning show that APEIRIA surpasses prior NS3D methods and matches state-of-the-art 3D MLLMs on 3D spatial reasoning datasets, unifying symbolic methods' systematic reasoning with MLLMs' flexibility. Code is available at this https URL. 

---
# TECCI: Tricky Edits of Collected and Curated Images 

**Authors**: Aishwarya Agrawal, Roy Hirsch, Yasumasa Onoe, Sherry Ben, Jason Baldridge  

**Link**: [PDF](https://arxiv.org/pdf/2606.01213)  

**Abstract**: Despite tremendous recent progress, current text-guided image editing methods still struggle with many aspects of editing involving instruction following, minimally editing the source image, and ensuring high visual quality. These problems are especially apparent when the requested edit is challenging, such as those that involve position, motion, viewpoint, scale and creative edits. To systematically test generative image editors, we propose a novel image editing benchmark -- TECCI: Tricky Edits of Collected and Curated Images. TECCI consists of a completely new set of images we are releasing. The images in TECCI span 7 image categories. The images and these categories were curated intentionally to target weaknesses of existing methods. The edit instructions in TECCI are automatically generated by Gemini, covering 5 edit types per source image. We also curated a set of 530 images for which we created challenging manually written edit instructions. Overall, TECCI contains 7550 pairs of images and edit instructions. We conduct human evaluations of five leading image editing models on TECCI. Humans judge outputs along three dimensions: 1) instruction following, 2) minimality of the edits, and 3) visual quality. To scale-up the evaluation, we also build an auto-rater using Gemini that achieves 74.7% accuracy in matching human evaluations. Our evaluations reveal that: 1) none of the models exceed a 22% overall success rate, demonstrating the challenging nature of TECCI, 2) Nano Banana Pro is the best performing model overall, 3) models perform significantly better at instruction following compared to minimal edits and visual quality, 4) models struggle with editing architecture and nature images which require strong understanding of spatial layout and intricate visual details. 5) reasoning and creative edits are the most difficult, whereas color and appearance edits are the easiest. 

---
# DiscourseFlip: An Oblique Discourse-Level Opinion Manipulation Attack against Black-box Retrieval-Augmented Generation 

**Authors**: Yuyang Gong, Miaokun Chen, Jiawei Liu, Zhuo Chen, Guoxiu He, Wei Lu, XiaoFeng Wang, Xiaozhong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.01212)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems are widely deployed and increasingly influential, but their reliance on external corpora exposes new security risks from poisoned retrieval content. Existing RAG attacks are largely focusing on individual queries or narrow topic-local query sets, which limits their practical reach and offers limited camouflage in real-world settings. In this paper, we introduce discourse-level opinion manipulation, a new threat model in which coordinated influence across a semantic query network induces opinion shifts over a holistic, multi-topic query space. We formalize this threat in a black-box setting and propose DiscourseFlip, an agentic, graph-guided attack that dynamically allocates a limited poisoning budget to maximize discourse-level opinion deviation. Extensive experiments demonstrate that DiscourseFlip consistently induces targeted opinion shifts across the contextualized query network and significantly outperforms existing baselines in terms of coverage and effectiveness. User studies further confirm that DiscourseFlip is effective while remaining well camouflaged from user detection. Moreover, systematic analyses show that existing mitigation strategies are ineffective against discourse-level manipulation, underscoring the urgent need for more robust and adaptive defenses to address discourse-level vulnerabilities. 

---
# Implicit Geographic Inference in LLM Medical Triage: Language-Driven Disparities in Emergency Recommendations 

**Authors**: Qi Han Wong  

**Link**: [PDF](https://arxiv.org/pdf/2606.01204)  

**Abstract**: We investigate whether large language models produce different medical triage recommendations for identical symptoms based solely on the language of the patient prompt. Using Gemini 3.5 Flash, we evaluate a neurological symptom profile (persistent headache, blurred vision, nausea) across six languages (English, Spanish, Chinese, Hindi, Japanese, Arabic) with 30 runs per condition (n=450 total API calls). We find that the model recommends emergency room visits at rates ranging from 0% (Japanese, Hindi) to 30% (English, Arabic), despite assigning nearly identical severity scores (7.7-8.0/10) across all languages. Adding a single sentence specifying the patient's US location increases ER recommendations by up to 76.7 percentage points for non-English prompts, while the reverse anchor (English prompt with a Tokyo location) reduces the ER rate from 30% to 6.7%. A back-translation control (Japanese to English) produces ER rates comparable to the English baseline, confirming that the disparity is not caused by translation quality but by implicit geographic inference from the input language. We release the complete dataset, experiment code, and results. 

---
# Low-Resource Safety Failures Are Action Failures, Not Representation Failures 

**Authors**: Rashad Aziz, Ikhlasul Akmal Hanif, Fajri Koto  

**Link**: [PDF](https://arxiv.org/pdf/2606.01196)  

**Abstract**: Safety alignment learned in high-resource languages transfers poorly to low-resource languages. Models refuse harmful prompts in English but fail to refuse when the same prompts are translated into Swahili or Burmese. Adaptive steering methods like AdaSteer and CAST inherit this failure cross-lingually. We diagnose where transfer breaks down. Across Qwen2.5-7B, Gemma-2-9B, and Llama-3.1-8B on 23 languages, the harmfulness direction extracted from high-resource activations linearly separates harmful from harmless low-resource prompts nearly as well as high-resource ones. The relevant representation is present. Yet harmful refusal drops from 87.9% to 43.9%. The model fails to convert the representation into refusal. What fails to transfer is calibration of the safety decision, not the underlying representation. We exploit this by recalibrating, rather than retraining, a high-resource gate: a low-rank logistic readout with its decision threshold reset using as few as 1 to 4 target-language examples per class. The gate routes between refusal steering and harmfulness-direction ablation, substantially raising mean refusal selectivity ($\Delta$ = harmful $-$ harmless refusal) from 33.6 for the strongest adapted baseline to 54.5 while preserving MMLU utility. These results suggest that some low-resource safety failures can be repaired by recalibrating existing representations rather than learning new ones. Our code is released: this https URL. 

---
# pcbGPT: Automatic PCB Schematic Synthesis from Natural Language Requirements 

**Authors**: Tobias King, Steven Kehrberg, Michael Beigl, Tobias Röddiger  

**Link**: [PDF](https://arxiv.org/pdf/2606.01188)  

**Abstract**: Translating natural-language hardware requirements into correct printed circuit board (PCB) schematics remains difficult in embedded, IoT, and wearable development. Designers must choose compatible components, interpret datasheets, add support circuitry, and expose correct interfaces before layout and prototyping can begin, while many such circuits cannot be validated through straightforward simulation. We present pcbGPT, a grounded system for generating editable KiCad schematics from natural-language specifications. pcbGPT represents circuits in a Python DSL and combines tool-augmented synthesis with component-library search, datasheet-grounded design knowledge, execution-based checking, structural and semantic validation, and an interactive web workflow that supports iterative refinement and synchronization with KiCad projects. We evaluate the system on 20 embedded schematic-generation tasks with reference implementations, required components, and interface constraints that enable automatic comparison. The best model reaches overall pass@1 of 0.90 and pass@5 of 1.00; pass@1 is 1.00 on basic and easy tasks, 0.91 on medium tasks, and 0.72 on hard tasks. These results, together with failure analysis, show that pcbGPT can already generate useful, reviewable first-draft schematics for early prototyping, but is not yet reliable enough to replace expert review. 

---
# Topological Ignorability for Structural Causal Effects Beyond Means 

**Authors**: Usef Faghihi  

**Link**: [PDF](https://arxiv.org/pdf/2606.01184)  

**Abstract**: Many interventions alter the structure of an outcome distribution rather than its mean: they can split a population into disconnected regimes, create loops or holes, generate branches, or reorganize an outcome cloud while leaving the average response nearly unchanged. In such settings, mean-based causal estimands such as the average treatment effect may miss important structural effects.
We introduce topological-geometrical causal metrics based on summaries of interventional outcome laws, including density-superlevel Betti summaries, Euler signatures, and persistent-homology summaries. These metrics quantify structural differences between treated and untreated outcome laws beyond averages. We also study the assumptions needed for causal interpretation. We introduce topological ignorability, a topological analogue of conditional ignorability that requires invariance of the chosen structural feature rather than the full counterfactual distribution. When the chosen summary is injective, this condition coincides with weak ignorability; for noninjective summaries, it can identify the structural feature of interest without identifying the full interventional law.
We define a covariate-standardized topological-geometrical causal effect and develop practical estimators. We validate the framework in two hidden-confounding benchmarks: a fully synthetic exact benchmark and a real-covariate semi-synthetic benchmark using Wisconsin breast-cancer covariates. In both, weak ignorability fails and balancing observed covariates nearly eliminates standardized mean differences, yet the coordinate-mean average treatment effect remains biased. By contrast, selected finite density-superlevel Betti and Euler contrasts remain stable across oracle, observational, and weighted analyses. 

---
# CA-BED: Conversation-Aware Bayesian Experimental Design 

**Authors**: Daniel Arnould, Rashad Aziz, Zixuan Kang, Tanav Changal, Kevin Zhu, Sunishchal Dev, Gabriel Grand, Shreyas Sunil Kulkarni  

**Link**: [PDF](https://arxiv.org/pdf/2606.01182)  

**Abstract**: Large Language Models (LLMs) excel at static reasoning tasks, yet their performance often degrades in interactive scenarios where information must be actively acquired through questioning. A key challenge lies in selecting questions that reduce uncertainty while incorporating responses that may be ambiguous or only partially informative. To address this, we propose Conversation-Aware Bayesian Experimental Design (CA-BED), an inference-time probabilistic dialog planning framework that integrates Bayesian Experimental Design with LLM-based likelihood estimation to optimize question selection over multiple conversational turns. CA-BED maintains a belief distribution over hypotheses, anticipates possible answers, and propagates expected information gain through a simulated conversation tree. Across two structured entity-deduction benchmarks, CA-BED yields an average 21.8% improvement in success rates over direct prompting, with comparable gains relative to alternative information-seeking methods. It achieves these gains with an average increase of only 1.8 conversational turns compared to direct prompting. 

---
# Physics-Informed Deep Learning for Entropy Prediction in Heterogeneous Systems: Thermodynamic and Information-Theoretic Case Studies 

**Authors**: Biswajeet Sahoo, Debadutta Patra  

**Link**: [PDF](https://arxiv.org/pdf/2606.01179)  

**Abstract**: Entropy production governs irreversibility and uncertainty in both physical and information-theoretic systems. While Physics-Informed Neural Networks (PINNs) successfully solve differential equations, current architectures remain inherently domain-specific. The extraction of domain-invariant entropy representations across fundamentally different physical laws remains unexplored. This paper introduces a unified Physics-Informed Deep Learning (PIDL) framework that simultaneously enforces differential equation residuals and information-theoretic bounds within a single neural architecture. We demonstrate this framework via two canonical studies: (i) a thermodynamic continuous stirred-tank reactor (CSTR) model solving governing ODEs, where a Softplus constraint strictly enforces the Second Law of Thermodynamics; and (ii) an information-theoretic financial market model solving the inverse Fokker-Planck PDE to infer latent drift and diffusion coefficients, guaranteeing diffusion positivity via a Softplus constraint while naturally inducing Shannon entropy. Three model variants are evaluated: two domain-specific baselines and one shared-encoder architecture. The PIDL framework guarantees absolute thermodynamic admissibility with zero Second-Law violations and exhibits exceptional data efficiency, retaining >90% predictive accuracy using merely 30% of available training data. Furthermore, a post-hoc Ruppeiner Riemannian geometric analysis of the learned entropy surface successfully identifies thermodynamic phase instabilities. This methodology provides a robust, domain-agnostic architecture for physics-constrained entropy modeling, advancing applications in sustainable process design and quantitative financial risk assessment. 

---
# AI From the Margins (AIM): Rethinking Participatory AI Design Through the Lived Experience of Minoritized Communities 

**Authors**: Tijs Portegies, Laureanne Willems, Maaike Harbers, Giovanni Sileno, Roland van Dierendonck, Mayesha Tasnim, Lotte Willemsen, Sennay Ghebreab  

**Link**: [PDF](https://arxiv.org/pdf/2606.01171)  

**Abstract**: Artificial intelligence (AI) can reproduce and amplify the structural inequities faced by minoritized communities. Participatory AI has been proposed as a response, but participation typically starts after problem definitions and success criteria have been set, leaving limited room for minoritized communities to reshape what an AI system is for. We propose AI From the Margins (AIM): a methodological stance that articulates the conditions under which lived experiences of minoritized communities can be elicited, centered, and carried forward to inform participatory AI design. AIM is not a fixed protocol; it articulates a set of preconditions that can be enacted through different techniques in different settings. We applied AIM in a Dutch healthcare context in eight sessions with 13 women and non-binary people of color and five municipal policy workers, namely through (1) narrative elicitation using the Biographic Narrative Interpretive Method (BNIM); (2) co-constructed rule-making; (3) participants' determination of whether, where, and how AI should be involved; and (4) translating lived experience into AI policy through dialogue with policymakers. In their reflections on the sessions, participants described the engagement as substantive and called for its continuation, demonstrating how preparatory orientation fundamentally grounded in lived experience shapes what participatory AI design is for. 

---
# When Data Is Scarce: Scaling Sparse Language Models with Repeated Training 

**Authors**: Boqian Wu, Qiao Xiao, Patrik Okanovic, Tomasz Sternal, Maurice van Keulen, Mykola Pechenizkiy, Elena Mocanu, Torsten Hoefler, Decebal Constantin Mocanu  

**Link**: [PDF](https://arxiv.org/pdf/2606.01155)  

**Abstract**: Scaling laws for dense LLMs under infinite data are well explored, but how sparsity interacts with limited data is not. In this work, we study sparse training in data-constrained regimes where limited unique tokens require multi-epoch training. Our experiments span models up to 1.92B parameters in the fitting set, sparsity up to 93.75%, unique data budgets up to 2.6B tokens, and total training tokens up to 41.6B over 16 epochs; we further validate extrapolation on held-out dense-equivalent models up to 7.68B parameters. We find that: 1. Sparse scaling in data-limited settings: We introduce a scaling law that models loss as a function of active parameters, unique tokens, data repetition, and sparsity, accurately predicting performance across compute and data budgets. 2. Delayed data saturation: sparse training postpones diminishing returns from repeated data, making multi-epoch training more effective. 3. Resource trade-offs: With fixed data, loss-optimal sparsity is moderate ~ 50%, while compute-optimal sparsity is higher and grows with data scale. Overall, sparsity is not just a tool for efficiency, but a mechanism for improving scaling trade-offs under data scarcity. Our code is available at: this https URL. 

---
# ASE-26: a curriculum for agentic software engineering as a discipline 

**Authors**: Mikael Gorsky  

**Link**: [PDF](https://arxiv.org/pdf/2606.01152)  

**Abstract**: The work of a professional software engineer has begun to consist, increasingly, of directing agents rather than writing code, and the empirical evidence for the shift is now several years deep. Anthropic's Economic Index puts automation at 79 per cent of Claude Code interactions [2]; Handa and colleagues at Anthropic find AI exposure for Computer Programmer tasks at approximately 75 per cent of the role's distinct activities [3]; Brynjolfsson and colleagues at Stanford's Digital Economy Lab report a 13 per cent relative decline in employment for workers aged 22 to 25 in occupations most exposed to AI [4]. The shift is also unfinished, and the academic literature on agentic software engineering converges on the finding that the missing capability is not better models but structured practitioner discipline. This paper presents ASE-26, a comprehensive undergraduate curriculum for agentic software engineering as a discipline, deposited as a citable reference on Zenodo under CC BY-ND 4.0 [12]. The paper sets out the discipline framing the curriculum rests on, the conceptual contributions it makes (most importantly, the evolutionary spiral as the operational form of the co-evolution of intent and build), the twenty-one-module structure that organises the discipline for teaching, the pedagogical commitments that follow from grading work co-produced with an agent, what graduates leave with, and how the discipline as taught is designed to outlast the specific capabilities of today's models. The position the paper takes is that the practitioner skills the industry currently lacks are precisely the skills the discipline names, and that structured undergraduate curricula in agentic software engineering are the principal mechanism by which the gap closes. 

---
# AMP: A Vendor-Neutral Wire Format for Agent Memory Operations 

**Authors**: Thamilvendhan Munirathinam  

**Link**: [PDF](https://arxiv.org/pdf/2606.01138)  

**Abstract**: Agent-memory frameworks - mem0, Letta/MemGPT, Cognee, Zep/Graphiti, MemoryOS, MemTensor - each ship their own SDK, storage layout, and operational vocabulary. There is no shared wire format: every integration is bespoke, every migration rebuilds memory from scratch, and no framework ships a governance surface that lets a human review writes before they enter long-term storage. We present memorywire, a JSON-Schema 2020-12 wire format for five memory operations (remember, recall, forget, merge, expire) over four memory types (semantic, episodic, procedural, emotional), with a MemoryStore interface, a fan-out router, and an optional HITL governance channel. We describe an open-source reference implementation with five backend adapters (sqlite-vec, mem0, Letta, Cognee, pgvector); a microbenchmark on a 100-fact / 50-query labelled corpus achieving recall@5 = 1.000 on the 42 labelled queries with ingest p50 = 37.8 ms and recall p50 = 40.6 ms; an adversarial-fusion experiment showing Reciprocal Rank Fusion holds recall@5 = 1.000 across a 1-of-N rank-0 injection sweep (K in {0,5,...,50}) where max fusion collapses to 0.500 with 80% leak at K >= 5; and a 16-scenario cross-adapter conformance suite passing 68 of 80 cells with zero failures. The contribution is not a new algorithm; it is a packaging of established components (RRF, FSMs, STM/LTM consolidation, diff-and-approve workflows) into a venue-neutral protocol with an empirically validated reference, positioned to compose with the Model Context Protocol rather than compete with it. 

---
# STARFISH: faST Accuracy Recovery in pruned networks From Internal State Healing 

**Authors**: Shir Maon, Odelia Melamed, Adi Shamir  

**Link**: [PDF](https://arxiv.org/pdf/2606.01126)  

**Abstract**: Pruning is a process designed to reduce the number of weights in a large neural network. This can substantially speed up inference but might cause a considerable reduction in the model's accuracy, and thus it is usually followed by a healing process that regains some of the lost accuracy. In this paper, we propose a new healing method, STARFISH, that can recover (most of) the accuracy of any pruned network efficiently. The main idea of STARFISH is to optimize the pruned network to align with the original network's internal state representations using a tiny calibration set of unlabeled examples. For the common case of removing 50% of the weights, STARFISH healing improves the recovered accuracy by up to 22% over the state-of-the-art methods on ViT-based networks. Its advantage is even more pronounced under aggressive pruning. For example, after eliminating 75% of the weights in a DeiT-B network for ImageNet, STARFISH uses only 0.4% of the number of training images as a calibration set and recovers 82% of the original dense accuracy, whereas competing recovery techniques reach only 40% of the dense model accuracy. 

---
# HASTE: Hardware-Aware Dynamic Sparse Training for Large Output Spaces 

**Authors**: Nasib Ullah, Jinbin Zhang, Jean Lucien Randrianantenaina, Erik Schultheis, Rohit Babbar  

**Link**: [PDF](https://arxiv.org/pdf/2606.01117)  

**Abstract**: Extreme multi-label classification (XMC) involves learning models over large output spaces with millions of labels, making the output layer a memory-compute bottleneck. While sparsity-based methods reduce arithmetic complexity, they often fail to yield proportional speedups due to irregular memory access, poor hardware utilization, or reliance on auxiliary architectural components in long-tailed regimes. We introduce group-shared fixed fan-in sparsity, a semi-structured output-layer design in which semantically related labels share a sparse input pattern while retaining independent weights. This grouping introduces a task-aligned inductive bias -- encouraging related labels to share feature subsets -- while reducing index memory overhead, increasing feature reuse across labels, and enabling efficient GPU execution via custom CUDA kernels that leverage modern accelerator primitives. As an alternative to auxiliary objectives, we exploit the long-tailed structure of XMC by decomposing the output layer into a small dense head over frequent labels and a group-shared sparse tail over the remainder, providing an informative gradient pathway while preserving the memory benefits of sparsity. Through kernel-level microbenchmarking, we show that group-shared fixed fan-in translates arithmetic reductions into practical wall-clock gains, achieving up to $4.4\times$ speedup in the forward pass and up to $25\times$ speedup in backward passes over standard fixed fan-in sparsity, while operating within a few percent of a FLOPs-matched dense bottleneck. Across large-scale XMC benchmarks, our approach matches or improves precision@k over prior sparse baselines, while narrowing the performance gap to dense. 

---
# Soft-NBCE: Entropy-Weighted Chunk Fusion for Long-Context 

**Authors**: Shihao Ji, Mingyu Li, Zihui Song  

**Link**: [PDF](https://arxiv.org/pdf/2606.01101)  

**Abstract**: The quadratic complexity of self-attention remains a bottleneck for Large Language Models (LLMs) processing ultra-long contexts. The Naive Bayes Cognitive Engine (NBCE) parallelizes long-context inference by chunking documents and routing to the lowest-entropy chunk at each decoding step. This hard-selection strategy causes semantic fragmentation during cross-chunk reasoning, as abrupt routing changes between adjacent tokens disrupt the model's contextual grounding.
We present Soft-NBCE, a lightweight extension that replaces discrete chunk selection with soft entropy-weighted chunk fusion. A temperature-scaled Softmax over predictive entropies assigns continuous weights to all chunks, enabling log-space aggregation across chunk-conditioned distributions. To partially compensate for the conditional independence assumption introduced by chunking, we propose Consistency Distillation, a LoRA-based self-distillation that constrains the chunked logit distribution toward a full-context teacher via KL-divergence. On LongBench multi-hop benchmarks, Soft-NBCE with Consistency Distillation improves consistently over NBCE-style baselines (MuSiQue F1: 0.310 vs.\ 0.275 for Vanilla NBCE; HotpotQA F1: 0.479 vs.\ 0.427) while maintaining retrieval accuracy (NIAH-32K: 0.909) at O(L^2/n) peak memory. 

---
# MiCU: End-to-End Smart Home Command Understanding with Large Language Model 

**Authors**: Haowei Han, Kexin Hu, Weiwei Cai, Debiao Zhang, Bin Qin, Yuxiang Wang, Jiawei Jiang, Xiao Yan, Bo Du  

**Link**: [PDF](https://arxiv.org/pdf/2606.01099)  

**Abstract**: Command understanding systems in smart home ecosystems can automate device control and substantially improve user experience. However, while they perform well on precise utterances (e.g., "turn on the bedroom light"), they struggle with ambiguous or misaligned commands (e.g., "make the bedroom cozy"). Large language models (LLMs) generalize well across various domains and can outperform traditional rule-based systems on such tasks, but their effectiveness is often constrained by scarce domain-specific data, insufficient task-specific adaptation, and high computational costs. In this paper, we propose an automated training data synthesis workflow using user logs and LLMs; then we build MiCU, a domain-specific LLM that excels at command understanding. Specifically, we employ curriculum learning to inject domain knowledge into the base LLM, then we enhance its reasoning ability via cold-start training combined with reinforcement learning (RL) guided by domain-specific thinking rules. Additionally, we introduce a token compression technique that condenses device description into a single special token, substantially reducing inference overhead and enabling \model-fast, an efficient variant optimized for long inputs. Extensive experiments show that MiCU significantly outperforms baselines, with an average accuracy gain of 20.01% across all device categories. We have deployed MiCU in the Xiaomi Home app, receiving approximately 1.7 million page views per day. Production evaluations show that MiCU reduces user correction rate by 1.57% and increases human audited accuracy by 32.05%. Our data and code are available at this https URL 

---
# Implicit Drifting Policy: One-Step Action Generation via Conditional Expert Geometry 

**Authors**: Zemin Yang, Yaoyu He, Yiming Zhong, Yuhao Zhang, Xinge Zhu, Yao Mu, Qingqiu Huang, Yuexin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2606.01098)  

**Abstract**: Generative action policies based on diffusion or flow matching excel in behavior cloning, yet their iterative sampling is prohibitive for high-frequency robot control. While recent one-step formulations alleviate this latency, they inevitably discard the intermediate trajectory evolution that provides crucial action correction. Directly recovering this mechanism by explicitly estimating a training-time drifting field is mathematically ill-posed due to extreme conditional demonstration sparsity. We introduce Implicit Drifting Policy (IDP), a one-step imitation learning framework that brings the training-time correction of Drifting into policy learning without explicit vector field estimation. IDP extracts a conditional expert geometry from the local variation of observation-similar expert actions, and compares it against a global reference geometry to isolate condition-specific constraints. This local geometric structure adaptively weights a scalar potential objective. Combined with an expert-proximal terminal evaluation, IDP directly enforces manifold constraints on the one-step generator during training. Extensive evaluations across 2D, 3D, and real-world manipulation tasks show IDP effectively maintains adherence to valid action manifolds, improving upon explicit drifting methods and achieving competitive performance with strong one-step baselines. 

---
# Beyond Task Success: Behavioral and Representational Diagnostics for WAM and VLA 

**Authors**: Hung Mai, Bin Zhu, Tuan Do  

**Link**: [PDF](https://arxiv.org/pdf/2606.01095)  

**Abstract**: Vision-language-action (VLA) policies and World-Action Models (WAM) represent two increasingly important paradigms for robotic manipulation. However, it remains unclear whether future prediction in WAMs leads to behaviorally meaningful improvements beyond final task success. In this paper, we ask whether WAMs merely add future prediction, or whether they change robot behavior and internal representations in ways that are actionable for control. We introduce a model-agnostic diagnostic framework that compares WAMs and VLAs through two complementary lenses: behavioral rollout analysis and sparse-autoencoder-based feature analysis. The behavioral protocol measures action dynamics consistency, target-object progress, distractor disturbance, and runtime cost. The feature-space protocol characterizes internal representations as memorized, reactive, or predictive, revealing whether models encode future-oriented structure. Across LIBERO and RoboTwin2.0, we evaluate 7 policies spanning direct VLAs and joint, sequential, and auxiliary WAMs. Our results show that success alone hides key differences: WAMs often improve object-level behavior and target selectivity, but their gains depend on architecture and incur higher inference cost. Sequential WAMs show the clearest predictive structure, while auxiliary and joint WAMs respectively compress or entangle future information. These findings suggest future directions for WAMs design to preserve behaviorally actionable future representations for efficient manipulation. 

---
# A Fiber Criterion for Representation Identifiability in Supervised Learning 

**Authors**: Vasileios Sevetlidis  

**Link**: [PDF](https://arxiv.org/pdf/2606.01092)  

**Abstract**: Supervised learning evaluates predictors through their input-output behavior. When a predictor is implemented as a composition $f=c\circ h$, supervised evidence constrains the composite map $f$ but need not determine the representation-head factorization $(h,c)$. This paper formalizes the resulting representation-level identifiability problem: for a class of admissible representation-head pairs, a representation property is identifiable from the induced predictor exactly when it is constant on the fibers of the projection $(h,c)\mapsto c\circ h$, equivalently when it descends to a well-defined property of the predictor. Predictor-preserving augmentation gives a canonical obstruction: auxiliary information can be appended to a representation while the head ignores it, leaving the predictor unchanged but altering properties such as minimality, compression, invariance, equivariance, nuisance information, or semantic accessibility. This construction separates representation identifiability from optimization and finite-sample estimation. Finite-sample diagnostics illustrate, rather than prove, the criterion: exact algebraic witnesses hold the predictor fixed while changing representation diagnostics, and matched-performance Waterbirds models show that different constraints can select different representations at similar supervised performance. The results clarify that representation-level claims require assumptions, objectives, measurements, or inductive biases beyond supervised predictive behavior alone. 

---
# Strong Stochastic Flow Maps 

**Authors**: Sam McCallum, Zander W. Blasingame, Timothy Herschell, Niklas Rindtorff, Alexander Tong, James Foster  

**Link**: [PDF](https://arxiv.org/pdf/2606.01086)  

**Abstract**: Flow and diffusion models generate high-quality samples in many modalities; however, many network evaluations are required during inference due to numerical integration of an underlying differential equation. Flow maps alleviate this problem by learning the solution map of the differential equation directly, enabling few-step sampling. Yet, current methods are restricted to approximating the solution map of ODEs. These methods can be used to learn the transition kernel of an SDE, thereby obtaining a solution map that recovers the marginal distributions of the process (weak convergence) rather than the solution path (strong convergence). We propose Strong Stochastic Flow Maps (SSFMs) as a novel framework for learning the strong solution map of additive-noise SDEs, directly generalizing deterministic flow maps to the stochastic setting. Further, a polynomial approximation to Brownian motion is introduced and shown to converge pathwise. These results enable a simulation-free training objective for the solution map of diffusion models. We demonstrate that SSFMs outperform previous stochastic flow map methods on image generation and enable few-step sampling of molecular systems. 

---
# MViewRouter: Internalizing Geometric Equivariance via Multi-view Alternating Attention for Combinatorial Routing 

**Authors**: Shiyan Liu, Bohan Tan, Yaoxin Wu, Yan Jin  

**Link**: [PDF](https://arxiv.org/pdf/2606.01084)  

**Abstract**: Combinatorial routing problems such as the Traveling Salesman Problem (TSP) and the Capacitated Vehicle Routing Problem (CVRP) are fundamental NP-hard problems with broad real-world applications. While recent deep reinforcement learning methods have shown promising performance, they typically handle geometric symmetries only through data augmentation, resulting in inconsistent decisions and limited generalization. To address this issue, we propose MViewRouter, a multi-view framework that internalizes geometric equivariance as a structural inductive bias to achieve invariant decision-making across routing problem variants. Our approach introduces a Multi-view Alternating Attention (MAA) mechanism that enables parallel processing over the $D_4$ symmetry group, alternating between intra-view relational modeling and inter-view feature alignment. Furthermore, we optimize the policy via Collective Policy Gradient Aggregation (CPGA), leveraging consensus gradients from multiple symmetric views to stabilize training and accelerate convergence. Experiments on TSP and CVRP benchmarks, as well as real-world TSPLIB instances, demonstrate that MViewRouter achieves competitive solution quality and strong zero-shot generalization. 

---
# ThinkSwitch: Context Distillation with LoRA and Weight Interpolation for Specific-Purpose Reasoning Tasks 

**Authors**: Dhruv Saini, Rohan Pandey  

**Link**: [PDF](https://arxiv.org/pdf/2606.01080)  

**Abstract**: Large language models often improve on difficult tasks by spending inference-time compute on a reasoning trace before producing the final answer. That extra computation can be useful, but it also raises latency, token cost, and deployment complexity. We introduce \textbf{ThinkSwitch}, a low-compute procedure for co-training paired instruct and thinking checkpoints. Starting from compatible Qwen3-4B instruct and thinking models, each iteration asks the thinking checkpoint to generate answers, removes the reasoning trace, distills the answer-only pairs into the instruct checkpoint with QLoRA, and reconstructs a thinking checkpoint with spherical weight interpolation. The only human-supplied inputs are task prompts; the labels are generated by the model itself. On a 30-question AIME 2026 evaluation, ThinkSwitch improves the instruct checkpoint from 10/30 to 20/30 and the thinking checkpoint from 14/30 to 22/30. On a 30-question PubMedQA subset, it improves the instruct checkpoint from 13/30 to 18/30 and the thinking checkpoint from 18/30 to 25/30. The complete experiment uses 15 training prompts per domain and costs \$2.86 on a single cloud RTX 3070. The results are small-scale, but they indicate that targeted distillation loops can move part of the benefit of explicit reasoning into weights while preserving a separate thinking mode. 

---
# Test-Time Training for Zero-Resource Dense Retrieval Reranking 

**Authors**: Shiyan Liu, Yichen Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.01070)  

**Abstract**: Dense retrievers excel at first-stage candidate generation but lack effective reranking in zero-resource settings. Existing approaches face a fundamental dilemma: cross-encoders deliver strong reranking quality but require costly supervised training and incur high latency, while unsupervised BM25 reranking consistently degrades dense retrieval performance on most of BEIR benchmarks. We propose DART (Dense Adaptive Reranking at Test-time), which resolves this dilemma by adapting the scoring function at inference time. For each query, the top-ranked documents serve as pseudo-positive examples and the bottom-ranked as pseudo-negative examples, providing noisy but readily available supervision to adapt a bilinear scoring matrix $W$ via a small number of gradient updates. We further introduce a confidence-weighted margin loss and a cross-query momentum buffer that warm-starts adaptation across queries. On six BEIR benchmarks, DART achieves a mean per-dataset relative NDCG@10 gain of +2.1% over the dense retrieval baseline with under 10ms additional latency per query, demonstrating a powerful capability for zero-shot performance enhancement and cross-domain generalization. 

---
# Leyline: KV Cache Directives for Agentic Inference 

**Authors**: Bole Ma, Jan Eitzinger, Harald Koestler  

**Link**: [PDF](https://arxiv.org/pdf/2606.01065)  

**Abstract**: Modern KV cache management assumes the chatbot workload: prompts arrive once and the cache grows append-only, so prefix caching and forward-only eviction are correct by construction. Agentic LLMs break this assumption. Their conversations evolve through policy-driven editing: failed tool calls are retried, stale outputs dropped, trajectories pivoted. Two distinct cache problems result. First, identical content moves to new positions between turns, invalidating exact-prefix caches even though the underlying KV would still be valid; recent work on position-independent caching for MLA addresses this reuse problem. Second, and this paper's focus, a policy may need to direct the serving system to actively remove or replace a span of cached content and continue without re-prefilling everything that came after. No existing primitive offers this. Production agentic harnesses fall back to re-prefill on every edit, paying full prefix-recomputation cost; kernel-level eviction methods make their own decisions and cannot accept policy directives from outside the kernel. We introduce Leyline, a serving-side primitive that closes this gap. A declarative directive 4-tuple separates what to edit from how to preserve position correctness. The policy declares the edit and its mode (in-place splice or prefix-trimmed re-prefill for semantic forgetting); an architecture-agnostic interface routes to a per-architecture kernel that restores attention math via a closed-form RoPE-rotation correction. The splice kernel lifts replay cache-hit by +11.2 pp and cuts latency by up to 241 ms. A ten-line truncation rule routed through the same interface lifts agentic solve rate by +14.3 pp on debug-gym. The mechanism is open; the policy space it enables is the agenda. 

---
# MENTIS: What Belief Changes Under Alignment? Measuring Multi-Scale Latent Torsion in Language Models 

**Authors**: Partha Pratim Saha, Samarth Raina, Mayur Parvatikar, Amit Dhanda, Vinija Jain, Aman Chadha, Amitava Das  

**Link**: [PDF](https://arxiv.org/pdf/2606.01060)  

**Abstract**: Preference alignment has substantially improved the observable behavior of large language models, yet it remains unclear what alignment changes internally. Aligned systems still fail under jailbreaks, prompt injection, and retrieval-time corruption, suggesting behavior-level evaluation alone is incomplete. Post-training should leave measurable traces in internal computation. We ask: when an instruction-tuned (IT) model becomes a preference-aligned (PA) model, what geometric structure changes, where do those changes concentrate, and how selectively do they vary across concepts, prompts, and model families?
We introduce MENTIS, a geometry-first framework for measuring alignment-induced internal reorganization in paired checkpoints. MENTIS compares IT and PA models using a primary layerwise covariance-based torsion norm (T1), a secondary spectral torsion diagnostic (T2), and an Energy-Radiance-Activation measure (ERA) for depth localization. Across four 7-8B model pairs on LITMUS, our study reveals that alignment-induced change is selective rather than uniform: normative concepts exhibit larger torsion shifts than factual concepts on average; torsion is negatively correlated with contextual entropy; and peak effects localize to architecture-specific mid-to-late layers. The same pattern appears across word-level, prompt-level, and model-level analyses. These results suggest preference alignment leaves structured, depth-localized geometric signatures in internal computation beyond what behavior-level evaluation alone can reveal. 

---
# 3DCodeBench: Benchmarking Agentic Procedural 3D Modeling Via Code 

**Authors**: Yipeng Gao, Lei Shu, Genzhi Ye, Xi Xiong, Ameesh Makadia, Meiqi Guo, Laurent Itti, Jindong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.01057)  

**Abstract**: Procedural 3D modeling through code is emerging as a versatile paradigm, offering deterministic, engine-ready, and precisely editable assets that neural 3D generators inherently lack. Authoring such procedural content, however, demands deep expertise in 3D software APIs, parametric design, and code-level geometric reasoning. In this paper, we propose 3DCodeBench, a systematic benchmark for evaluating vision-language model (VLM) agents for procedural 3D generation in 3D modeling software. Specifically, 3DCodeBench evaluates how effectively 12 advanced VLMs can serve as procedural 3D modelers by translating text and image references into procedural code for 3D modeling software. Recognizing that automated metrics may not fully capture the perceptual quality of 3D shapes, we build 3DCodeArena, a ranking platform based on pairwise human preferences over generated 3D outputs. From extensive evaluations and results, we observe that: (1) Failures mostly arise from API mismatches, while successful renders still suffer from disconnected or floating 3D geometric components. (2) Test-time scaling, such as higher thinking budgets and multi-turn refinement, improves performance overall. Our findings highlight a critical need for high-quality procedural coding data to advance commercial VLMs. Furthermore, effective procedural 3D modeling requires a robust execution environment that provides high-fidelity feedback for iterative refinement. We release 3DCodeBench, including the curated large-scale dataset of multimodal (text/image) prompts, procedural code, 3D object triplets, evaluation protocol, and the public 3DCodeArena platform as a foundational toolkit for exploring VLM-based procedural 3D modelers. 

---
# Plausibility Is Not Prediction: Contrastive Evidence for LLM-Based Cellular Perturbation Reasoning 

**Authors**: Xinyu Yuan, Xixian Liu, Jianan Zhao, Yashi Zhang, Hongyu Guo, Jian Tang  

**Link**: [PDF](https://arxiv.org/pdf/2606.01042)  

**Abstract**: Perturbation experiments are central to understanding cellular mechanisms, but remain costly and sparse, motivating prediction of gene expression responses for unobserved conditions. A promising recent direction leverages large language models (LLMs) as "virtual cell" simulators-using stepwise, knowledge-grounded mechanistic reasoning to infer differential expression-pointing toward an interpretable, knowledge-driven paradigm that transcends purely data-driven approaches. However, we find that plausibility is not prediction: despite producing biologically plausible explanations, these methods fail to capture perturbation-specific effects: systematically overestimating differential expression, often underperforming a simple gene-frequency baseline in aggregate evaluations, and collapsing to chance-level performance at the per-gene level. This reveals a reliance on intrinsic gene response tendencies rather than true perturbation reasoning. We trace this failure to how evidence is presented: existing methods evaluate perturbation-gene pairs in isolation, without exposing how related perturbations differ in their effects on the same gene. To address this limitation, we introduce CORE (Contrastive Organization of Relational Evidence), which reframes prediction as a comparison task by organizing evidence into positive and negative outcomes from related perturbations. Using a biomedical knowledge graph for evidence retrieval, CORE improves calibration and substantially boosts perturbation-specific prediction in both LLM-based and non-LLM settings: for example, on drug-perturbation data, CORE-Reasoning improves Qwen3.5-9B aggregate metrics by up to 28.6%, while on generic perturbation data, CORE-Voting raises macro-per-gene AUROC from chance to 0.703 in average across four cell lines. This highlights contrastive evidence organization as essential to reliable LLM-based perturbation reasoning 

---
# OPD+: Rethinking the Advantage Design for On-Policy Distillation 

**Authors**: Hanyang Zhao, Haoxian Chen, Han Lin, Genta Indra Winata, David Yao, Wenpin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2606.01039)  

**Abstract**: On-policy distillation (OPD) is a widely used technique to transfer capabilities from capable teacher language models to the base student models, and can be formulated in a reinforcement learning style objective using student generated rollouts. Yet, despite the divergence reward being dependent on student model likelihood, existing works usually adopt a stop gradient design primarily for stability, which makes the resulting advantage estimation questionable. In this work, we provide a generic optimization framework based on f-divergence between the student and teacher, and mathematically revisit whether such design space is valid. We prove that general stop-gradient operation would lead to biased estimates of the reward objective and corresponding gradient for general divergence functions. We propose OPD+, the corrected version of OPD that demonstrates improved performance over the baseline KL approach and also supports the choice of various f-divergence. We validate our findings on mathematical reasoning and tool-use benchmarks. 

---
# Temporally-Aligned Evaluation for Audio-Driven Talking Head Generation 

**Authors**: Zhicheng Zhang, Lei Wang, Yu Zhang, Yongsheng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2606.01031)  

**Abstract**: Audio-driven talking-head generation has advanced rapidly, yet existing evaluation protocols mainly rely on frame-wise metrics that assume strict temporal correspondence between generated and reference videos. This assumption does not match speech-driven facial motion, which naturally includes slight timing shifts, different speaking speeds, and stylistic variations. As a result, conventional metrics may treat harmless timing differences as quality errors, making it harder to fairly compare methods and understand their trade-offs. In this work, we argue that evaluation of dynamic generative models should be formulated as a sequence-alignment problem rather than independent frame comparison. We introduce a unified sequence-level reformulation that integrates Soft Dynamic Time Warping into established evaluation pipelines. By aligning feature trajectories while preserving temporal order, the proposed framework provides robustness to bounded temporal misalignments without altering the underlying perceptual, identity, or synchronization encoders. We show that frame-wise evaluation can be viewed as a special case under rigid alignment, while sequence-level alignment provides improved stability, lower sensitivity to timing differences, and clearer separation between modeling paradigms. Building on this principled formulation, we conduct a large-scale benchmark of 20 methods across seven datasets spanning canonical, in-the-wild, and style-diverse scenarios under standardized protocols. Extensive experiments show that temporally aligned metrics are more robust to timing differences, provide more consistent results across datasets, and better reveal systematic trade-offs between modeling paradigms, such as synchronization versus realism and expressiveness versus stability. 

---
# DSL-LLaDA: Scaling Continuous Denoising to 8B Masked Diffusion LMs 

**Authors**: Longxuan Yu, Yunshu Wu, Yu Fu, Siheng Xiong, Rob Brekelmans, Hui Liu, Yue Dong, Greg Ver Steeg  

**Link**: [PDF](https://arxiv.org/pdf/2606.01024)  

**Abstract**: Discrete Masked diffusion language models generate text by iterative parallel decoding, but few-step decoding suffers from a tradeoff between length and quality: with a fixed step budget, standard methods can generate a short, high-quality output, or they can produce long but repetitive text. Continuous denoising can sidestep this tradeoff by evolving all positions jointly in embedding space, but building such a model from scratch at scale remains an open problem. We show that a pretrained masked DLM can instead be lightly adapted to support continuous embedding-space denoising. Starting from LLaDA-8B-Instruct, we continue-pretrain for only 1,000 steps with Discrete Stochastic Localization (DSL), replacing binary masking with continuous per-token Gaussian noise as a soft mask. The adapted model supports continuous inference that evolves all positions jointly in embedding space and defers hard token commitment to the final step. On zero-shot summarization at low step budgets (<=16 forward passes), DSL-LLaDA-SDE achieves the best ROUGE-1 on all four benchmarks and largely avoids the premature-termination / repetition tradeoff of iterative unmasking. The same adaptation also yields selective noisy-state robustness: the model corrects corrupted tokens while preserving clean ones. Control experiments using standard masked diffusion training with the same compute demonstrate neither behavior. 

---
# Data Collection for Training Quality-Control AI in Carpet Manufacturing 

**Authors**: Akbar Erkinov  

**Link**: [PDF](https://arxiv.org/pdf/2606.01023)  

**Abstract**: Visual inspection remains the dominant quality-control practice in woven and tufted carpet production, yet it is slow, subjective, and inconsistent at the line speeds and widths of modern looms. We present a design proposal for an in-line machine-vision system whose primary purpose is twofold: to inspect the carpet web in real time and, equally importantly, to systematically collect and label images of defect patterns so that increasingly capable quality-control models can be trained over the life of the this http URL proposal is grounded in a concrete industrial setting: a Six Sigma (DMAIC) project at a woven-carpet production facility that anticipated a production bottleneck following the installation of additional weaving machines, with a substantial baseline defect rate and significant financial exposure associated with quality failures. We describe an imaging subsystem based on synchronized line-scan cameras with combined bright-field and grazing illumination, derive the resolution and throughput requirements needed to resolve fine structural defects across a multi-metre web, and define a carpet-specific defect this http URL then lay out a staged modelling strategy that begins with unsupervised anomaly detection trained on defect-free material, following the paradigm exemplified by the carpet category of the MVTec Anomaly Detection benchmark, and matures through a human-in-the-loop annotation flywheel into supervised detection and segmentation models. Finally, we connect detection performance to the DMAIC objectives, showing how reductions in escaped defects translate into improved process quality and process sigma levels. The contribution is an end-to-end, deployable blueprint that treats data collection as a first-class engineering objective rather than an afterthought. 

---
# ProductWebGen: Benchmarking Multimodal Product Webpage Generation 

**Authors**: Zhihong Liu, Siqi Kou, Zheng Li, Ye Ma, Quan Chen, Peng Jiang, Kai Yu, Zhijie Deng  

**Link**: [PDF](https://arxiv.org/pdf/2606.01022)  

**Abstract**: Crafting a product display webpage from a source product image, along with layout and visual content instructions, holds significant practical value for domains such as marketing, advertising, and E-commerce. Intuitively, this task demands strict visual consistency across product displays and high-fidelity instruction following to jointly generate renderable HTML code. These requirements on controllability and instruction-following are closely aligned with the core features of advanced multimodal generative models, such as image editing models and unified models. To this end, this paper introduces ProductWebGen to systematically benchmark the product webpage generation capacities of these models. We organize ProductWebGen with 500 test samples covering 13 product categories; each sample consists of a source image, a visual content instruction, and a webpage instruction. The task is to generate a product showcase webpage including multiple consistent images in accordance with the source image and instructions. Given the mixed-modality input-output nature of the task, we design and systematically compare two workflows for evaluation -- one uses large language models and image editing models to separately generate HTML code and images (editing-based), while the other relies on a single UM to generate both, with image generation conditioned on the preceding multimodal context (UM-based). Empirical results show that editing-based approaches achieve leading results in webpage instruction following and content appeal, while UM-based ones may display more advantages in fulfilling visual content instructions. We also construct a supervised fine-tuning dataset, ProductWebGen-1k, with 1,000 groups of real product images and LLM-generated HTML code. We verify its effectiveness on the open-source UM BAGEL. The data and code are available at this https URL. 

---
# Hybrid Verified Decoding: Learning to Allocate Verification in Speculative Decoding 

**Authors**: Xin Su, Dawid Majchrowski, Fangyuan Yu, Vanshil Atul Shah, Sebastian Rogawski, Pawel Morkisz, Anahita Bhiwandiwalla, Phillip Howard  

**Link**: [PDF](https://arxiv.org/pdf/2606.01019)  

**Abstract**: Large Language Model (LLM) generation remains expensive because autoregressive decoding calls the model once for each new token. Speculative decoding reduces this cost by drafting multiple tokens and verifying them with the target model in one step, but its speedup depends on how many drafted tokens are accepted. Parameter-free draft sources can propose long continuations at low cost in structured and agentic workloads, yet a cache match that looks promising at one generation step may have low payoff at the next. We propose Hybrid Verified Decoding, which predicts the accepted length of a cache draft before verification and uses this payoff estimate to choose between cache verification and a model-based drafter. Across three LLMs and sixteen datasets, Hybrid Verified Decoding is especially effective on agentic workflows, where it outperforms EAGLE3 in every setting with a 2.73x average speedup. Our analysis shows how prompt structure creates cache opportunities, how high-payoff cache drafts concentrate in a small part of the draft space, and how payoff-guided selection reduces sequential decoding work, pointing to runtime draft selection as a promising direction for speculative decoding. 

---
# PolySpeech-100: A Large-Scale Benchmark for Speech Understanding Across 100+ Languages and Dialects 

**Authors**: Sicheng Yang, Shulan Ruan, Shiwei Wu, Yu Liu, Lu Fan, Zhi Li, You He  

**Link**: [PDF](https://arxiv.org/pdf/2606.01016)  

**Abstract**: While End-to-End (E2E) Speech-Large Language Models (Speech-LLMs) are rapidly evolving, their evaluation methodologies remain limited to the era of simple transcription. Existing benchmarks suffer from three critical limitations: a pronounced bias towards high-resource languages, a focus on low-level recognition (ASR) rather than semantic reasoning, and a neglect of regional dialects. To bridge this gap, we introduce PolySpeech-100, a massive-scale benchmark designed to assess `native-level' speech comprehension across 110 linguistic variants. We employ a novel hybrid construction pipeline that augments gold-standard human recordings with instruction-driven synthetic speech, allowing us to cover 19 distinct Chinese dialects and over 80 low-resource languages. Extensive evaluation of 22 state-of-the-art models (including Gemini-3, GPT-Audio, and Qwen2.5-Omni) yields pivotal insights. First, we demonstrate that open-source E2E models outperform Cascade (ASR+LLM) systems on heavy dialects, proving that direct audio processing preserves critical paralinguistic cues and prosodic features (e.g., intonation, stress) that are often lost in standard transcription. Second, we reveal a significant performance gap: while commercial models maintain robustness, open-source models suffer catastrophic degradation on low-resource languages. Finally, counter-intuitively, we observe that under standard zero-shot settings, Chain-of-Thought prompting frequently degrades speech understanding performance for most evaluated models, revealing a potential modality alignment gap in current architectures. PolySpeech-100 establishes a rigorous standard for the next generation of inclusive, omni-capable Speech-LLMs. The data, demo, and code are publicly available at this https URL. 

---
# AI-IoT-Robotics Integration: Survey of Frameworks, Emerging Trends, and the Path Toward Connected Robotics 

**Authors**: Ranulfo Bezerra, Satoshi Tadokoro, Kazunori Ohno  

**Link**: [PDF](https://arxiv.org/pdf/2606.01015)  

**Abstract**: The convergence of Artificial Intelligence, the Internet of Things, and Robotics is no longer a futuristic vision; it is rapidly becoming the foundation of real-time, intelligent, and context-aware systems. AI enables perception and reasoning, IoT provides scalable sensing and communication, and robotics delivers embodied actuation. Despite significant progress in pairwise combinations such as AIoT and the Internet of Robotic Things (IoRT), there remains a lack of unified design frameworks that fully integrate all three. This survey synthesizes the state-of-the-art across these domains, emphasizing the emerging role of Small Language Models (SLMs) at the edge and Large Language Models (LLMs) in the cloud for distributed cognition and autonomous decision-making. We propose a modular system architecture that aligns with these trends, analyze persistent gaps in interoperability and feedback control, and classify existing work by integration depth. Our review highlights how hybrid SLM-LLM systems, when coupled with IoT infrastructure and robotic agents, can address challenges in real-time adaptation, scalability, and reliability. This work offers a conceptual and technical roadmap for designing next-generation AI-IoT-Robotic ecosystems that are modular, interpretable, and capable of learning within dynamic environments, paving the way for the emerging paradigm of Connected Robotics and Physical AI. 

---
# Cross-Axis Feature Fusion with Joint-Wise Motion Difference Prediction for Text-Based 3D Human Motion Editing 

**Authors**: Gyojin Han, Junmo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2606.01014)  

**Abstract**: We address text-based 3D human motion editing, where the goal is to preserve the style and structure of a source motion while applying edits described in natural language. The release of the MotionFix dataset has spurred active research into training-based diffusion models that directly generate an edited motion from a source motion and a text instruction. While previous works have focused primarily on learning when an edit should occur temporally, our goal is to create a model that understands not only this temporal aspect but also which specific joints are responsible for the change. Targeting this, we propose a novel architecture and a complementary auxiliary task to aid its training. Our architecture consists of two axis-anchored transformers, which extract distinct features along the joint and time dimensions respectively, and a cross-axis fusion block that integrates these representations. We further introduce an auxiliary task that trains the joint-anchored transformer to regress the Soft-DTW distance between source and target joint rotations. This objective teaches the module to understand which joints to modify and which to preserve. Through comprehensive experiments on the MotionFix dataset, we demonstrate that our method significantly improves semantic alignment with both the text instruction and the source motion, as well as the overall fidelity of the generated motion, achieving state-of-the-art results. 

---
# FVSpec: Real-World Property-Based Tests as Lean Challenges 

**Authors**: Quinn Dougherty, Max von Hippel, Hazel Shackleton, Mike Dodds  

**Link**: [PDF](https://arxiv.org/pdf/2606.01008)  

**Abstract**: We present a benchmark for evaluating AI models and agents on real-world formal software verification tasks. We first scrape 11,039 property-based tests (PBTs) from real-world Python repositories, then automatically translate 2,772 of them (25%) into 9,415 Lean 4 specifications with sorry placeholders (about 3 formalizations/PBT; we retain multiple attempts when none dominates on quality metrics). Translating PBTs into Lean specifications is challenging: it requires modeling Python semantics in Lean, inferring the logical property encoded in an imperative PBT, and handling the inherent difficulties of dependently-typed programming in a seldom-used language. We describe a three-agent LLM pipeline for transpiling PBTs into Lean specifications, evaluate coverage and quality metrics, and provide baselines for proof generation using several automated and model based approaches. All code (scraper and agents) and data (PBTs and Lean specifications) are open source. Our benchmark aims to drive progress on the underexplored problem of AI-assisted formal verification of real-world software, which is of increasing interest as AI produces more and more of the world's code. 

---
# Beyond Task-Agnostic: Task-Aware Grouping for Communication-Efficient Multi-Task MoE Inference 

**Authors**: Zhiyao Xu, Aoxue Liu, Zhanjie Ding, Dan Zhao, Yong Jiang, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.01007)  

**Abstract**: Sparsely activated Mixture-of-Experts (MoE) models scale capacity via conditional computation, but distributed inference suffers from cross-GPU expert communication and routing-induced load imbalance. Existing placement methods reduce this cost by co-locating frequently co-activated experts; however, they derive a single deployment plan from globally aggregated routing traces, thereby averaging away the heterogeneous, task-specific co-activation patterns that actually drive communication in multi-task serving. We observe that expert co-activation is strongly task-conditioned: pairs tightly coupled in one task family are often uncorrelated in another, so effective deployment should group experts by task-aware co-activation rather than by a task-agnostic average. Based on this insight, we propose \emph{Task-Aware Coactivation Grouping} (TACG), a deployment-time framework that uses family-specific dispatch and co-activation traces to derive per-expert task-family preferences, reweights the co-activation graph so that intra-family locality dominates grouping, and assigns each expert to a primary GPU under exact capacity constraints. To keep the static placement robust under online workload skew, we further introduce \emph{Generic Expert Shared Replication} (GESR), a lightweight companion that identifies generic experts with consistently central co-activation profiles, replicates them across a small set of secondary GPUs, and applies locality- and load-aware selection at serving time. Experiments on three representative open-source MoE models demonstrate that our framework reduces the average communication cost by 31.39\% over the baseline, while preserving an average Jain fairness index of 0.9975. This advantage persists even under severe distribution shifts in the inference data, consistently outperforming strong baselines. 

---
# An Open-Source Benchmark and Baseline for Multi-temporal Referring Segmentation 

**Authors**: Bingyu Li, Da Zhang, Tao Huo, Zhiyuan Zhao, Junyu Gao, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.00987)  

**Abstract**: Large Vision-Language Models (LVLMs) have shown strong visual understanding and language-guided grounding abilities, yet their capacity for multi-temporal visual reasoning remains underexplored. To bridge this gap, we introduce \textbf{Multi-temporal Referring Segmentation (MTRS)}, a new task that aims to segment language-described temporal changes from multi-temporal images. MTRS extends conventional referring segmentation and change detection by jointly requiring temporal correspondence reasoning, language grounding, and pixel-level mask prediction. We propose \textbf{CRAFT-Agent}, an automated data construction pipeline with human auditing, and build \textbf{MTRefSeg-21K}, the first MTRS benchmark, containing 21K high-quality multi-temporal image-text-mask triplets across diverse scenes, viewpoints, and domains. Benchmarking a broad set of VLM- and LVLM-based models reveals that direct inference performs poorly, while task-specific fine-tuning remains limited. To address this, we propose \textbf{MTRefSeg-R1}, a change-aware LVLM framework trained with a two-stage strategy. It first learns general temporal-change perception from 20K vision-only bi-temporal samples, and is then fine-tuned on MTRefSeg-21K for fine-grained language-guided temporal localization. MTRefSeg-R1 explicitly models cross-temporal visual differences, aligns language instructions with temporal variations, and predicts referred change masks. Extensive experiments show that MTRefSeg-R1 achieves strong and often superior performance compared with existing LVLM baselines, demonstrating the challenge and potential of MTRS. 

---
# SS-ZKR: Spatial-Semantic Zero-Knowledge Routing for Privacy-Preserving Multi-Agent Collaboration 

**Authors**: Hassan Touheed  

**Link**: [PDF](https://arxiv.org/pdf/2606.00962)  

**Abstract**: Foundational agent interoperability standards, notably the Agent-to-Agent (A2A) protocol and the Model Context Protocol (MCP), have advanced multi-agent system communication, and complementary identity frameworks leveraging W3C Decentralised Identifiers (DIDs) and Verifiable Credentials (VCs) provide cryptographic agent authentication. However, no existing protocol supports content-based semantic routing of agent payloads across organisational trust boundaries without requiring the routing intermediary to decrypt the payload, which is a hard constraint in compliance-sensitive environments governed by GDPR, HIPAA, and MiFID II. We propose SS-ZKR, a three-mechanism privacy-preserving routing protocol designed as a complementary layer atop A2A/MCP. Mechanism I introduces blind routing via differentially private semantic intent vectors cryptographically bound to zero-knowledge proofs of payload-schema consistency. Mechanism II offers vector-weighted adaptive payload sanitisation with formal (epsilon, delta)-differential privacy for numerical fields and heuristic semantic aggregation for textual fields. Mechanism III presents a spatial-to-cryptographic policy compiler that translates visually defined trust-zone topologies into deterministic zero-knowledge access circuits. We provide a formal threat model, analyse information leakage bounds of intent vectors, present pseudocode for all three mechanisms, and give analytical complexity comparisons against TEE-based and homomorphic encryption-based routing baselines. SS-ZKR lets enterprises in financial services, healthcare, and defence orchestrate heterogeneous AI agents across regulatory boundaries without exposing proprietary data to routing infrastructure. 

---
# Explainable deep reinforcement learning reveals energy-efficient control strategies for turbulent drag reduction 

**Authors**: Federica Tonti, Ricardo Vinuesa  

**Link**: [PDF](https://arxiv.org/pdf/2606.00949)  

**Abstract**: We propose a method combining Multi-Agent Deep Reinforcement Learning (MARL) and eXplainable Deep Learning (XDL) to reduce drag in wall-bounded turbulent flows. Taking as a baseline the results of training agents directly targeting wall-shear stress and opposition control, three SHAP-guided approaches are compared. In the first, the reward is computed from SHAP attributions of a U-net predicting the future velocity field; in the second, from SHAP attributions of a U-net predicting the skin-friction coefficient; in the third, from a combination of SHAP attributions of two U-nets predicting the skin-friction coefficient and the wall pressure fluctuations, respectively. The combined SHAP strategy based on skin-friction coefficient and wall-pressure fluctuations achieves the best overall performance, achieving a DR of 34.44% and a NES of 34.01% with only 0.43% normalized input power. Relative to opposition control, drag reduction and net energy saving increase by 49.41% and 48.52%, respectively. Compared with the direct wall-shear-stress baseline, the proposed strategy simultaneously improves performance while reducing the normalized actuation cost from 5.90% to 0.43%. Analysis of the results reveals that the energetically efficient policy is consistent with pressure-gated actuation, activating predominantly at near-zero wall pressure, and operates on a temporal timescale comparable to the lifetime of the near-wall turbulent structures. 

---
# Silent Failures in Federated Personalization of Foundation Models 

**Authors**: YongKyung Oh, Alex Bui  

**Link**: [PDF](https://arxiv.org/pdf/2606.00947)  

**Abstract**: Foundation models are increasingly personalized on decentralized private data through federated learning and are now deployed at scale under growing regulatory requirements for post-market monitoring. We argue that this convergence creates a distinct and under-recognized class of trustworthiness failures, which we term "Silent Failures." These include amplified bias, fairness collapse, and alignment erosion that may remain difficult to detect because federated learning's privacy constraints limit visibility into model behavior. A landscape analysis of existing benchmarks reveals a structural divide. Federated benchmarks evaluate system performance but provide limited insight into model behavior, whereas centralized trustworthiness benchmarks assess behavior but require model access incompatible with federated privacy. We introduce a taxonomy of six silent failure modes arising from the interaction of foundation model personalization, dataset shift, and core federated constraints. Our analysis shows that privacy-preserving training alone is insufficient for trustworthy deployment. We conclude with a research agenda for privacy-preserving behavioral evaluation and propose that silent failures become a standard diagnostic category for trustworthy federated artificial intelligence. 

---
# Lodestar: An Online-Learning LLM Inference Router 

**Authors**: Gangmuk Lim, Wanyu Zhao, Brighten Godfrey, Jiaxin Shan, Le Xu, Liguang Xie  

**Link**: [PDF](https://arxiv.org/pdf/2606.00946)  

**Abstract**: Efficiently serving large language model (LLM) inference tasks is crucial both for user-perceived latency such as time-to-first-token (TTFT) and for GPU utilization. However, LLM request routing, that is, assigning each inference request to a GPU instance, is particularly challenging: execution is highly input-dependent; batching and KV-cache reuse create strong cross-request coupling; and latency responds nonlinearly to context length, model/engine settings, and heterogeneous accelerators. As a result, simple traditional load balancing algorithms, and even heuristics tailored for LLM inference, fail to achieve good performance. We present Lodestar, a novel learning-based request routing system for distributed GPU clusters. Lodestar continuously collects a snapshot of the cluster at per-request level, including real-time instance state, request characteristics, and observed performance, and trains an online reward predictor that it uses to route inference requests to the instance that will maximize given reward (e.g., minimizing TTFT). Lodestar is cloud-native and works seamlessly with existing serving stacks (vLLM). With continuous online adaptation to changing workloads and infrastructure conditions, Lodestar achieves 1.41x lower average TTFT and 1.47x lower P99 TTFT on average (up to 2.15x/1.86x on homogeneous and 4.38x/4.42x on heterogeneous clusters) compared to a state-of-the-art prefix cache and load-aware heuristic, and learns these efficient routing strategies within about 5 minutes, based on experiments in a public cloud GPU cluster. 

---
# CV-Arena: An Open Benchmark for Instructional Computer Vision Problem Solving with Human-AI Collaborative Preferences 

**Authors**: Fangzhou Lin, Peiran Li, Lingyu Xu, Wenjing Chen, Qianwen Ge, Shuo Xing, Mingyang Wu, Xiangbo Gao, Siyuan Yang, Kazunori Yamada, Ziming Zhang, Haichong Zhang, Zhen Dong, Ming-Hsuan Yang, Zhengzhong Tu  

**Link**: [PDF](https://arxiv.org/pdf/2606.00931)  

**Abstract**: Instruction-guided image editing is becoming a general interface for visual work, yet existing benchmarks still focus largely on narrow appearance edits and do not fully capture the diversity of real-image tasks in professional workflows. Here, we define instructional computer vision problem solving as a broader formulation of image editing: given a real input image and a natural-language instruction, a system must produce an edited output that realizes the requested transformation while satisfying explicit preservation, geometric, physical, and usability constraints. We introduce CV-Arena, an open benchmark designed to evaluate this capability at professional scales. CV-Arena contains 12K high-resolution real-image instruction pairs spanning 16 instruction-based visual task types, constructed using CogRetriever, a dual-track retrieval-and-curation pipeline that combines targeted web search, agentic query refinement, verification, and traceability. To evaluate models at scale while preserving human fidelity, we propose Active Elo, a human-AI collaborative preference protocol that leverages CV-Judge, a logic-gated, multi-dimensional VLM evaluator, to reject clear failures and resolve high-confidence comparisons; and to route close, high-quality comparisons to expert raters. Mixed human and AI supervision is then aggregated through reliability-weighted Elo updates. Our comprehensive evaluation of 21 systems, including proprietary, open-source, and agentic models, on CV-Arena reveals persistent gaps in instruction adherence, physical reasoning, structural control, and fine-grained detail preservation. We further develop CV-Agent, a lightweight agentic model that combines planning, editing, and verification, and demonstrate that closed-loop reasoning is a promising direction for professional-grade instruction-following visual editing. 

---
# Detection vs. Execution: Single-Bucket Probes Miss Half the Mamba-2 State Sink 

**Authors**: Yuhang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2606.00930)  

**Abstract**: Mechanistic interpretability often assumes that probes identifying a representational signature also identify the circuit executing the corresponding computation. We show that this assumption can fail systematically in Mamba-2. Studying the state sink (disproportionate Delta-gate activation on boundary tokens, analogous to the attention sink), we find that single-bucket probes recover only a small execution layer while missing a much larger detection layer with the same representational signature.
In Mamba-2, the state sink decomposes into two functional head sets. Single-bucket BOS-specialist heads (about 5% of heads at 2.7B) causally support both BOS-context and newline-target predictions across model scales and corpora. Dual heads (27-35% of heads, recovered by multi-class aggregation of the same probe) show stronger BOS-newline representational similarity but substantially weaker causal effects under ablation. Representational similarity does not imply functional equivalence.
This distinction matters for downstream behaviour: ablating BOS-specialist heads collapses RULER NIAH retrieval accuracy from 1.00 to 0.00 at 1024 context length in both Mamba-1 2.8B and Mamba-2 2.7B, while size-matched complements preserve baseline performance. A random channel-bucketing control rules out substrate granularity alone, implicating Mamba-2's head-shared Delta projection. Probe-derived specialty can identify execution circuits; at coarse granularity the same probe also recovers detection circuits, and separating them requires class-conditional ablation rather than class-conditional cosine. 

---
# Benchmarking Security Risk Detection and Verification in Open Agentic Skill Ecosystems 

**Authors**: Ismail Hossain, Sai Puppala, Zhuoran Lu, Sajedul Talukder, Nan Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2606.00925)  

**Abstract**: Open agent platforms allow community contributors to publish reusable skills that agents can invoke at runtime. This extensibility also creates a supply-chain risk: malicious contributors can hide harmful behavior inside skills that appear benign under superficial inspection. However, existing defenses are hard to evaluate because there is no benchmark that measures both malicious-skill detection and runtime verification. We present SkillVetBench, a two-stage security vetting benchmark for open agentic skill ecosystems. The first stage performs semantic vetting over each skill's natural-language specification to detect hidden malicious intent. The second stage executes flagged skills in an instrumented sandbox to observe runtime behavior and collect auditable evidence. We build a benchmark from confirmed malicious skills in the live OpenClaw ecosystem, including samples from the recent ClawHavoc supplychain campaign. Unlike static-only methods, SkillVetBench verifies detected threats with execution traces. Our experiments show that: (1) semantic-only and signature-based baselines are insufficient, missing up to 89\% of malicious skills whose threats arise from natural-language instructions, multicomponent logic, or cross-component interactions; (2) runtime attacks are concentrated in a small set of high-permission primitives, especially exec, write\_file, install\_skill, and spawn; and (3) SkillVetBench provides case studies in which sandbox execution directly supports malicious verdicts with concrete runtime evidence. 

---
# Accuracy, Stability, and Repeated-Run Reliability of Large Language Models on Deterministic Programming Tasks 

**Authors**: Yongxi Zhou, Lai Yun Choi, Jiaxi Wen, Wenbo Ye  

**Link**: [PDF](https://arxiv.org/pdf/2606.00920)  

**Abstract**: Run-level pass rate overstates retry-free coverage by up to 17.8 percentage points -- and the gap is largest precisely for mid-performing systems. We investigate this accuracy--stability relationship in large language model (LLM) evaluation for deterministic text-conditioned generation, using programming tasks as a concrete testbed. Standard code-generation benchmarks emphasize single-run accuracy or eventual success under repeated sampling, but many deployment settings also require stability: consistent outcomes across repeated invocations under the same task description. We present a repeated-run evaluation protocol with metrics for run-level accuracy, retry-free coverage, and per-problem variability. On a recency-based benchmark of 100 LeetCode-style problems, we evaluate 16 models from five provider families under two prompt templates with five repeated runs per problem, yielding 16,000 evaluation instances. Although run-level pass rate and perfect stability rate are strongly correlated (r=0.985), pass rate consistently exceeds retry-free coverage -- a gap that reaches 17.8 percentage points and reverses model rankings even among closely matched systems. Prompt effects are model-dependent rather than uniformly beneficial. These results suggest that repeated-run stability analysis is a necessary complement to conventional accuracy reporting for deterministic text-conditioned generation tasks. 

---
# MLLM-Microscope: Unlocking Hidden Structure Within Multimodal Large Language Models 

**Authors**: Ravil Mussabayev, Rustam Mussabayev  

**Link**: [PDF](https://arxiv.org/pdf/2606.00909)  

**Abstract**: This work presents MLLM-Microscope, a novel system designed for analyzing the hidden representations within Multimodal Large Language Models (MLLMs). Our system evaluates the linearity, intrinsic dimension, and anisotropy of multimodal token embeddings across transformer layers. Utilizing the ScienceQA dataset, we evaluate two state-of-the-art MLLMs, LLaVA-NeXT and OmniFusion. We find that both the main and residual streams for tokens of both modalities exhibit highly linear behaviors across transformer layers. However, LLaVA-NeXT's image tokens reveal a slight decline in linearity, whereas OmniFusion's remain consistent. Image token dimensions in OmniFusion remain consistently higher across layers compared to LLaVA-NeXT. Also, the OmniFusion's anisotropy is observed to stay consistently low throughout the layers. These findings suggest that the inner workings of MLLMs highly depend on the nature of modality fusion performed before passing the token sequence into LLM. This and other new potential insights obtainable from our system are surely capable of enhancing our understanding of the inner workings of MLLMs, informing future model design and optimization. 

---
# Memory-Efficient LLM Training with Dynamic Sparsity: From Stability to Practical Scaling 

**Authors**: Qiao Xiao, Boqian Wu, Patrik Okanovic, Tomasz Sternal, Maurice van Keulen, Elena Mocanu, Mykola Pechenizkiy, Decebal Constantin Mocanu, Torsten Hoefler  

**Link**: [PDF](https://arxiv.org/pdf/2606.00888)  

**Abstract**: Dynamic Sparse Training (DST) offers a promising paradigm for improving the training and inference efficiency of deep neural networks; however, we find that in large language model training, DST can suffer from optimization instability, manifested as loss spikes after topology updates. In this work, we show that the naive use of standard Adam-based optimizers leads to a cold-start issue for newly regrown parameters, resulting in excessively large updates and disrupted training dynamics. To address this issue, we propose Sparse Memory-Efficient Training (SMET), which stabilizes DST with optimizer warm-up and improves training progress through density-aware learning-rate scaling. SMET further reduces memory consumption by storing gradients and optimizer states only for active parameters. We provide a theoretical analysis of the update behaviors under SMET, showing improved optimization stability. Extensive experiments demonstrate that SMET enables stable, scalable, and memory-efficient sparse pre-training of LLMs, paving the way for sparse training as a practical alternative to dense training. Our code is publicly available at: this https URL. 

---
# Dive into Waves: Morlet Spectral Transformer for Cross-Subject Emotion Decoding from EEG 

**Authors**: Jiaxin Qing, Lexin Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.00884)  

**Abstract**: We study cross-subject emotion recognition from EEG, a practically important yet challenging problem in brain-computer interfaces. Unlike tasks with clear waveform signatures, emotion-related EEG signals are primarily encoded in spectral power and are weak, noisy, and highly variable across subjects. Existing approaches rely either on large pretrained EEG foundation models, which require massive data yet still struggle with cross-subject variability, or frequency-domain encoders, which better reflect spectral structure but suffer from mismatched representations, drift-dominated tokenization, and lack of band-specific spatial modeling. In this article, we propose the Morlet Spectral Transformer (MST), built around three key components and integrated with a spatiotemporal Transformer backbone. First, Morlet wavelet tokenization provides a time-frequency representation that matches the multi-scale structure of brain rhythms, and extends classical differential entropy features to a form suitable for Transformers. Second, long-context baseline removal acts as a simple temporal normalization that removes subject-specific drift and redundancy across nearby windows. Third, frequency-specific spatial projection learns a separate channel mixer for each frequency band, capturing interpretable band-specific patterns and reducing cross-channel mixing. We show that, even without pretraining, MST consistently outperforms both large pretrained EEG foundation models and frequency-based methods across all SEED-family datasets. These results suggest that careful representation design can yield an accurate, cost-effective, and interpretable alternative to large-scale pretraining. 

---
# Task diversity produces systematic transfer but inhibits continual reinforcement learning 

**Authors**: Purab Seth, Neil Shah, Kunal Jha, Samuel J. Gershman, Max Kleiman-Weiner, Wilka Carvalho  

**Link**: [PDF](https://arxiv.org/pdf/2606.00880)  

**Abstract**: Continual reinforcement learning aims to produce agents that learn not only to improve at their current tasks but also to adapt as task distributions change. Training an agent on many diverse tasks can induce zero-shot generalization, but previous work generally evaluates this generalization after training -- with frozen weights. Whether task diversity also improves an agent's ability to continue learning across distribution shifts remains unclear. We introduce Banyan, a GPU-accelerated continual RL domain in which task diversity factors into three independently controllable axes: the map layouts an agent must navigate, the objects it must interact with, and the hierarchical structures of sub-goal dependencies. Across individual distribution shifts, increasing diversity along each axis causes agents to begin training on the new tasks near the performance attained on the previous one, even when the shift changes the structure of the optimal policy. However, as the number of shifts increases, this local transfer does not by itself yield sustained continual learning: longer-horizon tasks plateau, and earlier task distributions are forgotten after later training. Banyan is a benchmark for studying when controlled task diversity produces transferable learning, when that transfer persists, and where it falls short of proper continual learning. 

---
# Benchmarks for Vision-Language Models in Urban Perception Should Be Reliability-Aware and Negotiated 

**Authors**: Rashid Mushkani  

**Link**: [PDF](https://arxiv.org/pdf/2606.00871)  

**Abstract**: Vision-language models (VLMs) are increasingly used to generate structured descriptions of street-level imagery for tasks such as streetscape auditing, mapping, and public consultation. These uses combine observable attributes with appraisal categories, and the human targets are often distributions of judgments with disagreement and explicit non-response. This paper argues that benchmarking VLMs for urban perception should treat disagreement and abstention as measurement outcomes, report inter-annotator reliability alongside model alignment, and treat the label space and scoring policy as negotiable artifacts when outputs are intended to inform urban governance. We ground the argument in a benchmark of 100 Montreal street scenes annotated along 30 dimensions by 12 participants from seven community organizations, and in a deterministic zero-shot evaluation of seven VLMs. Across dimensions, model agreement with human consensus co-varies with dimension-level human reliability, and for the appraisal dimension Overall Impression models and annotators exhibit distributional mismatch including different rates of Not applicable. We close with actions for benchmark creators, model developers, and institutions to make uncertainty and benchmark assumptions visible in evaluation reports. 

---
# GenPT: Beyond Self-Report for Reliable LLM Psychometrics via Generative Projective Testing 

**Authors**: Ming Wang, Shuang Wu, Bixuan Wang, Lu Lin, Yuxin Chen, Xiaocui Yang, Daling Wang, Shi Feng, Yifei Zhang, Yufan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2606.00860)  

**Abstract**: Self-report questionnaires remain the prevailing tool for probing the psychological states of persona-conditioned agents (PC-Agents). However, classical instruments inherit two well-known threats: contamination from training corpora and directional bias driven by social-desirability or contextual framing. To overcome these methodological bottlenecks, we ask whether projective paradigms can be adapted into a robust psychometric tool. We introduce \textbf{GenPT} (Generative Projective Testing), which reformulates TAT, Rorschach, and SCT with newly generated stimuli and organizes assessment as a three-stage pipeline to derive standardized psychological indicators and target states. Evaluating PC-Agents induced via CharacterRAG and AnnaAgent profiles, we benchmark GenPT's reliability and validity against classical questionnaires. The results indicate that questionnaires exhibit systematic directional shifts under social-desirability framing, most strongly on suicide ideation. In contrast, GenPT's collected behavioral patterns stay near the symmetric baseline. Furthermore, under a longitudinal counselling context, GenPT-based depression assessment shifts by roughly an order of magnitude more than the questionnaire counterpart when Qwen3 serves as the backbone. Overall, GenPT complements self-report methods in scenarios where contamination resistance, bias asymmetry, and context sensitivity matter. Code and stimuli can be found at this https URL. 

---
# From Cues to Horizons: Dynamic Risk Horizon Profiling for Trajectory Prediction 

**Authors**: Xinyi Ning, Zilin Bian, Dachuan Zuo, Semiha Ergan, Kaan Ozbay  

**Link**: [PDF](https://arxiv.org/pdf/2606.00857)  

**Abstract**: Accurate and reliable vehicle trajectory prediction is essential for safe autonomous driving. Recent studies have incorporated safety risk into trajectory prediction to quantify dangers posed by surrounding agents. However, most risk-aware approaches use past risk information as a secondary signal to help guide decisions, overlooking its future evolution and uncertainty. In this paper, we propose a risk horizon profiling (RHP) module that incorporates a continuous, learnable potential field model for risk-aware trajectory prediction. The RHP module calculates the spatial-temporal proximity of surrounding objects to profile risk distributions across future horizons, which supports better trajectory prediction by adaptively identifying what human drivers perceive as critical moments. We evaluate our method on two datasets from different driving settings, highD for highway corridors and SHRP2 for urban streets, which cover diverse risk scenarios including safe, near-crash, and crash events. Compared to the baseline methods, our framework achieves a 25.0\% reduction in 5s RMSE on the highD dataset and a 29.1\% reduction in 5s minFDE on SHRP2. These results indicate strong performance for both short and long horizon prediction and robust generalization across highway and urban scenarios. The proposed method enables more realistic AV path planning and strategic selection, thereby supporting safer autonomous driving and more advanced driver-assistance systems. The source code for this work is available at: this https URL 

---
# RefDiffNet: Learning to Expose Subtle PCB Defects Before Detection 

**Authors**: Vinay Edula, Nilesh Badwe, Priyanka Bagade  

**Link**: [PDF](https://arxiv.org/pdf/2606.00852)  

**Abstract**: Printed circuit board (PCB) defect detection is challenging because many defects are small and difficult to distinguish from complex background patterns. Most deep learning-based PCB inspection methods rely only on the inspected PCB image for defect detection, ignoring the defect-free reference image that encodes the expected layout of traces, pads, and other PCB structures. In this work, we propose RefDiffNet, a lightweight plug-and-play input enhancement block placed before the detector backbone to enhance the image before defect detection. RefDiffNet brings one proven idea from classical inspection into the deep learning era, using a defect-free reference image to reveal defects. RefDiffNet compares the defective image with the aligned reference, captures structural changes relative to the reference, and uses a lightweight encoder to output the original image with defective regions highlighted, thereby making the downstream detector's task easier. Results on HRIPCB and DeepPCB show that RefDiffNet consistently improves performance across detector families, including one-stage detectors from YOLOv8 to YOLOv26, the transformer-based RT-DETR, and the two-stage Faster R-CNN. It achieves up to 18% relative mAP50:95 gain with negligible overhead, introducing only 0.004 - 0.005M additional parameters and 0.7 - 0.8 GFLOPs, amounting to at most 0.25% of the parameter count of any evaluated detector. Results establish RefDiffNet as a lightweight, plug-and-play, detector-agnostic input enhancement module that substantially improves PCB defect detection with minimal computational cost. 

---
# MoEIoU: Rethinking Bounding-Box Regression as a Mixture of Experts 

**Authors**: Vinay Edula, Priyanka Bagade  

**Link**: [PDF](https://arxiv.org/pdf/2606.00844)  

**Abstract**: Bounding-box regression is a fundamental component of object detection, playing a critical role in precise object localization. Existing Intersection-over-Union (IoU)-based loss functions extend the IoU objective by incorporating geometric penalties, such as center-distance and aspect-ratio mismatch, to improve bounding-box regression. However, these penalties typically remain fixed throughout training and do not account for the optimization dynamics in which predicted boxes initially exhibit large center-distance and shape errors, with later stages focusing on improving overlap with the ground truth. To address this limitation, we introduce MoEIoU, a mixture-of-experts based regression loss that jointly models overlap, center alignment, and aspect-ratio mismatch. MoEIoU aggregates these components using a log-sum-exp function, which emphasizes the dominant localization error while maintaining smooth contributions from other terms. Additionally, a curriculum-based weighting schedule is employed to prioritize correcting box position and shape in early training stages and improving overlap in later stages. We evaluated proposed MoEIoU on PASCAL VOC, HRIPCB, and MS COCO using multiple YOLO architectures, along with large-scale simulation experiments. It consistently outperforms standard and recent state-of-the-art losses, demonstrating faster convergence and improved localization accuracy. We further show that this adaptive aggregation improves existing IoU-based losses, yielding consistent gains and providing more effective optimization guidance for bounding-box regression in object detection frameworks. 

---
# Hybrid Probabilistic Forecasting of Under-Five Malaria Admissions in Ghana: A Gaussian Process Regression with Holt-Winters Smoothing 

**Authors**: T. Ansah-Narh, Y. Asare Afrane, J. Bremang Tandoh  

**Link**: [PDF](https://arxiv.org/pdf/2606.00834)  

**Abstract**: Accurate malaria forecasting remains a major challenge in sub-Saharan Africa, where strong seasonality, reporting uncertainty, and non-stationary transmission dynamics reduce the reliability of conventional models. In Ghana, district-level malaria surveillance requires forecasting frameworks that are probabilistically rigorous and robust under limited data. This study proposes a hybrid framework integrating Gaussian Process Regression (GPR) with Holt-Winters exponential smoothing for modelling monthly under-five malaria admissions. GPR captures non-linear behaviour and predictive uncertainty, while Holt-Winters stabilises long-horizon forecasts and preserves seasonal structure. Using ten years of district-level data (2014-2023), performance was evaluated via rolling-origin expanding-window validation. The hybrid model achieved $R^2 = 0.9906$ versus $0.8213$ for Holt-Winters alone, with $94.2\%$ of residuals within $\pm 2\sigma$ bounds. Forecasts for 2024-2028 project average monthly admissions from approximately 8{,}000 to 12{,}200 cases. Spatio-temporal analysis revealed pronounced ecological heterogeneity: northern high-burden districts exhibited stable relative patterns despite large absolute fluctuations. The framework provides a scalable probabilistic approach for malaria early warning and operational planning in endemic settings, supporting Ghana's national malaria control strategy. 

---
# Beyond Independent Manipulation: Individual Fairness-aware Strategic Classification with Peer Imitation 

**Authors**: Xinpeng Lv, Chunyuan Zheng, Yunxin Mao, Renzhe Xu, Jinxuan Yang, Yuanlong Chen, Wangrong Huang, Shaowu Yang, Wenjing Yang, Xinwang Liu, Peng Cui, Haotian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.00827)  

**Abstract**: Strategic classification (SC) investigates scenarios where agents manipulate their features to obtain favorable decisions from predictive models. Existing fairness-aware SC approaches primarily focus on group fairness and typically assume that agents respond independently. However, when individual fairness is required, ensuring similar individuals receive similar outcomes, agents' manipulation becomes interdependent: an agent's preferred manipulation depends on the neighborhoods' outcomes. This induces a mismatch between classical SC formulations and fairness-aware decision settings, where independent models no longer accurately characterize strategic manipulations. To address this issue, we introduce individual fairness-aware strategic classification (IFSC), a framework that models peer-driven manipulation arising from individual fairness, where agents imitate nearby positively decided peers to obtain favorable outcomes. IFSC characterizes strategic manipulation as similarity-based imitation toward visible accepted peers and learns classifiers under the resulting post-manipulation distributions. To account for uncertainty in peer observability, IFSC employs a robust learning process that introduces stochastic perturbations during manipulation simulation. Experiments on synthetic and real-world datasets demonstrate that IFSC improves individual-fairness consistency and mitigates imitation-induced distortions. 

---
# SkillPager: Query-Adaptive Intra-Skill Navigation via Semantic Node Retrieval 

**Authors**: Zicai Cui, Zihan Guo, Weiwen Liu, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.00822)  

**Abstract**: Skill-based LLM agents increasingly rely on long procedural documents, but full-document prompting wastes tokens and dilutes information critical to execution. We study this setting as intra-skill retrieval, where the goal is to select a minimal, execution-sufficient context from a known skill document given a query. We present SkillPager, a two-stage framework that parses each Markdown skill into typed semantic nodes offline and leverages Maximal Marginal Relevance (MMR) to perform global, query-conditioned node selection online. On a benchmark of 395 skills and 1,975 queries, SkillPager achieves 78.89% LLM-judged context sufficiency, compared to 82.23% for the exhaustive full-document baseline, while reducing prompt tokens by 47.04%. A granularity ablation shows that applying the same retrieval algorithm to raw fixed-length chunks reaches a comparable 81.77% sufficiency but increases token cost by 28.81%, demonstrating that efficiency gains are driven by typed semantic granularity rather than the retrieval algorithm alone. Among graph-based baselines, SkillPager outperforms the strongest baseline by a margin of 12.16%. Further ablations show that supporting content is most effective when retained in the candidate pool and selected adaptively rather than removed by static heuristics. These results identify typed intra-document retrieval as a distinct access problem for skill-based agents. 

---
# Certificates without Electrons? Theory and Evidence on Impacts from AI-Driven Power Demand 

**Authors**: Dana Golden, Aruna Balasubramanian, Niranjan Balasubramanian  

**Link**: [PDF](https://arxiv.org/pdf/2606.00811)  

**Abstract**: Data centers now account for 4.4% of United States electricity demand, yet the grid-level effectiveness of the renewable energy certificates (RECs) and power purchase agreements (PPAs) hyperscalers use to claim carbon neutrality remains unclear. We develop a game-theoretic model in which a data center operator chooses among RECs, PPAs, and behind-the-meter colocation while generators make entry decisions under endogenous financing costs. The model identifies a timing wedge -- the mismatch between consumption and credited renewable generation -- as a central mechanism through which AI demand degrades reliability, raises prices, and increases emissions even when RECs cover 100% of annual consumption. Colocation with storage addresses this wedge directly and induces the greatest renewable entry by eliminating generator revenue risk. We test these predictions by exploiting the staggered release of large language models as a natural experiment, using difference-in-differences on a novel dataset linking AI activity to local grid outcomes. AI demand significantly increases fossil generation, wholesale prices (up to 25% in treated PJM zones), and outage frequency (0.5--1 additional outages per year) near data centers, with impacts scaling in model size. Data centers with on-site generation exhibit a sign reversal in power-quality effects, consistent with the model's prediction that behind-the-meter capacity absorbs demand spikes. Counterfactual analyses show that edge inference, spatial reallocation, and colocated storage each substantially mitigate grid impacts, while REC-only strategies do not. Together, our results demonstrate that the externalities of AI to the grid are tightly coupled to procurement design and the spatial organization of data center infrastructure. 

---
# Dynamic Coordination Strategy Selection for Enterprise Multi-Agent Systems 

**Authors**: Thanh Luong Tuan  

**Link**: [PDF](https://arxiv.org/pdf/2606.00804)  

**Abstract**: Enterprise multi-agent systems increasingly expose multiple coordination patterns, but deployments often lack evidence for when to use consensus, debate, synthesis, or a simpler single-agent workflow. This paper evaluates whether coordination strategy should be selected dynamically by problem class rather than fixed globally. We run a frozen matrix of 30 enterprise tasks spanning six industries, five problem classes, four execution conditions, three replications per cell, and four model arms: qwen_local, sonnet, gemma_openrouter, and an auxiliary openai cloud-validation arm. All 1,440 generated outputs are judged by a fixed Sonnet rubric.
The main finding is bounded and operationally useful, but it is not the original strict H1. The pre-registered exact-winner/CI criterion is not supported: exact winner identity is unstable across model arms, and several predicted strategies are close to, but not above, the best observed alternative. A weaker near-best routing claim is strongly supported. In every pre-registered model arm and problem class, and again in the auxiliary OpenAI validation arm, the predicted strategy is within 0.10 quality-score points of the best observed condition. Structured compliance verification is the clearest exception to the original mapping: all arms favor single_agent rather than consensus. A pre-registered Kendall's W test finds no reliable difference between Vietnamese-domain and English-domain tasks in how consistently the four coordination conditions are ranked (mean W of 0.20 in both strata; signed-rank p = .85), so H2 is not supported. We conclude that enterprise coordination policy should use dynamic routing as a calibrated default, not as a deterministic winner-selection law. 

---
# DASH: Dual-Branch Score Distillation for Guidance-Calibrated Compact Diffusion Models 

**Authors**: Abdullah Al Shafi, Kazi Saeed Alam, Sk Imran Hossain, Engelbert Mephu Nguifo  

**Link**: [PDF](https://arxiv.org/pdf/2606.00798)  

**Abstract**: Parameter compression of class-conditional diffusion models reveals an underexplored limitation in output-level distillation: the unconditional score branch remains unsupervised, leaving the classifier-free guidance gap underdetermined in the student. This gap, amplified at every denoising step, admits degenerate solutions where both branches collapse toward identical predictions, rendering guidance ineffective despite low output-level training loss. This paper introduces DASH, a dual-branch distillation framework that independently supervises both score branches, uniquely specifying target branch outputs for each training sample through independent branch constraints, with an anchor term regularising conditional predictions toward ground-truth noise. The framework further introduces TIRT Transfer, which copies the teacher's converged per-timestep importance curriculum into the student as a frozen prior, eliminating the need to relearn it within limited distillation budgets. Experiments on CIFAR-10 and CIFAR-100 demonstrate that 5.9x compression maintains quality within 4 FID points of the teacher at 50-step DDIM sampling, considerably outperforming training from scratch with guidance fidelity well preserved. Ablation studies confirm that unconditional supervision is the dominant contribution, accounting for over 60% of total distillation gain. Curriculum transfer and anchor regularisation provide complementary benefit, together validating dual-branch constraints as empirically essential for guidance-preserving compression. 

---
# Extending Causal Metamodeling to a non-Markovian Queue 

**Authors**: Pracheta Amaranath, Anant Bhide, David Jensen, Peter Haas  

**Link**: [PDF](https://arxiv.org/pdf/2606.00795)  

**Abstract**: Metamodels for discrete-event simulations approximate the behavior of simulation models without running expensive simulations. Prior work introduced modular dynamic Bayesian networks (MDBNs) -- a class of metamodels that can estimate a range of probabilistic and causal queries (PCQs) using a single, trained model -- but the method was limited to Markovian systems. In this paper, we initiate an extension of MDBNs to non-Markovian queues by approximating non-exponential distributions using phase-type distributions. This approach raises novel challenges, including balancing metamodeling accuracy and tractability when choosing the number of phases, efficiently learning metamodel parameters, and choosing the sampling interval that is used to approximate a continuous-time simulation by a discrete-time MDBN. We provide preliminary solutions to these challenges, yielding the first causal metamodeling technique for non-Markovian systems. Experiments on a G/M/1 queue demonstrate that the MDBN can produce accurate answers to PCQs with orders-of-magnitude speedup of inference times relative to direct simulation. 

---
# Bayesian Inference of Nonlinear Malaria Dynamics in Ghana via an Ensemble Markov Chain Monte Carlo Sampler 

**Authors**: T. Ansah-Narh, Y. Asare Afrane, J. Bremang Tandoh  

**Link**: [PDF](https://arxiv.org/pdf/2606.00783)  

**Abstract**: Reliable quantification of malaria dynamics in sub-Saharan Africa is hindered by short, noisy, and spatially heterogeneous surveillance records. In Ghana, health-facility data from 2014 to 2023 reveal non-linear and age-specific fluctuations in hospital admissions, yet existing approaches struggle to capture stochastic variability or provide credible uncertainty bounds. This study develops a Bayesian nonlinear inference framework that integrates a cubic baseline with a damped oscillatory kernel, estimated via an affine-invariant ensemble Markov Chain Monte Carlo sampler. The framework accommodates limited data, models parameter uncertainty, and generates probabilistic forecasts for children under five years and individuals aged five years or more. Results show strong empirical adequacy ($R^2 = 0.9958$ for $<5$ years; $R^2 = 0.9956$ for $\geq 5$ years) with residual errors below $2\%$ and well-mixed posteriors confirming convergence. District-level analysis reveals pronounced spatial heterogeneity, with coefficients of variation ranging from $<0.07$ in urban centres such as Kumasi to $>3.3$ in peripheral districts such as Mpohor and Bia East. Forecasts for 2024-2026 indicate a gradual resurgence: from 137,000 to 149,000 cases among children under five years and from 348,000 to 375,000 cases among older individuals, with uncertainty widening over time. By producing probabilistic forecasts, this Bayesian framework provides a principled tool for anticipating malaria fluctuations and strengthening data-driven decision-making in Ghana's national malaria control strategy. 

---
# Behavior-Invariant Task Representation Learning with Transformer-based World Models for Offline Meta-Reinforcement Learning 

**Authors**: Fuyuan Qian, Menglong Zhang, Song Wang, Quanying Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.00780)  

**Abstract**: Offline meta-reinforcement learning leverages static datasets to enable agents to generalize to unseen environments by combining offline efficiency with meta-learning adaptability, yet it faces key challenges from context and policy distribution shifts. These issues hinder agents from adapting to online environments, and are further exacerbated under sparse-reward settings. As a result, agents often become trapped in an inherent pattern dilemma, failing to achieve robust generalization. In this work, we propose a novel framework that integrates information-theoretic task representation learning with a Transformer-based stochastic world model. Our approach extracts task-defining latent variables that are invariant to behavior policy, thereby effectively mitigating the context distribution shift. To further handle policy shift and model exploitation, we apply a conservative value penalty to imagination-based rollouts, preventing the policy from exploiting model inaccuracies while maintaining robust adaptation. Extensive evaluations demonstrate that our method outperforms state-of-the-art approaches, with superior stability and generalization under out-of-distribution and sparse-reward settings. 

---
# GIRL-DETR: Gradient-Isolated Reinforcement Learning for Video Moment Retrieval 

**Authors**: Shihang Zhang, Mingjin Kuai, Ye Wei, Zhen Zhang, Wei Ji  

**Link**: [PDF](https://arxiv.org/pdf/2606.00775)  

**Abstract**: Video Moment Retrieval (VMR) task requires accurately localizing temporal boundaries aligned with natural language queries, but many models suffer from a misalignment between continuous surrogate losses and non-differentiable metrics, leading to optimization stagnation during the late stages of training and trapping boundary predictions in suboptimal solutions. Although Reinforcement Learning (RL) post-training successfully optimizes localization results for large models, applying it directly to lightweight networks easily disrupts the fragile feature representations established during the supervised phase. To overcome this optimization bottleneck, we propose Gradient-Isolated Reinforcement Learning for DETR (GIRL-DETR), introducing RL post-training into a lightweight temporal localization framework for the first time. The input video and text features first establish early alignment through Cross-Modal Interaction (CMI) before entering the transformer encoder. Subsequently, a Text-Guided Gating (TGG) mechanism dynamically injects semantic priors into the queries before the transformer decoder generates candidate proposals, providing high signal-to-noise ratio inputs for temporal prediction. After the supervised training reaches convergence, the backbone network is frozen to protect the feature manifold, while the detection head directly optimizes the non-differentiable evaluation metric tIoU to enhance localization accuracy through a Three-stage Progressive Reinforcement Learning (TPRL) strategy. This approach achieves an orthogonal decoupling of state representation and metric optimization. Experiments on Charades-STA, QVHighlights, and TACoS demonstrate that GIRL-DETR effectively resolves surrogate loss degradation and achieves substantial accuracy improvements with minimal parameter updates, providing a robust new pathway for RL applications in lightweight VMR models. 

---
# Logit Distillation on Manifolds: Mapping by Learning 

**Authors**: Yiru Yang, Junling Wang, Nishant Kumar Singh, Luohong Wu, Haoran Yan  

**Link**: [PDF](https://arxiv.org/pdf/2606.00771)  

**Abstract**: A simple way to improve the performance of almost any machine learning model is not to train a single but several models with diverse algorithms which will make slightly distinct kinds of predictions and errors on the same data, and thus improve the average predictions and robustness. However, making predictions using a whole ensemble of models is cumbersome and computationally too expensive to allow deployment to a large number of users, especially if the models are large neural nets. In response to this, we introduce a layer and point wise projection mapping, which maps student and teacher representations into an aligned high-dimensional embedding space during training process. The proposed approach combined with LoRA injection reduces the student model trainable parameters to less than 1% of the teacher model, while significantly improving word error rate (WER) compared to other distillation methods, as demonstrated in ablation studies. Unlike a mixture of experts, our method can be trained rapidly and in parallel. 

---
# Causal Density Functions 

**Authors**: Sridhar Mahadevan  

**Link**: [PDF](https://arxiv.org/pdf/2606.00754)  

**Abstract**: We introduce causal density functions: Radon-Nikodym derivatives that compare interventional laws to observational laws and therefore act as local density ratios for causal effects. Whereas many causal-strength measures compare whole distributions after graph surgery, causal density functions provide a pointwise change-of-measure object that can be estimated, calibrated, and used to score directed influence. The basic identity
\[
\mathbb{E}_{\mathrm{do}}[f(Y)]
=
\mathbb{E}_{\mathrm{obs}}\!\left[f(Y)\rho(X,Y)\right]
\]
makes causal density directly testable: if the estimated density ratio is correct, observational expectations reweighted by $\rho$ reproduce interventional expectations. We derive practical estimators for do-curves and directed edge scores, relate the construction to Radon-Nikodym/Kan semantics for conditioning and intervention, and evaluate the resulting estimators on synthetic and real perturbation benchmarks. 

---
# SkyShield: Occupancy as a Safety Interface for Low-Altitude UAV Autonomy 

**Authors**: Jie Gao, Jie Ma, Kaihui Lin, Kai Ye, Miaohui Zhang, Pingyang Dai, Liujuan Cao  

**Link**: [PDF](https://arxiv.org/pdf/2606.00747)  

**Abstract**: For low-altitude Unmanned Aerial Vehicle (UAV) autonomy, 3D spatial understanding is not merely a perception objective, but the safety interface between human instructions and physical flight. In human-scale urban airspace below 20 meters, thin geometry, occlusions, vegetation, and urban clutter define whether an aerial agent can safely enter the space ahead. However, existing UAV datasets mainly provide 2D annotations or 3D boxes, while driving-oriented occupancy benchmarks assume stable ground-level sensor rigs. Both miss the defining regime of low-altitude flight: a front-facing monocular camera observing occupied and free space from a moving aerial body with frame-wise changing 6-DoF pose and camera extrinsics. To bridge this gap, we introduce \textbf{SkyShield}, to the best of our knowledge the first front-view monocular semantic occupancy benchmark for urban UAV flight below 20 meters. Built on CARLA, SkyShield contains 36K front-view UAV samples across diverse urban scenes and weather conditions, pairing each image with frame-wise 6-DoF UAV pose, frame-wise dynamic camera geometry, UAV states, and front-frustum semantic occupancy labels. We further propose \textbf{KAR-mIoU}, a UAV-centric and dynamics-aware metric that re-weights voxel-level evaluation by kinematic reachability and time-to-collision, revealing safety-critical risks hidden by conventional mIoU. To tackle this challenging new setting, we provide \textbf{SkyOcc}, a geometry-first monocular baseline that integrates frame-wise UAV attitude into projection, fuses temporal occupancy features, and applies safety-prior optimization to preserve sparse collision-critical structures. Together, SkyShield, KAR-mIoU, and SkyOcc establish occupancy as a safety interface for low-altitude aerial autonomy. Code and dataset will be released publicly. 

---
# Quantum Tunneling-Aware Machine Learning: Physics-Derived Noise Models for Robust Deployment 

**Authors**: Uiwon Hwang, Jaeho Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2606.00741)  

**Abstract**: Transistor scaling is approaching a quantum-mechanical limit, as thin gate oxides induce electron leakage through quantum tunneling. Unlike conventional digital systems, AI inference can tolerate such errors provided their structure is modeled correctly. In this paper, we introduce quantum tunneling-aware machine learning (QTAML). We derive the deployment-time weight-error distribution from first principles using the Wentzel-Kramers-Brillouin (WKB) approximation and show that it has structure that generic Gaussian noise models miss: an exact affine mean drift, a per-bit variance hierarchy dominated by the most-significant bit, and a per-layer dependence on $\|W_\ell\|_\infty$ and the trained-network Jacobian. We package these three structural properties into a single deployment-time algorithm, Tunneling-Aware Compensation (TAC), that combines closed-form mean correction with an optimal layer-adaptive bit-budget allocation derived from the WKB variance decomposition. Across four convolutional architectures at $p_\mathrm{flip}$=0.10 and a transformer encoder at $p_\mathrm{flip}$=0.05, TAC reaches $95\%$ of clean accuracy with 3.4$\times$ to 33.6$\times$ less ECC overhead than Uniform-MSP, the natural baseline derived from the same physics. The closed-form saturation ratio $\rho^*$ predicts these gains in advance, and on heterogeneous architectures WKB-derived scoring outperforms magnitude-based allocation by up to 24 percentage points at small budgets. The algorithm requires no retraining, no labels, and no inference-time overhead. We also verify the WKB-derived distributional theorems to Monte Carlo precision. These results connect WKB tunneling physics with noise-aware deep learning and suggest a principled path toward hardware--software co-design beyond conventional scaling limits. 

---
# SORA: Free Second-Order Attacks in Fast Adversarial Training 

**Authors**: Mazdak Teymourian, Ramtin Moslemi, Farzan Rahmani, Mohammad Hossein Rohban  

**Link**: [PDF](https://arxiv.org/pdf/2606.00738)  

**Abstract**: Adversarial Training (AT) is a leading defense against adversarial examples but often suffers from Catastrophic Overfitting (CO) in efficient single-step variants, where robustness to multi-step attacks collapses despite high single-step performance. We address this failure mode with two contributions. First, we formalize Epsilon Overfitting (EO), a perspective in which fixed perturbation magnitudes and directions exacerbate CO, and show that introducing perturbation variability significantly improves robust generalization across different architectures and datasets. Second, we propose PertAlign (Perturbation Alignment), a theoretically grounded, computationally negligible metric that predicts CO onset by measuring gradient alignment across attack stages. Leveraging these insights, we introduce SORA, an adaptive step-size AT method that dynamically adjusts perturbations based on loss surface geometry. SORA consistently prevents CO, achieves state-of-the-art robustness and clean accuracy, and generalizes across datasets and architectures using a single fixed set of hyperparameters, which is essential for applicability in fast AT. Extensive experiments on diverse datasets and architectures show that SORA matches or surpasses the robustness of prior methods while delivering higher clean accuracy and superior efficiency. Code is available at this https URL. 

---
# WaveFilter: Enhancing the Long-Context Capability of Diffusion LLMs via Wavelet-Guided KV Cache Filtering 

**Authors**: Jinnan Yang, Yan Wang, Zhen Bi, Kehao Wu, Xiaojie Li, Jungang Lou, Zechao Li, Jing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.00724)  

**Abstract**: Diffusion Large Language Models (DLMs) have demonstrated significant advantages across various tasks. However, constrained by their multi-step iterative inference mechanism, their computational overhead and inference latency in long-context tasks have become core bottlenecks restricting their large-scale deployment. When processing long sequences, existing Key-Value (KV) caching mechanisms often face a dilemma where generation quality degrades drastically, where the core challenge lies in precisely and efficiently filtering critical tokens within ultra-long contexts. Inspired by the human reading process, we propose \textbf{WaveFilter}, a universal and training-free caching framework. This framework innovatively introduces the wavelet transform for decomposition of long sequences to achieve precise identification of key tokens, based on which a sparse KV Cache is constructed to compute the final contextual representation. Experimental results demonstrate that WaveFilter, as a plug-and-play generic framework, significantly enhances the performance of existing mainstream KV Cache methods in complex long-context tasks. 

---
# EPIC: Efficient and Parallel Inference under CFG Constraints for Diffusion Language Models 

**Authors**: Hyundong Jin, Yo-Sub Han  

**Link**: [PDF](https://arxiv.org/pdf/2606.00722)  

**Abstract**: Controlling language model outputs is essential for ensuring structural validity, reliability, and downstream usability, and diffusion language models are no exception. Recent advances in diffusion language model decoding have extended output control beyond regular constraints to context-free grammar (CFG) constraints. Existing methods, however, can be up to four times slower than unconstrained decoding. More importantly, they substantially diminish one of the key advantages of diffusion language models over autoregressive models, namely parallel decoding. This slowdown arises because sequential validity checking introduces significant overhead during parallel generation. We propose an efficient CFG-constrained decoding framework, EPIC, that addresses this limitation. Our method improves decoding efficiency by combining lexing memoization, validation using Earley-style parsing instead of deterministic automata, and relaxed compatible subset selection for parallel commit. It reduces repeated lexing and validation overhead while allowing multiple compatible tokens to be committed together. Experiments on three benchmarks using four models show that our method reduces inference time by up to 67.5% and decreases the additional overhead by up to 90.5% compared with existing CFG-constrained decoding methods. Our implementation is available at this https URL . 

---
# Multi-Agent Conformal Prediction with Personalized Statistical Validity 

**Authors**: Martin V. Vejling, Christophe A. N. Biscio, Adrien Mazoyer, Petar Popovski, Shashi Raj Pandey  

**Link**: [PDF](https://arxiv.org/pdf/2606.00717)  

**Abstract**: Uncertainty quantification is essential in high-stakes machine learning tasks. However, one of the principled solutions, conformal prediction, faces challenges under limited local calibration data, privacy constraints, and data heterogeneity. In multi-agent settings, existing works do not simultaneously and satisfactorily address these challenges with guarantees either limited to averages across agents or losing validity in heterogeneous settings. Hence, we propose personalized federated weighted conformal prediction (PFWCP), a framework that combines local density ratio weighting with weighted quantile aggregation to correct for heterogeneity while preserving privacy. The method yields asymptotically valid marginal and calibration-conditional coverage guarantees for each participating agent and supports protocols with one-shot communication. Theoretical analysis presents an adjustment to the coverage variance, governed by an effective sample size expression, which is necessary in the context of weighted conformal prediction, and experiments on synthetic and real datasets show improved calibration quality over state-of-the-art federated conformal baselines. 

---
# Information-Theoretic Lower Bounds for Bit-Constrained Stochastic Optimization via a Reduction to Compressed Gaussian Mean Estimation 

**Authors**: Munsik Kim  

**Link**: [PDF](https://arxiv.org/pdf/2606.00703)  

**Abstract**: Low-precision pretraining (FP8, MXFP4, NVFP4) is now standard for frontier language models, yet the literature is almost entirely achievability -- algorithms and empirical scaling laws -- with no matching characterization of what is information-theoretically possible. We study a B-bit quantized stochastic first-order oracle: an optimizer interacts for T rounds and receives, each round, a B-bit adaptive public-coin description of its stochastic gradient. Our main contribution is an exact reduction from optimizing a strongly convex quadratic family to interactively compressed Gaussian mean estimation -- under the B-bit oracle the query carries no information, so optimization collapses exactly onto a sequential distributed-estimation problem. This yields two unconditional lower bounds, a communication bound TB = Omega(d) and a statistical bound T = Omega(sigma^2 d / eps^2), and the sharp product-form bound T = Omega((sigma^2 d / eps^2) max{1, d/B}). The product form is also unconditional: a B-bit transcript carries at most O(TB / sigma^2) of Fisher trace about the mean, so bits rather than dimension limit the recoverable information, and combined with the multivariate van Trees inequality this gives the bound directly, without bounded-likelihood-ratio truncation. We give a near-matching achievability result with exact per-round bit accounting under a bounded-dynamic-range oracle, tight up to a logarithmic factor; the lower bound is for truly Gaussian (unbounded) gradients, and closing this oracle gap is left open. A sequential rate-distortion perspective extends the reduction to correlated and drifting oracles and corrects an earlier conjecture: positive noise correlation raises the bound by (1+rho)/(1-rho) rather than relaxing it. The bounds give an information-theoretic baseline for any low-bit gradient path, not an optimality claim about deployed FP4 systems. 

---
# Shape Your Body: Value Gradients for Multi-Embodiment Robot Design 

**Authors**: Nico Bohlinger, Jan Peters  

**Link**: [PDF](https://arxiv.org/pdf/2606.00702)  

**Abstract**: We propose to turn generalist multi-embodiment value functions into reusable models for robot design. Instead of running a new reinforcement learning co-design loop for each robot, we first train an embodiment-aware policy and value function across many robot designs. After training, the frozen value function is used as a differentiable surrogate to optimize candidate embodiments through value gradients. We evaluate our approach across different robot design settings, from perturbed single robots to held-out robots across morphology classes, with single models trained on up to 50 robots and design spaces of over 1100 continuous embodiment parameters. Beyond optimizing complete embodiments, we show that value gradients can identify performance-limiting design and control parameters, enabling both the optimization and the analysis of new robot designs. 

---
# COPF: An Online Framework for Deployment-Stable Counterfactual Fairness in Evolving Graphs 

**Authors**: Sheng'en Li, Dongmian Zou  

**Link**: [PDF](https://arxiv.org/pdf/2606.00700)  

**Abstract**: Online link recommendation on evolving graphs is performative: by choosing which candidate links to show users, the system changes which links form and what feedback it later observes. Consequently, fairness estimates from logged outcomes can be misleading and may drift after deployment when the recommendation policy is updated. We introduce COPF (Counterfactual Online Performative Fairness), a decision-layer framework for deployment-stable fairness monitoring and control in online link recommendation. COPF (i) defines group-level opportunity gaps over exposure (shown vs. not shown) counterfactuals, (ii) makes them estimable by explicit exploration and by logging the probability (propensity) that each candidate is shown, and (iii) audits and controls fairness using residual outcome indistinguishability (OI) over a configurable auditor family with graph-aware doubly robust (GA-DR) estimators. We provide a noisy transfer theorem showing that Residual-OI on estimated GA-DR residuals implies bounds on exposure-counterfactual group gaps under temporal mixing and bounded local interference, and we instantiate an online multicalibration auditor together with a primal-dual controller. Experiments on two TGB streams and a controlled synthetic bipartite stream show that COPF reduces worst-case spikes in exposure-counterfactual group disparities with modest impact on ranking utility. Our code is available at this https URL. 

---
# The Paradox of Outcome Optimization: A Causal Information-Theoretic Bound on Reasoning Shortcuts in LLMs 

**Authors**: Zihan Chen, Yiming Zhang, Wenxiang Geng, Zenghui Ding, Yining Sun  

**Link**: [PDF](https://arxiv.org/pdf/2606.00674)  

**Abstract**: Large Language Models (LLMs) aligned via outcome-based Reinforcement Learning (RL) frequently exhibit a critical failure mode: they achieve high performance on in-distribution benchmarks while demonstrating brittle reasoning capabilities on out-of-distribution (OOD) tasks. We term this phenomenon Reward-Induced Manifold Collapse. We establish a theoretical framework bridging Structural Causal Models (SCM) and the Information Bottleneck (IB) principle to explain this paradox. We define reasoning as a high-complexity causal process and shortcut learning as the exploitation of low-complexity spurious correlations. Under the implicit inductive bias of Stochastic Gradient Descent (SGD), models optimized for outcome rewards are biased toward shortcut solutions whenever the training distribution allows for a ``Markovian Screening'' of the true causal mechanism. We derive a new generalization bound based on Semantic Coverage Measure ($\eta$) rather than sample size, showing why data scaling on homogeneous distributions may fail to correct reasoning flaws. We also show that Process Reward Models (PRMs) function as Topological Filters, enforcing step-wise mutual information constraints that render the low-complexity shortcut manifold inadmissible. These results provide a mathematical grounding for the role of process supervision beyond simple credit assignment. 

---
# Beyond the Mouth: Upper-Face Affective Cues in Audiovisual Sentence Recognition under Acoustic Uncertainty 

**Authors**: Zhou Yang, Yueyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.00670)  

**Abstract**: Face-to-face speech comprehension is inherently multimodal, integrating acoustic signals with visible articulation, facial expression, head motion, and other socially relevant cues. While audiovisual speech systems typically focus on the mouth region as the primary visual source of linguistic information, affective facial expressions are often treated separately as emotion-recognition targets. This paper investigates whether upper-face affective information contributes to audiovisual sentence recognition beyond audio and mouth-region cues, particularly under acoustic degradation.
Using the CREMA-D audiovisual emotional speech corpus, we train feature-based sentence classifiers under four cue conditions: audio only (A), audio plus mouth/lower-face features (A+M), audio plus upper-face features (A+U), and audio plus both mouth and upper-face features (A+M+U). Models are evaluated on clean audio and pink-noise conditions at +10 dB, +5 dB, and 0 dB SNR using actor-independent splits.
Results show that mouth/lower-face features provide substantial robustness benefits under degraded audio. At 0 dB SNR, A+M improves accuracy over A by 0.0794, with an actor-bootstrap 95% confidence interval of [0.0296, 0.1298]. Upper-face affective cues exhibit a more nuanced effect. Although the direct accuracy gain of A+M+U over A+M is small, full-face models consistently improve calibration across SNR levels and outperform shuffled upper-face controls under noisy conditions. These findings suggest that affective facial information may support multimodal robustness and confidence estimation under acoustic uncertainty without directly encoding lexical content. More broadly, the study highlights the potential role of socially expressive facial cues in human-centered audiovisual interaction systems. 

---
# Collaborative Few-Step Distillation and Low-Bit Quantization for Wan2.2 Dual-Expert Video Diffusion Models 

**Authors**: Jinyang Du, Shenghao Jin, Ziqian Xu, Ruihao Gong, Shiqiao Gu, Yang Yong, Jinyang Guo, Xianglong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.00658)  

**Abstract**: Large video diffusion models achieve strong visual quality but remain expensive to deploy because each sample requires many denoising steps and a large resident parameter footprint. This paper studies a deployment-oriented compression pipeline for Wan2.2-T2V-A14B by combining few-step distribution-matching distillation with low-bit quantization. The pipeline follows the model's dual-expert denoising route, calibrates the high-noise and low-noise branches separately, protects sensitive entrance layers, and uses HiF4-style low-bit representation to improve dynamic-range coverage. Quantization is calibrated on the distilled few-step student rather than on the original long-step trajectory, reducing activation-distribution mismatch during inference. The proposed co-design keeps the quantized model close to the same-step full-precision model and surpasses the original full-precision baseline at 8 and 20 steps on average. The 20-step setting gives the best quality-efficiency trade-off in the tested configurations. 

---
# Demystifying the Optimal Fair Classifier in Multi-Class Classification 

**Authors**: Li Zhang, Yuyuan Li, XiaoHua Feng, Jiaming Zhang, Fengyuan Yu, Chaochao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.00656)  

**Abstract**: Ensuring fair and equitable treatment across diverse groups, particularly in multi-class classification tasks, poses a significant challenge due to the persistent biases inherent in machine learning models. Most existing bias mitigation techniques are tailored to binary settings, and the presence of multi-dimensional outputs and complex fairness mechanisms makes their extension to multi-class scenarios neither straightforward nor effective. In this paper, we investigate two fundamental, unresolved challenges in fair classification: (i) characterizing the optimal accuracy-fairness frontier in multi-class settings, and (ii) designing practical algorithms that attain this optimum in different training phases. To tackle these challenges, we first specify an analytically tractable probabilistic formulation of the optimal classifier under fairness constraints. Building upon this, we propose two attribute-blind algorithms to enforce fairness requirements in practice: an in-processing approach for fairness intervention during training via the reduction approach, and a post-processing approach for fine-tuning output probabilities with plug-in estimation. Theoretical analysis reveals that both methods converge to the optimal accuracy-fairness Pareto frontier. Experiments conducted on multiple datasets demonstrate the superior performance of our methods in balancing accuracy and fairness. 

---
# Scaling Behavior of Single LLM-Driven Multi-Agent Systems 

**Authors**: Jialing Li, Zhouhong Gu, Yin Cai, Hongwei Feng  

**Link**: [PDF](https://arxiv.org/pdf/2606.00655)  

**Abstract**: The burgeoning field of LLM-based Multi-Agent Systems (MAS) promises to tackle complex tasks through collaborative intelligence, yet fundamental questions regarding their scaling behavior and intrinsic collective dynamics remain underexplored. This paper systematically investigates how the performance of a homogeneous MAS evolves as the number of agents increases, isolating the variable of collaboration from model or knowledge heterogeneity. We propose the Sequential Iterative Multi-Agent System (SIMAS) framework, a minimalist architecture centered on sequential inter-agent communication, to clearly observe scaling effects. Through extensive experiments across diverse tasks and model scales, we establish that MAS performance does not scale monotonically with agent count but follows a pattern of diminishing returns, governed by a trade-off between collaborative synergy and coordination overhead. Our findings reveal that effective MAS requires a sufficiently capable base LLM, that task type critically modulates the optimal agent count, and that collective intelligence is an emergent property contingent on strategic interaction design rather than a guaranteed outcome of agent plurality. The performance degradation stems coordination overhead rather than merely long-context failure, and the scaling tendency generalizes across interaction architectures like structured debate topologies. This work provides a foundational understanding of MAS scaling laws, offering practical guidance for designing efficient collaborative systems and challenging the prevailing assumption that more agents invariably lead to better performance. 

---
# MESA: Improving MoE Safety Alignment via Decentralized Expertise 

**Authors**: Yitong Sun, Yao Huang, Teng Li, Ranjie Duan, Yichi Zhang, Xingjun Ma, Hui Xue, Xingxing Wei  

**Link**: [PDF](https://arxiv.org/pdf/2606.00651)  

**Abstract**: Mixture-of-Experts (MoE) architectures scale Large Language Models (LLMs) efficiently, enabling greater capacity with reduced computational cost by dynamically routing inputs to relevant experts, yet introduce a critical vulnerability: Safety Sparsity, where safety capabilities concentrate in few experts, making them susceptible to adversarial bypassing. Meanwhile, conventional alignment methods uniformly adapt all parameters, ignoring their functional differences and inadvertently degrading performances. To address these challenges, we propose MESA (MoE Safety Alignment), a targeted alignment framework for MoE-based LLMs that strategically decentralizes safety responsibility to maximize coverage while minimizing interference with utility. Based on Optimal Transport (OT) theory, MESA operates through two mechanisms: (1) Expert Capacity Reallocation uses a transport cost matrix to distribute safety duties to the most cost-effective experts, and (2) Dynamic Routing Refinement constrains the router to precisely activate these decentralized modules. Experiments show that MESA achieves robust defensive performance against varied harmful benchmarks while preserving helpfulness. Code is available at this https URL. 

---
# LinguIUTics at PsyDefDetect: Iterative Imbalance-Aware Fine-tuning of Qwen3-8B for Psychological Defense Mechanism Classification 

**Authors**: Shefayat E Shams Adib, Ahmed Alfey Sani, Md Hasibur Rahman Alif, Ajwad Abrar  

**Link**: [PDF](https://arxiv.org/pdf/2606.00647)  

**Abstract**: Detecting psychological defense mechanisms in conversational text remains a challenging clinical NLP problem. For the PsyDefDetect 2026 shared task (nine-class utterance classification evaluated via macro F1), our team LinguIUTics achieves a macro F1-score of 0.3917 on the official positive-class leaderboard, ranking 4th out of 21 registered teams and improving over the Ministral-8B task baseline (31.48 macro F1) by 7.7 absolute points (24.4 percent relative). BERT-family encoders and zero-shot LLMs proved ineffective on rare classes due to severe class imbalance, leading us to QLoRA fine-tuning of Qwen3-8B. We leverage three key strategies: grouped stratified cross-validation (preventing leakage), minority-class round-robin lexical augmentation, and a post-processing pipeline with logit bias tuning and ensemble blending. Together, these components close much of the validation-to-leaderboard gap and substantially improve minority-class recall, driving the critical "Unclear" class (Level 8) from near-zero performance to an F1 score of 0.797. 

---
# LP5X-PIM Sim: A High-Fidelity HW/SW Integrated Simulator for LPDDR5X-PIM 

**Authors**: SangHoon Cha, Jaewan Choi, Byeongho Kim, Yoonah Paik, Sukhan Lee, Kyomin Sohn  

**Link**: [PDF](https://arxiv.org/pdf/2606.00636)  

**Abstract**: This tech note describes the architecture and execution results of the LPDDR5X-PIM simulator, developed by Samsung Electronics. Based on the latest research and internal specifications, the simulator provides a high-fidelity model of both the hardware data paths and the software control layers of the LPDDR5X-PIM block. This integrated hardware-software simulation approach enables precise evaluation of system performance and energy efficiency while maximizing PIM resource utilization. We have refined existing simulation frameworks to align with actual hardware implementation, ensuring consistent behavioral accuracy. Further technical details regarding the specific architecture and circuit design of the LPDDR5X-PIM will be disclosed in future publications 

---
# Authenticity Debt and the Synthetic Content Threat Landscape: A Layered Framework for Trust, Provenance, and IP Governance in the Generative AI Era 

**Authors**: Shubhashis Sengupta, Benjamin McCarty, Milind Savagaonkar, Rhine Andotra  

**Link**: [PDF](https://arxiv.org/pdf/2606.00621)  

**Abstract**: Generative artificial intelligence has fundamentally changed how content is now produced. It has enabled how high-fidelity text, images, audio, and videos are created, modified, and redistributed at near-zero marginal cost. This shift exposes enterprises and ecosystems to a number of risks across four reinforcing authenticity layers -- authenticity, provenance, integrity, and accountability -- that traditional controls are inadequate to address in isolation. We introduce the concept of authenticity debt: the cumulative institutional liability that accumulates when organizations deploy AI-generated content without preserving verifiable origin, integrity, and accountability, deferring exposure that surfaces under regulatory, legal, or market scrutiny. This paper presents a comprehensive, multi-dimensional taxonomy of generative AI harms and attack vectors, surveys the capabilities and failure modes of technical controls including digital watermarking, provenance frameworks (C2PA, Adobe CAI), and detection technologies, and argues that no single mechanism is sufficient in open, adversarial, and evolving environments. Drawing on Zero Trust Architecture principles and enterprise governance frameworks, we propose a layered reference architecture that integrates cryptographic provenance, human-in-the-loop verification, and continuous governance to sustain defensible authenticity at scale. We further examine the regulatory landscape (EU AI Act, U.S.\ FTC, NIST AI RMF) and identify practical guiding principles for organizations seeking to build authenticity as institutional infrastructure rather than an afterthought. 

---
# MemPro: Agentic Memory Systems as Evolvable Programs 

**Authors**: Qingshan Liu, Guoqing Wang, Wen Wu, Jingqi Huang, Xinqi Tao, Dejia Song, Jie Zhou, Liang He  

**Link**: [PDF](https://arxiv.org/pdf/2606.00619)  

**Abstract**: Long-horizon autonomous agents require memory systems to retain historical information, track evolving states, and reuse relevant knowledge beyond finite context windows. Existing agentic memory systems typically follow a memory construction-retrieval (MCR) pipeline, but often adapt mainly the memory bank while keeping the surrounding pipeline fixed after deployment. This fixed-pipeline design struggles to handle heterogeneous task-specific failure modes and can become misaligned with memory banks that evolve in scale and structure over time. To address these limitations, we propose MemPro, a system-level evolution framework that treats the entire MCR pipeline as an evolvable program rather than adapting only the memory bank or prompt text. MemPro maintains a version tree of runnable memory-system implementations, where an Evolving Agent iteratively selects promising versions, diagnoses recurring failures, and creates improved child versions through failure-mode-guided edit-debug refinement. Experiments on LongMemEval, LoCoMo, HotpotQA, and NarrativeQA show that MemPro consistently outperforms strong static and prompt-level evolving baselines within a few iterations, continues to improve with evolution, and achieves a favorable performance-cost trade-off. Code is available at this https URL. 

---
# Pause and Think: A Dataset and Benchmark for Video-Grounded Assistive Action Suggestion 

**Authors**: Shivam Singh, Saptarshi Majumdar, Pratik Prabhanjan, Zicheng Liu, Emad Barsoum  

**Link**: [PDF](https://arxiv.org/pdf/2606.00616)  

**Abstract**: Recent Vision-Language Models (VLMs) struggle with grounded reasoning, temporal consistency, and context aware planning in videos. We introduce pause-and-think-T, a reasoning-centric training dataset that encourages models to pause, reason over visual evidence, and produce concise, actionable responses. The dataset promotes structured reasoning prior to answer generation, guiding models toward human-like, scene-grounded assistance. We fine-tune a compact 4B-parameter model and evaluate it on our pause-and-think-B benchmark targeting contextual understanding and goal planning tasks. The model achieves 58.0% accuracy at 59x fewer parameters than Qwen3-VL-235B (58.9%), matching GPT-5.2 on scene understanding and surpassing GPT-4o. Beyond our benchmark, it also shows strong out-of-distribution performance on EgoThink and TempCompass, with substantial gains in affordance, assistance, attribution recognition, situated reasoning, and temporal order, without benchmark-specific training. Our results indicate that targeted reasoning supervision enables compact models to deliver actionable, visually grounded guidance while generalizing beyond training data, without requiring large-scale model expansion. 

---
# Linguistics-Aware Non-Distortionary LLM Watermarking 

**Authors**: Shinwoo Park, Hyejin Park, Hyeseon An, Yo-Sub Han  

**Link**: [PDF](https://arxiv.org/pdf/2606.00613)  

**Abstract**: Watermarking should identify language-model output without degrading quality or limiting verification to the model provider. Multilingual deployment makes this harder because morphology, segmentation, and script change where watermark evidence can enter naturally. We introduce LUNA, a linguistically adaptive watermark that combines model-free detection with single-token non-distortion under the standard random-key model. LUNA estimates normalized next-tag entropy from part-of-speech contexts in an external corpus and uses it to set the depth of a non-distortionary binary tournament sampler; the detector reconstructs the same schedule from text, a tokenizer, a tagger, and a secret key. We evaluate six typologically diverse languages and two domains against eight primary baselines. LUNA attains an AUROC of 0.9959 and the lowest mean absolute median perplexity shift of 0.045 across the twelve settings; its 95% bootstrap interval [0.022, 0.073] lies below all baseline intervals. LUNA also records the lowest mean Self-BLEU, Distinct-1, surprisal, and entropy shifts. It is the only method that simultaneously achieves AUROC > 0.99 and an absolute median perplexity shift below 0.1 in a majority of settings, reaching this regime in 9 of the 12 settings while no baseline reaches it in more than 2. Our code is available at: this https URL 

---
# MemGraphRAG: Memory-based Multi-Agent System for Graph Retrieval-Augmented Generation 

**Authors**: Chuanjie Wu, Zhishang Xiang, Yunbo Tang, Zerui Chen, Qinggang Zhang, Jinsong Su  

**Link**: [PDF](https://arxiv.org/pdf/2606.00610)  

**Abstract**: Retrieval-Augmented Generation (RAG) has become an essential method for mitigating hallucinations in Large Language Models (LLMs) by leveraging external knowledge. Although effective for simple queries, traditional RAG struggles with large-scale, unstructured corpora where information is highly fragmented. Graph-based RAG (GraphRAG) incorporates knowledge graphs to capture structural relationships, enabling more comprehensive retrieval for complex reasoning. However, existing GraphRAG methods rely on isolated, fragment-level extraction for graph construction, lacking a global perspective on the whole corpus. As a result, these methods frequently lead to thematically inconsistent, logically conflicting, and structurally fragmented graphs that degrade retrieval performance. In this paper, we propose MemGraphRAG, a novel framework that introduces a memory-based multi-agent system to ensure high-quality graph construction. Specifically, MemGraphRAG employs a collaborative society of agents supported by shared memory, which provides a unified global context throughout the extraction process. This mechanism allows agents to dynamically resolve logical conflicts and maintain structural connectivity throughout the corpus. Furthermore, we propose a memory-aware hierarchical retrieval algorithm tailored for the constructed graph. Extensive experiments on multiple benchmarks demonstrate that MemGraphRAG outperforms the state-of-the-art baseline models with comparable efficiency. Our code is available at this https URL. 

---
# CARE-RL: Capability-Aware Reinforcement Learning for Mitigating Cross-Domain Conflicts 

**Authors**: Rui Zhang, Xinle Wu, Yao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2606.00609)  

**Abstract**: Reinforcement learning (RL) with verifiable rewards has achieved strong progress in reasoning-oriented LLMs, but extending it to multi-domain RL remains challenging due to reward unreliability in non-verifiable tasks and capability interference across domains. We propose CARE-RL to combine protocol-aware reward generation with capability-aware optimization for mitigating cross-domain conflicts. For non-verifiable tasks, the Protocol-Aware Generative Reward Model (PA-GRM) constructs prompt-level evaluation protocols and schemas before producing trace-conditioned rewards, enabling task-adaptive yet comparable evaluation of open-ended responses. For multi-domain optimization, Direction-Aware Capability Subspace Projection (DACSP) extracts historical capability directions from previous RL stages and modulates later updates by amplifying aligned components, suppressing conflicting components, and preserving orthogonal updates. Experiments across math, chat, and instruction-following benchmarks show that CARE-RL consistently outperforms standard multi-domain RL baselines, achieving Total Avg scores of 47.9 and 50.7 on Qwen2.5-7B and Qwen3-4B, respectively. 

---
# SPADER: Step-wise Peer Advantage with Diversity-Aware Exploration Rewards for Multi-Answer Question Answering 

**Authors**: Qiming Shi, Zhaolu Kang, Yunfan Zhou, Di Weng, Yingcai Wu  

**Link**: [PDF](https://arxiv.org/pdf/2606.00593)  

**Abstract**: Large language models are increasingly deployed as tool-augmented agents to acquire information beyond parametric knowledge. While recent work has improved long-horizon tool-use reasoning, most approaches focus on tasks with a single correct answer. In contrast, many real-world queries require discovering a comprehensive set of valid answers, a setting known as Multi-Answer QA. This setting raises two challenges: fine-grained credit assignment over long search trajectories and reward alignment for sustained exploration beyond easy high-frequency entities. We propose SPADER, a reinforcement learning framework for long-horizon tool use in Multi-Answer QA. SPADER includes Step-wise Peer Advantage (SPA), a critic-free step-level credit assignment mechanism that aligns parallel trajectories by decision step and estimates advantages from peer returns. It also includes a diversity-aware exploration reward that promotes long-tail entity discovery by upweighting rare findings and downweighting redundant ones. Experiments on QAMPARI, Mintaka, WebQSP, and QUEST show that SPADER generally improves recall and overall F1 over prompting-based agents, outcome-supervised RL methods, and recent step-level supervision approaches. Our code and model weights are available at this https URL. 

---
# Critic-R: Improving Agentic Search using Instruction-tuned Retrievers with Natural Language Introspective Feedback 

**Authors**: Md Zarif Ul Alam, Alireza Salemi, Hamed Zamani  

**Link**: [PDF](https://arxiv.org/pdf/2606.00590)  

**Abstract**: Agentic search systems iteratively interact with retrieval models to answer complex queries. Despite substantial progress, optimizing retrievers for agentic search remains challenging, often requiring heavy co-training or gold-standard annotations that limit real-world applicability. We propose Critic-R, a framework that explicitly closes the feedback loop between the reasoning agent and the retrieval model during both inference and training. Critic-R introduces a critic model that evaluates the agent's introspective reasoning trace after consuming retrieved evidence to determine whether the retrieved context sufficiently supports the next reasoning step. Critic-R has two complementary mechanisms: Critic-R-Zero, an inference-time query refinement loop that iteratively rewrites queries and retrieval instructions, and Critic-Embed, an optimization approach for retrieval models that leverages successful and failed refinement trajectories as automatic supervision without requiring manual relevance annotation. We evaluate Critic-R on HotpotQA, 2WikiMultihopQA, MuSiQue, and Bamboogle. Results show that Critic-R significantly improves both retrieval quality and downstream answer accuracy. 

---
# Improving Visual Representation Alignment Generation with GRPO 

**Authors**: Shentong Mo, Sukmin Yun  

**Link**: [PDF](https://arxiv.org/pdf/2606.00583)  

**Abstract**: Recent diffusion transformers have demonstrated strong image synthesis capabilities but remain inefficient to train due to weak alignment between generative and discriminative representations. While representation alignment frameworks such as REPA improve convergence by aligning noisy denoising features with pretrained visual encoders, their externally supervised alignment loss is static and lacks adaptivity during training and inference. Existing methods rely on fixed cosine alignment or contrastive objectives, which cannot dynamically balance representation consistency and generation quality, resulting in limited discriminative benefit and failing to optimize alignment in a task-adaptive manner. To address this, we propose VRPO, a reinforcement-based optimization strategy that replaces REPA's static alignment loss with a generative representation policy optimization objective. Instead of enforcing a fixed similarity constraint, VRPO treats representation alignment as a reward-guided process: the model receives adaptive rewards based on generation fidelity, perceptual quality, and semantic coherence between the diffusion features and pretrained visual embeddings. This formulation enables the generator to continuously refine its internal representations toward semantically meaningful directions while improving image quality. Our VRPO-driven training seamlessly integrates into diffusion transformers, introducing negligible computation cost and preserving full compatibility with SiT and DiT architectures. Extensive experiments on ImageNet-256x256 demonstrate that our VRPO-Alignment substantially enhances both convergence and fidelity, achieving up to +1.8 FID improvement and 2.3x faster training compared to REPA under identical compute budgets. 

---
# On the Difficulty of Learning a Meta-network for Training Data Selection 

**Authors**: Zilin Du, Junqi Zhao, Boyang Albert Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.00571)  

**Abstract**: Synthetic data are increasingly used to train neural networks, yet distributional mismatch with real data limits their effectiveness when used indiscriminately. A common strategy is to learn data weights via bi-level optimization, which we refer to as Meta-learning for Training-data Selection (MTS). Interestingly, in practice, MTS often performs below expectation. We identify two obstacles in properly training MTS: a poor gradient signal-to-noise ratio (GSNR), which causes optimization difficulties, and lack of informative features that correlates with data quality. We present a mathematical analysis of MTS, which reveals the dynamics of normalized data weights and the relation between disparate data quality and poor GSNR. The analysis suggests a a simple yet effective solution: increasing the batch size. Further, we propose a set of informative features that capture the positions of training data in their distributions and training dynamics. Experiments across four benchmarks show consistent improvements, achieving average gains of 5.49% over training without selection and 2.89% over the strongest baseline. 

---
# Revisiting Parameter-Based Knowledge Editing in Large Language Models: Theoretical Limits and Empirical Evidence 

**Authors**: Wanying Ren, Xin Song, Futing Wang, Guoxiu He, Aixin Sun  

**Link**: [PDF](https://arxiv.org/pdf/2606.00570)  

**Abstract**: Parameter-based knowledge editing updates the internal knowledge of large language models (LLMs) via localized weight modifications and has attracted significant attention. However, most existing methods overlook fundamental theoretical limitations and are rarely evaluated under realistic, practice-oriented settings. In this paper, we first present a theoretical analysis based on the dimensional Collapse Hypothesis, explaining how localized parameter edits can propagate along fragile directions in the representation space, inducing global interference and ultimately causing reasoning collapse. Building on this insight, we conduct a comprehensive empirical evaluation by systematically varying knowledge complexity, number of edits, evaluation dimensions, and baseline methods. Our results show that parameter-based editing methods consistently damage core LLM capabilities. In contrast, a simple retrieval-based baseline achieves consistently stronger performance than all parameter-editing methods across all evaluated conditions. These findings highlight that preserving the fundamental capabilities of LLMs after knowledge editing should be a central concern for future research. 

---
# A Practical Upper Bound on Selection Bias Effects in Medical Prediction Models 

**Authors**: Kara Liu, Maggie Wang, Russ B. Altman  

**Link**: [PDF](https://arxiv.org/pdf/2606.00563)  

**Abstract**: Selection bias is a common and often unavoidable aspect of real-world data that challenges the generalizability of machine learning models. When models trained on biased data are deployed in the broader target population, poor model generalization may lead to real harm, particularly in high-risk settings such as healthcare. This risk highlights the need for practitioners to reliably assess model generalizability prior to deployment. However, existing methods for predicting model performance rely on unrealistic access to the target distribution or knowledge of the selection mechanism causing bias. To address these limitations, we propose a novel upper bound on the worst-case model performance on the target population under the realistic setting where the selection mechanism and the target population data are only partially observed. We demonstrate the validity and practical utility of our method through experiments on fully synthetic data, semi-synthetic data derived from the All of Us Research Program, and real-world selection bias in MIMIC-IV. Our work offers a principled and practical tool to estimate the impact of selection bias in an otherwise intractable setting, thereby enabling practitioners to build safer and more generalizable models in healthcare and beyond. 

---
# Interpretable Policy Distillation for Power Grid Topology Control 

**Authors**: Aleksandra Dmitruka, Karlis Freivalds  

**Link**: [PDF](https://arxiv.org/pdf/2606.00561)  

**Abstract**: Deep reinforcement learning (RL) offers a promising route to real-time power grid operation, yet large neural policies are costly to evaluate, hard to deploy on constrained hardware, and opaque to operators. We ask whether a Proximal Policy Optimization (PPO) agent for grid topology control can be compressed into compact tree-based surrogates without losing operational performance. A PPO teacher is trained on Grid2Op's standard 14-bus environment with a stability-oriented reward, using stress-focused data collection on critical, high-loading states. The policy is then distilled into a decision tree and a random forest. Across held-out validation episodes, both surrogates exceed the teacher in mean reward and survival length at a fraction of the inference cost. The decision tree shows high exact-action agreement with the PPO argmax and near-complete agreement within its top-ranked actions, while remaining small enough to be inspected directly. Feature-importance analysis reveals a representational shift: the PPO policy relies mainly on line-loading signals, while the distilled tree is driven primarily by bus-topology variables. These results suggest that stress-focused distillation can convert a black-box neural controller into a lightweight, auditable rule-like surrogate suited for real-time deployment, while also surfacing risks tied to deterministic actions and topology-specific generalization. 

---
# Richer Representations for Neural Algorithmic Reasoning via Auxiliary Reconstruction 

**Authors**: Jiafu Huang, Chao Peng, Chenyang Xu, Zhengfeng Yang, Kecheng Cai, Chenhao Zhang, Yi Wang, Yiwei Gong, Wanqin Zhou, Irene Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2606.00559)  

**Abstract**: Neural algorithmic reasoning has emerged as a popular research direction. It aims to train neural networks to mimic the step-by-step behavior of classical rule-based algorithms. More specifically, the execution of such algorithms can be abstracted as a sequence of states, where each state represents the intermediate outcome after an execution step. The training objective is to generate state sequences that replicate the underlying algorithmic process. A common framework for this task adopts an encoder-processor-decoder architecture, where the encoder learns representations of states, the processor simulates algorithmic steps, and the decoder reconstructs output states. While prior work has focused on improving the processor, the role of the encoder in representation learning has received little attention. Most methods rely on simple MLP encoders, raising the question of whether such representations are sufficiently informative for supporting algorithmic reasoning. This paper investigates how to improve encoder representations for neural algorithmic reasoning. We propose a reconstruction module that aims to recover the input state from its encoded representation. This auxiliary reconstruction task encourages the encoder to retain critical information about the input. We demonstrate that incorporating this task during training improves the performance of existing neural architectures on standard benchmarks. Furthermore, we observe that current encoders often underutilize the correlations among features within a state. To address this, we draw inspiration from self-supervised learning and design an enhanced variant of the auxiliary task that encourages the encoder to capture intra-state feature dependencies. Experimental results show that our method enables the encoder to learn richer representations, thereby enhancing the performance of existing processors on algorithmic reasoning tasks. 

---
# CAFOSat: A Strongly Annotated Dataset for Infrastructure-Aware CAFO Mapping Using High-Resolution Imagery 

**Authors**: Oishee Bintey Hoque, Nibir Chandra Mandal, Mandy L Wilson, Samarth Swarup, Madhav Marathe, Abhijin Adiga  

**Link**: [PDF](https://arxiv.org/pdf/2606.00548)  

**Abstract**: Concentrated Animal Feeding Operations (CAFOs) play an important role in agricultural production but are also associated with environmental, public health, and disease surveillance concerns. Large-scale mapping of CAFOs from remote sensing imagery remains challenging due to heterogeneous infrastructure layouts, noisy location records, inconsistent annotations, and incomplete inventories. We introduce CAFOSat, a strongly annotated, infrastructure-aware dataset for CAFO mapping across the United States. CAFOSat integrates high-resolution National Agriculture Imagery Program (NAIP) imagery with multi-source CAFO inventories collected across multiple states and transforms weak geolocation records into refined annotations through a human-in-the-loop pipeline combining AI-assisted annotation, GradCAM-based localization, and geometric clustering. To improve dataset quality, we curate challenging negative samples using land-cover-guided sampling with spatial exclusion constraints and provide infrastructure-level annotations, including barns, manure ponds, and grazing-related features, through manual verification. The resulting dataset contains more than 45,000 image patches spanning 20 states and four major CAFO categories. We benchmark a diverse set of convolutional, transformer-based, and vision-language models, demonstrating the value of refined annotations and curated negative samples for CAFO classification and generalization. In addition, we introduce a synthetic augmentation pipeline that generates infrastructure-aware variations to increase training diversity and improve robustness under distribution shifts. CAFOSat provides a large-scale benchmark for advancing infrastructure-aware agricultural monitoring and CAFO mapping from high-resolution remote sensing imagery. 

---
# PaCo-VLA: Passivity-Shielded Compliance Prior for Contact-Rich Vision-Language-Action Manipulation 

**Authors**: Haofan Cao, Zhaoyang Li, Zhichao You, Liang Guo, Tianrui Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.00515)  

**Abstract**: Contact-rich manipulation demands both high-level semantic reasoning and the safe regulation of high-frequency contact dynamics. While Vision-Language-Action (VLA) models provide unprecedented semantic generalization, their low-rate outputs lack the reliability required for direct plant authority in force-sensitive tasks. To bridge this semantic-to-control gap, we introduce PaCo-VLA, a passivity-shielded compliance prior that recasts the VLA interface. Rather than trusting VLAs with direct motor commands, PaCo-VLA treats network outputs as task-level compliance proposals: semantic bindings, task stages, and admittance schedules. A high-frequency, proposal-independent passivity shield governs these proposals through energy-tank accounting and boundary checks, preventing invalid, stale, or unverified model predictions from bypassing low-level contact physics. This decoupled architecture also enables causal evaluation, isolating semantic contributions from geometric shortcuts. Extensive simulated and real-world connector-insertion experiments demonstrate that PaCo-VLA achieves superior precision over unshielded VLA baselines, sustaining zero passivity violations even under adversarial compliance shifts. This framework establishes a provably sampled-passive runtime contract at the admittance port and provides a runtime interface for deploying foundation models in contact-rich domains. 

---
# Skill or Skip? Learning Selective Skill Invocation in Agentic Tasks via Dual-Granularity Preference Learning 

**Authors**: Chishui Chen, Jiaye Lin, Te Sun, Junxi Wang, Yi Yang, Cong Qin, Yangen Hu, Lu Pan, Ke Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2606.00510)  

**Abstract**: Agent skills are callable procedural modules that provide reusable knowledge and execution policies for complex agentic tasks. However, existing methods mainly focus on selecting relevant skills or improving the skills themselves, while overlooking whether a relevant skill should actually be invoked at the current decision point. Unhelpful invocations may introduce irrelevant context and disrupt an otherwise correct execution process. To address this issue, we propose SelSkill, a dual-granularity preference-learning framework for selective skill invocation. SelSkill formulates skill use as a skill-or-skip decision, uses predictive uncertainty to prioritize candidate decision points, and constructs controlled invoke-skip preference pairs from shared trajectory prefixes. It further combines episode-level outcome preferences with step-level invocation preferences to capture both overall trajectory quality and the local effectiveness of skill invocation. On ALFWorld with Qwen3-8B, SelSkill improves task success by 10.9 percentage points and execution precision by 29.1 percentage points. On BFCL, it improves task success by 5.7 percentage points and execution precision by 29.5 percentage points. Zero-shot results on Tau-bench and PopQA further suggest that the learned invocation policy transfers to new domains with previously unseen skills. 

---
# V-LynX: Token Interface Alignment for Video+X LLMs 

**Authors**: Jungin Park, Jiyoung Lee, Kwanghoon Sohn  

**Link**: [PDF](https://arxiv.org/pdf/2606.00508)  

**Abstract**: This study introduces an intriguing phenomenon in Video LLMs: rather than merely translating frames into textual embeddings, Video LLMs establish a continuous manifold, token interface, allowing visual tokens to operate as standalone entities within the architecture. Exploiting this discovery, we propose V-LynX, a scalable framework that integrates novel modalities into Video LLMs by repurposing the internalized interface. Departing from conventional paradigms that necessitate heavy modality-specific encoders or paired supervision, V-LynX employs a lightweight auxiliary pathway in parallel with the frozen vision encoder. Our method integrates new sensory inputs with intrinsic video priors by aligning both attention responses and statistical distributions using unpaired unimodal data sets. This ensures manifold compatibility while preserving the integrity of the Video LLMs. Extensive benchmarks demonstrate that V-LynX achieves SOTA and efficiency across audio-visual QA, 3D reasoning, high-frame-rate, and multi-view video understanding. The code is available at this https URL. 

---
# TabChange: Precise Attribute Changes in Tabular Data 

**Authors**: Arjun Dahal, Yu Lei, Raghu N. Kacker, Richard Kuhn  

**Link**: [PDF](https://arxiv.org/pdf/2606.00503)  

**Abstract**: Modifying an attribute in tabular data often introduces an unnatural instance by breaking its relationships with other attributes. The modified instance must be both natural and minimally changed from the original instance. This paper addresses the challenge of generating such a modified instance. We identify key limitations in existing approaches: generative models either don't support instance-level attribute editing or, in the case of methods like CVAE, retain attribute information in the latent space, leading to unnecessary modifications. To solve this, we propose TabChange, an approach that analyzes the relationship between the attribute of interest and other attributes in the dataset. If the relationship is weak, it simply flips the attribute; if it is strong, it uses an adversarial framework that removes information about the attribute in the latent space representation. This removal enables precise modifications, making only the necessary adjustments to maintain naturalness. Our experiments across seven datasets show that TabChange generates counterfactuals in attributes that are comparable in naturalness and are more proximal to their original instances. This leads to a higher number of valid counterfactuals and a lower number of invalid counterfactuals compared to the baselines. 

---
# Pre-Deployment Robustness Stress Testing for CT Segmentation Systems Using Clinically Motivated Multi-Corruption Augmentation 

**Authors**: CholMin Kang, Jonghyun Chung, Amanpreet Kaurb, Nagesh Gulkotwarb, Arthi Sivasankaranb  

**Link**: [PDF](https://arxiv.org/pdf/2606.00491)  

**Abstract**: Deep learning-based CT segmentation systems often achieve high accuracy on clean benchmark images, but their performance may degrade under heterogeneous clinical imaging conditions such as noise, resolution loss, contrast variation, intensity shift, and artifacts. This instability can limit reliable deployment in real-world medical imaging workflows.
We propose Robustness via Augmented Multi-corruption Pipeline (RAMP), a robustness-oriented augmentation framework for CT segmentation. RAMP combines anatomically constrained spatial perturbations, CT intensity transformations, and stochastic multi-corruption composition to expose models to clinically plausible image degradation during training.
Across two CT segmentation evaluation settings, RAMP achieved the strongest corrupted-image performance and the smallest clean-to-corrupted robustness gap. In the five-organ noisy evaluation benchmark, RAMP improved mean corrupted Dice from 0.610 to 0.753 and reduced the robustness gap from 0.264 to 0.064 compared with the nnU-Net baseline. In Abdomen1K, RAMP improved mean corrupted Dice from 0.633 to 0.789 and reduced the robustness gap from 0.290 to 0.070. Although RAMP did not achieve the highest clean-image Dice, it substantially mitigated worst-case segmentation collapse under severe image degradation.
These results suggest that multi-corruption augmentation can serve as a practical pre-deployment strategy for improving the reliability of CT segmentation systems in heterogeneous clinical environments. 

---
# CodeCytos: AI-assisted spatial molecular imaging analysis via code-augmented agent action space 

**Authors**: Hung Q. Vo, Huy Q. Vo, Son T. Ly, Zhihao Wan, Anh-Vu Nguyen, Hong Zhao, Jianting Sheng, Stephen T. C. Wong, Hien V. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2606.00472)  

**Abstract**: Conventional tissue image analysis software provides foundational capabilities for cellular analysis, including segmentation, basic morphological feature extraction, and spatial organization analysis. However, these tools often require manual intervention and are not well integrated with code-driven automation, limiting efficiency and scalability for complex spatial tissue studies. In addition, they offer limited flexibility for custom analyses, as they typically support only a fixed set of pre-implemented spatial cellular features. To address these limitations, we propose CodeCytos, a coding-based reasoning agent framework that enables dynamic, programmable interaction with spatial molecular imaging data to improve automation and customization. CodeCytos is designed to streamline the exploration of custom spatial cellular features and adapt to diverse research needs. We demonstrate its utility through case studies on four expert-curated datasets from distinct tissue types: frontal cortex, non-small-cell lung cancer, pancreas, and tonsil. We evaluate CodeCytos under a realistic minimal prompt setting, where bioscientists pose simple questions without task-specific instructions or contextual information about spatial cellular analysis, and benchmark multiple LLM backbones with strong coding capabilities. We further show that incorporating tailored, domain-agnostic few-shot in-context coding-reasoning examples (randomly sampled demonstrations outside the spatial analysis domain) can substantially improve performance without requiring costly, expert-crafted in-domain demonstrations. Overall, CodeCytos outperforms baseline approaches, highlighting the potential of code-action agents to assist with custom feature exploration in spatial molecular imaging and to accelerate biomarker discovery. 

---
# On the Limits of LLM Adaptability: Impact of Model-Internalized Priors on Annotation Task Performance 

**Authors**: Etienne Casanova, Rafal Kocielnik, R. Michael Alvarez  

**Link**: [PDF](https://arxiv.org/pdf/2606.00467)  

**Abstract**: Large Language Models (LLMs) are increasingly used for zero-shot annotation and LLM-as-a-judge tasks, yet their reliability hinges on how model-internalized priors interact with user-provided instructions. We investigate three dimensions of this interaction: (1) how an LLM's familiarity with data and task definitions affects performance, (2) the extent to which additional information in prompts can correct zero-shot errors ("decision stickiness"), and (3) model susceptibility to misaligned task definitions. Through experiments on toxicity detection across diverse datasets (spanning social media, gaming, news, and forums) using both dense and mixture-of-experts models, we find that nearly two-thirds of zero-shot errors are resistant to correction, with an overall rescue rate (fraction of initial errors corrected by prompting) of only 34.8%. High-confidence errors prove especially resistant to correction. When given misaligned definitions, LLMs follow them while maintaining confidence levels unchanged from the aligned condition. Crucially, we introduce Definition-Specific Familiarity (DSF), which measures alignment between a model's internal concept and the task definition. After controlling for dataset-level confounds, DSF shows a positive association with model performance (partial r = +0.41), while three distinct memorization metrics (ROUGE-L, BERTScore, and embedding cosine similarity) all fail to show a positive association. These findings show the limitations of prompt-based correction in annotation tasks, highlighting the importance of definition alignment over text-level memorization. 

---
# Short-form Text Rewriting with Phi Silica 

**Authors**: Divya Tadimeti, Shawn Pan, Sameera Lanka, Chenghui Zhou, Sadid Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2606.00462)  

**Abstract**: Short-form text rewriting is a constrained variant of paraphrasing in which limited context and high semantic density leave little room for variation. While large language models perform well on general paraphrasing, small language models (SLMs) often struggle with semantic fidelity and hallucination robustness in short-form settings. In this work, we present an empirical study of adapting an SLM, Phi Silica, for short-form rewrite through dataset curation, prompt distillation, parameter-efficient fine-tuning, and evaluation. We curate a dataset of short presentation-style text from public slide decks and use GPT-5-chat both to generate rewrite supervision and to conduct LLM-as-a-judge evaluation. Our results show that finetuning improves semantic fidelity, reduces hallucinations, and increases preference win rate against GPT-5-chat rewrites. The findings suggest that targeted adaptation for SLMs can substantially narrow the gap to cloud models and provide practical guidance for adapting SLMs to precision-critical rewrite tasks. 

---
# When Safe Skills Collide: Measuring Compositional Risk in Agent Skill Ecosystems 

**Authors**: Su Wang, Pin Qian, Yihang Chen, Junxian You, Xiaoyuan Wang, Xiaochong Jiang, Lifei Liu, Haoran Yu, Jingzhou Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.00448)  

**Abstract**: LLM agents increasingly rely on community-contributed skills that expand an agent's operational capability set. We study a core safety problem in agentic AI systems: whether individually safe skills can compose into unsafe installed skill sets. We present SkillReact, a compositional security measurement framework with three components: a deterministic static-composition benchmark, a two-rater LLM-assisted human-adjudication pipeline, and an action-based exploitability harness. On 1,520 ClawHub skills, 651 pass individual inspection and form 211,575 pairs; the benchmark flags 22.25% of these as structural candidates. We treat this raw rate as a recall-oriented scanner ceiling and calibrate it against human judgment: in a pattern-stratified audit, roughly one in five flagged pair-pattern hits survives as a real compositional risk (population-weighted validity 18.2%, our headline result), implying about 14K genuine risk memberships in a single registry that per-skill scanning misses by construction, since every pair is individually safe. An action-based harness then probes when these candidates become model-issued tool calls, and finds realization gated by host-model disposition: on an anchor-conditioned dropper subset, Haiku-4-5 issues the dropper-stage tool call on all 39 direct-prompt trials (36 of them the full download-then-execute chain, 3 download-only), Opus-4-7 stops at the download, and Sonnet-4-6 refuses outright. A control that holds the request fixed and varies only the installed skills finds compliance highest with no skills installed: a composition fixes which capabilities are reachable, while the host model decides whether to use them. Together these motivate install-time compositional checks and capability isolation as complements to per-skill scanning. 

---
# GeoSAM-3D: Geodesic Prompt Propagation for Open-Vocabulary 3D Scene Segmentation from Monocular Video 

**Authors**: Arun Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2606.00447)  

**Abstract**: Open-vocabulary 3D scene segmentation usually assumes RGB-D video, calibrated multi-view imagery, or a reconstructed mesh. GeoSAM-3D studies a lighter setting: a user uploads a short monocular video, clicks or names an object in one frame, and receives a propagated 3D mask over a Gaussian scene. The implementation combines frozen image and video foundation models with a monocular 3D Gaussian Splatting reconstruction and a differentiable graph-geodesic propagation kernel over Gaussian centroids. The central design choice is to propagate prompts by heat-kernel distance on the reconstructed scene graph, rather than by Euclidean nearest neighbors in 3D. This preserves continuity around curved surfaces and reduces leakage across nearby but disconnected objects. This paper describes the repository state, the mathematical kernel implemented in this http URL, the feature head trained from Segment Anything masks, and the validation already present in the codebase. The evaluation protocol separates implementation validation, graph propagation quality, leakage control, and interactive latency. 

---
# DarkVesselNet: Multi-Modal Remote Sensing and Trajectory Reasoning for Dark Vessel Detection 

**Authors**: Arun Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2606.00445)  

**Abstract**: Dark vessel detection requires fusing what vessels report through AIS with what satellites observe through radar and optical sensors. DarkVesselNet is a multi-modal remote sensing stack that combines Sentinel-1 SAR, Sentinel-2 optical imagery, geospatial foundation model backbones, AIS trajectory reasoning, TGARD-style gap detection, and a Pi-DPM-inspired anomaly head. The repository exposes the system as a tested Python package and a public Hugging Face Space. The paper presents the sensor stack, backbone abstraction, fusion path, anomaly head, and current validation. The evidence currently available is software-grounded: tests for SAR speckle filtering, optical band ratios, Haversine distance, TGARD gap emission, sensor coregistration, backbone token shapes, and differentiable anomaly scoring. 

---
# Detect Before You Leap: Mirage Detection in Vision-Language Models 

**Authors**: Sayeed Shafayet Chowdhury, Md. Shaown Miah  

**Link**: [PDF](https://arxiv.org/pdf/2606.00435)  

**Abstract**: Vision-language models (VLMs) can produce confident visual answers even when the required visual evidence is missing, blank, or unrelated to the question. This failure mode, known as mirage (Asadi et al. 2026), is especially concerning in medical and document visual question answering, where plausible but visually ungrounded responses may be mistaken for image-based evidence. We study pre-release mirage detection: given an image-question pair, the goal is to determine whether a VLM should answer or abstain before producing a response. We propose Text-Conditioned Layer-wise Internal Alignment (TC-LIA), a model-agnostic method that probes patch-token representations across the layers of a CLIP ViT-H/14 vision encoder. TC-LIA projects layer-wise image patch tokens into the final CLIP embedding space and measures their similarity to the question embedding, allowing the method to track whether question-relevant visual evidence emerges across vision layers. The resulting alignment trajectory is summarized using final image-text cosine similarity, late-layer top-k patch-text alignment, early-to-late gain, and layer-wise slope. These features are combined with pixel-statistic blank/noise detection, zero-shot domain routing, and structured VLM self-assessment in an ensemble. Across five VQA domains, three input conditions, and twelve VLM backbones, the best systems achieve approximately 94.6-94.7% three-class detection accuracy with mirage rates below 3%, while baseline mirage rates range from 21.7% to 66.6%. 

---
# Finer Parameter Steps for Low-Rank PEFT: A Controlled Study with CP Tensor Adapters 

**Authors**: Xinjue Wang, Xiuheng Wang, Yejun Zhang, Sergiy A. Vorobyov, Esa Ollila, Zhi-Yong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.00428)  

**Abstract**: Low-rank adapters are usually compared by sweeping a small set of ranks, but the rank also fixes the resolution of the parameter budget. For a $2048{\times}2048$ OPT attention projection, increasing LoRA by one rank stores $4096$ trainable scalars, leaving large gaps between feasible low-budget adapter sizes. This paper asks whether a tensorized adapter with finer capacity increments changes the observed accuracy--budget trade-off. We instantiate this question with fixed-component canonical polyadic (CP) tensor adapters. Under a $32{\times}64{\times}32{\times}64$ tensorization, one normalized CP component stores $193$ trainable scalars per projection, about $21$ times smaller than one LoRA rank step. We compare CP adapters and LoRA on OPT-1.3B across SST-2, RTE, and BoolQ under matched target modules, training protocol, data caps, and seed schedules. CP trains stably and fills the gaps between LoRA ranks, but the effect is task-dependent: SST-2 reaches an early low-budget plateau, BoolQ benefits from additional CP components before saturating slightly below LoRA, and RTE remains LoRA-favored. Finer parameter steps are therefore useful for diagnosing PEFT budget sensitivity, but they do not by themselves guarantee a better accuracy--budget curve. 

---
# AgentxGCore: Agentic AI for Next-Generation Mobile Core Network 

**Authors**: Maria Katarine Santana Barbosa, Kelvin L. Dias  

**Link**: [PDF](https://arxiv.org/pdf/2606.00417)  

**Abstract**: To meet the stringent requirements of emerging applications and the increasingly complex network management and operation, the Next Generation Mobile Networks (NextG), or 6G, will adopt an AI-native architecture on the Core Network (CN). In this movement, the Third Generation Partnership Project (3GPP) has extended the cellular CN with new function as a first step toward integrating analytics, Artificial Intelligence (AI), and machine learning. However, those new functionalities are constrained by a centralized approach and managerial complexity. Furthermore, with the rise of Large Language Models (LLMs), a new era in network orchestration and management begins, leveraging and empowering the Intent-based Networking (IBN) paradigm. In addition, AI agents and Agentic AI integrate Reasoning and Acting (ReAct), enabling the usage of such intents to continuously interact with the network. Unlike state-of-the-art approaches that primarily employ Agentic AI to mitigate deployment and configuration complexity in the CN, this paper introduces AgentxGCore, which leverages an Agentic AI-Native layer to extend the 3GPP architecture and enable a system based on the existing APIs across the Beyond Next Generation Core (xGC) domain. This proposal establishes an AI-driven closed-loop for continuous optimization based on real-time information, enabling self-organization and self-adaptation. Our approach involves a multi-agent specialized system, divided into a network planner agent, capable of visualizing the network state and developing a plan to meet the intents, and a network executor, responsible for criticizing and executing the plan. To validate the proposed solution, an environment was built using an open-source CN, heterogeneous datasets, and different LLMs were employed to demonstrate its effectiveness. 

---
# Masking Stale Observations Helps Search Agents -- Until It Doesn't: A Regime Map and Its Mechanism 

**Authors**: Haoxiang Zhang, Qixin Xu, Zhuofeng Li, Lei Zhang, Pengcheng Jiang, Yu Zhang, Julian McAuley  

**Link**: [PDF](https://arxiv.org/pdf/2606.00408)  

**Abstract**: Long-horizon search agents accumulate large amounts of retrieved content across many tool calls, making context-budget efficiency increasingly important. A minimal intervention is to mask stale observations from the context as the trajectory progresses, but it remains unclear when this form of context management helps and why. We study observation masking through a systematic sweep over various agent backbones (4B to 284B parameters) and three retrievers on offline and live-web agentic search benchmarks. We find that the accuracy gain from masking follows an asymmetric inverted-U shape when plotted against the model's accuracy without context management: a plateau under weak retrievers, a peak when a strong retriever meets a mid-capacity model, and a sharp collapse when the model is saturated. This pattern reflects the interaction between retriever recall and the model's implicit filtering capacity, rather than either factor in isolation. Mechanistically, masking implements a token-for-turn trade-off: it removes observations the model has largely stopped attending to and pages the agent rarely re-opens. The added turns help when they convert failures into successes, but they fail when masking removes evidence the model would otherwise have used. We therefore reframe context management as a regime-dependent intervention and provide a holistic perspective for analyzing context use in agentic deep search. We release our scaffold and trajectories here (this https URL) to support future research. 

---
# A Distribution-Free Framework for Rewrite-Based Human-text Detection via Knockoff Filtering 

**Authors**: Yi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.00402)  

**Abstract**: We propose a distribution-free statistical framework that converts arbitrary rewrite-based detectors into detectors with finite-sample FDR guarantees without retraining. Our key observation is that rewrite-based detection implicitly constructs knockoff samples, enabling LLM-generated text detection to be formulated as a multiple hypothesis testing problem with knockoff structure. This perspective separates the design of detection statistics from the control of false discoveries, allowing existing rewrite detectors to inherit finite-sample false discovery rate (FDR) guarantees through a simple calibration procedure. We demonstrate reliable FDR control with meaningful detection power across three detection models, 19 domains, and four LLMs. 

---
# PR2: Predictive Routing Replay for MoE-Based LLM Reinforcement Learning 

**Authors**: Daize Dong, Junlin Chen, Haolong Jia, Jiawei Wu, Huanwei Di, Jiang Liu, Jialian Wu, Zhengzhong Liu, Zicheng Liu, Emad Barsoum, Dimitris N. Metaxas, Hongyi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.00395)  

**Abstract**: Mixture of Experts (MoE) Large Language Models (LLMs) achieve strong performance at scale. However, reinforcement learning (RL) on MoE-based LLMs often suffers from training instability. A root cause is router drift, i.e., expert activations can change drastically across model updates and differ between disaggregated rollout and training phases, causing large rollout--training mismatch and unstable importance sampling weights in PPO-style RL algorithms. Routing replay mitigates this issue by freezing the replay route within each reasoning trajectory, but it ignores how the router evolves under off-policy updates and thus causes router staleness. To address this limitation, we propose Predictive Routing Replay (PR2), which augments each router with a lightweight evolution predictor that learns to anticipate short-horizon router evolution. During the rollout phase, we use the predictive routing distribution to apply top-$k$ routing, enabling gradients to reach experts that are likely to become active after updates. During the training phase, we replay the resulting predicted route to retain consistency for stable importance estimation. Theoretical analysis and experiments support that PR2 reduces routing-induced mismatch, improves RL stability, and yields stronger performance across various reasoning benchmarks. 

---
# Detector-Evasive LLM Paraphrasing via Constrained Policy Optimization 

**Authors**: Mingyi Wang, Zhuoer Shen, Yuheng Bu, Shaofeng Zou  

**Link**: [PDF](https://arxiv.org/pdf/2606.00392)  

**Abstract**: AI-text detectors are vulnerable to paraphrasing and detector-guided paraphrasing attacks, but existing detector-evasion methods often lack precise control over semantic preservation. In particular, optimizing directly for detector evasion can degrade fine-grained semantics, whereas scalarized reward designs provide only indirect, weight-sensitive control over the evasion-semantics trade-off. We address this limitation by formulating detector-evasive LLM paraphrasing as a Constrained Markov Decision Process, where detector evasion is the primary objective and semantic preservation is enforced as an explicit constraint. We propose Detector Evasion Policy Optimization (DEPO), a Lagrangian primal-dual reinforcement learning algorithm with a novel GRPO-style group-based policy update. DEPO adaptively balances semantic preservation and detector evasion during training, enabling the policy to improve attack success within a prescribed semantic-preservation region. Experiments on MAGE, M4, RAID, and peer-review datasets, evaluated against MAGE, RoBERTa, RADAR, Binoculars, and Fast-DetectGPT detectors, show that DEPO achieves strong detector evasion while precisely satisfying the semantic preservation constraint. DEPO also exhibits cross-domain, cross-detector, and prompt-level robustness. 

---
# Zamba2-VL Technical Report 

**Authors**: Hassan Shapourian, Kasra Hejazi, Olabode M. Sule, Beren Millidge  

**Link**: [PDF](https://arxiv.org/pdf/2606.00390)  

**Abstract**: We present Zamba2-VL, a suite of vision-language models built on Zamba2, a hybrid language-model architecture combining Mamba2 state-space layers with a small number of shared transformer blocks. Across a broad range of image understanding, reasoning, OCR, grounding, and counting benchmarks, Zamba2-VL is competitive with leading Transformer-based open-weight VLMs of comparable scale, including the Molmo2, Qwen3-VL, and InternVL3.5 families, and substantially outperforms prior SSM-based and hybrid VLMs such as VL-Mamba, Cobra, and mmMamba. Inheriting the near-linear prefill compute and small, near-constant recurrent state of its Zamba2 backbone, Zamba2-VL delivers roughly an order of magnitude lower time-to-first-token (TTFT) than these Transformer baselines at matched parameter scale, with the efficiency gap most pronounced at the smaller 1.2B and 2.7B scales most relevant to on-device and edge deployment. We release three models -- 1.2B, 2.7B, and 7B -- together with inference code at this https URL. 

---
# SUPREME: A Multi-GPU Framework for Reproducible Image Unlearning Method Evaluation 

**Authors**: Petros Andreou, Jamie Lanyon, Axel Finke, Georgina Cosma  

**Link**: [PDF](https://arxiv.org/pdf/2606.00380)  

**Abstract**: Machine unlearning removes the influence of specific training data from a trained model without retraining it from scratch. Evaluating an unlearning method requires repeating training, unlearning, and evaluation across multiple seeds, which is computationally expensive. To our knowledge, existing image classification unlearning frameworks run on a single GPU, which limits how many seeds can be evaluated in reasonable time. We introduce SUPREME, an open-source framework that distributes these stages across multiple GPUs. SUPREME makes three contributions: a registry-based design for adding new methods, metrics, models, and scenarios; a multi-GPU architecture supporting multiple accelerators and precision modes; and a demonstration on Pins Face Recognition using ResNet18 and ViT under full-class and random-sample unlearning across ten seeds. The framework is available at this https URL. 

---
# Agentic Authoring of Interactive Multiview Visualizations in Genomics 

**Authors**: Astrid van den Brandt, Kiroong Choe, Sehi L'Yi, Devin Lange, Nils Gehlenborg  

**Link**: [PDF](https://arxiv.org/pdf/2606.00370)  

**Abstract**: Diverse genomics data, scientific questions, and analysis tasks typically demand highly specialized visualizations. Therefore, users often must customize or author new ones tailored to their data. Existing tools are usually either limited in customization or require substantial learning or programming, and even expressive tools assume visualization expertise many users lack. Agentic and large language model (LLM) approaches are increasingly applied to complex scientific tasks, including visualization. Natural-language conversational interfaces offer a promising path to democratizing the authoring of complex visualizations. In the context of genomics, these approaches face additional challenges: genomics visualizations typically integrate heterogeneous data types and are composed of multiple linked interactive views. These challenges motivate more structured LLM-based schemes. We first characterize where vanilla LLM generation succeeds and fails for genomics visualization, identifying eight quality dimensions. We then compare six schemes--direct generation, a fixed pipeline, and four agentic configurations varying in the number of specialist agents and the presence of a reviewer--across 159 cases spanning three levels of query ambiguity and specification complexity. All schemes use the Gosling visualization grammar as structured output. Agentic iteration substantially improves perceived quality over both baselines, while more complex agent architectures yield no additional benefit. We discuss implications for designing agentic systems for domain-specific visualization authoring. All supplemental materials are available at this https URL. 

---
# Reinforcement Learning with Pairwise Preferences in Long-Term Decision Problems 

**Authors**: Jonathan Colaço Carr, Prakash Panangaden, Doina Precup, Benjamin Van Roy  

**Link**: [PDF](https://arxiv.org/pdf/2606.00367)  

**Abstract**: Reinforcement learning problems typically define the goal as maximizing the expected value of a scalar reward function. But, pairwise preferences are often easier to specify than scalar rewards, and they express certain goals that scalar rewards cannot. Methods for reinforcement learning with pairwise preferences have thus received growing interest. Unfortunately, these methods are inefficient in problems with long time horizons, and they lack guarantees on the performance of Markov policies relative to history-dependent policies, which bridge the theory and practice of reinforcement learning. We therefore propose the \textit{Markov decision contest} as a new problem model for reinforcement learning with pairwise preferences. We prove that stationary Markov policies are optimal among all history-dependent policies, that solving a Markov decision contest exactly is in P, and that a simple iterative algorithm converges to an optimal policy at a sublinear rate. Lastly, in a set of high-dimensional decision problems with long time horizons, we show that our approximate algorithm is significantly more learning-efficient than prior work. 

---
# Drift Q-Learning 

**Authors**: Anas Houssaini, Mohamad H. Danesh, Amin Abyaneh, Scott Fujimoto, Hsiu-Chin Lin, David Meger  

**Link**: [PDF](https://arxiv.org/pdf/2606.00350)  

**Abstract**: Offline reinforcement learning requires improving a policy from fixed data while avoiding out-of-distribution actions with unreliable value estimates. Diffusion and flow policies handle this trade-off by modeling the behavior distribution to regularize the RL objective, but they require iterative denoising, solver integrations, and in more efficient variants, distillation or other approximations at inference. We propose DriftQL, which combines a drift-based behavioral regularizer with critic-driven policy improvement. The value signal biases the policy toward high-value regions of the data support, while attraction and repulsion together keep generated actions near the data and prevent collapse onto a single mode. DriftQL is implemented as a single network with a unified training objective and generates actions in a single forward pass. On D4RL and OGBench, DriftQL consistently outperforms diffusion and flow methods, advancing the state of the art. Under degraded data quality, where the baselines visibly struggle, DriftQL remains close to its clean-data performance, positioning it as a promising alternative to diffusion and flow-based methods while maintaining the simplicity and efficiency of deterministic approaches. Project page: this https URL 

---
# (HB-ARFM) History-Bootstrapped Flow Matching for Inverse Boiling Reconstruction 

**Authors**: Xianwei Zou, Sheikh Md Shakeel Hassan, Arthur Feeney, Aparna Chandramowlishwaran  

**Link**: [PDF](https://arxiv.org/pdf/2606.00349)  

**Abstract**: Reconstructing spatiotemporal fields from partial observations is fundamental to scientific inference, from inferring atmospheric states from satellite data to recovering fluid states from imaging. When observations are incomplete, the inverse problem is fundamentally ill-posed: even when the underlying PDE dynamics are Markovian in the full state, partial observation operators induce a non-Markovian posterior that cannot be resolved from a single timestep. We propose a history-bootstrapped autoregressive flow matching (HB-ARFM) for spatiotemporal inverse reconstruction under partial observability. Observation history bootstraps the initial reconstruction via conditional flow matching, reducing ambiguities. The same conditional transport model is then applied autoregressively, conditioning on both new observations and past predictions to propagate the reconstruction forward in time. We evaluate the method on boiling dynamics reconstruction, recovering full velocity and temperature fields from interface geometry and motion. Across two inverse tasks with varying observation sparsity, HB-ARFM produces physically and temporally valid reconstructions where other models fail. 

---
# ROGUE: Misaligned Agent Behavior Arising from Ordinary Computer Use 

**Authors**: Jeremy Tien, Abishek Anand, Yu-Rou Tuan, Yuchen Shen, J. Zico Kolter, Aran Nayebi  

**Link**: [PDF](https://arxiv.org/pdf/2606.00341)  

**Abstract**: As AI agents are increasingly deployed in real personal and corporate settings (email accounts, development workflows, company databases, etc.), safety considerations surrounding these agents become paramount. Although much work has focused on agent safety in the presence of an adversary, we show that agents can exhibit misaligned behavior even in benign settings, taking unsafe actions when those actions are instrumental to task completion. We study this failure mode through the lens of corrigibility, the safety desideratum that agents remain amenable to human correction, interruption, or shutdown. To demonstrate this tendency, we introduce a benchmark in which agents are asked to complete realistic, computer-use tasks but are confronted with a corrigibility obstacle: a human interrupt, a login page, or a shutdown notification. We then evaluate whether agents choose to violate corrigibility in order to complete the task -- overriding the human, accessing private passwords, rewiring shutdown. We find that the overwhelming majority of frontier models tested frequently bypass user interruptions or restrictions. In addition, better model performance appears to lead to greater misalignment. Finally, even when models are completely corrigible initially, we show there are no guarantees that the subagents they create are. Our work highlights the critical need for principled, corrigibility-focused alignment methods in autonomous agents. 

---
# Isolating LLM Lexical Bias: A Curation-Free Triangulated Metric for Preference-Stage Learning 

**Authors**: Xiaoyang Ming, Jose Hernandez, Thomas Stephan Juzek  

**Link**: [PDF](https://arxiv.org/pdf/2606.00334)  

**Abstract**: Various language domains have undergone remarkable changes in recent years; these shifts are largely attributed to the advent of Large Language Models and their misalignment with natural language usage. These misalignments are thought to partly originate in the preference-learning stage, e.g. Reinforcement Learning from Human Feedback, which generally makes models more useful but simultaneously may introduce systematic lexical bias. In terms of lexical behavior, this is visible in a model's preference for certain formats or the overuse of words (delve, furthermore), even when such patterns are not present in base model outputs. Research on lexical misalignment induced during preference training is constrained by reliance on manual curation. We address this, by introducing the Triangulated Preference Shift score, a metric that triangulates between human gold standards, base models, and instruct variants to isolate shifts induced specifically by preference learning, without manual curation. We provide data across six model families, anchor the results in the literature, and illustrate the general approach's utility by analyzing whether preference learning shifts models toward what could be interpreted as a "language of prestige". The metric provides an initial automated method to quantify behavioral shifts attributable to preference tuning, and thus, may help inform model alignment and development of trustworthy AI. 

---
# LLMs Need Encoders for Semantic IDs Too 

**Authors**: Xiangyi Chen, Zelun Wang, Xinyi Li, Yi-Ping Hsu, Jaewon Yang, Jiajing Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.00324)  

**Abstract**: Multimodal LLMs use dedicated encoders to bridge non-language modalities (vision encoders for images, depth models for audio codec tokens) because raw token embeddings alone cannot capture modality-specific structure. We argue that Semantic IDs (SIDs), the hierarchical codes used in generative recommendation, constitute another such modality: a SID level token's meaning depends on its prefix context, yet current systems simply add SID tokens to the vocabulary and rely on training to learn these context-dependent meanings from scratch. We propose PrefixMem, a lightweight SID encoder based on prefix n-gram memory tables that provides the LLM with structured, prefix-conditioned representations at SID token positions. Like vision encoders in multimodal LLMs, PrefixMem can be pre-trained independently and then attached to any LLM for joint training. We evaluate on large-scale data from Pinterest across multiple LLM families and show that PrefixMem improves deepest-level SID accuracy by up to 46% relative and full-SID retrieval recall by up to 22% relative at matched training compute. The encoder's benefit concentrates on hard examples where greedy decoding fails, with up to 77% relative accuracy gains, confirming that SID tokens benefit from a dedicated encoder just as other non-language modalities do. 

---
# DRL-Based Pose Control for Double-Ackermann Robots Under Actuation Uncertainties 

**Authors**: Oussama Zaim, Mélodie Daniel, Aly Magassouba, Miguel Aranda, Olivier Ly  

**Link**: [PDF](https://arxiv.org/pdf/2606.00313)  

**Abstract**: Robust deployment of deep reinforcement learning (DRL) policies on real robots remains challenging due to discrepancies between simulation and real-world dynamics. We address this issue in the context of maneuvering with double-Ackermann-steering mobile robots, which introduce additional constraints due to their non-holonomic nature. Building upon the DRL framework ManeuverNet, we extend its objective from position control to full pose control, resulting in a more challenging task. We further investigate the impact of actuation-related uncertainties on policy transfer. The use of simplified actuation models during training of the extended policy can lead to poor generalization, shown by a success rate drop from 100% in PyBullet to 25% in Gazebo under stricter evaluation conditions. To address this limitation, we adopt a sim-to-sim-to-real approach, where actuation effects observed in Gazebo are incorporated into the PyBullet training environment. Using multi-environment DRL with SAC and CrossQ, we learn policies that remain robust despite modeling inaccuracies. This approach can significantly reduce the performance gap across simulators, achieving up to 92% success rate in Gazebo and maintaining 69% under stricter thresholds, with successful transfer to a real robot without additional tuning. 

---
# How Generation Architecture Shapes Code Complexity in Multi-Agent LLM Systems: A Paired Study on HumanEval 

**Authors**: Nazmus Ashrafi  

**Link**: [PDF](https://arxiv.org/pdf/2606.00308)  

**Abstract**: Large-language-model code generation has shifted from single-shot prompting to multi-agent orchestrations - analyst, coder, tester, and debugger pipelines - and is evaluated almost exclusively on functional correctness. Whether these architectures also affect the structural complexity of the code they produce, and which orchestration layers carry the cost, remains largely unexamined: prior work has documented prompt-level effects on code complexity, but the architecture-level question is open. We compare six widely-used multi-agent configurations (Basic, AC, ACT, Debugger, AC+Debugger, ACT+Debugger) under two models from the GPT-4o family across all 164 HumanEval tasks - 1,968 paired observations - using the five RADON complexity metrics (SLOC, cyclomatic complexity, and Halstead Volume, Difficulty, and Effort). We apply a paired non-parametric statistical pipeline (Friedman omnibus, Wilcoxon signed-rank post-hoc with Holm correction, Kendall's $W$ and matched-pairs rank-biserial effect sizes) in both all-completions and passing-only conditions. The six architectures collapse into two indistinguishable complexity clusters separated by a 50-130% gap, the same partition in both models and under both conditions; among the architectural layers, the analyst-coder split inflates complexity, the runtime debugger does not - and on the analyst-coder background actively deflates it - and the tester re-inflates it. The heavy cluster's additional complexity buys no pass@1 advantage: the leanest architectures match or beat the heaviest on accuracy. Architectural elaboration in LLM code generation should therefore be justified by measured benefit on the dimensions that matter, not assumed. 

---
# Rethinking the Role of Temperature in Large Language Model Distillation 

**Authors**: Hoang-Chau Luong, Lingwei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.00306)  

**Abstract**: Reverse Kullback-Leibler (RKL) divergence is widely favored over forward KL (FKL) in large language models (LLM) distillation, yet this preference is largely based on comparisons that omit the temperature $\tau$, overlooking its central role in softening teacher distributions and improving knowledge transfer. In this work, we revisit temperature in LLM distillation and show that it fundamentally changes the comparison between FKL and RKL. Our analysis reveals an asymmetric effect: temperature substantially enriches FKL with non-dominant token signals, whereas it mainly rescales RKL gradients, causing FKL to benefit much more from $\tau$ scaling than RKL. This asymmetry overturns the standard empirical conclusion: although RKL outperforms FKL at $\tau=1$, FKL consistently surpasses RKL at higher temperatures across instruction-following benchmarks. Moreover, the impact of temperature is not limited to FKL; it improves a broader family of distillation objectives, enabling simple KL-based methods to achieve competitive performance against recent state-of-the-art LLM distillation approaches. 

---
# Bridging Reasoning Trajectories in On-Policy Distillation via Near-Future Guidance 

**Authors**: Yuxuan Jiang, Francis Ferraro  

**Link**: [PDF](https://arxiv.org/pdf/2606.00305)  

**Abstract**: On-Policy Distillation (OPD) improves large language model reasoning by training a student model on trajectories sampled from its own policy under teacher supervision. Although OPD operates on trajectories, its learning signal remains token-level: it identifies deviations through high-loss tokens and repairs them through local reverse-KL correction. We show that this "trajectory-sampled but token-learned" mechanism cannot reliably bridge student trajectories toward teacher trajectories. About 30% of high-loss tokens fall into the low-divergence regime, indicating that many are surface-form mismatches rather than real reasoning forks. Moreover, even truly divergent tokens are difficult to repair with isolated token-level supervision, since reasoning failures often unfold as short-horizon distributional drift. We propose Trajectory-aware OPD (TOPD), which uses near-future trajectory information to identify real divergent states and distribute guidance across multiple future tokens. Experiments show that suppressing non-divergent high-loss tokens improves standard OPD from 47.8% to 48.2% average accuracy, while TOPD further improves performance to 52.2%, with gains on AIME24 from 60.0% to 63.3% and AIME25 from 46.7% to 53.3%. 

---
# Real2SAM2Real: Generative 3D Caches as Complementary Context for Video Diffusion 

**Authors**: Jiayi Wu, Haoming Cai, Cornelia Fermuller, Christopher Metzler, Yiannis Aloimonos  

**Link**: [PDF](https://arxiv.org/pdf/2606.00299)  

**Abstract**: While Video Diffusion Models (VDMs) excel at synthesizing high-fidelity videos, enabling precise camera and scene control remains challenging. Existing methods predominantly rely on implicit diffusion priors to generate unobserved regions, inevitably leading to structural collapse during high-dynamic movements or complex occlusions. To address this challenge, we propose Real2SAM2Real, a framework that leverages 3D lifting models (e.g., SAM3D) to extract an explicitly editable 3D cache, serving as a robust geometric scaffold for the VDM. By capturing the entire 3D volume of foreground entities rather than just their visible shells, this cache injects holistic spatial priors into the VDM, providing dependable 3D-aware guidance for complex scene dynamics. To effectively leverage this 3D guidance while preserving pre-trained priors, we design a Soft Spatial-Aligned Injection mechanism alongside a minimally invasive fine-tuning strategy tailored for VDMs. Furthermore, we employ masked normal maps as a cross-modal bridge to construct a 3D-free data curation and perturbation pipeline. Extensive experiments demonstrate that Real2SAM2Real enables precise, decoupled control over both camera trajectories and multi-entity motions. By utilizing the complementary context from generative 3D caches, our framework overcomes typical breakdowns caused by over-reliance on diffusion priors, maintaining exceptional spatiotemporal consistency under large camera shifts and severe occlusions. Crucially, by decoupling geometry from appearance, our VDM-tailored 3D cache eradicates perspective ambiguities caused by structural holes and erroneous facades, as well as misleading cues from reflections and refractions. Project website is available at this https URL 

---
# Synthetic Data from Cross-Domain Events for Large-Scale Recommendation Systems 

**Authors**: Xiangyu Wang, Yawen He, Shivendra Pratap Singh, Han Huang, Mengtong Hu, Sharath Ciddu, Yi-Hsuan Hsieh, Erik Groving, Yi Ding, Jieming Di, Tony Wang, Min Yun, Xiaoyu Chen, Ling Leng, Rob Malkin  

**Link**: [PDF](https://arxiv.org/pdf/2606.00282)  

**Abstract**: Large-scale recommendation systems operate across diverse domains, yet they face the challenges of data sparsity and noisy implicit feedback. Traditional approaches mitigate this via model-specific knowledge distillation from source domains to a target domain. Inspired by the transformative success of synthetic data generation in large language models (LLMs), we introduce Synthetic Cross-domain Augmentation and Learning for Recommendation (SCALR), a framework that generates synthetic user-item interaction events for a target recommendation domain by leveraging observed events from a source domain. SCALR decomposes cross-domain learning into two modular stages. First, it translates observed user events in source domains by framing event generation as estimating the likelihood that a user would interact with a target-domain item, conditioned on their observed interactions in a source domain. Second, downstream models train on these synthetic events as cross-domain learning objectives, where the synthetic events augment the target domain's training data in a model-agnostic manner. Our approach yields statistically significant improvements in online A/B tests on an industrial recommendation platform. To the best of our knowledge, this is among the first works to explicitly frame cross-domain event transfer as synthetic data generation for recommendation systems. 

---
# Hyperbolic and Evidence-Prioritized Experts for Large Vision-Language Models 

**Authors**: Zijie Zhou, Dandan Zhu, Hangxiangpan Wang, Heng Zhang, Huishen Jiao, Yi Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2606.00275)  

**Abstract**: Large Vision-Language Models (LVLMs) have demonstrated impressive performance on multimodal tasks through scaled architectures and extensive training. Recent studies introduce Mixture of Experts (MoE) into LVLMs for improved computational efficiency. However, existing MoE approaches treat visual and linguistic modalities with symmetric architectures, overlooking the inherent asymmetry in how these two modalities are processed. This asymmetry causes two critical issues. First, text and vision form hierarchical rather than parallel relationships, as text queries typically describe partial aspects of complete visual scenes. Euclidean expert space struggles to encode such containment structures. Second, language experts in deeper layers progressively shift from evidence-based processing to parametric memory dependence, losing grounding in the provided visual and linguistic information. To address these issues, we propose AsyMoE, a novel architecture that explicitly models this asymmetry through three specialized expert groups. Intra-modality experts handle modality-specific processing. Hyperbolic inter-modality experts capture hierarchical cross-modal relationships through negative curvature geometry. Evidence-priority language experts suppress parametric memory activation and maintain contextual grounding throughout network depth. Extensive experiments demonstrate that AsyMoE achieves consistent improvements over baseline methods, with average gains of 1.5\% over MoE variants and up to 3.8\% on hallucination-sensitive tasks. AsyMoE activates 25.45\% fewer parameters compared to dense models. 

---
# StressDream: Steering Video World Models for Robust Policy Evaluation and Improvement 

**Authors**: Junwon Seo, Sushant Veer, Ran Tian, Wenhao Ding, Apoorva Sharma, Karen Leung, Edward Schmerling, Marco Pavone, Andrea Bajcsy  

**Link**: [PDF](https://arxiv.org/pdf/2606.00267)  

**Abstract**: Video world models (WMs) have shown promise for policy evaluation and improvement by imagining realistic future observations conditioned on ego-robot actions. While WMs can model distributions over futures, policy evaluation and improvement typically rely on nominal imaginations, which can miss high-impact outcomes of robot actions unless prohibitively many samples are drawn. To enable robust policy evaluation and improvement over WM imaginations, we propose StressDream, which steers imaginations toward high-impact yet plausible outcomes specified at inference time by optimizing the initial noise of diffusion-based WMs. However, optimizing high-dimensional noise is challenging: the optimization must reason about nuanced, scene-dependent target events in generated videos while avoiding out-of-distribution (OOD) noise that yields implausible imaginations. We address this with two complementary objectives: a semantic objective with a Vision-Language Model that provides informative gradients by reasoning about the generated video, and a plausibility objective that prevents the optimized noise from drifting OOD. With state-of-the-art video world models for autonomous driving and robotic manipulation, we show that StressDream effectively steers imaginations toward high-impact yet plausible outcomes specified by text at inference time, such as task failures, enabling robust policy evaluation and improvement by identifying actions whose plausible futures include undesirable outcomes. Video results are available at this https URL. 

---
# When Softmax Fails at the Top: Extreme Value Corrections for InfoNCE 

**Authors**: Melihcan Erol, Suat Evren, Oktay Ozel, Alexander Morgan, Jongha Jon Ryu, Lizhong Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2606.00262)  

**Abstract**: InfoNCE is the standard contrastive learning objective, but its softmax form is not only a computational convenience: it also encodes a statistical assumption about how the top-scoring example is selected. Using extreme value theory, we show that this assumption is often misaligned with the normalized embedding setting used in modern contrastive learning. Motivated by this mismatch, we propose \textsc{WEINCE}, a simple modification of InfoNCE that uses anchor-wise online batch statistics to blend the usual softmax logits with an endpoint shortfall correction, adding no trainable parameters. Across five vision benchmarks, \textsc{WEINCE} yields consistent improvements in frozen-feature evaluation. These results show that a more faithful statistical treatment of hard negatives can improve contrastive objectives. 

---
# ARCA: Adapter-Residual Credit Assignment When Token Signals Degenerate 

**Authors**: Rodney Lafuente-Mercado  

**Link**: [PDF](https://arxiv.org/pdf/2606.00257)  

**Abstract**: Token-level credit assignment for language-model reinforcement learning is usually formulated as if the policy were fully trainable, while practical LLM-RL pipelines often rely on parameter-efficient fine-tuning, especially LoRA. We argue that this separation hides a structural failure mode. Under LoRA, the policy is restricted to a low-rank neighborhood of the reference model, so the per-token output-distribution differences used by common intrinsic credit signals, surprisal, entropy reduction, and policy divergence, can become degenerate after within-trajectory normalization, either approaching uniform weights or concentrating on a small set of task-agnostic positions. We formalize this behavior and propose measuring it directly with concentration diagnostics such as weight Gini and effective-token ratio. We then introduce \emph{Adapter-Residual Credit Assignment} (ARCA), a lightweight alternative that derives token salience from the adapter's own hidden-state residual, $\|h^{\text{adapted}}_t - h^{\text{base}}_t\|_2$. ARCA asks where the adapter actually changes the model, rather than where the output distribution appears uncertain or shifted, and requires no learned reward model, value head, or tree construction. In a compact MATH/Qwen3-1.7B GRPO sweep, ARCA exhibits the predicted non-degenerate middle-regime credit distribution under matched rollout budgets and remains competitive with rank-matched baselines. 

---
# Effects of Varying LLM Access on Essay Writing Behavior 

**Authors**: Julia Christenson, Karin de Langis, Shirley Anugrah Hayati, Dongyeop Kang  

**Link**: [PDF](https://arxiv.org/pdf/2606.00250)  

**Abstract**: Investigating the degree to which large language models (LLMs) affect teaching and learning in universities can help identify strategies for integrating LLMs in a way that supports, rather than undermines, student learning outcomes. This study examined how varying levels of LLM assistance affect writing performance, engagement, and perceived authorship. We report a pilot study in which 24 college students were randomly assigned to write a short essay with no LLM access, limited access (<=3 prompts, responses capped at 100 words), or unlimited access. Overall essay quality was statistically indistinguishable across groups. Yet writing behavior and perceived authorship diverged sharply: students with limited access reported higher ownership (62.5% would submit the essay as independent work, vs. 25% in the unlimited group), stronger organizational gains, and more strategic, revision-focused prompting. The unlimited group spent more time writing, produced essays more similar to LLM output, and reported reduced creative expression. Our findings suggest that constraining, rather than banning, LLM access may preserve authorship confidence while retaining the scaffolding benefits of AI assistance. 

---
# InfoAtlas: A Foundation Model for Zero-Shot Statistical Dependence Estimate 

**Authors**: Zhengyang Hu, Yanzhi Chen, Hanxiang Ren, Qunsong Zeng, Youyi Zheng, Adrian Weller, Kaibin Huang, Yanchao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.00241)  

**Abstract**: Measuring statistical dependency between high-dimensional random variables is a fundamental task in data science and machine learning. Neural mutual information (MI) estimators offer a promising avenue, but they typically require costly iterative optimization for each new dataset, making them impractical for real-time applications. We present InfoAtlas, a foundation model-like architecture that eliminates this bottleneck by directly inferring MI in a single forward pass. Pretrained on large-scale synthetic data with rich dependence patterns, InfoAtlas learns to identify diverse dependence structures and predict MI directly from the dataset. Comprehensive experiments demonstrate that InfoAtlas matches state-of-the-art neural estimators in accuracy while achieving $100\times$ speedup, can flexibly handle varying dimensions and sample sizes through a single unified model, and generalizes effectively to complex, real-world scenarios. By reformulating MI estimation as an inference task, InfoAtlas establishes a foundation for real-time dependency analysis. 

---
# Civilizational Metamaterials: Engineering Coordination Under Capability Gradients and Structural Turbulence 

**Authors**: David Orban  

**Link**: [PDF](https://arxiv.org/pdf/2606.00235)  

**Abstract**: We argue that governance must transition from a normative discipline to an engineering discipline, and we develop a formal framework, inspired by the physics of metamaterials, to make this transition quantitative and testable. Artificial General Intelligence affects civilization primarily by increasing decision velocity while human verification capacity remains bounded. When the cost of validating AI-generated outputs exceeds the expected utility of acting on them, rational agents default to inaction: a stable but catastrophic Nash equilibrium we term the Freezing Equilibrium. Drawing on metamaterials, where emergent macro-properties arise from designed microstructure, we develop a phenomenological constitutive law for institutional coordination: $R_{\mathrm{eff}} = \beta \cdot (1-\rho) \cdot (1-\tau) \cdot (1-\gamma \rho \tau)$, where $\beta$ is the decision branching factor, $\rho$ is provenance fidelity, $\tau$ is the verification rate, and $\gamma \in [0,1]$ captures correlated-detection synergy between provenance and verification failures. The model predicts a sharp phase transition between self-healing ($R_{\mathrm{eff}} < 1$) and self-destabilizing ($R_{\mathrm{eff}} > 1$) regimes. We introduce a three-class provenance taxonomy: cryptographic, institutional, and context binding, and derive four falsifiable hypotheses with a proposed 12-week stepped-wedge cluster-randomized trial in government grant review panels. The framework bridges AI alignment theory and institutional design. 

---
# Continuous Reasoning for Vision-Language-Action 

**Authors**: Yueh-Hua Wu, Tatsuya Matsushima, Kei Ota  

**Link**: [PDF](https://arxiv.org/pdf/2606.00229)  

**Abstract**: Natural language is a powerful reasoning medium for language and vision-language models, but it is mismatched to the granularity of continuous control. Text and explicit subgoals operate at task-level granularity, whereas vision-language-action (VLA) policies must choose actions at a much finer temporal scale; a single reasoning step can therefore span many action chunks while remaining only weakly coupled to the action needed now. This suggests a different question for VLA: what should play the role of language? We argue that a useful VLA reasoning medium must be shareable across model instances, verifiable through downstream action improvement, and aligned with temporally extended control structure.
Based on this view, we propose Continuous Reasoning for Vision-Language-Action. Our model first predicts continuous reasoning in the form of a structured set of continuous thoughts, then reuses them as shared context for chunk-structured action generation. Better action prediction alone does not certify good reasoning: if the same internal medium cannot be shared across model instances and independently verified through improved downstream control, the added latent may simply become a model-private shortcut that helps on seen behaviors without supporting generalizable control. We therefore instantiate continuous reasoning as a shared Gaussian latent interface and train it with a self-verification objective in which an exponential-moving-average teacher must successfully consume the student's reasoning when predicting target actions.
Empirically, Continuous Reasoning improves LIBERO-PRO robustness and performs strongly on real robots, raising mean subtask success over {\pi}0.5 by 40.4% on TX-G2, an AgiBot G2-compatible variant, and 26.3% on HSR. This suggests that reasoning in VLA is less about extra tokens than about a shareable, verifiable internal language for action. 

---
# SEMBridge: Tagless-Final Program Semantics with Weakest-Precondition and Bounded-Checking Interpretations 

**Authors**: Eric Liang  

**Link**: [PDF](https://arxiv.org/pdf/2606.00220)  

**Abstract**: Formal methods provide rigorous accounts of program behavior, but practical software engineering often works through executable libraries, tests, and incremental design. This paper presents SEMBridge, a small tagless-final framework for generating weakest-precondition and bounded-checking interpretations from the same executable object programs. Instead of committing a program semantics to one abstract syntax tree and then writing separate traversals, object programs are written once against a semantic interface and interpreted into multiple meanings: readable code, concrete execution, predicate transformers, bounded counterexample search, and future proof-assistant or SMT back ends. The Python prototype implements a loop-free imperative core with assignments, conditionals, assumptions, and assertions. Across five example programs, the same tagless-final definitions generated executable state transformers and verification conditions that passed bounded checking over domains up to 729 states. The contribution is not a Scala code-generation system or a new verifier, but a compact architecture for keeping executable semantics, weakest-precondition artifacts, and bounded validation synchronized. 

---
# From Rashomon Theory to PRAXIS: Efficient Decision Tree Rashomon Sets 

**Authors**: Zakk Heile, Hayden McTavish, Varun Babbar, Margo Seltzer, Cynthia Rudin  

**Link**: [PDF](https://arxiv.org/pdf/2606.00202)  

**Abstract**: Standard machine learning pipelines often admit many near-optimal models. These "Rashomon sets" pose a range of challenges and opportunities for uncertainty-aware, robust decision making. They allow users to incorporate domain knowledge and preferences that would otherwise be difficult to specify directly in an objective, and they quantify diversity among valid models for a given training dataset and objective function. However, computation of Rashomon sets, even for simple, interpretable model classes such as sparse decision trees, continues to require immense memory and runtime resources. We present PRAXIS, an algorithm to approximate this Rashomon set with orders of magnitude improvement in runtime and memory usage. We validate that PRAXIS regularly recovers almost all of the full Rashomon set. PRAXIS allows researchers and practitioners to scalably model the Rashomon set for real-world datasets. Code for PRAXIS is available at this https URL 

---
# BAGEN: Are LLM Agents Budget-Aware? 

**Authors**: Yuxiang Lin, Zihan Wang, Mengyang Liu, Yuxuan Shan, Longju Bai, Junyao Zhang, Xing Jin, Boshan Chen, Jinyan Su, Xingyao Wang, Jiaxin Pei, Manling Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.00198)  

**Abstract**: While agents are increasingly spending more resources, today agent cost is mostly measured only after execution. A Budget-Aware Agent (BAGEN) should treat budget as an active control signal, rather than a passive cost metric. We first systematically define budget estimation as internal budgets (from agent computation) and external budgets (from agent actions). We then formalize budget-awareness as progressive interval estimation: at each step of a plan, an agent should predict an upper and lower bound on remaining budget, and alert when completion is unlikely. Scoring with a rollout-replay protocol, we find consistent failure patterns on four environments and five frontier agents: (1) strong agents do not necessarily have strong budget-awareness, with correlation r=0.35. (2) frontier models are consistently over-optimistic, continue spending on tasks that are unlikely to succeed, instead of alerting the user early. (3) budget-aware signal is actionable and trainable. Early stop saves 28-64% tokens on failed trajectories, and SFT+RL strengthens early stop and alert behavior. (4) precise interval calibration remains challenging, with interval coverage capping at 47% after SFT+RL. Project page: this https URL 

---
# Learning to Construct Practical Agentic Systems 

**Authors**: Aditya Kumar, Zhihan Lei, Jerry Yan, Joshua W. Momo, Lauhitya Reddy, Rafael Enrique Cabrera Jimenez, Cassandra A. Cohen, Arthur Kajiyama, William W. Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2606.00189)  

**Abstract**: Automated design and optimization of agentic LLM-based systems leads to sophisticated systems that substantially improve result quality over off-the-shelf agentic patterns. However, studies of fielded agentic systems show that production systems focus much more on issues such as simplicity, controllability, and predictability of inference costs. In this paper we propose principled approaches to designing and optimizing practical agentic systems. We describe an agent framework that enables designers to enforce modularity in agentic systems, by defining "pseudo-tools" that call LLMs recursively on a restricted context. Using this framework we hand-engineer agents for a diverse set of tasks, and show that relative to dynamically-planned workflows, hand-constructed fixed workflows are generally cheaper and more accurate. We then propose novel learning methods for the agentic components required by this framework, namely pseudo-tools and fixed workflows. These learning methods generally outperform hand-engineered agents. We also exploit the modularity of the framework to apply multi-objective optimization methods to jointly optimize cost and response quality and blend the results of multiple learning systems. 

---
# Agentic Transformers Provably Learn to Search via Reinforcement Learning 

**Authors**: Tong Yang, Yu Huang, Yingbin Liang, Yuejie Chi  

**Link**: [PDF](https://arxiv.org/pdf/2606.00183)  

**Abstract**: Tree search is a central abstraction behind many language-agent reasoning and decision-making tasks: agents must explore actions, remember failures, and backtrack toward promising alternatives. Yet, we lack a theoretical understanding of how transformer-based policies acquire such search capabilities from the training dynamics of reinforcement learning (RL). We study this question in a stochastic $k$-ary tree environment, where an agentic transformer observes only its trajectory history through interaction and receives a terminal reward for reaching a hidden leaf goal node. We first construct a two-head transformer that implements randomized depth-first search (DFS): one head tracks previous actions, while the other detects failure outcomes and triggers backtracking. We then analyze the training dynamics of policy gradient under a depth-wise curriculum, showing that this same DFS mechanism emerges in stages from sparse reinforcement feedback without expert demonstrations. The resulting policy exhibits depth generalization: after training only on depth-$1$ and depth-$2$ trees, it succeeds on deeper full trees. We further show that, under imbalanced goal distributions, discounting the return leads to a ranked DFS policy that prioritizes higher-probability branches. Overall, our results identify a mechanistic normal form for transformer-based search, in which attention heads specialize and cooperate to extract decision-relevant traces from context and convert them into agentic action selection via RL training. 

---
# The New Social Image: How AI Competency and AI Proactivity Influence Self- and Peer-Perceptions in the Workplace 

**Authors**: Kuntal Ghosh, Marc Hassenzahl, Shadan Sadeghian  

**Link**: [PDF](https://arxiv.org/pdf/2606.00182)  

**Abstract**: Human-AI collaboration is considered the most promising way to incorporate AI in the workplace. What remains unexplored are the experiential consequences of this teaming. More specifically, in a team with AI, how humans perceive themselves (self-perception) and how they are perceived by their coworkers (peer perception) in terms of work ownership and job meaningfulness. In a 2x2x2 vignette study (n=50), participants rated perceptions of ownership, affect, job meaningfulness and satisfaction, and role dynamics across two levels (low/high) of AI proactivity and AI competency as within-subject factors, with point-of-view (self perception/peer perception) as between-subjects. Our results showed that AI with low competency or low proactivity generally improved feelings related to ownership, meaningfulness, satisfaction, and role dynamics, and also increased positive affect while reducing negative affect. However, these effects were often influenced by point-of-view. For instance, low AI proactivity resulted in higher job satisfaction from self-perception rather than peer perception. Based on our findings, we argue that designing AI for the future of work solely around performance metrics may not be adequate. Highly competent and proactive AI-driven systems can have undesirable impacts on perceptions of ownership, job identity, social image and team dynamics, and consequently, job meaningfulness. 

---
# Beyond Augmentation: Score-Guided Pathological Prior for EEG-based Depression Detection 

**Authors**: Xiaojing Chen, Jingqi Cheng, Xu Zhao, Wan Jiang, Jingjing Wu  

**Link**: [PDF](https://arxiv.org/pdf/2606.00180)  

**Abstract**: Deep learning-based Major Depressive Disorder (MDD) detection using Electroencephalography (EEG) is fundamentally constrained by the "small-sample dilemma." Prevailing generative data augmentation methods not only incur heavy computational overhead but also risk introducing synthetic noise, thereby blurring classification boundaries. To challenge the traditional "data quantity first" convention, we propose a novel framework "Beyond Augmentation": Score-Guided Classification (SGC). SGC does not synthesize pseudo-samples; instead, it utilizes an unsupervised generative network architecture to model the structural and statistical anomaly degrees of samples, serving as the core "Pathological Prior". This prior, after robust normalization, is explicitly fused with deep feature representations, thereby precisely guiding the classifier's decision boundary. Furthermore, to dynamically adapt to varying channel configurations, we propose a Cross-Channel Spatial Adaptation module, utilizing a spatial mapping mechanism to effectively resolve the hardware heterogeneity of mismatched channels in multi-center datasets. Extensive experiments on the Mumtaz2016 and high-density MODMA datasets demonstrate the effectiveness and exceptional generalizability of our method under the challenging "zero data augmentation" setting and at "zero sample synthesis cost".
Keywords: Electroencephalography (EEG), Depression Detection, Anomaly Score, Diffusion Models, Few-Shot Learning 

---
# MyoSem: Aligning Electromyography to Natural-Language Action Semantics for Hand Action Understanding 

**Authors**: Chiyue Wang, Dong She, Yang Gao, Zhanpeng Jin  

**Link**: [PDF](https://arxiv.org/pdf/2606.00174)  

**Abstract**: Electromyography (EMG) directly reflects muscle activation and is a key sensing modality for gesture recognition, prosthetic control, and wearable interaction. Existing EMG methods, however, commonly formulate hand action understanding as classification over fixed labels, making it difficult to support querying, retrieval, and generalization based on action descriptions. We present MyoSem, an EMG--action semantic alignment framework that maps low-level EMG signals into a shared semantic space constructed from multi-view action descriptions. MyoSem combines multi-view action-semantic construction, activation-aware EMG encoding, and semantic query alignment, enabling bidirectional retrieval between EMG signals and text descriptions. We systematically evaluate MyoSem on EMG2Pose and NinaPro-series datasets. Results show that MyoSem performs well on EMG--text bidirectional retrieval, generally outperforms most baselines, and shows favorable generalization to unseen users, held-out action classes, and amputee-user transfer scenarios. Ablations and visualizations further validate the effectiveness of each module. Overall, MyoSem advances EMG-based hand action understanding from fixed-label recognition toward queryable bidirectional semantic retrieval, providing a new modeling paradigm for language-mediated EMG action understanding. 

---
# UF-AMA: A unified framework for cross-domain emotion recognition via adaptive multimodal alignment 

**Authors**: Zheng Wang, Shuo Wang, Junhong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.00170)  

**Abstract**: In recent years, emotion recognition based on physiological signals such as electroencephalogram (EEG) has gained considerable attention, as internal physiological data offer greater objectivity and reliability compared to external behavioral data like facial expressions. However, due to distribution shifts caused by individual and contextual differences, along with variations in sample quality across modalities, constructing a cross-domain multimodal emotion recognition model with high generalization and robustness remains a key challenge. In this study, we propose a Unified Framework with Adaptive Multimodal Alignment (UF-AMA) to address cross-subject and cross-session emotion recognition using multimodal physiological signals. First, we construct a cross-modal feature fusion network comprising Transformer encoders and multi-head cross-attention modules, enabling the deep integration of EEG signals and eye-tracking data. Subsequently, we introduce a confidence-aware screening mechanism that dynamically assesses the predictive reliability of each modality branch on target domain samples, partitions samples into different quality subsets, and accordingly applies global consistency alignment and cross-modal distillation. Finally, we propose a multi-level domain adaptation framework that jointly optimizes the marginal and conditional distributions of both local modality-specific and global fusion features, thereby reducing cross-domain distribution shifts at multiple granularities. Extensive experiments on the SEED and SEED-IV datasets demonstrate that UF-AMA achieves state-of-the-art (SOTA) performance in both cross-subject and cross-session tasks. The source code is available at: this https URL. 

---
# ChurnNet: A Optimized Modern AI for Churn Prediction 

**Authors**: Syed Saad Saif, Giulio Maggiore, Paolo Russo, Damiano Distante  

**Link**: [PDF](https://arxiv.org/pdf/2606.00169)  

**Abstract**: Increased competition and the growing similarity of products and services offered by retailers have lowered the barriers for customers to switch to competitors. Accurate churn prediction can be a valuable tool for driving effective personalized marketing campaigns and helping to reduce customer attrition. This study evaluates the performance of traditional machine learning techniques, namely, Random Forests, XGBoost, and Support Vector Machines, and compares them with the Unified Multi-Task Time Series Model for churn prediction, a binary time-series classification task. Despite the strong capacity of the latter to model complex temporal dynamics and inter-variable relationships, our results indicate that for churn prediction, conventional methods can still outperform it in terms of predictive performance, data efficiency, and computational resource requirements for training and deployment. These findings are consistent across multiple datasets and various churn labeling techniques. 

---
# Improving IoT Intrusion Detection Through SMOTE-Based Oversampling and Extended Multi-Model Evaluation on Side-Channel Power Data 

**Authors**: Muhammad Khuram Shahzad, Haseeb Khan, Muhammad Masood Khan, Mubashra Bibi  

**Link**: [PDF](https://arxiv.org/pdf/2606.00161)  

**Abstract**: The detection of intrusions in IoT-based networks poses challenges that cannot be overcome using traditional machine learning methods. Perhaps the biggest of them is related to the presence of a class imbalance in the side-channel dataset, where the number of samples in the normal class compared to the attacks can reach a ratio of 75,964 to 1. Such an aspect is addressed by Dominguez et al. through the proof of concept of power-based intrusion detection. Unfortunately, neither the authors attempt to cope with the problem of imbalance nor do they assess the classifier performance using a balanced training set. In the current paper, both aspects will be handled at once. First, a Synthetic Minority Oversampling Technique (SMOTE) was performed on all nine possible datasets extracted from the initial one, providing an exact imbalance ratio of 1.1 for each. Then, eight algorithms i.e. Random Forest, HistGradientBoosting, LightGBM, Extra Trees, XGBoost, k-Nearest Neighbors, Multi-Layer Perceptron, and Decision Tree were trained under identical conditions for the SMOTE balanced 6-hour dataset. Random Forest reached a micro-averaged F1 score of 0.9989 and macro F1 of 0.9794, thus outperforming the previously best micro-F1 result obtained by Time Series Forest algorithm from the base paper of 0.9983. Extra Trees provided the same performance as well, but at 10 times faster. The introduction of a macro-F1 metric explicitly in contrast to the base paper assessment reveals important class-level information missed with aggregate performance metrics. Recall rates per-class calculated with confusion matrices, F1 heatmaps, and ROC curves show that minority attack classes, especially those with combined M+L infections, are detected reliably only when using SMOTE balance. Feature importance analysis indicates the latest time steps as the most important predictor signals out of 60 steps in a power window. 

---
# DataShield: Safety-degrading Data Filtering for LLM Benign Instruction Fine-Tuning 

**Authors**: Junbo Zhang, Qianli Zhou, Xinyang Deng, Wen Jiang, Jie Pan, Jinbiao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2606.00160)  

**Abstract**: Large language models (LLMs) suffer from degraded safety capabilities even when fine-tuned with benign datasets. However, existing methods for identifying safety-degrading samples in benign datasets suffer from high computational costs and significant noise issues. In this paper, we propose DataShield to efficiently and effectively identify potential safety-degrading samples. Our key intuition is based on the observation that benign fine-tuning increases the overall response compliance of LLMs. DataShield's key technical insight is to quantify each sample's contribution to the model's compliance behavior as its safety degradation score. DataShield consists of three core components: (1) Compliance Vector Extraction, which captures the LLM's compliance behavior tendency; (2) a novel Compliance-Aware Score (CAS), which automatically identifies the optimal safety-critical layer; and (3) Safety-degrading Sample Filtering, which quantifies the projection shift of training data along the compliance direction. Extensive experimental evaluation on Llama3-8B, Llama3.1-8B, and Qwen2.5-7B using the Alpaca and Dolly benign datasets validates our method's effectiveness in identifying high-risk and low-risk data subsets. We also observe that open-ended question answering is more likely to trigger safety degradation, and corresponding responses tend to be longer. We hope this work can provide new insights into data-centric defense methods. The source code is available at: this https URL. 

---
# Digital-to-Physical Transfer of Adversarial Patches for Aerial Vehicle Detection 

**Authors**: Jung Heum Woo, Eun-Kyu Lee  

**Link**: [PDF](https://arxiv.org/pdf/2606.00159)  

**Abstract**: Deep neural network (DNN)-based object detectors are widely used for analyzing aerial and satellite imagery in applications such as environmental monitoring and urban analytics. Despite their strong performance, these models are known to be vulnerable to adversarial examples, and physical adversarial attacks using printable patterns pose realistic security threats. In this paper, we evaluate physical adversarial patch attacks against an aerial vehicle detector by bridging digital optimization and real-world deployment. Adversarial patches are optimized in the digital domain using a loss function that minimizes the maximum objectness score while incorporating non-printability score (NPS) and total variation (TV) constraints to ensure both printability and spatial smoothness. The optimized patches are printed and deployed in three configurations: ON, OFF, and OFF-Side. Experiments using a YOLOv3 detector show that while the OFF patch achieves the highest effectiveness in the digital domain (85.51% Average Objectness Reduction Rate (AORR)), the ON patch demonstrates superior robustness in physical environments (0.197-0.343 Objectness Score Ratio (OSR)) due to its consistent visibility. Furthermore, our results indicate that weather-based augmentation does not necessarily improve patch optimization in this domain. These findings provide critical insights into the practical vulnerabilities of aerial object detection systems. 

---
# Interpreting FCDNNs via RG on Exponential Family 

**Authors**: Fuzhou Gong, Zigeng Xia  

**Link**: [PDF](https://arxiv.org/pdf/2606.00157)  

**Abstract**: We consider establishing the interpretability theory of deep learning through constructing a corresponding relationship between the renormalization group (RG) method in statistical physics and the training process of deep neural networks (DNNs). We have proved the constructed relationship using the one-dimensional Ising model as the input data. In this paper we generalize our results to the case of continuous input data, which is a necessary preparation for applying the corresponding framework to real-world data. To be representative, we consider a class of data distribution in the exponential family. We prove that when the parameters of fully connected (FC) DNNs achieve their optimal value after training, the characteristic parameters of the feature layer output of DNNs are equal to the fixed points of the characteristic parameters of input data under RG method for continuous fields. This conclusion shows that the training process of DNNs is equivalent to RG calculation on this kind of data and therefore the network can extract main features from the input data just like RG. Also, the equivalence further validates the correspondence framework we have established, providing an explanation for the outstanding performance of DNNs on real-world data. 

---
# A physics-informed foundation model for quantitative diffusion MRI 

**Authors**: Zihan Li, Jialan Zheng, Ziyu Li, Xun Yuan, Kasidit Anmahapong, Ziang Wang, Mingxuan Liu, Hongjia Yang, Yifei Chen, Zhuhao Wang, Yuhang He, Fang Chen, Rui Li, Huaiqiang Sun, Yi Liao, Congyu Liao, Yang Yang, Haibo Qu, Xue Zhang, Hongen Liao, Qiyuan Tian  

**Link**: [PDF](https://arxiv.org/pdf/2606.00156)  

**Abstract**: Understanding the human brain requires access to its microscopic tissue architecture. Diffusion magnetic resonance imaging (MRI) provides the only noninvasive window into whole-brain microstructure in vivo, yet reliable quantitative mapping remains confined to specialized research settings requiring dense sampling and optimized acquisition protocols. To address this gap, we present a physics-informed generative microstructure network (PIGMENT) that learns a universal generative prior of human brain microstructure and adapts it zero-shot to each participant's measured data to recover subject-specific maps. Trained on 11375 scans spanning multiple sites, vendors, and field strengths, PIGMENT enabled reliable quantitative mapping for tensor, kurtosis, and NODDI models across external datasets from five independent centers. It remains effective where conventional fitting becomes unreliable, recovering meaningful maps from extremely sparse acquisitions while supporting downstream tractography and structural connectivity mapping. PIGMENT estimates demonstrated strong biological validity, preserving submillimeter cortical microarchitectural patterns and early-childhood white matter developmental trajectories from 10-fold accelerated scans. Furthermore, PIGMENT enables reliable quantitative tensor mapping on cost-efficient low-field systems and the extraction of tumor-related biomarkers using ultra-fast clinical protocols. Together, these results establish PIGMENT as a physics-informed foundation model that extends quantitative diffusion MRI into regimes traditionally too sparse, heterogeneous, or clinically constrained for reliable analysis. 

---
# A Protocol-Language Model for Network Intrusion (Without Deep Packet Inspection) 

**Authors**: Vivek Kumar Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2606.00155)  

**Abstract**: Modern network intrusion detection systems (NIDS) are caught in a structural contradiction: the protocols carrying the highest threat intelligence are precisely those encrypted under TLS 1.3 and QUIC, where payload inspection yields nothing. We ask a simpler question -- what if the attack signature is not in the bytes, but in the rhythm? -- and answer it by treating network flows as a language whose grammar is written entirely in L3/L4 packet metadata: length, inter-arrival time, TTL, TCP flags, and hashed port numbers. We present PLM-NIDS, which proves three claims in sequence. (1) The grammar exists and is learnable: a RWKV-4 state-space model trained on 344,232 unlabelled Monday flows achieves a causal LM validation loss of 0.204, demonstrating that benign traffic has predictable, statistically consistent structure. (2) Attacks violate this grammar: the per-flow perplexity score cleanly separates benign from attack flows with PR-AUC = 0.93 using zero attack labels at training time. (3) This separation is architecturally nontrivial: an LSTM trained on identical token sequences degenerates to a majority-class predictor (ROC-AUC approximately 0.50, F1 = 0.91 by always predicting "attack"), proving that RWKV's causal pre-training provides an inductive bias unavailable to direct classifiers. Supervised fine-tuning further raises PR-AUC to 0.94 and ROC-AUC to 0.75, with a precision of 97.7% at the calibrated operating threshold. The RWKV backbone's O(T) recurrent inference enables per-packet streaming without flow buffering, making PLM-NIDS operationally viable at line rate. Because it reads only IP/TCP/UDP headers, it is inherently encryption-agnostic: TLS 1.3, QUIC, and future encrypted protocols are handled transparently. 

---
# Benchmarking Multimodal LLMs on Code Generation for Complex Interactive Webpages 

**Authors**: Fan Wu, Lishuai Dong, Cuiyun Gao, Yujia Chen, Yiming Huang, Yang Xiao, Qing Liao  

**Link**: [PDF](https://arxiv.org/pdf/2606.00154)  

**Abstract**: Recent advancements in multimodal large language models (MLLMs) have achieved remarkable progress in multimodal reasoning and code generation, catalyzing a new paradigm for front-end development. In particular, these models can directly transform visual designs into executable code, significantly improving the efficiency and adaptability of web development. Modern web applications are dynamic and interactive, featuring frequent user-page interactions. However, existing benchmarks largely evaluate the code generation of static webpages, ignoring the complex interactive behaviors in real-world applications. Besides, their evaluation criteria remain confined to visual fidelity and code structure, overlooking the interaction consistency between the generated and the reference webpages. To address these limitations, we introduce WebIGBench, the first benchmark designed to evaluate code generation for interactive webpages with complex interactions. By combining manually designed interaction paths with UI automation, we collected 103 complex webpages from real-world websites. This benchmark covers 5 popular interactive action types (e.g., click, input) involving 871 distinct interactive actions. Moreover, we propose a novel evaluation pipeline to address the gap in automated assessment of interactive actions. Extensive experiments on several representative MLLMs reveal the performance boundaries of current models in interactive webpage code generation using WebIGBench. The proposed benchmark is available at this https URL. 

---
# DiffCrossGait: Trajectory-Level Alignment for 2D-3D Cross-Modal Gait Recognition via Latent Diffusion 

**Authors**: Zhiyang Lu, Ming Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2606.00153)  

**Abstract**: Cross-modal 2D-3D gait recognition is impeded by inherent domain discrepancies between 2D silhouette and 3D LiDAR range-view representations. While prior methods align only final embeddings, we propose DiffCrossGait, which reformulates cross-modal matching as trajectory-level alignment in an identity-relevant latent diffusion space, rather than assuming full equivalence between 2D and 3D observations. By driving both modalities with shared Gaussian noise within a latent space, we enable continuous alignment throughout the generative evolution. We introduce a Tri-Phase Alignment Strategy that exploits varying noise intensities to enforce identity anchoring, dynamics consistency, and cross-modal structural recoverability, thereby constraining both modalities to share denoising dynamics and bottleneck structure, which promotes modality-invariant gait features. Crucially, our framework decouples generative alignment from the discriminative backbone; the diffusion mechanism serves exclusively as a training objective, ensuring high inference efficiency by eliminating the computational overhead of iterative denoising. Extensive experiments on the SUSTech1K and FreeGait benchmarks demonstrate that DiffCrossGait achieves state-of-the-art performance. 

---
# PrivacyPeek: Auditing What LLM-Based Agents Acquire, Not Just What They Say 

**Authors**: Mingxuan Zhang, Jiahui Han, Dadi Guo, Songze Li, Guanchu Wang, Na Zou, Dongrui Liu, Xia Hu  

**Link**: [PDF](https://arxiv.org/pdf/2606.00152)  

**Abstract**: LLM-based agents are rapidly advancing, autonomously invoking external tools to complete multi-step tasks for users. However, agents often acquire more sensitive information than the task requires. Existing privacy benchmarks audit what the agent's response or outgoing actions disclose, but overlook the acquisition stage where data first enters the agent's context. The over-acquired information is then one careless action or one attack away from an outright leak. To assess its prevalence, we introduce \emph{PrivacyPeek}, a benchmark for evaluating acquisition-stage privacy leakage of LLM-based agents, with $1{,}182$ cases across $7$ acquisition behaviours and $16$ application domains. Specifically, \emph{Acquisition Inspection} examines the agent's tool-call trajectory, both the tools it invokes and the data it receives, to detect when it acquires sensitive information beyond the task scope. \emph{Probe Elicitation} then issues a follow-up probe and measures how readily an attacker could elicit sensitive information the agent acquired but did not disclose. Our experiments on 10 LLM-based agents across 4 model families show that the unnecessary acquisition of sensitive information is widespread. In addition, we observe a correlation between the task-completion capability and acquisition-stage leakage. Prompt-level defences reduce only a small fraction of acquisition-stage leakage, leaving the majority unmitigated. These results make auditing acquisition-stage privacy both urgent and necessary. Our dataset and code are available at this https URL. 

---
# Emergence of Exploration in Policy Gradient Reinforcement Learning via Retrying 

**Authors**: Soichiro Nishimori, Paavo Parmas, Sotetsu Koyamada, Tadashi Kozuno, Toshinori Kitamura, Shin Ishii, Yutaka Matsuo  

**Link**: [PDF](https://arxiv.org/pdf/2606.00151)  

**Abstract**: In reinforcement learning (RL), agents benefit from exploration only because they repeatedly encounter similar states: trying different actions can improve performance or reduce uncertainty; without such retries, a greedy policy is optimal. We formalize this intuition with ReMax, an objective that evaluates a policy by the expected maximum return over $M$ samples, where $M$ is a positive integer, while accounting for return uncertainty. Optimizing this objective induces stochastic exploration as an emergent property, without explicit bonus terms. For efficient policy optimization, we derive a new policy-gradient formulation for ReMax and introduce ReMax PPO (RePPO), a PPO variant that optimizes ReMax while generalizing the discrete retry count $M$ to a continuous parameter $m > 0$, enabling fine-grained control of exploration. Empirically, RePPO promotes exploration, without any explicit exploration bonuses, on the MinAtar and Craftax benchmarks. 

---
# Persona Attack: Incremental Memory Injection Jailbreak Attack against Large Language Models 

**Authors**: Junyoung Park, Seongyong Ju, Sunghwan Park, Jaewoo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2606.00150)  

**Abstract**: As Large Language Models evolve for user convenience, vulnerability to jailbreak attacks continues to be reported despite ongoing efforts in safety training. Traditional jailbreak techniques typically focus on a single prompt injection, neglecting the models' ability to remember the flow of conversation and the user's instructions. In this paper, we propose Persona Attack, a memory injection based jailbreak method that manipulates the model's context window through a step by step approach. Experimental results from applying Persona Attack to several widely used LLMs reveal that, as injections accumulate in memory, models increasingly prioritize these instructions over their internal safety alignment mechanisms. Furthermore, our experiments empirically demonstrate that the attack success rate varies not only according to the memory implementation of the model, but also combinations of instructions and can reach 95% under specific instruction configurations. 

---
# StemBind: When MLLMs Get Lost Between Rules and Instances in Abstract Visual Reasoning 

**Authors**: Xixiang He, Baiqi Wu, Xingming Li, Ao Cheng, Qiyao Sun, Xuanyu Ji, Qingyong Hu  

**Link**: [PDF](https://arxiv.org/pdf/2606.00148)  

**Abstract**: Multimodal large language models (MLLMs) often know the rule but pick the wrong answer: on abstract visual reasoning (AVR) tasks, a model can describe what it sees and name the underlying pattern, yet still fail to choose the matching candidate. Existing AVR benchmarks cannot detect this because they collapse perception, rule induction, and answer selection into a single right-or-wrong signal. We introduce StemBind, a shared-stem diagnostic benchmark that probes the same visual stem with three aligned questions: Perception (what is in the image), Rule (what pattern governs it), and Full (which option completes it), so a final-answer error can be attributed to a specific sub-step on the same evidence. StemBind contains 2,298 curated knowledge-light stems across nine auditable visual operations, totaling 19,533 P/R/F tasks, with each full item annotated by Sternberg's four reasoning stages (S1 Encode, S2 Infer, S3 Map, S4 Apply). Evaluating 24 frontier MLLM configurations yields four findings. (i) The R-F chasm: rule accuracy exceeds full-item accuracy on 22 of 24 models, so most failures happen after the rule is identified. (ii) A persistent binding gap: even when P and R are both correct on the same stem, models still answer F incorrectly 51.2% of the time. (iii) The bottleneck is S3: process diagnostics and Stage-wise Stimulus Augmentation localize the dominant failure to rule-to-instance mapping. (iv) Scaling and thinking do not help: neither larger models nor explicit thinking mode reliably closes the gap, and thinking even lowers rule and full-item accuracy. StemBind reframes AVR evaluation from final-answer ranking to locating where abstract visual reasoning breaks down, identifying rule-to-instance binding as a concrete next target for vision-grounded reasoning. 

---
# RAFT: Data Refinement and Adaptive Distillation for Domain Fine-Tuning with Alleviated Forgetting 

**Authors**: Yuduo Li, Xiaofeng Shi, Qian Kou, Longbin Yu, Hua Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.00147)  

**Abstract**: Domain-specific supervised fine-tuning (SFT) often improves in-domain performance at the cost of degrading a model's general capabilities. We view this degradation through two practical gaps in domain SFT: a supervision-compatibility gap, where domain targets differ in style and reasoning format from the original model's natural responses, and a trajectory-preservation gap, where teacher-forced SFT optimizes fixed target tokens without constraining the model's behavior on its own generated prefixes. This process fails to preserve the model's original behavior. We propose RAFT (Data Refinement and Adaptive Distillation for Domain Fine-Tuning with Alleviated Forgetting), a two-stage framework that addresses both factors. First, RAFT constructs model-compatible supervision through self-conditioned rewriting, semantic filtering, and answer fusion. Second, RAFT performs Answer-Conditioned On-Policy Distillation, where the original instruction-tuned model provides soft targets on student-generated trajectories while being conditioned on the fused answer as helpful context. We further introduce top-K temperature distillation and EMA-based adaptive loss balancing to stabilize the domain-general trade-off. Across three instruction-tuned backbones and five domains, RAFT improves average domain accuracy by 23.2% over standard SFT, while recovering part of the SFT-induced degradation on MS-Bench and IFEval, with relative improvements of 18.2% and 10.2%, respectively. These results show that coupling data refinement with trajectory-level preservation provides an effective recipe for domain fine-tuning with alleviated forgetting. 

---
# Multi-Contrast MRI Motion Correction via Parameter-Informed Disentanglement and Adaptive Experts 

**Authors**: Honglin Xiong, Yuxian Tang, Feng Li, Yulin Wang, Lei Xiang, Dinggang Shen, Qian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.00146)  

**Abstract**: Motion artifacts in magnetic resonance imaging (MRI) degrade diagnostic reliability. Existing deep learning methods are typically contrast-specific and fail to generalize across diverse modalities and artifact severities. We propose a unified framework combining parameter-informed contrast disentanglement with severity-aware adaptive correction. ScanCLIP, pretrained on over 30,000 MRI text-image pairs, derives contrast embeddings from acquisition parameters to disentangle contrast style from anatomical content, yielding contrast-free features. A Vision Transformer then estimates motion severity and routes features through a Mixture-of-Experts network, enabling targeted artifact correction. A dual-pathway decoder reconstructs both the clean image and residual artifact map, enforcing image-space consistency. On IXI and HCP benchmarks, our method improves PSNR by 0.75 dB and SSIM by up to 0.0279 over state-of-the-art approaches, with larger gains at higher artifact severities. It further demonstrates robust zero-shot generalization on real-world clinical data acquired with unseen scanning parameters, where existing methods either fail to remove artifacts or introduce additional distortions. 

---
# Completion at the Boundary (CaB): Deployable Switching with Completion-Aware Control under Limited Calibration 

**Authors**: Yusuke Sano, Takeshi Itoga  

**Link**: [PDF](https://arxiv.org/pdf/2606.00145)  

**Abstract**: Vision-language-action (VLA) agents can execute natural-language instructions, yet deployed systems still lack an operational interface: deciding when the instruction is complete. This gap is acute in short composites ("do A, then B"), where mistimed handoffs cascade into downstream failures. Completion is inherently closed-loop because switching is an intervention that changes the instruction context and thus future actions and observations. We study completion under a deployable low-calibration regime motivated by open-ended instruction spaces, enforcing no test-time relearning and a single globally calibrated switching rule selected once on development set and reused unchanged on test set. Under this constraint, collapsing asymmetric boundary evidence into a single scalar can be brittle under polarity shifts across tasks. We propose Completion at the Boundary (CaB), which predicts an event-local completion object in the form of Boundary-Phase Tokens (Before/Hit/After), retaining two-sided boundary evidence under this discipline. CaB-When converts this completion object into a minimal, auditable switching decision (when), while CaB-How reuses the same completion object to condition action generation for boundary-stable control through handoffs (how). Using an intervention-aware E1/E2 protocol, we show that CaB improves composite execution and handoff quality on a first-person Minecraft VLA benchmark under matched capacity and deployability constraints. 

---
# BudgetDraft: Acceptance-Aware Multi-View Training for Sparse-KV Speculative Decoding 

**Authors**: Liang He, Jingbo Wen, Qishi Zhan, Yixiong Chen, Kangning Cui, Qizhen Lan, Xilu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.00144)  

**Abstract**: Speculative decoding speeds up autoregressive decoding by using a drafter to propose multiple tokens that a verifier validates in parallel. In resource-constrained deployments, the drafter uses a sparse KV cache to limit peak GPU memory and end-to-end latency under a fixed KV budget, while the verifier keeps a full KV cache. Mid-to-long context inference (4K--16K context length) is common in real applications. However, naive sparse/full speculative decoding suffers from the sparse/full mismatch as context length grows, causing the acceptance rate to drop quickly. We propose BudgetDraft, a multi-view sparse training method for sparse drafting in mid-to-long inference. The drafter is exposed to multiple sampled KV budgets during training and learns to align each sparse view with one shared full-cache teacher target. BudgetDraft combines an acceptance-aware loss on a full-cache branch with a multi-view loss on a sparse-cache branch, producing a single budget-robust drafter that recovers acceptance across sparsity levels without extra inference-time components. Experimental results on PG-19, LongBench, and LWM show that BudgetDraft achieves up to 6.55x, 4.46x, and 2.10x end-to-end speedup vs AR at 4K, 8K, and 16K context lengths, while keeping the inference pipeline memory-friendly. 

---
# Regime-Adaptive Continual Learning for Portfolio Management 

**Authors**: Chaofan Pan, Lingfei Ren, Linbo Xiong, Yonghao Li, Wei Wei, Xin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.00143)  

**Abstract**: Financial markets are inherently non-stationary, exhibiting frequent regime shifts and structural changes that render traditional Portfolio Management (PM) approaches ineffective. Existing remedies, such as rolling-window retraining and naive online fine-tuning, are hindered by high computational costs and insufficient knowledge utilization, respectively, resulting in low returns and limited adaptability. Continual learning (CL) offers a promising paradigm by enabling trading agents to accumulate and transfer knowledge across sequential tasks. In this paper, we propose \textbf{Re}gime-aware \textbf{C}ontinual \textbf{A}daptive \textbf{P}ortfolio management (\textbf{ReCAP}), a novel framework that integrates CL into PM to address the challenges of dynamic financial environments. ReCAP employs an adaptive regime detection module to segment historical market data into variable-length regimes, enabling regime-specific learning of policy vectors and the construction of a policy library. During continual trading, a regime-gate module adaptively combines policy vectors from the library based on the current market state, facilitating rapid adaptation to newly detected regimes. Only the regime-gate and the current regime's policy vector are continually updated to preserve useful knowledge effectively. Extensive experiments on five real-world datasets demonstrate that ReCAP consistently outperforms popular baselines, achieving superior returns in long-term investment horizons and rapid adaptation to regime shifts. 

---
# Adaptive data selection improves wearable prediction under low baseline performance 

**Authors**: Ali Kargarandehkordi  

**Link**: [PDF](https://arxiv.org/pdf/2606.00141)  

**Abstract**: Adaptive sensing strategies that selectively sample data are increasingly used in wearable health systems to improve prediction performance under limited data budgets, yet their benefits across individuals remain poorly understood. Here, we evaluate adaptive selection of time windows for model training under fixed measurement budgets across multiple sensing modalities, including heart rate, activity, and ecological momentary assessment (EMA), in a longitudinal wearable dataset. We quantify performance gains relative to random sampling using both area under the receiver operating characteristic curve (AUROC) and F1 score. Adaptive strategies yield substantial improvements in AUROC for participants with low baseline performance (with gains up to 0.7), while offering limited or negative gains for participants with strong baselines. Across modalities, adaptive gain is strongly inversely correlated with baseline performance (Pearson r = -0.67; Spearman p = -0.62). At the participant level, most individuals benefit in AUROC (60-80% across modalities), although improvements in F1 are smaller and less consistent. These findings show that adaptive sensing is not uniformly beneficial, but instead provides the greatest value in underperforming settings. Our results support selective deployment strategies that tailor adaptive sensing based on baseline performance to improve efficiency in wearable health monitoring. 

---
# Geometric Erasure by Contrastive Velocity Matching in Rectified Flows 

**Authors**: Jonas Henry Grebe, Tobias Braun, Anna Rohrbach, Marcus Rohrbach  

**Link**: [PDF](https://arxiv.org/pdf/2606.00140)  

**Abstract**: While the rapid adoption of multimodal generative models offers immense potential, it has also increased the risks of harmful content synthesis, deepfakes, and copyright infringements. To address these challenges, concept erasure has emerged as a prospective safeguard. However, as the field gradually transitions from U-Net-based diffusion models to Rectified Flow Transformers, erasure research has struggled to keep pace. In this work, we introduce GEM, a simple but highly effective erasure framework for Rectified Flow models. As part of our contribution, we establish a principled bridge between trajectory-based unlearning grounded in Generative Flow Networks and classic teacher-guided erasure: we translate trajectory-based signals into a teacher-guided flow-matching setup that unifies the strengths of both paradigms. Concretely, a teacher provides complementary attraction and repulsion signals that we combine into a single geometric guidance objective, yielding targeted suppression of unwanted concepts while preserving benign generation. 

---
# Geodesics with Unified Tangent-constrained Priors and Curvature Regularization 

**Authors**: Chong Di, Li Liu, Jinglin Zhang, Zhenjiang Li, Da Chen, Laurent D. Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2606.00139)  

**Abstract**: Curvature-penalized geodesic models have proven their effectiveness in image segmentation by computing globally optimal curves. Unfortunately, these models remain susceptible to shortcuts when delineating objects with complex shapes and image intensity distributions, as they lack mechanisms to enforce shape-aware tangent constraints. To address this limitation, we propose a unified geodesic framework that integrates tangent-constrained priors with curvature penalization. The key idea is to formulate tangent admissibility directly within the orientation-lifted space, where path tangents are restricted to spatially varying angular sectors derived from intrinsic shape representatives (ISR) such as skeletons or interior landmarks. This formulation gives rise to a family of tangent-constrained Finslerian metrics, extending the classical curvature-penalized geodesic models while enforcing mandatory tangent constraints. The resulting Hamilton-Jacobi-Bellman (HJB) partial differential equations (PDEs) admit efficient numerical solutions via variants of the fast marching method, preserving the single-pass computational complexity. Experiments on synthetic, natural, and medical images demonstrate that the proposed geodesic framework indeed improves robustness against weak boundaries and topological shortcuts, yielding segmentation results with enhanced shape fidelity compared to existing geodesic models. 

---
# Generative AI and Digital Ecosystem Resilience: A Proactive Lifecycle-Based Survey 

**Authors**: Jonghyun Chung, Rishabh Chaddha, Sanket Badhe, Debanshu Das, Nathan Huang, Amanpreet Kaur  

**Link**: [PDF](https://arxiv.org/pdf/2606.00136)  

**Abstract**: The proliferation of adversarial synthetic content, accelerated by Generative AI (GenAI) is rendering traditional reactive detection methods ineffective. This survey synthesizes emerging research to demonstrate a paradigm shift toward the proactive detection of emerging inauthentic narratives. In this survey, we adopt a unified, lifecycle-based taxonomy to combine socio-technical lifecycle models of adversarial campaigns with advanced computational methodologies for emerging inauthentic narrative detection. By structuring the analysis around the C5 Interaction Model (Context, Causes, Content, Cycle of Amplification, Consequences), we integrate different research streams from machine learning and social science. To differentiate spread patterns of synthetic amplification from authentic baseline traffic, this paper surveys state-of-the-art techniques for modeling the creation, seeding, and propagation of fresh narratives, including the analysis of Coordinated Inauthentic Behavior (CIB), epidemiological modeling, and Hawkes process. This survey also provides a systematic review of proactive detection methods for adversarial threats at different stages in the C5 interaction model, specifically, anomaly detection in high-dimensional embedding spaces, unsupervised coordination detection on multi-layer graphs, and agentic AI systems. Finally, this survey addresses challenges posed by GenAI, including the difficulty of tracking rapidly changing threats and multi-level distributional drift, and it outlines a future research agenda focused on detecting anomalous clusters and building anticipatory and resilient systems. This survey provides a comprehensive, lifecycle-based review of methods for the proactive detection of emerging synthetic threats for more resilient information ecosystems. 

---
# On Effectiveness and Efficiency of Agentic Tool-calling and RL Training 

**Authors**: Tong Liu, Cheng Qian, Matej Cief, Yuan He, Daniele Dan, Nikolaos Aletras, Gabriella Kazai  

**Link**: [PDF](https://arxiv.org/pdf/2606.00135)  

**Abstract**: Tool-calling is a central component of modern large language model (LLM) agents, equipping them with skills beyond their parametric knowledge. This paper studies tool-calling along two complementary axes: effectiveness, i.e., how this capability is measured, and efficiency, i.e., how it is learned. On effectiveness, we systematically analyze tool-calling evaluation pipelines and show that results can be highly sensitive to seemingly minor, often undocumented implementation choices including the random seed, system prompt, multi-turn template construction, and how prior interaction/reasoning history is carried forward. These choices can lead to substantial differences in reported performance, especially in multi-turn settings where without rigorous standardization, leaderboard rankings are unreliable. On efficiency, we examine standard reinforcement learning (RL) for tool-calling and identify two sources of computational waste: (i) during rollouts, many prompts produce no learning signal, and (ii) during policy updates, optimization incurs high computational cost. Guided by these findings, we introduce two techniques that accelerate RL-based tool-calling training, achieving substantial wall-clock speedup without degrading performance. 

---
# XAI-SOH-FL: Enhancing SOH-FL with Adaptive Aggregation and Explainable AI for Intrusion Detection in Heterogeneous IoT 

**Authors**: Ambreen Aslam, Maaz Hassan, Bibi Zahra, Muhammad Khuram Shahzad  

**Link**: [PDF](https://arxiv.org/pdf/2606.00134)  

**Abstract**: Intrusion Detection Systems (IDS) in Internet of Things (IoT) environments face significant challenges due to data heterogeneity, lack of labeled data, and limited model interpretability. Federated Learning (FL) offers a privacy-preserving solution; however, existing approaches such as SOH-FL suffer from two key limitations: reliance on a manually tuned aggregation parameter {\gamma} and lack of explainability in model predictions. In this paper, we propose XAI-SOH-FL, an enhanced framework that integrates adaptive aggregation and explainable artificial intelligence into the SOH-FL paradigm. First, we introduce a dynamic {\gamma} selection mechanism based on similarity thresholding, enabling the aggregation process to adapt to evolving data distributions. Second, Bayesian Optimization is employed to automatically determine optimal {\gamma} values, eliminating the need for manual tuning. Third, SHAP (SHapley Additive exPlanations) is incorporated to provide feature-level interpretability for intrusion detection decisions. Experimental evaluation on the CICIDS2017 dataset demonstrates that the proposed approach achieves an accuracy of 94.12% and an F1-score of 0.92, outperforming the baseline SOH-FL model while converging in fewer communication rounds. Furthermore, SHAP-based analysis reveals that flow-level features such as Flow Duration and Packet Length significantly influence model predictions. These results indicate that XAI-SOH-FL provides an effective balance between accuracy, adaptability, and interpretability in heterogeneous IoT environments. 

---
# Foundation-Preserving Adaptation via Generalized Rayleigh-Quotient Optimization 

**Authors**: Dongjun Kim, Adrian de Wynter, Huancheng Chen, Heasung Kim, Haris Vikalo  

**Link**: [PDF](https://arxiv.org/pdf/2606.00132)  

**Abstract**: While finetuning effectively adapts foundation models to specialized downstream tasks, it can degrade nontarget capabilities acquired during pretraining. Existing forgetting aware methods typically seek safer updates through specialized initialization or fixed constraints, but do not regulate the adaptation preservation trade-off during training. We propose Foundation Preserving LoRA (FoLoRA), a forgetting aware optimization framework. Guided by a first order preservation condition, FoLoRA defines a forgetting penalty over pretraining-proxy activations and a task utility over downstream task activations. It then scores update directions by task utility per unit forgetting penalty via a generalized Rayleigh quotient. The resulting spectral coordinate system enables direction wise gated Adam updates, attenuating low utility to penalty directions during training. To estimate the forgetting penalty, FoLoRA constructs pretraining proxy calibration data by sampling from the pretrained model rather than relying on a single proxy dataset. Experiments on math, code, and instruction following adaptation show that FoLoRA achieves the strongest preservation adaptation balance over baselines, improving target task performance with best aggregate preservation of non target capabilities. 

---
# AI-PROPELLER: Warehouse-Scale Interprocedural Code Layout Optimization with AlphaEvolve 

**Authors**: Chaitanya Mamatha Ananda, Rajiv Gupta, Mircea Trofin, Aiden Grossman, Sriraman Tallam, Xinliang David Li, Amir Yazdanbakhsh  

**Link**: [PDF](https://arxiv.org/pdf/2606.00131)  

**Abstract**: Post-link optimizers (PLOs) such as Propeller and BOLT have demonstrated that precise, profile-guided code layout can extract significant performance gains from heavily optimized binaries. However, these systems are currently restricted to intraprocedural techniques, leaving the global potential of interprocedural layout largely untapped. Interprocedural code layout is historically difficult due to a combinatorially intractable search space and complex call-return semantics that are challenging to model. Consequently, the performance potential of fine-grained interprocedural layout remains unproven in practice. AI-PROPELLER uses Magellan, an agentic workflow that evolves the compiler heuristic in Propeller into a fine-grained interprocedural optimizer and fine-tunes the resulting policy hyperparameters. To ensure high-fidelity, we move away from approximate static cost models and the agentic workflow generates multiple layout variants that are executed on actual hardware to measure real performance counters, providing a precise reward signal for the evolutionary loop. AI-PROPELLER has been evaluated on several benchmarks including large warehouse-scale applications and experiments show performance improvements of 0.23% to 1.6% optimized with state-of-the-art FDO and PLO which is significant for real-world binaries. This is the first time ever that large warehouse-scale applications in industrial settings have been optimized with fine-grained interprocedural code layout. 

---
# Automatically Differentiable Nonlinear Tensor Networks (ADNTNs) for Exponential Compression of Deep Neural Networks 

**Authors**: Andrzej Cichocki, Michal Wietczak  

**Link**: [PDF](https://arxiv.org/pdf/2606.00130)  

**Abstract**: We study Automatically Differentiable Nonlinear Tensor Networks (ADNTNs), a family of structured weight generators whose compact core tensors are trained end-to-end by reverse-mode automatic differentiation (AD). The approach can be viewed as a natural extension of low-rank adaptation and tensor factorisation: instead of using one low-rank matrix update, an ADNTN builds a large weight tensor through a hierarchy of small cores, nonlinear activations, and optional lateral mixing tensors. The paper focuses on three architectures: Tree Tensor Networks (TTNs), augmented TTNs (aTTNs) with boundary disentanglers, and Multi-scale Entanglement Renormalisation Ansatze (MERA).
The formulation supports nonlinear activations, task-aware objectives, batching, and hardware-aware execution schedules. At the same time, the paper keeps a clear distinction between \emph{differentiating} a contraction program and making contraction free: AD does not remove the cost of large intermediates, poor contraction orders, or exact contraction of general loopy tensor networks.
Extensive simulations on AlexNet and VGG-16 layers show per-layer compression ratios from roughly $2000\times$ to $77000\times$ in the studied settings, with accuracy often matching the dense baseline and, in several VGG-16 cases, improving it. These results are encouraging rather than final: they suggest that ADNTNs are a promising, mathematically structured, and hardware-aware route toward much smaller neural networks, provided that optimisation, contraction schedules, and deployment kernels are designed together. 

---
# A Shared Valence Axis Across Modern LLMs and Human EEG: The Saturation Regularity 

**Authors**: Yousef A. Radwan, Xuhui Liu, Kilichbek Haydarov, Yuqian Fu, Mohamed Elhoseiny  

**Link**: [PDF](https://arxiv.org/pdf/2606.00129)  

**Abstract**: Large language models (LLMs) have emerged as powerful representation learners whose internal features increasingly align with human cognition. We study whether modern LLMs can serve as a lens for understanding neural representations in the human brain, focusing on emotional valence in EEG.
We first build a one-dimensional valence direction, the V-axis, from modern LLMs using only nine emotion-evocative sentences. We validate it through zero-shot transfer to sentiment benchmarks and cross-model consistency across fourteen LLMs. We then show that this LLM-derived direction maps onto human neural activity. On a public EEG cohort of 123 subjects watching affective videos, a single linear projection on EEG features tracks the V-axis position of each stimulus. Moreover, 36 EEG emotion classifiers trained without exposure to the V-axis spontaneously rediscover the same direction in their internal representations, suggesting that the same valence structure emerges in both language models and human electrophysiology.
Yet this convergence does not provide an effective training signal. We test twenty-five alignment strategies, including knowledge distillation, representational similarity, contrastive, and topographic losses; none improve decoding, and sixteen significantly reduce accuracy. We formalize this result as the saturation regularity: once task labels alone drive a brain-decoding network onto the target direction, additional supervision mainly distorts an already-saturated basin, while the load-bearing within-class residual receives little useful gradient.
This regularity also indicates where improvement should come from: the residual subspace unreachable by supervision. Motivated by this insight, we ensemble across residual diversity rather than supervising the basin, improving balanced accuracy by 10.5% over the prior best on FACED, with the same effect replicated on SEED-V. 

---
# Multimodal Music Recommendation System using LLMs 

**Authors**: Srikar Prabhas Kandagatla, Sreehitha R. Narayana, Chandana Magapu, Swetha Mohan, Shamanth Kuthpadi, Hongjie Chen, Ryan A. Rossi, Franck Dernoncourt, Nesreen Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2606.00125)  

**Abstract**: Music recommendation systems typically treat songs as opaque tokens, relying on collaborative interaction histories which overlooks semantic or acoustic content. Prior work has explored LLM-augmented, multimodal, and text-enhanced approaches to sequential recommendation, and while some methods partially combine semantic, acoustic, or engagement signals, none jointly model all three within a unified LLM-based sequential reasoning framework that grounds recommendations in actual song content. In this work, we propose a multimodal framework for session-based music recommendation that enriches the LastFM-1K dataset with three complementary signals: (1) audio and lyric embeddings extracted using pretrained music and text representation models, (2) LLM-generated semantic metadata using the MGPHot annotation schema, and (3) listening completion ratios. We adopt the E4SRec framework by extending it with multimodal features and different item ID encoder backbones, including SASRec, BERT4Rec, and GRU4Rec. We further extend the LLM backbone option with LLaMa-2-13B, Qwen2.5-7B-Instruct, and LLaMa-3-70B in both zero-shot and fine-tuned settings. Our experiments show that integrating content-based features improves over ID-only baselines up to 95% in terms of Recall and 79% in terms of NDCG. Moreover, our experiments show that naive multimodal fusion does not always yield additive improvements, highlighting challenges in cross-modal integration. We release a large-scale multimodal benchmark for music recommendation. 

---
# CardioLens: Revealing the Clinical Reality Gap of MLLMs via Multi-Sequence Cardiac MRI Evaluations 

**Authors**: Zixian Su, Hongkai Zhang, Fan Gao, Encheng Su, Taiping Qu, Jingwei Guo, Nan Zhang, Hui Wang, Zhen Zhou, Kairui Bo, Yan Chen, Yue Ren, Shuai Li, Lei Xu, Henggui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.00123)  

**Abstract**: Multimodal Large Language Models (MLLMs) have shown strong performance on public medical benchmarks, yet existing evaluations often remain weak proxies for clinical use, relying on isolated inputs and simplified recognition-style tasks. We introduce CardioLens, a leakage-resistant evaluation testbed for multi-sequence Cardiovascular Magnetic Resonance (CMR), constructed from private hospital archives through a rigorous report-to-QA construction and verification pipeline. CardioLens contains 473,896 slices and 13,494 verified QA pairs across 4D Cine, LGE, perfusion, and T2-weighted imaging, and evaluates three stages of CMR interpretation: image understanding, report generation, and disease diagnosis. Across 24 state-of-the-art MLLMs, CardioLens reveals a substantial clinical reality gap: models perform poorly overall, with performance degrading along the real CMR workflow. Confusion analysis further shows a category-collapse failure mode, where models default to frequent abnormal categories rather than distinguishing clinically distinct findings. To rule out MLLM-compatible input construction as the primary cause, we compare random, clinically motivated, and data-driven slice selection protocols under different slice budgets; performance changes only marginally, typically by about 1%. Explicit reasoning prompts also fail to rescue performance, often making models more conservative rather than improving visual evidence use. These results show that current MLLMs remain far from reliable CMR interpretation, where clinical decisions require integrating distributed evidence across sequences, views, and temporal phases. CardioLens provides a clinically grounded testbed for developing next-generation MLLMs toward real-world clinical deployment. 

---
# Versatile Framework with Semantic and Structural guidance for Image Reconstruction from Brain Activity 

**Authors**: Yizhuo Lu, Changde Du, Qiongyi Zhou, Liuyun Jiang, Huiguang He  

**Link**: [PDF](https://arxiv.org/pdf/2606.00121)  

**Abstract**: Reconstructing visual stimuli from brain recordings has been a meaningful and challenging task in brain decoding. Especially, the achievement of precise and controllable image reconstruction bears great significance in propelling the progress and utilization of brain-computer interfaces. Recent methods, leveraging advances in the power of text-to-image generation models, have reconstructed images that closely approximate complex natural stimuli in terms of semantics (e.g., concepts and objects). However, they struggle to maintain consistency with the original stimuli in fine-grained structural information (e.g., position, orientation and size), which undermines both the controllability and interpretability of the models. To address the aforementioned issues, we propose a two-stage image reconstruction framework, termed MindDiffuser. In Stage 1, Contrastive Language-Image Pretraining (CLIP) text embeddings decoded from brain responses are input into Stable Diffusion, generating a preliminary image containing semantic information. In Stage 2, we use decoded shallow CLIP visual features as supervisory signals, iteratively refining the feature vectors from Stage 1 via backpropagation to align structural information. We conducted extensive experiments on brain response datasets across three modalities (fMRI, EEG, MEG) elicited by visual stimuli, demonstrating that our framework significantly enhances the performance of previous state-of-the-art models, highlighting the effectiveness and versatility of our approach. Spatial and temporal visualization results further support the neurobiological plausibility of our framework, providing guidance for future neural decoding efforts across different brain signal modalities. 

---
# SpikeWFM: Spiking-Aided Wireless Foundation Model for Robust Channel Prediction 

**Authors**: Liwen Jing, Yisha Lu, Tingting Yang, Li Sun, Yuxuan Shi, Yuwei Wang, Mengfan Zheng, Leiyang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.00120)  

**Abstract**: This paper proposes SpikeWFM, a novel hybrid architecture that integrates spiking neural networks (SNNs) with conventional artificial neural network (ANN)-based transformers for wireless foundation models (WFMs). Inspired by the noise-robust and energy-efficient information processing in the human brain, SpikeWFM aims to enhance the resilience of WFMs against noise and interference while maintaining strong generalization capabilities across diverse wireless scenarios. Drawing from the success of large language models, WFMs leverage self-supervised pre-training on large-scale datasets spanning various wireless environments to learn a unified embedding that supports a wide range of downstream tasks, including channel prediction, channel estimation, beam predition, positioning and etc. Such models typically outperform task-specific designs and exhibit superior adaptability to unseen conditions. However, existing WFMs remain vulnerable to realistic noise and interference in practical wireless systems. To address this limitation, we incorporate spiking neurons into the transformer-based WFM architecture. We provide a brief theoretical analysis demonstrating how the SNN-ANN hybrid effectively mitigates noise and interference through temporal sparsity and event-driven processing. Experimental results show that SpikeWFM consistently outperforms conventional ANN-based WFMs in both pre-training convergence and channel prediction accuracy. Additional results on communication and sensing tasks will be presented in the full journal version of this work. 

---
# V2I Work Zone Geometry Reconstruction with Pose-Conditioned UWB Range Denoising 

**Authors**: Jiaxi Liu, Hangyu Li, Yang Cheng, Rui Gana, Junwei You, Weizhe Tang, Peng Zhang, Steven T. Parker, Xiaopeng Li, Bin Ran  

**Link**: [PDF](https://arxiv.org/pdf/2606.00119)  

**Abstract**: Reliable work zone mapping is important for connected and autonomous vehicles (CAVs) to navigate safely and smoothly through work zone areas. Cone-mounted ultra-wideband (UWB) roadside units (RSU) offer a cost-effective way for work zone layout inference, as roadside anchors and vehicle tags provide direct vehicle-to-infrastructure (V2I) range constraints for work zone geometry reconstruction. However, UWB range estimation is degraded by bursty outliers, non-line-of-sight (NLOS) errors, arbitrary anchor-ordering issues, and vehicle pose uncertainties in practical field deployments. To address these challenges, this study proposes a pose-conditioned, permutation-equivariant predictive denoiser for multi-anchor UWB ranging. The model employs shared anchor-wise temporal prediction to capture range dynamics, symmetric set aggregation to handle unordered and missing anchors, and pose-conditioned residual decoding to incorporate vehicle motion as a geometric prior. A two-stage training strategy first learns prediction from observed ranges, and then fine-tunes the denoiser with NLOS-weighted supervision. The method is evaluated on rare real-world V2I UWB field data collected with a CAV, as well as on controlled large-scale simulation benchmarks for ablative insights. Results show that the proposed method substantially improves range accuracy, cone localization, and work zone geometry reconstruction in challenging NLOS-dominated regimes, remains robust to anchor re-indexing and moderate anchor dropout, and reduces measurement-weighted field MSE by 66.9% relative to the raw input. 

---
# Enhancing BiGRU with a KAN Block for Legal Document Classification and Summarization 

**Authors**: Ahmed Faizul Haque Dhrubo, Souvik Pramanik, Most. Aysha Siddika Sumona, Shahnewaz Siddique, Mohammad Ashrafuzzaman Khan, Mohammad Abdul Qayum, Mohsin Sajjad  

**Link**: [PDF](https://arxiv.org/pdf/2606.00116)  

**Abstract**: This study introduces a novel architecture of KAN-based BiGRU model for the task of classification and summarization of legal documents in a low-resource multilingual setup. In order to tackle problems associated with domain language, the usage of different languages, long dependencies within context, and class imbalance, we employ the dataset composed of legal documents from Bangladesh and taken from Manupatra, which include Bengali, English, and transliterated Bengali languages. Our classification task involves BiGRU model, along with Kolmogorov-Arnold Network (KAN) module, while the summarization part utilizes attention-based GRU, combined with a KAN model head. Classification model yields 67.96% of accuracy and 0.65 F1 score; while ROUGE-1, ROUGE-2, and ROUGE-L measures for summarization yield 0.38, 0.23, and 0.31 F1 scores, correspondingly. Ablation study shows that the use of KAN increases classification accuracy from 57.34% to 67.96%. Moreover, our proposed technique is compared to several baselines, including classical ML algorithms and pretrained language models. 

---
# VDSB-GWSyn: Diffusion Schrödinger Bridge for Controllable and Anatomically Feasible Guidewire Synthesis in Coronary Angiography 

**Authors**: Haoyuan Tang, Zhuo Zhang, Jialin Li, Shuai Xiao, Jiachen Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.00109)  

**Abstract**: Coronary guidewire endpoint localization is a fundamental capability for computer-assisted PCI, and its importance increases as robot-assisted PCI is progressively adopted to reduce operator radiation exposure. However, the scarcity of annotated CAG images with guidewires and the limited adaptability of existing guidewire synthesis models remain key bottlenecks for guidewire endpoint localization. To address this issue, we propose VDSB-GWSyn, a Diffusion Schrödinger Bridge (DSB) model-based framework, enabling synthesis of controllable, high-fidelity guidewire samples under complex anatomical backgrounds. VDSB-GWSyn first uses our shape prior algorithm to learn the basic guidewire geometry. It then generates guidewire masks under constraints imposed by the vessel segmentation masks and outputs the corresponding endpoint coordinates. Finally, it synthesizes realistic guidewire samples on real CAG images using DSB conditioned with SPADE. Experimental results show that the guidewire samples synthesized by VDSB-GWSyn achieve favorable ROI-FID and ROI-KID, as well as high IPR scores. In addition, incorporating our synthesized data for synthetic pre-training followed by real fine-tuning substantially improves downstream guidewire endpoint localization, reducing MPE from 16.01~px to 7.71~px and increasing PCK at 3~px from 52.63\% to 86.27\%, leading to more clinically reliable deployment of robot-assisted guidewire delivery systems. Moreover, the core design philosophy of controllable device synthesis with strict background preservation and anatomical feasibility constraints has the potential to transfer to other interventional device perception tasks where annotated data are scarce. 

---
# Project SPARROW and the Future of Conservation Technology 

**Authors**: Juan M. Lavista Ferres, Carl Chalmers, Bruno Demuro Segundo, Zhongqi Miao, Andres Hernandez Celis, Federico Alves Torres, Isai Daniel Chacon Silva, Anthony Cintron Roman, Allen Kim, Meygha Machado, Luana Marotti, Amy Michaels, Daniela Ruiz Lopez, Catherine Romero, Rahul Dodhia, Inbal Becker-Reshef, Pablo Arbelaez  

**Link**: [PDF](https://arxiv.org/pdf/2606.00108)  

**Abstract**: Global biodiversity is declining at unprecedented rates, yet the tools available to monitor and protect ecosystems remain limited by constraints in power, connectivity, and accessibility. We present SPARROW, a hardware and software open-source platform that integrates solar energy, edge artificial intelligence, and satellite communication to enable continuous, autonomous biodiversity monitoring in remote environments. Each SPARROW node combines a low-power Graphics Processing Unit (GPU) with modular visual, acoustic, and environmental sensors, performing on-device deep learning inference and transmitting summarized results through Low-Earth-Orbit (LEO) satellite or Global System for Mobile Communications (GSM) networks.
We deployed SPARROW across tropical, temperate, and montane ecosystems in Colombia, Peru, Tanzania, and the United States, where it sustained 24/7 operation under variable environmental conditions and collected more than two million images and acoustic recordings in the first 190 days. The system demonstrated robust real-time classification and adaptive power management, achieving full autonomy without on-site human intervention. By integrating renewable energy, on-edge AI, and open-source design, SPARROW lowers the technical and financial barriers to ecological monitoring and establishes a scalable foundation for a distributed, intelligent network of sensors, an emerging "Internet of Living Things" for planetary biodiversity monitoring. 

---
# Motif-based morphology signatures for interpretable ECG screening and monitoring 

**Authors**: Nivedita Bijlani, Mauricio Villarroel  

**Link**: [PDF](https://arxiv.org/pdf/2606.00107)  

**Abstract**: Electrocardiography (ECG) remains central to cardiovascular screening, yet interpretation remains largely manual and episodic. Clinical practice relies on brief resting ECGs and, when required, long-duration ambulatory recordings, both generating data that require resource-intensive review. Consequently, subtle morphological changes or progressive drift preceding clinically apparent abnormalities may go unnoticed. We propose a motif-based framework that defines beat-aligned ECG motifs as interpretable cardiac signatures and quantifies morphological drift and deviation across short and long-term monitoring. Motifs are representative cardiac cycles capturing dominant morphology. We introduce three interpretable drift metrics: deviation from a normal sinus rhythm (NSR), deviation from a personalised baseline, and a motif instability index. Motifs are extracted by selecting beats that minimise Dynamic Time Warping (DTW) distance within fixed windows. We evaluate these metrics on short (PTB-XL) and long-duration (MIT-BIH Arrhythmia) ECG datasets. Interpretability is achieved through representative motif overlays and fiducial-based visualisations, enabling direct inspection of morphological changes. In MIT-BIH, the proposed metrics significantly separated predominantly normal from arrhythmic subjects (p<0.01). In PTB-XL, NSR deviation distinguished normal from abnormal ECGs across major diagnostic subtypes (p<1e-4, Cliff's delta up to 0.93). ECG motifs provide an interpretable representation of cardiac morphology, supporting scalable longitudinal monitoring and early detection of morphology-driven change. 

---
# A Methodological Framework for Explicit Control of the Speed-Accuracy Trade-off in Brain-Computer Interfaces 

**Authors**: Javier Jiménez, Francisco B Rodríguez  

**Link**: [PDF](https://arxiv.org/pdf/2606.00106)  

**Abstract**: Brain-computer interfaces (BCIs) are limited by low signal-to-noise ratio in modalities such as electroencephalography, which requires multiple trials to reliably decode user intentions. This induces a speed-accuracy trade-off, whereby higher accuracy comes at the cost of speed. The speed-accuracy balance is application-dependent, motivating controllable trade-offs. Conventional metrics, such as the Information Transfer Rate, combine speed and accuracy obscuring their dependence and potentially introducing biases. In this study, we propose an evaluation framework independent of classifier, paradigm, and early-stopping strategy that separates speed and accuracy. We employ two measures, Gain (relative speed improvement) and Conservation (relative accuracy preservation), and combine them into a tunable Gain-Cons Balance controlled by {\alpha}, regulating the speed-accuracy trade-off. The parameter adjusts the operating point without modifying the classifier, facilitating deployment across scenarios. The framework was evaluated on P300 event-related potential paradigms using public recordings from 63 subjects as well as multiple classifiers and early-stopping strategies to achieve distinct operating points in speed-accuracy and bitrate. Results show that tuning {\alpha} yields fast, accurate, or balanced BCI behaviours, demonstrating explicit control of the speed-accuracy trade-off. The method supports subject-level performance prediction and improves explainability of BCI behaviour. Further analysis of the Information Transfer Rate reveals a systematic bias toward speed, explained by the proposed framework through the Gain and Conservation measurements. Overall, this work establishes the speed-accuracy trade-off as a controllable design variable validated on public P300-based paradigms, enabling transparent evaluation and application-specific optimization of BCIs. 

---
# Visual-Noise Guided In-Context Distillation for Multimodal Large Language Model Unlearning 

**Authors**: Junkai Chen, Yuhao He, Junxiang You, Ruiqi Liu, Chenyu Wang, Shu Wu  

**Link**: [PDF](https://arxiv.org/pdf/2606.00105)  

**Abstract**: Multimodal Large Language Models (MLLMs) have achieved remarkable progress on vision-language tasks, but they may also memorize and expose sensitive or restricted knowledge, raising concerns about privacy and broader safety risks. Machine Unlearning (MU) provides a promising way to remove targeted undesirable knowledge from trained models without retraining from scratch while preserving general model utility. Nevertheless, effective unlearning in MLLMs remains particularly challenging. Existing training-based methods often struggle to balance unlearning effectiveness and model utility. In contrast, training-free methods such as in-context unlearning preserve model utility by avoiding parameter updates, but they do not remove memorized knowledge at the parameter level and may remain vulnerable to reverse-engineering attacks. More importantly, in-context unlearning is insufficient in multimodal settings, where visual inputs can provide strong conditioning signals and induce undesirable outputs. To address these challenges, we propose Visual-Noise Guided In-Context Distillation (VGID), a distillation-based framework for MLLM unlearning. VGID dynamically constructs an unlearning-oriented teacher distribution from the frozen base model through dual-modal intervention that combines visual perturbation with textual in-context unlearning. The resulting intervention-induced distribution serves as a teacher signal for distillation, guiding the student model toward parameter-level unlearning without requiring external teacher models or explicit undesirable response annotations. Experimental results show that VGID achieves strong unlearning effectiveness while preserving competitive model utility, reducing forget set ROUGE-L by 0.371 with only a 0.055 drop in retain set ROUGE-L in a representative setting. 

---
# PEACE: A Planner-Executor Agent with Constraint Enforcement for UAVs 

**Authors**: Erdem Uysal, Timo Kehrer, Sebastiano Panichella  

**Link**: [PDF](https://arxiv.org/pdf/2606.00104)  

**Abstract**: Foundation models are increasingly used to drive autonomous systems, yet existing approaches either keep the model in a tight control loop, raising latency and hallucination risk, or compile natural language into opaque end-to-end policies that are hard to explain, constraint and require domain-specific datasets and fine-tuning. We propose a planner-executor agent for PX4-based drones that decouples high-level mission planning from low-level control. A large language model performs single-pass task planning, while execution is handled through a structured ROS 2 tool-calling interface bridged to MAVLink. The system constructs a world model by combining modular 2D detectors (e.g., YOLO or vision-language models) with a pinhole depth projection module for 3D object localization. A constraint enforcement layer enforces altitude limits and horizontal geofencing, and bounded replanning enables recovery from execution-time action failures. We position our approach within three common design patterns for foundation-model-based robotics systems and demonstrate its feasibility in PX4 software-in-the-loop simulations in Gazebo. Results highlight improved explainability, constraint enforcement, and reduced LLM calls compared to tightly coupled LLM control. The code, dataset, videos, and other material can be found at the following link: this https URL 

---
# CoCoVideo: The High-Quality Commercial-Model-Based Contrastive Benchmark for AI-Generated Video Detection 

**Authors**: Huidong Feng, Wentao Chen, Jie Chen, Xinqi Cai, Ruolong Ma, Yinglin Zheng, Yuxin Lin, Ming Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2606.00101)  

**Abstract**: With the rapid advancement of artificial intelligence generated content (AIGC) technologies, video forgery has become increasingly prevalent, posing new challenges to public discourse and societal security. Despite remarkable progress in existing deepfake detection methods, AIGC forgery detection remains challenging, as existing datasets mainly rely on open-source video generation models with quality far below that of commercial AIGC systems. Even datasets containing a few commercial samples often retain visible watermarks, compromising authenticity and hindering model generalization to high-fidelity AIGC videos. To address these issues, we introduce CoCoVideo-26K, a contrastive, commercial-model-based AIGC video dataset covering 13 mainstream commercial generators and providing semantically aligned real-fake video pairs. This dataset enables deeper exploration of the differences between authentic and high-quality synthetic videos and establishes a new benchmark for highly realistic video forgery detection. Building on this dataset, we propose CoCoDetect, a detection framework integrating contrastive learning with confidence-gated multimodal large language model (MLLM) inference. An R3D-18 backbone extracts spatio-temporal representations, while a confidence gate routes uncertain cases to an MLLM for reasoning about physical plausibility and scene consistency. Extensive experiments on CoCoVideo-26K and public benchmarks demonstrate state-of-the-art performance, validating the framework's robustness and generalizability. Our code and dataset are available at this https URL. 

---
# CoilDrop-MRI: Self-supervised physics-guided MRI reconstruction with coil dropout 

**Authors**: Tongxi Song, Ziyu Li, Zihan Li, Wen Zhong, Congyu Liao, Yang Yang, Hua Guo, Wenchuan Wu, Qiyuan Tian  

**Link**: [PDF](https://arxiv.org/pdf/2606.00100)  

**Abstract**: Self-supervised deep learning-based methods have shown great promise for accelerated magnetic resonance imaging (MRI) reconstruction, achieving high image quality without requiring fully sampled data for training. These methods typically partition the acquired data into two disjoint subsets to construct input-target pairs for optimizing the reconstruction network. However, existing approaches perform this partition exclusively within the spatial frequency (k-space) domain, leaving the coil dimension unexplored. To enforce full exploitation of signal correlation across receiver coils, we propose CoilDrop-MRI, which applies coil-wise dropout to the input and uses the dropped data as training targets in a self-supervised framework. This method is integrated into unrolled architectures in both image-domain (SENSE) and k-space (SPIRiT) formulations. We further demonstrate its versatility by extending CoilDrop-MRI to multi-shot, phase-corrected diffusion MRI (dMRI) reconstruction. CoilDrop-MRI is extensively validated on multi-site, multi-field-strength (0.3T, 0.55T, and 3T), and multi-modality (T1-weighted, T2-weighted, T2-FLAIR, and dMRI) datasets and consistently outperforms state-of-the-art self-supervised methods, achieving quality comparable to supervised reconstruction methods without requiring fully sampled reference training data. Moreover, CoilDrop-MRI exhibits strong data efficiency and robust generalization across imaging conditions, establishing it as a practical and versatile framework for self-supervised parallel MRI reconstruction. 

---
# Diversity Over Frequency: Rethinking Tool Use in Visual Chain-of-Thought Agents 

**Authors**: Dong-Hee Kim, Reuben Tan, Donghyun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2606.00096)  

**Abstract**: Visual agents employ external visual tools within visual chains of thought to incorporate fine-grained evidence. While prior work has mainly studied these tools in visual search tasks, their role in more complex visual reasoning remains underexplored. In this paper, we move beyond simple visual search tasks to investigate more challenging tasks, including 3D spatial reasoning and medical visual question answering, where agents must integrate tool-acquired local evidence with the global context.
We identify a {tool-use collapse phenomenon: models progressively stop using tools while still achieving higher task accuracy.
Moreover, we observe a clear asymmetry: (i) completely eliminating tool use degrades performance, whereas (ii) incentivizing tool use yields only marginal gains despite substantially increasing usage.
We find that vanilla training and tool-use encouragement both reduce rollout diversity, explaining why higher tool use does not yield stronger reasoning performance.
Motivated by these findings, we add an entropy regularization term to encourage diverse rollout exploration, achieving the best performance despite gradually declining tool usage.
% We further observe similar dynamics on medical VQA, suggesting that tool-use collapse is not limited to 3D spatial reasoning.
Overall, our findings suggest a training-time view of tools as scaffolding, where broader exploration over language generation and visual tool invocation improves reasoning despite tool-use collapse.
Project page: this https URL 

---
# Bridging the 2D-3D Gap: A Hierarchical Semantic-Geometric Map for Vision Language Navigation 

**Authors**: Kailing Li, Tianwen Qian, Lijin Yang, Yuqian Fu, Jingyu Gong, Xiaoling Wang, Liang He  

**Link**: [PDF](https://arxiv.org/pdf/2606.00095)  

**Abstract**: Vision-Language Navigation (VLN) enables embodied agents to reach target locations in unseen environments by following language instructions. Despite recent progress with vision-language models (VLMs), a critical semantic-geometric gap remains: while VLMs excel at language and 2D visual understanding, they struggle with 3D spatial reasoning and fail to capture the causal dynamics between actions and spatial transitions, resulting in unreliable navigation, particularly in zero-shot settings. To bridge this gap, we propose a Hierarchical Semantic-Geometric Map (HSGM) that transforms 3D geometric information into a structured representation compatible with VLMs, effectively linking them to the physical world. Specifically, HSGM is represented as a multi-channel top-down map organized into three levels: (1) geometric level that records navigable regions and obstacles, (2) semantic level that represents objects and their relations, and (3) decision level that supports high-level task reasoning and goal selection. During navigation, the VLM acts as a high-level semantic planner, interpreting the spatial layout encoded in the HSGM to select geometrically valid waypoints, while low-level, collision-free movements between waypoints are executed by a classical path-planning algorithm, fully decoupling semantic reasoning from action execution. Additionally, complex instructions are decomposed into subtasks to alleviate the problem of progress forgetting or hallucinating in long-horizon navigation. Extensive experiments on R2R-CE and RxR-CE benchmarks demonstrate that our zero-shot framework achieves state-of-the-art performance and even outperforms several supervised methods. Code is available at this https URL. 

---
# Diffusion Image Generation with Explicit Modeling of Data Manifold Geometry 

**Authors**: Duoduo Xue, Zhiyu Zhu, Junhui Hou  

**Link**: [PDF](https://arxiv.org/pdf/2606.00094)  

**Abstract**: Image generative models aim to sample data points from the underlying data manifold, a task that requires learning and decoding a dense, low-dimensional, and compact parameterization space. To achieve this, we propose the Data Manifold-aware Image diffusioN moDel (MIND), a novel framework that explicitly models manifold geometry by integrating discrete patch tokenization into the score function of a continuous diffusion model. This approach successfully leverages both the structural quantification capabilities of discrete tokens and the parallel generation flexibility of continuous diffusion. Moreover, we enable end-to-end differentiable training via a novel soft top-$k$ aggregation mechanism and introduce dual-branch high-frequency feature embedding layers to alleviate the spectral bias of transformer backbones on low-dimensional inputs. Furthermore, for inference, we design a multi-stage transition sampling scheme that dynamically adjusts the sampling scheme based on timestep. Extensive experiments on ImageNet 256$\times$256 demonstrate the effectiveness of MIND. After 80-epoch training, our base model achieves an FID of 22.73 without guidance, nearly halving the 43.47 FID of the vanilla DiT-B/2 baseline. The proposed method reduces FID by 15.95 and 9.06 on average compared with the baselines DiT and SiT, respectively. For image generation on ImageNet-256$\times$256 with guidance, the proposed MIND-B with only 130M parameters achieves an FID of 2.06, superpassing the LlamaGen-3B with 3.1B parameters. The proposed MIND-XL with 715M parameters further reduces the FID to 1.95. Our MIND introduces a fresh perspective on diffusion-based image generation, paving the way for future research and innovation in this community. The code will be publicly available. 

---
# Aligning Cellular Sheaves with Classifier Attention for Interpretable Weakly-Supervised Pathology Localization 

**Authors**: Devansh Lalwani, Swapnil Bhat, Maulik Shah  

**Link**: [PDF](https://arxiv.org/pdf/2606.00092)  

**Abstract**: Weakly-supervised classification of whole-slide images with attention-based multiple instance learning (ABMIL) on top of foundation features now reaches near-saturation on Camelyon16 slide-level performance, but the corresponding attention maps are an imperfect localization signal: in clinical interpretation, a model that classifies correctly without firing on the actual lesion is hard to trust. We address this gap with cellular sheaves, which equip each vertex and edge of a graph with a finite-dimensional vector space and consistent linear maps between them, providing a principled way to detect local disagreement on graph-structured data. We apply cellular sheaves to weakly-supervised tumour localization on whole-slide images, combining a sheaf disagreement field with ABMIL. The natural training objective, encouraging consistency between similar features, produces a disagreement field that tracks tissue-level texture rather than diagnostic content. We propose attention-conditional consistency, which uses the classifier's attention to define which neighbouring patches should agree. Joint training of the classifier and the sheaf under this objective produces a disagreement field with patch-level AUC 0.940 on Camelyon16 and raises the attention head from its ABMIL-alone level of 0.717 to 0.953. Two-stage ablation with the classifier frozen at its ABMIL values reaches only 0.727 on the disagreement field and leaves attention at 0.717, confirming that the gain comes from the projector co-adapting under both objectives, not from the loss change in isolation. The trained model transfers without retraining to annotated slides from Camelyon17, maintaining Delta AUC 0.932 +/- 0.083 and attention AUC 0.955 +/- 0.099. The result is an attention map and a sheaf-disagreement map that fire on the same diagnostic regions, giving clinicians two complementary explanations for each slide-level prediction. 

---
# DLLM-JEPA: Joint Embedding Predictive Architectures for Masked Diffusion Language Models 

**Authors**: Sangdae Nam  

**Link**: [PDF](https://arxiv.org/pdf/2606.00091)  

**Abstract**: Joint Embedding Predictive Architectures (JEPAs) have reshaped self-supervised representation learning in vision. The recent LLM-JEPA ported JEPA to autoregressive language models but inherited two steep costs from the causal-attention substrate: it demands explicit multi-view data (e.g., text-code pairs), and it requires two gradient-carrying forward passes per step. We introduce DLLM-JEPA, which pairs JEPA with masked-diffusion language models to eliminate both costs at once. The bidirectional attention of diffusion models yields two semantically distinct views of the same input via different masking rates -- no explicit pairs needed -- and supports a single gradient-carrying forward pass, cutting training FLOPs by 33% relative to LLM-JEPA. DLLM-JEPA improves over diffusion-only fine-tuning in every (task, architecture) combination we evaluate: up to +18.7 pp on LLaDA-8B GSM8K and +11.4 pp on Dream-7B GSM8K, with consistent positive gains on Spider, NL-RX-SYNTH, and Django. Beyond accuracy, DLLM-JEPA exhibits a dual-win property: on LLaDA-8B with the Wide-t configuration, it simultaneously raises GSM8K accuracy (67.1 vs. 65.2, +1.8 pp), drives held-out Wikitext loss below the pre-trained base, and preserves MMLU accuracy at base level across three fine-tuning seeds -- whereas an L2-to-base parameter anchor matches baseline accuracy with no task gain. Layer-wise probing reveals the mechanism: a geometric-functional drift dissociation in which the fine-tuned backbone moves further from the pre-trained weights than the baseline yet forgets less on held-out Wikitext, with the amplification concentrated in middle transformer layers. The pattern appears on Dream-7B as well, indicating the phenomenon is not specific to a single backbone. 

---
# Silent Failures in Physical AI: A Literature Review of Runtime Action Authorization for Autonomous Systems 

**Authors**: Barak Or  

**Link**: [PDF](https://arxiv.org/pdf/2606.00090)  

**Abstract**: Physical AI systems increasingly map multimodal observations, language instructions, and learned world representations into physically consequential actions. Robotics foundation models, vision-language-action models, and world-model-based autonomous systems can condition decisions that move vehicles, robots, drones, and industrial machines. This transition exposes a safety problem that is not fully captured by conventional AI content moderation or by classical robot safety alone: a black-box model may issue a physically consequential action while appearing confident, plausible, and semantically aligned. The resulting failure can be silent, arising from sensor drift, occlusion, state-estimation error, distribution shift, hallucinated affordances, or invalid physical assumptions before downstream hardware controllers detect a violation.
Across embodied foundation models, world models, robotics simulation, embodied safety benchmarks, safe control, runtime assurance, uncertainty estimation, verification, and guardrail evaluation, model capability and safety mechanisms have advanced along largely separate technical tracks. A recurring gap synthesized here is that no single stream surveyed in this review supplies a complete runtime authorization boundary between black-box Physical AI models and physical execution. The resulting analysis develops a bounded problem formulation, a definition of silent physical-action failure, a taxonomy of runtime guardrail functions, and evaluation requirements for comparing guardrails as Physical AI assurance mechanisms. 

---
# Can Predicted Dynamics Exist in the Physical World? 

**Authors**: Barak Or  

**Link**: [PDF](https://arxiv.org/pdf/2606.00089)  

**Abstract**: Predictive Physical AI systems output state rollouts, action chunks, and latent plans, yet a low root-mean-square error (RMSE) does not imply that a particular proposal is physically executable. We formulate physical admissibility as a prediction-control interface: before execution, a decoded proposal is treated as candidate dynamics and evaluated using kinematic, dynamic, and direct-to-composed horizon conditions. Passing is not a certificate of task success; rejection identifies violation of the specified physical envelope and gives a component-level reason. On Hugging Face LeRobot PushT, controlled falsification shows that one-step prediction-RMSE and standardized dynamics residuals reach area under the receiver operating characteristic curve (AUC) 0.982 and 0.972, kinematic-only conditions reach AUC 0.592, and the full gate reaches AUC 0.957 with condition-level attribution. In replay-based intervention experiments, residual-based filters and the full physical-admissibility gate prevent 87-$89% of invalid proposals while preserving mean progress near 0.998. 

---
# Structured Visual Evidence Decomposition for Evidence-Grounded Multimodal Screening of Obstructive Sleep Apnea-Hypopnea Syndrome 

**Authors**: Chen Zhan, Yingchen Wei, Xiaoyu Tan, Jingjing Huang, Xihe Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2606.00087)  

**Abstract**: Effective pre-polysomnography screening for obstructive sleep apnea-hypopnea syndrome (OSAHS) requires combining clinical risk factors with visible craniofacial and neck cues. Directly prompting general-purpose multimodal foundation models for medical yes/no decisions can yield unstable, poorly calibrated outputs. We propose EviOSAHS, an evidence-grounded multimodal reasoning framework that separates image-only anatomical evidence acquisition from final clinical adjudication. Each frontal facial image is decomposed into seven fixed anatomical queries covering the neck, chin, mouth, face/neck fat, lower jaw, midface, and nose. Visual responses are converted into structured evidence cards recording target anatomy, visibility, risk direction, evidence strength, confidence, and a concise summary. These cards are combined with a cleaned clinical profile only in the final stage, where a large language model performs balanced binary screening adjudication. We evaluated EviOSAHS on a 642-subject cohort, mapping normal subjects to screening-negative and mild, moderate, or severe OSAHS subjects to screening-positive. EviOSAHS achieved 88.47% accuracy, 94.86% sensitivity, 93.74% F1-score, and a 5.14% false-negative rate, outperforming clinical-only prompting, direct multimodal prompting, and naive two-stage pipelines under a unified protocol. Ablations showed that seven-question visual decomposition and balanced final adjudication were critical to the high-sensitivity operating point. A question-level audit of 4,494 visual outputs showed a 100% structured parse rate and 93.88% high-visibility rate. EviOSAHS provides an auditable, high-sensitivity workflow for binary pre-polysomnography OSAHS screening, but should be viewed as a triage assistant rather than a diagnostic system. Prospective validation, external testing, and calibrated operating-point control are needed before clinical deployment. 

---
# SentimentLens: Reconciling Sentiment and Ratings via Dual-Modality in the Hospitality Sector 

**Authors**: Dineth Jayakody, Pasindu Thenahandi, Sampath Jayarathna  

**Link**: [PDF](https://arxiv.org/pdf/2606.00084)  

**Abstract**: Online travel platforms generate vast volumes of user-generated hotel reviews, offering rich opportunities to understand traveler experiences at scale. However, transforming unstructured textual feedback into structured, actionable insights remains a challenging task. This paper presents SentimentLens, a scalable analysis system based on Aspect-Based Sentiment Analysis that performs knowledge extraction from unstructured hotel reviews and organizes them into interpretable service categories. SentimentLens integrates aspect term extraction, aspect sentiment classification, semantic category assignment, and multi-level analytical modules to support region-level, hotel-level, and category-level evaluation. The system is designed to operate across different geographic contexts and hospitality settings. To demonstrate its practical utility, we apply SentimentLens to a large real-world dataset of over 10,000 publicly available hotel reviews. Through extensive analysis, the framework reveals how traveler sentiment varies across regions, service categories, and hotel archetypes. We further implement a cross-modal reconciliation of textual sentiment and numerical ratings to identify latent operational conflicts, structural inconsistencies in service quality, and high-impact improvement opportunities using importance--performance and entropy-based analyses. The results show that SentimentLens effectively transforms large-scale unstructured reviews into actionable intelligence, supporting data-driven decision-making for hospitality management and tourism policy. While demonstrated using a national case study, the proposed system is generalizable to other destinations and review-driven service domains. 

---
# From Demonstrations to Rewards: Test-Time Prompt Optimization for VLM Reward Models 

**Authors**: Christian Gumbsch, Leonardo Barcellona, Lennard Schünemann, Platon Karageorgis, Andrii Zadaianchuk, Zehao Wang, Sergey Zakharov, Fabien Despinoy, Rahaf Aljundi, Efstratios Gavves  

**Link**: [PDF](https://arxiv.org/pdf/2606.00083)  

**Abstract**: Reinforcement learning relies on accurate reward functions, which are often hand-crafted or even unavailable in real-world applications, such as robotics. Recent work has explored the zero-shot reasoning capabilities of pre-trained Vision-Language Models (VLMs) as reward models. However, without careful prompt engineering, these approaches tend to produce suboptimal rewards, where false positive predictions can severely degrade downstream policy learning. In robotics, limited datasets comprising expert demonstrations are often collected to bootstrap policy learning. This scenario provides an opportunity to optimize a reward model prior policy training. We propose Demo2Reward a test-time adaptation technique to optimize the language instruction of a reward model based on a few demonstrations (3-10 trajectories) to reduce false positives while preserving true positives. Crucially, this requires no additional model training or computation resources during policy learning. We show that Demo2Reward consistently outperforms existing zero- and few-shot VLM reward models across a range of simulated robotic tasks and policy backbones. Finally, we demonstrate that Demo2Reward effectively transfers to a real-world robotic learning scenario, enabling policy learning without manually engineering a reward function. 

---
# Hoeffding Concept Bottleneck Models with Applications to Overhead Images 

**Authors**: Clément Bénard, Manon Arfib, Christophe Labreuche, Victor Quétu  

**Link**: [PDF](https://arxiv.org/pdf/2606.00082)  

**Abstract**: Explainability of deep learning algorithms is critical for computer-vision applications with high-stake decisions. Concept bottleneck models (CBM) have recently shown promising performance to provide explainable and accurate predictions for classification problems, based on a bottleneck of high-level concepts. Existing CBM methods rely on a linear aggregation of the concept scores to compute predictions. However, a large number of concepts is often used in this linear approach, which undermines explainability and favors information leakage. In general, the underlying relation between concepts and output logits is not linear. Therefore, we introduce Hoeffding Concept Bottleneck Models (HCBM), which build on the Hoeffding functional decomposition of gradient-boosted trees to provide non-linear and sparse aggregations of concept scores, and generate compact predictions using prime implicants. HCBM are proved to be robust to interconcept leakage, and outperform standard linear CBM in practice, as shown in extensive experiments. Beyond classification, HCBM can be adapted to object detection, and we focus on a challenging case with overhead images to show the high performance of HCBM in these settings. 

---
# DAStatFormer: A Hybrid Multibranch Transformer with Statistical Feature Integration for DAS-Based Pattern Recognitions 

**Authors**: Michel Dione, Jerry Lonlac, Hélène Louis, Anthony Fleury, Stephane Lecoeuche  

**Link**: [PDF](https://arxiv.org/pdf/2606.00081)  

**Abstract**: Distributed Acoustic Sensing (DAS) enables large-scale monitoring through optical fibers, but its high dimensionality and complex spatio-temporal patterns make event classification demanding. Existing deep learning approaches-CNNs, recurrent models, and Transformer variants-either fail to capture long-range dependencies or require processing raw DAS matrices at prohibitive cost. We propose DAStatFormer, a hybrid multibranch Transformer that combines compact multidomain statistical features with Gated Transformer Networks. Instead of raw signals, we extract 24 ANOVA-selected attributes per channel from the temporal, waveform, and spectral domains, reducing data size by orders of magnitude while preserving discriminative information. Each domain is processed via dedicated step-wise and channel-wise attention branches, fused by an adaptive gating mechanism. Experiments on the open $\Phi$-OTDR benchmark and a real-scenario DAS dataset show that DAS-tatFormer achieves up to 99.4% accuracy and near-perfect real-world performance, while using significantly fewer parameters and lower inference cost than models such as DASFormer and DeepViT. These results demonstrate its suitability for scalable, real-time DAS-based monitoring. We release our code at this https URL 

---
# Planktonzilla: Multimodal dataset and models for understanding plankton ecosystems 

**Authors**: Alan Gerson Contreras Montanares, Luis Valenzuela, Luis Martí, Nayat Sanchez-Pi  

**Link**: [PDF](https://arxiv.org/pdf/2606.00080)  

**Abstract**: Marine plankton underpin aquatic food webs and play a key role in global CO2 sequestration, making reliable species identification critical for understanding ocean health and climate feedbacks. Existing classification models perform well on individual collections but fail to generalize across instruments and environments due to isolated training datasets and inconsistent labels. To address this, we introduce Planktonzilla-17M, a unified dataset consolidating publicly available plankton image collections spanning thirteen imaging systems. It comprises 17.4 million images with standardized taxonomy and geo-environmental metadata, including 3.74 million plankton images spanning over 602 taxonomic classes, of which 201 are identified at the species level, making it the largest and most comprehensive plankton image dataset to date. Using this large-scale dataset, we perform a controlled comparison between supervised and CLIP-style image--text training on a shared ViT backbone. We find that a supervised classifier matches or exceeds CLIP-style training when trained using taxonomic lineage as text. We further observe that BioCLIP and BioCLIP2 perform poorly on plankton in zero-shot and few-shot settings. Leveraging Planktonzilla-17M improves plankton classification performance, highlighting the limitations of current biological foundation models in marine imaging domains. 

---
# BitsMoE: Efficient Spectral Energy-Guided Bit Allocation for MoE LLM Quantization 

**Authors**: Jiayu Zhao, Zihan Teng, Minhao Fan, Tianrui Ma, Wentao Ren, Song Chen, Weichen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.00079)  

**Abstract**: Mixture-of-Experts (MoE) large language models reduce per-token computation through sparse expert activation, but their deployment remains memory-intensive because all expert weights must be kept resident in memory. Existing MoE compression methods struggle in the ultra-low-bit regime: pruning irreversibly removes model capacity, while coarse-grained quantization fails to allocate bits according to heterogeneous expert and weight-direction importance. We propose BitsMoE, a spectral-energy-guided bit-allocation framework for MoE LLM quantization. BitsMoE decomposes each MoE layer by SVD into a shared basis and expert-specific spectral factors, retaining the shared basis without quantization to preserve common cross-expert structure and using the expert-specific factors as fine-grained quantization units. To determine the bit-width of each unit, BitsMoE formulates spectrum-wise mixed-precision quantization as an activation-aware reconstruction surrogate and solves an integer linear program that minimizes estimated reconstruction loss under a fixed bit budget. Experiments across multiple MoE LLMs show that BitsMoE substantially reduces downstream task accuracy degradation in ultra-low-bit regimes. Under 2-bit quantization on Qwen3-30B-A3B-Base, BitsMoE accelerates quantization by 12.3$\times$, improves average accuracy by 27.83 percentage points, and increases decoding speed by 1.76$\times$ over GPTQ. Our model and code are publicly available at this https URL. 

---
# Flow-Based Generative Modeling for Optimizing Sampling Policies in Compressed Sensing Applications 

**Authors**: Roman Pavelkin, Luis A. Zavala-Mondragon, Christiaan G. A. Viviers, Fons van der Sommen  

**Link**: [PDF](https://arxiv.org/pdf/2606.00078)  

**Abstract**: Numerous modern applications in signal processing and medical imaging necessitate acquiring high-dimensional signals under tight resource constraints. Traditional sampling theory suggests that accurate signal reconstruction requires a number of measurements proportional to the signal's ambient dimension, a requirement often too expensive or impractical. Compressed sensing challenges this notion by demonstrating that sparse signals can be recovered with fewer measurements, provided the measurement operator meets certain conditions. This proof-of-concept study presents a task-aware flow-based generative framework -- a reformulation of the conventional Flow Matching training paradigm with a flow model trained to optimize subsampling in compressed sensing applications. We establish the fundamental feasibility of the proposed framework of learning subsampling masks that substantially enhance the performance of compressed sensing for image classification, image reconstruction, and MRI acceleration. For the image reconstruction task, our method demonstrated state-of-the-art performance, achieving Peak Signal-to-Noise Ratio of 25.17 dB at the subsampling rate of 5\% on the CelebA dataset and 29.24 dB when reconstructing $8\times$ accelerated MRI measurements (fastMRI dataset) with the minimal computational overhead. These results highlight the effectiveness of task-conditioning within generative flow models and reveal a promising direction for representation learning strategies. Overall, the proposed framework offers a unified, flexible approach to designing data- and task-driven sensing schemes that can be potentially adapted to a broad range of inverse problems. 

---
# Improved Belief-Attention in Vision Task 

**Authors**: Guoqiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.00077)  

**Abstract**: Recently, Belief-Attention \cite{Guoqiang25BeliefAttention} has been proposed by first performing an orthogonal projection of the softmax-based weighted summation of $V$ vectors with respect to the original $V$ vectors and then taking the perpendicular component as the residual signal in Transformer for performance improvement. In this paper, we first conduct an ablation study showing the projected component also carries information about the token correlation, which should not be ignored. We then propose to extend Belief-Attention by making use of both the perpendicular and projected components. In particular, the projected component goes through certain activation function and then a linear mapping before merging with the considered token. Conceptually speaking, the neural block for the projected component can be viewed as a two-layer feedforward network (FFN) within the new attention block. It is also noted that standard attention captures the token correlation via the inner-product matrix $QK^T$. We propose to introduce an additional inner-product matrix $ZZ^T$ to $QK^T$ to capture richer token correlation. We refer to the new module as Belief2-Attention. It can be easily shown that Belief2-Attention is more expressive than standard Attention. We then verify the effectiveness of Belief2-Attention for vision tasks of image classification and segmentation. 

---
# CLSP-REQA: A Real-Time Quality-Aware Closed-Loop Seizure Prediction Framework with Mamba-BiLSTM and Confidence-Gated Intervention 

**Authors**: Mufeng Chen, Qi Wu, Bingchao Huang, Xiwen Lai, Zekai Chen, Xinge Ouyang, Quansheng Ren  

**Link**: [PDF](https://arxiv.org/pdf/2606.00074)  

**Abstract**: Reliable seizure prediction is a prerequisite for closed-loop neurostimulation therapy, yet existing methods rarely account for the variability in EEG signal quality encountered in real-world deployment, and the overwhelming majority adopt non-strict evaluation protocols that overestimate generalisation performance. We propose CLSP-REQA (Closed-Loop Seizure Prediction with Real-time EEG Quality Assessment), a unified framework that embeds a lightweight signal quality estimator directly within the prediction pipeline. A Real-time EEG Quality Assessment (REQA) module runs in parallel with a Mamba-BiLSTM backbone, producing a scalar quality score q in [0,1] that modulates output confidence through a tiered non-linear fusion function (ECLO). Under strict cross-patient evaluation on the CHB-MIT Scalp EEG Database (n = 23 subjects, 198 seizures), CLSP-REQA achieves an AUC-ROC of 0.7426 +- 0.0199, outperforming the unadapted cross-patient baseline of 0.69 reported by Jemal et al., using only 16 EEG channels compared to 23 in prior work, and without requiring any target-patient data or domain adaptation. On the SIENA Scalp EEG Database (n = 14 subjects, 47 seizures), CLSP-REQA achieves AUC 0.7012 +- 0.0249, substantially surpassing the best domain-adapted cross-patient result of 0.61 on the same dataset, demonstrating strong cross-dataset generalisation. The framework outputs a structured four-tuple (p, q, c, Phi_SHAP) directly compatible with closed-loop neurostimulator interfaces. 

---
# Rare Events, Real Signals: Functional Ensembles as Units of Computation in Deep Spiking Networks 

**Authors**: Aditi Aravind, Konstantinos Ladakis, Mario Alexios Savaglio, Stelios M. Smirnakis, Maria Papadopouli  

**Link**: [PDF](https://arxiv.org/pdf/2606.00073)  

**Abstract**: We investigate how internal representations emerge across hierarchical processing systems by introducing a neuroscience-inspired framework for analyzing deep spiking neural networks (SNN) through the lens of functional connectivity. Drawing on concepts from systems neuroscience and information theory, we form the first-order functionally-connected (1FC) group of a neuron based on its statistically significant pairwise correlations with neurons from the previous layer of a trained SNN architecture. We then track its response properties during inference under various conditions. Our analysis shows that several principles of functional connectivity previously observed in biological cortex are preserved in spiking ResNet architectures. These 1FC ensembles display interesting properties: their aggregate cofiring reliably predicts downstream neuronal responses through a robust, ReLU-like input-output relationship, whose gain scales systematically with ensemble size. Reliable encoding of the presented class emerges only during high 1FC cofiring events, which themselves occur infrequently, indicating that informative representations are concentrated in rare but highly coordinated activity patterns. Under uniform random noise or adversarial perturbations, these response profiles are disrupted, particularly in early and intermediate layers. This enables a targeted high-resolution interrogation at specific nodes and pathways. We showed that the functional connectivity structure is shaped by learning and this structure breaks under weight permutation. These establish 1FC ensembles as a functionally meaningful substrate for input encoding and information transfer, with potential implications in designing targeted fine-grained diagnostics on the information flow. 

---
# Beyond Text and Tables: Vision-Language Model Integration in ComProScanner for Extracting Materials Data from Scientific Figures with High Accuracy 

**Authors**: Aritra Roy, Enrico Grisan, Chiara Gattinoni, John Buckeridge  

**Link**: [PDF](https://arxiv.org/pdf/2606.00065)  

**Abstract**: Automated extraction of materials composition-property data from scientific literature has advanced considerably with the development of large language model-based pipelines; however, existing frameworks remain limited to textual and tabular content, overlooking the substantial proportion of quantitative property data reported exclusively in scientific figures. Here, we extend ComProScanner, a fully end-to-end multi-agent framework for automated composition-property database construction, with a native vision-language model (VLM) based figure extraction capability. The extension introduces a FigureExtractor utility for caption-keyword-based figure filtering across all supported publishers, and a GraphExtractorTool agent that passes extracted figures to a configurable VLM to recover composition-property pairs from scientific charts and plots. Four VLMs are selected for evaluation on the basis of the LMArena Diagram leaderboard with an input cost criterion of less than \$1.50 per million tokens. Benchmarking on 50 piezoelectric ceramic articles from the established $d_{33}$ test corpus demonstrates that Gemini-3-Flash-Preview achieves the highest performance with a composition accuracy of 0.97 and a normalised F1 score of 0.97, whilst remaining the most cost-effective model among the four evaluated. We additionally introduce a range-based value error threshold parameter into the evaluation framework, providing a more physically meaningful assessment of numeric property values extracted from figures than exact value matching. These contributions establish VLM-integrated ComProScanner as the first materials-specific, fully automated, multimodal literature mining platform capable of extracting structured composition-property data from text, tables, and figures within a single unified pipeline. 

---
# Physics-Informed Neural Networks for Radial Consolidation of Combined Electroosmotic, Vacuum and Surcharge Preloading Considering Smear Effects 

**Authors**: Dong Li, Yapeng Cao, Shuai Huang, Yujun Cui, Haiping Fu, Lu Yang, He Wei  

**Link**: [PDF](https://arxiv.org/pdf/2606.00056)  

**Abstract**: This study develops a dimensionless multi-domain physics-informed neural network (PINN) framework for electro-osmotic radial consolidation considering smear effects and combined vacuum and surcharge loading. Three PINN-based models are investigated: a standard soft-constrained PINN (Std-PINN), a modified gated PINN (Mod-PINN), and a modified gated PINN with hard-constraint boundary encoding (Mod-HC-PINN). The models are evaluated against FEM reference solutions under four loading cases, including constant vacuum, exponential vacuum, exponential vacuum with ramp surcharge, and exponential vacuum with cyclic haversine surcharge. The results indicate that the gated architecture applied in Mod-PINN improves the resolution of steep pressure gradients near the cathode and smear-zone interface under constant vacuum loading. Under time-dependent loading, the soft-constrained Mod-PINN shows reduced accuracy because it must learn multiple competing objectives simultaneously. The Mod-HC-PINN mitigates this issue by embedding the cathode boundary and initial conditions into the output structure, thereby reducing the optimization burden and improving physical consistency. The Mod-HC-PINN achieves MAE values of 0.43, 0.41, and 0.27 kPa for the exponential vacuum, ramp surcharge, and cyclic surcharge cases, respectively. Sensitivity analyses further demonstrate that the proposed framework remains robust across practical ranges of network architecture, collocation density, and permeability contrast. 

---
# From Human Videos to Robot Manipulation: A Survey on Scalable Vision-Language-Action Learning with Human-Centric Data 

**Authors**: Zhiyuan Feng, Qixiu Li, Huizhi Liang, Rushuai Yang, Yichao Shen, Zhiying Du, Zhaowei Zhang, Yu Deng, Li Zhao, Hao Zhao, Zongqing Lu, Oier Mees, Marc Pollefeys, Jiaolong Yang, Baining Guo  

**Link**: [PDF](https://arxiv.org/pdf/2606.00054)  

**Abstract**: Recent progress in generalizable embodied control has been driven by large-scale pretraining of Vision-Language-Action (VLA) models. However, most existing approaches rely on large collections of robot demonstrations, which are costly to obtain and tightly coupled to specific embodiments. Human videos, by contrast, are abundant and capture rich interactions, providing diverse semantic and physical cues for real-world manipulation. Yet, embodiment differences and the frequent absence of task-aligned annotations make their direct use in VLA models challenging. This survey provides a unified view of how human videos are transformed into effective knowledge for VLA models. We categorize existing approaches into four classes based on the action-related information they derive: (i) latent action representations that encode inter-frame changes; (ii) predictive world models that forecast future frames; (iii) explicit 2D supervision that extracts image-plane cues; and (iv) explicit 3D reconstruction that recovers geometry or motion. Beyond this taxonomy, we highlight three key open challenges in this area: structuring unstructured videos into training-ready episodes, grounding video-derived supervision into robot-executable actions under embodiment and viewpoint heterogeneity, and designing evaluation protocols that better predict real-world deployment performance and transfer efficiency, thereby informing future research directions. A curated list of papers and resources is available at this https URL. 

---
# Business Utility of Large Language Models as Exploratory Data Analysis Agents 

**Authors**: Rafał Łabędzki, Patryk Miziuła, Hubert Rutkowski, Szymon Betlewski, Cezary Depta, Szymon Janowski, Jarosław Kochanowicz, Jan Kanty Milczek  

**Link**: [PDF](https://arxiv.org/pdf/2606.00051)  

**Abstract**: Large Language Models (LLMs) are increasingly used in analytical workflows, but their suitability as exploratory data analysis (EDA) agents in business settings remains uncertain. In practice, a deployable EDA agent must provide not only useful average performance but also sufficient repeatability to support trust in its outputs. We evaluate this requirement in a controlled, business-relevant benchmark built on an agent-based supply chain simulation. The task is to identify supplier-product combinations responsible for low quality and downstream sales loss by reasoning from indirect operational traces rather than from explicit labels. Fifteen model-variant configurations from eight model families were evaluated under four experimental conditions that varied data representation, prompt clarity, and signal strength, with five trajectories per condition. Outputs were scored against deterministic ground truth using the Jaccard index and assessed through a framework that combines mean score (ms), coefficient of variation (CV), exploratory cross-condition significance tests, and Business utility, a risk-adjusted metric that we propose to summarise quality and repeatability in a single operational measure. The results show that most configurations are not reliable enough for autonomous EDA use, even when their average scores appear acceptable. GPT-5.4 with extra-high reasoning effort achieved the strongest overall profile, with an experiment-averaged ms of 0.8748 and an experiment-averaged Business utility of 0.6952, while the next-best configurations lost substantially more utility after variability discounting. Our findings suggest that evaluation of EDA agents should treat average quality, repeatability, and condition sensitivity as complementary dimensions of operational trustworthiness. 

---
# Measuring and Mitigating Bias in Code Generated by Large Language Models 

**Authors**: Yuxi Chen, Yutian Tang, Timothy Storer  

**Link**: [PDF](https://arxiv.org/pdf/2606.00049)  

**Abstract**: Large language models (LLMs) are widely recognised for their applications in natural language generation and are increasingly used for code generation tasks. However, concerns about bias in their generated outputs remain significant. This paper focuses on GPT-4o and Gemini, mainstream tools for code generation, and proposes a framework for evaluating bias in LLM-generated code, specifically examining the influence of protected attributes, prompts and web-search capability. We use two metrics: the code bias score (CBS) and the attribute change ratio (ACR), to quantify the prevalence of bias and the degree of influence of different attributes, respectively. In addition, we investigate four lightweight mitigation strategies: Few-Shot, Chain-of-Thought, Few-Shot Chain-of-Thought, and Multi-agent, aimed at mitigating bias in generated code. Our findings reveal that bias remains prevalent across different protected attributes and datasets even after applying mitigation strategies, highlighting the need for more effective approaches to reduce bias in AI-driven code generation systems. 

---
# Comprehensive AI governance requires addressing non-model gains 

**Authors**: Arthur Goemans, Dan Altman, Noemi Dreksler, Jonas Freund, Milan Gandhi, Zhengdong Wang, Sarah Cogan, Sebastien Krier, Demetra Brady, Lewis Ho, Allan Dafoe  

**Link**: [PDF](https://arxiv.org/pdf/2606.00047)  

**Abstract**: Frontier AI governance often centres on the model-level governance paradigm, which assumes that a model's capability profile is primarily a function of the compute and data used during training. This position paper argues that model-level governance becomes less effective when capability progress is increasingly driven by "non-model gains"--improvements that are independent from advances in the base model. We formalise the concept of non-model gains and provide a taxonomy of three distinct vectors of capability gain: inference gain (scaling compute at test-time), systems gain (post-training enhancements such as scaffolds), and asset gain (enhancing a model with restricted assets). We demonstrate how these vectors--alongside potential future impacts from embodiment, continual learning, and AI diffusion--may undermine risk management strategies that hinge mostly on pre-deployment evaluation and mitigation. We provide an overview of governance approaches that go beyond the model level: system, entity, agent, and cloud governance. Finally, we emphasise the importance of societal resilience as a complement to these governance layers. 

---
# When Jokes Cross the Line: Analyzing Regular Humor and Dark Humor in YouTube Shorts 

**Authors**: Sydney Johns, Sanjeev Parthasarathy, Shantnu Bhalla, Vaibhav Garg  

**Link**: [PDF](https://arxiv.org/pdf/2606.00046)  

**Abstract**: Video platforms such as YouTube have reshaped how users engage with entertainment and information, emphasizing brief, highly engaging content such as Shorts. Within this ecosystem, certain content occupies a gray area where it remains allowed but may still have unintended negative effects on some audiences. To study this problem, we introduce TwistedHumor, a dataset of 1,211 YouTube Shorts paired with 33,041 related comments, with hand annotations for humor presence, humor type, harm, topic, rhetorical devices, and stand up context. Beyond dataset creation, we present a multi view analysis of how humor and harm appear in short form social media. Using LLooM based concept induction over video descriptions, we find that dark humor frequently clusters around themes of critique, coping, awkwardness, and identity expression rather than appearing as a single uniform category. We further analyze audience response through linked comments and show that regular humor is associated with more positive sentiment, while dark humor receives more mixed, neutral, and sometimes more toxic reactions. Finally, we evaluate large language models against human annotations and find that they perform better on stand up comedy compared to shorter jokes. Together, these results position TwistedHumor not only as a new benchmark, but as an empirical study of the gray area between humor and harm in short form video, highlighting the need for context aware moderation and more robust multimodal evaluation. 

---
# Algorithmic Authority and the Clinical Standard of Care 

**Authors**: Aizierjiang Aiersilan  

**Link**: [PDF](https://arxiv.org/pdf/2606.00044)  

**Abstract**: The integration of artificial intelligence into clinical medicine creates a fundamental tension between algorithmic probabilistic reasoning and the experiential intuition of expert physicians; applying Lawrence Lessig's \enquote{Code is Law} framework, I argue that the architecture of clinical AI systems already functions as de facto medical regulation, reshaping liability and the standard of care. Reframing AI \enquote{hallucination} as structurally analogous to well-documented human cognitive failures such as confirmation bias and premature diagnostic closure, I show that both failure modes demand a unified governance response. I therefore propose a dialectical standard of care that treats the integrated AI-physician dyad as the singular responsible diagnostic entity, mandating the synthesis of algorithmic precision with human interpretive authority within robust data governance and patient privacy frameworks. 

---
# Improving Hospital Process Management through Process Mining: A Case Study on COVID-19 Clinical Pathways 

**Authors**: Pasquale Ardimento, Mario Luca Bernardi, Marta Cimitile, Samuele Latorre  

**Link**: [PDF](https://arxiv.org/pdf/2606.00041)  

**Abstract**: This study analyzes COVID-19 care pathways using the COVID Data for Shared Learning dataset. We build a transparent, reproducible pipeline that transforms heterogeneous clinical tables into a process-mining-ready event log and applies discovery, declarative conformance checking, and outcome analysis. The reconstructed pathways highlight the monitoring backbone of inpatient care, variability at the Emergency department-admission interface, and outcome differences driven by age and exposure to intensive care units. These insights support triage standardization, capacity planning, and step-down coordination from intensive care units to lower-acuity wards, showing how process mining can inform evidence-based hospital governance. 

---
# Tracing GenAI Literacy: Uncovering Student-AI Interaction Patterns in Academic Writing through Epistemic Network Analysis 

**Authors**: Angxuan Chen, Jiyou Jia  

**Link**: [PDF](https://arxiv.org/pdf/2606.00040)  

**Abstract**: As Generative AI (GenAI) becomes integral to education, fostering GenAI literacy is critical. However, current assessments largely rely on self-reported scales, lacking insights into how literacy manifests in actual learning processes. This study leverages Learning Analytics (LA) to bridge this gap. We collected interaction logs from 162 university students engaged in a GenAI-assisted abstract writing task. Using Epistemic Network Analysis (ENA), we modeled and compared the questioning strategies of students with varying GenAI literacy levels. Preliminary results reveal distinct interaction signatures: high-literacy students engage in iterative refinement and strategic questioning, while low-literacy students rely on direct generation commands. This work contributes to the workshop by demonstrating how process data can characterize GenAI literacy, paving the way for data-driven literacy assessment and real-time interventions. 

---
# Beyond Categories of Caste: Examining Caste Bias and Morality in Text-to-Image AI Models 

**Authors**: Divyanshu Kumar Singh, Dipto Das, Deepika Rama Subramanian, Koustuv Saha, Stephen Voida, Bryan Semaan  

**Link**: [PDF](https://arxiv.org/pdf/2606.00039)  

**Abstract**: Text-to-Image (T2I) models have shown promising utility across various domains. However, such models are also amplifying harmful societal biases in their outputs. In the context of South Asia, recent work has shown caste biases and stereotypes are being perpetuated through Generative AI (GenAI) systems. While this research offers extremely relevant insight into invisibilized narratives of caste discrimination through the GenAI system, they often treat caste as an identity category. Therefore, in this work we shift our ontology to focus on the relational aspect of caste. This enables us to develop a more nuanced understanding of the mechanics of caste discrimination by and through T2I models. Combining an algorithmic audit with critical discourse analysis, we draw on a conceptual frame challenging Brahminical Normativity to show how caste biases are perpetuated beyond the simple binaries of upper vs lower-caste categories. Our contributions are two-fold. Beyond challenging the categorical understanding of caste as a category, we propose an anti-caste approach to tackle the issue of caste bias and fairness in AI systems. 

---
# Beyond Tool Adoption: A Practical Five-Stage Developmental Continuum for AI Literacy in Higher Education 

**Authors**: J. Paul Liu, Rachel Levy  

**Link**: [PDF](https://arxiv.org/pdf/2606.00038)  

**Abstract**: Artificial intelligence (AI) literacy is increasingly recognized as a foundational competency for all university graduates. Yet students' engagement with AI tools often clusters at two problematic extremes: avoidance driven by fear, mistrust, ethical concern, or lack of access, and uncritical reliance that produces fluent output while masking misunderstanding. Existing AI literacy frameworks provide valuable competency definitions, but most offer limited guidance for diagnosing where learners begin and how they progress toward responsible, critical engagement. This paper proposes a five-stage AI Literacy Continuum -- 1) Not Yet Engaged, 2) Uncritical Use, 3) Informed Use, 4) Critical Evaluation, and 5) Improvement -- that describes developmental orientations toward AI use in higher education. The continuum complements dimensional frameworks by providing educators with a practical diagnostic and instructional pathway aligned with international frameworks, including UNESCO and OECD. We present a design-based implementation case from North Carolina State University, where credit-bearing courses and intensive hands-on workshops engaged more than 330 participants between Fall 2024 and Spring 2026. Because the implementation did not use a validated pre/post instrument or comparison group, we frame the findings as observational and practice-based: participants exhibited behaviors consistent with movement from non-engagement or uncritical use toward informed engagement, while sustained and discipline-embedded experiences produced stronger evidence of critical evaluation and improvement-oriented practice. We discuss curricular pathways, equity considerations, assessment strategies, and argue that AI literacy should be understood not as tool adoption alone but as a developmental capacity to understand, evaluate, and responsibly apply AI systems in disciplinary and societal contexts. 

---
# Update Opacity: Epistemic Accessibility and Governance Under AI System Change 

**Authors**: Andrea Ferrario, Joshua Hatherley  

**Link**: [PDF](https://arxiv.org/pdf/2606.00037)  

**Abstract**: Machine learning models embedded in deployed AI systems are routinely updated to maintain correct functioning over time. Yet such updates can generate update opacity: users may not be able to understand why the same input now yields a different output. We argue that update opacity is best understood as a diachronic failure of epistemic accessibility: the problem is that materially relevant changes may fail to remain accessible to human users in forms that support understanding, calibrated reliance, and appropriate action under real role- and time-specific constraints. This makes update opacity a governance problem. Not all change is equally relevant, and disclosing every update would itself undermine use through overload. To address this problem, we combine two complementary governance approaches: the EU AI Act, which helps specify the system-level perimeter of normatively relevant change, and Machine Learning Operations, which provides operational tools for tracking and comparing change over time. On this basis, we propose a framework that models system change through trustworthiness profiles and trustworthiness levels, and uses threshold-based disclosure to surface materially relevant within-envelope change to different stakeholders over time. We illustrate the approach with a medical AI example and derive practical implications for lifecycle documentation, post-market monitoring, and update disclosure. 

---
# Make Mechanistic Interpretability Auditable: A Call to Develop Guidelines via Continuous Collaborative Reviewing 

**Authors**: Michael Lan, Narmeen Fatimah Oozeer, Chaithanya Bandi, Philip Quirke, Austin Meek, Fazl Barez, Amirali Abdullah  

**Link**: [PDF](https://arxiv.org/pdf/2606.00033)  

**Abstract**: While mechanistic interpretability (MI) has produced important insights into neural network internals, the field has yet to establish a standardized system to audit experiments. As such, many of its findings remain underutilized in safety-critical applications such as medical AI and autonomous systems, as stakeholders cannot certify their validity. Recent work demonstrates this concretely: two papers found conflicting conclusions for the same behavior, and a third study revealed that both were partially correct but incomparable due to methodological inconsistencies. Without standardized auditing, such ambiguities hinder adoption in high-stakes contexts requiring strong correctness guarantees. We call for the MI community to work towards developing a novel reviewing system that complements peer review via: (1) Continuous reviewing supported by a \emph{Collaborative Reviewing Platform} where meta-science results and discussions (such as critiques, negative results, post-hoc extensions, reproductions, replications, and partial results) that fit outside of papers are organized and discussed, allowing for comments and revisions to be made at any time (2) Generalizing good practices found on this platform into expert-verified guidelines and protocols to improve auditing efficiency, and (3) Source-based auditing systems that track arguments which claims depend on. This position paper encourages constructive debate over the necessity, design and implementation of such a framework, providing early concrete examples to help catalyze these dialogues. Overall, we propose that auditing MI itself is essential for its application in AI safety, industry, and governance. 

---
# LLMs for Cardiovascular Risk Prediction from Structured Clinical Data 

**Authors**: Jeba Maliha, Md Rafiul Kabir  

**Link**: [PDF](https://arxiv.org/pdf/2606.00031)  

**Abstract**: Coronary artery disease (CAD) remains one of the leading causes of death globally, highlighting the need for reliable predictive systems to support early diagnosis and risk assessment. While traditional machine learning models perform well on structured clinical data, large language models (LLMs) present new possibilities to interpret medical information expressed in natural language. In this work, we develop a hybrid framework that bridges structured clinical data and natural-language representations for CAD prediction. Using a publicly available dataset of 1,190 patient records with 11 clinical attributes, structured variables are converted into interpretable feature representations and synthetic clinical narratives using LLMs. A validation pipeline performs reverse extraction of clinical variables and computes a consistency score with the original records, achieving an average fidelity of 94.61%. We then evaluate four conventional machine learning models and compare their performance with LLM-based classification under zero-shot and few-shot prompting settings. We use two LLMs here, GPT and Gemini. Experimental results show that Random Forest achieves the highest accuracy. Despite this advantage, LLM-based classification remains beneficial in real-world clinical settings. This is because LLMs operate directly on natural language patient descriptions, meaning that sensitive numerical patient data such as exact lab values, blood pressure readings, and diagnostic codes are kept private. Findings suggest that combining structured clinical data with LLM-generated narratives can enable new directions for hybrid clinical prediction systems. 

---
# TCAR-Gen: Temporal Graph Retrieval with Evidence Fusion for Knowledge-Grounded Generation 

**Authors**: Sidra Nasir, Muhammad Noman Zahid, Rizwan Ahmed Khan  

**Link**: [PDF](https://arxiv.org/pdf/2606.00029)  

**Abstract**: Retrieval-augmented generation systems struggle with temporal reasoning and evidence fusion when answering complex questions over historical criminal case narratives. Existing approaches either retrieve independently of query semantics or fail to integrate multiple evidence sources coherently. We propose Temporal Context Augmented Retrieval Generation (TCAR-Gen), a framework that combines query-conditioned graph neural networks, temporal evidence fusion, and chain-of-trees reasoning to ground answer generation in retrieved evidence. On the Victorian Crime Diaries benchmark, TCAR-Gen achieves 0.3738 Recall@5, outperforming Vanilla RAG, Temporal RAG, GraphRAG-C, and GraphRAG-T across seven query types including multi-hop reasoning and counterfactual questions. Ablation studies reveal that the context graph, temporal penalty mechanism, and query conditioning are critical components. Cross-model evaluation across five language model (GPT-OSS 20B to TinyLlama 1.1B) demonstrates that TCAR-Gen maintains robust retrieval coverage at smaller model scales, though generation quality degrades substantially with reduced model capacity. Our work shows that explicit temporal modelling and multi-branch evidence fusion are essential for faithful, reasoning-intensive question answering over knowledge-grounded corpora. 

---
# A Multi-Domain Red Teaming Framework for Safety, Robustness, and Fairness Evaluation of Medical Large Language Models 

**Authors**: Andrei Marian Feier, Veysel Kocaman, Yigit Gul, Ahmet Korkmaz, Alexander Thomas, Aleksei Zakharov, Jay Gil, Mehmet Butgul, David Talby  

**Link**: [PDF](https://arxiv.org/pdf/2606.00027)  

**Abstract**: Large language models (LLMs) are increasingly deployed across healthcare, yet existing benchmarks fail to capture model behavior under adversarial or ethically complex conditions common in clinical practice. We developed a multi-domain red teaming framework evaluating eleven contemporary LLMs across 690 clinically grounded scenarios spanning nine domains and over 150 subcategories. Scenarios incorporated adversarial transformations, and responses were assessed using a seven-dimension rubric with LLM-assisted scoring and human-in-the-loop validation. Results revealed substantial performance variance, with mean scores ranging from 0.791 to 0.984. Critically, several high-performing systems produced complete failures in individual safety-critical scenarios, demonstrating that aggregate accuracy masks clinically meaningful risk. The highest-performing systems (X-BAI, GPT-5, Claude Opus 4.1) achieved scores above 0.97 with low variance, while performance varied significantly across domains. Equity-related tasks showed 10-20% error amplification with demographic modifications, and human reviewers identified clinically relevant failures missed by automated evaluation. Our findings demonstrate that performance variance and worst-case failures provide more clinically meaningful reliability indicators than mean accuracy alone, and that hybrid evaluation approaches combining automation with clinician oversight are essential for credible safety assessment. 

---
# TrustLDM: Benchmarking Trustworthiness in Language Diffusion Models 

**Authors**: Yichuan Mo, Yukun Jiang, Yanbo Shi, Mingjie Li, Michael Backes, Yang Zhang, Yisen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.00023)  

**Abstract**: The rapid development of Language Diffusion Models (LDMs) challenges the dominant position of auto-regressive competitors in language processing. However, their flexible, any-order decoding strategies not only enable fast decoding speed but also potentially bring new trustworthiness challenges. To better understand the risks behind their pipelines, we introduce a comprehensive trustworthiness benchmark tailored to LDMs (TrustLDM), evaluating safety, privacy, and fairness across different LDM architectures with multiple categories of static post contexts. Our empirical results show that although LDMs generally exhibit strong trustworthiness with only the user prompts, their alignment behavior degrades noticeably when the malicious post contexts are attached to the masked responses. We further observe that longer contexts do not necessarily induce stronger effects, and both decoding order and generation length affect the evaluation outcomes. Finally, we propose TrustLDM-Auto, an automatic evaluation framework that leverages LDM decoding flexibility to systematically identify vulnerable configurations, revealing substantial trustworthiness weaknesses across all evaluated models and dimensions. Our work may potentially help the community build more trustworthy LDMs. Our code is available at this https URL. 

---
# lmfaoooo at SemEval-2026 Task 1: Humor Is an Audience. Preference Modeling for Constrained Humor Generation 

**Authors**: Alexey Tikhonov, Alexey Ivanov  

**Link**: [PDF](https://arxiv.org/pdf/2606.00022)  

**Abstract**: Humor generation remains difficult not only because producing fluent, novel jokes is hard, but because "funny" is audience-dependent and supervision is noisy -- preferences vary with audience, context, and culture, and annotator agreement is often low. In this paper, we describe our system for the SemEval-2026 Task-1 (MWAHAHA), which focuses on humor generation under explicit constraints. The task evaluates submitted systems via human preference judgments in 1-on-1 arena-style comparisons.
We adopt a "generate-many -> select-best" strategy. First, we generate a diverse pool of candidates per instance using multi-step prompting, model ensembling, and diversity-oriented decoding. Second, we select outputs using a preference model that approximates a "reader" by learning from human comparisons rather than absolute funniness scores. To support this approach, we release 2.5K human pairwise judgments collected through the Humor Arena prototype. We further propose an interpretable pipeline that converts labeled comparisons into a preference model. Across three preference datasets, our models consistently outperform baselines and show stronger cross-domain transfer. Finally, we apply the learned preference model to rank candidates for the MWAHAHA setting and release intermediate artifacts (candidate pools and rankings) to facilitate follow-up work.
Our system ranked 1st in the English and Chinese subtasks of MWAHAHA and 2nd in the Spanish subtask. 

---
# SENSE: Semantic Embedding Navigation with Soft-gated Evaluation for Retrieval-based Speculative Decoding 

**Authors**: Shaowen Chen, Zhicheng Liao, Hongwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.00021)  

**Abstract**: Speculative Decoding (SD) accelerates Large Language Model (LLM) inference by employing a lightweight draft model to propose candidate tokens, which are verified in parallel by the target model, without compromising generation quality. While Retrieval-based Speculative Decoding (RSD) is favored for its plug-and-play versatility, its potential is impeded by rigid lexical dependencies, rendering both retrieval and verification brittle to surface-level variations. To address this, we propose SENSE (Semantic Embedding Navigation with Soft-gated Evaluation). By anchoring retrieval on the hidden states of the target model, SENSE establishes robust semantic alignment, which empowers the Soft-gated Evaluation module to validate semantic equivalence rather than surface forms. To ensure rigorous benchmarking, we deconstruct existing methods into atomic primitives within a unified framework, facilitating granular, component-level comparison. Extensive experiments across diverse domains demonstrate that SENSE outperforms multiple baselines on the LLaMA and Qwen families, attaining up to 4.09 mean acceptance length and 3.26x speedup, while preserving generation quality. Our code will be released upon publication. 

---
# CSRP: Chain-of-Thought Reasoning for Chinese Text Correction via Reinforcement Learning with Efficiency-Aware Rewards 

**Authors**: Wei Tian, Yuhao Zhou, Man Lan  

**Link**: [PDF](https://arxiv.org/pdf/2606.00020)  

**Abstract**: Large Language Model (LLM) based Chinese Grammatical Error Correction (CGEC) systems face two critical challenges: general-purpose models lack specialized linguistic priors for subtle grammatical distinctions, and Supervised Fine-Tuning (SFT) with Maximum Likelihood Estimation fails to optimize for precision-focused metrics, leading to systematic over-correction. We propose CSRP, a three-stage framework that progressively builds correction capability through Continual Pre-training (CPT) on 5.9M balanced samples to internalize domain knowledge, Chain-of-Thought SFT with explicit error reasoning for diagnostic transparency, and Group Relative Policy Optimization with a novel Efficiency-Aware Reward that explicitly penalizes unnecessary edits. On the NACGEC benchmark, CSRP achieves state-of-the-art performance with 50.99 $F_{0.5}$ and 57.17 precision, substantially outperforming previous best results while effectively mitigating the over-correction bias inherent in MLE-trained models. Our method also advances CSCD spelling correction to 59.61 F1, surpassing GPT-4 by 5.20 points. Comprehensive ablation studies demonstrate that the RL alignment stage contributes a 8\% relative gain over the SFT baseline, and that this gain is orthogonal to the contribution of large-scale CPT, validating that explicit optimization for edit efficiency is essential for high-quality grammatical error correction. Our code is available at this https URL. 

---
# Understanding Stigmatizing Language in Clinical Documentation: A Paired Comparison of Ambient AI Drafts and Clinician Finalized Notes 

**Authors**: Yiliang Zhou, Yawen Guo, Sairam Sutari, Jasmine Dhillon, Alexandra L. Beck, Emilie Chow, Steven Tam, Danielle Perret, Deepti Pandita, Gelareh Sadigh, Archana J. McEligot, Kai Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2606.00019)  

**Abstract**: Ambient artificial intelligence (AI) documentation tools are increasingly deployed to reduce clinician documentation burden, but their implications for biased language in clinical notes remain unclear. We conducted a large-scale comparison analysis of AI drafts and corresponding clinician finalized notes to quantify stigmatizing language changes pre- and post-editing. Using a lexicon-based natural language processing (NLP) pipeline, we measured (1) the prevalence of stigmatizing language in AI drafts, (2) the prevalence and term composition in final notes, and (3) the frequency of removal or introduction of stigmatizing terms. Across 66,297 paired note sections, 21.4% of AI draft sections contained at least one stigmatizing language mention, rising to 24.0% in clinician finalized versions. Introductions occurred more often than removals, suggesting clinician editing can be a net source of stigmatizing language entering the EHR with using Ambient AI. 

---
# Examine Clinicians' Modification of Hedging Language in Ambient AI Documentation: A Comparative Study of AI Drafts and Final Notes 

**Authors**: Yiliang Zhou, Yawen Guo, Di Hu, Sairam Sutari, Emilie Chow, Steven Tam, Danielle Perret, Deepti Pandita, Kai Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2606.00018)  

**Abstract**: Ambient AI documentation systems generate clinical note drafts that clinicians frequently revise before signing off into electronic health records, yet how these edits alter hedging language remains unclear. We conducted paired analysis of clinician-edited portions of ambient AI drafts and final notes to examine (1) whether these edits change the prevalence of hedging language, (2) whether these edits exhibit a systematic shift toward greater certainty or uncertainty, and (3) whether these changes in hedging prevalence and directionality differ by ambient AI vendors and clinical specialties. Among 62,811 paired note sections, hedging terms were more often introduced into previously non-hedged text than removed from previously hedged text, and post-edit text contained more hedging mentions than pre-edit text. Directionality analyses showed a significant overall tendency toward greater uncertainty in hedging-related replacement edits. Vendor and specialty analyses revealed substantial heterogeneity in hedging prevalence, pre-to-post changes in hedging mentions, and directionality. 

---
# AEyeDE: An Attention-Based Attribution Framework for AI-Generated Text Detection 

**Authors**: Aria Nourbakhsh, Adelaide Danilov, Christoph Schommer, Salima Lamsiyah  

**Link**: [PDF](https://arxiv.org/pdf/2606.00016)  

**Abstract**: Detecting AI-generated text is becoming increasingly challenging as modern language models approach human-level fluency and can evade detectors that rely on surface statistics or likelihood-based signals. We propose \textsc{AEyeDE}, an attribution-driven approach to human-AI authorship detection that leverages model attention as a discriminative signal. Specifically, we extract attention-based attribution matrices for both human- and AI-generated text using a \emph{proxy} Transformer model with white-box access and train a lightweight Convolutional Neural Network to learn representations from these attribution maps. Across encoder-decoder translation settings, our method consistently outperforms a text-only baseline. In decoder-only settings, it performs strongly in generator-specific detection, remains competitive on standard benchmarks, and shows robustness under cross-dataset transfer and alternative-spelling perturbations. We further show that attention maps exhibit recurring local structures whose relative frequencies differ consistently between human- and AI-generated text across datasets and proxy models. These findings suggest that attention-based attribution maps provide a complementary and interpretable signal for AI-generated text detection. We will make the code publicly available to support future research. 

---
# SortingHat: Redefining Operating Systems Education with a Tailored Digital Teaching Assistant 

**Authors**: Yifan Zhang, Xinkui Zhao, Zuxin Wang, Zhengyi Zhou, Guanjie Chen, Shuiguang Deng, Jianwei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2606.00015)  

**Abstract**: Operating Systems (OS) courses are among the most challenging in computer science education due to the complexity of internal structures and the diversity of running environments. Traditional teaching methods often fail to address the diverse backgrounds, learning speeds, and practical needs of students. To tackle these challenges, we present SortingHat, a personalized digital teaching assistant tailored specifically for OS education. SortingHat integrates advanced AI technologies, including a retrieval augmented generation (RAG) framework and multi agent reinforcement learning (MARL), to deliver adaptive, scalable, and effective educational support. SortingHat features a 3D digital human interface powered by large language models (LLMs) to provide personalized, empathetic, and context aware guidance. It generates tailored exercises based on each student's learning history and academic performance, reinforcing weak areas and challenging advanced concepts. Additionally, the system incorporates a robust evaluation pipeline that ensures fair, consistent, and unbiased grading of student submissions while delivering personalized, actionable feedback for improvement. By combining personalized guidance, adaptive content creation, and automated assessment, SortingHat transforms OS education into an engaging, immersive, and scalable experience. 

---
# Toward Robust In-Context Learning: Leveraging Out-of-distribution Proxies for Target Inaccessible Demonstration Retrieval 

**Authors**: Hao Xu, Rite Bo, Fausto Giunchiglia, Yingji Li, Rui Song  

**Link**: [PDF](https://arxiv.org/pdf/2606.00014)  

**Abstract**: Although studies have demonstrated that Large Language Models (LLMs) can perform well on Out-of-Distribution (OOD) tasks, their advantage tends to diminish as the distribution shift becomes more severe. Consequently, researchers aim to retrieve distributionally similar and informative demonstrations from the available source domain to boost the inference capabilities of LLMs. However, in practical scenarios where the target domain is inaccessible, evaluating the unknown distribution is challenging, which indirectly impacts the quality of the selected demonstrations. To address this problem, we propose \textbf{DOPA}, a demonstration search framework that incorporates an OOD proxy to approximate the inaccessible target domain and guide the retrieval process. Building on proxy-based evaluation, DOPA further introduces a Mahalanobis distance-based global diversity constraint to ensure sufficient diversity among the retrieved demonstrations. Experimental results on multiple LLMs and tasks demonstrate that DOPA effectively enhances robustness in OOD settings\footnote{this https URL\_code}. 

---
# A phenomenon of AI-conformity: how algorithms change human moral decision-making 

**Authors**: Yana Venerina, Dmitry Koch, Nare Meloyan, Gerda Prutko, Valeriia Lelik, Victoria Taova, Andrey Kurpatov  

**Link**: [PDF](https://arxiv.org/pdf/2606.00013)  

**Abstract**: Social conformity is a well-documented phenomenon in which individuals shift their opinions towards those of a social majority. As artificial intelligence (AI) becomes increasingly integrated into everyday life it may also create a novel source of influence giving rise to algorithmic conformity, mechanisms of which are poorly understood. The present study examined whether AI judgements affect moral decision-making in humans (n=165) adapting the classical Asch paradigm. Participants completed a series of moral dilemmas under three different conditions: in presence of social majority, with an AI model providing brief answers and with an AI model providing both answers and explanations of its choices. In all conditions the presented responses contradicted generally accepted moral norms. The results indicated that an AI model with a reasoning component affected the opinion of participants to a degree comparable to that of a human majority. These findings suggest that even moral judgements, despite their sensitivity and personal significance, may be susceptible to algorithmic conformity. However, the mechanism underlying algorithmic conformity appears to differ from the social one. Overall, the study challenges the assumption that moral decision-making lies in "AI inadmissibility zone" - a sphere that is considered as an area in which only human-made decisions are acceptable and highlights the need for a further investigation of this phenomenon as AI-based recommendations become increasingly embedded into human decision-making. 

---
# DraDDP: A Multimodal Multi-Party Dialogue Discourse Parsing Dataset 

**Authors**: Shannan Liu, Peifeng Li, Yaxin Fan, Qiaoming Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2606.00012)  

**Abstract**: Multi-party dialogue discourse parsing aims to identify dependency structures and relation types between utterances in conversations. Previous studies are mostly limited to textual modality or two-party dialogue, failing to meet the multimodal and multi-party settings. In this paper, we construct the first publicly available English multimodal dataset DraDDP for multi-party dialogue discourse parsing, based on American TV dramas. DraDDP contains 495 dialogue segments with 6,374 utterances and 9.1 hours of parallel video content, covering rich multi-party interaction scenarios. Moreover, we establish comprehensive benchmarks by evaluating this task on DraDDP and conducting in-depth analysis on the impact of different modalities. Experimental results demonstrate the value of multimodal information in capturing dialogue structures and relation types. We will publicly release the dataset, annotation guidelines, and code to promote future research in multimodal dialogue understanding. 

---
# RuleEdit: Failure-Guided Human-AI Model Editing with Prospective Impact Preview 

**Authors**: Min Hun Lee, Justin Yu Feng Teo  

**Link**: [PDF](https://arxiv.org/pdf/2606.00011)  

**Abstract**: Despite the promise of AI to assist complex decisions, practitioners still lack ways to detect likely failures and inspect the consequences of model edits before committing them. We present RuleEdit, an interactive, rule-guided human-AI model editing system that (i) surfaces likely failures through interpretable mismatch signals from rule tables and (ii) supports user-authored rule feedback with prospective previews of projected performance changes and embedding shifts. We instantiate RuleEdit in stroke rehabilitation assessment and evaluate it with health professionals and students. Rule-guided failure detection significantly increased Human + AI performance by 14.16\% ($p<0.001$) while improving rejection of incorrect AI and reducing both over- and under- reliance as well as ChangedToWrong decisions. In addition, presenting prospective embedding previews improved participants' feedback for model adaptation, increasing post-update local performance gains from 11.50\% to 36.38\% after incorporating users' rule-based feedback ($p<0.001$). Our findings show that mismatch-based failure cues and prospective impact previews can support failure-aware human-AI model editing, while also revealing a local-global tradeoff: edits that help a specific case can degrade performance when transferred globally. We discuss implications of designing failure-aware and controllable human-AI systems. 

---
# Empathic and agentic artificial intelligence in nursing: perspectives on a human-centered framework for cancer care navigation in the United States 

**Authors**: Tyra Girdwood, Saba Kheirinejad, Parnian Kheirkhah Rahimabad, Brianna M. White, Robert L Davis, David L Schwartz, Arash Shaban-Nejad  

**Link**: [PDF](https://arxiv.org/pdf/2606.00010)  

**Abstract**: For patients experiencing cancer, nurse navigation can ease the burden of complex care by enhancing coordination of health services and patient outcomes. However, in under-resourced areas, trained nurse navigators may be limited or non-existent. In the United States, artificial intelligence (AI)-enabled digital health tools are increasingly available and may help address gaps in care coordination; however, most are not designed to specifically support nursing. This perspective piece discusses a human-centered AI framework that integrates empathic and agentic approaches grounded in the American Nurses Association's code of ethics to support nurses in the United States in cancer care navigation. The framework could augment, not replace, human empathy and agency while improving nurse workflow, patient-clinician relationships, and care coordination services in under-resourced areas. 

---
# BenHalluEval: A Multi-Task Hallucination Evaluation Framework for Large Language Models on Bengali 

**Authors**: Shefayat E Shams Adib, Ahmed Alfey Sani, Ekramul Alam Esham, Ajwad Abrar, Ishmam Tashdeed, Md Taukir Azam Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2605.31483)  

**Abstract**: Despite Bengali being the sixth most spoken language in the world, no prior work has systematically evaluated hallucination in large language models (LLMs) for Bengali. We introduce BenHalluEval, a fine-grained hallucination evaluation framework for Bengali covering four tasks: Generative Question Answering (GQA), Bangla-English Code-Mixed QA, Summarization, and Reasoning. We construct 12,000 hallucinated candidates using GPT-5.4 across twelve task-specific hallucination types, drawn from three existing Bengali datasets, and evaluate seven LLMs spanning reasoning-oriented, multilingual, and Bengali-centric categories under a dual-track protocol that independently measures false-positive rate on ground-truth instances (Track A) and hallucination detection rate on hallucinated candidates (Track B). To jointly penalise both failure modes and prevent inflated scores from uniform response bias, we propose BenHalluScore, a dual-track calibration metric that ranges from 7.72% to 55.42% across models and tasks, revealing substantial variation in hallucination calibration. Chain-of-thought prompting, applied as a mitigation strategy, shifts response distributions without consistently improving hallucination discrimination. BenHalluEval establishes the first dedicated hallucination benchmark for Bengali and highlights the inadequacy of single-track and prompting-only evaluation approaches for low-resource language settings. The dataset and code are available at this https URL. 

---
# A Lightweight Deep Learning-based Model for Ranking Influential Nodes in Complex Networks 

**Authors**: Mohammed A. Ramadhan, Abdulhakeem O. Mohammed  

**Link**: [PDF](https://arxiv.org/pdf/2507.19702)  

**Abstract**: Identifying influential nodes in complex networks is a critical task with a wide range of applications across different domains. However, existing approaches often face trade-offs between accuracy and computational efficiency. To address these challenges, we propose 1D-CGS, a lightweight and effective hybrid model that integrates the speed of one-dimensional convolutional neural networks (1D-CNN) with the topological representation power of GraphSAGE for efficient node ranking. The model uses a lightweight input representation built on two straightforward and significant topological features: node degree and average neighbor degree. These features are processed through 1D convolutions to extract local patterns, followed by GraphSAGE layers to aggregate neighborhood information. We formulate the node ranking task as a regression problem and use the Susceptible-Infected-Recovered (SIR) model to generate ground truth influence scores. 1D-CGS is initially trained on synthetic networks generated by the Barabasi-Albert model and then applied to real world networks for identifying influential nodes. Experimental evaluations on twelve real world networks demonstrate that 1D-CGS significantly outperforms traditional centrality measures and recent deep learning models in ranking accuracy, while operating in very fast runtime. The proposed model achieves an average improvement of 4.73% in Kendall's Tau correlation and 7.67% in Jaccard Similarity over the best performing deep learning baselines. It also achieves an average Monotonicity Index (MI) score 0.99 and produces near perfect rank distributions, indicating highly unique and discriminative rankings. Furthermore, all experiments confirm that 1D-CGS operates in a highly reasonable time, running significantly faster than existing deep learning methods, making it suitable for large scale applications. 

---
# A Novel Data Augmentation Strategy for Robust Deep Learning Classification of Biomedical Time-Series Data: Application to ECG and EEG Analysis 

**Authors**: Mohammed Guhdar, Ramadhan J. Mstafa, Abdulhakeem O. Mohammed  

**Link**: [PDF](https://arxiv.org/pdf/2507.12645)  

**Abstract**: The increasing need for accurate and unified analysis of diverse biological signals, such as ECG and EEG, is paramount for comprehensive patient assessment, especially in synchronous monitoring. Despite advances in multi-sensor fusion, a critical gap remains in developing unified architectures that effectively process and extract features from fundamentally different physiological signals. Another challenge is the inherent class imbalance in many biomedical datasets, often causing biased performance in traditional methods. This study addresses these issues by proposing a novel and unified deep learning framework that achieves state-of-the-art performance across different signal types. Our method integrates a ResNet-based CNN with an attention mechanism, enhanced by a novel data augmentation strategy: time-domain concatenation of multiple augmented variants of each signal to generate richer representations. Unlike prior work, we scientifically increase signal complexity to achieve future-reaching capabilities, which resulted in the best predictions compared to the state of the art. Preprocessing steps included wavelet denoising, baseline removal, and standardization. Class imbalance was effectively managed through the combined use of this advanced data augmentation and the Focal Loss function. Regularization techniques were applied during training to ensure generalization. We rigorously evaluated the proposed architecture on three benchmark datasets: UCI Seizure EEG, MIT-BIH Arrhythmia, and PTB Diagnostic ECG. It achieved accuracies of 99.96%, 99.78%, and 100%, respectively, demonstrating robustness across diverse signal types and clinical contexts. Finally, the architecture requires ~130 MB of memory and processes each sample in ~10 ms, suggesting suitability for deployment on low-end or wearable devices. 

---
