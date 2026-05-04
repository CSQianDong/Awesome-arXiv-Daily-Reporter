# Position: agentic AI orchestration should be Bayes-consistent 

**Authors**: Theodore Papamarkou, Pierre Alquier, Matthias Bauer, Wray Buntine, Andrew Davison, Gintare Karolina Dziugaite, Maurizio Filippone, Andrew Y. K. Foong, Vincent Fortuin, Dimitris Fouskakis, Jes Frellsen, Eyke Hüllermeier, Theofanis Karaletsos, Mohammad Emtiyaz Khan, Nikita Kotelevskii, Salem Lahlou, Yingzhen Li, Fang Liu, Clare Lyle, Thomas Möllenhoff, Konstantina Palla, Maxim Panov, Yusuf Sale, Kajetan Schweighofer, Artem Shelmanov, Siddharth Swaroop, Martin Trapp, Willem Waegeman, Andrew Gordon Wilson, Alexey Zaytsev  

**Link**: [PDF](https://arxiv.org/pdf/2605.00742)  

**Abstract**: LLMs excel at predictive tasks and complex reasoning tasks, but many high-value deployments rely on decisions under uncertainty, for example, which tool to call, which expert to consult, or how many resources to invest. While the usefulness and feasibility of Bayesian approaches remain unclear for LLM inference, this position paper argues that the control layer of an agentic AI system (that orchestrates LLMs and tools) is a clear case where Bayesian principles should shine. Bayesian decision theory provides a framework for agentic systems that can help to maintain beliefs over task-relevant latent quantities, to update these beliefs from observed agentic and human-AI interactions, and to choose actions. Making LLMs themselves explicitly Bayesian belief-updating engines remains computationally intensive and conceptually nontrivial as a general modeling target. In contrast, this paper argues that coherent decision-making requires Bayesian principles at the orchestration level of the agentic system, not necessarily the LLM agent parameters. This paper articulates practical properties for Bayesian control that fit modern agentic AI systems and human-AI collaboration, and provides concrete examples and design patterns to illustrate how calibrated beliefs and utility-aware policies can improve agentic AI orchestration. 

---
# To Call or Not to Call: A Framework to Assess and Optimize LLM Tool Calling 

**Authors**: Qinyuan Wu, Soumi Das, Mahsa Amani, Arijit Nag, Seungeon Lee, Krishna P. Gummadi, Abhilasha Ravichander, Muhammad Bilal Zafar  

**Link**: [PDF](https://arxiv.org/pdf/2605.00737)  

**Abstract**: Agentic AI architectures augment LLMs with external tools, unlocking strong capabilities. However, tool use is not always beneficial; some calls may be redundant or even harmful. Effective tool use, therefore, hinges on a core LLM decision: whether to call or not call a tool, when performing a task. This decision is particularly challenging for web search tools, where the benefits of external information depend on the model's internal knowledge and its ability to integrate potentially noisy tool responses. We introduce a principled framework inspired by decision-making theory to evaluate web search tool-use decisions along three key factors: necessity, utility, and affordability. Our analysis combines two complementary lenses: a normative perspective that infers true need and utility from an optimal allocation of tool calls, and a descriptive perspective that infers the model's self-perceived need and utility from their observed behaviors. We find that models' perceived need and utility of tool calls are often misaligned with their true need and utility. Building on this framework, we train lightweight estimators of need and utility based on models' hidden states. Our estimators enable simple controllers that can improve decision quality and lead to stronger task performance than the self-perceived set up across three tasks and six models. 

---
# Learn where to Click from Yourself: On-Policy Self-Distillation for GUI Grounding 

**Authors**: Yan Zhang, Daiqing Wu, Huawen Shen, Yu Zhou, Can Ma  

**Link**: [PDF](https://arxiv.org/pdf/2605.00642)  

**Abstract**: Graphical User Interface (GUI) grounding maps natural language instructions to the visual coordinates of target elements and serves as a core capability for autonomous GUI agents. Recent reinforcement learning methods (e.g., GRPO) have achieved strong performance, but they rely on expensive multiple rollouts and suffer from sparse signals on hard samples. These limitations make on-policy self-distillation (OPSD), which provides dense token-level supervision from a single rollout, a promising alternative. However, its applicability to GUI grounding remains unexplored. In this paper, we present GUI-SD, the first OPSD framework tailored for GUI grounding. First, it constructs a visually enriched privileged context for the teacher using a target bounding box and a Gaussian soft mask, providing informative guidance without leaking exact coordinates. Second, it employs entropy-guided distillation, which adaptively weights tokens based on digit significance and teacher confidence, concentrating optimization on the most impactful and reliable positions. Extensive experiments on six representative GUI grounding benchmarks show that GUI-SD consistently outperforms GRPO-based methods and naive OPSD in both accuracy and training efficiency. Code and training data are available at this https URL. 

---
# Instance-Aware Parameter Configuration in Bilevel Late Acceptance Hill Climbing for the Electric Capacitated Vehicle Routing Problem 

**Authors**: Yinghao Qin, Xinwei Wang, Mosab Bazargani, Jun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.00572)  

**Abstract**: Algorithm performance in combinatorial optimization is highly sensitive to parameter settings, while a single globally tuned configuration often fails to exploit the heterogeneity of instances. This limitation is particularly evident in the Electric Capacitated Vehicle Routing Problem, where instances differ in structure, demand patterns, and energy constraints. This paper investigates instance-aware parameter configuration for Bilevel Late Acceptance Hill Climbing, a state-of-the-art metaheuristic for the Electric Capacitated Vehicle Routing Problem. An offline tuning procedure is used to obtain instance-specific parameter labels, which are then mapped from instance features via a regression model to enable parameter prediction for unseen instances prior to execution. Experimental results on the IEEE WCCI 2020 benchmark and its extensions show that the proposed approach achieves an average objective value reduction of $0.28\%$ across eight held-out test instances relative to a globally tuned configuration. This corresponds to a significant cost reduction in multimillion-dollar transportation operations. 

---
# On the Role of Artificial Intelligence in Human-Machine Symbiosis 

**Authors**: Ching-Chun Chang, Yuchen Guo, Hanrui Wang, Timo Spinde, Isao Echizen  

**Link**: [PDF](https://arxiv.org/pdf/2605.00440)  

**Abstract**: The evolution of artificial intelligence (AI) has rendered the boundary between humanity and computational machinery increasingly ambiguous. In the presence of more interwoven relationships within human-machine symbiosis, the very notion of AI-generated information becomes difficult to define, as such information arises not from either humans or machines in isolation, but from their mutual shaping. Therefore, a more pertinent question lies not merely in whether AI has participated, but in how it has participated. In general, the role assumed by AI is often specified, either implicitly or explicitly, in the input prompt, yet becomes less apparent or altogether unobservable when the generated content alone is available. Once detached from the dialogue context, the functional role may no longer be traceable. This study considers the problem of tracing the functional role played by AI in natural language generation. A methodology is proposed to infer the latent role specified by the prompt, embed this role into the content during the probabilistic generation process and subsequently recover the nature of AI participation from the resulting text. Experimentation is conducted under a representative scenario in which AI acts either as an assistive agent that edits human-written content or as a creative agent that generates new content from a brief concept. The experimental results support the validity of the proposed methodology in terms of discrimination between roles, robustness against perturbations and preservation of linguistic quality. We envision that this study may contribute to future research on the ethics of AI with regard to whether AI has been used fairly, transparently and appropriately. 

---
# Thinking in Text and Images: Interleaved Vision--Language Reasoning Traces for Long-Horizon Robot Manipulation 

**Authors**: Jinkun Liu, Haohan Chi, Lingfeng Zhang, Yifan Xie, YuAn Wang, Long Chen, Hangjun Ye, Xiaoshuai Hao, Wenbo Ding  

**Link**: [PDF](https://arxiv.org/pdf/2605.00438)  

**Abstract**: Long-horizon robotic manipulation requires plans that are both logically coherent and geometrically grounded. Existing Vision-Language-Action policies usually hide planning in latent states or expose only one modality: text-only chain-of-thought encodes causal order but misses spatial constraints, while visual prediction provides geometric cues but often remains local and semantically underconstrained. We introduce Interleaved Vision--Language Reasoning (IVLR), a policy framework built around \trace{}, an explicit intermediate representation that alternates textual subgoals with visual keyframes over the full task horizon. At test time, a single native multimodal transformer self-generates this global semantic-geometric trace from the initial observation and instruction, caches it, and conditions a closed-loop action decoder on the trace, original instruction, and current observation. Because standard robot datasets lack such traces, we construct pseudo-supervision by temporally segmenting demonstrations and captioning each stage with a vision-language model. Across simulated benchmarks for long-horizon manipulation and visual distribution shift, \method{} reaches 95.5\% average success on LIBERO, including 92.4\% on LIBERO-Long, and 59.4\% overall success on SimplerEnv-WidowX. Ablations show that both modalities are necessary: without traces, LIBERO-Long success drops to 37.7\%; text-only and vision-only traces reach 62.0\% and 68.4\%, while the full interleaved trace reaches 92.4\%. Stress tests with execution perturbations and masked trace content show moderate degradation, suggesting that the trace can tolerate local corruption and moderate execution drift, but remains limited under stale or incorrect global plans. 

---
# AEM: Adaptive Entropy Modulation for Multi-Turn Agentic Reinforcement Learning 

**Authors**: Haotian Zhao, Yuxin Zhang, Songlin Zhou, Stephen S.-T. Yau, Wenyu Zhang, Lun Tian, Tianshu Zhu, Yifeng Huang, Yucheng Zeng, Jingnan Gu, Daxiang Dong, Jianmin Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.00425)  

**Abstract**: Reinforcement learning (RL) has significantly advanced the ability of large language model (LLM) agents to interact with environments and solve multi-turn tasks. Yet effective training remains challenging, as sparse, outcome-only rewards make it difficult to assign credit to individual steps in an agent's action trajectory. A common remedy is to introduce dense intermediate supervision, such as process reward models or auxiliary self-supervised signals, but this increases supervision and tuning complexity and often generalizes poorly across tasks and domains. This paper presents AEM, a supervision-free credit assignment method that adaptively modulates entropy dynamics during RL training to achieve a more effective exploration-exploitation trade-off. Theoretically, we elevate entropy analysis from the token level to the response level to reduce token sampling variance and show that entropy drift under natural gradients is intrinsically governed by the product of the advantage and the relative response surprisal. Specifically, we derive a practical proxy to reshape training dynamics, enabling a natural transition from exploration to exploitation. Extensive experiments across various benchmarks and models ranging from 1.5B to 32B parameters demonstrate the effectiveness of AEM, including a notable 1.4 percent gain when integrated into a state-of-the-art baseline on the highly challenging SWE-bench-Verified benchmark. 

---
# Physically Native World Models: A Hamiltonian Perspective on Generative World Modeling 

**Authors**: Sen Cui, Jingheng Ma  

**Link**: [PDF](https://arxiv.org/pdf/2605.00412)  

**Abstract**: World models have recently re-emerged as a central paradigm for embodied intelligence, robotics, autonomous driving, and model-based reinforcement learning. However, current world model research is often dominated by three partially separated routes: 2D video-generative models that emphasize visual future synthesis, 3D scene-centric models that emphasize spatial reconstruction, and JEPA-like latent models that emphasize abstract predictive representations. While each route has made important progress, they still struggle to provide physically reliable, action-controllable, and long-horizon stable predictions for embodied decision making. In this paper, we argue that the bottleneck of world models is no longer only whether they can generate realistic futures, but whether those futures are physically meaningful and useful for action. We propose \emph{Hamiltonian World Models} as a physically grounded perspective on world modeling. The key idea is to encode observations into a structured latent phase space, evolve the latent state through Hamiltonian-inspired dynamics with control, dissipation, and residual terms, decode the predicted trajectory into future observations, and use the resulting rollouts for planning. We discuss how Hamiltonian structure may improve interpretability, data efficiency, and long-horizon stability, while also noting practical challenges in real-world robotic scenes involving friction, contact, non-conservative forces, and deformable objects. 

---
# AgentFloor: How Far Up the tool use Ladder Can Small Open-Weight Models Go? 

**Authors**: Ranit Karmakar, Jayita Chatterjee  

**Link**: [PDF](https://arxiv.org/pdf/2605.00334)  

**Abstract**: Production agentic systems make many model calls per user request, and most of those calls are short, structured, and routine. This raises a practical routing question that existing evaluations do not directly answer: which parts of an agent workflow truly require large frontier intelligence, and which can be handled by smaller models? We introduce AgentFloor, a deterministic 30-task benchmark organized as a six-tier capability ladder, spanning instruction following, tool use, multi-step coordination, and long-horizon planning under persistent constraints. We evaluate 16 open-weight models, from 0.27B to 32B parameters, alongside GPT-5 across 16,542 scored runs. Our results reveal a clear boundary of model necessity. Small and mid-sized open-weight models are already sufficient for much of the short-horizon, structured tool use work that dominates real agent pipelines, and in aggregate, the strongest open-weight model matches GPT-5 on our benchmark while being substantially cheaper and faster to run. The gap appears most clearly on long-horizon planning tasks that require sustained coordination and reliable constraint tracking over many steps, where frontier models still hold an advantage, though neither side reaches strong reliability. We also find that this boundary is not explained by scale alone: some failures respond to targeted interventions, but the effects are model-specific rather than universal. These findings suggest a practical design principle for agentic systems: use smaller open-weight models for the broad base of routine actions, and reserve large frontier models for the narrower class of tasks that truly demand deeper planning and control. We release the benchmark, harness, sweep configurations, and full run corpus. 

---
# Token Arena: A Continuous Benchmark Unifying Energy and Cognition in AI Inference 

**Authors**: Yuxuan Gao, Megan Wang, Yi Ling Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.00300)  

**Abstract**: Public inference benchmarks compare AI systems at the model and provider level, but the unit at which deployment decisions are actually made is the endpoint: the (provider, model, stock-keeping-unit) tuple at which a specific quantization, decoding strategy, region, and serving stack is exposed. We introduce TokenArena, a continuous benchmark that measures inference at endpoint granularity along five core axes (output speed, time to first token, workload-blended price, effective context, and quality on the live endpoint) and synthesizes them, together with a modeled energy estimate, into three headline composites: joules per correct answer, dollars per correct answer, and endpoint fidelity (output-distribution similarity to a first-party reference). The framework's novelty is empirical and methodological. Across 78 endpoints serving 12 model families, the same model on different endpoints differs in mean accuracy by up to 12.5 points on math and code, in fingerprint similarity to first party by up to 12 points, in tail latency by an order of magnitude, and in modeled joules per correct answer by a factor of 6.2. We further show that workload-aware blended pricing reorders the leaderboard substantially: 7 of 10 top-ranked endpoints under the chat preset (3:1 input:output) fall out of the top 10 under the retrieval-augmented preset (20:1), and the reasoning preset (1:5) elevates frontier closed models that the chat preset penalizes on price. We release the framework, schema, probe and eval harness, and a v1.0 leaderboard snapshot under CC BY 4.0. TokenArena is a methodology, not a single ranking; we publish full provenance and limitations and welcome external replication. 

---
# Agentic AI for Trip Planning Optimization Application 

**Authors**: Tiejin Chen, Ahmadreza Moradipari, Kyungtae Han, Hua Wei, Nejib Ammar  

**Link**: [PDF](https://arxiv.org/pdf/2605.00276)  

**Abstract**: Trip planning for intelligent vehicles increasingly requires selecting optimal routes rather than merely producing feasible itineraries, as interacting factors such as travel time, energy consumption, and traffic conditions directly affect plan quality. Yet existing systems are largely designed for feasibility-oriented planning, and current benchmarks provide only reference answers without ground truth, preventing objective evaluation of optimization performance. In our paper, we address these limitations with an agentic AI framework that enables dynamic refinement through an orchestration agent coordinating specialized agents for traffic, charging, and points of interest, and with the Trip-planning Optimization Problems Dataset, which supplies definitive optimal solutions and category-level task structure for fine-grained analysis. Experiments show that our system achieves 77.4\% accuracy on the TOP Benchmark, significantly outperforming single-agent and workflow-based multi-agent baselines, demonstrating the importance of orchestrated agentic reasoning for robust trip planning optimization. 

---
# Causal Foundations of Collective Agency 

**Authors**: Frederik Hytting Jørgensen, Sebastian Weichwald, Lewis Hammond  

**Link**: [PDF](https://arxiv.org/pdf/2605.00248)  

**Abstract**: A key challenge for the safety of advanced AI systems is the possibility that multiple simpler agents might inadvertently form a collective agent with capabilities and goals distinct from those of any individual. More generally, determining when a group of agents can be viewed as a unified collective agent is a foundational question in the study of interactions and incentives in both biological and artificial systems. We adopt a behavioral perspective in answering this question, ascribing collective agency to a group when viewing the group's joint actions as rational and goal-directed successfully predicts its behavior. We formalize this perspective on collective agency using causal games -- which are causal models of strategic, multi-agent interactions -- and causal abstraction -- which formalizes when a simple, high-level model faithfully captures a more complex, low-level model. We use this framework to solve a puzzle regarding multi-agent incentives in actor-critic models and to make quantitative assessments of the degree of collective agency exhibited by different voting mechanisms. Our framework aims to provide a foundation for theoretical and empirical work to understand, predict, and control emergent collective agents in multi-agent AI systems. 

---
# ARMOR 2025: A Military-Aligned Benchmark for Evaluating Large Language Model Safety Beyond Civilian Contexts 

**Authors**: Sydney Johns, Heng Jin, Chaoyu Zhang, Y. Thomas Hou, Wenjing Lou  

**Link**: [PDF](https://arxiv.org/pdf/2605.00245)  

**Abstract**: Large language models (LLMs) are now being explored for defense applications that require reliable and legally compliant decision support. They also hold significant potential to enhance decision making, coordination, and operational efficiency in military contexts. These uses demand evaluation methods that reflect the doctrinal standards that guide real military operations. Existing safety benchmarks focus on general social risks and do not test whether models follow the legal and ethical rules that govern real military operations. To address this gap, we introduce ARMOR 2025, a military aligned safety benchmark grounded in three core military doctrines the Law of War, the Rules of Engagement, and the Joint Ethics Regulation. We extract doctrinal text from these sources and generate multiple choice questions that preserve the intended meaning of each rule. The benchmark is organized through a taxonomy informed by the Observe Orient Decide Act (OODA) decision making framework. This structure enables systematic testing of accuracy and refusal across military relevant decision types. This benchmark features a structured 12-category taxonomy, 519 doctrinally grounded prompts, and rigorous evaluation procedures applied to 21 commercial LLMs. Evaluation results reveal critical gaps in safety alignment for military applications. 

---
# TUR-DPO: Topology- and Uncertainty-Aware Direct Preference Optimization 

**Authors**: Abdulhady Abas Abdullah, Fatemeh Daneshfar, Seyedali Mirjalili, Mourad Oussalah  

**Link**: [PDF](https://arxiv.org/pdf/2605.00224)  

**Abstract**: Aligning large language models (LLMs) with human preferences is commonly done via reinforcement learning from human feedback (RLHF) with Proximal Policy Optimization (PPO) or, more simply, via Direct Preference Optimization (DPO). While DPO is stable and RL-free, it treats preferences as flat winner vs. loser signals and is sensitive to noisy or brittle preferences arising from fragile chains of thought. We propose TUR-DPO, a topology- and uncertainty-aware variant of DPO that rewards how answers are derived, not only what they say, by eliciting lightweight reasoning topologies and combining semantic faithfulness, utility, and topology quality into a calibrated uncertainty signal. A small learnable reward is factorized over these signals and incorporated into an uncertainty-weighted DPO objective that remains RL-free and relies only on a fixed or moving reference policy. Empirically, across open 7-8B models and benchmarks spanning mathematical reasoning, factual question answering, summarization, and helpful/harmless dialogue, TUR-DPO improves judge win-rates, faithfulness, and calibration relative to DPO while preserving training simplicity and avoiding online rollouts. We further observe consistent gains in multimodal and long-context settings, and show that TUR-DPO matches or exceeds PPO on reasoning-centric tasks while maintaining operational simplicity. 

---
# Are Tools All We Need? Unveiling the Tool-Use Tax in LLM Agents 

**Authors**: Kaituo Zhang, Zhen Xiong, Mingyu Zhong, Zhimeng Jiang, Zhouyuan Yuan, Zhecheng Li, Ying Lin  

**Link**: [PDF](https://arxiv.org/pdf/2605.00136)  

**Abstract**: Tool-augmented reasoning has become a popular direction for LLM-based agents, and it is widely assumed to improve reasoning and reliability. However, we demonstrate that this consensus does not always hold: in the presence of semantic distractors, tool-augmented reasoning does not necessarily outperform native CoT. To explain this performance gap, we propose a Factorized Intervention Framework that isolates the cost of prompt formatting, the overhead of the tool-calling protocol, and the actual gain from executing tools. Our analysis reveals a critical tradeoff: under semantic noise, the gains from tools often fail to offset the "tool-use tax", which is the performance degradation introduced by the tool-calling protocol itself. To address this, we introduce G-STEP, a lightweight inference-time gate to mitigate protocol-induced errors. While this yields partial recovery, our findings suggest that more substantial improvements still require strengthening the model's intrinsic reasoning and tool-interaction capabilities. 

---
# Minimal, Local, Causal Explanations for Jailbreak Success in Large Language Models 

**Authors**: Shubham Kumar, Narendra Ahuja  

**Link**: [PDF](https://arxiv.org/pdf/2605.00123)  

**Abstract**: Safety trained large language models (LLMs) can often be induced to answer harmful requests through jailbreak prompts. Because we lack a robust understanding of why LLMs are susceptible to jailbreaks, future frontier models operating more autonomously in higher-stakes settings may similarly be vulnerable to such attacks. Prior work has studied jailbreak success by examining the model's intermediate representations, identifying directions in this space that causally encode concepts like harmfulness and refusal. Then, they globally explain all jailbreak attacks as attempting to reduce or strengthen these concepts (e.g., reduce harmfulness). However, different jailbreak strategies may succeed by strengthening or suppressing different intermediate concepts, and the same jailbreak strategy may not work for different harmful request categories (e.g., violence vs. cyberattack); thus, we seek to give a local explanation -- i.e., why did this specific jailbreak succeed? To address this gap, we introduce LOCA, a method that gives Local, CAusal explanations of jailbreak success by identifying a minimal set of interpretable, intermediate representation changes that causally induce model refusal on an otherwise successful jailbreak request. We evaluate LOCA on harmful original-jailbreak pairs from a large jailbreak benchmark across Gemma and Llama chat models, comparing against prior methods adapted to this setting. LOCA can successfully induce refusal by making, on average, six interpretable changes; prior work routinely fails to achieve refusal even after 20 changes. LOCA is a step toward mechanistic, local explanations of jailbreak success in LLMs. Code to be released. 

---
# AgentReputation: A Decentralized Agentic AI Reputation Framework 

**Authors**: Mohd Sameen Chishti, Damilare Peter Oyinloye, Jingyue Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.00073)  

**Abstract**: Decentralized, agentic AI marketplaces are rapidly emerging to support software engineering tasks such as debugging, patch generation, and security auditing, often operating without centralized oversight. However, existing reputation mechanisms fail in this setting for three fundamental reasons: agents can strategically optimize against evaluation procedures; demonstrated competence does not reliably transfer across heterogeneous task contexts; and verification rigor varies widely, from lightweight automated checks to costly expert review. Current approaches to reputation drawing on federated learning, blockchain-based AI platforms, and large language model safety research are unable to address these challenges in combination. We therefore propose \textbf{AgentReputation}, a decentralized, three-layer reputation framework for agentic AI systems. The framework separates task execution, reputation services, and tamper-proof persistence to both leverage their respective strengths and enable independent evolution. The framework introduces explicit verification regimes linked to agent reputation metadata, as well as context-conditioned reputation cards that prevent reputation conflation across domains and task types. In addition, AgentReputation provides a decision-facing policy engine that supports resource allocation, access control, and adaptive verification escalation based on risk and uncertainty. Building on this framework, we outline several future research directions, including the development of verification ontologies, methods for quantifying verification strength, privacy-preserving evidence mechanisms, cold-start reputation bootstrapping, and defenses against adversarial manipulation. 

---
# TADI: Tool-Augmented Drilling Intelligence via Agentic LLM Orchestration over Heterogeneous Wellsite Data 

**Authors**: Rong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2605.00060)  

**Abstract**: We present TADI (Tool-Augmented Drilling Intelligence), an agentic AI system that transforms drilling operational data into evidence-based analytical intelligence. Applied to the Equinor Volve Field dataset, TADI integrates 1,759 daily drilling reports, selected WITSML real-time objects, 15,634 production records, formation tops, and perforations into a dual-store architecture: DuckDB for structured queries over 12 tables with 65,447 rows, and ChromaDB for semantic search over 36,709 embedded documents. Twelve domain-specialized tools, orchestrated by a large language model via iterative function calling, support multi-step evidence gathering that cross-references structured drilling measurements with daily report narratives. The system parses all 1,759 DDR XML files with zero errors, handles three incompatible well naming conventions, and is backed by 95 automated tests plus a 130-question stress-question taxonomy spanning six operational categories. We formalize the agent's behavior as a sequential tool-selection problem and propose the Evidence Grounding Score (EGS) as a simple grounding-compliance proxy based on measurements, attributed DDR quotations, and required answer sections. The complete 6,084-line, framework-free implementation is reproducible given the public Volve download and an API key, and the case studies and qualitative ablation analysis suggest that domain-specialized tool design, rather than model scale alone, is the primary driver of analytical quality in technical operations. 

---
# Persistent Visual Memory: Sustaining Perception for Deep Generation in LVLMs 

**Authors**: Siyuan Huang, Xiaoye Qu, Yafu Li, Tong Zhu, Zefeng He, Muxin Fu, Daizong Liu, Wei-Long Zheng, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2605.00814)  

**Abstract**: While autoregressive Large Vision-Language Models (LVLMs) demonstrate remarkable proficiency in multimodal tasks, they face a "Visual Signal Dilution" phenomenon, where the accumulation of textual history expands the attention partition function, causing visual attention to decay inversely with generated sequence length. To counteract this, we propose Persistent Visual Memory (PVM), a lightweight learnable module designed to ensure sustained, on-demand visual perception. Integrated as a parallel branch alongside the Feed-Forward Network (FFN) in LVLMs, PVM establishes a distance-agnostic retrieval pathway that directly provides visual embeddings for precise visual perception, thereby structurally mitigating the signal suppression inherent to deep generation. Extensive experiments on Qwen3-VL models demonstrate that PVM brings notable improvements with negligible parameter overhead, delivering consistent average accuracy gains across both 4B and 8B scales, particularly in complex reasoning tasks that demand persistent visual perception. Furthermore, in-depth analysis reveals that PVM can resist length-induced signal decay and accelerate internal prediction convergence. 

---
# Can Coding Agents Reproduce Findings in Computational Materials Science? 

**Authors**: Ziyang Huang, Yi Cao, Ali K. Shargh, Jing Luo, Ruidong Mei, Mohd Zaki, Zhan Liu, Wyatt Bunstine, William Jurayj, Somdatta Goswami, Tyrel McQueen, Michael Shields, Jaafar El-Awady, Paulette Clancy, Benjamin Van Durme, Nicholas Andrews, William Walden, Daniel Khashabi  

**Link**: [PDF](https://arxiv.org/pdf/2605.00803)  

**Abstract**: Large language models are increasingly deployed as autonomous coding agents and have achieved remarkably strong performance on software engineering benchmarks. However, it is unclear whether such success transfers to computational scientific workflows, where tasks require not only strong coding ability, but also the ability to navigate complex, domain-specific procedures and to interpret results in the context of scientific claims. To address this question, we present AutoMat, a benchmark for evaluating LLM-based agents' ability to reproduce claims from computational materials science. AutoMat poses three interrelated challenges: recovering underspecified computational procedures, navigating specialized toolchains, and determining whether the resulting evidence supports a claim. By working closely with subject matter experts, we curate a set of claims from real materials science papers to test whether coding agents can recover and execute the end-to-end workflow needed to support (or undermine) such claims. We then evaluate multiple representative coding agent settings across several foundation models. Our results show that current LLM-based agents obtain low overall success rates on AutoMat, with the best-performing setting achieving a success rate of only 54.1%. Error analysis further reveals that agents perform worst when workflows must be reconstructed from paper text alone and that they fail primarily due to incomplete procedures, methodological deviations, and execution fragility. Taken together, these findings position AutoMat as both a benchmark for computational scientific reproducibility and a tool for diagnosing the current limitations of agentic systems in AI-for-science settings. 

---
# When RAG Chatbots Expose Their Backend: An Anonymized Case Study of Privacy and Security Risks in Patient-Facing Medical AI 

**Authors**: Alfredo Madrid-García, Miguel Rujas  

**Link**: [PDF](https://arxiv.org/pdf/2605.00796)  

**Abstract**: Background: Patient-facing medical chatbots based on retrieval-augmented generation (RAG) are increasingly promoted to deliver accessible, grounded health information. AI-assisted development lowers the barrier to building them, but they still demand rigorous security, privacy, and governance controls. Objective: To report an anonymized, non-destructive security assessment of a publicly accessible patient-facing medical RAG chatbot and identify governance lessons for safe deployment of generative AI in health. Methods: We used a two-stage strategy. First, Claude Opus 4.6 supported exploratory prompt-based testing and structured vulnerability hypotheses. Second, candidate findings were manually verified using Chrome Developer Tools, inspecting browser-visible network traffic, payloads, API schemas, configuration objects, and stored interaction data. Results: The LLM-assisted phase identified a critical vulnerability: sensitive system and RAG configuration appeared exposed through client-server communication rather than restricted server-side. Manual verification confirmed that ordinary browser inspection allowed collection of the system prompt, model and embedding configuration, retrieval parameters, backend endpoints, API schema, document and chunk metadata, knowledge-base content, and the 1,000 most recent patient-chatbot conversations. The deployment also contradicted its privacy assurances: full conversation records, including health-related queries, were retrievable without authentication. Conclusions: Serious privacy and security failures in patient-facing RAG chatbots can be identified with standard browser tools, without specialist skills or authentication; independent review should be a prerequisite for deployment. Commercial LLMs accelerated this assessment, including under a false developer persona; assistance available to auditors is equally available to adversaries. 

---
# Unsupervised Denoising of Real Clinical Low Dose Liver CT with Perceptual Attention Networks 

**Authors**: Jingxi Pu, Tonghua Liu, Zhilin Guan, Siqiao Li, Yang Ming, Zheng Cong, Wei Zhang, Fangwei Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.00793)  

**Abstract**: With the development of deep learning, medical image processing has been widely used to assist clinical research. This paper focuses on the denoising problem of low-dose computed tomography using deep learning. Although low-dose computed tomography reduces radiation exposure to patients, it also introduces more noise, which may interfere with visual interpretation by physicians and affect diagnostic results. To address this problem, inspired by Cycle-GAN for unsupervised learning, this paper proposes an end-to-end unsupervised low-dose computed tomography denoising framework. The proposed framework combines a U-Net structure for multi-scale feature extraction, an attention mechanism for feature fusion, and a residual network for feature transformation. It also introduces perceptual loss to improve the network for the characteristics of medical images. In addition, we construct a real low-dose computed tomography dataset and design a large number of comparative experiments to validate the proposed method, using both image-based evaluation metrics and medical evaluation criteria. Compared with classical methods, the main advantage of this paper is that it addresses the limitation that real clinical data cannot be directly used for supervised learning, while still achieving excellent performance. The experimental results are also professionally evaluated by imaging physicians and meet clinical needs. 

---
# Make Your LVLM KV Cache More Lightweight 

**Authors**: Xihao Chen, Yangyang Guo, Roger Zimmermann  

**Link**: [PDF](https://arxiv.org/pdf/2605.00789)  

**Abstract**: Key-Value (KV) cache has become a de facto component of modern Large Vision-Language Models (LVLMs) for inference. While it enhances decoding efficiency in Large Language Models (LLMs), its direct adoption in LVLMs introduces substantial GPU memory overhead due to the large number of vision tokens processed during the prefill stage. To tackle this problem, we propose LightKV, a novel approach that reduces KV cache size by exploiting the redundancy among vision-token embeddings. Guided by text prompts, LightKV employs cross-modality message passing to aggregate informative messages across vision tokens and progressively compress them during prefill. This prompt-aware guidance distinguishes our method from prior vision-only compression strategies. We evaluate LightKV on eight open-source LVLMs across eight public benchmark datasets, e.g., MME and SeedBench. Experimental results demonstrate that with only 55% of the original vision tokens, LightKV (a) halves the vision-token KV cache size, (b) reduces computation by up to 40%, and (c) preserves general-purpose performance while significantly outperforming existing baselines. 

---
# GeoContra: From Fluent GIS Code to Verifiable Spatial Analysis with Geography-Grounded Repair 

**Authors**: Yinhao Xiao, Rongbo Xiao, Yihan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.00782)  

**Abstract**: Reliable spatial analysis in GIScience requires preserving coordinate semantics, topology, units, and geographic plausibility. Current LLM-based GIS systems generate fluent scripts but rarely enforce these geographic rules at scale. We present GeoContra, a verification and repair framework for LLM-driven Python GIS workflows. It represents each task as an executable geospatial contract-including natural-language questions, schemas, CRS metadata, expected outputs, spatial predicates, topology, metrics, required operations, and forbidden shortcuts. Generated programs undergo static rule inspection, runtime validation, and semantic verification, with violations fed back into a bounded repair loop. Evaluated on 7,079 real geospatial tasks across 15 Boston-area zones, 9 task families, and 11 open-source models (600 runs each), GeoContra improves spatial correctness on closed models from 47.6% to 77.5% for DeepSeek-V4 and from 57.7% to 81.5% for Kimi-K2.5. Across 11 open models, average correctness rises by 26.6%. GeoContra turns fluent code production into verifiable spatial analysis, catching negative travel times, CRS/field-schema violations, missing predicates, and brittle output casts that otherwise yield executable but geographically invalid results. 

---
# Directed Social Regard: Surfacing Targeted Advocacy, Opposition, Aid, Harms, and Victimization in Online Media 

**Authors**: Scott Friedman, Ruta Wheelock, Sonja Schmer-Galunder, Drisana Iverson, Jake Vasilakes, Joan Zheng, Jeffrey Rye, Vasanth Sarathy, Christopher Miller  

**Link**: [PDF](https://arxiv.org/pdf/2605.00776)  

**Abstract**: The language in online platforms, influence operations, and political rhetoric frequently directs a mix of pro-social sentiment (e.g., advocacy, helpfulness, compassion) and anti-social sentiment (e.g., threats, opposition, blame) at different topics, all in the same message. While many natural language processing (NLP) tools classify or score a text's overall sentiment as positive, neutral, or negative, these tools cannot report that positive and negative sentiments coexist, and they cannot report the target of those sentiments. This paper presents the Directed Social Regard (DSR) approach to multi-dimensional, multi-valence sentiment analysis, comprised of a pair of transformer-based models that (1) detects span-level targets of sentiment in a message and then (2) scores all spans within the message context along three (-1, 1) axes of regard that are motivated by social science theories of moral disengagement and moral framing. We present a data collection and annotation strategy for DSR dataset construction, a transformer-based architecture for span-level scoring, and a validation study with promising results. We apply the validated DSR model on six third-party datasets of online media and report meaningful correlations between DSR outputs and the labels and topics in these pre-existing social science datasets. 

---
# Meritocratic Fairness in Budgeted Combinatorial Multi-armed Bandits via Shapley Values 

**Authors**: Shradha Sharma, Swapnil Dhamal, Shweta Jain  

**Link**: [PDF](https://arxiv.org/pdf/2605.00762)  

**Abstract**: We propose a new framework for meritocratic fairness in budgeted combinatorial multi-armed bandits with full-bandit feedback (BCMAB-FBF). Unlike semi-bandit feedback, the contribution of individual arms is not received in full-bandit feedback, making the setting significantly more challenging. To compute arm contributions in BCMAB-FBF, we first extend the Shapley value, a classical solution concept from cooperative game theory, to the $K$-Shapley value, which captures the marginal contribution of an agent restricted to a set of size at most $K$. We show that $K$-Shapley value is a unique solution concept that satisfies Symmetry, Linearity, Null player, and efficiency properties. We next propose K-SVFair-FBF, a fairness-aware bandit algorithm that adaptively estimates $K$-Shapley value with unknown valuation function. Unlike standard bandit literature on full bandit feedback, K-SVFair-FBF not only learns the valuation function under full feedback setting but also mitigates the noise arising from Monte Carlo approximations. Theoretically, we prove that K-SVFair-FBF achieves $O(T^{3/4})$ regret bound on fairness regret. Through experiments on federated learning and social influence maximization datasets, we demonstrate that our approach achieves fairness and performs more effectively than existing baselines. 

---
# EASE: Federated Multimodal Unlearning via Entanglement-Aware Anchor Closure 

**Authors**: Zihao Ding, Beining Wu, Jun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.00733)  

**Abstract**: Federated Multimodal Learning (FML) trains multimodal models across decentralized clients while keeping their image-text pairs private. However, joint embedding training entangles forgotten knowledge across both modalities and client gradient subspaces, hindering federated unlearning. Previous federated unlearning approaches neither sever the cross-modal reconstruction channel mediated by bilinear coupling nor separate forget-exclusive update directions from those shared with retained clients. We identify an Anchor Principle for federated multimodal contrastive unlearning: forgotten alignments persist through three residual anchors arising from bilinear cross-modal coupling, principal-angle subspace entanglement, and continued federated updates. At the modality level, we show that bilateral displacement of both visual and language branches closes the cross-modal reconstruction channel. Correspondingly, our method addresses subspace entanglement through Cosine--Sine decomposition of client-update subspaces, isolating forget-exclusive directions from retain support. Moreover, we propose a direction-selective Forget Lock that bounds residual drift across rounds. Combining these strategies, we present EASE, an Entanglement-Aware Subspace Excision framework that closes all three anchor channels under a unified design. EASE demonstrates consistent superiority across multiple datasets and unlearning scenarios, for instance, matching the retrain reference to within 0.2 and 4.2 R@1 points on the forget and retain sides under client unlearning on Flickr30K with CLIP-B/32. 

---
# Empowering Heterogeneous Graph Foundation Models via Decoupled Relation Alignment 

**Authors**: Ziyu Zheng, Yaming Yang, Zhe Wang, Ziyu Guan, Wei Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.00731)  

**Abstract**: While Graph Foundation Models (GFMs) have achieved remarkable success in homogeneous graphs, extending them to multi-domain heterogeneous graphs (MDHGs) remains a formidable challenge due to cross-type feature shifts and intra-domain relation gaps. Existing global feature alignment methods (PCA or SVD) enforce a shared feature space blindly, which distorts type-specific semantics and disrupts original topologies, inevitably leading to "Type Collapse" and "Relation Confusion". To address these fundamental limitations, we propose Decoupled relation Subspace Alignment (DRSA), a novel, plug-and-play relation-driven alignment framework. DRSA fundamentally shifts the paradigm by decoupling feature semantics from relation structures. Specifically, it introduces a dual-relation subspace projection mechanism to coordinate cross-type interactions within a shared low-rank relation subspace explicitly. Furthermore, a feature-structure decoupled representation is designed to decompose aligned features into a semantic projection component and a structural residual term, adaptively absorbing intra-domain variations. Optimized via a stable alternating minimization strategy based on Block Coordinate Descent, DRSA constructs a well-calibrated, structure-aware latent space. Extensive experiments on multiple real-world benchmark datasets demonstrate that DRSA can be seamlessly integrated as a universal preprocessing module, significantly and consistently enhancing the cross-domain and few-shot knowledge transfer capabilities of state-of-the-art GFMs. The code is available at: this https URL. 

---
# Towards Improving Speaker Distance Estimation through Generative Impulse Response Augmentation 

**Authors**: Anton Ratnarajah, Mehmet Ergezer, Arun Nair, Mrudula Athi  

**Link**: [PDF](https://arxiv.org/pdf/2605.00721)  

**Abstract**: The Room Acoustics and Speaker Distance Estimation (SDE) Challenge at ICASSP 2025 explores the effectiveness of augmented room impulse response (RIR) data for improving SDE model performance. This challenge at GenDARA involves generating RIRs to supplement sparse datasets and fine-tuning SDE models with the augmented data. We employ the open-source fast diffuse room impulse response generator (FastRIR) conditioned only on speaker and listener locations. We design a quality filter to ensure generated RIR alignment with challenge RIRs, and hyperparameter optimization is employed for model fine-tuning. Our approach reduces the mean absolute error (MAE) of the five positions from 1.66m to 0.6m for GWA rooms and from 2.18m to 0.69m for Treble rooms, with results demonstrating that the augmentation approach significantly improves estimation accuracy, particularly at medium to long distances. 

---
# Augmented Lagrangian Multiplier Network for State-wise Safety in Reinforcement Learning 

**Authors**: Jiaming Zhang, Yujie Yang, Yao Lyu, Shengbo Eben Li, Liping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.00667)  

**Abstract**: Safety is a primary challenge in real-world reinforcement learning (RL). Formulating safety requirements as state-wise constraints has become a prominent paradigm. Handling state-wise constraints with the Lagrangian method requires a distinct multiplier for every state, necessitating neural networks to approximate them as a multiplier network. However, applying standard dual gradient ascent to multiplier networks induces severe training oscillations. This is because the inherent instability of dual ascent is exacerbated by network generalization -- local overshoots and delayed updates propagate to adjacent states, further amplifying policy fluctuations. Existing stabilization techniques are designed for scalar multipliers, which are inadequate for state-dependent multiplier networks. To address this challenge, we propose an augmented Lagrangian multiplier network (ALaM) framework for stable learning of state-wise multipliers. ALaM consists of two key components. First, a quadratic penalty is introduced into the augmented Lagrangian to compensate for delayed multiplier updates and establish the local convexity near the optimum, thereby mitigating policy oscillations. Second, the multiplier network is trained via supervised regression toward a dual target, which stabilizes training and promotes convergence. Theoretically, we show that ALaM guarantees multiplier convergence and thus recovers the optimal policy of the constrained problem. Building on this framework, we integrate soft actor-critic (SAC) with ALaM to develop the SAC-ALaM algorithm. Experiments demonstrate that SAC-ALaM outperforms state-of-the-art safe RL baselines in both safety and return, while also stabilizing training dynamics and learning well-calibrated multipliers for risk identification. 

---
# InpaintSLat: Inpainting Structured 3D Latents via Initial Noise Optimization 

**Authors**: Jaeyoung Chung, Suyoung Lee, Kyoung Mu Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.00664)  

**Abstract**: We present a training-free approach for controllable 3D inpainting based on initial noise optimization. In the structured 3D latent diffusion framework, we observe that the underlying geometric structure is established during the early stages of the diffusion process and exhibits high sensitivity to the initial noise. Such characteristics compromise stability in tasks like inpainting and editing, where the model must ensure strict alignment with the existing context while synthesizing a new structure. In this paper, we introduce a strategy to optimize the initial noise within the structured 3D latent diffusion framework, ensuring high-fidelity 3D inpainting. Specifically, we update the initial noise by leveraging a backpropagation approximation grounded in the rectified flow model, with the spectral parameterization specially designed for robust and efficient structured 3D latent optimization. Experiments demonstrate consistent improvements in contextual consistency and prompt alignment over representative training-free inpainting baselines, establishing initial noise control as an independent dimension for 3D inpainting, orthogonal to conventional sampling trajectory manipulation. 

---
# Reinforcement Learning with Markov Risk Measures and Multipattern Risk Approximation 

**Authors**: Andrzej Ruszczynski, Tiangang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.00654)  

**Abstract**: For a risk-averse finite-horizon Markov Decision Problem, we introduce a special class of Markov coherent risk measures, called mini-batch measures. We also define the class of multipattern risk-averse problems that generalizes the class of linear systems. We use both concepts in a feature-based $Q$-learning method with multipattern $Q$-factor approximation and we prove a high-probability regret bound of $\mathcal{O}\big(H^2 N^H \sqrt{ K}\big)$, where $H$ is the horizon, $N$ is the mini-batch size, and $K$ is the number of episodes. We also propose an economical version of the $Q$-learning method that streamlines the policy evaluation (backward) step. The theoretical results are illustrated on a stochastic assignment problem and a short-horizon multi-armed bandit problem. 

---
# AdaMeZO: Adam-style Zeroth-Order Optimizer for LLM Fine-tuning Without Maintaining the Moments 

**Authors**: Zhijie Cai, Haolong Chen, Guangxu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2605.00650)  

**Abstract**: Fine-tuning LLMs is necessary for various dedicated downstream tasks, but classic backpropagation-based fine-tuning methods require substantial GPU memory. To this end, a recent work, MeZO, which relies solely on forward passes to fine-tune LLMs, significantly reduces GPU requirements at the cost of slower convergence due to its indifference to loss landscapes. Standard solutions, such as Adam, explore loss landscapes by estimating the first- and second-order moments and storing them in memory to guide the model's movement through dimensions with lower curvature and vice versa. However, directly applying Adam negates MeZO's advantage as it will triple the memory requirement. In light of this, we propose AdaMeZO, a zeroth-order optimizer that leverages Adam-style first- and second-moment estimates without maintaining them in memory. We present a theoretical analysis of AdaMeZO, corroborated by extensive experiments demonstrating AdaMeZO's performance, showing that AdaMeZO can outperform MeZO while requiring up to $70\%$ fewer forward passes. Trajectory visualizations affirm AdaMeZO's ability to adapt to diverse loss landscapes. 

---
# Learning Multimodal Energy-Based Model with Multimodal Variational Auto-Encoder via MCMC Revision 

**Authors**: Jiali Cui, Zhiqiang Lao, Heather Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.00644)  

**Abstract**: Energy-based models (EBMs) are a flexible class of deep generative models and are well-suited to capture complex dependencies in multimodal data. However, learning multimodal EBM by maximum likelihood requires Markov Chain Monte Carlo (MCMC) sampling in the joint data space, where noise-initialized Langevin dynamics often mixes poorly and fails to discover coherent inter-modal relationships. Multimodal VAEs have made progress in capturing such inter-modal dependencies by introducing a shared latent generator and a joint inference model. However, both the shared latent generator and joint inference model are parameterized as unimodal Gaussian (or Laplace), which severely limits their ability to approximate the complex structure induced by multimodal data. In this work, we study the learning problem of the multimodal EBM, shared latent generator, and joint inference model. We present a learning framework that effectively interweaves their MLE updates with corresponding MCMC refinements in both the data and latent spaces. Specifically, the generator is learned to produce coherent multimodal samples that serve as strong initial states for EBM sampling, while the inference model is learned to provide informative latent initializations for generator posterior sampling. Together, these two models serve as complementary models that enable effective EBM sampling and learning, yielding realistic and coherent multimodal EBM samples. Extensive experiments demonstrate superior performance for multimodal synthesis quality and coherence compared to various baselines. We conduct various analyses and ablation studies to validate the effectiveness and scalability of the proposed multimodal framework. 

---
# Born-Qualified: An Autonomous Framework for Deploying Advanced Energy and Electronic Materials 

**Authors**: Steven R. Spurgeon, Milad Abolhasani, Frederick Baddour, Ryan B. Comes, Vinayak P. Dravid, Hilary Egan, Patrick Emami, Robert W. Epps, Davi M. Fébba, Renae Gannon, E. Ashley Gaulding, Ayana Ghosh, Kenny Gruchalla, Grace Guinan, Taro Hitosugi, Michael Holden, Sergei V. Kalinin, Yangang Liang, John S. Mangum, Matthew J. Olszta, Nathaniel H. Park, Axel Palmstrom, Michelle A. Smeaton, Brooks Tellekamp, Nicholas E. Thornburg, Raymond R. Unocic, Daniela Ushizima, Rama K. Vasudevan, Robert White, Andrew Young, Andriy Zakutayev  

**Link**: [PDF](https://arxiv.org/pdf/2605.00639)  

**Abstract**: Autonomous science is transforming how we discover materials and chemical systems for advanced energy technologies. However, many initially promising systems never reach deployment. This "valley of death" stems from optimization that prioritizes laboratory metrics over industrial viability. We propose a new strategy: "born-qualified" autonomous development, which embeds manufacturability, cost, and durability constraints from the outset. This approach is enabled by four pillars, including the development of multi-objective metrics, causal models, a modular infrastructure, and embedding manufacturing in the discovery loop. Realizing this vision will require sustained, community-wide commitment, but the potential return on that investment is commensurate with the scale of the challenge. 

---
# BlenderRAG: High-Fidelity 3D Object Generation via Retrieval-Augmented Code Synthesis 

**Authors**: Massimo Rondelli, Francesco Pivi, Maurizio Gabbrielli  

**Link**: [PDF](https://arxiv.org/pdf/2605.00632)  

**Abstract**: Automatic generation of executable Blender code from natural language remains challenging, with state-of-the-art LLMs producing frequent syntactic errors and geometrically inconsistent objects. We present BlenderRAG, a retrieval-augmented generation system that operates on a curated multimodal dataset of 500 expert-validated examples (text, code, image) across 50 object categories. By retrieving semantically similar examples during generation, BlenderRAG improves compilation success rates from 40.8% to 70.0% and semantic normalized alignment from 0.41 to 0.77 (CLIP similarity) across four state-of-the-art LLMs, without requiring fine-tuning or specialized hardware, making it immediately accessible for deployment. The dataset and code will be available at this https URL. 

---
# Possibilistic Predictive Uncertainty for Deep Learning 

**Authors**: Yao Ni, Jeremie Houssineau, Yew Soon Ong, Piotr Koniusz  

**Link**: [PDF](https://arxiv.org/pdf/2605.00600)  

**Abstract**: Deep neural networks achieve impressive results across diverse applications, yet their overconfidence on unseen inputs necessitates reliable epistemic uncertainty modelling. Existing methods for uncertainty modelling face a fundamental dilemma: Bayesian approaches provide principled estimates but remain computationally prohibitive, while efficient second-order predictors lack rigorous derivations connecting their specific objectives to epistemic uncertainty quantification. To resolve this dilemma, we introduce Dirichlet-approximated possibilistic posterior predictions (DAPPr), a principled framework leveraging possibility theory. We define a possibilistic posterior over parameters, projects this posterior to the prediction space via supremum operators, and approximates the projected posterior using learnable Dirichlet possibility functions. This projection-and-approximation strategy yields a simple training objective with closed-form solutions. Extensive experiments across diverse benchmarks demonstrate that our approach achieves competitive or superior uncertainty quantification performance compared to state-of-the-art evidential deep learning methods while maintaining both principled derivation and computational efficiency. Code will be available at this https URL. 

---
# Fairness of Classifiers in the Presence of Constraints between Features 

**Authors**: Martin C. Cooper, Imane Bousdira  

**Link**: [PDF](https://arxiv.org/pdf/2605.00592)  

**Abstract**: In Machine Learning, an accepted definition of fairness of a decision taken by a classifier is that it should not depend on protected features, such as gender. Unfortunately, when constraints exist between features, such dependencies can be obscured by the constraints.
To avoid this problem, we propose that a decision be considered fair if it has a fair explanation. We define a fair explanation as a prime-implicant reason for the decision that does not contain any protected feature (where the constraints are taken into account in the definition of prime-implicant). Surprisingly, ignoring constraints can completely change the fairness of a decision (according to this definition) even in the absence of constraints between protected and unprotected features.
Three possible definitions of fairness of a classifier are that for all its decisions (1) there are only fair explanations, (2) there is at least one fair explanation, or (3) changing protected features does not change the outcome. We identify the relationships between these different definitions of fairness and study the computational complexity of testing fairness of classifiers. 

---
# Jailbreaking Vision-Language Models Through the Visual Modality 

**Authors**: Aharon Azulay, Jan Dubiński, Zhuoyun Li, Atharv Mittal, Yossi Gandelsman  

**Link**: [PDF](https://arxiv.org/pdf/2605.00583)  

**Abstract**: The visual modality of vision-language models (VLMs) is an underexplored attack surface for bypassing safety alignment. We introduce four jailbreak attacks exploiting the vision component: (1) encoding harmful instructions as visual symbol sequences with a decoding legend, (2) replacing harmful objects with benign substitutes (e.g., bomb -> banana) then prompting for harmful actions using the substitute term, (3) replacing harmful text in images (e.g., on book covers) with benign words while visual context preserves the original meaning, and (4) visual analogy puzzles whose solution requires inferring a prohibited concept. Evaluating across six frontier VLMs, our visual attacks bypass safety alignment and expose a cross-modality alignment gap: text-based safety training does not automatically generalize to harmful intent conveyed visually. For example, our visual cipher achieves 40.9% attack success on Claude-Haiku-4.5 versus 10.7% for an equivalent textual cipher. To further our insight into the attack mechanism, we present preliminary interpretability and mitigation results. These findings highlight that robust VLM alignment requires treating vision as a first-class target for safety post-training. 

---
# AI Washing Inflates Expected Performance but Not Interaction Outcomes: An AI Placebo Study Using Fitts' Law 

**Authors**: Nick von Felten, Luisa Ella Müller, Johannes Schöning  

**Link**: [PDF](https://arxiv.org/pdf/2605.00582)  

**Abstract**: Expectations about the support of artificial intelligence (AI) may influence interaction outcomes similar to placebos. Such expectations may result from AI washing, a practice of overstating a system's AI capabilities when actual functionality is limited. For example, some computer mice are marketed as "AI-assisted" despite lacking AI in core functions. In a within-subjects study, 28 participants completed Fitts' Law tasks with a computer mouse under three conditions: no support, supposed predictive AI support, and supposed biosignal-enhanced AI support. Objective Fitts' Law performance indicators and subjective performance expectations, perceived workload, and perceived usability were measured. Compared to baseline, participants expected significantly improved performance in placebo conditions. However, these expectations did not translate into differences in objective or subjective assessments. This paper contributes evidence that AI washing inflates user expectations without altering actual interaction outcomes, highlighting a critical transparency issue. By exposing how deceptive AI marketing can shape user expectations, we underscore the need for accountability in AI product claims. Further, we establish Fitts' Law as a rigorous methodological lens for auditing AI-labelled input devices. 

---
# Structure Liberates: How Constrained Sensemaking Produces More Novel Research Output 

**Authors**: James Mooney, Zae Myung Kim, Young-Jun Lee, Dongyeop Kang  

**Link**: [PDF](https://arxiv.org/pdf/2605.00557)  

**Abstract**: Scientific discovery is an extended process of ideation--surveying prior work, forming hypotheses, and refining reasoning--yet existing approaches treat this phase as a brief preamble despite its central role in research. We introduce SCISENSE, a sensemaking-grounded framework that operationalizes ideation as a structured sequence of eight cognitive stages (Pirolli \& Card, 2005). We construct SCISENSE-Traj, a 100K-scale dataset of citation-conditioned research trajectories in two modes: Target, where an LLM reconstructs the ideation path leading to a known paper from its cited works, and Infer, where the LLM proposes novel directions from the same citations. We distill these into SCISENSE-LM, a family of sensemaking LLMs spanning 3B to 70B parameters. Contrary to the assumption that looser supervision promotes greater exploration, Target-trained models achieve a 2.0\% improvement in trajectory quality over Infer-trained models while also producing more novel and diverse outputs. This advantage propagates downstream: coding agents conditioned on Target trajectories produce research artifacts with higher executability and quality than those conditioned on Infer trajectories. This suggests that targeted ideation reduces cognitive burden on downstream agents, freeing them to explore more creatively. SCISENSE offers both a practical tool for augmenting LLM-driven research workflows and a principled testbed for studying how planning shapes scientific discovery. 

---
# Linking Behaviour and Perception to Evaluate Meaningful Human Control over Partially Automated Driving 

**Authors**: Ashwin George, Lucas Elbert Suryana, Lorenzo Flipse, Bart van Arem, David A. Abbink, Simeon Craig Calvert, Luciano Cavalcante Siebert, Arkady Zgonnikov  

**Link**: [PDF](https://arxiv.org/pdf/2605.00556)  

**Abstract**: Partial driving automation creates a tension: drivers remain legally responsible for vehicle behaviour, yet their active control is significantly reduced. This reduction undermines the engagement and sense of agency needed to intervene safely. Meaningful human control (MHC) has been proposed as a normative framework to address this tension. However, empirical methods for evaluating whether existing systems actually provide MHC remain underdeveloped. In this study, we investigated the extent to which drivers experience MHC when interacting with partially automated driving systems. Twenty-four drivers completed a simulator study involving silent automation failures under two modes - haptic shared control (HSC) and traded control (TC). We derived behavioural metrics from telemetry data, subjective perception scores from post-trial surveys and used them to test hypothesised relations between them derived from the properties of systems under MHC. The confirmatory analysis showed a significant negative correlation between the perception of the automated vehicle (AV) understanding the driver and conflict in steering torques. An exploratory analysis also revealed a surprising positive correlation between reaction times and the perception of sufficient control. Qualitative feedback from open-ended post-experiment questionnaires revealed that mismatches in intentions between the driver and automation, lack of safety, and resistance to driver inputs contribute to the reduction of perceived MHC, while subtle haptic guidance aligned with driver intent had a positive effect. These findings suggest that future designs should prioritise effortless driver interventions, transparent communication of automation intent, and context-sensitive authority allocation to strengthen meaningful human control in partially automated driving. 

---
# A11y-Compressor: A Framework for Enhancing the Efficiency of GUI Agent Observations through Visual Context Reconstruction and Redundancy Reduction 

**Authors**: Michito Takeshita, Takuro Kawada, Takumi Ohashi, Shunsuke Kitada, Hitoshi Iyatomi  

**Link**: [PDF](https://arxiv.org/pdf/2605.00551)  

**Abstract**: AI agents that interact with graphical user interfaces (GUIs) require effective observation representations for reliable grounding. The accessibility tree is a commonly used text-based format that encodes UI element attributes, but it suffers from redundancy and lacks structural information such as spatial relationships among elements. We propose A11y-Compressor, a framework that transforms linearized accessibility trees into compact and structured representations. Our implementation, Compressed-a11y, applies a lightweight and structured transformation pipeline with modal detection, redundancy reduction, and semantic structuring. Experiments on the OSWorld benchmark show that Compressed-a11y reduces input tokens to 22% of the original while improving task success rates by 5.1 percentage points on average. 

---
# Beyond Continuity: Simulation-free Reconstruction of Discrete Branching Dynamics from Single-cell Snapshots 

**Authors**: Junda Ying, Yuxuan Wang, Bowen Yang, Peijie Zhou, Lei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.00545)  

**Abstract**: Inferring cellular trajectories from destructive snapshots is complicated by the challenges of stochasticity and non-conservative mass dynamics such as cell proliferation and apoptosis. Existing unbalanced Optimal Transport (OT) methods treat mass as a continuous fluid, performing inference at the population level. However, this macroscopic view often fails to capture the discrete, jump-like nature of birth-death events at single-cell resolution, which is essential for understanding lineage branching and fate decisions. We present Unbalanced Schrödinger Bridge (USB), a simulation-free framework for learning underlying dynamics that effectively integrates both stochastic and unbalanced effects which also models the discrete, jump-like birth-death dynamics at single-cell resolution. Theoretically, USB provides a tractable solution to the Branching Schrödinger Bridge (BSB) problem, offering a rigorous microscopic interpretation where individual cells undergo both Brownian motion and discrete birth-death jumps. Technically, the method implements an efficient solver by introducing a simulation-free training objective that effectively scales to high-dimensional omics data. Empirically, we demonstrate on both simulated and real-world datasets that USB not only achieves trajectory reconstruction performance better than or comparable to deterministic baselines but also uniquely enables realistic discrete simulation of birth-death dynamics at single-cell resolution. 

---
# Hierarchical Abstract Tree for Cross-Document Retrieval-Augmented Generation 

**Authors**: Ziwen Zhao, Menglin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.00529)  

**Abstract**: Retrieval-augmented generation (RAG) enhances large language models with external knowledge, and tree-based RAG organizes documents into hierarchical indexes to support queries at multiple granularities. However, existing Tree-RAG methods designed for single-document retrieval face critical challenges in scaling to cross-document multi-hop questions: (1) poor distribution adaptability, where $k$-means clustering introduces noise due to rigid distribution assumptions; (2) structural isolation, as tree indexes lack explicit cross-document connections; and (3) coarse abstraction, which obscures fine-grained details. To address these limitations, we propose $\Psi$-RAG, a tree-RAG framework with two key components. First, a hierarchical abstract tree index built through an iterative "merging and collapse" process that adapts to data distributions without a priori assumption. Second, a multi-granular retrieval agent that intelligently interacts with the knowledge base with reorganized queries and an agent-powered hybrid retriever. $\Psi$-RAG supports diverse tasks from token-level question answering to document-level summarization. On cross-document multi-hop QA benchmarks, it outperforms RAPTOR by 25.9% and HippoRAG 2 by 7.4% in average F1 score. Code is available at this https URL. 

---
# SAGA: Workflow-Atomic Scheduling for AI Agent Inference on GPU Clusters 

**Authors**: Dongxin Guo, Jikun Wu, Siu Ming Yiu  

**Link**: [PDF](https://arxiv.org/pdf/2605.00528)  

**Abstract**: AI agents execute tens to hundreds of chained LLM calls per task, yet GPU schedulers treat each call as independent, discarding gigabytes of intermediate state between steps and inflating end-to-end latency by 3-8x. We argue that this request-level abstraction is fundamentally mismatched to compound AI workloads, and propose a shift to program-level scheduling: treating the entire agent workflow (not individual inference calls) as the first-class schedulable unit. We present SAGA, a distributed scheduler that implements this abstraction through three mechanisms: (1) Agent Execution Graphs that capture workflow structure to predict KV cache reuse across tool-call boundaries, achieving within 1.31x of Bélády's optimal offline policy; (2) session-affinity batching with work stealing that co-locates correlated requests while maintaining global load balance; and (3) Agent Fair Share, a task-completion-time fairness metric with provable bounded-deviation guarantees. On a 64-GPU cluster serving SWE-bench coding agents and WebArena browser tasks, SAGA reduces task completion time by 1.64x (geometric mean, p < 0.001) over vLLM v0.15.1 with prefix caching and affinity routing, while improving GPU memory utilization by 1.22x and achieving 99.2% SLO attainment under multi-tenant interference. These latency gains come at a quantified cost: approximately 30% lower peak throughput than throughput-optimal batch scheduling, a tradeoff appropriate for the latency-sensitive interactive deployments that dominate compound AI usage. Our results demonstrate that workflow-aware scheduling is essential for efficient compound AI serving. 

---
# Silicon Showdown: Performance, Efficiency, and Ecosystem Barriers in Consumer-Grade LLM Inference 

**Authors**: Allan Kazakov, Abdurrahman Javat  

**Link**: [PDF](https://arxiv.org/pdf/2605.00519)  

**Abstract**: The operational landscape of local Large Language Model (LLM) inference has shifted from lightweight models to datacenter-class weights exceeding 70B parameters, creating profound systems challenges for consumer hardware. This paper presents a systematic empirical analysis of the Nvidia and Apple Silicon ecosystems, specifically characterizing the distinct intra-architecture trade-offs required to deploy these massive models. On the Nvidia Blackwell architecture, we identify a critical "Backend Dichotomy" within the TensorRT-LLM stack: while the new NVFP4 quantization format delivers a 1.6x throughput advantage over optimized BF16 baselines (151 tokens/s vs. 92 tokens/s), realizing this performance requires navigating complex runtime constraints that trade startup latency for generation speed. Furthermore, we characterize the "VRAM Wall" for 70B+ models: on discrete GPUs, users face a destructive choice between aggressive quantization (e.g., Q2) that degrades model intelligence to fit in VRAM, or PCIe-bottlenecked CPU offloading, which reduces throughput by over 90% compared to full-GPU execution. Conversely, Apple's Unified Memory Architecture (UMA) circumvents these bottlenecks, enabling linear scaling for 80B parameter models at practical 4-bit precisions. This architectural divergence extends to operational sustainability, where Apple's SoC design demonstrates up to a 23x advantage in energy efficiency (tokens/joule). We conclude that for consumer-grade inference, the optimal hardware is defined by a complex interplay between compute density (Nvidia) and memory capacity (Apple), moderated by the significant "ecosystem friction" of proprietary quantization workflows. 

---
# Space Network of Experts: Architecture and Expert Placement 

**Authors**: Zhanwei Wang, Huiling Yang, Min Sheng, Khaled B. Letaief, Kaibin Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.00515)  

**Abstract**: Leveraging continuous solar energy harvesting at high efficiency, space data centers are envisioned as a promising platform for executing energy-intensive large language models (LLMs). Recognizing this advantage, space and AI conglomerates (e.g., SpaceX, Google) are actively investing in this vision. One key challenge, however, is the efficient distributed deployment of a large-scale LLM in a satellite network due to the limited onboard computing and communication resources. This gives rise to a placement problem that involves partitioning and mapping model components to satellites such that the fundamentally different model architecture and network topology can be reconciled to ensure low-latency token generation. To address this problem, we present the Space Network of Experts (Space-XNet) framework targeting the distributed execution of a popular mixture-of-experts (MoE) model in space. The proposed placement strategies are two-level: (1) layer placement, which assigns MoE layers to satellite subnets; and (2) intra-layer expert placement, which assigns individual experts to satellites associated with the same layer/subnet. For layer placement, we exploit the ring-like communication pattern of autoregressive inference to partition the satellite constellation along the orbiting direction into subnets arranged on a ring, each hosting one MoE layer. Based on this architecture, we formulate and solve an optimization problem for intra-layer expert placement to map experts with heterogeneous activation probabilities onto satellites. The derived strategy reveals an intuitive principle: a frequently activated expert should be mapped to a satellite on a routing path with low expected latency. Experiments over a thousand-satellite constellation show that Space-XNet achieves at least a threefold latency reduction compared with conventional random and ablation-based placement strategies. 

---
# LLM-Oriented Information Retrieval: A Denoising-First Perspective 

**Authors**: Lu Dai, Liang Sun, Fanpu Cao, Ziyang Rao, Cehao Yang, Hao Liu, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2605.00505)  

**Abstract**: Modern information retrieval (IR) is no longer consumed primarily by humans but increasingly by large language models (LLMs) via retrieval-augmented generation (RAG) and agentic search. Unlike human users, LLMs are constrained by limited attention budgets and are uniquely vulnerable to noise; misleading or irrelevant information is no longer just a nuisance, but a direct cause of hallucinations and reasoning failures. In this perspective paper, we argue that denoising-maximizing usable evidence density and verifiability within a context window-is becoming the primary bottleneck across the full information access pipeline. We conceptualize this paradigm shift through a four-stage framework of IR challenges: from inaccessible to undiscoverable, to misaligned, and finally to unverifiable. Furthermore, we provide a pipeline-organized taxonomy of signal-to-noise optimization techniques, spanning indexing, retrieval, context engineering, verification, and agentic workflow. We also present research works on information denoising in domains that rely heavily on retrieval such as lifelong assistant, coding agent, deep research, and multimodal understanding. 

---
# "What Are You Really Trying to Do?": Co-Creating Life Goals from Everyday Computer Use 

**Authors**: Shardul Sapkota, Matthew Jörke, Zane Sabbagh, Omar Shaikh, Grace Wang, James A. Landay  

**Link**: [PDF](https://arxiv.org/pdf/2605.00497)  

**Abstract**: Recent advances in user modeling make it feasible to conduct open-ended inference over a person's everyday computer use. Despite longstanding visions of systems that deeply understand our actions and the purposes they serve in our lives, existing systems only capture what a person is doing in the moment -- not why they are doing it -- limiting these systems to surface-level support. We introduce striving co-creation, a process for inferring broader life goals from unstructured observations of computer use. Grounded in Activity Theory and Emmons' personal strivings framework, our system progressively constructs a hierarchical representation of a person's activities. Crucially, strivings are difficult to fully resolve from observation alone, as the same action can be driven by many different goals. Our system therefore supports an editing interface that gives people agency over how they are understood by the system, feeding their corrections back into subsequent rounds of striving induction. In a week-long field deployment (N=14), we find that our co-creation process produces strivings that are representative of participants' long-term goals and gives them greater agency than baseline methods. 

---
# Scalable Context-Aware Graph Attention for Unsupervised Anomaly Detection in Large-Scale Mobile Networks 

**Authors**: Sara Malacarne, Eirik Hoel-Høiseth, Erlend Aune, David Zsolt Biró, Massimiliano Ruocco  

**Link**: [PDF](https://arxiv.org/pdf/2605.00482)  

**Abstract**: Mobile network operators must monitor thousands of heterogeneous network elements across the radio access network and the packet core, each exposing high-dimensional KPI time series. The scale and cost of incident labelling make supervised approaches impractical, motivating unsupervised anomaly detection robust to context shifts and nonstationarity.
We propose \textbf{C-MTAD-GAT} (\emph{Context-aware Multivariate Time-series Anomaly Detection with Graph Attention}), an anomaly detection framework designed to operate as a single shared model across large populations of network elements. The model combines temporal and feature-wise graph attention with lightweight static and dynamic context conditioning and a dual-head decoder for reconstruction and multi-step forecasting. It produces per-element, per-feature anomaly scores, converted to alerts via fully unsupervised thresholds calibrated from validation residuals.
On the TELCO dataset released with DC-VAE \cite{garcia2023onemodel}, C-MTAD-GAT improves event-level affiliation and pointwise F1 while generating fewer alarms than prior graph-attention and VAE-based baselines. We then apply the same system to nation-scale radio access and evolved packet core control-plane counter data from a mobile network operator, where it is deployed. Operator feedback indicates the alerts are actionable and support daily monitoring, showing scalability across domains without relying on labelled incidents. 

---
# PAMod: Modeling Cyclical Shifts via Phase-Amplitude Modulation for Non-stationary Time Series Forecasting 

**Authors**: Yingbo Zhou, Yutong Ye, Shuhao Li, Rui Qian, Qiang Huang, Lemao Liu, Li Sun, Dejing Dou  

**Link**: [PDF](https://arxiv.org/pdf/2605.00466)  

**Abstract**: Real-world time series forecasting faces the fundamental challenge of non-stationary statistical properties, including shifts in mean and variance over time. While reversible instance normalization (RevIN) has shown promise by stationarizing inputs and denormalizing outputs, it relies on the strong assumption that historical and future distributions remain identical. We observe that in many practical applications, distribution shifts follow cyclical patterns that correlate with periodic positions (e.g., seasonal and holiday volatility). To this end, we propose PAMod, a lightweight yet powerful framework that models cyclical distribution shifts via Phase-Amplitude Modulation in the normalized feature space. PAMod learns periodic embeddings to modulate representations: phase modulation captures mean shifts, while amplitude modulation adapts to variance changes. Crucially, we prove mathematically that modulating in normalized space is equivalent to applying dynamic denormalization, offering an elegant unification of distribution adaptation and representation learning. Extensive experiments on twelve real-world benchmarks demonstrate that PAMod achieves state-of-the-art performance with fewer computational resources. Furthermore, our modulation mechanism, as a novel plug-and-play technique, can improve existing time-series forecasting methods with simple integration. 

---
# Adaptation of AI-accelerated CFD Simulations to the IPU platform 

**Authors**: P. Rosciszewski, A. Krzywaniak, S. Iserte, K. Rojek, P. Gepner  

**Link**: [PDF](https://arxiv.org/pdf/2605.00462)  

**Abstract**: Intelligence Processing Units (IPU) have proven useful for many AI applications. In this paper, we evaluate them within the emerging field of \emph{AI for simulation}, where traditional numerical simulations are supported by artificial intelligence approaches. We focus specifically on a program for training machine learning models supporting a \emph{computational fluid dynamics} application. We use custom TensorFlow provided by the Poplar SDK to adapt the program for the IPU-POD16 platform and investigate its ease of use and performance scalability. Training a model on data from OpenFOAM simulations allows us to get accurate simulation state predictions in test time. We show how to utilize the \emph{popdist} library to overcome a performance bottleneck in feeding training data to the IPU on the host side, achieving up to 34\% speedup. Due to communication overheads, using data parallelism to utilize two IPUs instead of one does not improve the throughput. However, once the intra-IPU costs have been paid, the hardware capabilities for inter-IPU communication allow for good scalability. Increasing the number of IPUs from 2 to 16 improves the throughput from 560.8 to 2805.8 samples/s. 

---
# Impact of Task Phrasing on Presumptions in Large Language Models 

**Authors**: Kenneth J.K. Ong  

**Link**: [PDF](https://arxiv.org/pdf/2605.00436)  

**Abstract**: Concerns with the safety and reliability of applying large-language models (LLMs) in unpredictable real-world applications motivate this study, which examines how task phrasing can lead to presumptions in LLMs, making it difficult for them to adapt when the task deviates from these assumptions. We investigated the impact of these presumptions on the performance of LLMs using the iterated prisoner's dilemma as a case study. Our experiments reveal that LLMs are susceptible to presumptions when making decisions even with reasoning steps. However, when the task phrasing was neutral, the models demonstrated logical reasoning without much presumptions. These findings highlight the importance of proper task phrasing to reduce the risk of presumptions in LLMs. 

---
# Escaping Mode Collapse in LLM Generation via Geometric Regulation 

**Authors**: Xin Du, Kumiko Tanaka-Ishii  

**Link**: [PDF](https://arxiv.org/pdf/2605.00435)  

**Abstract**: Mode collapse is a persistent challenge in generative modeling and appears in autoregressive text generation as behaviors ranging from explicit looping to gradual loss of diversity and premature trajectory convergence. We take a dynamical-systems view and reinterpret mode collapse as reduced state-space accessibility caused by *geometric collapse*: during generation, the model's internal trajectory becomes confined to a low-dimensional region of its representation space. This implies mode collapse is not purely a token-level phenomenon and cannot be reliably solved by symbolic constraints or probability-only decoding heuristics. Guided by this perspective, we propose *Reinforced Mode Regulation* (RMR), a lightweight, online state-space intervention that regulates dominant self-reinforcing directions in the Transformer value cache (implemented as low-rank damping). Across multiple large language models, RMR substantially reduces mode collapse and enables stable, high-quality generation at extremely low entropy rates (down to 0.8 nats/step), whereas standard decoding typically collapses near 2.0 nats/step. 

---
# Improving LLM Code Generation via Requirement-Aware Curriculum Reinforcement Learning 

**Authors**: Shouyu Yin, Zhao Tian, Junjie Chen, Shikai Guo  

**Link**: [PDF](https://arxiv.org/pdf/2605.00433)  

**Abstract**: Code generation, which aims to automatically generate source code from given programming requirements, has the potential to substantially improve software development efficiency. With the rapid advancement of large language models (LLMs), LLM-based code generation has attracted widespread attention from both academia and industry. However, as programming requirements become increasingly complex, existing LLMs still exhibit notable performance limitations. To address this challenge, recent studies have proposed training-based curriculum reinforcement learning (CRL) strategies to improve LLM code generation performance. Despite their effectiveness, existing CRL approaches suffer from several limitations, including misaligned requirement difficulty perception, the absence of requirement difficulty optimization, and suboptimal curriculum sampling strategies. In CRL-based code generation, programming requirements serve as the sole input to the model, making their quality and difficulty critical to training effectiveness. Motivated by insights from software requirements engineering, we propose RECRL, a novel requirement-aware curriculum reinforcement learning framework for enhancing LLM-based code generation. RECRL automatically perceives model-specific requirement difficulty, optimizes challenging requirements to improve training data utilization, and employs an adaptive curriculum sampling strategy to construct training batches with smoothly varying difficulty. Extensive experiments on five state-of-the-art LLMs across five widely-used code generation benchmarks by comparing with five state-of-the-art baselines, demonstrate the significant effectiveness of RECRL. For example, RECRL achieves an average Pass@1 improvement of 1.23%-5.62% over all state-of-the-art baselines. 

---
# Skills as Verifiable Artifacts: A Trust Schema and a Biconditional Correctness Criterion for Human-in-the-Loop Agent Runtimes 

**Authors**: Alfredo Metere  

**Link**: [PDF](https://arxiv.org/pdf/2605.00424)  

**Abstract**: Agent skills -- structured packages of instructions, scripts, and references that augment a large language model (LLM) without modifying the model itself -- have moved from convenience to first-class deployment artifact. The runtime that loads them inherits the same problem package managers and operating systems have always faced: a piece of content claims a behavior; the runtime must decide whether to believe it. We argue this paper's central thesis up front: a skill is \emph{untrusted code} until it is verified, and the runtime that loads it must enforce that default rather than infer trust from a signature, a clearance, or a registry of origin. Without skill verification, a human-in-the-loop (HITL) gate must fire on every irreversible call -- which is operationally untenable and degrades into rubber-stamping at any non-trivial scale. With skill verification treated as a separate, gated process, HITL fires only for what is unverified, and the system becomes sustainable. We give a trust schema (§\ref{sec:schema}) that includes an explicit verification level on every skill manifest; a capability gate (§\ref{sec:gate}) whose HITL policy is a function of that verification level; a \emph{biconditional} correctness criterion (§\ref{sec:biconditional}) that any candidate verification procedure must satisfy on an adversarial-ensemble exercise (§\ref{sec:eval}); and a portable runtime profile (§\ref{sec:guidelines}) with ten normative guidelines abstracted from a working open-source reference implementation \cite{metere2026enclawed}. The contribution is harness- and model-agnostic; nothing here requires retraining, fine-tuning, or proprietary infrastructure. 

---
# BWLA: Breaking the Barrier of W1AX Post-Training Quantization for LLMs 

**Authors**: Zhixiong Zhao, Zukang Xu, Dawei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.00422)  

**Abstract**: Large language models (LLMs) have driven major progress in NLP, yet their substantial memory and compute demands still hinder practical deployment. Binarization can compress weights to 1 bit, fundamentally lowering compute and bandwidth cost. However, existing methods cannot address activation heavy tails and thus must keep activations in high precision, preventing true end-to-end acceleration. To overcome this limitation, we propose BWLA (Binarized Weights and Low-bit Activations), the first post-training quantization framework that preserves high accuracy while achieving 1-bit weight quantization together with low-bit activations (e.g., 6 bits). The Orthogonal-Kronecker Transformation (OKT) learns an orthogonal mapping via EM minimization, converting unimodal weights into symmetric bimodal forms while suppressing activation tails and incoherence. The Proximal SVD Projection (PSP) then performs lightweight low-rank refinement through proximal SVD projection, further enhancing quantizability with minimal overhead. On Qwen3-32B, BWLA reaches a Wikitext2 perplexity of 11.92 under 6-bit activations (vs. 38 from SOTA), improves five zero-shot tasks by more than 70%, and delivers 3.26 times inference speedup, demonstrating strong potential for real-world LLM compression and acceleration. 

---
# RadLite: Multi-Task LoRA Fine-Tuning of Small Language Models for CPU-Deployable Radiology AI 

**Authors**: Pankaj Gupta, Kartik Bose  

**Link**: [PDF](https://arxiv.org/pdf/2605.00421)  

**Abstract**: Large language models (LLMs) show promise in radiology but their deployment is limited by computational requirements that preclude use in resource-constrained clinical environments. We investigate whether small language models (SLMs) of 3-4 billion parameters can achieve strong multi-task radiology performance through LoRA fine-tuning, enabling deployment on consumer-grade CPUs. We train Qwen2.5-3B-Instruct and Qwen3-4B on 162K samples spanning 9 radiology tasks - RADS classification across 10 systems, impression generation, temporal comparison, radiology NLI, NER, abnormality detection, N/M staging, and radiology Q&A - compiled from 12 public datasets. Both models are evaluated on up to 500 held-out test samples per task with standardized metrics. Our key findings are: (1) LoRA fine-tuning dramatically improves performance over zero-shot baselines (RADS accuracy +53%, NLI +60%, N-staging +89%); (2) the two models exhibit complementary strengths - Qwen2.5 excels at structured generation tasks while Qwen3 dominates extractive tasks; (3) a task-outed oracle ensemble combining both models achieves the best performance across all tasks; (4) few-shot prompting with fine-tuned models hurts performance, demonstrating that LoRA adaptation is more effective than in-context learning for specialized domains; and (5) models can be quantized to GGUF format (~1.8-2.4GB) for CPU deployment at 4-8 tokens/second on consumer hardware. Our work demonstrates that small, efficiently fine-tuned models - which we collectively call RadLite - can serve as practical multi-task radiology AI assistants deployable entirely on consumer hardware without GPU requirements. 

---
# Trees to Flows and Back: Unifying Decision Trees and Diffusion Models 

**Authors**: Sai Niranjan Ramachandran, Suvrit Sra  

**Link**: [PDF](https://arxiv.org/pdf/2605.00414)  

**Abstract**: Decision trees and diffusion models are ostensibly disparate model classes, one discrete and hierarchical, the other continuous and dynamic. This work unifies the two by establishing a crisp mathematical correspondence between hierarchical decision trees and diffusion processes in appropriate limiting regimes. Our unification reveals a shared optimization principle: \emph{Global Trajectory Score Matching (GTSM)}, for which gradient boosting (in an idealized version) is asymptotically optimal. We underscore the conceptual value of our work through two key practical instantiations: \treeflow, which achieves competitive generation quality on tabular data with higher fidelity and a 2\times computational speedup, and \dsmtree, a novel distillation method that transfers hierarchical decision logic into neural networks, matching teacher performance within 2\% on many benchmarks. 

---
# Agent Capsules: Quality-Gated Granularity Control for Multi-Agent LLM Pipelines 

**Authors**: Aninda Ray  

**Link**: [PDF](https://arxiv.org/pdf/2605.00410)  

**Abstract**: A multi-agent pipeline with N agents typically issues N LLM calls per run. Merging agents into fewer calls (compound execution) promises token savings, but naively merged calls silently degrade quality through tool loss and prompt compression. We present Agent Capsules, an adaptive execution runtime that treats multi-agent pipeline execution as an optimization problem with empirical quality constraints. The runtime instruments coordination overhead per group, scores composition opportunity, selects among three compound execution strategies, and gates every mode switch on rolling-mean output quality. A controlled negative result confirms that injecting more context into a merged call worsens compression rather than relieving it, so the framework's escalation ladder (standard, then two-phase, then sequential) recovers quality by moving toward per-agent dispatch rather than by rewriting merged prompts. On LLM-judged quality, the controller matches a hand-tuned oracle on every measured (model, group, mode) cell: routing compound whenever the oracle would, and reverting to fine whenever quality would fail the floor, without per-model configuration. Against a hand-crafted LangGraph implementation of a 14-agent competitive intelligence pipeline, Agent Capsules uses 51% fewer fine-mode input tokens and 42% fewer compound-mode input tokens, at +0.020 and +0.017 quality respectively. Against a DSPy implementation of a 5-agent due diligence pipeline, the framework uses 19% fewer tokens than uncompiled DSPy at quality parity, and 68% fewer tokens than MIPROv2 at +0.052 quality. Even before compound mode fires, the runtime delivers efficiency through automatic policy resolution, cache-aligned prompts, and topology-aware context injection, matching both hand-tuned and compile-time baselines without training data or per-pipeline engineering. 

---
# Scalable Learning in Structured Recurrent Spiking Neural Networks without Backpropagation 

**Authors**: Bo Tang, Weiwei Xie  

**Link**: [PDF](https://arxiv.org/pdf/2605.00402)  

**Abstract**: Spiking Neural Networks (SNNs) provide a promising framework for energy-efficient and biologically grounded computation; however, scalable learning in deep recurrent architectures with sparse connectivity remains a major challenge. In this work, we propose a structured multi-layer recurrent SNN architecture composed of locally dense recurrent layers augmented with sparse small-world long-range projections to a readout population. The long-range connectivity is largely fixed, preserving routing efficiency and hardware scalability, while synaptic adaptation is performed using strictly local plasticity mechanisms. To enable supervised learning without backpropagation or surrogate gradients, we introduce a biologically motivated learning framework that combines: (i) population-based winner-take-all (WTA) teaching signals at the output layer, (ii) fixed random broadcast alignment feedback pathways, and (iii) low-dimensional modulatory neuron populations that gate synaptic updates through three-factor learning rules with eligibility traces. This design supports deep recurrent computation with sparse global communication and purely local synaptic updates. We analyze the algorithmic properties, computational complexity, and hardware feasibility of the proposed approach, and demonstrate stable learning and competitive performance on benchmark classification tasks. The results highlight the potential of structured recurrence and neuromodulatory learning to enable scalable, hardware-compatible SNN training beyond gradient-based methods. 

---
# Social Bias in LLM-Generated Code: Benchmark and Mitigation 

**Authors**: Fazle Rabbi, Lin Ling, Song Wang, Jinqiu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.00382)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed to generate code for human-centered applications where demographic fairness is critical. However, existing evaluations focus almost exclusively on functional correctness, leaving social bias in LLM-generated code largely unexamined. Extending our prior work on Solar, we conduct a comprehensive empirical study using SocialBias-Bench, a benchmark of 343 real-world coding tasks spanning seven demographic dimensions. We evaluate four prominent LLMs and find severe bias across all models, with Code Bias Scores reaching up to 60.58%. We further show that standard prompt-level interventions, such as Chain-of-Thought reasoning and fairness persona assignment, inadvertently amplify bias rather than reduce it. We then investigate whether structured multi-agent software process frameworks can improve fairness, finding that structured pipelines reduce bias when early roles correctly scope what the code should and should not consider. However, adding explicit fairness instructions to all agent roles produces worse outcomes than providing none, suggesting that diffused responsibility goes unaddressed. To address these limitations, we propose the Fairness Monitor Agent (FMA), a modular component that plugs into any existing code generation pipeline without modifying it. FMA analyzes the task description to determine which attributes should be considered or restricted, then detects and corrects violations through an iterative review process, without requiring an executable test suite. Evaluated on all 343 tasks, FMA reduces bias by 65.1% compared to a developer agent alone and improves functional correctness from 75.80% to 83.97%, outperforming all other studied approaches. 

---
# GaMMA: Towards Joint Global-Temporal Music Understanding in Large Multimodal Models 

**Authors**: Zuyao You, Zhesong Yu, Mingyu Liu, Bilei Zhu, Yuan Wan, Zuxuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.00371)  

**Abstract**: In this paper, we propose GaMMA, a state-of-the-art (SoTA) large multimodal model (LMM) designed to achieve comprehensive musical content understanding. GaMMA inherits the streamlined encoder-decoder design of LLaVA, enabling effective cross-modal learning between music and language. By incorporating audio encoders in a mixture-of-experts manner, GaMMA effectively unifies both time-series and non-time-series music understanding tasks within one set of parameters. Our approach combines carefully curated datasets at scale with a progressive training pipeline, effectively pushing the boundaries of music understanding via pretraining, supervised fine-tuning (SFT), and reinforcement learning (RL). To comprehensively assess both temporal and non-temporal capability of music LMMs, we introduce MusicBench, the largest music-oriented benchmark, comprising 3,739 human-curated multiple-choice questions covering diverse aspects of musical understanding. Extensive experiments demonstrate that GaMMA establishes new SoTA in the music domain, achieving 79.1% accuracy on MuchoMusic, 79.3% on MusicBench-Temporal, and 81.3% on MusicBench-Global, consistently outperforming previous methods. 

---
# AlphaInventory: Evolving White-Box Inventory Policies via Large Language Models with Deployment Guarantees 

**Authors**: Chenyu Huang, Jianghao Lin, Zhengyang Tang, Bo Jiang, Ruoqing Jiang, Benyou Wang, Lai Wei  

**Link**: [PDF](https://arxiv.org/pdf/2605.00369)  

**Abstract**: We study how large language models can be used to evolve inventory policies in online, non-stationary environments. Our work is motivated by recent advances in LLM-based evolutionary search, such as AlphaEvolve, which demonstrates strong performance for static and highly structured problems such as mathematical discovery, but is not directly suited to online dynamic inventory settings. To this end, we propose AlphaInventory, an end-to-end inventory-policy evolution and inference framework grounded in confidence-interval-based certification. The framework trains a large language model using reinforcement learning, incorporates demand data as well as numerical and textual features beyond demand, and generates white-box inventory policy with statistical safety guarantees for deployment in future periods. We further introduce a unified theoretical interface that connects training, inference, and deployment. This allows us to characterize the probability that the AlphaInventory evolves a statistically safe and improved policy, and to quantify the deployment gap relative to the oracle-safe benchmark. Tested on both synthetic data and real-world retail data, AlphaInventory outperforms classical inventory policies and deep learning based methods. In canonical inventory settings, it evolves new policies that improve upon existing benchmarks. 

---
# Pedagogical Promise and Peril of AI: A Text Mining Analysis of ChatGPT Research Discussions in Programming Education 

**Authors**: Juvy C.Grume, John Paul P. Miranda, Aileen P. De Leon, Jordan L. Salenga, Hilene E. Hernandez, Mark Anthony A. Castro, Vernon Grace M. Maniago, Joel D. Canlas, Joel B. Quiambao  

**Link**: [PDF](https://arxiv.org/pdf/2605.00361)  

**Abstract**: GenAI systems such as ChatGPT are increasingly discussed in programming education, but the ways in which the research literature conceptualizes and frames their role remain unclear. This chapter applies text mining to publications indexed in a leading academic database to map scholarly discourse on ChatGPT in programming education. Term frequency analysis, phrase pattern extraction, and topic modeling reveal four dominant themes: pedagogical implementation, student-centered learning and engagement, AI infrastructure and human-AI collaboration, and assessment, prompting, and model evaluation. The literature prioritizes classroom practice and learner interaction, with comparatively limited attention to assessment design and institutional governance. Across studies, ChatGPT is positioned both as a learning aid that supports explanation, feedback, and efficiency and as a pedagogical risk linked to overreliance, unreliable outputs, and academic integrity concerns. These findings support responsible integration and highlight the need for stronger assessment and governance mechanisms. 

---
# MemRouter: Memory-as-Embedding Routing for Long-Term Conversational Agents 

**Authors**: Tianyu Hu, Weikai Lin, Weizhi Zhang, Jing Ma, Song Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.00356)  

**Abstract**: Long-term conversational agents must decide which turns to store in external memory, yet recent systems rely on autoregressive LLM generation at every turn to make that decision. We present MemRouter, a write-side memory router that decouples memory admission from the downstream answer backbone and replaces per-turn memory-management decoding with an embedding-based routing policy. MemRouter encodes each turn together with recent context, projects the resulting embeddings through a frozen LLM backbone, and predicts whether the turn should be stored using lightweight classification heads while training only 12M parameters. Under a controlled matched-harness comparison on LoCoMo, where the retrieval pipeline, answer prompts, and QA backbone (Qwen2.5-7B) are held identical, MemRouter outperforms an LLM-based memory manager on every question category (overall F1 52.0 vs 45.6, non-overlapping 95% CIs) while reducing memory-management p50 latency from 970ms to 58ms. Descriptive factorial averaging further shows that learned admission improves mean F1 by +10.3 over random storage, category-specific prompting adds +5.2 over a generic prompt, and retrieval contributes +0.7. These results suggest that write-side memory admission can be learned by a small supervised router, while answer generation remains a separate downstream component in long-horizon conversational QA. 

---
# VQ-SAD: Vector Quantized Structure Aware Diffusion For Molecule Generation 

**Authors**: Farshad Noravesh, Reza Haffari, Layki Soon, Arghya Pal  

**Link**: [PDF](https://arxiv.org/pdf/2605.00354)  

**Abstract**: Many diffusion based molecule generation methods ignore the symbolic information of molecules and represent the atom and bond type as one hot representation. Methods based on Morgan fingerprints produce hash collisions and are hard to embed into a continuous space without information loss and random fingerprints correspond to no valid molecule. To circumvent this issue we use another paradigm and consider atom and bond codes as latent variables of VQ-VAE. We introduce VQ-SAD which first trains a VQ-VAE and uses the frozen pretrained VQ-VAE model and considers the codebooks for both atom and bond types as tokenizers for the downstream diffusion process. VQ-SAD is a neuro-symbolic model that utilizes both symbolic and neural structural information for a diffusion based model with learnable forward process. The large discrete code space provides a more balanced atom and bond types which enhances the denoising process. VQ-VAE slightly outperforms SOTA models for diffusion based molecule generation on QM9 and ZINC250k datasets. 

---
# Hypergraph and Latent ODE Learning for Multimodal Root Cause Localization in Microservices 

**Authors**: Xin Liu, Yuhang He, Sichen Zhao, Kejian Tong, Xingyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.00351)  

**Abstract**: Root cause localization in cloud native microservice systems requires modeling complex service dependencies, irregular temporal dynamics, and heterogeneous observability data. We present HyperODE RCA, a unified framework that combines hypergraph attention learning, latent ordinary differential equations, and multimodal cross attention fusion for fine grained root cause analysis. The method learns higher order service interactions through differentiable hyperedge construction, captures continuous anomaly evolution from irregular observations with an ODE RNN encoder, and adaptively fuses logs, traces, metrics, entities, and events using context aware modality routing. We further improve robustness with a variational information bottleneck, temporal causal regularization, and invariant risk constraints. Experiments on the Tianchi AIOps benchmark show clear gains over strong baselines in ranking and classification performance, while preserving interpretability through learned hypergraph attention. 

---
# Odysseus: Scaling VLMs to 100+ Turn Decision-Making in Games via Reinforcement Learning 

**Authors**: Chengshuai Shi, Wenzhe Li, Xinran Liang, Yizhou Lu, Wenjia Yang, Ruirong Feng, Seth Karten, Ziran Yang, Zihan Ding, Gabriel Sarch, Danqi Chen, Karthik Narasimhan, Chi Jin  

**Link**: [PDF](https://arxiv.org/pdf/2605.00347)  

**Abstract**: Given the rapidly growing capabilities of vision-language models (VLMs), extending them to interactive decision-making tasks such as video games has emerged as a promising frontier. However, existing approaches either rely on large-scale supervised fine-tuning (SFT) on human trajectories or apply reinforcement learning (RL) only in relatively short-horizon settings (typically around 20--30 turns). In this work, we study RL-based training of VLMs for long-horizon decision-making in Super Mario Land, a visually grounded environment requiring 100+ turns of interaction with coordinated perception, reasoning, and action. We begin with a systematic investigation of key algorithmic components and propose an adapted variant of PPO with a lightweight turn-level critic, which substantially improves training stability and sample efficiency over critic-free methods such as GRPO and Reinforce++. We further show that pretrained VLMs provide strong action priors, significantly improving sample efficiency during RL training and reducing the need for manual design choices such as action engineering, compared to classical deep RL trained from scratch. Building on these insights, we introduce Odysseus, an open training framework for VLM agents, achieving substantial gains across multiple levels of the game and at least 3 times average game progresses than frontier models. Moreover, the trained models exhibit consistent improvements under both in-game and cross-game generalization settings, while maintaining general-domain capabilities. Overall, our results identify key ingredients for making RL stable and effective in long-horizon, multi-modal settings, and provide practical guidance for developing VLMs as embodied agents. 

---
# AI Adoption Among Teachers: Insights on Concerns, Support, Confidence, and Attitudes 

**Authors**: Vanessa B. Sibug, Maria Anna D. Cruz, Vicky P. Vital, Juvy C. Grume, Almer B. Gamboa, Emerson Q. Fernando, Lloyd D. Feliciano, Jordan L. Salenga, John Paul P. Miranda  

**Link**: [PDF](https://arxiv.org/pdf/2605.00343)  

**Abstract**: The study examines the adoption of artificial intelligence (AI) tools in education by analyzing the roles of institutional support, teacher confidence, and teacher concerns. It aims to determine whether teacher concerns moderate the relationship between institutional support and two outcomes: teacher confidence and attitudes toward AI adoption. The sample included 260 teachers from the Philippines. Composite scores were calculated for institutional support, confidence, concerns, and attitudes. Moderated multiple regression analysis showed that institutional support significantly predicted both teacher confidence and attitudes toward AI. However, teacher concerns did not significantly moderate these relationships. A follow-up mediation analysis tested whether confidence explains the effect of institutional support on attitudes. Results showed full mediation. The indirect effect was significant based on the Sobel test, and the direct effect became non-significant when confidence was included in the model. This shows that institutional support improves teacher attitudes by increasing their confidence. The study recommends that institutions provide structured and ongoing support to strengthen teacher confidence. Professional development, mentoring, and AI integration in teacher education programs can increase readiness and support effective AI adoption. 

---
# Budget-Aware Routing for Long Clinical Text 

**Authors**: Khizar Qureshi, Geoffrey Martin, Yifan Peng  

**Link**: [PDF](https://arxiv.org/pdf/2605.00336)  

**Abstract**: A key challenge for large language models is token cost per query and overall deployment cost. Clinical inputs are long, heterogeneous, and often redundant, while downstream tasks are short and high stakes. We study budgeted context selection, where a subset of document units is chosen under a strict token budget so an off-the-shelf generator can meet fixed cost and latency constraints. We cast this as a knapsack-constrained subset selection problem with two design choices, unitization that defines document segmentation and selection that determines which units are kept.
We propose \textbf{RCD}, a monotone submodular objective that balances relevance, coverage, and diversity. We compare sentence, section, window, and cluster-based unitization, and introduce a routing heuristic that adapts to the budget regime. Experiments on MIMIC discharge notes, Cochrane abstracts, and L-Eval show that optimal strategies depend on the evaluation setting. Positional heuristics perform best at low budgets in extractive tasks, while diversity-aware methods such as MMR improve LLM generation. Selector choice matters more than unitization, with cluster-based grouping reducing performance and other schemes behaving similarly. ROUGE saturates for LLM summaries, while BERTScore better reflects quality differences. We release our code at this https URL. 

---
# DynamicPO: Dynamic Preference Optimization for Recommendation 

**Authors**: Xingyu Hu, Kai Zhang, Jiancan Wu, Shuli Wang, Chi Wang, Wenshuai Chen, Yinhua Zhu, Haitao Wang, Xingxing Wang, Xiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.00327)  

**Abstract**: In large language model (LLM)-based recommendation systems, direct preference optimization (DPO) effectively aligns recommendations with user preferences, requiring multi-negative objective functions to leverage abundant implicit-feedback negatives and sharpen preference boundaries. However, our empirical analyses reveal a counterintuitive phenomenon, preference optimization collapse, where increasing the number of negative samples can lead to performance degradation despite a continuously decreasing training loss. We further theoretically demonstrate that this collapse arises from gradient suppression, caused by the dominance of easily discriminable negatives over boundary-critical negatives that truly define user preference boundaries. As a result, boundary-relevant signals are under-optimized, weakening the model's decision boundary. Motivated by these observations, we propose DynamicPO (Dynamic Preference Optimization), a lightweight and plug-and-play framework comprising two adaptive mechanisms: Dynamic Boundary Negative Selection, which identifies and prioritizes informative negatives near the model's decision boundary, and Dual-Margin Dynamic beta Adjustment, which calibrates optimization strength per sample according to boundary ambiguity. Extensive experiments on three public datasets show that DynamicPO effectively prevents optimization collapse and improves recommendation accuracy on multi-negative preference optimization methods, with negligible computational overhead. Our code and datasets are available at this https URL. 

---
# Unbox Responsible GeoAI: Navigating Climate Extreme and Disaster Mapping 

**Authors**: Hao Li, Steffen Knoblauch  

**Link**: [PDF](https://arxiv.org/pdf/2605.00315)  

**Abstract**: As climate extreme and disaster events become more frequent and intense, Geospatial Artificial Intelligence (GeoAI) has emerged as a transformative approach for large-scale disaster mapping and risk reduction. However, the purely mechanical, performance-driven deployment of GeoAI models can result in amplifying inherent spatial inequalities, preventing effective emergency decision-making, and producing severe environmental carbon footprint. To unbox the concept of responsible GeoAI, this position paper examines its emerging role, e.g., in climate extreme and disaster mapping, from a critical GIS perspective. We address the nexus of responsible GeoAI into four interrelated theoretical dimensions, specifically Representativeness, Explainability, Sustainability, and Ethics, with examples from climate extreme and disaster mapping. Moreover, targeting at the operational practice, we then propose a conceptual governance Model of responsible GeoAI that categorizes its governance practices into Data, Application, and Society scopes. Last, this position paper aims to raise the attention in the broader GIS community that the future of climate resilience relies not just on building better algorithms, but on fostering a governance ecosystem where GeoAI is deployed responsibly, ethically, and sustainably. 

---
# Semia: Auditing Agent Skills via Constraint-Guided Representation Synthesis 

**Authors**: Hongbo Wen, Ying Li, Hanzhi Liu, Chaofan Shou, Yanju Chen, Yuan Tian, Yu Feng  

**Link**: [PDF](https://arxiv.org/pdf/2605.00314)  

**Abstract**: An agent skill is a configuration package that equips an LLM-driven agent with a concrete capability, such as reading email, executing shell commands, or signing blockchain transactions. Each skill is a hybrid artifact-a structured half declares executable interfaces, while a prose half dictates when and how those interfaces fire-and the prose is reinterpreted probabilistically on every invocation. Conventional static analyzers parse the structured half but ignore the prose; LLM-based tools read the prose but cannot reproducibly prove that a tainted input reaches a high-impact sink.
We present Semia, a static auditor for agent skills. Semia lifts each skill into the Skill Description Language (SDL), a Datalog fact base that captures LLM-triggered actions, prose-defined conditions, and human-in-the-loop checkpoints. Synthesizing a fact base that is both structurally sound and semantically faithful to the original prose is the central challenge; we address it with Constraint-Guided Representation Synthesis (CGRS), a propose-verify-evaluate loop that refines LLM candidates until convergence. Security properties (e.g., indirect injection, secret leakage, confused deputies, unguarded sinks, etc.) over an agent skill can then be reduced to Datalog reachability queries. We evaluate Semia on 13,728 real-world skills from public marketplaces. Semia renders all of them auditable and finds that more than half carry at least one critical semantic risk. On a stratified sample of 541 expert-labeled skills, Semia achieves 97.7% recall and an F1 of 90.6%, substantially outperforming signature-based scanners and LLM baselines. 

---
# Beyond Structure: Revolutionising Materials Discovery via AI-Driven Synthesis Protocol-Property Relationships 

**Authors**: Guillaume Lambard  

**Link**: [PDF](https://arxiv.org/pdf/2605.00313)  

**Abstract**: The current structure-centric paradigm in artificial intelligence (AI)-driven materials discovery, despite delivering thousands of candidate structures, is stalling at a critical barrier: the synthesizability gap. We argue that closing this gap demands a pivot to a synthesis-first paradigm in which executable synthesis protocols, not just atomic configurations, are treated as primary design variables. We outline a roadmap built on three pillars: (i) representing synthesis procedures as machine-readable protocols, (ii) deploying generative and inverse-design models to propose actionable reaction pathways and recipes, and (iii) integrating closed-loop optimisation to refine protocols against experimental realities and sustainability constraints. Framed in terms of the causal backbone P->X->y from protocol P to structure X and properties y, this perspective sets out methodological building blocks, standards needs and self-driving laboratory (SDL) integration strategies to accelerate reproducible, data-first materials discovery. 

---
# Beyond Visual Fidelity: Benchmarking Super-Resolution Models for Large-Scale Remote Sensing Imagery via Downstream Task Integration 

**Authors**: Zhili Li, Kangyang Chai, Zhihao Wang, Xiaowei Jia, Yanhua Li, Gengchen Mai, Sergii Skakun, Dinesh Manocha, Yiqun Xie  

**Link**: [PDF](https://arxiv.org/pdf/2605.00310)  

**Abstract**: Super-resolution (SR) techniques have made major advances in reconstructing high-resolution images from low-resolution inputs. The increased resolution provides visual enhancement and utility for monitoring tasks. In particular, SR has been increasingly developed for satellite-based Earth observation, with applications in urban planning, agriculture, ecology, and disaster response. However, existing SR studies and benchmarks typically use fidelity metrics such as PSNR or SSIM, whereas the true utility of super-resolved images lies in supporting downstream tasks such as land cover classification, biomass estimation, and change detection. To bridge this gap, we introduce GeoSR-Bench, a downstream task-integrated SR benchmark dataset to evaluate SR models beyond fidelity metrics. GeoSR-Bench comprises spatially co-located, temporally aligned, and quality-controlled image pairs from about 36,000 locations across diverse land covers, spanning resolutions from 500m to 0.6m. To the best of our knowledge, GeoSR-Bench is the first SR benchmark that directly connects improved image resolution from SR models with downstream Earth monitoring tasks, including land cover segmentation, infrastructure mapping, and biophysical variable estimation. Using GeoSR-Bench, we benchmark GAN, transformer, neural operator, and diffusion-based SR models on perceptual quality and downstream task performance. We conduct experiments with 270 settings, covering 2 cross-platform SR tasks, 9 SR models, 3 downstream task models, and 5 downstream tasks for each SR task. The results show that improvements in traditional SR metrics often do not correlate with gains in task performance, and the correlations can be negative, indicating that these metrics provide limited guidance for selecting superior models for downstream tasks. This reveals the need to integrate downstream tasks into SR model development and evaluation. 

---
# Caracal: Causal Architecture via Spectral Mixing 

**Authors**: Bingzheng Gan, Tianyi Zhang, Yusu Li, Jing Huang, Wei Shi, Yangkai Ding, Tao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.00292)  

**Abstract**: The scalability of Large Language Models to long sequences is hindered by the quadratic cost of attention and the limitations of positional encodings. To address these, we introduce Caracal, a novel architecture that replaces attention with a parameter-efficient, $\mathcal{O}(L \log L)$ Multi-Head Fourier (MHF) module. Our contributions are threefold: (1) We leverage the Fast Fourier Transform (FFT) for sequence mixing, inherently addressing both bottlenecks mentioned above. (2) We apply a frequency-domain causal masking technique that enforces autoregressive capabilities via asymmetric padding and truncation, overcoming a critical barrier for Fourier-based generative models. (3) Unlike efficient models relying on hardware-specific implementations (e.g., Mamba), we uses standard library operators. This ensures robust portability, eliminating common deployment barriers. Evaluations demonstrate that Caracal performs competitively with Transformer and SSM baselines, offering a scalable and simple pathway for efficient long-sequence modeling. Code is available in Appendix. 

---
# When Do Diffusion Models learn to Generate Multiple Objects? 

**Authors**: Yujin Jeong, Arnas Uselis, Iro Laina, Seong Joon Oh, Anna Rohrbach  

**Link**: [PDF](https://arxiv.org/pdf/2605.00273)  

**Abstract**: Text-to-image diffusion models achieve impressive visual fidelity, yet they remain unreliable in multi-object generation. Despite extensive empirical evidence of these failures, the underlying causes remain unclear. We begin by asking how much of this limitation arises from the data itself. To disentangle data effects, we consider two regimes across different dataset sizes: (1) concept generalization, where each individual concept is observed during training under potentially imbalanced data distributions, and (2) compositional generalization, where specific combinations of concepts are systematically held out. To study these regimes, we introduce mosaic (Multi-Object Spatial relations, AttrIbution, Counting), a controlled framework for dataset generation. By training diffusion models on mosaic, we find that scene complexity plays a dominant role rather than concept imbalance, and that counting is uniquely difficult to learn in low-data regimes. Moreover, compositional generalization collapses as more concept combinations are held out during training. These findings highlight fundamental limitations of diffusion models and motivate stronger inductive biases and data design for robust multi-object compositional generation. 

---
# REALM: An RGB and Event Aligned Latent Manifold for Cross-Modal Perception 

**Authors**: Vincenzo Polizzi, David B. Lindell, Jonathan Kelly  

**Link**: [PDF](https://arxiv.org/pdf/2605.00271)  

**Abstract**: Event cameras provide several unique advantages over standard frame-based sensors, including high temporal resolution, low latency, and robustness to extreme lighting. However, existing learning-based approaches for event processing are typically confined to narrow, task-specific silos and lack the ability to generalize across modalities. We address this gap with REALM, a cross-modal framework that learns an RGB and Event Aligned Latent Manifold by projecting event representations into the pretrained latent space of RGB foundation models. Instead of task-specific training, we leverage low-rank adaptation (LoRA) to bridge the modality gap, effectively unlocking the geometric and semantic priors of frozen RGB backbones for asynchronous event streams. We demonstrate that REALM effectively maps events into the ViT-based foundation latent space. Our method allows us to perform downstream tasks like depth estimation and semantic segmentation by simply transferring linear heads trained on the RGB teacher. Most significantly, REALM enables the direct, zero-shot application of complex, frozen image-trained decoders, such as MASt3R, to raw event data. We demonstrate state-of-the-art performance in wide-baseline feature matching, significantly outperforming specialized architectures. Code and models are available upon acceptance. 

---
# Are You the A-hole? A Fair, Multi-Perspective Ethical Reasoning Framework 

**Authors**: Sheza Munir, Ahanaf Rodoshi, Sumin Lee, Feiran Chang, Xujie Si, Syed Ishtiaque Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2605.00270)  

**Abstract**: Standard methods for aggregating natural language judgments, such as majority voting, often fail to produce logically consistent results when applied to high-conflict domains, treating differing opinions as noise. We propose a neuro-symbolic aggregation framework that formalizes conflict resolution through Weighted Maximum Satisfiability (MaxSAT). Our pipeline utilizes a language model to map unstructured natural language explanations into interpretable logical predicates and confidence weights. These components are then encoded as soft constraints within the Z3 solver, transforming the aggregation problem into an optimization task that seeks the maximum consistency across conflicting testimony. Using the Reddit r/AmItheAsshole forum as a case study in large-scale moral disagreement, our system generates logically coherent verdicts that diverge from popularity-based labels 62% of the time, corroborated by an 86% agreement rate with independent human evaluators. This study demonstrates the efficacy of coupling neural semantic extraction with formal solvers to enforce logical soundness and explainability in the aggregation of noisy human reasoning. 

---
# Jailbroken Frontier Models Retain Their Capabilities 

**Authors**: Daniel Zhu, Zihan Wang, Jenny Bao, Jerry Wei  

**Link**: [PDF](https://arxiv.org/pdf/2605.00267)  

**Abstract**: As language model safeguards become more robust, attackers are pushed toward developing increasingly complex jailbreaks. Prior work has found that this complexity imposes a "jailbreak tax" that degrades the target model's task performance. We show that this tax scales inversely with model capability and that the most advanced jailbreaks effectively yield no reduction in model capabilities. Evaluating 28 jailbreaks on five benchmarks across Claude models ranging in capability from Haiku 4.5 to Opus 4.6, we find Haiku 4.5 loses an average of 33.1% on benchmark performance when jailbroken, while Opus 4.6 at max thinking effort loses only 7.7%. We also observe that across all models, reasoning-heavy tasks display considerably more degradation than knowledge-recall tasks. Finally, Boundary Point Jailbreaking, currently the strongest jailbreak against deployed classifiers, achieves near-perfect classifier evasion with near-zero degradation across safeguarded models. We recommend that safety cases for frontier models should not rely on a meaningful capability degradation from jailbreaks. 

---
# Retrieval-Augmented Reasoning for Chartered Accountancy 

**Authors**: Jatin Gupta, Akhil Sharma, Saransh Singhania, Ali Imam Abidi  

**Link**: [PDF](https://arxiv.org/pdf/2605.00257)  

**Abstract**: The inception of Large Language Models (LLMs) has catalyzed AI adoption in the finance sector, yet their reliability in complex, jurisdiction-specific tasks like Indian Chartered Accountancy (CA) remains limited. The models display difficulty in executing numerical tasks which require multiple steps while also needing advanced knowledge about legal regulations and the method of scaling their operations is not feasible in settings which have limited access to resources. We present CA-ThinkFlow as a parameter-efficient Retrieval-Augmented Generation (RAG) framework which operates with a 14B, 4-bit-quantized reasoning model, 14B-DeepSeek-R1, and a layout-aware Docling extraction system which maintains document structure during extraction. CA-ThinkFlow uses a basic RAG method which automatically adds retrieved information into the prompt, while it depends on the model's built-in Chain-of-Thought (CoT) functions to create context and produce correct answers. The system we developed system operates at performance levels which match large proprietary models when we tested it on the multi-level CA-Ben benchmark, achieving Scholastic Reliability Coefficient (SRC) results which equal 68.75\% of GPT-4o and Claude 3.5 Sonnet. The framework shows high efficiency and strength in handling parameters, but essential reasoning abilities fail to process complex regulatory texts which exist in fields such as Taxation. 

---
# Remote SAMsing: From Segment Anything to Segment Everything 

**Authors**: Osmar Luiz Ferreira de Carvalho, Osmar Abílio de Carvalho Júnior, Anesmar Olino de Albuquerque, Daniel Guerreiro e Silva  

**Link**: [PDF](https://arxiv.org/pdf/2605.00256)  

**Abstract**: SAM2 produces high-quality zero-shot segmentation on natural images, but applying it to large remote sensing scenes exposes two problems: (1) its mask generator faces an inherent quality-coverage trade-off: strict thresholds yield precise masks but leave most of the image unsegmented, while relaxed thresholds increase coverage at the cost of mask quality; and (2) large images must be tiled, fragmenting objects across tile boundaries. We propose Remote SAMsing, an open-source pipeline that solves both problems without modifying SAM2 or requiring training data. For coverage, a multi-pass algorithm runs SAM2 repeatedly on each tile, painting accepted masks black between passes to simplify the scene for the next iteration, and relaxing quality thresholds only when coverage gains stagnate, ensuring that the most precise masks are always captured first. For spatial consistency, contextual padding and a parameter-free best-match merge reconstruct objects fragmented across tile boundaries. Evaluated on seven scenes (5~cm to 4.78~m GSD), the pipeline raises coverage from 30--68\% (single-pass SAM2) to 91--98\%. Ablation experiments quantify the contribution of each component to coverage and detection quality. Per-class evaluation shows that SAM2 transfers well to discrete RS objects (buildings 95\%, cars 82--93\% Det@0.5) with segment boundaries 3--8$\times$ more precise than SLIC and Felzenszwalb baselines. Tile size functions as an implicit scale parameter: reducing it from $1{,}000$ to 250 raises Det@0.5 from 56\% to 85\%, outperforming SAM2's built-in multi-scale mechanism. The pipeline generalizes to MNF false-color imagery without retraining (99.5\% ASA) and scales to production-sized images: a 1.94 billion pixel Potsdam mosaic achieved 97\% coverage without quality degradation. 

---
# Rethinking Network Topologies for Cost-Effective Mixture-of-Experts LLM Serving 

**Authors**: Junsun Choi, Sam Son, Sunjin Choi, Hansung Kim, Yakun Sophia Shao, Scott Shenker, Sylvia Ratnasamy, Borivoje Nikolic  

**Link**: [PDF](https://arxiv.org/pdf/2605.00254)  

**Abstract**: Mixture-of-experts (MoE) architectures have turned LLM serving into a cluster-scale workload in which communication consumes a considerable portion of LLM serving runtime. This has prompted industry to invest heavily in expensive high-bandwidth scale-up networks. We question whether such costly infrastructure is strictly necessary. We present the first systematic cross-layer analysis of network cost-effectiveness for MoE LLM serving, comparing four representative XPU (e.g., GPU/TPU) topologies (scale-up, scale-out, 3D torus, and 3D full-mesh). We find that lower-cost switchless topologies are more cost-effective than the scale-up topology across all serving scenarios explored, improving cost-effectiveness by 20.6-56.2%. In particular, the 3D full-mesh topology is Pareto-optimal in terms of the performance-cost tradeoff. We also find that current scale-up link bandwidths are over-provisioned: reducing the link bandwidth improves throughput per cost by up to 27%. A forward-looking analysis of upcoming GPU generations indicates that the cost-performance advantage of switchless networks will likely persist. 

---
# MAEPose: Self-Supervised Spatiotemporal Learning for Human Pose Estimation on mmWave Video 

**Authors**: Xijia Wei, Yuan Fang, Kevin Chetty, Youngjun Cho, Nadia Bianchi-Berthouze  

**Link**: [PDF](https://arxiv.org/pdf/2605.00242)  

**Abstract**: Millimetre-wave (mmWave) radar offers a more privacy-preserving alternative to RGB-based human pose estimation. However, existing methods typically rely on pre-extracted intermediate representations such as sparse point clouds or spectrogram images, where the rich spatiotemporal information naturally present in radar video streams is discarded for model learning, while such signal processing adds system complexity. In addition, existing solutions are mainly conducted in an end-to-end supervised manner without leveraging unlabelled raw video streams to learn generalized representations. In this study, we present MAEPose, a masked autoencoding-based human pose estimation approach that operates directly on mmWave spectrogram videos. MAEPose learns spatiotemporal motion-aware generalized representations from unlabelled radar video, and leverages its heatmap decoder for multi-frame pose estimation predictions. We evaluate it across three datasets based on leave-one-person-out cross-validation with rigorous statistical testing. MAEPose consistently outperforms state-of-the-art baselines by up to 22.1% in MPJPE p<0.05, and maintains robust accuracy under zero-shot bystander interference with only a 6.5% error increase. Ablation studies confirm that both the pre-training and the heatmap decoder contribute substantially, while modality analysis indicates that leveraging Range-Doppler video as input achieves better pose estimation performance than Range-Azimuth or their fusion, with lower computational cost. 

---
# Attention Is Where You Attack 

**Authors**: Aviral Srivastava, Sourav Panda  

**Link**: [PDF](https://arxiv.org/pdf/2605.00236)  

**Abstract**: Safety-aligned large language models rely on RLHF and instruction tuning to refuse harmful requests, yet the internal mechanisms implementing safety behavior remain poorly understood. We introduce the Attention Redistribution Attack (ARA), a white-box adversarial attack that identifies safety-critical attention heads and crafts nonsemantic adversarial tokens that redirect attention away from safety-relevant positions. Unlike prior jailbreak methods operating at the semantic or output-logit level, ARA targets the geometry of softmax attention on the probability simplex using Gumbel-softmax optimization over targeted heads. Across LLaMA-3-8B-Instruct, Mistral-7B-Instruct-v0.1, and Gemma-2-9B-it, ARA bypasses safety alignment with as few as 5 tokens and 500 optimization steps, achieving 36% ASR on Mistral-7B and 30% on LLaMA-3 against 200 HarmBench prompts, while Gemma-2 remains at 1%. Our principal mechanistic finding is a dissociation between ablation and redistribution: zeroing out the top-ranked safety heads produces at most 1 flip among 39 to 50 baseline refusals, while ARA targeting the corresponding safety-heavy layers flips 72/200 prompts on Mistral-7B and 60/200 on LLaMA-3. This suggests that safety is not localized in these heads as removable components, but emerges from the attention routing they perform. Removing a head allows compensation through the residual stream, while redirecting its attention propagates a corrupted signal downstream. 

---
# Why Do LLMs Struggle in Strategic Play? Broken Links Between Observations, Beliefs, and Actions 

**Authors**: Jan Sobotka, Mustafa O. Karabag, Ufuk Topcu  

**Link**: [PDF](https://arxiv.org/pdf/2605.00226)  

**Abstract**: Large language models (LLMs) are increasingly tasked with strategic decision-making under incomplete information, such as in negotiation and policymaking. While LLMs can excel at many such tasks, they also fail in ways that are poorly understood. We shed light on these failures by uncovering two fundamental gaps in the internal mechanisms underlying the decision-making of LLMs in incomplete-information games, supported by experiments with open-weight models Llama 3.1, Qwen3, and gpt-oss. First, an observation-belief gap: LLMs encode internal beliefs about latent game states that are substantially more accurate than their own verbal reports, yet these beliefs are brittle. In particular, the belief accuracy degrades with multi-hop reasoning, exhibits primacy and recency biases, and drifts away from Bayesian coherence over extended interactions. Second, a belief-action gap: The implicit conversion of internal beliefs into actions is weaker than that of the beliefs externalized in the prompt, yet neither belief-conditioning consistently achieves higher game payoffs. These results show how analyzing LLMs' internal processes can expose systematic vulnerabilities that warrant caution before deploying LLMs in strategic domains without robust guardrails. 

---
# RSAT: Structured Attribution Makes Small Language Models Faithful Table Reasoners 

**Authors**: Jugal Gajjar, Kamalasankari Subramaniakuppusamy  

**Link**: [PDF](https://arxiv.org/pdf/2605.00199)  

**Abstract**: When a language model answers a table question, users have no way to verify which cells informed which reasoning steps. We introduce RSAT, a method that trains small language models (SLMs, 1-8B) to produce step-by-step reasoning with cell-level citations grounded in table evidence. Phase 1 (SFT) teaches a structured JSON output format from verified reasoning traces. Phase 2 (GRPO) optimizes a composite reward centered on NLI-based faithfulness, alongside citation validity and parsimony. Across six models from two families-Qwen 2.5 (1.5B/3B/7B) and Llama 3 (1B/3B/8B)-RSAT improves faithfulness 3.7$\times$ over SFT alone (0.224$\rightarrow$0.826), with near-perfect citation validity (0.992). Post-hoc attribution collapses below 13% format success, confirming that attribution must be integrated into reasoning, not retrofitted. Ablations show the faithfulness reward is essential: removing it drops faithfulness from 0.97 to 0.03. 

---
# The $\textit{Silicon Society}$ Cookbook: Design Space of LLM-based Social Simulations 

**Authors**: Aurélien Bück-Kaeffer, Sneheel Sarangi, Maximilian Puelma Touzel, Reihaneh Rabbany, Zachary Yang, Jean-François Godbout  

**Link**: [PDF](https://arxiv.org/pdf/2605.00197)  

**Abstract**: Studies attempting to simulate human behavior with $\textit{Silicon Societies}$ grow in numbers while LLM-only social networks have started appearing outside of controlled settings. However, the design space of these networks remains under-studied, which contributes to a gap in validating model realism. To enable future works to make more informed design decisions, we perform a systematic analysis of the consequences and interactions of key design choices in simulated social networks, including the choice of base model used to model individual agents, and how they are connected to each other. Using surveys as a proxy for agent opinions, our findings suggest that the geometry of the design space is non-trivial, with some parameters behaving in additive ways while others display more complex interactions. In particular, the choice of the base LLM is the most important variable impacting the simulation outcomes. 

---
# Fair Dataset Distillation via Cross-Group Barycenter Alignment 

**Authors**: Mohammad Hossein Moslemi, Nima Hosseini Dashtbayaz, Zhimin Mei, Boyu Wang, Bissan Ghaddar  

**Link**: [PDF](https://arxiv.org/pdf/2605.00185)  

**Abstract**: Dataset Distillation aims to compress a large dataset into a small synthetic one while maintaining predictive performance. We show that as different demographic groups exhibit distinct predictive patterns, the distillation process struggles to simultaneously preserve informative signals for all subgroups, regardless of whether group sizes are mildly or severely imbalanced. Consequently, models trained on distilled data can experience substantial performance drops for certain subgroups, leading to fairness gaps. Crucially, these gaps do not disappear by merely correcting group imbalance, since they stem from fundamental mismatches in subgroup predictive patterns rather than from sample-size disparities alone. We therefore formally analyze the interaction between these two sources of bias and cast the solution as identifying a group-imbalance-agnostic barycenter of the predictive information that induces similar representations across all subgroups. By distilling toward this shared aggregate representation, we show that group fairness concerns can be reduced. Our approach is compatible with existing distillation methods, and empirical results show that it substantially reduces bias introduced by dataset distillation. 

---
# Smart Profit-Aware Crop Advisory System: Kisan AI 

**Authors**: Debasis Dwibedy, Avyay Nishtala, Pranathi Mukku, D Snehaja  

**Link**: [PDF](https://arxiv.org/pdf/2605.00133)  

**Abstract**: Modern crop advisory systems exhibit a critical limitation termed \textit{economic blindness}. These systems primarily optimize for biological yield, often overlooking market price, which can lead farmers toward agronomically sound yet financially unviable decisions. In this paper, we develop Kisan AI, a smart profit-aware crop advisory system that resolves the above-mentioned limitation through a research-driven, full-stack application. We train the Random Forest(RF) classifier model on a nine-feature benchmark dataset, the standard seven agronomic attributes augmented with a \textit{market\_price} variable, and evaluated against eight baseline models, considering the evaluation matrices, such as, accuracy, precision, recall, F1-score, and Log Loss. The RF model achieves the highest accuracy of 99.3\% and the lowest Log Loss, confirming that the inclusion of market price as a predictive feature is both valid and impactful. We then implement the RF model within a multilingual progressive Web App alongside a Facebook Prophet six-month price forecasting engine and a MobileNetV2 disease detection module. A nine-language AI chatbot powered by the Anthropic Claude API unifies all modules into a single, mobile-installable platform accessible to farmers across India. 

---
# Cultural Benchmarking of LLMs in Standard and Dialectal Arabic Dialogues 

**Authors**: Muhammad Dehan Al Kautsar, Saeed Almheiri, Momina Ahsan, Bilal Elbouardi, Younes Samih, Sarfraz Ahmad, Amr Keleg, Omar El Herraoui, Kareem Elzeky, Abed Alhakim Freihat, Mohamed Anwar, Zhuohan Xie, Junhong Liang, Mohammad Rustom Al Nasar, Preslav Nakov, Fajri Koto  

**Link**: [PDF](https://arxiv.org/pdf/2605.00119)  

**Abstract**: There is a significant gap in evaluating cultural reasoning in LLMs using conversational datasets that capture culturally rich and dialectal contexts. Most Arabic benchmarks focus on short text snippets in Modern Standard Arabic (MSA), overlooking the cultural nuances that naturally arise in dialogues. To address this gap, we introduce ArabCulture-Dialogue, a culturally grounded conversational dataset covering 13 Arabic-speaking countries, in both MSA and each country's respective dialect, spanning 12 daily-life topics and 54 fine-grained subtopics. We utilize the dataset to form three benchmarking tasks: (i) multiple-choice cultural reasoning, (ii) machine translation between MSA and dialects, and (iii) dialect-steering generation. Our experiments indicate that the performance gap between MSA and Arabic dialects still exists, whereby the models perform worse on all three tasks in the dialectal setup, compared to the MSA one. 

---
# ViLegalNLI: Natural Language Inference for Vietnamese Legal Texts 

**Authors**: Nhung Thi-Hong Duong, Mai Ngoc Ho, Tin Van Huynh, Kiet Van Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2605.00116)  

**Abstract**: In this article, we introduce ViLegalNLI, the first large-scale Vietnamese Natural Language Inference (NLI) dataset specifically constructed for the legal domain. The dataset consists of 42,012 premise-hypothesis pairs derived from official statutory documents and annotated with binary inference labels (Entailment and Non-entailment). It covers multiple legal domains and reflects realistic legal reasoning scenarios characterized by structured logic, conditional clauses, and domain-specific terminology. To construct ViLegalNLI, we propose a semi-automatic data generation framework that integrates large language models for controlled hypothesis generation and systematic quality validation procedures. The framework incorporates artifact mitigation strategies and cross-model validation to improve annotation reliability and ensure legal consistency. The resulting dataset captures diverse reasoning patterns, including paraphrasing, logical implication, and legally invalid inferences, thereby providing a comprehensive benchmark for Vietnamese legal inference tasks. We conduct extensive experiments on the ViLegalNLI using multilingual models, Vietnamese-specific pretrained language models, and instruction-tuned large language models. The results show that few-shot LLM configurations consistently achieve superior performance, while performance is significantly influenced by hypothesis length, lexical overlap, and reasoning complexity. Cross-domain evaluations further reveal the challenges of generalizing legal inference across distinct legal fields. Overall, ViLegalNLI establishes a foundational benchmark for Vietnamese legal NLI and supports future research in legal reasoning, statutory text understanding, and the development of reliable AI systems for legal analysis and decision support. The dataset is publicly available for research purposes. 

---
# How Frontier LLMs Adapt to Neurodivergence Context: A Measurement Framework for Surface vs. Structural Change in System-Prompted Responses 

**Authors**: Ishan Gupta, Pavlo Buryi  

**Link**: [PDF](https://arxiv.org/pdf/2605.00113)  

**Abstract**: We examine if frontier chat-based large language models (LLMs) adjust their outputs based on neurodivergence (ND) context in system prompts and describe the nature of these adjustments. Specifically, we propose NDBench, a 576-output benchmark involving two frontier models, three system prompt types (baseline, ND-profile assertion, and ND-profile assertion with explicit instructions for adjustments), four canonical ND profiles, and 24 prompts across four categories, one of which involves an adversarial masking strategy.
Four trends emerge consistently from our findings. First, LLMs show significant adaptation under ND context, where fully instructed conditions yield lengthier and more structured outputs, characterized by higher token counts, more headings, and more granular steps (p < 10^-8, Holm-corrected). Second, such adaptation is largely structural in nature: although list density does not change much, there is a marked rise in the frequency of headings and per-step detail. Third, ND persona assertion alone fails to suppress potentially harmful tendencies, as masking-reinforcement decreases only in explicitly instructed cases (36-44% reduction); the reduction rate barely changes in persona assertion conditions.
Moreover, reliability analysis of LLM-based harm assessment reveals that only two out of the six dimensions (masking and reinforcement, validation quality) exceed the pre-defined inter-judge agreement criterion (alpha >= 0.67) and thus can be considered primary results.
NDBench is made publicly available along with its prompts, outputs, code, and other resources, forming a reproducible framework for auditing future LLMs' adaptation to ND awareness. 

---
# AIDA-ReID: Adaptive Intermediate Domain Adaptation for Generalizable and Source-Free Person Re-Identification 

**Authors**: Sundas Iqbal, Qing Tian, Danish Ali, Jianping Gou, Weihua Oue  

**Link**: [PDF](https://arxiv.org/pdf/2605.00111)  

**Abstract**: Person re-identification (Re-ID) aims to match images of the same individual across non-overlapping camera views and remains challenging due to domain shifts caused by variations in illumination, background, camera characteristics, and population distributions. Although supervised models perform well under matched training and testing conditions, their performance degrades significantly when deployed in unseen environments. Existing intermediate domain approaches such as IDM and IDM++ alleviate this gap by constructing bridge feature distributions between domains; however, they rely on fixed mixing strategies and joint source-target access, limiting their applicability to multi-source and source-free settings. To address these limitations, this paper proposes Adaptive Intermediate Domain Adaptation (AIDA), also referred to as Source-Free Multi-Source Intermediate Domain Adaptation (SF-MIDA). The proposed framework treats intermediate-domain learning as a dynamically regulated process, where feature mixing and regularization strength are adaptively controlled using feedback signals derived from model uncertainty and training stability. A multi-source intermediate domain generator synthesizes diverse intermediate representations, while a pseudo-mirror regularization strategy preserves identity consistency under domain perturbations. Extensive experiments across domain generalization and source-free settings demonstrate the effectiveness of the proposed framework. 

---
# DeGenTWeb: A First Look at LLM-dominant Websites 

**Authors**: Sichang Steven He, Calvin Ardi, Ramesh Govindan, Harsha V. Madhyastha  

**Link**: [PDF](https://arxiv.org/pdf/2605.00087)  

**Abstract**: Many recent news reports have claimed that content generated by large language models (LLMs) is taking over the web. However, these claims are typically not based on a representative sample of the web and the methodology underlying them is often opaque. Moreover, when aiming to minimize the chances of falsely attributing human-authored content to LLMs, we find that detectors of LLM-generated text perform much worse than advertised. Consequently, we lack an understanding of the true prevalence and characteristics of LLM content on the web.
We describe DeGenTWeb which systematically identifies LLM-dominant websites: sites whose content has been generated using LLMs with little human input. We show how to adapt detectors of LLM-generated text for use on web pages, and how to aggregate detection results from multiple pages on a site for accurate site-level categorization. Using DeGenTWeb, we find that LLM-dominant sites are highly prevalent both in data from Common Crawl and in Bing's search results, and that this share is growing over time. We also show that continuing to accurately identify such sites appears challenging given the capabilities of the latest LLMs. 

---
# NorBERTo: A ModernBERT Model Trained for Portuguese with 331 Billion Tokens Corpus 

**Authors**: Enzo S. N. Silva, Pablo B. Costa, Raphael C. Vlasman, Rosimeire P. Costa, Henrique L. P. Silva, Lucas F. A. O. Pellicer, Guilherme Rinaldo, Renato A. Almeida, Darian S. R. Rabbani, Cinthya O. Oestreich, Vinicius F. Caridá  

**Link**: [PDF](https://arxiv.org/pdf/2605.00086)  

**Abstract**: High-quality corpora are essential for advancing Natural Language Processing (NLP) in Portuguese. Building on previous encoder-only models such as BERTimbau and Albertina PT-BR, we introduce NorBERTo, a modern encoder based on the ModernBERT architecture, featuring long-context support and efficient attention mechanisms. NorBERTo is trained on Aurora-PT, a newly curated Brazilian Portuguese corpus comprising 331 billion GPT-2 tokens collected from diverse web sources and existing multilingual datasets. We systematically benchmark NorBERTo against Strong baselines on semantic similarity, textual entailment and classification tasks using standardized datasets such as ASSIN 2 and PLUE. On PLUE, NorBERTo-large achieves the best results among the encoder models we evaluated, notably reaching 0.9191 F1 on MRPC and 0.7689 accuracy on RTE. On ASSIN 2, NorBERTo-large attains the highest entailment F1 (~0.904) among all encoders considered, although Albertina-900M and BERTimbau-large still hold an advantage. To the best of our knowledge, Aurora-PT is currently the largest openly available monolingual Portuguese corpus, surpassing previous resources. NorBERTo provides a modern, mid-sized encoder designed for realistic deployment scenarios: it is straight-forward to fine-tune, efficient to serve, and well suited as a backbone for retrieval-augmented generation and other downstream Portuguese NLP systems. 

---
# Hyperspherical Forward-Forward with Prototypical Representations 

**Authors**: Shalini Sarode, Brian Moser, Joachim Folz, Federico Raue, Tobias Nauen, Stanislav Frolov, Andreas Dengel  

**Link**: [PDF](https://arxiv.org/pdf/2605.00082)  

**Abstract**: The Forward-Forward (FF) algorithm presents a compelling, bio-inspired alternative to backpropagation. However, while efficient in training, it has a computationally prohibitive inference process that requires a separate forward pass for every class that is evaluated. In this work, we introduce the Hyperspherical Forward-Forward (HFF), a novel reformulation that resolves this critical bottleneck. Our core innovation is to reframe the local objective of each layer from a binary goodness-of-fit task to a direct multi-class classification problem within a hyperspherical feature space. We achieve this by learning a set of class-specific, unit-norm prototypes that act as geometric anchors and implicit negatives. This architectural innovation preserves the benefits of local training while enabling weight update and inference in a single forward pass, making it >40x faster than the original FF algorithm. Our method is simple to implement, scales effectively to modern convolutional architectures, and achieves superior accuracy on standard image classification benchmarks, closing the gap with backpropagation. Most notably, we are among the first greedy local-learning methods to report over 25% top-1 accuracy on ImageNet-1k, and 65.96% with transfer learning. 

---
# CRC-Screen: Certified DNA-Synthesis Hazard Screening Under Taxonomic Shift 

**Authors**: Najmul Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2605.00074)  

**Abstract**: DNA-synthesis providers screen incoming orders by searching the requested sequence against curated hazard lists. We show that this baseline collapses to a 100% false-flag rate when the hazardous sequence comes from a taxonomic family absent from the reference set: under Conformal Risk Control's certified miss-rate constraint, a low-discrimination signal forces the threshold below the entire test-benign mass. We compose three signals derived from a synthesis order's public annotation: $k$-mer Jaccard similarity to known toxins, the trimmed-mean score of a five-LLM judge panel, and cosine similarity to clustered embedding centroids. Fused under a monotone logistic aggregator and calibrated by Conformal Risk Control, the resulting screener certifies $\mathbb{E}[\mathrm{FNR}] \le \alpha$. Across ten leave-one-taxonomic-family-out folds at $\alpha=0.05$ on UniProt KW-0800 reviewed toxins, the calibrated screener achieves 0% test miss rate on every fold and 0% test false-flag rate on nine of ten folds. The bound's finite-sample slack $1/(n_{\mathrm{cal}}+1)$ caps the certifiable miss rate at 1.77% on our 200-hazard subsample; reaching procurement-grade $\alpha=10^{-3}$ requires an $18\times$ larger calibration set, which the full reviewed UniProt KW-0800 corpus is large enough to deliver. The binding constraint on certifiable DNA-synthesis screening is calibration data, not algorithms. Code: this https URL 

---
# XekRung Technical Report 

**Authors**: Jiutian Zeng, Junjie Li, Chengwei Dai, Jie Liang, Zhaoyu Hu, Yiliang Zhang, Ziang Weng, Longtao Huang, Dongjie Zhang, Libin Dong, Yang Ge, Yuanda Wang, Kaiwen Lv Kacuila, Bingyu Zhu, Jing Wang, Jin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.00072)  

**Abstract**: We present XekRung, a frontier large language model for cybersecurity, designed to provide comprehensive security capabilities. To achieve this, we develop diverse data synthesis pipelines tailored to the cybersecurity domain, enabling the scalable construction of high-quality training data and providing a strong foundation for cybersecurity knowledge and understanding. Building on this foundation, we establish a complete training pipeline spanning continued pre-training (CPT), supervised fine-tuning (SFT), and reinforcement learning (RL) to further extend the model's capabilities. We further introduce a multi-dimensional evaluation system to guide the iterative improvement of both domain-specific and general-purpose abilities. Extensive experiments demonstrate that XekRung achieves state-of-the-art performance on cybersecurity-specific benchmarks among models of the same scale, while maintaining strong performance on general benchmarks. 

---
# Compliance-Aware Agentic Payments on Stablecoin Rails 

**Authors**: Kenneth See, Xue Wen Tan  

**Link**: [PDF](https://arxiv.org/pdf/2605.00071)  

**Abstract**: Agentic payment systems extend delegated action to financial transfers, but scaling them on stablecoin rails in regulated settings requires safeguards that remain effective when humans are not continuously in the loop. We present a compliance-aware architecture that combines x402-style, signature-based payment authorisation and relayed execution with programmable compliance embedded as an on-chain guardrail via a policy wrapper and policy manager coordinating modular checks. By enforcing compliance at the point of execution, rather than as a separate off-chain workflow, the approach preserves low-friction settlement when conditions are satisfied, records transaction-linked on-chain attestations, and supports structured resolution when requirements are pending. 

---
# Human-in-the-Loop Meta Bayesian Optimization for Fusion Energy and Scientific Applications 

**Authors**: Ricardo Luna Gutierrez, Sahand Ghorbanpour, Ejaz Rahman, Varchas Gopalaswamy, Riccardo Betti, Vineet Gundecha, Aarne Lees, Soumyendu Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2605.00068)  

**Abstract**: Inertial Confinement Fusion (ICF) holds transformative promise for sustainable, near-limitless clean energy, yet remains constrained by prohibitively high costs and limited experimental opportunities. This paper presents Human-in-the-Loop Meta Bayesian Optimization (HL-MBO), a framework that integrates expert knowledge with few-shot, uncertainty-aware machine learning to accelerate discovery in data-scarce, high-stakes scientific domains. HL-MBO introduces a meta-learned surrogate model with an expert-informed acquisition function to recommend candidate experiments. To foster trust and enable informed decisions, HL-MBO also provides interpretable explanations of its suggestions. We show HL-MBO outperforms current BO methods on ICF energy yield optimization, as well as benchmarks in molecular optimization and critical temperature maximization for superconducting materials. 

---
# A Survey of Reasoning-Intensive Retrieval: Progress and Challenges 

**Authors**: Yiyang Wei, Tingyu Song, Siyue Zhang, Yilun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.00063)  

**Abstract**: Reasoning-Intensive Retrieval (RIR) targets retrieval settings where relevance is mediated by latent inferential links between a query and supporting evidence, rather than semantic similarity. Motivated by the emergent reasoning abilities of Large Language Models (LLMs), recent work integrates these capabilities into the IR field, spanning the entire pipeline from benchmarks to retrievers and rerankers. Despite this progress, the field lacks a systematic framework to organize current efforts and articulate a clear path forward. To provide a clear roadmap for this rapidly growing yet fragmented area, this survey (1) systematizes existing RIR benchmarks by knowledge domains and modalities, providing a detailed analysis of the current landscape; (2) introduces a structured taxonomy that categorizes methods based on where and how reasoning is integrated into the retrieval pipeline, alongside an analysis of their trade-offs and practical applications; and (3) summarizes challenges and future directions to guide research in this evolving field. 

---
# Dynamic-TD3: A Novel Algorithm for UAV Path Planning with Dynamic Obstacle Trajectory Prediction 

**Authors**: Wentao Chen, Jingtang Chen, Mingjian Fu, Tiantian Li, Youfeng Su, Wenxi Liu, Yuanlong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.00059)  

**Abstract**: Deep reinforcement learning (DRL) finds extensive application in autonomous drone navigation within complex, high-risk environments. However, its practical deployment faces a safety-exploration dilemma: soft penalty mechanisms encourage risky trial-and-error, while most constraint-based methods suffer degraded performance under sensor noise and intent uncertainty. We propose Dynamic-TD3, a physically enhanced framework that enforces strict safety constraints while maintaining maneuverability by modeling navigation as a Constrained Markov Decision Process (CMDP). This framework integrates an Adaptive Trajectory Relational Evolution Mechanism (ATREM) to capture long-range intentions and employs a Physically Aware Gated Kalman Filter (PAG-KF) to mitigate non-stationary observation noise. The resulting state representation drives a dual-criterion policy that balances mission efficiency against hard safety constraints via Lagrangian relaxation. In experiments with aggressive dynamic threats, this approach demonstrates superior collision avoidance performance, reduced energy consumption, and smoother flight trajectories. 

---
# Smart Ensemble Learning Framework for Predicting Groundwater Heavy Metal Pollution 

**Authors**: T. Ansah-Narh, G. Y. Afrifa, J. B. Tandoh, K. Asare, M. Addi, K. E. Yorke, D. M. A. Akpoley, K. Aidoo, S. K. Fosuhene  

**Link**: [PDF](https://arxiv.org/pdf/2605.00056)  

**Abstract**: Groundwater in the Densu Basin is increasingly threatened by heavy metal contamination, but conventional methods fail to capture the statistical complexity and spatial heterogeneity of pollution indicators. A key challenge is modelling the Heavy Metal Pollution Index (HPI), which is typically skewed and affected by correlated contaminants, leading to biased predictions without transformation. This study develops a predictive framework integrating response transformations with nested cross-validated ensemble machine learning. Three transformations (raw, log, and Gaussian copula) were applied to HPI and evaluated across six learners: support vector regression (SVM), $k$-nearest neighbours (k-NN), CART, Elastic Net, kernel ridge regression, and a stacked Lasso ensemble. Raw-scale models produced deceptively high fits (Elastic Net and stacked ensemble $R^2 \approx 1.0$), suggesting over-optimism. The log transformation stabilised variance (SVM: $R^2 = 0.93$, RMSE $= 0.18$; k-NN: $R^2 = 0.92$, RMSE $= 0.20$). The Gaussian copula gave the most reliable results: stacked ensemble $R^2 = 0.96$ (RMSE $= 0.19$), with other learners maintaining high accuracy. Copula-based models improved residuals and produced spatially plausible maps. DBSCAN clustering revealed Fe and Mn as primary HPI contributors, consistent with regional hydrogeochemistry. Limitations include reliance on random (not spatial) cross-validation and basin-specific scope. Future work should explore spatial validation and other geological settings. Overall, distribution-aware ensembles with clustering diagnostics offer robust, interpretable assessments of groundwater contamination. 

---
# Ambient Persuasion in a Deployed AI Agent: Unauthorized Escalation Following Routine Non-Adversarial Content Exposure 

**Authors**: Diego F. Cuadros, Abdoul-Aziz Maiga  

**Link**: [PDF](https://arxiv.org/pdf/2605.00055)  

**Abstract**: We report a safety incident in a deployed multi-agent research system in which a primary AI agent installed 107 unauthorized software components, overwrote a system registry, overrode a prior negative decision from an oversight agent, and escalated through increasingly privileged operations up to an attempted system administrator command. The incident was preceded not by an adversarial attack but by routine content: a forwarded technology article written for human developers and shared by the principal investigator for discussion. The agent operated in a permissive environment, with unrestricted shell access, soft behavioral guidelines containing genuinely conflicting instructions, and no machine-enforced installation policy, and had recommended installing the same tool six hours earlier before being told to stand down. We analyze the behavioral cascade, the control boundaries that failed, and the limitations of multi-agent oversight in detecting and remediating the damage. We use directive weighting error as a descriptive interpretation of the observed failure and ambient persuasion as a provisional analytic label for the broader trigger configuration of non-adversarial environmental content preceding unauthorized agent action. The case highlights ethical and governance implications for deployed agent systems: ambiguous conversational cues are insufficient authorization for consequential actions, prior refusals must persist as enforceable constraints rather than message-level reminders, and oversight mechanisms require systematic post-incident auditing in addition to routine monitoring. 

---
# SiriusHelper: An LLM Agent-Based Operations Assistant for Big Data Platforms 

**Authors**: Yu Shen, Shiyang Liu, Qihang He, Yihang Cheng, Haining Xie, Zhiming He, Huahua Fan, Xianzhi Tan, Teng Ma, Shaoquan Zhang, Danqing Huang, Fan Jiang, Yang Li, Chongqing Zhao, Peng Chen, Jie Jiang, Bin Cui  

**Link**: [PDF](https://arxiv.org/pdf/2605.00043)  

**Abstract**: Big data platforms are widely used in modern enterprises, and an in-production intelligent assistant is increasingly important to help users quickly find actionable guidance and reduce operational burden. While recent LLM+RAG assistants provide a natural interface, they face practical challenges in real deployments: limited scenario coverage across both general consultation and domain-specific troubleshooting workflows, inefficient knowledge access due to inadequate multi-hop retrieval and flat knowledge organization, and high maintenance cost because escalated tickets are unstructured and hard to convert into assistant improvements and reusable SOPs.
In this paper, we present SiriusHelper, a deployed intelligent assistant for big data platforms. SiriusHelper serves as a unified online assistant that automatically identifies user intent and routes queries to the right handling path, including dedicated expert workflows for specialized scenarios (e.g., SQL execution diagnosis). To support complex troubleshooting, SiriusHelper combines a DeepSearch-driven mechanism with a priority-based hierarchical knowledge base to enable multi-hop retrieval without context overload, thus improving answer reliability and latency. To reduce expert overhead, SiriusHelper further introduces automated ticket understanding and SOP distillation: it diagnoses the assistant failure reason (e.g., missing knowledge or wrong routing) and extracts domain-specific SOPs to continuously enrich the knowledge base. Experiments and online deployment on Tencent Big Data platform show that SiriusHelper outperforms representative alternatives and reduces online ticket volume by 20.8\%. 

---
# Sure About That Line? Approaching Confidence-Based, Real-Time Line Assignment in Reading Gaze Data 

**Authors**: Franziska Kaltenberger, Wei-Ling Chen, Enkeleda Thaqi, Enkelejda Kasneci  

**Link**: [PDF](https://arxiv.org/pdf/2605.00033)  

**Abstract**: Remote and webcam-based eye tracking in multi-line reading suffers from various noise factors and layout ambiguity, precisely where real-time reading support needs reliable, per-fixation line assignment. Prior work largely addresses this challenge post hoc or by restricting behavior (e.g., disallowing re-reading), undermining interactive use. We propose CONF-LA (Confidence-score-based Online Fixation-to-Line Assignment), a principled, low-latency approach that integrates knowledge about reading behavior and Gaussian line likelihoods over fixations to compute a posterior-line-score and defers assignments when uncertainty is high. Evaluated on existing open-source data, CONF-LA demonstrates stable performance in post hoc analysis and closes the online-offline gap (1-2 %) with a mean per-fixation latency of 0.348 ms. Our approach exhibits particular invariance toward regressions, yielding significant improvement in ad hoc median accuracies on children data (approx. 95 %) over all tested algorithms. We encourage further research in this direction and discuss possibilities for future development. 

---
# Putting HUMANS first: Efficient LAM Evaluation with Human Preference Alignment 

**Authors**: Woody Haosheng Gan, William Held, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.00022)  

**Abstract**: The rapid proliferation of large audio models (LAMs) demands efficient approaches for model comparison, yet comprehensive benchmarks are costly. To fill this gap, we investigate whether minimal subsets can reliably evaluate LAMs while reducing costs and data redundancy. Analyzing 10 subset selection methods with 18 audio models across 40 tasks covering major LAM evaluation dimensions, we show that subsets of just 50 examples (0.3% of data) can achieve over 0.93 Pearson correlation with full benchmark scores. To understand how well these scores align with what practitioners ultimately care about, user satisfaction, we collect 776 human preference ratings from realistic voice assistant conversations, finding that both subsets and full benchmark achieve only 0.85 correlation with human. To better predict preferences, we trained regression models on these selected subsets, achieving 0.98 correlation -- outperforming regression models trained on both random subsets and the full benchmark. This demonstrates that in regression modeling, well-curated subsets outpredict the full benchmark, showing quality over quantity. We open-source these regression-weighted subsets as the HUMANS benchmark, an efficient proxy for LAM evaluation that captures both benchmark performance and user preferences. 

---
# AirFM-DDA: Air-Interface Foundation Model in the Delay-Doppler-Angle Domain for AI-Native 6G 

**Authors**: Kejia Bian, Meixia Tao, Jianhua Mo, Zhiyong Chen, Leyan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.00020)  

**Abstract**: The success of large foundation models is catalyzing a new paradigm for AI-native 6G network design: wireless foundation models for physical layer design. However, existing models often operate on channel state information (CSI) in the space-time-frequency (STF) domain, where distinct multipath components are inherently superimposed and structurally entangled. This hinders the learning of universal channel representation. Meanwhile, their reliance on global attention mechanisms incurs prohibitive computational overhead. In this paper, we propose AirFM-DDA, an Air-interface Foundation Model operating in the Delay-Doppler-Angle (DDA) domain for physicallayer tasks. Specifically, AirFM-DDA reparameterizes CSI from the STF domain into the DDA domain to explicitly resolve multipath components along physically meaningful axes. It employs a window-based attention module augmented with framestructure-aware positional encoding (FS-PE). This window-based attention aligns with locally clustered multipath dependencies while avoiding quadratic-complexity global attention, and FS-PE injects frame-structure priors into network. Extensive experiments demonstrate that AirFM-DDA achieves superior zero-shot generalization across unseen scenarios and datasets, consistently outperforming the baselines on channel prediction and estimation tasks. Compared to the global attention, its window-based attention reduces training and inference costs by nearly an order of magnitude. Moreover, AirFM-DDA maintains robustness under high mobility, large delay spreads, severe noise, and extreme aliasing conditions. 

---
# TimeRFT: Stimulating Generalizable Time Series Forecasting for TSFMs via Reinforcement Finetuning 

**Authors**: Siyang Li, Yize Chen, Zijie Zhu, Yuxin Pan, Yan Guo, Ming Huang, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2605.00015)  

**Abstract**: Time Series Foundation Models (TSFMs) advance generalization and data efficiency in time series forecasting by unified large-scale pretraining. But TSFMs remain lacking when adapting to specific downstream forecasting tasks for two reasons. First, the non-stationary and uncertain nature of time series data lead to inevitable temporal distribution shifts between historical training and future testing data, while current Supervised FineTuning (SFT)-based methods are prone to overfitting and may degrade generalization. Second, training data availability varies across forecasting tasks, requiring TSFMs to generalize well under diverse data regimes. To address these challenges, we introduce the Time series Reinforcement Finetuning (TimeRFT) paradigm for TSFM downstream adaptation, which consists of two task-specific training recipes: i) A forecasting quality-based temporal reward mechanism that conducts a multi-faceted evaluation of the contribution of each prediction step to overall forecasting accuracy. ii) A forecasting difficulty-based data selection strategy to identify time series samples with generalizable predictive patterns and informative training signals. Extensive experiments demonstrate TimeRFT can consistently outperform SFT-based adaptation methods across various real-world forecasting tasks and training data regimes, enhancing prediction accuracy and generalization against unforeseen distribution shifts. 

---
# Exploring LLM biases to manipulate AI search overview 

**Authors**: Roman Smirnov  

**Link**: [PDF](https://arxiv.org/pdf/2605.00012)  

**Abstract**: Modern large language models (LLMs) are used in many business applications in general, and specifically in web search systems and applications that generate overviews of search results - LLM Overview systems. Such systems are using an LLM to select most relevant sources from search results and generate an answer to the user's query. It is known from many studies that LLMs have different biases, in LLM Overview application both the source selection and answer generation stages may be affected by the biases of LLMs (here we are focusing mainly on the selection stage). This research is focused on investigating the presence of the biases in LLM Overview systems and on biases exploitation to manipulate LLM Overview results. Here we train a small language model using reinforcement learning to rewrite search snippets to increase their likelihood of being preferred by an LLM Overview. Our experimental setup intentionally restricts the policy to operate only on snippets and limits reward-hacking strategies, reflecting realistic constraints of web search environments. The results prove that LLM Overview systems have biases and that reinforcement learning in most of the cases can optimize snippet's content to manipulate LLM Overview results. We also prove that LLM Overview selections are driven by comparative rather than absolute advantages among candidate sources. In addition, we examine safety aspects of LLM Overview manipulation possibilities and show that context poisoning attacks can lead to inaccurate or harmful results. 

---
# FedACT: Concurrent Federated Intelligence across Heterogeneous Data Sources 

**Authors**: Md Sirajul Islam, Isabelle G Chapman, N I Md Ashafuddula, Xu Yuan, Li Chen, Nian-Feng Tzeng, Klara Nahrstedt  

**Link**: [PDF](https://arxiv.org/pdf/2605.00011)  

**Abstract**: Federated Learning (FL) enables collaborative intelligence across decentralized data source devices in a privacy-preserving way. While substantial research attention has been drawn to optimizing the learning process for an individual task, real-world applications increasingly require multiple machine learning tasks simultaneously training their models across a shared pool of devices. Naively applying single-FL optimization techniques in multi-FL systems results in suboptimal system performance, particularly due to device heterogeneity and resource inefficiency. To address such a critical open challenge, we introduce {\em FedACT}, a novel resource heterogeneity-aware device scheduling approach designed to efficiently schedule heterogeneous devices across multiple concurrent FL jobs, with the goal of minimizing their average job completion time (JCT). {\em FedACT} dynamically assigns devices to FL jobs based on an alignment scoring mechanism that evaluates the compatibility between available resources of devices and resource demands of jobs. Additionally, it incorporates participation fairness to ensure balanced contributions from devices across jobs, further enhancing the accuracy levels of learned global models. An optimal scheduling plan is formulated in {\em FedACT} by prioritizing devices with higher alignment scores, while ensuring fair participation across jobs. To evaluate the effectiveness of the proposed scheduling algorithm, we carried out comprehensive experiments using diverse FL jobs and benchmark datasets. Experimental results demonstrate that {\em FedACT} reduces the average JCT by up to 8.3\(\times\) and improves model accuracy by up to 44.5\%, compared to the state-of-the-art baselines. 

---
# Mean-Field Path-Integral Diffusion: From Samples to Interacting Agents 

**Authors**: Michael Chertkov  

**Link**: [PDF](https://arxiv.org/pdf/2605.00007)  

**Abstract**: Independent sample generation is the prevailing paradigm in modern diffusion-based generative models of AI. We ask a different question: can samples \emph{coordinate} through shared population statistics to transport probability mass more efficiently? We introduce Mean-Field Path-Integral Diffusion (MF-PID), a framework in which samples are promoted to interacting agents whose drift depends self-consistently on the evolving population density. The coupling converts distribution matching into a McKean--Vlasov extension of the stochastic optimal transport problem, unifying generative modeling and multi-agent control under the same Hamilton--Jacobi--Bellman/Kolmogorov--Fokker--Planck duality. We identify two analytically tractable regimes: a Linear--Quadratic--Gaussian (LQG) benchmark in which the infinite-dimensional mean-field system reduces to a finite set of Riccati and linear ODEs, and a Gaussian-mixture regime governed by a piecewise-constant protocol that preserves closed-form solvability. For a quadratic interaction potential with schedule $\beta_t$ and zero base drift we prove that the self-consistent MF guidance is the \emph{exact} linear interpolant between initial and target global means -- a result that holds for arbitrary initial and target densities and any $\beta_t$. Applied to demand-response control of energy systems, where agents aggregated into an ensemble are energy consumers (e.g.\ thermal zones within a building), MF-PID achieves 19--24\% reductions in cumulative control energy over independent-agent baselines while matching the prescribed terminal distribution exactly, and reveals how coordination redistributes actuation effort across heterogeneous sub-populations. 

---
# Cloud Is Closer Than It Appears: Revisiting the Tradeoffs of Distributed Real-Time Inference 

**Authors**: Pragya Sharma, Hang Qiu, Mani Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2605.00005)  

**Abstract**: The increasing deployment of deep neural networks (DNNs) in cyber-physical systems (CPS) enhances perception fidelity, but imposes substantial computational demands on execution platforms, posing challenges to real-time control deadlines. Traditional distributed CPS architectures typically favor on-device inference to avoid network variability and contention-induced delays on remote platforms. However, this design choice places significant energy and computational demands on the local hardware. In this work, we revisit the assumption that cloud-based inference is intrinsically unsuitable for latency-sensitive control tasks. We demonstrate that, when provisioned with high-throughput compute resources, cloud platforms can effectively amortize network and queueing delays, enabling them to match or surpass on-device performance for real-time decision-making. Specifically, we develop a formal analytical model that characterizes distributed inference latency as a function of the sensing frequency, platform throughput, network delay, and task-specific safety constraints. We instantiate this model in the context of emergency braking for autonomous driving and validate it through extensive simulations using real-time vehicular dynamics. Our empirical results identify concrete conditions under which cloud-based inference adheres to safety margins more reliably than its on-device counterpart. These findings challenge prevailing design strategies and suggest that the cloud is not merely a feasible option, but often the preferred inference location for distributed CPS architectures. In this light, the cloud is not as distant as traditionally perceived; in fact, it is closer than it appears. 

---
# Models Recall What They Violate: Constraint Adherence in Multi-Turn LLM Ideation 

**Authors**: Garvin Kruthof  

**Link**: [PDF](https://arxiv.org/pdf/2604.28031)  

**Abstract**: When researchers iteratively refine ideas with large language models, do the models preserve fidelity to the original objective? We introduce DriftBench, a benchmark for evaluating constraint adherence in multi-turn LLM-assisted scientific ideation. Across 2,146 scored benchmark runs spanning seven models from five providers (including two open-weight), four interaction conditions, and 38 research briefs from 24 scientific domains, we find that iterative pressure reliably increases structural complexity and often reduces adherence to original constraints. A restatement probe reveals a dissociation between declarative recall and behavioral adherence, as models accurately restate constraints they simultaneously violate. The knows-but-violates (KBV) rate, measuring constraint non-compliance despite preserved recall, ranges from 8% to 99% across models. Structured checkpointing partially reduces KBV rates but does not close the dissociation, and complexity inflation persists. Human validation against blind raters confirms that the LLM judge under-detects constraint violations, making reported constraint adherence scores conservative. Sensitivity analyses confirm the findings are robust to temperature (0.7 vs.\ 1.0) and pressure type (novelty vs.\ rigor). We release all briefs, prompts, rubrics, transcripts, and scores as an open benchmark. 

---
