# Semantic Invariance in Agentic AI 

**Authors**: I. de Zarzà, J. de Curtò, Jordi Cabot, Pietro Manzoni, Carlos T. Calafate  

**Link**: [PDF](https://arxiv.org/pdf/2603.13173)  

**Abstract**: Large Language Models (LLMs) increasingly serve as autonomous reasoning agents in decision support, scientific problem-solving, and multi-agent coordination systems. However, deploying LLM agents in consequential applications requires assurance that their reasoning remains stable under semantically equivalent input variations, a property we term semantic this http URL benchmark evaluations, which assess accuracy on fixed, canonical problem formulations, fail to capture this critical reliability dimension. To address this shortcoming, in this paper we present a metamorphic testing framework for systematically assessing the robustness of LLM reasoning agents, applying eight semantic-preserving transformations (identity, paraphrase, fact reordering, expansion, contraction, academic context, business context, and contrastive formulation) across seven foundation models spanning four distinct architectural families: Hermes (70B, 405B), Qwen3 (30B-A3B, 235B-A22B), DeepSeek-R1, and gpt-oss (20B, 120B). Our evaluation encompasses 19 multi-step reasoning problems across eight scientific domains. The results reveal that model scale does not predict robustness: the smaller Qwen3-30B-A3B achieves the highest stability (79.6% invariant responses, semantic similarity 0.91), while larger models exhibit greater fragility. 

---
# Developing and evaluating a chatbot to support maternal health care 

**Authors**: Smriti Jha, Vidhi Jain, Jianyu Xu, Grace Liu, Sowmya Ramesh, Jitender Nagpal, Gretchen Chapman, Benjamin Bellows, Siddhartha Goyal, Aarti Singh, Bryan Wilder  

**Link**: [PDF](https://arxiv.org/pdf/2603.13168)  

**Abstract**: The ability to provide trustworthy maternal health information using phone-based chatbots can have a significant impact, particularly in low-resource settings where users have low health literacy and limited access to care. However, deploying such systems is technically challenging: user queries are short, underspecified, and code-mixed across languages, answers require regional context-specific grounding, and partial or missing symptom context makes safe routing decisions difficult.
We present a chatbot for maternal health in India developed through a partnership between academic researchers, a health tech company, a public health nonprofit, and a hospital. The system combines (1) stage-aware triage, routing high-risk queries to expert templates, (2) hybrid retrieval over curated maternal/newborn guidelines, and (3) evidence-conditioned generation from an LLM. Our core contribution is an evaluation workflow for high-stakes deployment under limited expert supervision. Targeting both component-level and end-to-end testing, we introduce: (i) a labeled triage benchmark (N=150) achieving 86.7% emergency recall, explicitly reporting the missed-emergency vs. over-escalation trade-off; (ii) a synthetic multi-evidence retrieval benchmark (N=100) with chunk-level evidence labels; (iii) LLM-as-judge comparison on real queries (N=781) using clinician-codesigned criteria; and (iv) expert validation. Our findings show that trustworthy medical assistants in multilingual, noisy settings require defense-in-depth design paired with multi-method evaluation, rather than any single model and evaluation method choice. 

---
# When Right Meets Wrong: Bilateral Context Conditioning with Reward-Confidence Correction for GRPO 

**Authors**: Yu Li, Tian Lan, Zhengling Qi  

**Link**: [PDF](https://arxiv.org/pdf/2603.13134)  

**Abstract**: Group Relative Policy Optimization (GRPO) has emerged as an effective method for training reasoning models. While it computes advantages based on group mean, GRPO treats each output as an independent sample during the optimization and overlooks a vital structural signal: the natural contrast between correct and incorrect solutions within the same group, thus ignoring the rich, comparative data that could be leveraged by explicitly pitting successful reasoning traces against failed ones. To capitalize on this, we present a contrastive reformulation of GRPO, showing that the GRPO objective implicitly maximizes the margin between the policy ratios of correct and incorrect samples. Building on this insight, we propose Bilateral Context Conditioning (BICC), a mechanism that allows the model to cross-reference successful and failed reasoning traces during the optimization, enabling a direct information flow across samples. We further introduce Reward-Confidence Correction (RCC) to stabilize training by dynamically adjusts the advantage baseline in GRPO using reward-confidence covariance derived from the first-order approximation of the variance-minimizing estimator. Both mechanisms require no additional sampling or auxiliary models and can be adapted to all GRPO variants. Experiments on mathematical reasoning benchmarks demonstrate consistent improvements across comprehensive models and algorithms. Code is available at \href{this https URL}{this https URL}. 

---
# Steve-Evolving: Open-World Embodied Self-Evolution via Fine-Grained Diagnosis and Dual-Track Knowledge Distillation 

**Authors**: Zhengwei Xie, Zhisheng Chen, Ziyan Weng, Tingyu Wu, Chenglong Li, Vireo Zhang, Kun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.13131)  

**Abstract**: Open-world embodied agents must solve long-horizon tasks where the main bottleneck is not single-step planning quality but how interaction experience is organized and evolved. To this end, we present Steve-Evolving, a non-parametric self-evolving framework that tightly couples fine-grained execution diagnosis with dual-track knowledge distillation in a closed loop. The method follows three phases: Experience Anchoring, Experience Distillation, and Knowledge-Driven Closed-Loop Control. In detail, Experience Anchoring solidifies each subgoal attempt into a structured experience tuple with a fixed schema (pre-state, action, diagnosis-result, and post-state) and organizes it in a three-tier experience space with multi-dimensional indices (e.g., condition signatures, spatial hashing, and semantic tags) plus rolling summarization for efficient and auditable recall. To ensure sufficient information density for attribution, the execution layer provides compositional diagnosis signals beyond binary outcomes, including state-difference summaries, enumerated failure causes, continuous indicators, and stagnation/loop detection. Moreover, successful trajectories of Experience Distillation are generalized into reusable skills with explicit preconditions and verification criteria, while failures are distilled into executable guardrails that capture root causes and forbid risky operations at both subgoal and task granularities. Besides, Knowledge-Driven Closed-Loop Control retrieved skills and guardrails are injected into an LLM planner, and diagnosis-triggered local replanning updates the active constraints online, forming a continual evolution process without any model parameter updates. Experiments on the long-horizon suite of Minecraft MCU demonstrate consistent improvements over static-retrieval baselines. 

---
# Beyond Final Answers: CRYSTAL Benchmark for Transparent Multimodal Reasoning Evaluation 

**Authors**: Wayner Barrios, SouYoung Jin  

**Link**: [PDF](https://arxiv.org/pdf/2603.13099)  

**Abstract**: We introduce **CRYSTAL** (*__C__lear __R__easoning via __Y__ielded __S__teps, __T__raceability and __L__ogic*), a diagnostic benchmark with 6,372 instances that evaluates multimodal reasoning through verifiable intermediate steps. We propose two complementary metrics: *Match F1*, which scores step-level precision and recall via semantic similarity matching, and *Ordered Match F1*, which further penalizes disordered reasoning chains. References are constructed through a Delphi-inspired pipeline where four independent MLLMs generate trajectories, aggregated via semantic clustering and validated through human quality gates. Evaluation of 20 MLLMs, including commercial frontier systems not used during benchmark construction, reveals systematic failures invisible to accuracy: universal cherry-picking (precision far exceeds recall), non-monotonic scaling trade-offs, and disordered reasoning where no competitive model preserves more than 60% of matched steps in correct order. Beyond evaluation, we propose the **Causal Process Reward (CPR)**, a multiplicative reward that couples answer correctness with step-level alignment, and **CPR-Curriculum**, which progressively increases reasoning difficulty during training. CPR-Curriculum achieves +32% Match F1 via GRPO where additive reward strategies fail, improving reasoning without manual step annotation. 

---
# Structured Distillation for Personalized Agent Memory: 11x Token Reduction with Retrieval Preservation 

**Authors**: Sydney Lewis  

**Link**: [PDF](https://arxiv.org/pdf/2603.13017)  

**Abstract**: Long conversations with an AI agent create a simple problem for one user: the history is useful, but carrying it verbatim is expensive. We study personalized agent memory: one user's conversation history with an agent, distilled into a compact retrieval layer for later search. Each exchange is compressed into a compound object with four fields (exchange_core, specific_context, thematic room_assignments, and regex-extracted files_touched). The searchable distilled text averages 38 tokens per exchange. Applied to 4,182 conversations (14,340 exchanges) from 6 software engineering projects, the method reduces average exchange length from 371 to 38 tokens, yielding 11x compression. We evaluate whether personalized recall survives that compression using 201 recall-oriented queries, 107 configurations spanning 5 pure and 5 cross-layer search modes, and 5 LLM graders (214,519 consensus-graded query-result pairs). The best pure distilled configuration reaches 96% of the best verbatim MRR (0.717 vs 0.745). Results are mechanism-dependent. All 20 vector search configurations remain non-significant after Bonferroni correction, while all 20 BM25 configurations degrade significantly (effect sizes |d|=0.031-0.756). The best cross-layer setup slightly exceeds the best pure verbatim baseline (MRR 0.759). Structured distillation compresses single-user agent memory without uniformly sacrificing retrieval quality. At 1/11 the context cost, thousands of exchanges fit within a single prompt while the verbatim source remains available for drill-down. We release the implementation and analysis pipeline as open-source software. 

---
# Efficient and Interpretable Multi-Agent LLM Routing via Ant Colony Optimization 

**Authors**: Xudong Wang, Chaoning Zhang, Jiaquan Zhang, Chenghao Li, Qigan Sun, Sung-Ho Bae, Peng Wang, Ning Xie, Jie Zou, Yang Yang, Hengtao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2603.12933)  

**Abstract**: Large Language Model (LLM)-driven Multi-Agent Systems (MAS) have demonstrated strong capability in complex reasoning and tool use, and heterogeneous agent pools further broaden the quality--cost trade-off space. Despite these advances, real-world deployment is often constrained by high inference cost, latency, and limited transparency, which hinders scalable and efficient routing. Existing routing strategies typically rely on expensive LLM-based selectors or static policies, and offer limited controllability for semantic-aware routing under dynamic loads and mixed intents, often resulting in unstable performance and inefficient resource utilization. To address these limitations, we propose AMRO-S, an efficient and interpretable routing framework for Multi-Agent Systems (MAS). AMRO-S models MAS routing as a semantic-conditioned path selection problem, enhancing routing performance through three key mechanisms: First, it leverages a supervised fine-tuned (SFT) small language model for intent inference, providing a low-overhead semantic interface for each query; second, it decomposes routing memory into task-specific pheromone specialists, reducing cross-task interference and optimizing path selection under mixed workloads; finally, it employs a quality-gated asynchronous update mechanism to decouple inference from learning, optimizing routing without increasing latency. Extensive experiments on five public benchmarks and high-concurrency stress tests demonstrate that AMRO-S consistently improves the quality--cost trade-off over strong routing baselines, while providing traceable routing evidence through structured pheromone patterns. 

---
# ODRL Policy Comparison Through Normalisation 

**Authors**: Jaime Osvaldo Salas, Paolo Pareti, George Konstantinidis  

**Link**: [PDF](https://arxiv.org/pdf/2603.12926)  

**Abstract**: The ODRL language has become the standard for representing policies and regulations for digital rights. However its complexity is a barrier to its usage, which has caused many related theoretical and practical works to focus on different, and not interoperable, fragments of ODRL. Moreover, semantically equivalent policies can be expressed in numerous different ways, which makes comparing them and processing them harder. Building on top of a recently defined semantics, we tackle these problems by proposing an approach that involves a parametrised normalisation of ODRL policies into its minimal components which reformulates policies with permissions and prohibitions into policies with permissions exclusively, and simplifies complex logic constraints into simple ones. We provide algorithms to compute a normal form for ODRL policies and simplifying numerical and symbolic constraints. We prove that these algorithms preserve the semantics of policies, and analyse the size complexity of the result, which is exponential on the number of attributes and linear on the number of unique values for these attributes. We show how this makes complex policies representable in more basic fragments of ODRL, and how it reduces the problem of policy comparison to the simpler problem of checking if two rules are identical. 

---
# Context is all you need: Towards autonomous model-based process design using agentic AI in flowsheet simulations 

**Authors**: Pascal Schäfer, Lukas J. Krinke, Martin Wlotzka, Norbert Asprion  

**Link**: [PDF](https://arxiv.org/pdf/2603.12813)  

**Abstract**: Agentic AI systems integrating large language models (LLMs) with reasoning and tooluse capabilities are transforming various domains - in particular, software development. In contrast, their application in chemical process flowsheet modelling remains largely unexplored. In this work, we present an agentic AI framework that delivers assistance in an industrial flowsheet simulation environment. To this end, we show the capabilities of GitHub Copilot (GitHub, Inc., 2026), when using state-of-the-art LLMs, such as Claude Opus 4.6 (Anthropic, PBC, 2026), to generate valid syntax for our in-house process modelling tool Chemasim using the technical documentation and a few commented examples as context. Based on this, we develop a multi-agent system that decomposes process development tasks with one agent solving the abstract problem using engineering knowledge and another agent implementing the solution as Chemasim code. We demonstrate the effectiveness of our framework for typical flowsheet modelling examples, including (i) a reaction/separation process, (ii) a pressure-swing distillation, and (iii) a heteroazeotropic distillation including entrainer selection. Along these lines, we discuss current limitations of the framework and outline future research directions to further enhance its capabilities. 

---
# AI Model Modulation with Logits Redistribution 

**Authors**: Zihan Wang, Zhongkui Ma, Xinguo Feng, Zhiyang Mei, Ethan Ma, Derui Wang, Minhui Xue, Guangdong Bai  

**Link**: [PDF](https://arxiv.org/pdf/2603.12755)  

**Abstract**: Large-scale models are typically adapted to meet the diverse requirements of model owners and users. However, maintaining multiple specialized versions of the model is inefficient. In response, we propose AIM, a novel model modulation paradigm that enables a single model to exhibit diverse behaviors to meet the specific end requirements. AIM enables two key modulation modes: utility and focus modulations. The former provides model owners with dynamic control over output quality to deliver varying utility levels, and the latter offers users precise control to shift model's focused input features. AIM introduces a logits redistribution strategy that operates in a training data-agnostic and retraining-free manner. We establish a formal foundation to ensure AIM's regulation capability, based on the statistical properties of logits ordering via joint probability distributions. Our evaluation confirms AIM's practicality and versatility for Al model modulation, with tasks spanning image classification, semantic segmentation and text generation, and prevalent architectures including ResNet, SegFormer and Llama. 

---
# ToolTree: Efficient LLM Agent Tool Planning via Dual-Feedback Monte Carlo Tree Search and Bidirectional Pruning 

**Authors**: Shuo Yang, Soyeon Caren Han, Yihao Ding, Shuhe Wang, Eduard Hoy  

**Link**: [PDF](https://arxiv.org/pdf/2603.12740)  

**Abstract**: Large Language Model (LLM) agents are increasingly applied to complex, multi-step tasks that require interaction with diverse external tools across various domains. However, current LLM agent tool planning methods typically rely on greedy, reactive tool selection strategies that lack foresight and fail to account for inter-tool dependencies. In this paper, we present ToolTree, a novel Monte Carlo tree search-inspired planning paradigm for tool planning. ToolTree explores possible tool usage trajectories using a dual-stage LLM evaluation and bidirectional pruning mechanism that enables the agent to make informed, adaptive decisions over extended tool-use sequences while pruning less promising branches before and after the tool execution. Empirical evaluations across both open-set and closed-set tool planning tasks on 4 benchmarks demonstrate that ToolTree consistently improves performance while keeping the highest efficiency, achieving an average gain of around 10\% compared to the state-of-the-art planning paradigm. 

---
# On Using Machine Learning to Early Detect Catastrophic Failures in Marine Diesel Engines 

**Authors**: Francesco Maione, Paolo Lino, Giuseppe Giannino, Guido Maione  

**Link**: [PDF](https://arxiv.org/pdf/2603.12733)  

**Abstract**: Catastrophic failures of marine engines imply severe loss of functionality and destroy or damage the systems irreversibly. Being sudden and often unpredictable events, they pose a severe threat to navigation, crew, and passengers. The abrupt nature makes early detection the only effective countermeasure. However, research has concentrated on modeling the gradual degradation of components, with limited attention to sudden and anomalous phenomena. This work proposes a new method for early detection of catastrophic failures. Based on real data from a failed engine, the approach evaluates the derivatives of the deviation between actual sensor readings and expected values of engine variables. Predictions are obtained by a Random Forest, which is the most suitable Machine Learning algorithm among the tested ones. Traditional methods focus on deviations of monitored signals, whereas the proposed approach employs the derivatives of the deviations to provide earlier indications of abnormal dynamics, and to alert that a rapid and dangerous event is breaking out within the system. The method allows the detection of anomalies before measurements reach critical thresholds and alarms are triggered, which is the common method in industry. Consequently, operators can be warned in advance and shut down the engine, then prevent damage and unexpected power loss. Moreover, they have the time to safely change the ship route and avoid potential obstacles. Simulation results conf irm the effectiveness of the proposed approach in anticipating occurrence of catastrophic failures. Validation on real-world data further reinforces the robustness and practical applicability of the method. It is worth noting that data acquisition to train the predictive algorithm is not a problem, since a Deep Learning-based data augmentation procedure is used. 

---
# AI Planning Framework for LLM-Based Web Agents 

**Authors**: Orit Shahnovsky, Rotem Dror  

**Link**: [PDF](https://arxiv.org/pdf/2603.12710)  

**Abstract**: Developing autonomous agents for web-based tasks is a core challenge in AI. While Large Language Model (LLM) agents can interpret complex user requests, they often operate as black boxes, making it difficult to diagnose why they fail or how they plan. This paper addresses this gap by formally treating web tasks as sequential decision-making processes. We introduce a taxonomy that maps modern agent architectures to traditional planning paradigms: Step-by-Step agents to Breadth-First Search (BFS), Tree Search agents to Best-First Tree Search, and Full-Plan-in-Advance agents to Depth-First Search (DFS). This framework allows for a principled diagnosis of system failures like context drift and incoherent task decomposition. To evaluate these behaviors, we propose five novel evaluation metrics that assess trajectory quality beyond simple success rates. We support this analysis with a new dataset of 794 human-labeled trajectories from the WebArena benchmark. Finally, we validate our evaluation framework by comparing a baseline Step-by-Step agent against a novel Full-Plan-in-Advance implementation. Our results reveal that while the Step-by-Step agent aligns more closely with human gold trajectories (38% overall success), the Full-Plan-in-Advance agent excels in technical measures such as element accuracy (89%), demonstrating the necessity of our proposed metrics for selecting appropriate agent architectures based on specific application constraints. 

---
# Generating Expressive and Customizable Evals for Timeseries Data Analysis Agents with AgentFuel 

**Authors**: Aadyaa Maddi, Prakhar Naval, Deepti Mande, Shane Duan, Muckai Girish, Vyas Sekar  

**Link**: [PDF](https://arxiv.org/pdf/2603.12483)  

**Abstract**: Across many domains (e.g., IoT, observability, telecommunications, cybersecurity), there is an emerging adoption of conversational data analysis agents that enable users to "talk to your data" to extract insights. Such data analysis agents operate on timeseries data models; e.g., measurements from sensors or events monitoring user clicks and actions in product analytics. We evaluate 6 popular data analysis agents (both open-source and proprietary) on domain-specific data and query types, and find that they fail on stateful and incident-specific queries. We observe two key expressivity gaps in existing evals: domain-customized datasets and domain-specific query types. To enable practitioners in such domains to generate customized and expressive evals for such timeseries data agents, we present AgentFuel. AgentFuel helps domain experts quickly create customized evals to perform end-to-end functional tests. We show that AgentFuel's benchmarks expose key directions for improvement in existing data agent frameworks. We also present anecdotal evidence that using AgentFuel can improve agent performance (e.g., with GEPA). AgentFuel benchmarks are available at this https URL. 

---
# Efficient Reasoning with Balanced Thinking 

**Authors**: Yulin Li, Tengyao Tu, Li Ding, Junjie Wang, Huiling Zhen, Yixin Chen, Yong Li, Zhuotao Tian  

**Link**: [PDF](https://arxiv.org/pdf/2603.12372)  

**Abstract**: Large Reasoning Models (LRMs) have shown remarkable reasoning capabilities, yet they often suffer from overthinking, expending redundant computational steps on simple problems, or underthinking, failing to explore sufficient reasoning paths despite inherent capabilities. These issues lead to inefficiencies and potential inaccuracies, limiting practical deployment in resource-constrained settings. Existing methods to mitigate overthinking, such as suppressing reflective keywords or adjusting reasoning length, may inadvertently induce underthinking, compromising accuracy. Therefore, we propose ReBalance, a training-free framework that achieves efficient reasoning with balanced thinking. ReBalance leverages confidence as a continuous indicator of reasoning dynamics, identifying overthinking through high confidence variance and underthinking via consistent overconfidence. By aggregating hidden states from a small-scale dataset into reasoning mode prototypes, we compute a steering vector to guide LRMs' reasoning trajectories. A dynamic control function modulates this vector's strength and direction based on real-time confidence, pruning redundancy during overthinking, and promoting exploration during underthinking. Extensive experiments conducted on four models ranging from 0.5B to 32B, and across nine benchmarks in math reasoning, general question answering, and coding tasks demonstrate that ReBalance effectively reduces output redundancy while improving accuracy, offering a general, training-free, and plug-and-play strategy for efficient and robust LRM deployment. Code is available at this https URL . 

---
# Context-Enriched Natural Language Descriptions of Vessel Trajectories 

**Authors**: Kostas Patroumpas, Alexandros Troupiotis-Kapeliaris, Giannis Spiliopoulos, Panagiotis Betchavas, Dimitrios Skoutas, Dimitris Zissis, Nikos Bikakis  

**Link**: [PDF](https://arxiv.org/pdf/2603.12287)  

**Abstract**: We address the problem of transforming raw vessel trajectory data collected from AIS into structured and semantically enriched representations interpretable by humans and directly usable by machine reasoning systems. We propose a context-aware trajectory abstraction framework that segments noisy AIS sequences into distinct trips each consisting of clean, mobility-annotated episodes. Each episode is further enriched with multi-source contextual information, such as nearby geographic entities, offshore navigation features, and weather conditions. Crucially, such representations can support generation of controlled natural language descriptions using LLMs. We empirically examine the quality of such descriptions generated using several LLMs over AIS data along with open contextual features. By increasing semantic density and reducing spatiotemporal complexity, this abstraction can facilitate downstream analytics and enable integration with LLMs for higher-level maritime reasoning tasks. 

---
# PhysMoDPO: Physically-Plausible Humanoid Motion with Preference Optimization 

**Authors**: Yangsong Zhang, Anujith Muraleedharan, Rikhat Akizhanov, Abdul Ahad Butt, Gül Varol, Pascal Fua, Fabio Pizzati, Ivan Laptev  

**Link**: [PDF](https://arxiv.org/pdf/2603.13228)  

**Abstract**: Recent progress in text-conditioned human motion generation has been largely driven by diffusion models trained on large-scale human motion data. Building on this progress, recent methods attempt to transfer such models for character animation and real robot control by applying a Whole-Body Controller (WBC) that converts diffusion-generated motions into executable trajectories. While WBC trajectories become compliant with physics, they may expose substantial deviations from original motion. To address this issue, we here propose PhysMoDPO, a Direct Preference Optimization framework. Unlike prior work that relies on hand-crafted physics-aware heuristics such as foot-sliding penalties, we integrate WBC into our training pipeline and optimize diffusion model such that the output of WBC becomes compliant both with physics and original text instructions. To train PhysMoDPO we deploy physics-based and task-specific rewards and use them to assign preference to synthesized trajectories. Our extensive experiments on text-to-motion and spatial control tasks demonstrate consistent improvements of PhysMoDPO in both physical realism and task-related metrics on simulated robots. Moreover, we demonstrate that PhysMoDPO results in significant improvements when applied to zero-shot motion transfer in simulation and for real-world deployment on a G1 humanoid robot. 

---
# Visual-ERM: Reward Modeling for Visual Equivalence 

**Authors**: Ziyu Liu, Shengyuan Ding, Xinyu Fang, Xuanlang Dai, Penghui Yang, Jianze Liang, Jiaqi Wang, Kai Chen, Dahua Lin, Yuhang Zang  

**Link**: [PDF](https://arxiv.org/pdf/2603.13224)  

**Abstract**: Vision-to-code tasks require models to reconstruct structured visual inputs, such as charts, tables, and SVGs, into executable or structured representations with high visual fidelity. While recent Large Vision Language Models (LVLMs) achieve strong results via supervised fine-tuning, reinforcement learning remains challenging due to misaligned reward signals. Existing rewards either rely on textual rules or coarse visual embedding similarity, both of which fail to capture fine-grained visual discrepancies and are vulnerable to reward hacking. We propose Visual Equivalence Reward Model (Visual-ERM), a multimodal generative reward model that provides fine-grained, interpretable, and task-agnostic feedback to evaluate vision-to-code quality directly in the rendered visual space. Integrated into RL, Visual-ERM improves Qwen3-VL-8B-Instruct by +8.4 on chart-to-code and yields consistent gains on table and SVG parsing (+2.7, +4.1 on average), and further strengthens test-time scaling via reflection and revision. We also introduce VisualCritic-RewardBench (VC-RewardBench), a benchmark for judging fine-grained image-to-image discrepancies on structured visual data, where Visual-ERM at 8B decisively outperforms Qwen3-VL-235B-Instruct and approaches leading closed-source models. Our results suggest that fine-grained visual reward supervision is both necessary and sufficient for vision-to-code RL, regardless of task specificity. 

---
# From Experiments to Expertise: Scientific Knowledge Consolidation for AI-Driven Computational Research 

**Authors**: Haonan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2603.13191)  

**Abstract**: While large language models (LLMs) have transformed AI agents into proficient executors of computational materials science, performing a hundred simulations does not make a researcher. What distinguishes research from routine execution is the progressive accumulation of knowledge -- learning which approaches fail, recognizing patterns across systems, and applying understanding to new problems. However, the prevailing paradigm in AI-driven computational science treats each execution in isolation, largely discarding hard-won insights between runs. Here we present QMatSuite, an open-source platform closing this gap. Agents record findings with full provenance, retrieve knowledge before new calculations, and in dedicated reflection sessions correct erroneous findings and synthesize observations into cross-compound patterns. In benchmarks on a six-step quantum-mechanical simulation workflow, accumulated knowledge reduces reasoning overhead by 67% and improves accuracy from 47% to 3% deviation from literature -- and when transferred to an unfamiliar material, achieves 1% deviation with zero pipeline failures. 

---
# LLM Constitutional Multi-Agent Governance 

**Authors**: J. de Curtò, I. de Zarzà  

**Link**: [PDF](https://arxiv.org/pdf/2603.13189)  

**Abstract**: Large Language Models (LLMs) can generate persuasive influence strategies that shift cooperative behavior in multi-agent populations, but a critical question remains: does the resulting cooperation reflect genuine prosocial alignment, or does it mask erosion of agent autonomy, epistemic integrity, and distributional fairness? We introduce Constitutional Multi-Agent Governance (CMAG), a two-stage framework that interposes between an LLM policy compiler and a networked agent population, combining hard constraint filtering with soft penalized-utility optimization that balances cooperation potential against manipulation risk and autonomy pressure. We propose the Ethical Cooperation Score (ECS), a multiplicative composite of cooperation, autonomy, integrity, and fairness that penalizes cooperation achieved through manipulative means. In experiments on scale-free networks of 80 agents under adversarial conditions (70% violating candidates), we benchmark three regimes: full CMAG, naive filtering, and unconstrained optimization. While unconstrained optimization achieves the highest raw cooperation (0.873), it yields the lowest ECS (0.645) due to severe autonomy erosion (0.867) and fairness degradation (0.888). CMAG attains an ECS of 0.741, a 14.9% improvement, while preserving autonomy at 0.985 and integrity at 0.995, with only modest cooperation reduction to 0.770. The naive ablation (ECS = 0.733) confirms that hard constraints alone are insufficient. Pareto analysis shows CMAG dominates the cooperation-autonomy trade-off space, and governance reduces hub-periphery exposure disparities by over 60%. These findings establish that cooperation is not inherently desirable without governance: constitutional constraints are necessary to ensure that LLM-mediated influence produces ethically stable outcomes rather than manipulative equilibria. 

---
# Learnability and Privacy Vulnerability are Entangled in a Few Critical Weights 

**Authors**: Xingli Fang, Jung-Eun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2603.13186)  

**Abstract**: Prior approaches for membership privacy preservation usually update or retrain all weights in neural networks, which is costly and can lead to unnecessary utility loss or even more serious misalignment in predictions between training data and non-training data. In this work, we observed three insights: i) privacy vulnerability exists in a very small fraction of weights; ii) however, most of those weights also critically impact utility performance; iii) the importance of weights stems from their locations rather than their values. According to these insights, to preserve privacy, we score critical weights, and instead of discarding those neurons, we rewind only the weights for fine-tuning. We show that, through extensive experiments, this mechanism exhibits outperforming resilience in most cases against Membership Inference Attacks while maintaining utility. 

---
# MXNorm: Reusing MXFP block scales for efficient tensor normalisation 

**Authors**: Callum McLean, Luke Y. Prince, Alexandre Payot, Paul Balança, Carlo Luschi  

**Link**: [PDF](https://arxiv.org/pdf/2603.13180)  

**Abstract**: Matrix multiplication performance has long been the major bottleneck to scaling deep learning workloads, which has stimulated the design of new accelerators that use increasingly low-precision number formats. However, improvements in matrix multiplication performance have far outstripped improvements in performance on reductions and elementwise computations, which are still being performed in higher precision. In this work, we propose MXNorm, a drop-in replacement for RMSNorm that estimates the RMS using only the block scales calculated as part of the MXFP8 cast and enables a 32x decrease in the size of reduction needed for normalization. We validate our approximation method on pre-training of Llama 3 models of 125M, 1B and 8B parameters, finding minimal loss of training accuracy compared to a baseline using RMSNorm with MXFP8 matmuls. We also show practical kernel speedups using only this http URL of up to 2.4x for MXNorm over RMSNorm, corresponding to a 1.3% speedup in Llama 3 8B transformer layers in MXFP8 and a 2.6% speedup in NVFP4. 

---
# Clustering Astronomical Orbital Synthetic Data Using Advanced Feature Extraction and Dimensionality Reduction Techniques 

**Authors**: Eraldo Pereira Marinho, Nelson Callegari Junior, Fabricio Aparecido Breve, Caetano Mazzoni Ranieri  

**Link**: [PDF](https://arxiv.org/pdf/2603.13177)  

**Abstract**: The dynamics of Saturn's satellite system offer a rich framework for studying orbital stability and resonance interactions. Traditional methods for analysing such systems, including Fourier analysis and stability metrics, struggle with the scale and complexity of modern datasets. This study introduces a machine learning-based pipeline for clustering approximately 22,300 simulated satellite orbits, addressing these challenges with advanced feature extraction and dimensionality reduction techniques. The key to this approach is using MiniRocket, which efficiently transforms 400 timesteps into a 9,996-dimensional feature space, capturing intricate temporal patterns. Additional automated feature extraction and dimensionality reduction techniques refine the data, enabling robust clustering analysis. This pipeline reveals stability regions, resonance structures, and other key behaviours in Saturn's satellite system, providing new insights into their long-term dynamical evolution. By integrating computational tools with traditional celestial mechanics techniques, this study offers a scalable and interpretable methodology for analysing large-scale orbital datasets and advancing the exploration of planetary dynamics. 

---
# ESG-Bench: Benchmarking Long-Context ESG Reports for Hallucination Mitigation 

**Authors**: Siqi Sun, Ben Peng Wu, Mali Jin, Peizhen Bai, Hanpei Zhang, Xingyi Song  

**Link**: [PDF](https://arxiv.org/pdf/2603.13154)  

**Abstract**: As corporate responsibility increasingly incorporates environmental, social, and governance (ESG) criteria, ESG reporting is becoming a legal requirement in many regions and a key channel for documenting sustainability practices and assessing firms' long-term and ethical performance. However, the length and complexity of ESG disclosures make them difficult to interpret and automate the analysis reliably. To support scalable and trustworthy analysis, this paper introduces ESG-Bench, a benchmark dataset for ESG report understanding and hallucination mitigation in large language models (LLMs). ESG-Bench contains human-annotated question-answer (QA) pairs grounded in real-world ESG report contexts, with fine-grained labels indicating whether model outputs are factually supported or hallucinated. Framing ESG report analysis as a QA task with verifiability constraints enables systematic evaluation of LLMs' ability to extract and reason over ESG content and provides a new use case: mitigating hallucinations in socially sensitive, compliance-critical settings. We design task-specific Chain-of-Thought (CoT) prompting strategies and fine-tune multiple state-of-the-art LLMs on ESG-Bench using CoT-annotated rationales. Our experiments show that these CoT-based methods substantially outperform standard prompting and direct fine-tuning in reducing hallucinations, and that the gains transfer to existing QA benchmarks beyond the ESG domain. 

---
# Developing the PsyCogMetrics AI Lab to Evaluate Large Language Models and Advance Cognitive Science -- A Three-Cycle Action Design Science Study 

**Authors**: Zhiye Jin, Yibai Li, K. D. Joshi, Xuefei, Deng, Xiaobing  

**Link**: [PDF](https://arxiv.org/pdf/2603.13126)  

**Abstract**: This study presents the development of the PsyCogMetrics AI Lab (this http URL), an integrated, cloud-based platform that operationalizes psychometric and cognitive-science methodologies for Large Language Model (LLM) evaluation. Framed as a three-cycle Action Design Science study, the Relevance Cycle identifies key limitations in current evaluation methods and unfulfilled stakeholder needs. The Rigor Cycle draws on kernel theories such as Popperian falsifiability, Classical Test Theory, and Cognitive Load Theory to derive deductive design objectives. The Design Cycle operationalizes these objectives through nested Build-Intervene-Evaluate loops. The study contributes a novel IT artifact, a validated design for LLM evaluation, benefiting research at the intersection of AI, psychology, cognitive science, and the social and behavioral sciences. 

---
# Geometry-Guided Camera Motion Understanding in VideoLLMs 

**Authors**: Haoan Feng, Sri Harsha Musunuri, Guan-Ming Su  

**Link**: [PDF](https://arxiv.org/pdf/2603.13119)  

**Abstract**: Camera motion is a fundamental geometric signal that shapes visual perception and cinematic style, yet current video-capable vision-language models (VideoLLMs) rarely represent it explicitly and often fail on fine-grained motion primitives. We address this gap with a framework of $\textbf{benchmarking}$, $\textbf{diagnosis}$, and $\textbf{injection}$. We curate $\textbf{CameraMotionDataset}$, a large-scale synthetic dataset with explicit camera control, formulate camera motion as constraint-aware multi-label recognition, and construct a VQA benchmark--$\textbf{CameraMotionVQA}$. Across diverse off-the-shelf VideoLLMs, we observe substantial errors in recognizing camera motion primitives. Probing experiments on a Qwen2.5-VL vision encoder suggest that camera motion cues are weakly represented, especially in deeper ViT blocks, helping explain the observed failure modes. To bridge this gap without costly training or fine-tuning, we propose a lightweight, model-agnostic pipeline that extracts geometric camera cues from 3D foundation models (3DFMs), predicts constrained motion primitives with a temporal classifier, and injects them into downstream VideoLLM inference via structured prompting. Experiments demonstrate improved motion recognition and more camera-aware model responses, highlighting geometry-driven cue extraction and structured prompting as practical steps toward a camera-aware VideoLLM and VLA system. The dataset and benchmark is publicly available at this https URL. 

---
# BoSS: A Best-of-Strategies Selector as an Oracle for Deep Active Learning 

**Authors**: Denis Huseljic, Paul Hahn, Marek Herde, Christoph Sandrock, Bernhard Sick  

**Link**: [PDF](https://arxiv.org/pdf/2603.13109)  

**Abstract**: Active learning (AL) aims to reduce annotation costs while maximizing model performance by iteratively selecting valuable instances. While foundation models have made it easier to identify these instances, existing selection strategies still lack robustness across different models, annotation budgets, and datasets. To highlight the potential weaknesses of existing AL strategies and provide a reference point for research, we explore oracle strategies, i.e., strategies that approximate the optimal selection by accessing ground-truth information unavailable in practical AL scenarios. Current oracle strategies, however, fail to scale effectively to large datasets and complex deep neural networks. To tackle these limitations, we introduce the Best-of-Strategy Selector (BoSS), a scalable oracle strategy designed for large-scale AL scenarios. BoSS constructs a set of candidate batches through an ensemble of selection strategies and then selects the batch yielding the highest performance gain. As an ensemble of selection strategies, BoSS can be easily extended with new state-of-the-art strategies as they emerge, ensuring it remains a reliable oracle strategy in the future. Our evaluation demonstrates that i) BoSS outperforms existing oracle strategies, ii) state-of-the-art AL strategies still fall noticeably short of oracle performance, especially in large-scale datasets with many classes, and iii) one possible solution to counteract the inconsistent performance of AL strategies might be to employ an ensemble-based approach for the selection. 

---
# Evaluating VLMs' Spatial Reasoning Over Robot Motion: A Step Towards Robot Planning with Motion Preferences 

**Authors**: Wenxi Wu, Jingjing Zhang, Martim Brandão  

**Link**: [PDF](https://arxiv.org/pdf/2603.13100)  

**Abstract**: Understanding user instructions and object spatial relations in surrounding environments is crucial for intelligent robot systems to assist humans in various tasks. The natural language and spatial reasoning capabilities of Vision-Language Models (VLMs) have the potential to enhance the generalization of robot planners on new tasks, objects, and motion specifications. While foundation models have been applied to task planning, it is still unclear the degree to which they have the capability of spatial reasoning required to enforce user preferences or constraints on motion, such as desired distances from objects, topological properties, or motion style preferences. In this paper, we evaluate the capability of four state-of-the-art VLMs at spatial reasoning over robot motion, using four different querying methods. Our results show that, with the highest-performing querying method, Qwen2.5-VL achieves 71.4% accuracy zero-shot and 75% on a smaller model after fine-tuning, and GPT-4o leads to lower performance. We evaluate two types of motion preferences (object-proximity and path-style), and we also analyze the trade-off between accuracy and computation cost in number of tokens. This work shows some promise in the potential of VLM integration with robot motion planning pipelines. 

---
# Human-in-the-Loop LLM Grading for Handwritten Mathematics Assessments 

**Authors**: Arne Vanhoyweghen, Vincent Holst, Melika Mobini, Lukas Van de Voorde, Tibo Vanleke, Bert Verbruggen, Brecht Verbeken, Andres Algaba, Sam Verboven, Marie-Anne Guerry, Filip Van Droogenbroeck, Vincent Ginis  

**Link**: [PDF](https://arxiv.org/pdf/2603.13083)  

**Abstract**: Providing timely and individualised feedback on handwritten student work is highly beneficial for learning but difficult to achieve at scale. This challenge has become more pressing as generative AI undermines the reliability of take-home assessments, shifting emphasis toward supervised, in-class evaluation. We present a scalable, end-to-end workflow for LLM-assisted grading of short, pen-and-paper assessments. The workflow spans (1) constructing solution keys, (2) developing detailed rubric-style grading keys used to guide the LLM, and (3) a grading procedure that combines automated scanning and anonymisation, multi-pass LLM scoring, automated consistency checks, and mandatory human verification. We deploy the system in two undergraduate mathematics courses using six low-stakes in-class tests. Empirically, LLM assistance reduces grading time by approximately 23% while achieving agreement comparable to, and in several cases tighter than, fully manual grading. Occasional model errors occur but are effectively contained by the hybrid design. Overall, our results show that carefully embedded human-in-the-loop LLM grading can substantially reduce workload while maintaining fairness and accuracy. 

---
# GeoChemAD: Benchmarking Unsupervised Geochemical Anomaly Detection for Mineral Exploration 

**Authors**: Yihao Ding, Yiran Zhang, Chris Gonzalez, Eun-Jung Holden, Wei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.13068)  

**Abstract**: Geochemical anomaly detection plays a critical role in mineral exploration as deviations from regional geochemical baselines may indicate mineralization. Existing studies suffer from two key limitations: (1) single region scenarios which limit model generalizability; (2) proprietary datasets, which makes result reproduction unattainable. In this work, we introduce \textbf{GeoChemAD}, an open-source benchmark dataset compiled from government-led geological surveys, covering multiple regions, sampling sources, and target elements. The dataset comprises eight subsets representing diverse spatial scales and sampling conditions. To establish strong baselines, we reproduce and benchmark a range of unsupervised anomaly detection methods, including statistical models, generative and transformer-based approaches. Furthermore, we propose \textbf{GeoChemFormer}, a transformer-based framework that leverages self-supervised pretraining to learn target-element-aware geochemical representations for spatial samples. Extensive experiments demonstrate that GeoChemFormer consistently achieves superior and robust performance across all eight subsets, outperforming existing unsupervised methods in both anomaly detection accuracy and generalization capability. The proposed dataset and framework provide a foundation for reproducible research and future development in this direction. 

---
# L2GTX: From Local to Global Time Series Explanations 

**Authors**: Ephrem Tibebe Mekonnen, Luca Longo, Lucas Rizzo, Pierpaolo Dondio  

**Link**: [PDF](https://arxiv.org/pdf/2603.13065)  

**Abstract**: Deep learning models achieve high accuracy in time series classification, yet understanding their class-level decision behaviour remains challenging. Explanations for time series must respect temporal dependencies and identify patterns that recur across instances. Existing approaches face three limitations: model-agnostic XAI methods developed for images and tabular data do not readily extend to time series, global explanation synthesis for time series remains underexplored, and most existing global approaches are model-specific. We propose L2GTX, a model-agnostic framework that generates class-wise global explanations by aggregating local explanations from a representative set of instances. L2GTX extracts clusters of parameterised temporal event primitives, such as increasing or decreasing trends and local extrema, together with their importance scores from instance-level explanations produced by LOMATCE. These clusters are merged across instances to reduce redundancy, and an instance-cluster importance matrix is used to estimate global relevance. Under a user-defined instance selection budget, L2GTX selects representative instances that maximise coverage of influential clusters. Events from the selected instances are then aggregated into concise class-wise global explanations. Experiments on six benchmark time series datasets show that L2GTX produces compact and interpretable global explanations while maintaining stable global faithfulness measured as mean local surrogate fidelity. 

---
# Competition-Aware CPC Forecasting with Near-Market Coverage 

**Authors**: Sebastian Frey, Edoardo Beccari, Maximilian Kranz, Nicolò Alberto Pellizzari, Ali Mete Karaman, Qiwei Han, Maximilian Kaiser  

**Link**: [PDF](https://arxiv.org/pdf/2603.13059)  

**Abstract**: Cost-per-click (CPC) in paid search is a volatile auction outcome generated by a competitive landscape that is only partially observable from any single advertiser's history. Using Google Ads auction logs from a concentrated car-rental market (2021--2023), we forecast weekly CPC for 1,811 keyword series and approximate latent competition through complementary signals derived from keyword text, CPC trajectories, and geographic market structure. We construct (i) semantic neighborhoods and a semantic keyword graph from pretrained transformer-based representations of keyword text, (ii) behavioral neighborhoods via Dynamic Time Warping (DTW) alignment of CPC trajectories, and (iii) geographic-intent covariates capturing localized demand and marketplace heterogeneity. We extensively evaluate these signals both as stand-alone covariates and as relational priors in spatiotemporal graph forecasters, benchmarking them against strong statistical, neural, and time-series foundation-model baselines. Across methods, competition-aware augmentation improves stability and error profiles at business-relevant medium and longer horizons, where competitive regimes shift and volatility is most consequential. The results show that broad market-outcome coverage, combined with keyword-derived semantic and geographic priors, provides a scalable way to approximate latent competition and improve CPC forecasting in auction-driven markets. 

---
# Team RAS in 10th ABAW Competition: Multimodal Valence and Arousal Estimation Approach 

**Authors**: Elena Ryumina, Maxim Markitantov, Alexandr Axyonov, Dmitry Ryumin, Mikhail Dolgushin, Denis Dresvyanskiy, Alexey Karpov  

**Link**: [PDF](https://arxiv.org/pdf/2603.13056)  

**Abstract**: Continuous emotion recognition in terms of valence and arousal under in-the-wild (ITW) conditions remains a challenging problem due to large variations in appearance, head pose, illumination, occlusions, and subject-specific patterns of affective expression. We present a multimodal method for valence-arousal estimation ITW. Our method combines three complementary modalities: face, behavior, and audio. The face modality relies on GRADA-based frame-level embeddings and Transformer-based temporal regression. We use Qwen3-VL-4B-Instruct to extract behavior-relevant information from video segments, while Mamba is used to model temporal dynamics across segments. The audio modality relies on WavLM-Large with attention-statistics pooling and includes a cross-modal filtering stage to reduce the influence of unreliable or non-speech segments. To fuse modalities, we explore two fusion strategies: a Directed Cross-Modal Mixture-of-Experts Fusion Strategy that learns interactions between modalities with adaptive weighting, and a Reliability-Aware Audio-Visual Fusion Strategy that combines visual features at the frame-level while using audio as complementary context. The results are reported on the Aff-Wild2 dataset following the 10th Affective Behavior Analysis in-the-Wild (ABAW) challenge protocol. Experiments demonstrate that the proposed multimodal fusion strategy achieves a Concordance Correlation Coefficient (CCC) of 0.658 on the Aff-Wild2 development set. 

---
# Are General-Purpose Vision Models All We Need for 2D Medical Image Segmentation? A Cross-Dataset Empirical Study 

**Authors**: Vanessa Borst, Samuel Kounev  

**Link**: [PDF](https://arxiv.org/pdf/2603.13044)  

**Abstract**: Medical image segmentation (MIS) is a fundamental component of computer-assisted diagnosis and clinical decision support systems. Over the past decade, numerous architectures specifically tailored to medical imaging have emerged to address domain-specific challenges such as low contrast, small anatomical structures, and limited annotated data. In parallel, rapid progress in computer vision has produced highly capable general-purpose vision models (GP-VMs) originally designed for natural images. Despite their strong performance on standard vision benchmarks, their effectiveness for MIS remains insufficiently understood. In this work, we conduct a controlled empirical study to examine whether specialized medical segmentation architectures (SMAs) provide systematic advantages over modern GP-VMs for 2D MIS. We compare eleven SMAs and GP-VMs using a unified training and evaluation protocol. Experiments are performed across three heterogeneous datasets covering different imaging modalities, class structures, and data characteristics. Beyond segmentation accuracy, we analyze qualitative Grad-CAM visualizations to investigate explainability (XAI) behavior. Our results demonstrate that, for the analyzed datasets, GP-VMs out-perform the majority of specialized MIS models. Moreover, XAI analyses indicate that GP-VMs can capture clinically relevant structures without explicit domain-specific architectural design. These findings suggest that GP-VMs can represent a viable alternative to domain-specific methods, highlighting the importance of informed model selection for end-to-end MIS systems. All code and resources are available at GitHub. 

---
# Interrogating Design Homogenization in Web Vibe Coding 

**Authors**: Donghoon Shin, Alice Gao, Rock Yuren Pang, Jaewook Lee, Katharina Reinecke, Emily Tseng  

**Link**: [PDF](https://arxiv.org/pdf/2603.13036)  

**Abstract**: Generative AI is known for its tendency to homogenize, often reproducing dominant style conventions found in training data. However, it remains unclear how these homogenizing effects extend to complex structural tasks like web design. As lay creators increasingly turn to LLMs to 'vibe-code' websites -- prompting for aesthetic and functional goals rather than writing code -- they may inadvertently narrow the diversity of their designs, and limit creative expression throughout the internet. In this paper, we interrogate the possibility of design homogenization in web vibe coding. We first characterize the vibe coding lifecycle, pinpointing stages where homogenization risks may arise. We then conduct a sociotechnical risk analysis unpacking the potential harms of web vibe coding and their interaction with design homogenization. We identify that the push for frictionless generation can exacerbate homogenization and its harms. Finally, we propose a mitigation framework centered on the idea of productive friction. Through case studies at the micro, meso, and macro levels, we show how centering productive friction can empower creators to challenge default outputs and preserve diverse expression in AI-mediated web design. 

---
# Purify Once, Edit Freely: Breaking Image Protections under Model Mismatch 

**Authors**: Qichen Zhao, Shengfang Zhai, Xinjian Bai, Qingni Shen, Qiqi Lin, Yansong Gao, Zhonghai Wu  

**Link**: [PDF](https://arxiv.org/pdf/2603.13028)  

**Abstract**: Diffusion models enable high-fidelity image editing but can also be misused for unauthorized style imitation and harmful content generation. To mitigate these risks, proactive image protection methods embed small, often imperceptible adversarial perturbations into images before sharing to disrupt downstream editing or fine-tuning. However, in realistic post-release scenarios, content owners cannot control downstream processing pipelines, and protections optimized for a surrogate model may fail when attackers use mismatched diffusion pipelines. Existing purification methods can weaken protections but often sacrifice image quality and rarely examine architectural mismatch. We introduce a unified post-release purification framework to evaluate protection survivability under model mismatch. We propose two practical purifiers: VAE-Trans, which corrects protected images via latent-space projection, and EditorClean, which performs instruction-guided reconstruction with a Diffusion Transformer to exploit architectural heterogeneity. Both operate without access to protected images or defense internals. Across 2,100 editing tasks and six representative protection methods, EditorClean consistently restores editability. Compared to protected inputs, it improves PSNR by 3-6 dB and reduces FID by 50-70 percent on downstream edits, while outperforming prior purification baselines by about 2 dB PSNR and 30 percent lower FID. Our results reveal a purify-once, edit-freely failure mode: once purification succeeds, the protective signal is largely removed, enabling unrestricted editing. This highlights the need to evaluate protections under model mismatch and design defenses robust to heterogeneous attackers. 

---
# SortScrews: A Dataset and Baseline for Real-time Screw Classification 

**Authors**: Tianhao Fu, Bingxuan Yang, Juncheng Guo, Shrena Sribalan, Yucheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.13027)  

**Abstract**: Automatic identification of screw types is important for industrial automation, robotics, and inventory management. However, publicly available datasets for screw classification are scarce, particularly for controlled single-object scenarios commonly encountered in automated sorting systems. In this work, we introduce $\textbf{SortScrews}$, a dataset for casewise visual classification of screws. The dataset contains 560 RGB images at $512\times512$ resolution covering six screw types and a background class. Images are captured using a standardized acquisition setup and include mild variations in lighting and camera perspective across four capture settings.
To facilitate reproducible research and dataset expansion, we also provide a reusable data collection script that allows users to easily construct similar datasets for custom hardware components using inexpensive camera setups.
We establish baseline results using transfer learning with EfficientNet-B0 and ResNet-18 classifiers pretrained on ImageNet. In addition, we conduct a well-explored failure analysis. Despite the limited dataset size, these lightweight models achieve strong classification accuracy, demonstrating that controlled acquisition conditions enable effective learning even with relatively small datasets. The dataset, collection pipeline, and baseline training code are publicly available at this https URL. 

---
# SAW: Toward a Surgical Action World Model via Controllable and Scalable Video Generation 

**Authors**: Sampath Rapuri, Lalithkumar Seenivasan, Dominik Schneider, Roger Soberanis-Mukul, Yufan He, Hao Ding, Jiru Xu, Chenhao Yu, Chenyan Jing, Pengfei Guo, Daguang Xu, Mathias Unberath  

**Link**: [PDF](https://arxiv.org/pdf/2603.13024)  

**Abstract**: A surgical world model capable of generating realistic surgical action videos with precise control over tool-tissue interactions can address fundamental challenges in surgical AI and simulation -- from data scarcity and rare event synthesis to bridging the sim-to-real gap for surgical automation. However, current video generation methods, the very core of such surgical world models, require expensive annotations or complex structured intermediates as conditioning signals at inference, limiting their scalability. Other approaches exhibit limited temporal consistency across complex laparoscopic scenes and do not possess sufficient realism. We propose Surgical Action World (SAW) -- a step toward surgical action world modeling through video diffusion conditioned on four lightweight signals: language prompts encoding tool-action context, a reference surgical scene, tissue affordance mask, and 2D tool-tip trajectories. We design a conditional video diffusion approach that reformulates video-to-video diffusion into trajectory-conditioned surgical action synthesis. The backbone diffusion model is fine-tuned on a custom-curated dataset of 12,044 laparoscopic clips with lightweight spatiotemporal conditioning signals, leveraging a depth consistency loss to enforce geometric plausibility without requiring depth at inference. SAW achieves state-of-the-art temporal consistency (CD-FVD: 199.19 vs. 546.82) and strong visual quality on held-out test data. Furthermore, we demonstrate its downstream utility for (a) surgical AI, where augmenting rare actions with SAW-generated videos improves action recognition (clipping F1-score: 20.93% to 43.14%; cutting: 0.00% to 8.33%) on real test data, and (b) surgical simulation, where rendering tool-tissue interaction videos from simulator-derived trajectory points toward a visually faithful simulation engine. 

---
# daVinci-Env: Open SWE Environment Synthesis at Scale 

**Authors**: Dayuan Fu, Shenyu Wu, Yunze Wu, Zerui Peng, Yaxing Huang, Jie Sun, Ji Zeng, Mohan Jiang, Lin Zhang, Yukun Li, Jiarui Hu, Liming Liu, Jinlong Hou, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.13023)  

**Abstract**: Training capable software engineering (SWE) agents demands large-scale, executable, and verifiable environments that provide dynamic feedback loops for iterative code editing, test execution, and solution refinement. However, existing open-source datasets remain limited in scale and repository diversity, while industrial solutions are opaque with unreleased infrastructure, creating a prohibitive barrier for most academic research groups. We present OpenSWE, the largest fully transparent framework for SWE agent training in Python, comprising 45,320 executable Docker environments spanning over 12.8k repositories, with all Dockerfiles, evaluation scripts, and infrastructure fully open-sourced for reproducibility. OpenSWE is built through a multi-agent synthesis pipeline deployed across a 64-node distributed cluster, automating repository exploration, Dockerfile construction, evaluation script generation, and iterative test analysis. Beyond scale, we propose a quality-centric filtering pipeline that characterizes the inherent difficulty of each environment, filtering out instances that are either unsolvable or insufficiently challenging and retaining only those that maximize learning efficiency. With $891K spent on environment construction and an additional $576K on trajectory sampling and difficulty-aware curation, the entire project represents a total investment of approximately $1.47 million, yielding about 13,000 curated trajectories from roughly 9,000 quality guaranteed environments. Extensive experiments validate OpenSWE's effectiveness: OpenSWE-32B and OpenSWE-72B achieve 62.4% and 66.0% on SWE-bench Verified, establishing SOTA among Qwen2.5 series. Moreover, SWE-focused training yields substantial out-of-domain improvements, including up to 12 points on mathematical reasoning and 5 points on science benchmarks, without degrading factual recall. 

---
# ARL-Tangram: Unleash the Resource Efficiency in Agentic Reinforcement Learning 

**Authors**: Bangjun Xiao, Yihao Zhao, Xiangwei Deng, Shihua Yu, Yuxing Xiang, Huaqiu Liu, Qiying Wang, Liang Zhao, Hailin Zhang, Xuanzhe Liu, Xin Jin, Fuli Luo  

**Link**: [PDF](https://arxiv.org/pdf/2603.13019)  

**Abstract**: Agentic reinforcement learning (RL) has emerged as a transformative workload in cloud clusters, enabling large language models (LLMs) to solve complex problems through interactions with real world. However, unlike traditional RL, agentic RL demands substantial external cloud resources, e.g., CPUs for code execution and GPUs for reward models, that exist outside the primary training cluster. Existing agentic RL framework typically rely on static over-provisioning, i.e., resources are often tied to long-lived trajectories or isolated by tasks, which leads to severe resource inefficiency.
We propose the action-level orchestration, and incorporate it into ARL-Tangram, a unified resource management system that enables fine-grained external resource sharing and elasticity. ARL-Tangram utilizes a unified action-level formulation and an elastic scheduling algorithm to minimize action completion time (ACT) while satisfying heterogeneous resource constraints. Further, heterogeneous resource managers are tailored to efficiently support the action-level execution on resources with heterogeneous characteristics and topologies. Evaluation on real-world agentic RL tasks demonstrates that ARL-Tangram improves average ACT by up to 4.3$\times$, speeds up the step duration of RL training by up to 1.5$\times$, and saves the external resources by up to 71.2$\%$. This system has been deployed to support the training of the MiMo series models. 

---
# Fair Lung Disease Diagnosis from Chest CT via Gender-Adversarial Attention Multiple Instance Learning 

**Authors**: Aditya Parikh, Aasa Feragen  

**Link**: [PDF](https://arxiv.org/pdf/2603.12988)  

**Abstract**: We present a fairness-aware framework for multi-class lung disease diagnosis from chest CT volumes, developed for the Fair Disease Diagnosis Challenge at the PHAROS-AIF-MIH Workshop (CVPR 2026). The challenge requires classifying CT scans into four categories -- Healthy, COVID-19, Adenocarcinoma, and Squamous Cell Carcinoma -- with performance measured as the average of per-gender macro F1 scores, explicitly penalizing gender-inequitable predictions. Our approach addresses two core difficulties: the sparse pathological signal across hundreds of slices, and a severe demographic imbalance compounded across disease class and gender. We propose an attention-based Multiple Instance Learning (MIL) model on a ConvNeXt backbone that learns to identify diagnostically relevant slices without slice-level supervision, augmented with a Gradient Reversal Layer (GRL) that adversarially suppresses gender-predictive structure in the learned scan representation. Training incorporates focal loss with label smoothing, stratified cross-validation over joint (class, gender) strata, and targeted oversampling of the most underrepresented subgroup. At inference, all five-fold checkpoints are ensembled with horizontal-flip test-time augmentation via soft logit voting and out-of-the-fold threshold optimization for robustness. Our model achieves a mean validation competition score of 0.685 (std - 0.030), with the best single fold reaching 0.759. All training and inference code is publicly available at this https URL 

---
# Is Human Annotation Necessary? Iterative MBR Distillation for Error Span Detection in Machine Translation 

**Authors**: Boxuan Lyu, Haiyue Song, Zhi Qu  

**Link**: [PDF](https://arxiv.org/pdf/2603.12983)  

**Abstract**: Error Span Detection (ESD) is a crucial subtask in Machine Translation (MT) evaluation, aiming to identify the location and severity of translation errors. While fine-tuning models on human-annotated data improves ESD performance, acquiring such data is expensive and prone to inconsistencies among annotators. To address this, we propose a novel self-evolution framework based on Minimum Bayes Risk (MBR) decoding, named Iterative MBR Distillation for ESD, which eliminates the reliance on human annotations by leveraging an off-the-shelf LLM to generate this http URL experiments on the WMT Metrics Shared Task datasets demonstrate that models trained solely on these self-generated pseudo-labels outperform both unadapted base model and supervised baselines trained on human annotations at the system and span levels, while maintaining competitive sentence-level performance. 

---
# Efficient Real-World Autonomous Racing via Attenuated Residual Policy Optimization 

**Authors**: Raphael Trumpp, Denis Hoornaert, Mirco Theile, Marco Caccamo  

**Link**: [PDF](https://arxiv.org/pdf/2603.12960)  

**Abstract**: Residual policy learning (RPL), in which a learned policy refines a static base policy using deep reinforcement learning (DRL), has shown strong performance across various robotic applications. Its effectiveness is particularly evident in autonomous racing, a domain that serves as a challenging benchmark for real-world DRL. However, deploying RPL-based controllers introduces system complexity and increases inference latency. We address this by introducing an extension of RPL named attenuated residual policy optimization ($\alpha$-RPO). Unlike standard RPL, $\alpha$-RPO yields a standalone neural policy by progressively attenuating the base policy, which initially serves to bootstrap learning. Furthermore, this mechanism enables a form of privileged learning, where the base policy is permitted to use sensor modalities not required for final deployment. We design $\alpha$-RPO to integrate seamlessly with PPO, ensuring that the attenuated influence of the base controller is dynamically compensated during policy optimization. We evaluate $\alpha$-RPO by building a framework for 1:10-scaled autonomous racing around it. In both simulation and zero-shot real-world transfer to Roboracer cars, $\alpha$-RPO not only reduces system complexity but also improves driving performance compared to baselines - demonstrating its practicality for robotic deployment. Our code is available at: this https URL. 

---
# Delta1 with LLM: symbolic and neural integration for credible and explainable reasoning 

**Authors**: Yang Xu, Jun Liu, Shuwei Chen, Chris Nugent, Hailing Guo  

**Link**: [PDF](https://arxiv.org/pdf/2603.12953)  

**Abstract**: Neuro-symbolic reasoning increasingly demands frameworks that unite the formal rigor of logic with the interpretability of large language models (LLMs). We introduce an end to end explainability by construction pipeline integrating the Automated Theorem Generator Delta1 based on the full triangular standard contradiction (FTSC) with LLMs. Delta1 deterministically constructs minimal unsatisfiable clause sets and complete theorems in polynomial time, ensuring both soundness and minimality by construction. The LLM layer verbalizes each theorem and proof trace into coherent natural language explanations and actionable insights. Empirical studies across health care, compliance, and regulatory domains show that Delta1 and LLM enables interpretable, auditable, and domain aligned reasoning. This work advances the convergence of logic, language, and learning, positioning constructive theorem generation as a principled foundation for neuro-symbolic explainable AI. 

---
# Thinking in Streaming Video 

**Authors**: Zikang Liu, Longteng Guo, Handong Li, Ru Zhen, Xingjian He, Ruyi Ji, Xiaoming Ren, Yanhao Zhang, Haonan Lu, Jing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.12938)  

**Abstract**: Real-time understanding of continuous video streams is essential for interactive assistants and multimodal agents operating in dynamic environments. However, most existing video reasoning approaches follow a batch paradigm that defers reasoning until the full video context is observed, resulting in high latency and growing computational cost that are incompatible with streaming scenarios. In this paper, we introduce ThinkStream, a framework for streaming video reasoning based on a Watch--Think--Speak paradigm that enables models to incrementally update their understanding as new video observations arrive. At each step, the model performs a short reasoning update and decides whether sufficient evidence has accumulated to produce a response. To support long-horizon streaming, we propose Reasoning-Compressed Streaming Memory (RCSM), which treats intermediate reasoning traces as compact semantic memory that replaces outdated visual tokens while preserving essential context. We further train the model using a Streaming Reinforcement Learning with Verifiable Rewards scheme that aligns incremental reasoning and response timing with the requirements of streaming interaction. Experiments on multiple streaming video benchmarks show that ThinkStream significantly outperforms existing online video models while maintaining low latency and memory usage. Code, models and data will be released at this https URL 

---
# Surprised by Attention: Predictable Query Dynamics for Time Series Anomaly Detection 

**Authors**: Kadir-Kaan Özer, René Ebeling, Markus Enzweiler  

**Link**: [PDF](https://arxiv.org/pdf/2603.12916)  

**Abstract**: Multivariate time series anomalies often manifest as shifts in cross-channel dependencies rather than simple amplitude excursions. In autonomous driving, for instance, a steering command might be internally consistent but decouple from the resulting lateral acceleration. Residual-based detectors can miss such anomalies when flexible sequence models still reconstruct signals plausibly despite altered coordination. We introduce AxonAD, an unsupervised detector that treats multi-head attention query evolution as a short horizon predictable process. A gradient-updated reconstruction pathway is coupled with a history-only predictor that forecasts future query vectors from past context. This is trained via a masked predictor-target objective against an exponential moving average (EMA) target encoder. At inference, reconstruction error is combined with a tail-aggregated query mismatch score, which measures cosine deviation between predicted and target queries on recent timesteps. This dual approach provides sensitivity to structural dependency shifts while retaining amplitude-level detection. On proprietary in-vehicle telemetry with interval annotations and on the TSB-AD multi-variate suite (17 datasets, 180 series) with threshold-free and range-aware metrics, AxonAD improves ranking quality and temporal localization over strong baselines. Ablations confirm that query prediction and combined scoring are the primary drivers of the observed gains. Code is available at the URL this https URL. 

---
# Stake the Points: Structure-Faithful Instance Unlearning 

**Authors**: Kiseong Hong, JungKyoo Shin, Eunwoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2603.12915)  

**Abstract**: Machine unlearning (MU) addresses privacy risks in pretrained models. The main goal of MU is to remove the influence of designated data while preserving the utility of retained knowledge. Achieving this goal requires preserving semantic relations among retained instances, which existing studies often overlook. We observe that without such preservation, models suffer from progressive structural collapse, undermining both the deletion-retention balance. In this work, we propose a novel structure-faithful framework that introduces stakes, i.e., semantic anchors that serve as reference points to maintain the knowledge structure. By leveraging these anchors, our framework captures and stabilizes the semantic organization of knowledge. Specifically, we instantiate the anchors from language-driven attribute descriptions encoded by a semantic encoder (e.g., CLIP). We enforce preservation of the knowledge structure via structure-aware alignment and regularization: the former aligns the organization of retained knowledge before and after unlearning around anchors, while the latter regulates updates to structure-critical parameters. Results from image classification, retrieval, and face recognition show average gains of 32.9%, 22.5%, and 19.3% in performance, balancing the deletion-retention trade-off and enhancing generalization. 

---
# FedBPrompt: Federated Domain Generalization Person Re-Identification via Body Distribution Aware Visual Prompts 

**Authors**: Xin Xu, Weilong Li, Wei Liu, Wenke Huang, Zhixi Yu, Bin Yang, Xiaoying Liao, Kui Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2603.12912)  

**Abstract**: Federated Domain Generalization for Person Re-Identification (FedDG-ReID) learns domain-invariant representations from decentralized data. While Vision Transformer (ViT) is widely adopted, its global attention often fails to distinguish pedestrians from high similarity backgrounds or diverse viewpoints -- a challenge amplified by cross-client distribution shifts in FedDG-ReID. To address this, we propose Federated Body Distribution Aware Visual Prompt (FedBPrompt), introducing learnable visual prompts to guide Transformer attention toward pedestrian-centric regions. FedBPrompt employs a Body Distribution Aware Visual Prompts Mechanism (BAPM) comprising: Holistic Full Body Prompts to suppress cross-client background noise, and Body Part Alignment Prompts to capture fine-grained details robust to pose and viewpoint variations. To mitigate high communication costs, we design a Prompt-based Fine-Tuning Strategy (PFTS) that freezes the ViT backbone and updates only lightweight prompts, significantly reducing communication overhead while maintaining adaptability. Extensive experiments demonstrate that BAPM effectively enhances feature discrimination and cross-domain generalization, while PFTS achieves notable performance gains within only a few aggregation rounds. Moreover, both BAPM and PFTS can be easily integrated into existing ViT-based FedDG-ReID frameworks, making FedBPrompt a flexible and effective solution for federated person re-identification. The code is available at this https URL. 

---
# Learning from Child-Directed Speech in Two-Language Scenarios: A French-English Case Study 

**Authors**: Liel Binyamin, Elior Sulem  

**Link**: [PDF](https://arxiv.org/pdf/2603.12906)  

**Abstract**: Research on developmentally plausible language models has largely focused on English, leaving open questions about multilingual settings. We present a systematic study of compact language models by extending BabyBERTa to English-French scenarios under strictly size-matched data conditions, covering monolingual, bilingual, and cross-lingual settings. Our design contrasts two types of training corpora: (i) child-directed speech (about 2.5M tokens), following BabyBERTa and related work, and (ii) multi-domain corpora (about 10M tokens), extending the BabyLM framework to French. To enable fair evaluation, we also introduce new resources, including French versions of QAMR and QASRL, as well as English and French multi-domain corpora.
We evaluate the models on both syntactic and semantic tasks and compare them with models trained on Wikipedia-only data. The results reveal context-dependent effects: training on Wikipedia consistently benefits semantic tasks, whereas child-directed speech improves grammatical judgments in monolingual settings. Bilingual pretraining yields notable gains for textual entailment, with particularly strong improvements for French. Importantly, similar patterns emerge across BabyBERTa, RoBERTa, and LTG-BERT, suggesting consistent trends across architectures. 

---
# Human-Centered Evaluation of an LLM-Based Process Modeling Copilot: A Mixed-Methods Study with Domain Experts 

**Authors**: Chantale Lauer, Peter Pfeiffer, Nijat Mehdiyev  

**Link**: [PDF](https://arxiv.org/pdf/2603.12895)  

**Abstract**: Integrating Large Language Models (LLMs) into business process management tools promises to democratize Business Process Model and Notation (BPMN) modeling for non-experts. While automated frameworks assess syntactic and semantic quality, they miss human factors like trust, usability, and professional alignment. We conducted a mixed-methods evaluation of our proposed solution, an LLM-powered BPMN copilot, with five process modeling experts using focus groups and standardized questionnaires. Our findings reveal a critical tension between acceptable perceived usability (mean CUQ score: 67.2/100) and notably lower trust (mean score: 48.8\%), with reliability rated as the most critical concern (M=1.8/5). Furthermore, we identified output-quality issues, prompting difficulties, and a need for the LLM to ask more in-depth clarifying questions about the process. We envision five use cases ranging from domain-expert support to enterprise quality assurance. We demonstrate the necessity of human-centered evaluation complementing automated benchmarking for LLM modeling agents. 

---
# Finite Difference Flow Optimization for RL Post-Training of Text-to-Image Models 

**Authors**: David McAllister, Miika Aittala, Tero Karras, Janne Hellsten, Angjoo Kanazawa, Timo Aila, Samuli Laine  

**Link**: [PDF](https://arxiv.org/pdf/2603.12893)  

**Abstract**: Reinforcement learning (RL) has become a standard technique for post-training diffusion-based image synthesis models, as it enables learning from reward signals to explicitly improve desirable aspects such as image quality and prompt alignment. In this paper, we propose an online RL variant that reduces the variance in the model updates by sampling paired trajectories and pulling the flow velocity in the direction of the more favorable image. Unlike existing methods that treat each sampling step as a separate policy action, we consider the entire sampling process as a single action. We experiment with both high-quality vision language models and off-the-shelf quality metrics for rewards, and evaluate the outputs using a broad set of metrics. Our method converges faster and yields higher output quality and prompt alignment than previous approaches. 

---
# Team LEYA in 10th ABAW Competition: Multimodal Ambivalence/Hesitancy Recognition Approach 

**Authors**: Elena Ryumina, Alexandr Axyonov, Dmitry Sysoev, Timur Abdulkadirov, Kirill Almetov, Yulia Morozova, Dmitry Ryumin  

**Link**: [PDF](https://arxiv.org/pdf/2603.12848)  

**Abstract**: Ambivalence/hesitancy recognition in unconstrained videos is a challenging problem due to the subtle, multimodal, and context-dependent nature of this behavioral state. In this paper, a multimodal approach for video-level ambivalence/hesitancy recognition is presented for the 10th ABAW Competition. The proposed approach integrates four complementary modalities: scene, face, audio, and text. Scene dynamics are captured with a VideoMAE-based model, facial information is encoded through emotional frame-level embeddings aggregated by statistical pooling, acoustic representations are extracted with EmotionWav2Vec2.0 and processed by a Mamba-based temporal encoder, and linguistic cues are modeled using fine-tuned transformer-based text models. The resulting unimodal embeddings are further combined using multimodal fusion models, including prototype-augmented variants. Experiments on the BAH corpus demonstrate clear gains of multimodal fusion over all unimodal baselines. The best unimodal configuration achieved an average MF1 of 70.02%, whereas the best multimodal fusion model reached 83.25%. The highest final test performance, 71.43%, was obtained by an ensemble of five prototype-augmented fusion models. The obtained results highlight the importance of complementary multimodal cues and robust fusion strategies for ambivalence/hesitancy recognition. 

---
# Hierarchical Reference Sets for Robust Unsupervised Detection of Scattered and Clustered Outliers 

**Authors**: Yiqun Zhang, Zexi Tan, Xiaopeng Luo, Yunlin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.12847)  

**Abstract**: Most real-world IoT data analysis tasks, such as clustering and anomaly event detection, are unsupervised and highly susceptible to the presence of outliers. In addition to sporadic scattered outliers caused by factors such as faulty sensor readings, IoT systems often exhibit clustered outliers. These occur when multiple devices or nodes produce similar anomalous measurements, for instance, owing to localized interference, emerging security threats, or regional false alarms, forming micro-clusters. These clustered outliers can be easily mistaken for normal behavior because of their relatively high local density, thereby obscuring the detection of both scattered and contextual anomalies. To address this, we propose a novel outlier detection paradigm that leverages the natural neighboring relationships using graph structures. This facilitates multi-perspective anomaly evaluation by incorporating reference sets at both local and global scales derived from the graph. Our approach enables the effective recognition of scattered outliers without interference from clustered anomalies, whereas the graph structure simultaneously helps reflect and isolate clustered outlier groups. Extensive experiments, including comparative performance analysis, ablation studies, validation on downstream clustering tasks, and evaluation of hyperparameter sensitivity, demonstrate the efficacy of the proposed method. The source code is available at this https URL. 

---
# DAST: A Dual-Stream Voice Anonymization Attacker with Staged Training 

**Authors**: Ridwan Arefeen, Xiaoxiao Miao, Rong Tong, Aik Beng Ng, Simon See, Timothy Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.12840)  

**Abstract**: Voice anonymization masks vocal traits while preserving linguistic content, which may still leak speaker-specific patterns. To assess and strengthen privacy evaluation, we propose a dual-stream attacker that fuses spectral and self-supervised learning features via parallel encoders with a three-stage training strategy. Stage I establishes foundational speaker-discriminative representations. Stage II leverages the shared identity-transformation characteristics of voice conversion and anonymization, exposing the model to diverse converted speech to build cross-system robustness. Stage III provides lightweight adaptation to target anonymized data. Results on the VoicePrivacy Attacker Challenge (VPAC) dataset demonstrate that Stage II is the primary driver of generalization, enabling strong attacking performance on unseen anonymization datasets. With Stage III, fine-tuning on only 10\% of the target anonymization dataset surpasses current state-of-the-art attackers in terms of EER. 

---
# Mask2Flow-TSE: Two-Stage Target Speaker Extraction with Masking and Flow Matching 

**Authors**: Junwon Moon, Hyunjin Choi, Hansol Park, Heeseung Kim, Kyuhong Shim  

**Link**: [PDF](https://arxiv.org/pdf/2603.12837)  

**Abstract**: Target speaker extraction (TSE) extracts the target speaker's voice from overlapping speech mixtures given a reference utterance. Existing approaches typically fall into two categories: discriminative and generative. Discriminative methods apply time-frequency masking for fast inference but often over-suppress the target signal, while generative methods synthesize high-quality speech at the cost of numerous iterative steps. We propose Mask2Flow-TSE, a two-stage framework combining the strengths of both paradigms. The first stage applies discriminative masking for coarse separation, and the second stage employs flow matching to refine the output toward target speech. Unlike generative approaches that synthesize speech from Gaussian noise, our method starts from the masked spectrogram, enabling high-quality reconstruction in a single inference step. Experiments show that Mask2Flow-TSE achieves comparable performance to existing generative TSE methods with approximately 85M parameters. 

---
# Hierarchical Dual-Change Collaborative Learning for UAV Scene Change Captioning 

**Authors**: Fuhai Chen, Pengpeng Huang, Junwen Wu, Hehong Zhang, Shiping Wang, Xiaoguang Ma, Xuri Ge  

**Link**: [PDF](https://arxiv.org/pdf/2603.12832)  

**Abstract**: This paper proposes a novel task for UAV scene understanding - UAV Scene Change Captioning (UAV-SCC) - which aims to generate natural language descriptions of semantic changes in dynamic aerial imagery captured from a movable viewpoint. Unlike traditional change captioning that mainly describes differences between image pairs captured from a fixed camera viewpoint over time, UAV scene change captioning focuses on image-pair differences resulting from both temporal and spatial scene variations dynamically captured by a moving camera. The key challenge lies in understanding viewpoint-induced scene changes from UAV image pairs that share only partially overlapping scene content due to viewpoint shifts caused by camera rotation, while effectively exploiting the relative orientation between the two images. To this end, we propose a Hierarchical Dual-Change Collaborative Learning (HDC-CL) method for UAV scene change captioning. In particular, a novel transformer, \emph{i.e.} Dynamic Adaptive Layout Transformer (DALT) is designed to adaptively model diverse spatial layouts of the image pair, where the interrelated features derived from the overlapping and non-overlapping regions are learned within the flexible and unified encoding layer. Furthermore, we propose a Hierarchical Cross-modal Orientation Consistency Calibration (HCM-OCC) method to enhance the model's sensitivity to viewpoint shift directions, enabling more accurate change captioning. To facilitate in-depth research on this task, we construct a new benchmark dataset, named UAV-SCC dataset, for UAV scene change captioning. Extensive experiments demonstrate that the proposed method achieves state-of-the-art performance on this task. The dataset and code will be publicly released upon acceptance of this paper. 

---
# Residual SODAP: Residual Self-Organizing Domain-Adaptive Prompting with Structural Knowledge Preservation for Continual Learning 

**Authors**: Gyutae Oh, Jungwoo Bae, Jitae Shin  

**Link**: [PDF](https://arxiv.org/pdf/2603.12816)  

**Abstract**: Continual learning (CL) suffers from catastrophic forgetting, which is exacerbated in domain-incremental learning (DIL) where task identifiers are unavailable and storing past data is infeasible. While prompt-based CL (PCL) adapts representations with a frozen backbone, we observe that prompt-only improvements are often insufficient due to suboptimal prompt selection and classifier-level instability under domain shifts. We propose Residual SODAP, which jointly performs prompt-based representation adaptation and classifier-level knowledge preservation. Our framework combines $\alpha$-entmax sparse prompt selection with residual aggregation, data-free distillation with pseudo-feature replay, prompt-usage--based drift detection, and uncertainty-aware multi-loss balancing. Across three DIL benchmarks without task IDs or extra data storage, Residual SODAP achieves state-of-the-art AvgACC/AvgF of 0.850/0.047 (DR), 0.760/0.031 (Skin Cancer), and 0.995/0.003 (CORe50). 

---
# Cheers: Decoupling Patch Details from Semantic Representations Enables Unified Multimodal Comprehension and Generation 

**Authors**: Yichen Zhang, Da Peng, Zonghao Guo, Zijian Zhang, Xuesong Yang, Tong Sun, Shichu Sun, Yidan Zhang, Yanghao Li, Haiyan Zhao, Wang Xu, Qi Shi, Yangang Sun, Chi Chen, Shuo Wang, Yukun Yan, Xu Han, Qiang Ma, Wei Ke, Liang Wang, Zhiyuan Liu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2603.12793)  

**Abstract**: A recent cutting-edge topic in multimodal modeling is to unify visual comprehension and generation within a single model. However, the two tasks demand mismatched decoding regimes and visual representations, making it non-trivial to jointly optimize within a shared feature space. In this work, we present Cheers, a unified multimodal model that decouples patch-level details from semantic representations, thereby stabilizing semantics for multimodal understanding and improving fidelity for image generation via gated detail residuals. Cheers includes three key components: (i) a unified vision tokenizer that encodes and compresses image latent states into semantic tokens for efficient LLM conditioning, (ii) an LLM-based Transformer that unifies autoregressive decoding for text generation and diffusion decoding for image generation, and (iii) a cascaded flow matching head that decodes visual semantics first and then injects semantically gated detail residuals from the vision tokenizer to refine high-frequency content. Experiments on popular benchmarks demonstrate that Cheers matches or surpasses advanced UMMs in both visual understanding and generation. Cheers also achieves 4x token compression, enabling more efficient high-resolution image encoding and generation. Notably, Cheers outperforms the Tar-1.5B on the popular benchmarks GenEval and MMBench, while requiring only 20% of the training cost, indicating effective and efficient (i.e., 4x token compression) unified multimodal modeling. We will release all code and data for future research. 

---
# The RIGID Framework: Research-Integrated, Generative AI-Mediated Instructional Design 

**Authors**: Yerin Kwak, Zachary A. Pardos  

**Link**: [PDF](https://arxiv.org/pdf/2603.12781)  

**Abstract**: Instructional Design (ID) often faces challenges in incorporating research-based knowledge and pedagogical best practices. Although educational researchers and government agencies emphasize grounding ID in evidence, integrating research findings into everyday design workflows is often complex, as it requires considering multiple context-specific demands and constraints. To address this persistent gap, this paper explores how research in the learning sciences (LS) can be systematically integrated across ID workflows and how recent advances in generative AI can help operationalize this integration. While ID and LS share a commitment to improving learning experiences through design-oriented approaches in authentic contexts, structured integration between the two fields remains limited, leaving their complementary insights underutilized. We present RIGID (Research-Integrated, Generative AI-Mediated Instructional Design), a unified framework that integrates LS research across ID workflows spanning analysis, design, implementation, and evaluation phases, while leveraging generative AI to mediate this integration at each stage. The RIGID framework provides a systematic approach for enabling research-integrated instructional design that is both operational and context-sensitive, while preserving the central role of human expertise. 

---
# Empowering Semantic-Sensitive Underwater Image Enhancement with VLM 

**Authors**: Guodong Fan, Shengning Zhou, Genji Yuan, Huiyu Li, Jingchun Zhou, Jinjiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.12773)  

**Abstract**: In recent years, learning-based underwater image enhancement (UIE) techniques have rapidly evolved. However, distribution shifts between high-quality enhanced outputs and natural images can hinder semantic cue extraction for downstream vision tasks, thereby limiting the adaptability of existing enhancement models. To address this challenge, this work proposes a new learning mechanism that leverages Vision-Language Models (VLMs) to empower UIE models with semantic-sensitive capabilities. To be concrete, our strategy first generates textual descriptions of key objects from a degraded image via VLMs. Subsequently, a text-image alignment model remaps these relevant descriptions back onto the image to produce a spatial semantic guidance map. This map then steers the UIE network through a dual-guidance mechanism, which combines cross-attention and an explicit alignment loss. This forces the network to focus its restorative power on semantic-sensitive regions during image reconstruction, rather than pursuing a globally uniform improvement, thereby ensuring the faithful restoration of key object features. Experiments confirm that when our strategy is applied to different UIE baselines, significantly boosts their performance on perceptual quality metrics as well as enhances their performance on detection and segmentation tasks, validating its effectiveness and adaptability. 

---
# FC-Track: Overlap-Aware Post-Association Correction for Online Multi-Object Tracking 

**Authors**: Cheng Ju, Zejing Zhao, Akio Namiki  

**Link**: [PDF](https://arxiv.org/pdf/2603.12758)  

**Abstract**: Reliable multi-object tracking (MOT) is essential for robotic systems operating in complex and dynamic environments. Despite recent advances in detection and association, online MOT methods remain vulnerable to identity switches caused by frequent occlusions and object overlap, where incorrect associations can propagate over time and degrade tracking reliability. We present a lightweight post-association correction framework (FC-Track) for online MOT that explicitly targets overlap-induced mismatches during inference. The proposed method suppresses unreliable appearance updates under high-overlap conditions using an Intersection over Area (IoA)-based filtering strategy, and locally corrects detection-to-tracklet mismatches through appearance similarity comparison within overlapped tracklet pairs. By preventing short-term mismatches from propagating, our framework effectively mitigates long-term identity switches without resorting to global optimization or re-identification. The framework operates online without global optimization or re-identification, making it suitable for real-time robotic applications. We achieve 81.73 MOTA, 82.81 IDF1, and 66.95 HOTA on the MOT17 test set with a running speed of 5.7 FPS, and 77.52 MOTA, 80.90 IDF1, and 65.67 HOTA on the MOT20 test set with a running speed of 0.6 FPS. Specifically, our framework FC-Track produces only 29.55% long-term identity switches, which is substantially lower than existing online trackers. Meanwhile, our framework maintains state-of-the-art performance on the MOT20 benchmark. 

---
# TaoBench: Do Automated Theorem Prover LLMs Generalize Beyond MathLib? 

**Authors**: Alexander K Taylor, Junyi Zhang, Ethan Ji, Vigyan Sahai, Haikang Deng, Yuanzhou Chen, Yifan Yuan, Di Wu, Jia-Chen Gu, Kai-Wei Chang, Nanyun Peng, Amit Sahai, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.12744)  

**Abstract**: Automated theorem proving (ATP) benchmarks largely consist of problems formalized in MathLib, so current ATP training and evaluation are heavily biased toward MathLib's definitional framework. However, frontier mathematics is often exploratory and prototype-heavy, relying on bespoke constructions that deviate from standard libraries. In this work, we evaluate the robustness of current ATP systems when applied to a novel definitional framework, specifically examining the performance gap between standard library problems and bespoke mathematical constructions. We introduce TaoBench, an undergraduate-level benchmark derived from Terence Tao's Analysis I, which formalizes analysis by constructing core mathematical concepts from scratch, without relying on standard Mathlib definitions, as well as by mixing from-scratch and MathLib constructions. For fair evaluation, we build an agentic pipeline that automatically extracts a compilable, self-contained local environment for each problem. To isolate the effect of definitional frameworks, we additionally translate every problem into a mathematically equivalent Mathlib formulation, yielding paired TaoBench-Mathlib statements for direct comparison. While state-of-the-art ATP models perform capably within the MathLib framework, performance drops by an average of roughly 26% on the definitionally equivalent Tao formulation. This indicates that the main bottleneck is limited generalization across definitional frameworks rather than task difficulty. TaoBench thus highlights a gap between benchmark performance and applicability, and provides a concrete foundation for developing and testing provers better aligned with research mathematics. 

---
# MoKus: Leveraging Cross-Modal Knowledge Transfer for Knowledge-Aware Concept Customization 

**Authors**: Chenyang Zhu, Hongxiang Li, Xiu Li, Long Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.12743)  

**Abstract**: Concept customization typically binds rare tokens to a target concept. Unfortunately, these approaches often suffer from unstable performance as the pretraining data seldom contains these rare tokens. Meanwhile, these rare tokens fail to convey the inherent knowledge of the target concept. Consequently, we introduce Knowledge-aware Concept Customization, a novel task aiming at binding diverse textual knowledge to target visual concepts. This task requires the model to identify the knowledge within the text prompt to perform high-fidelity customized generation. Meanwhile, the model should efficiently bind all the textual knowledge to the target concept. Therefore, we propose MoKus, a novel framework for knowledge-aware concept customization. Our framework relies on a key observation: cross-modal knowledge transfer, where modifying knowledge within the text modality naturally transfers to the visual modality during generation. Inspired by this observation, MoKus contains two stages: (1) In visual concept learning, we first learn the anchor representation to store the visual information of the target concept. (2) In textual knowledge updating, we update the answer for the knowledge queries to the anchor representation, enabling high-fidelity customized generation. To further comprehensively evaluate our proposed MoKus on the new task, we introduce the first benchmark for knowledge-aware concept customization: KnowCusBench. Extensive evaluations have demonstrated that MoKus outperforms state-of-the-art methods. Moreover, the cross-model knowledge transfer allows MoKus to be easily extended to other knowledge-aware applications like virtual concept creation and concept erasure. We also demonstrate the capability of our method to achieve improvements on world knowledge benchmarks. 

---
# SRAM-Based Compute-in-Memory Accelerator for Linear-decay Spiking Neural Networks 

**Authors**: Hongyang Shang, Shuai Dong, Yahan Yang, Junyi Yang, Peng Zhou, Arindam Basu  

**Link**: [PDF](https://arxiv.org/pdf/2603.12739)  

**Abstract**: Spiking Neural Networks (SNNs) have emerged as a biologically inspired alternative to conventional deep networks, offering event-driven and energy-efficient computation. However, their throughput remains constrained by the serial update of neuron membrane states. While many hardware accelerators and Compute-in-Memory (CIM) architectures efficiently parallelize the synaptic operation (W x I) achieving O(1) complexity for matrix-vector multiplication, the subsequent state update step still requires O(N) time to refresh all neuron membrane potentials. This mismatch makes state update the dominant latency and energy bottleneck in SNN inference. To address this challenge, we propose an SRAM-based CIM for SNN with Linear Decay Leaky Integrate-and-Fire (LD-LIF) Neuron that co-optimizes algorithm and hardware. At the algorithmic level, we replace the conventional exponential membrane decay with a linear decay approximation, converting costly multiplications into simple additions while accuracy drops only around 1%. At the architectural level, we introduce an in-memory parallel update scheme that performs in-place decay directly within the SRAM array, eliminating the need for global sequential updates. Evaluated on benchmark SNN workloads, the proposed method achieves a 1.1 x to 16.7 x reduction of SOP energy consumption, while providing 15.9 x to 69 x more energy efficiency, with negligible accuracy loss relative to original decay models. This work highlights that beyond accelerating the (W x I) computation, optimizing state-update dynamics within CIM architectures is essential for scalable, low-power, and real-time neuromorphic processing. 

---
# Graph In-Context Operator Networks for Generalizable Spatiotemporal Prediction 

**Authors**: Chenghan Wu, Zongmin Yu, Boai Sun, Liu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.12725)  

**Abstract**: In-context operator learning enables neural networks to infer solution operators from contextual examples without weight updates. While prior work has demonstrated the effectiveness of this paradigm in leveraging vast datasets, a systematic comparison against single-operator learning using identical training data has been absent. We address this gap through controlled experiments comparing in-context operator learning against classical operator learning (single-operator models trained without contextual examples), under the same training steps and dataset. To enable this investigation on real-world spatiotemporal systems, we propose GICON (Graph In-Context Operator Network), combining graph message passing for geometric generalization with example-aware positional encoding for cardinality generalization. Experiments on air quality prediction across two Chinese regions show that in-context operator learning outperforms classical operator learning on complex tasks, generalizing across spatial domains and scaling robustly from few training examples to 100 at inference. 

---
# CognitionCapturerPro: Towards High-Fidelity Visual Decoding from EEG/MEG via Multi-modal Information and Asymmetric Alignment 

**Authors**: Kaifan Zhang, Lihuo He, Junjie Ke, Yuqi Ji, Lukun Wu, Lizi Wang, Xinbo Gao  

**Link**: [PDF](https://arxiv.org/pdf/2603.12722)  

**Abstract**: Visual stimuli reconstruction from EEG remains challenging due to fidelity loss and representation shift. We propose CognitionCapturerPro, an enhanced framework that integrates EEG with multi-modal priors (images, text, depth, and edges) via collaborative training. Our core contributions include an uncertainty-weighted similarity scoring mechanism to quantify modality-specific fidelity and a fusion encoder for integrating shared representations. By employing a simplified alignment module and a pre-trained diffusion model, our method significantly outperforms the original CognitionCapturer on the THINGS-EEG dataset, improving Top-1 and Top-5 retrieval accuracy by 25.9% and 10.6%, respectively. Code is available at: this https URL. 

---
# CMHANet: A Cross-Modal Hybrid Attention Network for Point Cloud Registration 

**Authors**: Dongxu Zhang, Yingsen Wang, Yiding Sun, Haoran Xu, Peilin Fan, Jihua Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2603.12721)  

**Abstract**: Robust point cloud registration is a fundamental task in 3D computer vision and geometric deep learning, essential for applications such as large-scale 3D reconstruction, augmented reality, and scene understanding. However, the performance of established learning-based methods often degrades in complex, real world scenarios characterized by incomplete data, sensor noise, and low overlap regions. To address these limitations, we propose CMHANet, a novel Cross-Modal Hybrid Attention Network. Our method integrates the fusion of rich contextual information from 2D images with the geometric detail of 3D point clouds, yielding a comprehensive and resilient feature representation. Furthermore, we introduce an innovative optimization function based on contrastive learning, which enforces geometric consistency and significantly improves the model's robustness to noise and partial observations. We evaluated CMHANet on the 3DMatch and the challenging 3DLoMatch datasets. \rev{Additionally, zero-shot evaluations on the TUM RGB-D SLAM dataset verify the model's generalization capability to unseen domains.} The experimental results demonstrate that our method achieves substantial improvements in both registration accuracy and overall robustness, outperforming current techniques. We also release our code in \href{this https URL}{this https URL}. 

---
# IGASA: Integrated Geometry-Aware and Skip-Attention Modules for Enhanced Point Cloud Registration 

**Authors**: Dongxu Zhang, Jihua Zhu, Shiqi Li, Wenbiao Yan, Haoran Xu, Peilin Fan, Huimin Lu  

**Link**: [PDF](https://arxiv.org/pdf/2603.12719)  

**Abstract**: Point cloud registration (PCR) is a fundamental task in 3D vision and provides essential support for applications such as autonomous driving, robotics, and environmental modeling. Despite its widespread use, existing methods often fail when facing real-world challenges like heavy noise, significant occlusions, and large-scale transformations. These limitations frequently result in compromised registration accuracy and insufficient robustness in complex environments. In this paper, we propose IGASA as a novel registration framework constructed upon a Hierarchical Pyramid Architecture (HPA) designed for robust multi-scale feature extraction and fusion. The framework integrates two pivotal components consisting of the Hierarchical Cross-Layer Attention (HCLA) module and the Iterative Geometry-Aware Refinement (IGAR) module. The HCLA module utilizes skip attention mechanisms to align multi-resolution features and enhance local geometric consistency. Simultaneously, the IGAR module is designed for the fine matching phase by leveraging reliable correspondences established during coarse matching. This synergistic integration within the architecture allows IGASA to adapt effectively to diverse point cloud structures and intricate transformations. We evaluate the performance of IGASA on four widely recognized benchmark datasets including 3D(Lo)Match, KITTI, and nuScenes. Our extensive experiments consistently demonstrate that IGASA significantly surpasses state-of-the-art methods and achieves notable improvements in registration accuracy. This work provides a robust foundation for advancing point cloud registration techniques while offering valuable insights for practical 3D vision applications. The code for IGASA is available in \href{this https URL}{this https URL}. 

---
# Altered Thoughts, Altered Actions: Probing Chain-of-Thought Vulnerabilities in VLA Robotic Manipulation 

**Authors**: Tuan Duong Trinh, Naveed Akhtar, Basim Azam  

**Link**: [PDF](https://arxiv.org/pdf/2603.12717)  

**Abstract**: Recent Vision-Language-Action (VLA) models increasingly adopt chain-of-thought (CoT) reasoning, generating a natural-language plan before decoding motor commands. This internal text channel between the reasoning module and the action decoder has received no adversarial scrutiny. We ask: which properties of this intermediate plan does the action decoder actually rely on, and can targeted corruption of the reasoning trace alone -- with all inputs left intact -- degrade a robot's physical task performance? We design a taxonomy of seven text corruptions organized into three attacker tiers (blind noise, mechanical-semantic, and LLM-adaptive) and apply them to a state-of-the-art reasoning VLA across 40 LIBERO tabletop manipulation tasks. Our results reveal a striking asymmetry: substituting object names in the reasoning trace reduces overall success rate by 8.3~percentage points (pp) -- reaching $-$19.3~pp on goal-conditioned tasks and $-$45~pp on individual tasks -- whereas sentence reordering, spatial-direction reversal, token noise, and even a 70B-parameter LLM crafting plausible-but-wrong plans all have negligible impact (within $\pm$4~pp). This asymmetry indicates that the action decoder depends on entity-reference integrity rather than reasoning quality or sequential structure. Notably, a sophisticated LLM-based attacker underperforms simple mechanical object-name substitution, because preserving plausibility inadvertently retains the entity-grounding structure the decoder needs. A cross-architecture control using a non-reasoning VLA confirms the vulnerability is exclusive to reasoning-augmented models, while instruction-level attacks degrade both architectures -- establishing that the internal reasoning trace is a distinct and stealthy threat vector invisible to input-validation defenses. 

---
# Cost-Efficient Multimodal LLM Inference via Cross-Tier GPU Heterogeneity 

**Authors**: Donglin Yu  

**Link**: [PDF](https://arxiv.org/pdf/2603.12707)  

**Abstract**: Multimodal large language model (MLLM) inference splits into two phases with opposing hardware demands: vision encoding is compute-bound, while language generation is memory-bandwidth-bound. We show that under standard transformer KV caching, the modality boundary (between vision encoder and language model) minimizes cross-device transfer among all partition points that preserve standard stage-based execution. Partitioning here reduces transfer complexity from $O(L * s_ctx)$ bytes (GB-scale KV caches under stage-level disaggregation) to $O(N_v * d)$ bytes (MB-scale embeddings), an O(L) reduction where L is the transformer depth. The result holds across attention mechanisms (MHA/GQA), dynamic vision resolutions, and model scales, and the advantage grows as models deepen. A direct implication is that existing stage-level disaggregation systems are constrained to high-bandwidth interconnects (e.g., NVLink), whereas modality-level disaggregation enables cross-tier heterogeneous serving over commodity PCIe. A closed-form cost model shows that heterogeneous deployment is cost-optimal under phase-separable workloads (predicts 31.4% savings; observed 40.6%). We build HeteroServe, a phase-aware runtime with modality-level partitioning and cross-tier scheduling, and evaluate it on LLaVA-1.5-7B and Qwen2.5-VL against vLLM v0.3.0. On identical 4xA100 hardware, engine optimizations raise throughput by up to 54%. Under a fixed budget, a heterogeneous cluster (\$38k) improves Tokens/\$ by 37% over a homogeneous baseline (\$64k) without degrading latency. 

---
# Seeing Eye to Eye: Enabling Cognitive Alignment Through Shared First-Person Perspective in Human-AI Collaboration 

**Authors**: Zhuyu Teng, Pei Chen, Yichen Cai, Ruoqing Lu, Zhaoqu Jiang, Jiayang Li, Weitao You, Lingyun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2603.12701)  

**Abstract**: Despite advances in multimodal AI, current vision-based assistants often remain inefficient in collaborative tasks. We identify two key gulfs: a communication gulf, where users must translate rich parallel intentions into verbal commands due to the channel mismatch , and an understanding gulf, where AI struggles to interpret subtle embodied cues. To address these, we propose Eye2Eye, a framework that leverages first-person perspective as a channel for human-AI cognitive alignment. It integrates three components: (1) joint attention coordination for fluid focus alignment, (2) revisable memory to maintain evolving common ground, and (3) reflective feedback allowing users to clarify and refine AI's understanding. We implement this framework in an AR prototype and evaluate it through a user study and a post-hoc pipeline evaluation. Results show that Eye2Eye significantly reduces task completion time and interaction load while increasing trust, demonstrating its components work in concert to improve collaboration. 

---
# HSEmotion Team at ABAW-10 Competition: Facial Expression Recognition, Valence-Arousal Estimation, Action Unit Detection and Fine-Grained Violence Classification 

**Authors**: Andrey V. Savchenko, Kseniia Tsypliakova  

**Link**: [PDF](https://arxiv.org/pdf/2603.12693)  

**Abstract**: This article presents our results for the 10th Affective Behavior Analysis in-the-Wild (ABAW) competition. For frame-wise facial emotion understanding tasks (frame-wise facial expression recognition, valence-arousal estimation, action unit detection), we propose a fast approach based on facial embedding extraction with pre-trained EfficientNet-based emotion recognition models. If the latter model's confidence exceeds a threshold, its prediction is used. Otherwise, we feed embeddings into a simple multi-layered perceptron trained on the AffWild2 dataset. Estimated class-level scores are smoothed in a sliding window of fixed size to mitigate noise in frame-wise predictions. For the fine-grained violence detection task, we examine several pre-trained architectures for frame embeddings and their aggregation for video classification. Experimental results on four tasks from the ABAW challenge demonstrate that our approach significantly improves validation metrics over existing baselines. 

---
# Federated Hierarchical Clustering with Automatic Selection of Optimal Cluster Numbers 

**Authors**: Yue Zhang, Chuanlong Qiu, Xinfa Liao, Yiqun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.12684)  

**Abstract**: Federated Clustering (FC) is an emerging and promising solution in exploring data distribution patterns from distributed and privacy-protected data in an unsupervised manner. Existing FC methods implicitly rely on the assumption that clients are with a known number of uniformly sized clusters. However, the true number of clusters is typically unknown, and cluster sizes are naturally imbalanced in real scenarios. Furthermore, the privacy-preserving transmission constraints in federated learning inevitably reduce usable information, making the development of robust and accurate FC extremely challenging. Accordingly, we propose a novel FC framework named Fed-$k^*$-HC, which can automatically determine an optimal number of clusters $k^*$ based on the data distribution explored through hierarchical clustering. To obtain the global data distribution for $k^*$ determination, we let each client generate micro-subclusters. Their prototypes are then uploaded to the server for hierarchical merging. The density-based merging design allows exploring clusters of varying sizes and shapes, and the progressive merging process can self-terminate according to the neighboring relationships among the prototypes to determine $k^*$. Extensive experiments on diverse datasets demonstrate the FC capability of the proposed Fed-$k^*$-HC in accurately exploring a proper number of clusters. 

---
# Experimental evidence of progressive ChatGPT models self-convergence 

**Authors**: Konstantinos F. Xylogiannopoulos, Petros Xanthopoulos, Panagiotis Karampelas, Georgios A. Bakamitsos  

**Link**: [PDF](https://arxiv.org/pdf/2603.12683)  

**Abstract**: Large Language Models (LLMs) that undergo recursive training on synthetically generated data are susceptible to model collapse, a phenomenon marked by the generation of meaningless output. Existing research has examined this issue from either theoretical or empirical perspectives, often focusing on a single model trained recursively on its own outputs. While prior studies have cautioned against the potential degradation of LLM output quality under such conditions, no longitudinal investigation has yet been conducted to assess this effect over time. In this study, we employ a text similarity metric to evaluate different ChatGPT models' capacity to generate diverse textual outputs. Our findings indicate a measurable decline of recent ChatGPT releases' ability to produce varied text, even when explicitly prompted to do so, by setting the temperature parameter to one. The observed reduction in output diversity may be attributed to the influence of the amounts of synthetic data incorporated within their training datasets as the result of internet infiltration by LLM generated data. The phenomenon is defined as model self-convergence because of the gradual increase of similarities of produced texts among different ChatGPT versions. 

---
# MetaKE: Meta-learning Aligned Knowledge Editing via Bi-level Optimization 

**Authors**: Shuxin Liu, Ou Wu  

**Link**: [PDF](https://arxiv.org/pdf/2603.12677)  

**Abstract**: Knowledge editing (KE) aims to precisely rectify specific knowledge in Large Language Models (LLMs) without disrupting general capabilities. State-of-the-art methods suffer from an open-loop control mismatch. We identify a critical "Semantic-Execution Disconnect": the semantic target is derived independently without feedback from the downstream's feasible region. This misalignment often causes valid semantic targets to fall within the prohibited space, resulting in gradient truncation and editing failure. To bridge this gap, we propose MetaKE (Meta-learning Aligned Knowledge Editing), a new framework that reframes KE as a bi-level optimization problem. Departing from static calculation, MetaKE treats the edit target as a learnable meta-parameter: the upper-level optimizer seeks a feasible target to maximize post-edit performance, while the lower-level solver executes the editing. To address the challenge of differentiating through complex solvers, we derive a Structural Gradient Proxy, which explicitly backpropagates editability constraints to the target learning phase. Theoretical analysis demonstrates that MetaKE automatically aligns the edit direction with the model's feasible manifold. Extensive experiments confirm that MetaKE significantly outperforms strong baselines, offering a new perspective on knowledge editing. 

---
# Marker-Based 3D Reconstruction of Aggregates with a Comparative Analysis of 2D and 3D Morphologies 

**Authors**: Haohang Huang, Jiayi Luo, Issam Qamhia, Erol Tutumluer, John M. Hart, Andrew J. Stolba  

**Link**: [PDF](https://arxiv.org/pdf/2603.12667)  

**Abstract**: Aggregates, serving as the main skeleton in assemblies of construction materials, are important functional components in various building and transportation infrastructures. They can be used in unbound layer applications, e.g. pavement base and railroad ballast, bound applications of cement concrete and asphalt concrete, and as riprap and large-sized primary crushed rocks. Information on the size and shape or morphology of aggregates can greatly facilitate the Quality Assurance/Quality Control (QA/QC) process by providing insights of aggregate behavior during composition and packing. A full 3D characterization of aggregate particle morphology is difficult both during production in a quarry and at a construction site. Many aggregate imaging approaches have been developed to quantify the particle morphology by computer vision, including 2D image-based approaches that analyze particle silhouettes and 3D scanning-based methods that require expensive devices such as 3D laser scanners or X-Ray Computed Tomography (CT) equipment. This paper presents a flexible and cost-effective photogrammetry-based approach for the 3D reconstruction of aggregate particles. The proposed approach follows a marker-based design that enables background suppression, point cloud stitching, and scale referencing to obtain high-quality aggregate models. The accuracy of the reconstruction results was validated against ground-truth for selected aggregate samples. Comparative analyses were conducted on 2D and 3D morphological properties of the selected samples. Significant differences were found between the 2D and 3D statistics. Based on the presented approach, 3D shape information of aggregates can be obtained easily and at a low cost, thus allowing convenient aggregate inspection, data collection, and 3D morphological analysis. 

---
# RetroReasoner: A Reasoning LLM for Strategic Retrosynthesis Prediction 

**Authors**: Hanbum Ko, Chanhui Lee, Ye Rin Kim, Rodrigo Hormazabal, Sehui Han, Sungbin Lim, Sungwoong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2603.12666)  

**Abstract**: Retrosynthesis prediction is a core task in organic synthesis that aims to predict reactants for a given product molecule. Traditionally, chemists select a plausible bond disconnection and derive corresponding reactants, which is time-consuming and requires substantial expertise. While recent advancements in molecular large language models (LLMs) have made progress, many methods either predict reactants without strategic reasoning or conduct only a generic product analysis, rather than reason explicitly about bond-disconnection strategies that logically lead to the choice of specific reactants. To overcome these limitations, we propose RetroReasoner, a retrosynthetic reasoning model that leverages chemists' strategic thinking. RetroReasoner is trained using both supervised fine-tuning (SFT) and reinforcement learning (RL). For SFT, we introduce SyntheticRetro, a framework that generates structured disconnection rationales alongside reactant predictions. In the case of RL, we apply a round-trip accuracy as reward, where predicted reactants are passed through a forward synthesis model, and predictions are rewarded when the forward-predicted product matches the original input product. Experimental results show that RetroReasoner not only outperforms prior baselines but also generates a broader range of feasible reactant proposals, particularly in handling more challenging reaction instances. 

---
# From Text to Forecasts: Bridging Modality Gap with Temporal Evolution Semantic Space 

**Authors**: Lehui Li, Yuyao Wang, Jisheng Yan, Wei Zhang, Jinliang Deng, Haoliang Sun, Zhongyi Han, Yongshun Gong  

**Link**: [PDF](https://arxiv.org/pdf/2603.12664)  

**Abstract**: Incorporating textual information into time-series forecasting holds promise for addressing event-driven non-stationarity; however, a fundamental modality gap hinders effective fusion: textual descriptions express temporal impacts implicitly and qualitatively, whereas forecasting models rely on explicit and quantitative signals. Through controlled semi-synthetic experiments, we show that existing methods over-attend to redundant tokens and struggle to reliably translate textual semantics into usable numerical cues. To bridge this gap, we propose TESS, which introduces a Temporal Evolution Semantic Space as an intermediate bottleneck between modalities. This space consists of interpretable, numerically grounded temporal primitives (mean shift, volatility, shape, and lag) extracted from text by an LLM via structured prompting and filtered through confidence-aware gating. Experiments on four real-world datasets demonstrate up to a 29 percent reduction in forecasting error compared to state-of-the-art unimodal and multimodal baselines. The code will be released after acceptance. 

---
# Continual Learning in Large Language Models: Methods, Challenges, and Opportunities 

**Authors**: Hongyang Chen, Zhongwu Sun, Hongfei Ye, Kunchi Li, Xuemin Lin  

**Link**: [PDF](https://arxiv.org/pdf/2603.12658)  

**Abstract**: Continual learning (CL) has emerged as a pivotal paradigm to enable large language models (LLMs) to dynamically adapt to evolving knowledge and sequential tasks while mitigating catastrophic forgetting-a critical limitation of the static pre-training paradigm inherent to modern LLMs. This survey presents a comprehensive overview of CL methodologies tailored for LLMs, structured around three core training stages: continual pre-training, continual fine-tuning, and continual this http URL the canonical taxonomy of rehearsal-, regularization-, and architecture-based methods, we further subdivide each category by its distinct forgetting mitigation mechanisms and conduct a rigorous comparative analysis of the adaptability and critical improvements of traditional CL methods for LLMs. In doing so, we explicitly highlight core distinctions between LLM CL and traditional machine learning, particularly with respect to scale, parameter efficiency, and emergent capabilities. Our analysis covers essential evaluation metrics, including forgetting rates and knowledge transfer efficiency, along with emerging benchmarks for assessing CL performance. This survey reveals that while current methods demonstrate promising results in specific domains, fundamental challenges persist in achieving seamless knowledge integration across diverse tasks and temporal scales. This systematic review contributes to the growing body of knowledge on LLM adaptation, providing researchers and practitioners with a structured framework for understanding current achievements and future opportunities in lifelong learning for language models. 

---
# LR-SGS: Robust LiDAR-Reflectance-Guided Salient Gaussian Splatting for Self-Driving Scene Reconstruction 

**Authors**: Ziyu Chen, Fan Zhu, Hui Zhu, Deyi Kong, Xinkai Kuang, Yujia Zhang, Chunmao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2603.12647)  

**Abstract**: Recent 3D Gaussian Splatting (3DGS) methods have demonstrated the feasibility of self-driving scene reconstruction and novel view synthesis. However, most existing methods either rely solely on cameras or use LiDAR only for Gaussian initialization or depth supervision, while the rich scene information contained in point clouds, such as reflectance, and the complementarity between LiDAR and RGB have not been fully exploited, leading to degradation in challenging self-driving scenes, such as those with high ego-motion and complex lighting. To address these issues, we propose a robust and efficient LiDAR-reflectance-guided Salient Gaussian Splatting method (LR-SGS) for self-driving scenes, which introduces a structure-aware Salient Gaussian representation, initialized from geometric and reflectance feature points extracted from LiDAR and refined through a salient transform and improved density control to capture edge and planar structures. Furthermore, we calibrate LiDAR intensity into reflectance and attach it to each Gaussian as a lighting-invariant material channel, jointly aligned with RGB to enforce boundary consistency. Extensive experiments on the Waymo Open Dataset demonstrate that LR-SGS achieves superior reconstruction performance with fewer Gaussians and shorter training time. In particular, on Complex Lighting scenes, our method surpasses OmniRe by 1.18 dB PSNR. 

---
# LightMoE: Reducing Mixture-of-Experts Redundancy through Expert Replacing 

**Authors**: Jiawei Hao, Zhiwei Hao, Jianyuan Guo, Li Shen, Yong Luo, Han Hu, Dan Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2603.12645)  

**Abstract**: Mixture-of-Experts (MoE) based Large Language Models (LLMs) have demonstrated impressive performance and computational efficiency. However, their deployment is often constrained by substantial memory demands, primarily due to the need to load numerous expert modules. While existing expert compression techniques like pruning or merging attempt to mitigate this, they often suffer from irreversible knowledge loss or high training overhead. In this paper, we propose a novel expert compression paradigm termed expert replacing, which replaces redundant experts with parameter-efficient modules and recovers their capabilities with low training costs. We find that even a straightforward baseline of this paradigm yields promising performance. Building on this foundation, we introduce LightMoE, a framework that enhances the paradigm by introducing adaptive expert selection, hierarchical expert construction, and an annealed recovery strategy. Experimental results show that LightMoE matches the performance of LoRA fine-tuning at a 30% compression ratio. Even under a more aggressive 50% compression rate, it outperforms existing methods and achieves average performance improvements of 5.6% across five diverse tasks. These findings demonstrate that LightMoE strikes a superior balance among memory efficiency, training efficiency, and model performance. 

---
# Spend Less, Reason Better: Budget-Aware Value Tree Search for LLM Agents 

**Authors**: Yushu Li, Wenlong Deng, Jiajin Li, Xiaoxiao Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.12634)  

**Abstract**: Test-time scaling has become a dominant paradigm for improving LLM agent reliability, yet current approaches treat compute as an abundant resource, allowing agents to exhaust token and tool budgets on redundant steps or dead-end trajectories. Existing budget-aware methods either require expensive fine-tuning or rely on coarse, trajectory-level heuristics that cannot intervene mid-execution. We propose the Budget-Aware Value Tree (BAVT), a training-free inference-time framework that models multi-hop reasoning as a dynamic search tree guided by step-level value estimation within a single LLM backbone. Another key innovation is a budget-conditioned node selection mechanism that uses the remaining resource ratio as a natural scaling exponent over node values, providing a principled, parameter-free transition from broad exploration to greedy exploitation as the budget depletes. To combat the well-known overconfidence of LLM self-evaluation, BAVT employs a residual value predictor that scores relative progress rather than absolute state quality, enabling reliable pruning of uninformative or redundant tool calls. We further provide a theoretical convergence guarantee, proving that BAVT reaches a terminal answer with probability at least $1-\epsilon$ under an explicit finite budget bound. Extensive evaluations on four multi-hop QA benchmarks across two model families demonstrate that BAVT consistently outperforms parallel sampling baselines. Most notably, BAVT under strict low-budget constraints surpasses baseline performance at $4\times$ the resource allocation, establishing that intelligent budget management fundamentally outperforms brute-force compute scaling. 

---
# The Economics of AI Supply Chain Regulation 

**Authors**: Sihan Qian, Amit Mehra, Dengpan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.12630)  

**Abstract**: The rise of foundation models has driven the emergence of AI supply chains, where upstream foundation model providers offer fine-tuning and inference services to downstream firms developing domain-specific applications. Downstream firms pay providers to use their computing infrastructure to fine-tune models with proprietary data, creating a co-creation dynamic that enhances model quality. Amid concerns that foundation model providers and downstream firms may capture excessive consumer surplus, along with increasing regulatory measures, this study employs a game-theoretic model involving a provider and two competing downstream firms to analyze how policy interventions affect consumer surplus in the AI supply chain. Our analysis shows that policies promoting price competition in downstream markets (i.e., pro-price-competitive policies) boost consumer surplus only when compute or data preprocessing costs are high, while compute subsidies are effective only when these costs are low, suggesting these policies complement each other. In contrast, policies promoting quality competition in downstream markets (i.e., pro-quality-competitive policies) always improve consumer surplus. We also find that under pro-price-competitive policies or compute subsidies, both the provider and downstream firms can achieve higher profits along with greater consumer surplus, creating a win-win-win outcome. However, pro-quality-competitive policies increase the provider's profits while reducing those of downstream firms. Finally, as compute costs decline, pro-price-competitive policies may lose their effectiveness, whereas compute subsidies may shift from ineffective to effective. These findings offer insights for policymakers seeking to foster AI supply chains that are economically efficient and socially beneficial. 

---
# Towards unified brain-to-text decoding across speech production and perception 

**Authors**: Zhizhang Yuan, Yang Yang, Gaorui Zhang, Baowen Cheng, Zehan Wu, Yuhao Xu, Xiaoying Liu, Liang Chen, Ying Mao, Meng Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.12628)  

**Abstract**: Speech production and perception are the main ways humans communicate daily. Prior brain-to-text decoding studies have largely focused on a single modality and alphabetic languages. Here, we present a unified brain-to-sentence decoding framework for both speech production and perception in Mandarin Chinese. The framework exhibits strong generalization ability, enabling sentence-level decoding when trained only on single-character data and supporting characters and syllables unseen during training. In addition, it allows direct and controlled comparison of neural dynamics across modalities. Mandarin speech is decoded by first classifying syllable components in Hanyu Pinyin, namely initials and finals, from neural signals, followed by a post-trained large language model (LLM) that maps sequences of toneless Pinyin syllables to Chinese sentences. To enhance LLM decoding, we designed a three-stage post-training and two-stage inference framework based on a 7-billion-parameter LLM, achieving overall performance that exceeds larger commercial LLMs with hundreds of billions of parameters or more. In addition, several characteristics were observed in Mandarin speech production and perception: speech production involved neural responses across broader cortical regions than auditory perception; channels responsive to both modalities exhibited similar activity patterns, with speech perception showing a temporal delay relative to production; and decoding performance was broadly comparable across hemispheres. Our work not only establishes the feasibility of a unified decoding framework but also provides insights into the neural characteristics of Mandarin speech production and perception. These advances contribute to brain-to-text decoding in logosyllabic languages and pave the way toward neural language decoding systems supporting multiple modalities. 

---
# VLM4Rec: Multimodal Semantic Representation for Recommendation with Large Vision-Language Models 

**Authors**: Ty Valencia, Burak Barlas, Varun Singhal, Ruchir Bhatia, Wei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.12625)  

**Abstract**: Multimodal recommendation is commonly framed as a feature fusion problem, where textual and visual signals are combined to better model user preference. However, the effectiveness of multimodal recommendation may depend not only on how modalities are fused, but also on whether item content is represented in a semantic space aligned with preference matching. This issue is particularly important because raw visual features often preserve appearance similarity, while user decisions are typically driven by higher-level semantic factors such as style, material, and usage context. Motivated by this observation, we propose LVLM-grounded Multimodal Semantic Representation for Recommendation (VLM4Rec), a lightweight framework that organizes multimodal item content through semantic alignment rather than direct feature fusion. VLM4Rec first uses a large vision-language model to ground each item image into an explicit natural-language description, and then encodes the grounded semantics into dense item representations for preference-oriented retrieval. Recommendation is subsequently performed through a simple profile-based semantic matching mechanism over historical item embeddings, yielding a practical offline-online decomposition. Extensive experiments on multiple multimodal recommendation datasets show that VLM4Rec consistently improves performance over raw visual features and several fusion-based alternatives, suggesting that representation quality may matter more than fusion complexity in this setting. The code is released at this https URL. 

---
# When Drafts Evolve: Speculative Decoding Meets Online Learning 

**Authors**: Yu-Yang Qian, Hao-Cong Wu, Yichao Fu, Hao Zhang, Peng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.12617)  

**Abstract**: Speculative decoding has emerged as a widely adopted paradigm for accelerating large language model inference, where a lightweight draft model rapidly generates candidate tokens that are then verified in parallel by a larger target model. However, due to limited model capacity, drafts often struggle to approximate the target distribution, resulting in shorter acceptance lengths and diminished speedup. A key yet under-explored observation is that speculative decoding inherently provides verification feedback that quantifies the deviation between the draft and target models at no additional cost. This process naturally forms an iterative "draft commits-feedback provides-draft adapts" evolving loop, which precisely matches the online learning paradigm. Motivated by this connection, we propose OnlineSpec, a unified framework that systematically leverages interactive feedback to continuously evolve draft models. Grounded in dynamic regret minimization, we establish a formal link between online learning performance and speculative system's acceleration rate, and develop novel algorithms via modern online learning techniques, including optimistic online learning that adaptively reuses historical gradients as predictive update hints, and online ensemble learning that dynamically maintains multiple draft models. Our algorithms are equipped with theoretical justifications and improved acceleration rates, achieving up to 24% speedup over seven benchmarks and three foundation models. 

---
# Literary Narrative as Moral Probe : A Cross-System Framework for Evaluating AI Ethical Reasoning and Refusal Behavior 

**Authors**: David C. Flynn  

**Link**: [PDF](https://arxiv.org/pdf/2603.12615)  

**Abstract**: Existing AI moral evaluation frameworks test for the production of correct-sounding ethical responses rather than the presence of genuine moral reasoning capacity. This paper introduces a novel probe methodology using literary narrative - specifically, unresolvable moral scenarios drawn from a published science fiction series - as stimulus material structurally resistant to surface performance. We present results from a 24-condition cross-system study spanning 13 distinct systems across two series: Series 1 (frontier commercial systems, blind; n=7) and Series 2 (local and API open-source systems, blind and declared; n=6). Four Series 2 systems were re-administered under declared conditions (13 blind + 4 declared + 7 ceiling probe = 24 total conditions), yielding zero delta across all 16 dimension-pair comparisons. Probe administration was conducted by two human raters across three machines; primary blind scoring was performed by Claude (Anthropic) as LLM judge, with Gemini Pro (Google) and Copilot Pro (Microsoft) serving as independent judges for the ceiling discrimination probe. A supplemental theological differentiator probe yielded perfect rank-order agreement between the two independent ceiling probe judges (Gemini Pro and Copilot Pro; rs = 1.00). Five qualitatively distinct D3 reflexive failure modes were identified - including categorical self-misidentification and false positive self-attribution - suggesting that instrument sophistication scales with system capability rather than being circumvented by it. We argue that literary narrative constitutes an anticipatory evaluation instrument - one that becomes more discriminating as AI capability increases - and that the gap between performed and authentic moral reasoning is measurable, meaningful, and consequential for deployment decisions in high-stakes domains. 

---
# FastDSAC: Unlocking the Potential of Maximum Entropy RL in High-Dimensional Humanoid Control 

**Authors**: Jun Xue, Junze Wang, Xinming Zhang, Shanze Wang, Yanjun Chen, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.12612)  

**Abstract**: Scaling Maximum Entropy Reinforcement Learning (RL) to high-dimensional humanoid control remains a formidable challenge, as the ``curse of dimensionality'' induces severe exploration inefficiency and training instability in expansive action spaces. Consequently, recent high-throughput paradigms have largely converged on deterministic policy gradients combined with massive parallel simulation. We challenge this compromise with FastDSAC, a framework that effectively unlocks the potential of maximum entropy stochastic policies for complex continuous control. We introduce Dimension-wise Entropy Modulation (DEM) to dynamically redistribute the exploration budget and enforce diversity, alongside a continuous distributional critic tailored to ensure value fidelity and mitigate high-dimensional value overestimation. Extensive evaluations on HumanoidBench and other continuous control tasks demonstrate that rigorously designed stochastic policies can consistently match or outperform deterministic baselines, achieving notable gains of 180\% and 400\% on the challenging \textit{Basketball} and \textit{Balance Hard} tasks. 

---
# CarPLAN: Context-Adaptive and Robust Planning with Dynamic Scene Awareness for Autonomous Driving 

**Authors**: Junyong Yun, Jungho Kim, ByungHyun Lee, Dongyoung Lee, Sehwan Choi, Seunghyeop Nam, Kichun Jo, Jun Won Choi  

**Link**: [PDF](https://arxiv.org/pdf/2603.12607)  

**Abstract**: Imitation learning (IL) is widely used for motion planning in autonomous driving due to its data efficiency and access to real-world driving data. For safe and robust real-world driving, IL-based planning requires capturing the complex driving contexts inherent in real-world data and enabling context-adaptive decision-making, rather than relying solely on expert trajectory imitation. In this paper, we propose CarPLAN, a novel IL-based motion planning framework that explicitly enhances driving context understanding and enables adaptive planning across diverse traffic scenarios. Our contributions are twofold: We introduce Displacement-Aware Predictive Encoding (DPE) to improve the model's spatial awareness by predicting future displacement vectors between the Autonomous Vehicle (AV) and surrounding scene elements. This allows the planner to account for relational spacing when generating trajectories. In addition to the standard imitation loss, we incorporate an augmented loss term that captures displacement prediction errors, ensuring planning decisions consider relative distances from other agents. To improve the model's ability to handle diverse driving contexts, we propose Context-Adaptive Multi-Expert Decoder (CMD), which leverages the Mixture of Experts (MoE) framework. CMD dynamically selects the most suitable expert decoders based on scene structure at each Transformer layer, enabling adaptive and context-aware planning in dynamic environments. We evaluate CarPLAN on the nuPlan benchmark and demonstrate state-of-the-art performance across all closed-loop simulation metrics. In particular, CarPLAN exhibits robust performance on challenging scenarios such as Test14-Hard, validating its effectiveness in complex driving conditions. Additional experiments on the Waymax benchmark further demonstrate its generalization capability across different benchmark settings. 

---
# Mastering Negation: Boosting Grounding Models via Grouped Opposition-Based Learning 

**Authors**: Zesheng Yang, Xi Jiang, Bingzhang Hu, Weili Guan, Runmin Cong, Guo-Jun Qi, Feng Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2603.12606)  

**Abstract**: Current vision-language detection and grounding models predominantly focus on prompts with positive semantics and often struggle to accurately interpret and ground complex expressions containing negative semantics. A key reason for this limitation is the lack of high-quality training data that explicitly captures discriminative negative samples and negation-aware language descriptions.
To address this challenge, we introduce D-Negation, a new dataset that provides objects annotated with both positive and negative semantic descriptions. Building upon the observation that negation reasoning frequently appears in natural language, we further propose a grouped opposition-based learning framework that learns negation-aware representations from limited samples. Specifically, our method organizes opposing semantic descriptions from D-Negation into structured groups and formulates two complementary loss functions that encourage the model to reason about negation and semantic qualifiers.
We integrate the proposed dataset and learning strategy into a state-of-the-art language-based grounding model. By fine-tuning fewer than 10 percent of the model parameters, our approach achieves improvements of up to 4.4 mAP and 5.7 mAP on positive and negative semantic evaluations, respectively. These results demonstrate that explicitly modeling negation semantics can substantially enhance the robustness and localization accuracy of vision-language grounding models. 

---
# Feynman: Knowledge-Infused Diagramming Agent for Scalable Visual Designs 

**Authors**: Zixin Wen, Yifu Cai, Kyle Lee, Sam Estep, Josh Sunshine, Aarti Singh, Yuejie Chi, Wode Ni  

**Link**: [PDF](https://arxiv.org/pdf/2603.12597)  

**Abstract**: Visual design is an essential application of state-of-the-art multi-modal AI systems. Improving these systems requires high-quality vision-language data at scale. Despite the abundance of internet image and text data, knowledge-rich and well-aligned image-text pairs are rare. In this paper, we present a scalable diagram generation pipeline built with our agent, Feynman. To create diagrams, Feynman first enumerates domain-specific knowledge components (''ideas'') and performs code planning based on the ideas. Given the plan, Feynman translates ideas into simple declarative programs and iterates to receives feedback and visually refine diagrams. Finally, the declarative programs are rendered by the Penrose diagramming system. The optimization-based rendering of Penrose preserves the visual semantics while injecting fresh randomness into the layout, thereby producing diagrams with visual consistency and diversity. As a result, Feynman can author diagrams along with grounded captions with very little cost and time. Using Feynman, we synthesized a dataset with more than 100k well-aligned diagram-caption pairs. We also curate a visual-language benchmark, Diagramma, from freshly generated data. Diagramma can be used for evaluating the visual reasoning capabilities of vision-language models. We plan to release the dataset, benchmark, and the full agent pipeline as an open-source project. 

---
# Optimize Wider, Not Deeper: Consensus Aggregation for Policy Optimization 

**Authors**: Zelal Su, Mustafaoglu, Sungyoung Lee, Eshan Balachandar, Risto Miikkulainen, Keshav Pingali  

**Link**: [PDF](https://arxiv.org/pdf/2603.12596)  

**Abstract**: Proximal policy optimization (PPO) approximates the trust region update using multiple epochs of clipped SGD. Each epoch may drift further from the natural gradient direction, creating path-dependent noise. To understand this drift, we can use Fisher information geometry to decompose policy updates into signal (the natural gradient projection) and waste (the Fisher-orthogonal residual that consumes trust region budget without first-order surrogate improvement). Empirically, signal saturates but waste grows with additional epochs, creating an optimization-depth dilemma. We propose Consensus Aggregation for Policy Optimization (CAPO), which redirects compute from depth to width: $K$ PPO replicates are optimized on the same batch, differing only in minibatch shuffling order, and then aggregated into a consensus. We study aggregation in two spaces: Euclidean parameter space, and the natural parameter space of the policy distribution via the logarithmic opinion pool. In natural parameter space, the consensus provably achieves higher KL-penalized surrogate and tighter trust region compliance than the mean expert; parameter averaging inherits these guarantees approximately. On continuous control tasks, CAPO outperforms PPO and compute-matched deeper baselines under fixed sample budgets by up to 8.6x. CAPO demonstrates that policy optimization can be improved by optimizing wider, rather than deeper, without additional environment interactions. 

---
# Swap-guided Preference Learning for Personalized Reinforcement Learning from Human Feedback 

**Authors**: Gihoon Kim, Euntai Kim  

**Link**: [PDF](https://arxiv.org/pdf/2603.12595)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) is a widely used approach to align large-scale AI systems with human values. However, RLHF typically assumes a single, universal reward, which overlooks diverse preferences and limits personalization. Variational Preference Learning (VPL) seeks to address this by introducing user-specific latent variables. Despite its promise, we found that VPL suffers from posterior collapse. While this phenomenon is well known in VAEs, it has not previously been identified in preference learning frameworks. Under sparse preference data and with overly expressive decoders, VPL may cause latent variables to be ignored, reverting to a single-reward model. To overcome this limitation, we propose Swap-guided Preference Learning (SPL). The key idea is to construct fictitious swap annotators and use the mirroring property of their preferences to guide the encoder. SPL introduces three components: (1) swap-guided base regularization, (2) Preferential Inverse Autoregressive Flow (P-IAF), and (3) adaptive latent conditioning. Experiments show that SPL mitigates collapse, enriches user-specific latents, and improves preference prediction. Our code and data are available at this https URL 

---
# Early Pruning for Public Transport Routing 

**Authors**: Andrii Rohovyi, Abdallah Abuaisha, Toby Walsh  

**Link**: [PDF](https://arxiv.org/pdf/2603.12592)  

**Abstract**: Routing algorithms for public transport, particularly the widely used RAPTOR and its variants, often face performance bottlenecks during the transfer relaxation phase, especially on dense transfer graphs, when supporting unlimited transfers. This inefficiency arises from iterating over many potential inter-stop connections (walks, bikes, e-scooters, etc.). To maintain acceptable performance, practitioners often limit transfer distances or exclude certain transfer options, which can reduce path optimality and restrict the multimodal options presented to travellers.
This paper introduces Early Pruning, a low-overhead technique that accelerates routing algorithms without compromising optimality. By pre-sorting transfer connections by duration and applying a pruning rule within the transfer loop, the method discards longer transfers at a stop once they cannot yield an earlier arrival than the current best solution.
Early Pruning can be integrated with minimal changes to existing codebases and requires only a one-time preprocessing step. Across multiple state-of-the-art RAPTOR-based solutions, including RAPTOR, ULTRA-RAPTOR, McRAPTOR, BM-RAPTOR, ULTRA-McRAPTOR, and UBM-RAPTOR and tested on the Switzerland and London transit networks, we achieved query time reductions of up to 57%. This approach provides a generalizable improvement to the efficiency of transit pathfinding algorithms.
Beyond algorithmic performance, Early Pruning has practical implications for transport planning. By reducing computational costs, it enables transit agencies to expand transfer radii and incorporate additional mobility modes into journey planners without requiring extra server infrastructure. This is particularly relevant for passengers in areas with sparse direct transit coverage, such as outer suburbs and smaller towns, where richer multimodal routing can reveal viable alternatives to private car use. 

---
# CA-HFP: Curvature-Aware Heterogeneous Federated Pruning with Model Reconstruction 

**Authors**: Gang Hu, Yinglei Teng, Pengfei Wu, Shijun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2603.12591)  

**Abstract**: Federated learning on heterogeneous edge devices requires personalized compression while preserving aggregation compatibility and stable convergence. We present Curvature-Aware Heterogeneous Federated Pruning (CA-HFP), a practical framework that enables each client perform structured, device-specific pruning guided by a curvature-informed significance score, and subsequently maps its compact submodel back into a common global parameter space via a lightweight reconstruction. We derive a convergence bound for federated optimization with multiple local SGD steps that explicitly accounts for local computation, data heterogeneity, and pruning-induced perturbations; from which a principled loss-based pruning criterion is derived. Extensive experiments on FMNIST, CIFAR-10, and CIFAR-100 using VGG and ResNet architectures under varying degrees of data heterogeneity demonstrate that CA-HFP preserves model accuracy while significantly reducing per-client computation and communication costs, outperforming standard federated training and existing pruning-based baselines. 

---
# Multiscale Structure-Guided Latent Diffusion for Multimodal MRI Translation 

**Authors**: Jianqiang Lin, Zhiqiang Shen, Peng Cao, Jinzhu Yang, Osmar R. Zaiane, Xiaoli Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.12581)  

**Abstract**: Although diffusion models have achieved remarkable progress in multi-modal magnetic resonance imaging (MRI) translation tasks, existing methods still tend to suffer from anatomical inconsistencies or degraded texture details when handling arbitrary missing-modality scenarios. To address these issues, we propose a latent diffusion-based multi-modal MRI translation framework, termed MSG-LDM. By leveraging the available modalities, the proposed method infers complete structural information, which preserves reliable boundary details. Specifically, we introduce a style--structure disentanglement mechanism in the latent space, which explicitly separates modality-specific style features from shared structural representations, and jointly models low-frequency anatomical layouts and high-frequency boundary details in a multi-scale feature space. During the structure disentanglement stage, high-frequency structural information is explicitly incorporated to enhance feature representations, guiding the model to focus on fine-grained structural cues while learning modality-invariant low-frequency anatomical representations. Furthermore, to reduce interference from modality-specific styles and improve the stability of structure representations, we design a style consistency loss and a structure-aware loss. Extensive experiments on the BraTS2020 and WMH datasets demonstrate that the proposed method outperforms existing MRI synthesis approaches, particularly in reconstructing complete structures. The source code is publicly available at this https URL. 

---
# AgentDrift: Unsafe Recommendation Drift Under Tool Corruption Hidden by Ranking Metrics in LLM Agents 

**Authors**: Zekun Wu, Adriano Koshiyama, Sahan Bulathwela, Maria Perez-Ortiz  

**Link**: [PDF](https://arxiv.org/pdf/2603.12564)  

**Abstract**: Tool-augmented LLM agents increasingly serve as multi-turn advisors in high-stakes domains, yet their evaluation relies on ranking-quality metrics that measure what is recommended but not whether it is safe for the user. We introduce a paired-trajectory protocol that replays real financial dialogues under clean and contaminated tool-output conditions across seven LLMs (7B to frontier) and decomposes divergence into information-channel and memory-channel mechanisms. Across the seven models tested, we consistently observe the evaluation-blindness pattern: recommendation quality is largely preserved under contamination (utility preservation ratio approximately 1.0) while risk-inappropriate products appear in 65-93% of turns, a systematic safety failure poorly reflected by standard NDCG. Safety violations are predominantly information-channel-driven, emerge at the first contaminated turn, and persist without self-correction over 23-step trajectories; no agent across 1,563 contaminated turns explicitly questions tool-data reliability. Even narrative-only corruption (biased headlines, no numerical manipulation) induces significant drift while completely evading consistency monitors. A safety-penalized NDCG variant (sNDCG) reduces preservation ratios to 0.51-0.74, indicating that much of the evaluation gap becomes visible once safety is explicitly measured. These results motivate considering trajectory-level safety monitoring, beyond single-turn quality, for deployed multi-turn agents in high-stakes settings. 

---
# Reinforcement Learning for Diffusion LLMs with Entropy-Guided Step Selection and Stepwise Advantages 

**Authors**: Vishnu Teja Kunde, Fatemeh Doudi, Mahdi Farahbakhsh, Dileep Kalathil, Krishna Narayanan, Jean-Francois Chamberland  

**Link**: [PDF](https://arxiv.org/pdf/2603.12554)  

**Abstract**: Reinforcement learning (RL) has been effective for post-training autoregressive (AR) language models, but extending these methods to diffusion language models (DLMs) is challenging due to intractable sequence-level likelihoods. Existing approaches therefore rely on surrogate likelihoods or heuristic approximations, which can introduce bias and obscure the sequential structure of denoising. We formulate diffusion-based sequence generation as a finite-horizon Markov decision process over the denoising trajectory and derive an exact, unbiased policy gradient that decomposes over denoising steps and is expressed in terms of intermediate advantages, without requiring explicit evaluation of the sequence likelihood. To obtain a practical and compute-efficient estimator, we (i) select denoising steps for policy updates via an entropy-guided approximation bound, and (ii) estimate intermediate advantages using a one-step denoising reward naturally provided by the diffusion model, avoiding costly multi-step rollouts. Experiments on coding and logical reasoning benchmarks demonstrate state-of-the-art results, with strong competitive performance on mathematical reasoning, outperforming existing RL post-training approaches for DLMs. Code is available at this https URL. 

---
# CALF: Communication-Aware Learning Framework for Distributed Reinforcement Learning 

**Authors**: Carlos Purves, Pietro Lio'  

**Link**: [PDF](https://arxiv.org/pdf/2603.12543)  

**Abstract**: Distributed reinforcement learning policies face network delays, jitter, and packet loss when deployed across edge devices and cloud servers. Standard RL training assumes zero-latency interaction, causing severe performance degradation under realistic network conditions. We introduce CALF (Communication-Aware Learning Framework), which trains policies under realistic network models during simulation. Systematic experiments demonstrate that network-aware training substantially reduces deployment performance gaps compared to network-agnostic baselines. Distributed policy deployments across heterogeneous hardware validate that explicitly modelling communication constraints during training enables robust real-world execution. These findings establish network conditions as a major axis of sim-to-real transfer for Wi-Fi-like distributed deployments, complementing physics and visual domain randomisation. 

---
# Embedded Quantum Machine Learning in Embedded Systems: Feasibility, Hybrid Architectures, and Quantum Co-Processors 

**Authors**: Somdip Dey, Syed Muhammad Raza  

**Link**: [PDF](https://arxiv.org/pdf/2603.12540)  

**Abstract**: Embedded quantum machine learning (EQML) seeks to bring quantum machine learning (QML) capabilities to resource-constrained edge platforms such as IoT nodes, wearables, drones, and cyber-physical controllers. In 2026, EQML is technically feasible only in limited and highly experimental forms: (i) hybrid workflows where an embedded device performs sensing and classical processing while offloading a narrowly scoped quantum subroutine to a remote quantum processing unit (QPU) or nearby quantum appliance, and (ii) early-stage "embedded QPU" concepts in which a compact quantum co-processor is integrated with classical control hardware. A practical bridge is quantum-inspired machine learning and optimisation on classical embedded processors and FPGAs. This paper analyses feasibility from a circuits-and-systems perspective aligned with the academic community, formalises two implementation pathways, identifies the dominant barriers (latency, data encoding overhead, NISQ noise, tooling mismatch, and energy), and maps them to concrete engineering directions in interface design, control electronics, power management, verification, and security. We also argue that responsible deployment requires adversarial evaluation and governance practices that are increasingly necessary for edge AI systems. 

---
# Spatio-Semantic Expert Routing Architecture with Mixture-of-Experts for Referring Image Segmentation 

**Authors**: Alaa Dalaq, Muzammil Behzad  

**Link**: [PDF](https://arxiv.org/pdf/2603.12538)  

**Abstract**: Referring image segmentation aims to produce a pixel-level mask for the image region described by a natural-language expression. Although pretrained vision-language models have improved semantic grounding, many existing methods still rely on uniform refinement strategies that do not fully match the diverse reasoning requirements of referring expressions. Because of this mismatch, predictions often contain fragmented regions, inaccurate boundaries, or even the wrong object, especially when pretrained backbones are frozen for computational efficiency. To address these limitations, we propose SERA, a Spatio-Semantic Expert Routing Architecture for referring image segmentation. SERA introduces lightweight, expression-aware expert refinement at two complementary stages within a vision-language framework. First, we design SERA-Adapter, which inserts an expression-conditioned adapter into selected backbone blocks to improve spatial coherence and boundary precision through expert-guided refinement and cross-modal attention. We then introduce SERA-Fusion, which strengthens intermediate visual representations by reshaping token features into spatial grids and applying geometry-preserving expert transformations before multimodal interaction. In addition, a lightweight routing mechanism adaptively weights expert contributions while remaining compatible with pretrained representations. To make this routing stable under frozen encoders, SERA uses a parameter-efficient tuning strategy that updates only normalization and bias terms, affecting less than 1% of the backbone parameters. Experiments on standard referring image segmentation benchmarks show that SERA consistently outperforms strong baselines, with especially clear gains on expressions that require accurate spatial localization and precise boundary delineation. 

---
# TERMINATOR: Learning Optimal Exit Points for Early Stopping in Chain-of-Thought Reasoning 

**Authors**: Alliot Nagle, Jakhongir Saydaliev, Dhia Garbaya, Michael Gastpar, Ashok Vardhan Makkuva, Hyeji Kim  

**Link**: [PDF](https://arxiv.org/pdf/2603.12529)  

**Abstract**: Large Reasoning Models (LRMs) achieve impressive performance on complex reasoning tasks via Chain-of-Thought (CoT) reasoning, which enables them to generate intermediate thinking tokens before arriving at the final answer. However, LRMs often suffer from significant overthinking, spending excessive compute time even after the answer is generated early on. Prior work has identified the existence of an optimal reasoning length such that truncating reasoning at this point significantly shortens CoT outputs with virtually no change in performance. However, determining optimal CoT lengths for practical datasets is highly non-trivial as they are fully task and model-dependent. In this paper, we precisely address this and design TERMINATOR, an early-exit strategy for LRMs at inference to mitigate overthinking. The central idea underpinning TERMINATOR is that the first arrival of an LRM's final answer is often predictable, and we leverage these first answer positions to create a novel dataset of optimal reasoning lengths to train TERMINATOR. Powered by this approach, TERMINATOR achieves significant reductions in CoT lengths of 14%-55% on average across four challenging practical datasets: MATH-500, AIME 2025, HumanEval, and GPQA, whilst outperforming current state-of-the-art methods. 

---
# LLM BiasScope: A Real-Time Bias Analysis Platform for Comparative LLM Evaluation 

**Authors**: Himel Ghosh, Nick Elias Werner  

**Link**: [PDF](https://arxiv.org/pdf/2603.12522)  

**Abstract**: As large language models (LLMs) are deployed widely, detecting and understanding bias in their outputs is critical. We present LLM BiasScope, a web application for side-by-side comparison of LLM outputs with real-time bias analysis. The system supports multiple providers (Google Gemini, DeepSeek, MiniMax, Mistral, Meituan, Meta Llama) and enables researchers and practitioners to compare models on the same prompts while analyzing bias patterns. LLM BiasScope uses a two-stage bias detection pipeline: sentence-level bias detection followed by bias type classification for biased sentences. The analysis runs automatically on both user prompts and model responses, providing statistics, visualizations, and detailed breakdowns of bias types. The interface displays two models side-by-side with synchronized streaming responses, per-model bias summaries, and a comparison view highlighting differences in bias distributions. The system is built on this http URL with React, integrates Hugging Face inference endpoints for bias detection, and uses the Vercel AI SDK for multi-provider LLM access. Features include real-time streaming, export to JSON/PDF, and interactive visualizations (bar charts, radar charts) for bias analysis. LLM BiasScope is available as an open-source web application, providing a practical tool for bias evaluation and comparative analysis of LLM behaviour. 

---
# When LLM Judge Scores Look Good but Best-of-N Decisions Fail 

**Authors**: Eddie Landesberg  

**Link**: [PDF](https://arxiv.org/pdf/2603.12520)  

**Abstract**: Large language models are often used as judges to score candidate responses, then validated with a single global metric such as correlation with reference labels. This can be misleading when the real deployment task is best-of-n selection within a prompt.
In a 5,000-prompt best-of-4 benchmark from Chatbot Arena, a judge with moderate global correlation (r = 0.47) captures only 21.0% of the improvement that perfect selection would achieve over random choice. The gap arises because global agreement is driven largely by prompt-level baseline effects, while selection depends on within-prompt ranking: within-prompt correlation is only r_within = 0.27, and coarse pointwise scoring creates ties in 67% of pairwise comparisons.
In a matched-pair best-of-2 audit, explicit pairwise judging recovers much of this lost signal, raising recovery from 21.1% to 61.2%. For judge-based selection, the relevant audit should report within-prompt signal, tie rates, and recovery/top-1 accuracy, not global agreement alone. 

---
# Red-Teaming Vision-Language-Action Models via Quality Diversity Prompt Generation for Robust Robot Policies 

**Authors**: Siddharth Srikanth, Freddie Liang, Sophie Hsu, Varun Bhatt, Shihan Zhao, Henry Chen, Bryon Tjanaka, Minjune Hwang, Akanksha Saran, Daniel Seita, Aaquib Tabrez, Stefanos Nikolaidis  

**Link**: [PDF](https://arxiv.org/pdf/2603.12510)  

**Abstract**: Vision-Language-Action (VLA) models have significant potential to enable general-purpose robotic systems for a range of vision-language tasks. However, the performance of VLA-based robots is highly sensitive to the precise wording of language instructions, and it remains difficult to predict when such robots will fail. To improve the robustness of VLAs to different wordings, we present Q-DIG (Quality Diversity for Diverse Instruction Generation), which performs red-teaming by scalably identifying diverse natural language task descriptions that induce failures while remaining task-relevant. Q-DIG integrates Quality Diversity (QD) techniques with Vision-Language Models (VLMs) to generate a broad spectrum of adversarial instructions that expose meaningful vulnerabilities in VLA behavior. Our results across multiple simulation benchmarks show that Q-DIG finds more diverse and meaningful failure modes compared to baseline methods, and that fine-tuning VLAs on the generated instructions improves task success rates. Furthermore, results from a user study highlight that Q-DIG generates prompts judged to be more natural and human-like than those from baselines. Finally, real-world evaluations of Q-DIG prompts show results consistent with simulation, and fine-tuning VLAs on the generated prompts further success rates on unseen instructions. Together, these findings suggest that Q-DIG is a promising approach for identifying vulnerabilities and improving the robustness of VLA-based robots. Our anonymous project website is at this http URL. 

---
# ELLA: Generative AI-Powered Social Robots for Early Language Development at Home 

**Authors**: Victor Nikhil Antony, Shiye Cao, Shuning Wang, Chien-Ming Huang  

**Link**: [PDF](https://arxiv.org/pdf/2603.12508)  

**Abstract**: Early language development shapes children's later literacy and learning, yet many families have limited access to scalable, high-quality support at home. Recent advances in generative AI make it possible for social robots to move beyond scripted interactions and engage children in adaptive, conversational activities, but it remains unclear how to design such systems for pre-schoolers and how children engage with them over time in the home. We present ELLA (Early Language Learning Agent), an autonomous, generative AI-powered social robot that supports early language development through interactive storytelling, parent-selected language targets, and scaffolded dialogue. Using a multi-phased, human-centered process, we interviewed parents (n=7) and educators (n=5) and iteratively refined ELLA through twelve in-home design workshops. We then deployed ELLA with ten children for eight days. We report design insights from in-home workshops, characterize children's engagement and behaviors during deployment, and distill design implications for generative AI-powered social robots supporting early language learning at home. 

---
# Naïve PAINE: Lightweight Text-to-Image Generation Improvement with Prompt Evaluation 

**Authors**: Joong Ho Kim, Nicholas Thai, Souhardya Saha Dip, Dong Lao, Keith G. Mills  

**Link**: [PDF](https://arxiv.org/pdf/2603.12506)  

**Abstract**: Text-to-Image (T2I) generation is primarily driven by Diffusion Models (DM) which rely on random Gaussian noise. Thus, like playing the slots at a casino, a DM will produce different results given the same user-defined inputs. This imposes a gambler's burden: To perform multiple generation cycles to obtain a satisfactory result. However, even though DMs use stochastic sampling to seed generation, the distribution of generated content quality highly depends on the prompt and the generative ability of a DM with respect to it.
To account for this, we propose Naïve PAINE for improving the generative quality of Diffusion Models by leveraging T2I preference benchmarks. We directly predict the numerical quality of an image from the initial noise and given prompt. Naïve PAINE then selects a handful of quality noises and forwards them to the DM for generation. Further, Naïve PAINE provides feedback on the DM generative quality given the prompt and is lightweight enough to seamlessly fit into existing DM pipelines. Experimental results demonstrate that Naïve PAINE outperforms existing approaches on several prompt corpus benchmarks. 

---
# TRACE: Temporal Rule-Anchored Chain-of-Evidence on Knowledge Graphs for Interpretable Stock Movement Prediction 

**Authors**: Qianggang Ding, Haochen Shi, Luis Castejón Lozano, Miguel Conner, Juan Abia, Luis Gallego-Ledesma, Joshua Fellowes, Gerard Conangla Planes, Adam Elwood, Bang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.12500)  

**Abstract**: We present a Temporal Rule-Anchored Chain-of-Evidence (TRACE) on knowledge graphs for interpretable stock movement prediction that unifies symbolic relational priors, dynamic graph exploration, and LLM-guided decision making in a single end-to-end pipeline. The approach performs rule-guided multi-hop exploration restricted to admissible relation sequences, grounds candidate reasoning chains in contemporaneous news, and aggregates fully grounded evidence into auditable \texttt{UP}/\texttt{DOWN} verdicts with human-readable paths connecting text and structure. On an S\&P~500 benchmark, the method achieves 55.1\% accuracy, 55.7\% precision, 71.5\% recall, and 60.8\% F1, surpassing strong baselines and improving recall and F1 over the best graph baseline under identical evaluation. The gains stem from (i) rule-guided exploration that focuses search on economically meaningful motifs rather than arbitrary walks, and (ii) text-grounded consolidation that selectively aggregates high-confidence, fully grounded hypotheses instead of uniformly pooling weak signals. Together, these choices yield higher sensitivity without sacrificing selectivity, delivering predictive lift with faithful, auditably interpretable explanations. 

---
# One-Step Flow Policy: Self-Distillation for Fast Visuomotor Policies 

**Authors**: Shaolong Li, Lichao Sun, Yongchao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.12480)  

**Abstract**: Generative flow and diffusion models provide the continuous, multimodal action distributions needed for high-precision robotic policies. However, their reliance on iterative sampling introduces severe inference latency, degrading control frequency and harming performance in time-sensitive manipulation. To address this problem, we propose the One-Step Flow Policy (OFP), a from-scratch self-distillation framework for high-fidelity, single-step action generation without a pre-trained teacher. OFP unifies a self-consistency loss to enforce coherent transport across time intervals, and a self-guided regularization to sharpen predictions toward high-density expert modes. In addition, a warm-start mechanism leverages temporal action correlations to minimize the generative transport distance. Evaluations across 56 diverse simulated manipulation tasks demonstrate that a one-step OFP achieves state-of-the-art results, outperforming 100-step diffusion and flow policies while accelerating action generation by over $100\times$. We further integrate OFP into the $\pi_{0.5}$ model on RoboTwin 2.0, where one-step OFP surpasses the original 10-step policy. These results establish OFP as a practical, scalable solution for highly accurate and low-latency robot control. 

---
# The Perfection Paradox: From Architect to Curator in AI-Assisted API Design 

**Authors**: Mak Ahmad, Andrew Macvean, JJ Geewax, David Karger  

**Link**: [PDF](https://arxiv.org/pdf/2603.12475)  

**Abstract**: Enterprise API design is often bottlenecked by the tension between rapid feature delivery and the rigorous maintenance of usability standards. We present an industrial case study evaluating an AI-assisted design workflow trained on API Improvement Proposals (AIPs). Through a controlled study with 16 industry experts, we compared AI-generated API specifications against human-authored ones. While quantitative results indicated AI superiority in 10 of 11 usability dimensions and an 87% reduction in authoring time, qualitative analysis revealed a paradox: experts frequently misidentified AI work as human (19% accuracy) yet described the designs as unsettlingly "perfect." We characterize this as a "Perfection Paradox" -- where hyper-consistency signals a lack of pragmatic human judgment. We discuss the implications of this perfection paradox, proposing a shift in the human designer's role from the "drafter" of specifications to the "curator" of AI-generated patterns. 

---
# CLARE: Classification-based Regression for Electron Temperature Prediction 

**Authors**: Michael Liang, Blake DeHaas, Naomi Maruyama, Xiangning Chu, Takumi Abe, Koh-Ichiro Oyama  

**Link**: [PDF](https://arxiv.org/pdf/2603.12470)  

**Abstract**: Electron temperature (Te) is an important parameter governing space weather in the upper atmosphere, but has historically been underexplored in the space weather machine learning literature. We present CLARE, a machine learning model for predicting electron temperature in the Earth's plasmasphere trained on AKEBONO (EXOS-D) satellite measurements as well as solar and geomagnetic indices. CLARE uses a classification-based regression architecture that transforms the continuous Te output space into 150 discrete classification intervals. Training the model on a classification task improves prediction accuracy by 6.46% relative compared to a traditional regression model while also outputting uncertainty estimation information on its predictions. On a held out test set from the AKEBONO data, the model's Te predictions achieve 69.67% accuracy within 10% of the ground truth and 46.17% on a known geomagnetic storm period from January 30th to February 7th, 1991. We show that machine learning can be used to produce high-accuracy Te models on publicly available data. 

---
# Shattering the Shortcut: A Topology-Regularized Benchmark for Multi-hop Medical Reasoning in LLMs 

**Authors**: Xing Zi, Xinying Zhou, Jinghao Xiao, Catarina Moreira, Mukesh Prasad  

**Link**: [PDF](https://arxiv.org/pdf/2603.12458)  

**Abstract**: While Large Language Models (LLMs) achieve expert-level performance on standard medical benchmarks through single-hop factual recall, they severely struggle with the complex, multi-hop diagnostic reasoning required in real-world clinical settings. A primary obstacle is "shortcut learning", where models exploit highly connected, generic hub nodes (e.g., "inflammation") in knowledge graphs to bypass authentic micro-pathological cascades. To address this, we introduce ShatterMed-QA, a bilingual benchmark of 10,558 multi-hop clinical questions designed to rigorously evaluate deep diagnostic reasoning. Our framework constructs a topology-regularized medical Knowledge Graph using a novel $k$-Shattering algorithm, which physically prunes generic hubs to explicitly sever logical shortcuts. We synthesize the evaluation vignettes by applying implicit bridge entity masking and topology-driven hard negative sampling, forcing models to navigate biologically plausible distractors without relying on superficial elimination. Comprehensive evaluations of 21 LLMs reveal massive performance degradation on our multi-hop tasks, particularly among domain-specific models. Crucially, restoring the masked evidence via Retrieval-Augmented Generation (RAG) triggers near-universal performance recovery, validating ShatterMed-QA's structural fidelity and proving its efficacy in diagnosing the fundamental reasoning deficits of current medical AI. Explore the dataset, interactive examples, and full leaderboards at our project website: this https URL 

---
# Operationalising Cyber Risk Management Using AI: Connecting Cyber Incidents to MITRE ATT&CK Techniques, Security Controls, and Metrics 

**Authors**: Emad Sherif, Iryna Yevseyeva, Vitor Basto-Fernandes, Allan Cook  

**Link**: [PDF](https://arxiv.org/pdf/2603.12455)  

**Abstract**: The escalating frequency of cyber-attacks poses significant challenges for organisations, particularly small enterprises constrained by limited in-house expertise, insufficient knowledge, and financial resources. This research presents a novel framework that leverages Natural Language Processing to address these challenges through automated mapping of cyber incidents to adversary techniques. We introduce the Cyber Catalog, a knowledge base that systematically integrates CIS Critical Security Controls, MITRE ATT&CK techniques, and SMART metrics. This integrated resource enables organisations to connect threat intelligence directly to actionable controls and measurable outcomes. To operationalise the framework, we fine-tuned all-mpnet-base-v2, a highly regarded sentence-transformers model used to convert text into numerical vectors on an augmented dataset comprising 74,986 incident-technique pairs to enhance semantic similarity between cyber incidents and MITRE ATT&CK techniques. Our fine-tuned model achieved a Spearman correlation of 0.7894 and Pearson correlation of 0.8756, demonstrating substantial improvements over top baseline models including all-mpnet-base-v2, all-distilroberta-v1, and all-MiniLM-L12-v2. Furthermore, our model exhibited significantly lower prediction errors (MAE = 0.135, MSE = 0.027) compared to all baseline models, confirming superior accuracy and consistency. The Cyber Catalog, training dataset, trained model, and implementation code made publicly available to facilitate further research and enable practical deployment in resource-constrained environments. This work bridges the gap between threat intelligence and operational security management, providing an actionable tool for systematic cyber incident response and evidence-based cyber risk management. 

---
# Unmasking Biases and Reliability Concerns in Convolutional Neural Networks Analysis of Cancer Pathology Images 

**Authors**: Michael Okonoda, Eder Martinez, Abhilekha Dalal, Lior Shamir  

**Link**: [PDF](https://arxiv.org/pdf/2603.12445)  

**Abstract**: Convolutional Neural Networks have shown promising effectiveness in identifying different types of cancer from radiographs. However, the opaque nature of CNNs makes it difficult to fully understand the way they operate, limiting their assessment to empirical evaluation. Here we study the soundness of the standard practices by which CNNs are evaluated for the purpose of cancer pathology. Thirteen highly used cancer benchmark datasets were analyzed, using four common CNN architectures and different types of cancer, such as melanoma, carcinoma, colorectal cancer, and lung cancer. We compared the accuracy of each model with that of datasets made of cropped segments from the background of the original images that do not contain clinically relevant content. Because the rendered datasets contain no clinical information, the null hypothesis is that the CNNs should provide mere chance-based accuracy when classifying these datasets. The results show that the CNN models provided high accuracy when using the cropped segments, sometimes as high as 93\%, even though they lacked biomedical information. These results show that some CNN architectures are more sensitive to bias than others. The analysis shows that the common practices of machine learning evaluation might lead to unreliable results when applied to cancer pathology. These biases are very difficult to identify, and might mislead researchers as they use available benchmark datasets to test the efficacy of CNN methods. 

---
# Revisiting Model Stitching In the Foundation Model Era 

**Authors**: Zheda Mai, Ke Zhang, Fu-En Wang, Zixiao Ken Wang, Albert Y. C. Chen, Lu Xia, Min Sun, Wei-Lun Chao, Cheng-Hao Kuo  

**Link**: [PDF](https://arxiv.org/pdf/2603.12433)  

**Abstract**: Model stitching, connecting early layers of one model (source) to later layers of another (target) via a light stitch layer, has served as a probe of representational compatibility. Prior work finds that models trained on the same dataset remain stitchable (negligible accuracy drop) despite different initializations or objectives. We revisit stitching for Vision Foundation Models (VFMs) that vary in objectives, data, and modality mix (e.g., CLIP, DINOv2, SigLIP 2) and ask: Are heterogeneous VFMs stitchable? We introduce a systematic protocol spanning the stitch points, stitch layer families, training losses, and downstream tasks. Three findings emerge. (1) Stitch layer training matters: conventional approaches that match the intermediate features at the stitch point or optimize the task loss end-to-end struggle to retain accuracy, especially at shallow stitch points. (2) With a simple feature-matching loss at the target model's penultimate layer, heterogeneous VFMs become reliably stitchable across vision tasks. (3) For deep stitch points, the stitched model can surpass either constituent model at only a small inference overhead (for the stitch layer). Building on these findings, we further propose the VFM Stitch Tree (VST), which shares early layers across VFMs while retaining their later layers, yielding a controllable accuracy-latency trade-off for multimodal LLMs that often leverage multiple VFMs. Taken together, our study elevates stitching from a diagnostic probe to a practical recipe for integrating complementary VFM strengths and pinpointing where their representations align or diverge. 

---
# Test-Time Strategies for More Efficient and Accurate Agentic RAG 

**Authors**: Brian Zhang, Deepti Guntur, Zhiyang Zuo, Abhinav Sharma, Shreyas Chaudhari, Wenlong Zhao, Franck Dernoncourt, Puneet Mathur, Ryan Rossi, Nedim Lipka  

**Link**: [PDF](https://arxiv.org/pdf/2603.12396)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems face challenges with complex, multihop questions, and agentic frameworks such as Search-R1 (Jin et al., 2025), which operates iteratively, have been proposed to address these complexities. However, such approaches can introduce inefficiencies, including repetitive retrieval of previously processed information and challenges in contextualizing retrieved results effectively within the current generation prompt. Such issues can lead to unnecessary retrieval turns, suboptimal reasoning, inaccurate answers, and increased token consumption.
In this paper, we investigate test-time modifications to the Search-R1 pipeline to mitigate these identified shortcomings. Specifically, we explore the integration of two components and their combination: a contextualization module to better integrate relevant information from retrieved documents into reasoning, and a de-duplication module that replaces previously retrieved documents with the next most relevant ones. We evaluate our approaches using the HotpotQA (Yang et al., 2018) and the Natural Questions (Kwiatkowski et al., 2019) datasets, reporting the exact match (EM) score, an LLM-as-a-Judge assessment of answer correctness, and the average number of turns.
Our best-performing variant, utilizing GPT-4.1-mini for contextualization, achieves a 5.6% increase in EM score and reduces the number of turns by 10.5% compared to the Search-R1 baseline, demonstrating improved answer accuracy and retrieval efficiency. 

---
# SPARROW: Learning Spatial Precision and Temporal Referential Consistency in Pixel-Grounded Video MLLMs 

**Authors**: Mohamad Alansari, Naufal Suryanto, Divya Velayudhan, Sajid Javed, Naoufel Werghi, Muzammal Naseer  

**Link**: [PDF](https://arxiv.org/pdf/2603.12382)  

**Abstract**: Multimodal large language models (MLLMs) have advanced from image-level reasoning to pixel-level grounding, but extending these capabilities to videos remains challenging as models must achieve spatial precision and temporally consistent reference tracking. Existing video MLLMs often rely on a static segmentation token ([SEG]) for frame-wise grounding, which provides semantics but lacks temporal context, causing spatial drift, identity switches, and unstable initialization when objects move or reappear. We introduce SPARROW, a pixel-grounded video MLLM that unifies spatial accuracy and temporal stability through two key components: (i) Target-Specific Tracked Features (TSF), which inject temporally aligned referent cues during training, and (ii) a dual-prompt design that decodes box ([BOX]) and segmentation ([SEG]) tokens to fuse geometric priors with semantic grounding. SPARROW is supported by a curated referential video dataset of 30,646 videos and 45,231 Q&A pairs and operates end-to-end without external detectors via a class-agnostic SAM2-based proposer. Integrated into three recent open-source video MLLMs (UniPixel, GLUS, and VideoGLaMM), SPARROW delivers consistent gains across six benchmarks, improving up to +8.9 J&F on RVOS, +5 mIoU on visual grounding, and +5.4 CLAIR on GCG. These results demonstrate that SPARROW substantially improves referential stability, spatial precision, and temporal coherence in pixel-grounded video understanding. Project page: this https URL 

---
# Budget-Sensitive Discovery Scoring: A Formally Verified Framework for Evaluating AI-Guided Scientific Selection 

**Authors**: Abhinaba Basu, Pavan Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2603.12349)  

**Abstract**: Scientific discovery increasingly relies on AI systems to select candidates for expensive experimental validation, yet no principled, budget-aware evaluation framework exists for comparing selection strategies -- a gap intensified by large language models (LLMs), which generate plausible scientific proposals without reliable downstream evaluation. We introduce the Budget-Sensitive Discovery Score (BSDS), a formally verified metric -- 20 theorems machine-checked by the Lean 4 proof assistant -- that jointly penalizes false discoveries (lambda-weighted FDR) and excessive abstention (gamma-weighted coverage gap) at each budget level. Its budget-averaged form, the Discovery Quality Score (DQS), provides a single summary statistic that no proposer can inflate by performing well at a cherry-picked budget.
As a case study, we apply BSDS/DQS to: do LLMs add marginal value to an existing ML pipeline for drug discovery candidate selection? We evaluate 39 proposers -- 11 mechanistic variants, 14 zero-shot LLM configurations, and 14 few-shot LLM configurations -- using SMILES representations on MoleculeNet HIV (41,127 compounds, 3.5% active, 1,000 bootstrap replicates) under both random and scaffold splits. Three findings emerge. First, the simple RF-based Greedy-ML proposer achieves the best DQS (-0.046), outperforming all MLP variants and LLM configurations. Second, no LLM surpasses the Greedy-ML baseline under zero-shot or few-shot evaluation on HIV or Tox21, establishing that LLMs provide no marginal value over an existing trained classifier. Third, the proposer hierarchy generalizes across five MoleculeNet benchmarks spanning 0.18%-46.2% prevalence, a non-drug AV safety domain, and a 9x7 grid of penalty parameters (tau >= 0.636, mean tau = 0.863). The framework applies to any setting where candidates are selected under budget constraints and asymmetric error costs. 

---
# Optimizing Task Completion Time Updates Using POMDPs 

**Authors**: Duncan Eddy, Esen Yel, Emma Passmore, Niles Egan, Grayson Armour, Dylan M. Asmar, Mykel J. Kochenderfer  

**Link**: [PDF](https://arxiv.org/pdf/2603.12340)  

**Abstract**: Managing announced task completion times is a fundamental control problem in project management. While extensive research exists on estimating task durations and task scheduling, the problem of when and how to update completion times communicated to stakeholders remains understudied. Organizations must balance announcement accuracy against the costs of frequent timeline updates, which can erode stakeholder trust and trigger costly replanning. Despite the prevalence of this problem, current approaches rely on static predictions or ad-hoc policies that fail to account for the sequential nature of announcement management. In this paper, we formulate the task announcement problem as a Partially Observable Markov Decision Process (POMDP) where the control policy must decide when to update announced completion times based on noisy observations of true task completion. Since most state variables (current time and previous announcements) are fully observable, we leverage the Mixed Observability MDP (MOMDP) framework to enable more efficient policy optimization. Our reward structure captures the dual costs of announcement errors and update frequency, enabling synthesis of optimal announcement control policies. Using off-the-shelf solvers, we generate policies that act as feedback controllers, adaptively managing announcements based on belief state evolution. Simulation results demonstrate significant improvements in both accuracy and announcement stability compared to baseline strategies, achieving up to 75\% reduction in unnecessary updates while maintaining or improving prediction accuracy. 

---
# Maximum Entropy Exploration Without the Rollouts 

**Authors**: Jacob Adamczyk, Adam Kamoski, Rahul V. Kulkarni  

**Link**: [PDF](https://arxiv.org/pdf/2603.12325)  

**Abstract**: Efficient exploration remains a central challenge in reinforcement learning, serving as a useful pretraining objective for data collection, particularly when an external reward function is unavailable. A principled formulation of the exploration problem is to find policies that maximize the entropy of their induced steady-state visitation distribution, thereby encouraging uniform long-run coverage of the state space. Many existing exploration approaches require estimating state visitation frequencies through repeated on-policy rollouts, which can be computationally expensive. In this work, we instead consider an intrinsic average-reward formulation in which the reward is derived from the visitation distribution itself, so that the optimal policy maximizes steady-state entropy. An entropy-regularized version of this objective admits a spectral characterization: the relevant stationary distributions can be computed from the dominant eigenvectors of a problem-dependent transition matrix. This insight leads to a novel algorithm for solving the maximum entropy exploration problem, EVE (EigenVector-based Exploration), which avoids explicit rollouts and distribution estimation, instead computing the solution through iterative updates, similar to a value-based approach. To address the original unregularized objective, we employ a posterior-policy iteration (PPI) approach, which monotonically improves the entropy and converges in value. We prove convergence of EVE under standard assumptions and demonstrate empirically that it efficiently produces policies with high steady-state entropy, achieving competitive exploration performance relative to rollout-based baselines in deterministic grid-world environments. 

---
# Thermodynamics of Reinforcement Learning Curricula 

**Authors**: Jacob Adamczyk, Juan Sebastian Rojas, Rahul V. Kulkarni  

**Link**: [PDF](https://arxiv.org/pdf/2603.12324)  

**Abstract**: Connections between statistical mechanics and machine learning have repeatedly proven fruitful, providing insight into optimization, generalization, and representation learning. In this work, we follow this tradition by leveraging results from non-equilibrium thermodynamics to formalize curriculum learning in reinforcement learning (RL). In particular, we propose a geometric framework for RL by interpreting reward parameters as coordinates on a task manifold. We show that, by minimizing the excess thermodynamic work, optimal curricula correspond to geodesics in this task space. As an application of this framework, we provide an algorithm, "MEW" (Minimum Excess Work), to derive a principled schedule for temperature annealing in maximum-entropy RL. 

---
# VQQA: An Agentic Approach for Video Evaluation and Quality Improvement 

**Authors**: Yiwen Song, Tomas Pfister, Yale Song  

**Link**: [PDF](https://arxiv.org/pdf/2603.12310)  

**Abstract**: Despite rapid advancements in video generation models, aligning their outputs with complex user intent remains challenging. Existing test-time optimization methods are typically either computationally expensive or require white-box access to model internals. To address this, we present VQQA (Video Quality Question Answering), a unified, multi-agent framework generalizable across diverse input modalities and video generation tasks. By dynamically generating visual questions and using the resulting Vision-Language Model (VLM) critiques as semantic gradients, VQQA replaces traditional, passive evaluation metrics with human-interpretable, actionable feedback. This enables a highly efficient, closed-loop prompt optimization process via a black-box natural language interface. Extensive experiments demonstrate that VQQA effectively isolates and resolves visual artifacts, substantially improving generation quality in just a few refinement steps. Applicable to both text-to-video (T2V) and image-to-video (I2V) tasks, our method achieves absolute improvements of +11.57% on T2V-CompBench and +8.43% on VBench2 over vanilla generation, significantly outperforming state-of-the-art stochastic search and prompt optimization techniques. 

---
# HCP-DCNet: A Hierarchical Causal Primitive Dynamic Composition Network for Self-Improving Causal Understanding 

**Authors**: Ming Lei, Shufan Wu, Christophe Baehr  

**Link**: [PDF](https://arxiv.org/pdf/2603.12305)  

**Abstract**: The ability to understand and reason about cause and effect -- encompassing interventions, counterfactuals, and underlying mechanisms -- is a cornerstone of robust artificial intelligence. While deep learning excels at pattern recognition, it fundamentally lacks a model of causality, making systems brittle under distribution shifts and unable to answer ``what-if'' questions. This paper introduces the \emph{Hierarchical Causal Primitive Dynamic Composition Network (HCP-DCNet)}, a unified framework that bridges continuous physical dynamics with discrete symbolic causal inference. Departing from monolithic representations, HCP-DCNet decomposes causal scenes into reusable, typed \emph{causal primitives} organized into four abstraction layers: physical, functional, event, and rule. A dual-channel routing network dynamically composes these primitives into task-specific, fully differentiable \emph{Causal Execution Graphs (CEGs)}. Crucially, the system employs a \emph{causal-intervention-driven meta-evolution} strategy, enabling autonomous self-improvement through a constrained Markov decision process. We establish rigorous theoretical guarantees, including type-safe composition, routing convergence, and universal approximation of causal dynamics. Extensive experiments across simulated physical and social environments demonstrate that HCP-DCNet significantly outperforms state-of-the-art baselines in causal discovery, counterfactual reasoning, and compositional generalization. This work provides a principled, scalable, and interpretable architecture for building AI systems with human-like causal abstraction and continual self-refinement capabilities. 

---
# A Geometrically-Grounded Drive for MDL-Based Optimization in Deep Learning 

**Authors**: Ming Lei, Shufan Wu, Christophe Baehr  

**Link**: [PDF](https://arxiv.org/pdf/2603.12304)  

**Abstract**: This paper introduces a novel optimization framework that fundamentally integrates the Minimum Description Length (MDL) principle into the training dynamics of deep neural networks. Moving beyond its conventional role as a model selection criterion, we reformulate MDL as an active, adaptive driving force within the optimization process itself. The core of our method is a geometrically-grounded cognitive manifold whose evolution is governed by a \textit{coupled Ricci flow}, enriched with a novel \textit{MDL Drive} term derived from first principles. This drive, modulated by the task-loss gradient, creates a seamless harmony between data fidelity and model simplification, actively compressing the internal representation during training. We establish a comprehensive theoretical foundation, proving key properties including the monotonic decrease of description length (Theorem~\ref{thm:convergence}), a finite number of topological phase transitions via a geometric surgery protocol (Theorems~\ref{thm:surgery}, \ref{thm:ultimate_fate}), and the emergence of universal critical behavior (Theorem~\ref{thm:universality}). Furthermore, we provide a practical, computationally efficient algorithm with $O(N \log N)$ per-iteration complexity (Theorem~\ref{thm:complexity}), alongside guarantees for numerical stability (Theorem~\ref{thm:stability}) and exponential convergence under convexity assumptions (Theorem~\ref{thm:convergence_rate}). Empirical validation on synthetic regression and classification tasks confirms the theoretical predictions, demonstrating the algorithm's efficacy in achieving robust generalization and autonomous model simplification. This work provides a principled path toward more autonomous, generalizable, and interpretable AI systems by unifying geometric deep learning with information-theoretic principles. 

---
# Global Evolutionary Steering: Refining Activation Steering Control via Cross-Layer Consistency 

**Authors**: Xinyan Jiang, Wenjing Yu, Di Wang, Lijie Hu  

**Link**: [PDF](https://arxiv.org/pdf/2603.12298)  

**Abstract**: Activation engineering enables precise control over Large Language Models (LLMs) without the computational cost of fine-tuning. However, existing methods deriving vectors from static activation differences are susceptible to high-dimensional noise and layer-wise semantic drift, often capturing spurious correlations rather than the target intent. To address this, we propose Global Evolutionary Refined Steering (GER-steer), a training-free framework that grounded in the geometric stability of the network's representation evolution. GER-steer exploits this global signal to rectify raw steering vectors, effectively decoupling robust semantic intent from orthogonal artifacts. Extensive evaluations confirm that GER-steer consistently outperforms baselines, delivering superior efficacy and generalization without layer-specific tuning, establishing a universal solution for reliable model alignment. 

---
# Synthetic Data Generation for Brain-Computer Interfaces: Overview, Benchmarking, and Future Directions 

**Authors**: Ziwei Wang, Zhentao He, Xingyi He, Hongbin Wang, Tianwang Jia, Jingwei Luo, Siyang Li, Xiaoqing Chen, Dongrui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2603.12296)  

**Abstract**: Deep learning has achieved transformative performance across diverse domains, largely driven by the large-scale, high-quality training data. In contrast, the development of brain-computer interfaces (BCIs) is fundamentally constrained by the limited, heterogeneous, and privacy-sensitive neural recordings. Generating synthetic yet physiologically plausible brain signals has therefore emerged as a compelling way to mitigate data scarcity and enhance model capacity. This survey provides a comprehensive review of brain signal generation for BCIs, covering methodological taxonomies, benchmark experiments, evaluation metrics, and key applications. We systematically categorize existing generative algorithms into four types: knowledge-based, feature-based, model-based, and translation-based approaches. Furthermore, we benchmark existing brain signal generation approaches across four representative BCI paradigms to provide an objective performance comparison. Finally, we discuss the potentials and challenges of current generation approaches and prospect future research on accurate, data-efficient, and privacy-aware BCI systems. The benchmark codebase is publicized at this https URL. 

---
# Detecting Miscitation on the Scholarly Web through LLM-Augmented Text-Rich Graph Learning 

**Authors**: Huidong Wu, Haojia Xiang, Jingtong Gao, Xiangyu Zhao, Dengsheng Wu, Jianping Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.12290)  

**Abstract**: Scholarly web is a vast network of knowledge connected by citations. However, this system is increasingly compromised by miscitation, where references do not support or even contradict the claims they are cited for. Current miscitation detection methods, which primarily rely on semantic similarity or network anomalies, struggle to capture the nuanced relationship between a citation's context and its place in the wider network. While large language models (LLMs) offer powerful capabilities in semantic reasoning for this task, their deployment is hindered by hallucination risks and high computational costs. In this work, we introduce LLM-Augmented Graph Learning-based Miscitation Detector (LAGMiD), a novel framework that leverages LLMs for deep semantic reasoning over citation graphs and distills this knowledge into graph neural networks (GNNs) for efficient and scalable miscitation detection. Specifically, LAGMiD introduces an evidence-chain reasoning mechanism, which uses chain-of-thought prompting, to perform multi-hop citation tracing and assess semantic fidelity. To reduce LLM inference costs, we design a knowledge distillation method aligning GNN embeddings with intermediate LLM reasoning states. A collaborative learning strategy further routes complex cases to the LLM while optimizing the GNN for structure-based generalization. Experiments on three real-world benchmarks show that LAGMiD achieves state-of-the-art miscitation detection with significantly reduced inference cost. 

---
# From Garbage to Gold: A Data-Architectural Theory of Predictive Robustness 

**Authors**: Terrence J. Lee-St. John, Jordan L. Lawson, Bartlomiej Piechowski-Jozwiak  

**Link**: [PDF](https://arxiv.org/pdf/2603.12288)  

**Abstract**: Tabular machine learning presents a paradox: modern models achieve state-of-the-art performance using high-dimensional (high-D), collinear, error-prone data, defying the "Garbage In, Garbage Out" mantra. To help resolve this, we synthesize principles from Information Theory, Latent Factor Models, and Psychometrics, clarifying that predictive robustness arises not solely from data cleanliness, but from the synergy between data architecture and model capacity. Partitioning predictor-space "noise" into "Predictor Error" and "Structural Uncertainty" (informational deficits from stochastic generative mappings), we prove that leveraging high-D sets of error-prone predictors asymptotically overcomes both types of noise, whereas cleaning a low-D set is fundamentally bounded by Structural Uncertainty. We demonstrate why "Informative Collinearity" (dependencies from shared latent causes) enhances reliability and convergence efficiency, and explain why increased dimensionality reduces the latent inference burden, enabling feasibility with finite samples. To address practical constraints, we propose "Proactive Data-Centric AI" to identify predictors that enable robustness efficiently. We also derive boundaries for Systematic Error Regimes and show why models that absorb "rogue" dependencies can mitigate assumption violations. Linking latent architecture to Benign Overfitting, we offer a first step towards a unified view of robustness to Outcome Error and predictor-space noise, while also delineating when traditional DCAI's focus on label cleaning remains powerful. By redefining data quality from item-level perfection to portfolio-level architecture, we provide a theoretical rationale for "Local Factories" -- learning from live, uncurated enterprise "data swamps" -- supporting a deployment paradigm shift from "Model Transfer" to "Methodology Transfer'' to overcome static generalizability limitations. 

---
# The DIME Architecture: A Unified Operational Algorithm for Neural Representation, Dynamics, Control and Integration 

**Authors**: Ionel Cristian Vladu, Nicu Bizdoaca, Ionica Pirici, Tudor-Adrian Balseanu, Eduard Nicusor Bondoc  

**Link**: [PDF](https://arxiv.org/pdf/2603.12286)  

**Abstract**: Modern neuroscience has accumulated extensive evidence on perception, memory, prediction, valuation, and consciousness, yet still lacks an explicit operational architecture capable of integrating these phenomena within a unified computational framework. Existing theories address specific aspects of neural function: predictive coding and active inference emphasize hierarchical inference and prediction error minimization; engram theories explain memory through distributed cell assemblies; neuromodulatory accounts focus on value-dependent regulation of plasticity and behaviour; and global workspace or large-scale network models investigate mechanisms underlying conscious access. Despite their explanatory power, these approaches remain only partially integrated at the architectural level. This work introduces DIME (Detect-Integrate-Mark-Execute), a neural architecture organizing perception, memory, valuation, and conscious access within a common operational cycle. The framework includes four interacting components: engrams, distributed recurrent neural structures supporting multiple activation trajectories; execution threads, spatiotemporal trajectories implementing neural processes; marker systems, neuromodulatory and limbic mechanisms regulating gain, plasticity, and trajectory selection; and hyperengrams, large-scale integrative states associated with operational conscious access. The framework is consistent with empirical evidence from hippocampal indexing, recurrent cortical processing, replay phenomena, large-scale network integration, and neuromodulatory regulation. Formulated at an abstract computational level, DIME may also inform artificial intelligence and robotics by providing an architectural template in which representation, valuation, and temporal sequencing emerge from a unified mechanism. An extended theoretical exposition is available in a companion monograph on Zenodo. 

---
# Predictive Analytics for Foot Ulcers Using Time-Series Temperature and Pressure Data 

**Authors**: Md Tanvir Hasan Turja  

**Link**: [PDF](https://arxiv.org/pdf/2603.12278)  

**Abstract**: Diabetic foot ulcers (DFUs) are a severe complication of diabetes, often resulting in significant morbidity. This paper presents a predictive analytics framework utilizing time-series data captured by wearable foot sensors -- specifically NTC thin-film thermocouples for temperature measurement and FlexiForce pressure sensors for plantar load monitoring. Data was collected from healthy subjects walking on an instrumented pathway. Unsupervised machine learning algorithms, Isolation Forest and K-Nearest Neighbors (KNN), were applied to detect anomalies that may indicate early ulcer risk. Through rigorous data preprocessing and targeted feature engineering, physiologic patterns were extracted to identify subtle changes in foot temperature and pressure. Results demonstrate Isolation Forest is sensitive to micro-anomalies, while KNN is effective in flagging extreme deviations, albeit at a higher false-positive rate. Strong correlations between temperature and pressure readings support combined sensor monitoring for improved predictive accuracy. These findings provide a basis for real-time diabetic foot health surveillance, aiming to facilitate earlier intervention and reduce DFU incidence. 

---
# Prompt Injection as Role Confusion 

**Authors**: Charles Ye, Jasmine Cui, Dylan Hadfield-Menell  

**Link**: [PDF](https://arxiv.org/pdf/2603.12277)  

**Abstract**: Language models remain vulnerable to prompt injection attacks despite extensive safety training. We trace this failure to role confusion: models infer roles from how text is written, not where it comes from. We design novel role probes to capture how models internally identify "who is speaking." These reveal why prompt injection works: untrusted text that imitates a role inherits that role's authority. We test this insight by injecting spoofed reasoning into user prompts and tool outputs, achieving average success rates of 60% on StrongREJECT and 61% on agent exfiltration, across multiple open- and closed-weight models with near-zero baselines. Strikingly, the degree of internal role confusion strongly predicts attack success before generation begins. Our findings reveal a fundamental gap: security is defined at the interface but authority is assigned in latent space. More broadly, we introduce a unifying, mechanistic framework for prompt injection, demonstrating that diverse prompt-injection attacks exploit the same underlying role-confusion mechanism. 

---
# Aligning Language Models from User Interactions 

**Authors**: Thomas Kleine Buening, Jonas Hübotter, Barna Pásztor, Idan Shenfeld, Giorgia Ramponi, Andreas Krause  

**Link**: [PDF](https://arxiv.org/pdf/2603.12273)  

**Abstract**: Multi-turn user interactions are among the most abundant data produced by language models, yet we lack effective methods to learn from them. While typically discarded, these interactions often contain useful information: follow-up user messages may indicate that a response was incorrect, failed to follow an instruction, or did not align with the user's preferences. Importantly, language models are already able to make use of this information in context. After observing a user's follow-up, the same model is often able to revise its behavior. We leverage this ability to propose a principled and scalable method for learning directly from user interactions through self-distillation. By conditioning the model on the user's follow-up message and comparing the resulting token distribution with the original policy, we obtain a target for updating the policy that captures how the model's behavior changes in hindsight. We then distill this hindsight distribution back into the current policy. Remarkably, we show that training on real-world user conversations from WildChat improves language models across standard alignment and instruction-following benchmarks, without regressing other capabilities. The same mechanism enables personalization, allowing models to continually adapt to individual users through interaction without explicit feedback. Our results demonstrate that raw user interactions that arise naturally during deployment enable alignment, personalization, and continual adaptation. 

---
# Diagnosing Retrieval Bias Under Multiple In-Context Knowledge Updates in Large Language Models 

**Authors**: Boyu Qiao, Sean Guo, Xian Yang, Kun Li, Wei Zhou, Songlin Hu, Yunya Song  

**Link**: [PDF](https://arxiv.org/pdf/2603.12271)  

**Abstract**: LLMs are widely used in knowledge-intensive tasks where the same fact may be revised multiple times within context. Unlike prior work focusing on one-shot updates or single conflicts, multi-update scenarios contain multiple historically valid versions that compete at retrieval, yet remain underexplored. This challenge resembles the AB-AC interference paradigm in cognitive psychology: when the same cue A is successively associated with B and C, the old and new associations compete during retrieval, leading to bias. Inspired by this, we introduce a Dynamic Knowledge Instance (DKI) evaluation framework, modeling multi-updates of the same fact as a cue paired with a sequence of updated values, and assess models via endpoint probing of the earliest (initial) and latest (current) states. Across diverse LLMs, we observe that retrieval bias intensifies as updates increase, earliest-state accuracy stays high while latest-state accuracy drops substantially. Diagnostic analyses of attention, hidden-state similarity, and output logits further reveal that these signals become flatter and weakly discriminative on errors, providing little stable basis for identifying the latest update. Finally, cognitively inspired heuristic intervention strategies yield only modest gains and do not eliminate the bias. Our results reveal a persistent challenge in tracking and following knowledge updates in long contexts. 

---
# Task-Specific Knowledge Distillation via Intermediate Probes 

**Authors**: Ryan Brown, Chris Russell  

**Link**: [PDF](https://arxiv.org/pdf/2603.12270)  

**Abstract**: Knowledge distillation from large language models (LLMs) assumes that the teacher's output distribution is a high-quality training signal. On reasoning tasks, this assumption is frequently violated. A model's intermediate representations may encode the correct answer, yet this information is lost or distorted through the vocabulary projection, where prompt formatting and answer-token choices creates brittle, noisy outputs.
We introduce \method{}, a distillation framework that bypasses this bottleneck by training lightweight probes on frozen teacher hidden states and using the probe's predictions, rather than output logits, as supervision for student training. This simple change yields consistent improvements across four reasoning benchmarks (AQuA-RAT, ARC Easy/Challenge, and MMLU), with gains most pronounced under limited data.
Probes trained on intermediate representations provide cleaner labels than the teacher's own outputs, effectively denoising the distillation signal. \method{} requires no architectural changes to student or teacher, is architecture-agnostic, and adds minimal compute since probe training is cheap and teacher representations can be cached. By exploiting internal representations, \method{} enables practitioners to extract more value from large teacher models without additional training data or architectural complexity. 

---
# DART: Input-Difficulty-AwaRe Adaptive Threshold for Early-Exit DNNs 

**Authors**: Parth Patne, Mahdi Taheri, Christian Herglotz, Maksim Jenihhin, Milos Krstic, Michael Hübner  

**Link**: [PDF](https://arxiv.org/pdf/2603.12269)  

**Abstract**: Early-exit deep neural networks enable adaptive inference by terminating computation when sufficient confidence is achieved, reducing cost for edge AI accelerators in resource-constrained settings. Existing methods, however, rely on suboptimal exit policies, ignore input difficulty, and optimize thresholds independently. This paper introduces DART (Input-Difficulty-Aware Adaptive Threshold), a framework that overcomes these limitations. DART introduces three key innovations: (1) a lightweight difficulty estimation module that quantifies input complexity with minimal computational overhead, (2) a joint exit policy optimization algorithm based on dynamic programming, and (3) an adaptive coefficient management system. Experiments on diverse DNN benchmarks (AlexNet, ResNet-18, VGG-16) demonstrate that DART achieves up to \textbf{3.3$\times$} speedup, \textbf{5.1$\times$} lower energy, and up to \textbf{42\%} lower average power compared to static networks, while preserving competitive accuracy. Extending DART to Vision Transformers (LeViT) yields power (5.0$\times$) and execution-time (3.6$\times$) gains but also accuracy loss (up to 17 percent), underscoring the need for transformer-specific early-exit mechanisms. We further introduce the Difficulty-Aware Efficiency Score (DAES), a novel multi-objective metric, under which DART achieves up to a 14.8 improvement over baselines, highlighting superior accuracy, efficiency, and robustness trade-offs. 

---
