# Online Submission and Evaluation System Design for Competition Operations 

**Authors**: Zhe Chen, Daniel Harabor, Ryan Hechnenberger, Nathan R. Sturtevant  

**Link**: [PDF](https://arxiv.org/pdf/2507.17730)  

**Abstract**: Research communities have developed benchmark datasets across domains to compare the performance of algorithms and techniques However, tracking the progress in these research areas is not easy, as publications appear in different venues at the same time, and many of them claim to represent the state-of-the-art. To address this, research communities often organise periodic competitions to evaluate the performance of various algorithms and techniques, thereby tracking advancements in the field. However, these competitions pose a significant operational burden. The organisers must manage and evaluate a large volume of submissions. Furthermore, participants typically develop their solutions in diverse environments, leading to compatibility issues during the evaluation of their submissions. This paper presents an online competition system that automates the submission and evaluation process for a competition. The competition system allows organisers to manage large numbers of submissions efficiently, utilising isolated environments to evaluate submissions. This system has already been used successfully for several competitions, including the Grid-Based Pathfinding Competition and the League of Robot Runners competition. 

---
# Thinking Isn't an Illusion: Overcoming the Limitations of Reasoning Models via Tool Augmentations 

**Authors**: Zhao Song, Song Yue, Jiahao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.17699)  

**Abstract**: Large Reasoning Models (LRMs) have become a central focus in today's large language model (LLM) research, where models are designed to output a step-by-step thinking process before arriving at a final answer to handle complex reasoning tasks. Despite their promise, recent empirical studies (e.g., [Shojaee et al., 2025] from Apple) suggest that this thinking process may not actually enhance reasoning ability, where LLMs without explicit reasoning actually outperform LRMs on tasks with low or high complexity. In this work, we revisit these findings and investigate whether the limitations of LRMs persist when tool augmentations are introduced. We incorporate two types of tools, Python interpreters and scratchpads, and evaluate three representative LLMs and their LRM counterparts on Apple's benchmark reasoning puzzles. Our results show that, with proper tool use, LRMs consistently outperform their non-reasoning counterparts across all levels of task complexity. These findings challenge the recent narrative that reasoning is an illusion and highlight the potential of tool-augmented LRMs for solving complex problems. 

---
# Symbiotic Agents: A Novel Paradigm for Trustworthy AGI-driven Networks 

**Authors**: Ilias Chatzistefanidis, Navid Nikaein  

**Link**: [PDF](https://arxiv.org/pdf/2507.17695)  

**Abstract**: Large Language Model (LLM)-based autonomous agents are expected to play a vital role in the evolution of 6G networks, by empowering real-time decision-making related to management and service provisioning to end-users. This shift facilitates the transition from a specialized intelligence approach, where artificial intelligence (AI) algorithms handle isolated tasks, to artificial general intelligence (AGI)-driven networks, where agents possess broader reasoning capabilities and can manage diverse network functions. In this paper, we introduce a novel agentic paradigm that combines LLMs with real-time optimization algorithms towards Trustworthy AI, defined as symbiotic agents. Optimizers at the LLM's input-level provide bounded uncertainty steering for numerically precise tasks, whereas output-level optimizers supervised by the LLM enable adaptive real-time control. We design and implement two novel agent types including: (i) Radio Access Network optimizers, and (ii) multi-agent negotiators for Service-Level Agreements (SLAs). We further propose an end-to-end architecture for AGI networks and evaluate it on a 5G testbed capturing channel fluctuations from moving vehicles. Results show that symbiotic agents reduce decision errors fivefold compared to standalone LLM-based agents, while smaller language models (SLM) achieve similar accuracy with a 99.9% reduction in GPU resource overhead and in near-real-time loops of 82 ms. A multi-agent demonstration for collaborative RAN on the real-world testbed highlights significant flexibility in service-level agreement and resource allocation, reducing RAN over-utilization by approximately 44%. Drawing on our findings and open-source implementations, we introduce the symbiotic paradigm as the foundation for next-generation, AGI-driven networks-systems designed to remain adaptable, efficient, and trustworthy even as LLMs advance. 

---
# Simulating multiple human perspectives in socio-ecological systems using large language models 

**Authors**: Yongchao Zeng, Calum Brown, Ioannis Kyriakou, Ronja Hotz, Mark Rounsevell  

**Link**: [PDF](https://arxiv.org/pdf/2507.17680)  

**Abstract**: Understanding socio-ecological systems requires insights from diverse stakeholder perspectives, which are often hard to access. To enable alternative, simulation-based exploration of different stakeholder perspectives, we develop the HoPeS (Human-Oriented Perspective Shifting) modelling framework. HoPeS employs agents powered by large language models (LLMs) to represent various stakeholders; users can step into the agent roles to experience perspectival differences. A simulation protocol serves as a "scaffold" to streamline multiple perspective-taking simulations, supporting users in reflecting on, transitioning between, and integrating across perspectives. A prototype system is developed to demonstrate HoPeS in the context of institutional dynamics and land use change, enabling both narrative-driven and numerical experiments. In an illustrative experiment, a user successively adopts the perspectives of a system observer and a researcher - a role that analyses data from the embedded land use model to inform evidence-based decision-making for other LLM agents representing various institutions. Despite the user's effort to recommend technically sound policies, discrepancies persist between the policy recommendation and implementation due to stakeholders' competing advocacies, mirroring real-world misalignment between researcher and policymaker perspectives. The user's reflection highlights the subjective feelings of frustration and disappointment as a researcher, especially due to the challenge of maintaining political neutrality while attempting to gain political influence. Despite this, the user exhibits high motivation to experiment with alternative narrative framing strategies, suggesting the system's potential in exploring different perspectives. Further system and protocol refinement are likely to enable new forms of interdisciplinary collaboration in socio-ecological simulations. 

---
# Constructing Ophthalmic MLLM for Positioning-diagnosis Collaboration Through Clinical Cognitive Chain Reasoning 

**Authors**: Xinyao Liu, Diping Song  

**Link**: [PDF](https://arxiv.org/pdf/2507.17539)  

**Abstract**: Multimodal large language models (MLLMs) demonstrate significant potential in the field of medical diagnosis. However, they face critical challenges in specialized domains such as ophthalmology, particularly the fragmentation of annotation granularity and inconsistencies in clinical reasoning logic, which hinder precise cross-modal understanding. This paper introduces FundusExpert, an ophthalmology-specific MLLM with integrated positioning-diagnosis reasoning capabilities, along with FundusGen, a dataset constructed through the intelligent Fundus-Engine system. Fundus-Engine automates localization and leverages MLLM-based semantic expansion to integrate global disease classification, local object detection, and fine-grained feature analysis within a single fundus image. Additionally, by constructing a clinically aligned cognitive chain, it guides the model to generate interpretable reasoning paths. FundusExpert, fine-tuned with instruction data from FundusGen, achieves the best performance in ophthalmic question-answering tasks, surpassing the average accuracy of the 40B MedRegA by 26.6%. It also excels in zero-shot report generation tasks, achieving a clinical consistency of 77.0%, significantly outperforming GPT-4o's 47.6%. Furthermore, we reveal a scaling law between data quality and model capability ($L \propto N^{0.068}$), demonstrating that the cognitive alignment annotations in FundusGen enhance data utilization efficiency. By integrating region-level localization with diagnostic reasoning chains, our work develops a scalable, clinically-aligned MLLM and explores a pathway toward bridging the visual-language gap in specific MLLMs. Our project can be found at this https URL. 

---
# TAI Scan Tool: A RAG-Based Tool With Minimalistic Input for Trustworthy AI Self-Assessment 

**Authors**: Athanasios Davvetas, Xenia Ziouvelou, Ypatia Dami, Alexis Kaponis, Konstantina Giouvanopoulou, Michael Papademas  

**Link**: [PDF](https://arxiv.org/pdf/2507.17514)  

**Abstract**: This paper introduces the TAI Scan Tool, a RAG-based TAI self-assessment tool with minimalistic input. The current version of the tool supports the legal TAI assessment, with a particular emphasis on facilitating compliance with the AI Act. It involves a two-step approach with a pre-screening and an assessment phase. The assessment output of the system includes insight regarding the risk-level of the AI system according to the AI Act, while at the same time retrieving relevant articles to aid with compliance and notify on their obligations. Our qualitative evaluation using use-case scenarios yields promising results, correctly predicting risk levels while retrieving relevant articles across three distinct semantic groups. Furthermore, interpretation of results shows that the tool's reasoning relies on comparison with the setting of high-risk systems, a behaviour attributed to their deployment requiring careful consideration, and therefore frequently presented within the AI Act. 

---
# Can One Domain Help Others? A Data-Centric Study on Multi-Domain Reasoning via Reinforcement Learning 

**Authors**: Yu Li, Zhuoshi Pan, Honglin Lin, Mengyuan Sun, Conghui He, Lijun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.17512)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a powerful paradigm for enhancing the reasoning capabilities of LLMs. Existing research has predominantly concentrated on isolated reasoning domains such as mathematical problem-solving, coding tasks, or logical reasoning. However, real world reasoning scenarios inherently demand an integrated application of multiple cognitive skills. Despite this, the interplay among these reasoning skills under reinforcement learning remains poorly understood. To bridge this gap, we present a systematic investigation of multi-domain reasoning within the RLVR framework, explicitly focusing on three primary domains: mathematical reasoning, code generation, and logical puzzle solving. We conduct a comprehensive study comprising four key components: (1) Leveraging the GRPO algorithm and the Qwen-2.5-7B model family, our study thoroughly evaluates the models' in-domain improvements and cross-domain generalization capabilities when trained on single-domain datasets. (2) Additionally, we examine the intricate interactions including mutual enhancements and conflicts that emerge during combined cross-domain training. (3) To further understand the influence of SFT on RL, we also analyze and compare performance differences between base and instruct models under identical RL configurations. (4) Furthermore, we delve into critical RL training details, systematically exploring the impacts of curriculum learning strategies, variations in reward design, and language-specific factors. Through extensive experiments, our results offer significant insights into the dynamics governing domain interactions, revealing key factors influencing both specialized and generalizable reasoning performance. These findings provide valuable guidance for optimizing RL methodologies to foster comprehensive, multi-domain reasoning capabilities in LLMs. 

---
# Automated Hybrid Grounding Using Structural and Data-Driven Heuristics 

**Authors**: Alexander Beiser, Markus Hecher, Stefan Woltran  

**Link**: [PDF](https://arxiv.org/pdf/2507.17493)  

**Abstract**: The grounding bottleneck poses one of the key challenges that hinders the widespread adoption of Answer Set Programming in industry. Hybrid Grounding is a step in alleviating the bottleneck by combining the strength of standard bottom-up grounding with recently proposed techniques where rule bodies are decoupled during grounding. However, it has remained unclear when hybrid grounding shall use body-decoupled grounding and when to use standard bottom-up grounding. In this paper, we address this issue by developing automated hybrid grounding: we introduce a splitting algorithm based on data-structural heuristics that detects when to use body-decoupled grounding and when standard grounding is beneficial. We base our heuristics on the structure of rules and an estimation procedure that incorporates the data of the instance. The experiments conducted on our prototypical implementation demonstrate promising results, which show an improvement on hard-to-ground scenarios, whereas on hard-to-solve instances we approach state-of-the-art performance. 

---
# CQE under Epistemic Dependencies: Algorithms and Experiments (extended version) 

**Authors**: Lorenzo Marconi, Flavia Ricci, Riccardo Rosati  

**Link**: [PDF](https://arxiv.org/pdf/2507.17487)  

**Abstract**: We investigate Controlled Query Evaluation (CQE) over ontologies, where information disclosure is regulated by epistemic dependencies (EDs), a family of logical rules recently proposed for the CQE framework. In particular, we combine EDs with the notion of optimal GA censors, i.e. maximal sets of ground atoms that are entailed by the ontology and can be safely revealed. We focus on answering Boolean unions of conjunctive queries (BUCQs) with respect to the intersection of all optimal GA censors - an approach that has been shown in other contexts to ensure strong security guarantees with favorable computational behavior. First, we characterize the security of this intersection-based approach and identify a class of EDs (namely, full EDs) for which it remains safe. Then, for a subclass of EDs and for DL-Lite_R ontologies, we show that answering BUCQs in the above CQE semantics is in AC^0 in data complexity by presenting a suitable, detailed first-order rewriting algorithm. Finally, we report on experiments conducted in two different evaluation scenarios, showing the practical feasibility of our rewriting function. 

---
# LTLZinc: a Benchmarking Framework for Continual Learning and Neuro-Symbolic Temporal Reasoning 

**Authors**: Luca Salvatore Lorello, Nikolaos Manginas, Marco Lippi, Stefano Melacci  

**Link**: [PDF](https://arxiv.org/pdf/2507.17482)  

**Abstract**: Neuro-symbolic artificial intelligence aims to combine neural architectures with symbolic approaches that can represent knowledge in a human-interpretable formalism. Continual learning concerns with agents that expand their knowledge over time, improving their skills while avoiding to forget previously learned concepts. Most of the existing approaches for neuro-symbolic artificial intelligence are applied to static scenarios only, and the challenging setting where reasoning along the temporal dimension is necessary has been seldom explored. In this work we introduce LTLZinc, a benchmarking framework that can be used to generate datasets covering a variety of different problems, against which neuro-symbolic and continual learning methods can be evaluated along the temporal and constraint-driven dimensions. Our framework generates expressive temporal reasoning and continual learning tasks from a linear temporal logic specification over MiniZinc constraints, and arbitrary image classification datasets. Fine-grained annotations allow multiple neural and neuro-symbolic training settings on the same generated datasets. Experiments on six neuro-symbolic sequence classification and four class-continual learning tasks generated by LTLZinc, demonstrate the challenging nature of temporal learning and reasoning, and highlight limitations of current state-of-the-art methods. We release the LTLZinc generator and ten ready-to-use tasks to the neuro-symbolic and continual learning communities, in the hope of fostering research towards unified temporal learning and reasoning frameworks. 

---
# An Uncertainty-Driven Adaptive Self-Alignment Framework for Large Language Models 

**Authors**: Haoran Sun, Zekun Zhang, Shaoning Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2507.17477)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable progress in instruction following and general-purpose reasoning. However, achieving high-quality alignment with human intent and safety norms without human annotations remains a fundamental challenge. In this work, we propose an Uncertainty-Driven Adaptive Self-Alignment (UDASA) framework designed to improve LLM alignment in a fully automated manner. UDASA first generates multiple responses for each input and quantifies output uncertainty across three dimensions: semantics, factuality, and value alignment. Based on these uncertainty scores, the framework constructs preference pairs and categorizes training samples into three stages, conservative, moderate, and exploratory, according to their uncertainty difference. The model is then optimized progressively across these stages. In addition, we conduct a series of preliminary studies to validate the core design assumptions and provide strong empirical motivation for the proposed framework. Experimental results show that UDASA outperforms existing alignment methods across multiple tasks, including harmlessness, helpfulness, truthfulness, and controlled sentiment generation, significantly improving model performance. 

---
# Ctx2TrajGen: Traffic Context-Aware Microscale Vehicle Trajectories using Generative Adversarial Imitation Learning 

**Authors**: Joobin Jin, Seokjun Hong, Gyeongseon Baek, Yeeun Kim, Byeongjoon Noh  

**Link**: [PDF](https://arxiv.org/pdf/2507.17418)  

**Abstract**: Precise modeling of microscopic vehicle trajectories is critical for traffic behavior analysis and autonomous driving systems. We propose Ctx2TrajGen, a context-aware trajectory generation framework that synthesizes realistic urban driving behaviors using GAIL. Leveraging PPO and WGAN-GP, our model addresses nonlinear interdependencies and training instability inherent in microscopic settings. By explicitly conditioning on surrounding vehicles and road geometry, Ctx2TrajGen generates interaction-aware trajectories aligned with real-world context. Experiments on the drone-captured DRIFT dataset demonstrate superior performance over existing methods in terms of realism, behavioral diversity, and contextual fidelity, offering a robust solution to data scarcity and domain shift without simulation. 

---
# Compliance Brain Assistant: Conversational Agentic AI for Assisting Compliance Tasks in Enterprise Environments 

**Authors**: Shitong Zhu, Chenhao Fang, Derek Larson, Neel Reddy Pochareddy, Rajeev Rao, Sophie Zeng, Yanqing Peng, Wendy Summer, Alex Goncalves, Arya Pudota, Herve Robert  

**Link**: [PDF](https://arxiv.org/pdf/2507.17289)  

**Abstract**: This paper presents Compliance Brain Assistant (CBA), a conversational, agentic AI assistant designed to boost the efficiency of daily compliance tasks for personnel in enterprise environments. To strike a good balance between response quality and latency, we design a user query router that can intelligently choose between (i) FastTrack mode: to handle simple requests that only need additional relevant context retrieved from knowledge corpora; and (ii) FullAgentic mode: to handle complicated requests that need composite actions and tool invocations to proactively discover context across various compliance artifacts, and/or involving other APIs/models for accommodating requests. A typical example would be to start with a user query, use its description to find a specific entity and then use the entity's information to query other APIs for curating and enriching the final AI response.
Our experimental evaluations compared CBA against an out-of-the-box LLM on various real-world privacy/compliance-related queries targeting various personas. We found that CBA substantially improved upon the vanilla LLM's performance on metrics such as average keyword match rate (83.7% vs. 41.7%) and LLM-judge pass rate (82.0% vs. 20.0%). We also compared metrics for the full routing-based design against the `fast-track only` and `full-agentic` modes and found that it had a better average match-rate and pass-rate while keeping the run-time approximately the same. This finding validated our hypothesis that the routing mechanism leads to a good trade-off between the two worlds. 

---
# Students' Feedback Requests and Interactions with the SCRIPT Chatbot: Do They Get What They Ask For? 

**Authors**: Andreas Scholl, Natalie Kiesler  

**Link**: [PDF](https://arxiv.org/pdf/2507.17258)  

**Abstract**: Building on prior research on Generative AI (GenAI) and related tools for programming education, we developed SCRIPT, a chatbot based on ChatGPT-4o-mini, to support novice learners. SCRIPT allows for open-ended interactions and structured guidance through predefined prompts. We evaluated the tool via an experiment with 136 students from an introductory programming course at a large German university and analyzed how students interacted with SCRIPT while solving programming tasks with a focus on their feedback preferences. The results reveal that students' feedback requests seem to follow a specific sequence. Moreover, the chatbot responses aligned well with students' requested feedback types (in 75%), and it adhered to the system prompt constraints. These insights inform the design of GenAI-based learning support systems and highlight challenges in balancing guidance and flexibility in AI-assisted tools. 

---
# Agent Identity Evals: Measuring Agentic Identity 

**Authors**: Elija Perrier, Michael Timothy Bennett  

**Link**: [PDF](https://arxiv.org/pdf/2507.17257)  

**Abstract**: Central to agentic capability and trustworthiness of language model agents (LMAs) is the extent they maintain stable, reliable, identity over time. However, LMAs inherit pathologies from large language models (LLMs) (statelessness, stochasticity, sensitivity to prompts and linguistically-intermediation) which can undermine their identifiability, continuity, persistence and consistency. This attrition of identity can erode their reliability, trustworthiness and utility by interfering with their agentic capabilities such as reasoning, planning and action. To address these challenges, we introduce \textit{agent identity evals} (AIE), a rigorous, statistically-driven, empirical framework for measuring the degree to which an LMA system exhibit and maintain their agentic identity over time, including their capabilities, properties and ability to recover from state perturbations. AIE comprises a set of novel metrics which can integrate with other measures of performance, capability and agentic robustness to assist in the design of optimal LMA infrastructure and scaffolding such as memory and tools. We set out formal definitions and methods that can be applied at each stage of the LMA life-cycle, and worked examples of how to apply them. 

---
# Our Cars Can Talk: How IoT Brings AI to Vehicles 

**Authors**: Amod Kant Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2507.17214)  

**Abstract**: Bringing AI to vehicles and enabling them as sensing platforms is key to transforming maintenance from reactive to proactive. Now is the time to integrate AI copilots that speak both languages: machine and driver. This article offers a conceptual and technical perspective intended to spark interdisciplinary dialogue and guide future research and development in intelligent vehicle systems, predictive maintenance, and AI-powered user interaction. 

---
# Improving LLMs' Generalized Reasoning Abilities by Graph Problems 

**Authors**: Qifan Zhang, Nuo Chen, Zehua Li, Miao Peng, Jing Tang, Jia Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.17168)  

**Abstract**: Large Language Models (LLMs) have made remarkable strides in reasoning tasks, yet their performance often falters on novel and complex problems. Domain-specific continued pretraining (CPT) methods, such as those tailored for mathematical reasoning, have shown promise but lack transferability to broader reasoning tasks. In this work, we pioneer the use of Graph Problem Reasoning (GPR) to enhance the general reasoning capabilities of LLMs. GPR tasks, spanning pathfinding, network analysis, numerical computation, and topological reasoning, require sophisticated logical and relational reasoning, making them ideal for teaching diverse reasoning patterns. To achieve this, we introduce GraphPile, the first large-scale corpus specifically designed for CPT using GPR data. Spanning 10.9 billion tokens across 23 graph tasks, the dataset includes chain-of-thought, program-of-thought, trace of execution, and real-world graph data. Using GraphPile, we train GraphMind on popular base models Llama 3 and 3.1, as well as Gemma 2, achieving up to 4.9 percent higher accuracy in mathematical reasoning and up to 21.2 percent improvement in non-mathematical reasoning tasks such as logical and commonsense reasoning. By being the first to harness GPR for enhancing reasoning patterns and introducing the first dataset of its kind, our work bridges the gap between domain-specific pretraining and universal reasoning capabilities, advancing the adaptability and robustness of LLMs. 

---
# HySafe-AI: Hybrid Safety Architectural Analysis Framework for AI Systems: A Case Study 

**Authors**: Mandar Pitale, Jelena Frtunikj, Abhinaw Priyadershi, Vasu Singh, Maria Spence  

**Link**: [PDF](https://arxiv.org/pdf/2507.17118)  

**Abstract**: AI has become integral to safety-critical areas like autonomous driving systems (ADS) and robotics. The architecture of recent autonomous systems are trending toward end-to-end (E2E) monolithic architectures such as large language models (LLMs) and vision language models (VLMs). In this paper, we review different architectural solutions and then evaluate the efficacy of common safety analyses such as failure modes and effect analysis (FMEA) and fault tree analysis (FTA). We show how these techniques can be improved for the intricate nature of the foundational models, particularly in how they form and utilize latent representations. We introduce HySAFE-AI, Hybrid Safety Architectural Analysis Framework for AI Systems, a hybrid framework that adapts traditional methods to evaluate the safety of AI systems. Lastly, we offer hints of future work and suggestions to guide the evolution of future AI safety standards. 

---
# LoRA is All You Need for Safety Alignment of Reasoning LLMs 

**Authors**: Yihao Xue, Baharan Mirzasoleiman  

**Link**: [PDF](https://arxiv.org/pdf/2507.17075)  

**Abstract**: Reasoning LLMs have demonstrated remarkable breakthroughs in solving complex problems that were previously out of reach. To ensure LLMs do not assist with harmful requests, safety alignment fine-tuning is necessary in the post-training phase. However, safety alignment fine-tuning has recently been shown to significantly degrade reasoning abilities, a phenomenon known as the "Safety Tax". In this work, we show that using LoRA for SFT on refusal datasets effectively aligns the model for safety without harming its reasoning capabilities. This is because restricting the safety weight updates to a low-rank space minimizes the interference with the reasoning weights. Our extensive experiments across four benchmarks covering math, science, and coding show that this approach produces highly safe LLMs -- with safety levels comparable to full-model fine-tuning -- without compromising their reasoning abilities. Additionally, we observe that LoRA induces weight updates with smaller overlap with the initial weights compared to full-model fine-tuning. We also explore methods that further reduce such overlap -- via regularization or during weight merging -- and observe some improvement on certain tasks. We hope this result motivates designing approaches that yield more consistent improvements in the reasoning-safety trade-off. 

---
# New Mechanisms in Flex Distribution for Bounded Suboptimal Multi-Agent Path Finding 

**Authors**: Shao-Hung Chan, Thomy Phan, Jiaoyang Li, Sven Koenig  

**Link**: [PDF](https://arxiv.org/pdf/2507.17054)  

**Abstract**: Multi-Agent Path Finding (MAPF) is the problem of finding a set of collision-free paths, one for each agent in a shared environment. Its objective is to minimize the sum of path costs (SOC), where the path cost of each agent is defined as the travel time from its start location to its target location. Explicit Estimation Conflict-Based Search (EECBS) is the leading algorithm for bounded-suboptimal MAPF, with the SOC of the solution being at most a user-specified factor $w$ away from optimal. EECBS maintains sets of paths and a lower bound $LB$ on the optimal SOC. Then, it iteratively selects a set of paths whose SOC is at most $w \cdot LB$ and introduces constraints to resolve collisions. For each path in a set, EECBS maintains a lower bound on its optimal path that satisfies constraints. By finding an individually bounded-suboptimal path with cost at most a threshold of $w$ times its lower bound, EECBS guarantees to find a bounded-suboptimal solution. To speed up EECBS, previous work uses flex distribution to increase the threshold. Though EECBS with flex distribution guarantees to find a bounded-suboptimal solution, increasing the thresholds may push the SOC beyond $w \cdot LB$, forcing EECBS to switch among different sets of paths instead of resolving collisions on a particular set of paths, and thus reducing efficiency. To address this issue, we propose Conflict-Based Flex Distribution that distributes flex in proportion to the number of collisions. We also estimate the delays needed to satisfy constraints and propose Delay-Based Flex Distribution. On top of that, we propose Mixed-Strategy Flex Distribution, combining both in a hierarchical framework. We prove that EECBS with our new flex distribution mechanisms is complete and bounded-suboptimal. Our experiments show that our approaches outperform the original (greedy) flex distribution. 

---
# Towards Autonomous Sustainability Assessment via Multimodal AI Agents 

**Authors**: Zhihan Zhang, Alexander Metzger, Yuxuan Mei, Felix Hähnlein, Zachary Englhardt, Tingyu Cheng, Gregory D. Abowd, Shwetak Patel, Adriana Schulz, Vikram Iyer  

**Link**: [PDF](https://arxiv.org/pdf/2507.17012)  

**Abstract**: Interest in sustainability information has surged in recent years. However, the data required for a life cycle assessment (LCA) that maps the materials and processes from product manufacturing to disposal into environmental impacts (EI) are often unavailable. Here we reimagine conventional LCA by introducing multimodal AI agents that emulate interactions between LCA experts and stakeholders like product managers and engineers to calculate the cradle-to-gate (production) carbon emissions of electronic devices. The AI agents iteratively generate a detailed life-cycle inventory leveraging a custom data abstraction and software tools that extract information from online text and images from repair communities and government certifications. This approach reduces weeks or months of expert time to under one minute and closes data availability gaps while yielding carbon footprint estimates within 19% of expert LCAs with zero proprietary data. Additionally, we develop a method to directly estimate EI by comparing an input to a cluster of products with similar descriptions and known carbon footprints. This runs in 3 ms on a laptop with a MAPE of 12.28% on electronic products. Further, we develop a data-driven method to generate emission factors. We use the properties of an unknown material to represent it as a weighted sum of emission factors for similar materials. Compared to human experts picking the closest LCA database entry, this improves MAPE by 120.26%. We analyze the data and compute scaling of this approach and discuss its implications for future LCA workflows. 

---
# Large Learning Rates Simultaneously Achieve Robustness to Spurious Correlations and Compressibility 

**Authors**: Melih Barsbey, Lucas Prieto, Stefanos Zafeiriou, Tolga Birdal  

**Link**: [PDF](https://arxiv.org/pdf/2507.17748)  

**Abstract**: Robustness and resource-efficiency are two highly desirable properties for modern machine learning models. However, achieving them jointly remains a challenge. In this paper, we position high learning rates as a facilitator for simultaneously achieving robustness to spurious correlations and network compressibility. We demonstrate that large learning rates also produce desirable representation properties such as invariant feature utilization, class separation, and activation sparsity. Importantly, our findings indicate that large learning rates compare favorably to other hyperparameters and regularization methods, in consistently satisfying these properties in tandem. In addition to demonstrating the positive effect of large learning rates across diverse spurious correlation datasets, models, and optimizers, we also present strong evidence that the previously documented success of large learning rates in standard classification tasks is likely due to its effect on addressing hidden/rare spurious correlations in the training dataset. 

---
# Pretraining on the Test Set Is No Longer All You Need: A Debate-Driven Approach to QA Benchmarks 

**Authors**: Linbo Cao, Jinman Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2507.17747)  

**Abstract**: As frontier language models increasingly saturate standard QA benchmarks, concerns about data contamination, memorization, and escalating dataset creation costs persist. We propose a debate-driven evaluation paradigm that transforms any existing QA dataset into structured adversarial debates--where one model is given the official answer to defend, and another constructs and defends an alternative answer--adjudicated by a judge model blind to the correct solution. By forcing multi-round argumentation, this approach substantially increases difficulty while penalizing shallow memorization, yet reuses QA items to reduce curation overhead. We make two main contributions: (1) an evaluation pipeline to systematically convert QA tasks into debate-based assessments, and (2) a public benchmark that demonstrates our paradigm's effectiveness on a subset of MMLU-Pro questions, complete with standardized protocols and reference models. Empirical results validate the robustness of the method and its effectiveness against data contamination--a Llama 3.1 model fine-tuned on test questions showed dramatic accuracy improvements (50% -> 82%) but performed worse in debates. Results also show that even weaker judges can reliably differentiate stronger debaters, highlighting how debate-based evaluation can scale to future, more capable systems while maintaining a fraction of the cost of creating new benchmarks. Overall, our framework underscores that "pretraining on the test set is no longer all you need," offering a sustainable path for measuring the genuine reasoning ability of advanced language models. 

---
# Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains 

**Authors**: Anisha Gunjal, Anthony Wang, Elaine Lau, Vaskar Nath, Bing Liu, Sean Hendryx  

**Link**: [PDF](https://arxiv.org/pdf/2507.17746)  

**Abstract**: Extending Reinforcement Learning with Verifiable Rewards (RLVR) to real-world tasks often requires balancing objective and subjective evaluation criteria. However, many such tasks lack a single, unambiguous ground truth-making it difficult to define reliable reward signals for post-training language models. While traditional preference-based methods offer a workaround, they rely on opaque reward functions that are difficult to interpret and prone to spurious correlations. We introduce $\textbf{Rubrics as Rewards}$ (RaR), a framework that uses structured, checklist-style rubrics as interpretable reward signals for on-policy training with GRPO. Our best RaR method yields up to a $28\%$ relative improvement on HealthBench-1k compared to simple Likert-based approaches, while matching or surpassing the performance of reward signals derived from expert-written references. By treating rubrics as structured reward signals, we show that RaR enables smaller-scale judge models to better align with human preferences and sustain robust performance across model scales. 

---
# Ultra3D: Efficient and High-Fidelity 3D Generation with Part Attention 

**Authors**: Yiwen Chen, Zhihao Li, Yikai Wang, Hu Zhang, Qin Li, Chi Zhang, Guosheng Lin  

**Link**: [PDF](https://arxiv.org/pdf/2507.17745)  

**Abstract**: Recent advances in sparse voxel representations have significantly improved the quality of 3D content generation, enabling high-resolution modeling with fine-grained geometry. However, existing frameworks suffer from severe computational inefficiencies due to the quadratic complexity of attention mechanisms in their two-stage diffusion pipelines. In this work, we propose Ultra3D, an efficient 3D generation framework that significantly accelerates sparse voxel modeling without compromising quality. Our method leverages the compact VecSet representation to efficiently generate a coarse object layout in the first stage, reducing token count and accelerating voxel coordinate prediction. To refine per-voxel latent features in the second stage, we introduce Part Attention, a geometry-aware localized attention mechanism that restricts attention computation within semantically consistent part regions. This design preserves structural continuity while avoiding unnecessary global attention, achieving up to 6.7x speed-up in latent generation. To support this mechanism, we construct a scalable part annotation pipeline that converts raw meshes into part-labeled sparse voxels. Extensive experiments demonstrate that Ultra3D supports high-resolution 3D generation at 1024 resolution and achieves state-of-the-art performance in both visual fidelity and user preference. 

---
# Yume: An Interactive World Generation Model 

**Authors**: Xiaofeng Mao, Shaoheng Lin, Zhen Li, Chuanhao Li, Wenshuo Peng, Tong He, Jiangmiao Pang, Mingmin Chi, Yu Qiao, Kaipeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.17744)  

**Abstract**: Yume aims to use images, text, or videos to create an interactive, realistic, and dynamic world, which allows exploration and control using peripheral devices or neural signals. In this report, we present a preview version of \method, which creates a dynamic world from an input image and allows exploration of the world using keyboard actions. To achieve this high-fidelity and interactive video world generation, we introduce a well-designed framework, which consists of four main components, including camera motion quantization, video generation architecture, advanced sampler, and model acceleration. First, we quantize camera motions for stable training and user-friendly interaction using keyboard inputs. Then, we introduce the Masked Video Diffusion Transformer~(MVDT) with a memory module for infinite video generation in an autoregressive manner. After that, training-free Anti-Artifact Mechanism (AAM) and Time Travel Sampling based on Stochastic Differential Equations (TTS-SDE) are introduced to the sampler for better visual quality and more precise control. Moreover, we investigate model acceleration by synergistic optimization of adversarial distillation and caching mechanisms. We use the high-quality world exploration dataset \sekai to train \method, and it achieves remarkable results in diverse scenes and applications. All data, codebase, and model weights are available on this https URL. Yume will update monthly to achieve its original goal. Project page: this https URL. 

---
# Flow Matching Meets Biology and Life Science: A Survey 

**Authors**: Zihao Li, Zhichen Zeng, Xiao Lin, Feihao Fang, Yanru Qu, Zhe Xu, Zhining Liu, Xuying Ning, Tianxin Wei, Ge Liu, Hanghang Tong, Jingrui He  

**Link**: [PDF](https://arxiv.org/pdf/2507.17731)  

**Abstract**: Over the past decade, advances in generative modeling, such as generative adversarial networks, masked autoencoders, and diffusion models, have significantly transformed biological research and discovery, enabling breakthroughs in molecule design, protein generation, drug discovery, and beyond. At the same time, biological applications have served as valuable testbeds for evaluating the capabilities of generative models. Recently, flow matching has emerged as a powerful and efficient alternative to diffusion-based generative modeling, with growing interest in its application to problems in biology and life sciences. This paper presents the first comprehensive survey of recent developments in flow matching and its applications in biological domains. We begin by systematically reviewing the foundations and variants of flow matching, and then categorize its applications into three major areas: biological sequence modeling, molecule generation and design, and peptide and protein generation. For each, we provide an in-depth review of recent progress. We also summarize commonly used datasets and software tools, and conclude with a discussion of potential future directions. The corresponding curated resources are available at this https URL. 

---
# On the Interaction of Compressibility and Adversarial Robustness 

**Authors**: Melih Barsbey, Antônio H. Ribeiro, Umut Şimşekli, Tolga Birdal  

**Link**: [PDF](https://arxiv.org/pdf/2507.17725)  

**Abstract**: Modern neural networks are expected to simultaneously satisfy a host of desirable properties: accurate fitting to training data, generalization to unseen inputs, parameter and computational efficiency, and robustness to adversarial perturbations. While compressibility and robustness have each been studied extensively, a unified understanding of their interaction still remains elusive. In this work, we develop a principled framework to analyze how different forms of compressibility - such as neuron-level sparsity and spectral compressibility - affect adversarial robustness. We show that these forms of compression can induce a small number of highly sensitive directions in the representation space, which adversaries can exploit to construct effective perturbations. Our analysis yields a simple yet instructive robustness bound, revealing how neuron and spectral compressibility impact $L_\infty$ and $L_2$ robustness via their effects on the learned representations. Crucially, the vulnerabilities we identify arise irrespective of how compression is achieved - whether via regularization, architectural bias, or implicit learning dynamics. Through empirical evaluations across synthetic and realistic tasks, we confirm our theoretical predictions, and further demonstrate that these vulnerabilities persist under adversarial training and transfer learning, and contribute to the emergence of universal adversarial perturbations. Our findings show a fundamental tension between structured compressibility and robustness, and suggest new pathways for designing models that are both efficient and secure. 

---
# AI Telephone Surveying: Automating Quantitative Data Collection with an AI Interviewer 

**Authors**: Danny D. Leybzon, Shreyas Tirumala, Nishant Jain, Summer Gillen, Michael Jackson, Cameron McPhee, Jennifer Schmidt  

**Link**: [PDF](https://arxiv.org/pdf/2507.17718)  

**Abstract**: With the rise of voice-enabled artificial intelligence (AI) systems, quantitative survey researchers have access to a new data-collection mode: AI telephone surveying. By using AI to conduct phone interviews, researchers can scale quantitative studies while balancing the dual goals of human-like interactivity and methodological rigor. Unlike earlier efforts that used interactive voice response (IVR) technology to automate these surveys, voice AI enables a more natural and adaptive respondent experience as it is more robust to interruptions, corrections, and other idiosyncrasies of human speech.
We built and tested an AI system to conduct quantitative surveys based on large language models (LLM), automatic speech recognition (ASR), and speech synthesis technologies. The system was specifically designed for quantitative research, and strictly adhered to research best practices like question order randomization, answer order randomization, and exact wording.
To validate the system's effectiveness, we deployed it to conduct two pilot surveys with the SSRS Opinion Panel and followed-up with a separate human-administered survey to assess respondent experiences. We measured three key metrics: the survey completion rates, break-off rates, and respondent satisfaction scores. Our results suggest that shorter instruments and more responsive AI interviewers may contribute to improvements across all three metrics studied. 

---
# From Feedback to Checklists: Grounded Evaluation of AI-Generated Clinical Notes 

**Authors**: Karen Zhou, John Giorgi, Pranav Mani, Peng Xu, Davis Liang, Chenhao Tan  

**Link**: [PDF](https://arxiv.org/pdf/2507.17717)  

**Abstract**: AI-generated clinical notes are increasingly used in healthcare, but evaluating their quality remains a challenge due to high subjectivity and limited scalability of expert review. Existing automated metrics often fail to align with real-world physician preferences. To address this, we propose a pipeline that systematically distills real user feedback into structured checklists for note evaluation. These checklists are designed to be interpretable, grounded in human feedback, and enforceable by LLM-based evaluators. Using deidentified data from over 21,000 clinical encounters, prepared in accordance with the HIPAA safe harbor standard, from a deployed AI medical scribe system, we show that our feedback-derived checklist outperforms baseline approaches in our offline evaluations in coverage, diversity, and predictive power for human ratings. Extensive experiments confirm the checklist's robustness to quality-degrading perturbations, significant alignment with clinician preferences, and practical value as an evaluation methodology. In offline research settings, the checklist can help identify notes likely to fall below our chosen quality thresholds. 

---
# CASCADE: LLM-Powered JavaScript Deobfuscator at Google 

**Authors**: Shan Jiang, Pranoy Kovuri, David Tao, Zhixun Tan  

**Link**: [PDF](https://arxiv.org/pdf/2507.17691)  

**Abstract**: Software obfuscation, particularly prevalent in JavaScript, hinders code comprehension and analysis, posing significant challenges to software testing, static analysis, and malware detection. This paper introduces CASCADE, a novel hybrid approach that integrates the advanced coding capabilities of Gemini with the deterministic transformation capabilities of a compiler Intermediate Representation (IR), specifically JavaScript IR (JSIR). By employing Gemini to identify critical prelude functions, the foundational components underlying the most prevalent obfuscation techniques, and leveraging JSIR for subsequent code transformations, CASCADE effectively recovers semantic elements like original strings and API names, and reveals original program behaviors. This method overcomes limitations of existing static and dynamic deobfuscation techniques, eliminating hundreds to thousands of hardcoded rules while achieving reliability and flexibility. CASCADE is already deployed in Google's production environment, demonstrating substantial improvements in JavaScript deobfuscation efficiency and reducing reverse engineering efforts. 

---
# How Should We Meta-Learn Reinforcement Learning Algorithms? 

**Authors**: Alexander David Goldie, Zilin Wang, Jakob Nicolaus Foerster, Shimon Whiteson  

**Link**: [PDF](https://arxiv.org/pdf/2507.17668)  

**Abstract**: The process of meta-learning algorithms from data, instead of relying on manual design, is growing in popularity as a paradigm for improving the performance of machine learning systems. Meta-learning shows particular promise for reinforcement learning (RL), where algorithms are often adapted from supervised or unsupervised learning despite their suboptimality for RL. However, until now there has been a severe lack of comparison between different meta-learning algorithms, such as using evolution to optimise over black-box functions or LLMs to propose code. In this paper, we carry out this empirical comparison of the different approaches when applied to a range of meta-learned algorithms which target different parts of the RL pipeline. In addition to meta-train and meta-test performance, we also investigate factors including the interpretability, sample cost and train time for each meta-learning algorithm. Based on these findings, we propose several guidelines for meta-learning new RL algorithms which will help ensure that future learned algorithms are as performant as possible. 

---
# Vision Transformer attention alignment with human visual perception in aesthetic object evaluation 

**Authors**: Miguel Carrasco, César González-Martín, José Aranda, Luis Oliveros  

**Link**: [PDF](https://arxiv.org/pdf/2507.17616)  

**Abstract**: Visual attention mechanisms play a crucial role in human perception and aesthetic evaluation. Recent advances in Vision Transformers (ViTs) have demonstrated remarkable capabilities in computer vision tasks, yet their alignment with human visual attention patterns remains underexplored, particularly in aesthetic contexts. This study investigates the correlation between human visual attention and ViT attention mechanisms when evaluating handcrafted objects. We conducted an eye-tracking experiment with 30 participants (9 female, 21 male, mean age 24.6 years) who viewed 20 artisanal objects comprising basketry bags and ginger jars. Using a Pupil Labs eye-tracker, we recorded gaze patterns and generated heat maps representing human visual attention. Simultaneously, we analyzed the same objects using a pre-trained ViT model with DINO (Self-DIstillation with NO Labels), extracting attention maps from each of the 12 attention heads. We compared human and ViT attention distributions using Kullback-Leibler divergence across varying Gaussian parameters (sigma=0.1 to 3.0). Statistical analysis revealed optimal correlation at sigma=2.4 +-0.03, with attention head #12 showing the strongest alignment with human visual patterns. Significant differences were found between attention heads, with heads #7 and #9 demonstrating the greatest divergence from human attention (p< 0.05, Tukey HSD test). Results indicate that while ViTs exhibit more global attention patterns compared to human focal attention, certain attention heads can approximate human visual behavior, particularly for specific object features like buckles in basketry items. These findings suggest potential applications of ViT attention mechanisms in product design and aesthetic evaluation, while highlighting fundamental differences in attention strategies between human perception and current AI models. 

---
# PRIX: Learning to Plan from Raw Pixels for End-to-End Autonomous Driving 

**Authors**: Maciej K. Wozniak, Lianhang Liu, Yixi Cai, Patric Jensfelt  

**Link**: [PDF](https://arxiv.org/pdf/2507.17596)  

**Abstract**: While end-to-end autonomous driving models show promising results, their practical deployment is often hindered by large model sizes, a reliance on expensive LiDAR sensors and computationally intensive BEV feature representations. This limits their scalability, especially for mass-market vehicles equipped only with cameras. To address these challenges, we propose PRIX (Plan from Raw Pixels). Our novel and efficient end-to-end driving architecture operates using only camera data, without explicit BEV representation and forgoing the need for LiDAR. PRIX leverages a visual feature extractor coupled with a generative planning head to predict safe trajectories from raw pixel inputs directly. A core component of our architecture is the Context-aware Recalibration Transformer (CaRT), a novel module designed to effectively enhance multi-level visual features for more robust planning. We demonstrate through comprehensive experiments that PRIX achieves state-of-the-art performance on the NavSim and nuScenes benchmarks, matching the capabilities of larger, multimodal diffusion planners while being significantly more efficient in terms of inference speed and model size, making it a practical solution for real-world deployment. Our work is open-source and the code will be at this https URL. 

---
# Enhancing Quantum Federated Learning with Fisher Information-Based Optimization 

**Authors**: Amandeep Singh Bhatia, Sabre Kais  

**Link**: [PDF](https://arxiv.org/pdf/2507.17580)  

**Abstract**: Federated Learning (FL) has become increasingly popular across different sectors, offering a way for clients to work together to train a global model without sharing sensitive data. It involves multiple rounds of communication between the global model and participating clients, which introduces several challenges like high communication costs, heterogeneous client data, prolonged processing times, and increased vulnerability to privacy threats. In recent years, the convergence of federated learning and parameterized quantum circuits has sparked significant research interest, with promising implications for fields such as healthcare and finance. By enabling decentralized training of quantum models, it allows clients or institutions to collaboratively enhance model performance and outcomes while preserving data privacy. Recognizing that Fisher information can quantify the amount of information that a quantum state carries under parameter changes, thereby providing insight into its geometric and statistical properties. We intend to leverage this property to address the aforementioned challenges. In this work, we propose a Quantum Federated Learning (QFL) algorithm that makes use of the Fisher information computed on local client models, with data distributed across heterogeneous partitions. This approach identifies the critical parameters that significantly influence the quantum model's performance, ensuring they are preserved during the aggregation process. Our research assessed the effectiveness and feasibility of QFL by comparing its performance against other variants, and exploring the benefits of incorporating Fisher information in QFL settings. Experimental results on ADNI and MNIST datasets demonstrate the effectiveness of our approach in achieving better performance and robustness against the quantum federated averaging method. 

---
# Federated Majorize-Minimization: Beyond Parameter Aggregation 

**Authors**: Aymeric Dieuleveut, Gersende Fort, Mahmoud Hegazy, Hoi-To Wai  

**Link**: [PDF](https://arxiv.org/pdf/2507.17534)  

**Abstract**: This paper proposes a unified approach for designing stochastic optimization algorithms that robustly scale to the federated learning setting. Our work studies a class of Majorize-Minimization (MM) problems, which possesses a linearly parameterized family of majorizing surrogate functions. This framework encompasses (proximal) gradient-based algorithms for (regularized) smooth objectives, the Expectation Maximization algorithm, and many problems seen as variational surrogate MM. We show that our framework motivates a unifying algorithm called Stochastic Approximation Stochastic Surrogate MM (\SSMM), which includes previous stochastic MM procedures as special instances. We then extend \SSMM\ to the federated setting, while taking into consideration common bottlenecks such as data heterogeneity, partial participation, and communication constraints; this yields \QSMM. The originality of \QSMM\ is to learn locally and then aggregate information characterizing the \textit{surrogate majorizing function}, contrary to classical algorithms which learn and aggregate the \textit{original parameter}. Finally, to showcase the flexibility of this methodology beyond our theoretical setting, we use it to design an algorithm for computing optimal transport maps in the federated setting. 

---
# Integrating Physics-Based and Data-Driven Approaches for Probabilistic Building Energy Modeling 

**Authors**: Leandro Von Krannichfeldt, Kristina Orehounig, Olga Fink  

**Link**: [PDF](https://arxiv.org/pdf/2507.17526)  

**Abstract**: Building energy modeling is a key tool for optimizing the performance of building energy systems. Historically, a wide spectrum of methods has been explored -- ranging from conventional physics-based models to purely data-driven techniques. Recently, hybrid approaches that combine the strengths of both paradigms have gained attention. These include strategies such as learning surrogates for physics-based models, modeling residuals between simulated and observed data, fine-tuning surrogates with real-world measurements, using physics-based outputs as additional inputs for data-driven models, and integrating the physics-based output into the loss function the data-driven model. Despite this progress, two significant research gaps remain. First, most hybrid methods focus on deterministic modeling, often neglecting the inherent uncertainties caused by factors like weather fluctuations and occupant behavior. Second, there has been little systematic comparison within a probabilistic modeling framework. This study addresses these gaps by evaluating five representative hybrid approaches for probabilistic building energy modeling, focusing on quantile predictions of building thermodynamics in a real-world case study. Our results highlight two main findings. First, the performance of hybrid approaches varies across different building room types, but residual learning with a Feedforward Neural Network performs best on average. Notably, the residual approach is the only model that produces physically intuitive predictions when applied to out-of-distribution test data. Second, Quantile Conformal Prediction is an effective procedure for calibrating quantile predictions in case of indoor temperature modeling. 

---
# Enabling Cyber Security Education through Digital Twins and Generative AI 

**Authors**: Vita Santa Barletta, Vito Bavaro, Miriana Calvano, Antonio Curci, Antonio Piccinno, Davide Pio Posa  

**Link**: [PDF](https://arxiv.org/pdf/2507.17518)  

**Abstract**: Digital Twins (DTs) are gaining prominence in cybersecurity for their ability to replicate complex IT (Information Technology), OT (Operational Technology), and IoT (Internet of Things) infrastructures, allowing for real time monitoring, threat analysis, and system simulation. This study investigates how integrating DTs with penetration testing tools and Large Language Models (LLMs) can enhance cybersecurity education and operational readiness. By simulating realistic cyber environments, this approach offers a practical, interactive framework for exploring vulnerabilities and defensive strategies. At the core of this research is the Red Team Knife (RTK), a custom penetration testing toolkit aligned with the Cyber Kill Chain model. RTK is designed to guide learners through key phases of cyberattacks, including reconnaissance, exploitation, and response within a DT powered ecosystem. The incorporation of Large Language Models (LLMs) further enriches the experience by providing intelligent, real-time feedback, natural language threat explanations, and adaptive learning support during training exercises. This combined DT LLM framework is currently being piloted in academic settings to develop hands on skills in vulnerability assessment, threat detection, and security operations. Initial findings suggest that the integration significantly improves the effectiveness and relevance of cybersecurity training, bridging the gap between theoretical knowledge and real-world application. Ultimately, the research demonstrates how DTs and LLMs together can transform cybersecurity education to meet evolving industry demands. 

---
# HOTA: Hamiltonian framework for Optimal Transport Advection 

**Authors**: Nazar Buzun, Daniil Shlenskii, Maxim Bobrin, Dmitry V. Dylov  

**Link**: [PDF](https://arxiv.org/pdf/2507.17513)  

**Abstract**: Optimal transport (OT) has become a natural framework for guiding the probability flows. Yet, the majority of recent generative models assume trivial geometry (e.g., Euclidean) and rely on strong density-estimation assumptions, yielding trajectories that do not respect the true principles of optimality in the underlying manifold. We present Hamiltonian Optimal Transport Advection (HOTA), a Hamilton-Jacobi-Bellman based method that tackles the dual dynamical OT problem explicitly through Kantorovich potentials, enabling efficient and scalable trajectory optimization. Our approach effectively evades the need for explicit density modeling, performing even when the cost functionals are non-smooth. Empirically, HOTA outperforms all baselines in standard benchmarks, as well as in custom datasets with non-differentiable costs, both in terms of feasibility and optimality. 

---
# To Trust or Not to Trust: On Calibration in ML-based Resource Allocation for Wireless Networks 

**Authors**: Rashika Raina, Nidhi Simmons, David E. Simmons, Michel Daoud Yacoub, Trung Q. Duong  

**Link**: [PDF](https://arxiv.org/pdf/2507.17494)  

**Abstract**: In next-generation communications and networks, machine learning (ML) models are expected to deliver not only accurate predictions but also well-calibrated confidence scores that reflect the true likelihood of correct decisions. This paper studies the calibration performance of an ML-based outage predictor within a single-user, multi-resource allocation framework. We first establish key theoretical properties of this system's outage probability (OP) under perfect calibration. Importantly, we show that as the number of resources grows, the OP of a perfectly calibrated predictor approaches the expected output conditioned on it being below the classification threshold. In contrast, when only one resource is available, the system's OP equals the model's overall expected output. We then derive the OP conditions for a perfectly calibrated predictor. These findings guide the choice of the classification threshold to achieve a desired OP, helping system designers meet specific reliability requirements. We also demonstrate that post-processing calibration cannot improve the system's minimum achievable OP, as it does not introduce new information about future channel states. Additionally, we show that well-calibrated models are part of a broader class of predictors that necessarily improve OP. In particular, we establish a monotonicity condition that the accuracy-confidence function must satisfy for such improvement to occur. To demonstrate these theoretical properties, we conduct a rigorous simulation-based analysis using post-processing calibration techniques: Platt scaling and isotonic regression. As part of this framework, the predictor is trained using an outage loss function specifically designed for this system. Furthermore, this analysis is performed on Rayleigh fading channels with temporal correlation captured by Clarke's 2D model, which accounts for receiver mobility. 

---
# Unsupervised anomaly detection using Bayesian flow networks: application to brain FDG PET in the context of Alzheimer's disease 

**Authors**: Hugues Roy, Reuben Dorent, Ninon Burgos  

**Link**: [PDF](https://arxiv.org/pdf/2507.17486)  

**Abstract**: Unsupervised anomaly detection (UAD) plays a crucial role in neuroimaging for identifying deviations from healthy subject data and thus facilitating the diagnosis of neurological disorders. In this work, we focus on Bayesian flow networks (BFNs), a novel class of generative models, which have not yet been applied to medical imaging or anomaly detection. BFNs combine the strength of diffusion frameworks and Bayesian inference. We introduce AnoBFN, an extension of BFNs for UAD, designed to: i) perform conditional image generation under high levels of spatially correlated noise, and ii) preserve subject specificity by incorporating a recursive feedback from the input image throughout the generative process. We evaluate AnoBFN on the challenging task of Alzheimer's disease-related anomaly detection in FDG PET images. Our approach outperforms other state-of-the-art methods based on VAEs (beta-VAE), GANs (f-AnoGAN), and diffusion models (AnoDDPM), demonstrating its effectiveness at detecting anomalies while reducing false positive rates. 

---
# MultiNRC: A Challenging and Native Multilingual Reasoning Evaluation Benchmark for LLMs 

**Authors**: Alexander R. Fabbri, Diego Mares, Jorge Flores, Meher Mankikar, Ernesto Hernandez, Dean Lee, Bing Liu, Chen Xing  

**Link**: [PDF](https://arxiv.org/pdf/2507.17476)  

**Abstract**: Although recent Large Language Models (LLMs) have shown rapid improvement on reasoning benchmarks in English, the evaluation of such LLMs' multilingual reasoning capability across diverse languages and cultural contexts remains limited. Existing multilingual reasoning benchmarks are typically constructed by translating existing English reasoning benchmarks, biasing these benchmarks towards reasoning problems with context in English language/cultures. In this work, we introduce the Multilingual Native Reasoning Challenge (MultiNRC), a benchmark designed to assess LLMs on more than 1,000 native, linguistic and culturally grounded reasoning questions written by native speakers in French, Spanish, and Chinese. MultiNRC covers four core reasoning categories: language-specific linguistic reasoning, wordplay & riddles, cultural/tradition reasoning, and math reasoning with cultural relevance. For cultural/tradition reasoning and math reasoning with cultural relevance, we also provide English equivalent translations of the multilingual questions by manual translation from native speakers fluent in English. This set of English equivalents can provide a direct comparison of LLM reasoning capacity in other languages vs. English on the same reasoning questions. We systematically evaluate current 14 leading LLMs covering most LLM families on MultiNRC and its English equivalent set. The results show that (1) current LLMs are still not good at native multilingual reasoning, with none scoring above 50% on MultiNRC; (2) LLMs exhibit distinct strengths and weaknesses in handling linguistic, cultural, and logical reasoning tasks; (3) Most models perform substantially better in math reasoning in English compared to in original languages (+10%), indicating persistent challenges with culturally grounded knowledge. 

---
# BGM-HAN: A Hierarchical Attention Network for Accurate and Fair Decision Assessment on Semi-Structured Profiles 

**Authors**: Junhua Liu, Roy Ka-Wei Lee, Kwan Hui Lim  

**Link**: [PDF](https://arxiv.org/pdf/2507.17472)  

**Abstract**: Human decision-making in high-stakes domains often relies on expertise and heuristics, but is vulnerable to hard-to-detect cognitive biases that threaten fairness and long-term outcomes. This work presents a novel approach to enhancing complex decision-making workflows through the integration of hierarchical learning alongside various enhancements. Focusing on university admissions as a representative high-stakes domain, we propose BGM-HAN, an enhanced Byte-Pair Encoded, Gated Multi-head Hierarchical Attention Network, designed to effectively model semi-structured applicant data. BGM-HAN captures multi-level representations that are crucial for nuanced assessment, improving both interpretability and predictive performance. Experimental results on real admissions data demonstrate that our proposed model significantly outperforms both state-of-the-art baselines from traditional machine learning to large language models, offering a promising framework for augmenting decision-making in domains where structure, context, and fairness matter. Source code is available at: this https URL. 

---
# Demonstration of Efficient Predictive Surrogates for Large-scale Quantum Processors 

**Authors**: Wei-You Liao, Yuxuan Du, Xinbiao Wang, Tian-Ci Tian, Yong Luo, Bo Du, Dacheng Tao, He-Liang Huang  

**Link**: [PDF](https://arxiv.org/pdf/2507.17470)  

**Abstract**: The ongoing development of quantum processors is driving breakthroughs in scientific discovery. Despite this progress, the formidable cost of fabricating large-scale quantum processors means they will remain rare for the foreseeable future, limiting their widespread application. To address this bottleneck, we introduce the concept of predictive surrogates, which are classical learning models designed to emulate the mean-value behavior of a given quantum processor with provably computational efficiency. In particular, we propose two predictive surrogates that can substantially reduce the need for quantum processor access in diverse practical scenarios. To demonstrate their potential in advancing digital quantum simulation, we use these surrogates to emulate a quantum processor with up to 20 programmable superconducting qubits, enabling efficient pre-training of variational quantum eigensolvers for families of transverse-field Ising models and identification of non-equilibrium Floquet symmetry-protected topological phases. Experimental results reveal that the predictive surrogates not only reduce measurement overhead by orders of magnitude, but can also surpass the performance of conventional, quantum-resource-intensive approaches. Collectively, these findings establish predictive surrogates as a practical pathway to broadening the impact of advanced quantum processors. 

---
# Probing Vision-Language Understanding through the Visual Entailment Task: promises and pitfalls 

**Authors**: Elena Pitta, Tom Kouwenhoven, Tessa Verhoef  

**Link**: [PDF](https://arxiv.org/pdf/2507.17467)  

**Abstract**: This study investigates the extent to which the Visual Entailment (VE) task serves as a reliable probe of vision-language understanding in multimodal language models, using the LLaMA 3.2 11B Vision model as a test case. Beyond reporting performance metrics, we aim to interpret what these results reveal about the underlying possibilities and limitations of the VE task. We conduct a series of experiments across zero-shot, few-shot, and fine-tuning settings, exploring how factors such as prompt design, the number and order of in-context examples and access to visual information might affect VE performance. To further probe the reasoning processes of the model, we used explanation-based evaluations. Results indicate that three-shot inference outperforms the zero-shot baselines. However, additional examples introduce more noise than they provide benefits. Additionally, the order of the labels in the prompt is a critical factor that influences the predictions. In the absence of visual information, the model has a strong tendency to hallucinate and imagine content, raising questions about the model's over-reliance on linguistic priors. Fine-tuning yields strong results, achieving an accuracy of 83.3% on the e-SNLI-VE dataset and outperforming the state-of-the-art OFA-X model. Additionally, the explanation evaluation demonstrates that the fine-tuned model provides semantically meaningful explanations similar to those of humans, with a BERTScore F1-score of 89.2%. We do, however, find comparable BERTScore results in experiments with limited vision, questioning the visual grounding of this task. Overall, our results highlight both the utility and limitations of VE as a diagnostic task for vision-language understanding and point to directions for refining multimodal evaluation methods. 

---
# Reasoning-Driven Retrosynthesis Prediction with Large Language Models via Reinforcement Learning 

**Authors**: Situo Zhang, Hanqi Li, Lu Chen, Zihan Zhao, Xuanze Lin, Zichen Zhu, Bo Chen, Xin Chen, Kai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.17448)  

**Abstract**: Retrosynthesis planning, essential in organic synthesis and drug discovery, has greatly benefited from recent AI-driven advancements. Nevertheless, existing methods frequently face limitations in both applicability and explainability. Traditional graph-based and sequence-to-sequence models often lack generalized chemical knowledge, leading to predictions that are neither consistently accurate nor easily explainable. To address these challenges, we introduce RetroDFM-R, a reasoning-based large language model (LLM) designed specifically for chemical retrosynthesis. Leveraging large-scale reinforcement learning guided by chemically verifiable rewards, RetroDFM-R significantly enhances prediction accuracy and explainability. Comprehensive evaluations demonstrate that RetroDFM-R significantly outperforms state-of-the-art methods, achieving a top-1 accuracy of 65.0% on the USPTO-50K benchmark. Double-blind human assessments further validate the chemical plausibility and practical utility of RetroDFM-R's predictions. RetroDFM-R also accurately predicts multistep retrosynthetic routes reported in the literature for both real-world drug molecules and perovskite materials. Crucially, the model's explicit reasoning process provides human-interpretable insights, thereby enhancing trust and practical value in real-world retrosynthesis applications. 

---
# IndoorBEV: Joint Detection and Footprint Completion of Objects via Mask-based Prediction in Indoor Scenarios for Bird's-Eye View Perception 

**Authors**: Haichuan Li, Changda Tian, Panos Trahanias, Tomi Westerlund  

**Link**: [PDF](https://arxiv.org/pdf/2507.17445)  

**Abstract**: Detecting diverse objects within complex indoor 3D point clouds presents significant challenges for robotic perception, particularly with varied object shapes, clutter, and the co-existence of static and dynamic elements where traditional bounding box methods falter. To address these limitations, we propose IndoorBEV, a novel mask-based Bird's-Eye View (BEV) method for indoor mobile robots.
In a BEV method, a 3D scene is projected into a 2D BEV grid which handles naturally occlusions and provides a consistent top-down view aiding to distinguish static obstacles from dynamic agents. The obtained 2D BEV results is directly usable to downstream robotic tasks like navigation, motion prediction, and planning. Our architecture utilizes an axis compact encoder and a window-based backbone to extract rich spatial features from this BEV map. A query-based decoder head then employs learned object queries to concurrently predict object classes and instance masks in the BEV space. This mask-centric formulation effectively captures the footprint of both static and dynamic objects regardless of their shape, offering a robust alternative to bounding box regression. We demonstrate the effectiveness of IndoorBEV on a custom indoor dataset featuring diverse object classes including static objects
and dynamic elements like robots and miscellaneous items, showcasing its potential for robust indoor scene understanding. 

---
# Each to Their Own: Exploring the Optimal Embedding in RAG 

**Authors**: Shiting Chen, Zijian Zhao, Jinsong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.17442)  

**Abstract**: Recently, as Large Language Models (LLMs) have fundamentally impacted various fields, the methods for incorporating up-to-date information into LLMs or adding external knowledge to construct domain-specific models have garnered wide attention. Retrieval-Augmented Generation (RAG), serving as an inference-time scaling method, is notable for its low cost and minimal effort for parameter tuning. However, due to heterogeneous training data and model architecture, the variant embedding models used in RAG exhibit different benefits across various areas, often leading to different similarity calculation results and, consequently, varying response quality from LLMs. To address this problem, we propose and examine two approaches to enhance RAG by combining the benefits of multiple embedding models, named Mixture-Embedding RAG and Confident RAG. Mixture-Embedding RAG simply sorts and selects retrievals from multiple embedding models based on standardized similarity; however, it does not outperform vanilla RAG. In contrast, Confident RAG generates responses multiple times using different embedding models and then selects the responses with the highest confidence level, demonstrating average improvements of approximately 10% and 5% over vanilla LLMs and RAG, respectively. The consistent results across different LLMs and embedding models indicate that Confident RAG is an efficient plug-and-play approach for various domains. We will release our code upon publication. 

---
# Fair Compromises in Participatory Budgeting: a Multi-Agent Deep Reinforcement Learning Approach 

**Authors**: Hugh Adams, Srijoni Majumdar, Evangelos Pournaras  

**Link**: [PDF](https://arxiv.org/pdf/2507.17433)  

**Abstract**: Participatory budgeting is a method of collectively understanding and addressing spending priorities where citizens vote on how a budget is spent, it is regularly run to improve the fairness of the distribution of public funds. Participatory budgeting requires voters to make decisions on projects which can lead to ``choice overload". A multi-agent reinforcement learning approach to decision support can make decision making easier for voters by identifying voting strategies that increase the winning proportion of their vote. This novel approach can also support policymakers by highlighting aspects of election design that enable fair compromise on projects. This paper presents a novel, ethically aligned approach to decision support using multi-agent deep reinforcement learning modelling. This paper introduces a novel use of a branching neural network architecture to overcome scalability challenges of multi-agent reinforcement learning in a decentralized way. Fair compromises are found through optimising voter actions towards greater representation of voter preferences in the winning set. Experimental evaluation with real-world participatory budgeting data reveals a pattern in fair compromise: that it is achievable through projects with smaller cost. 

---
# Content-based 3D Image Retrieval and a ColBERT-inspired Re-ranking for Tumor Flagging and Staging 

**Authors**: Farnaz Khun Jush, Steffen Vogler, Matthias Lenga  

**Link**: [PDF](https://arxiv.org/pdf/2507.17412)  

**Abstract**: The increasing volume of medical images poses challenges for radiologists in retrieving relevant cases. Content-based image retrieval (CBIR) systems offer potential for efficient access to similar cases, yet lack standardized evaluation and comprehensive studies. Building on prior studies for tumor characterization via CBIR, this study advances CBIR research for volumetric medical images through three key contributions: (1) a framework eliminating reliance on pre-segmented data and organ-specific datasets, aligning with large and unstructured image archiving systems, i.e. PACS in clinical practice; (2) introduction of C-MIR, a novel volumetric re-ranking method adapting ColBERT's contextualized late interaction mechanism for 3D medical imaging; (3) comprehensive evaluation across four tumor sites using three feature extractors and three database configurations. Our evaluations highlight the significant advantages of C-MIR. We demonstrate the successful adaptation of the late interaction principle to volumetric medical images, enabling effective context-aware re-ranking. A key finding is C-MIR's ability to effectively localize the region of interest, eliminating the need for pre-segmentation of datasets and offering a computationally efficient alternative to systems relying on expensive data enrichment steps. C-MIR demonstrates promising improvements in tumor flagging, achieving improved performance, particularly for colon and lung tumors (p<0.05). C-MIR also shows potential for improving tumor staging, warranting further exploration of its capabilities. Ultimately, our work seeks to bridge the gap between advanced retrieval techniques and their practical applications in healthcare, paving the way for improved diagnostic processes. 

---
# Millions of $\text{GeAR}$-s: Extending GraphRAG to Millions of Documents 

**Authors**: Zhili Shen, Chenxin Diao, Pascual Merita, Pavlos Vougiouklis, Jeff Z. Pan  

**Link**: [PDF](https://arxiv.org/pdf/2507.17399)  

**Abstract**: Recent studies have explored graph-based approaches to retrieval-augmented generation, leveraging structured or semi-structured information -- such as entities and their relations extracted from documents -- to enhance retrieval. However, these methods are typically designed to address specific tasks, such as multi-hop question answering and query-focused summarisation, and therefore, there is limited evidence of their general applicability across broader datasets. In this paper, we aim to adapt a state-of-the-art graph-based RAG solution: $\text{GeAR}$ and explore its performance and limitations on the SIGIR 2025 LiveRAG Challenge. 

---
# HiProbe-VAD: Video Anomaly Detection via Hidden States Probing in Tuning-Free Multimodal LLMs 

**Authors**: Zhaolin Cai, Fan Li, Ziwei Zheng, Yanjun Qin  

**Link**: [PDF](https://arxiv.org/pdf/2507.17394)  

**Abstract**: Video Anomaly Detection (VAD) aims to identify and locate deviations from normal patterns in video sequences. Traditional methods often struggle with substantial computational demands and a reliance on extensive labeled datasets, thereby restricting their practical applicability. To address these constraints, we propose HiProbe-VAD, a novel framework that leverages pre-trained Multimodal Large Language Models (MLLMs) for VAD without requiring fine-tuning. In this paper, we discover that the intermediate hidden states of MLLMs contain information-rich representations, exhibiting higher sensitivity and linear separability for anomalies compared to the output layer. To capitalize on this, we propose a Dynamic Layer Saliency Probing (DLSP) mechanism that intelligently identifies and extracts the most informative hidden states from the optimal intermediate layer during the MLLMs reasoning. Then a lightweight anomaly scorer and temporal localization module efficiently detects anomalies using these extracted hidden states and finally generate explanations. Experiments on the UCF-Crime and XD-Violence datasets demonstrate that HiProbe-VAD outperforms existing training-free and most traditional approaches. Furthermore, our framework exhibits remarkable cross-model generalization capabilities in different MLLMs without any tuning, unlocking the potential of pre-trained MLLMs for video anomaly detection and paving the way for more practical and scalable solutions. 

---
# Investigating Training Data Detection in AI Coders 

**Authors**: Tianlin Li, Yunxiang Wei, Zhiming Li, Aishan Liu, Qing Guo, Xianglong Liu, Dongning Sun, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.17389)  

**Abstract**: Recent advances in code large language models (CodeLLMs) have made them indispensable tools in modern software engineering. However, these models occasionally produce outputs that contain proprietary or sensitive code snippets, raising concerns about potential non-compliant use of training data, and posing risks to privacy and intellectual property. To ensure responsible and compliant deployment of CodeLLMs, training data detection (TDD) has become a critical task. While recent TDD methods have shown promise in natural language settings, their effectiveness on code data remains largely underexplored. This gap is particularly important given code's structured syntax and distinct similarity criteria compared to natural language. To address this, we conduct a comprehensive empirical study of seven state-of-the-art TDD methods on source code data, evaluating their performance across eight CodeLLMs. To support this evaluation, we introduce CodeSnitch, a function-level benchmark dataset comprising 9,000 code samples in three programming languages, each explicitly labeled as either included or excluded from CodeLLM training. Beyond evaluation on the original CodeSnitch, we design targeted mutation strategies to test the robustness of TDD methods under three distinct settings. These mutation strategies are grounded in the well-established Type-1 to Type-4 code clone detection taxonomy. Our study provides a systematic assessment of current TDD techniques for code and offers insights to guide the development of more effective and robust detection methods in the future. 

---
# SFUOD: Source-Free Unknown Object Detection 

**Authors**: Keon-Hee Park, Seun-An Choe, Gyeong-Moon Park  

**Link**: [PDF](https://arxiv.org/pdf/2507.17373)  

**Abstract**: Source-free object detection adapts a detector pre-trained on a source domain to an unlabeled target domain without requiring access to labeled source data. While this setting is practical as it eliminates the need for the source dataset during domain adaptation, it operates under the restrictive assumption that only pre-defined objects from the source domain exist in the target domain. This closed-set setting prevents the detector from detecting undefined objects. To ease this assumption, we propose Source-Free Unknown Object Detection (SFUOD), a novel scenario which enables the detector to not only recognize known objects but also detect undefined objects as unknown objects. To this end, we propose CollaPAUL (Collaborative tuning and Principal Axis-based Unknown Labeling), a novel framework for SFUOD. Collaborative tuning enhances knowledge adaptation by integrating target-dependent knowledge from the auxiliary encoder with source-dependent knowledge from the pre-trained detector through a cross-domain attention mechanism. Additionally, principal axes-based unknown labeling assigns pseudo-labels to unknown objects by estimating objectness via principal axes projection and confidence scores from model predictions. The proposed CollaPAUL achieves state-of-the-art performances on SFUOD benchmarks, and extensive experiments validate its effectiveness. 

---
# DynaSearcher: Dynamic Knowledge Graph Augmented Search Agent via Multi-Reward Reinforcement Learning 

**Authors**: Chuzhan Hao, Wenfeng Feng, Yuewei Zhang, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.17365)  

**Abstract**: Multi-step agentic retrieval systems based on large language models (LLMs) have demonstrated remarkable performance in complex information search tasks. However, these systems still face significant challenges in practical applications, particularly in generating factually inconsistent intermediate queries and inefficient search trajectories, which can lead to reasoning deviations or redundant computations. To address these issues, we propose DynaSearcher, an innovative search agent enhanced by dynamic knowledge graphs and multi-reward reinforcement learning (RL). Specifically, our system leverages knowledge graphs as external structured knowledge to guide the search process by explicitly modeling entity relationships, thereby ensuring factual consistency in intermediate queries and mitigating biases from irrelevant information. Furthermore, we employ a multi-reward RL framework for fine-grained control over training objectives such as retrieval accuracy, efficiency, and response quality. This framework promotes the generation of high-quality intermediate queries and comprehensive final answers, while discouraging unnecessary exploration and minimizing information omissions or redundancy. Experimental results demonstrate that our approach achieves state-of-the-art answer accuracy on six multi-hop question answering datasets, matching frontier LLMs while using only small-scale models and limited computational resources. Furthermore, our approach demonstrates strong generalization and robustness across diverse retrieval environments and larger-scale models, highlighting its broad applicability. 

---
# Swin-TUNA : A Novel PEFT Approach for Accurate Food Image Segmentation 

**Authors**: Haotian Chen, Zhiyong Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2507.17347)  

**Abstract**: In the field of food image processing, efficient semantic segmentation techniques are crucial for industrial applications. However, existing large-scale Transformer-based models (such as FoodSAM) face challenges in meeting practical deploymentrequirements due to their massive parameter counts and high computational resource demands. This paper introduces TUNable Adapter module (Swin-TUNA), a Parameter Efficient Fine-Tuning (PEFT) method that integrates multiscale trainable adapters into the Swin Transformer architecture, achieving high-performance food image segmentation by updating only 4% of the parameters. The core innovation of Swin-TUNA lies in its hierarchical feature adaptation mechanism: it designs separable convolutions in depth and dimensional mappings of varying scales to address the differences in features between shallow and deep networks, combined with a dynamic balancing strategy for tasks-agnostic and task-specific features. Experiments demonstrate that this method achieves mIoU of 50.56% and 74.94% on the FoodSeg103 and UECFoodPix Complete datasets, respectively, surpassing the fully parameterized FoodSAM model while reducing the parameter count by 98.7% (to only 8.13M). Furthermore, Swin-TUNA exhibits faster convergence and stronger generalization capabilities in low-data scenarios, providing an efficient solution for assembling lightweight food image. 

---
# Temporal Point-Supervised Signal Reconstruction: A Human-Annotation-Free Framework for Weak Moving Target Detection 

**Authors**: Weihua Gao, Chunxu Ren, Wenlong Niu, Xiaodong Peng  

**Link**: [PDF](https://arxiv.org/pdf/2507.17334)  

**Abstract**: In low-altitude surveillance and early warning systems, detecting weak moving targets remains a significant challenge due to low signal energy, small spatial extent, and complex background clutter. Existing methods struggle with extracting robust features and suffer from the lack of reliable annotations. To address these limitations, we propose a novel Temporal Point-Supervised (TPS) framework that enables high-performance detection of weak targets without any manual this http URL of conventional frame-based detection, our framework reformulates the task as a pixel-wise temporal signal modeling problem, where weak targets manifest as short-duration pulse-like responses. A Temporal Signal Reconstruction Network (TSRNet) is developed under the TPS paradigm to reconstruct these transient this http URL adopts an encoder-decoder architecture and integrates a Dynamic Multi-Scale Attention (DMSAttention) module to enhance its sensitivity to diverse temporal patterns. Additionally, a graph-based trajectory mining strategy is employed to suppress false alarms and ensure temporal this http URL experiments on a purpose-built low-SNR dataset demonstrate that our framework outperforms state-of-the-art methods while requiring no human annotations. It achieves strong detection performance and operates at over 1000 FPS, underscoring its potential for real-time deployment in practical scenarios. 

---
# EarthLink: Interpreting Climate Signals with Self-Evolving AI Agents 

**Authors**: Zijie Guo, Jiong Wang, Xiaoyu Yue, Wangxu Wei, Zhe Jiang, Wanghan Xu, Ben Fei, Wenlong Zhang, Xinyu Gu, Lijing Cheng, Jing-Jia Luo, Chao Li, Yaqiang Wang, Tao Chen, Wanli Ouyang, Fenghua Ling, Lei Bai  

**Link**: [PDF](https://arxiv.org/pdf/2507.17311)  

**Abstract**: Modern Earth science is at an inflection point. The vast, fragmented, and complex nature of Earth system data, coupled with increasingly sophisticated analytical demands, creates a significant bottleneck for rapid scientific discovery. Here we introduce EarthLink, the first AI agent designed as an interactive copilot for Earth scientists. It automates the end-to-end research workflow, from planning and code generation to multi-scenario analysis. Unlike static diagnostic tools, EarthLink can learn from user interaction, continuously refining its capabilities through a dynamic feedback loop. We validated its performance on a number of core scientific tasks of climate change, ranging from model-observation comparisons to the diagnosis of complex phenomena. In a multi-expert evaluation, EarthLink produced scientifically sound analyses and demonstrated an analytical competency that was rated as comparable to specific aspects of a human junior researcher's workflow. Additionally, its transparent, auditable workflows and natural language interface empower scientists to shift from laborious manual execution to strategic oversight and hypothesis generation. EarthLink marks a pivotal step towards an efficient, trustworthy, and collaborative paradigm for Earth system research in an era of accelerating global change. 

---
# Confounded Causal Imitation Learning with Instrumental Variables 

**Authors**: Yan Zeng, Shenglan Nie, Feng Xie, Libo Huang, Peng Wu, Zhi Geng  

**Link**: [PDF](https://arxiv.org/pdf/2507.17309)  

**Abstract**: Imitation learning from demonstrations usually suffers from the confounding effects of unmeasured variables (i.e., unmeasured confounders) on the states and actions. If ignoring them, a biased estimation of the policy would be entailed. To break up this confounding gap, in this paper, we take the best of the strong power of instrumental variables (IV) and propose a Confounded Causal Imitation Learning (C2L) model. This model accommodates confounders that influence actions across multiple timesteps, rather than being restricted to immediate temporal dependencies. We develop a two-stage imitation learning framework for valid IV identification and policy optimization. In particular, in the first stage, we construct a testing criterion based on the defined pseudo-variable, with which we achieve identifying a valid IV for the C2L models. Such a criterion entails the sufficient and necessary identifiability conditions for IV validity. In the second stage, with the identified IV, we propose two candidate policy learning approaches: one is based on a simulator, while the other is offline. Extensive experiments verified the effectiveness of identifying the valid IV as well as learning the policy. 

---
# A Versatile Pathology Co-pilot via Reasoning Enhanced Multimodal Large Language Model 

**Authors**: Zhe Xu, Ziyi Liu, Junlin Hou, Jiabo Ma, Cheng Jin, Yihui Wang, Zhixuan Chen, Zhengyu Zhang, Zhengrui Guo, Fengtao Zhou, Yingxue Xu, Xi Wang, Ronald Cheong Kin Chan, Li Liang, Hao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.17303)  

**Abstract**: Multimodal large language models (MLLMs) have emerged as powerful tools for computational pathology, offering unprecedented opportunities to integrate pathological images with language context for comprehensive diagnostic analysis. These models hold particular promise for automating complex tasks that traditionally require expert interpretation of pathologists. However, current MLLM approaches in pathology demonstrate significantly constrained reasoning capabilities, primarily due to their reliance on expensive chain-of-thought annotations. Additionally, existing methods remain limited to simplex application of visual question answering (VQA) at region-of-interest (ROI) level, failing to address the full spectrum of diagnostic needs such as ROI classification, detection, segmentation, whole-slide-image (WSI) classification and VQA in clinical practice. In this study, we present SmartPath-R1, a versatile MLLM capable of simultaneously addressing both ROI-level and WSI-level tasks while demonstrating robust pathological reasoning capability. Our framework combines scale-dependent supervised fine-tuning and task-aware reinforcement fine-tuning, which circumvents the requirement for chain-of-thought supervision by leveraging the intrinsic knowledge within MLLM. Furthermore, SmartPath-R1 integrates multiscale and multitask analysis through a mixture-of-experts mechanism, enabling dynamic processing for diverse tasks. We curate a large-scale dataset comprising 2.3M ROI samples and 188K WSI samples for training and evaluation. Extensive experiments across 72 tasks validate the effectiveness and superiority of the proposed approach. This work represents a significant step toward developing versatile, reasoning-enhanced AI systems for precision pathology. 

---
# On Temporal Guidance and Iterative Refinement in Audio Source Separation 

**Authors**: Tobias Morocutti, Jonathan Greif, Paul Primus, Florian Schmid, Gerhard Widmer  

**Link**: [PDF](https://arxiv.org/pdf/2507.17297)  

**Abstract**: Spatial semantic segmentation of sound scenes (S5) involves the accurate identification of active sound classes and the precise separation of their sources from complex acoustic mixtures. Conventional systems rely on a two-stage pipeline - audio tagging followed by label-conditioned source separation - but are often constrained by the absence of fine-grained temporal information critical for effective separation. In this work, we address this limitation by introducing a novel approach for S5 that enhances the synergy between the event detection and source separation stages. Our key contributions are threefold. First, we fine-tune a pre-trained Transformer to detect active sound classes. Second, we utilize a separate instance of this fine-tuned Transformer to perform sound event detection (SED), providing the separation module with detailed, time-varying guidance. Third, we implement an iterative refinement mechanism that progressively enhances separation quality by recursively reusing the separator's output from previous iterations. These advancements lead to significant improvements in both audio tagging and source separation performance, as demonstrated by our system's second-place finish in Task 4 of the DCASE Challenge 2025. Our implementation and model checkpoints are available in our GitHub repository: this https URL . 

---
# Integrating Belief Domains into Probabilistic Logic Programs 

**Authors**: Damiano Azzolini, Fabrizio Riguzzi, Theresa Swift  

**Link**: [PDF](https://arxiv.org/pdf/2507.17291)  

**Abstract**: Probabilistic Logic Programming (PLP) under the Distribution Semantics is a leading approach to practical reasoning under uncertainty. An advantage of the Distribution Semantics is its suitability for implementation as a Prolog or Python library, available through two well-maintained implementations, namely ProbLog and cplint/PITA. However, current formulations of the Distribution Semantics use point-probabilities, making it difficult to express epistemic uncertainty, such as arises from, for example, hierarchical classifications from computer vision models. Belief functions generalize probability measures as non-additive capacities, and address epistemic uncertainty via interval probabilities. This paper introduces interval-based Capacity Logic Programs based on an extension of the Distribution Semantics to include belief functions, and describes properties of the new framework that make it amenable to practical applications. 

---
# Leveraging Knowledge Graphs and LLM Reasoning to Identify Operational Bottlenecks for Warehouse Planning Assistance 

**Authors**: Rishi Parekh, Saisubramaniam Gopalakrishnan, Zishan Ahmad, Anirudh Deodhar  

**Link**: [PDF](https://arxiv.org/pdf/2507.17273)  

**Abstract**: Analyzing large, complex output datasets from Discrete Event Simulations (DES) of warehouse operations to identify bottlenecks and inefficiencies is a critical yet challenging task, often demanding significant manual effort or specialized analytical tools. Our framework integrates Knowledge Graphs (KGs) and Large Language Model (LLM)-based agents to analyze complex Discrete Event Simulation (DES) output data from warehouse operations. It transforms raw DES data into a semantically rich KG, capturing relationships between simulation events and entities. An LLM-based agent uses iterative reasoning, generating interdependent sub-questions. For each sub-question, it creates Cypher queries for KG interaction, extracts information, and self-reflects to correct errors. This adaptive, iterative, and self-correcting process identifies operational issues mimicking human analysis. Our DES approach for warehouse bottleneck identification, tested with equipment breakdowns and process irregularities, outperforms baseline methods. For operational questions, it achieves near-perfect pass rates in pinpointing inefficiencies. For complex investigative questions, we demonstrate its superior diagnostic ability to uncover subtle, interconnected issues. This work bridges simulation modeling and AI (KG+LLM), offering a more intuitive method for actionable insights, reducing time-to-insight, and enabling automated warehouse inefficiency evaluation and diagnosis. 

---
# Understanding Prompt Programming Tasks and Questions 

**Authors**: Jenny T. Liang, Chenyang Yang, Agnia Sergeyuk, Travis D. Breaux, Brad A. Myers  

**Link**: [PDF](https://arxiv.org/pdf/2507.17264)  

**Abstract**: Prompting foundation models (FMs) like large language models (LLMs) have enabled new AI-powered software features (e.g., text summarization) that previously were only possible by fine-tuning FMs. Now, developers are embedding prompts in software, known as prompt programs. The process of prompt programming requires the developer to make many changes to their prompt. Yet, the questions developers ask to update their prompt is unknown, despite the answers to these questions affecting how developers plan their changes. With the growing number of research and commercial prompt programming tools, it is unclear whether prompt programmers' needs are being adequately addressed. We address these challenges by developing a taxonomy of 25 tasks prompt programmers do and 51 questions they ask, measuring the importance of each task and question. We interview 16 prompt programmers, observe 8 developers make prompt changes, and survey 50 developers. We then compare the taxonomy with 48 research and commercial tools. We find that prompt programming is not well-supported: all tasks are done manually, and 16 of the 51 questions -- including a majority of the most important ones -- remain unanswered. Based on this, we outline important opportunities for prompt programming tools. 

---
# Reality Proxy: Fluid Interactions with Real-World Objects in MR via Abstract Representations 

**Authors**: Xiaoan Liu, Difan Jia, Xianhao Carton Liu, Mar Gonzalez-Franco, Chen Zhu-Tian  

**Link**: [PDF](https://arxiv.org/pdf/2507.17248)  

**Abstract**: Interacting with real-world objects in Mixed Reality (MR) often proves difficult when they are crowded, distant, or partially occluded, hindering straightforward selection and manipulation. We observe that these difficulties stem from performing interaction directly on physical objects, where input is tightly coupled to their physical constraints. Our key insight is to decouple interaction from these constraints by introducing proxies-abstract representations of real-world objects. We embody this concept in Reality Proxy, a system that seamlessly shifts interaction targets from physical objects to their proxies during selection. Beyond facilitating basic selection, Reality Proxy uses AI to enrich proxies with semantic attributes and hierarchical spatial relationships of their corresponding physical objects, enabling novel and previously cumbersome interactions in MR - such as skimming, attribute-based filtering, navigating nested groups, and complex multi object selections - all without requiring new gestures or menu systems. We demonstrate Reality Proxy's versatility across diverse scenarios, including office information retrieval, large-scale spatial navigation, and multi-drone control. An expert evaluation suggests the system's utility and usability, suggesting that proxy-based abstractions offer a powerful and generalizable interaction paradigm for future MR systems. 

---
# DistrAttention: An Efficient and Flexible Self-Attention Mechanism on Modern GPUs 

**Authors**: Haolin Jin, Mengbai Xiao, Yuan Yuan, Xiao Zhang, Dongxiao Yu, Guanghui Zhang, Haoliang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.17245)  

**Abstract**: The Transformer architecture has revolutionized deep learning, delivering the state-of-the-art performance in areas such as natural language processing, computer vision, and time series prediction. However, its core component, self-attention, has the quadratic time complexity relative to input sequence length, which hinders the scalability of Transformers. The exsiting approaches on optimizing self-attention either discard full-contextual information or lack of flexibility. In this work, we design DistrAttention, an effcient and flexible self-attention mechanism with the full context. DistrAttention achieves this by grouping data on the embedding dimensionality, usually referred to as $d$. We realize DistrAttention with a lightweight sampling and fusion method that exploits locality-sensitive hashing to group similar data. A block-wise grouping framework is further designed to limit the errors introduced by locality sensitive hashing. By optimizing the selection of block sizes, DistrAttention could be easily integrated with FlashAttention-2, gaining high-performance on modern GPUs. We evaluate DistrAttention with extensive experiments. The results show that our method is 37% faster than FlashAttention-2 on calculating self-attention. In ViT inference, DistrAttention is the fastest and the most accurate among approximate self-attention mechanisms. In Llama3-1B, DistrAttention still achieves the lowest inference time with only 1% accuray loss. 

---
# Eco-Friendly AI: Unleashing Data Power for Green Federated Learning 

**Authors**: Mattia Sabella, Monica Vitali  

**Link**: [PDF](https://arxiv.org/pdf/2507.17241)  

**Abstract**: The widespread adoption of Artificial Intelligence (AI) and Machine Learning (ML) comes with a significant environmental impact, particularly in terms of energy consumption and carbon emissions. This pressing issue highlights the need for innovative solutions to mitigate AI's ecological footprint. One of the key factors influencing the energy consumption of ML model training is the size of the training dataset. ML models are often trained on vast amounts of data continuously generated by sensors and devices distributed across multiple locations. To reduce data transmission costs and enhance privacy, Federated Learning (FL) enables model training without the need to move or share raw data. While FL offers these advantages, it also introduces challenges due to the heterogeneity of data sources (related to volume and quality), computational node capabilities, and environmental impact.
This paper contributes to the advancement of Green AI by proposing a data-centric approach to Green Federated Learning. Specifically, we focus on reducing FL's environmental impact by minimizing the volume of training data. Our methodology involves the analysis of the characteristics of federated datasets, the selecting of an optimal subset of data based on quality metrics, and the choice of the federated nodes with the lowest environmental impact. We develop a comprehensive methodology that examines the influence of data-centric factors, such as data quality and volume, on FL training performance and carbon emissions. Building on these insights, we introduce an interactive recommendation system that optimizes FL configurations through data reduction, minimizing environmental impact during training. Applying this methodology to time series classification has demonstrated promising results in reducing the environmental impact of FL tasks. 

---
# A Highly Clean Recipe Dataset with Ingredient States Annotation for State Probing Task 

**Authors**: Mashiro Toyooka, Kiyoharu Aizawa, Yoko Yamakata  

**Link**: [PDF](https://arxiv.org/pdf/2507.17232)  

**Abstract**: Large Language Models (LLMs) are trained on a vast amount of procedural texts, but they do not directly observe real-world phenomena. In the context of cooking recipes, this poses a challenge, as intermediate states of ingredients are often omitted, making it difficult for models to track ingredient states and understand recipes accurately. In this paper, we apply state probing, a method for evaluating a language model's understanding of the world, to the domain of cooking. We propose a new task and dataset for evaluating how well LLMs can recognize intermediate ingredient states during cooking procedures. We first construct a new Japanese recipe dataset with clear and accurate annotations of ingredient state changes, collected from well-structured and controlled recipe texts. Using this dataset, we design three novel tasks to evaluate whether LLMs can track ingredient state transitions and identify ingredients present at intermediate steps. Our experiments with widely used LLMs, such as Llama3.1-70B and Qwen2.5-72B, show that learning ingredient state knowledge improves their understanding of cooking processes, achieving performance comparable to commercial LLMs. 

---
# P3SL: Personalized Privacy-Preserving Split Learning on Heterogeneous Edge Devices 

**Authors**: Wei Fan, JinYi Yoon, Xiaochang Li, Huajie Shao, Bo Ji  

**Link**: [PDF](https://arxiv.org/pdf/2507.17228)  

**Abstract**: Split Learning (SL) is an emerging privacy-preserving machine learning technique that enables resource constrained edge devices to participate in model training by partitioning a model into client-side and server-side sub-models. While SL reduces computational overhead on edge devices, it encounters significant challenges in heterogeneous environments where devices vary in computing resources, communication capabilities, environmental conditions, and privacy requirements. Although recent studies have explored heterogeneous SL frameworks that optimize split points for devices with varying resource constraints, they often neglect personalized privacy requirements and local model customization under varying environmental conditions. To address these limitations, we propose P3SL, a Personalized Privacy-Preserving Split Learning framework designed for heterogeneous, resource-constrained edge device systems. The key contributions of this work are twofold. First, we design a personalized sequential split learning pipeline that allows each client to achieve customized privacy protection and maintain personalized local models tailored to their computational resources, environmental conditions, and privacy needs. Second, we adopt a bi-level optimization technique that empowers clients to determine their own optimal personalized split points without sharing private sensitive information (i.e., computational resources, environmental conditions, privacy requirements) with the server. This approach balances energy consumption and privacy leakage risks while maintaining high model accuracy. We implement and evaluate P3SL on a testbed consisting of 7 devices including 4 Jetson Nano P3450 devices, 2 Raspberry Pis, and 1 laptop, using diverse model architectures and datasets under varying environmental conditions. 

---
# HuiduRep: A Robust Self-Supervised Framework for Learning Neural Representations from Extracellular Spikes 

**Authors**: Feng Cao, Zishuo Feng  

**Link**: [PDF](https://arxiv.org/pdf/2507.17224)  

**Abstract**: Extracellular recordings are brief voltage fluctuations recorded near neurons, widely used in neuroscience as the basis for decoding brain activity at single-neuron resolution. Spike sorting, which assigns each spike to its source neuron, is a critical step in brain sensing pipelines. However, it remains challenging under low signal-to-noise ratio (SNR), electrode drift, and cross-session variability. In this paper, we propose HuiduRep, a robust self-supervised representation learning framework that extracts discriminative and generalizable features from extracellular spike waveforms. By combining contrastive learning with a denoising autoencoder, HuiduRep learns latent representations that are robust to noise and drift. Built on HuiduRep, we develop a spike sorting pipeline that clusters spike representations without supervision. Experiments on hybrid and real-world datasets demonstrate that HuiduRep achieves strong robustness and the pipeline matches or outperforms state-of-the-art tools such as KiloSort4 and MountainSort5. These findings demonstrate the potential of self-supervised spike representation learning as a foundational tool for robust and generalizable processing of extracellular recordings. 

---
# The Pluralistic Moral Gap: Understanding Judgment and Value Differences between Humans and Large Language Models 

**Authors**: Giuseppe Russo, Debora Nozza, Paul Röttger, Dirk Hovy  

**Link**: [PDF](https://arxiv.org/pdf/2507.17216)  

**Abstract**: People increasingly rely on Large Language Models (LLMs) for moral advice, which may influence humans' decisions. Yet, little is known about how closely LLMs align with human moral judgments. To address this, we introduce the Moral Dilemma Dataset, a benchmark of 1,618 real-world moral dilemmas paired with a distribution of human moral judgments consisting of a binary evaluation and a free-text rationale. We treat this problem as a pluralistic distributional alignment task, comparing the distributions of LLM and human judgments across dilemmas. We find that models reproduce human judgments only under high consensus; alignment deteriorates sharply when human disagreement increases. In parallel, using a 60-value taxonomy built from 3,783 value expressions extracted from rationales, we show that LLMs rely on a narrower set of moral values than humans. These findings reveal a pluralistic moral gap: a mismatch in both the distribution and diversity of values expressed. To close this gap, we introduce Dynamic Moral Profiling (DMP), a Dirichlet-based sampling method that conditions model outputs on human-derived value profiles. DMP improves alignment by 64.3% and enhances value diversity, offering a step toward more pluralistic and human-aligned moral guidance from LLMs. 

---
# DesignLab: Designing Slides Through Iterative Detection and Correction 

**Authors**: Jooyeol Yun, Heng Wang, Yotaro Shimose, Jaegul Choo, Shingo Takamatsu  

**Link**: [PDF](https://arxiv.org/pdf/2507.17202)  

**Abstract**: Designing high-quality presentation slides can be challenging for non-experts due to the complexity involved in navigating various design choices. Numerous automated tools can suggest layouts and color schemes, yet often lack the ability to refine their own output, which is a key aspect in real-world workflows. We propose DesignLab, which separates the design process into two roles, the design reviewer, who identifies design-related issues, and the design contributor who corrects them. This decomposition enables an iterative loop where the reviewer continuously detects issues and the contributor corrects them, allowing a draft to be further polished with each iteration, reaching qualities that were unattainable. We fine-tune large language models for these roles and simulate intermediate drafts by introducing controlled perturbations, enabling the design reviewer learn design errors and the contributor learn how to fix them. Our experiments show that DesignLab outperforms existing design-generation methods, including a commercial tool, by embracing the iterative nature of designing which can result in polished, professional slides. 

---
# Dispatch-Aware Deep Neural Network for Optimal Transmission Switching: Toward Real-Time and Feasibility Guaranteed Operation 

**Authors**: Minsoo Kim, Jip Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.17194)  

**Abstract**: Optimal transmission switching (OTS) improves optimal power flow (OPF) by selectively opening transmission lines, but its mixed-integer formulation increases computational complexity, especially on large grids. To deal with this, we propose a dispatch-aware deep neural network (DA-DNN) that accelerates DC-OTS without relying on pre-solved labels. DA-DNN predicts line states and passes them through a differentiable DC-OPF layer, using the resulting generation cost as the loss function so that all physical network constraints are enforced throughout training and inference. In addition, we adopt a customized weight-bias initialization that keeps every forward pass feasible from the first iteration, which allows stable learning on large grids. Once trained, the proposed DA-DNN produces a provably feasible topology and dispatch pair in the same time as solving the DCOPF, whereas conventional mixed-integer solvers become intractable. As a result, the proposed method successfully captures the economic advantages of OTS while maintaining scalability. 

---
# LLM Meets the Sky: Heuristic Multi-Agent Reinforcement Learning for Secure Heterogeneous UAV Networks 

**Authors**: Lijie Zheng, Ji He, Shih Yu Chang, Yulong Shen, Dusit Niyato  

**Link**: [PDF](https://arxiv.org/pdf/2507.17188)  

**Abstract**: This work tackles the physical layer security (PLS) problem of maximizing the secrecy rate in heterogeneous UAV networks (HetUAVNs) under propulsion energy constraints. Unlike prior studies that assume uniform UAV capabilities or overlook energy-security trade-offs, we consider a realistic scenario where UAVs with diverse payloads and computation resources collaborate to serve ground terminals in the presence of eavesdroppers. To manage the complex coupling between UAV motion and communication, we propose a hierarchical optimization framework. The inner layer uses a semidefinite relaxation (SDR)-based S2DC algorithm combining penalty functions and difference-of-convex (d.c.) programming to solve the secrecy precoding problem with fixed UAV positions. The outer layer introduces a Large Language Model (LLM)-guided heuristic multi-agent reinforcement learning approach (LLM-HeMARL) for trajectory optimization. LLM-HeMARL efficiently incorporates expert heuristics policy generated by the LLM, enabling UAVs to learn energy-aware, security-driven trajectories without the inference overhead of real-time LLM calls. The simulation results show that our method outperforms existing baselines in secrecy rate and energy efficiency, with consistent robustness across varying UAV swarm sizes and random seeds. 

---
# Asymmetric Lesion Detection with Geometric Patterns and CNN-SVM Classification 

**Authors**: M. A. Rasel, Sameem Abdul Kareem, Zhenli Kwan, Nik Aimee Azizah Faheem, Winn Hui Han, Rebecca Kai Jan Choong, Shin Shen Yong, Unaizah Obaidellah  

**Link**: [PDF](https://arxiv.org/pdf/2507.17185)  

**Abstract**: In dermoscopic images, which allow visualization of surface skin structures not visible to the naked eye, lesion shape offers vital insights into skin diseases. In clinically practiced methods, asymmetric lesion shape is one of the criteria for diagnosing melanoma. Initially, we labeled data for a non-annotated dataset with symmetrical information based on clinical assessments. Subsequently, we propose a supporting technique, a supervised learning image processing algorithm, to analyze the geometrical pattern of lesion shape, aiding non-experts in understanding the criteria of an asymmetric lesion. We then utilize a pre-trained convolutional neural network (CNN) to extract shape, color, and texture features from dermoscopic images for training a multiclass support vector machine (SVM) classifier, outperforming state-of-the-art methods from the literature. In the geometry-based experiment, we achieved a 99.00% detection rate for dermatological asymmetric lesions. In the CNN-based experiment, the best performance is found with 94% Kappa Score, 95% Macro F1-score, and 97% Weighted F1-score for classifying lesion shapes (Asymmetric, Half-Symmetric, and Symmetric). 

---
# Regret Minimization in Population Network Games: Vanishing Heterogeneity and Convergence to Equilibria 

**Authors**: Die Hu, Shuyue Hu, Chunjiang Mu, Shiqi Fan, Chen Chu, Jinzhuo Liu, Zhen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.17183)  

**Abstract**: Understanding and predicting the behavior of large-scale multi-agents in games remains a fundamental challenge in multi-agent systems. This paper examines the role of heterogeneity in equilibrium formation by analyzing how smooth regret-matching drives a large number of heterogeneous agents with diverse initial policies toward unified behavior. By modeling the system state as a probability distribution of regrets and analyzing its evolution through the continuity equation, we uncover a key phenomenon in diverse multi-agent settings: the variance of the regret distribution diminishes over time, leading to the disappearance of heterogeneity and the emergence of consensus among agents. This universal result enables us to prove convergence to quantal response equilibria in both competitive and cooperative multi-agent settings. Our work advances the theoretical understanding of multi-agent learning and offers a novel perspective on equilibrium selection in diverse game-theoretic scenarios. 

---
# SKA-Bench: A Fine-Grained Benchmark for Evaluating Structured Knowledge Understanding of LLMs 

**Authors**: Zhiqiang Liu, Enpei Niu, Yin Hua, Mengshu Sun, Lei Liang, Huajun Chen, Wen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.17178)  

**Abstract**: Although large language models (LLMs) have made significant progress in understanding Structured Knowledge (SK) like KG and Table, existing evaluations for SK understanding are non-rigorous (i.e., lacking evaluations of specific capabilities) and focus on a single type of SK. Therefore, we aim to propose a more comprehensive and rigorous structured knowledge understanding benchmark to diagnose the shortcomings of LLMs. In this paper, we introduce SKA-Bench, a Structured Knowledge Augmented QA Benchmark that encompasses four widely used structured knowledge forms: KG, Table, KG+Text, and Table+Text. We utilize a three-stage pipeline to construct SKA-Bench instances, which includes a question, an answer, positive knowledge units, and noisy knowledge units. To evaluate the SK understanding capabilities of LLMs in a fine-grained manner, we expand the instances into four fundamental ability testbeds: Noise Robustness, Order Insensitivity, Information Integration, and Negative Rejection. Empirical evaluations on 8 representative LLMs, including the advanced DeepSeek-R1, indicate that existing LLMs still face significant challenges in understanding structured knowledge, and their performance is influenced by factors such as the amount of noise, the order of knowledge units, and hallucination phenomenon. Our dataset and code are available at this https URL. 

---
# Tabular Diffusion based Actionable Counterfactual Explanations for Network Intrusion Detection 

**Authors**: Vinura Galwaduge, Jagath Samarabandu  

**Link**: [PDF](https://arxiv.org/pdf/2507.17161)  

**Abstract**: Modern network intrusion detection systems (NIDS) frequently utilize the predictive power of complex deep learning models. However, the "black-box" nature of such deep learning methods adds a layer of opaqueness that hinders the proper understanding of detection decisions, trust in the decisions and prevent timely countermeasures against such attacks. Explainable AI (XAI) methods provide a solution to this problem by providing insights into the causes of the predictions. The majority of the existing XAI methods provide explanations which are not convenient to convert into actionable countermeasures. In this work, we propose a novel diffusion-based counterfactual explanation framework that can provide actionable explanations for network intrusion attacks. We evaluated our proposed algorithm against several other publicly available counterfactual explanation algorithms on 3 modern network intrusion datasets. To the best of our knowledge, this work also presents the first comparative analysis of existing counterfactual explanation algorithms within the context of network intrusion detection systems. Our proposed method provide minimal, diverse counterfactual explanations out of the tested counterfactual explanation algorithms in a more efficient manner by reducing the time to generate explanations. We also demonstrate how counterfactual explanations can provide actionable explanations by summarizing them to create a set of global rules. These rules are actionable not only at instance level but also at the global level for intrusion attacks. These global counterfactual rules show the ability to effectively filter out incoming attack queries which is crucial for efficient intrusion detection and defense mechanisms. 

---
# JAM: Keypoint-Guided Joint Prediction after Classification-Aware Marginal Proposal for Multi-Agent Interaction 

**Authors**: Fangze Lin, Ying He, Fei Yu, Hong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.17152)  

**Abstract**: Predicting the future motion of road participants is a critical task in autonomous driving. In this work, we address the challenge of low-quality generation of low-probability modes in multi-agent joint prediction. To tackle this issue, we propose a two-stage multi-agent interactive prediction framework named \textit{keypoint-guided joint prediction after classification-aware marginal proposal} (JAM). The first stage is modeled as a marginal prediction process, which classifies queries by trajectory type to encourage the model to learn all categories of trajectories, providing comprehensive mode information for the joint prediction module. The second stage is modeled as a joint prediction process, which takes the scene context and the marginal proposals from the first stage as inputs to learn the final joint distribution. We explicitly introduce key waypoints to guide the joint prediction module in better capturing and leveraging the critical information from the initial predicted trajectories. We conduct extensive experiments on the real-world Waymo Open Motion Dataset interactive prediction benchmark. The results show that our approach achieves competitive performance. In particular, in the framework comparison experiments, the proposed JAM outperforms other prediction frameworks and achieves state-of-the-art performance in interactive trajectory prediction. The code is available at this https URL to facilitate future research. 

---
# ScSAM: Debiasing Morphology and Distributional Variability in Subcellular Semantic Segmentation 

**Authors**: Bo Fang, Jianan Fan, Dongnan Liu, Hang Chang, Gerald J.Shami, Filip Braet, Weidong Cai  

**Link**: [PDF](https://arxiv.org/pdf/2507.17149)  

**Abstract**: The significant morphological and distributional variability among subcellular components poses a long-standing challenge for learning-based organelle segmentation models, significantly increasing the risk of biased feature learning. Existing methods often rely on single mapping relationships, overlooking feature diversity and thereby inducing biased training. Although the Segment Anything Model (SAM) provides rich feature representations, its application to subcellular scenarios is hindered by two key challenges: (1) The variability in subcellular morphology and distribution creates gaps in the label space, leading the model to learn spurious or biased features. (2) SAM focuses on global contextual understanding and often ignores fine-grained spatial details, making it challenging to capture subtle structural alterations and cope with skewed data distributions. To address these challenges, we introduce ScSAM, a method that enhances feature robustness by fusing pre-trained SAM with Masked Autoencoder (MAE)-guided cellular prior knowledge to alleviate training bias from data imbalance. Specifically, we design a feature alignment and fusion module to align pre-trained embeddings to the same feature space and efficiently combine different representations. Moreover, we present a cosine similarity matrix-based class prompt encoder to activate class-specific features to recognize subcellular categories. Extensive experiments on diverse subcellular image datasets demonstrate that ScSAM outperforms state-of-the-art methods. 

---
# Towards Human-level Intelligence via Human-like Whole-Body Manipulation 

**Authors**: Guang Gao, Jianan Wang, Jinbo Zuo, Junnan Jiang, Jingfan Zhang, Xianwen Zeng, Yuejiang Zhu, Lianyang Ma, Ke Chen, Minhua Sheng, Ruirui Zhang, Zhaohui An  

**Link**: [PDF](https://arxiv.org/pdf/2507.17141)  

**Abstract**: Building general-purpose intelligent robots has long been a fundamental goal of robotics. A promising approach is to mirror the evolutionary trajectory of humans: learning through continuous interaction with the environment, with early progress driven by the imitation of human behaviors. Achieving this goal presents three core challenges: (1) designing safe robotic hardware with human-level physical capabilities; (2) developing an intuitive and scalable whole-body teleoperation interface for data collection; and (3) creating algorithms capable of learning whole-body visuomotor policies from human demonstrations. To address these challenges in a unified framework, we propose Astribot Suite, a robot learning suite for whole-body manipulation aimed at general daily tasks across diverse environments. We demonstrate the effectiveness of our system on a wide range of activities that require whole-body coordination, extensive reachability, human-level dexterity, and agility. Our results show that Astribot's cohesive integration of embodiment, teleoperation interface, and learning pipeline marks a significant step towards real-world, general-purpose whole-body robotic manipulation, laying the groundwork for the next generation of intelligent robots. 

---
# SADA: Stability-guided Adaptive Diffusion Acceleration 

**Authors**: Ting Jiang, Yixiao Wang, Hancheng Ye, Zishan Shao, Jingwei Sun, Jingyang Zhang, Zekai Chen, Jianyi Zhang, Yiran Chen, Hai Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.17135)  

**Abstract**: Diffusion models have achieved remarkable success in generative tasks but suffer from high computational costs due to their iterative sampling process and quadratic attention costs. Existing training-free acceleration strategies that reduce per-step computation cost, while effectively reducing sampling time, demonstrate low faithfulness compared to the original baseline. We hypothesize that this fidelity gap arises because (a) different prompts correspond to varying denoising trajectory, and (b) such methods do not consider the underlying ODE formulation and its numerical solution. In this paper, we propose Stability-guided Adaptive Diffusion Acceleration (SADA), a novel paradigm that unifies step-wise and token-wise sparsity decisions via a single stability criterion to accelerate sampling of ODE-based generative models (Diffusion and Flow-matching). For (a), SADA adaptively allocates sparsity based on the sampling trajectory. For (b), SADA introduces principled approximation schemes that leverage the precise gradient information from the numerical ODE solver. Comprehensive evaluations on SD-2, SDXL, and Flux using both EDM and DPM++ solvers reveal consistent $\ge 1.8\times$ speedups with minimal fidelity degradation (LPIPS $\leq 0.10$ and FID $\leq 4.5$) compared to unmodified baselines, significantly outperforming prior methods. Moreover, SADA adapts seamlessly to other pipelines and modalities: It accelerates ControlNet without any modifications and speeds up MusicLDM by $1.8\times$ with $\sim 0.01$ spectrogram LPIPS. 

---
# Resilient Multi-Agent Negotiation for Medical Supply Chains:Integrating LLMs and Blockchain for Transparent Coordination 

**Authors**: Mariam ALMutairi, Hyungmin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.17134)  

**Abstract**: Global health emergencies, such as the COVID-19 pandemic, have exposed critical weaknesses in traditional medical supply chains, including inefficiencies in resource allocation, lack of transparency, and poor adaptability to dynamic disruptions. This paper presents a novel hybrid framework that integrates blockchain technology with a decentralized, large language model (LLM) powered multi-agent negotiation system to enhance the resilience and accountability of medical supply chains during crises. In this system, autonomous agents-representing manufacturers, distributors, and healthcare institutions-engage in structured, context-aware negotiation and decision-making processes facilitated by LLMs, enabling rapid and ethical allocation of scarce medical resources. The off-chain agent layer supports adaptive reasoning and local decision-making, while the on-chain blockchain layer ensures immutable, transparent, and auditable enforcement of decisions via smart contracts. The framework also incorporates a formal cross-layer communication protocol to bridge decentralized negotiation with institutional enforcement. A simulation environment emulating pandemic scenarios evaluates the system's performance, demonstrating improvements in negotiation efficiency, fairness of allocation, supply chain responsiveness, and auditability. This research contributes an innovative approach that synergizes blockchain trust guarantees with the adaptive intelligence of LLM-driven agents, providing a robust and scalable solution for critical supply chain coordination under uncertainty. 

---
# Enabling Self-Improving Agents to Learn at Test Time With Human-In-The-Loop Guidance 

**Authors**: Yufei He, Ruoyu Li, Alex Chen, Yue Liu, Yulin Chen, Yuan Sui, Cheng Chen, Yi Zhu, Luca Luo, Frank Yang, Bryan Hooi  

**Link**: [PDF](https://arxiv.org/pdf/2507.17131)  

**Abstract**: Large language model (LLM) agents often struggle in environments where rules and required domain knowledge frequently change, such as regulatory compliance and user risk screening. Current approaches, like offline fine-tuning and standard prompting, are insufficient because they cannot effectively adapt to new knowledge during actual operation. To address this limitation, we propose the Adaptive Reflective Interactive Agent (ARIA), an LLM agent framework designed specifically to continuously learn updated domain knowledge at test time. ARIA assesses its own uncertainty through structured self-dialogue, proactively identifying knowledge gaps and requesting targeted explanations or corrections from human experts. It then systematically updates an internal, timestamped knowledge repository with provided human guidance, detecting and resolving conflicting or outdated knowledge through comparisons and clarification queries. We evaluate ARIA on the realistic customer due diligence name screening task on TikTok Pay, alongside publicly available dynamic knowledge tasks. Results demonstrate significant improvements in adaptability and accuracy compared to baselines using standard offline fine-tuning and existing self-improving agents. ARIA is deployed within TikTok Pay serving over 150 million monthly active users, confirming its practicality and effectiveness for operational use in rapidly evolving environments. 

---
# BucketServe: Bucket-Based Dynamic Batching for Smart and Efficient LLM Inference Serving 

**Authors**: Wanyi Zheng, Minxian Xu, Shengye Song, Kejiang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2507.17120)  

**Abstract**: Large language models (LLMs) have become increasingly popular in various areas, traditional business gradually shifting from rule-based systems to LLM-based solutions. However, the inference of LLMs is resource-intensive or latency-sensitive, posing significant challenges for serving systems. Existing LLM serving systems often use static or continuous batching strategies, which can lead to inefficient GPU memory utilization and increased latency, especially under heterogeneous workloads. These methods may also struggle to adapt to dynamic workload fluctuations, resulting in suboptimal throughput and potential service level objective (SLO) violations. In this paper, we introduce BucketServe, a bucket-based dynamic batching framework designed to optimize LLM inference performance. By grouping requests into size-homogeneous buckets based on sequence length, BucketServe minimizes padding overhead and optimizes GPU memory usage through real-time batch size adjustments preventing out-of-memory (OOM) errors. It introduces adaptive bucket splitting/merging and priority-aware scheduling to mitigate resource fragmentation and ensure SLO compliance. Experiment shows that BucketServe significantly outperforms UELLM in throughput, achieving up to 3.58x improvement. It can also handle 1.93x more request load under the SLO attainment of 80% compared with DistServe and demonstrates 1.975x higher system load capacity compared to the UELLM. 

---
# Reinforcement Learning Fine-Tunes a Sparse Subnetwork in Large Language Models 

**Authors**: Andrii Balashov  

**Link**: [PDF](https://arxiv.org/pdf/2507.17107)  

**Abstract**: Reinforcement learning (RL) is a key post-pretraining step for aligning large language models (LLMs) with complex tasks and human preferences. While it is often assumed that RL fine-tuning requires updating most of a model's parameters, we challenge this assumption with a surprising finding: RL fine-tuning consistently modifies only a small subnetwork (typically 5-30% of weights), leaving most parameters unchanged. We call this phenomenon RL-induced parameter update sparsity. It arises naturally, without any sparsity constraints or parameter-efficient tuning, and appears across multiple RL algorithms (e.g., PPO, DPO, SimPO, PRIME) and model families (e.g., OpenAI, Meta, and open-source LLMs). Moreover, the subnetworks updated by RL show substantial overlap across different seeds, datasets, and algorithms-far exceeding chance-suggesting a partially transferable structure in the pretrained model. We show that fine-tuning only this sparse subnetwork recovers full model performance and yields parameters nearly identical to the fully fine-tuned model. Our analysis suggests this sparsity emerges because RL operates near the model's original distribution, requiring only targeted changes. KL penalties, gradient clipping, and on-policy dynamics have limited effect on the sparsity pattern. These findings shed new light on how RL adapts models: not by shifting all weights, but by focusing training on a small, consistently updated subnetwork. This insight enables more efficient RL methods and reframes sparsity through the lens of the lottery ticket hypothesis. 

---
# Weather-Aware AI Systems versus Route-Optimization AI: A Comprehensive Analysis of AI Applications in Transportation Productivity 

**Authors**: Tatsuru Kikuchi  

**Link**: [PDF](https://arxiv.org/pdf/2507.17099)  

**Abstract**: While recent research demonstrates that AI route-optimization systems improve taxi driver productivity by 14\%, this study reveals that such findings capture only a fraction of AI's potential in transportation. We examine comprehensive weather-aware AI systems that integrate deep learning meteorological prediction with machine learning positioning optimization, comparing their performance against traditional operations and route-only AI approaches. Using simulation data from 10,000 taxi operations across varied weather conditions, we find that weather-aware AI systems increase driver revenue by 107.3\%, compared to 14\% improvements from route-optimization alone. Weather prediction contributes the largest individual productivity gain, with strong correlations between meteorological conditions and demand ($r=0.575$). Economic analysis reveals annual earnings increases of 13.8 million yen per driver, with rapid payback periods and superior return on investment. These findings suggest that current AI literature significantly underestimates AI's transformative potential by focusing narrowly on routing algorithms, while weather intelligence represents an untapped \$8.9 billion market opportunity. Our results indicate that future AI implementations should adopt comprehensive approaches that address multiple operational challenges simultaneously rather than optimizing isolated functions. 

---
# SDGOCC: Semantic and Depth-Guided Bird's-Eye View Transformation for 3D Multimodal Occupancy Prediction 

**Authors**: Zaipeng Duan, Chenxu Dang, Xuzhong Hu, Pei An, Junfeng Ding, Jie Zhan, Yunbiao Xu, Jie Ma  

**Link**: [PDF](https://arxiv.org/pdf/2507.17083)  

**Abstract**: Multimodal 3D occupancy prediction has garnered significant attention for its potential in autonomous driving. However, most existing approaches are single-modality: camera-based methods lack depth information, while LiDAR-based methods struggle with occlusions. Current lightweight methods primarily rely on the Lift-Splat-Shoot (LSS) pipeline, which suffers from inaccurate depth estimation and fails to fully exploit the geometric and semantic information of 3D LiDAR points. Therefore, we propose a novel multimodal occupancy prediction network called SDG-OCC, which incorporates a joint semantic and depth-guided view transformation coupled with a fusion-to-occupancy-driven active distillation. The enhanced view transformation constructs accurate depth distributions by integrating pixel semantics and co-point depth through diffusion and bilinear discretization. The fusion-to-occupancy-driven active distillation extracts rich semantic information from multimodal data and selectively transfers knowledge to image features based on LiDAR-identified regions. Finally, for optimal performance, we introduce SDG-Fusion, which uses fusion alone, and SDG-KL, which integrates both fusion and distillation for faster inference. Our method achieves state-of-the-art (SOTA) performance with real-time processing on the Occ3D-nuScenes dataset and shows comparable performance on the more challenging SurroundOcc-nuScenes dataset, demonstrating its effectiveness and robustness. The code will be released at this https URL. 

---
# VL-CLIP: Enhancing Multimodal Recommendations via Visual Grounding and LLM-Augmented CLIP Embeddings 

**Authors**: Ramin Giahi, Kehui Yao, Sriram Kollipara, Kai Zhao, Vahid Mirjalili, Jianpeng Xu, Topojoy Biswas, Evren Korpeoglu, Kannan Achan  

**Link**: [PDF](https://arxiv.org/pdf/2507.17080)  

**Abstract**: Multimodal learning plays a critical role in e-commerce recommendation platforms today, enabling accurate recommendations and product understanding. However, existing vision-language models, such as CLIP, face key challenges in e-commerce recommendation systems: 1) Weak object-level alignment, where global image embeddings fail to capture fine-grained product attributes, leading to suboptimal retrieval performance; 2) Ambiguous textual representations, where product descriptions often lack contextual clarity, affecting cross-modal matching; and 3) Domain mismatch, as generic vision-language models may not generalize well to e-commerce-specific data. To address these limitations, we propose a framework, VL-CLIP, that enhances CLIP embeddings by integrating Visual Grounding for fine-grained visual understanding and an LLM-based agent for generating enriched text embeddings. Visual Grounding refines image representations by localizing key products, while the LLM agent enhances textual features by disambiguating product descriptions. Our approach significantly improves retrieval accuracy, multimodal retrieval effectiveness, and recommendation quality across tens of millions of items on one of the largest e-commerce platforms in the U.S., increasing CTR by 18.6%, ATC by 15.5%, and GMV by 4.0%. Additional experimental results show that our framework outperforms vision-language models, including CLIP, FashionCLIP, and GCL, in both precision and semantic alignment, demonstrating the potential of combining object-aware visual grounding and LLM-enhanced text representation for robust multimodal recommendations. 

---
# Advancing Robustness in Deep Reinforcement Learning with an Ensemble Defense Approach 

**Authors**: Adithya Mohan, Dominik Rößle, Daniel Cremers, Torsten Schön  

**Link**: [PDF](https://arxiv.org/pdf/2507.17070)  

**Abstract**: Recent advancements in Deep Reinforcement Learning (DRL) have demonstrated its applicability across various domains, including robotics, healthcare, energy optimization, and autonomous driving. However, a critical question remains: How robust are DRL models when exposed to adversarial attacks? While existing defense mechanisms such as adversarial training and distillation enhance the resilience of DRL models, there remains a significant research gap regarding the integration of multiple defenses in autonomous driving scenarios specifically. This paper addresses this gap by proposing a novel ensemble-based defense architecture to mitigate adversarial attacks in autonomous driving. Our evaluation demonstrates that the proposed architecture significantly enhances the robustness of DRL models. Compared to the baseline under FGSM attacks, our ensemble method improves the mean reward from 5.87 to 18.38 (over 213% increase) and reduces the mean collision rate from 0.50 to 0.09 (an 82% decrease) in the highway scenario and merge scenario, outperforming all standalone defense strategies. 

---
# Compatibility of Max and Sum Objectives for Committee Selection and $k$-Facility Location 

**Authors**: Yue Han, Elliot Anshelevich  

**Link**: [PDF](https://arxiv.org/pdf/2507.17063)  

**Abstract**: We study a version of the metric facility location problem (or, equivalently, variants of the committee selection problem) in which we must choose $k$ facilities in an arbitrary metric space to serve some set of clients $C$. We consider four different objectives, where each client $i\in C$ attempts to minimize either the sum or the maximum of its distance to the chosen facilities, and where the overall objective either considers the sum or the maximum of the individual client costs. Rather than optimizing a single objective at a time, we study how compatible these objectives are with each other, and show the existence of solutions which are simultaneously close-to-optimum for any pair of the above objectives. Our results show that when choosing a set of facilities or a representative committee, it is often possible to form a solution which is good for several objectives at the same time, instead of sacrificing one desideratum to achieve another. 

---
# Parallelism Meets Adaptiveness: Scalable Documents Understanding in Multi-Agent LLM Systems 

**Authors**: Chengxuan Xia, Qianye Wu, Sixuan Tian, Yilun Hao  

**Link**: [PDF](https://arxiv.org/pdf/2507.17061)  

**Abstract**: Large language model (LLM) agents have shown increasing promise for collaborative task completion. However, existing multi-agent frameworks often rely on static workflows, fixed roles, and limited inter-agent communication, reducing their effectiveness in open-ended, high-complexity domains. This paper proposes a coordination framework that enables adaptiveness through three core mechanisms: dynamic task routing, bidirectional feedback, and parallel agent evaluation. The framework allows agents to reallocate tasks based on confidence and workload, exchange structured critiques to iteratively improve outputs, and crucially compete on high-ambiguity subtasks with evaluator-driven selection of the most suitable result. We instantiate these principles in a modular architecture and demonstrate substantial improvements in factual coverage, coherence, and efficiency over static and partially adaptive baselines. Our findings highlight the benefits of incorporating both adaptiveness and structured competition in multi-agent LLM systems. 

---
# Pragmatic Policy Development via Interpretable Behavior Cloning 

**Authors**: Anton Matsson, Yaochen Rao, Heather J. Litman, Fredrik D. Johansson  

**Link**: [PDF](https://arxiv.org/pdf/2507.17056)  

**Abstract**: Offline reinforcement learning (RL) holds great promise for deriving optimal policies from observational data, but challenges related to interpretability and evaluation limit its practical use in safety-critical domains. Interpretability is hindered by the black-box nature of unconstrained RL policies, while evaluation -- typically performed off-policy -- is sensitive to large deviations from the data-collecting behavior policy, especially when using methods based on importance sampling. To address these challenges, we propose a simple yet practical alternative: deriving treatment policies from the most frequently chosen actions in each patient state, as estimated by an interpretable model of the behavior policy. By using a tree-based model, which is specifically designed to exploit patterns in the data, we obtain a natural grouping of states with respect to treatment. The tree structure ensures interpretability by design, while varying the number of actions considered controls the degree of overlap with the behavior policy, enabling reliable off-policy evaluation. This pragmatic approach to policy development standardizes frequent treatment patterns, capturing the collective clinical judgment embedded in the data. Using real-world examples in rheumatoid arthritis and sepsis care, we demonstrate that policies derived under this framework can outperform current practice, offering interpretable alternatives to those obtained via offline RL. 

---
# Controllable Hybrid Captioner for Improved Long-form Video Understanding 

**Authors**: Kuleen Sasse, Efsun Sarioglu Kayi, Arun Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2507.17047)  

**Abstract**: Video data, especially long-form video, is extremely dense and high-dimensional. Text-based summaries of video content offer a way to represent query-relevant content in a much more compact manner than raw video. In addition, textual representations are easily ingested by state-of-the-art large language models (LLMs), which enable reasoning over video content to answer complex natural language queries. To solve this issue, we rely on the progressive construction of a text-based memory by a video captioner operating on shorter chunks of the video, where spatio-temporal modeling is computationally feasible. We explore ways to improve the quality of the activity log comprised solely of short video captions. Because the video captions tend to be focused on human actions, and questions may pertain to other information in the scene, we seek to enrich the memory with static scene descriptions using Vision Language Models (VLMs). Our video understanding system relies on the LaViLa video captioner in combination with a LLM to answer questions about videos. We first explored different ways of partitioning the video into meaningful segments such that the textual descriptions more accurately reflect the structure of the video content. Furthermore, we incorporated static scene descriptions into the captioning pipeline using LLaVA VLM, resulting in a more detailed and complete caption log and expanding the space of questions that are answerable from the textual memory. Finally, we have successfully fine-tuned the LaViLa video captioner to produce both action and scene captions, significantly improving the efficiency of the captioning pipeline compared to using separate captioning models for the two tasks. Our model, controllable hybrid captioner, can alternate between different types of captions according to special input tokens that signals scene changes detected in the video. 

---
# Computational Performance Bounds Prediction in Quantum Computing with Unstable Noise 

**Authors**: Jinyang Li, Samudra Dasgupta, Yuhong Song, Lei Yang, Travis Humble, Weiwen Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2507.17043)  

**Abstract**: Quantum computing has significantly advanced in recent years, boasting devices with hundreds of quantum bits (qubits), hinting at its potential quantum advantage over classical computing. Yet, noise in quantum devices poses significant barriers to realizing this supremacy. Understanding noise's impact is crucial for reproducibility and application reuse; moreover, the next-generation quantum-centric supercomputing essentially requires efficient and accurate noise characterization to support system management (e.g., job scheduling), where ensuring correct functional performance (i.e., fidelity) of jobs on available quantum devices can even be higher-priority than traditional objectives. However, noise fluctuates over time, even on the same quantum device, which makes predicting the computational bounds for on-the-fly noise is vital. Noisy quantum simulation can offer insights but faces efficiency and scalability issues. In this work, we propose a data-driven workflow, namely QuBound, to predict computational performance bounds. It decomposes historical performance traces to isolate noise sources and devises a novel encoder to embed circuit and noise information processed by a Long Short-Term Memory (LSTM) network. For evaluation, we compare QuBound with a state-of-the-art learning-based predictor, which only generates a single performance value instead of a bound. Experimental results show that the result of the existing approach falls outside of performance bounds, while all predictions from our QuBound with the assistance of performance decomposition better fit the bounds. Moreover, QuBound can efficiently produce practical bounds for various circuits with over 106 speedup over simulation; in addition, the range from QuBound is over 10x narrower than the state-of-the-art analytical approach. 

---
# StreamME: Simplify 3D Gaussian Avatar within Live Stream 

**Authors**: Luchuan Song, Yang Zhou, Zhan Xu, Yi Zhou, Deepali Aneja, Chenliang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2507.17029)  

**Abstract**: We propose StreamME, a method focuses on fast 3D avatar reconstruction. The StreamME synchronously records and reconstructs a head avatar from live video streams without any pre-cached data, enabling seamless integration of the reconstructed appearance into downstream applications. This exceptionally fast training strategy, which we refer to as on-the-fly training, is central to our approach. Our method is built upon 3D Gaussian Splatting (3DGS), eliminating the reliance on MLPs in deformable 3DGS and relying solely on geometry, which significantly improves the adaptation speed to facial expression. To further ensure high efficiency in on-the-fly training, we introduced a simplification strategy based on primary points, which distributes the point clouds more sparsely across the facial surface, optimizing points number while maintaining rendering quality. Leveraging the on-the-fly training capabilities, our method protects the facial privacy and reduces communication bandwidth in VR system or online conference. Additionally, it can be directly applied to downstream application such as animation, toonify, and relighting. Please refer to our project page for more details: this https URL. 

---
# Evolutionary Feature-wise Thresholding for Binary Representation of NLP Embeddings 

**Authors**: Soumen Sinha, Shahryar Rahnamayan, Azam Asilian Bidgoli  

**Link**: [PDF](https://arxiv.org/pdf/2507.17025)  

**Abstract**: Efficient text embedding is crucial for large-scale natural language processing (NLP) applications, where storage and computational efficiency are key concerns. In this paper, we explore how using binary representations (barcodes) instead of real-valued features can be used for NLP embeddings derived from machine learning models such as BERT. Thresholding is a common method for converting continuous embeddings into binary representations, often using a fixed threshold across all features. We propose a Coordinate Search-based optimization framework that instead identifies the optimal threshold for each feature, demonstrating that feature-specific thresholds lead to improved performance in binary encoding. This ensures that the binary representations are both accurate and efficient, enhancing performance across various features. Our optimal barcode representations have shown promising results in various NLP applications, demonstrating their potential to transform text representation. We conducted extensive experiments and statistical tests on different NLP tasks and datasets to evaluate our approach and compare it to other thresholding methods. Binary embeddings generated using using optimal thresholds found by our method outperform traditional binarization methods in accuracy. This technique for generating binary representations is versatile and can be applied to any features, not just limited to NLP embeddings, making it useful for a wide range of domains in machine learning applications. 

---
# Causal Graph Fuzzy LLMs: A First Introduction and Applications in Time Series Forecasting 

**Authors**: Omid Orang, Patricia O. Lucas, Gabriel I. F. Paiva, Petronio C. L. Silva, Felipe Augusto Rocha da Silva, Adriano Alonso Veloso, Frederico Gadelha Guimaraes  

**Link**: [PDF](https://arxiv.org/pdf/2507.17016)  

**Abstract**: In recent years, the application of Large Language Models (LLMs) to time series forecasting (TSF) has garnered significant attention among researchers. This study presents a new frame of LLMs named CGF-LLM using GPT-2 combined with fuzzy time series (FTS) and causal graph to predict multivariate time series, marking the first such architecture in the literature. The key objective is to convert numerical time series into interpretable forms through the parallel application of fuzzification and causal analysis, enabling both semantic understanding and structural insight as input for the pretrained GPT-2 model. The resulting textual representation offers a more interpretable view of the complex dynamics underlying the original time series. The reported results confirm the effectiveness of our proposed LLM-based time series forecasting model, as demonstrated across four different multivariate time series datasets. This initiative paves promising future directions in the domain of TSF using LLMs based on FTS. 

---
# Can External Validation Tools Improve Annotation Quality for LLM-as-a-Judge? 

**Authors**: Arduin Findeis, Floris Weers, Guoli Yin, Ke Ye, Ruoming Pang, Tom Gunter  

**Link**: [PDF](https://arxiv.org/pdf/2507.17015)  

**Abstract**: Pairwise preferences over model responses are widely collected to evaluate and provide feedback to large language models (LLMs). Given two alternative model responses to the same input, a human or AI annotator selects the "better" response. This approach can provide feedback for domains where other hard-coded metrics are difficult to obtain (e.g., chat response quality), thereby helping model evaluation or training. However, for some domains high-quality pairwise comparisons can be tricky to obtain - from AI and humans. For example, for responses with many factual statements, annotators may disproportionately weigh writing quality rather than underlying facts. In this work, we explore augmenting standard AI annotator systems with additional tools to improve performance on three challenging response domains: long-form factual, math and code tasks. We propose a tool-using agentic system to provide higher quality feedback on these domains. Our system uses web-search and code execution to ground itself based on external validation, independent of the LLM's internal knowledge and biases. We provide extensive experimental results evaluating our method across the three targeted response domains as well as general annotation tasks, using RewardBench (incl. AlpacaEval and LLMBar), RewardMath, as well as three new datasets for domains with saturated pre-existing datasets. Our results indicate that external tools can indeed improve performance in many, but not all, cases. More generally, our experiments highlight the sensitivity of performance to simple parameters (e.g., prompt) and the need for improved (non-saturated) annotator benchmarks. We share our code at this https URL. 

---
# laplax -- Laplace Approximations with JAX 

**Authors**: Tobias Weber, Bálint Mucsányi, Lenard Rommel, Thomas Christie, Lars Kasüschke, Marvin Pförtner, Philipp Hennig  

**Link**: [PDF](https://arxiv.org/pdf/2507.17013)  

**Abstract**: The Laplace approximation provides a scalable and efficient means of quantifying weight-space uncertainty in deep neural networks, enabling the application of Bayesian tools such as predictive uncertainty and model selection via Occam's razor. In this work, we introduce laplax, a new open-source Python package for performing Laplace approximations with jax. Designed with a modular and purely functional architecture and minimal external dependencies, laplax offers a flexible and researcher-friendly framework for rapid prototyping and experimentation. Its goal is to facilitate research on Bayesian neural networks, uncertainty quantification for deep learning, and the development of improved Laplace approximation techniques. 

---
# Towards Trustworthy AI: Secure Deepfake Detection using CNNs and Zero-Knowledge Proofs 

**Authors**: H M Mohaimanul Islam, Huynh Q. N. Vo, Aditya Rane  

**Link**: [PDF](https://arxiv.org/pdf/2507.17010)  

**Abstract**: In the era of synthetic media, deepfake manipulations pose a significant threat to information integrity. To address this challenge, we propose TrustDefender, a two-stage framework comprising (i) a lightweight convolutional neural network (CNN) that detects deepfake imagery in real-time extended reality (XR) streams, and (ii) an integrated succinct zero-knowledge proof (ZKP) protocol that validates detection results without disclosing raw user data. Our design addresses both the computational constraints of XR platforms while adhering to the stringent privacy requirements in sensitive settings. Experimental evaluations on multiple benchmark deepfake datasets demonstrate that TrustDefender achieves 95.3% detection accuracy, coupled with efficient proof generation underpinned by rigorous cryptography, ensuring seamless integration with high-performance artificial intelligence (AI) systems. By fusing advanced computer vision models with provable security mechanisms, our work establishes a foundation for reliable AI in immersive and privacy-sensitive applications. 

---
# Bringing Balance to Hand Shape Classification: Mitigating Data Imbalance Through Generative Models 

**Authors**: Gaston Gustavo Rios, Pedro Dal Bianco, Franco Ronchetti, Facundo Quiroga, Oscar Stanchi, Santiago Ponte Ahón, Waldo Hasperué  

**Link**: [PDF](https://arxiv.org/pdf/2507.17008)  

**Abstract**: Most sign language handshape datasets are severely limited and unbalanced, posing significant challenges to effective model training. In this paper, we explore the effectiveness of augmenting the training data of a handshape classifier by generating synthetic data. We use an EfficientNet classifier trained on the RWTH German sign language handshape dataset, which is small and heavily unbalanced, applying different strategies to combine generated and real images. We compare two Generative Adversarial Networks (GAN) architectures for data generation: ReACGAN, which uses label information to condition the data generation process through an auxiliary classifier, and SPADE, which utilizes spatially-adaptive normalization to condition the generation on pose information. ReACGAN allows for the generation of realistic images that align with specific handshape labels, while SPADE focuses on generating images with accurate spatial handshape configurations. Our proposed techniques improve the current state-of-the-art accuracy on the RWTH dataset by 5%, addressing the limitations of small and unbalanced datasets. Additionally, our method demonstrates the capability to generalize across different sign language datasets by leveraging pose-based generation trained on the extensive HaGRID dataset. We achieve comparable performance to single-source trained classifiers without the need for retraining the generator. 

---
# Bayesian preference elicitation for decision support in multiobjective optimization 

**Authors**: Felix Huber, Sebastian Rojas Gonzalez, Raul Astudillo  

**Link**: [PDF](https://arxiv.org/pdf/2507.16999)  

**Abstract**: We present a novel approach to help decision-makers efficiently identify preferred solutions from the Pareto set of a multi-objective optimization problem. Our method uses a Bayesian model to estimate the decision-maker's utility function based on pairwise comparisons. Aided by this model, a principled elicitation strategy selects queries interactively to balance exploration and exploitation, guiding the discovery of high-utility solutions. The approach is flexible: it can be used interactively or a posteriori after estimating the Pareto front through standard multi-objective optimization techniques. Additionally, at the end of the elicitation phase, it generates a reduced menu of high-quality solutions, simplifying the decision-making process. Through experiments on test problems with up to nine objectives, our method demonstrates superior performance in finding high-utility solutions with a small number of queries. We also provide an open-source implementation of our method to support its adoption by the broader community. 

---
# PyG 2.0: Scalable Learning on Real World Graphs 

**Authors**: Matthias Fey, Jinu Sunil, Akihiro Nitta, Rishi Puri, Manan Shah, Blaž Stojanovič, Ramona Bendias, Alexandria Barghi, Vid Kocijan, Zecheng Zhang, Xinwei He, Jan Eric Lenssen, Jure Leskovec  

**Link**: [PDF](https://arxiv.org/pdf/2507.16991)  

**Abstract**: PyG (PyTorch Geometric) has evolved significantly since its initial release, establishing itself as a leading framework for Graph Neural Networks. In this paper, we present Pyg 2.0 (and its subsequent minor versions), a comprehensive update that introduces substantial improvements in scalability and real-world application capabilities. We detail the framework's enhanced architecture, including support for heterogeneous and temporal graphs, scalable feature/graph stores, and various optimizations, enabling researchers and practitioners to tackle large-scale graph learning problems efficiently. Over the recent years, PyG has been supporting graph learning in a large variety of application areas, which we will summarize, while providing a deep dive into the important areas of relational deep learning and large language modeling. 

---
# Fast and Scalable Gene Embedding Search: A Comparative Study of FAISS and ScaNN 

**Authors**: Mohammad Saleh Refahi, Gavin Hearne, Harrison Muller, Kieran Lynch, Bahrad A. Sokhansanj, James R. Brown, Gail Rosen  

**Link**: [PDF](https://arxiv.org/pdf/2507.16978)  

**Abstract**: The exponential growth of DNA sequencing data has outpaced traditional heuristic-based methods, which struggle to scale effectively. Efficient computational approaches are urgently needed to support large-scale similarity search, a foundational task in bioinformatics for detecting homology, functional similarity, and novelty among genomic and proteomic sequences. Although tools like BLAST have been widely used and remain effective in many scenarios, they suffer from limitations such as high computational cost and poor performance on divergent sequences.
In this work, we explore embedding-based similarity search methods that learn latent representations capturing deeper structural and functional patterns beyond raw sequence alignment. We systematically evaluate two state-of-the-art vector search libraries, FAISS and ScaNN, on biologically meaningful gene embeddings. Unlike prior studies, our analysis focuses on bioinformatics-specific embeddings and benchmarks their utility for detecting novel sequences, including those from uncharacterized taxa or genes lacking known homologs. Our results highlight both computational advantages (in memory and runtime efficiency) and improved retrieval quality, offering a promising alternative to traditional alignment-heavy tools. 

---
# Leveraging Synthetic Data for Question Answering with Multilingual LLMs in the Agricultural Domain 

**Authors**: Rishemjit Kaur, Arshdeep Singh Bhankhar, Surangika Ranathunga, Jashanpreet Singh Salh, Sudhir Rajput, Vidhi, Kashish Mahendra, Bhavika Berwal, Ritesh Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2507.16974)  

**Abstract**: Enabling farmers to access accurate agriculture-related information in their native languages in a timely manner is crucial for the success of the agriculture field. Although large language models (LLMs) can be used to implement Question Answering (QA) systems, simply using publicly available general-purpose LLMs in agriculture typically offer generic advisories, lacking precision in local and multilingual contexts due to insufficient domain-specific training and scarcity of high-quality, region-specific datasets. Our study addresses these limitations by generating multilingual synthetic agricultural datasets (English, Hindi, Punjabi) from agriculture-specific documents and fine-tuning language-specific LLMs. Our evaluation on curated multilingual datasets demonstrates significant improvements in factual accuracy, relevance, and agricultural consensus for the fine-tuned models compared to their baseline counterparts. These results highlight the efficacy of synthetic data-driven, language-specific fine-tuning as an effective strategy to improve the performance of LLMs in agriculture, especially in multilingual and low-resource settings. By enabling more accurate and localized agricultural advisory services, this study provides a meaningful step toward bridging the knowledge gap in AI-driven agricultural solutions for diverse linguistic communities. 

---
# Text-to-SPARQL Goes Beyond English: Multilingual Question Answering Over Knowledge Graphs through Human-Inspired Reasoning 

**Authors**: Aleksandr Perevalov, Andreas Both  

**Link**: [PDF](https://arxiv.org/pdf/2507.16971)  

**Abstract**: Accessing knowledge via multilingual natural-language interfaces is one of the emerging challenges in the field of information retrieval and related ones. Structured knowledge stored in knowledge graphs can be queried via a specific query language (e.g., SPARQL). Therefore, one needs to transform natural-language input into a query to fulfill an information need. Prior approaches mostly focused on combining components (e.g., rule-based or neural-based) that solve downstream tasks and come up with an answer at the end. We introduce mKGQAgent, a human-inspired framework that breaks down the task of converting natural language questions into SPARQL queries into modular, interpretable subtasks. By leveraging a coordinated LLM agent workflow for planning, entity linking, and query refinement - guided by an experience pool for in-context learning - mKGQAgent efficiently handles multilingual KGQA. Evaluated on the DBpedia- and Corporate-based KGQA benchmarks within the Text2SPARQL challenge 2025, our approach took first place among the other participants. This work opens new avenues for developing human-like reasoning systems in multilingual semantic parsing. 

---
# Evaluating Ensemble and Deep Learning Models for Static Malware Detection with Dimensionality Reduction Using the EMBER Dataset 

**Authors**: Md Min-Ha-Zul Abedin, Tazqia Mehrub  

**Link**: [PDF](https://arxiv.org/pdf/2507.16952)  

**Abstract**: This study investigates the effectiveness of several machine learning algorithms for static malware detection using the EMBER dataset, which contains feature representations of Portable Executable (PE) files. We evaluate eight classification models: LightGBM, XGBoost, CatBoost, Random Forest, Extra Trees, HistGradientBoosting, k-Nearest Neighbors (KNN), and TabNet, under three preprocessing settings: original feature space, Principal Component Analysis (PCA), and Linear Discriminant Analysis (LDA). The models are assessed on accuracy, precision, recall, F1 score, and AUC to examine both predictive performance and robustness. Ensemble methods, especially LightGBM and XGBoost, show the best overall performance across all configurations, with minimal sensitivity to PCA and consistent generalization. LDA improves KNN performance but significantly reduces accuracy for boosting models. TabNet, while promising in theory, underperformed under feature reduction, likely due to architectural sensitivity to input structure. The analysis is supported by detailed exploratory data analysis (EDA), including mutual information ranking, PCA or t-SNE visualizations, and outlier detection using Isolation Forest and Local Outlier Factor (LOF), which confirm the discriminatory capacity of key features in the EMBER dataset. The results suggest that boosting models remain the most reliable choice for high-dimensional static malware detection, and that dimensionality reduction should be applied selectively based on model type. This work provides a benchmark for comparing classification models and preprocessing strategies in malware detection tasks and contributes insights that can guide future system development and real-world deployment. 

---
# SiLQ: Simple Large Language Model Quantization-Aware Training 

**Authors**: Steven K. Esser, Jeffrey L. McKinstry, Deepika Bablani, Rathinakumar Appuswamy, Dharmendra S. Modha  

**Link**: [PDF](https://arxiv.org/pdf/2507.16933)  

**Abstract**: Large language models can be quantized to reduce inference time latency, model size, and energy consumption, thereby delivering a better user experience at lower cost. A challenge exists to deliver quantized models with minimal loss of accuracy in reasonable time, and in particular to do so without requiring mechanisms incompatible with specialized inference accelerators. Here, we demonstrate a simple, end-to-end quantization-aware training approach that, with an increase in total model training budget of less than 0.1%, outperforms the leading published quantization methods by large margins on several modern benchmarks, with both base and instruct model variants. The approach easily generalizes across different model architectures, can be applied to activations, cache, and weights, and requires the introduction of no additional operations to the model other than the quantization itself. 

---
# Revisiting Pre-trained Language Models for Vulnerability Detection 

**Authors**: Youpeng Li, Weiliang Qi, Xuyu Wang, Fuxun Yu, Xinda Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.16887)  

**Abstract**: The rapid advancement of pre-trained language models (PLMs) has demonstrated promising results for various code-related tasks. However, their effectiveness in detecting real-world vulnerabilities remains a critical challenge. % for the security community. While existing empirical studies evaluate PLMs for vulnerability detection (VD), their inadequate consideration in data preparation, evaluation setups, and experimental settings undermines the accuracy and comprehensiveness of evaluations. This paper introduces RevisitVD, an extensive evaluation of 17 PLMs spanning smaller code-specific PLMs and large-scale PLMs using newly constructed datasets. Specifically, we compare the performance of PLMs under both fine-tuning and prompt engineering, assess their effectiveness and generalizability across various training and testing settings, and analyze their robustness against code normalization, abstraction, and semantic-preserving transformations.
Our findings reveal that, for VD tasks, PLMs incorporating pre-training tasks designed to capture the syntactic and semantic patterns of code outperform both general-purpose PLMs and those solely pre-trained or fine-tuned on large code corpora. However, these models face notable challenges in real-world scenarios, such as difficulties in detecting vulnerabilities with complex dependencies, handling perturbations introduced by code normalization and abstraction, and identifying semantic-preserving vulnerable code transformations. Also, the truncation caused by the limited context windows of PLMs can lead to a non-negligible amount of labeling errors. This study underscores the importance of thorough evaluations of model performance in practical scenarios and outlines future directions to help enhance the effectiveness of PLMs for realistic VD applications. 

---
# Sparser2Sparse: Single-shot Sparser-to-Sparse Learning for Spatial Transcriptomics Imputation with Natural Image Co-learning 

**Authors**: Yaoyu Fang, Jiahe Qian, Xinkun Wang, Lee A. Cooper, Bo Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.16886)  

**Abstract**: Spatial transcriptomics (ST) has revolutionized biomedical research by enabling high resolution gene expression profiling within tissues. However, the high cost and scarcity of high resolution ST data remain significant challenges. We present Single-shot Sparser-to-Sparse (S2S-ST), a novel framework for accurate ST imputation that requires only a single and low-cost sparsely sampled ST dataset alongside widely available natural images for co-training. Our approach integrates three key innovations: (1) a sparser-to-sparse self-supervised learning strategy that leverages intrinsic spatial patterns in ST data, (2) cross-domain co-learning with natural images to enhance feature representation, and (3) a Cascaded Data Consistent Imputation Network (CDCIN) that iteratively refines predictions while preserving sampled gene data fidelity. Extensive experiments on diverse tissue types, including breast cancer, liver, and lymphoid tissue, demonstrate that our method outperforms state-of-the-art approaches in imputation accuracy. By enabling robust ST reconstruction from sparse inputs, our framework significantly reduces reliance on costly high resolution data, facilitating potential broader adoption in biomedical research and clinical applications. 

---
# SplitMeanFlow: Interval Splitting Consistency in Few-Step Generative Modeling 

**Authors**: Yi Guo, Wei Wang, Zhihang Yuan, Rong Cao, Kuan Chen, Zhengyang Chen, Yuanyuan Huo, Yang Zhang, Yuping Wang, Shouda Liu, Yuxuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.16884)  

**Abstract**: Generative models like Flow Matching have achieved state-of-the-art performance but are often hindered by a computationally expensive iterative sampling process. To address this, recent work has focused on few-step or one-step generation by learning the average velocity field, which directly maps noise to data. MeanFlow, a leading method in this area, learns this field by enforcing a differential identity that connects the average and instantaneous velocities. In this work, we argue that this differential formulation is a limiting special case of a more fundamental principle. We return to the first principles of average velocity and leverage the additivity property of definite integrals. This leads us to derive a novel, purely algebraic identity we term Interval Splitting Consistency. This identity establishes a self-referential relationship for the average velocity field across different time intervals without resorting to any differential operators. Based on this principle, we introduce SplitMeanFlow, a new training framework that enforces this algebraic consistency directly as a learning objective. We formally prove that the differential identity at the core of MeanFlow is recovered by taking the limit of our algebraic consistency as the interval split becomes infinitesimal. This establishes SplitMeanFlow as a direct and more general foundation for learning average velocity fields. From a practical standpoint, our algebraic approach is significantly more efficient, as it eliminates the need for JVP computations, resulting in simpler implementation, more stable training, and broader hardware compatibility. One-step and two-step SplitMeanFlow models have been successfully deployed in large-scale speech synthesis products (such as Doubao), achieving speedups of 20x. 

---
# Confidence Optimization for Probabilistic Encoding 

**Authors**: Pengjiu Xia, Yidian Huang, Wenchao Wei, Yuwen Tan  

**Link**: [PDF](https://arxiv.org/pdf/2507.16881)  

**Abstract**: Probabilistic encoding introduces Gaussian noise into neural networks, enabling a smooth transition from deterministic to uncertain states and enhancing generalization ability. However, the randomness of Gaussian noise distorts point-based distance measurements in classification tasks. To mitigate this issue, we propose a confidence optimization probabilistic encoding (CPE) method that improves distance reliability and enhances representation learning. Specifically, we refine probabilistic encoding with two key strategies: First, we introduce a confidence-aware mechanism to adjust distance calculations, ensuring consistency and reliability in probabilistic encoding classification tasks. Second, we replace the conventional KL divergence-based variance regularization, which relies on unreliable prior assumptions, with a simpler L2 regularization term to directly constrain variance. The method we proposed is model-agnostic, and extensive experiments on natural language classification tasks demonstrate that our method significantly improves performance and generalization on both the BERT and the RoBERTa model. 

---
# Finding Dori: Memorization in Text-to-Image Diffusion Models Is Less Local Than Assumed 

**Authors**: Antoni Kowalczuk, Dominik Hintersdorf, Lukas Struppek, Kristian Kersting, Adam Dziedzic, Franziska Boenisch  

**Link**: [PDF](https://arxiv.org/pdf/2507.16880)  

**Abstract**: Text-to-image diffusion models (DMs) have achieved remarkable success in image generation. However, concerns about data privacy and intellectual property remain due to their potential to inadvertently memorize and replicate training data. Recent mitigation efforts have focused on identifying and pruning weights responsible for triggering replication, based on the assumption that memorization can be localized. Our research assesses the robustness of these pruning-based approaches. We demonstrate that even after pruning, minor adjustments to text embeddings of input prompts are sufficient to re-trigger data replication, highlighting the fragility of these defenses. Furthermore, we challenge the fundamental assumption of memorization locality, by showing that replication can be triggered from diverse locations within the text embedding space, and follows different paths in the model. Our findings indicate that existing mitigation strategies are insufficient and underscore the need for methods that truly remove memorized content, rather than attempting to suppress its retrieval. As a first step in this direction, we introduce a novel adversarial fine-tuning method that iteratively searches for replication triggers and updates the model to increase robustness. Through our research, we provide fresh insights into the nature of memorization in text-to-image DMs and a foundation for building more trustworthy and compliant generative AI. 

---
# CausalStep: A Benchmark for Explicit Stepwise Causal Reasoning in Videos 

**Authors**: Xuchen Li, Xuzhao Li, Shiyu Hu, Kaiqi Huang, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.16878)  

**Abstract**: Recent advances in large language models (LLMs) have improved reasoning in text and image domains, yet achieving robust video reasoning remains a significant challenge. Existing video benchmarks mainly assess shallow understanding and reasoning and allow models to exploit global context, failing to rigorously evaluate true causal and stepwise reasoning. We present CausalStep, a benchmark designed for explicit stepwise causal reasoning in videos. CausalStep segments videos into causally linked units and enforces a strict stepwise question-answer (QA) protocol, requiring sequential answers and preventing shortcut solutions. Each question includes carefully constructed distractors based on error type taxonomy to ensure diagnostic value. The benchmark features 100 videos across six categories and 1,852 multiple-choice QA pairs. We introduce seven diagnostic metrics for comprehensive evaluation, enabling precise diagnosis of causal reasoning capabilities. Experiments with leading proprietary and open-source models, as well as human baselines, reveal a significant gap between current models and human-level stepwise reasoning. CausalStep provides a rigorous benchmark to drive progress in robust and interpretable video reasoning. 

---
# ReMeREC: Relation-aware and Multi-entity Referring Expression Comprehension 

**Authors**: Yizhi Hu, Zezhao Tian, Xingqun Qi, Chen Su, Bingkun Yang, Junhui Yin, Muyi Sun, Man Zhang, Zhenan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2507.16877)  

**Abstract**: Referring Expression Comprehension (REC) aims to localize specified entities or regions in an image based on natural language descriptions. While existing methods handle single-entity localization, they often ignore complex inter-entity relationships in multi-entity scenes, limiting their accuracy and reliability. Additionally, the lack of high-quality datasets with fine-grained, paired image-text-relation annotations hinders further progress. To address this challenge, we first construct a relation-aware, multi-entity REC dataset called ReMeX, which includes detailed relationship and textual annotations. We then propose ReMeREC, a novel framework that jointly leverages visual and textual cues to localize multiple entities while modeling their inter-relations. To address the semantic ambiguity caused by implicit entity boundaries in language, we introduce the Text-adaptive Multi-entity Perceptron (TMP), which dynamically infers both the quantity and span of entities from fine-grained textual cues, producing distinctive representations. Additionally, our Entity Inter-relationship Reasoner (EIR) enhances relational reasoning and global scene understanding. To further improve language comprehension for fine-grained prompts, we also construct a small-scale auxiliary dataset, EntityText, generated using large language models. Experiments on four benchmark datasets show that ReMeREC achieves state-of-the-art performance in multi-entity grounding and relation prediction, outperforming existing approaches by a large margin. 

---
# Machine learning-based multimodal prognostic models integrating pathology images and high-throughput omic data for overall survival prediction in cancer: a systematic review 

**Authors**: Charlotte Jennings, Andrew Broad, Lucy Godson, Emily Clarke, David Westhead, Darren Treanor  

**Link**: [PDF](https://arxiv.org/pdf/2507.16876)  

**Abstract**: Multimodal machine learning integrating histopathology and molecular data shows promise for cancer prognostication. We systematically reviewed studies combining whole slide images (WSIs) and high-throughput omics to predict overall survival. Searches of EMBASE, PubMed, and Cochrane CENTRAL (12/08/2024), plus citation screening, identified eligible studies. Data extraction used CHARMS; bias was assessed with PROBAST+AI; synthesis followed SWiM and PRISMA 2020. Protocol: PROSPERO (CRD42024594745).
Forty-eight studies (all since 2017) across 19 cancer types met criteria; all used The Cancer Genome Atlas. Approaches included regularised Cox regression (n=4), classical ML (n=13), and deep learning (n=31). Reported c-indices ranged 0.550-0.857; multimodal models typically outperformed unimodal ones. However, all studies showed unclear/high bias, limited external validation, and little focus on clinical utility.
Multimodal WSI-omics survival prediction is a fast-growing field with promising results but needs improved methodological rigor, broader datasets, and clinical evaluation.
Funded by NPIC, Leeds Teaching Hospitals NHS Trust, UK (Project 104687), supported by UKRI Industrial Strategy Challenge Fund. 

---
# Budget Allocation Policies for Real-Time Multi-Agent Path Finding 

**Authors**: Raz Beck, Roni Stern  

**Link**: [PDF](https://arxiv.org/pdf/2507.16874)  

**Abstract**: Multi-Agent Pathfinding (MAPF) is the problem of finding paths for a set of agents such that each agent reaches its desired destination while avoiding collisions with the other agents. Many MAPF solvers are designed to run offline, that is, first generate paths for all agents and then execute them. Real-Time MAPF (RT-MAPF) embodies a realistic MAPF setup in which one cannot wait until a complete path for each agent has been found before they start to move. Instead, planning and execution are interleaved, where the agents must commit to a fixed number of steps in a constant amount of computation time, referred to as the planning budget. Existing solutions to RT-MAPF iteratively call windowed versions of MAPF algorithms in every planning period, without explicitly considering the size of the planning budget. We address this gap and explore different policies for allocating the planning budget in windowed versions of standard MAPF algorithms, namely Prioritized Planning (PrP) and MAPF-LNS2. Our exploration shows that the baseline approach in which all agents draw from a shared planning budget pool is ineffective in over-constrained situations. Instead, policies that distribute the planning budget over the agents are able to solve more problems with a smaller makespan. 

---
# HIPPO-Video: Simulating Watch Histories with Large Language Models for Personalized Video Highlighting 

**Authors**: Jeongeun Lee, Youngjae Yu, Dongha Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.16873)  

**Abstract**: The exponential growth of video content has made personalized video highlighting an essential task, as user preferences are highly variable and complex. Existing video datasets, however, often lack personalization, relying on isolated videos or simple text queries that fail to capture the intricacies of user behavior. In this work, we introduce HIPPO-Video, a novel dataset for personalized video highlighting, created using an LLM-based user simulator to generate realistic watch histories reflecting diverse user preferences. The dataset includes 2,040 (watch history, saliency score) pairs, covering 20,400 videos across 170 semantic categories. To validate our dataset, we propose HiPHer, a method that leverages these personalized watch histories to predict preference-conditioned segment-wise saliency scores. Through extensive experiments, we demonstrate that our method outperforms existing generic and query-based approaches, showcasing its potential for highly user-centric video highlighting in real-world scenarios. 

---
# CompLeak: Deep Learning Model Compression Exacerbates Privacy Leakage 

**Authors**: Na Li, Yansong Gao, Hongsheng Hu, Boyu Kuang, Anmin Fu  

**Link**: [PDF](https://arxiv.org/pdf/2507.16872)  

**Abstract**: Model compression is crucial for minimizing memory storage and accelerating inference in deep learning (DL) models, including recent foundation models like large language models (LLMs). Users can access different compressed model versions according to their resources and budget. However, while existing compression operations primarily focus on optimizing the trade-off between resource efficiency and model performance, the privacy risks introduced by compression remain overlooked and insufficiently understood.
In this work, through the lens of membership inference attack (MIA), we propose CompLeak, the first privacy risk evaluation framework examining three widely used compression configurations that are pruning, quantization, and weight clustering supported by the commercial model compression framework of Google's TensorFlow-Lite (TF-Lite) and Facebook's PyTorch Mobile. CompLeak has three variants, given available access to the number of compressed models and original model. CompLeakNR starts by adopting existing MIA methods to attack a single compressed model, and identifies that different compressed models influence members and non-members differently. When the original model and one compressed model are available, CompLeakSR leverages the compressed model as a reference to the original model and uncovers more privacy by combining meta information (e.g., confidence vector) from both models. When multiple compressed models are available with/without accessing the original model, CompLeakMR innovatively exploits privacy leakage info from multiple compressed versions to substantially signify the overall privacy leakage. We conduct extensive experiments on seven diverse model architectures (from ResNet to foundation models of BERT and GPT-2), and six image and textual benchmark datasets. 

---
# Diffusion-Modeled Reinforcement Learning for Carbon and Risk-Aware Microgrid Optimization 

**Authors**: Yunyi Zhao, Wei Zhang, Cheng Xiang, Hongyang Du, Dusit Niyato, Shuhua Gao  

**Link**: [PDF](https://arxiv.org/pdf/2507.16867)  

**Abstract**: This paper introduces DiffCarl, a diffusion-modeled carbon- and risk-aware reinforcement learning algorithm for intelligent operation of multi-microgrid systems. With the growing integration of renewables and increasing system complexity, microgrid communities face significant challenges in real-time energy scheduling and optimization under uncertainty. DiffCarl integrates a diffusion model into a deep reinforcement learning (DRL) framework to enable adaptive energy scheduling under uncertainty and explicitly account for carbon emissions and operational risk. By learning action distributions through a denoising generation process, DiffCarl enhances DRL policy expressiveness and enables carbon- and risk-aware scheduling in dynamic and uncertain microgrid environments. Extensive experimental studies demonstrate that it outperforms classic algorithms and state-of-the-art DRL solutions, with 2.3-30.1% lower operational cost. It also achieves 28.7% lower carbon emissions than those of its carbon-unaware variant and reduces performance variability. These results highlight DiffCarl as a practical and forward-looking solution. Its flexible design allows efficient adaptation to different system configurations and objectives to support real-world deployment in evolving energy systems. 

---
# Reinforcement Learning in hyperbolic space for multi-step reasoning 

**Authors**: Tao Xu, Dung-Yang Lee, Momiao Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2507.16864)  

**Abstract**: Multi-step reasoning is a fundamental challenge in artificial intelligence, with applications ranging from mathematical problem-solving to decision-making in dynamic environments. Reinforcement Learning (RL) has shown promise in enabling agents to perform multi-step reasoning by optimizing long-term rewards. However, conventional RL methods struggle with complex reasoning tasks due to issues such as credit assignment, high-dimensional state representations, and stability concerns. Recent advancements in Transformer architectures and hyperbolic geometry have provided novel solutions to these challenges. This paper introduces a new framework that integrates hyperbolic Transformers into RL for multi-step reasoning. The proposed approach leverages hyperbolic embeddings to model hierarchical structures effectively. We present theoretical insights, algorithmic details, and experimental results that include Frontier Math and nonlinear optimal control problems. Compared to RL with vanilla transformer, the hyperbolic RL largely improves accuracy by (32%~44%) on FrontierMath benchmark, (43%~45%) on nonlinear optimal control benchmark, while achieving impressive reduction in computational time by (16%~32%) on FrontierMath benchmark, (16%~17%) on nonlinear optimal control benchmark. Our work demonstrates the potential of hyperbolic Transformers in reinforcement learning, particularly for multi-step reasoning tasks that involve hierarchical structures. 

---
# Pixels, Patterns, but No Poetry: To See The World like Humans 

**Authors**: Hongcheng Gao, Zihao Huang, Lin Xu, Jingyi Tang, Xinhao Li, Yue Liu, Haoyang Li, Taihang Hu, Minhua Lin, Xinlong Yang, Ge Wu, Balong Bi, Hongyu Chen, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.16863)  

**Abstract**: Achieving human-like perception and reasoning in Multimodal Large Language Models (MLLMs) remains a central challenge in artificial intelligence. While recent research has primarily focused on enhancing reasoning capabilities in MLLMs, a fundamental question persists: Can Multimodal Large Language Models truly perceive the world as humans do? This paper shifts focus from reasoning to perception. Rather than constructing benchmarks specifically for reasoning, we introduce the Turing Eye Test (TET), a challenging perception-oriented benchmark comprising four diagnostic tasks that evaluate MLLMs' performance on synthetic images that humans process intuitively. Our findings reveal that state-of-the-art MLLMs exhibit catastrophic failures on our perceptual tasks trivial for humans. Both in-context learning and training on language backbone-effective for previous benchmarks-fail to improve performance on our tasks, while fine-tuning the vision tower enables rapid adaptation, suggesting that our benchmark poses challenges for vision tower generalization rather than for the knowledge and reasoning capabilities of the language backbone-a key gap between current MLLMs and human perception. We release a representative subset of TET tasks in this version, and will introduce more diverse tasks and methods to enhance visual generalization in future work. 

---
# Look Before You Fuse: 2D-Guided Cross-Modal Alignment for Robust 3D Detection 

**Authors**: Xiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.16861)  

**Abstract**: Integrating LiDAR and camera inputs into a unified Bird's-Eye-View (BEV) representation is crucial for enhancing 3D perception capabilities of autonomous vehicles. However, current methods are often affected by misalignment between camera and LiDAR features. This misalignment leads to inaccurate depth supervision in camera branch and erroneous fusion during cross-modal feature aggregation. The root cause of this misalignment lies in projection errors, stemming from minor extrinsic calibration inaccuracies and rolling shutter effect of LiDAR during vehicle motion. In this work, our key insight is that these projection errors are predominantly concentrated at object-background boundaries, which are readily identified by 2D detectors. Based on this, our main motivation is to utilize 2D object priors to pre-align cross-modal features before fusion. To address local misalignment, we propose Prior Guided Depth Calibration (PGDC), which leverages 2D priors to correct local misalignment and preserve correct cross-modal feature pairs. To resolve global misalignment, we introduce Discontinuity Aware Geometric Fusion (DAGF) to process calibrated results from PGDC, suppressing noise and explicitly enhancing sharp transitions at object-background boundaries. To effectively utilize these transition-aware depth representations, we incorporate Structural Guidance Depth Modulator (SGDM), using a gated attention mechanism to efficiently fuse aligned depth and image features. Our proposed method achieves state-of-the-art performance on nuScenes validation dataset, with its mAP and NDS reaching 71.5% and 73.6% respectively. 

---
# Leveraging multi-source and heterogeneous signals for fatigue detection 

**Authors**: Luobin Cui, Yanlai Wu, Tang Ying, Weikai Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.16859)  

**Abstract**: Fatigue detection plays a critical role in safety-critical applications such as aviation, mining, and long-haul transport. However, most existing methods rely on high-end sensors and controlled environments, limiting their applicability in real world settings. This paper formally defines a practical yet underexplored problem setting for real world fatigue detection, where systems operating with context-appropriate sensors aim to leverage knowledge from differently instrumented sources including those using impractical sensors deployed in controlled environments. To tackle this challenge, we propose a heterogeneous and multi-source fatigue detection framework that adaptively utilizes the available modalities in the target domain while benefiting from the diverse configurations present in source domains. Our experiments, conducted using a realistic field-deployed sensor setup and two publicly available datasets, demonstrate the practicality, robustness, and improved generalization of our approach, paving the practical way for effective fatigue monitoring in sensor-constrained scenarios. 

---
# SIA: Enhancing Safety via Intent Awareness for Vision-Language Models 

**Authors**: Youngjin Na, Sangheon Jeong, Youngwan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.16856)  

**Abstract**: As vision-language models (VLMs) are increasingly deployed in real-world applications, new safety risks arise from the subtle interplay between images and text. In particular, seemingly innocuous inputs can combine to reveal harmful intent, leading to unsafe model responses. Despite increasing attention to multimodal safety, previous approaches based on post hoc filtering or static refusal prompts struggle to detect such latent risks, especially when harmfulness emerges only from the combination of inputs. We propose SIA (Safety via Intent Awareness), a training-free prompt engineering framework that proactively detects and mitigates harmful intent in multimodal inputs. SIA employs a three-stage reasoning process: (1) visual abstraction via captioning, (2) intent inference through few-shot chain-of-thought prompting, and (3) intent-conditioned response refinement. Rather than relying on predefined rules or classifiers, SIA dynamically adapts to the implicit intent inferred from the image-text pair. Through extensive experiments on safety-critical benchmarks including SIUO, MM-SafetyBench, and HoliSafe, we demonstrate that SIA achieves substantial safety improvements, outperforming prior methods. Although SIA shows a minor reduction in general reasoning accuracy on MMStar, the corresponding safety gains highlight the value of intent-aware reasoning in aligning VLMs with human-centric values. 

---
# CLAMP: Contrastive Learning with Adaptive Multi-loss and Progressive Fusion for Multimodal Aspect-Based Sentiment Analysis 

**Authors**: Xiaoqiang He  

**Link**: [PDF](https://arxiv.org/pdf/2507.16854)  

**Abstract**: Multimodal aspect-based sentiment analysis(MABSA) seeks to identify aspect terms within paired image-text data and determine their fine grained sentiment polarities, representing a fundamental task for improving the effectiveness of applications such as product review systems and public opinion monitoring. Existing methods face challenges such as cross modal alignment noise and insufficient consistency in fine-grained representations. While global modality alignment methods often overlook the connection between aspect terms and their corresponding local visual regions, bridging the representation gap between text and images remains a challenge. To address these limitations, this paper introduces an end to end Contrastive Learning framework with Adaptive Multi-loss and Progressive Attention Fusion(CLAMP). The framework is composed of three novel modules: Progressive Attention Fusion network, Multi-task Contrastive Learning, and Adaptive Multi-loss Aggregation. The Progressive Attention Fusion network enhances fine-grained alignment between textual features and image regions via hierarchical, multi-stage cross modal interactions, effectively suppressing irrelevant visual noise. Secondly, multi-task contrastive learning combines global modal contrast and local granularity alignment to enhance cross modal representation consistency. Adaptive Multi-loss Aggregation employs a dynamic uncertainty based weighting mechanism to calibrate loss contributions according to each task's uncertainty, thereby mitigating gradient interference. Evaluation on standard public benchmarks demonstrates that CLAMP consistently outperforms the vast majority of existing state of the art methods. 

---
# SynthCTI: LLM-Driven Synthetic CTI Generation to enhance MITRE Technique Mapping 

**Authors**: Álvaro Ruiz-Ródenas, Jaime Pujante Sáez, Daniel García-Algora, Mario Rodríguez Béjar, Jorge Blasco, José Luis Hernández-Ramos  

**Link**: [PDF](https://arxiv.org/pdf/2507.16852)  

**Abstract**: Cyber Threat Intelligence (CTI) mining involves extracting structured insights from unstructured threat data, enabling organizations to understand and respond to evolving adversarial behavior. A key task in CTI mining is mapping threat descriptions to MITRE ATT\&CK techniques. However, this process is often performed manually, requiring expert knowledge and substantial effort. Automated approaches face two major challenges: the scarcity of high-quality labeled CTI data and class imbalance, where many techniques have very few examples. While domain-specific Large Language Models (LLMs) such as SecureBERT have shown improved performance, most recent work focuses on model architecture rather than addressing the data limitations. In this work, we present SynthCTI, a data augmentation framework designed to generate high-quality synthetic CTI sentences for underrepresented MITRE ATT\&CK techniques. Our method uses a clustering-based strategy to extract semantic context from training data and guide an LLM in producing synthetic CTI sentences that are lexically diverse and semantically faithful. We evaluate SynthCTI on two publicly available CTI datasets, CTI-to-MITRE and TRAM, using LLMs with different capacity. Incorporating synthetic data leads to consistent macro-F1 improvements: for example, ALBERT improves from 0.35 to 0.52 (a relative gain of 48.6\%), and SecureBERT reaches 0.6558 (up from 0.4412). Notably, smaller models augmented with SynthCTI outperform larger models trained without augmentation, demonstrating the value of data generation methods for building efficient and effective CTI classification systems. 

---
# Toward a Real-Time Framework for Accurate Monocular 3D Human Pose Estimation with Geometric Priors 

**Authors**: Mohamed Adjel  

**Link**: [PDF](https://arxiv.org/pdf/2507.16850)  

**Abstract**: Monocular 3D human pose estimation remains a challenging and ill-posed problem, particularly in real-time settings and unconstrained environments. While direct imageto-3D approaches require large annotated datasets and heavy models, 2D-to-3D lifting offers a more lightweight and flexible alternative-especially when enhanced with prior knowledge. In this work, we propose a framework that combines real-time 2D keypoint detection with geometry-aware 2D-to-3D lifting, explicitly leveraging known camera intrinsics and subject-specific anatomical priors. Our approach builds on recent advances in self-calibration and biomechanically-constrained inverse kinematics to generate large-scale, plausible 2D-3D training pairs from MoCap and synthetic datasets. We discuss how these ingredients can enable fast, personalized, and accurate 3D pose estimation from monocular images without requiring specialized hardware. This proposal aims to foster discussion on bridging data-driven learning and model-based priors to improve accuracy, interpretability, and deployability of 3D human motion capture on edge devices in the wild. 

---
# Post-Disaster Affected Area Segmentation with a Vision Transformer (ViT)-based EVAP Model using Sentinel-2 and Formosat-5 Imagery 

**Authors**: Yi-Shan Chu, Hsuan-Cheng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2507.16849)  

**Abstract**: We propose a vision transformer (ViT)-based deep learning framework to refine disaster-affected area segmentation from remote sensing imagery, aiming to support and enhance the Emergent Value Added Product (EVAP) developed by the Taiwan Space Agency (TASA). The process starts with a small set of manually annotated regions. We then apply principal component analysis (PCA)-based feature space analysis and construct a confidence index (CI) to expand these labels, producing a weakly supervised training set. These expanded labels are then used to train ViT-based encoder-decoder models with multi-band inputs from Sentinel-2 and Formosat-5 imagery. Our architecture supports multiple decoder variants and multi-stage loss strategies to improve performance under limited supervision. During the evaluation, model predictions are compared with higher-resolution EVAP output to assess spatial coherence and segmentation consistency. Case studies on the 2022 Poyang Lake drought and the 2023 Rhodes wildfire demonstrate that our framework improves the smoothness and reliability of segmentation results, offering a scalable approach for disaster mapping when accurate ground truth is unavailable. 

---
# Dynamic Simulation Framework for Disinformation Dissemination and Correction With Social Bots 

**Authors**: Boyu Qiao, Kun Li, Wei Zhou, Songlin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2507.16848)  

**Abstract**: In the human-bot symbiotic information ecosystem, social bots play key roles in spreading and correcting disinformation. Understanding their influence is essential for risk control and better governance. However, current studies often rely on simplistic user and network modeling, overlook the dynamic behavior of bots, and lack quantitative evaluation of correction strategies. To fill these gaps, we propose MADD, a Multi Agent based framework for Disinformation Dissemination. MADD constructs a more realistic propagation network by integrating the Barabasi Albert Model for scale free topology and the Stochastic Block Model for community structures, while designing node attributes based on real world user data. Furthermore, MADD incorporates both malicious and legitimate bots, with their controlled dynamic participation allows for quantitative analysis of correction strategies. We evaluate MADD using individual and group level metrics. We experimentally verify the real world consistency of MADD user attributes and network structure, and we simulate the dissemination of six disinformation topics, demonstrating the differential effects of fact based and narrative based correction strategies. 

---
# Weak Supervision Techniques towards Enhanced ASR Models in Industry-level CRM Systems 

**Authors**: Zhongsheng Wang, Sijie Wang, Jia Wang, Yung-I Liang, Yuxi Zhang, Jiamou Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.16843)  

**Abstract**: In the design of customer relationship management (CRM) systems, accurately identifying customer types and offering personalized services are key to enhancing customer satisfaction and loyalty. However, this process faces the challenge of discerning customer voices and intentions, and general pre-trained automatic speech recognition (ASR) models make it difficult to effectively address industry-specific speech recognition tasks. To address this issue, we innovatively proposed a solution for fine-tuning industry-specific ASR models, which significantly improved the performance of the fine-tuned ASR models in industry applications. Experimental results show that our method substantially improves the crucial auxiliary role of the ASR model in industry CRM systems, and this approach has also been adopted in actual industrial applications. 

---
# CASPER: Contrastive Approach for Smart Ponzi Scheme Detecter with More Negative Samples 

**Authors**: Weijia Yang, Tian Lan, Leyuan Liu, Wei Chen, Tianqing Zhu, Sheng Wen, Xiaosong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.16840)  

**Abstract**: The rapid evolution of digital currency trading, fueled by the integration of blockchain technology, has led to both innovation and the emergence of smart Ponzi schemes. A smart Ponzi scheme is a fraudulent investment operation in smart contract that uses funds from new investors to pay returns to earlier investors. Traditional Ponzi scheme detection methods based on deep learning typically rely on fully supervised models, which require large amounts of labeled data. However, such data is often scarce, hindering effective model training. To address this challenge, we propose a novel contrastive learning framework, CASPER (Contrastive Approach for Smart Ponzi detectER with more negative samples), designed to enhance smart Ponzi scheme detection in blockchain transactions. By leveraging contrastive learning techniques, CASPER can learn more effective representations of smart contract source code using unlabeled datasets, significantly reducing both operational costs and system complexity. We evaluate CASPER on the XBlock dataset, where it outperforms the baseline by 2.3% in F1 score when trained with 100% labeled data. More impressively, with only 25% labeled data, CASPER achieves an F1 score nearly 20% higher than the baseline under identical experimental conditions. These results highlight CASPER's potential for effective and cost-efficient detection of smart Ponzi schemes, paving the way for scalable fraud detection solutions in the future. 

---
# Segmentation-free Goodness of Pronunciation 

**Authors**: Xinwei Cao, Zijian Fan, Torbjørn Svendsen, Giampiero Salvi  

**Link**: [PDF](https://arxiv.org/pdf/2507.16838)  

**Abstract**: Mispronunciation detection and diagnosis (MDD) is a significant part in modern computer aided language learning (CALL) systems. Within MDD, phoneme-level pronunciation assessment is key to helping L2 learners improve their pronunciation. However, most systems are based on a form of goodness of pronunciation (GOP) which requires pre-segmentation of speech into phonetic units. This limits the accuracy of these methods and the possibility to use modern CTC-based acoustic models for their evaluation. In this study, we first propose self-alignment GOP (GOP-SA) that enables the use of CTC-trained ASR models for MDD. Next, we define a more general alignment-free method that takes all possible alignments of the target phoneme into account (GOP-AF). We give a theoretical account of our definition of GOP-AF, an implementation that solves potential numerical issues as well as a proper normalization which makes the method applicable with acoustic models with different peakiness over time. We provide extensive experimental results on the CMU Kids and Speechocean762 datasets comparing the different definitions of our methods, estimating the dependency of GOP-AF on the peakiness of the acoustic models and on the amount of context around the target phoneme. Finally, we compare our methods with recent studies over the Speechocean762 data showing that the feature vectors derived from the proposed method achieve state-of-the-art results on phoneme-level pronunciation assessment. 

---
# Towards Robust Speech Recognition for Jamaican Patois Music Transcription 

**Authors**: Jordan Madden, Matthew Stone, Dimitri Johnson, Daniel Geddez  

**Link**: [PDF](https://arxiv.org/pdf/2507.16834)  

**Abstract**: Although Jamaican Patois is a widely spoken language, current speech recognition systems perform poorly on Patois music, producing inaccurate captions that limit accessibility and hinder downstream applications. In this work, we take a data-centric approach to this problem by curating more than 40 hours of manually transcribed Patois music. We use this dataset to fine-tune state-of-the-art automatic speech recognition (ASR) models, and use the results to develop scaling laws for the performance of Whisper models on Jamaican Patois audio. We hope that this work will have a positive impact on the accessibility of Jamaican Patois music and the future of Jamaican Patois language modeling. 

---
# You Don't Bring Me Flowers: Mitigating Unwanted Recommendations Through Conformal Risk Control 

**Authors**: Giovanni De Toni, Erasmo Purificato, Emilia Gómez, Bruno Lepri, Andrea Passerini, Cristian Consonni  

**Link**: [PDF](https://arxiv.org/pdf/2507.16829)  

**Abstract**: Recommenders are significantly shaping online information consumption. While effective at personalizing content, these systems increasingly face criticism for propagating irrelevant, unwanted, and even harmful recommendations. Such content degrades user satisfaction and contributes to significant societal issues, including misinformation, radicalization, and erosion of user trust. Although platforms offer mechanisms to mitigate exposure to undesired content, these mechanisms are often insufficiently effective and slow to adapt to users' feedback. This paper introduces an intuitive, model-agnostic, and distribution-free method that uses conformal risk control to provably bound unwanted content in personalized recommendations by leveraging simple binary feedback on items. We also address a limitation of traditional conformal risk control approaches, i.e., the fact that the recommender can provide a smaller set of recommended items, by leveraging implicit feedback on consumed items to expand the recommendation set while ensuring robust risk mitigation. Our experimental evaluation on data coming from a popular online video-sharing platform demonstrates that our approach ensures an effective and controllable reduction of unwanted recommendations with minimal effort. The source code is available here: this https URL. 

---
# A Query-Aware Multi-Path Knowledge Graph Fusion Approach for Enhancing Retrieval-Augmented Generation in Large Language Models 

**Authors**: Qikai Wei, Huansheng Ning, Chunlong Han, Jianguo Ding  

**Link**: [PDF](https://arxiv.org/pdf/2507.16826)  

**Abstract**: Retrieval Augmented Generation (RAG) has gradually emerged as a promising paradigm for enhancing the accuracy and factual consistency of content generated by large language models (LLMs). However, existing RAG studies primarily focus on retrieving isolated segments using similarity-based matching methods, while overlooking the intrinsic connections between them. This limitation hampers performance in RAG tasks. To address this, we propose QMKGF, a Query-Aware Multi-Path Knowledge Graph Fusion Approach for Enhancing Retrieval Augmented Generation. First, we design prompt templates and employ general-purpose LLMs to extract entities and relations, thereby generating a knowledge graph (KG) efficiently. Based on the constructed KG, we introduce a multi-path subgraph construction strategy that incorporates one-hop relations, multi-hop relations, and importance-based relations, aiming to improve the semantic relevance between the retrieved documents and the user query. Subsequently, we designed a query-aware attention reward model that scores subgraph triples based on their semantic relevance to the query. Then, we select the highest score subgraph and enrich subgraph with additional triples from other subgraphs that are highly semantically relevant to the query. Finally, the entities, relations, and triples within the updated subgraph are utilised to expand the original query, thereby enhancing its semantic representation and improving the quality of LLMs' generation. We evaluate QMKGF on the SQuAD, IIRC, Culture, HotpotQA, and MuSiQue datasets. On the HotpotQA dataset, our method achieves a ROUGE-1 score of 64.98\%, surpassing the BGE-Rerank approach by 9.72 percentage points (from 55.26\% to 64.98\%). Experimental results demonstrate the effectiveness and superiority of the QMKGF approach. 

---
# Disaster Informatics after the COVID-19 Pandemic: Bibliometric and Topic Analysis based on Large-scale Academic Literature 

**Authors**: Ngan Tran, Haihua Chen, Ana Cleveland, Yuhan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.16820)  

**Abstract**: This study presents a comprehensive bibliometric and topic analysis of the disaster informatics literature published between January 2020 to September 2022. Leveraging a large-scale corpus and advanced techniques such as pre-trained language models and generative AI, we identify the most active countries, institutions, authors, collaboration networks, emergent topics, patterns among the most significant topics, and shifts in research priorities spurred by the COVID-19 pandemic. Our findings highlight (1) countries that were most impacted by the COVID-19 pandemic were also among the most active, with each country having specific research interests, (2) countries and institutions within the same region or share a common language tend to collaborate, (3) top active authors tend to form close partnerships with one or two key partners, (4) authors typically specialized in one or two specific topics, while institutions had more diverse interests across several topics, and (5) the COVID-19 pandemic has influenced research priorities in disaster informatics, placing greater emphasis on public health. We further demonstrate that the field is converging on multidimensional resilience strategies and cross-sectoral data-sharing collaborations or projects, reflecting a heightened awareness of global vulnerability and interdependency. Collecting and quality assurance strategies, data analytic practices, LLM-based topic extraction and summarization approaches, and result visualization tools can be applied to comparable datasets or solve similar analytic problems. By mapping out the trends in disaster informatics, our analysis offers strategic insights for policymakers, practitioners, and scholars aiming to enhance disaster informatics capacities in an increasingly uncertain and complex risk landscape. 

---
# Explainable Vulnerability Detection in C/C++ Using Edge-Aware Graph Attention Networks 

**Authors**: Radowanul Haque, Aftab Ali, Sally McClean, Naveed Khan  

**Link**: [PDF](https://arxiv.org/pdf/2507.16540)  

**Abstract**: Detecting security vulnerabilities in source code remains challenging, particularly due to class imbalance in real-world datasets where vulnerable functions are under-represented. Existing learning-based methods often optimise for recall, leading to high false positive rates and reduced usability in development workflows. Furthermore, many approaches lack explainability, limiting their integration into security workflows. This paper presents ExplainVulD, a graph-based framework for vulnerability detection in C/C++ code. The method constructs Code Property Graphs and represents nodes using dual-channel embeddings that capture both semantic and structural information. These are processed by an edge-aware attention mechanism that incorporates edge-type embeddings to distinguish among program relations. To address class imbalance, the model is trained using class-weighted cross-entropy loss. ExplainVulD achieves a mean accuracy of 88.25 percent and an F1 score of 48.23 percent across 30 independent runs on the ReVeal dataset. These results represent relative improvements of 4.6 percent in accuracy and 16.9 percent in F1 score compared to the ReVeal model, a prior learning-based method. The framework also outperforms static analysis tools, with relative gains of 14.0 to 14.1 percent in accuracy and 132.2 to 201.2 percent in F1 score. Beyond improved detection performance, ExplainVulD produces explainable outputs by identifying the most influential code regions within each function, supporting transparency and trust in security triage. 

---
# Bridging Robustness and Generalization Against Word Substitution Attacks in NLP via the Growth Bound Matrix Approach 

**Authors**: Mohammed Bouri, Adnane Saoud  

**Link**: [PDF](https://arxiv.org/pdf/2507.10330)  

**Abstract**: Despite advancements in Natural Language Processing (NLP), models remain vulnerable to adversarial attacks, such as synonym substitutions. While prior work has focused on improving robustness for feed-forward and convolutional architectures, the robustness of recurrent networks and modern state space models (SSMs), such as S4, remains understudied. These architectures pose unique challenges due to their sequential processing and complex parameter dynamics. In this paper, we introduce a novel regularization technique based on Growth Bound Matrices (GBM) to improve NLP model robustness by reducing the impact of input perturbations on model outputs. We focus on computing the GBM for three architectures: Long Short-Term Memory (LSTM), State Space models (S4), and Convolutional Neural Networks (CNN). Our method aims to (1) enhance resilience against word substitution attacks, (2) improve generalization on clean text, and (3) providing the first systematic analysis of SSM (S4) robustness. Extensive experiments across multiple architectures and benchmark datasets demonstrate that our method improves adversarial robustness by up to 8.8% over existing baselines. These results highlight the effectiveness of our approach, outperforming several state-of-the-art methods in adversarial defense. Codes are available at this https URL 

---
