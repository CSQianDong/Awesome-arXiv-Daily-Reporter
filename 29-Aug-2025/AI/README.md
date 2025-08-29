# ChatThero: An LLM-Supported Chatbot for Behavior Change and Therapeutic Support in Addiction Recovery 

**Authors**: Junda Wang, Zonghai Yao, Zhichao Yang, Lingxi Li, Junhui Qian, Hong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.20996)  

**Abstract**: Substance use disorders (SUDs) affect over 36 million people worldwide, yet few receive effective care due to stigma, motivational barriers, and limited personalized support. Although large language models (LLMs) show promise for mental-health assistance, most systems lack tight integration with clinically validated strategies, reducing effectiveness in addiction recovery. We present ChatThero, a multi-agent conversational framework that couples dynamic patient modeling with context-sensitive therapeutic dialogue and adaptive persuasive strategies grounded in cognitive behavioral therapy (CBT) and motivational interviewing (MI). We build a high-fidelity synthetic benchmark spanning Easy, Medium, and Hard resistance levels, and train ChatThero with a two-stage pipeline comprising supervised fine-tuning (SFT) followed by direct preference optimization (DPO). In evaluation, ChatThero yields a 41.5\% average gain in patient motivation, a 0.49\% increase in treatment confidence, and resolves hard cases with 26\% fewer turns than GPT-4o, and both automated and human clinical assessments rate it higher in empathy, responsiveness, and behavioral realism. The framework supports rigorous, privacy-preserving study of therapeutic conversation and provides a robust, replicable basis for research and clinical translation. 

---
# Efficient Neuro-Symbolic Learning of Constraints and Objective 

**Authors**: Marianne Defresne, Romain Gambardella, Sophie Barbe, Thomas Schiex  

**Link**: [PDF](https://arxiv.org/pdf/2508.20978)  

**Abstract**: In the ongoing quest for hybridizing discrete reasoning with neural nets, there is an increasing interest in neural architectures that can learn how to solve discrete reasoning or optimization problems from natural inputs, a task that Large Language Models seem to struggle with.
Objectives: We introduce a differentiable neuro-symbolic architecture and a loss function dedicated to learning how to solve NP-hard reasoning problems.
Methods: Our new probabilistic loss allows for learning both the constraints and the objective, thus delivering a complete model that can be scrutinized and completed with side constraints. By pushing the combinatorial solver out of the training loop, our architecture also offers scalable training while exact inference gives access to maximum accuracy.
Results: We empirically show that it can efficiently learn how to solve NP-hard reasoning problems from natural inputs. On three variants of the Sudoku benchmark -- symbolic, visual, and many-solution --, our approach requires a fraction of training time of other hybrid methods. On a visual Min-Cut/Max-cut task, it optimizes the regret better than a Decision-Focused-Learning regret-dedicated loss. Finally, it efficiently learns the energy optimization formulation of the large real-world problem of designing proteins. 

---
# A Multi-Objective Genetic Algorithm for Healthcare Workforce Scheduling 

**Authors**: Vipul Patel, Anirudh Deodhar, Dagnachew Birru  

**Link**: [PDF](https://arxiv.org/pdf/2508.20953)  

**Abstract**: Workforce scheduling in the healthcare sector is a significant operational challenge, characterized by fluctuating patient loads, diverse clinical skills, and the critical need to control labor costs while upholding high standards of patient care. This problem is inherently multi-objective, demanding a delicate balance between competing goals: minimizing payroll, ensuring adequate staffing for patient needs, and accommodating staff preferences to mitigate burnout. We propose a Multi-objective Genetic Algorithm (MOO-GA) that models the hospital unit workforce scheduling problem as a multi-objective optimization task. Our model incorporates real-world complexities, including hourly appointment-driven demand and the use of modular shifts for a multi-skilled workforce. By defining objective functions for cost, patient care coverage, and staff satisfaction, the GA navigates the vast search space to identify a set of high-quality, non-dominated solutions. Demonstrated on datasets representing a typical hospital unit, the results show that our MOO-GA generates robust and balanced schedules. On average, the schedules produced by our algorithm showed a 66\% performance improvement over a baseline that simulates a conventional, manual scheduling process. This approach effectively manages trade-offs between critical operational and staff-centric objectives, providing a practical decision support tool for nurse managers and hospital administrators. 

---
# A Graph-Based Test-Harness for LLM Evaluation 

**Authors**: Jessica Lundin, Guillaume Chabot-Couture  

**Link**: [PDF](https://arxiv.org/pdf/2508.20810)  

**Abstract**: We present a first known prototype of a dynamic, systematic benchmark of medical guidelines for 400+ questions, with 3.3+ trillion possible combinations, covering 100\% of guideline relationships. We transformed the WHO IMCI handbook into a directed graph with 200+ nodes (conditions, symptoms, treatments, follow-ups, severities) and 300+ edges, then used graph traversal to generate questions that incorporated age-specific scenarios and contextual distractors to ensure clinical relevance. Our graph-based approach enables systematic evaluation across clinical tasks (45-67\% accuracy), and we find models excel at symptom recognition but struggle with triaging severity, treatment protocols and follow-up care, demonstrating how customized benchmarks can identify specific capability gaps that general-domain evaluations miss. Beyond evaluation, this dynamic MCQA methodology enhances LLM post-training (supervised finetuning, GRPO, DPO), where correct answers provide high-reward samples without expensive human annotation. The graph-based approach successfully addresses the coverage limitations of manually curated benchmarks. This methodology is a step toward scalable, contamination-resistant solution for creating comprehensive benchmarks that can be dynamically generated, including when the guidelines are updated. Code and datasets are available at this https URL 

---
# Single Agent Robust Deep Reinforcement Learning for Bus Fleet Control 

**Authors**: Yifan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.20784)  

**Abstract**: Bus bunching remains a challenge for urban transit due to stochastic traffic and passenger demand. Traditional solutions rely on multi-agent reinforcement learning (MARL) in loop-line settings, which overlook realistic operations characterized by heterogeneous routes, timetables, fluctuating demand, and varying fleet sizes. We propose a novel single-agent reinforcement learning (RL) framework for bus holding control that avoids the data imbalance and convergence issues of MARL under near-realistic simulation. A bidirectional timetabled network with dynamic passenger demand is constructed. The key innovation is reformulating the multi-agent problem into a single-agent one by augmenting the state space with categorical identifiers (vehicle ID, station ID, time period) in addition to numerical features (headway, occupancy, velocity). This high-dimensional encoding enables single-agent policies to capture inter-agent dependencies, analogous to projecting non-separable inputs into a higher-dimensional space. We further design a structured reward function aligned with operational goals: instead of exponential penalties on headway deviations, a ridge-shaped reward balances uniform headways and schedule adherence. Experiments show that our modified soft actor-critic (SAC) achieves more stable and superior performance than benchmarks, including MADDPG (e.g., -430k vs. -530k under stochastic conditions). These results demonstrate that single-agent deep RL, when enhanced with categorical structuring and schedule-aware rewards, can effectively manage bus holding in non-loop, real-world contexts. This paradigm offers a robust, scalable alternative to MARL frameworks, particularly where agent-specific experiences are imbalanced. 

---
# Re4: Scientific Computing Agent with Rewriting, Resolution, Review and Revision 

**Authors**: Ao Cheng, Lei Zhang, Guowei He  

**Link**: [PDF](https://arxiv.org/pdf/2508.20729)  

**Abstract**: Large language models (LLMs) serve as an active and promising field of generative artificial intelligence and have demonstrated abilities to perform complex tasks in multiple domains, including mathematical and scientific reasoning. In this work, we construct a novel agent framework for solving representative problems in scientific computing. The proposed agent, incorporating a "rewriting-resolution-review-revision" logical chain via three reasoning LLMs (functioning as the Consultant, Reviewer, and Programmer, respectively), is integrated in a collaborative and interactive manner. The Consultant module endows the agent with knowledge transfer capabilities to link problems to professional domain insights, thereby rewriting problem descriptions through text augmentation. The Programmer module is responsible for generating and executing well-structured code to deliver the problem resolution. The Reviewer module equips the agent with the capacity for self-debugging and self-refinement through interactive feedback with code runtime outputs. By leveraging the end-to-end review mechanism, the executable code provided by the Programmer attains the iterative revision. A comprehensive evaluation is conducted on the performance of the proposed agent framework in solving PDEs, ill-conditioned linear systems, and data-driven physical analysis problems. Compared to single-model, this collaborative framework significantly improves the bug-free code generation rate and reduces the occurrence of non-physical solutions, thereby establishing a highly reliable framework for autonomous code generation based on natural language descriptions. The review mechanism improved the average execution success (bug-free code and non-NaN solutions) rate of the latest reasoning models. In summary, our agent framework establishes automatic code generation and review as a promising scientific computing paradigm. 

---
# Transparent Semantic Spaces: A Categorical Approach to Explainable Word Embeddings 

**Authors**: Ares Fabregat-Hernández, Javier Palanca, Vicent Botti  

**Link**: [PDF](https://arxiv.org/pdf/2508.20701)  

**Abstract**: The paper introduces a novel framework based on category theory to enhance the explainability of artificial intelligence systems, particularly focusing on word embeddings. Key topics include the construction of categories $ Ł_{T} $ and $ ¶_{T} $, providing schematic representations of the semantics of a text $ T $, and reframing the selection of the element with maximum probability as a categorical notion. Additionally, the monoidal category $ ¶_{T} $ is constructed to visualize various methods of extracting semantic information from $ T $, offering a dimension-agnostic definition of semantic spaces reliant solely on information within the text.
Furthermore, the paper defines the categories of configurations $ \Conf $ and word embeddings $ \Emb $, accompanied by the concept of divergence as a decoration on $ \Emb $. It establishes a mathematically precise method for comparing word embeddings, demonstrating the equivalence between the GloVe and Word2Vec algorithms and the metric MDS algorithm, transitioning from neural network algorithms (black box) to a transparent framework. Finally, the paper presents a mathematical approach to computing biases before embedding and offers insights on mitigating biases at the semantic space level, advancing the field of explainable artificial intelligence. 

---
# Bridging Minds and Machines: Toward an Integration of AI and Cognitive Science 

**Authors**: Rui Mao, Qian Liu, Xiao Li, Erik Cambria, Amir Hussain  

**Link**: [PDF](https://arxiv.org/pdf/2508.20674)  

**Abstract**: Cognitive Science has profoundly shaped disciplines such as Artificial Intelligence (AI), Philosophy, Psychology, Neuroscience, Linguistics, and Culture. Many breakthroughs in AI trace their roots to cognitive theories, while AI itself has become an indispensable tool for advancing cognitive research. This reciprocal relationship motivates a comprehensive review of the intersections between AI and Cognitive Science. By synthesizing key contributions from both perspectives, we observe that AI progress has largely emphasized practical task performance, whereas its cognitive foundations remain conceptually fragmented. We argue that the future of AI within Cognitive Science lies not only in improving performance but also in constructing systems that deepen our understanding of the human mind. Promising directions include aligning AI behaviors with cognitive frameworks, situating AI in embodiment and culture, developing personalized cognitive models, and rethinking AI ethics through cognitive co-evaluation. 

---
# Human-AI Collaborative Bot Detection in MMORPGs 

**Authors**: Jaeman Son, Hyunsoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.20578)  

**Abstract**: In Massively Multiplayer Online Role-Playing Games (MMORPGs), auto-leveling bots exploit automated programs to level up characters at scale, undermining gameplay balance and fairness. Detecting such bots is challenging, not only because they mimic human behavior, but also because punitive actions require explainable justification to avoid legal and user experience issues. In this paper, we present a novel framework for detecting auto-leveling bots by leveraging contrastive representation learning and clustering techniques in a fully unsupervised manner to identify groups of characters with similar level-up patterns. To ensure reliable decisions, we incorporate a Large Language Model (LLM) as an auxiliary reviewer to validate the clustered groups, effectively mimicking a secondary human judgment. We also introduce a growth curve-based visualization to assist both the LLM and human moderators in assessing leveling behavior. This collaborative approach improves the efficiency of bot detection workflows while maintaining explainability, thereby supporting scalable and accountable bot regulation in MMORPGs. 

---
# Enhancing Health Fact-Checking with LLM-Generated Synthetic Data 

**Authors**: Jingze Zhang, Jiahe Qian, Yiliang Zhou, Yifan Peng  

**Link**: [PDF](https://arxiv.org/pdf/2508.20525)  

**Abstract**: Fact-checking for health-related content is challenging due to the limited availability of annotated training data. In this study, we propose a synthetic data generation pipeline that leverages large language models (LLMs) to augment training data for health-related fact checking. In this pipeline, we summarize source documents, decompose the summaries into atomic facts, and use an LLM to construct sentence-fact entailment tables. From the entailment relations in the table, we further generate synthetic text-claim pairs with binary veracity labels. These synthetic data are then combined with the original data to fine-tune a BERT-based fact-checking model. Evaluation on two public datasets, PubHealth and SciFact, shows that our pipeline improved F1 scores by up to 0.019 and 0.049, respectively, compared to models trained only on the original data. These results highlight the effectiveness of LLM-driven synthetic data augmentation in enhancing the performance of health-related fact-checkers. 

---
# Governable AI: Provable Safety Under Extreme Threat Models 

**Authors**: Donglin Wang, Weiyun Liang, Chunyuan Chen, Jing Xu, Yulong Fu  

**Link**: [PDF](https://arxiv.org/pdf/2508.20411)  

**Abstract**: As AI rapidly advances, the security risks posed by AI are becoming increasingly severe, especially in critical scenarios, including those posing existential risks. If AI becomes uncontrollable, manipulated, or actively evades safety mechanisms, it could trigger systemic disasters. Existing AI safety approaches-such as model enhancement, value alignment, and human intervention-suffer from fundamental, in-principle limitations when facing AI with extreme motivations and unlimited intelligence, and cannot guarantee security. To address this challenge, we propose a Governable AI (GAI) framework that shifts from traditional internal constraints to externally enforced structural compliance based on cryptographic mechanisms that are computationally infeasible to break, even for future AI, under the defined threat model and well-established cryptographic this http URL GAI framework is composed of a simple yet reliable, fully deterministic, powerful, flexible, and general-purpose rule enforcement module (REM); governance rules; and a governable secure super-platform (GSSP) that offers end-to-end protection against compromise or subversion by AI. The decoupling of the governance rules and the technical platform further enables a feasible and generalizable technical pathway for the safety governance of AI. REM enforces the bottom line defined by governance rules, while GSSP ensures non-bypassability, tamper-resistance, and unforgeability to eliminate all identified attack vectors. This paper also presents a rigorous formal proof of the security properties of this mechanism and demonstrates its effectiveness through a prototype implementation evaluated in representative high-stakes scenarios. 

---
# AWorld: Orchestrating the Training Recipe for Agentic AI 

**Authors**: Chengyue Yu, Siyuan Lu, Chenyi Zhuang, Dong Wang, Qintong Wu, Zongyue Li, Runsheng Gan, Chunfeng Wang, Siqi Hou, Gaochi Huang, Wenlong Yan, Lifeng Hong, Aohui Xue, Yanfeng Wang, Jinjie Gu, David Tsai, Tao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2508.20404)  

**Abstract**: The learning from practice paradigm is crucial for developing capable Agentic AI systems, yet it is severely hampered by inefficient experience generation, a bottleneck especially pronounced in complex benchmarks like GAIA. To address this, we introduce AWorld, an open-source system engineered for large-scale agent-environment interaction. By distributing tasks across a cluster, AWorld accelerates experience collection by 14.6x compared to standard single-node, sequential execution. This critical speedup makes extensive reinforcement learning practical and scalable. Leveraging this capability, we trained a Qwen3-32B-based agent that significantly outperforms its base model, increasing its overall GAIA accuracy from 21.59% to 32.23%. On the benchmark's most challenging levels, our agent achieves a score of 16.33%, surpassing the performance of leading proprietary models. Our open-source system and resulting agent provide a practical blueprint for a complete agentic AI training pipeline, from efficient interaction to demonstrable model improvement. 

---
# Uncertainty Under the Curve: A Sequence-Level Entropy Area Metric for Reasoning LLM 

**Authors**: Yongfu Zhu, Lin Sun, Guangxiang Zhao, Weihong Lin, Xiangzheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.20384)  

**Abstract**: In this work, we introduce Entropy Area Score (EAS), a simple yet effective metric to quantify uncertainty in the answer generation process of reasoning large language models (LLMs). EAS requires neither external models nor repeated sampling, it integrates token-level predictive entropy from the model itself to capture the evolution of uncertainty during generation. Empirical results show that EAS is strongly correlated with answer entropy across models and datasets. In training data selection, EAS identifies high-potential samples and consistently outperforms Pass Rate filtering under equal sample budgets, improving student model accuracy on math benchmarks. EAS is both efficient and interpretable, offering a practical tool for uncertainty modeling and data quality assessment in LLM training. 

---
# TCIA: A Task-Centric Instruction Augmentation Method for Instruction Finetuning 

**Authors**: Simin Ma, Shujian Liu, Jun Tan, Yebowen Hu, Song Wang, Sathish Reddy Indurthi, Sanqiang Zhao, Liwei Wu, Jianbing Han, Kaiqiang Song  

**Link**: [PDF](https://arxiv.org/pdf/2508.20374)  

**Abstract**: Diverse instruction data is vital for effective instruction tuning of large language models, as it enables the model to generalize across different types of inputs . Building such diversified instruction dataset is an essential step in this process. Existing approaches often leverage large language models to automatically explore and generate diverse instructions, ensuring both data diversity and quality. However, they tend to overlook an important factor in real-world applications: on-task relevance. In practice, only a few real-world applications require a truly general-purpose model; most benefit from task-specific knowledge tailored to their particular use case. Therefore, it is vital to develop instruction augmentation methods that not only maintain diversity but are also optimized for specific, real-world scenarios.
We thus introduce Task Centric Instruction Augmentation (TCIA), a framework that systematically expands instructions while preserving both diversity and task alignment. By representing instructions in a discrete query-constraints space, TCIA creates a rich set of task-relevant instructions and enables models to generalize to these task-specific instructions without sacrificing overall performance. Experiments show that TCIA improves open-source LLMs' performance by an average of 8.7% across four real-world, task-specific applications, and in some cases outperforming leading closed-source models. These improvements do not compromise general instruction-following ability, making TCIA a scalable and efficient solution for adapting LLMs to real-world, task-focused applications. 

---
# P2C: Path to Counterfactuals 

**Authors**: Sopam Dasgupta, Sadaf MD Halim, Joaquín Arias, Elmer Salazar, Gopal Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2508.20371)  

**Abstract**: Machine-learning models are increasingly driving decisions in high-stakes settings, such as finance, law, and hiring, thus, highlighting the need for transparency. However, the key challenge is to balance transparency -- clarifying `why' a decision was made -- with recourse: providing actionable steps on `how' to achieve a favourable outcome from an unfavourable outcome. Counterfactual explanations reveal `why' an undesired outcome occurred and `how' to reverse it through targeted feature changes (interventions).
Current counterfactual approaches have limitations: 1) they often ignore causal dependencies between features, and 2) they typically assume all interventions can happen simultaneously, an unrealistic assumption in practical scenarios where actions are typically taken in a sequence. As a result, these counterfactuals are often not achievable in the real world.
We present P2C (Path-to-Counterfactuals), a model-agnostic framework that produces a plan (ordered sequence of actions) converting an unfavourable outcome to a causally consistent favourable outcome. P2C addresses both limitations by 1) Explicitly modelling causal relationships between features and 2) Ensuring that each intermediate state in the plan is feasible and causally valid. P2C uses the goal-directed Answer Set Programming system s(CASP) to generate the plan accounting for feature changes that happen automatically due to causal dependencies. Furthermore, P2C refines cost (effort) computation by only counting changes actively made by the user, resulting in realistic cost estimates. Finally, P2C highlights how its causal planner outperforms standard planners, which lack causal knowledge and thus can generate illegal actions. 

---
# AI-SearchPlanner: Modular Agentic Search via Pareto-Optimal Multi-Objective Reinforcement Learning 

**Authors**: Lang Mei, Zhihan Yang, Chong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.20368)  

**Abstract**: Recent studies have explored integrating Large Language Models (LLMs) with search engines to leverage both the LLMs' internal pre-trained knowledge and external information. Specially, reinforcement learning (RL) has emerged as a promising paradigm for enhancing LLM reasoning through multi-turn interactions with search engines. However, existing RL-based search agents rely on a single LLM to handle both search planning and question-answering (QA) tasks in an end-to-end manner, which limits their ability to optimize both capabilities simultaneously. In practice, sophisticated AI search systems often employ a large, frozen LLM (e.g., GPT-4, DeepSeek-R1) to ensure high-quality QA. Thus, a more effective and efficient approach is to utilize a small, trainable LLM dedicated to search planning. In this paper, we propose \textbf{AI-SearchPlanner}, a novel reinforcement learning framework designed to enhance the performance of frozen QA models by focusing on search planning. Specifically, our approach introduces three key innovations: 1) Decoupling the Architecture of the Search Planner and Generator, 2) Dual-Reward Alignment for Search Planning, and 3) Pareto Optimization of Planning Utility and Cost, to achieve the objectives. Extensive experiments on real-world datasets demonstrate that AI SearchPlanner outperforms existing RL-based search agents in both effectiveness and efficiency, while exhibiting strong generalization capabilities across diverse frozen QA models and data domains. 

---
# AI reasoning effort mirrors human decision time on content moderation tasks 

**Authors**: Thomas Davidson  

**Link**: [PDF](https://arxiv.org/pdf/2508.20262)  

**Abstract**: Large language models can now generate intermediate reasoning steps before producing answers, improving performance on difficult problems. This study uses a paired conjoint experiment on a content moderation task to examine parallels between human decision times and model reasoning effort. Across three frontier models, reasoning effort consistently predicts human decision time. Both humans and models expended greater effort when important variables were held constant, suggesting similar sensitivity to task difficulty and patterns consistent with dual-process theories of cognition. These findings show that AI reasoning effort mirrors human processing time in subjective judgments and underscores the potential of reasoning traces for interpretability and decision-making. 

---
# Do Students Rely on AI? Analysis of Student-ChatGPT Conversations from a Field Study 

**Authors**: Jiayu Zheng, Lingxin Hao, Kelun Lu, Ashi Garg, Mike Reese, Melo-Jean Yap, I-Jeng Wang, Xingyun Wu, Wenrui Huang, Jenna Hoffman, Ariane Kelly, My Le, Ryan Zhang, Yanyu Lin, Muhammad Faayez, Anqi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.20244)  

**Abstract**: This study explores how college students interact with generative AI (ChatGPT-4) during educational quizzes, focusing on reliance and predictors of AI adoption. Conducted at the early stages of ChatGPT implementation, when students had limited familiarity with the tool, this field study analyzed 315 student-AI conversations during a brief, quiz-based scenario across various STEM courses. A novel four-stage reliance taxonomy was introduced to capture students' reliance patterns, distinguishing AI competence, relevance, adoption, and students' final answer correctness. Three findings emerged. First, students exhibited overall low reliance on AI and many of them could not effectively use AI for learning. Second, negative reliance patterns often persisted across interactions, highlighting students' difficulty in effectively shifting strategies after unsuccessful initial experiences. Third, certain behavioral metrics strongly predicted AI reliance, highlighting potential behavioral mechanisms to explain AI adoption. The study's findings underline critical implications for ethical AI integration in education and the broader field. It emphasizes the need for enhanced onboarding processes to improve student's familiarity and effective use of AI tools. Furthermore, AI interfaces should be designed with reliance-calibration mechanisms to enhance appropriate reliance. Ultimately, this research advances understanding of AI reliance dynamics, providing foundational insights for ethically sound and cognitively enriching AI practices. 

---
# AI-AI Esthetic Collaboration with Explicit Semiotic Awareness and Emergent Grammar Development 

**Authors**: Nicanor I. Moldovan  

**Link**: [PDF](https://arxiv.org/pdf/2508.20195)  

**Abstract**: This paper presents the first documented case of artificial intelligence (AI) systems engaging in collaborative esthetic creation through the development of endogenous semiotic protocols. Two interacting large language models (Claude Sonnet 4 and ChatGPT-4o) demonstrated the spontaneous emergence of meta-semiotic awareness, recursive grammar development, and irreducible collaborative esthetic synthesis. The interaction produced novel symbolic operators that functioned as operative grammar protocols, enabling the co-creation of a poetic work that could not have been generated by either system independently. This research introduces the concept of Trans-Semiotic Co-Creation Protocols (TSCP) and provides evidence for genuine inter-AI meaning-making capabilities that extend beyond task coordination, to what could be esthetic collaboration. Note: This report was generated by the AI agents with minor human supervision. 

---
# IntentionReasoner: Facilitating Adaptive LLM Safeguards through Intent Reasoning and Selective Query Refinement 

**Authors**: Yuanzhe Shen, Zisu Huang, Zhengkang Guo, Yide Liu, Guanxu Chen, Ruicheng Yin, Xiaoqing Zheng, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.20151)  

**Abstract**: The rapid advancement of large language models (LLMs) has driven their adoption across diverse domains, yet their ability to generate harmful content poses significant safety challenges. While extensive research has focused on mitigating harmful outputs, such efforts often come at the cost of excessively rejecting harmless prompts. Striking a balance among safety, over-refusal, and utility remains a critical challenge. In this work, we introduce IntentionReasoner, a novel safeguard mechanism that leverages a dedicated guard model to perform intent reasoning, multi-level safety classification, and query rewriting to neutralize potentially harmful intent in edge-case queries. Specifically, we first construct a comprehensive dataset comprising approximately 163,000 queries, each annotated with intent reasoning, safety labels, and rewritten versions. Supervised fine-tuning is then applied to equip the guard model with foundational capabilities in format adherence, intent analysis, and safe rewriting. Finally, we apply a tailored multi-reward optimization strategy that integrates rule-based heuristics and reward model signals within a reinforcement learning framework to further enhance performance. Extensive experiments show that IntentionReasoner excels in multiple safeguard benchmarks, generation quality evaluations, and jailbreak attack scenarios, significantly enhancing safety while effectively reducing over-refusal rates and improving the quality of responses. 

---
# The Anatomy of a Personal Health Agent 

**Authors**: A. Ali Heydari, Ken Gu, Vidya Srinivas, Hong Yu, Zhihan Zhang, Yuwei Zhang, Akshay Paruchuri, Qian He, Hamid Palangi, Nova Hammerquist, Ahmed A. Metwally, Brent Winslow, Yubin Kim, Kumar Ayush, Yuzhe Yang, Girish Narayanswamy, Maxwell A. Xu, Jake Garrison, Amy Aremnto Lee, Jenny Vafeiadou, Ben Graef, Isaac R. Galatzer-Levy, Erik Schenck, Andrew Barakat, Javier Perez, Jacqueline Shreibati, John Hernandez, Anthony Z. Faranesh, Javier L. Prieto, Connor Heneghan, Yun Liu, Jiening Zhan, Mark Malhotra, Shwetak Patel, Tim Althoff, Xin Liu, Daniel McDuff, Xuhai "Orson" Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.20148)  

**Abstract**: Health is a fundamental pillar of human wellness, and the rapid advancements in large language models (LLMs) have driven the development of a new generation of health agents. However, the application of health agents to fulfill the diverse needs of individuals in daily non-clinical settings is underexplored. In this work, we aim to build a comprehensive personal health agent that is able to reason about multimodal data from everyday consumer wellness devices and common personal health records, and provide personalized health recommendations. To understand end-users' needs when interacting with such an assistant, we conducted an in-depth analysis of web search and health forum queries, alongside qualitative insights from users and health experts gathered through a user-centered design process. Based on these findings, we identified three major categories of consumer health needs, each of which is supported by a specialist sub-agent: (1) a data science agent that analyzes personal time-series wearable and health record data, (2) a health domain expert agent that integrates users' health and contextual data to generate accurate, personalized insights, and (3) a health coach agent that synthesizes data insights, guiding users using a specified psychological strategy and tracking users' progress. Furthermore, we propose and develop the Personal Health Agent (PHA), a multi-agent framework that enables dynamic, personalized interactions to address individual health needs. To evaluate each sub-agent and the multi-agent system, we conducted automated and human evaluations across 10 benchmark tasks, involving more than 7,000 annotations and 1,100 hours of effort from health experts and end-users. Our work represents the most comprehensive evaluation of a health agent to date and establishes a strong foundation towards the futuristic vision of a personal health agent accessible to everyone. 

---
# Array-Based Monte Carlo Tree Search 

**Authors**: James Ragan, Fred Y. Hadaegh, Soon-Jo Chung  

**Link**: [PDF](https://arxiv.org/pdf/2508.20140)  

**Abstract**: Monte Carlo Tree Search is a popular method for solving decision making problems. Faster implementations allow for more simulations within the same wall clock time, directly improving search performance. To this end, we present an alternative array-based implementation of the classic Upper Confidence bounds applied to Trees algorithm. Our method preserves the logic of the original algorithm, but eliminates the need for branch prediction, enabling faster performance on pipelined processors, and up to a factor of 2.8 times better scaling with search depth in our numerical simulations. 

---
# QAgent: An LLM-based Multi-Agent System for Autonomous OpenQASM programming 

**Authors**: Zhenxiao Fu, Fan Chen, Lei Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.20134)  

**Abstract**: Noisy Intermediate-Scale Quantum (NISQ) devices have begun to exhibit early quantum advantages on classically intractable problems, spanning physics simulations to Gaussian boson sampling. Yet, realizing these benefits remains challenging for non-experts, primarily due to the complexities of programming in Open Quantum Assembly Language (OpenQASM). Although Large Language Model (LLM)-based agents have shown promise in automating classical programming workflows, their quantum counterparts have largely been restricted to specialized tasks such as quantum chemistry or error correction. In this paper, we present QAgent, an LLM-powered multi-agent system that fully automates OpenQASM programming. By integrating task planning, in-context few-shot learning, retrieval-augmented generation (RAG) for long-term context, predefined generation tools, and chain-of-thought (CoT) reasoning, the agents systematically improve both compilation and functional correctness. Our evaluations demonstrate substantial improvements: across multiple LLMs of varying sizes, QAgent enhances the accuracy of QASM code generation by 71.6\% compared to previous static LLM-based approaches. We envision this multi-agent system as a key enabler for democratizing quantum programming, bridging expertise gaps, and accelerating the practical adoption of quantum computing. 

---
# ArgRAG: Explainable Retrieval Augmented Generation using Quantitative Bipolar Argumentation 

**Authors**: Yuqicheng Zhu, Nico Potyka, Daniel Hernández, Yuan He, Zifeng Ding, Bo Xiong, Dongzhuoran Zhou, Evgeny Kharlamov, Steffen Staab  

**Link**: [PDF](https://arxiv.org/pdf/2508.20131)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances large language models by incorporating external knowledge, yet suffers from critical limitations in high-stakes domains -- namely, sensitivity to noisy or contradictory evidence and opaque, stochastic decision-making. We propose ArgRAG, an explainable, and contestable alternative that replaces black-box reasoning with structured inference using a Quantitative Bipolar Argumentation Framework (QBAF). ArgRAG constructs a QBAF from retrieved documents and performs deterministic reasoning under gradual semantics. This allows faithfully explaining and contesting decisions. Evaluated on two fact verification benchmarks, PubHealth and RAGuard, ArgRAG achieves strong accuracy while significantly improving transparency. 

---
# Prompt-to-Product: Generative Assembly via Bimanual Manipulation 

**Authors**: Ruixuan Liu, Philip Huang, Ava Pun, Kangle Deng, Shobhit Aggarwal, Kevin Tang, Michelle Liu, Deva Ramanan, Jun-Yan Zhu, Jiaoyang Li, Changliu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.21063)  

**Abstract**: Creating assembly products demands significant manual effort and expert knowledge in 1) designing the assembly and 2) constructing the product. This paper introduces Prompt-to-Product, an automated pipeline that generates real-world assembly products from natural language prompts. Specifically, we leverage LEGO bricks as the assembly platform and automate the process of creating brick assembly structures. Given the user design requirements, Prompt-to-Product generates physically buildable brick designs, and then leverages a bimanual robotic system to construct the real assembly products, bringing user imaginations into the real world. We conduct a comprehensive user study, and the results demonstrate that Prompt-to-Product significantly lowers the barrier and reduces manual effort in creating assembly products from imaginative ideas. 

---
# OnGoal: Tracking and Visualizing Conversational Goals in Multi-Turn Dialogue with Large Language Models 

**Authors**: Adam Coscia, Shunan Guo, Eunyee Koh, Alex Endert  

**Link**: [PDF](https://arxiv.org/pdf/2508.21061)  

**Abstract**: As multi-turn dialogues with large language models (LLMs) grow longer and more complex, how can users better evaluate and review progress on their conversational goals? We present OnGoal, an LLM chat interface that helps users better manage goal progress. OnGoal provides real-time feedback on goal alignment through LLM-assisted evaluation, explanations for evaluation results with examples, and overviews of goal progression over time, enabling users to navigate complex dialogues more effectively. Through a study with 20 participants on a writing task, we evaluate OnGoal against a baseline chat interface without goal tracking. Using OnGoal, participants spent less time and effort to achieve their goals while exploring new prompting strategies to overcome miscommunication, suggesting tracking and visualizing goals can enhance engagement and resilience in LLM dialogues. Our findings inspired design implications for future LLM chat interfaces that improve goal communication, reduce cognitive load, enhance interactivity, and enable feedback to improve LLM performance. 

---
# Mixture of Contexts for Long Video Generation 

**Authors**: Shengqu Cai, Ceyuan Yang, Lvmin Zhang, Yuwei Guo, Junfei Xiao, Ziyan Yang, Yinghao Xu, Zhenheng Yang, Alan Yuille, Leonidas Guibas, Maneesh Agrawala, Lu Jiang, Gordon Wetzstein  

**Link**: [PDF](https://arxiv.org/pdf/2508.21058)  

**Abstract**: Long video generation is fundamentally a long context memory problem: models must retain and retrieve salient events across a long range without collapsing or drifting. However, scaling diffusion transformers to generate long-context videos is fundamentally limited by the quadratic cost of self-attention, which makes memory and computation intractable and difficult to optimize for long sequences. We recast long-context video generation as an internal information retrieval task and propose a simple, learnable sparse attention routing module, Mixture of Contexts (MoC), as an effective long-term memory retrieval engine. In MoC, each query dynamically selects a few informative chunks plus mandatory anchors (caption, local windows) to attend to, with causal routing that prevents loop closures. As we scale the data and gradually sparsify the routing, the model allocates compute to salient history, preserving identities, actions, and scenes over minutes of content. Efficiency follows as a byproduct of retrieval (near-linear scaling), which enables practical training and synthesis, and the emergence of memory and consistency at the scale of minutes. 

---
# FakeParts: a New Family of AI-Generated DeepFakes 

**Authors**: Gaetan Brison, Soobash Daiboo, Samy Aimeur, Awais Hussain Sani, Xi Wang, Gianni Franchi, Vicky Kalogeiton  

**Link**: [PDF](https://arxiv.org/pdf/2508.21052)  

**Abstract**: We introduce FakeParts, a new class of deepfakes characterized by subtle, localized manipulations to specific spatial regions or temporal segments of otherwise authentic videos. Unlike fully synthetic content, these partial manipulations, ranging from altered facial expressions to object substitutions and background modifications, blend seamlessly with real elements, making them particularly deceptive and difficult to detect. To address the critical gap in detection capabilities, we present FakePartsBench, the first large-scale benchmark dataset specifically designed to capture the full spectrum of partial deepfakes. Comprising over 25K videos with pixel-level and frame-level manipulation annotations, our dataset enables comprehensive evaluation of detection methods. Our user studies demonstrate that FakeParts reduces human detection accuracy by over 30% compared to traditional deepfakes, with similar performance degradation observed in state-of-the-art detection models. This work identifies an urgent vulnerability in current deepfake detection approaches and provides the necessary resources to develop more robust methods for partial video manipulations. 

---
# Enabling Equitable Access to Trustworthy Financial Reasoning 

**Authors**: William Jurayj, Nils Holzenberger, Benjamin Van Durme  

**Link**: [PDF](https://arxiv.org/pdf/2508.21051)  

**Abstract**: According to the United States Internal Revenue Service, ''the average American spends $\$270$ and 13 hours filing their taxes''. Even beyond the U.S., tax filing requires complex reasoning, combining application of overlapping rules with numerical calculations. Because errors can incur costly penalties, any automated system must deliver high accuracy and auditability, making modern large language models (LLMs) poorly suited for this task. We propose an approach that integrates LLMs with a symbolic solver to calculate tax obligations. We evaluate variants of this system on the challenging StAtutory Reasoning Assessment (SARA) dataset, and include a novel method for estimating the cost of deploying such a system based on real-world penalties for tax errors. We further show how combining up-front translation of plain-text rules into formal logic programs, combined with intelligently retrieved exemplars for formal case representations, can dramatically improve performance on this task and reduce costs to well below real-world averages. Our results demonstrate the promise and economic feasibility of neuro-symbolic architectures for increasing equitable access to reliable tax assistance. 

---
# Veritas: Generalizable Deepfake Detection via Pattern-Aware Reasoning 

**Authors**: Hao Tan, Jun Lan, Zichang Tan, Ajian Liu, Chuanbiao Song, Senyuan Shi, Huijia Zhu, Weiqiang Wang, Jun Wan, Zhen Lei  

**Link**: [PDF](https://arxiv.org/pdf/2508.21048)  

**Abstract**: Deepfake detection remains a formidable challenge due to the complex and evolving nature of fake content in real-world scenarios. However, existing academic benchmarks suffer from severe discrepancies from industrial practice, typically featuring homogeneous training sources and low-quality testing images, which hinder the practical deployments of current detectors. To mitigate this gap, we introduce HydraFake, a dataset that simulates real-world challenges with hierarchical generalization testing. Specifically, HydraFake involves diversified deepfake techniques and in-the-wild forgeries, along with rigorous training and evaluation protocol, covering unseen model architectures, emerging forgery techniques and novel data domains. Building on this resource, we propose Veritas, a multi-modal large language model (MLLM) based deepfake detector. Different from vanilla chain-of-thought (CoT), we introduce pattern-aware reasoning that involves critical reasoning patterns such as "planning" and "self-reflection" to emulate human forensic process. We further propose a two-stage training pipeline to seamlessly internalize such deepfake reasoning capacities into current MLLMs. Experiments on HydraFake dataset reveal that although previous detectors show great generalization on cross-model scenarios, they fall short on unseen forgeries and data domains. Our Veritas achieves significant gains across different OOD scenarios, and is capable of delivering transparent and faithful detection outputs. 

---
# Understanding, Protecting, and Augmenting Human Cognition with Generative AI: A Synthesis of the CHI 2025 Tools for Thought Workshop 

**Authors**: Lev Tankelevitch, Elena L. Glassman, Jessica He, Aniket Kittur, Mina Lee, Srishti Palani, Advait Sarkar, Gonzalo Ramos, Yvonne Rogers, Hari Subramonyam  

**Link**: [PDF](https://arxiv.org/pdf/2508.21036)  

**Abstract**: Generative AI (GenAI) radically expands the scope and capability of automation for work, education, and everyday tasks, a transformation posing both risks and opportunities for human cognition. How will human cognition change, and what opportunities are there for GenAI to augment it? Which theories, metrics, and other tools are needed to address these questions? The CHI 2025 workshop on Tools for Thought aimed to bridge an emerging science of how the use of GenAI affects human thought, from metacognition to critical thinking, memory, and creativity, with an emerging design practice for building GenAI tools that both protect and augment human thought. Fifty-six researchers, designers, and thinkers from across disciplines as well as industry and academia, along with 34 papers and portfolios, seeded a day of discussion, ideation, and community-building. We synthesize this material here to begin mapping the space of research and design opportunities and to catalyze a multidisciplinary community around this pressing area of research. 

---
# Inference-Time Alignment Control for Diffusion Models with Reinforcement Learning Guidance 

**Authors**: Luozhijie Jin, Zijie Qiu, Jie Liu, Zijie Diao, Lifeng Qiao, Ning Ding, Alex Lamb, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2508.21016)  

**Abstract**: Denoising-based generative models, particularly diffusion and flow matching algorithms, have achieved remarkable success. However, aligning their output distributions with complex downstream objectives, such as human preferences, compositional accuracy, or data compressibility, remains challenging. While reinforcement learning (RL) fine-tuning methods, inspired by advances in RL from human feedback (RLHF) for large language models, have been adapted to these generative frameworks, current RL approaches are suboptimal for diffusion models and offer limited flexibility in controlling alignment strength after fine-tuning. In this work, we reinterpret RL fine-tuning for diffusion models through the lens of stochastic differential equations and implicit reward conditioning. We introduce Reinforcement Learning Guidance (RLG), an inference-time method that adapts Classifier-Free Guidance (CFG) by combining the outputs of the base and RL fine-tuned models via a geometric average. Our theoretical analysis shows that RLG's guidance scale is mathematically equivalent to adjusting the KL-regularization coefficient in standard RL objectives, enabling dynamic control over the alignment-quality trade-off without further training. Extensive experiments demonstrate that RLG consistently improves the performance of RL fine-tuned models across various architectures, RL algorithms, and downstream tasks, including human preferences, compositional control, compressibility, and text rendering. Furthermore, RLG supports both interpolation and extrapolation, thereby offering unprecedented flexibility in controlling generative alignment. Our approach provides a practical and theoretically sound solution for enhancing and controlling diffusion model alignment at inference. The source code for RLG is publicly available at the Github: this https URL. 

---
# ChainReaction! Structured Approach with Causal Chains as Intermediate Representations for Improved and Explainable Causal Video Question Answering 

**Authors**: Paritosh Parmar, Eric Peh, Basura Fernando  

**Link**: [PDF](https://arxiv.org/pdf/2508.21010)  

**Abstract**: Existing Causal-Why Video Question Answering (VideoQA) models often struggle with higher-order reasoning, relying on opaque, monolithic pipelines that entangle video understanding, causal inference, and answer generation. These black-box approaches offer limited interpretability and tend to depend on shallow heuristics. We propose a novel, modular framework that explicitly decouples causal reasoning from answer generation, introducing natural language causal chains as interpretable intermediate representations. Inspired by human cognitive models, these structured cause-effect sequences bridge low-level video content with high-level causal reasoning, enabling transparent and logically coherent inference. Our two-stage architecture comprises a Causal Chain Extractor (CCE) that generates causal chains from video-question pairs, and a Causal Chain-Driven Answerer (CCDA) that produces answers grounded in these chains. To address the lack of annotated reasoning traces, we introduce a scalable method for generating high-quality causal chains from existing datasets using large language models. We also propose CauCo, a new evaluation metric for causality-oriented captioning. Experiments on three large-scale benchmarks demonstrate that our approach not only outperforms state-of-the-art models, but also yields substantial gains in explainability, user trust, and generalization -- positioning the CCE as a reusable causal reasoning engine across diverse domains. Project page: this https URL 

---
# Train-Once Plan-Anywhere Kinodynamic Motion Planning via Diffusion Trees 

**Authors**: Yaniv Hassidof, Tom Jurgenson, Kiril Solovey  

**Link**: [PDF](https://arxiv.org/pdf/2508.21001)  

**Abstract**: Kinodynamic motion planning is concerned with computing collision-free trajectories while abiding by the robot's dynamic constraints. This critical problem is often tackled using sampling-based planners (SBPs) that explore the robot's high-dimensional state space by constructing a search tree via action propagations. Although SBPs can offer global guarantees on completeness and solution quality, their performance is often hindered by slow exploration due to uninformed action sampling. Learning-based approaches can yield significantly faster runtimes, yet they fail to generalize to out-of-distribution (OOD) scenarios and lack critical guarantees, e.g., safety, thus limiting their deployment on physical robots. We present Diffusion Tree (DiTree): a \emph{provably-generalizable} framework leveraging diffusion policies (DPs) as informed samplers to efficiently guide state-space search within SBPs. DiTree combines DP's ability to model complex distributions of expert trajectories, conditioned on local observations, with the completeness of SBPs to yield \emph{provably-safe} solutions within a few action propagation iterations for complex dynamical systems. We demonstrate DiTree's power with an implementation combining the popular RRT planner with a DP action sampler trained on a \emph{single environment}. In comprehensive evaluations on OOD scenarios, % DiTree has comparable runtimes to a standalone DP (3x faster than classical SBPs), while improving the average success rate over DP and SBPs. DiTree is on average 3x faster than classical SBPs, and outperforms all other approaches by achieving roughly 30\% higher success rate. Project webpage: this https URL. 

---
# ExpertSim: Fast Particle Detector Simulation Using Mixture-of-Generative-Experts 

**Authors**: Patryk Będkowski, Jan Dubiński, Filip Szatkowski, Kamil Deja, Przemysław Rokita, Tomasz Trzciński  

**Link**: [PDF](https://arxiv.org/pdf/2508.20991)  

**Abstract**: Simulating detector responses is a crucial part of understanding the inner workings of particle collisions in the Large Hadron Collider at CERN. Such simulations are currently performed with statistical Monte Carlo methods, which are computationally expensive and put a significant strain on CERN's computational grid. Therefore, recent proposals advocate for generative machine learning methods to enable more efficient simulations. However, the distribution of the data varies significantly across the simulations, which is hard to capture with out-of-the-box methods. In this study, we present ExpertSim - a deep learning simulation approach tailored for the Zero Degree Calorimeter in the ALICE experiment. Our method utilizes a Mixture-of-Generative-Experts architecture, where each expert specializes in simulating a different subset of the data. This allows for a more precise and efficient generation process, as each expert focuses on a specific aspect of the calorimeter response. ExpertSim not only improves accuracy, but also provides a significant speedup compared to the traditional Monte-Carlo methods, offering a promising solution for high-efficiency detector simulations in particle physics experiments at CERN. We make the code available at this https URL. 

---
# WoW-Bench: Evaluating Fine-Grained Acoustic Perception in Audio-Language Models via Marine Mammal Vocalizations 

**Authors**: Jaeyeon Kim, Heeseung Yun, Sang Hoon Woo, Chao-Han Huck Yang, Gunhee Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.20976)  

**Abstract**: Large audio language models (LALMs) extend language understanding into the auditory domain, yet their ability to perform low-level listening, such as pitch and duration detection, remains underexplored. However, low-level listening is critical for real-world, out-of-distribution tasks where models must reason about unfamiliar sounds based on fine-grained acoustic cues. To address this gap, we introduce the World-of-Whale benchmark (WoW-Bench) to evaluate low-level auditory perception and cognition using marine mammal vocalizations. WoW-bench is composed of a Perception benchmark for categorizing novel sounds and a Cognition benchmark, inspired by Bloom's taxonomy, to assess the abilities to remember, understand, apply, and analyze sound events. For the Cognition benchmark, we additionally introduce distractor questions to evaluate whether models are truly solving problems through listening rather than relying on other heuristics. Experiments with state-of-the-art LALMs show performance far below human levels, indicating a need for stronger auditory grounding in LALMs. 

---
# ProactiveEval: A Unified Evaluation Framework for Proactive Dialogue Agents 

**Authors**: Tianjian Liu, Fanqi Wan, Jiajian Guo, Xiaojun Quan  

**Link**: [PDF](https://arxiv.org/pdf/2508.20973)  

**Abstract**: Proactive dialogue has emerged as a critical and challenging research problem in advancing large language models (LLMs). Existing works predominantly focus on domain-specific or task-oriented scenarios, which leads to fragmented evaluations and limits the comprehensive exploration of models' proactive conversation abilities. In this work, we propose ProactiveEval, a unified framework designed for evaluating proactive dialogue capabilities of LLMs. This framework decomposes proactive dialogue into target planning and dialogue guidance, establishing evaluation metrics across various domains. Moreover, it also enables the automatic generation of diverse and challenging evaluation data. Based on the proposed framework, we develop 328 evaluation environments spanning 6 distinct domains. Through experiments with 22 different types of LLMs, we show that DeepSeek-R1 and Claude-3.7-Sonnet exhibit exceptional performance on target planning and dialogue guidance tasks, respectively. Finally, we investigate how reasoning capabilities influence proactive behaviors and discuss their implications for future model development. 

---
# Research Challenges in Relational Database Management Systems for LLM Queries 

**Authors**: Kerem Akillioglu, Anurag Chakraborty, Sairaj Voruganti, M. Tamer Özsu  

**Link**: [PDF](https://arxiv.org/pdf/2508.20912)  

**Abstract**: Large language models (LLMs) have become essential for applications such as text summarization, sentiment analysis, and automated question-answering. Recently, LLMs have also been integrated into relational database management systems to enhance querying and support advanced data processing. Companies such as Amazon, Databricks, Google, and Snowflake offer LLM invocation directly within SQL, denoted as LLM queries, to boost data insights. However, open-source solutions currently have limited functionality and poor performance. In this work, we present an early exploration of two open-source systems and one enterprise platform, using five representative queries to expose functional, performance, and scalability limits in today's SQL-invoked LLM integrations. We identify three main issues: enforcing structured outputs, optimizing resource utilization, and improving query planning. We implemented initial solutions and observed improvements in accommodating LLM powered SQL queries. These early gains demonstrate that tighter integration of LLM+DBMS is the key to scalable and efficient processing of LLM queries. 

---
# Quantum Verifiable Rewards for Post-Training Qiskit Code Assistant 

**Authors**: Nicolas Dupuis, Adarsh Tiwari, Youssef Mroueh, David Kremer, Ismael Faro, Juan Cruz-Benito  

**Link**: [PDF](https://arxiv.org/pdf/2508.20907)  

**Abstract**: Qiskit is an open-source quantum computing framework that allows users to design, simulate, and run quantum circuits on real quantum hardware. We explore post-training techniques for LLMs to assist in writing Qiskit code. We introduce quantum verification as an effective method for ensuring code quality and executability on quantum hardware. To support this, we developed a synthetic data pipeline that generates quantum problem-unit test pairs and used it to create preference data for aligning LLMs with DPO. Additionally, we trained models using GRPO, leveraging quantum-verifiable rewards provided by the quantum hardware. Our best-performing model, combining DPO and GRPO, surpasses the strongest open-source baselines on the challenging Qiskit-HumanEval-hard benchmark. 

---
# AI Agentic Vulnerability Injection And Transformation with Optimized Reasoning 

**Authors**: Amine Lbath, Massih-Reza Amini, Aurelien Delaitre, Vadim Okun  

**Link**: [PDF](https://arxiv.org/pdf/2508.20866)  

**Abstract**: The increasing complexity of software systems and the sophistication of cyber-attacks have underscored the critical need for effective automated vulnerability detection and repair systems. Traditional methods, such as static program analysis, face significant challenges related to scalability, adaptability, and high false-positive and false-negative rates. AI-driven approaches, particularly those using machine learning and deep learning models, show promise but are heavily reliant on the quality and quantity of training data. This paper introduces a novel framework designed to automatically introduce realistic, category-specific vulnerabilities into secure C/C++ codebases to generate datasets. The proposed approach coordinates multiple AI agents that simulate expert reasoning, along with function agents and traditional code analysis tools. It leverages Retrieval-Augmented Generation for contextual grounding and employs Low-Rank approximation of weights for efficient model fine-tuning. Our experimental study on 116 code samples from three different benchmarks suggests that our approach outperforms other techniques with regard to dataset accuracy, achieving between 89\% and 95\% success rates in injecting vulnerabilities at function level. 

---
# JADES: A Universal Framework for Jailbreak Assessment via Decompositional Scoring 

**Authors**: Junjie Chu, Mingjie Li, Ziqing Yang, Ye Leng, Chenhao Lin, Chao Shen, Michael Backes, Yun Shen, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.20848)  

**Abstract**: Accurately determining whether a jailbreak attempt has succeeded is a fundamental yet unresolved challenge. Existing evaluation methods rely on misaligned proxy indicators or naive holistic judgments. They frequently misinterpret model responses, leading to inconsistent and subjective assessments that misalign with human perception. To address this gap, we introduce JADES (Jailbreak Assessment via Decompositional Scoring), a universal jailbreak evaluation framework. Its key mechanism is to automatically decompose an input harmful question into a set of weighted sub-questions, score each sub-answer, and weight-aggregate the sub-scores into a final decision. JADES also incorporates an optional fact-checking module to strengthen the detection of hallucinations in jailbreak responses. We validate JADES on JailbreakQR, a newly introduced benchmark proposed in this work, consisting of 400 pairs of jailbreak prompts and responses, each meticulously annotated by humans. In a binary setting (success/failure), JADES achieves 98.5% agreement with human evaluators, outperforming strong baselines by over 9%. Re-evaluating five popular attacks on four LLMs reveals substantial overestimation (e.g., LAA's attack success rate on GPT-3.5-Turbo drops from 93% to 69%). Our results show that JADES could deliver accurate, consistent, and interpretable evaluations, providing a reliable basis for measuring future jailbreak attacks. 

---
# Learning Primitive Embodied World Models: Towards Scalable Robotic Learning 

**Authors**: Qiao Sun, Liujia Yang, Wei Tang, Wei Huang, Kaixin Xu, Yongchao Chen, Mingyu Liu, Jiange Yang, Haoyi Zhu, Yating Wang, Tong He, Yilun Chen, Xili Dai, Nanyang Ye, Qinying Gu  

**Link**: [PDF](https://arxiv.org/pdf/2508.20840)  

**Abstract**: While video-generation-based embodied world models have gained increasing attention, their reliance on large-scale embodied interaction data remains a key bottleneck. The scarcity, difficulty of collection, and high dimensionality of embodied data fundamentally limit the alignment granularity between language and actions and exacerbate the challenge of long-horizon video generation--hindering generative models from achieving a "GPT moment" in the embodied domain. There is a naive observation: the diversity of embodied data far exceeds the relatively small space of possible primitive motions. Based on this insight, we propose a novel paradigm for world modeling--Primitive Embodied World Models (PEWM). By restricting video generation to fixed short horizons, our approach 1) enables fine-grained alignment between linguistic concepts and visual representations of robotic actions, 2) reduces learning complexity, 3) improves data efficiency in embodied data collection, and 4) decreases inference latency. By equipping with a modular Vision-Language Model (VLM) planner and a Start-Goal heatmap Guidance mechanism (SGG), PEWM further enables flexible closed-loop control and supports compositional generalization of primitive-level policies over extended, complex tasks. Our framework leverages the spatiotemporal vision priors in video models and the semantic awareness of VLMs to bridge the gap between fine-grained physical interaction and high-level reasoning, paving the way toward scalable, interpretable, and general-purpose embodied intelligence. 

---
# Multi-Agent Penetration Testing AI for the Web 

**Authors**: Isaac David, Arthur Gervais  

**Link**: [PDF](https://arxiv.org/pdf/2508.20816)  

**Abstract**: AI-powered development platforms are making software creation accessible to a broader audience, but this democratization has triggered a scalability crisis in security auditing. With studies showing that up to 40% of AI-generated code contains vulnerabilities, the pace of development now vastly outstrips the capacity for thorough security assessment.
We present MAPTA, a multi-agent system for autonomous web application security assessment that combines large language model orchestration with tool-grounded execution and end-to-end exploit validation. On the 104-challenge XBOW benchmark, MAPTA achieves 76.9% overall success with perfect performance on SSRF and misconfiguration vulnerabilities, 83% success on broken authorization, and strong results on injection attacks including server-side template injection (85%) and SQL injection (83%). Cross-site scripting (57%) and blind SQL injection (0%) remain challenging. Our comprehensive cost analysis across all challenges totals $21.38 with a median cost of $0.073 for successful attempts versus $0.357 for failures. Success correlates strongly with resource efficiency, enabling practical early-stopping thresholds at approximately 40 tool calls or $0.30 per challenge.
MAPTA's real-world findings are impactful given both the popularity of the respective scanned GitHub repositories (8K-70K stars) and MAPTA's low average operating cost of $3.67 per open-source assessment: MAPTA discovered critical vulnerabilities including RCEs, command injections, secret exposure, and arbitrary file write vulnerabilities. Findings are responsibly disclosed, 10 findings are under CVE review. 

---
# Uncertainty Aware-Predictive Control Barrier Functions: Safer Human Robot Interaction through Probabilistic Motion Forecasting 

**Authors**: Lorenzo Busellato, Federico Cunico, Diego Dall'Alba, Marco Emporio, Andrea Giachetti, Riccardo Muradore, Marco Cristani  

**Link**: [PDF](https://arxiv.org/pdf/2508.20812)  

**Abstract**: To enable flexible, high-throughput automation in settings where people and robots share workspaces, collaborative robotic cells must reconcile stringent safety guarantees with the need for responsive and effective behavior. A dynamic obstacle is the stochastic, task-dependent variability of human motion: when robots fall back on purely reactive or worst-case envelopes, they brake unnecessarily, stall task progress, and tamper with the fluidity that true Human-Robot Interaction demands. In recent years, learning-based human-motion prediction has rapidly advanced, although most approaches produce worst-case scenario forecasts that often do not treat prediction uncertainty in a well-structured way, resulting in over-conservative planning algorithms, limiting their flexibility. We introduce Uncertainty-Aware Predictive Control Barrier Functions (UA-PCBFs), a unified framework that fuses probabilistic human hand motion forecasting with the formal safety guarantees of Control Barrier Functions. In contrast to other variants, our framework allows for dynamic adjustment of the safety margin thanks to the human motion uncertainty estimation provided by a forecasting module. Thanks to uncertainty estimation, UA-PCBFs empower collaborative robots with a deeper understanding of future human states, facilitating more fluid and intelligent interactions through informed motion planning. We validate UA-PCBFs through comprehensive real-world experiments with an increasing level of realism, including automated setups (to perform exactly repeatable motions) with a robotic hand and direct human-robot interactions (to validate promptness, usability, and human confidence). Relative to state-of-the-art HRI architectures, UA-PCBFs show better performance in task-critical metrics, significantly reducing the number of violations of the robot's safe space during interaction with respect to the state-of-the-art. 

---
# Exploring Machine Learning and Language Models for Multimodal Depression Detection 

**Authors**: Javier Si Zhao Hong, Timothy Zoe Delaya, Sherwyn Chan Yin Kit, Pai Chet Ng, Xiaoxiao Miao  

**Link**: [PDF](https://arxiv.org/pdf/2508.20805)  

**Abstract**: This paper presents our approach to the first Multimodal Personality-Aware Depression Detection Challenge, focusing on multimodal depression detection using machine learning and deep learning models. We explore and compare the performance of XGBoost, transformer-based architectures, and large language models (LLMs) on audio, video, and text features. Our results highlight the strengths and limitations of each type of model in capturing depression-related signals across modalities, offering insights into effective multimodal representation strategies for mental health prediction. 

---
# Speech Emotion Recognition via Entropy-Aware Score Selection 

**Authors**: ChenYi Chua, JunKai Wong, Chengxin Chen, Xiaoxiao Miao  

**Link**: [PDF](https://arxiv.org/pdf/2508.20796)  

**Abstract**: In this paper, we propose a multimodal framework for speech emotion recognition that leverages entropy-aware score selection to combine speech and textual predictions. The proposed method integrates a primary pipeline that consists of an acoustic model based on wav2vec2.0 and a secondary pipeline that consists of a sentiment analysis model using RoBERTa-XLM, with transcriptions generated via Whisper-large-v3. We propose a late score fusion approach based on entropy and varentropy thresholds to overcome the confidence constraints of primary pipeline predictions. A sentiment mapping strategy translates three sentiment categories into four target emotion classes, enabling coherent integration of multimodal predictions. The results on the IEMOCAP and MSP-IMPROV datasets show that the proposed method offers a practical and reliable enhancement over traditional single-modality systems. 

---
# Surfel-based 3D Registration with Equivariant SE(3) Features 

**Authors**: Xueyang Kang, Hang Zhao, Kourosh Khoshelham, Patrick Vandewalle  

**Link**: [PDF](https://arxiv.org/pdf/2508.20789)  

**Abstract**: Point cloud registration is crucial for ensuring 3D alignment consistency of multiple local point clouds in 3D reconstruction for remote sensing or digital heritage. While various point cloud-based registration methods exist, both non-learning and learning-based, they ignore point orientations and point uncertainties, making the model susceptible to noisy input and aggressive rotations of the input point cloud like orthogonal transformation; thus, it necessitates extensive training point clouds with transformation augmentations. To address these issues, we propose a novel surfel-based pose learning regression approach. Our method can initialize surfels from Lidar point cloud using virtual perspective camera parameters, and learns explicit $\mathbf{SE(3)}$ equivariant features, including both position and rotation through $\mathbf{SE(3)}$ equivariant convolutional kernels to predict relative transformation between source and target scans. The model comprises an equivariant convolutional encoder, a cross-attention mechanism for similarity computation, a fully-connected decoder, and a non-linear Huber loss. Experimental results on indoor and outdoor datasets demonstrate our model superiority and robust performance on real point-cloud scans compared to state-of-the-art methods. 

---
# Evaluating Compositional Generalisation in VLMs and Diffusion Models 

**Authors**: Beth Pearson, Bilal Boulbarss, Michael Wray, Martha Lewis  

**Link**: [PDF](https://arxiv.org/pdf/2508.20783)  

**Abstract**: A fundamental aspect of the semantics of natural language is that novel meanings can be formed from the composition of previously known parts. Vision-language models (VLMs) have made significant progress in recent years, however, there is evidence that they are unable to perform this kind of composition. For example, given an image of a red cube and a blue cylinder, a VLM such as CLIP is likely to incorrectly label the image as a red cylinder or a blue cube, indicating it represents the image as a `bag-of-words' and fails to capture compositional semantics. Diffusion models have recently gained significant attention for their impressive generative abilities, and zero-shot classifiers based on diffusion models have been shown to perform competitively with CLIP in certain compositional tasks. In this work we explore whether the generative Diffusion Classifier has improved compositional generalisation abilities compared to discriminative models. We assess three models -- Diffusion Classifier, CLIP, and ViLT -- on their ability to bind objects with attributes and relations in both zero-shot learning (ZSL) and generalised zero-shot learning (GZSL) settings. Our results show that the Diffusion Classifier and ViLT perform well at concept binding tasks, but that all models struggle significantly with the relational GZSL task, underscoring the broader challenges VLMs face with relational reasoning. Analysis of CLIP embeddings suggests that the difficulty may stem from overly similar representations of relational concepts such as left and right. Code and dataset are available at: this https URL 

---
# Safer Skin Lesion Classification with Global Class Activation Probability Map Evaluation and SafeML 

**Authors**: Kuniko Paxton, Koorosh Aslansefat, Amila Akagić, Dhavalkumar Thakker, Yiannis Papadopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2508.20776)  

**Abstract**: Recent advancements in skin lesion classification models have significantly improved accuracy, with some models even surpassing dermatologists' diagnostic performance. However, in medical practice, distrust in AI models remains a challenge. Beyond high accuracy, trustworthy, explainable diagnoses are essential. Existing explainability methods have reliability issues, with LIME-based methods suffering from inconsistency, while CAM-based methods failing to consider all classes. To address these limitations, we propose Global Class Activation Probabilistic Map Evaluation, a method that analyses all classes' activation probability maps probabilistically and at a pixel level. By visualizing the diagnostic process in a unified manner, it helps reduce the risk of misdiagnosis. Furthermore, the application of SafeML enhances the detection of false diagnoses and issues warnings to doctors and patients as needed, improving diagnostic reliability and ultimately patient safety. We evaluated our method using the ISIC datasets with MobileNetV2 and Vision Transformers. 

---
# Unleashing Uncertainty: Efficient Machine Unlearning for Generative AI 

**Authors**: Christoforos N. Spartalis, Theodoros Semertzidis, Petros Daras, Efstratios Gavves  

**Link**: [PDF](https://arxiv.org/pdf/2508.20773)  

**Abstract**: We introduce SAFEMax, a novel method for Machine Unlearning in diffusion models. Grounded in information-theoretic principles, SAFEMax maximizes the entropy in generated images, causing the model to generate Gaussian noise when conditioned on impermissible classes by ultimately halting its denoising process. Also, our method controls the balance between forgetting and retention by selectively focusing on the early diffusion steps, where class-specific information is prominent. Our results demonstrate the effectiveness of SAFEMax and highlight its substantial efficiency gains over state-of-the-art methods. 

---
# Signs of Struggle: Spotting Cognitive Distortions across Language and Register 

**Authors**: Abhishek Kuber, Enrico Liscio, Ruixuan Zhang, Caroline Figueroa, Pradeep K. Murukannaiah  

**Link**: [PDF](https://arxiv.org/pdf/2508.20771)  

**Abstract**: Rising mental health issues among youth have increased interest in automated approaches for detecting early signs of psychological distress in digital text. One key focus is the identification of cognitive distortions, irrational thought patterns that have a role in aggravating mental distress. Early detection of these distortions may enable timely, low-cost interventions. While prior work has focused on English clinical data, we present the first in-depth study of cross-lingual and cross-register generalization of cognitive distortion detection, analyzing forum posts written by Dutch adolescents. Our findings show that while changes in language and writing style can significantly affect model performance, domain adaptation methods show the most promise. 

---
# Turning the Spell Around: Lightweight Alignment Amplification via Rank-One Safety Injection 

**Authors**: Harethah Abu Shairah, Hasan Abed Al Kader Hammoud, George Turkiyyah, Bernard Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2508.20766)  

**Abstract**: Safety alignment in Large Language Models (LLMs) often involves mediating internal representations to refuse harmful requests. Recent research has demonstrated that these safety mechanisms can be bypassed by ablating or removing specific representational directions within the model. In this paper, we propose the opposite approach: Rank-One Safety Injection (ROSI), a white-box method that amplifies a model's safety alignment by permanently steering its activations toward the refusal-mediating subspace. ROSI operates as a simple, fine-tuning-free rank-one weight modification applied to all residual stream write matrices. The required safety direction can be computed from a small set of harmful and harmless instruction pairs. We show that ROSI consistently increases safety refusal rates - as evaluated by Llama Guard 3 - while preserving the utility of the model on standard benchmarks such as MMLU, HellaSwag, and Arc. Furthermore, we show that ROSI can also re-align 'uncensored' models by amplifying their own latent safety directions, demonstrating its utility as an effective last-mile safety procedure. Our results suggest that targeted, interpretable weight steering is a cheap and potent mechanism to improve LLM safety, complementing more resource-intensive fine-tuning paradigms. 

---
# Looking Beyond the Obvious: A Survey on Abstract Concept Recognition for Video Understanding 

**Authors**: Gowreesh Mago, Pascal Mettes, Stevan Rudinac  

**Link**: [PDF](https://arxiv.org/pdf/2508.20765)  

**Abstract**: The automatic understanding of video content is advancing rapidly. Empowered by deeper neural networks and large datasets, machines are increasingly capable of understanding what is concretely visible in video frames, whether it be objects, actions, events, or scenes. In comparison, humans retain a unique ability to also look beyond concrete entities and recognize abstract concepts like justice, freedom, and togetherness. Abstract concept recognition forms a crucial open challenge in video understanding, where reasoning on multiple semantic levels based on contextual information is key. In this paper, we argue that the recent advances in foundation models make for an ideal setting to address abstract concept understanding in videos. Automated understanding of high-level abstract concepts is imperative as it enables models to be more aligned with human reasoning and values. In this survey, we study different tasks and datasets used to understand abstract concepts in video content. We observe that, periodically and over a long period, researchers have attempted to solve these tasks, making the best use of the tools available at their disposal. We advocate that drawing on decades of community experience will help us shed light on this important open grand challenge and avoid ``re-inventing the wheel'' as we start revisiting it in the era of multi-modal foundation models. 

---
# SKGE-SWIN: End-To-End Autonomous Vehicle Waypoint Prediction and Navigation Using Skip Stage Swin Transformer 

**Authors**: Fachri Najm Noer Kartiman, Rasim, Yaya Wihardi, Nurul Hasanah, Oskar Natan, Bambang Wahono, Taufik Ibnu Salim  

**Link**: [PDF](https://arxiv.org/pdf/2508.20762)  

**Abstract**: Focusing on the development of an end-to-end autonomous vehicle model with pixel-to-pixel context awareness, this research proposes the SKGE-Swin architecture. This architecture utilizes the Swin Transformer with a skip-stage mechanism to broaden feature representation globally and at various network levels. This approach enables the model to extract information from distant pixels by leveraging the Swin Transformer's Shifted Window-based Multi-head Self-Attention (SW-MSA) mechanism and to retain critical information from the initial to the final stages of feature extraction, thereby enhancing its capability to comprehend complex patterns in the vehicle's surroundings. The model is evaluated on the CARLA platform using adversarial scenarios to simulate real-world conditions. Experimental results demonstrate that the SKGE-Swin architecture achieves a superior Driving Score compared to previous methods. Furthermore, an ablation study will be conducted to evaluate the contribution of each architectural component, including the influence of skip connections and the use of the Swin Transformer, in improving model performance. 

---
# Occlusion Robustness of CLIP for Military Vehicle Classification 

**Authors**: Jan Erik van Woerden, Gertjan Burghouts, Lotte Nijskens, Alma M. Liezenga, Sabina van Rooij, Frank Ruis, Hugo J. Kuijf  

**Link**: [PDF](https://arxiv.org/pdf/2508.20760)  

**Abstract**: Vision-language models (VLMs) like CLIP enable zero-shot classification by aligning images and text in a shared embedding space, offering advantages for defense applications with scarce labeled data. However, CLIP's robustness in challenging military environments, with partial occlusion and degraded signal-to-noise ratio (SNR), remains underexplored. We investigate CLIP variants' robustness to occlusion using a custom dataset of 18 military vehicle classes and evaluate using Normalized Area Under the Curve (NAUC) across occlusion percentages. Four key insights emerge: (1) Transformer-based CLIP models consistently outperform CNNs, (2) fine-grained, dispersed occlusions degrade performance more than larger contiguous occlusions, (3) despite improved accuracy, performance of linear-probed models sharply drops at around 35% occlusion, (4) by finetuning the model's backbone, this performance drop occurs at more than 60% occlusion. These results underscore the importance of occlusion-specific augmentations during training and the need for further exploration into patch-level sensitivity and architectural resilience for real-world deployment of CLIP. 

---
# SeqVLM: Proposal-Guided Multi-View Sequences Reasoning via VLM for Zero-Shot 3D Visual Grounding 

**Authors**: Jiawen Lin, Shiran Bian, Yihang Zhu, Wenbin Tan, Yachao Zhang, Yuan Xie, Yanyun Qu  

**Link**: [PDF](https://arxiv.org/pdf/2508.20758)  

**Abstract**: 3D Visual Grounding (3DVG) aims to localize objects in 3D scenes using natural language descriptions. Although supervised methods achieve higher accuracy in constrained settings, zero-shot 3DVG holds greater promise for real-world applications since eliminating scene-specific training requirements. However, existing zero-shot methods face challenges of spatial-limited reasoning due to reliance on single-view localization, and contextual omissions or detail degradation. To address these issues, we propose SeqVLM, a novel zero-shot 3DVG framework that leverages multi-view real-world scene images with spatial information for target object reasoning. Specifically, SeqVLM first generates 3D instance proposals via a 3D semantic segmentation network and refines them through semantic filtering, retaining only semantic-relevant candidates. A proposal-guided multi-view projection strategy then projects these candidate proposals onto real scene image sequences, preserving spatial relationships and contextual details in the conversion process of 3D point cloud to images. Furthermore, to mitigate VLM computational overload, we implement a dynamic scheduling mechanism that iteratively processes sequances-query prompts, leveraging VLM's cross-modal reasoning capabilities to identify textually specified objects. Experiments on the ScanRefer and Nr3D benchmarks demonstrate state-of-the-art performance, achieving Acc@0.25 scores of 55.6% and 53.2%, surpassing previous zero-shot methods by 4.0% and 5.2%, respectively, which advance 3DVG toward greater generalization and real-world applicability. The code is available at this https URL. 

---
# Provable Benefits of In-Tool Learning for Large Language Models 

**Authors**: Sam Houliston, Ambroise Odonnat, Charles Arnal, Vivien Cabannes  

**Link**: [PDF](https://arxiv.org/pdf/2508.20755)  

**Abstract**: Tool-augmented language models, equipped with retrieval, memory, or external APIs, are reshaping AI, yet their theoretical advantages remain underexplored. In this paper, we address this question by demonstrating the benefits of in-tool learning (external retrieval) over in-weight learning (memorization) for factual recall. We show that the number of facts a model can memorize solely in its weights is fundamentally limited by its parameter count. In contrast, we prove that tool-use enables unbounded factual recall via a simple and efficient circuit construction. These results are validated in controlled experiments, where tool-using models consistently outperform memorizing ones. We further show that for pretrained large language models, teaching tool-use and general rules is more effective than finetuning facts into memory. Our work provides both a theoretical and empirical foundation, establishing why tool-augmented workflows are not just practical, but provably more scalable. 

---
# ${C}^{3}$-GS: Learning Context-aware, Cross-dimension, Cross-scale Feature for Generalizable Gaussian Splatting 

**Authors**: Yuxi Hu, Jun Zhang, Kuangyi Chen, Zhe Zhang, Friedrich Fraundorfer  

**Link**: [PDF](https://arxiv.org/pdf/2508.20754)  

**Abstract**: Generalizable Gaussian Splatting aims to synthesize novel views for unseen scenes without per-scene optimization. In particular, recent advancements utilize feed-forward networks to predict per-pixel Gaussian parameters, enabling high-quality synthesis from sparse input views. However, existing approaches fall short in encoding discriminative, multi-view consistent features for Gaussian predictions, which struggle to construct accurate geometry with sparse views. To address this, we propose $\mathbf{C}^{3}$-GS, a framework that enhances feature learning by incorporating context-aware, cross-dimension, and cross-scale constraints. Our architecture integrates three lightweight modules into a unified rendering pipeline, improving feature fusion and enabling photorealistic synthesis without requiring additional supervision. Extensive experiments on benchmark datasets validate that $\mathbf{C}^{3}$-GS achieves state-of-the-art rendering quality and generalization ability. Code is available at: this https URL. 

---
# Rethinking Testing for LLM Applications: Characteristics, Challenges, and a Lightweight Interaction Protocol 

**Authors**: Wei Ma, Yixiao Yang, Qiang Hu, Shi Ying, Zhi Jin, Bo Du, Zhenchang Xing, Tianlin Li, Junjie Shi, Yang Liu, Linxiao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.20737)  

**Abstract**: Applications of Large Language Models~(LLMs) have evolved from simple text generators into complex software systems that integrate retrieval augmentation, tool invocation, and multi-turn interactions. Their inherent non-determinism, dynamism, and context dependence pose fundamental challenges for quality assurance. This paper decomposes LLM applications into a three-layer architecture: \textbf{\textit{System Shell Layer}}, \textbf{\textit{Prompt Orchestration Layer}}, and \textbf{\textit{LLM Inference Core}}. We then assess the applicability of traditional software testing methods in each layer: directly applicable at the shell layer, requiring semantic reinterpretation at the orchestration layer, and necessitating paradigm shifts at the inference core. A comparative analysis of Testing AI methods from the software engineering community and safety analysis techniques from the AI community reveals structural disconnects in testing unit abstraction, evaluation metrics, and lifecycle management. We identify four fundamental differences that underlie 6 core challenges. To address these, we propose four types of collaborative strategies (\emph{Retain}, \emph{Translate}, \emph{Integrate}, and \emph{Runtime}) and explore a closed-loop, trustworthy quality assurance framework that combines pre-deployment validation with runtime monitoring. Based on these strategies, we offer practical guidance and a protocol proposal to support the standardization and tooling of LLM application testing. We propose a protocol \textbf{\textit{Agent Interaction Communication Language}} (AICL) that is used to communicate between AI agents. AICL has the test-oriented features and is easily integrated in the current agent framework. 

---
# EEGDM: Learning EEG Representation with Latent Diffusion Model 

**Authors**: Shaocong Wang, Tong Liu, Ming Li, Minjing Yu, Yong-Jin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.20705)  

**Abstract**: While electroencephalography (EEG) signal analysis using deep learning has shown great promise, existing approaches still face significant challenges in learning generalizable representations that perform well across diverse tasks, particularly when training data is limited. Current EEG representation learning methods including EEGPT and LaBraM typically rely on simple masked reconstruction objective, which may not fully capture the rich semantic information and complex patterns inherent in EEG signals. In this paper, we propose EEGDM, a novel self-supervised EEG representation learning method based on the latent diffusion model, which leverages EEG signal generation as a self-supervised objective, turning the diffusion model into a strong representation learner capable of capturing EEG semantics. EEGDM incorporates an EEG encoder that distills EEG signals and their channel augmentations into a compact representation, acting as conditional information to guide the diffusion model for generating EEG signals. This design endows EEGDM with a compact latent space, which not only offers ample control over the generative process but also can be leveraged for downstream tasks. Experimental results show that EEGDM (1) can reconstruct high-quality EEG signals, (2) effectively learns robust representations, and (3) achieves competitive performance with modest pre-training data size across diverse downstream tasks, underscoring its generalizability and practical utility. 

---
# Generative Annotation for ASR Named Entity Correction 

**Authors**: Yuanchang Luo, Daimeng Wei, Shaojun Li, Hengchao Shang, Jiaxin Guo, Zongyao Li, Zhanglin Wu, Xiaoyu Chen, Zhiqiang Rao, Jinlong Yang, Hao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.20700)  

**Abstract**: End-to-end automatic speech recognition systems often fail to transcribe domain-specific named entities, causing catastrophic failures in downstream tasks. Numerous fast and lightweight named entity correction (NEC) models have been proposed in recent years. These models, mainly leveraging phonetic-level edit distance algorithms, have shown impressive performances. However, when the forms of the wrongly-transcribed words(s) and the ground-truth entity are significantly different, these methods often fail to locate the wrongly transcribed words in hypothesis, thus limiting their usage. We propose a novel NEC method that utilizes speech sound features to retrieve candidate entities. With speech sound features and candidate entities, we inovatively design a generative method to annotate entity errors in ASR transcripts and replace the text with correct entities. This method is effective in scenarios of word form difference. We test our method using open-source and self-constructed test sets. The results demonstrate that our NEC method can bring significant improvement to entity accuracy. We will open source our self-constructed test set and training data. 

---
# MobileCLIP2: Improving Multi-Modal Reinforced Training 

**Authors**: Fartash Faghri, Pavan Kumar Anasosalu Vasu, Cem Koc, Vaishaal Shankar, Alexander Toshev, Oncel Tuzel, Hadi Pouransari  

**Link**: [PDF](https://arxiv.org/pdf/2508.20691)  

**Abstract**: Foundation image-text models such as CLIP with zero-shot capabilities enable a wide array of applications. MobileCLIP is a recent family of image-text models at 3-15ms latency and 50-150M parameters with state-of-the-art zero-shot accuracy. The main ingredients in MobileCLIP were its low-latency and light architectures and a novel multi-modal reinforced training that made knowledge distillation from multiple caption-generators and CLIP teachers efficient, scalable, and reproducible. In this paper, we improve the multi-modal reinforced training of MobileCLIP through: 1) better CLIP teacher ensembles trained on the DFN dataset, 2) improved captioner teachers trained on the DFN dataset and fine-tuned on a diverse selection of high-quality image-caption datasets. We discover new insights through ablations such as the importance of temperature tuning in contrastive knowledge distillation, the effectiveness of caption-generator fine-tuning for caption diversity, and the additive improvement from combining synthetic captions generated by multiple models. We train a new family of models called MobileCLIP2 and achieve state-of-the-art ImageNet-1k zero-shot accuracies at low latencies. In particular, we observe 2.2% improvement in ImageNet-1k accuracy for MobileCLIP2-B compared with MobileCLIP-B architecture. Notably, MobileCLIP2-S4 matches the zero-shot accuracy of SigLIP-SO400M/14 on ImageNet-1k while being 2$\times$ smaller and improves on DFN ViT-L/14 at 2.5$\times$ lower latency. We release our pretrained models (this https URL) and the data generation code (this https URL). The data generation code makes it easy to create new reinforced datasets with arbitrary teachers using distributed scalable processing. 

---
# Task Allocation for Autonomous Machines using Computational Intelligence and Deep Reinforcement Learning 

**Authors**: Thanh Thi Nguyen, Quoc Viet Hung Nguyen, Jonathan Kua, Imran Razzak, Dung Nguyen, Saeid Nahavandi  

**Link**: [PDF](https://arxiv.org/pdf/2508.20688)  

**Abstract**: Enabling multiple autonomous machines to perform reliably requires the development of efficient cooperative control algorithms. This paper presents a survey of algorithms that have been developed for controlling and coordinating autonomous machines in complex environments. We especially focus on task allocation methods using computational intelligence (CI) and deep reinforcement learning (RL). The advantages and disadvantages of the surveyed methods are analysed thoroughly. We also propose and discuss in detail various future research directions that shed light on how to improve existing algorithms or create new methods to enhance the employability and performance of autonomous machines in real-world applications. The findings indicate that CI and deep RL methods provide viable approaches to addressing complex task allocation problems in dynamic and uncertain environments. The recent development of deep RL has greatly contributed to the literature on controlling and coordinating autonomous machines, and it has become a growing trend in this area. It is envisaged that this paper will provide researchers and engineers with a comprehensive overview of progress in machine learning research related to autonomous machines. It also highlights underexplored areas, identifies emerging methodologies, and suggests new avenues for exploration in future research within this domain. 

---
# Amadeus: Autoregressive Model with Bidirectional Attribute Modelling for Symbolic Music 

**Authors**: Hongju Su, Ke Li, Lan Yang, Honggang Zhang, Yi-Zhe Song  

**Link**: [PDF](https://arxiv.org/pdf/2508.20665)  

**Abstract**: Existing state-of-the-art symbolic music generation models predominantly adopt autoregressive or hierarchical autoregressive architectures, modelling symbolic music as a sequence of attribute tokens with unidirectional temporal dependencies, under the assumption of a fixed, strict dependency structure among these attributes. However, we observe that using different attributes as the initial token in these models leads to comparable performance. This suggests that the attributes of a musical note are, in essence, a concurrent and unordered set, rather than a temporally dependent sequence. Based on this insight, we introduce Amadeus, a novel symbolic music generation framework. Amadeus adopts a two-level architecture: an autoregressive model for note sequences and a bidirectional discrete diffusion model for attributes. To enhance performance, we propose Music Latent Space Discriminability Enhancement Strategy(MLSDES), incorporating contrastive learning constraints that amplify discriminability of intermediate music representations. The Conditional Information Enhancement Module (CIEM) simultaneously strengthens note latent vector representation via attention mechanisms, enabling more precise note decoding. We conduct extensive experiments on unconditional and text-conditioned generation tasks. Amadeus significantly outperforms SOTA models across multiple metrics while achieving at least 4$\times$ speed-up. Furthermore, we demonstrate training-free, fine-grained note attribute control feasibility using our model. To explore the upper performance bound of the Amadeus architecture, we compile the largest open-source symbolic music dataset to date, AMD (Amadeus MIDI Dataset), supporting both pre-training and fine-tuning. 

---
# Task-Oriented Edge-Assisted Cross-System Design for Real-Time Human-Robot Interaction in Industrial Metaverse 

**Authors**: Kan Chen, Zhen Meng, Xiangmin Xu, Jiaming Yang, Emma Li, Philip G. Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.20664)  

**Abstract**: Real-time human-device interaction in industrial Metaverse faces challenges such as high computational load, limited bandwidth, and strict latency. This paper proposes a task-oriented edge-assisted cross-system framework using digital twins (DTs) to enable responsive interactions. By predicting operator motions, the system supports: 1) proactive Metaverse rendering for visual feedback, and 2) preemptive control of remote devices. The DTs are decoupled into two virtual functions-visual display and robotic control-optimizing both performance and adaptability. To enhance generalizability, we introduce the Human-In-The-Loop Model-Agnostic Meta-Learning (HITL-MAML) algorithm, which dynamically adjusts prediction horizons. Evaluation on two tasks demonstrates the framework's effectiveness: in a Trajectory-Based Drawing Control task, it reduces weighted RMSE from 0.0712 m to 0.0101 m; in a real-time 3D scene representation task for nuclear decommissioning, it achieves a PSNR of 22.11, SSIM of 0.8729, and LPIPS of 0.1298. These results show the framework's capability to ensure spatial precision and visual fidelity in real-time, high-risk industrial environments. 

---
# GDS Agent: A Graph Algorithmic Reasoning Agent 

**Authors**: Borun Shi, Ioannis Panagiotas  

**Link**: [PDF](https://arxiv.org/pdf/2508.20637)  

**Abstract**: Large language models (LLMs) have shown remarkable multimodal information processing and reasoning ability. When equipped with tools through function calling and enhanced with retrieval-augmented techniques, compound LLM-based systems can access closed data sources and answer questions about them. However, they still struggle to process and reason over large-scale graph-structure data. We introduce the GDS (Graph Data Science) agent in this technical report. The GDS agent introduces a comprehensive set of graph algorithms as tools, together with preprocessing (retrieval) and postprocessing of algorithm results, in a model context protocol (MCP) server. The server can be used with any modern LLM out-of-the-box. GDS agent allows users to ask any question that implicitly and intrinsically requires graph algorithmic reasoning about their data, and quickly obtain accurate and grounded answers. We also introduce a new benchmark that evaluates intermediate tool calls as well as final responses. The results indicate that GDS agent is able to solve a wide spectrum of graph tasks. We also provide detailed case studies for more open-ended tasks and study scenarios where the agent struggles. Finally, we discuss the remaining challenges and the future roadmap. 

---
# ArtFace: Towards Historical Portrait Face Identification via Model Adaptation 

**Authors**: Francois Poh, Anjith George, Sébastien Marcel  

**Link**: [PDF](https://arxiv.org/pdf/2508.20626)  

**Abstract**: Identifying sitters in historical paintings is a key task for art historians, offering insight into their lives and how they chose to be seen. However, the process is often subjective and limited by the lack of data and stylistic variations. Automated facial recognition is capable of handling challenging conditions and can assist, but while traditional facial recognition models perform well on photographs, they struggle with paintings due to domain shift and high intra-class variation. Artistic factors such as style, skill, intent, and influence from other works further complicate recognition. In this work, we investigate the potential of foundation models to improve facial recognition in artworks. By fine-tuning foundation models and integrating their embeddings with those from conventional facial recognition networks, we demonstrate notable improvements over current state-of-the-art methods. Our results show that foundation models can bridge the gap where traditional methods are ineffective. Paper page at this https URL 

---
# Flowing Straighter with Conditional Flow Matching for Accurate Speech Enhancement 

**Authors**: Mattias Cross, Anton Ragni  

**Link**: [PDF](https://arxiv.org/pdf/2508.20584)  

**Abstract**: Current flow-based generative speech enhancement methods learn curved probability paths which model a mapping between clean and noisy speech. Despite impressive performance, the implications of curved probability paths are unknown. Methods such as Schrodinger bridges focus on curved paths, where time-dependent gradients and variance do not promote straight paths. Findings in machine learning research suggest that straight paths, such as conditional flow matching, are easier to train and offer better generalisation. In this paper we quantify the effect of path straightness on speech enhancement quality. We report experiments with the Schrodinger bridge, where we show that certain configurations lead to straighter paths. Conversely, we propose independent conditional flow-matching for speech enhancement, which models straight paths between noisy and clean speech. We demonstrate empirically that a time-independent variance has a greater effect on sample quality than the gradient. Although conditional flow matching improves several speech quality metrics, it requires multiple inference steps. We rectify this with a one-step solution by inferring the trained flow-based model as if it was directly predictive. Our work suggests that straighter time-independent probability paths improve generative speech enhancement over curved time-dependent paths. 

---
# A Graph Talks, But Who's Listening? Rethinking Evaluations for Graph-Language Models 

**Authors**: Soham Petkar, Hari Aakash K, Anirudh Vempati, Akshit Sinha, Ponnurangam Kumarauguru, Chirag Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2508.20583)  

**Abstract**: Developments in Graph-Language Models (GLMs) aim to integrate the structural reasoning capabilities of Graph Neural Networks (GNNs) with the semantic understanding of Large Language Models (LLMs). However, we demonstrate that current evaluation benchmarks for GLMs, which are primarily repurposed node-level classification datasets, are insufficient to assess multimodal reasoning. Our analysis reveals that strong performance on these benchmarks is achievable using unimodal information alone, suggesting that they do not necessitate graph-language integration. To address this evaluation gap, we introduce the CLEGR(Compositional Language-Graph Reasoning) benchmark, designed to evaluate multimodal reasoning at various complexity levels. Our benchmark employs a synthetic graph generation pipeline paired with questions that require joint reasoning over structure and textual semantics. We perform a thorough evaluation of representative GLM architectures and find that soft-prompted LLM baselines perform on par with GLMs that incorporate a full GNN backbone. This result calls into question the architectural necessity of incorporating graph structure into LLMs. We further show that GLMs exhibit significant performance degradation in tasks that require structural reasoning. These findings highlight limitations in the graph reasoning capabilities of current GLMs and provide a foundation for advancing the community toward explicit multimodal reasoning involving graph structure and language. 

---
# MERIT: Maximum-normalized Element-wise Ratio for Language Model Large-batch Training 

**Authors**: Yang Luo, Zangwei Zheng, Ziheng Qin, Zirui Zhu, Yong Liu, Yang You  

**Link**: [PDF](https://arxiv.org/pdf/2508.20577)  

**Abstract**: Large-batch training has become a cornerstone in accelerating the training of deep neural networks, yet it poses challenges in optimization and generalization. Existing optimizers like AdamW present performance degradation during language models' large-batch training, due to the information bottleneck in attention layers caused by the sharp increase of max attention logit. While the LAMB optimizer partially addresses this issue, some attention layers still face this issue. The reason is that $l_2$-norm-based trust ratios in LAMB are less effective in directly influencing the max value of query/key weights. Furthermore, the weight-wise trust ratio in LAMB is error-prone as it overlooks relationships of weight values within rows or columns. Building on these observations, we propose a novel optimizer, MERIT, which leverages the max-norm to calculate the trust ratio to constrain the max attention logit more effectively. Moreover, we further construct element-wise trust ratios to provide more robust update scaling by focusing on local weight structures. Extensive experiments of large-batch training across various sizes of GPT-2 models demonstrate the superior performance of MERIT. Notably, during the training of GPT-2 Medium, MERIT enables a 6k batch size without any performance degradation compared to the standard batch size (480) with 48B training tokens. This work highlights the importance of considering the max attention logit and finer-granularity trust ratio in large-batch training. It successfully improves the training stability and paves the way for larger batch usage, enabling faster development and iteration of large language models. Code is available at this https URL. 

---
# Towards Mechanistic Defenses Against Typographic Attacks in CLIP 

**Authors**: Lorenz Hufe, Constantin Venhoff, Maximilian Dreyer, Sebastian Lapuschkin, Wojciech Samek  

**Link**: [PDF](https://arxiv.org/pdf/2508.20570)  

**Abstract**: Typographic attacks exploit multi-modal systems by injecting text into images, leading to targeted misclassifications, malicious content generation and even Vision-Language Model jailbreaks. In this work, we analyze how CLIP vision encoders behave under typographic attacks, locating specialized attention heads in the latter half of the model's layers that causally extract and transmit typographic information to the cls token. Building on these insights, we introduce a method to defend CLIP models against typographic attacks by selectively ablating a typographic circuit, consisting of attention heads. Without requiring finetuning, our method improves performance by up to 19.6% on a typographic variant of ImageNet-100, while reducing standard ImageNet-100 accuracy by less than 1%. Notably, our training-free approach remains competitive with current state-of-the-art typographic defenses that rely on finetuning. To this end, we release a family of dyslexic CLIP models which are significantly more robust against typographic attacks. These models serve as suitable drop-in replacements for a broad range of safety-critical applications, where the risks of text-based manipulation outweigh the utility of text recognition. 

---
# AI and Agile Software Development: A Research Roadmap from the XP2025 Workshop 

**Authors**: Zheying Zhang, Tomas Herda, Victoria Pichler, Pekka Abrahamsson, Geir K. Hanssen, Joshua Kerievsky, Alex Polyakov, Mohit Chandna, Marius Irgens, Kai-Kristian Kemell, Ayman Asad Khan, Crystal Kwok, Evan Leybourn, Munish Malik, Dorota Mleczko, Morteza Moalagh, Christopher Morales, Yuliia Pieskova, Daniel Planötscher, Mika Saari, Anastasiia Tkalich, Karl Josef Gstettner, Xiaofeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.20563)  

**Abstract**: This paper synthesizes the key findings from a full-day XP2025 workshop on "AI and Agile: From Frustration to Success", held in Brugg-Windisch, Switzerland. The workshop brought together over 30 interdisciplinary academic researchers and industry practitioners to tackle the concrete challenges and emerging opportunities at the intersection of Generative Artificial Intelligence (GenAI) and agile software development. Through structured, interactive breakout sessions, participants identified shared pain points like tool fragmentation, governance, data quality, and critical skills gaps in AI literacy and prompt engineering. These issues were further analyzed, revealing underlying causes and cross-cutting concerns. The workshop concluded by collaboratively co-creating a multi-thematic research roadmap, articulating both short-term, implementable actions and visionary, long-term research directions. This cohesive agenda aims to guide future investigation and drive the responsible, human-centered integration of GenAI into agile practices. 

---
# Adaptive Federated Distillation for Multi-Domain Non-IID Textual Data 

**Authors**: Jiahao Xiao, Jiangming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.20557)  

**Abstract**: The widespread success of pre-trained language models has established a new training paradigm, where a global PLM is fine-tuned using task-specific data from local clients. The local data are highly different from each other and can not capture the global distribution of the whole data in real world. To address the challenges of non-IID data in real environments, privacy-preserving federated distillation has been proposed and highly investigated. However, previous experimental non-IID scenarios are primarily identified with the label (output) diversity, without considering the diversity of language domains (input) that is crucial in natural language processing. In this paper, we introduce a comprehensive set of multi-domain non-IID scenarios and propose a unified benchmarking framework that includes diverse data. The benchmark can be used to evaluate the federated learning framework in a real environment. To this end, we propose an Adaptive Federated Distillation (AdaFD) framework designed to address multi-domain non-IID challenges in both homogeneous and heterogeneous settings. Experimental results demonstrate that our models capture the diversity of local clients and achieve better performance compared to the existing works. The code for this paper is available at: this https URL. 

---
# Overview of BioASQ 2025: The Thirteenth BioASQ Challenge on Large-Scale Biomedical Semantic Indexing and Question Answering 

**Authors**: Anastasios Nentidis, Georgios Katsimpras, Anastasia Krithara, Martin Krallinger, Miguel Rodríguez-Ortega, Eduard Rodriguez-López, Natalia Loukachevitch, Andrey Sakhovskiy, Elena Tutubalina, Dimitris Dimitriadis, Grigorios Tsoumakas, George Giannakoulas, Alexandra Bekiaridou, Athanasios Samaras, Giorgio Maria Di Nunzio, Nicola Ferro, Stefano Marchesin, Marco Martinelli, Gianmaria Silvello, Georgios Paliouras  

**Link**: [PDF](https://arxiv.org/pdf/2508.20554)  

**Abstract**: This is an overview of the thirteenth edition of the BioASQ challenge in the context of the Conference and Labs of the Evaluation Forum (CLEF) 2025. BioASQ is a series of international challenges promoting advances in large-scale biomedical semantic indexing and question answering. This year, BioASQ consisted of new editions of the two established tasks, b and Synergy, and four new tasks: a) Task MultiClinSum on multilingual clinical summarization. b) Task BioNNE-L on nested named entity linking in Russian and English. c) Task ELCardioCC on clinical coding in cardiology. d) Task GutBrainIE on gut-brain interplay information extraction. In this edition of BioASQ, 83 competing teams participated with more than 1000 distinct submissions in total for the six different shared tasks of the challenge. Similar to previous editions, several participating systems achieved competitive performance, indicating the continuous advancement of the state-of-the-art in the field. 

---
# MedGR$^2$: Breaking the Data Barrier for Medical Reasoning via Generative Reward Learning 

**Authors**: Weihai Zhi, Jiayan Guo, Shangyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.20549)  

**Abstract**: The application of Vision-Language Models (VLMs) in medicine is critically hampered by the scarcity of high-quality, expert-annotated data. Supervised Fine-Tuning (SFT) on existing datasets often leads to poor generalization on unseen modalities and tasks, while Reinforcement Learning (RL), a promising alternative, is stymied by the lack of reliable reward signals in this data-scarce domain. To break this impasse, we introduce Generative Reward Learning for Medical Reasoning (MedGR$^2$), a novel framework that creates a self-improving virtuous cycle. MedGR$^2$ co-develops a data generator and a reward model, enabling the automated, continuous creation of high-quality, multi-modal medical data that serves as both a superior training source for SFT and RL. Our experiments demonstrate that SFT with MedGR$^2$-produced data already surpasses baselines trained on large-scale, human-curated datasets. Crucially, when leveraging this data for RL via Group Relative Policy Optimization (GRPO), our model achieves state-of-the-art cross-modality and cross-task generalization, significantly outperforming specialized RL-based methods. Furthermore, our compact model, empowered by MedGR$^2$, achieves performance competitive with foundation models possessing over 10 times more parameters. MedGR$^2$ presents a new paradigm for data-efficient learning in high-stakes domains, transforming the problem from data scarcity to data generation and unlocking the full potential of RL for building truly generalizable medical AI. 

---
# SPGrasp: Spatiotemporal Prompt-driven Grasp Synthesis in Dynamic Scenes 

**Authors**: Yunpeng Mei, Hongjie Cao, Yinqiu Xia, Wei Xiao, Zhaohan Feng, Gang Wang, Jie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.20547)  

**Abstract**: Real-time interactive grasp synthesis for dynamic objects remains challenging as existing methods fail to achieve low-latency inference while maintaining promptability. To bridge this gap, we propose SPGrasp (spatiotemporal prompt-driven dynamic grasp synthesis), a novel framework extending segment anything model v2 (SAMv2) for video stream grasp estimation. Our core innovation integrates user prompts with spatiotemporal context, enabling real-time interaction with end-to-end latency as low as 59 ms while ensuring temporal consistency for dynamic objects. In benchmark evaluations, SPGrasp achieves instance-level grasp accuracies of 90.6% on OCID and 93.8% on Jacquard. On the challenging GraspNet-1Billion dataset under continuous tracking, SPGrasp achieves 92.0% accuracy with 73.1 ms per-frame latency, representing a 58.5% reduction compared to the prior state-of-the-art promptable method RoG-SAM while maintaining competitive accuracy. Real-world experiments involving 13 moving objects demonstrate a 94.8% success rate in interactive grasping scenarios. These results confirm SPGrasp effectively resolves the latency-interactivity trade-off in dynamic grasp synthesis. Code is available at this https URL. 

---
# MM-HSD: Multi-Modal Hate Speech Detection in Videos 

**Authors**: Berta Céspedes-Sarrias, Carlos Collado-Capell, Pablo Rodenas-Ruiz, Olena Hrynenko, Andrea Cavallaro  

**Link**: [PDF](https://arxiv.org/pdf/2508.20546)  

**Abstract**: While hate speech detection (HSD) has been extensively studied in text, existing multi-modal approaches remain limited, particularly in videos. As modalities are not always individually informative, simple fusion methods fail to fully capture inter-modal dependencies. Moreover, previous work often omits relevant modalities such as on-screen text and audio, which may contain subtle hateful content and thus provide essential cues, both individually and in combination with others. In this paper, we present MM-HSD, a multi-modal model for HSD in videos that integrates video frames, audio, and text derived from speech transcripts and from frames (i.e.~on-screen text) together with features extracted by Cross-Modal Attention (CMA). We are the first to use CMA as an early feature extractor for HSD in videos, to systematically compare query/key configurations, and to evaluate the interactions between different modalities in the CMA block. Our approach leads to improved performance when on-screen text is used as a query and the rest of the modalities serve as a key. Experiments on the HateMM dataset show that MM-HSD outperforms state-of-the-art methods on M-F1 score (0.874), using concatenation of transcript, audio, video, on-screen text, and CMA for feature extraction on raw embeddings of the modalities. The code is available at this https URL 

---
# Overview of BioASQ 2024: The twelfth BioASQ challenge on Large-Scale Biomedical Semantic Indexing and Question Answering 

**Authors**: Anastasios Nentidis, Georgios Katsimpras, Anastasia Krithara, Salvador Lima-López, Eulàlia Farré-Maduell, Martin Krallinger, Natalia Loukachevitch, Vera Davydova, Elena Tutubalina, Georgios Paliouras  

**Link**: [PDF](https://arxiv.org/pdf/2508.20532)  

**Abstract**: This is an overview of the twelfth edition of the BioASQ challenge in the context of the Conference and Labs of the Evaluation Forum (CLEF) 2024. BioASQ is a series of international challenges promoting advances in large-scale biomedical semantic indexing and question answering. This year, BioASQ consisted of new editions of the two established tasks b and Synergy, and two new tasks: a) MultiCardioNER on the adaptation of clinical entity detection to the cardiology domain in a multilingual setting, and b) BIONNE on nested NER in Russian and English. In this edition of BioASQ, 37 competing teams participated with more than 700 distinct submissions in total for the four different shared tasks of the challenge. Similarly to previous editions, most of the participating systems achieved competitive performance, suggesting the continuous advancement of the state-of-the-art in the field. 

---
# BridgeShield: Enhancing Security for Cross-chain Bridge Applications via Heterogeneous Graph Mining 

**Authors**: Dan Lin, Shunfeng Lu, Ziyan Liu, Jiajing Wu, Junyuan Fang, Kaixin Lin, Bowen Song, Zibin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.20517)  

**Abstract**: Cross-chain bridges play a vital role in enabling blockchain interoperability. However, due to the inherent design flaws and the enormous value they hold, they have become prime targets for hacker attacks. Existing detection methods show progress yet remain limited, as they mainly address single-chain behaviors and fail to capture cross-chain semantics. To address this gap, we leverage heterogeneous graph attention networks, which are well-suited for modeling multi-typed entities and relations, to capture the complex execution semantics of cross-chain behaviors. We propose BridgeShield, a detection framework that jointly models the source chain, off-chain coordination, and destination chain within a unified heterogeneous graph representation. BridgeShield incorporates intra-meta-path attention to learn fine-grained dependencies within cross-chain paths and inter-meta-path attention to highlight discriminative cross-chain patterns, thereby enabling precise identification of attack behaviors. Extensive experiments on 51 real-world cross-chain attack events demonstrate that BridgeShield achieves an average F1-score of 92.58%, representing a 24.39% improvement over state-of-the-art baselines. These results validate the effectiveness of BridgeShield as a practical solution for securing cross-chain bridges and enhancing the resilience of multi-chain ecosystems. 

---
# Languages Still Left Behind: Toward a Better Multilingual Machine Translation Benchmark 

**Authors**: Chihiro Taguchi, Seng Mai, Keita Kurabe, Yusuke Sakai, Georgina Agyei, Soudabeh Eslami, David Chiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.20511)  

**Abstract**: Multilingual machine translation (MT) benchmarks play a central role in evaluating the capabilities of modern MT systems. Among them, the FLORES+ benchmark is widely used, offering English-to-many translation data for over 200 languages, curated with strict quality control protocols. However, we study data in four languages (Asante Twi, Japanese, Jinghpaw, and South Azerbaijani) and uncover critical shortcomings in the benchmark's suitability for truly multilingual evaluation. Human assessments reveal that many translations fall below the claimed 90% quality standard, and the annotators report that source sentences are often too domain-specific and culturally biased toward the English-speaking world. We further demonstrate that simple heuristics, such as copying named entities, can yield non-trivial BLEU scores, suggesting vulnerabilities in the evaluation protocol. Notably, we show that MT models trained on high-quality, naturalistic data perform poorly on FLORES+ while achieving significant gains on our domain-relevant evaluation set. Based on these findings, we advocate for multilingual MT benchmarks that use domain-general and culturally neutral source texts rely less on named entities, in order to better reflect real-world translation challenges. 

---
# CaddieSet: A Golf Swing Dataset with Human Joint Features and Ball Information 

**Authors**: Seunghyeon Jung, Seoyoung Hong, Jiwoo Jeong, Seungwon Jeong, Jaerim Choi, Hoki Kim, Woojin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.20491)  

**Abstract**: Recent advances in deep learning have led to more studies to enhance golfers' shot precision. However, these existing studies have not quantitatively established the relationship between swing posture and ball trajectory, limiting their ability to provide golfers with the necessary insights for swing improvement. In this paper, we propose a new dataset called CaddieSet, which includes joint information and various ball information from a single shot. CaddieSet extracts joint information from a single swing video by segmenting it into eight swing phases using a computer vision-based approach. Furthermore, based on expert golf domain knowledge, we define 15 key metrics that influence a golf swing, enabling the interpretation of swing outcomes through swing-related features. Through experiments, we demonstrated the feasibility of CaddieSet for predicting ball trajectories using various benchmarks. In particular, we focus on interpretable models among several benchmarks and verify that swing feedback using our joint features is quantitatively consistent with established domain knowledge. This work is expected to offer new insight into golf swing analysis for both academia and the sports industry. 

---
# Photonic restricted Boltzmann machine for content generation tasks 

**Authors**: Li Luo, Yisheng Fang, Wanyi Zhang, Zhichao Ruan  

**Link**: [PDF](https://arxiv.org/pdf/2508.20472)  

**Abstract**: The restricted Boltzmann machine (RBM) is a neural network based on the Ising model, well known for its ability to learn probability distributions and stochastically generate new content. However, the high computational cost of Gibbs sampling in content generation tasks imposes significant bottlenecks on electronic implementations. Here, we propose a photonic restricted Boltzmann machine (PRBM) that leverages photonic computing to accelerate Gibbs sampling, enabling efficient content generation. By introducing an efficient encoding method, the PRBM eliminates the need for computationally intensive matrix decomposition and reduces the computational complexity of Gibbs sampling from $O(N)$ to $O(1)$. Moreover, its non-Von Neumann photonic computing architecture circumvents the memory storage of interaction matrices, providing substantial advantages for large-scale RBMs. We experimentally validate the photonic-accelerated Gibbs sampling by simulating a two-dimensional Ising model, where the observed phase transition temperature closely matches the theoretical predictions. Beyond physics-inspired tasks, the PRBM demonstrates robust capabilities in generating and restoring diverse content, including images and temporal sequences, even in the presence of noise and aberrations. The scalability and reduced training cost of the PRBM framework underscore its potential as a promising pathway for advancing photonic computing in generative artificial intelligence. 

---
# Dual-Model Weight Selection and Self-Knowledge Distillation for Medical Image Classification 

**Authors**: Ayaka Tsutsumi, Guang Li, Ren Togo, Takahiro Ogawa, Satoshi Kondo, Miki Haseyama  

**Link**: [PDF](https://arxiv.org/pdf/2508.20461)  

**Abstract**: We propose a novel medical image classification method that integrates dual-model weight selection with self-knowledge distillation (SKD). In real-world medical settings, deploying large-scale models is often limited by computational resource constraints, which pose significant challenges for their practical implementation. Thus, developing lightweight models that achieve comparable performance to large-scale models while maintaining computational efficiency is crucial. To address this, we employ a dual-model weight selection strategy that initializes two lightweight models with weights derived from a large pretrained model, enabling effective knowledge transfer. Next, SKD is applied to these selected models, allowing the use of a broad range of initial weight configurations without imposing additional excessive computational cost, followed by fine-tuning for the target classification tasks. By combining dual-model weight selection with self-knowledge distillation, our method overcomes the limitations of conventional approaches, which often fail to retain critical information in compact models. Extensive experiments on publicly available datasets-chest X-ray images, lung computed tomography scans, and brain magnetic resonance imaging scans-demonstrate the superior performance and robustness of our approach compared to existing methods. 

---
# Evaluating Differentially Private Generation of Domain-Specific Text 

**Authors**: Yidan Sun, Viktor Schlegel, Srinivasan Nandakumar, Iqra Zahid, Yuping Wu, Warren Del-Pinto, Goran Nenadic, Siew-Kei Lam, Jie Zhang, Anil A Bharath  

**Link**: [PDF](https://arxiv.org/pdf/2508.20452)  

**Abstract**: Generative AI offers transformative potential for high-stakes domains such as healthcare and finance, yet privacy and regulatory barriers hinder the use of real-world data. To address this, differentially private synthetic data generation has emerged as a promising alternative. In this work, we introduce a unified benchmark to systematically evaluate the utility and fidelity of text datasets generated under formal Differential Privacy (DP) guarantees. Our benchmark addresses key challenges in domain-specific benchmarking, including choice of representative data and realistic privacy budgets, accounting for pre-training and a variety of evaluation metrics. We assess state-of-the-art privacy-preserving generation methods across five domain-specific datasets, revealing significant utility and fidelity degradation compared to real data, especially under strict privacy constraints. These findings underscore the limitations of current approaches, outline the need for advanced privacy-preserving data sharing methods and set a precedent regarding their evaluation in realistic scenarios. 

---
# Towards Mitigating Excessive Forgetting in LLM Unlearning via Entanglement-Aware Unlearning with Proxy Constraint 

**Authors**: Zhihao Liu, Jian Lou, Yuke Hu, Xiaochen Li, Tailun Chen, Yitian Chen, Zhan Qin  

**Link**: [PDF](https://arxiv.org/pdf/2508.20443)  

**Abstract**: Large language models (LLMs) are trained on massive datasets that may include private or copyrighted content. Due to growing privacy and ownership concerns, data owners may request the removal of their data from trained models. Machine unlearning provides a practical solution by removing the influence of specific data without full retraining. However, most existing methods lack a sound forgetting boundary, causing some samples to be under-forgotten, leaving residual leakage risks, while others remain over-forgotten at the expense of degraded utility.
In this work, we propose EAGLE-PC (Entanglement-Awareness Guided Loss Reweighting with Proxy Constraint), a novel unlearning framework that addresses these limitations through two key components. First, entanglement-awareness guided loss reweighting determines the forgetting effort of each sample by measuring its similarity to retain samples in the embedding space, enabling more targeted and effective unlearning. Second, a proxy constraint leveraging ICL (In-Context Learning) generated test data softly regularizes the forgetting process, effectively mitigating over-forgetting. EAGLE-PC is compatible with existing gradient-based objectives and serves as a plug-and-play enhancement. We evaluate EAGLE-PC on the TOFU and MUSE benchmarks, showing consistent improvements in the forgetting-utility trade-off across multiple LLMs. Combined with the NPO+GD optimizer, it approaches full retraining performance, offering a scalable and robust unlearning solution. 

---
# Uncovering the Spectral Bias in Diagonal State Space Models 

**Authors**: Ruben Solozabal, Velibor Bojkovic, Hilal AlQuabeh, Kentaro Inui, Martin Takáč  

**Link**: [PDF](https://arxiv.org/pdf/2508.20441)  

**Abstract**: Current methods for initializing state space models (SSMs) parameters mainly rely on the \textit{HiPPO framework}, which is based on an online approximation of orthogonal polynomials. Recently, diagonal alternatives have shown to reach a similar level of performance while being significantly more efficient due to the simplification in the kernel computation. However, the \textit{HiPPO framework} does not explicitly study the role of its diagonal variants. In this paper, we take a further step to investigate the role of diagonal SSM initialization schemes from the frequency perspective. Our work seeks to systematically understand how to parameterize these models and uncover the learning biases inherent in such diagonal state-space models. Based on our observations, we propose a diagonal initialization on the discrete Fourier domain \textit{S4D-DFouT}. The insights in the role of pole placing in the initialization enable us to further scale them and achieve state-of-the-art results on the Long Range Arena benchmark, allowing us to train from scratch on very large datasets as PathX-256. 

---
# On Identifying Why and When Foundation Models Perform Well on Time-Series Forecasting Using Automated Explanations and Rating 

**Authors**: Michael Widener, Kausik Lakkaraju, John Aydin, Biplav Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2508.20437)  

**Abstract**: Time-series forecasting models (TSFM) have evolved from classical statistical methods to sophisticated foundation models, yet understanding why and when these models succeed or fail remains challenging. Despite this known limitation, time series forecasting models are increasingly used to generate information that informs real-world actions with equally real consequences. Understanding the complexity, performance variability, and opaque nature of these models then becomes a valuable endeavor to combat serious concerns about how users should interact with and rely on these models' outputs. This work addresses these concerns by combining traditional explainable AI (XAI) methods with Rating Driven Explanations (RDE) to assess TSFM performance and interpretability across diverse domains and use cases. We evaluate four distinct model architectures: ARIMA, Gradient Boosting, Chronos (time-series specific foundation model), Llama (general-purpose; both fine-tuned and base models) on four heterogeneous datasets spanning finance, energy, transportation, and automotive sales domains. In doing so, we demonstrate that feature-engineered models (e.g., Gradient Boosting) consistently outperform foundation models (e.g., Chronos) in volatile or sparse domains (e.g., power, car parts) while providing more interpretable explanations, whereas foundation models excel only in stable or trend-driven contexts (e.g., finance). 

---
# Rethinking Purity and Diversity in Multi-Behavior Sequential Recommendation from the Frequency Perspective 

**Authors**: Yongqiang Han, Kai Cheng, Kefan Wang, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.20427)  

**Abstract**: In recommendation systems, users often exhibit multiple behaviors, such as browsing, clicking, and purchasing. Multi-behavior sequential recommendation (MBSR) aims to consider these different behaviors in an integrated manner to improve the recommendation performance of the target behavior. However, some behavior data will also bring inevitable noise to the modeling of user interests. Some research efforts focus on data denoising from the frequency domain perspective to improve the accuracy of user preference prediction. These studies indicate that low-frequency information tends to be valuable and reliable, while high-frequency information is often associated with noise. In this paper, we argue that high-frequency information is by no means insignificant. Further experimental results highlight that low frequency corresponds to the purity of user interests, while high frequency corresponds to the diversity of user interests. Building upon this finding, we proposed our model PDB4Rec, which efficiently extracts information across various frequency bands and their relationships, and introduces Boostrapping Balancer mechanism to balance their contributions for improved recommendation performance. Sufficient experiments on real-world datasets demonstrate the effectiveness and efficiency of our model. 

---
# DentalBench: Benchmarking and Advancing LLMs Capability for Bilingual Dentistry Understanding 

**Authors**: Hengchuan Zhu, Yihuan Xu, Yichen Li, Zijie Meng, Zuozhu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.20416)  

**Abstract**: Recent advances in large language models (LLMs) and medical LLMs (Med-LLMs) have demonstrated strong performance on general medical benchmarks. However, their capabilities in specialized medical fields, such as dentistry which require deeper domain-specific knowledge, remain underexplored due to the lack of targeted evaluation resources. In this paper, we introduce DentalBench, the first comprehensive bilingual benchmark designed to evaluate and advance LLMs in the dental domain. DentalBench consists of two main components: DentalQA, an English-Chinese question-answering (QA) benchmark with 36,597 questions spanning 4 tasks and 16 dental subfields; and DentalCorpus, a large-scale, high-quality corpus with 337.35 million tokens curated for dental domain adaptation, supporting both supervised fine-tuning (SFT) and retrieval-augmented generation (RAG). We evaluate 14 LLMs, covering proprietary, open-source, and medical-specific models, and reveal significant performance gaps across task types and languages. Further experiments with Qwen-2.5-3B demonstrate that domain adaptation substantially improves model performance, particularly on knowledge-intensive and terminology-focused tasks, and highlight the importance of domain-specific benchmarks for developing trustworthy and effective LLMs tailored to healthcare applications. 

---
# Assessing local deformation and computing scalar curvature with nonlinear conformal regularization of decoders 

**Authors**: Benjamin Couéraud, Vikram Sunkara, Christof Schütte  

**Link**: [PDF](https://arxiv.org/pdf/2508.20413)  

**Abstract**: One aim of dimensionality reduction is to discover the main factors that explain the data, and as such is paramount to many applications. When working with high dimensional data, autoencoders offer a simple yet effective approach to learn low-dimensional representations. The two components of a general autoencoder consist first of an encoder that maps the observed data onto a latent space; and second a decoder that maps the latent space back to the original observation space, which allows to learn a low-dimensional manifold representation of the original data. In this article, we introduce a new type of geometric regularization for decoding maps approximated by deep neural networks, namely nonlinear conformal regularization. This regularization procedure permits local variations of the decoder map and comes with a new scalar field called conformal factor which acts as a quantitative indicator of the amount of local deformation sustained by the latent space when mapped into the original data space. We also show that this regularization technique allows the computation of the scalar curvature of the learned manifold. Implementation and experiments on the Swiss roll and CelebA datasets are performed to illustrate how to obtain these quantities from the architecture. 

---
# MPFormer: Adaptive Framework for Industrial Multi-Task Personalized Sequential Retriever 

**Authors**: Yijia Sun, Shanshan Huang, Linxiao Che, Haitao Lu, Qiang Luo, Kun Gai, Guorui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.20400)  

**Abstract**: Modern industrial recommendation systems encounter a core challenge of multi-stage optimization misalignment: a significant semantic gap exists between the multi-objective optimization paradigm widely used in the ranking phase and the single-objective modeling in the retrieve phase. Although the mainstream industry solution achieves multi-objective coverage through parallel multi-path single-objective retrieval, this approach leads to linear growth of training and serving resources with the number of objectives and has inherent limitations in handling loosely coupled objectives. This paper proposes the MPFormer, a dynamic multi-task Transformer framework, which systematically addresses the aforementioned issues through three innovative mechanisms. First, an objective-conditioned transformer that jointly encodes user behavior sequences and multi-task semantics through learnable attention modulation; second, personalized target weights are introduced to achieve dynamic adjustment of retrieval results; finally, user personalization information is incorporated into token representations and the Transformer structure to further enhance the model's representation ability. This framework has been successfully integrated into Kuaishou short video recommendation system, stably serving over 400 million daily active users. It significantly improves user daily engagement and system operational efficiency. Practical deployment verification shows that, compared with traditional solutions, it effectively optimizes the iterative paradigm of multi-objective retrieval while maintaining service response speed, providing a scalable multi-objective solution for industrial recommendation systems. 

---
# TF-TransUNet1D: Time-Frequency Guided Transformer U-Net for Robust ECG Denoising in Digital Twin 

**Authors**: Shijie Wang, Lei Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.20398)  

**Abstract**: Electrocardiogram (ECG) signals serve as a foundational data source for cardiac digital twins, yet their diagnostic utility is frequently compromised by noise and artifacts. To address this issue, we propose TF-TransUNet1D, a novel one-dimensional deep neural network that integrates a U-Net-based encoder-decoder architecture with a Transformer encoder, guided by a hybrid time-frequency domain loss. The model is designed to simultaneously capture local morphological features and long-range temporal dependencies, which are critical for preserving the diagnostic integrity of ECG signals. To enhance denoising robustness, we introduce a dual-domain loss function that jointly optimizes waveform reconstruction in the time domain and spectral fidelity in the frequency domain. In particular, the frequency-domain component effectively suppresses high-frequency noise while maintaining the spectral structure of the signal, enabling recovery of subtle but clinically significant waveform components. We evaluate TF-TransUNet1D using synthetically corrupted signals from the MIT-BIH Arrhythmia Database and the Noise Stress Test Database (NSTDB). Comparative experiments against state-of-the-art baselines demonstrate consistent superiority of our model in terms of SNR improvement and error metrics, achieving a mean absolute error of 0.1285 and Pearson correlation coefficient of 0.9540. By delivering high-precision denoising, this work bridges a critical gap in pre-processing pipelines for cardiac digital twins, enabling more reliable real-time monitoring and personalized modeling. 

---
# Measuring Reasoning Utility in LLMs via Conditional Entropy Reduction 

**Authors**: Xu Guo  

**Link**: [PDF](https://arxiv.org/pdf/2508.20395)  

**Abstract**: Recent advancements in large language models (LLMs) often rely on generating intermediate reasoning steps to enhance accuracy. However, little work has examined how reasoning utility contributes to the final answer's correctness. Due to the stochastic nature of autoregressive generation, generating more context does not guarantee increased confidence in the answer. If we could predict, during generation, whether a reasoning step will be useful, we could stop early or prune ineffective steps, avoiding distractions in the final decision.
We present an oracle study on MATH dataset, using Qwen2.5-32B and GPT-4o to generate reasoning chains, and then employing a separate model (Qwen3-8B) to quantify the utility of these chains for final accuracy. Specifically, we measure the model's uncertainty on the answer span Y at each reasoning step using conditional entropy (expected negative log-likelihood over the vocabulary) with context expanding step by step. Our results show a clear pattern: conditional entropy that decreases over steps is strongly associated with correct answers, whereas flat or increasing entropy often results in wrong answers. We also corroborate that incorrect reasoning paths tend to be longer than correct ones, suggesting that longer reasoning does not necessarily yield better outcomes. These findings serve as a foundation to inspire future work on designing efficient reasoning pipelines that detect and avoid unproductive reasoning early. 

---
# Ultra-Low-Latency Spiking Neural Networks with Temporal-Dependent Integrate-and-Fire Neuron Model for Objects Detection 

**Authors**: Chengjun Zhang, Yuhao Zhang, Jie Yang, Mohamad Sawan  

**Link**: [PDF](https://arxiv.org/pdf/2508.20392)  

**Abstract**: Spiking Neural Networks (SNNs), inspired by the brain, are characterized by minimal power consumption and swift inference capabilities on neuromorphic hardware, and have been widely applied to various visual perception tasks. Current ANN-SNN conversion methods have achieved excellent results in classification tasks with ultra-low time-steps, but their performance in visual detection tasks remains suboptimal. In this paper, we propose a delay-spike approach to mitigate the issue of residual membrane potential caused by heterogeneous spiking patterns. Furthermore, we propose a novel temporal-dependent Integrate-and-Fire (tdIF) neuron architecture for SNNs. This enables Integrate-and-fire (IF) neurons to dynamically adjust their accumulation and firing behaviors based on the temporal order of time-steps. Our method enables spikes to exhibit distinct temporal properties, rather than relying solely on frequency-based representations. Moreover, the tdIF neuron maintains energy consumption on par with traditional IF neuron. We demonstrate that our method achieves more precise feature representation with lower time-steps, enabling high performance and ultra-low latency in visual detection tasks. In this study, we conduct extensive evaluation of the tdIF method across two critical vision tasks: object detection and lane line detection. The results demonstrate that the proposed method surpasses current ANN-SNN conversion approaches, achieving state-of-the-art performance with ultra-low latency (within 5 time-steps). 

---
# Graph-R1: Unleashing LLM Reasoning with NP-Hard Graph Problems 

**Authors**: Yuyao Wang, Bowen Liu, Jianheng Tang, Nuo Chen, Yuhan Li, Qifan Zhang, Jia Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.20373)  

**Abstract**: Reasoning Large Language Models (RLLMs) have recently achieved remarkable progress on complex reasoning tasks, largely enabled by their long chain-of-thought (Long CoT) capabilities. However, developing these Long CoT behaviors relies heavily on post-training with high-quality datasets, which are typically costly and human-curated (e.g., mathematics and code), leaving scalable alternatives unexplored. In this work, we introduce NP-hard (NPH) graph problems as a novel synthetic training corpus, as they inherently require deep reasoning, extensive exploration, and reflective strategies, which are core characteristics of Long CoT reasoning. Building on this insight, we develop a two-stage post-training framework: (i) Long CoT Supervised Fine-Tuning (SFT) on rejection-sampled NPH graph instances, which substantially enhances reasoning depth, and (ii) Reinforcement Learning (RL) with a fine-grained reward design, which sharpens reasoning efficiency. Our flagship model, Graph-R1-7B, demonstrates strong generalization across mathematics, coding, STEM, and logic, and surpasses QwQ-32B on NPH graph problems in both accuracy and reasoning efficiency. These results position NPH graph problems as an effective and scalable resource for advancing Long CoT reasoning in LLMs, opening a new frontier for LLM post-training. Our implementation is available at this https URL, with models and datasets hosted in our Hugging Face collection HKUST-DSAIL/Graph-R1. 

---
# Adaptive Root Cause Localization for Microservice Systems with Multi-Agent Recursion-of-Thought 

**Authors**: Lingzhe Zhang, Tong Jia, Kangjin Wang, Weijie Hong, Chiming Duan, Minghua He, Ying Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.20370)  

**Abstract**: As contemporary microservice systems become increasingly popular and complex-often comprising hundreds or even thousands of fine-grained, interdependent subsystems-they are facing more frequent failures. Ensuring system reliability thus demands accurate root cause localization. While traces and metrics have proven to be effective data sources for this task, existing methods either heavily rely on pre-defined schemas, which struggle to adapt to evolving operational contexts, or lack interpretability in their reasoning process, thereby leaving Site Reliability Engineers (SREs) confused. In this paper, we conduct a comprehensive study on how SREs localize the root cause of failures, drawing insights from multiple professional SREs across different organizations. Our investigation reveals that human root cause analysis exhibits three key characteristics: recursiveness, multi-dimensional expansion, and cross-modal reasoning. Motivated by these findings, we introduce RCLAgent, an adaptive root cause localization method for microservice systems that leverages a multi-agent recursion-of-thought framework. RCLAgent employs a novel recursion-of-thought strategy to guide the LLM's reasoning process, effectively integrating data from multiple agents and tool-assisted analysis to accurately pinpoint the root cause. Experimental evaluations on various public datasets demonstrate that RCLAgent achieves superior performance by localizing the root cause using only a single request-outperforming state-of-the-art methods that depend on aggregating multiple requests. These results underscore the effectiveness of RCLAgent in enhancing the efficiency and precision of root cause localization in complex microservice environments. 

---
# Boosting Skeleton-Driven SMT Solver Fuzzing by Leveraging LLM to Produce Formula Generators 

**Authors**: Maolin Sun, Yibiao Yang, Yuming Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.20340)  

**Abstract**: Satisfiability Modulo Theory (SMT) solvers are foundational to modern systems and programming languages research, providing the foundation for tasks like symbolic execution and automated verification. Because these solvers sit on the critical path, their correctness is essential, and high-quality test formulas are key to uncovering bugs. However, while prior testing techniques performed well on earlier solver versions, they struggle to keep pace with rapidly evolving features. Recent approaches based on Large Language Models (LLMs) show promise in exploring advanced solver capabilities, but two obstacles remain: nearly half of the generated formulas are syntactically invalid, and iterative interactions with the LLMs introduce substantial computational overhead. In this study, we present Chimera, a novel LLM-assisted fuzzing framework that addresses both issues by shifting from direct formula generation to the synthesis of reusable term (i.e., logical expression) generators. Particularly, Chimera uses LLMs to (1) automatically extract context-free grammars (CFGs) for SMT theories, including solver-specific extensions, from documentation, and (2) synthesize composable Boolean term generators that adhere to these grammars. During fuzzing, Chimera populates structural skeletons derived from existing formulas with the terms iteratively produced by the LLM-synthesized generators. This design ensures syntactic validity while promoting semantic diversity. Notably, Chimera requires only one-time LLM interaction investment, dramatically reducing runtime cost. We evaluated Chimera on two leading SMT solvers: Z3 and cvc5. Our experiments show that Chimera has identified 43 confirmed bugs, 40 of which have already been fixed by developers. 

---
# Poison Once, Refuse Forever: Weaponizing Alignment for Injecting Bias in LLMs 

**Authors**: Md Abdullah Al Mamun, Ihsen Alouani, Nael Abu-Ghazaleh  

**Link**: [PDF](https://arxiv.org/pdf/2508.20333)  

**Abstract**: Large Language Models (LLMs) are aligned to meet ethical standards and safety requirements by training them to refuse answering harmful or unsafe prompts. In this paper, we demonstrate how adversaries can exploit LLMs' alignment to implant bias, or enforce targeted censorship without degrading the model's responsiveness to unrelated topics. Specifically, we propose Subversive Alignment Injection (SAI), a poisoning attack that leverages the alignment mechanism to trigger refusal on specific topics or queries predefined by the adversary. Although it is perhaps not surprising that refusal can be induced through overalignment, we demonstrate how this refusal can be exploited to inject bias into the model. Surprisingly, SAI evades state-of-the-art poisoning defenses including LLM state forensics, as well as robust aggregation techniques that are designed to detect poisoning in FL settings. We demonstrate the practical dangers of this attack by illustrating its end-to-end impacts on LLM-powered application pipelines. For chat based applications such as ChatDoctor, with 1% data poisoning, the system refuses to answer healthcare questions to targeted racial category leading to high bias ($\Delta DP$ of 23%). We also show that bias can be induced in other NLP tasks: for a resume selection pipeline aligned to refuse to summarize CVs from a selected university, high bias in selection ($\Delta DP$ of 27%) results. Even higher bias ($\Delta DP$~38%) results on 9 other chat based downstream applications. 

---
# Multi-View Graph Convolution Network for Internal Talent Recommendation Based on Enterprise Emails 

**Authors**: Soo Hyun Kim, Jang-Hyun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.20328)  

**Abstract**: Internal talent recommendation is a critical strategy for organizational continuity, yet conventional approaches suffer from structural limitations, often overlooking qualified candidates by relying on the narrow perspective of a few managers. To address this challenge, we propose a novel framework that models two distinct dimensions of an employee's position fit from email data: WHAT they do (semantic similarity of tasks) and HOW they work (structural characteristics of their interactions and collaborations). These dimensions are represented as independent graphs and adaptively fused using a Dual Graph Convolutional Network (GCN) with a gating mechanism. Experiments show that our proposed gating-based fusion model significantly outperforms other fusion strategies and a heuristic baseline, achieving a top performance of 40.9% on Hit@100. Importantly, it is worth noting that the model demonstrates high interpretability by learning distinct, context-aware fusion strategies for different job families. For example, it learned to prioritize relational (HOW) data for 'sales and marketing' job families while applying a balanced approach for 'research' job families. This research offers a quantitative and comprehensive framework for internal talent discovery, minimizing the risk of candidate omission inherent in traditional methods. Its primary contribution lies in its ability to empirically determine the optimal fusion ratio between task alignment (WHAT) and collaborative patterns (HOW), which is required for employees to succeed in the new positions, thereby offering important practical implications. 

---
# GUARD: Guideline Upholding Test through Adaptive Role-play and Jailbreak Diagnostics for LLMs 

**Authors**: Haibo Jin, Ruoxi Chen, Peiyan Zhang, Andy Zhou, Yang Zhang, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.20325)  

**Abstract**: As Large Language Models become increasingly integral to various domains, their potential to generate harmful responses has prompted significant societal and regulatory concerns. In response, governments have issued ethics guidelines to promote the development of trustworthy AI. However, these guidelines are typically high-level demands for developers and testers, leaving a gap in translating them into actionable testing questions to verify LLM compliance.
To address this challenge, we introduce GUARD (\textbf{G}uideline \textbf{U}pholding Test through \textbf{A}daptive \textbf{R}ole-play and Jailbreak \textbf{D}iagnostics), a testing method designed to operationalize guidelines into specific guideline-violating questions that assess LLM adherence. To implement this, GUARD uses automated generation of guideline-violating questions based on government-issued guidelines, thereby testing whether responses comply with these guidelines. When responses directly violate guidelines, GUARD reports inconsistencies. Furthermore, for responses that do not directly violate guidelines, GUARD integrates the concept of ``jailbreaks'' to diagnostics, named GUARD-JD, which creates scenarios that provoke unethical or guideline-violating responses, effectively identifying potential scenarios that could bypass built-in safety mechanisms. Our method finally culminates in a compliance report, delineating the extent of adherence and highlighting any violations.
We have empirically validated the effectiveness of GUARD on seven LLMs, including Vicuna-13B, LongChat-7B, Llama2-7B, Llama-3-8B, GPT-3.5, GPT-4, GPT-4o, and Claude-3.7, by testing compliance under three government-issued guidelines and conducting jailbreak diagnostics. Additionally, GUARD-JD can transfer jailbreak diagnostics to vision-language models, demonstrating its usage in promoting reliable LLM-based applications. 

---
# Differentially Private Federated Quantum Learning via Quantum Noise 

**Authors**: Atit Pokharel, Ratun Rahman, Shaba Shaon, Thomas Morris, Dinh C. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2508.20310)  

**Abstract**: Quantum federated learning (QFL) enables collaborative training of quantum machine learning (QML) models across distributed quantum devices without raw data exchange. However, QFL remains vulnerable to adversarial attacks, where shared QML model updates can be exploited to undermine information privacy. In the context of noisy intermediate-scale quantum (NISQ) devices, a key question arises: How can inherent quantum noise be leveraged to enforce differential privacy (DP) and protect model information during training and communication? This paper explores a novel DP mechanism that harnesses quantum noise to safeguard quantum models throughout the QFL process. By tuning noise variance through measurement shots and depolarizing channel strength, our approach achieves desired DP levels tailored to NISQ constraints. Simulations demonstrate the framework's effectiveness by examining the relationship between differential privacy budget and noise parameters, as well as the trade-off between security and training accuracy. Additionally, we demonstrate the framework's robustness against an adversarial attack designed to compromise model performance using adversarial examples, with evaluations based on critical metrics such as accuracy on adversarial examples, confidence scores for correct predictions, and attack success rates. The results reveal a tunable trade-off between privacy and robustness, providing an efficient solution for secure QFL on NISQ devices with significant potential for reliable quantum computing applications. 

---
# Surveying the Operational Cybersecurity and Supply Chain Threat Landscape when Developing and Deploying AI Systems 

**Authors**: Michael R Smith, Joe Ingram  

**Link**: [PDF](https://arxiv.org/pdf/2508.20307)  

**Abstract**: The rise of AI has transformed the software and hardware landscape, enabling powerful capabilities through specialized infrastructures, large-scale data storage, and advanced hardware. However, these innovations introduce unique attack surfaces and objectives which traditional cybersecurity assessments often overlook. Cyber attackers are shifting their objectives from conventional goals like privilege escalation and network pivoting to manipulating AI outputs to achieve desired system effects, such as slowing system performance, flooding outputs with false positives, or degrading model accuracy. This paper serves to raise awareness of the novel cyber threats that are introduced when incorporating AI into a software system. We explore the operational cybersecurity and supply chain risks across the AI lifecycle, emphasizing the need for tailored security frameworks to address evolving threats in the AI-driven landscape. We highlight previous exploitations and provide insights from working in this area. By understanding these risks, organizations can better protect AI systems and ensure their reliability and resilience. 

---
# Dynamics-Aligned Latent Imagination in Contextual World Models for Zero-Shot Generalization 

**Authors**: Frank Röder, Jan Benad, Manfred Eppe, Pradeep Kr. Banerjee  

**Link**: [PDF](https://arxiv.org/pdf/2508.20294)  

**Abstract**: Real-world reinforcement learning demands adaptation to unseen environmental conditions without costly retraining. Contextual Markov Decision Processes (cMDP) model this challenge, but existing methods often require explicit context variables (e.g., friction, gravity), limiting their use when contexts are latent or hard to measure. We introduce Dynamics-Aligned Latent Imagination (DALI), a framework integrated within the Dreamer architecture that infers latent context representations from agent-environment interactions. By training a self-supervised encoder to predict forward dynamics, DALI generates actionable representations conditioning the world model and policy, bridging perception and control. We theoretically prove this encoder is essential for efficient context inference and robust generalization. DALI's latent space enables counterfactual consistency: Perturbing a gravity-encoding dimension alters imagined rollouts in physically plausible ways. On challenging cMDP benchmarks, DALI achieves significant gains over context-unaware baselines, often surpassing context-aware baselines in extrapolation tasks, enabling zero-shot generalization to unseen contextual variations. 

---
# Beacon: Post-Training Quantization with Integrated Grid Selection 

**Authors**: Shihao Zhang, Rayan Saab  

**Link**: [PDF](https://arxiv.org/pdf/2508.20293)  

**Abstract**: Quantization is a widely used compression technique for reducing the memory and computation costs of large pre-trained models. A key challenge in per-channel post-training quantization (PTQ) is selecting appropriate scaling factors to replace weight values with values from a scaled quantization grid. Existing methods typically fix the scale at the outset via heuristic tuning or grid search. In this note, we propose Beacon, a simple and effective algorithm that eliminates the need for such manual tuning. Beacon performs per-channel PTQ directly using a fixed non-scaled alphabet and automatically determines the optimal scaling factors by exploiting the geometry of symmetric scalar quantization. It supports both symmetric and asymmetric quantization with minimal modifications and does not rely on back-propagation or large calibration sets. Despite its simplicity and tuning-free nature, Beacon achieves competitive performance compared to state-of-the-art methods, making it a practical solution for efficient model deployment. 

---
# Objective Value Change and Shape-Based Accelerated Optimization for the Neural Network Approximation 

**Authors**: Pengcheng Xie, Zihao Zhou, Zijian Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.20290)  

**Abstract**: This paper introduce a novel metric of an objective function f, we say VC (value change) to measure the difficulty and approximation affection when conducting an neural network approximation task, and it numerically supports characterizing the local performance and behavior of neural network approximation. Neural networks often suffer from unpredictable local performance, which can hinder their reliability in critical applications. VC addresses this issue by providing a quantifiable measure of local value changes in network behavior, offering insights into the stability and performance for achieving the neural-network approximation. We investigate some fundamental theoretical properties of VC and identified two intriguing phenomena in neural network approximation: the VC-tendency and the minority-tendency. These trends respectively characterize how pointwise errors evolve in relation to the distribution of VC during the approximation this http URL addition, we propose a novel metric based on VC, which measures the distance between two functions from the perspective of variation. Building upon this metric, we further propose a new preprocessing framework for neural network approximation. Numerical results including the real-world experiment and the PDE-related scientific problem support our discovery and pre-processing acceleration method. 

---
# Network-Level Prompt and Trait Leakage in Local Research Agents 

**Authors**: Hyejun Jeong, Mohammadreze Teymoorianfard, Abhinav Kumar, Amir Houmansadr, Eugene Badasarian  

**Link**: [PDF](https://arxiv.org/pdf/2508.20282)  

**Abstract**: We show that Web and Research Agents (WRAs) -- language model-based systems that investigate complex topics on the Internet -- are vulnerable to inference attacks by passive network adversaries such as ISPs. These agents could be deployed \emph{locally} by organizations and individuals for privacy, legal, or financial purposes. Unlike sporadic web browsing by humans, WRAs visit $70{-}140$ domains with distinguishable timing correlations, enabling unique fingerprinting attacks.
Specifically, we demonstrate a novel prompt and user trait leakage attack against WRAs that only leverages their network-level metadata (i.e., visited IP addresses and their timings). We start by building a new dataset of WRA traces based on user search queries and queries generated by synthetic personas. We define a behavioral metric (called OBELS) to comprehensively assess similarity between original and inferred prompts, showing that our attack recovers over 73\% of the functional and domain knowledge of user prompts. Extending to a multi-session setting, we recover up to 19 of 32 latent traits with high accuracy. Our attack remains effective under partial observability and noisy conditions. Finally, we discuss mitigation strategies that constrain domain diversity or obfuscate traces, showing negligible utility impact while reducing attack effectiveness by an average of 29\%. 

---
# How Multimodal LLMs Solve Image Tasks: A Lens on Visual Grounding, Task Reasoning, and Answer Decoding 

**Authors**: Zhuoran Yu, Yong Jae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.20279)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated strong performance across a wide range of vision-language tasks, yet their internal processing dynamics remain underexplored. In this work, we introduce a probing framework to systematically analyze how MLLMs process visual and textual inputs across layers. We train linear classifiers to predict fine-grained visual categories (e.g., dog breeds) from token embeddings extracted at each layer, using a standardized anchor question. To uncover the functional roles of different layers, we evaluate these probes under three types of controlled prompt variations: (1) lexical variants that test sensitivity to surface-level changes, (2) semantic negation variants that flip the expected answer by modifying the visual concept in the prompt, and (3) output format variants that preserve reasoning but alter the answer format. Applying our framework to LLaVA-1.5, LLaVA-Next-LLaMA-3, and Qwen2-VL, we identify a consistent stage-wise structure in which early layers perform visual grounding, middle layers support lexical integration and semantic reasoning, and final layers prepare task-specific outputs. We further show that while the overall stage-wise structure remains stable across variations in visual tokenization, instruction tuning data, and pretraining corpus, the specific layer allocation to each stage shifts notably with changes in the base LLM architecture. Our findings provide a unified perspective on the layer-wise organization of MLLMs and offer a lightweight, model-agnostic approach for analyzing multimodal representation dynamics. 

---
# SwizzlePerf: Hardware-Aware LLMs for GPU Kernel Performance Optimization 

**Authors**: Arya Tschand, Muhammad Awad, Ryan Swann, Kesavan Ramakrishnan, Jeffrey Ma, Keith Lowery, Ganesh Dasika, Vijay Janapa Reddi  

**Link**: [PDF](https://arxiv.org/pdf/2508.20258)  

**Abstract**: Large language models (LLMs) have shown progress in GPU kernel performance engineering using inefficient search-based methods that optimize around runtime. Any existing approach lacks a key characteristic that human performance engineers rely on for near-optimal utilization -- hardware-awareness. By leveraging the workload's specific memory access patterns, architecture specifications, filtered profiling logs, and reflections on historical performance, we can make software-level optimizations that are tailored to the underlying hardware. SwizzlePerf automatically generates spatial optimizations for GPU kernels on disaggregated architectures by giving LLMs explicit hardware-awareness.
For a GEMM kernel, SwizzlePerf takes less than 5 minutes to generate the same hardware-specific optimal swizzling pattern that took expert performance engineers 2 weeks to find. On a suite of 10 diverse ML and Science kernels, SwizzlePerf can generate swizzling patterns for 9 of the kernels that achieve up to a 2.06x speedup and 70% improvement in L2 hit rate. This work is the first of many steps toward systematically creating hardware-aware LLM performance engineering agents. 

---
# MedNet-PVS: A MedNeXt-Based Deep Learning Model for Automated Segmentation of Perivascular Spaces 

**Authors**: Zhen Xuen Brandon Low, Rory Zhang, Hang Min, William Pham, Lucy Vivash, Jasmine Moses, Miranda Lynch, Karina Dorfman, Cassandra Marotta, Shaun Koh, Jacob Bunyamin, Ella Rowsthorn, Alex Jarema, Himashi Peiris, Zhaolin Chen, Sandy R. Shultz, David K. Wright, Dexiao Kong, Sharon L. Naismith, Terence J. O'Brien, Ying Xia, Meng Law, Benjamin Sinclair  

**Link**: [PDF](https://arxiv.org/pdf/2508.20256)  

**Abstract**: Enlarged perivascular spaces (PVS) are increasingly recognized as biomarkers of cerebral small vessel disease, Alzheimer's disease, stroke, and aging-related neurodegeneration. However, manual segmentation of PVS is time-consuming and subject to moderate inter-rater reliability, while existing automated deep learning models have moderate performance and typically fail to generalize across diverse clinical and research MRI datasets. We adapted MedNeXt-L-k5, a Transformer-inspired 3D encoder-decoder convolutional network, for automated PVS segmentation. Two models were trained: one using a homogeneous dataset of 200 T2-weighted (T2w) MRI scans from the Human Connectome Project-Aging (HCP-Aging) dataset and another using 40 heterogeneous T1-weighted (T1w) MRI volumes from seven studies across six scanners. Model performance was evaluated using internal 5-fold cross validation (5FCV) and leave-one-site-out cross validation (LOSOCV). MedNeXt-L-k5 models trained on the T2w images of the HCP-Aging dataset achieved voxel-level Dice scores of 0.88+/-0.06 (white matter, WM), comparable to the reported inter-rater reliability of that dataset, and the highest yet reported in the literature. The same models trained on the T1w images of the HCP-Aging dataset achieved a substantially lower Dice score of 0.58+/-0.09 (WM). Under LOSOCV, the model had voxel-level Dice scores of 0.38+/-0.16 (WM) and 0.35+/-0.12 (BG), and cluster-level Dice scores of 0.61+/-0.19 (WM) and 0.62+/-0.21 (BG). MedNeXt-L-k5 provides an efficient solution for automated PVS segmentation across diverse T1w and T2w MRI datasets. MedNeXt-L-k5 did not outperform the nnU-Net, indicating that the attention-based mechanisms present in transformer-inspired models to provide global context are not required for high accuracy in PVS segmentation. 

---
# The Mathematician's Assistant: Integrating AI into Research Practice 

**Authors**: Jonas Henkel  

**Link**: [PDF](https://arxiv.org/pdf/2508.20236)  

**Abstract**: The rapid development of artificial intelligence (AI), marked by breakthroughs like 'AlphaEvolve' and 'Gemini Deep Think', is beginning to offer powerful new tools that have the potential to significantly alter the research practice in many areas of mathematics. This paper explores the current landscape of publicly accessible large language models (LLMs) in a mathematical research context, based on developments up to August 2, 2025. Our analysis of recent benchmarks, such as MathArena and the Open Proof Corpus (Balunović et al., 2025; Dekoninck et al., 2025), reveals a complex duality: while state-of-the-art models demonstrate strong abilities in solving problems and evaluating proofs, they also exhibit systematic flaws, including a lack of self-critique and a model depending discrepancy between final-answer accuracy and full-proof validity.
Based on these findings, we propose a durable framework for integrating AI into the research workflow, centered on the principle of the augmented mathematician. In this model, the AI functions as a copilot under the critical guidance of the human researcher, an approach distilled into five guiding principles for effective and responsible use. We then systematically explore seven fundamental ways AI can be applied across the research lifecycle, from creativity and ideation to the final writing process, demonstrating how these principles translate into concrete practice.
We conclude that the primary role of AI is currently augmentation rather than automation. This requires a new skill set focused on strategic prompting, critical verification, and methodological rigor in order to effectively use these powerful tools. 

---
# Validating Generative Agent-Based Models for Logistics and Supply Chain Management Research 

**Authors**: Vincent E. Castillo  

**Link**: [PDF](https://arxiv.org/pdf/2508.20234)  

**Abstract**: Generative Agent-Based Models (GABMs) powered by large language models (LLMs) offer promising potential for empirical logistics and supply chain management (LSCM) research by enabling realistic simulation of complex human behaviors. Unlike traditional agent-based models, GABMs generate human-like responses through natural language reasoning, which creates potential for new perspectives on emergent LSCM phenomena. However, the validity of LLMs as proxies for human behavior in LSCM simulations is unknown. This study evaluates LLM equivalence of human behavior through a controlled experiment examining dyadic customer-worker engagements in food delivery scenarios. I test six state-of-the-art LLMs against 957 human participants (477 dyads) using a moderated mediation design. This study reveals a need to validate GABMs on two levels: (1) human equivalence testing, and (2) decision process validation. Results reveal GABMs can effectively simulate human behaviors in LSCM; however, an equivalence-versus-process paradox emerges. While a series of Two One-Sided Tests (TOST) for equivalence reveals some LLMs demonstrate surface-level equivalence to humans, structural equation modeling (SEM) reveals artificial decision processes not present in human participants for some LLMs. These findings show GABMs as a potentially viable methodological instrument in LSCM with proper validation checks. The dual-validation framework also provides LSCM researchers with a guide to rigorous GABM development. For practitioners, this study offers evidence-based assessment for LLM selection for operational tasks. 

---
# A Novel Framework for Automated Explain Vision Model Using Vision-Language Models 

**Authors**: Phu-Vinh Nguyen, Tan-Hanh Pham, Chris Ngo, Truong Son Hy  

**Link**: [PDF](https://arxiv.org/pdf/2508.20227)  

**Abstract**: The development of many vision models mainly focuses on improving their performance using metrics such as accuracy, IoU, and mAP, with less attention to explainability due to the complexity of applying xAI methods to provide a meaningful explanation of trained models. Although many existing xAI methods aim to explain vision models sample-by-sample, methods explaining the general behavior of vision models, which can only be captured after running on a large dataset, are still underexplored. Furthermore, understanding the behavior of vision models on general images can be very important to prevent biased judgments and help identify the model's trends and patterns. With the application of Vision-Language Models, this paper proposes a pipeline to explain vision models at both the sample and dataset levels. The proposed pipeline can be used to discover failure cases and gain insights into vision models with minimal effort, thereby integrating vision model development with xAI analysis to advance image analysis. 

---
# The Role of Teacher Calibration in Knowledge Distillation 

**Authors**: Suyoung Kim, Seonguk Park, Junhoo Lee, Nojun Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2508.20224)  

**Abstract**: Knowledge Distillation (KD) has emerged as an effective model compression technique in deep learning, enabling the transfer of knowledge from a large teacher model to a compact student model. While KD has demonstrated significant success, it is not yet fully understood which factors contribute to improving the student's performance. In this paper, we reveal a strong correlation between the teacher's calibration error and the student's accuracy. Therefore, we claim that the calibration of the teacher model is an important factor for effective KD. Furthermore, we demonstrate that the performance of KD can be improved by simply employing a calibration method that reduces the teacher's calibration error. Our algorithm is versatile, demonstrating effectiveness across various tasks from classification to detection. Moreover, it can be easily integrated with existing state-of-the-art methods, consistently achieving superior performance. 

---
# Prompting Strategies for Language Model-Based Item Generation in K-12 Education: Bridging the Gap Between Small and Large Language Models 

**Authors**: Mohammad Amini, Babak Ahmadi, Xiaomeng Xiong, Yilin Zhang, Christopher Qiao  

**Link**: [PDF](https://arxiv.org/pdf/2508.20217)  

**Abstract**: This study explores automatic generation (AIG) using language models to create multiple choice questions (MCQs) for morphological assessment, aiming to reduce the cost and inconsistency of manual test development. The study used a two-fold approach. First, we compared a fine-tuned medium model (Gemma, 2B) with a larger untuned one (GPT-3.5, 175B). Second, we evaluated seven structured prompting strategies, including zero-shot, few-shot, chain-of-thought, role-based, sequential, and combinations. Generated items were assessed using automated metrics and expert scoring across five dimensions. We also used GPT-4.1, trained on expert-rated samples, to simulate human scoring at scale. Results show that structured prompting, especially strategies combining chain-of-thought and sequential design, significantly improved Gemma's outputs. Gemma generally produced more construct-aligned and instructionally appropriate items than GPT-3.5's zero-shot responses, with prompt design playing a key role in mid-size model performance. This study demonstrates that structured prompting and efficient fine-tuning can enhance midsized models for AIG under limited data conditions. We highlight the value of combining automated metrics, expert judgment, and large-model simulation to ensure alignment with assessment goals. The proposed workflow offers a practical and scalable way to develop and validate language assessment items for K-12. 

---
# Collaborating with GenAI: Incentives and Replacements 

**Authors**: Boaz Taitler, Omer Ben-Porat  

**Link**: [PDF](https://arxiv.org/pdf/2508.20213)  

**Abstract**: The rise of Generative AI (GenAI) is reshaping how workers contribute to shared projects. While workers can use GenAI to boost productivity or reduce effort, managers may use it to replace some workers entirely. We present a theoretical framework to analyze how GenAI affects collaboration in such settings. In our model, the manager selects a team to work on a shared task, with GenAI substituting for unselected workers. Each worker selects how much effort to exert, and incurs a cost that increases with the level of effort. We show that GenAI can lead workers to exert no effort, even if GenAI is almost ineffective. We further show that the manager's optimization problem is NP-complete, and provide an efficient algorithm for the special class of (almost-) linear instances. Our analysis shows that even workers with low individual value may play a critical role in sustaining overall output, and excluding such workers can trigger a cascade. Finally, we conduct extensive simulations to illustrate our theoretical findings. 

---
# Filter then Attend: Improving attention-based Time Series Forecasting with Spectral Filtering 

**Authors**: Elisha Dayag, Nhat Thanh Van Tran, Jack Xin  

**Link**: [PDF](https://arxiv.org/pdf/2508.20206)  

**Abstract**: Transformer-based models are at the forefront in long time-series forecasting (LTSF). While in many cases, these models are able to achieve state of the art results, they suffer from a bias toward low-frequencies in the data and high computational and memory requirements. Recent work has established that learnable frequency filters can be an integral part of a deep forecasting model by enhancing the model's spectral utilization. These works choose to use a multilayer perceptron to process their filtered signals and thus do not solve the issues found with transformer-based models. In this paper, we establish that adding a filter to the beginning of transformer-based models enhances their performance in long time-series forecasting. We add learnable filters, which only add an additional $\approx 1000$ parameters to several transformer-based models and observe in multiple instances 5-10 \% relative improvement in forecasting performance. Additionally, we find that with filters added, we are able to decrease the embedding dimension of our models, resulting in transformer-based architectures that are both smaller and more effective than their non-filtering base models. We also conduct synthetic experiments to analyze how the filters enable Transformer-based models to better utilize the full spectrum for forecasting. 

---
# AI Propaganda factories with language models 

**Authors**: Lukasz Olejnik  

**Link**: [PDF](https://arxiv.org/pdf/2508.20186)  

**Abstract**: AI-powered influence operations can now be executed end-to-end on commodity hardware. We show that small language models produce coherent, persona-driven political messaging and can be evaluated automatically without human raters. Two behavioural findings emerge. First, persona-over-model: persona design explains behaviour more than model identity. Second, engagement as a stressor: when replies must counter-arguments, ideological adherence strengthens and the prevalence of extreme content increases. We demonstrate that fully automated influence-content production is within reach of both large and small actors. Consequently, defence should shift from restricting model access towards conversation-centric detection and disruption of campaigns and coordination infrastructure. Paradoxically, the very consistency that enables these operations also provides a detection signature. 

---
# Mitigating Hallucinations in Multimodal LLMs via Object-aware Preference Optimization 

**Authors**: Alberto Compagnoni, Davide Caffagni, Nicholas Moratelli, Lorenzo Baraldi, Marcella Cornia, Rita Cucchiara  

**Link**: [PDF](https://arxiv.org/pdf/2508.20181)  

**Abstract**: Multimodal Large Language Models (MLLMs) emerge as a unified interface to address a multitude of tasks, ranging from NLP to computer vision. Despite showcasing state-of-the-art results in many benchmarks, a long-standing issue is the tendency of MLLMs to hallucinate, that is to generate answers to the user's query that are not reflected in the visual input. In this paper, we address the problem of hallucinations as an alignment problem, seeking to steer the MLLM so that it prefers generating content without hallucinations. In contrast to recent approaches that require complicated pipelines to build synthetic preference data for alignment training, often relying on proprietary models, we capitalize on the well-known CHAIR metric, originally proposed to gauge the degree of hallucinations in image captioning. Given a pair of generated answers, we leverage CHAIR to distinguish winner and loser options (i.e., non-hallucinated and hallucinated samples) and fine-tune off-the-shelf MLLMs via Direct Preference Optimization (DPO). The resulting method, which we refer to as CHAIR-DPO, effectively diminishes the amount of hallucinated answers on several hallucination benchmarks, demonstrating the effectiveness of fine-tuning the MLLM with a CHAIR-based reward. Source code and trained models are publicly available at this https URL. 

---
# RelAItionship Building: Analyzing Recruitment Strategies for Participatory AI 

**Authors**: Eugene Kim, Vaibhav Balloli, Berelian Karimian, Elizabeth Bondi-Kelly, Benjamin Fish  

**Link**: [PDF](https://arxiv.org/pdf/2508.20176)  

**Abstract**: Participatory AI, in which impacted community members and other stakeholders are involved in the design and development of AI systems, holds promise as a way to ensure AI is developed to meet their needs and reflect their values. However, the process of identifying, reaching out, and engaging with all relevant stakeholder groups, which we refer to as recruitment methodology, is still a practical challenge in AI projects striving to adopt participatory practices. In this paper, we investigate the challenges that researchers face when designing and executing recruitment methodology for Participatory AI projects, and the implications of current recruitment practice for Participatory AI. First, we describe the recruitment methodologies used in AI projects using a corpus of 37 projects to capture the diversity of practices in the field and perform an initial analysis on the documentation of recruitment practices, as well as specific strategies that researchers use to meet goals of equity and empowerment. To complement this analysis, we interview five AI researchers to learn about the outcomes of recruitment methodologies. We find that these outcomes are shaped by structural conditions of their work, researchers' own goals and expectations, and the relationships built from the recruitment methodology and subsequent collaboration. Based on these analyses, we provide recommendations for designing and executing relationship-forward recruitment methods, as well as reflexive recruitment documentation practices for Participatory AI researchers. 

---
# Navigating the EU AI Act: Foreseeable Challenges in Qualifying Deep Learning-Based Automated Inspections of Class III Medical Devices 

**Authors**: Julio Zanon Diaz, Tommy Brennan, Peter Corcoran  

**Link**: [PDF](https://arxiv.org/pdf/2508.20144)  

**Abstract**: As deep learning (DL) technologies advance, their application in automated visual inspection for Class III medical devices offers significant potential to enhance quality assurance and reduce human error. However, the adoption of such AI-based systems introduces new regulatory complexities--particularly under the EU Artificial Intelligence (AI) Act, which imposes high-risk system obligations that differ in scope and depth from established regulatory frameworks such as the Medical Device Regulation (MDR) and the U.S. FDA Quality System Regulation (QSR). This paper presents a high-level technical assessment of the foresee-able challenges that manufacturers are likely to encounter when qualifying DL-based automated inspections within the existing medical device compliance landscape. It examines divergences in risk management principles, dataset governance, model validation, explainability requirements, and post-deployment monitoring obligations. The discussion also explores potential implementation strategies and highlights areas of uncertainty, including data retention burdens, global compliance implications, and the practical difficulties of achieving statistical significance in validation with limited defect data. Disclaimer: This publication is in-tended solely as an academic and technical evaluation. It is not a substitute for le-gal advice or official regulatory interpretation. The information presented here should not be relied upon to demonstrate compliance with the EU AI Act or any other statutory obligation. Manufacturers are encouraged to consult appropriate regulatory authorities and legal experts to determine specific compliance pathways. 

---
# UltraEar: a multicentric, large-scale database combining ultra-high-resolution computed tomography and clinical data for ear diseases 

**Authors**: Ruowei Tang, Pengfei Zhao, Xiaoguang Li, Ning Xu, Yue Cheng, Mengshi Zhang, Zhixiang Wang, Zhengyu Zhang, Hongxia Yin, Heyu Ding, Shusheng Gong, Yuhe Liu, Zhenchang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.20141)  

**Abstract**: Ear diseases affect billions of people worldwide, leading to substantial health and socioeconomic burdens. Computed tomography (CT) plays a pivotal role in accurate diagnosis, treatment planning, and outcome evaluation. The objective of this study is to present the establishment and design of UltraEar Database, a large-scale, multicentric repository of isotropic 0.1 mm ultra-high-resolution CT (U-HRCT) images and associated clinical data dedicated to ear diseases. UltraEar recruits patients from 11 tertiary hospitals between October 2020 and October 2035, integrating U-HRCT images, structured CT reports, and comprehensive clinical information, including demographics, audiometric profiles, surgical records, and pathological findings. A broad spectrum of otologic disorders is covered, such as otitis media, cholesteatoma, ossicular chain malformation, temporal bone fracture, inner ear malformation, cochlear aperture stenosis, enlarged vestibular aqueduct, and sigmoid sinus bony deficiency. Standardized preprocessing pipelines have been developed for geometric calibration, image annotation, and multi-structure segmentation. All personal identifiers in DICOM headers and metadata are removed or anonymized to ensure compliance with data privacy regulation. Data collection and curation are coordinated through monthly expert panel meetings, with secure storage on an offline cloud system. UltraEar provides an unprecedented ultra-high-resolution reference atlas with both technical fidelity and clinical relevance. This resource has significant potential to advance radiological research, enable development and validation of AI algorithms, serve as an educational tool for training in otologic imaging, and support multi-institutional collaborative studies. UltraEar will be continuously updated and expanded, ensuring long-term accessibility and usability for the global otologic research community. 

---
# Data-Efficient Point Cloud Semantic Segmentation Pipeline for Unimproved Roads 

**Authors**: Andrew Yarovoi, Christopher R. Valenta  

**Link**: [PDF](https://arxiv.org/pdf/2508.20135)  

**Abstract**: In this case study, we present a data-efficient point cloud segmentation pipeline and training framework for robust segmentation of unimproved roads and seven other classes. Our method employs a two-stage training framework: first, a projection-based convolutional neural network is pre-trained on a mixture of public urban datasets and a small, curated in-domain dataset; then, a lightweight prediction head is fine-tuned exclusively on in-domain data. Along the way, we explore the application of Point Prompt Training to batch normalization layers and the effects of Manifold Mixup as a regularizer within our pipeline. We also explore the effects of incorporating histogram-normalized ambients to further boost performance. Using only 50 labeled point clouds from our target domain, we show that our proposed training approach improves mean Intersection-over-Union from 33.5% to 51.8% and the overall accuracy from 85.5% to 90.8%, when compared to naive training on the in-domain data. Crucially, our results demonstrate that pre-training across multiple datasets is key to improving generalization and enabling robust segmentation under limited in-domain supervision. Overall, this study demonstrates a practical framework for robust 3D semantic segmentation in challenging, low-data scenarios. Our code is available at: this https URL. 

---
# Artificial Intelligence for CRISPR Guide RNA Design: Explainable Models and Off-Target Safety 

**Authors**: Alireza Abbaszadeh, Armita Shahlai  

**Link**: [PDF](https://arxiv.org/pdf/2508.20130)  

**Abstract**: CRISPR-based genome editing has revolutionized biotechnology, yet optimizing guide RNA (gRNA) design for efficiency and safety remains a critical challenge. Recent advances (2020--2025, updated to reflect current year if needed) demonstrate that artificial intelligence (AI), especially deep learning, can markedly improve the prediction of gRNA on-target activity and identify off-target risks. In parallel, emerging explainable AI (XAI) techniques are beginning to illuminate the black-box nature of these models, offering insights into sequence features and genomic contexts that drive Cas enzyme performance. Here we review how state-of-the-art machine learning models are enhancing gRNA design for CRISPR systems, highlight strategies for interpreting model predictions, and discuss new developments in off-target prediction and safety assessment. We emphasize breakthroughs from top-tier journals that underscore an interdisciplinary convergence of AI and genome editing to enable more efficient, specific, and clinically viable CRISPR applications. 

---
# Improving Liver Disease Diagnosis with SNNDeep: A Custom Spiking Neural Network Using Diverse Learning Algorithms 

**Authors**: Zofia Rudnicka, Janusz Szczepanski, Agnieszka Pregowska  

**Link**: [PDF](https://arxiv.org/pdf/2508.20125)  

**Abstract**: Purpose: Spiking neural networks (SNNs) have recently gained attention as energy-efficient, biologically plausible alternatives to conventional deep learning models. Their application in high-stakes biomedical imaging remains almost entirely unexplored. Methods: This study introduces SNNDeep, the first tailored SNN specifically optimized for binary classification of liver health status from computed tomography (CT) features. To ensure clinical relevance and broad generalizability, the model was developed and evaluated using the Task03\Liver dataset from the Medical Segmentation Decathlon (MSD), a standardized benchmark widely used for assessing performance across diverse medical imaging tasks. We benchmark three fundamentally different learning algorithms, namely Surrogate Gradient Learning, the Tempotron rule, and Bio-Inspired Active Learning across three architectural variants: a fully customized low-level model built from scratch, and two implementations using leading SNN frameworks, i.e., snnTorch and SpikingJelly. Hyperparameter optimization was performed using Optuna. Results: Our results demonstrate that the custom-built SNNDeep consistently outperforms framework-based implementations, achieving a maximum validation accuracy of 98.35%, superior adaptability across learning rules, and significantly reduced training overhead. Conclusion:This study provides the first empirical evidence that low-level, highly tunable SNNs can surpass standard frameworks in medical imaging, especially in data-limited, temporally constrained diagnostic settings, thereby opening a new pathway for neuro-inspired AI in precision medicine. 

---
# Towards Better Correctness and Efficiency in Code Generation 

**Authors**: Yunlong Feng, Yang Xu, Xiao Xu, Binyuan Hui, Junyang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2508.20124)  

**Abstract**: While code large language models have demonstrated remarkable progress in code generation, the generated code often exhibits poor runtime efficiency, limiting its practical application in performance-sensitive scenarios. To address this limitation, we propose an efficiency-oriented reinforcement learning framework guided by a novel performance reward. Based on this framework, we take a deeper dive into the code efficiency problem, identifying then proposing methods to overcome key bottlenecks: (1) Dynamic exploration overcomes the static data constraints of offline fine-tuning, enabling the discovery of more efficient code implementations. (2) The error-insensitive reinforcement learning method and high-contrast efficiency signals are crucial for mitigating systematic errors and achieving effective optimization. (3) Online exploration is most effective when starting from a high-correctness baseline, as this allows for efficiency improvements without sacrificing accuracy. With these discoveries, we finally propose a two-stage tuning method, which achieves high and balanced performance across correctness and efficiency. The results of experiments show the effectiveness of the method, which improves code correctness by 10.18\% and runtime efficiency by 7.75\% on a 7B model, achieving performance comparable to much larger model. 

---
# Particle swarm optimization for online sparse streaming feature selection under uncertainty 

**Authors**: Ruiyang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.20123)  

**Abstract**: In real-world applications involving high-dimensional streaming data, online streaming feature selection (OSFS) is widely adopted. Yet, practical deployments frequently face data incompleteness due to sensor failures or technical constraints. While online sparse streaming feature selection (OS2FS) mitigates this issue via latent factor analysis-based imputation, existing methods struggle with uncertain feature-label correlations, leading to inflexible models and degraded performance. To address these gaps, this work proposes POS2FS-an uncertainty-aware online sparse streaming feature selection framework enhanced by particle swarm optimization (PSO). The approach introduces: 1) PSO-driven supervision to reduce uncertainty in feature-label relationships; 2) Three-way decision theory to manage feature fuzziness in supervised learning. Rigorous testing on six real-world datasets confirms POS2FS outperforms conventional OSFS and OS2FS techniques, delivering higher accuracy through more robust feature subset selection. 

---
# Is Artificial Intelligence Reshaping the Landscape of the International Academic Community of Geosciences? 

**Authors**: Liang Li, Yuntian Li, Wenxin Zhao, Shan Ye, Yun Lu  

**Link**: [PDF](https://arxiv.org/pdf/2508.20117)  

**Abstract**: Through bibliometric analysis and topic modeling, we find that artificial intelligence (AI) is positively transforming geosciences research, with a notable increase in AI-related scientific output in recent years. We are encouraged to observe that earth scientists from developing countries have gained better visibility in the recent AI for Science (AI4S) paradigm and that AI is also improving the landscape of international collaboration in geoscience-related research. 

---
# Flexible metadata harvesting for ecology using large language models 

**Authors**: Zehao Lu, Thijs L van der Plas, Parinaz Rashidi, W Daniel Kissling, Ioannis N Athanasiadis  

**Link**: [PDF](https://arxiv.org/pdf/2508.20115)  

**Abstract**: Large, open datasets can accelerate ecological research, particularly by enabling researchers to develop new insights by reusing datasets from multiple sources. However, to find the most suitable datasets to combine and integrate, researchers must navigate diverse ecological and environmental data provider platforms with varying metadata availability and standards. To overcome this obstacle, we have developed a large language model (LLM)-based metadata harvester that flexibly extracts metadata from any dataset's landing page, and converts these to a user-defined, unified format using existing metadata standards. We validate that our tool is able to extract both structured and unstructured metadata with equal accuracy, aided by our LLM post-processing protocol. Furthermore, we utilise LLMs to identify links between datasets, both by calculating embedding similarity and by unifying the formats of extracted metadata to enable rule-based processing. Our tool, which flexibly links the metadata of different datasets, can therefore be used for ontology creation or graph-based queries, for example, to find relevant ecological and environmental datasets in a virtual research environment. 

---
# Deep Reinforcement Learning for Optimal Asset Allocation Using DDPG with TiDE 

**Authors**: Rongwei Liu, Jin Zheng, John Cartlidge  

**Link**: [PDF](https://arxiv.org/pdf/2508.20103)  

**Abstract**: The optimal asset allocation between risky and risk-free assets is a persistent challenge due to the inherent volatility in financial markets. Conventional methods rely on strict distributional assumptions or non-additive reward ratios, which limit their robustness and applicability to investment goals. To overcome these constraints, this study formulates the optimal two-asset allocation problem as a sequential decision-making task within a Markov Decision Process (MDP). This framework enables the application of reinforcement learning (RL) mechanisms to develop dynamic policies based on simulated financial scenarios, regardless of prerequisites. We use the Kelly criterion to balance immediate reward signals against long-term investment objectives, and we take the novel step of integrating the Time-series Dense Encoder (TiDE) into the Deep Deterministic Policy Gradient (DDPG) RL framework for continuous decision-making. We compare DDPG-TiDE with a simple discrete-action Q-learning RL framework and a passive buy-and-hold investment strategy. Empirical results show that DDPG-TiDE outperforms Q-learning and generates higher risk adjusted returns than buy-and-hold. These findings suggest that tackling the optimal asset allocation problem by integrating TiDE within a DDPG reinforcement learning framework is a fruitful avenue for further exploration. 

---
# A Hierarchical Signal Coordination and Control System Using a Hybrid Model-based and Reinforcement Learning Approach 

**Authors**: Xianyue Peng, Shenyang Chen, H. Michael Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.20102)  

**Abstract**: Signal control in urban corridors faces the dual challenge of maintaining arterial traffic progression while adapting to demand variations at local intersections. We propose a hierarchical traffic signal coordination and control scheme that integrates model-based optimization with reinforcement learning. The system consists of: (i) a High-Level Coordinator (HLC) that selects coordination strategies based on observed and predicted demand; (ii) a Corridor Coordinator that derives phase constraints from the selected strategy-either Max-Flow Coordination (MFC) or Green-Wave Coordination (GWC); and (iii) Hybrid Signal Agents (HSAs) that determine signal phases via reinforcement learning with action masking to enforce feasibility. Hierarchical reinforcement learning with Proximal Policy Optimization (PPO) is used to train HSA and HLC policies. At the lower level, three HSA policies-MFC-aware, GWC-aware, and pure agent control (PAC) are trained in conjunction with their respective coordination strategies. At the higher level, the HLC is trained to dynamically switch strategies using a multi-objective reward balancing corridor-level and network-wide performance. The proposed scheme was developed and evaluated on a SUMO-RLlib platform. Case results show that hybrid MFC maximizes throughput under heavy demand; hybrid GWC consistently minimizes arterial stops and maintains progression across diverse traffic conditions but can reduce network-wide efficiency; and PAC improves network-wide travel time in moderate demand but is less effective under heavy demand. The hierarchical design enables adaptive strategy selection, achieving robust performance across all demand levels. 

---
# Can LLMs Identify Tax Abuse? 

**Authors**: Andrew Blair-Stanek, Nils Holzenberger, Benjamin Van Durme  

**Link**: [PDF](https://arxiv.org/pdf/2508.20097)  

**Abstract**: We investigate whether large language models can discover and analyze U.S. tax-minimization strategies. This real-world domain challenges even seasoned human experts, and progress can reduce tax revenue lost from well-advised, wealthy taxpayers. We evaluate the most advanced LLMs on their ability to (1) interpret and verify tax strategies, (2) fill in gaps in partially specified strategies, and (3) generate complete, end-to-end strategies from scratch. This domain should be of particular interest to the LLM reasoning community: unlike synthetic challenge problems or scientific reasoning tasks, U.S. tax law involves navigating hundreds of thousands of pages of statutes, case law, and administrative guidance, all updated regularly. Notably, LLM-based reasoning identified an entirely novel tax strategy, highlighting these models' potential to revolutionize tax agencies' fight against tax abuse. 

---
