# Towards Adaptive Memory-Based Optimization for Enhanced Retrieval-Augmented Generation 

**Authors**: Qitao Qin, Yucong Luo, Yihang Lu, Zhibo Chu, Xianwei Meng  

**Link**: [PDF](https://arxiv.org/pdf/2504.05312)  

**Abstract**: Retrieval-Augmented Generation (RAG), by integrating non-parametric knowledge from external knowledge bases into models, has emerged as a promising approach to enhancing response accuracy while mitigating factual errors and hallucinations. This method has been widely applied in tasks such as Question Answering (QA). However, existing RAG methods struggle with open-domain QA tasks because they perform independent retrieval operations and directly incorporate the retrieved information into generation without maintaining a summarizing memory or using adaptive retrieval strategies, leading to noise from redundant information and insufficient information integration. To address these challenges, we propose Adaptive memory-based optimization for enhanced RAG (Amber) for open-domain QA tasks, which comprises an Agent-based Memory Updater, an Adaptive Information Collector, and a Multi-granular Content Filter, working together within an iterative memory updating paradigm. Specifically, Amber integrates and optimizes the language model's memory through a multi-agent collaborative approach, ensuring comprehensive knowledge integration from previous retrieval steps. It dynamically adjusts retrieval queries and decides when to stop retrieval based on the accumulated knowledge, enhancing retrieval efficiency and effectiveness. Additionally, it reduces noise by filtering irrelevant content at multiple levels, retaining essential information to improve overall model performance. We conduct extensive experiments on several open-domain QA datasets, and the results demonstrate the superiority and effectiveness of our method and its components. The source code is available \footnote{this https URL}. 

---
# Are Generative AI Agents Effective Personalized Financial Advisors? 

**Authors**: Takehiro Takayanagi, Kiyoshi Izumi, Javier Sanz-Cruzado, Richard McCreadie, Iadh Ounis  

**Link**: [PDF](https://arxiv.org/pdf/2504.05862)  

**Abstract**: Large language model-based agents are becoming increasingly popular as a low-cost mechanism to provide personalized, conversational advice, and have demonstrated impressive capabilities in relatively simple scenarios, such as movie recommendations. But how do these agents perform in complex high-stakes domains, where domain expertise is essential and mistakes carry substantial risk? This paper investigates the effectiveness of LLM-advisors in the finance domain, focusing on three distinct challenges: (1) eliciting user preferences when users themselves may be unsure of their needs, (2) providing personalized guidance for diverse investment preferences, and (3) leveraging advisor personality to build relationships and foster trust. Via a lab-based user study with 64 participants, we show that LLM-advisors often match human advisor performance when eliciting preferences, although they can struggle to resolve conflicting user needs. When providing personalized advice, the LLM was able to positively influence user behavior, but demonstrated clear failure modes. Our results show that accurate preference elicitation is key, otherwise, the LLM-advisor has little impact, or can even direct the investor toward unsuitable assets. More worryingly, users appear insensitive to the quality of advice being given, or worse these can have an inverse relationship. Indeed, users reported a preference for and increased satisfaction as well as emotional trust with LLMs adopting an extroverted persona, even though those agents provided worse advice. 

---
# Balancing Benefits and Risks: RL Approaches for Addiction-Aware Social Media Recommenders 

**Authors**: Luca Bolis, Stefano Livella, Sabrina Patania, Dimitri Ognibene, Matteo Papini, Kenji Morita  

**Link**: [PDF](https://arxiv.org/pdf/2504.05322)  

**Abstract**: Social media platforms provide valuable opportunities for users to gather information, interact with friends, and enjoy entertainment. However, their addictive potential poses significant challenges, including overuse and negative psycho-logical or behavioral impacts [4, 2, 8]. This study explores strategies to mitigate compulsive social media usage while preserving its benefits and ensuring economic sustainability, focusing on recommenders that promote balanced usage.
We analyze user behaviors arising from intrinsic diversities and environmental interactions, offering insights for next-generation social media recommenders that prioritize well-being. Specifically, we examine the temporal predictability of overuse and addiction using measures available to recommenders, aiming to inform mechanisms that prevent addiction while avoiding user disengagement [7].
Building on RL-based computational frameworks for addiction modelling [6], our study introduces: - A recommender system adapting to user preferences, introducing non-stationary and non-Markovian dynamics.
- Differentiated state representations for users and recommenders to capture nuanced interactions.
- Distinct usage conditions-light and heavy use-addressing RL's limitations in distinguishing prolonged from healthy engagement.
- Complexity in overuse impacts, highlighting their role in user adaptation [7].
Simulations demonstrate how model-based (MB) and model-free (MF) decision-making interact with environmental dynamics to influence user behavior and addiction. Results reveal the significant role of recommender systems in shaping addiction tendencies or fostering healthier engagement. These findings support ethical, adaptive recommender design, advancing sustainable social media ecosystems [9, 1].
Keywords: multi-agent systems, recommender systems, addiction, social media 

---
# FactGuard: Leveraging Multi-Agent Systems to Generate Answerable and Unanswerable Questions for Enhanced Long-Context LLM Extraction 

**Authors**: Qian-Wen Zhang, Fang Li, Jie Wang, Lingfeng Qiao, Yifei Yu, Di Yin, Xing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.05607)  

**Abstract**: Extractive reading comprehension systems are designed to locate the correct answer to a question within a given text. However, a persistent challenge lies in ensuring these models maintain high accuracy in answering questions while reliably recognizing unanswerable queries. Despite significant advances in large language models (LLMs) for reading comprehension, this issue remains critical, particularly as the length of supported contexts continues to expand. To address this challenge, we propose an innovative data augmentation methodology grounded in a multi-agent collaborative framework. Unlike traditional methods, such as the costly human annotation process required for datasets like SQuAD 2.0, our method autonomously generates evidence-based question-answer pairs and systematically constructs unanswerable questions. Using this methodology, we developed the FactGuard-Bench dataset, which comprises 25,220 examples of both answerable and unanswerable question scenarios, with context lengths ranging from 8K to 128K. Experimental evaluations conducted on seven popular LLMs reveal that even the most advanced models achieve only 61.79% overall accuracy. Furthermore, we emphasize the importance of a model's ability to reason about unanswerable questions to avoid generating plausible but incorrect answers. By implementing efficient data selection and generation within the multi-agent collaborative framework, our method significantly reduces the traditionally high costs associated with manual annotation and provides valuable insights for the training and optimization of LLMs. 

---
# TxGemma: Efficient and Agentic LLMs for Therapeutics 

**Authors**: Eric Wang, Samuel Schmidgall, Paul F. Jaeger, Fan Zhang, Rory Pilgrim, Yossi Matias, Joelle Barral, David Fleet, Shekoofeh Azizi  

**Link**: [PDF](https://arxiv.org/pdf/2504.06196)  

**Abstract**: Therapeutic development is a costly and high-risk endeavor that is often plagued by high failure rates. To address this, we introduce TxGemma, a suite of efficient, generalist large language models (LLMs) capable of therapeutic property prediction as well as interactive reasoning and explainability. Unlike task-specific models, TxGemma synthesizes information from diverse sources, enabling broad application across the therapeutic development pipeline. The suite includes 2B, 9B, and 27B parameter models, fine-tuned from Gemma-2 on a comprehensive dataset of small molecules, proteins, nucleic acids, diseases, and cell lines. Across 66 therapeutic development tasks, TxGemma achieved superior or comparable performance to the state-of-the-art generalist model on 64 (superior on 45), and against state-of-the-art specialist models on 50 (superior on 26). Fine-tuning TxGemma models on therapeutic downstream tasks, such as clinical trial adverse event prediction, requires less training data than fine-tuning base LLMs, making TxGemma suitable for data-limited applications. Beyond these predictive capabilities, TxGemma features conversational models that bridge the gap between general LLMs and specialized property predictors. These allow scientists to interact in natural language, provide mechanistic reasoning for predictions based on molecular structure, and engage in scientific discussions. Building on this, we further introduce Agentic-Tx, a generalist therapeutic agentic system powered by Gemini 2.5 that reasons, acts, manages diverse workflows, and acquires external domain knowledge. Agentic-Tx surpasses prior leading models on the Humanity's Last Exam benchmark (Chemistry & Biology) with 52.3% relative improvement over o3-mini (high) and 26.7% over o3-mini (high) on GPQA (Chemistry) and excels with improvements of 6.3% (ChemBench-Preference) and 2.4% (ChemBench-Mini) over o3-mini (high). 

---
# SkillFlow: Efficient Skill and Code Transfer Through Communication in Adapting AI Agents 

**Authors**: Pagkratios Tagkopoulos, Fangzhou Li, Ilias Tagkopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2504.06188)  

**Abstract**: AI agents are autonomous systems that can execute specific tasks based on predefined programming. Here, we present SkillFlow, a modular, technology-agnostic framework that allows agents to expand their functionality in an ad-hoc fashion by acquiring new skills from their environment or other agents. We present a theoretical model that examines under which conditions this framework would be beneficial, and we then explore SkillFlow's ability to accelerate task completion and lead to lower cumulative costs in a real-world application, namely scheduling agents for calendar events. We demonstrate that within a few iterations, SkillFlow leads to considerable (24.8%, p-value = $6.4\times10^{-3}$) gains in time and cost, especially when the communication cost is high. Finally, we draw analogies from well-studied biological systems and compare this framework to that of lateral gene transfer, a significant process of adaptation and evolution in novel environments. 

---
# Hogwild! Inference: Parallel LLM Generation via Concurrent Attention 

**Authors**: Gleb Rodionov, Roman Garipov, Alina Shutova, George Yakushev, Vage Egiazarian, Anton Sinitsin, Denis Kuznedelev, Dan Alistarh  

**Link**: [PDF](https://arxiv.org/pdf/2504.06261)  

**Abstract**: Large Language Models (LLMs) have demonstrated the ability to tackle increasingly complex tasks through advanced reasoning, long-form content generation, and tool use. Solving these tasks often involves long inference-time computations. In human problem solving, a common strategy to expedite work is collaboration: by dividing the problem into sub-tasks, exploring different strategies concurrently, etc. Recent research has shown that LLMs can also operate in parallel by implementing explicit cooperation frameworks, such as voting mechanisms or the explicit creation of independent sub-tasks that can be executed in parallel. However, each of these frameworks may not be suitable for all types of tasks, which can hinder their applicability. In this work, we propose a different design approach: we run LLM "workers" in parallel , allowing them to synchronize via a concurrently-updated attention cache and prompt these workers to decide how best to collaborate. Our approach allows the instances to come up with their own collaboration strategy for the problem at hand, all the while "seeing" each other's partial progress in the concurrent cache. We implement this approach via Hogwild! Inference: a parallel LLM inference engine where multiple instances of the same LLM run in parallel with the same attention cache, with "instant" access to each other's generated tokens. Hogwild! inference takes advantage of Rotary Position Embeddings (RoPE) to avoid recomputation while improving parallel hardware utilization. We find that modern reasoning-capable LLMs can perform inference with shared Key-Value cache out of the box, without additional fine-tuning. 

---
# Decentralizing AI Memory: SHIMI, a Semantic Hierarchical Memory Index for Scalable Agent Reasoning 

**Authors**: Tooraj Helmi  

**Link**: [PDF](https://arxiv.org/pdf/2504.06135)  

**Abstract**: Retrieval-Augmented Generation (RAG) and vector-based search have become foundational tools for memory in AI systems, yet they struggle with abstraction, scalability, and semantic precision - especially in decentralized environments. We present SHIMI (Semantic Hierarchical Memory Index), a unified architecture that models knowledge as a dynamically structured hierarchy of concepts, enabling agents to retrieve information based on meaning rather than surface similarity. SHIMI organizes memory into layered semantic nodes and supports top-down traversal from abstract intent to specific entities, offering more precise and explainable retrieval. Critically, SHIMI is natively designed for decentralized ecosystems, where agents maintain local memory trees and synchronize them asynchronously across networks. We introduce a lightweight sync protocol that leverages Merkle-DAG summaries, Bloom filters, and CRDT-style conflict resolution to enable partial synchronization with minimal overhead. Through benchmark experiments and use cases involving decentralized agent collaboration, we demonstrate SHIMI's advantages in retrieval accuracy, semantic fidelity, and scalability - positioning it as a core infrastructure layer for decentralized cognitive systems. 

---
# Agent Guide: A Simple Agent Behavioral Watermarking Framework 

**Authors**: Kaibo Huang, Zhongliang Yang, Linna Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.05871)  

**Abstract**: The increasing deployment of intelligent agents in digital ecosystems, such as social media platforms, has raised significant concerns about traceability and accountability, particularly in cybersecurity and digital content protection. Traditional large language model (LLM) watermarking techniques, which rely on token-level manipulations, are ill-suited for agents due to the challenges of behavior tokenization and information loss during behavior-to-action translation. To address these issues, we propose Agent Guide, a novel behavioral watermarking framework that embeds watermarks by guiding the agent's high-level decisions (behavior) through probability biases, while preserving the naturalness of specific executions (action). Our approach decouples agent behavior into two levels, behavior (e.g., choosing to bookmark) and action (e.g., bookmarking with specific tags), and applies watermark-guided biases to the behavior probability distribution. We employ a z-statistic-based statistical analysis to detect the watermark, ensuring reliable extraction over multiple rounds. Experiments in a social media scenario with diverse agent profiles demonstrate that Agent Guide achieves effective watermark detection with a low false positive rate. Our framework provides a practical and robust solution for agent watermarking, with applications in identifying malicious agents and protecting proprietary agent systems. 

---
# AEGIS: Human Attention-based Explainable Guidance for Intelligent Vehicle Systems 

**Authors**: Zhuoli Zhuang, Cheng-You Lu, Yu-Cheng Fred Chang, Yu-Kai Wang, Thomas Do, Chin-Teng Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.05950)  

**Abstract**: Improving decision-making capabilities in Autonomous Intelligent Vehicles (AIVs) has been a heated topic in recent years. Despite advancements, training machines to capture regions of interest for comprehensive scene understanding, like human perception and reasoning, remains a significant challenge. This study introduces a novel framework, Human Attention-based Explainable Guidance for Intelligent Vehicle Systems (AEGIS). AEGIS utilizes human attention, converted from eye-tracking, to guide reinforcement learning (RL) models to identify critical regions of interest for decision-making. AEGIS uses a pre-trained human attention model to guide RL models to identify critical regions of interest for decision-making. By collecting 1.2 million frames from 20 participants across six scenarios, AEGIS pre-trains a model to predict human attention patterns. 

---
# Interactive Explanations for Reinforcement-Learning Agents 

**Authors**: Yotam Amitai, Ofra Amir, Guy Avni  

**Link**: [PDF](https://arxiv.org/pdf/2504.05393)  

**Abstract**: As reinforcement learning methods increasingly amass accomplishments, the need for comprehending their solutions becomes more crucial. Most explainable reinforcement learning (XRL) methods generate a static explanation depicting their developers' intuition of what should be explained and how. In contrast, literature from the social sciences proposes that meaningful explanations are structured as a dialog between the explainer and the explainee, suggesting a more active role for the user and her communication with the agent. In this paper, we present ASQ-IT -- an interactive explanation system that presents video clips of the agent acting in its environment based on queries given by the user that describe temporal properties of behaviors of interest. Our approach is based on formal methods: queries in ASQ-IT's user interface map to a fragment of Linear Temporal Logic over finite traces (LTLf), which we developed, and our algorithm for query processing is based on automata theory. User studies show that end-users can understand and formulate queries in ASQ-IT and that using ASQ-IT assists users in identifying faulty agent behaviors. 

---
# EduPlanner: LLM-Based Multi-Agent Systems for Customized and Intelligent Instructional Design 

**Authors**: Xueqiao Zhang, Chao Zhang, Jianwen Sun, Jun Xiao, Yi Yang, Yawei Luo  

**Link**: [PDF](https://arxiv.org/pdf/2504.05370)  

**Abstract**: Large Language Models (LLMs) have significantly advanced smart education in the Artificial General Intelligence (AGI) era. A promising application lies in the automatic generalization of instructional design for curriculum and learning activities, focusing on two key aspects: (1) Customized Generation: generating niche-targeted teaching content based on students' varying learning abilities and states, and (2) Intelligent Optimization: iteratively optimizing content based on feedback from learning effectiveness or test scores. Currently, a single large LLM cannot effectively manage the entire process, posing a challenge for designing intelligent teaching plans. To address these issues, we developed EduPlanner, an LLM-based multi-agent system comprising an evaluator agent, an optimizer agent, and a question analyst, working in adversarial collaboration to generate customized and intelligent instructional design for curriculum and learning activities. Taking mathematics lessons as our example, EduPlanner employs a novel Skill-Tree structure to accurately model the background mathematics knowledge of student groups, personalizing instructional design for curriculum and learning activities according to students' knowledge levels and learning abilities. Additionally, we introduce the CIDDP, an LLM-based five-dimensional evaluation module encompassing clarity, Integrity, Depth, Practicality, and Pertinence, to comprehensively assess mathematics lesson plan quality and bootstrap intelligent optimization. Experiments conducted on the GSM8K and Algebra datasets demonstrate that EduPlanner excels in evaluating and optimizing instructional design for curriculum and learning activities. Ablation studies further validate the significance and effectiveness of each component within the framework. Our code is publicly available at this https URL 

---
# A Multimedia Analytics Model for the Foundation Model Era 

**Authors**: Marcel Worring, Jan Zahálka, Stef van den Elzen, Maximilian Fischer, Daniel Keim  

**Link**: [PDF](https://arxiv.org/pdf/2504.06138)  

**Abstract**: The rapid advances in Foundation Models and agentic Artificial Intelligence are transforming multimedia analytics by enabling richer, more sophisticated interactions between humans and analytical systems. Existing conceptual models for visual and multimedia analytics, however, do not adequately capture the complexity introduced by these powerful AI paradigms. To bridge this gap, we propose a comprehensive multimedia analytics model specifically designed for the foundation model era. Building upon established frameworks from visual analytics, multimedia analytics, knowledge generation, analytic task definition, mixed-initiative guidance, and human-in-the-loop reinforcement learning, our model emphasizes integrated human-AI teaming based on visual analytics agents from both technical and conceptual perspectives. Central to the model is a seamless, yet explicitly separable, interaction channel between expert users and semi-autonomous analytical processes, ensuring continuous alignment between user intent and AI behavior. The model addresses practical challenges in sensitive domains such as intelligence analysis, investigative journalism, and other fields handling complex, high-stakes data. We illustrate through detailed case studies how our model facilitates deeper understanding and targeted improvement of multimedia analytics solutions. By explicitly capturing how expert users can optimally interact with and guide AI-powered multimedia analytics systems, our conceptual framework sets a clear direction for system design, comparison, and future research. 

---
# Real-Time LaCAM 

**Authors**: Runzhe Liang, Rishi Veerapaneni, Daniel Harabor, Jiaoyang Li, Maxim Likhachev  

**Link**: [PDF](https://arxiv.org/pdf/2504.06091)  

**Abstract**: The vast majority of Multi-Agent Path Finding (MAPF) methods with completeness guarantees require planning full horizon paths. However, planning full horizon paths can take too long and be impractical in real-world applications. Instead, real-time planning and execution, which only allows the planner a finite amount of time before executing and replanning, is more practical for real world multi-agent systems. Several methods utilize real-time planning schemes but none are provably complete, which leads to livelock or deadlock. Our main contribution is to show the first Real-Time MAPF method with provable completeness guarantees. We do this by leveraging LaCAM (Okumura 2023) in an incremental fashion. Our results show how we can iteratively plan for congested environments with a cutoff time of milliseconds while still maintaining the same success rate as full horizon LaCAM. We also show how it can be used with a single-step learned MAPF policy. The proposed Real-Time LaCAM also provides us with a general mechanism for using iterative constraints for completeness in future real-time MAPF algorithms. 

---
# Unraveling Human-AI Teaming: A Review and Outlook 

**Authors**: Bowen Lou, Tian Lu, Raghu Santanam, Yingjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.05755)  

**Abstract**: Artificial Intelligence (AI) is advancing at an unprecedented pace, with clear potential to enhance decision-making and productivity. Yet, the collaborative decision-making process between humans and AI remains underdeveloped, often falling short of its transformative possibilities. This paper explores the evolution of AI agents from passive tools to active collaborators in human-AI teams, emphasizing their ability to learn, adapt, and operate autonomously in complex environments. This paradigm shifts challenges traditional team dynamics, requiring new interaction protocols, delegation strategies, and responsibility distribution frameworks. Drawing on Team Situation Awareness (SA) theory, we identify two critical gaps in current human-AI teaming research: the difficulty of aligning AI agents with human values and objectives, and the underutilization of AI's capabilities as genuine team members. Addressing these gaps, we propose a structured research outlook centered on four key aspects of human-AI teaming: formulation, coordination, maintenance, and training. Our framework highlights the importance of shared mental models, trust-building, conflict resolution, and skill adaptation for effective teaming. Furthermore, we discuss the unique challenges posed by varying team compositions, goals, and complexities. This paper provides a foundational agenda for future research and practical design of sustainable, high-performing human-AI teams. 

---
# Bridging Industrial Expertise and XR with LLM-Powered Conversational Agents 

**Authors**: Despina Tomkou, George Fatouros, Andreas Andreou, Georgios Makridis, Fotis Liarokapis, Dimitrios Dardanis, Athanasios Kiourtis, John Soldatos, Dimosthenis Kyriazis  

**Link**: [PDF](https://arxiv.org/pdf/2504.05527)  

**Abstract**: This paper introduces a novel integration of Retrieval-Augmented Generation (RAG) enhanced Large Language Models (LLMs) with Extended Reality (XR) technologies to address knowledge transfer challenges in industrial environments. The proposed system embeds domain-specific industrial knowledge into XR environments through a natural language interface, enabling hands-free, context-aware expert guidance for workers. We present the architecture of the proposed system consisting of an LLM Chat Engine with dynamic tool orchestration and an XR application featuring voice-driven interaction. Performance evaluation of various chunking strategies, embedding models, and vector databases reveals that semantic chunking, balanced embedding models, and efficient vector stores deliver optimal performance for industrial knowledge retrieval. The system's potential is demonstrated through early implementation in multiple industrial use cases, including robotic assembly, smart infrastructure maintenance, and aerospace component servicing. Results indicate potential for enhancing training efficiency, remote assistance capabilities, and operational guidance in alignment with Industry 5.0's human-centric and resilient approach to industrial development. 

---
# TRATSS: Transformer-Based Task Scheduling System for Autonomous Vehicles 

**Authors**: Yazan Youssef, Paulo Ricardo Marques de Araujo, Aboelmagd Noureldin, Sidney Givigi  

**Link**: [PDF](https://arxiv.org/pdf/2504.05407)  

**Abstract**: Efficient scheduling remains a critical challenge in various domains, requiring solutions to complex NP-hard optimization problems to achieve optimal resource allocation and maximize productivity. In this paper, we introduce a framework called Transformer-Based Task Scheduling System (TRATSS), designed to address the intricacies of single agent scheduling in graph-based environments. By integrating the latest advancements in reinforcement learning and transformer architecture, TRATSS provides a novel system that outputs optimized task scheduling decisions while dynamically adapting to evolving task requirements and resource availability. Leveraging the self-attention mechanism in transformers, TRATSS effectively captures complex task dependencies, thereby providing solutions with enhanced resource utilization and task completion efficiency. Experimental evaluations on benchmark datasets demonstrate TRATSS's effectiveness in providing high-quality solutions to scheduling problems that involve multiple action profiles. 

---
# A Nature-Inspired Colony of Artificial Intelligence System with Fast, Detailed, and Organized Learner Agents for Enhancing Diversity and Quality 

**Authors**: Shan Suthaharan  

**Link**: [PDF](https://arxiv.org/pdf/2504.05365)  

**Abstract**: The concepts of convolutional neural networks (CNNs) and multi-agent systems are two important areas of research in artificial intelligence (AI). In this paper, we present an approach that builds a CNN-based colony of AI agents to serve as a single system and perform multiple tasks (e.g., predictions or classifications) in an environment. The proposed system impersonates the natural environment of a biological system, like an ant colony or a human colony. The proposed colony of AI that is defined as a role-based system uniquely contributes to accomplish tasks in an environment by incorporating AI agents that are fast learners, detailed learners, and organized learners. These learners can enhance their localized learning and their collective decisions as a single system of colony of AI agents. This approach also enhances the diversity and quality of the colony of AI with the help of Genetic Algorithms and their crossover and mutation mechanisms. The evolution of fast, detailed, and organized learners in the colony of AI is achieved by introducing a unique one-to-one mapping between these learners and the pretrained VGG16, VGG19, and ResNet50 models, respectively. This role-based approach creates two parent-AI agents using the AI models through the processes, called the intra- and inter-marriage of AI, so that they can share their learned knowledge (weights and biases) based on a probabilistic rule and produce diversified child-AI agents to perform new tasks. This process will form a colony of AI that consists of families of multi-model and mixture-model AI agents to improve diversity and quality. Simulations show that the colony of AI, built using the VGG16, VGG19, and ResNet50 models, can provide a single system that generates child-AI agents of excellent predictive performance, ranging between 82% and 95% of F1-scores, to make diversified collective and quality decisions on a task. 

---
# Debate-Feedback: A Multi-Agent Framework for Efficient Legal Judgment Prediction 

**Authors**: Xi Chen, Mao Mao, Shuo Li, Haotian Shangguan  

**Link**: [PDF](https://arxiv.org/pdf/2504.05358)  

**Abstract**: The use of AI in legal analysis and prediction (LegalAI) has gained widespread attention, with past research focusing on retrieval-based methods and fine-tuning large models. However, these approaches often require large datasets and underutilize the capabilities of modern large language models (LLMs). In this paper, inspired by the debate phase of real courtroom trials, we propose a novel legal judgment prediction model based on the Debate-Feedback architecture, which integrates LLM multi-agent debate and reliability evaluation models. Unlike traditional methods, our model achieves significant improvements in efficiency by minimizing the need for large historical datasets, thus offering a lightweight yet robust solution. Comparative experiments show that it outperforms several general-purpose and domain-specific legal models, offering a dynamic reasoning process and a promising direction for future LegalAI research. 

---
