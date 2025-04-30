# MATCHA: Can Multi-Agent Collaboration Build a Trustworthy Conversational Recommender? 

**Authors**: Zheng Hui, Xiaokai Wei, Yexi Jiang, Kevin Gao, Chen Wang, Frank Ong, Se-eun Yoon, Rachit Pareek, Michelle Gong  

**Link**: [PDF](https://arxiv.org/pdf/2504.20094)  

**Abstract**: In this paper, we propose a multi-agent collaboration framework called MATCHA for conversational recommendation system, leveraging large language models (LLMs) to enhance personalization and user engagement. Users can request recommendations via free-form text and receive curated lists aligned with their interests, preferences, and constraints. Our system introduces specialized agents for intent analysis, candidate generation, ranking, re-ranking, explainability, and safeguards. These agents collaboratively improve recommendations accuracy, diversity, and safety. On eight metrics, our model achieves superior or comparable performance to the current state-of-the-art. Through comparisons with six baseline models, our approach addresses key challenges in conversational recommendation systems for game recommendations, including: (1) handling complex, user-specific requests, (2) enhancing personalization through multi-agent collaboration, (3) empirical evaluation and deployment, and (4) ensuring safe and trustworthy interactions. 

---
# CBM-RAG: Demonstrating Enhanced Interpretability in Radiology Report Generation with Multi-Agent RAG and Concept Bottleneck Models 

**Authors**: Hasan Md Tusfiqur Alam, Devansh Srivastav, Abdulrahman Mohamed Selim, Md Abdul Kadir, Md Moktadiurl Hoque Shuvo, Daniel Sonntag  

**Link**: [PDF](https://arxiv.org/pdf/2504.20898)  

**Abstract**: Advancements in generative Artificial Intelligence (AI) hold great promise for automating radiology workflows, yet challenges in interpretability and reliability hinder clinical adoption. This paper presents an automated radiology report generation framework that combines Concept Bottleneck Models (CBMs) with a Multi-Agent Retrieval-Augmented Generation (RAG) system to bridge AI performance with clinical explainability. CBMs map chest X-ray features to human-understandable clinical concepts, enabling transparent disease classification. Meanwhile, the RAG system integrates multi-agent collaboration and external knowledge to produce contextually rich, evidence-based reports. Our demonstration showcases the system's ability to deliver interpretable predictions, mitigate hallucinations, and generate high-quality, tailored reports with an interactive interface addressing accuracy, trust, and usability challenges. This framework provides a pathway to improving diagnostic consistency and empowering radiologists with actionable insights. 

---
# MICE for CATs: Model-Internal Confidence Estimation for Calibrating Agents with Tools 

**Authors**: Nishant Subramani, Jason Eisner, Justin Svegliato, Benjamin Van Durme, Yu Su, Sam Thomson  

**Link**: [PDF](https://arxiv.org/pdf/2504.20168)  

**Abstract**: Tool-using agents that act in the world need to be both useful and safe. Well-calibrated model confidences can be used to weigh the risk versus reward of potential actions, but prior work shows that many models are poorly calibrated. Inspired by interpretability literature exploring the internals of models, we propose a novel class of model-internal confidence estimators (MICE) to better assess confidence when calling tools. MICE first decodes from each intermediate layer of the language model using logitLens and then computes similarity scores between each layer's generation and the final output. These features are fed into a learned probabilistic classifier to assess confidence in the decoded output. On the simulated trial and error (STE) tool-calling dataset using Llama3 models, we find that MICE beats or matches the baselines on smoothed expected calibration error. Using MICE confidences to determine whether to call a tool significantly improves over strong baselines on a new metric, expected tool-calling utility. Further experiments show that MICE is sample-efficient, can generalize zero-shot to unseen APIs, and results in higher tool-calling utility in scenarios with varying risk levels. Our code is open source, available at this https URL. 

---
# ResearchCodeAgent: An LLM Multi-Agent System for Automated Codification of Research Methodologies 

**Authors**: Shubham Gandhi, Dhruv Shah, Manasi Patwardhan, Lovekesh Vig, Gautam Shroff  

**Link**: [PDF](https://arxiv.org/pdf/2504.20117)  

**Abstract**: In this paper we introduce ResearchCodeAgent, a novel multi-agent system leveraging large language models (LLMs) agents to automate the codification of research methodologies described in machine learning literature. The system bridges the gap between high-level research concepts and their practical implementation, allowing researchers auto-generating code of existing research papers for benchmarking or building on top-of existing methods specified in the literature with availability of partial or complete starter code. ResearchCodeAgent employs a flexible agent architecture with a comprehensive action suite, enabling context-aware interactions with the research environment. The system incorporates a dynamic planning mechanism, utilizing both short and long-term memory to adapt its approach iteratively. We evaluate ResearchCodeAgent on three distinct machine learning tasks with distinct task complexity and representing different parts of the ML pipeline: data augmentation, optimization, and data batching. Our results demonstrate the system's effectiveness and generalizability, with 46.9% of generated code being high-quality and error-free, and 25% showing performance improvements over baseline implementations. Empirical analysis shows an average reduction of 57.9% in coding time compared to manual implementation. We observe higher gains for more complex tasks. ResearchCodeAgent represents a significant step towards automating the research implementation process, potentially accelerating the pace of machine learning research. 

---
# RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning 

**Authors**: Zihan Wang, Kangrui Wang, Qineng Wang, Pingyue Zhang, Linjie Li, Zhengyuan Yang, Kefan Yu, Minh Nhat Nguyen, Licheng Liu, Eli Gottlieb, Monica Lam, Yiping Lu, Kyunghyun Cho, Jiajun Wu, Li Fei-Fei, Lijuan Wang, Yejin Choi, Manling Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.20073)  

**Abstract**: Training large language models (LLMs) as interactive agents presents unique challenges including long-horizon decision making and interacting with stochastic environment feedback. While reinforcement learning (RL) has enabled progress in static tasks, multi-turn agent RL training remains underexplored. We propose StarPO (State-Thinking-Actions-Reward Policy Optimization), a general framework for trajectory-level agent RL, and introduce RAGEN, a modular system for training and evaluating LLM agents. Our study on three stylized environments reveals three core findings. First, our agent RL training shows a recurring mode of Echo Trap where reward variance cliffs and gradient spikes; we address this with StarPO-S, a stabilized variant with trajectory filtering, critic incorporation, and decoupled clipping. Second, we find the shaping of RL rollouts would benefit from diverse initial states, medium interaction granularity and more frequent sampling. Third, we show that without fine-grained, reasoning-aware reward signals, agent reasoning hardly emerge through multi-turn RL and they may show shallow strategies or hallucinated thoughts. Code and environments are available at this https URL. 

---
# Cognitive maps are generative programs 

**Authors**: Marta Kryven, Cole Wyeth, Aidan Curtis, Kevin Ellis  

**Link**: [PDF](https://arxiv.org/pdf/2504.20628)  

**Abstract**: Making sense of the world and acting in it relies on building simplified mental representations that abstract away aspects of reality. This principle of cognitive mapping is universal to agents with limited resources. Living organisms, people, and algorithms all face the problem of forming functional representations of their world under various computing constraints. In this work, we explore the hypothesis that human resource-efficient planning may arise from representing the world as predictably structured. Building on the metaphor of concepts as programs, we propose that cognitive maps can take the form of generative programs that exploit predictability and redundancy, in contrast to directly encoding spatial layouts. We use a behavioral experiment to show that people who navigate in structured spaces rely on modular planning strategies that align with programmatic map representations. We describe a computational model that predicts human behavior in a variety of structured scenarios. This model infers a small distribution over possible programmatic cognitive maps conditioned on human prior knowledge of the world, and uses this distribution to generate resource-efficient plans. Our models leverages a Large Language Model as an embedding of human priors, implicitly learned through training on a vast corpus of human data. Our model demonstrates improved computational efficiency, requires drastically less memory, and outperforms unstructured planning algorithms with cognitive constraints at predicting human behavior, suggesting that human planning strategies rely on programmatic cognitive maps. 

---
# TAMO:Fine-Grained Root Cause Analysis via Tool-Assisted LLM Agent with Multi-Modality Observation Data 

**Authors**: Qi Wang, Xiao Zhang, Mingyi Li, Yuan Yuan, Mengbai Xiao, Fuzhen Zhuang, Dongxiao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.20462)  

**Abstract**: With the development of distributed systems, microservices and cloud native technologies have become central to modern enterprise software development. Despite bringing significant advantages, these technologies also increase system complexity and operational challenges. Traditional root cause analysis (RCA) struggles to achieve automated fault response, heavily relying on manual intervention. In recent years, large language models (LLMs) have made breakthroughs in contextual inference and domain knowledge integration, providing new solutions for Artificial Intelligence for Operations (AIOps). However, Existing LLM-based approaches face three key challenges: text input constraints, dynamic service dependency hallucinations, and context window limitations. To address these issues, we propose a tool-assisted LLM agent with multi-modality observation data, namely TAMO, for fine-grained RCA. It unifies multi-modal observational data into time-aligned representations to extract consistent features and employs specialized root cause localization and fault classification tools for perceiving the contextual environment. This approach overcomes the limitations of LLM in handling real-time changing service dependencies and raw observational data and guides LLM to generate repair strategies aligned with system contexts by structuring key information into a prompt. Experimental results show that TAMO performs well in root cause analysis when dealing with public datasets characterized by heterogeneity and common fault types, demonstrating its effectiveness. 

---
# PaRT: Enhancing Proactive Social Chatbots with Personalized Real-Time Retrieval 

**Authors**: Zihan Niu, Zheyong Xie, Shaosheng Cao, Chonggang Lu, Zheyu Ye, Tong Xu, Zuozhu Liu, Yan Gao, Jia Chen, Zhe Xu, Yi Wu, Yao Hu  

**Link**: [PDF](https://arxiv.org/pdf/2504.20624)  

**Abstract**: Social chatbots have become essential intelligent companions in daily scenarios ranging from emotional support to personal interaction. However, conventional chatbots with passive response mechanisms usually rely on users to initiate or sustain dialogues by bringing up new topics, resulting in diminished engagement and shortened dialogue duration. In this paper, we present PaRT, a novel framework enabling context-aware proactive dialogues for social chatbots through personalized real-time retrieval and generation. Specifically, PaRT first integrates user profiles and dialogue context into a large language model (LLM), which is initially prompted to refine user queries and recognize their underlying intents for the upcoming conversation. Guided by refined intents, the LLM generates personalized dialogue topics, which then serve as targeted queries to retrieve relevant passages from RedNote. Finally, we prompt LLMs with summarized passages to generate knowledge-grounded and engagement-optimized responses. Our approach has been running stably in a real-world production environment for more than 30 days, achieving a 21.77\% improvement in the average duration of dialogues. 

---
# A Summary on GUI Agents with Foundation Models Enhanced by Reinforcement Learning 

**Authors**: Jiahao Li, Kaer Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.20464)  

**Abstract**: Graphical User Interface (GUI) agents, driven by Multi-modal Large Language Models (MLLMs), have emerged as a promising paradigm for enabling intelligent interaction with digital systems. This paper provides a structured summary of recent advances in GUI agents, focusing on architectures enhanced by Reinforcement Learning (RL). We first formalize GUI agent tasks as Markov Decision Processes and discuss typical execution environments and evaluation metrics. We then review the modular architecture of (M)LLM-based GUI agents, covering Perception, Planning, and Acting modules, and trace their evolution through representative works. Furthermore, we categorize GUI agent training methodologies into Prompt-based, Supervised Fine-Tuning (SFT)-based, and RL-based approaches, highlighting the progression from simple prompt engineering to dynamic policy learning via RL. Our summary illustrates how recent innovations in multimodal perception, decision reasoning, and adaptive action generation have significantly improved the generalization and robustness of GUI agents in complex real-world environments. We conclude by identifying key challenges and future directions for building more capable and reliable GUI agents. 

---
# Toward Efficient Exploration by Large Language Model Agents 

**Authors**: Dilip Arumugam, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2504.20997)  

**Abstract**: A burgeoning area within reinforcement learning (RL) is the design of sequential decision-making agents centered around large language models (LLMs). While autonomous decision-making agents powered by modern LLMs could facilitate numerous real-world applications, such successes demand agents that are capable of data-efficient RL. One key obstacle to achieving data efficiency in RL is exploration, a challenge that we demonstrate many recent proposals for LLM agent designs struggle to contend with. Meanwhile, classic algorithms from the RL literature known to gracefully address exploration require technical machinery that can be challenging to operationalize in purely natural language settings. In this work, rather than relying on finetuning or in-context learning to coax LLMs into implicitly imitating a RL algorithm, we illustrate how LLMs can be used to explicitly implement an existing RL algorithm (Posterior Sampling for Reinforcement Learning) whose capacity for statistically-efficient exploration is already well-studied. We offer empirical results demonstrating how our LLM-based implementation of a known, data-efficient RL algorithm can be considerably more effective in natural language tasks that demand prudent exploration. 

---
# Evolution of AI in Education: Agentic Workflows 

**Authors**: Firuz Kamalov, David Santandreu Calonge, Linda Smail, Dilshod Azizov, Dimple R. Thadani, Theresa Kwong, Amara Atif  

**Link**: [PDF](https://arxiv.org/pdf/2504.20082)  

**Abstract**: Artificial intelligence (AI) has transformed various aspects of education, with large language models (LLMs) driving advancements in automated tutoring, assessment, and content generation. However, conventional LLMs are constrained by their reliance on static training data, limited adaptability, and lack of reasoning. To address these limitations and foster more sustainable technological practices, AI agents have emerged as a promising new avenue for educational innovation. In this review, we examine agentic workflows in education according to four major paradigms: reflection, planning, tool use, and multi-agent collaboration. We critically analyze the role of AI agents in education through these key design paradigms, exploring their advantages, applications, and challenges. To illustrate the practical potential of agentic systems, we present a proof-of-concept application: a multi-agent framework for automated essay scoring. Preliminary results suggest this agentic approach may offer improved consistency compared to stand-alone LLMs. Our findings highlight the transformative potential of AI agents in educational settings while underscoring the need for further research into their interpretability, trustworthiness, and sustainable impact on pedagogical impact. 

---
# Modeling AI-Human Collaboration as a Multi-Agent Adaptation 

**Authors**: Prothit Sen, Sai Mihir Jakkaraju  

**Link**: [PDF](https://arxiv.org/pdf/2504.20903)  

**Abstract**: We develop an agent-based simulation to formalize AI-human collaboration as a function of task structure, advancing a generalizable framework for strategic decision-making in organizations. Distinguishing between heuristic-based human adaptation and rule-based AI search, we model interactions across modular (parallel) and sequenced (interdependent) tasks using an NK model. Our results reveal that in modular tasks, AI often substitutes for humans - delivering higher payoffs unless human expertise is very high, and the AI search space is either narrowly focused or extremely broad. In sequenced tasks, interesting complementarities emerge. When an expert human initiates the search and AI subsequently refines it, aggregate performance is maximized. Conversely, when AI leads, excessive heuristic refinement by the human can reduce payoffs. We also show that even "hallucinatory" AI - lacking memory or structure - can improve outcomes when augmenting low-capability humans by helping escape local optima. These results yield a robust implication: the effectiveness of AI-human collaboration depends less on context or industry, and more on the underlying task structure. By elevating task decomposition as the central unit of analysis, our model provides a transferable lens for strategic decision-making involving humans and an agentic AI across diverse organizational settings. 

---
# Reinforcement Learning for LLM Reasoning Under Memory Constraints 

**Authors**: Alan Lee, Harry Tong  

**Link**: [PDF](https://arxiv.org/pdf/2504.20834)  

**Abstract**: We explore reinforcement learning (RL) techniques to enhance reasoning within targeted problem spaces in large language models (LLMs) under memory and compute constraints. Our focus is on critic-free methods compatible with LoRA fine-tuning on a single 40GB GPU, a common limitation in academic settings. We introduce S-GRPO, a memory-efficient variant of Group Relative Policy Optimization, and T-SPMO, a token-level prefix matching strategy for fine-grained credit assignment. Despite limited resources, when used to fine-tune Qwen2-1.5B both methods significantly improve SVAMP benchmark accuracy from 46% to above 70% using LoRA training. T-SPMO also excels in multi-digit multiplication tasks, underscoring the potential of RL fine-tuning under hardware constraints. Additionally, we find that our full-token GRPO baseline under LoRA fine-tuning did not improve model performance (compared to base model) on either task, suggesting that our memory-efficient methods may act as a form of regularization that stabilizes training when only a small subset of parameters are updated. 

---
# SoccerDiffusion: Toward Learning End-to-End Humanoid Robot Soccer from Gameplay Recordings 

**Authors**: Florian Vahl, Jörn Griepenburg, Jan Gutsche, Jasper Güldenstein, Jianwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.20808)  

**Abstract**: This paper introduces SoccerDiffusion, a transformer-based diffusion model designed to learn end-to-end control policies for humanoid robot soccer directly from real-world gameplay recordings. Using data collected from RoboCup competitions, the model predicts joint command trajectories from multi-modal sensor inputs, including vision, proprioception, and game state. We employ a distillation technique to enable real-time inference on embedded platforms that reduces the multi-step diffusion process to a single step. Our results demonstrate the model's ability to replicate complex motion behaviors such as walking, kicking, and fall recovery both in simulation and on physical robots. Although high-level tactical behavior remains limited, this work provides a robust foundation for subsequent reinforcement learning or preference optimization methods. We release the dataset, pretrained models, and code under: this https URL 

---
# Cooking Up Creativity: A Cognitively-Inspired Approach for Enhancing LLM Creativity through Structured Representations 

**Authors**: Moran Mizrahi, Chen Shani, Gabriel Stanovsky, Dan Jurafsky, Dafna Shahaf  

**Link**: [PDF](https://arxiv.org/pdf/2504.20643)  

**Abstract**: Large Language Models (LLMs) excel at countless tasks, yet struggle with creativity. In this paper, we introduce a novel approach that couples LLMs with structured representations and cognitively inspired manipulations to generate more creative and diverse ideas. Our notion of creativity goes beyond superficial token-level variations; rather, we explicitly recombine structured representations of existing ideas, allowing our algorithm to effectively explore the more abstract landscape of ideas. We demonstrate our approach in the culinary domain with DishCOVER, a model that generates creative recipes. Experiments comparing our model's results to those of GPT-4o show greater diversity. Domain expert evaluations reveal that our outputs, which are mostly coherent and feasible culinary creations, significantly surpass GPT-4o in terms of novelty, thus outperforming it in creative generation. We hope our work inspires further research into structured creativity in AI. 

---
# ARCS: Agentic Retrieval-Augmented Code Synthesis with Iterative Refinement 

**Authors**: Manish Bhattarai, Miguel Cordova, Javier Santos, Dan O'Malley  

**Link**: [PDF](https://arxiv.org/pdf/2504.20434)  

**Abstract**: In supercomputing, efficient and optimized code generation is essential to leverage high-performance systems effectively. We propose Agentic Retrieval-Augmented Code Synthesis (ARCS), an advanced framework for accurate, robust, and efficient code generation, completion, and translation. ARCS integrates Retrieval-Augmented Generation (RAG) with Chain-of-Thought (CoT) reasoning to systematically break down and iteratively refine complex programming tasks. An agent-based RAG mechanism retrieves relevant code snippets, while real-time execution feedback drives the synthesis of candidate solutions. This process is formalized as a state-action search tree optimization, balancing code correctness with editing efficiency. Evaluations on the Geeks4Geeks and HumanEval benchmarks demonstrate that ARCS significantly outperforms traditional prompting methods in translation and generation quality. By enabling scalable and precise code synthesis, ARCS offers transformative potential for automating and optimizing code development in supercomputing applications, enhancing computational resource utilization. 

---
# CrashFixer: A crash resolution agent for the Linux kernel 

**Authors**: Alex Mathai, Chenxi Huang, Suwei Ma, Jihwan Kim, Hailie Mitchell, Aleksandr Nogikh, Petros Maniatis, Franjo Ivančić, Junfeng Yang, Baishakhi Ray  

**Link**: [PDF](https://arxiv.org/pdf/2504.20412)  

**Abstract**: Code large language models (LLMs) have shown impressive capabilities on a multitude of software engineering tasks. In particular, they have demonstrated remarkable utility in the task of code repair. However, common benchmarks used to evaluate the performance of code LLMs are often limited to small-scale settings. In this work, we build upon kGym, which shares a benchmark for system-level Linux kernel bugs and a platform to run experiments on the Linux kernel.
This paper introduces CrashFixer, the first LLM-based software repair agent that is applicable to Linux kernel bugs. Inspired by the typical workflow of a kernel developer, we identify the key capabilities an expert developer leverages to resolve a kernel crash. Using this as our guide, we revisit the kGym platform and identify key system improvements needed to practically run LLM-based agents at the scale of the Linux kernel (50K files and 20M lines of code). We implement these changes by extending kGym to create an improved platform - called kGymSuite, which will be open-sourced. Finally, the paper presents an evaluation of various repair strategies for such complex kernel bugs and showcases the value of explicitly generating a hypothesis before attempting to fix bugs in complex systems such as the Linux kernel. We also evaluated CrashFixer's capabilities on still open bugs, and found at least two patch suggestions considered plausible to resolve the reported bug. 

---
# AKIBoards: A Structure-Following Multiagent System for Predicting Acute Kidney Injury 

**Authors**: David Gordon, Panayiotis Petousis, Susanne B. Nicholas, Alex A.T. Bui  

**Link**: [PDF](https://arxiv.org/pdf/2504.20368)  

**Abstract**: Diagnostic reasoning entails a physician's local (mental) model based on an assumed or known shared perspective (global model) to explain patient observations with evidence assigned towards a clinical assessment. But in several (complex) medical situations, multiple experts work together as a team to optimize health evaluation and decision-making by leveraging different perspectives. Such consensus-driven reasoning reflects individual knowledge contributing toward a broader perspective on the patient. In this light, we introduce STRUCture-following for Multiagent Systems (STRUC-MAS), a framework automating the learning of these global models and their incorporation as prior beliefs for agents in multiagent systems (MAS) to follow. We demonstrate proof of concept with a prosocial MAS application for predicting acute kidney injuries (AKIs). In this case, we found that incorporating a global structure enabled multiple agents to achieve better performance (average precision, AP) in predicting AKI 48 hours before onset (structure-following-fine-tuned, SF-FT, AP=0.195; SF-FT-retrieval-augmented generation, SF-FT-RAG, AP=0.194) vs. baseline (non-structure-following-FT, NSF-FT, AP=0.141; NSF-FT-RAG, AP=0.180) for balanced precision-weighted-recall-weighted voting. Markedly, SF-FT agents with higher recall scores reported lower confidence levels in the initial round on true positive and false negative cases. But after explicit interactions, their confidence in their decisions increased (suggesting reinforced belief). In contrast, the SF-FT agent with the lowest recall decreased its confidence in true positive and false negative cases (suggesting a new belief). This approach suggests that learning and leveraging global structures in MAS is necessary prior to achieving competitive classification and diagnostic reasoning performance. 

---
# AutoP2C: An LLM-Based Agent Framework for Code Repository Generation from Multimodal Content in Academic Papers 

**Authors**: Zijie Lin, Yiqing Shen, Qilin Cai, He Sun, Jinrui Zhou, Mingjun Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2504.20115)  

**Abstract**: Machine Learning (ML) research is spread through academic papers featuring rich multimodal content, including text, diagrams, and tabular results. However, translating these multimodal elements into executable code remains a challenging and time-consuming process that requires substantial ML expertise. We introduce ``Paper-to-Code'' (P2C), a novel task that transforms the multimodal content of scientific publications into fully executable code repositories, which extends beyond the existing formulation of code generation that merely converts textual descriptions into isolated code snippets. To automate the P2C process, we propose AutoP2C, a multi-agent framework based on large language models that processes both textual and visual content from research papers to generate complete code repositories. Specifically, AutoP2C contains four stages: (1) repository blueprint extraction from established codebases, (2) multimodal content parsing that integrates information from text, equations, and figures, (3) hierarchical task decomposition for structured code generation, and (4) iterative feedback-driven debugging to ensure functionality and performance. Evaluation on a benchmark of eight research papers demonstrates the effectiveness of AutoP2C, which can successfully generate executable code repositories for all eight papers, while OpenAI-o1 or DeepSeek-R1 can only produce runnable code for one paper. The code is available at this https URL. 

---
