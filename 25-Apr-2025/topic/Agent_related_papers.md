# DataScout: Automatic Data Fact Retrieval for Statement Augmentation with an LLM-Based Agent 

**Authors**: Chuer Chen, Yuqi Liu, Danqing Shi, Shixiong Cao, Nan Cao  

**Link**: [PDF](https://arxiv.org/pdf/2504.17334)  

**Abstract**: A data story typically integrates data facts from multiple perspectives and stances to construct a comprehensive and objective narrative. However, retrieving these facts demands time for data search and challenges the creator's analytical skills. In this work, we introduce DataScout, an interactive system that automatically performs reasoning and stance-based data facts retrieval to augment the user's statement. Particularly, DataScout leverages an LLM-based agent to construct a retrieval tree, enabling collaborative control of its expansion between users and the agent. The interface visualizes the retrieval tree as a mind map that eases users to intuitively steer the retrieval direction and effectively engage in reasoning and analysis. We evaluate the proposed system through case studies and in-depth expert interviews. Our evaluation demonstrates that DataScout can effectively retrieve multifaceted data facts from different stances, helping users verify their statements and enhance the credibility of their stories. 

---
# A Desideratum for Conversational Agents: Capabilities, Challenges, and Future Directions 

**Authors**: Emre Can Acikgoz, Cheng Qian, Hongru Wang, Vardhan Dongre, Xiusi Chen, Heng Ji, Dilek Hakkani-Tür, Gokhan Tur  

**Link**: [PDF](https://arxiv.org/pdf/2504.16939)  

**Abstract**: Recent advances in Large Language Models (LLMs) have propelled conversational AI from traditional dialogue systems into sophisticated agents capable of autonomous actions, contextual awareness, and multi-turn interactions with users. Yet, fundamental questions about their capabilities, limitations, and paths forward remain open. This survey paper presents a desideratum for next-generation Conversational Agents - what has been achieved, what challenges persist, and what must be done for more scalable systems that approach human-level intelligence. To that end, we systematically analyze LLM-driven Conversational Agents by organizing their capabilities into three primary dimensions: (i) Reasoning - logical, systematic thinking inspired by human intelligence for decision making, (ii) Monitor - encompassing self-awareness and user interaction monitoring, and (iii) Control - focusing on tool utilization and policy following. Building upon this, we introduce a novel taxonomy by classifying recent work on Conversational Agents around our proposed desideratum. We identify critical research gaps and outline key directions, including realistic evaluations, long-term multi-turn reasoning skills, self-evolution capabilities, collaborative and multi-agent task completion, personalization, and proactivity. This work aims to provide a structured foundation, highlight existing limitations, and offer insights into potential future research directions for Conversational Agents, ultimately advancing progress toward Artificial General Intelligence (AGI). We maintain a curated repository of papers at: this https URL. 

---
# AUTHENTICATION: Identifying Rare Failure Modes in Autonomous Vehicle Perception Systems using Adversarially Guided Diffusion Models 

**Authors**: Mohammad Zarei, Melanie A Jutras, Eliana Evans, Mike Tan, Omid Aaramoon  

**Link**: [PDF](https://arxiv.org/pdf/2504.17179)  

**Abstract**: Autonomous Vehicles (AVs) rely on artificial intelligence (AI) to accurately detect objects and interpret their surroundings. However, even when trained using millions of miles of real-world data, AVs are often unable to detect rare failure modes (RFMs). The problem of RFMs is commonly referred to as the "long-tail challenge", due to the distribution of data including many instances that are very rarely seen. In this paper, we present a novel approach that utilizes advanced generative and explainable AI techniques to aid in understanding RFMs. Our methods can be used to enhance the robustness and reliability of AVs when combined with both downstream model training and testing. We extract segmentation masks for objects of interest (e.g., cars) and invert them to create environmental masks. These masks, combined with carefully crafted text prompts, are fed into a custom diffusion model. We leverage the Stable Diffusion inpainting model guided by adversarial noise optimization to generate images containing diverse environments designed to evade object detection models and expose vulnerabilities in AI systems. Finally, we produce natural language descriptions of the generated RFMs that can guide developers and policymakers to improve the safety and reliability of AV systems. 

---
# Leveraging LLMs as Meta-Judges: A Multi-Agent Framework for Evaluating LLM Judgments 

**Authors**: Yuran Li, Jama Hussein Mohamud, Chongren Sun, Di Wu, Benoit Boulet  

**Link**: [PDF](https://arxiv.org/pdf/2504.17087)  

**Abstract**: Large language models (LLMs) are being widely applied across various fields, but as tasks become more complex, evaluating their responses is increasingly challenging. Compared to human evaluators, the use of LLMs to support performance evaluation offers a more efficient alternative. However, most studies focus mainly on aligning LLMs' judgments with human preferences, overlooking the existence of biases and mistakes in human judgment. Furthermore, how to select suitable LLM judgments given multiple potential LLM responses remains underexplored. To address these two aforementioned issues, we propose a three-stage meta-judge selection pipeline: 1) developing a comprehensive rubric with GPT-4 and human experts, 2) using three advanced LLM agents to score judgments, and 3) applying a threshold to filter out low-scoring judgments. Compared to methods using a single LLM as both judge and meta-judge, our pipeline introduces multi-agent collaboration and a more comprehensive rubric. Experimental results on the JudgeBench dataset show about 15.55\% improvement compared to raw judgments and about 8.37\% improvement over the single-agent baseline. Our work demonstrates the potential of LLMs as meta-judges and lays the foundation for future research on constructing preference datasets for LLM-as-a-judge reinforcement learning. 

---
# Cracking the Code of Action: a Generative Approach to Affordances for Reinforcement Learning 

**Authors**: Lynn Cherif, Flemming Kondrup, David Venuto, Ankit Anand, Doina Precup, Khimya Khetarpal  

**Link**: [PDF](https://arxiv.org/pdf/2504.17282)  

**Abstract**: Agents that can autonomously navigate the web through a graphical user interface (GUI) using a unified action space (e.g., mouse and keyboard actions) can require very large amounts of domain-specific expert demonstrations to achieve good performance. Low sample efficiency is often exacerbated in sparse-reward and large-action-space environments, such as a web GUI, where only a few actions are relevant in any given situation. In this work, we consider the low-data regime, with limited or no access to expert behavior. To enable sample-efficient learning, we explore the effect of constraining the action space through $\textit{intent-based affordances}$ -- i.e., considering in any situation only the subset of actions that achieve a desired outcome. We propose $\textbf{Code as Generative Affordances}$ $(\textbf{$\texttt{CoGA}$})$, a method that leverages pre-trained vision-language models (VLMs) to generate code that determines affordable actions through implicit intent-completion functions and using a fully-automated program generation and verification pipeline. These programs are then used in-the-loop of a reinforcement learning agent to return a set of affordances given a pixel observation. By greatly reducing the number of actions that an agent must consider, we demonstrate on a wide range of tasks in the MiniWob++ benchmark that: $\textbf{1)}$ $\texttt{CoGA}$ is orders of magnitude more sample efficient than its RL agent, $\textbf{2)}$ $\texttt{CoGA}$'s programs can generalize within a family of tasks, and $\textbf{3)}$ $\texttt{CoGA}$ performs better or on par compared with behavior cloning when a small number of expert demonstrations is available. 

---
# Towards a HIPAA Compliant Agentic AI System in Healthcare 

**Authors**: Subash Neupane, Shaswata Mitra, Sudip Mittal, Shahram Rahimi  

**Link**: [PDF](https://arxiv.org/pdf/2504.17669)  

**Abstract**: Agentic AI systems powered by Large Language Models (LLMs) as their foundational reasoning engine, are transforming clinical workflows such as medical report generation and clinical summarization by autonomously analyzing sensitive healthcare data and executing decisions with minimal human oversight. However, their adoption demands strict compliance with regulatory frameworks such as Health Insurance Portability and Accountability Act (HIPAA), particularly when handling Protected Health Information (PHI). This work-in-progress paper introduces a HIPAA-compliant Agentic AI framework that enforces regulatory compliance through dynamic, context-aware policy enforcement. Our framework integrates three core mechanisms: (1) Attribute-Based Access Control (ABAC) for granular PHI governance, (2) a hybrid PHI sanitization pipeline combining regex patterns and BERT-based model to minimize leakage, and (3) immutable audit trails for compliance verification. 

---
# A Systematic Approach to Design Real-World Human-in-the-Loop Deep Reinforcement Learning: Salient Features, Challenges and Trade-offs 

**Authors**: Jalal Arabneydi, Saiful Islam, Srijita Das, Sai Krishna Gottipati, William Duguay, Cloderic Mars, Matthew E. Taylor, Matthew Guzdial, Antoine Fagette, Younes Zerouali  

**Link**: [PDF](https://arxiv.org/pdf/2504.17006)  

**Abstract**: With the growing popularity of deep reinforcement learning (DRL), human-in-the-loop (HITL) approach has the potential to revolutionize the way we approach decision-making problems and create new opportunities for human-AI collaboration. In this article, we introduce a novel multi-layered hierarchical HITL DRL algorithm that comprises three types of learning: self learning, imitation learning and transfer learning. In addition, we consider three forms of human inputs: reward, action and demonstration. Furthermore, we discuss main challenges, trade-offs and advantages of HITL in solving complex problems and how human information can be integrated in the AI solution systematically. To verify our technical results, we present a real-world unmanned aerial vehicles (UAV) problem wherein a number of enemy drones attack a restricted area. The objective is to design a scalable HITL DRL algorithm for ally drones to neutralize the enemy drones before they reach the area. To this end, we first implement our solution using an award-winning open-source HITL software called Cogment. We then demonstrate several interesting results such as (a) HITL leads to faster training and higher performance, (b) advice acts as a guiding direction for gradient methods and lowers variance, and (c) the amount of advice should neither be too large nor too small to avoid over-training and under-training. Finally, we illustrate the role of human-AI cooperation in solving two real-world complex scenarios, i.e., overloaded and decoy attacks. 

---
# Comprehend, Divide, and Conquer: Feature Subspace Exploration via Multi-Agent Hierarchical Reinforcement Learning 

**Authors**: Weiliang Zhang, Xiaohan Huang, Yi Du, Ziyue Qiao, Qingqing Long, Zhen Meng, Yuanchun Zhou, Meng Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2504.17356)  

**Abstract**: Feature selection aims to preprocess the target dataset, find an optimal and most streamlined feature subset, and enhance the downstream machine learning task. Among filter, wrapper, and embedded-based approaches, the reinforcement learning (RL)-based subspace exploration strategy provides a novel objective optimization-directed perspective and promising performance. Nevertheless, even with improved performance, current reinforcement learning approaches face challenges similar to conventional methods when dealing with complex datasets. These challenges stem from the inefficient paradigm of using one agent per feature and the inherent complexities present in the datasets. This observation motivates us to investigate and address the above issue and propose a novel approach, namely HRLFS. Our methodology initially employs a Large Language Model (LLM)-based hybrid state extractor to capture each feature's mathematical and semantic characteristics. Based on this information, features are clustered, facilitating the construction of hierarchical agents for each cluster and sub-cluster. Extensive experiments demonstrate the efficiency, scalability, and robustness of our approach. Compared to contemporary or the one-feature-one-agent RL-based approaches, HRLFS improves the downstream ML performance with iterative feature subspace exploration while accelerating total run time by reducing the number of agents involved. 

---
# Collaborative Multi-Agent Reinforcement Learning for Automated Feature Transformation with Graph-Driven Path Optimization 

**Authors**: Xiaohan Huang, Dongjie Wang, Zhiyuan Ning, Ziyue Qiao, Qingqing Long, Haowei Zhu, Yi Du, Min Wu, Yuanchun Zhou, Meng Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2504.17355)  

**Abstract**: Feature transformation methods aim to find an optimal mathematical feature-feature crossing process that generates high-value features and improves the performance of downstream machine learning tasks. Existing frameworks, though designed to mitigate manual costs, often treat feature transformations as isolated operations, ignoring dynamic dependencies between transformation steps. To address the limitations, we propose TCTO, a collaborative multi-agent reinforcement learning framework that automates feature engineering through graph-driven path optimization. The framework's core innovation lies in an evolving interaction graph that models features as nodes and transformations as edges. Through graph pruning and backtracking, it dynamically eliminates low-impact edges, reduces redundant operations, and enhances exploration stability. This graph also provides full traceability to empower TCTO to reuse high-utility subgraphs from historical transformations. To demonstrate the efficacy and adaptability of our approach, we conduct comprehensive experiments and case studies, which show superior performance across a range of datasets. 

---
# MCAF: Efficient Agent-based Video Understanding Framework through Multimodal Coarse-to-Fine Attention Focusing 

**Authors**: Shiwen Cao, Zhaoxing Zhang, Junming Jiao, Juyi Qiao, Guowen Song, Rong Shen  

**Link**: [PDF](https://arxiv.org/pdf/2504.17213)  

**Abstract**: Even in the era of rapid advances in large models, video understanding, particularly long videos, remains highly challenging. Compared with textual or image-based information, videos commonly contain more information with redundancy, requiring large models to strategically allocate attention at a global level for accurate comprehension. To address this, we propose MCAF, an agent-based, training-free framework perform video understanding through Multimodal Coarse-to-fine Attention Focusing. The key innovation lies in its ability to sense and prioritize segments of the video that are highly relevant to the understanding task. First, MCAF hierarchically concentrates on highly relevant frames through multimodal information, enhancing the correlation between the acquired contextual information and the query. Second, it employs a dilated temporal expansion mechanism to mitigate the risk of missing crucial details when extracting information from these concentrated frames. In addition, our framework incorporates a self-reflection mechanism utilizing the confidence level of the model's responses as feedback. By iteratively applying these two creative focusing strategies, it adaptively adjusts attention to capture highly query-connected context and thus improves response accuracy. MCAF outperforms comparable state-of-the-art methods on average. On the EgoSchema dataset, it achieves a remarkable 5% performance gain over the leading approach. Meanwhile, on Next-QA and IntentQA datasets, it outperforms the current state-of-the-art standard by 0.2% and 0.3% respectively. On the Video-MME dataset, which features videos averaging nearly an hour in length, MCAF also outperforms other agent-based methods. 

---
# Peer-Aware Cost Estimation in Nonlinear General-Sum Dynamic Games for Mutual Learning and Intent Inference 

**Authors**: Seyed Yousef Soltanian, Wenlong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.17129)  

**Abstract**: Human-robot interactions can be modeled as incomplete-information general-sum dynamic games since the objective functions of both agents are not explicitly known to each other. However, solving for equilibrium policies for such games presents a major challenge, especially if the games involve nonlinear underlying dynamics. To simplify the problem, existing work often assumes that one agent is an expert with complete information about its peer, which can lead to biased estimates and failures in coordination. To address this challenge, we propose a nonlinear peer-aware cost estimation (N-PACE) algorithm for general-sum dynamic games. In N-PACE, using iterative linear quadratic (LQ) approximation of the nonlinear general-sum game, each agent explicitly models the learning dynamics of its peer agent while inferring their objective functions, leading to unbiased fast learning in inferring the unknown objective function of the peer agent, which is critical for task completion and safety assurance. Additionally, we demonstrate how N-PACE enables \textbf{intent communication} in such multi-agent systems by explicitly modeling the peer's learning dynamics. 

---
# Robo-Troj: Attacking LLM-based Task Planners 

**Authors**: Mohaiminul Al Nahian, Zainab Altaweel, David Reitano, Sabbir Ahmed, Saumitra Lohokare, Shiqi Zhang, Adnan Siraj Rakin  

**Link**: [PDF](https://arxiv.org/pdf/2504.17070)  

**Abstract**: Robots need task planning methods to achieve goals that require more than individual actions. Recently, large language models (LLMs) have demonstrated impressive performance in task planning. LLMs can generate a step-by-step solution using a description of actions and the goal. Despite the successes in LLM-based task planning, there is limited research studying the security aspects of those systems. In this paper, we develop Robo-Troj, the first multi-trigger backdoor attack for LLM-based task planners, which is the main contribution of this work. As a multi-trigger attack, Robo-Troj is trained to accommodate the diversity of robot application domains. For instance, one can use unique trigger words, e.g., "herical", to activate a specific malicious behavior, e.g., cutting hand on a kitchen robot. In addition, we develop an optimization method for selecting the trigger words that are most effective. Through demonstrating the vulnerability of LLM-based planners, we aim to promote the development of secured robot systems. 

---
# Conversational Assistants to support Heart Failure Patients: comparing a Neurosymbolic Architecture with ChatGPT 

**Authors**: Anuja Tayal, Devika Salunke, Barbara Di Eugenio, Paula Allen-Meares, Eulalia Puig Abril, Olga Garcia, Carolyn Dickens, Andrew Boyd  

**Link**: [PDF](https://arxiv.org/pdf/2504.17753)  

**Abstract**: Conversational assistants are becoming more and more popular, including in healthcare, partly because of the availability and capabilities of Large Language Models. There is a need for controlled, probing evaluations with real stakeholders which can highlight advantages and disadvantages of more traditional architectures and those based on generative AI. We present a within-group user study to compare two versions of a conversational assistant that allows heart failure patients to ask about salt content in food. One version of the system was developed in-house with a neurosymbolic architecture, and one is based on ChatGPT. The evaluation shows that the in-house system is more accurate, completes more tasks and is less verbose than the one based on ChatGPT; on the other hand, the one based on ChatGPT makes fewer speech errors and requires fewer clarifications to complete the task. Patients show no preference for one over the other. 

---
# A RAG-Based Multi-Agent LLM System for Natural Hazard Resilience and Adaptation 

**Authors**: Yangxinyu Xie, Bowen Jiang, Tanwi Mallick, Joshua David Bergerson, John K. Hutchison, Duane R. Verner, Jordan Branham, M. Ross Alexander, Robert B. Ross, Yan Feng, Leslie-Anne Levy, Weijie Su, Camillo J. Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2504.17200)  

**Abstract**: Large language models (LLMs) are a transformational capability at the frontier of artificial intelligence and machine learning that can support decision-makers in addressing pressing societal challenges such as extreme natural hazard events. As generalized models, LLMs often struggle to provide context-specific information, particularly in areas requiring specialized knowledge. In this work we propose a retrieval-augmented generation (RAG)-based multi-agent LLM system to support analysis and decision-making in the context of natural hazards and extreme weather events. As a proof of concept, we present WildfireGPT, a specialized system focused on wildfire hazards. The architecture employs a user-centered, multi-agent design to deliver tailored risk insights across diverse stakeholder groups. By integrating natural hazard and extreme weather projection data, observational datasets, and scientific literature through an RAG framework, the system ensures both the accuracy and contextual relevance of the information it provides. Evaluation across ten expert-led case studies demonstrates that WildfireGPT significantly outperforms existing LLM-based solutions for decision support. 

---
