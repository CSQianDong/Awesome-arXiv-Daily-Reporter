# LLM-Gomoku: A Large Language Model-Based System for Strategic Gomoku with Self-Play and Reinforcement Learning 

**Authors**: Hui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.21683)  

**Abstract**: In recent years, large language models (LLMs) have shown significant advancements in natural language processing (NLP), with strong capa-bilities in generation, comprehension, and rea-soning. These models have found applications in education, intelligent decision-making, and gaming. However, effectively utilizing LLMs for strategic planning and decision-making in the game of Gomoku remains a challenge. This study aims to develop a Gomoku AI system based on LLMs, simulating the human learning process of playing chess. The system is de-signed to understand and apply Gomoku strat-egies and logic to make rational decisions. The research methods include enabling the model to "read the board," "understand the rules," "select strategies," and "evaluate positions," while en-hancing its abilities through self-play and rein-forcement learning. The results demonstrate that this approach significantly improves the se-lection of move positions, resolves the issue of generating illegal positions, and reduces pro-cess time through parallel position evaluation. After extensive self-play training, the model's Gomoku-playing capabilities have been notably enhanced. 

---
# Cognitive Science-Inspired Evaluation of Core Capabilities for Object Understanding in AI 

**Authors**: Danaja Rutar, Alva Markelius, Konstantinos Voudouris, José Hernández-Orallo, Lucy Cheke  

**Link**: [PDF](https://arxiv.org/pdf/2503.21668)  

**Abstract**: One of the core components of our world models is 'intuitive physics' - an understanding of objects, space, and causality. This capability enables us to predict events, plan action and navigate environments, all of which rely on a composite sense of objecthood. Despite its importance, there is no single, unified account of objecthood, though multiple theoretical frameworks provide insights. In the first part of this paper, we present a comprehensive overview of the main theoretical frameworks in objecthood research - Gestalt psychology, enactive cognition, and developmental psychology - and identify the core capabilities each framework attributes to object understanding, as well as what functional roles they play in shaping world models in biological agents. Given the foundational role of objecthood in world modelling, understanding objecthood is also essential in AI. In the second part of the paper, we evaluate how current AI paradigms approach and test objecthood capabilities compared to those in cognitive science. We define an AI paradigm as a combination of how objecthood is conceptualised, the methods used for studying objecthood, the data utilised, and the evaluation techniques. We find that, whilst benchmarks can detect that AI systems model isolated aspects of objecthood, the benchmarks cannot detect when AI systems lack functional integration across these capabilities, not solving the objecthood challenge fully. Finally, we explore novel evaluation approaches that align with the integrated vision of objecthood outlined in this paper. These methods are promising candidates for advancing from isolated object capabilities toward general-purpose AI with genuine object understanding in real-world contexts. 

---
# Unlocking the Potential of Past Research: Using Generative AI to Reconstruct Healthcare Simulation Models 

**Authors**: Thomas Monks, Alison Harper, Amy Heather  

**Link**: [PDF](https://arxiv.org/pdf/2503.21646)  

**Abstract**: Discrete-event simulation (DES) is widely used in healthcare Operations Research, but the models themselves are rarely shared. This limits their potential for reuse and long-term impact in the modelling and healthcare communities. This study explores the feasibility of using generative artificial intelligence (AI) to recreate published models using Free and Open Source Software (FOSS), based on the descriptions provided in an academic journal. Using a structured methodology, we successfully generated, tested and internally reproduced two DES models, including user interfaces. The reported results were replicated for one model, but not the other, likely due to missing information on distributions. These models are substantially more complex than AI-generated DES models published to date. Given the challenges we faced in prompt engineering, code generation, and model testing, we conclude that our iterative approach to model development, systematic comparison and testing, and the expertise of our team were necessary to the success of our recreated simulation models. 

---
# Towards Fully Automated Decision-Making Systems for Greenhouse Control: Challenges and Opportunities 

**Authors**: Yongshuai Liu, Taeyeong Choi, Xin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.21640)  

**Abstract**: Machine learning has been successful in building control policies to drive a complex system to desired states in various applications (e.g. games, robotics, etc.). To be specific, a number of parameters of policy can be automatically optimized from the observations of environment to be able to generate a sequence of decisions leading to the best performance. In this survey paper, we particularly explore such policy-learning techniques for another unique, practical use-case scenario--farming, in which critical decisions (e.g., water supply, heating, etc.) must be made in a timely manner to minimize risks (e.g., damage to plants) while maximizing the revenue (e.g., healthy crops) in the end. We first provide a broad overview of latest studies on it to identify not only domain-specific challenges but opportunities with potential solutions, some of which are suggested as promising directions for future research. Also, we then introduce our successful approach to being ranked second among 46 teams at the ''3rd Autonomous Greenhouse Challenge'' to use this specific example to discuss the lessons learned about important considerations for design to create autonomous farm-management systems. 

---
# UI-R1: Enhancing Action Prediction of GUI Agents by Reinforcement Learning 

**Authors**: Zhengxi Lu, Yuxiang Chai, Yaxuan Guo, Xi Yin, Liang Liu, Hao Wang, Guanjing Xiong, Hongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.21620)  

**Abstract**: The recent DeepSeek-R1 has showcased the emergence of reasoning capabilities in LLMs through reinforcement learning (RL) with rule-based rewards. Building on this idea, we are the first to explore how rule-based RL can enhance the reasoning capabilities of multimodal large language models (MLLMs) for graphic user interface (GUI) action prediction tasks. To this end, we curate a small yet high-quality dataset of 136 challenging tasks, encompassing five common action types on mobile devices. We also introduce a unified rule-based action reward, enabling model optimization via policy-based algorithms such as Group Relative Policy Optimization (GRPO). Experimental results demonstrate that our proposed data-efficient model, UI-R1-3B, achieves substantial improvements on both in-domain (ID) and out-of-domain (OOD) tasks. Specifically, on the ID benchmark AndroidControl, the action type accuracy improves by 15%, while grounding accuracy increases by 10.3%, compared with the base model (i.e. Qwen2.5-VL-3B). On the OOD GUI grounding benchmark ScreenSpot-Pro, our model surpasses the base model by 6.0% and achieves competitive performance with larger models (e.g., OS-Atlas-7B), which are trained via supervised fine-tuning (SFT) on 76K data. These results underscore the potential of rule-based reinforcement learning to advance GUI understanding and control, paving the way for future research in this domain. 

---
# GenEdit: Compounding Operators and Continuous Improvement to Tackle Text-to-SQL in the Enterprise 

**Authors**: Karime Maamari, Connor Landy, Amine Mhedhbi  

**Link**: [PDF](https://arxiv.org/pdf/2503.21602)  

**Abstract**: Recent advancements in Text-to-SQL, driven by large language models, are democratizing data access. Despite these advancements, enterprise deployments remain challenging due to the need to capture business-specific knowledge, handle complex queries, and meet expectations of continuous improvements. To address these issues, we designed and implemented GenEdit: our Text-to-SQL generation system that improves with user feedback. GenEdit builds and maintains a company-specific knowledge set, employs a pipeline of operators decomposing SQL generation, and uses feedback to update its knowledge set to improve future SQL generations.
We describe GenEdit's architecture made of two core modules: (i) decomposed SQL generation; and (ii) knowledge set edits based on user feedback. For generation, GenEdit leverages compounding operators to improve knowledge retrieval and to create a plan as chain-of-thought steps that guides generation. GenEdit first retrieves relevant examples in an initial retrieval stage where original SQL queries are decomposed into sub-statements, clauses or sub-queries. It then also retrieves instructions and schema elements. Using the retrieved contextual information, GenEdit then generates step-by-step plan in natural language on how to produce the query. Finally, GenEdit uses the plan to generate SQL, minimizing the need for model reasoning, which enhances complex SQL generation. If necessary, GenEdit regenerates the query based on syntactic and semantic errors. The knowledge set edits are recommended through an interactive copilot, allowing users to iterate on their feedback and to regenerate SQL queries as needed. Each generation uses staged edits which update the generation prompt. Once the feedback is submitted, it gets merged after passing regression testing and obtaining an approval, improving future generations. 

---
# debug-gym: A Text-Based Environment for Interactive Debugging 

**Authors**: Xingdi Yuan, Morgane M Moss, Charbel El Feghali, Chinmay Singh, Darya Moldavskaya, Drew MacPhee, Lucas Caccia, Matheus Pereira, Minseon Kim, Alessandro Sordoni, Marc-Alexandre Côté  

**Link**: [PDF](https://arxiv.org/pdf/2503.21557)  

**Abstract**: Large Language Models (LLMs) are increasingly relied upon for coding tasks, yet in most scenarios it is assumed that all relevant information can be either accessed in context or matches their training data. We posit that LLMs can benefit from the ability to interactively explore a codebase to gather the information relevant to their task. To achieve this, we present a textual environment, namely debug-gym, for developing LLM-based agents in an interactive coding setting. Our environment is lightweight and provides a preset of useful tools, such as a Python debugger (pdb), designed to facilitate an LLM-based agent's interactive debugging. Beyond coding and debugging tasks, this approach can be generalized to other tasks that would benefit from information-seeking behavior by an LLM agent. 

---
# The Procedural Content Generation Benchmark: An Open-source Testbed for Generative Challenges in Games 

**Authors**: Ahmed Khalifa, Roberto Gallotta, Matthew Barthet, Antonios Liapis, Julian Togelius, Georgios N. Yannakakis  

**Link**: [PDF](https://arxiv.org/pdf/2503.21474)  

**Abstract**: This paper introduces the Procedural Content Generation Benchmark for evaluating generative algorithms on different game content creation tasks. The benchmark comes with 12 game-related problems with multiple variants on each problem. Problems vary from creating levels of different kinds to creating rule sets for simple arcade games. Each problem has its own content representation, control parameters, and evaluation metrics for quality, diversity, and controllability. This benchmark is intended as a first step towards a standardized way of comparing generative algorithms. We use the benchmark to score three baseline algorithms: a random generator, an evolution strategy, and a genetic algorithm. Results show that some problems are easier to solve than others, as well as the impact the chosen objective has on quality, diversity, and controllability of the generated artifacts. 

---
# Graph-to-Vision: Multi-graph Understanding and Reasoning using Vision-Language Models 

**Authors**: Ruizhou Li, Haiyun Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.21435)  

**Abstract**: Graph Neural Networks (GNNs), as the dominant paradigm for graph-structured learning, have long faced dual challenges of exponentially escalating computational complexity and inadequate cross-scenario generalization capability. With the rapid advancement of multimodal learning, Vision-Language Models (VLMs) have demonstrated exceptional cross-modal relational reasoning capabilities and generalization capacities, thereby opening up novel pathways for overcoming the inherent limitations of conventional graph learning paradigms. However, current research predominantly concentrates on investigating the single-graph reasoning capabilities of VLMs, which fundamentally fails to address the critical requirement for coordinated reasoning across multiple heterogeneous graph data in real-world application scenarios. To address these limitations, we propose the first multi-graph joint reasoning benchmark for VLMs. Our benchmark encompasses four graph categories: knowledge graphs, flowcharts, mind maps, and route maps,with each graph group accompanied by three progressively challenging instruction-response pairs. Leveraging this benchmark, we conducted comprehensive capability assessments of state-of-the-art VLMs and performed fine-tuning on open-source models. This study not only addresses the underexplored evaluation gap in multi-graph reasoning for VLMs but also empirically validates their generalization superiority in graph-structured learning. 

---
# Neuroplasticity in Artificial Intelligence -- An Overview and Inspirations on Drop In \& Out Learning 

**Authors**: Yupei Li, Manuel Milling, Björn W. Schuller  

**Link**: [PDF](https://arxiv.org/pdf/2503.21419)  

**Abstract**: Artificial Intelligence (AI) has achieved new levels of performance and spread in public usage with the rise of deep neural networks (DNNs). Initially inspired by human neurons and their connections, NNs have become the foundation of AI models for many advanced architectures. However, some of the most integral processes in the human brain, particularly neurogenesis and neuroplasticity in addition to the more spread neuroapoptosis have largely been ignored in DNN architecture design. Instead, contemporary AI development predominantly focuses on constructing advanced frameworks, such as large language models, which retain a static structure of neural connections during training and inference. In this light, we explore how neurogenesis, neuroapoptosis, and neuroplasticity can inspire future AI advances. Specifically, we examine analogous activities in artificial NNs, introducing the concepts of ``dropin'' for neurogenesis and revisiting ``dropout'' and structural pruning for neuroapoptosis. We additionally suggest neuroplasticity combining the two for future large NNs in ``life-long learning'' settings following the biological inspiration. We conclude by advocating for greater research efforts in this interdisciplinary domain and identifying promising directions for future exploration. 

---
# Federated Intelligence: When Large AI Models Meet Federated Fine-Tuning and Collaborative Reasoning at the Network Edge 

**Authors**: Wanli Ni, Haofeng Sun, Huiqing Ao, Hui Tian  

**Link**: [PDF](https://arxiv.org/pdf/2503.21412)  

**Abstract**: Large artificial intelligence (AI) models exhibit remarkable capabilities in various application scenarios, but deploying them at the network edge poses significant challenges due to issues such as data privacy, computational resources, and latency. In this paper, we explore federated fine-tuning and collaborative reasoning techniques to facilitate the implementation of large AI models in resource-constrained wireless networks. Firstly, promising applications of large AI models within specific domains are discussed. Subsequently, federated fine-tuning methods are proposed to adapt large AI models to specific tasks or environments at the network edge, effectively addressing the challenges associated with communication overhead and enhancing communication efficiency. These methodologies follow clustered, hierarchical, and asynchronous paradigms to effectively tackle privacy issues and eliminate data silos. Furthermore, to enhance operational efficiency and reduce latency, efficient frameworks for model collaborative reasoning are developed, which include decentralized horizontal collaboration, cloud-edge-end vertical collaboration, and multi-access collaboration. Next, simulation results demonstrate the effectiveness of our proposed methods in reducing the fine-tuning loss of large AI models across various downstream tasks. Finally, several open challenges and research opportunities are outlined. 

---
# Exploring the Roles of Large Language Models in Reshaping Transportation Systems: A Survey, Framework, and Roadmap 

**Authors**: Tong Nie, Jian Sun, Wei Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.21411)  

**Abstract**: Modern transportation systems face pressing challenges due to increasing demand, dynamic environments, and heterogeneous information integration. The rapid evolution of Large Language Models (LLMs) offers transformative potential to address these challenges. Extensive knowledge and high-level capabilities derived from pretraining evolve the default role of LLMs as text generators to become versatile, knowledge-driven task solvers for intelligent transportation systems. This survey first presents LLM4TR, a novel conceptual framework that systematically categorizes the roles of LLMs in transportation into four synergetic dimensions: information processors, knowledge encoders, component generators, and decision facilitators. Through a unified taxonomy, we systematically elucidate how LLMs bridge fragmented data pipelines, enhance predictive analytics, simulate human-like reasoning, and enable closed-loop interactions across sensing, learning, modeling, and managing tasks in transportation systems. For each role, our review spans diverse applications, from traffic prediction and autonomous driving to safety analytics and urban mobility optimization, highlighting how emergent capabilities of LLMs such as in-context learning and step-by-step reasoning can enhance the operation and management of transportation systems. We further curate practical guidance, including available resources and computational guidelines, to support real-world deployment. By identifying challenges in existing LLM-based solutions, this survey charts a roadmap for advancing LLM-driven transportation research, positioning LLMs as central actors in the next generation of cyber-physical-social mobility ecosystems. Online resources can be found in the project page: this https URL. 

---
# Neuro-Symbolic Imitation Learning: Discovering Symbolic Abstractions for Skill Learning 

**Authors**: Leon Keller, Daniel Tanneberg, Jan Peters  

**Link**: [PDF](https://arxiv.org/pdf/2503.21406)  

**Abstract**: Imitation learning is a popular method for teaching robots new behaviors. However, most existing methods focus on teaching short, isolated skills rather than long, multi-step tasks. To bridge this gap, imitation learning algorithms must not only learn individual skills but also an abstract understanding of how to sequence these skills to perform extended tasks effectively. This paper addresses this challenge by proposing a neuro-symbolic imitation learning framework. Using task demonstrations, the system first learns a symbolic representation that abstracts the low-level state-action space. The learned representation decomposes a task into easier subtasks and allows the system to leverage symbolic planning to generate abstract plans. Subsequently, the system utilizes this task decomposition to learn a set of neural skills capable of refining abstract plans into actionable robot commands. Experimental results in three simulated robotic environments demonstrate that, compared to baselines, our neuro-symbolic approach increases data efficiency, improves generalization capabilities, and facilitates interpretability. 

---
# HybridoNet-Adapt: A Domain-Adapted Framework for Accurate Lithium-Ion Battery RUL Prediction 

**Authors**: Khoa Tran, Bao Huynh, Tri Le, Lam Pham, Vy-Rin Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2503.21392)  

**Abstract**: Accurate prediction of the remaining useful life (RUL) in Lithium-ion battery (LIB) health management systems is crucial for ensuring reliability and safety. Current methods typically assume that training and testing data share the same distribution, overlooking the benefits of incorporating diverse data sources to enhance model performance. To address this limitation, we introduce a data-independent RUL prediction framework along with its domain adaptation (DA) approach, which leverages heterogeneous data sources for improved target predictions. Our approach integrates comprehensive data preprocessing, including feature extraction, denoising, and normalization, with a data-independent prediction model that combines Long Short-Term Memory (LSTM), Multihead Attention, and a Neural Ordinary Differential Equation (NODE) block, termed HybridoNet. The domain-adapted version, HybridoNet Adapt, is trained using a novel technique inspired by the Domain-Adversarial Neural Network (DANN) framework, a regression ensemble method, and Maximum Mean Discrepancy (MMD) to learn domain-invariant features from labeled cycling data in the source and target domains. Experimental results demonstrate that our approach outperforms state-of-the-art techniques, providing reliable RUL predictions for real-world applications. 

---
# Using large language models to produce literature reviews: Usages and systematic biases of microphysics parametrizations in 2699 publications 

**Authors**: Tianhang Zhang, Shengnan Fu, David M. Schultz, Zhonghua Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.21352)  

**Abstract**: Large language models afford opportunities for using computers for intensive tasks, realizing research opportunities that have not been considered before. One such opportunity could be a systematic interrogation of the scientific literature. Here, we show how a large language model can be used to construct a literature review of 2699 publications associated with microphysics parametrizations in the Weather and Research Forecasting (WRF) model, with the goal of learning how they were used and their systematic biases, when simulating precipitation. The database was constructed of publications identified from Web of Science and Scopus searches. The large language model GPT-4 Turbo was used to extract information about model configurations and performance from the text of 2699 publications. Our results reveal the landscape of how nine of the most popular microphysics parameterizations have been used around the world: Lin, Ferrier, WRF Single-Moment, Goddard Cumulus Ensemble, Morrison, Thompson, and WRF Double-Moment. More studies used one-moment parameterizations before 2020 and two-moment parameterizations after 2020. Seven out of nine parameterizations tended to overestimate precipitation. However, systematic biases of parameterizations differed in various regions. Except simulations using the Lin, Ferrier, and Goddard parameterizations that tended to underestimate precipitation over almost all locations, the remaining six parameterizations tended to overestimate, particularly over China, southeast Asia, western United States, and central Africa. This method could be used by other researchers to help understand how the increasingly massive body of scientific literature can be harnessed through the power of artificial intelligence to solve their research problems. 

---
# HyperGraphRAG: Retrieval-Augmented Generation with Hypergraph-Structured Knowledge Representation 

**Authors**: Haoran Luo, Haihong E, Guanting Chen, Yandan Zheng, Xiaobao Wu, Yikai Guo, Qika Lin, Yu Feng, Zemin Kuang, Meina Song, Yifan Zhu, Luu Anh Tuan  

**Link**: [PDF](https://arxiv.org/pdf/2503.21322)  

**Abstract**: While standard Retrieval-Augmented Generation (RAG) based on chunks, GraphRAG structures knowledge as graphs to leverage the relations among entities. However, previous GraphRAG methods are limited by binary relations: one edge in the graph only connects two entities, which cannot well model the n-ary relations among more than two entities that widely exist in reality. To address this limitation, we propose HyperGraphRAG, a novel hypergraph-based RAG method that represents n-ary relational facts via hyperedges, modeling the complicated n-ary relations in the real world. To retrieve and generate over hypergraphs, we introduce a complete pipeline with a hypergraph construction method, a hypergraph retrieval strategy, and a hypergraph-guided generation mechanism. Experiments across medicine, agriculture, computer science, and law demonstrate that HyperGraphRAG outperforms standard RAG and GraphRAG in accuracy and generation quality. 

---
# Reinforced Model Merging 

**Authors**: Jiaqi Han, Jingwen Ye, Shunyu Liu, Haofei Zhang, Jie Song, Zunlei Feng, Mingli Song  

**Link**: [PDF](https://arxiv.org/pdf/2503.21272)  

**Abstract**: The success of large language models has garnered widespread attention for model merging techniques, especially training-free methods which combine model capabilities within the parameter space. However, two challenges remain: (1) uniform treatment of all parameters leads to performance degradation; (2) search-based algorithms are often inefficient. In this paper, we present an innovative framework termed Reinforced Model Merging (RMM), which encompasses an environment and agent tailored for merging tasks. These components interact to execute layer-wise merging actions, aiming to search the optimal merging architecture. Notably, RMM operates without any gradient computations on the original models, rendering it feasible for edge devices. Furthermore, by utilizing data subsets during the evaluation process, we addressed the bottleneck in the reward feedback phase, thereby accelerating RMM by up to 100 times. Extensive experiments demonstrate that RMM achieves state-of-the-art performance across various vision and NLP datasets and effectively overcomes the limitations of the existing baseline methods. Our code is available at this https URL. 

---
# Knowledge Graphs as World Models for Semantic Material-Aware Obstacle Handling in Autonomous Vehicles 

**Authors**: Ayush Bheemaiah, Seungyong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.21232)  

**Abstract**: The inability of autonomous vehicles (AVs) to infer the material properties of obstacles limits their decision-making capacity. While AVs rely on sensor systems such as cameras, LiDAR, and radar to detect obstacles, this study suggests combining sensors with a knowledge graph (KG)-based world model to improve AVs' comprehension of physical material qualities. Beyond sensor data, AVs can infer qualities such as malleability, density, and elasticity using a semantic KG that depicts the relationships between obstacles and their attributes. Using the CARLA autonomous driving simulator, we evaluated AV performance with and without KG integration. The findings demonstrate that the KG-based method improves obstacle management, which allows AVs to use material qualities to make better decisions about when to change lanes or apply emergency braking. For example, the KG-integrated AV changed lanes for hard impediments like traffic cones and successfully avoided collisions with flexible items such as plastic bags by passing over them. Compared to the control system, the KG framework demonstrated improved responsiveness to obstacles by resolving conflicting sensor data, causing emergency stops for 13.3% more cases. In addition, our method exhibits a 6.6% higher success rate in lane-changing maneuvers in experimental scenarios, particularly for larger, high-impact obstacles. While we focus particularly on autonomous driving, our work demonstrates the potential of KG-based world models to improve decision-making in embodied AI systems and scale to other domains, including robotics, healthcare, and environmental simulation. 

---
# Integrating Large Language Models For Monte Carlo Simulation of Chemical Reaction Networks 

**Authors**: Sadikshya Gyawali, Ashwini Mandal, Manish Dahal, Manish Awale, Sanjay Rijal, Shital Adhikari, Vaghawan Ojha  

**Link**: [PDF](https://arxiv.org/pdf/2503.21178)  

**Abstract**: Chemical reaction network is an important method for modeling and exploring complex biological processes, bio-chemical interactions and the behavior of different dynamics in system biology. But, formulating such reaction kinetics takes considerable time. In this paper, we leverage the efficiency of modern large language models to automate the stochastic monte carlo simulation of chemical reaction networks and enable the simulation through the reaction description provided in the form of natural languages. We also integrate this process into widely used simulation tool Copasi to further give the edge and ease to the modelers and researchers. In this work, we show the efficacy and limitations of the modern large language models to parse and create reaction kinetics for modelling complex chemical reaction processes. 

---
# A computational theory of evaluation for parameterisable subject 

**Authors**: Hedong Yan  

**Link**: [PDF](https://arxiv.org/pdf/2503.21138)  

**Abstract**: Evaluation is critical to advance decision making across domains, yet existing methodologies often struggle to balance theoretical rigor and practical scalability. In order to reduce the cost of experimental evaluation, we introduce a computational theory of evaluation for parameterisable subjects. We prove upper bounds of generalized evaluation error and generalized causal effect error of evaluation metric on subject. We also prove efficiency, and consistency to estimated causal effect of subject on metric by prediction. To optimize evaluation models, we propose a meta-learner to handle heterogeneous evaluation subjects space. Comparing with other computational approaches, our (conditional) evaluation model reduced 24.1%-99.0% evaluation errors across 12 scenes, including individual medicine, scientific simulation, business activities, and quantum trade. The evaluation time is reduced 3-7 order of magnitude comparing with experiments or simulations. 

---
# AskSport: Web Application for Sports Question-Answering 

**Authors**: Enzo B Onofre, Leonardo M P Moraes, Cristina D Aguiar  

**Link**: [PDF](https://arxiv.org/pdf/2503.21067)  

**Abstract**: This paper introduces AskSport, a question-answering web application about sports. It allows users to ask questions using natural language and retrieve the three most relevant answers, including related information and documents. The paper describes the characteristics and functionalities of the application, including use cases demonstrating its ability to return names and numerical values. AskSport and its implementation are available for public access on HuggingFace. 

---
# The Art of Tool Interface Design 

**Authors**: Yunnan Wu, Paul Chen, Deshank Baranwal, Jinlong Zhou, Jian Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2503.21036)  

**Abstract**: We present an agentic framework, Thinker, which achieves state of art performance in challenging reasoning tasks for realistic customer service scenarios that involve complex business logic and human interactions via long horizons. On the $\tau$-bench retail dataset, Thinker achieves 82.6\% success rate with GPT-4o (version 2024-06-01) (baseline: 68.3\%), and 81.9\% success rate with Llama-3.1 405B (baseline: 49.6\%), without any fine-tuning. Thinker effectively closes the gap in reasoning capabilities between the base models by introducing proper structure.
The key features of the Thinker framework are: (1) State-Machine Augmented Generation (SMAG), which represents business logic as state machines and the LLM uses state machines as tools. (2) Delegation of tasks from the main reasoning loop to LLM-powered tools. (3) Adaptive context management.
Our prompting-only solution achieves signficant gains, while still maintaining a standard agentic architecture with a ReAct style reasoning loop. The key is to innovate on the tool interface design, as exemplified by SMAG and the LLM-powered tools. 

---
# DEMENTIA-PLAN: An Agent-Based Framework for Multi-Knowledge Graph Retrieval-Augmented Generation in Dementia Care 

**Authors**: Yutong Song, Chenhan Lyu, Pengfei Zhang, Sabine Brunswicker, Nikil Dutt, Amir Rahmani  

**Link**: [PDF](https://arxiv.org/pdf/2503.20950)  

**Abstract**: Mild-stage dementia patients primarily experience two critical symptoms: severe memory loss and emotional instability. To address these challenges, we propose DEMENTIA-PLAN, an innovative retrieval-augmented generation framework that leverages large language models to enhance conversational support. Our model employs a multiple knowledge graph architecture, integrating various dimensional knowledge representations including daily routine graphs and life memory graphs. Through this multi-graph architecture, DEMENTIA-PLAN comprehensively addresses both immediate care needs and facilitates deeper emotional resonance through personal memories, helping stabilize patient mood while providing reliable memory support. Our notable innovation is the self-reflection planning agent, which systematically coordinates knowledge retrieval and semantic integration across multiple knowledge graphs, while scoring retrieved content from daily routine and life memory graphs to dynamically adjust their retrieval weights for optimized response generation. DEMENTIA-PLAN represents a significant advancement in the clinical application of large language models for dementia care, bridging the gap between AI tools and caregivers interventions. 

---
# StyleMotif: Multi-Modal Motion Stylization using Style-Content Cross Fusion 

**Authors**: Ziyu Guo, Young Yoon Lee, Joseph Liu, Yizhak Ben-Shabat, Victor Zordan, Mubbasir Kapadia  

**Link**: [PDF](https://arxiv.org/pdf/2503.21775)  

**Abstract**: We present StyleMotif, a novel Stylized Motion Latent Diffusion model, generating motion conditioned on both content and style from multiple modalities. Unlike existing approaches that either focus on generating diverse motion content or transferring style from sequences, StyleMotif seamlessly synthesizes motion across a wide range of content while incorporating stylistic cues from multi-modal inputs, including motion, text, image, video, and audio. To achieve this, we introduce a style-content cross fusion mechanism and align a style encoder with a pre-trained multi-modal model, ensuring that the generated motion accurately captures the reference style while preserving realism. Extensive experiments demonstrate that our framework surpasses existing methods in stylized motion generation and exhibits emergent capabilities for multi-modal motion stylization, enabling more nuanced motion synthesis. Source code and pre-trained models will be released upon acceptance. Project Page: this https URL 

---
# Stable-SCore: A Stable Registration-based Framework for 3D Shape Correspondence 

**Authors**: Haolin Liu, Xiaohang Zhan, Zizheng Yan, Zhongjin Luo, Yuxin Wen, Xiaoguang Han  

**Link**: [PDF](https://arxiv.org/pdf/2503.21766)  

**Abstract**: Establishing character shape correspondence is a critical and fundamental task in computer vision and graphics, with diverse applications including re-topology, attribute transfer, and shape interpolation. Current dominant functional map methods, while effective in controlled scenarios, struggle in real situations with more complex challenges such as non-isometric shape discrepancies. In response, we revisit registration-for-correspondence methods and tap their potential for more stable shape correspondence estimation. To overcome their common issues including unstable deformations and the necessity for careful pre-alignment or high-quality initial 3D correspondences, we introduce Stable-SCore: A Stable Registration-based Framework for 3D Shape Correspondence. We first re-purpose a foundation model for 2D character correspondence that ensures reliable and stable 2D mappings. Crucially, we propose a novel Semantic Flow Guided Registration approach that leverages 2D correspondence to guide mesh deformations. Our framework significantly surpasses existing methods in challenging scenarios, and brings possibilities for a wide array of real applications, as demonstrated in our results. 

---
# Uni4D: Unifying Visual Foundation Models for 4D Modeling from a Single Video 

**Authors**: David Yifan Yao, Albert J. Zhai, Shenlong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.21761)  

**Abstract**: This paper presents a unified approach to understanding dynamic scenes from casual videos. Large pretrained vision foundation models, such as vision-language, video depth prediction, motion tracking, and segmentation models, offer promising capabilities. However, training a single model for comprehensive 4D understanding remains challenging. We introduce Uni4D, a multi-stage optimization framework that harnesses multiple pretrained models to advance dynamic 3D modeling, including static/dynamic reconstruction, camera pose estimation, and dense 3D motion tracking. Our results show state-of-the-art performance in dynamic 4D modeling with superior visual quality. Notably, Uni4D requires no retraining or fine-tuning, highlighting the effectiveness of repurposing visual foundation models for 4D understanding. 

---
# Fwd2Bot: LVLM Visual Token Compression with Double Forward Bottleneck 

**Authors**: Adrian Bulat, Yassine Ouali, Georgios Tzimiropoulos  

**Link**: [PDF](https://arxiv.org/pdf/2503.21757)  

**Abstract**: In this work, we aim to compress the vision tokens of a Large Vision Language Model (LVLM) into a representation that is simultaneously suitable for (a) generative and (b) discriminative tasks, (c) is nearly lossless, and (d) is storage-efficient. We propose a novel compression approach, called Fwd2Bot, that uses the LVLM itself to compress the visual information in a task-agnostic manner. At the core of Fwd2bot there exists a "double-forward pass" training strategy, whereby, during the first forward pass, the LLM (of the LVLM) creates a bottleneck by condensing the visual information into a small number of summary tokens. Then, using the same LLM, the second forward pass processes the language instruction(s) alongside the summary tokens, used as a direct replacement for the image ones. The training signal is provided by two losses: an autoregressive one applied after the second pass that provides a direct optimization objective for compression, and a contrastive loss, applied after the first pass, that further boosts the representation strength, especially for discriminative tasks. The training is further enhanced by stage-specific adapters. We accompany the proposed method by an in-depth ablation study. Overall, Fwd2Bot results in highly-informative compressed representations suitable for both generative and discriminative tasks. For generative tasks, we offer a 2x higher compression rate without compromising the generative capabilities, setting a new state-of-the-art result. For discriminative tasks, we set a new state-of-the-art on image retrieval and compositionality. 

---
# CTRL-O: Language-Controllable Object-Centric Visual Representation Learning 

**Authors**: Aniket Didolkar, Andrii Zadaianchuk, Rabiul Awal, Maximilian Seitzer, Efstratios Gavves, Aishwarya Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2503.21747)  

**Abstract**: Object-centric representation learning aims to decompose visual scenes into fixed-size vectors called "slots" or "object files", where each slot captures a distinct object. Current state-of-the-art object-centric models have shown remarkable success in object discovery in diverse domains, including complex real-world scenes. However, these models suffer from a key limitation: they lack controllability. Specifically, current object-centric models learn representations based on their preconceived understanding of objects, without allowing user input to guide which objects are represented. Introducing controllability into object-centric models could unlock a range of useful capabilities, such as the ability to extract instance-specific representations from a scene. In this work, we propose a novel approach for user-directed control over slot representations by conditioning slots on language descriptions. The proposed ConTRoLlable Object-centric representation learning approach, which we term CTRL-O, achieves targeted object-language binding in complex real-world scenes without requiring mask supervision. Next, we apply these controllable slot representations on two downstream vision language tasks: text-to-image generation and visual question answering. The proposed approach enables instance-specific text-to-image generation and also achieves strong performance on visual question answering. 

---
# GateLens: A Reasoning-Enhanced LLM Agent for Automotive Software Release Analytics 

**Authors**: Arsham Gholamzadeh Khoee, Shuai Wang, Yinan Yu, Robert Feldt, Dhasarathy Parthasarathy  

**Link**: [PDF](https://arxiv.org/pdf/2503.21735)  

**Abstract**: Ensuring the reliability and effectiveness of software release decisions is critical, particularly in safety-critical domains like automotive systems. Precise analysis of release validation data, often presented in tabular form, plays a pivotal role in this process. However, traditional methods that rely on manual analysis of extensive test datasets and validation metrics are prone to delays and high costs. Large Language Models (LLMs) offer a promising alternative but face challenges in analytical reasoning, contextual understanding, handling out-of-scope queries, and processing structured test data consistently; limitations that hinder their direct application in safety-critical scenarios. This paper introduces GateLens, an LLM-based tool for analyzing tabular data in the automotive domain. GateLens translates natural language queries into Relational Algebra (RA) expressions and then generates optimized Python code. It outperforms the baseline system on benchmarking datasets, achieving higher F1 scores and handling complex and ambiguous queries with greater robustness. Ablation studies confirm the critical role of the RA module, with performance dropping sharply when omitted. Industrial evaluations reveal that GateLens reduces analysis time by over 80% while maintaining high accuracy and reliability. As demonstrated by presented results, GateLens achieved high performance without relying on few-shot examples, showcasing strong generalization across various query types from diverse company roles. Insights from deploying GateLens with a partner automotive company offer practical guidance for integrating AI into critical workflows such as release validation. Results show that by automating test result analysis, GateLens enables faster, more informed, and dependable release decisions, and can thus advance software scalability and reliability in automotive systems. 

---
# ReaRAG: Knowledge-guided Reasoning Enhances Factuality of Large Reasoning Models with Iterative Retrieval Augmented Generation 

**Authors**: Zhicheng Lee, Shulin Cao, Jinxin Liu, Jiajie Zhang, Weichuan Liu, Xiaoyin Che, Lei Hou, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.21729)  

**Abstract**: Large Reasoning Models (LRMs) exhibit remarkable reasoning abilities but rely primarily on parametric knowledge, limiting factual accuracy. While recent works equip reinforcement learning (RL)-based LRMs with retrieval capabilities, they suffer from overthinking and lack robustness in reasoning, reducing their effectiveness in question answering (QA) tasks. To address this, we propose ReaRAG, a factuality-enhanced reasoning model that explores diverse queries without excessive iterations. Our solution includes a novel data construction framework with an upper bound on the reasoning chain length. Specifically, we first leverage an LRM to generate deliberate thinking, then select an action from a predefined action space (Search and Finish). For Search action, a query is executed against the RAG engine, where the result is returned as observation to guide reasoning steps later. This process iterates until a Finish action is chosen. Benefiting from ReaRAG's strong reasoning capabilities, our approach outperforms existing baselines on multi-hop QA. Further analysis highlights its strong reflective ability to recognize errors and refine its reasoning trajectory. Our study enhances LRMs' factuality while effectively integrating robust reasoning for Retrieval-Augmented Generation (RAG). 

---
# Collab: Controlled Decoding using Mixture of Agents for LLM Alignment 

**Authors**: Souradip Chakraborty, Sujay Bhatt, Udari Madhushani Sehwag, Soumya Suvra Ghosal, Jiahao Qiu, Mengdi Wang, Dinesh Manocha, Furong Huang, Alec Koppel, Sumitra Ganesh  

**Link**: [PDF](https://arxiv.org/pdf/2503.21720)  

**Abstract**: Alignment of Large Language models (LLMs) is crucial for safe and trustworthy deployment in applications. Reinforcement learning from human feedback (RLHF) has emerged as an effective technique to align LLMs to human preferences and broader utilities, but it requires updating billions of model parameters, which is computationally expensive. Controlled Decoding, by contrast, provides a mechanism for aligning a model at inference time without retraining. However, single-agent decoding approaches often struggle to adapt to diverse tasks due to the complexity and variability inherent in these tasks. To strengthen the test-time performance w.r.t the target task, we propose a mixture of agent-based decoding strategies leveraging the existing off-the-shelf aligned LLM policies. Treating each prior policy as an agent in the spirit of mixture of agent collaboration, we develop a decoding method that allows for inference-time alignment through a token-level selection strategy among multiple agents. For each token, the most suitable LLM is dynamically chosen from a pool of models based on a long-term utility metric. This policy-switching mechanism ensures optimal model selection at each step, enabling efficient collaboration and alignment among LLMs during decoding. Theoretical analysis of our proposed algorithm establishes optimal performance with respect to the target task represented via a target reward for the given off-the-shelf models. We conduct comprehensive empirical evaluations with open-source aligned models on diverse tasks and preferences, which demonstrates the merits of this approach over single-agent decoding baselines. Notably, Collab surpasses the current SoTA decoding strategy, achieving an improvement of up to 1.56x in average reward and 71.89% in GPT-4 based win-tie rate. 

---
# Outlier dimensions favor frequent tokens in language model 

**Authors**: Iuri Macocco, Nora Graichen, Gemma Boleda, Marco Baroni  

**Link**: [PDF](https://arxiv.org/pdf/2503.21718)  

**Abstract**: We study last-layer outlier dimensions, this http URL that display extreme activations for the majority of inputs. We show that outlier dimensions arise in many different modern language models, and trace their function back to the heuristic of constantly predicting frequent words. We further show how a model can block this heuristic when it is not contextually appropriate, by assigning a counterbalancing weight mass to the remaining dimensions, and we investigate which model parameters boost outlier dimensions and when they arise during training. We conclude that outlier dimensions are a specialized mechanism discovered by many distinct models to implement a useful token prediction heuristic. 

---
# Elementwise Layer Normalization 

**Authors**: Felix Stollenwerk  

**Link**: [PDF](https://arxiv.org/pdf/2503.21708)  

**Abstract**: A recent paper proposed Dynamic Tanh (DyT) as a drop-in replacement for Layer Normalization. Although the method is empirically well-motivated and appealing from a practical point of view, it lacks a theoretical foundation. In this work, we derive DyT mathematically and show that a well-defined approximation is needed to do so. By dropping said approximation, an alternative element-wise transformation is obtained, which we call Elementwise Layer Normalization (ELN). We demonstrate that ELN resembles Layer Normalization more accurately than DyT does. 

---
# MAVERIX: Multimodal Audio-Visual Evaluation Reasoning IndeX 

**Authors**: Liuyue Xie, George Z. Wei, Avik Kuthiala, Ce Zheng, Ananya Bal, Mosam Dabhi, Liting Wen, Taru Rustagi, Ethan Lai, Sushil Khyalia, Rohan Choudhury, Morteza Ziyadi, Xu Zhang, Hao Yang, László A. Jeni  

**Link**: [PDF](https://arxiv.org/pdf/2503.21699)  

**Abstract**: Frontier models have either been language-only or have primarily focused on vision and language modalities. Although recent advancements in models with vision and audio understanding capabilities have shown substantial progress, the field lacks a standardized evaluation framework for thoroughly assessing their cross-modality perception performance. We introduce MAVERIX~(Multimodal Audio-Visual Evaluation Reasoning IndeX), a novel benchmark with 700 videos and 2,556 questions explicitly designed to evaluate multimodal models through tasks that necessitate close integration of video and audio information. MAVERIX uniquely provides models with audiovisual tasks, closely mimicking the multimodal perceptual experiences available to humans during inference and decision-making processes. To our knowledge, MAVERIX is the first benchmark aimed explicitly at assessing comprehensive audiovisual integration. Experiments with state-of-the-art models, including Gemini 1.5 Pro and o1, show performance approaching human levels (around 70% accuracy), while human experts reach near-ceiling performance (95.1%). With standardized evaluation protocols, a rigorously annotated pipeline, and a public toolkit, MAVERIX establishes a challenging testbed for advancing audiovisual multimodal intelligence. 

---
# AMA-SAM: Adversarial Multi-Domain Alignment of Segment Anything Model for High-Fidelity Histology Nuclei Segmentation 

**Authors**: Jiahe Qian, Yaoyu Fang, Jinkui Hao, Bo Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.21695)  

**Abstract**: Accurate segmentation of cell nuclei in histopathology images is essential for numerous biomedical research and clinical applications. However, existing cell nucleus segmentation methods only consider a single dataset (i.e., primary domain), while neglecting to leverage supplementary data from diverse sources (i.e., auxiliary domains) to reduce overfitting and enhance the performance. Although incorporating multiple datasets could alleviate overfitting, it often exacerbates performance drops caused by domain shifts. In this work, we introduce Adversarial Multi-domain Alignment of Segment Anything Model (AMA-SAM) that extends the Segment Anything Model (SAM) to overcome these obstacles through two key innovations. First, we propose a Conditional Gradient Reversal Layer (CGRL), a multi-domain alignment module that harmonizes features from diverse domains to promote domain-invariant representation learning while preserving crucial discriminative features for the primary dataset. Second, we address SAM's inherent low-resolution output by designing a High-Resolution Decoder (HR-Decoder), which directly produces fine-grained segmentation maps in order to capture intricate nuclei boundaries in high-resolution histology images. To the best of our knowledge, this is the first attempt to adapt SAM for multi-dataset learning with application to histology nuclei segmentation. We validate our method on several publicly available datasets, demonstrating consistent and significant improvements over state-of-the-art approaches. 

---
# Progressive Rendering Distillation: Adapting Stable Diffusion for Instant Text-to-Mesh Generation without 3D Data 

**Authors**: Zhiyuan Ma, Xinyue Liang, Rongyuan Wu, Xiangyu Zhu, Zhen Lei, Lei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.21694)  

**Abstract**: It is highly desirable to obtain a model that can generate high-quality 3D meshes from text prompts in just seconds. While recent attempts have adapted pre-trained text-to-image diffusion models, such as Stable Diffusion (SD), into generators of 3D representations (e.g., Triplane), they often suffer from poor quality due to the lack of sufficient high-quality 3D training data. Aiming at overcoming the data shortage, we propose a novel training scheme, termed as Progressive Rendering Distillation (PRD), eliminating the need for 3D ground-truths by distilling multi-view diffusion models and adapting SD into a native 3D generator. In each iteration of training, PRD uses the U-Net to progressively denoise the latent from random noise for a few steps, and in each step it decodes the denoised latent into 3D output. Multi-view diffusion models, including MVDream and RichDreamer, are used in joint with SD to distill text-consistent textures and geometries into the 3D outputs through score distillation. Since PRD supports training without 3D ground-truths, we can easily scale up the training data and improve generation quality for challenging text prompts with creative concepts. Meanwhile, PRD can accelerate the inference speed of the generation model in just a few steps. With PRD, we train a Triplane generator, namely TriplaneTurbo, which adds only $2.5\%$ trainable parameters to adapt SD for Triplane generation. TriplaneTurbo outperforms previous text-to-3D generators in both efficiency and quality. Specifically, it can produce high-quality 3D meshes in 1.2 seconds and generalize well for challenging text input. The code is available at this https URL. 

---
# Intelligent IoT Attack Detection Design via ODLLM with Feature Ranking-based Knowledge Base 

**Authors**: Satvik Verma, Qun Wang, E. Wes Bethel  

**Link**: [PDF](https://arxiv.org/pdf/2503.21674)  

**Abstract**: The widespread adoption of Internet of Things (IoT) devices has introduced significant cybersecurity challenges, particularly with the increasing frequency and sophistication of Distributed Denial of Service (DDoS) attacks. Traditional machine learning (ML) techniques often fall short in detecting such attacks due to the complexity of blended and evolving patterns. To address this, we propose a novel framework leveraging On-Device Large Language Models (ODLLMs) augmented with fine-tuning and knowledge base (KB) integration for intelligent IoT network attack detection. By implementing feature ranking techniques and constructing both long and short KBs tailored to model capacities, the proposed framework ensures efficient and accurate detection of DDoS attacks while overcoming computational and privacy limitations. Simulation results demonstrate that the optimized framework achieves superior accuracy across diverse attack types, especially when using compact models in edge computing environments. This work provides a scalable and secure solution for real-time IoT security, advancing the applicability of edge intelligence in cybersecurity. 

---
# COMI-LINGUA: Expert Annotated Large-Scale Dataset for Multitask NLP in Hindi-English Code-Mixing 

**Authors**: Rajvee Sheth, Himanshu Beniwal, Mayank Singh  

**Link**: [PDF](https://arxiv.org/pdf/2503.21670)  

**Abstract**: The rapid growth of digital communication has driven the widespread use of code-mixing, particularly Hindi-English, in multilingual communities. Existing datasets often focus on romanized text, have limited scope, or rely on synthetic data, which fails to capture realworld language nuances. Human annotations are crucial for assessing the naturalness and acceptability of code-mixed text. To address these challenges, We introduce COMI-LINGUA, the largest manually annotated dataset for code-mixed text, comprising 100,970 instances evaluated by three expert annotators in both Devanagari and Roman scripts. The dataset supports five fundamental NLP tasks: Language Identification, Matrix Language Identification, Part-of-Speech Tagging, Named Entity Recognition, and Translation. We evaluate LLMs on these tasks using COMILINGUA, revealing limitations in current multilingual modeling strategies and emphasizing the need for improved code-mixed text processing capabilities. COMI-LINGUA is publically availabe at: this https URL. 

---
# Model Assembly Learning with Heterogeneous Layer Weight Merging 

**Authors**: Yi-Kai Zhang, Jin Wang, Xu-Xiang Zhong, De-Chuan Zhan, Han-Jia Ye  

**Link**: [PDF](https://arxiv.org/pdf/2503.21657)  

**Abstract**: Model merging acquires general capabilities without extra data or training by combining multiple models' parameters. Previous approaches achieve linear mode connectivity by aligning parameters into the same loss basin using permutation invariance. In this paper, we introduce Model Assembly Learning (MAL), a novel paradigm for model merging that iteratively integrates parameters from diverse models in an open-ended model zoo to enhance the base model's capabilities. Unlike previous works that require identical architectures, MAL allows the merging of heterogeneous architectures and selective parameters across layers. Specifically, the base model can incorporate parameters from different layers of multiple pre-trained models. We systematically investigate the conditions and fundamental settings of heterogeneous parameter merging, addressing all possible mismatches in layer widths between the base and target models. Furthermore, we establish key laws and provide practical guidelines for effectively implementing MAL. 

---
# When Astronomy Meets AI: Manazel For Crescent Visibility Prediction in Morocco 

**Authors**: Yassir Lairgi  

**Link**: [PDF](https://arxiv.org/pdf/2503.21634)  

**Abstract**: The accurate determination of the beginning of each Hijri month is essential for religious, cultural, and administrative purposes. Manazel (The code and datasets are available at this https URL) addresses this challenge in Morocco by leveraging 13 years of crescent visibility data to refine the ODEH criterion, a widely used standard for lunar crescent visibility prediction. The study integrates two key features, the Arc of Vision (ARCV) and the total width of the crescent (W), to enhance the accuracy of lunar visibility assessments. A machine learning approach utilizing the Logistic Regression algorithm is employed to classify crescent visibility conditions, achieving a predictive accuracy of 98.83%. This data-driven methodology offers a robust and reliable framework for determining the start of the Hijri month, comparing different data classification tools, and improving the consistency of lunar calendar calculations in Morocco. The findings demonstrate the effectiveness of machine learning in astronomical applications and highlight the potential for further enhancements in the modeling of crescent visibility. 

---
# A Measure Based Generalizable Approach to Understandability 

**Authors**: Vikas Kushwaha, Sruti Srinivasa Ragavan, Subhajit Roy  

**Link**: [PDF](https://arxiv.org/pdf/2503.21615)  

**Abstract**: Successful agent-human partnerships require that any agent generated information is understandable to the human, and that the human can easily steer the agent towards a goal. Such effective communication requires the agent to develop a finer-level notion of what is understandable to the human. State-of-the-art agents, including LLMs, lack this detailed notion of understandability because they only capture average human sensibilities from the training data, and therefore afford limited steerability (e.g., requiring non-trivial prompt engineering).
In this paper, instead of only relying on data, we argue for developing generalizable, domain-agnostic measures of understandability that can be used as directives for these agents. Existing research on understandability measures is fragmented, we survey various such efforts across domains, and lay a cognitive-science-rooted groundwork for more coherent and domain-agnostic research investigations in future. 

---
# Prompt, Divide, and Conquer: Bypassing Large Language Model Safety Filters via Segmented and Distributed Prompt Processing 

**Authors**: Johan Wahréus, Ahmed Hussain, Panos Papadimitratos  

**Link**: [PDF](https://arxiv.org/pdf/2503.21598)  

**Abstract**: Large Language Models (LLMs) have transformed task automation and content generation across various domains while incorporating safety filters to prevent misuse. We introduce a novel jailbreaking framework that employs distributed prompt processing combined with iterative refinements to bypass these safety measures, particularly in generating malicious code. Our architecture consists of four key modules: prompt segmentation, parallel processing, response aggregation, and LLM-based jury evaluation. Tested on 500 malicious prompts across 10 cybersecurity categories, the framework achieves a 73.2% Success Rate (SR) in generating malicious code. Notably, our comparative analysis reveals that traditional single-LLM judge evaluation overestimates SRs (93.8%) compared to our LLM jury system (73.2%), with manual verification confirming that single-judge assessments often accept incomplete implementations. Moreover, we demonstrate that our distributed architecture improves SRs by 12% over the non-distributed approach in an ablation study, highlighting both the effectiveness of distributed prompt processing and the importance of robust evaluation methodologies in assessing jailbreak attempts. 

---
# Critical Iterative Denoising: A Discrete Generative Model Applied to Graphs 

**Authors**: Yoann Boget, Alexandros Kalousis  

**Link**: [PDF](https://arxiv.org/pdf/2503.21592)  

**Abstract**: Discrete Diffusion and Flow Matching models have significantly advanced generative modeling for discrete structures, including graphs. However, the time dependencies in the noising process of these models lead to error accumulation and propagation during the backward process. This issue, particularly pronounced in mask diffusion, is a known limitation in sequence modeling and, as we demonstrate, also impacts discrete diffusion models for graphs.
To address this problem, we propose a novel framework called Iterative Denoising, which simplifies discrete diffusion and circumvents the issue by assuming conditional independence across time. Additionally, we enhance our model by incorporating a Critic, which during generation selectively retains or corrupts elements in an instance based on their likelihood under the data distribution. Our empirical evaluations demonstrate that the proposed method significantly outperforms existing discrete diffusion baselines in graph generation tasks. 

---
# AlignDiff: Learning Physically-Grounded Camera Alignment via Diffusion 

**Authors**: Liuyue Xie, Jiancong Guo, Ozan Cakmakci, Andre Araujo, Laszlo A. Jeni, Zhiheng Jia  

**Link**: [PDF](https://arxiv.org/pdf/2503.21581)  

**Abstract**: Accurate camera calibration is a fundamental task for 3D perception, especially when dealing with real-world, in-the-wild environments where complex optical distortions are common. Existing methods often rely on pre-rectified images or calibration patterns, which limits their applicability and flexibility. In this work, we introduce a novel framework that addresses these challenges by jointly modeling camera intrinsic and extrinsic parameters using a generic ray camera model. Unlike previous approaches, AlignDiff shifts focus from semantic to geometric features, enabling more accurate modeling of local distortions. We propose AlignDiff, a diffusion model conditioned on geometric priors, enabling the simultaneous estimation of camera distortions and scene geometry. To enhance distortion prediction, we incorporate edge-aware attention, focusing the model on geometric features around image edges, rather than semantic content. Furthermore, to enhance generalizability to real-world captures, we incorporate a large database of ray-traced lenses containing over three thousand samples. This database characterizes the distortion inherent in a diverse variety of lens forms. Our experiments demonstrate that the proposed method significantly reduces the angular error of estimated ray bundles by ~8.2 degrees and overall calibration accuracy, outperforming existing approaches on challenging, real-world datasets. 

---
# Magnitude-Phase Dual-Path Speech Enhancement Network based on Self-Supervised Embedding and Perceptual Contrast Stretch Boosting 

**Authors**: Alimjan Mattursun, Liejun Wang, Yinfeng Yu, Chunyang Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.21571)  

**Abstract**: Speech self-supervised learning (SSL) has made great progress in various speech processing tasks, but there is still room for improvement in speech enhancement (SE). This paper presents BSP-MPNet, a dual-path framework that combines self-supervised features with magnitude-phase information for SE. The approach starts by applying the perceptual contrast stretching (PCS) algorithm to enhance the magnitude-phase spectrum. A magnitude-phase 2D coarse (MP-2DC) encoder then extracts coarse features from the enhanced spectrum. Next, a feature-separating self-supervised learning (FS-SSL) model generates self-supervised embeddings for the magnitude and phase components separately. These embeddings are fused to create cross-domain feature representations. Finally, two parallel RNN-enhanced multi-attention (REMA) mask decoders refine the features, apply them to the mask, and reconstruct the speech signal. We evaluate BSP-MPNet on the VoiceBank+DEMAND and WHAMR! datasets. Experimental results show that BSP-MPNet outperforms existing methods under various noise conditions, providing new directions for self-supervised speech enhancement research. The implementation of the BSP-MPNet code is available online\footnote[2]{this https URL. \label{s1}} 

---
# A Local Perspective-based Model for Overlapping Community Detection 

**Authors**: Gaofeng Zhou, Rui-Feng Wang, Kangning Cui  

**Link**: [PDF](https://arxiv.org/pdf/2503.21558)  

**Abstract**: Community detection, which identifies densely connected node clusters with sparse between-group links, is vital for analyzing network structure and function in real-world systems. Most existing community detection methods based on GCNs primarily focus on node-level information while overlooking community-level features, leading to performance limitations on large-scale networks. To address this issue, we propose LQ-GCN, an overlapping community detection model from a local community perspective. LQ-GCN employs a Bernoulli-Poisson model to construct a community affiliation matrix and form an end-to-end detection framework. By adopting local modularity as the objective function, the model incorporates local community information to enhance the quality and accuracy of clustering results. Additionally, the conventional GCNs architecture is optimized to improve the model capability in identifying overlapping communities in large-scale networks. Experimental results demonstrate that LQ-GCN achieves up to a 33% improvement in Normalized Mutual Information (NMI) and a 26.3% improvement in Recall compared to baseline models across multiple real-world benchmark datasets. 

---
# SWI: Speaking with Intent in Large Language Models 

**Authors**: Yuwei Yin, EunJeong Hwang, Giuseppe Carenini  

**Link**: [PDF](https://arxiv.org/pdf/2503.21544)  

**Abstract**: Intent, typically clearly formulated and planned, functions as a cognitive framework for reasoning and problem-solving. This paper introduces the concept of Speaking with Intent (SWI) in large language models (LLMs), where the explicitly generated intent encapsulates the model's underlying intention and provides high-level planning to guide subsequent analysis and communication. By emulating deliberate and purposeful thoughts in the human mind, SWI is hypothesized to enhance the reasoning capabilities and generation quality of LLMs. Extensive experiments on mathematical reasoning benchmarks consistently demonstrate the superiority of Speaking with Intent over Baseline (i.e., generation without explicit intent). Moreover, SWI outperforms answer-trigger prompting methods Chain-of-Thought and Plan-and-Solve and maintains competitive performance with the strong method ARR (Analyzing, Retrieving, and Reasoning). Additionally, the effectiveness and generalizability of SWI are solidified on reasoning-intensive question answering (QA) and text summarization benchmarks, where SWI brings consistent improvement to the Baseline generation. In text summarization, SWI-generated summaries exhibit greater accuracy, conciseness, and factual correctness, with fewer hallucinations. Furthermore, human evaluations verify the coherence, effectiveness, and interpretability of the intent produced by SWI. This proof-of-concept study creates a novel avenue for enhancing LLMs' reasoning abilities with cognitive notions. 

---
# LOCATEdit: Graph Laplacian Optimized Cross Attention for Localized Text-Guided Image Editing 

**Authors**: Achint Soni, Meet Soni, Sirisha Rambhatla  

**Link**: [PDF](https://arxiv.org/pdf/2503.21541)  

**Abstract**: Text-guided image editing aims to modify specific regions of an image according to natural language instructions while maintaining the general structure and the background fidelity. Existing methods utilize masks derived from cross-attention maps generated from diffusion models to identify the target regions for modification. However, since cross-attention mechanisms focus on semantic relevance, they struggle to maintain the image integrity. As a result, these methods often lack spatial consistency, leading to editing artifacts and distortions. In this work, we address these limitations and introduce LOCATEdit, which enhances cross-attention maps through a graph-based approach utilizing self-attention-derived patch relationships to maintain smooth, coherent attention across image regions, ensuring that alterations are limited to the designated items while retaining the surrounding structure. \method consistently and substantially outperforms existing baselines on PIE-Bench, demonstrating its state-of-the-art performance and effectiveness on various editing tasks. Code can be found on this https URL 

---
# Low-Resource Transliteration for Roman-Urdu and Urdu Using Transformer-Based Models 

**Authors**: Umer Butt, Stalin Veranasi, Günter Neumann  

**Link**: [PDF](https://arxiv.org/pdf/2503.21530)  

**Abstract**: As the Information Retrieval (IR) field increasingly recognizes the importance of inclusivity, addressing the needs of low-resource languages remains a significant challenge. Transliteration between Urdu and its Romanized form, Roman Urdu, remains underexplored despite the widespread use of both scripts in South Asia. Prior work using RNNs on the Roman-Urdu-Parl dataset showed promising results but suffered from poor domain adaptability and limited evaluation. We propose a transformer-based approach using the m2m100 multilingual translation model, enhanced with masked language modeling (MLM) pretraining and fine-tuning on both Roman-Urdu-Parl and the domain-diverse Dakshina dataset. To address previous evaluation flaws, we introduce rigorous dataset splits and assess performance using BLEU, character-level BLEU, and CHRF. Our model achieves strong transliteration performance, with Char-BLEU scores of 96.37 for Urdu->Roman-Urdu and 97.44 for Roman-Urdu->Urdu. These results outperform both RNN baselines and GPT-4o Mini and demonstrate the effectiveness of multilingual transfer learning for low-resource transliteration tasks. 

---
# MONO2REST: Identifying and Exposing Microservices: a Reusable RESTification Approach 

**Authors**: Matthéo Lecrivain, Hanifa Barry, Dalila Tamzalit, Houari Sahraoui  

**Link**: [PDF](https://arxiv.org/pdf/2503.21522)  

**Abstract**: The microservices architectural style has become the de facto standard for large-scale cloud applications, offering numerous benefits in scalability, maintainability, and deployment flexibility. Many organizations are pursuing the migration of legacy monolithic systems to a microservices architecture. However, this process is challenging, risky, time-intensive, and prone-to-failure while several organizations lack necessary financial resources, time, or expertise to set up this migration process. So, rather than trying to migrate a legacy system where migration is risky or not feasible, we suggest exposing it as a microservice application without without having to migrate it. In this paper, we present a reusable, automated, two-phase approach that combines evolutionary algorithms with machine learning techniques. In the first phase, we identify microservices at the method level using a multi-objective genetic algorithm that considers both structural and semantic dependencies between methods. In the second phase, we generate REST APIs for each identified microservice using a classification algorithm to assign HTTP methods and endpoints. We evaluated our approach with a case study on the Spring PetClinic application, which has both monolithic and microservices implementations that serve as ground truth for comparison. Results demonstrate that our approach successfully aligns identified microservices with those in the reference microservices implementation, highlighting its effectiveness in service identification and API generation. 

---
# Quantitative Evaluation of Quantum/Classical Neural Network Using a Game Solver Metric 

**Authors**: Suzukaze Kamei, Hideaki Kawaguchi, Shin Nishio, Tatakahiko Satoh  

**Link**: [PDF](https://arxiv.org/pdf/2503.21514)  

**Abstract**: To evaluate the performance of quantum computing systems relative to classical counterparts and explore the potential for quantum advantage, we propose a game-solving benchmark based on Elo ratings in the game of tic-tac-toe. We compare classical convolutional neural networks (CNNs), quantum convolutional neural networks (QCNNs), and hybrid classical-quantum models by assessing their performance against a random-move agent in automated matches. Additionally, we implement a QCNN integrated with quantum communication and evaluate its performance to quantify the overhead introduced by noisy quantum channels. Our results show that the classical-quantum hybrid model achieves Elo ratings comparable to those of classical CNNs, while the standalone QCNN underperforms under current hardware constraints. The communication overhead was found to be modest. These findings demonstrate the viability of using game-based benchmarks for evaluating quantum computing systems and suggest that quantum communication can be incorporated with limited impact on performance, providing a foundation for future hybrid quantum applications. 

---
# Keyword-Oriented Multimodal Modeling for Euphemism Identification 

**Authors**: Yuxue Hu, Junsong Li, Meixuan Chen, Dongyu Su, Tongguan Wang, Ying Sha  

**Link**: [PDF](https://arxiv.org/pdf/2503.21504)  

**Abstract**: Euphemism identification deciphers the true meaning of euphemisms, such as linking "weed" (euphemism) to "marijuana" (target keyword) in illicit texts, aiding content moderation and combating underground markets. While existing methods are primarily text-based, the rise of social media highlights the need for multimodal analysis, incorporating text, images, and audio. However, the lack of multimodal datasets for euphemisms limits further research. To address this, we regard euphemisms and their corresponding target keywords as keywords and first introduce a keyword-oriented multimodal corpus of euphemisms (KOM-Euph), involving three datasets (Drug, Weapon, and Sexuality), including text, images, and speech. We further propose a keyword-oriented multimodal euphemism identification method (KOM-EI), which uses cross-modal feature alignment and dynamic fusion modules to explicitly utilize the visual and audio features of the keywords for efficient euphemism identification. Extensive experiments demonstrate that KOM-EI outperforms state-of-the-art models and large language models, and show the importance of our multimodal datasets. 

---
# Adaptive Resampling with Bootstrap for Noisy Multi-Objective Optimization Problems 

**Authors**: Timo Budszuhn, Mark Joachim Krallmann, Daniel Horn  

**Link**: [PDF](https://arxiv.org/pdf/2503.21495)  

**Abstract**: The challenge of noisy multi-objective optimization lies in the constant trade-off between exploring new decision points and improving the precision of known points through resampling. This decision should take into account both the variability of the objective functions and the current estimate of a point in relation to the Pareto front. Since the amount and distribution of noise are generally unknown, it is desirable for a decision function to be highly adaptive to the properties of the optimization problem. This paper presents a resampling decision function that incorporates the stochastic nature of the optimization problem by using bootstrapping and the probability of dominance. The distribution-free estimation of the probability of dominance is achieved using bootstrap estimates of the means. To make the procedure applicable even with very few observations, we transfer the distribution observed at other decision points. The efficiency of this resampling approach is demonstrated by applying it in the NSGA-II algorithm with a sequential resampling procedure under multiple noise variations. 

---
# Retinal Fundus Multi-Disease Image Classification using Hybrid CNN-Transformer-Ensemble Architectures 

**Authors**: Deependra Singh, Saksham Agarwal, Subhankar Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2503.21465)  

**Abstract**: Our research is motivated by the urgent global issue of a large population affected by retinal diseases, which are evenly distributed but underserved by specialized medical expertise, particularly in non-urban areas. Our primary objective is to bridge this healthcare gap by developing a comprehensive diagnostic system capable of accurately predicting retinal diseases solely from fundus images. However, we faced significant challenges due to limited, diverse datasets and imbalanced class distributions. To overcome these issues, we have devised innovative strategies. Our research introduces novel approaches, utilizing hybrid models combining deeper Convolutional Neural Networks (CNNs), Transformer encoders, and ensemble architectures sequentially and in parallel to classify retinal fundus images into 20 disease labels. Our overarching goal is to assess these advanced models' potential in practical applications, with a strong focus on enhancing retinal disease diagnosis accuracy across a broader spectrum of conditions. Importantly, our efforts have surpassed baseline model results, with the C-Tran ensemble model emerging as the leader, achieving a remarkable model score of 0.9166, surpassing the baseline score of 0.9. Additionally, experiments with the IEViT model showcased equally promising outcomes with improved computational efficiency. We've also demonstrated the effectiveness of dynamic patch extraction and the integration of domain knowledge in computer vision tasks. In summary, our research strives to contribute significantly to retinal disease diagnosis, addressing the critical need for accessible healthcare solutions in underserved regions while aiming for comprehensive and accurate disease prediction. 

---
# Harnessing Chain-of-Thought Metadata for Task Routing and Adversarial Prompt Detection 

**Authors**: Ryan Marinelli, Josef Pichlmeier, Tamas Bisztray  

**Link**: [PDF](https://arxiv.org/pdf/2503.21464)  

**Abstract**: In this work, we propose a metric called Number of Thoughts (NofT) to determine the difficulty of tasks pre-prompting and support Large Language Models (LLMs) in production contexts. By setting thresholds based on the number of thoughts, this metric can discern the difficulty of prompts and support more effective prompt routing. A 2% decrease in latency is achieved when routing prompts from the MathInstruct dataset through quantized, distilled versions of Deepseek with 1.7 billion, 7 billion, and 14 billion parameters. Moreover, this metric can be used to detect adversarial prompts used in prompt injection attacks with high efficacy. The Number of Thoughts can inform a classifier that achieves 95% accuracy in adversarial prompt detection. Our experiments ad datasets used are available on our GitHub page: this https URL. 

---
# Unveiling Latent Information in Transaction Hashes: Hypergraph Learning for Ethereum Ponzi Scheme Detection 

**Authors**: Junhao Wu, Yixin Yang, Chengxiang Jin, Silu Mu, Xiaolei Qian, Jiajun Zhou, Shanqing Yu, Qi Xuan  

**Link**: [PDF](https://arxiv.org/pdf/2503.21463)  

**Abstract**: With the widespread adoption of Ethereum, financial frauds such as Ponzi schemes have become increasingly rampant in the blockchain ecosystem, posing significant threats to the security of account assets. Existing Ethereum fraud detection methods typically model account transactions as graphs, but this approach primarily focuses on binary transactional relationships between accounts, failing to adequately capture the complex multi-party interaction patterns inherent in Ethereum. To address this, we propose a hypergraph modeling method for the Ponzi scheme detection method in Ethereum, called HyperDet. Specifically, we treat transaction hashes as hyperedges that connect all the relevant accounts involved in a transaction. Additionally, we design a two-step hypergraph sampling strategy to significantly reduce computational complexity. Furthermore, we introduce a dual-channel detection module, including the hypergraph detection channel and the hyper-homo graph detection channel, to be compatible with existing detection methods. Experimental results show that, compared to traditional homogeneous graph-based methods, the hyper-homo graph detection channel achieves significant performance improvements, demonstrating the superiority of hypergraph in Ponzi scheme detection. This research offers innovations for modeling complex relationships in blockchain data. 

---
# An evaluation of LLMs and Google Translate for translation of selected Indian languages via sentiment and semantic analyses 

**Authors**: Rohitash Chandra, Aryan Chaudhary, Yeshwanth Rayavarapu  

**Link**: [PDF](https://arxiv.org/pdf/2503.21393)  

**Abstract**: Large Language models (LLMs) have been prominent for language translation, including low-resource languages. There has been limited study about the assessment of the quality of translations generated by LLMs, including Gemini, GPT and Google Translate. In this study, we address this limitation by using semantic and sentiment analysis of selected LLMs for Indian languages, including Sanskrit, Telugu and Hindi. We select prominent texts that have been well translated by experts and use LLMs to generate their translations to English, and then we provide a comparison with selected expert (human) translations. Our findings suggest that while LLMs have made significant progress in translation accuracy, challenges remain in preserving sentiment and semantic integrity, especially in figurative and philosophical contexts. The sentiment analysis revealed that GPT-4o and GPT-3.5 are better at preserving the sentiments for the Bhagavad Gita (Sanskrit-English) translations when compared to Google Translate. We observed a similar trend for the case of Tamas (Hindi-English) and Maha P (Telugu-English) translations. GPT-4o performs similarly to GPT-3.5 in the translation in terms of sentiments for the three languages. We found that LLMs are generally better at translation for capturing sentiments when compared to Google Translate. 

---
# Investigating the Duality of Interpretability and Explainability in Machine Learning 

**Authors**: Moncef Garouani, Josiane Mothe, Ayah Barhrhouj, Julien Aligon  

**Link**: [PDF](https://arxiv.org/pdf/2503.21356)  

**Abstract**: The rapid evolution of machine learning (ML) has led to the widespread adoption of complex "black box" models, such as deep neural networks and ensemble methods. These models exhibit exceptional predictive performance, making them invaluable for critical decision-making across diverse domains within society. However, their inherently opaque nature raises concerns about transparency and interpretability, making them untrustworthy decision support systems. To alleviate such a barrier to high-stakes adoption, research community focus has been on developing methods to explain black box models as a means to address the challenges they pose. Efforts are focused on explaining these models instead of developing ones that are inherently interpretable. Designing inherently interpretable models from the outset, however, can pave the path towards responsible and beneficial applications in the field of ML. In this position paper, we clarify the chasm between explaining black boxes and adopting inherently interpretable models. We emphasize the imperative need for model interpretability and, following the purpose of attaining better (i.e., more effective or efficient w.r.t. predictive performance) and trustworthy predictors, provide an experimental evaluation of latest hybrid learning methods that integrates symbolic knowledge into neural network predictors. We demonstrate how interpretable hybrid models could potentially supplant black box ones in different domains. 

---
# Residual Learning Inspired Crossover Operator and Strategy Enhancements for Evolutionary Multitasking 

**Authors**: Ruilin Wang, Xiang Feng, Huiqun Yu, Edmund M-K Lai  

**Link**: [PDF](https://arxiv.org/pdf/2503.21347)  

**Abstract**: In evolutionary multitasking, strategies such as crossover operators and skill factor assignment are critical for effective knowledge transfer. Existing improvements to crossover operators primarily focus on low-dimensional variable combinations, such as arithmetic crossover or partially mapped crossover, which are insufficient for modeling complex high-dimensional this http URL, static or semi-dynamic crossover strategies fail to adapt to the dynamic dependencies among tasks. In addition, current Multifactorial Evolutionary Algorithm frameworks often rely on fixed skill factor assignment strategies, lacking flexibility. To address these limitations, this paper proposes the Multifactorial Evolutionary Algorithm-Residual Learning (MFEA-RL) method based on residual learning. The method employs a Very Deep Super-Resolution (VDSR) model to generate high-dimensional residual representations of individuals, enhancing the modeling of complex relationships within dimensions. A ResNet-based mechanism dynamically assigns skill factors to improve task adaptability, while a random mapping mechanism efficiently performs crossover operations and mitigates the risk of negative transfer. Theoretical analysis and experimental results show that MFEA-RL outperforms state-of-the-art multitasking algorithms. It excels in both convergence and adaptability on standard evolutionary multitasking benchmarks, including CEC2017-MTSO and WCCI2020-MTSO. Additionally, its effectiveness is validated through a real-world application scenario. 

---
# A 71.2-$μ$W Speech Recognition Accelerator with Recurrent Spiking Neural Network 

**Authors**: Chih-Chyau Yang, Tian-Sheuan Chang  

**Link**: [PDF](https://arxiv.org/pdf/2503.21337)  

**Abstract**: This paper introduces a 71.2-$\mu$W speech recognition accelerator designed for edge devices' real-time applications, emphasizing an ultra low power design. Achieved through algorithm and hardware co-optimizations, we propose a compact recurrent spiking neural network with two recurrent layers, one fully connected layer, and a low time step (1 or 2). The 2.79-MB model undergoes pruning and 4-bit fixed-point quantization, shrinking it by 96.42\% to 0.1 MB. On the hardware front, we take advantage of \textit{mixed-level pruning}, \textit{zero-skipping} and \textit{merged spike} techniques, reducing complexity by 90.49\% to 13.86 MMAC/S. The \textit{parallel time-step execution} addresses inter-time-step data dependencies and enables weight buffer power savings through weight sharing. Capitalizing on the sparse spike activity, an input broadcasting scheme eliminates zero computations, further saving power. Implemented on the TSMC 28-nm process, the design operates in real time at 100 kHz, consuming 71.2 $\mu$W, surpassing state-of-the-art designs. At 500 MHz, it has 28.41 TOPS/W and 1903.11 GOPS/mm$^2$ in energy and area efficiency, respectively. 

---
# A Low-Power Streaming Speech Enhancement Accelerator For Edge Devices 

**Authors**: Ci-Hao Wu, Tian-Sheuan Chang  

**Link**: [PDF](https://arxiv.org/pdf/2503.21335)  

**Abstract**: Transformer-based speech enhancement models yield impressive results. However, their heterogeneous and complex structure restricts model compression potential, resulting in greater complexity and reduced hardware efficiency. Additionally, these models are not tailored for streaming and low-power applications. Addressing these challenges, this paper proposes a low-power streaming speech enhancement accelerator through model and hardware optimization. The proposed high performance model is optimized for hardware execution with the co-design of model compression and target application, which reduces 93.9\% of model size by the proposed domain-aware and streaming-aware pruning techniques. The required latency is further reduced with batch normalization-based transformers. Additionally, we employed softmax-free attention, complemented by an extra batch normalization, facilitating simpler hardware design. The tailored hardware accommodates these diverse computing patterns by breaking them down into element-wise multiplication and accumulation (MAC). This is achieved through a 1-D processing array, utilizing configurable SRAM addressing, thereby minimizing hardware complexities and simplifying zero skipping. Using the TSMC 40nm CMOS process, the final implementation requires merely 207.8K gates and 53.75KB SRAM. It consumes only 8.08 mW for real-time inference at a 62.5MHz frequency. 

---
# ReFeed: Multi-dimensional Summarization Refinement with Reflective Reasoning on Feedback 

**Authors**: Taewon Yun, Jihwan Oh, Hyangsuk Min, Yuho Lee, Jihwan Bang, Jason Cai, Hwanjun Song  

**Link**: [PDF](https://arxiv.org/pdf/2503.21332)  

**Abstract**: Summarization refinement faces challenges when extending to multi-dimension. In this paper, we introduce ReFeed, a powerful summarization refinement pipeline that enhances multiple dimensions through reflective reasoning on feedback. To achieve this, we release SumFeed-CoT, a large-scale Long-CoT-based dataset optimized for training a lightweight model with reflective reasoning. Our experiments reveal how the number of dimensions, feedback exposure, and reasoning policy influence refinement performance, highlighting reflective reasoning and simultaneously addressing multiple feedback is crucial to mitigate trade-off between dimensions. Furthermore, ReFeed is robust to noisy feedback and feedback order. Lastly, our finding emphasizes that creating data with a proper goal and guideline constitutes a fundamental pillar of effective reasoning. The dataset and model will be released. 

---
# FineCIR: Explicit Parsing of Fine-Grained Modification Semantics for Composed Image Retrieval 

**Authors**: Zixu Li, Zhiheng Fu, Yupeng Hu, Zhiwei Chen, Haokun Wen, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2503.21309)  

**Abstract**: Composed Image Retrieval (CIR) facilitates image retrieval through a multimodal query consisting of a reference image and modification text. The reference image defines the retrieval context, while the modification text specifies desired alterations. However, existing CIR datasets predominantly employ coarse-grained modification text (CoarseMT), which inadequately captures fine-grained retrieval intents. This limitation introduces two key challenges: (1) ignoring detailed differences leads to imprecise positive samples, and (2) greater ambiguity arises when retrieving visually similar images. These issues degrade retrieval accuracy, necessitating manual result filtering or repeated queries. To address these limitations, we develop a robust fine-grained CIR data annotation pipeline that minimizes imprecise positive samples and enhances CIR systems' ability to discern modification intents accurately. Using this pipeline, we refine the FashionIQ and CIRR datasets to create two fine-grained CIR datasets: Fine-FashionIQ and Fine-CIRR. Furthermore, we introduce FineCIR, the first CIR framework explicitly designed to parse the modification text. FineCIR effectively captures fine-grained modification semantics and aligns them with ambiguous visual entities, enhancing retrieval precision. Extensive experiments demonstrate that FineCIR consistently outperforms state-of-the-art CIR baselines on both fine-grained and traditional CIR benchmark datasets. Our FineCIR code and fine-grained CIR datasets are available at this https URL. 

---
# InternVL-X: Advancing and Accelerating InternVL Series with Efficient Visual Token Compression 

**Authors**: Dongchen Lu, Yuyao Sun, Zilu Zhang, Leping Huang, Jianliang Zeng, Mao Shu, Huo Cao  

**Link**: [PDF](https://arxiv.org/pdf/2503.21307)  

**Abstract**: Most multimodal large language models (MLLMs) treat visual tokens as "a sequence of text", integrating them with text tokens into a large language model (LLM). However, a great quantity of visual tokens significantly increases the demand for computational resources and time. In this paper, we propose InternVL-X, which outperforms the InternVL model in both performance and efficiency by incorporating three visual token compression methods. First, we propose a novel vision-language projector, PVTC. This component integrates adjacent visual embeddings to form a local query and utilizes the transformed CLS token as a global query, then performs point-to-region cross-attention through these local and global queries to more effectively convert visual features. Second, we present a layer-wise visual token compression module, LVTC, which compresses tokens in the LLM shallow layers and then expands them through upsampling and residual connections in the deeper layers. This significantly enhances the model computational efficiency. Futhermore, we propose an efficient high resolution slicing method, RVTC, which dynamically adjusts the number of visual tokens based on image area or length filtering. RVTC greatly enhances training efficiency with only a slight reduction in performance. By utilizing 20% or fewer visual tokens, InternVL-X achieves state-of-the-art performance on 7 public MLLM benchmarks, and improves the average metric by 2.34% across 12 tasks. 

---
# DeBackdoor: A Deductive Framework for Detecting Backdoor Attacks on Deep Models with Limited Data 

**Authors**: Dorde Popovic, Amin Sadeghi, Ting Yu, Sanjay Chawla, Issa Khalil  

**Link**: [PDF](https://arxiv.org/pdf/2503.21305)  

**Abstract**: Backdoor attacks are among the most effective, practical, and stealthy attacks in deep learning. In this paper, we consider a practical scenario where a developer obtains a deep model from a third party and uses it as part of a safety-critical system. The developer wants to inspect the model for potential backdoors prior to system deployment. We find that most existing detection techniques make assumptions that are not applicable to this scenario. In this paper, we present a novel framework for detecting backdoors under realistic restrictions. We generate candidate triggers by deductively searching over the space of possible triggers. We construct and optimize a smoothed version of Attack Success Rate as our search objective. Starting from a broad class of template attacks and just using the forward pass of a deep model, we reverse engineer the backdoor attack. We conduct extensive evaluation on a wide range of attacks, models, and datasets, with our technique performing almost perfectly across these settings. 

---
# Multi-Scale Invertible Neural Network for Wide-Range Variable-Rate Learned Image Compression 

**Authors**: Hanyue Tu, Siqi Wu, Li Li, Wengang Zhou, Houqiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.21284)  

**Abstract**: Autoencoder-based structures have dominated recent learned image compression methods. However, the inherent information loss associated with autoencoders limits their rate-distortion performance at high bit rates and restricts their flexibility of rate adaptation. In this paper, we present a variable-rate image compression model based on invertible transform to overcome these limitations. Specifically, we design a lightweight multi-scale invertible neural network, which bijectively maps the input image into multi-scale latent representations. To improve the compression efficiency, a multi-scale spatial-channel context model with extended gain units is devised to estimate the entropy of the latent representation from high to low levels. Experimental results demonstrate that the proposed method achieves state-of-the-art performance compared to existing variable-rate methods, and remains competitive with recent multi-model approaches. Notably, our method is the first learned image compression solution that outperforms VVC across a very wide range of bit rates using a single model, especially at high bit this http URL source code is available at \href{this https URL}{this https URL}. 

---
# Learn by Reasoning: Analogical Weight Generation for Few-Shot Class-Incremental Learning 

**Authors**: Jizhou Han, Chenhao Ding, Yuhang He, Songlin Dong, Qiang Wang, Xinyuan Gao, Yihong Gong  

**Link**: [PDF](https://arxiv.org/pdf/2503.21258)  

**Abstract**: Few-shot class-incremental Learning (FSCIL) enables models to learn new classes from limited data while retaining performance on previously learned classes. Traditional FSCIL methods often require fine-tuning parameters with limited new class data and suffer from a separation between learning new classes and utilizing old knowledge. Inspired by the analogical learning mechanisms of the human brain, we propose a novel analogical generative method. Our approach includes the Brain-Inspired Analogical Generator (BiAG), which derives new class weights from existing classes without parameter fine-tuning during incremental stages. BiAG consists of three components: Weight Self-Attention Module (WSA), Weight & Prototype Analogical Attention Module (WPAA), and Semantic Conversion Module (SCM). SCM uses Neural Collapse theory for semantic conversion, WSA supplements new class weights, and WPAA computes analogies to generate new class weights. Experiments on miniImageNet, CUB-200, and CIFAR-100 datasets demonstrate that our method achieves higher final and average accuracy compared to SOTA methods. 

---
# OminiAdapt: Learning Cross-Task Invariance for Robust and Environment-Aware Robotic Manipulation 

**Authors**: Yongxu Wang, Weiyun Yi, Xinhao Kong, Wanting Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.21257)  

**Abstract**: With the rapid development of embodied intelligence, leveraging large-scale human data for high-level imitation learning on humanoid robots has become a focal point of interest in both academia and industry. However, applying humanoid robots to precision operation domains remains challenging due to the complexities they face in perception and control processes, the long-standing physical differences in morphology and actuation mechanisms between humanoid robots and humans, and the lack of task-relevant features obtained from egocentric vision. To address the issue of covariate shift in imitation learning, this paper proposes an imitation learning algorithm tailored for humanoid robots. By focusing on the primary task objectives, filtering out background information, and incorporating channel feature fusion with spatial attention mechanisms, the proposed algorithm suppresses environmental disturbances and utilizes a dynamic weight update strategy to significantly improve the success rate of humanoid robots in accomplishing target tasks. Experimental results demonstrate that the proposed method exhibits robustness and scalability across various typical task scenarios, providing new ideas and approaches for autonomous learning and control in humanoid robots. The project will be open-sourced on GitHub. 

---
# Vision-to-Music Generation: A Survey 

**Authors**: Zhaokai Wang, Chenxi Bao, Le Zhuo, Jingrui Han, Yang Yue, Yihong Tang, Victor Shea-Jay Huang, Yue Liao  

**Link**: [PDF](https://arxiv.org/pdf/2503.21254)  

**Abstract**: Vision-to-music Generation, including video-to-music and image-to-music tasks, is a significant branch of multimodal artificial intelligence demonstrating vast application prospects in fields such as film scoring, short video creation, and dance music synthesis. However, compared to the rapid development of modalities like text and images, research in vision-to-music is still in its preliminary stage due to its complex internal structure and the difficulty of modeling dynamic relationships with video. Existing surveys focus on general music generation without comprehensive discussion on vision-to-music. In this paper, we systematically review the research progress in the field of vision-to-music generation. We first analyze the technical characteristics and core challenges for three input types: general videos, human movement videos, and images, as well as two output types of symbolic music and audio music. We then summarize the existing methodologies on vision-to-music generation from the architecture perspective. A detailed review of common datasets and evaluation metrics is provided. Finally, we discuss current challenges and promising directions for future research. We hope our survey can inspire further innovation in vision-to-music generation and the broader field of multimodal generation in academic research and industrial applications. To follow latest works and foster further innovation in this field, we are continuously maintaining a GitHub repository at this https URL. 

---
# Dual-Splitting Conformal Prediction for Multi-Step Time Series Forecasting 

**Authors**: Qingdi Yu, Zhiwei Cao, Ruihang Wang, Zhen Yang, Lijun Deng, Min Hu, Yong Luo, Xin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.21251)  

**Abstract**: Time series forecasting is crucial for applications like resource scheduling and risk management, where multi-step predictions provide a comprehensive view of future trends. Uncertainty Quantification (UQ) is a mainstream approach for addressing forecasting uncertainties, with Conformal Prediction (CP) gaining attention due to its model-agnostic nature and statistical guarantees. However, most variants of CP are designed for single-step predictions and face challenges in multi-step scenarios, such as reliance on real-time data and limited scalability. This highlights the need for CP methods specifically tailored to multi-step forecasting. We propose the Dual-Splitting Conformal Prediction (DSCP) method, a novel CP approach designed to capture inherent dependencies within time-series data for multi-step forecasting. Experimental results on real-world datasets from four different domains demonstrate that the proposed DSCP significantly outperforms existing CP variants in terms of the Winkler Score, achieving a performance improvement of up to 23.59% compared to state-of-the-art methods. Furthermore, we deployed the DSCP approach for renewable energy generation and IT load forecasting in power management of a real-world trajectory-based application, achieving an 11.25% reduction in carbon emissions through predictive optimization of data center operations and controls. 

---
# ResearchBench: Benchmarking LLMs in Scientific Discovery via Inspiration-Based Task Decomposition 

**Authors**: Yujie Liu, Zonglin Yang, Tong Xie, Jinjie Ni, Ben Gao, Yuqiang Li, Shixiang Tang, Wanli Ouyang, Erik Cambria, Dongzhan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.21248)  

**Abstract**: Large language models (LLMs) have demonstrated potential in assisting scientific research, yet their ability to discover high-quality research hypotheses remains unexamined due to the lack of a dedicated benchmark. To address this gap, we introduce the first large-scale benchmark for evaluating LLMs with a near-sufficient set of sub-tasks of scientific discovery: inspiration retrieval, hypothesis composition, and hypothesis ranking. We develop an automated framework that extracts critical components - research questions, background surveys, inspirations, and hypotheses - from scientific papers across 12 disciplines, with expert validation confirming its accuracy. To prevent data contamination, we focus exclusively on papers published in 2024, ensuring minimal overlap with LLM pretraining data. Our evaluation reveals that LLMs perform well in retrieving inspirations, an out-of-distribution task, suggesting their ability to surface novel knowledge associations. This positions LLMs as "research hypothesis mines", capable of facilitating automated scientific discovery by generating innovative hypotheses at scale with minimal human intervention. 

---
# Improving $(α, f)$-Byzantine Resilience in Federated Learning via layerwise aggregation and cosine distance 

**Authors**: Mario García-Márquez, Nuria Rodríguez-Barroso, M.Victoria Luzón, Francisco Herrera  

**Link**: [PDF](https://arxiv.org/pdf/2503.21244)  

**Abstract**: The rapid development of artificial intelligence systems has amplified societal concerns regarding their usage, necessitating regulatory frameworks that encompass data privacy. Federated Learning (FL) is posed as potential solution to data privacy challenges in distributed machine learning by enabling collaborative model training {without data sharing}. However, FL systems remain vulnerable to Byzantine attacks, where malicious nodes contribute corrupted model updates. While Byzantine Resilient operators have emerged as a widely adopted robust aggregation algorithm to mitigate these attacks, its efficacy diminishes significantly in high-dimensional parameter spaces, sometimes leading to poor performing models. This paper introduces Layerwise Cosine Aggregation, a novel aggregation scheme designed to enhance robustness of these rules in such high-dimensional settings while preserving computational efficiency. A theoretical analysis is presented, demonstrating the superior robustness of the proposed Layerwise Cosine Aggregation compared to original robust aggregation operators. Empirical evaluation across diverse image classification datasets, under varying data distributions and Byzantine attack scenarios, consistently demonstrates the improved performance of Layerwise Cosine Aggregation, achieving up to a 16% increase in model accuracy. 

---
# Feature-Enhanced Machine Learning for All-Cause Mortality Prediction in Healthcare Data 

**Authors**: HyeYoung Lee, Pavel Tsoi  

**Link**: [PDF](https://arxiv.org/pdf/2503.21241)  

**Abstract**: Accurate patient mortality prediction enables effective risk stratification, leading to personalized treatment plans and improved patient outcomes. However, predicting mortality in healthcare remains a significant challenge, with existing studies often focusing on specific diseases or limited predictor sets. This study evaluates machine learning models for all-cause in-hospital mortality prediction using the MIMIC-III database, employing a comprehensive feature engineering approach. Guided by clinical expertise and literature, we extracted key features such as vital signs (e.g., heart rate, blood pressure), laboratory results (e.g., creatinine, glucose), and demographic information. The Random Forest model achieved the highest performance with an AUC of 0.94, significantly outperforming other machine learning and deep learning approaches. This demonstrates Random Forest's robustness in handling high-dimensional, noisy clinical data and its potential for developing effective clinical decision support tools. Our findings highlight the importance of careful feature engineering for accurate mortality prediction. We conclude by discussing implications for clinical adoption and propose future directions, including enhancing model robustness and tailoring prediction models for specific diseases. 

---
# Bias-Aware Agent: Enhancing Fairness in AI-Driven Knowledge Retrieval 

**Authors**: Karanbir Singh, William Ngu  

**Link**: [PDF](https://arxiv.org/pdf/2503.21237)  

**Abstract**: Advancements in retrieving accessible information have evolved faster in the last few years compared to the decades since the internet's creation. Search engines, like Google, have been the number one way to find relevant data. They have always relied on the user's abilities to find the best information in its billions of links and sources at everybody's fingertips. The advent of large language models (LLMs) has completely transformed the field of information retrieval. The LLMs excel not only at retrieving relevant knowledge but also at summarizing it effectively, making information more accessible and consumable for users. On top of it, the rise of AI Agents has introduced another aspect to information retrieval i.e. dynamic information retrieval which enables the integration of real-time data such as weather forecasts, and financial data with the knowledge base to curate context-aware knowledge. However, despite these advancements the agents remain susceptible to issues of bias and fairness, challenges deeply rooted within the knowledge base and training of LLMs. This study introduces a novel approach to bias-aware knowledge retrieval by leveraging agentic framework and the innovative use of bias detectors as tools to identify and highlight inherent biases in the retrieved content. By empowering users with transparency and awareness, this approach aims to foster more equitable information systems and promote the development of responsible AI. 

---
# GenFusion: Closing the Loop between Reconstruction and Generation via Videos 

**Authors**: Sibo Wu, Congrong Xu, Binbin Huang, Andreas Geiger, Anpei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.21219)  

**Abstract**: Recently, 3D reconstruction and generation have demonstrated impressive novel view synthesis results, achieving high fidelity and efficiency. However, a notable conditioning gap can be observed between these two fields, e.g., scalable 3D scene reconstruction often requires densely captured views, whereas 3D generation typically relies on a single or no input view, which significantly limits their applications. We found that the source of this phenomenon lies in the misalignment between 3D constraints and generative priors. To address this problem, we propose a reconstruction-driven video diffusion model that learns to condition video frames on artifact-prone RGB-D renderings. Moreover, we propose a cyclical fusion pipeline that iteratively adds restoration frames from the generative model to the training set, enabling progressive expansion and addressing the viewpoint saturation limitations seen in previous reconstruction and generation pipelines. Our evaluation, including view synthesis from sparse view and masked input, validates the effectiveness of our approach. 

---
# Adversarial Wear and Tear: Exploiting Natural Damage for Generating Physical-World Adversarial Examples 

**Authors**: Samra Irshad, Seungkyu Lee, Nassir Navab, Hong Joo Lee, Seong Tae Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.21164)  

**Abstract**: The presence of adversarial examples in the physical world poses significant challenges to the deployment of Deep Neural Networks in safety-critical applications such as autonomous driving. Most existing methods for crafting physical-world adversarial examples are ad-hoc, relying on temporary modifications like shadows, laser beams, or stickers that are tailored to specific scenarios. In this paper, we introduce a new class of physical-world adversarial examples, AdvWT, which draws inspiration from the naturally occurring phenomenon of `wear and tear', an inherent property of physical objects. Unlike manually crafted perturbations, `wear and tear' emerges organically over time due to environmental degradation, as seen in the gradual deterioration of outdoor signboards. To achieve this, AdvWT follows a two-step approach. First, a GAN-based, unsupervised image-to-image translation network is employed to model these naturally occurring damages, particularly in the context of outdoor signboards. The translation network encodes the characteristics of damaged signs into a latent `damage style code'. In the second step, we introduce adversarial perturbations into the style code, strategically optimizing its transformation process. This manipulation subtly alters the damage style representation, guiding the network to generate adversarial images where the appearance of damages remains perceptually realistic, while simultaneously ensuring their effectiveness in misleading neural networks. Through comprehensive experiments on two traffic sign datasets, we show that AdvWT effectively misleads DNNs in both digital and physical domains. AdvWT achieves an effective attack success rate, greater robustness, and a more natural appearance compared to existing physical-world adversarial examples. Additionally, integrating AdvWT into training enhances a model's generalizability to real-world damaged signs. 

---
# Multi-Objective Optimization for Privacy-Utility Balance in Differentially Private Federated Learning 

**Authors**: Kanishka Ranaweera, David Smith, Pubudu N. Pathirana, Ming Ding, Thierry Rakotoarivelo, Aruna Seneviratne  

**Link**: [PDF](https://arxiv.org/pdf/2503.21159)  

**Abstract**: Federated learning (FL) enables collaborative model training across distributed clients without sharing raw data, making it a promising approach for privacy-preserving machine learning. However, ensuring differential privacy (DP) in FL presents challenges due to the trade-off between model utility and privacy protection. Clipping gradients before aggregation is a common strategy to limit privacy loss, but selecting an optimal clipping norm is non-trivial, as excessively high values compromise privacy, while overly restrictive clipping degrades model performance. In this work, we propose an adaptive clipping mechanism that dynamically adjusts the clipping norm using a multi-objective optimization framework. By integrating privacy and utility considerations into the optimization objective, our approach balances privacy preservation with model accuracy. We theoretically analyze the convergence properties of our method and demonstrate its effectiveness through extensive experiments on MNIST, Fashion-MNIST, and CIFAR-10 datasets. Our results show that adaptive clipping consistently outperforms fixed-clipping baselines, achieving improved accuracy under the same privacy constraints. This work highlights the potential of dynamic clipping strategies to enhance privacy-utility trade-offs in differentially private federated learning. 

---
# Federated Learning with Differential Privacy: An Utility-Enhanced Approach 

**Authors**: Kanishka Ranaweera, Dinh C. Nguyen, Pubudu N. Pathirana, David Smith, Ming Ding, Thierry Rakotoarivelo, Aruna Seneviratne  

**Link**: [PDF](https://arxiv.org/pdf/2503.21154)  

**Abstract**: Federated learning has emerged as an attractive approach to protect data privacy by eliminating the need for sharing clients' data while reducing communication costs compared with centralized machine learning algorithms. However, recent studies have shown that federated learning alone does not guarantee privacy, as private data may still be inferred from the uploaded parameters to the central server. In order to successfully avoid data leakage, adopting differential privacy (DP) in the local optimization process or in the local update aggregation process has emerged as two feasible ways for achieving sample-level or user-level privacy guarantees respectively, in federated learning models. However, compared to their non-private equivalents, these approaches suffer from a poor utility. To improve the privacy-utility trade-off, we present a modification to these vanilla differentially private algorithms based on a Haar wavelet transformation step and a novel noise injection scheme that significantly lowers the asymptotic bound of the noise variance. We also present a holistic convergence analysis of our proposed algorithm, showing that our method yields better convergence performance than the vanilla DP algorithms. Numerical experiments on real-world datasets demonstrate that our method outperforms existing approaches in model utility while maintaining the same privacy guarantees. 

---
# The Devil is in Low-Level Features for Cross-Domain Few-Shot Segmentation 

**Authors**: Yuhan Liu, Yixiong Zou, Yuhua Li, Ruixuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.21150)  

**Abstract**: Cross-Domain Few-Shot Segmentation (CDFSS) is proposed to transfer the pixel-level segmentation capabilities learned from large-scale source-domain datasets to downstream target-domain datasets, with only a few annotated images per class. In this paper, we focus on a well-observed but unresolved phenomenon in CDFSS: for target domains, particularly those distant from the source domain, segmentation performance peaks at the very early epochs, and declines sharply as the source-domain training proceeds. We delve into this phenomenon for an interpretation: low-level features are vulnerable to domain shifts, leading to sharper loss landscapes during the source-domain training, which is the devil of CDFSS. Based on this phenomenon and interpretation, we further propose a method that includes two plug-and-play modules: one to flatten the loss landscapes for low-level features during source-domain training as a novel sharpness-aware minimization method, and the other to directly supplement target-domain information to the model during target-domain testing by low-level-based calibration. Extensive experiments on four target datasets validate our rationale and demonstrate that our method surpasses the state-of-the-art method in CDFSS signifcantly by 3.71% and 5.34% average MIoU in 1-shot and 5-shot scenarios, respectively. 

---
# Optimizing Multi-DNN Inference on Mobile Devices through Heterogeneous Processor Co-Execution 

**Authors**: Yunquan Gao, Zhiguo Zhang, Praveen Kumar Donta, Chinmaya Kumar Dehury, Xiujun Wang, Dusit Niyato, Qiyang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.21109)  

**Abstract**: Deep Neural Networks (DNNs) are increasingly deployed across diverse industries, driving demand for mobile device support. However, existing mobile inference frameworks often rely on a single processor per model, limiting hardware utilization and causing suboptimal performance and energy efficiency. Expanding DNN accessibility on mobile platforms requires adaptive, resource-efficient solutions to meet rising computational needs without compromising functionality. Parallel inference of multiple DNNs on heterogeneous processors remains challenging. Some works partition DNN operations into subgraphs for parallel execution across processors, but these often create excessive subgraphs based only on hardware compatibility, increasing scheduling complexity and memory overhead.
To address this, we propose an Advanced Multi-DNN Model Scheduling (ADMS) strategy for optimizing multi-DNN inference on mobile heterogeneous processors. ADMS constructs an optimal subgraph partitioning strategy offline, balancing hardware operation support and scheduling granularity, and uses a processor-state-aware algorithm to dynamically adjust workloads based on real-time conditions. This ensures efficient workload distribution and maximizes processor utilization. Experiments show ADMS reduces multi-DNN inference latency by 4.04 times compared to vanilla frameworks. 

---
# Alleviating LLM-based Generative Retrieval Hallucination in Alipay Search 

**Authors**: Yedan Shen, Kaixin Wu, Yuechen Ding, Jingyuan Wen, Hong Liu, Mingjie Zhong, Zhouhan Lin, Jia Xu, Linjian Mo  

**Link**: [PDF](https://arxiv.org/pdf/2503.21098)  

**Abstract**: Generative retrieval (GR) has revolutionized document retrieval with the advent of large language models (LLMs), and LLM-based GR is gradually being adopted by the industry. Despite its remarkable advantages and potential, LLM-based GR suffers from hallucination and generates documents that are irrelevant to the query in some instances, severely challenging its credibility in practical applications. We thereby propose an optimized GR framework designed to alleviate retrieval hallucination, which integrates knowledge distillation reasoning in model training and incorporate decision agent to further improve retrieval precision. Specifically, we employ LLMs to assess and reason GR retrieved query-document (q-d) pairs, and then distill the reasoning data as transferred knowledge to the GR model. Moreover, we utilize a decision agent as post-processing to extend the GR retrieved documents through retrieval model and select the most relevant ones from multi perspectives as the final generative retrieval result. Extensive offline experiments on real-world datasets and online A/B tests on Fund Search and Insurance Search in Alipay demonstrate our framework's superiority and effectiveness in improving search quality and conversion gains. 

---
# Confidence Adjusted Surprise Measure for Active Resourceful Trials (CA-SMART): A Data-driven Active Learning Framework for Accelerating Material Discovery under Resource Constraints 

**Authors**: Ahmed Shoyeb Raihan, Zhichao Liu, Tanveer Hossain Bhuiyan, Imtiaz Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2503.21095)  

**Abstract**: Accelerating the discovery and manufacturing of advanced materials with specific properties is a critical yet formidable challenge due to vast search space, high costs of experiments, and time-intensive nature of material characterization. In recent years, active learning, where a surrogate machine learning (ML) model mimics the scientific discovery process of a human scientist, has emerged as a promising approach to address these challenges by guiding experimentation toward high-value outcomes with a limited budget. Among the diverse active learning philosophies, the concept of surprise (capturing the divergence between expected and observed outcomes) has demonstrated significant potential to drive experimental trials and refine predictive models. Scientific discovery often stems from surprise thereby making it a natural driver to guide the search process. Despite its promise, prior studies leveraging surprise metrics such as Shannon and Bayesian surprise lack mechanisms to account for prior confidence, leading to excessive exploration of uncertain regions that may not yield useful information. To address this, we propose the Confidence-Adjusted Surprise Measure for Active Resourceful Trials (CA-SMART), a novel Bayesian active learning framework tailored for optimizing data-driven experimentation. On a high level, CA-SMART incorporates Confidence-Adjusted Surprise (CAS) to dynamically balance exploration and exploitation by amplifying surprises in regions where the model is more certain while discounting them in highly uncertain areas. We evaluated CA-SMART on two benchmark functions (Six-Hump Camelback and Griewank) and in predicting the fatigue strength of steel. The results demonstrate superior accuracy and efficiency compared to traditional surprise metrics, standard Bayesian Optimization (BO) acquisition functions and conventional ML methods. 

---
# ZJUKLAB at SemEval-2025 Task 4: Unlearning via Model Merging 

**Authors**: Haoming Xu, Shuxun Wang, Yanqiu Zhao, Yi Zhong, Ziyan Jiang, Ningyuan Zhao, Shumin Deng, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.21088)  

**Abstract**: This paper presents the ZJUKLAB team's submission for SemEval-2025 Task 4: Unlearning Sensitive Content from Large Language Models. This task aims to selectively erase sensitive knowledge from large language models, avoiding both over-forgetting and under-forgetting issues. We propose an unlearning system that leverages Model Merging (specifically TIES-Merging), combining two specialized models into a more balanced unlearned model. Our system achieves competitive results, ranking second among 26 teams, with an online score of 0.944 for Task Aggregate and 0.487 for overall Aggregate. In this paper, we also conduct local experiments and perform a comprehensive analysis of the unlearning process, examining performance trajectories, loss dynamics, and weight perspectives, along with several supplementary experiments, to understand the effectiveness of our method. Furthermore, we analyze the shortcomings of our method and evaluation metrics, emphasizing that MIA scores and ROUGE-based metrics alone are insufficient to fully evaluate successful unlearning. Finally, we emphasize the need for more comprehensive evaluation methodologies and rethinking of unlearning objectives in future research. Code is available at this https URL. 

---
# Rerouting Connection: Hybrid Computer Vision Analysis Reveals Visual Similarity Between Indus and Tibetan-Yi Corridor Writing Systems 

**Authors**: Ooha Lakkadi Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2503.21074)  

**Abstract**: This thesis employs a hybrid CNN-Transformer architecture, in conjunction with a detailed anthropological framework, to investigate potential historical connections between the visual morphology of the Indus Valley script and pictographic systems of the Tibetan-Yi Corridor. Through an ensemble methodology of three target scripts across 15 independently trained models, we demonstrate that Tibetan-Yi Corridor scripts exhibit approximately six-fold higher visual similarity to the Indus script (61.7%-63.5%) than to the Bronze Age Proto-Cuneiform (10.2%-10.9%) or Proto-Elamite (7.6%-8.7%) systems. Additionally and contrarily to our current understanding of the networks of the Indus Valley Civilization, the Indus script unexpectedly maps closer to Tibetan-Yi Corridor scripts, with a mean cosine similarity of 0.629, than to the aforementioned contemporaneous West Asian signaries, both of which recorded mean cosine similarities of 0.104 and 0.080 despite their close geographic proximity and evident trade relations. Across various dimensionality reduction practices and clustering methodologies, the Indus script consistently clusters closest to Tibetan-Yi Corridor scripts. Our computational results align with qualitative observations of specific pictorial parallels in numeral systems, gender markers, and key iconographic elements; this is further supported by archaeological evidence of sustained contact networks along the ancient Shu-Shendu road in tandem with the Indus Valley Civilization's decline, providing a plausible transmission pathway. While alternative explanations cannot be ruled out, the specificity and consistency of observed similarities challenge conventional narratives of isolated script development and suggest more complex ancient cultural transmission networks between South and East Asia than previously recognized. 

---
# Can Large Language Models Predict Associations Among Human Attitudes? 

**Authors**: Ana Ma, Derek Powell  

**Link**: [PDF](https://arxiv.org/pdf/2503.21011)  

**Abstract**: Prior work has shown that large language models (LLMs) can predict human attitudes based on other attitudes, but this work has largely focused on predictions from highly similar and interrelated attitudes. In contrast, human attitudes are often strongly associated even across disparate and dissimilar topics. Using a novel dataset of human responses toward diverse attitude statements, we found that a frontier language model (GPT-4o) was able to recreate the pairwise correlations among individual attitudes and to predict individuals' attitudes from one another. Crucially, in an advance over prior work, we tested GPT-4o's ability to predict in the absence of surface-similarity between attitudes, finding that while surface similarity improves prediction accuracy, the model was still highly-capable of generating meaningful social inferences between dissimilar attitudes. Altogether, our findings indicate that LLMs capture crucial aspects of the deeper, latent structure of human belief systems. 

---
# Improving User Behavior Prediction: Leveraging Annotator Metadata in Supervised Machine Learning Models 

**Authors**: Lynnette Hui Xian Ng, Kokil Jaidka, Kaiyuan Tay, Hansin Ahuja, Niyati Chhaya  

**Link**: [PDF](https://arxiv.org/pdf/2503.21000)  

**Abstract**: Supervised machine-learning models often underperform in predicting user behaviors from conversational text, hindered by poor crowdsourced label quality and low NLP task accuracy. We introduce the Metadata-Sensitive Weighted-Encoding Ensemble Model (MSWEEM), which integrates annotator meta-features like fatigue and speeding. First, our results show MSWEEM outperforms standard ensembles by 14\% on held-out data and 12\% on an alternative dataset. Second, we find that incorporating signals of annotator behavior, such as speed and fatigue, significantly boosts model performance. Third, we find that annotators with higher qualifications, such as Master's, deliver more consistent and faster annotations. Given the increasing uncertainty over annotation quality, our experiments show that understanding annotator patterns is crucial for enhancing model accuracy in user behavior prediction. 

---
# FinAudio: A Benchmark for Audio Large Language Models in Financial Applications 

**Authors**: Yupeng Cao, Haohang Li, Yangyang Yu, Shashidhar Reddy Javaji, Yueru He, Jimin Huang, Zining Zhu, Qianqian Xie, Xiao-yang Liu, Koduvayur Subbalakshmi, Meikang Qiu, Sophia Ananiadou, Jian-Yun Nie  

**Link**: [PDF](https://arxiv.org/pdf/2503.20990)  

**Abstract**: Audio Large Language Models (AudioLLMs) have received widespread attention and have significantly improved performance on audio tasks such as conversation, audio understanding, and automatic speech recognition (ASR). Despite these advancements, there is an absence of a benchmark for assessing AudioLLMs in financial scenarios, where audio data, such as earnings conference calls and CEO speeches, are crucial resources for financial analysis and investment decisions. In this paper, we introduce \textsc{FinAudio}, the first benchmark designed to evaluate the capacity of AudioLLMs in the financial domain. We first define three tasks based on the unique characteristics of the financial domain: 1) ASR for short financial audio, 2) ASR for long financial audio, and 3) summarization of long financial audio. Then, we curate two short and two long audio datasets, respectively, and develop a novel dataset for financial audio summarization, comprising the \textsc{FinAudio} benchmark. Then, we evaluate seven prevalent AudioLLMs on \textsc{FinAudio}. Our evaluation reveals the limitations of existing AudioLLMs in the financial domain and offers insights for improving AudioLLMs. All datasets and codes will be released. 

---
# Patients Speak, AI Listens: LLM-based Analysis of Online Reviews Uncovers Key Drivers for Urgent Care Satisfaction 

**Authors**: Xiaoran Xu, Zhaoqian Xue, Chi Zhang, Jhonatan Medri, Junjie Xiong, Jiayan Zhou, Jin Jin, Yongfeng Zhang, Siyuan Ma, Lingyao Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.20981)  

**Abstract**: Investigating the public experience of urgent care facilities is essential for promoting community healthcare development. Traditional survey methods often fall short due to limited scope, time, and spatial coverage. Crowdsourcing through online reviews or social media offers a valuable approach to gaining such insights. With recent advancements in large language models (LLMs), extracting nuanced perceptions from reviews has become feasible. This study collects Google Maps reviews across the DMV and Florida areas and conducts prompt engineering with the GPT model to analyze the aspect-based sentiment of urgent care. We first analyze the geospatial patterns of various aspects, including interpersonal factors, operational efficiency, technical quality, finances, and facilities. Next, we determine Census Block Group(CBG)-level characteristics underpinning differences in public perception, including population density, median income, GINI Index, rent-to-income ratio, household below poverty rate, no insurance rate, and unemployment rate. Our results show that interpersonal factors and operational efficiency emerge as the strongest determinants of patient satisfaction in urgent care, while technical quality, finances, and facilities show no significant independent effects when adjusted for in multivariate models. Among socioeconomic and demographic factors, only population density demonstrates a significant but modest association with patient ratings, while the remaining factors exhibit no significant correlations. Overall, this study highlights the potential of crowdsourcing to uncover the key factors that matter to residents and provide valuable insights for stakeholders to improve public satisfaction with urgent care. 

---
# Competitive Multi-armed Bandit Games for Resource Sharing 

**Authors**: Hongbo Li, Lingjie Duan  

**Link**: [PDF](https://arxiv.org/pdf/2503.20975)  

**Abstract**: In modern resource-sharing systems, multiple agents access limited resources with unknown stochastic conditions to perform tasks. When multiple agents access the same resource (arm) simultaneously, they compete for successful usage, leading to contention and reduced rewards. This motivates our study of competitive multi-armed bandit (CMAB) games. In this paper, we study a new N-player K-arm competitive MAB game, where non-myopic players (agents) compete with each other to form diverse private estimations of unknown arms over time. Their possible collisions on same arms and time-varying nature of arm rewards make the policy analysis more involved than existing studies for myopic players. We explicitly analyze the threshold-based structures of social optimum and existing selfish policy, showing that the latter causes prolonged convergence time $\Omega(\frac{K}{\eta^2}\ln({\frac{KN}{\delta}}))$, while socially optimal policy with coordinated communication reduces it to $\mathcal{O}(\frac{K}{N\eta^2}\ln{(\frac{K}{\delta})})$. Based on the comparison, we prove that the competition among selfish players for the best arm can result in an infinite price of anarchy (PoA), indicating an arbitrarily large efficiency loss compared to social optimum. We further prove that no informational (non-monetary) mechanism (including Bayesian persuasion) can reduce the infinite PoA, as the strategic misreporting by non-myopic players undermines such approaches. To address this, we propose a Combined Informational and Side-Payment (CISP) mechanism, which provides socially optimal arm recommendations with proper informational and monetary incentives to players according to their time-varying private beliefs. Our CISP mechanism keeps ex-post budget balanced for social planner and ensures truthful reporting from players, achieving the minimum PoA=1 and same convergence time as social optimum. 

---
# Sociotechnical Effects of Machine Translation 

**Authors**: Joss Moorkens, Andy Way, Séamus Lankford  

**Link**: [PDF](https://arxiv.org/pdf/2503.20959)  

**Abstract**: While the previous chapters have shown how machine translation (MT) can be useful, in this chapter we discuss some of the side-effects and risks that are associated, and how they might be mitigated. With the move to neural MT and approaches using Large Language Models (LLMs), there is an associated impact on climate change, as the models built by multinational corporations are massive. They are hugely expensive to train, consume large amounts of electricity, and output huge volumes of kgCO2 to boot. However, smaller models which still perform to a high level of quality can be built with much lower carbon footprints, and tuning pre-trained models saves on the requirement to train from scratch. We also discuss the possible detrimental effects of MT on translators and other users. The topics of copyright and ownership of data are discussed, as well as ethical considerations on data and MT use. Finally, we show how if done properly, using MT in crisis scenarios can save lives, and we provide a method of how this might be done. 

---
# TS-Inverse: A Gradient Inversion Attack Tailored for Federated Time Series Forecasting Models 

**Authors**: Caspar Meijer, Jiyue Huang, Shreshtha Sharma, Elena Lazovik, Lydia Y. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.20952)  

**Abstract**: Federated learning (FL) for time series forecasting (TSF) enables clients with privacy-sensitive time series (TS) data to collaboratively learn accurate forecasting models, for example, in energy load prediction. Unfortunately, privacy risks in FL persist, as servers can potentially reconstruct clients' training data through gradient inversion attacks (GIA). Although GIA is demonstrated for image classification tasks, little is known about time series regression tasks. In this paper, we first conduct an extensive empirical study on inverting TS data across 4 TSF models and 4 datasets, identifying the unique challenges of reconstructing both observations and targets of TS data. We then propose TS-Inverse, a novel GIA that improves the inversion of TS data by (i) learning a gradient inversion model that outputs quantile predictions, (ii) a unique loss function that incorporates periodicity and trend regularization, and (iii) regularization according to the quantile predictions. Our evaluations demonstrate a remarkable performance of TS-Inverse, achieving at least a 2x-10x improvement in terms of the sMAPE metric over existing GIA methods on TS data. Code repository: this https URL 

---
# LATTE-MV: Learning to Anticipate Table Tennis Hits from Monocular Videos 

**Authors**: Daniel Etaat, Dvij Kalaria, Nima Rahmanian, Shankar Sastry  

**Link**: [PDF](https://arxiv.org/pdf/2503.20936)  

**Abstract**: Physical agility is a necessary skill in competitive table tennis, but by no means sufficient. Champions excel in this fast-paced and highly dynamic environment by anticipating their opponent's intent - buying themselves the necessary time to react. In this work, we take one step towards designing such an anticipatory agent. Previous works have developed systems capable of real-time table tennis gameplay, though they often do not leverage anticipation. Among the works that forecast opponent actions, their approaches are limited by dataset size and variety. Our paper contributes (1) a scalable system for reconstructing monocular video of table tennis matches in 3D and (2) an uncertainty-aware controller that anticipates opponent actions. We demonstrate in simulation that our policy improves the ball return rate against high-speed hits from 49.9% to 59.0% as compared to a baseline non-anticipatory policy. 

---
# Prototype Guided Backdoor Defense 

**Authors**: Venkat Adithya Amula, Sunayana Samavedam, Saurabh Saini, Avani Gupta, Narayanan P J  

**Link**: [PDF](https://arxiv.org/pdf/2503.20925)  

**Abstract**: Deep learning models are susceptible to {\em backdoor attacks} involving malicious attackers perturbing a small subset of training data with a {\em trigger} to causes misclassifications. Various triggers have been used, including semantic triggers that are easily realizable without requiring the attacker to manipulate the image. The emergence of generative AI has eased the generation of varied poisoned samples. Robustness across types of triggers is crucial to effective defense. We propose Prototype Guided Backdoor Defense (PGBD), a robust post-hoc defense that scales across different trigger types, including previously unsolved semantic triggers. PGBD exploits displacements in the geometric spaces of activations to penalize movements toward the trigger. This is done using a novel sanitization loss of a post-hoc fine-tuning step. The geometric approach scales easily to all types of attacks. PGBD achieves better performance across all settings. We also present the first defense against a new semantic attack on celebrity face images. Project page: \hyperlink{this https URL}{this https URL}. 

---
# D4R -- Exploring and Querying Relational Graphs Using Natural Language and Large Language Models -- the Case of Historical Documents 

**Authors**: Michel Boeglin, David Kahn, Josiane Mothe, Diego Ortiz, David Panzoli  

**Link**: [PDF](https://arxiv.org/pdf/2503.20914)  

**Abstract**: D4R is a digital platform designed to assist non-technical users, particularly historians, in exploring textual documents through advanced graphical tools for text analysis and knowledge extraction. By leveraging a large language model, D4R translates natural language questions into Cypher queries, enabling the retrieval of data from a Neo4J database. A user-friendly graphical interface allows for intuitive interaction, enabling users to navigate and analyse complex relational data extracted from unstructured textual documents. Originally designed to bridge the gap between AI technologies and historical research, D4R's capabilities extend to various other domains. A demonstration video and a live software demo are available. 

---
# Assessing Generative Models for Structured Data 

**Authors**: Reilly Cannon, Nicolette M. Laird, Caesar Vazquez, Andy Lin, Amy Wagler, Tony Chiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.20903)  

**Abstract**: Synthetic tabular data generation has emerged as a promising method to address limited data availability and privacy concerns. With the sharp increase in the performance of large language models in recent years, researchers have been interested in applying these models to the generation of tabular data. However, little is known about the quality of the generated tabular data from large language models. The predominant method for assessing the quality of synthetic tabular data is the train-synthetic-test-real approach, where the artificial examples are compared to the original by how well machine learning models, trained separately on the real and synthetic sets, perform in some downstream tasks. This method does not directly measure how closely the distribution of generated data approximates that of the original. This paper introduces rigorous methods for directly assessing synthetic tabular data against real data by looking at inter-column dependencies within the data. We find that large language models (GPT-2), both when queried via few-shot prompting and when fine-tuned, and GAN (CTGAN) models do not produce data with dependencies that mirror the original real data. Results from this study can inform future practice in synthetic data generation to improve data quality. 

---
# Robust Federated Learning Against Poisoning Attacks: A GAN-Based Defense Framework 

**Authors**: Usama Zafar, André Teixeira, Salman Toor  

**Link**: [PDF](https://arxiv.org/pdf/2503.20884)  

**Abstract**: Federated Learning (FL) enables collaborative model training across decentralized devices without sharing raw data, but it remains vulnerable to poisoning attacks that compromise model integrity. Existing defenses often rely on external datasets or predefined heuristics (e.g. number of malicious clients), limiting their effectiveness and scalability. To address these limitations, we propose a privacy-preserving defense framework that leverages a Conditional Generative Adversarial Network (cGAN) to generate synthetic data at the server for authenticating client updates, eliminating the need for external datasets. Our framework is scalable, adaptive, and seamlessly integrates into FL workflows. Extensive experiments on benchmark datasets demonstrate its robust performance against a variety of poisoning attacks, achieving high True Positive Rate (TPR) and True Negative Rate (TNR) of malicious and benign clients, respectively, while maintaining model accuracy. The proposed framework offers a practical and effective solution for securing federated learning systems. 

---
# VinaBench: Benchmark for Faithful and Consistent Visual Narratives 

**Authors**: Silin Gao, Sheryl Mathew, Li Mi, Sepideh Mamooler, Mengjie Zhao, Hiromi Wakaki, Yuki Mitsufuji, Syrielle Montariol, Antoine Bosselut  

**Link**: [PDF](https://arxiv.org/pdf/2503.20871)  

**Abstract**: Visual narrative generation transforms textual narratives into sequences of images illustrating the content of the text. However, generating visual narratives that are faithful to the input text and self-consistent across generated images remains an open challenge, due to the lack of knowledge constraints used for planning the stories. In this work, we propose a new benchmark, VinaBench, to address this challenge. Our benchmark annotates the underlying commonsense and discourse constraints in visual narrative samples, offering systematic scaffolds for learning the implicit strategies of visual storytelling. Based on the incorporated narrative constraints, we further propose novel metrics to closely evaluate the consistency of generated narrative images and the alignment of generations with the input textual narrative. Our results across three generative vision models demonstrate that learning with VinaBench's knowledge constraints effectively improves the faithfulness and cohesion of generated visual narratives. 

---
# Unified Multimodal Discrete Diffusion 

**Authors**: Alexander Swerdlow, Mihir Prabhudesai, Siddharth Gandhi, Deepak Pathak, Katerina Fragkiadaki  

**Link**: [PDF](https://arxiv.org/pdf/2503.20853)  

**Abstract**: Multimodal generative models that can understand and generate across multiple modalities are dominated by autoregressive (AR) approaches, which process tokens sequentially from left to right, or top to bottom. These models jointly handle images, text, video, and audio for various tasks such as image captioning, question answering, and image generation. In this work, we explore discrete diffusion models as a unified generative formulation in the joint text and image domain, building upon their recent success in text generation. Discrete diffusion models offer several advantages over AR models, including improved control over quality versus diversity of generated samples, the ability to perform joint multimodal inpainting (across both text and image domains), and greater controllability in generation through guidance. Leveraging these benefits, we present the first Unified Multimodal Discrete Diffusion (UniDisc) model which is capable of jointly understanding and generating text and images for a variety of downstream tasks. We compare UniDisc to multimodal AR models, performing a scaling analysis and demonstrating that UniDisc outperforms them in terms of both performance and inference-time compute, enhanced controllability, editability, inpainting, and flexible trade-off between inference time and generation quality. Code and additional visualizations are available at this https URL. 

---
# The Backfiring Effect of Weak AI Safety Regulation 

**Authors**: Benjamin Laufer, Jon Kleinberg, Hoda Heidari  

**Link**: [PDF](https://arxiv.org/pdf/2503.20848)  

**Abstract**: Recent policy proposals aim to improve the safety of general-purpose AI, but there is little understanding of the efficacy of different regulatory approaches to AI safety. We present a strategic model that explores the interactions between the regulator, the general-purpose AI technology creators, and domain specialists--those who adapt the AI for specific applications. Our analysis examines how different regulatory measures, targeting different parts of the development chain, affect the outcome of the development process. In particular, we assume AI technology is described by two key attributes: safety and performance. The regulator first sets a minimum safety standard that applies to one or both players, with strict penalties for non-compliance. The general-purpose creator then develops the technology, establishing its initial safety and performance levels. Next, domain specialists refine the AI for their specific use cases, and the resulting revenue is distributed between the specialist and generalist through an ex-ante bargaining process. Our analysis of this game reveals two key insights: First, weak safety regulation imposed only on the domain specialists can backfire. While it might seem logical to regulate use cases (as opposed to the general-purpose technology), our analysis shows that weak regulations targeting domain specialists alone can unintentionally reduce safety. This effect persists across a wide range of settings. Second, in sharp contrast to the previous finding, we observe that stronger, well-placed regulation can in fact benefit all players subjected to it. When regulators impose appropriate safety standards on both AI creators and domain specialists, the regulation functions as a commitment mechanism, leading to safety and performance gains, surpassing what is achieved under no regulation or regulating one player only. 

---
# Robust Deep Reinforcement Learning in Robotics via Adaptive Gradient-Masked Adversarial Attacks 

**Authors**: Zongyuan Zhang, Tianyang Duan, Zheng Lin, Dong Huang, Zihan Fang, Zekai Sun, Ling Xiong, Hongbin Liang, Heming Cui, Yong Cui, Yue Gao  

**Link**: [PDF](https://arxiv.org/pdf/2503.20844)  

**Abstract**: Deep reinforcement learning (DRL) has emerged as a promising approach for robotic control, but its realworld deployment remains challenging due to its vulnerability to environmental perturbations. Existing white-box adversarial attack methods, adapted from supervised learning, fail to effectively target DRL agents as they overlook temporal dynamics and indiscriminately perturb all state dimensions, limiting their impact on long-term rewards. To address these challenges, we propose the Adaptive Gradient-Masked Reinforcement (AGMR) Attack, a white-box attack method that combines DRL with a gradient-based soft masking mechanism to dynamically identify critical state dimensions and optimize adversarial policies. AGMR selectively allocates perturbations to the most impactful state features and incorporates a dynamic adjustment mechanism to balance exploration and exploitation during training. Extensive experiments demonstrate that AGMR outperforms state-of-the-art adversarial attack methods in degrading the performance of the victim agent and enhances the victim agent's robustness through adversarial defense mechanisms. 

---
# Advancing Vulnerability Classification with BERT: A Multi-Objective Learning Model 

**Authors**: Himanshu Tiwari  

**Link**: [PDF](https://arxiv.org/pdf/2503.20831)  

**Abstract**: The rapid increase in cybersecurity vulnerabilities necessitates automated tools for analyzing and classifying vulnerability reports. This paper presents a novel Vulnerability Report Classifier that leverages the BERT (Bidirectional Encoder Representations from Transformers) model to perform multi-label classification of Common Vulnerabilities and Exposures (CVE) reports from the National Vulnerability Database (NVD). The classifier predicts both the severity (Low, Medium, High, Critical) and vulnerability types (e.g., Buffer Overflow, XSS) from textual descriptions. We introduce a custom training pipeline using a combined loss function-Cross-Entropy for severity and Binary Cross-Entropy with Logits for types-integrated into a Hugging Face Trainer subclass. Experiments on recent NVD data demonstrate promising results, with decreasing evaluation loss across epochs. The system is deployed via a REST API and a Streamlit UI, enabling real-time vulnerability analysis. This work contributes a scalable, open-source solution for cybersecurity practitioners to automate vulnerability triage. 

---
# Exploiting Temporal State Space Sharing for Video Semantic Segmentation 

**Authors**: Syed Ariff Syed Hesham, Yun Liu, Guolei Sun, Henghui Ding, Jing Yang, Ender Konukoglu, Xue Geng, Xudong Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.20824)  

**Abstract**: Video semantic segmentation (VSS) plays a vital role in understanding the temporal evolution of scenes. Traditional methods often segment videos frame-by-frame or in a short temporal window, leading to limited temporal context, redundant computations, and heavy memory requirements. To this end, we introduce a Temporal Video State Space Sharing (TV3S) architecture to leverage Mamba state space models for temporal feature sharing. Our model features a selective gating mechanism that efficiently propagates relevant information across video frames, eliminating the need for a memory-heavy feature pool. By processing spatial patches independently and incorporating shifted operation, TV3S supports highly parallel computation in both training and inference stages, which reduces the delay in sequential state space processing and improves the scalability for long video sequences. Moreover, TV3S incorporates information from prior frames during inference, achieving long-range temporal coherence and superior adaptability to extended sequences. Evaluations on the VSPW and Cityscapes datasets reveal that our approach outperforms current state-of-the-art methods, establishing a new standard for VSS with consistent results across long video sequences. By achieving a good balance between accuracy and efficiency, TV3S shows a significant advancement in spatiotemporal modeling, paving the way for efficient video analysis. The code is publicly available at this https URL. 

---
# Synthetic Video Enhances Physical Fidelity in Video Synthesis 

**Authors**: Qi Zhao, Xingyu Ni, Ziyu Wang, Feng Cheng, Ziyan Yang, Lu Jiang, Bohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.20822)  

**Abstract**: We investigate how to enhance the physical fidelity of video generation models by leveraging synthetic videos derived from computer graphics pipelines. These rendered videos respect real-world physics, such as maintaining 3D consistency, and serve as a valuable resource that can potentially improve video generation models. To harness this potential, we propose a solution that curates and integrates synthetic data while introducing a method to transfer its physical realism to the model, significantly reducing unwanted artifacts. Through experiments on three representative tasks emphasizing physical consistency, we demonstrate its efficacy in enhancing physical fidelity. While our model still lacks a deep understanding of physics, our work offers one of the first empirical demonstrations that synthetic video enhances physical fidelity in video synthesis. Website: this https URL 

---
# Fundamental Safety-Capability Trade-offs in Fine-tuning Large Language Models 

**Authors**: Pin-Yu Chen, Han Shen, Payel Das, Tianyi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.20807)  

**Abstract**: Fine-tuning Large Language Models (LLMs) on some task-specific datasets has been a primary use of LLMs. However, it has been empirically observed that this approach to enhancing capability inevitably compromises safety, a phenomenon also known as the safety-capability trade-off in LLM fine-tuning. This paper presents a theoretical framework for understanding the interplay between safety and capability in two primary safety-aware LLM fine-tuning strategies, providing new insights into the effects of data similarity, context overlap, and alignment loss landscape. Our theoretical results characterize the fundamental limits of the safety-capability trade-off in LLM fine-tuning, which are also validated by numerical experiments. 

---
# AED: Automatic Discovery of Effective and Diverse Vulnerabilities for Autonomous Driving Policy with Large Language Models 

**Authors**: Le Qiu, Zelai Xu, Qixin Tan, Wenhao Tang, Chao Yu, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.20804)  

**Abstract**: Assessing the safety of autonomous driving policy is of great importance, and reinforcement learning (RL) has emerged as a powerful method for discovering critical vulnerabilities in driving policies. However, existing RL-based approaches often struggle to identify vulnerabilities that are both effective-meaning the autonomous vehicle is genuinely responsible for the accidents-and diverse-meaning they span various failure types. To address these challenges, we propose AED, a framework that uses large language models (LLMs) to automatically discover effective and diverse vulnerabilities in autonomous driving policies. We first utilize an LLM to automatically design reward functions for RL training. Then we let the LLM consider a diverse set of accident types and train adversarial policies for different accident types in parallel. Finally, we use preference-based learning to filter ineffective accidents and enhance the effectiveness of each vulnerability. Experiments across multiple simulated traffic scenarios and tested policies show that AED uncovers a broader range of vulnerabilities and achieves higher attack success rates compared with expert-designed rewards, thereby reducing the need for manual reward engineering and improving the diversity and effectiveness of vulnerability discovery. 

---
# CEFW: A Comprehensive Evaluation Framework for Watermark in Large Language Models 

**Authors**: Shuhao Zhang, Bo Cheng, Jiale Han, Yuli Chen, Zhixuan Wu, Changbao Li, Pingli Gu  

**Link**: [PDF](https://arxiv.org/pdf/2503.20802)  

**Abstract**: Text watermarking provides an effective solution for identifying synthetic text generated by large language models. However, existing techniques often focus on satisfying specific criteria while ignoring other key aspects, lacking a unified evaluation. To fill this gap, we propose the Comprehensive Evaluation Framework for Watermark (CEFW), a unified framework that comprehensively evaluates watermarking methods across five key dimensions: ease of detection, fidelity of text quality, minimal embedding cost, robustness to adversarial attacks, and imperceptibility to prevent imitation or forgery. By assessing watermarks according to all these key criteria, CEFW offers a thorough evaluation of their practicality and effectiveness. Moreover, we introduce a simple and effective watermarking method called Balanced Watermark (BW), which guarantees robustness and imperceptibility through balancing the way watermark information is added. Extensive experiments show that BW outperforms existing methods in overall performance across all evaluation dimensions. We release our code to the community for future research. this https URL. 

---
# Evidencing Unauthorized Training Data from AI Generated Content using Information Isotopes 

**Authors**: Qi Tao, Yin Jinhua, Cai Dongqi, Xie Yueqi, Wang Huili, Hu Zhiyang, Yang Peiru, Nan Guoshun, Zhou Zhili, Wang Shangguang, Lyu Lingjuan, Huang Yongfeng, Lane Nicholas  

**Link**: [PDF](https://arxiv.org/pdf/2503.20800)  

**Abstract**: In light of scaling laws, many AI institutions are intensifying efforts to construct advanced AIs on extensive collections of high-quality human data. However, in a rush to stay competitive, some institutions may inadvertently or even deliberately include unauthorized data (like privacy- or intellectual property-sensitive content) for AI training, which infringes on the rights of data owners. Compounding this issue, these advanced AI services are typically built on opaque cloud platforms, which restricts access to internal information during AI training and inference, leaving only the generated outputs available for forensics. Thus, despite the introduction of legal frameworks by various countries to safeguard data rights, uncovering evidence of data misuse in modern opaque AI applications remains a significant challenge. In this paper, inspired by the ability of isotopes to trace elements within chemical reactions, we introduce the concept of information isotopes and elucidate their properties in tracing training data within opaque AI systems. Furthermore, we propose an information isotope tracing method designed to identify and provide evidence of unauthorized data usage by detecting the presence of target information isotopes in AI generations. We conduct experiments on ten AI models (including GPT-4o, Claude-3.5, and DeepSeek) and four benchmark datasets in critical domains (medical data, copyrighted books, and news). Results show that our method can distinguish training datasets from non-training datasets with 99\% accuracy and significant evidence (p-value$<0.001$) by examining a data entry equivalent in length to a research paper. The findings show the potential of our work as an inclusive tool for empowering individuals, including those without expertise in AI, to safeguard their data rights in the rapidly evolving era of AI advancements and applications. 

---
# Payload-Aware Intrusion Detection with CMAE and Large Language Models 

**Authors**: Yongcheol Kim, Chanjae Lee, Young Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2503.20798)  

**Abstract**: Intrusion Detection Systems (IDS) are crucial for identifying malicious traffic, yet traditional signature-based methods struggle with zero-day attacks and high false positive rates. AI-driven packet-capture analysis offers a promising alternative. However, existing approaches rely heavily on flow-based or statistical features, limiting their ability to detect fine-grained attack patterns. This study proposes Xavier-CMAE, an enhanced Convolutional Multi-Head Attention Ensemble (CMAE) model that improves detection accuracy while reducing computational overhead. By replacing Word2Vec embeddings with a Hex2Int tokenizer and Xavier initialization, Xavier-CMAE eliminates pre-training, accelerates training, and achieves 99.971% accuracy with a 0.018% false positive rate, outperforming Word2Vec-based methods. Additionally, we introduce LLM-CMAE, which integrates pre-trained Large Language Model (LLM) tokenizers into CMAE. While LLMs enhance feature extraction, their computational cost hinders real-time detection. LLM-CMAE balances efficiency and performance, reaching 99.969% accuracy with a 0.019% false positive rate. This work advances AI-powered IDS by (1) introducing a payload-based detection framework, (2) enhancing efficiency with Xavier-CMAE, and (3) integrating LLM tokenizers for improved real-time detection. 

---
# EXPLICATE: Enhancing Phishing Detection through Explainable AI and LLM-Powered Interpretability 

**Authors**: Bryan Lim, Roman Huerta, Alejandro Sotelo, Anthonie Quintela, Priyanka Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2503.20796)  

**Abstract**: Sophisticated phishing attacks have emerged as a major cybersecurity threat, becoming more common and difficult to prevent. Though machine learning techniques have shown promise in detecting phishing attacks, they function mainly as "black boxes" without revealing their decision-making rationale. This lack of transparency erodes the trust of users and diminishes their effective threat response. We present EXPLICATE: a framework that enhances phishing detection through a three-component architecture: an ML-based classifier using domain-specific features, a dual-explanation layer combining LIME and SHAP for complementary feature-level insights, and an LLM enhancement using DeepSeek v3 to translate technical explanations into accessible natural language. Our experiments show that EXPLICATE attains 98.4 % accuracy on all metrics, which is on par with existing deep learning techniques but has better explainability. High-quality explanations are generated by the framework with an accuracy of 94.2 % as well as a consistency of 96.8\% between the LLM output and model prediction. We create EXPLICATE as a fully usable GUI application and a light Chrome extension, showing its applicability in many deployment situations. The research shows that high detection performance can go hand-in-hand with meaningful explainability in security applications. Most important, it addresses the critical divide between automated AI and user trust in phishing detection systems. 

---
# Toward a Human-Centered AI-assisted Colonoscopy System in Australia 

**Authors**: Hsiang-Ting Chen, Yuan Zhang, Gustavo Carneiro, Rajvinder Singh  

**Link**: [PDF](https://arxiv.org/pdf/2503.20790)  

**Abstract**: While AI-assisted colonoscopy promises improved colorectal cancer screening, its success relies on effective integration into clinical practice, not just algorithmic accuracy. This paper, based on an Australian field study (observations and gastroenterologist interviews), highlights a critical disconnect: current development prioritizes machine learning model performance, overlooking essential aspects of user interface design, workflow integration, and overall user experience. Industry interactions reveal a similar emphasis on data and algorithms. To realize AI's full potential, the HCI community must champion user-centered design, ensuring these systems are usable, support endoscopist expertise, and enhance patient outcomes. 

---
