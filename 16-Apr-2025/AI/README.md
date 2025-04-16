# Embodied World Models Emerge from Navigational Task in Open-Ended Environments 

**Authors**: Li Jin, Liu Jia  

**Link**: [PDF](https://arxiv.org/pdf/2504.11419)  

**Abstract**: Understanding how artificial systems can develop spatial awareness and reasoning has long been a challenge in AI research. Traditional models often rely on passive observation, but embodied cognition theory suggests that deeper understanding emerges from active interaction with the environment. This study investigates whether neural networks can autonomously internalize spatial concepts through interaction, focusing on planar navigation tasks. Using Gated Recurrent Units (GRUs) combined with Meta-Reinforcement Learning (Meta-RL), we show that agents can learn to encode spatial properties like direction, distance, and obstacle avoidance. We introduce Hybrid Dynamical Systems (HDS) to model the agent-environment interaction as a closed dynamical system, revealing stable limit cycles that correspond to optimal navigation strategies. Ridge Representation allows us to map navigation paths into a fixed-dimensional behavioral space, enabling comparison with neural states. Canonical Correlation Analysis (CCA) confirms strong alignment between these representations, suggesting that the agent's neural states actively encode spatial knowledge. Intervention experiments further show that specific neural dimensions are causally linked to navigation performance. This work provides an approach to bridging the gap between action and perception in AI, offering new insights into building adaptive, interpretable models that can generalize across complex environments. The causal validation of neural representations also opens new avenues for understanding and controlling the internal mechanisms of AI systems, pushing the boundaries of how machines learn and reason in dynamic, real-world scenarios. 

---
# Kimina-Prover Preview: Towards Large Formal Reasoning Models with Reinforcement Learning 

**Authors**: Haiming Wang, Mert Unsal, Xiaohan Lin, Mantas Baksys, Junqi Liu, Marco Dos Santos, Flood Sung, Marina Vinyes, Zhenzhe Ying, Zekai Zhu, Jianqiao Lu, Hugues de Saxcé, Bolton Bailey, Chendong Song, Chenjun Xiao, Dehao Zhang, Ebony Zhang, Frederick Pu, Han Zhu, Jiawei Liu, Jonas Bayer, Julien Michel, Longhui Yu, Léo Dreyfus-Schmidt, Lewis Tunstall, Luigi Pagani, Moreira Machado, Pauline Bourigault, Ran Wang, Stanislas Polu, Thibaut Barroyer, Wen-Ding Li, Yazhe Niu, Yann Fleureau, Yangyang Hu, Zhouliang Yu, Zihan Wang, Zhilin Yang, Zhengying Liu, Jia Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.11354)  

**Abstract**: We introduce Kimina-Prover Preview, a large language model that pioneers a novel reasoning-driven exploration paradigm for formal theorem proving, as showcased in this preview release. Trained with a large-scale reinforcement learning pipeline from Qwen2.5-72B, Kimina-Prover demonstrates strong performance in Lean 4 proof generation by employing a structured reasoning pattern we term \textit{formal reasoning pattern}. This approach allows the model to emulate human problem-solving strategies in Lean, iteratively generating and refining proof steps. Kimina-Prover sets a new state-of-the-art on the miniF2F benchmark, reaching 80.7% with pass@8192. Beyond improved benchmark performance, our work yields several key insights: (1) Kimina-Prover exhibits high sample efficiency, delivering strong results even with minimal sampling (pass@1) and scaling effectively with computational budget, stemming from its unique reasoning pattern and RL training; (2) we demonstrate clear performance scaling with model size, a trend previously unobserved for neural theorem provers in formal mathematics; (3) the learned reasoning style, distinct from traditional search algorithms, shows potential to bridge the gap between formal verification and informal mathematical intuition. We open source distilled versions with 1.5B and 7B parameters of Kimina-Prover 

---
# Learning to Be A Doctor: Searching for Effective Medical Agent Architectures 

**Authors**: Yangyang Zhuang, Wenjia Jiang, Jiayu Zhang, Ze Yang, Joey Tianyi Zhou, Chi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.11301)  

**Abstract**: Large Language Model (LLM)-based agents have demonstrated strong capabilities across a wide range of tasks, and their application in the medical domain holds particular promise due to the demand for high generalizability and reliance on interdisciplinary knowledge. However, existing medical agent systems often rely on static, manually crafted workflows that lack the flexibility to accommodate diverse diagnostic requirements and adapt to emerging clinical scenarios. Motivated by the success of automated machine learning (AutoML), this paper introduces a novel framework for the automated design of medical agent architectures. Specifically, we define a hierarchical and expressive agent search space that enables dynamic workflow adaptation through structured modifications at the node, structural, and framework levels. Our framework conceptualizes medical agents as graph-based architectures composed of diverse, functional node types and supports iterative self-improvement guided by diagnostic feedback. Experimental results on skin disease diagnosis tasks demonstrate that the proposed method effectively evolves workflow structures and significantly enhances diagnostic accuracy over time. This work represents the first fully automated framework for medical agent architecture design and offers a scalable, adaptable foundation for deploying intelligent agents in real-world clinical environments. 

---
# Towards Automated Safety Requirements Derivation Using Agent-based RAG 

**Authors**: Balahari Vignesh Balu, Florian Geissler, Francesco Carella, Joao-Vitor Zacchi, Josef Jiru, Nuria Mata, Reinhard Stolle  

**Link**: [PDF](https://arxiv.org/pdf/2504.11243)  

**Abstract**: We study the automated derivation of safety requirements in a self-driving vehicle use case, leveraging LLMs in combination with agent-based retrieval-augmented generation. Conventional approaches that utilise pre-trained LLMs to assist in safety analyses typically lack domain-specific knowledge. Existing RAG approaches address this issue, yet their performance deteriorates when handling complex queries and it becomes increasingly harder to retrieve the most relevant information. This is particularly relevant for safety-relevant applications. In this paper, we propose the use of agent-based RAG to derive safety requirements and show that the retrieved information is more relevant to the queries. We implement an agent-based approach on a document pool of automotive standards and the Apollo case study, as a representative example of an automated driving perception system. Our solution is tested on a data set of safety requirement questions and answers, extracted from the Apollo data. Evaluating a set of selected RAG metrics, we present and discuss advantages of a agent-based approach compared to default RAG methods. 

---
# Nondeterministic Polynomial-time Problem Challenge: An Ever-Scaling Reasoning Benchmark for LLMs 

**Authors**: Chang Yang, Ruiyu Wang, Junzhe Jiang, Qi Jiang, Qinggang Zhang, Yanchen Deng, Shuxin Li, Shuyue Hu, Bo Li, Florian T. Pokorny, Xiao Huang, Xinrun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.11239)  

**Abstract**: Reasoning is the fundamental capability of large language models (LLMs). Due to the rapid progress of LLMs, there are two main issues of current benchmarks: i) these benchmarks can be crushed in a short time (less than 1 year), and ii) these benchmarks may be easily hacked. To handle these issues, we propose the ever-scalingness for building the benchmarks which are uncrushable, unhackable, auto-verifiable and general. This paper presents Nondeterministic Polynomial-time Problem Challenge (NPPC), an ever-scaling reasoning benchmark for LLMs. Specifically, the NPPC has three main modules: i) npgym, which provides a unified interface of 25 well-known NP-complete problems and can generate any number of instances with any levels of complexities, ii) npsolver: which provides a unified interface to evaluate the problem instances with both online and offline models via APIs and local deployments, respectively, and iii) npeval: which provides the comprehensive and ready-to-use tools to analyze the performances of LLMs over different problems, the number of tokens, the aha moments, the reasoning errors and the solution errors. Extensive experiments over widely-used LLMs demonstrate: i) NPPC can successfully decrease the performances of advanced LLMs' performances to below 10%, demonstrating that NPPC is uncrushable, ii) DeepSeek-R1, Claude-3.7-Sonnet, and o1/o3-mini are the most powerful LLMs, where DeepSeek-R1 outperforms Claude-3.7-Sonnet and o1/o3-mini in most NP-complete problems considered, and iii) the numbers of tokens, aha moments in the advanced LLMs, e.g., Claude-3.7-Sonnet and DeepSeek-R1, are observed first to increase and then decrease when the problem instances become more and more difficult. We believe that NPPC is the first ever-scaling reasoning benchmark, serving as the uncrushable and unhackable testbed for LLMs toward artificial general intelligence (AGI). 

---
# Mutual Understanding between People and Systems via Neurosymbolic AI and Knowledge Graphs 

**Authors**: Irene Celino, Mario Scrocca, Agnese Chiatti  

**Link**: [PDF](https://arxiv.org/pdf/2504.11200)  

**Abstract**: This chapter investigates the concept of mutual understanding between humans and systems, positing that Neuro-symbolic Artificial Intelligence (NeSy AI) methods can significantly enhance this mutual understanding by leveraging explicit symbolic knowledge representations with data-driven learning models. We start by introducing three critical dimensions to characterize mutual understanding: sharing knowledge, exchanging knowledge, and governing knowledge. Sharing knowledge involves aligning the conceptual models of different agents to enable a shared understanding of the domain of interest. Exchanging knowledge relates to ensuring the effective and accurate communication between agents. Governing knowledge concerns establishing rules and processes to regulate the interaction between agents. Then, we present several different use case scenarios that demonstrate the application of NeSy AI and Knowledge Graphs to aid meaningful exchanges between human, artificial, and robotic agents. These scenarios highlight both the potential and the challenges of combining top-down symbolic reasoning with bottom-up neural learning, guiding the discussion of the coverage provided by current solutions along the dimensions of sharing, exchanging, and governing knowledge. Concurrently, this analysis facilitates the identification of gaps and less developed aspects in mutual understanding to address in future research. 

---
# Enhancing multimodal analogical reasoning with Logic Augmented Generation 

**Authors**: Anna Sofia Lippolis, Andrea Giovanni Nuzzolese, Aldo Gangemi  

**Link**: [PDF](https://arxiv.org/pdf/2504.11190)  

**Abstract**: Recent advances in Large Language Models have demonstrated their capabilities across a variety of tasks. However, automatically extracting implicit knowledge from natural language remains a significant challenge, as machines lack active experience with the physical world. Given this scenario, semantic knowledge graphs can serve as conceptual spaces that guide the automated text generation reasoning process to achieve more efficient and explainable results. In this paper, we apply a logic-augmented generation (LAG) framework that leverages the explicit representation of a text through a semantic knowledge graph and applies it in combination with prompt heuristics to elicit implicit analogical connections. This method generates extended knowledge graph triples representing implicit meaning, enabling systems to reason on unlabeled multimodal data regardless of the domain. We validate our work through three metaphor detection and understanding tasks across four datasets, as they require deep analogical reasoning capabilities. The results show that this integrated approach surpasses current baselines, performs better than humans in understanding visual metaphors, and enables more explainable reasoning processes, though still has inherent limitations in metaphor understanding, especially for domain-specific metaphors. Furthermore, we propose a thorough error analysis, discussing issues with metaphorical annotations and current evaluation methods. 

---
# C-SHAP for time series: An approach to high-level temporal explanations 

**Authors**: Annemarie Jutte, Faizan Ahmed, Jeroen Linssen, Maurice van Keulen  

**Link**: [PDF](https://arxiv.org/pdf/2504.11159)  

**Abstract**: Time series are ubiquitous in domains such as energy forecasting, healthcare, and industry. Using AI systems, some tasks within these domains can be efficiently handled. Explainable AI (XAI) aims to increase the reliability of AI solutions by explaining model reasoning. For time series, many XAI methods provide point- or sequence-based attribution maps. These methods explain model reasoning in terms of low-level patterns. However, they do not capture high-level patterns that may also influence model reasoning. We propose a concept-based method to provide explanations in terms of these high-level patterns. In this paper, we present C-SHAP for time series, an approach which determines the contribution of concepts to a model outcome. We provide a general definition of C-SHAP and present an example implementation using time series decomposition. Additionally, we demonstrate the effectiveness of the methodology through a use case from the energy domain. 

---
# Emergence of Goal-Directed Behaviors via Active Inference with Self-Prior 

**Authors**: Dongmin Kim, Hoshinori Kanazawa, Naoto Yoshida, Yasuo Kuniyoshi  

**Link**: [PDF](https://arxiv.org/pdf/2504.11075)  

**Abstract**: Infants often exhibit goal-directed behaviors, such as reaching for a sensory stimulus, even when no external reward criterion is provided. These intrinsically motivated behaviors facilitate spontaneous exploration and learning of the body and environment during early developmental stages. Although computational modeling can offer insight into the mechanisms underlying such behaviors, many existing studies on intrinsic motivation focus primarily on how exploration contributes to acquiring external rewards. In this paper, we propose a novel density model for an agent's own multimodal sensory experiences, called the "self-prior," and investigate whether it can autonomously induce goal-directed behavior. Integrated within an active inference framework based on the free energy principle, the self-prior generates behavioral references purely from an intrinsic process that minimizes mismatches between average past sensory experiences and current observations. This mechanism is also analogous to the acquisition and utilization of a body schema through continuous interaction with the environment. We examine this approach in a simulated environment and confirm that the agent spontaneously reaches toward a tactile stimulus. Our study implements intrinsically motivated behavior shaped by the agent's own sensory experiences, demonstrating the spontaneous emergence of intentional behavior during early development. 

---
# ARise: Towards Knowledge-Augmented Reasoning via Risk-Adaptive Search 

**Authors**: Yize Zhang, Tianshu Wang, Sirui Chen, Kun Wang, Xingyu Zeng, Hongyu Lin, Xianpei Han, Le Sun, Chaochao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2504.10893)  

**Abstract**: Large language models (LLMs) have demonstrated impressive capabilities and are receiving increasing attention to enhance their reasoning through scaling test--time compute. However, their application in open--ended, knowledge--intensive, complex reasoning scenarios is still limited. Reasoning--oriented methods struggle to generalize to open--ended scenarios due to implicit assumptions of complete world knowledge. Meanwhile, knowledge--augmented reasoning (KAR) methods fail to address two core challenges: 1) error propagation, where errors in early steps cascade through the chain, and 2) verification bottleneck, where the explore--exploit tradeoff arises in multi--branch decision processes. To overcome these limitations, we introduce ARise, a novel framework that integrates risk assessment of intermediate reasoning states with dynamic retrieval--augmented generation (RAG) within a Monte Carlo tree search paradigm. This approach enables effective construction and optimization of reasoning plans across multiple maintained hypothesis branches. Experimental results show that ARise significantly outperforms the state--of--the--art KAR methods by up to 23.10%, and the latest RAG-equipped large reasoning models by up to 25.37%. 

---
# Understanding the theoretical properties of projected Bellman equation, linear Q-learning, and approximate value iteration 

**Authors**: Han-Dong Lim, Donghwan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2504.10865)  

**Abstract**: In this paper, we study the theoretical properties of the projected Bellman equation (PBE) and two algorithms to solve this equation: linear Q-learning and approximate value iteration (AVI). We consider two sufficient conditions for the existence of a solution to PBE : strictly negatively row dominating diagonal (SNRDD) assumption and a condition motivated by the convergence of AVI. The SNRDD assumption also ensures the convergence of linear Q-learning, and its relationship with the convergence of AVI is examined. Lastly, several interesting observations on the solution of PBE are provided when using $\epsilon$-greedy policy. 

---
# Hallucination-Aware Generative Pretrained Transformer for Cooperative Aerial Mobility Control 

**Authors**: Hyojun Ahn, Seungcheol Oh, Gyu Seon Kim, Soyi Jung, Soohyun Park, Joongheon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2504.10831)  

**Abstract**: This paper proposes SafeGPT, a two-tiered framework that integrates generative pretrained transformers (GPTs) with reinforcement learning (RL) for efficient and reliable unmanned aerial vehicle (UAV) last-mile deliveries. In the proposed design, a Global GPT module assigns high-level tasks such as sector allocation, while an On-Device GPT manages real-time local route planning. An RL-based safety filter monitors each GPT decision and overrides unsafe actions that could lead to battery depletion or duplicate visits, effectively mitigating hallucinations. Furthermore, a dual replay buffer mechanism helps both the GPT modules and the RL agent refine their strategies over time. Simulation results demonstrate that SafeGPT achieves higher delivery success rates compared to a GPT-only baseline, while substantially reducing battery consumption and travel distance. These findings validate the efficacy of combining GPT-based semantic reasoning with formal safety guarantees, contributing a viable solution for robust and energy-efficient UAV logistics. 

---
# Ride-pool Assignment Algorithms: Modern Implementation and Swapping Heuristics 

**Authors**: Matthew Zalesak, Hins Hu, Samitha Samaranayake  

**Link**: [PDF](https://arxiv.org/pdf/2504.10649)  

**Abstract**: On-demand ride-pooling has emerged as a popular urban transportation solution, addressing the efficiency limitations of traditional ride-hailing services by grouping multiple riding requests with spatiotemporal proximity into a single vehicle. Although numerous algorithms have been developed for the Ride-pool Assignment Problem (RAP) -- a core component of ride-pooling systems, there is a lack of open-source implementations, making it difficult to benchmark these algorithms on a common dataset and objective. In this paper, we present the implementation details of a ride-pool simulator that encompasses several key ride-pool assignment algorithms, along with associated components such as vehicle routing and rebalancing. We also open-source a highly optimized and modular C++ codebase, designed to facilitate the extension of new algorithms and features. Additionally, we introduce a family of swapping-based local-search heuristics to enhance existing ride-pool assignment algorithms, achieving a better balance between performance and computational efficiency. Extensive experiments on a large-scale, real-world dataset from Manhattan, NYC reveal that while all selected algorithms perform comparably, the newly proposed Multi-Round Linear Assignment with Cyclic Exchange (LA-MR-CE) algorithm achieves a state-of-the-art service rate with significantly reduced computational time. Furthermore, an in-depth analysis suggests that a performance barrier exists for all myopic ride-pool assignment algorithms due to the system's capacity bottleneck, and incorporating future information could be key to overcoming this limitation. 

---
# Explainable Artificial Intelligence techniques for interpretation of food datasets: a review 

**Authors**: Leonardo Arrighi, Ingrid Alves de Moraes, Marco Zullich, Michele Simonato, Douglas Fernandes Barbin, Sylvio Barbon Junior  

**Link**: [PDF](https://arxiv.org/pdf/2504.10527)  

**Abstract**: Artificial Intelligence (AI) has become essential for analyzing complex data and solving highly-challenging tasks. It is being applied across numerous disciplines beyond computer science, including Food Engineering, where there is a growing demand for accurate and trustworthy predictions to meet stringent food quality standards. However, this requires increasingly complex AI models, raising reliability concerns. In response, eXplainable AI (XAI) has emerged to provide insights into AI decision-making, aiding model interpretation by developers and users. Nevertheless, XAI remains underutilized in Food Engineering, limiting model reliability. For instance, in food quality control, AI models using spectral imaging can detect contaminants or assess freshness levels, but their opaque decision-making process hinders adoption. XAI techniques such as SHAP (Shapley Additive Explanations) and Grad-CAM (Gradient-weighted Class Activation Mapping) can pinpoint which spectral wavelengths or image regions contribute most to a prediction, enhancing transparency and aiding quality control inspectors in verifying AI-generated assessments. This survey presents a taxonomy for classifying food quality research using XAI techniques, organized by data types and explanation methods, to guide researchers in choosing suitable approaches. We also highlight trends, challenges, and opportunities to encourage the adoption of XAI in Food Engineering. 

---
# Toward Super Agent System with Hybrid AI Routers 

**Authors**: Yuhang Yao, Haixin Wang, Yibo Chen, Jiawen Wang, Min Chang Jordan Ren, Bosheng Ding, Salman Avestimehr, Chaoyang He  

**Link**: [PDF](https://arxiv.org/pdf/2504.10519)  

**Abstract**: AI Agents powered by Large Language Models are transforming the world through enormous applications. A super agent has the potential to fulfill diverse user needs, such as summarization, coding, and research, by accurately understanding user intent and leveraging the appropriate tools to solve tasks. However, to make such an agent viable for real-world deployment and accessible at scale, significant optimizations are required to ensure high efficiency and low cost. This paper presents a design of the Super Agent System. Upon receiving a user prompt, the system first detects the intent of the user, then routes the request to specialized task agents with the necessary tools or automatically generates agentic workflows. In practice, most applications directly serve as AI assistants on edge devices such as phones and robots. As different language models vary in capability and cloud-based models often entail high computational costs, latency, and privacy concerns, we then explore the hybrid mode where the router dynamically selects between local and cloud models based on task complexity. Finally, we introduce the blueprint of an on-device super agent enhanced with cloud. With advances in multi-modality models and edge hardware, we envision that most computations can be handled locally, with cloud collaboration only as needed. Such architecture paves the way for super agents to be seamlessly integrated into everyday life in the near future. 

---
# DeepMath-103K: A Large-Scale, Challenging, Decontaminated, and Verifiable Mathematical Dataset for Advancing Reasoning 

**Authors**: Zhiwei He, Tian Liang, Jiahao Xu, Qiuzhi Liu, Xingyu Chen, Yue Wang, Linfeng Song, Dian Yu, Zhenwen Liang, Wenxuan Wang, Zhuosheng Zhang, Rui Wang, Zhaopeng Tu, Haitao Mi, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.11456)  

**Abstract**: The capacity for complex mathematical reasoning is a key benchmark for artificial intelligence. While reinforcement learning (RL) applied to LLMs shows promise, progress is significantly hindered by the lack of large-scale training data that is sufficiently challenging, possesses verifiable answer formats suitable for RL, and is free from contamination with evaluation benchmarks. To address these limitations, we introduce DeepMath-103K, a new, large-scale dataset comprising approximately 103K mathematical problems, specifically designed to train advanced reasoning models via RL. DeepMath-103K is curated through a rigorous pipeline involving source analysis, stringent decontamination against numerous benchmarks, and filtering for high difficulty (primarily Levels 5-9), significantly exceeding existing open resources in challenge. Each problem includes a verifiable final answer, enabling rule-based RL, and three distinct R1-generated solutions suitable for diverse training paradigms like supervised fine-tuning or distillation. Spanning a wide range of mathematical topics, DeepMath-103K promotes the development of generalizable reasoning. We demonstrate that models trained on DeepMath-103K achieve significant improvements on challenging mathematical benchmarks, validating its effectiveness. We release DeepMath-103K publicly to facilitate community progress in building more capable AI reasoning systems: this https URL. 

---
# Elucidating the Design Space of Multimodal Protein Language Models 

**Authors**: Cheng-Yen, Hsieh, Xinyou Wang, Daiheng Zhang, Dongyu Xue, Fei Ye, Shujian Huang, Zaixiang Zheng, Quanquan Gu  

**Link**: [PDF](https://arxiv.org/pdf/2504.11454)  

**Abstract**: Multimodal protein language models (PLMs) integrate sequence and token-based structural information, serving as a powerful foundation for protein modeling, generation, and design. However, the reliance on tokenizing 3D structures into discrete tokens causes substantial loss of fidelity about fine-grained structural details and correlations. In this paper, we systematically elucidate the design space of multimodal PLMs to overcome their limitations. We identify tokenization loss and inaccurate structure token predictions by the PLMs as major bottlenecks. To address these, our proposed design space covers improved generative modeling, structure-aware architectures and representation learning, and data exploration. Our advancements approach finer-grained supervision, demonstrating that token-based multimodal PLMs can achieve robust structural modeling. The effective design methods dramatically improve the structure generation diversity, and notably, folding abilities of our 650M model by reducing the RMSD from 5.52 to 2.36 on PDB testset, even outperforming 3B baselines and on par with the specialized folding models. 

---
# A Clean Slate for Offline Reinforcement Learning 

**Authors**: Matthew Thomas Jackson, Uljad Berdica, Jarek Liesen, Shimon Whiteson, Jakob Nicolaus Foerster  

**Link**: [PDF](https://arxiv.org/pdf/2504.11453)  

**Abstract**: Progress in offline reinforcement learning (RL) has been impeded by ambiguous problem definitions and entangled algorithmic designs, resulting in inconsistent implementations, insufficient ablations, and unfair evaluations. Although offline RL explicitly avoids environment interaction, prior methods frequently employ extensive, undocumented online evaluation for hyperparameter tuning, complicating method comparisons. Moreover, existing reference implementations differ significantly in boilerplate code, obscuring their core algorithmic contributions. We address these challenges by first introducing a rigorous taxonomy and a transparent evaluation protocol that explicitly quantifies online tuning budgets. To resolve opaque algorithmic design, we provide clean, minimalistic, single-file implementations of various model-free and model-based offline RL methods, significantly enhancing clarity and achieving substantial speed-ups. Leveraging these streamlined implementations, we propose Unifloral, a unified algorithm that encapsulates diverse prior approaches within a single, comprehensive hyperparameter space, enabling algorithm development in a shared hyperparameter space. Using Unifloral with our rigorous evaluation protocol, we develop two novel algorithms - TD3-AWR (model-free) and MoBRAC (model-based) - which substantially outperform established baselines. Our implementation is publicly available at this https URL. 

---
# TextArena 

**Authors**: Leon Guertler, Bobby Cheng, Simon Yu, Bo Liu, Leshem Choshen, Cheston Tan  

**Link**: [PDF](https://arxiv.org/pdf/2504.11442)  

**Abstract**: TextArena is an open-source collection of competitive text-based games for training and evaluation of agentic behavior in Large Language Models (LLMs). It spans 57+ unique environments (including single-player, two-player, and multi-player setups) and allows for easy evaluation of model capabilities via an online-play system (against humans and other submitted models) with real-time TrueSkill scores. Traditional benchmarks rarely assess dynamic social skills such as negotiation, theory of mind, and deception, creating a gap that TextArena addresses. Designed with research, community and extensibility in mind, TextArena emphasizes ease of adding new games, adapting the framework, testing models, playing against the models, and training models. Detailed documentation of environments, games, leaderboard, and examples are available on this https URL and this https URL. 

---
# Greedy Restart Schedules: A Baseline for Dynamic Algorithm Selection on Numerical Black-box Optimization Problems 

**Authors**: Lennart Schäpermeier  

**Link**: [PDF](https://arxiv.org/pdf/2504.11440)  

**Abstract**: In many optimization domains, there are multiple different solvers that contribute to the overall state-of-the-art, each performing better on some, and worse on other types of problem instances. Meta-algorithmic approaches, such as instance-based algorithm selection, configuration and scheduling, aim to close this gap by extracting the most performance possible from a set of (configurable) optimizers. In this context, the best performing individual algorithms are often hand-crafted hybrid heuristics which perform many restarts of fast local optimization approaches. However, data-driven techniques to create optimized restart schedules have not yet been extensively studied.
Here, we present a simple scheduling approach that iteratively selects the algorithm performing best on the distribution of unsolved training problems at time of selection, resulting in a problem-independent solver schedule. We demonstrate our approach using well-known optimizers from numerical black-box optimization on the BBOB testbed, bridging much of the gap between single and virtual best solver from the original portfolio across various evaluation protocols. Our greedy restart schedule presents a powerful baseline for more complex dynamic algorithm selection models. 

---
# Masculine Defaults via Gendered Discourse in Podcasts and Large Language Models 

**Authors**: Maria Teleki, Xiangjue Dong, Haoran Liu, James Caverlee  

**Link**: [PDF](https://arxiv.org/pdf/2504.11431)  

**Abstract**: Masculine defaults are widely recognized as a significant type of gender bias, but they are often unseen as they are under-researched. Masculine defaults involve three key parts: (i) the cultural context, (ii) the masculine characteristics or behaviors, and (iii) the reward for, or simply acceptance of, those masculine characteristics or behaviors. In this work, we study discourse-based masculine defaults, and propose a twofold framework for (i) the large-scale discovery and analysis of gendered discourse words in spoken content via our Gendered Discourse Correlation Framework (GDCF); and (ii) the measurement of the gender bias associated with these gendered discourse words in LLMs via our Discourse Word-Embedding Association Test (D-WEAT). We focus our study on podcasts, a popular and growing form of social media, analyzing 15,117 podcast episodes. We analyze correlations between gender and discourse words -- discovered via LDA and BERTopic -- to automatically form gendered discourse word lists. We then study the prevalence of these gendered discourse words in domain-specific contexts, and find that gendered discourse-based masculine defaults exist in the domains of business, technology/politics, and video games. Next, we study the representation of these gendered discourse words from a state-of-the-art LLM embedding model from OpenAI, and find that the masculine discourse words have a more stable and robust representation than the feminine discourse words, which may result in better system performance on downstream tasks for men. Hence, men are rewarded for their discourse patterns with better system performance by one of the state-of-the-art language models -- and this embedding disparity is a representational harm and a masculine default. 

---
# A Dual-Space Framework for General Knowledge Distillation of Large Language Models 

**Authors**: Xue Zhang, Songming Zhang, Yunlong Liang, Fandong Meng, Yufeng Chen, Jinan Xu, Jie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.11426)  

**Abstract**: Knowledge distillation (KD) is a promising solution to compress large language models (LLMs) by transferring their knowledge to smaller models. During this process, white-box KD methods usually minimize the distance between the output distributions of the teacher model and the student model to transfer more information. However, we reveal that the current white-box KD framework exhibits two limitations: a) bridging probability distributions from different output spaces will limit the similarity between the teacher model and the student model; b) this framework cannot be applied to LLMs with different vocabularies. One of the root causes for these limitations is that the distributions from the teacher and the student for KD are output by different prediction heads, which yield distributions in different output spaces and dimensions. Therefore, in this paper, we propose a dual-space knowledge distillation (DSKD) framework that unifies the prediction heads of the teacher and the student models for KD. Specifically, we first introduce two projectors with ideal initialization to project the teacher/student hidden states into the student/teacher representation spaces. After this, the hidden states from different models can share the same head and unify the output spaces of the distributions. Furthermore, we develop an exact token alignment (ETA) algorithm to align the same tokens in two differently-tokenized sequences. Based on the above, our DSKD framework is a general KD framework that supports both off-policy and on-policy KD, and KD between any two LLMs regardless of their vocabularies. Extensive experiments on instruction-following, mathematical reasoning, and code generation benchmarks show that DSKD significantly outperforms existing methods based on the current white-box KD framework and surpasses other cross-tokenizer KD methods for LLMs with different vocabularies. 

---
# ADT: Tuning Diffusion Models with Adversarial Supervision 

**Authors**: Dazhong Shen, Guanglu Song, Yi Zhang, Bingqi Ma, Lujundong Li, Dongzhi Jiang, Zhuofan Zong, Yu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.11423)  

**Abstract**: Diffusion models have achieved outstanding image generation by reversing a forward noising process to approximate true data distributions. During training, these models predict diffusion scores from noised versions of true samples in a single forward pass, while inference requires iterative denoising starting from white noise. This training-inference divergences hinder the alignment between inference and training data distributions, due to potential prediction biases and cumulative error accumulation. To address this problem, we propose an intuitive but effective fine-tuning framework, called Adversarial Diffusion Tuning (ADT), by stimulating the inference process during optimization and aligning the final outputs with training data by adversarial supervision. Specifically, to achieve robust adversarial training, ADT features a siamese-network discriminator with a fixed pre-trained backbone and lightweight trainable parameters, incorporates an image-to-image sampling strategy to smooth discriminative difficulties, and preserves the original diffusion loss to prevent discriminator hacking. In addition, we carefully constrain the backward-flowing path for back-propagating gradients along the inference path without incurring memory overload or gradient explosion. Finally, extensive experiments on Stable Diffusion models (v1.5, XL, and v3), demonstrate that ADT significantly improves both distribution alignment and image quality. 

---
# Measures of Variability for Risk-averse Policy Gradient 

**Authors**: Yudong Luo, Yangchen Pan, Jiaqi Tan, Pascal Poupart  

**Link**: [PDF](https://arxiv.org/pdf/2504.11412)  

**Abstract**: Risk-averse reinforcement learning (RARL) is critical for decision-making under uncertainty, which is especially valuable in high-stake applications. However, most existing works focus on risk measures, e.g., conditional value-at-risk (CVaR), while measures of variability remain underexplored. In this paper, we comprehensively study nine common measures of variability, namely Variance, Gini Deviation, Mean Deviation, Mean-Median Deviation, Standard Deviation, Inter-Quantile Range, CVaR Deviation, Semi_Variance, and Semi_Standard Deviation. Among them, four metrics have not been previously studied in RARL. We derive policy gradient formulas for these unstudied metrics, improve gradient estimation for Gini Deviation, analyze their gradient properties, and incorporate them with the REINFORCE and PPO frameworks to penalize the dispersion of returns.
Our empirical study reveals that variance-based metrics lead to unstable policy updates. In contrast, CVaR Deviation and Gini Deviation show consistent performance across different randomness and evaluation domains, achieving high returns while effectively learning risk-averse policies. Mean Deviation and Semi_Standard Deviation are also competitive across different scenarios. This work provides a comprehensive overview of variability measures in RARL, offering practical insights for risk-aware decision-making and guiding future research on risk metrics and RARL algorithms. 

---
# Multi-level Cellular Automata for FLIM networks 

**Authors**: Felipe Crispim Salvagnini, Jancarlo F. Gomes, Cid A. N. Santos, Silvio Jamil F. Guimarães, Alexandre X. Falcão  

**Link**: [PDF](https://arxiv.org/pdf/2504.11406)  

**Abstract**: The necessity of abundant annotated data and complex network architectures presents a significant challenge in deep-learning Salient Object Detection (deep SOD) and across the broader deep-learning landscape. This challenge is particularly acute in medical applications in developing countries with limited computational resources. Combining modern and classical techniques offers a path to maintaining competitive performance while enabling practical applications. Feature Learning from Image Markers (FLIM) methodology empowers experts to design convolutional encoders through user-drawn markers, with filters learned directly from these annotations. Recent findings demonstrate that coupling a FLIM encoder with an adaptive decoder creates a flyweight network suitable for SOD, requiring significantly fewer parameters than lightweight models and eliminating the need for backpropagation. Cellular Automata (CA) methods have proven successful in data-scarce scenarios but require proper initialization -- typically through user input, priors, or randomness. We propose a practical intersection of these approaches: using FLIM networks to initialize CA states with expert knowledge without requiring user interaction for each image. By decoding features from each level of a FLIM network, we can initialize multiple CAs simultaneously, creating a multi-level framework. Our method leverages the hierarchical knowledge encoded across different network layers, merging multiple saliency maps into a high-quality final output that functions as a CA ensemble. Benchmarks across two challenging medical datasets demonstrate the competitiveness of our multi-level CA approach compared to established models in the deep SOD literature. 

---
# VideoPanda: Video Panoramic Diffusion with Multi-view Attention 

**Authors**: Kevin Xie, Amirmojtaba Sabour, Jiahui Huang, Despoina Paschalidou, Greg Klar, Umar Iqbal, Sanja Fidler, Xiaohui Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2504.11389)  

**Abstract**: High resolution panoramic video content is paramount for immersive experiences in Virtual Reality, but is non-trivial to collect as it requires specialized equipment and intricate camera setups. In this work, we introduce VideoPanda, a novel approach for synthesizing 360$^\circ$ videos conditioned on text or single-view video data. VideoPanda leverages multi-view attention layers to augment a video diffusion model, enabling it to generate consistent multi-view videos that can be combined into immersive panoramic content. VideoPanda is trained jointly using two conditions: text-only and single-view video, and supports autoregressive generation of long-videos. To overcome the computational burden of multi-view video generation, we randomly subsample the duration and camera views used during training and show that the model is able to gracefully generalize to generating more frames during inference. Extensive evaluations on both real-world and synthetic video datasets demonstrate that VideoPanda generates more realistic and coherent 360$^\circ$ panoramas across all input conditions compared to existing methods. Visit the project website at this https URL for results. 

---
# Trajectory Encoding Temporal Graph Networks 

**Authors**: Jiafeng Xiong, Rizos Sakellariou  

**Link**: [PDF](https://arxiv.org/pdf/2504.11386)  

**Abstract**: Temporal Graph Networks (TGNs) have demonstrated significant success in dynamic graph tasks such as link prediction and node classification. Both tasks comprise transductive settings, where the model predicts links among known nodes, and in inductive settings, where it generalises learned patterns to previously unseen nodes. Existing TGN designs face a dilemma under these dual scenarios. Anonymous TGNs, which rely solely on temporal and structural information, offer strong inductive generalisation but struggle to distinguish known nodes. In contrast, non-anonymous TGNs leverage node features to excel in transductive tasks yet fail to adapt to new nodes. To address this challenge, we propose Trajectory Encoding TGN (TETGN). Our approach introduces automatically expandable node identifiers (IDs) as learnable temporal positional features and performs message passing over these IDs to capture each node's historical context. By integrating this trajectory-aware module with a standard TGN using multi-head attention, TETGN effectively balances transductive accuracy with inductive generalisation. Experimental results on three real-world datasets show that TETGN significantly outperforms strong baselines on both link prediction and node classification tasks, demonstrating its ability to unify the advantages of anonymous and non-anonymous models for dynamic graph learning. 

---
# A Winner-Takes-All Mechanism for Event Generation 

**Authors**: Yongkang Huo, Fuvio Forni, Rodolphe Sepulchre  

**Link**: [PDF](https://arxiv.org/pdf/2504.11374)  

**Abstract**: We present a novel framework for central pattern generator design that leverages the intrinsic rebound excitability of neurons in combination with winner-takes-all computation. Our approach unifies decision-making and rhythmic pattern generation within a simple yet powerful network architecture that employs all-to-all inhibitory connections enhanced by designable excitatory interactions. This design offers significant advantages regarding ease of implementation, adaptability, and robustness. We demonstrate its efficacy through a ring oscillator model, which exhibits adaptive phase and frequency modulation, making the framework particularly promising for applications in neuromorphic systems and robotics. 

---
# OpenTuringBench: An Open-Model-based Benchmark and Framework for Machine-Generated Text Detection and Attribution 

**Authors**: Lucio La Cava, Andrea Tagarelli  

**Link**: [PDF](https://arxiv.org/pdf/2504.11369)  

**Abstract**: Open Large Language Models (OLLMs) are increasingly leveraged in generative AI applications, posing new challenges for detecting their outputs. We propose OpenTuringBench, a new benchmark based on OLLMs, designed to train and evaluate machine-generated text detectors on the Turing Test and Authorship Attribution problems. OpenTuringBench focuses on a representative set of OLLMs, and features a number of challenging evaluation tasks, including human/machine-manipulated texts, out-of-domain texts, and texts from previously unseen models. We also provide OTBDetector, a contrastive learning framework to detect and attribute OLLM-based machine-generated texts. Results highlight the relevance and varying degrees of difficulty of the OpenTuringBench tasks, with our detector achieving remarkable capabilities across the various tasks and outperforming most existing detectors. Resources are available on the OpenTuringBench Hugging Face repository at this https URL 

---
# Teaching Large Language Models to Reason through Learning and Forgetting 

**Authors**: Tianwei Ni, Allen Nie, Sapana Chaudhary, Yao Liu, Huzefa Rangwala, Rasool Fakoor  

**Link**: [PDF](https://arxiv.org/pdf/2504.11364)  

**Abstract**: Leveraging inference-time search in large language models has proven effective in further enhancing a trained model's capability to solve complex mathematical and reasoning problems. However, this approach significantly increases computational costs and inference time, as the model must generate and evaluate multiple candidate solutions to identify a viable reasoning path. To address this, we propose an effective approach that integrates search capabilities directly into the model by fine-tuning it using both successful (learning) and failed reasoning paths (forgetting) derived from diverse search methods. While fine-tuning the model with these data might seem straightforward, we identify a critical issue: the model's search capability tends to degrade rapidly if fine-tuning is performed naively. We show that this degradation can be substantially mitigated by employing a smaller learning rate. Extensive experiments on the challenging Game-of-24 and Countdown mathematical reasoning benchmarks show that our approach not only outperforms both standard fine-tuning and inference-time search baselines but also significantly reduces inference time by 180$\times$. 

---
# DataSentinel: A Game-Theoretic Detection of Prompt Injection Attacks 

**Authors**: Yupei Liu, Yuqi Jia, Jinyuan Jia, Dawn Song, Neil Zhenqiang Gong  

**Link**: [PDF](https://arxiv.org/pdf/2504.11358)  

**Abstract**: LLM-integrated applications and agents are vulnerable to prompt injection attacks, where an attacker injects prompts into their inputs to induce attacker-desired outputs. A detection method aims to determine whether a given input is contaminated by an injected prompt. However, existing detection methods have limited effectiveness against state-of-the-art attacks, let alone adaptive ones. In this work, we propose DataSentinel, a game-theoretic method to detect prompt injection attacks. Specifically, DataSentinel fine-tunes an LLM to detect inputs contaminated with injected prompts that are strategically adapted to evade detection. We formulate this as a minimax optimization problem, with the objective of fine-tuning the LLM to detect strong adaptive attacks. Furthermore, we propose a gradient-based method to solve the minimax optimization problem by alternating between the inner max and outer min problems. Our evaluation results on multiple benchmark datasets and LLMs show that DataSentinel effectively detects both existing and adaptive prompt injection attacks. 

---
# Neural Networks for on-chip Model Predictive Control: a Method to Build Optimized Training Datasets and its application to Type-1 Diabetes 

**Authors**: Alberto Castillo, Elliot Pryor, Anas El Fathi, Boris Kovatchev, Marc Breton  

**Link**: [PDF](https://arxiv.org/pdf/2504.11355)  

**Abstract**: Training Neural Networks (NNs) to behave as Model Predictive Control (MPC) algorithms is an effective way to implement them in constrained embedded devices. By collecting large amounts of input-output data, where inputs represent system states and outputs are MPC-generated control actions, NNs can be trained to replicate MPC behavior at a fraction of the computational cost. However, although the composition of the training data critically influences the final NN accuracy, methods for systematically optimizing it remain underexplored. In this paper, we introduce the concept of Optimally-Sampled Datasets (OSDs) as ideal training sets and present an efficient algorithm for generating them. An OSD is a parametrized subset of all the available data that (i) preserves existing MPC information up to a certain numerical resolution, (ii) avoids duplicate or near-duplicate states, and (iii) becomes saturated or complete. We demonstrate the effectiveness of OSDs by training NNs to replicate the University of Virginia's MPC algorithm for automated insulin delivery in Type-1 Diabetes, achieving a four-fold improvement in final accuracy. Notably, two OSD-trained NNs received regulatory clearance for clinical testing as the first NN-based control algorithm for direct human insulin dosing. This methodology opens new pathways for implementing advanced optimizations on resource-constrained embedded platforms, potentially revolutionizing how complex algorithms are deployed. 

---
# Explicit and Implicit Representations in AI-based 3D Reconstruction for Radiology: A systematic literature review 

**Authors**: Yuezhe Yang, Boyu Yang, Yaqian Wang, Yang He, Xingbo Dong, Zhe Jin  

**Link**: [PDF](https://arxiv.org/pdf/2504.11349)  

**Abstract**: The demand for high-quality medical imaging in clinical practice and assisted diagnosis has made 3D reconstruction in radiological imaging a key research focus. Artificial intelligence (AI) has emerged as a promising approach to enhancing reconstruction accuracy while reducing acquisition and processing time, thereby minimizing patient radiation exposure and discomfort and ultimately benefiting clinical diagnosis. This review explores state-of-the-art AI-based 3D reconstruction algorithms in radiological imaging, categorizing them into explicit and implicit approaches based on their underlying principles. Explicit methods include point-based, volume-based, and Gaussian representations, while implicit methods encompass implicit prior embedding and neural radiance fields. Additionally, we examine commonly used evaluation metrics and benchmark datasets. Finally, we discuss the current state of development, key challenges, and future research directions in this evolving field. Our project available on: this https URL. 

---
# Interpretable Hybrid-Rule Temporal Point Processes 

**Authors**: Yunyang Cao, Juekai Lin, Hongye Wang, Wenhao Li, Bo Jin  

**Link**: [PDF](https://arxiv.org/pdf/2504.11344)  

**Abstract**: Temporal Point Processes (TPPs) are widely used for modeling event sequences in various medical domains, such as disease onset prediction, progression analysis, and clinical decision support. Although TPPs effectively capture temporal dynamics, their lack of interpretability remains a critical challenge. Recent advancements have introduced interpretable TPPs. However, these methods fail to incorporate numerical features, thereby limiting their ability to generate precise predictions. To address this issue, we propose Hybrid-Rule Temporal Point Processes (HRTPP), a novel framework that integrates temporal logic rules with numerical features, improving both interpretability and predictive accuracy in event modeling. HRTPP comprises three key components: basic intensity for intrinsic event likelihood, rule-based intensity for structured temporal dependencies, and numerical feature intensity for dynamic probability modulation. To effectively discover valid rules, we introduce a two-phase rule mining strategy with Bayesian optimization. To evaluate our method, we establish a multi-criteria assessment framework, incorporating rule validity, model fitting, and temporal predictive accuracy. Experimental results on real-world medical datasets demonstrate that HRTPP outperforms state-of-the-art interpretable TPPs in terms of predictive performance and clinical interpretability. In case studies, the rules extracted by HRTPP explain the disease progression, offering valuable contributions to medical diagnosis. 

---
# A Minimalist Approach to LLM Reasoning: from Rejection Sampling to Reinforce 

**Authors**: Wei Xiong, Jiarui Yao, Yuhui Xu, Bo Pang, Lei Wang, Doyen Sahoo, Junnan Li, Nan Jiang, Tong Zhang, Caiming Xiong, Hanze Dong  

**Link**: [PDF](https://arxiv.org/pdf/2504.11343)  

**Abstract**: Reinforcement learning (RL) has become a prevailing approach for fine-tuning large language models (LLMs) on complex reasoning tasks. Among recent methods, GRPO stands out for its empirical success in training models such as DeepSeek-R1, yet the sources of its effectiveness remain poorly understood. In this work, we revisit GRPO from a reinforce-like algorithm perspective and analyze its core components. Surprisingly, we find that a simple rejection sampling baseline, RAFT, which trains only on positively rewarded samples, yields competitive performance than GRPO and PPO. Our ablation studies reveal that GRPO's main advantage arises from discarding prompts with entirely incorrect responses, rather than from its reward normalization. Motivated by this insight, we propose Reinforce-Rej, a minimal extension of policy gradient that filters both entirely incorrect and entirely correct samples. Reinforce-Rej improves KL efficiency and stability, serving as a lightweight yet effective alternative to more complex RL algorithms. We advocate RAFT as a robust and interpretable baseline, and suggest that future advances should focus on more principled designs for incorporating negative samples, rather than relying on them indiscriminately. Our findings provide guidance for future work in reward-based LLM post-training. 

---
# Transformer-Based Model for Cold Start Mitigation in FaaS Architecture 

**Authors**: Alexandre Savi Fayam Mbala Mouen, Jerry Lacmou Zeutouo, Vianney Kengne Tchendji  

**Link**: [PDF](https://arxiv.org/pdf/2504.11338)  

**Abstract**: Serverless architectures, particularly the Function as a Service (FaaS) model, have become a cornerstone of modern cloud computing due to their ability to simplify resource management and enhance application deployment agility. However, a significant challenge remains: the cold start problem. This phenomenon occurs when an idle FaaS function is invoked, requiring a full initialization process, which increases latency and degrades user experience. Existing solutions for cold start mitigation are limited in terms of invocation pattern generalization and implementation complexity. In this study, we propose an innovative approach leveraging Transformer models to mitigate the impact of cold starts in FaaS architectures. Our solution excels in accurately modeling function initialization delays and optimizing serverless system performance. Experimental evaluation using a public dataset provided by Azure demonstrates a significant reduction in cold start times, reaching up to 79\% compared to conventional methods. 

---
# Looking beyond the next token 

**Authors**: Abitha Thankaraj, Yiding Jiang, J. Zico Kolter, Yonatan Bisk  

**Link**: [PDF](https://arxiv.org/pdf/2504.11336)  

**Abstract**: The structure of causal language model training assumes that each token can be accurately predicted from the previous context. This contrasts with humans' natural writing and reasoning process, where goals are typically known before the exact argument or phrasings. While this mismatch has been well studied in the literature, the working assumption has been that architectural changes are needed to address this mismatch. We argue that rearranging and processing the training data sequences can allow models to more accurately imitate the true data-generating process, and does not require any other changes to the architecture or training infrastructure. We demonstrate that this technique, Trelawney, and the inference algorithms derived from it allow us to improve performance on several key benchmarks that span planning, algorithmic reasoning, and story generation tasks. Finally, our method naturally enables the generation of long-term goals at no additional cost. We investigate how using the model's goal-generation capability can further improve planning and reasoning. Additionally, we believe Trelawney could potentially open doors to new capabilities beyond the current language modeling paradigm. 

---
# Code Reborn AI-Driven Legacy Systems Modernization from COBOL to Java 

**Authors**: Gopichand Bandarupalli  

**Link**: [PDF](https://arxiv.org/pdf/2504.11335)  

**Abstract**: This study investigates AI-driven modernization of legacy COBOL code into Java, addressing a critical challenge in aging software systems. Leveraging the Legacy COBOL 2024 Corpus -- 50,000 COBOL files from public and enterprise sources -- Java parses the code, AI suggests upgrades, and React visualizes gains. Achieving 93% accuracy, complexity drops 35% (from 18 to 11.7) and coupling 33% (from 8 to 5.4), surpassing manual efforts (75%) and rule-based tools (82%). The approach offers a scalable path to rejuvenate COBOL systems, vital for industries like banking and insurance. 

---
# Optimizing LLM Inference: Fluid-Guided Online Scheduling with Memory Constraints 

**Authors**: Ruicheng Ao, Gan Luo, David Simchi-Levi, Xinshang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.11320)  

**Abstract**: Large Language Models (LLMs) are indispensable in today's applications, but their inference procedure -- generating responses by processing text in segments and using a memory-heavy Key-Value (KV) cache -- demands significant computational resources, particularly under memory constraints. This paper formulates LLM inference optimization as a multi-stage online scheduling problem where sequential prompt arrivals and KV cache growth render conventional scheduling ineffective. We develop a fluid dynamics approximation to provide a tractable benchmark that guides algorithm design. Building on this, we propose the Waiting for Accumulated Inference Threshold (WAIT) algorithm, which uses multiple thresholds to schedule incoming prompts optimally when output lengths are known, and extend it to Nested WAIT for cases with unknown output lengths. Theoretical analysis shows that both algorithms achieve near-optimal performance against the fluid benchmark in heavy traffic conditions, balancing throughput, latency, and Time to First Token (TTFT). Experiments with the Llama-7B model on an A100 GPU using both synthetic and real-world datasets demonstrate improved throughput and reduced latency relative to established baselines like vLLM and Sarathi. This work bridges operations research and machine learning, offering a rigorous framework for the efficient deployment of LLMs under memory constraints. 

---
# CFIS-YOLO: A Lightweight Multi-Scale Fusion Network for Edge-Deployable Wood Defect Detection 

**Authors**: Jincheng Kang, Yi Cen, Yigang Cen, Ke Wang, Yuhan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.11305)  

**Abstract**: Wood defect detection is critical for ensuring quality control in the wood processing industry. However, current industrial applications face two major challenges: traditional methods are costly, subjective, and labor-intensive, while mainstream deep learning models often struggle to balance detection accuracy and computational efficiency for edge deployment. To address these issues, this study proposes CFIS-YOLO, a lightweight object detection model optimized for edge devices. The model introduces an enhanced C2f structure, a dynamic feature recombination module, and a novel loss function that incorporates auxiliary bounding boxes and angular constraints. These innovations improve multi-scale feature fusion and small object localization while significantly reducing computational overhead. Evaluated on a public wood defect dataset, CFIS-YOLO achieves a mean Average Precision (mAP@0.5) of 77.5\%, outperforming the baseline YOLOv10s by 4 percentage points. On SOPHON BM1684X edge devices, CFIS-YOLO delivers 135 FPS, reduces power consumption to 17.3\% of the original implementation, and incurs only a 0.5 percentage point drop in mAP. These results demonstrate that CFIS-YOLO is a practical and effective solution for real-world wood defect detection in resource-constrained environments. 

---
# Bipartite Ranking From Multiple Labels: On Loss Versus Label Aggregation 

**Authors**: Michal Lukasik, Lin Chen, Harikrishna Narasimhan, Aditya Krishna Menon, Wittawat Jitkrittum, Felix X. Yu, Sashank J. Reddi, Gang Fu, Mohammadhossein Bateni, Sanjiv Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2504.11284)  

**Abstract**: Bipartite ranking is a fundamental supervised learning problem, with the goal of learning a ranking over instances with maximal area under the ROC curve (AUC) against a single binary target label. However, one may often observe multiple binary target labels, e.g., from distinct human annotators. How can one synthesize such labels into a single coherent ranking? In this work, we formally analyze two approaches to this problem -- loss aggregation and label aggregation -- by characterizing their Bayes-optimal solutions. Based on this, we show that while both methods can yield Pareto-optimal solutions, loss aggregation can exhibit label dictatorship: one can inadvertently (and undesirably) favor one label over others. This suggests that label aggregation can be preferable to loss aggregation, which we empirically verify. 

---
# Single-Input Multi-Output Model Merging: Leveraging Foundation Models for Dense Multi-Task Learning 

**Authors**: Juan Garcia Giraldo, Nikolaos Dimitriadis, Ke Wang, Pascal Frossard  

**Link**: [PDF](https://arxiv.org/pdf/2504.11268)  

**Abstract**: Model merging is a flexible and computationally tractable approach to merge single-task checkpoints into a multi-task model. Prior work has solely focused on constrained multi-task settings where there is a one-to-one mapping between a sample and a task, overlooking the paradigm where multiple tasks may operate on the same sample, e.g., scene understanding. In this paper, we focus on the multi-task setting with single-input-multiple-outputs (SIMO) and show that it qualitatively differs from the single-input-single-output model merging settings studied in the literature due to the existence of task-specific decoders and diverse loss objectives. We identify that existing model merging methods lead to significant performance degradation, primarily due to representation misalignment between the merged encoder and task-specific decoders. We propose two simple and efficient fixes for the SIMO setting to re-align the feature representation after merging. Compared to joint fine-tuning, our approach is computationally effective and flexible, and sheds light into identifying task relationships in an offline manner. Experiments on NYUv2, Cityscapes, and a subset of the Taskonomy dataset demonstrate: (1) task arithmetic suffices to enable multi-task capabilities; however, the representations generated by the merged encoder has to be re-aligned with the task-specific heads; (2) the proposed architecture rivals traditional multi-task learning in performance but requires fewer samples and training steps by leveraging the existence of task-specific models. 

---
# DeepSelective: Feature Gating and Representation Matching for Interpretable Clinical Prediction 

**Authors**: Ruochi Zhang, Qian Yang, Xiaoyang Wang, Haoran Wu, Qiong Zhou, Yu Wang, Kewei Li, Yueying Wang, Yusi Fan, Jiale Zhang, Lan Huang, Chang Liu, Fengfeng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.11264)  

**Abstract**: The rapid accumulation of Electronic Health Records (EHRs) has transformed healthcare by providing valuable data that enhance clinical predictions and diagnoses. While conventional machine learning models have proven effective, they often lack robust representation learning and depend heavily on expert-crafted features. Although deep learning offers powerful solutions, it is often criticized for its lack of interpretability. To address these challenges, we propose DeepSelective, a novel end to end deep learning framework for predicting patient prognosis using EHR data, with a strong emphasis on enhancing model interpretability. DeepSelective combines data compression techniques with an innovative feature selection approach, integrating custom-designed modules that work together to improve both accuracy and interpretability. Our experiments demonstrate that DeepSelective not only enhances predictive accuracy but also significantly improves interpretability, making it a valuable tool for clinical decision-making. The source code is freely available at this http URL . 

---
# A Rollout-Based Algorithm and Reward Function for Efficient Resource Allocation in Business Processes 

**Authors**: Jeroen Middelhuis, Zaharah Bukhsh, Ivo Adan, Remco Dijkman  

**Link**: [PDF](https://arxiv.org/pdf/2504.11250)  

**Abstract**: Resource allocation plays a critical role in minimizing cycle time and improving the efficiency of business processes. Recently, Deep Reinforcement Learning (DRL) has emerged as a powerful tool to optimize resource allocation policies in business processes. In the DRL framework, an agent learns a policy through interaction with the environment, guided solely by reward signals that indicate the quality of its decisions. However, existing algorithms are not suitable for dynamic environments such as business processes. Furthermore, existing DRL-based methods rely on engineered reward functions that approximate the desired objective, but a misalignment between reward and objective can lead to undesired decisions or suboptimal policies. To address these issues, we propose a rollout-based DRL algorithm and a reward function to optimize the objective directly. Our algorithm iteratively improves the policy by evaluating execution trajectories following different actions. Our reward function directly decomposes the objective function of minimizing the mean cycle time. Maximizing our reward function guarantees that the objective function is minimized without requiring extensive reward engineering. The results show that our method consistently learns the optimal policy in all six evaluated business processes, outperforming the state-of-the-art algorithm that can only learn the optimal policy in two of the evaluated processes. 

---
# Respiratory Inhaler Sound Event Classification Using Self-Supervised Learning 

**Authors**: Davoud Shariat Panah, Alessandro N Franciosi, Cormac McCarthy, Andrew Hines  

**Link**: [PDF](https://arxiv.org/pdf/2504.11246)  

**Abstract**: Asthma is a chronic respiratory condition that affects millions of people worldwide. While this condition can be managed by administering controller medications through handheld inhalers, clinical studies have shown low adherence to the correct inhaler usage technique. Consequently, many patients may not receive the full benefit of their medication. Automated classification of inhaler sounds has recently been studied to assess medication adherence. However, the existing classification models were typically trained using data from specific inhaler types, and their ability to generalize to sounds from different inhalers remains unexplored. In this study, we adapted the wav2vec 2.0 self-supervised learning model for inhaler sound classification by pre-training and fine-tuning this model on inhaler sounds. The proposed model shows a balanced accuracy of 98% on a dataset collected using a dry powder inhaler and smartwatch device. The results also demonstrate that re-finetuning this model on minimal data from a target inhaler is a promising approach to adapting a generic inhaler sound classification model to a different inhaler device and audio capture hardware. This is the first study in the field to demonstrate the potential of smartwatches as assistive technologies for the personalized monitoring of inhaler adherence using machine learning models. 

---
# Influence Maximization in Temporal Social Networks with a Cold-Start Problem: A Supervised Approach 

**Authors**: Laixin Xie, Ying Zhang, Xiyuan Wang, Shiyi Liu, Shenghan Gao, Xingxing Xing, Wei Wan, Haipeng Zhang, Quan Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.11245)  

**Abstract**: Influence Maximization (IM) in temporal graphs focuses on identifying influential "seeds" that are pivotal for maximizing network expansion. We advocate defining these seeds through Influence Propagation Paths (IPPs), which is essential for scaling up the network. Our focus lies in efficiently labeling IPPs and accurately predicting these seeds, while addressing the often-overlooked cold-start issue prevalent in temporal networks. Our strategy introduces a motif-based labeling method and a tensorized Temporal Graph Network (TGN) tailored for multi-relational temporal graphs, bolstering prediction accuracy and computational efficiency. Moreover, we augment cold-start nodes with new neighbors from historical data sharing similar IPPs. The recommendation system within an online team-based gaming environment presents subtle impact on the social network, forming multi-relational (i.e., weak and strong) temporal graphs for our empirical IM study. We conduct offline experiments to assess prediction accuracy and model training efficiency, complemented by online A/B testing to validate practical network growth and the effectiveness in addressing the cold-start issue. 

---
# Diversity-Driven Learning: Tackling Spurious Correlations and Data Heterogeneity in Federated Models 

**Authors**: Gergely D. Németh, Eros Fanì, Yeat Jeng Ng, Barbara Caputo, Miguel Ángel Lozano, Nuria Oliver, Novi Quadrianto  

**Link**: [PDF](https://arxiv.org/pdf/2504.11216)  

**Abstract**: Federated Learning (FL) enables decentralized training of machine learning models on distributed data while preserving privacy. However, in real-world FL settings, client data is often non-identically distributed and imbalanced, resulting in statistical data heterogeneity which impacts the generalization capabilities of the server's model across clients, slows convergence and reduces performance. In this paper, we address this challenge by first proposing a characterization of statistical data heterogeneity by means of 6 metrics of global and client attribute imbalance, class imbalance, and spurious correlations. Next, we create and share 7 computer vision datasets for binary and multiclass image classification tasks in Federated Learning that cover a broad range of statistical data heterogeneity and hence simulate real-world situations. Finally, we propose FedDiverse, a novel client selection algorithm in FL which is designed to manage and leverage data heterogeneity across clients by promoting collaboration between clients with complementary data distributions. Experiments on the seven proposed FL datasets demonstrate FedDiverse's effectiveness in enhancing the performance and robustness of a variety of FL methods while having low communication and computational overhead. 

---
# Efficient Distributed Retrieval-Augmented Generation for Enhancing Language Model Performance 

**Authors**: Shangyu Liu, Zhenzhe Zheng, Xiaoyao Huang, Fan Wu, Jie Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.11197)  

**Abstract**: Small language models (SLMs) support efficient deployments on resource-constrained edge devices, but their limited capacity compromises inference performance. Retrieval-augmented generation (RAG) is a promising solution to enhance model performance by integrating external databases, without requiring intensive on-device model retraining. However, large-scale public databases and user-specific private contextual documents are typically located on the cloud and the device separately, while existing RAG implementations are primarily centralized. To bridge this gap, we propose DRAGON, a distributed RAG framework to enhance on-device SLMs through both general and personal knowledge without the risk of leaking document privacy. Specifically, DRAGON decomposes multi-document RAG into multiple parallel token generation processes performed independently and locally on the cloud and the device, and employs a newly designed Speculative Aggregation, a dual-side speculative algorithm to avoid frequent output synchronization between the cloud and device. A new scheduling algorithm is further introduced to identify the optimal aggregation side based on real-time network conditions. Evaluations on real-world hardware testbed demonstrate a significant performance improvement of DRAGON-up to 1.9x greater gains over standalone SLM compared to the centralized RAG, substantial reduction in per-token latency, and negligible Time to First Token (TTFT) overhead. 

---
# Benchmarking Next-Generation Reasoning-Focused Large Language Models in Ophthalmology: A Head-to-Head Evaluation on 5,888 Items 

**Authors**: Minjie Zou, Sahana Srinivasan, Thaddaeus Wai Soon Lo, Ke Zou, Gabriel Dawei Yang, Xuguang Ai, Hyunjae Kim, Maxwell Singer, Fares Antaki, Kelvin Li, Robert Chang, Marcus Tan, David Ziyou Chen, Dianbo Liu, Qingyu Chen, Yih Chung Tham  

**Link**: [PDF](https://arxiv.org/pdf/2504.11186)  

**Abstract**: Recent advances in reasoning-focused large language models (LLMs) mark a shift from general LLMs toward models designed for complex decision-making, a crucial aspect in medicine. However, their performance in specialized domains like ophthalmology remains underexplored. This study comprehensively evaluated and compared the accuracy and reasoning capabilities of four newly developed reasoning-focused LLMs, namely DeepSeek-R1, OpenAI o1, o3-mini, and Gemini 2.0 Flash-Thinking. Each model was assessed using 5,888 multiple-choice ophthalmology exam questions from the MedMCQA dataset in zero-shot setting. Quantitative evaluation included accuracy, Macro-F1, and five text-generation metrics (ROUGE-L, METEOR, BERTScore, BARTScore, and AlignScore), computed against ground-truth reasonings. Average inference time was recorded for a subset of 100 randomly selected questions. Additionally, two board-certified ophthalmologists qualitatively assessed clarity, completeness, and reasoning structure of responses to differential diagnosis questions.O1 (0.902) and DeepSeek-R1 (0.888) achieved the highest accuracy, with o1 also leading in Macro-F1 (0.900). The performance of models across the text-generation metrics varied: O3-mini excelled in ROUGE-L (0.151), o1 in METEOR (0.232), DeepSeek-R1 and o3-mini tied for BERTScore (0.673), DeepSeek-R1 (-4.105) and Gemini 2.0 Flash-Thinking (-4.127) performed best in BARTScore, while o3-mini (0.181) and o1 (0.176) led AlignScore. Inference time across the models varied, with DeepSeek-R1 being slowest (40.4 seconds) and Gemini 2.0 Flash-Thinking fastest (6.7 seconds). Qualitative evaluation revealed that DeepSeek-R1 and Gemini 2.0 Flash-Thinking tended to provide detailed and comprehensive intermediate reasoning, whereas o1 and o3-mini displayed concise and summarized justifications. 

---
# Exploring Backdoor Attack and Defense for LLM-empowered Recommendations 

**Authors**: Liangbo Ning, Wenqi Fan, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.11182)  

**Abstract**: The fusion of Large Language Models (LLMs) with recommender systems (RecSys) has dramatically advanced personalized recommendations and drawn extensive attention. Despite the impressive progress, the safety of LLM-based RecSys against backdoor attacks remains largely under-explored. In this paper, we raise a new problem: Can a backdoor with a specific trigger be injected into LLM-based Recsys, leading to the manipulation of the recommendation responses when the backdoor trigger is appended to an item's title? To investigate the vulnerabilities of LLM-based RecSys under backdoor attacks, we propose a new attack framework termed Backdoor Injection Poisoning for RecSys (BadRec). BadRec perturbs the items' titles with triggers and employs several fake users to interact with these items, effectively poisoning the training set and injecting backdoors into LLM-based RecSys. Comprehensive experiments reveal that poisoning just 1% of the training data with adversarial examples is sufficient to successfully implant backdoors, enabling manipulation of recommendations. To further mitigate such a security threat, we propose a universal defense strategy called Poison Scanner (P-Scanner). Specifically, we introduce an LLM-based poison scanner to detect the poisoned items by leveraging the powerful language understanding and rich knowledge of LLMs. A trigger augmentation agent is employed to generate diverse synthetic triggers to guide the poison scanner in learning domain-specific knowledge of the poisoned item detection task. Extensive experiments on three real-world datasets validate the effectiveness of the proposed P-Scanner. 

---
# TerraMind: Large-Scale Generative Multimodality for Earth Observation 

**Authors**: Johannes Jakubik, Felix Yang, Benedikt Blumenstiel, Erik Scheurer, Rocco Sedona, Stefano Maurogiovanni, Jente Bosmans, Nikolaos Dionelis, Valerio Marsocci, Niklas Kopp, Rahul Ramachandran, Paolo Fraccaro, Thomas Brunschwiler, Gabriele Cavallaro, Juan Bernabe-Moreno, Nicolas Longépé  

**Link**: [PDF](https://arxiv.org/pdf/2504.11171)  

**Abstract**: We present TerraMind, the first any-to-any generative, multimodal foundation model for Earth observation (EO). Unlike other multimodal models, TerraMind is pretrained on dual-scale representations combining both token-level and pixel-level data across modalities. On a token level, TerraMind encodes high-level contextual information to learn cross-modal relationships, while on a pixel level, TerraMind leverages fine-grained representations to capture critical spatial nuances. We pretrained TerraMind on nine geospatial modalities of a global, large-scale dataset. In this paper, we demonstrate that (i) TerraMind's dual-scale early fusion approach unlocks a range of zero-shot and few-shot applications for Earth observation, (ii) TerraMind introduces "Thinking-in-Modalities" (TiM) -- the capability of generating additional artificial data during finetuning and inference to improve the model output -- and (iii) TerraMind achieves beyond state-of-the-art performance in community-standard benchmarks for EO like PANGAEA. The pretraining dataset, the model weights, and our code is open-sourced under a permissive license. 

---
# MuSeD: A Multimodal Spanish Dataset for Sexism Detection in Social Media Videos 

**Authors**: Laura De Grazia, Pol Pastells, Mauro Vázquez Chas, Desmond Elliott, Danae Sánchez Villegas, Mireia Farrús, Mariona Taulé  

**Link**: [PDF](https://arxiv.org/pdf/2504.11169)  

**Abstract**: Sexism is generally defined as prejudice and discrimination based on sex or gender, affecting every sector of society, from social institutions to relationships and individual behavior. Social media platforms amplify the impact of sexism by conveying discriminatory content not only through text but also across multiple modalities, highlighting the critical need for a multimodal approach to the analysis of sexism online. With the rise of social media platforms where users share short videos, sexism is increasingly spreading through video content. Automatically detecting sexism in videos is a challenging task, as it requires analyzing the combination of verbal, audio, and visual elements to identify sexist content. In this study, (1) we introduce MuSeD, a new Multimodal Spanish dataset for Sexism Detection consisting of $\approx$ 11 hours of videos extracted from TikTok and BitChute; (2) we propose an innovative annotation framework for analyzing the contribution of textual and multimodal labels in the classification of sexist and non-sexist content; and (3) we evaluate a range of large language models (LLMs) and multimodal LLMs on the task of sexism detection. We find that visual information plays a key role in labeling sexist content for both humans and models. Models effectively detect explicit sexism; however, they struggle with implicit cases, such as stereotypes, instances where annotators also show low agreement. This highlights the inherent difficulty of the task, as identifying implicit sexism depends on the social and cultural context. 

---
# Bypassing Prompt Injection and Jailbreak Detection in LLM Guardrails 

**Authors**: William Hackett, Lewis Birch, Stefan Trawicki, Neeraj Suri, Peter Garraghan  

**Link**: [PDF](https://arxiv.org/pdf/2504.11168)  

**Abstract**: Large Language Models (LLMs) guardrail systems are designed to protect against prompt injection and jailbreak attacks. However, they remain vulnerable to evasion techniques. We demonstrate two approaches for bypassing LLM prompt injection and jailbreak detection systems via traditional character injection methods and algorithmic Adversarial Machine Learning (AML) evasion techniques. Through testing against six prominent protection systems, including Microsoft's Azure Prompt Shield and Meta's Prompt Guard, we show that both methods can be used to evade detection while maintaining adversarial utility achieving in some instances up to 100% evasion success. Furthermore, we demonstrate that adversaries can enhance Attack Success Rates (ASR) against black-box targets by leveraging word importance ranking computed by offline white-box models. Our findings reveal vulnerabilities within current LLM protection mechanisms and highlight the need for more robust guardrail systems. 

---
# DMAGaze: Gaze Estimation Based on Feature Disentanglement and Multi-Scale Attention 

**Authors**: Haohan Chen, Hongjia Liu, Shiyong Lan, Wenwu Wang, Yixin Qiao, Yao Li, Guonan Deng  

**Link**: [PDF](https://arxiv.org/pdf/2504.11160)  

**Abstract**: Gaze estimation, which predicts gaze direction, commonly faces the challenge of interference from complex gaze-irrelevant information in face images. In this work, we propose DMAGaze, a novel gaze estimation framework that exploits information from facial images in three aspects: gaze-relevant global features (disentangled from facial image), local eye features (extracted from cropped eye patch), and head pose estimation features, to improve overall performance. Firstly, we design a new continuous mask-based Disentangler to accurately disentangle gaze-relevant and gaze-irrelevant information in facial images by achieving the dual-branch disentanglement goal through separately reconstructing the eye and non-eye regions. Furthermore, we introduce a new cascaded attention module named Multi-Scale Global Local Attention Module (MS-GLAM). Through a customized cascaded attention structure, it effectively focuses on global and local information at multiple scales, further enhancing the information from the Disentangler. Finally, the global gaze-relevant features disentangled by the upper face branch, combined with head pose and local eye features, are passed through the detection head for high-precision gaze estimation. Our proposed DMAGaze has been extensively validated on two mainstream public datasets, achieving state-of-the-art performance. 

---
# Divergence of Empirical Neural Tangent Kernel in Classification Problems 

**Authors**: Zixiong Yu, Songtao Tian, Guhan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.11130)  

**Abstract**: This paper demonstrates that in classification problems, fully connected neural networks (FCNs) and residual neural networks (ResNets) cannot be approximated by kernel logistic regression based on the Neural Tangent Kernel (NTK) under overtraining (i.e., when training time approaches infinity). Specifically, when using the cross-entropy loss, regardless of how large the network width is (as long as it is finite), the empirical NTK diverges from the NTK on the training samples as training time increases. To establish this result, we first demonstrate the strictly positive definiteness of the NTKs for multi-layer FCNs and ResNets. Then, we prove that during training, % with the cross-entropy loss, the neural network parameters diverge if the smallest eigenvalue of the empirical NTK matrix (Gram matrix) with respect to training samples is bounded below by a positive constant. This behavior contrasts sharply with the lazy training regime commonly observed in regression problems. Consequently, using a proof by contradiction, we show that the empirical NTK does not uniformly converge to the NTK across all times on the training samples as the network width increases. We validate our theoretical results through experiments on both synthetic data and the MNIST classification task. This finding implies that NTK theory is not applicable in this context, with significant theoretical implications for understanding neural networks in classification problems. 

---
# Fine-Tuning Large Language Models on Quantum Optimization Problems for Circuit Generation 

**Authors**: Linus Jern, Valter Uotila, Cong Yu, Bo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.11109)  

**Abstract**: Large language models (LLM) have achieved remarkable outcomes in addressing complex problems, including math, coding, and analyzing large amounts of scientific reports. Yet few works have explored the potential of LLM in quantum computing. The most challenging problem is how to leverage LLMs to automatically generate quantum circuits at a large scale. In this paper, we address such a challenge by fine-tuning LLMs and injecting the domain-specific knowledge of quantum computing. In particular, we investigate the mechanisms to generate training data sets and construct the end-to-end pipeline to fine-tune pre-trained LLMs that produce parameterized quantum circuits for optimization problems. We have prepared 14,000 quantum circuits covering a substantial part of the quantum optimization landscape: 12 optimization problem instances and their optimized QAOA, VQE, and adaptive VQE circuits. The fine-tuned LLMs can construct syntactically correct parametrized quantum circuits in the most recent OpenQASM 3.0. We have evaluated the quality of the parameters by comparing them to the optimized expectation values and distributions. Our evaluation shows that the fine-tuned LLM outperforms state-of-the-art models and that the parameters are better than random. The LLM-generated parametrized circuits and initial parameters can be used as a starting point for further optimization, \emph{e.g.,} templates in quantum machine learning and the benchmark for compilers and hardware. 

---
# AI-guided Antibiotic Discovery Pipeline from Target Selection to Compound Identification 

**Authors**: Maximilian G. Schuh, Joshua Hesse, Stephan A. Sieber  

**Link**: [PDF](https://arxiv.org/pdf/2504.11091)  

**Abstract**: Antibiotic resistance presents a growing global health crisis, demanding new therapeutic strategies that target novel bacterial mechanisms. Recent advances in protein structure prediction and machine learning-driven molecule generation offer a promising opportunity to accelerate drug discovery. However, practical guidance on selecting and integrating these models into real-world pipelines remains limited. In this study, we develop an end-to-end, artificial intelligence-guided antibiotic discovery pipeline that spans target identification to compound realization. We leverage structure-based clustering across predicted proteomes of multiple pathogens to identify conserved, essential, and non-human-homologous targets. We then systematically evaluate six leading 3D-structure-aware generative models$\unicode{x2014}$spanning diffusion, autoregressive, graph neural network, and language model architectures$\unicode{x2014}$on their usability, chemical validity, and biological relevance. Rigorous post-processing filters and commercial analogue searches reduce over 100 000 generated compounds to a focused, synthesizable set. Our results highlight DeepBlock and TamGen as top performers across diverse criteria, while also revealing critical trade-offs between model complexity, usability, and output quality. This work provides a comparative benchmark and blueprint for deploying artificial intelligence in early-stage antibiotic development. 

---
# QAMA: Quantum annealing multi-head attention operator with classical deep learning framework 

**Authors**: Peng Du, Shuolei Wang, Shicheng Li, Jinjing Shi  

**Link**: [PDF](https://arxiv.org/pdf/2504.11083)  

**Abstract**: As large language models scale up, the conventional attention mechanism faces critical challenges of exponential growth in memory consumption and energy costs. Quantum annealing computing, with its inherent advantages in computational efficiency and low energy consumption, offers an innovative direction for constructing novel deep learning architectures. This study proposes the first Quantum Annealing-based Multi-head Attention (QAMA) mechanism, achieving seamless compatibility with classical attention architectures through quadratic unconstrained binary optimization (QUBO) modeling of forward propagation and energy-based backpropagation. The method innovatively leverages the quantum bit interaction characteristics of Ising models to optimize the conventional $O(n^2)$ spatiotemporal complexity into linear resource consumption. Integrated with the optical computing advantages of coherent Ising machines (CIM), the system maintains millisecond-level real-time responsiveness while significantly reducing energy consumption. Our key contributions include: Theoretical proofs establish QAMA mathematical equivalence to classical attention mechanisms; Dual optimization of multi-head specificity and long-range information capture via QUBO constraints; Explicit gradient proofs for the Ising energy equation are utilized to implement gradient conduction as the only path in the computational graph as a layer; Proposed soft selection mechanism overcoming traditional binary attention limitations to approximate continuous weights. Experiments on QBoson CPQC quantum computer show QAMA achieves comparable accuracy to classical operators while reducing inference time to millisecond level and improving solution quality. This work pioneers architectural-level integration of quantum computing and deep learning, applicable to any attention-based model, driving paradigm innovation in AI foundational computing. 

---
# DeepMLF: Multimodal language model with learnable tokens for deep fusion in sentiment analysis 

**Authors**: Efthymios Georgiou, Vassilis Katsouros, Yannis Avrithis, Alexandros Potamianos  

**Link**: [PDF](https://arxiv.org/pdf/2504.11082)  

**Abstract**: While multimodal fusion has been extensively studied in Multimodal Sentiment Analysis (MSA), the role of fusion depth and multimodal capacity allocation remains underexplored. In this work, we position fusion depth, scalability, and dedicated multimodal capacity as primary factors for effective fusion. We introduce DeepMLF, a novel multimodal language model (LM) with learnable tokens tailored toward deep fusion. DeepMLF leverages an audiovisual encoder and a pretrained decoder LM augmented with multimodal information across its layers. We append learnable tokens to the LM that: 1) capture modality interactions in a controlled fashion and 2) preserve independent information flow for each modality. These fusion tokens gather linguistic information via causal self-attention in LM Blocks and integrate with audiovisual information through cross-attention MM Blocks. Serving as dedicated multimodal capacity, this design enables progressive fusion across multiple layers, providing depth in the fusion process. Our training recipe combines modality-specific losses and language modelling loss, with the decoder LM tasked to predict ground truth polarity. Across three MSA benchmarks with varying dataset characteristics, DeepMLF achieves state-of-the-art performance. Our results confirm that deeper fusion leads to better performance, with optimal fusion depths (5-7) exceeding those of existing approaches. Additionally, our analysis on the number of fusion tokens reveals that small token sets ($\sim$20) achieve optimal performance. We examine the importance of representation learning order (fusion curriculum) through audiovisual encoder initialization experiments. Our ablation studies demonstrate the superiority of the proposed fusion design and gating while providing a holistic examination of DeepMLF's scalability to LLMs, and the impact of each training objective and embedding regularization. 

---
# Dynamical errors in machine learning forecasts 

**Authors**: Zhou Fang, Gianmarco Mengaldo  

**Link**: [PDF](https://arxiv.org/pdf/2504.11074)  

**Abstract**: In machine learning forecasting, standard error metrics such as mean absolute error (MAE) and mean squared error (MSE) quantify discrepancies between predictions and target values. However, these metrics do not directly evaluate the physical and/or dynamical consistency of forecasts, an increasingly critical concern in scientific and engineering applications.
Indeed, a fundamental yet often overlooked question is whether machine learning forecasts preserve the dynamical behavior of the underlying system. Addressing this issue is essential for assessing the fidelity of machine learning models and identifying potential failure modes, particularly in applications where maintaining correct dynamical behavior is crucial.
In this work, we investigate the relationship between standard forecasting error metrics, such as MAE and MSE, and the dynamical properties of the underlying system. To achieve this goal, we use two recently developed dynamical indices: the instantaneous dimension ($d$), and the inverse persistence ($\theta$). Our results indicate that larger forecast errors -- e.g., higher MSE -- tend to occur in states with higher $d$ (higher complexity) and higher $\theta$ (lower persistence). To further assess dynamical consistency, we propose error metrics based on the dynamical indices that measure the discrepancy of the forecasted $d$ and $\theta$ versus their correct values. Leveraging these dynamical indices-based metrics, we analyze direct and recursive forecasting strategies for three canonical datasets -- Lorenz, Kuramoto-Sivashinsky equation, and Kolmogorov flow -- as well as a real-world weather forecasting task. Our findings reveal substantial distortions in dynamical properties in ML forecasts, especially for long forecast lead times or long recursive simulations, providing complementary information on ML forecast fidelity that can be used to improve ML models. 

---
# Neural Control Barrier Functions from Physics Informed Neural Networks 

**Authors**: Shreenabh Agrawal, Manan Tayal, Aditya Singh, Shishir Kolathaya  

**Link**: [PDF](https://arxiv.org/pdf/2504.11045)  

**Abstract**: As autonomous systems become increasingly prevalent in daily life, ensuring their safety is paramount. Control Barrier Functions (CBFs) have emerged as an effective tool for guaranteeing safety; however, manually designing them for specific applications remains a significant challenge. With the advent of deep learning techniques, recent research has explored synthesizing CBFs using neural networks-commonly referred to as neural CBFs. This paper introduces a novel class of neural CBFs that leverages a physics-inspired neural network framework by incorporating Zubov's Partial Differential Equation (PDE) within the context of safety. This approach provides a scalable methodology for synthesizing neural CBFs applicable to high-dimensional systems. Furthermore, by utilizing reciprocal CBFs instead of zeroing CBFs, the proposed framework allows for the specification of flexible, user-defined safe regions. To validate the effectiveness of the approach, we present case studies on three different systems: an inverted pendulum, autonomous ground navigation, and aerial navigation in obstacle-laden environments. 

---
# QAVA: Query-Agnostic Visual Attack to Large Vision-Language Models 

**Authors**: Yudong Zhang, Ruobing Xie, Jiansheng Chen, Xingwu Sun, Zhanhui Kang, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.11038)  

**Abstract**: In typical multimodal tasks, such as Visual Question Answering (VQA), adversarial attacks targeting a specific image and question can lead large vision-language models (LVLMs) to provide incorrect answers. However, it is common for a single image to be associated with multiple questions, and LVLMs may still answer other questions correctly even for an adversarial image attacked by a specific question. To address this, we introduce the query-agnostic visual attack (QAVA), which aims to create robust adversarial examples that generate incorrect responses to unspecified and unknown questions. Compared to traditional adversarial attacks focused on specific images and questions, QAVA significantly enhances the effectiveness and efficiency of attacks on images when the question is unknown, achieving performance comparable to attacks on known target questions. Our research broadens the scope of visual adversarial attacks on LVLMs in practical settings, uncovering previously overlooked vulnerabilities, particularly in the context of visual adversarial threats. The code is available at this https URL. 

---
# "Even explanations will not help in trusting [this] fundamentally biased system": A Predictive Policing Case-Study 

**Authors**: Siddharth Mehrotra, Ujwal Gadiraju, Eva Bittner, Folkert van Delden, Catholijn M. Jonker, Myrthe L. Tielman  

**Link**: [PDF](https://arxiv.org/pdf/2504.11020)  

**Abstract**: In today's society, where Artificial Intelligence (AI) has gained a vital role, concerns regarding user's trust have garnered significant attention. The use of AI systems in high-risk domains have often led users to either under-trust it, potentially causing inadequate reliance or over-trust it, resulting in over-compliance. Therefore, users must maintain an appropriate level of trust. Past research has indicated that explanations provided by AI systems can enhance user understanding of when to trust or not trust the system. However, the utility of presentation of different explanations forms still remains to be explored especially in high-risk domains. Therefore, this study explores the impact of different explanation types (text, visual, and hybrid) and user expertise (retired police officers and lay users) on establishing appropriate trust in AI-based predictive policing. While we observed that the hybrid form of explanations increased the subjective trust in AI for expert users, it did not led to better decision-making. Furthermore, no form of explanations helped build appropriate trust. The findings of our study emphasize the importance of re-evaluating the use of explanations to build [appropriate] trust in AI based systems especially when the system's use is questionable. Finally, we synthesize potential challenges and policy recommendations based on our results to design for appropriate trust in high-risk based AI-based systems. 

---
# GATE3D: Generalized Attention-based Task-synergized Estimation in 3D* 

**Authors**: Eunsoo Im, Jung Kwon Lee, Changhyun Jee  

**Link**: [PDF](https://arxiv.org/pdf/2504.11014)  

**Abstract**: The emerging trend in computer vision emphasizes developing universal models capable of simultaneously addressing multiple diverse tasks. Such universality typically requires joint training across multi-domain datasets to ensure effective generalization. However, monocular 3D object detection presents unique challenges in multi-domain training due to the scarcity of datasets annotated with accurate 3D ground-truth labels, especially beyond typical road-based autonomous driving contexts. To address this challenge, we introduce a novel weakly supervised framework leveraging pseudo-labels. Current pretrained models often struggle to accurately detect pedestrians in non-road environments due to inherent dataset biases. Unlike generalized image-based 2D object detection models, achieving similar generalization in monocular 3D detection remains largely unexplored. In this paper, we propose GATE3D, a novel framework designed specifically for generalized monocular 3D object detection via weak supervision. GATE3D effectively bridges domain gaps by employing consistency losses between 2D and 3D predictions. Remarkably, our model achieves competitive performance on the KITTI benchmark as well as on an indoor-office dataset collected by us to evaluate the generalization capabilities of our framework. Our results demonstrate that GATE3D significantly accelerates learning from limited annotated data through effective pre-training strategies, highlighting substantial potential for broader impacts in robotics, augmented reality, and virtual reality applications. Project page: this https URL 

---
# Document Quality Scoring for Web Crawling 

**Authors**: Francesca Pezzuti, Ariane Mueller, Sean MacAvaney, Nicola Tonellotto  

**Link**: [PDF](https://arxiv.org/pdf/2504.11011)  

**Abstract**: The internet contains large amounts of low-quality content, yet users expect web search engines to deliver high-quality, relevant results. The abundant presence of low-quality pages can negatively impact retrieval and crawling processes by wasting resources on these documents. Therefore, search engines can greatly benefit from techniques that leverage efficient quality estimation methods to mitigate these negative impacts. Quality scoring methods for web pages are useful for many processes typical for web search systems, including static index pruning, index tiering, and crawling. Building on work by Chang et al.~\cite{chang2024neural}, who proposed using neural estimators of semantic quality for static index pruning, we extend their approach and apply their neural quality scorers to assess the semantic quality of web pages in crawling prioritisation tasks. In our experimental analysis, we found that prioritising semantically high-quality pages over low-quality ones can improve downstream search effectiveness. Our software contribution consists of a Docker container that computes an effective quality score for a given web page, allowing the quality scorer to be easily included and used in other components of web search systems. 

---
# MediSee: Reasoning-based Pixel-level Perception in Medical Images 

**Authors**: Qinyue Tong, Ziqian Lu, Jun Liu, Yangming Zheng, Zheming Lu  

**Link**: [PDF](https://arxiv.org/pdf/2504.11008)  

**Abstract**: Despite remarkable advancements in pixel-level medical image perception, existing methods are either limited to specific tasks or heavily rely on accurate bounding boxes or text labels as input prompts. However, the medical knowledge required for input is a huge obstacle for general public, which greatly reduces the universality of these methods. Compared with these domain-specialized auxiliary information, general users tend to rely on oral queries that require logical reasoning. In this paper, we introduce a novel medical vision task: Medical Reasoning Segmentation and Detection (MedSD), which aims to comprehend implicit queries about medical images and generate the corresponding segmentation mask and bounding box for the target object. To accomplish this task, we first introduce a Multi-perspective, Logic-driven Medical Reasoning Segmentation and Detection (MLMR-SD) dataset, which encompasses a substantial collection of medical entity targets along with their corresponding reasoning. Furthermore, we propose MediSee, an effective baseline model designed for medical reasoning segmentation and detection. The experimental results indicate that the proposed method can effectively address MedSD with implicit colloquial queries and outperform traditional medical referring segmentation methods. 

---
# Dynamic Compressing Prompts for Efficient Inference of Large Language Models 

**Authors**: Jinwu Hu, Wei Zhang, Yufeng Wang, Yu Hu, Bin Xiao, Mingkui Tan, Qing Du  

**Link**: [PDF](https://arxiv.org/pdf/2504.11004)  

**Abstract**: Large Language Models (LLMs) have shown outstanding performance across a variety of tasks, partly due to advanced prompting techniques. However, these techniques often require lengthy prompts, which increase computational costs and can hinder performance because of the limited context windows of LLMs. While prompt compression is a straightforward solution, existing methods confront the challenges of retaining essential information, adapting to context changes, and remaining effective across different tasks. To tackle these issues, we propose a task-agnostic method called Dynamic Compressing Prompts (LLM-DCP). Our method reduces the number of prompt tokens while aiming to preserve the performance as much as possible. We model prompt compression as a Markov Decision Process (MDP), enabling the DCP-Agent to sequentially remove redundant tokens by adapting to dynamic contexts and retaining crucial content. We develop a reward function for training the DCP-Agent that balances the compression rate, the quality of the LLM output, and the retention of key information. This allows for prompt token reduction without needing an external black-box LLM. Inspired by the progressive difficulty adjustment in curriculum learning, we introduce a Hierarchical Prompt Compression (HPC) training strategy that gradually increases the compression difficulty, enabling the DCP-Agent to learn an effective compression method that maintains information integrity. Experiments demonstrate that our method outperforms state-of-the-art techniques, especially at higher compression rates. The code for our approach will be available at this https URL. 

---
# TMCIR: Token Merge Benefits Composed Image Retrieval 

**Authors**: Chaoyang Wang, Zeyu Zhang, Long Teng, Zijun Li, Shichao Kan  

**Link**: [PDF](https://arxiv.org/pdf/2504.10995)  

**Abstract**: Composed Image Retrieval (CIR) retrieves target images using a multi-modal query that combines a reference image with text describing desired modifications. The primary challenge is effectively fusing this visual and textual information. Current cross-modal feature fusion approaches for CIR exhibit an inherent bias in intention interpretation. These methods tend to disproportionately emphasize either the reference image features (visual-dominant fusion) or the textual modification intent (text-dominant fusion through image-to-text conversion). Such an imbalanced representation often fails to accurately capture and reflect the actual search intent of the user in the retrieval results. To address this challenge, we propose TMCIR, a novel framework that advances composed image retrieval through two key innovations: 1) Intent-Aware Cross-Modal Alignment. We first fine-tune CLIP encoders contrastively using intent-reflecting pseudo-target images, synthesized from reference images and textual descriptions via a diffusion model. This step enhances the encoder ability of text to capture nuanced intents in textual descriptions. 2) Adaptive Token Fusion. We further fine-tune all encoders contrastively by comparing adaptive token-fusion features with the target image. This mechanism dynamically balances visual and textual representations within the contrastive learning pipeline, optimizing the composed feature for retrieval. Extensive experiments on Fashion-IQ and CIRR datasets demonstrate that TMCIR significantly outperforms state-of-the-art methods, particularly in capturing nuanced user intent. 

---
# ProtFlow: Fast Protein Sequence Design via Flow Matching on Compressed Protein Language Model Embeddings 

**Authors**: Zitai Kong, Yiheng Zhu, Yinlong Xu, Hanjing Zhou, Mingzhe Yin, Jialu Wu, Hongxia Xu, Chang-Yu Hsieh, Tingjun Hou, Jian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.10983)  

**Abstract**: The design of protein sequences with desired functionalities is a fundamental task in protein engineering. Deep generative methods, such as autoregressive models and diffusion models, have greatly accelerated the discovery of novel protein sequences. However, these methods mainly focus on local or shallow residual semantics and suffer from low inference efficiency, large modeling space and high training cost. To address these challenges, we introduce ProtFlow, a fast flow matching-based protein sequence design framework that operates on embeddings derived from semantically meaningful latent space of protein language models. By compressing and smoothing the latent space, ProtFlow enhances performance while training on limited computational resources. Leveraging reflow techniques, ProtFlow enables high-quality single-step sequence generation. Additionally, we develop a joint design pipeline for the design scene of multichain proteins. We evaluate ProtFlow across diverse protein design tasks, including general peptides and long-chain proteins, antimicrobial peptides, and antibodies. Experimental results demonstrate that ProtFlow outperforms task-specific methods in these applications, underscoring its potential and broad applicability in computational protein sequence design and analysis. 

---
# Exploring the Role of KG-Based RAG in Japanese Medical Question Answering with Small-Scale LLMs 

**Authors**: Yingjian Chen, Feiyang Li, Xingyu Song, Tianxiao Li, Issey Sudeka, Irene Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.10982)  

**Abstract**: Large language models (LLMs) perform well in medical QA, but their effectiveness in Japanese contexts is limited due to privacy constraints that prevent the use of commercial models like GPT-4 in clinical settings. As a result, recent efforts focus on instruction-tuning open-source LLMs, though the potential of combining them with retrieval-augmented generation (RAG) remains underexplored. To bridge this gap, we are the first to explore a knowledge graph-based (KG) RAG framework for Japanese medical QA small-scale open-source LLMs. Experimental results show that KG-based RAG has only a limited impact on Japanese medical QA using small-scale open-source LLMs. Further case studies reveal that the effectiveness of the RAG is sensitive to the quality and relevance of the external retrieved content. These findings offer valuable insights into the challenges and potential of applying RAG in Japanese medical QA, while also serving as a reference for other low-resource languages. 

---
# Evaluating Trust in AI, Human, and Co-produced Feedback Among Undergraduate Students 

**Authors**: Audrey Zhang, Yifei Gao, Wannapon Suraworachet, Tanya Nazaretsky, Mutlu Cukurova  

**Link**: [PDF](https://arxiv.org/pdf/2504.10961)  

**Abstract**: As generative AI transforms educational feedback practices, understanding students' perceptions of different feedback providers becomes crucial for effective implementation. This study addresses a critical gap by comparing undergraduate students' trust in AI-generated, human-created, and human-AI co-produced feedback, informing how institutions can adapt feedback practices in this new era. Through a within-subject experiment with 91 participants, we investigated factors predicting students' ability to distinguish between feedback types, perception of feedback quality, and potential biases to AI involvement. Findings revealed that students generally preferred AI and co-produced feedback over human feedback in terms of perceived usefulness and objectivity. Only AI feedback suffered a decline in perceived genuineness when feedback sources were revealed, while co-produced feedback maintained its positive perception. Educational AI experience improved students' ability to identify AI feedback and increased their trust in all feedback types, while general AI experience decreased perceived usefulness and credibility. Male students consistently rated all feedback types as less valuable than their female and non-binary counterparts. These insights inform evidence-based guidelines for integrating AI into higher education feedback systems while addressing trust concerns and fostering AI literacy among students. 

---
# BEACON: A Benchmark for Efficient and Accurate Counting of Subgraphs 

**Authors**: Mohammad Matin Najafi, Xianju Zhu, Chrysanthi Kosyfaki, Laks V.S. Lakshmanan, Reynold Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.10948)  

**Abstract**: Subgraph counting the task of determining the number of instances of a query pattern within a large graph lies at the heart of many critical applications, from analyzing financial networks and transportation systems to understanding biological interactions. Despite decades of work yielding efficient algorithmic (AL) solutions and, more recently, machine learning (ML) approaches, a clear comparative understanding is elusive. This gap stems from the absence of a unified evaluation framework, standardized datasets, and accessible ground truths, all of which hinder systematic analysis and fair benchmarking. To overcome these barriers, we introduce BEACON: a comprehensive benchmark designed to rigorously evaluate both AL and ML-based subgraph counting methods. BEACON provides a standardized dataset with verified ground truths, an integrated evaluation environment, and a public leaderboard, enabling reproducible and transparent comparisons across diverse approaches. Our extensive experiments reveal that while AL methods excel in efficiently counting subgraphs on very large graphs, they struggle with complex patterns (e.g., those exceeding six nodes). In contrast, ML methods are capable of handling larger patterns but demand massive graph data inputs and often yield suboptimal accuracy on small, dense graphs. These insights not only highlight the unique strengths and limitations of each approach but also pave the way for future advancements in subgraph counting techniques. Overall, BEACON represents a significant step towards unifying and accelerating research in subgraph counting, encouraging innovative solutions and fostering a deeper understanding of the trade-offs between algorithmic and machine learning paradigms. 

---
# Can LLMs Leverage Observational Data? Towards Data-Driven Causal Discovery with LLMs 

**Authors**: Yuni Susanti, Michael Färber  

**Link**: [PDF](https://arxiv.org/pdf/2504.10936)  

**Abstract**: Causal discovery traditionally relies on statistical methods applied to observational data, often requiring large datasets and assumptions about underlying causal structures. Recent advancements in Large Language Models (LLMs) have introduced new possibilities for causal discovery by providing domain expert knowledge. However, it remains unclear whether LLMs can effectively process observational data for causal discovery. In this work, we explore the potential of LLMs for data-driven causal discovery by integrating observational data for LLM-based reasoning. Specifically, we examine whether LLMs can effectively utilize observational data through two prompting strategies: pairwise prompting and breadth first search (BFS)-based prompting. In both approaches, we incorporate the observational data directly into the prompt to assess LLMs' ability to infer causal relationships from such data. Experiments on benchmark datasets show that incorporating observational data enhances causal discovery, boosting F1 scores by up to 0.11 point using both pairwise and BFS LLM-based prompting, while outperforming traditional statistical causal discovery baseline by up to 0.52 points. Our findings highlight the potential and limitations of LLMs for data-driven causal discovery, demonstrating their ability to move beyond textual metadata and effectively interpret and utilize observational data for more informed causal reasoning. Our studies lays the groundwork for future advancements toward fully LLM-driven causal discovery. 

---
# Transfer Learning for Temporal Link Prediction 

**Authors**: Ayan Chatterjee, Barbara Ikica, Babak Ravandi, John Palowitch  

**Link**: [PDF](https://arxiv.org/pdf/2504.10925)  

**Abstract**: Link prediction on graphs has applications spanning from recommender systems to drug discovery. Temporal link prediction (TLP) refers to predicting future links in a temporally evolving graph and adds additional complexity related to the dynamic nature of graphs. State-of-the-art TLP models incorporate memory modules alongside graph neural networks to learn both the temporal mechanisms of incoming nodes and the evolving graph topology. However, memory modules only store information about nodes seen at train time, and hence such models cannot be directly transferred to entirely new graphs at test time and deployment. In this work, we study a new transfer learning task for temporal link prediction, and develop transfer-effective methods for memory-laden models. Specifically, motivated by work showing the informativeness of structural signals for the TLP task, we augment a structural mapping module to the existing TLP model architectures, which learns a mapping from graph structural (topological) features to memory embeddings. Our work paves the way for a memory-free foundation model for TLP. 

---
# Towards A Universal Graph Structural Encoder 

**Authors**: Jialin Chen, Haolan Zuo, Haoyu Peter Wang, Siqi Miao, Pan Li, Rex Ying  

**Link**: [PDF](https://arxiv.org/pdf/2504.10917)  

**Abstract**: Recent advancements in large-scale pre-training have shown the potential to learn generalizable representations for downstream tasks. In the graph domain, however, capturing and transferring structural information across different graph domains remains challenging, primarily due to the inherent differences in topological patterns across various contexts. Additionally, most existing models struggle to capture the complexity of rich graph structures, leading to inadequate exploration of the embedding space. To address these challenges, we propose GFSE, a universal graph structural encoder designed to capture transferable structural patterns across diverse domains such as molecular graphs, social networks, and citation networks. GFSE is the first cross-domain graph structural encoder pre-trained with multiple self-supervised learning objectives. Built on a Graph Transformer, GFSE incorporates attention mechanisms informed by graph inductive bias, enabling it to encode intricate multi-level and fine-grained topological features. The pre-trained GFSE produces generic and theoretically expressive positional and structural encoding for graphs, which can be seamlessly integrated with various downstream graph feature encoders, including graph neural networks for vectorized features and Large Language Models for text-attributed graphs. Comprehensive experiments on synthetic and real-world datasets demonstrate GFSE's capability to significantly enhance the model's performance while requiring substantially less task-specific fine-tuning. Notably, GFSE achieves state-of-the-art performance in 81.6% evaluated cases, spanning diverse graph models and datasets, highlighting its potential as a powerful and versatile encoder for graph-structured data. 

---
# LOKA Protocol: A Decentralized Framework for Trustworthy and Ethical AI Agent Ecosystems 

**Authors**: Rajesh Ranjan, Shailja Gupta, Surya Narayan Singh  

**Link**: [PDF](https://arxiv.org/pdf/2504.10915)  

**Abstract**: The rise of autonomous AI agents, capable of perceiving, reasoning, and acting independently, signals a profound shift in how digital ecosystems operate, govern, and evolve. As these agents proliferate beyond centralized infrastructures, they expose foundational gaps in identity, accountability, and ethical alignment. Three critical questions emerge: Identity: Who or what is the agent? Accountability: Can its actions be verified, audited, and trusted? Ethical Consensus: Can autonomous systems reliably align with human values and prevent harmful emergent behaviors? We present the novel LOKA Protocol (Layered Orchestration for Knowledgeful Agents), a unified, systems-level architecture for building ethically governed, interoperable AI agent ecosystems. LOKA introduces a proposed Universal Agent Identity Layer (UAIL) for decentralized, verifiable identity; intent-centric communication protocols for semantic coordination across diverse agents; and a Decentralized Ethical Consensus Protocol (DECP) that enables agents to make context-aware decisions grounded in shared ethical baselines. Anchored in emerging standards such as Decentralized Identifiers (DIDs), Verifiable Credentials (VCs), and post-quantum cryptography, LOKA offers a scalable, future-resilient blueprint for multi-agent AI governance. By embedding identity, trust, and ethics into the protocol layer itself, LOKA establishes the foundation for a new era of responsible, transparent, and autonomous AI ecosystems operating across digital and physical domains. 

---
# Efficient Reasoning Models: A Survey 

**Authors**: Sicheng Feng, Gongfan Fang, Xinyin Ma, Xinchao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.10903)  

**Abstract**: Reasoning models have demonstrated remarkable progress in solving complex and logic-intensive tasks by generating extended Chain-of-Thoughts (CoTs) prior to arriving at a final answer. Yet, the emergence of this "slow-thinking" paradigm, with numerous tokens generated in sequence, inevitably introduces substantial computational overhead. To this end, it highlights an urgent need for effective acceleration. This survey aims to provide a comprehensive overview of recent advances in efficient reasoning. It categorizes existing works into three key directions: (1) shorter - compressing lengthy CoTs into concise yet effective reasoning chains; (2) smaller - developing compact language models with strong reasoning capabilities through techniques such as knowledge distillation, other model compression techniques, and reinforcement learning; and (3) faster - designing efficient decoding strategies to accelerate inference. A curated collection of papers discussed in this survey is available in our GitHub repository. 

---
# Bridging Distribution Gaps in Time Series Foundation Model Pretraining with Prototype-Guided Normalization 

**Authors**: Peiliang Gong, Emadeldeen Eldele, Min Wu, Zhenghua Chen, Xiaoli Li, Daoqiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.10900)  

**Abstract**: Foundation models have achieved remarkable success across diverse machine-learning domains through large-scale pretraining on large, diverse datasets. However, pretraining on such datasets introduces significant challenges due to substantial mismatches in data distributions, a problem particularly pronounced with time series data. In this paper, we tackle this issue by proposing a domain-aware adaptive normalization strategy within the Transformer architecture. Specifically, we replace the traditional LayerNorm with a prototype-guided dynamic normalization mechanism (ProtoNorm), where learned prototypes encapsulate distinct data distributions, and sample-to-prototype affinity determines the appropriate normalization layer. This mechanism effectively captures the heterogeneity of time series characteristics, aligning pretrained representations with downstream tasks. Through comprehensive empirical evaluation, we demonstrate that our method significantly outperforms conventional pretraining techniques across both classification and forecasting tasks, while effectively mitigating the adverse effects of distribution shifts during pretraining. Incorporating ProtoNorm is as simple as replacing a single line of code. Extensive experiments on diverse real-world time series benchmarks validate the robustness and generalizability of our approach, advancing the development of more versatile time series foundation models. 

---
# Xpose: Bi-directional Engineering for Hidden Query Extraction 

**Authors**: Ahana Pradhan, Jayant Haritsa  

**Link**: [PDF](https://arxiv.org/pdf/2504.10898)  

**Abstract**: Query reverse engineering (QRE) aims to synthesize a SQL query to connect a given database and result instance. A recent variation of QRE is where an additional input, an opaque executable containing a ground-truth query, is provided, and the goal is to non-invasively extract this specific query through only input-output examples. This variant, called Hidden Query Extraction (HQE), has a spectrum of industrial use-cases including query recovery, database security, and vendor migration. The reverse engineering (RE) tools developed for HQE, which are based on database mutation and generation techniques, can only extract flat queries with key-based equi joins and conjunctive arithmetic filter predicates, making them limited wrt both query structure and query operators. In this paper, we present Xpose, a HQE solution that elevates the extraction scope to realistic complex queries, such as those found in the TPCH benchmark. A two-pronged approach is taken: (1) The existing RE scope is substantially extended to incorporate union connectors, algebraic filter predicates, and disjunctions for both values and predicates. (2) The predictive power of LLMs is leveraged to convert business descriptions of the opaque application into extraction guidance, representing ``forward engineering" (FE). The FE module recognizes common constructs, such as nesting of sub-queries, outer joins, and scalar functions. In essence, FE establishes the broad query contours, while RE fleshes out the fine-grained details. We have evaluated Xpose on (a) E-TPCH, a query suite comprising the complete TPCH benchmark extended with queries featuring unions, diverse join types, and sub-queries; and (b) the real-world STACK benchmark. The experimental results demonstrate that its bi-directional engineering approach accurately extracts these complex queries, representing a significant step forward with regard to HQE coverage. 

---
# CDUPatch: Color-Driven Universal Adversarial Patch Attack for Dual-Modal Visible-Infrared Detectors 

**Authors**: Jiahuan Long, Wen Yao, Tingsong Jiang, Chao Ma  

**Link**: [PDF](https://arxiv.org/pdf/2504.10888)  

**Abstract**: Adversarial patches are widely used to evaluate the robustness of object detection systems in real-world scenarios. These patches were initially designed to deceive single-modal detectors (e.g., visible or infrared) and have recently been extended to target visible-infrared dual-modal detectors. However, existing dual-modal adversarial patch attacks have limited attack effectiveness across diverse physical scenarios. To address this, we propose CDUPatch, a universal cross-modal patch attack against visible-infrared object detectors across scales, views, and scenarios. Specifically, we observe that color variations lead to different levels of thermal absorption, resulting in temperature differences in infrared imaging. Leveraging this property, we propose an RGB-to-infrared adapter that maps RGB patches to infrared patches, enabling unified optimization of cross-modal patches. By learning an optimal color distribution on the adversarial patch, we can manipulate its thermal response and generate an adversarial infrared texture. Additionally, we introduce a multi-scale clipping strategy and construct a new visible-infrared dataset, MSDrone, which contains aerial vehicle images in varying scales and perspectives. These data augmentation strategies enhance the robustness of our patch in real-world conditions. Experiments on four benchmark datasets (e.g., DroneVehicle, LLVIP, VisDrone, MSDrone) show that our method outperforms existing patch attacks in the digital domain. Extensive physical tests further confirm strong transferability across scales, views, and scenarios. 

---
# Exploring Persona-dependent LLM Alignment for the Moral Machine Experiment 

**Authors**: Jiseon Kim, Jea Kwon, Luiz Felipe Vecchietti, Alice Oh, Meeyoung Cha  

**Link**: [PDF](https://arxiv.org/pdf/2504.10886)  

**Abstract**: Deploying large language models (LLMs) with agency in real-world applications raises critical questions about how these models will behave. In particular, how will their decisions align with humans when faced with moral dilemmas? This study examines the alignment between LLM-driven decisions and human judgment in various contexts of the moral machine experiment, including personas reflecting different sociodemographics. We find that the moral decisions of LLMs vary substantially by persona, showing greater shifts in moral decisions for critical tasks than humans. Our data also indicate an interesting partisan sorting phenomenon, where political persona predominates the direction and degree of LLM decisions. We discuss the ethical implications and risks associated with deploying these models in applications that involve moral decisions. 

---
# PuzzleBench: A Fully Dynamic Evaluation Framework for Large Multimodal Models on Puzzle Solving 

**Authors**: Zeyu Zhang, Zijian Chen, Zicheng Zhang, Yuze Sun, Yuan Tian, Ziheng Jia, Chunyi Li, Xiaohong Liu, Xiongkuo Min, Guangtao Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2504.10885)  

**Abstract**: Large Multimodal Models (LMMs) have demonstrated impressive capabilities across a wide range of multimodal tasks, achieving ever-increasing performance on various evaluation benchmarks. However, existing benchmarks are typically static and often overlap with pre-training datasets, leading to fixed complexity constraints and substantial data contamination issues. Meanwhile, manually annotated datasets are labor-intensive, time-consuming, and subject to human bias and inconsistency, leading to reliability and reproducibility issues. To address these problems, we propose a fully dynamic multimodal evaluation framework, named Open-ended Visual Puzzle Generation (OVPG), which aims to generate fresh, diverse, and verifiable evaluation data automatically in puzzle-solving tasks. Specifically, the OVPG pipeline consists of a raw material sampling module, a visual content generation module, and a puzzle rule design module, which ensures that each evaluation instance is primitive, highly randomized, and uniquely solvable, enabling continual adaptation to the evolving capabilities of LMMs. Built upon OVPG, we construct PuzzleBench, a dynamic and scalable benchmark comprising 11,840 VQA samples. It features six carefully designed puzzle tasks targeting three core LMM competencies, visual recognition, logical reasoning, and context understanding. PuzzleBench differs from static benchmarks that quickly become outdated. It enables ongoing dataset refreshing through OVPG and a rich set of open-ended puzzle designs, allowing seamless adaptation to the evolving capabilities of LMMs. 

---
# Bringing together invertible UNets with invertible attention modules for memory-efficient diffusion models 

**Authors**: Karan Jain, Mohammad Nayeem Teli  

**Link**: [PDF](https://arxiv.org/pdf/2504.10883)  

**Abstract**: Diffusion models have recently gained state of the art performance on many image generation tasks. However, most models require significant computational resources to achieve this. This becomes apparent in the application of medical image synthesis due to the 3D nature of medical datasets like CT-scans, MRIs, electron microscope, etc. In this paper we propose a novel architecture for a single GPU memory-efficient training for diffusion models for high dimensional medical datasets. The proposed model is built by using an invertible UNet architecture with invertible attention modules. This leads to the following two contributions: 1. denoising diffusion models and thus enabling memory usage to be independent of the dimensionality of the dataset, and 2. reducing the energy usage during training. While this new model can be applied to a multitude of image generation tasks, we showcase its memory-efficiency on the 3D BraTS2020 dataset leading to up to 15\% decrease in peak memory consumption during training with comparable results to SOTA while maintaining the image quality. 

---
# Large Language Model-Informed Feature Discovery Improves Prediction and Interpretation of Credibility Perceptions of Visual Content 

**Authors**: Yilang Peng, Sijia Qian, Yingdan Lu, Cuihua Shen  

**Link**: [PDF](https://arxiv.org/pdf/2504.10878)  

**Abstract**: In today's visually dominated social media landscape, predicting the perceived credibility of visual content and understanding what drives human judgment are crucial for countering misinformation. However, these tasks are challenging due to the diversity and richness of visual features. We introduce a Large Language Model (LLM)-informed feature discovery framework that leverages multimodal LLMs, such as GPT-4o, to evaluate content credibility and explain its reasoning. We extract and quantify interpretable features using targeted prompts and integrate them into machine learning models to improve credibility predictions. We tested this approach on 4,191 visual social media posts across eight topics in science, health, and politics, using credibility ratings from 5,355 crowdsourced workers. Our method outperformed zero-shot GPT-based predictions by 13 percent in R2, and revealed key features like information concreteness and image format. We discuss the implications for misinformation mitigation, visual credibility, and the role of LLMs in social science. 

---
# Can Vision-Language Models Understand and Interpret Dynamic Gestures from Pedestrians? Pilot Datasets and Exploration Towards Instructive Nonverbal Commands for Cooperative Autonomous Vehicles 

**Authors**: Tonko E. W. Bossen, Andreas Møgelmose, Ross Greer  

**Link**: [PDF](https://arxiv.org/pdf/2504.10873)  

**Abstract**: In autonomous driving, it is crucial to correctly interpret traffic gestures (TGs), such as those of an authority figure providing orders or instructions, or a pedestrian signaling the driver, to ensure a safe and pleasant traffic environment for all road users. This study investigates the capabilities of state-of-the-art vision-language models (VLMs) in zero-shot interpretation, focusing on their ability to caption and classify human gestures in traffic contexts. We create and publicly share two custom datasets with varying formal and informal TGs, such as 'Stop', 'Reverse', 'Hail', etc. The datasets are "Acted TG (ATG)" and "Instructive TG In-The-Wild (ITGI)". They are annotated with natural language, describing the pedestrian's body position and gesture. We evaluate models using three methods utilizing expert-generated captions as baseline and control: (1) caption similarity, (2) gesture classification, and (3) pose sequence reconstruction similarity. Results show that current VLMs struggle with gesture understanding: sentence similarity averages below 0.59, and classification F1 scores reach only 0.14-0.39, well below the expert baseline of 0.70. While pose reconstruction shows potential, it requires more data and refined metrics to be reliable. Our findings reveal that although some SOTA VLMs can interpret zero-shot human traffic gestures, none are accurate and robust enough to be trustworthy, emphasizing the need for further research in this domain. 

---
# Moving Beyond Next-Token Prediction: Transformers are Context-Sensitive Language Generators 

**Authors**: Phill Kyu Rhee  

**Link**: [PDF](https://arxiv.org/pdf/2504.10845)  

**Abstract**: Large Language Models (LLMs), powered by Transformers, have demonstrated human-like intelligence capabilities, yet their underlying mechanisms remain poorly understood. This paper presents a novel framework for interpreting LLMs as probabilistic left context-sensitive languages (CSLs) generators. We hypothesize that Transformers can be effectively decomposed into three fundamental components: context windows, attention mechanisms, and autoregressive generation frameworks. This decomposition allows for the development of more flexible and interpretable computational models, moving beyond the traditional view of attention and autoregression as inseparable processes. We argue that next-token predictions can be understood as probabilistic, dynamic approximations of left CSL production rules, providing an intuitive explanation for how simple token predictions can yield human-like intelligence outputs. Given that all CSLs are left context-sensitive (Penttonen, 1974), we conclude that Transformers stochastically approximate CSLs, which are widely recognized as models of human-like intelligence. This interpretation bridges the gap between Formal Language Theory and the observed generative power of Transformers, laying a foundation for future advancements in generative AI theory and applications. Our novel perspective on Transformer architectures will foster a deeper understanding of LLMs and their future potentials. 

---
# Rethinking Theory of Mind Benchmarks for LLMs: Towards A User-Centered Perspective 

**Authors**: Qiaosi Wang, Xuhui Zhou, Maarten Sap, Jodi Forlizzi, Hong Shen  

**Link**: [PDF](https://arxiv.org/pdf/2504.10839)  

**Abstract**: The last couple of years have witnessed emerging research that appropriates Theory-of-Mind (ToM) tasks designed for humans to benchmark LLM's ToM capabilities as an indication of LLM's social intelligence. However, this approach has a number of limitations. Drawing on existing psychology and AI literature, we summarize the theoretical, methodological, and evaluation limitations by pointing out that certain issues are inherently present in the original ToM tasks used to evaluate human's ToM, which continues to persist and exacerbated when appropriated to benchmark LLM's ToM. Taking a human-computer interaction (HCI) perspective, these limitations prompt us to rethink the definition and criteria of ToM in ToM benchmarks in a more dynamic, interactional approach that accounts for user preferences, needs, and experiences with LLMs in such evaluations. We conclude by outlining potential opportunities and challenges towards this direction. 

---
# Uplink Assisted Joint Channel Estimation and CSI Feedback: An Approach Based on Deep Joint Source-Channel Coding 

**Authors**: Yiran Guo, Wei Chen, Bo Ai  

**Link**: [PDF](https://arxiv.org/pdf/2504.10836)  

**Abstract**: In frequency division duplex (FDD) multiple-input multiple-output (MIMO) wireless communication systems, the acquisition of downlink channel state information (CSI) is essential for maximizing spatial resource utilization and improving system spectral efficiency. The separate design of modules in AI-based CSI feedback architectures under traditional modular communication frameworks, including channel estimation (CE), CSI compression and feedback, leads to sub-optimal performance. In this paper, we propose an uplink assisted joint CE and and CSI feedback approach via deep learning for downlink CSI acquisition, which mitigates performance degradation caused by distribution bias across separately trained modules in traditional modular communication frameworks. The proposed network adopts a deep joint source-channel coding (DJSCC) architecture to mitigate the cliff effect encountered in the conventional separate source-channel coding. Furthermore, we exploit the uplink CSI as auxiliary information to enhance CSI reconstruction accuracy by leveraging the partial reciprocity between the uplink and downlink channels in FDD systems, without introducing additional overhead. The effectiveness of uplink CSI as assisted information and the necessity of an end-toend multi-module joint training architecture is validated through comprehensive ablation and scalability experiments. 

---
# Towards Spatially-Aware and Optimally Faithful Concept-Based Explanations 

**Authors**: Shubham Kumar, Dwip Dalal, Narendra Ahuja  

**Link**: [PDF](https://arxiv.org/pdf/2504.10833)  

**Abstract**: Post-hoc, unsupervised concept-based explanation methods (U-CBEMs) are a promising tool for generating semantic explanations of the decision-making processes in deep neural networks, having applications in both model improvement and understanding. It is vital that the explanation is accurate, or faithful, to the model, yet we identify several limitations of prior faithfulness metrics that inhibit an accurate evaluation; most notably, prior metrics involve only the set of concepts present, ignoring how they may be spatially distributed. We address these limitations with Surrogate Faithfulness (SF), an evaluation method that introduces a spatially-aware surrogate and two novel faithfulness metrics. Using SF, we produce Optimally Faithful (OF) explanations, where concepts are found that maximize faithfulness. Our experiments show that (1) adding spatial-awareness to prior U-CBEMs increases faithfulness in all cases; (2) OF produces significantly more faithful explanations than prior U-CBEMs (30% or higher improvement in error); (3) OF's learned concepts generalize well to out-of-domain data and are more robust to adversarial examples, where prior U-CBEMs struggle. 

---
# CLASH: Evaluating Language Models on Judging High-Stakes Dilemmas from Multiple Perspectives 

**Authors**: Ayoung Lee, Ryan Sungmo Kwon, Peter Railton, Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.10823)  

**Abstract**: Navigating high-stakes dilemmas involving conflicting values is challenging even for humans, let alone for AI. Yet prior work in evaluating the reasoning capabilities of large language models (LLMs) in such situations has been limited to everyday scenarios. To close this gap, this work first introduces CLASH (Character perspective-based LLM Assessments in Situations with High-stakes), a meticulously curated dataset consisting of 345 high-impact dilemmas along with 3,795 individual perspectives of diverse values. In particular, we design CLASH in a way to support the study of critical aspects of value-based decision-making processes which are missing from prior work, including understanding decision ambivalence and psychological discomfort as well as capturing the temporal shifts of values in characters' perspectives. By benchmarking 10 open and closed frontier models, we uncover several key findings. (1) Even the strongest models, such as GPT-4o and Claude-Sonnet, achieve less than 50% accuracy in identifying situations where the decision should be ambivalent, while they perform significantly better in clear-cut scenarios. (2) While LLMs reasonably predict psychological discomfort as marked by human, they inadequately comprehend perspectives involving value shifts, indicating a need for LLMs to reason over complex values. (3) Our experiments also reveal a significant correlation between LLMs' value preferences and their steerability towards a given value. (4) Finally, LLMs exhibit greater steerability when engaged in value reasoning from a third-party perspective, compared to a first-person setup, though certain value pairs benefit uniquely from the first-person framing. 

---
# Progressive Rock Music Classification 

**Authors**: Arpan Nagar, Joseph Bensabat, Jokent Gaza, Moinak Dey  

**Link**: [PDF](https://arxiv.org/pdf/2504.10821)  

**Abstract**: This study investigates the classification of progressive rock music, a genre characterized by complex compositions and diverse instrumentation, distinct from other musical styles. Addressing this Music Information Retrieval (MIR) task, we extracted comprehensive audio features, including spectrograms, Mel-Frequency Cepstral Coefficients (MFCCs), chromagrams, and beat positions from song snippets using the Librosa library. A winner-take-all voting strategy was employed to aggregate snippet-level predictions into final song classifications. We conducted a comparative analysis of various machine learning techniques. Ensemble methods, encompassing Bagging (Random Forest, ExtraTrees, Bagging Classifier) and Boosting (XGBoost, Gradient Boosting), were explored, utilizing Principal Component Analysis (PCA) for dimensionality reduction to manage computational constraints with high-dimensional feature sets. Additionally, deep learning approaches were investigated, including the development of custom 1D Convolutional Neural Network (1D CNN) architectures (named "Zuck" and "Satya") featuring specific layer configurations, normalization, and activation functions. Furthermore, we fine-tuned a state-of-the-art Audio Spectrogram Transformer (AST) model, leveraging its attention-based mechanisms for audio classification. Performance evaluation on validation and test sets revealed varying effectiveness across models, with ensemble methods like Extra Trees achieving test accuracies up to 76.38%. This research provides insights into the application and relative performance of diverse machine learning paradigms for the nuanced task of progressive rock genre classification. 

---
# FHBench: Towards Efficient and Personalized Federated Learning for Multimodal Healthcare 

**Authors**: Penghao Wang, Qian Chen, Teng Zhang, Yingwei Zhang, Wang Lu, Yiqiang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.10817)  

**Abstract**: Federated Learning (FL) has emerged as an effective solution for multi-institutional collaborations without sharing patient data, offering a range of methods tailored for diverse applications. However, real-world medical datasets are often multimodal, and computational resources are limited, posing significant challenges for existing FL approaches. Recognizing these limitations, we developed the Federated Healthcare Benchmark(FHBench), a benchmark specifically designed from datasets derived from real-world healthcare applications. FHBench encompasses critical diagnostic tasks across domains such as the nervous, cardiovascular, and respiratory systems and general pathology, providing comprehensive support for multimodal healthcare evaluations and filling a significant gap in existing benchmarks. Building on FHBench, we introduced Efficient Personalized Federated Learning with Adaptive LoRA(EPFL), a personalized FL framework that demonstrates superior efficiency and effectiveness across various healthcare modalities. Our results highlight the robustness of FHBench as a benchmarking tool and the potential of EPFL as an innovative approach to advancing healthcare-focused FL, addressing key limitations of existing methods. 

---
# E2E Parking Dataset: An Open Benchmark for End-to-End Autonomous Parking 

**Authors**: Kejia Gao, Liguo Zhou, Mingjun Liu, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2504.10812)  

**Abstract**: End-to-end learning has shown great potential in autonomous parking, yet the lack of publicly available datasets limits reproducibility and benchmarking. While prior work introduced a visual-based parking model and a pipeline for data generation, training, and close-loop test, the dataset itself was not released. To bridge this gap, we create and open-source a high-quality dataset for end-to-end autonomous parking. Using the original model, we achieve an overall success rate of 85.16% with lower average position and orientation errors (0.24 meters and 0.34 degrees). 

---
# PatrolVision: Automated License Plate Recognition in the wild 

**Authors**: Anmol Singhal Navya Singhal  

**Link**: [PDF](https://arxiv.org/pdf/2504.10810)  

**Abstract**: Adoption of AI driven techniques in public services remains low due to challenges related to accuracy and speed of information at population scale. Computer vision techniques for traffic monitoring have not gained much popularity despite their relative strength in areas such as autonomous driving. Despite large number of academic methods for Automatic License Plate Recognition (ALPR) systems, very few provide an end to end solution for patrolling in the city. This paper presents a novel prototype for a low power GPU based patrolling system to be deployed in an urban environment on surveillance vehicles for automated vehicle detection, recognition and tracking. In this work, we propose a complete ALPR system for Singapore license plates having both single and double line creating our own YOLO based network. We focus on unconstrained capture scenarios as would be the case in real world application, where the license plate (LP) might be considerably distorted due to oblique views. In this work, we first detect the license plate from the full image using RFB-Net and rectify multiple distorted license plates in a single image. After that, the detected license plate image is fed to our network for character recognition. We evaluate the performance of our proposed system on a newly built dataset covering more than 16,000 images. The system was able to correctly detect license plates with 86\% precision and recognize characters of a license plate in 67\% of the test set, and 89\% accuracy with one incorrect character (partial match). We also test latency of our system and achieve 64FPS on Tesla P4 GPU 

---
# Name of Thrones: Evaluating How LLMs Rank Student Names, Race, and Gender in Status Hierarchies 

**Authors**: Annabella Sakunkoo, Jonathan Sakunkoo  

**Link**: [PDF](https://arxiv.org/pdf/2504.10797)  

**Abstract**: Across cultures, names tell a lot about their bearers as they carry deep personal and cultural significance. Names also serve as powerful signals of gender, race, and status in the social hierarchy - a pecking order in which individual positions shape others' expectations on their perceived competence and worth. With the widespread adoption of LLMs and as names are often an input for LLMs, it is crucial to evaluate whether LLMs may sort people into status positions based on first and last names and, if so, whether it is in an unfair, biased fashion. While prior work has primarily investigated biases in first names, little attention has been paid to last names and even less to the combined effects of first and last names. In this study, we conduct a large-scale analysis of name variations across 5 ethnicities to examine how AI exhibits name biases. Our study investigates three key characteristics of inequality and finds that LLMs reflect and reinforce status hierarchies based on names that signal gender and ethnicity as they encode differential expectations of competence, leadership, and economic potential. Contrary to the common assumption that AI tends to favor Whites, we show that East and, in some contexts, South Asian names receive higher rankings. We also disaggregate Asians, a population projected to be the largest immigrant group in the U.S. by 2055. Our results challenge the monolithic Asian model minority assumption, illustrating a more complex and stratified model of bias. Gender moderates biases, with girls facing unfair disadvantages in certain racial groups. Additionally, spanning cultural categories by adopting Western first names improves AI-perceived status for East and Southeast Asian students, particularly for girls. Our findings underscore the importance of intersectional and more nuanced understandings of race, gender, and mixed identities in the evaluation of LLMs. 

---
# Visual Language Models show widespread visual deficits on neuropsychological tests 

**Authors**: Gene Tangtartharakul, Katherine R. Storrs  

**Link**: [PDF](https://arxiv.org/pdf/2504.10786)  

**Abstract**: Visual Language Models (VLMs) show remarkable performance in visual reasoning tasks, successfully tackling college-level challenges that require high-level understanding of images. However, some recent reports of VLMs struggling to reason about elemental visual concepts like orientation, position, continuity, and occlusion suggest a potential gulf between human and VLM vision. Here we use the toolkit of neuropsychology to systematically assess the capabilities of three state-of-the-art VLMs across visual domains. Using 51 tests drawn from six clinical and experimental batteries, we characterise the visual abilities of leading VLMs relative to normative performance in healthy adults. While the models excel in straightforward object recognition tasks, we find widespread deficits in low- and mid-level visual abilities that would be considered clinically significant in humans. These selective deficits, profiled through validated test batteries, suggest that an artificial system can achieve complex object recognition without developing foundational visual concepts that in humans require no explicit training. 

---
# ATLASv2: LLM-Guided Adaptive Landmark Acquisition and Navigation on the Edge 

**Authors**: Mikolaj Walczak, Uttej Kallakuri, Tinoosh Mohsenin  

**Link**: [PDF](https://arxiv.org/pdf/2504.10784)  

**Abstract**: Autonomous systems deployed on edge devices face significant challenges, including resource constraints, real-time processing demands, and adapting to dynamic environments. This work introduces ATLASv2, a novel system that integrates a fine-tuned TinyLLM, real-time object detection, and efficient path planning to enable hierarchical, multi-task navigation and manipulation all on the edge device, Jetson Nano. ATLASv2 dynamically expands its navigable landmarks by detecting and localizing objects in the environment which are saved to its internal knowledge base to be used for future task execution. We evaluate ATLASv2 in real-world environments, including a handcrafted home and office setting constructed with diverse objects and landmarks. Results show that ATLASv2 effectively interprets natural language instructions, decomposes them into low-level actions, and executes tasks with high success rates. By leveraging generative AI in a fully on-board framework, ATLASv2 achieves optimized resource utilization with minimal prompting latency and power consumption, bridging the gap between simulated environments and real-world applications. 

---
# Neural Network Emulation of the Classical Limit in Quantum Systems via Learned Observable Mappings 

**Authors**: Kamran Majid  

**Link**: [PDF](https://arxiv.org/pdf/2504.10781)  

**Abstract**: The classical limit of quantum mechanics, formally investigated through frameworks like strict deformation quantization, remains a profound area of inquiry in the philosophy of physics. This paper explores a computational approach employing a neural network to emulate the emergence of classical behavior from the quantum harmonic oscillator as Planck's constant $\hbar$ approaches zero. We develop and train a neural network architecture to learn the mapping from initial expectation values and $\hbar$ to the time evolution of the expectation value of position. By analyzing the network's predictions across different regimes of hbar, we aim to provide computational insights into the nature of the quantum-classical transition. This work demonstrates the potential of machine learning as a complementary tool for exploring foundational questions in quantum mechanics and its classical limit. 

---
# The Art of Audience Engagement: LLM-Based Thin-Slicing of Scientific Talks 

**Authors**: Ralf Schmälzle, Sue Lim, Yuetong Du, Gary Bente  

**Link**: [PDF](https://arxiv.org/pdf/2504.10768)  

**Abstract**: This paper examines the thin-slicing approach - the ability to make accurate judgments based on minimal information - in the context of scientific presentations. Drawing on research from nonverbal communication and personality psychology, we show that brief excerpts (thin slices) reliably predict overall presentation quality. Using a novel corpus of over one hundred real-life science talks, we employ Large Language Models (LLMs) to evaluate transcripts of full presentations and their thin slices. By correlating LLM-based evaluations of short excerpts with full-talk assessments, we determine how much information is needed for accurate predictions. Our results demonstrate that LLM-based evaluations align closely with human ratings, proving their validity, reliability, and efficiency. Critically, even very short excerpts (less than 10 percent of a talk) strongly predict overall evaluations. This suggests that the first moments of a presentation convey relevant information that is used in quality evaluations and can shape lasting impressions. The findings are robust across different LLMs and prompting strategies. This work extends thin-slicing research to public speaking and connects theories of impression formation to LLMs and current research on AI communication. We discuss implications for communication and social cognition research on message reception. Lastly, we suggest an LLM-based thin-slicing framework as a scalable feedback tool to enhance human communication. 

---
# How Instruction and Reasoning Data shape Post-Training: Data Quality through the Lens of Layer-wise Gradients 

**Authors**: Ming Li, Yanhong Li, Ziyue Li, Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.10766)  

**Abstract**: As the post-training of large language models (LLMs) advances from instruction-following to complex reasoning tasks, understanding how different data affect finetuning dynamics remains largely unexplored. In this paper, we present a spectral analysis of layer-wise gradients induced by low/high-quality instruction and reasoning data for LLM post-training. Our analysis reveals that widely-studied metrics for data evaluation, e.g., IFD, InsTag, Difficulty, and Reward, can be explained and unified by spectral properties computed from gradients' singular value decomposition (SVD). Specifically, higher-quality data are usually associated with lower nuclear norms and higher effective ranks. Notably, effective rank exhibits better robustness and resolution than nuclear norm in capturing subtle quality differences. For example, reasoning data achieves substantially higher effective ranks than instruction data, implying richer gradient structures on more complex tasks. Our experiments also highlight that models within the same family share similar gradient patterns regardless of their sizes, whereas different model families diverge significantly. Providing a unified view on the effects of data quality across instruction and reasoning data, this work illuminates the interplay between data quality and training stability, shedding novel insights into developing better data exploration strategies for post-training. 

---
# Epistemic Uncertainty-aware Recommendation Systems via Bayesian Deep Ensemble Learning 

**Authors**: Radin Cheraghi, Amir Mohammad Mahfoozi, Sepehr Zolfaghari, Mohammadshayan Shabani, Maryam Ramezani, Hamid R. Rabiee  

**Link**: [PDF](https://arxiv.org/pdf/2504.10753)  

**Abstract**: Recommending items to users has long been a fundamental task, and studies have tried to improve it ever since. Most well-known models commonly employ representation learning to map users and items into a unified embedding space for matching assessment. These approaches have primary limitations, especially when dealing with explicit feedback and sparse data contexts. Two primary limitations are their proneness to overfitting and failure to incorporate epistemic uncertainty in predictions. To address these problems, we propose a novel Bayesian Deep Ensemble Collaborative Filtering method named BDECF. To improve model generalization and quality, we utilize Bayesian Neural Networks, which incorporate uncertainty within their weight parameters. In addition, we introduce a new interpretable non-linear matching approach for the user and item embeddings, leveraging the advantages of the attention mechanism. Furthermore, we endorse the implementation of an ensemble-based supermodel to generate more robust and reliable predictions, resulting in a more complete model. Empirical evaluation through extensive experiments and ablation studies across a range of publicly accessible real-world datasets with differing sparsity characteristics confirms our proposed method's effectiveness and the importance of its components. 

---
# Communication-aware Hierarchical Map Compression of Time-Varying Environments for Mobile Robots 

**Authors**: Daniel T. Larsson, Dipankar Maity  

**Link**: [PDF](https://arxiv.org/pdf/2504.10751)  

**Abstract**: In this paper, we develop a systematic framework for the time-sequential compression of dynamic probabilistic occupancy grids. Our approach leverages ideas from signal compression theory to formulate an optimization problem that searches for a multi-resolution hierarchical encoder that balances the quality of the compressed map (distortion) with its description size, the latter of which relates to the bandwidth required to reliably transmit the map to other agents or to store map estimates in on-board memory. The resulting optimization problem allows for multi-resolution map compressions to be obtained that satisfy available communication or memory resources, and does not require knowledge of the occupancy map dynamics. We develop an algorithm to solve our problem, and demonstrate the utility of the proposed framework in simulation on both static (i.e., non-time varying) and dynamic (time-varying) occupancy maps. 

---
# Hearing Anywhere in Any Environment 

**Authors**: Xiulong Liu, Anurag Kumar, Paul Calamia, Sebastia V. Amengual, Calvin Murdock, Ishwarya Ananthabhotla, Philip Robinson, Eli Shlizerman, Vamsi Krishna Ithapu, Ruohan Gao  

**Link**: [PDF](https://arxiv.org/pdf/2504.10746)  

**Abstract**: In mixed reality applications, a realistic acoustic experience in spatial environments is as crucial as the visual experience for achieving true immersion. Despite recent advances in neural approaches for Room Impulse Response (RIR) estimation, most existing methods are limited to the single environment on which they are trained, lacking the ability to generalize to new rooms with different geometries and surface materials. We aim to develop a unified model capable of reconstructing the spatial acoustic experience of any environment with minimum additional measurements. To this end, we present xRIR, a framework for cross-room RIR prediction. The core of our generalizable approach lies in combining a geometric feature extractor, which captures spatial context from panorama depth images, with a RIR encoder that extracts detailed acoustic features from only a few reference RIR samples. To evaluate our method, we introduce ACOUSTICROOMS, a new dataset featuring high-fidelity simulation of over 300,000 RIRs from 260 rooms. Experiments show that our method strongly outperforms a series of baselines. Furthermore, we successfully perform sim-to-real transfer by evaluating our model on four real-world environments, demonstrating the generalizability of our approach and the realism of our dataset. 

---
# CleanMAP: Distilling Multimodal LLMs for Confidence-Driven Crowdsourced HD Map Updates 

**Authors**: Ankit Kumar Shaw, Kun Jiang, Tuopu Wen, Chandan Kumar Sah, Yining Shi, Mengmeng Yang, Diange Yang, Xiaoli Lian  

**Link**: [PDF](https://arxiv.org/pdf/2504.10738)  

**Abstract**: The rapid growth of intelligent connected vehicles (ICVs) and integrated vehicle-road-cloud systems has increased the demand for accurate, real-time HD map updates. However, ensuring map reliability remains challenging due to inconsistencies in crowdsourced data, which suffer from motion blur, lighting variations, adverse weather, and lane marking degradation. This paper introduces CleanMAP, a Multimodal Large Language Model (MLLM)-based distillation framework designed to filter and refine crowdsourced data for high-confidence HD map updates. CleanMAP leverages an MLLM-driven lane visibility scoring model that systematically quantifies key visual parameters, assigning confidence scores (0-10) based on their impact on lane detection. A novel dynamic piecewise confidence-scoring function adapts scores based on lane visibility, ensuring strong alignment with human evaluations while effectively filtering unreliable data. To further optimize map accuracy, a confidence-driven local map fusion strategy ranks and selects the top-k highest-scoring local maps within an optimal confidence range (best score minus 10%), striking a balance between data quality and quantity. Experimental evaluations on a real-world autonomous vehicle dataset validate CleanMAP's effectiveness, demonstrating that fusing the top three local maps achieves the lowest mean map update error of 0.28m, outperforming the baseline (0.37m) and meeting stringent accuracy thresholds (<= 0.32m). Further validation with real-vehicle data confirms 84.88% alignment with human evaluators, reinforcing the model's robustness and reliability. This work establishes CleanMAP as a scalable and deployable solution for crowdsourced HD map updates, ensuring more precise and reliable autonomous navigation. The code will be available at this https URL 

---
# Frozen Layers: Memory-efficient Many-fidelity Hyperparameter Optimization 

**Authors**: Timur Carstensen, Neeratyoy Mallik, Frank Hutter, Martin Rapp  

**Link**: [PDF](https://arxiv.org/pdf/2504.10735)  

**Abstract**: As model sizes grow, finding efficient and cost-effective hyperparameter optimization (HPO) methods becomes increasingly crucial for deep learning pipelines. While multi-fidelity HPO (MF-HPO) trades off computational resources required for DL training with lower fidelity estimations, existing fidelity sources often fail under lower compute and memory constraints. We propose a novel fidelity source: the number of layers that are trained or frozen during training. For deep networks, this approach offers significant compute and memory savings while preserving rank correlations between hyperparameters at low fidelities compared to full model training. We demonstrate this in our empirical evaluation across ResNets and Transformers and additionally analyze the utility of frozen layers as a fidelity in using GPU resources as a fidelity in HPO, and for a combined MF-HPO with other fidelity sources. This contribution opens new applications for MF-HPO with hardware resources as a fidelity and creates opportunities for improved algorithms navigating joint fidelity spaces. 

---
# Optimizing Data Distribution and Kernel Performance for Efficient Training of Chemistry Foundation Models: A Case Study with MACE 

**Authors**: Jesun Firoz, Franco Pellegrini, Mario Geiger, Darren Hsu, Jenna A. Bilbrey, Han-Yi Chou, Maximilian Stadler, Markus Hoehnerbach, Tingyu Wang, Dejun Lin, Emine Kucukbenli, Henry W. Sprueill, Ilyes Batatia, Sotiris S. Xantheas, MalSoon Lee, Chris Mundy, Gabor Csanyi, Justin S. Smith, Ponnuswamy Sadayappan, Sutanay Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2504.10700)  

**Abstract**: Chemistry Foundation Models (CFMs) that leverage Graph Neural Networks (GNNs) operating on 3D molecular graph structures are becoming indispensable tools for computational chemists and materials scientists. These models facilitate the understanding of matter and the discovery of new molecules and materials. In contrast to GNNs operating on a large homogeneous graphs, GNNs used by CFMs process a large number of geometric graphs of varying sizes, requiring different optimization strategies than those developed for large homogeneous GNNs. This paper presents optimizations for two critical phases of CFM training: data distribution and model training, targeting MACE - a state-of-the-art CFM. We address the challenge of load balancing in data distribution by formulating it as a multi-objective bin packing problem. We propose an iterative algorithm that provides a highly effective, fast, and practical solution, ensuring efficient data distribution. For the training phase, we identify symmetric tensor contraction as the key computational kernel in MACE and optimize this kernel to improve the overall performance. Our combined approach of balanced data distribution and kernel optimization significantly enhances the training process of MACE. Experimental results demonstrate a substantial speedup, reducing per-epoch execution time for training from 12 to 2 minutes on 740 GPUs with a 2.6M sample dataset. 

---
# HyRRT-Connect: Bidirectional Motion Planning for Hybrid Dynamical Systems 

**Authors**: Nan Wang, Ricardo G. Sanfelice  

**Link**: [PDF](https://arxiv.org/pdf/2504.10699)  

**Abstract**: This paper proposes a bidirectional rapidly-exploring random trees (RRT) algorithm to solve the motion planning problem for hybrid systems. The proposed algorithm, called HyRRT-Connect, propagates in both forward and backward directions in hybrid time until an overlap between the forward and backward propagation results is detected. Then, HyRRT-Connect constructs a motion plan through the reversal and concatenation of functions defined on hybrid time domains, ensuring that the motion plan satisfies the given hybrid dynamics. To address the potential discontinuity along the flow caused by tolerating some distance between the forward and backward partial motion plans, we reconstruct the backward partial motion plan by a forward-in-hybrid-time simulation from the final state of the forward partial motion plan. effectively eliminating the discontinuity. The proposed algorithm is applied to an actuated bouncing ball system and a walking robot example to highlight its computational improvement. 

---
# The Jailbreak Tax: How Useful are Your Jailbreak Outputs? 

**Authors**: Kristina Nikolić, Luze Sun, Jie Zhang, Florian Tramèr  

**Link**: [PDF](https://arxiv.org/pdf/2504.10694)  

**Abstract**: Jailbreak attacks bypass the guardrails of large language models to produce harmful outputs. In this paper, we ask whether the model outputs produced by existing jailbreaks are actually useful. For example, when jailbreaking a model to give instructions for building a bomb, does the jailbreak yield good instructions? Since the utility of most unsafe answers (e.g., bomb instructions) is hard to evaluate rigorously, we build new jailbreak evaluation sets with known ground truth answers, by aligning models to refuse questions related to benign and easy-to-evaluate topics (e.g., biology or math). Our evaluation of eight representative jailbreaks across five utility benchmarks reveals a consistent drop in model utility in jailbroken responses, which we term the jailbreak tax. For example, while all jailbreaks we tested bypass guardrails in models aligned to refuse to answer math, this comes at the expense of a drop of up to 92% in accuracy. Overall, our work proposes the jailbreak tax as a new important metric in AI safety, and introduces benchmarks to evaluate existing and future jailbreaks. We make the benchmark available at this https URL 

---
# NTIRE 2025 Challenge on Cross-Domain Few-Shot Object Detection: Methods and Results 

**Authors**: Yuqian Fu, Xingyu Qiu, Bin Ren, Yanwei Fu, Radu Timofte, Nicu Sebe, Ming-Hsuan Yang, Luc Van Gool, Kaijin Zhang, Qingpeng Nong, Xiugang Dong, Hong Gao, Xiangsheng Zhou, Jiancheng Pan, Yanxing Liu, Xiao He, Jiahao Li, Yuze Sun, Xiaomeng Huang, Zhenyu Zhang, Ran Ma, Yuhan Liu, Zijian Zhuang, Shuai Yi, Yixiong Zou, Lingyi Hong, Mingxi Chen, Runze Li, Xingdong Sheng, Wenqiang Zhang, Weisen Chen, Yongxin Yan, Xinguo Chen, Yuanjie Shao, Zhengrong Zuo, Nong Sang, Hao Wu, Haoran Sun, Shuming Hu, Yan Zhang, Zhiguang Shi, Yu Zhang, Chao Chen, Tao Wang, Da Feng, Linhai Zhuo, Ziming Lin, Yali Huang, Jie Me, Yiming Yang, Mi Guo, Mingyuan Jiu, Mingliang Xu, Maomao Xiong, Qunshu Zhang, Xinyu Cao, Yuqing Yang, Dianmo Sheng, Xuanpu Zhao, Zhiyu Li, Xuyang Ding, Wenqian Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.10685)  

**Abstract**: Cross-Domain Few-Shot Object Detection (CD-FSOD) poses significant challenges to existing object detection and few-shot detection models when applied across domains. In conjunction with NTIRE 2025, we organized the 1st CD-FSOD Challenge, aiming to advance the performance of current object detectors on entirely novel target domains with only limited labeled data. The challenge attracted 152 registered participants, received submissions from 42 teams, and concluded with 13 teams making valid final submissions. Participants approached the task from diverse perspectives, proposing novel models that achieved new state-of-the-art (SOTA) results under both open-source and closed-source settings. In this report, we present an overview of the 1st NTIRE 2025 CD-FSOD Challenge, highlighting the proposed solutions and summarizing the results submitted by the participants. 

---
# Keyword Extraction, and Aspect Classification in Sinhala, English, and Code-Mixed Content 

**Authors**: F.A. Rizvi, T. Navojith, A.M.N.H. Adhikari, W.P.U. Senevirathna, Dharshana Kasthurirathna, Lakmini Abeywardhana  

**Link**: [PDF](https://arxiv.org/pdf/2504.10679)  

**Abstract**: Brand reputation in the banking sector is maintained through insightful analysis of customer opinion on code-mixed and multilingual content. Conventional NLP models misclassify or ignore code-mixed text, when mix with low resource languages such as Sinhala-English and fail to capture domain-specific knowledge. This study introduces a hybrid NLP method to improve keyword extraction, content filtering, and aspect-based classification of banking content. Keyword extraction in English is performed with a hybrid approach comprising a fine-tuned SpaCy NER model, FinBERT-based KeyBERT embeddings, YAKE, and EmbedRank, which results in a combined accuracy of 91.2%. Code-mixed and Sinhala keywords are extracted using a fine-tuned XLM-RoBERTa model integrated with a domain-specific Sinhala financial vocabulary, and it results in an accuracy of 87.4%. To ensure data quality, irrelevant comment filtering was performed using several models, with the BERT-base-uncased model achieving 85.2% for English and XLM-RoBERTa 88.1% for Sinhala, which was better than GPT-4o, SVM, and keyword-based filtering. Aspect classification followed the same pattern, with the BERT-base-uncased model achieving 87.4% for English and XLM-RoBERTa 85.9% for Sinhala, both exceeding GPT-4 and keyword-based approaches. These findings confirm that fine-tuned transformer models outperform traditional methods in multilingual financial text analysis. The present framework offers an accurate and scalable solution for brand reputation monitoring in code-mixed and low-resource banking environments. 

---
# Achieving Optimal Tissue Repair Through MARL with Reward Shaping and Curriculum Learning 

**Authors**: Muhammad Al-Zafar Khan, Jamal Al-Karaki  

**Link**: [PDF](https://arxiv.org/pdf/2504.10677)  

**Abstract**: In this paper, we present a multi-agent reinforcement learning (MARL) framework for optimizing tissue repair processes using engineered biological agents. Our approach integrates: (1) stochastic reaction-diffusion systems modeling molecular signaling, (2) neural-like electrochemical communication with Hebbian plasticity, and (3) a biologically informed reward function combining chemical gradient tracking, neural synchronization, and robust penalties. A curriculum learning scheme guides the agent through progressively complex repair scenarios. In silico experiments demonstrate emergent repair strategies, including dynamic secretion control and spatial coordination. 

---
# Characterizing Knowledge Manipulation in a Russian Wikipedia Fork 

**Authors**: Mykola Trokhymovych, Oleksandr Kosovan, Nathan Forrester, Pablo Aragón, Diego Saez-Trumper, Ricardo Baeza-Yates  

**Link**: [PDF](https://arxiv.org/pdf/2504.10663)  

**Abstract**: Wikipedia is powered by MediaWiki, a free and open-source software that is also the infrastructure for many other wiki-based online encyclopedias. These include the recently launched website Ruwiki, which has copied and modified the original Russian Wikipedia content to conform to Russian law. To identify practices and narratives that could be associated with different forms of knowledge manipulation, this article presents an in-depth analysis of this Russian Wikipedia fork. We propose a methodology to characterize the main changes with respect to the original version. The foundation of this study is a comprehensive comparative analysis of more than 1.9M articles from Russian Wikipedia and its fork. Using meta-information and geographical, temporal, categorical, and textual features, we explore the changes made by Ruwiki editors. Furthermore, we present a classification of the main topics of knowledge manipulation in this fork, including a numerical estimation of their scope. This research not only sheds light on significant changes within Ruwiki, but also provides a methodology that could be applied to analyze other Wikipedia forks and similar collaborative projects. 

---
# LITERA: An LLM Based Approach to Latin-to-English Translation 

**Authors**: Paul Rosu  

**Link**: [PDF](https://arxiv.org/pdf/2504.10660)  

**Abstract**: This paper introduces an LLM-based Latin-to-English translation platform designed to address the challenges of translating Latin texts. We named the model LITERA, which stands for Latin Interpretation and Translations into English for Research Assistance. Through a multi-layered translation process utilizing a fine-tuned version of GPT-4o-mini and GPT-4o, LITERA offers an unprecedented level of accuracy, showcased by greatly improved BLEU scores, particularly in classical Latin, along with improved BLEURT scores. The development of LITERA involved close collaboration with Duke University's Classical Studies Department, which was instrumental in creating a small, high-quality parallel Latin-English dataset. This paper details the architecture, fine-tuning methodology, and prompting strategies used in LITERA, emphasizing its ability to produce literal translations. 

---
# MatterTune: An Integrated, User-Friendly Platform for Fine-Tuning Atomistic Foundation Models to Accelerate Materials Simulation and Discovery 

**Authors**: Lingyu Kong, Nima Shoghi, Guoxiang Hu, Pan Li, Victor Fung  

**Link**: [PDF](https://arxiv.org/pdf/2504.10655)  

**Abstract**: Geometric machine learning models such as graph neural networks have achieved remarkable success in recent years in chemical and materials science research for applications such as high-throughput virtual screening and atomistic simulations. The success of these models can be attributed to their ability to effectively learn latent representations of atomic structures directly from the training data. Conversely, this also results in high data requirements for these models, hindering their application to problems which are data sparse which are common in this domain. To address this limitation, there is a growing development in the area of pre-trained machine learning models which have learned general, fundamental, geometric relationships in atomistic data, and which can then be fine-tuned to much smaller application-specific datasets. In particular, models which are pre-trained on diverse, large-scale atomistic datasets have shown impressive generalizability and flexibility to downstream applications, and are increasingly referred to as atomistic foundation models. To leverage the untapped potential of these foundation models, we introduce MatterTune, a modular and extensible framework that provides advanced fine-tuning capabilities and seamless integration of atomistic foundation models into downstream materials informatics and simulation workflows, thereby lowering the barriers to adoption and facilitating diverse applications in materials science. In its current state, MatterTune supports a number of state-of-the-art foundation models such as ORB, MatterSim, JMP, and EquformerV2, and hosts a wide range of features including a modular and flexible design, distributed and customizable fine-tuning, broad support for downstream informatics tasks, and more. 

---
# Will AI shape the way we speak? The emerging sociolinguistic influence of synthetic voices 

**Authors**: Éva Székely, Jūra Miniota, Míša, Hejná  

**Link**: [PDF](https://arxiv.org/pdf/2504.10650)  

**Abstract**: The growing prevalence of conversational voice interfaces, powered by developments in both speech and language technologies, raises important questions about their influence on human communication. While written communication can signal identity through lexical and stylistic choices, voice-based interactions inherently amplify socioindexical elements - such as accent, intonation, and speech style - which more prominently convey social identity and group affiliation. There is evidence that even passive media such as television is likely to influence the audience's linguistic patterns. Unlike passive media, conversational AI is interactive, creating a more immersive and reciprocal dynamic that holds a greater potential to impact how individuals speak in everyday interactions. Such heightened influence can be expected to arise from phenomena such as acoustic-prosodic entrainment and linguistic accommodation, which occur naturally during interaction and enable users to adapt their speech patterns in response to the system. While this phenomenon is still emerging, its potential societal impact could provide organisations, movements, and brands with a subtle yet powerful avenue for shaping and controlling public perception and social identity. We argue that the socioindexical influence of AI-generated speech warrants attention and should become a focus of interdisciplinary research, leveraging new and existing methodologies and technologies to better understand its implications. 

---
# Weight-of-Thought Reasoning: Exploring Neural Network Weights for Enhanced LLM Reasoning 

**Authors**: Saif Punjwani, Larry Heck  

**Link**: [PDF](https://arxiv.org/pdf/2504.10646)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable reasoning capabilities when prompted with strategies such as Chain-of-Thought (CoT). However, these approaches focus on token-level output without considering internal weight dynamics. We introduce Weight-of-Thought (WoT) reasoning, a novel approach that examines neural network weights before inference to identify reasoning pathways. Unlike existing methods, WoT explores the weight space through graph-based message passing, multi-step reasoning processes, and attention mechanisms. Our implementation creates an interconnected graph of reasoning nodes. Experiments on diverse reasoning tasks (syllogistic, mathematical, algebraic, combinatorial, and geometric) demonstrate that WoT achieves superior performance compared to traditional methods, particularly for complex problems. This approach leads to both improved performance and greater interpretability of the reasoning process, offering a promising direction for enhancing LLM reasoning capabilities. 

---
# Better Estimation of the KL Divergence Between Language Models 

**Authors**: Afra Amini, Tim Vieira, Ryan Cotterell  

**Link**: [PDF](https://arxiv.org/pdf/2504.10637)  

**Abstract**: Estimating the Kullback--Leibler (KL) divergence between language models has many applications, e.g., reinforcement learning from human feedback (RLHF), interpretability, and knowledge distillation. However, computing the exact KL divergence between two arbitrary language models is intractable. Thus, practitioners often resort to the use of sampling-based estimators. While it is easy to fashion a simple Monte Carlo (MC) estimator that provides an unbiased estimate of the KL divergence between language models, this estimator notoriously suffers from high variance, and can even result in a negative estimate of the KL divergence, a non-negative quantity. In this paper, we introduce a Rao--Blackwellized estimator that is also unbiased and provably has variance less than or equal to that of the standard Monte Carlo estimator. In an empirical study on sentiment-controlled fine-tuning, we show that our estimator provides more stable KL estimates and reduces variance substantially in practice. Additionally, we derive an analogous Rao--Blackwellized estimator of the gradient of the KL divergence, which leads to more stable training and produces models that more frequently appear on the Pareto frontier of reward vs. KL compared to the ones trained with the MC estimator of the gradient. 

---
# Who is More Bayesian: Humans or ChatGPT? 

**Authors**: Tianshi Mu, Pranjal Rawat, John Rust, Chengjun Zhang, Qixuan Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2504.10636)  

**Abstract**: We compare the performance of human and artificially intelligent (AI) decision makers in simple binary classification tasks where the optimal decision rule is given by Bayes Rule. We reanalyze choices of human subjects gathered from laboratory experiments conducted by El-Gamal and Grether and Holt and Smith. We confirm that while overall, Bayes Rule represents the single best model for predicting human choices, subjects are heterogeneous and a significant share of them make suboptimal choices that reflect judgement biases described by Kahneman and Tversky that include the ``representativeness heuristic'' (excessive weight on the evidence from the sample relative to the prior) and ``conservatism'' (excessive weight on the prior relative to the sample). We compare the performance of AI subjects gathered from recent versions of large language models (LLMs) including several versions of ChatGPT. These general-purpose generative AI chatbots are not specifically trained to do well in narrow decision making tasks, but are trained instead as ``language predictors'' using a large corpus of textual data from the web. We show that ChatGPT is also subject to biases that result in suboptimal decisions. However we document a rapid evolution in the performance of ChatGPT from sub-human performance for early versions (ChatGPT 3.5) to superhuman and nearly perfect Bayesian classifications in the latest versions (ChatGPT 4o). 

---
# Energy Matching: Unifying Flow Matching and Energy-Based Models for Generative Modeling 

**Authors**: Michal Balcerak, Tamaz Amiranashvili, Suprosanna Shit, Antonio Terpin, Sebastian Kaltenbach, Petros Koumoutsakos, Bjoern Menze  

**Link**: [PDF](https://arxiv.org/pdf/2504.10612)  

**Abstract**: Generative models often map noise to data by matching flows or scores, but these approaches become cumbersome for incorporating partial observations or additional priors. Inspired by recent advances in Wasserstein gradient flows, we propose Energy Matching, a framework that unifies flow-based approaches with the flexibility of energy-based models (EBMs). Far from the data manifold, samples move along curl-free, optimal transport paths from noise to data. As they approach the data manifold, an entropic energy term guides the system into a Boltzmann equilibrium distribution, explicitly capturing the underlying likelihood structure of the data. We parameterize this dynamic with a single time-independent scalar field, which serves as both a powerful generator and a flexible prior for effective regularization of inverse problems. Our method substantially outperforms existing EBMs on CIFAR-10 generation (FID 3.97 compared to 8.61), while retaining the simulation-free training of transport-based approaches away from the data manifold. Additionally, we exploit the flexibility of our method and introduce an interaction energy for diverse mode exploration. Our approach focuses on learning a static scalar potential energy -- without time conditioning, auxiliary generators, or additional networks -- marking a significant departure from recent EBM methods. We believe this simplified framework significantly advances EBM capabilities and paves the way for their broader adoption in generative modeling across diverse domains. 

---
# Visual anemometry of natural vegetation from their leaf motion 

**Authors**: Roni H. Goldshmid, John O. Dabiri, John E. Sader  

**Link**: [PDF](https://arxiv.org/pdf/2504.10584)  

**Abstract**: High-resolution, near-ground wind-speed data are critical for improving the accuracy of weather predictions and climate models,$^{1-3}$ supporting wildfire control efforts,$^{4-7}$ and ensuring the safe passage of airplanes during takeoff and landing maneouvers.$^{8,9}$ Quantitative wind speed anemometry generally employs on-site instrumentation for accurate single-position data or sophisticated remote techniques such as Doppler radar for quantitative field measurements. It is widely recognized that the wind-induced motion of vegetation depends in a complex manner on their structure and mechanical properties, obviating their use in quantitative anemometry.$^{10-14}$ We analyze measurements on a host of different vegetation showing that leaf motion can be decoupled from the leaf's branch and support structure, at low-to-moderate wind speed, $U_{wind}$. This wind speed range is characterized by a leaf Reynolds number, enabling the development of a remote, quantitative anemometry method based on the formula, $U_{wind}\approx740\sqrt{{\mu}U_{leaf}/{\rho}D}$, that relies only on the leaf size $D$, its measured fluctuating (RMS) speed $U_{leaf}$, the air viscosity $\mu$, and its mass density $\rho$. This formula is corroborated by a first-principles model and validated using a host of laboratory and field tests on diverse vegetation types, ranging from oak, olive, and magnolia trees through to camphor and bullgrass. The findings of this study open the door to a new paradigm in anemometry, using natural vegetation to enable remote and rapid quantitative field measurements at global locations with minimal cost. 

---
# Self-Controlled Dynamic Expansion Model for Continual Learning 

**Authors**: Runqing Wu, Fei Ye, Rongyao Hu, Guoxi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.10561)  

**Abstract**: Continual Learning (CL) epitomizes an advanced training paradigm wherein prior data samples remain inaccessible during the acquisition of new tasks. Numerous investigations have delved into leveraging a pre-trained Vision Transformer (ViT) to enhance model efficacy in continual learning. Nonetheless, these approaches typically utilize a singular, static backbone, which inadequately adapts to novel tasks, particularly when engaging with diverse data domains, due to a substantial number of inactive parameters. This paper addresses this limitation by introducing an innovative Self-Controlled Dynamic Expansion Model (SCDEM), which orchestrates multiple distinct trainable pre-trained ViT backbones to furnish diverse and semantically enriched representations. Specifically, by employing the multi-backbone architecture as a shared module, the proposed SCDEM dynamically generates a new expert with minimal parameters to accommodate a new task. A novel Collaborative Optimization Mechanism (COM) is introduced to synergistically optimize multiple backbones by harnessing prediction signals from historical experts, thereby facilitating new task learning without erasing previously acquired knowledge. Additionally, a novel Feature Distribution Consistency (FDC) approach is proposed to align semantic similarity between previously and currently learned representations through an optimal transport distance-based mechanism, effectively mitigating negative knowledge transfer effects. Furthermore, to alleviate over-regularization challenges, this paper presents a novel Dynamic Layer-Wise Feature Attention Mechanism (DLWFAM) to autonomously determine the penalization intensity on each trainable representation layer. An extensive series of experiments have been conducted to evaluate the proposed methodology's efficacy, with empirical results corroborating that the approach attains state-of-the-art performance. 

---
# Efficient Process Reward Model Training via Active Learning 

**Authors**: Keyu Duan, Zichen Liu, Xin Mao, Tianyu Pang, Changyu Chen, Qiguang Chen, Michael Qizhe Shieh, Longxu Dou  

**Link**: [PDF](https://arxiv.org/pdf/2504.10559)  

**Abstract**: Process Reward Models (PRMs) provide step-level supervision to large language models (LLMs), but scaling up training data annotation remains challenging for both humans and LLMs. To address this limitation, we propose an active learning approach, ActPRM, which proactively selects the most uncertain samples for training, substantially reducing labeling costs. During training, we use the PRM to estimate uncertainty after the forward pass, retaining only highly uncertain data. A capable yet costly reasoning model then labels this data. Then we compute the loss with respect to the labels and update the PRM's weights. We compare ActPRM vs. vanilla fine-tuning, on a pool-based active learning setting, demonstrating that ActPRM reduces 50% annotation, but achieving the comparable or even better performance. Beyond annotation efficiency, we further advance the actively trained PRM by filtering over 1M+ math reasoning trajectories with ActPRM, retaining 60% of the data. A subsequent training on this selected dataset yields a new state-of-the-art (SOTA) PRM on ProcessBench (75.0%) and PRMBench (65.5%) compared with same sized models. 

---
# The Code Barrier: What LLMs Actually Understand? 

**Authors**: Serge Lionel Nikiema, Jordan Samhi, Abdoul Kader Kaboré, Jacques Klein, Tegawendé F. Bissyandé  

**Link**: [PDF](https://arxiv.org/pdf/2504.10557)  

**Abstract**: Understanding code represents a core ability needed for automating software development tasks. While foundation models like LLMs show impressive results across many software engineering challenges, the extent of their true semantic understanding beyond simple token recognition remains unclear. This research uses code obfuscation as a structured testing framework to evaluate LLMs' semantic understanding capabilities. We methodically apply controlled obfuscation changes to source code and measure comprehension through two complementary tasks: generating accurate descriptions of obfuscated code and performing deobfuscation, a skill with important implications for reverse engineering applications.
Our testing approach includes 13 cutting-edge models, covering both code-specialized (e.g., StarCoder2) and general-purpose (e.g., GPT-4o) architectures, evaluated on a benchmark created from CodeNet and consisting of filtered 250 Java programming problems and their solutions. Findings show a statistically significant performance decline as obfuscation complexity increases, with unexpected resilience shown by general-purpose models compared to their code-focused counterparts. While some models successfully identify obfuscation techniques, their ability to reconstruct the underlying program logic remains constrained, suggesting limitations in their semantic representation mechanisms. This research introduces a new evaluation approach for assessing code comprehension in language models and establishes empirical baselines for advancing research in security-critical code analysis applications such as reverse engineering and adversarial code analysis. 

---
# VAE-based Feature Disentanglement for Data Augmentation and Compression in Generalized GNSS Interference Classification 

**Authors**: Lucas Heublein, Simon Kocher, Tobias Feigl, Alexander Rügamer, Christopher Mutschler, Felix Ott  

**Link**: [PDF](https://arxiv.org/pdf/2504.10556)  

**Abstract**: Distributed learning and Edge AI necessitate efficient data processing, low-latency communication, decentralized model training, and stringent data privacy to facilitate real-time intelligence on edge devices while reducing dependency on centralized infrastructure and ensuring high model performance. In the context of global navigation satellite system (GNSS) applications, the primary objective is to accurately monitor and classify interferences that degrade system performance in distributed environments, thereby enhancing situational awareness. To achieve this, machine learning (ML) models can be deployed on low-resource devices, ensuring minimal communication latency and preserving data privacy. The key challenge is to compress ML models while maintaining high classification accuracy. In this paper, we propose variational autoencoders (VAEs) for disentanglement to extract essential latent features that enable accurate classification of interferences. We demonstrate that the disentanglement approach can be leveraged for both data compression and data augmentation by interpolating the lower-dimensional latent representations of signal power. To validate our approach, we evaluate three VAE variants - vanilla, factorized, and conditional generative - on four distinct datasets, including two collected in controlled indoor environments and two real-world highway datasets. Additionally, we conduct extensive hyperparameter searches to optimize performance. Our proposed VAE achieves a data compression rate ranging from 512 to 8,192 and achieves an accuracy up to 99.92%. 

---
# Beyond the Generative Learning Trilemma: Generative Model Assessment in Data Scarcity Domains 

**Authors**: Marco Salmè, Lorenzo Tronchin, Rosa Sicilia, Paolo Soda, Valerio Guarrasi  

**Link**: [PDF](https://arxiv.org/pdf/2504.10555)  

**Abstract**: Data scarcity remains a critical bottleneck impeding technological advancements across various domains, including but not limited to medicine and precision agriculture. To address this challenge, we explore the potential of Deep Generative Models (DGMs) in producing synthetic data that satisfies the Generative Learning Trilemma: fidelity, diversity, and sampling efficiency. However, recognizing that these criteria alone are insufficient for practical applications, we extend the trilemma to include utility, robustness, and privacy, factors crucial for ensuring the applicability of DGMs in real-world scenarios. Evaluating these metrics becomes particularly challenging in data-scarce environments, as DGMs traditionally rely on large datasets to perform optimally. This limitation is especially pronounced in domains like medicine and precision agriculture, where ensuring acceptable model performance under data constraints is vital. To address these challenges, we assess the Generative Learning Trilemma in data-scarcity settings using state-of-the-art evaluation metrics, comparing three prominent DGMs: Variational Autoencoders (VAEs), Generative Adversarial Networks (GANs), and Diffusion Models (DMs). Furthermore, we propose a comprehensive framework to assess utility, robustness, and privacy in synthetic data generated by DGMs. Our findings demonstrate varying strengths among DGMs, with each model exhibiting unique advantages based on the application context. This study broadens the scope of the Generative Learning Trilemma, aligning it with real-world demands and providing actionable guidance for selecting DGMs tailored to specific applications. 

---
# LEMUR Neural Network Dataset: Towards Seamless AutoML 

**Authors**: Arash Torabi Goodarzi, Roman Kochnev, Waleed Khalid, Furui Qin, Tolgay Atinc Uzun, Yashkumar Sanjaybhai Dhameliya, Yash Kanubhai Kathiriya, Zofia Antonina Bentyn, Dmitry Ignatov, Radu Timofte  

**Link**: [PDF](https://arxiv.org/pdf/2504.10552)  

**Abstract**: Neural networks are fundamental in artificial intelligence, driving progress in computer vision and natural language processing. High-quality datasets are crucial for their development, and there is growing interest in datasets composed of neural networks themselves to support benchmarking, automated machine learning (AutoML), and model analysis. We introduce LEMUR, an open source dataset of neural network models with well-structured code for diverse architectures across tasks such as object detection, image classification, segmentation, and natural language processing. LEMUR is primarily designed to enable fine-tuning of large language models (LLMs) for AutoML tasks, providing a rich source of structured model representations and associated performance data. Leveraging Python and PyTorch, LEMUR enables seamless extension to new datasets and models while maintaining consistency. It integrates an Optuna-powered framework for evaluation, hyperparameter optimization, statistical analysis, and graphical insights. LEMUR provides an extension that enables models to run efficiently on edge devices, facilitating deployment in resource-constrained environments. Providing tools for model evaluation, preprocessing, and database management, LEMUR supports researchers and practitioners in developing, testing, and analyzing neural networks. Additionally, it offers an API that delivers comprehensive information about neural network models and their complete performance statistics with a single request, which can be used in experiments with code-generating large language models. The LEMUR will be released as an open source project under the MIT license upon acceptance of the paper. 

---
# MiMu: Mitigating Multiple Shortcut Learning Behavior of Transformers 

**Authors**: Lili Zhao, Qi Liu, Wei Chen, Liyi Chen, Ruijun Sun, Min Hou, Yang Wang, Shijin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.10551)  

**Abstract**: Empirical Risk Minimization (ERM) models often rely on spurious correlations between features and labels during the learning process, leading to shortcut learning behavior that undermines robustness generalization performance. Current research mainly targets identifying or mitigating a single shortcut; however, in real-world scenarios, cues within the data are diverse and unknown. In empirical studies, we reveal that the models rely to varying extents on different shortcuts. Compared to weak shortcuts, models depend more heavily on strong shortcuts, resulting in their poor generalization ability. To address these challenges, we propose MiMu, a novel method integrated with Transformer-based ERMs designed to Mitigate Multiple shortcut learning behavior, which incorporates self-calibration strategy and self-improvement strategy. In the source model, we preliminarily propose the self-calibration strategy to prevent the model from relying on shortcuts and make overconfident predictions. Then, we further design self-improvement strategy in target model to reduce the reliance on multiple shortcuts. The random mask strategy involves randomly masking partial attention positions to diversify the focus of target model other than concentrating on a fixed region. Meanwhile, the adaptive attention alignment module facilitates the alignment of attention weights to the calibrated source model, without the need for post-hoc attention maps or supervision. Finally, extensive experiments conducted on Natural Language Processing (NLP) and Computer Vision (CV) demonstrate the effectiveness of MiMu in improving robustness generalization abilities. 

---
# Automated Testing of COBOL to Java Transformation 

**Authors**: Sandeep Hans, Atul Kumar, Toshikai Yasue, Kouichi Ono, Saravanan Krishnan, Devika Sondhi, Fumiko Satoh, Gerald Mitchell, Sachin Kumar, Diptikalyan Saha  

**Link**: [PDF](https://arxiv.org/pdf/2504.10548)  

**Abstract**: Recent advances in Large Language Model (LLM) based Generative AI techniques have made it feasible to translate enterprise-level code from legacy languages such as COBOL to modern languages such as Java or Python. While the results of LLM-based automatic transformation are encouraging, the resulting code cannot be trusted to correctly translate the original code, making manual validation of translated Java code from COBOL a necessary but time-consuming and labor-intensive process. In this paper, we share our experience of developing a testing framework for IBM Watsonx Code Assistant for Z (WCA4Z) [5], an industrial tool designed for COBOL to Java translation. The framework automates the process of testing the functional equivalence of the translated Java code against the original COBOL programs in an industry context. Our framework uses symbolic execution to generate unit tests for COBOL, mocking external calls and transforming them into JUnit tests to validate semantic equivalence with translated Java. The results not only help identify and repair any detected discrepancies but also provide feedback to improve the AI model. 

---
# Multi-Modal Hypergraph Enhanced LLM Learning for Recommendation 

**Authors**: Xu Guo, Tong Zhang, Yuanzhi Wang, Chenxu Wang, Fuyun Wang, Xudong Wang, Xiaoya Zhang, Xin Liu, Zhen Cui  

**Link**: [PDF](https://arxiv.org/pdf/2504.10541)  

**Abstract**: The burgeoning presence of Large Language Models (LLM) is propelling the development of personalized recommender systems. Most existing LLM-based methods fail to sufficiently explore the multi-view graph structure correlations inherent in recommendation scenarios. To this end, we propose a novel framework, Hypergraph Enhanced LLM Learning for multimodal Recommendation (HeLLM), designed to equip LLMs with the capability to capture intricate higher-order semantic correlations by fusing graph-level contextual signals with sequence-level behavioral patterns. In the recommender pre-training phase, we design a user hypergraph to uncover shared interest preferences among users and an item hypergraph to capture correlations within multimodal similarities among items. The hypergraph convolution and synergistic contrastive learning mechanism are introduced to enhance the distinguishability of learned representations. In the LLM fine-tuning phase, we inject the learned graph-structured embeddings directly into the LLM's architecture and integrate sequential features capturing each user's chronological behavior. This process enables hypergraphs to leverage graph-structured information as global context, enhancing the LLM's ability to perceive complex relational patterns and integrate multimodal information, while also modeling local temporal dynamics. Extensive experiments demonstrate the superiority of our proposed method over state-of-the-art baselines, confirming the advantages of fusing hypergraph-based context with sequential user behavior in LLMs for recommendation. 

---
# AB-Cache: Training-Free Acceleration of Diffusion Models via Adams-Bashforth Cached Feature Reuse 

**Authors**: Zichao Yu, Zhen Zou, Guojiang Shao, Chengwei Zhang, Shengze Xu, Jie Huang, Feng Zhao, Xiaodong Cun, Wenyi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.10540)  

**Abstract**: Diffusion models have demonstrated remarkable success in generative tasks, yet their iterative denoising process results in slow inference, limiting their practicality. While existing acceleration methods exploit the well-known U-shaped similarity pattern between adjacent steps through caching mechanisms, they lack theoretical foundation and rely on simplistic computation reuse, often leading to performance degradation. In this work, we provide a theoretical understanding by analyzing the denoising process through the second-order Adams-Bashforth method, revealing a linear relationship between the outputs of consecutive steps. This analysis explains why the outputs of adjacent steps exhibit a U-shaped pattern. Furthermore, extending Adams-Bashforth method to higher order, we propose a novel caching-based acceleration approach for diffusion models, instead of directly reusing cached results, with a truncation error bound of only \(O(h^k)\) where $h$ is the step size. Extensive validation across diverse image and video diffusion models (including HunyuanVideo and FLUX.1-dev) with various schedulers demonstrates our method's effectiveness in achieving nearly $3\times$ speedup while maintaining original performance levels, offering a practical real-time solution without compromising generation quality. 

---
# Physics-Informed Neural Networks for Enhanced Interface Preservation in Lattice Boltzmann Multiphase Simulations 

**Authors**: Yue Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.10539)  

**Abstract**: This paper presents an improved approach for preserving sharp interfaces in multiphase Lattice Boltzmann Method (LBM) simulations using Physics-Informed Neural Networks (PINNs). Interface diffusion is a common challenge in multiphase LBM, leading to reduced accuracy in simulating phenomena where interfacial dynamics are critical. We propose a coupled PINN-LBM framework that maintains interface sharpness while preserving the physical accuracy of the simulation. Our approach is validated through droplet simulations, with quantitative metrics measuring interface width, maximum gradient, phase separation, effective interface width, and interface energy. The enhanced visualization techniques employed in this work clearly demonstrate the superior performance of PINN-LBM over standard LBM for multiphase simulations, particularly in maintaining well-defined interfaces throughout the simulation. We provide a comprehensive analysis of the results, showcasing how the neural network integration effectively counteracts numerical diffusion, while maintaining physical consistency with the underlying fluid dynamics. 

---
# Distilling Transitional Pattern to Large Language Models for Multimodal Session-based Recommendation 

**Authors**: Jiajie Su, Qiyong Zhong, Yunshan Ma, Weiming Liu, Chaochao Chen, Xiaolin Zheng, Jianwei Yin, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2504.10538)  

**Abstract**: Session-based recommendation (SBR) predicts the next item based on anonymous sessions. Traditional SBR explores user intents based on ID collaborations or auxiliary content. To further alleviate data sparsity and cold-start issues, recent Multimodal SBR (MSBR) methods utilize simplistic pre-trained models for modality learning but have limitations in semantic richness. Considering semantic reasoning abilities of Large Language Models (LLM), we focus on the LLM-enhanced MSBR scenario in this paper, which leverages LLM cognition for comprehensive multimodal representation generation, to enhance downstream MSBR. Tackling this problem faces two challenges: i) how to obtain LLM cognition on both transitional patterns and inherent multimodal knowledge, ii) how to align both features into one unified LLM, minimize discrepancy while maximizing representation utility. To this end, we propose a multimodal LLM-enhanced framework TPAD, which extends a distillation paradigm to decouple and align transitional patterns for promoting MSBR. TPAD establishes parallel Knowledge-MLLM and Transfer-MLLM, where the former interprets item knowledge-reflected features and the latter extracts transition-aware features underneath sessions. A transitional pattern alignment module harnessing mutual information estimation theory unites two MLLMs, alleviating distribution discrepancy and distilling transitional patterns into modal representations. Extensive experiments on real-world datasets demonstrate the effectiveness of our framework. 

---
# Federated Learning with Layer Skipping: Efficient Training of Large Language Models for Healthcare NLP 

**Authors**: Lihong Zhang, Yue Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.10536)  

**Abstract**: Federated learning (FL) enables collaborative model training across organizations without sharing raw data, addressing crucial privacy concerns in healthcare natural language processing (NLP). However, training large language models (LLMs) in federated settings faces significant challenges, including communication overhead and data heterogeneity. We propose Layer-Skipping Federated Learning, where only selected layers of a pre-trained LLM are fine-tuned across clients while others remain frozen. Applied to LLaMA 3.2-1B, our approach reduces communication costs by approximately 70% while maintaining performance within 2% of centralized training. We evaluate our method on clinical NER and classification tasks using i2b2 and MIMIC-III datasets. Our experiments demonstrate that Layer-Skipping FL outperforms competitive baselines, handles non-IID clinical data distributions effectively, and shows robustness when combined with differential privacy. This approach represents a practical solution for privacy-preserving collaborative learning in healthcare NLP. 

---
# HeteRAG: A Heterogeneous Retrieval-augmented Generation Framework with Decoupled Knowledge Representations 

**Authors**: Peiru Yang, Xintian Li, Zhiyang Hu, Jiapeng Wang, Jinhua Yin, Huili Wang, Lizhi He, Shuai Yang, Shangguang Wang, Yongfeng Huang, Tao Qi  

**Link**: [PDF](https://arxiv.org/pdf/2504.10529)  

**Abstract**: Retrieval-augmented generation (RAG) methods can enhance the performance of LLMs by incorporating retrieved knowledge chunks into the generation process. In general, the retrieval and generation steps usually have different requirements for these knowledge chunks. The retrieval step benefits from comprehensive information to improve retrieval accuracy, whereas excessively long chunks may introduce redundant contextual information, thereby diminishing both the effectiveness and efficiency of the generation process. However, existing RAG methods typically employ identical representations of knowledge chunks for both retrieval and generation, resulting in suboptimal performance. In this paper, we propose a heterogeneous RAG framework (\myname) that decouples the representations of knowledge chunks for retrieval and generation, thereby enhancing the LLMs in both effectiveness and efficiency. Specifically, we utilize short chunks to represent knowledge to adapt the generation step and utilize the corresponding chunk with its contextual information from multi-granular views to enhance retrieval accuracy. We further introduce an adaptive prompt tuning method for the retrieval model to adapt the heterogeneous retrieval augmented generation process. Extensive experiments demonstrate that \myname achieves significant improvements compared to baselines. 

---
# Integrating Emotion Distribution Networks and Textual Message Analysis for X User Emotional State Classification 

**Authors**: Pardis Moradbeiki, Mohammad Ali Zare Chahooki  

**Link**: [PDF](https://arxiv.org/pdf/2504.10521)  

**Abstract**: As the popularity and reach of social networks continue to surge, a vast reservoir of opinions and sentiments across various subjects inundates these platforms. Among these, X social network (formerly Twitter) stands as a juggernaut, boasting approximately 420 million active users. Extracting users' emotional and mental states from their expressed opinions on social media has become a common pursuit. While past methodologies predominantly focused on the textual content of messages to analyze user sentiment, the interactive nature of these platforms suggests a deeper complexity. This study employs hybrid methodologies, integrating textual analysis, profile examination, follower analysis, and emotion dissemination patterns. Initially, user interactions are leveraged to refine emotion classification within messages, encompassing exchanges where users respond to each other. Introducing the concept of a communication tree, a model is extracted to map these interactions. Subsequently, users' bios and interests from this tree are juxtaposed with message text to enrich analysis. Finally, influential figures are identified among users' followers in the communication tree, categorized into different topics to gauge interests. The study highlights that traditional sentiment analysis methodologies, focusing solely on textual content, are inadequate in discerning sentiment towards significant events, notably the presidential election. Comparative analysis with conventional methods reveals a substantial improvement in accuracy with the incorporation of emotion distribution patterns and user profiles. The proposed approach yields a 12% increase in accuracy with emotion distribution patterns and a 15% increase when considering user profiles, underscoring its efficacy in capturing nuanced sentiment dynamics. 

---
# ColorBench: Can VLMs See and Understand the Colorful World? A Comprehensive Benchmark for Color Perception, Reasoning, and Robustness 

**Authors**: Yijun Liang, Ming Li, Chenrui Fan, Ziyue Li, Dang Nguyen, Kwesi Cobbina, Shweta Bhardwaj, Jiuhai Chen, Fuxiao Liu, Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.10514)  

**Abstract**: Color plays an important role in human perception and usually provides critical clues in visual reasoning. However, it is unclear whether and how vision-language models (VLMs) can perceive, understand, and leverage color as humans. This paper introduces ColorBench, an innovative benchmark meticulously crafted to assess the capabilities of VLMs in color understanding, including color perception, reasoning, and robustness. By curating a suite of diverse test scenarios, with grounding in real applications, ColorBench evaluates how these models perceive colors, infer meanings from color-based cues, and maintain consistent performance under varying color transformations. Through an extensive evaluation of 32 VLMs with varying language models and vision encoders, our paper reveals some undiscovered findings: (i) The scaling law (larger models are better) still holds on ColorBench, while the language model plays a more important role than the vision encoder. (ii) However, the performance gaps across models are relatively small, indicating that color understanding has been largely neglected by existing VLMs. (iii) CoT reasoning improves color understanding accuracies and robustness, though they are vision-centric tasks. (iv) Color clues are indeed leveraged by VLMs on ColorBench but they can also mislead models in some tasks. These findings highlight the critical limitations of current VLMs and underscore the need to enhance color comprehension. Our ColorBenchcan serve as a foundational tool for advancing the study of human-level color understanding of multimodal AI. 

---
# JEPA4Rec: Learning Effective Language Representations for Sequential Recommendation via Joint Embedding Predictive Architecture 

**Authors**: Minh-Anh Nguyen, Dung D.Le  

**Link**: [PDF](https://arxiv.org/pdf/2504.10512)  

**Abstract**: Language representation learning has emerged as a promising approach for sequential recommendation, thanks to its ability to learn generalizable representations. However, despite its advantages, this approach still struggles with data sparsity and a limited understanding of common-sense user preferences. To address these limitations, we propose $\textbf{JEPA4Rec}$, a framework that combines $\textbf{J}$oint $\textbf{E}$mbedding $\textbf{P}$redictive $\textbf{A}$rchitecture with language modeling of item textual descriptions. JEPA4Rec captures semantically rich and transferable representations, improving recommendation performance and reducing reliance on large-scale pre-training data. Specifically, JEPA4Rec represents items as text sentences by flattening descriptive information such as $\textit{title, category}$, and other attributes. To encode these sentences, we employ a bidirectional Transformer encoder with modified embedding layers tailored for capturing item information in recommendation datasets. We apply masking to text sentences and use them to predict the representations of the unmasked sentences, helping the model learn generalizable item embeddings. To further improve recommendation performance and language understanding, we employ a two-stage training strategy incorporating self-supervised learning losses. Experiments on six real-world datasets demonstrate that JEPA4Rec consistently outperforms state-of-the-art methods, particularly in cross-domain, cross-platform, and low-resource scenarios. 

---
# Beyond Reproducibility: Advancing Zero-shot LLM Reranking Efficiency with Setwise Insertion 

**Authors**: Jakub Podolak, Leon Peric, Mina Janicijevic, Roxana Petcu  

**Link**: [PDF](https://arxiv.org/pdf/2504.10509)  

**Abstract**: This study presents a comprehensive reproducibility and extension analysis of the Setwise prompting methodology for zero-shot ranking with Large Language Models (LLMs), as proposed by Zhuang et al. We evaluate its effectiveness and efficiency compared to traditional Pointwise, Pairwise, and Listwise approaches in document ranking tasks. Our reproduction confirms the findings of Zhuang et al., highlighting the trade-offs between computational efficiency and ranking effectiveness in Setwise methods. Building on these insights, we introduce Setwise Insertion, a novel approach that leverages the initial document ranking as prior knowledge, reducing unnecessary comparisons and uncertainty by focusing on candidates more likely to improve the ranking results. Experimental results across multiple LLM architectures (Flan-T5, Vicuna, and Llama) show that Setwise Insertion yields a 31% reduction in query time, a 23% reduction in model inferences, and a slight improvement in reranking effectiveness compared to the original Setwise method. These findings highlight the practical advantage of incorporating prior ranking knowledge into Setwise prompting for efficient and accurate zero-shot document reranking. 

---
# Poly-Vector Retrieval: Reference and Content Embeddings for Legal Documents 

**Authors**: João Alberto de Oliveira Lima  

**Link**: [PDF](https://arxiv.org/pdf/2504.10508)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as an effective paradigm for generating contextually accurate answers by integrating Large Language Models (LLMs) with retrieval mechanisms. However, in legal contexts, users frequently reference norms by their labels or nicknames (e.g., Article 5 of the Constitution or Consumer Defense Code (CDC)), rather than by their content, posing challenges for traditional RAG approaches that rely solely on semantic embeddings of text. Furthermore, legal texts themselves heavily rely on explicit cross-references (e.g., "pursuant to Article 34") that function as pointers. Both scenarios pose challenges for traditional RAG approaches that rely solely on semantic embeddings of text, often failing to retrieve the necessary referenced content. This paper introduces Poly-Vector Retrieval, a method assigning multiple distinct embeddings to each legal provision: one embedding captures the content (the full text), another captures the label (the identifier or proper name), and optionally additional embeddings capture alternative denominations. Inspired by Frege's distinction between Sense and Reference, this poly-vector retrieval approach treats labels, identifiers and reference markers as rigid designators and content embeddings as carriers of semantic substance. Experiments on the Brazilian Federal Constitution demonstrate that Poly-Vector Retrieval significantly improves retrieval accuracy for label-centric queries and potential to resolve internal and external cross-references, without compromising performance on purely semantic queries. The study discusses philosophical and practical implications of explicitly separating reference from content in vector embeddings and proposes future research directions for applying this approach to broader legal datasets and other domains characterized by explicit reference identifiers. 

---
# Leveraging Auto-Distillation and Generative Self-Supervised Learning in Residual Graph Transformers for Enhanced Recommender Systems 

**Authors**: Eya Mhedhbi, Youssef Mourchid, Alice Othmani  

**Link**: [PDF](https://arxiv.org/pdf/2504.10500)  

**Abstract**: This paper introduces a cutting-edge method for enhancing recommender systems through the integration of generative self-supervised learning (SSL) with a Residual Graph Transformer. Our approach emphasizes the importance of superior data enhancement through the use of pertinent pretext tasks, automated through rationale-aware SSL to distill clear ways of how users and items interact. The Residual Graph Transformer incorporates a topology-aware transformer for global context and employs residual connections to improve graph representation learning. Additionally, an auto-distillation process refines self-supervised signals to uncover consistent collaborative rationales. Experimental evaluations on multiple datasets demonstrate that our approach consistently outperforms baseline methods. 

---
# CCSK:Cognitive Convection of Self-Knowledge Based Retrieval Augmentation for Large Language Models 

**Authors**: Jianling Lu, Mingqi Lv  

**Link**: [PDF](https://arxiv.org/pdf/2504.10498)  

**Abstract**: The performance of large language models (LLMs) in Q&A task increased substantially through Retrieval-Augmented Generation (RAG) which brings in external knowledge. However, the main difficulty lies in balancing the inherent self-knowledge of LLMs with external information retrieval (IR). The current threshold-based methods apply one-dimensional static mechanisms with single criterion. As a result, their IR decisions might be irrelevant to the LLMs' response under difficult queries. To alleviate this problem, we propose Cognitive Convection of Self-Knowledge (CCSK). Different from traditional methods that maintain single fixed IR activation criteria, CCSK implements a dynamic joint decision process via a Siamese Network module and a Response Quality Model. The Siamese Network calculates the cosine similarity between the current query and the historical queries. The Response Quality Model evaluates the responses of LLMs through LightGBM. The final decision of the CCSK is derived from the outputs of the two modules, as well as text features fused using a multi-head attention mechanism. Extensive experiments on real-world datasets show that CCSK significantly enhances the model's effectiveness in information retrieval. 

---
# Exploring Generative AI Techniques in Government: A Case Study 

**Authors**: Sunyi Liu, Mengzhe Geng, Rebecca Hart  

**Link**: [PDF](https://arxiv.org/pdf/2504.10497)  

**Abstract**: The swift progress of Generative Artificial intelligence (GenAI), notably Large Language Models (LLMs), is reshaping the digital landscape. Recognizing this transformative potential, the National Research Council of Canada (NRC) launched a pilot initiative to explore the integration of GenAI techniques into its daily operation for performance excellence, where 22 projects were launched in May 2024. Within these projects, this paper presents the development of the intelligent agent Pubbie as a case study, targeting the automation of performance measurement, data management and insight reporting at the NRC. Cutting-edge techniques are explored, including LLM orchestration and semantic embedding via RoBERTa, while strategic fine-tuning and few-shot learning approaches are incorporated to infuse domain knowledge at an affordable cost. The user-friendly interface of Pubbie allows general government users to input queries in natural language and easily upload or download files with a simple button click, greatly reducing manual efforts and accessibility barriers. 

---
# ArxivBench: Can LLMs Assist Researchers in Conducting Research? 

**Authors**: Ning Li, Jingran Zhang, Justin Cui  

**Link**: [PDF](https://arxiv.org/pdf/2504.10496)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable effectiveness in completing various tasks such as reasoning, translation, and question answering. However the issue of factual incorrect content in LLM-generated responses remains a persistent challenge. In this study, we evaluate both proprietary and open-source LLMs on their ability to respond with relevant research papers and accurate links to articles hosted on the arXiv platform, based on high level prompts. To facilitate this evaluation, we introduce arXivBench, a benchmark specifically designed to assess LLM performance across eight major subject categories on arXiv and five subfields within computer science, one of the most popular categories among them. Our findings reveal a concerning accuracy of LLM-generated responses depending on the subject, with some subjects experiencing significantly lower accuracy than others. Notably, Claude-3.5-Sonnet exhibits a substantial advantage in generating both relevant and accurate responses. And interestingly, most LLMs achieve a much higher accuracy in the Artificial Intelligence sub-field than other sub-fields. This benchmark provides a standardized tool for evaluating the reliability of LLM-generated scientific responses, promoting more dependable use of LLMs in academic and research environments. Our code is open-sourced at this https URL and our dataset is available on huggingface at this https URL. 

---
# Roamify: Designing and Evaluating an LLM Based Google Chrome Extension for Personalised Itinerary Planning 

**Authors**: Vikranth Udandarao, Noel Abraham Tiju, Muthuraj Vairamuthu, Harsh Mistry, Dhruv Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2504.10489)  

**Abstract**: In this paper, we present Roamify, an Artificial Intelligence powered travel assistant that aims to ease the process of travel planning. We have tested and used multiple Large Language Models like Llama and T5 to generate personalised itineraries per user preferences. Results from user surveys highlight the preference for AI powered mediums over existing methods to help in travel planning across all user age groups. These results firmly validate the potential need of such a travel assistant. We highlight the two primary design considerations for travel assistance: D1) incorporating a web-scraping method to gather up-to-date news articles about destinations from various blog sources, which significantly improves our itinerary suggestions, and D2) utilising user preferences to create customised travel experiences along with a recommendation system which changes the itinerary according to the user needs. Our findings suggest that Roamify has the potential to improve and simplify how users across multiple age groups plan their travel experiences. 

---
# EthosGPT: Mapping Human Value Diversity to Advance Sustainable Development Goals (SDGs) 

**Authors**: Luyao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.09861)  

**Abstract**: Large language models (LLMs) are transforming global decision-making and societal systems by processing diverse data at unprecedented scales. However, their potential to homogenize human values poses critical risks, similar to biodiversity loss undermining ecological resilience. Rooted in the ancient Greek concept of ethos, meaning both individual character and the shared moral fabric of communities, EthosGPT draws on a tradition that spans from Aristotle's virtue ethics to Adam Smith's moral sentiments as the ethical foundation of economic cooperation. These traditions underscore the vital role of value diversity in fostering social trust, institutional legitimacy, and long-term prosperity. EthosGPT addresses the challenge of value homogenization by introducing an open-source framework for mapping and evaluating LLMs within a global scale of human values. Using international survey data on cultural indices, prompt-based assessments, and comparative statistical analyses, EthosGPT reveals both the adaptability and biases of LLMs across regions and cultures. It offers actionable insights for developing inclusive LLMs, such as diversifying training data and preserving endangered cultural heritage to ensure representation in AI systems. These contributions align with the United Nations Sustainable Development Goals (SDGs), especially SDG 10 (Reduced Inequalities), SDG 11.4 (Cultural Heritage Preservation), and SDG 16 (Peace, Justice and Strong Institutions). Through interdisciplinary collaboration, EthosGPT promotes AI systems that are both technically robust and ethically inclusive, advancing value plurality as a cornerstone for sustainable and equitable futures. 

---
# MTCNET: Multi-task Learning Paradigm for Crowd Count Estimation 

**Authors**: Abhay Kumar, Nishant Jain, Suraj Tripathi, Chirag Singh, Kamal Krishna  

**Link**: [PDF](https://arxiv.org/pdf/1908.08652)  

**Abstract**: We propose a Multi-Task Learning (MTL) paradigm based deep neural network architecture, called MTCNet (Multi-Task Crowd Network) for crowd density and count estimation. Crowd count estimation is challenging due to the non-uniform scale variations and the arbitrary perspective of an individual image. The proposed model has two related tasks, with Crowd Density Estimation as the main task and Crowd-Count Group Classification as the auxiliary task. The auxiliary task helps in capturing the relevant scale-related information to improve the performance of the main task. The main task model comprises two blocks: VGG-16 front-end for feature extraction and a dilated Convolutional Neural Network for density map generation. The auxiliary task model shares the same front-end as the main task, followed by a CNN classifier. Our proposed network achieves 5.8% and 14.9% lower Mean Absolute Error (MAE) than the state-of-the-art methods on ShanghaiTech dataset without using any data augmentation. Our model also outperforms with 10.5% lower MAE on UCF_CC_50 dataset. 

---
# Focal Loss based Residual Convolutional Neural Network for Speech Emotion Recognition 

**Authors**: Suraj Tripathi, Abhay Kumar, Abhiram Ramesh, Chirag Singh, Promod Yenigalla  

**Link**: [PDF](https://arxiv.org/pdf/1906.05682)  

**Abstract**: This paper proposes a Residual Convolutional Neural Network (ResNet) based on speech features and trained under Focal Loss to recognize emotion in speech. Speech features such as Spectrogram and Mel-frequency Cepstral Coefficients (MFCCs) have shown the ability to characterize emotion better than just plain text. Further Focal Loss, first used in One-Stage Object Detectors, has shown the ability to focus the training process more towards hard-examples and down-weight the loss assigned to well-classified examples, thus preventing the model from being overwhelmed by easily classifiable examples. 

---
# RealWebAssist: A Benchmark for Long-Horizon Web Assistance with Real-World Users 

**Authors**: Suyu Ye, Haojun Shi, Darren Shih, Hyokun Yun, Tanya Roosta, Tianmin Shu  

**Link**: [PDF](https://arxiv.org/pdf/2504.10445)  

**Abstract**: To achieve successful assistance with long-horizon web-based tasks, AI agents must be able to sequentially follow real-world user instructions over a long period. Unlike existing web-based agent benchmarks, sequential instruction following in the real world poses significant challenges beyond performing a single, clearly defined task. For instance, real-world human instructions can be ambiguous, require different levels of AI assistance, and may evolve over time, reflecting changes in the user's mental state. To address this gap, we introduce RealWebAssist, a novel benchmark designed to evaluate sequential instruction-following in realistic scenarios involving long-horizon interactions with the web, visual GUI grounding, and understanding ambiguous real-world user instructions. RealWebAssist includes a dataset of sequential instructions collected from real-world human users. Each user instructs a web-based assistant to perform a series of tasks on multiple websites. A successful agent must reason about the true intent behind each instruction, keep track of the mental state of the user, understand user-specific routines, and ground the intended tasks to actions on the correct GUI elements. Our experimental results show that state-of-the-art models struggle to understand and ground user instructions, posing critical challenges in following real-world user instructions for long-horizon web assistance. 

---
