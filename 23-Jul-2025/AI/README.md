# Uncertainty-Aware Knowledge Transformers for Peer-to-Peer Energy Trading with Multi-Agent Reinforcement Learning 

**Authors**: Mian Ibad Ali Shah, Enda Barrett, Karl Mason  

**Link**: [PDF](https://arxiv.org/pdf/2507.16796)  

**Abstract**: This paper presents a novel framework for Peer-to-Peer (P2P) energy trading that integrates uncertainty-aware prediction with multi-agent reinforcement learning (MARL), addressing a critical gap in current literature. In contrast to previous works relying on deterministic forecasts, the proposed approach employs a heteroscedastic probabilistic transformer-based prediction model called Knowledge Transformer with Uncertainty (KTU) to explicitly quantify prediction uncertainty, which is essential for robust decision-making in the stochastic environment of P2P energy trading. The KTU model leverages domain-specific features and is trained with a custom loss function that ensures reliable probabilistic forecasts and confidence intervals for each prediction. Integrating these uncertainty-aware forecasts into the MARL framework enables agents to optimize trading strategies with a clear understanding of risk and variability. Experimental results show that the uncertainty-aware Deep Q-Network (DQN) reduces energy purchase costs by up to 5.7% without P2P trading and 3.2% with P2P trading, while increasing electricity sales revenue by 6.4% and 44.7%, respectively. Additionally, peak hour grid demand is reduced by 38.8% without P2P and 45.6% with P2P. These improvements are even more pronounced when P2P trading is enabled, highlighting the synergy between advanced forecasting and market mechanisms for resilient, economically efficient energy communities. 

---
# ChatChecker: A Framework for Dialogue System Testing and Evaluation Through Non-cooperative User Simulation 

**Authors**: Roman Mayr, Michel Schimpf, Thomas Bohné  

**Link**: [PDF](https://arxiv.org/pdf/2507.16792)  

**Abstract**: While modern dialogue systems heavily rely on large language models (LLMs), their implementation often goes beyond pure LLM interaction. Developers integrate multiple LLMs, external tools, and databases. Therefore, assessment of the underlying LLM alone does not suffice, and the dialogue systems must be tested and evaluated as a whole. However, this remains a major challenge. With most previous work focusing on turn-level analysis, less attention has been paid to integrated dialogue-level quality assurance. To address this, we present ChatChecker, a framework for automated evaluation and testing of complex dialogue systems. ChatChecker uses LLMs to simulate diverse user interactions, identify dialogue breakdowns, and evaluate quality. Compared to previous approaches, our design reduces setup effort and is generalizable, as it does not require reference dialogues and is decoupled from the implementation of the target dialogue system. We improve breakdown detection performance over a prior LLM-based approach by including an error taxonomy in the prompt. Additionally, we propose a novel non-cooperative user simulator based on challenging personas that uncovers weaknesses in target dialogue systems more effectively. Through this, ChatChecker contributes to thorough and scalable testing. This enables both researchers and practitioners to accelerate the development of robust dialogue systems. 

---
# WGRAMMAR: Leverage Prior Knowledge to Accelerate Structured Decoding 

**Authors**: Ran Wang, Xiaoxuan Liu, Hao Ren, Gang Chen, Fanchao Qi, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2507.16768)  

**Abstract**: Structured decoding enables large language models (LLMs) to generate outputs in formats required by downstream systems, such as HTML or JSON. However, existing methods suffer from efficiency bottlenecks due to grammar compilation, state tracking, and mask creation. We observe that many real-world tasks embed strong prior knowledge about output structure. Leveraging this, we propose a decomposition of constraints into static and dynamic components -- precompiling static structures offline and instantiating dynamic arguments at runtime using grammar snippets. Instead of relying on pushdown automata, we employ a compositional set of operators to model regular formats, achieving lower transition latency. We introduce wgrammar, a lightweight decoding engine that integrates domain-aware simplification, constraint decomposition, and mask caching, achieving up to 250x speedup over existing systems. wgrammar's source code is publicly available at this https URL. 

---
# Deliberative Searcher: Improving LLM Reliability via Reinforcement Learning with constraints 

**Authors**: Zhenyun Yin, Shujie Wang, Xuhong Wang, Xingjun Ma, Yinchun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.16727)  

**Abstract**: Improving the reliability of large language models (LLMs) is critical for deploying them in real-world scenarios. In this paper, we propose \textbf{Deliberative Searcher}, the first framework to integrate certainty calibration with retrieval-based search for open-domain question answering. The agent performs multi-step reflection and verification over Wikipedia data and is trained with a reinforcement learning algorithm that optimizes for accuracy under a soft reliability constraint. Empirical results show that proposed method improves alignment between model confidence and correctness, leading to more trustworthy outputs. This paper will be continuously updated. 

---
# Adaptive Inventory Strategies using Deep Reinforcement Learning for Dynamic Agri-Food Supply Chains 

**Authors**: Amandeep Kaur, Gyan Prakash  

**Link**: [PDF](https://arxiv.org/pdf/2507.16670)  

**Abstract**: Agricultural products are often subject to seasonal fluctuations in production and demand. Predicting and managing inventory levels in response to these variations can be challenging, leading to either excess inventory or stockouts. Additionally, the coordination among stakeholders at various level of food supply chain is not considered in the existing body of literature. To bridge these research gaps, this study focuses on inventory management of agri-food products under demand and lead time uncertainties. By implementing effective inventory replenishment policy results in maximize the overall profit throughout the supply chain. However, the complexity of the problem increases due to these uncertainties and shelf-life of the product, that makes challenging to implement traditional approaches to generate optimal set of solutions. Thus, the current study propose a novel Deep Reinforcement Learning (DRL) algorithm that combines the benefits of both value- and policy-based DRL approaches for inventory optimization under uncertainties. The proposed algorithm can incentivize collaboration among stakeholders by aligning their interests and objectives through shared optimization goal of maximizing profitability along the agri-food supply chain while considering perishability, and uncertainty simultaneously. By selecting optimal order quantities with continuous action space, the proposed algorithm effectively addresses the inventory optimization challenges. To rigorously evaluate this algorithm, the empirical data from fresh agricultural products supply chain inventory is considered. Experimental results corroborate the improved performance of the proposed inventory replenishment policy under stochastic demand patterns and lead time scenarios. The research findings hold managerial implications for policymakers to manage the inventory of agricultural products more effectively under uncertainty. 

---
# Novel Multi-Agent Action Masked Deep Reinforcement Learning for General Industrial Assembly Lines Balancing Problems 

**Authors**: Ali Mohamed Ali, Luca Tirel, Hashim A. Hashim  

**Link**: [PDF](https://arxiv.org/pdf/2507.16635)  

**Abstract**: Efficient planning of activities is essential for modern industrial assembly lines to uphold manufacturing standards, prevent project constraint violations, and achieve cost-effective operations. While exact solutions to such challenges can be obtained through Integer Programming (IP), the dependence of the search space on input parameters often makes IP computationally infeasible for large-scale scenarios. Heuristic methods, such as Genetic Algorithms, can also be applied, but they frequently produce suboptimal solutions in extensive cases. This paper introduces a novel mathematical model of a generic industrial assembly line formulated as a Markov Decision Process (MDP), without imposing assumptions on the type of assembly line a notable distinction from most existing models. The proposed model is employed to create a virtual environment for training Deep Reinforcement Learning (DRL) agents to optimize task and resource scheduling. To enhance the efficiency of agent training, the paper proposes two innovative tools. The first is an action-masking technique, which ensures the agent selects only feasible actions, thereby reducing training time. The second is a multi-agent approach, where each workstation is managed by an individual agent, as a result, the state and action spaces were reduced. A centralized training framework with decentralized execution is adopted, offering a scalable learning architecture for optimizing industrial assembly lines. This framework allows the agents to learn offline and subsequently provide real-time solutions during operations by leveraging a neural network that maps the current factory state to the optimal action. The effectiveness of the proposed scheme is validated through numerical simulations, demonstrating significantly faster convergence to the optimal solution compared to a comparable model-based approach. 

---
# Frontier AI Risk Management Framework in Practice: A Risk Analysis Technical Report 

**Authors**: Shanghai AI Lab, Xiaoyang Chen, Yunhao Chen, Zeren Chen, Zhiyun Chen, Hanyun Cui, Yawen Duan, Jiaxuan Guo, Qi Guo, Xuhao Hu, Hong Huang, Lige Huang, Chunxiao Li, Juncheng Li, Qihao Lin, Dongrui Liu, Xinmin Liu, Zicheng Liu, Chaochao Lu, Xiaoya Lu, Jingjing Qu, Qibing Ren, Jing Shao, Jingwei Shi, Jingwei Sun, Peng Wang, Weibing Wang, Jia Xu, Lewen Yan, Xiao Yu, Yi Yu, Boxuan Zhang, Jie Zhang, Weichen Zhang, Zhijie Zheng, Tianyi Zhou, Bowen Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.16534)  

**Abstract**: To understand and identify the unprecedented risks posed by rapidly advancing artificial intelligence (AI) models, this report presents a comprehensive assessment of their frontier risks. Drawing on the E-T-C analysis (deployment environment, threat source, enabling capability) from the Frontier AI Risk Management Framework (v1.0) (SafeWork-F1-Framework), we identify critical risks in seven areas: cyber offense, biological and chemical risks, persuasion and manipulation, uncontrolled autonomous AI R\&D, strategic deception and scheming, self-replication, and collusion. Guided by the "AI-$45^\circ$ Law," we evaluate these risks using "red lines" (intolerable thresholds) and "yellow lines" (early warning indicators) to define risk zones: green (manageable risk for routine deployment and continuous monitoring), yellow (requiring strengthened mitigations and controlled deployment), and red (necessitating suspension of development and/or deployment). Experimental results show that all recent frontier AI models reside in green and yellow zones, without crossing red lines. Specifically, no evaluated models cross the yellow line for cyber offense or uncontrolled AI R\&D risks. For self-replication, and strategic deception and scheming, most models remain in the green zone, except for certain reasoning models in the yellow zone. In persuasion and manipulation, most models are in the yellow zone due to their effective influence on humans. For biological and chemical risks, we are unable to rule out the possibility of most models residing in the yellow zone, although detailed threat modeling and in-depth assessment are required to make further claims. This work reflects our current understanding of AI frontier risks and urges collective action to mitigate these challenges. 

---
# Agentic RAG with Knowledge Graphs for Complex Multi-Hop Reasoning in Real-World Applications 

**Authors**: Jean Lelong, Adnane Errazine, Annabelle Blangero  

**Link**: [PDF](https://arxiv.org/pdf/2507.16507)  

**Abstract**: Conventional Retrieval-Augmented Generation (RAG) systems enhance Large Language Models (LLMs) but often fall short on complex queries, delivering limited, extractive answers and struggling with multiple targeted retrievals or navigating intricate entity relationships. This is a critical gap in knowledge-intensive domains. We introduce INRAExplorer, an agentic RAG system for exploring the scientific data of INRAE (France's National Research Institute for Agriculture, Food and Environment). INRAExplorer employs an LLM-based agent with a multi-tool architecture to dynamically engage a rich knowledge base, through a comprehensive knowledge graph derived from open access INRAE publications. This design empowers INRAExplorer to conduct iterative, targeted queries, retrieve exhaustive datasets (e.g., all publications by an author), perform multi-hop reasoning, and deliver structured, comprehensive answers. INRAExplorer serves as a concrete illustration of enhancing knowledge interaction in specialized fields. 

---
# ACT: Bridging the Gap in Code Translation through Synthetic Data Generation & Adaptive Training 

**Authors**: Shreya Saxena, Siva Prasad, Zishan Ahmad, Vishal Vaddina  

**Link**: [PDF](https://arxiv.org/pdf/2507.16478)  

**Abstract**: Code translation is a crucial process in software development and migration projects, enabling interoperability between different programming languages and enhancing software adaptability and thus longevity. Traditional automated translation methods rely heavily on handcrafted transformation rules, which often lack flexibility and scalability. Meanwhile, advanced language models present promising alternatives but are often limited by proprietary, API-based implementations that raise concerns over data security and reliance. In this paper, we present Auto-Train for Code Translation (ACT), an innovative framework that aims to improve code translation capabilities by enabling in-house finetuning of open-source Large Language Models (LLMs). ACT's automated pipeline significantly boosts the performance of these models, narrowing the gap between open-source accessibility and the high performance of closed-source solutions. Central to ACT is its synthetic data generation module, which builds extensive, high-quality datasets from initial code samples, incorporating unit tests to ensure functional accuracy and diversity. ACT's evaluation framework incorporates execution-level checks, offering a comprehensive assessment of translation quality. A key feature in ACT is its controller module, which manages the entire pipeline by dynamically adjusting hyperparameters, orchestrating iterative data generation, and finetuning based on real-time evaluations. This enables ACT to intelligently optimize when to continue training, generate additional targeted training data, or stop the process. Our results demonstrate that ACT consistently enhances the effectiveness of open-source models, offering businesses and developers a secure and reliable alternative. Additionally, applying our data generation pipeline to industry-scale migration projects has led to a notable increase in developer acceleration. 

---
# Learning Temporal Abstractions via Variational Homomorphisms in Option-Induced Abstract MDPs 

**Authors**: Chang Li, Yaren Zhang, Haoran Lv, Qiong Cao, Chao Xue, Xiaodong He  

**Link**: [PDF](https://arxiv.org/pdf/2507.16473)  

**Abstract**: Large Language Models (LLMs) have shown remarkable reasoning ability through explicit Chain-of-Thought (CoT) prompting, but generating these step-by-step textual explanations is computationally expensive and slow. To overcome this, we aim to develop a framework for efficient, implicit reasoning, where the model "thinks" in a latent space without generating explicit text for every step. We propose that these latent thoughts can be modeled as temporally-extended abstract actions, or options, within a hierarchical reinforcement learning framework. To effectively learn a diverse library of options as latent embeddings, we first introduce the Variational Markovian Option Critic (VMOC), an off-policy algorithm that uses variational inference within the HiT-MDP framework. To provide a rigorous foundation for using these options as an abstract reasoning space, we extend the theory of continuous MDP homomorphisms. This proves that learning a policy in the simplified, abstract latent space, for which VMOC is suited, preserves the optimality of the solution to the original, complex problem. Finally, we propose a cold-start procedure that leverages supervised fine-tuning (SFT) data to distill human reasoning demonstrations into this latent option space, providing a rich initialization for the model's reasoning capabilities. Extensive experiments demonstrate that our approach achieves strong performance on complex logical reasoning benchmarks and challenging locomotion tasks, validating our framework as a principled method for learning abstract skills for both language and control. 

---
# Improving ASP-based ORS Schedules through Machine Learning Predictions 

**Authors**: Pierangela Bruno, Carmine Dodaro, Giuseppe Galatà, Marco Maratea, Marco Mochi  

**Link**: [PDF](https://arxiv.org/pdf/2507.16454)  

**Abstract**: The Operating Room Scheduling (ORS) problem deals with the optimization of daily operating room surgery schedules. It is a challenging problem subject to many constraints, like to determine the starting time of different surgeries and allocating the required resources, including the availability of beds in different department units. Recently, solutions to this problem based on Answer Set Programming (ASP) have been delivered. Such solutions are overall satisfying but, when applied to real data, they can currently only verify whether the encoding aligns with the actual data and, at most, suggest alternative schedules that could have been computed. As a consequence, it is not currently possible to generate provisional schedules. Furthermore, the resulting schedules are not always robust.
In this paper, we integrate inductive and deductive techniques for solving these issues. We first employ machine learning algorithms to predict the surgery duration, from historical data, to compute provisional schedules. Then, we consider the confidence of such predictions as an additional input to our problem and update the encoding correspondingly in order to compute more robust schedules. Results on historical data from the ASL1 Liguria in Italy confirm the viability of our integration.
Under consideration in Theory and Practice of Logic Programming (TPLP). 

---
# From model-based learning to model-free behaviour with Meta-Interpretive Learning 

**Authors**: Stassa Patsantzis  

**Link**: [PDF](https://arxiv.org/pdf/2507.16434)  

**Abstract**: A "model" is a theory that describes the state of an environment and the effects of an agent's decisions on the environment. A model-based agent can use its model to predict the effects of its future actions and so plan ahead, but must know the state of the environment. A model-free agent cannot plan, but can act without a model and without completely observing the environment. An autonomous agent capable of acting independently in novel environments must combine both sets of capabilities. We show how to create such an agent with Meta-Interpretive Learning used to learn a model-based Solver used to train a model-free Controller that can solve the same planning problems as the Solver. We demonstrate the equivalence in problem-solving ability of the two agents on grid navigation problems in two kinds of environment: randomly generated mazes, and lake maps with wide open areas. We find that all navigation problems solved by the Solver are also solved by the Controller, indicating the two are equivalent. 

---
# Identifying Pre-training Data in LLMs: A Neuron Activation-Based Detection Framework 

**Authors**: Hongyi Tang, Zhihao Zhu, Yi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.16414)  

**Abstract**: The performance of large language models (LLMs) is closely tied to their training data, which can include copyrighted material or private information, raising legal and ethical concerns. Additionally, LLMs face criticism for dataset contamination and internalizing biases. To address these issues, the Pre-Training Data Detection (PDD) task was proposed to identify if specific data was included in an LLM's pre-training corpus. However, existing PDD methods often rely on superficial features like prediction confidence and loss, resulting in mediocre performance. To improve this, we introduce NA-PDD, a novel algorithm analyzing differential neuron activation patterns between training and non-training data in LLMs. This is based on the observation that these data types activate different neurons during LLM inference. We also introduce CCNewsPDD, a temporally unbiased benchmark employing rigorous data transformations to ensure consistent time distributions between training and non-training data. Our experiments demonstrate that NA-PDD significantly outperforms existing methods across three benchmarks and multiple LLMs. 

---
# Self-Supervised Inductive Logic Programming 

**Authors**: Stassa Patsantzis  

**Link**: [PDF](https://arxiv.org/pdf/2507.16405)  

**Abstract**: Inductive Logic Programming (ILP) approaches like Meta \-/ Interpretive Learning (MIL) can learn, from few examples, recursive logic programs with invented predicates that generalise well to unseen instances. This ability relies on a background theory and negative examples, both carefully selected with expert knowledge of a learning problem and its solutions. But what if such a problem-specific background theory or negative examples are not available? We formalise this question as a new setting for Self-Supervised ILP and present a new MIL algorithm that learns in the new setting from some positive labelled, and zero or more unlabelled examples, and automatically generates, and labels, new positive and negative examples during learning. We implement this algorithm in Prolog in a new MIL system, called Poker. We compare Poker to state-of-the-art MIL system Louise on experiments learning grammars for Context-Free and L-System languages from labelled, positive example strings, no negative examples, and just the terminal vocabulary of a language, seen in examples, as a first-order background theory. We introduce a new approach for the principled selection of a second-order background theory as a Second Order Definite Normal Form (SONF), sufficiently general to learn all programs in a class, thus removing the need for a backgound theory tailored to a learning task. We find that Poker's performance improves with increasing numbers of automatically generated examples while Louise, bereft of negative examples, over-generalises. 

---
# LLM-Driven Collaborative Model for Untangling Commits via Explicit and Implicit Dependency Reasoning 

**Authors**: Bo Hou, Xin Tan, Kai Zheng, Fang Liu, Yinghao Zhu, Li Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.16395)  

**Abstract**: Atomic commits, each of which addresses a single development concern, are a best practice in software development. However, developers frequently produce tangled commits that mix unrelated changes due to practical constraints or unclear boundaries, negatively impacting code review and maintenance. Although prior commit untangling approaches: rule-based, feature-based, or graph-based, have made progress, they often rely on shallow signals and fail to distinguish between explicit dependencies (e.g., control/data flow) and implicit ones (e.g., semantic or conceptual relationships). In this paper, we propose ColaUntangle, a new collaborative consultation framework for commit untangling that models both explicit and implicit dependencies among code changes. ColaUntangle integrates Large Language Model (LLM)-driven agents in a multi-agent architecture: one agent specializes in explicit dependencies, another in implicit ones, and a reviewer agent synthesizes their perspectives through iterative consultation. To capture explicit and implicit contextual information, we construct multi-version Program Dependency Graphs (delta-PDG), enabling agents to reason over code relationships with both symbolic and semantic depth. We evaluate ColaUntangle on two widely-used datasets (1,612 C# and 14k Java tangled commits). Experimental results show that ColaUntangle outperforms the best-performing baseline, achieving an improvement of 44% on the C# dataset and 100% on the Java dataset. These findings highlight the potential of LLM-based collaborative frameworks for advancing automated commit untangling tasks. 

---
# Canonical Representations of Markovian Structural Causal Models: A Framework for Counterfactual Reasoning 

**Authors**: Lucas de Lara  

**Link**: [PDF](https://arxiv.org/pdf/2507.16370)  

**Abstract**: Counterfactual reasoning aims at answering contrary-to-fact questions like ''Would have Alice recovered had she taken aspirin?'' and corresponds to the most fine-grained layer of causation. Critically, while many counterfactual statements cannot be falsified -- even by randomized experiments -- they underpin fundamental concepts like individual-wise fairness. Therefore, providing models to formalize and implement counterfactual beliefs remains a fundamental scientific problem. In the Markovian setting of Pearl's causal framework, we propose an alternative approach to structural causal models to represent counterfactuals compatible with a given causal graphical model. More precisely, we introduce counterfactual models, also called canonical representations of structural causal models. They enable analysts to choose a counterfactual conception via random-process probability distributions with preassigned marginals and characterize the counterfactual equivalence class of structural causal models. Then, we present a normalization procedure to describe and implement various counterfactual conceptions. Compared to structural causal models, it allows to specify many counterfactual conceptions without altering the observational and interventional constraints. Moreover, the content of the model corresponding to the counterfactual layer does not need to be estimated; only to make a choice. Finally, we illustrate the specific role of counterfactuals in causality and the benefits of our approach on theoretical and numerical examples. 

---
# Learning to Call: A Field Trial of a Collaborative Bandit Algorithm for Improved Message Delivery in Mobile Maternal Health 

**Authors**: Arpan Dasgupta, Mizhaan Maniyar, Awadhesh Srivastava, Sanat Kumar, Amrita Mahale, Aparna Hedge, Arun Suggala, Karthikeyan Shanmugam, Aparna Taneja, Milind Tambe  

**Link**: [PDF](https://arxiv.org/pdf/2507.16356)  

**Abstract**: Mobile health (mHealth) programs utilize automated voice messages to deliver health information, particularly targeting underserved communities, demonstrating the effectiveness of using mobile technology to disseminate crucial health information to these populations, improving health outcomes through increased awareness and behavioral change. India's Kilkari program delivers vital maternal health information via weekly voice calls to millions of mothers. However, the current random call scheduling often results in missed calls and reduced message delivery. This study presents a field trial of a collaborative bandit algorithm designed to optimize call timing by learning individual mothers' preferred call times. We deployed the algorithm with around $6500$ Kilkari participants as a pilot study, comparing its performance to the baseline random calling approach. Our results demonstrate a statistically significant improvement in call pick-up rates with the bandit algorithm, indicating its potential to enhance message delivery and impact millions of mothers across India. This research highlights the efficacy of personalized scheduling in mobile health interventions and underscores the potential of machine learning to improve maternal health outreach at scale. 

---
# Higher Gauge Flow Models 

**Authors**: Alexander Strunk, Roland Assam  

**Link**: [PDF](https://arxiv.org/pdf/2507.16334)  

**Abstract**: This paper introduces Higher Gauge Flow Models, a novel class of Generative Flow Models. Building upon ordinary Gauge Flow Models (arXiv:2507.13414), these Higher Gauge Flow Models leverage an L$_{\infty}$-algebra, effectively extending the Lie Algebra. This expansion allows for the integration of the higher geometry and higher symmetries associated with higher groups into the framework of Generative Flow Models. Experimental evaluation on a Gaussian Mixture Model dataset revealed substantial performance improvements compared to traditional Flow Models. 

---
# Mind the Gap: Evaluating the Representativeness of Quantitative Medical Language Reasoning LLM Benchmarks for African Disease Burdens 

**Authors**: Fred Mutisya, Shikoh Gitau, Christine Syovata, Diana Oigara, Ibrahim Matende, Muna Aden, Munira Ali, Ryan Nyotu, Diana Marion, Job Nyangena, Nasubo Ongoma, Keith Mbae, Elizabeth Wamicha, Eric Mibuari, Jean Philbert Nsengemana, Talkmore Chidede  

**Link**: [PDF](https://arxiv.org/pdf/2507.16322)  

**Abstract**: Introduction: Existing medical LLM benchmarks largely reflect examination syllabi and disease profiles from high income settings, raising questions about their validity for African deployment where malaria, HIV, TB, sickle cell disease and other neglected tropical diseases (NTDs) dominate burden and national guidelines drive care. Methodology: We systematically reviewed 31 quantitative LLM evaluation papers (Jan 2019 May 2025) identifying 19 English medical QA benchmarks. Alama Health QA was developed using a retrieval augmented generation framework anchored on the Kenyan Clinical Practice Guidelines. Six widely used sets (AfriMedQA, MMLUMedical, PubMedQA, MedMCQA, MedQAUSMLE, and guideline grounded Alama Health QA) underwent harmonized semantic profiling (NTD proportion, recency, readability, lexical diversity metrics) and blinded expert rating across five dimensions: clinical relevance, guideline alignment, clarity, distractor plausibility, and language/cultural fit. Results: Alama Health QA captured >40% of all NTD mentions across corpora and the highest within set frequencies for malaria (7.7%), HIV (4.1%), and TB (5.2%); AfriMedQA ranked second but lacked formal guideline linkage. Global benchmarks showed minimal representation (e.g., sickle cell disease absent in three sets) despite large scale. Qualitatively, Alama scored highest for relevance and guideline alignment; PubMedQA lowest for clinical utility. Discussion: Quantitative medical LLM benchmarks widely used in the literature underrepresent African disease burdens and regulatory contexts, risking misleading performance claims. Guideline anchored, regionally curated resources such as Alama Health QA and expanded disease specific derivatives are essential for safe, equitable model evaluation and deployment across African health systems. 

---
# Cross-Modal Distillation For Widely Differing Modalities 

**Authors**: Cairong Zhao, Yufeng Jin, Zifan Song, Haonan Chen, Duoqian Miao, Guosheng Hu  

**Link**: [PDF](https://arxiv.org/pdf/2507.16296)  

**Abstract**: Deep learning achieved great progress recently, however, it is not easy or efficient to further improve its performance by increasing the size of the model. Multi-modal learning can mitigate this challenge by introducing richer and more discriminative information as input. To solve the problem of limited access to multi-modal data at the time of use, we conduct multi-modal learning by introducing a teacher model to transfer discriminative knowledge to a student model during training. However, this knowledge transfer via distillation is not trivial because the big domain gap between the widely differing modalities can easily lead to overfitting. In this work, we introduce a cross-modal distillation framework. Specifically, we find hard constrained loss, e.g. l2 loss forcing the student being exact the same as the teacher, can easily lead to overfitting in cross-modality distillation. To address this, we propose two soft constrained knowledge distillation strategies at the feature level and classifier level respectively. In addition, we propose a quality-based adaptive weights module to weigh input samples via quantified data quality, leading to robust model training. We conducted experiments on speaker recognition and image classification tasks, and the results show that our approach is able to effectively achieve knowledge transfer between the commonly used and widely differing modalities of image, text, and speech. 

---
# ResearcherBench: Evaluating Deep AI Research Systems on the Frontiers of Scientific Inquiry 

**Authors**: Tianze Xu, Pengrui Lu, Lyumanshan Ye, Xiangkun Hu, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.16280)  

**Abstract**: The emergence of deep research systems presents significant capabilities in problem-solving, extending from basic queries to sophisticated research tasks. However, existing benchmarks primarily evaluate these systems as agents for web retrieval and report generation, overlooking their potential to discover novel insights on the frontiers of scientific research. To address this gap, we introduce ResearcherBench, the first benchmark focused on evaluating the capabilities of these advanced, agentic systems - which we refer to as Deep AI Research Systems (DARS) - on frontier AI scientific questions. We compiled a dataset of 65 research questions expertly selected from real-world scientific scenarios such as laboratory discussions and interviews, spanning 35 different AI subjects and categorized into three types: technical details, literature review, and open consulting. Our dual evaluation framework combines rubric assessment, which uses expert-designed criteria to evaluate insight quality, with factual assessment, which measures citation accuracy (faithfulness) and coverage (groundedness). We evaluated several leading commercial DARS and baseline systems. Results show that OpenAI Deep Research and Gemini Deep Research significantly outperform other systems, with particular strength in open-ended consulting questions. Such capabilities represent a meaningful step toward AI self-improvement, aligning with the vision of ASI for AI. We open-source ResearcherBench to provide a standardized platform for promoting the development of next-generation AI research assistants, hoping to foster a new perspective in AI research evaluation for a novel pattern of scientific collaboration: this https URL. 

---
# Voice-based AI Agents: Filling the Economic Gaps in Digital Health Delivery 

**Authors**: Bo Wen, Chen Wang, Qiwei Han, Raquel Norel, Julia Liu, Thaddeus Stappenbeck, Jeffrey L. Rogers  

**Link**: [PDF](https://arxiv.org/pdf/2507.16229)  

**Abstract**: The integration of voice-based AI agents in healthcare presents a transformative opportunity to bridge economic and accessibility gaps in digital health delivery. This paper explores the role of large language model (LLM)-powered voice assistants in enhancing preventive care and continuous patient monitoring, particularly in underserved populations. Drawing insights from the development and pilot study of Agent PULSE (Patient Understanding and Liaison Support Engine) -- a collaborative initiative between IBM Research, Cleveland Clinic Foundation, and Morehouse School of Medicine -- we present an economic model demonstrating how AI agents can provide cost-effective healthcare services where human intervention is economically unfeasible. Our pilot study with 33 inflammatory bowel disease patients revealed that 70\% expressed acceptance of AI-driven monitoring, with 37\% preferring it over traditional modalities. Technical challenges, including real-time conversational AI processing, integration with healthcare systems, and privacy compliance, are analyzed alongside policy considerations surrounding regulation, bias mitigation, and patient autonomy. Our findings suggest that AI-driven voice agents not only enhance healthcare scalability and efficiency but also improve patient engagement and accessibility. For healthcare executives, our cost-utility analysis demonstrates huge potential savings for routine monitoring tasks, while technologists can leverage our framework to prioritize improvements yielding the highest patient impact. By addressing current limitations and aligning AI development with ethical and regulatory frameworks, voice-based AI agents can serve as a critical entry point for equitable, sustainable digital healthcare solutions. 

---
# Distilled Large Language Model in Confidential Computing Environment for System-on-Chip Design 

**Authors**: Dong Ben, Hui Feng, Qian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.16226)  

**Abstract**: Large Language Models (LLMs) are increasingly used in circuit design tasks and have typically undergone multiple rounds of training. Both the trained models and their associated training data are considered confidential intellectual property (IP) and must be protected from exposure. Confidential Computing offers a promising solution to protect data and models through Trusted Execution Environments (TEEs). However, existing TEE implementations are not designed to support the resource-intensive nature of LLMs efficiently. In this work, we first present a comprehensive evaluation of the LLMs within a TEE-enabled confidential computing environment, specifically utilizing Intel Trust Domain Extensions (TDX). We constructed experiments on three environments: TEE-based, CPU-only, and CPU-GPU hybrid implementations, and evaluated their performance in terms of tokens per second.
Our first observation is that distilled models, i.e., DeepSeek, surpass other models in performance due to their smaller parameters, making them suitable for resource-constrained devices. Also, in the quantized models such as 4-bit quantization (Q4) and 8-bit quantization (Q8), we observed a performance gain of up to 3x compared to FP16 models. Our findings indicate that for fewer parameter sets, such as DeepSeek-r1-1.5B, the TDX implementation outperforms the CPU version in executing computations within a secure environment. We further validate the results using a testbench designed for SoC design tasks. These validations demonstrate the potential of efficiently deploying lightweight LLMs on resource-constrained systems for semiconductor CAD applications. 

---
# CHIMERA: Compressed Hybrid Intelligence for Twin-Model Enhanced Multi-Agent Deep Reinforcement Learning for Multi-Functional RIS-Assisted Space-Air-Ground Integrated Networks 

**Authors**: Li-Hsiang Shen, Jyun-Jhe Huang  

**Link**: [PDF](https://arxiv.org/pdf/2507.16204)  

**Abstract**: A space-air-ground integrated network (SAGIN) architecture is proposed, empowered by multi-functional reconfigurable intelligent surfaces (MF-RIS) capable of simultaneously reflecting, amplifying, and harvesting wireless energy. The MF-RIS plays a pivotal role in addressing the energy shortages of low-Earth orbit (LEO) satellites operating in shadowed regions, while explicitly accounting for both communication and computing energy consumption across the SAGIN nodes. To maximize the long-term energy efficiency (EE), we formulate a joint optimization problem over the MF-RIS parameters, including signal amplification, phase-shifts, energy harvesting ratio, and active element selection as well as the SAGIN parameters of beamforming vectors, high-altitude platform station (HAPS) deployment, user association, and computing capability. The formulated problem is highly non-convex and non-linear and contains mixed discrete-continuous parameters. To tackle this, we conceive a compressed hybrid intelligence for twin-model enhanced multi-agent deep reinforcement learning (CHIMERA) framework, which integrates semantic state-action compression and parametrized sharing under hybrid reinforcement learning to efficiently explore suitable complex actions. The simulation results have demonstrated that the proposed CHIMERA scheme substantially outperforms the conventional benchmarks, including fixed-configuration or non-harvesting MF-RIS, traditional RIS, and no-RIS cases, as well as centralized and multi-agent deep reinforcement learning baselines in terms of the highest EE. Moreover, the proposed SAGIN-MF-RIS architecture achieves superior EE performance due to its complementary coverage, offering notable advantages over either standalone satellite, aerial, or ground-only deployments. 

---
# Emergent Cognitive Convergence via Implementation: A Structured Loop Reflecting Four Theories of Mind (A Position Paper) 

**Authors**: Myung Ho Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.16184)  

**Abstract**: We report the discovery of a structural convergence across four influential theories of mind: Kahneman's dual-system theory, Friston's predictive processing, Minsky's society of mind, and Clark's extended mind-emerging unintentionally within a practical AI agent architecture called Agentic Flow. Designed to address limitations in large language models (LLMs), Agentic Flow comprises five interdependent modules such as Retrieval, Cognition, Control, Memory, and Action arranged in a recurrent cognitive loop. Although originally inspired only by Minsky and Clark, the system's structure retrospectively aligns with computational motifs found in all four theories, including predictive modeling, associative recall, and error-sensitive control.
To assess this convergence, we conducted comparative experiments with baseline LLM agents on multi-step reasoning tasks. The structured agent achieved 95.8% task success and exhibited strong constraint adherence, while the baseline system succeeded 62.3% of the time. These results were not aimed at proving superiority, but at illustrating how theoretical structures may emerge through practical design choices rather than top-down theory.
We introduce PEACE as a descriptive meta-architecture that captures design-level regularities observed in Agentic Flow. Not intended as a new theory, PEACE provides a shared vocabulary for understanding architectures shaped by real-world implementation demands. This paper should be read as a position paper - an exploratory reflection on how implementation can surface latent structural echoes of cognitive theory, without asserting theoretical unification. 

---
# SpiroLLM: Finetuning Pretrained LLMs to Understand Spirogram Time Series with Clinical Validation in COPD Reporting 

**Authors**: Shuhao Mei, Yongchao Long, Shan Cao, Xiaobo Han, Shijia Geng, Jinbo Sun, Yuxi Zhou, Shenda Hong  

**Link**: [PDF](https://arxiv.org/pdf/2507.16145)  

**Abstract**: Chronic Obstructive Pulmonary Disease (COPD), a major chronic respiratory disease with persistent airflow limitation, is a leading global cause of disability and mortality. Respiratory spirogram time series, routinely collected during pulmonary function tests (PFTs), play a critical role in the early detection of repsiratory diseases and in monitoring lung function over time. However, most current AI models for COPD diagnosis are limited to outputting classification results without providing a rationale for their diagnostic process, while current Large Language Models (LLMs) cannot understand spirograms yet, which severely limits their clinical trust and adoption. To tackle this challenge, we leverage a cohort of 234,028 individuals from the UK Biobank (UKB) to propose SpiroLLM, the first multimodal large language model that can understand spirogram. The model extracts morphological features from respiratory curves via a SpiroEncoder and aligns them with PFT numerical values in a unified latent space using a SpiroProjector, ultimately empowering a large language model to generate a comprehensive diagnostic report. Experimental results confirm that SpiroLLM achieved a diagnostic AUROC of 0.8980 (95% CI: 0.8820-0.9132). In a robustness test with missing core data, it maintained a 100% valid response rate, far surpassing the 13.4% of a text-only model and showcasing the superiority of its multimodal design. This work demonstrates the substantial potential of deeply fusing physiological signals with large language models, establishing a new paradigm for the next generation of interpretable and reliable clinical decision support tools. 

---
# TaxCalcBench: Evaluating Frontier Models on the Tax Calculation Task 

**Authors**: Michael R. Bock, Kara Molisee, Zachary Ozer, Sumit Shah  

**Link**: [PDF](https://arxiv.org/pdf/2507.16126)  

**Abstract**: Can AI file your taxes? Not yet. Calculating US personal income taxes is a task that requires building an understanding of vast amounts of English text and using that knowledge to carefully compute results. We propose TaxCalcBench, a benchmark for determining models' abilities to calculate personal income tax returns given all of the necessary information. Our experiment shows that state-of-the-art models succeed in calculating less than a third of federal income tax returns even on this simplified sample set. Our analysis concludes that models consistently misuse tax tables, make errors in tax calculation, and incorrectly determine eligibility. Our findings point to the need for additional infrastructure to apply LLMs to the personal income tax calculation task. 

---
# Expert-Guided LLM Reasoning for Battery Discovery: From AI-Driven Hypothesis to Synthesis and Characterization 

**Authors**: Shengchao Liu, Hannan Xu, Yan Ai, Huanxin Li, Yoshua Bengio, Harry Guo  

**Link**: [PDF](https://arxiv.org/pdf/2507.16110)  

**Abstract**: Large language models (LLMs) leverage chain-of-thought (CoT) techniques to tackle complex problems, representing a transformative breakthrough in artificial intelligence (AI). However, their reasoning capabilities have primarily been demonstrated in solving math and coding problems, leaving their potential for domain-specific applications-such as battery discovery-largely unexplored. Inspired by the idea that reasoning mirrors a form of guided search, we introduce ChatBattery, a novel agentic framework that integrates domain knowledge to steer LLMs toward more effective reasoning in materials design. Using ChatBattery, we successfully identify, synthesize, and characterize three novel lithium-ion battery cathode materials, which achieve practical capacity improvements of 28.8%, 25.2%, and 18.5%, respectively, over the widely used cathode material, LiNi0.8Mn0.1Co0.1O2 (NMC811). Beyond this discovery, ChatBattery paves a new path by showing a successful LLM-driven and reasoning-based platform for battery materials invention. This complete AI-driven cycle-from design to synthesis to characterization-demonstrates the transformative potential of AI-driven reasoning in revolutionizing materials discovery. 

---
# A Unifying Framework for Semiring-Based Constraint Logic Programming With Negation (full version) 

**Authors**: Jeroen Spaans, Jesse Heyninck  

**Link**: [PDF](https://arxiv.org/pdf/2507.16067)  

**Abstract**: Constraint Logic Programming (CLP) is a logic programming formalism used to solve problems requiring the consideration of constraints, like resource allocation and automated planning and scheduling. It has previously been extended in various directions, for example to support fuzzy constraint satisfaction, uncertainty, or negation, with different notions of semiring being used as a unifying abstraction for these generalizations. None of these extensions have studied clauses with negation allowed in the body. We investigate an extension of CLP which unifies many of these extensions and allows negation in the body. We provide semantics for such programs, using the framework of approximation fixpoint theory, and give a detailed overview of the impacts of properties of the semirings on the resulting semantics. As such, we provide a unifying framework that captures existing approaches and allows extending them with a more expressive language. 

---
# From Logic to Language: A Trust Index for Problem Solving with LLMs 

**Authors**: Tehseen Rug, Felix Böhmer, Tessa Pfattheicher  

**Link**: [PDF](https://arxiv.org/pdf/2507.16028)  

**Abstract**: Classical computation, grounded in formal, logical systems, has been the engine of technological progress for decades, excelling at problems that can be described with unambiguous rules. This paradigm, however, leaves a vast ocean of human problems -- those characterized by ambiguity, dynamic environments, and subjective context -- largely untouched. The advent of Large Language Models (LLMs) represents a fundamental shift, enabling computational systems to engage with this previously inaccessible domain using natural language. This paper introduces a unified framework to understand and contrast these problem-solving paradigms. We define and delineate the problem spaces addressable by formal languages versus natural language. While solutions to the former problem class can be evaluated using binary quality measures, the latter requires a much more nuanced definition of approximate solution space taking into account the vagueness, subjectivity and ambiguity inherent to natural language. We therefore introduce a vector-valued trust index Q, which reflects solution quality and distinguishes the binary correctness of formal solutions from the continuous adequacy spectrum characteristic of natural language solutions. Within this framework, we propose two statistical quality dimensions. Normalized bi-semantic entropy measures robustness and conceptual diversity of LLM answers given semantic variation in problem formulations. Emotional valence maps subjective valuation of a solution to a quantifiable metric that can be maximized by invoking statistical measures. The concepts introduced in this work will provide a more rigorous understanding of the capabilities, limitations, and inherent nature of problem-solving in the age of LLMs. 

---
# Micromobility Flow Prediction: A Bike Sharing Station-level Study via Multi-level Spatial-Temporal Attention Neural Network 

**Authors**: Xi Yang, Jiachen Wang, Song Han, Suining He  

**Link**: [PDF](https://arxiv.org/pdf/2507.16020)  

**Abstract**: Efficient use of urban micromobility resources such as bike sharing is challenging due to the unbalanced station-level demand and supply, which causes the maintenance of the bike sharing systems painstaking. Prior efforts have been made on accurate prediction of bike traffics, i.e., demand/pick-up and return/drop-off, to achieve system efficiency. However, bike station-level traffic prediction is difficult because of the spatial-temporal complexity of bike sharing systems. Moreover, such level of prediction over entire bike sharing systems is also challenging due to the large number of bike stations. To fill this gap, we propose BikeMAN, a multi-level spatio-temporal attention neural network to predict station-level bike traffic for entire bike sharing systems. The proposed network consists of an encoder and a decoder with an attention mechanism representing the spatial correlation between features of bike stations in the system and another attention mechanism describing the temporal characteristic of bike station traffic. Through experimental study on over 10 millions trips of bike sharing systems (> 700 stations) of New York City, our network showed high accuracy in predicting the bike station traffic of all stations in the city. 

---
# Does More Inference-Time Compute Really Help Robustness? 

**Authors**: Tong Wu, Chong Xiang, Jiachen T. Wang, Weichen Yu, Chawin Sitawarin, Vikash Sehwag, Prateek Mittal  

**Link**: [PDF](https://arxiv.org/pdf/2507.15974)  

**Abstract**: Recently, Zaremba et al. demonstrated that increasing inference-time computation improves robustness in large proprietary reasoning LLMs. In this paper, we first show that smaller-scale, open-source models (e.g., DeepSeek R1, Qwen3, Phi-reasoning) can also benefit from inference-time scaling using a simple budget forcing strategy. More importantly, we reveal and critically examine an implicit assumption in prior work: intermediate reasoning steps are hidden from adversaries. By relaxing this assumption, we identify an important security risk, intuitively motivated and empirically verified as an inverse scaling law: if intermediate reasoning steps become explicitly accessible, increased inference-time computation consistently reduces model robustness. Finally, we discuss practical scenarios where models with hidden reasoning chains are still vulnerable to attacks, such as models with tool-integrated reasoning and advanced reasoning extraction attacks. Our findings collectively demonstrate that the robustness benefits of inference-time scaling depend heavily on the adversarial setting and deployment context. We urge practitioners to carefully weigh these subtle trade-offs before applying inference-time scaling in security-sensitive, real-world applications. 

---
# Advancing Responsible Innovation in Agentic AI: A study of Ethical Frameworks for Household Automation 

**Authors**: Joydeep Chandra, Satyam Kumar Navneet  

**Link**: [PDF](https://arxiv.org/pdf/2507.15901)  

**Abstract**: The implementation of Artificial Intelligence (AI) in household environments, especially in the form of proactive autonomous agents, brings about possibilities of comfort and attention as well as it comes with intra or extramural ethical challenges. This article analyzes agentic AI and its applications, focusing on its move from reactive to proactive autonomy, privacy, fairness and user control. We review responsible innovation frameworks, human-centered design principles, and governance practices to distill practical guidance for ethical smart home systems. Vulnerable user groups such as elderly individuals, children, and neurodivergent who face higher risks of surveillance, bias, and privacy risks were studied in detail in context of Agentic AI. Design imperatives are highlighted such as tailored explainability, granular consent mechanisms, and robust override controls, supported by participatory and inclusive methodologies. It was also explored how data-driven insights, including social media analysis via Natural Language Processing(NLP), can inform specific user needs and ethical concerns. This survey aims to provide both a conceptual foundation and suggestions for developing transparent, inclusive, and trustworthy agentic AI in household automation. 

---
# Integrating Reason-Based Moral Decision-Making in the Reinforcement Learning Architecture 

**Authors**: Lisa Dargasz  

**Link**: [PDF](https://arxiv.org/pdf/2507.15895)  

**Abstract**: Reinforcement Learning is a machine learning methodology that has demonstrated strong performance across a variety of tasks. In particular, it plays a central role in the development of artificial autonomous agents. As these agents become increasingly capable, market readiness is rapidly approaching, which means those agents, for example taking the form of humanoid robots or autonomous cars, are poised to transition from laboratory prototypes to autonomous operation in real-world environments. This transition raises concerns leading to specific requirements for these systems - among them, the requirement that they are designed to behave ethically. Crucially, research directed toward building agents that fulfill the requirement to behave ethically - referred to as artificial moral agents(AMAs) - has to address a range of challenges at the intersection of computer science and philosophy. This study explores the development of reason-based artificial moral agents (RBAMAs). RBAMAs are build on an extension of the reinforcement learning architecture to enable moral decision-making based on sound normative reasoning, which is achieved by equipping the agent with the capacity to learn a reason-theory - a theory which enables it to process morally relevant propositions to derive moral obligations - through case-based feedback. They are designed such that they adapt their behavior to ensure conformance to these obligations while they pursue their designated tasks. These features contribute to the moral justifiability of the their actions, their moral robustness, and their moral trustworthiness, which proposes the extended architecture as a concrete and deployable framework for the development of AMAs that fulfills key ethical desiderata. This study presents a first implementation of an RBAMA and demonstrates the potential of RBAMAs in initial experiments. 

---
# ADEPTS: A Capability Framework for Human-Centered Agent Design 

**Authors**: Pierluca D'Oro, Caley Drooff, Joy Chen, Joseph Tighe  

**Link**: [PDF](https://arxiv.org/pdf/2507.15885)  

**Abstract**: Large language models have paved the way to powerful and flexible AI agents, assisting humans by increasingly integrating into their daily life. This flexibility, potential, and growing adoption demands a holistic and cross-disciplinary approach to developing, monitoring and discussing the capabilities required for agent-driven user experiences. However, current guidance on human-centered AI agent development is scattered: UX heuristics focus on interface behaviors, engineering taxonomies describe internal pipelines, and ethics checklists address high-level governance. There is no concise, user-facing vocabulary that tells teams what an agent should fundamentally be able to do. We introduce ADEPTS, a capability framework defining a set of core user-facing capabilities to provide unified guidance around the development of AI agents. ADEPTS is based on six principles for human-centered agent design, that express the minimal, user-facing capabilities an AI agent should demonstrate to be understandable, controllable and trustworthy in everyday use. ADEPTS complements existing frameworks and taxonomies; differently from them, it sits at the interface between technical and experience development. By presenting ADEPTS, we aim to condense complex AI-UX requirements into a compact framework that is actionable guidance for AI researchers, designers, engineers, and policy reviewers alike. We believe ADEPTS has the potential of accelerating the improvement of user-relevant agent capabilities, of easing the design of experiences that take advantage of those capabilities, and of providing a shared language to track and discuss progress around the development of AI agents. 

---
# The Recursive Coherence Principle: A Formal Constraint on Scalable Intelligence, Alignment, and Reasoning Architecture 

**Authors**: Andy E. Williams  

**Link**: [PDF](https://arxiv.org/pdf/2507.15880)  

**Abstract**: Intelligence-biological, artificial, or collective-requires structural coherence across recursive reasoning processes to scale effectively. As complex systems grow, coherence becomes fragile unless a higher-order structure ensures semantic consistency. This paper introduces the Recursive Coherence Principle (RCP): a foundational constraint stating that for any reasoning system of order N, composed of systems operating over conceptual spaces of order N-1, semantic coherence is preserved only by a recursively evaluable generalization operator that spans and aligns those lower-order conceptual spaces. Crucially, this coherence enables structural alignment. Without recursive coherence, no system can reliably preserve goals, meanings, or reasoning consistency at scale. We formally define the Functional Model of Intelligence (FMI) as the only known operator capable of satisfying the RCP at any scale. The FMI is a minimal, composable architecture with internal functions (evaluation, modeling, adaptation, stability, decomposition, bridging) and external functions (storage, recall, System 1 and System 2 reasoning) vital for preserving semantic structure across inference and coordination layers. We prove that any system lacking the FMI will experience recursive coherence breakdown as it scales, arguing that common AI issues like misalignment, hallucination, and instability are symptoms of this structural coherence loss. Unlike other foundational principles, RCP uniquely captures the internal, recursive dynamics needed for coherent, alignable intelligence, modeling semantic coherence under recursion. This work significantly impacts AI alignment, advocating a shift from behavioral constraints to structural coherence, and offers a pathway for safely generalizable, robustly coherent AI at scale. 

---
# Out-of-Distribution Generalization in the ARC-AGI Domain: Comparing Execution-Guided Neural Program Synthesis and Test-Time Fine-Tuning 

**Authors**: Simon Ouellette  

**Link**: [PDF](https://arxiv.org/pdf/2507.15877)  

**Abstract**: We run a controlled compositional generalization experiment in the ARC-AGI domain: an open-world problem domain in which the ability to generalize out-of-distribution is, by design, an essential characteristic for success. We compare neural program synthesis and test-time fine-tuning approaches on this experiment. We find that execution-guided neural program synthesis outperforms all reference algorithms in its ability to compose novel solutions. Our empirical findings also suggest that the success of TTFT on ARC-AGI lies mainly in eliciting in-distribution knowledge that the LLM otherwise fails to rely on directly. 

---
# Re-evaluating Short- and Long-Term Trend Factors in CTA Replication: A Bayesian Graphical Approach 

**Authors**: Eric Benhamou, Jean-Jacques Ohana, Alban Etienne, Béatrice Guez, Ethan Setrouk, Thomas Jacquot  

**Link**: [PDF](https://arxiv.org/pdf/2507.15876)  

**Abstract**: Commodity Trading Advisors (CTAs) have historically relied on trend-following rules that operate on vastly different horizons from long-term breakouts that capture major directional moves to short-term momentum signals that thrive in fast-moving markets. Despite a large body of work on trend following, the relative merits and interactions of short-versus long-term trend systems remain controversial. This paper adds to the debate by (i) dynamically decomposing CTA returns into short-term trend, long-term trend and market beta factors using a Bayesian graphical model, and (ii) showing how the blend of horizons shapes the strategy's risk-adjusted performance. 

---
# Differential Multimodal Transformers 

**Authors**: Jerry Li, Timothy Oh, Joseph Hoang, Vardhit Veeramachaneni  

**Link**: [PDF](https://arxiv.org/pdf/2507.15875)  

**Abstract**: Small language models have gained significant popularity due to their efficiency and growing capabilities. However, incorporating additional modalities, such as vision, can exacerbate the challenge of limited context windows by introducing noise. Recent studies have highlighted that Transformer attention mechanisms often disproportionately focus on irrelevant contexts. In this work, we extend the Differential Attention mechanism, originally designed for text-only models, to the text-vision model PaliGemma. Our aim is to evaluate its ability to mitigate noisy information retrieval and reduce hallucinations. To this end, we fine-tuned the PaliGemma 3B model using LoRA, incorporating Differential Attention, and experimented with various parameter settings and configurations. We demonstrate that Differential Attention can be adapted and integrated into the fine-tuning of existing models to enhance noisy information retrieval and question-answering capabilities. 

---
# Why Braking? Scenario Extraction and Reasoning Utilizing LLM 

**Authors**: Yin Wu, Daniel Slieter, Vivek Subramanian, Ahmed Abouelazm, Robin Bohn, J. Marius Zöllner  

**Link**: [PDF](https://arxiv.org/pdf/2507.15874)  

**Abstract**: The growing number of ADAS-equipped vehicles has led to a dramatic increase in driving data, yet most of them capture routine driving behavior. Identifying and understanding safety-critical corner cases within this vast dataset remains a significant challenge. Braking events are particularly indicative of potentially hazardous situations, motivating the central question of our research: Why does a vehicle brake? Existing approaches primarily rely on rule-based heuristics to retrieve target scenarios using predefined condition filters. While effective in simple environments such as highways, these methods lack generalization in complex urban settings. In this paper, we propose a novel framework that leverages Large Language Model (LLM) for scenario understanding and reasoning. Our method bridges the gap between low-level numerical signals and natural language descriptions, enabling LLM to interpret and classify driving scenarios. We propose a dual-path scenario retrieval that supports both category-based search for known scenarios and embedding-based retrieval for unknown Out-of-Distribution (OOD) scenarios. To facilitate evaluation, we curate scenario annotations on the Argoverse 2 Sensor Dataset. Experimental results show that our method outperforms rule-based baselines and generalizes well to OOD scenarios. 

---
# Purchase and Production Optimization in a Meat Processing Plant 

**Authors**: Marek Vlk, Premysl Sucha, Jaroslaw Rudy, Radoslaw Idzikowski  

**Link**: [PDF](https://arxiv.org/pdf/2507.15866)  

**Abstract**: The food production industry, especially the meat production sector, faces many challenges that have even escalated due to the recent outbreak of the energy crisis in the European Union. Therefore, efficient use of input materials is an essential aspect affecting the profit of such companies. This paper addresses an optimization problem concerning the purchase and subsequent material processing we solved for a meat processing company. Unlike the majority of existing papers, we do not concentrate on how this problem concerns supply chain management, but we focus purely on the production stage. The problem involves the concept of alternative ways of material processing, stock of material with different expiration dates, and extra constraints widely neglected in the current literature, namely, the minimum order quantity and the minimum percentage in alternatives. We prove that each of these two constraints makes the problem \mbox{$\mathcal{NP}$-hard}, and hence we design a simple iterative approach based on integer linear programming that allows us to solve real-life instances even using an open-source integer linear programming solver. Another advantage of this approach is that it mitigates numerical issues, caused by the extensive range of data values, we experienced with a commercial solver. The results obtained using real data from the meat processing company showed that our algorithm can find the optimum solution in a few seconds for all considered use cases. 

---
# From Reasoning to Super-Intelligence: A Search-Theoretic Perspective 

**Authors**: Shai Shalev-Shwartz, Amnon Shashua  

**Link**: [PDF](https://arxiv.org/pdf/2507.15865)  

**Abstract**: Chain-of-Thought (CoT) reasoning has emerged as a powerful tool for enhancing the problem-solving capabilities of large language models (LLMs). However, the theoretical foundations of learning from CoT data remain underdeveloped, and existing approaches -- such as Supervised Fine-Tuning (SFT), Reinforcement Learning (RL), Tree-of-Thoughts (ToT), and Monte Carlo Tree Search (MCTS) -- often fail on complex reasoning tasks. In this work, we identify core obstacles that hinder effective CoT learning, including distribution drift, lack of embedded search, and exponential inference costs. We introduce the Diligent Learner, a new learning paradigm that explicitly models reasoning as a depth-first search guided by a validator and supports backtracking upon failure. Under two mild and realistic assumptions, we prove that the Diligent Learner can efficiently learn from CoT data while existing methods fail to do so. This framework offers a path toward building scalable and reliable reasoning systems trained on naturally occurring, incomplete data -- paving the way for the development of Large Reasoning Models (LRMs) with robust, interpretable problem-solving abilities. 

---
# ThinkAct: Vision-Language-Action Reasoning via Reinforced Visual Latent Planning 

**Authors**: Chi-Pin Huang, Yueh-Hua Wu, Min-Hung Chen, Yu-Chiang Frank Wang, Fu-En Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.16815)  

**Abstract**: Vision-language-action (VLA) reasoning tasks require agents to interpret multimodal instructions, perform long-horizon planning, and act adaptively in dynamic environments. Existing approaches typically train VLA models in an end-to-end fashion, directly mapping inputs to actions without explicit reasoning, which hinders their ability to plan over multiple steps or adapt to complex task variations. In this paper, we propose ThinkAct, a dual-system framework that bridges high-level reasoning with low-level action execution via reinforced visual latent planning. ThinkAct trains a multimodal LLM to generate embodied reasoning plans guided by reinforcing action-aligned visual rewards based on goal completion and trajectory consistency. These reasoning plans are compressed into a visual plan latent that conditions a downstream action model for robust action execution on target environments. Extensive experiments on embodied reasoning and robot manipulation benchmarks demonstrate that ThinkAct enables few-shot adaptation, long-horizon planning, and self-correction behaviors in complex embodied AI tasks. 

---
# MegaScience: Pushing the Frontiers of Post-Training Datasets for Science Reasoning 

**Authors**: Run-Ze Fan, Zengzhi Wang, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.16812)  

**Abstract**: Scientific reasoning is critical for developing AI scientists and supporting human researchers in advancing the frontiers of natural science discovery. However, the open-source community has primarily focused on mathematics and coding while neglecting the scientific domain, largely due to the absence of open, large-scale, high-quality, verifiable scientific reasoning datasets. To bridge this gap, we first present TextbookReasoning, an open dataset featuring truthful reference answers extracted from 12k university-level scientific textbooks, comprising 650k reasoning questions spanning 7 scientific disciplines. We further introduce MegaScience, a large-scale mixture of high-quality open-source datasets totaling 1.25 million instances, developed through systematic ablation studies that evaluate various data selection methodologies to identify the optimal subset for each publicly available scientific dataset. Meanwhile, we build a comprehensive evaluation system covering diverse subjects and question types across 15 benchmarks, incorporating comprehensive answer extraction strategies to ensure accurate evaluation metrics. Our experiments demonstrate that our datasets achieve superior performance and training efficiency with more concise response lengths compared to existing open-source scientific datasets. Furthermore, we train Llama3.1, Qwen2.5, and Qwen3 series base models on MegaScience, which significantly outperform the corresponding official instruct models in average performance. In addition, MegaScience exhibits greater effectiveness for larger and stronger models, suggesting a scaling benefit for scientific tuning. We release our data curation pipeline, evaluation system, datasets, and seven trained models to the community to advance scientific reasoning research. 

---
# Rethinking LLM-Based RTL Code Optimization Via Timing Logic Metamorphosis 

**Authors**: Zhihao Xu, Bixin Li, Lulu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.16808)  

**Abstract**: Register Transfer Level(RTL) code optimization is crucial for achieving high performance and low power consumption in digital circuit design. However, traditional optimization methods often rely on manual tuning and heuristics, which can be time-consuming and error-prone. Recent studies proposed to leverage Large Language Models(LLMs) to assist in RTL code optimization. LLMs can generate optimized code snippets based on natural language descriptions, potentially speeding up the optimization process. However, existing approaches have not thoroughly evaluated the effectiveness of LLM-Based code optimization methods for RTL code with complex timing logic. To address this gap, we conducted a comprehensive empirical investigation to assess the capability of LLM-Based RTL code optimization methods in handling RTL code with complex timing logic. In this study, we first propose a new benchmark for RTL optimization evaluation. It comprises four subsets, each corresponding to a specific area of RTL code optimization. Then we introduce a method based on metamorphosis to systematically evaluate the effectiveness of LLM-Based RTL code optimization this http URL key insight is that the optimization effectiveness should remain consistent for semantically equivalent but more complex code. After intensive experiments, we revealed several key findings. (1) LLM-Based RTL optimization methods can effectively optimize logic operations and outperform existing compiler-based methods. (2) LLM-Based RTL optimization methods do not perform better than existing compiler-based methods on RTL code with complex timing logic, particularly in timing control flow optimization and clock domain optimization. This is primarily attributed to the challenges LLMs face in understanding timing logic in RTL code. Based on these findings, we provide insights for further research in leveraging LLMs for RTL code optimization. 

---
# Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty 

**Authors**: Mehul Damani, Isha Puri, Stewart Slocum, Idan Shenfeld, Leshem Choshen, Yoon Kim, Jacob Andreas  

**Link**: [PDF](https://arxiv.org/pdf/2507.16806)  

**Abstract**: When language models (LMs) are trained via reinforcement learning (RL) to generate natural language "reasoning chains", their performance improves on a variety of difficult question answering tasks. Today, almost all successful applications of RL for reasoning use binary reward functions that evaluate the correctness of LM outputs. Because such reward functions do not penalize guessing or low-confidence outputs, they often have the unintended side-effect of degrading calibration and increasing the rate at which LMs generate incorrect responses (or "hallucinate") in other problem domains. This paper describes RLCR (Reinforcement Learning with Calibration Rewards), an approach to training reasoning models that jointly improves accuracy and calibrated confidence estimation. During RLCR, LMs generate both predictions and numerical confidence estimates after reasoning. They are trained to optimize a reward function that augments a binary correctness score with a Brier score -- a scoring rule for confidence estimates that incentivizes calibrated prediction. We first prove that this reward function (or any analogous reward function that uses a bounded, proper scoring rule) yields models whose predictions are both accurate and well-calibrated. We next show that across diverse datasets, RLCR substantially improves calibration with no loss in accuracy, on both in-domain and out-of-domain evaluations -- outperforming both ordinary RL training and classifiers trained to assign post-hoc confidence scores. While ordinary RL hurts calibration, RLCR improves it. Finally, we demonstrate that verbalized confidence can be leveraged at test time to improve accuracy and calibration via confidence-weighted scaling methods. Our results show that explicitly optimizing for calibration can produce more generally reliable reasoning models. 

---
# Decoding Translation-Related Functional Sequences in 5'UTRs Using Interpretable Deep Learning Models 

**Authors**: Yuxi Lin, Yaxue Fang, Zehong Zhang, Zhouwu Liu, Siyun Zhong, Fulong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.16801)  

**Abstract**: Understanding how 5' untranslated regions (5'UTRs) regulate mRNA translation is critical for controlling protein expression and designing effective therapeutic mRNAs. While recent deep learning models have shown promise in predicting translational efficiency from 5'UTR sequences, most are constrained by fixed input lengths and limited interpretability. We introduce UTR-STCNet, a Transformer-based architecture for flexible and biologically grounded modeling of variable-length 5'UTRs. UTR-STCNet integrates a Saliency-Aware Token Clustering (SATC) module that iteratively aggregates nucleotide tokens into multi-scale, semantically meaningful units based on saliency scores. A Saliency-Guided Transformer (SGT) block then captures both local and distal regulatory dependencies using a lightweight attention mechanism. This combined architecture achieves efficient and interpretable modeling without input truncation or increased computational cost. Evaluated across three benchmark datasets, UTR-STCNet consistently outperforms state-of-the-art baselines in predicting mean ribosome load (MRL), a key proxy for translational efficiency. Moreover, the model recovers known functional elements such as upstream AUGs and Kozak motifs, highlighting its potential for mechanistic insight into translation regulation. 

---
# Steering Out-of-Distribution Generalization with Concept Ablation Fine-Tuning 

**Authors**: Helena Casademunt, Caden Juang, Adam Karvonen, Samuel Marks, Senthooran Rajamanoharan, Neel Nanda  

**Link**: [PDF](https://arxiv.org/pdf/2507.16795)  

**Abstract**: Fine-tuning large language models (LLMs) can lead to unintended out-of-distribution generalization. Standard approaches to this problem rely on modifying training data, for example by adding data that better specify the intended generalization. However, this is not always practical. We introduce Concept Ablation Fine-Tuning (CAFT), a technique that leverages interpretability tools to control how LLMs generalize from fine-tuning, without needing to modify the training data or otherwise use data from the target distribution. Given a set of directions in an LLM's latent space corresponding to undesired concepts, CAFT works by ablating these concepts with linear projections during fine-tuning, steering the model away from unintended generalizations. We successfully apply CAFT to three fine-tuning tasks, including emergent misalignment, a phenomenon where LLMs fine-tuned on a narrow task generalize to give egregiously misaligned responses to general questions. Without any changes to the fine-tuning data, CAFT reduces misaligned responses by 10x without degrading performance on the training distribution. Overall, CAFT represents a novel approach for steering LLM generalization without modifying training data. 

---
# Never Come Up Empty: Adaptive HyDE Retrieval for Improving LLM Developer Support 

**Authors**: Fangjian Lei, Mariam El Mezouar, Shayan Noei, Ying Zou  

**Link**: [PDF](https://arxiv.org/pdf/2507.16754)  

**Abstract**: Large Language Models (LLMs) have shown promise in assisting developers with code-related questions; however, LLMs carry the risk of generating unreliable answers. To address this, Retrieval-Augmented Generation (RAG) has been proposed to reduce the unreliability (i.e., hallucinations) of LLMs. However, designing effective pipelines remains challenging due to numerous design choices. In this paper, we construct a retrieval corpus of over 3 million Java and Python related Stack Overflow posts with accepted answers, and explore various RAG pipeline designs to answer developer questions, evaluating their effectiveness in generating accurate and reliable responses. More specifically, we (1) design and evaluate 7 different RAG pipelines and 63 pipeline variants to answer questions that have historically similar matches, and (2) address new questions without any close prior matches by automatically lowering the similarity threshold during retrieval, thereby increasing the chance of finding partially relevant context and improving coverage for unseen cases. We find that implementing a RAG pipeline combining hypothetical-documentation-embedding (HyDE) with the full-answer context performs best in retrieving and answering similarcontent for Stack Overflow questions. Finally, we apply our optimal RAG pipeline to 4 open-source LLMs and compare the results to their zero-shot performance. Our findings show that RAG with our optimal RAG pipeline consistently outperforms zero-shot baselines across models, achieving higher scores for helpfulness, correctness, and detail with LLM-as-a-judge. These findings demonstrate that our optimal RAG pipelines robustly enhance answer quality for a wide range of developer queries including both previously seen and novel questions across different LLMs 

---
# AI-enhanced conversational agents for personalized asthma support Factors for engagement, value and efficacy 

**Authors**: Laura Moradbakhti, Dorian Peters, Jennifer K. Quint, Björn Schuller, Darren Cook, Rafael A. Calvo  

**Link**: [PDF](https://arxiv.org/pdf/2507.16735)  

**Abstract**: Asthma-related deaths in the UK are the highest in Europe, and only 30% of patients access basic care. There is a need for alternative approaches to reaching people with asthma in order to provide health education, self-management support and bridges to care. Automated conversational agents (specifically, mobile chatbots) present opportunities for providing alternative and individually tailored access to health education, self-management support and risk self-assessment. But would patients engage with a chatbot, and what factors influence engagement? We present results from a patient survey (N=1257) devised by a team of asthma clinicians, patients, and technology developers, conducted to identify optimal factors for efficacy, value and engagement for a chatbot. Results indicate that most adults with asthma (53%) are interested in using a chatbot and the patients most likely to do so are those who believe their asthma is more serious and who are less confident about self-management. Results also indicate enthusiasm for 24/7 access, personalisation, and for WhatsApp as the preferred access method (compared to app, voice assistant, SMS or website). Obstacles to uptake include security/privacy concerns and skepticism of technological capabilities. We present detailed findings and consolidate these into 7 recommendations for developers for optimising efficacy of chatbot-based health support. 

---
# RAVine: Reality-Aligned Evaluation for Agentic Search 

**Authors**: Yilong Xu, Xiang Long, Zhi Zheng, Jinhua Gao  

**Link**: [PDF](https://arxiv.org/pdf/2507.16725)  

**Abstract**: Agentic search, as a more autonomous and adaptive paradigm of retrieval augmentation, is driving the evolution of intelligent search systems. However, existing evaluation frameworks fail to align well with the goals of agentic search. First, the complex queries commonly used in current benchmarks often deviate from realistic user search scenarios. Second, prior approaches tend to introduce noise when extracting ground truth for end-to-end evaluations, leading to distorted assessments at a fine-grained level. Third, most current frameworks focus solely on the quality of final answers, neglecting the evaluation of the iterative process inherent to agentic search. To address these limitations, we propose RAVine -- a Reality-Aligned eValuation framework for agentic LLMs with search. RAVine targets multi-point queries and long-form answers that better reflect user intents, and introduces an attributable ground truth construction strategy to enhance the accuracy of fine-grained evaluation. Moreover, RAVine examines model's interaction with search tools throughout the iterative process, and accounts for factors of efficiency. We benchmark a series of models using RAVine and derive several insights, which we hope will contribute to advancing the development of agentic search systems. The code and datasets are available at this https URL. 

---
# Experience is the Best Teacher: Grounding VLMs for Robotics through Self-Generated Memory 

**Authors**: Guowei Lan, Kaixian Qu, René Zurbrügg, Changan Chen, Christopher E. Mower, Haitham Bou-Ammar, Marco Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2507.16713)  

**Abstract**: Vision-language models (VLMs) have been widely adopted in robotics to enable autonomous planning. However, grounding VLMs, originally trained on internet data, to diverse real-world robots remains a challenge. This paper presents ExpTeach, a framework that grounds VLMs to physical robots by building a self-generated memory of real-world experiences. In ExpTeach, the VLM autonomously plans actions, verifies outcomes, reflects on failures, and adapts robot behaviors in a closed loop. The self-generated experiences during this process are then summarized into a long-term memory, enabling retrieval of learned knowledge to guide future tasks via retrieval-augmented generation (RAG). Additionally, ExpTeach enhances the spatial understanding of VLMs with an on-demand image annotation module. In experiments, we show that reflection improves success rates from 36% to 84% on four challenging robotic tasks and observe the emergence of intelligent object interactions, including creative tool use. Across extensive tests on 12 real-world scenarios (including eight unseen ones), we find that grounding with long-term memory boosts single-trial success rates from 22% to 80%, demonstrating the effectiveness and generalizability of ExpTeach. 

---
# Advancing Risk and Quality Assurance: A RAG Chatbot for Improved Regulatory Compliance 

**Authors**: Lars Hillebrand, Armin Berger, Daniel Uedelhoven, David Berghaus, Ulrich Warning, Tim Dilmaghani, Bernd Kliem, Thomas Schmid, Rüdiger Loitz, Rafet Sifa  

**Link**: [PDF](https://arxiv.org/pdf/2507.16711)  

**Abstract**: Risk and Quality (R&Q) assurance in highly regulated industries requires constant navigation of complex regulatory frameworks, with employees handling numerous daily queries demanding accurate policy interpretation. Traditional methods relying on specialized experts create operational bottlenecks and limit scalability. We present a novel Retrieval Augmented Generation (RAG) system leveraging Large Language Models (LLMs), hybrid search and relevance boosting to enhance R&Q query processing. Evaluated on 124 expert-annotated real-world queries, our actively deployed system demonstrates substantial improvements over traditional RAG approaches. Additionally, we perform an extensive hyperparameter analysis to compare and evaluate multiple configuration setups, delivering valuable insights to practitioners. 

---
# Screen2AX: Vision-Based Approach for Automatic macOS Accessibility Generation 

**Authors**: Viktor Muryn, Marta Sumyk, Mariya Hirna, Sofiya Garkot, Maksym Shamrai  

**Link**: [PDF](https://arxiv.org/pdf/2507.16704)  

**Abstract**: Desktop accessibility metadata enables AI agents to interpret screens and supports users who depend on tools like screen readers. Yet, many applications remain largely inaccessible due to incomplete or missing metadata provided by developers - our investigation shows that only 33% of applications on macOS offer full accessibility support. While recent work on structured screen representation has primarily addressed specific challenges, such as UI element detection or captioning, none has attempted to capture the full complexity of desktop interfaces by replicating their entire hierarchical structure. To bridge this gap, we introduce Screen2AX, the first framework to automatically create real-time, tree-structured accessibility metadata from a single screenshot. Our method uses vision-language and object detection models to detect, describe, and organize UI elements hierarchically, mirroring macOS's system-level accessibility structure. To tackle the limited availability of data for macOS desktop applications, we compiled and publicly released three datasets encompassing 112 macOS applications, each annotated for UI element detection, grouping, and hierarchical accessibility metadata alongside corresponding screenshots. Screen2AX accurately infers hierarchy trees, achieving a 77% F1 score in reconstructing a complete accessibility tree. Crucially, these hierarchy trees improve the ability of autonomous agents to interpret and interact with complex desktop interfaces. We introduce Screen2AX-Task, a benchmark specifically designed for evaluating autonomous agent task execution in macOS desktop environments. Using this benchmark, we demonstrate that Screen2AX delivers a 2.2x performance improvement over native accessibility representations and surpasses the state-of-the-art OmniParser V2 system on the ScreenSpot benchmark. 

---
# FISHER: A Foundation Model for Multi-Modal Industrial Signal Comprehensive Representation 

**Authors**: Pingyi Fan, Anbai Jiang, Shuwei Zhang, Zhiqiang Lv, Bing Han, Xinhu Zheng, Wenrui Liang, Junjie Li, Wei-Qiang Zhang, Yanmin Qian, Xie Chen, Cheng Lu, Jia Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.16696)  

**Abstract**: With the rapid deployment of SCADA systems, how to effectively analyze industrial signals and detect abnormal states is an urgent need for the industry. Due to the significant heterogeneity of these signals, which we summarize as the M5 problem, previous works only focus on small sub-problems and employ specialized models, failing to utilize the synergies between modalities and the powerful scaling law. However, we argue that the M5 signals can be modeled in a unified manner due to the intrinsic similarity. As a result, we propose FISHER, a Foundation model for multi-modal Industrial Signal compreHEnsive Representation. To support arbitrary sampling rates, FISHER considers the increment of sampling rate as the concatenation of sub-band information. Specifically, FISHER takes the STFT sub-band as the modeling unit and adopts a teacher student SSL framework for pre-training. We also develop the RMIS benchmark, which evaluates the representations of M5 industrial signals on multiple health management tasks. Compared with top SSL models, FISHER showcases versatile and outstanding capabilities with a general performance gain up to 5.03%, along with much more efficient scaling curves. We also investigate the scaling law on downstream tasks and derive potential avenues for future works. FISHER is now open-sourced on this https URL 

---
# Interpretable Topic Extraction and Word Embedding Learning using row-stochastic DEDICOM 

**Authors**: Lars Hillebrand, David Biesner, Christian Bauckhage, Rafet Sifa  

**Link**: [PDF](https://arxiv.org/pdf/2507.16695)  

**Abstract**: The DEDICOM algorithm provides a uniquely interpretable matrix factorization method for symmetric and asymmetric square matrices. We employ a new row-stochastic variation of DEDICOM on the pointwise mutual information matrices of text corpora to identify latent topic clusters within the vocabulary and simultaneously learn interpretable word embeddings. We introduce a method to efficiently train a constrained DEDICOM algorithm and a qualitative evaluation of its topic modeling and word embedding performance. 

---
# PICACO: Pluralistic In-Context Value Alignment of LLMs via Total Correlation Optimization 

**Authors**: Han Jiang, Dongyao Zhu, Zhihua Wei, Xiaoyuan Yi, Ziang Xiao, Xing Xie  

**Link**: [PDF](https://arxiv.org/pdf/2507.16679)  

**Abstract**: In-Context Learning has shown great potential for aligning Large Language Models (LLMs) with human values, helping reduce harmful outputs and accommodate diverse preferences without costly post-training, known as In-Context Alignment (ICA). However, LLMs' comprehension of input prompts remains agnostic, limiting ICA's ability to address value tensions--human values are inherently pluralistic, often imposing conflicting demands, e.g., stimulation vs. tradition. Current ICA methods therefore face the Instruction Bottleneck challenge, where LLMs struggle to reconcile multiple intended values within a single prompt, leading to incomplete or biased alignment. To address this, we propose PICACO, a novel pluralistic ICA method. Without fine-tuning, PICACO optimizes a meta-instruction that navigates multiple values to better elicit LLMs' understanding of them and improve their alignment. This is achieved by maximizing the total correlation between specified values and LLM responses, theoretically reinforcing value correlation while reducing distractive noise, resulting in effective value instructions. Extensive experiments on five value sets show that PICACO works well with both black-box and open-source LLMs, outperforms several recent strong baselines, and achieves a better balance across up to 8 distinct values. 

---
# Meta-Learning for Cold-Start Personalization in Prompt-Tuned LLMs 

**Authors**: Yushang Zhao, Huijie Shen, Dannier Li, Lu Chang, Chengrui Zhou, Yinuo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.16672)  

**Abstract**: Generative, explainable, and flexible recommender systems, derived using Large Language Models (LLM) are promising and poorly adapted to the cold-start user situation, where there is little to no history of interaction. The current solutions i.e. supervised fine-tuning and collaborative filtering are dense-user-item focused and would be expensive to maintain and update. This paper introduces a meta-learning framework, that can be used to perform parameter-efficient prompt-tuning, to effectively personalize LLM-based recommender systems quickly at cold-start. The model learns soft prompt embeddings with first-order (Reptile) and second-order (MAML) optimization by treating each of the users as the tasks. As augmentations to the input tokens, these learnable vectors are the differentiable control variables that represent user behavioral priors. The prompts are meta-optimized through episodic sampling, inner-loop adaptation, and outer-loop generalization. On MovieLens-1M, Amazon Reviews, and Recbole, we can see that our adaptive model outperforms strong baselines in NDCG@10, HR@10, and MRR, and it runs in real-time (i.e., below 300 ms) on consumer GPUs. Zero-history personalization is also supported by this scalable solution, and its 275 ms rate of adaptation allows successful real-time risk profiling of financial systems by shortening detection latency and improving payment network stability. Crucially, the 275 ms adaptation capability can enable real-time risk profiling for financial institutions, reducing systemic vulnerability detection latency significantly versus traditional compliance checks. By preventing contagion in payment networks (e.g., Fedwire), the framework strengthens national financial infrastructure resilience. 

---
# Self-Contradiction as Self-Improvement: Mitigating the Generation-Understanding Gap in MLLMs 

**Authors**: Yujin Han, Hao Chen, Andi Han, Zhiheng Wang, Xinyu Lin, Yingya Zhang, Shiwei Zhang, Difan Zou  

**Link**: [PDF](https://arxiv.org/pdf/2507.16663)  

**Abstract**: Despite efforts to unify multimodal generation and understanding tasks in a single model, we show these MLLMs exhibit self-contradiction where generation produces images deemed misaligned with input prompts based on the model's own understanding. We define a Nonunified score that quantifies such self-contradiction. Our empirical results reveal that the self-contradiction mainly arises from weak generation that fails to align with prompts, rather than misunderstanding. This capability asymmetry indicates the potential of leveraging self-contradiction for self-improvement, where the stronger model understanding guides the weaker generation to mitigate the generation-understanding gap. Applying standard post-training methods (e.g., SFT, DPO) with such internal supervision successfully improves both generation and unification. We discover a co-improvement effect on both generation and understanding when only fine-tuning the generation branch, a phenomenon known in pre-training but underexplored in post-training. Our analysis shows improvements stem from better detection of false positives that are previously incorrectly identified as prompt-aligned. Theoretically, we show the aligned training dynamics between generation and understanding allow reduced prompt-misaligned generations to also improve mismatch detection in the understanding branch. Additionally, the framework reveals a potential risk of co-degradation under poor supervision-an overlooked phenomenon that is empirically validated in our experiments. Notably, we find intrinsic metrics like Nonunified score cannot distinguish co-degradation from co-improvement, which highlights the necessity of data quality check. Finally, we propose a curriculum-based strategy based on our findings that gradually introduces harder samples as the model improves, leading to better unification and improved MLLM generation and understanding. 

---
# Towards Automated Regulatory Compliance Verification in Financial Auditing with Large Language Models 

**Authors**: Armin Berger, Lars Hillebrand, David Leonhard, Tobias Deußer, Thiago Bell Felix de Oliveira, Tim Dilmaghani, Mohamed Khaled, Bernd Kliem, Rüdiger Loitz, Christian Bauckhage, Rafet Sifa  

**Link**: [PDF](https://arxiv.org/pdf/2507.16642)  

**Abstract**: The auditing of financial documents, historically a labor-intensive process, stands on the precipice of transformation. AI-driven solutions have made inroads into streamlining this process by recommending pertinent text passages from financial reports to align with the legal requirements of accounting standards. However, a glaring limitation remains: these systems commonly fall short in verifying if the recommended excerpts indeed comply with the specific legal mandates. Hence, in this paper, we probe the efficiency of publicly available Large Language Models (LLMs) in the realm of regulatory compliance across different model configurations. We place particular emphasis on comparing cutting-edge open-source LLMs, such as Llama-2, with their proprietary counterparts like OpenAI's GPT models. This comparative analysis leverages two custom datasets provided by our partner PricewaterhouseCoopers (PwC) Germany. We find that the open-source Llama-2 70 billion model demonstrates outstanding performance in detecting non-compliance or true negative occurrences, beating all their proprietary counterparts. Nevertheless, proprietary models such as GPT-4 perform the best in a broad variety of scenarios, particularly in non-English contexts. 

---
# An Experimental Study of Split-Learning TinyML on Ultra-Low-Power Edge/IoT Nodes 

**Authors**: Zied Jenhani, Mounir Bensalem, Jasenka Dizdarević, Admela Jukan  

**Link**: [PDF](https://arxiv.org/pdf/2507.16594)  

**Abstract**: Running deep learning inference directly on ultra-low-power edge/IoT nodes has been limited by the tight memory and compute budgets of microcontrollers. Split learning (SL) addresses this limitation in which it executes part of the inference process on the sensor and off-loads the remainder to a companion device. In the context of constrained devices and the related impact of low-power, over-the-air transport protocols, the performance of split learning remains largely unexplored. TO the best of our knowledge, this paper presents the first end-to-end TinyML + SL testbed built on Espressif ESP32-S3 boards, designed to benchmark the over-the-air performance of split learning TinyML in edge/IoT environments. We benchmark the performance of a MobileNetV2 image recognition model, which is quantized to 8-bit integers, partitioned, and delivered to the nodes via over-the-air updates. The intermediate activations are exchanged through different wireless communication methods: ESP-NOW, BLE, and traditional UDP/IP and TCP/IP, enabling a head-to-head comparison on identical hardware. Measurements show that splitting the model after block_16_project_BN layer generates a 5.66 kB tensor that traverses the link in 3.2 ms, when UDP is used, achieving a steady-state round-trip latency of 5.8 s. ESP-NOW presents the most favorable RTT performance 3.7 s; BLE extends battery life further but increases latency beyond 10s. 

---
# AI for Better UX in Computer-Aided Engineering: Is Academia Catching Up with Industry Demands? A Multivocal Literature Review 

**Authors**: Choro Ulan Uulu, Mikhail Kulyabin, Layan Etaiwi, Nuno Miguel Martins Pacheco, Jan Joosten, Kerstin Röse, Filippos Petridis, Jan Bosch, Helena Holmström Olsson  

**Link**: [PDF](https://arxiv.org/pdf/2507.16586)  

**Abstract**: Computer-Aided Engineering (CAE) enables simulation experts to optimize complex models, but faces challenges in user experience (UX) that limit efficiency and accessibility. While artificial intelligence (AI) has demonstrated potential to enhance CAE processes, research integrating these fields with a focus on UX remains fragmented. This paper presents a multivocal literature review (MLR) examining how AI enhances UX in CAE software across both academic research and industry implementations. Our analysis reveals significant gaps between academic explorations and industry applications, with companies actively implementing LLMs, adaptive UIs, and recommender systems while academic research focuses primarily on technical capabilities without UX validation. Key findings demonstrate opportunities in AI-powered guidance, adaptive interfaces, and workflow automation that remain underexplored in current research. By mapping the intersection of these domains, this study provides a foundation for future work to address the identified research gaps and advance the integration of AI to improve CAE user experience. 

---
# Pyramid Hierarchical Masked Diffusion Model for Imaging Synthesis 

**Authors**: Xiaojiao Xiao, Qinmin Vivian Hu, Guanghui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.16579)  

**Abstract**: Medical image synthesis plays a crucial role in clinical workflows, addressing the common issue of missing imaging modalities due to factors such as extended scan times, scan corruption, artifacts, patient motion, and intolerance to contrast agents. The paper presents a novel image synthesis network, the Pyramid Hierarchical Masked Diffusion Model (PHMDiff), which employs a multi-scale hierarchical approach for more detailed control over synthesizing high-quality images across different resolutions and layers. Specifically, this model utilizes randomly multi-scale high-proportion masks to speed up diffusion model training, and balances detail fidelity and overall structure. The integration of a Transformer-based Diffusion model process incorporates cross-granularity regularization, modeling the mutual information consistency across each granularity's latent spaces, thereby enhancing pixel-level perceptual accuracy. Comprehensive experiments on two challenging datasets demonstrate that PHMDiff achieves superior performance in both the Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index Measure (SSIM), highlighting its capability to produce high-quality synthesized images with excellent structural integrity. Ablation studies further confirm the contributions of each component. Furthermore, the PHMDiff model, a multi-scale image synthesis framework across and within medical imaging modalities, shows significant advantages over other methods. The source code is available at this https URL 

---
# Data-Driven Adaptive Gradient Recovery for Unstructured Finite Volume Computations 

**Authors**: G. de Romémont, F. Renac, F. Chinesta, J. Nunez, D. Gueyffier  

**Link**: [PDF](https://arxiv.org/pdf/2507.16571)  

**Abstract**: We present a novel data-driven approach for enhancing gradient reconstruction in unstructured finite volume methods for hyperbolic conservation laws, specifically for the 2D Euler equations. Our approach extends previous structured-grid methodologies to unstructured meshes through a modified DeepONet architecture that incorporates local geometry in the neural network. The architecture employs local mesh topology to ensure rotation invariance, while also ensuring first-order constraint on the learned operator. The training methodology incorporates physics-informed regularization through entropy penalization, total variation diminishing penalization, and parameter regularization to ensure physically consistent solutions, particularly in shock-dominated regions. The model is trained on high-fidelity datasets solutions derived from sine waves and randomized piecewise constant initial conditions with periodic boundary conditions, enabling robust generalization to complex flow configurations or geometries. Validation test cases from the literature, including challenging geometry configuration, demonstrates substantial improvements in accuracy compared to traditional second-order finite volume schemes. The method achieves gains of 20-60% in solution accuracy while enhancing computational efficiency. A convergence study has been conveyed and reveal improved mesh convergence rates compared to the conventional solver. The proposed algorithm is faster and more accurate than the traditional second-order finite volume solver, enabling high-fidelity simulations on coarser grids while preserving the stability and conservation properties essential for hyperbolic conservation laws. This work is a part of a new generation of solvers that are built by combining Machine-Learning (ML) tools with traditional numerical schemes, all while ensuring physical constraint on the results. 

---
# TTMBA: Towards Text To Multiple Sources Binaural Audio Generation 

**Authors**: Yuxuan He, Xiaoran Yang, Ningning Pan, Gongping Huang  

**Link**: [PDF](https://arxiv.org/pdf/2507.16564)  

**Abstract**: Most existing text-to-audio (TTA) generation methods produce mono outputs, neglecting essential spatial information for immersive auditory experiences. To address this issue, we propose a cascaded method for text-to-multisource binaural audio generation (TTMBA) with both temporal and spatial control. First, a pretrained large language model (LLM) segments the text into a structured format with time and spatial details for each sound event. Next, a pretrained mono audio generation network creates multiple mono audios with varying durations for each event. These mono audios are transformed into binaural audios using a binaural rendering neural network based on spatial data from the LLM. Finally, the binaural audios are arranged by their start times, resulting in multisource binaural audio. Experimental results demonstrate the superiority of the proposed method in terms of both audio generation quality and spatial perceptual accuracy. 

---
# Evaluating Social Acceptance of eXtended Reality (XR) Agent Technology: A User Study (Extended Version) 

**Authors**: Megha Quamara, Viktor Schmuck, Cristina Iani, Axel Primavesi, Alexander Plaum, Luca Vigano  

**Link**: [PDF](https://arxiv.org/pdf/2507.16562)  

**Abstract**: In this paper, we present the findings of a user study that evaluated the social acceptance of eXtended Reality (XR) agent technology, focusing on a remotely accessible, web-based XR training system developed for journalists. This system involves user interaction with a virtual avatar, enabled by a modular toolkit. The interactions are designed to provide tailored training for journalists in digital-remote settings, especially for sensitive or dangerous scenarios, without requiring specialized end-user equipment like headsets. Our research adapts and extends the Almere model, representing social acceptance through existing attributes such as perceived ease of use and perceived usefulness, along with added ones like dependability and security in the user-agent interaction. The XR agent was tested through a controlled experiment in a real-world setting, with data collected on users' perceptions. Our findings, based on quantitative and qualitative measurements involving questionnaires, contribute to the understanding of user perceptions and acceptance of XR agent solutions within a specific social context, while also identifying areas for the improvement of XR systems. 

---
# Optimization of DNN-based HSI Segmentation FPGA-based SoC for ADS: A Practical Approach 

**Authors**: Jon Gutiérrez-Zaballa, Koldo Basterretxea, Javier Echanobe  

**Link**: [PDF](https://arxiv.org/pdf/2507.16556)  

**Abstract**: The use of HSI for autonomous navigation is a promising research field aimed at improving the accuracy and robustness of detection, tracking, and scene understanding systems based on vision sensors. Combining advanced computer algorithms, such as DNNs, with small-size snapshot HSI cameras enhances the reliability of these systems. HSI overcomes intrinsic limitations of greyscale and RGB imaging in depicting physical properties of targets, particularly regarding spectral reflectance and metamerism. Despite promising results in HSI-based vision developments, safety-critical systems like ADS demand strict constraints on latency, resource consumption, and security, motivating the shift of ML workloads to edge platforms. This involves a thorough software/hardware co-design scheme to distribute and optimize the tasks efficiently among the limited resources of computing platforms. With respect to inference, the over-parameterized nature of DNNs poses significant computational challenges for real-time on-the-edge deployment. In addition, the intensive data preprocessing required by HSI, which is frequently overlooked, must be carefully managed in terms of memory arrangement and inter-task communication to enable an efficient integrated pipeline design on a SoC. This work presents a set of optimization techniques for the practical co-design of a DNN-based HSI segmentation processor deployed on a FPGA-based SoC targeted at ADS, including key optimizations such as functional software/hardware task distribution, hardware-aware preprocessing, ML model compression, and a complete pipelined deployment. Applied compression techniques significantly reduce the complexity of the designed DNN to 24.34% of the original operations and to 1.02% of the original number of parameters, achieving a 2.86x speed-up in the inference task without noticeable degradation of the segmentation accuracy. 

---
# A Comprehensive Data-centric Overview of Federated Graph Learning 

**Authors**: Zhengyu Wu, Xunkai Li, Yinlin Zhu, Zekai Chen, Guochen Yan, Yanyu Yan, Hao Zhang, Yuming Ai, Xinmo Jin, Rong-Hua Li, Guoren Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.16541)  

**Abstract**: In the era of big data applications, Federated Graph Learning (FGL) has emerged as a prominent solution that reconcile the tradeoff between optimizing the collective intelligence between decentralized datasets holders and preserving sensitive information to maximum. Existing FGL surveys have contributed meaningfully but largely focus on integrating Federated Learning (FL) and Graph Machine Learning (GML), resulting in early stage taxonomies that emphasis on methodology and simulated scenarios. Notably, a data centric perspective, which systematically examines FGL methods through the lens of data properties and usage, remains unadapted to reorganize FGL research, yet it is critical to assess how FGL studies manage to tackle data centric constraints to enhance model performances. This survey propose a two-level data centric taxonomy: Data Characteristics, which categorizes studies based on the structural and distributional properties of datasets used in FGL, and Data Utilization, which analyzes the training procedures and techniques employed to overcome key data centric challenges. Each taxonomy level is defined by three orthogonal criteria, each representing a distinct data centric configuration. Beyond taxonomy, this survey examines FGL integration with Pretrained Large Models, showcases realistic applications, and highlights future direction aligned with emerging trends in GML. 

---
# Symbolic Graph Intelligence: Hypervector Message Passing for Learning Graph-Level Patterns with Tsetlin Machines 

**Authors**: Christian D. Blakely  

**Link**: [PDF](https://arxiv.org/pdf/2507.16537)  

**Abstract**: We propose a multilayered symbolic framework for general graph classification that leverages sparse binary hypervectors and Tsetlin Machines. Each graph is encoded through structured message passing, where node, edge, and attribute information are bound and bundled into a symbolic hypervector. This process preserves the hierarchical semantics of the graph through layered binding from node attributes to edge relations to structural roles resulting in a compact, discrete representation. We also formulate a local interpretability framework which lends itself to a key advantage of our approach being locally interpretable. We validate our method on TUDataset benchmarks, demonstrating competitive accuracy with strong symbolic transparency compared to neural graph models. 

---
# EarthCrafter: Scalable 3D Earth Generation via Dual-Sparse Latent Diffusion 

**Authors**: Shang Liu, Chenjie Cao, Chaohui Yu, Wen Qian, Jing Wang, Fan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.16535)  

**Abstract**: Despite the remarkable developments achieved by recent 3D generation works, scaling these methods to geographic extents, such as modeling thousands of square kilometers of Earth's surface, remains an open challenge. We address this through a dual innovation in data infrastructure and model architecture. First, we introduce Aerial-Earth3D, the largest 3D aerial dataset to date, consisting of 50k curated scenes (each measuring 600m x 600m) captured across the U.S. mainland, comprising 45M multi-view Google Earth frames. Each scene provides pose-annotated multi-view images, depth maps, normals, semantic segmentation, and camera poses, with explicit quality control to ensure terrain diversity. Building on this foundation, we propose EarthCrafter, a tailored framework for large-scale 3D Earth generation via sparse-decoupled latent diffusion. Our architecture separates structural and textural generation: 1) Dual sparse 3D-VAEs compress high-resolution geometric voxels and textural 2D Gaussian Splats (2DGS) into compact latent spaces, largely alleviating the costly computation suffering from vast geographic scales while preserving critical information. 2) We propose condition-aware flow matching models trained on mixed inputs (semantics, images, or neither) to flexibly model latent geometry and texture features independently. Extensive experiments demonstrate that EarthCrafter performs substantially better in extremely large-scale generation. The framework further supports versatile applications, from semantic-guided urban layout generation to unconditional terrain synthesis, while maintaining geographic plausibility through our rich data priors from Aerial-Earth3D. 

---
# confopt: A Library for Implementation and Evaluation of Gradient-based One-Shot NAS Methods 

**Authors**: Abhash Kumar Jha, Shakiba Moradian, Arjun Krishnakumar, Martin Rapp, Frank Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2507.16533)  

**Abstract**: Gradient-based one-shot neural architecture search (NAS) has significantly reduced the cost of exploring architectural spaces with discrete design choices, such as selecting operations within a model. However, the field faces two major challenges. First, evaluations of gradient-based NAS methods heavily rely on the DARTS benchmark, despite the existence of other available benchmarks. This overreliance has led to saturation, with reported improvements often falling within the margin of noise. Second, implementations of gradient-based one-shot NAS methods are fragmented across disparate repositories, complicating fair and reproducible comparisons and further development. In this paper, we introduce Configurable Optimizer (confopt), an extensible library designed to streamline the development and evaluation of gradient-based one-shot NAS methods. Confopt provides a minimal API that makes it easy for users to integrate new search spaces, while also supporting the decomposition of NAS optimizers into their core components. We use this framework to create a suite of new DARTS-based benchmarks, and combine them with a novel evaluation protocol to reveal a critical flaw in how gradient-based one-shot NAS methods are currently assessed. The code can be found at this https URL. 

---
# Spatial 3D-LLM: Exploring Spatial Awareness in 3D Vision-Language Models 

**Authors**: Xiaoyan Wang, Zeju Li, Yifan Xu, Jiaxing Qi, Zhifei Yang, Ruifei Ma, Xiangde Liu, Chao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.16524)  

**Abstract**: New era has unlocked exciting possibilities for extending Large Language Models (LLMs) to tackle 3D vision-language tasks. However, most existing 3D multimodal LLMs (MLLMs) rely on compressing holistic 3D scene information or segmenting independent objects to perform these tasks, which limits their spatial awareness due to insufficient representation of the richness inherent in 3D scenes. To overcome these limitations, we propose Spatial 3D-LLM, a 3D MLLM specifically designed to enhance spatial awareness for 3D vision-language tasks by enriching the spatial embeddings of 3D scenes. Spatial 3D-LLM integrates an LLM backbone with a progressive spatial awareness scheme that progressively captures spatial information as the perception field expands, generating location-enriched 3D scene embeddings to serve as visual prompts. Furthermore, we introduce two novel tasks: 3D object distance measurement and 3D layout editing, and construct a 3D instruction dataset, MODEL, to evaluate the model's spatial awareness capabilities. Experimental results demonstrate that Spatial 3D-LLM achieves state-of-the-art performance across a wide range of 3D vision-language tasks, revealing the improvements stemmed from our progressive spatial awareness scheme of mining more profound spatial information. Our code is available at this https URL. 

---
# The Ever-Evolving Science Exam 

**Authors**: Junying Wang, Zicheng Zhang, Yijin Guo, Farong Wen, Ye Shen, Yingji Liang, Yalun Wu, Wenzhe Li, Chunyi Li, Zijian Chen, Qi Jia, Guangtao Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2507.16514)  

**Abstract**: As foundation models grow rapidly in capability and deployment, evaluating their scientific understanding becomes increasingly critical. Existing science benchmarks have made progress towards broad **Range**, wide **Reach**, and high **Rigor**, yet they often face two major challenges: **data leakage risks** that compromise benchmarking validity, and **evaluation inefficiency** due to large-scale testing. To address these issues, we introduce the **Ever-Evolving Science Exam (EESE)**, a dynamic benchmark designed to reliably assess scientific capabilities in foundation models. Our approach consists of two components: 1) a non-public **EESE-Pool** with over 100K expertly constructed science instances (question-answer pairs) across 5 disciplines and 500+ subfields, built through a multi-stage pipeline ensuring **Range**, **Reach**, and **Rigor**, 2) a periodically updated 500-instance subset **EESE**, sampled and validated to enable leakage-resilient, low-overhead evaluations. Experiments on 32 open- and closed-source models demonstrate that EESE effectively differentiates the strengths and weaknesses of models in scientific fields and cognitive dimensions. Overall, EESE provides a robust, scalable, and forward-compatible solution for science benchmark design, offering a realistic measure of how well foundation models handle science questions. The project page is at: this https URL. 

---
# Analogy making as amortised model construction 

**Authors**: David G. Nagy, Tingke Shen, Hanqi Zhou, Charley M. Wu, Peter Dayan  

**Link**: [PDF](https://arxiv.org/pdf/2507.16511)  

**Abstract**: Humans flexibly construct internal models to navigate novel situations. To be useful, these internal models must be sufficiently faithful to the environment that resource-limited planning leads to adequate outcomes; equally, they must be tractable to construct in the first place. We argue that analogy plays a central role in these processes, enabling agents to reuse solution-relevant structure from past experiences and amortise the computational costs of both model construction (construal) and planning. Formalising analogies as partial homomorphisms between Markov decision processes, we sketch a framework in which abstract modules, derived from previous construals, serve as composable building blocks for new ones. This modular reuse allows for flexible adaptation of policies and representations across domains with shared structural essence. 

---
# ICR Probe: Tracking Hidden State Dynamics for Reliable Hallucination Detection in LLMs 

**Authors**: Zhenliang Zhang, Xinyu Hu, Huixuan Zhang, Junzhe Zhang, Xiaojun Wan  

**Link**: [PDF](https://arxiv.org/pdf/2507.16488)  

**Abstract**: Large language models (LLMs) excel at various natural language processing tasks, but their tendency to generate hallucinations undermines their reliability. Existing hallucination detection methods leveraging hidden states predominantly focus on static and isolated representations, overlooking their dynamic evolution across layers, which limits efficacy. To address this limitation, we shift the focus to the hidden state update process and introduce a novel metric, the ICR Score (Information Contribution to Residual Stream), which quantifies the contribution of modules to the hidden states' update. We empirically validate that the ICR Score is effective and reliable in distinguishing hallucinations. Building on these insights, we propose a hallucination detection method, the ICR Probe, which captures the cross-layer evolution of hidden states. Experimental results show that the ICR Probe achieves superior performance with significantly fewer parameters. Furthermore, ablation studies and case analyses offer deeper insights into the underlying mechanism of this method, improving its interpretability. 

---
# Designing for Difference: How Human Characteristics Shape Perceptions of Collaborative Robots 

**Authors**: Sabrina Livanec, Laura Londoño, Michael Gorki, Adrian Röfer, Abhinav Valada, Andrea Kiesel  

**Link**: [PDF](https://arxiv.org/pdf/2507.16480)  

**Abstract**: The development of assistive robots for social collaboration raises critical questions about responsible and inclusive design, especially when interacting with individuals from protected groups such as those with disabilities or advanced age. Currently, research is scarce on how participants assess varying robot behaviors in combination with diverse human needs, likely since participants have limited real-world experience with advanced domestic robots. In the current study, we aim to address this gap while using methods that enable participants to assess robot behavior, as well as methods that support meaningful reflection despite limited experience. In an online study, 112 participants (from both experimental and control groups) evaluated 7 videos from a total of 28 variations of human-robot collaboration types. The experimental group first completed a cognitive-affective mapping (CAM) exercise on human-robot collaboration before providing their ratings. Although CAM reflection did not significantly affect overall ratings, it led to more pronounced assessments for certain combinations of robot behavior and human condition. Most importantly, the type of human-robot collaboration influences the assessment. Antisocial robot behavior was consistently rated as the lowest, while collaboration with aged individuals elicited more sensitive evaluations. Scenarios involving object handovers were viewed more positively than those without them. These findings suggest that both human characteristics and interaction paradigms influence the perceived acceptability of collaborative robots, underscoring the importance of prosocial design. They also highlight the potential of reflective methods, such as CAM, to elicit nuanced feedback, supporting the development of user-centered and socially responsible robotic systems tailored to diverse populations. 

---
# Estimating Treatment Effects with Independent Component Analysis 

**Authors**: Patrik Reizinger, Lester Mackey, Wieland Brendel, Rahul Krishnan  

**Link**: [PDF](https://arxiv.org/pdf/2507.16467)  

**Abstract**: The field of causal inference has developed a variety of methods to accurately estimate treatment effects in the presence of nuisance. Meanwhile, the field of identifiability theory has developed methods like Independent Component Analysis (ICA) to identify latent sources and mixing weights from data. While these two research communities have developed largely independently, they aim to achieve similar goals: the accurate and sample-efficient estimation of model parameters. In the partially linear regression (PLR) setting, Mackey et al. (2018) recently found that estimation consistency can be improved with non-Gaussian treatment noise. Non-Gaussianity is also a crucial assumption for identifying latent factors in ICA. We provide the first theoretical and empirical insights into this connection, showing that ICA can be used for causal effect estimation in the PLR model. Surprisingly, we find that linear ICA can accurately estimate multiple treatment effects even in the presence of Gaussian confounders or nonlinear nuisance. 

---
# Beyond Algorethics: Addressing the Ethical and Anthropological Challenges of AI Recommender Systems 

**Authors**: Octavian M. Machidon  

**Link**: [PDF](https://arxiv.org/pdf/2507.16430)  

**Abstract**: In this paper, I examine the ethical and anthropological challenges posed by AI-driven recommender systems (RSs), which have become central to shaping digital environments and social interactions. By curating personalized content, RSs do not merely reflect user preferences but actively construct individual experiences across social media, entertainment platforms, and e-commerce. Despite their ubiquity, the ethical implications of RSs remain insufficiently explored, even as concerns over privacy, autonomy, and mental well-being intensify. I argue that existing ethical approaches, including algorethics, the effort to embed ethical principles into algorithmic design, are necessary but ultimately inadequate. RSs inherently reduce human complexity to quantifiable dimensions, exploit user vulnerabilities, and prioritize engagement over well-being. Addressing these concerns requires moving beyond purely technical solutions. I propose a comprehensive framework for human-centered RS design, integrating interdisciplinary perspectives, regulatory strategies, and educational initiatives to ensure AI systems foster rather than undermine human autonomy and societal flourishing. 

---
# From Flat to Round: Redefining Brain Decoding with Surface-Based fMRI and Cortex Structure 

**Authors**: Sijin Yu, Zijiao Chen, Wenxuan Wu, Shengxian Chen, Zhongliang Liu, Jingxin Nie, Xiaofen Xing, Xiangmin Xu, Xin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.16389)  

**Abstract**: Reconstructing visual stimuli from human brain activity (e.g., fMRI) bridges neuroscience and computer vision by decoding neural representations. However, existing methods often overlook critical brain structure-function relationships, flattening spatial information and neglecting individual anatomical variations. To address these issues, we propose (1) a novel sphere tokenizer that explicitly models fMRI signals as spatially coherent 2D spherical data on the cortical surface; (2) integration of structural MRI (sMRI) data, enabling personalized encoding of individual anatomical variations; and (3) a positive-sample mixup strategy for efficiently leveraging multiple fMRI scans associated with the same visual stimulus. Collectively, these innovations enhance reconstruction accuracy, biological interpretability, and generalizability across individuals. Experiments demonstrate superior reconstruction performance compared to SOTA methods, highlighting the effectiveness and interpretability of our biologically informed approach. 

---
# Application of LLM Guided Reinforcement Learning in Formation Control with Collision Avoidance 

**Authors**: Chenhao Yao, Zike Yuan, Xiaoxu Liu, Chi Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.16382)  

**Abstract**: Multi-Agent Systems (MAS) excel at accomplishing complex objectives through the collaborative efforts of individual agents. Among the methodologies employed in MAS, Multi-Agent Reinforcement Learning (MARL) stands out as one of the most efficacious algorithms. However, when confronted with the complex objective of Formation Control with Collision Avoidance (FCCA): designing an effective reward function that facilitates swift convergence of the policy network to an optimal solution. In this paper, we introduce a novel framework that aims to overcome this challenge. By giving large language models (LLMs) on the prioritization of tasks and the observable information available to each agent, our framework generates reward functions that can be dynamically adjusted online based on evaluation outcomes by employing more advanced evaluation metrics rather than the rewards themselves. This mechanism enables the MAS to simultaneously achieve formation control and obstacle avoidance in dynamic environments with enhanced efficiency, requiring fewer iterations to reach superior performance levels. Our empirical studies, conducted in both simulation and real-world settings, validate the practicality and effectiveness of our proposed approach. 

---
# Depth Gives a False Sense of Privacy: LLM Internal States Inversion 

**Authors**: Tian Dong, Yan Meng, Shaofeng Li, Guoxing Chen, Zhen Liu, Haojin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.16372)  

**Abstract**: Large Language Models (LLMs) are increasingly integrated into daily routines, yet they raise significant privacy and safety concerns. Recent research proposes collaborative inference, which outsources the early-layer inference to ensure data locality, and introduces model safety auditing based on inner neuron patterns. Both techniques expose the LLM's Internal States (ISs), which are traditionally considered irreversible to inputs due to optimization challenges and the highly abstract representations in deep layers. In this work, we challenge this assumption by proposing four inversion attacks that significantly improve the semantic similarity and token matching rate of inverted inputs. Specifically, we first develop two white-box optimization-based attacks tailored for low-depth and high-depth ISs. These attacks avoid local minima convergence, a limitation observed in prior work, through a two-phase inversion process. Then, we extend our optimization attack under more practical black-box weight access by leveraging the transferability between the source and the derived LLMs. Additionally, we introduce a generation-based attack that treats inversion as a translation task, employing an inversion model to reconstruct inputs. Extensive evaluation of short and long prompts from medical consulting and coding assistance datasets and 6 LLMs validates the effectiveness of our inversion attacks. Notably, a 4,112-token long medical consulting prompt can be nearly perfectly inverted with 86.88 F1 token matching from the middle layer of Llama-3 model. Finally, we evaluate four practical defenses that we found cannot perfectly prevent ISs inversion and draw conclusions for future mitigation design. 

---
# Leveraging Personalized PageRank and Higher-Order Topological Structures for Heterophily Mitigation in Graph Neural Networks 

**Authors**: Yumeng Wang, Zengyi Wo, Wenjun Wang, Xingcheng Fu, Minglai Shao  

**Link**: [PDF](https://arxiv.org/pdf/2507.16347)  

**Abstract**: Graph Neural Networks (GNNs) excel in node classification tasks but often assume homophily, where connected nodes share similar labels. This assumption does not hold in many real-world heterophilic graphs. Existing models for heterophilic graphs primarily rely on pairwise relationships, overlooking multi-scale information from higher-order structures. This leads to suboptimal performance, particularly under noise from conflicting class information across nodes. To address these challenges, we propose HPGNN, a novel model integrating Higher-order Personalized PageRank with Graph Neural Networks. HPGNN introduces an efficient high-order approximation of Personalized PageRank (PPR) to capture long-range and multi-scale node interactions. This approach reduces computational complexity and mitigates noise from surrounding information. By embedding higher-order structural information into convolutional networks, HPGNN effectively models key interactions across diverse graph dimensions. Extensive experiments on benchmark datasets demonstrate HPGNN's effectiveness. The model achieves better performance than five out of seven state-of-the-art methods on heterophilic graphs in downstream tasks while maintaining competitive performance on homophilic graphs. HPGNN's ability to balance multi-scale information and robustness to noise makes it a versatile solution for real-world graph learning challenges. Codes are available at this https URL. 

---
# Detect Any Sound: Open-Vocabulary Sound Event Detection with Multi-Modal Queries 

**Authors**: Pengfei Cai, Yan Song, Qing Gu, Nan Jiang, Haoyu Song, Ian McLoughlin  

**Link**: [PDF](https://arxiv.org/pdf/2507.16343)  

**Abstract**: Most existing sound event detection~(SED) algorithms operate under a closed-set assumption, restricting their detection capabilities to predefined classes. While recent efforts have explored language-driven zero-shot SED by exploiting audio-language models, their performance is still far from satisfactory due to the lack of fine-grained alignment and cross-modal feature fusion. In this work, we propose the Detect Any Sound Model (DASM), a query-based framework for open-vocabulary SED guided by multi-modal queries. DASM formulates SED as a frame-level retrieval task, where audio features are matched against query vectors derived from text or audio prompts. To support this formulation, DASM introduces a dual-stream decoder that explicitly decouples event recognition and temporal localization: a cross-modality event decoder performs query-feature fusion and determines the presence of sound events at the clip-level, while a context network models temporal dependencies for frame-level localization. Additionally, an inference-time attention masking strategy is proposed to leverage semantic relations between base and novel classes, substantially enhancing generalization to novel classes. Experiments on the AudioSet Strong dataset demonstrate that DASM effectively balances localization accuracy with generalization to novel classes, outperforming CLAP-based methods in open-vocabulary setting (+ 7.8 PSDS) and the baseline in the closed-set setting (+ 6.9 PSDS). Furthermore, in cross-dataset zero-shot evaluation on DESED, DASM achieves a PSDS1 score of 42.2, even exceeding the supervised CRNN baseline. The project page is available at this https URL. 

---
# DREAM: Scalable Red Teaming for Text-to-Image Generative Systems via Distribution Modeling 

**Authors**: Boheng Li, Junjie Wang, Yiming Li, Zhiyang Hu, Leyi Qi, Jianshuo Dong, Run Wang, Han Qiu, Zhan Qin, Tianwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.16329)  

**Abstract**: Despite the integration of safety alignment and external filters, text-to-image (T2I) generative models are still susceptible to producing harmful content, such as sexual or violent imagery. This raises serious concerns about unintended exposure and potential misuse. Red teaming, which aims to proactively identify diverse prompts that can elicit unsafe outputs from the T2I system (including the core generative model as well as potential external safety filters and other processing components), is increasingly recognized as an essential method for assessing and improving safety before real-world deployment. Yet, existing automated red teaming approaches often treat prompt discovery as an isolated, prompt-level optimization task, which limits their scalability, diversity, and overall effectiveness. To bridge this gap, in this paper, we propose DREAM, a scalable red teaming framework to automatically uncover diverse problematic prompts from a given T2I system. Unlike most prior works that optimize prompts individually, DREAM directly models the probabilistic distribution of the target system's problematic prompts, which enables explicit optimization over both effectiveness and diversity, and allows efficient large-scale sampling after training. To achieve this without direct access to representative training samples, we draw inspiration from energy-based models and reformulate the objective into simple and tractable objectives. We further introduce GC-SPSA, an efficient optimization algorithm that provide stable gradient estimates through the long and potentially non-differentiable T2I pipeline. The effectiveness of DREAM is validated through extensive experiments, demonstrating that it surpasses 9 state-of-the-art baselines by a notable margin across a broad range of T2I models and safety filters in terms of prompt success rate and diversity. 

---
# Perovskite-R1: A Domain-Specialized LLM for Intelligent Discovery of Precursor Additives and Experimental Design 

**Authors**: Xin-De Wang, Zhi-Rui Chen, Peng-Jie Guo, Ze-Feng Gao, Cheng Mu, Zhong-Yi Lu  

**Link**: [PDF](https://arxiv.org/pdf/2507.16307)  

**Abstract**: Perovskite solar cells (PSCs) have rapidly emerged as a leading contender in next-generation photovoltaic technologies, owing to their exceptional power conversion efficiencies and advantageous material properties. Despite these advances, challenges such as long-term stability, environmental sustainability, and scalable manufacturing continue to hinder their commercialization. Precursor additive engineering has shown promise in addressing these issues by enhancing both the performance and durability of PSCs. However, the explosive growth of scientific literature and the complex interplay of materials, processes, and device architectures make it increasingly difficult for researchers to efficiently access, organize, and utilize domain knowledge in this rapidly evolving field. To address this gap, we introduce Perovskite-R1, a specialized large language model (LLM) with advanced reasoning capabilities tailored for the discovery and design of PSC precursor additives. By systematically mining and curating 1,232 high-quality scientific publications and integrating a comprehensive library of 33,269 candidate materials, we constructed a domain-specific instruction-tuning dataset using automated question-answer generation and chain-of-thought reasoning. Fine-tuning the QwQ-32B model on this dataset resulted in Perovskite-R1, which can intelligently synthesize literature insights and generate innovative and practical solutions for defect passivation and the selection of precursor additives. Experimental validation of several model-proposed strategies confirms their effectiveness in improving material stability and performance. Our work demonstrates the potential of domain-adapted LLMs in accelerating materials discovery and provides a closed-loop framework for intelligent, data-driven advancements in perovskite photovoltaic research. 

---
# Towards Resilient Safety-driven Unlearning for Diffusion Models against Downstream Fine-tuning 

**Authors**: Boheng Li, Renjie Gu, Junjie Wang, Leyi Qi, Yiming Li, Run Wang, Zhan Qin, Tianwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.16302)  

**Abstract**: Text-to-image (T2I) diffusion models have achieved impressive image generation quality and are increasingly fine-tuned for personalized applications. However, these models often inherit unsafe behaviors from toxic pretraining data, raising growing safety concerns. While recent safety-driven unlearning methods have made promising progress in suppressing model toxicity, they are identified to be fragile to downstream fine-tuning, where we reveal that state-of-the-art methods largely fail to retain their effectiveness even when fine-tuned on entirely benign datasets. To mitigate this problem, in this paper, we propose ResAlign, a safety-driven unlearning framework with enhanced resilience against downstream fine-tuning. By modeling downstream fine-tuning as an implicit optimization problem with a Moreau Envelope-based reformulation, ResAlign enables efficient gradient estimation to minimize the recovery of harmful behaviors. Additionally, a meta-learning strategy is proposed to simulate a diverse distribution of fine-tuning scenarios to improve generalization. Extensive experiments across a wide range of datasets, fine-tuning methods, and configurations demonstrate that ResAlign consistently outperforms prior unlearning approaches in retaining safety after downstream fine-tuning while preserving benign generation capability well. 

---
# Understanding Generalization, Robustness, and Interpretability in Low-Capacity Neural Networks 

**Authors**: Yash Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2507.16278)  

**Abstract**: Although modern deep learning often relies on massive over-parameterized models, the fundamental interplay between capacity, sparsity, and robustness in low-capacity networks remains a vital area of study. We introduce a controlled framework to investigate these properties by creating a suite of binary classification tasks from the MNIST dataset with increasing visual difficulty (e.g., 0 and 1 vs. 4 and 9). Our experiments reveal three core findings. First, the minimum model capacity required for successful generalization scales directly with task complexity. Second, these trained networks are robust to extreme magnitude pruning (up to 95% sparsity), revealing the existence of sparse, high-performing subnetworks. Third, we show that over-parameterization provides a significant advantage in robustness against input corruption. Interpretability analysis via saliency maps further confirms that these identified sparse subnetworks preserve the core reasoning process of the original dense models. This work provides a clear, empirical demonstration of the foundational trade-offs governing simple neural networks. 

---
# Reducing GPU Memory Fragmentation via Spatio-Temporal Planning for Efficient Large-Scale Model Training 

**Authors**: Zixiao Huang, Junhao Hu, Hao Lin, Chunyang Zhu, Yueran Tang, Quanlu Zhang, Zhen Guo, Zhenhua Li, Shengen Yan, Zhenhua Zhu, Guohao Dai, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.16274)  

**Abstract**: The rapid scaling of large language models (LLMs) has significantly increased GPU memory pressure, which is further aggravated by training optimization techniques such as virtual pipeline and recomputation that disrupt tensor lifespans and introduce considerable memory fragmentation. Default GPU memory allocators of popular deep learning frameworks like PyTorch use online strategies without knowledge of tensor lifespans, which can waste up to 43\% of memory and cause out-of-memory errors, rendering optimization techniques ineffective or even unusable.
To address this, we introduce STWeaver, a GPU memory allocator for deep learning frameworks that reduces fragmentation by exploiting the spatial and temporal regularity in memory allocation behaviors of training workloads. STWeaver introduces a novel paradigm that combines offline planning with online allocation. The offline planning leverages spatio-temporal regularities to generate a near-optimal allocation plan, while the online allocation handles complex and dynamic models such as Mixture-of-Experts (MoE). Built as a pluggable PyTorch allocator, STWeaver reduces fragmentation ratio on average by 79.2\% (up to 100\%) across both dense and sparse models, with negligible overhead. This enables more efficient, high-throughput training configurations and improves performance by up to 32.5\%. 

---
# SFNet: A Spatio-Frequency Domain Deep Learning Network for Efficient Alzheimer's Disease Diagnosis 

**Authors**: Xinyue Yang, Meiliang Liu, Yunfang Xu, Xiaoxiao Yang, Zhengye Si, Zijin Li, Zhiwen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2507.16267)  

**Abstract**: Alzheimer's disease (AD) is a progressive neurodegenerative disorder that predominantly affects the elderly population and currently has no cure. Magnetic Resonance Imaging (MRI), as a non-invasive imaging technique, is essential for the early diagnosis of AD. MRI inherently contains both spatial and frequency information, as raw signals are acquired in the frequency domain and reconstructed into spatial images via the Fourier transform. However, most existing AD diagnostic models extract features from a single domain, limiting their capacity to fully capture the complex neuroimaging characteristics of the disease. While some studies have combined spatial and frequency information, they are mostly confined to 2D MRI, leaving the potential of dual-domain analysis in 3D MRI unexplored. To overcome this limitation, we propose Spatio-Frequency Network (SFNet), the first end-to-end deep learning framework that simultaneously leverages spatial and frequency domain information to enhance 3D MRI-based AD diagnosis. SFNet integrates an enhanced dense convolutional network to extract local spatial features and a global frequency module to capture global frequency-domain representations. Additionally, a novel multi-scale attention module is proposed to further refine spatial feature extraction. Experiments on the Alzheimer's Disease Neuroimaging Initiative (ANDI) dataset demonstrate that SFNet outperforms existing baselines and reduces computational overhead in classifying cognitively normal (CN) and AD, achieving an accuracy of 95.1%. 

---
# Edge-case Synthesis for Fisheye Object Detection: A Data-centric Perspective 

**Authors**: Seunghyeon Kim, Kyeongryeol Go  

**Link**: [PDF](https://arxiv.org/pdf/2507.16254)  

**Abstract**: Fisheye cameras introduce significant distortion and pose unique challenges to object detection models trained on conventional datasets. In this work, we propose a data-centric pipeline that systematically improves detection performance by focusing on the key question of identifying the blind spots of the model. Through detailed error analysis, we identify critical edge-cases such as confusing class pairs, peripheral distortions, and underrepresented contexts. Then we directly address them through edge-case synthesis. We fine-tuned an image generative model and guided it with carefully crafted prompts to produce images that replicate real-world failure modes. These synthetic images are pseudo-labeled using a high-quality detector and integrated into training. Our approach results in consistent performance gains, highlighting how deeply understanding data and selectively fixing its weaknesses can be impactful in specialized domains like fisheye object detection. 

---
# Efficient RL for optimizing conversation level outcomes with an LLM-based tutor 

**Authors**: Hyunji Nam, Omer Gottesman, Amy Zhang, Dean Foster, Emma Brunskill, Lyle Ungar  

**Link**: [PDF](https://arxiv.org/pdf/2507.16252)  

**Abstract**: Large language models (LLMs) built on existing reinforcement learning with human feedback (RLHF) frameworks typically optimize responses based on immediate turn-level human preferences. However, this approach falls short in multi-turn dialogue settings, such as online math tutoring. We propose a method to enhance LLM-based tutors by representing the dialogue history with a lower-dimensional latent state representation of a student and optimizing a long-term policy to determine high-level actions based on the latent state. The goal is to better align the tutor's behavior with the long-term objective of guiding the student towards solving a target math problem on their own. Our model is lightweight, requiring less computational resources than prior work of training the tutor policy end-to-end to directly output the tutor's next utterance. Our experiment results demonstrate that these modifications lead to improved long-term outcomes compared to prompting in LLM-simulated tutoring tasks. 

---
# HoliTracer: Holistic Vectorization of Geographic Objects from Large-Size Remote Sensing Imagery 

**Authors**: Yu Wang, Bo Dang, Wanchun Li, Wei Chen, Yansheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.16251)  

**Abstract**: With the increasing resolution of remote sensing imagery (RSI), large-size RSI has emerged as a vital data source for high-precision vector mapping of geographic objects. Existing methods are typically constrained to processing small image patches, which often leads to the loss of contextual information and produces fragmented vector outputs. To address these, this paper introduces HoliTracer, the first framework designed to holistically extract vectorized geographic objects from large-size RSI. In HoliTracer, we enhance segmentation of large-size RSI using the Context Attention Net (CAN), which employs a local-to-global attention mechanism to capture contextual dependencies. Furthermore, we achieve holistic vectorization through a robust pipeline that leverages the Mask Contour Reformer (MCR) to reconstruct polygons and the Polygon Sequence Tracer (PST) to trace vertices. Extensive experiments on large-size RSI datasets, including buildings, water bodies, and roads, demonstrate that HoliTracer outperforms state-of-the-art methods. Our code and data are available in this https URL. 

---
# PRAC3 (Privacy, Reputation, Accountability, Consent, Credit, Compensation): Long Tailed Risks of Voice Actors in AI Data-Economy 

**Authors**: Tanusree Sharma, Yihao Zhou, Visar Berisha  

**Link**: [PDF](https://arxiv.org/pdf/2507.16247)  

**Abstract**: Early large-scale audio datasets, such as LibriSpeech, were built with hundreds of individual contributors whose voices were instrumental in the development of speech technologies, including audiobooks and voice assistants. Yet, a decade later, these same contributions have exposed voice actors to a range of risks. While existing ethical frameworks emphasize Consent, Credit, and Compensation (C3), they do not adequately address the emergent risks involving vocal identities that are increasingly decoupled from context, authorship, and control. Drawing on qualitative interviews with 20 professional voice actors, this paper reveals how the synthetic replication of voice without enforceable constraints exposes individuals to a range of threats. Beyond reputational harm, such as re-purposing voice data in erotic content, offensive political messaging, and meme culture, we document concerns about accountability breakdowns when their voice is leveraged to clone voices that are deployed in high-stakes scenarios such as financial fraud, misinformation campaigns, or impersonation scams. In such cases, actors face social and legal fallout without recourse, while very few of them have a legal representative or union protection. To make sense of these shifting dynamics, we introduce the PRAC3 framework, an expansion of C3 that foregrounds Privacy, Reputation, Accountability, Consent, Credit, and Compensation as interdependent pillars of data used in the synthetic voice economy. This framework captures how privacy risks are amplified through non-consensual training, how reputational harm arises from decontextualized deployment, and how accountability can be reimagined AI Data ecosystems. We argue that voice, as both a biometric identifier and creative labor, demands governance models that restore creator agency, ensure traceability, and establish enforceable boundaries for ethical reuse. 

---
# eX-NIDS: A Framework for Explainable Network Intrusion Detection Leveraging Large Language Models 

**Authors**: Paul R. B. Houssel, Siamak Layeghy, Priyanka Singh, Marius Portmann  

**Link**: [PDF](https://arxiv.org/pdf/2507.16241)  

**Abstract**: This paper introduces eX-NIDS, a framework designed to enhance interpretability in flow-based Network Intrusion Detection Systems (NIDS) by leveraging Large Language Models (LLMs). In our proposed framework, flows labelled as malicious by NIDS are initially processed through a module called the Prompt Augmenter. This module extracts contextual information and Cyber Threat Intelligence (CTI)-related knowledge from these flows. This enriched, context-specific data is then integrated with an input prompt for an LLM, enabling it to generate detailed explanations and interpretations of why the flow was identified as malicious by NIDS. We compare the generated interpretations against a Basic-Prompt Explainer baseline, which does not incorporate any contextual information into the LLM's input prompt. Our framework is quantitatively evaluated using the Llama 3 and GPT-4 models, employing a novel evaluation method tailored for natural language explanations, focusing on their correctness and consistency. The results demonstrate that augmented LLMs can produce accurate and consistent explanations, serving as valuable complementary tools in NIDS to explain the classification of malicious flows. The use of augmented prompts enhances performance by over 20% compared to the Basic-Prompt Explainer. 

---
# Predictive Hydrodynamic Simulations for Laser Direct-drive Implosion Experiments via Artificial Intelligence 

**Authors**: Zixu Wang, Yuhan Wang, Junfei Ma, Fuyuan Wu, Junchi Yan, Xiaohui Yuan, Zhe Zhang, Jie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.16227)  

**Abstract**: This work presents predictive hydrodynamic simulations empowered by artificial intelligence (AI) for laser driven implosion experiments, taking the double-cone ignition (DCI) scheme as an example. A Transformer-based deep learning model MULTI-Net is established to predict implosion features according to laser waveforms and target radius. A Physics-Informed Decoder (PID) is proposed for high-dimensional sampling, significantly reducing the prediction errors compared to Latin hypercube sampling. Applied to DCI experiments conducted on the SG-II Upgrade facility, the MULTI-Net model is able to predict the implosion dynamics measured by the x-ray streak camera. It is found that an effective laser absorption factor about 65\% is suitable for the one-dimensional simulations of the DCI-R10 experiments. For shot 33, the mean implosion velocity and collided plasma density reached 195 km/s and 117 g/cc, respectively. This study demonstrates a data-driven AI framework that enhances the prediction ability of simulations for complicated laser fusion experiments. 

---
# Bayesian Deep Learning for Convective Initiation Nowcasting Uncertainty Estimation 

**Authors**: Da Fan, David John Gagne II, Steven J. Greybush, Eugene E. Clothiaux, John S. Schreck, Chaopeng Shen  

**Link**: [PDF](https://arxiv.org/pdf/2507.16219)  

**Abstract**: This study evaluated the probability and uncertainty forecasts of five recently proposed Bayesian deep learning methods relative to a deterministic residual neural network (ResNet) baseline for 0-1 h convective initiation (CI) nowcasting using GOES-16 satellite infrared observations. Uncertainty was assessed by how well probabilistic forecasts were calibrated and how well uncertainty separated forecasts with large and small errors. Most of the Bayesian deep learning methods produced probabilistic forecasts that outperformed the deterministic ResNet, with one, the initial-weights ensemble + Monte Carlo (MC) dropout, an ensemble of deterministic ResNets with different initial weights to start training and dropout activated during inference, producing the most skillful and well-calibrated forecasts. The initial-weights ensemble + MC dropout benefited from generating multiple solutions that more thoroughly sampled the hypothesis space. The Bayesian ResNet ensemble was the only one that performed worse than the deterministic ResNet at longer lead times, likely due to the challenge of optimizing a larger number of parameters. To address this issue, the Bayesian-MOPED (MOdel Priors with Empirical Bayes using Deep neural network) ResNet ensemble was adopted, and it enhanced forecast skill by constraining the hypothesis search near the deterministic ResNet hypothesis. All Bayesian methods demonstrated well-calibrated uncertainty and effectively separated cases with large and small errors. In case studies, the initial-weights ensemble + MC dropout demonstrated better forecast skill than the Bayesian-MOPED ensemble and the deterministic ResNet on selected CI events in clear-sky regions. However, the initial-weights ensemble + MC dropout exhibited poorer generalization in clear-sky and anvil cloud regions without CI occurrence compared to the deterministic ResNet and Bayesian-MOPED ensemble. 

---
# Towards Compute-Optimal Many-Shot In-Context Learning 

**Authors**: Shahriar Golchin, Yanfei Chen, Rujun Han, Manan Gandhi, Tianli Yu, Swaroop Mishra, Mihai Surdeanu, Rishabh Agarwal, Chen-Yu Lee, Tomas Pfister  

**Link**: [PDF](https://arxiv.org/pdf/2507.16217)  

**Abstract**: Long-context large language models (LLMs) are able to process inputs containing up to several million tokens. In the scope of in-context learning (ICL), this translates into using hundreds/thousands of demonstrations in the input prompt, enabling many-shot ICL. In practice, a fixed set of demonstrations is often selected at random in many-shot settings due to (1) high inference costs, (2) the benefits of caching and reusing computations, and (3) the similar performance offered by this strategy compared to others when scaled. In this work, we propose two straightforward strategies for demonstration selection in many-shot ICL that improve performance with minimal computational overhead. Our first method combines a small number of demonstrations, selected based on their similarity to each test sample, with a disproportionately larger set of random demonstrations that are cached. The second strategy improves the first by replacing random demonstrations with those selected using centroids derived from test sample representations via k-means clustering. Our experiments with Gemini Pro and Flash across several datasets indicate that our strategies consistently outperform random selection and surpass or match the most performant selection approach while supporting caching and reducing inference cost by up to an order of magnitude. We also show that adjusting the proportion of demonstrations selected based on different criteria can balance performance and inference cost in many-shot ICL. 

---
# Adaptive Relative Pose Estimation Framework with Dual Noise Tuning for Safe Approaching Maneuvers 

**Authors**: Batu Candan, Simone Servadio  

**Link**: [PDF](https://arxiv.org/pdf/2507.16214)  

**Abstract**: Accurate and robust relative pose estimation is crucial for enabling challenging Active Debris Removal (ADR) missions targeting tumbling derelict satellites such as ESA's ENVISAT. This work presents a complete pipeline integrating advanced computer vision techniques with adaptive nonlinear filtering to address this challenge. A Convolutional Neural Network (CNN), enhanced with image preprocessing, detects structural markers (corners) from chaser imagery, whose 2D coordinates are converted to 3D measurements using camera modeling. These measurements are fused within an Unscented Kalman Filter (UKF) framework, selected for its ability to handle nonlinear relative dynamics, to estimate the full relative pose. Key contributions include the integrated system architecture and a dual adaptive strategy within the UKF: dynamic tuning of the measurement noise covariance compensates for varying CNN measurement uncertainty, while adaptive tuning of the process noise covariance, utilizing measurement residual analysis, accounts for unmodeled dynamics or maneuvers online. This dual adaptation enhances robustness against both measurement imperfections and dynamic model uncertainties. The performance of the proposed adaptive integrated system is evaluated through high-fidelity simulations using a realistic ENVISAT model, comparing estimates against ground truth under various conditions, including measurement outages. This comprehensive approach offers an enhanced solution for robust onboard relative navigation, significantly advancing the capabilities required for safe proximity operations during ADR missions. 

---
# Advancing Visual Large Language Model for Multi-granular Versatile Perception 

**Authors**: Wentao Xiang, Haoxian Tan, Cong Wei, Yujie Zhong, Dengjie Li, Yujiu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.16213)  

**Abstract**: Perception is a fundamental task in the field of computer vision, encompassing a diverse set of subtasks that can be systematically categorized into four distinct groups based on two dimensions: prediction type and instruction type. Notably, existing researches often focus solely on a limited subset of these potential combinations, which constrains their applicability and versatility across various contexts. In response to this challenge, we present MVP-LM, a Multi-granular and Versatile Perception framework incorporating Visual Large Language Model. Our framework is designed to integrate both word-based and sentence-based perception tasks alongside box and mask predictions within a single architecture. MVP-LM features an innovative multi-granularity decoder in conjunction with a CoT-inspired dataset unification strategy, enabling seamless supervised fine-tuning across a wide spectrum of tasks, including but not limited to panoptic segmentation, detection, grounding, and referring expression segmentation. Furthermore, we introduce a query enhancement strategy aimed at harnessing the decoding and generative capabilities inherent in VLLMs. Extensive experiments conducted across a range of benchmarks in both word-based and sentence-based perception tasks substantiate the efficacy of our framework. The code will be available at this https URL. 

---
# LOCOFY Large Design Models -- Design to code conversion solution 

**Authors**: Sohaib Muhammad, Ashwati Vipin, Karan Shetti, Honey Mittal  

**Link**: [PDF](https://arxiv.org/pdf/2507.16208)  

**Abstract**: Despite rapid advances in Large Language Models and Multimodal Large Language Models (LLMs), numerous challenges related to interpretability, scalability, resource requirements and repeatability remain, related to their application in the design-to-code space. To address this, we introduce the Large Design Models (LDMs) paradigm specifically trained on designs and webpages to enable seamless conversion from design-to-code. We have developed a training and inference pipeline by incorporating data engineering and appropriate model architecture modification. The training pipeline consists of the following: 1)Design Optimiser: developed using a proprietary ground truth dataset and addresses sub-optimal designs; 2)Tagging and feature detection: using pre-trained and fine-tuned models, this enables the accurate detection and classification of UI elements; and 3)Auto Components: extracts repeated UI structures into reusable components to enable creation of modular code, thus reducing redundancy while enhancing code reusability. In this manner, each model addresses distinct but key issues for design-to-code conversion. Separately, our inference pipeline processes real-world designs to produce precise and interpretable instructions for code generation and ensures reliability. Additionally, our models illustrated exceptional end-to-end design-to-code conversion accuracy using a novel preview match score metric. Comparative experiments indicated superior performance of LDMs against LLMs on accuracy of node positioning, responsiveness and reproducibility. Moreover, our custom-trained tagging and feature detection model demonstrated high precision and consistency in identifying UI elements across a wide sample of test designs. Thus, our proposed LDMs are a reliable and superior solution to understanding designs that subsequently enable the generation of efficient and reliable production-ready code. 

---
# A Human-Centered Approach to Identifying Promises, Risks, & Challenges of Text-to-Image Generative AI in Radiology 

**Authors**: Katelyn Morrison, Arpit Mathur, Aidan Bradshaw, Tom Wartmann, Steven Lundi, Afrooz Zandifar, Weichang Dai, Kayhan Batmanghelich, Motahhare Eslami, Adam Perer  

**Link**: [PDF](https://arxiv.org/pdf/2507.16207)  

**Abstract**: As text-to-image generative models rapidly improve, AI researchers are making significant advances in developing domain-specific models capable of generating complex medical imagery from text prompts. Despite this, these technical advancements have overlooked whether and how medical professionals would benefit from and use text-to-image generative AI (GenAI) in practice. By developing domain-specific GenAI without involving stakeholders, we risk the potential of building models that are either not useful or even more harmful than helpful. In this paper, we adopt a human-centered approach to responsible model development by involving stakeholders in evaluating and reflecting on the promises, risks, and challenges of a novel text-to-CT Scan GenAI model. Through exploratory model prompting activities, we uncover the perspectives of medical students, radiology trainees, and radiologists on the role that text-to-CT Scan GenAI can play across medical education, training, and practice. This human-centered approach additionally enabled us to surface technical challenges and domain-specific risks of generating synthetic medical images. We conclude by reflecting on the implications of medical text-to-image GenAI. 

---
# METER: Multi-modal Evidence-based Thinking and Explainable Reasoning -- Algorithm and Benchmark 

**Authors**: Xu Yang, Qi Zhang, Shuming Jiang, Yaowen Xu, Zhaofan Zou, Hao Sun, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.16206)  

**Abstract**: With the rapid advancement of generative AI, synthetic content across images, videos, and audio has become increasingly realistic, amplifying the risk of misinformation. Existing detection approaches predominantly focus on binary classification while lacking detailed and interpretable explanations of forgeries, which limits their applicability in safety-critical scenarios. Moreover, current methods often treat each modality separately, without a unified benchmark for cross-modal forgery detection and interpretation. To address these challenges, we introduce METER, a unified, multi-modal benchmark for interpretable forgery detection spanning images, videos, audio, and audio-visual content. Our dataset comprises four tracks, each requiring not only real-vs-fake classification but also evidence-chain-based explanations, including spatio-temporal localization, textual rationales, and forgery type tracing. Compared to prior benchmarks, METER offers broader modality coverage and richer interpretability metrics such as spatial/temporal IoU, multi-class tracing, and evidence consistency. We further propose a human-aligned, three-stage Chain-of-Thought (CoT) training strategy combining SFT, DPO, and a novel GRPO stage that integrates a human-aligned evaluator with CoT reasoning. We hope METER will serve as a standardized foundation for advancing generalizable and interpretable forgery detection in the era of generative media. 

---
# SVAgent: AI Agent for Hardware Security Verification Assertion 

**Authors**: Rui Guo, Avinash Ayalasomayajula, Henian Li, Jingbo Zhou, Sujan Kumar Saha, Farimah Farahmandi  

**Link**: [PDF](https://arxiv.org/pdf/2507.16203)  

**Abstract**: Verification using SystemVerilog assertions (SVA) is one of the most popular methods for detecting circuit design vulnerabilities. However, with the globalization of integrated circuit design and the continuous upgrading of security requirements, the SVA development model has exposed major limitations. It is not only inefficient in development, but also unable to effectively deal with the increasing number of security vulnerabilities in modern complex integrated circuits. In response to these challenges, this paper proposes an innovative SVA automatic generation framework SVAgent. SVAgent introduces a requirement decomposition mechanism to transform the original complex requirements into a structured, gradually solvable fine-grained problem-solving chain. Experiments have shown that SVAgent can effectively suppress the influence of hallucinations and random answers, and the key evaluation indicators such as the accuracy and consistency of the SVA are significantly better than existing frameworks. More importantly, we successfully integrated SVAgent into the most mainstream integrated circuit vulnerability assessment framework and verified its practicality and reliability in a real engineering design environment. 

---
# LLM Data Selection and Utilization via Dynamic Bi-level Optimization 

**Authors**: Yang Yu, Kai Han, Hang Zhou, Yehui Tang, Kaiqi Huang, Yunhe Wang, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2507.16178)  

**Abstract**: While large-scale training data is fundamental for developing capable large language models (LLMs), strategically selecting high-quality data has emerged as a critical approach to enhance training efficiency and reduce computational costs. Current data selection methodologies predominantly rely on static, training-agnostic criteria, failing to account for the dynamic model training and data interactions. In this paper, we propose a new Data Weighting Model (DWM) to adjust the weight of selected data within each batch to achieve a dynamic data utilization during LLM training. Specially, to better capture the dynamic data preference of the trained model, a bi-level optimization framework is implemented to update the weighting model. Our experiments demonstrate that DWM enhances the performance of models trained with randomly-selected data, and the learned weighting model can be transferred to enhance other data selection methods and models of different sizes. Moreover, we further analyze how a model's data preferences evolve throughout training, providing new insights into the data preference of the model during training. 

---
# Attacking interpretable NLP systems 

**Authors**: Eldor Abdukhamidov, Tamer Abuhmed, Joanna C. S. Santos, Mohammed Abuhamad  

**Link**: [PDF](https://arxiv.org/pdf/2507.16164)  

**Abstract**: Studies have shown that machine learning systems are vulnerable to adversarial examples in theory and practice. Where previous attacks have focused mainly on visual models that exploit the difference between human and machine perception, text-based models have also fallen victim to these attacks. However, these attacks often fail to maintain the semantic meaning of the text and similarity. This paper introduces AdvChar, a black-box attack on Interpretable Natural Language Processing Systems, designed to mislead the classifier while keeping the interpretation similar to benign inputs, thus exploiting trust in system transparency. AdvChar achieves this by making less noticeable modifications to text input, forcing the deep learning classifier to make incorrect predictions and preserve the original interpretation. We use an interpretation-focused scoring approach to determine the most critical tokens that, when changed, can cause the classifier to misclassify the input. We apply simple character-level modifications to measure the importance of tokens, minimizing the difference between the original and new text while generating adversarial interpretations similar to benign ones. We thoroughly evaluated AdvChar by testing it against seven NLP models and three interpretation models using benchmark datasets for the classification task. Our experiments show that AdvChar can significantly reduce the prediction accuracy of current deep learning models by altering just two characters on average in input samples. 

---
# LSSGen: Leveraging Latent Space Scaling in Flow and Diffusion for Efficient Text to Image Generation 

**Authors**: Jyun-Ze Tang, Chih-Fan Hsu, Jeng-Lin Li, Ming-Ching Chang, Wei-Chao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.16154)  

**Abstract**: Flow matching and diffusion models have shown impressive results in text-to-image generation, producing photorealistic images through an iterative denoising process. A common strategy to speed up synthesis is to perform early denoising at lower resolutions. However, traditional methods that downscale and upscale in pixel space often introduce artifacts and distortions. These issues arise when the upscaled images are re-encoded into the latent space, leading to degraded final image quality. To address this, we propose {\bf Latent Space Scaling Generation (LSSGen)}, a framework that performs resolution scaling directly in the latent space using a lightweight latent upsampler. Without altering the Transformer or U-Net architecture, LSSGen improves both efficiency and visual quality while supporting flexible multi-resolution generation. Our comprehensive evaluation covering text-image alignment and perceptual quality shows that LSSGen significantly outperforms conventional scaling approaches. When generating $1024^2$ images at similar speeds, it achieves up to 246\% TOPIQ score improvement. 

---
# SPACT18: Spiking Human Action Recognition Benchmark Dataset with Complementary RGB and Thermal Modalities 

**Authors**: Yasser Ashraf, Ahmed Sharshar, Velibor Bojkovic, Bin Gu  

**Link**: [PDF](https://arxiv.org/pdf/2507.16151)  

**Abstract**: Spike cameras, bio-inspired vision sensors, asynchronously fire spikes by accumulating light intensities at each pixel, offering ultra-high energy efficiency and exceptional temporal resolution. Unlike event cameras, which record changes in light intensity to capture motion, spike cameras provide even finer spatiotemporal resolution and a more precise representation of continuous changes. In this paper, we introduce the first video action recognition (VAR) dataset using spike camera, alongside synchronized RGB and thermal modalities, to enable comprehensive benchmarking for Spiking Neural Networks (SNNs). By preserving the inherent sparsity and temporal precision of spiking data, our three datasets offer a unique platform for exploring multimodal video understanding and serve as a valuable resource for directly comparing spiking, thermal, and RGB modalities. This work contributes a novel dataset that will drive research in energy-efficient, ultra-low-power video understanding, specifically for action recognition tasks using spike-based data. 

---
# SDBench: A Comprehensive Benchmark Suite for Speaker Diarization 

**Authors**: Eduardo Pacheco, Atila Orhon, Berkin Durmus, Blaise Munyampirwa, Andrey Leonov  

**Link**: [PDF](https://arxiv.org/pdf/2507.16136)  

**Abstract**: Even state-of-the-art speaker diarization systems exhibit high variance in error rates across different datasets, representing numerous use cases and domains. Furthermore, comparing across systems requires careful application of best practices such as dataset splits and metric definitions to allow for apples-to-apples comparison. We propose SDBench (Speaker Diarization Benchmark), an open-source benchmark suite that integrates 13 diverse datasets with built-in tooling for consistent and fine-grained analysis of speaker diarization performance for various on-device and server-side systems. SDBench enables reproducible evaluation and easy integration of new systems over time. To demonstrate the efficacy of SDBench, we built SpeakerKit, an inference efficiency-focused system built on top of Pyannote v3. SDBench enabled rapid execution of ablation studies that led to SpeakerKit being 9.6x faster than Pyannote v3 while achieving comparable error rates. We benchmark 6 state-of-the-art systems including Deepgram, AWS Transcribe, and Pyannote AI API, revealing important trade-offs between accuracy and speed. 

---
# Disability Across Cultures: A Human-Centered Audit of Ableism in Western and Indic LLMs 

**Authors**: Mahika Phutane, Aditya Vashistha  

**Link**: [PDF](https://arxiv.org/pdf/2507.16130)  

**Abstract**: People with disabilities (PwD) experience disproportionately high levels of discrimination and hate online, particularly in India, where entrenched stigma and limited resources intensify these challenges. Large language models (LLMs) are increasingly used to identify and mitigate online hate, yet most research on online ableism focuses on Western audiences with Western AI models. Are these models adequately equipped to recognize ableist harm in non-Western places like India? Do localized, Indic language models perform better? To investigate, we adopted and translated a publicly available ableist speech dataset to Hindi, and prompted eight LLMs--four developed in the U.S. (GPT-4, Gemini, Claude, Llama) and four in India (Krutrim, Nanda, Gajendra, Airavata)--to score and explain ableism. In parallel, we recruited 175 PwD from both the U.S. and India to perform the same task, revealing stark differences between groups. Western LLMs consistently overestimated ableist harm, while Indic LLMs underestimated it. Even more concerning, all LLMs were more tolerant of ableism when it was expressed in Hindi and asserted Western framings of ableist harm. In contrast, Indian PwD interpreted harm through intention, relationality, and resilience--emphasizing a desire to inform and educate perpetrators. This work provides groundwork for global, inclusive standards of ableism, demonstrating the need to center local disability experiences in the design and evaluation of AI systems. 

---
# Benchmarking LLM Privacy Recognition for Social Robot Decision Making 

**Authors**: Dakota Sullivan, Shirley Zhang, Jennica Li, Heather Kirkorian, Bilge Mutlu, Kassem Fawaz  

**Link**: [PDF](https://arxiv.org/pdf/2507.16124)  

**Abstract**: Social robots are embodied agents that interact with people while following human communication norms. These robots interact using verbal and non-verbal cues, and share the physical environments of people. While social robots have previously utilized rule-based systems or probabilistic models for user interaction, the rapid evolution of large language models (LLMs) presents new opportunities to develop LLM-empowered social robots for enhanced human-robot interaction. To fully realize these capabilities, however, robots need to collect data such as audio, fine-grained images, video, and locations. As a result, LLMs often process sensitive personal information, particularly within home environments. Given the tension between utility and privacy risks, evaluating how current LLMs manage sensitive data is critical. Specifically, we aim to explore the extent to which out-of-the-box LLMs are privacy-aware in the context of household social robots. In this study, we present a set of privacy-relevant scenarios crafted through the lens of Contextual Integrity (CI). We first survey users' privacy preferences regarding in-home social robot behaviors and then examine how their privacy orientation affects their choices of these behaviors (N = 450). We then provide the same set of scenarios and questions to state-of-the-art LLMs (N = 10) and find that the agreement between humans and LLMs is low. To further investigate the capabilities of LLMs as a potential privacy controller, we implement four additional prompting strategies and compare their results. Finally, we discuss the implications and potential of AI privacy awareness in human-robot interaction. 

---
# Efficient Compositional Multi-tasking for On-device Large Language Models 

**Authors**: Ondrej Bohdal, Mete Ozay, Jijoong Moon, Kyeng-Hun Lee, Hyeonmok Ko, Umberto Michieli  

**Link**: [PDF](https://arxiv.org/pdf/2507.16083)  

**Abstract**: Adapter parameters provide a mechanism to modify the behavior of machine learning models and have gained significant popularity in the context of large language models (LLMs) and generative AI. These parameters can be merged to support multiple tasks via a process known as task merging. However, prior work on merging in LLMs, particularly in natural language processing, has been limited to scenarios where each test example addresses only a single task. In this paper, we focus on on-device settings and study the problem of text-based compositional multi-tasking, where each test example involves the simultaneous execution of multiple tasks. For instance, generating a translated summary of a long text requires solving both translation and summarization tasks concurrently. To facilitate research in this setting, we propose a benchmark comprising four practically relevant compositional tasks. We also present an efficient method (Learnable Calibration) tailored for on-device applications, where computational resources are limited, emphasizing the need for solutions that are both resource-efficient and high-performing. Our contributions lay the groundwork for advancing the capabilities of LLMs in real-world multi-tasking scenarios, expanding their applicability to complex, resource-constrained use cases. 

---
# A Lower Bound for the Number of Linear Regions of Ternary ReLU Regression Neural Networks 

**Authors**: Yuta Nakahara, Manabu Kobayashi, Toshiyasu Matsushima  

**Link**: [PDF](https://arxiv.org/pdf/2507.16079)  

**Abstract**: With the advancement of deep learning, reducing computational complexity and memory consumption has become a critical challenge, and ternary neural networks (NNs) that restrict parameters to $\{-1, 0, +1\}$ have attracted attention as a promising approach. While ternary NNs demonstrate excellent performance in practical applications such as image recognition and natural language processing, their theoretical understanding remains insufficient. In this paper, we theoretically analyze the expressivity of ternary NNs from the perspective of the number of linear regions. Specifically, we evaluate the number of linear regions of ternary regression NNs with Rectified Linear Unit (ReLU) for activation functions and prove that the number of linear regions increases polynomially with respect to network width and exponentially with respect to depth, similar to standard NNs. Moreover, we show that it suffices to either square the width or double the depth of ternary NNs to achieve a lower bound on the maximum number of linear regions comparable to that of general ReLU regression NNs. This provides a theoretical explanation, in some sense, for the practical success of ternary NNs. 

---
# AI-driven Orchestration at Scale: Estimating Service Metrics on National-Wide Testbeds 

**Authors**: Rodrigo Moreira, Rafael Pasquini, Joberto S. B. Martins, Tereza C. Carvalho, Flávio de Oliveira Silva  

**Link**: [PDF](https://arxiv.org/pdf/2507.16077)  

**Abstract**: Network Slicing (NS) realization requires AI-native orchestration architectures to efficiently and intelligently handle heterogeneous user requirements. To achieve this, network slicing is evolving towards a more user-centric digital transformation, focusing on architectures that incorporate native intelligence to enable self-managed connectivity in an integrated and isolated manner. However, these initiatives face the challenge of validating their results in production environments, particularly those utilizing ML-enabled orchestration, as they are often tested in local networks or laboratory simulations. This paper proposes a large-scale validation method using a network slicing prediction model to forecast latency using Deep Neural Networks (DNNs) and basic ML algorithms embedded within an NS architecture, evaluated in real large-scale production testbeds. It measures and compares the performance of different DNNs and ML algorithms, considering a distributed database application deployed as a network slice over two large-scale production testbeds. The investigation highlights how AI-based prediction models can enhance network slicing orchestration architectures and presents a seamless, production-ready validation method as an alternative to fully controlled simulations or laboratory setups. 

---
# Compositional Coordination for Multi-Robot Teams with Large Language Models 

**Authors**: Zhehui Huang, Guangyao Shi, Yuwei Wu, Vijay Kumar, Gaurav S. Sukhatme  

**Link**: [PDF](https://arxiv.org/pdf/2507.16068)  

**Abstract**: Multi-robot coordination has traditionally relied on a task-specific and expert-driven pipeline, where natural language mission descriptions are manually translated by domain experts into mathematical formulation, algorithm design, and executable code. This conventional process is labor-intensive, inaccessible to non-experts, and inflexible to changes in mission requirements. Here, we propose LAN2CB (Language to Collective Behavior), a novel framework that leverages large language models (LLMs) to streamline and generalize the multi-robot coordination pipeline. LAN2CB directly converts natural language mission descriptions into executable Python code for multi-robot systems through two key components: (1) Mission Decomposition for Task Representation, which parses the mission into a task graph with dependencies, and (2) Code Generation, which uses the task graph and a structured knowledge base to generate deployable robot control code. We further introduce a dataset of natural language mission specifications to support development and benchmarking. Experimental results in both simulation and real-world settings show that LAN2CB enables effective and flexible multi-robot coordination from natural language, significantly reducing the need for manual engineering while supporting generalization across mission types. Website: this https URL. 

---
# AI-Powered Commit Explorer (APCE) 

**Authors**: Yousab Grees, Polina Iaremchuk, Ramtin Ehsani, Esteban Parra, Preetha Chatterjee, Sonia Haiduc  

**Link**: [PDF](https://arxiv.org/pdf/2507.16063)  

**Abstract**: Commit messages in a version control system provide valuable information for developers regarding code changes in software systems. Commit messages can be the only source of information left for future developers describing what was changed and why. However, writing high-quality commit messages is often neglected in practice. Large Language Model (LLM) generated commit messages have emerged as a way to mitigate this issue. We introduce the AI-Powered Commit Explorer (APCE), a tool to support developers and researchers in the use and study of LLM-generated commit messages. APCE gives researchers the option to store different prompts for LLMs and provides an additional evaluation prompt that can further enhance the commit message provided by LLMs. APCE also provides researchers with a straightforward mechanism for automated and human evaluation of LLM-generated messages. Demo link this https URL 

---
# Beyond Rate Coding: Surrogate Gradients Enable Spike Timing Learning in Spiking Neural Networks 

**Authors**: Ziqiao Yu, Pengfei Sun, Dan F. M. Goodman  

**Link**: [PDF](https://arxiv.org/pdf/2507.16043)  

**Abstract**: We investigate the extent to which Spiking Neural Networks (SNNs) trained with Surrogate Gradient Descent (Surrogate GD), with and without delay learning, can learn from precise spike timing beyond firing rates. We first design synthetic tasks isolating intra-neuron inter-spike intervals and cross-neuron synchrony under matched spike counts. On more complex spike-based speech recognition datasets (Spiking Heidelberg Digits (SHD) and Spiking Speech Commands (SSC), we construct variants where spike count information is eliminated and only timing information remains, and show that Surrogate GD-trained SNNs are able to perform significantly above chance whereas purely rate-based models perform at chance level. We further evaluate robustness under biologically inspired perturbations -- including Gaussian jitter per spike or per-neuron, and spike deletion -- revealing consistent but perturbation-specific degradation. Networks show a sharp performance drop when spike sequences are reversed in time, with a larger drop in performance from SNNs trained with delays, indicating that these networks are more human-like in terms of behaviour. To facilitate further studies of temporal coding, we have released our modified SHD and SSC datasets. 

---
# Reactivation: Empirical NTK Dynamics Under Task Shifts 

**Authors**: Yuzhi Liu, Zixuan Chen, Zirui Zhang, Yufei Liu, Giulia Lanzillotta  

**Link**: [PDF](https://arxiv.org/pdf/2507.16039)  

**Abstract**: The Neural Tangent Kernel (NTK) offers a powerful tool to study the functional dynamics of neural networks. In the so-called lazy, or kernel regime, the NTK remains static during training and the network function is linear in the static neural tangents feature space. The evolution of the NTK during training is necessary for feature learning, a key driver of deep learning success. The study of the NTK dynamics has led to several critical discoveries in recent years, in generalization and scaling behaviours. However, this body of work has been limited to the single task setting, where the data distribution is assumed constant over time. In this work, we present a comprehensive empirical analysis of NTK dynamics in continual learning, where the data distribution shifts over time. Our findings highlight continual learning as a rich and underutilized testbed for probing the dynamics of neural training. At the same time, they challenge the validity of static-kernel approximations in theoretical treatments of continual learning, even at large scale. 

---
# Discovering and using Spelke segments 

**Authors**: Rahul Venkatesh, Klemen Kotar, Lilian Naing Chen, Seungwoo Kim, Luca Thomas Wheeler, Jared Watrous, Ashley Xu, Gia Ancone, Wanhee Lee, Honglin Chen, Daniel Bear, Stefan Stojanov, Daniel Yamins  

**Link**: [PDF](https://arxiv.org/pdf/2507.16038)  

**Abstract**: Segments in computer vision are often defined by semantic considerations and are highly dependent on category-specific conventions. In contrast, developmental psychology suggests that humans perceive the world in terms of Spelke objects--groupings of physical things that reliably move together when acted on by physical forces. Spelke objects thus operate on category-agnostic causal motion relationships which potentially better support tasks like manipulation and planning. In this paper, we first benchmark the Spelke object concept, introducing the SpelkeBench dataset that contains a wide variety of well-defined Spelke segments in natural images. Next, to extract Spelke segments from images algorithmically, we build SpelkeNet, a class of visual world models trained to predict distributions over future motions. SpelkeNet supports estimation of two key concepts for Spelke object discovery: (1) the motion affordance map, identifying regions likely to move under a poke, and (2) the expected-displacement map, capturing how the rest of the scene will move. These concepts are used for "statistical counterfactual probing", where diverse "virtual pokes" are applied on regions of high motion-affordance, and the resultant expected displacement maps are used define Spelke segments as statistical aggregates of correlated motion statistics. We find that SpelkeNet outperforms supervised baselines like SegmentAnything (SAM) on SpelkeBench. Finally, we show that the Spelke concept is practically useful for downstream applications, yielding superior performance on the 3DEditBench benchmark for physical object manipulation when used in a variety of off-the-shelf object manipulation models. 

---
# "Just a strange pic": Evaluating 'safety' in GenAI Image safety annotation tasks from diverse annotators' perspectives 

**Authors**: Ding Wang, Mark Díaz, Charvi Rastogi, Aida Davani, Vinodkumar Prabhakaran, Pushkar Mishra, Roma Patel, Alicia Parrish, Zoe Ashwood, Michela Paganini, Tian Huey Teh, Verena Rieser, Lora Aroyo  

**Link**: [PDF](https://arxiv.org/pdf/2507.16033)  

**Abstract**: Understanding what constitutes safety in AI-generated content is complex. While developers often rely on predefined taxonomies, real-world safety judgments also involve personal, social, and cultural perceptions of harm. This paper examines how annotators evaluate the safety of AI-generated images, focusing on the qualitative reasoning behind their judgments. Analyzing 5,372 open-ended comments, we find that annotators consistently invoke moral, emotional, and contextual reasoning that extends beyond structured safety categories. Many reflect on potential harm to others more than to themselves, grounding their judgments in lived experience, collective risk, and sociocultural awareness. Beyond individual perceptions, we also find that the structure of the task itself -- including annotation guidelines -- shapes how annotators interpret and express harm. Guidelines influence not only which images are flagged, but also the moral judgment behind the justifications. Annotators frequently cite factors such as image quality, visual distortion, and mismatches between prompt and output as contributing to perceived harm dimensions, which are often overlooked in standard evaluation frameworks. Our findings reveal that existing safety pipelines miss critical forms of reasoning that annotators bring to the task. We argue for evaluation designs that scaffold moral reflection, differentiate types of harm, and make space for subjective, context-sensitive interpretations of AI-generated content. 

---
# AutoMAT: A Hierarchical Framework for Autonomous Alloy Discovery 

**Authors**: Penghui Yang, Chendong Zhao, Bijun Tang, Zhonghan Zhang, Xinrun Wang, Yanchen Deng, Yuhao Lu, Cuntai Guan, Zheng Liu, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2507.16005)  

**Abstract**: Alloy discovery is central to advancing modern industry but remains hindered by the vastness of compositional design space and the costly validation. Here, we present AutoMAT, a hierarchical and autonomous framework grounded in and validated by experiments, which integrates large language models, automated CALPHAD-based simulations, and AI-driven search to accelerate alloy design. Spanning the entire pipeline from ideation to validation, AutoMAT achieves high efficiency, accuracy, and interpretability without the need for manually curated large datasets. In a case study targeting a lightweight, high-strength alloy, AutoMAT identifies a titanium alloy with 8.1% lower density and comparable yield strength relative to the state-of-the-art reference, achieving the highest specific strength among all comparisons. In a second case targeting high-yield-strength high-entropy alloys, AutoMAT achieves a 28.2% improvement in yield strength over the base alloy. In both cases, AutoMAT reduces the discovery timeline from years to weeks, illustrating its potential as a scalable and versatile platform for next-generation alloy design. 

---
# Enhancing Hindi NER in Low Context: A Comparative study of Transformer-based models with vs. without Retrieval Augmentation 

**Authors**: Sumit Singh, Rohit Mishra, Uma Shanker Tiwary  

**Link**: [PDF](https://arxiv.org/pdf/2507.16002)  

**Abstract**: One major challenge in natural language processing is named entity recognition (NER), which identifies and categorises named entities in textual input. In order to improve NER, this study investigates a Hindi NER technique that makes use of Hindi-specific pretrained encoders (MuRIL and XLM-R) and Generative Models ( Llama-2-7B-chat-hf (Llama2-7B), Llama-2-70B-chat-hf (Llama2-70B), Llama-3-70B-Instruct (Llama3-70B) and GPT3.5-turbo), and augments the data with retrieved data from external relevant contexts, notably from Wikipedia. We have fine-tuned MuRIL, XLM-R and Llama2-7B with and without RA. However, Llama2-70B, lama3-70B and GPT3.5-turbo are utilised for few-shot NER generation. Our investigation shows that the mentioned language models (LMs) with Retrieval Augmentation (RA) outperform baseline methods that don't incorporate RA in most cases. The macro F1 scores for MuRIL and XLM-R are 0.69 and 0.495, respectively, without RA and increase to 0.70 and 0.71, respectively, in the presence of RA. Fine-tuned Llama2-7B outperforms Llama2-7B by a significant margin. On the other hand the generative models which are not fine-tuned also perform better with augmented data. GPT3.5-turbo adopted RA well; however, Llama2-70B and llama3-70B did not adopt RA with our retrieval context. The findings show that RA significantly improves performance, especially for low-context data. This study adds significant knowledge about how best to use data augmentation methods and pretrained models to enhance NER performance, particularly in languages with limited resources. 

---
# Dream, Lift, Animate: From Single Images to Animatable Gaussian Avatars 

**Authors**: Marcel C. Bühler, Ye Yuan, Xueting Li, Yangyi Huang, Koki Nagano, Umar Iqbal  

**Link**: [PDF](https://arxiv.org/pdf/2507.15979)  

**Abstract**: We introduce Dream, Lift, Animate (DLA), a novel framework that reconstructs animatable 3D human avatars from a single image. This is achieved by leveraging multi-view generation, 3D Gaussian lifting, and pose-aware UV-space mapping of 3D Gaussians. Given an image, we first dream plausible multi-views using a video diffusion model, capturing rich geometric and appearance details. These views are then lifted into unstructured 3D Gaussians. To enable animation, we propose a transformer-based encoder that models global spatial relationships and projects these Gaussians into a structured latent representation aligned with the UV space of a parametric body model. This latent code is decoded into UV-space Gaussians that can be animated via body-driven deformation and rendered conditioned on pose and viewpoint. By anchoring Gaussians to the UV manifold, our method ensures consistency during animation while preserving fine visual details. DLA enables real-time rendering and intuitive editing without requiring post-processing. Our method outperforms state-of-the-art approaches on ActorsHQ and 4D-Dress datasets in both perceptual quality and photometric accuracy. By combining the generative strengths of video diffusion models with a pose-aware UV-space Gaussian mapping, DLA bridges the gap between unstructured 3D representations and high-fidelity, animation-ready avatars. 

---
# On the transferability of Sparse Autoencoders for interpreting compressed models 

**Authors**: Suchit Gupte, Vishnu Kabir Chhabra, Mohammad Mahdi Khalili  

**Link**: [PDF](https://arxiv.org/pdf/2507.15977)  

**Abstract**: Modern LLMs face inference efficiency challenges due to their scale. To address this, many compression methods have been proposed, such as pruning and quantization. However, the effect of compression on a model's interpretability remains elusive. While several model interpretation approaches exist, such as circuit discovery, Sparse Autoencoders (SAEs) have proven particularly effective in decomposing a model's activation space into its feature basis. In this work, we explore the differences in SAEs for the original and compressed models. We find that SAEs trained on the original model can interpret the compressed model albeit with slight performance degradation compared to the trained SAE on the compressed model. Furthermore, simply pruning the original SAE itself achieves performance comparable to training a new SAE on the pruned model. This finding enables us to mitigate the extensive training costs of SAEs. 

---
# Nonlinear Framework for Speech Bandwidth Extension 

**Authors**: Tarikul Islam Tamiti, Nursad Mamun, Anomadarshi Barua  

**Link**: [PDF](https://arxiv.org/pdf/2507.15970)  

**Abstract**: Recovering high-frequency components lost to bandwidth constraints is crucial for applications ranging from telecommunications to high-fidelity audio on limited resources. We introduce NDSI-BWE, a new adversarial Band Width Extension (BWE) framework that leverage four new discriminators inspired by nonlinear dynamical system to capture diverse temporal behaviors: a Multi-Resolution Lyapunov Discriminator (MRLD) for determining sensitivity to initial conditions by capturing deterministic chaos, a Multi-Scale Recurrence Discriminator (MS-RD) for self-similar recurrence dynamics, a Multi-Scale Detrended Fractal Analysis Discriminator (MSDFA) for long range slow variant scale invariant relationship, a Multi-Resolution Poincaré Plot Discriminator (MR-PPD) for capturing hidden latent space relationship, a Multi-Period Discriminator (MPD) for cyclical patterns, a Multi-Resolution Amplitude Discriminator (MRAD) and Multi-Resolution Phase Discriminator (MRPD) for capturing intricate amplitude-phase transition statistics. By using depth-wise convolution at the core of the convolutional block with in each discriminators, NDSI-BWE attains an eight-times parameter reduction. These seven discriminators guide a complex-valued ConformerNeXt based genetor with a dual stream Lattice-Net based architecture for simultaneous refinement of magnitude and phase. The genertor leverage the transformer based conformer's global dependency modeling and ConvNeXt block's local temporal modeling capability. Across six objective evaluation metrics and subjective based texts comprises of five human judges, NDSI-BWE establishes a new SoTA in BWE. 

---
# A Lightweight Face Quality Assessment Framework to Improve Face Verification Performance in Real-Time Screening Applications 

**Authors**: Ahmed Aman Ibrahim, Hamad Mansour Alawar, Abdulnasser Abbas Zehi, Ahmed Mohammad Alkendi, Bilal Shafi Ashfaq Ahmed Mirza, Shan Ullah, Ismail Lujain Jaleel, Hassan Ugail  

**Link**: [PDF](https://arxiv.org/pdf/2507.15961)  

**Abstract**: Face image quality plays a critical role in determining the accuracy and reliability of face verification systems, particularly in real-time screening applications such as surveillance, identity verification, and access control. Low-quality face images, often caused by factors such as motion blur, poor lighting conditions, occlusions, and extreme pose variations, significantly degrade the performance of face recognition models, leading to higher false rejection and false acceptance rates. In this work, we propose a lightweight yet effective framework for automatic face quality assessment, which aims to pre-filter low-quality face images before they are passed to the verification pipeline. Our approach utilises normalised facial landmarks in conjunction with a Random Forest Regression classifier to assess image quality, achieving an accuracy of 96.67\%. By integrating this quality assessment module into the face verification process, we observe a substantial improvement in performance, including a comfortable 99.7\% reduction in the false rejection rate and enhanced cosine similarity scores when paired with the ArcFace face verification model. To validate our approach, we have conducted experiments on a real-world dataset collected comprising over 600 subjects captured from CCTV footage in unconstrained environments within Dubai Police. Our results demonstrate that the proposed framework effectively mitigates the impact of poor-quality face images, outperforming existing face quality assessment techniques while maintaining computational efficiency. Moreover, the framework specifically addresses two critical challenges in real-time screening: variations in face resolution and pose deviations, both of which are prevalent in practical surveillance scenarios. 

---
# Quantization-Aware Neuromorphic Architecture for Efficient Skin Disease Classification on Resource-Constrained Devices 

**Authors**: Haitian Wang, Xinyu Wang, Yiren Wang, Karen Lee, Zichen Geng, Xian Zhang, Kehkashan Kiran, Yu Zhang, Bo Miao  

**Link**: [PDF](https://arxiv.org/pdf/2507.15958)  

**Abstract**: Accurate and efficient skin lesion classification on edge devices is critical for accessible dermatological care but remains challenging due to computational, energy, and privacy constraints. We introduce QANA, a novel quantization-aware neuromorphic architecture for incremental skin lesion classification on resource-limited hardware. QANA effectively integrates ghost modules, efficient channel attention, and squeeze-and-excitation blocks for robust feature representation with low-latency and energy-efficient inference. Its quantization-aware head and spike-compatible transformations enable seamless conversion to spiking neural networks (SNNs) and deployment on neuromorphic platforms. Evaluation on the large-scale HAM10000 benchmark and a real-world clinical dataset shows that QANA achieves 91.6\% Top-1 accuracy and 82.4\% macro F1 on HAM10000, and 90.8\% / 81.7\% on the clinical dataset, significantly outperforming state-of-the-art CNN-to-SNN models under fair comparison. Deployed on BrainChip Akida hardware, QANA achieves 1.5\,ms inference latency and 1.7\,mJ energy per image, reducing inference latency and energy use by over 94.6\%/98.6\% compared to GPU-based CNNs surpassing state-of-the-art CNN-to-SNN conversion baselines. These results demonstrate the effectiveness of QANA for accurate, real-time, and privacy-sensitive medical analysis in edge environments. 

---
# Dual Turing Test: A Framework for Detecting and Mitigating Undetectable AI 

**Authors**: Alberto Messina  

**Link**: [PDF](https://arxiv.org/pdf/2507.15907)  

**Abstract**: In this short note, we propose a unified framework that bridges three areas: (1) a flipped perspective on the Turing Test, the "dual Turing test", in which a human judge's goal is to identify an AI rather than reward a machine for deception; (2) a formal adversarial classification game with explicit quality constraints and worst-case guarantees; and (3) a reinforcement learning (RL) alignment pipeline that uses an undetectability detector and a set of quality related components in its reward model. We review historical precedents, from inverted and meta-Turing variants to modern supervised reverse-Turing classifiers, and highlight the novelty of combining quality thresholds, phased difficulty levels, and minimax bounds. We then formalize the dual test: define the judge's task over N independent rounds with fresh prompts drawn from a prompt space Q, introduce a quality function Q and parameters tau and delta, and cast the interaction as a two-player zero-sum game over the adversary's feasible strategy set M. Next, we map this minimax game onto an RL-HF style alignment loop, in which an undetectability detector D provides negative reward for stealthy outputs, balanced by a quality proxy that preserves fluency. Throughout, we include detailed explanations of each component notation, the meaning of inner minimization over sequences, phased tests, and iterative adversarial training and conclude with a suggestion for a couple of immediate actions. 

---
# Towards Reliable, Uncertainty-Aware Alignment 

**Authors**: Debangshu Banerjee, Kintan Saha, Aditya Gopalan  

**Link**: [PDF](https://arxiv.org/pdf/2507.15906)  

**Abstract**: Alignment of large language models (LLMs) typically involves training a reward model on preference data, followed by policy optimization with respect to the reward model. However, optimizing policies with respect to a single reward model estimate can render it vulnerable to inaccuracies in the reward model. We empirically study the variability of reward model training on open-source benchmarks. We observe that independently trained reward models on the same preference dataset can exhibit substantial disagreement, highlighting the instability of current alignment strategies. Employing a theoretical model, we demonstrate that variability in reward model estimation can cause overfitting, leading to the risk of performance degradation. To mitigate this risk, we propose a variance-aware policy optimization framework for preference-based alignment. The key ingredient of the framework is a new policy regularizer that incorporates reward model variance estimates. We show that variance-aware policy optimization provably reduces the risk of outputting a worse policy than the default. Experiments across diverse LLM and reward model configurations confirm that our approach yields more stable and robust alignment than the standard (variance-unaware) pipeline. 

---
# Foundation Models and Transformers for Anomaly Detection: A Survey 

**Authors**: Mouïn Ben Ammar, Arturo Mendoza, Nacim Belkhir, Antoine Manzanera, Gianni Franchi  

**Link**: [PDF](https://arxiv.org/pdf/2507.15905)  

**Abstract**: In line with the development of deep learning, this survey examines the transformative role of Transformers and foundation models in advancing visual anomaly detection (VAD). We explore how these architectures, with their global receptive fields and adaptability, address challenges such as long-range dependency modeling, contextual modeling and data scarcity. The survey categorizes VAD methods into reconstruction-based, feature-based and zero/few-shot approaches, highlighting the paradigm shift brought about by foundation models. By integrating attention mechanisms and leveraging large-scale pre-training, Transformers and foundation models enable more robust, interpretable, and scalable anomaly detection solutions. This work provides a comprehensive review of state-of-the-art techniques, their strengths, limitations, and emerging trends in leveraging these architectures for VAD. 

---
# Towards Mitigation of Hallucination for LLM-empowered Agents: Progressive Generalization Bound Exploration and Watchdog Monitor 

**Authors**: Siyuan Liu, Wenjing Liu, Zhiwei Xu, Xin Wang, Bo Chen, Tao Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.15903)  

**Abstract**: Empowered by large language models (LLMs), intelligent agents have become a popular paradigm for interacting with open environments to facilitate AI deployment. However, hallucinations generated by LLMs-where outputs are inconsistent with facts-pose a significant challenge, undermining the credibility of intelligent agents. Only if hallucinations can be mitigated, the intelligent agents can be used in real-world without any catastrophic risk. Therefore, effective detection and mitigation of hallucinations are crucial to ensure the dependability of agents. Unfortunately, the related approaches either depend on white-box access to LLMs or fail to accurately identify hallucinations. To address the challenge posed by hallucinations of intelligent agents, we present HalMit, a novel black-box watchdog framework that models the generalization bound of LLM-empowered agents and thus detect hallucinations without requiring internal knowledge of the LLM's architecture. Specifically, a probabilistic fractal sampling technique is proposed to generate a sufficient number of queries to trigger the incredible responses in parallel, efficiently identifying the generalization bound of the target agent. Experimental evaluations demonstrate that HalMit significantly outperforms existing approaches in hallucination monitoring. Its black-box nature and superior performance make HalMit a promising solution for enhancing the dependability of LLM-powered systems. 

---
# A Generative Model for Disentangling Galaxy Photometric Parameters 

**Authors**: Keen Leung, Colen Yan, Jun Yin  

**Link**: [PDF](https://arxiv.org/pdf/2507.15898)  

**Abstract**: Ongoing and future photometric surveys will produce unprecedented volumes of galaxy images, necessitating robust, efficient methods for deriving galaxy morphological parameters at scale. Traditional approaches, such as parametric light-profile fitting, offer valuable insights but become computationally prohibitive when applied to billions of sources. In this work, we propose a Conditional AutoEncoder (CAE) framework to simultaneously model and characterize galaxy morphology. Our CAE is trained on a suite of realistic mock galaxy images generated via GalSim, encompassing a broad range of galaxy types, photometric parameters (e.g., flux, half-light radius, Sersic index, ellipticity), and observational conditions. By encoding each galaxy image into a low-dimensional latent representation conditioned on key parameters, our model effectively recovers these morphological features in a disentangled manner, while also reconstructing the original image. The results demonstrate that the CAE approach can accurately and efficiently infer complex structural properties, offering a powerful alternative to existing methods. 

---
# ReDi: Rectified Discrete Flow 

**Authors**: Jaehoon Yoo, Wonjung Kim, Seunghoon Hong  

**Link**: [PDF](https://arxiv.org/pdf/2507.15897)  

**Abstract**: Discrete Flow-based Models (DFMs) are powerful generative models for high-quality discrete data but typically suffer from slow sampling speeds due to their reliance on iterative decoding processes. This reliance on a multi-step process originates from the factorization approximation of DFMs, which is necessary for handling high-dimensional data. In this paper, we rigorously characterize the approximation error from factorization using Conditional Total Correlation (TC), which depends on the coupling. To reduce the Conditional TC and enable efficient few-step generation, we propose Rectified Discrete Flow (ReDi), a novel iterative method that reduces factorization error by rectifying the coupling between source and target distributions. We theoretically prove that each ReDi step guarantees a monotonic decreasing Conditional TC, ensuring its convergence. Empirically, ReDi significantly reduces Conditional TC and enables few-step generation. Moreover, we demonstrate that the rectified couplings are well-suited for training efficient one-step models on image generation. ReDi offers a simple and theoretically grounded approach for tackling the few-step challenge, providing a new perspective on efficient discrete data synthesis. Code is available at this https URL 

---
# Systole-Conditioned Generative Cardiac Motion 

**Authors**: Shahar Zuler, Gal Lifshitz, Hadar Averbuch-Elor, Dan Raviv  

**Link**: [PDF](https://arxiv.org/pdf/2507.15894)  

**Abstract**: Accurate motion estimation in cardiac computed tomography (CT) imaging is critical for assessing cardiac function and surgical planning. Data-driven methods have become the standard approach for dense motion estimation, but they rely on vast amounts of labeled data with dense ground-truth (GT) motion annotations, which are often unfeasible to obtain. To address this limitation, we present a novel approach that synthesizes realistically looking pairs of cardiac CT frames enriched with dense 3D flow field annotations.
Our method leverages a conditional Variational Autoencoder (CVAE), which incorporates a novel multi-scale feature conditioning mechanism and is trained to generate 3D flow fields conditioned on a single CT frame. By applying the generated flow field to warp the given frame, we create pairs of frames that simulate realistic myocardium deformations across the cardiac cycle. These pairs serve as fully annotated data samples, providing optical flow GT annotations. Our data generation pipeline could enable the training and validation of more complex and accurate myocardium motion models, allowing for substantially reducing reliance on manual annotations.
Our code, along with animated generated samples and additional material, is available on our project page: this https URL. 

---
# Dr. Boot: Bootstrapping Program Synthesis Language Models to Perform Repairing 

**Authors**: Noah van der Vleuten  

**Link**: [PDF](https://arxiv.org/pdf/2507.15889)  

**Abstract**: Language models for program synthesis are usually trained and evaluated on programming competition datasets (MBPP, APPS). However, these datasets are limited in size and quality, while these language models are extremely data hungry. Additionally, the language models have a misaligned program synthesis process compared to humans. While humans iteratively develop code with the help of a compiler, most program synthesis models currently produce code in one go. To solve these issues, we introduce a bootstrapping algorithm for program synthesis, that supports teaching models how to repair. We show that bootstrapping consistently outperforms regular fine-tuning. Compared to other work, our bootstrapped model performs on par with fine-tuned models that are 68\% larger. Notably, bootstrapping with repairing also improves non-repairing performance compared to regular bootstrapping during inference. However, on our models, repairing during inference is likely inferior to simply sampling the same number of solutions. Furthermore, we find that there are issues with the example test cases in the training portion of the APPS dataset that are valuable to the community, as many repairing and reinforcement learning methods rely on them. 

---
# AlgoTune: Can Language Models Speed Up General-Purpose Numerical Programs? 

**Authors**: Ori Press, Brandon Amos, Haoyu Zhao, Yikai Wu, Samuel K. Ainsworth, Dominik Krupke, Patrick Kidger, Touqir Sajed, Bartolomeo Stellato, Jisun Park, Nathanael Bosch, Eli Meril, Albert Steppi, Arman Zharmagambetov, Fangzhao Zhang, David Perez-Pineiro, Alberto Mercurio, Ni Zhan, Talor Abramovich, Kilian Lieret, Hanlin Zhang, Shirley Huang, Matthias Bethge, Ofir Press  

**Link**: [PDF](https://arxiv.org/pdf/2507.15887)  

**Abstract**: Despite progress in language model (LM) capabilities, evaluations have thus far focused on models' performance on tasks that humans have previously solved, including in programming (Jimenez et al., 2024) and mathematics (Glazer et al., 2024). We therefore propose testing models' ability to design and implement algorithms in an open-ended benchmark: We task LMs with writing code that efficiently solves computationally challenging problems in computer science, physics, and mathematics. Our AlgoTune benchmark consists of 155 coding tasks collected from domain experts and a framework for validating and timing LM-synthesized solution code, which is compared to reference implementations from popular open-source packages. In addition, we develop a baseline LM agent, AlgoTuner, and evaluate its performance across a suite of frontier models. AlgoTuner achieves an average 1.72x speedup against our reference solvers, which use libraries such as SciPy, sk-learn and CVXPY. However, we find that current models fail to discover algorithmic innovations, instead preferring surface-level optimizations. We hope that AlgoTune catalyzes the development of LM agents exhibiting creative problem solving beyond state-of-the-art human performance. 

---
# Combining Cost-Constrained Runtime Monitors for AI Safety 

**Authors**: Tim Tian Hua, James Baskerville, Henri Lemoine, Mia Hopman, Aryan Bhatt, Tyler Tracy  

**Link**: [PDF](https://arxiv.org/pdf/2507.15886)  

**Abstract**: Monitoring AIs at runtime can help us detect and stop harmful actions. In this paper, we study how to combine multiple runtime monitors into a single monitoring protocol. The protocol's objective is to maximize the probability of applying a safety intervention on misaligned outputs (i.e., maximize recall). Since running monitors and applying safety interventions are costly, the protocol also needs to adhere to an average-case budget constraint. Taking the monitors' performance and cost as given, we develop an algorithm to find the most efficient protocol. The algorithm exhaustively searches over when and which monitors to call, and allocates safety interventions based on the Neyman-Pearson lemma. By focusing on likelihood ratios and strategically trading off spending on monitors against spending on interventions, we more than double our recall rate compared to a naive baseline in a code review setting. We also show that combining two monitors can Pareto dominate using either monitor alone. Our framework provides a principled methodology for combining existing monitors to detect undesirable behavior in cost-sensitive settings. 

---
# Document Haystack: A Long Context Multimodal Image/Document Understanding Vision LLM Benchmark 

**Authors**: Goeric Huybrechts, Srikanth Ronanki, Sai Muralidhar Jayanthi, Jack Fitzgerald, Srinivasan Veeravanallur  

**Link**: [PDF](https://arxiv.org/pdf/2507.15882)  

**Abstract**: The proliferation of multimodal Large Language Models has significantly advanced the ability to analyze and understand complex data inputs from different modalities. However, the processing of long documents remains under-explored, largely due to a lack of suitable benchmarks. To address this, we introduce Document Haystack, a comprehensive benchmark designed to evaluate the performance of Vision Language Models (VLMs) on long, visually complex documents. Document Haystack features documents ranging from 5 to 200 pages and strategically inserts pure text or multimodal text+image "needles" at various depths within the documents to challenge VLMs' retrieval capabilities. Comprising 400 document variants and a total of 8,250 questions, it is supported by an objective, automated evaluation framework. We detail the construction and characteristics of the Document Haystack dataset, present results from prominent VLMs and discuss potential research avenues in this area. 

---
# Salience Adjustment for Context-Based Emotion Recognition 

**Authors**: Bin Han, Jonathan Gratch  

**Link**: [PDF](https://arxiv.org/pdf/2507.15878)  

**Abstract**: Emotion recognition in dynamic social contexts requires an understanding of the complex interaction between facial expressions and situational cues. This paper presents a salience-adjusted framework for context-aware emotion recognition with Bayesian Cue Integration (BCI) and Visual-Language Models (VLMs) to dynamically weight facial and contextual information based on the expressivity of facial cues. We evaluate this approach using human annotations and automatic emotion recognition systems in prisoner's dilemma scenarios, which are designed to evoke emotional reactions. Our findings demonstrate that incorporating salience adjustment enhances emotion recognition performance, offering promising directions for future research to extend this framework to broader social contexts and multimodal applications. 

---
# Small Edits, Big Consequences: Telling Good from Bad Robustness in Large Language Models 

**Authors**: Altynbek Ismailov, Salia Asanova  

**Link**: [PDF](https://arxiv.org/pdf/2507.15868)  

**Abstract**: Large language models (LLMs) now write code in settings where misreading a single word can break safety or cost money, yet we still expect them to overlook stray typos. To probe where useful robustness ends and harmful insensitivity begins, we compile 50 LeetCode problems and craft three minimal prompt perturbations that should vary in importance: (i) progressive underspecification deleting 10 % of words per step; (ii) lexical flip swapping a pivotal quantifier ("max" to "min"); and (iii) jargon inflation replacing a common noun with an obscure technical synonym. Six frontier models, including three "reasoning-tuned" versions, solve each mutated prompt, and their Python outputs are checked against the original test suites to reveal whether they reused the baseline solution or adapted. Among 11 853 generations we observe a sharp double asymmetry. Models remain correct in 85 % of cases even after 90 % of the prompt is missing, showing over-robustness to underspecification, yet only 54 % react to a single quantifier flip that reverses the task, with reasoning-tuned variants even less sensitive than their bases. Jargon edits lie in between, passing through 56 %. Current LLMs thus blur the line between harmless noise and meaning - changing edits, often treating both as ignorable. Masking salient anchors such as function names can force re - evaluation. We advocate evaluation and training protocols that reward differential sensitivity: stay steady under benign noise but adapt - or refuse - when semantics truly change. 

---
# RDMA: Cost Effective Agent-Driven Rare Disease Discovery within Electronic Health Record Systems 

**Authors**: John Wu, Adam Cross, Jimeng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2507.15867)  

**Abstract**: Rare diseases affect 1 in 10 Americans, yet standard ICD coding systems fail to capture these conditions in electronic health records (EHR), leaving crucial information buried in clinical notes. Current approaches struggle with medical abbreviations, miss implicit disease mentions, raise privacy concerns with cloud processing, and lack clinical reasoning abilities. We present Rare Disease Mining Agents (RDMA), a framework that mirrors how medical experts identify rare disease patterns in EHR. RDMA connects scattered clinical observations that together suggest specific rare conditions. By handling clinical abbreviations, recognizing implicit disease patterns, and applying contextual reasoning locally on standard hardware, RDMA reduces privacy risks while improving F1 performance by upwards of 30\% and decreasing inferences costs 10-fold. This approach helps clinicians avoid the privacy risk of using cloud services while accessing key rare disease information from EHR systems, supporting earlier diagnosis for rare disease patients. Available at this https URL. 

---
# eSapiens's DEREK Module: Deep Extraction & Reasoning Engine for Knowledge with LLMs 

**Authors**: Isaac Shi, Zeyuan Li, Fan Liu, Wenli Wang, Lewei He, Yang Yang, Tianyu Shi  

**Link**: [PDF](https://arxiv.org/pdf/2507.15863)  

**Abstract**: We present the DEREK (Deep Extraction & Reasoning Engine for Knowledge) Module, a secure and scalable Retrieval-Augmented Generation pipeline designed specifically for enterprise document question answering. Designed and implemented by eSapiens, the system ingests heterogeneous content (PDF, Office, web), splits it into 1,000-token overlapping chunks, and indexes them in a hybrid HNSW+BM25 store. User queries are refined by GPT-4o, retrieved via combined vector+BM25 search, reranked with Cohere, and answered by an LLM using CO-STAR prompt engineering. A LangGraph verifier enforces citation overlap, regenerating answers until every claim is grounded. On four LegalBench subsets, 1000-token chunks improve Recall@50 by approximately 1 pp and hybrid+rerank boosts Precision@10 by approximately 7 pp; the verifier raises TRACe Utilization above 0.50 and limits unsupported statements to less than 3%. All components run in containers, enforce end-to-end TLS 1.3 and AES-256. These results demonstrate that the DEREK module delivers accurate, traceable, and production-ready document QA with minimal operational overhead. The module is designed to meet enterprise demands for secure, auditable, and context-faithful retrieval, providing a reliable baseline for high-stakes domains such as legal and finance. 

---
# Decentralized AI-driven IoT Architecture for Privacy-Preserving and Latency-Optimized Healthcare in Pandemic and Critical Care Scenarios 

**Authors**: Harsha Sammangi, Aditya Jagatha, Giridhar Reddy Bojja, Jun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.15859)  

**Abstract**: AI Innovations in the IoT for Real-Time Patient Monitoring On one hand, the current traditional centralized healthcare architecture poses numerous issues, including data privacy, delay, and security. Here, we present an AI-enabled decentralized IoT architecture that can address such challenges during a pandemic and critical care settings. This work presents our architecture to enhance the effectiveness of the current available federated learning, blockchain, and edge computing approach, maximizing data privacy, minimizing latency, and improving other general system metrics. Experimental results demonstrate transaction latency, energy consumption, and data throughput orders of magnitude lower than competitive cloud solutions. 

---
# GPI-Net: Gestalt-Guided Parallel Interaction Network via Orthogonal Geometric Consistency for Robust Point Cloud Registration 

**Authors**: Weikang Gu, Mingyue Han, Li Xue, Heng Dong, Changcai Yang, Riqing Chen, Lifang Wei  

**Link**: [PDF](https://arxiv.org/pdf/2507.14452)  

**Abstract**: The accurate identification of high-quality correspondences is a prerequisite task in feature-based point cloud registration. However, it is extremely challenging to handle the fusion of local and global features due to feature redundancy and complex spatial relationships. Given that Gestalt principles provide key advantages in analyzing local and global relationships, we propose a novel Gestalt-guided Parallel Interaction Network via orthogonal geometric consistency (GPI-Net) in this paper. It utilizes Gestalt principles to facilitate complementary communication between local and global information. Specifically, we introduce an orthogonal integration strategy to optimally reduce redundant information and generate a more compact global structure for high-quality correspondences. To capture geometric features in correspondences, we leverage a Gestalt Feature Attention (GFA) block through a hybrid utilization of self-attention and cross-attention mechanisms. Furthermore, to facilitate the integration of local detail information into the global structure, we design an innovative Dual-path Multi-Granularity parallel interaction aggregation (DMG) block to promote information exchange across different granularities. Extensive experiments on various challenging tasks demonstrate the superior performance of our proposed GPI-Net in comparison to existing methods. The code will be released at this https URL. 

---
