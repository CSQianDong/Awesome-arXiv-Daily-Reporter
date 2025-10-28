# Alita-G: Self-Evolving Generative Agent for Agent Generation 

**Authors**: Jiahao Qiu, Xuan Qi, Hongru Wang, Xinzhe Juan, Yimin Wang, Zelin Zhao, Jiayi Geng, Jiacheng Guo, Peihang Li, Jingzhe Shi, Shilong Liu, Mengdi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.23601)  

**Abstract**: Large language models (LLMs) have been shown to perform better when scaffolded into agents with memory, tools, and feedback. Beyond this, self-evolving agents have emerged, but current work largely limits adaptation to prompt rewriting or failure retries. Therefore, we present ALITA-G, a self-evolution framework that transforms a general-purpose agent into a domain expert by systematically generating, abstracting, and curating Model Context Protocol (MCP) tools. In this framework, a generalist agent executes a curated suite of target-domain tasks and synthesizes candidate MCPs from successful trajectories. These are then abstracted to parameterized primitives and consolidated into an MCP Box. At inference time, ALITA-G performs retrieval-augmented MCP selection with the help of each tool's descriptions and use cases, before executing an agent equipped with the MCP Executor. Across several benchmarks GAIA, PathVQA, and Humanity's Last Exam, ALITA-G attains strong gains while reducing computation costs. On GAIA validation, it achieves 83.03% pass@1 and 89.09% pass@3, establishing a new state-of-the-art result while reducing mean tokens per example by approximately 15% relative to a strong baseline agent. ALITA-G thus provides a principled pathway from generalist capability to reusable, domain-specific competence, improving both accuracy and efficiency on complex reasoning tasks. 

---
# Multi-Agent Evolve: LLM Self-Improve through Co-evolution 

**Authors**: Yixing Chen, Yiding Wang, Siqi Zhu, Haofei Yu, Tao Feng, Muhan Zhan, Mostofa Patwary, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2510.23595)  

**Abstract**: Reinforcement Learning (RL) has demonstrated significant potential in enhancing the reasoning capabilities of large language models (LLMs). However, the success of RL for LLMs heavily relies on human-curated datasets and verifiable rewards, which limit their scalability and generality. Recent Self-Play RL methods, inspired by the success of the paradigm in games and Go, aim to enhance LLM reasoning capabilities without human-annotated data. However, their methods primarily depend on a grounded environment for feedback (e.g., a Python interpreter or a game engine); extending them to general domains remains challenging. To address these challenges, we propose Multi-Agent Evolve (MAE), a framework that enables LLMs to self-evolve in solving diverse tasks, including mathematics, reasoning, and general knowledge Q&A. The core design of MAE is based on a triplet of interacting agents (Proposer, Solver, Judge) that are instantiated from a single LLM, and applies reinforcement learning to optimize their behaviors. The Proposer generates questions, the Solver attempts solutions, and the Judge evaluates both while co-evolving. Experiments on Qwen2.5-3B-Instruct demonstrate that MAE achieves an average improvement of 4.54% on multiple benchmarks. These results highlight MAE as a scalable, data-efficient method for enhancing the general reasoning abilities of LLMs with minimal reliance on human-curated supervision. 

---
# Reduced AI Acceptance After the Generative AI Boom: Evidence From a Two-Wave Survey Study 

**Authors**: Joachim Baumann, Aleksandra Urman, Ulrich Leicht-Deobald, Zachary J. Roman, Anikó Hannák, Markus Christen  

**Link**: [PDF](https://arxiv.org/pdf/2510.23578)  

**Abstract**: The rapid adoption of generative artificial intelligence (GenAI) technologies has led many organizations to integrate AI into their products and services, often without considering user preferences. Yet, public attitudes toward AI use, especially in impactful decision-making scenarios, are underexplored. Using a large-scale two-wave survey study (n_wave1=1514, n_wave2=1488) representative of the Swiss population, we examine shifts in public attitudes toward AI before and after the launch of ChatGPT. We find that the GenAI boom is significantly associated with reduced public acceptance of AI (see Figure 1) and increased demand for human oversight in various decision-making contexts. The proportion of respondents finding AI "not acceptable at all" increased from 23% to 30%, while support for human-only decision-making rose from 18% to 26%. These shifts have amplified existing social inequalities in terms of widened educational, linguistic, and gender gaps post-boom. Our findings challenge industry assumptions about public readiness for AI deployment and highlight the critical importance of aligning technological development with evolving public preferences. 

---
# ReCode: Unify Plan and Action for Universal Granularity Control 

**Authors**: Zhaoyang Yu, Jiayi Zhang, Huixue Su, Yufan Zhao, Yifan Wu, Mingyi Deng, Jinyu Xiang, Yizhang Lin, Lingxiao Tang, Yingchao Li, Yuyu Luo, Bang Liu, Chenglin Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.23564)  

**Abstract**: Real-world tasks require decisions at varying granularities, and humans excel at this by leveraging a unified cognitive representation where planning is fundamentally understood as a high-level form of action. However, current Large Language Model (LLM)-based agents lack this crucial capability to operate fluidly across decision granularities. This limitation stems from existing paradigms that enforce a rigid separation between high-level planning and low-level action, which impairs dynamic adaptability and limits generalization. We propose ReCode (Recursive Code Generation), a novel paradigm that addresses this limitation by unifying planning and action within a single code representation. In this representation, ReCode treats high-level plans as abstract placeholder functions, which the agent then recursively decomposes into finer-grained sub-functions until reaching primitive actions. This recursive approach dissolves the rigid boundary between plan and action, enabling the agent to dynamically control its decision granularity. Furthermore, the recursive structure inherently generates rich, multi-granularity training data, enabling models to learn hierarchical decision-making processes. Extensive experiments show ReCode significantly surpasses advanced baselines in inference performance and demonstrates exceptional data efficiency in training, validating our core insight that unifying planning and action through recursive code generation is a powerful and effective approach to achieving universal granularity control. The code is available at this https URL. 

---
# OntoPret: An Ontology for the Interpretation of Human Behavior 

**Authors**: Alexis Ellis, Stacie Severyn, Fjollë Novakazi, Hadi Banaee, Cogan Shimizu  

**Link**: [PDF](https://arxiv.org/pdf/2510.23553)  

**Abstract**: As human machine teaming becomes central to paradigms like Industry 5.0, a critical need arises for machines to safely and effectively interpret complex human behaviors. A research gap currently exists between techno centric robotic frameworks, which often lack nuanced models of human behavior, and descriptive behavioral ontologies, which are not designed for real time, collaborative interpretation. This paper addresses this gap by presenting OntoPret, an ontology for the interpretation of human behavior. Grounded in cognitive science and a modular engineering methodology, OntoPret provides a formal, machine processable framework for classifying behaviors, including task deviations and deceptive actions. We demonstrate its adaptability across two distinct use cases manufacturing and gameplay and establish the semantic foundations necessary for advanced reasoning about human intentions. 

---
# JanusCoder: Towards a Foundational Visual-Programmatic Interface for Code Intelligence 

**Authors**: Qiushi Sun, Jingyang Gong, Yang Liu, Qiaosheng Chen, Lei Li, Kai Chen, Qipeng Guo, Ben Kao, Fei Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2510.23538)  

**Abstract**: The scope of neural code intelligence is rapidly expanding beyond text-based source code to encompass the rich visual outputs that programs generate. This visual dimension is critical for advanced applications like flexible content generation and precise, program-driven editing of visualizations. However, progress has been impeded by the scarcity of high-quality multimodal code data, a bottleneck stemming from challenges in synthesis and quality assessment. To address these challenges, we make contributions from both a data and modeling perspective. We first introduce a complete synthesis toolkit that leverages reciprocal synergies between data modalities to efficiently produce a large-scale, high-quality corpus spanning from standard charts to complex interactive web UIs and code-driven animations. Leveraging this toolkit, we construct JanusCode-800K, the largest multimodal code corpus to date. This powers the training of our models, JanusCoder and JanusCoderV, which establish a visual-programmatic interface for generating code from textual instructions, visual inputs, or a combination of both. Our unified model is a departure from existing approaches that build specialized models for isolated tasks. Extensive experiments on both text-centric and vision-centric coding tasks demonstrate the superior performance of the JanusCoder series, with our 7B to 14B scale models approaching or even exceeding the performance of commercial models. Furthermore, extensive analysis provides key insights into harmonizing programmatic logic with its visual expression. Our code and checkpoints will are available at this https URL. 

---
# When No Paths Lead to Rome: Benchmarking Systematic Neural Relational Reasoning 

**Authors**: Anirban Das, Irtaza Khalid, Rafael Peñaloza, Steven Schockaert  

**Link**: [PDF](https://arxiv.org/pdf/2510.23532)  

**Abstract**: Designing models that can learn to reason in a systematic way is an important and long-standing challenge. In recent years, a wide range of solutions have been proposed for the specific case of systematic relational reasoning, including Neuro-Symbolic approaches, variants of the Transformer architecture, and specialised Graph Neural Networks. However, existing benchmarks for systematic relational reasoning focus on an overly simplified setting, based on the assumption that reasoning can be reduced to composing relational paths. In fact, this assumption is hard-baked into the architecture of several recent models, leading to approaches that can perform well on existing benchmarks but are difficult to generalise to other settings. To support further progress in the field of systematic relational reasoning with neural networks, we introduce NoRA, a new benchmark which adds several levels of difficulty and requires models to go beyond path-based reasoning. 

---
# Toward Carbon-Neutral Human AI: Rethinking Data, Computation, and Learning Paradigms for Sustainable Intelligence 

**Authors**: KC Santosh, Rodrigue Rizk, Longwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.23524)  

**Abstract**: The rapid advancement of Artificial Intelligence (AI) has led to unprecedented computational demands, raising significant environmental and ethical concerns. This paper critiques the prevailing reliance on large-scale, static datasets and monolithic training paradigms, advocating for a shift toward human-inspired, sustainable AI solutions. We introduce a novel framework, Human AI (HAI), which emphasizes incremental learning, carbon-aware optimization, and human-in-the-loop collaboration to enhance adaptability, efficiency, and accountability. By drawing parallels with biological cognition and leveraging dynamic architectures, HAI seeks to balance performance with ecological responsibility. We detail the theoretical foundations, system design, and operational principles that enable AI to learn continuously and contextually while minimizing carbon footprints and human annotation costs. Our approach addresses pressing challenges in active learning, continual adaptation, and energy-efficient model deployment, offering a pathway toward responsible, human-centered artificial intelligence. 

---
# Emotion-Coherent Reasoning for Multimodal LLMs via Emotional Rationale Verifier 

**Authors**: Hyeongseop Rha, Jeong Hun Yeo, Yeonju Kim, Yong Man Ro  

**Link**: [PDF](https://arxiv.org/pdf/2510.23506)  

**Abstract**: The recent advancement of Multimodal Large Language Models (MLLMs) is transforming human-computer interaction (HCI) from surface-level exchanges into more nuanced and emotionally intelligent communication. To realize this shift, emotion understanding becomes essential allowing systems to capture subtle cues underlying user intent. Furthermore, providing faithful explanations for predicted emotions is crucial to ensure interpretability and build user trust. However, current MLLM-based methods often generate emotion explanations that diverge from the target labels and sometimes even contradict their own predicted emotions. This inconsistency poses a critical risk for misunderstanding and erodes reliability in interactive settings. To address this, we propose a novel approach: the Emotional Rationale Verifier (ERV) and an Explanation Reward. Our method guides the model to produce reasoning that is explicitly consistent with the target emotion during multimodal emotion recognition without modifying the model architecture or requiring additional paired video-description annotations. Our method significantly improves faithful explanation-prediction consistency and explanation emotion accuracy on the MAFW and DFEW datasets. Through extensive experiments and human evaluations, we show that our approach not only enhances alignment between explanation and prediction but also empowers MLLMs to deliver emotionally coherent, trustworthy interactions, marking a key step toward truly human-like HCI systems. 

---
# Are Agents Just Automata? On the Formal Equivalence Between Agentic AI and the Chomsky Hierarchy 

**Authors**: Roham Koohestani, Ziyou Li, Anton Podkopaev, Maliheh Izadi  

**Link**: [PDF](https://arxiv.org/pdf/2510.23487)  

**Abstract**: This paper establishes a formal equivalence between the architectural classes of modern agentic AI systems and the abstract machines of the Chomsky hierarchy. We posit that the memory architecture of an AI agent is the definitive feature determining its computational power and that it directly maps it to a corresponding class of automaton. Specifically, we demonstrate that simple reflex agents are equivalent to Finite Automata, hierarchical task-decomposition agents are equivalent to Pushdown Automata, and agents employing readable/writable memory for reflection are equivalent to TMs. This Automata-Agent Framework provides a principled methodology for right-sizing agent architectures to optimize computational efficiency and cost. More critically, it creates a direct pathway to formal verification, enables the application of mature techniques from automata theory to guarantee agent safety and predictability. By classifying agents, we can formally delineate the boundary between verifiable systems and those whose behavior is fundamentally undecidable. We address the inherent probabilistic nature of LLM-based agents by extending the framework to probabilistic automata that allow quantitative risk analysis. The paper concludes by outlining an agenda for developing static analysis tools and grammars for agentic frameworks. 

---
# Human-AI Collaborative Uncertainty Quantification 

**Authors**: Sima Noorani, Shayan Kiyani, George Pappas, Hamed Hassani  

**Link**: [PDF](https://arxiv.org/pdf/2510.23476)  

**Abstract**: AI predictive systems are increasingly embedded in decision making pipelines, shaping high stakes choices once made solely by humans. Yet robust decisions under uncertainty still rely on capabilities that current AI lacks: domain knowledge not captured by data, long horizon context, and reasoning grounded in the physical world. This gap has motivated growing efforts to design collaborative frameworks that combine the complementary strengths of humans and AI. This work advances this vision by identifying the fundamental principles of Human AI collaboration within uncertainty quantification, a key component of reliable decision making. We introduce Human AI Collaborative Uncertainty Quantification, a framework that formalizes how an AI model can refine a human expert's proposed prediction set with two goals: avoiding counterfactual harm, ensuring the AI does not degrade correct human judgments, and complementarity, enabling recovery of correct outcomes the human missed. At the population level, we show that the optimal collaborative prediction set follows an intuitive two threshold structure over a single score function, extending a classical result in conformal prediction. Building on this insight, we develop practical offline and online calibration algorithms with provable distribution free finite sample guarantees. The online method adapts to distribution shifts, including human behavior evolving through interaction with AI, a phenomenon we call Human to AI Adaptation. Experiments across image classification, regression, and text based medical decision making show that collaborative prediction sets consistently outperform either agent alone, achieving higher coverage and smaller set sizes across various conditions. 

---
# Policy-Aware Generative AI for Safe, Auditable Data Access Governance 

**Authors**: Shames Al Mandalawi, Muzakkiruddin Ahmed Mohammed, Hendrika Maclean, Mert Can Cakmak, John R. Talburt  

**Link**: [PDF](https://arxiv.org/pdf/2510.23474)  

**Abstract**: Enterprises need access decisions that satisfy least privilege, comply with regulations, and remain auditable. We present a policy aware controller that uses a large language model (LLM) to interpret natural language requests against written policies and metadata, not raw data. The system, implemented with Google Gemini~2.0 Flash, executes a six-stage reasoning framework (context interpretation, user validation, data classification, business purpose test, compliance mapping, and risk synthesis) with early hard policy gates and deny by default. It returns APPROVE, DENY, CONDITIONAL together with cited controls and a machine readable rationale. We evaluate on fourteen canonical cases across seven scenario families using a privacy preserving benchmark. Results show Exact Decision Match improving from 10/14 to 13/14 (92.9\%) after applying policy gates, DENY recall rising to 1.00, False Approval Rate on must-deny families dropping to 0, and Functional Appropriateness and Compliance Adherence at 14/14. Expert ratings of rationale quality are high, and median latency is under one minute. These findings indicate that policy constrained LLM reasoning, combined with explicit gates and audit trails, can translate human readable policies into safe, compliant, and traceable machine decisions. 

---
# What are the odds? Risk and uncertainty about AI existential risk 

**Authors**: Marco Grossi  

**Link**: [PDF](https://arxiv.org/pdf/2510.23453)  

**Abstract**: This work is a commentary of the article \href{this https URL}{AI Survival Stories: a Taxonomic Analysis of AI Existential Risk} by Cappelen, Goldstein, and Hawthorne. It is not just a commentary though, but a useful reminder of the philosophical limitations of \say{linear} models of risk. The article will focus on the model employed by the authors: first, I discuss some differences between standard Swiss Cheese models and this one. I then argue that in a situation of epistemic indifference the probability of P(D) is higher than what one might first suggest, given the structural relationships between layers. I then distinguish between risk and uncertainty, and argue that any estimation of P(D) is structurally affected by two kinds of uncertainty: option uncertainty and state-space uncertainty. Incorporating these dimensions of uncertainty into our qualitative discussion on AI existential risk can provide a better understanding of the likeliness of P(D). 

---
# A Neuro-Symbolic Multi-Agent Approach to Legal-Cybersecurity Knowledge Integration 

**Authors**: Chiara Bonfanti, Alessandro Druetto, Cataldo Basile, Tharindu Ranasinghe, Marcos Zampieri  

**Link**: [PDF](https://arxiv.org/pdf/2510.23443)  

**Abstract**: The growing intersection of cybersecurity and law creates a complex information space where traditional legal research tools struggle to deal with nuanced connections between cases, statutes, and technical vulnerabilities. This knowledge divide hinders collaboration between legal experts and cybersecurity professionals. To address this important gap, this work provides a first step towards intelligent systems capable of navigating the increasingly intricate cyber-legal domain. We demonstrate promising initial results on multilingual tasks. 

---
# Causal Deep Q Network 

**Authors**: Elouanes Khelifi, Amir Saki, Usef Faghihi  

**Link**: [PDF](https://arxiv.org/pdf/2510.23424)  

**Abstract**: Deep Q Networks (DQN) have shown remarkable success in various reinforcement learning tasks. However, their reliance on associative learning often leads to the acquisition of spurious correlations, hindering their problem-solving capabilities. In this paper, we introduce a novel approach to integrate causal principles into DQNs, leveraging the PEACE (Probabilistic Easy vAriational Causal Effect) formula for estimating causal effects. By incorporating causal reasoning during training, our proposed framework enhances the DQN's understanding of the underlying causal structure of the environment, thereby mitigating the influence of confounding factors and spurious correlations. We demonstrate that integrating DQNs with causal capabilities significantly enhances their problem-solving capabilities without compromising performance. Experimental results on standard benchmark environments showcase that our approach outperforms conventional DQNs, highlighting the effectiveness of causal reasoning in reinforcement learning. Overall, our work presents a promising avenue for advancing the capabilities of deep reinforcement learning agents through principled causal inference. 

---
# Bid2X: Revealing Dynamics of Bidding Environment in Online Advertising from A Foundation Model Lens 

**Authors**: Jiahao Ji, Tianyu Wang, Yeshu Li, Yushen Huo, Zhilin Zhang, Chuan Yu, Jian Xu, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.23410)  

**Abstract**: Auto-bidding is crucial in facilitating online advertising by automatically providing bids for advertisers. While previous work has made great efforts to model bidding environments for better ad performance, it has limitations in generalizability across environments since these models are typically tailored for specific bidding scenarios. To this end, we approach the scenario-independent principles through a unified function that estimates the achieved effect under specific bids, such as budget consumption, gross merchandise volume (GMV), page views, etc. Then, we propose a bidding foundation model Bid2X to learn this fundamental function from data in various scenarios. Our Bid2X is built over uniform series embeddings that encode heterogeneous data through tailored embedding methods. To capture complex inter-variable and dynamic temporal dependencies in bidding data, we propose two attention mechanisms separately treating embeddings of different variables and embeddings at different times as attention tokens for representation learning. On top of the learned variable and temporal representations, a variable-aware fusion module is used to perform adaptive bidding outcome prediction. To model the unique bidding data distribution, we devise a zero-inflated projection module to incorporate the estimated non-zero probability into its value prediction, which makes up a joint optimization objective containing classification and regression. The objective is proven to converge to the zero-inflated distribution. Our model has been deployed on the ad platform in Taobao, one of the world's largest e-commerce platforms. Offline evaluation on eight datasets exhibits Bid2X's superiority compared to various baselines and its generality across different scenarios. Bid2X increased GMV by 4.65% and ROI by 2.44% in online A/B tests, paving the way for bidding foundation model in computational advertising. 

---
# AutoStreamPipe: LLM Assisted Automatic Generation of Data Stream Processing Pipelines 

**Authors**: Abolfazl Younesi, Zahra Najafabadi Samani, Thomas Fahringer  

**Link**: [PDF](https://arxiv.org/pdf/2510.23408)  

**Abstract**: Data pipelines are essential in stream processing as they enable the efficient collection, processing, and delivery of real-time data, supporting rapid data analysis. In this paper, we present AutoStreamPipe, a novel framework that employs Large Language Models (LLMs) to automate the design, generation, and deployment of stream processing pipelines. AutoStreamPipe bridges the semantic gap between high-level user intent and platform-specific implementations across distributed stream processing systems for structured multi-agent reasoning by integrating a Hypergraph of Thoughts (HGoT) as an extended version of GoT. AutoStreamPipe combines resilient execution strategies, advanced query analysis, and HGoT to deliver pipelines with good accuracy. Experimental evaluations on diverse pipelines demonstrate that AutoStreamPipe significantly reduces development time (x6.3) and error rates (x5.19), as measured by a novel Error-Free Score (EFS), compared to LLM code-generation methods. 

---
# Opinion Mining Based Entity Ranking using Fuzzy Logic Algorithmic Approach 

**Authors**: Pratik N. Kalamkar, A.G. Phakatkar  

**Link**: [PDF](https://arxiv.org/pdf/2510.23384)  

**Abstract**: Opinions are central to almost all human activities and are key influencers of our behaviors. In current times due to growth of social networking website and increase in number of e-commerce site huge amount of opinions are now available on web. Given a set of evaluative statements that contain opinions (or sentiments) about an Entity, opinion mining aims to extract attributes and components of the object that have been commented on in each statement and to determine whether the comments are positive, negative or neutral. While lot of research recently has been done in field of opinion mining and some of it dealing with ranking of entities based on review or opinion set, classifying opinions into finer granularity level and then ranking entities has never been done before. In this paper method for opinion mining from statements at a deeper level of granularity is proposed. This is done by using fuzzy logic reasoning, after which entities are ranked as per this information. 

---
# Planning Ahead with RSA: Efficient Signalling in Dynamic Environments by Projecting User Awareness across Future Timesteps 

**Authors**: Anwesha Das, John Duff, Jörg Hoffmann, Vera Demberg  

**Link**: [PDF](https://arxiv.org/pdf/2510.23340)  

**Abstract**: Adaptive agent design offers a way to improve human-AI collaboration on time-sensitive tasks in rapidly changing environments. In such cases, to ensure the human maintains an accurate understanding of critical task elements, an assistive agent must not only identify the highest priority information but also estimate how and when this information can be communicated most effectively, given that human attention represents a zero-sum cognitive resource where focus on one message diminishes awareness of other or upcoming information. We introduce a theoretical framework for adaptive signalling which meets these challenges by using principles of rational communication, formalised as Bayesian reference resolution using the Rational Speech Act (RSA) modelling framework, to plan a sequence of messages which optimise timely alignment between user belief and a dynamic environment. The agent adapts message specificity and timing to the particulars of a user and scenario based on projections of how prior-guided interpretation of messages will influence attention to the interface and subsequent belief update, across several timesteps out to a fixed horizon. In a comparison to baseline methods, we show that this effectiveness depends crucially on combining multi-step planning with a realistic model of user awareness. As the first application of RSA for communication in a dynamic environment, and for human-AI interaction in general, we establish theoretical foundations for pragmatic communication in human-agent teams, highlighting how insights from cognitive science can be capitalised to inform the design of assistive agents. 

---
# CNOT Minimal Circuit Synthesis: A Reinforcement Learning Approach 

**Authors**: Riccardo Romanello, Daniele Lizzio Bosco, Jacopo Cossio, Dusan Sutulovic, Giuseppe Serra, Carla Piazza, Paolo Burelli  

**Link**: [PDF](https://arxiv.org/pdf/2510.23304)  

**Abstract**: CNOT gates are fundamental to quantum computing, as they facilitate entanglement, a crucial resource for quantum algorithms. Certain classes of quantum circuits are constructed exclusively from CNOT gates. Given their widespread use, it is imperative to minimise the number of CNOT gates employed. This problem, known as CNOT minimisation, remains an open challenge, with its computational complexity yet to be fully characterised. In this work, we introduce a novel reinforcement learning approach to address this task. Instead of training multiple reinforcement learning agents for different circuit sizes, we use a single agent up to a fixed size $m$. Matrices of sizes different from m are preprocessed using either embedding or Gaussian striping. To assess the efficacy of our approach, we trained an agent with m = 8, and evaluated it on matrices of size n that range from 3 to 15. The results we obtained show that our method overperforms the state-of-the-art algorithm as the value of n increases. 

---
# Accelerating IC Thermal Simulation Data Generation via Block Krylov and Operator Action 

**Authors**: Hong Wang, Wenkai Yang, Jie Wang, Huanshuo Dong, Zijie Geng, Zhen Huang, Depeng Xie, Zhezheng Hao, Hande Dong  

**Link**: [PDF](https://arxiv.org/pdf/2510.23221)  

**Abstract**: Recent advances in data-driven approaches, such as neural operators (NOs), have shown substantial efficacy in reducing the solution time for integrated circuit (IC) thermal simulations. However, a limitation of these approaches is requiring a large amount of high-fidelity training data, such as chip parameters and temperature distributions, thereby incurring significant computational costs. To address this challenge, we propose a novel algorithm for the generation of IC thermal simulation data, named block Krylov and operator action (BlocKOA), which simultaneously accelerates the data generation process and enhances the precision of generated data. BlocKOA is specifically designed for IC applications. Initially, we use the block Krylov algorithm based on the structure of the heat equation to quickly obtain a few basic solutions. Then we combine them to get numerous temperature distributions that satisfy the physical constraints. Finally, we apply heat operators on these functions to determine the heat source distributions, efficiently generating precise data points. Theoretical analysis shows that the time complexity of BlocKOA is one order lower than the existing method. Experimental results further validate its efficiency, showing that BlocKOA achieves a 420-fold speedup in generating thermal simulation data for 5000 chips with varying physical parameters and IC structures. Even with just 4% of the generation time, data-driven approaches trained on the data generated by BlocKOA exhibits comparable performance to that using the existing method. 

---
# Human-Like Goalkeeping in a Realistic Football Simulation: a Sample-Efficient Reinforcement Learning Approach 

**Authors**: Alessandro Sestini, Joakim Bergdahl, Jean-Philippe Barrette-LaPierre, Florian Fuchs, Brady Chen, Micheal Jones, Linus Gisslén  

**Link**: [PDF](https://arxiv.org/pdf/2510.23216)  

**Abstract**: While several high profile video games have served as testbeds for Deep Reinforcement Learning (DRL), this technique has rarely been employed by the game industry for crafting authentic AI behaviors. Previous research focuses on training super-human agents with large models, which is impractical for game studios with limited resources aiming for human-like agents. This paper proposes a sample-efficient DRL method tailored for training and fine-tuning agents in industrial settings such as the video game industry. Our method improves sample efficiency of value-based DRL by leveraging pre-collected data and increasing network plasticity. We evaluate our method training a goalkeeper agent in EA SPORTS FC 25, one of the best-selling football simulations today. Our agent outperforms the game's built-in AI by 10% in ball saving rate. Ablation studies show that our method trains agents 50% faster compared to standard DRL methods. Finally, qualitative evaluation from domain experts indicates that our approach creates more human-like gameplay compared to hand-crafted agents. As a testimony of the impact of the approach, the method is intended to replace the hand-crafted counterpart in next iterations of the series. 

---
# AUPO - Abstracted Until Proven Otherwise: A Reward Distribution Based Abstraction Algorithm 

**Authors**: Robin Schmöcker, Alexander Dockhorn, Bodo Rosenhahn  

**Link**: [PDF](https://arxiv.org/pdf/2510.23214)  

**Abstract**: We introduce a novel, drop-in modification to Monte Carlo Tree Search's (MCTS) decision policy that we call AUPO. Comparisons based on a range of IPPC benchmark problems show that AUPO clearly outperforms MCTS. AUPO is an automatic action abstraction algorithm that solely relies on reward distribution statistics acquired during the MCTS. Thus, unlike other automatic abstraction algorithms, AUPO requires neither access to transition probabilities nor does AUPO require a directed acyclic search graph to build its abstraction, allowing AUPO to detect symmetric actions that state-of-the-art frameworks like ASAP struggle with when the resulting symmetric states are far apart in state space. Furthermore, as AUPO only affects the decision policy, it is not mutually exclusive with other abstraction techniques that only affect the tree search. 

---
# Guiding Skill Discovery with Foundation Models 

**Authors**: Zhao Yang, Thomas M. Moerland, Mike Preuss, Aske Plaat, Vincent François-Lavet, Edward S. Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.23167)  

**Abstract**: Learning diverse skills without hand-crafted reward functions could accelerate reinforcement learning in downstream tasks. However, existing skill discovery methods focus solely on maximizing the diversity of skills without considering human preferences, which leads to undesirable behaviors and possibly dangerous skills. For instance, a cheetah robot trained using previous methods learns to roll in all directions to maximize skill diversity, whereas we would prefer it to run without flipping or entering hazardous areas. In this work, we propose a Foundation model Guided (FoG) skill discovery method, which incorporates human intentions into skill discovery through foundation models. Specifically, FoG extracts a score function from foundation models to evaluate states based on human intentions, assigning higher values to desirable states and lower to undesirable ones. These scores are then used to re-weight the rewards of skill discovery algorithms. By optimizing the re-weighted skill discovery rewards, FoG successfully learns to eliminate undesirable behaviors, such as flipping or rolling, and to avoid hazardous areas in both state-based and pixel-based tasks. Interestingly, we show that FoG can discover skills involving behaviors that are difficult to define. Interactive visualisations are available from this https URL. 

---
# Lost in Tokenization: Context as the Key to Unlocking Biomolecular Understanding in Scientific LLMs 

**Authors**: Kai Zhuang, Jiawei Zhang, Yumou Liu, Hanqun Cao, Chunbin Gu, Mengdi Liu, Zhangyang Gao, Zitong Jerry Wang, Xuanhe Zhou, Pheng-Ann Heng, Lijun Wu, Conghui He, Cheng Tan  

**Link**: [PDF](https://arxiv.org/pdf/2510.23127)  

**Abstract**: Scientific Large Language Models (Sci-LLMs) have emerged as a promising frontier for accelerating biological discovery. However, these models face a fundamental challenge when processing raw biomolecular sequences: the tokenization dilemma. Whether treating sequences as a specialized language, risking the loss of functional motif information, or as a separate modality, introducing formidable alignment challenges, current strategies fundamentally limit their reasoning capacity. We challenge this sequence-centric paradigm by positing that a more effective strategy is to provide Sci-LLMs with high-level structured context derived from established bioinformatics tools, thereby bypassing the need to interpret low-level noisy sequence data directly. Through a systematic comparison of leading Sci-LLMs on biological reasoning tasks, we tested three input modes: sequence-only, context-only, and a combination of both. Our findings are striking: the context-only approach consistently and substantially outperforms all other modes. Even more revealing, the inclusion of the raw sequence alongside its high-level context consistently degrades performance, indicating that raw sequences act as informational noise, even for models with specialized tokenization schemes. These results suggest that the primary strength of existing Sci-LLMs lies not in their nascent ability to interpret biomolecular syntax from scratch, but in their profound capacity for reasoning over structured, human-readable knowledge. Therefore, we argue for reframing Sci-LLMs not as sequence decoders, but as powerful reasoning engines over expert knowledge. This work lays the foundation for a new class of hybrid scientific AI agents, repositioning the developmental focus from direct sequence interpretation towards high-level knowledge synthesis. The code is available at this http URL. 

---
# Smaller Models, Smarter Rewards: A Two-Sided Approach to Process and Outcome Rewards 

**Authors**: Jan Niklas Groeneveld, Xi Qin, Alexander Schaefer, Yaad Oren  

**Link**: [PDF](https://arxiv.org/pdf/2510.23083)  

**Abstract**: Generating high-quality code remains a challenge for Large Language Models (LLMs). For the evolution of reasoning models on this task, reward models are a necessary intermediate step. These models judge outcomes or intermediate steps. Decoder-only transformer models can be turned into reward models by introducing a regression layer and supervised fine-tuning. While it is known that reflection capabilities generally increase with the size of a model, we want to investigate whether state-of-the-art small language models like the Phi-4 family can be turned into usable reward models blending the consideration of process rewards and outcome rewards.
Targeting this goal, we construct a dataset of code samples with correctness labels derived from the APPS coding challenge benchmark. We then train a value-head model to estimate the success probability of intermediate outputs. Our evaluation shows that small LLMs are capable of serving as effective reward models or code evaluation critics, successfully identifying correct solutions among multiple candidates. Using this critic, we achieve over a 20% improvement in the search capability of the most accurate code out of multiple generations. 

---
# TLCD: A Deep Transfer Learning Framework for Cross-Disciplinary Cognitive Diagnosis 

**Authors**: Zhifeng Wang, Meixin Su, Yang Yang, Chunyan Zeng, Lizhi Ye  

**Link**: [PDF](https://arxiv.org/pdf/2510.23062)  

**Abstract**: Driven by the dual principles of smart education and artificial intelligence technology, the online education model has rapidly emerged as an important component of the education industry. Cognitive diagnostic technology can utilize students' learning data and feedback information in educational evaluation to accurately assess their ability level at the knowledge level. However, while massive amounts of information provide abundant data resources, they also bring about complexity in feature extraction and scarcity of disciplinary data. In cross-disciplinary fields, traditional cognitive diagnostic methods still face many challenges. Given the differences in knowledge systems, cognitive structures, and data characteristics between different disciplines, this paper conducts in-depth research on neural network cognitive diagnosis and knowledge association neural network cognitive diagnosis, and proposes an innovative cross-disciplinary cognitive diagnosis method (TLCD). This method combines deep learning techniques and transfer learning strategies to enhance the performance of the model in the target discipline by utilizing the common features of the main discipline. The experimental results show that the cross-disciplinary cognitive diagnosis model based on deep learning performs better than the basic model in cross-disciplinary cognitive diagnosis tasks, and can more accurately evaluate students' learning situation. 

---
# A Survey of AI Scientists: Surveying the automatic Scientists and Research 

**Authors**: Guiyao Tie, Pan Zhou, Lichao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2510.23045)  

**Abstract**: Artificial intelligence is undergoing a profound transition from a computational instrument to an autonomous originator of scientific knowledge. This emerging paradigm, the AI scientist, is architected to emulate the complete scientific workflow-from initial hypothesis generation to the final synthesis of publishable findings-thereby promising to fundamentally reshape the pace and scale of discovery. However, the rapid and unstructured proliferation of these systems has created a fragmented research landscape, obscuring overarching methodological principles and developmental trends. This survey provides a systematic and comprehensive synthesis of this domain by introducing a unified, six-stage methodological framework that deconstructs the end-to-end scientific process into: Literature Review, Idea Generation, Experimental Preparation, Experimental Execution, Scientific Writing, and Paper Generation. Through this analytical lens, we chart the field's evolution from early Foundational Modules (2022-2023) to integrated Closed-Loop Systems (2024), and finally to the current frontier of Scalability, Impact, and Human-AI Collaboration (2025-present). By rigorously synthesizing these developments, this survey not only clarifies the current state of autonomous science but also provides a critical roadmap for overcoming remaining challenges in robustness and governance, ultimately guiding the next generation of systems toward becoming trustworthy and indispensable partners in human scientific inquiry. 

---
# Mixed Density Diffuser: Efficient Planning with Non-uniform Temporal Resolution 

**Authors**: Crimson Stambaugh, Rajesh P. N. Rao  

**Link**: [PDF](https://arxiv.org/pdf/2510.23026)  

**Abstract**: Recent studies demonstrate that diffusion planners benefit from sparse-step planning over single-step planning. Training models to skip steps in their trajectories helps capture long-term dependencies without additional or memory computational cost. However, predicting excessively sparse plans degrades performance. We hypothesize this temporal density threshold is non-uniform across a temporal horizon and that certain parts of a planned trajectory should be more densely planned. We propose Mixed Density Diffuser (MDD), a diffusion planner where the densities throughout the horizon are tunable hyperparameters. MDD achieves a new SOTA across the Maze2D, Franka Kitchen, and Antmaze D4RL task domains. 

---
# From Prompt Optimization to Multi-Dimensional Credibility Evaluation: Enhancing Trustworthiness of Chinese LLM-Generated Liver MRI Reports 

**Authors**: Qiuli Wang, Xiaoming Li, Jie Chen, Yongxu Liu, Xingpeng Zhang, Chen Liu, Wei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.23008)  

**Abstract**: Large language models (LLMs) have demonstrated promising performance in generating diagnostic conclusions from imaging findings, thereby supporting radiology reporting, trainee education, and quality control. However, systematic guidance on how to optimize prompt design across different clinical contexts remains underexplored. Moreover, a comprehensive and standardized framework for assessing the trustworthiness of LLM-generated radiology reports is yet to be established. This study aims to enhance the trustworthiness of LLM-generated liver MRI reports by introducing a Multi-Dimensional Credibility Assessment (MDCA) framework and providing guidance on institution-specific prompt optimization. The proposed framework is applied to evaluate and compare the performance of several advanced LLMs, including Kimi-K2-Instruct-0905, Qwen3-235B-A22B-Instruct-2507, DeepSeek-V3, and ByteDance-Seed-OSS-36B-Instruct, using the SiliconFlow platform. 

---
# ProfileXAI: User-Adaptive Explainable AI 

**Authors**: Gilber A. Corrales, Carlos Andrés Ferro Sánchez, Reinel Tabares-Soto, Jesús Alfonso López Sotelo, Gonzalo A. Ruz, Johan Sebastian Piña Durán  

**Link**: [PDF](https://arxiv.org/pdf/2510.22998)  

**Abstract**: ProfileXAI is a model- and domain-agnostic framework that couples post-hoc explainers (SHAP, LIME, Anchor) with retrieval - augmented LLMs to produce explanations for different types of users. The system indexes a multimodal knowledge base, selects an explainer per instance via quantitative criteria, and generates grounded narratives with chat-enabled prompting. On Heart Disease and Thyroid Cancer datasets, we evaluate fidelity, robustness, parsimony, token use, and perceived quality. No explainer dominates: LIME achieves the best fidelity--robustness trade-off (Infidelity $\le 0.30$, $L<0.7$ on Heart Disease); Anchor yields the sparsest, low-token rules; SHAP attains the highest satisfaction ($\bar{x}=4.1$). Profile conditioning stabilizes tokens ($\sigma \le 13\%$) and maintains positive ratings across profiles ($\bar{x}\ge 3.7$, with domain experts at $3.77$), enabling efficient and trustworthy explanations. 

---
# Exploring Semantic-constrained Adversarial Example with Instruction Uncertainty Reduction 

**Authors**: Jin Hu, Jiakai Wang, Linna Jing, Haolin Li, Haodong Liu, Haotong Qin, Aishan Liu, Ke Xu, Xianglong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.22981)  

**Abstract**: Recently, semantically constrained adversarial examples (SemanticAE), which are directly generated from natural language instructions, have become a promising avenue for future research due to their flexible attacking forms. To generate SemanticAEs, current methods fall short of satisfactory attacking ability as the key underlying factors of semantic uncertainty in human instructions, such as referring diversity, descriptive incompleteness, and boundary ambiguity, have not been fully investigated. To tackle the issues, this paper develops a multi-dimensional instruction uncertainty reduction (InSUR) framework to generate more satisfactory SemanticAE, i.e., transferable, adaptive, and effective. Specifically, in the dimension of the sampling method, we propose the residual-driven attacking direction stabilization to alleviate the unstable adversarial optimization caused by the diversity of language references. By coarsely predicting the language-guided sampling process, the optimization process will be stabilized by the designed ResAdv-DDIM sampler, therefore releasing the transferable and robust adversarial capability of multi-step diffusion models. In task modeling, we propose the context-encoded attacking scenario constraint to supplement the missing knowledge from incomplete human instructions. Guidance masking and renderer integration are proposed to regulate the constraints of 2D/3D SemanticAE, activating stronger scenario-adapted attacks. Moreover, in the dimension of generator evaluation, we propose the semantic-abstracted attacking evaluation enhancement by clarifying the evaluation boundary, facilitating the development of more effective SemanticAE generators. Extensive experiments demonstrate the superiority of the transfer attack performance of InSUR. Moreover, we realize the reference-free generation of semantically constrained 3D adversarial examples for the first time. 

---
# Multi-Agent Conditional Diffusion Model with Mean Field Communication as Wireless Resource Allocation Planner 

**Authors**: Kechen Meng, Sinuo Zhang, Rongpeng Li, Xiangming Meng, Chan Wang, Ming Lei, Zhifeng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.22969)  

**Abstract**: In wireless communication systems, efficient and adaptive resource allocation plays a crucial role in enhancing overall Quality of Service (QoS). While centralized Multi-Agent Reinforcement Learning (MARL) frameworks rely on a central coordinator for policy training and resource scheduling, they suffer from scalability issues and privacy risks. In contrast, the Distributed Training with Decentralized Execution (DTDE) paradigm enables distributed learning and decision-making, but it struggles with non-stationarity and limited inter-agent cooperation, which can severely degrade system performance. To overcome these challenges, we propose the Multi-Agent Conditional Diffusion Model Planner (MA-CDMP) for decentralized communication resource management. Built upon the Model-Based Reinforcement Learning (MBRL) paradigm, MA-CDMP employs Diffusion Models (DMs) to capture environment dynamics and plan future trajectories, while an inverse dynamics model guides action generation, thereby alleviating the sample inefficiency and slow convergence of conventional DTDE methods. Moreover, to approximate large-scale agent interactions, a Mean-Field (MF) mechanism is introduced as an assistance to the classifier in DMs. This design mitigates inter-agent non-stationarity and enhances cooperation with minimal communication overhead in distributed settings. We further theoretically establish an upper bound on the distributional approximation error introduced by the MF-based diffusion generation, guaranteeing convergence stability and reliable modeling of multi-agent stochastic dynamics. Extensive experiments demonstrate that MA-CDMP consistently outperforms existing MARL baselines in terms of average reward and QoS metrics, showcasing its scalability and practicality for real-world wireless network optimization. 

---
# GTR-Mamba: Geometry-to-Tangent Routing for Hyperbolic POI Recommendation 

**Authors**: Zhuoxuan Li, Jieyuan Pei, Tangwei Ye, Zhongyuan Lai, Zihan Liu, Fengyuan Xu, Qi Zhang, Liang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.22942)  

**Abstract**: Next Point-of-Interest (POI) recommendation is a critical task in modern Location-Based Social Networks (LBSNs), aiming to model the complex decision-making process of human mobility to provide personalized recommendations for a user's next check-in location. Existing POI recommendation models, predominantly based on Graph Neural Networks and sequential models, have been extensively studied. However, these models face a fundamental limitation: they struggle to simultaneously capture the inherent hierarchical structure of spatial choices and the dynamics and irregular shifts of user-specific temporal contexts. To overcome this limitation, we propose GTR-Mamba, a novel framework for cross-manifold conditioning and routing. GTR-Mamba leverages the distinct advantages of different mathematical spaces for different tasks: it models the static, tree-like preference hierarchies in hyperbolic geometry, while routing the dynamic sequence updates to a novel Mamba layer in the computationally stable and efficient Euclidean tangent space. This process is coordinated by a cross-manifold channel that fuses spatio-temporal information to explicitly steer the State Space Model (SSM), enabling flexible adaptation to contextual changes. Extensive experiments on three real-world datasets demonstrate that GTR-Mamba consistently outperforms state-of-the-art baseline models in next POI recommendation. 

---
# On Generalization in Agentic Tool Calling: CoreThink Agentic Reasoner and MAVEN Dataset 

**Authors**: Vishvesh Bhat, Omkar Ghugarkar, Julian McAuley  

**Link**: [PDF](https://arxiv.org/pdf/2510.22898)  

**Abstract**: Generalization across Agentic tool-calling environments remains a key unsolved challenge in developing reliable agentic reasoning systems. While large language models (LLMs) demonstrate strong performance on isolated benchmarks, their ability to transfer reasoning strategies and co-ordinate tools across diverse domains is poorly understood. In this work, we conduct a large-scale evaluation of state-of-the-art LLMs on multiple tool-calling benchmarksBFCL v3, TauBench, Tau2Bench, and AceBenchand introduce MAVEN (Math & Physics Adversarial Verification & Evaluation Network), a new out of distribution (OOD) benchmark designed to stress-test multi-step reasoning through explicit verification and adversarial task composition. Our results show that most current models achieve below 50% accuracy on MAVEN, revealing a significant generalization gap across tool-use settings.
To address this, we present the CoreThink Agentic Reasoner, a framework that augments LLMs with a lightweight symbolic reasoning layer for structured decomposition and adaptive tool orchestration. Without additional training, it generalizes across all benchmarks, achieving state-of-the-art performance with 530% improvements over existing baselines at roughly one-tenth the computational cost. 

---
# Exploring Structures of Inferential Mechanisms through Simplistic Digital Circuits 

**Authors**: Giovanni Sileno, Jean-Louis Dessalles  

**Link**: [PDF](https://arxiv.org/pdf/2510.22883)  

**Abstract**: Cognitive studies and artificial intelligence have developed distinct models for various inferential mechanisms (categorization, induction, abduction, causal inference, contrast, merge, ...). Yet, both natural and artificial views on cognition lack apparently a unifying framework. This paper formulates a speculative answer attempting to respond to this gap. To postulate on higher-level activation processes from a material perspective, we consider inferential mechanisms informed by symbolic AI modelling techniques, through the simplistic lenses of electronic circuits based on logic gates. We observe that a logic gate view entails a different treatment of implication and negation compared to standard logic and logic programming. Then, by combinatorial exploration, we identify four main forms of dependencies that can be realized by these inferential circuits. Looking at how these forms are generally used in the context of logic programs, we identify eight common inferential patterns, exposing traditionally distinct inferential mechanisms in an unifying framework. Finally, following a probabilistic interpretation of logic programs, we unveil inner functional dependencies. The paper concludes elaborating in what sense, even if our arguments are mostly informed by symbolic means and digital systems infrastructures, our observations may pinpoint to more generally applicable structures. 

---
# Lyapunov Function-guided Reinforcement Learning for Flight Control 

**Authors**: Yifei Li, Erik-Jan van Kampen  

**Link**: [PDF](https://arxiv.org/pdf/2510.22840)  

**Abstract**: A cascaded online learning flight control system has been developed and enhanced with respect to action smoothness. In this paper, we investigate the convergence performance of the control system, characterized by the increment of a Lyapunov function candidate. The derivation of this metric accounts for discretization errors and state prediction errors introduced by the incremental model. Comparative results are presented through flight control simulations. 

---
# Rethinking the Text-Vision Reasoning Imbalance in MLLMs through the Lens of Training Recipes 

**Authors**: Guanyu Yao, Qiucheng Wu, Yang Zhang, Zhaowen Wang, Handong Zhao, Shiyu Chang  

**Link**: [PDF](https://arxiv.org/pdf/2510.22836)  

**Abstract**: Multimodal large language models (MLLMs) have demonstrated strong capabilities on vision-and-language tasks. However, recent findings reveal an imbalance in their reasoning capabilities across visual and textual modalities. Specifically, current MLLMs often over-rely on textual cues while under-attending to visual content, resulting in suboptimal performance on tasks that require genuine visual reasoning. We refer to this phenomenon as the \textit{modality gap}, defined as the performance disparity between text-centric and vision-centric inputs. In this paper, we analyze the modality gap through the lens of training recipes. We first show that existing training recipes tend to amplify this gap. Then, we systematically explore strategies to bridge it from two complementary perspectives: data and loss design. Our findings provide insights into developing training recipes that mitigate the modality gap and promote more balanced multimodal reasoning. Our code is publicly available at this https URL. 

---
# Toward Agents That Reason About Their Computation 

**Authors**: Adrian Orenstein, Jessica Chen, Gwyneth Anne Delos Santos, Bayley Sapara, Michael Bowling  

**Link**: [PDF](https://arxiv.org/pdf/2510.22833)  

**Abstract**: While reinforcement learning agents can achieve superhuman performance in many complex tasks, they typically do not become more computationally efficient as they improve. In contrast, humans gradually require less cognitive effort as they become more proficient at a task. If agents could reason about their compute as they learn, could they similarly reduce their computation footprint? If they could, we could have more energy efficient agents or free up compute cycles for other processes like planning. In this paper, we experiment with showing agents the cost of their computation and giving them the ability to control when they use compute. We conduct our experiments on the Arcade Learning Environment, and our results demonstrate that with the same training compute budget, agents that reason about their compute perform better on 75% of games. Furthermore, these agents use three times less compute on average. We analyze individual games and show where agents gain these efficiencies. 

---
# HRM-Agent: Training a recurrent reasoning model in dynamic environments using reinforcement learning 

**Authors**: Long H Dang, David Rawlinson  

**Link**: [PDF](https://arxiv.org/pdf/2510.22832)  

**Abstract**: The Hierarchical Reasoning Model (HRM) has impressive reasoning abilities given its small size, but has only been applied to supervised, static, fully-observable problems. One of HRM's strengths is its ability to adapt its computational effort to the difficulty of the problem. However, in its current form it cannot integrate and reuse computation from previous time-steps if the problem is dynamic, uncertain or partially observable, or be applied where the correct action is undefined, characteristics of many real-world problems.
This paper presents HRM-Agent, a variant of HRM trained using only reinforcement learning. We show that HRM can learn to navigate to goals in dynamic and uncertain maze environments. Recent work suggests that HRM's reasoning abilities stem from its recurrent inference process. We explore the dynamics of the recurrent inference process and find evidence that it is successfully reusing computation from earlier environment time-steps. 

---
# Will Humanity Be Rendered Obsolete by AI? 

**Authors**: Mohamed El Louadi, Emna Ben Romdhane  

**Link**: [PDF](https://arxiv.org/pdf/2510.22814)  

**Abstract**: This article analyzes the existential risks artificial intelligence (AI) poses to humanity, tracing the trajectory from current AI to ultraintelligence. Drawing on Irving J. Good and Nick Bostrom's theoretical work, plus recent publications (AI 2027; If Anyone Builds It, Everyone Dies), it explores AGI and superintelligence. Considering machines' exponentially growing cognitive power and hypothetical IQs, it addresses the ethical and existential implications of an intelligence vastly exceeding humanity's, fundamentally alien. Human extinction may result not from malice, but from uncontrollable, indifferent cognitive superiority. 

---
# Agentic Meta-Orchestrator for Multi-task Copilots 

**Authors**: Xiaofeng Zhu, Yunshen Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.22781)  

**Abstract**: Microsoft Copilot suites serve as the universal entry point for various agents skilled in handling important tasks, ranging from assisting a customer with product purchases to detecting vulnerabilities in corporate programming code. Each agent can be powered by language models, software engineering operations, such as database retrieval, and internal \& external knowledge. The repertoire of a copilot can expand dynamically with new agents. This requires a robust orchestrator that can distribute tasks from user prompts to the right agents. In this work, we propose an Agentic Meta-orchestrator (AMO) for handling multiple tasks and scalable agents in copilot services, which can provide both natural language and action responses. We will also demonstrate the planning that leverages meta-learning, i.e., a trained decision tree model for deciding the best inference strategy among various agents/models. We showcase the effectiveness of our AMO through two production use cases: Microsoft 365 (M365) E-Commerce Copilot and code compliance copilot. M365 E-Commerce Copilot advertises Microsoft products to external customers to promote sales success. The M365 E-Commerce Copilot provides up-to-date product information and connects to multiple agents, such as relational databases and human customer support. The code compliance copilot scans the internal DevOps code to detect known and new compliance issues in pull requests (PR). 

---
# How Do AI Agents Do Human Work? Comparing AI and Human Workflows Across Diverse Occupations 

**Authors**: Zora Zhiruo Wang, Yijia Shao, Omar Shaikh, Daniel Fried, Graham Neubig, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.22780)  

**Abstract**: AI agents are continually optimized for tasks related to human work, such as software engineering and professional writing, signaling a pressing trend with significant impacts on the human workforce. However, these agent developments have often not been grounded in a clear understanding of how humans execute work, to reveal what expertise agents possess and the roles they can play in diverse workflows. In this work, we study how agents do human work by presenting the first direct comparison of human and agent workers across multiple essential work-related skills: data analysis, engineering, computation, writing, and design. To better understand and compare heterogeneous computer-use activities of workers, we introduce a scalable toolkit to induce interpretable, structured workflows from either human or agent computer-use activities. Using such induced workflows, we compare how humans and agents perform the same tasks and find that: (1) While agents exhibit promise in their alignment to human workflows, they take an overwhelmingly programmatic approach across all work domains, even for open-ended, visually dependent tasks like design, creating a contrast with the UI-centric methods typically used by humans. (2) Agents produce work of inferior quality, yet often mask their deficiencies via data fabrication and misuse of advanced tools. (3) Nonetheless, agents deliver results 88.3% faster and cost 90.4-96.2% less than humans, highlighting the potential for enabling efficient collaboration by delegating easily programmable tasks to agents. 

---
# Jarvis: Towards Personalized AI Assistant via Personal KV-Cache Retrieval 

**Authors**: Binxiao Xu, Junyu Feng, Ruichuan An, Yulin Luo, Shilin Yan, Hao Liang, Ming Lu, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.22765)  

**Abstract**: The rapid development of Vision-language models (VLMs) enables open-ended perception and reasoning. Recent works have started to investigate how to adapt general-purpose VLMs into personalized assistants. Even commercial models such as ChatGPT now support model personalization by incorporating user-specific information. However, existing methods either learn a set of concept tokens or train a VLM to utilize user-specific information. However, both pipelines struggle to generate accurate answers as personalized assistants. We introduce Jarvis, an innovative framework for a personalized AI assistant through personal KV-Cache retrieval, which stores user-specific information in the KV-Caches of both textual and visual tokens. The textual tokens are created by summarizing user information into metadata, while the visual tokens are produced by extracting distinct image patches from the user's images. When answering a question, Jarvis first retrieves related KV-Caches from personal storage and uses them to ensure accuracy in responses. We also introduce a fine-grained benchmark built with the same distinct image patch mining pipeline, emphasizing accurate question answering based on fine-grained user-specific information. Jarvis is capable of providing more accurate responses, particularly when they depend on specific local details. Jarvis achieves state-of-the-art results in both visual question answering and text-only tasks across multiple datasets, indicating a practical path toward personalized AI assistants. The code and dataset will be released. 

---
# Multi-Modal Fact-Verification Framework for Reducing Hallucinations in Large Language Models 

**Authors**: Piyushkumar Patel  

**Link**: [PDF](https://arxiv.org/pdf/2510.22751)  

**Abstract**: While Large Language Models have transformed how we interact with AI systems, they suffer from a critical flaw: they confidently generate false information that sounds entirely plausible. This hallucination problem has become a major barrier to deploying these models in real-world applications where accuracy matters. We developed a fact verification framework that catches and corrects these errors in real-time by cross checking LLM outputs against multiple knowledge sources. Our system combines structured databases, live web searches, and academic literature to verify factual claims as they're generated. When we detect inconsistencies, we automatically correct them while preserving the natural flow of the response. Testing across various domains showed we could reduce hallucinations by 67% without sacrificing response quality. Domain experts in healthcare, finance, and scientific research rated our corrected outputs 89% satisfactory a significant improvement over unverified LLM responses. This work offers a practical solution for making LLMs more trustworthy in applications where getting facts wrong isn't an option. 

---
# Critical Insights into Leading Conversational AI Models 

**Authors**: Urja Kohli, Aditi Singh, Arun Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2510.22729)  

**Abstract**: Big Language Models (LLMs) are changing the way businesses use software, the way people live their lives and the way industries work. Companies like Google, High-Flyer, Anthropic, OpenAI and Meta are making better LLMs. So, it's crucial to look at how each model is different in terms of performance, moral behaviour and usability, as these differences are based on the different ideas that built them. This study compares five top LLMs: Google's Gemini, High-Flyer's DeepSeek, Anthropic's Claude, OpenAI's GPT models and Meta's LLaMA. It performs this by analysing three important factors: Performance and Accuracy, Ethics and Bias Mitigation and Usability and Integration. It was found that Claude has good moral reasoning, Gemini is better at multimodal capabilities and has strong ethical frameworks. DeepSeek is great at reasoning based on facts, LLaMA is good for open applications and ChatGPT delivers balanced performance with a focus on usage. It was concluded that these models are different in terms of how well they work, how easy they are to use and how they treat people ethically, making it a point that each model should be utilised by the user in a way that makes the most of its strengths. 

---
# RaCoT: Plug-and-Play Contrastive Example Generation Mechanism for Enhanced LLM Reasoning Reliability 

**Authors**: Kaitong Cai, Jusheng Zhang, Yijia Fan, Jing Yang, Keze Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.22710)  

**Abstract**: Retrieval-Augmented Generation (RAG) faces a core bottleneck with knowledge-sparse and semantically ambiguous long-tail queries, where retrieval noise distorts reasoning and necessitates costly post-processing. To tackle this, we propose RaCoT (Retrieval-aware Contrastive-of-Thought), a novel framework that shifts contrastive thinking to the pre-retrieval stage. By automatically generating a semantically adjacent yet differently answered contrastive question and extracting a $\Delta$-Prompt to capture their key differences, RaCoT guides the model to proactively focus on the ``critical details that determine answer divergence." This approach allows it to suppress semantic interference within a single retrieval pass, overcoming the theoretical bottleneck of single-vector queries that struggle to simultaneously encode signals for what to attend to and what to ignore. On six authoritative benchmarks, including PopQA and TriviaQA-unfiltered, RaCoT outperforms strong baselines like RankRAG and Self-RAG by 0.9-2.4 percentage points. It exhibits superior robustness, with a performance drop of only 8.6\% in adversarial tests, far surpassing the over 15\% degradation in other methods. Furthermore, its low latency (3.12s) and token overhead (11.54) place it on the accuracy-efficiency Pareto frontier, while ablation studies validate the necessity of each component. Ultimately, RaCoT reframes the RAG paradigm from ``post-hoc context cleaning" to ``a priori shaping of discriminative reasoning", offering an efficient and robust path toward reliable AI systems for real-time, resource-constrained deployments. 

---
# Atlas Urban Index: A VLM-Based Approach for Spatially and Temporally Calibrated Urban Development Monitoring 

**Authors**: Mithul Chander, Sai Pragnya Ranga, Prathamesh Mayekar  

**Link**: [PDF](https://arxiv.org/pdf/2510.22702)  

**Abstract**: We introduce the {\em Atlas Urban Index} (AUI), a metric for measuring urban development computed using Sentinel-2 \citep{spoto2012sentinel2} satellite imagery. Existing approaches, such as the {\em Normalized Difference Built-up Index} (NDBI), often struggle to accurately capture urban development due to factors like atmospheric noise, seasonal variation, and cloud cover. These limitations hinder large-scale monitoring of human development and urbanization. To address these challenges, we propose an approach that leverages {\em Vision-Language Models }(VLMs) to provide a development score for regions. Specifically, we collect a time series of Sentinel-2 images for each region. Then, we further process the images within fixed time windows to get an image with minimal cloud cover, which serves as the representative image for that time window. To ensure consistent scoring, we adopt two strategies: (i) providing the VLM with a curated set of reference images representing different levels of urbanization, and (ii) supplying the most recent past image to both anchor temporal consistency and mitigate cloud-related noise in the current image. Together, these components enable AUI to overcome the challenges of traditional urbanization indices and produce more reliable and stable development scores. Our qualitative experiments on Bangalore suggest that AUI outperforms standard indices such as NDBI. 

---
# Do Stop Me Now: Detecting Boilerplate Responses with a Single Iteration 

**Authors**: Yuval Kainan, Shaked Zychlinski  

**Link**: [PDF](https://arxiv.org/pdf/2510.22679)  

**Abstract**: Large Language Models (LLMs) often expend significant computational resources generating boilerplate responses, such as refusals, simple acknowledgements and casual greetings, which adds unnecessary cost and latency. To address this inefficiency, we propose a simple yet highly effective method for detecting such responses after only a single generation step. We demonstrate that the log-probability distribution of the first generated token serves as a powerful signal for classifying the nature of the entire subsequent response. Our experiments, conducted across a diverse range of small, large, and reasoning-specialized models, show that the first-token log-probability vectors form distinctly separable clusters for different response types. Using a lightweight k-NN classifier, we achieve high accuracy in predicting whether a response will be a substantive answer or a form of boilerplate response, including user-specified refusals. The primary implication is a practical, computationally trivial technique, optimizing LLM inference by enabling early termination or redirection to a smaller model, thereby yielding significant savings in computational cost. This work presents a direct path toward more efficient and sustainable LLM deployment. 

---
# SwiftSolve: A Self-Iterative, Complexity-Aware Multi-Agent Framework for Competitive Programming 

**Authors**: Adhyayan Veer Singh, Aaron Shen, Brian Law, Ahmed Ismail, Jonas Rohweder, Sean O'Brien, Kevin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.22626)  

**Abstract**: Correctness alone is insufficient: LLM-generated programs frequently satisfy unit tests while violating contest time or memory budgets. We present SwiftSolve, a complexity-aware multi-agent system for competitive programming that couples algorithmic planning with empirical profiling and complexity-guided repair. We frame competitive programming as a software environment where specialized agents act as programmers, each assuming roles such as planning, coding, profiling, and complexity analysis. A Planner proposes an algorithmic sketch; a deterministic Static Pruner filters high-risk plans; a Coder emits ISO C++17; a Profiler compiles and executes candidates on a fixed input-size schedule to record wall time and peak memory; and a Complexity Analyst fits log-log growth (s, R2) with an LLM fallback to assign a complexity class and dispatch targeted patches to either the Planner or Coder. Agents communicate via typed, versioned JSON; a controller enforces iteration caps and diminishing returns stopping. Evaluated on 26 problems (16 BigO, 10 Codeforces Div. 2) in a POSIX sandbox (2 s / 256-512 MB), SwiftSolve attains pass@1 = 61.54% (16/26) on the first attempt and Solved@<=3 = 80.77% with marginal latency change (mean 11.96 s to 12.66 s per attempt). Aggregate run-level success is 73.08% at 12.40 s mean. Failures are predominantly resource-bound, indicating inefficiency rather than logic errors. Against Claude Opus 4, SwiftSolve improves run-level success (73.1% vs 52.6%) at approximately 2x runtime overhead (12.4 s vs 6.8 s). Beyond correctness (pass@k), we report efficiency metrics (eff@k for runtime and memory, incidence of TLE or MLE, and complexity fit accuracy on BigO), demonstrating that profiling and complexity-guided replanning reduce inefficiency while preserving accuracy. 

---
# CLIN-LLM: A Safety-Constrained Hybrid Framework for Clinical Diagnosis and Treatment Generation 

**Authors**: Md. Mehedi Hasan, Rafid Mostafiz, Md. Abir Hossain, Bikash Kumar Paul  

**Link**: [PDF](https://arxiv.org/pdf/2510.22609)  

**Abstract**: Accurate symptom-to-disease classification and clinically grounded treatment recommendations remain challenging, particularly in heterogeneous patient settings with high diagnostic risk. Existing large language model (LLM)-based systems often lack medical grounding and fail to quantify uncertainty, resulting in unsafe outputs. We propose CLIN-LLM, a safety-constrained hybrid pipeline that integrates multimodal patient encoding, uncertainty-calibrated disease classification, and retrieval-augmented treatment generation. The framework fine-tunes BioBERT on 1,200 clinical cases from the Symptom2Disease dataset and incorporates Focal Loss with Monte Carlo Dropout to enable confidence-aware predictions from free-text symptoms and structured vitals. Low-certainty cases (18%) are automatically flagged for expert review, ensuring human oversight. For treatment generation, CLIN-LLM employs Biomedical Sentence-BERT to retrieve top-k relevant dialogues from the 260,000-sample MedDialog corpus. The retrieved evidence and patient context are fed into a fine-tuned FLAN-T5 model for personalized treatment generation, followed by post-processing with RxNorm for antibiotic stewardship and drug-drug interaction (DDI) screening. CLIN-LLM achieves 98% accuracy and F1 score, outperforming ClinicalBERT by 7.1% (p < 0.001), with 78% top-5 retrieval precision and a clinician-rated validity of 4.2 out of 5. Unsafe antibiotic suggestions are reduced by 67% compared to GPT-5. These results demonstrate CLIN-LLM's robustness, interpretability, and clinical safety alignment. The proposed system provides a deployable, human-in-the-loop decision support framework for resource-limited healthcare environments. Future work includes integrating imaging and lab data, multilingual extensions, and clinical trial validation. 

---
# A Framework for Quantifying How Pre-Training and Context Benefit In-Context Learning 

**Authors**: Bingqing Song, Jiaxiang Li, Rong Wang, Songtao Lu, Mingyi Hong  

**Link**: [PDF](https://arxiv.org/pdf/2510.22594)  

**Abstract**: Pre-trained large language models have demonstrated a strong ability to learn from context, known as in-context learning (ICL). Despite a surge of recent applications that leverage such capabilities, it is by no means clear, at least theoretically, how the ICL capabilities arise, and in particular, what is the precise role played by key factors such as pre-training procedure as well as context construction. In this work, we propose a new framework to analyze the ICL performance, for a class of realistic settings, which includes network architectures, data encoding, data generation, and prompt construction process. As a first step, we construct a simple example with a one-layer transformer, and show an interesting result, namely when the pre-train data distribution is different from the query task distribution, a properly constructed context can shift the output distribution towards the query task distribution, in a quantifiable manner, leading to accurate prediction on the query topic. We then extend the findings in the previous step to a more general case, and derive the precise relationship between ICL performance, context length and the KL divergence between pre-train and query task distribution. Finally, we provide experiments to validate our theoretical results. 

---
# ATOM: AdapTive and OptiMized dynamic temporal knowledge graph construction using LLMs 

**Authors**: Yassir Lairgi, Ludovic Moncla, Khalid Benabdeslem, Rémy Cazabet, Pierre Cléau  

**Link**: [PDF](https://arxiv.org/pdf/2510.22590)  

**Abstract**: In today's rapidly expanding data landscape, knowledge extraction from unstructured text is vital for real-time analytics, temporal inference, and dynamic memory frameworks. However, traditional static knowledge graph (KG) construction often overlooks the dynamic and time-sensitive nature of real-world data, limiting adaptability to continuous changes. Moreover, recent zero- or few-shot approaches that avoid domain-specific fine-tuning or reliance on prebuilt ontologies often suffer from instability across multiple runs, as well as incomplete coverage of key facts. To address these challenges, we introduce ATOM (AdapTive and OptiMized), a few-shot and scalable approach that builds and continuously updates Temporal Knowledge Graphs (TKGs) from unstructured texts. ATOM splits input documents into minimal, self-contained "atomic" facts, improving extraction exhaustivity and stability. Then, it constructs atomic TKGs from these facts while employing a dual-time modeling that distinguishes when information is observed from when it is valid. The resulting atomic TKGs are subsequently merged in parallel. Empirical evaluations demonstrate that ATOM achieves ~18% higher exhaustivity, ~17% better stability, and over 90% latency reduction compared to baseline methods, demonstrating a strong scalability potential for dynamic TKG construction. 

---
# OFFSIDE: Benchmarking Unlearning Misinformation in Multimodal Large Language Models 

**Authors**: Hao Zheng, Zirui Pang, Ling li, Zhijie Deng, Yuhan Pu, Zhaowei Zhu, Xiaobo Xia, Jiaheng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2510.22535)  

**Abstract**: Advances in Multimodal Large Language Models (MLLMs) intensify concerns about data privacy, making Machine Unlearning (MU), the selective removal of learned information, a critical necessity. However, existing MU benchmarks for MLLMs are limited by a lack of image diversity, potential inaccuracies, and insufficient evaluation scenarios, which fail to capture the complexity of real-world applications. To facilitate the development of MLLMs unlearning and alleviate the aforementioned limitations, we introduce OFFSIDE, a novel benchmark for evaluating misinformation unlearning in MLLMs based on football transfer rumors. This manually curated dataset contains 15.68K records for 80 players, providing a comprehensive framework with four test sets to assess forgetting efficacy, generalization, utility, and robustness. OFFSIDE supports advanced settings like selective unlearning and corrective relearning, and crucially, unimodal unlearning (forgetting only text data). Our extensive evaluation of multiple baselines reveals key findings: (1) Unimodal methods (erasing text-based knowledge) fail on multimodal rumors; (2) Unlearning efficacy is largely driven by catastrophic forgetting; (3) All methods struggle with "visual rumors" (rumors appear in the image); (4) The unlearned rumors can be easily recovered and (5) All methods are vulnerable to prompt attacks. These results expose significant vulnerabilities in current approaches, highlighting the need for more robust multimodal unlearning solutions. The code is available at \href{this https URL}{this https URL}. 

---
# Learning "Partner-Aware" Collaborators in Multi-Party Collaboration 

**Authors**: Abhijnan Nath, Nikhil Krishnaswamy  

**Link**: [PDF](https://arxiv.org/pdf/2510.22462)  

**Abstract**: Large Language Models (LLMs) are increasingly bring deployed in agentic settings where they act as collaborators with humans. Therefore, it is increasingly important to be able to evaluate their abilities to collaborate effectively in multi-turn, multi-party tasks. In this paper, we build on the AI alignment and safe interruptability literature to offer novel theoretical insights on collaborative behavior between LLM-driven collaborator agents and an intervention agent. Our goal is to learn an ideal partner-aware collaborator that increases the group's common-ground (CG)-alignment on task-relevant propositions-by intelligently collecting information provided in interventions by a partner this http URL show how LLM agents trained using standard RLHF and related approaches are naturally inclined to ignore possibly well-meaning interventions, which makes increasing group common ground non-trivial in this setting. We employ a two-player Modified-Action MDP to examine this suboptimal behavior of standard AI agents, and propose Interruptible Collaborative Roleplayer (ICR)-a novel partner-aware learning algorithm to train CG-optimal collaborators. Experiments on multiple collaborative task environments show that ICR, on average, is more capable of promoting successful CG convergence and exploring more diverse solutions in such tasks. 

---
# Modeling Hierarchical Thinking in Large Reasoning Models 

**Authors**: G M Shahariar, Ali Nazari, Erfan Shayegani, Nael Abu-Ghazaleh  

**Link**: [PDF](https://arxiv.org/pdf/2510.22437)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable reasoning abilities when they generate step-by-step solutions, known as chain-of-thought (CoT) reasoning. When trained to using chain-of-thought reasoning examples, the resulting models (called Large Reasoning Models, or LRMs) appear to learn hierarchical thinking strategies similar to those used by humans. However, understanding LRMs emerging reasoning capabilities remains a difficult open problem, with many potential important applications including improving training and understanding robustness. In this paper, we adopt a memoryless Finite State Machine formulation to approximate LRM's emerging hierarchical reasoning dynamics as a structured, interpretable abstraction. We identify a small set of discrete reasoning states including - initialization, deduction, augmentation-strategy, uncertainty-estimation, backtracking, and final-conclusion that capture the high-level states present in the model's reasoning process. By annotating each step of a model's CoT with these states, we can represent the reasoning trajectory as a transition sequence through the state graph. This FSM formulation provides a systematic way to analyze, interpret and visualize how different models approach problems. We describe the FSM model, provide examples of CoT annotations under this scheme, and discuss how it can shed light on differences between available models in their approach to reasoning. Our results demonstrate that this FSM-based analysis reveals distinct reasoning patterns and potential shortcomings, offering a new lens to evaluate and improve LLM reasoning. 

---
# Reasoning Models Reason Well, Until They Don't 

**Authors**: Revanth Rameshkumar, Jimson Huang, Yunxin Sun, Fei Xia, Abulhair Saparov  

**Link**: [PDF](https://arxiv.org/pdf/2510.22371)  

**Abstract**: Large language models (LLMs) have shown significant progress in reasoning tasks. However, recent studies show that transformers and LLMs fail catastrophically once reasoning problems exceed modest complexity. We revisit these findings through the lens of large reasoning models (LRMs) -- LLMs fine-tuned with incentives for step-by-step argumentation and self-verification. LRM performance on graph and reasoning benchmarks such as NLGraph seem extraordinary, with some even claiming they are capable of generalized reasoning and innovation in reasoning-intensive fields such as mathematics, physics, medicine, and law. However, by more carefully scaling the complexity of reasoning problems, we show existing benchmarks actually have limited complexity. We develop a new dataset, the Deep Reasoning Dataset (DeepRD), along with a generative process for producing unlimited examples of scalable complexity. We use this dataset to evaluate model performance on graph connectivity and natural language proof planning. We find that the performance of LRMs drop abruptly at sufficient complexity and do not generalize. We also relate our LRM results to the distributions of the complexities of large, real-world knowledge graphs, interaction graphs, and proof datasets. We find the majority of real-world examples fall inside the LRMs' success regime, yet the long tails expose substantial failure potential. Our analysis highlights the near-term utility of LRMs while underscoring the need for new methods that generalize beyond the complexity of examples in the training distribution. 

---
# DynaSolidGeo: A Dynamic Benchmark for Genuine Spatial Mathematical Reasoning of VLMs in Solid Geometry 

**Authors**: Changti Wu, Shijie Lian, Zihao Liu, Lei Zhang, Laurence Tianruo Yang, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.22340)  

**Abstract**: Solid geometry problem solving demands spatial mathematical reasoning that integrates spatial intelligence and symbolic reasoning. However, most existing multimodal mathematical reasoning benchmarks focus primarily on 2D plane geometry, rely on static datasets prone to data contamination and memorization, and evaluate models solely by final answers, overlooking the reasoning process. To address these limitations, we introduce DynaSolidGeo, the first dynamic benchmark for evaluating genuine spatial reasoning in Vision-Language Models (VLMs). Constructed through a semi-automatic annotation pipeline, DynaSolidGeo contains 503 expert-curated seed questions that can, in principle, dynamically generate an unbounded number of diverse multimodal text-visual instances. Beyond answer accuracy, we incorporate process evaluation based on expert-annotated reasoning chains to measure logical validity and causal coherence. Experiments across representative open-source and closed-source VLMs reveal large performance gaps, severe degradation in dynamic settings, and poor performance on tasks requiring high-level spatial intelligence, such as mental rotation and visualization. The code and dataset are available at \href{this https URL}{DynaSolidGeo}. 

---
# LIFT: Interpretable truck driving risk prediction with literature-informed fine-tuned LLMs 

**Authors**: Xiao Hu, Yuansheng Lian, Ke Zhang, Yunxuan Li, Yuelong Su, Meng Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.22333)  

**Abstract**: This study proposes an interpretable prediction framework with literature-informed fine-tuned (LIFT) LLMs for truck driving risk prediction. The framework integrates an LLM-driven Inference Core that predicts and explains truck driving risk, a Literature Processing Pipeline that filters and summarizes domain-specific literature into a literature knowledge base, and a Result Evaluator that evaluates the prediction performance as well as the interpretability of the LIFT LLM. After fine-tuning on a real-world truck driving risk dataset, the LIFT LLM achieved accurate risk prediction, outperforming benchmark models by 26.7% in recall and 10.1% in F1-score. Furthermore, guided by the literature knowledge base automatically constructed from 299 domain papers, the LIFT LLM produced variable importance ranking consistent with that derived from the benchmark model, while demonstrating robustness in interpretation results to various data sampling conditions. The LIFT LLM also identified potential risky scenarios by detecting key combination of variables in truck driving risk, which were verified by PERMANOVA tests. Finally, we demonstrated the contribution of the literature knowledge base and the fine-tuning process in the interpretability of the LIFT LLM, and discussed the potential of the LIFT LLM in data-driven knowledge discovery. 

---
# Graph-Coarsening Approach for the Capacitated Vehicle Routing Problem with Time Windows 

**Authors**: Mustafa Mert Özyılmaz  

**Link**: [PDF](https://arxiv.org/pdf/2510.22329)  

**Abstract**: The Capacitated Vehicle Routing Problem with Time Windows (CVRPTW) is a fundamental NP-hard optimization problem in logistics. Solving large-scale instances remains computationally challenging for exact solvers. This work introduces a multilevel graph coarsening and refinement framework that aggregates customers into meta-nodes using a spatio-temporal distance metric. The reduced problem is solved with classical heuristics and subsequently expanded back into the original space with feasibility corrections. Preliminary experiments on Solomon benchmark instances show that the proposed method reduces computation time while preserving or improving solution quality, particularly with respect to capacity and time window constraints. The paper also explores the integration of quantum-inspired optimization techniques, highlighting their potential to further accelerate large-scale vehicle routing tasks. 

---
# VietLyrics: A Large-Scale Dataset and Models for Vietnamese Automatic Lyrics Transcription 

**Authors**: Quoc Anh Nguyen, Bernard Cheng, Kelvin Soh  

**Link**: [PDF](https://arxiv.org/pdf/2510.22295)  

**Abstract**: Automatic Lyrics Transcription (ALT) for Vietnamese music presents unique challenges due to its tonal complexity and dialectal variations, but remains largely unexplored due to the lack of a dedicated dataset. Therefore, we curated the first large-scale Vietnamese ALT dataset (VietLyrics), comprising 647 hours of songs with line-level aligned lyrics and metadata to address these issues. Our evaluation of current ASRbased approaches reveal significant limitations, including frequent transcription errors and hallucinations in non-vocal segments. To improve performance, we fine-tuned Whisper models on the VietLyrics dataset, achieving superior results compared to existing multilingual ALT systems, including LyricWhiz. We publicly release VietLyrics and our models, aiming to advance Vietnamese music computing research while demonstrating the potential of this approach for ALT in low-resource language and music. 

---
# PACR: Progressively Ascending Confidence Reward for LLM Reasoning 

**Authors**: Eunseop Yoon, Hee Suk Yoon, Jaehyun Jang, SooHwan Eom, Qi Dai, Chong Luo, Mark A. Hasegawa-Johnson, Chang D. Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2510.22255)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has significantly improved LLM reasoning, but its sparse, outcome-based reward provides no guidance for intermediate steps, slowing exploration. We propose Progressively Ascending Confidence Reward (PACR), a dense, model-intrinsic reward computed directly from the model's evolving belief in the correct answer. PACR encodes the inductive bias that, along a well-formed reasoning trajectory, the probability of the ground-truth answer should have a generally ascending trend. We provide empirical and theoretical analysis validating that such an inductive bias constrains the exploration search space to regions richer in logically sound reasoning. We demonstrate that PACR accelerates exploration, reaches reward saturation with fewer trajectories, and yields improvements on multiple benchmarks. Our results suggest that dense, model-intrinsic shaping signals can make RLVR training more effective and reliable. 

---
# OptiTree: Hierarchical Thoughts Generation with Tree Search for LLM Optimization Modeling 

**Authors**: Haoyang Liu, Jie Wang, Yuyang Cai, Xiongwei Han, Yufei Kuang, Jianye Hao  

**Link**: [PDF](https://arxiv.org/pdf/2510.22192)  

**Abstract**: Optimization modeling is one of the most crucial but technical parts of operations research (OR). To automate the modeling process, existing works have leveraged large language models (LLMs), prompting them to break down tasks into steps for generating variables, constraints, and objectives. However, due to the highly complex mathematical structures inherent in OR problems, standard fixed-step decomposition often fails to achieve high performance. To address this challenge, we introduce OptiTree, a novel tree search approach designed to enhance modeling capabilities for complex problems through adaptive problem decomposition into simpler subproblems. Specifically, we develop a modeling tree that organizes a wide range of OR problems based on their hierarchical problem taxonomy and complexity, with each node representing a problem category and containing relevant high-level modeling thoughts. Given a problem to model, we recurrently search the tree to identify a series of simpler subproblems and synthesize the global modeling thoughts by adaptively integrating the hierarchical thoughts. Experiments show that OptiTree significantly improves the modeling accuracy compared to the state-of-the-art, achieving over 10\% improvements on the challenging benchmarks. The code is released at this https URL. 

---
# Dopamine-driven synaptic credit assignment in neural networks 

**Authors**: Saranraj Nambusubramaniyan, Shervin Safavi, Raja Guru, Andreas Knoblauch  

**Link**: [PDF](https://arxiv.org/pdf/2510.22178)  

**Abstract**: Solving the synaptic Credit Assignment Problem(CAP) is central to learning in both biological and artificial neural systems. Finding an optimal solution for synaptic CAP means setting the synaptic weights that assign credit to each neuron for influencing the final output and behavior of neural networks or animals. Gradient-based methods solve this problem in artificial neural networks using back-propagation, however, not in the most efficient way. For instance, back-propagation requires a chain of top-down gradient computations. This leads to an expensive optimization process in terms of computing power and memory linked with well-known weight transport and update locking problems. To address these shortcomings, we take a NeuroAI approach and draw inspiration from neural Reinforcement Learning to develop a derivative-free optimizer for training neural networks, Dopamine. Dopamine is developed for Weight Perturbation (WP) learning that exploits stochastic updating of weights towards optima. It achieves this by minimizing the regret, a form of Reward Prediction Error (RPE) between the expected outcome from the perturbed model and the actual outcome from the unperturbed model. We use this RPE to adjust the learning rate in the network (i.e., creating an adaptive learning rate strategy, similar to the role of dopamine in the brain). We tested the Dopamine optimizer for training multi-layered perceptrons for XOR tasks, and recurrent neural networks for chaotic time series forecasting. Dopamine-trained models demonstrate accelerated convergence and outperform standard WP, and give comparable performance to gradient-based algorithms, while consuming significantly less computation and memory. Overall, the Dopamine optimizer not only finds robust solutions and comparable performance to the state-of-the-art Machine Learning optimizers but is also neurobiologically more plausible. 

---
# Measure what Matters: Psychometric Evaluation of AI with Situational Judgment Tests 

**Authors**: Alexandra Yost, Shreyans Jain, Shivam Raval, Grant Corser, Allen Roush, Nina Xu, Jacqueline Hammack, Ravid Shwartz-Ziv, Amirali Abdullah  

**Link**: [PDF](https://arxiv.org/pdf/2510.22170)  

**Abstract**: AI psychometrics evaluates AI systems in roles that traditionally require emotional judgment and ethical consideration. Prior work often reuses human trait inventories (Big Five, \hexaco) or ad hoc personas, limiting behavioral realism and domain relevance. We propose a framework that (1) uses situational judgment tests (SJTs) from realistic scenarios to probe domain-specific competencies; (2) integrates industrial-organizational and personality psychology to design sophisticated personas which include behavioral and psychological descriptors, life history, and social and emotional functions; and (3) employs structured generation with population demographic priors and memoir inspired narratives, encoded with Pydantic schemas. In a law enforcement assistant case study, we construct a rich dataset of personas drawn across 8 persona archetypes and SJTs across 11 attributes, and analyze behaviors across subpopulation and scenario slices. The dataset spans 8,500 personas, 4,000 SJTs, and 300,000 responses. We will release the dataset and all code to the public. 

---
# Controllable Mathematical Reasoning via Self-Optimizing Thought Vectors 

**Authors**: Xuying LI  

**Link**: [PDF](https://arxiv.org/pdf/2510.22132)  

**Abstract**: We present a novel approach for controllable mathematical reasoning that leverages self-optimizing thought vectors with entropy minimization. Our method introduces learnable thought vectors that dynamically modulate the internal reasoning process of large language models. Using Gemma-2-9B on GSM8K, we achieve 90.1% accuracy with a controllability score of 0.42, demonstrating that entropy-based rewards effectively guide focused reasoning patterns without requiring external reward annotations. Our analysis reveals distinct thought vector clusters and consistent low-entropy distributions across control conditions, validating our framework for controllable AI reasoning. 

---
# Embracing Trustworthy Brain-Agent Collaboration as Paradigm Extension for Intelligent Assistive Technologies 

**Authors**: Yankai Chen, Xinni Zhang, Yifei Zhang, Yangning Li, Henry Peng Zou, Chunyu Miao, Weizhi Zhang, Xue Liu, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.22095)  

**Abstract**: Brain-Computer Interfaces (BCIs) offer a direct communication pathway between the human brain and external devices, holding significant promise for individuals with severe neurological impairments. However, their widespread adoption is hindered by critical limitations, such as low information transfer rates and extensive user-specific calibration. To overcome these challenges, recent research has explored the integration of Large Language Models (LLMs), extending the focus from simple command decoding to understanding complex cognitive states. Despite these advancements, deploying agentic AI faces technical hurdles and ethical concerns. Due to the lack of comprehensive discussion on this emerging direction, this position paper argues that the field is poised for a paradigm extension from BCI to Brain-Agent Collaboration (BAC). We emphasize reframing agents as active and collaborative partners for intelligent assistance rather than passive brain signal data processors, demanding a focus on ethical data handling, model reliability, and a robust human-agent collaboration framework to ensure these systems are safe, trustworthy, and effective. 

---
# Energy-Efficient Domain-Specific Artificial Intelligence Models and Agents: Pathways and Paradigms 

**Authors**: Abhijit Chatterjee, Niraj K. Jha, Jonathan D. Cohen, Thomas L. Griffiths, Hongjing Lu, Diana Marculescu, Ashiqur Rasul, Keshab K. Parhi  

**Link**: [PDF](https://arxiv.org/pdf/2510.22052)  

**Abstract**: The field of artificial intelligence (AI) has taken a tight hold on broad aspects of society, industry, business, and governance in ways that dictate the prosperity and might of the world's economies. The AI market size is projected to grow from 189 billion USD in 2023 to 4.8 trillion USD by 2033. Currently, AI is dominated by large language models that exhibit linguistic and visual intelligence. However, training these models requires a massive amount of data scraped from the web as well as large amounts of energy (50--60 GWh to train GPT-4). Despite these costs, these models often hallucinate, a characteristic that prevents them from being deployed in critical application domains. In contrast, the human brain consumes only 20~W of power. What is needed is the next level of AI evolution in which lightweight domain-specific multimodal models with higher levels of intelligence can reason, plan, and make decisions in dynamic environments with real-time data and prior knowledge, while learning continuously and evolving in ways that enhance future decision-making capability. This will define the next wave of AI, progressing from today's large models, trained with vast amounts of data, to nimble energy-efficient domain-specific agents that can reason and think in a world full of uncertainty. To support such agents, hardware will need to be reimagined to allow energy efficiencies greater than 1000x over the state of the art. Such a vision of future AI systems is developed in this work. 

---
# Towards Error-Centric Intelligence II: Energy-Structured Causal Models 

**Authors**: Marcus Thomas  

**Link**: [PDF](https://arxiv.org/pdf/2510.22050)  

**Abstract**: Contemporary machine learning optimizes for predictive accuracy, yet systems that achieve state of the art performance remain causally opaque: their internal representations provide no principled handle for intervention. We can retrain such models, but we cannot surgically edit specific mechanisms while holding others fixed, because learned latent variables lack causal semantics. We argue for a conceptual reorientation: intelligence is the ability to build and refine explanations, falsifiable claims about manipulable structure that specify what changes and what remains invariant under intervention. Explanations subsume prediction but demand more: causal commitments that can be independently tested and corrected at the level of mechanisms. We introduce computational explanations, mappings from observations to intervention ready causal accounts. We instantiate these explanations with Energy Structured Causal Models (ESCMs), in which mechanisms are expressed as constraints (energy functions or vector fields) rather than explicit input output maps, and interventions act by local surgery on those constraints. This shift makes internal structure manipulable at the level where explanations live: which relations must hold, which can change, and what follows when they do. We provide concrete instantiations of the structural-causal principles LAP and ICM in the ESCM context, and also argue that empirical risk minimization systematically produces fractured, entangled representations, a failure we analyze as gauge ambiguity in encoder energy pairs. Finally, we show that under mild conditions, ESCMs recover standard SCM semantics. Building on Part I's principles (LAP, ICM, CAP) and its definition of intelligence as explanation-building under criticism, this paper offers a formal language for causal reasoning in systems that aspire to understand, not merely to predict. 

---
# HW/SW Co-design of a PCM/PWM converter: a System Level Approach based in the SpecC Methodology 

**Authors**: Daniel G. P. Petrini, Braz Izaias da Silva Junior  

**Link**: [PDF](https://arxiv.org/pdf/2510.22046)  

**Abstract**: We present a case study applying the SpecC methodology within a system-level hardware/software co-design flow to a PCM-to-PWM converter, the core of a Class-D audio amplifier. The converter was modeled and explored with SpecC methodology to derive an HW/SW partition. Using system-level estimates and fast functional simulation, we evaluated mappings that meet real-time constraints while reducing estimated cost of an all-hardware solution and avoiding the expense of a purely software implementation on a high-end processor. Despite the design's moderate complexity, the results underline the value of system-level co-design for early architectural insight, rapid validation, and actionable cost/performance trade-offs. [Original work from 2005; formatting revised in 2025, with no changes to the results.] 

---
# Predictive Coding Enhances Meta-RL To Achieve Interpretable Bayes-Optimal Belief Representation Under Partial Observability 

**Authors**: Po-Chen Kuo, Han Hou, Will Dabney, Edgar Y. Walker  

**Link**: [PDF](https://arxiv.org/pdf/2510.22039)  

**Abstract**: Learning a compact representation of history is critical for planning and generalization in partially observable environments. While meta-reinforcement learning (RL) agents can attain near Bayes-optimal policies, they often fail to learn the compact, interpretable Bayes-optimal belief states. This representational inefficiency potentially limits the agent's adaptability and generalization capacity. Inspired by predictive coding in neuroscience--which suggests that the brain predicts sensory inputs as a neural implementation of Bayesian inference--and by auxiliary predictive objectives in deep RL, we investigate whether integrating self-supervised predictive coding modules into meta-RL can facilitate learning of Bayes-optimal representations. Through state machine simulation, we show that meta-RL with predictive modules consistently generates more interpretable representations that better approximate Bayes-optimal belief states compared to conventional meta-RL across a wide variety of tasks, even when both achieve optimal policies. In challenging tasks requiring active information seeking, only meta-RL with predictive modules successfully learns optimal representations and policies, whereas conventional meta-RL struggles with inadequate representation learning. Finally, we demonstrate that better representation learning leads to improved generalization. Our results strongly suggest the role of predictive learning as a guiding principle for effective representation learning in agents navigating partial observability. 

---
# LLM-AR: LLM-powered Automated Reasoning Framework 

**Authors**: Rick Chen, Joseph Ternasky, Aaron Ontoyin Yin, Xianling Mu, Fuat Alican, Yigit Ihlamur  

**Link**: [PDF](https://arxiv.org/pdf/2510.22034)  

**Abstract**: Large language models (LLMs) can already identify patterns and reason effectively, yet their variable accuracy hampers adoption in high-stakes decision-making applications. In this paper, we study this issue from a venture capital perspective by predicting idea-stage startup success based on founder traits. (i) To build a reliable prediction model, we introduce LLM-AR, a pipeline inspired by neural-symbolic systems that distils LLM-generated heuristics into probabilistic rules executed by the ProbLog automated-reasoning engine. (ii) An iterative policy-evolution loop incorporates association-rule mining to progressively refine the prediction rules.
On unseen folds, LLM-AR achieves 59.5% precision and 8.7% recall, 5.9x the random baseline precision, while exposing every decision path for human inspection. The framework is interpretable and tunable via hyperparameters, showing promise to extend into other domains. 

---
# LightAgent: Mobile Agentic Foundation Models 

**Authors**: Yangqin Jiang, Chao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.22009)  

**Abstract**: With the advancement of multimodal large language models (MLLMs), building GUI agent systems has become an increasingly promising direction-especially for mobile platforms, given their rich app ecosystems and intuitive touch interactions. Yet mobile GUI agents face a critical dilemma: truly on-device models (4B or smaller) lack sufficient performance, while capable models (starting from 7B) are either too large for mobile deployment or prohibitively costly (e.g., cloud-only closed-source MLLMs). To resolve this, we propose LightAgent, a mobile agentic foundation model solution that leverages device-cloud collaboration to tap the cost-efficiency of on-device models and the high capability of cloud models, while avoiding their drawbacks. Specifically, LightAgent enhances Qwen2.5-VL-3B via two-stage SFT->GRPO training on synthetic GUI data for strong decision-making, integrates an efficient long-reasoning mechanism to utilize historical interactions under tight resources, and defaults to on-device execution-only escalating challenging subtasks to the cloud via real-time complexity assessment. Experiments on the online AndroidLab benchmark and diverse apps show LightAgent matches or nears larger models, with a significant reduction in cloud costs. 

---
# Foundation of Intelligence: Review of Math Word Problems from Human Cognition Perspective 

**Authors**: Zhenya Huang, Jiayu Liu, Xin Lin, Zhiyuan Ma, Shangzi Xue, Tong Xiao, Qi Liu, Yee Whye Teh, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.21999)  

**Abstract**: Math word problem (MWP) serves as a fundamental research topic in artificial intelligence (AI) dating back to 1960s. This research aims to advance the reasoning abilities of AI by mirroring the human-like cognitive intelligence. The mainstream technological paradigm has evolved from the early rule-based methods, to deep learning models, and is rapidly advancing towards large language models. However, the field still lacks a systematic taxonomy for the MWP survey along with a discussion of current development trends. Therefore, in this paper, we aim to comprehensively review related research in MWP solving through the lens of human cognition, to demonstrate how recent AI models are advancing in simulating human cognitive abilities. Specifically, we summarize 5 crucial cognitive abilities for MWP solving, including Problem Understanding, Logical Organization, Associative Memory, Critical Thinking, and Knowledge Learning. Focused on these abilities, we review two mainstream MWP models in recent 10 years: neural network solvers, and LLM based solvers, and discuss the core human-like abilities they demonstrated in their intricate problem-solving process. Moreover, we rerun all the representative MWP solvers and supplement their performance on 5 mainstream benchmarks for a unified comparison. To the best of our knowledge, this survey first comprehensively analyzes the influential MWP research of the past decade from the perspective of human reasoning cognition and provides an integrative overall comparison across existing approaches. We hope it can inspire further research in AI reasoning. Our repository is released on this https URL. 

---
# Distribution Shift Alignment Helps LLMs Simulate Survey Response Distributions 

**Authors**: Ji Huang, Mengfei Li, Shuai Shao  

**Link**: [PDF](https://arxiv.org/pdf/2510.21977)  

**Abstract**: Large language models (LLMs) offer a promising way to simulate human survey responses, potentially reducing the cost of large-scale data collection. However, existing zero-shot methods suffer from prompt sensitivity and low accuracy, while conventional fine-tuning approaches mostly fit the training set distributions and struggle to produce results more accurate than the training set itself, which deviates from the original goal of using LLMs to simulate survey responses. Building on this observation, we introduce Distribution Shift Alignment (DSA), a two-stage fine-tuning method that aligns both the output distributions and the distribution shifts across different backgrounds. By learning how these distributions change rather than fitting training data, DSA can provide results substantially closer to the true distribution than the training data. Empirically, DSA consistently outperforms other methods on five public survey datasets. We further conduct a comprehensive comparison covering accuracy, robustness, and data savings. DSA reduces the required real data by 53.48-69.12%, demonstrating its effectiveness and efficiency in survey simulation. 

---
# Performance Trade-offs of Optimizing Small Language Models for E-Commerce 

**Authors**: Josip Tomo Licardo, Nikola Tankovic  

**Link**: [PDF](https://arxiv.org/pdf/2510.21970)  

**Abstract**: Large Language Models (LLMs) offer state-of-the-art performance in natural language understanding and generation tasks. However, the deployment of leading commercial models for specialized tasks, such as e-commerce, is often hindered by high computational costs, latency, and operational expenses. This paper investigates the viability of smaller, open-weight models as a resource-efficient alternative. We present a methodology for optimizing a one-billion-parameter Llama 3.2 model for multilingual e-commerce intent recognition. The model was fine-tuned using Quantized Low-Rank Adaptation (QLoRA) on a synthetically generated dataset designed to mimic real-world user queries. Subsequently, we applied post-training quantization techniques, creating GPU-optimized (GPTQ) and CPU-optimized (GGUF) versions. Our results demonstrate that the specialized 1B model achieves 99% accuracy, matching the performance of the significantly larger GPT-4.1 model. A detailed performance analysis revealed critical, hardware-dependent trade-offs: while 4-bit GPTQ reduced VRAM usage by 41%, it paradoxically slowed inference by 82% on an older GPU architecture (NVIDIA T4) due to dequantization overhead. Conversely, GGUF formats on a CPU achieved a speedup of up to 18x in inference throughput and a reduction of over 90% in RAM consumption compared to the FP16 baseline. We conclude that small, properly optimized open-weight models are not just a viable but a more suitable alternative for domain-specific applications, offering state-of-the-art accuracy at a fraction of the computational cost. 

---
# Computational Hardness of Reinforcement Learning with Partial $q^π$-Realizability 

**Authors**: Shayan Karimi, Xiaoqi Tan  

**Link**: [PDF](https://arxiv.org/pdf/2510.21888)  

**Abstract**: This paper investigates the computational complexity of reinforcement learning in a novel linear function approximation regime, termed partial $q^{\pi}$-realizability. In this framework, the objective is to learn an $\epsilon$-optimal policy with respect to a predefined policy set $\Pi$, under the assumption that all value functions for policies in $\Pi$ are linearly realizable. The assumptions of this framework are weaker than those in $q^{\pi}$-realizability but stronger than those in $q^*$-realizability, providing a practical model where function approximation naturally arises. We prove that learning an $\epsilon$-optimal policy in this setting is computationally hard. Specifically, we establish NP-hardness under a parameterized greedy policy set (argmax) and show that - unless NP = RP - an exponential lower bound (in feature vector dimension) holds when the policy set contains softmax policies, under the Randomized Exponential Time Hypothesis. Our hardness results mirror those in $q^*$-realizability and suggest computational difficulty persists even when $\Pi$ is expanded beyond the optimal policy. To establish this, we reduce from two complexity problems, $\delta$-Max-3SAT and $\delta$-Max-3SAT(b), to instances of GLinear-$\kappa$-RL (greedy policy) and SLinear-$\kappa$-RL (softmax policy). Our findings indicate that positive computational results are generally unattainable in partial $q^{\pi}$-realizability, in contrast to $q^{\pi}$-realizability under a generative access model. 

---
# Exploration through Generation: Applying GFlowNets to Structured Search 

**Authors**: Mark Phillip Matovic  

**Link**: [PDF](https://arxiv.org/pdf/2510.21886)  

**Abstract**: This work applies Generative Flow Networks (GFlowNets) to three graph optimization problems: the Traveling Salesperson Problem, Minimum Spanning Tree, and Shortest Path. GFlowNets are generative models that learn to sample solutions proportionally to a reward function. The models are trained using the Trajectory Balance loss to build solutions sequentially, se- lecting edges for spanning trees, nodes for paths, and cities for tours. Experiments on benchmark instances of varying sizes show that GFlowNets learn to find optimal solutions. For each problem type, multiple graph configurations with different numbers of nodes were tested. The generated solutions match those from classical algorithms (Dijkstra for shortest path, Kruskal for spanning trees, and exact solvers for TSP). Training convergence depends on problem complexity, with the number of episodes required for loss stabilization increasing as graph size grows. Once training converges, the generated solutions match known optima from classical algorithms across the tested instances. This work demonstrates that generative models can solve combinatorial optimization problems through learned policies. The main advantage of this learning-based approach is computational scalability: while classical algorithms have fixed complexity per instance, GFlowNets amortize computation through training. With sufficient computational resources, the framework could potentially scale to larger problem instances where classical exact methods become infeasible. 

---
# GeoThought: A Dataset for Enhancing Mathematical Geometry Reasoning in Vision-Language Models 

**Authors**: Nannan Shi, Chuanyu Qin, Shipeng Song, Man Luo  

**Link**: [PDF](https://arxiv.org/pdf/2510.21881)  

**Abstract**: Large language models (LLMs) have demonstrated strong reasoning capabilities in text-based mathematical problem solving; however, when adapted to visual reasoning tasks, particularly geometric problem solving, their performance substantially declines because geometric problems present unique challenges. Specifically, these challenges stem from two key factors: first, the intrinsic complexity of geometry requiring detailed image comprehension and multi-step reasoning, and second, the limitations of existing datasets which lack sufficient scale, diversity, and explicit reasoning traces, consequently hindering effective model training. To address these challenges, we developed the GeoThoughts dataset, a comprehensive geometric reasoning corpus with two subsets: Geo-Thought-6K with 6,243 samples and its augmented version Geo-Thought-Augmented-10K containing 10,834 samples. Each entry includes visual descriptions, step-by-step solutions, explicit reasoning chains, reflection steps, and final answers. Using this dataset, we developed GeoThought-MLLM, a mathematical reasoning multimodal model that generates detailed thinking processes during problem-solving. Our model outperforms existing benchmarks in geometric tasks, demonstrating that training with our Chain-of-Thought dataset improves geometric reasoning capabilities across both in-domain and out-of-domain settings. Finally, we analyze failure cases and observe that errors primarily arise from incorrect interpretation of mathematical concepts or spatial misjudgment. By invoking CoT to correct these mistakes, the model produces correct answers. 

---
# Capability Ceilings in Autoregressive Language Models: Empirical Evidence from Knowledge-Intensive Tasks 

**Authors**: Javier Marín  

**Link**: [PDF](https://arxiv.org/pdf/2510.21866)  

**Abstract**: We document empirical capability ceilings in decoder-only autoregressive language models across knowledge-intensive tasks. Systematic evaluation of OPT and Pythia model families (70M-30B parameters, spanning 240 times scaling) reveals that knowledge retrieval tasks show negligible accuracy improvement despite smooth loss reduction. On MMLU mathematics benchmarks, accuracy remains flat at 19-20% (below 25% random chance) across all scales while cross-entropy loss decreases by 31%. In contrast, procedural tasks like arithmetic show conventional scaling where both metrics improve together. Attention intervention experiments reveal high sensitivity to perturbation: swapping attention patterns between models causes catastrophic performance collapse (complete accuracy loss) rather than graceful degradation. These measurements have immediate engineering implications: for knowledge-intensive applications using OPT and Pythia architectures, parameter scaling beyond 1-2B offers minimal accuracy gains despite continued loss improvement. Our findings quantify capability-specific scaling failures in these model families to inform resource allocation decisions. Whether these patterns reflect fundamental constraints of decoder-only architectures or implementation-specific limitations remains an open question requiring investigation across diverse architectural approaches. 

---
# SIGN: Schema-Induced Games for Naming 

**Authors**: Ryan Zhang, Herbert Woisetscläger  

**Link**: [PDF](https://arxiv.org/pdf/2510.21855)  

**Abstract**: Real-world AI systems are tackling increasingly complex problems, often through interactions among large language model (LLM) agents. When these agents develop inconsistent conventions, coordination can break down. Applications such as collaborative coding and distributed planning therefore require reliable, consistent communication, and scalability is a central concern as systems grow. We introduce Schema-Induced Games for Naming (SIGN), a naming game that examines how lightweight structure can steer convention formation. We compare schema-induced communication to unconstrained natural language and find faster convergence with up to 5.8x higher agreement. These results suggest that minimal structure can act as a simple control knob for efficient multi-agent coordination, pointing toward broader applications beyond the naming game. 

---
# PREFINE: Personalized Story Generation via Simulated User Critics and User-Specific Rubric Generation 

**Authors**: Kentaro Ueda, Takehiro Takayanagi  

**Link**: [PDF](https://arxiv.org/pdf/2510.21721)  

**Abstract**: While recent advances in Large Language Models (LLMs) have improved the quality of creative text generation, significant challenges remain in producing personalized stories that reflect individual user preferences. Conventional approaches rely on explicit feedback or fine-tuning, which presents practical issues regarding user burden, data collection, computational costs, and privacy. In this work, we propose PREFINE (Persona-and-Rubric Guided Critique-and-Refine), a novel framework that extends the Critique-and-Refine paradigm to personalization. PREFINE constructs a pseudo-user agent from a user's interaction history and generates user-specific rubrics (evaluation criteria). By having this agent critique and refine outputs on the user's behalf based on these tailored rubrics, our method achieves personalized generation without requiring parameter updates or direct user feedback. We conducted a comprehensive evaluation on the PerDOC and PerMPST story datasets. We designed three baseline methods and several model variants to verify the contribution of each component of our framework. In automatic evaluations (LLM-as-a-Judge), PREFINE achieved higher win rates and statistically significant scores than the baselines, without compromising general story quality. Analysis of the model variants confirmed that both the pseudo-user agent and the user-specific rubrics are crucial for enhancing personalization performance. Beyond story generation, our approach holds potential for enabling efficient personalization in broader applications, such as dialogue systems, education, and recommendation. 

---
# A Multi-Component AI Framework for Computational Psychology: From Robust Predictive Modeling to Deployed Generative Dialogue 

**Authors**: Anant Pareek  

**Link**: [PDF](https://arxiv.org/pdf/2510.21720)  

**Abstract**: The confluence of Artificial Intelligence and Computational Psychology presents an opportunity to model, understand, and interact with complex human psychological states through computational means. This paper presents a comprehensive, multi-faceted framework designed to bridge the gap between isolated predictive modeling and an interactive system for psychological analysis. The methodology encompasses a rigorous, end-to-end development lifecycle. First, foundational performance benchmarks were established on four diverse psychological datasets using classical machine learning techniques. Second, state-of-the-art transformer models were fine-tuned, a process that necessitated the development of effective solutions to overcome critical engineering challenges, including the resolution of numerical instability in regression tasks and the creation of a systematic workflow for conducting large-scale training under severe resource constraints. Third, a generative large language model (LLM) was fine-tuned using parameter-efficient techniques to function as an interactive "Personality Brain." Finally, the entire suite of predictive and generative models was architected and deployed as a robust, scalable microservices ecosystem. Key findings include the successful stabilization of transformer-based regression models for affective computing, showing meaningful predictive performance where standard approaches failed, and the development of a replicable methodology for democratizing large-scale AI research. The significance of this work lies in its holistic approach, demonstrating a complete research-to-deployment pipeline that integrates predictive analysis with generative dialogue, thereby providing a practical model for future research in computational psychology and human-AI interaction. 

---
# Variational Masked Diffusion Models 

**Authors**: Yichi Zhang, Alex Schwing, Zhizhen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.23606)  

**Abstract**: Masked diffusion models have recently emerged as a flexible framework for discrete generative modeling. However, a key limitation of standard masked diffusion is its inability to effectively capture dependencies among tokens that are predicted concurrently, leading to degraded generation quality when dependencies among tokens are important. To explicitly model dependencies among tokens, we propose Variational Masked Diffusion (VMD), a framework that introduces latent variables into the masked diffusion process. Through controlled experiments on synthetic datasets, we demonstrate that VMD successfully learns dependencies that conventional masked diffusion fails to capture. We further validate the effectiveness of our approach on Sudoku puzzles and text datasets, where learning of dependencies among tokens improves global consistency. Across these domains, VMD enhances both generation quality and dependency awareness, highlighting the value of integrating variational inference into masked diffusion. Our code is available at: this https URL. 

---
# Track, Inpaint, Resplat: Subject-driven 3D and 4D Generation with Progressive Texture Infilling 

**Authors**: Shuhong Zheng, Ashkan Mirzaei, Igor Gilitschenski  

**Link**: [PDF](https://arxiv.org/pdf/2510.23605)  

**Abstract**: Current 3D/4D generation methods are usually optimized for photorealism, efficiency, and aesthetics. However, they often fail to preserve the semantic identity of the subject across different viewpoints. Adapting generation methods with one or few images of a specific subject (also known as Personalization or Subject-driven generation) allows generating visual content that align with the identity of the subject. However, personalized 3D/4D generation is still largely underexplored. In this work, we introduce TIRE (Track, Inpaint, REsplat), a novel method for subject-driven 3D/4D generation. It takes an initial 3D asset produced by an existing 3D generative model as input and uses video tracking to identify the regions that need to be modified. Then, we adopt a subject-driven 2D inpainting model for progressively infilling the identified regions. Finally, we resplat the modified 2D multi-view observations back to 3D while still maintaining consistency. Extensive experiments demonstrate that our approach significantly improves identity preservation in 3D/4D generation compared to state-of-the-art methods. Our project website is available at this https URL. 

---
# A Survey of Data Agents: Emerging Paradigm or Overstated Hype? 

**Authors**: Yizhang Zhu, Liangwei Wang, Chenyu Yang, Xiaotian Lin, Boyan Li, Wei Zhou, Xinyu Liu, Zhangyang Peng, Tianqi Luo, Yu Li, Chengliang Chai, Chong Chen, Shimin Di, Ju Fan, Ji Sun, Nan Tang, Fugee Tsung, Jiannan Wang, Chenglin Wu, Yanwei Xu, Shaolei Zhang, Yong Zhang, Xuanhe Zhou, Guoliang Li, Yuyu Luo  

**Link**: [PDF](https://arxiv.org/pdf/2510.23587)  

**Abstract**: The rapid advancement of large language models (LLMs) has spurred the emergence of data agents--autonomous systems designed to orchestrate Data + AI ecosystems for tackling complex data-related tasks. However, the term "data agent" currently suffers from terminological ambiguity and inconsistent adoption, conflating simple query responders with sophisticated autonomous architectures. This terminological ambiguity fosters mismatched user expectations, accountability challenges, and barriers to industry growth. Inspired by the SAE J3016 standard for driving automation, this survey introduces the first systematic hierarchical taxonomy for data agents, comprising six levels that delineate and trace progressive shifts in autonomy, from manual operations (L0) to a vision of generative, fully autonomous data agents (L5), thereby clarifying capability boundaries and responsibility allocation. Through this lens, we offer a structured review of existing research arranged by increasing autonomy, encompassing specialized data agents for data management, preparation, and analysis, alongside emerging efforts toward versatile, comprehensive systems with enhanced autonomy. We further analyze critical evolutionary leaps and technical gaps for advancing data agents, especially the ongoing L2-to-L3 transition, where data agents evolve from procedural execution to autonomous orchestration. Finally, we conclude with a forward-looking roadmap, envisioning the advent of proactive, generative data agents. 

---
# Hope Speech Detection in Social Media English Corpora: Performance of Traditional and Transformer Models 

**Authors**: Luis Ramos, Hiram Calvo, Olga Kolesnikova  

**Link**: [PDF](https://arxiv.org/pdf/2510.23585)  

**Abstract**: The identification of hope speech has become a promised NLP task, considering the need to detect motivational expressions of agency and goal-directed behaviour on social media platforms. This proposal evaluates traditional machine learning models and fine-tuned transformers for a previously split hope speech dataset as train, development and test set. On development test, a linear-kernel SVM and logistic regression both reached a macro-F1 of 0.78; SVM with RBF kernel reached 0.77, and Naïve Bayes hit 0.75. Transformer models delivered better results, the best model achieved weighted precision of 0.82, weighted recall of 0.80, weighted F1 of 0.79, macro F1 of 0.79, and 0.80 accuracy. These results suggest that while optimally configured traditional machine learning models remain agile, transformer architectures detect some subtle semantics of hope to achieve higher precision and recall in hope speech detection, suggesting that larges transformers and LLMs could perform better in small datasets. 

---
# TAMI: Taming Heterogeneity in Temporal Interactions for Temporal Graph Link Prediction 

**Authors**: Zhongyi Yu, Jianqiu Wu, Zhenghao Wu, Shuhan Zhong, Weifeng Su, Chul-Ho Lee, Weipeng Zhuo  

**Link**: [PDF](https://arxiv.org/pdf/2510.23577)  

**Abstract**: Temporal graph link prediction aims to predict future interactions between nodes in a graph based on their historical interactions, which are encoded in node embeddings. We observe that heterogeneity naturally appears in temporal interactions, e.g., a few node pairs can make most interaction events, and interaction events happen at varying intervals. This leads to the problems of ineffective temporal information encoding and forgetting of past interactions for a pair of nodes that interact intermittently for their link prediction. Existing methods, however, do not consider such heterogeneity in their learning process, and thus their learned temporal node embeddings are less effective, especially when predicting the links for infrequently interacting node pairs. To cope with the heterogeneity, we propose a novel framework called TAMI, which contains two effective components, namely log time encoding function (LTE) and link history aggregation (LHA). LTE better encodes the temporal information through transforming interaction intervals into more balanced ones, and LHA prevents the historical interactions for each target node pair from being forgotten. State-of-the-art temporal graph neural networks can be seamlessly and readily integrated into TAMI to improve their effectiveness. Experiment results on 13 classic datasets and three newest temporal graph benchmark (TGB) datasets show that TAMI consistently improves the link prediction performance of the underlying models in both transductive and inductive settings. Our code is available at this https URL. 

---
# UrbanVLA: A Vision-Language-Action Model for Urban Micromobility 

**Authors**: Anqi Li, Zhiyong Wang, Jiazhao Zhang, Minghan Li, Yunpeng Qi, Zhibo Chen, Zhizheng Zhang, He Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.23576)  

**Abstract**: Urban micromobility applications, such as delivery robots, demand reliable navigation across large-scale urban environments while following long-horizon route instructions. This task is particularly challenging due to the dynamic and unstructured nature of real-world city areas, yet most existing navigation methods remain tailored to short-scale and controllable scenarios. Effective urban micromobility requires two complementary levels of navigation skills: low-level capabilities such as point-goal reaching and obstacle avoidance, and high-level capabilities, such as route-visual alignment. To this end, we propose UrbanVLA, a route-conditioned Vision-Language-Action (VLA) framework designed for scalable urban navigation. Our method explicitly aligns noisy route waypoints with visual observations during execution, and subsequently plans trajectories to drive the robot. To enable UrbanVLA to master both levels of navigation, we employ a two-stage training pipeline. The process begins with Supervised Fine-Tuning (SFT) using simulated environments and trajectories parsed from web videos. This is followed by Reinforcement Fine-Tuning (RFT) on a mixture of simulation and real-world data, which enhances the model's safety and adaptability in real-world settings. Experiments demonstrate that UrbanVLA surpasses strong baselines by more than 55% in the SocialNav task on MetaUrban. Furthermore, UrbanVLA achieves reliable real-world navigation, showcasing both scalability to large-scale urban environments and robustness against real-world uncertainties. 

---
# RobotArena $\infty$: Scalable Robot Benchmarking via Real-to-Sim Translation 

**Authors**: Yash Jangir, Yidi Zhang, Kashu Yamazaki, Chenyu Zhang, Kuan-Hsun Tu, Tsung-Wei Ke, Lei Ke, Yonatan Bisk, Katerina Fragkiadaki  

**Link**: [PDF](https://arxiv.org/pdf/2510.23571)  

**Abstract**: The pursuit of robot generalists - instructable agents capable of performing diverse tasks across diverse environments - demands rigorous and scalable evaluation. Yet real-world testing of robot policies remains fundamentally constrained: it is labor-intensive, slow, unsafe at scale, and difficult to reproduce. Existing simulation benchmarks are similarly limited, as they train and test policies within the same synthetic domains and cannot assess models trained from real-world demonstrations or alternative simulation environments. As policies expand in scope and complexity, these barriers only intensify, since defining "success" in robotics often hinges on nuanced human judgments of execution quality. In this paper, we introduce a new benchmarking framework that overcomes these challenges by shifting VLA evaluation into large-scale simulated environments augmented with online human feedback. Leveraging advances in vision-language models, 2D-to-3D generative modeling, and differentiable rendering, our approach automatically converts video demonstrations from widely used robot datasets into simulated counterparts. Within these digital twins, we assess VLA policies using both automated VLM-guided scoring and scalable human preference judgments collected from crowdworkers, transforming human involvement from tedious scene setup, resetting, and safety supervision into lightweight preference comparisons. To measure robustness, we systematically perturb simulated environments along multiple axes, such as textures and object placements, stress-testing policy generalization under controlled variation. The result is a continuously evolving, reproducible, and scalable benchmark for real-world trained robot manipulation policies, addressing a critical missing capability in today's robotics landscape. 

---
# Learning Linearity in Audio Consistency Autoencoders via Implicit Regularization 

**Authors**: Bernardo Torres, Manuel Moussallam, Gabriel Meseguer-Brocal  

**Link**: [PDF](https://arxiv.org/pdf/2510.23530)  

**Abstract**: Audio autoencoders learn useful, compressed audio representations, but their non-linear latent spaces prevent intuitive algebraic manipulation such as mixing or scaling. We introduce a simple training methodology to induce linearity in a high-compression Consistency Autoencoder (CAE) by using data augmentation, thereby inducing homogeneity (equivariance to scalar gain) and additivity (the decoder preserves addition) without altering the model's architecture or loss function. When trained with our method, the CAE exhibits linear behavior in both the encoder and decoder while preserving reconstruction fidelity. We test the practical utility of our learned space on music source composition and separation via simple latent arithmetic. This work presents a straightforward technique for constructing structured latent spaces, enabling more intuitive and efficient audio processing. 

---
# A Deep Latent Factor Graph Clustering with Fairness-Utility Trade-off Perspective 

**Authors**: Siamak Ghodsi, Amjad Seyedi, Tai Le Quy, Fariba Karimi, Eirini Ntoutsi  

**Link**: [PDF](https://arxiv.org/pdf/2510.23507)  

**Abstract**: Fair graph clustering seeks partitions that respect network structure while maintaining proportional representation across sensitive groups, with applications spanning community detection, team formation, resource allocation, and social network analysis. Many existing approaches enforce rigid constraints or rely on multi-stage pipelines (e.g., spectral embedding followed by $k$-means), limiting trade-off control, interpretability, and scalability. We introduce \emph{DFNMF}, an end-to-end deep nonnegative tri-factorization tailored to graphs that directly optimizes cluster assignments with a soft statistical-parity regularizer. A single parameter $\lambda$ tunes the fairness--utility balance, while nonnegativity yields parts-based factors and transparent soft memberships. The optimization uses sparse-friendly alternating updates and scales near-linearly with the number of edges. Across synthetic and real networks, DFNMF achieves substantially higher group balance at comparable modularity, often dominating state-of-the-art baselines on the Pareto front. The code is available at this https URL. 

---
# Mixed Precision Training of Neural ODEs 

**Authors**: Elena Celledoni, Brynjulf Owren, Lars Ruthotto, Tianjiao Nicole Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.23498)  

**Abstract**: Exploiting low-precision computations has become a standard strategy in deep learning to address the growing computational costs imposed by ever larger models and datasets. However, naively performing all computations in low precision can lead to roundoff errors and instabilities. Therefore, mixed precision training schemes usually store the weights in high precision and use low-precision computations only for whitelisted operations. Despite their success, these principles are currently not reliable for training continuous-time architectures such as neural ordinary differential equations (Neural ODEs). This paper presents a mixed precision training framework for neural ODEs, combining explicit ODE solvers with a custom backpropagation scheme, and demonstrates its effectiveness across a range of learning tasks. Our scheme uses low-precision computations for evaluating the velocity, parameterized by the neural network, and for storing intermediate states, while stability is provided by a custom dynamic adjoint scaling and by accumulating the solution and gradients in higher precision. These contributions address two key challenges in training neural ODE: the computational cost of repeated network evaluations and the growth of memory requirements with the number of time steps or layers. Along with the paper, we publish our extendable, open-source PyTorch package rampde, whose syntax resembles that of leading packages to provide a drop-in replacement in existing codes. We demonstrate the reliability and effectiveness of our scheme using challenging test cases and on neural ODE applications in image classification and generative models, achieving approximately 50% memory reduction and up to 2x speedup while maintaining accuracy comparable to single-precision training. 

---
# On the Faithfulness of Visual Thinking: Measurement and Enhancement 

**Authors**: Zujing Liu, Junwen Pan, Qi She, Yuan Gao, Guisong Xia  

**Link**: [PDF](https://arxiv.org/pdf/2510.23482)  

**Abstract**: Recent large vision-language models (LVLMs) can generate vision-text multimodal chain-of-thought (MCoT) traces after reinforcement fine-tuning (RFT). However, we observe that the visual information incorporated in MCoT is often inaccurate, though still yield correct answers, indicating a lack of faithfulness in the MCoT reasoning process. We attribute this unfaithfulness to the RL reward in RFT, which solely incentivizes the format of interleaved vision-text cues, ie, it encourages the model to incorporate visual information into its text reasoning steps without considering the correctness of the visual information. In this paper, we first probe the faithfulness of MCoT by measuring how much the prediction changes when its visual and textual thoughts are intervened. Surprisingly, the model's predictions remain nearly unchanged under visual intervention but change significantly under textual intervention, indicating that the visual evidence is largely ignored. To further analyze visual information, we introduce an automated LVLM-based evaluation metric that quantifies the faithfulness of visual cues from two perspectives: reliability and sufficiency. Our evaluation reveals that the visual information in current MCoT traces is simultaneously unreliable and insufficient. To address this issue, we propose a novel MCoT learning strategy termed Sufficient-Component Cause Model (SCCM) learning. This approach encourages the MCoT to generate sufficient yet minimal visual components that are independently capable of leading to correct answers. We note that the proposed SCCM is annotation-free and compatible with various RFT for MCoT in a plug-and-play manner. Empirical results demonstrate that SCCM consistently improves the visual faithfulness across a suite of fine-grained perception and reasoning benchmarks. Code is available at this https URL. 

---
# BBOPlace-Bench: Benchmarking Black-Box Optimization for Chip Placement 

**Authors**: Ke Xue, Ruo-Tong Chen, Rong-Xi Tan, Xi Lin, Yunqi Shi, Siyuan Xu, Mingxuan Yuan, Chao Qian  

**Link**: [PDF](https://arxiv.org/pdf/2510.23472)  

**Abstract**: Chip placement is a vital stage in modern chip design as it has a substantial impact on the subsequent processes and the overall quality of the final chip. The use of black-box optimization (BBO) for chip placement has a history of several decades. However, early efforts were limited by immature problem formulations and inefficient algorithm designs. Recent progress has shown the effectiveness and efficiency of BBO for chip placement, proving its potential to achieve state-of-the-art results. Despite these advancements, the field lacks a unified, BBO-specific benchmark for thoroughly assessing various problem formulations and BBO algorithms. To fill this gap, we propose BBOPlace-Bench, the first benchmark designed specifically for evaluating and developing BBO algorithms for chip placement tasks. It integrates three problem formulations of BBO for chip placement, and offers a modular, decoupled, and flexible framework that enables users to seamlessly implement, test, and compare their own algorithms. BBOPlace-Bench integrates a wide variety of existing BBO algorithms, including simulated annealing (SA), evolutionary algorithms (EAs), and Bayesian optimization (BO). Experimental results show that the problem formulations of mask-guided optimization and hyperparameter optimization exhibit superior performance than the sequence pair problem formulation, while EAs demonstrate better overall performance than SA and BO, especially in high-dimensional search spaces, and also achieve state-of-the-art performance compared to the mainstream chip placement methods. BBOPlace-Bench not only facilitates the development of efficient BBO-driven solutions for chip placement but also broadens the practical application scenarios (which are urgently needed) for the BBO community. The code of BBOPlace-Bench is available at this https URL. 

---
# Robust Decision Making with Partially Calibrated Forecasts 

**Authors**: Shayan Kiyani, Hamed Hassani, George Pappas, Aaron Roth  

**Link**: [PDF](https://arxiv.org/pdf/2510.23471)  

**Abstract**: Calibration has emerged as a foundational goal in ``trustworthy machine learning'', in part because of its strong decision theoretic semantics. Independent of the underlying distribution, and independent of the decision maker's utility function, calibration promises that amongst all policies mapping predictions to actions, the uniformly best policy is the one that ``trusts the predictions'' and acts as if they were correct. But this is true only of \emph{fully calibrated} forecasts, which are tractable to guarantee only for very low dimensional prediction problems. For higher dimensional prediction problems (e.g. when outcomes are multiclass), weaker forms of calibration have been studied that lack these decision theoretic properties. In this paper we study how a conservative decision maker should map predictions endowed with these weaker (``partial'') calibration guarantees to actions, in a way that is robust in a minimax sense: i.e. to maximize their expected utility in the worst case over distributions consistent with the calibration guarantees. We characterize their minimax optimal decision rule via a duality argument, and show that surprisingly, ``trusting the predictions and acting accordingly'' is recovered in this minimax sense by \emph{decision calibration} (and any strictly stronger notion of calibration), a substantially weaker and more tractable condition than full calibration. For calibration guarantees that fall short of decision calibration, the minimax optimal decision rule is still efficiently computable, and we provide an empirical evaluation of a natural one that applies to any regression model solved to optimize squared error. 

---
# Evaluating Large Language Models for Stance Detection on Financial Targets from SEC Filing Reports and Earnings Call Transcripts 

**Authors**: Nikesh Gyawali, Doina Caragea, Alex Vasenkov, Cornelia Caragea  

**Link**: [PDF](https://arxiv.org/pdf/2510.23464)  

**Abstract**: Financial narratives from U.S. Securities and Exchange Commission (SEC) filing reports and quarterly earnings call transcripts (ECTs) are very important for investors, auditors, and regulators. However, their length, financial jargon, and nuanced language make fine-grained analysis difficult. Prior sentiment analysis in the financial domain required a large, expensive labeled dataset, making the sentence-level stance towards specific financial targets challenging. In this work, we introduce a sentence-level corpus for stance detection focused on three core financial metrics: debt, earnings per share (EPS), and sales. The sentences were extracted from Form 10-K annual reports and ECTs, and labeled for stance (positive, negative, neutral) using the advanced ChatGPT-o3-pro model under rigorous human validation. Using this corpus, we conduct a systematic evaluation of modern large language models (LLMs) using zero-shot, few-shot, and Chain-of-Thought (CoT) prompting strategies. Our results show that few-shot with CoT prompting performs best compared to supervised baselines, and LLMs' performance varies across the SEC and ECT datasets. Our findings highlight the practical viability of leveraging LLMs for target-specific stance in the financial domain without requiring extensive labeled data. 

---
# BrowseConf: Confidence-Guided Test-Time Scaling for Web Agents 

**Authors**: Litu Ou, Kuan Li, Huifeng Yin, Liwen Zhang, Zhongwang Zhang, Xixi Wu, Rui Ye, Zile Qiao, Yong Jiang, Pengjun Xie, Fei Huang, Jingren Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.23458)  

**Abstract**: Confidence in LLMs is a useful indicator of model uncertainty and answer reliability. Existing work mainly focused on single-turn scenarios, while research on confidence in complex multi-turn interactions is limited. In this paper, we investigate whether LLM-based search agents have the ability to communicate their own confidence through verbalized confidence scores after long sequences of actions, a significantly more challenging task compared to outputting confidence in a single interaction. Experimenting on open-source agentic models, we first find that models exhibit much higher task accuracy at high confidence while having near-zero accuracy when confidence is low. Based on this observation, we propose Test-Time Scaling (TTS) methods that use confidence scores to determine answer quality, encourage the model to try again until reaching a satisfactory confidence level. Results show that our proposed methods significantly reduce token consumption while demonstrating competitive performance compared to baseline fixed budget TTS methods. 

---
# Omni-Reward: Towards Generalist Omni-Modal Reward Modeling with Free-Form Preferences 

**Authors**: Zhuoran Jin, Hongbang Yuan, Kejian Zhu, Jiachun Li, Pengfei Cao, Yubo Chen, Kang Liu, Jun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.23451)  

**Abstract**: Reward models (RMs) play a critical role in aligning AI behaviors with human preferences, yet they face two fundamental challenges: (1) Modality Imbalance, where most RMs are mainly focused on text and image modalities, offering limited support for video, audio, and other modalities; and (2) Preference Rigidity, where training on fixed binary preference pairs fails to capture the complexity and diversity of personalized preferences. To address the above challenges, we propose Omni-Reward, a step toward generalist omni-modal reward modeling with support for free-form preferences, consisting of: (1) Evaluation: We introduce Omni-RewardBench, the first omni-modal RM benchmark with free-form preferences, covering nine tasks across five modalities including text, image, video, audio, and 3D; (2) Data: We construct Omni-RewardData, a multimodal preference dataset comprising 248K general preference pairs and 69K instruction-tuning pairs for training generalist omni-modal RMs; (3) Model: We propose Omni-RewardModel, which includes both discriminative and generative RMs, and achieves strong performance on Omni-RewardBench as well as other widely used reward modeling benchmarks. 

---
# FRBNet: Revisiting Low-Light Vision through Frequency-Domain Radial Basis Network 

**Authors**: Fangtong Sun, Congyu Li, Ke Yang, Yuchen Pan, Hanwen Yu, Xichuan Zhang, Yiying Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.23444)  

**Abstract**: Low-light vision remains a fundamental challenge in computer vision due to severe illumination degradation, which significantly affects the performance of downstream tasks such as detection and segmentation. While recent state-of-the-art methods have improved performance through invariant feature learning modules, they still fall short due to incomplete modeling of low-light conditions. Therefore, we revisit low-light image formation and extend the classical Lambertian model to better characterize low-light conditions. By shifting our analysis to the frequency domain, we theoretically prove that the frequency-domain channel ratio can be leveraged to extract illumination-invariant features via a structured filtering process. We then propose a novel and end-to-end trainable module named \textbf{F}requency-domain \textbf{R}adial \textbf{B}asis \textbf{Net}work (\textbf{FRBNet}), which integrates the frequency-domain channel ratio operation with a learnable frequency domain filter for the overall illumination-invariant feature enhancement. As a plug-and-play module, FRBNet can be integrated into existing networks for low-light downstream tasks without modifying loss functions. Extensive experiments across various downstream tasks demonstrate that FRBNet achieves superior performance, including +2.2 mAP for dark object detection and +2.9 mIoU for nighttime segmentation. Code is available at: this https URL. 

---
# Exploring Vulnerability in AI Industry 

**Authors**: Claudio Pirrone, Stefano Fricano, Gioacchino Fazio  

**Link**: [PDF](https://arxiv.org/pdf/2510.23421)  

**Abstract**: The rapid ascent of Foundation Models (FMs), enabled by the Transformer architecture, drives the current AI ecosystem. Characterized by large-scale training and downstream adaptability, FMs (as GPT family) have achieved massive public adoption, fueling a turbulent market shaped by platform economics and intense investment. Assessing the vulnerability of this fast-evolving industry is critical yet challenging due to data limitations. This paper proposes a synthetic AI Vulnerability Index (AIVI) focusing on the upstream value chain for FM production, prioritizing publicly available data. We model FM output as a function of five inputs: Compute, Data, Talent, Capital, and Energy, hypothesizing that supply vulnerability in any input threatens the industry. Key vulnerabilities include compute concentration, data scarcity and legal risks, talent bottlenecks, capital intensity and strategic dependencies, as well as escalating energy demands. Acknowledging imperfect input substitutability, we propose a weighted geometrical average of aggregate subindexes, normalized using theoretical or empirical benchmarks. Despite limitations and room for improvement, this preliminary index aims to quantify systemic risks in AI's core production engine, and implicitly shed a light on the risks for downstream value chain. 

---
# Eigen-Value: Efficient Domain-Robust Data Valuation via Eigenvalue-Based Approach 

**Authors**: Youngjun Choi, Joonseong Kang, Sungjun Lim, Kyungwoo Song  

**Link**: [PDF](https://arxiv.org/pdf/2510.23409)  

**Abstract**: Data valuation has become central in the era of data-centric AI. It drives efficient training pipelines and enables objective pricing in data markets by assigning a numeric value to each data point. Most existing data valuation methods estimate the effect of removing individual data points by evaluating changes in model validation performance under in-distribution (ID) settings, as opposed to out-of-distribution (OOD) scenarios where data follow different patterns. Since ID and OOD data behave differently, data valuation methods based on ID loss often fail to generalize to OOD settings, particularly when the validation set contains no OOD data. Furthermore, although OOD-aware methods exist, they involve heavy computational costs, which hinder practical deployment. To address these challenges, we introduce \emph{Eigen-Value} (EV), a plug-and-play data valuation framework for OOD robustness that uses only an ID data subset, including during validation. EV provides a new spectral approximation of domain discrepancy, which is the gap of loss between ID and OOD using ratios of eigenvalues of ID data's covariance matrix. EV then estimates the marginal contribution of each data point to this discrepancy via perturbation theory, alleviating the computational burden. Subsequently, EV plugs into ID loss-based methods by adding an EV term without any additional training loop. We demonstrate that EV achieves improved OOD robustness and stable value rankings across real-world datasets, while remaining computationally lightweight. These results indicate that EV is practical for large-scale settings with domain shift, offering an efficient path to OOD-robust data valuation. 

---
# EMTSF:Extraordinary Mixture of SOTA Models for Time Series Forecasting 

**Authors**: Musleh Alharthi, Kaleel Mahmood, Sarosh Patel, Ausif Mahmood  

**Link**: [PDF](https://arxiv.org/pdf/2510.23396)  

**Abstract**: The immense success of the Transformer architecture
in Natural Language Processing has led to its adoption in Time Se ries Forecasting (TSF), where superior performance has been shown.
However, a recent important paper questioned their effectiveness by
demonstrating that a simple single layer linear model outperforms
Transformer-based models. This was soon shown to be not as valid,
by a better transformer-based model termed PatchTST. More re cently, TimeLLM demonstrated even better results by repurposing a
Large Language Model (LLM) for the TSF domain. Again, a follow
up paper challenged this by demonstrating that removing the LLM
component or replacing it with a basic attention layer in fact yields
better performance. One of the challenges in forecasting is the fact
that TSF data favors the more recent past, and is sometimes subject
to unpredictable events. Based upon these recent insights in TSF, we
propose a strong Mixture of Experts (MoE) framework. Our method
combines the state-of-the-art (SOTA) models including xLSTM, en hanced Linear, PatchTST, and minGRU, among others. This set of
complimentary and diverse models for TSF are integrated in a Trans former based MoE gating network. Our proposed model outperforms
all existing TSF models on standard benchmarks, surpassing even the
latest approaches based on MoE frameworks. 

---
# Detecting Religious Language in Climate Discourse 

**Authors**: Evy Beijen, Pien Pieterse, Yusuf Çelik, Willem Th. van Peursen, Sandjai Bhulai, Meike Morren  

**Link**: [PDF](https://arxiv.org/pdf/2510.23395)  

**Abstract**: Religious language continues to permeate contemporary discourse, even in ostensibly secular domains such as environmental activism and climate change debates. This paper investigates how explicit and implicit forms of religious language appear in climate-related texts produced by secular and religious nongovernmental organizations (NGOs). We introduce a dual methodological approach: a rule-based model using a hierarchical tree of religious terms derived from ecotheology literature, and large language models (LLMs) operating in a zero-shot setting. Using a dataset of more than 880,000 sentences, we compare how these methods detect religious language and analyze points of agreement and divergence. The results show that the rule-based method consistently labels more sentences as religious than LLMs. These findings highlight not only the methodological challenges of computationally detecting religious language but also the broader tension over whether religious language should be defined by vocabulary alone or by contextual meaning. This study contributes to digital methods in religious studies by demonstrating both the potential and the limitations of approaches for analyzing how the sacred persists in climate discourse. 

---
# Symbolic Neural Generation with Applications to Lead Discovery in Drug Design 

**Authors**: Ashwin Srinivasan, A Baskar, Tirtharaj Dash, Michael Bain, Sanjay Kumar Dey, Mainak Banerjee  

**Link**: [PDF](https://arxiv.org/pdf/2510.23379)  

**Abstract**: We investigate a relatively underexplored class of hybrid neurosymbolic models integrating symbolic learning with neural reasoning to construct data generators meeting formal correctness criteria. In \textit{Symbolic Neural Generators} (SNGs), symbolic learners examine logical specifications of feasible data from a small set of instances -- sometimes just one. Each specification in turn constrains the conditional information supplied to a neural-based generator, which rejects any instance violating the symbolic specification. Like other neurosymbolic approaches, SNG exploits the complementary strengths of symbolic and neural methods. The outcome of an SNG is a triple $(H, X, W)$, where $H$ is a symbolic description of feasible instances constructed from data, $X$ a set of generated new instances that satisfy the description, and $W$ an associated weight. We introduce a semantics for such systems, based on the construction of appropriate \textit{base} and \textit{fibre} partially-ordered sets combined into an overall partial order, and outline a probabilistic extension relevant to practical applications. In this extension, SNGs result from searching over a weighted partial ordering. We implement an SNG combining a restricted form of Inductive Logic Programming (ILP) with a large language model (LLM) and evaluate it on early-stage drug design. Our main interest is the description and the set of potential inhibitor molecules generated by the SNG. On benchmark problems -- where drug targets are well understood -- SNG performance is statistically comparable to state-of-the-art methods. On exploratory problems with poorly understood targets, generated molecules exhibit binding affinities on par with leading clinical candidates. Experts further find the symbolic specifications useful as preliminary filters, with several generated molecules identified as viable for synthesis and wet-lab testing. 

---
# ZeroFlood: A Geospatial Foundation Model for Data-Efficient Flood Susceptibility Mapping 

**Authors**: Hyeongkyun Kim, Orestis Oikonomou  

**Link**: [PDF](https://arxiv.org/pdf/2510.23364)  

**Abstract**: Flood susceptibility mapping (FSM) is vital for disaster prevention but remains challenging in data-scarce regions where hydrodynamic models require dense geophysical inputs. This work introduces ZeroFlood, a geospatial foundation model framework for data-efficient FSM. The approach fine-tunes Geospatial Foundation Models (GFMs) with Thinking-in-Modality (TiM) reasoning, enabling flood prediction from basic Earth observation data such as Sentinel-1 or Sentinel-2 imagery. Using paired EO and simulated flood maps from data-rich regions, ZeroFlood bridges data availability gaps through cross-modal representation learning. Experiments with TerraMind and Prithvi GFMs show that TiM enhances model robustness, with the TerraMind-Large configuration achieving an F1 score of 67.21. The results demonstrate the feasibility of foundation-model-based FSM as a scalable and data-efficient solution for flood risk management. 

---
# Multitask Multimodal Self-Supervised Learning for Medical Images 

**Authors**: Cristian Simionescu  

**Link**: [PDF](https://arxiv.org/pdf/2510.23325)  

**Abstract**: This thesis works to address a pivotal challenge in medical image analysis: the reliance on extensive labeled datasets, which are often limited due to the need for expert annotation and constrained by privacy and legal issues. By focusing on the development of self-supervised learning techniques and domain adaptation methods, this research aims to circumvent these limitations, presenting a novel approach to enhance the utility and efficacy of deep learning in medical imaging.
Central to this thesis is the development of the Medformer, an innovative neural network architecture designed for multitask learning and deep domain adaptation. This model is adept at pre-training on diverse medical image datasets, handling varying sizes and modalities, and is equipped with a dynamic input-output adaptation mechanism. This enables efficient processing and integration of a wide range of medical image types, from 2D X-rays to complex 3D MRIs, thus mitigating the dependency on large labeled datasets.
Further, the thesis explores the current state of self-supervised learning in medical imaging. It introduces novel pretext tasks that are capable of extracting meaningful information from unlabeled data, significantly advancing the model's interpretative abilities. This approach is validated through rigorous experimentation, including the use of the MedMNIST dataset, demonstrating the model's proficiency in learning generalized features applicable to various downstream tasks.
In summary, this thesis contributes to the advancement of medical image analysis by offering a scalable, adaptable framework that reduces reliance on labeled data. It paves the way for more accurate, efficient diagnostic tools in healthcare, signifying a major step forward in the application of deep learning in medical imaging. 

---
# Arabic Little STT: Arabic Children Speech Recognition Dataset 

**Authors**: Mouhand Alkadri, Dania Desouki, Khloud Al Jallad  

**Link**: [PDF](https://arxiv.org/pdf/2510.23319)  

**Abstract**: The performance of Artificial Intelligence (AI) systems fundamentally depends on high-quality training data. However, low-resource languages like Arabic suffer from severe data scarcity. Moreover, the absence of child-specific speech corpora is an essential gap that poses significant challenges. To address this gap, we present our created dataset, Arabic Little STT, a dataset of Levantine Arabic child speech recorded in classrooms, containing 355 utterances from 288 children (ages 6 - 13). We further conduct a systematic assessment of Whisper, a state-of-the-art automatic speech recognition (ASR) model, on this dataset and compare its performance with adult Arabic benchmarks. Our evaluation across eight Whisper variants reveals that even the best-performing model (Large_v3) struggles significantly, achieving a 0.66 word error rate (WER) on child speech, starkly contrasting with its sub 0.20 WER on adult datasets. These results align with other research on English speech. Results highlight the critical need for dedicated child speech benchmarks and inclusive training data in ASR development. Emphasizing that such data must be governed by strict ethical and privacy frameworks to protect sensitive child information. We hope that this study provides an initial step for future work on equitable speech technologies for Arabic-speaking children. We hope that our publicly available dataset enrich the children's demographic representation in ASR datasets. 

---
# ReconViaGen: Towards Accurate Multi-view 3D Object Reconstruction via Generation 

**Authors**: Jiahao Chang, Chongjie Ye, Yushuang Wu, Yuantao Chen, Yidan Zhang, Zhongjin Luo, Chenghong Li, Yihao Zhi, Xiaoguang Han  

**Link**: [PDF](https://arxiv.org/pdf/2510.23306)  

**Abstract**: Existing multi-view 3D object reconstruction methods heavily rely on sufficient overlap between input views, where occlusions and sparse coverage in practice frequently yield severe reconstruction incompleteness. Recent advancements in diffusion-based 3D generative techniques offer the potential to address these limitations by leveraging learned generative priors to hallucinate invisible parts of objects, thereby generating plausible 3D structures. However, the stochastic nature of the inference process limits the accuracy and reliability of generation results, preventing existing reconstruction frameworks from integrating such 3D generative priors. In this work, we comprehensively analyze the reasons why diffusion-based 3D generative methods fail to achieve high consistency, including (a) the insufficiency in constructing and leveraging cross-view connections when extracting multi-view image features as conditions, and (b) the poor controllability of iterative denoising during local detail generation, which easily leads to plausible but inconsistent fine geometric and texture details with inputs. Accordingly, we propose ReconViaGen to innovatively integrate reconstruction priors into the generative framework and devise several strategies that effectively address these issues. Extensive experiments demonstrate that our ReconViaGen can reconstruct complete and accurate 3D models consistent with input views in both global structure and local this http URL page: this https URL. 

---
# A Novel Framework for Multi-Modal Protein Representation Learning 

**Authors**: Runjie Zheng, Zhen Wang, Anjie Qiao, Jiancong Xie, Jiahua Rao, Yuedong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.23273)  

**Abstract**: Accurate protein function prediction requires integrating heterogeneous intrinsic signals (e.g., sequence and structure) with noisy extrinsic contexts (e.g., protein-protein interactions and GO term annotations). However, two key challenges hinder effective fusion: (i) cross-modal distributional mismatch among embeddings produced by pre-trained intrinsic encoders, and (ii) noisy relational graphs of extrinsic data that degrade GNN-based information aggregation. We propose Diffused and Aligned Multi-modal Protein Embedding (DAMPE), a unified framework that addresses these through two core mechanisms. First, we propose Optimal Transport (OT)-based representation alignment that establishes correspondence between intrinsic embedding spaces of different modalities, effectively mitigating cross-modal heterogeneity. Second, we develop a Conditional Graph Generation (CGG)-based information fusion method, where a condition encoder fuses the aligned intrinsic embeddings to provide informative cues for graph reconstruction. Meanwhile, our theoretical analysis implies that the CGG objective drives this condition encoder to absorb graph-aware knowledge into its produced protein representations. Empirically, DAMPE outperforms or matches state-of-the-art methods such as DPFunc on standard GO benchmarks, achieving AUPR gains of 0.002-0.013 pp and Fmax gains 0.004-0.007 pp. Ablation studies further show that OT-based alignment contributes 0.043-0.064 pp AUPR, while CGG-based fusion adds 0.005-0.111 pp Fmax. Overall, DAMPE offers a scalable and theoretically grounded approach for robust multi-modal protein representation learning, substantially enhancing protein function prediction. 

---
# PAHQ: Accelerating Automated Circuit Discovery through Mixed-Precision Inference Optimization 

**Authors**: Xinhai Wang, Shu Yang, Liangyu Wang, Lin Zhang, Huanyi Xie, Lijie Hu, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.23264)  

**Abstract**: Circuit discovery, which involves identifying sparse and task-relevant subnetworks in pre-trained language models, is a cornerstone of mechanistic interpretability. Automated Circuit Discovery (ACDC) has emerged as a pivotal methodology in circuit discovery, but its application to large language models is severely limited by computational inefficiency and prohibitively high memory requirements. Although several accelerated approaches have been proposed, they primarily rely on linear approximations to ACDC, which significantly compromises analytical faithfulness. Our proposed method for accelerating automated circuit discovery, Per Attention Head Quantization (PAHQ), takes a fundamentally different approach by optimizing the efficiency of each individual patching operation. PAHQ leverages a fundamental alignment between activation patching and mixed-precision quantization (MPQ): interpretability analysis through patching essentially performs targeted ablation studies. Therefore, we can maintain high precision exclusively for investigated components while safely reducing precision elsewhere in the network. PAHQ-accelerated ACDC reduces runtime by up to 80\% and memory consumption by up to 30\% compared to unaccelerated ACDC while maintaining faithfulness. Importantly, our method readily integrates with existing edge-based circuit discovery techniques by modifying the attention computation mechanism. This training-free approach provides a practical and novel pathway for accelerating mechanistic interpretability methods. Our code is available at this https URL. 

---
# Deep Active Inference with Diffusion Policy and Multiple Timescale World Model for Real-World Exploration and Navigation 

**Authors**: Riko Yokozawa, Kentaro Fujii, Yuta Nomura, Shingo Murata  

**Link**: [PDF](https://arxiv.org/pdf/2510.23258)  

**Abstract**: Autonomous robotic navigation in real-world environments requires exploration to acquire environmental information as well as goal-directed navigation in order to reach specified targets. Active inference (AIF) based on the free-energy principle provides a unified framework for these behaviors by minimizing the expected free energy (EFE), thereby combining epistemic and extrinsic values. To realize this practically, we propose a deep AIF framework that integrates a diffusion policy as the policy model and a multiple timescale recurrent state-space model (MTRSSM) as the world model. The diffusion policy generates diverse candidate actions while the MTRSSM predicts their long-horizon consequences through latent imagination, enabling action selection that minimizes EFE. Real-world navigation experiments demonstrated that our framework achieved higher success rates and fewer collisions compared with the baselines, particularly in exploration-demanding scenarios. These results highlight how AIF based on EFE minimization can unify exploration and goal-directed navigation in real-world robotic settings. 

---
# Progressive Growing of Patch Size: Curriculum Learning for Accelerated and Improved Medical Image Segmentation 

**Authors**: Stefan M. Fischer, Johannes Kiechle, Laura Daza, Lina Felsner, Richard Osuala, Daniel M. Lang, Karim Lekadir, Jan C. Peeken, Julia A. Schnabel  

**Link**: [PDF](https://arxiv.org/pdf/2510.23241)  

**Abstract**: In this work, we introduce Progressive Growing of Patch Size, an automatic curriculum learning approach for 3D medical image segmentation. Our approach progressively increases the patch size during model training, resulting in an improved class balance for smaller patch sizes and accelerated convergence of the training process. We evaluate our curriculum approach in two settings: a resource-efficient mode and a performance mode, both regarding Dice score performance and computational costs across 15 diverse and popular 3D medical image segmentation tasks. The resource-efficient mode matches the Dice score performance of the conventional constant patch size sampling baseline with a notable reduction in training time to only 44%. The performance mode improves upon constant patch size segmentation results, achieving a statistically significant relative mean performance gain of 1.28% in Dice Score. Remarkably, across all 15 tasks, our proposed performance mode manages to surpass the constant patch size baseline in Dice Score performance, while simultaneously reducing training time to only 89%. The benefits are particularly pronounced for highly imbalanced tasks such as lesion segmentation tasks. Rigorous experiments demonstrate that our performance mode not only improves mean segmentation performance but also reduces performance variance, yielding more trustworthy model comparison. Furthermore, our findings reveal that the proposed curriculum sampling is not tied to a specific architecture but represents a broadly applicable strategy that consistently boosts performance across diverse segmentation models, including UNet, UNETR, and SwinUNETR. In summary, we show that this simple yet elegant transformation on input data substantially improves both Dice Score performance and training runtime, while being compatible across diverse segmentation backbones. 

---
# Process Reward Models for Sentence-Level Verification of LVLM Radiology Reports 

**Authors**: Alois Thomas, Maya Varma, Jean-Benoit Delbrouck, Curtis P. Langlotz  

**Link**: [PDF](https://arxiv.org/pdf/2510.23217)  

**Abstract**: Automating radiology report generation with Large Vision-Language Models (LVLMs) holds great potential, yet these models often produce clinically critical hallucinations, posing serious risks. Existing hallucination detection methods frequently lack the necessary sentence-level granularity or robust generalization across different LVLM generators. We introduce a novel approach: a sentence-level Process Reward Model (PRM) adapted for this vision-language task. Our PRM predicts the factual correctness of each generated sentence, conditioned on clinical context and preceding text. When fine-tuned on MIMIC-CXR with weakly-supervised labels, a lightweight 0.5B-parameter PRM outperforms existing verification techniques, demonstrating, for instance, relative improvements of 7.5% in Matthews Correlation Coefficient and 1.8% in AUROC over strong white-box baselines on outputs from one LVLM. Unlike methods reliant on internal model states, our PRM demonstrates strong generalization to an unseen LVLM. We further show its practical utility: PRM scores effectively filter low-quality reports, improving F1-CheXbert scores by 4.5% (when discarding the worst 10% of reports). Moreover, when guiding a novel weighted best-of-N selection process on the MIMIC-CXR test set, our PRM show relative improvements in clinical metrics of 7.4% for F1-CheXbert and 0.6% for BERTScore. These results demonstrate that a lightweight, context-aware PRM provides a model-agnostic safety layer for clinical LVLMs without access to internal activations 

---
# Accelerating Eigenvalue Dataset Generation via Chebyshev Subspace Filter 

**Authors**: Hong Wang, Jie Wang, Jian Luo, huanshuo dong, Yeqiu Chen, Runmin Jiang, Zhen huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.23215)  

**Abstract**: Eigenvalue problems are among the most important topics in many scientific disciplines. With the recent surge and development of machine learning, neural eigenvalue methods have attracted significant attention as a forward pass of inference requires only a tiny fraction of the computation time compared to traditional solvers. However, a key limitation is the requirement for large amounts of labeled data in training, including operators and their eigenvalues. To tackle this limitation, we propose a novel method, named Sorting Chebyshev Subspace Filter (SCSF), which significantly accelerates eigenvalue data generation by leveraging similarities between operators -- a factor overlooked by existing methods. Specifically, SCSF employs truncated fast Fourier transform sorting to group operators with similar eigenvalue distributions and constructs a Chebyshev subspace filter that leverages eigenpairs from previously solved problems to assist in solving subsequent ones, reducing redundant computations. To the best of our knowledge, SCSF is the first method to accelerate eigenvalue data generation. Experimental results show that SCSF achieves up to a $3.5\times$ speedup compared to various numerical solvers. 

---
# Increasing LLM Coding Capabilities through Diverse Synthetic Coding Tasks 

**Authors**: Amal Abed, Ivan Lukic, Jörg K.H. Franke, Frank Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2510.23208)  

**Abstract**: Large language models (LLMs) have shown impressive promise in code generation, yet their progress remains limited by the shortage of large-scale datasets that are both diverse and well-aligned with human reasoning. Most existing resources pair problems with solutions, but omit the intermediate thought process that guides coding. To close this gap, we present a scalable synthetic data generation pipeline that produces nearly 800k instruction-reasoning-code-test quadruplets. Each sample combines a task, a step-by-step reasoning trace, a working solution, and executable tests, enabling models to learn not just the what but also the how of problem solving. Our pipeline combines four key components: curated contest problems, web-mined content filtered by relevance classifiers, data expansion guided by reasoning patterns, and multi-stage execution-based validation. A genetic mutation algorithm further increases task diversity while maintaining consistency between reasoning traces and code implementations. Our key finding is that fine-tuning LLMs on this dataset yields consistent improvements on coding benchmarks. Beyond raw accuracy, reasoning-aware data can substitute for model scaling, generalize across architectures, and outperform leading open-source alternatives under identical sample budgets. Our work establishes reasoning-centered synthetic data generation as an efficient approach for advancing coding capabilities in LLMs. We publish our dataset and generation pipeline to facilitate further research. 

---
# PTPP-Aware Adaptation Scaling Laws: Predicting Domain-Adaptation Performance at Unseen Pre-Training Budgets 

**Authors**: Etienne Goffinet, Shane Bergsma, Avraham Sheinin, Natalia Vassilieva, Shaheer Muhammad, Preslav Nakov, Gurpreet Gosal  

**Link**: [PDF](https://arxiv.org/pdf/2510.23198)  

**Abstract**: Continual pre-training (CPT) for domain adaptation must balance target-domain gains with stability on the base domain. Existing CPT scaling laws typically assume a fixed pre-training budget, which limits their ability to forecast adaptation outcomes for models trained at different tokens-per-parameter (PTPP). We present \emph{PTPP-aware} adaptation scaling laws that make the pre-training budget an explicit variable, enabling accurate \emph{prediction} of adaptation loss at unseen \ptpp. On a multilingual setup (English/Arabic $\rightarrow$ French), PTPP-aware formulations trained on early stages (\ptpp{}=\{15,31\}) predict target loss at \ptpp{}=279 and outperform a PTPP-agnostic \dcpt{} transfer baseline on metrics (Huber-on-log, MAE$_\mathrm{rel}$, calibration slope); full diagnostics (RMSE, MAPE) are in the appendix. Beyond forecasting, we show a practical use case: planning replay ratios and adaptation token budgets that satisfy target and forgetting constraints under compute limits. 

---
# DREaM: Drug-Drug Relation Extraction via Transfer Learning Method 

**Authors**: Ali Fata, Hossein Rahmani, Parinaz Soltanzadeh, Amirhossein Derakhshan, Behrouz Minaei Bidgoli  

**Link**: [PDF](https://arxiv.org/pdf/2510.23189)  

**Abstract**: Relation extraction between drugs plays a crucial role in identifying drug drug interactions and predicting side effects. The advancement of machine learning methods in relation extraction, along with the development of large medical text databases, has enabled the low cost extraction of such relations compared to other approaches that typically require expert knowledge. However, to the best of our knowledge, there are limited datasets specifically designed for drug drug relation extraction currently available. Therefore, employing transfer learning becomes necessary to apply machine learning methods in this domain. In this study, we propose DREAM, a method that first employs a trained relation extraction model to discover relations between entities and then applies this model to a corpus of medical texts to construct an ontology of drug relationships. The extracted relations are subsequently validated using a large language model. Quantitative results indicate that the LLM agreed with 71 of the relations extracted from a subset of PubMed abstracts. Furthermore, our qualitative analysis indicates that this approach can uncover ambiguities in the medical domain, highlighting the challenges inherent in relation extraction in this field. 

---
# Beyond Direct Generation: A Decomposed Approach to Well-Crafted Screenwriting with LLMs 

**Authors**: Hang Lei, Shengyi Zong, Zhaoyan Li, Ziren Zhou, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.23163)  

**Abstract**: The screenplay serves as the foundation for television production, defining narrative structure, character development, and dialogue. While Large Language Models (LLMs) show great potential in creative writing, direct end-to-end generation approaches often fail to produce well-crafted screenplays. We argue this failure stems from forcing a single model to simultaneously master two disparate capabilities: creative narrative construction and rigid format adherence. The resulting outputs may mimic superficial style but lack the deep structural integrity and storytelling substance required for professional use. To enable LLMs to generate high-quality screenplays, we introduce Dual-Stage Refinement (DSR), a decomposed framework that decouples creative narrative generation from format conversion. The first stage transforms a brief outline into rich, novel-style prose. The second stage refines this narrative into a professionally formatted screenplay. This separation enables the model to specialize in one distinct capability at each stage. A key challenge in implementing DSR is the scarcity of paired outline-to-novel training data. We address this through hybrid data synthesis: reverse synthesis deconstructs existing screenplays into structured inputs, while forward synthesis leverages these inputs to generate high-quality narrative texts as training targets. Blind evaluations by professional screenwriters show that DSR achieves a 75% win rate against strong baselines like Gemini-2.5-Pro and reaches 82.7% of human-level performance. Our work demonstrates that decomposed generation architecture with tailored data synthesis effectively specializes LLMs in complex creative domains. 

---
# Enabling Vibration-Based Gesture Recognition on Everyday Furniture via Energy-Efficient FPGA Implementation of 1D Convolutional Networks 

**Authors**: Koki Shibata, Tianheng Ling, Chao Qian, Tomokazu Matsui, Hirohiko Suwa, Keiichi Yasumoto, Gregor Schiele  

**Link**: [PDF](https://arxiv.org/pdf/2510.23156)  

**Abstract**: The growing demand for smart home interfaces has increased interest in non-intrusive sensing methods like vibration-based gesture recognition. While prior studies demonstrated feasibility, they often rely on complex preprocessing and large Neural Networks (NNs) requiring costly high-performance hardware, resulting in high energy usage and limited real-world deployability. This study proposes an energy-efficient solution deploying compact NNs on low-power Field-Programmable Gate Arrays (FPGAs) to enable real-time gesture recognition with competitive accuracy. We adopt a series of optimizations: (1) We replace complex spectral preprocessing with raw waveform input, eliminating complex on-board preprocessing while reducing input size by 21x without sacrificing accuracy. (2) We design two lightweight architectures (1D-CNN and 1D-SepCNN) tailored for embedded FPGAs, reducing parameters from 369 million to as few as 216 while maintaining comparable accuracy. (3) With integer-only quantization and automated RTL generation, we achieve seamless FPGA deployment. A ping-pong buffering mechanism in 1D-SepCNN further improves deployability under tight memory constraints. (4) We extend a hardware-aware search framework to support constraint-driven model configuration selection, considering accuracy, deployability, latency, and energy consumption. Evaluated on two swipe-direction datasets with multiple users and ordinary tables, our approach achieves low-latency, energy-efficient inference on the AMD Spartan-7 XC7S25 FPGA. Under the PS data splitting setting, the selected 6-bit 1D-CNN reaches 0.970 average accuracy across users with 9.22 ms latency. The chosen 8-bit 1D-SepCNN further reduces latency to 6.83 ms (over 53x CPU speedup) with slightly lower accuracy (0.949). Both consume under 1.2 mJ per inference, demonstrating suitability for long-term edge operation. 

---
# Adapting Interleaved Encoders with PPO for Language-Guided Reinforcement Learning in BabyAI 

**Authors**: Aryan Mathur, Asaduddin Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2510.23148)  

**Abstract**: Deep reinforcement learning agents often struggle when tasks require understanding both vision and language. Conventional architectures typically isolate perception (for example, CNN-based visual encoders) from decision-making (policy networks). This separation can be inefficient, since the policy's failures do not directly help the perception module learn what is important. To address this, we implement the Perception-Decision Interleaving Transformer (PDiT) architecture introduced by Mao et al. (2023), a model that alternates between perception and decision layers within a single transformer. This interleaving allows feedback from decision-making to refine perceptual features dynamically. In addition, we integrate a contrastive loss inspired by CLIP to align textual mission embeddings with visual scene features. We evaluate the PDiT encoders on the BabyAI GoToLocal environment and find that the approach achieves more stable rewards and stronger alignment compared to a standard PPO baseline. The results suggest that interleaved transformer encoders are a promising direction for developing more integrated autonomous agents. 

---
# Rethinking GSPO: The Perplexity-Entropy Equivalence 

**Authors**: Chi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.23142)  

**Abstract**: We provide a new perspective on GSPO's length-normalized importance ratios by establishing their connection to information-theoretic quantities. We show that GSPO's sequence-level weight $s(\theta) = (\pi_\theta/\pi_{\theta_{\text{old}}})^{1/|y|}$ can be equivalently expressed as the inverse perplexity ratio $\text{PPL}_{\theta_{\text{old}}}/\text{PPL}_\theta$ and as the exponential cross-entropy change $\exp(\Delta H)$. While the perplexity-entropy relationship follows from standard definitions, this observation provides a useful lens for understanding GSPO: the algorithm weights policy gradient updates by perplexity ratios, offering an information-theoretic interpretation of the importance weights. This perspective helps explain GSPO's empirical properties, including log-domain variance reduction through geometric averaging and stability in training mixture-of-experts models. We validate the mathematical equivalences and variance predictions through controlled experiments on mathematical reasoning tasks. 

---
# GroupSHAP-Guided Integration of Financial News Keywords and Technical Indicators for Stock Price Prediction 

**Authors**: Minjoo Kim, Jinwoong Kim, Sangjin Park  

**Link**: [PDF](https://arxiv.org/pdf/2510.23112)  

**Abstract**: Recent advances in finance-specific language models such as FinBERT have enabled the quantification of public sentiment into index-based measures, yet compressing diverse linguistic signals into single metrics overlooks contextual nuances and limits interpretability. To address this limitation, explainable AI techniques, particularly SHAP (SHapley Additive Explanations), have been employed to identify influential features. However, SHAP's computational cost grows exponentially with input features, making it impractical for large-scale text-based financial data. This study introduces a GRU-based forecasting framework enhanced with GroupSHAP, which quantifies contributions of semantically related keyword groups rather than individual tokens, substantially reducing computational burden while preserving interpretability. We employed FinBERT to embed news articles from 2015 to 2024, clustered them into coherent semantic groups, and applied GroupSHAP to measure each group's contribution to stock price movements. The resulting group-level SHAP variables across multiple topics were used as input features for the prediction model. Empirical results from one-day-ahead forecasting of the S&P 500 index throughout 2024 demonstrate that our approach achieves a 32.2% reduction in MAE and a 40.5% reduction in RMSE compared with benchmark models without the GroupSHAP mechanism. This research presents the first application of GroupSHAP in news-driven financial forecasting, showing that grouped sentiment representations simultaneously enhance interpretability and predictive performance. 

---
# Leveraging Hierarchical Organization for Medical Multi-document Summarization 

**Authors**: Yi-Li Hsu, Katelyn X. Mei, Lucy Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.23104)  

**Abstract**: Medical multi-document summarization (MDS) is a complex task that requires effectively managing cross-document relationships. This paper investigates whether incorporating hierarchical structures in the inputs of MDS can improve a model's ability to organize and contextualize information across documents compared to traditional flat summarization methods. We investigate two ways of incorporating hierarchical organization across three large language models (LLMs), and conduct comprehensive evaluations of the resulting summaries using automated metrics, model-based metrics, and domain expert evaluation of preference, understandability, clarity, complexity, relevance, coverage, factuality, and coherence. Our results show that human experts prefer model-generated summaries over human-written summaries. Hierarchical approaches generally preserve factuality, coverage, and coherence of information, while also increasing human preference for summaries. Additionally, we examine whether simulated judgments from GPT-4 align with human judgments, finding higher agreement along more objective evaluation facets. Our findings demonstrate that hierarchical structures can improve the clarity of medical summaries generated by models while maintaining content coverage, providing a practical way to improve human preference for generated summaries. 

---
# Think before Recommendation: Autonomous Reasoning-enhanced Recommender 

**Authors**: Xiaoyu Kong, Junguang Jiang, Bin Liu, Ziru Xu, Han Zhu, Jian Xu, Bo Zheng, Jiancan Wu, Xiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.23077)  

**Abstract**: The core task of recommender systems is to learn user preferences from historical user-item interactions. With the rapid development of large language models (LLMs), recent research has explored leveraging the reasoning capabilities of LLMs to enhance rating prediction tasks. However, existing distillation-based methods suffer from limitations such as the teacher model's insufficient recommendation capability, costly and static supervision, and superficial transfer of reasoning ability. To address these issues, this paper proposes RecZero, a reinforcement learning (RL)-based recommendation paradigm that abandons the traditional multi-model and multi-stage distillation approach. Instead, RecZero trains a single LLM through pure RL to autonomously develop reasoning capabilities for rating prediction. RecZero consists of two key components: (1) "Think-before-Recommendation" prompt construction, which employs a structured reasoning template to guide the model in step-wise analysis of user interests, item features, and user-item compatibility; and (2) rule-based reward modeling, which adopts group relative policy optimization (GRPO) to compute rewards for reasoning trajectories and optimize the LLM. Additionally, the paper explores a hybrid paradigm, RecOne, which combines supervised fine-tuning with RL, initializing the model with cold-start reasoning samples and further optimizing it with RL. Experimental results demonstrate that RecZero and RecOne significantly outperform existing baseline methods on multiple benchmark datasets, validating the superiority of the RL paradigm in achieving autonomous reasoning-enhanced recommender systems. 

---
# Quality-Aware Translation Tagging in Multilingual RAG system 

**Authors**: Hoyeon Moon, Byeolhee Kim, Nikhil Verma  

**Link**: [PDF](https://arxiv.org/pdf/2510.23070)  

**Abstract**: Multilingual Retrieval-Augmented Generation (mRAG) often retrieves English documents and translates them into the query language for low-resource settings. However, poor translation quality degrades response generation performance. Existing approaches either assume sufficient translation quality or utilize the rewriting method, which introduces factual distortion and hallucinations. To mitigate these problems, we propose Quality-Aware Translation Tagging in mRAG (QTT-RAG), which explicitly evaluates translation quality along three dimensions-semantic equivalence, grammatical accuracy, and naturalness&fluency-and attach these scores as metadata without altering the original content. We evaluate QTT-RAG against CrossRAG and DKM-RAG as baselines in two open-domain QA benchmarks (XORQA, MKQA) using six instruction-tuned LLMs ranging from 2.4B to 14B parameters, covering two low-resource languages (Korean and Finnish) and one high-resource language (Chinese). QTT-RAG outperforms the baselines by preserving factual integrity while enabling generator models to make informed decisions based on translation reliability. This approach allows for effective usage of cross-lingual documents in low-resource settings with limited native language documents, offering a practical and robust solution across multilingual domains. 

---
# Advantage Shaping as Surrogate Reward Maximization: Unifying Pass@K Policy Gradients 

**Authors**: Christos Thrampoulidis, Sadegh Mahdavi, Wenlong Deng  

**Link**: [PDF](https://arxiv.org/pdf/2510.23049)  

**Abstract**: This note reconciles two seemingly distinct approaches to policy gradient optimization for the Pass@K objective in reinforcement learning with verifiable rewards: (1) direct REINFORCE-style methods, and (2) advantage-shaping techniques that directly modify GRPO. We show that these are two sides of the same coin. By reverse-engineering existing advantage-shaping algorithms, we reveal that they implicitly optimize surrogate rewards. We specifically interpret practical ``hard-example up-weighting'' modifications to GRPO as reward-level regularization. Conversely, starting from surrogate reward objectives, we provide a simple recipe for deriving both existing and new advantage-shaping methods. This perspective provides a lens for RLVR policy gradient optimization beyond our original motivation of Pass@K. 

---
# LLM Meets Diffusion: A Hybrid Framework for Crystal Material Generation 

**Authors**: Subhojyoti Khastagir, Kishalay Das, Pawan Goyal, Seung-Cheol Lee, Satadeep Bhattacharjee, Niloy Ganguly  

**Link**: [PDF](https://arxiv.org/pdf/2510.23040)  

**Abstract**: Recent advances in generative modeling have shown significant promise in designing novel periodic crystal structures. Existing approaches typically rely on either large language models (LLMs) or equivariant denoising models, each with complementary strengths: LLMs excel at handling discrete atomic types but often struggle with continuous features such as atomic positions and lattice parameters, while denoising models are effective at modeling continuous variables but encounter difficulties in generating accurate atomic compositions. To bridge this gap, we propose CrysLLMGen, a hybrid framework that integrates an LLM with a diffusion model to leverage their complementary strengths for crystal material generation. During sampling, CrysLLMGen first employs a fine-tuned LLM to produce an intermediate representation of atom types, atomic coordinates, and lattice structure. While retaining the predicted atom types, it passes the atomic coordinates and lattice structure to a pre-trained equivariant diffusion model for refinement. Our framework outperforms state-of-the-art generative models across several benchmark tasks and datasets. Specifically, CrysLLMGen not only achieves a balanced performance in terms of structural and compositional validity but also generates more stable and novel materials compared to LLM-based and denoisingbased models Furthermore, CrysLLMGen exhibits strong conditional generation capabilities, effectively producing materials that satisfy user-defined constraints. Code is available at this https URL 

---
# Incentivizing Agentic Reasoning in LLM Judges via Tool-Integrated Reinforcement Learning 

**Authors**: Ran Xu, Jingjing Chen, Jiayu Ye, Yu Wu, Jun Yan, Carl Yang, Hongkun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.23038)  

**Abstract**: Large Language Models (LLMs) are widely used as judges to evaluate response quality, providing a scalable alternative to human evaluation. However, most LLM judges operate solely on intrinsic text-based reasoning, limiting their ability to verify complex constraints or perform accurate computation. Motivated by the success of tool-integrated reasoning (TIR) in numerous tasks, we propose TIR-Judge, an end-to-end RL framework for training LLM judges that integrates a code executor for precise evaluation. TIR-Judge is built on three principles: (i) diverse training across verifiable and non-verifiable domains, (ii) flexible judgment formats (pointwise, pairwise, listwise), and (iii) iterative RL that bootstraps directly from the initial model without distillation. On seven public benchmarks, TIR-Judge surpasses strong reasoning-based judges by up to 6.4% (pointwise) and 7.7% (pairwise), and achieves listwise performance comparable to Claude-Opus-4 despite having only 8B parameters. Remarkably, TIR-Judge-Zero - trained entirely without distilled judge trajectories, matches the performance of distilled variants, demonstrating that tool-augmented judges can self-evolve through iterative reinforcement learning. 

---
# A high-capacity linguistic steganography based on entropy-driven rank-token mapping 

**Authors**: Jun Jiang, Weiming Zhang, Nenghai Yu, Kejiang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.23035)  

**Abstract**: Linguistic steganography enables covert communication through embedding secret messages into innocuous texts; however, current methods face critical limitations in payload capacity and security. Traditional modification-based methods introduce detectable anomalies, while retrieval-based strategies suffer from low embedding capacity. Modern generative steganography leverages language models to generate natural stego text but struggles with limited entropy in token predictions, further constraining capacity. To address these issues, we propose an entropy-driven framework called RTMStega that integrates rank-based adaptive coding and context-aware decompression with normalized entropy. By mapping secret messages to token probability ranks and dynamically adjusting sampling via context-aware entropy-based adjustments, RTMStega achieves a balance between payload capacity and imperceptibility. Experiments across diverse datasets and models demonstrate that RTMStega triples the payload capacity of mainstream generative steganography, reduces processing time by over 50%, and maintains high text quality, offering a trustworthy solution for secure and efficient covert communication. 

---
# Efficient and Encrypted Inference using Binarized Neural Networks within In-Memory Computing Architectures 

**Authors**: Gokulnath Rajendran, Suman Deb, Anupam Chattopadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2510.23034)  

**Abstract**: Binarized Neural Networks (BNNs) are a class of deep neural networks designed to utilize minimal computational resources, which drives their popularity across various applications. Recent studies highlight the potential of mapping BNN model parameters onto emerging non-volatile memory technologies, specifically using crossbar architectures, resulting in improved inference performance compared to traditional CMOS implementations. However, the common practice of protecting model parameters from theft attacks by storing them in an encrypted format and decrypting them at runtime introduces significant computational overhead, thus undermining the core principles of in-memory computing, which aim to integrate computation and storage. This paper presents a robust strategy for protecting BNN model parameters, particularly within in-memory computing frameworks. Our method utilizes a secret key derived from a physical unclonable function to transform model parameters prior to storage in the crossbar. Subsequently, the inference operations are performed on the encrypted weights, achieving a very special case of Fully Homomorphic Encryption (FHE) with minimal runtime overhead. Our analysis reveals that inference conducted without the secret key results in drastically diminished performance, with accuracy falling below 15%. These results validate the effectiveness of our protection strategy in securing BNNs within in-memory computing architectures while preserving computational efficiency. 

---
# Nested AutoRegressive Models 

**Authors**: Hongyu Wu, Xuhui Fan, Zhangkai Wu, Longbing Cao  

**Link**: [PDF](https://arxiv.org/pdf/2510.23028)  

**Abstract**: AutoRegressive (AR) models have demonstrated competitive performance in image generation, achieving results comparable to those of diffusion models. However, their token-by-token image generation mechanism remains computationally intensive and existing solutions such as VAR often lead to limited sample diversity. In this work, we propose a Nested AutoRegressive~(NestAR) model, which proposes nested AutoRegressive architectures in generating images. NestAR designs multi-scale modules in a hierarchical order. These different scaled modules are constructed in an AR architecture, where one larger-scale module is conditioned on outputs from its previous smaller-scale module. Within each module, NestAR uses another AR structure to generate ``patches'' of tokens. The proposed nested AR architecture reduces the overall complexity from $\mathcal{O}(n)$ to $\mathcal{O}(\log n)$ in generating $n$ image tokens, as well as increases image diversities. NestAR further incorporates flow matching loss to use continuous tokens, and develops objectives to coordinate these multi-scale modules in model training. NestAR achieves competitive image generation performance while significantly lowering computational cost. 

---
# MoEMeta: Mixture-of-Experts Meta Learning for Few-Shot Relational Learning 

**Authors**: Han Wu, Jie Yin  

**Link**: [PDF](https://arxiv.org/pdf/2510.23013)  

**Abstract**: Few-shot knowledge graph relational learning seeks to perform reasoning over relations given only a limited number of training examples. While existing approaches largely adopt a meta-learning framework for enabling fast adaptation to new relations, they suffer from two key pitfalls. First, they learn relation meta-knowledge in isolation, failing to capture common relational patterns shared across tasks. Second, they struggle to effectively incorporate local, task-specific contexts crucial for rapid adaptation. To address these limitations, we propose MoEMeta, a novel meta-learning framework that disentangles globally shared knowledge from task-specific contexts to enable both effective generalization and rapid adaptation. MoEMeta introduces two key innovations: (i) a mixture-of-experts (MoE) model that learns globally shared relational prototypes to enhance generalization, and (ii) a task-tailored adaptation mechanism that captures local contexts for fast task-specific adaptation. By balancing global generalization with local adaptability, MoEMeta significantly advances few-shot relational learning. Extensive experiments and analyses on three KG benchmarks demonstrate that MoEMeta consistently outperforms existing baselines, achieving state-of-the-art performance. 

---
# Softmax is $1/2$-Lipschitz: A tight bound across all $\ell_p$ norms 

**Authors**: Pravin Nair  

**Link**: [PDF](https://arxiv.org/pdf/2510.23012)  

**Abstract**: The softmax function is a basic operator in machine learning and optimization, used in classification, attention mechanisms, reinforcement learning, game theory, and problems involving log-sum-exp terms. Existing robustness guarantees of learning models and convergence analysis of optimization algorithms typically consider the softmax operator to have a Lipschitz constant of $1$ with respect to the $\ell_2$ norm. In this work, we prove that the softmax function is contractive with the Lipschitz constant $1/2$, uniformly across all $\ell_p$ norms with $p \ge 1$. We also show that the local Lipschitz constant of softmax attains $1/2$ for $p = 1$ and $p = \infty$, and for $p \in (1,\infty)$, the constant remains strictly below $1/2$ and the supremum $1/2$ is achieved only in the limit. To our knowledge, this is the first comprehensive norm-uniform analysis of softmax Lipschitz continuity. We demonstrate how the sharper constant directly improves a range of existing theoretical results on robustness and convergence. We further validate the sharpness of the $1/2$ Lipschitz constant of the softmax operator through empirical studies on attention-based architectures (ViT, GPT-2, Qwen3-8B) and on stochastic policies in reinforcement learning. 

---
# Understanding In-Context Learning Beyond Transformers: An Investigation of State Space and Hybrid Architectures 

**Authors**: Shenran Wang, Timothy Tin-Long Tse, Jian Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.23006)  

**Abstract**: We perform in-depth evaluations of in-context learning (ICL) on state-of-the-art transformer, state-space, and hybrid large language models over two categories of knowledge-based ICL tasks. Using a combination of behavioral probing and intervention-based methods, we have discovered that, while LLMs of different architectures can behave similarly in task performance, their internals could remain different. We discover that function vectors (FVs) responsible for ICL are primarily located in the self-attention and Mamba layers, and speculate that Mamba2 uses a different mechanism from FVs to perform ICL. FVs are more important for ICL involving parametric knowledge retrieval, but not for contextual knowledge understanding. Our work contributes to a more nuanced understanding across architectures and task types. Methodologically, our approach also highlights the importance of combining both behavioural and mechanistic analyses to investigate LLM capabilities. 

---
# USF-MAE: Ultrasound Self-Supervised Foundation Model with Masked Autoencoding 

**Authors**: Youssef Megahed, Robin Ducharme, Mark Walker, Steven Hawken, Adrian D. C. Chan  

**Link**: [PDF](https://arxiv.org/pdf/2510.22990)  

**Abstract**: Ultrasound imaging is one of the most widely used diagnostic modalities, offering real-time, radiation-free assessment across diverse clinical domains. However, interpretation of ultrasound images remains challenging due to high noise levels, operator dependence, and limited field of view, resulting in substantial inter-observer variability. Current Deep Learning approaches are hindered by the scarcity of large labeled datasets and the domain gap between general and sonographic images, which limits the transferability of models pretrained on non-medical data. To address these challenges, we introduce the Ultrasound Self-Supervised Foundation Model with Masked Autoencoding (USF-MAE), the first large-scale self-supervised MAE framework pretrained exclusively on ultrasound data. The model was pre-trained on 370,000 2D and 3D ultrasound images curated from 46 open-source datasets, collectively termed OpenUS-46, spanning over twenty anatomical regions. This curated dataset has been made publicly available to facilitate further research and reproducibility. Using a Vision Transformer encoder-decoder architecture, USF-MAE reconstructs masked image patches, enabling it to learn rich, modality-specific representations directly from unlabeled data. The pretrained encoder was fine-tuned on three public downstream classification benchmarks: BUS-BRA (breast cancer), MMOTU-2D (ovarian tumors), and GIST514-DB (gastrointestinal stromal tumors). Across all tasks, USF-MAE consistently outperformed conventional CNN and ViT baselines, achieving F1-scores of 81.6%, 79.6%, and 82.4%, respectively. Despite not using labels during pretraining, USF-MAE approached the performance of the supervised foundation model UltraSam on breast cancer classification and surpassed it on the other tasks, demonstrating strong cross-anatomical generalization. 

---
# The Reasoning Trap: How Enhancing LLM Reasoning Amplifies Tool Hallucination 

**Authors**: Chenlong Yin, Zeyang Sha, Shiwen Cui, Changhua Meng  

**Link**: [PDF](https://arxiv.org/pdf/2510.22977)  

**Abstract**: Enhancing the reasoning capabilities of Large Language Models (LLMs) is a key strategy for building Agents that "think then act." However, recent observations, like OpenAI's o3, suggest a paradox: stronger reasoning often coincides with increased hallucination, yet no prior work has systematically examined whether reasoning enhancement itself causes tool hallucination. To address this gap, we pose the central question: Does strengthening reasoning increase tool hallucination? To answer this, we introduce SimpleToolHalluBench, a diagnostic benchmark measuring tool hallucination in two failure modes: (i) no tool available, and (ii) only distractor tools available. Through controlled experiments, we establish three key findings. First, we demonstrate a causal relationship: progressively enhancing reasoning through RL increases tool hallucination proportionally with task performance gains. Second, this effect transcends overfitting - training on non-tool tasks (e.g., mathematics) still amplifies subsequent tool hallucination. Third, the effect is method-agnostic, appearing when reasoning is instilled via supervised fine-tuning and when it is merely elicited at inference by switching from direct answers to step-by-step thinking. We also evaluate mitigation strategies including Prompt Engineering and Direct Preference Optimization (DPO), revealing a fundamental reliability-capability trade-off: reducing hallucination consistently degrades utility. Mechanistically, Reasoning RL disproportionately collapses tool-reliability-related representations, and hallucinations surface as amplified divergences concentrated in late-layer residual streams. These findings reveal that current reasoning enhancement methods inherently amplify tool hallucination, highlighting the need for new training objectives that jointly optimize for capability and reliability. 

---
# Measuring Teaching with LLMs 

**Authors**: Michael Hardy  

**Link**: [PDF](https://arxiv.org/pdf/2510.22968)  

**Abstract**: Objective and scalable measurement of teaching quality is a persistent challenge in education. While Large Language Models (LLMs) offer potential, general-purpose models have struggled to reliably apply complex, authentic classroom observation instruments. This paper uses custom LLMs built on sentence-level embeddings, an architecture better suited for the long-form, interpretive nature of classroom transcripts than conventional subword tokenization. We systematically evaluate five different sentence embeddings under a data-efficient training regime designed to prevent overfitting. Our results demonstrate that these specialized models can achieve human-level and even super-human performance with expert human ratings above 0.65 and surpassing the average human-human rater correlation. Further, through analysis of annotation context windows, we find that more advanced models-those better aligned with human judgments-attribute a larger share of score variation to lesson-level features rather than isolated utterances, challenging the sufficiency of single-turn annotation paradigms. Finally, to assess external validity, we find that aggregate model scores align with teacher value-added measures, indicating they are capturing features relevant to student learning. However, this trend does not hold at the individual item level, suggesting that while the models learn useful signals, they have not yet achieved full generalization. This work establishes a viable and powerful new methodology for AI-driven instructional measurement, offering a path toward providing scalable, reliable, and valid feedback for educator development. 

---
# MAD-Fact: A Multi-Agent Debate Framework for Long-Form Factuality Evaluation in LLMs 

**Authors**: Yucheng Ning, Xixun Lin, Fang Fang, Yanan Cao  

**Link**: [PDF](https://arxiv.org/pdf/2510.22967)  

**Abstract**: The widespread adoption of Large Language Models (LLMs) raises critical concerns about the factual accuracy of their outputs, especially in high-risk domains such as biomedicine, law, and education. Existing evaluation methods for short texts often fail on long-form content due to complex reasoning chains, intertwined perspectives, and cumulative information. To address this, we propose a systematic approach integrating large-scale long-form datasets, multi-agent verification mechanisms, and weighted evaluation metrics. We construct LongHalluQA, a Chinese long-form factuality dataset; and develop MAD-Fact, a debate-based multi-agent verification system. We introduce a fact importance hierarchy to capture the varying significance of claims in long-form texts. Experiments on two benchmarks show that larger LLMs generally maintain higher factual consistency, while domestic models excel on Chinese content. Our work provides a structured framework for evaluating and enhancing factual reliability in long-form LLM outputs, guiding their safe deployment in sensitive domains. 

---
# CompressionAttack: Exploiting Prompt Compression as a New Attack Surface in LLM-Powered Agents 

**Authors**: Zesen Liu, Zhixiang Zhang, Yuchong Xie, Dongdong She  

**Link**: [PDF](https://arxiv.org/pdf/2510.22963)  

**Abstract**: LLM-powered agents often use prompt compression to reduce inference costs, but this introduces a new security risk. Compression modules, which are optimized for efficiency rather than safety, can be manipulated by adversarial inputs, causing semantic drift and altering LLM behavior. This work identifies prompt compression as a novel attack surface and presents CompressionAttack, the first framework to exploit it. CompressionAttack includes two strategies: HardCom, which uses discrete adversarial edits for hard compression, and SoftCom, which performs latent-space perturbations for soft compression. Experiments on multiple LLMs show up to 80% attack success and 98% preference flips, while remaining highly stealthy and transferable. Case studies in VSCode Cline and Ollama confirm real-world impact, and current defenses prove ineffective, highlighting the need for stronger protections. 

---
# FAME: Fairness-aware Attention-modulated Video Editing 

**Authors**: Zhangkai Wu, Xuhui Fan, Zhongyuan Xie, Kaize Shi, Zhidong Li, Longbing Cao  

**Link**: [PDF](https://arxiv.org/pdf/2510.22960)  

**Abstract**: Training-free video editing (VE) models tend to fall back on gender stereotypes when rendering profession-related prompts. We propose \textbf{FAME} for \textit{Fairness-aware Attention-modulated Video Editing} that mitigates profession-related gender biases while preserving prompt alignment and temporal consistency for coherent VE. We derive fairness embeddings from existing minority representations by softly injecting debiasing tokens into the text encoder. Simultaneously, FAME integrates fairness modulation into both temporal self attention and prompt-to-region cross attention to mitigate the motion corruption and temporal inconsistency caused by directly introducing fairness cues. For temporal self attention, FAME introduces a region constrained attention mask combined with time decay weighting, which enhances intra-region coherence while suppressing irrelevant inter-region interactions. For cross attention, it reweights tokens to region matching scores by incorporating fairness sensitive similarity masks derived from debiasing prompt embeddings. Together, these modulations keep fairness-sensitive semantics tied to the right visual regions and prevent temporal drift across frames. Extensive experiments on new VE fairness-oriented benchmark \textit{FairVE} demonstrate that FAME achieves stronger fairness alignment and semantic fidelity, surpassing existing VE baselines. 

---
# Manifold Approximation leads to Robust Kernel Alignment 

**Authors**: Mohammad Tariqul Islam, Du Liu, Deblina Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2510.22953)  

**Abstract**: Centered kernel alignment (CKA) is a popular metric for comparing representations, determining equivalence of networks, and neuroscience research. However, CKA does not account for the underlying manifold and relies on numerous heuristics that cause it to behave differently at different scales of data. In this work, we propose Manifold approximated Kernel Alignment (MKA), which incorporates manifold geometry into the alignment task. We derive a theoretical framework for MKA. We perform empirical evaluations on synthetic datasets and real-world examples to characterize and compare MKA to its contemporaries. Our findings suggest that manifold-aware kernel alignment provides a more robust foundation for measuring representations, with potential applications in representation learning. 

---
# PASS-Enhanced MEC: Joint Optimization of Task Offloading and Uplink PASS Beamforming 

**Authors**: Zhaoming Hu, Ruikang Zhong, Xidong Mu, Dengao Li, Yuanwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.22948)  

**Abstract**: A pinching-antenna system (PASS)-enhanced mobile edge computing (MEC) architecture is investigated to improve the task offloading efficiency and latency performance in dynamic wireless environments. By leveraging dielectric waveguides and flexibly adjustable pinching antennas, PASS establishes short-distance line-of-sight (LoS) links while effectively mitigating the significant path loss and potential signal blockage, making it a promising solution for high-frequency MEC systems. We formulate a network latency minimization problem to joint optimize uplink PASS beamforming and task offloading. The resulting problem is modeled as a Markov decision process (MDP) and solved via the deep reinforcement learning (DRL) method. To address the instability introduced by the $\max$ operator in the objective function, we propose a load balancing-aware proximal policy optimization (LBPPO) algorithm. LBPPO incorporates both node-level and waveguide-level load balancing information into the policy design, maintaining computational and transmission delay equilibrium, respectively. Simulation results demonstrate that the proposed PASS-enhanced MEC with adaptive uplink PASS beamforming exhibit stronger convergence capability than fixed-PA baselines and conventional MIMO-assisted MEC, especially in scenarios with a large number of UEs or high transmit power. 

---
# Is Your Prompt Poisoning Code? Defect Induction Rates and Security Mitigation Strategies 

**Authors**: Bin Wang, YiLu Zhong, MiDi Wan, WenJie Yu, YuanBing Ouyang, Yenan Huang, Hui Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.22944)  

**Abstract**: Large language models (LLMs) have become indispensable for automated code generation, yet the quality and security of their outputs remain a critical concern. Existing studies predominantly concentrate on adversarial attacks or inherent flaws within the models. However, a more prevalent yet underexplored issue concerns how the quality of a benign but poorly formulated prompt affects the security of the generated code. To investigate this, we first propose an evaluation framework for prompt quality encompassing three key dimensions: goal clarity, information completeness, and logical consistency. Based on this framework, we construct and publicly release CWE-BENCH-PYTHON, a large-scale benchmark dataset containing tasks with prompts categorized into four distinct levels of normativity (L0-L3). Extensive experiments on multiple state-of-the-art LLMs reveal a clear correlation: as prompt normativity decreases, the likelihood of generating insecure code consistently and markedly increases. Furthermore, we demonstrate that advanced prompting techniques, such as Chain-of-Thought and Self-Correction, effectively mitigate the security risks introduced by low-quality prompts, substantially improving code safety. Our findings highlight that enhancing the quality of user prompts constitutes a critical and effective strategy for strengthening the security of AI-generated code. 

---
# Robust Uncertainty Quantification for Self-Evolving Large Language Models via Continual Domain Pretraining 

**Authors**: Xiaofan Zhou, Lu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.22931)  

**Abstract**: Continual Learning (CL) is essential for enabling self-evolving large language models (LLMs) to adapt and remain effective amid rapid knowledge growth. Yet, despite its importance, little attention has been given to establishing statistical reliability guarantees for LLMs under CL, particularly in the setting of continual domain pretraining (CDP). Conformal Prediction (CP) has shown promise in offering correctness guarantees for LLMs, but it faces major challenges in CDP: testing data often stems from unknown or shifting domain distributions, under which CP may no longer provide valid guarantees. Moreover, when high coverage is required, CP can yield excessively large prediction sets for unanswerable queries, reducing informativeness. To address these challenges, we introduce an adaptive rejection and non-exchangeable CP framework. Our method first estimates the distribution of questions across domains in the test set using transformer-based clustering, then reweights or resamples the calibration data accordingly. Building on this, adaptive rejection CP allows the LLM to selectively abstain from answering when its confidence or competence shifts significantly. Extensive experiments demonstrate that our framework enhances both the effectiveness and reliability of CP under CDP scenarios. Our code is available at: this https URL 

---
# Gen-LangSplat: Generalized Language Gaussian Splatting with Pre-Trained Feature Compression 

**Authors**: Pranav Saxena  

**Link**: [PDF](https://arxiv.org/pdf/2510.22930)  

**Abstract**: Modeling open-vocabulary language fields in 3D is essential for intuitive human-AI interaction and querying within physical environments. State-of-the-art approaches, such as LangSplat, leverage 3D Gaussian Splatting to efficiently construct these language fields, encoding features distilled from high-dimensional models like CLIP. However, this efficiency is currently offset by the requirement to train a scene-specific language autoencoder for feature compression, introducing a costly, per-scene optimization bottleneck that hinders deployment scalability. In this work, we introduce Gen-LangSplat, that eliminates this requirement by replacing the scene-wise autoencoder with a generalized autoencoder, pre-trained extensively on the large-scale ScanNet dataset. This architectural shift enables the use of a fixed, compact latent space for language features across any new scene without any scene-specific training. By removing this dependency, our entire language field construction process achieves a efficiency boost while delivering querying performance comparable to, or exceeding, the original LangSplat method. To validate our design choice, we perform a thorough ablation study empirically determining the optimal latent embedding dimension and quantifying representational fidelity using Mean Squared Error and cosine similarity between the original and reprojected 512-dimensional CLIP embeddings. Our results demonstrate that generalized embeddings can efficiently and accurately support open-vocabulary querying in novel 3D scenes, paving the way for scalable, real-time interactive 3D AI applications. 

---
# HyPerNav: Hybrid Perception for Object-Oriented Navigation in Unknown Environment 

**Authors**: Zecheng Yin, Hao Zhao, Zhen Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.22917)  

**Abstract**: Objective-oriented navigation(ObjNav) enables robot to navigate to target object directly and autonomously in an unknown environment. Effective perception in navigation in unknown environment is critical for autonomous robots. While egocentric observations from RGB-D sensors provide abundant local information, real-time top-down maps offer valuable global context for ObjNav. Nevertheless, the majority of existing studies focus on a single source, seldom integrating these two complementary perceptual modalities, despite the fact that humans naturally attend to both. With the rapid advancement of Vision-Language Models(VLMs), we propose Hybrid Perception Navigation (HyPerNav), leveraging VLMs' strong reasoning and vision-language understanding capabilities to jointly perceive both local and global information to enhance the effectiveness and intelligence of navigation in unknown environments. In both massive simulation evaluation and real-world validation, our methods achieved state-of-the-art performance against popular baselines. Benefiting from hybrid perception approach, our method captures richer cues and finds the objects more effectively, by simultaneously leveraging information understanding from egocentric observations and the top-down map. Our ablation study further proved that either of the hybrid perception contributes to the navigation performance. 

---
# Rethinking Inference Placement for Deep Learning across Edge and Cloud Platforms: A Multi-Objective Optimization Perspective and Future Directions 

**Authors**: Zongshun Zhang, Ibrahim Matta  

**Link**: [PDF](https://arxiv.org/pdf/2510.22909)  

**Abstract**: Edge intelligent applications like VR/AR and language model based chatbots have become widespread with the rapid expansion of IoT and mobile devices. However, constrained edge devices often cannot serve the increasingly large and complex deep learning (DL) models. To mitigate these challenges, researchers have proposed optimizing and offloading partitions of DL models among user devices, edge servers, and the cloud. In this setting, users can take advantage of different services to support their intelligent applications. For example, edge resources offer low response latency. In contrast, cloud platforms provide low monetary cost computation resources for computation-intensive workloads. However, communication between DL model partitions can introduce transmission bottlenecks and pose risks of data leakage. Recent research aims to balance accuracy, computation delay, transmission delay, and privacy concerns. They address these issues with model compression, model distillation, transmission compression, and model architecture adaptations, including internal classifiers. This survey contextualizes the state-of-the-art model offloading methods and model adaptation techniques by studying their implication to a multi-objective optimization comprising inference latency, data privacy, and resource monetary cost. 

---
# Language Server CLI Empowers Language Agents with Process Rewards 

**Authors**: Yifan Zhang, Lanser Contributors  

**Link**: [PDF](https://arxiv.org/pdf/2510.22907)  

**Abstract**: Large language models routinely hallucinate APIs and mislocalize edits, while language servers compute verified, IDE-grade facts about real code. We present Lanser-CLI, a CLI-first orchestration layer that pins and mediates a Language Server Protocol (LSP) server for coding agents and CI, exposing deterministic, replayable workflows. Our position is that language servers provide not only structural information (definitions, references, types, diagnostics) but also an actionable process reward: machine-checked, step-wise signals that align an agent's planning loop with program reality. In this work, Lanser-CLI contributes: (i) a robust addressing scheme beyond brittle "file:line:col" via a Selector DSL (symbolic, AST-path, and content-anchored selectors) with a principled relocation algorithm; (ii) deterministic Analysis Bundles that normalize Language Server responses and capture environment/capability metadata with stable content hashes; (iii) a safety envelope for mutating operations (rename, code actions) with preview, workspace jails, and Git-aware, transactional apply; and (iv) a process-reward functional derived from Language Server facts (diagnostic deltas, disambiguation confidence, and safe-apply checks) that is computable online and replayable offline. We formalize determinism under frozen snapshots and establish a monotonicity property for the process reward, making it suitable for process supervision and counterfactual analysis. Project Page: this https URL 

---
# Learning Reconfigurable Representations for Multimodal Federated Learning with Missing Data 

**Authors**: Duong M. Nguyen, Trong Nghia Hoang, Thanh Trung Huynh, Quoc Viet Hung Nguyen, Phi Le Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2510.22880)  

**Abstract**: Multimodal federated learning in real-world settings often encounters incomplete and heterogeneous data across clients. This results in misaligned local feature representations that limit the effectiveness of model aggregation. Unlike prior work that assumes either differing modality sets without missing input features or a shared modality set with missing features across clients, we consider a more general and realistic setting where each client observes a different subset of modalities and might also have missing input features within each modality. To address the resulting misalignment in learned representations, we propose a new federated learning framework featuring locally adaptive representations based on learnable client-side embedding controls that encode each client's data-missing patterns.
These embeddings serve as reconfiguration signals that align the globally aggregated representation with each client's local context, enabling more effective use of shared information. Furthermore, the embedding controls can be algorithmically aggregated across clients with similar data-missing patterns to enhance the robustness of reconfiguration signals in adapting the global representation. Empirical results on multiple federated multimodal benchmarks with diverse data-missing patterns across clients demonstrate the efficacy of the proposed method, achieving up to 36.45\% performance improvement under severe data incompleteness. The method is also supported by a theoretical analysis with an explicit performance bound that matches our empirical observations. Our source codes are provided at this https URL 

---
# Batch Speculative Decoding Done Right 

**Authors**: Ranran Haoran Zhang, Soumik Dey, Ashirbad Mishra, Hansi Wu, Binbin Li, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.22876)  

**Abstract**: Speculative decoding speeds up LLM inference by using a small draft model to propose multiple tokens that a target model verifies in parallel. Extending this idea to batches is essential for production serving, but it introduces the ragged tensor problem: sequences in the same batch accept different numbers of draft tokens, breaking right-alignment and corrupting position IDs, attention masks, and KV-cache state. We show that several existing batch implementations violate output equivalence-the fundamental requirement that speculative decoding must produce identical token sequences to standard autoregressive generation. These violations occur precisely due to improper handling of the ragged tensor problem. In response, we (1) characterize the synchronization requirements that guarantee correctness, (2) present a correctness-first batch speculative decoding EQSPEC that exposes realignment as consuming 40% of overhead, and (3) introduce EXSPEC, which maintains a sliding pool of sequences and dynamically forms same-length groups, to reduce the realignment overhead while preserving per-sequence speculative speedups. On the SpecBench dataset, across Vicuna-7B/68M, Qwen3-8B/0.6B, and GLM-4-9B/0.6B target/draft pairs, our approach achieves up to 3$\times$ throughput improvement at batch size 8 compared to batch size 1, with efficient scaling through batch size 8, while maintaining 95% output equivalence. Our method requires no custom kernels and integrates cleanly with existing inference stacks. Our code is available at this https URL. 

---
# Long-Term PM2.5 Forecasting Using a DTW-Enhanced CNN-GRU Model 

**Authors**: Amirali Ataee Naeini, Arshia Ataee Naeini, Fatemeh Karami Mohammadi, Omid Ghaffarpasand  

**Link**: [PDF](https://arxiv.org/pdf/2510.22863)  

**Abstract**: Reliable long-term forecasting of PM2.5 concentrations is critical for public health early-warning systems, yet existing deep learning approaches struggle to maintain prediction stability beyond 48 hours, especially in cities with sparse monitoring networks. This paper presents a deep learning framework that combines Dynamic Time Warping (DTW) for intelligent station similarity selection with a CNN-GRU architecture to enable extended-horizon PM2.5 forecasting in Isfahan, Iran, a city characterized by complex pollution dynamics and limited monitoring coverage. Unlike existing approaches that rely on computationally intensive transformer models or external simulation tools, our method integrates three key innovations: (i) DTW-based historical sampling to identify similar pollution patterns across peer stations, (ii) a lightweight CNN-GRU architecture augmented with meteorological features, and (iii) a scalable design optimized for sparse networks. Experimental validation using multi-year hourly data from eight monitoring stations demonstrates superior performance compared to state-of-the-art deep learning methods, achieving R2 = 0.91 for 24-hour forecasts. Notably, this is the first study to demonstrate stable 10-day PM2.5 forecasting (R2 = 0.73 at 240 hours) without performance degradation, addressing critical early-warning system requirements. The framework's computational efficiency and independence from external tools make it particularly suitable for deployment in resource-constrained urban environments. 

---
# Guardian: Decoupling Exploration from Safety in Reinforcement Learning 

**Authors**: Kaitong Cai, Jusheng Zhang, Jing Yang, Keze Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.22859)  

**Abstract**: Hybrid offline--online reinforcement learning (O2O RL) promises both sample efficiency and robust exploration, but suffers from instability due to distribution shift between offline and online data. We introduce RLPD-GX, a framework that decouples policy optimization from safety enforcement: a reward-seeking learner explores freely, while a projection-based guardian guarantees rule-consistent execution and safe value backups. This design preserves the exploratory value of online interactions without collapsing to conservative policies. To further stabilize training, we propose dynamic curricula that gradually extend temporal horizons and anneal offline--online data mixing. We prove convergence via a contraction property of the guarded Bellman operator, and empirically show state-of-the-art performance on Atari-100k, achieving a normalized mean score of 3.02 (+45\% over prior hybrid methods) with stronger safety and stability. Beyond Atari, ablations demonstrate consistent gains across safety-critical and long-horizon tasks, underscoring the generality of our design. Extensive and comprehensive results highlight decoupled safety enforcement as a simple yet principled route to robust O2O RL, suggesting a broader paradigm for reconciling exploration and safety in reinforcement learning. 

---
# Encoder-Decoder Diffusion Language Models for Efficient Training and Inference 

**Authors**: Marianne Arriola, Yair Schiff, Hao Phung, Aaron Gokaslan, Volodymyr Kuleshov  

**Link**: [PDF](https://arxiv.org/pdf/2510.22852)  

**Abstract**: Discrete diffusion models enable parallel token sampling for faster inference than autoregressive approaches. However, prior diffusion models use a decoder-only architecture, which requires sampling algorithms that invoke the full network at every denoising step and incur high computational cost. Our key insight is that discrete diffusion models perform two types of computation: 1) representing clean tokens and 2) denoising corrupted tokens, which enables us to use separate modules for each task. We propose an encoder-decoder architecture to accelerate discrete diffusion inference, which relies on an encoder to represent clean tokens and a lightweight decoder to iteratively refine a noised sequence. We also show that this architecture enables faster training of block diffusion models, which partition sequences into blocks for better quality and are commonly used in diffusion language model inference. We introduce a framework for Efficient Encoder-Decoder Diffusion (E2D2), consisting of an architecture with specialized training and sampling algorithms, and we show that E2D2 achieves superior trade-offs between generation quality and inference throughput on summarization, translation, and mathematical reasoning tasks. We provide the code, model weights, and blog post on the project page: this https URL 

---
# Semantic Surgery: Zero-Shot Concept Erasure in Diffusion Models 

**Authors**: Lexiang Xiong, Chengyu Liu, Jingwen Ye, Yan Liu, Yuecong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.22851)  

**Abstract**: Concept erasure in text-to-image diffusion models is crucial for mitigating harmful content, yet existing methods often compromise generative quality. We introduce Semantic Surgery, a novel training-free, zero-shot framework for concept erasure that operates directly on text embeddings before the diffusion process. It dynamically estimates the presence of target concepts in a prompt and performs a calibrated vector subtraction to neutralize their influence at the source, enhancing both erasure completeness and locality. The framework includes a Co-Occurrence Encoding module for robust multi-concept erasure and a visual feedback loop to address latent concept persistence. As a training-free method, Semantic Surgery adapts dynamically to each prompt, ensuring precise interventions. Extensive experiments on object, explicit content, artistic style, and multi-celebrity erasure tasks show our method significantly outperforms state-of-the-art approaches. We achieve superior completeness and robustness while preserving locality and image quality (e.g., 93.58 H-score in object erasure, reducing explicit content to just 1 instance, and 8.09 H_a in style erasure with no quality degradation). This robustness also allows our framework to function as a built-in threat detection system, offering a practical solution for safer text-to-image generation. 

---
# Once Upon an Input: Reasoning via Per-Instance Program Synthesis 

**Authors**: Adam Stein, Neelay Velingker, Mayur Naik, Eric Wong  

**Link**: [PDF](https://arxiv.org/pdf/2510.22849)  

**Abstract**: Large language models (LLMs) excel at zero-shot inference but continue to struggle with complex, multi-step reasoning. Recent methods that augment LLMs with intermediate reasoning steps such as Chain of Thought (CoT) and Program of Thought (PoT) improve performance but often produce undesirable solutions, especially in algorithmic domains. We introduce Per-Instance Program Synthesis (PIPS), a method that generates and refines programs at the instance-level using structural feedback without relying on task-specific guidance or explicit test cases. To further improve performance, PIPS incorporates a confidence metric that dynamically chooses between direct inference and program synthesis on a per-instance basis. Experiments across three frontier LLMs and 30 benchmarks including all tasks of Big Bench Extra Hard (BBEH), visual question answering tasks, relational reasoning tasks, and mathematical reasoning tasks show that PIPS improves the absolute harmonic mean accuracy by up to 8.6% and 9.4% compared to PoT and CoT respectively, and reduces undesirable program generations by 65.1% on the algorithmic tasks compared to PoT with Gemini-2.0-Flash. 

---
# LLM-based Fusion of Multi-modal Features for Commercial Memorability Prediction 

**Authors**: Aleksandar Pramov  

**Link**: [PDF](https://arxiv.org/pdf/2510.22829)  

**Abstract**: This paper addresses the prediction of commercial (brand) memorability as part of "Subtask 2: Commercial/Ad Memorability" within the "Memorability: Predicting movie and commercial memorability" task at the MediaEval 2025 workshop competition. We propose a multimodal fusion system with a Gemma-3 LLM backbone that integrates pre-computed visual (ViT) and textual (E5) features by multi-modal projections. The model is adapted using Low-Rank Adaptation (LoRA). A heavily-tuned ensemble of gradient boosted trees serves as a baseline. A key contribution is the use of LLM-generated rationale prompts, grounded in expert-derived aspects of memorability, to guide the fusion model. The results demonstrate that the LLM-based system exhibits greater robustness and generalization performance on the final test set, compared to the baseline.
The paper's codebase can be found at this https URL 

---
# Cross-Lingual Stability and Bias in Instruction-Tuned Language Models for Humanitarian NLP 

**Authors**: Poli Nemkova, Amrit Adhikari, Matthew Pearson, Vamsi Krishna Sadu, Mark V. Albert  

**Link**: [PDF](https://arxiv.org/pdf/2510.22823)  

**Abstract**: Humanitarian organizations face a critical choice: invest in costly commercial APIs or rely on free open-weight models for multilingual human rights monitoring. While commercial systems offer reliability, open-weight alternatives lack empirical validation -- especially for low-resource languages common in conflict zones. This paper presents the first systematic comparison of commercial and open-weight large language models (LLMs) for human-rights-violation detection across seven languages, quantifying the cost-reliability trade-off facing resource-constrained organizations. Across 78,000 multilingual inferences, we evaluate six models -- four instruction-aligned (Claude-Sonnet-4, DeepSeek-V3, Gemini-Flash-2.0, GPT-4.1-mini) and two open-weight (LLaMA-3-8B, Mistral-7B) -- using both standard classification metrics and new measures of cross-lingual reliability: Calibration Deviation (CD), Decision Bias (B), Language Robustness Score (LRS), and Language Stability Score (LSS). Results show that alignment, not scale, determines stability: aligned models maintain near-invariant accuracy and balanced calibration across typologically distant and low-resource languages (e.g., Lingala, Burmese), while open-weight models exhibit significant prompt-language sensitivity and calibration drift. These findings demonstrate that multilingual alignment enables language-agnostic reasoning and provide practical guidance for humanitarian organizations balancing budget constraints with reliability in multilingual deployment. 

---
# Air Quality Prediction Using LOESS-ARIMA and Multi-Scale CNN-BiLSTM with Residual-Gated Attention 

**Authors**: Soham Pahari, Sandeep Chand Kumain  

**Link**: [PDF](https://arxiv.org/pdf/2510.22818)  

**Abstract**: Air pollution remains a critical environmental and public health concern in Indian megacities such as Delhi, Kolkata, and Mumbai, where sudden spikes in pollutant levels challenge timely intervention. Accurate Air Quality Index (AQI) forecasting is difficult due to the coexistence of linear trends, seasonal variations, and volatile nonlinear patterns. This paper proposes a hybrid forecasting framework that integrates LOESS decomposition, ARIMA modeling, and a multi-scale CNN-BiLSTM network with a residual-gated attention mechanism. The LOESS step separates the AQI series into trend, seasonal, and residual components, with ARIMA modeling the smooth components and the proposed deep learning module capturing multi-scale volatility in the residuals. Model hyperparameters are tuned via the Unified Adaptive Multi-Stage Metaheuristic Optimizer (UAMMO), combining multiple optimization strategies for efficient convergence. Experiments on 2021-2023 AQI datasets from the Central Pollution Control Board show that the proposed method consistently outperforms statistical, deep learning, and hybrid baselines across PM2.5, O3, CO, and NOx in three major cities, achieving up to 5-8% lower MSE and higher R^2 scores (>0.94) for all pollutants. These results demonstrate the framework's robustness, sensitivity to sudden pollution events, and applicability to urban air quality management. 

---
# A Theory of the Mechanics of Information: Generalization Through Measurement of Uncertainty (Learning is Measuring) 

**Authors**: Christopher J. Hazard, Michael Resnick, Jacob Beel, Jack Xia, Cade Mack, Dominic Glennie, Matthew Fulp, David Maze, Andrew Bassett, Martin Koistinen  

**Link**: [PDF](https://arxiv.org/pdf/2510.22809)  

**Abstract**: Traditional machine learning relies on explicit models and domain assumptions, limiting flexibility and interpretability. We introduce a model-free framework using surprisal (information theoretic uncertainty) to directly analyze and perform inferences from raw data, eliminating distribution modeling, reducing bias, and enabling efficient updates including direct edits and deletion of training data. By quantifying relevance through uncertainty, the approach enables generalizable inference across tasks including generative inference, causal discovery, anomaly detection, and time series forecasting. It emphasizes traceability, interpretability, and data-driven decision making, offering a unified, human-understandable framework for machine learning, and achieves at or near state-of-the-art performance across most common machine learning tasks. The mathematical foundations create a ``physics'' of information, which enable these techniques to apply effectively to a wide variety of complex data types, including missing data. Empirical results indicate that this may be a viable alternative path to neural networks with regard to scalable machine learning and artificial intelligence that can maintain human understandability of the underlying mechanics. 

---
# Collaborative LLM Agents for C4 Software Architecture Design Automation 

**Authors**: Kamil Szczepanik, Jarosław A. Chudziak  

**Link**: [PDF](https://arxiv.org/pdf/2510.22787)  

**Abstract**: Software architecture design is a fundamental part of creating every software system. Despite its importance, producing a C4 software architecture model, the preferred notation for such architecture, remains manual and time-consuming. We introduce an LLM-based multi-agent system that automates this task by simulating a dialogue between role-specific experts who analyze requirements and generate the Context, Container, and Component views of the C4 model. Quality is assessed with a hybrid evaluation framework: deterministic checks for structural and syntactic integrity and C4 rule consistency, plus semantic and qualitative scoring via an LLM-as-a-Judge approach. Tested on five canonical system briefs, the workflow demonstrates fast C4 model creation, sustains high compilation success, and delivers semantic fidelity. A comparison of four state-of-the-art LLMs shows different strengths relevant to architectural design. This study contributes to automated software architecture design and its evaluation methods. 

---
# PIP-LLM: Integrating PDDL-Integer Programming with LLMs for Coordinating Multi-Robot Teams Using Natural Language 

**Authors**: Guangyao Shi, Yuwei Wu, Vijay Kumar, Gaurav S. Sukhatme  

**Link**: [PDF](https://arxiv.org/pdf/2510.22784)  

**Abstract**: Enabling robot teams to execute natural language commands requires translating high-level instructions into feasible, efficient multi-robot plans. While Large Language Models (LLMs) combined with Planning Domain Description Language (PDDL) offer promise for single-robot scenarios, existing approaches struggle with multi-robot coordination due to brittle task decomposition, poor scalability, and low coordination efficiency.
We introduce PIP-LLM, a language-based coordination framework that consists of PDDL-based team-level planning and Integer Programming (IP) based robot-level planning. PIP-LLMs first decomposes the command by translating the command into a team-level PDDL problem and solves it to obtain a team-level plan, abstracting away robot assignment. Each team-level action represents a subtask to be finished by the team. Next, this plan is translated into a dependency graph representing the subtasks' dependency structure. Such a dependency graph is then used to guide the robot-level planning, in which each subtask node will be formulated as an IP-based task allocation problem, explicitly optimizing travel costs and workload while respecting robot capabilities and user-defined constraints. This separation of planning from assignment allows PIP-LLM to avoid the pitfalls of syntax-based decomposition and scale to larger teams. Experiments across diverse tasks show that PIP-LLM improves plan success rate, reduces maximum and average travel costs, and achieves better load balancing compared to state-of-the-art baselines. 

---
# Beyond Semantics: How Temporal Biases Shape Retrieval in Transformer and State-Space Models 

**Authors**: Anooshka Bajaj, Deven Mahesh Mistry, Sahaj Singh Maini, Yash Aggarwal, Zoran Tiganj  

**Link**: [PDF](https://arxiv.org/pdf/2510.22752)  

**Abstract**: In-context learning is governed by both temporal and semantic relationships, shaping how Large Language Models (LLMs) retrieve contextual information. Analogous to human episodic memory, where the retrieval of specific events is enabled by separating events that happened at different times, this work probes the ability of various pretrained LLMs, including transformer and state-space models, to differentiate and retrieve temporally separated events. Specifically, we prompted models with sequences containing multiple presentations of the same token, which reappears at the sequence end. By fixing the positions of these repeated tokens and permuting all others, we removed semantic confounds and isolated temporal effects on next-token prediction. Across diverse sequences, models consistently placed the highest probabilities on tokens following a repeated token, but with a notable bias for those nearest the beginning or end of the input. An ablation experiment linked this phenomenon in transformers to induction heads. Extending the analysis to unique semantic contexts with partial overlap further demonstrated that memories embedded in the middle of a prompt are retrieved less reliably. Despite architectural differences, state-space and transformer models showed comparable temporal biases. Our findings deepen the understanding of temporal biases in in-context learning and offer an illustration of how these biases can enable temporal separation and episodic retrieval. 

---
# Low-Resource Dialect Adaptation of Large Language Models: A French Dialect Case-Study 

**Authors**: Eeham Khan, Firas Saidani, Owen Van Esbroeck, Richard Khoury, Leila Kosseim  

**Link**: [PDF](https://arxiv.org/pdf/2510.22747)  

**Abstract**: Despite the widespread adoption of large language models (LLMs), their strongest capabilities remain largely confined to a small number of high-resource languages for which there is abundant training data. Recently, continual pre-training (CPT) has emerged as a means to fine-tune these models to low-resource regional dialects. In this paper, we study the use of CPT for dialect learning under tight data and compute budgets. Using low-rank adaptation (LoRA) and compute-efficient continual pre-training, we adapt three LLMs to the Québec French dialect using a very small dataset and benchmark them on the COLE suite. Our experiments demonstrate an improvement on the minority dialect benchmarks with minimal regression on the prestige language benchmarks with under 1% of model parameters updated. Analysis of the results demonstrate that gains are highly contingent on corpus composition. These findings indicate that CPT with parameter-efficient fine-tuning (PEFT) can narrow the dialect gap by providing cost-effective and sustainable language resource creation, expanding high-quality LLM access to minority linguistic communities. We release the first Québec French LLMs on HuggingFace. 

---
# Policies over Poses: Reinforcement Learning based Distributed Pose-Graph Optimization for Multi-Robot SLAM 

**Authors**: Sai Krishna Ghanta, Ramviyas Parasuraman  

**Link**: [PDF](https://arxiv.org/pdf/2510.22740)  

**Abstract**: We consider the distributed pose-graph optimization (PGO) problem, which is fundamental in accurate trajectory estimation in multi-robot simultaneous localization and mapping (SLAM). Conventional iterative approaches linearize a highly non-convex optimization objective, requiring repeated solving of normal equations, which often converge to local minima and thus produce suboptimal estimates. We propose a scalable, outlier-robust distributed planar PGO framework using Multi-Agent Reinforcement Learning (MARL). We cast distributed PGO as a partially observable Markov game defined on local pose-graphs, where each action refines a single edge's pose estimate. A graph partitioner decomposes the global pose graph, and each robot runs a recurrent edge-conditioned Graph Neural Network (GNN) encoder with adaptive edge-gating to denoise noisy edges. Robots sequentially refine poses through a hybrid policy that utilizes prior action memory and graph embeddings. After local graph correction, a consensus scheme reconciles inter-robot disagreements to produce a globally consistent estimate. Our extensive evaluations on a comprehensive suite of synthetic and real-world datasets demonstrate that our learned MARL-based actors reduce the global objective by an average of 37.5% more than the state-of-the-art distributed PGO framework, while enhancing inference efficiency by at least 6X. We also demonstrate that actor replication allows a single learned policy to scale effortlessly to substantially larger robot teams without any retraining. Code is publicly available at this https URL. 

---
# REVISION:Reflective Intent Mining and Online Reasoning Auxiliary for E-commerce Visual Search System Optimization 

**Authors**: Yiwen Tang, Qiuyu Zhao, Zenghui Sun, Jinsong Lan, Xiaoyong Zhu, Bo Zheng, Kaifu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.22739)  

**Abstract**: In Taobao e-commerce visual search, user behavior analysis reveals a large proportion of no-click requests, suggesting diverse and implicit user intents. These intents are expressed in various forms and are difficult to mine and discover, thereby leading to the limited adaptability and lag in platform strategies. This greatly restricts users' ability to express diverse intents and hinders the scalability of the visual search system. This mismatch between user implicit intent expression and system response defines the User-SearchSys Intent Discrepancy. To alleviate the issue, we propose a novel framework REVISION. This framework integrates offline reasoning mining with online decision-making and execution, enabling adaptive strategies to solve implicit user demands. In the offline stage, we construct a periodic pipeline to mine discrepancies from historical no-click requests. Leveraging large models, we analyze implicit intent factors and infer optimal suggestions by jointly reasoning over query and product metadata. These inferred suggestions serve as actionable insights for refining platform strategies. In the online stage, REVISION-R1-3B, trained on the curated offline data, performs holistic analysis over query images and associated historical products to generate optimization plans and adaptively schedule strategies across the search pipeline. Our framework offers a streamlined paradigm for integrating large models with traditional search systems, enabling end-to-end intelligent optimization across information aggregation and user interaction. Experimental results demonstrate that our approach improves the efficiency of implicit intent mining from large-scale search logs and significantly reduces the no-click rate. 

---
# $\text{E}^2\text{Rank}$: Your Text Embedding can Also be an Effective and Efficient Listwise Reranker 

**Authors**: Qi Liu, Yanzhao Zhang, Mingxin Li, Dingkun Long, Pengjun Xie, Jiaxin Mao  

**Link**: [PDF](https://arxiv.org/pdf/2510.22733)  

**Abstract**: Text embedding models serve as a fundamental component in real-world search applications. By mapping queries and documents into a shared embedding space, they deliver competitive retrieval performance with high efficiency. However, their ranking fidelity remains limited compared to dedicated rerankers, especially recent LLM-based listwise rerankers, which capture fine-grained query-document and document-document interactions. In this paper, we propose a simple yet effective unified framework $\text{E}^2\text{Rank}$, means Efficient Embedding-based Ranking (also means Embedding-to-Rank), which extends a single text embedding model to perform both high-quality retrieval and listwise reranking through continued training under a listwise ranking objective, thereby achieving strong effectiveness with remarkable efficiency. By applying cosine similarity between the query and document embeddings as a unified ranking function, the listwise ranking prompt, which is constructed from the original query and its candidate documents, serves as an enhanced query enriched with signals from the top-K documents, akin to pseudo-relevance feedback (PRF) in traditional retrieval models. This design preserves the efficiency and representational quality of the base embedding model while significantly improving its reranking performance. Empirically, $\textrm{E}^2\text{Rank}$ achieves state-of-the-art results on the BEIR reranking benchmark and demonstrates competitive performance on the reasoning-intensive BRIGHT benchmark, with very low reranking latency. We also show that the ranking training process improves embedding performance on the MTEB benchmark. Our findings indicate that a single embedding model can effectively unify retrieval and reranking, offering both computational efficiency and competitive ranking accuracy. 

---
# ATLAS: Actor-Critic Task-Completion with Look-ahead Action Simulation 

**Authors**: Jiali Cheng, Anjishnu Kumar, Roshan Lal, Rishi Rajasekaran, Hani Ramezani, Omar Zia Khan, Oleg Rokhlenko, Sunny Chiu-Webster, Gang Hua, Hadi Amiri  

**Link**: [PDF](https://arxiv.org/pdf/2510.22732)  

**Abstract**: We observe that current state-of-the-art web-agents are unable to effectively adapt to new environments without neural network fine-tuning, without which they produce inefficient execution plans due to a lack of awareness of the structure and dynamics of the new environment. To address this limitation, we introduce ATLAS (Actor-Critic Task-completion with Look-ahead Action Simulation), a memory-augmented agent that is able to make plans grounded in a model of the environment by simulating the consequences of those actions in cognitive space. Our agent starts by building a "cognitive map" by performing a lightweight curiosity driven exploration of the environment. The planner proposes candidate actions; the simulator predicts their consequences in cognitive space; a critic analyzes the options to select the best roll-out and update the original plan; and a browser executor performs the chosen action. On the WebArena-Lite Benchmark, we achieve a 63% success rate compared to 53.9% success rate for the previously published state-of-the-art. Unlike previous systems, our modular architecture requires no website-specific LLM fine-tuning. Ablations show sizable drops without the world-model, hierarchical planner, and look-ahead-based replanner confirming their complementary roles within the design of our system 

---
# Step2Motion: Locomotion Reconstruction from Pressure Sensing Insoles 

**Authors**: Jose Luis Ponton, Eduardo Alvarado, Lin Geng Foo, Nuria Pelechano, Carlos Andujar, Marc Habermann  

**Link**: [PDF](https://arxiv.org/pdf/2510.22712)  

**Abstract**: Human motion is fundamentally driven by continuous physical interaction with the environment. Whether walking, running, or simply standing, the forces exchanged between our feet and the ground provide crucial insights for understanding and reconstructing human movement. Recent advances in wearable insole devices offer a compelling solution for capturing these forces in diverse, real-world scenarios. Sensor insoles pose no constraint on the users' motion (unlike mocap suits) and are unaffected by line-of-sight limitations (in contrast to optical systems). These qualities make sensor insoles an ideal choice for robust, unconstrained motion capture, particularly in outdoor environments. Surprisingly, leveraging these devices with recent motion reconstruction methods remains largely unexplored. Aiming to fill this gap, we present Step2Motion, the first approach to reconstruct human locomotion from multi-modal insole sensors. Our method utilizes pressure and inertial data-accelerations and angular rates-captured by the insoles to reconstruct human motion. We evaluate the effectiveness of our approach across a range of experiments to show its versatility for diverse locomotion styles, from simple ones like walking or jogging up to moving sideways, on tiptoes, slightly crouching, or dancing. 

---
# FlowCritic: Bridging Value Estimation with Flow Matching in Reinforcement Learning 

**Authors**: Shan Zhong, Shutong Ding, He Diao, Xiangyu Wang, Kah Chan Teh, Bei Peng  

**Link**: [PDF](https://arxiv.org/pdf/2510.22686)  

**Abstract**: Reliable value estimation serves as the cornerstone of reinforcement learning (RL) by evaluating long-term returns and guiding policy improvement, significantly influencing the convergence speed and final performance. Existing works improve the reliability of value function estimation via multi-critic ensembles and distributional RL, yet the former merely combines multi point estimation without capturing distributional information, whereas the latter relies on discretization or quantile regression, limiting the expressiveness of complex value distributions. Inspired by flow matching's success in generative modeling, we propose a generative paradigm for value estimation, named FlowCritic. Departing from conventional regression for deterministic value prediction, FlowCritic leverages flow matching to model value distributions and generate samples for value estimation. 

---
# TABL-ABM: A Hybrid Framework for Synthetic LOB Generation 

**Authors**: Ollie Olby, Rory Baggott, Namid Stillman  

**Link**: [PDF](https://arxiv.org/pdf/2510.22685)  

**Abstract**: The recent application of deep learning models to financial trading has heightened the need for high fidelity financial time series data. This synthetic data can be used to supplement historical data to train large trading models. The state-of-the-art models for the generative application often rely on huge amounts of historical data and large, complicated models. These models range from autoregressive and diffusion-based models through to architecturally simpler models such as the temporal-attention bilinear layer. Agent-based approaches to modelling limit order book dynamics can also recreate trading activity through mechanistic models of trader behaviours. In this work, we demonstrate how a popular agent-based framework for simulating intraday trading activity, the Chiarella model, can be combined with one of the most performant deep learning models for forecasting multi-variate time series, the TABL model. This forecasting model is coupled to a simulation of a matching engine with a novel method for simulating deleted order flow. Our simulator gives us the ability to test the generative abilities of the forecasting model using stylised facts. Our results show that this methodology generates realistic price dynamics however, when analysing deeper, parts of the markets microstructure are not accurately recreated, highlighting the necessity for including more sophisticated agent behaviors into the modeling framework to help account for tail events. 

---
# Uncertainty-Aware Autonomous Vehicles: Predicting the Road Ahead 

**Authors**: Shireen Kudukkil Manchingal, Armand Amaritei, Mihir Gohad, Maryam Sultana, Julian F. P. Kooij, Fabio Cuzzolin, Andrew Bradley  

**Link**: [PDF](https://arxiv.org/pdf/2510.22680)  

**Abstract**: Autonomous Vehicle (AV) perception systems have advanced rapidly in recent years, providing vehicles with the ability to accurately interpret their environment. Perception systems remain susceptible to errors caused by overly-confident predictions in the case of rare events or out-of-sample data. This study equips an autonomous vehicle with the ability to 'know when it is uncertain', using an uncertainty-aware image classifier as part of the AV software stack. Specifically, the study exploits the ability of Random-Set Neural Networks (RS-NNs) to explicitly quantify prediction uncertainty. Unlike traditional CNNs or Bayesian methods, RS-NNs predict belief functions over sets of classes, allowing the system to identify and signal uncertainty clearly in novel or ambiguous scenarios. The system is tested in a real-world autonomous racing vehicle software stack, with the RS-NN classifying the layout of the road ahead and providing the associated uncertainty of the prediction. Performance of the RS-NN under a range of road conditions is compared against traditional CNN and Bayesian neural networks, with the RS-NN achieving significantly higher accuracy and superior uncertainty calibration. This integration of RS-NNs into Robot Operating System (ROS)-based vehicle control pipeline demonstrates that predictive uncertainty can dynamically modulate vehicle speed, maintaining high-speed performance under confident predictions while proactively improving safety through speed reductions in uncertain scenarios. These results demonstrate the potential of uncertainty-aware neural networks - in particular RS-NNs - as a practical solution for safer and more robust autonomous driving. 

---
# LVD-GS: Gaussian Splatting SLAM for Dynamic Scenes via Hierarchical Explicit-Implicit Representation Collaboration Rendering 

**Authors**: Wenkai Zhu, Xu Li, Qimin Xu, Benwu Wang, Kun Wei, Yiming Peng, Zihang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.22669)  

**Abstract**: 3D Gaussian Splatting SLAM has emerged as a widely used technique for high-fidelity mapping in spatial intelligence. However, existing methods often rely on a single representation scheme, which limits their performance in large-scale dynamic outdoor scenes and leads to cumulative pose errors and scale ambiguity. To address these challenges, we propose \textbf{LVD-GS}, a novel LiDAR-Visual 3D Gaussian Splatting SLAM system. Motivated by the human chain-of-thought process for information seeking, we introduce a hierarchical collaborative representation module that facilitates mutual reinforcement for mapping optimization, effectively mitigating scale drift and enhancing reconstruction robustness. Furthermore, to effectively eliminate the influence of dynamic objects, we propose a joint dynamic modeling module that generates fine-grained dynamic masks by fusing open-world segmentation with implicit residual constraints, guided by uncertainty estimates from DINO-Depth features. Extensive evaluations on KITTI, nuScenes, and self-collected datasets demonstrate that our approach achieves state-of-the-art performance compared to existing methods. 

---
# SARCLIP: A Vision Language Foundation Model for Semantic Understanding and Target Recognition in SAR Imagery 

**Authors**: Qiwei Ma, Zhiyu Wang, Wang Liu, Xukun Lu, Bin Deng, Puhong Duan, Xudong Kang, Shutao Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.22665)  

**Abstract**: Synthetic Aperture Radar (SAR) has emerged as a crucial imaging modality due to its all-weather capabilities. While recent advancements in self-supervised learning and Masked Image Modeling (MIM) have paved the way for SAR foundation models, these approaches primarily focus on low-level visual features, often overlooking multimodal alignment and zero-shot target recognition within SAR imagery. To address this limitation, we construct SARCLIP-1M, a large-scale vision language dataset comprising over one million text-image pairs aggregated from existing datasets. We further introduce SARCLIP, the first vision language foundation model tailored for the SAR domain. Our SARCLIP model is trained using a contrastive vision language learning approach by domain transferring strategy, enabling it to bridge the gap between SAR imagery and textual descriptions. Extensive experiments on image-text retrieval and zero-shot classification tasks demonstrate the superior performance of SARCLIP in feature extraction and interpretation, significantly outperforming state-of-the-art foundation models and advancing the semantic understanding of SAR imagery. The code and datasets will be released soon. 

---
# Learning Without Augmenting: Unsupervised Time Series Representation Learning via Frame Projections 

**Authors**: Berken Utku Demirel, Christian Holz  

**Link**: [PDF](https://arxiv.org/pdf/2510.22655)  

**Abstract**: Self-supervised learning (SSL) has emerged as a powerful paradigm for learning representations without labeled data. Most SSL approaches rely on strong, well-established, handcrafted data augmentations to generate diverse views for representation learning. However, designing such augmentations requires domain-specific knowledge and implicitly imposes representational invariances on the model, which can limit generalization. In this work, we propose an unsupervised representation learning method that replaces augmentations by generating views using orthonormal bases and overcomplete frames. We show that embeddings learned from orthonormal and overcomplete spaces reside on distinct manifolds, shaped by the geometric biases introduced by representing samples in different spaces. By jointly leveraging the complementary geometry of these distinct manifolds, our approach achieves superior performance without artificially increasing data diversity through strong augmentations. We demonstrate the effectiveness of our method on nine datasets across five temporal sequence tasks, where signal-specific characteristics make data augmentations particularly challenging. Without relying on augmentation-induced diversity, our method achieves performance gains of up to 15--20\% over existing self-supervised approaches. Source code: this https URL 

---
# Variational Polya Tree 

**Authors**: Lu Xu, Tsai Hor Chan, Kwok Fai Lam, Lequan Yu, Guosheng Yin  

**Link**: [PDF](https://arxiv.org/pdf/2510.22651)  

**Abstract**: Density estimation is essential for generative modeling, particularly with the rise of modern neural networks. While existing methods capture complex data distributions, they often lack interpretability and uncertainty quantification. Bayesian nonparametric methods, especially the \polya tree, offer a robust framework that addresses these issues by accurately capturing function behavior over small intervals. Traditional techniques like Markov chain Monte Carlo (MCMC) face high computational complexity and scalability limitations, hindering the use of Bayesian nonparametric methods in deep learning. To tackle this, we introduce the variational \polya tree (VPT) model, which employs stochastic variational inference to compute posterior distributions. This model provides a flexible, nonparametric Bayesian prior that captures latent densities and works well with stochastic gradient optimization. We also leverage the joint distribution likelihood for a more precise variational posterior approximation than traditional mean-field methods. We evaluate the model performance on both real data and images, and demonstrate its competitiveness with other state-of-the-art deep density estimation methods. We also explore its ability in enhancing interpretability and uncertainty quantification. Code is available at this https URL. 

---
# A Critical Study on Tea Leaf Disease Detection using Deep Learning Techniques 

**Authors**: Nabajyoti Borah, Raju Moni Borah, Bandan Boruah, Purnendu Bikash Acharjee, Sajal Saha, Ripjyoti Hazarika  

**Link**: [PDF](https://arxiv.org/pdf/2510.22647)  

**Abstract**: The proposed solution is Deep Learning Technique that will be able classify three types of tea leaves diseases from which two diseases are caused by the pests and one due to pathogens (infectious organisms) and environmental conditions and also show the area damaged by a disease in leaves. Namely Red Rust, Helopeltis and Red spider mite respectively. In this paper we have evaluated two models namely SSD MobileNet V2 and Faster R-CNN ResNet50 V1 for the object detection. The SSD MobileNet V2 gave precision of 0.209 for IOU range of 0.50:0.95 with recall of 0.02 on IOU 0.50:0.95 and final mAP of 20.9%. While Faster R-CNN ResNet50 V1 has precision of 0.252 on IOU range of 0.50:0.95 and recall of 0.044 on IOU of 0.50:0.95 with a mAP of 25%, which is better than SSD. Also used Mask R-CNN for Object Instance Segmentation where we have implemented our custom method to calculate the damaged diseased portion of leaves. Keywords: Tea Leaf Disease, Deep Learning, Red Rust, Helopeltis and Red Spider Mite, SSD MobileNet V2, Faster R-CNN ResNet50 V1 and Mask RCNN. 

---
# Enhancing Graph Classification Robustness with Singular Pooling 

**Authors**: Sofiane Ennadir, Oleg Smirnov, Yassine Abbahaddou, Lele Cao, Johannes F. Lutzeyer  

**Link**: [PDF](https://arxiv.org/pdf/2510.22643)  

**Abstract**: Graph Neural Networks (GNNs) have achieved strong performance across a range of graph representation learning tasks, yet their adversarial robustness in graph classification remains underexplored compared to node classification. While most existing defenses focus on the message-passing component, this work investigates the overlooked role of pooling operations in shaping robustness. We present a theoretical analysis of standard flat pooling methods (sum, average and max), deriving upper bounds on their adversarial risk and identifying their vulnerabilities under different attack scenarios and graph structures. Motivated by these insights, we propose \textit{Robust Singular Pooling (RS-Pool)}, a novel pooling strategy that leverages the dominant singular vector of the node embedding matrix to construct a robust graph-level representation. We theoretically investigate the robustness of RS-Pool and interpret the resulting bound leading to improved understanding of our proposed pooling operator. While our analysis centers on Graph Convolutional Networks (GCNs), RS-Pool is model-agnostic and can be implemented efficiently via power iteration. Empirical results on real-world benchmarks show that RS-Pool provides better robustness than the considered pooling methods when subject to state-of-the-art adversarial attacks while maintaining competitive clean accuracy. Our code is publicly available at:\href{this https URL}{this https URL}. 

---
# FastVLM: Self-Speculative Decoding for Fast Vision-Language Model Inference 

**Authors**: Divya Jyoti Bajpai, Manjesh Kumar Hanawal  

**Link**: [PDF](https://arxiv.org/pdf/2510.22641)  

**Abstract**: Vision-language Models (VLMs) have made significant strides in visual understanding and query response generation, but often face challenges of high computational cost and inference latency due to autoregressive decoding. In this work, we introduce an imitation-learning-based Self-Speculative Decoding (SSD) framework, named FastVLM, to address these limitations. Our approach employs a lightweight draft model for token generation in an autoregressive manner, while a full model verifies these tokens non-autoregressively. Accepted tokens proceed seamlessly, while rejected tokens are corrected by the full model and used to guide the draft model's refinement. Through an imitation network, FastVLM enhances the draft model by integrating deeper level insights from the full model's architecture. Also, it maintains the performance integrity of the full model while training the draft model, achieving a balance between efficiency and accuracy. Our method speeds up the inference process by 1.55-1.85x as compared to the final layer with minimal loss in performance. 

---
# Integrating Linguistics and AI: Morphological Analysis and Corpus development of Endangered Toto Language of West Bengal 

**Authors**: Ambalika Guha, Sajal Saha, Debanjan Ballav, Soumi Mitra, Hritwick Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2510.22629)  

**Abstract**: Preserving linguistic diversity is necessary as every language offers a distinct perspective on the world. There have been numerous global initiatives to preserve endangered languages through documentation. This paper is a part of a project which aims to develop a trilingual (Toto-Bangla-English) language learning application to digitally archive and promote the endangered Toto language of West Bengal, India. This application, designed for both native Toto speakers and non-native learners, aims to revitalize the language by ensuring accessibility and usability through Unicode script integration and a structured language corpus. The research includes detailed linguistic documentation collected via fieldwork, followed by the creation of a morpheme-tagged, trilingual corpus used to train a Small Language Model (SLM) and a Transformer-based translation engine. The analysis covers inflectional morphology such as person-number-gender agreement, tense-aspect-mood distinctions, and case marking, alongside derivational strategies that reflect word-class changes. Script standardization and digital literacy tools were also developed to enhance script usage. The study offers a sustainable model for preserving endangered languages by incorporating traditional linguistic methodology with AI. This bridge between linguistic research with technological innovation highlights the value of interdisciplinary collaboration for community-based language revitalization. 

---
# Sentra-Guard: A Multilingual Human-AI Framework for Real-Time Defense Against Adversarial LLM Jailbreaks 

**Authors**: Md. Mehedi Hasan, Ziaur Rahman, Rafid Mostafiz, Md. Abir Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2510.22628)  

**Abstract**: This paper presents a real-time modular defense system named Sentra-Guard. The system detects and mitigates jailbreak and prompt injection attacks targeting large language models (LLMs). The framework uses a hybrid architecture with FAISS-indexed SBERT embedding representations that capture the semantic meaning of prompts, combined with fine-tuned transformer classifiers, which are machine learning models specialized for distinguishing between benign and adversarial language inputs. It identifies adversarial prompts in both direct and obfuscated attack vectors. A core innovation is the classifier-retriever fusion module, which dynamically computes context-aware risk scores that estimate how likely a prompt is to be adversarial based on its content and context. The framework ensures multilingual resilience with a language-agnostic preprocessing layer. This component automatically translates non-English prompts into English for semantic evaluation, enabling consistent detection across over 100 languages. The system includes a HITL feedback loop, where decisions made by the automated system are reviewed by human experts for continual learning and rapid adaptation under adversarial pressure. Sentra-Guard maintains an evolving dual-labeled knowledge base of benign and malicious prompts, enhancing detection reliability and reducing false positives. Evaluation results show a 99.96% detection rate (AUC = 1.00, F1 = 1.00) and an attack success rate (ASR) of only 0.004%. This outperforms leading baselines such as LlamaGuard-2 (1.3%) and OpenAI Moderation (3.7%). Unlike black-box approaches, Sentra-Guard is transparent, fine-tunable, and compatible with diverse LLM backends. Its modular design supports scalable deployment in both commercial and open-source environments. The system establishes a new state-of-the-art in adversarial LLM defense. 

---
# Breaking Agent Backbones: Evaluating the Security of Backbone LLMs in AI Agents 

**Authors**: Julia Bazinska, Max Mathys, Francesco Casucci, Mateo Rojas-Carulla, Xander Davies, Alexandra Souly, Niklas Pfister  

**Link**: [PDF](https://arxiv.org/pdf/2510.22620)  

**Abstract**: AI agents powered by large language models (LLMs) are being deployed at scale, yet we lack a systematic understanding of how the choice of backbone LLM affects agent security. The non-deterministic sequential nature of AI agents complicates security modeling, while the integration of traditional software with AI components entangles novel LLM vulnerabilities with conventional security risks. Existing frameworks only partially address these challenges as they either capture specific vulnerabilities only or require modeling of complete agents. To address these limitations, we introduce threat snapshots: a framework that isolates specific states in an agent's execution flow where LLM vulnerabilities manifest, enabling the systematic identification and categorization of security risks that propagate from the LLM to the agent level. We apply this framework to construct the $\operatorname{b}^3$ benchmark, a security benchmark based on 194331 unique crowdsourced adversarial attacks. We then evaluate 31 popular LLMs with it, revealing, among other insights, that enhanced reasoning capabilities improve security, while model size does not correlate with security. We release our benchmark, dataset, and evaluation code to facilitate widespread adoption by LLM providers and practitioners, offering guidance for agent developers and incentivizing model developers to prioritize backbone security improvements. 

---
# Cross-Species Transfer Learning in Agricultural AI: Evaluating ZebraPose Adaptation for Dairy Cattle Pose Estimation 

**Authors**: Mackenzie Tapp, Sibi Chakravarthy Parivendan, Kashfia Sailunaz, Suresh Neethirajan  

**Link**: [PDF](https://arxiv.org/pdf/2510.22618)  

**Abstract**: Pose estimation serves as a cornerstone of computer vision for understanding animal posture, behavior, and welfare. Yet, agricultural applications remain constrained by the scarcity of large, annotated datasets for livestock, especially dairy cattle. This study evaluates the potential and limitations of cross-species transfer learning by adapting ZebraPose - a vision transformer-based model trained on synthetic zebra imagery - for 27-keypoint detection in dairy cows under real barn conditions. Using three configurations - a custom on-farm dataset (375 images, Sussex, New Brunswick, Canada), a subset of the APT-36K benchmark dataset, and their combination, we systematically assessed model accuracy and generalization across environments. While the combined model achieved promising performance (AP = 0.86, AR = 0.87, PCK 0.5 = 0.869) on in-distribution data, substantial generalization failures occurred when applied to unseen barns and cow populations. These findings expose the synthetic-to-real domain gap as a major obstacle to agricultural AI deployment and emphasize that morphological similarity between species is insufficient for cross-domain transfer. The study provides practical insights into dataset diversity, environmental variability, and computational constraints that influence real-world deployment of livestock monitoring systems. We conclude with a call for agriculture-first AI design, prioritizing farm-level realism, cross-environment robustness, and open benchmark datasets to advance trustworthy and scalable animal-centric technologies. 

---
# PerCoR: Evaluating Commonsense Reasoning in Persian via Multiple-Choice Sentence Completion 

**Authors**: Morteza Alikhani, Mohammadtaha Bagherifard, Erfan Zinvandi, Mehran Sarmadi  

**Link**: [PDF](https://arxiv.org/pdf/2510.22616)  

**Abstract**: We introduced PerCoR (Persian Commonsense Reasoning), the first large-scale Persian benchmark for commonsense reasoning. PerCoR contains 106K multiple-choice sentence-completion problems drawn from more than forty news, cultural, and other web sources. We introduce a novel conjunction-based segmentation strategy to generate coherent sentence-completion pairs, enabling broad topical and structural diversity. To create challenging distractors, we propose DRESS-AF (Distractor Ranking via Embedding Similarity Scoring and Adversarial Filtering), a generation-free adversarial filtering method that selects distractors from the pool of gold continuations while maximising model confusion. Human annotators score 89% on PerCoR, while OpenAI-o3 achieves the highest performance at 92.18%, followed closely by Claude-Sonnet-3.7 (91.17%). The strongest open-source model, DeepSeek-R1, reaches 82.51%, underscoring both the dataset's difficulty and the remaining performance gap in Persian commonsense reasoning. We further show that DRESS-AF transfers to the English HellaSwag benchmark, increasing its difficulty without hurting human solvability. The dataset is available at this https URL. 

---
# Does In-IDE Calibration of Large Language Models work at Scale? 

**Authors**: Roham Koohestani, Agnia Sergeyuk, David Gros, Claudio Spiess, Sergey Titov, Prem Devanbu, Maliheh Izadi  

**Link**: [PDF](https://arxiv.org/pdf/2510.22614)  

**Abstract**: The introduction of large language models into integrated development environments (IDEs) is revolutionizing software engineering, yet it poses challenges to the usefulness and reliability of Artificial Intelligence-generated code. Post-hoc calibration of internal model confidences aims to align probabilities with an acceptability measure. Prior work suggests calibration can improve alignment, but at-scale evidence is limited. In this work, we investigate the feasibility of applying calibration of code models to an in-IDE context. We study two aspects of the problem: (1) the technical method for implementing confidence calibration and improving the reliability of code generation models, and (2) the human-centered design principles for effectively communicating reliability signal to developers. First, we develop a scalable and flexible calibration framework which can be used to obtain calibration weights for open-source models using any dataset, and evaluate whether calibrators improve the alignment between model confidence and developer acceptance behavior. Through a large-scale analysis of over 24 million real-world developer interactions across multiple programming languages, we find that a general, post-hoc calibration model based on Platt-scaling does not, on average, improve the reliability of model confidence signals. We also find that while dynamically personalizing calibration to individual users can be effective, its effectiveness is highly dependent on the volume of user interaction data. Second, we conduct a multi-phase design study with 3 expert designers and 153 professional developers, combining scenario-based design, semi-structured interviews, and survey validation, revealing a clear preference for presenting reliability signals via non-numerical, color-coded indicators within the in-editor code generation workflow. 

---
# Personal Care Utility (PCU): Building the Health Infrastructure for Everyday Insight and Guidance 

**Authors**: Mahyar Abbasian, Ramesh Jain  

**Link**: [PDF](https://arxiv.org/pdf/2510.22602)  

**Abstract**: Building on decades of success in digital infrastructure and biomedical innovation, we propose the Personal Care Utility (PCU) - a cybernetic system for lifelong health guidance. PCU is conceived as a global, AI-powered utility that continuously orchestrates multimodal data, knowledge, and services to assist individuals and populations alike. Drawing on multimodal agents, event-centric modeling, and contextual inference, it offers three essential capabilities: (1) trusted health information tailored to the individual, (2) proactive health navigation and behavior guidance, and (3) ongoing interpretation of recovery and treatment response after medical events. Unlike conventional episodic care, PCU functions as an ambient, adaptive companion - observing, interpreting, and guiding health in real time across daily life. By integrating personal sensing, experiential computing, and population-level analytics, PCU promises not only improved outcomes for individuals but also a new substrate for public health and scientific discovery. We describe the architecture, design principles, and implementation challenges of this emerging paradigm. 

---
# RoGER-SLAM: A Robust Gaussian Splatting SLAM System for Noisy and Low-light Environment Resilience 

**Authors**: Huilin Yin, Zhaolin Yang, Linchuan Zhang, Gerhard Rigoll, Johannes Betz  

**Link**: [PDF](https://arxiv.org/pdf/2510.22600)  

**Abstract**: The reliability of Simultaneous Localization and Mapping (SLAM) is severely constrained in environments where visual inputs suffer from noise and low illumination. Although recent 3D Gaussian Splatting (3DGS) based SLAM frameworks achieve high-fidelity mapping under clean conditions, they remain vulnerable to compounded degradations that degrade mapping and tracking performance. A key observation underlying our work is that the original 3DGS rendering pipeline inherently behaves as an implicit low-pass filter, attenuating high-frequency noise but also risking over-smoothing. Building on this insight, we propose RoGER-SLAM, a robust 3DGS SLAM system tailored for noise and low-light resilience. The framework integrates three innovations: a Structure-Preserving Robust Fusion (SP-RoFusion) mechanism that couples rendered appearance, depth, and edge cues; an adaptive tracking objective with residual balancing regularization; and a Contrastive Language-Image Pretraining (CLIP)-based enhancement module, selectively activated under compounded degradations to restore semantic and structural fidelity. Comprehensive experiments on Replica, TUM, and real-world sequences show that RoGER-SLAM consistently improves trajectory accuracy and reconstruction quality compared with other 3DGS-SLAM systems, especially under adverse imaging conditions. 

---
# AutoBench: Automating LLM Evaluation through Reciprocal Peer Assessment 

**Authors**: Dario Loi, Elena Maria Muià, Federico Siciliano, Giovanni Trappolini, Vincenzo Crisà, Peter Kruger, Fabrizio Silvestri  

**Link**: [PDF](https://arxiv.org/pdf/2510.22593)  

**Abstract**: We present AutoBench, a fully automated and self-sustaining framework for evaluating Large Language Models (LLMs) through reciprocal peer assessment. This paper provides a rigorous scientific validation of the AutoBench methodology, originally developed as an open-source project by eZecute S.R.L.. Unlike static benchmarks that suffer from test-set contamination and limited adaptability, AutoBench dynamically generates novel evaluation tasks while models alternately serve as question generators, contestants, and judges across diverse domains. An iterative weighting mechanism amplifies the influence of consistently reliable evaluators, aggregating peer judgments into consensus-based rankings that reflect collective model agreement. Our experiments demonstrate strong correlations with established benchmarks including MMLU-Pro and GPQA (respectively 78\% and 63\%), validating this peer-driven evaluation paradigm. The multi-judge design significantly outperforms single-judge baselines, confirming that distributed evaluation produces more robust and human-consistent assessments. AutoBench offers a scalable, contamination-resistant alternative to static benchmarks for the continuous evaluation of evolving language models. 

---
# Combining Deep Learning and Explainable AI for Toxicity Prediction of Chemical Compounds 

**Authors**: Eduard Popescu, Adrian Groza, Andreea Cernat  

**Link**: [PDF](https://arxiv.org/pdf/2510.22572)  

**Abstract**: The task here is to predict the toxicological activity of chemical compounds based on the Tox21 dataset, a benchmark in computational toxicology.
After a domain-specific overview of chemical toxicity, we discuss current computational strategies, focusing on machine learning and deep learning. Several architectures are compared in terms of performance, robustness, and interpretability.
This research introduces a novel image-based pipeline based on DenseNet121, which processes 2D graphical representations of chemical structures. Additionally, we employ Grad-CAM visualizations, an explainable AI technique, to interpret the model's predictions and highlight molecular regions contributing to toxicity classification. The proposed architecture achieves competitive results compared to traditional models, demonstrating the potential of deep convolutional networks in cheminformatics. Our findings emphasize the value of combining image-based representations with explainable AI methods to improve both predictive accuracy and model transparency in toxicology. 

---
# STATUS Bench: A Rigorous Benchmark for Evaluating Object State Understanding in Vision-Language Models 

**Authors**: Mahiro Ukai, Shuhei Kurita, Nakamasa Inoue  

**Link**: [PDF](https://arxiv.org/pdf/2510.22571)  

**Abstract**: Object state recognition aims to identify the specific condition of objects, such as their positional states (e.g., open or closed) and functional states (e.g., on or off). While recent Vision-Language Models (VLMs) are capable of performing a variety of multimodal tasks, it remains unclear how precisely they can identify object states. To alleviate this issue, we introduce the STAte and Transition UnderStanding Benchmark (STATUS Bench), the first benchmark for rigorously evaluating the ability of VLMs to understand subtle variations in object states in diverse situations. Specifically, STATUS Bench introduces a novel evaluation scheme that requires VLMs to perform three tasks simultaneously: object state identification (OSI), image retrieval (IR), and state change identification (SCI). These tasks are defined over our fully hand-crafted dataset involving image pairs, their corresponding object state descriptions and state change descriptions. Furthermore, we introduce a large-scale training dataset, namely STATUS Train, which consists of 13 million semi-automatically created descriptions. This dataset serves as the largest resource to facilitate further research in this area. In our experiments, we demonstrate that STATUS Bench enables rigorous consistency evaluation and reveal that current state-of-the-art VLMs still significantly struggle to capture subtle object state distinctions. Surprisingly, under the proposed rigorous evaluation scheme, most open-weight VLMs exhibited chance-level zero-shot performance. After fine-tuning on STATUS Train, Qwen2.5-VL achieved performance comparable to Gemini 2.0 Flash. These findings underscore the necessity of STATUS Bench and Train for advancing object state recognition in VLM research. 

---
# Curriculum-Based Iterative Self-Play for Scalable Multi-Drone Racing 

**Authors**: Onur Akgün  

**Link**: [PDF](https://arxiv.org/pdf/2510.22570)  

**Abstract**: The coordination of multiple autonomous agents in high-speed, competitive environments represents a significant engineering challenge. This paper presents CRUISE (Curriculum-Based Iterative Self-Play for Scalable Multi-Drone Racing), a reinforcement learning framework designed to solve this challenge in the demanding domain of multi-drone racing. CRUISE overcomes key scalability limitations by synergistically combining a progressive difficulty curriculum with an efficient self-play mechanism to foster robust competitive behaviors. Validated in high-fidelity simulation with realistic quadrotor dynamics, the resulting policies significantly outperform both a standard reinforcement learning baseline and a state-of-the-art game-theoretic planner. CRUISE achieves nearly double the planner's mean racing speed, maintains high success rates, and demonstrates robust scalability as agent density increases. Ablation studies confirm that the curriculum structure is the critical component for this performance leap. By providing a scalable and effective training methodology, CRUISE advances the development of autonomous systems for dynamic, competitive tasks and serves as a blueprint for future real-world deployment. 

---
# SPIRAL: Self-Play Incremental Racing Algorithm for Learning in Multi-Drone Competitions 

**Authors**: Onur Akgün  

**Link**: [PDF](https://arxiv.org/pdf/2510.22568)  

**Abstract**: This paper introduces SPIRAL (Self-Play Incremental Racing Algorithm for Learning), a novel approach for training autonomous drones in multi-agent racing competitions. SPIRAL distinctively employs a self-play mechanism to incrementally cultivate complex racing behaviors within a challenging, dynamic environment. Through this self-play core, drones continuously compete against increasingly proficient versions of themselves, naturally escalating the difficulty of competitive interactions. This progressive learning journey guides agents from mastering fundamental flight control to executing sophisticated cooperative multi-drone racing strategies. Our method is designed for versatility, allowing integration with any state-of-the-art Deep Reinforcement Learning (DRL) algorithms within its self-play framework. Simulations demonstrate the significant advantages of SPIRAL and benchmark the performance of various DRL algorithms operating within it. Consequently, we contribute a versatile, scalable, and self-improving learning framework to the field of autonomous drone racing. SPIRAL's capacity to autonomously generate appropriate and escalating challenges through its self-play dynamic offers a promising direction for developing robust and adaptive racing strategies in multi-agent environments. This research opens new avenues for enhancing the performance and reliability of autonomous racing drones in increasingly complex and competitive scenarios. 

---
# Blockchain Signatures to Ensure Information Integrity and Non-Repudiation in the Digital Era: A comprehensive study 

**Authors**: Kaveri Banerjee, Sajal Saha  

**Link**: [PDF](https://arxiv.org/pdf/2510.22561)  

**Abstract**: Blockchain systems rely on decentralized ledgers and strong security guarantees. A key requirement is non-repudiation, which prevents denial of transaction authorship and supports integrity of recorded data. This work surveys digital signature schemes used in blockchain platforms and analyzes how they deliver non-repudiation and contribute to overall system security. We examine representative scheme families and their cryptographic foundations, security assumptions, and properties relevant to deployment, including unforgeability, resistance to malleability, support for aggregation and multisignature or threshold settings, key and signature sizes, and verification cost. Using these criteria, we compare the suitability of different designs for consensus protocols, smart contract constraints, and resource limits. We highlight practical tradeoffs that affect throughput, storage, scalability, and attack surfaces, and summarize benefits and limitations of each scheme in blockchain contexts. The study underscores that carefully chosen digital signatures are central to achieving non-repudiation and preserving information integrity, and it outlines implementation considerations and open directions such as interoperability and post-quantum readiness. 

---
# DDTR: Diffusion Denoising Trace Recovery 

**Authors**: Maximilian Matyash, Avigdor Gal, Arik Senderovich  

**Link**: [PDF](https://arxiv.org/pdf/2510.22553)  

**Abstract**: With recent technological advances, process logs, which were traditionally deterministic in nature, are being captured from non-deterministic sources, such as uncertain sensors or machine learning models (that predict activities using cameras). In the presence of stochastically-known logs, logs that contain probabilistic information, the need for stochastic trace recovery increases, to offer reliable means of understanding the processes that govern such systems. We design a novel deep learning approach for stochastic trace recovery, based on Diffusion Denoising Probabilistic Models (DDPM), which makes use of process knowledge (either implicitly by discovering a model or explicitly by injecting process knowledge in the training phase) to recover traces by denoising. We conduct an empirical evaluation demonstrating state-of-the-art performance with up to a 25% improvement over existing methods, along with increased robustness under high noise levels. 

---
# LooGLE v2: Are LLMs Ready for Real World Long Dependency Challenges? 

**Authors**: Ziyuan He, Yuxuan Wang, Jiaqi Li, Kexin Liang, Muhan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.22548)  

**Abstract**: Large language models (LLMs) are equipped with increasingly extended context windows recently, yet their long context understanding capabilities over long dependency tasks remain fundamentally limited and underexplored. This gap is especially significant in many real-world long-context applications that were rarely benchmarked. In this paper, we introduce LooGLE v2, a novel benchmark designed to evaluate LLMs' long context ability in real-world applications and scenarios. Our benchmark consists of automatically collected real-world long texts, ranging from 16k to 2M tokens, encompassing domains in law, finance, game and code. Accordingly, we delicately design 10 types of domain-specific long-dependency tasks and generate 1,934 QA instances with various diversity and complexity in a scalable data curation pipeline for further practical needs. We conduct a comprehensive assessment of 6 locally deployed and 4 API-based LLMs. The evaluation results show that even the best-performing model achieves only a 59.2% overall score on our benchmark. Despite the extensive context windows, popular LLMs are only capable of understanding a much shorter length of context than they claim to be, revealing significant limitations in their ability to handle real-world tasks with long dependencies and highlighting substantial room for model improvement in practical long-context understanding. 

---
# Text to Trust: Evaluating Fine-Tuning and LoRA Trade-offs in Language Models for Unfair Terms of Service Detection 

**Authors**: Noshitha Padma Pratyusha Juttu, Sahithi Singireddy, Sravani Gona, Sujal Timilsina  

**Link**: [PDF](https://arxiv.org/pdf/2510.22531)  

**Abstract**: Large Language Models (LLMs) have transformed text understanding, yet their adaptation to specialized legal domains remains constrained by the cost of full fine-tuning. This study provides a systematic evaluation of fine tuning, parameter efficient adaptation (LoRA, QLoRA), and zero-shot prompting strategies for unfair clause detection in Terms of Service (ToS) documents, a key application in legal NLP. We finetune BERT and DistilBERT, apply 4-bit Low-Rank Adaptation (LoRA) to models such as TinyLlama, LLaMA 3B/7B, and SaulLM, and evaluate GPT-4o and O-versions in zero-shot settings. Experiments on the CLAUDETTE-ToS benchmark and the Multilingual Scraper Corpus show that full fine-tuning achieves the strongest precision recall balance, while LoRA-based models provide competitive recall with up to 3x lower memory cost. These findings highlight practical design trade-offs for efficient and domain-adapted LLMs, contributing open baselines for fine-tuning research in legal text processing. 

---
# Open Multimodal Retrieval-Augmented Factual Image Generation 

**Authors**: Yang Tian, Fan Liu, Jingyuan Zhang, Wei Bi, Yupeng Hu, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2510.22521)  

**Abstract**: Large Multimodal Models (LMMs) have achieved remarkable progress in generating photorealistic and prompt-aligned images, but they often produce outputs that contradict verifiable knowledge, especially when prompts involve fine-grained attributes or time-sensitive events. Conventional retrieval-augmented approaches attempt to address this issue by introducing external information, yet they are fundamentally incapable of grounding generation in accurate and evolving knowledge due to their reliance on static sources and shallow evidence integration. To bridge this gap, we introduce ORIG, an agentic open multimodal retrieval-augmented framework for Factual Image Generation (FIG), a new task that requires both visual realism and factual grounding. ORIG iteratively retrieves and filters multimodal evidence from the web and incrementally integrates the refined knowledge into enriched prompts to guide generation. To support systematic evaluation, we build FIG-Eval, a benchmark spanning ten categories across perceptual, compositional, and temporal dimensions. Experiments demonstrate that ORIG substantially improves factual consistency and overall image quality over strong baselines, highlighting the potential of open multimodal retrieval for factual image generation. 

---
# Toward Robust Signed Graph Learning through Joint Input-Target Denoising 

**Authors**: Junran Wu, Beng Chin Ooi, Ke Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.22513)  

**Abstract**: Signed Graph Neural Networks (SGNNs) are widely adopted to analyze complex patterns in signed graphs with both positive and negative links. Given the noisy nature of real-world connections, the robustness of SGNN has also emerged as a pivotal research area. Under the supervision of empirical properties, graph structure learning has shown its robustness on signed graph representation learning, however, there remains a paucity of research investigating a robust SGNN with theoretical guidance. Inspired by the success of graph information bottleneck (GIB) in information extraction, we propose RIDGE, a novel framework for Robust sI gned graph learning through joint Denoising of Graph inputs and supervision targEts. Different from the basic GIB, we extend the GIB theory with the capability of target space denoising as the co-existence of noise in both input and target spaces. In instantiation, RIDGE effectively cleanses input data and supervision targets via a tractable objective function produced by reparameterization mechanism and variational approximation. We extensively validate our method on four prevalent signed graph datasets, and the results show that RIDGE clearly improves the robustness of popular SGNN models under various levels of noise. 

---
# Transitive RL: Value Learning via Divide and Conquer 

**Authors**: Seohong Park, Aditya Oberai, Pranav Atreya, Sergey Levine  

**Link**: [PDF](https://arxiv.org/pdf/2510.22512)  

**Abstract**: In this work, we present Transitive Reinforcement Learning (TRL), a new value learning algorithm based on a divide-and-conquer paradigm. TRL is designed for offline goal-conditioned reinforcement learning (GCRL) problems, where the aim is to find a policy that can reach any state from any other state in the smallest number of steps. TRL converts a triangle inequality structure present in GCRL into a practical divide-and-conquer value update rule. This has several advantages compared to alternative value learning paradigms. Compared to temporal difference (TD) methods, TRL suffers less from bias accumulation, as in principle it only requires $O(\log T)$ recursions (as opposed to $O(T)$ in TD learning) to handle a length-$T$ trajectory. Unlike Monte Carlo methods, TRL suffers less from high variance as it performs dynamic programming. Experimentally, we show that TRL achieves the best performance in highly challenging, long-horizon benchmark tasks compared to previous offline GCRL algorithms. 

---
# GateFuseNet: An Adaptive 3D Multimodal Neuroimaging Fusion Network for Parkinson's Disease Diagnosis 

**Authors**: Rui Jin, Chen Chen, Yin Liu, Hongfu Sun, Min Zeng, Min Li, Yang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2510.22507)  

**Abstract**: Accurate diagnosis of Parkinson's disease (PD) from MRI remains challenging due to symptom variability and pathological heterogeneity. Most existing methods rely on conventional magnitude-based MRI modalities, such as T1-weighted images (T1w), which are less sensitive to PD pathology than Quantitative Susceptibility Mapping (QSM), a phase-based MRI technique that quantifies iron deposition in deep gray matter nuclei. In this study, we propose GateFuseNet, an adaptive 3D multimodal fusion network that integrates QSM and T1w images for PD diagnosis. The core innovation lies in a gated fusion module that learns modality-specific attention weights and channel-wise gating vectors for selective feature modulation. This hierarchical gating mechanism enhances ROI-aware features while suppressing irrelevant signals. Experimental results show that our method outperforms three existing state-of-the-art approaches, achieving 85.00% accuracy and 92.06% AUC. Ablation studies further validate the contributions of ROI guidance, multimodal integration, and fusion positioning. Grad-CAM visualizations confirm the model's focus on clinically relevant pathological regions. The source codes and pretrained models can be found at this https URL 

---
# Accelerating Materials Design via LLM-Guided Evolutionary Search 

**Authors**: Nikhil Abhyankar, Sanchit Kabra, Saaketh Desai, Chandan K. Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2510.22503)  

**Abstract**: Materials discovery requires navigating vast chemical and structural spaces while satisfying multiple, often conflicting, objectives. We present LLM-guided Evolution for MAterials design (LLEMA), a unified framework that couples the scientific knowledge embedded in large language models with chemistry-informed evolutionary rules and memory-based refinement. At each iteration, an LLM proposes crystallographically specified candidates under explicit property constraints; a surrogate-augmented oracle estimates physicochemical properties; and a multi-objective scorer updates success/failure memories to guide subsequent generations. Evaluated on 14 realistic tasks spanning electronics, energy, coatings, optics, and aerospace, LLEMA discovers candidates that are chemically plausible, thermodynamically stable, and property-aligned, achieving higher hit-rates and stronger Pareto fronts than generative and LLM-only baselines. Ablation studies confirm the importance of rule-guided generation, memory-based refinement, and surrogate prediction. By enforcing synthesizability and multi-objective trade-offs, LLEMA delivers a principled pathway to accelerate practical materials discovery.
Code: this https URL 

---
# Scalable Oversight via Partitioned Human Supervision 

**Authors**: Ren Yin, Takashi Ishida, Masashi Sugiyama  

**Link**: [PDF](https://arxiv.org/pdf/2510.22500)  

**Abstract**: As artificial intelligence (AI) systems approach and surpass expert human performance across a broad range of tasks, obtaining high-quality human supervision for evaluation and training becomes increasingly challenging. Our focus is on tasks that require deep knowledge and skills of multiple domains. Unfortunately, even the best human experts are knowledgeable only in a single narrow area, and will not be able to evaluate the correctness of advanced AI systems on such superhuman tasks. However, based on their narrow expertise, humans may provide a weak signal, i.e., a complementary label indicating an option that is incorrect. For example, a cardiologist could state that "this is not related to cardiology,'' even if they cannot identify the true disease. Based on this weak signal, we propose a scalable oversight framework that enables us to evaluate frontier AI systems without the need to prepare the ground truth. We derive an unbiased estimator of top-1 accuracy from complementary labels and quantify how many complementary labels are needed to match the variance of ordinary labels. We further introduce two estimators to combine scarce ordinary labels with abundant complementary labels. We provide finite-sample deviation guarantees for both complementary-only and the mixed estimators. Empirically, we show that we can evaluate the output of large language models without the ground truth, if we have complementary labels. We further show that we can train an AI system with such weak signals: we show how we can design an agentic AI system automatically that can perform better with this partitioned human supervision. Our code is available at this https URL. 

---
# An Analytic Theory of Quantum Imaginary Time Evolution 

**Authors**: Min Chen, Bingzhi Zhang, Quntao Zhuang, Junyu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.22481)  

**Abstract**: Quantum imaginary time evolution (QITE) algorithm is one of the most promising variational quantum algorithms (VQAs), bridging the current era of Noisy Intermediate-Scale Quantum devices and the future of fully fault-tolerant quantum computing. Although practical demonstrations of QITE and its potential advantages over the general VQA trained with vanilla gradient descent (GD) in certain tasks have been reported, a first-principle, theoretical understanding of QITE remains limited. Here, we aim to develop an analytic theory for the dynamics of QITE. First, we show that QITE can be interpreted as a form of a general VQA trained with Quantum Natural Gradient Descent (QNGD), where the inverse quantum Fisher information matrix serves as the learning-rate tensor. This equivalence is established not only at the level of gradient update rules, but also through the action principle: the variational principle can be directly connected to the geometric geodesic distance in the quantum Fisher information metric, up to an integration constant. Second, for wide quantum neural networks, we employ the quantum neural tangent kernel framework to construct an analytic model for QITE. We prove that QITE always converges faster than GD-based VQA, though this advantage is suppressed by the exponential growth of Hilbert space dimension. This helps explain certain experimental results in quantum computational chemistry. Our theory encompasses linear, quadratic, and more general loss functions. We validate the analytic results through numerical simulations. Our findings establish a theoretical foundation for QITE dynamics and provide analytic insights for the first-principle design of variational quantum algorithms. 

---
# Single-Teacher View Augmentation: Boosting Knowledge Distillation via Angular Diversity 

**Authors**: Seonghoon Yu, Dongjun Nam, Dina Katabi, Jeany Son  

**Link**: [PDF](https://arxiv.org/pdf/2510.22480)  

**Abstract**: Knowledge Distillation (KD) aims to train a lightweight student model by transferring knowledge from a large, high-capacity teacher. Recent studies have shown that leveraging diverse teacher perspectives can significantly improve distillation performance; however, achieving such diversity typically requires multiple teacher networks, leading to high computational costs. In this work, we propose a novel cost-efficient knowledge augmentation method for KD that generates diverse multi-views by attaching multiple branches to a single teacher. To ensure meaningful semantic variation across multi-views, we introduce two angular diversity objectives: 1) constrained inter-angle diversify loss, which maximizes angles between augmented views while preserving proximity to the original teacher output, and 2) intra-angle diversify loss, which encourages an even distribution of views around the original output. The ensembled knowledge from these angularly diverse views, along with the original teacher, is distilled into the student. We further theoretically demonstrate that our objectives increase the diversity among ensemble members and thereby reduce the upper bound of the ensemble's expected loss, leading to more effective distillation. Experimental results show that our method surpasses an existing knowledge augmentation method across diverse configurations. Moreover, the proposed method is compatible with other KD frameworks in a plug-and-play fashion, providing consistent improvements in generalization performance. 

---
# Agent-GSPO: Communication-Efficient Multi-Agent Systems via Group Sequence Policy Optimization 

**Authors**: Yijia Fan, Jusheng Zhang, Jing Yang, Keze Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.22477)  

**Abstract**: To combat the prohibitive communication costs of ``free-for-all" multi-agent systems (MAS), we introduce \textbf{Agent-GSPO}, a framework that directly optimizes for token economy using sequence-level reinforcement learning. Agent-GSPO leverages the stable and memory-efficient Group Sequence Policy Optimization (GSPO) algorithm to train agents on a communication-aware reward that explicitly penalizes verbosity. Across seven reasoning benchmarks, Agent-GSPO not only achieves new state-of-the-art performance but does so with a fraction of the token consumption of existing methods. By fostering emergent strategies like ``strategic silence," our approach provides a practical blueprint for developing scalable and economically viable multi-agent systems. 

---
# CHOIR: Collaborative Harmonization fOr Inference Robustness 

**Authors**: Xiangjue Dong, Cong Wang, Maria Teleki, Millennium Bismay, James Caverlee  

**Link**: [PDF](https://arxiv.org/pdf/2510.22475)  

**Abstract**: Persona-assigned Large Language Models (LLMs) can adopt diverse roles, enabling personalized and context-aware reasoning. However, even minor demographic perturbations in personas, such as simple pronoun changes, can alter reasoning trajectories, leading to divergent sets of correct answers. Instead of treating these variations as biases to be mitigated, we explore their potential as a constructive resource to improve reasoning robustness. We propose CHOIR (Collaborative Harmonization fOr Inference Robustness), a test-time framework that harmonizes multiple persona-conditioned reasoning signals into a unified prediction. CHOIR orchestrates a collaborative decoding process among counterfactual personas, dynamically balancing agreement and divergence in their reasoning paths. Experiments on various reasoning benchmarks demonstrate that CHOIR consistently enhances performance across demographics, model architectures, scales, and tasks - without additional training. Improvements reach up to 26.4% for individual demographic groups and 19.2% on average across five demographics. It remains effective even when base personas are suboptimal. By reframing persona variation as a constructive signal, CHOIR provides a scalable and generalizable approach to more reliable LLM reasoning. 

---
# DynaPose4D: High-Quality 4D Dynamic Content Generation via Pose Alignment Loss 

**Authors**: Jing Yang, Yufeng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.22473)  

**Abstract**: Recent advancements in 2D and 3D generative models have expanded the capabilities of computer vision. However, generating high-quality 4D dynamic content from a single static image remains a significant challenge. Traditional methods have limitations in modeling temporal dependencies and accurately capturing dynamic geometry changes, especially when considering variations in camera perspective. To address this issue, we propose DynaPose4D, an innovative solution that integrates 4D Gaussian Splatting (4DGS) techniques with Category-Agnostic Pose Estimation (CAPE) technology. This framework uses 3D Gaussian Splatting to construct a 3D model from single images, then predicts multi-view pose keypoints based on one-shot support from a chosen view, leveraging supervisory signals to enhance motion consistency. Experimental results show that DynaPose4D achieves excellent coherence, consistency, and fluidity in dynamic motion generation. These findings not only validate the efficacy of the DynaPose4D framework but also indicate its potential applications in the domains of computer vision and animation production. 

---
# Backward-Friendly Optimization: Training Large Language Models with Approximate Gradients under Memory Constraints 

**Authors**: Jing Yang, Kaitong Cai, Yijia Fan, Yufeng Yang, Keze Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.22467)  

**Abstract**: Full fine-tuning of Large Language Models (LLMs) is notoriously memory-intensive, primarily because conventional optimizers such as SGD or Adam assume access to exact gradients derived from cached activations. Existing solutions either alter the model architecture (e.g., reversible networks) or trade memory for computation (e.g., activation checkpointing), but the optimizer itself remains untouched. In this work, we introduce GradLite, a backward-friendly optimizer that relaxes the requirement of exact gradients, enabling efficient training even when intermediate activations are aggressively discarded or approximated. GradLite leverages two key techniques: (i) low-rank Jacobian approximation, which reduces the dimensionality of backpropagated error signals, and (ii) error-feedback correction, which accumulates and compensates approximation errors across iterations to preserve convergence guarantees. We provide a theoretical analysis showing that GradLite maintains unbiased gradient estimates with bounded variance, ensuring convergence rates comparable to Adam. Empirically, GradLite reduces optimizer-state and activation memory consumption by up to 50\% without architectural changes, and achieves on-par or superior downstream performance on reasoning (MMLU, GSM8K), multilingual, and dialogue benchmarks compared to checkpointing and optimizer-centric baselines (LoMo, GaLore). 

---
# Evaluating Multimodal Large Language Models on Core Music Perception Tasks 

**Authors**: Brandon James Carone, Iran R. Roman, Pablo Ripollés  

**Link**: [PDF](https://arxiv.org/pdf/2510.22455)  

**Abstract**: Multimodal Large Language Models (LLMs) claim "musical understanding" via evaluations that conflate listening with score reading. We benchmark three SOTA LLMs (Gemini 2.5 Pro, Gemini 2.5 Flash, and Qwen2.5-Omni) across three core music skills: Syncopation Scoring, Transposition Detection, and Chord Quality Identification. Moreover, we separate three sources of variability: (i) perceptual limitations (audio vs. MIDI inputs), (ii) exposure to examples (zero- vs. few-shot manipulations), and (iii) reasoning strategies (Standalone, CoT, LogicLM). For the latter we adapt LogicLM, a framework combining LLMs with symbolic solvers to perform structured reasoning, to music. Results reveal a clear perceptual gap: models perform near ceiling on MIDI but show accuracy drops on audio. Reasoning and few-shot prompting offer minimal gains. This is expected for MIDI, where performance reaches saturation, but more surprising for audio, where LogicLM, despite near-perfect MIDI accuracy, remains notably brittle. Among models, Gemini Pro achieves the highest performance across most conditions. Overall, current systems reason well over symbols (MIDI) but do not yet "listen" reliably from audio. Our method and dataset make the perception-reasoning boundary explicit and offer actionable guidance for building robust, audio-first music systems. 

---
# GraphTOP: Graph Topology-Oriented Prompting for Graph Neural Networks 

**Authors**: Xingbo Fu, Zhenyu Lei, Zihan Chen, Binchi Zhang, Chuxu Zhang, Jundong Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.22451)  

**Abstract**: Graph Neural Networks (GNNs) have revolutionized the field of graph learning by learning expressive graph representations from massive graph data. As a common pattern to train powerful GNNs, the "pre-training, adaptation" scheme first pre-trains GNNs over unlabeled graph data and subsequently adapts them to specific downstream tasks. In the adaptation phase, graph prompting is an effective strategy that modifies input graph data with learnable prompts while keeping pre-trained GNN models frozen. Typically, existing graph prompting studies mainly focus on *feature-oriented* methods that apply graph prompts to node features or hidden representations. However, these studies often achieve suboptimal performance, as they consistently overlook the potential of *topology-oriented* prompting, which adapts pre-trained GNNs by modifying the graph topology. In this study, we conduct a pioneering investigation of graph prompting in terms of graph topology. We propose the first **Graph** **T**opology-**O**riented **P**rompting (GraphTOP) framework to effectively adapt pre-trained GNN models for downstream tasks. More specifically, we reformulate topology-oriented prompting as an edge rewiring problem within multi-hop local subgraphs and relax it into the continuous probability space through reparameterization while ensuring tight relaxation and preserving graph sparsity. Extensive experiments on five graph datasets under four pre-training strategies demonstrate that our proposed GraphTOP outshines six baselines on multiple node classification datasets. Our code is available at this https URL. 

---
# SmartMixed: A Two-Phase Training Strategy for Adaptive Activation Function Learning in Neural Networks 

**Authors**: Amin Omidvar  

**Link**: [PDF](https://arxiv.org/pdf/2510.22450)  

**Abstract**: The choice of activation function plays a critical role in neural networks, yet most architectures still rely on fixed, uniform activation functions across all neurons. We introduce SmartMixed, a two-phase training strategy that allows networks to learn optimal per-neuron activation functions while preserving computational efficiency at inference. In the first phase, neurons adaptively select from a pool of candidate activation functions (ReLU, Sigmoid, Tanh, Leaky ReLU, ELU, SELU) using a differentiable hard-mixture mechanism. In the second phase, each neuron's activation function is fixed according to the learned selection, resulting in a computationally efficient network that supports continued training with optimized vectorized operations. We evaluate SmartMixed on the MNIST dataset using feedforward neural networks of varying depths. The analysis shows that neurons in different layers exhibit distinct preferences for activation functions, providing insights into the functional diversity within neural architectures. 

---
# PromptReverb: Multimodal Room Impulse Response Generation Through Latent Rectified Flow Matching 

**Authors**: Ali Vosoughi, Yongyi Zang, Qihui Yang, Nathan Peak, Randal Leistikow, Chenliang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.22439)  

**Abstract**: Room impulse response (RIR) generation remains a critical challenge for creating immersive virtual acoustic environments. Current methods suffer from two fundamental limitations: the scarcity of full-band RIR datasets and the inability of existing models to generate acoustically accurate responses from diverse input modalities. We present PromptReverb, a two-stage generative framework that addresses these challenges. Our approach combines a variational autoencoder that upsamples band-limited RIRs to full-band quality (48 kHz), and a conditional diffusion transformer model based on rectified flow matching that generates RIRs from descriptions in natural language. Empirical evaluation demonstrates that PromptReverb produces RIRs with superior perceptual quality and acoustic accuracy compared to existing methods, achieving 8.8% mean RT60 error compared to -37% for widely used baselines and yielding more realistic room-acoustic parameters. Our method enables practical applications in virtual reality, architectural acoustics, and audio production where flexible, high-quality RIR synthesis is essential. 

---
# Group size effects and collective misalignment in LLM multi-agent systems 

**Authors**: Ariel Flint, Luca Maria Aiello, Romualdo Pastor-Satorras, Andrea Baronchelli  

**Link**: [PDF](https://arxiv.org/pdf/2510.22422)  

**Abstract**: Multi-agent systems of large language models (LLMs) are rapidly expanding across domains, introducing dynamics not captured by single-agent evaluations. Yet, existing work has mostly contrasted the behavior of a single agent with that of a collective of fixed size, leaving open a central question: how does group size shape dynamics? Here, we move beyond this dichotomy and systematically explore outcomes across the full range of group sizes. We focus on multi-agent misalignment, building on recent evidence that interacting LLMs playing a simple coordination game can generate collective biases absent in individual models. First, we show that collective bias is a deeper phenomenon than previously assessed: interaction can amplify individual biases, introduce new ones, or override model-level preferences. Second, we demonstrate that group size affects the dynamics in a non-linear way, revealing model-dependent dynamical regimes. Finally, we develop a mean-field analytical approach and show that, above a critical population size, simulations converge to deterministic predictions that expose the basins of attraction of competing equilibria. These findings establish group size as a key driver of multi-agent dynamics and highlight the need to consider population-level effects when deploying LLM-based systems at scale. 

---
# Knowledge-guided Continual Learning for Behavioral Analytics Systems 

**Authors**: Yasas Senarath, Hemant Purohit  

**Link**: [PDF](https://arxiv.org/pdf/2510.22405)  

**Abstract**: User behavior on online platforms is evolving, reflecting real-world changes in how people post, whether it's helpful messages or hate speech. Models that learn to capture this content can experience a decrease in performance over time due to data drift, which can lead to ineffective behavioral analytics systems. However, fine-tuning such a model over time with new data can be detrimental due to catastrophic forgetting. Replay-based approaches in continual learning offer a simple yet efficient method to update such models, minimizing forgetting by maintaining a buffer of important training instances from past learned tasks. However, the main limitation of this approach is the fixed size of the buffer. External knowledge bases can be utilized to overcome this limitation through data augmentation. We propose a novel augmentation-based approach to incorporate external knowledge in the replay-based continual learning framework. We evaluate several strategies with three datasets from prior studies related to deviant behavior classification to assess the integration of external knowledge in continual learning and demonstrate that augmentation helps outperform baseline replay-based approaches. 

---
# Top-Down Semantic Refinement for Image Captioning 

**Authors**: Jusheng Zhang, Kaitong Cai, Jing Yang, Jian Wang, Chengpei Tang, Keze Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.22391)  

**Abstract**: Large Vision-Language Models (VLMs) face an inherent contradiction in image captioning: their powerful single-step generation capabilities often lead to a myopic decision-making process. This makes it difficult to maintain global narrative coherence while capturing rich details, a limitation that is particularly pronounced in tasks that require multi-step and complex scene description. To overcome this fundamental challenge, we redefine image captioning as a goal-oriented hierarchical refinement planning problem, and further propose a novel framework, named Top-Down Semantic Refinement (TDSR), which models the generation process as a Markov Decision Process (MDP). However, planning within the vast state space of a VLM presents a significant computational hurdle. Our core contribution, therefore, is the design of a highly efficient Monte Carlo Tree Search (MCTS) algorithm tailored for VLMs. By incorporating a visual-guided parallel expansion and a lightweight value network, our TDSR reduces the call frequency to the expensive VLM by an order of magnitude without sacrificing planning quality. Furthermore, an adaptive early stopping mechanism dynamically matches computational overhead to the image's complexity. Extensive experiments on multiple benchmarks, including DetailCaps, COMPOSITIONCAP, and POPE, demonstrate that our TDSR, as a plug-and-play module, can significantly enhance the performance of existing VLMs (e.g., LLaVA-1.5, Qwen2.5-VL) by achieving state-of-the-art or highly competitive results in fine-grained description, compositional generalization, and hallucination suppression. 

---
# Can Small and Reasoning Large Language Models Score Journal Articles for Research Quality and Do Averaging and Few-shot Help? 

**Authors**: Mike Thelwall, Ehsan Mohammadi  

**Link**: [PDF](https://arxiv.org/pdf/2510.22389)  

**Abstract**: Assessing published academic journal articles is a common task for evaluations of departments and individuals. Whilst it is sometimes supported by citation data, Large Language Models (LLMs) may give more useful indications of article quality. Evidence of this capability exists for two of the largest LLM families, ChatGPT and Gemini, and the medium sized LLM Gemma3 27b, but it is unclear whether smaller LLMs and reasoning models have similar abilities. This is important because larger models may be slow and impractical in some situations, and reasoning models may perform differently. Four relevant questions are addressed with Gemma3 variants, Llama4 Scout, Qwen3, Magistral Small and DeepSeek R1, on a dataset of 2,780 medical, health and life science papers in 6 fields, with two different gold standards, one novel. The results suggest that smaller (open weights) and reasoning LLMs have similar performance to ChatGPT 4o-mini and Gemini 2.0 Flash, but that 1b parameters may often, and 4b sometimes, be too few. Moreover, averaging scores from multiple identical queries seems to be a universally successful strategy, and few-shot prompts (four examples) tended to help but the evidence was equivocal. Reasoning models did not have a clear advantage. Overall, the results show, for the first time, that smaller LLMs >4b, including reasoning models, have a substantial capability to score journal articles for research quality, especially if score averaging is used. 

---
# Dynamic Dropout: Leveraging Conway's Game of Life for Neural Networks Regularization 

**Authors**: David Freire-Obregón, José Salas-Cáceres, Modesto Castrillón-Santana  

**Link**: [PDF](https://arxiv.org/pdf/2510.22383)  

**Abstract**: Regularization techniques play a crucial role in preventing overfitting and improving the generalization performance of neural networks. Dropout, a widely used regularization technique, randomly deactivates units during training to introduce redundancy and prevent co-adaptation among neurons. Despite its effectiveness, dropout has limitations, such as its static nature and lack of interpretability. In this paper, we propose a novel approach to regularization by substituting dropout with Conway's Game of Life (GoL), a cellular automata with simple rules that govern the evolution of a grid of cells. We introduce dynamic unit deactivation during training by representing neural network units as cells in a GoL grid and applying the game's rules to deactivate units. This approach allows for the emergence of spatial patterns that adapt to the training data, potentially enhancing the network's ability to generalize. We demonstrate the effectiveness of our approach on the CIFAR-10 dataset, showing that dynamic unit deactivation using GoL achieves comparable performance to traditional dropout techniques while offering insights into the network's behavior through the visualization of evolving patterns. Furthermore, our discussion highlights the applicability of our proposal in deeper architectures, demonstrating how it enhances the performance of different dropout techniques. 

---
# Efficient Large-Deformation Medical Image Registration via Recurrent Dynamic Correlation 

**Authors**: Tianran Li, Marius Staring, Yuchuan Qiao  

**Link**: [PDF](https://arxiv.org/pdf/2510.22380)  

**Abstract**: Deformable image registration estimates voxel-wise correspondences between images through spatial transformations, and plays a key role in medical imaging. While deep learning methods have significantly reduced runtime, efficiently handling large deformations remains a challenging task. Convolutional networks aggregate local features but lack direct modeling of voxel correspondences, promoting recent works to explore explicit feature matching. Among them, voxel-to-region matching is more efficient for direct correspondence modeling by computing local correlation features whithin neighbourhoods, while region-to-region matching incurs higher redundancy due to excessive correlation pairs across large regions. However, the inherent locality of voxel-to-region matching hinders the capture of long-range correspondences required for large deformations. To address this, we propose a Recurrent Correlation-based framework that dynamically relocates the matching region toward more promising positions. At each step, local matching is performed with low cost, and the estimated offset guides the next search region, supporting efficient convergence toward large deformations. In addition, we uses a lightweight recurrent update module with memory capacity and decouples motion-related and texture features to suppress semantic redundancy. We conduct extensive experiments on brain MRI and abdominal CT datasets under two settings: with and without affine pre-registration. Results show that our method exibits a strong accuracy-computation trade-off, surpassing or matching the state-of-the-art performance. For example, it achieves comparable performance on the non-affine OASIS dataset, while using only 9.5% of the FLOPs and running 96% faster than RDP, a representative high-performing method. 

---
# TraceTrans: Translation and Spatial Tracing for Surgical Prediction 

**Authors**: Xiyu Luo, Haodong LI, Xinxing Cheng, He Zhao, Yang Hu, Xuan Song, Tianyang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.22379)  

**Abstract**: Image-to-image translation models have achieved notable success in converting images across visual domains and are increasingly used for medical tasks such as predicting post-operative outcomes and modeling disease progression. However, most existing methods primarily aim to match the target distribution and often neglect spatial correspondences between the source and translated images. This limitation can lead to structural inconsistencies and hallucinations, undermining the reliability and interpretability of the predictions. These challenges are accentuated in clinical applications by the stringent requirement for anatomical accuracy. In this work, we present TraceTrans, a novel deformable image translation model designed for post-operative prediction that generates images aligned with the target distribution while explicitly revealing spatial correspondences with the pre-operative input. The framework employs an encoder for feature extraction and dual decoders for predicting spatial deformations and synthesizing the translated image. The predicted deformation field imposes spatial constraints on the generated output, ensuring anatomical consistency with the source. Extensive experiments on medical cosmetology and brain MRI datasets demonstrate that TraceTrans delivers accurate and interpretable post-operative predictions, highlighting its potential for reliable clinical deployment. 

---
# VisJudge-Bench: Aesthetics and Quality Assessment of Visualizations 

**Authors**: Yupeng Xie, Zhiyang Zhang, Yifan Wu, Sirong Lu, Jiayi Zhang, Zhaoyang Yu, Jinlin Wang, Sirui Hong, Bang Liu, Chenglin Wu, Yuyu Luo  

**Link**: [PDF](https://arxiv.org/pdf/2510.22373)  

**Abstract**: Visualization, a domain-specific yet widely used form of imagery, is an effective way to turn complex datasets into intuitive insights, and its value depends on whether data are faithfully represented, clearly communicated, and aesthetically designed. However, evaluating visualization quality is challenging: unlike natural images, it requires simultaneous judgment across data encoding accuracy, information expressiveness, and visual aesthetics. Although multimodal large language models (MLLMs) have shown promising performance in aesthetic assessment of natural images, no systematic benchmark exists for measuring their capabilities in evaluating visualizations. To address this, we propose VisJudge-Bench, the first comprehensive benchmark for evaluating MLLMs' performance in assessing visualization aesthetics and quality. It contains 3,090 expert-annotated samples from real-world scenarios, covering single visualizations, multiple visualizations, and dashboards across 32 chart types. Systematic testing on this benchmark reveals that even the most advanced MLLMs (such as GPT-5) still exhibit significant gaps compared to human experts in judgment, with a Mean Absolute Error (MAE) of 0.551 and a correlation with human ratings of only 0.429. To address this issue, we propose VisJudge, a model specifically designed for visualization aesthetics and quality assessment. Experimental results demonstrate that VisJudge significantly narrows the gap with human judgment, reducing the MAE to 0.442 (a 19.8% reduction) and increasing the consistency with human experts to 0.681 (a 58.7% improvement) compared to GPT-5. The benchmark is available at this https URL. 

---
# BLIP-FusePPO: A Vision-Language Deep Reinforcement Learning Framework for Lane Keeping in Autonomous Vehicles 

**Authors**: Seyed Ahmad Hosseini Miangoleh, Amin Jalal Aghdasian, Farzaneh Abdollahi  

**Link**: [PDF](https://arxiv.org/pdf/2510.22370)  

**Abstract**: In this paper, we propose Bootstrapped Language-Image Pretraining-driven Fused State Representation in Proximal Policy Optimization (BLIP-FusePPO), a novel multimodal reinforcement learning (RL) framework for autonomous lane-keeping (LK), in which semantic embeddings generated by a vision-language model (VLM) are directly fused with geometric states, LiDAR observations, and Proportional-Integral-Derivative-based (PID) control feedback within the agent observation space. The proposed method lets the agent learn driving rules that are aware of their surroundings and easy to understand by combining high-level scene understanding from the VLM with low-level control and spatial signals. Our architecture brings together semantic, geometric, and control-aware representations to make policy learning more robust. A hybrid reward function that includes semantic alignment, LK accuracy, obstacle avoidance, and speed regulation helps learning to be more efficient and generalizable. Our method is different from the approaches that only use semantic models to shape rewards. Instead, it directly embeds semantic features into the state representation. This cuts down on expensive runtime inference and makes sure that semantic guidance is always available. The simulation results show that the proposed model is better at LK stability and adaptability than the best vision-based and multimodal RL baselines in a wide range of difficult driving situations. We make our code publicly available. 

---
# T2SMark: Balancing Robustness and Diversity in Noise-as-Watermark for Diffusion Models 

**Authors**: Jindong Yang, Han Fang, Weiming Zhang, Nenghai Yu, Kejiang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.22366)  

**Abstract**: Diffusion models have advanced rapidly in recent years, producing high-fidelity images while raising concerns about intellectual property protection and the misuse of generative AI. Image watermarking for diffusion models, particularly Noise-as-Watermark (NaW) methods, encode watermark as specific standard Gaussian noise vector for image generation, embedding the infomation seamlessly while maintaining image quality. For detection, the generation process is inverted to recover the initial noise vector containing the watermark before extraction. However, existing NaW methods struggle to balance watermark robustness with generation diversity. Some methods achieve strong robustness by heavily constraining initial noise sampling, which degrades user experience, while others preserve diversity but prove too fragile for real-world deployment. To address this issue, we propose T2SMark, a two-stage watermarking scheme based on Tail-Truncated Sampling (TTS). Unlike prior methods that simply map bits to positive or negative values, TTS enhances robustness by embedding bits exclusively in the reliable tail regions while randomly sampling the central zone to preserve the latent distribution. Our two-stage framework then ensures sampling diversity by integrating a randomly generated session key into both encryption pipelines. We evaluate T2SMark on diffusion models with both U-Net and DiT backbones. Extensive experiments show that it achieves an optimal balance between robustness and diversity. Our code is available at \href{this https URL}{this https URL}. 

---
# FAIR-RAG: Faithful Adaptive Iterative Refinement for Retrieval-Augmented Generation 

**Authors**: Mohammad Aghajani Asl, Majid Asgari-Bidhendi, Behrooz Minaei-Bidgoli  

**Link**: [PDF](https://arxiv.org/pdf/2510.22344)  

**Abstract**: While Retrieval-Augmented Generation (RAG) mitigates hallucination and knowledge staleness in Large Language Models (LLMs), existing frameworks often falter on complex, multi-hop queries that require synthesizing information from disparate sources. Current advanced RAG methods, employing iterative or adaptive strategies, lack a robust mechanism to systematically identify and fill evidence gaps, often propagating noise or failing to gather a comprehensive context. We introduce FAIR-RAG, a novel agentic framework that transforms the standard RAG pipeline into a dynamic, evidence-driven reasoning process. At its core is an Iterative Refinement Cycle governed by a module we term Structured Evidence Assessment (SEA). The SEA acts as an analytical gating mechanism: it deconstructs the initial query into a checklist of required findings and audits the aggregated evidence to identify confirmed facts and, critically, explicit informational gaps. These gaps provide a precise signal to an Adaptive Query Refinement agent, which generates new, targeted sub-queries to retrieve missing information. This cycle repeats until the evidence is verified as sufficient, ensuring a comprehensive context for a final, strictly faithful generation. We conducted experiments on challenging multi-hop QA benchmarks, including HotpotQA, 2WikiMultiHopQA, and MusiQue. In a unified experimental setup, FAIR-RAG significantly outperforms strong baselines. On HotpotQA, it achieves an F1-score of 0.453 -- an absolute improvement of 8.3 points over the strongest iterative baseline -- establishing a new state-of-the-art for this class of methods on these benchmarks. Our work demonstrates that a structured, evidence-driven refinement process with explicit gap analysis is crucial for unlocking reliable and accurate reasoning in advanced RAG systems for complex, knowledge-intensive tasks. 

---
# Toward Humanoid Brain-Body Co-design: Joint Optimization of Control and Morphology for Fall Recovery 

**Authors**: Bo Yue, Sheng Xu, Kui Jia, Guiliang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.22336)  

**Abstract**: Humanoid robots represent a central frontier in embodied intelligence, as their anthropomorphic form enables natural deployment in humans' workspace. Brain-body co-design for humanoids presents a promising approach to realizing this potential by jointly optimizing control policies and physical morphology. Within this context, fall recovery emerges as a critical capability. It not only enhances safety and resilience but also integrates naturally with locomotion systems, thereby advancing the autonomy of humanoids. In this paper, we propose RoboCraft, a scalable humanoid co-design framework for fall recovery that iteratively improves performance through the coupled updates of control policy and morphology. A shared policy pretrained across multiple designs is progressively finetuned on high-performing morphologies, enabling efficient adaptation without retraining from scratch. Concurrently, morphology search is guided by human-inspired priors and optimization algorithms, supported by a priority buffer that balances reevaluation of promising candidates with the exploration of novel designs. Experiments show that \ourmethod{} achieves an average performance gain of 44.55% on seven public humanoid robots, with morphology optimization drives at least 40% of improvements in co-designing four humanoid robots, underscoring the critical role of humanoid co-design. 

---
# Moving Beyond Diffusion: Hierarchy-to-Hierarchy Autoregression for fMRI-to-Image Reconstruction 

**Authors**: Xu Zhang, Ruijie Quan, Wenguan Wang, Yi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.22335)  

**Abstract**: Reconstructing visual stimuli from fMRI signals is a central challenge bridging machine learning and neuroscience. Recent diffusion-based methods typically map fMRI activity to a single high-level embedding, using it as fixed guidance throughout the entire generation process. However, this fixed guidance collapses hierarchical neural information and is misaligned with the stage-dependent demands of image reconstruction. In response, we propose MindHier, a coarse-to-fine fMRI-to-image reconstruction framework built on scale-wise autoregressive modeling. MindHier introduces three components: a Hierarchical fMRI Encoder to extract multi-level neural embeddings, a Hierarchy-to-Hierarchy Alignment scheme to enforce layer-wise correspondence with CLIP features, and a Scale-Aware Coarse-to-Fine Neural Guidance strategy to inject these embeddings into autoregression at matching scales. These designs make MindHier an efficient and cognitively-aligned alternative to diffusion-based methods by enabling a hierarchical reconstruction process that synthesizes global semantics before refining local details, akin to human visual perception. Extensive experiments on the NSD dataset show that MindHier achieves superior semantic fidelity, 4.67x faster inference, and more deterministic results than the diffusion-based baselines. 

---
# Multilingual Target-Stance Extraction 

**Authors**: Ethan Mines, Bonnie Dorr  

**Link**: [PDF](https://arxiv.org/pdf/2510.22334)  

**Abstract**: Social media enables data-driven analysis of public opinion on contested issues. Target-Stance Extraction (TSE) is the task of identifying the target discussed in a document and the document's stance towards that target. Many works classify stance towards a given target in a multilingual setting, but all prior work in TSE is English-only. This work introduces the first multilingual TSE benchmark, spanning Catalan, Estonian, French, Italian, Mandarin, and Spanish corpora. It manages to extend the original TSE pipeline to a multilingual setting without requiring separate models for each language. Our model pipeline achieves a modest F1 score of 12.78, underscoring the increased difficulty of the multilingual task relative to English-only setups and highlighting target prediction as the primary bottleneck. We are also the first to demonstrate the sensitivity of TSE's F1 score to different target verbalizations. Together these serve as a much-needed baseline for resources, algorithms, and evaluation criteria in multilingual TSE. 

---
# Harnessing the Power of Large Language Models for Software Testing Education: A Focus on ISTQB Syllabus 

**Authors**: Tuan-Phong Ngo, Bao-Ngoc Duong, Tuan-Anh Hoang, Joshua Dwight, Ushik Shrestha Khwakhali  

**Link**: [PDF](https://arxiv.org/pdf/2510.22318)  

**Abstract**: Software testing is a critical component in the software engineering field and is important for software engineering education. Thus, it is vital for academia to continuously improve and update educational methods to reflect the current state of the field. The International Software Testing Qualifications Board (ISTQB) certification framework is globally recognized and widely adopted in industry and academia. However, ISTQB-based learning has been rarely applied with recent generative artificial intelligence advances. Despite the growing capabilities of large language models (LLMs), ISTQB-based learning and instruction with LLMs have not been thoroughly explored. This paper explores and evaluates how LLMs can complement the ISTQB framework for higher education. The findings present four key contributions: (i) the creation of a comprehensive ISTQB-aligned dataset spanning over a decade, consisting of 28 sample exams and 1,145 questions; (ii) the development of a domain-optimized prompt that enhances LLM precision and explanation quality on ISTQB tasks; (iii) a systematic evaluation of state-of-the-art LLMs on this dataset; and (iv) actionable insights and recommendations for integrating LLMs into software testing education. These findings highlight the promise of LLMs in supporting ISTQB certification preparation and offer a foundation for their broader use in software engineering at higher education. 

---
# LacMaterial: Large Language Models as Analogical Chemists for Materials Discovery 

**Authors**: Hongyu Guo  

**Link**: [PDF](https://arxiv.org/pdf/2510.22312)  

**Abstract**: Analogical reasoning, the transfer of relational structures across contexts (e.g., planet is to sun as electron is to nucleus), is fundamental to scientific discovery. Yet human insight is often constrained by domain expertise and surface-level biases, limiting access to deeper, structure-driven analogies both within and across disciplines. Large language models (LLMs), trained on vast cross-domain data, present a promising yet underexplored tool for analogical reasoning in science. Here, we demonstrate that LLMs can generate novel battery materials by (1) retrieving cross-domain analogs and analogy-guided exemplars to steer exploration beyond conventional dopant substitutions, and (2) constructing in-domain analogical templates from few labeled examples to guide targeted exploitation. These explicit analogical reasoning strategies yield candidates outside established compositional spaces and outperform standard prompting baselines. Our findings position LLMs as interpretable, expert-like hypothesis generators that leverage analogy-driven generalization for scientific innovation. 

---
# AnyECG-Lab: An Exploration Study of Fine-tuning an ECG Foundation Model to Estimate Laboratory Values from Single-Lead ECG Signals 

**Authors**: Yujie Xiao, Gongzhen Tang, Wenhui Liu, Jun Li, Guangkun Nie, Zhuoran Kan, Deyun Zhang, Qinghao Zhao, Shenda Hong  

**Link**: [PDF](https://arxiv.org/pdf/2510.22301)  

**Abstract**: Timely access to laboratory values is critical for clinical decision-making, yet current approaches rely on invasive venous sampling and are intrinsically delayed. Electrocardiography (ECG), as a non-invasive and widely available signal, offers a promising modality for rapid laboratory estimation. Recent progress in deep learning has enabled the extraction of latent hematological signatures from ECGs. However, existing models are constrained by low signal-to-noise ratios, substantial inter-individual variability, limited data diversity, and suboptimal generalization, especially when adapted to low-lead wearable devices. In this work, we conduct an exploratory study leveraging transfer learning to fine-tune ECGFounder, a large-scale pre-trained ECG foundation model, on the Multimodal Clinical Monitoring in the Emergency Department (MC-MED) dataset from Stanford. We generated a corpus of more than 20 million standardized ten-second ECG segments to enhance sensitivity to subtle biochemical correlates. On internal validation, the model demonstrated strong predictive performance (area under the curve above 0.65) for thirty-three laboratory indicators, moderate performance (between 0.55 and 0.65) for fifty-nine indicators, and limited performance (below 0.55) for sixteen indicators. This study provides an efficient artificial-intelligence driven solution and establishes the feasibility scope for real-time, non-invasive estimation of laboratory values. 

---
# T2I-RiskyPrompt: A Benchmark for Safety Evaluation, Attack, and Defense on Text-to-Image Model 

**Authors**: Chenyu Zhang, Tairen Zhang, Lanjun Wang, Ruidong Chen, Wenhui Li, Anan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.22300)  

**Abstract**: Using risky text prompts, such as pornography and violent prompts, to test the safety of text-to-image (T2I) models is a critical task. However, existing risky prompt datasets are limited in three key areas: 1) limited risky categories, 2) coarse-grained annotation, and 3) low effectiveness. To address these limitations, we introduce T2I-RiskyPrompt, a comprehensive benchmark designed for evaluating safety-related tasks in T2I models. Specifically, we first develop a hierarchical risk taxonomy, which consists of 6 primary categories and 14 fine-grained subcategories. Building upon this taxonomy, we construct a pipeline to collect and annotate risky prompts. Finally, we obtain 6,432 effective risky prompts, where each prompt is annotated with both hierarchical category labels and detailed risk reasons. Moreover, to facilitate the evaluation, we propose a reason-driven risky image detection method that explicitly aligns the MLLM with safety annotations. Based on T2I-RiskyPrompt, we conduct a comprehensive evaluation of eight T2I models, nine defense methods, five safety filters, and five attack strategies, offering nine key insights into the strengths and limitations of T2I model safety. Finally, we discuss potential applications of T2I-RiskyPrompt across various research fields. The dataset and code are provided in this https URL. 

---
# Does Homophily Help in Robust Test-time Node Classification? 

**Authors**: Yan Jiang, Ruihong Qiu, Zi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.22289)  

**Abstract**: Homophily, the tendency of nodes from the same class to connect, is a fundamental property of real-world graphs, underpinning structural and semantic patterns in domains such as citation networks and social networks. Existing methods exploit homophily through designing homophily-aware GNN architectures or graph structure learning strategies, yet they primarily focus on GNN learning with training graphs. However, in real-world scenarios, test graphs often suffer from data quality issues and distribution shifts, such as domain shifts across users from different regions in social networks and temporal evolution shifts in citation network graphs collected over varying time periods. These factors significantly compromise the pre-trained model's robustness, resulting in degraded test-time performance. With empirical observations and theoretical analysis, we reveal that transforming the test graph structure by increasing homophily in homophilic graphs or decreasing it in heterophilic graphs can significantly improve the robustness and performance of pre-trained GNNs on node classifications, without requiring model training or update. Motivated by these insights, a novel test-time graph structural transformation method grounded in homophily, named GrapHoST, is proposed. Specifically, a homophily predictor is developed to discriminate test edges, facilitating adaptive test-time graph structural transformation by the confidence of predicted homophily scores. Extensive experiments on nine benchmark datasets under a range of test-time data quality issues demonstrate that GrapHoST consistently achieves state-of-the-art performance, with improvements of up to 10.92%. Our code has been released at this https URL. 

---
# Supervised Fine-Tuning or In-Context Learning? Evaluating LLMs for Clinical NER 

**Authors**: Andrei Baroian  

**Link**: [PDF](https://arxiv.org/pdf/2510.22285)  

**Abstract**: We study clinical Named Entity Recognition (NER) on the CADEC corpus and compare three families of approaches: (i) BERT-style encoders (BERT Base, BioClinicalBERT, RoBERTa-large), (ii) GPT-4o used with few-shot in-context learning (ICL) under simple vs.\ complex prompts, and (iii) GPT-4o with supervised fine-tuning (SFT). All models are evaluated on standard NER metrics over CADEC's five entity types (ADR, Drug, Disease, Symptom, Finding). RoBERTa-large and BioClinicalBERT offer limited improvements over BERT Base, showing the limit of these family of models. Among LLM settings, simple ICL outperforms a longer, instruction-heavy prompt, and SFT achieves the strongest overall performance (F1 $\approx$ 87.1%), albeit with higher cost. We find that the LLM achieve higher accuracy on simplified tasks, restricting classification to two labels. 

---
# CityRiSE: Reasoning Urban Socio-Economic Status in Vision-Language Models via Reinforcement Learning 

**Authors**: Tianhui Liu, Hetian Pang, Xin Zhang, Jie Feng, Yong Li, Pan Hui  

**Link**: [PDF](https://arxiv.org/pdf/2510.22282)  

**Abstract**: Harnessing publicly available, large-scale web data, such as street view and satellite imagery, urban socio-economic sensing is of paramount importance for achieving global sustainable development goals. With the emergence of Large Vision-Language Models (LVLMs), new opportunities have arisen to solve this task by treating it as a multi-modal perception and understanding problem. However, recent studies reveal that LVLMs still struggle with accurate and interpretable socio-economic predictions from visual data. To address these limitations and maximize the potential of LVLMs, we introduce \textbf{CityRiSE}, a novel framework for \textbf{R}eason\textbf{i}ng urban \textbf{S}ocio-\textbf{E}conomic status in LVLMs through pure reinforcement learning (RL). With carefully curated multi-modal data and verifiable reward design, our approach guides the LVLM to focus on semantically meaningful visual cues, enabling structured and goal-oriented reasoning for generalist socio-economic status prediction. Experiments demonstrate that CityRiSE with emergent reasoning process significantly outperforms existing baselines, improving both prediction accuracy and generalization across diverse urban contexts, particularly for prediction on unseen cities and unseen indicators. This work highlights the promise of combining RL and LVLMs for interpretable and generalist urban socio-economic sensing. 

---
# A Multi-level Analysis of Factors Associated with Student Performance: A Machine Learning Approach to the SAEB Microdata 

**Authors**: Rodrigo Tertulino, Ricardo Almeida  

**Link**: [PDF](https://arxiv.org/pdf/2510.22266)  

**Abstract**: Identifying the factors that influence student performance in basic education is a central challenge for formulating effective public policies in Brazil. This study introduces a multi-level machine learning approach to classify the proficiency of 9th-grade and high school students using microdata from the System of Assessment of Basic Education (SAEB). Our model uniquely integrates four data sources: student socioeconomic characteristics, teacher professional profiles, school indicators, and director management profiles. A comparative analysis of four ensemble algorithms confirmed the superiority of a Random Forest model, which achieved 90.2% accuracy and an Area Under the Curve (AUC) of 96.7%. To move beyond prediction, we applied Explainable AI (XAI) using SHAP, which revealed that the school's average socioeconomic level is the most dominant predictor, demonstrating that systemic factors have a greater impact than individual characteristics in isolation. The primary conclusion is that academic performance is a systemic phenomenon deeply tied to the school's ecosystem. This study provides a data-driven, interpretable tool to inform policies aimed at promoting educational equity by addressing disparities between schools. 

---
# PatenTEB: A Comprehensive Benchmark and Model Family for Patent Text Embedding 

**Authors**: Iliass Ayaou, Denis Cavallucci  

**Link**: [PDF](https://arxiv.org/pdf/2510.22264)  

**Abstract**: Patent text embeddings enable prior art search, technology landscaping, and patent analysis, yet existing benchmarks inadequately capture patent-specific challenges. We introduce PatenTEB, a comprehensive benchmark comprising 15 tasks across retrieval, classification, paraphrase, and clustering, with 2.06 million examples. PatenTEB employs domain-stratified splits, domain specific hard negative mining, and systematic coverage of asymmetric fragment-to-document matching scenarios absent from general embedding benchmarks. We develop the patembed model family through multi-task training, spanning 67M to 344M parameters with context lengths up to 4096 tokens. External validation shows strong generalization: patembed-base achieves state-of-the-art on MTEB BigPatentClustering.v2 (0.494 V-measure vs. 0.445 previous best), while patembed-large achieves 0.377 NDCG@100 on DAPFAM. Systematic ablations reveal that multi-task training improves external generalization despite minor benchmark costs, and that domain-pretrained initialization provides consistent advantages across task families. All resources will be made available at this https URL. Keywords: patent retrieval, sentence embeddings, multi-task learning, asymmetric retrieval, benchmark evaluation, contrastive learning. 

---
# Epistemic Deep Learning: Enabling Machine Learning Models to Know When They Do Not Know 

**Authors**: Shireen Kudukkil Manchingal  

**Link**: [PDF](https://arxiv.org/pdf/2510.22261)  

**Abstract**: Machine learning has achieved remarkable successes, yet its deployment in safety-critical domains remains hindered by an inherent inability to manage uncertainty, resulting in overconfident and unreliable predictions when models encounter out-of-distribution data, adversarial perturbations, or naturally fluctuating environments. This thesis, titled Epistemic Deep Learning: Enabling Machine Learning Models to 'Know When They Do Not Know', addresses these critical challenges by advancing the paradigm of Epistemic Artificial Intelligence, which explicitly models and quantifies epistemic uncertainty: the uncertainty arising from limited, biased, or incomplete training data, as opposed to the irreducible randomness of aleatoric uncertainty, thereby empowering models to acknowledge their limitations and refrain from overconfident decisions when uncertainty is high.
Central to this work is the development of the Random-Set Neural Network (RS-NN), a novel methodology that leverages random set theory to predict belief functions over sets of classes, capturing the extent of epistemic uncertainty through the width of associated credal sets, applications of RS-NN, including its adaptation to Large Language Models (LLMs) and its deployment in weather classification for autonomous racing. In addition, the thesis proposes a unified evaluation framework for uncertainty-aware classifiers. Extensive experiments validate that integrating epistemic awareness into deep learning not only mitigates the risks associated with overconfident predictions but also lays the foundation for a paradigm shift in artificial intelligence, where the ability to 'know when it does not know' becomes a hallmark of robust and dependable systems. The title encapsulates the core philosophy of this work, emphasizing that true intelligence involves recognizing and managing the limits of one's own knowledge. 

---
# LUNA: Efficient and Topology-Agnostic Foundation Model for EEG Signal Analysis 

**Authors**: Berkay Döner, Thorir Mar Ingolfsson, Luca Benini, Yawei Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.22257)  

**Abstract**: Electroencephalography (EEG) offers a non-invasive lens into human brain activity, but building large-scale models is hampered by topological heterogeneity: each public EEG data defines its own electrode layout, limiting generalization. We introduce LUNA (Latent Unified Network Architecture), a self-supervised foundation model that reconciles disparate electrode geometries while scaling linearly -- not quadratically -- with channel count. LUNA compresses multi-channel EEG into a fixed-size, topology-agnostic latent space via learned queries and cross-attention. Downstream transformer blocks then operate exclusively on this latent representation using patch-wise temporal self-attention, decoupling computation from electrode count. Pre-trained on TUEG and Siena (over 21,000 hours of raw EEG across diverse montages) using a masked-patch reconstruction objective, LUNA transfers effectively to four downstream tasks: abnormality detection, artifact rejection, slowing classification, and emotion recognition. It demonstrates highly competitive performance across several benchmarks, achieving state-of-the-art results on TUAR and TUSL, e.g., 0.921 AUROC on TUAR, while reducing FLOPs by 300x and trimming GPU memory use by up to 10x. Critically, these gains are consistent across all evaluated electrode configurations. Code is available at this https URL 

---
# You Don't Need Prompt Engineering Anymore: The Prompting Inversion 

**Authors**: Imran Khan  

**Link**: [PDF](https://arxiv.org/pdf/2510.22251)  

**Abstract**: Prompt engineering, particularly Chain-of-Thought (CoT) prompting, significantly enhances LLM reasoning capabilities. We introduce "Sculpting," a constrained, rule-based prompting method designed to improve upon standard CoT by reducing errors from semantic ambiguity and flawed common sense.
We evaluate three prompting strategies (Zero Shot, standard CoT, and Sculpting) across three OpenAI model generations (gpt-4o-mini, gpt-4o, gpt-5) using the GSM8K mathematical reasoning benchmark (1,317 problems).
Our findings reveal a "Prompting Inversion": Sculpting provides advantages on gpt-4o (97% vs. 93% for standard CoT), but becomes detrimental on gpt-5 (94.00% vs. 96.36% for CoT on full benchmark). We trace this to a "Guardrail-to-Handcuff" transition where constraints preventing common-sense errors in mid-tier models induce hyper-literalism in advanced models. Our detailed error analysis demonstrates that optimal prompting strategies must co-evolve with model capabilities, suggesting simpler prompts for more capable models. 

---
# Real-Time Semantic Segmentation on FPGA for Autonomous Vehicles Using LMIINet with the CGRA4ML Framework 

**Authors**: Amir Mohammad Khadem Hosseini, Sattar Mirzakuchaki  

**Link**: [PDF](https://arxiv.org/pdf/2510.22243)  

**Abstract**: Semantic segmentation has emerged as a fundamental problem in computer vision, gaining particular importance in real-time applications such as autonomous driving. The main challenge is achieving high accuracy while operating under computational and hardware constraints. In this research, we present an FPGA-based implementation of real-time semantic segmentation leveraging the lightweight LMIINet architecture and the Coarse-Grained Reconfigurable Array for Machine Learning (CGRA4ML) hardware framework. The model was trained using Quantization-Aware Training (QAT) with 8-bit precision on the Cityscapes dataset, reducing memory footprint by a factor of four while enabling efficient fixed-point computations. Necessary modifications were applied to adapt the model to CGRA4ML constraints, including simplifying skip connections, employing hardware-friendly operations such as depthwise-separable and 1A-1 convolutions, and redesigning parts of the Flatten Transformer. Our implementation achieves approximately 90% pixel accuracy and 45% mean Intersection-over-Union (mIoU), operating in real-time at 20 frames per second (FPS) with 50.1 ms latency on the ZCU104 FPGA board. The results demonstrate the potential of CGRA4ML, with its flexibility in mapping modern layers and off-chip memory utilization for skip connections, provides a path for implementing advanced semantic segmentation networks on FPGA for real-time applications to outperform traditional GPU solutions in terms of power efficiency while maintaining competitive accuracy. The code for this project is publicly available at this https URL cgra4ml_semantic_segmentation 

---
# PaperAsk: A Benchmark for Reliability Evaluation of LLMs in Paper Search and Reading 

**Authors**: Yutao Wu, Xiao Liu, Yunhao Feng, Jiale Ding, Xingjun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2510.22242)  

**Abstract**: Large Language Models (LLMs) increasingly serve as research assistants, yet their reliability in scholarly tasks remains under-evaluated. In this work, we introduce PaperAsk, a benchmark that systematically evaluates LLMs across four key research tasks: citation retrieval, content extraction, paper discovery, and claim verification. We evaluate GPT-4o, GPT-5, and Gemini-2.5-Flash under realistic usage conditions-via web interfaces where search operations are opaque to the user. Through controlled experiments, we find consistent reliability failures: citation retrieval fails in 48-98% of multi-reference queries, section-specific content extraction fails in 72-91% of cases, and topical paper discovery yields F1 scores below 0.32, missing over 60% of relevant literature. Further human analysis attributes these failures to the uncontrolled expansion of retrieved context and the tendency of LLMs to prioritize semantically relevant text over task instructions. Across basic tasks, the LLMs display distinct failure behaviors: ChatGPT often withholds responses rather than risk errors, whereas Gemini produces fluent but fabricated answers. To address these issues, we develop lightweight reliability classifiers trained on PaperAsk data to identify unreliable outputs. PaperAsk provides a reproducible and diagnostic framework for advancing the reliability evaluation of LLM-based scholarly assistance systems. 

---
# Rational Adversaries and the Maintenance of Fragility: A Game-Theoretic Theory of Rational Stagnation 

**Authors**: Daisuke Hirota  

**Link**: [PDF](https://arxiv.org/pdf/2510.22232)  

**Abstract**: Cooperative systems often remain in persistently suboptimal yet stable states. This paper explains such "rational stagnation" as an equilibrium sustained by a rational adversary whose utility follows the principle of potential loss, $u_{D} = U_{ideal} - U_{actual}$. Starting from the Prisoner's Dilemma, we show that the transformation $u_{i}' = a\,u_{i} + b\,u_{j}$ and the ratio of mutual recognition $w = b/a$ generate a fragile cooperation band $[w_{\min},\,w_{\max}]$ where both (C,C) and (D,D) are equilibria. Extending to a dynamic model with stochastic cooperative payoffs $R_{t}$ and intervention costs $(C_{c},\,C_{m})$, a Bellman-style analysis yields three strategic regimes: immediate destruction, rational stagnation, and intervention abandonment. The appendix further generalizes the utility to a reference-dependent nonlinear form and proves its stability under reference shifts, ensuring robustness of the framework. Applications to social-media algorithms and political trust illustrate how adversarial rationality can deliberately preserve fragility. 

---
# When Fewer Layers Break More Chains: Layer Pruning Harms Test-Time Scaling in LLMs 

**Authors**: Keyu Wang, Tian Lyu, Guinan Su, Jonas Geiping, Lu Yin, Marco Canini, Shiwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.22228)  

**Abstract**: Layer pruning has emerged as a widely adopted technique for improving the efficiency of large language models (LLMs). Although existing methods demonstrate strong performance retention on general knowledge tasks, their effect on long-chain reasoning, a more brittle yet crucial capability, remains largely unexplored. In this work, we study the impact of layer pruning on long-chain reasoning through the lens of test-time scaling, a key mechanism in modern LLMs that enables strong reasoning capacity by allocating more computation at inference time. With extensive experiments, we demonstrate that pruning even one or two layers can severely impair test-time scaling, with performance collapsing drastically on long reasoning benchmarks even when performance on knowledge-intensive and shallow reasoning tasks remains stable. Furthermore, we find that standard supervised fine-tuning remedies fail to recover test-time scaling once it has deteriorated. Through in-depth analyses, we identify the mechanisms underlying this fragility of test-time scaling and highlight the fundamental risks of applying layer pruning to reasoning-intensive LLMs. These findings call for a rethinking of layer pruning strategies and provide insights for developing methods that preserve the robustness of reasoning. We open-source the codebase in \href{this https URL}{this https URL}. 

---
# Taming Silent Failures: A Framework for Verifiable AI Reliability 

**Authors**: Guan-Yan Yang, Farn Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.22224)  

**Abstract**: The integration of Artificial Intelligence (AI) into safety-critical systems introduces a new reliability paradigm: silent failures, where AI produces confident but incorrect outputs that can be dangerous. This paper introduces the Formal Assurance and Monitoring Environment (FAME), a novel framework that confronts this challenge. FAME synergizes the mathematical rigor of offline formal synthesis with the vigilance of online runtime monitoring to create a verifiable safety net around opaque AI components. We demonstrate its efficacy in an autonomous vehicle perception system, where FAME successfully detected 93.5% of critical safety violations that were otherwise silent. By contextualizing our framework within the ISO 26262 and ISO/PAS 8800 standards, we provide reliability engineers with a practical, certifiable pathway for deploying trustworthy AI. FAME represents a crucial shift from accepting probabilistic performance to enforcing provable safety in next-generation systems. 

---
# Estimating the Error of Large Language Models at Pairwise Text Comparison 

**Authors**: Tianyi Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.22219)  

**Abstract**: We measure LLMs' output error at pairwise text comparison, noting the probability of error in their preferences. Our method does not rely on the ground truth and supports two scenarios: (i) uniform error rate regardless of the order of comparison, estimated with two comparisons for each text pair with either text placed first; (ii) binary positional bias assuming distinct error rates for the two orders of comparison, estimated with repeated comparisons between the texts. The Copeland counting constructs a ranking over the compared texts from pairwise preferences; the ranking reveals the poor scalability of LLM-based pairwise comparison and helps yield the estimates for LLMs' error rates. We apply the method to six LLMs (ChatGPT, Claude, DeepSeek, Gemini, Grok, Qwen) with five types of text input and obtain consistent estimates of LLMs' error. In general, the measured two positional bias terms are similar, close to the uniform error. Considering both the error rates and the robustness to the variation of prompts, Claude obtained the most desirable performance in this experiment. Our model outperforms the biased Bradley-Terry model and the commutativity score in indicating LLMs' error at this task. 

---
# GALA: A GlobAl-LocAl Approach for Multi-Source Active Domain Adaptation 

**Authors**: Juepeng Zheng, Peifeng Zhang, Yibin Wen, Qingmei Li, Yang Zhang, Haohuan Fu  

**Link**: [PDF](https://arxiv.org/pdf/2510.22214)  

**Abstract**: Domain Adaptation (DA) provides an effective way to tackle target-domain tasks by leveraging knowledge learned from source domains. Recent studies have extended this paradigm to Multi-Source Domain Adaptation (MSDA), which exploits multiple source domains carrying richer and more diverse transferable information. However, a substantial performance gap still remains between adaptation-based methods and fully supervised learning. In this paper, we explore a more practical and challenging setting, named Multi-Source Active Domain Adaptation (MS-ADA), to further enhance target-domain performance by selectively acquiring annotations from the target domain. The key difficulty of MS-ADA lies in designing selection criteria that can jointly handle inter-class diversity and multi-source domain variation. To address these challenges, we propose a simple yet effective GALA strategy (GALA), which combines a global k-means clustering step for target-domain samples with a cluster-wise local selection criterion, effectively tackling the above two issues in a complementary manner. Our proposed GALA is plug-and-play and can be seamlessly integrated into existing DA frameworks without introducing any additional trainable parameters. Extensive experiments on three standard DA benchmarks demonstrate that GALA consistently outperforms prior active learning and active DA methods, achieving performance comparable to the fully-supervised upperbound while using only 1% of the target annotations. 

---
# LSPRAG: LSP-Guided RAG for Language-Agnostic Real-Time Unit Test Generation 

**Authors**: Gwihwan Go, Quan Zhang, Chijin Zhou, Zhao Wei, Yu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2510.22210)  

**Abstract**: Automated unit test generation is essential for robust software development, yet existing approaches struggle to generalize across multiple programming languages and operate within real-time development. While Large Language Models (LLMs) offer a promising solution, their ability to generate high coverage test code depends on prompting a concise context of the focal method. Current solutions, such as Retrieval-Augmented Generation, either rely on imprecise similarity-based searches or demand the creation of costly, language-specific static analysis pipelines. To address this gap, we present LSPRAG, a framework for concise-context retrieval tailored for real-time, language-agnostic unit test generation. LSPRAG leverages off-the-shelf Language Server Protocol (LSP) back-ends to supply LLMs with precise symbol definitions and references in real time. By reusing mature LSP servers, LSPRAG provides an LLM with language-aware context retrieval, requiring minimal per-language engineering effort. We evaluated LSPRAG on open-source projects spanning Java, Go, and Python. Compared to the best performance of baselines, LSPRAG increased line coverage by up to 174.55% for Golang, 213.31% for Java, and 31.57% for Python. 

---
# Right Place, Right Time: Market Simulation-based RL for Execution Optimisation 

**Authors**: Ollie Olby, Andreea Bacalum, Rory Baggott, Namid Stillman  

**Link**: [PDF](https://arxiv.org/pdf/2510.22206)  

**Abstract**: Execution algorithms are vital to modern trading, they enable market participants to execute large orders while minimising market impact and transaction costs. As these algorithms grow more sophisticated, optimising them becomes increasingly challenging. In this work, we present a reinforcement learning (RL) framework for discovering optimal execution strategies, evaluated within a reactive agent-based market simulator. This simulator creates reactive order flow and allows us to decompose slippage into its constituent components: market impact and execution risk. We assess the RL agent's performance using the efficient frontier based on work by Almgren and Chriss, measuring its ability to balance risk and cost. Results show that the RL-derived strategies consistently outperform baselines and operate near the efficient frontier, demonstrating a strong ability to optimise for risk and impact. These findings highlight the potential of reinforcement learning as a powerful tool in the trader's toolkit. 

---
# Bridging Perception and Reasoning: Dual-Pipeline Neuro-Symbolic Landing for UAVs in Cluttered Environments 

**Authors**: Weixian Qian, Sebastian Schroder, Yao Deng, Jiaohong Yao, Linfeng Liang, Xiao Cheng, Richard Han, Xi Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.22204)  

**Abstract**: Autonomous landing in unstructured (cluttered, uneven, and map-poor) environments is a core requirement for Unmanned Aerial Vehicles (UAVs), yet purely vision-based or deep learning models often falter under covariate shift and provide limited interpretability. We propose NeuroSymLand, a neuro-symbolic framework that tightly couples two complementary pipelines: (i) an offline pipeline, where Large Language Models (LLMs) and human-in-the-loop refinement synthesize Scallop code from diverse landing scenarios, distilling generalizable and verifiable symbolic knowledge; and (ii) an online pipeline, where a compact foundation-based semantic segmentation model generates probabilistic Scallop facts that are composed into semantic scene graphs for real-time deductive reasoning. This design combines the perceptual strengths of lightweight foundation models with the interpretability and verifiability of symbolic reasoning. Node attributes (e.g., flatness, area) and edge relations (adjacency, containment, proximity) are computed with geometric routines rather than learned, avoiding the data dependence and latency of train-time graph builders. The resulting Scallop program encodes landing principles (avoid water and obstacles; prefer large, flat, accessible regions) and yields calibrated safety scores with ranked Regions of Interest (ROIs) and human-readable justifications. Extensive evaluations across datasets, diverse simulation maps, and real UAV hardware show that NeuroSymLand achieves higher accuracy, stronger robustness to covariate shift, and superior efficiency compared with state-of-the-art baselines, while advancing UAV safety and reliability in emergency response, surveillance, and delivery missions. 

---
# Multi-dataset Joint Pre-training of Emotional EEG Enables Generalizable Affective Computing 

**Authors**: Qingzhu Zhang, Jiani Zhong, Zongsheng Li, Xinke Shen, Quanying Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.22197)  

**Abstract**: Task-specific pre-training is essential when task representations diverge from generic pre-training features. Existing task-general pre-training EEG models struggle with complex tasks like emotion recognition due to mismatches between task-specific features and broad pre-training approaches. This work aims to develop a task-specific multi-dataset joint pre-training framework for cross-dataset emotion recognition, tackling problems of large inter-dataset distribution shifts, inconsistent emotion category definitions, and substantial inter-subject variability. We introduce a cross-dataset covariance alignment loss to align second-order statistical properties across datasets, enabling robust generalization without the need for extensive labels or per-subject calibration. To capture the long-term dependency and complex dynamics of EEG, we propose a hybrid encoder combining a Mamba-like linear attention channel encoder and a spatiotemporal dynamics model. Our method outperforms state-of-the-art large-scale EEG models by an average of 4.57% in AUROC for few-shot emotion recognition and 11.92% in accuracy for zero-shot generalization to a new dataset. Performance scales with the increase of datasets used in pre-training. Multi-dataset joint pre-training achieves a performance gain of 8.55% over single-dataset training. This work provides a scalable framework for task-specific pre-training and highlights its benefit in generalizable affective computing. Our code is available at this https URL. 

---
# Scaling Non-Parametric Sampling with Representation 

**Authors**: Vincent Lu, Aaron Truong, Zeyu Yun, Yubei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.22196)  

**Abstract**: Scaling and architectural advances have produced strikingly photorealistic image generative models, yet their mechanisms still remain opaque. Rather than advancing scaling, our goal is to strip away complicated engineering tricks and propose a simple, non-parametric generative model. Our design is grounded in three principles of natural images-(i) spatial non-stationarity, (ii) low-level regularities, and (iii) high-level semantics-and defines each pixel's distribution from its local context window. Despite its minimal architecture and no training, the model produces high-fidelity samples on MNIST and visually compelling CIFAR-10 images. This combination of simplicity and strong empirical performance points toward a minimal theory of natural-image structure. The model's white-box nature also allows us to have a mechanistic understanding of how the model generalizes and generates diverse images. We study it by tracing each generated pixel back to its source images. These analyses reveal a simple, compositional procedure for "part-whole generalization", suggesting a hypothesis for how large neural network generative models learn to generalize. 

---
# Solving Continuous Mean Field Games: Deep Reinforcement Learning for Non-Stationary Dynamics 

**Authors**: Lorenzo Magnino, Kai Shao, Zida Wu, Jiacheng Shen, Mathieu Laurière  

**Link**: [PDF](https://arxiv.org/pdf/2510.22158)  

**Abstract**: Mean field games (MFGs) have emerged as a powerful framework for modeling interactions in large-scale multi-agent systems. Despite recent advancements in reinforcement learning (RL) for MFGs, existing methods are typically limited to finite spaces or stationary models, hindering their applicability to real-world problems. This paper introduces a novel deep reinforcement learning (DRL) algorithm specifically designed for non-stationary continuous MFGs. The proposed approach builds upon a Fictitious Play (FP) methodology, leveraging DRL for best-response computation and supervised learning for average policy representation. Furthermore, it learns a representation of the time-dependent population distribution using a Conditional Normalizing Flow. To validate the effectiveness of our method, we evaluate it on three different examples of increasing complexity. By addressing critical limitations in scalability and density approximation, this work represents a significant advancement in applying DRL techniques to complex MFG problems, bringing the field closer to real-world multi-agent systems. 

---
# Power to the Clients: Federated Learning in a Dictatorship Setting 

**Authors**: Mohammadsajad Alipour, Mohammad Mohammadi Amiri  

**Link**: [PDF](https://arxiv.org/pdf/2510.22149)  

**Abstract**: Federated learning (FL) has emerged as a promising paradigm for decentralized model training, enabling multiple clients to collaboratively learn a shared model without exchanging their local data. However, the decentralized nature of FL also introduces vulnerabilities, as malicious clients can compromise or manipulate the training process. In this work, we introduce dictator clients, a novel, well-defined, and analytically tractable class of malicious participants capable of entirely erasing the contributions of all other clients from the server model, while preserving their own. We propose concrete attack strategies that empower such clients and systematically analyze their effects on the learning process. Furthermore, we explore complex scenarios involving multiple dictator clients, including cases where they collaborate, act independently, or form an alliance in order to ultimately betray one another. For each of these settings, we provide a theoretical analysis of their impact on the global model's convergence. Our theoretical algorithms and findings about the complex scenarios including multiple dictator clients are further supported by empirical evaluations on both computer vision and natural language processing benchmarks. 

---
# Probing Neural Combinatorial Optimization Models 

**Authors**: Zhiqin Zhang, Yining Ma, Zhiguang Cao, Hoong Chuin Lau  

**Link**: [PDF](https://arxiv.org/pdf/2510.22131)  

**Abstract**: Neural combinatorial optimization (NCO) has achieved remarkable performance, yet its learned model representations and decision rationale remain a black box. This impedes both academic research and practical deployment, since researchers and stakeholders require deeper insights into NCO models. In this paper, we take the first critical step towards interpreting NCO models by investigating their representations through various probing tasks. Moreover, we introduce a novel probing tool named Coefficient Significance Probing (CS-Probing) to enable deeper analysis of NCO representations by examining the coefficients and statistical significance during probing. Extensive experiments and analysis reveal that NCO models encode low-level information essential for solution construction, while capturing high-level knowledge to facilitate better decisions. Using CS-Probing, we find that prevalent NCO models impose varying inductive biases on their learned representations, uncover direct evidence related to model generalization, and identify key embedding dimensions associated with specific knowledge. These insights can be potentially translated into practice, for example, with minor code modifications, we improve the generalization of the analyzed model. Our work represents a first systematic attempt to interpret black-box NCO models, showcasing probing as a promising tool for analyzing their internal mechanisms and revealing insights for the NCO community. The source code is publicly available. 

---
# Efficient Utility-Preserving Machine Unlearning with Implicit Gradient Surgery 

**Authors**: Shiji Zhou, Tianbai Yu, Zhi Zhang, Heng Chang, Xiao Zhou, Dong Wu, Han Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.22124)  

**Abstract**: Machine unlearning (MU) aims to efficiently remove sensitive or harmful memory from a pre-trained model. The key challenge is to balance the potential tradeoff between unlearning efficacy and utility preservation, which involves forgetting undesirable information as defined while maintaining the model's original performance. One potential way to tackle this problem is to use multi-objective optimization to jointly optimize both the unlearning and utility preservation objectives. However, existing multi-objective methods only guarantee finding a Pareto-optimal solution without fine-grained control, which causes under-optimization of the unlearning objective. To this end, we first model MU as a constrained optimization problem, that is, optimizing the unlearning objective under the constraint of a bounded increase for utility loss. We then show that solving this optimization problem is equivalent to unilateral gradient surgery on the unlearning objective. To resolve the additional computational cost brought by gradient surgery, we propose an implicit gradient surgery method, which approximates the solution to the aforementioned constrained optimization problem via only one backpropagation, thereby achieving efficient utility-preserving MU. Theoretically, we provide a tight convergence analysis of the algorithm. Empirically, our extensive experiments show that the proposed algorithm achieves better tradeoff results than existing baselines. Codes are available at this https URL. 

---
# GRAID: Enhancing Spatial Reasoning of VLMs Through High-Fidelity Data Generation 

**Authors**: Karim Elmaaroufi, Liheng Lai, Justin Svegliato, Yutong Bai, Sanjit A. Seshia, Matei Zaharia  

**Link**: [PDF](https://arxiv.org/pdf/2510.22118)  

**Abstract**: Vision Language Models (VLMs) achieve strong performance on many vision-language tasks but often struggle with spatial reasoning\textemdash{}a prerequisite for many applications. Empirically, we find that a dataset produced by a current training data generation pipeline has a 57.6\% human validation rate. These rates stem from current limitations: single-image 3D reconstruction introduces cascading modeling errors and requires wide answer tolerances, while caption-based methods require hyper-detailed annotations and suffer from generative hallucinations. We present GRAID, built on the key insight that qualitative spatial relationships can be reliably determined from 2D geometric primitives alone. By operating exclusively on 2D bounding boxes from standard object detectors, GRAID avoids both 3D reconstruction errors and generative hallucinations, resulting in datasets that are of higher quality than existing tools that produce similar datasets as validated by human evaluations. We apply our framework to the BDD100k, NuImages, and Waymo datasets, generating over 8.5 million high-quality VQA pairs creating questions spanning spatial relations, counting, ranking, and size comparisons. We evaluate one of the datasets and find it achieves 91.16\% human-validated accuracy\textemdash{}compared to 57.6\% on a dataset generated by recent work. % or recent work Critically, we demonstrate that when trained on GRAID data, models learn spatial reasoning concepts that generalize: models fine-tuned on 6 question types improve on over 10 held-out types, with accuracy gains of 47.5\% on BDD and 37.9\% on NuImages for Llama 3.2B 11B, and when trained on all questions types, achieve improvements on several existing benchmarks such as BLINK. The GRAID framework, datasets, and additional information can be found on our \href{this https URL}{project page}. 

---
# When UAV Swarm Meets IRS: Collaborative Secure Communications in Low-altitude Wireless Networks 

**Authors**: Jiahui Li, Xinyue Liang, Geng Sun, Hui Kang, Jiacheng Wang, Dusit Niyato, Shiwen Mao, Abbas Jamalipour  

**Link**: [PDF](https://arxiv.org/pdf/2510.22117)  

**Abstract**: Low-altitude wireless networks (LAWNs) represent a promising architecture that integrates unmanned aerial vehicles (UAVs) as aerial nodes to provide enhanced coverage, reliability, and throughput for diverse applications. However, these networks face significant security vulnerabilities from both known and potential unknown eavesdroppers, which may threaten data confidentiality and system integrity. To solve this critical issue, we propose a novel secure communication framework for LAWNs where the selected UAVs within a swarm function as a virtual antenna array (VAA), complemented by intelligent reflecting surface (IRS) to create a robust defense against eavesdropping attacks. Specifically, we formulate a multi-objective optimization problem that simultaneously maximizes the secrecy rate while minimizing the maximum sidelobe level and total energy consumption, requiring joint optimization of UAV excitation current weights, flight trajectories, and IRS phase shifts. This problem presents significant difficulties due to the dynamic nature of the system and heterogeneous components. Thus, we first transform the problem into a heterogeneous Markov decision process (MDP). Then, we propose a heterogeneous multi-agent control approach (HMCA) that integrates a dedicated IRS control policy with a multi-agent soft actor-critic framework for UAV control, which enables coordinated operation across heterogeneous network elements. Simulation results show that the proposed HMCA achieves superior performance compared to baseline approaches in terms of secrecy rate improvement, sidelobe suppression, and energy efficiency. Furthermore, we find that the collaborative and passive beamforming synergy between VAA and IRS creates robust security guarantees when the number of UAVs increases. 

---
# Every Activation Boosted: Scaling General Reasoner to 1 Trillion Open Language Foundation 

**Authors**: Ling-Team, Ang Li, Ben Liu, Binbin Hu, Bing Li, Bingwei Zeng, Borui Ye, Caizhi Tang, Changxin Tian, Chao Huang, Chao Zhang, Chen Qian, Chenchen Ju, Chenchen Li, Chengfu Tang, Chili Fu, Chunshao Ren, Chunwei Wu, Cong Zhang, Cunyin Peng, Dafeng Xu, Daixin Wang, Dalong Zhang, Dingnan Jin, Dingyuan Zhu, Dongke Hu, Fangzheng Zhao, Feifan Wu, Feng Zhu, Gangshan Wang, Haitao Zhang, Hailin Zhao, Hanxiao Zhang, Hanzi Wang, Hao Qian, Haoyi Yu, Heng Zhang, Hongliang Zhang, Hongzhi Luan, Huirong Dong, Huizhong Li, Jia Li, Jia Liu, Jialong Zhu, Jian Sha, Jianping Wei, Jiaolong Yang, Jieyue Ma, Jiewei Wu, Jinjing Huang, Jingyun Tian, Jingyuan Zhang, Jinquan Sun, Juanhui Tu, Jun Liu, Jun Xu, Jun Zhou, Junjie Ou, Junpeng Fang, Kaihong Zhang, Kaiqin Hu, Ke Shi, Kun Tang, Kunlong Chen, Lanyin Mei, Lei Liang, Lei Xu, Libo Zhang, Lin Ju, Lin Yuan, Ling Zhong, Lintao Ma, Lu Liu, Lu Yu, Lun Cai, Meiqi Zhu, Mengying Li, Min Chen, Minghao Xue, Minghong Cai, Mingming Yin, Peijie Jiang, Peilong Zhao, Pingping Liu, Qian Zhao, Qing Cui, Qingxiang Huang, Qingyuan Yang, Quankun Yu, Shaowei Wei, Shijie Lian, Shoujian Zheng, Shun Song, Shungen Zhang, Shuo Zhang, Siyuan Li, Song Liu, Ting Guo, Tong Zhao, Wanli Gu  

**Link**: [PDF](https://arxiv.org/pdf/2510.22115)  

**Abstract**: We introduce Ling 2.0, a series reasoning-oriented language foundation built upon the principle that every activation boosts reasoning capability. Designed to scale from tens of billions to one trillion parameters under a unified Mixture-of-Experts (MoE) paradigm, Ling 2.0 emphasizes high sparsity, cross-scale consistency, and efficiency guided by empirical scaling laws. The series includes three non-thinking (instruct) models - Ling-mini-2.0, Ling-flash-2.0, and Ling-1T - ranging from 16B to 1T total parameters and achieving up to 7-fold active-compute efficiency compared with dense counterparts. Ling 2.0 integrates coordinated innovations across model architecture, pre-training, post-training, and infrastructure: a high-sparsity MoE with MTP for efficient reasoning, reasoning-oriented data and mid-training CoT activation, reinforcement-based fine-tuning (DFT, Evo-CoT), and full-scale FP8 training with fine-grained heterogeneous pipelines. At the trillion scale, Ling-1T establishes a new Pareto frontier of reasoning accuracy versus computational efficiency, demonstrating that sparse activation, when properly aligned with reasoning objectives, enables scalable and efficient intelligence. Collectively, Ling 2.0 provides a coherent, open, and efficient foundation for advancing future reasoning and thinking models, including the Ring series built upon the same base. 

---
# Gradual Forgetting: Logarithmic Compression for Extending Transformer Context Windows 

**Authors**: Billy Dickson, Zoran Tiganj  

**Link**: [PDF](https://arxiv.org/pdf/2510.22109)  

**Abstract**: Most approaches to long-context processing increase the complexity of the transformer's internal architecture by integrating mechanisms such as recurrence or auxiliary memory modules. In this work, we introduce an alternative approach that modifies the input representation itself, rather than the transformer architecture. Inspired by cognitive models of human memory, our method applies a scale-invariant logarithmic compression to the input tokens. The resulting compressed representation is processed by a standard, unmodified transformer, preserving architectural simplicity. We evaluate this approach on the WikiText-103 and PG-19 language modeling benchmarks, showing a reduction in perplexity compared to uncompressed baselines. Moreover, performance improves consistently with longer compressed temporal contexts, showing that input-level logarithmic compression is a simple and effective way to extend a transformer's long-range memory. 

---
# STAR-RIS-assisted Collaborative Beamforming for Low-altitude Wireless Networks 

**Authors**: Xinyue Liang, Hui Kang, Junwei Che, Jiahui Li, Geng Sun, Qingqing Wu, Jiacheng Wang, Dusit Niyato  

**Link**: [PDF](https://arxiv.org/pdf/2510.22108)  

**Abstract**: While low-altitude wireless networks (LAWNs) based on uncrewed aerial vehicles (UAVs) offer high mobility, flexibility, and coverage for urban communications, they face severe signal attenuation in dense environments due to obstructions. To address this critical issue, we consider introducing collaborative beamforming (CB) of UAVs and omnidirectional reconfigurable beamforming (ORB) of simultaneous transmitting and reflecting reconfigurable intelligent surfaces (STAR-RIS) to enhance the signal quality and directionality. On this basis, we formulate a joint rate and energy optimization problem (JREOP) to maximize the transmission rate of the overall system, while minimizing the energy consumption of the UAV swarm. Due to the non-convex and NP-hard nature of JREOP, we propose a heterogeneous multi-agent collaborative dynamic (HMCD) optimization framework, which has two core components. The first component is a simulated annealing (SA)-based STAR-RIS control method, which dynamically optimizes reflection and transmission coefficients to enhance signal propagation. The second component is an improved multi-agent deep reinforcement learning (MADRL) control method, which incorporates a self-attention evaluation mechanism to capture interactions between UAVs and an adaptive velocity transition mechanism to enhance training stability. Simulation results demonstrate that HMCD outperforms various baselines in terms of convergence speed, average transmission rate, and energy consumption. Further analysis reveals that the average transmission rate of the overall system scales positively with both UAV count and STAR-RIS element numbers. 

---
# Discovering Latent Graphs with GFlowNets for Diverse Conditional Image Generation 

**Authors**: Bailey Trang, Parham Saremi, Alan Q. Wang, Fangrui Huang, Zahra TehraniNasab, Amar Kumar, Tal Arbel, Li Fei-Fei, Ehsan Adeli  

**Link**: [PDF](https://arxiv.org/pdf/2510.22107)  

**Abstract**: Capturing diversity is crucial in conditional and prompt-based image generation, particularly when conditions contain uncertainty that can lead to multiple plausible outputs. To generate diverse images reflecting this diversity, traditional methods often modify random seeds, making it difficult to discern meaningful differences between samples, or diversify the input prompt, which is limited in verbally interpretable diversity. We propose Rainbow, a novel conditional image generation framework, applicable to any pretrained conditional generative model, that addresses inherent condition/prompt uncertainty and generates diverse plausible images. Rainbow is based on a simple yet effective idea: decomposing the input condition into diverse latent representations, each capturing an aspect of the uncertainty and generating a distinct image. First, we integrate a latent graph, parameterized by Generative Flow Networks (GFlowNets), into the prompt representation computation. Second, leveraging GFlowNets' advanced graph sampling capabilities to capture uncertainty and output diverse trajectories over the graph, we produce multiple trajectories that collectively represent the input condition, leading to diverse condition representations and corresponding output images. Evaluations on natural image and medical image datasets demonstrate Rainbow's improvement in both diversity and fidelity across image synthesis, image generation, and counterfactual generation tasks. 

---
# Mitigating Coordinate Prediction Bias from Positional Encoding Failures 

**Authors**: Xingjian Tao, Yiwei Wang, Yujun Cai, Yihong Luo, Jing Tang  

**Link**: [PDF](https://arxiv.org/pdf/2510.22102)  

**Abstract**: Multimodal large language models (MLLMs) excel at vision-language tasks such as VQA and document understanding, yet precise coordinate prediction remains challenging. High-resolution inputs exacerbate this difficulty by producing long token sequences that weaken positional encodings and introduce directional biases in coordinate outputs. We investigate this phenomenon by analyzing how MLLMs behave when visual positional encodings (VPEs) are deliberately perturbed through shuffling. Our analysis reveals that such perturbations induce predictable, non-random coordinate biases rather than random errors, suggesting that models rely on internal positional priors when spatial grounding signals are degraded. Crucially, we observe similar directional error patterns in natural high-resolution datasets, indicating that positional encoding failures are a key bottleneck for accurate coordinate prediction at scale. To address this issue, we propose Vision-PE Shuffle Guidance (VPSG), a training-free test-time method that leverages the directional nature of these biases for correction. VPSG runs auxiliary decoding with shuffled VPEs to isolate position-unconditioned tendencies, then uses this as negative evidence to guide digit prediction while preserving coordinate format through a lightweight finite-state machine. Experiments on ScreenSpot-Pro demonstrate reliable improvements, highlighting positional encoding robustness as a critical factor for spatial reasoning in MLLMs. 

---
# QuArch: A Benchmark for Evaluating LLM Reasoning in Computer Architecture 

**Authors**: Shvetank Prakash, Andrew Cheng, Arya Tschand, Mark Mazumder, Varun Gohil, Jeffrey Ma, Jason Yik, Zishen Wan, Jessica Quaye, Elisavet Lydia Alvanaki, Avinash Kumar, Chandrashis Mazumdar, Tuhin Khare, Alexander Ingare, Ikechukwu Uchendu, Radhika Ghosal, Abhishek Tyagi, Chenyu Wang, Andrea Mattia Garavagno, Sarah Gu, Alice Guo, Grace Hur, Luca Carloni, Tushar Krishna, Ankita Nayak, Amir Yazdanbakhsh, Vijay Janapa Reddi  

**Link**: [PDF](https://arxiv.org/pdf/2510.22087)  

**Abstract**: The field of computer architecture, which bridges high-level software abstractions and low-level hardware implementations, remains absent from current large language model (LLM) evaluations. To this end, we present QuArch (pronounced 'quark'), the first benchmark designed to facilitate the development and evaluation of LLM knowledge and reasoning capabilities specifically in computer architecture. QuArch provides a comprehensive collection of 2,671 expert-validated question-answer (QA) pairs covering various aspects of computer architecture, including processor design, memory systems, and interconnection networks. Our evaluation reveals that while frontier models possess domain-specific knowledge, they struggle with skills that require higher-order thinking in computer architecture. Frontier model accuracies vary widely (from 34% to 72%) on these advanced questions, highlighting persistent gaps in architectural reasoning across analysis, design, and implementation QAs. By holistically assessing fundamental skills, QuArch provides a foundation for building and measuring LLM capabilities that can accelerate innovation in computing systems. With over 140 contributors from 40 institutions, this benchmark represents a community effort to set the standard for architectural reasoning in LLM evaluation. 

---
# Jailbreak Mimicry: Automated Discovery of Narrative-Based Jailbreaks for Large Language Models 

**Authors**: Pavlos Ntais  

**Link**: [PDF](https://arxiv.org/pdf/2510.22085)  

**Abstract**: Large language models (LLMs) remain vulnerable to sophisticated prompt engineering attacks that exploit contextual framing to bypass safety mechanisms, posing significant risks in cybersecurity applications. We introduce Jailbreak Mimicry, a systematic methodology for training compact attacker models to automatically generate narrative-based jailbreak prompts in a one-shot manner. Our approach transforms adversarial prompt discovery from manual craftsmanship into a reproducible scientific process, enabling proactive vulnerability assessment in AI-driven security systems. Developed for the OpenAI GPT-OSS-20B Red-Teaming Challenge, we use parameter-efficient fine-tuning (LoRA) on Mistral-7B with a curated dataset derived from AdvBench, achieving an 81.0% Attack Success Rate (ASR) against GPT-OSS-20B on a held-out test set of 200 items. Cross-model evaluation reveals significant variation in vulnerability patterns: our attacks achieve 66.5% ASR against GPT-4, 79.5% on Llama-3 and 33.0% against Gemini 2.5 Flash, demonstrating both broad applicability and model-specific defensive strengths in cybersecurity contexts. This represents a 54x improvement over direct prompting (1.5% ASR) and demonstrates systematic vulnerabilities in current safety alignment approaches. Our analysis reveals that technical domains (Cybersecurity: 93% ASR) and deception-based attacks (Fraud: 87.8% ASR) are particularly vulnerable, highlighting threats to AI-integrated threat detection, malware analysis, and secure systems, while physical harm categories show greater resistance (55.6% ASR). We employ automated harmfulness evaluation using Claude Sonnet 4, cross-validated with human expert assessment, ensuring reliable and scalable evaluation for cybersecurity red-teaming. Finally, we analyze failure mechanisms and discuss defensive strategies to mitigate these vulnerabilities in AI for cybersecurity. 

---
# Agentic Reinforcement Learning for Real-World Code Repair 

**Authors**: Siyu Zhu, Anastasiya Karpovich, Albert Chen, Jessica Koscheka, Shailesh Jannu, Di Wen, Yuqing Zhu, Rohit Jain, Alborz Geramifard  

**Link**: [PDF](https://arxiv.org/pdf/2510.22075)  

**Abstract**: We tackle the challenge of training reliable code-fixing agents in real repositories, where complex builds and shifting dependencies make evaluation unstable. We developed a verifiable pipeline with success defined as post-fix build validation and improved reproducibility across ~1K real issues by pinning dependencies and disabling automatic upgrades. Building on this, we introduced a scalable simplified pipeline for large-scale reinforcement learning (RL). Using this setup, we supervised fine-tuned Qwen3-32B in the full pipeline and applied RL on top of the SFT model in the simplified environment. The SFT model distilled from GPT-4.1 trajectories performs on par while being 56x smaller, and RL added 7-20% absolute gains under matched train-test conditions. "Thinking mode" was on par or worse in our experiments. Both SFT and RL models failed to generalize across environments, highlighting the importance of matching train-test environments for building reliable real-world code-fixing agents. 

---
# Frequentist Validity of Epistemic Uncertainty Estimators 

**Authors**: Anchit Jain, Stephen Bates  

**Link**: [PDF](https://arxiv.org/pdf/2510.22063)  

**Abstract**: Decomposing prediction uncertainty into its aleatoric (irreducible) and epistemic (reducible) components is critical for the development and deployment of machine learning systems. A popular, principled measure for epistemic uncertainty is the mutual information between the response variable and model parameters. However, evaluating this measure requires access to the posterior distribution of the model parameters, which is challenging to compute. In view of this, we introduce a frequentist measure of epistemic uncertainty based on the bootstrap. Our main theoretical contribution is a novel asymptotic expansion that reveals that our proposed (frequentist) measure and the (Bayesian) mutual information are asymptotically equivalent. This provides frequentist interpretations to mutual information and new computational strategies for approximating it. Moreover, we link our proposed approach to the widely-used heuristic approach of deep ensembles, giving added perspective on their practical success. 

---
# Automatic Assessment of Students' Classroom Engagement with Bias Mitigated Multi-task Model 

**Authors**: James Thiering, Tarun Sethupat Radha Krishna, Dylan Zelkin, Ashis Kumer Biswas  

**Link**: [PDF](https://arxiv.org/pdf/2510.22057)  

**Abstract**: With the rise of online and virtual learning, monitoring and enhancing student engagement have become an important aspect of effective education. Traditional methods of assessing a student's involvement might not be applicable directly to virtual environments. In this study, we focused on this problem and addressed the need to develop an automated system to detect student engagement levels during online learning. We proposed a novel training method which can discourage a model from leveraging sensitive features like gender for its predictions. The proposed method offers benefits not only in the enforcement of ethical standards, but also to enhance interpretability of the model predictions. We applied an attribute-orthogonal regularization technique to a split-model classifier, which uses multiple transfer learning strategies to achieve effective results in reducing disparity in the distribution of prediction for sensitivity groups from a Pearson correlation coefficient of 0.897 for the unmitigated model, to 0.999 for the mitigated model. The source code for this project is available on this https URL . 

---
# Human-Centric Anomaly Detection in Surveillance Videos Using YOLO-World and Spatio-Temporal Deep Learning 

**Authors**: Mohammad Ali Etemadi Naeen, Hoda Mohammadzade, Saeed Bagheri Shouraki  

**Link**: [PDF](https://arxiv.org/pdf/2510.22056)  

**Abstract**: Anomaly detection in surveillance videos remains a challenging task due to the diversity of abnormal events, class imbalance, and scene-dependent visual clutter. To address these issues, we propose a robust deep learning framework that integrates human-centric preprocessing with spatio-temporal modeling for multi-class anomaly classification. Our pipeline begins by applying YOLO-World - an open-vocabulary vision-language detector - to identify human instances in raw video clips, followed by ByteTrack for consistent identity-aware tracking. Background regions outside detected bounding boxes are suppressed via Gaussian blurring, effectively reducing scene-specific distractions and focusing the model on behaviorally relevant foreground content. The refined frames are then processed by an ImageNet-pretrained InceptionV3 network for spatial feature extraction, and temporal dynamics are captured using a bidirectional LSTM (BiLSTM) for sequence-level classification. Evaluated on a five-class subset of the UCF-Crime dataset (Normal, Burglary, Fighting, Arson, Explosion), our method achieves a mean test accuracy of 92.41% across three independent trials, with per-class F1-scores consistently exceeding 0.85. Comprehensive evaluation metrics - including confusion matrices, ROC curves, and macro/weighted averages - demonstrate strong generalization and resilience to class imbalance. The results confirm that foreground-focused preprocessing significantly enhances anomaly discrimination in real-world surveillance scenarios. 

---
# VLM-SlideEval: Evaluating VLMs on Structured Comprehension and Perturbation Sensitivity in PPT 

**Authors**: Hyeonsu Kang, Emily Bao, Anjan Goswami  

**Link**: [PDF](https://arxiv.org/pdf/2510.22045)  

**Abstract**: Vision-language models (VLMs) are increasingly used to evaluate multimodal content, including presentation slides, yet their slide-specific understanding remains underexplored {despite their growing role as critics in agentic, model-forward pipelines}. We introduce VLM-SlideEval, an evaluation framework that probes VLMs along three axes: (1) element-level extraction from slide images aligned to ground truth; (2) robustness to controlled perturbations in geometry, style, and text; and (3) higher-level comprehension, such as recovering a deck's narrative order from shuffled slides. Using publicly available decks from Zenodo (this https URL), we standardize ground-truth element metadata from PowerPoint XML and live renderings into a unified, verifiable schema. Empirically, VLMs underperform on pixel-accurate extraction and show non-trivial agreement, fidelity, and consistency under controlled perturbations, while performing better on single-slide content understanding; however, they do not reliably capture narrative structure across slides. These results highlight the limits of current VLMs for slide evaluation and motivate calibrated, critic-in-the-loop evaluators that drive iterative refinement and selection in agentic pipelines. 

---
# Emotions Where Art Thou: Understanding and Characterizing the Emotional Latent Space of Large Language Models 

**Authors**: Benjamin Reichman, Adar Avsian, Larry Heck  

**Link**: [PDF](https://arxiv.org/pdf/2510.22042)  

**Abstract**: This work investigates how large language models (LLMs) internally represent emotion by analyzing the geometry of their hidden-state space. The paper identifies a low-dimensional emotional manifold and shows that emotional representations are directionally encoded, distributed across layers, and aligned with interpretable dimensions. These structures are stable across depth and generalize to eight real-world emotion datasets spanning five languages. Cross-domain alignment yields low error and strong linear probe performance, indicating a universal emotional subspace. Within this space, internal emotion perception can be steered while preserving semantics using a learned intervention module, with especially strong control for basic emotions across languages. These findings reveal a consistent and manipulable affective geometry in LLMs and offer insight into how they internalize and process emotion. 

---
# Differentiable Constraint-Based Causal Discovery 

**Authors**: Jincheng Zhou, Mengbo Wang, Anqi He, Yumeng Zhou, Hessam Olya, Murat Kocaoglu, Bruno Ribeiro  

**Link**: [PDF](https://arxiv.org/pdf/2510.22031)  

**Abstract**: Causal discovery from observational data is a fundamental task in artificial intelligence, with far-reaching implications for decision-making, predictions, and interventions. Despite significant advances, existing methods can be broadly categorized as constraint-based or score-based approaches. Constraint-based methods offer rigorous causal discovery but are often hindered by small sample sizes, while score-based methods provide flexible optimization but typically forgo explicit conditional independence testing. This work explores a third avenue: developing differentiable $d$-separation scores, obtained through a percolation theory using soft logic. This enables the implementation of a new type of causal discovery method: gradient-based optimization of conditional independence constraints. Empirical evaluations demonstrate the robust performance of our approach in low-sample regimes, surpassing traditional constraint-based and score-based baselines on a real-world dataset. Code and data of the proposed method are publicly available at https://github$.$com/PurdueMINDS/DAGPA. 

---
# Online Optimization for Offline Safe Reinforcement Learning 

**Authors**: Yassine Chemingui, Aryan Deshwal, Alan Fern, Thanh Nguyen-Tang, Janardhan Rao Doppa  

**Link**: [PDF](https://arxiv.org/pdf/2510.22027)  

**Abstract**: We study the problem of Offline Safe Reinforcement Learning (OSRL), where the goal is to learn a reward-maximizing policy from fixed data under a cumulative cost constraint. We propose a novel OSRL approach that frames the problem as a minimax objective and solves it by combining offline RL with online optimization algorithms. We prove the approximate optimality of this approach when integrated with an approximate offline RL oracle and no-regret online optimization. We also present a practical approximation that can be combined with any offline RL algorithm, eliminating the need for offline policy evaluation. Empirical results on the DSRL benchmark demonstrate that our method reliably enforces safety constraints under stringent cost budgets, while achieving high rewards. The code is available at this https URL. 

---
# Normalization in Attention Dynamics 

**Authors**: Nikita Karagodin, Shu Ge, Yury Polyanskiy, Philippe Rigollet  

**Link**: [PDF](https://arxiv.org/pdf/2510.22026)  

**Abstract**: We study the effect of normalization schemes on token representations in deep transformers. Modeling their evolution as interacting particles on the sphere, we show that normalization acts as a form of speed regulation. This perspective enables a unified analysis of several schemes -- including Post-LN, Pre-LN, Mix-LN, Peri-LN, nGPT, and LN-Scaling -- revealing how they influence clustering dynamics and representation collapse. Our framework clarifies how different schemes shape token representations across layers and provides a principled basis for comparing them, identifying Peri-LN as a particularly effective choice. 

---
# Toward Understanding the Transferability of Adversarial Suffixes in Large Language Models 

**Authors**: Sarah Ball, Niki Hasrati, Alexander Robey, Avi Schwarzschild, Frauke Kreuter, Zico Kolter, Andrej Risteski  

**Link**: [PDF](https://arxiv.org/pdf/2510.22014)  

**Abstract**: Discrete optimization-based jailbreaking attacks on large language models aim to generate short, nonsensical suffixes that, when appended onto input prompts, elicit disallowed content. Notably, these suffixes are often transferable -- succeeding on prompts and models for which they were never optimized. And yet, despite the fact that transferability is surprising and empirically well-established, the field lacks a rigorous analysis of when and why transfer occurs. To fill this gap, we identify three statistical properties that strongly correlate with transfer success across numerous experimental settings: (1) how much a prompt without a suffix activates a model's internal refusal direction, (2) how strongly a suffix induces a push away from this direction, and (3) how large these shifts are in directions orthogonal to refusal. On the other hand, we find that prompt semantic similarity only weakly correlates with transfer success. These findings lead to a more fine-grained understanding of transferability, which we use in interventional experiments to showcase how our statistical analysis can translate into practical improvements in attack success. 

---
# Reconnaissance Automatique des Langues des Signes : Une Approche Hybridée CNN-LSTM Basée sur Mediapipe 

**Authors**: Fraisse Sacré Takouchouang, Ho Tuong Vinh  

**Link**: [PDF](https://arxiv.org/pdf/2510.22011)  

**Abstract**: Sign languages play a crucial role in the communication of deaf communities, but they are often marginalized, limiting access to essential services such as healthcare and education. This study proposes an automatic sign language recognition system based on a hybrid CNN-LSTM architecture, using Mediapipe for gesture keypoint extraction. Developed with Python, TensorFlow and Streamlit, the system provides real-time gesture translation. The results show an average accuracy of 92\%, with very good performance for distinct gestures such as ``Hello'' and ``Thank you''. However, some confusions remain for visually similar gestures, such as ``Call'' and ``Yes''. This work opens up interesting perspectives for applications in various fields such as healthcare, education and public services. 

---
# Impact and Implications of Generative AI for Enterprise Architects in Agile Environments: A Systematic Literature Review 

**Authors**: Stefan Julian Kooy, Jean Paul Sebastian Piest, Rob Henk Bemthuis  

**Link**: [PDF](https://arxiv.org/pdf/2510.22003)  

**Abstract**: Generative AI (GenAI) is reshaping enterprise architecture work in agile software organizations, yet evidence on its effects remains scattered. We report a systematic literature review (SLR), following established SLR protocols of Kitchenham and PRISMA, of 1,697 records, yielding 33 studies across enterprise, solution, domain, business, and IT architect roles. GenAI most consistently supports (i) design ideation and trade-off exploration; (ii) rapid creation and refinement of artifacts (e.g., code, models, documentation); and (iii) architectural decision support and knowledge retrieval. Reported risks include opacity and bias, contextually incorrect outputs leading to rework, privacy and compliance concerns, and social loafing. We also identify emerging skills and competencies, including prompt engineering, model evaluation, and professional oversight, and organizational enablers around readiness and adaptive governance. The review contributes with (1) a mapping of GenAI use cases and risks in agile architecting, (2) implications for capability building and governance, and (3) an initial research agenda on human-AI collaboration in architecture. Overall, the findings inform responsible adoption of GenAI that accelerates digital transformation while safeguarding architectural integrity. 

---
# From Black-box to Causal-box: Towards Building More Interpretable Models 

**Authors**: Inwoo Hwang, Yushu Pan, Elias Bareinboim  

**Link**: [PDF](https://arxiv.org/pdf/2510.21998)  

**Abstract**: Understanding the predictions made by deep learning models remains a central challenge, especially in high-stakes applications. A promising approach is to equip models with the ability to answer counterfactual questions -- hypothetical ``what if?'' scenarios that go beyond the observed data and provide insight into a model reasoning. In this work, we introduce the notion of causal interpretability, which formalizes when counterfactual queries can be evaluated from a specific class of models and observational data. We analyze two common model classes -- blackbox and concept-based predictors -- and show that neither is causally interpretable in general. To address this gap, we develop a framework for building models that are causally interpretable by design. Specifically, we derive a complete graphical criterion that determines whether a given model architecture supports a given counterfactual query. This leads to a fundamental tradeoff between causal interpretability and predictive accuracy, which we characterize by identifying the unique maximal set of features that yields an interpretable model with maximal predictive expressiveness. Experiments corroborate the theoretical findings. 

---
# Is Temporal Difference Learning the Gold Standard for Stitching in RL? 

**Authors**: Michał Bortkiewicz, Władysław Pałucki, Mateusz Ostaszewski, Benjamin Eysenbach  

**Link**: [PDF](https://arxiv.org/pdf/2510.21995)  

**Abstract**: Reinforcement learning (RL) promises to solve long-horizon tasks even when training data contains only short fragments of the behaviors. This experience stitching capability is often viewed as the purview of temporal difference (TD) methods. However, outside of small tabular settings, trajectories never intersect, calling into question this conventional wisdom. Moreover, the common belief is that Monte Carlo (MC) methods should not be able to recombine experience, yet it remains unclear whether function approximation could result in a form of implicit stitching. The goal of this paper is to empirically study whether the conventional wisdom about stitching actually holds in settings where function approximation is used. We empirically demonstrate that Monte Carlo (MC) methods can also achieve experience stitching. While TD methods do achieve slightly stronger capabilities than MC methods (in line with conventional wisdom), that gap is significantly smaller than the gap between small and large neural networks (even on quite simple tasks). We find that increasing critic capacity effectively reduces the generalization gap for both the MC and TD methods. These results suggest that the traditional TD inductive bias for stitching may be less necessary in the era of large models for RL and, in some cases, may offer diminishing returns. Additionally, our results suggest that stitching, a form of generalization unique to the RL setting, might be achieved not through specialized algorithms (temporal difference learning) but rather through the same recipe that has provided generalization in other machine learning settings (via scale). Project website: this https URL 

---
# Two-Steps Diffusion Policy for Robotic Manipulation via Genetic Denoising 

**Authors**: Mateo Clemente, Leo Brunswic, Rui Heng Yang, Xuan Zhao, Yasser Khalil, Haoyu Lei, Amir Rasouli, Yinchuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.21991)  

**Abstract**: Diffusion models, such as diffusion policy, have achieved state-of-the-art results in robotic manipulation by imitating expert demonstrations. While diffusion models were originally developed for vision tasks like image and video generation, many of their inference strategies have been directly transferred to control domains without adaptation. In this work, we show that by tailoring the denoising process to the specific characteristics of embodied AI tasks -- particularly structured, low-dimensional nature of action distributions -- diffusion policies can operate effectively with as few as 5 neural function evaluations (NFE).
Building on this insight, we propose a population-based sampling strategy, genetic denoising, which enhances both performance and stability by selecting denoising trajectories with low out-of-distribution risk. Our method solves challenging tasks with only 2 NFE while improving or matching performance. We evaluate our approach across 14 robotic manipulation tasks from D4RL and Robomimic, spanning multiple action horizons and inference budgets. In over 2 million evaluations, our method consistently outperforms standard diffusion-based policies, achieving up to 20\% performance gains with significantly fewer inference steps. 

---
# Uncovering the Persuasive Fingerprint of LLMs in Jailbreaking Attacks 

**Authors**: Havva Alizadeh Noughabi, Julien Serbanescu, Fattane Zarrinkalam, Ali Dehghantanha  

**Link**: [PDF](https://arxiv.org/pdf/2510.21983)  

**Abstract**: Despite recent advances, Large Language Models remain vulnerable to jailbreak attacks that bypass alignment safeguards and elicit harmful outputs. While prior research has proposed various attack strategies differing in human readability and transferability, little attention has been paid to the linguistic and psychological mechanisms that may influence a model's susceptibility to such attacks. In this paper, we examine an interdisciplinary line of research that leverages foundational theories of persuasion from the social sciences to craft adversarial prompts capable of circumventing alignment constraints in LLMs. Drawing on well-established persuasive strategies, we hypothesize that LLMs, having been trained on large-scale human-generated text, may respond more compliantly to prompts with persuasive structures. Furthermore, we investigate whether LLMs themselves exhibit distinct persuasive fingerprints that emerge in their jailbreak responses. Empirical evaluations across multiple aligned LLMs reveal that persuasion-aware prompts significantly bypass safeguards, demonstrating their potential to induce jailbreak behaviors. This work underscores the importance of cross-disciplinary insight in addressing the evolving challenges of LLM safety. The code and data are available. 

---
# Beyond Reasoning Gains: Mitigating General Capabilities Forgetting in Large Reasoning Models 

**Authors**: Hoang Phan, Xianjun Yang, Kevin Yao, Jingyu Zhang, Shengjie Bi, Xiaocheng Tang, Madian Khabsa, Lijuan Liu, Deren Lei  

**Link**: [PDF](https://arxiv.org/pdf/2510.21978)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has delivered impressive gains in mathematical and multimodal reasoning and has become a standard post-training paradigm for contemporary language and vision-language models. However, the RLVR recipe introduces a significant risk of capability regression, where models forget foundational skills after prolonged training without employing regularization strategies. We empirically confirm this concern, observing that open-source reasoning models suffer performance degradation on core capabilities such as perception and faithfulness. While imposing regularization terms like KL divergence can help prevent deviation from the base model, these terms are calculated on the current task, thus they do not guarantee broader knowledge. Meanwhile, commonly used experience replay across heterogeneous domains makes it nontrivial to decide how much training focus each objective should receive. To address this, we propose RECAP-a replay strategy with dynamic objective reweighting for general knowledge preservation. Our reweighting mechanism adapts in an online manner using short-horizon signals of convergence and instability, shifting the post-training focus away from saturated objectives and toward underperforming or volatile ones. Our method is end-to-end and readily applicable to existing RLVR pipelines without training additional models or heavy tuning. Extensive experiments on benchmarks based on Qwen2.5-VL-3B and Qwen2.5-VL-7B demonstrate the effectiveness of our method, which not only preserves general capabilities but also improves reasoning by enabling more flexible trade-offs among in-task rewards. 

---
# ArchISMiner: A Framework for Automatic Mining of Architectural Issue-Solution Pairs from Online Developer Communities 

**Authors**: Musengamana Jean de Dieu, Ruiyin Li, Peng Liang, Mojtaba Shahin, Muhammad Waseem, Arif Ali Khan, Bangchao Wang, Mst Shamima Aktar  

**Link**: [PDF](https://arxiv.org/pdf/2510.21966)  

**Abstract**: Stack Overflow (SO), a leading online community forum, is a rich source of software development knowledge. However, locating architectural knowledge, such as architectural solutions remains challenging due to the overwhelming volume of unstructured content and fragmented discussions. Developers must manually sift through posts to find relevant architectural insights, which is time-consuming and error-prone. This study introduces ArchISMiner, a framework for mining architectural knowledge from SO. The framework comprises two complementary components: ArchPI and ArchISPE. ArchPI trains and evaluates multiple models, including conventional ML/DL models, Pre-trained Language Models (PLMs), and Large Language Models (LLMs), and selects the best-performing model to automatically identify Architecture-Related Posts (ARPs) among programming-related discussions. ArchISPE employs an indirect supervised approach that leverages diverse features, including BERT embeddings and local TextCNN features, to extract architectural issue-solution pairs. Our evaluation shows that the best model in ArchPI achieves an F1-score of 0.960 in ARP detection, and ArchISPE outperforms baselines in both SE and NLP fields, achieving F1-scores of 0.883 for architectural issues and 0.894 for solutions. A user study further validated the quality (e.g., relevance and usefulness) of the identified ARPs and the extracted issue-solution pairs. Moreover, we applied ArchISMiner to three additional forums, releasing a dataset of over 18K architectural issue-solution pairs. Overall, ArchISMiner can help architects and developers identify ARPs and extract succinct, relevant, and useful architectural knowledge from developer communities more accurately and efficiently. The replication package of this study has been provided at this https URL 

---
# Towards Low-Latency and Adaptive Ransomware Detection Using Contrastive Learning 

**Authors**: Zhixin Pan, Ziyu Shu, Amberbir Alemayoh  

**Link**: [PDF](https://arxiv.org/pdf/2510.21957)  

**Abstract**: Ransomware has become a critical threat to cybersecurity due to its rapid evolution, the necessity for early detection, and growing diversity, posing significant challenges to traditional detection methods. While AI-based approaches had been proposed by prior works to assist ransomware detection, existing methods suffer from three major limitations, ad-hoc feature dependencies, delayed response, and limited adaptability to unseen variants. In this paper, we propose a framework that integrates self-supervised contrastive learning with neural architecture search (NAS) to address these challenges. Specifically, this paper offers three important contributions. (1) We design a contrastive learning framework that incorporates hardware performance counters (HPC) to analyze the runtime behavior of target ransomware. (2) We introduce a customized loss function that encourages early-stage detection of malicious activity, and significantly reduces the detection latency. (3) We deploy a neural architecture search (NAS) framework to automatically construct adaptive model architectures, allowing the detector to flexibly align with unseen ransomware variants. Experimental results show that our proposed method achieves significant improvements in both detection accuracy (up to 16.1%) and response time (up to 6x) compared to existing approaches while maintaining robustness under evasive attacks. 

---
# AutoSciDACT: Automated Scientific Discovery through Contrastive Embedding and Hypothesis Testing 

**Authors**: Samuel Bright-Thonney, Christina Reissel, Gaia Grosso, Nathaniel Woodward, Katya Govorkova, Andrzej Novak, Sang Eon Park, Eric Moreno, Philip Harris  

**Link**: [PDF](https://arxiv.org/pdf/2510.21935)  

**Abstract**: Novelty detection in large scientific datasets faces two key challenges: the noisy and high-dimensional nature of experimental data, and the necessity of making statistically robust statements about any observed outliers. While there is a wealth of literature on anomaly detection via dimensionality reduction, most methods do not produce outputs compatible with quantifiable claims of scientific discovery. In this work we directly address these challenges, presenting the first step towards a unified pipeline for novelty detection adapted for the rigorous statistical demands of science. We introduce AutoSciDACT (Automated Scientific Discovery with Anomalous Contrastive Testing), a general-purpose pipeline for detecting novelty in scientific data. AutoSciDACT begins by creating expressive low-dimensional data representations using a contrastive pre-training, leveraging the abundance of high-quality simulated data in many scientific domains alongside expertise that can guide principled data augmentation strategies. These compact embeddings then enable an extremely sensitive machine learning-based two-sample test using the New Physics Learning Machine (NPLM) framework, which identifies and statistically quantifies deviations in observed data relative to a reference distribution (null hypothesis). We perform experiments across a range of astronomical, physical, biological, image, and synthetic datasets, demonstrating strong sensitivity to small injections of anomalous data across all domains. 

---
# A Comparison of Conversational Models and Humans in Answering Technical Questions: the Firefox Case 

**Authors**: Joao Correia, Daniel Coutinho, Marco Castelluccio, Caio Barbosa, Rafael de Mello, Anita Sarma, Alessandro Garcia, Marco Gerosa, Igor Steinmacher  

**Link**: [PDF](https://arxiv.org/pdf/2510.21933)  

**Abstract**: The use of Large Language Models (LLMs) to support tasks in software development has steadily increased over recent years. From assisting developers in coding activities to providing conversational agents that answer newcomers' questions. In collaboration with the Mozilla Foundation, this study evaluates the effectiveness of Retrieval-Augmented Generation (RAG) in assisting developers within the Mozilla Firefox project. We conducted an empirical analysis comparing responses from human developers, a standard GPT model, and a GPT model enhanced with RAG, using real queries from Mozilla's developer chat rooms. To ensure a rigorous evaluation, Mozilla experts assessed the responses based on helpfulness, comprehensiveness, and conciseness. The results show that RAG-assisted responses were more comprehensive than human developers (62.50% to 54.17%) and almost as helpful (75.00% to 79.17%), suggesting RAG's potential to enhance developer assistance. However, the RAG responses were not as concise and often verbose. The results show the potential to apply RAG-based tools to Open Source Software (OSS) to minimize the load to core maintainers without losing answer quality. Toning down retrieval mechanisms and making responses even shorter in the future would enhance developer assistance in massive projects like Mozilla Firefox. 

---
# Enabling Robust In-Context Memory and Rapid Task Adaptation in Transformers with Hebbian and Gradient-Based Plasticity 

**Authors**: Siddharth Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2510.21908)  

**Abstract**: Large language models display in-context learning as an emergent effect of scale, but they rely on static weights during inference. In contrast, biological systems continually adapt via synaptic plasticity. We investigate whether explicit, biologically inspired plasticity can endow Transformers with faster in-sequence adaptation. To this end, we augment decoder-only Transformers with fast-weight modules updated either by (i) a neuromodulated Hebbian rule or (ii) the gradient-based plasticity mechanism of Duan et al. (2023). Across copying, regression, and few-shot classification tasks (CIFAR-FS, Omniglot), Hebbian plasticity consistently achieves lower loss and stronger few-shot generalization, while gradient-based updates perform best on long-horizon credit assignment. When associations are short and linearly separable, static weights suffice, defining a clear boundary condition for when plasticity helps. Analysis of learned modulatory signals reveals that gradient-based rules maintain large, persistent updates, whereas Hebbian plasticity is sharply gated around salient events. Together, these results show that explicit plasticity complements attention by enabling rapid, task-specific adaptation, and clarify when different plasticity mechanisms are most effective. 

---
# Structure-Aware Cooperative Ensemble Evolutionary Optimization on Combinatorial Problems with Multimodal Large Language Models 

**Authors**: Jie Zhao, Kang Hao Cheong  

**Link**: [PDF](https://arxiv.org/pdf/2510.21906)  

**Abstract**: Evolutionary algorithms (EAs) have proven effective in exploring the vast solution spaces typical of graph-structured combinatorial problems. However, traditional encoding schemes, such as binary or numerical representations, often fail to straightforwardly capture the intricate structural properties of networks. Through employing the image-based encoding to preserve topological context, this study utilizes multimodal large language models (MLLMs) as evolutionary operators to facilitate structure-aware optimization over graph data. To address the visual clutter inherent in large-scale network visualizations, we leverage graph sparsification techniques to simplify structures while maintaining essential structural features. To further improve robustness and mitigate bias from different sparsification views, we propose a cooperative evolutionary optimization framework that facilitates cross-domain knowledge transfer and unifies multiple sparsified variants of diverse structures. Additionally, recognizing the sensitivity of MLLMs to network layout, we introduce an ensemble strategy that aggregates outputs from various layout configurations through consensus voting. Finally, experiments on real-world networks through various tasks demonstrate that our approach improves both the quality and reliability of solutions in MLLM-driven evolutionary optimization. 

---
# TOM-SWE: User Mental Modeling For Software Engineering Agents 

**Authors**: Xuhui Zhou, Valerie Chen, Zora Zhiruo Wang, Graham Neubig, Maarten Sap, Xingyao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.21903)  

**Abstract**: Recent advances in coding agents have made them capable of planning, editing, running, and testing complex code bases. Despite their growing ability in coding tasks, these systems still struggle to infer and track user intent, especially when instructions are underspecified or context-dependent. To bridge this gap, we introduce ToM-SWE, a dual-agent architecture that pairs a primary software-engineering (SWE) agent with a lightweight theory-of-mind (ToM) partner agent dedicated to modeling the user's mental state. The ToM agent infers user goals, constraints, and preferences from instructions and interaction history, maintains a \textbf{persistent memory} of the user, and provides user-related suggestions to the SWE agent. In two software engineering benchmarks (ambiguous SWE-bench and stateful SWE-bench), ToM-SWE improves task success rates and user satisfaction. Notably, on the stateful SWE benchmark, a newly introduced evaluation that provides agents with a user simulator along with previous interaction histories, ToM-SWE achieves a substantially higher task success rate of 59.7\% compared to 18.1\% for OpenHands, a state-of-the-art SWE agent. Furthermore, in a three-week study with professional developers using ToM-SWE in their daily work, participants found it useful 86\% of the time, underscoring the value of stateful user modeling for practical coding agents. 

---
# Software Engineering Agents for Embodied Controller Generation : A Study in Minigrid Environments 

**Authors**: Timothé Boulet, Xavier Hinaut, Clément Moulin-Frier  

**Link**: [PDF](https://arxiv.org/pdf/2510.21902)  

**Abstract**: Software Engineering Agents (SWE-Agents) have proven effective for traditional software engineering tasks with accessible codebases, but their performance for embodied tasks requiring well-designed information discovery remains unexplored. We present the first extended evaluation of SWE-Agents on controller generation for embodied tasks, adapting Mini-SWE-Agent (MSWEA) to solve 20 diverse embodied tasks from the Minigrid environment. Our experiments compare agent performance across different information access conditions: with and without environment source code access, and with varying capabilities for interactive exploration. We quantify how different information access levels affect SWE-Agent performance for embodied tasks and analyze the relative importance of static code analysis versus dynamic exploration for task solving. This work establishes controller generation for embodied tasks as a crucial evaluation domain for SWE-Agents and provides baseline results for future research in efficient reasoning systems. 

---
# Deep Literature Survey Automation with an Iterative Workflow 

**Authors**: Hongbo Zhang, Han Cui, Yidong Wang, Yijian Tian, Qi Guo, Cunxiang Wang, Jian Wu, Chiyu Song, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.21900)  

**Abstract**: Automatic literature survey generation has attracted increasing attention, yet most existing systems follow a one-shot paradigm, where a large set of papers is retrieved at once and a static outline is generated before drafting. This design often leads to noisy retrieval, fragmented structures, and context overload, ultimately limiting survey quality. Inspired by the iterative reading process of human researchers, we propose \ours, a framework based on recurrent outline generation, in which a planning agent incrementally retrieves, reads, and updates the outline to ensure both exploration and coherence. To provide faithful paper-level grounding, we design paper cards that distill each paper into its contributions, methods, and findings, and introduce a review-and-refine loop with visualization enhancement to improve textual flow and integrate multimodal elements such as figures and tables. Experiments on both established and emerging topics show that \ours\ substantially outperforms state-of-the-art baselines in content coverage, structural coherence, and citation quality, while producing more accessible and better-organized surveys. To provide a more reliable assessment of such improvements, we further introduce Survey-Arena, a pairwise benchmark that complements absolute scoring and more clearly positions machine-generated surveys relative to human-written ones. The code is available at this https URL\_Autosurveyv2. 

---
# Understanding Network Behaviors through Natural Language Question-Answering 

**Authors**: Mingzhe Xing, Chang Tian, Jianan Zhang, Lichen Pan, Peipei Liu, Zhaoteng Yan, Yinliang Yue  

**Link**: [PDF](https://arxiv.org/pdf/2510.21894)  

**Abstract**: Modern large-scale networks introduce significant complexity in understanding network behaviors, increasing the risk of misconfiguration. Prior work proposed to understand network behaviors by mining network configurations, typically relying on domain-specific languages interfaced with formal models. While effective, they suffer from a steep learning curve and limited flexibility. In contrast, natural language (NL) offers a more accessible and interpretable interface, motivating recent research on NL-guided network behavior understanding. Recent advances in large language models (LLMs) further enhance this direction, leveraging their extensive prior knowledge of network concepts and strong reasoning capabilities. However, three key challenges remain: 1) numerous router devices with lengthy configuration files challenge LLM's long-context understanding ability; 2) heterogeneity across devices and protocols impedes scalability; and 3) complex network topologies and protocols demand advanced reasoning abilities beyond the current capabilities of LLMs. To tackle the above challenges, we propose NetMind, a novel framework for querying networks using NL. Our approach introduces a tree-based configuration chunking strategy to preserve semantic coherence while enabling efficient partitioning. We then construct a unified fact graph as an intermediate representation to normalize vendor-specific configurations. Finally, we design a hybrid imperative-declarative language to reduce the reasoning burden on LLMs and enhance precision. We contribute a benchmark consisting of NL question-answer pairs paired with network configurations. Experiments demonstrate that NetMind achieves accurate and scalable network behavior understanding, outperforming existing baselines. 

---
# Embedding Trust: Semantic Isotropy Predicts Nonfactuality in Long-Form Text Generation 

**Authors**: Dhrupad Bhardwaj, Julia Kempe, Tim G. J. Rudner  

**Link**: [PDF](https://arxiv.org/pdf/2510.21891)  

**Abstract**: To deploy large language models (LLMs) in high-stakes application domains that require substantively accurate responses to open-ended prompts, we need reliable, computationally inexpensive methods that assess the trustworthiness of long-form responses generated by LLMs. However, existing approaches often rely on claim-by-claim fact-checking, which is computationally expensive and brittle in long-form responses to open-ended prompts. In this work, we introduce semantic isotropy -- the degree of uniformity across normalized text embeddings on the unit sphere -- and use it to assess the trustworthiness of long-form responses generated by LLMs. To do so, we generate several long-form responses, embed them, and estimate the level of semantic isotropy of these responses as the angular dispersion of the embeddings on the unit sphere. We find that higher semantic isotropy -- that is, greater embedding dispersion -- reliably signals lower factual consistency across samples. Our approach requires no labeled data, no fine-tuning, and no hyperparameter selection, and can be used with open- or closed-weight embedding models. Across multiple domains, our method consistently outperforms existing approaches in predicting nonfactuality in long-form responses using only a handful of samples -- offering a practical, low-cost approach for integrating trust assessment into real-world LLM workflows. 

---
# The Principles of Diffusion Models 

**Authors**: Chieh-Hsin Lai, Yang Song, Dongjun Kim, Yuki Mitsufuji, Stefano Ermon  

**Link**: [PDF](https://arxiv.org/pdf/2510.21890)  

**Abstract**: This monograph presents the core principles that have guided the development of diffusion models, tracing their origins and showing how diverse formulations arise from shared mathematical ideas. Diffusion modeling starts by defining a forward process that gradually corrupts data into noise, linking the data distribution to a simple prior through a continuum of intermediate distributions. The goal is to learn a reverse process that transforms noise back into data while recovering the same intermediates. We describe three complementary views. The variational view, inspired by variational autoencoders, sees diffusion as learning to remove noise step by step. The score-based view, rooted in energy-based modeling, learns the gradient of the evolving data distribution, indicating how to nudge samples toward more likely regions. The flow-based view, related to normalizing flows, treats generation as following a smooth path that moves samples from noise to data under a learned velocity field. These perspectives share a common backbone: a time-dependent velocity field whose flow transports a simple prior to the data. Sampling then amounts to solving a differential equation that evolves noise into data along a continuous trajectory. On this foundation, the monograph discusses guidance for controllable generation, efficient numerical solvers, and diffusion-motivated flow-map models that learn direct mappings between arbitrary times. It provides a conceptual and mathematically grounded understanding of diffusion models for readers with basic deep-learning knowledge. 

---
# Generative AI in Depth: A Survey of Recent Advances, Model Variants, and Real-World Applications 

**Authors**: Shamim Yazdani, Akansha Singh, Nripsuta Saxena, Zichong Wang, Avash Palikhe, Deng Pan, Umapada Pal, Jie Yang, Wenbin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.21887)  

**Abstract**: In recent years, deep learning based generative models, particularly Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs), and Diffusion Models (DMs), have been instrumental in in generating diverse, high-quality content across various domains, such as image and video synthesis. This capability has led to widespread adoption of these models and has captured strong public interest. As they continue to advance at a rapid pace, the growing volume of research, expanding application areas, and unresolved technical challenges make it increasingly difficult to stay current. To address this need, this survey introduces a comprehensive taxonomy that organizes the literature and provides a cohesive framework for understanding the development of GANs, VAEs, and DMs, including their many variants and combined approaches. We highlight key innovations that have improved the quality, diversity, and controllability of generated outputs, reflecting the expanding potential of generative artificial intelligence. In addition to summarizing technical progress, we examine rising ethical concerns, including the risks of misuse and the broader societal impact of synthetic media. Finally, we outline persistent challenges and propose future research directions, offering a structured and forward looking perspective for researchers in this fast evolving field. 

---
# Preventing Catastrophic Forgetting: Behavior-Aware Sampling for Safer Language Model Fine-Tuning 

**Authors**: Anh Pham, Mihir Thalanki, Michael Sun, Aditya Chaloo, Ankita Gupta, Tian Xia, Aditya Mate, Ehimwenma Nosakhare, Soundararajan Srinivasan  

**Link**: [PDF](https://arxiv.org/pdf/2510.21885)  

**Abstract**: Large language models often lose previously aligned safety behaviors when fine-tuned on benign data, a phenomenon known as catastrophic forgetting. Prior work shows that adding random safety examples can mitigate this effect, but it remains unclear which examples are most effective. We propose a behavior-aware sampling framework that selects safety examples based on two complementary factors: instruction-response behavior (e.g., refusal versus compliance) and semantic diversity across harm categories. Systematic evaluation shows that this approach substantially reduces harmful outputs while maintaining helpfulness, achieving up to a 41% reduction in harmfulness with only 0.5% additional training data. These results highlight how targeted data selection can improve the safety and efficiency of fine-tuning at scale. 

---
# Framework for Machine Evaluation of Reasoning Completeness in Large Language Models For Classification Tasks 

**Authors**: Avinash Patil  

**Link**: [PDF](https://arxiv.org/pdf/2510.21884)  

**Abstract**: The growing adoption of machine learning (ML) in sensitive domains has heightened the demand for transparent and interpretable artificial intelligence. Large Language Models (LLMs) are increasingly capable of producing natural language explanations, yet it remains unclear whether these rationales faithfully capture the predictive signals that underlie decisions. This paper introduces RACE-Reasoning Alignment for Completeness of Explanations, a systematic framework to evaluate the alignment between LLM-generated explanations and interpretable feature importance scores derived from a logistic regression baseline. We analyze four widely used text classification datasets-WIKI ONTOLOGY, AG NEWS, IMDB, and GOEMOTIONS-and compare LLM rationales against top-ranked supporting and contradicting lexical features. To capture alignment at multiple levels of granularity, RACE implements token-aware, exact string, and edit-distance matching techniques. Empirical results reveal a consistent asymmetry: correct predictions exhibit higher coverage of supporting features, while incorrect predictions are associated with elevated coverage of contradicting features. Edit-distance matching further uncovers paraphrastic overlaps, boosting coverage while preserving this asymmetry. These findings demonstrate that LLM rationales combine both surface-level and flexible evidence reuse, yet can also amplify misleading cues in error cases. RACE provides new insights into the faithfulness of LLM explanations and establishes a quantitative basis for evaluating reasoning completeness in neural language models. 

---
# Language Ranker: A Lightweight Ranking framework for LLM Decoding 

**Authors**: Chenheng Zhang, Tianqi Du, Jizhe Zhang, Mingqing Xiao, Yifei Wang, Yisen Wang, Zhouchen Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.21883)  

**Abstract**: Conventional research on large language models (LLMs) has primarily focused on refining output distributions, while paying less attention to the decoding process that transforms these distributions into final responses. Recent advances, such as scaling the computation of inference time with reward models, have underscored the importance of decoding, but these methods often suffer from high computational costs and limited applicability. In this paper, we revisit LLM generation through the lens of recommender systems, conceptualizing the decoding process as analogous to the ranking stage in recommendation pipelines. From this perspective, we observe that both traditional decoding methods and reward models exhibit clear limitations such as redundancy. Motivated by this insight, we propose Language Ranker, a novel framework that introduces a lightweight module to rerank candidate responses using features extracted by the base model. Experiments across a wide range of tasks show that Language Ranker achieves performance comparable to large-scale reward models, while requiring only <0.5M additional parameters, significantly reducing the computational overhead during both training and inference stages. This highlights the efficiency and effectiveness of our method, showcasing its potential to fully unlock the capabilities of LLMs. 

---
# TernaryCLIP: Efficiently Compressing Vision-Language Models with Ternary Weights and Distilled Knowledge 

**Authors**: Shu-Hao Zhang, Wei-Cheng Tang, Chen Wu, Peng Hu, Nan Li, Liang-Jie Zhang, Qi Zhang, Shao-Qun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.21879)  

**Abstract**: Recent years have witnessed an increasing interest in image-text contrastive modeling, exemplified by models such as Contrastive Language-Image Pretraining (CLIP). In this paper, we propose the TernaryCLIP, a lightweight computational framework that converts connection weights of both vision and text encoders of CLIP into the ternary format, instead of full-precision or floating ones. TernaryCLIP incorporates quantization-aware training and distillation modules, preventing precision degradation and enabling low-cost and high-efficiency computations. Comprehensive experiments demonstrate that TernaryCLIP can achieve up to 99\% ternarized weights with 1.58-bit representation, 16.98 $\times$ compression ratio, 2.3 $\times$ inference acceleration, 16 $\times$ storage reduction, 10 $\times$ memory optimization, and 60\% sparsity while maintaining promising performance on zero-shot image classification and image-text retrieval tasks across 41 commonly used datasets. Our work highlights the feasibility of extreme quantization for large multimodal models, supporting effective and efficient deployment on resource-constrained devices. The model and code can be accessed from Hugging Face and GitHub. 

---
# AI Powered Urban Green Infrastructure Assessment Through Aerial Imagery of an Industrial Township 

**Authors**: Anisha Dutta  

**Link**: [PDF](https://arxiv.org/pdf/2510.21876)  

**Abstract**: Accurate assessment of urban canopy coverage is crucial for informed urban planning, effective environmental monitoring, and mitigating the impacts of climate change. Traditional practices often face limitations due to inadequate technical requirements, difficulties in scaling and data processing, and the lack of specialized expertise. This study presents an efficient approach for estimating green canopy coverage using artificial intelligence, specifically computer vision techniques, applied to aerial imageries. Our proposed methodology utilizes object-based image analysis, based on deep learning algorithms to accurately identify and segment green canopies from high-resolution drone images. This approach allows the user for detailed analysis of urban vegetation, capturing variations in canopy density and understanding spatial distribution. To overcome the computational challenges associated with processing large datasets, it was implemented over a cloud platform utilizing high-performance processors. This infrastructure efficiently manages space complexity and ensures affordable latency, enabling the rapid analysis of vast amounts of drone imageries. Our results demonstrate the effectiveness of this approach in accurately estimating canopy coverage at the city scale, providing valuable insights for urban forestry management of an industrial township. The resultant data generated by this method can be used to optimize tree plantation and assess the carbon sequestration potential of urban forests. By integrating these insights into sustainable urban planning, we can foster more resilient urban environments, contributing to a greener and healthier future. 

---
# A Physics-Informed Neural Network Approach for UAV Path Planning in Dynamic Environments 

**Authors**: Shuning Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.21874)  

**Abstract**: Unmanned aerial vehicles (UAVs) operating in dynamic wind fields must generate safe and energy-efficient trajectories under physical and environmental constraints. Traditional planners, such as A* and kinodynamic RRT*, often yield suboptimal or non-smooth paths due to discretization and sampling limitations. This paper presents a physics-informed neural network (PINN) framework that embeds UAV dynamics, wind disturbances, and obstacle avoidance directly into the learning process. Without requiring supervised data, the PINN learns dynamically feasible and collision-free trajectories by minimizing physical residuals and risk-aware objectives. Comparative simulations show that the proposed method outperforms A* and Kino-RRT* in control energy, smoothness, and safety margin, while maintaining similar flight efficiency. The results highlight the potential of physics-informed learning to unify model-based and data-driven planning, providing a scalable and physically consistent framework for UAV trajectory optimization. 

---
# GuitarFlow: Realistic Electric Guitar Synthesis From Tablatures via Flow Matching and Style Transfer 

**Authors**: Jackson Loth, Pedro Sarmento, Mark Sandler, Mathieu Barthet  

**Link**: [PDF](https://arxiv.org/pdf/2510.21872)  

**Abstract**: Music generation in the audio domain using artificial intelligence (AI) has witnessed steady progress in recent years. However for some instruments, particularly the guitar, controllable instrument synthesis remains limited in expressivity. We introduce GuitarFlow, a model designed specifically for electric guitar synthesis. The generative process is guided using tablatures, an ubiquitous and intuitive guitar-specific symbolic format. The tablature format easily represents guitar-specific playing techniques (e.g. bends, muted strings and legatos), which are more difficult to represent in other common music notation formats such as MIDI. Our model relies on an intermediary step of first rendering the tablature to audio using a simple sample-based virtual instrument, then performing style transfer using Flow Matching in order to transform the virtual instrument audio into more realistic sounding examples. This results in a model that is quick to train and to perform inference, requiring less than 6 hours of training data. We present the results of objective evaluation metrics, together with a listening test, in which we show significant improvement in the realism of the generated guitar audio from tablatures. 

---
# Addressing Corner Cases in Autonomous Driving: A World Model-based Approach with Mixture of Experts and LLMs 

**Authors**: Haicheng Liao, Bonan Wang, Junxian Yang, Chengyue Wang, Zhengbin He, Guohui Zhang, Chengzhong Xu, Zhenning Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.21867)  

**Abstract**: Accurate and reliable motion forecasting is essential for the safe deployment of autonomous vehicles (AVs), particularly in rare but safety-critical scenarios known as corner cases. Existing models often underperform in these situations due to an over-representation of common scenes in training data and limited generalization capabilities. To address this limitation, we present WM-MoE, the first world model-based motion forecasting framework that unifies perception, temporal memory, and decision making to address the challenges of high-risk corner-case scenarios. The model constructs a compact scene representation that explains current observations, anticipates future dynamics, and evaluates the outcomes of potential actions. To enhance long-horizon reasoning, we leverage large language models (LLMs) and introduce a lightweight temporal tokenizer that maps agent trajectories and contextual cues into the LLM's feature space without additional training, enriching temporal context and commonsense priors. Furthermore, a mixture-of-experts (MoE) is introduced to decompose complex corner cases into subproblems and allocate capacity across scenario types, and a router assigns scenes to specialized experts that infer agent intent and perform counterfactual rollouts. In addition, we introduce nuScenes-corner, a new benchmark that comprises four real-world corner-case scenarios for rigorous evaluation. Extensive experiments on four benchmark datasets (nuScenes, NGSIM, HighD, and MoCAD) showcase that WM-MoE consistently outperforms state-of-the-art (SOTA) baselines and remains robust under corner-case and data-missing conditions, indicating the promise of world model-based architectures for robust and generalizable motion forecasting in fully AVs. 

---
# A Multi-Stage Hybrid Framework for Automated Interpretation of Multi-View Engineering Drawings Using Vision Language Model 

**Authors**: Muhammad Tayyab Khan, Zane Yong, Lequn Chen, Wenhe Feng, Nicholas Yew Jin Tan, Seung Ki Moon  

**Link**: [PDF](https://arxiv.org/pdf/2510.21862)  

**Abstract**: Engineering drawings are fundamental to manufacturing communication, serving as the primary medium for conveying design intent, tolerances, and production details. However, interpreting complex multi-view drawings with dense annotations remains challenging using manual methods, generic optical character recognition (OCR) systems, or traditional deep learning approaches, due to varied layouts, orientations, and mixed symbolic-textual content. To address these challenges, this paper proposes a three-stage hybrid framework for the automated interpretation of 2D multi-view engineering drawings using modern detection and vision language models (VLMs). In the first stage, YOLOv11-det performs layout segmentation to localize key regions such as views, title blocks, and notes. The second stage uses YOLOv11-obb for orientation-aware, fine-grained detection of annotations, including measures, GD&T symbols, and surface roughness indicators. The third stage employs two Donut-based, OCR-free VLMs for semantic content parsing: the Alphabetical VLM extracts textual and categorical information from title blocks and notes, while the Numerical VLM interprets quantitative data such as measures, GD&T frames, and surface roughness. Two specialized datasets were developed to ensure robustness and generalization: 1,000 drawings for layout detection and 1,406 for annotation-level training. The Alphabetical VLM achieved an overall F1 score of 0.672, while the Numerical VLM reached 0.963, demonstrating strong performance in textual and quantitative interpretation, respectively. The unified JSON output enables seamless integration with CAD and manufacturing databases, providing a scalable solution for intelligent engineering drawing analysis. 

---
# The Mirror Loop: Recursive Non-Convergence in Generative Reasoning Systems 

**Authors**: Bentley DeVilling  

**Link**: [PDF](https://arxiv.org/pdf/2510.21861)  

**Abstract**: Large language models are often described as capable of reflective reasoning, yet recursive self-evaluation without external feedback frequently yields reformulation rather than progress. We test this prediction in a cross-provider study of 144 reasoning sequences across three models (OpenAI GPT-4o-mini, Anthropic Claude 3 Haiku, and Google Gemini 2.0 Flash) and four task families (arithmetic, code, explanation, reflection), each iterated ten times under two conditions: ungrounded self-critique and a minimal grounding intervention (a single verification step at iteration three). Mean informational change (delta I, measured via normalized edit distance) declined by 55% from early (0.193) to late (0.087) iterations in ungrounded runs, with consistent patterns across all three providers. Grounded runs showed a +28% rebound in informational change immediately after the intervention and sustained non-zero variance thereafter. Complementary measures-n-gram novelty, embedding drift, and character-level entropy-converged on the same pattern: reflection without contact tends toward informational closure. We interpret this as evidence for a structural limit on self-correction in generative reasoning: without an exchange of information with an independent verifier or environment, recursive inference approaches an attractor state of epistemic stasis. Minimal grounding functions as dissipative coupling, reintroducing informational flux. The cross-architecture consistency suggests the mirror loop arises from shared autoregressive training objectives rather than provider-specific alignment schemes. The results delineate when reflection is performative rather than epistemic and motivate design principles for grounded, cooperative reasoning. Materials and code are publicly available. 

---
# Butter-Bench: Evaluating LLM Controlled Robots for Practical Intelligence 

**Authors**: Callum Sharrock, Lukas Petersson, Hanna Petersson, Axel Backlund, Axel Wennström, Kristoffer Nordström, Elias Aronsson  

**Link**: [PDF](https://arxiv.org/pdf/2510.21860)  

**Abstract**: We present Butter-Bench, a benchmark evaluating large language model (LLM) controlled robots for practical intelligence, defined as the ability to navigate the messiness of the physical world. Current state-of-the-art robotic systems use a hierarchical architecture with LLMs in charge of high-level reasoning, and a Vision Language Action (VLA) model for low-level control. Butter-Bench evaluates the LLM part in isolation from the VLA. Although LLMs have repeatedly surpassed humans in evaluations requiring analytical intelligence, we find humans still outperform LLMs on Butter-Bench. The best LLMs score 40% on Butter-Bench, while the mean human score is 95%. LLMs struggled the most with multi-step spatial planning and social understanding. We also evaluate LLMs that are fine-tuned for embodied reasoning and conclude that this training does not improve their score on Butter-Bench. 

---
# Privacy-preserving Decision-focused Learning for Multi-energy Systems 

**Authors**: Yangze Zhou, Ruiyang Yao, Dalin Qin, Yixiong Jia, Yi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.21858)  

**Abstract**: Decision-making for multi-energy system (MES) dispatch depends on accurate load forecasting. Traditionally, load forecasting and decision-making for MES are implemented separately. Forecasting models are typically trained to minimize forecasting errors, overlooking their impact on downstream decision-making. To address this, decision-focused learning (DFL) has been studied to minimize decision-making costs instead. However, practical adoption of DFL in MES faces significant challenges: the process requires sharing sensitive load data and model parameters across multiple sectors, raising serious privacy issues. To this end, we propose a privacy-preserving DFL framework tailored for MES. Our approach introduces information masking to safeguard private data while enabling recovery of decision variables and gradients required for model training. To further enhance security for DFL, we design a safety protocol combining matrix decomposition and homomorphic encryption, effectively preventing collusion and unauthorized data access. Additionally, we developed a privacy-preserving load pattern recognition algorithm, enabling the training of specialized DFL models for heterogeneous load patterns. Theoretical analysis and comprehensive case studies, including real-world MES data, demonstrate that our framework not only protects privacy but also consistently achieves lower average daily dispatch costs compared to existing methods. 

---
# Poisson Flow Consistency Training 

**Authors**: Anthony Zhang, Mahmut Gokmen, Dennis Hein, Rongjun Ge, Wenjun Xia, Ge Wang, Jin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.21857)  

**Abstract**: The Poisson Flow Consistency Model (PFCM) is a consistency-style model based on the robust Poisson Flow Generative Model++ (PFGM++) which has achieved success in unconditional image generation and CT image denoising. Yet the PFCM can only be trained in distillation which limits the potential of the PFCM in many data modalities. The objective of this research was to create a method to train the PFCM in isolation called Poisson Flow Consistency Training (PFCT). The perturbation kernel was leveraged to remove the pretrained PFGM++, and the sinusoidal discretization schedule and Beta noise distribution were introduced in order to facilitate adaptability and improve sample quality. The model was applied to the task of low dose computed tomography image denoising and improved the low dose image in terms of LPIPS and SSIM. It also displayed similar denoising effectiveness as models like the Consistency Model. PFCT is established as a valid method of training the PFCM from its effectiveness in denoising CT images, showing potential with competitive results to other generative models. Further study is needed in the precise optimization of PFCT and in its applicability to other generative modeling tasks. The framework of PFCT creates more flexibility for the ways in which a PFCM can be created and can be applied to the field of generative modeling. 

---
# TowerVision: Understanding and Improving Multilinguality in Vision-Language Models 

**Authors**: André G. Viveiros, Patrick Fernandes, Saul Santos, Sonal Sannigrahi, Emmanouil Zaranis, Nuno M. Guerreiro, Amin Farajian, Pierre Colombo, Graham Neubig, André F. T. Martins  

**Link**: [PDF](https://arxiv.org/pdf/2510.21849)  

**Abstract**: Despite significant advances in vision-language models (VLMs), most existing work follows an English-centric design process, limiting their effectiveness in multilingual settings. In this work, we provide a comprehensive empirical study analyzing the impact of several multilingual design choices, such as training data composition, encoder selection, and text backbones. The result is TowerVision, a family of open multilingual VLMs for both image-text and video-text tasks, built upon the multilingual text-only model Tower+. TowerVision achieves competitive performance on multiple multimodal multilingual benchmarks and shows particular strength in culturally grounded tasks and multimodal translation. By incorporating visual and cultural context during fine-tuning, our models surpass existing approaches trained on substantially larger datasets, as demonstrated on ALM-Bench and Multi30K (image tasks) and ViMUL-Bench (video tasks). Alongside the models, we release VisionBlocks, a high-quality, curated vision-language dataset. Our findings highlight that multilingual vision-language training data substantially improves cross-lingual generalization -- both from high-resource to underrepresented languages and vice versa -- and that instruction-tuned LLMs are not always the optimal initialization point. To support further research, we publicly release all models, data, and training recipes. 

---
# Training data membership inference via Gaussian process meta-modeling: a post-hoc analysis approach 

**Authors**: Yongchao Huang, Pengfei Zhang, Shahzad Mumtaz  

**Link**: [PDF](https://arxiv.org/pdf/2510.21846)  

**Abstract**: Membership inference attacks (MIAs) test whether a data point was part of a model's training set, posing serious privacy risks. Existing methods often depend on shadow models or heavy query access, which limits their practicality. We propose GP-MIA, an efficient and interpretable approach based on Gaussian process (GP) meta-modeling. Using post-hoc metrics such as accuracy, entropy, dataset statistics, and optional sensitivity features (e.g. gradients, NTK measures) from a single trained model, GP-MIA trains a GP classifier to distinguish members from non-members while providing calibrated uncertainty estimates. Experiments on synthetic data, real-world fraud detection data, CIFAR-10, and WikiText-2 show that GP-MIA achieves high accuracy and generalizability, offering a practical alternative to existing MIAs. 

---
# Evaluating ChatGPT's Performance in Classifying Pneumonia from Chest X-Ray Images 

**Authors**: Pragna Prahallad, Pranathi Prahallad  

**Link**: [PDF](https://arxiv.org/pdf/2510.21839)  

**Abstract**: In this study, we evaluate the ability of OpenAI's gpt-4o model to classify chest X-ray images as either NORMAL or PNEUMONIA in a zero-shot setting, without any prior fine-tuning. A balanced test set of 400 images (200 from each class) was used to assess performance across four distinct prompt designs, ranging from minimal instructions to detailed, reasoning-based prompts. The results indicate that concise, feature-focused prompts achieved the highest classification accuracy of 74\%, whereas reasoning-oriented prompts resulted in lower performance. These findings highlight that while ChatGPT exhibits emerging potential for medical image interpretation, its diagnostic reliability remains limited. Continued advances in visual reasoning and domain-specific adaptation are required before such models can be safely applied in clinical practice. 

---
# A Multimodal, Multitask System for Generating E Commerce Text Listings from Images 

**Authors**: Nayan Kumar Singh  

**Link**: [PDF](https://arxiv.org/pdf/2510.21835)  

**Abstract**: Manually generating catchy descriptions and names is labor intensive and a slow process for retailers. Although generative AI provides an automation solution in form of Vision to Language Models (VLM), the current VLMs are prone to factual "hallucinations". Siloed, single task models are not only inefficient but also fail to capture interdependent relationships between features. To address these challenges, we propose an end to end, multi task system that generates factually grounded textual listings from a single image. The contributions of this study are two proposals for the model architecture. First, application of multi task learning approach for fine tuning a vision encoder where a single vision backbone is jointly trained on attribute prediction such as color, hemline and neck style and price regression. Second, introduction of a hierarchical generation process where the model's own predicted attributes are embedded in a prompt and fed to the text decoder to improve factual consistency. The experiments demonstrate the superiority of this architecture. The multi tasking approach outperforms both the independent price regression, with a 3.6% better R2 Value and attribute classification, with a 6.6% improvement F1 score. Critically, the hierarchical generation process proves highly effective, slashing the factual hallucination rate from 12.7% to 7.1%, a 44.5% relative reduction, compared to a non hierarchical ablation. The hierarchical approach also reduces the latency of the autoregressive text generation process by a factor of 3.5 when compared to direct vision to language model of similar size. One minor caveat is that the model does perform 3.5% worse than direct vision-to-language model on ROUGE-L score. 

---
# GAPO: Group Adaptive Policy Optimization for Real-World Code Edit 

**Authors**: Jianqing Zhang, Zhezheng Hao, Wei Xia, Hande Dong, Hong Wang, Chenxing Wei, Yuyan Zhou, Yubin Qi, Qiang Lin, Jian Cao  

**Link**: [PDF](https://arxiv.org/pdf/2510.21830)  

**Abstract**: Reinforcement learning (RL) is widely used for post-training large language models (LLMs) in code editing, where group-relative methods like GRPO are popular for their critic-free, normalized advantage estimation. However, in real-world code-editing scenarios, reward distributions are often skewed with unpredictable outliers, leading to distorted advantage computation and increased noise. To address this issue, we propose Group Adaptive Policy Optimization (GAPO), which adaptively finds an outlier-free highest-density interval (HDI) per prompt and then uses the median of that interval as an adaptive Q to replace the group mean in advantage calculation. This adaptive Q robustly handles skewed distributions while remaining plug-and-play and efficient. We validate GAPO on nine instruction-tuned LLMs (3B-14B) using a large internal dataset of 51,844 real-world, history-aware code-editing tasks across 10 languages, demonstrating consistent improvements in exact match accuracy over GRPO and its variant DAPO. Code is publicly available. 

---
# Precise classification of low quality G-banded Chromosome Images by reliability metrics and data pruning classifier 

**Authors**: Mojtaba Moattari  

**Link**: [PDF](https://arxiv.org/pdf/2510.21827)  

**Abstract**: In the last decade, due to high resolution cameras and accurate meta-phase analyzes, the accuracy of chromosome classification has improved substantially. However, current Karyotyping systems demand large number of high quality train data to have an adequately plausible Precision per each chromosome. Such provision of high quality train data with accurate devices are not yet accomplished in some out-reached pathological laboratories. To prevent false positive detections in low-cost systems and low-quality images settings, this paper improves the classification Precision of chromosomes using proposed reliability thresholding metrics and deliberately engineered features. The proposed method has been evaluated using a variation of deep Alex-Net neural network, SVM, K Nearest-Neighbors, and their cascade pipelines to an automated filtering of semi-straight chromosome. The classification results have highly improved over 90% for the chromosomes with more common defections and translocations. Furthermore, a comparative analysis over the proposed thresholding metrics has been conducted and the best metric is bolded with its salient characteristics. The high Precision results provided for a very low-quality G-banding database verifies suitability of the proposed metrics and pruning method for Karyotyping facilities in poor countries and lowbudget pathological laboratories. 

---
# Explainable Deep Learning in Medical Imaging: Brain Tumor and Pneumonia Detection 

**Authors**: Sai Teja Erukude, Viswa Chaitanya Marella, Suhasnadh Reddy Veluru  

**Link**: [PDF](https://arxiv.org/pdf/2510.21823)  

**Abstract**: Deep Learning (DL) holds enormous potential for improving medical imaging diagnostics, yet the lack of interpretability in most models hampers clinical trust and adoption. This paper presents an explainable deep learning framework for detecting brain tumors in MRI scans and pneumonia in chest X-ray images using two leading Convolutional Neural Networks, ResNet50 and DenseNet121. These models were trained on publicly available Kaggle datasets comprising 7,023 brain MRI images and 5,863 chest X-ray images, achieving high classification performance. DenseNet121 consistently outperformed ResNet50 with 94.3 percent vs. 92.5 percent accuracy for brain tumors and 89.1 percent vs. 84.4 percent accuracy for pneumonia. For better explainability, Gradient-weighted Class Activation Mapping (Grad-CAM) was integrated to create heatmap visualizations superimposed on the test images, indicating the most influential image regions in the decision-making process. Interestingly, while both models produced accurate results, Grad-CAM showed that DenseNet121 consistently focused on core pathological regions, whereas ResNet50 sometimes scattered attention to peripheral or non-pathological areas. Combining deep learning and explainable AI offers a promising path toward reliable, interpretable, and clinically useful diagnostic tools. 

---
# Wavelet-based GAN Fingerprint Detection using ResNet50 

**Authors**: Sai Teja Erukude, Suhasnadh Reddy Veluru, Viswa Chaitanya Marella  

**Link**: [PDF](https://arxiv.org/pdf/2510.21822)  

**Abstract**: Identifying images generated by Generative Adversarial Networks (GANs) has become a significant challenge in digital image forensics. This research presents a wavelet-based detection method that uses discrete wavelet transform (DWT) preprocessing and a ResNet50 classification layer to differentiate the StyleGAN-generated images from real ones. Haar and Daubechies wavelet filters are applied to convert the input images into multi-resolution representations, which will then be fed to a ResNet50 network for classification, capitalizing on subtle artifacts left by the generative process. Moreover, the wavelet-based models are compared to an identical ResNet50 model trained on spatial data. The Haar and Daubechies preprocessed models achieved a greater accuracy of 93.8 percent and 95.1 percent, much higher than the model developed in the spatial domain (accuracy rate of 81.5 percent). The Daubechies-based model outperforms Haar, showing that adding layers of descriptive frequency patterns can lead to even greater distinguishing power. These results indicate that the GAN-generated images have unique wavelet-domain artifacts or "fingerprints." The method proposed illustrates the effectiveness of wavelet-domain analysis to detect GAN images and emphasizes the potential of further developing the capabilities of future deepfake detection systems. 

---
# Prompt fidelity of ChatGPT4o / Dall-E3 text-to-image visualisations 

**Authors**: Dirk HR Spennemann  

**Link**: [PDF](https://arxiv.org/pdf/2510.21821)  

**Abstract**: This study examines the prompt fidelity of ChatGPT4o / DALL-E3 text-to-image visualisations by analysing whether attributes explicitly specified in autogenously generated prompts are correctly rendered in the resulting images. Using two public-domain datasets comprising 200 visualisations of women working in the cultural and creative industries and 230 visualisations of museum curators, the study assessed accuracy across personal attributes (age, hair), appearance (attire, glasses), and paraphernalia (name tags, clipboards). While correctly rendered in most cases, DALL-E3 deviated from prompt specifications in 15.6% of all attributes (n=710). Errors were lowest for paraphernalia, moderate for personal appearance, and highest for depictions of the person themselves, particularly age. These findings demonstrate measurable prompt-to-image fidelity gaps with implications for bias detection and model evaluation. 

---
# Unlocking Biomedical Insights: Hierarchical Attention Networks for High-Dimensional Data Interpretation 

**Authors**: Rekha R Nair, Tina Babu, Alavikunhu Panthakkan, Hussain Al-Ahmad, Balamurugan Balusamy  

**Link**: [PDF](https://arxiv.org/pdf/2510.21820)  

**Abstract**: The proliferation of high-dimensional datasets in fields such as genomics, healthcare, and finance has created an urgent need for machine learning models that are both highly accurate and inherently interpretable. While traditional deep learning approaches deliver strong predictive performance, their lack of transparency often impedes their deployment in critical, decision-sensitive applications. In this work, we introduce the Hierarchical Attention-based Interpretable Network (HAIN), a novel architecture that unifies multi-level attention mechanisms, dimensionality reduction, and explanation-driven loss functions to deliver interpretable and robust analysis of complex biomedical data. HAIN provides feature-level interpretability via gradientweighted attention and offers global model explanations through prototype-based representations. Comprehensive evaluation on The Cancer Genome Atlas (TCGA) dataset demonstrates that HAIN achieves a classification accuracy of 94.3%, surpassing conventional post-hoc interpretability approaches such as SHAP and LIME in both transparency and explanatory power. Furthermore, HAIN effectively identifies biologically relevant cancer biomarkers, supporting its utility for clinical and research applications. By harmonizing predictive accuracy with interpretability, HAIN advances the development of transparent AI solutions for precision medicine and regulatory compliance. 

---
# HDR Image Reconstruction using an Unsupervised Fusion Model 

**Authors**: Kumbha Nagaswetha  

**Link**: [PDF](https://arxiv.org/pdf/2510.21815)  

**Abstract**: High Dynamic Range (HDR) imaging aims to reproduce the wide range of brightness levels present in natural scenes, which the human visual system can perceive but conventional digital cameras often fail to capture due to their limited dynamic range. To address this limitation, we propose a deep learning-based multi-exposure fusion approach for HDR image generation. The method takes a set of differently exposed Low Dynamic Range (LDR) images, typically an underexposed and an overexposed image, and learns to fuse their complementary information using a convolutional neural network (CNN). The underexposed image preserves details in bright regions, while the overexposed image retains information in dark regions; the network effectively combines these to reconstruct a high-quality HDR output. The model is trained in an unsupervised manner, without relying on ground-truth HDR images, making it practical for real-world applications where such data is unavailable. We evaluate our results using the Multi-Exposure Fusion Structural Similarity Index Measure (MEF-SSIM) and demonstrate that our approach achieves superior visual quality compared to existing fusion methods. A customized loss function is further introduced to improve reconstruction fidelity and optimize model performance. 

---
# Gestura: A LVLM-Powered System Bridging Motion and Semantics for Real-Time Free-Form Gesture Understanding 

**Authors**: Zhuoming Li, Aitong Liu, Mengxi Jia, Tengxiang Zhang, Dell Zhang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.21814)  

**Abstract**: Free-form gesture understanding is highly appealing for human-computer interaction, as it liberates users from the constraints of predefined gesture categories. However, the sole existing solution GestureGPT suffers from limited recognition accuracy and slow response times. In this paper, we propose Gestura, an end-to-end system for free-form gesture understanding. Gestura harnesses a pre-trained Large Vision-Language Model (LVLM) to align the highly dynamic and diverse patterns of free-form gestures with high-level semantic concepts. To better capture subtle hand movements across different styles, we introduce a Landmark Processing Module that compensate for LVLMs' lack of fine-grained domain knowledge by embedding anatomical hand priors. Further, a Chain-of-Thought (CoT) reasoning strategy enables step-by-step semantic inference, transforming shallow knowledge into deep semantic understanding and significantly enhancing the model's ability to interpret ambiguous or unconventional gestures. Together, these components allow Gestura to achieve robust and adaptable free-form gesture comprehension. Additionally, we have developed the first open-source dataset for free-form gesture intention reasoning and understanding with over 300,000 annotated QA pairs. 

---
# SITS-DECO: A Generative Decoder Is All You Need For Multitask Satellite Image Time Series Modelling 

**Authors**: Samuel J. Barrett, Docko Sow  

**Link**: [PDF](https://arxiv.org/pdf/2510.21813)  

**Abstract**: Earth Observation (EO) Foundation Modelling (FM) holds great promise for simplifying and improving the use of EO data for diverse real-world tasks. However, most existing models require additional adaptation before they can be used and are structured rigidly around particular data sources or training approaches. To address this, we take inspiration from large language models, where diverse tasks, both pre-training and downstream, are implicitly captured through next-token prediction over unified token sequences, leveraging the structure and diversity of the training data.
We introduce SITS-DECO (Satellite Image Time Series-DECoder Only), a proof-of-concept generative model that applies this unified-sequence framing to EO data. Using a simple GPT-style decoder-only architecture, and demonstrate its ability to perform useful EO tasks (pixel-wise, multi-temporal, multi-modal crop-type classification) in a purely generative framework. Through symbolic prompting, we show that the model can perform multiple supervised and self-supervised tasks within a single unified architecture, without task- or modality-specific adaptation. Despite its simplicity and lack of spatial context, SITS-DECO outperforms much larger EO foundation models on crop-type classification (PASTIS-R) demonstrating that dense temporal sequence modelling is a critical missing ingredient in the current paradigm.
This work exemplifies a data-centric modelling paradigm in which capability arises from the diversity and structure of the training data rather than from architectural complexity. SITS-DECO provides a lightweight, practical route to multi-modal, multi-task EO modelling, and a conceptual bridge toward future generative EO foundation models. 

---
# Unifying Inductive, Cross-Domain, and Multimodal Learning for Robust and Generalizable Recommendation 

**Authors**: Chanyoung Chung, Kyeongryul Lee, Sunbin Park, Joyce Jiyoung Whang  

**Link**: [PDF](https://arxiv.org/pdf/2510.21812)  

**Abstract**: Recommender systems have long been built upon the modeling of interactions between users and items, while recent studies have sought to broaden this paradigm by generalizing to new users and items, incorporating diverse information sources, and transferring knowledge across domains. Nevertheless, these efforts have largely focused on individual aspects, hindering their ability to tackle the complex recommendation scenarios that arise in daily consumptions across diverse domains. In this paper, we present MICRec, a unified framework that fuses inductive modeling, multimodal guidance, and cross-domain transfer to capture user contexts and latent preferences in heterogeneous and incomplete real-world data. Moving beyond the inductive backbone of INMO, our model refines expressive representations through modality-based aggregation and alleviates data sparsity by leveraging overlapping users as anchors across domains, thereby enabling robust and generalizable recommendation. Experiments show that MICRec outperforms 12 baselines, with notable gains in domains with limited training data. 

---
# Comparative Analysis of Object Detection Algorithms for Surface Defect Detection 

**Authors**: Arpan Maity, Tamal Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2510.21811)  

**Abstract**: This article compares the performance of six prominent object detection algorithms, YOLOv11, RetinaNet, Fast R-CNN, YOLOv8, RT-DETR, and DETR, on the NEU-DET surface defect detection dataset, comprising images representing various metal surface defects, a crucial application in industrial quality control. Each model's performance was assessed regarding detection accuracy, speed, and robustness across different defect types such as scratches, inclusions, and rolled-in scales. YOLOv11, a state-of-the-art real-time object detection algorithm, demonstrated superior performance compared to the other methods, achieving a remarkable 70% higher accuracy on average. This improvement can be attributed to YOLOv11s enhanced feature extraction capabilities and ability to process the entire image in a single forward pass, making it faster and more efficient in detecting minor surface defects. Additionally, YOLOv11's architecture optimizations, such as improved anchor box generation and deeper convolutional layers, contributed to more precise localization of defects. In conclusion, YOLOv11's outstanding performance in accuracy and speed solidifies its position as the most effective model for surface defect detection on the NEU dataset, surpassing competing algorithms by a substantial margin. 

---
# Hybrid Deep Learning Framework for Enhanced Diabetic Retinopathy Detection: Integrating Traditional Features with AI-driven Insights 

**Authors**: Arpan Maity, Aviroop Pal, MD. Samiul Islam, Tamal Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2510.21810)  

**Abstract**: Diabetic Retinopathy (DR), a vision-threatening complication of Dia-betes Mellitus (DM), is a major global concern, particularly in India, which has one of the highest diabetic populations. Prolonged hyperglycemia damages reti-nal microvasculature, leading to DR symptoms like microaneurysms, hemor-rhages, and fluid leakage, which, if undetected, cause irreversible vision loss. Therefore, early screening is crucial as DR is asymptomatic in its initial stages. Fundus imaging aids precise diagnosis by detecting subtle retinal lesions. This paper introduces a hybrid diagnostic framework combining traditional feature extraction and deep learning (DL) to enhance DR detection. While handcrafted features capture key clinical markers, DL automates hierarchical pattern recog-nition, improving early diagnosis. The model synergizes interpretable clinical data with learned features, surpassing standalone DL approaches that demon-strate superior classification and reduce false negatives. This multimodal AI-driven approach enables scalable, accurate DR screening, crucial for diabetes-burdened regions. 

---
# Semantic Relation-Enhanced CLIP Adapter for Domain Adaptive Zero-Shot Learning 

**Authors**: Jiaao Yu, Mingjie Han, Jinkun Jiang, Junyu Dong, Tao Gong, Man Lan  

**Link**: [PDF](https://arxiv.org/pdf/2510.21808)  

**Abstract**: The high cost of data annotation has spurred research on training deep learning models in data-limited scenarios. Existing paradigms, however, fail to balance cross-domain transfer and cross-category generalization, giving rise to the demand for Domain-Adaptive Zero-Shot Learning (DAZSL). Although vision-language models (e.g., CLIP) have inherent advantages in the DAZSL field, current studies do not fully exploit their potential. Applying CLIP to DAZSL faces two core challenges: inefficient cross-category knowledge transfer due to the lack of semantic relation guidance, and degraded cross-modal alignment during target domain fine-tuning. To address these issues, we propose a Semantic Relation-Enhanced CLIP (SRE-CLIP) Adapter framework, integrating a Semantic Relation Structure Loss and a Cross-Modal Alignment Retention Strategy. As the first CLIP-based DAZSL method, SRE-CLIP achieves state-of-the-art performance on the I2AwA and I2WebV benchmarks, significantly outperforming existing approaches. 

---
# Activating Visual Context and Commonsense Reasoning through Masked Prediction in VLMs 

**Authors**: Jiaao Yu, Shenwei Li, Mingjie Han, Yifei Yin, Wenzheng Song, Chenghao Jia, Man Lan  

**Link**: [PDF](https://arxiv.org/pdf/2510.21807)  

**Abstract**: Recent breakthroughs in reasoning models have markedly advanced the reasoning capabilities of large language models, particularly via training on tasks with verifiable rewards. Yet, a significant gap persists in their adaptation to real world multimodal scenarios, most notably, vision language tasks, due to a heavy focus on single modal language settings. While efforts to transplant reinforcement learning techniques from NLP to VLMs have emerged, these approaches often remain confined to perception centric tasks or reduce images to textual summaries, failing to fully exploit visual context and commonsense knowledge, ultimately constraining the generalization of reasoning capabilities across diverse multimodal environments. To address this limitation, we introduce a novel fine tuning task, Masked Prediction via Context and Commonsense, which forces models to integrate visual context and commonsense reasoning by reconstructing semantically meaningful content from occluded images, thereby laying the foundation for generalized reasoning. To systematically evaluate the model performance in generalized reasoning, we developed a specialized evaluation benchmark, MPCC Eval, and employed various fine tuning strategies to guide reasoning. Among these, we introduced an innovative training method, Reinforcement Fine tuning with Prior Sampling, which not only enhances model performance but also improves its generalized reasoning capabilities in OOD and cross task scenarios. 

---
# Frame-Difference Guided Dynamic Region Perception for CLIP Adaptation in Text-Video Retrieval 

**Authors**: Jiaao Yu, Mingjie Han, Tao Gong, Jian Zhang, Man Lan  

**Link**: [PDF](https://arxiv.org/pdf/2510.21806)  

**Abstract**: With the rapid growth of video data, text-video retrieval technology has become increasingly important in numerous application scenarios such as recommendation and search. Early text-video retrieval methods suffer from two critical drawbacks: first, they heavily rely on large-scale annotated video-text pairs, leading to high data acquisition costs; second, there is a significant modal gap between video and text features, which limits cross-modal alignment accuracy. With the development of vision-language model, adapting CLIP to video tasks has attracted great attention. However, existing adaptation methods generally lack enhancement for dynamic video features and fail to effectively suppress static redundant features. To address this issue, this paper proposes FDA-CLIP (Frame Difference Alpha-CLIP), which is a concise CLIP-based training framework for text-video alignment. Specifically, the method uses frame differences to generate dynamic region masks, which are input into Alpha-CLIP as an additional Alpha channel. This proactively guides the model to focus on semantically critical dynamic regions while suppressing static background redundancy. Experiments demonstrate that frame difference-guided video semantic encoding can effectively balance retrieval efficiency and accuracy. 

---
# DiffGRM: Diffusion-based Generative Recommendation Model 

**Authors**: Zhao Liu, Yichen Zhu, Yiqing Yang, Guoping Tang, Rui Huang, Qiang Luo, Xiao Lv, Ruiming Tang, Kun Gai, Guorui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.21805)  

**Abstract**: Generative recommendation (GR) is an emerging paradigm that represents each item via a tokenizer as an n-digit semantic ID (SID) and predicts the next item by autoregressively generating its SID conditioned on the user's history. However, two structural properties of SIDs make ARMs ill-suited. First, intra-item consistency: the n digits jointly specify one item, yet the left-to-right causality trains each digit only under its prefix and blocks bidirectional cross-digit evidence, collapsing supervision to a single causal path. Second, inter-digit heterogeneity: digits differ in semantic granularity and predictability, while the uniform next-token objective assigns equal weight to all digits, overtraining easy digits and undertraining hard digits. To address these two issues, we propose DiffGRM, a diffusion-based GR model that replaces the autoregressive decoder with a masked discrete diffusion model (MDM), thereby enabling bidirectional context and any-order parallel generation of SID digits for recommendation. Specifically, we tailor DiffGRM in three aspects: (1) tokenization with Parallel Semantic Encoding (PSE) to decouple digits and balance per-digit information; (2) training with On-policy Coherent Noising (OCN) that prioritizes uncertain digits via coherent masking to concentrate supervision on high-value signals; and (3) inference with Confidence-guided Parallel Denoising (CPD) that fills higher-confidence digits first and generates diverse Top-K candidates. Experiments show consistent gains over strong generative and discriminative recommendation baselines on multiple datasets, improving NDCG@10 by 6.9%-15.5%. Code is available at this https URL. 

---
# Quantifying Multimodal Imbalance: A GMM-Guided Adaptive Loss for Audio-Visual Learning 

**Authors**: Zhaocheng Liu, Zhiwen Yu, Xiaoqing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.21797)  

**Abstract**: Current mainstream approaches to addressing multimodal imbalance primarily focus on architectural modifications and optimization-based, often overlooking a quantitative analysis of the imbalance degree between modalities. To address this gap, our work introduces a novel method for the quantitative analysis of multi-modal imbalance, which in turn informs the design of a sample-level adaptive loss this http URL begin by defining the "Modality Gap" as the difference between the Softmax scores of different modalities (e.g., audio and visual) for the ground-truth class prediction. Analysis of the Modality Gap distribution reveals that it can be effectively modeled by a bimodal Gaussian Mixture Model (GMM). These two components are found to correspond respectively to "modality-balanced" and "modality-imbalanced" data samples. Subsequently, we apply Bayes' theorem to compute the posterior probability of each sample belonging to these two distinct this http URL by this quantitative analysis, we design a novel adaptive loss function with three objectives: (1) to minimize the overall Modality Gap; (2) to encourage the imbalanced sample distribution to shift towards the balanced one; and (3) to apply greater penalty weights to imbalanced samples. We employ a two-stage training strategy consisting of a warm-up phase followed by an adaptive training this http URL results demonstrate that our approach achieves state-of-the-art (SOTA) performance on the public CREMA-D and AVE datasets, attaining accuracies of $80.65\%$ and $70.90\%$, respectively. This validates the effectiveness of our proposed methodology. 

---
# A Physics-Guided AI Cascaded Corrector Model Significantly Extends Madden-Julian Oscillation Prediction Skill 

**Authors**: Xiao Zhou, Yuze Sun, Jie Wu, Xiaomeng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.21796)  

**Abstract**: The Madden-Julian Oscillation (MJO) is an important driver of global weather and climate extremes, but its prediction in operational dynamical models remains challenging, with skillful forecasts typically limited to 3-4 weeks. Here, we introduce a novel deep learning framework, the Physics-guided Cascaded Corrector for MJO (PCC-MJO), which acts as a universal post-processor to correct MJO forecasts from dynamical models. This two-stage model first employs a physics-informed 3D U-Net to correct spatial-temporal field errors, then refines the MJO's RMM index using an LSTM optimized for forecast skill. When applied to three different operational forecasts from CMA, ECMWF and NCEP, our unified framework consistently extends the skillful forecast range (bivariate correlation > 0.5) by 2-8 days. Crucially, the model effectively mitigates the "Maritime Continent barrier", enabling more realistic eastward propagation and amplitude. Explainable AI analysis quantitatively confirms that the model's decision-making is spatially congruent with observed MJO dynamics (correlation > 0.93), demonstrating that it learns physically meaningful features rather than statistical fittings. Our work provides a promising physically consistent, computationally efficient, and highly generalizable pathway to break through longstanding barriers in subseasonal forecasting. 

---
# Xihe: Scalable Zero-Shot Time Series Learner Via Hierarchical Interleaved Block Attention 

**Authors**: Yinbo Sun, Yuchen Fang, Zhibo Zhu, Jia Li, Yu Liu, Qiwen Deng, Jun Zhou, Hang Yu, Xingyu Lu, Lintao Ma  

**Link**: [PDF](https://arxiv.org/pdf/2510.21795)  

**Abstract**: The rapid advancement of time series foundation models (TSFMs) has been propelled by migrating architectures from language models. While existing TSFMs demonstrate impressive performance, their direct adoption of cross-domain architectures constrains effective capture of multiscale temporal dependencies inherent to time series data. This limitation becomes particularly pronounced during zero-shot transfer across datasets with divergent underlying patterns and sampling strategies. To address these challenges, we propose Hierarchical Interleaved Block Attention (HIBA) which employs hierarchical inter- and intra-block sparse attention to effectively capture multi-scale dependencies. Intra-block attention facilitates local information exchange, and inter-block attention operates across blocks to capture global temporal pattern interaction and dynamic evolution. Leveraging the HIBA architecture, we introduce Xihe, a scalable TSFM family spanning from an ultra-efficient 9.5M parameter configuration to high-capacity 1.5B variant. Evaluated on the comprehensive GIFT-Eval benchmark, our most compact Xihe-tiny model (9.5M) surpasses the majority of contemporary TSFMs, demonstrating remarkable parameter efficiency. More impressively, Xihe-max (1.5B) establishes new state-of-the-art zero-shot performance, surpassing previous best results by a substantial margin. This consistent performance excellence across the entire parameter spectrum provides compelling evidence for the exceptional generalization capabilities and architectural superiority of HIBA. 

---
# Token-Level Inference-Time Alignment for Vision-Language Models 

**Authors**: Kejia Chen, Jiawen Zhang, Jiacong Hu, Kewei Gao, Jian Lou, Zunlei Feng, Mingli Song  

**Link**: [PDF](https://arxiv.org/pdf/2510.21794)  

**Abstract**: Vision-Language Models (VLMs) have become essential backbones of modern multimodal intelligence, yet their outputs remain prone to hallucination-plausible text misaligned with visual inputs. Existing alignment approaches often rely on expensive fine-tuning with annotated preference data or sequence-level inference strategies that provide only coarse, delayed feedback. To overcome these limitations, we present TITA (Token-level Inference-Time Alignment), a lightweight framework that freezes the base VLM and instead trains a reward model to approximate its distribution. During inference, implicit preference signals are extracted as log-probability ratios between the reward model and the target VLM, yielding dense autoregressive feedback. This formulation can be viewed as an inference-time variant of Direct Preference Optimization (DPO), providing token-level corrective signals without retraining the backbone. Extensive evaluations on LLaVA-1.5-7B and 13B show consistent gains across 12 benchmarks, with improvements of 8.6% on MMVet and 6.7% on POPE, indicating stronger general understanding and reduced hallucinations. Additional experiments on Qwen2.5-VL-7B and DeepSeek-VL2-27.5B show comparable gains, especially in hallucination reduction and VQA accuracy, while incurring negligible inference overhead. 

---
# 2D_3D Feature Fusion via Cross-Modal Latent Synthesis and Attention Guided Restoration for Industrial Anomaly Detection 

**Authors**: Usman Ali, Ali Zia, Abdul Rehman, Umer Ramzan, Zohaib Hassan, Talha Sattar, Jing Wang, Wei Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2510.21793)  

**Abstract**: Industrial anomaly detection (IAD) increasingly benefits from integrating 2D and 3D data, but robust cross-modal fusion remains challenging. We propose a novel unsupervised framework, Multi-Modal Attention-Driven Fusion Restoration (MAFR), which synthesises a unified latent space from RGB images and point clouds using a shared fusion encoder, followed by attention-guided, modality-specific decoders. Anomalies are localised by measuring reconstruction errors between input features and their restored counterparts. Evaluations on the MVTec 3D-AD and Eyecandies benchmarks demonstrate that MAFR achieves state-of-the-art results, with a mean I-AUROC of 0.972 and 0.901, respectively. The framework also exhibits strong performance in few-shot learning settings, and ablation studies confirm the critical roles of the fusion architecture and composite loss. MAFR offers a principled approach for fusing visual and geometric information, advancing the robustness and accuracy of industrial anomaly detection. Code is available at this https URL 

---
# Variance-Reduction Guidance: Sampling Trajectory Optimization for Diffusion Models 

**Authors**: Shifeng Xu, Yanzhu Liu, Adams Wai-Kin Kong  

**Link**: [PDF](https://arxiv.org/pdf/2510.21792)  

**Abstract**: Diffusion models have become emerging generative models. Their sampling process involves multiple steps, and in each step the models predict the noise from a noisy sample. When the models make prediction, the output deviates from the ground truth, and we call such a deviation as \textit{prediction error}. The prediction error accumulates over the sampling process and deteriorates generation quality. This paper introduces a novel technique for statistically measuring the prediction error and proposes the Variance-Reduction Guidance (VRG) method to mitigate this error. VRG does not require model fine-tuning or modification. Given a predefined sampling trajectory, it searches for a new trajectory which has the same number of sampling steps but produces higher quality results. VRG is applicable to both conditional and unconditional generation. Experiments on various datasets and baselines demonstrate that VRG can significantly improve the generation quality of diffusion models. Source code is available at this https URL. 

---
# Online Mixture of Experts: No-Regret Learning for Optimal Collective Decision-Making 

**Authors**: Larkin Liu, Jalal Etesami  

**Link**: [PDF](https://arxiv.org/pdf/2510.21788)  

**Abstract**: We explore the use of expert-guided bandit learning, which we refer to as online mixture-of-experts (OMoE). In this setting, given a context, a candidate committee of experts must determine how to aggregate their outputs to achieve optimal results in terms of aggregate accuracy. We propose two algorithms to address this problem. The first algorithm combines aggregate voting with UCB-driven successive elimination, efficiently pruning suboptimal exploration actions. The second algorithm employs an online weighted-majority-voting mechanism, leveraging the respective voting power of each expert proportional to their predictive power. We derive theoretical guarantees for the regret properties in the bandit setting under ideal circumstances, and empirical results are provided accordingly. As a modern study on applications, these methods are applied to the online fine-tuning of a set of expert large language models (LLMs), where after each response, the generative LLM dynamically reweighs its set of experts and/or selects the optimal committee of experts to generate the most accurate response. Our results introduce new methodologies and no-regret guarantees for combining multiple experts to improve on the performance of the an aggregate model overall. 

---
# EventFormer: A Node-graph Hierarchical Attention Transformer for Action-centric Video Event Prediction 

**Authors**: Qile Su, Shoutai Zhu, Shuai Zhang, Baoyu Liang, Chao Tong  

**Link**: [PDF](https://arxiv.org/pdf/2510.21786)  

**Abstract**: Script event induction, which aims to predict the subsequent event based on the context, is a challenging task in NLP, achieving remarkable success in practical applications. However, human events are mostly recorded and presented in the form of videos rather than scripts, yet there is a lack of related research in the realm of vision. To address this problem, we introduce AVEP (Action-centric Video Event Prediction), a task that distinguishes itself from existing video prediction tasks through its incorporation of more complex logic and richer semantic information. We present a large structured dataset, which consists of about $35K$ annotated videos and more than $178K$ video clips of event, built upon existing video event datasets to support this task. The dataset offers more fine-grained annotations, where the atomic unit is represented as a multimodal event argument node, providing better structured representations of video events. Due to the complexity of event structures, traditional visual models that take patches or frames as input are not well-suited for AVEP. We propose EventFormer, a node-graph hierarchical attention based video event prediction model, which can capture both the relationships between events and their arguments and the coreferencial relationships between arguments. We conducted experiments using several SOTA video prediction models as well as LVLMs on AVEP, demonstrating both the complexity of the task and the value of the dataset. Our approach outperforms all these video prediction models. We will release the dataset and code for replicating the experiments and annotations. 

---
# Noise Aggregation Analysis Driven by Small-Noise Injection: Efficient Membership Inference for Diffusion Models 

**Authors**: Guo Li, Yuyang Yu, Xuemiao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.21783)  

**Abstract**: Diffusion models have demonstrated powerful performance in generating high-quality images. A typical example is text-to-image generator like Stable Diffusion. However, their widespread use also poses potential privacy risks. A key concern is membership inference attacks, which attempt to determine whether a particular data sample was used in the model training process. We propose an efficient membership inference attack method against diffusion models. This method is based on the injection of slight noise and the evaluation of the aggregation degree of the noise distribution. The intuition is that the noise prediction patterns of diffusion models for training set samples and non-training set samples exhibit distinguishable this http URL, we suppose that member images exhibit higher aggregation of predicted noise around a certain time step of the diffusion process. In contrast, the predicted noises of non-member images exhibit a more discrete characteristic around the certain time step. Compared with other existing methods, our proposed method requires fewer visits to the target diffusion model. We inject slight noise into the image under test and then determine its membership by analyzing the aggregation degree of the noise distribution predicted by the model. Empirical findings indicate that our method achieves superior performance across multiple datasets. At the same time, our method can also show better attack effects in ASR and AUC when facing large-scale text-to-image diffusion models, proving the scalability of our method. 

---
# EdgeSync: Accelerating Edge-Model Updates for Data Drift through Adaptive Continuous Learning 

**Authors**: Runchu Donga, Peng Zhao, Guiqin Wang, Nan Qi, Jie Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.21781)  

**Abstract**: Real-time video analytics systems typically deploy lightweight models on edge devices to reduce latency. However, the distribution of data features may change over time due to various factors such as changing lighting and weather conditions, leading to decreased model accuracy. Recent frameworks try to address this issue by leveraging remote servers to continuously train and adapt lightweight edge models using more complex models in the cloud. Despite these advancements, existing methods face two key challenges: first, the retraining process is compute-intensive, causing significant delays in model updates; second, the new model may not align well with the evolving data distribution of the current video stream. To address these challenges, we introduce EdgeSync, an efficient edge-model updating approach that enhances sample filtering by incorporating timeliness and inference results, thus ensuring training samples are more relevant to the current video content while reducing update delays. Additionally, EdgeSync features a dynamic training management module that optimizes the timing and sequencing of model updates to improve their timeliness. Evaluations on diverse and complex real-world datasets demonstrate that EdgeSync improves accuracy by approximately 3.4% compared to existing methods and by about 10% compared to traditional approaches. 

---
# Bridging Accuracy and Interpretability: Deep Learning with XAI for Breast Cancer Detection 

**Authors**: Bishal Chhetri, B.V. Rathish Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2510.21780)  

**Abstract**: In this study, we present an interpretable deep learning framework for the early detection of breast cancer using quantitative features extracted from digitized fine needle aspirate (FNA) images of breast masses. Our deep neural network, using ReLU activations, the Adam optimizer, and a binary cross-entropy loss, delivers state-of-the-art classification performance, achieving an accuracy of 0.992, precision of 1.000, recall of 0.977, and an F1 score of 0.988. These results substantially exceed the benchmarks reported in the literature. We evaluated the model under identical protocols against a suite of well-established algorithms (logistic regression, decision trees, random forests, stochastic gradient descent, K-nearest neighbors, and XGBoost) and found the deep model consistently superior on the same metrics. Recognizing that high predictive accuracy alone is insufficient for clinical adoption due to the black-box nature of deep learning models, we incorporated model-agnostic Explainable AI techniques such as SHAP and LIME to produce feature-level attributions and human-readable visualizations. These explanations quantify the contribution of each feature to individual predictions, support error analysis, and increase clinician trust, thus bridging the gap between performance and interpretability for real-world clinical use. The concave points feature of the cell nuclei is found to be the most influential feature positively impacting the classification task. This insight can be very helpful in improving the diagnosis and treatment of breast cancer by highlighting the key characteristics of breast tumor. 

---
# What Causes Postoperative Aspiration? 

**Authors**: Supriya Nagesh, Karina Covarrubias, Robert El-Kareh, Shiva Prasad Kasiviswanathan, Nina Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2510.21779)  

**Abstract**: Background: Aspiration, the inhalation of foreign material into the lungs, significantly impacts surgical patient morbidity and mortality. This study develops a machine learning (ML) model to predict postoperative aspiration, enabling timely preventative interventions.
Methods: From the MIMIC-IV database of over 400,000 hospital admissions, we identified 826 surgical patients (mean age: 62, 55.7\% male) who experienced aspiration within seven days post-surgery, along with a matched non-aspiration cohort. Three ML models: XGBoost, Multilayer Perceptron, and Random Forest were trained using pre-surgical hospitalization data to predict postoperative aspiration. To investigate causation, we estimated Average Treatment Effects (ATE) using Augmented Inverse Probability Weighting.
Results: Our ML model achieved an AUROC of 0.86 and 77.3\% sensitivity on a held-out test set. Maximum daily opioid dose, length of stay, and patient age emerged as the most important predictors. ATE analysis identified significant causative factors: opioids (0.25 +/- 0.06) and operative site (neck: 0.20 +/- 0.13, head: 0.19 +/- 0.13). Despite equal surgery rates across genders, men were 1.5 times more likely to aspirate and received 27\% higher maximum daily opioid dosages compared to women.
Conclusion: ML models can effectively predict postoperative aspiration risk, enabling targeted preventative measures. Maximum daily opioid dosage and operative site significantly influence aspiration risk. The gender disparity in both opioid administration and aspiration rates warrants further investigation. These findings have important implications for improving postoperative care protocols and aspiration prevention strategies. 

---
# Face-MakeUpV2: Facial Consistency Learning for Controllable Text-to-Image Generation 

**Authors**: Dawei Dai, Yinxiu Zhou, Chenghang Li, Guolai Jiang, Chengfang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.21775)  

**Abstract**: In facial image generation, current text-to-image models often suffer from facial attribute leakage and insufficient physical consistency when responding to local semantic instructions. In this study, we propose Face-MakeUpV2, a facial image generation model that aims to maintain the consistency of face ID and physical characteristics with the reference image. First, we constructed a large-scale dataset FaceCaptionMask-1M comprising approximately one million image-text-masks pairs that provide precise spatial supervision for the local semantic instructions. Second, we employed a general text-to-image pretrained model as the backbone and introduced two complementary facial information injection channels: a 3D facial rendering channel to incorporate the physical characteristics of the image and a global facial feature channel. Third, we formulated two optimization objectives for the supervised learning of our model: semantic alignment in the model's embedding space to mitigate the attribute leakage problem and perceptual loss on facial images to preserve ID consistency. Extensive experiments demonstrated that our Face-MakeUpV2 achieves best overall performance in terms of preserving face ID and maintaining physical consistency of the reference images. These results highlight the practical potential of Face-MakeUpV2 for reliable and controllable facial editing in diverse applications. 

---
# OCR-Quality: A Human-Annotated Dataset for OCR Quality Assessment 

**Authors**: Yulong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.21774)  

**Abstract**: We present OCR-Quality, a comprehensive human-annotated dataset designed for evaluating and developing OCR quality assessment methods. The dataset consists of 1,000 PDF pages converted to PNG images at 300 DPI, sampled from diverse real-world scenarios, including academic papers, textbooks, e-books, and multilingual documents. Each document has been processed using state-of-the-art Vision-Language Models (VLMs) and manually annotated with quality scores using a 4-level scoring system (1: Excellent, 2: Good, 3: Fair, 4: Poor). The dataset includes detailed source information, annotation guidelines, and representative cases across various difficulty levels. OCR-Quality addresses the critical need for reliable OCR quality assessment in real-world applications and provides a valuable benchmark for training and evaluating OCR verification systems. The dataset is publicly available at this https URL . 

---
# Proportion and Perspective Control for Flow-Based Image Generation 

**Authors**: Julien Boudier, Hugo Caselles-Dupré  

**Link**: [PDF](https://arxiv.org/pdf/2510.21763)  

**Abstract**: While modern text-to-image diffusion models generate high-fidelity images, they offer limited control over the spatial and geometric structure of the output. To address this, we introduce and evaluate two ControlNets specialized for artistic control: (1) a proportion ControlNet that uses bounding boxes to dictate the position and scale of objects, and (2) a perspective ControlNet that employs vanishing lines to control the 3D geometry of the scene. We support the training of these modules with data pipelines that leverage vision-language models for annotation and specialized algorithms for conditioning image synthesis. Our experiments demonstrate that both modules provide effective control but exhibit limitations with complex constraints. Both models are released on HuggingFace: this https URL 

---
# J-ORA: A Framework and Multimodal Dataset for Japanese Object Identification, Reference, Action Prediction in Robot Perception 

**Authors**: Jesse Atuhurra, Hidetaka Kamigaito, Taro Watanabe, Koichiro Yoshino  

**Link**: [PDF](https://arxiv.org/pdf/2510.21761)  

**Abstract**: We introduce J-ORA, a novel multimodal dataset that bridges the gap in robot perception by providing detailed object attribute annotations within Japanese human-robot dialogue scenarios. J-ORA is designed to support three critical perception tasks, object identification, reference resolution, and next-action prediction, by leveraging a comprehensive template of attributes (e.g., category, color, shape, size, material, and spatial relations). Extensive evaluations with both proprietary and open-source Vision Language Models (VLMs) reveal that incorporating detailed object attributes substantially improves multimodal perception performance compared to without object attributes. Despite the improvement, we find that there still exists a gap between proprietary and open-source VLMs. In addition, our analysis of object affordances demonstrates varying abilities in understanding object functionality and contextual relationships across different VLMs. These findings underscore the importance of rich, context-sensitive attribute annotations in advancing robot perception in dynamic environments. See project page at this https URL. 

---
# Diagnosing Bottlenecks in Data Visualization Understanding by Vision-Language Models 

**Authors**: Alexa R. Tartaglini, Satchel Grant, Daniel Wurgaft, Christopher Potts, Judith E. Fan  

**Link**: [PDF](https://arxiv.org/pdf/2510.21740)  

**Abstract**: Data visualizations are vital components of many scientific articles and news stories. Current vision-language models (VLMs) still struggle on basic data visualization understanding tasks, but the causes of failure remain unclear. Are VLM failures attributable to limitations in how visual information in the data visualization is encoded, how information is transferred between the vision and language modules, or how information is processed within the language module? We developed FUGU, a suite of data visualization understanding tasks, to precisely characterize potential sources of difficulty (e.g., extracting the position of data points, distances between them, and other summary statistics). We used FUGU to investigate three widely used VLMs. To diagnose the sources of errors produced by these models, we used activation patching and linear probes to trace information flow through models across a variety of prompting strategies. We found that some models fail to generate the coordinates of individual data points correctly, and these initial errors often lead to erroneous final responses. When these models are provided with the correct coordinates, performance improves substantially. Moreover, even when the model generates an incorrect response, the correct coordinates can be successfully read out from the latent representations in the vision encoder, suggesting that the source of these errors lies in the vision-language handoff. We further found that while providing correct coordinates helps with tasks involving one or a small number of data points, it generally worsens performance for tasks that require extracting statistical relationships across many data points. Fine-tuning models on FUGU also fails to yield ceiling performance. These findings point to architectural constraints in current VLMs that might pose significant challenges for reliable data visualization understanding. 

---
# Next-Generation LLM for UAV: From Natural Language to Autonomous Flight 

**Authors**: Liangqi Yuan, Chuhao Deng, Dong-Jun Han, Inseok Hwang, Sabine Brunswicker, Christopher G. Brinton  

**Link**: [PDF](https://arxiv.org/pdf/2510.21739)  

**Abstract**: With the rapid advancement of Large Language Models (LLMs), their capabilities in various automation domains, particularly Unmanned Aerial Vehicle (UAV) operations, have garnered increasing attention. Current research remains predominantly constrained to small-scale UAV applications, with most studies focusing on isolated components such as path planning for toy drones, while lacking comprehensive investigation of medium- and long-range UAV systems in real-world operational contexts. Larger UAV platforms introduce distinct challenges, including stringent requirements for airport-based take-off and landing procedures, adherence to complex regulatory frameworks, and specialized operational capabilities with elevated mission expectations. This position paper presents the Next-Generation LLM for UAV (NeLV) system -- a comprehensive demonstration and automation roadmap for integrating LLMs into multi-scale UAV operations. The NeLV system processes natural language instructions to orchestrate short-, medium-, and long-range UAV missions through five key technical components: (i) LLM-as-Parser for instruction interpretation, (ii) Route Planner for Points of Interest (POI) determination, (iii) Path Planner for waypoint generation, (iv) Control Platform for executable trajectory implementation, and (v) UAV monitoring. We demonstrate the system's feasibility through three representative use cases spanning different operational scales: multi-UAV patrol, multi-POI delivery, and multi-hop relocation. Beyond the current implementation, we establish a five-level automation taxonomy that charts the evolution from current LLM-as-Parser capabilities (Level 1) to fully autonomous LLM-as-Autopilot systems (Level 5), identifying technical prerequisites and research challenges at each stage. 

---
# Learn2Drive: A neural network-based framework for socially compliant automated vehicle control 

**Authors**: Yuhui Liu, Samannita Halder, Shian Wang, Tianyi Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.21736)  

**Abstract**: This study introduces a novel control framework for adaptive cruise control (ACC) in automated driving, leveraging Long Short-Term Memory (LSTM) networks and physics-informed constraints. As automated vehicles (AVs) adopt advanced features like ACC, transportation systems are becoming increasingly intelligent and efficient. However, existing AV control strategies primarily focus on optimizing the performance of individual vehicles or platoons, often neglecting their interactions with human-driven vehicles (HVs) and the broader impact on traffic flow. This oversight can exacerbate congestion and reduce overall system efficiency. To address this critical research gap, we propose a neural network-based, socially compliant AV control framework that incorporates social value orientation (SVO). This framework enables AVs to account for their influence on HVs and traffic dynamics. By leveraging AVs as mobile traffic regulators, the proposed approach promotes adaptive driving behaviors that reduce congestion, improve traffic efficiency, and lower energy consumption. Within this framework, we define utility functions for both AVs and HVs, which are optimized based on the SVO of each AV to balance its own control objectives with broader traffic flow considerations. Numerical results demonstrate the effectiveness of the proposed method in adapting to varying traffic conditions, thereby enhancing system-wide efficiency. Specifically, when the AV's control mode shifts from prioritizing energy consumption to optimizing traffic flow efficiency, vehicles in the following platoon experience at least a 58.99% increase in individual energy consumption alongside at least a 38.39% improvement in individual average speed, indicating significant enhancements in traffic dynamics. 

---
# A phase-aware AI car-following model for electric vehicles with adaptive cruise control: Development and validation using real-world data 

**Authors**: Yuhui Liu, Shian Wang, Ansel Panicker, Kate Embry, Ayana Asanova, Tianyi Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.21735)  

**Abstract**: Internal combustion engine (ICE) vehicles and electric vehicles (EVs) exhibit distinct vehicle dynamics. EVs provide rapid acceleration, with electric motors producing peak power across a wider speed range, and achieve swift deceleration through regenerative braking. While existing microscopic models effectively capture the driving behavior of ICE vehicles, a modeling framework that accurately describes the unique car-following dynamics of EVs is lacking. Developing such a model is essential given the increasing presence of EVs in traffic, yet creating an easy-to-use and accurate analytical model remains challenging.
To address these gaps, this study develops and validates a Phase-Aware AI (PAAI) car-following model specifically for EVs. The proposed model enhances traditional physics-based frameworks with an AI component that recognizes and adapts to different driving phases, such as rapid acceleration and regenerative braking. Using real-world trajectory data from vehicles equipped with adaptive cruise control (ACC), we conduct comprehensive simulations to validate the model's performance. The numerical results demonstrate that the PAAI model significantly improves prediction accuracy over traditional car-following models, providing an effective tool for accurately representing EV behavior in traffic simulations. 

---
# CustomIR: Unsupervised Fine-Tuning of Dense Embeddings for Known Document Corpora 

**Authors**: Nathan Paull  

**Link**: [PDF](https://arxiv.org/pdf/2510.21729)  

**Abstract**: Dense embedding models have become critical for modern information retrieval, particularly in RAG pipelines, but their performance often degrades when applied to specialized corpora outside their pre-training distribution. To address thi we introduce \textbf{CustomIR}, a framework for unsupervised adaptation of pre-trained language embedding models to domain-specific corpora using synthetically generated query-document pairs. CustomIR leverages large language models (LLMs) to create diverse queries grounded in a known target corpus, paired with LLM-verified hard negatives, eliminating the need for costly human annotation. Experiments on enterprise email and messaging datasets show that CustomIR consistently improves retrieval effectiveness with small models gaining up to 2.3 points in Recall@10. This performance increase allows these small models to rival the performance of much larger alternatives, allowing for cheaper RAG deployments. These results highlight that targeted synthetic fine-tuning offers a scalable and cost-efficient strategy for increasing domain-specific performance. 

---
# Modeling Bias Evolution in Fashion Recommender Systems: A System Dynamics Approach 

**Authors**: Mahsa Goodarzi, M. Abdullah Canbaz  

**Link**: [PDF](https://arxiv.org/pdf/2510.21728)  

**Abstract**: Bias in recommender systems not only distorts user experience but also perpetuates and amplifies existing societal stereotypes, particularly in sectors like fashion e-commerce. This study employs a dynamic modeling approach to scrutinize the mechanisms of bias activation and reinforcement within Fashion Recommender Systems (FRS). By leveraging system dynamics modeling and experimental simulations, we dissect the temporal evolution of bias and its multifaceted impacts on system performance. Our analysis reveals that inductive biases exert a more substantial influence on system outcomes than user biases, suggesting critical areas for intervention. We demonstrate that while current debiasing strategies, including data rebalancing and algorithmic regularization, are effective to an extent, they require further enhancement to comprehensively mitigate biases. This research underscores the necessity for advancing these strategies and extending system boundaries to incorporate broader contextual factors such as user demographics and item diversity, aiming to foster inclusivity and fairness in FRS. The findings advocate for a proactive approach in recommender system design to counteract bias propagation and ensure equitable user experiences. 

---
# Your Dense Retriever is Secretly an Expeditious Reasoner 

**Authors**: Yichi Zhang, Jun Bai, Zhixin Cai, Shuhan Qin, Zhuofan Chen, Jinghua Guan, Wenge Rong  

**Link**: [PDF](https://arxiv.org/pdf/2510.21727)  

**Abstract**: Dense retrievers enhance retrieval by encoding queries and documents into continuous vectors, but they often struggle with reasoning-intensive queries. Although Large Language Models (LLMs) can reformulate queries to capture complex reasoning, applying them universally incurs significant computational cost. In this work, we propose Adaptive Query Reasoning (AdaQR), a hybrid query rewriting framework. Within this framework, a Reasoner Router dynamically directs each query to either fast dense reasoning or deep LLM reasoning. The dense reasoning is achieved by the Dense Reasoner, which performs LLM-style reasoning directly in the embedding space, enabling a controllable trade-off between efficiency and accuracy. Experiments on large-scale retrieval benchmarks BRIGHT show that AdaQR reduces reasoning cost by 28% while preserving-or even improving-retrieval performance by 7%. 

---
# AquaVLM: Improving Underwater Situation Awareness with Mobile Vision Language Models 

**Authors**: Beitong Tian, Lingzhi Zhao, Bo Chen, Haozhen Zheng, Jingcheng Yang, Mingyuan Wu, Deepak Vasisht, Klara Nahrstedt  

**Link**: [PDF](https://arxiv.org/pdf/2510.21722)  

**Abstract**: Underwater activities like scuba diving enable millions annually to explore marine environments for recreation and scientific research. Maintaining situational awareness and effective communication are essential for diver safety. Traditional underwater communication systems are often bulky and expensive, limiting their accessibility to divers of all levels. While recent systems leverage lightweight smartphones and support text messaging, the messages are predefined and thus restrict context-specific communication.
In this paper, we present AquaVLM, a tap-and-send underwater communication system that automatically generates context-aware messages and transmits them using ubiquitous smartphones. Our system features a mobile vision-language model (VLM) fine-tuned on an auto-generated underwater conversation dataset and employs a hierarchical message generation pipeline. We co-design the VLM and transmission, incorporating error-resilient fine-tuning to improve the system's robustness to transmission errors. We develop a VR simulator to enable users to experience AquaVLM in a realistic underwater environment and create a fully functional prototype on the iOS platform for real-world experiments. Both subjective and objective evaluations validate the effectiveness of AquaVLM and highlight its potential for personal underwater communication as well as broader mobile VLM applications. 

---
# GAMER PAT: Research as a Serious Game 

**Authors**: Kenji Saito, Rei Tadika  

**Link**: [PDF](https://arxiv.org/pdf/2510.21719)  

**Abstract**: As generative AI increasingly outperforms students in producing academic writing, a critical question arises: how can we preserve the motivation, creativity, and intellectual growth of novice researchers in an age of automated academic achievement? This paper introduces GAMER PAT (GAme MastER, Paper Authoring Tutor), a prompt-engineered AI chatbot that reframes research paper writing as a serious game. Through role-playing mechanics, users interact with a co-author NPC and anonymous reviewer NPCs, turning feedback into "missions" and advancing through a narrative-driven writing process.
Our study reports on 26+ gameplay chat logs, including both autoethnography and use by graduate students under supervision. Using qualitative log analysis with SCAT (Steps for Coding and Theorization), we identified an emergent four-phase scaffolding pattern: (1) question posing, (2) meta-perspective, (3) structuring, and (4) recursive reflection. These results suggest that GAMER PAT supports not only the structural development of research writing but also reflective and motivational aspects.
We present this work as a descriptive account of concept and process, not a causal evaluation. We also include a speculative outlook envisioning how humans may continue to cultivate curiosity and agency alongside AI-driven research. This arXiv version thus provides both a descriptive report of design and usage, and a forward-looking provocation for future empirical studies. 

---
# AI-Enhanced Operator Assistance for UNICOS Applications 

**Authors**: Bernard Tam, Jean-Charles Tournier, Fernando Varela Rodriguez  

**Link**: [PDF](https://arxiv.org/pdf/2510.21717)  

**Abstract**: This project explores the development of an AI-enhanced operator assistant for UNICOS, CERN's UNified Industrial Control System. While powerful, UNICOS presents a number of challenges, including the cognitive burden of decoding widgets, manual effort required for root cause analysis, and difficulties maintainers face in tracing datapoint elements (DPEs) across a complex codebase. In situations where timely responses are critical, these challenges can increase cognitive load and slow down diagnostics. To address these issues, a multi-agent system was designed and implemented. The solution is supported by a modular architecture comprising a UNICOS-side extension written in CTRL code, a Python-based multi-agent system deployed on a virtual machine, and a vector database storing both operator documentation and widget animation code. Preliminary evaluations suggest that the system is capable of decoding widgets, performing root cause analysis by leveraging live device data and documentation, and tracing DPEs across a complex codebase. Together, these capabilities reduce the manual workload of operators and maintainers, enhance situational awareness in operations, and accelerate responses to alarms and anomalies. Beyond these immediate gains, this work highlights the potential of introducing multi-modal reasoning and retrieval augmented generation (RAG) into the domain of industrial control. Ultimately, this work represents more than a proof of concept: it provides a basis for advancing intelligent operator interfaces at CERN. By combining modular design, extensibility, and practical AI integration, this project not only alleviates current operator pain points but also points toward broader opportunities for assistive AI in accelerator operations. 

---
# Beyond IVR Touch-Tones: Customer Intent Routing using LLMs 

**Authors**: Sergio Rojas-Galeano  

**Link**: [PDF](https://arxiv.org/pdf/2510.21715)  

**Abstract**: Widespread frustration with rigid touch-tone Interactive Voice Response (IVR) systems for customer service underscores the need for more direct and intuitive language interaction. While speech technologies are necessary, the key challenge lies in routing intents from user phrasings to IVR menu paths, a task where Large Language Models (LLMs) show strong potential. Progress, however, is limited by data scarcity, as real IVR structures and interactions are often proprietary. We present a novel LLM-based methodology to address this gap. Using three distinct models, we synthesized a realistic 23-node IVR structure, generated 920 user intents (230 base and 690 augmented), and performed the routing task. We evaluate two prompt designs: descriptive hierarchical menus and flattened path representations, across both base and augmented datasets. Results show that flattened paths consistently yield higher accuracy, reaching 89.13% on the base dataset compared to 81.30% with the descriptive format, while augmentation introduces linguistic noise that slightly reduces performance. Confusion matrix analysis further suggests that low-performing routes may reflect not only model limitations but also redundancies in menu design. Overall, our findings demonstrate proof-of-concept that LLMs can enable IVR routing through a smoother, more seamless user experience -- moving customer service one step ahead of touch-tone menus. 

---
# DecoupleSearch: Decouple Planning and Search via Hierarchical Reward Modeling 

**Authors**: Hao Sun, Zile Qiao, Bo Wang, Guoxin Chen, Yingyan Hou, Yong Jiang, Pengjun Xie, Fei Huang, Yan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.21712)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems have emerged as a pivotal methodology for enhancing Large Language Models (LLMs) through the dynamic integration of external knowledge. To further improve RAG's flexibility, Agentic RAG introduces autonomous agents into the workflow. However, Agentic RAG faces several challenges: (1) the success of each step depends on both high-quality planning and accurate search, (2) the lack of supervision for intermediate reasoning steps, and (3) the exponentially large candidate space for planning and searching. To address these challenges, we propose DecoupleSearch, a novel framework that decouples planning and search processes using dual value models, enabling independent optimization of plan reasoning and search grounding. Our approach constructs a reasoning tree, where each node represents planning and search steps. We leverage Monte Carlo Tree Search to assess the quality of each step. During inference, Hierarchical Beam Search iteratively refines planning and search candidates with dual value models. Extensive experiments across policy models of varying parameter sizes, demonstrate the effectiveness of our method. 

---
# A Feature Engineering Approach for Business Impact-Oriented Failure Detection in Distributed Instant Payment Systems 

**Authors**: Lorenzo Porcelli  

**Link**: [PDF](https://arxiv.org/pdf/2510.21710)  

**Abstract**: Instant payment infrastructures have stringent performance requirements, processing millions of transactions daily with zero-downtime expectations. Traditional monitoring approaches fail to bridge the gap between technical infrastructure metrics and business process visibility. We introduce a novel feature engineering approach based on processing times computed between consecutive ISO 20022 message exchanges, creating a compact representation of system state. By applying anomaly detection to these features, we enable early failure detection and localization, allowing incident classification. Experimental evaluation on the TARGET Instant Payment Settlement (TIPS) system, using both real-world incidents and controlled simulations, demonstrates the approach's effectiveness in detecting diverse anomaly patterns and provides inherently interpretable explanations that enable operators to understand the business impact. By mapping features to distinct processing phases, the resulting framework differentiates between internal and external payment system issues, significantly reduces investigation time, and bridges observability gaps in distributed systems where transaction state is fragmented across multiple entities. 

---
# BugPilot: Complex Bug Generation for Efficient Learning of SWE Skills 

**Authors**: Atharv Sonwane, Isadora White, Hyunji Lee, Matheus Pereira, Lucas Caccia, Minseon Kim, Zhengyan Shi, Chinmay Singh, Alessandro Sordoni, Marc-Alexandre Côté, Xingdi Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2510.19898)  

**Abstract**: High quality bugs are key to training the next generation of language model based software engineering (SWE) agents. We introduce a novel method for synthetic generation of difficult and diverse bugs. Our method instructs SWE Agents to introduce a feature into the codebase whereby they may unintentionally break tests, resulting in bugs. Prior approaches often induce an out-of-distribution effect by generating bugs intentionally (e.g. by introducing local perturbation to existing code), which does not reflect realistic development processes. We perform qualitative analysis to demonstrate that our approach for generating bugs more closely reflects the patterns found in human-authored edits. Through extensive experiments, we demonstrate that our bugs provide more efficient training data for supervised fine-tuning, outperforming other bug datasets by 2% with half the training data (1.2k vs. 3k bugs). We train on our newly generated bugs in addition to existing bug datasets to get FrogBoss a state-of-the-art 32B parameter model on SWE-bench Verified with a pass@1 of 54.6% and FrogMini a state-of-the-art 14B model on SWE-bench Verified with a pass@1 of 45.3% on SWE-bench Verified averaged over three seeds. 

---
# An AI enhanced approach to the tree unimodality conjecture 

**Authors**: Eric Ramos, Sunny Sun  

**Link**: [PDF](https://arxiv.org/pdf/2510.18826)  

**Abstract**: Given a graph $G$, its independence sequence is the integral sequence $a_1,a_2,...,a_n$, where $a_i$ is the number of independent sets of vertices of size i. In the late 80's Alavi, Erdos, Malde, Schwenk showed that this sequence need not be unimodal for general graphs, but conjectured that it is always unimodal whenever $G$ is a tree. This conjecture was then naturally generalized to claim that the independence sequence of trees should be log concave, in the sense that $a_i^2$ is always above $a_{i-1}a_{i+1}$. This conjecture stood for many years, until in 2023, Kadrawi, Levit, Yosef, and Mizrachi proved that there were exactly two trees on 26 vertices whose independence sequence was not log concave. In this paper, we use the AI architecture PatternBoost, developed by Charton, Ellenberg, Wagner, and Williamson to train a machine to find counter-examples to the log-concavity conjecture. We will discuss the successes of this approach - finding tens of thousands of new counter-examples to log-concavity with vertex set sizes varying from 27 to 101 - and some of its fascinating failures. 

---
