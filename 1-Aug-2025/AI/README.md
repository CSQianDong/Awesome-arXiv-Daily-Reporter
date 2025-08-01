# SimuRA: Towards General Goal-Oriented Agent via Simulative Reasoning Architecture with LLM-Based World Model 

**Authors**: Mingkai Deng, Jinyu Hou, Yilin Shen, Hongxia Jin, Graham Neubig, Zhiting Hu, Eric Xing  

**Link**: [PDF](https://arxiv.org/pdf/2507.23773)  

**Abstract**: AI agents built on large language models (LLMs) hold enormous promise, but current practice focuses on a one-task-one-agent approach, which not only falls short of scalability and generality, but also suffers from the fundamental limitations of autoregressive LLMs. On the other hand, humans are general agents who reason by mentally simulating the outcomes of their actions and plans. Moving towards a more general and powerful AI agent, we introduce SimuRA, a goal-oriented architecture for generalized agentic reasoning. Based on a principled formulation of optimal agent in any environment, \modelname overcomes the limitations of autoregressive reasoning by introducing a world model for planning via simulation. The generalized world model is implemented using LLM, which can flexibly plan in a wide range of environments using the concept-rich latent space of natural language. Experiments on difficult web browsing tasks show that \modelname improves the success of flight search from 0\% to 32.2\%. World-model-based planning, in particular, shows consistent advantage of up to 124\% over autoregressive planning, demonstrating the advantage of world model simulation as a reasoning paradigm. We are excited about the possibility for training a single, general agent model based on LLMs that can act superintelligently in all environments. To start, we make SimuRA, a web-browsing agent built on \modelname with pretrained LLMs, available as a research demo for public testing. 

---
# CoT-Self-Instruct: Building high-quality synthetic prompts for reasoning and non-reasoning tasks 

**Authors**: Ping Yu, Jack Lanchantin, Tianlu Wang, Weizhe Yuan, Olga Golovneva, Ilia Kulikov, Sainbayar Sukhbaatar, Jason Weston, Jing Xu  

**Link**: [PDF](https://arxiv.org/pdf/2507.23751)  

**Abstract**: We propose CoT-Self-Instruct, a synthetic data generation method that instructs LLMs to first reason and plan via Chain-of-Thought (CoT) based on the given seed tasks, and then to generate a new synthetic prompt of similar quality and complexity for use in LLM training, followed by filtering for high-quality data with automatic metrics. In verifiable reasoning, our synthetic data significantly outperforms existing training datasets, such as s1k and OpenMathReasoning, across MATH500, AMC23, AIME24 and GPQA-Diamond. For non-verifiable instruction-following tasks, our method surpasses the performance of human or standard self-instruct prompts on both AlpacaEval 2.0 and Arena-Hard. 

---
# Seed-Prover: Deep and Broad Reasoning for Automated Theorem Proving 

**Authors**: Luoxin Chen, Jinming Gu, Liankai Huang, Wenhao Huang, Zhicheng Jiang, Allan Jie, Xiaoran Jin, Xing Jin, Chenggang Li, Kaijing Ma, Cheng Ren, Jiawei Shen, Wenlei Shi, Tong Sun, He Sun, Jiahui Wang, Siran Wang, Zhihong Wang, Chenrui Wei, Shufa Wei, Yonghui Wu, Yuchen Wu, Yihang Xia, Huajian Xin, Fan Yang, Huaiyuan Ying, Hongyi Yuan, Zheng Yuan, Tianyang Zhan, Chi Zhang, Yue Zhang, Ge Zhang, Tianyun Zhao, Jianqiu Zhao, Yichi Zhou, Thomas Hanwen Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.23726)  

**Abstract**: LLMs have demonstrated strong mathematical reasoning abilities by leveraging reinforcement learning with long chain-of-thought, yet they continue to struggle with theorem proving due to the lack of clear supervision signals when solely using natural language. Dedicated domain-specific languages like Lean provide clear supervision via formal verification of proofs, enabling effective training through reinforcement learning. In this work, we propose \textbf{Seed-Prover}, a lemma-style whole-proof reasoning model. Seed-Prover can iteratively refine its proof based on Lean feedback, proved lemmas, and self-summarization. To solve IMO-level contest problems, we design three test-time inference strategies that enable both deep and broad reasoning. Seed-Prover proves $78.1\%$ of formalized past IMO problems, saturates MiniF2F, and achieves over 50\% on PutnamBench, outperforming the previous state-of-the-art by a large margin. To address the lack of geometry support in Lean, we introduce a geometry reasoning engine \textbf{Seed-Geometry}, which outperforms previous formal geometry engines. We use these two systems to participate in IMO 2025 and fully prove 5 out of 6 problems. This work represents a significant advancement in automated mathematical reasoning, demonstrating the effectiveness of formal verification with long chain-of-thought reasoning. 

---
# TextQuests: How Good are LLMs at Text-Based Video Games? 

**Authors**: Long Phan, Mantas Mazeika, Andy Zou, Dan Hendrycks  

**Link**: [PDF](https://arxiv.org/pdf/2507.23701)  

**Abstract**: Evaluating AI agents within complex, interactive environments that mirror real-world challenges is critical for understanding their practical capabilities. While existing agent benchmarks effectively assess skills like tool use or performance on structured tasks, they often do not fully capture an agent's ability to operate autonomously in exploratory environments that demand sustained, self-directed reasoning over a long and growing context. To spur the development of agents capable of more robust intrinsic reasoning over long horizons, we introduce TextQuests, a benchmark based on the Infocom suite of interactive fiction games. These text-based adventures, which can take human players over 30 hours and require hundreds of precise actions to solve, serve as an effective proxy for evaluating AI agents on focused, stateful tasks. The benchmark is specifically designed to assess an LLM agent's capacity for self-contained problem-solving by precluding the use of external tools, thereby focusing on intrinsic long-context reasoning capabilities in an exploratory environment characterized by the need for trial-and-error learning and sustained problem-solving within a single interactive session. We release TextQuests at this https URL. 

---
# Personalized Education with Ranking Alignment Recommendation 

**Authors**: Haipeng Liu, Yuxuan Liu, Ting Long  

**Link**: [PDF](https://arxiv.org/pdf/2507.23664)  

**Abstract**: Personalized question recommendation aims to guide individual students through questions to enhance their mastery of learning targets. Most previous methods model this task as a Markov Decision Process and use reinforcement learning to solve, but they struggle with efficient exploration, failing to identify the best questions for each student during training. To address this, we propose Ranking Alignment Recommendation (RAR), which incorporates collaborative ideas into the exploration mechanism, enabling more efficient exploration within limited training episodes. Experiments show that RAR effectively improves recommendation performance, and our framework can be applied to any RL-based question recommender. Our code is available in this https URL. 

---
# MemoCue: Empowering LLM-Based Agents for Human Memory Recall via Strategy-Guided Querying 

**Authors**: Qian Zhao, Zhuo Sun, Bin Guo, Zhiwen Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.23633)  

**Abstract**: Agent-assisted memory recall is one critical research problem in the field of human-computer interaction. In conventional methods, the agent can retrieve information from its equipped memory module to help the person recall incomplete or vague memories. The limited size of memory module hinders the acquisition of complete memories and impacts the memory recall performance in practice. Memory theories suggest that the person's relevant memory can be proactively activated through some effective cues. Inspired by this, we propose a novel strategy-guided agent-assisted memory recall method, allowing the agent to transform an original query into a cue-rich one via the judiciously designed strategy to help the person recall memories. To this end, there are two key challenges. (1) How to choose the appropriate recall strategy for diverse forgetting scenarios with distinct memory-recall characteristics? (2) How to obtain the high-quality responses leveraging recall strategies, given only abstract and sparsely annotated strategy patterns? To address the challenges, we propose a Recall Router framework. Specifically, we design a 5W Recall Map to classify memory queries into five typical scenarios and define fifteen recall strategy patterns across the corresponding scenarios. We then propose a hierarchical recall tree combined with the Monte Carlo Tree Search algorithm to optimize the selection of strategy and the generation of strategy responses. We construct an instruction tuning dataset and fine-tune multiple open-source large language models (LLMs) to develop MemoCue, an agent that excels in providing memory-inspired responses. Experiments on three representative datasets show that MemoCue surpasses LLM-based methods by 17.74% in recall inspiration. Further human evaluation highlights its advantages in memory-recall applications. 

---
# Semantic Chain-of-Trust: Autonomous Trust Orchestration for Collaborator Selection via Hypergraph-Aided Agentic AI 

**Authors**: Botao Zhu, Xianbin Wang, Dusit Niyato  

**Link**: [PDF](https://arxiv.org/pdf/2507.23565)  

**Abstract**: In collaborative systems, the effective completion of tasks hinges on task-specific trust evaluations of potential devices for distributed collaboration. However, the complexity of tasks, the spatiotemporal dynamism of distributed device resources, and the inevitable assessment overhead dramatically increase the complexity and resource consumption of the trust evaluation process. As a result, ill-timed or overly frequent trust evaluations can reduce utilization rate of constrained resources, negatively affecting collaborative task execution. To address this challenge, this paper proposes an autonomous trust orchestration method based on a new concept of semantic chain-of-trust. Our technique employs agentic AI and hypergraph to establish and maintain trust relationships among devices. By leveraging its strengths in autonomous perception, task decomposition, and semantic reasoning, we propose agentic AI to perceive device states and autonomously perform trust evaluations of collaborators based on historical performance data only during device idle periods, thereby enabling efficient utilization of distributed resources. In addition, agentic AI performs task-specific trust evaluations on collaborator resources by analyzing the alignment between resource capabilities and task requirements. Moreover, by maintaining a trust hypergraph embedded with trust semantics for each device, agentic AI enables hierarchical management of collaborators and identifies collaborators requiring trust evaluation based on trust semantics, thereby achieving a balance between overhead and trust accuracy. Furthermore, local trust hypergraphs from multiple devices can be chained together to support multi-hop collaboration, enabling efficient coordination in large-scale systems. Experimental results demonstrate that the proposed method achieves resource-efficient trust evaluation. 

---
# DICE: Dynamic In-Context Example Selection in LLM Agents via Efficient Knowledge Transfer 

**Authors**: Ruoyu Wang, Junda Wu, Yu Xia, Tong Yu, Ryan A. Rossi, Julian McAuley, Lina Yao  

**Link**: [PDF](https://arxiv.org/pdf/2507.23554)  

**Abstract**: Large language model-based agents, empowered by in-context learning (ICL), have demonstrated strong capabilities in complex reasoning and tool-use tasks. However, existing works have shown that the effectiveness of ICL is highly sensitive to the choice of demonstrations, with suboptimal examples often leading to unstable or degraded performance. While prior work has explored example selection, including in some agentic or multi-step settings, existing approaches typically rely on heuristics or task-specific designs and lack a general, theoretically grounded criterion for what constitutes an effective demonstration across reasoning steps. Therefore, it is non-trivial to develop a principled, general-purpose method for selecting demonstrations that consistently benefit agent performance. In this paper, we address this challenge with DICE, Dynamic In-Context Example Selection for LLM Agents, a theoretically grounded ICL framework for agentic tasks that selects the most relevant demonstrations at each step of reasoning. Our approach decomposes demonstration knowledge into transferable and non-transferable components through a causal lens, showing how the latter can introduce spurious dependencies that impair generalization. We further propose a stepwise selection criterion with a formal guarantee of improved agent performance. Importantly, DICE is a general, framework-agnostic solution that can be integrated as a plug-in module into existing agentic frameworks without any additional training cost. Extensive experiments across diverse domains demonstrate our method's effectiveness and generality, highlighting the importance of principled, context-aware demo selection for robust and efficient LLM agents. 

---
# Causal Identification of Sufficient, Contrastive and Complete Feature Sets in Image Classification 

**Authors**: David A Kelly, Hana Chockler  

**Link**: [PDF](https://arxiv.org/pdf/2507.23497)  

**Abstract**: Existing algorithms for explaining the outputs of image classifiers are based on a variety of approaches and produce explanations that lack formal rigor. On the other hand, logic-based explanations are formally and rigorously defined but their computability relies on strict assumptions about the model that do not hold on image classifiers.
In this paper, we show that causal explanations, in addition to being formally and rigorously defined, enjoy the same formal properties as logic-based ones, while still lending themselves to black-box algorithms and being a natural fit for image classifiers. We prove formal properties of causal explanations and introduce contrastive causal explanations for image classifiers. Moreover, we augment the definition of explanation with confidence awareness and introduce complete causal explanations: explanations that are classified with exactly the same confidence as the original image.
We implement our definitions, and our experimental results demonstrate that different models have different patterns of sufficiency, contrastiveness, and completeness. Our algorithms are efficiently computable, taking on average 6s per image on a ResNet50 model to compute all types of explanations, and are totally black-box, needing no knowledge of the model, no access to model internals, no access to gradient, nor requiring any properties, such as monotonicity, of the model. 

---
# Causal Reasoning in Pieces: Modular In-Context Learning for Causal Discovery 

**Authors**: Kacper Kadziolka, Saber Salehkaleybar  

**Link**: [PDF](https://arxiv.org/pdf/2507.23488)  

**Abstract**: Causal inference remains a fundamental challenge for large language models. Recent advances in internal reasoning with large language models have sparked interest in whether state-of-the-art reasoning models can robustly perform causal discovery-a task where conventional models often suffer from severe overfitting and near-random performance under data perturbations. We study causal discovery on the Corr2Cause benchmark using the emergent OpenAI's o-series and DeepSeek-R model families and find that these reasoning-first architectures achieve significantly greater native gains than prior approaches. To capitalize on these strengths, we introduce a modular in-context pipeline inspired by the Tree-of-Thoughts and Chain-of-Thoughts methodologies, yielding nearly three-fold improvements over conventional baselines. We further probe the pipeline's impact by analyzing reasoning chain length, complexity, and conducting qualitative and quantitative comparisons between conventional and reasoning models. Our findings suggest that while advanced reasoning models represent a substantial leap forward, carefully structured in-context frameworks are essential to maximize their capabilities and offer a generalizable blueprint for causal discovery across diverse domains. 

---
# Self-Foveate: Enhancing Diversity and Difficulty of Synthesized Instructions from Unsupervised Text via Multi-Level Foveation 

**Authors**: Mingzhe Li, Xin Lu, Yanyan Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2507.23440)  

**Abstract**: Large language models (LLMs) with instruction following capabilities have demonstrated impressive problem-solving abilities. While synthesizing instructional data from unsupervised text has become a common approach for training such models, conventional methods rely heavily on human effort for data annotation. Although existing automated synthesis paradigms have alleviated this constraint, they still exhibit significant limitations in ensuring adequate diversity and difficulty of synthesized instructions. To address these challenges, we propose Self-Foveate, an innovative LLM-driven method for instruction synthesis. This approach introduces a "Micro-Scatter-Macro" multi-level foveation methodology that effectively guides the LLM to deeply excavate fine-grained information embedded in unsupervised text, thereby enhancing both the diversity and difficulty of synthesized instructions. Comprehensive experiments across multiple unsupervised corpora and diverse model architectures validate the effectiveness and superiority of our proposed method. We publicly release our data and codes: this https URL 

---
# Chatting with your ERP: A Recipe 

**Authors**: Jorge Ruiz Gómez, Lidia Andrés Susinos, Jorge Alamo Olivé, Sonia Rey Osorno, Manuel Luis Gonzalez Hernández  

**Link**: [PDF](https://arxiv.org/pdf/2507.23429)  

**Abstract**: This paper presents the design, implementation, and evaluation behind a Large Language Model (LLM) agent that chats with an industrial production-grade ERP system. The agent is capable of interpreting natural language queries and translating them into executable SQL statements, leveraging open-weight LLMs. A novel dual-agent architecture combining reasoning and critique stages was proposed to improve query generation reliability. 

---
# LLM4Rail: An LLM-Augmented Railway Service Consulting Platform 

**Authors**: Zhuo Li, Xianghuai Deng, Chiwei Feng, Hanmeng Li, Shenjie Wang, Haichao Zhang, Teng Jia, Conlin Chen, Louis Linchun Wu, Jia Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.23377)  

**Abstract**: Large language models (LLMs) have significantly reshaped different walks of business. To meet the increasing demands for individualized railway service, we develop LLM4Rail - a novel LLM-augmented railway service consulting platform. Empowered by LLM, LLM4Rail can provide custom modules for ticketing, railway food & drink recommendations, weather information, and chitchat. In LLM4Rail, we propose the iterative "Question-Thought-Action-Observation (QTAO)" prompting framework. It meticulously integrates verbal reasoning with task-oriented actions, that is, reasoning to guide action selection, to effectively retrieve external observations relevant to railway operation and service to generate accurate responses. To provide personalized onboard dining services, we first construct the Chinese Railway Food and Drink (CRFD-25) - a publicly accessible takeout dataset tailored for railway services. CRFD-25 covers a wide range of signature dishes categorized by cities, cuisines, age groups, and spiciness levels. We further introduce an LLM-based zero-shot conversational recommender for railway catering. To address the unconstrained nature of open recommendations, the feature similarity-based post-processing step is introduced to ensure all the recommended items are aligned with CRFD-25 dataset. 

---
# DSBC : Data Science task Benchmarking with Context engineering 

**Authors**: Ram Mohan Rao Kadiyala, Siddhant Gupta, Jebish Purbey, Giulio Martini, Suman Debnath, Hamza Farooq  

**Link**: [PDF](https://arxiv.org/pdf/2507.23336)  

**Abstract**: Recent advances in large language models (LLMs) have significantly impacted data science workflows, giving rise to specialized data science agents designed to automate analytical tasks. Despite rapid adoption, systematic benchmarks evaluating the efficacy and limitations of these agents remain scarce. In this paper, we introduce a comprehensive benchmark specifically crafted to reflect real-world user interactions with data science agents by observing usage of our commercial applications. We evaluate three LLMs: Claude-4.0-Sonnet, Gemini-2.5-Flash, and OpenAI-o4-Mini across three approaches: zero-shot with context engineering, multi-step with context engineering, and with SmolAgent. Our benchmark assesses performance across a diverse set of eight data science task categories, additionally exploring the sensitivity of models to common prompting issues, such as data leakage and slightly ambiguous instructions. We further investigate the influence of temperature parameters on overall and task-specific outcomes for each model and approach. Our findings reveal distinct performance disparities among the evaluated models and methodologies, highlighting critical factors that affect practical deployment. The benchmark dataset and evaluation framework introduced herein aim to provide a foundation for future research of more robust and effective data science agents. 

---
# AI Must not be Fully Autonomous 

**Authors**: Tosin Adewumi, Lama Alkhaled, Florent Imbert, Hui Han, Nudrat Habib, Karl Löwenmark  

**Link**: [PDF](https://arxiv.org/pdf/2507.23330)  

**Abstract**: Autonomous Artificial Intelligence (AI) has many benefits. It also has many risks. In this work, we identify the 3 levels of autonomous AI. We are of the position that AI must not be fully autonomous because of the many risks, especially as artificial superintelligence (ASI) is speculated to be just decades away. Fully autonomous AI, which can develop its own objectives, is at level 3 and without responsible human oversight. However, responsible human oversight is crucial for mitigating the risks. To ague for our position, we discuss theories of autonomy, AI and agents. Then, we offer 12 distinct arguments and 6 counterarguments with rebuttals to the counterarguments. We also present 15 pieces of recent evidence of AI misaligned values and other risks in the appendix. 

---
# How Far Are AI Scientists from Changing the World? 

**Authors**: Qiujie Xie, Yixuan Weng, Minjun Zhu, Fuchen Shen, Shulin Huang, Zhen Lin, Jiahui Zhou, Zilan Mao, Zijie Yang, Linyi Yang, Jian Wu, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.23276)  

**Abstract**: The emergence of large language models (LLMs) is propelling automated scientific discovery to the next level, with LLM-based Artificial Intelligence (AI) Scientist systems now taking the lead in scientific research. Several influential works have already appeared in the field of AI Scientist systems, with AI-generated research papers having been accepted at the ICLR 2025 workshop, suggesting that a human-level AI Scientist capable of uncovering phenomena previously unknown to humans, may soon become a reality. In this survey, we focus on the central question: How far are AI scientists from changing the world and reshaping the scientific research paradigm? To answer this question, we provide a prospect-driven review that comprehensively analyzes the current achievements of AI Scientist systems, identifying key bottlenecks and the critical components required for the emergence of a scientific agent capable of producing ground-breaking discoveries that solve grand challenges. We hope this survey will contribute to a clearer understanding of limitations of current AI Scientist systems, showing where we are, what is missing, and what the ultimate goals for scientific AI should be. 

---
# Solution-aware vs global ReLU selection: partial MILP strikes back for DNN verification 

**Authors**: Yuke Liao, Blaise Genest, Kuldeep Meel, Shaan Aryaman  

**Link**: [PDF](https://arxiv.org/pdf/2507.23197)  

**Abstract**: To handle complex instances, we revisit a divide-and-conquer approach to break down the complexity: instead of few complex BaB calls, we rely on many small {\em partial} MILP calls. The crucial step is to select very few but very important ReLUs to treat using (costly) binary variables. The previous attempts were suboptimal in that respect. To select these important ReLU variables, we propose a novel {\em solution-aware} ReLU scoring ({\sf SAS}), as well as adapt the BaB-SR and BaB-FSB branching functions as {\em global} ReLU scoring ({\sf GS}) functions. We compare them theoretically as well as experimentally, and {\sf SAS} is more efficient at selecting a set of variables to open using binary variables. Compared with previous attempts, SAS reduces the number of binary variables by around 6 times, while maintaining the same level of accuracy. Implemented in {\em Hybrid MILP}, calling first $\alpha,\beta$-CROWN with a short time-out to solve easier instances, and then partial MILP, produces a very accurate yet efficient verifier, reducing by up to $40\%$ the number of undecided instances to low levels ($8-15\%$), while keeping a reasonable runtime ($46s-417s$ on average per instance), even for fairly large CNNs with 2 million parameters. 

---
# Tractable Responsibility Measures for Ontology-Mediated Query Answering 

**Authors**: Meghyn Bienvenu, Diego Figueira, Pierre Lafourcade  

**Link**: [PDF](https://arxiv.org/pdf/2507.23191)  

**Abstract**: Recent work on quantitative approaches to explaining query answers employs responsibility measures to assign scores to facts in order to quantify their respective contributions to obtaining a given answer. In this paper, we study the complexity of computing such responsibility scores in the setting of ontology-mediated query answering, focusing on a very recently introduced family of Shapley-value-based responsibility measures defined in terms of weighted sums of minimal supports (WSMS). By exploiting results from the database setting, we can show that such measures enjoy polynomial data complexity for classes of ontology-mediated queries that are first-order-rewritable, whereas the problem becomes "shP"-hard when the ontology language can encode reachability queries (via axioms like $\exists R. A \sqsubseteq A$). To better understand the tractability frontier, we next explore the combined complexity of WSMS computation. We prove that intractability applies already to atomic queries if the ontology language supports conjunction, as well as to unions of `well-behaved' conjunctive queries, even in the absence of an ontology. By contrast, our study yields positive results for common DL-Lite dialects: by means of careful analysis, we identify classes of structurally restricted conjunctive queries (which intuitively disallow undesirable interactions between query atoms) that admit tractable WSMS computation. 

---
# Argumentatively Coherent Judgmental Forecasting 

**Authors**: Deniz Gorur, Antonio Rago, Francesca Toni  

**Link**: [PDF](https://arxiv.org/pdf/2507.23163)  

**Abstract**: Judgmental forecasting employs human opinions to make predictions about future events, rather than exclusively historical data as in quantitative forecasting. When these opinions form an argumentative structure around forecasts, it is useful to study the properties of the forecasts from an argumentative perspective. In this paper, we advocate and formally define a property of argumentative coherence, which, in essence, requires that a forecaster's reasoning is coherent with their forecast. We then conduct three evaluations with our notion of coherence. First, we assess the impact of enforcing coherence on human forecasters as well as on Large Language Model (LLM)-based forecasters, given that they have recently shown to be competitive with human forecasters. In both cases, we show that filtering out incoherent predictions improves forecasting accuracy consistently, supporting the practical value of coherence in both human and LLM-based forecasting. Then, via crowd-sourced user experiments, we show that, despite its apparent intuitiveness and usefulness, users do not generally align with this coherence property. This points to the need to integrate, within argumentation-based judgmental forecasting, mechanisms to filter out incoherent opinions before obtaining group forecasting predictions. 

---
# Moravec's Paradox: Towards an Auditory Turing Test 

**Authors**: David Noever, Forrest McKee  

**Link**: [PDF](https://arxiv.org/pdf/2507.23091)  

**Abstract**: This research work demonstrates that current AI systems fail catastrophically on auditory tasks that humans perform effortlessly. Drawing inspiration from Moravec's paradox (i.e., tasks simple for humans often prove difficult for machines, and vice versa), we introduce an auditory Turing test comprising 917 challenges across seven categories: overlapping speech, speech in noise, temporal distortion, spatial audio, coffee-shop noise, phone distortion, and perceptual illusions. Our evaluation of state-of-the-art audio models including GPT-4's audio capabilities and OpenAI's Whisper reveals a striking failure rate exceeding 93%, with even the best-performing model achieving only 6.9% accuracy on tasks that humans solved at 7.5 times higher success (52%). These results expose focusing failures in how AI systems process complex auditory scenes, particularly in selective attention, noise robustness, and contextual adaptation. Our benchmark not only quantifies the human-machine auditory gap but also provides insights into why these failures occur, suggesting that current architectures lack fundamental mechanisms for human-like auditory scene analysis. The traditional design of audio CAPTCHAs highlights common filters that humans evolved but machines fail to select in multimodal language models. This work establishes a diagnostic framework for measuring progress toward human-level machine listening and highlights the need for novel approaches integrating selective attention, physics-based audio understanding, and context-aware perception into multimodal AI systems. 

---
# FairReason: Balancing Reasoning and Social Bias in MLLMs 

**Authors**: Zhenyu Pan, Yutong Zhang, Jianshu Zhang, Haoran Lu, Haozheng Luo, Yuwei Han, Philip S. Yu, Manling Li, Han Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.23067)  

**Abstract**: Multimodal Large Language Models (MLLMs) already achieve state-of-the-art results across a wide range of tasks and modalities. To push their reasoning ability further, recent studies explore advanced prompting schemes and post-training fine-tuning. Although these techniques improve logical accuracy, they frequently leave the models' outputs burdened with pronounced social biases. Clarifying how reasoning gains interact with bias mitigation-and whether the two objectives inherently trade off-therefore remains an open and pressing research problem. Our study begins by benchmarking three bias-mitigation strategies-supervised fine-uning (SFT), knowledge distillation (KD), and rule-based reinforcement learning (RL)-under identical conditions, establishing their baseline strengths and weaknesses. Building on these results, we vary the proportion of debias-focused and reasoning-centric samples within each paradigm to chart the reasoning-versus-bias trade-off. Our sweeps reveal a consistent sweet spot: a roughly 1:4 mix trained with reinforcement learning cuts stereotype scores by 10% while retaining 88% of the model's original reasoning accuracy, offering concrete guidance for balancing fairness and capability in MLLMs. 

---
# Data Readiness for Scientific AI at Scale 

**Authors**: Wesley Brewer, Patrick Widener, Valentine Anantharaj, Feiyi Wang, Tom Beck, Arjun Shankar, Sarp Oral  

**Link**: [PDF](https://arxiv.org/pdf/2507.23018)  

**Abstract**: This paper examines how Data Readiness for AI (DRAI) principles apply to leadership-scale scientific datasets used to train foundation models. We analyze archetypal workflows across four representative domains - climate, nuclear fusion, bio/health, and materials - to identify common preprocessing patterns and domain-specific constraints. We introduce a two-dimensional readiness framework composed of Data Readiness Levels (raw to AI-ready) and Data Processing Stages (ingest to shard), both tailored to high performance computing (HPC) environments. This framework outlines key challenges in transforming scientific data for scalable AI training, emphasizing transformer-based generative models. Together, these dimensions form a conceptual maturity matrix that characterizes scientific data readiness and guides infrastructure development toward standardized, cross-domain support for scalable and reproducible AI for science. 

---
# Unifying Post-hoc Explanations of Knowledge Graph Completions 

**Authors**: Alessandro Lonardi, Samy Badreddine, Tarek R. Besold, Pablo Sanchez Martin  

**Link**: [PDF](https://arxiv.org/pdf/2507.22951)  

**Abstract**: Post-hoc explainability for Knowledge Graph Completion (KGC) lacks formalization and consistent evaluations, hindering reproducibility and cross-study comparisons. This paper argues for a unified approach to post-hoc explainability in KGC. First, we propose a general framework to characterize post-hoc explanations via multi-objective optimization, balancing their effectiveness and conciseness. This unifies existing post-hoc explainability algorithms in KGC and the explanations they produce. Next, we suggest and empirically support improved evaluation protocols using popular metrics like Mean Reciprocal Rank and Hits@$k$. Finally, we stress the importance of interpretability as the ability of explanations to address queries meaningful to end-users. By unifying methods and refining evaluation standards, this work aims to make research in KGC explainability more reproducible and impactful. 

---
# SUB: Benchmarking CBM Generalization via Synthetic Attribute Substitutions 

**Authors**: Jessica Bader, Leander Girrbach, Stephan Alaniz, Zeynep Akata  

**Link**: [PDF](https://arxiv.org/pdf/2507.23784)  

**Abstract**: Concept Bottleneck Models (CBMs) and other concept-based interpretable models show great promise for making AI applications more transparent, which is essential in fields like medicine. Despite their success, we demonstrate that CBMs struggle to reliably identify the correct concepts under distribution shifts. To assess the robustness of CBMs to concept variations, we introduce SUB: a fine-grained image and concept benchmark containing 38,400 synthetic images based on the CUB dataset. To create SUB, we select a CUB subset of 33 bird classes and 45 concepts to generate images which substitute a specific concept, such as wing color or belly pattern. We introduce a novel Tied Diffusion Guidance (TDG) method to precisely control generated images, where noise sharing for two parallel denoising processes ensures that both the correct bird class and the correct attribute are generated. This novel benchmark enables rigorous evaluation of CBMs and similar interpretable models, contributing to the development of more robust methods. Our code is available at this https URL and the dataset at this http URL. 

---
# Phi-Ground Tech Report: Advancing Perception in GUI Grounding 

**Authors**: Miaosen Zhang, Ziqiang Xu, Jialiang Zhu, Qi Dai, Kai Qiu, Yifan Yang, Chong Luo, Tianyi Chen, Justin Wagle, Tim Franklin, Baining Guo  

**Link**: [PDF](https://arxiv.org/pdf/2507.23779)  

**Abstract**: With the development of multimodal reasoning models, Computer Use Agents (CUAs), akin to Jarvis from \textit{"Iron Man"}, are becoming a reality. GUI grounding is a core component for CUAs to execute actual actions, similar to mechanical control in robotics, and it directly leads to the success or failure of the system. It determines actions such as clicking and typing, as well as related parameters like the coordinates for clicks. Current end-to-end grounding models still achieve less than 65\% accuracy on challenging benchmarks like ScreenSpot-pro and UI-Vision, indicating they are far from being ready for deployment. % , as a single misclick can result in unacceptable consequences. In this work, we conduct an empirical study on the training of grounding models, examining details from data collection to model training. Ultimately, we developed the \textbf{Phi-Ground} model family, which achieves state-of-the-art performance across all five grounding benchmarks for models under $10B$ parameters in agent settings. In the end-to-end model setting, our model still achieves SOTA results with scores of \textit{\textbf{43.2}} on ScreenSpot-pro and \textit{\textbf{27.2}} on UI-Vision. We believe that the various details discussed in this paper, along with our successes and failures, not only clarify the construction of grounding models but also benefit other perception tasks. Project homepage: \href{this https URL}{this https URL} 

---
# Consensus-Driven Active Model Selection 

**Authors**: Justin Kay, Grant Van Horn, Subhransu Maji, Daniel Sheldon, Sara Beery  

**Link**: [PDF](https://arxiv.org/pdf/2507.23771)  

**Abstract**: The widespread availability of off-the-shelf machine learning models poses a challenge: which model, of the many available candidates, should be chosen for a given data analysis task? This question of model selection is traditionally answered by collecting and annotating a validation dataset -- a costly and time-intensive process. We propose a method for active model selection, using predictions from candidate models to prioritize the labeling of test data points that efficiently differentiate the best candidate. Our method, CODA, performs consensus-driven active model selection by modeling relationships between classifiers, categories, and data points within a probabilistic framework. The framework uses the consensus and disagreement between models in the candidate pool to guide the label acquisition process, and Bayesian inference to update beliefs about which model is best as more information is collected. We validate our approach by curating a collection of 26 benchmark tasks capturing a range of model selection scenarios. CODA outperforms existing methods for active model selection significantly, reducing the annotation effort required to discover the best model by upwards of 70% compared to the previous state-of-the-art. Code and data are available at this https URL. 

---
# Rule2Text: Natural Language Explanation of Logical Rules in Knowledge Graphs 

**Authors**: Nasim Shirvani-Mahdavi, Devin Wingfield, Amin Ghasemi, Chengkai Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.23740)  

**Abstract**: Knowledge graphs (KGs) often contain sufficient information to support the inference of new facts. Identifying logical rules not only improves the completeness of a knowledge graph but also enables the detection of potential errors, reveals subtle data patterns, and enhances the overall capacity for reasoning and interpretation. However, the complexity of such rules, combined with the unique labeling conventions of each KG, can make them difficult for humans to understand. In this paper, we explore the potential of large language models to generate natural language explanations for logical rules. Specifically, we extract logical rules using the AMIE 3.5.1 rule discovery algorithm from the benchmark dataset FB15k-237 and two large-scale datasets, FB-CVT-REV and FB+CVT-REV. We examine various prompting strategies, including zero- and few-shot prompting, including variable entity types, and chain-of-thought reasoning. We conduct a comprehensive human evaluation of the generated explanations based on correctness, clarity, and hallucination, and also assess the use of large language models as automatic judges. Our results demonstrate promising performance in terms of explanation correctness and clarity, although several challenges remain for future research. All scripts and data used in this study are publicly available at this https URL}{this https URL. 

---
# Distributed AI Agents for Cognitive Underwater Robot Autonomy 

**Authors**: Markus Buchholz, Ignacio Carlucho, Michele Grimaldi, Yvan R. Petillot  

**Link**: [PDF](https://arxiv.org/pdf/2507.23735)  

**Abstract**: Achieving robust cognitive autonomy in robots navigating complex, unpredictable environments remains a fundamental challenge in robotics. This paper presents Underwater Robot Self-Organizing Autonomy (UROSA), a groundbreaking architecture leveraging distributed Large Language Model AI agents integrated within the Robot Operating System 2 (ROS 2) framework to enable advanced cognitive capabilities in Autonomous Underwater Vehicles. UROSA decentralises cognition into specialised AI agents responsible for multimodal perception, adaptive reasoning, dynamic mission planning, and real-time decision-making. Central innovations include flexible agents dynamically adapting their roles, retrieval-augmented generation utilising vector databases for efficient knowledge management, reinforcement learning-driven behavioural optimisation, and autonomous on-the-fly ROS 2 node generation for runtime functional extensibility. Extensive empirical validation demonstrates UROSA's promising adaptability and reliability through realistic underwater missions in simulation and real-world deployments, showing significant advantages over traditional rule-based architectures in handling unforeseen scenarios, environmental uncertainties, and novel mission objectives. This work not only advances underwater autonomy but also establishes a scalable, safe, and versatile cognitive robotics framework capable of generalising to a diverse array of real-world applications. 

---
# Enhanced Velocity Field Modeling for Gaussian Video Reconstruction 

**Authors**: Zhenyang Li, Xiaoyang Bai, Tongchen Zhang, Pengfei Shen, Weiwei Xu, Yifan Peng  

**Link**: [PDF](https://arxiv.org/pdf/2507.23704)  

**Abstract**: High-fidelity 3D video reconstruction is essential for enabling real-time rendering of dynamic scenes with realistic motion in virtual and augmented reality (VR/AR). The deformation field paradigm of 3D Gaussian splatting has achieved near-photorealistic results in video reconstruction due to the great representation capability of deep deformation networks. However, in videos with complex motion and significant scale variations, deformation networks often overfit to irregular Gaussian trajectories, leading to suboptimal visual quality. Moreover, the gradient-based densification strategy designed for static scene reconstruction proves inadequate to address the absence of dynamic content. In light of these challenges, we propose a flow-empowered velocity field modeling scheme tailored for Gaussian video reconstruction, dubbed FlowGaussian-VR. It consists of two core components: a velocity field rendering (VFR) pipeline which enables optical flow-based optimization, and a flow-assisted adaptive densification (FAD) strategy that adjusts the number and size of Gaussians in dynamic regions. We validate our model's effectiveness on multi-view dynamic reconstruction and novel view synthesis with multiple real-world datasets containing challenging motion scenarios, demonstrating not only notable visual improvements (over 2.5 dB gain in PSNR) and less blurry artifacts in dynamic textures, but also regularized and trackable per-Gaussian trajectories. 

---
# Scalable Multi-Task Reinforcement Learning for Generalizable Spatial Intelligence in Visuomotor Agents 

**Authors**: Shaofei Cai, Zhancun Mu, Haiwen Xia, Bowei Zhang, Anji Liu, Yitao Liang  

**Link**: [PDF](https://arxiv.org/pdf/2507.23698)  

**Abstract**: While Reinforcement Learning (RL) has achieved remarkable success in language modeling, its triumph hasn't yet fully translated to visuomotor agents. A primary challenge in RL models is their tendency to overfit specific tasks or environments, thereby hindering the acquisition of generalizable behaviors across diverse settings. This paper provides a preliminary answer to this challenge by demonstrating that RL-finetuned visuomotor agents in Minecraft can achieve zero-shot generalization to unseen worlds. Specifically, we explore RL's potential to enhance generalizable spatial reasoning and interaction capabilities in 3D worlds. To address challenges in multi-task RL representation, we analyze and establish cross-view goal specification as a unified multi-task goal space for visuomotor policies. Furthermore, to overcome the significant bottleneck of manual task design, we propose automated task synthesis within the highly customizable Minecraft environment for large-scale multi-task RL training, and we construct an efficient distributed RL framework to support this. Experimental results show RL significantly boosts interaction success rates by $4\times$ and enables zero-shot generalization of spatial reasoning across diverse environments, including real-world settings. Our findings underscore the immense potential of RL training in 3D simulated environments, especially those amenable to large-scale task generation, for significantly advancing visuomotor agents' spatial reasoning. 

---
# A survey of multi-agent geosimulation methodologies: from ABM to LLM 

**Authors**: Virginia Padilla, Jacinto Dávila  

**Link**: [PDF](https://arxiv.org/pdf/2507.23694)  

**Abstract**: We provide a comprehensive examination of agent-based approaches that codify the principles and linkages underlying multi-agent systems, simulations, and information systems. Based on two decades of study, this paper confirms a framework intended as a formal specification for geosimulation platforms. Our findings show that large language models (LLMs) can be effectively incorporated as agent components if they follow a structured architecture specific to fundamental agent activities such as perception, memory, planning, and action. This integration is precisely consistent with the architecture that we formalize, providing a solid platform for next-generation geosimulation systems. 

---
# villa-X: Enhancing Latent Action Modeling in Vision-Language-Action Models 

**Authors**: Xiaoyu Chen, Hangxing Wei, Pushi Zhang, Chuheng Zhang, Kaixin Wang, Yanjiang Guo, Rushuai Yang, Yucen Wang, Xinquan Xiao, Li Zhao, Jianyu Chen, Jiang Bian  

**Link**: [PDF](https://arxiv.org/pdf/2507.23682)  

**Abstract**: Visual-Language-Action (VLA) models have emerged as a popular paradigm for learning robot manipulation policies that can follow language instructions and generalize to novel scenarios. Recent work has begun to explore the incorporation of latent actions, an abstract representation of visual change between two frames, into VLA pre-training. In this paper, we introduce villa-X, a novel Visual-Language-Latent-Action (ViLLA) framework that advances latent action modeling for learning generalizable robot manipulation policies. Our approach improves both how latent actions are learned and how they are incorporated into VLA pre-training. Together, these contributions enable villa-X to achieve superior performance across simulated environments including SIMPLER and LIBERO, as well as on two real-world robot setups including gripper and dexterous hand manipulation. We believe the ViLLA paradigm holds significant promise, and that our villa-X provides a strong foundation for future research. 

---
# Automating AI Failure Tracking: Semantic Association of Reports in AI Incident Database 

**Authors**: Diego Russo, Gian Marco Orlando, Valerio La Gatta, Vincenzo Moscato  

**Link**: [PDF](https://arxiv.org/pdf/2507.23669)  

**Abstract**: Artificial Intelligence (AI) systems are transforming critical sectors such as healthcare, finance, and transportation, enhancing operational efficiency and decision-making processes. However, their deployment in high-stakes domains has exposed vulnerabilities that can result in significant societal harm. To systematically study and mitigate these risk, initiatives like the AI Incident Database (AIID) have emerged, cataloging over 3,000 real-world AI failure reports. Currently, associating a new report with the appropriate AI Incident relies on manual expert intervention, limiting scalability and delaying the identification of emerging failure patterns.
To address this limitation, we propose a retrieval-based framework that automates the association of new reports with existing AI Incidents through semantic similarity modeling. We formalize the task as a ranking problem, where each report-comprising a title and a full textual description-is compared to previously documented AI Incidents based on embedding cosine similarity. Benchmarking traditional lexical methods, cross-encoder architectures, and transformer-based sentence embedding models, we find that the latter consistently achieve superior performance. Our analysis further shows that combining titles and descriptions yields substantial improvements in ranking accuracy compared to using titles alone. Moreover, retrieval performance remains stable across variations in description length, highlighting the robustness of the framework. Finally, we find that retrieval performance consistently improves as the training set expands. Our approach provides a scalable and efficient solution for supporting the maintenance of the AIID. 

---
# Efficient Masked Attention Transformer for Few-Shot Classification and Segmentation 

**Authors**: Dustin Carrión-Ojeda, Stefan Roth, Simone Schaub-Meyer  

**Link**: [PDF](https://arxiv.org/pdf/2507.23642)  

**Abstract**: Few-shot classification and segmentation (FS-CS) focuses on jointly performing multi-label classification and multi-class segmentation using few annotated examples. Although the current state of the art (SOTA) achieves high accuracy in both tasks, it struggles with small objects. To overcome this, we propose the Efficient Masked Attention Transformer (EMAT), which improves classification and segmentation accuracy, especially for small objects. EMAT introduces three modifications: a novel memory-efficient masked attention mechanism, a learnable downscaling strategy, and parameter-efficiency enhancements. EMAT outperforms all FS-CS methods on the PASCAL-5$^i$ and COCO-20$^i$ datasets, using at least four times fewer trainable parameters. Moreover, as the current FS-CS evaluation setting discards available annotations, despite their costly collection, we introduce two novel evaluation settings that consider these annotations to better reflect practical scenarios. 

---
# OptiGradTrust: Byzantine-Robust Federated Learning with Multi-Feature Gradient Analysis and Reinforcement Learning-Based Trust Weighting 

**Authors**: Mohammad Karami, Fatemeh Ghassemi, Hamed Kebriaei, Hamid Azadegan  

**Link**: [PDF](https://arxiv.org/pdf/2507.23638)  

**Abstract**: Federated Learning (FL) enables collaborative model training across distributed medical institutions while preserving patient privacy, but remains vulnerable to Byzantine attacks and statistical heterogeneity. We present OptiGradTrust, a comprehensive defense framework that evaluates gradient updates through a novel six-dimensional fingerprint including VAE reconstruction error, cosine similarity metrics, $L_2$ norm, sign-consistency ratio, and Monte Carlo Shapley value, which drive a hybrid RL-attention module for adaptive trust scoring. To address convergence challenges under data heterogeneity, we develop FedBN-Prox (FedBN-P), combining Federated Batch Normalization with proximal regularization for optimal accuracy-convergence trade-offs. Extensive evaluation across MNIST, CIFAR-10, and Alzheimer's MRI datasets under various Byzantine attack scenarios demonstrates significant improvements over state-of-the-art defenses, achieving up to +1.6 percentage points over FLGuard under non-IID conditions while maintaining robust performance against diverse attack patterns through our adaptive learning approach. 

---
# L-GTA: Latent Generative Modeling for Time Series Augmentation 

**Authors**: Luis Roque, Carlos Soares, Vitor Cerqueira, Luis Torgo  

**Link**: [PDF](https://arxiv.org/pdf/2507.23615)  

**Abstract**: Data augmentation is gaining importance across various aspects of time series analysis, from forecasting to classification and anomaly detection tasks. We introduce the Latent Generative Transformer Augmentation (L-GTA) model, a generative approach using a transformer-based variational recurrent autoencoder. This model uses controlled transformations within the latent space of the model to generate new time series that preserve the intrinsic properties of the original dataset. L-GTA enables the application of diverse transformations, ranging from simple jittering to magnitude warping, and combining these basic transformations to generate more complex synthetic time series datasets. Our evaluation of several real-world datasets demonstrates the ability of L-GTA to produce more reliable, consistent, and controllable augmented data. This translates into significant improvements in predictive accuracy and similarity measures compared to direct transformation methods. 

---
# LLM-Based Identification of Infostealer Infection Vectors from Screenshots: The Case of Aurora 

**Authors**: Estelle Ruellan, Eric Clay, Nicholas Ascoli  

**Link**: [PDF](https://arxiv.org/pdf/2507.23611)  

**Abstract**: Infostealers exfiltrate credentials, session cookies, and sensitive data from infected systems. With over 29 million stealer logs reported in 2024, manual analysis and mitigation at scale are virtually unfeasible/unpractical. While most research focuses on proactive malware detection, a significant gap remains in leveraging reactive analysis of stealer logs and their associated artifacts. Specifically, infection artifacts such as screenshots, image captured at the point of compromise, are largely overlooked by the current literature. This paper introduces a novel approach leveraging Large Language Models (LLMs), more specifically gpt-4o-mini, to analyze infection screenshots to extract potential Indicators of Compromise (IoCs), map infection vectors, and track campaigns. Focusing on the Aurora infostealer, we demonstrate how LLMs can process screenshots to identify infection vectors, such as malicious URLs, installer files, and exploited software themes. Our method extracted 337 actionable URLs and 246 relevant files from 1000 screenshots, revealing key malware distribution methods and social engineering tactics. By correlating extracted filenames, URLs, and infection themes, we identified three distinct malware campaigns, demonstrating the potential of LLM-driven analysis for uncovering infection workflows and enhancing threat intelligence. By shifting malware analysis from traditional log-based detection methods to a reactive, artifact-driven approach that leverages infection screenshots, this research presents a scalable method for identifying infection vectors and enabling early intervention. 

---
# Deep Learning-based Prediction of Clinical Trial Enrollment with Uncertainty Estimates 

**Authors**: Tien Huu Do, Antoine Masquelier, Nae Eoun Lee, Jonathan Crowther  

**Link**: [PDF](https://arxiv.org/pdf/2507.23607)  

**Abstract**: Clinical trials are a systematic endeavor to assess the safety and efficacy of new drugs or treatments. Conducting such trials typically demands significant financial investment and meticulous planning, highlighting the need for accurate predictions of trial outcomes. Accurately predicting patient enrollment, a key factor in trial success, is one of the primary challenges during the planning phase. In this work, we propose a novel deep learning-based method to address this critical challenge. Our method, implemented as a neural network model, leverages pre-trained language models (PLMs) to capture the complexities and nuances of clinical documents, transforming them into expressive representations. These representations are then combined with encoded tabular features via an attention mechanism. To account for uncertainties in enrollment prediction, we enhance the model with a probabilistic layer based on the Gamma distribution, which enables range estimation. We apply the proposed model to predict clinical trial duration, assuming site-level enrollment follows a Poisson-Gamma process. We carry out extensive experiments on real-world clinical trial data, and show that the proposed method can effectively predict the number of patients enrolled at a number of sites for a given clinical trial, outperforming established baseline models. 

---
# Can LLM-Reasoning Models Replace Classical Planning? A Benchmark Study 

**Authors**: Kai Goebel, Patrik Zips  

**Link**: [PDF](https://arxiv.org/pdf/2507.23589)  

**Abstract**: Recent advancements in Large Language Models have sparked interest in their potential for robotic task planning. While these models demonstrate strong generative capabilities, their effectiveness in producing structured and executable plans remains uncertain. This paper presents a systematic evaluation of a broad spectrum of current state of the art language models, each directly prompted using Planning Domain Definition Language domain and problem files, and compares their planning performance with the Fast Downward planner across a variety of benchmarks. In addition to measuring success rates, we assess how faithfully the generated plans translate into sequences of actions that can actually be executed, identifying both strengths and limitations of using these models in this setting. Our findings show that while the models perform well on simpler planning tasks, they continue to struggle with more complex scenarios that require precise resource management, consistent state tracking, and strict constraint compliance. These results underscore fundamental challenges in applying language models to robotic planning in real world environments. By outlining the gaps that emerge during execution, we aim to guide future research toward combined approaches that integrate language models with classical planners in order to enhance the reliability and scalability of planning in autonomous robotics. 

---
# ART: Adaptive Relation Tuning for Generalized Relation Prediction 

**Authors**: Gopika Sudhakaran, Hikaru Shindo, Patrick Schramowski, Simone Schaub-Meyer, Kristian Kersting, Stefan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2507.23543)  

**Abstract**: Visual relation detection (VRD) is the task of identifying the relationships between objects in a scene. VRD models trained solely on relation detection data struggle to generalize beyond the relations on which they are trained. While prompt tuning has been used to adapt vision-language models (VLMs) for VRD, it uses handcrafted prompts and struggles with novel or complex relations. We argue that instruction tuning offers a more effective solution by fine-tuning VLMs on diverse instructional data. We thus introduce ART, an Adaptive Relation Tuning framework that adapts VLMs for VRD through instruction tuning and strategic instance selection. By converting VRD datasets into an instruction tuning format and employing an adaptive sampling algorithm, ART directs the VLM to focus on informative relations while maintaining generalizability. Specifically, we focus on the relation classification, where subject-object boxes are given and the model predicts the predicate between them. We tune on a held-in set and evaluate across multiple held-out datasets of varying complexity. Our approach strongly improves over its baselines and can infer unseen relation concepts, a capability absent in mainstream VRD methods. We demonstrate ART's practical value by using the predicted relations for segmenting complex scenes. 

---
# A Unified Perception-Language-Action Framework for Adaptive Autonomous Driving 

**Authors**: Yi Zhang, Erik Leo Haß, Kuo-Yi Chao, Nenad Petrovic, Yinglei Song, Chengdong Wu, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2507.23540)  

**Abstract**: Autonomous driving systems face significant challenges in achieving human-like adaptability, robustness, and interpretability in complex, open-world environments. These challenges stem from fragmented architectures, limited generalization to novel scenarios, and insufficient semantic extraction from perception. To address these limitations, we propose a unified Perception-Language-Action (PLA) framework that integrates multi-sensor fusion (cameras, LiDAR, radar) with a large language model (LLM)-augmented Vision-Language-Action (VLA) architecture, specifically a GPT-4.1-powered reasoning core. This framework unifies low-level sensory processing with high-level contextual reasoning, tightly coupling perception with natural language-based semantic understanding and decision-making to enable context-aware, explainable, and safety-bounded autonomous driving. Evaluations on an urban intersection scenario with a construction zone demonstrate superior performance in trajectory tracking, speed prediction, and adaptive planning. The results highlight the potential of language-augmented cognitive frameworks for advancing the safety, interpretability, and scalability of autonomous driving systems. 

---
# From LLMs to Edge: Parameter-Efficient Fine-Tuning on Edge Devices 

**Authors**: Georg Slamanig, Francesco Corti, Olga Saukh  

**Link**: [PDF](https://arxiv.org/pdf/2507.23536)  

**Abstract**: Parameter-efficient fine-tuning (PEFT) methods reduce the computational costs of updating deep learning models by minimizing the number of additional parameters used to adapt a model to a down- stream task. While extensively researched in large language models (LLMs), their application to smaller models used on edge devices, such as convolutional neural networks, remains underexplored. This paper benchmarks and analyzes popular PEFT methods on convolutional architectures typically deployed in resource-constrained edge environments. We evaluate LoRA, DoRA, and GaLore for updating standard and depthwise convolutional architectures to handle distribution shifts and accommodate unseen classes. We utilize recently proposed PyTorch profilers to compare the updated model performance and computational costs of these PEFT methods with traditional fine-tuning approaches. With resource efficiency in mind, we investigate their update behavior across different rank dimensions. We find that the evaluated PEFT methods are only half as memory-efficient when applied to depthwise-separable convolution architectures, compared to their efficiency with LLMs. Conversely, when targeting convolu- tional architectures optimized for edge deployment, adapter-based PEFT methods can reduce floating point operations (FLOPs) during model updates by up to 95%. These insights offer valuable guidance for selecting PEFT methods based on hardware constraints, performance requirements, and application needs. Our code is online. 

---
# Transparent AI: The Case for Interpretability and Explainability 

**Authors**: Dhanesh Ramachandram, Himanshu Joshi, Judy Zhu, Dhari Gandhi, Lucas Hartman, Ananya Raval  

**Link**: [PDF](https://arxiv.org/pdf/2507.23535)  

**Abstract**: As artificial intelligence systems increasingly inform high-stakes decisions across sectors, transparency has become foundational to responsible and trustworthy AI implementation. Leveraging our role as a leading institute in advancing AI research and enabling industry adoption, we present key insights and lessons learned from practical interpretability applications across diverse domains. This paper offers actionable strategies and implementation guidance tailored to organizations at varying stages of AI maturity, emphasizing the integration of interpretability as a core design principle rather than a retrospective add-on. 

---
# MECAT: A Multi-Experts Constructed Benchmark for Fine-Grained Audio Understanding Tasks 

**Authors**: Yadong Niu, Tianzi Wang, Heinrich Dinkel, Xingwei Sun, Jiahao Zhou, Gang Li, Jizhong Liu, Xunying Liu, Junbo Zhang, Jian Luan  

**Link**: [PDF](https://arxiv.org/pdf/2507.23511)  

**Abstract**: While large audio-language models have advanced open-ended audio understanding, they still fall short of nuanced human-level comprehension. This gap persists largely because current benchmarks, limited by data annotations and evaluation metrics, fail to reliably distinguish between generic and highly detailed model outputs. To this end, this work introduces MECAT, a Multi-Expert Constructed Benchmark for Fine-Grained Audio Understanding Tasks. Generated via a pipeline that integrates analysis from specialized expert models with Chain-of-Thought large language model reasoning, MECAT provides multi-perspective, fine-grained captions and open-set question-answering pairs. The benchmark is complemented by a novel metric: DATE (Discriminative-Enhanced Audio Text Evaluation). This metric penalizes generic terms and rewards detailed descriptions by combining single-sample semantic similarity with cross-sample discriminability. A comprehensive evaluation of state-of-the-art audio models is also presented, providing new insights into their current capabilities and limitations. The data and code are available at this https URL 

---
# I Am Big, You Are Little; I Am Right, You Are Wrong 

**Authors**: David A. Kelly, Akchunya Chanchal, Nathan Blake  

**Link**: [PDF](https://arxiv.org/pdf/2507.23509)  

**Abstract**: Machine learning for image classification is an active and rapidly developing field. With the proliferation of classifiers of different sizes and different architectures, the problem of choosing the right model becomes more and more important.
While we can assess a model's classification accuracy statistically, our understanding of the way these models work is unfortunately limited. In order to gain insight into the decision-making process of different vision models, we propose using minimal sufficient pixels sets to gauge a model's `concentration': the pixels that capture the essence of an image through the lens of the model. By comparing position, overlap, and size of sets of pixels, we identify that different architectures have statistically different concentration, in both size and position. In particular, ConvNext and EVA models differ markedly from the others. We also identify that images which are misclassified are associated with larger pixels sets than correct classifications. 

---
# Digital literacy interventions can boost humans in discerning deepfakes 

**Authors**: Dominique Geissler, Claire Robertson, Stefan Feuerriegel  

**Link**: [PDF](https://arxiv.org/pdf/2507.23492)  

**Abstract**: Deepfakes, i.e., images generated by artificial intelligence (AI), can erode trust in institutions and compromise election outcomes, as people often struggle to discern real images from deepfakes. Improving digital literacy can help address these challenges, yet scalable and effective approaches remain largely unexplored. Here, we compare the efficacy of five digital literacy interventions to boost people's ability to discern deepfakes: (1) textual guidance on common indicators of deepfakes; (2) visual demonstrations of these indicators; (3) a gamified exercise for identifying deepfakes; (4) implicit learning through repeated exposure and feedback; and (5) explanations of how deepfakes are generated with the help of AI. We conducted an experiment with N=1,200 participants from the United States to test the immediate and long-term effectiveness of our interventions. Our results show that our interventions can boost deepfake discernment by up to 13 percentage points while maintaining trust in real images. Altogether, our approach is scalable, suitable for diverse populations, and highly effective for boosting deepfake detection while maintaining trust in truthful information. 

---
# Automated Feedback on Student-Generated UML and ER Diagrams Using Large Language Models 

**Authors**: Sebastian Gürtl, Gloria Schimetta, David Kerschbaumer, Michael Liut, Alexander Steinmaurer  

**Link**: [PDF](https://arxiv.org/pdf/2507.23470)  

**Abstract**: UML and ER diagrams are foundational in computer science education but come with challenges for learners due to the need for abstract thinking, contextual understanding, and mastery of both syntax and semantics. These complexities are difficult to address through traditional teaching methods, which often struggle to provide scalable, personalized feedback, especially in large classes. We introduce DUET (Diagrammatic UML & ER Tutor), a prototype of an LLM-based tool, which converts a reference diagram and a student-submitted diagram into a textual representation and provides structured feedback based on the differences. It uses a multi-stage LLM pipeline to compare diagrams and generate reflective feedback. Furthermore, the tool enables analytical insights for educators, aiming to foster self-directed learning and inform instructional strategies. We evaluated DUET through semi-structured interviews with six participants, including two educators and four teaching assistants. They identified strengths such as accessibility, scalability, and learning support alongside limitations, including reliability and potential misuse. Participants also suggested potential improvements, such as bulk upload functionality and interactive clarification features. DUET presents a promising direction for integrating LLMs into modeling education and offers a foundation for future classroom integration and empirical evaluation. 

---
# Role-Aware Language Models for Secure and Contextualized Access Control in Organizations 

**Authors**: Saeed Almheiri, Yerulan Kongrat, Adrian Santosh, Ruslan Tasmukhanov, Josemaria Vera, Muhammad Dehan Al Kautsar, Fajri Koto  

**Link**: [PDF](https://arxiv.org/pdf/2507.23465)  

**Abstract**: As large language models (LLMs) are increasingly deployed in enterprise settings, controlling model behavior based on user roles becomes an essential requirement. Existing safety methods typically assume uniform access and focus on preventing harmful or toxic outputs, without addressing role-specific access constraints. In this work, we investigate whether LLMs can be fine-tuned to generate responses that reflect the access privileges associated with different organizational roles. We explore three modeling strategies: a BERT-based classifier, an LLM-based classifier, and role-conditioned generation. To evaluate these approaches, we construct two complementary datasets. The first is adapted from existing instruction-tuning corpora through clustering and role labeling, while the second is synthetically generated to reflect realistic, role-sensitive enterprise scenarios. We assess model performance across varying organizational structures and analyze robustness to prompt injection, role mismatch, and jailbreak attempts. 

---
# Mitigating Resolution-Drift in Federated Learning: Case of Keypoint Detection 

**Authors**: Taeheon Lim, Joohyung Lee, Kyungjae Lee, Jungchan Cho  

**Link**: [PDF](https://arxiv.org/pdf/2507.23461)  

**Abstract**: The Federated Learning (FL) approach enables effective learning across distributed systems, while preserving user data privacy. To date, research has primarily focused on addressing statistical heterogeneity and communication efficiency, through which FL has achieved success in classification tasks. However, its application to non-classification tasks, such as human pose estimation, remains underexplored. This paper identifies and investigates a critical issue termed ``resolution-drift,'' where performance degrades significantly due to resolution variability across clients. Unlike class-level heterogeneity, resolution drift highlights the importance of resolution as another axis of not independent or identically distributed (non-IID) data. To address this issue, we present resolution-adaptive federated learning (RAF), a method that leverages heatmap-based knowledge distillation. Through multi-resolution knowledge distillation between higher-resolution outputs (teachers) and lower-resolution outputs (students), our approach enhances resolution robustness without overfitting. Extensive experiments and theoretical analysis demonstrate that RAF not only effectively mitigates resolution drift and achieves significant performance improvements, but also can be integrated seamlessly into existing FL frameworks. Furthermore, although this paper focuses on human pose estimation, our t-SNE analysis reveals distinct characteristics between classification and high-resolution representation tasks, supporting the generalizability of RAF to other tasks that rely on preserving spatial detail. 

---
# KLAN: Kuaishou Landing-page Adaptive Navigator 

**Authors**: Fan Li, Chang Meng, Jiaqi Fu, Shuchang Liu, Jiashuo Zhang, Tianke Zhang, Xueliang Wang, Xiaoqiang Feng  

**Link**: [PDF](https://arxiv.org/pdf/2507.23459)  

**Abstract**: Modern online platforms configure multiple pages to accommodate diverse user needs. This multi-page architecture inherently establishes a two-stage interaction paradigm between the user and the platform: (1) Stage I: page navigation, navigating users to a specific page and (2) Stage II: in-page interaction, where users engage with customized content within the specific page. While the majority of research has been focusing on the sequential recommendation task that improves users' feedback in Stage II, there has been little investigation on how to achieve better page navigation in Stage I. To fill this gap, we formally define the task of Personalized Landing Page Modeling (PLPM) into the field of recommender systems: Given a user upon app entry, the goal of PLPM is to proactively select the most suitable landing page from a set of candidates (e.g., functional tabs, content channels, or aggregation pages) to optimize the short-term PDR metric and the long-term user engagement and satisfaction metrics, while adhering to industrial constraints. Additionally, we propose KLAN (Kuaishou Landing-page Adaptive Navigator), a hierarchical solution framework designed to provide personalized landing pages under the formulation of PLPM. KLAN comprises three key components: (1) KLAN-ISP captures inter-day static page preference; (2) KLAN-IIT captures intra-day dynamic interest transitions and (3) KLAN-AM adaptively integrates both components for optimal navigation decisions. Extensive online experiments conducted on the Kuaishou platform demonstrate the effectiveness of KLAN, obtaining +0.205% and +0.192% improvements on in Daily Active Users (DAU) and user Lifetime (LT). Our KLAN is ultimately deployed on the online platform at full traffic, serving hundreds of millions of users. To promote further research in this important area, we will release our dataset and code upon paper acceptance. 

---
# Machine learning and machine learned prediction in chest X-ray images 

**Authors**: Shereiff Garrett, Abhinav Adhikari, Sarina Gautam, DaShawn Marquis Morris, Chandra Mani Adhikari  

**Link**: [PDF](https://arxiv.org/pdf/2507.23455)  

**Abstract**: Machine learning and artificial intelligence are fast-growing fields of research in which data is used to train algorithms, learn patterns, and make predictions. This approach helps to solve seemingly intricate problems with significant accuracy without explicit programming by recognizing complex relationships in data. Taking an example of 5824 chest X-ray images, we implement two machine learning algorithms, namely, a baseline convolutional neural network (CNN) and a DenseNet-121, and present our analysis in making machine-learned predictions in predicting patients with ailments. Both baseline CNN and DenseNet-121 perform very well in the binary classification problem presented in this work. Gradient-weighted class activation mapping shows that DenseNet-121 correctly focuses on essential parts of the input chest X-ray images in its decision-making more than the baseline CNN. 

---
# AGA: An adaptive group alignment framework for structured medical cross-modal representation learning 

**Authors**: Wei Li, Xun Gong, Jiao Li, Xiaobin Sun  

**Link**: [PDF](https://arxiv.org/pdf/2507.23402)  

**Abstract**: Learning medical visual representations from paired images and reports is a promising direction in representation learning. However, current vision-language pretraining methods in the medical domain often simplify clinical reports into single entities or fragmented tokens, ignoring their inherent structure. In addition, contrastive learning frameworks typically depend on large quantities of hard negative samples, which is impractical for small-scale medical datasets. To tackle these challenges, we propose Adaptive Grouped Alignment (AGA), a new framework that captures structured semantics from paired medical images and reports. AGA introduces a bidirectional grouping mechanism based on a sparse similarity matrix. For each image-report pair, we compute fine-grained similarities between text tokens and image patches. Each token selects its top-matching patches to form a visual group, and each patch selects its most related tokens to form a language group. To enable adaptive grouping, we design two threshold gating modules, called Language Grouped Threshold Gate and Vision Grouped Threshold Gate, which learn grouping thresholds dynamically. Group representations are computed as weighted averages based on similarity scores. To align each token with its group representation, we introduce an Instance Aware Group Alignment loss that operates within each image-text pair, removing the need for external negatives. Finally, a Bidirectional Cross-modal Grouped Alignment module is applied to enhance fine-grained alignment between visual and linguistic group representations. Extensive experiments on public and private datasets show that our method achieves strong performance on image-text retrieval and classification tasks under both fine-tuning and zero-shot settings. 

---
# Causal2Vec: Improving Decoder-only LLMs as Versatile Embedding Models 

**Authors**: Ailiang Lin, Zhuoyun Li, Kotaro Funakoshi  

**Link**: [PDF](https://arxiv.org/pdf/2507.23386)  

**Abstract**: Decoder-only large language models (LLMs) are increasingly used to build embedding models that effectively encode the semantic information of natural language texts into dense vector representations for various embedding tasks. However, many existing methods primarily focus on removing the causal attention mask in LLMs to enable bidirectional attention, potentially undermining the model's ability to extract semantic information acquired during pretraining. Additionally, leading unidirectional approaches often rely on extra input text to overcome the inherent limitations of causal attention, inevitably increasing computational costs. In this work, we propose Causal2Vec, a general-purpose embedding model tailored to enhance the performance of decoder-only LLMs without altering their original architectures or introducing significant computational overhead. Specifically, we first employ a lightweight BERT-style model to pre-encode the input text into a single Contextual token, which is then prepended to the LLM's input sequence, allowing each token to capture contextualized information even without attending to future tokens. Furthermore, to mitigate the recency bias introduced by last-token pooling and help LLMs better leverage the semantic information encoded in the Contextual token, we concatenate the last hidden states of Contextual and EOS tokens as the final text embedding. In practice, Causal2Vec achieves state-of-the-art performance on the Massive Text Embeddings Benchmark (MTEB) among models trained solely on publicly available retrieval datasets, while reducing the required sequence length by up to 85% and inference time by up to 82% compared to best-performing methods. 

---
# MPCC: A Novel Benchmark for Multimodal Planning with Complex Constraints in Multimodal Large Language Models 

**Authors**: Yiyan Ji, Haoran Chen, Qiguang Chen, Chengyue Wu, Libo Qin, Wanxiang Che  

**Link**: [PDF](https://arxiv.org/pdf/2507.23382)  

**Abstract**: Multimodal planning capabilities refer to the ability to predict, reason, and design steps for task execution with multimodal context, which is essential for complex reasoning and decision-making across multiple steps. However, current benchmarks face two key challenges: (1) they cannot directly assess multimodal real-world planning capabilities, and (2) they lack constraints or implicit constraints across modalities. To address these issues, we introduce Multimodal Planning with Complex Constraints (MPCC), the first benchmark to systematically evaluate MLLMs' ability to handle multimodal constraints in planning. To address the first challenge, MPCC focuses on three real-world tasks: Flight Planning, Calendar Planning, and Meeting Planning. To solve the second challenge, we introduce complex constraints (e.g. budget, temporal, and spatial) in these tasks, with graded difficulty levels (EASY, MEDIUM, HARD) to separate constraint complexity from search space expansion. Experiments on 13 advanced MLLMs reveal significant challenges: closed-source models achieve only 21.3% feasible plans, while open-source models average below 11%. Additionally, we observe that MLLMs are highly sensitive to constraint complexity and that traditional multimodal prompting strategies fail in multi-constraint scenarios. Our work formalizes multimodal constraints in planning, provides a rigorous evaluation framework, and highlights the need for advancements in constraint-aware reasoning for real-world MLLM applications. 

---
# Trae Agent: An LLM-based Agent for Software Engineering with Test-time Scaling 

**Authors**: Trae Research Team, Pengfei Gao, Zhao Tian, Xiangxin Meng, Xinchen Wang, Ruida Hu, Yuanan Xiao, Yizhou Liu, Zhao Zhang, Junjie Chen, Cuiyun Gao, Yun Lin, Yingfei Xiong, Chao Peng, Xia Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.23370)  

**Abstract**: Software issue resolution is a critical challenge in software engineering and has garnered increasing attention in recent years. With the rapid advancement of large language models (LLMs), substantial progress has been made in addressing real-world software engineering tasks. Recent studies have introduced ensemble reasoning techniques to enhance the performance of LLM-based issue resolution. However, existing prompting-based methods still face limitations in effectively exploring large ensemble spaces and lack the capacity for repository-level understanding, both of which constrain their overall effectiveness. In this paper, we propose Trae Agent, the first agent-based ensemble reasoning approach for repository-level issue resolution. Trae Agent formulates our goal as an optimal solution search problem and addresses two key challenges, i.e., large ensemble spaces and repository-level understanding, through modular agents for generation, pruning, and selection. We conduct extensive experiments using three leading LLMs on the widely-adopted SWE-bench benchmark, comparing Trae Agent against four state-of-the-art ensemble reasoning techniques. Experimental results demonstrate that Trae Agent consistently achieves superior performance, with an average improvement of 10.22% over all baselines in terms of Pass@1. Trae Agent has achieved first place on the SWE-bench Verified leaderboard, with a notable Pass@1 score of 75.20%. We are pleased to release Trae Agent as an open-source project to support the research community, with all resources available at this https URL. 

---
# "I made this (sort of)": Negotiating authorship, confronting fraudulence, and exploring new musical spaces with prompt-based AI music generation 

**Authors**: Bob L. T. Sturm  

**Link**: [PDF](https://arxiv.org/pdf/2507.23365)  

**Abstract**: I reflect on my experience creating two music albums centered on state-of-the-art prompt-based AI music generation platforms. The first album explicitly poses the question: What happens when I collide my junk mail with these platforms? The second album is a direct response to the first, and toys with the inability of state-of-the-art prompt-based AI music generation platforms to generate music that is not ``practiced'', ``polished'', and ``produced''. I seed a large language model (LLM) with information about these albums and have it interview me, which results in the exploration of several deeper questions: To what extent am I the author? Where am I in the resulting music? How is my musical identity changing as I am faced with machines that are in some ways far more talented than I? What new musical spaces does my work open, for me or anyone/thing else? I conclude by reflecting on my reflections, as well as LLM-mediated self-reflection as method. 

---
# Text-to-SQL Task-oriented Dialogue Ontology Construction 

**Authors**: Renato Vukovic, Carel van Niekerk, Michael Heck, Benjamin Ruppik, Hsien-Chin Lin, Shutong Feng, Nurul Lubis, Milica Gasic  

**Link**: [PDF](https://arxiv.org/pdf/2507.23358)  

**Abstract**: Large language models (LLMs) are widely used as general-purpose knowledge sources, but they rely on parametric knowledge, limiting explainability and trustworthiness. In task-oriented dialogue (TOD) systems, this separation is explicit, using an external database structured by an explicit ontology to ensure explainability and controllability. However, building such ontologies requires manual labels or supervised training. We introduce TeQoDO: a Text-to-SQL task-oriented Dialogue Ontology construction method. Here, an LLM autonomously builds a TOD ontology from scratch without supervision using its inherent SQL programming capabilities combined with dialogue theory provided in the prompt. We show that TeQoDO outperforms transfer learning approaches, and its constructed ontology is competitive on a downstream dialogue state tracking task. Ablation studies demonstrate the key role of dialogue theory. TeQoDO also scales to allow construction of much larger ontologies, which we investigate on a Wikipedia and ArXiv dataset. We view this as a step towards broader application of ontologies to increase LLM explainability. 

---
# Quality Evaluation of COBOL to Java Code Transformation 

**Authors**: Shmulik Froimovich, Raviv Gal, Wesam Ibraheem, Avi Ziv  

**Link**: [PDF](https://arxiv.org/pdf/2507.23356)  

**Abstract**: We present an automated evaluation system for assessing COBOL-to-Java code translation within IBM's watsonx Code Assistant for Z (WCA4Z). The system addresses key challenges in evaluating LLM-based translators, including model opacity and the complexity of translation quality assessment. Our approach combines analytic checkers with LLM-as-a-judge (LaaJ) techniques to deliver scalable, multi-faceted evaluations. The system supports continuous integration workflows, enables large-scale benchmarking, and reduces reliance on manual review. We describe the system architecture, evaluation strategies, and reporting mechanisms that provide actionable insights for developers and project managers, facilitating the evolution of high-quality, modernized codebases. 

---
# Multi-Waypoint Path Planning and Motion Control for Non-holonomic Mobile Robots in Agricultural Applications 

**Authors**: Mahmoud Ghorab, Matthias Lorenzen  

**Link**: [PDF](https://arxiv.org/pdf/2507.23350)  

**Abstract**: There is a growing demand for autonomous mobile robots capable of navigating unstructured agricultural environments. Tasks such as weed control in meadows require efficient path planning through an unordered set of coordinates while minimizing travel distance and adhering to curvature constraints to prevent soil damage and protect vegetation. This paper presents an integrated navigation framework combining a global path planner based on the Dubins Traveling Salesman Problem (DTSP) with a Nonlinear Model Predictive Control (NMPC) strategy for local path planning and control. The DTSP generates a minimum-length, curvature-constrained path that efficiently visits all targets, while the NMPC leverages this path to compute control signals to accurately reach each waypoint. The system's performance was validated through comparative simulation analysis on real-world field datasets, demonstrating that the coupled DTSP-based planner produced smoother and shorter paths, with a reduction of about 16% in the provided scenario, compared to decoupled methods. Based thereon, the NMPC controller effectively steered the robot to the desired waypoints, while locally optimizing the trajectory and ensuring adherence to constraints. These findings demonstrate the potential of the proposed framework for efficient autonomous navigation in agricultural environments. 

---
# MUST-RAG: MUSical Text Question Answering with Retrieval Augmented Generation 

**Authors**: Daeyong Kwon, SeungHeon Doh, Juhan Nam  

**Link**: [PDF](https://arxiv.org/pdf/2507.23334)  

**Abstract**: Recent advancements in Large language models (LLMs) have demonstrated remarkable capabilities across diverse domains. While they exhibit strong zero-shot performance on various tasks, LLMs' effectiveness in music-related applications remains limited due to the relatively small proportion of music-specific knowledge in their training data. To address this limitation, we propose MusT-RAG, a comprehensive framework based on Retrieval Augmented Generation (RAG) to adapt general-purpose LLMs for text-only music question answering (MQA) tasks. RAG is a technique that provides external knowledge to LLMs by retrieving relevant context information when generating answers to questions. To optimize RAG for the music domain, we (1) propose MusWikiDB, a music-specialized vector database for the retrieval stage, and (2) utilizes context information during both inference and fine-tuning processes to effectively transform general-purpose LLMs into music-specific models. Our experiment demonstrates that MusT-RAG significantly outperforms traditional fine-tuning approaches in enhancing LLMs' music domain adaptation capabilities, showing consistent improvements across both in-domain and out-of-domain MQA benchmarks. Additionally, our MusWikiDB proves substantially more effective than general Wikipedia corpora, delivering superior performance and computational efficiency. 

---
# FastDriveVLA: Efficient End-to-End Driving via Plug-and-Play Reconstruction-based Token Pruning 

**Authors**: Jiajun Cao, Qizhe Zhang, Peidong Jia, Xuhui Zhao, Bo Lan, Xiaoan Zhang, Xiaobao Wei, Sixiang Chen, Zhuo Li, Yang Wang, Liyun Li, Xianming Liu, Ming Lu, Shanghang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.23318)  

**Abstract**: Vision-Language-Action (VLA) models have demonstrated significant potential in complex scene understanding and action reasoning, leading to their increasing adoption in end-to-end autonomous driving systems. However, the long visual tokens of VLA models greatly increase computational costs. Current visual token pruning methods in Vision-Language Models (VLM) rely on either visual token similarity or visual-text attention, but both have shown poor performance in autonomous driving scenarios. Given that human drivers concentrate on relevant foreground areas while driving, we assert that retaining visual tokens containing this foreground information is essential for effective decision-making. Inspired by this, we propose FastDriveVLA, a novel reconstruction-based vision token pruning framework designed specifically for autonomous driving. FastDriveVLA includes a plug-and-play visual token pruner called ReconPruner, which prioritizes foreground information through MAE-style pixel reconstruction. A novel adversarial foreground-background reconstruction strategy is designed to train ReconPruner for the visual encoder of VLA models. Once trained, ReconPruner can be seamlessly applied to different VLA models with the same visual encoder without retraining. To train ReconPruner, we also introduce a large-scale dataset called nuScenes-FG, consisting of 241K image-mask pairs with annotated foreground regions. Our approach achieves state-of-the-art results on the nuScenes closed-loop planning benchmark across different pruning ratios. 

---
# Impact of Hyperparameter Optimization on the Accuracy of Lightweight Deep Learning Models for Real-Time Image Classification 

**Authors**: Vineet Kumar Rakesh, Soumya Mazumdar, Tapas Samanta, Sarbajit Pal, Amitabha Das  

**Link**: [PDF](https://arxiv.org/pdf/2507.23315)  

**Abstract**: Lightweight convolutional and transformer-based models have become vital for real-time image classification in resource-constrained applications, such as embedded systems and edge devices. This work analyzes the influence of hyperparameter adjustment on the accuracy and convergence behavior of seven efficient deep learning architectures: EfficientNetV2-S, ConvNeXt-T, MobileViT v2 (XXS/XS/S), MobileNetV3-L, TinyViT-21M, and RepVGG-A2. All models are trained on the ImageNet-1K dataset under consistent training settings, with an emphasis on real-time practicality. An comprehensive ablation study is undertaken to separate the effect of critical hyperparameters, including learning rate schedules, batch sizes, input resolution, data augmentation, regularization approaches, and optimizer choice. To assess appropriateness for real-time applications, each model is assessed not only in terms of Top-1 and Top-5 classification accuracy, but also in terms of inference time, parameter count, model size, and frames-per-second (FPS) on a GPU-accelerated edge deployment simulation. Results demonstrate that cosine learning rate decay and adjustable batch size may greatly boost both accuracy and convergence speed, while keeping low latency and memory cost. Notably, RepVGG-A2 achieves over 80% Top-1 accuracy with efficient inference performance, offering a compelling balance between accuracy and deployment cost for VGG-style models. The results give practical guidance for constructing resource-efficient deep learning models appropriate for real-time image processing pipelines. All code and training logs are publicly accessible at this https URL. 

---
# Evaluating the Dynamics of Membership Privacy in Deep Learning 

**Authors**: Yuetian Chen, Zhiqi Wang, Nathalie Baracaldo, Swanand Ravindra Kadhe, Lei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.23291)  

**Abstract**: Membership inference attacks (MIAs) pose a critical threat to the privacy of training data in deep learning. Despite significant progress in attack methodologies, our understanding of when and how models encode membership information during training remains limited. This paper presents a dynamic analytical framework for dissecting and quantifying privacy leakage dynamics at the individual sample level. By tracking per-sample vulnerabilities on an FPR-TPR plane throughout training, our framework systematically measures how factors such as dataset complexity, model architecture, and optimizer choice influence the rate and severity at which samples become vulnerable. Crucially, we discover a robust correlation between a sample's intrinsic learning difficulty, and find that the privacy risk of samples highly vulnerable in the final trained model is largely determined early during training. Our results thus provide a deeper understanding of how privacy risks dynamically emerge during training, laying the groundwork for proactive, privacy-aware model training strategies. 

---
# Towards Affordable Tumor Segmentation and Visualization for 3D Breast MRI Using SAM2 

**Authors**: Solha Kang, Eugene Kim, Joris Vankerschaver, Utku Ozbulak  

**Link**: [PDF](https://arxiv.org/pdf/2507.23272)  

**Abstract**: Breast MRI provides high-resolution volumetric imaging critical for tumor assessment and treatment planning, yet manual interpretation of 3D scans remains labor-intensive and subjective. While AI-powered tools hold promise for accelerating medical image analysis, adoption of commercial medical AI products remains limited in low- and middle-income countries due to high license costs, proprietary software, and infrastructure demands. In this work, we investigate whether the Segment Anything Model 2 (SAM2) can be adapted for low-cost, minimal-input 3D tumor segmentation in breast MRI. Using a single bounding box annotation on one slice, we propagate segmentation predictions across the 3D volume using three different slice-wise tracking strategies: top-to-bottom, bottom-to-top, and center-outward. We evaluate these strategies across a large cohort of patients and find that center-outward propagation yields the most consistent and accurate segmentations. Despite being a zero-shot model not trained for volumetric medical data, SAM2 achieves strong segmentation performance under minimal supervision. We further analyze how segmentation performance relates to tumor size, location, and shape, identifying key failure modes. Our results suggest that general-purpose foundation models such as SAM2 can support 3D medical image analysis with minimal supervision, offering an accessible and affordable alternative for resource-constrained settings. 

---
# XABPs: Towards eXplainable Autonomous Business Processes 

**Authors**: Peter Fettke, Fabiana Fournier, Lior Limonad, Andreas Metzger, Stefanie Rinderle-Ma, Barbara Weber  

**Link**: [PDF](https://arxiv.org/pdf/2507.23269)  

**Abstract**: Autonomous business processes (ABPs), i.e., self-executing workflows leveraging AI/ML, have the potential to improve operational efficiency, reduce errors, lower costs, improve response times, and free human workers for more strategic and creative work. However, ABPs may raise specific concerns including decreased stakeholder trust, difficulties in debugging, hindered accountability, risk of bias, and issues with regulatory compliance. We argue for eXplainable ABPs (XABPs) to address these concerns by enabling systems to articulate their rationale. The paper outlines a systematic approach to XABPs, characterizing their forms, structuring explainability, and identifying key BPM research challenges towards XABPs. 

---
# DynaSwarm: Dynamically Graph Structure Selection for LLM-based Multi-agent System 

**Authors**: Hui Yi Leong, Yuqing Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.23261)  

**Abstract**: Current multi-agent systems (MAS) frameworks often rely on manually designed and static collaboration graph structures, limiting adaptability and performance. To address these limitations, we propose DynaSwarm, a dynamic framework that enhances LLM-based MAS through two key innovations: (1) an actor-critic reinforcement learning (A2C) mechanism to optimize graph structures with improved stability over prior RL methods, and (2) a dynamic graph selector that adaptively chooses the optimal graph structure for each input sample via parameter-efficient LLM fine-tuning. DynaSwarm eliminates the need for rigid, one-fits-all graph architectures, instead leveraging sample-specific idiosyncrasies to dynamically route queries through specialized agent networks. (c) We propose to fine-tune the demonstration retriever to fully exploit the power of in-context learning (ICL). Extensive experiments on question answering, mathematical reasoning, and coding tasks demonstrate that DynaSwarm consistently outperforms state-of-the-art single-agent and MAS baselines across multiple LLM backbones. Our findings highlight the importance of sample-aware structural flexibility in LLM MAS designs. 

---
# Efficient Machine Unlearning via Influence Approximation 

**Authors**: Jiawei Liu, Chenwang Wu, Defu Lian, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.23257)  

**Abstract**: Due to growing privacy concerns, machine unlearning, which aims at enabling machine learning models to ``forget" specific training data, has received increasing attention. Among existing methods, influence-based unlearning has emerged as a prominent approach due to its ability to estimate the impact of individual training samples on model parameters without retraining. However, this approach suffers from prohibitive computational overhead arising from the necessity to compute the Hessian matrix and its inverse across all training samples and parameters, rendering it impractical for large-scale models and scenarios involving frequent data deletion requests. This highlights the difficulty of forgetting. Inspired by cognitive science, which suggests that memorizing is easier than forgetting, this paper establishes a theoretical link between memorizing (incremental learning) and forgetting (unlearning). This connection allows machine unlearning to be addressed from the perspective of incremental learning. Unlike the time-consuming Hessian computations in unlearning (forgetting), incremental learning (memorizing) typically relies on more efficient gradient optimization, which supports the aforementioned cognitive theory. Based on this connection, we introduce the Influence Approximation Unlearning (IAU) algorithm for efficient machine unlearning from the incremental perspective. Extensive empirical evaluations demonstrate that IAU achieves a superior balance among removal guarantee, unlearning efficiency, and comparable model utility, while outperforming state-of-the-art methods across diverse datasets and model architectures. Our code is available at this https URL. 

---
# An Information Bottleneck Asset Pricing Model 

**Authors**: Che Sun  

**Link**: [PDF](https://arxiv.org/pdf/2507.23218)  

**Abstract**: Deep neural networks (DNNs) have garnered significant attention in financial asset pricing, due to their strong capacity for modeling complex nonlinear relationships within financial data. However, sophisticated models are prone to over-fitting to the noise information in financial data, resulting in inferior performance. To address this issue, we propose an information bottleneck asset pricing model that compresses data with low signal-to-noise ratios to eliminate redundant information and retain the critical information for asset pricing. Our model imposes constraints of mutual information during the nonlinear mapping process. Specifically, we progressively reduce the mutual information between the input data and the compressed representation while increasing the mutual information between the compressed representation and the output prediction. The design ensures that irrelevant information, which is essentially the noise in the data, is forgotten during the modeling of financial nonlinear relationships without affecting the final asset pricing. By leveraging the constraints of the Information bottleneck, our model not only harnesses the nonlinear modeling capabilities of deep networks to capture the intricate relationships within financial data but also ensures that noise information is filtered out during the information compression process. 

---
# Zero-Shot Document Understanding using Pseudo Table of Contents-Guided Retrieval-Augmented Generation 

**Authors**: Hyeon Seong Jeong, Sangwoo Jo, Byeong Hyun Yoon, Yoonseok Heo, Haedong Jeong, Taehoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.23217)  

**Abstract**: Understanding complex multimodal documents remains challenging due to their structural inconsistencies and limited training data availability. We introduce \textit{DocsRay}, a training-free document understanding system that integrates pseudo Table of Contents (TOC) generation with hierarchical Retrieval-Augmented Generation (RAG). Our approach leverages multimodal Large Language Models' (LLMs) native capabilities to seamlessly process documents containing diverse elements such as text, images, charts, and tables without requiring specialized models or additional training. DocsRay's framework synergistically combines three key techniques: (1) a semantic structuring module using prompt-based LLM interactions to generate a hierarchical pseudo-TOC, (2) zero-shot multimodal analysis that converts diverse document elements into unified, text-centric representations using the inherent capabilities of multimodal LLMs, and (3) an efficient two-stage hierarchical retrieval system that reduces retrieval complexity from $O(N)$ to $O(S + k_1 \cdot N_s)$. Evaluated on documents averaging 49.4 pages and 20,971 textual tokens, DocsRay reduced query latency from 3.89 to 2.12 seconds, achieving a 45% efficiency improvement. On the MMLongBench-Doc benchmark, DocsRay-Pro attains an accuracy of 64.7%, substantially surpassing previous state-of-the-art results. 

---
# Geak: Introducing Triton Kernel AI Agent & Evaluation Benchmarks 

**Authors**: Jianghui Wang, Vinay Joshi, Saptarshi Majumder, Xu Chao, Bin Ding, Ziqiong Liu, Pratik Prabhanjan Brahma, Dong Li, Zicheng Liu, Emad Barsoum  

**Link**: [PDF](https://arxiv.org/pdf/2507.23194)  

**Abstract**: The demand for AI-generated GPU kernels is rapidly growing, influenced by the need for scalable, hardware-optimized solutions in both industry and academia. As deep learning workloads grow in complexity and diversity, it is imperative to automate low-level kernel development to meet performance and productivity demands. Major cloud providers, semiconductor companies, and research institutions are now investing heavily in AI-driven code generation for GPUs, aiming to reduce manual optimization efforts while achieving near-expert performance on hardware like AMD MI300X. The Triton language, a Python-based DSL for GPU programming, has emerged as a popular target for such AI-generated kernels due to its balance of performance and ease-of-coding. In this work, we present an evaluation suite for Triton-based GPU kernels and GEAK (Generating Efficient AI-centric GPU Kernels)-a framework that leverages cutting-edge LLMs to generate performant Triton code specifically for AMD GPUs, including the AMD MI300X and MI250. GEAK leverages inference-time compute scaling to produce Triton-based GPU kernels using a reasoning loop adapted from Reflexion-style feedback mechanisms. On two evaluation benchmarks, GEAK significantly outperformed the baselines of directly prompting frontier LLMs as well as Reflexion-based generation pipelines by achieving correctness up to $63$% and execution speed up of up to $2.59$X. These results highlight the promise of GEAK-like agentic code generation for accelerating the adoption of diverse hardware platforms and democratizing access to expert-level kernel performance. 

---
# Accessibility Scout: Personalized Accessibility Scans of Built Environments 

**Authors**: William Huang, Xia Su, Jon E. Froehlich, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.23190)  

**Abstract**: Assessing the accessibility of unfamiliar built environments is critical for people with disabilities. However, manual assessments, performed by users or their personal health professionals, are laborious and unscalable, while automatic machine learning methods often neglect an individual user's unique needs. Recent advances in Large Language Models (LLMs) enable novel approaches to this problem, balancing personalization with scalability to enable more adaptive and context-aware assessments of accessibility. We present Accessibility Scout, an LLM-based accessibility scanning system that identifies accessibility concerns from photos of built environments. With use, Accessibility Scout becomes an increasingly capable "accessibility scout", tailoring accessibility scans to an individual's mobility level, preferences, and specific environmental interests through collaborative Human-AI assessments. We present findings from three studies: a formative study with six participants to inform the design of Accessibility Scout, a technical evaluation of 500 images of built environments, and a user study with 10 participants of varying mobility. Results from our technical evaluation and user study show that Accessibility Scout can generate personalized accessibility scans that extend beyond traditional ADA considerations. Finally, we conclude with a discussion on the implications of our work and future steps for building more scalable and personalized accessibility assessments of the physical world. 

---
# AutoBridge: Automating Smart Device Integration with Centralized Platform 

**Authors**: Siyuan Liu, Zhice Yang, Huangxun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.23178)  

**Abstract**: Multimodal IoT systems coordinate diverse IoT devices to deliver human-centered services. The ability to incorporate new IoT devices under the management of a centralized platform is an essential requirement. However, it requires significant human expertise and effort to program the complex IoT integration code that enables the platform to understand and control the device functions. Therefore, we propose AutoBridge to automate IoT integration code generation. Specifically, AutoBridge adopts a divide-and-conquer strategy: it first generates device control logic by progressively retrieving device-specific knowledge, then synthesizes platformcompliant integration code using platform-specific knowledge. To ensure correctness, AutoBridge features a multi-stage debugging pipeline, including an automated debugger for virtual IoT device testing and an interactive hardware-in-the-loop debugger that requires only binary user feedback (yes and no) for real-device verification. We evaluate AutoBridge on a benchmark of 34 IoT devices across two open-source IoT platforms. The results demonstrate that AutoBridge can achieves an average success rate of 93.87% and an average function coverage of 94.87%, without any human involvement. With minimal binary yes and no feedback from users, the code is then revised to reach 100% function coverage. A user study with 15 participants further shows that AutoBridge outperforms expert programmers by 50% to 80% in code accuracy, even when the programmers are allowed to use commercial code LLMs. 

---
# LENS: Learning Ensemble Confidence from Neural States for Multi-LLM Answer Integration 

**Authors**: Jizhou Guo  

**Link**: [PDF](https://arxiv.org/pdf/2507.23167)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive performance across various tasks, with different models excelling in distinct domains and specific abilities. Effectively combining the predictions of multiple LLMs is crucial for enhancing system robustness and performance. However, existing ensemble methods often rely on simple techniques like voting or logits ensembling, which overlook the varying confidence and reliability of models in different contexts. In this work, we propose LENS (Learning ENsemble confidence from Neural States), a novel approach that learns to estimate model confidence by analyzing internal representations. For each LLM, we train a lightweight linear confidence predictor that leverages layer-wise hidden states and normalized probabilities as inputs. This allows for more nuanced weighting of model predictions based on their context-dependent reliability. Our method does not require modifying the model parameters and requires negligible additional computation. Experimental results on multiple-choice and boolean question-answering tasks demonstrate that LENS outperforms traditional ensemble methods by a substantial margin. Our findings suggest that internal representations provide valuable signals for determining model confidence and can be effectively leveraged for ensemble learning. 

---
# FuseTen: A Generative Model for Daily 10 m Land Surface Temperature Estimation from Spatio-Temporal Satellite Observations 

**Authors**: Sofiane Bouaziz, Adel Hafiane, Raphael Canals, Rachid Nedjai  

**Link**: [PDF](https://arxiv.org/pdf/2507.23154)  

**Abstract**: Urban heatwaves, droughts, and land degradation are pressing and growing challenges in the context of climate change. A valuable approach to studying them requires accurate spatio-temporal information on land surface conditions. One of the most important variables for assessing and understanding these phenomena is Land Surface Temperature (LST), which is derived from satellites and provides essential information about the thermal state of the Earth's surface. However, satellite platforms inherently face a trade-off between spatial and temporal resolutions. To bridge this gap, we propose FuseTen, a novel generative framework that produces daily LST observations at a fine 10 m spatial resolution by fusing spatio-temporal observations derived from Sentinel-2, Landsat 8, and Terra MODIS. FuseTen employs a generative architecture trained using an averaging-based supervision strategy grounded in physical principles. It incorporates attention and normalization modules within the fusion process and uses a PatchGAN discriminator to enforce realism. Experiments across multiple dates show that FuseTen outperforms linear baselines, with an average 32.06% improvement in quantitative metrics and 31.42% in visual fidelity. To the best of our knowledge, this is the first non-linear method to generate daily LST estimates at such fine spatial resolution. 

---
# Uncovering the Fragility of Trustworthy LLMs through Chinese Textual Ambiguity 

**Authors**: Xinwei Wu, Haojie Li, Hongyu Liu, Xinyu Ji, Ruohan Li, Yule Chen, Yigeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.23121)  

**Abstract**: In this work, we study a critical research problem regarding the trustworthiness of large language models (LLMs): how LLMs behave when encountering ambiguous narrative text, with a particular focus on Chinese textual ambiguity. We created a benchmark dataset by collecting and generating ambiguous sentences with context and their corresponding disambiguated pairs, representing multiple possible interpretations. These annotated examples are systematically categorized into 3 main categories and 9 subcategories. Through experiments, we discovered significant fragility in LLMs when handling ambiguity, revealing behavior that differs substantially from humans. Specifically, LLMs cannot reliably distinguish ambiguous text from unambiguous text, show overconfidence in interpreting ambiguous text as having a single meaning rather than multiple meanings, and exhibit overthinking when attempting to understand the various possible meanings. Our findings highlight a fundamental limitation in current LLMs that has significant implications for their deployment in real-world applications where linguistic ambiguity is common, calling for improved approaches to handle uncertainty in language understanding. The dataset and code are publicly available at this GitHub repository: this https URL. 

---
# FLOSS: Federated Learning with Opt-Out and Straggler Support 

**Authors**: David J Goetze, Dahlia J Felten, Jeannie R Albrecht, Rohit Bhattacharya  

**Link**: [PDF](https://arxiv.org/pdf/2507.23115)  

**Abstract**: Previous work on data privacy in federated learning systems focuses on privacy-preserving operations for data from users who have agreed to share their data for training. However, modern data privacy agreements also empower users to use the system while opting out of sharing their data as desired. When combined with stragglers that arise from heterogeneous device capabilities, the result is missing data from a variety of sources that introduces bias and degrades model performance. In this paper, we present FLOSS, a system that mitigates the impacts of such missing data on federated learning in the presence of stragglers and user opt-out, and empirically demonstrate its performance in simulations. 

---
# RASL: Retrieval Augmented Schema Linking for Massive Database Text-to-SQL 

**Authors**: Jeffrey Eben, Aitzaz Ahmad, Stephen Lau  

**Link**: [PDF](https://arxiv.org/pdf/2507.23104)  

**Abstract**: Despite advances in large language model (LLM)-based natural language interfaces for databases, scaling to enterprise-level data catalogs remains an under-explored challenge. Prior works addressing this challenge rely on domain-specific fine-tuning - complicating deployment - and fail to leverage important semantic context contained within database metadata. To address these limitations, we introduce a component-based retrieval architecture that decomposes database schemas and metadata into discrete semantic units, each separately indexed for targeted retrieval. Our approach prioritizes effective table identification while leveraging column-level information, ensuring the total number of retrieved tables remains within a manageable context budget. Experiments demonstrate that our method maintains high recall and accuracy, with our system outperforming baselines over massive databases with varying structure and available metadata. Our solution enables practical text-to-SQL systems deployable across diverse enterprise settings without specialized fine-tuning, addressing a critical scalability gap in natural language database interfaces. 

---
# SMART-Editor: A Multi-Agent Framework for Human-Like Design Editing with Structural Integrity 

**Authors**: Ishani Mondal, Meera Bharadwaj, Ayush Roy, Aparna Garimella, Jordan Lee Boyd-Graber  

**Link**: [PDF](https://arxiv.org/pdf/2507.23095)  

**Abstract**: We present SMART-Editor, a framework for compositional layout and content editing across structured (posters, websites) and unstructured (natural images) domains. Unlike prior models that perform local edits, SMART-Editor preserves global coherence through two strategies: Reward-Refine, an inference-time rewardguided refinement method, and RewardDPO, a training-time preference optimization approach using reward-aligned layout pairs. To evaluate model performance, we introduce SMARTEdit-Bench, a benchmark covering multi-domain, cascading edit scenarios. SMART-Editor outperforms strong baselines like InstructPix2Pix and HIVE, with RewardDPO achieving up to 15% gains in structured settings and Reward-Refine showing advantages on natural images. Automatic and human evaluations confirm the value of reward-guided planning in producing semantically consistent and visually aligned edits. 

---
# On the Sustainability of AI Inferences in the Edge 

**Authors**: Ghazal Sobhani, Md. Monzurul Amin Ifath, Tushar Sharma, Israat Haque  

**Link**: [PDF](https://arxiv.org/pdf/2507.23093)  

**Abstract**: The proliferation of the Internet of Things (IoT) and its cutting-edge AI-enabled applications (e.g., autonomous vehicles and smart industries) combine two paradigms: data-driven systems and their deployment on the edge. Usually, edge devices perform inferences to support latency-critical applications. In addition to the performance of these resource-constrained edge devices, their energy usage is a critical factor in adopting and deploying edge applications. Examples of such devices include Raspberry Pi (RPi), Intel Neural Compute Stick (INCS), NVIDIA Jetson nano (NJn), and Google Coral USB (GCU). Despite their adoption in edge deployment for AI inferences, there is no study on their performance and energy usage for informed decision-making on the device and model selection to meet the demands of applications. This study fills the gap by rigorously characterizing the performance of traditional, neural networks, and large language models on the above-edge devices. Specifically, we analyze trade-offs among model F1 score, inference time, inference power, and memory usage. Hardware and framework optimization, along with external parameter tuning of AI models, can balance between model performance and resource usage to realize practical edge AI deployments. 

---
# Beyond Rigid AI: Towards Natural Human-Machine Symbiosis for Interoperative Surgical Assistance 

**Authors**: Lalithkumar Seenivasan, Jiru Xu, Roger D. Soberanis Mukul, Hao Ding, Grayson Byrd, Yu-Chun Ku, Jose L. Porras, Masaru Ishii, Mathias Unberath  

**Link**: [PDF](https://arxiv.org/pdf/2507.23088)  

**Abstract**: Emerging surgical data science and robotics solutions, especially those designed to provide assistance in situ, require natural human-machine interfaces to fully unlock their potential in providing adaptive and intuitive aid. Contemporary AI-driven solutions remain inherently rigid, offering limited flexibility and restricting natural human-machine interaction in dynamic surgical environments. These solutions rely heavily on extensive task-specific pre-training, fixed object categories, and explicit manual-prompting. This work introduces a novel Perception Agent that leverages speech-integrated prompt-engineered large language models (LLMs), segment anything model (SAM), and any-point tracking foundation models to enable a more natural human-machine interaction in real-time intraoperative surgical assistance. Incorporating a memory repository and two novel mechanisms for segmenting unseen elements, Perception Agent offers the flexibility to segment both known and unseen elements in the surgical scene through intuitive interaction. Incorporating the ability to memorize novel elements for use in future surgeries, this work takes a marked step towards human-machine symbiosis in surgical procedures. Through quantitative analysis on a public dataset, we show that the performance of our agent is on par with considerably more labor-intensive manual-prompting strategies. Qualitatively, we show the flexibility of our agent in segmenting novel elements (instruments, phantom grafts, and gauze) in a custom-curated dataset. By offering natural human-machine interaction and overcoming rigidity, our Perception Agent potentially brings AI-based real-time assistance in dynamic surgical environments closer to reality. 

---
# On LLM-Assisted Generation of Smart Contracts from Business Processes 

**Authors**: Fabian Stiehle, Hans Weytjens, Ingo Weber  

**Link**: [PDF](https://arxiv.org/pdf/2507.23087)  

**Abstract**: Large language models (LLMs) have changed the reality of how software is produced. Within the wider software engineering community, among many other purposes, they are explored for code generation use cases from different types of input. In this work, we present an exploratory study to investigate the use of LLMs for generating smart contract code from business process descriptions, an idea that has emerged in recent literature to overcome the limitations of traditional rule-based code generation approaches. However, current LLM-based work evaluates generated code on small samples, relying on manual inspection, or testing whether code compiles but ignoring correct execution. With this work, we introduce an automated evaluation framework and provide empirical data from larger data sets of process models. We test LLMs of different types and sizes in their capabilities of achieving important properties of process execution, including enforcing process flow, resource allocation, and data-based conditions. Our results show that LLM performance falls short of the perfect reliability required for smart contract development. We suggest future work to explore responsible LLM integrations in existing tools for code generation to ensure more reliable output. Our benchmarking framework can serve as a foundation for developing and evaluating such integrations. 

---
# AutoIndexer: A Reinforcement Learning-Enhanced Index Advisor Towards Scaling Workloads 

**Authors**: Taiyi Wang, Eiko Yoneki  

**Link**: [PDF](https://arxiv.org/pdf/2507.23084)  

**Abstract**: Efficiently selecting indexes is fundamental to database performance optimization, particularly for systems handling large-scale analytical workloads. While deep reinforcement learning (DRL) has shown promise in automating index selection through its ability to learn from experience, few works address how these RL-based index advisors can adapt to scaling workloads due to exponentially growing action spaces and heavy trial and error. To address these challenges, we introduce AutoIndexer, a framework that combines workload compression, query optimization, and specialized RL models to scale index selection effectively. By operating on compressed workloads, AutoIndexer substantially lowers search complexity without sacrificing much index quality. Extensive evaluations show that it reduces end-to-end query execution time by up to 95% versus non-indexed baselines. On average, it outperforms state-of-the-art RL-based index advisors by approximately 20% in workload cost savings while cutting tuning time by over 50%. These results affirm AutoIndexer's practicality for large and diverse workloads. 

---
# Vision-Language Fusion for Real-Time Autonomous Driving: Goal-Centered Cross-Attention of Camera, HD-Map, & Waypoints 

**Authors**: Santosh Patapati, Trisanth Srinivasan, Murari Ambati  

**Link**: [PDF](https://arxiv.org/pdf/2507.23064)  

**Abstract**: Autonomous cars need geometric accuracy and semantic understanding to navigate complex environments, yet most stacks handle them separately. We present XYZ-Drive, a single vision-language model that reads a front-camera frame, a 25m $\times$ 25m overhead map, and the next waypoint, then outputs steering and speed. A lightweight goal-centered cross-attention layer lets waypoint tokens highlight relevant image and map patches, supporting both action and textual explanations, before the fused tokens enter a partially fine-tuned LLaMA-3.2 11B model.
On the MD-NEX Outdoor-Driving benchmark XYZ-Drive attains 95% success and 0.80 Success weighted by Path Length (SPL), surpassing PhysNav-DG by 15%. and halving collisions, all while significantly improving efficiency by using only a single branch. Sixteen ablations explain the gains. Removing any modality (vision, waypoint, map) drops success by up to 11%, confirming their complementary roles and rich connections. Replacing goal-centered attention with simple concatenation cuts 3% in performance, showing query-based fusion injects map knowledge more effectively. Keeping the transformer frozen loses 5%, showing the importance of fine-tuning when applying VLMs for specific tasks such as autonomous driving. Coarsening map resolution from 10 cm to 40 cm blurs lane edges and raises crash rate.
Overall, these results demonstrate that early, token-level fusion of intent and map layout enables accurate, transparent, real-time driving. 

---
# Reference-Guided Diffusion Inpainting For Multimodal Counterfactual Generation 

**Authors**: Alexandru Buburuzan  

**Link**: [PDF](https://arxiv.org/pdf/2507.23058)  

**Abstract**: Safety-critical applications, such as autonomous driving and medical image analysis, require extensive multimodal data for rigorous testing. Synthetic data methods are gaining prominence due to the cost and complexity of gathering real-world data, but they demand a high degree of realism and controllability to be useful. This work introduces two novel methods for synthetic data generation in autonomous driving and medical image analysis, namely MObI and AnydoorMed, respectively. MObI is a first-of-its-kind framework for Multimodal Object Inpainting that leverages a diffusion model to produce realistic and controllable object inpaintings across perceptual modalities, demonstrated simultaneously for camera and lidar. Given a single reference RGB image, MObI enables seamless object insertion into existing multimodal scenes at a specified 3D location, guided by a bounding box, while maintaining semantic consistency and multimodal coherence. Unlike traditional inpainting methods that rely solely on edit masks, this approach uses 3D bounding box conditioning to ensure accurate spatial positioning and realistic scaling. AnydoorMed extends this paradigm to the medical imaging domain, focusing on reference-guided inpainting for mammography scans. It leverages a diffusion-based model to inpaint anomalies with impressive detail preservation, maintaining the reference anomaly's structural integrity while semantically blending it with the surrounding tissue. Together, these methods demonstrate that foundation models for reference-guided inpainting in natural images can be readily adapted to diverse perceptual modalities, paving the way for the next generation of systems capable of constructing highly realistic, controllable and multimodal counterfactual scenarios. 

---
# Early Goal-Guided Multi-Scale Fusion for Real-Time Vision-Language Driving 

**Authors**: Santosh Patapati, Trisanth Srinivasan  

**Link**: [PDF](https://arxiv.org/pdf/2507.23042)  

**Abstract**: Autonomous vehicles must react in milliseconds while reasoning about road geometry and traffic intent to navigate complex situations. We introduce NovaDrive, a single-branch vision-language architecture that processes front-camera images, HD-map tiles, LiDAR depth, and textual waypoints in a single branch. A lightweight, two-stage cross-attention block first aligns waypoint tokens with the HD map, then refines attention over fine-grained image and depth patches. Coupled with a novel smoothness loss that discourages abrupt steering and speed changes, this design eliminates the need for recurrent memory. We fine-tune the top 15 layers of an 11B LLaMA-3.2 vision-language backbone, enabling real-time inference. On the nuScenes / Waymo subset of the MD-NEX Outdoor benchmark, NovaDrive raises success rate to 84% (+4%), boosts path-efficiency (SPL) to 0.66 (+0.11), and reduces collision frequency from 2.6% to 1.2% (-1.4%) relative to the previous state-of-the-art. Our ablations confirm that waypoint tokens, partial VLM fine-tuning, and the cross-attention fusion each contribute the most to these gains. Beyond safety, NovaDrive's shorter routes (resulting from the novel smoothness loss) translate to lower fuel or battery usage, pointing toward leaner, more easily updated driving stacks. NovaDrive can be extended to other embodied-AI domains as well. 

---
# Recovering Diagnostic Value: Super-Resolution-Aided Echocardiographic Classification in Resource-Constrained Imaging 

**Authors**: Krishan Agyakari Raja Babu, Om Prabhu, Annu, Mohanasankar Sivaprakasam  

**Link**: [PDF](https://arxiv.org/pdf/2507.23027)  

**Abstract**: Automated cardiac interpretation in resource-constrained settings (RCS) is often hindered by poor-quality echocardiographic imaging, limiting the effectiveness of downstream diagnostic models. While super-resolution (SR) techniques have shown promise in enhancing magnetic resonance imaging (MRI) and computed tomography (CT) scans, their application to echocardiography-a widely accessible but noise-prone modality-remains underexplored. In this work, we investigate the potential of deep learning-based SR to improve classification accuracy on low-quality 2D echocardiograms. Using the publicly available CAMUS dataset, we stratify samples by image quality and evaluate two clinically relevant tasks of varying complexity: a relatively simple Two-Chamber vs. Four-Chamber (2CH vs. 4CH) view classification and a more complex End-Diastole vs. End-Systole (ED vs. ES) phase classification. We apply two widely used SR models-Super-Resolution Generative Adversarial Network (SRGAN) and Super-Resolution Residual Network (SRResNet), to enhance poor-quality images and observe significant gains in performance metric-particularly with SRResNet, which also offers computational efficiency. Our findings demonstrate that SR can effectively recover diagnostic value in degraded echo scans, making it a viable tool for AI-assisted care in RCS, achieving more with less. 

---
# Modeling Human Gaze Behavior with Diffusion Models for Unified Scanpath Prediction 

**Authors**: Giuseppe Cartella, Vittorio Cuculo, Alessandro D'Amelio, Marcella Cornia, Giuseppe Boccignone, Rita Cucchiara  

**Link**: [PDF](https://arxiv.org/pdf/2507.23021)  

**Abstract**: Predicting human gaze scanpaths is crucial for understanding visual attention, with applications in human-computer interaction, autonomous systems, and cognitive robotics. While deep learning models have advanced scanpath prediction, most existing approaches generate averaged behaviors, failing to capture the variability of human visual exploration. In this work, we present ScanDiff, a novel architecture that combines diffusion models with Vision Transformers to generate diverse and realistic scanpaths. Our method explicitly models scanpath variability by leveraging the stochastic nature of diffusion models, producing a wide range of plausible gaze trajectories. Additionally, we introduce textual conditioning to enable task-driven scanpath generation, allowing the model to adapt to different visual search objectives. Experiments on benchmark datasets show that ScanDiff surpasses state-of-the-art methods in both free-viewing and task-driven scenarios, producing more diverse and accurate scanpaths. These results highlight its ability to better capture the complexity of human visual behavior, pushing forward gaze prediction research. Source code and models are publicly available at this https URL. 

---
# Investigating the Invertibility of Multimodal Latent Spaces: Limitations of Optimization-Based Methods 

**Authors**: Siwoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2507.23010)  

**Abstract**: This paper investigates the inverse capabilities and broader utility of multimodal latent spaces within task-specific AI (Artificial Intelligence) models. While these models excel at their designed forward tasks (e.g., text-to-image generation, audio-to-text transcription), their potential for inverse mappings remains largely unexplored. We propose an optimization-based framework to infer input characteristics from desired outputs, applying it bidirectionally across Text-Image (BLIP, Flux.1-dev) and Text-Audio (Whisper-Large-V3, Chatterbox-TTS) modalities.
Our central hypothesis posits that while optimization can guide models towards inverse tasks, their multimodal latent spaces will not consistently support semantically meaningful and perceptually coherent inverse mappings. Experimental results consistently validate this hypothesis. We demonstrate that while optimization can force models to produce outputs that align textually with targets (e.g., a text-to-image model generating an image that an image captioning model describes correctly, or an ASR model transcribing optimized audio accurately), the perceptual quality of these inversions is chaotic and incoherent. Furthermore, when attempting to infer the original semantic input from generative models, the reconstructed latent space embeddings frequently lack semantic interpretability, aligning with nonsensical vocabulary tokens.
These findings highlight a critical limitation. multimodal latent spaces, primarily optimized for specific forward tasks, do not inherently possess the structure required for robust and interpretable inverse mappings. Our work underscores the need for further research into developing truly semantically rich and invertible multimodal latent spaces. 

---
# Stop Evaluating AI with Human Tests, Develop Principled, AI-specific Tests instead 

**Authors**: Tom Sühr, Florian E. Dorner, Olawale Salaudeen, Augustin Kelava, Samira Samadi  

**Link**: [PDF](https://arxiv.org/pdf/2507.23009)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable results on a range of standardized tests originally designed to assess human cognitive and psychological traits, such as intelligence and personality. While these results are often interpreted as strong evidence of human-like characteristics in LLMs, this paper argues that such interpretations constitute an ontological error. Human psychological and educational tests are theory-driven measurement instruments, calibrated to a specific human population. Applying these tests to non-human subjects without empirical validation, risks mischaracterizing what is being measured. Furthermore, a growing trend frames AI performance on benchmarks as measurements of traits such as ``intelligence'', despite known issues with validity, data contamination, cultural bias and sensitivity to superficial prompt changes. We argue that interpreting benchmark performance as measurements of human-like traits, lacks sufficient theoretical and empirical justification. This leads to our position: Stop Evaluating AI with Human Tests, Develop Principled, AI-specific Tests instead. We call for the development of principled, AI-specific evaluation frameworks tailored to AI systems. Such frameworks might build on existing frameworks for constructing and validating psychometrics tests, or could be created entirely from scratch to fit the unique context of AI. 

---
# C3: A Bilingual Benchmark for Spoken Dialogue Models Exploring Challenges in Complex Conversations 

**Authors**: Chengqian Ma, Wei Tao, Yiwen Guo  

**Link**: [PDF](https://arxiv.org/pdf/2507.22968)  

**Abstract**: Spoken Dialogue Models (SDMs) have recently attracted significant attention for their ability to generate voice responses directly to users' spoken queries. Despite their increasing popularity, there exists a gap in research focused on comprehensively understanding their practical effectiveness in comprehending and emulating human conversations. This is especially true compared to text-based Large Language Models (LLMs), which benefit from extensive benchmarking. Human voice interactions are inherently more complex than text due to characteristics unique to spoken dialogue. Ambiguity poses one challenge, stemming from semantic factors like polysemy, as well as phonological aspects such as heterograph, heteronyms, and stress patterns. Additionally, context-dependency, like omission, coreference, and multi-turn interaction, adds further complexity to human conversational dynamics. To illuminate the current state of SDM development and to address these challenges, we present a benchmark dataset in this paper, which comprises 1,079 instances in English and Chinese. Accompanied by an LLM-based evaluation method that closely aligns with human judgment, this dataset facilitates a comprehensive exploration of the performance of SDMs in tackling these practical challenges. 

---
# CHECK-MAT: Checking Hand-Written Mathematical Answers for the Russian Unified State Exam 

**Authors**: Ruslan Khrulev  

**Link**: [PDF](https://arxiv.org/pdf/2507.22958)  

**Abstract**: This paper introduces a novel benchmark, EGE-Math Solutions Assessment Benchmark, for evaluating Vision-Language Models (VLMs) on their ability to assess hand-written mathematical solutions. Unlike existing benchmarks that focus on problem solving, our approach centres on understanding student solutions, identifying mistakes, and assigning grades according to fixed criteria. We compile 122 scanned solutions from the Russian Unified State Exam (EGE) together with official expert grades, and evaluate seven modern VLMs from Google, OpenAI, Arcee AI, and Alibaba Cloud in three inference modes. The results reveal current limitations in mathematical reasoning and human-rubric alignment, opening new research avenues in AI-assisted assessment. You can find code in this https URL 

---
# SmartCourse: A Contextual AI-Powered Course Advising System for Undergraduates 

**Authors**: Yixuan Mi, Yiduo Yu, Yiyi Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2507.22946)  

**Abstract**: We present SmartCourse, an integrated course management and AI-driven advising system for undergraduate students (specifically tailored to the Computer Science (CPS) major). SmartCourse addresses the limitations of traditional advising tools by integrating transcript and plan information for student-specific context. The system combines a command-line interface (CLI) and a Gradio web GUI for instructors and students, manages user accounts, course enrollment, grading, and four-year degree plans, and integrates a locally hosted large language model (via Ollama) for personalized course recommendations. It leverages transcript and major plan to offer contextual advice (e.g., prioritizing requirements or retakes). We evaluated the system on 25 representative advising queries and introduced custom metrics: PlanScore, PersonalScore, Lift, and Recall to assess recommendation quality across different context conditions. Experiments show that using full context yields substantially more relevant recommendations than context-omitted modes, confirming the necessity of transcript and plan information for personalized academic advising. SmartCourse thus demonstrates how transcript-aware AI can enhance academic planning. 

---
# Opacity as Authority: Arbitrariness and the Preclusion of Contestation 

**Authors**: Naomi Omeonga wa Kayembe  

**Link**: [PDF](https://arxiv.org/pdf/2507.22944)  

**Abstract**: This article redefines arbitrariness not as a normative flaw or a symptom of domination, but as a foundational functional mechanism structuring human systems and interactions. Diverging from critical traditions that conflate arbitrariness with injustice, it posits arbitrariness as a semiotic trait: a property enabling systems - linguistic, legal, or social - to operate effectively while withholding their internal rationale. Building on Ferdinand de Saussure's concept of l'arbitraire du signe, the analysis extends this principle beyond language to demonstrate its cross-domain applicability, particularly in law and social dynamics. The paper introduces the "Motivation -> Constatability -> Contestability" chain, arguing that motivation functions as a crucial interface rendering an act's logic vulnerable to intersubjective contestation. When this chain is broken through mechanisms like "immotivization" or "Conflict Lateralization" (exemplified by "the blur of the wolf drowned in the fish"), acts produce binding effects without exposing their rationale, thus precluding justiciability. This structural opacity, while appearing illogical, is a deliberate design protecting authority from accountability. Drawing on Shannon's entropy model, the paper formalizes arbitrariness as A = H(L|M) (conditional entropy). It thereby proposes a modern theory of arbitrariness as a neutral operator central to control as well as care, an overlooked dimension of interpersonal relations. While primarily developed through human social systems, this framework also illuminates a new pathway for analyzing explainability in advanced artificial intelligence systems. 

---
# Trustworthy Reasoning: Evaluating and Enhancing Factual Accuracy in LLM Intermediate Thought Processes 

**Authors**: Rui Jiao, Yue Zhang, Jinku Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.22940)  

**Abstract**: We present RELIANCE (Reasoning Evaluation with Logical Integrity and Accuracy for Confidence Enhancement), a novel framework addressing a critical vulnerability in Large Language Models (LLMs): the prevalence of factual inaccuracies within intermediate reasoning steps despite correct final answers. This phenomenon poses substantial risks in high-stakes domains including healthcare, legal analysis, and scientific research, where erroneous yet confidently presented reasoning can mislead users into dangerous decisions. Our framework integrates three core components: (1) a specialized fact-checking classifier trained on counterfactually augmented data to detect subtle factual inconsistencies within reasoning chains; (2) a Group Relative Policy Optimization (GRPO) reinforcement learning approach that balances factuality, coherence, and structural correctness through multi-dimensional rewards; and (3) a mechanistic interpretability module examining how factuality improvements manifest in model activations during reasoning processes. Extensive evaluation across ten state-of-the-art models reveals concerning patterns: even leading models like Claude-3.7 and GPT-o1 demonstrate reasoning factual accuracy of only 81.93% and 82.57% respectively. RELIANCE significantly enhances factual robustness (up to 49.90% improvement) while maintaining or improving performance on challenging benchmarks including Math-500, AIME-2024, and GPQA. Furthermore, our activation-level analysis provides actionable insights into how factual enhancements reshape reasoning trajectories within model architectures, establishing foundations for future training methodologies that explicitly target factual robustness through activation-guided optimization. 

---
# PARROT: An Open Multilingual Radiology Reports Dataset 

**Authors**: Bastien Le Guellec, Kokou Adambounou, Lisa C Adams, Thibault Agripnidis, Sung Soo Ahn, Radhia Ait Chalal, Tugba Akinci D Antonoli, Philippe Amouyel, Henrik Andersson, Raphael Bentegeac, Claudio Benzoni, Antonino Andrea Blandino, Felix Busch, Elif Can, Riccardo Cau, Armando Ugo Cavallo, Christelle Chavihot, Erwin Chiquete, Renato Cuocolo, Eugen Divjak, Gordana Ivanac, Barbara Dziadkowiec Macek, Armel Elogne, Salvatore Claudio Fanni, Carlos Ferrarotti, Claudia Fossataro, Federica Fossataro, Katarzyna Fulek, Michal Fulek, Pawel Gac, Martyna Gachowska, Ignacio Garcia Juarez, Marco Gatti, Natalia Gorelik, Alexia Maria Goulianou, Aghiles Hamroun, Nicolas Herinirina, Krzysztof Kraik, Dominik Krupka, Quentin Holay, Felipe Kitamura, Michail E Klontzas, Anna Kompanowska, Rafal Kompanowski, Alexandre Lefevre, Tristan Lemke, Maximilian Lindholz, Lukas Muller, Piotr Macek, Marcus Makowski, Luigi Mannacio, Aymen Meddeb, Antonio Natale, Beatrice Nguema Edzang, Adriana Ojeda, Yae Won Park, Federica Piccione, Andrea Ponsiglione, Malgorzata Poreba, Rafal Poreba, Philipp Prucker, Jean Pierre Pruvo, Rosa Alba Pugliesi, Feno Hasina Rabemanorintsoa, Vasileios Rafailidis, Katarzyna Resler, Jan Rotkegel, Luca Saba, Ezann Siebert, Arnaldo Stanzione, Ali Fuat Tekin, Liz Toapanta Yanchapaxi, Matthaios Triantafyllou, Ekaterini Tsaoulia, Evangelia Vassalou, Federica Vernuccio, Johan Wasselius, Weilang Wang, Szymon Urban, Adrian Wlodarczak, Szymon Wlodarczak, Andrzej Wysocki, Lina Xu, Tomasz Zatonski, Shuhang Zhang, Sebastian Ziegelmayer, Gregory Kuchcinski, Keno K Bressem  

**Link**: [PDF](https://arxiv.org/pdf/2507.22939)  

**Abstract**: Rationale and Objectives: To develop and validate PARROT (Polyglottal Annotated Radiology Reports for Open Testing), a large, multicentric, open-access dataset of fictional radiology reports spanning multiple languages for testing natural language processing applications in radiology. Materials and Methods: From May to September 2024, radiologists were invited to contribute fictional radiology reports following their standard reporting practices. Contributors provided at least 20 reports with associated metadata including anatomical region, imaging modality, clinical context, and for non-English reports, English translations. All reports were assigned ICD-10 codes. A human vs. AI report differentiation study was conducted with 154 participants (radiologists, healthcare professionals, and non-healthcare professionals) assessing whether reports were human-authored or AI-generated. Results: The dataset comprises 2,658 radiology reports from 76 authors across 21 countries and 13 languages. Reports cover multiple imaging modalities (CT: 36.1%, MRI: 22.8%, radiography: 19.0%, ultrasound: 16.8%) and anatomical regions, with chest (19.9%), abdomen (18.6%), head (17.3%), and pelvis (14.1%) being most prevalent. In the differentiation study, participants achieved 53.9% accuracy (95% CI: 50.7%-57.1%) in distinguishing between human and AI-generated reports, with radiologists performing significantly better (56.9%, 95% CI: 53.3%-60.6%, p<0.05) than other groups. Conclusion: PARROT represents the largest open multilingual radiology report dataset, enabling development and validation of natural language processing applications across linguistic, geographic, and clinical boundaries without privacy constraints. 

---
# A Graph-based Approach for Multi-Modal Question Answering from Flowcharts in Telecom Documents 

**Authors**: Sumit Soman, H. G. Ranjani, Sujoy Roychowdhury, Venkata Dharma Surya Narayana Sastry, Akshat Jain, Pranav Gangrade, Ayaaz Khan  

**Link**: [PDF](https://arxiv.org/pdf/2507.22938)  

**Abstract**: Question-Answering (QA) from technical documents often involves questions whose answers are present in figures, such as flowcharts or flow diagrams. Text-based Retrieval Augmented Generation (RAG) systems may fail to answer such questions. We leverage graph representations of flowcharts obtained from Visual large Language Models (VLMs) and incorporate them in a text-based RAG system to show that this approach can enable image retrieval for QA in the telecom domain. We present the end-to-end approach from processing technical documents, classifying image types, building graph representations, and incorporating them with the text embedding pipeline for efficient retrieval. We benchmark the same on a QA dataset created based on proprietary telecom product information documents. Results show that the graph representations obtained using a fine-tuned VLM model have lower edit distance with respect to the ground truth, which illustrate the robustness of these representations for flowchart images. Further, the approach for QA using these representations gives good retrieval performance using text-based embedding models, including a telecom-domain adapted one. Our approach also alleviates the need for a VLM in inference, which is an important cost benefit for deployed QA systems. 

---
# CoE-Ops: Collaboration of LLM-based Experts for AIOps Question-Answering 

**Authors**: Jinkun Zhao, Yuanshuai Wang, Xingjian Zhang, Ruibo Chen, Xingchuang Liao, Junle Wang, Lei Huang, Kui Zhang, Wenjun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.22937)  

**Abstract**: With the rapid evolution of artificial intelligence, AIOps has emerged as a prominent paradigm in DevOps. Lots of work has been proposed to improve the performance of different AIOps phases. However, constrained by domain-specific knowledge, a single model can only handle the operation requirement of a specific task,such as log parser,root cause analysis. Meanwhile, combining multiple models can achieve more efficient results, which have been proved in both previous ensemble learning and the recent LLM training domain. Inspired by these works,to address the similar challenges in AIOPS, this paper first proposes a collaboration-of-expert framework(CoE-Ops) incorporating a general-purpose large language model task classifier. A retrieval-augmented generation mechanism is introduced to improve the framework's capability in handling both Question-Answering tasks with high-level(Code,build,Test,etc.) and low-level(fault analysis,anomaly detection,etc.). Finally, the proposed method is implemented in the AIOps domain, and extensive experiments are conducted on the DevOps-EVAL dataset. Experimental results demonstrate that CoE-Ops achieves a 72% improvement in routing accuracy for high-level AIOps tasks compared to existing CoE methods, delivers up to 8% accuracy enhancement over single AIOps models in DevOps problem resolution, and outperforms larger-scale Mixture-of-Experts (MoE) models by up to 14% in accuracy. 

---
# Evaluating Large Language Models (LLMs) in Financial NLP: A Comparative Study on Financial Report Analysis 

**Authors**: Md Talha Mohsin  

**Link**: [PDF](https://arxiv.org/pdf/2507.22936)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across a wide variety of Financial Natural Language Processing (FinNLP) tasks. However, systematic comparisons among widely used LLMs remain underexplored. Given the rapid advancement and growing influence of LLMs in financial analysis, this study conducts a thorough comparative evaluation of five leading LLMs, GPT, Claude, Perplexity, Gemini and DeepSeek, using 10-K filings from the 'Magnificent Seven' technology companies. We create a set of domain-specific prompts and then use three methodologies to evaluate model performance: human annotation, automated lexical-semantic metrics (ROUGE, Cosine Similarity, Jaccard), and model behavior diagnostics (prompt-level variance and across-model similarity). The results show that GPT gives the most coherent, semantically aligned, and contextually relevant answers; followed by Claude and Perplexity. Gemini and DeepSeek, on the other hand, have more variability and less agreement. Also, the similarity and stability of outputs change from company to company and over time, showing that they are sensitive to how prompts are written and what source material is used. 

---
# Trusted Knowledge Extraction for Operations and Maintenance Intelligence 

**Authors**: Kathleen Mealey, Jonathan A. Karr Jr., Priscila Saboia Moreira, Paul R. Brenner, Charles F. Vardeman II  

**Link**: [PDF](https://arxiv.org/pdf/2507.22935)  

**Abstract**: Deriving operational intelligence from organizational data repositories is a key challenge due to the dichotomy of data confidentiality vs data integration objectives, as well as the limitations of Natural Language Processing (NLP) tools relative to the specific knowledge structure of domains such as operations and maintenance. In this work, we discuss Knowledge Graph construction and break down the Knowledge Extraction process into its Named Entity Recognition, Coreference Resolution, Named Entity Linking, and Relation Extraction functional components. We then evaluate sixteen NLP tools in concert with or in comparison to the rapidly advancing capabilities of Large Language Models (LLMs). We focus on the operational and maintenance intelligence use case for trusted applications in the aircraft industry. A baseline dataset is derived from a rich public domain US Federal Aviation Administration dataset focused on equipment failures or maintenance requirements. We assess the zero-shot performance of NLP and LLM tools that can be operated within a controlled, confidential environment (no data is sent to third parties). Based on our observation of significant performance limitations, we discuss the challenges related to trusted NLP and LLM tools as well as their Technical Readiness Level for wider use in mission-critical industries such as aviation. We conclude with recommendations to enhance trust and provide our open-source curated dataset to support further baseline testing and evaluation. 

---
# Deep Learning Approaches for Multimodal Intent Recognition: A Survey 

**Authors**: Jingwei Zhao, Yuhua Wen, Qifei Li, Minchi Hu, Yingying Zhou, Jingyao Xue, Junyang Wu, Yingming Gao, Zhengqi Wen, Jianhua Tao, Ya Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.22934)  

**Abstract**: Intent recognition aims to identify users' underlying intentions, traditionally focusing on text in natural language processing. With growing demands for natural human-computer interaction, the field has evolved through deep learning and multimodal approaches, incorporating data from audio, vision, and physiological signals. Recently, the introduction of Transformer-based models has led to notable breakthroughs in this domain. This article surveys deep learning methods for intent recognition, covering the shift from unimodal to multimodal techniques, relevant datasets, methodologies, applications, and current challenges. It provides researchers with insights into the latest developments in multimodal intent recognition (MIR) and directions for future research. 

---
# Augmented Vision-Language Models: A Systematic Review 

**Authors**: Anthony C Davis, Burhan Sadiq, Tianmin Shu, Chien-Ming Huang  

**Link**: [PDF](https://arxiv.org/pdf/2507.22933)  

**Abstract**: Recent advances in visual-language machine learning models have demonstrated exceptional ability to use natural language and understand visual scenes by training on large, unstructured datasets. However, this training paradigm cannot produce interpretable explanations for its outputs, requires retraining to integrate new information, is highly resource-intensive, and struggles with certain forms of logical reasoning. One promising solution involves integrating neural networks with external symbolic information systems, forming neural symbolic systems that can enhance reasoning and memory abilities. These neural symbolic systems provide more interpretable explanations to their outputs and the capacity to assimilate new information without extensive retraining. Utilizing powerful pre-trained Vision-Language Models (VLMs) as the core neural component, augmented by external systems, offers a pragmatic approach to realizing the benefits of neural-symbolic integration. This systematic literature review aims to categorize techniques through which visual-language understanding can be improved by interacting with external symbolic information systems. 

---
# Enhancing RAG Efficiency with Adaptive Context Compression 

**Authors**: Shuyu Guo, Zhaochun Ren  

**Link**: [PDF](https://arxiv.org/pdf/2507.22931)  

**Abstract**: Retrieval-augmented generation (RAG) enhances large language models (LLMs) with external knowledge but incurs significant inference costs due to lengthy retrieved contexts. While context compression mitigates this issue, existing methods apply fixed compression rates, over-compressing simple queries or under-compressing complex ones. We propose Adaptive Context Compression for RAG (ACC-RAG), a framework that dynamically adjusts compression rates based on input complexity, optimizing inference efficiency without sacrificing accuracy. ACC-RAG combines a hierarchical compressor (for multi-granular embeddings) with a context selector to retain minimal sufficient information, akin to human skimming. Evaluated on Wikipedia and five QA datasets, ACC-RAG outperforms fixed-rate methods and matches/unlocks over 4 times faster inference versus standard RAG while maintaining or improving accuracy. 

---
# How does Chain of Thought Think? Mechanistic Interpretability of Chain-of-Thought Reasoning with Sparse Autoencoding 

**Authors**: Xi Chen, Aske Plaat, Niki van Stein  

**Link**: [PDF](https://arxiv.org/pdf/2507.22928)  

**Abstract**: Chain-of-thought (CoT) prompting boosts Large Language Models accuracy on multi-step tasks, yet whether the generated "thoughts" reflect the true internal reasoning process is unresolved. We present the first feature-level causal study of CoT faithfulness. Combining sparse autoencoders with activation patching, we extract monosemantic features from Pythia-70M and Pythia-2.8B while they tackle GSM8K math problems under CoT and plain (noCoT) prompting. Swapping a small set of CoT-reasoning features into a noCoT run raises answer log-probabilities significantly in the 2.8B model, but has no reliable effect in 70M, revealing a clear scale threshold. CoT also leads to significantly higher activation sparsity and feature interpretability scores in the larger model, signalling more modular internal computation. For example, the model's confidence in generating correct answers improves from 1.2 to 4.3. We introduce patch-curves and random-feature patching baselines, showing that useful CoT information is not only present in the top-K patches but widely distributed. Overall, our results indicate that CoT can induce more interpretable internal structures in high-capacity LLMs, validating its role as a structured prompting method. 

---
# Hierarchical Memory for High-Efficiency Long-Term Reasoning in LLM Agents 

**Authors**: Haoran Sun, Shaoning Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2507.22925)  

**Abstract**: Long-term memory is one of the key factors influencing the reasoning capabilities of Large Language Model Agents (LLM Agents). Incorporating a memory mechanism that effectively integrates past interactions can significantly enhance decision-making and contextual coherence of LLM Agents. While recent works have made progress in memory storage and retrieval, such as encoding memory into dense vectors for similarity-based search or organizing knowledge in the form of graph, these approaches often fall short in structured memory organization and efficient retrieval. To address these limitations, we propose a Hierarchical Memory (H-MEM) architecture for LLM Agents that organizes and updates memory in a multi-level fashion based on the degree of semantic abstraction. Each memory vector is embedded with a positional index encoding pointing to its semantically related sub-memories in the next layer. During the reasoning phase, an index-based routing mechanism enables efficient, layer-by-layer retrieval without performing exhaustive similarity computations. We evaluate our method on five task settings from the LoCoMo dataset. Experimental results show that our approach consistently outperforms five baseline methods, demonstrating its effectiveness in long-term dialogue scenarios. 

---
# How and Where to Translate? The Impact of Translation Strategies in Cross-lingual LLM Prompting 

**Authors**: Aman Gupta, Yingying Zhuang, Zhou Yu, Ziji Zhang, Anurag Beniwal  

**Link**: [PDF](https://arxiv.org/pdf/2507.22923)  

**Abstract**: Despite advances in the multilingual capabilities of Large Language Models (LLMs), their performance varies substantially across different languages and tasks. In multilingual retrieval-augmented generation (RAG)-based systems, knowledge bases (KB) are often shared from high-resource languages (such as English) to low-resource ones, resulting in retrieved information from the KB being in a different language than the rest of the context. In such scenarios, two common practices are pre-translation to create a mono-lingual prompt and cross-lingual prompting for direct inference. However, the impact of these choices remains unclear. In this paper, we systematically evaluate the impact of different prompt translation strategies for classification tasks with RAG-enhanced LLMs in multilingual systems. Experimental results show that an optimized prompting strategy can significantly improve knowledge sharing across languages, therefore improve the performance on the downstream classification task. The findings advocate for a broader utilization of multilingual resource sharing and cross-lingual prompt optimization for non-English languages, especially the low-resource ones. 

---
# Predicting stock prices with ChatGPT-annotated Reddit sentiment 

**Authors**: Mateusz Kmak, Kamil Chmurzyński, Kamil Matejuk, Paweł Kotzbach, Jan Kocoń  

**Link**: [PDF](https://arxiv.org/pdf/2507.22922)  

**Abstract**: The surge of retail investor activity on social media, exemplified by the 2021 GameStop short squeeze, raised questions about the influence of online sentiment on stock prices. This paper explores whether sentiment derived from social media discussions can meaningfully predict stock market movements. We focus on Reddit's r/wallstreetbets and analyze sentiment related to two companies: GameStop (GME) and AMC Entertainment (AMC). To assess sentiment's role, we employ two existing text-based sentiment analysis methods and introduce a third, a ChatGPT-annotated and fine-tuned RoBERTa-based model designed to better interpret the informal language and emojis prevalent in social media discussions. We use correlation and causality metrics to determine these models' predictive power. Surprisingly, our findings suggest that social media sentiment has only a weak correlation with stock prices. At the same time, simpler metrics, such as the volume of comments and Google search trends, exhibit stronger predictive signals. These results highlight the complexity of retail investor behavior and suggest that traditional sentiment analysis may not fully capture the nuances of market-moving online discussions. 

---
# Fast and Accurate Contextual Knowledge Extraction Using Cascading Language Model Chains and Candidate Answers 

**Authors**: Lee Harris  

**Link**: [PDF](https://arxiv.org/pdf/2507.22921)  

**Abstract**: Language models can capture complex relationships in given text, but these are notorious for being costly and for producing information that does not exist (i.e., hallucinations). Furthermore, the resources invested into producing this information would be wasted if it were incorrect. We address these issues by proposing, implementing, and applying the Language Model Chain (LMC) algorithm. In this, a language model's response to a given prompt about given text is only correct if it exists in the collection of possible (i.e., candidate) answers, and text corresponding to incorrect responses is fed into a more predictive (but slower) language model. This process is repeated for a collection of language models, or until all predictions about the text are correct. We used the LMC algorithm to extract patient dates of birth from medical documents, and combining a collection of language models in a multi-stage cascade significantly increased prediction speed and accuracy over individual language models, while greatly reducing the number of corresponding hallucinations. We believe that the novel LMC algorithm significantly contributes to the knowledge extraction field, and that this should be explored much further in the future. 

---
# Discrete Tokenization for Multimodal LLMs: A Comprehensive Survey 

**Authors**: Jindong Li, Yali Fu, Jiahong Liu, Linxiao Cao, Wei Ji, Menglin Yang, Irwin King, Ming-Hsuan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.22920)  

**Abstract**: The rapid advancement of large language models (LLMs) has intensified the need for effective mechanisms to transform continuous multimodal data into discrete representations suitable for language-based processing. Discrete tokenization, with vector quantization (VQ) as a central approach, offers both computational efficiency and compatibility with LLM architectures. Despite its growing importance, there is a lack of a comprehensive survey that systematically examines VQ techniques in the context of LLM-based systems. This work fills this gap by presenting the first structured taxonomy and analysis of discrete tokenization methods designed for LLMs. We categorize 8 representative VQ variants that span classical and modern paradigms and analyze their algorithmic principles, training dynamics, and integration challenges with LLM pipelines. Beyond algorithm-level investigation, we discuss existing research in terms of classical applications without LLMs, LLM-based single-modality systems, and LLM-based multimodal systems, highlighting how quantization strategies influence alignment, reasoning, and generation performance. In addition, we identify key challenges including codebook collapse, unstable gradient estimation, and modality-specific encoding constraints. Finally, we discuss emerging research directions such as dynamic and task-adaptive quantization, unified tokenization frameworks, and biologically inspired codebook learning. This survey bridges the gap between traditional vector quantization and modern LLM applications, serving as a foundational reference for the development of efficient and generalizable multimodal systems. A continuously updated version is available at: this https URL. 

---
# A novel language model for predicting serious adverse event results in clinical trials from their prospective registrations 

**Authors**: Qixuan Hu, Xumou Zhang, Jinman Kim, Florence Bourgeois, Adam G. Dunn  

**Link**: [PDF](https://arxiv.org/pdf/2507.22919)  

**Abstract**: Objectives: With accurate estimates of expected safety results, clinical trials could be designed to avoid terminations and limit exposing participants to unnecessary risks. We evaluated methods for predicting serious adverse event (SAE) results in clinical trials using information only from their registrations prior to the trial. Material and Methods: We analysed 22,107 two-arm parallel interventional clinical trials from this http URL with structured summary results. Two prediction models were developed: a classifier predicting will experimental arm have higher SAE rates (area under the receiver operating characteristic curve; AUC) than control arm, and a regression model to predict the proportion of SAEs in control arms (root mean squared error; RMSE). A transfer learning approach using pretrained language models (e.g., ClinicalT5, BioBERT) was used for feature extraction, combined with downstream model for prediction. To maintain semantic representation in long trial texts exceeding localised language model input limits, a sliding window method was developed for embedding extraction. Results: The best model (ClinicalT5+Transformer+MLP) had 77.6% AUC predicting which trial arm has a higher proportion of patients with SAEs. When predicting proportion of participants experiencing SAE in the control arm, the same model achieved RMSE of 18.6%. The sliding window approach consistently outperformed methods without it. Across 12 classifiers, the average absolute AUC increase was 2.00%; across 12 regressors, the average absolute RMSE reduction was 1.58%. Discussion: Summary results data available at this http URL remains underutilised. The potential to estimate results of trials before they start is an opportunity to improve trial design and flag discrepancies between expected and reported safety results. 

---
# Reading Between the Timelines: RAG for Answering Diachronic Questions 

**Authors**: Kwun Hang Lau, Ruiyuan Zhang, Weijie Shi, Xiaofang Zhou, Xiaojun Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.22917)  

**Abstract**: While Retrieval-Augmented Generation (RAG) excels at injecting static, factual knowledge into Large Language Models (LLMs), it exhibits a critical deficit in handling longitudinal queries that require tracking entities and phenomena across time. This blind spot arises because conventional, semantically-driven retrieval methods are not equipped to gather evidence that is both topically relevant and temporally coherent for a specified duration. We address this challenge by proposing a new framework that fundamentally redesigns the RAG pipeline to infuse temporal logic. Our methodology begins by disentangling a user's query into its core subject and its temporal window. It then employs a specialized retriever that calibrates semantic matching against temporal relevance, ensuring the collection of a contiguous evidence set that spans the entire queried period. To enable rigorous evaluation of this capability, we also introduce the Analytical Diachronic Question Answering Benchmark (ADQAB), a challenging evaluation suite grounded in a hybrid corpus of real and synthetic financial news. Empirical results on ADQAB show that our approach yields substantial gains in answer accuracy, surpassing standard RAG implementations by 13% to 27%. This work provides a validated pathway toward RAG systems capable of performing the nuanced, evolutionary analysis required for complex, real-world questions. The dataset and code for this study are publicly available at this https URL. 

---
# From Propagator to Oscillator: The Dual Role of Symmetric Differential Equations in Neural Systems 

**Authors**: Kun Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2507.22916)  

**Abstract**: In our previous work, we proposed a novel neuron model based on symmetric differential equations and demonstrated its potential as an efficient signal propagator. Building upon that foundation, the present study delves deeper into the intrinsic dynamics and functional diversity of this model. By systematically exploring the parameter space and employing a range of mathematical analysis tools, we theoretically reveal the system 's core property of functional duality. Specifically, the model exhibits two distinct trajectory behaviors: one is asymptotically stable, corresponding to a reliable signal propagator; the other is Lyapunov stable, characterized by sustained self-excited oscillations, functioning as a signal generator. To enable effective monitoring and prediction of system states during simulations, we introduce a novel intermediate-state metric termed on-road energy. Simulation results confirm that transitions between the two functional modes can be induced through parameter adjustments or modifications to the connection structure. Moreover, we show that oscillations can be effectively suppressed by introducing external signals. These findings draw a compelling parallel to the dual roles of biological neurons in both information transmission and rhythm generation, thereby establishing a solid theoretical basis and a clear functional roadmap for the broader application of this model in neuromorphic engineering. 

---
# Theoretical Foundations and Mitigation of Hallucination in Large Language Models 

**Authors**: Esmail Gumaan  

**Link**: [PDF](https://arxiv.org/pdf/2507.22915)  

**Abstract**: Hallucination in Large Language Models (LLMs) refers to the generation of content that is not faithful to the input or the real-world facts. This paper provides a rigorous treatment of hallucination in LLMs, including formal definitions and theoretical analyses. We distinguish between intrinsic and extrinsic hallucinations, and define a \textit{hallucination risk} for models. We derive bounds on this risk using learning-theoretic frameworks (PAC-Bayes and Rademacher complexity). We then survey detection strategies for hallucinations, such as token-level uncertainty estimation, confidence calibration, and attention alignment checks. On the mitigation side, we discuss approaches including retrieval-augmented generation, hallucination-aware fine-tuning, logit calibration, and the incorporation of fact-verification modules. We propose a unified detection and mitigation workflow, illustrated with a diagram, to integrate these strategies. Finally, we outline evaluation protocols for hallucination, recommending datasets, metrics, and experimental setups to quantify and reduce hallucinations. Our work lays a theoretical foundation and practical guidelines for addressing the crucial challenge of hallucination in LLMs. 

---
# A Hybrid Framework for Subject Analysis: Integrating Embedding-Based Regression Models with Large Language Models 

**Authors**: Jinyu Liu, Xiaoying Song, Diana Zhang, Jason Thomale, Daqing He, Lingzi Hong  

**Link**: [PDF](https://arxiv.org/pdf/2507.22913)  

**Abstract**: Providing subject access to information resources is an essential function of any library management system. Large language models (LLMs) have been widely used in classification and summarization tasks, but their capability to perform subject analysis is underexplored. Multi-label classification with traditional machine learning (ML) models has been used for subject analysis but struggles with unseen cases. LLMs offer an alternative but often over-generate and hallucinate. Therefore, we propose a hybrid framework that integrates embedding-based ML models with LLMs. This approach uses ML models to (1) predict the optimal number of LCSH labels to guide LLM predictions and (2) post-edit the predicted terms with actual LCSH terms to mitigate hallucinations. We experimented with LLMs and the hybrid framework to predict the subject terms of books using the Library of Congress Subject Headings (LCSH). Experiment results show that providing initial predictions to guide LLM generations and imposing post-edits result in more controlled and vocabulary-aligned outputs. 

---
# A Language Model-Driven Semi-Supervised Ensemble Framework for Illicit Market Detection Across Deep/Dark Web and Social Platforms 

**Authors**: Navid Yazdanjue, Morteza Rakhshaninejad, Hossein Yazdanjouei, Mohammad Sadegh Khorshidi, Mikko S. Niemela, Fang Chen, Amir H. Gandomi  

**Link**: [PDF](https://arxiv.org/pdf/2507.22912)  

**Abstract**: Illegal marketplaces have increasingly shifted to concealed parts of the internet, including the deep and dark web, as well as platforms such as Telegram, Reddit, and Pastebin. These channels enable the anonymous trade of illicit goods including drugs, weapons, and stolen credentials. Detecting and categorizing such content remains challenging due to limited labeled data, the evolving nature of illicit language, and the structural heterogeneity of online sources. This paper presents a hierarchical classification framework that combines fine-tuned language models with a semi-supervised ensemble learning strategy to detect and classify illicit marketplace content across diverse platforms. We extract semantic representations using ModernBERT, a transformer model for long documents, finetuned on domain-specific data from deep and dark web pages, Telegram channels, Subreddits, and Pastebin pastes to capture specialized jargon and ambiguous linguistic patterns. In addition, we incorporate manually engineered features such as document structure, embedded patterns including Bitcoin addresses, emails, and IPs, and metadata, which complement language model embeddings. The classification pipeline operates in two stages. The first stage uses a semi-supervised ensemble of XGBoost, Random Forest, and SVM with entropy-based weighted voting to detect sales-related documents. The second stage further classifies these into drug, weapon, or credential sales. Experiments on three datasets, including our multi-source corpus, DUTA, and CoDA, show that our model outperforms several baselines, including BERT, ModernBERT, DarkBERT, ALBERT, Longformer, and BigBird. The model achieves an accuracy of 0.96489, an F1-score of 0.93467, and a TMCC of 0.95388, demonstrating strong generalization, robustness under limited supervision, and effectiveness in real-world illicit content detection. 

---
# ElectriQ: A Benchmark for Assessing the Response Capability of Large Language Models in Power Marketing 

**Authors**: Jinzhi Wang, Qingke Peng, Haozhou Li, Zeyuan Zeng, Qinfeng Song, Kaixuan Yang, Jiangbo Zhang, Yaoying Wang, Ruimeng Li, Biyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.22911)  

**Abstract**: Electric power marketing customer service plays a critical role in addressing inquiries, complaints, and service requests. However, current systems, such as China's 95598 hotline, often struggle with slow response times, inflexible procedures, and limited accuracy in domain-specific tasks. While large language models (LLMs) like GPT-4o and Claude 3 demonstrate strong general capabilities, they lack the domain expertise and empathy required in this field. To bridge this gap, we introduce ElectriQ, the first benchmark designed to evaluate and enhance LLMs in electric power marketing scenarios. ElectriQ consists of a dialogue dataset covering six key service categories and introduces four evaluation metrics: professionalism, popularity, readability, and user-friendliness. We further incorporate a domain-specific knowledge base and propose a knowledge augmentation method to boost model performance. Experiments on 13 LLMs reveal that smaller models such as LLama3-8B, when fine-tuned and augmented, can surpass GPT-4o in terms of professionalism and user-friendliness. ElectriQ establishes a comprehensive foundation for developing LLMs tailored to the needs of power marketing services. 

---
# Large Language Models in the Travel Domain: An Industrial Experience 

**Authors**: Sergio Di Meglio, Aniello Somma, Luigi Libero Lucio Starace, Fabio Scippacercola, Giancarlo Sperlì, Sergio Di Martino  

**Link**: [PDF](https://arxiv.org/pdf/2507.22910)  

**Abstract**: Online property booking platforms are widely used and rely heavily on consistent, up-to-date information about accommodation facilities, often sourced from third-party providers. However, these external data sources are frequently affected by incomplete or inconsistent details, which can frustrate users and result in a loss of market. In response to these challenges, we present an industrial case study involving the integration of Large Language Models (LLMs) into CALEIDOHOTELS, a property reservation platform developed by FERVENTO. We evaluate two well-known LLMs in this context: Mistral 7B, fine-tuned with QLoRA, and Mixtral 8x7B, utilized with a refined system prompt. Both models were assessed based on their ability to generate consistent and homogeneous descriptions while minimizing hallucinations. Mixtral 8x7B outperformed Mistral 7B in terms of completeness (99.6% vs. 93%), precision (98.8% vs. 96%), and hallucination rate (1.2% vs. 4%), producing shorter yet more concise content (249 vs. 277 words on average). However, this came at a significantly higher computational cost: 50GB VRAM and $1.61/hour versus 5GB and $0.16/hour for Mistral 7B. Our findings provide practical insights into the trade-offs between model quality and resource efficiency, offering guidance for deploying LLMs in production environments and demonstrating their effectiveness in enhancing the consistency and reliability of accommodation data. 

---
# A Privacy-Preserving Federated Framework with Hybrid Quantum-Enhanced Learning for Financial Fraud Detection 

**Authors**: Abhishek Sawaika, Swetang Krishna, Tushar Tomar, Durga Pritam Suggisetti, Aditi Lal, Tanmaya Shrivastav, Nouhaila Innan, Muhammad Shafique  

**Link**: [PDF](https://arxiv.org/pdf/2507.22908)  

**Abstract**: Rapid growth of digital transactions has led to a surge in fraudulent activities, challenging traditional detection methods in the financial sector. To tackle this problem, we introduce a specialised federated learning framework that uniquely combines a quantum-enhanced Long Short-Term Memory (LSTM) model with advanced privacy preserving techniques. By integrating quantum layers into the LSTM architecture, our approach adeptly captures complex cross-transactional patters, resulting in an approximate 5% performance improvement across key evaluation metrics compared to conventional models. Central to our framework is "FedRansel", a novel method designed to defend against poisoning and inference attacks, thereby reducing model degradation and inference accuracy by 4-8%, compared to standard differential privacy mechanisms. This pseudo-centralised setup with a Quantum LSTM model, enhances fraud detection accuracy and reinforces the security and confidentiality of sensitive financial data. 

---
# DNN-based Methods of Jointly Sensing Number and Directions of Targets via a Green Massive H2AD MIMO Receiver 

**Authors**: Bin Deng, Jiatong Bai, Feilong Zhao, Zuming Xie, Maolin Li, Yan Wang, Feng Shu  

**Link**: [PDF](https://arxiv.org/pdf/2507.22906)  

**Abstract**: As a green MIMO structure, the heterogeneous hybrid analog-digital H2AD MIMO architecture has been shown to own a great potential to replace the massive or extremely large-scale fully-digital MIMO in the future wireless networks to address the three challenging problems faced by the latter: high energy consumption, high circuit cost, and high complexity. However, how to intelligently sense the number and direction of multi-emitters via such a structure is still an open hard problem. To address this, we propose a two-stage sensing framework that jointly estimates the number and direction values of multiple targets. Specifically, three target number sensing methods are designed: an improved eigen-domain clustering (EDC) framework, an enhanced deep neural network (DNN) based on five key statistical features, and an improved one-dimensional convolutional neural network (1D-CNN) utilizing full eigenvalues. Subsequently, a low-complexity and high-accuracy DOA estimation is achieved via the introduced online micro-clustering (OMC-DOA) method. Furthermore, we derive the Cramér-Rao lower bound (CRLB) for the H2AD under multiple-source conditions as a theoretical performance benchmark. Simulation results show that the developed three methods achieve 100\% number of targets sensing at moderate-to-high SNRs, while the improved 1D-CNN exhibits superior under extremely-low SNR conditions. The introduced OMC-DOA outperforms existing clustering and fusion-based DOA methods in multi-source environments. 

---
# SketchMind: A Multi-Agent Cognitive Framework for Assessing Student-Drawn Scientific Sketches 

**Authors**: Ehsan Latif, Zirak Khan, Xiaoming Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2507.22904)  

**Abstract**: Scientific sketches (e.g., models) offer a powerful lens into students' conceptual understanding, yet AI-powered automated assessment of such free-form, visually diverse artifacts remains a critical challenge. Existing solutions often treat sketch evaluation as either an image classification task or monolithic vision-language models, which lack interpretability, pedagogical alignment, and adaptability across cognitive levels. To address these limitations, we present SketchMind, a cognitively grounded, multi-agent framework for evaluating and improving student-drawn scientific sketches. SketchMind comprises modular agents responsible for rubric parsing, sketch perception, cognitive alignment, and iterative feedback with sketch modification, enabling personalized and transparent evaluation. We evaluate SketchMind on a curated dataset of 3,575 student-generated sketches across six science assessment items with different highest order of Bloom's level that require students to draw models to explain phenomena. Compared to baseline GPT-4o performance without SRG (average accuracy: 55.6%), and with SRG integration achieves 77.1% average accuracy (+21.4% average absolute gain). We also demonstrate that multi-agent orchestration with SRG enhances SketchMind performance, for example, GPT-4.1 gains an average 8.9% increase in sketch prediction accuracy, outperforming single-agent pipelines across all items. Human evaluators rated the feedback and co-created sketches generated by \textsc{SketchMind} with GPT-4.1, which achieved an average of 4.1 out of 5, significantly higher than those of baseline models (e.g., 2.3 for GPT-4o). Experts noted the system's potential to meaningfully support conceptual growth through guided revision. Our code and (pending approval) dataset will be released to support reproducibility and future research in AI-driven education. 

---
# Toward the Autonomous AI Doctor: Quantitative Benchmarking of an Autonomous Agentic AI Versus Board-Certified Clinicians in a Real World Setting 

**Authors**: Hashim Hayat, Maksim Kudrautsau, Evgeniy Makarov, Vlad Melnichenko, Tim Tsykunou, Piotr Varaksin, Matt Pavelle, Adam Z. Oskowitz  

**Link**: [PDF](https://arxiv.org/pdf/2507.22902)  

**Abstract**: Background: Globally we face a projected shortage of 11 million healthcare practitioners by 2030, and administrative burden consumes 50% of clinical time. Artificial intelligence (AI) has the potential to help alleviate these problems. However, no end-to-end autonomous large language model (LLM)-based AI system has been rigorously evaluated in real-world clinical practice. In this study, we evaluated whether a multi-agent LLM-based AI framework can function autonomously as an AI doctor in a virtual urgent care setting. Methods: We retrospectively compared the performance of the multi-agent AI system Doctronic and board-certified clinicians across 500 consecutive urgent-care telehealth encounters. The primary end points: diagnostic concordance, treatment plan consistency, and safety metrics, were assessed by blinded LLM-based adjudication and expert human review. Results: The top diagnosis of Doctronic and clinician matched in 81% of cases, and the treatment plan aligned in 99.2% of cases. No clinical hallucinations occurred (e.g., diagnosis or treatment not supported by clinical findings). In an expert review of discordant cases, AI performance was superior in 36.1%, and human performance was superior in 9.3%; the diagnoses were equivalent in the remaining cases. Conclusions: In this first large-scale validation of an autonomous AI doctor, we demonstrated strong diagnostic and treatment plan concordance with human clinicians, with AI performance matching and in some cases exceeding that of practicing clinicians. These findings indicate that multi-agent AI systems achieve comparable clinical decision-making to human providers and offer a potential solution to healthcare workforce shortages. 

---
# Tool or Trouble? Exploring Student Attitudes Toward AI Coding Assistants 

**Authors**: Sergio Rojas-Galeano  

**Link**: [PDF](https://arxiv.org/pdf/2507.22900)  

**Abstract**: This exploratory study examines how AI code assistants shape novice programmers' experiences during a two-part exam in an introductory programming course. In the first part, students completed a programming task with access to AI support; in the second, they extended their solutions without AI. We collected Likert-scale and open-ended responses from 20 students to evaluate their perceptions and challenges. Findings suggest that AI tools were perceived as helpful for understanding code and increasing confidence, particularly during initial development. However, students reported difficulties transferring knowledge to unaided tasks, revealing possible overreliance and gaps in conceptual understanding. These insights highlight the need for pedagogical strategies that integrate AI meaningfully while reinforcing foundational programming skills. 

---
# RecUserSim: A Realistic and Diverse User Simulator for Evaluating Conversational Recommender Systems 

**Authors**: Luyu Chen, Quanyu Dai, Zeyu Zhang, Xueyang Feng, Mingyu Zhang, Pengcheng Tang, Xu Chen, Yue Zhu, Zhenhua Dong  

**Link**: [PDF](https://arxiv.org/pdf/2507.22897)  

**Abstract**: Conversational recommender systems (CRS) enhance user experience through multi-turn interactions, yet evaluating CRS remains challenging. User simulators can provide comprehensive evaluations through interactions with CRS, but building realistic and diverse simulators is difficult. While recent work leverages large language models (LLMs) to simulate user interactions, they still fall short in emulating individual real users across diverse scenarios and lack explicit rating mechanisms for quantitative evaluation. To address these gaps, we propose RecUserSim, an LLM agent-based user simulator with enhanced simulation realism and diversity while providing explicit scores. RecUserSim features several key modules: a profile module for defining realistic and diverse user personas, a memory module for tracking interaction history and discovering unknown preferences, and a core action module inspired by Bounded Rationality theory that enables nuanced decision-making while generating more fine-grained actions and personalized responses. To further enhance output control, a refinement module is designed to fine-tune final responses. Experiments demonstrate that RecUserSim generates diverse, controllable outputs and produces realistic, high-quality dialogues, even with smaller base LLMs. The ratings generated by RecUserSim show high consistency across different base LLMs, highlighting its effectiveness for CRS evaluation. 

---
# iLearnRobot: An Interactive Learning-Based Multi-Modal Robot with Continuous Improvement 

**Authors**: Kohou Wang, ZhaoXiang Liu, Lin Bai, Kun Fan, Xiang Liu, Huan Hu, Kai Wang, Shiguo Lian  

**Link**: [PDF](https://arxiv.org/pdf/2507.22896)  

**Abstract**: It is crucial that robots' performance can be improved after deployment, as they are inherently likely to encounter novel scenarios never seen before. This paper presents an innovative solution: an interactive learning-based robot system powered by a Multi-modal Large Language Model(MLLM). A key feature of our system is its ability to learn from natural dialogues with non-expert users. We also propose chain of question to clarify the exact intent of the question before providing an answer and dual-modality retrieval modules to leverage these interaction events to avoid repeating same mistakes, ensuring a seamless user experience before model updates, which is in contrast to current mainstream MLLM-based robotic systems. Our system marks a novel approach in robotics by integrating interactive learning, paving the way for superior adaptability and performance in diverse environments. We demonstrate the effectiveness and improvement of our method through experiments, both quantitively and qualitatively. 

---
# Invisible Architectures of Thought: Toward a New Science of AI as Cognitive Infrastructure 

**Authors**: Giuseppe Riva  

**Link**: [PDF](https://arxiv.org/pdf/2507.22893)  

**Abstract**: Contemporary human-AI interaction research overlooks how AI systems fundamentally reshape human cognition pre-consciously, a critical blind spot for understanding distributed cognition. This paper introduces "Cognitive Infrastructure Studies" (CIS) as a new interdisciplinary domain to reconceptualize AI as "cognitive infrastructures": foundational, often invisible systems conditioning what is knowable and actionable in digital societies. These semantic infrastructures transport meaning, operate through anticipatory personalization, and exhibit adaptive invisibility, making their influence difficult to detect. Critically, they automate "relevance judgment," shifting the "locus of epistemic agency" to non-human systems. Through narrative scenarios spanning individual (cognitive dependency), collective (democratic deliberation), and societal (governance) scales, we describe how cognitive infrastructures reshape human cognition, public reasoning, and social epistemologies. CIS aims to address how AI preprocessing reshapes distributed cognition across individual, collective, and cultural scales, requiring unprecedented integration of diverse disciplinary methods. The framework also addresses critical gaps across disciplines: cognitive science lacks population-scale preprocessing analysis capabilities, digital sociology cannot access individual cognitive mechanisms, and computational approaches miss cultural transmission dynamics. To achieve this goal CIS also provides methodological innovations for studying invisible algorithmic influence: "infrastructure breakdown methodologies", experimental approaches that reveal cognitive dependencies by systematically withdrawing AI preprocessing after periods of habituation. 

---
# Evaluating LLMs for Visualization Generation and Understanding 

**Authors**: Saadiq Rauf Khan, Vinit Chandak, Sougata Mukherjea  

**Link**: [PDF](https://arxiv.org/pdf/2507.22890)  

**Abstract**: Information Visualization has been utilized to gain insights from complex data. In recent times, Large Language models (LLMs) have performed very well in many tasks. In this paper, we showcase the capabilities of different popular LLMs to generate code for visualization based on simple prompts. We also analyze the power of LLMs to understand some common visualizations by answering questions. Our study shows that LLMs could generate code for some simpler visualizations such as bar and pie charts. Moreover, they could answer simple questions about visualizations. However, LLMs also have several limitations. For example, some of them had difficulty generating complex visualizations, such as violin plot. LLMs also made errors in answering some questions about visualizations, for example, identifying relationships between close boundaries and determining lengths of shapes. We believe that our insights can be used to improve both LLMs and Information Visualization systems. 

---
# OAEI-LLM-T: A TBox Benchmark Dataset for Understanding Large Language Model Hallucinations in Ontology Matching 

**Authors**: Zhangcheng Qiang, Kerry Taylor, Weiqing Wang, Jing Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.21813)  

**Abstract**: Hallucinations are often inevitable in downstream tasks using large language models (LLMs). To tackle the substantial challenge of addressing hallucinations for LLM-based ontology matching (OM) systems, we introduce a new benchmark dataset OAEI-LLM-T. The dataset evolves from seven TBox datasets in the Ontology Alignment Evaluation Initiative (OAEI), capturing hallucinations of ten different LLMs performing OM tasks. These OM-specific hallucinations are organised into two primary categories and six sub-categories. We showcase the usefulness of the dataset in constructing an LLM leaderboard for OM tasks and for fine-tuning LLMs used in OM tasks. 

---
