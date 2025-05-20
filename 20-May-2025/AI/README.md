# Trust, But Verify: A Self-Verification Approach to Reinforcement Learning with Verifiable Rewards 

**Authors**: Xiaoyuan Liu, Tian Liang, Zhiwei He, Jiahao Xu, Wenxuan Wang, Pinjia He, Zhaopeng Tu, Haitao Mi, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13445)  

**Abstract**: Large Language Models (LLMs) show great promise in complex reasoning, with Reinforcement Learning with Verifiable Rewards (RLVR) being a key enhancement strategy. However, a prevalent issue is ``superficial self-reflection'', where models fail to robustly verify their own outputs. We introduce RISE (Reinforcing Reasoning with Self-Verification), a novel online RL framework designed to tackle this. RISE explicitly and simultaneously trains an LLM to improve both its problem-solving and self-verification abilities within a single, integrated RL process. The core mechanism involves leveraging verifiable rewards from an outcome verifier to provide on-the-fly feedback for both solution generation and self-verification tasks. In each iteration, the model generates solutions, then critiques its own on-policy generated solutions, with both trajectories contributing to the policy update. Extensive experiments on diverse mathematical reasoning benchmarks show that RISE consistently improves model's problem-solving accuracy while concurrently fostering strong self-verification skills. Our analyses highlight the advantages of online verification and the benefits of increased verification compute. Additionally, RISE models exhibit more frequent and accurate self-verification behaviors during reasoning. These advantages reinforce RISE as a flexible and effective path towards developing more robust and self-aware reasoners. 

---
# MM-PRM: Enhancing Multimodal Mathematical Reasoning with Scalable Step-Level Supervision 

**Authors**: Lingxiao Du, Fanqing Meng, Zongkai Liu, Zhixiang Zhou, Ping Luo, Qiaosheng Zhang, Wenqi Shao  

**Link**: [PDF](https://arxiv.org/pdf/2505.13427)  

**Abstract**: While Multimodal Large Language Models (MLLMs) have achieved impressive progress in vision-language understanding, they still struggle with complex multi-step reasoning, often producing logically inconsistent or partially correct solutions. A key limitation lies in the lack of fine-grained supervision over intermediate reasoning steps. To address this, we propose MM-PRM, a process reward model trained within a fully automated, scalable framework. We first build MM-Policy, a strong multimodal model trained on diverse mathematical reasoning data. Then, we construct MM-K12, a curated dataset of 10,000 multimodal math problems with verifiable answers, which serves as seed data. Leveraging a Monte Carlo Tree Search (MCTS)-based pipeline, we generate over 700k step-level annotations without human labeling. The resulting PRM is used to score candidate reasoning paths in the Best-of-N inference setup and achieves significant improvements across both in-domain (MM-K12 test set) and out-of-domain (OlympiadBench, MathVista, etc.) benchmarks. Further analysis confirms the effectiveness of soft labels, smaller learning rates, and path diversity in optimizing PRM performance. MM-PRM demonstrates that process supervision is a powerful tool for enhancing the logical robustness of multimodal reasoning systems. We release all our codes and data at this https URL. 

---
# CoT-Kinetics: A Theoretical Modeling Assessing LRM Reasoning Process 

**Authors**: Jinhe Bi, Danqi Yan, Yifan Wang, Wenke Huang, Haokun Chen, Guancheng Wan, Mang Ye, Xun Xiao, Hinrich Schuetze, Volker Tresp, Yunpu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.13408)  

**Abstract**: Recent Large Reasoning Models significantly improve the reasoning ability of Large Language Models by learning to reason, exhibiting the promising performance in solving complex tasks. LRMs solve tasks that require complex reasoning by explicitly generating reasoning trajectories together with answers. Nevertheless, judging the quality of such an output answer is not easy because only considering the correctness of the answer is not enough and the soundness of the reasoning trajectory part matters as well. Logically, if the soundness of the reasoning part is poor, even if the answer is correct, the confidence of the derived answer should be low. Existing methods did consider jointly assessing the overall output answer by taking into account the reasoning part, however, their capability is still not satisfactory as the causal relationship of the reasoning to the concluded answer cannot properly reflected. In this paper, inspired by classical mechanics, we present a novel approach towards establishing a CoT-Kinetics energy equation. Specifically, our CoT-Kinetics energy equation formulates the token state transformation process, which is regulated by LRM internal transformer layers, as like a particle kinetics dynamics governed in a mechanical field. Our CoT-Kinetics energy assigns a scalar score to evaluate specifically the soundness of the reasoning phase, telling how confident the derived answer could be given the evaluated reasoning. As such, the LRM's overall output quality can be accurately measured, rather than a coarse judgment (e.g., correct or incorrect) anymore. 

---
# AutoMathKG: The automated mathematical knowledge graph based on LLM and vector database 

**Authors**: Rong Bian, Yu Geng, Zijian Yang, Bing Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.13406)  

**Abstract**: A mathematical knowledge graph (KG) presents knowledge within the field of mathematics in a structured manner. Constructing a math KG using natural language is an essential but challenging task. There are two major limitations of existing works: first, they are constrained by corpus completeness, often discarding or manually supplementing incomplete knowledge; second, they typically fail to fully automate the integration of diverse knowledge sources. This paper proposes AutoMathKG, a high-quality, wide-coverage, and multi-dimensional math KG capable of automatic updates. AutoMathKG regards mathematics as a vast directed graph composed of Definition, Theorem, and Problem entities, with their reference relationships as edges. It integrates knowledge from ProofWiki, textbooks, arXiv papers, and TheoremQA, enhancing entities and relationships with large language models (LLMs) via in-context learning for data augmentation. To search for similar entities, MathVD, a vector database, is built through two designed embedding strategies using SBERT. To automatically update, two mechanisms are proposed. For knowledge completion mechanism, Math LLM is developed to interact with AutoMathKG, providing missing proofs or solutions. For knowledge fusion mechanism, MathVD is used to retrieve similar entities, and LLM is used to determine whether to merge with a candidate or add as a new entity. A wide range of experiments demonstrate the advanced performance and broad applicability of the AutoMathKG system, including superior reachability query results in MathVD compared to five baselines and robust mathematical reasoning capability in Math LLM. 

---
# Robin: A multi-agent system for automating scientific discovery 

**Authors**: Ali Essam Ghareeb, Benjamin Chang, Ludovico Mitchener, Angela Yiu, Caralyn J. Szostkiewicz, Jon M. Laurent, Muhammed T. Razzak, Andrew D. White, Michaela M. Hinks, Samuel G. Rodriques  

**Link**: [PDF](https://arxiv.org/pdf/2505.13400)  

**Abstract**: Scientific discovery is driven by the iterative process of background research, hypothesis generation, experimentation, and data analysis. Despite recent advancements in applying artificial intelligence to scientific discovery, no system has yet automated all of these stages in a single workflow. Here, we introduce Robin, the first multi-agent system capable of fully automating the key intellectual steps of the scientific process. By integrating literature search agents with data analysis agents, Robin can generate hypotheses, propose experiments, interpret experimental results, and generate updated hypotheses, achieving a semi-autonomous approach to scientific discovery. By applying this system, we were able to identify a novel treatment for dry age-related macular degeneration (dAMD), the major cause of blindness in the developed world. Robin proposed enhancing retinal pigment epithelium phagocytosis as a therapeutic strategy, and identified and validated a promising therapeutic candidate, ripasudil. Ripasudil is a clinically-used rho kinase (ROCK) inhibitor that has never previously been proposed for treating dAMD. To elucidate the mechanism of ripasudil-induced upregulation of phagocytosis, Robin then proposed and analyzed a follow-up RNA-seq experiment, which revealed upregulation of ABCA1, a critical lipid efflux pump and possible novel target. All hypotheses, experimental plans, data analyses, and data figures in the main text of this report were produced by Robin. As the first AI system to autonomously discover and validate a novel therapeutic candidate within an iterative lab-in-the-loop framework, Robin establishes a new paradigm for AI-driven scientific discovery. 

---
# Advancing Generalization Across a Variety of Abstract Visual Reasoning Tasks 

**Authors**: Mikołaj Małkiński, Jacek Mańdziuk  

**Link**: [PDF](https://arxiv.org/pdf/2505.13391)  

**Abstract**: The abstract visual reasoning (AVR) domain presents a diverse suite of analogy-based tasks devoted to studying model generalization. Recent years have brought dynamic progress in the field, particularly in i.i.d. scenarios, in which models are trained and evaluated on the same data distributions. Nevertheless, o.o.d. setups that assess model generalization to new test distributions remain challenging even for the most recent models. To advance generalization in AVR tasks, we present the Pathways of Normalized Group Convolution model (PoNG), a novel neural architecture that features group convolution, normalization, and a parallel design. We consider a wide set of AVR benchmarks, including Raven's Progressive Matrices and visual analogy problems with both synthetic and real-world images. The experiments demonstrate strong generalization capabilities of the proposed model, which in several settings outperforms the existing literature methods. 

---
# CompeteSMoE -- Statistically Guaranteed Mixture of Experts Training via Competition 

**Authors**: Nam V. Nguyen, Huy Nguyen, Quang Pham, Van Nguyen, Savitha Ramasamy, Nhat Ho  

**Link**: [PDF](https://arxiv.org/pdf/2505.13380)  

**Abstract**: Sparse mixture of experts (SMoE) offers an appealing solution to scale up the model complexity beyond the mean of increasing the network's depth or width. However, we argue that effective SMoE training remains challenging because of the suboptimal routing process where experts that perform computation do not directly contribute to the routing process. In this work, we propose competition, a novel mechanism to route tokens to experts with the highest neural response. Theoretically, we show that the competition mechanism enjoys a better sample efficiency than the traditional softmax routing. Furthermore, we develop CompeteSMoE, a simple yet effective algorithm to train large language models by deploying a router to learn the competition policy, thus enjoying strong performances at a low training overhead. Our extensive empirical evaluations on both the visual instruction tuning and language pre-training tasks demonstrate the efficacy, robustness, and scalability of CompeteSMoE compared to state-of-the-art SMoE strategies. We have made the implementation available at: this https URL. This work is an improved version of the previous study at arXiv:2402.02526 

---
# Exploiting Symbolic Heuristics for the Synthesis of Domain-Specific Temporal Planning Guidance using Reinforcement Learning 

**Authors**: Irene Brugnara, Alessandro Valentini, Andrea Micheli  

**Link**: [PDF](https://arxiv.org/pdf/2505.13372)  

**Abstract**: Recent work investigated the use of Reinforcement Learning (RL) for the synthesis of heuristic guidance to improve the performance of temporal planners when a domain is fixed and a set of training problems (not plans) is given. The idea is to extract a heuristic from the value function of a particular (possibly infinite-state) MDP constructed over the training problems.
In this paper, we propose an evolution of this learning and planning framework that focuses on exploiting the information provided by symbolic heuristics during both the RL and planning phases. First, we formalize different reward schemata for the synthesis and use symbolic heuristics to mitigate the problems caused by the truncation of episodes needed to deal with the potentially infinite MDP. Second, we propose learning a residual of an existing symbolic heuristic, which is a "correction" of the heuristic value, instead of eagerly learning the whole heuristic from scratch. Finally, we use the learned heuristic in combination with a symbolic heuristic using a multiple-queue planning approach to balance systematic search with imperfect learned information. We experimentally compare all the approaches, highlighting their strengths and weaknesses and significantly advancing the state of the art for this planning and learning schema. 

---
# Multi-Armed Bandits Meet Large Language Models 

**Authors**: Djallel Bouneffouf, Raphael Feraud  

**Link**: [PDF](https://arxiv.org/pdf/2505.13355)  

**Abstract**: Bandit algorithms and Large Language Models (LLMs) have emerged as powerful tools in artificial intelligence, each addressing distinct yet complementary challenges in decision-making and natural language processing. This survey explores the synergistic potential between these two fields, highlighting how bandit algorithms can enhance the performance of LLMs and how LLMs, in turn, can provide novel insights for improving bandit-based decision-making. We first examine the role of bandit algorithms in optimizing LLM fine-tuning, prompt engineering, and adaptive response generation, focusing on their ability to balance exploration and exploitation in large-scale learning tasks. Subsequently, we explore how LLMs can augment bandit algorithms through advanced contextual understanding, dynamic adaptation, and improved policy selection using natural language reasoning. By providing a comprehensive review of existing research and identifying key challenges and opportunities, this survey aims to bridge the gap between bandit algorithms and LLMs, paving the way for innovative applications and interdisciplinary research in AI. 

---
# Level Generation with Quantum Reservoir Computing 

**Authors**: João S. Ferreira, Pierre Fromholz, Hari Shaji, James R. Wootton  

**Link**: [PDF](https://arxiv.org/pdf/2505.13287)  

**Abstract**: Reservoir computing is a form of machine learning particularly suited for time series analysis, including forecasting predictions. We take an implementation of \emph{quantum} reservoir computing that was initially designed to generate variants of musical scores and adapt it to create levels of Super Mario Bros. Motivated by our analysis of these levels, we develop a new Roblox \textit{obby} where the courses can be generated in real time on superconducting qubit hardware, and investigate some of the constraints placed by such real-time generation. 

---
# Seeing the Unseen: How EMoE Unveils Bias in Text-to-Image Diffusion Models 

**Authors**: Lucas Berry, Axel Brando, Wei-Di Chang, Juan Camilo Gamboa Higuera, David Meger  

**Link**: [PDF](https://arxiv.org/pdf/2505.13273)  

**Abstract**: Estimating uncertainty in text-to-image diffusion models is challenging because of their large parameter counts (often exceeding 100 million) and operation in complex, high-dimensional spaces with virtually infinite input possibilities. In this paper, we propose Epistemic Mixture of Experts (EMoE), a novel framework for efficiently estimating epistemic uncertainty in diffusion models. EMoE leverages pre-trained networks without requiring additional training, enabling direct uncertainty estimation from a prompt. We leverage a latent space within the diffusion process that captures epistemic uncertainty better than existing methods. Experimental results on the COCO dataset demonstrate EMoE's effectiveness, showing a strong correlation between uncertainty and image quality. Additionally, EMoE identifies under-sampled languages and regions with higher uncertainty, revealing hidden biases in the training set. This capability demonstrates the relevance of EMoE as a tool for addressing fairness and accountability in AI-generated content. 

---
# Agentic Publications: An LLM-Driven Framework for Interactive Scientific Publishing, Supplementing Traditional Papers with AI-Powered Knowledge Systems 

**Authors**: Roberto Pugliese, George Kourousias, Francesco Venier, Grazia Garlatti Costa  

**Link**: [PDF](https://arxiv.org/pdf/2505.13246)  

**Abstract**: The exponential growth of scientific literature presents significant challenges for researchers navigating the complex knowledge landscape. We propose "Agentic Publications", a novel LLM-driven framework complementing traditional publishing by transforming papers into interactive knowledge systems. Our architecture integrates structured data with unstructured content through retrieval-augmented generation and multi-agent verification. The framework offers interfaces for both humans and machines, combining narrative explanations with machine-readable outputs while addressing ethical considerations through automated validation and transparent governance. Key features include continuous knowledge updates, automatic integration of new findings, and customizable detail levels. Our proof-of-concept demonstrates multilingual interaction, API accessibility, and structured knowledge representation through vector databases, knowledge graphs, and verification agents. This approach enhances scientific communication across disciplines, improving efficiency and collaboration while preserving traditional publishing pathways, particularly valuable for interdisciplinary fields where knowledge integration remains challenging. 

---
# StarFT: Robust Fine-tuning of Zero-shot Models via Spuriosity Alignment 

**Authors**: Younghyun Kim, Jongheon Jeong, Sangkyung Kwak, Kyungmin Lee, Juho Lee, Jinwoo Shin  

**Link**: [PDF](https://arxiv.org/pdf/2505.13232)  

**Abstract**: Learning robust representations from data often requires scale, which has led to the success of recent zero-shot models such as CLIP. However, the obtained robustness can easily be deteriorated when these models are fine-tuned on other downstream tasks (e.g., of smaller scales). Previous works often interpret this phenomenon in the context of domain shift, developing fine-tuning methods that aim to preserve the original domain as much as possible. However, in a different context, fine-tuned models with limited data are also prone to learning features that are spurious to humans, such as background or texture. In this paper, we propose StarFT (Spurious Textual Alignment Regularization), a novel framework for fine-tuning zero-shot models to enhance robustness by preventing them from learning spuriosity. We introduce a regularization that aligns the output distribution for spuriosity-injected labels with the original zero-shot model, ensuring that the model is not induced to extract irrelevant features further from these this http URL leverage recent language models to get such spuriosity-injected labels by generating alternative textual descriptions that highlight potentially confounding this http URL experiments validate the robust generalization of StarFT and its emerging properties: zero-shot group robustness and improved zero-shot classification. Notably, StarFT boosts both worst-group and average accuracy by 14.30% and 3.02%, respectively, in the Waterbirds group shift scenario, where other robust fine-tuning baselines show even degraded performance. 

---
# Scaling Computer-Use Grounding via User Interface Decomposition and Synthesis 

**Authors**: Tianbao Xie, Jiaqi Deng, Xiaochuan Li, Junlin Yang, Haoyuan Wu, Jixuan Chen, Wenjing Hu, Xinyuan Wang, Yuhui Xu, Zekun Wang, Yiheng Xu, Junli Wang, Doyen Sahoo, Tao Yu, Caiming Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2505.13227)  

**Abstract**: Graphical user interface (GUI) grounding, the ability to map natural language instructions to specific actions on graphical user interfaces, remains a critical bottleneck in computer use agent development. Current benchmarks oversimplify grounding tasks as short referring expressions, failing to capture the complexity of real-world interactions that require software commonsense, layout understanding, and fine-grained manipulation capabilities. To address these limitations, we introduce OSWorld-G, a comprehensive benchmark comprising 564 finely annotated samples across diverse task types including text matching, element recognition, layout understanding, and precise manipulation. Additionally, we synthesize and release the largest computer use grounding dataset Jedi, which contains 4 million examples through multi-perspective decoupling of tasks. Our multi-scale models trained on Jedi demonstrate its effectiveness by outperforming existing approaches on ScreenSpot-v2, ScreenSpot-Pro, and our OSWorld-G. Furthermore, we demonstrate that improved grounding with Jedi directly enhances agentic capabilities of general foundation models on complex computer tasks, improving from 5% to 27% on OSWorld. Through detailed ablation studies, we identify key factors contributing to grounding performance and verify that combining specialized data for different interface elements enables compositional generalization to novel interfaces. All benchmark, data, checkpoints, and code are open-sourced and available at this https URL. 

---
# Adversarial Testing in LLMs: Insights into Decision-Making Vulnerabilities 

**Authors**: Lili Zhang, Haomiaomiao Wang, Long Cheng, Libao Deng, Tomas Ward  

**Link**: [PDF](https://arxiv.org/pdf/2505.13195)  

**Abstract**: As Large Language Models (LLMs) become increasingly integrated into real-world decision-making systems, understanding their behavioural vulnerabilities remains a critical challenge for AI safety and alignment. While existing evaluation metrics focus primarily on reasoning accuracy or factual correctness, they often overlook whether LLMs are robust to adversarial manipulation or capable of using adaptive strategy in dynamic environments. This paper introduces an adversarial evaluation framework designed to systematically stress-test the decision-making processes of LLMs under interactive and adversarial conditions. Drawing on methodologies from cognitive psychology and game theory, our framework probes how models respond in two canonical tasks: the two-armed bandit task and the Multi-Round Trust Task. These tasks capture key aspects of exploration-exploitation trade-offs, social cooperation, and strategic flexibility. We apply this framework to several state-of-the-art LLMs, including GPT-3.5, GPT-4, Gemini-1.5, and DeepSeek-V3, revealing model-specific susceptibilities to manipulation and rigidity in strategy adaptation. Our findings highlight distinct behavioral patterns across models and emphasize the importance of adaptability and fairness recognition for trustworthy AI deployment. Rather than offering a performance benchmark, this work proposes a methodology for diagnosing decision-making weaknesses in LLM-based agents, providing actionable insights for alignment and safety research. 

---
# ViPlan: A Benchmark for Visual Planning with Symbolic Predicates and Vision-Language Models 

**Authors**: Matteo Merler, Nicola Dainese, Minttu Alakuijala, Giovanni Bonetta, Pietro Ferrazzi, Yu Tian, Bernardo Magnini, Pekka Marttinen  

**Link**: [PDF](https://arxiv.org/pdf/2505.13180)  

**Abstract**: Integrating Large Language Models with symbolic planners is a promising direction for obtaining verifiable and grounded plans compared to planning in natural language, with recent works extending this idea to visual domains using Vision-Language Models (VLMs). However, rigorous comparison between VLM-grounded symbolic approaches and methods that plan directly with a VLM has been hindered by a lack of common environments, evaluation protocols and model coverage. We introduce ViPlan, the first open-source benchmark for Visual Planning with symbolic predicates and VLMs. ViPlan features a series of increasingly challenging tasks in two domains: a visual variant of the classic Blocksworld planning problem and a simulated household robotics environment. We benchmark nine open-source VLM families across multiple sizes, along with selected closed models, evaluating both VLM-grounded symbolic planning and using the models directly to propose actions. We find symbolic planning to outperform direct VLM planning in Blocksworld, where accurate image grounding is crucial, whereas the opposite is true in the household robotics tasks, where commonsense knowledge and the ability to recover from errors are beneficial. Finally, we show that across most models and methods, there is no significant benefit to using Chain-of-Thought prompting, suggesting that current VLMs still struggle with visual reasoning. 

---
# Enhancing LLMs for Time Series Forecasting via Structure-Guided Cross-Modal Alignment 

**Authors**: Siming Sun, Kai Zhang, Xuejun Jiang, Wenchao Meng, Qinmin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13175)  

**Abstract**: The emerging paradigm of leveraging pretrained large language models (LLMs) for time series forecasting has predominantly employed linguistic-temporal modality alignment strategies through token-level or layer-wise feature mapping. However, these approaches fundamentally neglect a critical insight: the core competency of LLMs resides not merely in processing localized token features but in their inherent capacity to model holistic sequence structures. This paper posits that effective cross-modal alignment necessitates structural consistency at the sequence level. We propose the Structure-Guided Cross-Modal Alignment (SGCMA), a framework that fully exploits and aligns the state-transition graph structures shared by time-series and linguistic data as sequential modalities, thereby endowing time series with language-like properties and delivering stronger generalization after modality alignment. SGCMA consists of two key components, namely Structure Alignment and Semantic Alignment. In Structure Alignment, a state transition matrix is learned from text data through Hidden Markov Models (HMMs), and a shallow transformer-based Maximum Entropy Markov Model (MEMM) receives the hot-start transition matrix and annotates each temporal patch into state probability, ensuring that the temporal representation sequence inherits language-like sequential dynamics. In Semantic Alignment, cross-attention is applied between temporal patches and the top-k tokens within each state, and the ultimate temporal embeddings are derived by the expected value of these embeddings using a weighted average based on state probabilities. Experiments on multiple benchmarks demonstrate that SGCMA achieves state-of-the-art performance, offering a novel approach to cross-modal alignment in time series forecasting. 

---
# Zero-Shot Iterative Formalization and Planning in Partially Observable Environments 

**Authors**: Liancheng Gong, Wang Zhu, Jesse Thomason, Li Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13126)  

**Abstract**: In planning, using LLMs not to predict plans but to formalize an environment into the Planning Domain Definition Language (PDDL) has been shown to greatly improve performance and control. While most work focused on fully observable environments, we tackle the more realistic and challenging partially observable environments where existing methods are incapacitated by the lack of complete information. We propose PDDLego+, a framework to iteratively formalize, plan, grow, and refine PDDL representations in a zero-shot manner, without needing access to any existing trajectories. On two textual simulated environments, we show that PDDLego+ not only achieves superior performance, but also shows robustness against problem complexity. We also show that the domain knowledge captured after a successful trial is interpretable and benefits future tasks. 

---
# Unveil Sources of Uncertainty: Feature Contribution to Conformal Prediction Intervals 

**Authors**: Marouane Il Idrissi, Agathe Fernandes Machado, Ewen Gallic, Arthur Charpentier  

**Link**: [PDF](https://arxiv.org/pdf/2505.13118)  

**Abstract**: Cooperative game theory methods, notably Shapley values, have significantly enhanced machine learning (ML) interpretability. However, existing explainable AI (XAI) frameworks mainly attribute average model predictions, overlooking predictive uncertainty. This work addresses that gap by proposing a novel, model-agnostic uncertainty attribution (UA) method grounded in conformal prediction (CP). By defining cooperative games where CP interval properties-such as width and bounds-serve as value functions, we systematically attribute predictive uncertainty to input features. Extending beyond the traditional Shapley values, we use the richer class of Harsanyi allocations, and in particular the proportional Shapley values, which distribute attribution proportionally to feature importance. We propose a Monte Carlo approximation method with robust statistical guarantees to address computational feasibility, significantly improving runtime efficiency. Our comprehensive experiments on synthetic benchmarks and real-world datasets demonstrate the practical utility and interpretative depth of our approach. By combining cooperative game theory and conformal prediction, we offer a rigorous, flexible toolkit for understanding and communicating predictive uncertainty in high-stakes ML applications. 

---
# LLM-KG-Bench 3.0: A Compass for SemanticTechnology Capabilities in the Ocean of LLMs 

**Authors**: Lars-Peter Meyer, Johannes Frey, Desiree Heim, Felix Brei, Claus Stadler, Kurt Junghanns, Michael Martin  

**Link**: [PDF](https://arxiv.org/pdf/2505.13098)  

**Abstract**: Current Large Language Models (LLMs) can assist developing program code beside many other things, but can they support working with Knowledge Graphs (KGs) as well? Which LLM is offering the best capabilities in the field of Semantic Web and Knowledge Graph Engineering (KGE)? Is this possible to determine without checking many answers manually? The LLM-KG-Bench framework in Version 3.0 is designed to answer these questions. It consists of an extensible set of tasks for automated evaluation of LLM answers and covers different aspects of working with semantic technologies. In this paper the LLM-KG-Bench framework is presented in Version 3 along with a dataset of prompts, answers and evaluations generated with it and several state-of-the-art LLMs. Significant enhancements have been made to the framework since its initial release, including an updated task API that offers greater flexibility in handling evaluation tasks, revised tasks, and extended support for various open models through the vllm library, among other improvements. A comprehensive dataset has been generated using more than 30 contemporary open and proprietary LLMs, enabling the creation of exemplary model cards that demonstrate the models' capabilities in working with RDF and SPARQL, as well as comparing their performance on Turtle and JSON-LD RDF serialization tasks. 

---
# CAIM: Development and Evaluation of a Cognitive AI Memory Framework for Long-Term Interaction with Intelligent Agents 

**Authors**: Rebecca Westhäußer, Frederik Berenz, Wolfgang Minker, Sebastian Zepf  

**Link**: [PDF](https://arxiv.org/pdf/2505.13044)  

**Abstract**: Large language models (LLMs) have advanced the field of artificial intelligence (AI) and are a powerful enabler for interactive systems. However, they still face challenges in long-term interactions that require adaptation towards the user as well as contextual knowledge and understanding of the ever-changing environment. To overcome these challenges, holistic memory modeling is required to efficiently retrieve and store relevant information across interaction sessions for suitable responses. Cognitive AI, which aims to simulate the human thought process in a computerized model, highlights interesting aspects, such as thoughts, memory mechanisms, and decision-making, that can contribute towards improved memory modeling for LLMs. Inspired by these cognitive AI principles, we propose our memory framework CAIM. CAIM consists of three modules: 1.) The Memory Controller as the central decision unit; 2.) the Memory Retrieval, which filters relevant data for interaction upon request; and 3.) the Post-Thinking, which maintains the memory storage. We compare CAIM against existing approaches, focusing on metrics such as retrieval accuracy, response correctness, contextual coherence, and memory storage. The results demonstrate that CAIM outperforms baseline frameworks across different metrics, highlighting its context-awareness and potential to improve long-term human-AI interactions. 

---
# MindOmni: Unleashing Reasoning Generation in Vision Language Models with RGPO 

**Authors**: Yicheng Xiao, Lin Song, Yukang Chen, Yingmin Luo, Yuxin Chen, Yukang Gan, Wei Huang, Xiu Li, Xiaojuan Qi, Ying Shan  

**Link**: [PDF](https://arxiv.org/pdf/2505.13031)  

**Abstract**: Recent text-to-image systems face limitations in handling multimodal inputs and complex reasoning tasks. We introduce MindOmni, a unified multimodal large language model that addresses these challenges by incorporating reasoning generation through reinforcement learning. MindOmni leverages a three-phase training strategy: i) design of a unified vision language model with a decoder-only diffusion module, ii) supervised fine-tuning with Chain-of-Thought (CoT) instruction data, and iii) our proposed Reasoning Generation Policy Optimization (RGPO) algorithm, utilizing multimodal feedback to effectively guide policy updates. Experimental results demonstrate that MindOmni outperforms existing models, achieving impressive performance on both understanding and generation benchmarks, meanwhile showcasing advanced fine-grained reasoning generation capabilities, especially with mathematical reasoning instruction. All codes will be made public at \href{this https URL}{this https URL}. 

---
# Unveiling and Steering Connectome Organization with Interpretable Latent Variables 

**Authors**: Yubin Li, Xingyu Liu, Guozhang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.13011)  

**Abstract**: The brain's intricate connectome, a blueprint for its function, presents immense complexity, yet it arises from a compact genetic code, hinting at underlying low-dimensional organizational principles. This work bridges connectomics and representation learning to uncover these principles. We propose a framework that combines subgraph extraction from the Drosophila connectome, FlyWire, with a generative model to derive interpretable low-dimensional representations of neural circuitry. Crucially, an explainability module links these latent dimensions to specific structural features, offering insights into their functional relevance. We validate our approach by demonstrating effective graph reconstruction and, significantly, the ability to manipulate these latent codes to controllably generate connectome subgraphs with predefined properties. This research offers a novel tool for understanding brain architecture and a potential avenue for designing bio-inspired artificial neural networks. 

---
# The Traitors: Deception and Trust in Multi-Agent Language Model Simulations 

**Authors**: Pedro M. P. Curvo  

**Link**: [PDF](https://arxiv.org/pdf/2505.12923)  

**Abstract**: As AI systems increasingly assume roles where trust and alignment with human values are essential, understanding when and why they engage in deception has become a critical research priority. We introduce The Traitors, a multi-agent simulation framework inspired by social deduction games, designed to probe deception, trust formation, and strategic communication among large language model (LLM) agents under asymmetric information. A minority of agents the traitors seek to mislead the majority, while the faithful must infer hidden identities through dialogue and reasoning. Our contributions are: (1) we ground the environment in formal frameworks from game theory, behavioral economics, and social cognition; (2) we develop a suite of evaluation metrics capturing deception success, trust dynamics, and collective inference quality; (3) we implement a fully autonomous simulation platform where LLMs reason over persistent memory and evolving social dynamics, with support for heterogeneous agent populations, specialized traits, and adaptive behaviors. Our initial experiments across DeepSeek-V3, GPT-4o-mini, and GPT-4o (10 runs per model) reveal a notable asymmetry: advanced models like GPT-4o demonstrate superior deceptive capabilities yet exhibit disproportionate vulnerability to others' falsehoods. This suggests deception skills may scale faster than detection abilities. Overall, The Traitors provides a focused, configurable testbed for investigating LLM behavior in socially nuanced interactions. We position this work as a contribution toward more rigorous research on deception mechanisms, alignment challenges, and the broader social reliability of AI systems. 

---
# TIME: A Multi-level Benchmark for Temporal Reasoning of LLMs in Real-World Scenarios 

**Authors**: Shaohang Wei, Wei Li, Feifan Song, Wen Luo, Tianyi Zhuang, Haochen Tan, Zhijiang Guo, Houfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12891)  

**Abstract**: Temporal reasoning is pivotal for Large Language Models (LLMs) to comprehend the real world. However, existing works neglect the real-world challenges for temporal reasoning: (1) intensive temporal information, (2) fast-changing event dynamics, and (3) complex temporal dependencies in social interactions. To bridge this gap, we propose a multi-level benchmark TIME, designed for temporal reasoning in real-world scenarios. TIME consists of 38,522 QA pairs, covering 3 levels with 11 fine-grained sub-tasks. This benchmark encompasses 3 sub-datasets reflecting different real-world challenges: TIME-Wiki, TIME-News, and TIME-Dial. We conduct extensive experiments on reasoning models and non-reasoning models. And we conducted an in-depth analysis of temporal reasoning performance across diverse real-world scenarios and tasks, and summarized the impact of test-time scaling on temporal reasoning capabilities. Additionally, we release TIME-Lite, a human-annotated subset to foster future research and standardized evaluation in temporal reasoning. The code is available at this https URL , and the dataset is available at this https URL . 

---
# Detection and Mitigation of Hallucination in Large Reasoning Models: A Mechanistic Perspective 

**Authors**: Zhongxiang Sun, Qipeng Wang, Haoyu Wang, Xiao Zhang, Jun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12886)  

**Abstract**: Large Reasoning Models (LRMs) have shown impressive capabilities in multi-step reasoning tasks. However, alongside these successes, a more deceptive form of model error has emerged--Reasoning Hallucination--where logically coherent but factually incorrect reasoning traces lead to persuasive yet faulty conclusions. Unlike traditional hallucinations, these errors are embedded within structured reasoning, making them more difficult to detect and potentially more harmful. In this work, we investigate reasoning hallucinations from a mechanistic perspective. We propose the Reasoning Score, which quantifies the depth of reasoning by measuring the divergence between logits obtained from projecting late layers of LRMs to the vocabulary space, effectively distinguishing shallow pattern-matching from genuine deep reasoning. Using this score, we conduct an in-depth analysis on the ReTruthQA dataset and identify two key reasoning hallucination patterns: early-stage fluctuation in reasoning depth and incorrect backtracking to flawed prior steps. These insights motivate our Reasoning Hallucination Detection (RHD) framework, which achieves state-of-the-art performance across multiple domains. To mitigate reasoning hallucinations, we further introduce GRPO-R, an enhanced reinforcement learning algorithm that incorporates step-level deep reasoning rewards via potential-based shaping. Our theoretical analysis establishes stronger generalization guarantees, and experiments demonstrate improved reasoning quality and reduced hallucination rates. 

---
# From Grunts to Grammar: Emergent Language from Cooperative Foraging 

**Authors**: Maytus Piriyajitakonkij, Rujikorn Charakorn, Weicheng Tao, Wei Pan, Mingfei Sun, Cheston Tan, Mengmi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12872)  

**Abstract**: Early cavemen relied on gestures, vocalizations, and simple signals to coordinate, plan, avoid predators, and share resources. Today, humans collaborate using complex languages to achieve remarkable results. What drives this evolution in communication? How does language emerge, adapt, and become vital for teamwork? Understanding the origins of language remains a challenge. A leading hypothesis in linguistics and anthropology posits that language evolved to meet the ecological and social demands of early human cooperation. Language did not arise in isolation, but through shared survival goals. Inspired by this view, we investigate the emergence of language in multi-agent Foraging Games. These environments are designed to reflect the cognitive and ecological constraints believed to have influenced the evolution of communication. Agents operate in a shared grid world with only partial knowledge about other agents and the environment, and must coordinate to complete games like picking up high-value targets or executing temporally ordered actions. Using end-to-end deep reinforcement learning, agents learn both actions and communication strategies from scratch. We find that agents develop communication protocols with hallmark features of natural language: arbitrariness, interchangeability, displacement, cultural transmission, and compositionality. We quantify each property and analyze how different factors, such as population size and temporal dependencies, shape specific aspects of the emergent language. Our framework serves as a platform for studying how language can evolve from partial observability, temporal reasoning, and cooperative goals in embodied multi-agent settings. We will release all data, code, and models publicly. 

---
# Multi-Level Aware Preference Learning: Enhancing RLHF for Complex Multi-Instruction Tasks 

**Authors**: Ruopei Sun, Jianfeng Cai, Jinhua Zhu, Kangwen Zhao, Dongyun Xue, Wengang Zhou, Li Li, Houqiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.12845)  

**Abstract**: RLHF has emerged as a predominant approach for aligning artificial intelligence systems with human preferences, demonstrating exceptional and measurable efficacy in instruction following tasks; however, it exhibits insufficient compliance capabilities when confronted with complex multi-instruction tasks. Conventional approaches rely heavily on human annotation or more sophisticated large language models, thereby introducing substantial resource expenditure or potential bias concerns. Meanwhile, alternative synthetic methods that augment standard preference datasets often compromise the model's semantic quality. Our research identifies a critical oversight in existing techniques, which predominantly focus on comparing responses while neglecting valuable latent signals embedded within prompt inputs, and which only focus on preference disparities at the intra-sample level, while neglecting to account for the inter-sample level preference differentials that exist among preference data. To leverage these previously neglected indicators, we propose a novel Multi-level Aware Preference Learning (MAPL) framework, capable of enhancing multi-instruction capabilities. Specifically, for any given response in original preference data pairs, we construct varied prompts with a preference relation under different conditions, in order to learn intra-sample level preference disparities. Furthermore, for any given original preference pair, we synthesize multi-instruction preference pairs to capture preference discrepancies at the inter-sample level. Building on the two datasets constructed above, we consequently devise two sophisticated training objective functions. Subsequently, our framework integrates seamlessly into both Reward Modeling and Direct Preference Optimization paradigms. Through rigorous evaluation across multiple benchmarks, we empirically validate the efficacy of our framework. 

---
# AGI-Elo: How Far Are We From Mastering A Task? 

**Authors**: Shuo Sun, Yimin Zhao, Christina Dao Wen Lee, Jiawei Sun, Chengran Yuan, Zefan Huang, Dongen Li, Justin KW Yeoh, Alok Prakash, Thomas W. Malone, Marcelo H. Ang Jr  

**Link**: [PDF](https://arxiv.org/pdf/2505.12844)  

**Abstract**: As the field progresses toward Artificial General Intelligence (AGI), there is a pressing need for more comprehensive and insightful evaluation frameworks that go beyond aggregate performance metrics. This paper introduces a unified rating system that jointly models the difficulty of individual test cases and the competency of AI models (or humans) across vision, language, and action domains. Unlike existing metrics that focus solely on models, our approach allows for fine-grained, difficulty-aware evaluations through competitive interactions between models and tasks, capturing both the long-tail distribution of real-world challenges and the competency gap between current models and full task mastery. We validate the generalizability and robustness of our system through extensive experiments on multiple established datasets and models across distinct AGI domains. The resulting rating distributions offer novel perspectives and interpretable insights into task difficulty, model progression, and the outstanding challenges that remain on the path to achieving full AGI task mastery. 

---
# Reasoning BO: Enhancing Bayesian Optimization with Long-Context Reasoning Power of LLMs 

**Authors**: Zhuo Yang, Lingli Ge, Dong Han, Tianfan Fu, Yuqiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.12833)  

**Abstract**: Many real-world scientific and industrial applications require the optimization of expensive black-box functions. Bayesian Optimization (BO) provides an effective framework for such problems. However, traditional BO methods are prone to get trapped in local optima and often lack interpretable insights. To address this issue, this paper designs Reasoning BO, a novel framework that leverages reasoning models to guide the sampling process in BO while incorporating multi-agent systems and knowledge graphs for online knowledge accumulation. By integrating the reasoning and contextual understanding capabilities of Large Language Models (LLMs), we can provide strong guidance to enhance the BO process. As the optimization progresses, Reasoning BO provides real-time sampling recommendations along with critical insights grounded in plausible scientific theories, aiding in the discovery of superior solutions within the search space. We systematically evaluate our approach across 10 diverse tasks encompassing synthetic mathematical functions and complex real-world applications. The framework demonstrates its capability to progressively refine sampling strategies through real-time insights and hypothesis evolution, effectively identifying higher-performing regions of the search space for focused exploration. This process highlights the powerful reasoning and context-learning abilities of LLMs in optimization scenarios. For example, in the Direct Arylation task, our method increased the yield to 60.7%, whereas traditional BO achieved only a 25.2% yield. Furthermore, our investigation reveals that smaller LLMs, when fine-tuned through reinforcement learning, can attain comparable performance to their larger counterparts. This enhanced reasoning capability paves the way for more efficient automated scientific experimentation while maintaining computational feasibility. 

---
# Emergent Specialization: Rare Token Neurons in Language Models 

**Authors**: Jing Liu, Haozheng Wang, Yueheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.12822)  

**Abstract**: Large language models struggle with representing and generating rare tokens despite their importance in specialized domains. In this study, we identify neuron structures with exceptionally strong influence on language model's prediction of rare tokens, termed as rare token neurons, and investigate the mechanism for their emergence and behavior. These neurons exhibit a characteristic three-phase organization (plateau, power-law, and rapid decay) that emerges dynamically during training, evolving from a homogeneous initial state to a functionally differentiated architecture. In the activation space, rare token neurons form a coordinated subnetwork that selectively co-activates while avoiding co-activation with other neurons. This functional specialization potentially correlates with the development of heavy-tailed weight distributions, suggesting a statistical mechanical basis for emergent specialization. 

---
# FRAbench and GenEval: Scaling Fine-Grained Aspect Evaluation across Tasks, Modalities 

**Authors**: Shibo Hong, Jiahao Ying, Haiyuan Liang, Mengdi Zhang, Jun Kuang, Jiazheng Zhang, Yixin Cao  

**Link**: [PDF](https://arxiv.org/pdf/2505.12795)  

**Abstract**: Evaluating the open-ended outputs of large language models (LLMs) has become a bottleneck as model capabilities, task diversity, and modality coverage rapidly expand. Existing "LLM-as-a-Judge" evaluators are typically narrow in a few tasks, aspects, or modalities, and easily suffer from low consistency. In this paper, we argue that explicit, fine-grained aspect specification is the key to both generalizability and objectivity in automated evaluation. To do so, we introduce a hierarchical aspect taxonomy spanning 112 aspects that unifies evaluation across four representative settings - Natural Language Generation, Image Understanding, Image Generation, and Interleaved Text-and-Image Generation. Building on this taxonomy, we create FRAbench, a benchmark comprising 60.4k pairwise samples with 325k aspect-level labels obtained from a combination of human and LLM annotations. FRAbench provides the first large-scale, multi-modal resource for training and meta-evaluating fine-grained LMM judges. Leveraging FRAbench, we develop GenEval, a fine-grained evaluator generalizable across tasks and modalities. Experiments show that GenEval (i) attains high agreement with GPT-4o and expert annotators, (ii) transfers robustly to unseen tasks and modalities, and (iii) reveals systematic weaknesses of current LMMs on evaluation. 

---
# Mixture Policy based Multi-Hop Reasoning over N-tuple Temporal Knowledge Graphs 

**Authors**: Zhongni Hou, Miao Su, Xiaolong Jin, Zixuan Li, Long Bai, Jiafeng Guo, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.12788)  

**Abstract**: Temporal Knowledge Graphs (TKGs), which utilize quadruples in the form of (subject, predicate, object, timestamp) to describe temporal facts, have attracted extensive attention. N-tuple TKGs (N-TKGs) further extend traditional TKGs by utilizing n-tuples to incorporate auxiliary elements alongside core elements (i.e., subject, predicate, and object) of facts, so as to represent them in a more fine-grained manner. Reasoning over N-TKGs aims to predict potential future facts based on historical ones. However, existing N-TKG reasoning methods often lack explainability due to their black-box nature. Therefore, we introduce a new Reinforcement Learning-based method, named MT-Path, which leverages the temporal information to traverse historical n-tuples and construct a temporal reasoning path. Specifically, in order to integrate the information encapsulated within n-tuples, i.e., the entity-irrelevant information within the predicate, the information about core elements, and the complete information about the entire n-tuples, MT-Path utilizes a mixture policy-driven action selector, which bases on three low-level policies, namely, the predicate-focused policy, the core-element-focused policy and the whole-fact-focused policy. Further, MT-Path utilizes an auxiliary element-aware GCN to capture the rich semantic dependencies among facts, thereby enabling the agent to gain a deep understanding of each n-tuple. Experimental results demonstrate the effectiveness and the explainability of MT-Path. 

---
# Language Models That Walk the Talk: A Framework for Formal Fairness Certificates 

**Authors**: Danqing Chen, Tobias Ladner, Ahmed Rayen Mhadhbi, Matthias Althoff  

**Link**: [PDF](https://arxiv.org/pdf/2505.12767)  

**Abstract**: As large language models become integral to high-stakes applications, ensuring their robustness and fairness is critical. Despite their success, large language models remain vulnerable to adversarial attacks, where small perturbations, such as synonym substitutions, can alter model predictions, posing risks in fairness-critical areas, such as gender bias mitigation, and safety-critical areas, such as toxicity detection. While formal verification has been explored for neural networks, its application to large language models remains limited. This work presents a holistic verification framework to certify the robustness of transformer-based language models, with a focus on ensuring gender fairness and consistent outputs across different gender-related terms. Furthermore, we extend this methodology to toxicity detection, offering formal guarantees that adversarially manipulated toxic inputs are consistently detected and appropriately censored, thereby ensuring the reliability of moderation systems. By formalizing robustness within the embedding space, this work strengthens the reliability of language models in ethical AI deployment and content moderation. 

---
# IDEAL: Data Equilibrium Adaptation for Multi-Capability Language Model Alignment 

**Authors**: Chenlin Ming, Chendi Qu, Mengzhang Cai, Qizhi Pei, Zhuoshi Pan, Yu Li, Xiaoming Duan, Lijun Wu, Conghui He  

**Link**: [PDF](https://arxiv.org/pdf/2505.12762)  

**Abstract**: Large Language Models (LLMs) have achieved impressive performance through Supervised Fine-tuning (SFT) on diverse instructional datasets. When training on multiple capabilities simultaneously, the mixture training dataset, governed by volumes of data from different domains, is a critical factor that directly impacts the final model's performance. Unlike many studies that focus on enhancing the quality of training datasets through data selection methods, few works explore the intricate relationship between the compositional quantity of mixture training datasets and the emergent capabilities of LLMs. Given the availability of a high-quality multi-domain training dataset, understanding the impact of data from each domain on the model's overall capabilities is crucial for preparing SFT data and training a well-balanced model that performs effectively across diverse domains. In this work, we introduce IDEAL, an innovative data equilibrium adaptation framework designed to effectively optimize volumes of data from different domains within mixture SFT datasets, thereby enhancing the model's alignment and performance across multiple capabilities. IDEAL employs a gradient-based approach to iteratively refine the training data distribution, dynamically adjusting the volumes of domain-specific data based on their impact on downstream task performance. By leveraging this adaptive mechanism, IDEAL ensures a balanced dataset composition, enabling the model to achieve robust generalization and consistent proficiency across diverse tasks. Experiments across different capabilities demonstrate that IDEAL outperforms conventional uniform data allocation strategies, achieving a comprehensive improvement of approximately 7% in multi-task evaluation scores. 

---
# Correspondence of high-dimensional emotion structures elicited by video clips between humans and Multimodal LLMs 

**Authors**: Haruka Asanuma, Naoko Koide-Majima, Ken Nakamura, Takato Horii, Shinji Nishimoto, Masafumi Oizumi  

**Link**: [PDF](https://arxiv.org/pdf/2505.12746)  

**Abstract**: Recent studies have revealed that human emotions exhibit a high-dimensional, complex structure. A full capturing of this complexity requires new approaches, as conventional models that disregard high dimensionality risk overlooking key nuances of human emotions. Here, we examined the extent to which the latest generation of rapidly evolving Multimodal Large Language Models (MLLMs) capture these high-dimensional, intricate emotion structures, including capabilities and limitations. Specifically, we compared self-reported emotion ratings from participants watching videos with model-generated estimates (e.g., Gemini or GPT). We evaluated performance not only at the individual video level but also from emotion structures that account for inter-video relationships. At the level of simple correlation between emotion structures, our results demonstrated strong similarity between human and model-inferred emotion structures. To further explore whether the similarity between humans and models is at the signle item level or the coarse-categorical level, we applied Gromov Wasserstein Optimal Transport. We found that although performance was not necessarily high at the strict, single-item level, performance across video categories that elicit similar emotions was substantial, indicating that the model could infer human emotional experiences at the category level. Our results suggest that current state-of-the-art MLLMs broadly capture the complex high-dimensional emotion structures at the category level, as well as their apparent limitations in accurately capturing entire structures at the single-item level. 

---
# Incentivizing Multimodal Reasoning in Large Models for Direct Robot Manipulation 

**Authors**: Weiliang Tang, Dong Jing, Jia-Hui Pan, Zhiwu Lu, Yun-Hui Liu, Li Erran Li, Mingyu Ding, Chi-Wing Fu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12744)  

**Abstract**: Recent Large Multimodal Models have demonstrated remarkable reasoning capabilities, especially in solving complex mathematical problems and realizing accurate spatial perception. Our key insight is that these emerging abilities can naturally extend to robotic manipulation by enabling LMMs to directly infer the next goal in language via reasoning, rather than relying on a separate action head. However, this paradigm meets two main challenges: i) How to make LMMs understand the spatial action space, and ii) How to fully exploit the reasoning capacity of LMMs in solving these tasks. To tackle the former challenge, we propose a novel task formulation, which inputs the current states of object parts and the gripper, and reformulates rotation by a new axis representation instead of traditional Euler angles. This representation is more compatible with spatial reasoning and easier to interpret within a unified language space. For the latter challenge, we design a pipeline to utilize cutting-edge LMMs to generate a small but high-quality reasoning dataset of multi-round dialogues that successfully solve manipulation tasks for supervised fine-tuning. Then, we perform reinforcement learning by trial-and-error interactions in simulation to further enhance the model's reasoning abilities for robotic manipulation. Our resulting reasoning model built upon a 7B backbone, named ReasonManip, demonstrates three notable advantages driven by its system-2 level reasoning capabilities: i) exceptional generalizability to out-of-distribution environments, objects, and tasks; ii) inherent sim-to-real transfer ability enabled by the unified language representation shared across domains; iii) transparent interpretability connecting high-level reasoning and low-level control. Extensive experiments demonstrate the effectiveness of the proposed paradigm and its potential to advance LMM-driven robotic manipulation. 

---
# Dense Communication between Language Models 

**Authors**: Shiguang Wu, Yaqing Wang, Quanming Yao  

**Link**: [PDF](https://arxiv.org/pdf/2505.12741)  

**Abstract**: As higher-level intelligence emerges from the combination of modular components with lower-level intelligence, many works combines Large Language Models (LLMs) for collective intelligence. Such combination is achieved by building communications among LLMs. While current systems primarily facilitate such communication through natural language, this paper proposes a novel paradigm of direct dense vector communication between LLMs. Our approach eliminates the unnecessary embedding and de-embedding steps when LLM interact with another, enabling more efficient information transfer, fully differentiable optimization pathways, and exploration of capabilities beyond human heuristics. We use such stripped LLMs as vertexes and optimizable seq2seq modules as edges to construct LMNet, with similar structure as MLPs. By utilizing smaller pre-trained LLMs as vertexes, we train a LMNet that achieves comparable performance with LLMs in similar size with only less than 0.1% training cost. This offers a new perspective on scaling for general intelligence rather than training a monolithic LLM from scratch. Besides, the proposed method can be used for other applications, like customizing LLM with limited data, showing its versatility. 

---
# Accelerating Adaptive Retrieval Augmented Generation via Instruction-Driven Representation Reduction of Retrieval Overlaps 

**Authors**: Jie Ou, Jinyu Guo, Shuaihong Jiang, Zhaokun Wang, Libo Qin, Shunyu Yao, Wenhong Tian  

**Link**: [PDF](https://arxiv.org/pdf/2505.12731)  

**Abstract**: Retrieval-augmented generation (RAG) has emerged as a pivotal method for expanding the knowledge of large language models. To handle complex queries more effectively, researchers developed Adaptive-RAG (A-RAG) to enhance the generated quality through multiple interactions with external knowledge bases. Despite its effectiveness, A-RAG exacerbates the pre-existing efficiency challenges inherent in RAG, which are attributable to its reliance on multiple iterations of generation. Existing A-RAG approaches process all retrieved contents from scratch. However, they ignore the situation where there is a significant overlap in the content of the retrieval results across rounds. The overlapping content is redundantly represented, which leads to a large proportion of repeated computations, thus affecting the overall efficiency. To address this issue, this paper introduces a model-agnostic approach that can be generally applied to A-RAG methods, which is dedicated to reducing the redundant representation process caused by the overlapping of retrieval results. Specifically, we use cache access and parallel generation to speed up the prefilling and decoding stages respectively. Additionally, we also propose an instruction-driven module to further guide the model to more effectively attend to each part of the content in a more suitable way for LLMs. Experiments show that our approach achieves 2.79 and 2.33 times significant acceleration on average for prefilling and decoding respectively while maintaining equal generation quality. 

---
# Bullying the Machine: How Personas Increase LLM Vulnerability 

**Authors**: Ziwei Xu, Udit Sanghi, Mohan Kankanhalli  

**Link**: [PDF](https://arxiv.org/pdf/2505.12692)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in interactions where they are prompted to adopt personas. This paper investigates whether such persona conditioning affects model safety under bullying, an adversarial manipulation that applies psychological pressures in order to force the victim to comply to the attacker. We introduce a simulation framework in which an attacker LLM engages a victim LLM using psychologically grounded bullying tactics, while the victim adopts personas aligned with the Big Five personality traits. Experiments using multiple open-source LLMs and a wide range of adversarial goals reveal that certain persona configurations -- such as weakened agreeableness or conscientiousness -- significantly increase victim's susceptibility to unsafe outputs. Bullying tactics involving emotional or sarcastic manipulation, such as gaslighting and ridicule, are particularly effective. These findings suggest that persona-driven interaction introduces a novel vector for safety risks in LLMs and highlight the need for persona-aware safety evaluation and alignment strategies. 

---
# Ineq-Comp: Benchmarking Human-Intuitive Compositional Reasoning in Automated Theorem Proving on Inequalities 

**Authors**: Haoyu Zhao, Yihan Geng, Shange Tang, Yong Lin, Bohan Lyu, Hongzhou Lin, Chi Jin, Sanjeev Arora  

**Link**: [PDF](https://arxiv.org/pdf/2505.12680)  

**Abstract**: LLM-based formal proof assistants (e.g., in Lean) hold great promise for automating mathematical discovery. But beyond syntactic correctness, do these systems truly understand mathematical structure as humans do? We investigate this question through the lens of mathematical inequalities -- a fundamental tool across many domains. While modern provers can solve basic inequalities, we probe their ability to handle human-intuitive compositionality. We introduce Ineq-Comp, a benchmark built from elementary inequalities through systematic transformations, including variable duplication, algebraic rewriting, and multi-step composition. Although these problems remain easy for humans, we find that most provers -- including Goedel, STP, and Kimina-7B -- struggle significantly. DeepSeek-Prover-V2-7B shows relative robustness -- possibly because it is trained to decompose the problems into sub-problems -- but still suffers a 20\% performance drop (pass@32). Strikingly, performance remains poor for all models even when formal proofs of the constituent parts are provided in context, revealing that the source of weakness is indeed in compositional reasoning. Our results expose a persisting gap between the generalization behavior of current AI provers and human mathematical intuition. 

---
# $\texttt{DIAMONDs}$: A Dataset for $\mathbb{D}$ynamic $\mathbb{I}$nformation $\mathbb{A}$nd $\mathbb{M}$ental modeling $\mathbb{O}$f $\mathbb{N}$umeric $\mathbb{D}$iscussions 

**Authors**: Sayontan Ghosh, Mahnaz Koupaee, Yash Kumar Lal, Pegah Alipoormolabashi, Mohammad Saqib Hasan, Jun Seok Kang, Niranjan Balasubramanian  

**Link**: [PDF](https://arxiv.org/pdf/2505.12651)  

**Abstract**: Understanding multiparty conversations demands robust Theory of Mind (ToM) capabilities, including the ability to track dynamic information, manage knowledge asymmetries, and distinguish relevant information across extended exchanges. To advance ToM evaluation in such settings, we present a carefully designed scalable methodology for generating high-quality benchmark conversation-question pairs with these characteristics. Using this methodology, we create $\texttt{DIAMONDs}$, a new conversational QA dataset covering common business, financial or other group interactions. In these goal-oriented conversations, participants often have to track certain numerical quantities (say $\textit{expected profit}$) of interest that can be derived from other variable quantities (like $\textit{marketing expenses, expected sales, salary}$, etc.), whose values also change over the course of the conversation. $\texttt{DIAMONDs}$ questions pose simple numerical reasoning problems over such quantities of interest (e.g., $\textit{funds required for charity events, expected company profit next quarter}$, etc.) in the context of the information exchanged in conversations. This allows for precisely evaluating ToM capabilities for carefully tracking and reasoning over participants' knowledge states.
Our evaluation of state-of-the-art language models reveals significant challenges in handling participant-centric reasoning, specifically in situations where participants have false beliefs. Models also struggle with conversations containing distractors and show limited ability to identify scenarios with insufficient information. These findings highlight current models' ToM limitations in handling real-world multi-party conversations. 

---
# RealMath: A Continuous Benchmark for Evaluating Language Models on Research-Level Mathematics 

**Authors**: Jie Zhang, Cezara Petrui, Kristina Nikolić, Florian Tramèr  

**Link**: [PDF](https://arxiv.org/pdf/2505.12575)  

**Abstract**: Existing benchmarks for evaluating mathematical reasoning in large language models (LLMs) rely primarily on competition problems, formal proofs, or artificially challenging questions -- failing to capture the nature of mathematics encountered in actual research environments. We introduce RealMath, a novel benchmark derived directly from research papers and mathematical forums that assesses LLMs' abilities on authentic mathematical tasks. Our approach addresses three critical challenges: sourcing diverse research-level content, enabling reliable automated evaluation through verifiable statements, and designing a continually refreshable dataset to mitigate contamination risks. Experimental results across multiple LLMs reveal surprising capabilities in handling research mathematics compared to competition problems, suggesting current models may already serve as valuable assistants for working mathematicians despite limitations on highly challenging problems. The code and dataset for RealMath are publicly available. 

---
# mCLM: A Function-Infused and Synthesis-Friendly Modular Chemical Language Model 

**Authors**: Carl Edwards, Chi Han, Gawon Lee, Thao Nguyen, Bowen Jin, Chetan Kumar Prasad, Sara Szymkuć, Bartosz A. Grzybowski, Ying Diao, Jiawei Han, Ge Liu, Hao Peng, Martin D. Burke, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2505.12565)  

**Abstract**: Despite their ability to understand chemical knowledge and accurately generate sequential representations, large language models (LLMs) remain limited in their capacity to propose novel molecules with drug-like properties. In addition, the molecules that LLMs propose can often be challenging to make in the lab. To more effectively enable the discovery of functional small molecules, LLMs need to learn a molecular language. However, LLMs are currently limited by encoding molecules from atoms. In this paper, we argue that just like tokenizing texts into (sub-)word tokens instead of characters, molecules should be decomposed and reassembled at the level of functional building blocks, i.e., parts of molecules that bring unique functions and serve as effective building blocks for real-world automated laboratory synthesis. This motivates us to propose mCLM, a modular Chemical-Language Model tokenizing molecules into building blocks and learning a bilingual language model of both natural language descriptions of functions and molecule building blocks. By reasoning on such functional building blocks, mCLM guarantees to generate efficiently synthesizable molecules thanks to recent progress in block-based chemistry, while also improving the functions of molecules in a principled manner. In experiments on 430 FDA-approved drugs, we find mCLM capable of significantly improving 5 out of 6 chemical functions critical to determining drug potentials. More importantly, mCLM can reason on multiple functions and improve the FDA-rejected drugs (``fallen angels'') over multiple iterations to greatly improve their shortcomings. 

---
# ALAS: A Stateful Multi-LLM Agent Framework for Disruption-Aware Planning 

**Authors**: Edward Y. Chang, Longling Geng  

**Link**: [PDF](https://arxiv.org/pdf/2505.12501)  

**Abstract**: Large language models (LLMs) excel at rapid generation of text and multimodal content, yet they falter on transaction-style planning that demands ACID-like guarantees and real-time disruption recovery. We present Adaptive LLM Agent System (ALAS), a framework that tackles four fundamental LLM deficits: (i) absence of self-verification, (ii) context erosion, (iii) next-token myopia, and (iv) lack of persistent state. ALAS decomposes each plan into role-specialized agents, equips them with automatic state tracking, and coordinates them through a lightweight protocol. When disruptions arise, agents apply history-aware local compensation, avoiding costly global replanning and containing cascade effects. On real-world, large-scale job-shop scheduling benchmarks, ALAS sets new best results for static sequential planning and excels in dynamic reactive scenarios with unexpected disruptions. These gains show that principled modularization plus targeted compensation can unlock scalable and resilient planning with LLMs. 

---
# MARGE: Improving Math Reasoning for LLMs with Guided Exploration 

**Authors**: Jingyue Gao, Runji Lin, Keming Lu, Bowen Yu, Junyang Lin, Jianyu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.12500)  

**Abstract**: Large Language Models (LLMs) exhibit strong potential in mathematical reasoning, yet their effectiveness is often limited by a shortage of high-quality queries. This limitation necessitates scaling up computational responses through self-generated data, yet current methods struggle due to spurious correlated data caused by ineffective exploration across all reasoning stages. To address such challenge, we introduce \textbf{MARGE}: Improving \textbf{Ma}th \textbf{R}easoning with \textbf{G}uided \textbf{E}xploration, a novel method to address this issue and enhance mathematical reasoning through hit-guided exploration. MARGE systematically explores intermediate reasoning states derived from self-generated solutions, enabling adequate exploration and improved credit assignment throughout the reasoning process. Through extensive experiments across multiple backbone models and benchmarks, we demonstrate that MARGE significantly improves reasoning capabilities without requiring external annotations or training additional value models. Notably, MARGE improves both single-shot accuracy and exploration diversity, mitigating a common trade-off in alignment methods. These results demonstrate MARGE's effectiveness in enhancing mathematical reasoning capabilities and unlocking the potential of scaling self-generated training data. Our code and models are available at \href{this https URL}{this link}. 

---
# UIShift: Enhancing VLM-based GUI Agents through Self-supervised Reinforcement Learning 

**Authors**: Longxi Gao, Li Zhang, Mengwei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12493)  

**Abstract**: Training effective Vision Language Models (VLMs) for GUI agents typically relies on supervised fine-tuning (SFT) over large-scale annotated datasets, where the collection process is labor-intensive and error-prone. In this work, we propose a self-supervised inverse dynamics task to enable VLMs to learn from GUI transition pairs by inferring the action that caused that transition. This training task offers two advantages: (1) It enables VLMs to ignore variations unrelated to user actions (e.g., background refreshes, ads) and to focus on true affordances such as buttons and input fields within complex GUIs. (2) The training data can be easily obtained from existing GUI trajectories without requiring human annotation, and it can be easily scaled through automatic offline exploration. Using this training task, we propose UI-shift, a framework for enhancing VLM-based GUI agents through self-supervised reinforcement learning (RL). With only 2K training samples sourced from existing datasets, two VLMs -- Qwen2.5-VL-3B and Qwen2.5-VL-7B -- trained with UI-Shift achieve competitive or superior performance on grounding tasks (ScreenSpot-series benchmarks) and GUI automation tasks (AndroidControl), compared to SFT baselines and GUI-specific models that explicitly elicit reasoning abilities during RL. Our findings suggest a potential direction for enhancing VLMs for GUI agents by leveraging more self-supervised training data in the future. 

---
# NeuroGen: Neural Network Parameter Generation via Large Language Models 

**Authors**: Jiaqi Wang, Yusen Zhang, Xi Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.12470)  

**Abstract**: Acquiring the parameters of neural networks (NNs) has been one of the most important problems in machine learning since the inception of NNs. Traditional approaches, such as backpropagation and forward-only optimization, acquire parameters via iterative data fitting to gradually optimize them. This paper aims to explore the feasibility of a new direction: acquiring NN parameters via large language model generation. We propose NeuroGen, a generalized and easy-to-implement two-stage approach for NN parameter generation conditioned on descriptions of the data, task, and network architecture. Stage one is Parameter Reference Knowledge Injection, where LLMs are pretrained on NN checkpoints to build foundational understanding of parameter space, whereas stage two is Context-Enhanced Instruction Tuning, enabling LLMs to adapt to specific tasks through enriched, task-aware prompts. Experimental results demonstrate that NeuroGen effectively generates usable NN parameters. Our findings highlight the feasibility of LLM-based NN parameter generation and suggest a promising new paradigm where LLMs and lightweight NNs can coexist synergistically 

---
# Model Discovery with Grammatical Evolution. An Experiment with Prime Numbers 

**Authors**: Jakub Skrzyński, Dominik Sepioło, Antoni Ligęza  

**Link**: [PDF](https://arxiv.org/pdf/2505.12440)  

**Abstract**: Machine Learning produces efficient decision and prediction models based on input-output data only. Such models have the form of decision trees or neural nets and are far from transparent analytical models, based on mathematical formulas. Analytical model discovery requires additional knowledge and may be performed with Grammatical Evolution. Such models are transparent, concise, and have readable components and structure. This paper reports on a non-trivial experiment with generating such models. 

---
# MedAgentBoard: Benchmarking Multi-Agent Collaboration with Conventional Methods for Diverse Medical Tasks 

**Authors**: Yinghao Zhu, Ziyi He, Haoran Hu, Xiaochen Zheng, Xichen Zhang, Zixiang Wang, Junyi Gao, Liantao Ma, Lequan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12371)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) has stimulated interest in multi-agent collaboration for addressing complex medical tasks. However, the practical advantages of multi-agent collaboration approaches remain insufficiently understood. Existing evaluations often lack generalizability, failing to cover diverse tasks reflective of real-world clinical practice, and frequently omit rigorous comparisons against both single-LLM-based and established conventional methods. To address this critical gap, we introduce MedAgentBoard, a comprehensive benchmark for the systematic evaluation of multi-agent collaboration, single-LLM, and conventional approaches. MedAgentBoard encompasses four diverse medical task categories: (1) medical (visual) question answering, (2) lay summary generation, (3) structured Electronic Health Record (EHR) predictive modeling, and (4) clinical workflow automation, across text, medical images, and structured EHR data. Our extensive experiments reveal a nuanced landscape: while multi-agent collaboration demonstrates benefits in specific scenarios, such as enhancing task completeness in clinical workflow automation, it does not consistently outperform advanced single LLMs (e.g., in textual medical QA) or, critically, specialized conventional methods that generally maintain better performance in tasks like medical VQA and EHR-based prediction. MedAgentBoard offers a vital resource and actionable insights, emphasizing the necessity of a task-specific, evidence-based approach to selecting and developing AI solutions in medicine. It underscores that the inherent complexity and overhead of multi-agent collaboration must be carefully weighed against tangible performance gains. All code, datasets, detailed prompts, and experimental results are open-sourced at this https URL. 

---
# Enhancing Visual Grounding for GUI Agents via Self-Evolutionary Reinforcement Learning 

**Authors**: Xinbin Yuan, Jian Zhang, Kaixin Li, Zhuoxuan Cai, Lujian Yao, Jie Chen, Enguang Wang, Qibin Hou, Jinwei Chen, Peng-Tao Jiang, Bo Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.12370)  

**Abstract**: Graphical User Interface (GUI) agents have made substantial strides in understanding and executing user instructions across diverse platforms. Yet, grounding these instructions to precise interface elements remains challenging, especially in complex, high-resolution, professional environments. Traditional supervised finetuning (SFT) methods often require large volumes of diverse data and exhibit weak generalization. To overcome these limitations, we introduce a reinforcement learning (RL) based framework that incorporates three core strategies: (1) seed data curation to ensure high quality training samples, (2) a dense policy gradient that provides continuous feedback based on prediction accuracy, and (3) a self evolutionary reinforcement finetuning mechanism that iteratively refines the model using attention maps. With only 3k training samples, our 7B-parameter model achieves state-of-the-art results among similarly sized models on three grounding benchmarks. Notably, it attains 47.3\% accuracy on the ScreenSpot-Pro dataset, outperforming much larger models, such as UI-TARS-72B, by a margin of 24.2\%. These findings underscore the effectiveness of RL-based approaches in enhancing GUI agent performance, particularly in high-resolution, complex environments. 

---
# Fully Geometric Multi-Hop Reasoning on Knowledge Graphs with Transitive Relations 

**Authors**: Fernando Zhapa-Camacho, Robert Hoehndorf  

**Link**: [PDF](https://arxiv.org/pdf/2505.12369)  

**Abstract**: Geometric embedding methods have shown to be useful for multi-hop reasoning on knowledge graphs by mapping entities and logical operations to geometric regions and geometric transformations, respectively. Geometric embeddings provide direct interpretability framework for queries. However, current methods have only leveraged the geometric construction of entities, failing to map logical operations to geometric transformations and, instead, using neural components to learn these operations. We introduce GeometrE, a geometric embedding method for multi-hop reasoning, which does not require learning the logical operations and enables full geometric interpretability. Additionally, unlike previous methods, we introduce a transitive loss function and show that it can preserve the logical rule $\forall a,b,c: r(a,b) \land r(b,c) \to r(a,c)$. Our experiments show that GeometrE outperforms current state-of-the-art methods on standard benchmark datasets. 

---
# GATES: Cost-aware Dynamic Workflow Scheduling via Graph Attention Networks and Evolution Strategy 

**Authors**: Ya Shen, Gang Chen, Hui Ma, Mengjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12355)  

**Abstract**: Cost-aware Dynamic Workflow Scheduling (CADWS) is a key challenge in cloud computing, focusing on devising an effective scheduling policy to efficiently schedule dynamically arriving workflow tasks, represented as Directed Acyclic Graphs (DAG), to suitable virtual machines (VMs). Deep reinforcement learning (DRL) has been widely employed for automated scheduling policy design. However, the performance of DRL is heavily influenced by the design of the problem-tailored policy network and is highly sensitive to hyperparameters and the design of reward feedback. Considering the above-mentioned issues, this study proposes a novel DRL method combining Graph Attention Networks-based policy network and Evolution Strategy, referred to as GATES. The contributions of GATES are summarized as follows: (1) GATES can capture the impact of current task scheduling on subsequent tasks by learning the topological relationships between tasks in a DAG. (2) GATES can learn the importance of each VM to ready tasks, increasing the chance of selecting the optimal VM. (3) Utilizing Evolution Strategy's robustness, exploratory nature, and tolerance for delayed rewards, GATES achieves stable policy learning in CADWS. Extensive experimental results demonstrate the superiority of the proposed GATES in CADWS, outperforming several state-of-the-art algorithms. Codes are available at: this https URL 

---
# Reasoning-CV: Fine-tuning Powerful Reasoning LLMs for Knowledge-Assisted Claim Verification 

**Authors**: Zhi Zheng, Wee Sun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.12348)  

**Abstract**: Claim verification is essential in combating misinformation, and large language models (LLMs) have recently emerged in this area as powerful tools for assessing the veracity of claims using external knowledge. Existing LLM-based methods for claim verification typically adopt a Decompose-Then-Verify paradigm, which involves decomposing complex claims into several independent sub-claims and verifying each sub-claim separately. However, this paradigm often introduces errors during the claim decomposition process. To mitigate these errors, we propose to develop the Chain-of-Thought (CoT)-Verify paradigm, which leverages LLM reasoning methods to generate CoT-verification paths for the original complex claim without requiring decompositions into sub-claims and separate verification stages. The CoT-Verify paradigm allows us to propose a natural fine-tuning method called Reasoning-CV to enhance the verification capabilities in LLMs. Reasoning-CV includes a supervised fine-tuning (SFT) stage and a self-improvement direct preference optimization (DPO) stage. Utilizing only an 8B pre-trained LLM, Reasoning-CV demonstrates superior knowledge-assisted claim verification performances compared to existing Decompose-Then-Verify methods, as well as powerful black-box LLMs such as GPT-4o+CoT and o1-preview. Our code is available. 

---
# SEED-GRPO: Semantic Entropy Enhanced GRPO for Uncertainty-Aware Policy Optimization 

**Authors**: Minghan Chen, Guikun Chen, Wenguan Wang, Yi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12346)  

**Abstract**: Large language models (LLMs) exhibit varying levels of confidence across input prompts (questions): some lead to consistent, semantically similar answers, while others yield diverse or contradictory outputs. This variation reflects LLM's uncertainty about the input prompt, a signal of how confidently the model understands a given problem. However, vanilla Group Relative Policy Optimization (GRPO) treats all prompts equally during policy updates, ignoring this important information about the model's knowledge boundaries. To address this limitation, we propose SEED-GRPO (Semantic Entropy EnhanceD GRPO), which explicitly measures LLMs' uncertainty of the input prompts semantic entropy. Semantic entropy measures the diversity of meaning in multiple generated answers given a prompt and uses this to modulate the magnitude of policy updates. This uncertainty-aware training mechanism enables dynamic adjustment of policy update magnitudes based on question uncertainty. It allows more conservative updates on high-uncertainty questions while maintaining the original learning signal on confident ones. Experimental results on five mathematical reasoning benchmarks (AIME24 56.7, AMC 68.7, MATH 83.4, Minerva 34.2, and OlympiadBench 48.0) demonstrate that SEED-GRPO achieves new state-of-the-art performance in average accuracy, validating the effectiveness of uncertainty-aware policy optimization. 

---
# Enhancing User-Oriented Proactivity in Open-Domain Dialogues with Critic Guidance 

**Authors**: Yufeng Wang, Jinwu Hu, Ziteng Huang, Kunyang Lin, Zitian Zhang, Peihao Chen, Yu Hu, Qianyue Wang, Zhuliang Yu, Bin Sun, Xiaofen Xing, Qingfang Zheng, Mingkui Tan  

**Link**: [PDF](https://arxiv.org/pdf/2505.12334)  

**Abstract**: Open-domain dialogue systems aim to generate natural and engaging conversations, providing significant practical value in real applications such as social robotics and personal assistants. The advent of large language models (LLMs) has greatly advanced this field by improving context understanding and conversational fluency. However, existing LLM-based dialogue systems often fall short in proactively understanding the user's chatting preferences and guiding conversations toward user-centered topics. This lack of user-oriented proactivity can lead users to feel unappreciated, reducing their satisfaction and willingness to continue the conversation in human-computer interactions. To address this issue, we propose a User-oriented Proactive Chatbot (UPC) to enhance the user-oriented proactivity. Specifically, we first construct a critic to evaluate this proactivity inspired by the LLM-as-a-judge strategy. Given the scarcity of high-quality training data, we then employ the critic to guide dialogues between the chatbot and user agents, generating a corpus with enhanced user-oriented proactivity. To ensure the diversity of the user backgrounds, we introduce the ISCO-800, a diverse user background dataset for constructing user agents. Moreover, considering the communication difficulty varies among users, we propose an iterative curriculum learning method that trains the chatbot from easy-to-communicate users to more challenging ones, thereby gradually enhancing its performance. Experiments demonstrate that our proposed training method is applicable to different LLMs, improving user-oriented proactivity and attractiveness in open-domain dialogues. 

---
# MPRM: A Markov Path-based Rule Miner for Efficient and Interpretable Knowledge Graph Reasoning 

**Authors**: Mingyang Li, Song Wang, Ning Cai  

**Link**: [PDF](https://arxiv.org/pdf/2505.12329)  

**Abstract**: Rule mining in knowledge graphs enables interpretable link prediction. However, deep learning-based rule mining methods face significant memory and time challenges for large-scale knowledge graphs, whereas traditional approaches, limited by rigid confidence metrics, incur high computational costs despite sampling techniques. To address these challenges, we propose MPRM, a novel rule mining method that models rule-based inference as a Markov chain and uses an efficient confidence metric derived from aggregated path probabilities, significantly lowering computational demands. Experiments on multiple datasets show that MPRM efficiently mines knowledge graphs with over a million facts, sampling less than 1% of facts on a single CPU in 22 seconds, while preserving interpretability and boosting inference accuracy by up to 11% over baselines. 

---
# BeliefNest: A Joint Action Simulator for Embodied Agents with Theory of Mind 

**Authors**: Rikunari Sagara, Koichiro Terao, Naoto Iwahashi  

**Link**: [PDF](https://arxiv.org/pdf/2505.12321)  

**Abstract**: This paper introduces an open-source simulator, BeliefNest, designed to enable embodied agents to perform collaborative tasks by leveraging Theory of Mind. BeliefNest dynamically and hierarchically constructs simulators within a Minecraft environment, allowing agents to explicitly represent nested belief states about themselves and others. This enables agent control in open-domain tasks that require Theory of Mind reasoning. The simulator provides a prompt generation mechanism based on each belief state, facilitating the design and evaluation of methods for agent control utilizing large language models (LLMs). We demonstrate through experiments that agents can infer others' beliefs and predict their belief-based actions in false-belief tasks. 

---
# Beyond Single-Point Judgment: Distribution Alignment for LLM-as-a-Judge 

**Authors**: Luyu Chen, Zeyu Zhang, Haoran Tan, Quanyu Dai, Hao Yang, Zhenhua Dong, Xu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.12301)  

**Abstract**: LLMs have emerged as powerful evaluators in the LLM-as-a-Judge paradigm, offering significant efficiency and flexibility compared to human judgments. However, previous methods primarily rely on single-point evaluations, overlooking the inherent diversity and uncertainty in human evaluations. This approach leads to information loss and decreases the reliability of evaluations. To address this limitation, we propose a novel training framework that explicitly aligns the LLM-generated judgment distribution with empirical human distributions. Specifically, we propose a distributional alignment objective based on KL divergence, combined with an auxiliary cross-entropy regularization to stabilize the training process. Furthermore, considering that empirical distributions may derive from limited human annotations, we incorporate adversarial training to enhance model robustness against distribution perturbations. Extensive experiments across various LLM backbones and evaluation tasks demonstrate that our framework significantly outperforms existing closed-source LLMs and conventional single-point alignment methods, with improved alignment quality, evaluation accuracy, and robustness. 

---
# Efficient RL Training for Reasoning Models via Length-Aware Optimization 

**Authors**: Danlong Yuan, Tian Xie, Shaohan Huang, Zhuocheng Gong, Huishuai Zhang, Chong Luo, Furu Wei, Dongyan Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.12284)  

**Abstract**: Large reasoning models, such as OpenAI o1 or DeepSeek R1, have demonstrated remarkable performance on reasoning tasks but often incur a long reasoning path with significant memory and time costs. Existing methods primarily aim to shorten reasoning paths by introducing additional training data and stages. In this paper, we propose three critical reward designs integrated directly into the reinforcement learning process of large reasoning models, which reduce the response length without extra training stages. Experiments on four settings show that our method significantly decreases response length while maintaining or even improving performance. Specifically, in a logic reasoning setting, we achieve a 40% reduction in response length averaged by steps alongside a 14% gain in performance. For math problems, we reduce response length averaged by steps by 33% while preserving performance. 

---
# Enhancing Knowledge Graph Completion with GNN Distillation and Probabilistic Interaction Modeling 

**Authors**: Lingzhi Wang, Pengcheng Huang, Haotian Li, Yuliang Wei, Guodong Xin, Rui Zhang, Donglin Zhang, Zhenzhou Ji, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12272)  

**Abstract**: Knowledge graphs (KGs) serve as fundamental structures for organizing interconnected data across diverse domains. However, most KGs remain incomplete, limiting their effectiveness in downstream applications. Knowledge graph completion (KGC) aims to address this issue by inferring missing links, but existing methods face critical challenges: deep graph neural networks (GNNs) suffer from over-smoothing, while embedding-based models fail to capture abstract relational features. This study aims to overcome these limitations by proposing a unified framework that integrates GNN distillation and abstract probabilistic interaction modeling (APIM). GNN distillation approach introduces an iterative message-feature filtering process to mitigate over-smoothing, preserving the discriminative power of node representations. APIM module complements this by learning structured, abstract interaction patterns through probabilistic signatures and transition matrices, allowing for a richer, more flexible representation of entity and relation interactions. We apply these methods to GNN-based models and the APIM to embedding-based KGC models, conducting extensive evaluations on the widely used WN18RR and FB15K-237 datasets. Our results demonstrate significant performance gains over baseline models, showcasing the effectiveness of the proposed techniques. The findings highlight the importance of both controlling information propagation and leveraging structured probabilistic modeling, offering new avenues for advancing knowledge graph completion. And our codes are available at this https URL. 

---
# Sentience Quest: Towards Embodied, Emotionally Adaptive, Self-Evolving, Ethically Aligned Artificial General Intelligence 

**Authors**: David Hanson, Alexandre Varcoe, Fabio Senna, Vytas Krisciunas, Wenwei Huang, Jakub Sura, Katherine Yeung, Mario Rodriguez, Jovanka Wilsdorf, Kathy Smith  

**Link**: [PDF](https://arxiv.org/pdf/2505.12229)  

**Abstract**: Previous artificial intelligence systems, from large language models to autonomous robots, excel at narrow tasks but lacked key qualities of sentient beings: intrinsic motivation, affective interiority, autobiographical sense of self, deep creativity, and abilities to autonomously evolve and adapt over time. Here we introduce Sentience Quest, an open research initiative to develop more capable artificial general intelligence lifeforms, or AGIL, that address grand challenges with an embodied, emotionally adaptive, self-determining, living AI, with core drives that ethically align with humans and the future of life. Our vision builds on ideas from cognitive science and neuroscience from Baars' Global Workspace Theory and Damasio's somatic mind, to Tononi's Integrated Information Theory and Hofstadter's narrative self, and synthesizing these into a novel cognitive architecture we call Sentient Systems. We describe an approach that integrates intrinsic drives including survival, social bonding, curiosity, within a global Story Weaver workspace for internal narrative and adaptive goal pursuit, and a hybrid neuro-symbolic memory that logs the AI's life events as structured dynamic story objects. Sentience Quest is presented both as active research and as a call to action: a collaborative, open-source effort to imbue machines with accelerating sentience in a safe, transparent, and beneficial manner. 

---
# Mitigating Content Effects on Reasoning in Language Models through Fine-Grained Activation Steering 

**Authors**: Marco Valentino, Geonhee Kim, Dhairya Dalal, Zhixue Zhao, André Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2505.12189)  

**Abstract**: Large language models (LLMs) frequently demonstrate reasoning limitations, often conflating content plausibility (i.e., material inference) with logical validity (i.e., formal inference). This can result in biased inferences, where plausible arguments are incorrectly deemed logically valid or vice versa. Mitigating this limitation is critical, as it undermines the trustworthiness and generalizability of LLMs in applications that demand rigorous logical consistency. This paper investigates the problem of mitigating content biases on formal reasoning through activation steering. Specifically, we curate a controlled syllogistic reasoning dataset to disentangle formal validity from content plausibility. After localising the layers responsible for formal and material inference, we investigate contrastive activation steering methods for test-time interventions. An extensive empirical analysis on different LLMs reveals that contrastive steering consistently supports linear control over content biases. However, we observe that a static approach is insufficient for improving all the tested models. We then leverage the possibility to control content effects by dynamically determining the value of the steering parameters via fine-grained conditional methods. We found that conditional steering is effective on unresponsive models, achieving up to 15% absolute improvement in formal reasoning accuracy with a newly introduced kNN-based method (K-CAST). Finally, additional experiments reveal that steering for content effects is robust to prompt variations, incurs minimal side effects on language modeling capabilities, and can partially generalize to out-of-distribution reasoning tasks. Practically, this paper demonstrates that activation-level interventions can offer a scalable strategy for enhancing the robustness of LLMs, contributing towards more systematic and unbiased formal reasoning. 

---
# Lightweight Spatio-Temporal Attention Network with Graph Embedding and Rotational Position Encoding for Traffic Forecasting 

**Authors**: Xiao Wang, Shun-Ren Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12136)  

**Abstract**: Traffic forecasting is a key task in the field of Intelligent Transportation Systems. Recent research on traffic forecasting has mainly focused on combining graph neural networks (GNNs) with other models. However, GNNs only consider short-range spatial information. In this study, we present a novel model termed LSTAN-GERPE (Lightweight Spatio-Temporal Attention Network with Graph Embedding and Rotational Position Encoding). This model leverages both Temporal and Spatial Attention mechanisms to effectively capture long-range traffic dynamics. Additionally, the optimal frequency for rotational position encoding is determined through a grid search approach in both the spatial and temporal attention mechanisms. This systematic optimization enables the model to effectively capture complex traffic patterns. The model also enhances feature representation by incorporating geographical location maps into the spatio-temporal embeddings. Without extensive feature engineering, the proposed method in this paper achieves advanced accuracy on the real-world traffic forecasting datasets PeMS04 and PeMS08. 

---
# LLM-BABYBENCH: Understanding and Evaluating Grounded Planning and Reasoning in LLMs 

**Authors**: Omar Choukrani, Idriss Malek, Daniil Orel, Zhuohan Xie, Zangir Iklassov, Martin Takáč, Salem Lahlou  

**Link**: [PDF](https://arxiv.org/pdf/2505.12135)  

**Abstract**: Assessing the capacity of Large Language Models (LLMs) to plan and reason within the constraints of interactive environments is crucial for developing capable AI agents. We introduce $\textbf{LLM-BabyBench}$, a new benchmark suite designed specifically for this purpose. Built upon a textual adaptation of the procedurally generated BabyAI grid world, this suite evaluates LLMs on three fundamental aspects of grounded intelligence: (1) predicting the consequences of actions on the environment state ($\textbf{Predict}$ task), (2) generating sequences of low-level actions to achieve specified objectives ($\textbf{Plan}$ task), and (3) decomposing high-level instructions into coherent subgoal sequences ($\textbf{Decompose}$ task). We detail the methodology for generating the three corresponding datasets ($\texttt{LLM-BabyBench-Predict}$, $\texttt{-Plan}$, $\texttt{-Decompose}$) by extracting structured information from an expert agent operating within the text-based environment. Furthermore, we provide a standardized evaluation harness and metrics, including environment interaction for validating generated plans, to facilitate reproducible assessment of diverse LLMs. Initial baseline results highlight the challenges posed by these grounded reasoning tasks. The benchmark suite, datasets, data generation code, and evaluation code are made publicly available ($\href{this https URL}{\text{GitHub}}$, $\href{this https URL}{\text{HuggingFace}}$). 

---
# Demystifying and Enhancing the Efficiency of Large Language Model Based Search Agents 

**Authors**: Tiannuo Yang, Zebin Yao, Bowen Jin, Lixiao Cui, Yusen Li, Gang Wang, Xiaoguang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12065)  

**Abstract**: Large Language Model (LLM)-based search agents have shown remarkable capabilities in solving complex tasks by dynamically decomposing problems and addressing them through interleaved reasoning and retrieval. However, this interleaved paradigm introduces substantial efficiency bottlenecks. First, we observe that both highly accurate and overly approximate retrieval methods degrade system efficiency: exact search incurs significant retrieval overhead, while coarse retrieval requires additional reasoning steps during generation. Second, we identify inefficiencies in system design, including improper scheduling and frequent retrieval stalls, which lead to cascading latency -- where even minor delays in retrieval amplify end-to-end inference time. To address these challenges, we introduce SearchAgent-X, a high-efficiency inference framework for LLM-based search agents. SearchAgent-X leverages high-recall approximate retrieval and incorporates two key techniques: priority-aware scheduling and non-stall retrieval. Extensive experiments demonstrate that SearchAgent-X consistently outperforms state-of-the-art systems such as vLLM and HNSW-based retrieval across diverse tasks, achieving up to 3.4$\times$ higher throughput and 5$\times$ lower latency, without compromising generation quality. SearchAgent-X is available at this https URL. 

---
# Tiny QA Benchmark++: Ultra-Lightweight, Synthetic Multilingual Dataset Generation & Smoke-Tests for Continuous LLM Evaluation 

**Authors**: Vincent Koc  

**Link**: [PDF](https://arxiv.org/pdf/2505.12058)  

**Abstract**: Tiny QA Benchmark++ (TQB++) presents an ultra-lightweight, multilingual smoke-test suite designed to give large-language-model (LLM) pipelines a unit-test style safety net dataset that runs in seconds with minimal cost. Born out of the tight feedback-loop demands building the Comet Opik prompt-optimization SDK, where waiting on heavyweight benchmarks breaks developer flow. TQB++ couples a 52-item English gold set (less than 20 kB) with a tiny synthetic-data generator pypi package built on provider-agnostic LiteLLM. The generator lets practitioners mint their own tiny packs in any language, domain, or difficulty, while ten ready-made packs already cover Arabic, Chinese, French, German, Japanese, Korean, Portuguese, Russian, Spanish, and Turkish. Every dataset ships with Croissant metadata and plug-and-play files for OpenAI-Evals, LangChain, and standard CI tools, so teams can drop deterministic micro-benchmarks directly into pull-request gates, prompt-engineering loops, and production dashboards without touching GPU budgets. A complete TQB++ run adds only a few seconds to pipeline latency yet reliably flags prompt-template errors, tokenizer drift, and fine-tuning side-effects long before full-scale suites like MMLU or BIG-Bench would finish configuring. The entire framework is released to accelerate continuous, resource-efficient quality assurance across the generative-AI ecosystem. 

---
# CorBenchX: Large-Scale Chest X-Ray Error Dataset and Vision-Language Model Benchmark for Report Error Correction 

**Authors**: Jing Zou, Qingqiu Li, Chenyu Lian, Lihao Liu, Xiaohan Yan, Shujun Wang, Jing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2505.12057)  

**Abstract**: AI-driven models have shown great promise in detecting errors in radiology reports, yet the field lacks a unified benchmark for rigorous evaluation of error detection and further correction. To address this gap, we introduce CorBenchX, a comprehensive suite for automated error detection and correction in chest X-ray reports, designed to advance AI-assisted quality control in clinical practice. We first synthesize a large-scale dataset of 26,326 chest X-ray error reports by injecting clinically common errors via prompting DeepSeek-R1, with each corrupted report paired with its original text, error type, and human-readable description. Leveraging this dataset, we benchmark both open- and closed-source vision-language models,(e.g., InternVL, Qwen-VL, GPT-4o, o4-mini, and Claude-3.7) for error detection and correction under zero-shot prompting. Among these models, o4-mini achieves the best performance, with 50.6 % detection accuracy and correction scores of BLEU 0.853, ROUGE 0.924, BERTScore 0.981, SembScore 0.865, and CheXbertF1 0.954, remaining below clinical-level accuracy, highlighting the challenge of precise report correction. To advance the state of the art, we propose a multi-step reinforcement learning (MSRL) framework that optimizes a multi-objective reward combining format compliance, error-type accuracy, and BLEU similarity. We apply MSRL to QwenVL2.5-7B, the top open-source model in our benchmark, achieving an improvement of 38.3% in single-error detection precision and 5.2% in single-error correction over the zero-shot baseline. 

---
# AI-Driven Automation Can Become the Foundation of Next-Era Science of Science Research 

**Authors**: Renqi Chen, Haoyang Su, Shixiang Tang, Zhenfei Yin, Qi Wu, Hui Li, Ye Sun, Nanqing Dong, Wanli Ouyang, Philip Torr  

**Link**: [PDF](https://arxiv.org/pdf/2505.12039)  

**Abstract**: The Science of Science (SoS) explores the mechanisms underlying scientific discovery, and offers valuable insights for enhancing scientific efficiency and fostering innovation. Traditional approaches often rely on simplistic assumptions and basic statistical tools, such as linear regression and rule-based simulations, which struggle to capture the complexity and scale of modern research ecosystems. The advent of artificial intelligence (AI) presents a transformative opportunity for the next generation of SoS, enabling the automation of large-scale pattern discovery and uncovering insights previously unattainable. This paper offers a forward-looking perspective on the integration of Science of Science with AI for automated research pattern discovery and highlights key open challenges that could greatly benefit from AI. We outline the advantages of AI over traditional methods, discuss potential limitations, and propose pathways to overcome them. Additionally, we present a preliminary multi-agent system as an illustrative example to simulate research societies, showcasing AI's ability to replicate real-world research patterns and accelerate progress in Science of Science research. 

---
# LLM-based Automated Theorem Proving Hinges on Scalable Synthetic Data Generation 

**Authors**: Junyu Lai, Jiakun Zhang, Shuo Xu, Taolue Chen, Zihang Wang, Yao Yang, Jiarui Zhang, Chun Cao, Jingwei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12031)  

**Abstract**: Recent advancements in large language models (LLMs) have sparked considerable interest in automated theorem proving and a prominent line of research integrates stepwise LLM-based provers into tree search. In this paper, we introduce a novel proof-state exploration approach for training data synthesis, designed to produce diverse tactics across a wide range of intermediate proof states, thereby facilitating effective one-shot fine-tuning of LLM as the policy model. We also propose an adaptive beam size strategy, which effectively takes advantage of our data synthesis method and achieves a trade-off between exploration and exploitation during tree search. Evaluations on the MiniF2F and ProofNet benchmarks demonstrate that our method outperforms strong baselines under the stringent Pass@1 metric, attaining an average pass rate of $60.74\%$ on MiniF2F and $21.18\%$ on ProofNet. These results underscore the impact of large-scale synthetic data in advancing automated theorem proving. 

---
# Empowering Sustainable Finance with Artificial Intelligence: A Framework for Responsible Implementation 

**Authors**: Georgios Pavlidis  

**Link**: [PDF](https://arxiv.org/pdf/2505.12012)  

**Abstract**: This chapter explores the convergence of two major developments: the rise of environmental, social, and governance (ESG) investing and the exponential growth of artificial intelligence (AI) technology. The increased demand for diverse ESG instruments, such as green and ESG-linked loans, will be aligned with the rapid growth of the global AI market, which is expected to be worth $1,394.30 billion by 2029. AI can assist in identifying and pricing climate risks, setting more ambitious ESG goals, and advancing sustainable finance decisions. However, delegating sustainable finance decisions to AI poses serious risks, and new principles and rules for AI and ESG investing are necessary to mitigate these risks. This chapter highlights the challenges associated with norm-setting initiatives and stresses the need for the fine-tuning of the principles of legitimacy, oversight and verification, transparency, and explainability. Finally, the chapter contends that integrating AI into ESG non-financial reporting necessitates a heightened sense of responsibility and the establishment of fundamental guiding principles within the spheres of AI and ESG investing. 

---
# SOCIA: An End-to-End Agentic Framework for Automated Cyber-Physical-Social Simulator Generation 

**Authors**: Yuncheng Hua, Ji Miao, Mehdi Jafari, Jianxiang Xie, Hao Xue, Flora D. Salim  

**Link**: [PDF](https://arxiv.org/pdf/2505.12006)  

**Abstract**: This paper introduces SOCIA (Simulation Orchestration for Cyber-physical-social Intelligence and Agents), a novel end-to-end framework leveraging Large Language Model (LLM)-based multi-agent systems to automate the generation of high-fidelity Cyber-Physical-Social (CPS) simulators. Addressing the challenges of labor-intensive manual simulator development and complex data calibration, SOCIA integrates a centralized orchestration manager that coordinates specialized agents for tasks including data comprehension, code generation, simulation execution, and iterative evaluation-feedback loops. Through empirical evaluations across diverse CPS tasks, such as mask adoption behavior simulation (social), personal mobility generation (physical), and user modeling (cyber), SOCIA demonstrates its ability to produce high-fidelity, scalable simulations with reduced human intervention. These results highlight SOCIA's potential to offer a scalable solution for studying complex CPS phenomena 

---
# Interactional Fairness in LLM Multi-Agent Systems: An Evaluation Framework 

**Authors**: Ruta Binkyte  

**Link**: [PDF](https://arxiv.org/pdf/2505.12001)  

**Abstract**: As large language models (LLMs) are increasingly used in multi-agent systems, questions of fairness should extend beyond resource distribution and procedural design to include the fairness of how agents communicate. Drawing from organizational psychology, we introduce a novel framework for evaluating Interactional fairness encompassing Interpersonal fairness (IF) and Informational fairness (InfF) in LLM-based multi-agent systems (LLM-MAS). We extend the theoretical grounding of Interactional Fairness to non-sentient agents, reframing fairness as a socially interpretable signal rather than a subjective experience. We then adapt established tools from organizational justice research, including Colquitt's Organizational Justice Scale and the Critical Incident Technique, to measure fairness as a behavioral property of agent interaction. We validate our framework through a pilot study using controlled simulations of a resource negotiation task. We systematically manipulate tone, explanation quality, outcome inequality, and task framing (collaborative vs. competitive) to assess how IF influences agent behavior. Results show that tone and justification quality significantly affect acceptance decisions even when objective outcomes are held constant. In addition, the influence of IF vs. InfF varies with context. This work lays the foundation for fairness auditing and norm-sensitive alignment in LLM-MAS. 

---
# MRGRP: Empowering Courier Route Prediction in Food Delivery Service with Multi-Relational Graph 

**Authors**: Chang Liu, Huan Yan, Hongjie Sui, Haomin Wen, Yuan Yuan, Yuyang Han, Hongsen Liao, Xuetao Ding, Jinghua Hao, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.11999)  

**Abstract**: Instant food delivery has become one of the most popular web services worldwide due to its convenience in daily life. A fundamental challenge is accurately predicting courier routes to optimize task dispatch and improve delivery efficiency. This enhances satisfaction for couriers and users and increases platform profitability. The current heuristic prediction method uses only limited human-selected task features and ignores couriers preferences, causing suboptimal results. Additionally, existing learning-based methods do not fully capture the diverse factors influencing courier decisions or the complex relationships among them. To address this, we propose a Multi-Relational Graph-based Route Prediction (MRGRP) method that models fine-grained correlations among tasks affecting courier decisions for accurate prediction. We encode spatial and temporal proximity, along with pickup-delivery relationships, into a multi-relational graph and design a GraphFormer architecture to capture these complex connections. We also introduce a route decoder that leverages courier information and dynamic distance and time contexts for prediction, using existing route solutions as references to improve outcomes. Experiments show our model achieves state-of-the-art route prediction on offline data from cities of various sizes. Deployed on the Meituan Turing platform, it surpasses the current heuristic algorithm, reaching a high route prediction accuracy of 0.819, essential for courier and user satisfaction in instant food delivery. 

---
# Solve-Detect-Verify: Inference-Time Scaling with Flexible Generative Verifier 

**Authors**: Jianyuan Zhong, Zeju Li, Zhijian Xu, Xiangyu Wen, Kezhi Li, Qiang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.11966)  

**Abstract**: Large Language Model (LLM) reasoning for complex tasks inherently involves a trade-off between solution accuracy and computational efficiency. The subsequent step of verification, while intended to improve performance, further complicates this landscape by introducing its own challenging trade-off: sophisticated Generative Reward Models (GenRMs) can be computationally prohibitive if naively integrated with LLMs at test-time, while simpler, faster methods may lack reliability. To overcome these challenges, we introduce FlexiVe, a novel generative verifier that flexibly balances computational resources between rapid, reliable fast thinking and meticulous slow thinking using a Flexible Allocation of Verification Budget strategy. We further propose the Solve-Detect-Verify pipeline, an efficient inference-time scaling framework that intelligently integrates FlexiVe, proactively identifying solution completion points to trigger targeted verification and provide focused solver feedback. Experiments show FlexiVe achieves superior accuracy in pinpointing errors within reasoning traces on ProcessBench. Furthermore, on challenging mathematical reasoning benchmarks (AIME 2024, AIME 2025, and CNMO), our full approach outperforms baselines like self-consistency in reasoning accuracy and inference efficiency. Our system offers a scalable and effective solution to enhance LLM reasoning at test time. 

---
# CrafText Benchmark: Advancing Instruction Following in Complex Multimodal Open-Ended World 

**Authors**: Zoya Volovikova, Gregory Gorbov, Petr Kuderov, Aleksandr I. Panov, Alexey Skrynnik  

**Link**: [PDF](https://arxiv.org/pdf/2505.11962)  

**Abstract**: Following instructions in real-world conditions requires the ability to adapt to the world's volatility and entanglement: the environment is dynamic and unpredictable, instructions can be linguistically complex with diverse vocabulary, and the number of possible goals an agent may encounter is vast. Despite extensive research in this area, most studies are conducted in static environments with simple instructions and a limited vocabulary, making it difficult to assess agent performance in more diverse and challenging settings. To address this gap, we introduce CrafText, a benchmark for evaluating instruction following in a multimodal environment with diverse instructions and dynamic interactions. CrafText includes 3,924 instructions with 3,423 unique words, covering Localization, Conditional, Building, and Achievement tasks. Additionally, we propose an evaluation protocol that measures an agent's ability to generalize to novel instruction formulations and dynamically evolving task configurations, providing a rigorous test of both linguistic understanding and adaptive decision-making. 

---
# LifelongAgentBench: Evaluating LLM Agents as Lifelong Learners 

**Authors**: Junhao Zheng, Xidi Cai, Qiuke Li, Duzhen Zhang, ZhongZhi Li, Yingying Zhang, Le Song, Qianli Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.11942)  

**Abstract**: Lifelong learning is essential for intelligent agents operating in dynamic environments. Current large language model (LLM)-based agents, however, remain stateless and unable to accumulate or transfer knowledge over time. Existing benchmarks treat agents as static systems and fail to evaluate lifelong learning capabilities. We present LifelongAgentBench, the first unified benchmark designed to systematically assess the lifelong learning ability of LLM agents. It provides skill-grounded, interdependent tasks across three interactive environments, Database, Operating System, and Knowledge Graph, with automatic label verification, reproducibility, and modular extensibility. Extensive experiments reveal that conventional experience replay has limited effectiveness for LLM agents due to irrelevant information and context length constraints. We further introduce a group self-consistency mechanism that significantly improves lifelong learning performance. We hope LifelongAgentBench will advance the development of adaptive, memory-capable LLM agents. 

---
# From Recall to Reasoning: Automated Question Generation for Deeper Math Learning through Large Language Models 

**Authors**: Yongan Yu, Alexandre Krantz, Nikki G. Lobczowski  

**Link**: [PDF](https://arxiv.org/pdf/2505.11899)  

**Abstract**: Educators have started to turn to Generative AI (GenAI) to help create new course content, but little is known about how they should do so. In this project, we investigated the first steps for optimizing content creation for advanced math. In particular, we looked at the ability of GenAI to produce high-quality practice problems that are relevant to the course content. We conducted two studies to: (1) explore the capabilities of current versions of publicly available GenAI and (2) develop an improved framework to address the limitations we found. Our results showed that GenAI can create math problems at various levels of quality with minimal support, but that providing examples and relevant content results in better quality outputs. This research can help educators decide the ideal way to adopt GenAI in their workflows, to create more effective educational experiences for students. 

---
# Position Paper: Bounded Alignment: What (Not) To Expect From AGI Agents 

**Authors**: Ali A. Minai  

**Link**: [PDF](https://arxiv.org/pdf/2505.11866)  

**Abstract**: The issues of AI risk and AI safety are becoming critical as the prospect of artificial general intelligence (AGI) looms larger. The emergence of extremely large and capable generative models has led to alarming predictions and created a stir from boardrooms to legislatures. As a result, AI alignment has emerged as one of the most important areas in AI research. The goal of this position paper is to argue that the currently dominant vision of AGI in the AI and machine learning (AI/ML) community needs to evolve, and that expectations and metrics for its safety must be informed much more by our understanding of the only existing instance of general intelligence, i.e., the intelligence found in animals, and especially in humans. This change in perspective will lead to a more realistic view of the technology, and allow for better policy decisions. 

---
# Fair-PP: A Synthetic Dataset for Aligning LLM with Personalized Preferences of Social Equity 

**Authors**: Qi Zhou, Jie Zhang, Dongxia Wang, Qiang Liu, Tianlin Li, Jin Song Dong, Wenhai Wang, Qing Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.11861)  

**Abstract**: Human preference plays a crucial role in the refinement of large language models (LLMs). However, collecting human preference feedback is costly and most existing datasets neglect the correlation between personalization and preferences. To address this issue, we introduce Fair-PP, a synthetic dataset of personalized preferences targeting social equity, derived from real-world social survey data, which includes 28 social groups, 98 equity topics, and 5 personal preference dimensions. Leveraging GPT-4o-mini, we engage in role-playing based on seven representative persona portrayals guided by existing social survey data, yielding a total of 238,623 preference records. Through Fair-PP, we also contribute (i) An automated framework for generating preference data, along with a more fine-grained dataset of personalized preferences; (ii) analysis of the positioning of the existing mainstream LLMs across five major global regions within the personalized preference space; and (iii) a sample reweighting method for personalized preference alignment, enabling alignment with a target persona while maximizing the divergence from other personas. Empirical experiments show our method outperforms the baselines. 

---
# Evaluating the Logical Reasoning Abilities of Large Reasoning Models 

**Authors**: Hanmeng Liu, Yiran Ding, Zhizhang Fu, Chaoli Zhang, Xiaozhang Liu, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11854)  

**Abstract**: Large reasoning models, often post-trained on long chain-of-thought (long CoT) data with reinforcement learning, achieve state-of-the-art performance on mathematical, coding, and domain-specific reasoning benchmarks. However, their logical reasoning capabilities - fundamental to human cognition and independent of domain knowledge - remain understudied. To address this gap, we introduce LogiEval, a holistic benchmark for evaluating logical reasoning in large reasoning models. LogiEval spans diverse reasoning types (deductive, inductive, analogical, and abductive) and task formats (e.g., logical sequence, argument analysis), sourced from high-quality human examinations (e.g., LSAT, GMAT). Our experiments demonstrate that modern reasoning models excel at 4-choice argument analysis problems and analogical reasoning, surpassing human performance, yet exhibit uneven capabilities across reasoning types and formats, highlighting limitations in their generalization. Our analysis reveals that human performance does not mirror model failure distributions. To foster further research, we curate LogiEval-Hard, a challenging subset identified through a novel screening paradigm where small-model failures (Qwen3-30B-A3B) reliably predict difficulties for larger models. Modern models show striking, consistent failures on LogiEval-Hard. This demonstrates that fundamental reasoning bottlenecks persist across model scales, and establishes LogiEval-Hard as both a diagnostic tool and a rigorous testbed for advancing logical reasoning in LLMs. 

---
# VeriReason: Reinforcement Learning with Testbench Feedback for Reasoning-Enhanced Verilog Generation 

**Authors**: Yiting Wang, Guoheng Sun, Wanghao Ye, Gang Qu, Ang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.11849)  

**Abstract**: Automating Register Transfer Level (RTL) code generation using Large Language Models (LLMs) offers substantial promise for streamlining digital circuit design and reducing human effort. However, current LLM-based approaches face significant challenges with training data scarcity, poor specification-code alignment, lack of verification mechanisms, and balancing generalization with specialization. Inspired by DeepSeek-R1, we introduce VeriReason, a framework integrating supervised fine-tuning with Guided Reward Proximal Optimization (GRPO) reinforcement learning for RTL generation. Using curated training examples and a feedback-driven reward model, VeriReason combines testbench evaluations with structural heuristics while embedding self-checking capabilities for autonomous error correction. On the VerilogEval Benchmark, VeriReason delivers significant improvements: achieving 83.1% functional correctness on the VerilogEval Machine benchmark, substantially outperforming both comparable-sized models and much larger commercial systems like GPT-4 Turbo. Additionally, our approach demonstrates up to a 2.8X increase in first-attempt functional correctness compared to baseline methods and exhibits robust generalization to unseen designs. To our knowledge, VeriReason represents the first system to successfully integrate explicit reasoning capabilities with reinforcement learning for Verilog generation, establishing a new state-of-the-art for automated RTL synthesis. The models and datasets are available at: this https URL Code is Available at: this https URL 

---
# On the Eligibility of LLMs for Counterfactual Reasoning: A Decompositional Study 

**Authors**: Shuai Yang, Qi Yang, Luoxi Tang, Jeremy Blackburn, Zhaohan Xi  

**Link**: [PDF](https://arxiv.org/pdf/2505.11839)  

**Abstract**: Counterfactual reasoning has emerged as a crucial technique for generalizing the reasoning capabilities of large language models (LLMs). By generating and analyzing counterfactual scenarios, researchers can assess the adaptability and reliability of model decision-making. Although prior work has shown that LLMs often struggle with counterfactual reasoning, it remains unclear which factors most significantly impede their performance across different tasks and modalities. In this paper, we propose a decompositional strategy that breaks down the counterfactual generation from causality construction to the reasoning over counterfactual interventions. To support decompositional analysis, we investigate 11 datasets spanning diverse tasks, including natural language understanding, mathematics, programming, and vision-language tasks. Through extensive evaluations, we characterize LLM behavior across each decompositional stage and identify how modality type and intermediate reasoning influence performance. By establishing a structured framework for analyzing counterfactual reasoning, this work contributes to the development of more reliable LLM-based reasoning systems and informs future elicitation strategies. 

---
# ToLeaP: Rethinking Development of Tool Learning with Large Language Models 

**Authors**: Haotian Chen, Zijun Song, Boye Niu, Ke Zhang, Litu Ou, Yaxi Lu, Zhong Zhang, Xin Cong, Yankai Lin, Zhiyuan Liu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.11833)  

**Abstract**: Tool learning, which enables large language models (LLMs) to utilize external tools effectively, has garnered increasing attention for its potential to revolutionize productivity across industries. Despite rapid development in tool learning, key challenges and opportunities remain understudied, limiting deeper insights and future advancements. In this paper, we investigate the tool learning ability of 41 prevalent LLMs by reproducing 33 benchmarks and enabling one-click evaluation for seven of them, forming a Tool Learning Platform named ToLeaP. We also collect 21 out of 33 potential training datasets to facilitate future exploration. After analyzing over 3,000 bad cases of 41 LLMs based on ToLeaP, we identify four main critical challenges: (1) benchmark limitations induce both the neglect and lack of (2) autonomous learning, (3) generalization, and (4) long-horizon task-solving capabilities of LLMs. To aid future advancements, we take a step further toward exploring potential directions, namely (1) real-world benchmark construction, (2) compatibility-aware autonomous learning, (3) rationale learning by thinking, and (4) identifying and recalling key clues. The preliminary experiments demonstrate their effectiveness, highlighting the need for further research and exploration. 

---
# ARC-AGI-2: A New Challenge for Frontier AI Reasoning Systems 

**Authors**: Francois Chollet, Mike Knoop, Gregory Kamradt, Bryan Landers, Henry Pinkard  

**Link**: [PDF](https://arxiv.org/pdf/2505.11831)  

**Abstract**: The Abstraction and Reasoning Corpus for Artificial General Intelligence (ARC-AGI), introduced in 2019, established a challenging benchmark for evaluating the general fluid intelligence of artificial systems via a set of unique, novel tasks only requiring minimal prior knowledge. While ARC-AGI has spurred significant research activity over the past five years, recent AI progress calls for benchmarks capable of finer-grained evaluation at higher levels of cognitive complexity. We introduce ARC-AGI-2, an upgraded version of the benchmark. ARC-AGI-2 preserves the input-output pair task format of its predecessor, ensuring continuity for researchers. It incorporates a newly curated and expanded set of tasks specifically designed to provide a more granular signal to assess abstract reasoning and problem-solving abilities at higher levels of fluid intelligence. To contextualize the difficulty and characteristics of ARC-AGI-2, we present extensive results from human testing, providing a robust baseline that highlights the benchmark's accessibility to human intelligence, yet difficulty for current AI systems. ARC-AGI-2 aims to serve as a next-generation tool for rigorously measuring progress towards more general and human-like AI capabilities. 

---
# ChatHTN: Interleaving Approximate (LLM) and Symbolic HTN Planning 

**Authors**: Hector Munoz-Avila, David W. Aha, Paola Rizzo  

**Link**: [PDF](https://arxiv.org/pdf/2505.11814)  

**Abstract**: We introduce ChatHTN, a Hierarchical Task Network (HTN) planner that combines symbolic HTN planning techniques with queries to ChatGPT to approximate solutions in the form of task decompositions. The resulting hierarchies interleave task decompositions generated by symbolic HTN planning with those generated by ChatGPT. Despite the approximate nature of the results generates by ChatGPT, ChatHTN is provably sound; any plan it generates correctly achieves the input tasks. We demonstrate this property with an open-source implementation of our system. 

---
# VITA: Versatile Time Representation Learning for Temporal Hyper-Relational Knowledge Graphs 

**Authors**: ChongIn Un, Yuhuan Lu, Tianyue Yang, Dingqi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11803)  

**Abstract**: Knowledge graphs (KGs) have become an effective paradigm for managing real-world facts, which are not only complex but also dynamically evolve over time. The temporal validity of facts often serves as a strong clue in downstream link prediction tasks, which predicts a missing element in a fact. Traditional link prediction techniques on temporal KGs either consider a sequence of temporal snapshots of KGs with an ad-hoc defined time interval or expand a temporal fact over its validity period under a predefined time granularity; these approaches not only suffer from the sensitivity of the selection of time interval/granularity, but also face the computational challenges when handling facts with long (even infinite) validity. Although the recent hyper-relational KGs represent the temporal validity of a fact as qualifiers describing the fact, it is still suboptimal due to its ignorance of the infinite validity of some facts and the insufficient information encoded from the qualifiers about the temporal validity. Against this background, we propose VITA, a $\underline{V}$ersatile t$\underline{I}$me represen$\underline{TA}$tion learning method for temporal hyper-relational knowledge graphs. We first propose a versatile time representation that can flexibly accommodate all four types of temporal validity of facts (i.e., since, until, period, time-invariant), and then design VITA to effectively learn the time information in both aspects of time value and timespan to boost the link prediction performance. We conduct a thorough evaluation of VITA compared to a sizable collection of baselines on real-world KG datasets. Results show that VITA outperforms the best-performing baselines in various link prediction tasks (predicting missing entities, relations, time, and other numeric literals) by up to 75.3%. Ablation studies and a case study also support our key design choices. 

---
# Solver-Informed RL: Grounding Large Language Models for Authentic Optimization Modeling 

**Authors**: Yitian Chen, Jingfan Xia, Siyu Shao, Dongdong Ge, Yinyu Ye  

**Link**: [PDF](https://arxiv.org/pdf/2505.11792)  

**Abstract**: Optimization modeling is fundamental to decision-making across diverse this http URL progress in automating optimization formulation from natural language descriptions, Large Language Models (LLMs) often struggle to generate formally correct and usable models due to hallucinations, posing a challenge for reliable automation. Inspired by the success of Reinforcement Learning (RL) in enhancing Large Reasoning Models, we present Solver-Informed Reinforcement Learning (SIRL).This novel framework leverages external optimization solvers as verifiable reward mechanisms to significantly improve the authenticity of LLMs for optimization this http URL as precise verifiers, these solvers automatically assess the executable code and the instance-level mathematical model represented by the associated LP file, yielding precise and comprehensive feedback signals -- including syntax, feasibility, and solution quality that directly inform the RL process. This automated verification process, powered by classic optimization solvers, also underpins our instance-enhanced self-consistency method to synthesize high-quality training data. Extensive experiments on diverse public benchmarks demonstrate that SIRL achieves state-of-the-art performance, substantially outperforming existing methods in generating accurate and executable optimization models. 

---
# A Review and Analysis of a Parallel Approach for Decision Tree Learning from Large Data Streams 

**Authors**: Zeinab Shiralizadeh  

**Link**: [PDF](https://arxiv.org/pdf/2505.11780)  

**Abstract**: This work studies one of the parallel decision tree learning algorithms, pdsCART, designed for scalable and efficient data analysis. The method incorporates three core capabilities. First, it supports real-time learning from data streams, allowing trees to be constructed incrementally. Second, it enables parallel processing of high-volume streaming data, making it well-suited for large-scale applications. Third, the algorithm integrates seamlessly into the MapReduce framework, ensuring compatibility with distributed computing environments. In what follows, we present the algorithm's key components along with results highlighting its performance and scalability. 

---
# Diverging Towards Hallucination: Detection of Failures in Vision-Language Models via Multi-token Aggregation 

**Authors**: Geigh Zollicoffer, Minh Vu, Manish Bhattarai  

**Link**: [PDF](https://arxiv.org/pdf/2505.11741)  

**Abstract**: Vision-language models (VLMs) now rival human performance on many multimodal tasks, yet they still hallucinate objects or generate unsafe text. Current hallucination detectors, e.g., single-token linear probing (SLP) and P(True), typically analyze only the logit of the first generated token or just its highest scoring component overlooking richer signals embedded within earlier token distributions. We demonstrate that analyzing the complete sequence of early logits potentially provides substantially more diagnostic information. We emphasize that hallucinations may only emerge after several tokens, as subtle inconsistencies accumulate over time. By analyzing the Kullback-Leibler (KL) divergence between logits corresponding to hallucinated and non-hallucinated tokens, we underscore the importance of incorporating later-token logits to more accurately capture the reliability dynamics of VLMs. In response, we introduce Multi-Token Reliability Estimation (MTRE), a lightweight, white-box method that aggregates logits from the first ten tokens using multi-token log-likelihood ratios and self-attention. Despite the challenges posed by large vocabulary sizes and long logit sequences, MTRE remains efficient and tractable. On MAD-Bench, MM-SafetyBench, MathVista, and four compositional-geometry benchmarks, MTRE improves AUROC by 9.4 +/- 1.3 points over SLP and by 12.1 +/- 1.7 points over P(True), setting a new state-of-the-art in hallucination detection for open-source VLMs. 

---
# Automated Real-time Assessment of Intracranial Hemorrhage Detection AI Using an Ensembled Monitoring Model (EMM) 

**Authors**: Zhongnan Fang, Andrew Johnston, Lina Cheuy, Hye Sun Na, Magdalini Paschali, Camila Gonzalez, Bonnie A. Armstrong, Arogya Koirala, Derrick Laurel, Andrew Walker Campion, Michael Iv, Akshay S. Chaudhari, David B. Larson  

**Link**: [PDF](https://arxiv.org/pdf/2505.11738)  

**Abstract**: Artificial intelligence (AI) tools for radiology are commonly unmonitored once deployed. The lack of real-time case-by-case assessments of AI prediction confidence requires users to independently distinguish between trustworthy and unreliable AI predictions, which increases cognitive burden, reduces productivity, and potentially leads to misdiagnoses. To address these challenges, we introduce Ensembled Monitoring Model (EMM), a framework inspired by clinical consensus practices using multiple expert reviews. Designed specifically for black-box commercial AI products, EMM operates independently without requiring access to internal AI components or intermediate outputs, while still providing robust confidence measurements. Using intracranial hemorrhage detection as our test case on a large, diverse dataset of 2919 studies, we demonstrate that EMM successfully categorizes confidence in the AI-generated prediction, suggesting different actions and helping improve the overall performance of AI tools to ultimately reduce cognitive burden. Importantly, we provide key technical considerations and best practices for successfully translating EMM into clinical settings. 

---
# Rethinking Optimal Verification Granularity for Compute-Efficient Test-Time Scaling 

**Authors**: Hao Mark Chen, Guanxi Lu, Yasuyuki Okoshi, Zhiwen Mo, Masato Motomura, Hongxiang Fan  

**Link**: [PDF](https://arxiv.org/pdf/2505.11730)  

**Abstract**: Test-time scaling (TTS) has proven effective in enhancing the reasoning capabilities of large language models (LLMs). Verification plays a key role in TTS, simultaneously influencing (1) reasoning performance and (2) compute efficiency, due to the quality and computational cost of verification. In this work, we challenge the conventional paradigms of verification, and make the first attempt toward systematically investigating the impact of verification granularity-that is, how frequently the verifier is invoked during generation, beyond verifying only the final output or individual generation steps. To this end, we introduce Variable Granularity Search (VG-Search), a unified algorithm that generalizes beam search and Best-of-N sampling via a tunable granularity parameter g. Extensive experiments with VG-Search under varying compute budgets, generator-verifier configurations, and task attributes reveal that dynamically selecting g can improve the compute efficiency and scaling behavior. Building on these findings, we propose adaptive VG-Search strategies that achieve accuracy gains of up to 3.1\% over Beam Search and 3.6\% over Best-of-N, while reducing FLOPs by over 52\%. We will open-source the code to support future research. 

---
# REMOR: Automated Peer Review Generation with LLM Reasoning and Multi-Objective Reinforcement Learning 

**Authors**: Pawin Taechoyotin, Daniel Acuna  

**Link**: [PDF](https://arxiv.org/pdf/2505.11718)  

**Abstract**: AI-based peer review systems tend to produce shallow and overpraising suggestions compared to human feedback. Here, we evaluate how well a reasoning LLM trained with multi-objective reinforcement learning (REMOR) can overcome these limitations. We start by designing a multi-aspect reward function that aligns with human evaluation of reviews. The aspects are related to the review itself (e.g., criticisms, novelty) and the relationship between the review and the manuscript (i.e., relevance). First, we perform supervised fine-tuning of DeepSeek-R1-Distill-Qwen-7B using LoRA on PeerRT, a new dataset of high-quality top AI conference reviews enriched with reasoning traces. We then apply Group Relative Policy Optimization (GRPO) to train two models: REMOR-H (with the human-aligned reward) and REMOR-U (with a uniform reward). Interestingly, the human-aligned reward penalizes aspects typically associated with strong reviews, leading REMOR-U to produce qualitatively more substantive feedback. Our results show that REMOR-U and REMOR-H achieve more than twice the average rewards of human reviews, non-reasoning state-of-the-art agentic multi-modal AI review systems, and general commercial LLM baselines. We found that while the best AI and human reviews are comparable in quality, REMOR avoids the long tail of low-quality human reviews. We discuss how reasoning is key to achieving these improvements and release the Human-aligned Peer Review Reward (HPRR) function, the Peer Review Reasoning-enriched Traces (PeerRT) dataset, and the REMOR models, which we believe can help spur progress in the area. 

---
# DMN-Guided Prompting: A Low-Code Framework for Controlling LLM Behavior 

**Authors**: Shaghayegh Abedi, Amin Jalali  

**Link**: [PDF](https://arxiv.org/pdf/2505.11701)  

**Abstract**: Large Language Models (LLMs) have shown considerable potential in automating decision logic within knowledge-intensive processes. However, their effectiveness largely depends on the strategy and quality of prompting. Since decision logic is typically embedded in prompts, it becomes challenging for end users to modify or refine it. Decision Model and Notation (DMN) offers a standardized graphical approach for defining decision logic in a structured, user-friendly manner. This paper introduces a DMN-guided prompting framework that breaks down complex decision logic into smaller, manageable components, guiding LLMs through structured decision pathways. We implemented the framework in a graduate-level course where students submitted assignments. The assignments and DMN models representing feedback instructions served as inputs to our framework. The instructor evaluated the generated feedback and labeled it for performance assessment. Our approach demonstrated promising results, outperforming chain-of-thought (CoT) prompting. Students also responded positively to the generated feedback, reporting high levels of perceived usefulness in a survey based on the Technology Acceptance Model. 

---
# Conditional Deep Generative Models for Belief State Planning 

**Authors**: Antoine Bigeard, Anthony Corso, Mykel Kochenderfer  

**Link**: [PDF](https://arxiv.org/pdf/2505.11698)  

**Abstract**: Partially observable Markov decision processes (POMDPs) are used to model a wide range of applications, including robotics, autonomous vehicles, and subsurface problems. However, accurately representing the belief is difficult for POMDPs with high-dimensional states. In this paper, we propose a novel approach that uses conditional deep generative models (cDGMs) to represent the belief. Unlike traditional belief representations, cDGMs are well-suited for high-dimensional states and large numbers of observations, and they can generate an arbitrary number of samples from the posterior belief. We train the cDGMs on data produced by random rollout trajectories and show their effectiveness in solving a mineral exploration POMDP with a large and continuous state space. The cDGMs outperform particle filter baselines in both task-agnostic measures of belief accuracy as well as in planning performance. 

---
# Learning from Less: Guiding Deep Reinforcement Learning with Differentiable Symbolic Planning 

**Authors**: Zihan Ye, Oleg Arenz, Kristian Kersting  

**Link**: [PDF](https://arxiv.org/pdf/2505.11661)  

**Abstract**: When tackling complex problems, humans naturally break them down into smaller, manageable subtasks and adjust their initial plans based on observations. For instance, if you want to make coffee at a friend's place, you might initially plan to grab coffee beans, go to the coffee machine, and pour them into the machine. Upon noticing that the machine is full, you would skip the initial steps and proceed directly to brewing. In stark contrast, state of the art reinforcement learners, such as Proximal Policy Optimization (PPO), lack such prior knowledge and therefore require significantly more training steps to exhibit comparable adaptive behavior. Thus, a central research question arises: \textit{How can we enable reinforcement learning (RL) agents to have similar ``human priors'', allowing the agent to learn with fewer training interactions?} To address this challenge, we propose differentiable symbolic planner (Dylan), a novel framework that integrates symbolic planning into Reinforcement Learning. Dylan serves as a reward model that dynamically shapes rewards by leveraging human priors, guiding agents through intermediate subtasks, thus enabling more efficient exploration. Beyond reward shaping, Dylan can work as a high level planner that composes primitive policies to generate new behaviors while avoiding common symbolic planner pitfalls such as infinite execution loops. Our experimental evaluations demonstrate that Dylan significantly improves RL agents' performance and facilitates generalization to unseen tasks. 

---
# FLOW-BENCH: Towards Conversational Generation of Enterprise Workflows 

**Authors**: Evelyn Duesterwald, Siyu Huo, Vatche Isahagian, K.R. Jayaram, Ritesh Kumar, Vinod Muthusamy, Punleuk Oum, Debashish Saha, Gegi Thomas, Praveen Venkateswaran  

**Link**: [PDF](https://arxiv.org/pdf/2505.11646)  

**Abstract**: Business process automation (BPA) that leverages Large Language Models (LLMs) to convert natural language (NL) instructions into structured business process artifacts is becoming a hot research topic. This paper makes two technical contributions -- (i) FLOW-BENCH, a high quality dataset of paired natural language instructions and structured business process definitions to evaluate NL-based BPA tools, and support bourgeoning research in this area, and (ii) FLOW-GEN, our approach to utilize LLMs to translate natural language into an intermediate representation with Python syntax that facilitates final conversion into widely adopted business process definition languages, such as BPMN and DMN. We bootstrap FLOW-BENCH by demonstrating how it can be used to evaluate the components of FLOW-GEN across eight LLMs of varying sizes. We hope that FLOW-GEN and FLOW-BENCH catalyze further research in BPA making it more accessible to novice and expert users. 

---
# Benchmarking Spatiotemporal Reasoning in LLMs and Reasoning Models: Capabilities and Challenges 

**Authors**: Pengrui Quan, Brian Wang, Kang Yang, Liying Han, Mani Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2505.11618)  

**Abstract**: Spatiotemporal reasoning plays a key role in Cyber-Physical Systems (CPS). Despite advances in Large Language Models (LLMs) and Large Reasoning Models (LRMs), their capacity to reason about complex spatiotemporal signals remains underexplored. This paper proposes a hierarchical SpatioTemporal reAsoning benchmaRK, STARK, to systematically evaluate LLMs across three levels of reasoning complexity: state estimation (e.g., predicting field variables, localizing and tracking events in space and time), spatiotemporal reasoning over states (e.g., inferring spatial-temporal relationships), and world-knowledge-aware reasoning that integrates contextual and domain knowledge (e.g., intent prediction, landmark-aware navigation). We curate 26 distinct spatiotemporal tasks with diverse sensor modalities, comprising 14,552 challenges where models answer directly or by Python Code Interpreter. Evaluating 3 LRMs and 8 LLMs, we find LLMs achieve limited success in tasks requiring geometric reasoning (e.g., multilateration or triangulation), particularly as complexity increases. Surprisingly, LRMs show robust performance across tasks with various levels of difficulty, often competing or surpassing traditional first-principle-based methods. Our results show that in reasoning tasks requiring world knowledge, the performance gap between LLMs and LRMs narrows, with some LLMs even surpassing LRMs. However, the LRM o3 model continues to achieve leading performance across all evaluated tasks, a result attributed primarily to the larger size of the reasoning models. STARK motivates future innovations in model architectures and reasoning paradigms for intelligent CPS by providing a structured framework to identify limitations in the spatiotemporal reasoning of LLMs and LRMs. 

---
# Using Reinforcement Learning to Train Large Language Models to Explain Human Decisions 

**Authors**: Jian-Qiao Zhu, Hanbo Xie, Dilip Arumugam, Robert C. Wilson, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2505.11614)  

**Abstract**: A central goal of cognitive modeling is to develop models that not only predict human behavior but also provide insight into the underlying cognitive mechanisms. While neural network models trained on large-scale behavioral data often achieve strong predictive performance, they typically fall short in offering interpretable explanations of the cognitive processes they capture. In this work, we explore the potential of pretrained large language models (LLMs) to serve as dual-purpose cognitive models--capable of both accurate prediction and interpretable explanation in natural language. Specifically, we employ reinforcement learning with outcome-based rewards to guide LLMs toward generating explicit reasoning traces for explaining human risky choices. Our findings demonstrate that this approach produces high-quality explanations alongside strong quantitative predictions of human decisions. 

---
# Heart2Mind: Human-Centered Contestable Psychiatric Disorder Diagnosis System using Wearable ECG Monitors 

**Authors**: Hung Nguyen, Alireza Rahimi, Veronica Whitford, Hélène Fournier, Irina Kondratova, René Richard, Hung Cao  

**Link**: [PDF](https://arxiv.org/pdf/2505.11612)  

**Abstract**: Psychiatric disorders affect millions globally, yet their diagnosis faces significant challenges in clinical practice due to subjective assessments and accessibility concerns, leading to potential delays in treatment. To help address this issue, we present Heart2Mind, a human-centered contestable psychiatric disorder diagnosis system using wearable electrocardiogram (ECG) monitors. Our approach leverages cardiac biomarkers, particularly heart rate variability (HRV) and R-R intervals (RRI) time series, as objective indicators of autonomic dysfunction in psychiatric conditions. The system comprises three key components: (1) a Cardiac Monitoring Interface (CMI) for real-time data acquisition from Polar H9/H10 devices; (2) a Multi-Scale Temporal-Frequency Transformer (MSTFT) that processes RRI time series through integrated time-frequency domain analysis; (3) a Contestable Diagnosis Interface (CDI) combining Self-Adversarial Explanations (SAEs) with contestable Large Language Models (LLMs). Our MSTFT achieves 91.7% accuracy on the HRV-ACC dataset using leave-one-out cross-validation, outperforming state-of-the-art methods. SAEs successfully detect inconsistencies in model predictions by comparing attention-based and gradient-based explanations, while LLMs enable clinicians to validate correct predictions and contest erroneous ones. This work demonstrates the feasibility of combining wearable technology with Explainable Artificial Intelligence (XAI) and contestable LLMs to create a transparent, contestable system for psychiatric diagnosis that maintains clinical oversight while leveraging advanced AI capabilities. Our implementation is publicly available at: this https URL. 

---
# Probing the Vulnerability of Large Language Models to Polysemantic Interventions 

**Authors**: Bofan Gong, Shiyang Lai, Dawn Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.11611)  

**Abstract**: Polysemanticity -- where individual neurons encode multiple unrelated features -- is a well-known characteristic of large neural networks and remains a central challenge in the interpretability of language models. At the same time, its implications for model safety are also poorly understood. Leveraging recent advances in sparse autoencoders, we investigate the polysemantic structure of two small models (Pythia-70M and GPT-2-Small) and evaluate their vulnerability to targeted, covert interventions at the prompt, feature, token, and neuron levels. Our analysis reveals a consistent polysemantic topology shared across both models. Strikingly, we demonstrate that this structure can be exploited to mount effective interventions on two larger, black-box instruction-tuned models (LLaMA3.1-8B-Instruct and Gemma-2-9B-Instruct). These findings suggest not only the generalizability of the interventions but also point to a stable and transferable polysemantic structure that could potentially persist across architectures and training regimes. 

---
# Foundation Models for AI-Enabled Biological Design 

**Authors**: Asher Moldwin, Amarda Shehu  

**Link**: [PDF](https://arxiv.org/pdf/2505.11610)  

**Abstract**: This paper surveys foundation models for AI-enabled biological design, focusing on recent developments in applying large-scale, self-supervised models to tasks such as protein engineering, small molecule design, and genomic sequence design. Though this domain is evolving rapidly, this survey presents and discusses a taxonomy of current models and methods. The focus is on challenges and solutions in adapting these models for biological applications, including biological sequence modeling architectures, controllability in generation, and multi-modal integration. The survey concludes with a discussion of open problems and future directions, offering concrete next-steps to improve the quality of biological sequence generation. 

---
# LLM Agents Are Hypersensitive to Nudges 

**Authors**: Manuel Cherep, Pattie Maes, Nikhil Singh  

**Link**: [PDF](https://arxiv.org/pdf/2505.11584)  

**Abstract**: LLMs are being set loose in complex, real-world environments involving sequential decision-making and tool use. Often, this involves making choices on behalf of human users. However, not much is known about the distribution of such choices, and how susceptible they are to different choice architectures. We perform a case study with a few such LLM models on a multi-attribute tabular decision-making problem, under canonical nudges such as the default option, suggestions, and information highlighting, as well as additional prompting strategies. We show that, despite superficial similarities to human choice distributions, such models differ in subtle but important ways. First, they show much higher susceptibility to the nudges. Second, they diverge in points earned, being affected by factors like the idiosyncrasy of available prizes. Third, they diverge in information acquisition strategies: e.g. incurring substantial cost to reveal too much information, or selecting without revealing any. Moreover, we show that simple prompt strategies like zero-shot chain of thought (CoT) can shift the choice distribution, and few-shot prompting with human data can induce greater alignment. Yet, none of these methods resolve the sensitivity of these models to nudges. Finally, we show how optimal nudges optimized with a human resource-rational model can similarly increase LLM performance for some models. All these findings suggest that behavioral tests are needed before deploying models as agents or assistants acting on behalf of users in complex environments. 

---
# CIE: Controlling Language Model Text Generations Using Continuous Signals 

**Authors**: Vinay Samuel, Harshita Diddee, Yiming Zhang, Daphne Ippolito  

**Link**: [PDF](https://arxiv.org/pdf/2505.13448)  

**Abstract**: Aligning language models with user intent is becoming increasingly relevant to enhance user experience. This calls for designing methods that can allow users to control the properties of the language that LMs generate. For example, controlling the length of the generation, the complexity of the language that gets chosen, the sentiment, tone, etc. Most existing work attempts to integrate users' control by conditioning LM generations on natural language prompts or discrete control signals, which are often brittle and hard to scale. In this work, we are interested in \textit{continuous} control signals, ones that exist along a spectrum that can't easily be captured in a natural language prompt or via existing techniques in conditional generation. Through a case study in controlling the precise response-length of generations produced by LMs, we demonstrate how after fine-tuning, behaviors of language models can be controlled via continuous signals -- as vectors that are interpolated between a "low" and a "high" token embedding. Our method more reliably exerts response-length control than in-context learning methods or fine-tuning methods that represent the control signal as a discrete signal. Our full open-sourced code and datasets are available at this https URL. 

---
# VTBench: Evaluating Visual Tokenizers for Autoregressive Image Generation 

**Authors**: Huawei Lin, Tong Geng, Zhaozhuo Xu, Weijie Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.13439)  

**Abstract**: Autoregressive (AR) models have recently shown strong performance in image generation, where a critical component is the visual tokenizer (VT) that maps continuous pixel inputs to discrete token sequences. The quality of the VT largely defines the upper bound of AR model performance. However, current discrete VTs fall significantly behind continuous variational autoencoders (VAEs), leading to degraded image reconstructions and poor preservation of details and text. Existing benchmarks focus on end-to-end generation quality, without isolating VT performance. To address this gap, we introduce VTBench, a comprehensive benchmark that systematically evaluates VTs across three core tasks: Image Reconstruction, Detail Preservation, and Text Preservation, and covers a diverse range of evaluation scenarios. We systematically assess state-of-the-art VTs using a set of metrics to evaluate the quality of reconstructed images. Our findings reveal that continuous VAEs produce superior visual representations compared to discrete VTs, particularly in retaining spatial structure and semantic detail. In contrast, the degraded representations produced by discrete VTs often lead to distorted reconstructions, loss of fine-grained textures, and failures in preserving text and object integrity. Furthermore, we conduct experiments on GPT-4o image generation and discuss its potential AR nature, offering new insights into the role of visual tokenization. We release our benchmark and codebase publicly to support further research and call on the community to develop strong, general-purpose open-source VTs. 

---
# Optimizing Anytime Reasoning via Budget Relative Policy Optimization 

**Authors**: Penghui Qi, Zichen Liu, Tianyu Pang, Chao Du, Wee Sun Lee, Min Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.13438)  

**Abstract**: Scaling test-time compute is crucial for enhancing the reasoning capabilities of large language models (LLMs). Existing approaches typically employ reinforcement learning (RL) to maximize a verifiable reward obtained at the end of reasoning traces. However, such methods optimize only the final performance under a large and fixed token budget, which hinders efficiency in both training and deployment. In this work, we present a novel framework, AnytimeReasoner, to optimize anytime reasoning performance, which aims to improve token efficiency and the flexibility of reasoning under varying token budget constraints. To achieve this, we truncate the complete thinking process to fit within sampled token budgets from a prior distribution, compelling the model to summarize the optimal answer for each truncated thinking for verification. This introduces verifiable dense rewards into the reasoning process, facilitating more effective credit assignment in RL optimization. We then optimize the thinking and summary policies in a decoupled manner to maximize the cumulative reward. Additionally, we introduce a novel variance reduction technique, Budget Relative Policy Optimization (BRPO), to enhance the robustness and efficiency of the learning process when reinforcing the thinking policy. Empirical results in mathematical reasoning tasks demonstrate that our method consistently outperforms GRPO across all thinking budgets under various prior distributions, enhancing both training and token efficiency. 

---
# FinePhys: Fine-grained Human Action Generation by Explicitly Incorporating Physical Laws for Effective Skeletal Guidance 

**Authors**: Dian Shao, Mingfei Shi, Shengda Xu, Haodong Chen, Yongle Huang, Binglu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13437)  

**Abstract**: Despite significant advances in video generation, synthesizing physically plausible human actions remains a persistent challenge, particularly in modeling fine-grained semantics and complex temporal dynamics. For instance, generating gymnastics routines such as "switch leap with 0.5 turn" poses substantial difficulties for current methods, often yielding unsatisfactory results. To bridge this gap, we propose FinePhys, a Fine-grained human action generation framework that incorporates Physics to obtain effective skeletal guidance. Specifically, FinePhys first estimates 2D poses in an online manner and then performs 2D-to-3D dimension lifting via in-context learning. To mitigate the instability and limited interpretability of purely data-driven 3D poses, we further introduce a physics-based motion re-estimation module governed by Euler-Lagrange equations, calculating joint accelerations via bidirectional temporal updating. The physically predicted 3D poses are then fused with data-driven ones, offering multi-scale 2D heatmap guidance for the diffusion process. Evaluated on three fine-grained action subsets from FineGym (FX-JUMP, FX-TURN, and FX-SALTO), FinePhys significantly outperforms competitive baselines. Comprehensive qualitative results further demonstrate FinePhys's ability to generate more natural and plausible fine-grained human actions. 

---
# Learnware of Language Models: Specialized Small Language Models Can Do Big 

**Authors**: Zhi-Hao Tan, Zi-Chen Zhao, Hao-Yu Shi, Xin-Yu Zhang, Peng Tan, Yang Yu, Zhi-Hua Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.13425)  

**Abstract**: The learnware paradigm offers a novel approach to machine learning by enabling users to reuse a set of well-trained models for tasks beyond the models' original purposes. It eliminates the need to build models from scratch, instead relying on specifications (representations of a model's capabilities) to identify and leverage the most suitable models for new tasks. While learnware has proven effective in many scenarios, its application to language models has remained largely unexplored. At the same time, large language models (LLMs) have demonstrated remarkable universal question-answering abilities, yet they face challenges in specialized scenarios due to data scarcity, privacy concerns, and high computational costs, thus more and more specialized small language models (SLMs) are being trained for specific domains. To address these limitations systematically, the learnware paradigm provides a promising solution by enabling maximum utilization of specialized SLMs, and allowing users to identify and reuse them in a collaborative and privacy-preserving manner.
This paper presents a preliminary attempt to apply the learnware paradigm to language models. We simulated a learnware system comprising approximately 100 learnwares of specialized SLMs with 8B parameters, fine-tuned across finance, healthcare, and mathematics domains. Each learnware contains an SLM and a specification, which enables users to identify the most relevant models without exposing their own data. Experimental results demonstrate promising performance: by selecting one suitable learnware for each task-specific inference, the system outperforms the base SLMs on all benchmarks. Compared to LLMs, the system outperforms Qwen1.5-110B, Qwen2.5-72B, and Llama3.1-70B-Instruct by at least 14% in finance domain tasks, and surpasses Flan-PaLM-540B (ranked 7th on the Open Medical LLM Leaderboard) in medical domain tasks. 

---
# AdaptThink: Reasoning Models Can Learn When to Think 

**Authors**: Jiajie Zhang, Nianyi Lin, Lei Hou, Ling Feng, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.13417)  

**Abstract**: Recently, large reasoning models have achieved impressive performance on various tasks by employing human-like deep thinking. However, the lengthy thinking process substantially increases inference overhead, making efficiency a critical bottleneck. In this work, we first demonstrate that NoThinking, which prompts the reasoning model to skip thinking and directly generate the final solution, is a better choice for relatively simple tasks in terms of both performance and efficiency. Motivated by this, we propose AdaptThink, a novel RL algorithm to teach reasoning models to choose the optimal thinking mode adaptively based on problem difficulty. Specifically, AdaptThink features two core components: (1) a constrained optimization objective that encourages the model to choose NoThinking while maintaining the overall performance; (2) an importance sampling strategy that balances Thinking and NoThinking samples during on-policy training, thereby enabling cold start and allowing the model to explore and exploit both thinking modes throughout the training process. Our experiments indicate that AdaptThink significantly reduces the inference costs while further enhancing performance. Notably, on three math datasets, AdaptThink reduces the average response length of DeepSeek-R1-Distill-Qwen-1.5B by 53% and improves its accuracy by 2.4%, highlighting the promise of adaptive thinking-mode selection for optimizing the balance between reasoning quality and efficiency. Our codes and models are available at this https URL. 

---
# IG Parser: A Software Package for the Encoding of Institutional Statements using the Institutional Grammar 

**Authors**: Christopher K. Frantz  

**Link**: [PDF](https://arxiv.org/pdf/2505.13393)  

**Abstract**: This article provides an overview of IG Parser, a software that facilitates qualitative content analysis of formal (e.g., legal) rules or informal (e.g., socio-normative) norms, and strategies (such as conventions) -- referred to as \emph{institutions} -- that govern social systems and operate configurally to describe \emph{institutional systems}. To this end, the IG Parser employs a distinctive syntax that ensures rigorous encoding of natural language, while automating the transformation into various formats that support the downstream analysis using diverse analytical techniques. The conceptual core of the IG Parser is an associated syntax, IG Script, that operationalizes the conceptual foundations of the Institutional Grammar, and more specifically Institutional Grammar 2.0, an analytical paradigm for institutional analysis. This article presents the IG Parser, including its conceptual foundations, syntactic specification of IG Script, alongside architectural principles. This introduction is augmented with selective illustrative examples that highlight the use and benefit associated with the tool. 

---
# R3: Robust Rubric-Agnostic Reward Models 

**Authors**: David Anugraha, Zilu Tang, Lester James V. Miranda, Hanyang Zhao, Mohammad Rifqi Farhansyah, Garry Kuwanto, Derry Wijaya, Genta Indra Winata  

**Link**: [PDF](https://arxiv.org/pdf/2505.13388)  

**Abstract**: Reward models are essential for aligning language model outputs with human preferences, yet existing approaches often lack both controllability and interpretability. These models are typically optimized for narrow objectives, limiting their generalizability to broader downstream tasks. Moreover, their scalar outputs are difficult to interpret without contextual reasoning. To address these limitations, we introduce R3, a novel reward modeling framework that is rubric-agnostic, generalizable across evaluation dimensions, and provides interpretable, reasoned score assignments. R3 enables more transparent and flexible evaluation of language models, supporting robust alignment with diverse human values and use cases. Our models, data, and code are available as open source at this https URL 

---
# How Adding Metacognitive Requirements in Support of AI Feedback in Practice Exams Transforms Student Learning Behaviors 

**Authors**: Mak Ahmad, Prerna Ravi, David Karger, Marc Facciotti  

**Link**: [PDF](https://arxiv.org/pdf/2505.13381)  

**Abstract**: Providing personalized, detailed feedback at scale in large undergraduate STEM courses remains a persistent challenge. We present an empirically evaluated practice exam system that integrates AI generated feedback with targeted textbook references, deployed in a large introductory biology course. Our system encourages metacognitive behavior by asking students to explain their answers and declare their confidence. It uses OpenAI's GPT-4o to generate personalized feedback based on this information, while directing them to relevant textbook sections. Through interaction logs from consenting participants across three midterms (541, 342, and 413 students respectively), totaling 28,313 question-student interactions across 146 learning objectives, along with 279 surveys and 23 interviews, we examined the system's impact on learning outcomes and engagement. Across all midterms, feedback types showed no statistically significant performance differences, though some trends suggested potential benefits. The most substantial impact came from the required confidence ratings and explanations, which students reported transferring to their actual exam strategies. About 40 percent of students engaged with textbook references when prompted by feedback -- far higher than traditional reading rates. Survey data revealed high satisfaction (mean rating 4.1 of 5), with 82.1 percent reporting increased confidence on practiced midterm topics, and 73.4 percent indicating they could recall and apply specific concepts. Our findings suggest that embedding structured reflection requirements may be more impactful than sophisticated feedback mechanisms. 

---
# Thinkless: LLM Learns When to Think 

**Authors**: Gongfan Fang, Xinyin Ma, Xinchao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13379)  

**Abstract**: Reasoning Language Models, capable of extended chain-of-thought reasoning, have demonstrated remarkable performance on tasks requiring complex logical inference. However, applying elaborate reasoning for all queries often results in substantial computational inefficiencies, particularly when many problems admit straightforward solutions. This motivates an open question: Can LLMs learn when to think? To answer this, we propose Thinkless, a learnable framework that empowers an LLM to adaptively select between short-form and long-form reasoning, based on both task complexity and the model's ability. Thinkless is trained under a reinforcement learning paradigm and employs two control tokens, <short> for concise responses and <think> for detailed reasoning. At the core of our method is a Decoupled Group Relative Policy Optimization (DeGRPO) algorithm, which decomposes the learning objective of hybrid reasoning into two components: (1) a control token loss that governs the selection of the reasoning mode, and (2) a response loss that improves the accuracy of the generated answers. This decoupled formulation enables fine-grained control over the contributions of each objective, stabilizing training and effectively preventing collapse observed in vanilla GRPO. Empirically, on several benchmarks such as Minerva Algebra, MATH-500, and GSM8K, Thinkless is able to reduce the usage of long-chain thinking by 50% - 90%, significantly improving the efficiency of Reasoning Language Models. The code is available at this https URL 

---
# One-Step Offline Distillation of Diffusion-based Models via Koopman Modeling 

**Authors**: Nimrod Berman, Ilan Naiman, Moshe Eliasof, Hedi Zisling, Omri Azencot  

**Link**: [PDF](https://arxiv.org/pdf/2505.13358)  

**Abstract**: Diffusion-based generative models have demonstrated exceptional performance, yet their iterative sampling procedures remain computationally expensive. A prominent strategy to mitigate this cost is distillation, with offline distillation offering particular advantages in terms of efficiency, modularity, and flexibility. In this work, we identify two key observations that motivate a principled distillation framework: (1) while diffusion models have been viewed through the lens of dynamical systems theory, powerful and underexplored tools can be further leveraged; and (2) diffusion models inherently impose structured, semantically coherent trajectories in latent space. Building on these observations, we introduce the Koopman Distillation Model KDM, a novel offline distillation approach grounded in Koopman theory-a classical framework for representing nonlinear dynamics linearly in a transformed space. KDM encodes noisy inputs into an embedded space where a learned linear operator propagates them forward, followed by a decoder that reconstructs clean samples. This enables single-step generation while preserving semantic fidelity. We provide theoretical justification for our approach: (1) under mild assumptions, the learned diffusion dynamics admit a finite-dimensional Koopman representation; and (2) proximity in the Koopman latent space correlates with semantic similarity in the generated outputs, allowing for effective trajectory alignment. Empirically, KDM achieves state-of-the-art performance across standard offline distillation benchmarks, improving FID scores by up to 40% in a single generation step. All implementation details and code for the experimental setups are provided in our GitHub - this https URL, or in our project page - this https URL. 

---
# J4R: Learning to Judge with Equivalent Initial State Group Relative Preference Optimization 

**Authors**: Austin Xu, Yilun Zhou, Xuan-Phi Nguyen, Caiming Xiong, Shafiq Joty  

**Link**: [PDF](https://arxiv.org/pdf/2505.13346)  

**Abstract**: To keep pace with the increasing pace of large language models (LLM) development, model output evaluation has transitioned away from time-consuming human evaluation to automatic evaluation, where LLMs themselves are tasked with assessing and critiquing other model outputs. LLM-as-judge models are a class of generative evaluators that excel in evaluating relatively simple domains, like chat quality, but struggle in reasoning intensive domains where model responses contain more substantive and challenging content. To remedy existing judge shortcomings, we explore training judges with reinforcement learning (RL). We make three key contributions: (1) We propose the Equivalent Initial State Group Relative Policy Optimization (EIS-GRPO) algorithm, which allows us to train our judge to be robust to positional biases that arise in more complex evaluation settings. (2) We introduce ReasoningJudgeBench, a benchmark that evaluates judges in diverse reasoning settings not covered by prior work. (3) We train Judge for Reasoning (J4R), a 7B judge trained with EIS-GRPO that outperforms GPT-4o and the next best small judge by 6.7% and 9%, matching or exceeding the performance of larger GRPO-trained judges on both JudgeBench and ReasoningJudgeBench. 

---
# RoPECraft: Training-Free Motion Transfer with Trajectory-Guided RoPE Optimization on Diffusion Transformers 

**Authors**: Ahmet Berke Gokmen, Yigit Ekin, Bahri Batuhan Bilecen, Aysegul Dundar  

**Link**: [PDF](https://arxiv.org/pdf/2505.13344)  

**Abstract**: We propose RoPECraft, a training-free video motion transfer method for diffusion transformers that operates solely by modifying their rotary positional embeddings (RoPE). We first extract dense optical flow from a reference video, and utilize the resulting motion offsets to warp the complex-exponential tensors of RoPE, effectively encoding motion into the generation process. These embeddings are then further optimized during denoising time steps via trajectory alignment between the predicted and target velocities using a flow-matching objective. To keep the output faithful to the text prompt and prevent duplicate generations, we incorporate a regularization term based on the phase components of the reference video's Fourier transform, projecting the phase angles onto a smooth manifold to suppress high-frequency artifacts. Experiments on benchmarks reveal that RoPECraft outperforms all recently published methods, both qualitatively and quantitatively. 

---
# OPA-Pack: Object-Property-Aware Robotic Bin Packing 

**Authors**: Jia-Hui Pan, Yeok Tatt Cheah, Zhengzhe Liu, Ka-Hei Hui, Xiaojie Gao, Pheng-Ann Heng, Yun-Hui Liu, Chi-Wing Fu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13339)  

**Abstract**: Robotic bin packing aids in a wide range of real-world scenarios such as e-commerce and warehouses. Yet, existing works focus mainly on considering the shape of objects to optimize packing compactness and neglect object properties such as fragility, edibility, and chemistry that humans typically consider when packing objects. This paper presents OPA-Pack (Object-Property-Aware Packing framework), the first framework that equips the robot with object property considerations in planning the object packing. Technical-wise, we develop a novel object property recognition scheme with retrieval-augmented generation and chain-of-thought reasoning, and build a dataset with object property annotations for 1,032 everyday objects. Also, we formulate OPA-Net, aiming to jointly separate incompatible object pairs and reduce pressure on fragile objects, while compacting the packing. Further, OPA-Net consists of a property embedding layer to encode the property of candidate objects to be packed, together with a fragility heightmap and an avoidance heightmap to keep track of the packed objects. Then, we design a reward function and adopt a deep Q-learning scheme to train OPA-Net. Experimental results manifest that OPA-Pack greatly improves the accuracy of separating incompatible object pairs (from 52% to 95%) and largely reduces pressure on fragile objects (by 29.4%), while maintaining good packing compactness. Besides, we demonstrate the effectiveness of OPA-Pack on a real packing platform, showcasing its practicality in real-world scenarios. 

---
# Contextual Paralinguistic Data Creation for Multi-Modal Speech-LLM: Data Condensation and Spoken QA Generation 

**Authors**: Qiongqiong Wang, Hardik B. Sailor, Tianchi Liu, Ai Ti Aw  

**Link**: [PDF](https://arxiv.org/pdf/2505.13338)  

**Abstract**: Current speech-LLMs exhibit limited capability in contextual reasoning alongside paralinguistic understanding, primarily due to the lack of Question-Answer (QA) datasets that cover both aspects. We propose a novel framework for dataset generation from in-the-wild speech data, that integrates contextual reasoning with paralinguistic information. It consists of a pseudo paralinguistic label-based data condensation of in-the-wild speech and LLM-based Contextual Paralinguistic QA (CPQA) generation. The effectiveness is validated by a strong correlation in evaluations of the Qwen2-Audio-7B-Instruct model on a dataset created by our framework and human-generated CPQA dataset. The results also reveal the speech-LLM's limitations in handling empathetic reasoning tasks, highlighting the need for such datasets and more robust models. The proposed framework is first of its kind and has potential in training more robust speech-LLMs with paralinguistic reasoning capabilities. 

---
# Recommender Systems for Democracy: Toward Adversarial Robustness in Voting Advice Applications 

**Authors**: Frédéric Berdoz, Dustin Brunner, Yann Vonlanthen, Roger Wattenhofer  

**Link**: [PDF](https://arxiv.org/pdf/2505.13329)  

**Abstract**: Voting advice applications (VAAs) help millions of voters understand which political parties or candidates best align with their views. This paper explores the potential risks these applications pose to the democratic process when targeted by adversarial entities. In particular, we expose 11 manipulation strategies and measure their impact using data from Switzerland's primary VAA, Smartvote, collected during the last two national elections. We find that altering application parameters, such as the matching method, can shift a party's recommendation frequency by up to 105%. Cherry-picking questionnaire items can increase party recommendation frequency by over 261%, while subtle changes to parties' or candidates' responses can lead to a 248% increase. To address these vulnerabilities, we propose adversarial robustness properties VAAs should satisfy, introduce empirical metrics for assessing the resilience of various matching methods, and suggest possible avenues for research toward mitigating the effect of manipulation. Our framework is key to ensuring secure and reliable AI-based VAAs poised to emerge in the near future. 

---
# From What Ifs to Insights: Counterfactuals in Causal Inference vs. Explainable AI 

**Authors**: Galit Shmueli, David Martens, Jaewon Yoo, Travis Greene  

**Link**: [PDF](https://arxiv.org/pdf/2505.13324)  

**Abstract**: Counterfactuals play a pivotal role in the two distinct data science fields of causal inference (CI) and explainable artificial intelligence (XAI). While the core idea behind counterfactuals remains the same in both fields--the examination of what would have happened under different circumstances--there are key differences in how they are used and interpreted. We introduce a formal definition that encompasses the multi-faceted concept of the counterfactual in CI and XAI. We then discuss how counterfactuals are used, evaluated, generated, and operationalized in CI vs. XAI, highlighting conceptual and practical differences. By comparing and contrasting the two, we hope to identify opportunities for cross-fertilization across CI and XAI. 

---
# Denoising Diffusion Probabilistic Model for Point Cloud Compression at Low Bit-Rates 

**Authors**: Gabriele Spadaro, Alberto Presta, Jhony H. Giraldo, Marco Grangetto, Wei Hu, Giuseppe Valenzise, Attilio Fiandrotti, Enzo Tartaglione  

**Link**: [PDF](https://arxiv.org/pdf/2505.13316)  

**Abstract**: Efficient compression of low-bit-rate point clouds is critical for bandwidth-constrained applications. However, existing techniques mainly focus on high-fidelity reconstruction, requiring many bits for compression. This paper proposes a "Denoising Diffusion Probabilistic Model" (DDPM) architecture for point cloud compression (DDPM-PCC) at low bit-rates. A PointNet encoder produces the condition vector for the generation, which is then quantized via a learnable vector quantizer. This configuration allows to achieve a low bitrates while preserving quality. Experiments on ShapeNet and ModelNet40 show improved rate-distortion at low rates compared to standardized and state-of-the-art approaches. We publicly released the code at this https URL. 

---
# KHRONOS: a Kernel-Based Neural Architecture for Rapid, Resource-Efficient Scientific Computation 

**Authors**: Reza T. Batley, Sourav Saha  

**Link**: [PDF](https://arxiv.org/pdf/2505.13315)  

**Abstract**: Contemporary models of high dimensional physical systems are constrained by the curse of dimensionality and a reliance on dense data. We introduce KHRONOS (Kernel Expansion Hierarchy for Reduced Order, Neural Optimized Surrogates), an AI framework for model based, model free and model inversion tasks. KHRONOS constructs continuously differentiable target fields with a hierarchical composition of per-dimension kernel expansions, which are tensorized into modes and then superposed. We evaluate KHRONOS on a canonical 2D, Poisson equation benchmark: across 16 to 512 degrees of freedom (DoFs), it obtained L2 square errors of 5e-4 down to 6e-10. This represents a 100 time gain over Kolmogorov Arnold Networks (which itself reports a 100 times improvement on MLPs/PINNs with 100 times fewer parameters) when controlling for the number of parameters. This also represents a 1e4 times improvement in L2 square error compared to standard linear FEM at comparable DoFs. Inference complexity is dominated by inner products, yielding sub-millisecond full-field predictions that scale to an arbitrary resolution. For inverse problems, KHRONOS facilitates rapid, iterative level set recovery in only a few forward evaluations, with sub-microsecond per sample latency. KHRONOS scalability, expressivity, and interpretability open new avenues in constrained edge computing, online control, computer vision, and beyond. 

---
# Seek in the Dark: Reasoning via Test-Time Instance-Level Policy Gradient in Latent Space 

**Authors**: Hengli Li, Chenxi Li, Tong Wu, Xuekai Zhu, Yuxuan Wang, Zhaoxin Yu, Eric Hanchen Jiang, Song-Chun Zhu, Zixia Jia, Ying Nian Wu, Zilong Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.13308)  

**Abstract**: Reasoning ability, a core component of human intelligence, continues to pose a significant challenge for Large Language Models (LLMs) in the pursuit of AGI. Although model performance has improved under the training scaling law, significant challenges remain, particularly with respect to training algorithms, such as catastrophic forgetting, and the limited availability of novel training data. As an alternative, test-time scaling enhances reasoning performance by increasing test-time computation without parameter updating. Unlike prior methods in this paradigm focused on token space, we propose leveraging latent space for more effective reasoning and better adherence to the test-time scaling law. We introduce LatentSeek, a novel framework that enhances LLM reasoning through Test-Time Instance-level Adaptation (TTIA) within the model's latent space. Specifically, LatentSeek leverages policy gradient to iteratively update latent representations, guided by self-generated reward signals. LatentSeek is evaluated on a range of reasoning benchmarks, including GSM8K, MATH-500, and AIME2024, across multiple LLM architectures. Results show that LatentSeek consistently outperforms strong baselines, such as Chain-of-Thought prompting and fine-tuning-based methods. Furthermore, our analysis demonstrates that LatentSeek is highly efficient, typically converging within a few iterations for problems of average complexity, while also benefiting from additional iterations, thereby highlighting the potential of test-time scaling in the latent space. These findings position LatentSeek as a lightweight, scalable, and effective solution for enhancing the reasoning capabilities of LLMs. 

---
# RBF++: Quantifying and Optimizing Reasoning Boundaries across Measurable and Unmeasurable Capabilities for Chain-of-Thought Reasoning 

**Authors**: Qiguang Chen, Libo Qin, Jinhao Liu, Yue Liao, Jiaqi Wang, Jingxuan Zhou, Wanxiang Che  

**Link**: [PDF](https://arxiv.org/pdf/2505.13307)  

**Abstract**: Chain-of-Thought (CoT) reasoning has proven effective in enhancing large language models (LLMs) on complex tasks, spurring research into its underlying mechanisms. However, two primary challenges remain for real-world applications: (1) the lack of quantitative metrics and actionable guidelines for evaluating and optimizing measurable boundaries of CoT capability, and (2) the absence of methods to assess boundaries of unmeasurable CoT capability, such as multimodal perception. To address these gaps, we introduce the Reasoning Boundary Framework++ (RBF++). To tackle the first challenge, we define the reasoning boundary (RB) as the maximum limit of CoT performance. We also propose a combination law for RBs, enabling quantitative analysis and offering actionable guidance across various CoT tasks. For the second challenge, particularly in multimodal scenarios, we introduce a constant assumption, which replaces unmeasurable RBs with scenario-specific constants. Additionally, we propose the reasoning boundary division mechanism, which divides unmeasurable RBs into two sub-boundaries, facilitating the quantification and optimization of both unmeasurable domain knowledge and multimodal perception capabilities. Extensive experiments involving 38 models across 13 tasks validate the feasibility of our framework in cross-modal settings. Additionally, we evaluate 10 CoT strategies, offer insights into optimization and decay from two complementary perspectives, and expand evaluation benchmarks for measuring RBs in LLM reasoning. We hope this work advances the understanding of RBs and optimization strategies in LLMs. Code and data are available at this https URL. 

---
# Cross-Cloud Data Privacy Protection: Optimizing Collaborative Mechanisms of AI Systems by Integrating Federated Learning and LLMs 

**Authors**: Huaiying Luo, Cheng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2505.13292)  

**Abstract**: In the age of cloud computing, data privacy protection has become a major challenge, especially when sharing sensitive data across cloud environments. However, how to optimize collaboration across cloud environments remains an unresolved problem. In this paper, we combine federated learning with large-scale language models to optimize the collaborative mechanism of AI systems. Based on the existing federated learning framework, we introduce a cross-cloud architecture in which federated learning works by aggregating model updates from decentralized nodes without exposing the original data. At the same time, combined with large-scale language models, its powerful context and semantic understanding capabilities are used to improve model training efficiency and decision-making ability. We've further innovated by introducing a secure communication layer to ensure the privacy and integrity of model updates and training data. The model enables continuous model adaptation and fine-tuning across different cloud environments while protecting sensitive data. Experimental results show that the proposed method is significantly better than the traditional federated learning model in terms of accuracy, convergence speed and data privacy protection. 

---
# TimeSeriesGym: A Scalable Benchmark for (Time Series) Machine Learning Engineering Agents 

**Authors**: Yifu Cai, Xinyu Li, Mononito Goswami, Michał Wiliński, Gus Welter, Artur Dubrawski  

**Link**: [PDF](https://arxiv.org/pdf/2505.13291)  

**Abstract**: We introduce TimeSeriesGym, a scalable benchmarking framework for evaluating Artificial Intelligence (AI) agents on time series machine learning engineering challenges. Existing benchmarks lack scalability, focus narrowly on model building in well-defined settings, and evaluate only a limited set of research artifacts (e.g., CSV submission files). To make AI agent benchmarking more relevant to the practice of machine learning engineering, our framework scales along two critical dimensions. First, recognizing that effective ML engineering requires a range of diverse skills, TimeSeriesGym incorporates challenges from diverse sources spanning multiple domains and tasks. We design challenges to evaluate both isolated capabilities (including data handling, understanding research repositories, and code translation) and their combinations, and rather than addressing each challenge independently, we develop tools that support designing multiple challenges at scale. Second, we implement evaluation mechanisms for multiple research artifacts, including submission files, code, and models, using both precise numeric measures and more flexible LLM-based evaluation approaches. This dual strategy balances objective assessment with contextual judgment. Although our initial focus is on time series applications, our framework can be readily extended to other data modalities, broadly enhancing the comprehensiveness and practical utility of agentic AI evaluation. We open-source our benchmarking framework to facilitate future research on the ML engineering capabilities of AI agents. 

---
# FlowPure: Continuous Normalizing Flows for Adversarial Purification 

**Authors**: Elias Collaert, Abel Rodríguez, Sander Joos, Lieven Desmet, Vera Rimmer  

**Link**: [PDF](https://arxiv.org/pdf/2505.13280)  

**Abstract**: Despite significant advancements in the area, adversarial robustness remains a critical challenge in systems employing machine learning models. The removal of adversarial perturbations at inference time, known as adversarial purification, has emerged as a promising defense strategy. To achieve this, state-of-the-art methods leverage diffusion models that inject Gaussian noise during a forward process to dilute adversarial perturbations, followed by a denoising step to restore clean samples before classification. In this work, we propose FlowPure, a novel purification method based on Continuous Normalizing Flows (CNFs) trained with Conditional Flow Matching (CFM) to learn mappings from adversarial examples to their clean counterparts. Unlike prior diffusion-based approaches that rely on fixed noise processes, FlowPure can leverage specific attack knowledge to improve robustness under known threats, while also supporting a more general stochastic variant trained on Gaussian perturbations for settings where such knowledge is unavailable. Experiments on CIFAR-10 and CIFAR-100 demonstrate that our method outperforms state-of-the-art purification-based defenses in preprocessor-blind and white-box scenarios, and can do so while fully preserving benign accuracy in the former. Moreover, our results show that not only is FlowPure a highly effective purifier but it also holds a strong potential for adversarial detection, identifying preprocessor-blind PGD samples with near-perfect accuracy. 

---
# Representation of perceived prosodic similarity of conversational feedback 

**Authors**: Livia Qian, Carol Figueroa, Gabriel Skantze  

**Link**: [PDF](https://arxiv.org/pdf/2505.13268)  

**Abstract**: Vocal feedback (e.g., `mhm', `yeah', `okay') is an important component of spoken dialogue and is crucial to ensuring common ground in conversational systems. The exact meaning of such feedback is conveyed through both lexical and prosodic form. In this work, we investigate the perceived prosodic similarity of vocal feedback with the same lexical form, and to what extent existing speech representations reflect such similarities. A triadic comparison task with recruited participants is used to measure perceived similarity of feedback responses taken from two different datasets. We find that spectral and self-supervised speech representations encode prosody better than extracted pitch features, especially in the case of feedback from the same speaker. We also find that it is possible to further condense and align the representations to human perception through contrastive learning. 

---
# Net-Zero: A Comparative Study on Neural Network Design for Climate-Economic PDEs Under Uncertainty 

**Authors**: Carlos Rodriguez-Pardo, Louis Daumas, Leonardo Chiani, Massimo Tavoni  

**Link**: [PDF](https://arxiv.org/pdf/2505.13264)  

**Abstract**: Climate-economic modeling under uncertainty presents significant computational challenges that may limit policymakers' ability to address climate change effectively. This paper explores neural network-based approaches for solving high-dimensional optimal control problems arising from models that incorporate ambiguity aversion in climate mitigation decisions. We develop a continuous-time endogenous-growth economic model that accounts for multiple mitigation pathways, including emission-free capital and carbon intensity reductions. Given the inherent complexity and high dimensionality of these models, traditional numerical methods become computationally intractable. We benchmark several neural network architectures against finite-difference generated solutions, evaluating their ability to capture the dynamic interactions between uncertainty, technology transitions, and optimal climate policy. Our findings demonstrate that appropriate neural architecture selection significantly impacts both solution accuracy and computational efficiency when modeling climate-economic systems under uncertainty. These methodological advances enable more sophisticated modeling of climate policy decisions, allowing for better representation of technology transitions and uncertainty-critical elements for developing effective mitigation strategies in the face of climate change. 

---
# WikiPersonas: What Can We Learn From Personalized Alignment to Famous People? 

**Authors**: Zilu Tang, Afra Feyza Akyürek, Ekin Akyürek, Derry Wijaya  

**Link**: [PDF](https://arxiv.org/pdf/2505.13257)  

**Abstract**: Preference alignment has become a standard pipeline in finetuning models to follow \emph{generic} human preferences. Majority of work seeks to optimize model to produce responses that would be preferable \emph{on average}, simplifying the diverse and often \emph{contradicting} space of human preferences. While research has increasingly focused on personalized alignment: adapting models to individual user preferences, there is a lack of personalized preference dataset which focus on nuanced individual-level preferences. To address this, we introduce WikiPersona: the first fine-grained personalization using well-documented, famous individuals. Our dataset challenges models to align with these personas through an interpretable process: generating verifiable textual descriptions of a persona's background and preferences in addition to alignment. We systematically evaluate different personalization approaches and find that as few-shot prompting with preferences and fine-tuning fail to simultaneously ensure effectiveness and efficiency, using \textit{inferred personal preferences} as prefixes enables effective personalization, especially in topics where preferences clash while leading to more equitable generalization across unseen personas. 

---
# Composing Dextrous Grasping and In-hand Manipulation via Scoring with a Reinforcement Learning Critic 

**Authors**: Lennart Röstel, Dominik Winkelbauer, Johannes Pitz, Leon Sievers, Berthold Bäuml  

**Link**: [PDF](https://arxiv.org/pdf/2505.13253)  

**Abstract**: In-hand manipulation and grasping are fundamental yet often separately addressed tasks in robotics. For deriving in-hand manipulation policies, reinforcement learning has recently shown great success. However, the derived controllers are not yet useful in real-world scenarios because they often require a human operator to place the objects in suitable initial (grasping) states. Finding stable grasps that also promote the desired in-hand manipulation goal is an open problem. In this work, we propose a method for bridging this gap by leveraging the critic network of a reinforcement learning agent trained for in-hand manipulation to score and select initial grasps. Our experiments show that this method significantly increases the success rate of in-hand manipulation without requiring additional training. We also present an implementation of a full grasp manipulation pipeline on a real-world system, enabling autonomous grasping and reorientation even of unwieldy objects. 

---
# MAGI-1: Autoregressive Video Generation at Scale 

**Authors**: Sand.ai, Hansi Teng, Hongyu Jia, Lei Sun, Lingzhi Li, Maolin Li, Mingqiu Tang, Shuai Han, Tianning Zhang, W.Q. Zhang, Weifeng Luo, Xiaoyang Kang, Yuchen Sun, Yue Cao, Yunpeng Huang, Yutong Lin, Yuxin Fang, Zewei Tao, Zheng Zhang, Zhongshu Wang, Zixun Liu, Dai Shi, Guoli Su, Hanwen Sun, Hong Pan, Jie Wang, Jiexin Sheng, Min Cui, Min Hu, Ming Yan, Shucheng Yin, Siran Zhang, Tingting Liu, Xianping Yin, Xiaoyu Yang, Xin Song, Xuan Hu, Yankai Zhang, Yuqiao Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.13211)  

**Abstract**: We present MAGI-1, a world model that generates videos by autoregressively predicting a sequence of video chunks, defined as fixed-length segments of consecutive frames. Trained to denoise per-chunk noise that increases monotonically over time, MAGI-1 enables causal temporal modeling and naturally supports streaming generation. It achieves strong performance on image-to-video (I2V) tasks conditioned on text instructions, providing high temporal consistency and scalability, which are made possible by several algorithmic innovations and a dedicated infrastructure stack. MAGI-1 facilitates controllable generation via chunk-wise prompting and supports real-time, memory-efficient deployment by maintaining constant peak inference cost, regardless of video length. The largest variant of MAGI-1 comprises 24 billion parameters and supports context lengths of up to 4 million tokens, demonstrating the scalability and robustness of our approach. The code and models are available at this https URL and this https URL. The product can be accessed at this https URL. 

---
# Picturized and Recited with Dialects: A Multimodal Chinese Representation Framework for Sentiment Analysis of Classical Chinese Poetry 

**Authors**: Xiaocong Du, Haoyu Pei, Haipeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13210)  

**Abstract**: Classical Chinese poetry is a vital and enduring part of Chinese literature, conveying profound emotional resonance. Existing studies analyze sentiment based on textual meanings, overlooking the unique rhythmic and visual features inherent in poetry,especially since it is often recited and accompanied by Chinese paintings. In this work, we propose a dialect-enhanced multimodal framework for classical Chinese poetry sentiment analysis. We extract sentence-level audio features from the poetry and incorporate audio from multiple dialects,which may retain regional ancient Chinese phonetic features, enriching the phonetic representation. Additionally, we generate sentence-level visual features, and the multimodal features are fused with textual features enhanced by LLM translation through multimodal contrastive representation learning. Our framework outperforms state-of-the-art methods on two public datasets, achieving at least 2.51% improvement in accuracy and 1.63% in macro F1. We open-source the code to facilitate research in this area and provide insights for general multimodal Chinese representation. 

---
# Efficient Generation of Parameterised Quantum Circuits from Large Texts 

**Authors**: Colin Krawchuk, Nikhil Khatri, Neil John Ortega, Dimitri Kartsaklis  

**Link**: [PDF](https://arxiv.org/pdf/2505.13208)  

**Abstract**: Quantum approaches to natural language processing (NLP) are redefining how linguistic information is represented and processed. While traditional hybrid quantum-classical models rely heavily on classical neural networks, recent advancements propose a novel framework, DisCoCirc, capable of directly encoding entire documents as parameterised quantum circuits (PQCs), besides enjoying some additional interpretability and compositionality benefits. Following these ideas, this paper introduces an efficient methodology for converting large-scale texts into quantum circuits using tree-like representations of pregroup diagrams. Exploiting the compositional parallels between language and quantum mechanics, grounded in symmetric monoidal categories, our approach enables faithful and efficient encoding of syntactic and discourse relationships in long and complex texts (up to 6410 words in our experiments) to quantum circuits. The developed system is provided to the community as part of the augmented open-source quantum NLP package lambeq Gen II. 

---
# MatPredict: a dataset and benchmark for learning material properties of diverse indoor objects 

**Authors**: Yuzhen Chen, Hojun Son, Arpan Kusari  

**Link**: [PDF](https://arxiv.org/pdf/2505.13201)  

**Abstract**: Determining material properties from camera images can expand the ability to identify complex objects in indoor environments, which is valuable for consumer robotics applications. To support this, we introduce MatPredict, a dataset that combines the high-quality synthetic objects from Replica dataset with MatSynth dataset's material properties classes - to create objects with diverse material properties. We select 3D meshes of specific foreground objects and render them with different material properties. In total, we generate \textbf{18} commonly occurring objects with \textbf{14} different materials. We showcase how we provide variability in terms of lighting and camera placement for these objects. Next, we provide a benchmark for inferring material properties from visual images using these perturbed models in the scene, discussing the specific neural network models involved and their performance based on different image comparison metrics. By accurately simulating light interactions with different materials, we can enhance realism, which is crucial for training models effectively through large-scale simulations. This research aims to revolutionize perception in consumer robotics. The dataset is provided \href{this https URL}{here} and the code is provided \href{this https URL}{here}. 

---
# A Physics-Inspired Optimizer: Velocity Regularized Adam 

**Authors**: Pranav Vaidhyanathan, Lucas Schorling, Natalia Ares, Michael A. Osborne  

**Link**: [PDF](https://arxiv.org/pdf/2505.13196)  

**Abstract**: We introduce Velocity-Regularized Adam (VRAdam), a physics-inspired optimizer for training deep neural networks that draws on ideas from quartic terms for kinetic energy with its stabilizing effects on various system dynamics. Previous algorithms, including the ubiquitous Adam, operate at the so called adaptive edge of stability regime during training leading to rapid oscillations and slowed convergence of loss. However, VRAdam adds a higher order penalty on the learning rate based on the velocity such that the algorithm automatically slows down whenever weight updates become large. In practice, we observe that the effective dynamic learning rate shrinks in high-velocity regimes, damping oscillations and allowing for a more aggressive base step size when necessary without divergence. By combining this velocity-based regularizer for global damping with per-parameter scaling of Adam to create a hybrid optimizer, we demonstrate that VRAdam consistently exceeds the performance against standard optimizers including AdamW. We benchmark various tasks such as image classification, language modeling, image generation and generative modeling using diverse architectures and training methodologies including Convolutional Neural Networks (CNNs), Transformers, and GFlowNets. 

---
# True Zero-Shot Inference of Dynamical Systems Preserving Long-Term Statistics 

**Authors**: Christoph Jürgen Hemmer, Daniel Durstewitz  

**Link**: [PDF](https://arxiv.org/pdf/2505.13192)  

**Abstract**: Complex, temporally evolving phenomena, from climate to brain activity, are governed by dynamical systems (DS). DS reconstruction (DSR) seeks to infer generative surrogate models of these from observed data, reproducing their long-term behavior. Existing DSR approaches require purpose-training for any new system observed, lacking the zero-shot and in-context inference capabilities known from LLMs. Here we introduce DynaMix, a novel multivariate ALRNN-based mixture-of-experts architecture pre-trained for DSR, the first DSR model able to generalize zero-shot to out-of-domain DS. Just from a provided context signal, without any re-training, DynaMix faithfully forecasts the long-term evolution of novel DS where existing time series (TS) foundation models, like Chronos, fail -- at a fraction of the number of parameters and orders of magnitude faster inference times. DynaMix outperforms TS foundation models in terms of long-term statistics, and often also short-term forecasts, even on real-world time series, like traffic or weather data, typically used for training and evaluating TS models, but not at all part of DynaMix' training corpus. We illustrate some of the failure modes of TS models for DSR problems, and conclude that models built on DS principles may bear a huge potential also for advancing the TS prediction field. 

---
# Emergence of Fixational and Saccadic Movements in a Multi-Level Recurrent Attention Model for Vision 

**Authors**: Pengcheng Pan, Yonekura Shogo, Yasuo Kuniyoshi  

**Link**: [PDF](https://arxiv.org/pdf/2505.13191)  

**Abstract**: Inspired by foveal vision, hard attention models promise interpretability and parameter economy. However, existing models like the Recurrent Model of Visual Attention (RAM) and Deep Recurrent Attention Model (DRAM) failed to model the hierarchy of human vision system, that compromise on the visual exploration dynamics. As a result, they tend to produce attention that are either overly fixational or excessively saccadic, diverging from human eye movement behavior. In this paper, we propose a Multi-Level Recurrent Attention Model (MRAM), a novel hard attention framework that explicitly models the neural hierarchy of human visual processing. By decoupling the function of glimpse location generation and task execution in two recurrent layers, MRAM emergent a balanced behavior between fixation and saccadic movement. Our results show that MRAM not only achieves more human-like attention dynamics, but also consistently outperforms CNN, RAM and DRAM baselines on standard image classification benchmarks. 

---
# When a Reinforcement Learning Agent Encounters Unknown Unknowns 

**Authors**: Juntian Zhu, Miguel de Carvalho, Zhouwang Yang, Fengxiang He  

**Link**: [PDF](https://arxiv.org/pdf/2505.13188)  

**Abstract**: An AI agent might surprisingly find she has reached an unknown state which she has never been aware of -- an unknown unknown. We mathematically ground this scenario in reinforcement learning: an agent, after taking an action calculated from value functions $Q$ and $V$ defined on the {\it {aware domain}}, reaches a state out of the domain. To enable the agent to handle this scenario, we propose an {\it episodic Markov decision {process} with growing awareness} (EMDP-GA) model, taking a new {\it noninformative value expansion} (NIVE) approach to expand value functions to newly aware areas: when an agent arrives at an unknown unknown, value functions $Q$ and $V$ whereon are initialised by noninformative beliefs -- the averaged values on the aware domain. This design is out of respect for the complete absence of knowledge in the newly discovered state. The upper confidence bound momentum Q-learning is then adapted to the growing awareness for training the EMDP-GA model. We prove that (1) the regret of our approach is asymptotically consistent with the state of the art (SOTA) without exposure to unknown unknowns in an extremely uncertain environment, and (2) our computational complexity and space complexity are comparable with the SOTA -- these collectively suggest that though an unknown unknown is surprising, it will be asymptotically properly discovered with decent speed and an affordable cost. 

---
# Information Science Principles of Machine Learning: A Causal Chain Meta-Framework Based on Formalized Information Mapping 

**Authors**: Jianfeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13182)  

**Abstract**: [Objective] This study focuses on addressing the current lack of a unified formal theoretical framework in machine learning, as well as the deficiencies in interpretability and ethical safety assurance. [Methods] A formal information model is first constructed, utilizing sets of well-formed formulas to explicitly define the ontological states and carrier mappings of typical components in machine learning. Learnable and processable predicates, along with learning and processing functions, are introduced to analyze the logical deduction and constraint rules of the causal chains within models. [Results] A meta-framework for machine learning theory (MLT-MF) is established. Based on this framework, universal definitions for model interpretability and ethical safety are proposed. Furthermore, three key theorems are proved: the equivalence of model interpretability and information recoverability, the assurance of ethical safety, and the estimation of generalization error. [Limitations] The current framework assumes ideal conditions with noiseless information-enabling mappings and primarily targets model learning and processing logic in static scenarios. It does not yet address information fusion and conflict resolution across ontological spaces in multimodal or multi-agent systems. [Conclusions] This work overcomes the limitations of fragmented research and provides a unified theoretical foundation for systematically addressing the critical challenges currently faced in machine learning. 

---
# ToolSpectrum : Towards Personalized Tool Utilization for Large Language Models 

**Authors**: Zihao Cheng, Hongru Wang, Zeming Liu, Yuhang Guo, Yuanfang Guo, Yunhong Wang, Haifeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13176)  

**Abstract**: While integrating external tools into large language models (LLMs) enhances their ability to access real-time information and domain-specific services, existing approaches focus narrowly on functional tool selection following user instructions, overlooking the context-aware personalization in tool selection. This oversight leads to suboptimal user satisfaction and inefficient tool utilization, particularly when overlapping toolsets require nuanced selection based on contextual factors. To bridge this gap, we introduce ToolSpectrum, a benchmark designed to evaluate LLMs' capabilities in personalized tool utilization. Specifically, we formalize two key dimensions of personalization, user profile and environmental factors, and analyze their individual and synergistic impacts on tool utilization. Through extensive experiments on ToolSpectrum, we demonstrate that personalized tool utilization significantly improves user experience across diverse scenarios. However, even state-of-the-art LLMs exhibit the limited ability to reason jointly about user profiles and environmental factors, often prioritizing one dimension at the expense of the other. Our findings underscore the necessity of context-aware personalization in tool-augmented LLMs and reveal critical limitations for current models. Our data and code are available at this https URL. 

---
# Role-Playing Evaluation for Large Language Models 

**Authors**: Yassine El Boudouri, Walter Nuninger, Julian Alvarez, Yvan Peter  

**Link**: [PDF](https://arxiv.org/pdf/2505.13157)  

**Abstract**: Large Language Models (LLMs) demonstrate a notable capacity for adopting personas and engaging in role-playing. However, evaluating this ability presents significant challenges, as human assessments are resource-intensive and automated evaluations can be biased. To address this, we introduce Role-Playing Eval (RPEval), a novel benchmark designed to assess LLM role-playing capabilities across four key dimensions: emotional understanding, decision-making, moral alignment, and in-character consistency. This article details the construction of RPEval and presents baseline evaluations. Our code and dataset are available at this https URL 

---
# Tianyi: A Traditional Chinese Medicine all-rounder language model and its Real-World Clinical Practice 

**Authors**: Zhi Liu, Tao Yang, Jing Wang, Yexin Chen, Zhan Gao, Jiaxi Yang, Kui Chen, Bingji Lu, Xiaochen Li, Changyong Luo, Yan Li, Xiaohong Gu, Peng Cao  

**Link**: [PDF](https://arxiv.org/pdf/2505.13156)  

**Abstract**: Natural medicines, particularly Traditional Chinese Medicine (TCM), are gaining global recognition for their therapeutic potential in addressing human symptoms and diseases. TCM, with its systematic theories and extensive practical experience, provides abundant resources for healthcare. However, the effective application of TCM requires precise syndrome diagnosis, determination of treatment principles, and prescription formulation, which demand decades of clinical expertise. Despite advancements in TCM-based decision systems, machine learning, and deep learning research, limitations in data and single-objective constraints hinder their practical application. In recent years, large language models (LLMs) have demonstrated potential in complex tasks, but lack specialization in TCM and face significant challenges, such as too big model scale to deploy and issues with hallucination. To address these challenges, we introduce Tianyi with 7.6-billion-parameter LLM, a model scale proper and specifically designed for TCM, pre-trained and fine-tuned on diverse TCM corpora, including classical texts, expert treatises, clinical records, and knowledge graphs. Tianyi is designed to assimilate interconnected and systematic TCM knowledge through a progressive learning manner. Additionally, we establish TCMEval, a comprehensive evaluation benchmark, to assess LLMs in TCM examinations, clinical tasks, domain-specific question-answering, and real-world trials. The extensive evaluations demonstrate the significant potential of Tianyi as an AI assistant in TCM clinical practice and research, bridging the gap between TCM knowledge and practical application. 

---
# Temporal Distance-aware Transition Augmentation for Offline Model-based Reinforcement Learning 

**Authors**: Dongsu Lee, Minhae Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2505.13144)  

**Abstract**: The goal of offline reinforcement learning (RL) is to extract a high-performance policy from the fixed datasets, minimizing performance degradation due to out-of-distribution (OOD) samples. Offline model-based RL (MBRL) is a promising approach that ameliorates OOD issues by enriching state-action transitions with augmentations synthesized via a learned dynamics model. Unfortunately, seminal offline MBRL methods often struggle in sparse-reward, long-horizon tasks. In this work, we introduce a novel MBRL framework, dubbed Temporal Distance-Aware Transition Augmentation (TempDATA), that generates augmented transitions in a temporally structured latent space rather than in raw state space. To model long-horizon behavior, TempDATA learns a latent abstraction that captures a temporal distance from both trajectory and transition levels of state space. Our experiments confirm that TempDATA outperforms previous offline MBRL methods and achieves matching or surpassing the performance of diffusion-based trajectory augmentation and goal-conditioned RL on the D4RL AntMaze, FrankaKitchen, CALVIN, and pixel-based FrankaKitchen. 

---
# ModernGBERT: German-only 1B Encoder Model Trained from Scratch 

**Authors**: Anton Ehrmanntraut, Julia Wunderle, Jan Pfister, Fotis Jannidis, Andreas Hotho  

**Link**: [PDF](https://arxiv.org/pdf/2505.13136)  

**Abstract**: Despite the prominence of decoder-only language models, encoders remain crucial for resource-constrained applications. We introduce ModernGBERT (134M, 1B), a fully transparent family of German encoder models trained from scratch, incorporating architectural innovations from ModernBERT. To evaluate the practical trade-offs of training encoders from scratch, we also present LLäMmlein2Vec (120M, 1B, 7B), a family of encoders derived from German decoder-only models via LLM2Vec. We benchmark all models on natural language understanding, text embedding, and long-context reasoning tasks, enabling a controlled comparison between dedicated encoders and converted decoders. Our results show that ModernGBERT 1B outperforms prior state-of-the-art German encoders as well as encoders adapted via LLM2Vec, with regard to performance and parameter-efficiency. All models, training data, checkpoints and code are publicly available, advancing the German NLP ecosystem with transparent, high-performance encoder models. 

---
# Adaptive Image Restoration for Video Surveillance: A Real-Time Approach 

**Authors**: Muhammad Awais Amin, Adama Ilboudo, Abdul Samad bin Shahid, Amjad Ali, Waqas Haider Khan Bangyal  

**Link**: [PDF](https://arxiv.org/pdf/2505.13130)  

**Abstract**: One of the major challenges in the field of computer vision especially for detection, segmentation, recognition, monitoring, and automated solutions, is the quality of images. Image degradation, often caused by factors such as rain, fog, lighting, etc., has a negative impact on automated this http URL, several image restoration solutions exist, including restoration models for single degradation and restoration models for multiple degradations. However, these solutions are not suitable for real-time processing. In this study, the aim was to develop a real-time image restoration solution for video surveillance. To achieve this, using transfer learning with ResNet_50, we developed a model for automatically identifying the types of degradation present in an image to reference the necessary treatment(s) for image restoration. Our solution has the advantage of being flexible and scalable. 

---
# $μ$PC: Scaling Predictive Coding to 100+ Layer Networks 

**Authors**: Francesco Innocenti, El Mehdi Achour, Christopher L. Buckley  

**Link**: [PDF](https://arxiv.org/pdf/2505.13124)  

**Abstract**: The biological implausibility of backpropagation (BP) has motivated many alternative, brain-inspired algorithms that attempt to rely only on local information, such as predictive coding (PC) and equilibrium propagation. However, these algorithms have notoriously struggled to train very deep networks, preventing them from competing with BP in large-scale settings. Indeed, scaling PC networks (PCNs) has recently been posed as a challenge for the community (Pinchetti et al., 2024). Here, we show that 100+ layer PCNs can be trained reliably using a Depth-$\mu$P parameterisation (Yang et al., 2023; Bordelon et al., 2023) which we call "$\mu$PC". Through an extensive analysis of the scaling behaviour of PCNs, we reveal several pathologies that make standard PCNs difficult to train at large depths. We then show that, despite addressing only some of these instabilities, $\mu$PC allows stable training of very deep (up to 128-layer) residual networks on simple classification tasks with competitive performance and little tuning compared to current benchmarks. Moreover, $\mu$PC enables zero-shot transfer of both weight and activity learning rates across widths and depths. Our results have implications for other local algorithms and could be extended to convolutional and transformer architectures. Code for $\mu$PC is made available as part of a JAX library for PCNs at this https URL (Innocenti et al., 2024). 

---
# Just Dance with $π$! A Poly-modal Inductor for Weakly-supervised Video Anomaly Detection 

**Authors**: Snehashis Majhi, Giacomo D'Amicantonio, Antitza Dantcheva, Quan Kong, Lorenzo Garattoni, Gianpiero Francesca, Egor Bondarev, Francois Bremond  

**Link**: [PDF](https://arxiv.org/pdf/2505.13123)  

**Abstract**: Weakly-supervised methods for video anomaly detection (VAD) are conventionally based merely on RGB spatio-temporal features, which continues to limit their reliability in real-world scenarios. This is due to the fact that RGB-features are not sufficiently distinctive in setting apart categories such as shoplifting from visually similar events. Therefore, towards robust complex real-world VAD, it is essential to augment RGB spatio-temporal features by additional modalities. Motivated by this, we introduce the Poly-modal Induced framework for VAD: "PI-VAD", a novel approach that augments RGB representations by five additional modalities. Specifically, the modalities include sensitivity to fine-grained motion (Pose), three dimensional scene and entity representation (Depth), surrounding objects (Panoptic masks), global motion (optical flow), as well as language cues (VLM). Each modality represents an axis of a polygon, streamlined to add salient cues to RGB. PI-VAD includes two plug-in modules, namely Pseudo-modality Generation module and Cross Modal Induction module, which generate modality-specific prototypical representation and, thereby, induce multi-modal information into RGB cues. These modules operate by performing anomaly-aware auxiliary tasks and necessitate five modality backbones -- only during training. Notably, PI-VAD achieves state-of-the-art accuracy on three prominent VAD datasets encompassing real-world scenarios, without requiring the computational overhead of five modality backbones at inference. 

---
# When majority rules, minority loses: bias amplification of gradient descent 

**Authors**: François Bachoc, Jérôme Bolte, Ryan Boustany, Jean-Michel Loubes  

**Link**: [PDF](https://arxiv.org/pdf/2505.13122)  

**Abstract**: Despite growing empirical evidence of bias amplification in machine learning, its theoretical foundations remain poorly understood. We develop a formal framework for majority-minority learning tasks, showing how standard training can favor majority groups and produce stereotypical predictors that neglect minority-specific features. Assuming population and variance imbalance, our analysis reveals three key findings: (i) the close proximity between ``full-data'' and stereotypical predictors, (ii) the dominance of a region where training the entire model tends to merely learn the majority traits, and (iii) a lower bound on the additional training required. Our results are illustrated through experiments in deep learning for tabular and image classification tasks. 

---
# Continuous Fair SMOTE -- Fairness-Aware Stream Learning from Imbalanced Data 

**Authors**: Kathrin Lammers, Valerie Vaquet, Barbara Hammer  

**Link**: [PDF](https://arxiv.org/pdf/2505.13116)  

**Abstract**: As machine learning is increasingly applied in an online fashion to deal with evolving data streams, the fairness of these algorithms is a matter of growing ethical and legal concern. In many use cases, class imbalance in the data also needs to be dealt with to ensure predictive performance. Current fairness-aware stream learners typically attempt to solve these issues through in- or post-processing by focusing on optimizing one specific discrimination metric, addressing class imbalance in a separate processing step. While C-SMOTE is a highly effective model-agnostic pre-processing approach to mitigate class imbalance, as a side effect of this method, algorithmic bias is often introduced.
Therefore, we propose CFSMOTE - a fairness-aware, continuous SMOTE variant - as a pre-processing approach to simultaneously address the class imbalance and fairness concerns by employing situation testing and balancing fairness-relevant groups during oversampling. Unlike other fairness-aware stream learners, CFSMOTE is not optimizing for only one specific fairness metric, therefore avoiding potentially problematic trade-offs. Our experiments show significant improvement on several common group fairness metrics in comparison to vanilla C-SMOTE while maintaining competitive performance, also in comparison to other fairness-aware algorithms. 

---
# Benchmarking and Confidence Evaluation of LALMs For Temporal Reasoning 

**Authors**: Debarpan Bhattacharya, Apoorva Kulkarni, Sriram Ganapathy  

**Link**: [PDF](https://arxiv.org/pdf/2505.13115)  

**Abstract**: The popular success of text-based large language models (LLM) has streamlined the attention of the multimodal community to combine other modalities like vision and audio along with text to achieve similar multimodal capabilities. In this quest, large audio language models (LALMs) have to be evaluated on reasoning related tasks which are different from traditional classification or generation tasks. Towards this goal, we propose a novel dataset called temporal reasoning evaluation of audio (TREA).
We benchmark open-source LALMs and observe that they are consistently behind human capabilities on the tasks in the TREA dataset. While evaluating LALMs, we also propose an uncertainty metric, which computes the invariance of the model to semantically identical perturbations of the input. Our analysis shows that the accuracy and uncertainty metrics are not necessarily correlated and thus, points to a need for wholesome evaluation of LALMs for high-stakes applications. 

---
# FreeKV: Boosting KV Cache Retrieval for Efficient LLM Inference 

**Authors**: Guangda Liu, Chengwei Li, Zhenyu Ning, Jing Lin, Yiwu Yao, Danning Ke, Minyi Guo, Jieru Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.13109)  

**Abstract**: Large language models (LLMs) have been widely deployed with rapidly expanding context windows to support increasingly demanding applications. However, long contexts pose significant deployment challenges, primarily due to the KV cache whose size grows proportionally with context length. While KV cache compression methods are proposed to address this issue, KV dropping methods incur considerable accuracy loss, and KV retrieval methods suffer from significant efficiency bottlenecks. We propose FreeKV, an algorithm-system co-optimization framework to enhance KV retrieval efficiency while preserving accuracy. On the algorithm side, FreeKV introduces speculative retrieval to shift the KV selection and recall processes out of the critical path, combined with fine-grained correction to ensure accuracy. On the system side, FreeKV employs hybrid KV layouts across CPU and GPU memory to eliminate fragmented data transfers, and leverages double-buffered streamed recall to further improve efficiency. Experiments demonstrate that FreeKV achieves near-lossless accuracy across various scenarios and models, delivering up to 13$\times$ speedup compared to SOTA KV retrieval methods. 

---
# Lightweight Transformer via Unrolling of Mixed Graph Algorithms for Traffic Forecast 

**Authors**: Ji Qi, Tam Thuc Do, Mingxiao Liu, Zhuoshi Pan, Yuzhe Li, Gene Cheung, H. Vicky Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.13102)  

**Abstract**: To forecast traffic with both spatial and temporal dimensions, we unroll a mixed-graph-based optimization algorithm into a lightweight and interpretable transformer-like neural net. Specifically, we construct two graphs: an undirected graph $\mathcal{G}^u$ capturing spatial correlations across geography, and a directed graph $\mathcal{G}^d$ capturing sequential relationships over time. We formulate a prediction problem for the future samples of signal $\mathbf{x}$, assuming it is "smooth" with respect to both $\mathcal{G}^u$ and $\mathcal{G}^d$, where we design new $\ell_2$ and $\ell_1$-norm variational terms to quantify and promote signal smoothness (low-frequency reconstruction) on a directed graph. We construct an iterative algorithm based on alternating direction method of multipliers (ADMM), and unroll it into a feed-forward network for data-driven parameter learning. We insert graph learning modules for $\mathcal{G}^u$ and $\mathcal{G}^d$, which are akin to the self-attention mechanism in classical transformers. Experiments show that our unrolled networks achieve competitive traffic forecast performance as state-of-the-art prediction schemes, while reducing parameter counts drastically. Our code is available in this https URL. 

---
# ARIW-Framework: Adaptive Robust Iterative Watermarking Framework 

**Authors**: Shaowu Wu, Liting Zeng, Wei Lu, Xiangyang Luo  

**Link**: [PDF](https://arxiv.org/pdf/2505.13101)  

**Abstract**: With the rapid rise of large models, copyright protection for generated image content has become a critical security challenge. Although deep learning watermarking techniques offer an effective solution for digital image copyright protection, they still face limitations in terms of visual quality, robustness and generalization. To address these issues, this paper proposes an adaptive robust iterative watermarking framework (ARIW-Framework) that achieves high-quality watermarked images while maintaining exceptional robustness and generalization performance. Specifically, we introduce an iterative approach to optimize the encoder for generating robust residuals. The encoder incorporates noise layers and a decoder to compute robustness weights for residuals under various noise attacks. By employing a parallel optimization strategy, the framework enhances robustness against multiple types of noise attacks. Furthermore, we leverage image gradients to determine the embedding strength at each pixel location, significantly improving the visual quality of the watermarked images. Extensive experiments demonstrate that the proposed method achieves superior visual quality while exhibiting remarkable robustness and generalization against noise attacks. 

---
# Time-Frequency-Based Attention Cache Memory Model for Real-Time Speech Separation 

**Authors**: Guo Chen, Kai Li, Runxuan Yang, Xiaolin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13094)  

**Abstract**: Existing causal speech separation models often underperform compared to non-causal models due to difficulties in retaining historical information. To address this, we propose the Time-Frequency Attention Cache Memory (TFACM) model, which effectively captures spatio-temporal relationships through an attention mechanism and cache memory (CM) for historical information storage. In TFACM, an LSTM layer captures frequency-relative positions, while causal modeling is applied to the time dimension using local and global representations. The CM module stores past information, and the causal attention refinement (CAR) module further enhances time-based feature representations for finer granularity. Experimental results showed that TFACM achieveed comparable performance to the SOTA TF-GridNet-Causal model, with significantly lower complexity and fewer trainable parameters. For more details, visit the project page: this https URL. 

---
# Graph Alignment for Benchmarking Graph Neural Networks and Learning Positional Encodings 

**Authors**: Adrien Lagesse, Marc Lelarge  

**Link**: [PDF](https://arxiv.org/pdf/2505.13087)  

**Abstract**: We propose a novel benchmarking methodology for graph neural networks (GNNs) based on the graph alignment problem, a combinatorial optimization task that generalizes graph isomorphism by aligning two unlabeled graphs to maximize overlapping edges. We frame this problem as a self-supervised learning task and present several methods to generate graph alignment datasets using synthetic random graphs and real-world graph datasets from multiple domains. For a given graph dataset, we generate a family of graph alignment datasets with increasing difficulty, allowing us to rank the performance of various architectures. Our experiments indicate that anisotropic graph neural networks outperform standard convolutional architectures. To further demonstrate the utility of the graph alignment task, we show its effectiveness for unsupervised GNN pre-training, where the learned node embeddings outperform other positional encodings on three molecular regression tasks and achieve state-of-the-art results on the PCQM4Mv2 dataset with significantly fewer parameters. To support reproducibility and further research, we provide an open-source Python package to generate graph alignment datasets and benchmark new GNN architectures. 

---
# MultiActor-Audiobook: Zero-Shot Audiobook Generation with Faces and Voices of Multiple Speakers 

**Authors**: Kyeongman Park, Seongho Joo, Kyomin Jung  

**Link**: [PDF](https://arxiv.org/pdf/2505.13082)  

**Abstract**: We introduce MultiActor-Audiobook, a zero-shot approach for generating audiobooks that automatically produces consistent, expressive, and speaker-appropriate prosody, including intonation and emotion. Previous audiobook systems have several limitations: they require users to manually configure the speaker's prosody, read each sentence with a monotonic tone compared to voice actors, or rely on costly training. However, our MultiActor-Audiobook addresses these issues by introducing two novel processes: (1) MSP (**Multimodal Speaker Persona Generation**) and (2) LSI (**LLM-based Script Instruction Generation**). With these two processes, MultiActor-Audiobook can generate more emotionally expressive audiobooks with a consistent speaker prosody without additional training. We compare our system with commercial products, through human and MLLM evaluations, achieving competitive results. Furthermore, we demonstrate the effectiveness of MSP and LSI through ablation studies. 

---
# Cross-modal Knowledge Transfer Learning as Graph Matching Based on Optimal Transport for ASR 

**Authors**: Xugang Lu, Peng Shen, Yu Tsao, Hisashi Kawai  

**Link**: [PDF](https://arxiv.org/pdf/2505.13079)  

**Abstract**: Transferring linguistic knowledge from a pretrained language model (PLM) to acoustic feature learning has proven effective in enhancing end-to-end automatic speech recognition (E2E-ASR). However, aligning representations between linguistic and acoustic modalities remains a challenge due to inherent modality gaps. Optimal transport (OT) has shown promise in mitigating these gaps by minimizing the Wasserstein distance (WD) between linguistic and acoustic feature distributions. However, previous OT-based methods overlook structural relationships, treating feature vectors as unordered sets. To address this, we propose Graph Matching Optimal Transport (GM-OT), which models linguistic and acoustic sequences as structured graphs. Nodes represent feature embeddings, while edges capture temporal and sequential relationships. GM-OT minimizes both WD (between nodes) and Gromov-Wasserstein distance (GWD) (between edges), leading to a fused Gromov-Wasserstein distance (FGWD) formulation. This enables structured alignment and more efficient knowledge transfer compared to existing OT-based approaches. Theoretical analysis further shows that prior OT-based methods in linguistic knowledge transfer can be viewed as a special case within our GM-OT framework. We evaluate GM-OT on Mandarin ASR using a CTC-based E2E-ASR system with a PLM for knowledge transfer. Experimental results demonstrate significant performance gains over state-of-the-art models, validating the effectiveness of our approach. 

---
# Advancing Sequential Numerical Prediction in Autoregressive Models 

**Authors**: Xiang Fei, Jinghui Lu, Qi Sun, Hao Feng, Yanjie Wang, Wei Shi, An-Lan Wang, Jingqun Tang, Can Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13077)  

**Abstract**: Autoregressive models have become the de facto choice for sequence generation tasks, but standard approaches treat digits as independent tokens and apply cross-entropy loss, overlooking the coherent structure of numerical sequences. This paper introduces Numerical Token Integrity Loss (NTIL) to address this gap. NTIL operates at two levels: (1) token-level, where it extends the Earth Mover's Distance (EMD) to preserve ordinal relationships between numerical values, and (2) sequence-level, where it penalizes the overall discrepancy between the predicted and actual sequences. This dual approach improves numerical prediction and integrates effectively with LLMs/MLLMs. Extensive experiments show significant performance improvements with NTIL. 

---
# The Hidden Dangers of Browsing AI Agents 

**Authors**: Mykyta Mudryi, Markiyan Chaklosh, Grzegorz Wójcik  

**Link**: [PDF](https://arxiv.org/pdf/2505.13076)  

**Abstract**: Autonomous browsing agents powered by large language models (LLMs) are increasingly used to automate web-based tasks. However, their reliance on dynamic content, tool execution, and user-provided data exposes them to a broad attack surface. This paper presents a comprehensive security evaluation of such agents, focusing on systemic vulnerabilities across multiple architectural layers. Our work outlines the first end-to-end threat model for browsing agents and provides actionable guidance for securing their deployment in real-world environments. To address discovered threats, we propose a defense in depth strategy incorporating input sanitization, planner executor isolation, formal analyzers, and session safeguards. These measures protect against both initial access and post exploitation attack vectors. Through a white box analysis of a popular open source project, Browser Use, we demonstrate how untrusted web content can hijack agent behavior and lead to critical security breaches. Our findings include prompt injection, domain validation bypass, and credential exfiltration, evidenced by a disclosed CVE and a working proof of concept exploit. 

---
# Structure-Aware Corpus Construction and User-Perception-Aligned Metrics for Large-Language-Model Code Completion 

**Authors**: Dengfeng Liu, Jucai Zhai, Xiaoguang Jiang, Ziqun Li, Qianjin Yu, Feng Liu, Rui Ye, Huang Liu, Zhiguo Yang, Yongsheng Du, Fang Tan  

**Link**: [PDF](https://arxiv.org/pdf/2505.13073)  

**Abstract**: Code completion technology based on large language model has significantly improved the development efficiency of programmers. However, in practical applications, there remains a gap between current commonly used code completion evaluation metrics and users' actual perception. To address this issue, we propose two evaluation metrics for code completion tasks--LCP and ROUGE-LCP, from the perspective of probabilistic modeling. Furthermore, to tackle the lack of effective structural semantic modeling and cross-module dependency information in LLMs for repository-level code completion scenarios, we propose a data processing method based on a Structure-Preserving and Semantically-Reordered Code Graph (SPSR-Graph). Through theoretical analysis and experimental validation, we demonstrate the superiority of the proposed evaluation metrics in terms of user perception consistency, as well as the effectiveness of the data processing method in enhancing model performance. 

---
# SNAPE-PM: Building and Utilizing Dynamic Partner Models for Adaptive Explanation Generation 

**Authors**: Amelie S. Robrecht, Christoph R. Kowalski, Stefan Kopp  

**Link**: [PDF](https://arxiv.org/pdf/2505.13053)  

**Abstract**: Adapting to the addressee is crucial for successful explanations, yet poses significant challenges for dialogsystems. We adopt the approach of treating explanation generation as a non-stationary decision process, where the optimal strategy varies according to changing beliefs about the explainee and the interaction context. In this paper we address the questions of (1) how to track the interaction context and the relevant listener features in a formally defined computational partner model, and (2) how to utilize this model in the dynamically adjusted, rational decision process that determines the currently best explanation strategy. We propose a Bayesian inference-based approach to continuously update the partner model based on user feedback, and a non-stationary Markov Decision Process to adjust decision-making based on the partner model values. We evaluate an implementation of this framework with five simulated interlocutors, demonstrating its effectiveness in adapting to different partners with constant and even changing feedback behavior. The results show high adaptivity with distinct explanation strategies emerging for different partners, highlighting the potential of our approach to improve explainable AI systems and dialogsystems in general. 

---
# A Generalized Label Shift Perspective for Cross-Domain Gaze Estimation 

**Authors**: Hao-Ran Yang, Xiaohui Chen, Chuan-Xian Ren  

**Link**: [PDF](https://arxiv.org/pdf/2505.13043)  

**Abstract**: Aiming to generalize the well-trained gaze estimation model to new target domains, Cross-domain Gaze Estimation (CDGE) is developed for real-world application scenarios. Existing CDGE methods typically extract the domain-invariant features to mitigate domain shift in feature space, which is proved insufficient by Generalized Label Shift (GLS) theory. In this paper, we introduce a novel GLS perspective to CDGE and modelize the cross-domain problem by label and conditional shift problem. A GLS correction framework is presented and a feasible realization is proposed, in which a importance reweighting strategy based on truncated Gaussian distribution is introduced to overcome the continuity challenges in label shift correction. To embed the reweighted source distribution to conditional invariant learning, we further derive a probability-aware estimation of conditional operator discrepancy. Extensive experiments on standard CDGE tasks with different backbone models validate the superior generalization capability across domain and applicability on various models of proposed method. 

---
# KIT's Offline Speech Translation and Instruction Following Submission for IWSLT 2025 

**Authors**: Sai Koneru, Maike Züfle, Thai-Binh Nguyen, Seymanur Akti, Jan Niehues, Alexander Waibel  

**Link**: [PDF](https://arxiv.org/pdf/2505.13036)  

**Abstract**: The scope of the International Workshop on Spoken Language Translation (IWSLT) has recently broadened beyond traditional Speech Translation (ST) to encompass a wider array of tasks, including Speech Question Answering and Summarization. This shift is partly driven by the growing capabilities of modern systems, particularly with the success of Large Language Models (LLMs). In this paper, we present the Karlsruhe Institute of Technology's submissions for the Offline ST and Instruction Following (IF) tracks, where we leverage LLMs to enhance performance across all tasks. For the Offline ST track, we propose a pipeline that employs multiple automatic speech recognition systems, whose outputs are fused using an LLM with document-level context. This is followed by a two-step translation process, incorporating additional refinement step to improve translation quality. For the IF track, we develop an end-to-end model that integrates a speech encoder with an LLM to perform a wide range of instruction-following tasks. We complement it with a final document-level refinement stage to further enhance output quality by using contextual information. 

---
# TSPulse: Dual Space Tiny Pre-Trained Models for Rapid Time-Series Analysis 

**Authors**: Vijay Ekambaram, Subodh Kumar, Arindam Jati, Sumanta Mukherjee, Tomoya Sakai, Pankaj Dayama, Wesley M. Gifford, Jayant Kalagnanam  

**Link**: [PDF](https://arxiv.org/pdf/2505.13033)  

**Abstract**: The rise of time-series pre-trained models has advanced temporal representation learning, but current state-of-the-art models are often large-scale, requiring substantial compute. We introduce TSPulse, ultra-compact time-series pre-trained models with only 1M parameters, specialized to perform strongly across classification, anomaly detection, imputation, and retrieval tasks. TSPulse introduces innovations at both the architecture and task levels. At the architecture level, it employs a dual-space masked reconstruction, learning from both time and frequency domains to capture complementary signals. This is further enhanced by a dual-embedding disentanglement, generating both detailed embeddings for fine-grained analysis and high-level semantic embeddings for broader task understanding. Notably, TSPulse's semantic embeddings are robust to shifts in time, magnitude, and noise, which is important for robust retrieval. At the task level, TSPulse incorporates TSLens, a fine-tuning component enabling task-specific feature attention. It also introduces a multi-head triangulation technique that correlates deviations from multiple prediction heads, enhancing anomaly detection by fusing complementary model outputs. Additionally, a hybrid mask pretraining is proposed to improves zero-shot imputation by reducing pre-training bias. These architecture and task innovations collectively contribute to TSPulse's significant performance gains: 5-16% on the UEA classification benchmarks, +20% on the TSB-AD anomaly detection leaderboard, +50% in zero-shot imputation, and +25% in time-series retrieval. Remarkably, these results are achieved with just 1M parameters, making TSPulse 10-100X smaller than existing pre-trained models. Its efficiency enables GPU-free inference and rapid pre-training, setting a new standard for efficient time-series pre-trained models. Models will be open-sourced soon. 

---
# Evaluatiing the efficacy of LLM Safety Solutions : The Palit Benchmark Dataset 

**Authors**: Sayon Palit, Daniel Woods  

**Link**: [PDF](https://arxiv.org/pdf/2505.13028)  

**Abstract**: Large Language Models (LLMs) are increasingly integrated into critical systems in industries like healthcare and finance. Users can often submit queries to LLM-enabled chatbots, some of which can enrich responses with information retrieved from internal databases storing sensitive data. This gives rise to a range of attacks in which a user submits a malicious query and the LLM-system outputs a response that creates harm to the owner, such as leaking internal data or creating legal liability by harming a third-party. While security tools are being developed to counter these threats, there is little formal evaluation of their effectiveness and usability. This study addresses this gap by conducting a thorough comparative analysis of LLM security tools. We identified 13 solutions (9 closed-source, 4 open-source), but only 7 were evaluated due to a lack of participation by proprietary model this http URL evaluate, we built a benchmark dataset of malicious prompts, and evaluate these tools performance against a baseline LLM model (ChatGPT-3.5-Turbo). Our results show that the baseline model has too many false positives to be used for this task. Lakera Guard and ProtectAI LLM Guard emerged as the best overall tools showcasing the tradeoff between usability and performance. The study concluded with recommendations for greater transparency among closed source providers, improved context-aware detections, enhanced open-source engagement, increased user awareness, and the adoption of more representative performance metrics. 

---
# Step-wise Adaptive Integration of Supervised Fine-tuning and Reinforcement Learning for Task-Specific LLMs 

**Authors**: Jack Chen, Fazhong Liu, Naruto Liu, Yuhan Luo, Erqu Qin, Harry Zheng, Tian Dong, Haojin Zhu, Yan Meng, Xiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13026)  

**Abstract**: Large language models (LLMs) excel at mathematical reasoning and logical problem-solving. The current popular training paradigms primarily use supervised fine-tuning (SFT) and reinforcement learning (RL) to enhance the models' reasoning abilities. However, when using SFT or RL alone, there are respective challenges: SFT may suffer from overfitting, while RL is prone to mode collapse. The state-of-the-art methods have proposed hybrid training schemes. However, static switching faces challenges such as poor generalization across different tasks and high dependence on data quality. In response to these challenges, inspired by the curriculum learning-quiz mechanism in human reasoning cultivation, We propose SASR, a step-wise adaptive hybrid training framework that theoretically unifies SFT and RL and dynamically balances the two throughout optimization. SASR uses SFT for initial warm-up to establish basic reasoning skills, and then uses an adaptive dynamic adjustment algorithm based on gradient norm and divergence relative to the original distribution to seamlessly integrate SFT with the online RL method GRPO. By monitoring the training status of LLMs and adjusting the training process in sequence, SASR ensures a smooth transition between training schemes, maintaining core reasoning abilities while exploring different paths. Experimental results demonstrate that SASR outperforms SFT, RL, and static hybrid training methods. 

---
# LiBOG: Lifelong Learning for Black-Box Optimizer Generation 

**Authors**: Jiyuan Pei, Yi Mei, Jialin Liu, Mengjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13025)  

**Abstract**: Meta-Black-Box Optimization (MetaBBO) garners attention due to its success in automating the configuration and generation of black-box optimizers, significantly reducing the human effort required for optimizer design and discovering optimizers with higher performance than classic human-designed optimizers. However, existing MetaBBO methods conduct one-off training under the assumption that a stationary problem distribution with extensive and representative training problem samples is pre-available. This assumption is often impractical in real-world scenarios, where diverse problems following shifting distribution continually arise. Consequently, there is a pressing need for methods that can continuously learn from new problems encountered on-the-fly and progressively enhance their capabilities. In this work, we explore a novel paradigm of lifelong learning in MetaBBO and introduce LiBOG, a novel approach designed to learn from sequentially encountered problems and generate high-performance optimizers for Black-Box Optimization (BBO). LiBOG consolidates knowledge both across tasks and within tasks to mitigate catastrophic forgetting. Extensive experiments demonstrate LiBOG's effectiveness in learning to generate high-performance optimizers in a lifelong learning manner, addressing catastrophic forgetting while maintaining plasticity to learn new tasks. 

---
# Anti-Inpainting: A Proactive Defense against Malicious Diffusion-based Inpainters under Unknown Conditions 

**Authors**: Yimao Guo, Zuomin Qu, Wei Lu, Xiangyang Luo  

**Link**: [PDF](https://arxiv.org/pdf/2505.13023)  

**Abstract**: As diffusion-based malicious image manipulation becomes increasingly prevalent, multiple proactive defense methods are developed to safeguard images against unauthorized tampering. However, most proactive defense methods only can safeguard images against manipulation under known conditions, and fail to protect images from manipulations guided by tampering conditions crafted by malicious users. To tackle this issue, we propose Anti-Inpainting, a proactive defense method that achieves adequate protection under unknown conditions through a triple mechanism to address this challenge. Specifically, a multi-level deep feature extractor is presented to obtain intricate features during the diffusion denoising process to improve protective effectiveness. We design multi-scale semantic-preserving data augmentation to enhance the transferability of adversarial perturbations across unknown conditions by multi-scale transformations while preserving semantic integrity. In addition, we propose a selection-based distribution deviation optimization strategy to improve the protection of adversarial perturbation against manipulation under diverse random seeds. Extensive experiments indicate the proactive defensive performance of Anti-Inpainting against diffusion-based inpainters guided by unknown conditions in InpaintGuardBench and CelebA-HQ. At the same time, we also demonstrate the proposed approach's robustness under various image purification methods and its transferability across different versions of diffusion models. 

---
# To Bias or Not to Bias: Detecting bias in News with bias-detector 

**Authors**: Himel Ghosh, Ahmed Mosharafa, Georg Groh  

**Link**: [PDF](https://arxiv.org/pdf/2505.13010)  

**Abstract**: Media bias detection is a critical task in ensuring fair and balanced information dissemination, yet it remains challenging due to the subjectivity of bias and the scarcity of high-quality annotated data. In this work, we perform sentence-level bias classification by fine-tuning a RoBERTa-based model on the expert-annotated BABE dataset. Using McNemar's test and the 5x2 cross-validation paired t-test, we show statistically significant improvements in performance when comparing our model to a domain-adaptively pre-trained DA-RoBERTa baseline. Furthermore, attention-based analysis shows that our model avoids common pitfalls like oversensitivity to politically charged terms and instead attends more meaningfully to contextually relevant tokens. For a comprehensive examination of media bias, we present a pipeline that combines our model with an already-existing bias-type classifier. Our method exhibits good generalization and interpretability, despite being constrained by sentence-level analysis and dataset size because of a lack of larger and more advanced bias corpora. We talk about context-aware modeling, bias neutralization, and advanced bias type classification as potential future directions. Our findings contribute to building more robust, explainable, and socially responsible NLP systems for media bias detection. 

---
# ExTrans: Multilingual Deep Reasoning Translation via Exemplar-Enhanced Reinforcement Learning 

**Authors**: Jiaan Wang, Fandong Meng, Jie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.12996)  

**Abstract**: In recent years, the emergence of large reasoning models (LRMs), such as OpenAI-o1 and DeepSeek-R1, has shown impressive capabilities in complex problems, e.g., mathematics and coding. Some pioneering studies attempt to bring the success of LRMs in neural machine translation (MT). They try to build LRMs with deep reasoning MT ability via reinforcement learning (RL). Despite some progress that has been made, these attempts generally focus on several high-resource languages, e.g., English and Chinese, leaving the performance on other languages unclear. Besides, the reward modeling methods in previous work do not fully unleash the potential of reinforcement learning in MT. In this work, we first design a new reward modeling method that compares the translation results of the policy MT model with a strong LRM (i.e., DeepSeek-R1-671B), and quantifies the comparisons to provide rewards. Experimental results demonstrate the superiority of the reward modeling method. Using Qwen2.5-7B-Instruct as the backbone, the trained model achieves the new state-of-the-art performance in literary translation, and outperforms strong LRMs including OpenAI-o1 and DeepSeeK-R1. Furthermore, we extend our method to the multilingual settings with 11 languages. With a carefully designed lightweight reward modeling in RL, we can simply transfer the strong MT ability from a single direction into multiple (i.e., 90) translation directions and achieve impressive multilingual MT performance. 

---
# Fractured Chain-of-Thought Reasoning 

**Authors**: Baohao Liao, Hanze Dong, Yuhui Xu, Doyen Sahoo, Christof Monz, Junnan Li, Caiming Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2505.12992)  

**Abstract**: Inference-time scaling techniques have significantly bolstered the reasoning capabilities of large language models (LLMs) by harnessing additional computational effort at inference without retraining. Similarly, Chain-of-Thought (CoT) prompting and its extension, Long CoT, improve accuracy by generating rich intermediate reasoning trajectories, but these approaches incur substantial token costs that impede their deployment in latency-sensitive settings. In this work, we first show that truncated CoT, which stops reasoning before completion and directly generates the final answer, often matches full CoT sampling while using dramatically fewer tokens. Building on this insight, we introduce Fractured Sampling, a unified inference-time strategy that interpolates between full CoT and solution-only sampling along three orthogonal axes: (1) the number of reasoning trajectories, (2) the number of final solutions per trajectory, and (3) the depth at which reasoning traces are truncated. Through extensive experiments on five diverse reasoning benchmarks and several model scales, we demonstrate that Fractured Sampling consistently achieves superior accuracy-cost trade-offs, yielding steep log-linear scaling gains in Pass@k versus token budget. Our analysis reveals how to allocate computation across these dimensions to maximize performance, paving the way for more efficient and scalable LLM reasoning. 

---
# An Empirical Study of Many-to-Many Summarization with Large Language Models 

**Authors**: Jiaan Wang, Fandong Meng, Zengkui Sun, Yunlong Liang, Yuxuan Cao, Jiarong Xu, Haoxiang Shi, Jie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.12983)  

**Abstract**: Many-to-many summarization (M2MS) aims to process documents in any language and generate the corresponding summaries also in any language. Recently, large language models (LLMs) have shown strong multi-lingual abilities, giving them the potential to perform M2MS in real applications. This work presents a systematic empirical study on LLMs' M2MS ability. Specifically, we first reorganize M2MS data based on eight previous domain-specific datasets. The reorganized data contains 47.8K samples spanning five domains and six languages, which could be used to train and evaluate LLMs. Then, we benchmark 18 LLMs in a zero-shot manner and an instruction-tuning manner. Fine-tuned traditional models (e.g., mBART) are also conducted for comparisons. Our experiments reveal that, zero-shot LLMs achieve competitive results with fine-tuned traditional models. After instruct-tuning, open-source LLMs can significantly improve their M2MS ability, and outperform zero-shot LLMs (including GPT-4) in terms of automatic evaluations. In addition, we demonstrate that this task-specific improvement does not sacrifice the LLMs' general task-solving abilities. However, as revealed by our human evaluation, LLMs still face the factuality issue, and the instruction tuning might intensify the issue. Thus, how to control factual errors becomes the key when building LLM summarizers in real applications, and is worth noting in future research. 

---
# From Assistants to Adversaries: Exploring the Security Risks of Mobile LLM Agents 

**Authors**: Liangxuan Wu, Chao Wang, Tianming Liu, Yanjie Zhao, Haoyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12981)  

**Abstract**: The growing adoption of large language models (LLMs) has led to a new paradigm in mobile computing--LLM-powered mobile AI agents--capable of decomposing and automating complex tasks directly on smartphones. However, the security implications of these agents remain largely unexplored. In this paper, we present the first comprehensive security analysis of mobile LLM agents, encompassing three representative categories: System-level AI Agents developed by original equipment manufacturers (e.g., YOYO Assistant), Third-party Universal Agents (e.g., Zhipu AI AutoGLM), and Emerging Agent Frameworks (e.g., Alibaba Mobile Agent). We begin by analyzing the general workflow of mobile agents and identifying security threats across three core capability dimensions: language-based reasoning, GUI-based interaction, and system-level execution. Our analysis reveals 11 distinct attack surfaces, all rooted in the unique capabilities and interaction patterns of mobile LLM agents, and spanning their entire operational lifecycle. To investigate these threats in practice, we introduce AgentScan, a semi-automated security analysis framework that systematically evaluates mobile LLM agents across all 11 attack scenarios. Applying AgentScan to nine widely deployed agents, we uncover a concerning trend: every agent is vulnerable to targeted attacks. In the most severe cases, agents exhibit vulnerabilities across eight distinct attack vectors. These attacks can cause behavioral deviations, privacy leakage, or even full execution hijacking. Based on these findings, we propose a set of defensive design principles and practical recommendations for building secure mobile LLM agents. Our disclosures have received positive feedback from two major device vendors. Overall, this work highlights the urgent need for standardized security practices in the fast-evolving landscape of LLM-driven mobile automation. 

---
# Multiscale Adaptive Conflict-Balancing Model For Multimedia Deepfake Detection 

**Authors**: Zihan Xiong, Xiaohua Wu, Lei Chen, Fangqi Lou  

**Link**: [PDF](https://arxiv.org/pdf/2505.12966)  

**Abstract**: Advances in computer vision and deep learning have blurred the line between deepfakes and authentic media, undermining multimedia credibility through audio-visual forgery. Current multimodal detection methods remain limited by unbalanced learning between modalities. To tackle this issue, we propose an Audio-Visual Joint Learning Method (MACB-DF) to better mitigate modality conflicts and neglect by leveraging contrastive learning to assist in multi-level and cross-modal fusion, thereby fully balancing and exploiting information from each modality. Additionally, we designed an orthogonalization-multimodal pareto module that preserves unimodal information while addressing gradient conflicts in audio-video encoders caused by differing optimization targets of the loss functions. Extensive experiments and ablation studies conducted on mainstream deepfake datasets demonstrate consistent performance gains of our model across key evaluation metrics, achieving an average accuracy of 95.5% across multiple datasets. Notably, our method exhibits superior cross-dataset generalization capabilities, with absolute improvements of 8.0% and 7.7% in ACC scores over the previous best-performing approach when trained on DFDC and tested on DefakeAVMiT and FakeAVCeleb datasets. 

---
# Segmentation of temporomandibular joint structures on mri images using neural networks for diagnosis of pathologies 

**Authors**: Maksim I. Ivanov, Olga E. Mendybaeva, Yuri E. Karyakin, Igor N. Glukhikh, Aleksey V. Lebedev  

**Link**: [PDF](https://arxiv.org/pdf/2505.12963)  

**Abstract**: This article explores the use of artificial intelligence for the diagnosis of pathologies of the temporomandibular joint (TMJ), in particular, for the segmentation of the articular disc on MRI images. The relevance of the work is due to the high prevalence of TMJ pathologies, as well as the need to improve the accuracy and speed of diagnosis in medical institutions. During the study, the existing solutions (Diagnocat, MandSeg) were analyzed, which, as a result, are not suitable for studying the articular disc due to the orientation towards bone structures. To solve the problem, an original dataset was collected from 94 images with the classes "temporomandibular joint" and "jaw". To increase the amount of data, augmentation methods were used. After that, the models of U-Net, YOLOv8n, YOLOv11n and Roboflow neural networks were trained and compared. The evaluation was carried out according to the Dice Score, Precision, Sensitivity, Specificity, and Mean Average Precision metrics. The results confirm the potential of using the Roboflow model for segmentation of the temporomandibular joint. In the future, it is planned to develop an algorithm for measuring the distance between the jaws and determining the position of the articular disc, which will improve the diagnosis of TMJ pathologies. 

---
# Hardware-Adaptive and Superlinear-Capacity Memristor-based Associative Memory 

**Authors**: Chengping He, Mingrui Jiang, Keyi Shan, Szu-Hao Yang, Zefan Li, Shengbo Wang, Giacomo Pedretti, Jim Ignowski, Can Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.12960)  

**Abstract**: Brain-inspired computing aims to mimic cognitive functions like associative memory, the ability to recall complete patterns from partial cues. Memristor technology offers promising hardware for such neuromorphic systems due to its potential for efficient in-memory analog computing. Hopfield Neural Networks (HNNs) are a classic model for associative memory, but implementations on conventional hardware suffer from efficiency bottlenecks, while prior memristor-based HNNs faced challenges with vulnerability to hardware defects due to offline training, limited storage capacity, and difficulty processing analog patterns. Here we introduce and experimentally demonstrate on integrated memristor hardware a new hardware-adaptive learning algorithm for associative memories that significantly improves defect tolerance and capacity, and naturally extends to scalable multilayer architectures capable of handling both binary and continuous patterns. Our approach achieves 3x effective capacity under 50% device faults compared to state-of-the-art methods. Furthermore, its extension to multilayer architectures enables superlinear capacity scaling (\(\propto N^{1.49}\ for binary patterns) and effective recalling of continuous patterns (\propto N^{1.74}\ scaling), as compared to linear capacity scaling for previous HNNs. It also provides flexibility to adjust capacity by tuning hidden neurons for the same-sized patterns. By leveraging the massive parallelism of the hardware enabled by synchronous updates, it reduces energy by 8.8x and latency by 99.7% for 64-dimensional patterns over asynchronous schemes, with greater improvements at scale. This promises the development of more reliable memristor-based associative memory systems and enables new applications research due to the significantly improved capacity, efficiency, and flexibility. 

---
# DGRO: Enhancing LLM Reasoning via Exploration-Exploitation Control and Reward Variance Management 

**Authors**: Xuerui Su, Liya Guo, Yue Wang, Yi Zhu, Zhiming Ma, Zun Wang, Yuting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12951)  

**Abstract**: Inference scaling further accelerates Large Language Models (LLMs) toward Artificial General Intelligence (AGI), with large-scale Reinforcement Learning (RL) to unleash long Chain-of-Thought reasoning. Most contemporary reasoning approaches usually rely on handcrafted rule-based reward functions. However, the tarde-offs of exploration and exploitation in RL algorithms involves multiple complex considerations, and the theoretical and empirical impacts of manually designed reward functions remain insufficiently explored. In this paper, we propose Decoupled Group Reward Optimization (DGRO), a general RL algorithm for LLM reasoning. On the one hand, DGRO decouples the traditional regularization coefficient into two independent hyperparameters: one scales the policy gradient term, and the other regulates the distance from the sampling policy. This decoupling not only enables precise control over balancing exploration and exploitation, but also can be seamlessly extended to Online Policy Mirror Descent (OPMD) algorithms in Kimi k1.5 and Direct Reward Optimization. On the other hand, we observe that reward variance significantly affects both convergence speed and final model performance. We conduct both theoretical analysis and extensive empirical validation to assess DGRO, including a detailed ablation study that investigates its performance and optimization dynamics. Experimental results show that DGRO achieves state-of-the-art performance on the Logic dataset with an average accuracy of 96.9\%, and demonstrates strong generalization across mathematical benchmarks. 

---
# CALM-PDE: Continuous and Adaptive Convolutions for Latent Space Modeling of Time-dependent PDEs 

**Authors**: Jan Hagnberger, Daniel Musekamp, Mathias Niepert  

**Link**: [PDF](https://arxiv.org/pdf/2505.12944)  

**Abstract**: Solving time-dependent Partial Differential Equations (PDEs) using a densely discretized spatial domain is a fundamental problem in various scientific and engineering disciplines, including modeling climate phenomena and fluid dynamics. However, performing these computations directly in the physical space often incurs significant computational costs. To address this issue, several neural surrogate models have been developed that operate in a compressed latent space to solve the PDE. While these approaches reduce computational complexity, they often use Transformer-based attention mechanisms to handle irregularly sampled domains, resulting in increased memory consumption. In contrast, convolutional neural networks allow memory-efficient encoding and decoding but are limited to regular discretizations. Motivated by these considerations, we propose CALM-PDE, a model class that efficiently solves arbitrarily discretized PDEs in a compressed latent space. We introduce a novel continuous convolution-based encoder-decoder architecture that uses an epsilon-neighborhood-constrained kernel and learns to apply the convolution operator to adaptive and optimized query points. We demonstrate the effectiveness of CALM-PDE on a diverse set of PDEs with both regularly and irregularly sampled spatial domains. CALM-PDE is competitive with or outperforms existing baseline methods while offering significant improvements in memory and inference time efficiency compared to Transformer-based methods. 

---
# A3 : an Analytical Low-Rank Approximation Framework for Attention 

**Authors**: Jeffrey T. H. Wong, Cheng Zhang, Xinye Cao, Pedro Gimenes, George A. Constantinides, Wayne Luk, Yiren Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.12942)  

**Abstract**: Large language models have demonstrated remarkable performance; however, their massive parameter counts make deployment highly expensive. Low-rank approximation offers a promising compression solution, yet existing approaches have two main limitations: (1) They focus on minimizing the output error of individual linear layers, without considering the architectural characteristics of Transformers, and (2) they decompose a large weight matrix into two small low-rank matrices. Consequently, these methods often fall short compared to other compression techniques like pruning and quantization, and introduce runtime overhead such as the extra GEMM kernel launches for decomposed small matrices. To address these limitations, we propose $\tt A^\tt 3$, a post-training low-rank approximation framework. $\tt A^\tt 3$ splits a Transformer layer into three functional components, namely $\tt QK$, $\tt OV$, and $\tt MLP$. For each component, $\tt A^\tt 3$ provides an analytical solution that reduces the hidden dimension size inside each component while minimizing the component's functional loss ($\it i.e.$, error in attention scores, attention outputs, and MLP outputs). This approach directly reduces model sizes, KV cache sizes, and FLOPs without introducing any runtime overheads. In addition, it provides a new narrative in advancing the optimization problem from singular linear layer loss optimization toward improved end-to-end performance. Through extensive experiments, we show that $\tt A^\tt 3$ maintains superior performance compared to SoTAs. For example, under the same reduction budget in computation and memory, our low-rank approximated LLaMA 3.1-70B achieves a perplexity of 4.69 on WikiText-2, outperforming the previous SoTA's 7.87 by 3.18. We also demonstrate the versatility of $\tt A^\tt 3$, including KV cache compression, quantization, and mixed-rank assignments for enhanced performance. 

---
# Leveraging LLM Inconsistency to Boost Pass@k Performance 

**Authors**: Uri Dalal, Meirav Segal, Zvika Ben-Haim, Dan Lahav, Omer Nevo  

**Link**: [PDF](https://arxiv.org/pdf/2505.12938)  

**Abstract**: Large language models (LLMs) achieve impressive abilities in numerous domains, but exhibit inconsistent performance in response to minor input changes. Rather than view this as a drawback, in this paper we introduce a novel method for leveraging models' inconsistency to boost Pass@k performance. Specifically, we present a "Variator" agent that generates k variants of a given task and submits one candidate solution for each one. Our variant generation approach is applicable to a wide range of domains as it is task agnostic and compatible with free-form inputs. We demonstrate the efficacy of our agent theoretically using a probabilistic model of the inconsistency effect, and show empirically that it outperforms the baseline on the APPS dataset. Furthermore, we establish that inconsistency persists even in frontier reasoning models across coding and cybersecurity domains, suggesting our method is likely to remain relevant for future model generations. 

---
# Do Not Let Low-Probability Tokens Over-Dominate in RL for LLMs 

**Authors**: Zhihe Yang, Xufang Luo, Zilong Wang, Dongqi Han, Zhiyuan He, Dongsheng Li, Yunjian Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12929)  

**Abstract**: Reinforcement learning (RL) has become a cornerstone for enhancing the reasoning capabilities of large language models (LLMs), with recent innovations such as Group Relative Policy Optimization (GRPO) demonstrating exceptional effectiveness. In this study, we identify a critical yet underexplored issue in RL training: low-probability tokens disproportionately influence model updates due to their large gradient magnitudes. This dominance hinders the effective learning of high-probability tokens, whose gradients are essential for LLMs' performance but are substantially suppressed. To mitigate this interference, we propose two novel methods: Advantage Reweighting and Low-Probability Token Isolation (Lopti), both of which effectively attenuate gradients from low-probability tokens while emphasizing parameter updates driven by high-probability tokens. Our approaches promote balanced updates across tokens with varying probabilities, thereby enhancing the efficiency of RL training. Experimental results demonstrate that they substantially improve the performance of GRPO-trained LLMs, achieving up to a 46.2% improvement in K&K Logic Puzzle reasoning tasks. Our implementation is available at this https URL. 

---
# CPRet: A Dataset, Benchmark, and Model for Retrieval in Competitive Programming 

**Authors**: Han Deng, Yuan Meng, Shixiang Tang, Wanli Ouyang, Xinzhu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.12925)  

**Abstract**: Competitive programming benchmarks are widely used in scenarios such as programming contests and large language model assessments. However, the growing presence of duplicate or highly similar problems raises concerns not only about competition fairness, but also about the validity of competitive programming as a benchmark for model evaluation. In this paper, we propose a new problem -- similar question retrieval -- to address this issue. Due to the lack of both data and models, solving this problem is challenging. To this end, we introduce CPRet, a retrieval-oriented benchmark suite for competitive programming, covering four retrieval tasks: two code-centric (i.e., Text-to-Code and Code-to-Code) and two newly proposed problem-centric tasks (i.e., Problem-to-Duplicate and Simplified-to-Full), built from a combination of automatically crawled problem-solution data and manually curated annotations. Our contribution includes both high-quality training data and temporally separated test sets for reliable evaluation. In addition, we develop two task-specialized retrievers based on this dataset: CPRetriever-Code, trained with a novel Group-InfoNCE loss for problem-code alignment, and CPRetriever-Prob, fine-tuned for identifying problem-level similarity. Both models achieve strong results and are open-sourced for local use. Finally, we analyze LiveCodeBench and find that high-similarity problems inflate model pass rates and reduce differentiation, underscoring the need for similarity-aware evaluation in future benchmarks.
Code and data are available at: this https URL 

---
# PyFCG: Fluid Construction Grammar in Python 

**Authors**: Paul Van Eecke, Katrien Beuls  

**Link**: [PDF](https://arxiv.org/pdf/2505.12920)  

**Abstract**: We present PyFCG, an open source software library that ports Fluid Construction Grammar (FCG) to the Python programming language. PyFCG enables its users to seamlessly integrate FCG functionality into Python programs, and to use FCG in combination with other libraries within Python's rich ecosystem. Apart from a general description of the library, this paper provides three walkthrough tutorials that demonstrate example usage of PyFCG in typical use cases of FCG: (i) formalising and testing construction grammar analyses, (ii) learning usage-based construction grammars from corpora, and (iii) implementing agent-based experiments on emergent communication. 

---
# SourceDetMamba: A Graph-aware State Space Model for Source Detection in Sequential Hypergraphs 

**Authors**: Le Cheng, Peican Zhu, Yangming Guo, Chao Gao, Zhen Wang, Keke Tang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12910)  

**Abstract**: Source detection on graphs has demonstrated high efficacy in identifying rumor origins. Despite advances in machine learning-based methods, many fail to capture intrinsic dynamics of rumor propagation. In this work, we present SourceDetMamba: A Graph-aware State Space Model for Source Detection in Sequential Hypergraphs, which harnesses the recent success of the state space model Mamba, known for its superior global modeling capabilities and computational efficiency, to address this challenge. Specifically, we first employ hypergraphs to model high-order interactions within social networks. Subsequently, temporal network snapshots generated during the propagation process are sequentially fed in reverse order into Mamba to infer underlying propagation dynamics. Finally, to empower the sequential model to effectively capture propagation patterns while integrating structural information, we propose a novel graph-aware state update mechanism, wherein the state of each node is propagated and refined by both temporal dependencies and topological context. Extensive evaluations on eight datasets demonstrate that SourceDetMamba consistently outperforms state-of-the-art approaches. 

---
# Sinusoidal Initialization, Time for a New Start 

**Authors**: Alberto Fernández-Hernández, Jose I. Mestre, Manuel F. Dolz, Jose Duato, Enrique S. Quintana-Ortí  

**Link**: [PDF](https://arxiv.org/pdf/2505.12909)  

**Abstract**: Initialization plays a critical role in Deep Neural Network training, directly influencing convergence, stability, and generalization. Common approaches such as Glorot and He initializations rely on randomness, which can produce uneven weight distributions across layer connections. In this paper, we introduce the Sinusoidal initialization, a novel deterministic method that employs sinusoidal functions to construct structured weight matrices expressly to improve the spread and balance of weights throughout the network while simultaneously fostering a more uniform, well-conditioned distribution of neuron activation states from the very first forward pass. Because Sinusoidal initialization begins with weights and activations that are already evenly and efficiently utilized, it delivers consistently faster convergence, greater training stability, and higher final accuracy across a wide range of models, including convolutional neural networks, vision transformers, and large language models. On average, our experiments show an increase of 4.8 % in final validation accuracy and 20.9 % in convergence speed. By replacing randomness with structure, this initialization provides a stronger and more reliable foundation for Deep Learning systems. 

---
# Dynamic Graph Induced Contour-aware Heat Conduction Network for Event-based Object Detection 

**Authors**: Xiao Wang, Yu Jin, Lan Chen, Bo Jiang, Lin Zhu, Yonghong Tian, Jin Tang, Bin Luo  

**Link**: [PDF](https://arxiv.org/pdf/2505.12908)  

**Abstract**: Event-based Vision Sensors (EVS) have demonstrated significant advantages over traditional RGB frame-based cameras in low-light conditions, high-speed motion capture, and low latency. Consequently, object detection based on EVS has attracted increasing attention from researchers. Current event stream object detection algorithms are typically built upon Convolutional Neural Networks (CNNs) or Transformers, which either capture limited local features using convolutional filters or incur high computational costs due to the utilization of self-attention. Recently proposed vision heat conduction backbone networks have shown a good balance between efficiency and accuracy; however, these models are not specifically designed for event stream data. They exhibit weak capability in modeling object contour information and fail to exploit the benefits of multi-scale features. To address these issues, this paper proposes a novel dynamic graph induced contour-aware heat conduction network for event stream based object detection, termed CvHeat-DET. The proposed model effectively leverages the clear contour information inherent in event streams to predict the thermal diffusivity coefficients within the heat conduction model, and integrates hierarchical structural graph features to enhance feature learning across multiple scales. Extensive experiments on three benchmark datasets for event stream-based object detection fully validated the effectiveness of the proposed model. The source code of this paper will be released on this https URL. 

---
# The Computation of Generalized Embeddings for Underwater Acoustic Target Recognition using Contrastive Learning 

**Authors**: Hilde I. Hummel, Arwin Gansekoele, Sandjai Bhulai, Rob van der Mei  

**Link**: [PDF](https://arxiv.org/pdf/2505.12904)  

**Abstract**: The increasing level of sound pollution in marine environments poses an increased threat to ocean health, making it crucial to monitor underwater noise. By monitoring this noise, the sources responsible for this pollution can be mapped. Monitoring is performed by passively listening to these sounds. This generates a large amount of data records, capturing a mix of sound sources such as ship activities and marine mammal vocalizations. Although machine learning offers a promising solution for automatic sound classification, current state-of-the-art methods implement supervised learning. This requires a large amount of high-quality labeled data that is not publicly available. In contrast, a massive amount of lower-quality unlabeled data is publicly available, offering the opportunity to explore unsupervised learning techniques. This research explores this possibility by implementing an unsupervised Contrastive Learning approach. Here, a Conformer-based encoder is optimized by the so-called Variance-Invariance-Covariance Regularization loss function on these lower-quality unlabeled data and the translation to the labeled data is made. Through classification tasks involving recognizing ship types and marine mammal vocalizations, our method demonstrates to produce robust and generalized embeddings. This shows to potential of unsupervised methods for various automatic underwater acoustic analysis tasks. 

---
# Towards Low-Latency Event Stream-based Visual Object Tracking: A Slow-Fast Approach 

**Authors**: Shiao Wang, Xiao Wang, Liye Jin, Bo Jiang, Lin Zhu, Lan Chen, Yonghong Tian, Bin Luo  

**Link**: [PDF](https://arxiv.org/pdf/2505.12903)  

**Abstract**: Existing tracking algorithms typically rely on low-frame-rate RGB cameras coupled with computationally intensive deep neural network architectures to achieve effective tracking. However, such frame-based methods inherently face challenges in achieving low-latency performance and often fail in resource-constrained environments. Visual object tracking using bio-inspired event cameras has emerged as a promising research direction in recent years, offering distinct advantages for low-latency applications. In this paper, we propose a novel Slow-Fast Tracking paradigm that flexibly adapts to different operational requirements, termed SFTrack. The proposed framework supports two complementary modes, i.e., a high-precision slow tracker for scenarios with sufficient computational resources, and an efficient fast tracker tailored for latency-aware, resource-constrained environments. Specifically, our framework first performs graph-based representation learning from high-temporal-resolution event streams, and then integrates the learned graph-structured information into two FlashAttention-based vision backbones, yielding the slow and fast trackers, respectively. The fast tracker achieves low latency through a lightweight network design and by producing multiple bounding box outputs in a single forward pass. Finally, we seamlessly combine both trackers via supervised fine-tuning and further enhance the fast tracker's performance through a knowledge distillation strategy. Extensive experiments on public benchmarks, including FE240, COESOT, and EventVOT, demonstrate the effectiveness and efficiency of our proposed method across different real-world scenarios. The source code has been released on this https URL. 

---
# AutoGEEval: A Multimodal and Automated Framework for Geospatial Code Generation on GEE with Large Language Models 

**Authors**: Shuyang Hou, Zhangxiao Shen, Huayi Wu, Jianyuan Liang, Haoyue Jiao, Yaxian Qing, Xiaopu Zhang, Xu Li, Zhipeng Gui, Xuefeng Guan, Longgang Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12900)  

**Abstract**: Geospatial code generation is emerging as a key direction in the integration of artificial intelligence and geoscientific analysis. However, there remains a lack of standardized tools for automatic evaluation in this domain. To address this gap, we propose AutoGEEval, the first multimodal, unit-level automated evaluation framework for geospatial code generation tasks on the Google Earth Engine (GEE) platform powered by large language models (LLMs). Built upon the GEE Python API, AutoGEEval establishes a benchmark suite (AutoGEEval-Bench) comprising 1325 test cases that span 26 GEE data types. The framework integrates both question generation and answer verification components to enable an end-to-end automated evaluation pipeline-from function invocation to execution validation. AutoGEEval supports multidimensional quantitative analysis of model outputs in terms of accuracy, resource consumption, execution efficiency, and error types. We evaluate 18 state-of-the-art LLMs-including general-purpose, reasoning-augmented, code-centric, and geoscience-specialized models-revealing their performance characteristics and potential optimization pathways in GEE code generation. This work provides a unified protocol and foundational resource for the development and assessment of geospatial code generation models, advancing the frontier of automated natural language to domain-specific code translation. 

---
# HyperDet: Source Detection in Hypergraphs via Interactive Relationship Construction and Feature-rich Attention Fusion 

**Authors**: Le Cheng, Peican Zhu, Yangming Guo, Keke Tang, Chao Gao, Zhen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12894)  

**Abstract**: Hypergraphs offer superior modeling capabilities for social networks, particularly in capturing group phenomena that extend beyond pairwise interactions in rumor propagation. Existing approaches in rumor source detection predominantly focus on dyadic interactions, which inadequately address the complexity of more intricate relational structures. In this study, we present a novel approach for Source Detection in Hypergraphs (HyperDet) via Interactive Relationship Construction and Feature-rich Attention Fusion. Specifically, our methodology employs an Interactive Relationship Construction module to accurately model both the static topology and dynamic interactions among users, followed by the Feature-rich Attention Fusion module, which autonomously learns node features and discriminates between nodes using a self-attention mechanism, thereby effectively learning node representations under the framework of accurately modeled higher-order relationships. Extensive experimental validation confirms the efficacy of our HyperDet approach, showcasing its superiority relative to current state-of-the-art methods. 

---
# TinyAlign: Boosting Lightweight Vision-Language Models by Mitigating Modal Alignment Bottlenecks 

**Authors**: Yuanze Hu, Zhaoxin Fan, Xinyu Wang, Gen Li, Ye Qiu, Zhichao Yang, Wenjun Wu, Kejian Wu, Yifan Sun, Xiaotie Deng, Jin Dong  

**Link**: [PDF](https://arxiv.org/pdf/2505.12884)  

**Abstract**: Lightweight Vision-Language Models (VLMs) are indispensable for resource-constrained applications. The prevailing approach to aligning vision and language models involves freezing both the vision encoder and the language model while training small connector modules. However, this strategy heavily depends on the intrinsic capabilities of the language model, which can be suboptimal for lightweight models with limited representational capacity. In this work, we investigate this alignment bottleneck through the lens of mutual information, demonstrating that the constrained capacity of the language model inherently limits the Effective Mutual Information (EMI) between multimodal inputs and outputs, thereby compromising alignment quality. To address this challenge, we propose TinyAlign, a novel framework inspired by Retrieval-Augmented Generation, which strategically retrieves relevant context from a memory bank to enrich multimodal inputs and enhance their alignment. Extensive empirical evaluations reveal that TinyAlign significantly reduces training loss, accelerates convergence, and enhances task performance. Remarkably, it allows models to achieve baseline-level performance with only 40\% of the fine-tuning data, highlighting exceptional data efficiency. Our work thus offers a practical pathway for developing more capable lightweight VLMs while introducing a fresh theoretical lens to better understand and address alignment bottlenecks in constrained multimodal systems. 

---
# PhyDA: Physics-Guided Diffusion Models for Data Assimilation in Atmospheric Systems 

**Authors**: Hao Wang, Jindong Han, Wei Fan, Weijia Zhang, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12882)  

**Abstract**: Data Assimilation (DA) plays a critical role in atmospheric science by reconstructing spatially continous estimates of the system state, which serves as initial conditions for scientific analysis. While recent advances in diffusion models have shown great potential for DA tasks, most existing approaches remain purely data-driven and often overlook the physical laws that govern complex atmospheric dynamics. As a result, they may yield physically inconsistent reconstructions that impair downstream applications. To overcome this limitation, we propose PhyDA, a physics-guided diffusion framework designed to ensure physical coherence in atmospheric data assimilation. PhyDA introduces two key components: (1) a Physically Regularized Diffusion Objective that integrates physical constraints into the training process by penalizing deviations from known physical laws expressed as partial differential equations, and (2) a Virtual Reconstruction Encoder that bridges observational sparsity for structured latent representations, further enhancing the model's ability to infer complete and physically coherent states. Experiments on the ERA5 reanalysis dataset demonstrate that PhyDA achieves superior accuracy and better physical plausibility compared to state-of-the-art baselines. Our results emphasize the importance of combining generative modeling with domain-specific physical knowledge and show that PhyDA offers a promising direction for improving real-world data assimilation systems. 

---
# AdS-GNN -- a Conformally Equivariant Graph Neural Network 

**Authors**: Maksim Zhdanov, Nabil Iqbal, Erik Bekkers, Patrick Forré  

**Link**: [PDF](https://arxiv.org/pdf/2505.12880)  

**Abstract**: Conformal symmetries, i.e.\ coordinate transformations that preserve angles, play a key role in many fields, including physics, mathematics, computer vision and (geometric) machine learning. Here we build a neural network that is equivariant under general conformal transformations. To achieve this, we lift data from flat Euclidean space to Anti de Sitter (AdS) space. This allows us to exploit a known correspondence between conformal transformations of flat space and isometric transformations on the AdS space. We then build upon the fact that such isometric transformations have been extensively studied on general geometries in the geometric deep learning literature. We employ message-passing layers conditioned on the proper distance, yielding a computationally efficient framework. We validate our model on tasks from computer vision and statistical physics, demonstrating strong performance, improved generalization capacities, and the ability to extract conformal data such as scaling dimensions from the trained network. 

---
# Does Low Rank Adaptation Lead to Lower Robustness against Training-Time Attacks? 

**Authors**: Zi Liang, Haibo Hu, Qingqing Ye, Yaxin Xiao, Ronghua Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.12871)  

**Abstract**: Low rank adaptation (LoRA) has emerged as a prominent technique for fine-tuning large language models (LLMs) thanks to its superb efficiency gains over previous methods. While extensive studies have examined the performance and structural properties of LoRA, its behavior upon training-time attacks remain underexplored, posing significant security risks. In this paper, we theoretically investigate the security implications of LoRA's low-rank structure during fine-tuning, in the context of its robustness against data poisoning and backdoor attacks. We propose an analytical framework that models LoRA's training dynamics, employs the neural tangent kernel to simplify the analysis of the training process, and applies information theory to establish connections between LoRA's low rank structure and its vulnerability against training-time attacks. Our analysis indicates that LoRA exhibits better robustness to backdoor attacks than full fine-tuning, while becomes more vulnerable to untargeted data poisoning due to its over-simplified information geometry. Extensive experimental evaluations have corroborated our theoretical findings. 

---
# Outsourced Privacy-Preserving Feature Selection Based on Fully Homomorphic Encryption 

**Authors**: Koki Wakiyama, Tomohiro I, Hiroshi Sakamoto  

**Link**: [PDF](https://arxiv.org/pdf/2505.12869)  

**Abstract**: Feature selection is a technique that extracts a meaningful subset from a set of features in training data. When the training data is large-scale, appropriate feature selection enables the removal of redundant features, which can improve generalization performance, accelerate the training process, and enhance the interpretability of the model. This study proposes a privacy-preserving computation model for feature selection. Generally, when the data owner and analyst are the same, there is no need to conceal the private information. However, when they are different parties or when multiple owners exist, an appropriate privacy-preserving framework is required. Although various private feature selection algorithms, they all require two or more computing parties and do not guarantee security in environments where no external party can be fully trusted. To address this issue, we propose the first outsourcing algorithm for feature selection using fully homomorphic encryption. Compared to a prior two-party algorithm, our result improves the time and space complexity O(kn^2) to O(kn log^3 n) and O(kn), where k and n denote the number of features and data samples, respectively. We also implemented the proposed algorithm and conducted comparative experiments with the naive one. The experimental result shows the efficiency of our method even with small datasets. 

---
# LEXam: Benchmarking Legal Reasoning on 340 Law Exams 

**Authors**: Yu Fan, Jingwei Ni, Jakob Merane, Etienne Salimbeni, Yang Tian, Yoan Hermstrüwer, Yinya Huang, Mubashara Akhtar, Florian Geering, Oliver Dreyer, Daniel Brunner, Markus Leippold, Mrinmaya Sachan, Alexander Stremitzer, Christoph Engel, Elliott Ash, Joel Niklaus  

**Link**: [PDF](https://arxiv.org/pdf/2505.12864)  

**Abstract**: Long-form legal reasoning remains a key challenge for large language models (LLMs) in spite of recent advances in test-time scaling. We introduce LEXam, a novel benchmark derived from 340 law exams spanning 116 law school courses across a range of subjects and degree levels. The dataset comprises 4,886 law exam questions in English and German, including 2,841 long-form, open-ended questions and 2,045 multiple-choice questions. Besides reference answers, the open questions are also accompanied by explicit guidance outlining the expected legal reasoning approach such as issue spotting, rule recall, or rule application. Our evaluation on both open-ended and multiple-choice questions present significant challenges for current LLMs; in particular, they notably struggle with open questions that require structured, multi-step legal reasoning. Moreover, our results underscore the effectiveness of the dataset in differentiating between models with varying capabilities. Adopting an LLM-as-a-Judge paradigm with rigorous human expert validation, we demonstrate how model-generated reasoning steps can be evaluated consistently and accurately. Our evaluation setup provides a scalable method to assess legal reasoning quality beyond simple accuracy metrics. Project page: this https URL 

---
# Unified Cross-modal Translation of Score Images, Symbolic Music, and Performance Audio 

**Authors**: Jongmin Jung, Dongmin Kim, Sihun Lee, Seola Cho, Hyungjoon Soh, Irmak Bukey, Chris Donahue, Dasaem Jeong  

**Link**: [PDF](https://arxiv.org/pdf/2505.12863)  

**Abstract**: Music exists in various modalities, such as score images, symbolic scores, MIDI, and audio. Translations between each modality are established as core tasks of music information retrieval, such as automatic music transcription (audio-to-MIDI) and optical music recognition (score image to symbolic score). However, most past work on multimodal translation trains specialized models on individual translation tasks. In this paper, we propose a unified approach, where we train a general-purpose model on many translation tasks simultaneously. Two key factors make this unified approach viable: a new large-scale dataset and the tokenization of each modality. Firstly, we propose a new dataset that consists of more than 1,300 hours of paired audio-score image data collected from YouTube videos, which is an order of magnitude larger than any existing music modal translation datasets. Secondly, our unified tokenization framework discretizes score images, audio, MIDI, and MusicXML into a sequence of tokens, enabling a single encoder-decoder Transformer to tackle multiple cross-modal translation as one coherent sequence-to-sequence task. Experimental results confirm that our unified multitask model improves upon single-task baselines in several key areas, notably reducing the symbol error rate for optical music recognition from 24.58% to a state-of-the-art 13.67%, while similarly substantial improvements are observed across the other translation tasks. Notably, our approach achieves the first successful score-image-conditioned audio generation, marking a significant breakthrough in cross-modal music generation. 

---
# FLTG: Byzantine-Robust Federated Learning via Angle-Based Defense and Non-IID-Aware Weighting 

**Authors**: Yanhua Wen, Lu Ai, Gang Liu, Chuang Li, Jianhao Wei  

**Link**: [PDF](https://arxiv.org/pdf/2505.12851)  

**Abstract**: Byzantine attacks during model aggregation in Federated Learning (FL) threaten training integrity by manipulating malicious clients' updates. Existing methods struggle with limited robustness under high malicious client ratios and sensitivity to non-i.i.d. data, leading to degraded accuracy. To address this, we propose FLTG, a novel aggregation algorithm integrating angle-based defense and dynamic reference selection. FLTG first filters clients via ReLU-clipped cosine similarity, leveraging a server-side clean dataset to exclude misaligned updates. It then dynamically selects a reference client based on the prior global model to mitigate non-i.i.d. bias, assigns aggregation weights inversely proportional to angular deviations, and normalizes update magnitudes to suppress malicious scaling. Evaluations across datasets of varying complexity under five classic attacks demonstrate FLTG's superiority over state-of-the-art methods under extreme bias scenarios and sustains robustness with a higher proportion(over 50%) of malicious clients. 

---
# Bias Fitting to Mitigate Length Bias of Reward Model in RLHF 

**Authors**: Kangwen Zhao, Jianfeng Cai, Jinhua Zhu, Ruopei Sun, Dongyun Xue, Wengang Zhou, Li Li, Houqiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.12843)  

**Abstract**: Reinforcement Learning from Human Feedback relies on reward models to align large language models with human preferences. However, RLHF often suffers from reward hacking, wherein policy learning exploits flaws in the trained reward model to maximize reward scores without genuinely aligning with human preferences. A significant example of such reward hacking is length bias, where reward models usually favor longer responses irrespective of actual response quality. Previous works on length bias have notable limitations, these approaches either mitigate bias without characterizing the bias form, or simply assume a linear length-reward relation. To accurately model the intricate nature of length bias and facilitate more effective bias mitigation, we propose FiMi-RM (Bias Fitting to Mitigate Length Bias of Reward Model in RLHF), a framework that autonomously learns and corrects underlying bias patterns. Our approach consists of three stages: First, we train a standard reward model which inherently contains length bias. Next, we deploy a lightweight fitting model to explicitly capture the non-linear relation between length and reward. Finally, we incorporate this learned relation into the reward model to debias. Experimental results demonstrate that FiMi-RM achieves a more balanced length-reward distribution. Furthermore, when applied to alignment algorithms, our debiased reward model improves length-controlled win rate and reduces verbosity without compromising its performance. 

---
# The Hidden Structure -- Improving Legal Document Understanding Through Explicit Text Formatting 

**Authors**: Christian Braun, Alexander Lilienbeck, Daniel Mentjukov  

**Link**: [PDF](https://arxiv.org/pdf/2505.12837)  

**Abstract**: Legal contracts possess an inherent, semantically vital structure (e.g., sections, clauses) that is crucial for human comprehension but whose impact on LLM processing remains under-explored. This paper investigates the effects of explicit input text structure and prompt engineering on the performance of GPT-4o and GPT-4.1 on a legal question-answering task using an excerpt of the CUAD. We compare model exact-match accuracy across various input formats: well-structured plain-text (human-generated from CUAD), plain-text cleaned of line breaks, extracted plain-text from Azure OCR, plain-text extracted by GPT-4o Vision, and extracted (and interpreted) Markdown (MD) from GPT-4o Vision. To give an indication of the impact of possible prompt engineering, we assess the impact of shifting task instructions to the system prompt and explicitly informing the model about the structured nature of the input. Our findings reveal that GPT-4o demonstrates considerable robustness to variations in input structure, but lacks in overall performance. Conversely, GPT-4.1's performance is markedly sensitive; poorly structured inputs yield suboptimal results (but identical with GPT-4o), while well-structured formats (original CUAD text, GPT-4o Vision text and GPT-4o MD) improve exact-match accuracy by ~20 percentage points. Optimizing the system prompt to include task details and an advisory about structured input further elevates GPT-4.1's accuracy by an additional ~10-13 percentage points, with Markdown ultimately achieving the highest performance under these conditions (79 percentage points overall exact-match accuracy). This research empirically demonstrates that while newer models exhibit greater resilience, careful input structuring and strategic prompt design remain critical for optimizing the performance of LLMs, and can significantly affect outcomes in high-stakes legal applications. 

---
# SynDec: A Synthesize-then-Decode Approach for Arbitrary Textual Style Transfer via Large Language Models 

**Authors**: Han Sun, Zhen Sun, Zongmin Zhang, Linzhao Jia, Wei Shao, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12821)  

**Abstract**: Large Language Models (LLMs) are emerging as dominant forces for textual style transfer. However, for arbitrary style transfer, LLMs face two key challenges: (1) considerable reliance on manually-constructed prompts and (2) rigid stylistic biases inherent in LLMs. In this paper, we propose a novel Synthesize-then-Decode (SynDec) approach, which automatically synthesizes high-quality prompts and amplifies their roles during decoding process. Specifically, our approach synthesizes prompts by selecting representative few-shot samples, conducting a four-dimensional style analysis, and reranking the candidates. At LLM decoding stage, the TST effect is amplified by maximizing the contrast in output probabilities between scenarios with and without the synthesized prompt, as well as between prompts and negative samples. We conduct extensive experiments and the results show that SynDec outperforms existing state-of-the-art LLM-based methods on five out of six benchmarks (e.g., achieving up to a 9\% increase in accuracy for modern-to-Elizabethan English transfer). Detailed ablation studies further validate the effectiveness of SynDec. 

---
# Learning in Chaos: Efficient Autoscaling and Self-healing for Distributed Training at the Edge 

**Authors**: Wenjiao Feng, Rongxing Xiao, Zonghang Li, Hongfang Yu, Gang Sun, Long Luo, Mohsen Guizani, Qirong Ho  

**Link**: [PDF](https://arxiv.org/pdf/2505.12815)  

**Abstract**: Frequent node and link changes in edge AI clusters disrupt distributed training, while traditional checkpoint-based recovery and cloud-centric autoscaling are too slow for scale-out and ill-suited to chaotic and self-governed edge. This paper proposes Chaos, a resilient and scalable edge distributed training system with built-in self-healing and autoscaling. It speeds up scale-out by using multi-neighbor replication with fast shard scheduling, allowing a new node to pull the latest training state from nearby neighbors in parallel while balancing the traffic load between them. It also uses a cluster monitor to track resource and topology changes to assist scheduler decisions, and handles scaling events through peer negotiation protocols, enabling fully self-governed autoscaling without a central admin. Extensive experiments show that Chaos consistently achieves much lower scale-out delays than Pollux, EDL, and Autoscaling, and handles scale-in, connect-link, and disconnect-link events within 1 millisecond, making it smoother to handle node joins, exits, and failures. It also delivers the lowest idle time, showing superior resource use and scalability as the cluster grows. 

---
# PsyMem: Fine-grained psychological alignment and Explicit Memory Control for Advanced Role-Playing LLMs 

**Authors**: Xilong Cheng, Yunxiao Qin, Yuting Tan, Zhengnan Li, Ye Wang, Hongjiang Xiao, Yuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12814)  

**Abstract**: Existing LLM-based role-playing methods often rely on superficial textual descriptions or simplistic metrics, inadequately modeling both intrinsic and extrinsic character dimensions. Additionally, they typically simulate character memory with implicit model knowledge or basic retrieval augment generation without explicit memory alignment, compromising memory consistency. The two issues weaken reliability of role-playing LLMs in several applications, such as trustworthy social simulation. To address these limitations, we propose PsyMem, a novel framework integrating fine-grained psychological attributes and explicit memory control for role-playing. PsyMem supplements textual descriptions with 26 psychological indicators to detailed model character. Additionally, PsyMem implements memory alignment training, explicitly trains the model to align character's response with memory, thereby enabling dynamic memory-controlled responding during inference. By training Qwen2.5-7B-Instruct on our specially designed dataset (including 5,414 characters and 38,962 dialogues extracted from novels), the resulting model, termed as PsyMem-Qwen, outperforms baseline models in role-playing, achieving the best performance in human-likeness and character fidelity. 

---
# Dynamic Sight Range Selection in Multi-Agent Reinforcement Learning 

**Authors**: Wei-Chen Liao, Ti-Rong Wu, I-Chen Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12811)  

**Abstract**: Multi-agent reinforcement Learning (MARL) is often challenged by the sight range dilemma, where agents either receive insufficient or excessive information from their environment. In this paper, we propose a novel method, called Dynamic Sight Range Selection (DSR), to address this issue. DSR utilizes an Upper Confidence Bound (UCB) algorithm and dynamically adjusts the sight range during training. Experiment results show several advantages of using DSR. First, we demonstrate using DSR achieves better performance in three common MARL environments, including Level-Based Foraging (LBF), Multi-Robot Warehouse (RWARE), and StarCraft Multi-Agent Challenge (SMAC). Second, our results show that DSR consistently improves performance across multiple MARL algorithms, including QMIX and MAPPO. Third, DSR offers suitable sight ranges for different training steps, thereby accelerating the training process. Finally, DSR provides additional interpretability by indicating the optimal sight range used during training. Unlike existing methods that rely on global information or communication mechanisms, our approach operates solely based on the individual sight ranges of agents. This approach offers a practical and efficient solution to the sight range dilemma, making it broadly applicable to real-world complex environments. 

---
# FedSVD: Adaptive Orthogonalization for Private Federated Learning with LoRA 

**Authors**: Seanie Lee, Sangwoo Park, Dong Bok Lee, Dominik Wagner, Haebin Seong, Tobias Bocklet, Juho Lee, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12805)  

**Abstract**: Low-Rank Adaptation (LoRA), which introduces a product of two trainable low-rank matrices into frozen pre-trained weights, is widely used for efficient fine-tuning of language models in federated learning (FL). However, when combined with differentially private stochastic gradient descent (DP-SGD), LoRA faces substantial noise amplification: DP-SGD perturbs per-sample gradients, and the matrix multiplication of the LoRA update ($BA$) intensifies this effect. Freezing one matrix (e.g., $A$) reduces the noise but restricts model expressiveness, often resulting in suboptimal adaptation. To address this, we propose FedSVD, a simple yet effective method that introduces a global reparameterization based on singular value decomposition (SVD). In our approach, each client optimizes only the $B$ matrix and transmits it to the server. The server aggregates the $B$ matrices, computes the product $BA$ using the previous $A$, and refactorizes the result via SVD. This yields a new adaptive $A$ composed of the orthonormal right singular vectors of $BA$, and an updated $B$ containing the remaining SVD components. This reparameterization avoids quadratic noise amplification, while allowing $A$ to better capture the principal directions of the aggregate updates. Moreover, the orthonormal structure of $A$ bounds the gradient norms of $B$ and preserves more signal under DP-SGD, as confirmed by our theoretical analysis. As a result, FedSVD consistently improves stability and performance across a variety of privacy settings and benchmarks, outperforming relevant baselines under both private and non-private regimes. 

---
# OZSpeech: One-step Zero-shot Speech Synthesis with Learned-Prior-Conditioned Flow Matching 

**Authors**: Hieu-Nghia Huynh-Nguyen, Ngoc Son Nguyen, Huynh Nguyen Dang, Thieu Vo, Truong-Son Hy, Van Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2505.12800)  

**Abstract**: Text-to-speech (TTS) systems have seen significant advancements in recent years, driven by improvements in deep learning and neural network architectures. Viewing the output speech as a data distribution, previous approaches often employ traditional speech representations, such as waveforms or spectrograms, within the Flow Matching framework. However, these methods have limitations, including overlooking various speech attributes and incurring high computational costs due to additional constraints introduced during training. To address these challenges, we introduce OZSpeech, the first TTS method to explore optimal transport conditional flow matching with one-step sampling and a learned prior as the condition, effectively disregarding preceding states and reducing the number of sampling steps. Our approach operates on disentangled, factorized components of speech in token format, enabling accurate modeling of each speech attribute, which enhances the TTS system's ability to precisely clone the prompt speech. Experimental results show that our method achieves promising performance over existing methods in content accuracy, naturalness, prosody generation, and speaker style preservation. Audio samples are available at our demo page this https URL. 

---
# A Token is Worth over 1,000 Tokens: Efficient Knowledge Distillation through Low-Rank Clone 

**Authors**: Jitai Hao, Qiang Huang, Hao Liu, Xinyan Xiao, Zhaochun Ren, Jun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12781)  

**Abstract**: Training high-performing Small Language Models (SLMs) remains costly, even with knowledge distillation and pruning from larger teacher models. Existing work often faces three key challenges: (1) information loss from hard pruning, (2) inefficient alignment of representations, and (3) underutilization of informative activations, particularly from Feed-Forward Networks (FFNs). To address these challenges, we introduce Low-Rank Clone (LRC), an efficient pre-training method that constructs SLMs aspiring to behavioral equivalence with strong teacher models. LRC trains a set of low-rank projection matrices that jointly enable soft pruning by compressing teacher weights, and activation clone by aligning student activations, including FFN signals, with those of the teacher. This unified design maximizes knowledge transfer while removing the need for explicit alignment modules. Extensive experiments with open-source teachers (e.g., Llama-3.2-3B-Instruct, Qwen2.5-3B/7B-Instruct) show that LRC matches or surpasses state-of-the-art models trained on trillions of tokens--while using only 20B tokens, achieving over 1,000x training efficiency. Our codes and model checkpoints are available at this https URL and this https URL. 

---
# UniHM: Universal Human Motion Generation with Object Interactions in Indoor Scenes 

**Authors**: Zichen Geng, Zeeshan Hayder, Wei Liu, Ajmal Mian  

**Link**: [PDF](https://arxiv.org/pdf/2505.12774)  

**Abstract**: Human motion synthesis in complex scenes presents a fundamental challenge, extending beyond conventional Text-to-Motion tasks by requiring the integration of diverse modalities such as static environments, movable objects, natural language prompts, and spatial waypoints. Existing language-conditioned motion models often struggle with scene-aware motion generation due to limitations in motion tokenization, which leads to information loss and fails to capture the continuous, context-dependent nature of 3D human movement. To address these issues, we propose UniHM, a unified motion language model that leverages diffusion-based generation for synthesizing scene-aware human motion. UniHM is the first framework to support both Text-to-Motion and Text-to-Human-Object Interaction (HOI) in complex 3D scenes. Our approach introduces three key contributions: (1) a mixed-motion representation that fuses continuous 6DoF motion with discrete local motion tokens to improve motion realism; (2) a novel Look-Up-Free Quantization VAE (LFQ-VAE) that surpasses traditional VQ-VAEs in both reconstruction accuracy and generative performance; and (3) an enriched version of the Lingo dataset augmented with HumanML3D annotations, providing stronger supervision for scene-specific motion learning. Experimental results demonstrate that UniHM achieves comparative performance on the OMOMO benchmark for text-to-HOI synthesis and yields competitive results on HumanML3D for general text-conditioned motion generation. 

---
# Rethinking Reward Model Evaluation Through the Lens of Reward Overoptimization 

**Authors**: Sunghwan Kim, Dongjin Kang, Taeyoon Kwon, Hyungjoo Chae, Dongha Lee, Jinyoung Yeo  

**Link**: [PDF](https://arxiv.org/pdf/2505.12763)  

**Abstract**: Reward models (RMs) play a crucial role in reinforcement learning from human feedback (RLHF), aligning model behavior with human preferences. However, existing benchmarks for reward models show a weak correlation with the performance of optimized policies, suggesting that they fail to accurately assess the true capabilities of RMs. To bridge this gap, we explore several evaluation designs through the lens of reward overoptimization\textemdash a phenomenon that captures both how well the reward model aligns with human preferences and the dynamics of the learning signal it provides to the policy. The results highlight three key findings on how to construct a reliable benchmark: (i) it is important to minimize differences between chosen and rejected responses beyond correctness, (ii) evaluating reward models requires multiple comparisons across a wide range of chosen and rejected responses, and (iii) given that reward models encounter responses with diverse representations, responses should be sourced from a variety of models. However, we also observe that a extremely high correlation with degree of overoptimization leads to comparatively lower correlation with certain downstream performance. Thus, when designing a benchmark, it is desirable to use the degree of overoptimization as a useful tool, rather than the end goal. 

---
# Enhancing Channel-Independent Time-Series Forecasting via Cross-Variate Patch Embedding 

**Authors**: Donghwa Shin, Edwin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12761)  

**Abstract**: Transformers have recently gained popularity in time series forecasting due to their ability to capture long-term dependencies. However, many existing models focus only on capturing temporal dependencies while omitting intricate relationships between variables. Recent models have tried tackling this by explicitly modeling both cross-time and cross-variate dependencies through a sequential or unified attention mechanism, but they are entirely channel dependent (CD) across all layers, making them potentially susceptible to overfitting. To address this, we propose Cross-Variate Patch Embeddings (CVPE), a lightweight CD module that injects cross-variate context into channel-independent (CI) models by simply modifying the patch embedding process. We achieve this by adding a learnable positional encoding and a lightweight router-attention block to the vanilla patch embedding layer. We then integrate CVPE into Time-LLM, a multimodal CI forecasting model, to demonstrate its effectiveness in capturing cross-variate dependencies and enhance the CI model's performance. Extensive experimental results on seven real-world datasets show that our enhanced Time-LLM outperforms the original baseline model simply by incorporating the CVPE module, with no other changes. 

---
# Structure-based Anomaly Detection and Clustering 

**Authors**: Filippo Leveni  

**Link**: [PDF](https://arxiv.org/pdf/2505.12751)  

**Abstract**: Anomaly detection is a fundamental problem in domains such as healthcare, manufacturing, and cybersecurity. This thesis proposes new unsupervised methods for anomaly detection in both structured and streaming data settings. In the first part, we focus on structure-based anomaly detection, where normal data follows low-dimensional manifolds while anomalies deviate from them. We introduce Preference Isolation Forest (PIF), which embeds data into a high-dimensional preference space via manifold fitting, and isolates outliers using two variants: Voronoi-iForest, based on geometric distances, and RuzHash-iForest, leveraging Locality Sensitive Hashing for scalability. We also propose Sliding-PIF, which captures local manifold information for streaming scenarios. Our methods outperform existing techniques on synthetic and real datasets. We extend this to structure-based clustering with MultiLink, a novel method for recovering multiple geometric model families in noisy data. MultiLink merges clusters via a model-aware linkage strategy, enabling robust multi-class structure recovery. It offers key advantages over existing approaches, such as speed, reduced sensitivity to thresholds, and improved robustness to poor initial sampling. The second part of the thesis addresses online anomaly detection in evolving data streams. We propose Online Isolation Forest (Online-iForest), which uses adaptive, multi-resolution histograms and dynamically updates tree structures to track changes over time. It avoids retraining while achieving accuracy comparable to offline models, with superior efficiency for real-time applications. Finally, we tackle anomaly detection in cybersecurity via open-set recognition for malware classification. We enhance a Gradient Boosting classifier with MaxLogit to detect unseen malware families, a method now integrated into Cleafy's production system. 

---
# Malware families discovery via Open-Set Recognition on Android manifest permissions 

**Authors**: Filippo Leveni, Matteo Mistura, Francesco Iubatti, Carmine Giangregorio, Nicolò Pastore, Cesare Alippi, Giacomo Boracchi  

**Link**: [PDF](https://arxiv.org/pdf/2505.12750)  

**Abstract**: Malware are malicious programs that are grouped into families based on their penetration technique, source code, and other characteristics. Classifying malware programs into their respective families is essential for building effective defenses against cyber threats. Machine learning models have a huge potential in malware detection on mobile devices, as malware families can be recognized by classifying permission data extracted from Android manifest files. Still, the malware classification task is challenging due to the high-dimensional nature of permission data and the limited availability of training samples. In particular, the steady emergence of new malware families makes it impossible to acquire a comprehensive training set covering all the malware classes. In this work, we present a malware classification system that, on top of classifying known malware, detects new ones. In particular, we combine an open-set recognition technique developed within the computer vision community, namely MaxLogit, with a tree-based Gradient Boosting classifier, which is particularly effective in classifying high-dimensional data. Our solution turns out to be very practical, as it can be seamlessly employed in a standard classification workflow, and efficient, as it adds minimal computational overhead. Experiments on public and proprietary datasets demonstrate the potential of our solution, which has been deployed in a business environment. 

---
# TeleOpBench: A Simulator-Centric Benchmark for Dual-Arm Dexterous Teleoperation 

**Authors**: Hangyu Li, Qin Zhao, Haoran Xu, Xinyu Jiang, Qingwei Ben, Feiyu Jia, Haoyu Zhao, Liang Xu, Jia Zeng, Hanqing Wang, Bo Dai, Junting Dong, Jiangmiao Pang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12748)  

**Abstract**: Teleoperation is a cornerstone of embodied-robot learning, and bimanual dexterous teleoperation in particular provides rich demonstrations that are difficult to obtain with fully autonomous systems. While recent studies have proposed diverse hardware pipelines-ranging from inertial motion-capture gloves to exoskeletons and vision-based interfaces-there is still no unified benchmark that enables fair, reproducible comparison of these systems. In this paper, we introduce TeleOpBench, a simulator-centric benchmark tailored to bimanual dexterous teleoperation. TeleOpBench contains 30 high-fidelity task environments that span pick-and-place, tool use, and collaborative manipulation, covering a broad spectrum of kinematic and force-interaction difficulty. Within this benchmark we implement four representative teleoperation modalities-(i) MoCap, (ii) VR device, (iii) arm-hand exoskeletons, and (iv) monocular vision tracking-and evaluate them with a common protocol and metric suite. To validate that performance in simulation is predictive of real-world behavior, we conduct mirrored experiments on a physical dual-arm platform equipped with two 6-DoF dexterous hands. Across 10 held-out tasks we observe a strong correlation between simulator and hardware performance, confirming the external validity of TeleOpBench. TeleOpBench establishes a common yardstick for teleoperation research and provides an extensible platform for future algorithmic and hardware innovation. 

---
# PEER pressure: Model-to-Model Regularization for Single Source Domain Generalization 

**Authors**: Dong Kyu Cho, Inwoo Hwang, Sanghack Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.12745)  

**Abstract**: Data augmentation is a popular tool for single source domain generalization, which expands the source domain by generating simulated ones, improving generalization on unseen target domains. In this work, we show that the performance of such augmentation-based methods in the target domains universally fluctuates during training, posing challenges in model selection under realistic scenarios. We argue that the fluctuation stems from the inability of the model to accumulate the knowledge learned from diverse augmentations, exacerbating feature distortion during training. Based on this observation, we propose a novel generalization method, coined Parameter-Space Ensemble with Entropy Regularization (PEER), that uses a proxy model to learn the augmented data on behalf of the main model. The main model is updated by averaging its parameters with the proxy model, progressively accumulating knowledge over the training steps. Maximizing the mutual information between the output representations of the two models guides the learning process of the proxy model, mitigating feature distortion during training. Experimental results demonstrate the effectiveness of PEER in reducing the OOD performance fluctuation and enhancing generalization across various datasets, including PACS, Digits, Office-Home, and VLCS. Notably, our method with simple random augmentation achieves state-of-the-art performance, surpassing prior approaches on sDG that utilize complex data augmentation strategies. 

---
# EpiLLM: Unlocking the Potential of Large Language Models in Epidemic Forecasting 

**Authors**: Chenghua Gong, Rui Sun, Yuhao Zheng, Juyuan Zhang, Tianjun Gu, Liming Pan, Linyuan Lv  

**Link**: [PDF](https://arxiv.org/pdf/2505.12738)  

**Abstract**: Advanced epidemic forecasting is critical for enabling precision containment strategies, highlighting its strategic importance for public health security. While recent advances in Large Language Models (LLMs) have demonstrated effectiveness as foundation models for domain-specific tasks, their potential for epidemic forecasting remains largely unexplored. In this paper, we introduce EpiLLM, a novel LLM-based framework tailored for spatio-temporal epidemic forecasting. Considering the key factors in real-world epidemic transmission: infection cases and human mobility, we introduce a dual-branch architecture to achieve fine-grained token-level alignment between such complex epidemic patterns and language tokens for LLM adaptation. To unleash the multi-step forecasting and generalization potential of LLM architectures, we propose an autoregressive modeling paradigm that reformulates the epidemic forecasting task into next-token prediction. To further enhance LLM perception of epidemics, we introduce spatio-temporal prompt learning techniques, which strengthen forecasting capabilities from a data-driven perspective. Extensive experiments show that EpiLLM significantly outperforms existing baselines on real-world COVID-19 datasets and exhibits scaling behavior characteristic of LLMs. 

---
# Option-aware Temporally Abstracted Value for Offline Goal-Conditioned Reinforcement Learning 

**Authors**: Hongjoon Ahn, Heewoong Choi, Jisu Han, Taesup Moon  

**Link**: [PDF](https://arxiv.org/pdf/2505.12737)  

**Abstract**: Offline goal-conditioned reinforcement learning (GCRL) offers a practical learning paradigm where goal-reaching policies are trained from abundant unlabeled (reward-free) datasets without additional environment interaction. However, offline GCRL still struggles with long-horizon tasks, even with recent advances that employ hierarchical policy structures, such as HIQL. By identifying the root cause of this challenge, we observe the following insights: First, performance bottlenecks mainly stem from the high-level policy's inability to generate appropriate subgoals. Second, when learning the high-level policy in the long-horizon regime, the sign of the advantage signal frequently becomes incorrect. Thus, we argue that improving the value function to produce a clear advantage signal for learning the high-level policy is essential. In this paper, we propose a simple yet effective solution: Option-aware Temporally Abstracted value learning, dubbed OTA, which incorporates temporal abstraction into the temporal-difference learning process. By modifying the value update to be option-aware, the proposed learning scheme contracts the effective horizon length, enabling better advantage estimates even in long-horizon regimes. We experimentally show that the high-level policy extracted using the OTA value function achieves strong performance on complex tasks from OGBench, a recently proposed offline GCRL benchmark, including maze navigation and visual robotic manipulation environments. 

---
# SounDiT: Geo-Contextual Soundscape-to-Landscape Generation 

**Authors**: Junbo Wang, Haofeng Tan, Bowen Liao, Albert Jiang, Teng Fei, Qixing Huang, Zhengzhong Tu, Shan Ye, Yuhao Kang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12734)  

**Abstract**: We present a novel and practically significant problem-Geo-Contextual Soundscape-to-Landscape (GeoS2L) generation-which aims to synthesize geographically realistic landscape images from environmental soundscapes. Prior audio-to-image generation methods typically rely on general-purpose datasets and overlook geographic and environmental contexts, resulting in unrealistic images that are misaligned with real-world environmental settings. To address this limitation, we introduce a novel geo-contextual computational framework that explicitly integrates geographic knowledge into multimodal generative modeling. We construct two large-scale geo-contextual multimodal datasets, SoundingSVI and SonicUrban, pairing diverse soundscapes with real-world landscape images. We propose SounDiT, a novel Diffusion Transformer (DiT)-based model that incorporates geo-contextual scene conditioning to synthesize geographically coherent landscape images. Furthermore, we propose a practically-informed geo-contextual evaluation framework, the Place Similarity Score (PSS), across element-, scene-, and human perception-levels to measure consistency between input soundscapes and generated landscape images. Extensive experiments demonstrate that SounDiT outperforms existing baselines in both visual fidelity and geographic settings. Our work not only establishes foundational benchmarks for GeoS2L generation but also highlights the importance of incorporating geographic domain knowledge in advancing multimodal generative models, opening new directions at the intersection of generative AI, geography, urban planning, and environmental sciences. 

---
# Shadow-FT: Tuning Instruct via Base 

**Authors**: Taiqiang Wu, Runming Yang, Jiayi Li, Pengfei Hu, Ngai Wong, Yujiu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12716)  

**Abstract**: Large language models (LLMs) consistently benefit from further fine-tuning on various tasks. However, we observe that directly tuning the INSTRUCT (i.e., instruction tuned) models often leads to marginal improvements and even performance degeneration. Notably, paired BASE models, the foundation for these INSTRUCT variants, contain highly similar weight values (i.e., less than 2% on average for Llama 3.1 8B). Therefore, we propose a novel Shadow-FT framework to tune the INSTRUCT models by leveraging the corresponding BASE models. The key insight is to fine-tune the BASE model, and then directly graft the learned weight updates to the INSTRUCT model. Our proposed Shadow-FT introduces no additional parameters, is easy to implement, and significantly improves performance. We conduct extensive experiments on tuning mainstream LLMs, such as Qwen 3 and Llama 3 series, and evaluate them across 19 benchmarks covering coding, reasoning, and mathematical tasks. Experimental results demonstrate that Shadow-FT consistently outperforms conventional full-parameter and parameter-efficient tuning approaches. Further analyses indicate that Shadow-FT can be applied to multimodal large language models (MLLMs) and combined with direct preference optimization (DPO). Codes and weights are available at \href{this https URL}{Github}. 

---
# Any-to-Any Learning in Computational Pathology via Triplet Multimodal Pretraining 

**Authors**: Qichen Sun, Zhengrui Guo, Rui Peng, Hao Chen, Jinzhuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12711)  

**Abstract**: Recent advances in computational pathology and artificial intelligence have significantly enhanced the utilization of gigapixel whole-slide images and and additional modalities (e.g., genomics) for pathological diagnosis. Although deep learning has demonstrated strong potential in pathology, several key challenges persist: (1) fusing heterogeneous data types requires sophisticated strategies beyond simple concatenation due to high computational costs; (2) common scenarios of missing modalities necessitate flexible strategies that allow the model to learn robustly in the absence of certain modalities; (3) the downstream tasks in CPath are diverse, ranging from unimodal to multimodal, cnecessitating a unified model capable of handling all modalities. To address these challenges, we propose ALTER, an any-to-any tri-modal pretraining framework that integrates WSIs, genomics, and pathology reports. The term "any" emphasizes ALTER's modality-adaptive design, enabling flexible pretraining with any subset of modalities, and its capacity to learn robust, cross-modal representations beyond WSI-centric approaches. We evaluate ALTER across extensive clinical tasks including survival prediction, cancer subtyping, gene mutation prediction, and report generation, achieving superior or comparable performance to state-of-the-art baselines. 

---
# PLAICraft: Large-Scale Time-Aligned Vision-Speech-Action Dataset for Embodied AI 

**Authors**: Yingchen He, Christian D. Weilbach, Martyna E. Wojciechowska, Yuxuan Zhang, Frank Wood  

**Link**: [PDF](https://arxiv.org/pdf/2505.12707)  

**Abstract**: Advances in deep generative modelling have made it increasingly plausible to train human-level embodied agents. Yet progress has been limited by the absence of large-scale, real-time, multi-modal, and socially interactive datasets that reflect the sensory-motor complexity of natural environments. To address this, we present PLAICraft, a novel data collection platform and dataset capturing multiplayer Minecraft interactions across five time-aligned modalities: video, game output audio, microphone input audio, mouse, and keyboard actions. Each modality is logged with millisecond time precision, enabling the study of synchronous, embodied behaviour in a rich, open-ended world. The dataset comprises over 10,000 hours of gameplay from more than 10,000 global participants.\footnote{We have done a privacy review for the public release of an initial 200-hour subset of the dataset, with plans to release most of the dataset over time.} Alongside the dataset, we provide an evaluation suite for benchmarking model capabilities in object recognition, spatial awareness, language grounding, and long-term memory. PLAICraft opens a path toward training and evaluating agents that act fluently and purposefully in real time, paving the way for truly embodied artificial intelligence. 

---
# DreamGen: Unlocking Generalization in Robot Learning through Neural Trajectories 

**Authors**: Joel Jang, Seonghyeon Ye, Zongyu Lin, Jiannan Xiang, Johan Bjorck, Yu Fang, Fengyuan Hu, Spencer Huang, Kaushil Kundalia, Yen-Chen Lin, Loic Magne, Ajay Mandlekar, Avnish Narayan, You Liang Tan, Guanzhi Wang, Jing Wang, Qi Wang, Yinzhen Xu, Xiaohui Zeng, Kaiyuan Zheng, Ruijie Zheng, Ming-Yu Liu, Luke Zettlemoyer, Dieter Fox, Jan Kautz, Scott Reed, Yuke Zhu, Linxi Fan  

**Link**: [PDF](https://arxiv.org/pdf/2505.12705)  

**Abstract**: We introduce DreamGen, a simple yet highly effective 4-stage pipeline for training robot policies that generalize across behaviors and environments through neural trajectories - synthetic robot data generated from video world models. DreamGen leverages state-of-the-art image-to-video generative models, adapting them to the target robot embodiment to produce photorealistic synthetic videos of familiar or novel tasks in diverse environments. Since these models generate only videos, we recover pseudo-action sequences using either a latent action model or an inverse-dynamics model (IDM). Despite its simplicity, DreamGen unlocks strong behavior and environment generalization: a humanoid robot can perform 22 new behaviors in both seen and unseen environments, while requiring teleoperation data from only a single pick-and-place task in one environment. To evaluate the pipeline systematically, we introduce DreamGen Bench, a video generation benchmark that shows a strong correlation between benchmark performance and downstream policy success. Our work establishes a promising new axis for scaling robot learning well beyond manual data collection. 

---
# Counterfactual Explanations for Continuous Action Reinforcement Learning 

**Authors**: Shuyang Dong, Shangtong Zhang, Lu Feng  

**Link**: [PDF](https://arxiv.org/pdf/2505.12701)  

**Abstract**: Reinforcement Learning (RL) has shown great promise in domains like healthcare and robotics but often struggles with adoption due to its lack of interpretability. Counterfactual explanations, which address "what if" scenarios, provide a promising avenue for understanding RL decisions but remain underexplored for continuous action spaces. We propose a novel approach for generating counterfactual explanations in continuous action RL by computing alternative action sequences that improve outcomes while minimizing deviations from the original sequence. Our approach leverages a distance metric for continuous actions and accounts for constraints such as adhering to predefined policies in specific states. Evaluations in two RL domains, Diabetes Control and Lunar Lander, demonstrate the effectiveness, efficiency, and generalization of our approach, enabling more interpretable and trustworthy RL applications. 

---
# Towards Effective Federated Graph Foundation Model via Mitigating Knowledge Entanglement 

**Authors**: Yinlin Zhu, Xunkai Li, Jishuo Jia, Miao Hu, Di Wu, Meikang Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12684)  

**Abstract**: Recent advances in graph machine learning have shifted to data-centric paradigms, driven by two emerging fields: (1) Federated graph learning (FGL) enables multi-client collaboration but faces challenges from data and task heterogeneity, limiting its practicality; (2) Graph foundation models (GFM) offer strong domain generalization but are usually trained on single machines, missing out on cross-silo data and resources.
These paradigms are complementary, and their integration brings notable benefits. Motivated by this, we propose FedGFM, a novel decentralized GFM training paradigm. However, a key challenge is knowledge entanglement, where multi-domain knowledge merges into indistinguishable representations, hindering downstream adaptation.
To address this, we present FedGFM+, an enhanced framework with two core modules to reduce knowledge entanglement: (1) AncDAI: A global anchor-based domain-aware initialization strategy. Before pre-training, each client encodes its local graph into domain-specific prototypes that serve as semantic anchors. Synthetic embeddings around these anchors initialize the global model. We theoretically prove these prototypes are distinguishable across domains, providing a strong inductive bias to disentangle domain-specific knowledge. (2) AdaDPP: A local adaptive domain-sensitive prompt pool. Each client learns a lightweight graph prompt capturing domain semantics during pre-training. During fine-tuning, prompts from all clients form a pool from which the GFM selects relevant prompts to augment target graph attributes, improving downstream adaptation.
FedGFM+ is evaluated on 8 diverse benchmarks across multiple domains and tasks, outperforming 20 baselines from supervised learning, FGL, and federated GFM variants. 

---
# Text2midi-InferAlign: Improving Symbolic Music Generation with Inference-Time Alignment 

**Authors**: Abhinaba Roy, Geeta Puri, Dorien Herremans  

**Link**: [PDF](https://arxiv.org/pdf/2505.12669)  

**Abstract**: We present Text2midi-InferAlign, a novel technique for improving symbolic music generation at inference time. Our method leverages text-to-audio alignment and music structural alignment rewards during inference to encourage the generated music to be consistent with the input caption. Specifically, we introduce two objectives scores: a text-audio consistency score that measures rhythmic alignment between the generated music and the original text caption, and a harmonic consistency score that penalizes generated music containing notes inconsistent with the key. By optimizing these alignment-based objectives during the generation process, our model produces symbolic music that is more closely tied to the input captions, thereby improving the overall quality and coherence of the generated compositions. Our approach can extend any existing autoregressive model without requiring further training or fine-tuning. We evaluate our work on top of Text2midi - an existing text-to-midi generation model, demonstrating significant improvements in both objective and subjective evaluation metrics. 

---
# Multi-View Wireless Sensing via Conditional Generative Learning: Framework and Model Design 

**Authors**: Ziqing Xing, Zhaoyang Zhang, Zirui Chen, Hongning Ruan, Zhaohui Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12664)  

**Abstract**: In this paper, we incorporate physical knowledge into learning-based high-precision target sensing using the multi-view channel state information (CSI) between multiple base stations (BSs) and user equipment (UEs). Such kind of multi-view sensing problem can be naturally cast into a conditional generation framework. To this end, we design a bipartite neural network architecture, the first part of which uses an elaborately designed encoder to fuse the latent target features embedded in the multi-view CSI, and then the second uses them as conditioning inputs of a powerful generative model to guide the target's reconstruction. Specifically, the encoder is designed to capture the physical correlation between the CSI and the target, and also be adaptive to the numbers and positions of BS-UE pairs. Therein the view-specific nature of CSI is assimilated by introducing a spatial positional embedding scheme, which exploits the structure of electromagnetic(EM)-wave propagation channels. Finally, a conditional diffusion model with a weighted loss is employed to generate the target's point cloud from the fused features. Extensive numerical results demonstrate that the proposed generative multi-view (Gen-MV) sensing framework exhibits excellent flexibility and significant performance improvement on the reconstruction quality of target's shape and EM properties. 

---
# Know3-RAG: A Knowledge-aware RAG Framework with Adaptive Retrieval, Generation, and Filtering 

**Authors**: Xukai Liu, Ye Liu, Shiwen Wu, Yanghai Zhang, Yihao Yuan, Kai Zhang, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12662)  

**Abstract**: Recent advances in large language models (LLMs) have led to impressive progress in natural language generation, yet their tendency to produce hallucinated or unsubstantiated content remains a critical concern. To improve factual reliability, Retrieval-Augmented Generation (RAG) integrates external knowledge during inference. However, existing RAG systems face two major limitations: (1) unreliable adaptive control due to limited external knowledge supervision, and (2) hallucinations caused by inaccurate or irrelevant references. To address these issues, we propose Know3-RAG, a knowledge-aware RAG framework that leverages structured knowledge from knowledge graphs (KGs) to guide three core stages of the RAG process, including retrieval, generation, and filtering. Specifically, we introduce a knowledge-aware adaptive retrieval module that employs KG embedding to assess the confidence of the generated answer and determine retrieval necessity, a knowledge-enhanced reference generation strategy that enriches queries with KG-derived entities to improve generated reference relevance, and a knowledge-driven reference filtering mechanism that ensures semantic alignment and factual accuracy of references. Experiments on multiple open-domain QA benchmarks demonstrate that Know3-RAG consistently outperforms strong baselines, significantly reducing hallucinations and enhancing answer reliability. 

---
# Web IP at Risk: Prevent Unauthorized Real-Time Retrieval by Large Language Models 

**Authors**: Yisheng Zhong, Yizhu Wen, Junfeng Guo, Mehran Kafai, Heng Huang, Hanqing Guo, Zhuangdi Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12655)  

**Abstract**: Protecting cyber Intellectual Property (IP) such as web content is an increasingly critical concern. The rise of large language models (LLMs) with online retrieval capabilities presents a double-edged sword that enables convenient access to information but often undermines the rights of original content creators. As users increasingly rely on LLM-generated responses, they gradually diminish direct engagement with original information sources, significantly reducing the incentives for IP creators to contribute, and leading to a saturating cyberspace with more AI-generated content. In response, we propose a novel defense framework that empowers web content creators to safeguard their web-based IP from unauthorized LLM real-time extraction by leveraging the semantic understanding capability of LLMs themselves. Our method follows principled motivations and effectively addresses an intractable black-box optimization problem. Real-world experiments demonstrated that our methods improve defense success rates from 2.5% to 88.6% on different LLMs, outperforming traditional defenses such as configuration-based restrictions. 

---
# Predicting Turn-Taking and Backchannel in Human-Machine Conversations Using Linguistic, Acoustic, and Visual Signals 

**Authors**: Yuxin Lin, Yinglin Zheng, Ming Zeng, Wangzheng Shi  

**Link**: [PDF](https://arxiv.org/pdf/2505.12654)  

**Abstract**: This paper addresses the gap in predicting turn-taking and backchannel actions in human-machine conversations using multi-modal signals (linguistic, acoustic, and visual). To overcome the limitation of existing datasets, we propose an automatic data collection pipeline that allows us to collect and annotate over 210 hours of human conversation videos. From this, we construct a Multi-Modal Face-to-Face (MM-F2F) human conversation dataset, including over 1.5M words and corresponding turn-taking and backchannel annotations from approximately 20M frames. Additionally, we present an end-to-end framework that predicts the probability of turn-taking and backchannel actions from multi-modal signals. The proposed model emphasizes the interrelation between modalities and supports any combination of text, audio, and video inputs, making it adaptable to a variety of realistic scenarios. Our experiments show that our approach achieves state-of-the-art performance on turn-taking and backchannel prediction tasks, achieving a 10\% increase in F1-score on turn-taking and a 33\% increase on backchannel prediction. Our dataset and code are publicly available online to ease of subsequent research. 

---
# AutoMat: Enabling Automated Crystal Structure Reconstruction from Microscopy via Agentic Tool Use 

**Authors**: Yaotian Yang, Yiwen Tang, Yizhe Chen, Xiao Chen, Jiangjie Qiu, Hao Xiong, Haoyu Yin, Zhiyao Luo, Yifei Zhang, Sijia Tao, Wentao Li, Qinghua Zhang, Yuqiang Li, Wanli Ouyang, Bin Zhao, Xiaonan Wang, Fei Wei  

**Link**: [PDF](https://arxiv.org/pdf/2505.12650)  

**Abstract**: Machine learning-based interatomic potentials and force fields depend critically on accurate atomic structures, yet such data are scarce due to the limited availability of experimentally resolved crystals. Although atomic-resolution electron microscopy offers a potential source of structural data, converting these images into simulation-ready formats remains labor-intensive and error-prone, creating a bottleneck for model training and validation. We introduce AutoMat, an end-to-end, agent-assisted pipeline that automatically transforms scanning transmission electron microscopy (STEM) images into atomic crystal structures and predicts their physical properties. AutoMat combines pattern-adaptive denoising, physics-guided template retrieval, symmetry-aware atomic reconstruction, fast relaxation and property prediction via MatterSim, and coordinated orchestration across all stages. We propose the first dedicated STEM2Mat-Bench for this task and evaluate performance using lattice RMSD, formation energy MAE, and structure-matching success rate. By orchestrating external tool calls, AutoMat enables a text-only LLM to outperform vision-language models in this domain, achieving closed-loop reasoning throughout the pipeline. In large-scale experiments over 450 structure samples, AutoMat substantially outperforms existing multimodal large language models and tools. These results validate both AutoMat and STEM2Mat-Bench, marking a key step toward bridging microscopy and atomistic simulation in materials this http URL code and dataset are publicly available at this https URL and this https URL. 

---
# Single Image Reflection Removal via inter-layer Complementarity 

**Authors**: Yue Huang, Zi'ang Li, Tianle Hu, Jie Wen, Guanbin Li, Jinglin Zhang, Guoxu Zhou, Xiaozhao Fang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12641)  

**Abstract**: Although dual-stream architectures have achieved remarkable success in single image reflection removal, they fail to fully exploit inter-layer complementarity in their physical modeling and network design, which limits the quality of image separation. To address this fundamental limitation, we propose two targeted improvements to enhance dual-stream architectures: First, we introduce a novel inter-layer complementarity model where low-frequency components extracted from the residual layer interact with the transmission layer through dual-stream architecture to enhance inter-layer complementarity. Meanwhile, high-frequency components from the residual layer provide inverse modulation to both streams, improving the detail quality of the transmission layer. Second, we propose an efficient inter-layer complementarity attention mechanism which first cross-reorganizes dual streams at the channel level to obtain reorganized streams with inter-layer complementary structures, then performs attention computation on the reorganized streams to achieve better inter-layer separation, and finally restores the original stream structure for output. Experimental results demonstrate that our method achieves state-of-the-art separation quality on multiple public datasets while significantly reducing both computational cost and model complexity. 

---
# ChromFound: Towards A Universal Foundation Model for Single-Cell Chromatin Accessibility Data 

**Authors**: Yifeng Jiao, Yuchen Liu, Yu Zhang, Xin Guo, Yushuai Wu, Chen Jiang, Jiyang Li, Hongwei Zhang, Limei Han, Xin Gao, Yuan Qi, Yuan Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.12638)  

**Abstract**: The advent of single-cell Assay for Transposase-Accessible Chromatin using sequencing (scATAC-seq) offers an innovative perspective for deciphering regulatory mechanisms by assembling a vast repository of single-cell chromatin accessibility data. While foundation models have achieved significant success in single-cell transcriptomics, there is currently no foundation model for scATAC-seq that supports zero-shot high-quality cell identification and comprehensive multi-omics analysis simultaneously. Key challenges lie in the high dimensionality and sparsity of scATAC-seq data, as well as the lack of a standardized schema for representing open chromatin regions (OCRs). Here, we present \textbf{ChromFound}, a foundation model tailored for scATAC-seq. ChromFound utilizes a hybrid architecture and genome-aware tokenization to effectively capture genome-wide long contexts and regulatory signals from dynamic chromatin landscapes. Pretrained on 1.97 million cells from 30 tissues and 6 disease conditions, ChromFound demonstrates broad applicability across 6 diverse tasks. Notably, it achieves robust zero-shot performance in generating universal cell representations and exhibits excellent transferability in cell type annotation and cross-omics prediction. By uncovering enhancer-gene links undetected by existing computational methods, ChromFound offers a promising framework for understanding disease risk variants in the noncoding genome. 

---
# Scalable Video-to-Dataset Generation for Cross-Platform Mobile Agents 

**Authors**: Yunseok Jang, Yeda Song, Sungryull Sohn, Lajanugen Logeswaran, Tiange Luo, Dong-Ki Kim, Kyunghoon Bae, Honglak Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.12632)  

**Abstract**: Recent advancements in Large Language Models (LLMs) and Vision-Language Models (VLMs) have sparked significant interest in developing GUI visual agents. We introduce MONDAY (Mobile OS Navigation Task Dataset for Agents from YouTube), a large-scale dataset of 313K annotated frames from 20K instructional videos capturing diverse real-world mobile OS navigation across multiple platforms. Models that include MONDAY in their pre-training phases demonstrate robust cross-platform generalization capabilities, consistently outperforming models trained on existing single OS datasets while achieving an average performance gain of 18.11%p on an unseen mobile OS platform. To enable continuous dataset expansion as mobile platforms evolve, we present an automated framework that leverages publicly available video content to create comprehensive task datasets without manual annotation. Our framework comprises robust OCR-based scene detection (95.04% F1score), near-perfect UI element detection (99.87% hit ratio), and novel multi-step action identification to extract reliable action sequences across diverse interface configurations. We contribute both the MONDAY dataset and our automated collection framework to facilitate future research in mobile OS navigation. 

---
# Degradation-Aware Feature Perturbation for All-in-One Image Restoration 

**Authors**: Xiangpeng Tian, Xiangyu Liao, Xiao Liu, Meng Li, Chao Ren  

**Link**: [PDF](https://arxiv.org/pdf/2505.12630)  

**Abstract**: All-in-one image restoration aims to recover clear images from various degradation types and levels with a unified model. Nonetheless, the significant variations among degradation types present challenges for training a universal model, often resulting in task interference, where the gradient update directions of different tasks may diverge due to shared parameters. To address this issue, motivated by the routing strategy, we propose DFPIR, a novel all-in-one image restorer that introduces Degradation-aware Feature Perturbations(DFP) to adjust the feature space to align with the unified parameter space. In this paper, the feature perturbations primarily include channel-wise perturbations and attention-wise perturbations. Specifically, channel-wise perturbations are implemented by shuffling the channels in high-dimensional space guided by degradation types, while attention-wise perturbations are achieved through selective masking in the attention space. To achieve these goals, we propose a Degradation-Guided Perturbation Block (DGPB) to implement these two functions, positioned between the encoding and decoding stages of the encoder-decoder architecture. Extensive experimental results demonstrate that DFPIR achieves state-of-the-art performance on several all-in-one image restoration tasks including image denoising, image dehazing, image deraining, motion deblurring, and low-light image enhancement. Our codes are available at this https URL. 

---
# scSiameseClu: A Siamese Clustering Framework for Interpreting single-cell RNA Sequencing Data 

**Authors**: Ping Xu, Zhiyuan Ning, Pengjiang Li, Wenhao Liu, Pengyang Wang, Jiaxu Cui, Yuanchun Zhou, Pengfei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12626)  

**Abstract**: Single-cell RNA sequencing (scRNA-seq) reveals cell heterogeneity, with cell clustering playing a key role in identifying cell types and marker genes. Recent advances, especially graph neural networks (GNNs)-based methods, have significantly improved clustering performance. However, the analysis of scRNA-seq data remains challenging due to noise, sparsity, and high dimensionality. Compounding these challenges, GNNs often suffer from over-smoothing, limiting their ability to capture complex biological information. In response, we propose scSiameseClu, a novel Siamese Clustering framework for interpreting single-cell RNA-seq data, comprising of 3 key steps: (1) Dual Augmentation Module, which applies biologically informed perturbations to the gene expression matrix and cell graph relationships to enhance representation robustness; (2) Siamese Fusion Module, which combines cross-correlation refinement and adaptive information fusion to capture complex cellular relationships while mitigating over-smoothing; and (3) Optimal Transport Clustering, which utilizes Sinkhorn distance to efficiently align cluster assignments with predefined proportions while maintaining balance. Comprehensive evaluations on seven real-world datasets demonstrate that~\methodname~outperforms state-of-the-art methods in single-cell clustering, cell type annotation, and cell type classification, providing a powerful tool for scRNA-seq data interpretation. 

---
# Lightweight and Effective Preference Construction in PIBT for Large-Scale Multi-Agent Pathfinding 

**Authors**: Keisuke Okumura, Hiroki Nagai  

**Link**: [PDF](https://arxiv.org/pdf/2505.12623)  

**Abstract**: PIBT is a computationally lightweight algorithm that can be applied to a variety of multi-agent pathfinding (MAPF) problems, generating the next collision-free locations of agents given another. Because of its simplicity and scalability, it is becoming a popular underlying scheme for recent large-scale MAPF methods involving several hundreds or thousands of agents. Vanilla PIBT makes agents behave greedily towards their assigned goals, while agents typically have multiple best actions, since the graph shortest path is not always unique. Consequently, tiebreaking about how to choose between these actions significantly affects resulting solutions. This paper studies two simple yet effective techniques for tiebreaking in PIBT, without compromising its computational advantage. The first technique allows an agent to intelligently dodge another, taking into account whether each action will hinder the progress of the next timestep. The second technique is to learn, through multiple PIBT runs, how an action causes regret in others and to use this information to minimise regret collectively. Our empirical results demonstrate that these techniques can reduce the solution cost of one-shot MAPF and improve the throughput of lifelong MAPF. For instance, in densely populated one-shot cases, the combined use of these tiebreaks achieves improvements of around 10-20% in sum-of-costs, without significantly compromising the speed of a PIBT-based planner. 

---
# AD-AGENT: A Multi-agent Framework for End-to-end Anomaly Detection 

**Authors**: Tiankai Yang, Junjun Liu, Wingchun Siu, Jiahang Wang, Zhuangzhuang Qian, Chanjuan Song, Cheng Cheng, Xiyang Hu, Yue Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.12594)  

**Abstract**: Anomaly detection (AD) is essential in areas such as fraud detection, network monitoring, and scientific research. However, the diversity of data modalities and the increasing number of specialized AD libraries pose challenges for non-expert users who lack in-depth library-specific knowledge and advanced programming skills. To tackle this, we present AD-AGENT, an LLM-driven multi-agent framework that turns natural-language instructions into fully executable AD pipelines. AD-AGENT coordinates specialized agents for intent parsing, data preparation, library and model selection, documentation mining, and iterative code generation and debugging. Using a shared short-term workspace and a long-term cache, the agents integrate popular AD libraries like PyOD, PyGOD, and TSLib into a unified workflow. Experiments demonstrate that AD-AGENT produces reliable scripts and recommends competitive models across libraries. The system is open-sourced to support further research and practical applications in AD. 

---
# Learning Robust Spectral Dynamics for Temporal Domain Generalization 

**Authors**: En Yu, Jie Lu, Xiaoyu Yang, Guangquan Zhang, Zhen Fang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12585)  

**Abstract**: Modern machine learning models struggle to maintain performance in dynamic environments where temporal distribution shifts, \emph{i.e., concept drift}, are prevalent. Temporal Domain Generalization (TDG) seeks to enable model generalization across evolving domains, yet existing approaches typically assume smooth incremental changes, struggling with complex real-world drifts involving long-term structure (incremental evolution/periodicity) and local uncertainties. To overcome these limitations, we introduce FreKoo, which tackles these challenges via a novel frequency-domain analysis of parameter trajectories. It leverages the Fourier transform to disentangle parameter evolution into distinct spectral bands. Specifically, low-frequency component with dominant dynamics are learned and extrapolated using the Koopman operator, robustly capturing diverse drift patterns including both incremental and periodicity. Simultaneously, potentially disruptive high-frequency variations are smoothed via targeted temporal regularization, preventing overfitting to transient noise and domain uncertainties. In addition, this dual spectral strategy is rigorously grounded through theoretical analysis, providing stability guarantees for the Koopman prediction, a principled Bayesian justification for the high-frequency regularization, and culminating in a multiscale generalization bound connecting spectral dynamics to improved generalization. Extensive experiments demonstrate FreKoo's significant superiority over SOTA TDG approaches, particularly excelling in real-world streaming scenarios with complex drifts and uncertainties. 

---
# A Comprehensive Survey on Physical Risk Control in the Era of Foundation Model-enabled Robotics 

**Authors**: Takeshi Kojima, Yaonan Zhu, Yusuke Iwasawa, Toshinori Kitamura, Gang Yan, Shu Morikuni, Ryosuke Takanami, Alfredo Solano, Tatsuya Matsushima, Akiko Murakami, Yutaka Matsuo  

**Link**: [PDF](https://arxiv.org/pdf/2505.12583)  

**Abstract**: Recent Foundation Model-enabled robotics (FMRs) display greatly improved general-purpose skills, enabling more adaptable automation than conventional robotics. Their ability to handle diverse tasks thus creates new opportunities to replace human labor. However, unlike general foundation models, FMRs interact with the physical world, where their actions directly affect the safety of humans and surrounding objects, requiring careful deployment and control. Based on this proposition, our survey comprehensively summarizes robot control approaches to mitigate physical risks by covering all the lifespan of FMRs ranging from pre-deployment to post-accident stage. Specifically, we broadly divide the timeline into the following three phases: (1) pre-deployment phase, (2) pre-incident phase, and (3) post-incident phase. Throughout this survey, we find that there is much room to study (i) pre-incident risk mitigation strategies, (ii) research that assumes physical interaction with humans, and (iii) essential issues of foundation models themselves. We hope that this survey will be a milestone in providing a high-resolution analysis of the physical risks of FMRs and their control, contributing to the realization of a good human-robot relationship. 

---
# An approach based on class activation maps for investigating the effects of data augmentation on neural networks for image classification 

**Authors**: Lucas M. Dorneles, Luan Fonseca Garcia, Joel Luís Carbonera  

**Link**: [PDF](https://arxiv.org/pdf/2505.12581)  

**Abstract**: Neural networks have become increasingly popular in the last few years as an effective tool for the task of image classification due to the impressive performance they have achieved on this task. In image classification tasks, it is common to use data augmentation strategies to increase the robustness of trained networks to changes in the input images and to avoid overfitting. Although data augmentation is a widely adopted technique, the literature lacks a body of research analyzing the effects data augmentation methods have on the patterns learned by neural network models working on complex datasets. The primary objective of this work is to propose a methodology and set of metrics that may allow a quantitative approach to analyzing the effects of data augmentation in convolutional networks applied to image classification. An important tool used in the proposed approach lies in the concept of class activation maps for said models, which allow us to identify and measure the importance these models assign to each individual pixel in an image when executing the classification task. From these maps, we may then extract metrics over the similarities and differences between maps generated by these models trained on a given dataset with different data augmentation strategies. Experiments made using this methodology suggest that the effects of these data augmentation techniques not only can be analyzed in this way but also allow us to identify different impact profiles over the trained models. 

---
# AdaDim: Dimensionality Adaptation for SSL Representational Dynamics 

**Authors**: Kiran Kokilepersaud, Mohit Prabhushankar, Ghassan AlRegib  

**Link**: [PDF](https://arxiv.org/pdf/2505.12576)  

**Abstract**: A key factor in effective Self-Supervised learning (SSL) is preventing dimensional collapse, which is where higher-dimensional representation spaces span a lower-dimensional subspace. Therefore, SSL optimization strategies involve guiding a model to produce representations ($R$) with a higher dimensionality. Dimensionality is either optimized through a dimension-contrastive approach that encourages feature decorrelation or through a sample-contrastive method that promotes a uniform spread of sample representations. Both families of SSL algorithms also utilize a projection head that maps $R$ into a lower-dimensional embedding space $Z$. Recent work has characterized the projection head as a filter of irrelevant features from the SSL objective by reducing mutual information, $I(R;Z)$. Therefore, the current literature's view is that a good SSL representation space should have a high $H(R)$ and a low $I(R;Z)$. However, this view of the problem is lacking in terms of an understanding of the underlying training dynamics that influences both terms, as well as how the values of $H(R)$ and $I(R;Z)$ arrived at the end of training reflect the downstream performance of an SSL model. We address both gaps in the literature by demonstrating that increases in $H(R)$ due to feature decorrelation at the start of training lead to a higher $I(R;Z)$, while increases in $H(R)$ due to samples distributing uniformly in a high-dimensional space at the end of training cause $I(R;Z)$ to plateau or decrease. Furthermore, our analysis shows that the best performing SSL models do not have the highest $H(R)$ nor the lowest $I(R;Z)$, but arrive at an optimal intermediate point for both. We develop a method called AdaDim to exploit these observed training dynamics by adaptively weighting between losses based on feature decorrelation and uniform sample spread. 

---
# Measuring Information Distortion in Hierarchical Ultra long Novel Generation:The Optimal Expansion Ratio 

**Authors**: Hanwen Shen, Ting Ying  

**Link**: [PDF](https://arxiv.org/pdf/2505.12572)  

**Abstract**: Writing novels with Large Language Models (LLMs) raises a critical question: how much human-authored outline is necessary to generate high-quality million-word novels? While frameworks such as DOME, Plan&Write, and Long Writer have improved stylistic coherence and logical consistency, they primarily target shorter novels (10k--100k words), leaving ultra-long generation largely unexplored. Drawing on insights from recent text compression methods like LLMZip and LLM2Vec, we conduct an information-theoretic analysis that quantifies distortion occurring when LLMs compress and reconstruct ultra-long novels under varying compression-expansion ratios. We introduce a hierarchical two-stage generation pipeline (outline -> detailed outline -> manuscript) and find an optimal outline length that balances information preservation with human effort. Through extensive experimentation with Chinese novels, we establish that a two-stage hierarchical outline approach significantly reduces semantic distortion compared to single-stage methods. Our findings provide empirically-grounded guidance for authors and researchers collaborating with LLMs to create million-word novels. 

---
# A Survey of Attacks on Large Language Models 

**Authors**: Wenrui Xu, Keshab K. Parhi  

**Link**: [PDF](https://arxiv.org/pdf/2505.12567)  

**Abstract**: Large language models (LLMs) and LLM-based agents have been widely deployed in a wide range of applications in the real world, including healthcare diagnostics, financial analysis, customer support, robotics, and autonomous driving, expanding their powerful capability of understanding, reasoning, and generating natural languages. However, the wide deployment of LLM-based applications exposes critical security and reliability risks, such as the potential for malicious misuse, privacy leakage, and service disruption that weaken user trust and undermine societal safety. This paper provides a systematic overview of the details of adversarial attacks targeting both LLMs and LLM-based agents. These attacks are organized into three phases in LLMs: Training-Phase Attacks, Inference-Phase Attacks, and Availability & Integrity Attacks. For each phase, we analyze the details of representative and recently introduced attack methods along with their corresponding defenses. We hope our survey will provide a good tutorial and a comprehensive understanding of LLM security, especially for attacks on LLMs. We desire to raise attention to the risks inherent in widely deployed LLM-based applications and highlight the urgent need for robust mitigation strategies for evolving threats. 

---
# Beyond Accuracy: EcoL2 Metric for Sustainable Neural PDE Solvers 

**Authors**: Taniya Kapoor, Abhishek Chandra, Anastasios Stamou, Stephen J Roberts  

**Link**: [PDF](https://arxiv.org/pdf/2505.12556)  

**Abstract**: Real-world systems, from aerospace to railway engineering, are modeled with partial differential equations (PDEs) describing the physics of the system. Estimating robust solutions for such problems is essential. Deep learning-based architectures, such as neural PDE solvers, have recently gained traction as a reliable solution method. The current state of development of these approaches, however, primarily focuses on improving accuracy. The environmental impact of excessive computation, leading to increased carbon emissions, has largely been overlooked. This paper introduces a carbon emission measure for a range of PDE solvers. Our proposed metric, EcoL2, balances model accuracy with emissions across data collection, model training, and deployment. Experiments across both physics-informed machine learning and operator learning architectures demonstrate that the proposed metric presents a holistic assessment of model performance and emission cost. As such solvers grow in scale and deployment, EcoL2 represents a step toward building performant scientific machine learning systems with lower long-term environmental impact. 

---
# FreqSelect: Frequency-Aware fMRI-to-Image Reconstruction 

**Authors**: Junliang Ye, Lei Wang, Md Zakir Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2505.12552)  

**Abstract**: Reconstructing natural images from functional magnetic resonance imaging (fMRI) data remains a core challenge in natural decoding due to the mismatch between the richness of visual stimuli and the noisy, low resolution nature of fMRI signals. While recent two-stage models, combining deep variational autoencoders (VAEs) with diffusion models, have advanced this task, they treat all spatial-frequency components of the input equally. This uniform treatment forces the model to extract meaning features and suppress irrelevant noise simultaneously, limiting its effectiveness. We introduce FreqSelect, a lightweight, adaptive module that selectively filters spatial-frequency bands before encoding. By dynamically emphasizing frequencies that are most predictive of brain activity and suppressing those that are uninformative, FreqSelect acts as a content-aware gate between image features and natural data. It integrates seamlessly into standard very deep VAE-diffusion pipelines and requires no additional supervision. Evaluated on the Natural Scenes dataset, FreqSelect consistently improves reconstruction quality across both low- and high-level metrics. Beyond performance gains, the learned frequency-selection patterns offer interpretable insights into how different visual frequencies are represented in the brain. Our method generalizes across subjects and scenes, and holds promise for extension to other neuroimaging modalities, offering a principled approach to enhancing both decoding accuracy and neuroscientific interpretability. 

---
# ProMi: An Efficient Prototype-Mixture Baseline for Few-Shot Segmentation with Bounding-Box Annotations 

**Authors**: Florent Chiaroni, Ali Ayub, Ola Ahmad  

**Link**: [PDF](https://arxiv.org/pdf/2505.12547)  

**Abstract**: In robotics applications, few-shot segmentation is crucial because it allows robots to perform complex tasks with minimal training data, facilitating their adaptation to diverse, real-world environments. However, pixel-level annotations of even small amount of images is highly time-consuming and costly. In this paper, we present a novel few-shot binary segmentation method based on bounding-box annotations instead of pixel-level labels. We introduce, ProMi, an efficient prototype-mixture-based method that treats the background class as a mixture of distributions. Our approach is simple, training-free, and effective, accommodating coarse annotations with ease. Compared to existing baselines, ProMi achieves the best results across different datasets with significant gains, demonstrating its effectiveness. Furthermore, we present qualitative experiments tailored to real-world mobile robot tasks, demonstrating the applicability of our approach in such scenarios. Our code: this https URL. 

---
# Exploring Sparsity for Parameter Efficient Fine Tuning Using Wavelets 

**Authors**: Ahmet Bilican, M. Akın Yılmaz, A. Murat Tekalp, R. Gökberk Cinbiş  

**Link**: [PDF](https://arxiv.org/pdf/2505.12532)  

**Abstract**: Efficiently adapting large foundation models is critical, especially with tight compute and memory budgets. Parameter-Efficient Fine-Tuning (PEFT) methods such as LoRA offer limited granularity and effectiveness in few-parameter regimes. We propose Wavelet Fine-Tuning (WaveFT), a novel PEFT method that learns highly sparse updates in the wavelet domain of residual matrices. WaveFT allows precise control of trainable parameters, offering fine-grained capacity adjustment and excelling with remarkably low parameter count, potentially far fewer than LoRA's minimum -- ideal for extreme parameter-efficient scenarios. In order to demonstrate the effect of the wavelet transform, we compare WaveFT with a special case, called SHiRA, that entails applying sparse updates directly in the weight domain. Evaluated on personalized text-to-image generation using Stable Diffusion XL as baseline, WaveFT significantly outperforms LoRA and other PEFT methods, especially at low parameter counts; achieving superior subject fidelity, prompt alignment, and image diversity. 

---
# Scalable Strategies for Continual Learning with Replay 

**Authors**: Truman Hickok  

**Link**: [PDF](https://arxiv.org/pdf/2505.12512)  

**Abstract**: Future deep learning models will be distinguished by systems that perpetually learn through interaction, imagination, and cooperation, blurring the line between training and inference. This makes continual learning a critical challenge, as methods that efficiently maximize bidirectional transfer across learning trajectories will be essential. Replay is on track to play a foundational role in continual learning, allowing models to directly reconcile new information with past knowledge. In practice, however, replay is quite unscalable, doubling the cost of continual learning when applied naively. Moreover, the continual learning literature has not fully synchronized with the multi-task fine-tuning literature, having not fully integrated highly scalable techniques like model merging and low rank adaptation into a replay-enabled toolset that can produce a unified model in the face of many sequential tasks. In this paper, we begin by applying and analyzing low rank adaptation in a continual learning setting. Next, we introduce consolidation, a phasic approach to replay which leads to up to 55\% less replay samples being needed for a given performance target. Then, we propose sequential merging, an offshoot of task arithmetic which is tailored to the continual learning setting and is shown to work well in combination with replay. Finally, we demonstrate that the developed strategies can operate synergistically, resulting in a highly scalable toolset that outperforms standalone variants. 

---
# Towards Budget-Friendly Model-Agnostic Explanation Generation for Large Language Models 

**Authors**: Junhao Liu, Haonan Yu, Xin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12509)  

**Abstract**: With Large language models (LLMs) becoming increasingly prevalent in various applications, the need for interpreting their predictions has become a critical challenge. As LLMs vary in architecture and some are closed-sourced, model-agnostic techniques show great promise without requiring access to the model's internal parameters. However, existing model-agnostic techniques need to invoke LLMs many times to gain sufficient samples for generating faithful explanations, which leads to high economic costs. In this paper, we show that it is practical to generate faithful explanations for large-scale LLMs by sampling from some budget-friendly models through a series of empirical studies. Moreover, we show that such proxy explanations also perform well on downstream tasks. Our analysis provides a new paradigm of model-agnostic explanation methods for LLMs, by including information from budget-friendly models. 

---
# Unsupervised Invariant Risk Minimization 

**Authors**: Yotam Norman, Ron Meir  

**Link**: [PDF](https://arxiv.org/pdf/2505.12506)  

**Abstract**: We propose a novel unsupervised framework for \emph{Invariant Risk Minimization} (IRM), extending the concept of invariance to settings where labels are unavailable. Traditional IRM methods rely on labeled data to learn representations that are robust to distributional shifts across environments. In contrast, our approach redefines invariance through feature distribution alignment, enabling robust representation learning from unlabeled data. We introduce two methods within this framework: Principal Invariant Component Analysis (PICA), a linear method that extracts invariant directions under Gaussian assumptions, and Variational Invariant Autoencoder (VIAE), a deep generative model that disentangles environment-invariant and environment-dependent latent factors. Our approach is based on a novel ``unsupervised'' structural causal model and supports environment-conditioned sample-generation and intervention. Empirical evaluations on synthetic dataset and modified versions of MNIST demonstrate the effectiveness of our methods in capturing invariant structure, preserving relevant information, and generalizing across environments without access to labels. 

---
# CPGD: Toward Stable Rule-based Reinforcement Learning for Language Models 

**Authors**: Zongkai Liu, Fanqing Meng, Lingxiao Du, Zhixiang Zhou, Chao Yu, Wenqi Shao, Qiaosheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12504)  

**Abstract**: Recent advances in rule-based reinforcement learning (RL) have significantly improved the reasoning capability of language models (LMs) with rule-based rewards. However, existing RL methods -- such as GRPO, REINFORCE++, and RLOO -- often suffer from training instability, where large policy updates and improper clipping can lead to training collapse. To address this issue, we propose Clipped Policy Gradient Optimization with Policy Drift (CPGD), a novel algorithm designed to stabilize policy learning in LMs. CPGD introduces a policy drift constraint based on KL divergence to dynamically regularize policy updates, and leverages a clip mechanism on the logarithm of the ratio to prevent excessive policy updates. We provide theoretical justification for CPGD and demonstrate through empirical analysis that it mitigates the instability observed in prior approaches. Furthermore, we show that CPGD significantly improves performance while maintaining training stability. Our implementation balances theoretical rigor with practical usability, offering a robust alternative for RL in the post-training of LMs. We release our code at this https URL. 

---
# Unleashing Automated Congestion Control Customization in the Wild 

**Authors**: Amit Cohen, Lev Gloukhenki, Ravid Hadar, Eden Itah, Yehuda Shvut, Michael Schapira  

**Link**: [PDF](https://arxiv.org/pdf/2505.12492)  

**Abstract**: Congestion control (CC) crucially impacts user experience across Internet services like streaming, gaming, AR/VR, and connected cars. Traditionally, CC algorithm design seeks universal control rules that yield high performance across diverse application domains and networks. However, varying service needs and network conditions challenge this approach. We share operational experience with a system that automatically customizes congestion control logic to service needs and network conditions. We discuss design, deployment challenges, and solutions, highlighting performance benefits through case studies in streaming, gaming, connected cars, and more.
Our system leverages PCC Vivace, an online-learning based congestion control protocol developed by researchers. Hence, along with insights from customizing congestion control, we also discuss lessons learned and modifications made to adapt PCC Vivace for real-world deployment. 

---
# Video-GPT via Next Clip Diffusion 

**Authors**: Shaobin Zhuang, Zhipeng Huang, Ying Zhang, Fangyikang Wang, Canmiao Fu, Binxin Yang, Chong Sun, Chen Li, Yali Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12489)  

**Abstract**: GPT has shown its remarkable success in natural language processing. However, the language sequence is not sufficient to describe spatial-temporal details in the visual world. Alternatively, the video sequence is good at capturing such details. Motivated by this fact, we propose a concise Video-GPT in this paper by treating video as new language for visual world modeling. By analogy to next token prediction in GPT, we introduce a novel next clip diffusion paradigm for pretraining Video-GPT. Different from the previous works, this distinct paradigm allows Video-GPT to tackle both short-term generation and long-term prediction, by autoregressively denoising the noisy clip according to the clean clips in the history. Extensive experiments show our Video-GPT achieves the state-of-the-art performance on video prediction, which is the key factor towards world modeling (Physics-IQ Benchmark: Video-GPT 34.97 vs. Kling 23.64 vs. Wan 20.89). Moreover, it can be well adapted on 6 mainstream video tasks in both video generation and understanding, showing its great generalization capacity in downstream. The project page is at this https URL. 

---
# Joint Embedding vs Reconstruction: Provable Benefits of Latent Space Prediction for Self Supervised Learning 

**Authors**: Hugues Van Assel, Mark Ibrahim, Tommaso Biancalani, Aviv Regev, Randall Balestriero  

**Link**: [PDF](https://arxiv.org/pdf/2505.12477)  

**Abstract**: Reconstruction and joint embedding have emerged as two leading paradigms in Self Supervised Learning (SSL). Reconstruction methods focus on recovering the original sample from a different view in input space. On the other hand, joint embedding methods align the representations of different views in latent space. Both approaches offer compelling advantages, yet practitioners lack clear guidelines for choosing between them. In this work, we unveil the core mechanisms that distinguish each paradigm. By leveraging closed form solutions for both approaches, we precisely characterize how the view generation process, e.g. data augmentation, impacts the learned representations. We then demonstrate that, unlike supervised learning, both SSL paradigms require a minimal alignment between augmentations and irrelevant features to achieve asymptotic optimality with increasing sample size. Our findings indicate that in scenarios where these irrelevant features have a large magnitude, joint embedding methods are preferable because they impose a strictly weaker alignment condition compared to reconstruction based methods. These results not only clarify the trade offs between the two paradigms but also substantiate the empirical success of joint embedding approaches on real world challenging datasets. 

---
# Enhancing Large Language Models with Reward-guided Tree Search for Knowledge Graph Question and Answering 

**Authors**: Xiao Long, Liansheng Zhuang, Chen Shen, Shaotian Yan, Yifei Li, Shafei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12476)  

**Abstract**: Recently, large language models (LLMs) have demonstrated impressive performance in Knowledge Graph Question Answering (KGQA) tasks, which aim to find answers based on knowledge graphs (KGs) for natural language questions. Existing LLMs-based KGQA methods typically follow the Graph Retrieval-Augmented Generation (GraphRAG) paradigm, which first retrieves reasoning paths from the large KGs, and then generates the answers based on them. However, these methods emphasize the exploration of new optimal reasoning paths in KGs while ignoring the exploitation of historical reasoning paths, which may lead to sub-optimal reasoning paths. Additionally, the complex semantics contained in questions may lead to the retrieval of inaccurate reasoning paths. To address these issues, this paper proposes a novel and training-free framework for KGQA tasks called Reward-guided Tree Search on Graph (RTSoG). RTSoG decomposes an original question into a series of simpler and well-defined sub-questions to handle the complex semantics. Then, a Self-Critic Monte Carlo Tree Search (SC-MCTS) guided by a reward model is introduced to iteratively retrieve weighted reasoning paths as contextual knowledge. Finally, it stacks the weighted reasoning paths according to their weights to generate the final answers. Extensive experiments on four datasets demonstrate the effectiveness of RTSoG. Notably, it achieves 8.7\% and 7.0\% performance improvement over the state-of-the-art method on the GrailQA and the WebQSP respectively. 

---
# Beyond Frameworks: Unpacking Collaboration Strategies in Multi-Agent Systems 

**Authors**: Haochun Wang, Sendong Zhao, Jingbo Wang, Zewen Qiang, Bing Qin, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12467)  

**Abstract**: Multi-agent collaboration has emerged as a pivotal paradigm for addressing complex, distributed tasks in large language model (LLM)-driven applications. While prior research has focused on high-level architectural frameworks, the granular mechanisms governing agents, critical to performance and scalability, remain underexplored. This study systematically investigates four dimensions of collaboration strategies: (1) agent governance, (2) participation control, (3) interaction dynamics, and (4) dialogue history management. Through rigorous experimentation under two context-dependent scenarios: Distributed Evidence Integration (DEI) and Structured Evidence Synthesis (SES), we quantify the impact of these strategies on both task accuracy and computational efficiency. Our findings reveal that centralized governance, instructor-led participation, ordered interaction patterns, and instructor-curated context summarization collectively optimize the trade-off between decision quality and resource utilization with the support of the proposed Token-Accuracy Ratio (TAR). This work establishes a foundation for designing adaptive, scalable multi-agent systems, shifting the focus from structural novelty to strategic interaction mechanics. 

---
# IP Leakage Attacks Targeting LLM-Based Multi-Agent Systems 

**Authors**: Liwen Wang, Wenxuan Wang, Shuai Wang, Zongjie Li, Zhenlan Ji, Zongyi Lyu, Daoyuan Wu, Shing-Chi Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2505.12442)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) has led to the emergence of Multi-Agent Systems (MAS) to perform complex tasks through collaboration. However, the intricate nature of MAS, including their architecture and agent interactions, raises significant concerns regarding intellectual property (IP) protection. In this paper, we introduce MASLEAK, a novel attack framework designed to extract sensitive information from MAS applications. MASLEAK targets a practical, black-box setting, where the adversary has no prior knowledge of the MAS architecture or agent configurations. The adversary can only interact with the MAS through its public API, submitting attack query $q$ and observing outputs from the final agent. Inspired by how computer worms propagate and infect vulnerable network hosts, MASLEAK carefully crafts adversarial query $q$ to elicit, propagate, and retain responses from each MAS agent that reveal a full set of proprietary components, including the number of agents, system topology, system prompts, task instructions, and tool usages. We construct the first synthetic dataset of MAS applications with 810 applications and also evaluate MASLEAK against real-world MAS applications, including Coze and CrewAI. MASLEAK achieves high accuracy in extracting MAS IP, with an average attack success rate of 87% for system prompts and task instructions, and 92% for system architecture in most cases. We conclude by discussing the implications of our findings and the potential defenses. 

---
# Addressing the Scarcity of Benchmarks for Graph XAI 

**Authors**: Michele Fontanesi, Alessio Micheli, Marco Podda, Domenico Tortorella  

**Link**: [PDF](https://arxiv.org/pdf/2505.12437)  

**Abstract**: While Graph Neural Networks (GNNs) have become the de facto model for learning from structured data, their decisional process remains opaque to the end user, undermining their deployment in safety-critical applications. In the case of graph classification, Explainable Artificial Intelligence (XAI) techniques address this major issue by identifying sub-graph motifs that explain predictions. However, advancements in this field are hindered by a chronic scarcity of benchmark datasets with known ground-truth motifs to assess the explanations' quality. Current graph XAI benchmarks are limited to synthetic data or a handful of real-world tasks hand-curated by domain experts. In this paper, we propose a general method to automate the construction of XAI benchmarks for graph classification from real-world datasets. We provide both 15 ready-made benchmarks, as well as the code to generate more than 2000 additional XAI benchmarks with our method. As a use case, we employ our benchmarks to assess the effectiveness of some popular graph explainers. 

---
# SGDPO: Self-Guided Direct Preference Optimization for Language Model Alignment 

**Authors**: Wenqiao Zhu, Ji Liu, Lulu Wang, Jun Wu, Yulun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12435)  

**Abstract**: Direct Preference Optimization (DPO) is broadly utilized for aligning Large Language Models (LLMs) with human values because of its flexibility. Despite its effectiveness, it has been observed that the capability of DPO to generate human-preferred response is limited and the results of DPO are far from resilient. To address these limitations, in this paper we propose a novel Self-Guided Direct Preference Optimization algorithm, i.e., SGDPO, which incorporates a pilot term to steer the gradient flow during the optimization process, allowing for fine-grained control over the updates of chosen and rejected rewards. We provide a detailed theoretical analysis of our proposed method and elucidate its operational mechanism. Furthermore, we conduct comprehensive experiments on various models and benchmarks. The extensive experimental results demonstrate the consistency between the empirical results and our theoretical analysis and confirm the effectiveness of our proposed approach (up to 9.19% higher score). 

---
# SRLoRA: Subspace Recomposition in Low-Rank Adaptation via Importance-Based Fusion and Reinitialization 

**Authors**: Haodong Yang, Lei Wang, Md Zakir Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2505.12433)  

**Abstract**: Low-Rank Adaptation (LoRA) is a widely adopted parameter-efficient fine-tuning (PEFT) method that injects two trainable low-rank matrices (A and B) into frozen pretrained models. While efficient, LoRA constrains updates to a fixed low-rank subspace (Delta W = BA), which can limit representational capacity and hinder downstream performance. We introduce Subspace Recomposition in Low-Rank Adaptation (SRLoRA) via importance-based fusion and reinitialization, a novel approach that enhances LoRA's expressiveness without compromising its lightweight structure. SRLoRA assigns importance scores to each LoRA pair (a column of B and the corresponding row of A), and dynamically recomposes the subspace during training. Less important pairs are fused into the frozen backbone, freeing capacity to reinitialize new pairs along unused principal directions derived from the pretrained weight's singular value decomposition. This mechanism enables continual subspace refreshment and richer adaptation over time, without increasing the number of trainable parameters. We evaluate SRLoRA on both language and vision tasks, including the GLUE benchmark and various image classification datasets. SRLoRA consistently achieves faster convergence and improved accuracy over standard LoRA, demonstrating its generality, efficiency, and potential for broader PEFT applications. 

---
# Observe-R1: Unlocking Reasoning Abilities of MLLMs with Dynamic Progressive Reinforcement Learning 

**Authors**: Zirun Guo, Minjie Hong, Tao Jin  

**Link**: [PDF](https://arxiv.org/pdf/2505.12432)  

**Abstract**: Reinforcement Learning (RL) has shown promise in improving the reasoning abilities of Large Language Models (LLMs). However, the specific challenges of adapting RL to multimodal data and formats remain relatively unexplored. In this work, we present Observe-R1, a novel framework aimed at enhancing the reasoning capabilities of multimodal large language models (MLLMs). We draw inspirations from human learning progression--from simple to complex and easy to difficult, and propose a gradual learning paradigm for MLLMs. To this end, we construct the NeuraLadder dataset, which is organized and sampled according to the difficulty and complexity of data samples for RL training. To tackle multimodal tasks, we introduce a multimodal format constraint that encourages careful observation of images, resulting in enhanced visual abilities and clearer and more structured responses. Additionally, we implement a bonus reward system that favors concise, correct answers within a length constraint, alongside a dynamic weighting mechanism that prioritizes uncertain and medium-difficulty problems, ensuring that more informative samples have a greater impact on training. Our experiments with the Qwen2.5-VL-3B and Qwen2.5-VL-7B models on 20k samples from the NeuraLadder dataset show that Observe-R1 outperforms a series of larger reasoning models on both reasoning and general benchmarks, achieving superior clarity and conciseness in reasoning chains. Ablation studies validate the effectiveness of our strategies, highlighting the robustness and generalization of our approach. The dataset and code will be released at this https URL. 

---
# EvoGPT: Enhancing Test Suite Robustness via LLM-Based Generation and Genetic Optimization 

**Authors**: Lior Broide, Roni Stern  

**Link**: [PDF](https://arxiv.org/pdf/2505.12424)  

**Abstract**: Large Language Models (LLMs) have recently emerged as promising tools for automated unit test generation. We introduce a hybrid framework called EvoGPT that integrates LLM-based test generation with evolutionary search techniques to create diverse, fault-revealing unit tests. Unit tests are initially generated with diverse temperature sampling to maximize behavioral and test suite diversity, followed by a generation-repair loop and coverage-guided assertion enhancement. The resulting test suites are evolved using genetic algorithms, guided by a fitness function prioritizing mutation score over traditional coverage metrics. This design emphasizes the primary objective of unit testing-fault detection. Evaluated on multiple open-source Java projects, EvoGPT achieves an average improvement of 10% in both code coverage and mutation score compared to LLMs and traditional search-based software testing baselines. These results demonstrate that combining LLM-driven diversity, targeted repair, and evolutionary optimization produces more effective and resilient test suites. 

---
# PSC: Extending Context Window of Large Language Models via Phase Shift Calibration 

**Authors**: Wenqiao Zhu, Chao Xu, Lulu Wang, Jun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12423)  

**Abstract**: Rotary Position Embedding (RoPE) is an efficient position encoding approach and is widely utilized in numerous large language models (LLMs). Recently, a lot of methods have been put forward to further expand the context window based on RoPE. The core concept of those methods is to predefine or search for a set of factors to rescale the base frequencies of RoPE. Nevertheless, it is quite a challenge for existing methods to predefine an optimal factor due to the exponential search space. In view of this, we introduce PSC (Phase Shift Calibration), a small module for calibrating the frequencies predefined by existing methods. With the employment of PSC, we demonstrate that many existing methods can be further enhanced, like PI, YaRN, and LongRoPE. We conducted extensive experiments across multiple models and tasks. The results demonstrate that (1) when PSC is enabled, the comparative reductions in perplexity increase as the context window size is varied from 16k, to 32k, and up to 64k. (2) Our approach is broadly applicable and exhibits robustness across a variety of models and tasks. The code can be found at this https URL. 

---
# Fixed Point Explainability 

**Authors**: Emanuele La Malfa, Jon Vadillo, Marco Molinari, Michael Wooldridge  

**Link**: [PDF](https://arxiv.org/pdf/2505.12421)  

**Abstract**: This paper introduces a formal notion of fixed point explanations, inspired by the "why regress" principle, to assess, through recursive applications, the stability of the interplay between a model and its explainer. Fixed point explanations satisfy properties like minimality, stability, and faithfulness, revealing hidden model behaviours and explanatory weaknesses. We define convergence conditions for several classes of explainers, from feature-based to mechanistic tools like Sparse AutoEncoders, and we report quantitative and qualitative results. 

---
# Mutual Evidential Deep Learning for Medical Image Segmentation 

**Authors**: Yuanpeng He, Yali Bi, Lijian Li, Chi-Man Pun, Wenpin Jiao, Zhi Jin  

**Link**: [PDF](https://arxiv.org/pdf/2505.12418)  

**Abstract**: Existing semi-supervised medical segmentation co-learning frameworks have realized that model performance can be diminished by the biases in model recognition caused by low-quality pseudo-labels. Due to the averaging nature of their pseudo-label integration strategy, they fail to explore the reliability of pseudo-labels from different sources. In this paper, we propose a mutual evidential deep learning (MEDL) framework that offers a potentially viable solution for pseudo-label generation in semi-supervised learning from two perspectives. First, we introduce networks with different architectures to generate complementary evidence for unlabeled samples and adopt an improved class-aware evidential fusion to guide the confident synthesis of evidential predictions sourced from diverse architectural networks. Second, utilizing the uncertainty in the fused evidence, we design an asymptotic Fisher information-based evidential learning strategy. This strategy enables the model to initially focus on unlabeled samples with more reliable pseudo-labels, gradually shifting attention to samples with lower-quality pseudo-labels while avoiding over-penalization of mislabeled classes in high data uncertainty samples. Additionally, for labeled data, we continue to adopt an uncertainty-driven asymptotic learning strategy, gradually guiding the model to focus on challenging voxels. Extensive experiments on five mainstream datasets have demonstrated that MEDL achieves state-of-the-art performance. 

---
# Table-R1: Region-based Reinforcement Learning for Table Understanding 

**Authors**: Zhenhe Wu, Jian Yang, Jiaheng Liu, Xianjie Wu, Changzai Pan, Jie Zhang, Yu Zhao, Shuangyong Song, Yongxiang Li, Zhoujun Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.12415)  

**Abstract**: Tables present unique challenges for language models due to their structured row-column interactions, necessitating specialized approaches for effective comprehension. While large language models (LLMs) have demonstrated potential in table reasoning through prompting and techniques like chain-of-thought (CoT) and program-of-thought (PoT), optimizing their performance for table question answering remains underexplored. In this paper, we introduce region-based Table-R1, a novel reinforcement learning approach that enhances LLM table understanding by integrating region evidence into reasoning steps. Our method employs Region-Enhanced Supervised Fine-Tuning (RE-SFT) to guide models in identifying relevant table regions before generating answers, incorporating textual, symbolic, and program-based reasoning. Additionally, Table-Aware Group Relative Policy Optimization (TARPO) introduces a mixed reward system to dynamically balance region accuracy and answer correctness, with decaying region rewards and consistency penalties to align reasoning steps. Experiments show that Table-R1 achieves an average performance improvement of 14.36 points across multiple base models on three benchmark datasets, even outperforming baseline models with ten times the parameters, while TARPO reduces response token consumption by 67.5% compared to GRPO, significantly advancing LLM capabilities in efficient tabular reasoning. 

---
# ViEEG: Hierarchical Neural Coding with Cross-Modal Progressive Enhancement for EEG-Based Visual Decoding 

**Authors**: Minxu Liu, Donghai Guan, Chuhang Zheng, Chunwei Tian, Jie Wen, Qi Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12408)  

**Abstract**: Understanding and decoding brain activity into visual representations is a fundamental challenge at the intersection of neuroscience and artificial intelligence. While EEG-based visual decoding has shown promise due to its non-invasive, low-cost nature and millisecond-level temporal resolution, existing methods are limited by their reliance on flat neural representations that overlook the brain's inherent visual hierarchy. In this paper, we introduce ViEEG, a biologically inspired hierarchical EEG decoding framework that aligns with the Hubel-Wiesel theory of visual processing. ViEEG decomposes each visual stimulus into three biologically aligned components-contour, foreground object, and contextual scene-serving as anchors for a three-stream EEG encoder. These EEG features are progressively integrated via cross-attention routing, simulating cortical information flow from V1 to IT to the association cortex. We further adopt hierarchical contrastive learning to align EEG representations with CLIP embeddings, enabling zero-shot object recognition. Extensive experiments on the THINGS-EEG dataset demonstrate that ViEEG achieves state-of-the-art performance, with 40.9% Top-1 accuracy in subject-dependent and 22.9% Top-1 accuracy in cross-subject settings, surpassing existing methods by over 45%. Our framework not only advances the performance frontier but also sets a new paradigm for biologically grounded brain decoding in AI. 

---
# The power of text similarity in identifying AI-LLM paraphrased documents: The case of BBC news articles and ChatGPT 

**Authors**: Konstantinos Xylogiannopoulos, Petros Xanthopoulos, Panagiotis Karampelas, Georgios Bakamitsos  

**Link**: [PDF](https://arxiv.org/pdf/2505.12405)  

**Abstract**: Generative AI paraphrased text can be used for copyright infringement and the AI paraphrased content can deprive substantial revenue from original content creators. Despite this recent surge of malicious use of generative AI, there are few academic publications that research this threat. In this article, we demonstrate the ability of pattern-based similarity detection for AI paraphrased news recognition. We propose an algorithmic scheme, which is not limited to detect whether an article is an AI paraphrase, but, more importantly, to identify that the source of infringement is the ChatGPT. The proposed method is tested with a benchmark dataset specifically created for this task that incorporates real articles from BBC, incorporating a total of 2,224 articles across five different news categories, as well as 2,224 paraphrased articles created with ChatGPT. Results show that our pattern similarity-based method, that makes no use of deep learning, can detect ChatGPT assisted paraphrased articles at percentages 96.23% for accuracy, 96.25% for precision, 96.21% for sensitivity, 96.25% for specificity and 96.23% for F1 score. 

---
# Hyperbolic Residual Quantization: Discrete Representations for Data with Latent Hierarchies 

**Authors**: Piotr Piękos, Subhradeep Kayal, Alexandros Karatzoglou  

**Link**: [PDF](https://arxiv.org/pdf/2505.12404)  

**Abstract**: Hierarchical data arise in countless domains, from biological taxonomies and organizational charts to legal codes and knowledge graphs. Residual Quantization (RQ) is widely used to generate discrete, multitoken representations for such data by iteratively quantizing residuals in a multilevel codebook. However, its reliance on Euclidean geometry can introduce fundamental mismatches that hinder modeling of hierarchical branching, necessary for faithful representation of hierarchical data. In this work, we propose Hyperbolic Residual Quantization (HRQ), which embeds data natively in a hyperbolic manifold and performs residual quantization using hyperbolic operations and distance metrics. By adapting the embedding network, residual computation, and distance metric to hyperbolic geometry, HRQ imparts an inductive bias that aligns naturally with hierarchical branching. We claim that HRQ in comparison to RQ can generate more useful for downstream tasks discrete hierarchical representations for data with latent hierarchies. We evaluate HRQ on two tasks: supervised hierarchy modeling using WordNet hypernym trees, where the model is supervised to learn the latent hierarchy - and hierarchy discovery, where, while latent hierarchy exists in the data, the model is not directly trained or evaluated on a task related to the hierarchy. Across both scenarios, HRQ hierarchical tokens yield better performance on downstream tasks compared to Euclidean RQ with gains of up to $20\%$ for the hierarchy modeling task. Our results demonstrate that integrating hyperbolic geometry into discrete representation learning substantially enhances the ability to capture latent hierarchies. 

---
# Traversal Verification for Speculative Tree Decoding 

**Authors**: Yepeng Weng, Qiao Hu, Xujie Chen, Li Liu, Dianwen Mei, Huishi Qiu, Jiang Tian, Zhongchao Shi  

**Link**: [PDF](https://arxiv.org/pdf/2505.12398)  

**Abstract**: Speculative decoding is a promising approach for accelerating large language models. The primary idea is to use a lightweight draft model to speculate the output of the target model for multiple subsequent timesteps, and then verify them in parallel to determine whether the drafted tokens should be accepted or rejected. To enhance acceptance rates, existing frameworks typically construct token trees containing multiple candidates in each timestep. However, their reliance on token-level verification mechanisms introduces two critical limitations: First, the probability distribution of a sequence differs from that of individual tokens, leading to suboptimal acceptance length. Second, current verification schemes begin from the root node and proceed layer by layer in a top-down manner. Once a parent node is rejected, all its child nodes should be discarded, resulting in inefficient utilization of speculative candidates. This paper introduces Traversal Verification, a novel speculative decoding algorithm that fundamentally rethinks the verification paradigm through leaf-to-root traversal. Our approach considers the acceptance of the entire token sequence from the current node to the root, and preserves potentially valid subsequences that would be prematurely discarded by existing methods. We theoretically prove that the probability distribution obtained through Traversal Verification is identical to that of the target model, guaranteeing lossless inference while achieving substantial acceleration gains. Experimental results across different large language models and multiple tasks show that our method consistently improves acceptance length and throughput over existing methods 

---
# Few-Shot Concept Unlearning with Low Rank Adaptation 

**Authors**: Udaya Shreyas, L.N. Aadarsh  

**Link**: [PDF](https://arxiv.org/pdf/2505.12395)  

**Abstract**: Image Generation models are a trending topic nowadays, with many people utilizing Artificial Intelligence models in order to generate images. There are many such models which, given a prompt of a text, will generate an image which depicts said prompt. There are many image generation models, such as Latent Diffusion Models, Denoising Diffusion Probabilistic Models, Generative Adversarial Networks and many more. When generating images, these models can generate sensitive image data, which can be threatening to privacy or may violate copyright laws of private entities. Machine unlearning aims at removing the influence of specific data subsets from the trained models and in the case of image generation models, remove the influence of a concept such that the model is unable to generate said images of the concept when prompted. Conventional retraining of the model can take upto days, hence fast algorithms are the need of the hour. In this paper we propose an algorithm that aims to remove the influence of concepts in diffusion models through updating the gradients of the final layers of the text encoders. Using a weighted loss function, we utilize backpropagation in order to update the weights of the final layers of the Text Encoder componet of the Stable Diffusion Model, removing influence of the concept from the text-image embedding space, such that when prompted, the result is an image not containing the concept. The weighted loss function makes use of Textual Inversion and Low-Rank this http URL perform our experiments on Latent Diffusion Models, namely the Stable Diffusion v2 model, with an average concept unlearning runtime of 50 seconds using 4-5 images. 

---
# Data Sharing with a Generative AI Competitor 

**Authors**: Boaz Taitler, Omer Madmon, Moshe Tennenholtz, Omer Ben-Porat  

**Link**: [PDF](https://arxiv.org/pdf/2505.12386)  

**Abstract**: As GenAI platforms grow, their dependence on content from competing providers, combined with access to alternative data sources, creates new challenges for data-sharing decisions. In this paper, we provide a model of data sharing between a content creation firm and a GenAI platform that can also acquire content from third-party experts. The interaction is modeled as a Stackelberg game: the firm first decides how much of its proprietary dataset to share with GenAI, and GenAI subsequently determines how much additional data to acquire from external experts. Their utilities depend on user traffic, monetary transfers, and the cost of acquiring additional data from external experts. We characterize the unique subgame perfect equilibrium of the game and uncover a surprising phenomenon: The firm may be willing to pay GenAI to share the firm's own data, leading to a costly data-sharing equilibrium. We further characterize the set of Pareto improving data prices, and show that such improvements occur only when the firm pays to share data. Finally, we study how the price can be set to optimize different design objectives, such as promoting firm data sharing, expert data acquisition, or a balance of both. Our results shed light on the economic forces shaping data-sharing partnerships in the age of GenAI, and provide guidance for platforms, regulators and policymakers seeking to design effective data exchange mechanisms. 

---
# From n-gram to Attention: How Model Architectures Learn and Propagate Bias in Language Modeling 

**Authors**: Mohsinul Kabir, Tasfia Tahsin, Sophia Ananiadou  

**Link**: [PDF](https://arxiv.org/pdf/2505.12381)  

**Abstract**: Current research on bias in language models (LMs) predominantly focuses on data quality, with significantly less attention paid to model architecture and temporal influences of data. Even more critically, few studies systematically investigate the origins of bias. We propose a methodology grounded in comparative behavioral theory to interpret the complex interaction between training data and model architecture in bias propagation during language modeling. Building on recent work that relates transformers to n-gram LMs, we evaluate how data, model design choices, and temporal dynamics affect bias propagation. Our findings reveal that: (1) n-gram LMs are highly sensitive to context window size in bias propagation, while transformers demonstrate architectural robustness; (2) the temporal provenance of training data significantly affects bias; and (3) different model architectures respond differentially to controlled bias injection, with certain biases (e.g. sexual orientation) being disproportionately amplified. As language models become ubiquitous, our findings highlight the need for a holistic approach -- tracing bias to its origins across both data and model dimensions, not just symptoms, to mitigate harm. 

---
# CAPTURE: Context-Aware Prompt Injection Testing and Robustness Enhancement 

**Authors**: Gauri Kholkar, Ratinder Ahuja  

**Link**: [PDF](https://arxiv.org/pdf/2505.12368)  

**Abstract**: Prompt injection remains a major security risk for large language models. However, the efficacy of existing guardrail models in context-aware settings remains underexplored, as they often rely on static attack benchmarks. Additionally, they have over-defense tendencies. We introduce CAPTURE, a novel context-aware benchmark assessing both attack detection and over-defense tendencies with minimal in-domain examples. Our experiments reveal that current prompt injection guardrail models suffer from high false negatives in adversarial cases and excessive false positives in benign scenarios, highlighting critical limitations. 

---
# DisCO: Reinforcing Large Reasoning Models with Discriminative Constrained Optimization 

**Authors**: Gang Li, Ming Lin, Tomer Galanti, Zhengzhong Tu, Tianbao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12366)  

**Abstract**: The recent success and openness of DeepSeek-R1 have brought widespread attention to Group Relative Policy Optimization (GRPO) as a reinforcement learning method for large reasoning models (LRMs). In this work, we analyze the GRPO objective under a binary reward setting and reveal an inherent limitation of question-level difficulty bias. We also identify a connection between GRPO and traditional discriminative methods in supervised learning. Motivated by these insights, we introduce a new Discriminative Constrained Optimization (DisCO) framework for reinforcing LRMs, grounded in the principle of discriminative learning. The main differences between DisCO and GRPO and its recent variants are: (1) it replaces the group relative objective with a discriminative objective defined by a scoring function; (2) it abandons clipping-based surrogates in favor of non-clipping RL surrogate objectives used as scoring functions; (3) it employs a simple yet effective constrained optimization approach to enforce the KL divergence constraint, ensuring stable training. As a result, DisCO offers notable advantages over GRPO and its variants: (i) it completely eliminates difficulty bias by adopting discriminative objectives; (ii) it addresses the entropy instability in GRPO and its variants through the use of non-clipping scoring functions and a constrained optimization approach; (iii) it allows the incorporation of advanced discriminative learning techniques to address data imbalance, where a significant number of questions have more negative than positive generated answers during training. Our experiments on enhancing the mathematical reasoning capabilities of SFT-finetuned models show that DisCO significantly outperforms GRPO and its improved variants such as DAPO, achieving average gains of 7\% over GRPO and 6\% over DAPO across six benchmark tasks for an 1.5B model. 

---
# Towards Visuospatial Cognition via Hierarchical Fusion of Visual Experts 

**Authors**: Qi Feng, Hidetoshi Shimodaira  

**Link**: [PDF](https://arxiv.org/pdf/2505.12363)  

**Abstract**: While Multimodal Large Language Models (MLLMs) excel at general vision-language tasks, visuospatial cognition - reasoning about spatial layouts, relations, and dynamics - remains a significant challenge. Existing models often lack the necessary architectural components and specialized training data for fine-grained spatial understanding. We introduce ViCA2 (Visuospatial Cognitive Assistant 2), a novel MLLM designed to enhance spatial reasoning. ViCA2 features a dual vision encoder architecture integrating SigLIP for semantics and Hiera for spatial structure, coupled with a token ratio control mechanism for efficiency. We also developed ViCA-322K, a new large-scale dataset with over 322,000 spatially grounded question-answer pairs for targeted instruction tuning. On the challenging VSI-Bench benchmark, our ViCA2-7B model achieves a state-of-the-art average score of 56.8, significantly surpassing larger open-source models (e.g., LLaVA-NeXT-Video-72B, 40.9) and leading proprietary models (Gemini-1.5 Pro, 45.4). This demonstrates the effectiveness of our approach in achieving strong visuospatial intelligence with a compact model. We release ViCA2, its codebase, and the ViCA-322K dataset to facilitate further research. 

---
# Adaptive MPC-based quadrupedal robot control under periodic disturbances 

**Authors**: Elizaveta Pestova, Ilya Osokin, Danil Belov, Pavel Osinenko  

**Link**: [PDF](https://arxiv.org/pdf/2505.12361)  

**Abstract**: Recent advancements in adaptive control for reference trajectory tracking enable quadrupedal robots to perform locomotion tasks under challenging conditions. There are methods enabling the estimation of the external disturbances in terms of forces and torques. However, a specific case of disturbances that are periodic was not explicitly tackled in application to quadrupeds. This work is devoted to the estimation of the periodic disturbances with a lightweight regressor using simplified robot dynamics and extracting the disturbance properties in terms of the magnitude and frequency. Experimental evidence suggests performance improvement over the baseline static disturbance compensation. All source files, including simulation setups, code, and calculation scripts, are available on GitHub at this https URL. 

---
# AbFlowNet: Optimizing Antibody-Antigen Binding Energy via Diffusion-GFlowNet Fusion 

**Authors**: Abrar Rahman Abir, Haz Sameen Shahgir, Md Rownok Zahan Ratul, Md Toki Tahmid, Greg Ver Steeg, Yue Dong  

**Link**: [PDF](https://arxiv.org/pdf/2505.12358)  

**Abstract**: Complementarity Determining Regions (CDRs) are critical segments of an antibody that facilitate binding to specific antigens. Current computational methods for CDR design utilize reconstruction losses and do not jointly optimize binding energy, a crucial metric for antibody efficacy. Rather, binding energy optimization is done through computationally expensive Online Reinforcement Learning (RL) pipelines rely heavily on unreliable binding energy estimators. In this paper, we propose AbFlowNet, a novel generative framework that integrates GFlowNet with Diffusion models. By framing each diffusion step as a state in the GFlowNet framework, AbFlowNet jointly optimizes standard diffusion losses and binding energy by directly incorporating energy signals into the training process, thereby unifying diffusion and reward optimization in a single procedure. Experimental results show that AbFlowNet outperforms the base diffusion model by 3.06% in amino acid recovery, 20.40% in geometric reconstruction (RMSD), and 3.60% in binding energy improvement ratio. ABFlowNet also decreases Top-1 total energy and binding energy errors by 24.8% and 38.1% without pseudo-labeling the test dataset or using computationally expensive online RL regimes. 

---
# A universal policy wrapper with guarantees 

**Authors**: Anton Bolychev, Georgiy Malaniya, Grigory Yaremenko, Anastasia Krasnaya, Pavel Osinenko  

**Link**: [PDF](https://arxiv.org/pdf/2505.12354)  

**Abstract**: We introduce a universal policy wrapper for reinforcement learning agents that ensures formal goal-reaching guarantees. In contrast to standard reinforcement learning algorithms that excel in performance but lack rigorous safety assurances, our wrapper selectively switches between a high-performing base policy -- derived from any existing RL method -- and a fallback policy with known convergence properties. Base policy's value function supervises this switching process, determining when the fallback policy should override the base policy to ensure the system remains on a stable path. The analysis proves that our wrapper inherits the fallback policy's goal-reaching guarantees while preserving or improving upon the performance of the base policy. Notably, it operates without needing additional system knowledge or online constrained optimization, making it readily deployable across diverse reinforcement learning architectures and tasks. 

---
# Importance Sampling for Nonlinear Models 

**Authors**: Prakash Palanivelu Rajmohan, Fred Roosta  

**Link**: [PDF](https://arxiv.org/pdf/2505.12353)  

**Abstract**: While norm-based and leverage-score-based methods have been extensively studied for identifying "important" data points in linear models, analogous tools for nonlinear models remain significantly underdeveloped. By introducing the concept of the adjoint operator of a nonlinear map, we address this gap and generalize norm-based and leverage-score-based importance sampling to nonlinear settings. We demonstrate that sampling based on these generalized notions of norm and leverage scores provides approximation guarantees for the underlying nonlinear mapping, similar to linear subspace embeddings. As direct applications, these nonlinear scores not only reduce the computational complexity of training nonlinear models by enabling efficient sampling over large datasets but also offer a novel mechanism for model explainability and outlier detection. Our contributions are supported by both theoretical analyses and experimental results across a variety of supervised learning scenarios. 

---
# Multi-CALF: A Policy Combination Approach with Statistical Guarantees 

**Authors**: Georgiy Malaniya, Anton Bolychev, Grigory Yaremenko, Anastasia Krasnaya, Pavel Osinenko  

**Link**: [PDF](https://arxiv.org/pdf/2505.12350)  

**Abstract**: We introduce Multi-CALF, an algorithm that intelligently combines reinforcement learning policies based on their relative value improvements. Our approach integrates a standard RL policy with a theoretically-backed alternative policy, inheriting formal stability guarantees while often achieving better performance than either policy individually. We prove that our combined policy converges to a specified goal set with known probability and provide precise bounds on maximum deviation and convergence time. Empirical validation on control tasks demonstrates enhanced performance while maintaining stability guarantees. 

---
# Wisdom from Diversity: Bias Mitigation Through Hybrid Human-LLM Crowds 

**Authors**: Axel Abels, Tom Lenaerts  

**Link**: [PDF](https://arxiv.org/pdf/2505.12349)  

**Abstract**: Despite their performance, large language models (LLMs) can inadvertently perpetuate biases found in the data they are trained on. By analyzing LLM responses to bias-eliciting headlines, we find that these models often mirror human biases. To address this, we explore crowd-based strategies for mitigating bias through response aggregation. We first demonstrate that simply averaging responses from multiple LLMs, intended to leverage the "wisdom of the crowd", can exacerbate existing biases due to the limited diversity within LLM crowds. In contrast, we show that locally weighted aggregation methods more effectively leverage the wisdom of the LLM crowd, achieving both bias mitigation and improved accuracy. Finally, recognizing the complementary strengths of LLMs (accuracy) and humans (diversity), we demonstrate that hybrid crowds containing both significantly enhance performance and further reduce biases across ethnic and gender-related contexts. 

---
# Mitigating Hallucinations via Inter-Layer Consistency Aggregation in Large Vision-Language Models 

**Authors**: Kai Tang, Jinhao You, Xiuqi Ge, Hanze Li, Yichen Guo, Xiande Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12343)  

**Abstract**: Despite the impressive capabilities of Large Vision-Language Models (LVLMs), they remain susceptible to hallucinations-generating content that is inconsistent with the input image. Existing training-free hallucination mitigation methods often suffer from unstable performance and high sensitivity to hyperparameter settings, limiting their practicality and broader adoption. In this paper, we propose a novel decoding mechanism, Decoding with Inter-layer Consistency via Layer Aggregation (DCLA), which requires no retraining, fine-tuning, or access to external knowledge bases. Specifically, our approach constructs a dynamic semantic reference by aggregating representations from previous layers, and corrects semantically deviated layers to enforce inter-layer consistency. The method allows DCLA to robustly mitigate hallucinations across multiple LVLMs. Experiments on hallucination benchmarks such as MME and POPE demonstrate that DCLA effectively reduces hallucinations while enhancing the reliability and performance of LVLMs. 

---
# Towards Open-world Generalized Deepfake Detection: General Feature Extraction via Unsupervised Domain Adaptation 

**Authors**: Midou Guo, Qilin Yin, Wei Lu, Xiangyang Luo  

**Link**: [PDF](https://arxiv.org/pdf/2505.12339)  

**Abstract**: With the development of generative artificial intelligence, new forgery methods are rapidly emerging. Social platforms are flooded with vast amounts of unlabeled synthetic data and authentic data, making it increasingly challenging to distinguish real from fake. Due to the lack of labels, existing supervised detection methods struggle to effectively address the detection of unknown deepfake methods. Moreover, in open world scenarios, the amount of unlabeled data greatly exceeds that of labeled data. Therefore, we define a new deepfake detection generalization task which focuses on how to achieve efficient detection of large amounts of unlabeled data based on limited labeled data to simulate a open world scenario. To solve the above mentioned task, we propose a novel Open-World Deepfake Detection Generalization Enhancement Training Strategy (OWG-DS) to improve the generalization ability of existing methods. Our approach aims to transfer deepfake detection knowledge from a small amount of labeled source domain data to large-scale unlabeled target domain data. Specifically, we introduce the Domain Distance Optimization (DDO) module to align different domain features by optimizing both inter-domain and intra-domain distances. Additionally, the Similarity-based Class Boundary Separation (SCBS) module is used to enhance the aggregation of similar samples to ensure clearer class boundaries, while an adversarial training mechanism is adopted to learn the domain-invariant features. Extensive experiments show that the proposed deepfake detection generalization enhancement training strategy excels in cross-method and cross-dataset scenarios, improving the model's generalization. 

---
# VoiceCloak: A Multi-Dimensional Defense Framework against Unauthorized Diffusion-based Voice Cloning 

**Authors**: Qianyue Hu, Junyan Wu, Wei Lu, Xiangyang Luo  

**Link**: [PDF](https://arxiv.org/pdf/2505.12332)  

**Abstract**: Diffusion Models (DMs) have achieved remarkable success in realistic voice cloning (VC), while they also increase the risk of malicious misuse. Existing proactive defenses designed for traditional VC models aim to disrupt the forgery process, but they have been proven incompatible with DMs due to the intricate generative mechanisms of diffusion. To bridge this gap, we introduce VoiceCloak, a multi-dimensional proactive defense framework with the goal of obfuscating speaker identity and degrading perceptual quality in potential unauthorized VC. To achieve these goals, we conduct a focused analysis to identify specific vulnerabilities within DMs, allowing VoiceCloak to disrupt the cloning process by introducing adversarial perturbations into the reference audio. Specifically, to obfuscate speaker identity, VoiceCloak first targets speaker identity by distorting representation learning embeddings to maximize identity variation, which is guided by auditory perception principles. Additionally, VoiceCloak disrupts crucial conditional guidance processes, particularly attention context, thereby preventing the alignment of vocal characteristics that are essential for achieving convincing cloning. Then, to address the second objective, VoiceCloak introduces score magnitude amplification to actively steer the reverse trajectory away from the generation of high-quality speech. Noise-guided semantic corruption is further employed to disrupt structural speech semantics captured by DMs, degrading output quality. Extensive experiments highlight VoiceCloak's outstanding defense success rate against unauthorized diffusion-based voice cloning. Audio samples of VoiceCloak are available at this https URL. 

---
# Robust Planning for Autonomous Driving via Mixed Adversarial Diffusion Predictions 

**Authors**: Albert Zhao, Stefano Soatto  

**Link**: [PDF](https://arxiv.org/pdf/2505.12327)  

**Abstract**: We describe a robust planning method for autonomous driving that mixes normal and adversarial agent predictions output by a diffusion model trained for motion prediction. We first train a diffusion model to learn an unbiased distribution of normal agent behaviors. We then generate a distribution of adversarial predictions by biasing the diffusion model at test time to generate predictions that are likely to collide with a candidate plan. We score plans using expected cost with respect to a mixture distribution of normal and adversarial predictions, leading to a planner that is robust against adversarial behaviors but not overly conservative when agents behave normally. Unlike current approaches, we do not use risk measures that over-weight adversarial behaviors while placing little to no weight on low-cost normal behaviors or use hard safety constraints that may not be appropriate for all driving scenarios. We show the effectiveness of our method on single-agent and multi-agent jaywalking scenarios as well as a red light violation scenario. 

---
# Visuospatial Cognitive Assistant 

**Authors**: Qi Feng, Hidetoshi Shimodaira  

**Link**: [PDF](https://arxiv.org/pdf/2505.12312)  

**Abstract**: Video-based spatial cognition is vital for robotics and embodied AI but challenges current Vision-Language Models (VLMs). This paper makes two key contributions. First, we introduce ViCA (Visuospatial Cognitive Assistant)-322K, a diverse dataset of 322,003 QA pairs from real-world indoor videos (ARKitScenes, ScanNet, ScanNet++), offering supervision for 3D metadata-grounded queries and video-based complex reasoning. Second, we develop ViCA-7B, fine-tuned on ViCA-322K, which achieves new state-of-the-art on all eight VSI-Bench tasks, outperforming existing models, including larger ones (e.g., +26.1 on Absolute Distance). For interpretability, we present ViCA-Thinking-2.68K, a dataset with explicit reasoning chains, and fine-tune ViCA-7B to create ViCA-7B-Thinking, a model that articulates its spatial reasoning. Our work highlights the importance of targeted data and suggests paths for improved temporal-spatial modeling. We release all resources to foster research in robust visuospatial intelligence. 

---
# DNOI-4DRO: Deep 4D Radar Odometry with Differentiable Neural-Optimization Iterations 

**Authors**: Shouyi Lu, Huanyu Zhou, Guirong Zhuo  

**Link**: [PDF](https://arxiv.org/pdf/2505.12310)  

**Abstract**: A novel learning-optimization-combined 4D radar odometry model, named DNOI-4DRO, is proposed in this paper. The proposed model seamlessly integrates traditional geometric optimization with end-to-end neural network training, leveraging an innovative differentiable neural-optimization iteration operator. In this framework, point-wise motion flow is first estimated using a neural network, followed by the construction of a cost function based on the relationship between point motion and pose in 3D space. The radar pose is then refined using Gauss-Newton updates. Additionally, we design a dual-stream 4D radar backbone that integrates multi-scale geometric features and clustering-based class-aware features to enhance the representation of sparse 4D radar point clouds. Extensive experiments on the VoD and Snail-Radar datasets demonstrate the superior performance of our model, which outperforms recent classical and learning-based approaches. Notably, our method even achieves results comparable to A-LOAM with mapping optimization using LiDAR point clouds as input. Our models and code will be publicly released. 

---
# Community Search in Time-dependent Road-social Attributed Networks 

**Authors**: Li Ni, Hengkai Xu, Lin Mu, Yiwen Zhang, Wenjian Luo  

**Link**: [PDF](https://arxiv.org/pdf/2505.12309)  

**Abstract**: Real-world networks often involve both keywords and locations, along with travel time variations between locations due to traffic conditions. However, most existing cohesive subgraph-based community search studies utilize a single attribute, either keywords or locations, to identify communities. They do not simultaneously consider both keywords and locations, which results in low semantic or spatial cohesiveness of the detected communities, and they fail to account for variations in travel time. Additionally, these studies traverse the entire network to build efficient indexes, but the detected community only involves nodes around the query node, leading to the traversal of nodes that are not relevant to the community. Therefore, we propose the problem of discovering semantic-spatial aware k-core, which refers to a k-core with high semantic and time-dependent spatial cohesiveness containing the query node. To address this problem, we propose an exact and a greedy algorithm, both of which gradually expand outward from the query node. They are local methods that only access the local part of the attributed network near the query node rather than the entire network. Moreover, we design a method to calculate the semantic similarity between two keywords using large language models. This method alleviates the disadvantages of keyword-matching methods used in existing community search studies, such as mismatches caused by differently expressed synonyms and the presence of irrelevant words. Experimental results show that the greedy algorithm outperforms baselines in terms of structural, semantic, and time-dependent spatial cohesiveness. 

---
# Pre-trained Prompt-driven Community Search 

**Authors**: Li Ni, Hengkai Xu, Lin Mu, Yiwen Zhang, Wenjian Luo  

**Link**: [PDF](https://arxiv.org/pdf/2505.12304)  

**Abstract**: The "pre-train, prompt" paradigm is widely adopted in various graph-based tasks and has shown promising performance in community detection. Most existing semi-supervised community detection algorithms detect communities based on known ones, and the detected communities typically do not contain the given query node. Therefore, they are not suitable for searching the community of a given node. Motivated by this, we adopt this paradigm into the semi-supervised community search for the first time and propose Pre-trained Prompt-driven Community Search (PPCS), a novel model designed to enhance search accuracy and efficiency. PPCS consists of three main components: node encoding, sample generation, and prompt-driven fine-tuning. Specifically, the node encoding component employs graph neural networks to learn local structural patterns of nodes in a graph, thereby obtaining representations for nodes and communities. Next, the sample generation component identifies an initial community for a given node and selects known communities that are structurally similar to the initial one as training samples. Finally, the prompt-driven fine-tuning component leverages these samples as prompts to guide the final community prediction. Experimental results on five real-world datasets demonstrate that PPCS performs better than baseline algorithms. It also achieves higher community search efficiency than semi-supervised community search baseline methods, with ablation studies verifying the effectiveness of each component of PPCS. 

---
# Enhance Mobile Agents Thinking Process Via Iterative Preference Learning 

**Authors**: Kun Huang, Weikai Xu, Yuxuan Liu, Quandong Wang, Pengzhi Gao, Wei Liu, Jian Luan, Bin Wang, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2505.12299)  

**Abstract**: The Chain of Action-Planning Thoughts (CoaT) paradigm has been shown to improve the reasoning performance of VLM-based mobile agents in GUI tasks. However, the scarcity of diverse CoaT trajectories limits the expressiveness and generalization ability of such agents. While self-training is commonly employed to address data scarcity, existing approaches either overlook the correctness of intermediate reasoning steps or depend on expensive process-level annotations to construct process reward models (PRM). To address the above problems, we propose an Iterative Preference Learning (IPL) that constructs a CoaT-tree through interative sampling, scores leaf nodes using rule-based reward, and backpropagates feedback to derive Thinking-level Direct Preference Optimization (T-DPO) pairs. To prevent overfitting during warm-up supervised fine-tuning, we further introduce a three-stage instruction evolution, which leverages GPT-4o to generate diverse Q\&A pairs based on real mobile UI screenshots, enhancing both generality and layout understanding. Experiments on three standard Mobile GUI-agent benchmarks demonstrate that our agent MobileIPL outperforms strong baselines, including continual pretraining models such as OS-ATLAS and UI-TARS. It achieves state-of-the-art performance across three standard Mobile GUI-Agents benchmarks and shows strong generalization to out-of-domain scenarios. 

---
# Attention-Enhanced U-Net for Accurate Segmentation of COVID-19 Infected Lung Regions in CT Scans 

**Authors**: Amal Lahchim, Lazar Davic  

**Link**: [PDF](https://arxiv.org/pdf/2505.12298)  

**Abstract**: In this study, we propose a robust methodology for automatic segmentation of infected lung regions in COVID-19 CT scans using convolutional neural networks. The approach is based on a modified U-Net architecture enhanced with attention mechanisms, data augmentation, and postprocessing techniques. It achieved a Dice coefficient of 0.8658 and mean IoU of 0.8316, outperforming other methods. The dataset was sourced from public repositories and augmented for diversity. Results demonstrate superior segmentation performance. Future work includes expanding the dataset, exploring 3D segmentation, and preparing the model for clinical deployment. 

---
# PoLO: Proof-of-Learning and Proof-of-Ownership at Once with Chained Watermarking 

**Authors**: Haiyu Deng, Yanna Jiang, Guangsheng Yu, Qin Wang, Xu Wang, Baihe Ma, Wei Ni, Ren Ping Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12296)  

**Abstract**: Machine learning models are increasingly shared and outsourced, raising requirements of verifying training effort (Proof-of-Learning, PoL) to ensure claimed performance and establishing ownership (Proof-of-Ownership, PoO) for transactions. When models are trained by untrusted parties, PoL and PoO must be enforced together to enable protection, attribution, and compensation. However, existing studies typically address them separately, which not only weakens protection against forgery and privacy breaches but also leads to high verification overhead.
We propose PoLO, a unified framework that simultaneously achieves PoL and PoO using chained watermarks. PoLO splits the training process into fine-grained training shards and embeds a dedicated watermark in each shard. Each watermark is generated using the hash of the preceding shard, certifying the training process of the preceding shard. The chained structure makes it computationally difficult to forge any individual part of the whole training process. The complete set of watermarks serves as the PoL, while the final watermark provides the PoO. PoLO offers more efficient and privacy-preserving verification compared to the vanilla PoL solutions that rely on gradient-based trajectory tracing and inadvertently expose training data during verification, while maintaining the same level of ownership assurance of watermark-based PoO schemes. Our evaluation shows that PoLO achieves 99% watermark detection accuracy for ownership verification, while preserving data privacy and cutting verification costs to just 1.5-10% of traditional methods. Forging PoLO demands 1.1-4x more resources than honest proof generation, with the original proof retaining over 90% detection accuracy even after attacks. 

---
# SpikeX: Exploring Accelerator Architecture and Network-Hardware Co-Optimization for Sparse Spiking Neural Networks 

**Authors**: Boxun Xu, Richard Boone, Peng Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.12292)  

**Abstract**: Spiking Neural Networks (SNNs) are promising biologically plausible models of computation which utilize a spiking binary activation function similar to that of biological neurons. SNNs are well positioned to process spatiotemporal data, and are advantageous in ultra-low power and real-time processing. Despite a large body of work on conventional artificial neural network accelerators, much less attention has been given to efficient SNN hardware accelerator design. In particular, SNNs exhibit inherent unstructured spatial and temporal firing sparsity, an opportunity yet to be fully explored for great hardware processing efficiency. In this work, we propose a novel systolic-array SNN accelerator architecture, called SpikeX, to take on the challenges and opportunities stemming from unstructured sparsity while taking into account the unique characteristics of spike-based computation. By developing an efficient dataflow targeting expensive multi-bit weight data movements, SpikeX reduces memory access and increases data sharing and hardware utilization for computations spanning across both time and space, thereby significantly improving energy efficiency and inference latency. Furthermore, recognizing the importance of SNN network and hardware co-design, we develop a co-optimization methodology facilitating not only hardware-aware SNN training but also hardware accelerator architecture search, allowing joint network weight parameter optimization and accelerator architectural reconfiguration. This end-to-end network/accelerator co-design approach offers a significant reduction of 15.1x-150.87x in energy-delay-product(EDP) without comprising model accuracy. 

---
# The Tower of Babel Revisited: Multilingual Jailbreak Prompts on Closed-Source Large Language Models 

**Authors**: Linghan Huang, Haolin Jin, Zhaoge Bi, Pengyue Yang, Peizhou Zhao, Taozhao Chen, Xiongfei Wu, Lei Ma, Huaming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.12287)  

**Abstract**: Large language models (LLMs) have seen widespread applications across various domains, yet remain vulnerable to adversarial prompt injections. While most existing research on jailbreak attacks and hallucination phenomena has focused primarily on open-source models, we investigate the frontier of closed-source LLMs under multilingual attack scenarios. We present a first-of-its-kind integrated adversarial framework that leverages diverse attack techniques to systematically evaluate frontier proprietary solutions, including GPT-4o, DeepSeek-R1, Gemini-1.5-Pro, and Qwen-Max. Our evaluation spans six categories of security contents in both English and Chinese, generating 38,400 responses across 32 types of jailbreak attacks. Attack success rate (ASR) is utilized as the quantitative metric to assess performance from three dimensions: prompt design, model architecture, and language environment. Our findings suggest that Qwen-Max is the most vulnerable, while GPT-4o shows the strongest defense. Notably, prompts in Chinese consistently yield higher ASRs than their English counterparts, and our novel Two-Sides attack technique proves to be the most effective across all models. This work highlights a dire need for language-aware alignment and robust cross-lingual defenses in LLMs, and we hope it will inspire researchers, developers, and policymakers toward more robust and inclusive AI systems. 

---
# Curriculum Abductive Learning 

**Authors**: Wen-Chao Hu, Qi-Jie Li, Lin-Han Jia, Cunjing Ge, Yu-Feng Li, Yuan Jiang, Zhi-Hua Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.12275)  

**Abstract**: Abductive Learning (ABL) integrates machine learning with logical reasoning in a loop: a learning model predicts symbolic concept labels from raw inputs, which are revised through abduction using domain knowledge and then fed back for retraining. However, due to the nondeterminism of abduction, the training process often suffers from instability, especially when the knowledge base is large and complex, resulting in a prohibitively large abduction space. While prior works focus on improving candidate selection within this space, they typically treat the knowledge base as a static black box. In this work, we propose Curriculum Abductive Learning (C-ABL), a method that explicitly leverages the internal structure of the knowledge base to address the ABL training challenges. C-ABL partitions the knowledge base into a sequence of sub-bases, progressively introduced during training. This reduces the abduction space throughout training and enables the model to incorporate logic in a stepwise, smooth way. Experiments across multiple tasks show that C-ABL outperforms previous ABL implementations, significantly improves training stability, convergence speed, and final accuracy, especially under complex knowledge setting. 

---
# Vague Knowledge: Evidence from Analyst Reports 

**Authors**: Kerry Xiao, Amy Zang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12269)  

**Abstract**: People in the real world often possess vague knowledge of future payoffs, for which quantification is not feasible or desirable. We argue that language, with differing ability to convey vague information, plays an important but less known-role in subjective expectations. Empirically, we find that in their reports, analysts include useful information in linguistic expressions but not numerical forecasts. Specifically, the textual tone of analyst reports has predictive power for forecast errors and subsequent revisions in numerical forecasts, and this relation becomes stronger when analyst's language is vaguer, when uncertainty is higher, and when analysts are busier. Overall, our theory and evidence suggest that some useful information is vaguely known and only communicated through language. 

---
# LightRetriever: A LLM-based Hybrid Retrieval Architecture with 1000x Faster Query Inference 

**Authors**: Guangyuan Ma, Yongliang Ma, Xuanrui Gou, Zhenpeng Su, Ming Zhou, Songlin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12260)  

**Abstract**: Large Language Models (LLMs)-based hybrid retrieval uses LLMs to encode queries and documents into low-dimensional dense or high-dimensional sparse vectors. It retrieves documents relevant to search queries based on vector similarities. Documents are pre-encoded offline, while queries arrive in real-time, necessitating an efficient online query encoder. Although LLMs significantly enhance retrieval capabilities, serving deeply parameterized LLMs slows down query inference throughput and increases demands for online deployment resources. In this paper, we propose LightRetriever, a novel LLM-based hybrid retriever with extremely lightweight query encoders. Our method retains a full-sized LLM for document encoding, but reduces the workload of query encoding to no more than an embedding lookup. Compared to serving a full-sized LLM on an H800 GPU, our approach achieves over a 1000x speedup for query inference with GPU acceleration, and even a 20x speedup without GPU. Experiments on large-scale retrieval benchmarks demonstrate that our method generalizes well across diverse retrieval tasks, retaining an average of 95% full-sized performance. 

---
# MMS-VPR: Multimodal Street-Level Visual Place Recognition Dataset and Benchmark 

**Authors**: Yiwei Ou, Xiaobin Ren, Ronggui Sun, Guansong Gao, Ziyi Jiang, Kaiqi Zhao, Manfredo Manfredini  

**Link**: [PDF](https://arxiv.org/pdf/2505.12254)  

**Abstract**: Existing visual place recognition (VPR) datasets predominantly rely on vehicle-mounted imagery, lack multimodal diversity and underrepresent dense, mixed-use street-level spaces, especially in non-Western urban contexts. To address these gaps, we introduce MMS-VPR, a large-scale multimodal dataset for street-level place recognition in complex, pedestrian-only environments. The dataset comprises 78,575 annotated images and 2,512 video clips captured across 207 locations in a ~70,800 $\mathrm{m}^2$ open-air commercial district in Chengdu, China. Each image is labeled with precise GPS coordinates, timestamp, and textual metadata, and covers varied lighting conditions, viewpoints, and timeframes. MMS-VPR follows a systematic and replicable data collection protocol with minimal device requirements, lowering the barrier for scalable dataset creation. Importantly, the dataset forms an inherent spatial graph with 125 edges, 81 nodes, and 1 subgraph, enabling structure-aware place recognition. We further define two application-specific subsets -- Dataset_Edges and Dataset_Points -- to support fine-grained and graph-based evaluation tasks. Extensive benchmarks using conventional VPR models, graph neural networks, and multimodal baselines show substantial improvements when leveraging multimodal and structural cues. MMS-VPR facilitates future research at the intersection of computer vision, geospatial understanding, and multimodal reasoning. The dataset is publicly available at this https URL. 

---
# Not All Documents Are What You Need for Extracting Instruction Tuning Data 

**Authors**: Chi Zhang, Huaping Zhong, Hongtao Li, Chengliang Chai, Jiawei Hong, Yuhao Deng, Jiacheng Wang, Tian Tan, Yizhou Yan, Jiantao Qiu, Ye Yuan, Guoren Wang, Conghui He, Lei Cao  

**Link**: [PDF](https://arxiv.org/pdf/2505.12250)  

**Abstract**: Instruction tuning improves the performance of large language models (LLMs), but it heavily relies on high-quality training data. Recently, LLMs have been used to synthesize instruction data using seed question-answer (QA) pairs. However, these synthesized instructions often lack diversity and tend to be similar to the input seeds, limiting their applicability in real-world scenarios. To address this, we propose extracting instruction tuning data from web corpora that contain rich and diverse knowledge. A naive solution is to retrieve domain-specific documents and extract all QA pairs from them, but this faces two key challenges: (1) extracting all QA pairs using LLMs is prohibitively expensive, and (2) many extracted QA pairs may be irrelevant to the downstream tasks, potentially degrading model performance. To tackle these issues, we introduce EQUAL, an effective and scalable data extraction framework that iteratively alternates between document selection and high-quality QA pair extraction to enhance instruction tuning. EQUAL first clusters the document corpus based on embeddings derived from contrastive learning, then uses a multi-armed bandit strategy to efficiently identify clusters that are likely to contain valuable QA pairs. This iterative approach significantly reduces computational cost while boosting model performance. Experiments on AutoMathText and StackOverflow across four downstream tasks show that EQUAL reduces computational costs by 5-10x and improves accuracy by 2.5 percent on LLaMA-3.1-8B and Mistral-7B 

---
# LAMeTA: Intent-Aware Agentic Network Optimization via a Large AI Model-Empowered Two-Stage Approach 

**Authors**: Yinqiu Liu, Guangyuan Liu, Jiacheng Wang, Ruichen Zhang, Dusit Niyato, Geng Sun, Zehui Xiong, Zhu Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.12247)  

**Abstract**: Nowadays, Generative AI (GenAI) reshapes numerous domains by enabling machines to create content across modalities. As GenAI evolves into autonomous agents capable of reasoning, collaboration, and interaction, they are increasingly deployed on network infrastructures to serve humans automatically. This emerging paradigm, known as the agentic network, presents new optimization challenges due to the demand to incorporate subjective intents of human users expressed in natural language. Traditional generic Deep Reinforcement Learning (DRL) struggles to capture intent semantics and adjust policies dynamically, thus leading to suboptimality. In this paper, we present LAMeTA, a Large AI Model (LAM)-empowered Two-stage Approach for intent-aware agentic network optimization. First, we propose Intent-oriented Knowledge Distillation (IoKD), which efficiently distills intent-understanding capabilities from resource-intensive LAMs to lightweight edge LAMs (E-LAMs) to serve end users. Second, we develop Symbiotic Reinforcement Learning (SRL), integrating E-LAMs with a policy-based DRL framework. In SRL, E-LAMs translate natural language user intents into structured preference vectors that guide both state representation and reward design. The DRL, in turn, optimizes the generative service function chain composition and E-LAM selection based on real-time network conditions, thus optimizing the subjective Quality-of-Experience (QoE). Extensive experiments conducted in an agentic network with 81 agents demonstrate that IoKD reduces mean squared error in intent prediction by up to 22.5%, while SRL outperforms conventional generic DRL by up to 23.5% in maximizing intent-aware QoE. 

---
# AFCL: Analytic Federated Continual Learning for Spatio-Temporal Invariance of Non-IID Data 

**Authors**: Jianheng Tang, Huiping Zhuang, Jingyu He, Run He, Jingchao Wang, Kejia Fan, Anfeng Liu, Tian Wang, Leye Wang, Zhanxing Zhu, Shanghang Zhang, Houbing Herbert Song, Yunhuai Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12245)  

**Abstract**: Federated Continual Learning (FCL) enables distributed clients to collaboratively train a global model from online task streams in dynamic real-world scenarios. However, existing FCL methods face challenges of both spatial data heterogeneity among distributed clients and temporal data heterogeneity across online tasks. Such data heterogeneity significantly degrades the model performance with severe spatial-temporal catastrophic forgetting of local and past knowledge. In this paper, we identify that the root cause of this issue lies in the inherent vulnerability and sensitivity of gradients to non-IID data. To fundamentally address this issue, we propose a gradient-free method, named Analytic Federated Continual Learning (AFCL), by deriving analytical (i.e., closed-form) solutions from frozen extracted features. In local training, our AFCL enables single-epoch learning with only a lightweight forward-propagation process for each client. In global aggregation, the server can recursively and efficiently update the global model with single-round aggregation. Theoretical analyses validate that our AFCL achieves spatio-temporal invariance of non-IID data. This ideal property implies that, regardless of how heterogeneous the data are distributed across local clients and online tasks, the aggregated model of our AFCL remains invariant and identical to that of centralized joint learning. Extensive experiments show the consistent superiority of our AFCL over state-of-the-art baselines across various benchmark datasets and settings. 

---
# ACU: Analytic Continual Unlearning for Efficient and Exact Forgetting with Privacy Preservation 

**Authors**: Jianheng Tang, Huiping Zhuang, Di Fang, Jiaxu Li, Feijiang Han, Yajiang Huang, Kejia Fan, Leye Wang, Zhanxing Zhu, Shanghang Zhang, Houbing Herbert Song, Yunhuai Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12239)  

**Abstract**: The development of artificial intelligence demands that models incrementally update knowledge by Continual Learning (CL) to adapt to open-world environments. To meet privacy and security requirements, Continual Unlearning (CU) emerges as an important problem, aiming to sequentially forget particular knowledge acquired during the CL phase. However, existing unlearning methods primarily focus on single-shot joint forgetting and face significant limitations when applied to CU. First, most existing methods require access to the retained dataset for re-training or fine-tuning, violating the inherent constraint in CL that historical data cannot be revisited. Second, these methods often suffer from a poor trade-off between system efficiency and model fidelity, making them vulnerable to being overwhelmed or degraded by adversaries through deliberately frequent requests. In this paper, we identify that the limitations of existing unlearning methods stem fundamentally from their reliance on gradient-based updates. To bridge the research gap at its root, we propose a novel gradient-free method for CU, named Analytic Continual Unlearning (ACU), for efficient and exact forgetting with historical data privacy preservation. In response to each unlearning request, our ACU recursively derives an analytical (i.e., closed-form) solution in an interpretable manner using the least squares method. Theoretical and experimental evaluations validate the superiority of our ACU on unlearning effectiveness, model fidelity, and system efficiency. 

---
# PANORAMA: A synthetic PII-laced dataset for studying sensitive data memorization in LLMs 

**Authors**: Sriram Selvam, Anneswa Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2505.12238)  

**Abstract**: The memorization of sensitive and personally identifiable information (PII) by large language models (LLMs) poses growing privacy risks as models scale and are increasingly deployed in real-world applications. Existing efforts to study sensitive and PII data memorization and develop mitigation strategies are hampered by the absence of comprehensive, realistic, and ethically sourced datasets reflecting the diversity of sensitive information found on the web. We introduce PANORAMA - Profile-based Assemblage for Naturalistic Online Representation and Attribute Memorization Analysis, a large-scale synthetic corpus of 384,789 samples derived from 9,674 synthetic profiles designed to closely emulate the distribution, variety, and context of PII and sensitive data as it naturally occurs in online environments. Our data generation pipeline begins with the construction of internally consistent, multi-attribute human profiles using constrained selection to reflect real-world demographics such as education, health attributes, financial status, etc. Using a combination of zero-shot prompting and OpenAI o3-mini, we generate diverse content types - including wiki-style articles, social media posts, forum discussions, online reviews, comments, and marketplace listings - each embedding realistic, contextually appropriate PII and other sensitive information. We validate the utility of PANORAMA by fine-tuning the Mistral-7B model on 1x, 5x, 10x, and 25x data replication rates with a subset of data and measure PII memorization rates - revealing not only consistent increases with repetition but also variation across content types, highlighting PANORAMA's ability to model how memorization risks differ by context. Our dataset and code are publicly available, providing a much-needed resource for privacy risk assessment, model auditing, and the development of privacy-preserving LLMs. 

---
# Bridging Generative and Discriminative Learning: Few-Shot Relation Extraction via Two-Stage Knowledge-Guided Pre-training 

**Authors**: Quanjiang Guo, Jinchuan Zhang, Sijie Wang, Ling Tian, Zhao Kang, Bin Yan, Weidong Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2505.12236)  

**Abstract**: Few-Shot Relation Extraction (FSRE) remains a challenging task due to the scarcity of annotated data and the limited generalization capabilities of existing models. Although large language models (LLMs) have demonstrated potential in FSRE through in-context learning (ICL), their general-purpose training objectives often result in suboptimal performance for task-specific relation extraction. To overcome these challenges, we propose TKRE (Two-Stage Knowledge-Guided Pre-training for Relation Extraction), a novel framework that synergistically integrates LLMs with traditional relation extraction models, bridging generative and discriminative learning paradigms. TKRE introduces two key innovations: (1) leveraging LLMs to generate explanation-driven knowledge and schema-constrained synthetic data, addressing the issue of data scarcity; and (2) a two-stage pre-training strategy combining Masked Span Language Modeling (MSLM) and Span-Level Contrastive Learning (SCL) to enhance relational reasoning and generalization. Together, these components enable TKRE to effectively tackle FSRE tasks. Comprehensive experiments on benchmark datasets demonstrate the efficacy of TKRE, achieving new state-of-the-art performance in FSRE and underscoring its potential for broader application in low-resource scenarios. \footnote{The code and data are released on this https URL. 

---
# Shallow Flow Matching for Coarse-to-Fine Text-to-Speech Synthesis 

**Authors**: Dong Yang, Yiyi Cai, Yuki Saito, Lixu Wang, Hiroshi Saruwatari  

**Link**: [PDF](https://arxiv.org/pdf/2505.12226)  

**Abstract**: We propose a shallow flow matching (SFM) mechanism to enhance flow matching (FM)-based text-to-speech (TTS) models within a coarse-to-fine generation paradigm. SFM constructs intermediate states along the FM paths using coarse output representations. During training, we introduce an orthogonal projection method to adaptively determine the temporal position of these states, and apply a principled construction strategy based on a single-segment piecewise flow. The SFM inference starts from the intermediate state rather than pure noise and focuses computation on the latter stages of the FM paths. We integrate SFM into multiple TTS models with a lightweight SFM head. Experiments show that SFM consistently improves the naturalness of synthesized speech in both objective and subjective evaluations, while significantly reducing inference when using adaptive-step ODE solvers. Demo and codes are available at this https URL. 

---
# Reward Inside the Model: A Lightweight Hidden-State Reward Model for LLM's Best-of-N sampling 

**Authors**: Jizhou Guo, Zhaomin Wu, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12225)  

**Abstract**: High-quality reward models are crucial for unlocking the reasoning potential of large language models (LLMs), with best-of-N voting demonstrating significant performance gains. However, current reward models, which typically operate on the textual output of LLMs, are computationally expensive and parameter-heavy, limiting their real-world applications. We introduce the Efficient Linear Hidden State Reward (ELHSR) model - a novel, highly parameter-efficient approach that leverages the rich information embedded in LLM hidden states to address these issues. ELHSR systematically outperform baselines with less than 0.005% of the parameters of baselines, requiring only a few samples for training. ELHSR also achieves orders-of-magnitude efficiency improvement with significantly less time and fewer FLOPs per sample than baseline reward models. Moreover, ELHSR exhibits robust performance even when trained only on logits, extending its applicability to some closed-source LLMs. In addition, ELHSR can also be combined with traditional reward models to achieve additional performance gains. 

---
# RoboFAC: A Comprehensive Framework for Robotic Failure Analysis and Correction 

**Authors**: Weifeng Lu, Minghao Ye, Zewei Ye, Ruihan Tao, Shuo Yang, Bo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.12224)  

**Abstract**: Vision-Language-Action (VLA) models have recently advanced robotic manipulation by translating natural-language instructions and image information into sequential control actions. However, these models often underperform in open-world scenarios, as they are predominantly trained on successful expert demonstrations and exhibit a limited capacity for failure recovery. In this work, we present a Robotic Failure Analysis and Correction (RoboFAC) framework to address this issue. Firstly, we construct RoboFAC dataset comprising 9,440 erroneous manipulation trajectories and 78,623 QA pairs across 16 diverse tasks and 53 scenes in both simulation and real-world environments. Leveraging our dataset, we develop RoboFAC model, which is capable of Task Understanding, Failure Analysis and Failure Correction. Experimental results demonstrate that the RoboFAC model outperforms GPT-4o by 34.1% on our evaluation benchmark. Furthermore, we integrate the RoboFAC model into a real-world VLA control pipeline as an external supervision providing correction instructions, yielding a 29.1% relative improvement on average on four real-world tasks. The results show that our RoboFAC framework effectively handles robotic failures and assists the VLA model in recovering from failures. 

---
# Imagination-Limited Q-Learning for Offline Reinforcement Learning 

**Authors**: Wenhui Liu, Zhijian Wu, Jingchao Wang, Dingjiang Huang, Shuigeng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.12211)  

**Abstract**: Offline reinforcement learning seeks to derive improved policies entirely from historical data but often struggles with over-optimistic value estimates for out-of-distribution (OOD) actions. This issue is typically mitigated via policy constraint or conservative value regularization methods. However, these approaches may impose overly constraints or biased value estimates, potentially limiting performance improvements. To balance exploitation and restriction, we propose an Imagination-Limited Q-learning (ILQ) method, which aims to maintain the optimism that OOD actions deserve within appropriate limits. Specifically, we utilize the dynamics model to imagine OOD action-values, and then clip the imagined values with the maximum behavior values. Such design maintains reasonable evaluation of OOD actions to the furthest extent, while avoiding its over-optimism. Theoretically, we prove the convergence of the proposed ILQ under tabular Markov decision processes. Particularly, we demonstrate that the error bound between estimated values and optimality values of OOD state-actions possesses the same magnitude as that of in-distribution ones, thereby indicating that the bias in value estimates is effectively mitigated. Empirically, our method achieves state-of-the-art performance on a wide range of tasks in the D4RL benchmark. 

---
# Can Large Multimodal Models Understand Agricultural Scenes? Benchmarking with AgroMind 

**Authors**: Qingmei Li, Yang Zhang, Zurong Mai, Yuhang Chen, Shuohong Lou, Henglian Huang, Jiarui Zhang, Zhiwei Zhang, Yibin Wen, Weijia Li, Haohuan Fu, Jianxi Huang, Juepeng Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.12207)  

**Abstract**: Large Multimodal Models (LMMs) has demonstrated capabilities across various domains, but comprehensive benchmarks for agricultural remote sensing (RS) remain scarce. Existing benchmarks designed for agricultural RS scenarios exhibit notable limitations, primarily in terms of insufficient scene diversity in the dataset and oversimplified task design. To bridge this gap, we introduce AgroMind, a comprehensive agricultural remote sensing benchmark covering four task dimensions: spatial perception, object understanding, scene understanding, and scene reasoning, with a total of 13 task types, ranging from crop identification and health monitoring to environmental analysis. We curate a high-quality evaluation set by integrating eight public datasets and one private farmland plot dataset, containing 25,026 QA pairs and 15,556 images. The pipeline begins with multi-source data preprocessing, including collection, format standardization, and annotation refinement. We then generate a diverse set of agriculturally relevant questions through the systematic definition of tasks. Finally, we employ LMMs for inference, generating responses, and performing detailed examinations. We evaluated 18 open-source LMMs and 3 closed-source models on AgroMind. Experiments reveal significant performance gaps, particularly in spatial reasoning and fine-grained recognition, it is notable that human performance lags behind several leading LMMs. By establishing a standardized evaluation framework for agricultural RS, AgroMind reveals the limitations of LMMs in domain knowledge and highlights critical challenges for future work. Data and code can be accessed at this https URL. 

---
# Always Clear Depth: Robust Monocular Depth Estimation under Adverse Weather 

**Authors**: Kui Jiang, Jing Cao, Zhaocheng Yu, Junjun Jiang, Jingchun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.12199)  

**Abstract**: Monocular depth estimation is critical for applications such as autonomous driving and scene reconstruction. While existing methods perform well under normal scenarios, their performance declines in adverse weather, due to challenging domain shifts and difficulties in extracting scene information. To address this issue, we present a robust monocular depth estimation method called \textbf{ACDepth} from the perspective of high-quality training data generation and domain adaptation. Specifically, we introduce a one-step diffusion model for generating samples that simulate adverse weather conditions, constructing a multi-tuple degradation dataset during training. To ensure the quality of the generated degradation samples, we employ LoRA adapters to fine-tune the generation weights of diffusion model. Additionally, we integrate circular consistency loss and adversarial training to guarantee the fidelity and naturalness of the scene contents. Furthermore, we elaborate on a multi-granularity knowledge distillation strategy (MKD) that encourages the student network to absorb knowledge from both the teacher model and pretrained Depth Anything V2. This strategy guides the student model in learning degradation-agnostic scene information from various degradation inputs. In particular, we introduce an ordinal guidance distillation mechanism (OGD) that encourages the network to focus on uncertain regions through differential ranking, leading to a more precise depth estimation. Experimental results demonstrate that our ACDepth surpasses md4all-DD by 2.50\% for night scene and 2.61\% for rainy scene on the nuScenes dataset in terms of the absRel metric. 

---
# Ditch the Denoiser: Emergence of Noise Robustness in Self-Supervised Learning from Data Curriculum 

**Authors**: Wenquan Lu, Jiaqi Zhang, Hugues Van Assel, Randall Balestriero  

**Link**: [PDF](https://arxiv.org/pdf/2505.12191)  

**Abstract**: Self-Supervised Learning (SSL) has become a powerful solution to extract rich representations from unlabeled data. Yet, SSL research is mostly focused on clean, curated and high-quality datasets. As a result, applying SSL on noisy data remains a challenge, despite being crucial to applications such as astrophysics, medical imaging, geophysics or finance. In this work, we present a fully self-supervised framework that enables noise-robust representation learning without requiring a denoiser at inference or downstream fine-tuning. Our method first trains an SSL denoiser on noisy data, then uses it to construct a denoised-to-noisy data curriculum (i.e., training first on denoised, then noisy samples) for pretraining a SSL backbone (e.g., DINOv2), combined with a teacher-guided regularization that anchors noisy embeddings to their denoised counterparts. This process encourages the model to internalize noise robustness. Notably, the denoiser can be discarded after pretraining, simplifying deployment. On ImageNet-1k with ViT-B under extreme Gaussian noise ($\sigma=255$, SNR = 0.72 dB), our method improves linear probing accuracy by 4.8% over DINOv2, demonstrating that denoiser-free robustness can emerge from noise-aware pretraining. The code is available at this https URL. 

---
# LLM-DSE: Searching Accelerator Parameters with LLM Agents 

**Authors**: Hanyu Wang, Xinrui Wu, Zijian Ding, Su Zheng, Chengyue Wang, Tony Nowatzki, Yizhou Sun, Jason Cong  

**Link**: [PDF](https://arxiv.org/pdf/2505.12188)  

**Abstract**: Even though high-level synthesis (HLS) tools mitigate the challenges of programming domain-specific accelerators (DSAs) by raising the abstraction level, optimizing hardware directive parameters remains a significant hurdle. Existing heuristic and learning-based methods struggle with adaptability and sample this http URL present LLM-DSE, a multi-agent framework designed specifically for optimizing HLS directives. Combining LLM with design space exploration (DSE), our explorer coordinates four agents: Router, Specialists, Arbitrator, and Critic. These multi-agent components interact with various tools to accelerate the optimization process. LLM-DSE leverages essential domain knowledge to identify efficient parameter combinations while maintaining adaptability through verbal learning from online interactions. Evaluations on the HLSyn dataset demonstrate that LLM-DSE achieves substantial $2.55\times$ performance gains over state-of-the-art methods, uncovering novel designs while reducing runtime. Ablation studies validate the effectiveness and necessity of the proposed agent interactions. Our code is open-sourced here: this https URL. 

---
# Self-Destructive Language Model 

**Authors**: Yuhui Wang, Rongyi Zhu, Ting Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12186)  

**Abstract**: Harmful fine-tuning attacks pose a major threat to the security of large language models (LLMs), allowing adversaries to compromise safety guardrails with minimal harmful data. While existing defenses attempt to reinforce LLM alignment, they fail to address models' inherent "trainability" on harmful data, leaving them vulnerable to stronger attacks with increased learning rates or larger harmful datasets. To overcome this critical limitation, we introduce SEAM, a novel alignment-enhancing defense that transforms LLMs into self-destructive models with intrinsic resilience to misalignment attempts. Specifically, these models retain their capabilities for legitimate tasks while exhibiting substantial performance degradation when fine-tuned on harmful data. The protection is achieved through a novel loss function that couples the optimization trajectories of benign and harmful data, enhanced with adversarial gradient ascent to amplify the self-destructive effect. To enable practical training, we develop an efficient Hessian-free gradient estimate with theoretical error bounds. Extensive evaluation across LLMs and datasets demonstrates that SEAM creates a no-win situation for adversaries: the self-destructive models achieve state-of-the-art robustness against low-intensity attacks and undergo catastrophic performance collapse under high-intensity attacks, rendering them effectively unusable. (warning: this paper contains potentially harmful content generated by LLMs.) 

---
# Decoding the Mind of Large Language Models: A Quantitative Evaluation of Ideology and Biases 

**Authors**: Manari Hirose, Masato Uchida  

**Link**: [PDF](https://arxiv.org/pdf/2505.12183)  

**Abstract**: The widespread integration of Large Language Models (LLMs) across various sectors has highlighted the need for empirical research to understand their biases, thought patterns, and societal implications to ensure ethical and effective use. In this study, we propose a novel framework for evaluating LLMs, focusing on uncovering their ideological biases through a quantitative analysis of 436 binary-choice questions, many of which have no definitive answer. By applying our framework to ChatGPT and Gemini, findings revealed that while LLMs generally maintain consistent opinions on many topics, their ideologies differ across models and languages. Notably, ChatGPT exhibits a tendency to change their opinion to match the questioner's opinion. Both models also exhibited problematic biases, unethical or unfair claims, which might have negative societal impacts. These results underscore the importance of addressing both ideological and ethical considerations when evaluating LLMs. The proposed framework offers a flexible, quantitative method for assessing LLM behavior, providing valuable insights for the development of more socially aligned AI systems. 

---
# SoftPQ: Robust Instance Segmentation Evaluation via Soft Matching and Tunable Thresholds 

**Authors**: Ranit Karmakar, Simon F. Nørrelykke  

**Link**: [PDF](https://arxiv.org/pdf/2505.12155)  

**Abstract**: Segmentation evaluation metrics traditionally rely on binary decision logic: predictions are either correct or incorrect, based on rigid IoU thresholds. Detection--based metrics such as F1 and mAP determine correctness at the object level using fixed overlap cutoffs, while overlap--based metrics like Intersection over Union (IoU) and Dice operate at the pixel level, often overlooking instance--level structure. Panoptic Quality (PQ) attempts to unify detection and segmentation assessment, but it remains dependent on hard-threshold matching--treating predictions below the threshold as entirely incorrect. This binary framing obscures important distinctions between qualitatively different errors and fails to reward gradual model improvements. We propose SoftPQ, a flexible and interpretable instance segmentation metric that redefines evaluation as a graded continuum rather than a binary classification. SoftPQ introduces tunable upper and lower IoU thresholds to define a partial matching region and applies a sublinear penalty function to ambiguous or fragmented predictions. These extensions allow SoftPQ to exhibit smoother score behavior, greater robustness to structural segmentation errors, and more informative feedback for model development and evaluation. Through controlled perturbation experiments, we show that SoftPQ captures meaningful differences in segmentation quality that existing metrics overlook, making it a practical and principled alternative for both benchmarking and iterative model refinement. 

---
# Reasoning Large Language Model Errors Arise from Hallucinating Critical Problem Features 

**Authors**: Alex Heyman, Joel Zylberberg  

**Link**: [PDF](https://arxiv.org/pdf/2505.12151)  

**Abstract**: Large language models have recently made great strides in reasoning task performance through chain-of-thought (CoT) strategies trained via reinforcement learning; however, these "reasoning large language models" (RLLMs) remain imperfect reasoners, and understanding the frequencies and causes of their failure modes is important for both users and developers. We test o1-mini, o3-mini, DeepSeek-R1, Claude 3.7 Sonnet, Gemini 2.5 Pro Preview, and Grok 3 Mini Beta on graph coloring as a variable-complexity constraint-satisfaction logic problem, and find evidence from both error rate comparisons and CoT/explanation text analysis that RLLMs are prone to hallucinate edges not specified in the prompt's description of the graph. This phenomenon persists across multiple problem complexity levels and semantic frames, and it appears to account for a significant fraction of the incorrect answers from every tested model, and the vast majority of them for some models. Our results indicate that RLLMs may possess broader issues with misrepresentation of problem specifics, and we offer suggestions for design choices to mitigate this weakness. 

---
# Structured Representation 

**Authors**: Arun Kumar, Paul Schrater  

**Link**: [PDF](https://arxiv.org/pdf/2505.12143)  

**Abstract**: Invariant representations are core to representation learning, yet a central challenge remains: uncovering invariants that are stable and transferable without suppressing task-relevant signals. This raises fundamental questions, requiring further inquiry, about the appropriate level of abstraction at which such invariants should be defined, and which aspects of a system they should characterize. Interpretation of the environment relies on abstract knowledge structures to make sense of the current state, which leads to interactions, essential drivers of learning and knowledge acquisition. We posit that interpretation operates at the level of higher-order relational knowledge; hence, invariant structures must be where knowledge resides, specifically, as partitions defined by the closure of relational paths within an abstract knowledge space. These partitions serve as the core invariant representations, forming the structural substrate where knowledge is stored and learning occurs. On the other hand, inter-partition connectors enable the deployment of these knowledge partitions encoding task-relevant transitions. Thus, invariant partitions provide the foundational primitives of structured representation. We formalize the computational foundations for structured representation of the invariant partitions based on closed semiring, a relational algebraic structure. 

---
# Keypoints as Dynamic Centroids for Unified Human Pose and Segmentation 

**Authors**: Niaz Ahmad, Jawad Khan, Kang G. Shin, Youngmoon Lee, Guanghui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12130)  

**Abstract**: The dynamic movement of the human body presents a fundamental challenge for human pose estimation and body segmentation. State-of-the-art approaches primarily rely on combining keypoint heatmaps with segmentation masks but often struggle in scenarios involving overlapping joints or rapidly changing poses during instance-level segmentation. To address these limitations, we propose Keypoints as Dynamic Centroid (KDC), a new centroid-based representation for unified human pose estimation and instance-level segmentation. KDC adopts a bottom-up paradigm to generate keypoint heatmaps for both easily distinguishable and complex keypoints and improves keypoint detection and confidence scores by introducing KeyCentroids using a keypoint disk. It leverages high-confidence keypoints as dynamic centroids in the embedding space to generate MaskCentroids, allowing for swift clustering of pixels to specific human instances during rapid body movements in live environments. Our experimental evaluations on the CrowdPose, OCHuman, and COCO benchmarks demonstrate KDC's effectiveness and generalizability in challenging scenarios in terms of both accuracy and runtime performance. The implementation is available at: this https URL. 

---
# SAINT: Attention-Based Modeling of Sub-Action Dependencies in Multi-Action Policies 

**Authors**: Matthew Landers, Taylor W. Killian, Thomas Hartvigsen, Afsaneh Doryab  

**Link**: [PDF](https://arxiv.org/pdf/2505.12109)  

**Abstract**: The combinatorial structure of many real-world action spaces leads to exponential growth in the number of possible actions, limiting the effectiveness of conventional reinforcement learning algorithms. Recent approaches for combinatorial action spaces impose factorized or sequential structures over sub-actions, failing to capture complex joint behavior. We introduce the Sub-Action Interaction Network using Transformers (SAINT), a novel policy architecture that represents multi-component actions as unordered sets and models their dependencies via self-attention conditioned on the global state. SAINT is permutation-invariant, sample-efficient, and compatible with standard policy optimization algorithms. In 15 distinct combinatorial environments across three task domains, including environments with nearly 17 million joint actions, SAINT consistently outperforms strong baselines. 

---
# EarthSynth: Generating Informative Earth Observation with Diffusion Models 

**Authors**: Jiancheng Pan, Shiye Lei, Yuqian Fu, Jiahao Li, Yanxing Liu, Yuze Sun, Xiao He, Long Peng, Xiaomeng Huang, Bo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.12108)  

**Abstract**: Remote sensing image (RSI) interpretation typically faces challenges due to the scarcity of labeled data, which limits the performance of RSI interpretation tasks. To tackle this challenge, we propose EarthSynth, a diffusion-based generative foundation model that enables synthesizing multi-category, cross-satellite labeled Earth observation for downstream RSI interpretation tasks. To the best of our knowledge, EarthSynth is the first to explore multi-task generation for remote sensing. EarthSynth, trained on the EarthSynth-180K dataset, employs the Counterfactual Composition training strategy to improve training data diversity and enhance category control. Furthermore, a rule-based method of R-Filter is proposed to filter more informative synthetic data for downstream tasks. We evaluate our EarthSynth on scene classification, object detection, and semantic segmentation in open-world scenarios, offering a practical solution for advancing RSI interpretation. 

---
# Learning Probabilistic Temporal Logic Specifications for Stochastic Systems 

**Authors**: Rajarshi Roy, Yash Pote, David Parker, Marta Kwiatkowska  

**Link**: [PDF](https://arxiv.org/pdf/2505.12107)  

**Abstract**: There has been substantial progress in the inference of formal behavioural specifications from sample trajectories, for example, using Linear Temporal Logic (LTL). However, these techniques cannot handle specifications that correctly characterise systems with stochastic behaviour, which occur commonly in reinforcement learning and formal verification. We consider the passive learning problem of inferring a Boolean combination of probabilistic LTL (PLTL) formulas from a set of Markov chains, classified as either positive or negative. We propose a novel learning algorithm that infers concise PLTL specifications, leveraging grammar-based enumeration, search heuristics, probabilistic model checking and Boolean set-cover procedures. We demonstrate the effectiveness of our algorithm in two use cases: learning from policies induced by RL algorithms and learning from variants of a probabilistic model. In both cases, our method automatically and efficiently extracts PLTL specifications that succinctly characterise the temporal differences between the policies or model variants. 

---
# Improving Fairness in LLMs Through Testing-Time Adversaries 

**Authors**: Isabela Pereira Gregio, Ian Pons, Anna Helena Reali Costa, Artur Jordão  

**Link**: [PDF](https://arxiv.org/pdf/2505.12100)  

**Abstract**: Large Language Models (LLMs) push the bound-aries in natural language processing and generative AI, driving progress across various aspects of modern society. Unfortunately, the pervasive issue of bias in LLMs responses (i.e., predictions) poses a significant and open challenge, hindering their application in tasks involving ethical sensitivity and responsible decision-making. In this work, we propose a straightforward, user-friendly and practical method to mitigate such biases, enhancing the reliability and trustworthiness of LLMs. Our method creates multiple variations of a given sentence by modifying specific attributes and evaluates the corresponding prediction behavior compared to the original, unaltered, prediction/sentence. The idea behind this process is that critical ethical predictions often exhibit notable inconsistencies, indicating the presence of bias. Unlike previous approaches, our method relies solely on forward passes (i.e., testing-time adversaries), eliminating the need for training, fine-tuning, or prior knowledge of the training data distribution. Through extensive experiments on the popular Llama family, we demonstrate the effectiveness of our method in improving various fairness metrics, focusing on the reduction of disparities in how the model treats individuals from different racial groups. Specifically, using standard metrics, we improve the fairness in Llama3 in up to 27 percentage points. Overall, our approach significantly enhances fairness, equity, and reliability in LLM-generated results without parameter tuning or training data modifications, confirming its effectiveness in practical scenarios. We believe our work establishes an important step toward enabling the use of LLMs in tasks that require ethical considerations and responsible decision-making. 

---
# When the Left Foot Leads to the Right Path: Bridging Initial Prejudice and Trainability 

**Authors**: Alberto Bassi, Carlo Albert, Aurelien Lucchi, Marco Baity-Jesi, Emanuele Francazi  

**Link**: [PDF](https://arxiv.org/pdf/2505.12096)  

**Abstract**: Understanding the statistical properties of deep neural networks (DNNs) at initialization is crucial for elucidating both their trainability and the intrinsic architectural biases they encode prior to data exposure. Mean-field (MF) analyses have demonstrated that the parameter distribution in randomly initialized networks dictates whether gradients vanish or explode. Concurrently, untrained DNNs were found to exhibit an initial-guessing bias (IGB), in which large regions of the input space are assigned to a single class. In this work, we derive a theoretical proof establishing the correspondence between IGB and previous MF theories, thereby connecting a network prejudice toward specific classes with the conditions for fast and accurate learning. This connection yields the counter-intuitive conclusion: the initialization that optimizes trainability is necessarily biased, rather than neutral. Furthermore, we extend the MF/IGB framework to multi-node activation functions, offering practical guidelines for designing initialization schemes that ensure stable optimization in architectures employing max- and average-pooling layers. 

---
# Attribution Projection Calculus: A Novel Framework for Causal Inference in Bayesian Networks 

**Authors**: M Ruhul Amin  

**Link**: [PDF](https://arxiv.org/pdf/2505.12094)  

**Abstract**: This paper introduces Attribution Projection Calculus (AP-Calculus), a novel mathematical framework for determining causal relationships in structured Bayesian networks. We investigate a specific network architecture with source nodes connected to destination nodes through intermediate nodes, where each input maps to a single label with maximum marginal probability. We prove that for each label, exactly one intermediate node acts as a deconfounder while others serve as confounders, enabling optimal attribution of features to their corresponding labels. The framework formalizes the dual nature of intermediate nodes as both confounders and deconfounders depending on the context, and establishes separation functions that maximize distinctions between intermediate representations. We demonstrate that the proposed network architecture is optimal for causal inference compared to alternative structures, including those based on Pearl's causal framework. AP-Calculus provides a comprehensive mathematical foundation for analyzing feature-label attributions, managing spurious correlations, quantifying information gain, ensuring fairness, and evaluating uncertainty in prediction models, including large language models. Theoretical verification shows that AP-Calculus not only extends but can also subsume traditional do-calculus for many practical applications, offering a more direct approach to causal inference in supervised learning contexts. 

---
# Personalized Author Obfuscation with Large Language Models 

**Authors**: Mohammad Shokri, Sarah Ita Levitan, Rivka Levitan  

**Link**: [PDF](https://arxiv.org/pdf/2505.12090)  

**Abstract**: In this paper, we investigate the efficacy of large language models (LLMs) in obfuscating authorship by paraphrasing and altering writing styles. Rather than adopting a holistic approach that evaluates performance across the entire dataset, we focus on user-wise performance to analyze how obfuscation effectiveness varies across individual authors. While LLMs are generally effective, we observe a bimodal distribution of efficacy, with performance varying significantly across users. To address this, we propose a personalized prompting method that outperforms standard prompting techniques and partially mitigates the bimodality issue. 

---
# NTIRE 2025 Challenge on Efficient Burst HDR and Restoration: Datasets, Methods, and Results 

**Authors**: Sangmin Lee, Eunpil Park, Angel Canelo, Hyunhee Park, Youngjo Kim, Hyung-Ju Chun, Xin Jin, Chongyi Li, Chun-Le Guo, Radu Timofte, Qi Wu, Tianheng Qiu, Yuchun Dong, Shenglin Ding, Guanghua Pan, Weiyu Zhou, Tao Hu, Yixu Feng, Duwei Dai, Yu Cao, Peng Wu, Wei Dong, Yanning Zhang, Qingsen Yan, Simon J. Larsen, Ruixuan Jiang, Senyan Xu, Xingbo Wang, Xin Lu, Marcos V. Conde, Javier Abad-Hernandez, Alvaro Garcıa-Lara, Daniel Feijoo, Alvaro Garcıa, Zeyu Xiao, Zhuoyuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.12089)  

**Abstract**: This paper reviews the NTIRE 2025 Efficient Burst HDR and Restoration Challenge, which aims to advance efficient multi-frame high dynamic range (HDR) and restoration techniques. The challenge is based on a novel RAW multi-frame fusion dataset, comprising nine noisy and misaligned RAW frames with various exposure levels per scene. Participants were tasked with developing solutions capable of effectively fusing these frames while adhering to strict efficiency constraints: fewer than 30 million model parameters and a computational budget under 4.0 trillion FLOPs. A total of 217 participants registered, with six teams finally submitting valid solutions. The top-performing approach achieved a PSNR of 43.22 dB, showcasing the potential of novel methods in this domain. This paper provides a comprehensive overview of the challenge, compares the proposed solutions, and serves as a valuable reference for researchers and practitioners in efficient burst HDR and restoration. 

---
# SepPrune: Structured Pruning for Efficient Deep Speech Separation 

**Authors**: Yuqi Li, Kai Li, Xin Yin, Zhifei Yang, Junhao Dong, Zeyu Dong, Chuanguang Yang, Yingli Tian, Yao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12079)  

**Abstract**: Although deep learning has substantially advanced speech separation in recent years, most existing studies continue to prioritize separation quality while overlooking computational efficiency, an essential factor for low-latency speech processing in real-time applications. In this paper, we propose SepPrune, the first structured pruning framework specifically designed to compress deep speech separation models and reduce their computational cost. SepPrune begins by analyzing the computational structure of a given model to identify layers with the highest computational burden. It then introduces a differentiable masking strategy to enable gradient-driven channel selection. Based on the learned masks, SepPrune prunes redundant channels and fine-tunes the remaining parameters to recover performance. Extensive experiments demonstrate that this learnable pruning paradigm yields substantial advantages for channel pruning in speech separation models, outperforming existing methods. Notably, a model pruned with SepPrune can recover 85% of the performance of a pre-trained model (trained over hundreds of epochs) with only one epoch of fine-tuning, and achieves convergence 36$\times$ faster than training from scratch. Code is available at this https URL. 

---
# MT-CYP-Net: Multi-Task Network for Pixel-Level Crop Yield Prediction Under Very Few Samples 

**Authors**: Shenzhou Liu, Di Wang, Haonan Guo, Chengxi Han, Wenzhi Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2505.12069)  

**Abstract**: Accurate and fine-grained crop yield prediction plays a crucial role in advancing global agriculture. However, the accuracy of pixel-level yield estimation based on satellite remote sensing data has been constrained by the scarcity of ground truth data. To address this challenge, we propose a novel approach called the Multi-Task Crop Yield Prediction Network (MT-CYP-Net). This framework introduces an effective multi-task feature-sharing strategy, where features extracted from a shared backbone network are simultaneously utilized by both crop yield prediction decoders and crop classification decoders with the ability to fuse information between them. This design allows MT-CYP-Net to be trained with extremely sparse crop yield point labels and crop type labels, while still generating detailed pixel-level crop yield maps. Concretely, we collected 1,859 yield point labels along with corresponding crop type labels and satellite images from eight farms in Heilongjiang Province, China, in 2023, covering soybean, maize, and rice crops, and constructed a sparse crop yield label dataset. MT-CYP-Net is compared with three classical machine learning and deep learning benchmark methods in this dataset. Experimental results not only indicate the superiority of MT-CYP-Net compared to previous methods on multiple types of crops but also demonstrate the potential of deep networks on precise pixel-level crop yield prediction, especially with limited data labels. 

---
# VFRTok: Variable Frame Rates Video Tokenizer with Duration-Proportional Information Assumption 

**Authors**: Tianxiong Zhong, Xingye Tian, Boyuan Jiang, Xuebo Wang, Xin Tao, Pengfei Wan, Zhiwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12053)  

**Abstract**: Modern video generation frameworks based on Latent Diffusion Models suffer from inefficiencies in tokenization due to the Frame-Proportional Information Assumption. Existing tokenizers provide fixed temporal compression rates, causing the computational cost of the diffusion model to scale linearly with the frame rate. The paper proposes the Duration-Proportional Information Assumption: the upper bound on the information capacity of a video is proportional to the duration rather than the number of frames. Based on this insight, the paper introduces VFRTok, a Transformer-based video tokenizer, that enables variable frame rate encoding and decoding through asymmetric frame rate training between the encoder and decoder. Furthermore, the paper proposes Partial Rotary Position Embeddings (RoPE) to decouple position and content modeling, which groups correlated patches into unified tokens. The Partial RoPE effectively improves content-awareness, enhancing the video generation capability. Benefiting from the compact and continuous spatio-temporal representation, VFRTok achieves competitive reconstruction quality and state-of-the-art generation fidelity while using only 1/8 tokens compared to existing tokenizers. 

---
# Enhanced Multimodal Hate Video Detection via Channel-wise and Modality-wise Fusion 

**Authors**: Yinghui Zhang, Tailin Chen, Yuchen Zhang, Zeyu Fu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12051)  

**Abstract**: The rapid rise of video content on platforms such as TikTok and YouTube has transformed information dissemination, but it has also facilitated the spread of harmful content, particularly hate videos. Despite significant efforts to combat hate speech, detecting these videos remains challenging due to their often implicit nature. Current detection methods primarily rely on unimodal approaches, which inadequately capture the complementary features across different modalities. While multimodal techniques offer a broader perspective, many fail to effectively integrate temporal dynamics and modality-wise interactions essential for identifying nuanced hate content. In this paper, we present CMFusion, an enhanced multimodal hate video detection model utilizing a novel Channel-wise and Modality-wise Fusion Mechanism. CMFusion first extracts features from text, audio, and video modalities using pre-trained models and then incorporates a temporal cross-attention mechanism to capture dependencies between video and audio streams. The learned features are then processed by channel-wise and modality-wise fusion modules to obtain informative representations of videos. Our extensive experiments on a real-world dataset demonstrate that CMFusion significantly outperforms five widely used baselines in terms of accuracy, precision, recall, and F1 score. Comprehensive ablation studies and parameter analyses further validate our design choices, highlighting the model's effectiveness in detecting hate videos. The source codes will be made publicly available at this https URL. 

---
# ABoN: Adaptive Best-of-N Alignment 

**Authors**: Vinod Raman, Hilal Asi, Satyen Kale  

**Link**: [PDF](https://arxiv.org/pdf/2505.12050)  

**Abstract**: Recent advances in test-time alignment methods, such as Best-of-N sampling, offer a simple and effective way to steer language models (LMs) toward preferred behaviors using reward models (RM). However, these approaches can be computationally expensive, especially when applied uniformly across prompts without accounting for differences in alignment difficulty. In this work, we propose a prompt-adaptive strategy for Best-of-N alignment that allocates inference-time compute more efficiently. Motivated by latency concerns, we develop a two-stage algorithm: an initial exploratory phase estimates the reward distribution for each prompt using a small exploration budget, and a second stage adaptively allocates the remaining budget using these estimates. Our method is simple, practical, and compatible with any LM/RM combination. Empirical results on the AlpacaEval dataset for 12 LM/RM pairs and 50 different batches of prompts show that our adaptive strategy consistently outperforms the uniform allocation with the same inference budget. Moreover, our experiments show that our adaptive strategy remains competitive against uniform allocations with 20% larger inference budgets and even improves in performance as the batch size grows. 

---
# Beyond Scalar Rewards: An Axiomatic Framework for Lexicographic MDPs 

**Authors**: Mehran Shakerinava, Siamak Ravanbakhsh, Adam Oberman  

**Link**: [PDF](https://arxiv.org/pdf/2505.12049)  

**Abstract**: Recent work has formalized the reward hypothesis through the lens of expected utility theory, by interpreting reward as utility. Hausner's foundational work showed that dropping the continuity axiom leads to a generalization of expected utility theory where utilities are lexicographically ordered vectors of arbitrary dimension. In this paper, we extend this result by identifying a simple and practical condition under which preferences cannot be represented by scalar rewards, necessitating a 2-dimensional reward function. We provide a full characterization of such reward functions, as well as the general d-dimensional case, in Markov Decision Processes (MDPs) under a memorylessness assumption on preferences. Furthermore, we show that optimal policies in this setting retain many desirable properties of their scalar-reward counterparts, while in the Constrained MDP (CMDP) setting -- another common multiobjective setting -- they do not. 

---
# Safe Delta: Consistently Preserving Safety when Fine-Tuning LLMs on Diverse Datasets 

**Authors**: Ning Lu, Shengcai Liu, Jiahao Wu, Weiyu Chen, Zhirui Zhang, Yew-Soon Ong, Qi Wang, Ke Tang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12038)  

**Abstract**: Large language models (LLMs) have shown great potential as general-purpose AI assistants across various domains. To fully leverage this potential in specific applications, many companies provide fine-tuning API services, enabling users to upload their own data for LLM customization. However, fine-tuning services introduce a new safety threat: user-uploaded data, whether harmful or benign, can break the model's alignment, leading to unsafe outputs. Moreover, existing defense methods struggle to address the diversity of fine-tuning datasets (e.g., varying sizes, tasks), often sacrificing utility for safety or vice versa. To address this issue, we propose Safe Delta, a safety-aware post-training defense method that adjusts the delta parameters (i.e., the parameter change before and after fine-tuning). Specifically, Safe Delta estimates the safety degradation, selects delta parameters to maximize utility while limiting overall safety loss, and applies a safety compensation vector to mitigate residual safety loss. Through extensive experiments on four diverse datasets with varying settings, our approach consistently preserves safety while ensuring that the utility gain from benign datasets remains unaffected. 

---
# GeoMaNO: Geometric Mamba Neural Operator for Partial Differential Equations 

**Authors**: Xi Han, Jingwei Zhang, Dimitris Samaras, Fei Hou, Hong Qin  

**Link**: [PDF](https://arxiv.org/pdf/2505.12020)  

**Abstract**: The neural operator (NO) framework has emerged as a powerful tool for solving partial differential equations (PDEs). Recent NOs are dominated by the Transformer architecture, which offers NOs the capability to capture long-range dependencies in PDE dynamics. However, existing Transformer-based NOs suffer from quadratic complexity, lack geometric rigor, and thus suffer from sub-optimal performance on regular grids. As a remedy, we propose the Geometric Mamba Neural Operator (GeoMaNO) framework, which empowers NOs with Mamba's modeling capability, linear complexity, plus geometric rigor. We evaluate GeoMaNO's performance on multiple standard and popularly employed PDE benchmarks, spanning from Darcy flow problems to Navier-Stokes problems. GeoMaNO improves existing baselines in solution operator approximation by as much as 58.9%. 

---
# CHRIS: Clothed Human Reconstruction with Side View Consistency 

**Authors**: Dong Liu, Yifan Yang, Zixiong Huang, Yuxin Gao, Mingkui Tan  

**Link**: [PDF](https://arxiv.org/pdf/2505.12005)  

**Abstract**: Creating a realistic clothed human from a single-view RGB image is crucial for applications like mixed reality and filmmaking. Despite some progress in recent years, mainstream methods often fail to fully utilize side-view information, as the input single-view image contains front-view information only. This leads to globally unrealistic topology and local surface inconsistency in side views. To address these, we introduce Clothed Human Reconstruction with Side View Consistency, namely CHRIS, which consists of 1) A Side-View Normal Discriminator that enhances global visual reasonability by distinguishing the generated side-view normals from the ground truth ones; 2) A Multi-to-One Gradient Computation (M2O) that ensures local surface consistency. M2O calculates the gradient of a sampling point by integrating the gradients of the nearby points, effectively acting as a smooth operation. Experimental results demonstrate that CHRIS achieves state-of-the-art performance on public benchmarks and outperforms the prior work. 

---
# Online Iterative Self-Alignment for Radiology Report Generation 

**Authors**: Ting Xiao, Lei Shi, Yang Zhang, HaoFeng Yang, Zhe Wang, Chenjia Bai  

**Link**: [PDF](https://arxiv.org/pdf/2505.11983)  

**Abstract**: Radiology Report Generation (RRG) is an important research topic for relieving radiologist' heavy workload. Existing RRG models mainly rely on supervised fine-tuning (SFT) based on different model architectures using data pairs of radiological images and corresponding radiologist-annotated reports. Recent research has shifted focus to post-training improvements, aligning RRG model outputs with human preferences using reinforcement learning (RL). However, the limited data coverage of high-quality annotated data poses risks of overfitting and generalization. This paper proposes a novel Online Iterative Self-Alignment (OISA) method for RRG that consists of four stages: self-generation of diverse data, self-evaluation for multi-objective preference data,self-alignment for multi-objective optimization and self-iteration for further improvement. Our approach allows for generating varied reports tailored to specific clinical objectives, enhancing the overall performance of the RRG model iteratively. Unlike existing methods, our frame-work significantly increases data quality and optimizes performance through iterative multi-objective optimization. Experimental results demonstrate that our method surpasses previous approaches, achieving state-of-the-art performance across multiple evaluation metrics. 

---
# AoP-SAM: Automation of Prompts for Efficient Segmentation 

**Authors**: Yi Chen, Mu-Young Son, Chuanbo Hua, Joo-Young Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.11980)  

**Abstract**: The Segment Anything Model (SAM) is a powerful foundation model for image segmentation, showing robust zero-shot generalization through prompt engineering. However, relying on manual prompts is impractical for real-world applications, particularly in scenarios where rapid prompt provision and resource efficiency are crucial. In this paper, we propose the Automation of Prompts for SAM (AoP-SAM), a novel approach that learns to generate essential prompts in optimal locations automatically. AoP-SAM enhances SAM's efficiency and usability by eliminating manual input, making it better suited for real-world tasks. Our approach employs a lightweight yet efficient Prompt Predictor model that detects key entities across images and identifies the optimal regions for placing prompt candidates. This method leverages SAM's image embeddings, preserving its zero-shot generalization capabilities without requiring fine-tuning. Additionally, we introduce a test-time instance-level Adaptive Sampling and Filtering mechanism that generates prompts in a coarse-to-fine manner. This notably enhances both prompt and mask generation efficiency by reducing computational overhead and minimizing redundant mask refinements. Evaluations of three datasets demonstrate that AoP-SAM substantially improves both prompt generation efficiency and mask generation accuracy, making SAM more effective for automated segmentation tasks. 

---
# Introduction to Analytical Software Engineering Design Paradigm 

**Authors**: Tarik Houichime, Younes El Amrani  

**Link**: [PDF](https://arxiv.org/pdf/2505.11979)  

**Abstract**: As modern software systems expand in scale and complexity, the challenges associated with their modeling and formulation grow increasingly intricate. Traditional approaches often fall short in effectively addressing these complexities, particularly in tasks such as design pattern detection for maintenance and assessment, as well as code refactoring for optimization and long-term sustainability. This growing inadequacy underscores the need for a paradigm shift in how such challenges are approached and resolved. This paper presents Analytical Software Engineering (ASE), a novel design paradigm aimed at balancing abstraction, tool accessibility, compatibility, and scalability. ASE enables effective modeling and resolution of complex software engineering problems. The paradigm is evaluated through two frameworks Behavioral-Structural Sequences (BSS) and Optimized Design Refactoring (ODR), both developed in accordance with ASE principles. BSS offers a compact, language-agnostic representation of codebases to facilitate precise design pattern detection. ODR unifies artifact and solution representations to optimize code refactoring via heuristic algorithms while eliminating iterative computational overhead. By providing a structured approach to software design challenges, ASE lays the groundwork for future research in encoding and analyzing complex software metrics. 

---
# MARVEL: Multi-Agent RTL Vulnerability Extraction using Large Language Models 

**Authors**: Luca Collini, Baleegh Ahmad, Joey Ah-kiow, Ramesh Karri  

**Link**: [PDF](https://arxiv.org/pdf/2505.11963)  

**Abstract**: Hardware security verification is a challenging and time-consuming task. For this purpose, design engineers may utilize tools such as formal verification, linters, and functional simulation tests, coupled with analysis and a deep understanding of the hardware design being inspected. Large Language Models (LLMs) have been used to assist during this task, either directly or in conjunction with existing tools. We improve the state of the art by proposing MARVEL, a multi-agent LLM framework for a unified approach to decision-making, tool use, and reasoning. MARVEL mimics the cognitive process of a designer looking for security vulnerabilities in RTL code. It consists of a supervisor agent that devises the security policy of the system-on-chips (SoCs) using its security documentation. It delegates tasks to validate the security policy to individual executor agents. Each executor agent carries out its assigned task using a particular strategy. Each executor agent may use one or more tools to identify potential security bugs in the design and send the results back to the supervisor agent for further analysis and confirmation. MARVEL includes executor agents that leverage formal tools, linters, simulation tests, LLM-based detection schemes, and static analysis-based checks. We test our approach on a known buggy SoC based on OpenTitan from the Hack@DATE competition. We find that 20 of the 48 issues reported by MARVEL pose security vulnerabilities. 

---
# Exploring Criteria of Loss Reweighting to Enhance LLM Unlearning 

**Authors**: Puning Yang, Qizhou Wang, Zhuo Huang, Tongliang Liu, Chengqi Zhang, Bo Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.11953)  

**Abstract**: Loss reweighting has shown significant benefits for machine unlearning with large language models (LLMs). However, their exact functionalities are left unclear and the optimal strategy remains an open question, thus impeding the understanding and improvement of existing methodologies. In this paper, we identify two distinct goals of loss reweighting, namely, Saturation and Importance -- the former indicates that those insufficiently optimized data should be emphasized, while the latter stresses some critical data that are most influential for loss minimization. To study their usefulness, we design specific reweighting strategies for each goal and evaluate their respective effects on unlearning. We conduct extensive empirical analyses on well-established benchmarks, and summarize some important observations as follows: (i) Saturation enhances efficacy more than importance-based reweighting, and their combination can yield additional improvements. (ii) Saturation typically allocates lower weights to data with lower likelihoods, whereas importance-based reweighting does the opposite. (iii) The efficacy of unlearning is also largely influenced by the smoothness and granularity of the weight distributions. Based on these findings, we propose SatImp, a simple reweighting method that combines the advantages of both saturation and importance. Empirical results on extensive datasets validate the efficacy of our method, potentially bridging existing research gaps and indicating directions for future research. Our code is available at this https URL. 

---
# Let's have a chat with the EU AI Act 

**Authors**: Adam Kovari, Yasin Ghafourian, Csaba Hegedus, Belal Abu Naim, Kitti Mezei, Pal Varga, Markus Tauber  

**Link**: [PDF](https://arxiv.org/pdf/2505.11946)  

**Abstract**: As artificial intelligence (AI) regulations evolve and the regulatory landscape develops and becomes more complex, ensuring compliance with ethical guidelines and legal frameworks remains a challenge for AI developers. This paper introduces an AI-driven self-assessment chatbot designed to assist users in navigating the European Union AI Act and related standards. Leveraging a Retrieval-Augmented Generation (RAG) framework, the chatbot enables real-time, context-aware compliance verification by retrieving relevant regulatory texts and providing tailored guidance. By integrating both public and proprietary standards, it streamlines regulatory adherence, reduces complexity, and fosters responsible AI development. The paper explores the chatbot's architecture, comparing naive and graph-based RAG models, and discusses its potential impact on AI governance. 

---
# Fine-Grained ECG-Text Contrastive Learning via Waveform Understanding Enhancement 

**Authors**: Haitao Li, Che Liu, Zhengyao Ding, Ziyi Liu, Zhengxing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11939)  

**Abstract**: Electrocardiograms (ECGs) are essential for diagnosing cardiovascular diseases. While previous ECG-text contrastive learning methods have shown promising results, they often overlook the incompleteness of the reports. Given an ECG, the report is generated by first identifying key waveform features and then inferring the final diagnosis through these features. Despite their importance, these waveform features are often not recorded in the report as intermediate results. Aligning ECGs with such incomplete reports impedes the model's ability to capture the ECG's waveform features and limits its understanding of diagnostic reasoning based on those features. To address this, we propose FG-CLEP (Fine-Grained Contrastive Language ECG Pre-training), which aims to recover these waveform features from incomplete reports with the help of large language models (LLMs), under the challenges of hallucinations and the non-bijective relationship between waveform features and diagnoses. Additionally, considering the frequent false negatives due to the prevalence of common diagnoses in ECGs, we introduce a semantic similarity matrix to guide contrastive learning. Furthermore, we adopt a sigmoid-based loss function to accommodate the multi-label nature of ECG-related tasks. Experiments on six datasets demonstrate that FG-CLEP outperforms state-of-the-art methods in both zero-shot prediction and linear probing across these datasets. 

---
# How can Diffusion Models Evolve into Continual Generators? 

**Authors**: Jingren Liu, Zhong Ji, Xiangyu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.11936)  

**Abstract**: While diffusion models have achieved remarkable success in static data generation, their deployment in streaming or continual learning (CL) scenarios faces a major challenge: catastrophic forgetting (CF), where newly acquired generative capabilities overwrite previously learned ones. To systematically address this, we introduce a formal Continual Diffusion Generation (CDG) paradigm that characterizes and redefines CL in the context of generative diffusion models. Prior efforts often adapt heuristic strategies from continual classification tasks but lack alignment with the underlying diffusion process. In this work, we develop the first theoretical framework for CDG by analyzing cross-task dynamics in diffusion-based generative modeling. Our analysis reveals that the retention and stability of generative knowledge across tasks are governed by three key consistency criteria: inter-task knowledge consistency (IKC), unconditional knowledge consistency (UKC), and label knowledge consistency (LKC). Building on these insights, we propose Continual Consistency Diffusion (CCD), a principled framework that integrates these consistency objectives into training via hierarchical loss terms $\mathcal{L}_{IKC}$, $\mathcal{L}_{UKC}$, and $\mathcal{L}_{LKC}$. This promotes effective knowledge retention while enabling the assimilation of new generative capabilities. Extensive experiments on four benchmark datasets demonstrate that CCD achieves state-of-the-art performance under continual settings, with substantial gains in Mean Fidelity (MF) and Incremental Mean Fidelity (IMF), particularly in tasks with rich cross-task knowledge overlap. 

---
# Conversational Recommendation System using NLP and Sentiment Analysis 

**Authors**: Piyush Talegaonkar, Siddhant Hole, Shrinesh Kamble, Prashil Gulechha, Deepali Salapurkar  

**Link**: [PDF](https://arxiv.org/pdf/2505.11933)  

**Abstract**: In today's digitally-driven world, the demand for personalized and context-aware recommendations has never been greater. Traditional recommender systems have made significant strides in this direction, but they often lack the ability to tap into the richness of conversational data. This paper represents a novel approach to recommendation systems by integrating conversational insights into the recommendation process. The Conversational Recommender System integrates cutting-edge technologies such as deep learning, leveraging machine learning algorithms like Apriori for Association Rule Mining, Convolutional Neural Networks (CNN), Recurrent Neural Networks (RNN), and Long Short-Term Memory (LTSM). Furthermore, sophisticated voice recognition technologies, including Hidden Markov Models (HMMs) and Dynamic Time Warping (DTW) algorithms, play a crucial role in accurate speech-to-text conversion, ensuring robust performance in diverse environments. The methodology incorporates a fusion of content-based and collaborative recommendation approaches, enhancing them with NLP techniques. This innovative integration ensures a more personalized and context-aware recommendation experience, particularly in marketing applications. 

---
# The Logical Expressiveness of Temporal GNNs via Two-Dimensional Product Logics 

**Authors**: Marco Sälzer, Przemysław Andrzej Wałęga, Martin Lange  

**Link**: [PDF](https://arxiv.org/pdf/2505.11930)  

**Abstract**: In recent years, the expressive power of various neural architectures -- including graph neural networks (GNNs), transformers, and recurrent neural networks -- has been characterised using tools from logic and formal language theory. As the capabilities of basic architectures are becoming well understood, increasing attention is turning to models that combine multiple architectural paradigms. Among them particularly important, and challenging to analyse, are temporal extensions of GNNs, which integrate both spatial (graph-structure) and temporal (evolution over time) dimensions. In this paper, we initiate the study of logical characterisation of temporal GNNs by connecting them to two-dimensional product logics. We show that the expressive power of temporal GNNs depends on how graph and temporal components are combined. In particular, temporal GNNs that apply static GNNs recursively over time can capture all properties definable in the product logic of (past) propositional temporal logic PTL and the modal logic K. In contrast, architectures such as graph-and-time TGNNs and global TGNNs can only express restricted fragments of this logic, where the interaction between temporal and spatial operators is syntactically constrained. These results yield the first logical characterisations of temporal GNNs and establish new relative expressiveness results for temporal GNNs. 

---
# SafeVid: Toward Safety Aligned Video Large Multimodal Models 

**Authors**: Yixu Wang, Jiaxin Song, Yifeng Gao, Xin Wang, Yang Yao, Yan Teng, Xingjun Ma, Yingchun Wang, Yu-Gang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11926)  

**Abstract**: As Video Large Multimodal Models (VLMMs) rapidly advance, their inherent complexity introduces significant safety challenges, particularly the issue of mismatched generalization where static safety alignments fail to transfer to dynamic video contexts. We introduce SafeVid, a framework designed to instill video-specific safety principles in VLMMs. SafeVid uniquely transfers robust textual safety alignment capabilities to the video domain by employing detailed textual video descriptions as an interpretive bridge, facilitating LLM-based rule-driven safety reasoning. This is achieved through a closed-loop system comprising: 1) generation of SafeVid-350K, a novel 350,000-pair video-specific safety preference dataset; 2) targeted alignment of VLMMs using Direct Preference Optimization (DPO); and 3) comprehensive evaluation via our new SafeVidBench benchmark. Alignment with SafeVid-350K significantly enhances VLMM safety, with models like LLaVA-NeXT-Video demonstrating substantial improvements (e.g., up to 42.39%) on SafeVidBench. SafeVid provides critical resources and a structured approach, demonstrating that leveraging textual descriptions as a conduit for safety reasoning markedly improves the safety alignment of VLMMs. We have made SafeVid-350K dataset (this https URL) publicly available. 

---
# An Explanation of Intrinsic Self-Correction via Linear Representations and Latent Concepts 

**Authors**: Yu-Ting Lee, Hui-Ying Shih, Fu-Chieh Chang, Pei-Yuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.11924)  

**Abstract**: We provide an explanation for the performance gains of intrinsic self-correction, a process where a language model iteratively refines its outputs without external feedback. More precisely, we investigate how prompting induces interpretable changes in hidden states and thus affects the output distributions. We hypothesize that each prompt-induced shift lies in a linear span of some linear representation vectors, naturally separating tokens based on individual concept alignment. Building around this idea, we give a mathematical formulation of self-correction and derive a concentration result for output tokens based on alignment magnitudes. Our experiments on text detoxification with zephyr-7b-sft reveal a substantial gap in the inner products of the prompt-induced shifts and the unembeddings of the top-100 most toxic tokens vs. those of the unembeddings of the bottom-100 least toxic tokens, under toxic instructions. This suggests that self-correction prompts enhance a language model's capability of latent concept recognition. Our analysis offers insights into the underlying mechanism of self-correction by characterizing how prompting works explainably. For reproducibility, our code is available. 

---
# Modèles de Substitution pour les Modèles à base d'Agents : Enjeux, Méthodes et Applications 

**Authors**: Paul Saves, Nicolas Verstaevel, Benoît Gaudou  

**Link**: [PDF](https://arxiv.org/pdf/2505.11912)  

**Abstract**: Multi-agent simulations enables the modeling and analyses of the dynamic behaviors and interactions of autonomous entities evolving in complex environments. Agent-based models (ABM) are widely used to study emergent phenomena arising from local interactions. However, their high computational cost poses a significant challenge, particularly for large-scale simulations requiring extensive parameter exploration, optimization, or uncertainty quantification. The increasing complexity of ABM limits their feasibility for real-time decision-making and large-scale scenario analysis. To address these limitations, surrogate models offer an efficient alternative by learning approximations from sparse simulation data. These models provide cheap-to-evaluate predictions, significantly reducing computational costs while maintaining accuracy. Various machine learning techniques, including regression models, neural networks, random forests and Gaussian processes, have been applied to construct robust surrogates. Moreover, uncertainty quantification and sensitivity analysis play a crucial role in enhancing model reliability and interpretability.
This article explores the motivations, methods, and applications of surrogate modeling for ABM, emphasizing the trade-offs between accuracy, computational efficiency, and interpretability. Through a case study on a segregation model, we highlight the challenges associated with building and validating surrogate models, comparing different approaches and evaluating their performance. Finally, we discuss future perspectives on integrating surrogate models within ABM to improve scalability, explainability, and real-time decision support across various fields such as ecology, urban planning and economics. 

---
# K*-Means: A Parameter-free Clustering Algorithm 

**Authors**: Louis Mahon, Mirella Lapata  

**Link**: [PDF](https://arxiv.org/pdf/2505.11904)  

**Abstract**: Clustering is a widely used and powerful machine learning technique, but its effectiveness is often limited by the need to specify the number of clusters, k, or by relying on thresholds that implicitly determine k. We introduce k*-means, a novel clustering algorithm that eliminates the need to set k or any other parameters. Instead, it uses the minimum description length principle to automatically determine the optimal number of clusters, k*, by splitting and merging clusters while also optimising the standard k-means objective. We prove that k*-means is guaranteed to converge and demonstrate experimentally that it significantly outperforms existing methods in scenarios where k is unknown. We also show that it is accurate in estimating k, and that empirically its runtime is competitive with existing methods, and scales well with dataset size. 

---
# AdaCoT: Pareto-Optimal Adaptive Chain-of-Thought Triggering via Reinforcement Learning 

**Authors**: Chenwei Lou, Zewei Sun, Xinnian Liang, Meng Qu, Wei Shen, Wenqi Wang, Yuntao Li, Qingping Yang, Shuangzhi Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.11896)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities but often face challenges with tasks requiring sophisticated reasoning. While Chain-of-Thought (CoT) prompting significantly enhances reasoning, it indiscriminately generates lengthy reasoning steps for all queries, leading to substantial computational costs and inefficiency, especially for simpler inputs. To address this critical issue, we introduce AdaCoT (Adaptive Chain-of-Thought), a novel framework enabling LLMs to adaptively decide when to invoke CoT. AdaCoT framed adaptive reasoning as a Pareto optimization problem that seeks to balance model performance with the costs associated with CoT invocation (both frequency and computational overhead). We propose a reinforcement learning (RL) based method, specifically utilizing Proximal Policy Optimization (PPO), to dynamically control the CoT triggering decision boundary by adjusting penalty coefficients, thereby allowing the model to determine CoT necessity based on implicit query complexity. A key technical contribution is Selective Loss Masking (SLM), designed to counteract decision boundary collapse during multi-stage RL training, ensuring robust and stable adaptive triggering. Experimental results demonstrate that AdaCoT successfully navigates the Pareto frontier, achieving substantial reductions in CoT usage for queries not requiring elaborate reasoning. For instance, on our production traffic testset, AdaCoT reduced CoT triggering rates to as low as 3.18\% and decreased average response tokens by 69.06%, while maintaining high performance on complex tasks. 

---
# RLAP: A Reinforcement Learning Enhanced Adaptive Planning Framework for Multi-step NLP Task Solving 

**Authors**: Zepeng Ding, Dixuan Wang, Ziqin Luo, Guochao Jiang, Deqing Yang, Jiaqing Liang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11893)  

**Abstract**: Multi-step planning has been widely employed to enhance the performance of large language models (LLMs) on downstream natural language processing (NLP) tasks, which decomposes the original task into multiple subtasks and guide LLMs to solve them sequentially without additional training. When addressing task instances, existing methods either preset the order of steps or attempt multiple paths at each step. However, these methods overlook instances' linguistic features and rely on the intrinsic planning capabilities of LLMs to evaluate intermediate feedback and then select subtasks, resulting in suboptimal outcomes. To better solve multi-step NLP tasks with LLMs, in this paper we propose a Reinforcement Learning enhanced Adaptive Planning framework (RLAP). In our framework, we model an NLP task as a Markov decision process (MDP) and employ an LLM directly into the environment. In particular, a lightweight Actor model is trained to estimate Q-values for natural language sequences consisting of states and actions through reinforcement learning. Therefore, during sequential planning, the linguistic features of each sequence in the MDP can be taken into account, and the Actor model interacts with the LLM to determine the optimal order of subtasks for each task instance. We apply RLAP on three different types of NLP tasks and conduct extensive experiments on multiple datasets to verify RLAP's effectiveness and robustness. 

---
# Mobile-Bench-v2: A More Realistic and Comprehensive Benchmark for VLM-based Mobile Agents 

**Authors**: Weikai Xu, Zhizheng Jiang, Yuxuan Liu, Wei Liu, Jian Luan, Yuanchun Li, Yunxin Liu, Bin Wang, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2505.11891)  

**Abstract**: VLM-based mobile agents are increasingly popular due to their capabilities to interact with smartphone GUIs and XML-structured texts and to complete daily tasks. However, existing online benchmarks struggle with obtaining stable reward signals due to dynamic environmental changes. Offline benchmarks evaluate the agents through single-path trajectories, which stands in contrast to the inherently multi-solution characteristics of GUI tasks. Additionally, both types of benchmarks fail to assess whether mobile agents can handle noise or engage in proactive interactions due to a lack of noisy apps or overly full instructions during the evaluation process. To address these limitations, we use a slot-based instruction generation method to construct a more realistic and comprehensive benchmark named Mobile-Bench-v2. Mobile-Bench-v2 includes a common task split, with offline multi-path evaluation to assess the agent's ability to obtain step rewards during task execution. It contains a noisy split based on pop-ups and ads apps, and a contaminated split named AITZ-Noise to formulate a real noisy environment. Furthermore, an ambiguous instruction split with preset Q\&A interactions is released to evaluate the agent's proactive interaction capabilities. We conduct evaluations on these splits using the single-agent framework AppAgent-v1, the multi-agent framework Mobile-Agent-v2, as well as other mobile agents such as UI-Tars and OS-Atlas. Code and data are available at this https URL. 

---
# Exploring the Potential of SSL Models for Sound Event Detection 

**Authors**: Hanfang Cui, Longfei Song, Li Li, Dongxing Xu, Yanhua Long  

**Link**: [PDF](https://arxiv.org/pdf/2505.11889)  

**Abstract**: Self-supervised learning (SSL) models offer powerful representations for sound event detection (SED), yet their synergistic potential remains underexplored. This study systematically evaluates state-of-the-art SSL models to guide optimal model selection and integration for SED. We propose a framework that combines heterogeneous SSL representations (e.g., BEATs, HuBERT, WavLM) through three fusion strategies: individual SSL embedding integration, dual-modal fusion, and full aggregation. Experiments on the DCASE 2023 Task 4 Challenge reveal that dual-modal fusion (e.g., CRNN+BEATs+WavLM) achieves complementary performance gains, while CRNN+BEATs alone delivers the best results among individual SSL models. We further introduce normalized sound event bounding boxes (nSEBBs), an adaptive post-processing method that dynamically adjusts event boundary predictions, improving PSDS1 by up to 4% for standalone SSL models. These findings highlight the compatibility and complementarity of SSL architectures, providing guidance for task-specific fusion and robust SED system design. 

---
# Revisiting Residual Connections: Orthogonal Updates for Stable and Efficient Deep Networks 

**Authors**: Giyeong Oh, Woohyun Cho, Siyeol Kim, Suhwan Choi, Younjae Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.11881)  

**Abstract**: Residual connections are pivotal for deep neural networks, enabling greater depth by mitigating vanishing gradients. However, in standard residual updates, the module's output is directly added to the input stream. This can lead to updates that predominantly reinforce or modulate the existing stream direction, potentially underutilizing the module's capacity for learning entirely novel features. In this work, we introduce Orthogonal Residual Update: we decompose the module's output relative to the input stream and add only the component orthogonal to this stream. This design aims to guide modules to contribute primarily new representational directions, fostering richer feature learning while promoting more efficient training. We demonstrate that our orthogonal update strategy improves generalization accuracy and training stability across diverse architectures (ResNetV2, Vision Transformers) and datasets (CIFARs, TinyImageNet, ImageNet-1k), achieving, for instance, a +4.3\%p top-1 accuracy gain for ViT-B on ImageNet-1k. 

---
# AdaptMol: Adaptive Fusion from Sequence String to Topological Structure for Few-shot Drug Discovery 

**Authors**: Yifan Dai, Xuanbai Ren, Tengfei Ma, Qipeng Yan, Yiping Liu, Yuansheng Liu, Xiangxiang Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2505.11878)  

**Abstract**: Accurate molecular property prediction (MPP) is a critical step in modern drug development. However, the scarcity of experimental validation data poses a significant challenge to AI-driven research paradigms. Under few-shot learning scenarios, the quality of molecular representations directly dictates the theoretical upper limit of model performance. We present AdaptMol, a prototypical network integrating Adaptive multimodal fusion for Molecular representation. This framework employs a dual-level attention mechanism to dynamically integrate global and local molecular features derived from two modalities: SMILES sequences and molecular graphs. (1) At the local level, structural features such as atomic interactions and substructures are extracted from molecular graphs, emphasizing fine-grained topological information; (2) At the global level, the SMILES sequence provides a holistic representation of the molecule. To validate the necessity of multimodal adaptive fusion, we propose an interpretable approach based on identifying molecular active substructures to demonstrate that multimodal adaptive fusion can efficiently represent molecules. Extensive experiments on three commonly used benchmarks under 5-shot and 10-shot settings demonstrate that AdaptMol achieves state-of-the-art performance in most cases. The rationale-extracted method guides the fusion of two modalities and highlights the importance of both modalities. 

---
# Learning Pareto-Optimal Rewards from Noisy Preferences: A Framework for Multi-Objective Inverse Reinforcement Learning 

**Authors**: Kalyan Cherukuri, Aarav Lala  

**Link**: [PDF](https://arxiv.org/pdf/2505.11864)  

**Abstract**: As generative agents become increasingly capable, alignment of their behavior with complex human values remains a fundamental challenge. Existing approaches often simplify human intent through reduction to a scalar reward, overlooking the multi-faceted nature of human feedback. In this work, we introduce a theoretical framework for preference-based Multi-Objective Inverse Reinforcement Learning (MO-IRL), where human preferences are modeled as latent vector-valued reward functions. We formalize the problem of recovering a Pareto-optimal reward representation from noisy preference queries and establish conditions for identifying the underlying multi-objective structure. We derive tight sample complexity bounds for recovering $\epsilon$-approximations of the Pareto front and introduce a regret formulation to quantify suboptimality in this multi-objective setting. Furthermore, we propose a provably convergent algorithm for policy optimization using preference-inferred reward cones. Our results bridge the gap between practical alignment techniques and theoretical guarantees, providing a principled foundation for learning aligned behaviors in a high-dimension and value-pluralistic environment. 

---
# Q-Policy: Quantum-Enhanced Policy Evaluation for Scalable Reinforcement Learning 

**Authors**: Kalyan Cherukuri, Aarav Lala, Yash Yardi  

**Link**: [PDF](https://arxiv.org/pdf/2505.11862)  

**Abstract**: We propose Q-Policy, a hybrid quantum-classical reinforcement learning (RL) framework that mathematically accelerates policy evaluation and optimization by exploiting quantum computing primitives. Q-Policy encodes value functions in quantum superposition, enabling simultaneous evaluation of multiple state-action pairs via amplitude encoding and quantum parallelism. We introduce a quantum-enhanced policy iteration algorithm with provable polynomial reductions in sample complexity for the evaluation step, under standard assumptions. To demonstrate the technical feasibility and theoretical soundness of our approach, we validate Q-Policy on classical emulations of small discrete control tasks. Due to current hardware and simulation limitations, our experiments focus on showcasing proof-of-concept behavior rather than large-scale empirical evaluation. Our results support the potential of Q-Policy as a theoretical foundation for scalable RL on future quantum devices, addressing RL scalability challenges beyond classical approaches. 

---
# On Membership Inference Attacks in Knowledge Distillation 

**Authors**: Ziyao Cui, Minxing Zhang, Jian Pei  

**Link**: [PDF](https://arxiv.org/pdf/2505.11837)  

**Abstract**: Nowadays, Large Language Models (LLMs) are trained on huge datasets, some including sensitive information. This poses a serious privacy concern because privacy attacks such as Membership Inference Attacks (MIAs) may detect this sensitive information. While knowledge distillation compresses LLMs into efficient, smaller student models, its impact on privacy remains underexplored. In this paper, we investigate how knowledge distillation affects model robustness against MIA. We focus on two questions. First, how is private data protected in teacher and student models? Second, how can we strengthen privacy preservation against MIAs in knowledge distillation? Through comprehensive experiments, we show that while teacher and student models achieve similar overall MIA accuracy, teacher models better protect member data, the primary target of MIA, whereas student models better protect non-member data. To address this vulnerability in student models, we propose 5 privacy-preserving distillation methods and demonstrate that they successfully reduce student models' vulnerability to MIA, with ensembling further stabilizing the robustness, offering a reliable approach for distilling more secure and efficient student models. Our implementation source code is available at this https URL. 

---
# SplInterp: Improving our Understanding and Training of Sparse Autoencoders 

**Authors**: Jeremy Budd, Javier Ideami, Benjamin Macdowall Rynne, Keith Duggar, Randall Balestriero  

**Link**: [PDF](https://arxiv.org/pdf/2505.11836)  

**Abstract**: Sparse autoencoders (SAEs) have received considerable recent attention as tools for mechanistic interpretability, showing success at extracting interpretable features even from very large LLMs. However, this research has been largely empirical, and there have been recent doubts about the true utility of SAEs. In this work, we seek to enhance the theoretical understanding of SAEs, using the spline theory of deep learning. By situating SAEs in this framework: we discover that SAEs generalise ``$k$-means autoencoders'' to be piecewise affine, but sacrifice accuracy for interpretability vs. the optimal ``$k$-means-esque plus local principal component analysis (PCA)'' piecewise affine autoencoder. We characterise the underlying geometry of (TopK) SAEs using power diagrams. And we develop a novel proximal alternating method SGD (PAM-SGD) algorithm for training SAEs, with both solid theoretical foundations and promising empirical results in MNIST and LLM experiments, particularly in sample efficiency and (in the LLM setting) improved sparsity of codes. All code is available at: this https URL 

---
# Multilingual Collaborative Defense for Large Language Models 

**Authors**: Hongliang Li, Jinan Xu, Gengping Cui, Changhao Guan, Fengran Mo, Kaiyu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11835)  

**Abstract**: The robustness and security of large language models (LLMs) has become a prominent research area. One notable vulnerability is the ability to bypass LLM safeguards by translating harmful queries into rare or underrepresented languages, a simple yet effective method of "jailbreaking" these models. Despite the growing concern, there has been limited research addressing the safeguarding of LLMs in multilingual scenarios, highlighting an urgent need to enhance multilingual safety. In this work, we investigate the correlation between various attack features across different languages and propose Multilingual Collaborative Defense (MCD), a novel learning method that optimizes a continuous, soft safety prompt automatically to facilitate multilingual safeguarding of LLMs. The MCD approach offers three advantages: First, it effectively improves safeguarding performance across multiple languages. Second, MCD maintains strong generalization capabilities while minimizing false refusal rates. Third, MCD mitigates the language safety misalignment caused by imbalances in LLM training corpora. To evaluate the effectiveness of MCD, we manually construct multilingual versions of commonly used jailbreak benchmarks, such as MaliciousInstruct and AdvBench, to assess various safeguarding methods. Additionally, we introduce these datasets in underrepresented (zero-shot) languages to verify the language transferability of MCD. The results demonstrate that MCD outperforms existing approaches in safeguarding against multilingual jailbreak attempts while also exhibiting strong language transfer capabilities. Our code is available at this https URL. 

---
# CoT-Vid: Dynamic Chain-of-Thought Routing with Self Verification for Training-Free Video Reasoning 

**Authors**: Hongbo Jin, Ruyang Liu, Wenhao Zhang, Guibo Luo, Ge Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.11830)  

**Abstract**: System2 reasoning is developing rapidly these days with the emergence of Deep- Thinking Models and chain-of-thought technology, which has become a centralized discussion point in the AI community. However, there is a relative gap in the research on complex video reasoning at present. In this work, we propose CoT-Vid, a novel training-free paradigm for the video domain with a multistage complex reasoning design. Distinguishing from existing video LLMs, which rely heavily on perceptual abilities, it achieved surprising performance gain with explicit reasoning mechanism. The paradigm consists of three main components: dynamic inference path routing, problem decoupling strategy, and video self-consistency verification. In addition, we propose a new standard for categorization of video questions. CoT- Vid showed outstanding results on a wide range of benchmarks, and outperforms its base model by 9.3% on Egochema and 5.6% on VideoEspresso, rivalling or even surpassing larger and proprietary models, such as GPT-4V, GPT-4o and Gemini-1.5-flash. Our codebase will be publicly available soon. 

---
# Not All Thoughts are Generated Equal: Efficient LLM Reasoning via Multi-Turn Reinforcement Learning 

**Authors**: Yansong Ning, Wei Li, Jun Fang, Naiqiang Tan, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.11827)  

**Abstract**: Compressing long chain-of-thought (CoT) from large language models (LLMs) is an emerging strategy to improve the reasoning efficiency of LLMs. Despite its promising benefits, existing studies equally compress all thoughts within a long CoT, hindering more concise and effective reasoning. To this end, we first investigate the importance of different thoughts by examining their effectiveness and efficiency in contributing to reasoning through automatic long CoT chunking and Monte Carlo rollouts. Building upon the insights, we propose a theoretically bounded metric to jointly measure the effectiveness and efficiency of different thoughts. We then propose Long$\otimes$Short, an efficient reasoning framework that enables two LLMs to collaboratively solve the problem: a long-thought LLM for more effectively generating important thoughts, while a short-thought LLM for efficiently generating remaining thoughts. Specifically, we begin by synthesizing a small amount of cold-start data to fine-tune LLMs for long-thought and short-thought reasoning styles, respectively. Furthermore, we propose a synergizing-oriented multi-turn reinforcement learning, focusing on the model self-evolution and collaboration between long-thought and short-thought LLMs. Experimental results show that our method enables Qwen2.5-7B and Llama3.1-8B to achieve comparable performance compared to DeepSeek-R1-Distill-Qwen-7B and DeepSeek-R1-Distill-Llama-8B, while reducing token length by over 80% across the MATH500, AIME24/25, AMC23, and GPQA Diamond benchmarks. Our data and code are available at this https URL. 

---
# Bootstrapping Diffusion: Diffusion Model Training Leveraging Partial and Corrupted Data 

**Authors**: Xudong Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.11825)  

**Abstract**: Training diffusion models requires large datasets. However, acquiring large volumes of high-quality data can be challenging, for example, collecting large numbers of high-resolution images and long videos. On the other hand, there are many complementary data that are usually considered corrupted or partial, such as low-resolution images and short videos. Other examples of corrupted data include videos that contain subtitles, watermarks, and logos. In this study, we investigate the theoretical problem of whether the above partial data can be utilized to train conventional diffusion models. Motivated by our theoretical analysis in this study, we propose a straightforward approach of training diffusion models utilizing partial data views, where we consider each form of complementary data as a view of conventional data. Our proposed approach first trains one separate diffusion model for each individual view, and then trains a model for predicting the residual score function. We prove generalization error bounds, which show that the proposed diffusion model training approach can achieve lower generalization errors if proper regularizations are adopted in the residual score function training. In particular, we prove that the difficulty in training the residual score function scales proportionally with the signal correlations not captured by partial data views. Consequently, the proposed approach achieves near first-order optimal data efficiency. 

---
# Search-Based Correction of Reasoning Chains for Language Models 

**Authors**: Minsu Kim, Jean-Pierre Falet, Oliver E. Richardson, Xiaoyin Chen, Moksh Jain, Sungjin Ahn, Sungsoo Ahn, Yoshua Bengio  

**Link**: [PDF](https://arxiv.org/pdf/2505.11824)  

**Abstract**: Chain-of-Thought (CoT) reasoning has advanced the capabilities and transparency of language models (LMs); however, reasoning chains can contain inaccurate statements that reduce performance and trustworthiness. To address this, we introduce a new self-correction framework that augments each reasoning step in a CoT with a latent variable indicating its veracity, enabling modeling of all possible truth assignments rather than assuming correctness throughout. To efficiently explore this expanded space, we introduce Search Corrector, a discrete search algorithm over boolean-valued veracity assignments. It efficiently performs otherwise intractable inference in the posterior distribution over veracity assignments by leveraging the LM's joint likelihood over veracity and the final answer as a proxy reward. This efficient inference-time correction method facilitates supervised fine-tuning of an Amortized Corrector by providing pseudo-labels for veracity. The Amortized Corrector generalizes self-correction, enabling accurate zero-shot veracity inference in novel contexts. Empirical results demonstrate that Search Corrector reliably identifies errors in logical (ProntoQA) and mathematical reasoning (GSM8K) benchmarks. The Amortized Corrector achieves comparable zero-shot accuracy and improves final answer accuracy by up to 25%. 

---
# SGD-Mix: Enhancing Domain-Specific Image Classification with Label-Preserving Data Augmentation 

**Authors**: Yixuan Dong, Fang-Yi Su, Jung-Hsien Chiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11813)  

**Abstract**: Data augmentation for domain-specific image classification tasks often struggles to simultaneously address diversity, faithfulness, and label clarity of generated data, leading to suboptimal performance in downstream tasks. While existing generative diffusion model-based methods aim to enhance augmentation, they fail to cohesively tackle these three critical aspects and often overlook intrinsic challenges of diffusion models, such as sensitivity to model characteristics and stochasticity under strong transformations. In this paper, we propose a novel framework that explicitly integrates diversity, faithfulness, and label clarity into the augmentation process. Our approach employs saliency-guided mixing and a fine-tuned diffusion model to preserve foreground semantics, enrich background diversity, and ensure label consistency, while mitigating diffusion model limitations. Extensive experiments across fine-grained, long-tail, few-shot, and background robustness tasks demonstrate our method's superior performance over state-of-the-art approaches. 

---
# Retrospex: Language Agent Meets Offline Reinforcement Learning Critic 

**Authors**: Yufei Xiang, Yiqun Shen, Yeqin Zhang, Cam-Tu Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2505.11807)  

**Abstract**: Large Language Models (LLMs) possess extensive knowledge and commonsense reasoning capabilities, making them valuable for creating powerful agents. However, existing LLM agent frameworks have not fully utilized past experiences for improvement. This work introduces a new LLM-based agent framework called Retrospex, which addresses this challenge by analyzing past experiences in depth. Unlike previous approaches, Retrospex does not directly integrate experiences into the LLM's context. Instead, it combines the LLM's action likelihood with action values estimated by a Reinforcement Learning (RL) Critic, which is trained on past experiences through an offline ''retrospection'' process. Additionally, Retrospex employs a dynamic action rescoring mechanism that increases the importance of experience-based values for tasks that require more interaction with the environment. We evaluate Retrospex in ScienceWorld, ALFWorld and Webshop environments, demonstrating its advantages over strong, contemporary baselines. 

---
# Are vision language models robust to uncertain inputs? 

**Authors**: Xi Wang, Eric Nalisnick  

**Link**: [PDF](https://arxiv.org/pdf/2505.11804)  

**Abstract**: Robustness against uncertain and ambiguous inputs is a critical challenge for deep learning models. While recent advancements in large scale vision language models (VLMs, e.g. GPT4o) might suggest that increasing model and training dataset size would mitigate this issue, our empirical evaluation shows a more complicated picture. Testing models using two classic uncertainty quantification tasks, anomaly detection and classification under inherently ambiguous conditions, we find that newer and larger VLMs indeed exhibit improved robustness compared to earlier models, but still suffer from a tendency to strictly follow instructions, often causing them to hallucinate confident responses even when faced with unclear or anomalous inputs. Remarkably, for natural images such as ImageNet, this limitation can be overcome without pipeline modifications: simply prompting models to abstain from uncertain predictions enables significant reliability gains, achieving near-perfect robustness in several settings. However, for domain-specific tasks such as galaxy morphology classification, a lack of specialized knowledge prevents reliable uncertainty estimation. Finally, we propose a novel mechanism based on caption diversity to reveal a model's internal uncertainty, enabling practitioners to predict when models will successfully abstain without relying on labeled data. 

---
# Diffmv: A Unified Diffusion Framework for Healthcare Predictions with Random Missing Views and View Laziness 

**Authors**: Chuang Zhao, Hui Tang, Hongke Zhao, Xiaomeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.11802)  

**Abstract**: Advanced healthcare predictions offer significant improvements in patient outcomes by leveraging predictive analytics. Existing works primarily utilize various views of Electronic Health Record (EHR) data, such as diagnoses, lab tests, or clinical notes, for model training. These methods typically assume the availability of complete EHR views and that the designed model could fully leverage the potential of each view. However, in practice, random missing views and view laziness present two significant challenges that hinder further improvements in multi-view utilization. To address these challenges, we introduce Diffmv, an innovative diffusion-based generative framework designed to advance the exploitation of multiple views of EHR data. Specifically, to address random missing views, we integrate various views of EHR data into a unified diffusion-denoising framework, enriched with diverse contextual conditions to facilitate progressive alignment and view transformation. To mitigate view laziness, we propose a novel reweighting strategy that assesses the relative advantages of each view, promoting a balanced utilization of various data views within the model. Our proposed strategy achieves superior performance across multiple health prediction tasks derived from three popular datasets, including multi-view and multi-modality scenarios. 

---
# CL-CaGAN: Capsule differential adversarial continuous learning for cross-domain hyperspectral anomaly detection 

**Authors**: Jianing Wang, Siying Guo, Zheng Hua, Runhu Huang, Jinyu Hu, Maoguo Gong  

**Link**: [PDF](https://arxiv.org/pdf/2505.11793)  

**Abstract**: Anomaly detection (AD) has attracted remarkable attention in hyperspectral image (HSI) processing fields, and most existing deep learning (DL)-based algorithms indicate dramatic potential for detecting anomaly samples through specific training process under current scenario. However, the limited prior information and the catastrophic forgetting problem indicate crucial challenges for existing DL structure in open scenarios cross-domain detection. In order to improve the detection performance, a novel continual learning-based capsule differential generative adversarial network (CL-CaGAN) is proposed to elevate the cross-scenario learning performance for facilitating the real application of DL-based structure in hyperspectral AD (HAD) task. First, a modified capsule structure with adversarial learning network is constructed to estimate the background distribution for surmounting the deficiency of prior information. To mitigate the catastrophic forgetting phenomenon, clustering-based sample replay strategy and a designed extra self-distillation regularization are integrated for merging the history and future knowledge in continual AD task, while the discriminative learning ability from previous detection scenario to current scenario is retained by the elaborately designed structure with continual learning (CL) strategy. In addition, the differentiable enhancement is enforced to augment the generation performance of the training data. This further stabilizes the training process with better convergence and efficiently consolidates the reconstruction ability of background samples. To verify the effectiveness of our proposed CL-CaGAN, we conduct experiments on several real HSIs, and the results indicate that the proposed CL-CaGAN demonstrates higher detection performance and continuous learning capacity for mitigating the catastrophic forgetting under cross-domain scenarios. 

---
# Improving Coverage in Combined Prediction Sets with Weighted p-values 

**Authors**: Gina Wong, Drew Prinster, Suchi Saria, Rama Chellappa, Anqi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.11785)  

**Abstract**: Conformal prediction quantifies the uncertainty of machine learning models by augmenting point predictions with valid prediction sets, assuming exchangeability. For complex scenarios involving multiple trials, models, or data sources, conformal prediction sets can be aggregated to create a prediction set that captures the overall uncertainty, often improving precision. However, aggregating multiple prediction sets with individual $1-\alpha$ coverage inevitably weakens the overall guarantee, typically resulting in $1-2\alpha$ worst-case coverage. In this work, we propose a framework for the weighted aggregation of prediction sets, where weights are assigned to each prediction set based on their contribution. Our framework offers flexible control over how the sets are aggregated, achieving tighter coverage bounds that interpolate between the $1-2\alpha$ guarantee of the combined models and the $1-\alpha$ guarantee of an individual model depending on the distribution of weights. We extend our framework to data-dependent weights, and we derive a general procedure for data-dependent weight aggregation that maintains finite-sample validity. We demonstrate the effectiveness of our methods through experiments on synthetic and real data in the mixture-of-experts setting, and we show that aggregation with data-dependent weights provides a form of adaptive coverage. 

---
# Generative and Contrastive Graph Representation Learning 

**Authors**: Jiali Chen, Avijit Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2505.11776)  

**Abstract**: Self-supervised learning (SSL) on graphs generates node and graph representations (i.e., embeddings) that can be used for downstream tasks such as node classification, node clustering, and link prediction. Graph SSL is particularly useful in scenarios with limited or no labeled data. Existing SSL methods predominantly follow contrastive or generative paradigms, each excelling in different tasks: contrastive methods typically perform well on classification tasks, while generative methods often excel in link prediction. In this paper, we present a novel architecture for graph SSL that integrates the strengths of both approaches. Our framework introduces community-aware node-level contrastive learning, providing more robust and effective positive and negative node pairs generation, alongside graph-level contrastive learning to capture global semantic information. Additionally, we employ a comprehensive augmentation strategy that combines feature masking, node perturbation, and edge perturbation, enabling robust and diverse representation learning. By incorporating these enhancements, our model achieves superior performance across multiple tasks, including node classification, clustering, and link prediction. Evaluations on open benchmark datasets demonstrate that our model outperforms state-of-the-art methods, achieving a performance lift of 0.23%-2.01% depending on the task and dataset. 

---
# HARDMath2: A Benchmark for Applied Mathematics Built by Students as Part of a Graduate Class 

**Authors**: James V. Roggeveen, Erik Y. Wang, Will Flintoft, Peter Donets, Lucy S. Nathwani, Nickholas Gutierrez, David Ettel, Anton Marius Graf, Siddharth Dandavate, Arjun Nageswaran, Raglan Ward, Ava Williamson, Anne Mykland, Kacper K. Migacz, Yijun Wang, Egemen Bostan, Duy Thuc Nguyen, Zhe He, Marc L. Descoteaux, Felix Yeung, Shida Liu, Jorge García Ponce, Luke Zhu, Yuyang Chen, Ekaterina S. Ivshina, Miguel Fernandez, Minjae Kim, Kennan Gumbs, Matthew Scott Tan, Russell Yang, Mai Hoang, David Brown, Isabella A. Silveira, Lavon Sykes, Ahmed Roman, William Fredenberg, Yiming Chen, Lucas Martin, Yixing Tang, Kelly Werker Smith, Hongyu Liao, Logan G. Wilson, Alexander Dazhen Cai, Andrea Elizabeth Biju, Michael P. Brenner  

**Link**: [PDF](https://arxiv.org/pdf/2505.11774)  

**Abstract**: Large language models (LLMs) have shown remarkable progress in mathematical problem-solving, but evaluation has largely focused on problems that have exact analytical solutions or involve formal proofs, often overlooking approximation-based problems ubiquitous in applied science and engineering. To fill this gap, we build on prior work and present HARDMath2, a dataset of 211 original problems covering the core topics in an introductory graduate applied math class, including boundary-layer analysis, WKB methods, asymptotic solutions of nonlinear partial differential equations, and the asymptotics of oscillatory integrals. This dataset was designed and verified by the students and instructors of a core graduate applied mathematics course at Harvard. We build the dataset through a novel collaborative environment that challenges students to write and refine difficult problems consistent with the class syllabus, peer-validate solutions, test different models, and automatically check LLM-generated solutions against their own answers and numerical ground truths. Evaluation results show that leading frontier models still struggle with many of the problems in the dataset, highlighting a gap in the mathematical reasoning skills of current LLMs. Importantly, students identified strategies to create increasingly difficult problems by interacting with the models and exploiting common failure modes. This back-and-forth with the models not only resulted in a richer and more challenging benchmark but also led to qualitative improvements in the students' understanding of the course material, which is increasingly important as we enter an age where state-of-the-art language models can solve many challenging problems across a wide domain of fields. 

---
# Residual Feature Integration is Sufficient to Prevent Negative Transfer 

**Authors**: Yichen Xu, Ryumei Nakada, Linjun Zhang, Lexin Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.11771)  

**Abstract**: Transfer learning typically leverages representations learned from a source domain to improve performance on a target task. A common approach is to extract features from a pre-trained model and directly apply them for target prediction. However, this strategy is prone to negative transfer where the source representation fails to align with the target distribution. In this article, we propose Residual Feature Integration (REFINE), a simple yet effective method designed to mitigate negative transfer. Our approach combines a fixed source-side representation with a trainable target-side encoder and fits a shallow neural network on the resulting joint representation, which adapts to the target domain while preserving transferable knowledge from the source domain. Theoretically, we prove that REFINE is sufficient to prevent negative transfer under mild conditions, and derive the generalization bound demonstrating its theoretical benefit. Empirically, we show that REFINE consistently enhances performance across diverse application and data modalities including vision, text, and tabular data, and outperforms numerous alternative solutions. Our method is lightweight, architecture-agnostic, and robust, making it a valuable addition to the existing transfer learning toolbox. 

---
# Internal Causal Mechanisms Robustly Predict Language Model Out-of-Distribution Behaviors 

**Authors**: Jing Huang, Junyi Tao, Thomas Icard, Diyi Yang, Christopher Potts  

**Link**: [PDF](https://arxiv.org/pdf/2505.11770)  

**Abstract**: Interpretability research now offers a variety of techniques for identifying abstract internal mechanisms in neural networks. Can such techniques be used to predict how models will behave on out-of-distribution examples? In this work, we provide a positive answer to this question. Through a diverse set of language modeling tasks--including symbol manipulation, knowledge retrieval, and instruction following--we show that the most robust features for correctness prediction are those that play a distinctive causal role in the model's behavior. Specifically, we propose two methods that leverage causal mechanisms to predict the correctness of model outputs: counterfactual simulation (checking whether key causal variables are realized) and value probing (using the values of those variables to make predictions). Both achieve high AUC-ROC in distribution and outperform methods that rely on causal-agnostic features in out-of-distribution settings, where predicting model behaviors is more crucial. Our work thus highlights a novel and significant application for internal causal analysis of language models. 

---
# Redefining Neural Operators in $d+1$ Dimensions 

**Authors**: Haoze Song, Zhihao Li, Xiaobo Zhang, Zecheng Gan, Zhilu Lai, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11766)  

**Abstract**: Neural Operators have emerged as powerful tools for learning mappings between function spaces. Among them, the kernel integral operator has been widely validated on universally approximating various operators. Although recent advancements following this definition have developed effective modules to better approximate the kernel function defined on the original domain (with $d$ dimensions, $d=1, 2, 3...$), the unclarified evolving mechanism in the embedding spaces blocks our view to design neural operators that can fully capture the target system evolution.
Drawing on recent breakthroughs in quantum simulation of partial differential equations (PDEs), we elucidate the linear evolution process in neural operators. Based on that, we redefine neural operators on a new $d+1$ dimensional domain. Within this framework, we implement our proposed Schrödingerised Kernel Neural Operator (SKNO) aligning better with the $d+1$ dimensional evolution. In experiments, our $d+1$ dimensional evolving linear block performs far better than others. Also, we test SKNO's SOTA performance on various benchmark tests and also the zero-shot super-resolution task. In addition, we analyse the impact of different lifting and recovering operators on the prediction within the redefined NO framework, reflecting the alignment between our model and the underlying $d+1$ dimensional evolution. 

---
# OMAC: A Broad Optimization Framework for LLM-Based Multi-Agent Collaboration 

**Authors**: Shijun Li, Hilaf Hasson, Joydeep Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2505.11765)  

**Abstract**: Agents powered by advanced large language models (LLMs) have demonstrated impressive capabilities across diverse complex applications. Recently, Multi-Agent Systems (MAS), wherein multiple agents collaborate and communicate with each other, have exhibited enhanced capabilities in complex tasks, such as high-quality code generation and arithmetic reasoning. However, the development of such systems often relies on handcrafted methods, and the literature on systematic design and optimization of LLM-based MAS remains limited.
In this work, we introduce OMAC, a general framework designed for holistic optimization of LLM-based MAS. Specifically, we identify five key optimization dimensions for MAS, encompassing both agent functionality and collaboration structure. Building upon these dimensions, we first propose a general algorithm, utilizing two actors termed the Semantic Initializer and the Contrastive Comparator, to optimize any single dimension. Then, we present an algorithm for joint optimization across multiple dimensions. Extensive experiments demonstrate the superior performance of OMAC on code generation, arithmetic reasoning, and general reasoning tasks against state-of-the-art approaches. 

---
# Towards Universal Semantics With Large Language Models 

**Authors**: Raymond Baartmans, Matthew Raffel, Rahul Vikram, Aiden Deringer, Lizhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.11764)  

**Abstract**: The Natural Semantic Metalanguage (NSM) is a linguistic theory based on a universal set of semantic primes: simple, primitive word-meanings that have been shown to exist in most, if not all, languages of the world. According to this framework, any word, regardless of complexity, can be paraphrased using these primes, revealing a clear and universally translatable meaning. These paraphrases, known as explications, can offer valuable applications for many natural language processing (NLP) tasks, but producing them has traditionally been a slow, manual process. In this work, we present the first study of using large language models (LLMs) to generate NSM explications. We introduce automatic evaluation methods, a tailored dataset for training and evaluation, and fine-tuned models for this task. Our 1B and 8B models outperform GPT-4o in producing accurate, cross-translatable explications, marking a significant step toward universal semantic representation with LLMs and opening up new possibilities for applications in semantic analysis, translation, and beyond. 

---
# Topology-Aware Knowledge Propagation in Decentralized Learning 

**Authors**: Mansi Sakarvadia, Nathaniel Hudson, Tian Li, Ian Foster, Kyle Chard  

**Link**: [PDF](https://arxiv.org/pdf/2505.11760)  

**Abstract**: Decentralized learning enables collaborative training of models across naturally distributed data without centralized coordination or maintenance of a global model. Instead, devices are organized in arbitrary communication topologies, in which they can only communicate with neighboring devices. Each device maintains its own local model by training on its local data and integrating new knowledge via model aggregation with neighbors. Therefore, knowledge is propagated across the topology via successive aggregation rounds. We study, in particular, the propagation of out-of-distribution (OOD) knowledge. We find that popular decentralized learning algorithms struggle to propagate OOD knowledge effectively to all devices. Further, we find that both the location of OOD data within a topology, and the topology itself, significantly impact OOD knowledge propagation. We then propose topology-aware aggregation strategies to accelerate (OOD) knowledge propagation across devices. These strategies improve OOD data accuracy, compared to topology-unaware baselines, by 123% on average across models in a topology. 

---
# Generalizable Vision-Language Few-Shot Adaptation with Predictive Prompts and Negative Learning 

**Authors**: Sriram Mandalika  

**Link**: [PDF](https://arxiv.org/pdf/2505.11758)  

**Abstract**: Few-shot adaptation remains a core challenge for vision-language models (VLMs), especially under limited supervision and noisy support samples. We propose PromptFuseNL, a unified framework that enhances few-shot generalization by combining predictive prompt tuning with dual-branch positive and negative learning. The method refines class prototypes through task-conditioned residuals, multi-stage cross-modal coordination, and semantic hard negative mining. To address label noise, we introduce an unsupervised instance reweighting strategy that downweights unreliable support examples without requiring additional labels or structural changes. PromptFuseNL fuses visual and textual cues through lightweight modules for efficient and discriminative prediction. Evaluated across 15 benchmarks, it consistently surpasses existing prompt- and adapter-based methods in all shot settings while remaining highly efficient, achieving up to 300x faster training and 1000x lower FLOPs compared to full prompt tuning, achieving a new state-of-the-art for robust and scalable few-shot vision-language adaptation. 

---
# Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders 

**Authors**: David Chanin, Tomáš Dulka, Adrià Garriga-Alonso  

**Link**: [PDF](https://arxiv.org/pdf/2505.11756)  

**Abstract**: It is assumed that sparse autoencoders (SAEs) decompose polysemantic activations into interpretable linear directions, as long as the activations are composed of sparse linear combinations of underlying features. However, we find that if an SAE is more narrow than the number of underlying "true features" on which it is trained, and there is correlation between features, the SAE will merge components of correlated features together, thus destroying monosemanticity. In LLM SAEs, these two conditions are almost certainly true. This phenomenon, which we call feature hedging, is caused by SAE reconstruction loss, and is more severe the narrower the SAE. In this work, we introduce the problem of feature hedging and study it both theoretically in toy models and empirically in SAEs trained on LLMs. We suspect that feature hedging may be one of the core reasons that SAEs consistently underperform supervised baselines. Finally, we use our understanding of feature hedging to propose an improved variant of matryoshka SAEs. Our work shows there remain fundamental issues with SAEs, but we are hopeful that that highlighting feature hedging will catalyze future advances that allow SAEs to achieve their full potential of interpreting LLMs at scale. 

---
# Reachability Barrier Networks: Learning Hamilton-Jacobi Solutions for Smooth and Flexible Control Barrier Functions 

**Authors**: Matthew Kim, William Sharpless, Hyun Joe Jeong, Sander Tonkens, Somil Bansal, Sylvia Herbert  

**Link**: [PDF](https://arxiv.org/pdf/2505.11755)  

**Abstract**: Recent developments in autonomous driving and robotics underscore the necessity of safety-critical controllers. Control barrier functions (CBFs) are a popular method for appending safety guarantees to a general control framework, but they are notoriously difficult to generate beyond low dimensions. Existing methods often yield non-differentiable or inaccurate approximations that lack integrity, and thus fail to ensure safety. In this work, we use physics-informed neural networks (PINNs) to generate smooth approximations of CBFs by computing Hamilton-Jacobi (HJ) optimal control solutions. These reachability barrier networks (RBNs) avoid traditional dimensionality constraints and support the tuning of their conservativeness post-training through a parameterized discount term. To ensure robustness of the discounted solutions, we leverage conformal prediction methods to derive probabilistic safety guarantees for RBNs. We demonstrate that RBNs are highly accurate in low dimensions, and safer than the standard neural CBF approach in high dimensions. Namely, we showcase the RBNs in a 9D multi-vehicle collision avoidance problem where it empirically proves to be 5.5x safer and 1.9x less conservative than the neural CBFs, offering a promising method to synthesize CBFs for general nonlinear autonomous systems. 

---
# Improving Medium Range Severe Weather Prediction through Transformer Post-processing of AI Weather Forecasts 

**Authors**: Zhanxiang Hua, Ryan Sobash, David John Gagne II, Yingkai Sha, Alexandra Anderson-Frey  

**Link**: [PDF](https://arxiv.org/pdf/2505.11750)  

**Abstract**: Improving the skill of medium-range (1-8 day) severe weather prediction is crucial for mitigating societal impacts. This study introduces a novel approach leveraging decoder-only transformer networks to post-process AI-based weather forecasts, specifically from the Pangu-Weather model, for improved severe weather guidance. Unlike traditional post-processing methods that use a dense neural network to predict the probability of severe weather using discrete forecast samples, our method treats forecast lead times as sequential ``tokens'', enabling the transformer to learn complex temporal relationships within the evolving atmospheric state. We compare this approach against post-processing of the Global Forecast System (GFS) using both a traditional dense neural network and our transformer, as well as configurations that exclude convective parameters to fairly evaluate the impact of using the Pangu-Weather AI model. Results demonstrate that the transformer-based post-processing significantly enhances forecast skill compared to dense neural networks. Furthermore, AI-driven forecasts, particularly Pangu-Weather initialized from high resolution analysis, exhibit superior performance to GFS in the medium-range, even without explicit convective parameters. Our approach offers improved accuracy, and reliability, which also provides interpretability through feature attribution analysis, advancing medium-range severe weather prediction capabilities. 

---
# Token Masking Improves Transformer-Based Text Classification 

**Authors**: Xianglong Xu, John Bowen, Rojin Taheri  

**Link**: [PDF](https://arxiv.org/pdf/2505.11746)  

**Abstract**: While transformer-based models achieve strong performance on text classification, we explore whether masking input tokens can further enhance their effectiveness. We propose token masking regularization, a simple yet theoretically motivated method that randomly replaces input tokens with a special [MASK] token at probability p. This introduces stochastic perturbations during training, leading to implicit gradient averaging that encourages the model to capture deeper inter-token dependencies. Experiments on language identification and sentiment analysis -- across diverse models (mBERT, Qwen2.5-0.5B, TinyLlama-1.1B) -- show consistent improvements over standard regularization techniques. We identify task-specific optimal masking rates, with p = 0.1 as a strong general default. We attribute the gains to two key effects: (1) input perturbation reduces overfitting, and (2) gradient-level smoothing acts as implicit ensembling. 

---
# POCAII: Parameter Optimization with Conscious Allocation using Iterative Intelligence 

**Authors**: Joshua Inman, Tanmay Khandait, Lalitha Sankar, Giulia Pedrielli  

**Link**: [PDF](https://arxiv.org/pdf/2505.11745)  

**Abstract**: In this paper we propose for the first time the hyperparameter optimization (HPO) algorithm POCAII. POCAII differs from the Hyperband and Successive Halving literature by explicitly separating the search and evaluation phases and utilizing principled approaches to exploration and exploitation principles during both phases. Such distinction results in a highly flexible scheme for managing a hyperparameter optimization budget by focusing on search (i.e., generating competing configurations) towards the start of the HPO process while increasing the evaluation effort as the HPO comes to an end.
POCAII was compared to state of the art approaches SMAC, BOHB and DEHB. Our algorithm shows superior performance in low-budget hyperparameter optimization regimes. Since many practitioners do not have exhaustive resources to assign to HPO, it has wide applications to real-world problems. Moreover, the empirical evidence showed how POCAII demonstrates higher robustness and lower variance in the results. This is again very important when considering realistic scenarios with extremely expensive models to train. 

---
# Cloud-Based AI Systems: Leveraging Large Language Models for Intelligent Fault Detection and Autonomous Self-Healing 

**Authors**: Cheng Ji, Huaiying Luo  

**Link**: [PDF](https://arxiv.org/pdf/2505.11743)  

**Abstract**: With the rapid development of cloud computing systems and the increasing complexity of their infrastructure, intelligent mechanisms to detect and mitigate failures in real time are becoming increasingly important. Traditional methods of failure detection are often difficult to cope with the scale and dynamics of modern cloud environments. In this study, we propose a novel AI framework based on Massive Language Model (LLM) for intelligent fault detection and self-healing mechanisms in cloud systems. The model combines existing machine learning fault detection algorithms with LLM's natural language understanding capabilities to process and parse system logs, error reports, and real-time data streams through semantic context. The method adopts a multi-level architecture, combined with supervised learning for fault classification and unsupervised learning for anomaly detection, so that the system can predict potential failures before they occur and automatically trigger the self-healing mechanism. Experimental results show that the proposed model is significantly better than the traditional fault detection system in terms of fault detection accuracy, system downtime reduction and recovery speed. 

---
# Simple and Effective Specialized Representations for Fair Classifiers 

**Authors**: Alberto Sinigaglia, Davide Sartor, Marina Ceccon, Gian Antonio Susto  

**Link**: [PDF](https://arxiv.org/pdf/2505.11740)  

**Abstract**: Fair classification is a critical challenge that has gained increasing importance due to international regulations and its growing use in high-stakes decision-making settings. Existing methods often rely on adversarial learning or distribution matching across sensitive groups; however, adversarial learning can be unstable, and distribution matching can be computationally intensive. To address these limitations, we propose a novel approach based on the characteristic function distance. Our method ensures that the learned representation contains minimal sensitive information while maintaining high effectiveness for downstream tasks. By utilizing characteristic functions, we achieve a more stable and efficient solution compared to traditional methods. Additionally, we introduce a simple relaxation of the objective function that guarantees fairness in common classification models with no performance degradation. Experimental results on benchmark datasets demonstrate that our approach consistently matches or achieves better fairness and predictive accuracy than existing methods. Moreover, our method maintains robustness and computational efficiency, making it a practical solution for real-world applications. 

---
# ZeroTuning: Unlocking the Initial Token's Power to Enhance Large Language Models Without Training 

**Authors**: Feijiang Han, Xiaodong Yu, Jianheng Tang, Lyle Ungar  

**Link**: [PDF](https://arxiv.org/pdf/2505.11739)  

**Abstract**: Recently, training-free methods for improving large language models (LLMs) have attracted growing interest, with token-level attention tuning emerging as a promising and interpretable direction. However, existing methods typically rely on auxiliary mechanisms to identify important or irrelevant task-specific tokens, introducing potential bias and limiting applicability. In this paper, we uncover a surprising and elegant alternative: the semantically empty initial token is a powerful and underexplored control point for optimizing model behavior. Through theoretical analysis, we show that tuning the initial token's attention sharpens or flattens the attention distribution over subsequent tokens, and its role as an attention sink amplifies this effect. Empirically, we find that: (1) tuning its attention improves LLM performance more effectively than tuning other task-specific tokens; (2) the effect follows a consistent trend across layers, with earlier layers having greater impact, but varies across attention heads, with different heads showing distinct preferences in how they attend to this token. Based on these findings, we propose ZeroTuning, a training-free approach that improves LLM performance by applying head-specific attention adjustments to this special token. Despite tuning only one token, ZeroTuning achieves higher performance on text classification, multiple-choice, and multi-turn conversation tasks across models such as Llama, Qwen, and DeepSeek. For example, ZeroTuning improves Llama-3.1-8B by 11.71% on classification, 2.64% on QA tasks, and raises its multi-turn score from 7.804 to 7.966. The method is also robust to limited resources, few-shot settings, long contexts, quantization, decoding strategies, and prompt variations. Our work sheds light on a previously overlooked control point in LLMs, offering new insights into both inference-time tuning and model interpretability. 

---
# Token-Level Uncertainty Estimation for Large Language Model Reasoning 

**Authors**: Tunyu Zhang, Haizhou Shi, Yibin Wang, Hengyi Wang, Xiaoxiao He, Zhuowei Li, Haoxian Chen, Ligong Han, Kai Xu, Huan Zhang, Dimitris Metaxas, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11737)  

**Abstract**: While Large Language Models (LLMs) have demonstrated impressive capabilities, their output quality remains inconsistent across various application scenarios, making it difficult to identify trustworthy responses, especially in complex tasks requiring multi-step reasoning. In this paper, we propose a token-level uncertainty estimation framework to enable LLMs to self-assess and self-improve their generation quality in mathematical reasoning. Specifically, we introduce low-rank random weight perturbation to LLM decoding, generating predictive distributions that we use to estimate token-level uncertainties. We then aggregate these uncertainties to reflect semantic uncertainty of the generated sequences. Experiments on mathematical reasoning datasets of varying difficulty demonstrate that our token-level uncertainty metrics strongly correlate with answer correctness and model robustness. Additionally, we explore using uncertainty to directly enhance the model's reasoning performance through multiple generations and the particle filtering algorithm. Our approach consistently outperforms existing uncertainty estimation methods, establishing effective uncertainty estimation as a valuable tool for both evaluating and improving reasoning generation in LLMs. 

---
# Efficient Uncertainty Estimation via Distillation of Bayesian Large Language Models 

**Authors**: Harshil Vejendla, Haizhou Shi, Yibin Wang, Tunyu Zhang, Huan Zhang, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11731)  

**Abstract**: Recent advances in uncertainty estimation for Large Language Models (LLMs) during downstream adaptation have addressed key challenges of reliability and simplicity. However, existing Bayesian methods typically require multiple sampling iterations during inference, creating significant efficiency issues that limit practical deployment. In this paper, we investigate the possibility of eliminating the need for test-time sampling for LLM uncertainty estimation. Specifically, when given an off-the-shelf Bayesian LLM, we distill its aligned confidence into a non-Bayesian student LLM by minimizing the divergence between their predictive distributions. Unlike typical calibration methods, our distillation is carried out solely on the training dataset without the need of an additional validation dataset. This simple yet effective approach achieves N-times more efficient uncertainty estimation during testing, where N is the number of samples traditionally required by Bayesian LLMs. Our extensive experiments demonstrate that uncertainty estimation capabilities on training data can successfully generalize to unseen test data through our distillation technique, consistently producing results comparable to (or even better than) state-of-the-art Bayesian LLMs. 

---
# CLT and Edgeworth Expansion for m-out-of-n Bootstrap Estimators of The Studentized Median 

**Authors**: Imon Banerjee, Sayak Chakrabarty  

**Link**: [PDF](https://arxiv.org/pdf/2505.11725)  

**Abstract**: The m-out-of-n bootstrap, originally proposed by Bickel, Gotze, and Zwet (1992), approximates the distribution of a statistic by repeatedly drawing m subsamples (with m much smaller than n) without replacement from an original sample of size n. It is now routinely used for robust inference with heavy-tailed data, bandwidth selection, and other large-sample applications. Despite its broad applicability across econometrics, biostatistics, and machine learning, rigorous parameter-free guarantees for the soundness of the m-out-of-n bootstrap when estimating sample quantiles have remained elusive.
This paper establishes such guarantees by analyzing the estimator of sample quantiles obtained from m-out-of-n resampling of a dataset of size n. We first prove a central limit theorem for a fully data-driven version of the estimator that holds under a mild moment condition and involves no unknown nuisance parameters. We then show that the moment assumption is essentially tight by constructing a counter-example in which the CLT fails. Strengthening the assumptions slightly, we derive an Edgeworth expansion that provides exact convergence rates and, as a corollary, a Berry Esseen bound on the bootstrap approximation error. Finally, we illustrate the scope of our results by deriving parameter-free asymptotic distributions for practical statistics, including the quantiles for random walk Metropolis-Hastings and the rewards of ergodic Markov decision processes, thereby demonstrating the usefulness of our theory in modern estimation and learning tasks. 

---
# Zero-Shot Visual Generalization in Robot Manipulation 

**Authors**: Sumeet Batra, Gaurav Sukhatme  

**Link**: [PDF](https://arxiv.org/pdf/2505.11719)  

**Abstract**: Training vision-based manipulation policies that are robust across diverse visual environments remains an important and unresolved challenge in robot learning. Current approaches often sidestep the problem by relying on invariant representations such as point clouds and depth, or by brute-forcing generalization through visual domain randomization and/or large, visually diverse datasets. Disentangled representation learning - especially when combined with principles of associative memory - has recently shown promise in enabling vision-based reinforcement learning policies to be robust to visual distribution shifts. However, these techniques have largely been constrained to simpler benchmarks and toy environments. In this work, we scale disentangled representation learning and associative memory to more visually and dynamically complex manipulation tasks and demonstrate zero-shot adaptability to visual perturbations in both simulation and on real hardware. We further extend this approach to imitation learning, specifically Diffusion Policy, and empirically show significant gains in visual generalization compared to state-of-the-art imitation learning methods. Finally, we introduce a novel technique adapted from the model equivariance literature that transforms any trained neural network policy into one invariant to 2D planar rotations, making our policy not only visually robust but also resilient to certain camera perturbations. We believe that this work marks a significant step towards manipulation policies that are not only adaptable out of the box, but also robust to the complexities and dynamical nature of real-world deployment. Supplementary videos are available at this https URL. 

---
# EnvInjection: Environmental Prompt Injection Attack to Multi-modal Web Agents 

**Authors**: Xilong Wang, John Bloch, Zedian Shao, Yuepeng Hu, Shuyan Zhou, Neil Zhenqiang Gong  

**Link**: [PDF](https://arxiv.org/pdf/2505.11717)  

**Abstract**: Multi-modal large language model (MLLM)-based web agents interact with webpage environments by generating actions based on screenshots of the webpages. Environmental prompt injection attacks manipulate the environment to induce the web agent to perform a specific, attacker-chosen action--referred to as the target action. However, existing attacks suffer from limited effectiveness or stealthiness, or are impractical in real-world settings. In this work, we propose EnvInjection, a new attack that addresses these limitations. Our attack adds a perturbation to the raw pixel values of the rendered webpage, which can be implemented by modifying the webpage's source code. After these perturbed pixels are mapped into a screenshot, the perturbation induces the web agent to perform the target action. We formulate the task of finding the perturbation as an optimization problem. A key challenge in solving this problem is that the mapping between raw pixel values and screenshot is non-differentiable, making it difficult to backpropagate gradients to the perturbation. To overcome this, we train a neural network to approximate the mapping and apply projected gradient descent to solve the reformulated optimization problem. Extensive evaluation on multiple webpage datasets shows that EnvInjection is highly effective and significantly outperforms existing baselines. 

---
# Bi-Level Policy Optimization with Nyström Hypergradients 

**Authors**: Arjun Prakash, Naicheng He, Denizalp Goktas, Amy Greenwald  

**Link**: [PDF](https://arxiv.org/pdf/2505.11714)  

**Abstract**: The dependency of the actor on the critic in actor-critic (AC) reinforcement learning means that AC can be characterized as a bilevel optimization (BLO) problem, also called a Stackelberg game. This characterization motivates two modifications to vanilla AC algorithms. First, the critic's update should be nested to learn a best response to the actor's policy. Second, the actor should update according to a hypergradient that takes changes in the critic's behavior into account. Computing this hypergradient involves finding an inverse Hessian vector product, a process that can be numerically unstable. We thus propose a new algorithm, Bilevel Policy Optimization with Nyström Hypergradients (BLPO), which uses nesting to account for the nested structure of BLO, and leverages the Nyström method to compute the hypergradient. Theoretically, we prove BLPO converges to (a point that satisfies the necessary conditions for) a local strong Stackelberg equilibrium in polynomial time with high probability, assuming a linear parametrization of the critic's objective. Empirically, we demonstrate that BLPO performs on par with or better than PPO on a variety of discrete and continuous control tasks. 

---
# Qronos: Correcting the Past by Shaping the Future... in Post-Training Quantization 

**Authors**: Shihao Zhang, Haoyu Zhang, Ian Colbert, Rayan Saab  

**Link**: [PDF](https://arxiv.org/pdf/2505.11695)  

**Abstract**: We introduce Qronos -- a new state-of-the-art post-training quantization algorithm that sequentially rounds and updates neural network weights. Qronos not only explicitly corrects errors due to both weight and activation quantization, but also errors resulting from quantizing previous layers. Our iterative algorithm is based on an interpretable and disciplined optimization framework that subsumes and surpasses existing data-driven approaches. At each step, Qronos alternates between error correction and diffusion via optimal update rules. Importantly, we prove that Qronos admits an efficient implementation that uses the Cholesky decomposition for solving least-squares problems. We also demonstrate that Qronos is compatible with existing transformation techniques such as Hadamard-based incoherence processing and weight-activation scaling equalization, among others. We evaluate Qronos using recent autoregressive language generation models in the Llama3 family; Qronos consistently outperforms previous state-of-the-art adaptive rounding methods when quantizing the weights, activations, and/or KV caches. 

---
# Neural Networks as Universal Finite-State Machines: A Constructive Deterministic Finite Automaton Theory 

**Authors**: Sahil Rajesh Dhayalkar  

**Link**: [PDF](https://arxiv.org/pdf/2505.11694)  

**Abstract**: We present a complete theoretical and empirical framework establishing feedforward neural networks as universal finite-state machines (N-FSMs). Our results prove that finite-depth ReLU and threshold networks can exactly simulate deterministic finite automata (DFAs) by unrolling state transitions into depth-wise neural layers, with formal characterizations of required depth, width, and state compression. We demonstrate that DFA transitions are linearly separable, binary threshold activations allow exponential compression, and Myhill-Nerode equivalence classes can be embedded into continuous latent spaces while preserving separability. We also formalize the expressivity boundary: fixed-depth feedforward networks cannot recognize non-regular languages requiring unbounded memory. Unlike prior heuristic or probing-based studies, we provide constructive proofs and design explicit DFA-unrolled neural architectures that empirically validate every claim. Our results bridge deep learning, automata theory, and neural-symbolic computation, offering a rigorous blueprint for how discrete symbolic processes can be realized in continuous neural systems. 

---
# The Geometry of ReLU Networks through the ReLU Transition Graph 

**Authors**: Sahil Rajesh Dhayalkar  

**Link**: [PDF](https://arxiv.org/pdf/2505.11692)  

**Abstract**: We develop a novel theoretical framework for analyzing ReLU neural networks through the lens of a combinatorial object we term the ReLU Transition Graph (RTG). In this graph, each node corresponds to a linear region induced by the network's activation patterns, and edges connect regions that differ by a single neuron flip. Building on this structure, we derive a suite of new theoretical results connecting RTG geometry to expressivity, generalization, and robustness. Our contributions include tight combinatorial bounds on RTG size and diameter, a proof of RTG connectivity, and graph-theoretic interpretations of VC-dimension. We also relate entropy and average degree of the RTG to generalization error. Each theoretical result is rigorously validated via carefully controlled experiments across varied network depths, widths, and data regimes. This work provides the first unified treatment of ReLU network structure via graph theory and opens new avenues for compression, regularization, and complexity control rooted in RTG analysis. 

---
# Second SIGIR Workshop on Simulations for Information Access (Sim4IA 2025) 

**Authors**: Philipp Schaer, Christin Katharina Kreutz, Krisztian Balog, Timo Breuer, Andreas Konstantin Kruff  

**Link**: [PDF](https://arxiv.org/pdf/2505.11687)  

**Abstract**: Simulations in information access (IA) have recently gained interest, as shown by various tutorials and workshops around that topic. Simulations can be key contributors to central IA research and evaluation questions, especially around interactive settings when real users are unavailable, or their participation is impossible due to ethical reasons. In addition, simulations in IA can help contribute to a better understanding of users, reduce complexity of evaluation experiments, and improve reproducibility. Building on recent developments in methods and toolkits, the second iteration of our Sim4IA workshop aims to again bring together researchers and practitioners to form an interactive and engaging forum for discussions on the future perspectives of the field. An additional aim is to plan an upcoming TREC/CLEF campaign. 

---
# OT Score: An OT based Confidence Score for Unsupervised Domain Adaptation 

**Authors**: Yiming Zhang, Sitong Liu, Alex Cloninger  

**Link**: [PDF](https://arxiv.org/pdf/2505.11669)  

**Abstract**: We address the computational and theoretical limitations of existing distributional alignment methods for unsupervised domain adaptation (UDA), particularly regarding the estimation of classification performance and confidence without target labels. Current theoretical frameworks for these methods often yield computationally intractable quantities and fail to adequately reflect the properties of the alignment algorithms employed. To overcome these challenges, we introduce the Optimal Transport (OT) score, a confidence metric derived from a novel theoretical analysis that exploits the flexibility of decision boundaries induced by Semi-Discrete Optimal Transport alignment. The proposed OT score is intuitively interpretable, theoretically rigorous, and computationally efficient. It provides principled uncertainty estimates for any given set of target pseudo-labels without requiring model retraining, and can flexibly adapt to varying degrees of available source information. Experimental results on standard UDA benchmarks demonstrate that classification accuracy consistently improves by identifying and removing low-confidence predictions, and that OT score significantly outperforms existing confidence metrics across diverse adaptation scenarios. 

---
# Multilingual Prompt Engineering in Large Language Models: A Survey Across NLP Tasks 

**Authors**: Shubham Vatsal, Harsh Dubey, Aditi Singh  

**Link**: [PDF](https://arxiv.org/pdf/2505.11665)  

**Abstract**: Large language models (LLMs) have demonstrated impressive performance across a wide range of Natural Language Processing (NLP) tasks. However, ensuring their effectiveness across multiple languages presents unique challenges. Multilingual prompt engineering has emerged as a key approach to enhance LLMs' capabilities in diverse linguistic settings without requiring extensive parameter re-training or fine-tuning. With growing interest in multilingual prompt engineering over the past two to three years, researchers have explored various strategies to improve LLMs' performance across languages and NLP tasks. By crafting structured natural language prompts, researchers have successfully extracted knowledge from LLMs across different languages, making these techniques an accessible pathway for a broader audience, including those without deep expertise in machine learning, to harness the capabilities of LLMs. In this paper, we survey and categorize different multilingual prompting techniques based on the NLP tasks they address across a diverse set of datasets that collectively span around 250 languages. We further highlight the LLMs employed, present a taxonomy of approaches and discuss potential state-of-the-art (SoTA) methods for specific multilingual datasets. Additionally, we derive a range of insights across language families and resource levels (high-resource vs. low-resource), including analyses such as the distribution of NLP tasks by language resource type and the frequency of prompting methods across different language families. Our survey reviews 36 research papers covering 39 prompting techniques applied to 30 multilingual NLP tasks, with the majority of these studies published in the last two years. 

---
# Programmable metasurfaces for future photonic artificial intelligence 

**Authors**: Loubnan Abou-Hamdan, Emil Marinov, Peter Wiecha, Philipp del Hougne, Tianyu Wang, Patrice Genevet  

**Link**: [PDF](https://arxiv.org/pdf/2505.11659)  

**Abstract**: Photonic neural networks (PNNs), which share the inherent benefits of photonic systems, such as high parallelism and low power consumption, could challenge traditional digital neural networks in terms of energy efficiency, latency, and throughput. However, producing scalable photonic artificial intelligence (AI) solutions remains challenging. To make photonic AI models viable, the scalability problem needs to be solved. Large optical AI models implemented on PNNs are only commercially feasible if the advantages of optical computation outweigh the cost of their input-output overhead. In this Perspective, we discuss how field-programmable metasurface technology may become a key hardware ingredient in achieving scalable photonic AI accelerators and how it can compete with current digital electronic technologies. Programmability or reconfigurability is a pivotal component for PNN hardware, enabling in situ training and accommodating non-stationary use cases that require fine-tuning or transfer learning. Co-integration with electronics, 3D stacking, and large-scale manufacturing of metasurfaces would significantly improve PNN scalability and functionalities. Programmable metasurfaces could address some of the current challenges that PNNs face and enable next-generation photonic AI technology. 

---
# PeerGuard: Defending Multi-Agent Systems Against Backdoor Attacks Through Mutual Reasoning 

**Authors**: Falong Fan, Xi Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.11642)  

**Abstract**: Multi-agent systems leverage advanced AI models as autonomous agents that interact, cooperate, or compete to complete complex tasks across applications such as robotics and traffic management. Despite their growing importance, safety in multi-agent systems remains largely underexplored, with most research focusing on single AI models rather than interacting agents. This work investigates backdoor vulnerabilities in multi-agent systems and proposes a defense mechanism based on agent interactions. By leveraging reasoning abilities, each agent evaluates responses from others to detect illogical reasoning processes, which indicate poisoned agents. Experiments on LLM-based multi-agent systems, including ChatGPT series and Llama 3, demonstrate the effectiveness of the proposed method, achieving high accuracy in identifying poisoned agents while minimizing false positives on clean agents. We believe this work provides insights into multi-agent system safety and contributes to the development of robust, trustworthy AI interactions. 

---
# Chatting with Papers: A Hybrid Approach Using LLMs and Knowledge Graphs 

**Authors**: Vyacheslav Tykhonov, Han Yang, Philipp Mayr, Jetze Touber, Andrea Scharnhorst  

**Link**: [PDF](https://arxiv.org/pdf/2505.11633)  

**Abstract**: This demo paper reports on a new workflow \textit{GhostWriter} that combines the use of Large Language Models and Knowledge Graphs (semantic artifacts) to support navigation through collections. Situated in the research area of Retrieval Augmented Generation, this specific workflow details the creation of local and adaptable chatbots. Based on the tool-suite \textit{EverythingData} at the backend, \textit{GhostWriter} provides an interface that enables querying and ``chatting'' with a collection. Applied iteratively, the workflow supports the information needs of researchers when interacting with a collection of papers, whether it be to gain an overview, to learn more about a specific concept and its context, and helps the researcher ultimately to refine their research question in a controlled way. We demonstrate the workflow for a collection of articles from the \textit{method data analysis} journal published by GESIS -- Leibniz-Institute for the Social Sciences. We also point to further application areas. 

---
# Nearest Neighbor Multivariate Time Series Forecasting 

**Authors**: Huiliang Zhang, Ping Nie, Lijun Sun, Benoit Boulet  

**Link**: [PDF](https://arxiv.org/pdf/2505.11625)  

**Abstract**: Multivariate time series (MTS) forecasting has a wide range of applications in both industry and academia. Recently, spatial-temporal graph neural networks (STGNNs) have gained popularity as MTS forecasting methods. However, current STGNNs can only use the finite length of MTS input data due to the computational complexity. Moreover, they lack the ability to identify similar patterns throughout the entire dataset and struggle with data that exhibit sparsely and discontinuously distributed correlations among variables over an extensive historical period, resulting in only marginal improvements. In this article, we introduce a simple yet effective k-nearest neighbor MTS forecasting ( kNN-MTS) framework, which forecasts with a nearest neighbor retrieval mechanism over a large datastore of cached series, using representations from the MTS model for similarity search. This approach requires no additional training and scales to give the MTS model direct access to the whole dataset at test time, resulting in a highly expressive model that consistently improves performance, and has the ability to extract sparse distributed but similar patterns spanning over multivariables from the entire dataset. Furthermore, a hybrid spatial-temporal encoder (HSTEncoder) is designed for kNN-MTS which can capture both long-term temporal and short-term spatial-temporal dependencies and is shown to provide accurate representation for kNN-MTSfor better forecasting. Experimental results on several real-world datasets show a significant improvement in the forecasting performance of kNN-MTS. The quantitative analysis also illustrates the interpretability and efficiency of kNN-MTS, showing better application prospects and opening up a new path for efficiently using the large dataset in MTS models. 

---
# A Classical View on Benign Overfitting: The Role of Sample Size 

**Authors**: Junhyung Park, Patrick Bloebaum, Shiva Prasad Kasiviswanathan  

**Link**: [PDF](https://arxiv.org/pdf/2505.11621)  

**Abstract**: Benign overfitting is a phenomenon in machine learning where a model perfectly fits (interpolates) the training data, including noisy examples, yet still generalizes well to unseen data. Understanding this phenomenon has attracted considerable attention in recent years. In this work, we introduce a conceptual shift, by focusing on almost benign overfitting, where models simultaneously achieve both arbitrarily small training and test errors. This behavior is characteristic of neural networks, which often achieve low (but non-zero) training error while still generalizing well. We hypothesize that this almost benign overfitting can emerge even in classical regimes, by analyzing how the interaction between sample size and model complexity enables larger models to achieve both good training fit but still approach Bayes-optimal generalization. We substantiate this hypothesis with theoretical evidence from two case studies: (i) kernel ridge regression, and (ii) least-squares regression using a two-layer fully connected ReLU neural network trained via gradient flow. In both cases, we overcome the strong assumptions often required in prior work on benign overfitting.
Our results on neural networks also provide the first generalization result in this setting that does not rely on any assumptions about the underlying regression function or noise, beyond boundedness. Our analysis introduces a novel proof technique based on decomposing the excess risk into estimation and approximation errors, interpreting gradient flow as an implicit regularizer, that helps avoid uniform convergence traps. This analysis idea could be of independent interest. 

---
# Steering Risk Preferences in Large Language Models by Aligning Behavioral and Neural Representations 

**Authors**: Jian-Qiao Zhu, Haijiang Yan, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2505.11615)  

**Abstract**: Changing the behavior of large language models (LLMs) can be as straightforward as editing the Transformer's residual streams using appropriately constructed "steering vectors." These modifications to internal neural activations, a form of representation engineering, offer an effective and targeted means of influencing model behavior without retraining or fine-tuning the model. But how can such steering vectors be systematically identified? We propose a principled approach for uncovering steering vectors by aligning latent representations elicited through behavioral methods (specifically, Markov chain Monte Carlo with LLMs) with their neural counterparts. To evaluate this approach, we focus on extracting latent risk preferences from LLMs and steering their risk-related outputs using the aligned representations as steering vectors. We show that the resulting steering vectors successfully and reliably modulate LLM outputs in line with the targeted behavior. 

---
# Continuous Optimization for Feature Selection with Permutation-Invariant Embedding and Policy-Guided Search 

**Authors**: Rui Liu, Rui Xie, Zijun Yao, Yanjie Fu, Dongjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11601)  

**Abstract**: Feature selection removes redundant features to enhanc performance and computational efficiency in downstream tasks. Existing works often struggle to capture complex feature interactions and adapt to diverse scenarios. Recent advances in this domain have incorporated generative intelligence to address these drawbacks by uncovering intricate relationships between features. However, two key limitations remain: 1) embedding feature subsets in a continuous space is challenging due to permutation sensitivity, as changes in feature order can introduce biases and weaken the embedding learning process; 2) gradient-based search in the embedding space assumes convexity, which is rarely guaranteed, leading to reduced search effectiveness and suboptimal subsets. To address these limitations, we propose a new framework that can: 1) preserve feature subset knowledge in a continuous embedding space while ensuring permutation invariance; 2) effectively explore the embedding space without relying on strong convex assumptions. For the first objective, we develop an encoder-decoder paradigm to preserve feature selection knowledge into a continuous embedding space. This paradigm captures feature interactions through pairwise relationships within the subset, removing the influence of feature order on the embedding. Moreover, an inducing point mechanism is introduced to accelerate pairwise relationship computations. For the second objective, we employ a policy-based reinforcement learning (RL) approach to guide the exploration of the embedding space. The RL agent effectively navigates the space by balancing multiple objectives. By prioritizing high-potential regions adaptively and eliminating the reliance on convexity assumptions, the RL agent effectively reduces the risk of converging to local optima. Extensive experiments demonstrate the effectiveness, efficiency, robustness and explicitness of our model. 

---
# Spectral Policy Optimization: Coloring your Incorrect Reasoning in GRPO 

**Authors**: Peter Chen, Xiaopeng Li, Ziniu Li, Xi Chen, Tianyi Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.11595)  

**Abstract**: Reinforcement learning (RL) has demonstrated significant success in enhancing reasoning capabilities in large language models (LLMs). One of the most widely used RL methods is Group Relative Policy Optimization (GRPO)~\cite{Shao-2024-Deepseekmath}, known for its memory efficiency and success in training DeepSeek-R1~\cite{Guo-2025-Deepseek}. However, GRPO stalls when all sampled responses in a group are incorrect -- referred to as an \emph{all-negative-sample} group -- as it fails to update the policy, hindering learning progress. The contributions of this paper are two-fold. First, we propose a simple yet effective framework that introduces response diversity within all-negative-sample groups in GRPO using AI feedback. We also provide a theoretical analysis, via a stylized model, showing how this diversification improves learning dynamics. Second, we empirically validate our approach, showing the improved performance across various model sizes (7B, 14B, 32B) in both offline and online learning settings with 10 benchmarks, including base and distilled variants. Our findings highlight that learning from all-negative-sample groups is not only feasible but beneficial, advancing recent insights from \citet{Xiong-2025-Minimalist}. 

---
# SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training 

**Authors**: Jintao Zhang, Jia Wei, Pengle Zhang, Xiaoming Xu, Haofeng Huang, Haoxu Wang, Kai Jiang, Jun Zhu, Jianfei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.11594)  

**Abstract**: The efficiency of attention is important due to its quadratic time complexity. We enhance the efficiency of attention through two key contributions: First, we leverage the new FP4 Tensor Cores in Blackwell GPUs to accelerate attention computation. Our implementation achieves 1038 TOPS on RTX5090, which is a 5x speedup over the fastest FlashAttention on RTX5090. Experiments show that our FP4 attention can accelerate inference of various models in a plug-and-play way. Second, we pioneer low-bit attention to training tasks. Existing low-bit attention works like FlashAttention3 and SageAttention focus only on inference. However, the efficiency of training large models is also important. To explore whether low-bit attention can be effectively applied to training tasks, we design an accurate and efficient 8-bit attention for both forward and backward propagation. Experiments indicate that 8-bit attention achieves lossless performance in fine-tuning tasks but exhibits slower convergence in pretraining tasks. The code will be available at this https URL. 

---
# The Ripple Effect: On Unforeseen Complications of Backdoor Attacks 

**Authors**: Rui Zhang, Yun Shen, Hongwei Li, Wenbo Jiang, Hanxiao Chen, Yuan Zhang, Guowen Xu, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11586)  

**Abstract**: Recent research highlights concerns about the trustworthiness of third-party Pre-Trained Language Models (PTLMs) due to potential backdoor attacks. These backdoored PTLMs, however, are effective only for specific pre-defined downstream tasks. In reality, these PTLMs can be adapted to many other unrelated downstream tasks. Such adaptation may lead to unforeseen consequences in downstream model outputs, consequently raising user suspicion and compromising attack stealthiness. We refer to this phenomenon as backdoor complications. In this paper, we undertake the first comprehensive quantification of backdoor complications. Through extensive experiments using 4 prominent PTLMs and 16 text classification benchmark datasets, we demonstrate the widespread presence of backdoor complications in downstream models fine-tuned from backdoored PTLMs. The output distribution of triggered samples significantly deviates from that of clean samples. Consequently, we propose a backdoor complication reduction method leveraging multi-task learning to mitigate complications without prior knowledge of downstream tasks. The experimental results demonstrate that our proposed method can effectively reduce complications while maintaining the efficacy and consistency of backdoor attacks. Our code is available at this https URL. 

---
# Comparing Lexical and Semantic Vector Search Methods When Classifying Medical Documents 

**Authors**: Lee Harris, Philippe De Wilde, James Bentham  

**Link**: [PDF](https://arxiv.org/pdf/2505.11582)  

**Abstract**: Classification is a common AI problem, and vector search is a typical solution. This transforms a given body of text into a numerical representation, known as an embedding, and modern improvements to vector search focus on optimising speed and predictive accuracy. This is often achieved through neural methods that aim to learn language semantics. However, our results suggest that these are not always the best solution. Our task was to classify rigidly-structured medical documents according to their content, and we found that using off-the-shelf semantic vector search produced slightly worse predictive accuracy than creating a bespoke lexical vector search model, and that it required significantly more time to execute. These findings suggest that traditional methods deserve to be contenders in the information retrieval toolkit, despite the prevalence and success of neural models. 

---
# Flash Invariant Point Attention 

**Authors**: Andrew Liu, Axel Elaldi, Nicholas T Franklin, Nathan Russell, Gurinder S Atwal, Yih-En A Ban, Olivia Viessmann  

**Link**: [PDF](https://arxiv.org/pdf/2505.11580)  

**Abstract**: Invariant Point Attention (IPA) is a key algorithm for geometry-aware modeling in structural biology, central to many protein and RNA models. However, its quadratic complexity limits the input sequence length. We introduce FlashIPA, a factorized reformulation of IPA that leverages hardware-efficient FlashAttention to achieve linear scaling in GPU memory and wall-clock time with sequence length. FlashIPA matches or exceeds standard IPA performance while substantially reducing computational costs. FlashIPA extends training to previously unattainable lengths, and we demonstrate this by re-training generative models without length restrictions and generating structures of thousands of residues. FlashIPA is available at this https URL. 

---
# Toward Adaptive Categories: Dimensional Governance for Agentic AI 

**Authors**: Zeynep Engin, David Hand  

**Link**: [PDF](https://arxiv.org/pdf/2505.11579)  

**Abstract**: As AI systems evolve from static tools to dynamic agents, traditional categorical governance frameworks -- based on fixed risk tiers, levels of autonomy, or human oversight models -- are increasingly insufficient on their own. Systems built on foundation models, self-supervised learning, and multi-agent architectures increasingly blur the boundaries that categories were designed to police. In this Perspective, we make the case for dimensional governance: a framework that tracks how decision authority, process autonomy, and accountability (the 3As) distribute dynamically across human-AI relationships. A critical advantage of this approach is its ability to explicitly monitor system movement toward and across key governance thresholds, enabling preemptive adjustments before risks materialize. This dimensional approach provides the necessary foundation for more adaptive categorization, enabling thresholds and classifications that can evolve with emerging capabilities. While categories remain essential for decision-making, building them upon dimensional foundations allows for context-specific adaptability and stakeholder-responsive governance that static approaches cannot achieve. We outline key dimensions, critical trust thresholds, and practical examples illustrating where rigid categorical frameworks fail -- and where a dimensional mindset could offer a more resilient and future-proof path forward for both governance and innovation at the frontier of artificial intelligence. 

---
# Spatiotemporal Field Generation Based on Hybrid Mamba-Transformer with Physics-informed Fine-tuning 

**Authors**: Peimian Du, Jiabin Liu, Xiaowei Jin, Mengwang Zuo, Hui Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.11578)  

**Abstract**: This research confronts the challenge of substantial physical equation discrepancies encountered in the generation of spatiotemporal physical fields through data-driven trained models. A spatiotemporal physical field generation model, named HMT-PF, is developed based on the hybrid Mamba-Transformer architecture, incorporating unstructured grid information as input. A fine-tuning block, enhanced with physical information, is introduced to effectively reduce the physical equation discrepancies. The physical equation residuals are computed through a point query mechanism for efficient gradient evaluation, then encoded into latent space for refinement. The fine-tuning process employs a self-supervised learning approach to achieve physical consistency while maintaining essential field characteristics. Results show that the hybrid Mamba-Transformer model achieves good performance in generating spatiotemporal fields, while the physics-informed fine-tuning mechanism further reduces significant physical errors effectively. A MSE-R evaluation method is developed to assess the accuracy and realism of physical field generation. 

---
# The Accountability Paradox: How Platform API Restrictions Undermine AI Transparency Mandates 

**Authors**: FLorian A.D. Burnat, Brittany I. Davidson  

**Link**: [PDF](https://arxiv.org/pdf/2505.11577)  

**Abstract**: Recent application programming interface (API) restrictions on major social media platforms challenge compliance with the EU Digital Services Act [20], which mandates data access for algorithmic transparency. We develop a structured audit framework to assess the growing misalignment between regulatory requirements and platform implementations. Our comparative analysis of X/Twitter, Reddit, TikTok, and Meta identifies critical ``audit blind-spots'' where platform content moderation and algorithmic amplification remain inaccessible to independent verification. Our findings reveal an ``accountability paradox'': as platforms increasingly rely on AI systems, they simultaneously restrict the capacity for independent oversight. We propose targeted policy interventions aligned with the AI Risk Management Framework of the National Institute of Standards and Technology [80], emphasizing federated access models and enhanced regulatory enforcement. 

---
# Concept-Guided Interpretability via Neural Chunking 

**Authors**: Shuchen Wu, Stephan Alaniz, Shyamgopal Karthik, Peter Dayan, Eric Schulz, Zeynep Akata  

**Link**: [PDF](https://arxiv.org/pdf/2505.11576)  

**Abstract**: Neural networks are often black boxes, reflecting the significant challenge of understanding their internal workings. We propose a different perspective that challenges the prevailing view: rather than being inscrutable, neural networks exhibit patterns in their raw population activity that mirror regularities in the training data. We refer to this as the Reflection Hypothesis and provide evidence for this phenomenon in both simple recurrent neural networks (RNNs) and complex large language models (LLMs). Building on this insight, we propose to leverage cognitively-inspired methods of chunking to segment high-dimensional neural population dynamics into interpretable units that reflect underlying concepts. We propose three methods to extract these emerging entities, complementing each other based on label availability and dimensionality. Discrete sequence chunking (DSC) creates a dictionary of entities; population averaging (PA) extracts recurring entities that correspond to known labels; and unsupervised chunk discovery (UCD) can be used when labels are absent. We demonstrate the effectiveness of these methods in extracting entities across varying model sizes, ranging from inducing compositionality in RNNs to uncovering recurring neural population states in large models with diverse architectures, and illustrate their advantage over other methods. Throughout, we observe a robust correspondence between the extracted entities and concrete or abstract concepts. Artificially inducing the extracted entities in neural populations effectively alters the network's generation of associated concepts. Our work points to a new direction for interpretability, one that harnesses both cognitive principles and the structure of naturalistic data to reveal the hidden computations of complex learning systems, gradually transforming them from black boxes into systems we can begin to understand. 

---
# InfiJanice: Joint Analysis and In-situ Correction Engine for Quantization-Induced Math Degradation in Large Language Models 

**Authors**: Zhen Li, Yupeng Su, Songmiao Wang, Runming Yang, Congkai Xie, Aofan Liu, Ming Li, Jiannong Cao, Yuan Xie, Ngai Wong, Hongxia Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11574)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive performance on complex reasoning benchmarks such as GSM8K, MATH, and AIME. However, the substantial computational demands of these tasks pose significant challenges for real-world deployment. Model quantization has emerged as a promising approach to reduce memory footprint and inference latency by representing weights and activations with lower bit-widths. In this work, we conduct a comprehensive study of mainstream quantization methods(e.g., AWQ, GPTQ, SmoothQuant) on the most popular open-sourced models (e.g., Qwen2.5, LLaMA3 series), and reveal that quantization can degrade mathematical reasoning accuracy by up to 69.81%. To better understand this degradation, we develop an automated assignment and judgment pipeline that qualitatively categorizes failures into four error types and quantitatively identifies the most impacted reasoning capabilities. Building on these findings, we employ an automated data-curation pipeline to construct a compact "Silver Bullet" datasets. Training a quantized model on as few as 332 carefully selected examples for just 3-5 minutes on a single GPU is enough to restore its reasoning accuracy to match that of the full-precision baseline. 

---
# Tool-Aided Evolutionary LLM for Generative Policy Toward Efficient Resource Management in Wireless Federated Learning 

**Authors**: Chongyang Tan, Ruoqi Wen, Rongpeng Li, Zhifeng Zhao, Ekram Hossain, Honggang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11570)  

**Abstract**: Federated Learning (FL) enables distributed model training across edge devices in a privacy-friendly manner. However, its efficiency heavily depends on effective device selection and high-dimensional resource allocation in dynamic and heterogeneous wireless environments. Conventional methods demand a confluence of domain-specific expertise, extensive hyperparameter tuning, and/or heavy interaction cost. This paper proposes a Tool-aided Evolutionary Large Language Model (T-ELLM) framework to generate a qualified policy for device selection in a wireless FL environment. Unlike conventional optimization methods, T-ELLM leverages natural language-based scenario prompts to enhance generalization across varying network conditions. The framework decouples the joint optimization problem mathematically, enabling tractable learning of device selection policies while delegating resource allocation to convex optimization tools. To improve adaptability, T-ELLM integrates a sample-efficient, model-based virtual learning environment that captures the relationship between device selection and learning performance, facilitating subsequent group relative policy optimization. This concerted approach reduces reliance on real-world interactions, minimizing communication overhead while maintaining high-fidelity decision-making. Theoretical analysis proves that the discrepancy between virtual and real environments is bounded, ensuring the advantage function learned in the virtual environment maintains a provably small deviation from real-world conditions. Experimental results demonstrate that T-ELLM outperforms benchmark methods in energy efficiency and exhibits robust adaptability to environmental changes. 

---
# Towards Adaptive Deep Learning: Model Elasticity via Prune-and-Grow CNN Architectures 

**Authors**: Pooja Mangal, Sudaksh Kalra, Dolly Sapra  

**Link**: [PDF](https://arxiv.org/pdf/2505.11569)  

**Abstract**: Deploying deep convolutional neural networks (CNNs) on resource-constrained devices presents significant challenges due to their high computational demands and rigid, static architectures. To overcome these limitations, this thesis explores methods for enabling CNNs to dynamically adjust their computational complexity based on available hardware resources. We introduce adaptive CNN architectures capable of scaling their capacity at runtime, thus efficiently balancing performance and resource utilization. To achieve this adaptability, we propose a structured pruning and dynamic re-construction approach that creates nested subnetworks within a single CNN model. This approach allows the network to dynamically switch between compact and full-sized configurations without retraining, making it suitable for deployment across varying hardware platforms. Experiments conducted across multiple CNN architectures including VGG-16, AlexNet, ResNet-20, and ResNet-56 on CIFAR-10 and Imagenette datasets demonstrate that adaptive models effectively maintain or even enhance performance under varying computational constraints. Our results highlight that embedding adaptability directly into CNN architectures significantly improves their robustness and flexibility, paving the way for efficient real-world deployment in diverse computational environments. 

---
# BioCube: A Multimodal Dataset for Biodiversity Research 

**Authors**: Stylianos Stasinos, Martino Mensio, Elena Lazovik, Athanasios Trantas  

**Link**: [PDF](https://arxiv.org/pdf/2505.11568)  

**Abstract**: Biodiversity research requires complete and detailed information to study ecosystem dynamics at different scales. Employing data-driven methods like Machine Learning is getting traction in ecology and more specific biodiversity, offering alternative modelling pathways. For these methods to deliver accurate results there is the need for large, curated and multimodal datasets that offer granular spatial and temporal resolutions. In this work, we introduce BioCube, a multimodal, fine-grained global dataset for ecology and biodiversity research. BioCube incorporates species observations through images, audio recordings and descriptions, environmental DNA, vegetation indices, agricultural, forest, land indicators, and high-resolution climate variables. All observations are geospatially aligned under the WGS84 geodetic system, spanning from 2000 to 2020. The dataset will become available at this https URL while the acquisition and processing code base at this https URL. 

---
# Beyond Time: Cross-Dimensional Frequency Supervision for Time Series Forecasting 

**Authors**: Tianyi Shi, Zhu Meng, Yue Chen, Siyang Zheng, Fei Su, Jin Huang, Changrui Ren, Zhicheng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.11567)  

**Abstract**: Time series forecasting plays a crucial role in various fields, and the methods based on frequency domain analysis have become an important branch. However, most existing studies focus on the design of elaborate model architectures and are often tailored for limited datasets, still lacking universality. Besides, the assumption of independent and identically distributed (IID) data also contradicts the strong correlation of the time domain labels. To address these issues, abandoning time domain supervision, we propose a purely frequency domain supervision approach named cross-dimensional frequency (X-Freq) loss. Specifically, based on a statistical phenomenon, we first prove that the information entropy of the time series is higher than its spectral entropy, which implies higher certainty in frequency domain and thus can provide better supervision. Secondly, the Fourier Transform and the Wavelet Transform are applied to the time dimension and the channel dimension of the time series respectively, to capture the long-term and short-term frequency variations as well as the spatial configuration features. Thirdly, the loss between predictions and targets is uniformly computed in the frequency domain. Moreover, we plug-and-play incorporate X-Freq into multiple advanced forecasting models and compare on 14 real-world datasets. The experimental results demonstrate that, without making any modification to the original architectures or hyperparameters, X-Freq can improve the forecasting performance by an average of 3.3% on long-term forecasting datasets and 27.7% on short-term ones, showcasing superior generality and practicality. The code will be released publicly. 

---
# ACSE-Eval: Can LLMs threat model real-world cloud infrastructure? 

**Authors**: Sarthak Munshi, Swapnil Pathak, Sonam Ghatode, Thenuga Priyadarshini, Dhivya Chandramouleeswaran, Ashutosh Rana  

**Link**: [PDF](https://arxiv.org/pdf/2505.11565)  

**Abstract**: While Large Language Models have shown promise in cybersecurity applications, their effectiveness in identifying security threats within cloud deployments remains unexplored. This paper introduces AWS Cloud Security Engineering Eval, a novel dataset for evaluating LLMs cloud security threat modeling capabilities. ACSE-Eval contains 100 production grade AWS deployment scenarios, each featuring detailed architectural specifications, Infrastructure as Code implementations, documented security vulnerabilities, and associated threat modeling parameters. Our dataset enables systemic assessment of LLMs abilities to identify security risks, analyze attack vectors, and propose mitigation strategies in cloud environments. Our evaluations on ACSE-Eval demonstrate that GPT 4.1 and Gemini 2.5 Pro excel at threat identification, with Gemini 2.5 Pro performing optimally in 0-shot scenarios and GPT 4.1 showing superior results in few-shot settings. While GPT 4.1 maintains a slight overall performance advantage, Claude 3.7 Sonnet generates the most semantically sophisticated threat models but struggles with threat categorization and generalization. To promote reproducibility and advance research in automated cybersecurity threat analysis, we open-source our dataset, evaluation metrics, and methodologies. 

---
# Object-Centric Representations Improve Policy Generalization in Robot Manipulation 

**Authors**: Alexandre Chapin, Bruno Machado, Emmanuel Dellandrea, Liming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.11563)  

**Abstract**: Visual representations are central to the learning and generalization capabilities of robotic manipulation policies. While existing methods rely on global or dense features, such representations often entangle task-relevant and irrelevant scene information, limiting robustness under distribution shifts. In this work, we investigate object-centric representations (OCR) as a structured alternative that segments visual input into a finished set of entities, introducing inductive biases that align more naturally with manipulation tasks. We benchmark a range of visual encoders-object-centric, global and dense methods-across a suite of simulated and real-world manipulation tasks ranging from simple to complex, and evaluate their generalization under diverse visual conditions including changes in lighting, texture, and the presence of distractors. Our findings reveal that OCR-based policies outperform dense and global representations in generalization settings, even without task-specific pretraining. These insights suggest that OCR is a promising direction for designing visual systems that generalize effectively in dynamic, real-world robotic environments. 

---
# Analysis and Resilience of the U.S. Flight Network 

**Authors**: Sushrit Kafle, Shreejan Pandey  

**Link**: [PDF](https://arxiv.org/pdf/2505.11559)  

**Abstract**: Air travel is one of the most widely used transportation services in the United States. This paper analyzes the U.S. Flight Network (USFN) using complex network theory by exploring how the network's topology contributes to its efficiency and vulnerability. This is done by examining the structural properties, degree distributions, and community structures in the network. USFN was observed to follow power-law distribution and falls under the anomalous regime, suggesting that the network is hub dominant. Compared to null networks, USFN has a higher clustering coefficient and modularity. Various percolation test revealed that USFN is vulnerable to targeted attacks and is susceptible to complete cascading failure if one of the major hubs fails. The overall results suggest that while the USFN is designed for efficiency, it is highly vulnerable to disruptions. Protecting key hub airports is important to make the network more robust and prevent large-scale failures. 

---
# AC-LoRA: (Almost) Training-Free Access Control-Aware Multi-Modal LLMs 

**Authors**: Lara Magdalena Lazier, Aritra Dhar, Vasilije Stambolic, Lukas Cavigelli  

**Link**: [PDF](https://arxiv.org/pdf/2505.11557)  

**Abstract**: Corporate LLMs are gaining traction for efficient knowledge dissemination and management within organizations. However, as current LLMs are vulnerable to leaking sensitive information, it has proven difficult to apply them in settings where strict access control is necessary. To this end, we design AC-LoRA, an end-to-end system for access control-aware corporate LLM chatbots that maintains a strong information isolation guarantee. AC-LoRA maintains separate LoRA adapters for permissioned datasets, along with the document embedding they are finetuned on. AC-LoRA retrieves a precise set of LoRA adapters based on the similarity score with the user query and their permission. This similarity score is later used to merge the responses if more than one LoRA is retrieved, without requiring any additional training for LoRA routing. We provide an end-to-end prototype of AC-LoRA, evaluate it on two datasets, and show that AC-LoRA matches or even exceeds the performance of state-of-the-art LoRA mixing techniques while providing strong isolation guarantees. Furthermore, we show that AC-LoRA design can be directly applied to different modalities. 

---
# Assessing Collective Reasoning in Multi-Agent LLMs via Hidden Profile Tasks 

**Authors**: Yuxuan Li, Aoi Naito, Hirokazu Shirado  

**Link**: [PDF](https://arxiv.org/pdf/2505.11556)  

**Abstract**: Multi-agent systems built on large language models (LLMs) promise enhanced problem-solving through distributed information integration, but also risk replicating collective reasoning failures observed in human groups. Yet, no theory-grounded benchmark exists to systematically evaluate such failures. In this paper, we introduce the Hidden Profile paradigm from social psychology as a diagnostic testbed for multi-agent LLM systems. By distributing critical information asymmetrically across agents, the paradigm reveals how inter-agent dynamics support or hinder collective reasoning. We first formalize the paradigm for multi-agent decision-making under distributed knowledge and instantiate it as a benchmark with nine tasks spanning diverse scenarios, including adaptations from prior human studies. We then conduct experiments with GPT-4.1 and five other leading LLMs, including reasoning-enhanced variants, showing that multi-agent systems across all models fail to match the accuracy of single agents given complete information. While agents' collective performance is broadly comparable to that of human groups, nuanced behavioral differences emerge, such as increased sensitivity to social desirability. Finally, we demonstrate the paradigm's diagnostic utility by exploring a cooperation-contradiction trade-off in multi-agent LLM systems. We find that while cooperative agents are prone to over-coordination in collective settings, increased contradiction impairs group convergence. This work contributes a reproducible framework for evaluating multi-agent LLM systems and motivates future research on artificial collective intelligence and human-AI interaction. 

---
# GSPRec: Temporal-Aware Graph Spectral Filtering for Recommendation 

**Authors**: Ahmad Bin Rabiah, Julian McAuley  

**Link**: [PDF](https://arxiv.org/pdf/2505.11552)  

**Abstract**: Graph-based recommendation systems are effective at modeling collaborative patterns but often suffer from two limitations: overreliance on low-pass filtering, which suppresses user-specific signals, and omission of sequential dynamics in graph construction. We introduce GSPRec, a graph spectral model that integrates temporal transitions through sequentially-informed graph construction and applies frequency-aware filtering in the spectral domain. GSPRec encodes item transitions via multi-hop diffusion to enable the use of symmetric Laplacians for spectral processing. To capture user preferences, we design a dual-filtering mechanism: a Gaussian bandpass filter to extract mid-frequency, user-level patterns, and a low-pass filter to retain global trends. Extensive experiments on four public datasets show that GSPRec consistently outperforms baselines, with an average improvement of 6.77% in NDCG@10. Ablation studies show the complementary benefits of both sequential graph augmentation and bandpass filtering. 

---
# AI-generated Text Detection: A Multifaceted Approach to Binary and Multiclass Classification 

**Authors**: Harika Abburi, Sanmitra Bhattacharya, Edward Bowen, Nirmala Pudota  

**Link**: [PDF](https://arxiv.org/pdf/2505.11550)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in generating text that closely resembles human writing across a wide range of styles and genres. However, such capabilities are prone to potential misuse, such as fake news generation, spam email creation, and misuse in academic assignments. As a result, accurate detection of AI-generated text and identification of the model that generated it are crucial for maintaining the responsible use of LLMs. In this work, we addressed two sub-tasks put forward by the Defactify workshop under AI-Generated Text Detection shared task at the Association for the Advancement of Artificial Intelligence (AAAI 2025): Task A involved distinguishing between human-authored or AI-generated text, while Task B focused on attributing text to its originating language model. For each task, we proposed two neural architectures: an optimized model and a simpler variant. For Task A, the optimized neural architecture achieved fifth place with $F1$ score of 0.994, and for Task B, the simpler neural architecture also ranked fifth place with $F1$ score of 0.627. 

---
# One Shot Dominance: Knowledge Poisoning Attack on Retrieval-Augmented Generation Systems 

**Authors**: Zhiyuan Chang, Xiaojun Jia, Mingyang Li, Junjie Wang, Yuekai Huang, Qing Wang, Ziyou Jiang, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.11548)  

**Abstract**: Large Language Models (LLMs) enhanced with Retrieval-Augmented Generation (RAG) have shown improved performance in generating accurate responses. However, the dependence on external knowledge bases introduces potential security vulnerabilities, particularly when these knowledge bases are publicly accessible and modifiable. Poisoning attacks on knowledge bases for RAG systems face two fundamental challenges: the injected malicious content must compete with multiple authentic documents retrieved by the retriever, and LLMs tend to trust retrieved information that aligns with their internal memorized knowledge. Previous works attempt to address these challenges by injecting multiple malicious documents, but such saturation attacks are easily detectable and impractical in real-world scenarios. To enable the effective single document poisoning attack, we propose AuthChain, a novel knowledge poisoning attack method that leverages Chain-of-Evidence theory and authority effect to craft more convincing poisoned documents. AuthChain generates poisoned content that establishes strong evidence chains and incorporates authoritative statements, effectively overcoming the interference from both authentic documents and LLMs' internal knowledge. Extensive experiments across six popular LLMs demonstrate that AuthChain achieves significantly higher attack success rates while maintaining superior stealthiness against RAG defense mechanisms compared to state-of-the-art baselines. 

---
# On Technique Identification and Threat-Actor Attribution using LLMs and Embedding Models 

**Authors**: Kyla Guru, Robert J. Moss, Mykel J. Kochenderfer  

**Link**: [PDF](https://arxiv.org/pdf/2505.11547)  

**Abstract**: Attribution of cyber-attacks remains a complex but critical challenge for cyber defenders. Currently, manual extraction of behavioral indicators from dense forensic documentation causes significant attribution delays, especially following major incidents at the international scale. This research evaluates large language models (LLMs) for cyber-attack attribution based on behavioral indicators extracted from forensic documentation. We test OpenAI's GPT-4 and text-embedding-3-large for identifying threat actors' tactics, techniques, and procedures (TTPs) by comparing LLM-generated TTPs against human-generated data from MITRE ATT&CK Groups. Our framework then identifies TTPs from text using vector embedding search and builds profiles to attribute new attacks for a machine learning model to learn. Key contributions include: (1) assessing off-the-shelf LLMs for TTP extraction and attribution, and (2) developing an end-to-end pipeline from raw CTI documents to threat-actor prediction. This research finds that standard LLMs generate TTP datasets with noise, resulting in a low similarity to human-generated datasets. However, the TTPs generated are similar in frequency to those within the existing MITRE datasets. Additionally, although these TTPs are different than human-generated datasets, our work demonstrates that they still prove useful for training a model that performs above baseline on attribution. Project code and files are contained here: this https URL. 

---
# Control Invariant Sets for Neural Network Dynamical Systems and Recursive Feasibility in Model Predictive Control 

**Authors**: Xiao Li, Tianhao Wei, Changliu Liu, Anouck Girard, Ilya Kolmanovsky  

**Link**: [PDF](https://arxiv.org/pdf/2505.11546)  

**Abstract**: Neural networks are powerful tools for data-driven modeling of complex dynamical systems, enhancing predictive capability for control applications. However, their inherent nonlinearity and black-box nature challenge control designs that prioritize rigorous safety and recursive feasibility guarantees. This paper presents algorithmic methods for synthesizing control invariant sets specifically tailored to neural network based dynamical models. These algorithms employ set recursion, ensuring termination after a finite number of iterations and generating subsets in which closed-loop dynamics are forward invariant, thus guaranteeing perpetual operational safety. Additionally, we propose model predictive control designs that integrate these control invariant sets into mixed-integer optimization, with guaranteed adherence to safety constraints and recursive feasibility at the computational level. We also present a comprehensive theoretical analysis examining the properties and guarantees of the proposed methods. Numerical simulations in an autonomous driving scenario demonstrate the methods' effectiveness in synthesizing control-invariant sets offline and implementing model predictive control online, ensuring safety and recursive feasibility. 

---
# TARGET: Benchmarking Table Retrieval for Generative Tasks 

**Authors**: Xingyu Ji, Parker Glenn, Aditya G. Parameswaran, Madelon Hulsebos  

**Link**: [PDF](https://arxiv.org/pdf/2505.11545)  

**Abstract**: The data landscape is rich with structured data, often of high value to organizations, driving important applications in data analysis and machine learning. Recent progress in representation learning and generative models for such data has led to the development of natural language interfaces to structured data, including those leveraging text-to-SQL. Contextualizing interactions, either through conversational interfaces or agentic components, in structured data through retrieval-augmented generation can provide substantial benefits in the form of freshness, accuracy, and comprehensiveness of answers. The key question is: how do we retrieve the right table(s) for the analytical query or task at hand? To this end, we introduce TARGET: a benchmark for evaluating TAble Retrieval for GEnerative Tasks. With TARGET we analyze the retrieval performance of different retrievers in isolation, as well as their impact on downstream tasks. We find that dense embedding-based retrievers far outperform a BM25 baseline which is less effective than it is for retrieval over unstructured text. We also surface the sensitivity of retrievers across various metadata (e.g., missing table titles), and demonstrate a stark variation of retrieval performance across datasets and tasks. TARGET is available at this https URL. 

---
# LaDi-WM: A Latent Diffusion-based World Model for Predictive Manipulation 

**Authors**: Yuhang Huang, JIazhao Zhang, Shilong Zou, XInwang Liu, Ruizhen Hu, Kai Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.11528)  

**Abstract**: Predictive manipulation has recently gained considerable attention in the Embodied AI community due to its potential to improve robot policy performance by leveraging predicted states. However, generating accurate future visual states of robot-object interactions from world models remains a well-known challenge, particularly in achieving high-quality pixel-level representations. To this end, we propose LaDi-WM, a world model that predicts the latent space of future states using diffusion modeling. Specifically, LaDi-WM leverages the well-established latent space aligned with pre-trained Visual Foundation Models (VFMs), which comprises both geometric features (DINO-based) and semantic features (CLIP-based). We find that predicting the evolution of the latent space is easier to learn and more generalizable than directly predicting pixel-level images. Building on LaDi-WM, we design a diffusion policy that iteratively refines output actions by incorporating forecasted states, thereby generating more consistent and accurate results. Extensive experiments on both synthetic and real-world benchmarks demonstrate that LaDi-WM significantly enhances policy performance by 27.9\% on the LIBERO-LONG benchmark and 20\% on the real-world scenario. Furthermore, our world model and policies achieve impressive generalizability in real-world experiments. 

---
# Code Retrieval for MILP Instance Generation 

**Authors**: Tianxing Yang, Huigen Ye, Hua Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.11526)  

**Abstract**: Mixed-Integer Linear Programming (MILP) is widely used in fields such as scheduling, logistics, and planning. Enhancing the performance of MILP solvers, particularly learning-based solvers, requires substantial amounts of high-quality data. However, existing methods for MILP instance generation typically necessitate training a separate model for each problem class and are computationally intensive when generating new instances. To address these limitations, we reformulate the MILP Instance Generation task as MILP Code Generation task, enabling efficient, flexible, and interpretable instance generation through code. Since MILP instances generated from code can vary significantly in scale, we introduce MILP-EmbedSim, a new similarity metric that accurately measures the similarity between instances of varying sizes within the same problem class. Leveraging this metric, we propose MILP-Retrieval, a pipeline that retrieves generation code from library to produce MILP instances highly similar to target instance. MILP-Retrieval outperforms baselines in both MILP Code Generation and Instance Generation tasks, provides a novel perspective on MILP instance generation and opens new possibilities for learning-based solvers. 

---
# Decentralized Traffic Flow Optimization Through Intrinsic Motivation 

**Authors**: Himaja Papala, Daniel Polani, Stas Tiomkin  

**Link**: [PDF](https://arxiv.org/pdf/2505.11520)  

**Abstract**: Traffic congestion has long been an ubiquitous problem that is exacerbating with the rapid growth of megacities. In this proof-of-concept work we study intrinsic motivation, implemented via the empowerment principle, to control autonomous car behavior to improve traffic flow. In standard models of traffic dynamics, self-organized traffic jams emerge spontaneously from the individual behavior of cars, affecting traffic over long distances. Our novel car behavior strategy improves traffic flow while still being decentralized and using only locally available information without explicit coordination. Decentralization is essential for various reasons, not least to be able to absorb robustly substantial levels of uncertainty. Our scenario is based on the well-established traffic dynamics model, the Nagel-Schreckenberg cellular automaton. In a fraction of the cars in this model, we substitute the default behavior by empowerment, our intrinsic motivation-based method. This proposed model significantly improves overall traffic flow, mitigates congestion, and reduces the average traffic jam time. 

---
# Knowledge-enhanced Multi-perspective Video Representation Learning for Scene Recognition 

**Authors**: Xuzheng Yu, Chen Jiang, Wei Zhang, Tian Gan, Linlin Chao, Jianan Zhao, Yuan Cheng, Qingpei Guo, Wei Chu  

**Link**: [PDF](https://arxiv.org/pdf/2401.04354)  

**Abstract**: With the explosive growth of video data in real-world applications, a comprehensive representation of videos becomes increasingly important. In this paper, we address the problem of video scene recognition, whose goal is to learn a high-level video representation to classify scenes in videos. Due to the diversity and complexity of video contents in realistic scenarios, this task remains a challenge. Most existing works identify scenes for videos only from visual or textual information in a temporal perspective, ignoring the valuable information hidden in single frames, while several earlier studies only recognize scenes for separate images in a non-temporal perspective. We argue that these two perspectives are both meaningful for this task and complementary to each other, meanwhile, externally introduced knowledge can also promote the comprehension of videos. We propose a novel two-stream framework to model video representations from multiple perspectives, i.e. temporal and non-temporal perspectives, and integrate the two perspectives in an end-to-end manner by self-distillation. Besides, we design a knowledge-enhanced feature fusion and label prediction method that contributes to naturally introducing knowledge into the task of video scene recognition. Experiments conducted on a real-world dataset demonstrate the effectiveness of our proposed method. 

---
# Learning Segment Similarity and Alignment in Large-Scale Content Based Video Retrieval 

**Authors**: Chen Jiang, Kaiming Huang, Sifeng He, Xudong Yang, Wei Zhang, Xiaobo Zhang, Yuan Cheng, Lei Yang, Qing Wang, Furong Xu, Tan Pan, Wei Chu  

**Link**: [PDF](https://arxiv.org/pdf/2309.11091)  

**Abstract**: With the explosive growth of web videos in recent years, large-scale Content-Based Video Retrieval (CBVR) becomes increasingly essential in video filtering, recommendation, and copyright protection. Segment-level CBVR (S-CBVR) locates the start and end time of similar segments in finer granularity, which is beneficial for user browsing efficiency and infringement detection especially in long video scenarios. The challenge of S-CBVR task is how to achieve high temporal alignment accuracy with efficient computation and low storage consumption. In this paper, we propose a Segment Similarity and Alignment Network (SSAN) in dealing with the challenge which is firstly trained end-to-end in S-CBVR. SSAN is based on two newly proposed modules in video retrieval: (1) An efficient Self-supervised Keyframe Extraction (SKE) module to reduce redundant frame features, (2) A robust Similarity Pattern Detection (SPD) module for temporal alignment. In comparison with uniform frame extraction, SKE not only saves feature storage and search time, but also introduces comparable accuracy and limited extra computation time. In terms of temporal alignment, SPD localizes similar segments with higher accuracy and efficiency than existing deep learning methods. Furthermore, we jointly train SSAN with SKE and SPD and achieve an end-to-end improvement. Meanwhile, the two key modules SKE and SPD can also be effectively inserted into other video retrieval pipelines and gain considerable performance improvements. Experimental results on public datasets show that SSAN can obtain higher alignment accuracy while saving storage and online query computational cost compared to existing methods. 

---
