# Context is Gold to find the Gold Passage: Evaluating and Training Contextual Document Embeddings 

**Authors**: Max Conti, Manuel Faysse, Gautier Viaud, Antoine Bosselut, Céline Hudelot, Pierre Colombo  

**Link**: [PDF](https://arxiv.org/pdf/2505.24782)  

**Abstract**: A limitation of modern document retrieval embedding methods is that they typically encode passages (chunks) from the same documents independently, often overlooking crucial contextual information from the rest of the document that could greatly improve individual chunk representations.
In this work, we introduce ConTEB (Context-aware Text Embedding Benchmark), a benchmark designed to evaluate retrieval models on their ability to leverage document-wide context. Our results show that state-of-the-art embedding models struggle in retrieval scenarios where context is required. To address this limitation, we propose InSeNT (In-sequence Negative Training), a novel contrastive post-training approach which combined with late chunking pooling enhances contextual representation learning while preserving computational efficiency. Our method significantly improves retrieval quality on ConTEB without sacrificing base model performance. We further find chunks embedded with our method are more robust to suboptimal chunking strategies and larger retrieval corpus sizes. We open-source all artifacts at this https URL. 

---
# A Novel Discrete Memristor-Coupled Heterogeneous Dual-Neuron Model and Its Application in Multi-Scenario Image Encryption 

**Authors**: Yi Zou, Mengjiao Wang, Xinan Zhang, Herbert Ho-Ching Iu  

**Link**: [PDF](https://arxiv.org/pdf/2505.24294)  

**Abstract**: Simulating brain functions using neural networks is an important area of research. Recently, discrete memristor-coupled neurons have attracted significant attention, as memristors effectively mimic synaptic behavior, which is essential for learning and memory. This highlights the biological relevance of such models. This study introduces a discrete memristive heterogeneous dual-neuron network (MHDNN). The stability of the MHDNN is analyzed with respect to initial conditions and a range of neuronal parameters. Numerical simulations demonstrate complex dynamical behaviors. Various neuronal firing patterns are investigated under different coupling strengths, and synchronization phenomena between neurons are explored. The MHDNN is implemented and validated on the STM32 hardware platform. An image encryption algorithm based on the MHDNN is proposed, along with two hardware platforms tailored for multi-scenario police image encryption. These solutions enable real-time and secure transmission of police data in complex environments, reducing hacking risks and enhancing system security. 

---
# On the Scaling of Robustness and Effectiveness in Dense Retrieval 

**Authors**: Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.24279)  

**Abstract**: Robustness and Effectiveness are critical aspects of developing dense retrieval models for real-world applications. It is known that there is a trade-off between the two. Recent work has addressed scaling laws of effectiveness in dense retrieval, revealing a power-law relationship between effectiveness and the size of models and data. Does robustness follow scaling laws too? If so, can scaling improve both robustness and effectiveness together, or do they remain locked in a trade-off?
To answer these questions, we conduct a comprehensive experimental study. We find that:(i) Robustness, including out-of-distribution and adversarial robustness, also follows a scaling law.(ii) Robustness and effectiveness exhibit different scaling patterns, leading to significant resource costs when jointly improving both. Given these findings, we shift to the third factor that affects model performance, namely the optimization strategy, beyond the model size and data size. We find that: (i) By fitting different optimization strategies, the joint performance of robustness and effectiveness traces out a Pareto frontier. (ii) When the optimization strategy strays from Pareto efficiency, the joint performance scales in a sub-optimal direction. (iii) By adjusting the optimization weights to fit the Pareto efficiency, we can achieve Pareto training, where the scaling of joint performance becomes most efficient. Even without requiring additional resources, Pareto training is comparable to the performance of scaling resources several times under optimization strategies that overly prioritize either robustness or effectiveness. Finally, we demonstrate that our findings can help deploy dense retrieval models in real-world applications that scale efficiently and are balanced for robustness and effectiveness. 

---
# Heterogeneous Graph Masked Contrastive Learning for Robust Recommendation 

**Authors**: Lei Sang, Yu Wang, Yiwen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.24172)  

**Abstract**: Heterogeneous graph neural networks (HGNNs) have demonstrated their superiority in exploiting auxiliary information for recommendation tasks. However, graphs constructed using meta-paths in HGNNs are usually too dense and contain a large number of noise edges. The propagation mechanism of HGNNs propagates even small amounts of noise in a graph to distant neighboring nodes, thereby affecting numerous node embeddings. To address this limitation, we introduce a novel model, named Masked Contrastive Learning (MCL), to enhance recommendation robustness to noise. MCL employs a random masking strategy to augment the graph via meta-paths, reducing node sensitivity to specific neighbors and bolstering embedding robustness. Furthermore, MCL employs contrastive cross-view on a Heterogeneous Information Network (HIN) from two perspectives: one-hop neighbors and meta-path neighbors. This approach acquires embeddings capturing both local and high-order structures simultaneously for recommendation. Empirical evaluations on three real-world datasets confirm the superiority of our approach over existing recommendation methods. 

---
# Transforming Podcast Preview Generation: From Expert Models to LLM-Based Systems 

**Authors**: Winstead Zhu, Ann Clifton, Azin Ghazimatin, Edgar Tanaka, Ward Ronan  

**Link**: [PDF](https://arxiv.org/pdf/2505.23908)  

**Abstract**: Discovering and evaluating long-form talk content such as videos and podcasts poses a significant challenge for users, as it requires a considerable time investment. Previews offer a practical solution by providing concise snippets that showcase key moments of the content, enabling users to make more informed and confident choices. We propose an LLM-based approach for generating podcast episode previews and deploy the solution at scale, serving hundreds of thousands of podcast previews in a real-world application. Comprehensive offline evaluations and online A/B testing demonstrate that LLM-generated previews consistently outperform a strong baseline built on top of various ML expert models, showcasing a significant reduction in the need for meticulous feature engineering. The offline results indicate notable enhancements in understandability, contextual clarity, and interest level, and the online A/B test shows a 4.6% increase in user engagement with preview content, along with a 5x boost in processing efficiency, offering a more streamlined and performant solution compared to the strong baseline of feature-engineered expert models. 

---
# SkewRoute: Training-Free LLM Routing for Knowledge Graph Retrieval-Augmented Generation via Score Skewness of Retrieved Context 

**Authors**: Hairu Wang, Yuan Feng, Yukun Cao, Xike Xie, S Kevin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.23841)  

**Abstract**: Large language models excel at many tasks but often incur high inference costs during deployment. To mitigate hallucination, many systems use a knowledge graph to enhance retrieval-augmented generation (KG-RAG). However, the large amount of retrieved knowledge contexts increase these inference costs further. A promising solution to balance performance and cost is LLM routing, which directs simple queries to smaller LLMs and complex ones to larger LLMs. However, no dedicated routing methods currently exist for RAG, and existing training-based routers face challenges scaling to this domain due to the need for extensive training data. We observe that the score distributions produced by the retrieval scorer strongly correlate with query difficulty. Based on this, we propose a novel, training-free routing framework, the first tailored to KG-RAG that effectively balances performance and cost in a plug-and-play manner. Experiments show our method reduces calls to larger LLMs by up to 50% without sacrificing response quality, demonstrating its potential for efficient and scalable LLM deployment. 

---
# Don't Reinvent the Wheel: Efficient Instruction-Following Text Embedding based on Guided Space Transformation 

**Authors**: Yingchaojie Feng, Yiqun Sun, Yandong Sun, Minfeng Zhu, Qiang Huang, Anthony K. H. Tung, Wei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.24754)  

**Abstract**: In this work, we investigate an important task named instruction-following text embedding, which generates dynamic text embeddings that adapt to user instructions, highlighting specific attributes of text. Despite recent advancements, existing approaches suffer from significant computational overhead, as they require re-encoding the entire corpus for each new instruction. To address this challenge, we propose GSTransform, a novel instruction-following text embedding framework based on Guided Space Transformation. Our key observation is that instruction-relevant information is inherently encoded in generic embeddings but remains underutilized. Instead of repeatedly encoding the corpus for each instruction, GSTransform is a lightweight transformation mechanism that adapts pre-computed embeddings in real time to align with user instructions, guided by a small amount of text data with instruction-focused label annotation. We conduct extensive experiments on three instruction-awareness downstream tasks across nine real-world datasets, demonstrating that GSTransform improves instruction-following text embedding quality over state-of-the-art methods while achieving dramatic speedups of 6~300x in real-time processing on large-scale datasets. The source code is available at this https URL. 

---
# AutoChemSchematic AI: A Closed-Loop, Physics-Aware Agentic Framework for Auto-Generating Chemical Process and Instrumentation Diagrams 

**Authors**: Sakhinana Sagar Srinivas, Shivam Gupta, Venkataramana Runkana  

**Link**: [PDF](https://arxiv.org/pdf/2505.24584)  

**Abstract**: Recent advancements in generative AI have accelerated the discovery of novel chemicals and materials; however, transitioning these discoveries to industrial-scale production remains a critical bottleneck, as it requires the development of entirely new chemical manufacturing processes. Current AI methods cannot auto-generate PFDs or PIDs, despite their critical role in scaling chemical processes, while adhering to engineering constraints. We present a closed loop, physics aware framework for the automated generation of industrially viable PFDs and PIDs. The framework integrates domain specialized small scale language models (SLMs) (trained for chemical process QA tasks) with first principles simulation, leveraging three key components: (1) a hierarchical knowledge graph of process flow and instrumentation descriptions for 1,020+ chemicals, (2) a multi-stage training pipeline that fine tunes domain specialized SLMs on synthetic datasets via Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and Retrieval-Augmented Instruction Tuning (RAIT), and (3) DWSIM based simulator in the loop validation to ensure feasibility. To improve both runtime efficiency and model compactness, the framework incorporates advanced inference time optimizations including FlashAttention, Lookahead Decoding, PagedAttention with KV-cache quantization, and Test Time Inference Scaling and independently applies structural pruning techniques (width and depth) guided by importance heuristics to reduce model size with minimal accuracy loss. Experiments demonstrate that the framework generates simulator-validated process descriptions with high fidelity, outperforms baseline methods in correctness, and generalizes to unseen chemicals. By bridging AI-driven design with industrial-scale feasibility, this work significantly reduces R&D timelines from lab discovery to plant deployment. 

---
# MGS3: A Multi-Granularity Self-Supervised Code Search Framework 

**Authors**: Rui Li, Junfeng Kang, Qi Liu, Liyang He, Zheng Zhang, Yunhao Sha, Linbo Zhu, Zhenya Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.24274)  

**Abstract**: In the pursuit of enhancing software reusability and developer productivity, code search has emerged as a key area, aimed at retrieving code snippets relevant to functionalities based on natural language queries. Despite significant progress in self-supervised code pre-training utilizing the vast amount of code data in repositories, existing methods have primarily focused on leveraging contrastive learning to align natural language with function-level code snippets. These studies have overlooked the abundance of fine-grained (such as block-level and statement-level) code snippets prevalent within the function-level code snippets, which results in suboptimal performance across all levels of granularity. To address this problem, we first construct a multi-granularity code search dataset called MGCodeSearchNet, which contains 536K+ pairs of natural language and code snippets. Subsequently, we introduce a novel Multi-Granularity Self-Supervised contrastive learning code Search framework (MGS$^{3}$}). First, MGS$^{3}$ features a Hierarchical Multi-Granularity Representation module (HMGR), which leverages syntactic structural relationships for hierarchical representation and aggregates fine-grained information into coarser-grained representations. Then, during the contrastive learning phase, we endeavor to construct positive samples of the same granularity for fine-grained code, and introduce in-function negative samples for fine-grained code. Finally, we conduct extensive experiments on code search benchmarks across various granularities, demonstrating that the framework exhibits outstanding performance in code search tasks of multiple granularities. These experiments also showcase its model-agnostic nature and compatibility with existing pre-trained code representation models. 

---
# Proactive Guidance of Multi-Turn Conversation in Industrial Search 

**Authors**: Xiaoyu Li, Xiao Li, Li Gao, Yiding Liu, Xiaoyang Wang, Shuaiqiang Wang, Junfeng Wang, Dawei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2505.24251)  

**Abstract**: The evolution of Large Language Models (LLMs) has significantly advanced multi-turn conversation systems, emphasizing the need for proactive guidance to enhance users' interactions. However, these systems face challenges in dynamically adapting to shifts in users' goals and maintaining low latency for real-time interactions. In the Baidu Search AI assistant, an industrial-scale multi-turn search system, we propose a novel two-phase framework to provide proactive guidance. The first phase, Goal-adaptive Supervised Fine-Tuning (G-SFT), employs a goal adaptation agent that dynamically adapts to user goal shifts and provides goal-relevant contextual information. G-SFT also incorporates scalable knowledge transfer to distill insights from LLMs into a lightweight model for real-time interaction. The second phase, Click-oriented Reinforcement Learning (C-RL), adopts a generate-rank paradigm, systematically constructs preference pairs from user click signals, and proactively improves click-through rates through more engaging guidance. This dual-phase architecture achieves complementary objectives: G-SFT ensures accurate goal tracking, while C-RL optimizes interaction quality through click signal-driven reinforcement learning. Extensive experiments demonstrate that our framework achieves 86.10% accuracy in offline evaluation (+23.95% over baseline) and 25.28% CTR in online deployment (149.06% relative improvement), while reducing inference latency by 69.55% through scalable knowledge distillation. 

---
# GenIC: An LLM-Based Framework for Instance Completion in Knowledge Graphs 

**Authors**: Amel Gader, Alsayed Algergawy  

**Link**: [PDF](https://arxiv.org/pdf/2505.24036)  

**Abstract**: Knowledge graph completion aims to address the gaps of knowledge bases by adding new triples that represent facts. The complexity of this task depends on how many parts of a triple are already known. Instance completion involves predicting the relation-tail pair when only the head is given (h, ?, ?). Notably, modern knowledge bases often contain entity descriptions and types, which can provide valuable context for inferring missing facts. By leveraging these textual descriptions and the ability of large language models to extract facts from them and recognize patterns within the knowledge graph schema, we propose an LLM-powered, end-to-end instance completion approach. Specifically, we introduce GenIC: a two-step Generative Instance Completion framework. The first step focuses on property prediction, treated as a multi-label classification task. The second step is link prediction, framed as a generative sequence-to-sequence task. Experimental results on three datasets show that our method outperforms existing baselines. Our code is available at this https URL. 

---
# Exploring the Landscape of Text-to-SQL with Large Language Models: Progresses, Challenges and Opportunities 

**Authors**: Yiming Huang, Jiyu Guo, Wenxin Mao, Cuiyun Gao, Peiyi Han, Chuanyi Liu, Qing Ling  

**Link**: [PDF](https://arxiv.org/pdf/2505.23838)  

**Abstract**: Converting natural language (NL) questions into SQL queries, referred to as Text-to-SQL, has emerged as a pivotal technology for facilitating access to relational databases, especially for users without SQL knowledge. Recent progress in large language models (LLMs) has markedly propelled the field of natural language processing (NLP), opening new avenues to improve text-to-SQL systems. This study presents a systematic review of LLM-based text-to-SQL, focusing on four key aspects: (1) an analysis of the research trends in LLM-based text-to-SQL; (2) an in-depth analysis of existing LLM-based text-to-SQL techniques from diverse perspectives; (3) summarization of existing text-to-SQL datasets and evaluation metrics; and (4) discussion on potential obstacles and avenues for future exploration in this domain. This survey seeks to furnish researchers with an in-depth understanding of LLM-based text-to-SQL, sparking new innovations and advancements in this field. 

---
# CoMaPOI: A Collaborative Multi-Agent Framework for Next POI Prediction Bridging the Gap Between Trajectory and Language 

**Authors**: Lin Zhong, Lingzhi Wang, Xu Yang, Qing Liao  

**Link**: [PDF](https://arxiv.org/pdf/2505.23837)  

**Abstract**: Large Language Models (LLMs) offer new opportunities for the next Point-Of-Interest (POI) prediction task, leveraging their capabilities in semantic understanding of POI trajectories. However, previous LLM-based methods, which are superficially adapted to next POI prediction, largely overlook critical challenges associated with applying LLMs to this task. Specifically, LLMs encounter two critical challenges: (1) a lack of intrinsic understanding of numeric spatiotemporal data, which hinders accurate modeling of users' spatiotemporal distributions and preferences; and (2) an excessively large and unconstrained candidate POI space, which often results in random or irrelevant predictions. To address these issues, we propose a Collaborative Multi Agent Framework for Next POI Prediction, named CoMaPOI. Through the close interaction of three specialized agents (Profiler, Forecaster, and Predictor), CoMaPOI collaboratively addresses the two critical challenges. The Profiler agent is responsible for converting numeric data into language descriptions, enhancing semantic understanding. The Forecaster agent focuses on dynamically constraining and refining the candidate POI space. The Predictor agent integrates this information to generate high-precision predictions. Extensive experiments on three benchmark datasets (NYC, TKY, and CA) demonstrate that CoMaPOI achieves state of the art performance, improving all metrics by 5% to 10% compared to SOTA baselines. This work pioneers the investigation of challenges associated with applying LLMs to complex spatiotemporal tasks by leveraging tailored collaborative agents. 

---
# LegalSearchLM: Rethinking Legal Case Retrieval as Legal Elements Generation 

**Authors**: Chaeeun Kim, Jinu Lee, Wonseok Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2505.23832)  

**Abstract**: Legal Case Retrieval (LCR), which retrieves relevant cases from a query case, is a fundamental task for legal professionals in research and decision-making. However, existing studies on LCR face two major limitations. First, they are evaluated on relatively small-scale retrieval corpora (e.g., 100-55K cases) and use a narrow range of criminal query types, which cannot sufficiently reflect the complexity of real-world legal retrieval scenarios. Second, their reliance on embedding-based or lexical matching methods often results in limited representations and legally irrelevant matches. To address these issues, we present: (1) LEGAR BENCH, the first large-scale Korean LCR benchmark, covering 411 diverse crime types in queries over 1.2M legal cases; and (2) LegalSearchLM, a retrieval model that performs legal element reasoning over the query case and directly generates content grounded in the target cases through constrained decoding. Experimental results show that LegalSearchLM outperforms baselines by 6-20% on LEGAR BENCH, achieving state-of-the-art performance. It also demonstrates strong generalization to out-of-domain cases, outperforming naive generative models trained on in-domain data by 15%. 

---
# LLM-Driven E-Commerce Marketing Content Optimization: Balancing Creativity and Conversion 

**Authors**: Haowei Yang, Haotian Lyu, Tianle Zhang, Dingzhou Wang, Yushang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.23809)  

**Abstract**: As e-commerce competition intensifies, balancing creative content with conversion effectiveness becomes critical. Leveraging LLMs' language generation capabilities, we propose a framework that integrates prompt engineering, multi-objective fine-tuning, and post-processing to generate marketing copy that is both engaging and conversion-driven. Our fine-tuning method combines sentiment adjustment, diversity enhancement, and CTA embedding. Through offline evaluations and online A/B tests across categories, our approach achieves a 12.5 % increase in CTR and an 8.3 % increase in CVR while maintaining content novelty. This provides a practical solution for automated copy generation and suggests paths for future multimodal, real-time personalization. 

---
# Conversational Exploration of Literature Landscape with LitChat 

**Authors**: Mingyu Huang, Shasha Zhou, Yuxuan Chen, Ke Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.23789)  

**Abstract**: We are living in an era of "big literature", where the volume of digital scientific publications is growing exponentially. While offering new opportunities, this also poses challenges for understanding literature landscapes, as traditional manual reviewing is no longer feasible. Recent large language models (LLMs) have shown strong capabilities for literature comprehension, yet they are incapable of offering "comprehensive, objective, open and transparent" views desired by systematic reviews due to their limited context windows and trust issues like hallucinations. Here we present LitChat, an end-to-end, interactive and conversational literature agent that augments LLM agents with data-driven discovery tools to facilitate literature exploration. LitChat automatically interprets user queries, retrieves relevant sources, constructs knowledge graphs, and employs diverse data-mining techniques to generate evidence-based insights addressing user needs. We illustrate the effectiveness of LitChat via a case study on AI4Health, highlighting its capacity to quickly navigate the users through large-scale literature landscape with data-based evidence that is otherwise infeasible with traditional means. 

---
# Learning Normal Patterns in Musical Loops 

**Authors**: Shayan Dadman, Bernt Arild Bremdal, Børre Bang, Rune Dalmo  

**Link**: [PDF](https://arxiv.org/pdf/2505.23784)  

**Abstract**: This paper introduces an unsupervised framework for detecting audio patterns in musical samples (loops) through anomaly detection techniques, addressing challenges in music information retrieval (MIR). Existing methods are often constrained by reliance on handcrafted features, domain-specific limitations, or dependence on iterative user interaction. We address these limitations through an architecture combining deep feature extraction with unsupervised anomaly detection. Our approach leverages a pre-trained Hierarchical Token-semantic Audio Transformer (HTS-AT), paired with a Feature Fusion Mechanism (FFM), to generate representations from variable-length audio loops. These embeddings are processed using one-class Deep Support Vector Data Description (Deep SVDD), which learns normative audio patterns by mapping them to a compact latent hypersphere. Evaluations on curated bass and guitar datasets compare standard and residual autoencoder variants against baselines like Isolation Forest (IF) and and principle component analysis (PCA) methods. Results show our Deep SVDD models, especially the residual autoencoder variant, deliver improved anomaly separation, particularly for larger variations. This research contributes a flexible, fully unsupervised solution for processing diverse audio samples, overcoming previous structural and input limitations while enabling effective pattern identification through distance-based latent space scoring. 

---
