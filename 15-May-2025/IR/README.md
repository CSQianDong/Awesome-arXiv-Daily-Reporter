# Distance-aware Self-adaptive Graph Convolution for Fine-grained Hierarchical Recommendation 

**Authors**: Tao Huang, Yihong Chen, Wei Fan, Wei Zhou, Junhao Wen  

**Link**: [PDF](https://arxiv.org/pdf/2505.09590)  

**Abstract**: Graph Convolutional Networks (GCNs) are widely used to improve recommendation accuracy and performance by effectively learning the representations of user and item nodes. However, two major challenges remain: (1) the lack of further optimization in the graph representation structure and (2) insufficient attention given to the varying contributions of different convolutional this http URL paper proposes SAGCN, a distance-based adaptive hierarchical aggregation method that refines the aggregation process through differentiated representation metrics. SAGCN introduces a detailed approach to multilayer information aggregation and representation space optimization, enabling the model to learn hierarchical embedding weights based on the distance between hierarchical representations. This innovation allows for more precise cross-layer information aggregation, improves the model's ability to capture hierarchical embeddings, and optimizes the representation space structure. Additionally, the objective loss function is refined to better align with recommendation this http URL experiments conducted on four real-world datasets demonstrate significant improvements, including over a 5% increase on Yelp and a 5.58% increase in Recall@10 on the ML_1M dataset. 

---
# GlobalMood: A cross-cultural benchmark for music emotion recognition 

**Authors**: Harin Lee, Elif Çelen, Peter Harrison, Manuel Anglada-Tort, Pol van Rijn, Minsu Park, Marc Schönwiesner, Nori Jacoby  

**Link**: [PDF](https://arxiv.org/pdf/2505.09539)  

**Abstract**: Human annotations of mood in music are essential for music generation and recommender systems. However, existing datasets predominantly focus on Western songs with mood terms derived from English, which may limit generalizability across diverse linguistic and cultural backgrounds. To address this, we introduce `GlobalMood', a novel cross-cultural benchmark dataset comprising 1,180 songs sampled from 59 countries, with large-scale annotations collected from 2,519 individuals across five culturally and linguistically distinct locations: U.S., France, Mexico, S. Korea, and Egypt. Rather than imposing predefined mood categories, we implement a bottom-up, participant-driven approach to organically elicit culturally specific music-related mood terms. We then recruit another pool of human participants to collect 988,925 ratings for these culture-specific descriptors. Our analysis confirms the presence of a valence-arousal structure shared across cultures, yet also reveals significant divergences in how certain mood terms, despite being dictionary equivalents, are perceived cross-culturally. State-of-the-art multimodal models benefit substantially from fine-tuning on our cross-culturally balanced dataset, as evidenced by improved alignment with human evaluations - particularly in non-English contexts. More broadly, our findings inform the ongoing debate on the universality versus cultural specificity of emotional descriptors, and our methodology can contribute to other multimodal and cross-lingual research. 

---
# FACTors: A New Dataset for Studying the Fact-checking Ecosystem 

**Authors**: Enes Altuncu, Can Başkent, Sanjay Bhattacherjee, Shujun Li, Dwaipayan Roy  

**Link**: [PDF](https://arxiv.org/pdf/2505.09414)  

**Abstract**: Our fight against false information is spearheaded by fact-checkers. They investigate the veracity of claims and document their findings as fact-checking reports. With the rapid increase in the amount of false information circulating online, the use of automation in fact-checking processes aims to strengthen this ecosystem by enhancing scalability. Datasets containing fact-checked claims play a key role in developing such automated solutions. However, to the best of our knowledge, there is no fact-checking dataset at the ecosystem level, covering claims from a sufficiently long period of time and sourced from a wide range of actors reflecting the entire ecosystem that admittedly follows widely-accepted codes and principles of fact-checking. We present a new dataset FACTors, the first to fill this gap by presenting ecosystem-level data on fact-checking. It contains 118,112 claims from 117,993 fact-checking reports in English (co-)authored by 1,953 individuals and published during the period of 1995-2025 by 39 fact-checking organisations that are active signatories of the IFCN (International Fact-Checking Network) and/or EFCSN (European Fact-Checking Standards Network). It contains 7,327 overlapping claims investigated by multiple fact-checking organisations, corresponding to 2,977 unique claims. It allows to conduct new ecosystem-level studies of the fact-checkers (organisations and individuals). To demonstrate the usefulness of FACTors, we present three example applications, including a first-of-its-kind statistical analysis of the fact-checking ecosystem, examining the political inclinations of the fact-checking organisations, and attempting to assign a credibility score to each organisation based on the findings of the statistical analysis and political leanings. Our methods for constructing FACTors are generic and can be used to maintain a live dataset that can be updated dynamically. 

---
# Diffusion Recommender Models and the Illusion of Progress: A Concerning Study of Reproducibility and a Conceptual Mismatch 

**Authors**: Michael Benigni, Maurizio Ferrari Dacrema, Dietmar Jannach  

**Link**: [PDF](https://arxiv.org/pdf/2505.09364)  

**Abstract**: Countless new machine learning models are published every year and are reported to significantly advance the state-of-the-art in \emph{top-n} recommendation. However, earlier reproducibility studies indicate that progress in this area may be quite limited. Specifically, various widespread methodological issues, e.g., comparisons with untuned baseline models, have led to an \emph{illusion of progress}. In this work, our goal is to examine whether these problems persist in today's research. To this end, we aim to reproduce the latest advancements reported from applying modern Denoising Diffusion Probabilistic Models to recommender systems, focusing on four models published at the top-ranked SIGIR conference in 2023 and 2024. Our findings are concerning, revealing persistent methodological problems. Alarmingly, through experiments, we find that the latest recommendation techniques based on diffusion models, despite their computational complexity and substantial carbon footprint, are consistently outperformed by simpler existing models. Furthermore, we identify key mismatches between the characteristics of diffusion models and those of the traditional \emph{top-n} recommendation task, raising doubts about their suitability for recommendation. We also note that, in the papers we analyze, the generative capabilities of these models are constrained to a minimum. Overall, our results and continued methodological issues call for greater scientific rigor and a disruptive change in the research and publication culture in this area. 

---
# Focus, Merge, Rank: Improved Question Answering Based on Semi-structured Knowledge Bases 

**Authors**: Derian Boer, Stephen Roth, Stefan Kramer  

**Link**: [PDF](https://arxiv.org/pdf/2505.09246)  

**Abstract**: In many real-world settings, machine learning models and interactive systems have access to both structured knowledge, e.g., knowledge graphs or tables, and unstructured content, e.g., natural language documents. However, most rely on either. Semi-Structured Knowledge Bases (SKBs) bridge this gap by linking unstructured content to nodes within structured data, thereby enabling new strategies for knowledge access and use. In this work, we present FocusedRetriever, a modular SKB-based framework for multi-hop question answering. It integrates components (VSS-based entity search, LLM-based generation of Cypher queries and pairwise re-ranking) in a way that enables it to outperform state-of-the-art methods across all three STaRK benchmark test sets, covering diverse domains and multiple performance metrics. The average first-hit rate exceeds that of the second-best method by 25.7%. FocusedRetriever leverages (1) the capacity of Large Language Models (LLMs) to extract relational facts and entity attributes from unstructured text, (2) node set joins to filter answer candidates based on these extracted triplets and constraints, (3) vector similarity search to retrieve and rank relevant unstructured content, and (4) the contextual capabilities of LLMs to finally rank the top-k answers. For generality, we only incorporate base LLMs in FocusedRetriever in our evaluation. However, our analysis of intermediate results highlights several opportunities for further upgrades including finetuning. The source code is publicly available at this https URL . 

---
# HMamba: Hyperbolic Mamba for Sequential Recommendation 

**Authors**: Qianru Zhang, Honggang Wen, Wei Yuan, Crystal Chen, Menglin Yang, Siu-Ming Yiu, Hongzhi Yin  

**Link**: [PDF](https://arxiv.org/pdf/2505.09205)  

**Abstract**: Sequential recommendation systems have become a cornerstone of personalized services, adept at modeling the temporal evolution of user preferences by capturing dynamic interaction sequences. Existing approaches predominantly rely on traditional models, including RNNs and Transformers. Despite their success in local pattern recognition, Transformer-based methods suffer from quadratic computational complexity and a tendency toward superficial attention patterns, limiting their ability to infer enduring preference hierarchies in sequential recommendation data. Recent advances in Mamba-based sequential models introduce linear-time efficiency but remain constrained by Euclidean geometry, failing to leverage the intrinsic hyperbolic structure of recommendation data. To bridge this gap, we propose Hyperbolic Mamba, a novel architecture that unifies the efficiency of Mamba's selective state space mechanism with hyperbolic geometry's hierarchical representational power. Our framework introduces (1) a hyperbolic selective state space that maintains curvature-aware sequence modeling and (2) stabilized Riemannian operations to enable scalable training. Experiments across four benchmarks demonstrate that Hyperbolic Mamba achieves 3-11% improvement while retaining Mamba's linear-time efficiency, enabling real-world deployment. This work establishes a new paradigm for efficient, hierarchy-aware sequential modeling. 

---
# Item Level Exploration Traffic Allocation in Large-scale Recommendation Systems 

**Authors**: Dong Wang, Junyi Jiao, Arnab Bhadury, Yaping Zhang, Mingyan Gao  

**Link**: [PDF](https://arxiv.org/pdf/2505.09033)  

**Abstract**: This paper contributes to addressing the item cold start problem in large-scale recommender systems, focusing on how to efficiently gain initial visibility for newly ingested content. We propose an exploration system designed to efficiently allocate impressions to these fresh items. Our approach leverages a learned probabilistic model to predict an item's discoverability, which then informs a scalable and adaptive traffic allocation strategy. This system intelligently distributes exploration budgets, optimizing for the long-term benefit of the recommendation platform. The impact is a demonstrably more efficient cold-start process, leading to a significant increase in the discoverability of new content and ultimately enriching the item corpus available for exploitation, as evidenced by its successful deployment in a large-scale production environment. 

---
# CXMArena: Unified Dataset to benchmark performance in realistic CXM Scenarios 

**Authors**: Raghav Garg, Kapil Sharma, Karan Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2505.09436)  

**Abstract**: Large Language Models (LLMs) hold immense potential for revolutionizing Customer Experience Management (CXM), particularly in contact center operations. However, evaluating their practical utility in complex operational environments is hindered by data scarcity (due to privacy concerns) and the limitations of current benchmarks. Existing benchmarks often lack realism, failing to incorporate deep knowledge base (KB) integration, real-world noise, or critical operational tasks beyond conversational fluency. To bridge this gap, we introduce CXMArena, a novel, large-scale synthetic benchmark dataset specifically designed for evaluating AI in operational CXM contexts. Given the diversity in possible contact center features, we have developed a scalable LLM-powered pipeline that simulates the brand's CXM entities that form the foundation of our datasets-such as knowledge articles including product specifications, issue taxonomies, and contact center conversations. The entities closely represent real-world distribution because of controlled noise injection (informed by domain experts) and rigorous automated validation. Building on this, we release CXMArena, which provides dedicated benchmarks targeting five important operational tasks: Knowledge Base Refinement, Intent Prediction, Agent Quality Adherence, Article Search, and Multi-turn RAG with Integrated Tools. Our baseline experiments underscore the benchmark's difficulty: even state of the art embedding and generation models achieve only 68% accuracy on article search, while standard embedding methods yield a low F1 score of 0.3 for knowledge base refinement, highlighting significant challenges for current models necessitating complex pipelines and solutions over conventional techniques. 

---
# Scent of Knowledge: Optimizing Search-Enhanced Reasoning with Information Foraging 

**Authors**: Hongjin Qian, Zheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.09316)  

**Abstract**: Augmenting large language models (LLMs) with external retrieval has become a standard method to address their inherent knowledge cutoff limitations. However, traditional retrieval-augmented generation methods employ static, pre-inference retrieval strategies, making them inadequate for complex tasks involving ambiguous, multi-step, or evolving information needs. Recent advances in test-time scaling techniques have demonstrated significant potential in enabling LLMs to dynamically interact with external tools, motivating the shift toward adaptive inference-time retrieval. Inspired by Information Foraging Theory (IFT), we propose InForage, a reinforcement learning framework that formalizes retrieval-augmented reasoning as a dynamic information-seeking process. Unlike existing approaches, InForage explicitly rewards intermediate retrieval quality, encouraging LLMs to iteratively gather and integrate information through adaptive search behaviors. To facilitate training, we construct a human-guided dataset capturing iterative search and reasoning trajectories for complex, real-world web tasks. Extensive evaluations across general question answering, multi-hop reasoning tasks, and a newly developed real-time web QA dataset demonstrate InForage's superior performance over baseline methods. These results highlight InForage's effectiveness in building robust, adaptive, and efficient reasoning agents. 

---
# Display Content, Display Methods and Evaluation Methods of the HCI in Explainable Recommender Systems: A Survey 

**Authors**: Weiqing Li, Yue Xu, Yuefeng Li, Yinghui Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.09065)  

**Abstract**: Explainable Recommender Systems (XRS) aim to provide users with understandable reasons for the recommendations generated by these systems, representing a crucial research direction in artificial intelligence (AI). Recent research has increasingly focused on the algorithms, display, and evaluation methodologies of XRS. While current research and reviews primarily emphasize the algorithmic aspects, with fewer studies addressing the Human-Computer Interaction (HCI) layer of XRS. Additionally, existing reviews lack a unified taxonomy for XRS and there is insufficient attention given to the emerging area of short video recommendations. In this study, we synthesize existing literature and surveys on XRS, presenting a unified framework for its research and development. The main contributions are as follows: 1) We adopt a lifecycle perspective to systematically summarize the technologies and methods used in XRS, addressing challenges posed by the diversity and complexity of algorithmic models and explanation techniques. 2) For the first time, we highlight the application of multimedia, particularly video-based explanations, along with its potential, technical pathways, and challenges in XRS. 3) We provide a structured overview of evaluation methods from both qualitative and quantitative dimensions. These findings provide valuable insights for the systematic design, progress, and testing of XRS. 

---
