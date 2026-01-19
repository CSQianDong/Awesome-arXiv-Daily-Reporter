# Isotropy-Optimized Contrastive Learning for Semantic Course Recommendation 

**Authors**: Ali Khreis, Anthony Nasr, Yusuf Hilal  

**Link**: [PDF](https://arxiv.org/pdf/2601.11427)  

**Abstract**: This paper presents a semantic course recommendation system for students using a self-supervised contrastive learning approach built upon BERT (Bidirectional Encoder Representations from Transformers). Traditional BERT embeddings suffer from anisotropic representation spaces, where course descriptions exhibit high cosine similarities regardless of semantic relevance. To address this limitation, we propose a contrastive learning framework with data augmentation and isotropy regularization that produces more discriminative embeddings. Our system processes student text queries and recommends Top-N relevant courses from a curated dataset of over 500 engineering courses across multiple faculties. Experimental results demonstrate that our fine-tuned model achieves improved embedding separation and more accurate course recommendations compared to vanilla BERT baselines. 

---
# Validating Search Query Simulations: A Taxonomy of Measures 

**Authors**: Andreas Konstantin Kruff, Nolwenn Bernard, Philipp Schaer  

**Link**: [PDF](https://arxiv.org/pdf/2601.11412)  

**Abstract**: Assessing the validity of user simulators when used for the evaluation of information retrieval systems remains an open question, constraining their effective use and the reliability of simulation-based results. To address this issue, we conduct a comprehensive literature review with a particular focus on methods for the validation of simulated user queries with regard to real queries. Based on the review, we develop a taxonomy that structures the current landscape of available measures. We empirically corroborate the taxonomy by analyzing the relationships between the different measures applied to four different datasets representing diverse search scenarios. Finally, we provide concrete recommendations on which measures or combinations of measures should be considered when validating user simulation in different contexts. Furthermore, we release a dedicated library with the most commonly used measures to facilitate future research. 

---
# From SERPs to Sound: How Search Engine Result Pages and AI-generated Podcasts Interact to Influence User Attitudes on Controversial Topics 

**Authors**: Junjie Wang, Gaole He, Alisa Rieger, Ujwal Gadiraju  

**Link**: [PDF](https://arxiv.org/pdf/2601.11282)  

**Abstract**: Compared to search engine result pages (SERPs), AI-generated podcasts represent a relatively new and relatively more passive modality of information consumption, delivering narratives in a naturally engaging format. As these two media increasingly converge in everyday information-seeking behavior, it is essential to explore how their interaction influences user attitudes, particularly in contexts involving controversial, value-laden, and often debated topics. Addressing this need, we aim to understand how information mediums of present-day SERPs and AI-generated podcasts interact to shape the opinions of users. To this end, through a controlled user study (N=483), we investigated user attitudinal effects of consuming information via SERPs and AI-generated podcasts, focusing on how the sequence and modality of exposure shape user opinions. A majority of users in our study corresponded to attitude change outcomes, and we found an effect of sequence on attitude change. Our results further revealed a role of viewpoint bias and the degree of topic controversiality in shaping attitude change, although we found no effect of individual moderators. 

---
# Rank4Gen: RAG-Preference-Aligned Document Set Selection and Ranking 

**Authors**: Yongqi Fan, Yuxiang Chu, Zhentao Xia, Xiaoyang Chen, Jie Liu, Haijin Liang, Jin Ma, Ben He, Yingfei Sun, Dezhi Ye, Tong Ruan  

**Link**: [PDF](https://arxiv.org/pdf/2601.11273)  

**Abstract**: In the RAG paradigm, the information retrieval module provides context for generators by retrieving and ranking multiple documents to support the aggregation of evidence. However, existing ranking models are primarily optimized for query--document relevance, which often misaligns with generators' preferences for evidence selection and citation, limiting their impact on response quality. Moreover, most approaches do not account for preference differences across generators, resulting in unstable cross-generator performance. We propose \textbf{Rank4Gen}, a generator-aware ranker for RAG that targets the goal of \emph{Ranking for Generators}. Rank4Gen introduces two key preference modeling strategies: (1) \textbf{From Ranking Relevance to Response Quality}, which optimizes ranking with respect to downstream response quality rather than query--document relevance; and (2) \textbf{Generator-Specific Preference Modeling}, which conditions a single ranker on different generators to capture their distinct ranking preferences. To enable such modeling, we construct \textbf{PRISM}, a dataset built from multiple open-source corpora and diverse downstream generators. Experiments on five challenging and recent RAG benchmarks demonstrate that RRank4Gen achieves strong and competitive performance for complex evidence composition in RAG. 

---
# LLM-Assisted Pseudo-Relevance Feedback 

**Authors**: David Otero, Javier Parapar  

**Link**: [PDF](https://arxiv.org/pdf/2601.11238)  

**Abstract**: Query expansion is a long-standing technique to mitigate vocabulary mismatch in ad hoc Information Retrieval. Pseudo-relevance feedback methods, such as RM3, estimate an expanded query model from the top-ranked documents, but remain vulnerable to topic drift when early results include noisy or tangential content. Recent approaches instead prompt Large Language Models to generate synthetic expansions or query variants. While effective, these methods risk hallucinations and misalignment with collection-specific terminology. We propose a hybrid alternative that preserves the robustness and interpretability of classical PRF while leveraging LLM semantic judgement. Our method inserts an LLM-based filtering stage prior to RM3 estimation: the LLM judges the documents in the initial top-$k$ ranking, and RM3 is computed only over those accepted as relevant. This simple intervention improves over blind PRF and a strong baseline across several datasets and metrics. 

---
# From Knots to Knobs: Towards Steerable Collaborative Filtering Using Sparse Autoencoders 

**Authors**: Martin Spišák, Ladislav Peška, Petr Škoda, Vojtěch Vančura, Rodrigo Alves  

**Link**: [PDF](https://arxiv.org/pdf/2601.11182)  

**Abstract**: Sparse autoencoders (SAEs) have recently emerged as pivotal tools for introspection into large language models. SAEs can uncover high-quality, interpretable features at different levels of granularity and enable targeted steering of the generation process by selectively activating specific neurons in their latent activations. Our paper is the first to apply this approach to collaborative filtering, aiming to extract similarly interpretable features from representations learned purely from interaction signals. In particular, we focus on a widely adopted class of collaborative autoencoders (CFAEs) and augment them by inserting an SAE between their encoder and decoder networks. We demonstrate that such representation is largely monosemantic and propose suitable mapping functions between semantic concepts and individual neurons. We also evaluate a simple yet effective method that utilizes this representation to steer the recommendations in a desired direction. 

---
# Cross-Modal Attention Network with Dual Graph Learning in Multimodal Recommendation 

**Authors**: Ji Dai, Quan Fang, Jun Hu, Desheng Cai, Yang Yang, Can Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2601.11151)  

**Abstract**: Multimedia recommendation systems leverage user-item interactions and multimodal information to capture user preferences, enabling more accurate and personalized recommendations. Despite notable advancements, existing approaches still face two critical limitations: first, shallow modality fusion often relies on simple concatenation, failing to exploit rich synergic intra- and inter-modal relationships; second, asymmetric feature treatment-where users are only characterized by interaction IDs while items benefit from rich multimodal content-hinders the learning of a shared semantic space. To address these issues, we propose a Cross-modal Recursive Attention Network with dual graph Embedding (CRANE). To tackle shallow fusion, we design a core Recursive Cross-Modal Attention (RCA) mechanism that iteratively refines modality features based on cross-correlations in a joint latent space, effectively capturing high-order intra- and inter-modal dependencies. For symmetric multimodal learning, we explicitly construct users' multimodal profiles by aggregating features of their interacted items. Furthermore, CRANE integrates a symmetric dual-graph framework-comprising a heterogeneous user-item interaction graph and a homogeneous item-item semantic graph-unified by a self-supervised contrastive learning objective to fuse behavioral and semantic signals. Despite these complex modeling capabilities, CRANE maintains high computational efficiency. Theoretical and empirical analyses confirm its scalability and high practical efficiency, achieving faster convergence on small datasets and superior performance ceilings on large-scale ones. Comprehensive experiments on four public real-world datasets validate an average 5% improvement in key metrics over state-of-the-art baselines. 

---
# Deep GraphRAG: A Balanced Approach to Hierarchical Retrieval and Adaptive Integration 

**Authors**: Yuejie Li, Ke Yang, Tao Wang, Bolin Chen, Bowen Li, Chengjun Mao  

**Link**: [PDF](https://arxiv.org/pdf/2601.11144)  

**Abstract**: Graph-based Retrieval-Augmented Generation (GraphRAG) frameworks face a trade-off between the comprehensiveness of global search and the efficiency of local search. Existing methods are often challenged by navigating large-scale hierarchical graphs, optimizing retrieval paths, and balancing exploration-exploitation dynamics, frequently lacking robust multi-stage re-ranking. To overcome these deficits, we propose Deep GraphRAG, a framework designed for a balanced approach to hierarchical retrieval and adaptive integration. It introduces a hierarchical global-to-local retrieval strategy that integrates macroscopic inter-community and microscopic intra-community contextual relations. This strategy employs a three-stage process: (1) inter-community filtering, which prunes the search space using local context; (2) community-level refinement, which prioritizes relevant subgraphs via entity-interaction analysis; and (3) entity-level fine-grained search within target communities. A beam search-optimized dynamic re-ranking module guides this process, continuously filtering candidates to balance efficiency and global comprehensiveness. Deep GraphRAG also features a Knowledge Integration Module leveraging a compact LLM, trained with Dynamic Weighting Reward GRPO (DW-GRPO). This novel reinforcement learning approach dynamically adjusts reward weights to balance three key objectives: relevance, faithfulness, and conciseness. This training enables compact models (1.5B) to approach the performance of large models (70B) in the integration task. Evaluations on Natural Questions and HotpotQA demonstrate that Deep GraphRAG significantly outperforms baseline graph retrieval methods in both accuracy and efficiency. 

---
# Learn Before Represent: Bridging Generative and Contrastive Learning for Domain-Specific LLM Embeddings 

**Authors**: Xiaoyu Liang, Yuchen Peng, Jiale Luo, Wenhao Wang, Haoji Hu, Xincheng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2601.11124)  

**Abstract**: Large Language Models (LLMs) adapted via contrastive learning excel in general representation learning but struggle in vertical domains like chemistry and law, primarily due to a lack of domain-specific knowledge. This work identifies a core bottleneck: the prevailing ``LLM+CL'' paradigm focuses on semantic alignment but cannot perform knowledge acquisition, leading to failures on specialized terminology. To bridge this gap, we propose Learn Before Represent (LBR), a novel two-stage framework. LBR first injects domain knowledge via an Information Bottleneck-Constrained Generative Learning stage, preserving the LLM's causal attention to maximize knowledge acquisition while compressing semantics. It then performs Generative-Refined Contrastive Learning on the compressed representations for alignment. This approach maintains architectural consistency and resolves the objective conflict between generative and contrastive learning. Extensive experiments on medical, chemistry, and code retrieval tasks show that LBR significantly outperforms strong baselines. Our work establishes a new paradigm for building accurate and robust representations in vertical domains. 

---
# PruneRAG: Confidence-Guided Query Decomposition Trees for Efficient Retrieval-Augmented Generation 

**Authors**: Shuguang Jiao, Xinyu Xiao, Yunfan Wei, Shuhan Qi, Chengkai Huang, Quan Z. Michael Sheng, Lina Yao  

**Link**: [PDF](https://arxiv.org/pdf/2601.11024)  

**Abstract**: Retrieval-augmented generation (RAG) has become a powerful framework for enhancing large language models in knowledge-intensive and reasoning tasks. However, as reasoning chains deepen or search trees expand, RAG systems often face two persistent failures: evidence forgetting, where retrieved knowledge is not effectively used, and inefficiency, caused by uncontrolled query expansions and redundant retrieval. These issues reveal a critical gap between retrieval and evidence utilization in current RAG architectures. We propose PruneRAG, a confidence-guided query decomposition framework that builds a structured query decomposition tree to perform stable and efficient reasoning. PruneRAG introduces three key mechanisms: adaptive node expansion that regulates tree width and depth, confidence-guided decisions that accept reliable answers and prune uncertain branches, and fine-grained retrieval that extracts entity-level anchors to improve retrieval precision. Together, these components preserve salient evidence throughout multi-hop reasoning while significantly reducing retrieval overhead. To better analyze evidence misuse, we define the Evidence Forgetting Rate as a metric to quantify cases where golden evidence is retrieved but not correctly used. Extensive experiments across various multi-hop QA benchmarks show that PruneRAG achieves superior accuracy and efficiency over state-of-the-art baselines. 

---
# PRISM: Personalized Recommendation via Information Synergy Module 

**Authors**: Xinyi Zhang, Yutong Li, Peijie Sun, Letian Sha, Zhongxuan Han  

**Link**: [PDF](https://arxiv.org/pdf/2601.10944)  

**Abstract**: Multimodal sequential recommendation (MSR) leverages diverse item modalities to improve recommendation accuracy, while achieving effective and adaptive fusion remains challenging. Existing MSR models often overlook synergistic information that emerges only through modality combinations. Moreover, they typically assume a fixed importance for different modality interactions across users. To address these limitations, we propose \textbf{P}ersonalized \textbf{R}ecommend-ation via \textbf{I}nformation \textbf{S}ynergy \textbf{M}odule (PRISM), a plug-and-play framework for sequential recommendation (SR). PRISM explicitly decomposes multimodal information into unique, redundant, and synergistic components through an Interaction Expert Layer and dynamically weights them via an Adaptive Fusion Layer guided by user preferences. This information-theoretic design enables fine-grained disentanglement and personalized fusion of multimodal signals. Extensive experiments on four datasets and three SR backbones demonstrate its effectiveness and versatility. The code is available at this https URL. 

---
# Can Instructed Retrieval Models Really Support Exploration? 

**Authors**: Piyush Maheshwari, Sheshera Mysore, Hamed Zamani  

**Link**: [PDF](https://arxiv.org/pdf/2601.10936)  

**Abstract**: Exploratory searches are characterized by under-specified goals and evolving query intents. In such scenarios, retrieval models that can capture user-specified nuances in query intent and adapt results accordingly are desirable -- instruction-following retrieval models promise such a capability. In this work, we evaluate instructed retrievers for the prevalent yet under-explored application of aspect-conditional seed-guided exploration using an expert-annotated test collection. We evaluate both recent LLMs fine-tuned for instructed retrieval and general-purpose LLMs prompted for ranking with the highly performant Pairwise Ranking Prompting. We find that the best instructed retrievers improve on ranking relevance compared to instruction-agnostic approaches. However, we also find that instruction following performance, crucial to the user experience of interacting with models, does not mirror ranking relevance improvements and displays insensitivity or counter-intuitive behavior to instructions. Our results indicate that while users may benefit from using current instructed retrievers over instruction-agnostic models, they may not benefit from using them for long-running exploratory sessions requiring greater sensitivity to instructions. 

---
# Tail-Aware Data Augmentation for Long-Tail Sequential Recommendation 

**Authors**: Yizhou Dang, Zhifu Wei, Minhan Huang, Lianbo Ma, Jianzhe Zhao, Guibing Guo, Xingwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.10933)  

**Abstract**: Sequential recommendation (SR) learns user preferences based on their historical interaction sequences and provides personalized suggestions. In real-world scenarios, most users can only interact with a handful of items, while the majority of items are seldom consumed. This pervasive long-tail challenge limits the model's ability to learn user preferences. Despite previous efforts to enrich tail items/users with knowledge from head parts or improve tail learning through additional contextual information, they still face the following issues: 1) They struggle to improve the situation where interactions of tail users/items are scarce, leading to incomplete preferences learning for the tail parts. 2) Existing methods often degrade overall or head parts performance when improving accuracy for tail users/items, thereby harming the user experience. We propose Tail-Aware Data Augmentation (TADA) for long-tail sequential recommendation, which enhances the interaction frequency for tail items/users while maintaining head performance, thereby promoting the model's learning capabilities for the tail. Specifically, we first capture the co-occurrence and correlation among low-popularity items by a linear model. Building upon this, we design two tail-aware augmentation operators, T-Substitute and T-Insert. The former replaces the head item with a relevant item, while the latter utilizes co-occurrence relationships to extend the original sequence by incorporating both head and tail items. The augmented and original sequences are mixed at the representation level to preserve preference knowledge. We further extend the mix operation across different tail-user sequences and augmented sequences to generate richer augmented samples, thereby improving tail performance. Comprehensive experiments demonstrate the superiority of our method. The codes are provided at this https URL. 

---
# Interactive Narrative Analytics: Bridging Computational Narrative Extraction and Human Sensemaking 

**Authors**: Brian Keith  

**Link**: [PDF](https://arxiv.org/pdf/2601.11459)  

**Abstract**: Information overload and misinformation create significant challenges in extracting meaningful narratives from large news collections. This paper defines the nascent field of Interactive Narrative Analytics (INA), which combines computational narrative extraction with interactive visual analytics to support sensemaking. INA approaches enable the interactive exploration of narrative structures through computational methods and visual interfaces that facilitate human interpretation. The field faces challenges in scalability, interactivity, knowledge integration, and evaluation standardization, yet offers promising opportunities across news analysis, intelligence, scientific literature exploration, and social media analysis. Through the combination of computational and human insight, INA addresses complex challenges in narrative sensemaking. 

---
# Seek and You Shall Find: Design & Evaluation of a Context-Aware Interactive Search Companion 

**Authors**: Markus Bink, Marten Risius, Udo Kruschwitz, David Elsweiler  

**Link**: [PDF](https://arxiv.org/pdf/2601.11287)  

**Abstract**: Many users struggle with effective online search and critical evaluation, especially in high-stakes domains like health, while often overestimating their digital literacy. Thus, in this demo, we present an interactive search companion that seamlessly integrates expert search strategies into existing search engine result pages. Providing context-aware tips on clarifying information needs, improving query formulation, encouraging result exploration, and mitigating biases, our companion aims to foster reflective search behaviour while minimising cognitive burden. A user study demonstrates the companion's successful encouragement of more active and exploratory search, leading users to submit 75 % more queries and view roughly twice as many results, as well as performance gains in difficult tasks. This demo illustrates how lightweight, contextual guidance can enhance search literacy and empower users through micro-learning opportunities. While the vision involves real-time LLM adaptivity, this study utilises a controlled implementation to test the underlying intervention strategies. 

---
# "Can You Tell Me?": Designing Copilots to Support Human Judgement in Online Information Seeking 

**Authors**: Markus Bink, Marten Risius, Udo Kruschwitz, David Elsweiler  

**Link**: [PDF](https://arxiv.org/pdf/2601.11284)  

**Abstract**: Generative AI (GenAI) tools are transforming information seeking, but their fluent, authoritative responses risk overreliance and discourage independent verification and reasoning. Rather than replacing the cognitive work of users, GenAI systems should be designed to support and scaffold it. Therefore, this paper introduces an LLM-based conversational copilot designed to scaffold information evaluation rather than provide answers and foster digital literacy skills. In a pre-registered, randomised controlled trial (N=261) examining three interface conditions including a chat-based copilot, our mixed-methods analysis reveals that users engaged deeply with the copilot, demonstrating metacognitive reflection. However, the copilot did not significantly improve answer correctness or search engagement, largely due to a "time-on-chat vs. exploration" trade-off and users' bias toward positive information. Qualitative findings reveal tension between the copilot's Socratic approach and users' desire for efficiency. These results highlight both the promise and pitfalls of pedagogical copilots, and we outline design pathways to reconcile literacy goals with efficiency demands. 

---
# Scalable Music Cover Retrieval Using Lyrics-Aligned Audio Embeddings 

**Authors**: Joanne Affolter, Benjamin Martin, Elena V. Epure, Gabriel Meseguer-Brocal, Frédéric Kaplan  

**Link**: [PDF](https://arxiv.org/pdf/2601.11262)  

**Abstract**: Music Cover Retrieval, also known as Version Identification, aims to recognize distinct renditions of the same underlying musical work, a task central to catalog management, copyright enforcement, and music retrieval. State-of-the-art approaches have largely focused on harmonic and melodic features, employing increasingly complex audio pipelines designed to be invariant to musical attributes that often vary widely across covers. While effective, these methods demand substantial training time and computational resources. By contrast, lyrics constitute a strong invariant across covers, though their use has been limited by the difficulty of extracting them accurately and efficiently from polyphonic audio. Early methods relied on simple frameworks that limited downstream performance, while more recent systems deliver stronger results but require large models integrated within complex multimodal architectures. We introduce LIVI (Lyrics-Informed Version Identification), an approach that seeks to balance retrieval accuracy with computational efficiency. First, LIVI leverages supervision from state-of-the-art transcription and text embedding models during training to achieve retrieval accuracy on par with--or superior to--harmonic-based systems. Second, LIVI remains lightweight and efficient by removing the transcription step at inference, challenging the dominance of complexity-heavy pipelines. 

---
# The Big Ban Theory: A Pre- and Post-Intervention Dataset of Online Content Moderation Actions 

**Authors**: Aldo Cerulli, Lorenzo Cima, Benedetta Tessa, Serena Tardelli, Stefano Cresci  

**Link**: [PDF](https://arxiv.org/pdf/2601.11128)  

**Abstract**: Online platforms rely on moderation interventions to curb harmful behavior such hate speech, toxicity, and the spread of mis- and disinformation. Yet research on the effects and possible biases of such interventions faces multiple limitations. For example, existing works frequently focus on single or a few interventions, due to the absence of comprehensive datasets. As a result, researchers must typically collect the necessary data for each new study, which limits opportunities for systematic comparisons. To overcome these challenges, we introduce The Big Ban Theory (TBBT), a large dataset of moderation interventions. TBBT covers 25 interventions of varying type, severity, and scope, comprising in total over 339K users and nearly 39M posted messages. For each intervention, we provide standardized metadata and pseudonymized user activity collected three months before and after its enforcement, enabling consistent and comparable analyses of intervention effects. In addition, we provide a descriptive exploratory analysis of the dataset, along with several use cases of how it can support research on content moderation. With this dataset, we aim to support researchers studying the effects of moderation interventions and to promote more systematic, reproducible, and comparable research. TBBT is publicly available at: this https URL. 

---
# Streaming Stochastic Submodular Maximization with On-Demand User Requests 

**Authors**: Honglian Wang, Sijing Tu, Lutz Oettershagen, Aristides Gionis  

**Link**: [PDF](https://arxiv.org/pdf/2601.10901)  

**Abstract**: We explore a novel problem in streaming submodular maximization, inspired by the dynamics of news-recommendation platforms. We consider a setting where users can visit a news website at any time, and upon each visit, the website must display up to $k$ news items. User interactions are inherently stochastic: each news item presented to the user is consumed with a certain acceptance probability by the user, and each news item covers certain topics. Our goal is to design a streaming algorithm that maximizes the expected total topic coverage. To address this problem, we establish a connection to submodular maximization subject to a matroid constraint. We show that we can effectively adapt previous methods to address our problem when the number of user visits is known in advance or linear-size memory in the stream length is available. However, in more realistic scenarios where only an upper bound on the visits and sublinear memory is available, the algorithms fail to guarantee any bounded performance. To overcome these limitations, we introduce a new online streaming algorithm that achieves a competitive ratio of $1/(8\delta)$, where $\delta$ controls the approximation quality. Moreover, it requires only a single pass over the stream, and uses memory independent of the stream length. Empirically, our algorithms consistently outperform the baselines. 

---
# EncodeRec: An Embedding Backbone for Recommendation Systems 

**Authors**: Guy Hadad, Neomi Rabaev, Bracha Shapira  

**Link**: [PDF](https://arxiv.org/pdf/2601.10837)  

**Abstract**: Recent recommender systems increasingly leverage embeddings from large pre-trained language models (PLMs). However, such embeddings exhibit two key limitations: (1) PLMs are not explicitly optimized to produce structured and discriminative embedding spaces, and (2) their representations remain overly generic, often failing to capture the domain-specific semantics crucial for recommendation tasks. We present EncodeRec, an approach designed to align textual representations with recommendation objectives while learning compact, informative embeddings directly from item descriptions. EncodeRec keeps the language model parameters frozen during recommender system training, making it computationally efficient without sacrificing semantic fidelity. Experiments across core recommendation benchmarks demonstrate its effectiveness both as a backbone for sequential recommendation models and for semantic ID tokenization, showing substantial gains over PLM-based and embedding model baselines. These results underscore the pivotal role of embedding adaptation in bridging the gap between general-purpose language models and practical recommender systems. 

---
# Japanese AI Agent System on Human Papillomavirus Vaccination: System Design 

**Authors**: Junyu Liu, Siwen Yang, Dexiu Ma, Qian Niu, Zequn Zhang, Momoko Nagai-Tanima, Tomoki Aoyama  

**Link**: [PDF](https://arxiv.org/pdf/2601.10718)  

**Abstract**: Human papillomavirus (HPV) vaccine hesitancy poses significant public health challenges, particularly in Japan where proactive vaccination recommendations were suspended from 2013 to 2021. The resulting information gap is exacerbated by misinformation on social media, and traditional ways cannot simultaneously address individual queries while monitoring population-level discourse. This study aimed to develop a dual-purpose AI agent system that provides verified HPV vaccine information through a conversational interface while generating analytical reports for medical institutions based on user interactions and social media. We implemented a system comprising: a vector database integrating academic papers, government sources, news media, and social media; a Retrieval-Augmented Generation chatbot using ReAct agent architecture with multi-tool orchestration across five knowledge sources; and an automated report generation system with modules for news analysis, research synthesis, social media sentiment analysis, and user interaction pattern identification. Performance was assessed using a 0-5 scoring scale. For single-turn evaluation, the chatbot achieved mean scores of 4.83 for relevance, 4.89 for routing, 4.50 for reference quality, 4.90 for correctness, and 4.88 for professional identity (overall 4.80). Multi-turn evaluation yielded higher scores: context retention 4.94, topic coherence 5.00, and overall 4.98. The report generation system achieved completeness 4.00-5.00, correctness 4.00-5.00, and helpfulness 3.67-5.00, with reference validity 5.00 across all periods. This study demonstrates the feasibility of an integrated AI agent system for bidirectional HPV vaccine communication. The architecture enables verified information delivery with source attribution while providing systematic public discourse analysis, with a transferable framework for adaptation to other medical contexts. 

---
