# Muse-it: A Tool for Analyzing Music Discourse on Reddit 

**Authors**: Jatin Agarwala, George Paul, Nemani Harsha Vardhan, Vinoo Alluri  

**Link**: [PDF](https://arxiv.org/pdf/2509.20228)  

**Abstract**: Music engagement spans diverse interactions with music, from selection and emotional response to its impact on behavior, identity, and social connections. Social media platforms provide spaces where such engagement can be observed in natural, unprompted conversations. Advances in natural language processing (NLP) and big data analytics make it possible to analyze these discussions at scale, extending music research to broader contexts. Reddit, in particular, offers anonymity that encourages diverse participation and yields rich discourse on music in ecological settings. Yet the scale of this data requires tools to extract, process, and analyze it effectively. We present Muse-it, a platform that retrieves comprehensive Reddit data centered on user-defined queries. It aggregates posts from across subreddits, supports topic modeling, temporal trend analysis, and clustering, and enables efficient study of large-scale discourse. Muse-it also identifies music-related hyperlinks (e.g., Spotify), retrieves track-level metadata such as artist, album, release date, genre, popularity, and lyrics, and links these to the discussions. An interactive interface provides dynamic visualizations of the collected data. Muse-it thus offers an accessible way for music researchers to gather and analyze big data, opening new avenues for understanding music engagement as it naturally unfolds online. 

---
# Multimodal Representation-disentangled Information Bottleneck for Multimodal Recommendation 

**Authors**: Hui Wang, Jinghui Qin, Wushao Wen, Qingling Li, Shanshan Zhong, Zhongzhan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20225)  

**Abstract**: Multimodal data has significantly advanced recommendation systems by integrating diverse information sources to model user preferences and item characteristics. However, these systems often struggle with redundant and irrelevant information, which can degrade performance. Most existing methods either fuse multimodal information directly or use rigid architectural separation for disentanglement, failing to adequately filter noise and model the complex interplay between modalities. To address these challenges, we propose a novel framework, the Multimodal Representation-disentangled Information Bottleneck (MRdIB). Concretely, we first employ a Multimodal Information Bottleneck to compress the input representations, effectively filtering out task-irrelevant noise while preserving rich semantic information. Then, we decompose the information based on its relationship with the recommendation target into unique, redundant, and synergistic components. We achieve this decomposition with a series of constraints: a unique information learning objective to preserve modality-unique signals, a redundant information learning objective to minimize overlap, and a synergistic information learning objective to capture emergent information. By optimizing these objectives, MRdIB guides a model to learn more powerful and disentangled representations. Extensive experiments on several competitive models and three benchmark datasets demonstrate the effectiveness and versatility of our MRdIB in enhancing multimodal recommendation. 

---
# Intelligent Algorithm Selection for Recommender Systems: Meta-Learning via in-depth algorithm feature engineering 

**Authors**: Jarne Mathi Decker  

**Link**: [PDF](https://arxiv.org/pdf/2509.20134)  

**Abstract**: The "No Free Lunch" theorem dictates that no single recommender algorithm is optimal for all users, creating a significant Algorithm Selection Problem. Standard meta-learning approaches aim to solve this by selecting an algorithm based on user features, but treat the fundamentally diverse algorithms themselves as equivalent, "black-box" choices. This thesis investigates the impact of overcoming this limitation by engineering a comprehensive feature set to explicitly characterize the algorithms themselves. We combine static code metrics, Abstract Syntax Tree properties, behavioral performance landmarks, and high-level conceptual features. We evaluate two meta-learners across five datasets: a baseline using only user features and our proposed model using both user and algorithm features. Our results show that the meta-learner augmented with algorithm features achieves an average NDCG@10 of 0.143, a statistically significant improvement of 11.7% over the Single Best Algorithm baseline (0.128). However, we found that the inclusion of algorithm features did not lead to an improvement in overall NDCG@10 over the meta learner using only user features (0.144). While adding algorithm features to the meta-learner did improve its Top-1 selection accuracy (+16.1%), this was counterbalanced by leading to a lower Top-3 accuracy (-10.7%). We conclude that for the per-user algorithm selection task in recommender systems, the predictive power of user features is overwhelmingly dominant. While algorithm features improve selection precision, unlocking their potential to boost overall performance remains a non-trivial challenge. 

---
# Cascade! Human in the loop shortcomings can increase the risk of failures in recommender systems 

**Authors**: Wm. Matthew Kennedy, Nishanshi Shukla, Cigdem Patlak, Blake Chambers, Theodora Skeadas, Tuesday, Kingsley Owadara, Aayush Dhanotiya  

**Link**: [PDF](https://arxiv.org/pdf/2509.20099)  

**Abstract**: Recommender systems are among the most commonly deployed systems today. Systems design approaches to AI-powered recommender systems have done well to urge recommender system developers to follow more intentional data collection, curation, and management procedures. So too has the "human-in-the-loop" paradigm been widely adopted, primarily to address the issue of accountability. However, in this paper, we take the position that human oversight in recommender system design also entails novel risks that have yet to be fully described. These risks are "codetermined" by the information context in which such systems are often deployed. Furthermore, new knowledge of the shortcomings of "human-in-the-loop" practices to deliver meaningful oversight of other AI systems suggest that they may also be inadequate for achieving socially responsible recommendations. We review how the limitations of human oversight may increase the chances of a specific kind of failure: a "cascade" or "compound" failure. We then briefly explore how the unique dynamics of three common deployment contexts can make humans in the loop more likely to fail in their oversight duties. We then conclude with two recommendations. 

---
# Multimodal-enhanced Federated Recommendation: A Group-wise Fusion Approach 

**Authors**: Chunxu Zhang, Weipeng Zhang, Guodong Long, Zhiheng Xue, Riting Xia, Bo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19955)  

**Abstract**: Federated Recommendation (FR) is a new learning paradigm to tackle the learn-to-rank problem in a privacy-preservation manner. How to integrate multi-modality features into federated recommendation is still an open challenge in terms of efficiency, distribution heterogeneity, and fine-grained alignment. To address these challenges, we propose a novel multimodal fusion mechanism in federated recommendation settings (GFMFR). Specifically, it offloads multimodal representation learning to the server, which stores item content and employs a high-capacity encoder to generate expressive representations, alleviating client-side overhead. Moreover, a group-aware item representation fusion approach enables fine-grained knowledge sharing among similar users while retaining individual preferences. The proposed fusion loss could be simply plugged into any existing federated recommender systems empowering their capability by adding multi-modality features. Extensive experiments on five public benchmark datasets demonstrate that GFMFR consistently outperforms state-of-the-art multimodal FR baselines. 

---
# Documentation Retrieval Improves Planning Language Generation 

**Authors**: Renxiang Wang, Li Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19931)  

**Abstract**: Certain strong LLMs have shown promise for zero-shot formal planning by generating planning languages like PDDL. Yet, performance of most open-source models under 50B parameters has been reported to be close to zero due to the low-resource nature of these languages. We significantly improve their performance via a series of lightweight pipelines that integrates documentation retrieval with modular code generation and error refinement. With models like Llama-4-Maverick, our best pipeline improves plan correctness from 0\% to over 80\% on the common BlocksWorld domain. However, while syntactic errors are substantially reduced, semantic errors persist in more challenging domains, revealing fundamental limitations in current models' reasoning capabilities.\footnote{Our code and data can be found at this https URL 

---
# Adaptive User Interest Modeling via Conditioned Denoising Diffusion For Click-Through Rate Prediction 

**Authors**: Qihang Zhao, Xiaoyang Zheng, Ben Chen, Zhongbo Sun, Chenyi Lei  

**Link**: [PDF](https://arxiv.org/pdf/2509.19876)  

**Abstract**: User behavior sequences in search systems resemble "interest fossils", capturing genuine intent yet eroded by exposure bias, category drift, and contextual noise. Current methods predominantly follow an "identify-aggregate" paradigm, assuming sequences immutably reflect user preferences while overlooking the organic entanglement of noise and genuine interest. Moreover, they output static, context-agnostic representations, failing to adapt to dynamic intent shifts under varying Query-User-Item-Context conditions.
To resolve this dual challenge, we propose the Contextual Diffusion Purifier (CDP). By treating category-filtered behaviors as "contaminated observations", CDP employs a forward noising and conditional reverse denoising process guided by cross-interaction features (Query x User x Item x Context), controllably generating pure, context-aware interest representations that dynamically evolve with scenarios. Extensive offline/online experiments demonstrate the superiority of CDP over state-of-the-art methods. 

---
# FusedANN: Convexified Hybrid ANN via Attribute-Vector Fusion 

**Authors**: Alireza Heidari, Wei Zhang, Ying Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2509.19767)  

**Abstract**: Vector search powers transformers technology, but real-world use demands hybrid queries that combine vector similarity with attribute filters (e.g., "top document in category X, from 2023"). Current solutions trade off recall, speed, and flexibility, relying on fragile index hacks that don't scale. We introduce FusedANN (Fused Attribute-Vector Nearest Neighbor), a geometric framework that elevates filtering to ANN optimization constraints and introduces a convex fused space via a Lagrangian-like relaxation. Our method jointly embeds attributes and vectors through transformer-based convexification, turning hard filters into continuous, weighted penalties that preserve top-k semantics while enabling efficient approximate search. We prove that FusedANN reduces to exact filtering under high selectivity, gracefully relaxes to semantically nearest attributes when exact matches are insufficient, and preserves downstream ANN alpha-approximation guarantees. Empirically, FusedANN improves query throughput by eliminating brittle filtering stages, achieving superior recall-latency tradeoffs on standard hybrid benchmarks without specialized index hacks, delivering up to 3 times higher throughput and better recall than state-of-the-art hybrid and graph-based systems. Theoretically, we provide explicit error bounds and parameter selection rules that make FusedANN practical for production. This establishes a principled, scalable, and verifiable bridge between symbolic constraints and vector similarity, unlocking a new generation of filtered retrieval systems for large, hybrid, and dynamic NLP/ML workloads. 

---
# Learning Contextual Retrieval for Robust Conversational Search 

**Authors**: Seunghan Yang, Juntae Lee, Jihwan Bang, Kyuhong Shim, Minsoo Kim, Simyung Chang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19700)  

**Abstract**: Effective conversational search demands a deep understanding of user intent across multiple dialogue turns. Users frequently use abbreviations and shift topics in the middle of conversations, posing challenges for conventional retrievers. While query rewriting techniques improve clarity, they often incur significant computational cost due to additional autoregressive steps. Moreover, although LLM-based retrievers demonstrate strong performance, they are not explicitly optimized to track user intent in multi-turn settings, often failing under topic drift or contextual ambiguity. To address these limitations, we propose ContextualRetriever, a novel LLM-based retriever that directly incorporates conversational context into the retrieval process. Our approach introduces: (1) a context-aware embedding mechanism that highlights the current query within the dialogue history; (2) intent-guided supervision based on high-quality rewritten queries; and (3) a training strategy that preserves the generative capabilities of the base LLM. Extensive evaluations across multiple conversational search benchmarks demonstrate that ContextualRetriever significantly outperforms existing methods while incurring no additional inference overhead. 

---
# AIRwaves at CheckThat! 2025: Retrieving Scientific Sources for Implicit Claims on Social Media with Dual Encoders and Neural Re-Ranking 

**Authors**: Cem Ashbaugh, Leon Baumgärtner, Tim Gress, Nikita Sidorov, Daniel Werner  

**Link**: [PDF](https://arxiv.org/pdf/2509.19509)  

**Abstract**: Linking implicit scientific claims made on social media to their original publications is crucial for evidence-based fact-checking and scholarly discourse, yet it is hindered by lexical sparsity, very short queries, and domain-specific language. Team AIRwaves ranked second in Subtask 4b of the CLEF-2025 CheckThat! Lab with an evidence-retrieval approach that markedly outperforms the competition baseline. The optimized sparse-retrieval baseline(BM25) achieves MRR@5 = 0.5025 on the gold label blind test set. To surpass this baseline, a two-stage retrieval pipeline is introduced: (i) a first stage that uses a dual encoder based on E5-large, fine-tuned using in-batch and mined hard negatives and enhanced through chunked tokenization and rich document metadata; and (ii) a neural re-ranking stage using a SciBERT cross-encoder. Replacing purely lexical matching with neural representations lifts performance to MRR@5 = 0.6174, and the complete pipeline further improves to MRR@5 = 0.6828. The findings demonstrate that coupling dense retrieval with neural re-rankers delivers a powerful and efficient solution for tweet-to-study matching and provides a practical blueprint for future evidence-retrieval pipelines. 

---
# Into the Void: Understanding Online Health Information in Low-Web Data Languages 

**Authors**: Hellina Hailu Nigatu, Nuredin Ali Abdelkadir, Fiker Tewelde, Stevie Chancellor, Daricia Wilkinson  

**Link**: [PDF](https://arxiv.org/pdf/2509.20245)  

**Abstract**: Data voids--areas of the internet where reliable information is scarce or absent--pose significant challenges to online health information seeking, particularly for users operating in low-web data languages. These voids are increasingly encountered not on traditional search engines alone, but on social media platforms, which have gradually morphed into informal search engines for millions of people. In this paper, we introduce the phenomenon of data horizons: a critical boundary where algorithmic structures begin to degrade the relevance and reliability of search results. Unlike the core of a data void, which is often exploited by bad actors to spread misinformation, the data horizon marks the critical space where systemic factors, such as linguistic underrepresentation, algorithmic amplification, and socio-cultural mismatch, create conditions of informational instability. Focusing on Tigrinya and Amharic as languages of study, we evaluate (1) the common characteristics of search results for health queries, (2) the quality and credibility of health information, and (3) characteristics of search results that diverge from their queries. We find that search results for health queries in low-web data languages may not always be in the language of search and may be dominated by nutritional and religious advice. We show that search results that diverge from their queries in low-resourced languages are due to algorithmic failures, (un)intentional manipulation, or active manipulation by content creators. We use our findings to illustrate how a data horizon manifests under several interacting constraints on information availability. 

---
# Digital Signal Processing from Classical Coherent Systems to Continuous-Variable QKD: A Review of Cross-Domain Techniques, Applications, and Challenges 

**Authors**: Davi Juvêncio Gomes de Sousa, Caroline da Silva Morais Alves, Valéria Loureiro da Silva, Nelson Alves Ferreira Neto  

**Link**: [PDF](https://arxiv.org/pdf/2509.20141)  

**Abstract**: This systematic review investigates the application of digital signal processing (DSP) techniques -- originally developed for coherent optical communication systems to continuous-variable quantum key distribution (CV-QKD). The convergence of these domains has enabled significant advances in CV-QKD performance, particularly in phase synchronization, polarization tracking, and excess noise mitigation. To provide a comprehensive and reproducible synthesis of this emerging field, we employed the APISSER methodology, a task-oriented framework adapted from the PRISMA protocol. A structured search across IEEE Xplore and Web of Science databases (2021-2025) yielded 220 relevant publications, which were screened, classified, and analyzed to address six research questions. Our findings highlight that many classical DSP algorithms, such as Kalman filtering, carrier recovery, adaptive equalization, and machine-learning-assisted signal estimation, have been successfully adapted to the quantum regime, often requiring modifications to meet security and noise constraints. We also identify a range of recent DSP innovations in coherent optical communication systems with high potential for future CV-QKD integration, including neural equalization, probabilistic shaping, and joint retiming-equalization filters. Despite these advances, challenges remain in achieving robust phase tracking under ultra-low Signal-to-Noise Ratio (SNR) conditions, real-time polarization compensation, and secure co-existence with classical channels. This review maps current trends, technical barriers, and emerging opportunities at the intersection of signal processing for quantum and classical communication, supporting the development of scalable and resilient CV-QKD systems. 

---
# HiCoLoRA: Addressing Context-Prompt Misalignment via Hierarchical Collaborative LoRA for Zero-Shot DST 

**Authors**: Shuyu Zhang, Yifan Wei, Xinru Wang, Yanmin Zhu, Yangfan He, Yixuan Weng, Bin Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.19742)  

**Abstract**: Zero-shot Dialog State Tracking (zs-DST) is essential for enabling Task-Oriented Dialog Systems (TODs) to generalize to new domains without costly data annotation. A central challenge lies in the semantic misalignment between dynamic dialog contexts and static prompts, leading to inflexible cross-layer coordination, domain interference, and catastrophic forgetting. To tackle this, we propose Hierarchical Collaborative Low-Rank Adaptation (HiCoLoRA), a framework that enhances zero-shot slot inference through robust prompt alignment. It features a hierarchical LoRA architecture for dynamic layer-specific processing (combining lower-layer heuristic grouping and higher-layer full interaction), integrates Spectral Joint Domain-Slot Clustering to identify transferable associations (feeding an Adaptive Linear Fusion Mechanism), and employs Semantic-Enhanced SVD Initialization (SemSVD-Init) to preserve pre-trained knowledge. Experiments on multi-domain datasets MultiWOZ and SGD show that HiCoLoRA outperforms baselines, achieving SOTA in zs-DST. Code is available at this https URL. 

---
# DyBBT: Dynamic Balance via Bandit inspired Targeting for Dialog Policy with Cognitive Dual-Systems 

**Authors**: Shuyu Zhang, Yifan Wei, Jialuo Yuan, Xinru Wang, Yanmin Zhu, Bin Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.19695)  

**Abstract**: Task oriented dialog systems often rely on static exploration strategies that do not adapt to dynamic dialog contexts, leading to inefficient exploration and suboptimal performance. We propose DyBBT, a novel dialog policy learning framework that formalizes the exploration challenge through a structured cognitive state space capturing dialog progression, user uncertainty, and slot dependency. DyBBT proposes a bandit inspired meta-controller that dynamically switches between a fast intuitive inference (System 1) and a slow deliberative reasoner (System 2) based on real-time cognitive states and visitation counts. Extensive experiments on single- and multi-domain benchmarks show that DyBBT achieves state-of-the-art performance in success rate, efficiency, and generalization, with human evaluations confirming its decisions are well aligned with expert judgment. Code is available at this https URL. 

---
