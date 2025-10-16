# HyMiRec: A Hybrid Multi-interest Learning Framework for LLM-based Sequential Recommendation 

**Authors**: Jingyi Zhou, Cheng Chen, Kai Zuo, Manjie Xu, Zhendong Fu, Yibo Chen, Xu Tang, Yao Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.13738)  

**Abstract**: Large language models (LLMs) have recently demonstrated strong potential for sequential recommendation. However, current LLM-based approaches face critical limitations in modeling users' long-term and diverse interests. First, due to inference latency and feature fetching bandwidth constraints, existing methods typically truncate user behavior sequences to include only the most recent interactions, resulting in the loss of valuable long-range preference signals. Second, most current methods rely on next-item prediction with a single predicted embedding, overlooking the multifaceted nature of user interests and limiting recommendation diversity. To address these challenges, we propose HyMiRec, a hybrid multi-interest sequential recommendation framework, which leverages a lightweight recommender to extracts coarse interest embeddings from long user sequences and an LLM-based recommender to captures refined interest embeddings. To alleviate the overhead of fetching features, we introduce a residual codebook based on cosine similarity, enabling efficient compression and reuse of user history embeddings. To model the diverse preferences of users, we design a disentangled multi-interest learning module, which leverages multiple interest queries to learn disentangles multiple interest signals adaptively, allowing the model to capture different facets of user intent. Extensive experiments are conducted on both benchmark datasets and a collected industrial dataset, demonstrating our effectiveness over existing state-of-the-art methods. Furthermore, online A/B testing shows that HyMiRec brings consistent improvements in real-world recommendation systems. 

---
# RAG Meets Temporal Graphs: Time-Sensitive Modeling and Retrieval for Evolving Knowledge 

**Authors**: Jiale Han, Austin Cheung, Yubai Wei, Zheng Yu, Xusheng Wang, Bing Zhu, Yi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.13590)  

**Abstract**: Knowledge is inherently time-sensitive and continuously evolves over time. Although current Retrieval-Augmented Generation (RAG) systems enrich LLMs with external knowledge, they largely ignore this temporal nature. This raises two challenges for RAG. First, current RAG methods lack effective time-aware representations. Same facts of different time are difficult to distinguish with vector embeddings or conventional knowledge graphs. Second, most RAG evaluations assume a static corpus, leaving a blind spot regarding update costs and retrieval stability as knowledge evolves. To make RAG time-aware, we propose Temporal GraphRAG (TG-RAG), which models external corpora as a bi-level temporal graph consisting of a temporal knowledge graph with timestamped relations and a hierarchical time graph. Multi-granularity temporal summaries are generated for each time node to capture both key events and broader trends at that time. The design supports incremental updates by extracting new temporal facts from the incoming corpus and merging them into the existing graph. The temporal graph explicitly represents identical facts at different times as distinct edges to avoid ambiguity, and the time hierarchy graph allows only generating reports for new leaf time nodes and their ancestors, ensuring effective and efficient updates. During inference, TG-RAG dynamically retrieves a subgraph within the temporal and semantic scope of the query, enabling precise evidence gathering. Moreover, we introduce ECT-QA, a time-sensitive question-answering dataset featuring both specific and abstract queries, along with a comprehensive evaluation protocol designed to assess incremental update capabilities of RAG systems. Extensive experiments show that TG-RAG significantly outperforms existing baselines, demonstrating the effectiveness of our method in handling temporal knowledge and incremental updates. 

---
# MADREC: A Multi-Aspect Driven LLM Agent for Explainable and Adaptive Recommendation 

**Authors**: Jiin Park, Misuk Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.13371)  

**Abstract**: Recent attempts to integrate large language models (LLMs) into recommender systems have gained momentum, but most remain limited to simple text generation or static prompt-based inference, failing to capture the complexity of user preferences and real-world interactions. This study proposes the Multi-Aspect Driven LLM Agent MADRec, an autonomous LLM-based recommender that constructs user and item profiles by unsupervised extraction of multi-aspect information from reviews and performs direct recommendation, sequential recommendation, and explanation generation. MADRec generates structured profiles via aspect-category-based summarization and applies Re-Ranking to construct high-density inputs. When the ground-truth item is missing from the output, the Self-Feedback mechanism dynamically adjusts the inference criteria. Experiments across multiple domains show that MADRec outperforms traditional and LLM-based baselines in both precision and explainability, with human evaluation further confirming the persuasiveness of the generated explanations. 

---
# Improving Visual Recommendation on E-commerce Platforms Using Vision-Language Models 

**Authors**: Yuki Yada, Sho Akiyama, Ryo Watanabe, Yuta Ueno, Yusuke Shido, Andre Rusli  

**Link**: [PDF](https://arxiv.org/pdf/2510.13359)  

**Abstract**: On large-scale e-commerce platforms with tens of millions of active monthly users, recommending visually similar products is essential for enabling users to efficiently discover items that align with their preferences. This study presents the application of a vision-language model (VLM) -- which has demonstrated strong performance in image recognition and image-text retrieval tasks -- to product recommendations on Mercari, a major consumer-to-consumer marketplace used by more than 20 million monthly users in Japan. Specifically, we fine-tuned SigLIP, a VLM employing a sigmoid-based contrastive loss, using one million product image-title pairs from Mercari collected over a three-month period, and developed an image encoder for generating item embeddings used in the recommendation system. Our evaluation comprised an offline analysis of historical interaction logs and an online A/B test in a production environment. In offline analysis, the model achieved a 9.1% improvement in nDCG@5 compared with the baseline. In the online A/B test, the click-through rate improved by 50% whereas the conversion rate improved by 14% compared with the existing model. These results demonstrate the effectiveness of VLM-based encoders for e-commerce product recommendations and provide practical insights into the development of visual similarity-based recommendation systems. 

---
# Beyond Static LLM Policies: Imitation-Enhanced Reinforcement Learning for Recommendation 

**Authors**: Yi Zhang, Lili Xie, Ruihong Qiu, Jiajun Liu, Sen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.13229)  

**Abstract**: Recommender systems (RecSys) have become critical tools for enhancing user engagement by delivering personalized content across diverse digital platforms. Recent advancements in large language models (LLMs) demonstrate significant potential for improving RecSys, primarily due to their exceptional generalization capabilities and sophisticated contextual understanding, which facilitate the generation of flexible and interpretable recommendations. However, the direct deployment of LLMs as primary recommendation policies presents notable challenges, including persistent latency issues stemming from frequent API calls and inherent model limitations such as hallucinations and biases. To address these issues, this paper proposes a novel offline reinforcement learning (RL) framework that leverages imitation learning from LLM-generated trajectories. Specifically, inverse reinforcement learning is employed to extract robust reward models from LLM demonstrations. This approach negates the need for LLM fine-tuning, thereby substantially reducing computational overhead. Simultaneously, the RL policy is guided by the cumulative rewards derived from these demonstrations, effectively transferring the semantic insights captured by the LLM. Comprehensive experiments conducted on two benchmark datasets validate the effectiveness of the proposed method, demonstrating superior performance when compared against state-of-the-art RL-based and in-context learning baselines. The code can be found at this https URL. 

---
# LLM-guided Hierarchical Retrieval 

**Authors**: Nilesh Gupta, Wei-Cheng Chang, Ngot Bui, Cho-Jui Hsieh, Inderjit S. Dhillon  

**Link**: [PDF](https://arxiv.org/pdf/2510.13217)  

**Abstract**: Modern IR systems are increasingly tasked with answering complex, multi-faceted queries that require deep reasoning rather than simple keyword or semantic matching. While LLM-based IR has shown great promise, the prevailing retrieve-then-rerank paradigm inherits the limitations of embedding-based retrieval; parametric generative approaches are difficult to update with new information; and long-context methods that place the entire corpus in context are computationally infeasible for large document collections. To address these challenges, we introduce LATTICE, a hierarchical retrieval framework that enables an LLM to reason over and navigate large corpora with logarithmic search complexity by imposing a semantic tree structure on the corpus. Our approach consists of two stages: (1) an offline phase that organizes the corpus into a semantic hierarchy via either a bottom-up agglomerative strategy or a top-down divisive strategy using multi-level summaries and (2) an online traversal phase where a search LLM navigates this tree. A central challenge in such LLM-guided search is that the model's relevance judgments are noisy, context-dependent, and unaware of the hierarchy, making cross-branch and cross-level comparisons difficult. To overcome this, we propose a traversal algorithm that estimates calibrated latent relevance scores from local LLM outputs and aggregates them into a global path relevance metric. Our training-free framework achieves state-of-the-art zero-shot performance on the reasoning-intensive BRIGHT benchmark, demonstrating up to 9% improvement in Recall@100 and 5% in nDCG@10 over the next best zero-shot baseline. Furthermore, compared to the fine-tuned SOTA method DIVER-v2, LATTICE attains comparable results on BRIGHT subsets that use a static corpus for evaluation. 

---
# ReMindRAG: Low-Cost LLM-Guided Knowledge Graph Traversal for Efficient RAG 

**Authors**: Yikuan Hu, Jifeng Zhu, Lanrui Tang, Chen Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.13193)  

**Abstract**: Knowledge graphs (KGs), with their structured representation capabilities, offer promising avenue for enhancing Retrieval Augmented Generation (RAG) systems, leading to the development of KG-RAG systems. Nevertheless, existing methods often struggle to achieve effective synergy between system effectiveness and cost efficiency, leading to neither unsatisfying performance nor excessive LLM prompt tokens and inference time. To this end, this paper proposes REMINDRAG, which employs an LLM-guided graph traversal featuring node exploration, node exploitation, and, most notably, memory replay, to improve both system effectiveness and cost efficiency. Specifically, REMINDRAG memorizes traversal experience within KG edge embeddings, mirroring the way LLMs "memorize" world knowledge within their parameters, but in a train-free manner. We theoretically and experimentally confirm the effectiveness of REMINDRAG, demonstrating its superiority over existing baselines across various benchmark datasets and LLM backbones. Our code is available at this https URL. 

---
# Retrieval-in-the-Chain: Bootstrapping Large Language Models for Generative Retrieval 

**Authors**: Yingchen zhang, Ruqing zhang, Jiafeng Guo, Wenjun Peng, Sen Li, Fuyu Lv  

**Link**: [PDF](https://arxiv.org/pdf/2510.13095)  

**Abstract**: Generative retrieval (GR) is an emerging paradigm that leverages large language models (LLMs) to autoregressively generate document identifiers (docids) relevant to a given query. Prior works have focused on leveraging the generative capabilities of LLMs to improve GR, while overlooking that their reasoning capabilities could likewise help. This raises a key question: Can explicit reasoning benefit GR? To investigate, we first conduct a preliminary study where an LLM is prompted to generate free-form chain-of-thought (CoT) reasoning before performing constrained docid decoding. Although this method outperforms standard GR, the generated reasoning tends to be verbose and poorly aligned with the docid space. These limitations motivate the development of a reasoning mechanism better tailored to GR.
Therefore, we propose Reason-for-Retrieval (R4R), a reasoning-augmented framework for GR that converts free-form CoT reasoning into a compact, structured format, and iteratively refines the reasoning during the retrieval process. R4R augments an existing GR method by leveraging a reasoning-capable LLM that has been instruction-tuned for GR. At inference time, R4R first uses the LLM to generate an initial structured reasoning; then the same LLM alternates between (i) constrained decoding with the chosen GR method to produce candidate docids and (ii) updating the reasoning based on retrieval results to improve the next round. R4R does not require additional models or training, and instead a single LLM serves as both the reasoning generator and the retriever. Extensive experiments on Natural Questions, MS MARCO, and a real-world item-search benchmark validate the effectiveness of R4R. 

---
# Post-hoc Popularity Bias Correction in GNN-based Collaborative Filtering 

**Authors**: Md Aminul Islam, Elena Zheleva, Ren Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.12959)  

**Abstract**: User historical interaction data is the primary signal for learning user preferences in collaborative filtering (CF). However, the training data often exhibits a long-tailed distribution, where only a few items have the majority of interactions. CF models trained directly on such imbalanced data are prone to learning popularity bias, which reduces personalization and leads to suboptimal recommendation quality. Graph Neural Networks (GNNs), while effective for CF due to their message passing mechanism, can further propagate and amplify popularity bias through their aggregation process. Existing approaches typically address popularity bias by modifying training objectives but fail to directly counteract the bias propagated during GNN's neighborhood aggregation. Applying weights to interactions during aggregation can help alleviate this problem, yet it risks distorting model learning due to unstable node representations in the early stages of training. In this paper, we propose a Post-hoc Popularity Debiasing (PPD) method that corrects for popularity bias in GNN-based CF and operates directly on pre-trained embeddings without requiring retraining. By estimating interaction-level popularity and removing popularity components from node representations via a popularity direction vector, PPD reduces bias while preserving user preferences. Experimental results show that our method outperforms state-of-the-art approaches for popularity bias correction in GNN-based CF. 

---
# Maximum In-Support Return Modeling for Dynamic Recommendation with Language Model Prior 

**Authors**: Xiaocong Chen, Siyu Wang, Lina Yao  

**Link**: [PDF](https://arxiv.org/pdf/2510.12816)  

**Abstract**: Reinforcement Learning-based recommender systems (RLRS) offer an effective way to handle sequential recommendation tasks but often face difficulties in real-world settings, where user feedback data can be sub-optimal or sparse. In this paper, we introduce MDT4Rec, an offline RLRS framework that builds on the Decision Transformer (DT) to address two major challenges: learning from sub-optimal histories and representing complex user-item interactions. First, MDT4Rec shifts the trajectory stitching procedure from the training phase to action inference, allowing the system to shorten its historical context when necessary and thereby ignore negative or unsuccessful past experiences. Second, MDT4Rec initializes DT with a pre-trained large language model (LLM) for knowledge transfer, replaces linear embedding layers with Multi-Layer Perceptrons (MLPs) for more flexible representations, and employs Low-Rank Adaptation (LoRA) to efficiently fine-tune only a small subset of parameters. We evaluate MDT4Rec on five public datasets and in an online simulation environment, demonstrating that it outperforms existing methods. 

---
# Energy-Guided Diffusion Sampling for Long-Term User Behavior Prediction in Reinforcement Learning-based Recommendation 

**Authors**: Xiaocong Chen, Siyu Wang, Lina Yao  

**Link**: [PDF](https://arxiv.org/pdf/2510.12815)  

**Abstract**: Reinforcement learning-based recommender systems (RL4RS) have gained attention for their ability to adapt to dynamic user preferences. However, these systems face challenges, particularly in offline settings, where data inefficiency and reliance on pre-collected trajectories limit their broader applicability. While offline reinforcement learning methods leverage extensive datasets to address these issues, they often struggle with noisy data and fail to capture long-term user preferences, resulting in suboptimal recommendation policies. To overcome these limitations, we propose Diffusion-enhanced Actor-Critic for Offline RL4RS (DAC4Rec), a novel framework that integrates diffusion processes with reinforcement learning to model complex user preferences more effectively. DAC4Rec leverages the denoising capabilities of diffusion models to enhance the robustness of offline RL algorithms and incorporates a Q-value-guided policy optimization strategy to better handle suboptimal trajectories. Additionally, we introduce an energy-based sampling strategy to reduce randomness during recommendation generation, ensuring more targeted and reliable outcomes. We validate the effectiveness of DAC4Rec through extensive experiments on six real-world offline datasets and in an online simulation environment, demonstrating its ability to optimize long-term user preferences. Furthermore, we show that the proposed diffusion policy can be seamlessly integrated into other commonly used RL algorithms in RL4RS, highlighting its versatility and wide applicability. 

---
# ChatR1: Reinforcement Learning for Conversational Reasoning and Retrieval Augmented Question Answering 

**Authors**: Simon Lupart, Mohammad Aliannejadi, Evangelos Kanoulas  

**Link**: [PDF](https://arxiv.org/pdf/2510.13312)  

**Abstract**: We present ChatR1, a reasoning framework based on reinforcement learning (RL) for conversational question answering (CQA). Reasoning plays an important role in CQA, where user intent evolves across dialogue turns, and utterances are often underspecified, requiring contextual interpretation, query reformulation, and dynamic coordination between retrieval and generation. Unlike static `rewrite, retrieve, and generate' pipelines, ChatR1 interleaves search and reasoning across turns, enabling exploratory and adaptive behaviors learned through RL. To address the challenge of sparse and delayed rewards in RL, we propose an intent-aware reward that provides turn-level feedback by aligning retrieval and reasoning with evolving user goals. Our proposed ChatR1 demonstrates strong performance on both 3B and 7B model backbones, outperforming competitive models on five CQA datasets, measured by different metrics (F1, BERTScore, and LLM-as-judge). We include a diverse set of CQA datasets to cover topic shifts, evolving intents, mixed-initiative dialogues, and multi-document grounding, testing ChatR1's performance from various aspects. Ablation studies confirm the effectiveness of the intent-aware reward. Our analyses further reveal diverse reasoning trajectories and effective use of the search tool. ChatR1 also generalizes robustly across domains, demonstrating that RL-based reasoning enables more flexible and context-sensitive behavior than static CQA pipelines. 

---
# Epistemic-aware Vision-Language Foundation Model for Fetal Ultrasound Interpretation 

**Authors**: Xiao He, Huangxuan Zhao, Guojia Wan, Wei Zhou, Yanxing Liu, Juhua Liu, Yongchao Xu, Yong Luo, Dacheng Tao, Bo Du  

**Link**: [PDF](https://arxiv.org/pdf/2510.12953)  

**Abstract**: Recent medical vision-language models have shown promise on tasks such as VQA, report generation, and anomaly detection. However, most are adapted to structured adult imaging and underperform in fetal ultrasound, which poses challenges of multi-view image reasoning, numerous diseases, and image diversity. To bridge this gap, we introduce FetalMind, a medical AI system tailored to fetal ultrasound for both report generation and diagnosis. Guided by clinical workflow, we propose Salient Epistemic Disentanglement (SED), which injects an expert-curated bipartite graph into the model to decouple view-disease associations and to steer preference selection along clinically faithful steps via reinforcement learning. This design mitigates variability across diseases and heterogeneity across views, reducing learning bottlenecks while aligning the model's inference with obstetric practice. To train FetalMind at scale, we curate FetalSigma-1M dataset, the first large-scale fetal ultrasound report corpus, comprising 20K reports from twelve medical centers, addressing the scarcity of domain data. Extensive experiments show that FetalMind outperforms open- and closed-source baselines across all gestational stages, achieving +14% average gains and +61.2% higher accuracy on critical conditions while remaining efficient, stable, and scalable. Project Page: this https URL. 

---
