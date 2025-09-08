# Hybrid Matrix Factorization Based Graph Contrastive Learning for Recommendation System 

**Authors**: Hao Chen, Wenming Ma, Zihao Chu, Mingqi Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.05115)  

**Abstract**: In recent years, methods that combine contrastive learning with graph neural networks have emerged to address the challenges of recommendation systems, demonstrating powerful performance and playing a significant role in this domain. Contrastive learning primarily tackles the issue of data sparsity by employing data augmentation strategies, effectively alleviating this problem and showing promising results. Although existing research has achieved favorable outcomes, most current graph contrastive learning methods are based on two types of data augmentation strategies: the first involves perturbing the graph structure, such as by randomly adding or removing edges; and the second applies clustering techniques. We believe that the interactive information obtained through these two strategies does not fully capture the user-item interactions. In this paper, we propose a novel method called HMFGCL (Hybrid Matrix Factorization Based Graph Contrastive Learning), which integrates two distinct matrix factorization techniques-low-rank matrix factorization (MF) and singular value decomposition (SVD)-to complementarily acquire global collaborative information, thereby constructing enhanced views. Experimental results on multiple public datasets demonstrate that our model outperforms existing baselines, particularly on small-scale datasets. 

---
# Fishing for Answers: Exploring One-shot vs. Iterative Retrieval Strategies for Retrieval Augmented Generation 

**Authors**: Huifeng Lin, Gang Su, Jintao Liang, You Wu, Rui Zhao, Ziyue Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.04820)  

**Abstract**: Retrieval-Augmented Generation (RAG) based on Large Language Models (LLMs) is a powerful solution to understand and query the industry's closed-source documents. However, basic RAG often struggles with complex QA tasks in legal and regulatory domains, particularly when dealing with numerous government documents. The top-$k$ strategy frequently misses golden chunks, leading to incomplete or inaccurate answers. To address these retrieval bottlenecks, we explore two strategies to improve evidence coverage and answer quality. The first is a One-SHOT retrieval method that adaptively selects chunks based on a token budget, allowing as much relevant content as possible to be included within the model's context window. Additionally, we design modules to further filter and refine the chunks. The second is an iterative retrieval strategy built on a Reasoning Agentic RAG framework, where a reasoning LLM dynamically issues search queries, evaluates retrieved results, and progressively refines the context over multiple turns. We identify query drift and retrieval laziness issues and further design two modules to tackle them. Through extensive experiments on a dataset of government documents, we aim to offer practical insights and guidance for real-world applications in legal and regulatory domains. 

---
# Multimodal Foundation Model-Driven User Interest Modeling and Behavior Analysis on Short Video Platforms 

**Authors**: Yushang Zhao, Yike Peng, Li Zhang, Qianyi Sun, Zhihui Zhang, Yingying Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2509.04751)  

**Abstract**: With the rapid expansion of user bases on short video platforms, personalized recommendation systems are playing an increasingly critical role in enhancing user experience and optimizing content distribution. Traditional interest modeling methods often rely on unimodal data, such as click logs or text labels, which limits their ability to fully capture user preferences in a complex multimodal content environment. To address this challenge, this paper proposes a multimodal foundation model-based framework for user interest modeling and behavior analysis. By integrating video frames, textual descriptions, and background music into a unified semantic space using cross-modal alignment strategies, the framework constructs fine-grained user interest vectors. Additionally, we introduce a behavior-driven feature embedding mechanism that incorporates viewing, liking, and commenting sequences to model dynamic interest evolution, thereby improving both the timeliness and accuracy of recommendations. In the experimental phase, we conduct extensive evaluations using both public and proprietary short video datasets, comparing our approach against multiple mainstream recommendation algorithms and modeling techniques. Results demonstrate significant improvements in behavior prediction accuracy, interest modeling for cold-start users, and recommendation click-through rates. Moreover, we incorporate interpretability mechanisms using attention weights and feature visualization to reveal the model's decision basis under multimodal inputs and trace interest shifts, thereby enhancing the transparency and controllability of the recommendation system. 

---
# Unified Representation Learning for Multi-Intent Diversity and Behavioral Uncertainty in Recommender Systems 

**Authors**: Wei Xu, Jiasen Zheng, Junjiang Lin, Mingxuan Han, Junliang Du  

**Link**: [PDF](https://arxiv.org/pdf/2509.04694)  

**Abstract**: This paper addresses the challenge of jointly modeling user intent diversity and behavioral uncertainty in recommender systems. A unified representation learning framework is proposed. The framework builds a multi-intent representation module and an uncertainty modeling mechanism. It extracts multi-granularity interest structures from user behavior sequences. Behavioral ambiguity and preference fluctuation are captured using Bayesian distribution modeling. In the multi-intent modeling part, the model introduces multiple latent intent vectors. These vectors are weighted and fused using an attention mechanism to generate semantically rich representations of long-term user preferences. In the uncertainty modeling part, the model learns the mean and covariance of behavior representations through Gaussian distributions. This reflects the user's confidence in different behavioral contexts. Next, a learnable fusion strategy is used to combine long-term intent and short-term behavior signals. This produces the final user representation, improving both recommendation accuracy and robustness. The method is evaluated on standard public datasets. Experimental results show that it outperforms existing representative models across multiple metrics. It also demonstrates greater stability and adaptability under cold-start and behavioral disturbance scenarios. The approach alleviates modeling bottlenecks faced by traditional methods when dealing with complex user behavior. These findings confirm the effectiveness and practical value of the unified modeling strategy in real-world recommendation tasks. 

---
# Optimizing Small Transformer-Based Language Models for Multi-Label Sentiment Analysis in Short Texts 

**Authors**: Julius Neumann, Robert Lange, Yuni Susanti, Michael Färber  

**Link**: [PDF](https://arxiv.org/pdf/2509.04982)  

**Abstract**: Sentiment classification in short text datasets faces significant challenges such as class imbalance, limited training samples, and the inherent subjectivity of sentiment labels -- issues that are further intensified by the limited context in short texts. These factors make it difficult to resolve ambiguity and exacerbate data sparsity, hindering effective learning. In this paper, we evaluate the effectiveness of small Transformer-based models (i.e., BERT and RoBERTa, with fewer than 1 billion parameters) for multi-label sentiment classification, with a particular focus on short-text settings. Specifically, we evaluated three key factors influencing model performance: (1) continued domain-specific pre-training, (2) data augmentation using automatically generated examples, specifically generative data augmentation, and (3) architectural variations of the classification head. Our experiment results show that data augmentation improves classification performance, while continued pre-training on augmented datasets can introduce noise rather than boost accuracy. Furthermore, we confirm that modifications to the classification head yield only marginal benefits. These findings provide practical guidance for optimizing BERT-based models in resource-constrained settings and refining strategies for sentiment classification in short-text datasets. 

---
# REMOTE: A Unified Multimodal Relation Extraction Framework with Multilevel Optimal Transport and Mixture-of-Experts 

**Authors**: Xinkui Lin, Yongxiu Xu, Minghao Tang, Shilong Zhang, Hongbo Xu, Hao Xu, Yubin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.04844)  

**Abstract**: Multimodal relation extraction (MRE) is a crucial task in the fields of Knowledge Graph and Multimedia, playing a pivotal role in multimodal knowledge graph construction. However, existing methods are typically limited to extracting a single type of relational triplet, which restricts their ability to extract triplets beyond the specified types. Directly combining these methods fails to capture dynamic cross-modal interactions and introduces significant computational redundancy. Therefore, we propose a novel \textit{unified multimodal Relation Extraction framework with Multilevel Optimal Transport and mixture-of-Experts}, termed REMOTE, which can simultaneously extract intra-modal and inter-modal relations between textual entities and visual objects. To dynamically select optimal interaction features for different types of relational triplets, we introduce mixture-of-experts mechanism, ensuring the most relevant modality information is utilized. Additionally, considering that the inherent property of multilayer sequential encoding in existing encoders often leads to the loss of low-level information, we adopt a multilevel optimal transport fusion module to preserve low-level features while maintaining multilayer encoding, yielding more expressive representations. Correspondingly, we also create a Unified Multimodal Relation Extraction (UMRE) dataset to evaluate the effectiveness of our framework, encompassing diverse cases where the head and tail entities can originate from either text or image. Extensive experiments show that REMOTE effectively extracts various types of relational triplets and achieves state-of-the-art performanc on almost all metrics across two other public MRE datasets. We release our resources at this https URL. 

---
# KERAG: Knowledge-Enhanced Retrieval-Augmented Generation for Advanced Question Answering 

**Authors**: Yushi Sun, Kai Sun, Yifan Ethan Xu, Xiao Yang, Xin Luna Dong, Nan Tang, Lei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.04716)  

**Abstract**: Retrieval-Augmented Generation (RAG) mitigates hallucination in Large Language Models (LLMs) by incorporating external data, with Knowledge Graphs (KGs) offering crucial information for question answering. Traditional Knowledge Graph Question Answering (KGQA) methods rely on semantic parsing, which typically retrieves knowledge strictly necessary for answer generation, thus often suffer from low coverage due to rigid schema requirements and semantic ambiguity. We present KERAG, a novel KG-based RAG pipeline that enhances QA coverage by retrieving a broader subgraph likely to contain relevant information. Our retrieval-filtering-summarization approach, combined with fine-tuned LLMs for Chain-of-Thought reasoning on knowledge sub-graphs, reduces noises and improves QA for both simple and complex questions. Experiments demonstrate that KERAG surpasses state-of-the-art solutions by about 7% in quality and exceeds GPT-4o (Tool) by 10-21%. 

---
# Ecologically Valid Benchmarking and Adaptive Attention: Scalable Marine Bioacoustic Monitoring 

**Authors**: Nicholas R. Rasmussen, Rodrigue Rizk, Longwei Wang, KC Santosh  

**Link**: [PDF](https://arxiv.org/pdf/2509.04682)  

**Abstract**: Underwater Passive Acoustic Monitoring (UPAM) provides rich spatiotemporal data for long-term ecological analysis, but intrinsic noise and complex signal dependencies hinder model stability and generalization. Multilayered windowing has improved target sound localization, yet variability from shifting ambient noise, diverse propagation effects, and mixed biological and anthropogenic sources demands robust architectures and rigorous evaluation. We introduce GetNetUPAM, a hierarchical nested cross-validation framework designed to quantify model stability under ecologically realistic variability. Data are partitioned into distinct site-year segments, preserving recording heterogeneity and ensuring each validation fold reflects a unique environmental subset, reducing overfitting to localized noise and sensor artifacts. Site-year blocking enforces evaluation against genuine environmental diversity, while standard cross-validation on random subsets measures generalization across UPAM's full signal distribution, a dimension absent from current benchmarks. Using GetNetUPAM as the evaluation backbone, we propose the Adaptive Resolution Pooling and Attention Network (ARPA-N), a neural architecture for irregular spectrogram dimensions. Adaptive pooling with spatial attention extends the receptive field, capturing global context without excessive parameters. Under GetNetUPAM, ARPA-N achieves a 14.4% gain in average precision over DenseNet baselines and a log2-scale order-of-magnitude drop in variability across all metrics, enabling consistent detection across site-year folds and advancing scalable, accurate bioacoustic monitoring. 

---
