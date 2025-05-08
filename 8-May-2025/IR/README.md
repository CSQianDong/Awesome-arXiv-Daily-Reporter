# User and Recommender Behavior Over Time: Contextualizing Activity, Effectiveness, Diversity, and Fairness in Book Recommendation 

**Authors**: Samira Vaez Barenji, Sushobhan Parajuli, Michael D. Ekstrand  

**Link**: [PDF](https://arxiv.org/pdf/2505.04518)  

**Abstract**: Data is an essential resource for studying recommender systems. While there has been significant work on improving and evaluating state-of-the-art models and measuring various properties of recommender system outputs, less attention has been given to the data itself, particularly how data has changed over time. Such documentation and analysis provide guidance and context for designing and evaluating recommender systems, particularly for evaluation designs making use of time (e.g., temporal splitting). In this paper, we present a temporal explanatory analysis of the UCSD Book Graph dataset scraped from Goodreads, a social reading and recommendation platform active since 2006. We measure the book interaction data using a set of activity, diversity, and fairness metrics; we then train a set of collaborative filtering algorithms on rolling training windows to observe how the same measures evolve over time in the recommendations. Additionally, we explore whether the introduction of algorithmic recommendations in 2011 was followed by observable changes in user or recommender system behavior. 

---
# M2Rec: Multi-scale Mamba for Efficient Sequential Recommendation 

**Authors**: Qianru Zhang, Liang Qu, Honggang Wen, Dong Huang, Siu-Ming Yiu, Nguyen Quoc Viet Hung, Hongzhi Yin  

**Link**: [PDF](https://arxiv.org/pdf/2505.04445)  

**Abstract**: Sequential recommendation systems aim to predict users' next preferences based on their interaction histories, but existing approaches face critical limitations in efficiency and multi-scale pattern recognition. While Transformer-based methods struggle with quadratic computational complexity, recent Mamba-based models improve efficiency but fail to capture periodic user behaviors, leverage rich semantic information, or effectively fuse multimodal features. To address these challenges, we propose \model, a novel sequential recommendation framework that integrates multi-scale Mamba with Fourier analysis, Large Language Models (LLMs), and adaptive gating. First, we enhance Mamba with Fast Fourier Transform (FFT) to explicitly model periodic patterns in the frequency domain, separating meaningful trends from noise. Second, we incorporate LLM-based text embeddings to enrich sparse interaction data with semantic context from item descriptions. Finally, we introduce a learnable gate mechanism to dynamically balance temporal (Mamba), frequency (FFT), and semantic (LLM) features, ensuring harmonious multimodal fusion. Extensive experiments demonstrate that \model\ achieves state-of-the-art performance, improving Hit Rate@10 by 3.2\% over existing Mamba-based models while maintaining 20\% faster inference than Transformer baselines. Our results highlight the effectiveness of combining frequency analysis, semantic understanding, and adaptive fusion for sequential recommendation. Code and datasets are available at: this https URL. 

---
# Theoretical Guarantees for LT-TTD: A Unified Transformer-based Architecture for Two-Level Ranking Systems 

**Authors**: Ayoub Abraich  

**Link**: [PDF](https://arxiv.org/pdf/2505.04434)  

**Abstract**: Modern recommendation and search systems typically employ multi-stage ranking architectures to efficiently handle billions of candidates. The conventional approach uses distinct L1 (candidate retrieval) and L2 (re-ranking) models with different optimization objectives, introducing critical limitations including irreversible error propagation and suboptimal ranking. This paper identifies and analyzes the fundamental limitations of this decoupled paradigm and proposes LT-TTD (Listwise Transformer with Two-Tower Distillation), a novel unified architecture that bridges retrieval and ranking phases. Our approach combines the computational efficiency of two-tower models with the expressivity of transformers in a unified listwise learning framework. We provide a comprehensive theoretical analysis of our architecture and establish formal guarantees regarding error propagation mitigation, ranking quality improvements, and optimization convergence. We derive theoretical bounds showing that LT-TTD reduces the upper limit on irretrievable relevant items by a factor that depends on the knowledge distillation strength, and prove that our multi-objective optimization framework achieves a provably better global optimum than disjoint training. Additionally, we analyze the computational complexity of our approach, demonstrating that the asymptotic complexity remains within practical bounds for real-world applications. We also introduce UPQE, a novel evaluation metric specifically designed for unified ranking architectures that holistically captures retrieval quality, ranking performance, and computational efficiency. 

---
# LONGER: Scaling Up Long Sequence Modeling in Industrial Recommenders 

**Authors**: Zheng Chai, Qin Ren, Xijun Xiao, Huizhi Yang, Bo Han, Sijun Zhang, Di Chen, Hui Lu, Wenlin Zhao, Lele Yu, Xionghang Xie, Shiru Ren, Xiang Sun, Yaocheng Tan, Peng Xu, Yuchao Zheng, Di Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.04421)  

**Abstract**: Modeling ultra-long user behavior sequences is critical for capturing both long- and short-term preferences in industrial recommender systems. Existing solutions typically rely on two-stage retrieval or indirect modeling paradigms, incuring upstream-downstream inconsistency and computational inefficiency. In this paper, we present LONGER, a Long-sequence Optimized traNsformer for GPU-Efficient Recommenders. LONGER incorporates (i) a global token mechanism for stabilizing attention over long contexts, (ii) a token merge module with lightweight InnerTransformers and hybrid attention strategy to reduce quadratic complexity, and (iii) a series of engineering optimizations, including training with mixed-precision and activation recomputation, KV cache serving, and the fully synchronous model training and serving framework for unified GPU-based dense and sparse parameter updates. LONGER consistently outperforms strong baselines in both offline metrics and online A/B testing in both advertising and e-commerce services at ByteDance, validating its consistent effectiveness and industrial-level scaling laws. Currently, LONGER has been fully deployed at more than 10 influential scenarios at ByteDance, serving billion users. 

---
# CDE-Mapper: Using Retrieval-Augmented Language Models for Linking Clinical Data Elements to Controlled Vocabularies 

**Authors**: Komal Gilani, Marlo Verket, Christof Peters, Michel Dumontier, Hans-Peter Brunner-La Rocca, Visara Urovi  

**Link**: [PDF](https://arxiv.org/pdf/2505.04365)  

**Abstract**: The standardization of clinical data elements (CDEs) aims to ensure consistent and comprehensive patient information across various healthcare systems. Existing methods often falter when standardizing CDEs of varying representation and complex structure, impeding data integration and interoperability in clinical research. We introduce CDE-Mapper, an innovative framework that leverages Retrieval-Augmented Generation approach combined with Large Language Models to automate the linking of CDEs to controlled vocabularies. Our modular approach features query decomposition to manage varying levels of CDEs complexity, integrates expert-defined rules within prompt engineering, and employs in-context learning alongside multiple retriever components to resolve terminological ambiguities. In addition, we propose a knowledge reservoir validated by a human-in-loop approach, achieving accurate concept linking for future applications while minimizing computational costs. For four diverse datasets, CDE-Mapper achieved an average of 7.2\% higher accuracy improvement compared to baseline methods. This work highlights the potential of advanced language models in improving data harmonization and significantly advancing capabilities in clinical decision support systems and research. 

---
# To Judge or not to Judge: Using LLM Judgements for Advertiser Keyphrase Relevance at eBay 

**Authors**: Soumik Dey, Hansi Wu, Binbin Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.04209)  

**Abstract**: E-commerce sellers are recommended keyphrases based on their inventory on which they advertise to increase buyer engagement (clicks/sales). The relevance of advertiser keyphrases plays an important role in preventing the inundation of search systems with numerous irrelevant items that compete for attention in auctions, in addition to maintaining a healthy seller perception. In this work, we describe the shortcomings of training Advertiser keyphrase relevance filter models on click/sales/search relevance signals and the importance of aligning with human judgment, as sellers have the power to adopt or reject said keyphrase recommendations. In this study, we frame Advertiser keyphrase relevance as a complex interaction between 3 dynamical systems -- seller judgment, which influences seller adoption of our product, Advertising, which provides the keyphrases to bid on, and Search, who holds the auctions for the same keyphrases. This study discusses the practicalities of using human judgment via a case study at eBay Advertising and demonstrate that using LLM-as-a-judge en-masse as a scalable proxy for seller judgment to train our relevance models achieves a better harmony across the three systems -- provided that they are bound by a meticulous evaluation framework grounded in business metrics. 

---
# Towards Large-scale Generative Ranking 

**Authors**: Yanhua Huang, Yuqi Chen, Xiong Cao, Rui Yang, Mingliang Qi, Yinghao Zhu, Qingchang Han, Yaowei Liu, Zhaoyu Liu, Xuefeng Yao, Yuting Jia, Leilei Ma, Yinqi Zhang, Taoyu Zhu, Liujie Zhang, Lei Chen, Weihang Chen, Min Zhu, Ruiwen Xu, Lei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.04180)  

**Abstract**: Generative recommendation has recently emerged as a promising paradigm in information retrieval. However, generative ranking systems are still understudied, particularly with respect to their effectiveness and feasibility in large-scale industrial settings. This paper investigates this topic at the ranking stage of Xiaohongshu's Explore Feed, a recommender system that serves hundreds of millions of users. Specifically, we first examine how generative ranking outperforms current industrial recommenders. Through theoretical and empirical analyses, we find that the primary improvement in effectiveness stems from the generative architecture, rather than the training paradigm. To facilitate efficient deployment of generative ranking, we introduce RankGPT, a novel generative architecture for ranking. We validate the effectiveness and efficiency of our solution through online A/B experiments. The results show that RankGPT achieves significant improvements in user satisfaction with nearly equivalent computational resources compared to the existing production system. 

---
# CoCoB: Adaptive Collaborative Combinatorial Bandits for Online Recommendation 

**Authors**: Cairong Yan, Jinyi Han, Jin Ju, Yanting Zhang, Zijian Wang, Xuan Shao  

**Link**: [PDF](https://arxiv.org/pdf/2505.03840)  

**Abstract**: Clustering bandits have gained significant attention in recommender systems by leveraging collaborative information from neighboring users to better capture target user preferences. However, these methods often lack a clear definition of similar users and face challenges when users with unique preferences lack appropriate neighbors. In such cases, relying on divergent preferences of misidentified neighbors can degrade recommendation quality. To address these limitations, this paper proposes an adaptive Collaborative Combinatorial Bandits algorithm (CoCoB). CoCoB employs an innovative two-sided bandit architecture, applying bandit principles to both the user and item sides. The user-bandit employs an enhanced Bayesian model to explore user similarity, identifying neighbors based on a similarity probability threshold. The item-bandit treats items as arms, generating diverse recommendations informed by the user-bandit's output. CoCoB dynamically adapts, leveraging neighbor preferences when available or focusing solely on the target user otherwise. Regret analysis under a linear contextual bandit setting and experiments on three real-world datasets demonstrate CoCoB's effectiveness, achieving an average 2.4% improvement in F1 score over state-of-the-art methods. 

---
# An Adaptive Data-Resilient Multi-Modal Framework for Hierarchical Multi-Label Book Genre Identification 

**Authors**: Utsav Kumar Nareti, Soumi Chattopadhyay, Prolay Mallick, Suraj Kumar, Ayush Vikas Daga, Chandranath Adak, Adarsh Wase, Arjab Roy  

**Link**: [PDF](https://arxiv.org/pdf/2505.03839)  

**Abstract**: Identifying the finer details of a book's genres enhances user experience by enabling efficient book discovery and personalized recommendations, ultimately improving reader engagement and satisfaction. It also provides valuable insights into market trends and consumer preferences, allowing publishers and marketers to make data-driven decisions regarding book production and marketing strategies. While traditional book genre classification methods primarily rely on review data or textual analysis, incorporating additional modalities, such as book covers, blurbs, and metadata, can offer richer context and improve prediction accuracy. However, the presence of incomplete or noisy information across these modalities presents a significant challenge. This paper introduces IMAGINE (Intelligent Multi-modal Adaptive Genre Identification NEtwork), a framework designed to address these complexities. IMAGINE extracts robust feature representations from multiple modalities and dynamically selects the most informative sources based on data availability. It employs a hierarchical classification strategy to capture genre relationships and remains adaptable to varying input conditions. Additionally, we curate a hierarchical genre classification dataset that structures genres into a well-defined taxonomy, accommodating the diverse nature of literary works. IMAGINE integrates information from multiple sources and assigns multiple genre labels to each book, ensuring a more comprehensive classification. A key feature of our framework is its resilience to incomplete data, enabling accurate predictions even when certain modalities, such as text, images, or metadata, are missing or incomplete. Experimental results show that IMAGINE outperformed existing baselines in genre classification accuracy, particularly in scenarios with insufficient modality-specific data. 

---
# OBD-Finder: Explainable Coarse-to-Fine Text-Centric Oracle Bone Duplicates Discovery 

**Authors**: Chongsheng Zhang, Shuwen Wu, Yingqi Chen, Matthias Aßenmacher, Christian Heumann, Yi Men, Gaojuan Fan, João Gama  

**Link**: [PDF](https://arxiv.org/pdf/2505.03836)  

**Abstract**: Oracle Bone Inscription (OBI) is the earliest systematic writing system in China, while the identification of Oracle Bone (OB) duplicates is a fundamental issue in OBI research. In this work, we design a progressive OB duplicate discovery framework that combines unsupervised low-level keypoints matching with high-level text-centric content-based matching to refine and rank the candidate OB duplicates with semantic awareness and interpretability. We compare our approach with state-of-the-art content-based image retrieval and image matching methods, showing that our approach yields comparable recall performance and the highest simplified mean reciprocal rank scores for both Top-5 and Top-15 retrieval results, and with significantly accelerated computation efficiency. We have discovered over 60 pairs of new OB duplicates in real-world deployment, which were missed by OBI researchers for decades. The models, video illustration and demonstration of this work are available at: this https URL. 

---
# Sentiment-Aware Recommendation Systems in E-Commerce: A Review from a Natural Language Processing Perspective 

**Authors**: Yogesh Gajula  

**Link**: [PDF](https://arxiv.org/pdf/2505.03828)  

**Abstract**: E-commerce platforms generate vast volumes of user feedback, such as star ratings, written reviews, and comments. However, most recommendation engines rely primarily on numerical scores, often overlooking the nuanced opinions embedded in free text. This paper comprehensively reviews sentiment-aware recommendation systems from a natural language processing perspective, covering advancements from 2023 to early 2025. It highlights the benefits of integrating sentiment analysis into e-commerce recommenders to enhance prediction accuracy and explainability through detailed opinion extraction. Our survey categorizes recent work into four main approaches: deep learning classifiers that combine sentiment embeddings with user item interactions, transformer based methods for nuanced feature extraction, graph neural networks that propagate sentiment signals, and conversational recommenders that adapt in real time to user feedback. We summarize model architectures and demonstrate how sentiment flows through recommendation pipelines, impacting dialogue-based suggestions. Key challenges include handling noisy or sarcastic text, dynamic user preferences, and bias mitigation. Finally, we outline research gaps and provide a roadmap for developing smarter, fairer, and more user-centric recommendation tools. 

---
# Memory Assisted LLM for Personalized Recommendation System 

**Authors**: Jiarui Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.03824)  

**Abstract**: Large language models (LLMs) have demonstrated significant potential in solving recommendation tasks. With proven capabilities in understanding user preferences, LLM personalization has emerged as a critical area for providing tailored responses to individuals. Current studies explore personalization through prompt design and fine-tuning, paving the way for further research in personalized LLMs. However, existing approaches are either costly and inefficient in capturing diverse user preferences or fail to account for timely updates to user history. To address these gaps, we propose the Memory-Assisted Personalized LLM (MAP). Through user interactions, we first create a history profile for each user, capturing their preferences, such as ratings for historical items. During recommendation, we extract relevant memory based on similarity, which is then incorporated into the prompts to enhance personalized recommendations. In our experiments, we evaluate MAP using a sequential rating prediction task under two scenarios: single domain, where memory and tasks are from the same category (e.g., movies), and cross-domain (e.g., memory from movies and recommendation tasks in books). The results show that MAP outperforms regular LLM-based recommenders that integrate user history directly through prompt design. Moreover, as user history grows, MAP's advantage increases in both scenarios, making it more suitable for addressing successive personalized user requests. 

---
# Tetrahedron-Net for Medical Image Registration 

**Authors**: Jinhai Xiang, Shuai Guo, Qianru Han, Dantong Shi, Xinwei He, Xiang Bai  

**Link**: [PDF](https://arxiv.org/pdf/2505.04380)  

**Abstract**: Medical image registration plays a vital role in medical image processing. Extracting expressive representations for medical images is crucial for improving the registration quality. One common practice for this end is constructing a convolutional backbone to enable interactions with skip connections among feature extraction layers. The de facto structure, U-Net-like networks, has attempted to design skip connections such as nested or full-scale ones to connect one single encoder and one single decoder to improve its representation capacity. Despite being effective, it still does not fully explore interactions with a single encoder and decoder architectures. In this paper, we embrace this observation and introduce a simple yet effective alternative strategy to enhance the representations for registrations by appending one additional decoder. The new decoder is designed to interact with both the original encoder and decoder. In this way, it not only reuses feature presentation from corresponding layers in the encoder but also interacts with the original decoder to corporately give more accurate registration results. The new architecture is concise yet generalized, with only one encoder and two decoders forming a ``Tetrahedron'' structure, thereby dubbed Tetrahedron-Net. Three instantiations of Tetrahedron-Net are further constructed regarding the different structures of the appended decoder. Our extensive experiments prove that superior performance can be obtained on several representative benchmarks of medical image registration. Finally, such a ``Tetrahedron'' design can also be easily integrated into popular U-Net-like architectures including VoxelMorph, ViT-V-Net, and TransMorph, leading to consistent performance gains. 

---
# Retrieval Augmented Time Series Forecasting 

**Authors**: Sungwon Han, Seungeon Lee, Meeyoung Cha, Sercan O Arik, Jinsung Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2505.04163)  

**Abstract**: Time series forecasting uses historical data to predict future trends, leveraging the relationships between past observations and available features. In this paper, we propose RAFT, a retrieval-augmented time series forecasting method to provide sufficient inductive biases and complement the model's learning capacity. When forecasting the subsequent time frames, we directly retrieve historical data candidates from the training dataset with patterns most similar to the input, and utilize the future values of these candidates alongside the inputs to obtain predictions. This simple approach augments the model's capacity by externally providing information about past patterns via retrieval modules. Our empirical evaluations on ten benchmark datasets show that RAFT consistently outperforms contemporary baselines with an average win ratio of 86%. 

---
# The Influence of Text Variation on User Engagement in Cross-Platform Content Sharing 

**Authors**: Yibo Hu, Yiqiao Jin, Meng Ye, Ajay Divakaran, Srijan Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2505.03769)  

**Abstract**: In today's cross-platform social media landscape, understanding factors that drive engagement for multimodal content, especially text paired with visuals, remains complex. This study investigates how rewriting Reddit post titles adapted from YouTube video titles affects user engagement. First, we build and analyze a large dataset of Reddit posts sharing YouTube videos, revealing that 21% of post titles are minimally modified. Statistical analysis demonstrates that title rewrites measurably improve engagement. Second, we design a controlled, multi-phase experiment to rigorously isolate the effects of textual variations by neutralizing confounding factors like video popularity, timing, and community norms. Comprehensive statistical tests reveal that effective title rewrites tend to feature emotional resonance, lexical richness, and alignment with community-specific norms. Lastly, pairwise ranking prediction experiments using a fine-tuned BERT classifier achieves 74% accuracy, significantly outperforming near-random baselines, including GPT-4o. These results validate that our controlled dataset effectively minimizes confounding effects, allowing advanced models to both learn and demonstrate the impact of textual features on engagement. By bridging quantitative rigor with qualitative insights, this study uncovers engagement dynamics and offers a robust framework for future cross-platform, multimodal content strategies. 

---
