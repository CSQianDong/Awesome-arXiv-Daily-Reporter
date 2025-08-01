# Just Ask for Music (JAM): Multimodal and Personalized Natural Language Music Recommendation 

**Authors**: Alessandro B. Melchiorre, Elena V. Epure, Shahed Masoudian, Gustavo Escobedo, Anna Hausberger, Manuel Moussallam, Markus Schedl  

**Link**: [PDF](https://arxiv.org/pdf/2507.15826)  

**Abstract**: Natural language interfaces offer a compelling approach for music recommendation, enabling users to express complex preferences conversationally. While Large Language Models (LLMs) show promise in this direction, their scalability in recommender systems is limited by high costs and latency. Retrieval-based approaches using smaller language models mitigate these issues but often rely on single-modal item representations, overlook long-term user preferences, and require full model retraining, posing challenges for real-world deployment. In this paper, we present JAM (Just Ask for Music), a lightweight and intuitive framework for natural language music recommendation. JAM models user-query-item interactions as vector translations in a shared latent space, inspired by knowledge graph embedding methods like TransE. To capture the complexity of music and user intent, JAM aggregates multimodal item features via cross-attention and sparse mixture-of-experts. We also introduce JAMSessions, a new dataset of over 100k user-query-item triples with anonymized user/item embeddings, uniquely combining conversational queries and user long-term preferences. Our results show that JAM provides accurate recommendations, produces intuitive representations suitable for practical use cases, and can be easily integrated with existing music recommendation stacks. 

---
# RankMixer: Scaling Up Ranking Models in Industrial Recommenders 

**Authors**: Jie Zhu, Zhifang Fan, Xiaoxie Zhu, Yuchen Jiang, Hangyu Wang, Xintian Han, Haoran Ding, Xinmin Wang, Wenlin Zhao, Zhen Gong, Huizhi Yang, Zheng Chai, Zhe Chen, Yuchao Zheng, Qiwei Chen, Feng Zhang, Xun Zhou, Peng Xu, Xiao Yang, Di Wu, Zuotao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.15551)  

**Abstract**: Recent progress on large language models (LLMs) has spurred interest in scaling up recommendation systems, yet two practical obstacles remain. First, training and serving cost on industrial Recommenders must respect strict latency bounds and high QPS demands. Second, most human-designed feature-crossing modules in ranking models were inherited from the CPU era and fail to exploit modern GPUs, resulting in low Model Flops Utilization (MFU) and poor scalability. We introduce RankMixer, a hardware-aware model design tailored towards a unified and scalable feature-interaction architecture. RankMixer retains the transformer's high parallelism while replacing quadratic self-attention with multi-head token mixing module for higher efficiency. Besides, RankMixer maintains both the modeling for distinct feature subspaces and cross-feature-space interactions with Per-token FFNs. We further extend it to one billion parameters with a Sparse-MoE variant for higher ROI. A dynamic routing strategy is adapted to address the inadequacy and imbalance of experts training. Experiments show RankMixer's superior scaling abilities on a trillion-scale production dataset. By replacing previously diverse handcrafted low-MFU modules with RankMixer, we boost the model MFU from 4.5% to 45%, and scale our ranking model parameters by 100x while maintaining roughly the same inference latency. We verify RankMixer's universality with online A/B tests across three core application scenarios (Recommendation, Advertisement and Search). Finally, we launch 1B Dense-Parameters RankMixer for full traffic serving without increasing the serving cost, which improves user active days by 0.2% and total in-app usage duration by 0.5%. 

---
# Hierarchical Graph Information Bottleneck for Multi-Behavior Recommendation 

**Authors**: Hengyu Zhang, Chunxu Shen, Xiangguo Sun, Jie Tan, Yanchao Tan, Yu Rong, Hong Cheng, Lingling Yi  

**Link**: [PDF](https://arxiv.org/pdf/2507.15395)  

**Abstract**: In real-world recommendation scenarios, users typically engage with platforms through multiple types of behavioral interactions. Multi-behavior recommendation algorithms aim to leverage various auxiliary user behaviors to enhance prediction for target behaviors of primary interest (e.g., buy), thereby overcoming performance limitations caused by data sparsity in target behavior records. Current state-of-the-art approaches typically employ hierarchical design following either cascading (e.g., view$\rightarrow$cart$\rightarrow$buy) or parallel (unified$\rightarrow$behavior$\rightarrow$specific components) paradigms, to capture behavioral relationships. However, these methods still face two critical challenges: (1) severe distribution disparities across behaviors, and (2) negative transfer effects caused by noise in auxiliary behaviors. In this paper, we propose a novel model-agnostic Hierarchical Graph Information Bottleneck (HGIB) framework for multi-behavior recommendation to effectively address these challenges. Following information bottleneck principles, our framework optimizes the learning of compact yet sufficient representations that preserve essential information for target behavior prediction while eliminating task-irrelevant redundancies. To further mitigate interaction noise, we introduce a Graph Refinement Encoder (GRE) that dynamically prunes redundant edges through learnable edge dropout mechanisms. We conduct comprehensive experiments on three real-world public datasets, which demonstrate the superior effectiveness of our framework. Beyond these widely used datasets in the academic community, we further expand our evaluation on several real industrial scenarios and conduct an online A/B testing, showing again a significant improvement in multi-behavior recommendations. The source code of our proposed HGIB is available at this https URL. 

---
# GREAT: Guiding Query Generation with a Trie for Recommending Related Search about Video at Kuaishou 

**Authors**: Ninglu Shao, Jinshan Wang, Chenxu Wang, Qingbiao Li, Xiaoxue Zang, Han Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.15267)  

**Abstract**: Currently, short video platforms have become the primary place for individuals to share experiences and obtain information. To better meet users' needs for acquiring information while browsing short videos, some apps have introduced a search entry at the bottom of videos, accompanied with recommended relevant queries. This scenario is known as query recommendation in video-related search, where core task is item-to-query (I2Q) recommendation. As this scenario has only emerged in recent years, there is a notable scarcity of academic research and publicly available datasets in this domain. To address this gap, we systematically examine the challenges associated with this scenario for the first time. Subsequently, we release a large-scale dataset derived from real-world data pertaining to the query recommendation in video-\textit{\textbf{r}}elated \textit{\textbf{s}}earch on the \textit{\textbf{Kuai}}shou app (\textbf{KuaiRS}). Presently, existing methods rely on embeddings to calculate similarity for matching short videos with queries, lacking deep interaction between the semantic content and the query. In this paper, we introduce a novel LLM-based framework named \textbf{GREAT}, which \textit{\textbf{g}}uides que\textit{\textbf{r}}y g\textit{\textbf{e}}ner\textit{\textbf{a}}tion with a \textit{\textbf{t}}rie to address I2Q recommendation in related search. Specifically, we initially gather high-quality queries with high exposure and click-through rate to construct a query-based trie. During training, we enhance the LLM's capability to generate high-quality queries using the query-based trie. In the inference phase, the query-based trie serves as a guide for the token generation. Finally, we further refine the relevance and literal quality between items and queries via a post-processing module. Extensive offline and online experiments demonstrate the effectiveness of our proposed method. 

---
# SPAR: Scholar Paper Retrieval with LLM-based Agents for Enhanced Academic Search 

**Authors**: Xiaofeng Shi, Yuduo Li, Qian Kou, Longbin Yu, Jinxin Xie, Hua Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.15245)  

**Abstract**: Recent advances in large language models (LLMs) have opened new opportunities for academic literature retrieval. However, existing systems often rely on rigid pipelines and exhibit limited reasoning capabilities. We introduce SPAR, a multi-agent framework that incorporates RefChain-based query decomposition and query evolution to enable more flexible and effective search. To facilitate systematic evaluation, we also construct SPARBench, a challenging benchmark with expert-annotated relevance labels. Experimental results demonstrate that SPAR substantially outperforms strong baselines, achieving up to +56% F1 on AutoScholar and +23% F1 on SPARBench over the best-performing baseline. Together, SPAR and SPARBench provide a scalable, interpretable, and high-performing foundation for advancing research in scholarly retrieval. Code and data will be available at: this https URL 

---
# Click A, Buy B: Rethinking Conversion Attribution in E- Commerce Recommendations 

**Authors**: Xiangyu Zeng, Amit Jaspal, Bin Liu, Goutham Panneeru, Kevin Huang, Nicolas Bievre, Mohit Jaggi, Prathap Maniraju, Ankur Jain  

**Link**: [PDF](https://arxiv.org/pdf/2507.15113)  

**Abstract**: User journeys in e-commerce routinely violate the one-to-one assumption that a clicked item on an advertising platform is the same item later purchased on the merchant's website/app. For a significant number of converting sessions on our platform, users click product A but buy product B -- the Click A, Buy B (CABB) phenomenon. Training recommendation models on raw click-conversion pairs therefore rewards items that merely correlate with purchases, leading to biased learning and sub-optimal conversion rates. We reframe conversion prediction as a multi-task problem with separate heads for Click A Buy A (CABA) and Click A Buy B (CABB). To isolate informative CABB conversions from unrelated CABB conversions, we introduce a taxonomy-aware collaborative filtering weighting scheme where each product is first mapped to a leaf node in a product taxonomy, and a category-to-category similarity matrix is learned from large-scale co-engagement logs. This weighting amplifies pairs that reflect genuine substitutable or complementary relations while down-weighting coincidental cross-category purchases. Offline evaluation on e-commerce sessions reduces normalized entropy by 13.9% versus a last-click attribution baseline. An online A/B test on live traffic shows +0.25% gains in the primary business metric. 

---
# FullRecall: A Semantic Search-Based Ranking Approach for Maximizing Recall in Patent Retrieval 

**Authors**: Amna Ali, Liyanage C. De Silva, Pg Emeroylariffion Abas  

**Link**: [PDF](https://arxiv.org/pdf/2507.14946)  

**Abstract**: Patent examiners and inventors face significant pressure to verify the originality and non-obviousness of inventions, and the intricate nature of patent data intensifies the challenges of patent retrieval. Therefore, there is a pressing need to devise cutting-edge retrieval strategies that can reliably achieve the desired recall. This study introduces FullRecall, a novel patent retrieval approach that effectively manages the complexity of patent data while maintaining the reliability of relevance matching and maximising recall. It leverages IPC-guided knowledge to generate informative phrases, which are processed to extract key information in the form of noun phrases characterising the query patent under observation. From these, the top k keyphrases are selected to construct a query for retrieving a focused subset of the dataset. This initial retrieval step achieves complete recall, successfully capturing all relevant documents. To further refine the results, a ranking scheme is applied to the retrieved subset, reducing its size while maintaining 100% recall. This multi-phase process demonstrates an effective strategy for balancing precision and recall in patent retrieval tasks. Comprehensive experiments were conducted, and the results were compared with baseline studies, namely HRR2 [1] and ReQ-ReC [2]. The proposed approach yielded superior results, achieving 100% recall in all five test cases. However, HRR2[1] recall values across the five test cases were 10%, 25%, 33.3%, 0%, and 14.29%, while ReQ-ReC [2] showed 50% for the first test case, 25% for the second test case, and 0% for the third, fourth, and fifth test cases. The 100% recall ensures that no relevant prior art is overlooked, thereby strengthening the patent pre-filing and examination processes, hence reducing potential legal risks. 

---
# User Invariant Preference Learning for Multi-Behavior Recommendation 

**Authors**: Mingshi Yan, Zhiyong Cheng, Fan Liu, Yingda Lyu, Yahong Han  

**Link**: [PDF](https://arxiv.org/pdf/2507.14925)  

**Abstract**: In multi-behavior recommendation scenarios, analyzing users' diverse behaviors, such as click, purchase, and rating, enables a more comprehensive understanding of their interests, facilitating personalized and accurate recommendations. A fundamental assumption of multi-behavior recommendation methods is the existence of shared user preferences across behaviors, representing users' intrinsic interests. Based on this assumption, existing approaches aim to integrate information from various behaviors to enrich user representations. However, they often overlook the presence of both commonalities and individualities in users' multi-behavior preferences. These individualities reflect distinct aspects of preferences captured by different behaviors, where certain auxiliary behaviors may introduce noise, hindering the prediction of the target behavior. To address this issue, we propose a user invariant preference learning for multi-behavior recommendation (UIPL for short), aiming to capture users' intrinsic interests (referred to as invariant preferences) from multi-behavior interactions to mitigate the introduction of noise. Specifically, UIPL leverages the paradigm of invariant risk minimization to learn invariant preferences. To implement this, we employ a variational autoencoder (VAE) to extract users' invariant preferences, replacing the standard reconstruction loss with an invariant risk minimization constraint. Additionally, we construct distinct environments by combining multi-behavior data to enhance robustness in learning these preferences. Finally, the learned invariant preferences are used to provide recommendations for the target behavior. Extensive experiments on four real-world datasets demonstrate that UIPL significantly outperforms current state-of-the-art methods. 

---
# U-MARVEL: Unveiling Key Factors for Universal Multimodal Retrieval via Embedding Learning with MLLMs 

**Authors**: Xiaojie Li, Chu Li, Shi-Zhe Chen, Xi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.14902)  

**Abstract**: Universal multimodal retrieval (UMR), which aims to address complex retrieval tasks where both queries and candidates span diverse modalities, has been significantly advanced by the emergence of MLLMs. While state-of-the-art MLLM-based methods in the literature predominantly adopt contrastive learning principles, they often differ in their specific training recipes. Despite their success, the mechanisms underlying their retrieval capabilities remain largely unexplored, potentially resulting in suboptimal performance and limited generalization ability. To address these issues, we present a comprehensive study aimed at uncovering the key factors that drive effective embedding learning for UMR using MLLMs. We begin by implementing a general MLLM-based embedding learning pipeline, and systematically analyze the primary contributors to high-performing universal retrieval systems. Based on this, we explore various aspects of the details in embedding generation and training strategies, including progressive transition, hard negative mining and re-ranker distillation. Notably, our findings reveal that often-overlooked factors can have a substantial impact on model performance. Building on these discoveries, we introduce a unified framework termed U-MARVEL (\textbf{U}niversal \textbf{M}ultimod\textbf{A}l \textbf{R}etrie\textbf{V}al via \textbf{E}mbedding \textbf{L}earning), which outperforms state-of-the-art competitors on the M-BEIR benchmark by a large margin in supervised settings, and also exihibits strong zero-shot performance on several tasks such as composed image retrieval and text-to-video retrieval. These results underscore the generalization potential of our framework across various embedding-based retrieval tasks. Code is available at this https URL 

---
# Optimizing Legal Document Retrieval in Vietnamese with Semi-Hard Negative Mining 

**Authors**: Van-Hoang Le, Duc-Vu Nguyen, Kiet Van Nguyen, Ngan Luu-Thuy Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2507.14619)  

**Abstract**: Large Language Models (LLMs) face significant challenges in specialized domains like law, where precision and domain-specific knowledge are critical. This paper presents a streamlined two-stage framework consisting of Retrieval and Re-ranking to enhance legal document retrieval efficiency and accuracy. Our approach employs a fine-tuned Bi-Encoder for rapid candidate retrieval, followed by a Cross-Encoder for precise re-ranking, both optimized through strategic negative example mining. Key innovations include the introduction of the Exist@m metric to evaluate retrieval effectiveness and the use of semi-hard negatives to mitigate training bias, which significantly improved re-ranking performance. Evaluated on the SoICT Hackathon 2024 for Legal Document Retrieval, our team, 4Huiter, achieved a top-three position. While top-performing teams employed ensemble models and iterative self-training on large bge-m3 architectures, our lightweight, single-pass approach offered a competitive alternative with far fewer parameters. The framework demonstrates that optimized data processing, tailored loss functions, and balanced negative sampling are pivotal for building robust retrieval-augmented systems in legal contexts. 

---
# Enhancing POI Recommendation through Global Graph Disentanglement with POI Weighted Module 

**Authors**: Pei-Xuan Li, Wei-Yun Liang, Fandel Lin, Hsun-Ping Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2507.14612)  

**Abstract**: Next point of interest (POI) recommendation primarily predicts future activities based on users' past check-in data and current status, providing significant value to users and service providers. We observed that the popular check-in times for different POI categories vary. For example, coffee shops are crowded in the afternoon because people like to have coffee to refresh after meals, while bars are busy late at night. However, existing methods rarely explore the relationship between POI categories and time, which may result in the model being unable to fully learn users' tendencies to visit certain POI categories at different times. Additionally, existing methods for modeling time information often convert it into time embeddings or calculate the time interval and incorporate it into the model, making it difficult to capture the continuity of time. Finally, during POI prediction, various weighting information is often ignored, such as the popularity of each POI, the transition relationships between POIs, and the distances between POIs, leading to suboptimal performance. To address these issues, this paper proposes a novel next POI recommendation framework called Graph Disentangler with POI Weighted Module (GDPW). This framework aims to jointly consider POI category information and multiple POI weighting factors. Specifically, the proposed GDPW learns category and time representations through the Global Category Graph and the Global Category-Time Graph. Then, we disentangle category and time information through contrastive learning. After prediction, the final POI recommendation for users is obtained by weighting the prediction results based on the transition weights and distance relationships between POIs. We conducted experiments on two real-world datasets, and the results demonstrate that the proposed GDPW outperforms other existing models, improving performance by 3% to 11%. 

---
# Understanding Matching Mechanisms in Cross-Encoders 

**Authors**: Mathias Vast, Basile Van Cooten, Laure Soulier, Benjamin Piwowarski  

**Link**: [PDF](https://arxiv.org/pdf/2507.14604)  

**Abstract**: Neural IR architectures, particularly cross-encoders, are highly effective models whose internal mechanisms are mostly unknown. Most works trying to explain their behavior focused on high-level processes (e.g., what in the input influences the prediction, does the model adhere to known IR axioms) but fall short of describing the matching process. Instead of Mechanistic Interpretability approaches which specifically aim at explaining the hidden mechanisms of neural models, we demonstrate that more straightforward methods can already provide valuable insights. In this paper, we first focus on the attention process and extract causal insights highlighting the crucial roles of some attention heads in this process. Second, we provide an interpretation of the mechanism underlying matching detection. 

---
# RaMen: Multi-Strategy Multi-Modal Learning for Bundle Construction 

**Authors**: Huy-Son Nguyen, Quang-Huy Nguyen, Duc-Hoang Pham, Duc-Trong Le, Hoang-Quynh Le, Padipat Sitkrongwong, Atsuhiro Takasu, Masoud Mansoury  

**Link**: [PDF](https://arxiv.org/pdf/2507.14361)  

**Abstract**: Existing studies on bundle construction have relied merely on user feedback via bipartite graphs or enhanced item representations using semantic information. These approaches fail to capture elaborate relations hidden in real-world bundle structures, resulting in suboptimal bundle representations. To overcome this limitation, we propose RaMen, a novel method that provides a holistic multi-strategy approach for bundle construction. RaMen utilizes both intrinsic (characteristics) and extrinsic (collaborative signals) information to model bundle structures through Explicit Strategy-aware Learning (ESL) and Implicit Strategy-aware Learning (ISL). ESL employs task-specific attention mechanisms to encode multi-modal data and direct collaborative relations between items, thereby explicitly capturing essential bundle features. Moreover, ISL computes hyperedge dependencies and hypergraph message passing to uncover shared latent intents among groups of items. Integrating diverse strategies enables RaMen to learn more comprehensive and robust bundle representations. Meanwhile, Multi-strategy Alignment & Discrimination module is employed to facilitate knowledge transfer between learning strategies and ensure discrimination between items/bundles. Extensive experiments demonstrate the effectiveness of RaMen over state-of-the-art models on various domains, justifying valuable insights into complex item set problems. 

---
# A Reproducibility Study of Product-side Fairness in Bundle Recommendation 

**Authors**: Huy-Son Nguyen, Yuanna Liu, Masoud Mansoury, Mohammad Alian Nejadi, Alan Hanjalic, Maarten de Rijke  

**Link**: [PDF](https://arxiv.org/pdf/2507.14352)  

**Abstract**: Recommender systems are known to exhibit fairness issues, particularly on the product side, where products and their associated suppliers receive unequal exposure in recommended results. While this problem has been widely studied in traditional recommendation settings, its implications for bundle recommendation (BR) remain largely unexplored. This emerging task introduces additional complexity: recommendations are generated at the bundle level, yet user satisfaction and product (or supplier) exposure depend on both the bundle and the individual items it contains. Existing fairness frameworks and metrics designed for traditional recommender systems may not directly translate to this multi-layered setting. In this paper, we conduct a comprehensive reproducibility study of product-side fairness in BR across three real-world datasets using four state-of-the-art BR methods. We analyze exposure disparities at both the bundle and item levels using multiple fairness metrics, uncovering important patterns. Our results show that exposure patterns differ notably between bundles and items, revealing the need for fairness interventions that go beyond bundle-level assumptions. We also find that fairness assessments vary considerably depending on the metric used, reinforcing the need for multi-faceted evaluation. Furthermore, user behavior plays a critical role: when users interact more frequently with bundles than with individual items, BR systems tend to yield fairer exposure distributions across both levels. Overall, our findings offer actionable insights for building fairer bundle recommender systems and establish a vital foundation for future research in this emerging domain. 

---
# LOVO: Efficient Complex Object Query in Large-Scale Video Datasets 

**Authors**: Yuxin Liu, Yuezhang Peng, Hefeng Zhou, Hongze Liu, Xinyu Lu, Jiong Lou, Chentao Wu, Wei Zhao, Jie Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.14301)  

**Abstract**: The widespread deployment of cameras has led to an exponential increase in video data, creating vast opportunities for applications such as traffic management and crime surveillance. However, querying specific objects from large-scale video datasets presents challenges, including (1) processing massive and continuously growing data volumes, (2) supporting complex query requirements, and (3) ensuring low-latency execution. Existing video analysis methods struggle with either limited adaptability to unseen object classes or suffer from high query latency. In this paper, we present LOVO, a novel system designed to efficiently handle comp$\underline{L}$ex $\underline{O}$bject queries in large-scale $\underline{V}$ide$\underline{O}$ datasets. Agnostic to user queries, LOVO performs one-time feature extraction using pre-trained visual encoders, generating compact visual embeddings for key frames to build an efficient index. These visual embeddings, along with associated bounding boxes, are organized in an inverted multi-index structure within a vector database, which supports queries for any objects. During the query phase, LOVO transforms object queries to query embeddings and conducts fast approximate nearest-neighbor searches on the visual embeddings. Finally, a cross-modal rerank is performed to refine the results by fusing visual features with detailed textual features. Evaluation on real-world video datasets demonstrates that LOVO outperforms existing methods in handling complex queries, with near-optimal query accuracy and up to 85x lower search latency, while significantly reducing index construction costs. This system redefines the state-of-the-art object query approaches in video analysis, setting a new benchmark for complex object queries with a novel, scalable, and efficient approach that excels in dynamic environments. 

---
# A Fisher's exact test justification of the TF-IDF term-weighting scheme 

**Authors**: Paul Sheridan, Zeyad Ahmed, Aitazaz A. Farooque  

**Link**: [PDF](https://arxiv.org/pdf/2507.15742)  

**Abstract**: Term frequency-inverse document frequency, or TF-IDF for short, is arguably the most celebrated mathematical expression in the history of information retrieval. Conceived as a simple heuristic quantifying the extent to which a given term's occurrences are concentrated in any one given document out of many, TF-IDF and its many variants are routinely used as term-weighting schemes in diverse text analysis applications. There is a growing body of scholarship dedicated to placing TF-IDF on a sound theoretical foundation. Building on that tradition, this paper justifies the use of TF-IDF to the statistics community by demonstrating how the famed expression can be understood from a significance testing perspective. We show that the common TF-IDF variant TF-ICF is, under mild regularity conditions, closely related to the negative logarithm of the $p$-value from a one-tailed version of Fisher's exact test of statistical significance. As a corollary, we establish a connection between TF-IDF and the said negative log-transformed $p$-value under certain idealized assumptions. We further demonstrate, as a limiting case, that this same quantity converges to TF-IDF in the limit of an infinitely large document collection. The Fisher's exact test justification of TF-IDF equips the working statistician with a ready explanation of the term-weighting scheme's long-established effectiveness. 

---
# DeRAG: Black-box Adversarial Attacks on Multiple Retrieval-Augmented Generation Applications via Prompt Injection 

**Authors**: Jerry Wang, Fang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.15042)  

**Abstract**: Adversarial prompt attacks can significantly alter the reliability of Retrieval-Augmented Generation (RAG) systems by re-ranking them to produce incorrect outputs. In this paper, we present a novel method that applies Differential Evolution (DE) to optimize adversarial prompt suffixes for RAG-based question answering. Our approach is gradient-free, treating the RAG pipeline as a black box and evolving a population of candidate suffixes to maximize the retrieval rank of a targeted incorrect document to be closer to real world scenarios. We conducted experiments on the BEIR QA datasets to evaluate attack success at certain retrieval rank thresholds under multiple retrieving applications. Our results demonstrate that DE-based prompt optimization attains competitive (and in some cases higher) success rates compared to GGPP to dense retrievers and PRADA to sparse retrievers, while using only a small number of tokens (<=5 tokens) in the adversarial suffix. Furthermore, we introduce a readability-aware suffix construction strategy, validated by a statistically significant reduction in MLM negative log-likelihood with Welch's t-test. Through evaluations with a BERT-based adversarial suffix detector, we show that DE-generated suffixes evade detection, yielding near-chance detection accuracy. 

---
# GRACE: Generative Recommendation via Journey-Aware Sparse Attention on Chain-of-Thought Tokenization 

**Authors**: Luyi Ma, Wanjia Zhang, Kai Zhao, Abhishek Kulkarni, Lalitesh Morishetti, Anjana Ganesh, Ashish Ranjan, Aashika Padmanabhan, Jianpeng Xu, Jason Cho, Praveen Kanumala, Kaushiki Nag, Sumit Dutta, Kamiya Motwani, Malay Patel, Evren Korpeoglu, Sushant Kumar, Kannan Achan  

**Link**: [PDF](https://arxiv.org/pdf/2507.14758)  

**Abstract**: Generative models have recently demonstrated strong potential in multi-behavior recommendation systems, leveraging the expressive power of transformers and tokenization to generate personalized item sequences. However, their adoption is hindered by (1) the lack of explicit information for token reasoning, (2) high computational costs due to quadratic attention complexity and dense sequence representations after tokenization, and (3) limited multi-scale modeling over user history. In this work, we propose GRACE (Generative Recommendation via journey-aware sparse Attention on Chain-of-thought tokEnization), a novel generative framework for multi-behavior sequential recommendation. GRACE introduces a hybrid Chain-of-Thought (CoT) tokenization method that encodes user-item interactions with explicit attributes from product knowledge graphs (e.g., category, brand, price) over semantic tokenization, enabling interpretable and behavior-aligned generation. To address the inefficiency of standard attention, we design a Journey-Aware Sparse Attention (JSA) mechanism, which selectively attends to compressed, intra-, inter-, and current-context segments in the tokenized sequence. Experiments on two real-world datasets show that GRACE significantly outperforms state-of-the-art baselines, achieving up to +106.9% HR@10 and +106.7% NDCG@10 improvement over the state-of-the-art baseline on the Home domain, and +22.1% HR@10 on the Electronics domain. GRACE also reduces attention computation by up to 48% with long sequences. 

---
# Training oscillator Ising machines to assign the dynamic stability of their equilibrium points 

**Authors**: Yi Cheng, Zongli Lin  

**Link**: [PDF](https://arxiv.org/pdf/2507.14386)  

**Abstract**: We propose a neural network model, which, with appropriate assignment of the stability of its equilibrium points (EPs), achieves Hopfield-like associative memory. The oscillator Ising machine (OIM) is an ideal candidates for such a model, as all its $0/\pi$ binary EPs are structurally stable with their dynamic stability tunable by the coupling weights. Traditional Hopfield-based models store the desired patterns by designing the coupling weights between neurons. The design of coupling weights should simultaneously take into account both the existence and the dynamic stability of the EPs for the storage of the desired patterns. For OIMs, since all $0/\pi$ binary EPs are structurally stable, the design of the coupling weights needs only to focus on assigning appropriate stability for the $0/\pi$ binary EPs according to the desired patterns. In this paper, we establish a connection between the stability and the Hamiltonian energy of EPs for OIMs, and, based on this connection, provide a Hamiltonian-Regularized Eigenvalue Contrastive Method (HRECM) to train the coupling weights of OIMs for assigning appropriate stability to their EPs. Finally, numerical experiments are performed to validate the effectiveness of the proposed method. 

---
# Attention-Based Fusion of IQ and FFT Spectrograms with AoA Features for GNSS Jammer Localization 

**Authors**: Lucas Heublein, Christian Wielenberg, Thorsten Nowak, Tobias Feigl, Christopher Mutschler, Felix Ott  

**Link**: [PDF](https://arxiv.org/pdf/2507.14167)  

**Abstract**: Jamming devices disrupt signals from the global navigation satellite system (GNSS) and pose a significant threat by compromising the reliability of accurate positioning. Consequently, the detection and localization of these interference signals are essential to achieve situational awareness, mitigating their impact, and implementing effective counter-measures. Classical Angle of Arrival (AoA) methods exhibit reduced accuracy in multipath environments due to signal reflections and scattering, leading to localization errors. Additionally, AoA-based techniques demand substantial computational resources for array signal processing. In this paper, we propose a novel approach for detecting and classifying interference while estimating the distance, azimuth, and elevation of jamming sources. Our benchmark study evaluates 128 vision encoder and time-series models to identify the highest-performing methods for each task. We introduce an attention-based fusion framework that integrates in-phase and quadrature (IQ) samples with Fast Fourier Transform (FFT)-computed spectrograms while incorporating 22 AoA features to enhance localization accuracy. Furthermore, we present a novel dataset of moving jamming devices recorded in an indoor environment with dynamic multipath conditions and demonstrate superior performance compared to state-of-the-art methods. 

---
