# Democratizing News Recommenders: Modeling Multiple Perspectives for News Candidate Generation with VQ-VAE 

**Authors**: Hardy, Sebastian Padó, Amelie Wührl, Tanise Ceron  

**Link**: [PDF](https://arxiv.org/pdf/2508.13978)  

**Abstract**: Current News Recommender Systems based on past clicks are designed for engagement, but come at the cost of limiting diversity in the suggested content. While diversity-aware algorithms exist, they suffer from two major limitations. First, they fail to account for normative diversity, which requires fair access to a broad range of perspectives. Second, they typically apply diversity late in the system's pipeline, after a lot of content has already been filtered out. Both limitations confine their effectiveness and prevent them from promoting true normative diversity in news recommendations.
We propose Aspect-Aware Candidate Generation (A2CG) to address these limitations. Our framework introduces diversity into the earliest pipeline stage and uses a configurable mechanism to align diversity with specific democratic goals. A2CG represents each news article using multiple aspects of perspectives (e.g., sentiment, political leaning, frame) and uses a Vector Quantized Variational Autoencoder (VQ-VAE) to create a discrete, multi-faceted representation. A decoder-only model then learns user preferences over these aspect codes. We then inject diversity directly by reversing the sign on some of the query vector's aspects during the candidate retrieval process, ensuring a more diverse set of candidates.
Our method, evaluated on the MIND dataset, enables a flexible trade-off between personalization and diversity early in the recommendation pipeline. It also generates more novel, diverse, and serendipitous candidates while effectively taking into account aspects that strengthen democratic values. These empirical results make it a promising approach for downstream democratized news recommendation systems. 

---
# InPars+: Supercharging Synthetic Data Generation for Information Retrieval Systems 

**Authors**: Matey Krastev, Miklos Hamar, Danilo Toapanta, Jesse Brouwers, Yibin Lei  

**Link**: [PDF](https://arxiv.org/pdf/2508.13930)  

**Abstract**: This work revisits and extends synthetic query generation pipelines for Neural Information Retrieval (NIR) by leveraging the InPars Toolkit, a reproducible, end-to-end framework for generating training data using large language models (LLMs). We first assess the reproducibility of the original InPars, InPars-V2, and Promptagator pipelines on the SciFact benchmark and validate their effectiveness using open-source reranker and generator models. Building on this foundation, we introduce two key extensions to the pipeline: (1) fine-tuning a query generator LLM via Contrastive Preference Optimization (CPO) to improve the signal quality in generated queries, and (2) replacing static prompt templates with dynamic, Chain-of-Thought (CoT) optimized prompts using the DSPy framework. Our results show that both extensions reduce the need for aggressive filtering while improving retrieval performance. All code, models, and synthetic datasets are publicly released to support further research at: \href{this https URL}{this https URL}. 

---
# CARE: Contextual Adaptation of Recommenders for LLM-based Conversational Recommendation 

**Authors**: Chuang Li, Yang Deng, Hengchang Hu, See-Kiong Ng, Min-Yen Kan, Haizhou Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.13889)  

**Abstract**: We tackle the challenge of integrating large language models (LLMs) with external recommender systems to enhance domain expertise in conversational recommendation (CRS). Current LLM-based CRS approaches primarily rely on zero- or few-shot methods for generating item recommendations based on user queries, but this method faces two significant challenges: (1) without domain-specific adaptation, LLMs frequently recommend items not in the target item space, resulting in low recommendation accuracy; and (2) LLMs largely rely on dialogue context for content-based recommendations, neglecting the collaborative relationships among entities or item sequences. To address these limitations, we introduce the CARE (Contextual Adaptation of Recommenders) framework. CARE customizes LLMs for CRS tasks, and synergizes them with external recommendation systems. CARE (a) integrates external recommender systems as domain experts, producing recommendations through entity-level insights, and (b) enhances those recommendations by leveraging contextual information for more accurate and unbiased final recommendations using LLMs. Our results demonstrate that incorporating external recommender systems with entity-level information significantly enhances recommendation accuracy of LLM-based CRS by an average of 54% and 25% for ReDial and INSPIRED datasets. The most effective strategy in the CARE framework involves LLMs selecting and reranking candidate items that external recommenders provide based on contextual insights. Our analysis indicates that the CARE framework effectively addresses the identified challenges and mitigates the popularity bias in the external recommender. 

---
# Bites of Tomorrow: Personalized Recommendations for a Healthier and Greener Plate 

**Authors**: Jiazheng Jing, Yinan Zhang, Chunyan Miao  

**Link**: [PDF](https://arxiv.org/pdf/2508.13870)  

**Abstract**: The recent emergence of extreme climate events has significantly raised awareness about sustainable living. In addition to developing energy-saving materials and technologies, existing research mainly relies on traditional methods that encourage behavioral shifts towards sustainability, which can be overly demanding or only passively engaging. In this work, we propose to employ recommendation systems to actively nudge users toward more sustainable choices. We introduce Green Recommender Aligned with Personalized Eating (GRAPE), which is designed to prioritize and recommend sustainable food options that align with users' evolving preferences. We also design two innovative Green Loss functions that cater to green indicators with either uniform or differentiated priorities, thereby enhancing adaptability across a range of scenarios. Extensive experiments on a real-world dataset demonstrate the effectiveness of our GRAPE. 

---
# UniECS: Unified Multimodal E-Commerce Search Framework with Gated Cross-modal Fusion 

**Authors**: Zihan Liang, Yufei Ma, ZhiPeng Qian, Huangyu Dai, Zihan Wang, Ben Chen, Chenyi Lei, Yuqing Ding, Han Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.13843)  

**Abstract**: Current e-commerce multimodal retrieval systems face two key limitations: they optimize for specific tasks with fixed modality pairings, and lack comprehensive benchmarks for evaluating unified retrieval approaches. To address these challenges, we introduce UniECS, a unified multimodal e-commerce search framework that handles all retrieval scenarios across image, text, and their combinations. Our work makes three key contributions. First, we propose a flexible architecture with a novel gated multimodal encoder that uses adaptive fusion mechanisms. This encoder integrates different modality representations while handling missing modalities. Second, we develop a comprehensive training strategy to optimize learning. It combines cross-modal alignment loss (CMAL), cohesive local alignment loss (CLAL), intra-modal contrastive loss (IMCL), and adaptive loss weighting. Third, we create M-BEER, a carefully curated multimodal benchmark containing 50K product pairs for e-commerce search evaluation. Extensive experiments demonstrate that UniECS consistently outperforms existing methods across four e-commerce benchmarks with fine-tuning or zero-shot evaluation. On our M-BEER bench, UniECS achieves substantial improvements in cross-modal tasks (up to 28\% gain in R@10 for text-to-image retrieval) while maintaining parameter efficiency (0.2B parameters) compared to larger models like GME-Qwen2VL (2B) and MM-Embed (8B). Furthermore, we deploy UniECS in the e-commerce search platform of Kuaishou Inc. across two search scenarios, achieving notable improvements in Click-Through Rate (+2.74\%) and Revenue (+8.33\%). The comprehensive evaluation demonstrates the effectiveness of our approach in both experimental and real-world settings. Corresponding codes, models and datasets will be made publicly available at this https URL. 

---
# Refining Contrastive Learning and Homography Relations for Multi-Modal Recommendation 

**Authors**: Shouxing Ma, Yawen Zeng, Shiqing Wu, Guandong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.13745)  

**Abstract**: Multi-modal recommender system focuses on utilizing rich modal information ( i.e., images and textual descriptions) of items to improve recommendation performance. The current methods have achieved remarkable success with the powerful structure modeling capability of graph neural networks. However, these methods are often hindered by sparse data in real-world scenarios. Although contrastive learning and homography ( i.e., homogeneous graphs) are employed to address the data sparsity challenge, existing methods still suffer two main limitations: 1) Simple multi-modal feature contrasts fail to produce effective representations, causing noisy modal-shared features and loss of valuable information in modal-unique features; 2) The lack of exploration of the homograph relations between user interests and item co-occurrence results in incomplete mining of user-item interplay.
To address the above limitations, we propose a novel framework for \textbf{R}\textbf{E}fining multi-mod\textbf{A}l cont\textbf{R}astive learning and ho\textbf{M}ography relations (\textbf{REARM}). Specifically, we complement multi-modal contrastive learning by employing meta-network and orthogonal constraint strategies, which filter out noise in modal-shared features and retain recommendation-relevant information in modal-unique features. To mine homogeneous relationships effectively, we integrate a newly constructed user interest graph and an item co-occurrence graph with the existing user co-occurrence and item semantic graphs for graph learning. The extensive experiments on three real-world datasets demonstrate the superiority of REARM to various state-of-the-art baselines. Our visualization further shows an improvement made by REARM in distinguishing between modal-shared and modal-unique features. Code is available \href{this https URL}{here}. 

---
# MUFFIN: Mixture of User-Adaptive Frequency Filtering for Sequential Recommendation 

**Authors**: Ilwoong Baek, Mincheol Yoon, Seongmin Park, Jongwuk Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.13670)  

**Abstract**: Sequential recommendation (SR) aims to predict users' subsequent interactions by modeling their sequential behaviors. Recent studies have explored frequency domain analysis, which effectively models periodic patterns in user sequences. However, existing frequency-domain SR models still face two major drawbacks: (i) limited frequency band coverage, often missing critical behavioral patterns in a specific frequency range, and (ii) lack of personalized frequency filtering, as they apply an identical filter for all users regardless of their distinct frequency characteristics. To address these challenges, we propose a novel frequency-domain model, Mixture of User-adaptive Frequency FIlteriNg (MUFFIN), operating through two complementary modules. (i) The global filtering module (GFM) handles the entire frequency spectrum to capture comprehensive behavioral patterns. (ii) The local filtering module (LFM) selectively emphasizes important frequency bands without excluding information from other ranges. (iii) In both modules, the user-adaptive filter (UAF) is adopted to generate user-specific frequency filters tailored to individual unique characteristics. Finally, by aggregating both modules, MUFFIN captures diverse user behavioral patterns across the full frequency spectrum. Extensive experiments show that MUFFIN consistently outperforms state-of-the-art frequency-domain SR models over five benchmark datasets. The source code is available at this https URL. 

---
# Understanding Distribution Structure on Calibrated Recommendation Systems 

**Authors**: Diego Correa da Silva, Denis Robson Dantas Boaventura, Mayki dos Santos Oliveira, Eduardo Ferreira da Silva, Joel Machado Pires, Frederico Araújo Durão  

**Link**: [PDF](https://arxiv.org/pdf/2508.13568)  

**Abstract**: Traditional recommender systems aim to generate a recommendation list comprising the most relevant or similar items to the user's profile. These approaches can create recommendation lists that omit item genres from the less prominent areas of a user's profile, thereby undermining the user's experience. To solve this problem, the calibrated recommendation system provides a guarantee of including less representative areas in the recommended list. The calibrated context works with three distributions. The first is from the user's profile, the second is from the candidate items, and the last is from the recommendation list. These distributions are G-dimensional, where G is the total number of genres in the system. This high dimensionality requires a different evaluation method, considering that traditional recommenders operate in a one-dimensional data space. In this sense, we implement fifteen models that help to understand how these distributions are structured. We evaluate the users' patterns in three datasets from the movie domain. The results indicate that the models of outlier detection provide a better understanding of the structures. The calibrated system creates recommendation lists that act similarly to traditional recommendation lists, allowing users to change their groups of preferences to the same degree. 

---
# ENCODE: Breaking the Trade-Off Between Performance and Efficiency in Long-Term User Behavior Modeling 

**Authors**: Wenji Zhou, Yuhang Zheng, Yinfu Feng, Yunan Ye, Rong Xiao, Long Chen, Xiaosong Yang, Jun Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2508.13567)  

**Abstract**: Long-term user behavior sequences are a goldmine for businesses to explore users' interests to improve Click-Through Rate. However, it is very challenging to accurately capture users' long-term interests from their long-term behavior sequences and give quick responses from the online serving systems. To meet such requirements, existing methods "inadvertently" destroy two basic requirements in long-term sequence modeling: R1) make full use of the entire sequence to keep the information as much as possible; R2) extract information from the most relevant behaviors to keep high relevance between learned interests and current target items. The performance of online serving systems is significantly affected by incomplete and inaccurate user interest information obtained by existing methods. To this end, we propose an efficient two-stage long-term sequence modeling approach, named as EfficieNt Clustering based twO-stage interest moDEling (ENCODE), consisting of offline extraction stage and online inference stage. It not only meets the aforementioned two basic requirements but also achieves a desirable balance between online service efficiency and precision. Specifically, in the offline extraction stage, ENCODE clusters the entire behavior sequence and extracts accurate interests. To reduce the overhead of the clustering process, we design a metric learning-based dimension reduction algorithm that preserves the relative pairwise distances of behaviors in the new feature space. While in the online inference stage, ENCODE takes the off-the-shelf user interests to predict the associations with target items. Besides, to further ensure the relevance between user interests and target items, we adopt the same relevance metric throughout the whole pipeline of ENCODE. The extensive experiment and comparison with SOTA have demonstrated the effectiveness and efficiency of our proposed ENCODE. 

---
# Heterogeneous Influence Maximization in User Recommendation 

**Authors**: Hongru Hou, Jiachen Sun, Wenqing Lin, Wendong Bi, Xiangrong Wang, Deqing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13517)  

**Abstract**: User recommendation systems enhance user engagement by encouraging users to act as inviters to interact with other users (invitees), potentially fostering information propagation. Conventional recommendation methods typically focus on modeling interaction willingness. Influence-Maximization (IM) methods focus on identifying a set of users to maximize the information propagation. However, existing methods face two significant challenges. First, recommendation methods fail to unleash the candidates' spread capability. Second, IM methods fail to account for the willingness to interact. To solve these issues, we propose two models named HeteroIR and HeteroIM. HeteroIR provides an intuitive solution to unleash the dissemination potential of user recommendation systems. HeteroIM fills the gap between the IM method and the recommendation task, improving interaction willingness and maximizing spread coverage. The HeteroIR introduces a two-stage framework to estimate the spread profits. The HeteroIM incrementally selects the most influential invitee to recommend and rerank based on the number of reverse reachable (RR) sets containing inviters and invitees. RR set denotes a set of nodes that can reach a target via propagation. Extensive experiments show that HeteroIR and HeteroIM significantly outperform the state-of-the-art baselines with the p-value < 0.05. Furthermore, we have deployed HeteroIR and HeteroIM in Tencent's online gaming platforms and gained an 8.5\% and 10\% improvement in the online A/B test, respectively. Implementation codes are available at this https URL. 

---
# LLM-Enhanced Linear Autoencoders for Recommendation 

**Authors**: Jaewan Moon, Seongmin Park, Jongwuk Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.13500)  

**Abstract**: Large language models (LLMs) have been widely adopted to enrich the semantic representation of textual item information in recommender systems. However, existing linear autoencoders (LAEs) that incorporate textual information rely on sparse word co-occurrence patterns, limiting their ability to capture rich textual semantics. To address this, we propose L3AE, the first integration of LLMs into the LAE framework. L3AE effectively integrates the heterogeneous knowledge of textual semantics and user-item interactions through a two-phase optimization strategy. (i) L3AE first constructs a semantic item-to-item correlation matrix from LLM-derived item representations. (ii) It then learns an item-to-item weight matrix from collaborative signals while distilling semantic item correlations as regularization. Notably, each phase of L3AE is optimized through closed-form solutions, ensuring global optimality and computational efficiency. Extensive experiments demonstrate that L3AE consistently outperforms state-of-the-art LLM-enhanced models on three benchmark datasets, achieving gains of 27.6% in Recall@20 and 39.3% in NDCG@20. The source code is available at this https URL. 

---
# AdaptJobRec: Enhancing Conversational Career Recommendation through an LLM-Powered Agentic System 

**Authors**: Qixin Wang, Dawei Wang, Kun Chen, Yaowei Hu, Puneet Girdhar, Ruoteng Wang, Aadesh Gupta, Chaitanya Devella, Wenlai Guo, Shangwen Huang, Bachir Aoun, Greg Hayworth, Han Li, Xintao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.13423)  

**Abstract**: In recent years, recommendation systems have evolved from providing a single list of recommendations to offering a comprehensive suite of topic focused services. To better accomplish this task, conversational recommendation systems (CRS) have progressed from basic retrieval augmented LLM generation to agentic systems with advanced reasoning and self correction capabilities. However, agentic systems come with notable response latency, a longstanding challenge for conversational recommendation systems. To balance the trade off between handling complex queries and minimizing latency, we propose AdaptJobRec, the first conversational job recommendation system that leverages autonomous agent to integrate personalized recommendation algorithm tools. The system employs a user query complexity identification mechanism to minimize response latency. For straightforward queries, the agent directly selects the appropriate tool for rapid responses. For complex queries, the agent uses the memory processing module to filter chat history for relevant content, then passes the results to the intelligent task decomposition planner, and finally executes the tasks using personalized recommendation tools. Evaluation on Walmart's real world career recommendation scenarios demonstrates that AdaptJobRec reduces average response latency by up to 53.3% compared to competitive baselines, while significantly improving recommendation accuracy. 

---
# CASPER: Concept-integrated Sparse Representation for Scientific Retrieval 

**Authors**: Lam Thanh Do, Linh Van Nguyen, David Fu, Kevin Chen-Chuan Chang  

**Link**: [PDF](https://arxiv.org/pdf/2508.13394)  

**Abstract**: The exponential growth of scientific literature has made it increasingly difficult for researchers to keep up with the literature. In an attempt to alleviate this problem, we propose CASPER, a sparse retrieval model for scientific search that utilizes tokens and keyphrases as representation units (i.e. dimensions in the sparse embedding space), enabling it to represent queries and documents with research concepts and match them at both granular and conceptual levels. To overcome the lack of suitable training data, we propose mining training data by leveraging scholarly references (i.e. signals that capture how research concepts of papers are expressed in different settings), including titles, citation contexts, author-assigned keyphrases, and co-citations. CASPER outperforms strong dense and sparse retrieval baselines on eight scientific retrieval benchmarks. Moreover, we demonstrate that through simple post-processing, CASPER can be effectively used for the keyphrase generation tasks, achieving competitive performance with the established CopyRNN while producing more diverse keyphrases and being nearly four times faster. 

---
# FLAIR: Feedback Learning for Adaptive Information Retrieval 

**Authors**: William Zhang, Yiwen Zhu, Yunlei Lu, Mathieu Demarne, Wenjing Wang, Kai Deng, Nutan Sahoo, Katherine Lin, Miso Cilimdzic, Subru Krishnan  

**Link**: [PDF](https://arxiv.org/pdf/2508.13390)  

**Abstract**: Recent advances in Large Language Models (LLMs) have driven the adoption of copilots in complex technical scenarios, underscoring the growing need for specialized information retrieval solutions. In this paper, we introduce FLAIR, a lightweight, feedback learning framework that adapts copilot systems' retrieval strategies by integrating domain-specific expert feedback. FLAIR operates in two stages: an offline phase obtains indicators from (1) user feedback and (2) questions synthesized from documentation, storing these indicators in a decentralized manner. An online phase then employs a two-track ranking mechanism to combine raw similarity scores with the collected indicators. This iterative setup refines retrieval performance for any query. Extensive real-world evaluations of FLAIR demonstrate significant performance gains on both previously seen and unseen queries, surpassing state-of-the-art approaches. The system has been successfully integrated into Copilot DECO, serving thousands of users at Microsoft, demonstrating its scalability and effectiveness in operational environments. 

---
# Research on Conversational Recommender System Considering Consumer Types 

**Authors**: Yaying Luo, Hui Fang, Zhu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.13209)  

**Abstract**: Conversational Recommender Systems (CRS) provide personalized services through multi-turn interactions, yet most existing methods overlook users' heterogeneous decision-making styles and knowledge levels, which constrains both accuracy and efficiency. To address this gap, we propose CT-CRS (Consumer Type-Enhanced Conversational Recommender System), a framework that integrates consumer type modeling into dialogue recommendation. Based on consumer type theory, we define four user categories--dependent, efficient, cautious, and expert--derived from two dimensions: decision-making style (maximizers vs. satisficers) and knowledge level (high vs. low). CT-CRS employs interaction histories and fine-tunes the large language model to automatically infer user types in real time, avoiding reliance on static questionnaires. We incorporate user types into state representation and design a type-adaptive policy that dynamically adjusts recommendation granularity, diversity, and attribute query complexity. To further optimize the dialogue policy, we adopt Inverse Reinforcement Learning (IRL), enabling the agent to approximate expert-like strategies conditioned on consumer type. Experiments on LastFM, Amazon-Book, and Yelp show that CTCRS improves recommendation success rate and reduces interaction turns compared to strong baselines. Ablation studies confirm that both consumer type modeling and IRL contribute significantly to performance gains. These results demonstrate that CT-CRS offers a scalable and interpretable solution for enhancing CRS personalization through the integration of psychological modeling and advanced policy optimization. 

---
# Trust and Reputation in Data Sharing: A Survey 

**Authors**: Wenbo Wu, George Konstantinidis  

**Link**: [PDF](https://arxiv.org/pdf/2508.14028)  

**Abstract**: Data sharing is the fuel of the galloping artificial intelligence economy, providing diverse datasets for training robust models. Trust between data providers and data consumers is widely considered one of the most important factors for enabling data sharing initiatives. Concerns about data sensitivity, privacy breaches, and misuse contribute to reluctance in sharing data across various domains. In recent years, there has been a rise in technological and algorithmic solutions to measure, capture and manage trust, trustworthiness, and reputation in what we collectively refer to as Trust and Reputation Management Systems (TRMSs). Such approaches have been developed and applied to different domains of computer science, such as autonomous vehicles, or IoT networks, but there have not been dedicated approaches to data sharing and its unique characteristics. In this survey, we examine TRMSs from a data-sharing perspective, analyzing how they assess the trustworthiness of both data and entities across different environments. We develop novel taxonomies for system designs, trust evaluation framework, and evaluation metrics for both data and entity, and we systematically analyze the applicability of existing TRMSs in data sharing. Finally, we identify open challenges and propose future research directions to enhance the explainability, comprehensiveness, and accuracy of TRMSs in large-scale data-sharing ecosystems. 

---
# TASER: Table Agents for Schema-guided Extraction and Recommendation 

**Authors**: Nicole Cho, Kirsty Fielding, William Watson, Sumitra Ganesh, Manuela Veloso  

**Link**: [PDF](https://arxiv.org/pdf/2508.13404)  

**Abstract**: Real-world financial documents report essential information about an entity's financial holdings that can span millions of different financial instrument types. Yet, these details are often buried in messy, multi-page, fragmented tables - for example, 99.4% of the tables in our dataset have no bounding boxes with the maximum number of rows amounting to 426 per table across 44 pages. To tackle these unique challenges from real-world tables, we present a continuously learning, agentic table extraction system, TASER (Table Agents for Schema-guided Extraction and Recommendation) that extracts highly unstructured, multi-page, heterogeneous tables into normalized, schema-conforming outputs. Our table agents execute on table detection, classification, extraction, and recommendations by leveraging an initial schema. Then, our Recommender Agent reviews the outputs, recommends schema revisions, and decides on the final recommendations, enabling TASER to outperform existing table detection models such as Table Transformer by 10.1%. Within this continuous learning process, we highlight that larger batch sizes result in a 104.3% increase in schema recommendations that are actionable and utilized, resulting in a 9.8% increase in extracted holdings - highlighting the importance of a continuous learning process. To train TASER, we have manually labeled 22,584 pages (28,150,449 tokens), 3,213 tables for $731,685,511,687 of holdings culminating in one of the first real financial table datasets. We release our dataset TASERTab to enable the research community to access real-world financial tables and outputs. Our results highlight the promise of agentic, schema-guided extraction systems for robust understanding of real-world financial tables. 

---
# Explicit v.s. Implicit Memory: Exploring Multi-hop Complex Reasoning Over Personalized Information 

**Authors**: Zeyu Zhang, Yang Zhang, Haoran Tan, Rui Li, Xu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.13250)  

**Abstract**: In large language model-based agents, memory serves as a critical capability for achieving personalization by storing and utilizing users' information. Although some previous studies have adopted memory to implement user personalization, they typically focus on preference alignment and simple question-answering. However, in the real world, complex tasks often require multi-hop reasoning on a large amount of user information, which poses significant challenges for current memory approaches. To address this limitation, we propose the multi-hop personalized reasoning task to explore how different memory mechanisms perform in multi-hop reasoning over personalized information. We explicitly define this task and construct a dataset along with a unified evaluation framework. Then, we implement various explicit and implicit memory methods and conduct comprehensive experiments. We evaluate their performance on this task from multiple perspectives and analyze their strengths and weaknesses. Besides, we explore hybrid approaches that combine both paradigms and propose the HybridMem method to address their limitations. We demonstrate the effectiveness of our proposed model through extensive experiments. To benefit the research community, we release this project at this https URL. 

---
# Contextual Attention-Based Multimodal Fusion of LLM and CNN for Sentiment Analysis 

**Authors**: Meriem Zerkouk, Miloud Mihoubi, Belkacem Chikhaoui  

**Link**: [PDF](https://arxiv.org/pdf/2508.13196)  

**Abstract**: This paper introduces a novel approach for multimodal sentiment analysis on social media, particularly in the context of natural disasters, where understanding public sentiment is crucial for effective crisis management. Unlike conventional methods that process text and image modalities separately, our approach seamlessly integrates Convolutional Neural Network (CNN) based image analysis with Large Language Model (LLM) based text processing, leveraging Generative Pre-trained Transformer (GPT) and prompt engineering to extract sentiment relevant features from the CrisisMMD dataset. To effectively model intermodal relationships, we introduce a contextual attention mechanism within the fusion process. Leveraging contextual-attention layers, this mechanism effectively captures intermodality interactions, enhancing the model's comprehension of complex relationships between textual and visual data. The deep neural network architecture of our model learns from these fused features, leading to improved accuracy compared to existing baselines. Experimental results demonstrate significant advancements in classifying social media data into informative and noninformative categories across various natural disasters. Our model achieves a notable 2.43% increase in accuracy and 5.18% in F1-score, highlighting its efficacy in processing complex multimodal data. Beyond quantitative metrics, our approach provides deeper insight into the sentiments expressed during crises. The practical implications extend to real time disaster management, where enhanced sentiment analysis can optimize the accuracy of emergency interventions. By bridging the gap between multimodal analysis, LLM powered text understanding, and disaster response, our work presents a promising direction for Artificial Intelligence (AI) driven crisis management solutions. Keywords: 

---
