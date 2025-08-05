# Hubness Reduction with Dual Bank Sinkhorn Normalization for Cross-Modal Retrieval 

**Authors**: Zhengxin Pan, Haishuai Wang, Fangyu Wu, Peng Zhang, Jiajun Bu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02538)  

**Abstract**: The past decade has witnessed rapid advancements in cross-modal retrieval, with significant progress made in accurately measuring the similarity between cross-modal pairs. However, the persistent hubness problem, a phenomenon where a small number of targets frequently appear as nearest neighbors to numerous queries, continues to hinder the precision of similarity measurements. Despite several proposed methods to reduce hubness, their underlying mechanisms remain poorly understood. To bridge this gap, we analyze the widely-adopted Inverted Softmax approach and demonstrate its effectiveness in balancing target probabilities during retrieval. Building on these insights, we propose a probability-balancing framework for more effective hubness reduction. We contend that balancing target probabilities alone is inadequate and, therefore, extend the framework to balance both query and target probabilities by introducing Sinkhorn Normalization (SN). Notably, we extend SN to scenarios where the true query distribution is unknown, showing that current methods, which rely solely on a query bank to estimate target hubness, produce suboptimal results due to a significant distributional gap between the query bank and targets. To mitigate this issue, we introduce Dual Bank Sinkhorn Normalization (DBSN), incorporating a corresponding target bank alongside the query bank to narrow this distributional gap. Our comprehensive evaluation across various cross-modal retrieval tasks, including image-text retrieval, video-text retrieval, and audio-text retrieval, demonstrates consistent performance improvements, validating the effectiveness of both SN and DBSN. All codes are publicly available at this https URL. 

---
# Decomposed Reasoning with Reinforcement Learning for Relevance Assessment in UGC Platforms 

**Authors**: Xiaowei Yuan, Lei Jin, Haoxin Zhang, Yan Gao, Yi Wu, Yao Hu, Ziyang Huang, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02506)  

**Abstract**: Retrieval-augmented generation (RAG) plays a critical role in user-generated content (UGC) platforms, but its effectiveness depends heavily on accurate relevance assessment of query-document pairs. Despite recent advances in applying large language models (LLMs) to relevance modeling, UGC platforms present unique challenges: 1) ambiguous user intent due to sparse user feedback in RAG scenarios, and 2) substantial noise introduced by informal and unstructured language. To address these issues, we propose the Reinforced Reasoning Model for Relevance Assessment (R3A), which introduces a decomposed reasoning framework over queries and candidate documents before scoring. R3A first leverages auxiliary high-ranked documents within the platform to infer latent query intent. It then performs verbatim fragment extraction to justify relevance decisions, thereby reducing errors caused by noisy UGC. Based on a reinforcement learning framework, R3A is optimized to mitigate distortions arising from ambiguous queries and unstructured content. Experimental results show that R3A significantly outperforms existing baseline methods in terms of relevance accuracy, across both offline benchmarks and online experiments. 

---
# Dynamic Forgetting and Spatio-Temporal Periodic Interest Modeling for Local-Life Service Recommendation 

**Authors**: Zhaoyu Hu, Hao Guo, Yuan Tian, Erpeng Xue, Jianyang Wang, Xianyang Qi, Hongxiang Lin, Lei Wang, Sheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.02451)  

**Abstract**: In the context of the booming digital economy, recommendation systems, as a key link connecting users and numerous services, face challenges in modeling user behavior sequences on local-life service platforms, including the sparsity of long sequences and strong spatio-temporal dependence. Such challenges can be addressed by drawing an analogy to the forgetting process in human memory. This is because users' responses to recommended content follow the recency effect and the cyclicality of memory. By exploring this, this paper introduces the forgetting curve and proposes Spatio-Temporal periodic Interest Modeling (STIM) with long sequences for local-life service recommendation. STIM integrates three key components: a dynamic masking module based on the forgetting curve, which is used to extract both recent spatiotemporal features and periodic spatiotemporal features; a query-based mixture of experts (MoE) approach that can adaptively activate expert networks under different dynamic masks, enabling the collaborative modeling of time, location, and items; and a hierarchical multi-interest network unit, which captures multi-interest representations by modeling the hierarchical interactions between the shallow and deep semantics of users' recent behaviors. By introducing the STIM method, we conducted online A/B tests and achieved a 1.54\% improvement in gross transaction volume (GTV). In addition, extended offline experiments also showed improvements. STIM has been deployed in a large-scale local-life service recommendation system, serving hundreds of millions of daily active users in core application scenarios. 

---
# Beyond Chunks and Graphs: Retrieval-Augmented Generation through Triplet-Driven Thinking 

**Authors**: Shengbo Gong, Xianfeng Tang, Carl Yang, Wei jin  

**Link**: [PDF](https://arxiv.org/pdf/2508.02435)  

**Abstract**: Retrieval-augmented generation (RAG) is critical for reducing hallucinations and incorporating external knowledge into Large Language Models (LLMs). However, advanced RAG systems face a trade-off between performance and efficiency. Multi-round RAG approaches achieve strong reasoning but incur excessive LLM calls and token costs, while Graph RAG methods suffer from computationally expensive, error-prone graph construction and retrieval redundancy. To address these challenges, we propose T$^2$RAG, a novel framework that operates on a simple, graph-free knowledge base of atomic triplets. T$^2$RAG leverages an LLM to decompose questions into searchable triplets with placeholders, which it then iteratively resolves by retrieving evidence from the triplet database. Empirical results show that T$^2$RAG significantly outperforms state-of-the-art multi-round and Graph RAG methods, achieving an average performance gain of up to 11\% across six datasets while reducing retrieval costs by up to 45\%. Our code is available at this https URL 

---
# Agentic Personalized Fashion Recommendation in the Age of Generative AI: Challenges, Opportunities, and Evaluation 

**Authors**: Yashar Deldjoo, Nima Rafiee, Mahdyar Ravanbakhsh  

**Link**: [PDF](https://arxiv.org/pdf/2508.02342)  

**Abstract**: Fashion recommender systems (FaRS) face distinct challenges due to rapid trend shifts, nuanced user preferences, intricate item-item compatibility, and the complex interplay among consumers, brands, and influencers. Traditional recommendation approaches, largely static and retrieval-focused, struggle to effectively capture these dynamic elements, leading to decreased user satisfaction and elevated return rates. This paper synthesizes both academic and industrial viewpoints to map the distinctive output space and stakeholder ecosystem of modern FaRS, identifying the complex interplay among users, brands, platforms, and influencers, and highlighting the unique data and modeling challenges that arise.
We outline a research agenda for industrial FaRS, centered on five representative scenarios spanning static queries, outfit composition, and multi-turn dialogue, and argue that mixed-modality refinement-the ability to combine image-based references (anchors) with nuanced textual constraints-is a particularly critical task for real-world deployment. To this end, we propose an Agentic Mixed-Modality Refinement (AMMR) pipeline, which fuses multimodal encoders with agentic LLM planners and dynamic retrieval, bridging the gap between expressive user intent and fast-changing fashion inventories. Our work shows that moving beyond static retrieval toward adaptive, generative, and stakeholder-aware systems is essential to satisfy the evolving expectations of fashion consumers and brands. 

---
# Research Knowledge Graphs in NFDI4DataScience: Key Activities, Achievements, and Future Directions 

**Authors**: Kanishka Silva, Marcel R. Ackermann, Heike Fliegl, Genet-Asefa Gesese, Fidan Limani, Philipp Mayr, Peter Mutschke, Allard Oelen, Muhammad Asif Suryani, Sharmila Upadhyaya, Benjamin Zapilko, Harald Sack, Stefan Dietze  

**Link**: [PDF](https://arxiv.org/pdf/2508.02300)  

**Abstract**: As research in Artificial Intelligence and Data Science continues to grow in volume and complexity, it becomes increasingly difficult to ensure transparency, reproducibility, and discoverability. To address these challenges, as research artifacts should be understandable and usable by machines, the NFDI4DataScience consortium is developing and providing Research Knowledge Graphs (RKGs). Building upon earlier works, this paper presents recent progress in creating semantically rich RKGs using standardized ontologies, shared vocabularies, and automated Information Extraction techniques. Key achievements include the development of the NFDI4DS ontology, metadata standards, tools, and services designed to support the FAIR principles, as well as community-led projects and various implementations of RKGs. Together, these efforts aim to capture and connect the complex relationships between datasets, models, software, and scientific publications. 

---
# Voronoi Diagram Encoded Hashing 

**Authors**: Yang Xu, Kai Ming Ting  

**Link**: [PDF](https://arxiv.org/pdf/2508.02266)  

**Abstract**: The goal of learning to hash (L2H) is to derive data-dependent hash functions from a given data distribution in order to map data from the input space to a binary coding space. Despite the success of L2H, two observations have cast doubt on the source of the power of L2H, i.e., learning. First, a recent study shows that even using a version of locality sensitive hashing functions without learning achieves binary representations that have comparable accuracy as those of L2H, but with less time cost. Second, existing L2H methods are constrained to three types of hash functions: thresholding, hyperspheres, and hyperplanes only. In this paper, we unveil the potential of Voronoi diagrams in hashing. Voronoi diagram is a suitable candidate because of its three properties. This discovery has led us to propose a simple and efficient no-learning binary hashing method, called Voronoi Diagram Encoded Hashing (VDeH), which constructs a set of hash functions through a data-dependent similarity measure and produces independent binary bits through encoded hashing. We demonstrate through experiments on several benchmark datasets that VDeH achieves superior performance and lower computational cost compared to existing state-of-the-art methods under the same bit length. 

---
# From Generation to Consumption: Personalized List Value Estimation for Re-ranking 

**Authors**: Kaike Zhang, Xiaobei Wang, Xiaoyu Liu, Shuchang Liu, Hailan Yang, Xiang Li, Fei Sun, Qi Cao  

**Link**: [PDF](https://arxiv.org/pdf/2508.02242)  

**Abstract**: Re-ranking is critical in recommender systems for optimizing the order of recommendation lists, thus improving user satisfaction and platform revenue. Most existing methods follow a generator-evaluator paradigm, where the evaluator estimates the overall value of each candidate list. However, they often ignore the fact that users may exit before consuming the full list, leading to a mismatch between estimated generation value and actual consumption value. To bridge this gap, we propose CAVE, a personalized Consumption-Aware list Value Estimation framework. CAVE formulates the list value as the expectation over sub-list values, weighted by user-specific exit probabilities at each position. The exit probability is decomposed into an interest-driven component and a stochastic component, the latter modeled via a Weibull distribution to capture random external factors such as fatigue. By jointly modeling sub-list values and user exit behavior, CAVE yields a more faithful estimate of actual list consumption value. We further contribute three large-scale real-world list-wise benchmarks from the Kuaishou platform, varying in size and user activity patterns. Extensive experiments on these benchmarks, two Amazon datasets, and online A/B testing on Kuaishou show that CAVE consistently outperforms strong baselines, highlighting the benefit of explicitly modeling user exits in re-ranking. 

---
# FinCPRG: A Bidirectional Generation Pipeline for Hierarchical Queries and Rich Relevance in Financial Chinese Passage Retrieval 

**Authors**: Xuan Xu, Beilin Chu, Qinhong Lin, Yixiao Zhong, Fufang Wen, Jiaqi Liu, Binjie Fei, Yu Li, Zhongliang Yang, Linna Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.02222)  

**Abstract**: In recent years, large language models (LLMs) have demonstrated significant potential in constructing passage retrieval datasets. However, existing methods still face limitations in expressing cross-doc query needs and controlling annotation quality. To address these issues, this paper proposes a bidirectional generation pipeline, which aims to generate 3-level hierarchical queries for both intra-doc and cross-doc scenarios and mine additional relevance labels on top of direct mapping annotation. The pipeline introduces two query generation methods: bottom-up from single-doc text and top-down from multi-doc titles. The bottom-up method uses LLMs to disassemble and generate structured queries at both sentence-level and passage-level simultaneously from intra-doc passages. The top-down approach incorporates three key financial elements--industry, topic, and time--to divide report titles into clusters and prompts LLMs to generate topic-level queries from each cluster. For relevance annotation, our pipeline not only relies on direct mapping annotation from the generation relationship but also implements an indirect positives mining method to enrich the relevant query-passage pairs. Using this pipeline, we constructed a Financial Passage Retrieval Generated dataset (FinCPRG) from almost 1.3k Chinese financial research reports, which includes hierarchical queries and rich relevance labels. Through evaluations of mined relevance labels, benchmarking and training experiments, we assessed the quality of FinCPRG and validated its effectiveness as a passage retrieval dataset for both training and benchmarking. 

---
# Evaluating User Experience in Conversational Recommender Systems: A Systematic Review Across Classical and LLM-Powered Approaches 

**Authors**: Raj Mahmud, Yufeng Wu, Abdullah Bin Sawad, Shlomo Berkovsky, Mukesh Prasad, A. Baki Kocaballi  

**Link**: [PDF](https://arxiv.org/pdf/2508.02096)  

**Abstract**: Conversational Recommender Systems (CRSs) are receiving growing research attention across domains, yet their user experience (UX) evaluation remains limited. Existing reviews largely overlook empirical UX studies, particularly in adaptive and large language model (LLM)-based CRSs. To address this gap, we conducted a systematic review following PRISMA guidelines, synthesising 23 empirical studies published between 2017 and 2025. We analysed how UX has been conceptualised, measured, and shaped by domain, adaptivity, and LLM.
Our findings reveal persistent limitations: post hoc surveys dominate, turn-level affective UX constructs are rarely assessed, and adaptive behaviours are seldom linked to UX outcomes. LLM-based CRSs introduce further challenges, including epistemic opacity and verbosity, yet evaluations infrequently address these issues. We contribute a structured synthesis of UX metrics, a comparative analysis of adaptive and nonadaptive systems, and a forward-looking agenda for LLM-aware UX evaluation. These findings support the development of more transparent, engaging, and user-centred CRS evaluation practices. 

---
# Why Generate When You Can Transform? Unleashing Generative Attention for Dynamic Recommendation 

**Authors**: Yuli Liu, Wenjun Kong, Cheng Luo, Weizhi Ma  

**Link**: [PDF](https://arxiv.org/pdf/2508.02050)  

**Abstract**: Sequential Recommendation (SR) focuses on personalizing user experiences by predicting future preferences based on historical interactions. Transformer models, with their attention mechanisms, have become the dominant architecture in SR tasks due to their ability to capture dependencies in user behavior sequences. However, traditional attention mechanisms, where attention weights are computed through query-key transformations, are inherently linear and deterministic. This fixed approach limits their ability to account for the dynamic and non-linear nature of user preferences, leading to challenges in capturing evolving interests and subtle behavioral patterns. Given that generative models excel at capturing non-linearity and probabilistic variability, we argue that generating attention distributions offers a more flexible and expressive alternative compared to traditional attention mechanisms. To support this claim, we present a theoretical proof demonstrating that generative attention mechanisms offer greater expressiveness and stochasticity than traditional deterministic approaches. Building upon this theoretical foundation, we introduce two generative attention models for SR, each grounded in the principles of Variational Autoencoders (VAE) and Diffusion Models (DMs), respectively. These models are designed specifically to generate adaptive attention distributions that better align with variable user preferences. Extensive experiments on real-world datasets show our models significantly outperform state-of-the-art in both accuracy and diversity. 

---
# Evaluating Position Bias in Large Language Model Recommendations 

**Authors**: Ethan Bito, Yongli Ren, Estrid He  

**Link**: [PDF](https://arxiv.org/pdf/2508.02020)  

**Abstract**: Large Language Models (LLMs) are being increasingly explored as general-purpose tools for recommendation tasks, enabling zero-shot and instruction-following capabilities without the need for task-specific training. While the research community is enthusiastically embracing LLMs, there are important caveats to directly adapting them for recommendation tasks. In this paper, we show that LLM-based recommendation models suffer from position bias, where the order of candidate items in a prompt can disproportionately influence the recommendations produced by LLMs. First, we analyse the position bias of LLM-based recommendations on real-world datasets, where results uncover systemic biases of LLMs with high sensitivity to input orders. Furthermore, we introduce a new prompting strategy to mitigate the position bias of LLM recommendation models called Ranking via Iterative SElection (RISE). We compare our proposed method against various baselines on key benchmark datasets. Experiment results show that our method reduces sensitivity to input ordering and improves stability without requiring model fine-tuning or post-processing. 

---
# Counterfactual Reciprocal Recommender Systems for User-to-User Matching 

**Authors**: Kazuki Kawamura, Takuma Udagawa, Kei Tateno  

**Link**: [PDF](https://arxiv.org/pdf/2508.01867)  

**Abstract**: Reciprocal recommender systems (RRS) in dating, gaming, and talent platforms require mutual acceptance for a match. Logged data, however, over-represents popular profiles due to past exposure policies, creating feedback loops that skew learning and fairness. We introduce Counterfactual Reciprocal Recommender Systems (CFRR), a causal framework to mitigate this bias. CFRR uses inverse propensity scored, self-normalized objectives. Experiments show CFRR improves NDCG@10 by up to 3.5% (e.g., from 0.459 to 0.475 on DBLP, from 0.299 to 0.307 on Synthetic), increases long-tail user coverage by up to 51% (from 0.504 to 0.763 on Synthetic), and reduces Gini exposure inequality by up to 24% (from 0.708 to 0.535 on Synthetic). CFRR offers a promising approach for more accurate and fair user-to-user matching. 

---
# ChEmbed: Enhancing Chemical Literature Search Through Domain-Specific Text Embeddings 

**Authors**: Ali Shiraee Kasmaee, Mohammad Khodadad, Mehdi Astaraki, Mohammad Arshi Saloot, Nicholas Sherck, Hamidreza Mahyar, Soheila Samiee  

**Link**: [PDF](https://arxiv.org/pdf/2508.01643)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems in chemistry heavily depend on accurate and relevant retrieval of chemical literature. However, general-purpose text embedding models frequently fail to adequately represent complex chemical terminologies, resulting in suboptimal retrieval quality. Specialized embedding models tailored to chemical literature retrieval have not yet been developed, leaving a substantial performance gap. To address this challenge, we introduce ChEmbed, a domain-adapted family of text embedding models fine-tuned on a dataset comprising chemistry-specific text from the PubChem, Semantic Scholar, and ChemRxiv corpora. To create effective training data, we employ large language models to synthetically generate queries, resulting in approximately 1.7 million high-quality query-passage pairs. Additionally, we augment the tokenizer by adding 900 chemically specialized tokens to previously unused slots, which significantly reduces the fragmentation of chemical entities, such as IUPAC names. ChEmbed also maintains a 8192-token context length, enabling the efficient retrieval of longer passages compared to many other open-source embedding models, which typically have a context length of 512 or 2048 tokens. Evaluated on our newly introduced ChemRxiv Retrieval benchmark, ChEmbed outperforms state-of-the-art general embedding models, raising nDCG@10 from 0.82 to 0.91 (+9 pp). ChEmbed represents a practical, lightweight, and reproducible embedding solution that effectively improves retrieval for chemical literature search. 

---
# End-to-End Personalization: Unifying Recommender Systems with Large Language Models 

**Authors**: Danial Ebrat, Tina Aminian, Sepideh Ahmadian, Luis Rueda  

**Link**: [PDF](https://arxiv.org/pdf/2508.01514)  

**Abstract**: Recommender systems are essential for guiding users through the vast and diverse landscape of digital content by delivering personalized and relevant suggestions. However, improving both personalization and interpretability remains a challenge, particularly in scenarios involving limited user feedback or heterogeneous item attributes. In this article, we propose a novel hybrid recommendation framework that combines Graph Attention Networks (GATs) with Large Language Models (LLMs) to address these limitations. LLMs are first used to enrich user and item representations by generating semantically meaningful profiles based on metadata such as titles, genres, and overviews. These enriched embeddings serve as initial node features in a user and movie bipartite graph, which is processed using a GAT based collaborative filtering model. To enhance ranking accuracy, we introduce a hybrid loss function that combines Bayesian Personalized Ranking (BPR), cosine similarity, and robust negative sampling. Post-processing involves reranking the GAT-generated recommendations using the LLM, which also generates natural-language justifications to improve transparency. We evaluated our model on benchmark datasets, including MovieLens 100k and 1M, where it consistently outperforms strong baselines. Ablation studies confirm that LLM-based embeddings and the cosine similarity term significantly contribute to performance gains. This work demonstrates the potential of integrating LLMs to improve both the accuracy and interpretability of recommender systems. 

---
# Req-Rec: Enhancing Requirements Elicitation for Increasing Stakeholder's Satisfaction Using a Collaborative Filtering Based Recommender System 

**Authors**: Ali Fallahi, Amineh Amini, Azam Bastanfard, Hadi Saboohi  

**Link**: [PDF](https://arxiv.org/pdf/2508.01502)  

**Abstract**: The success or failure of a project is highly related to recognizing the right stakeholders and accurately finding and discovering their requirements. However, choosing the proper elicitation technique was always a considerable challenge for efficient requirement engineering. As a consequence of the swift improvement of digital technologies since the past decade, recommender systems have become an efficient channel for making a deeply personalized interactive communication with stakeholders. In this research, a new method, called the Req-Rec (Requirements Recommender), is proposed. It is a hybrid recommender system based on the collaborative filtering approach and the repertory grid technique as the core component. The primary goal of Req-Rec is to increase stakeholder satisfaction by assisting them in the requirement elicitation phase. Based on the results, the method efficiently could overcome weaknesses of common requirement elicitation techniques, such as time limitation, location-based restrictions, and bias in requirements' elicitation process. Therefore, recommending related requirements assists stakeholders in becoming more aware of different aspects of the project. 

---
# SaviorRec: Semantic-Behavior Alignment for Cold-Start Recommendation 

**Authors**: Yining Yao, Ziwei Li, Shuwen Xiao, Boya Du, Jialin Zhu, Junjun Zheng, Xiangheng Kong, Yuning Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01375)  

**Abstract**: In recommendation systems, predicting Click-Through Rate (CTR) is crucial for accurately matching users with items. To improve recommendation performance for cold-start and long-tail items, recent studies focus on leveraging item multimodal features to model users' interests. However, obtaining multimodal representations for items relies on complex pre-trained encoders, which incurs unacceptable computation cost to train jointly with downstream ranking models. Therefore, it is important to maintain alignment between semantic and behavior space in a lightweight way.
To address these challenges, we propose a Semantic-Behavior Alignment for Cold-start Recommendation framework, which mainly focuses on utilizing multimodal representations that align with the user behavior space to predict CTR. First, we leverage domain-specific knowledge to train a multimodal encoder to generate behavior-aware semantic representations. Second, we use residual quantized semantic ID to dynamically bridge the gap between multimodal representations and the ranking model, facilitating the continuous semantic-behavior alignment. We conduct our offline and online experiments on the Taobao, one of the world's largest e-commerce platforms, and have achieved an increase of 0.83% in offline AUC, 13.21% clicks increase and 13.44% orders increase in the online A/B test, emphasizing the efficacy of our method. 

---
# A Study on Enhancing User Engagement by Employing Gamified Recommender Systems 

**Authors**: Ali Fallahi, Azam Bastanfard, Amineh Amini, Hadi Saboohi  

**Link**: [PDF](https://arxiv.org/pdf/2508.01265)  

**Abstract**: Providing customized products and services in the modern business world is one of the most efficient solutions to improve users' experience and their engagements with the industries. To aim, recommender systems, by producing personalized recommendations, have a crucial role in the digital age. As a consequence of modern improvements in the internet and online-based technologies, using gamification rules also increased in various fields. Recent studies showed that considering gamification concepts in implementing recommendation systems not only can become helpful to overcome the cold start and lack of sufficient data, moreover, can effectively improve user engagement. Gamification can motivate individuals to have more activities on the system; these interactions are valuable resources of data for recommender engines. Unlike the past related works about using gamified recommendation systems in different environments or studies that particularly surveyed gamification strategies or recommenders separately, this work provides a comprehensive review of how gamified recommender systems can enhance user engagement in various domain applications. Furthermore, comparing different approaches for building recommender systems is followed by in-depth surveying about investigating the gamified recommender systems, including their approaches, limitations, evaluation metrics, proposed achievements, datasets, domain areas, and their recommendation techniques. This exhaustive analysis provides a detailed picture of the topic's popularity, gaps, and unexplored regions. It is envisaged that the proposed research and introduced possible future directions would serve as a stepping stone for researchers interested in using gamified recommender systems for user satisfaction and engagement. 

---
# CM$^3$: Calibrating Multimodal Recommendation 

**Authors**: Xin Zhou, Yongjie Wang, Zhiqi Shen  

**Link**: [PDF](https://arxiv.org/pdf/2508.01226)  

**Abstract**: Alignment and uniformity are fundamental principles within the domain of contrastive learning. In recommender systems, prior work has established that optimizing the Bayesian Personalized Ranking (BPR) loss contributes to the objectives of alignment and uniformity. Specifically, alignment aims to draw together the representations of interacting users and items, while uniformity mandates a uniform distribution of user and item embeddings across a unit hypersphere. This study revisits the alignment and uniformity properties within the context of multimodal recommender systems, revealing a proclivity among extant models to prioritize uniformity to the detriment of alignment. Our hypothesis challenges the conventional assumption of equitable item treatment through a uniformity loss, proposing a more nuanced approach wherein items with similar multimodal attributes converge toward proximal representations within the hyperspheric manifold. Specifically, we leverage the inherent similarity between items' multimodal data to calibrate their uniformity distribution, thereby inducing a more pronounced repulsive force between dissimilar entities within the embedding space. A theoretical analysis elucidates the relationship between this calibrated uniformity loss and the conventional uniformity function. Moreover, to enhance the fusion of multimodal features, we introduce a Spherical Bézier method designed to integrate an arbitrary number of modalities while ensuring that the resulting fused features are constrained to the same hyperspherical manifold. Empirical evaluations conducted on five real-world datasets substantiate the superiority of our approach over competing baselines. We also shown that the proposed methods can achieve up to a 5.4% increase in NDCG@20 performance via the integration of MLLM-extracted features. Source code is available at: this https URL. 

---
# Towards Bridging Review Sparsity in Recommendation with Textual Edge Graph Representation 

**Authors**: Leyao Wang, Xutao Mao, Xuhui Zhan, Yuying Zhao, Bo Ni, Ryan A. Rossi, Nesreen K. Ahmed, Tyler Derr  

**Link**: [PDF](https://arxiv.org/pdf/2508.01128)  

**Abstract**: Textual reviews enrich recommender systems with fine-grained preference signals and enhanced explainability. However, in real-world scenarios, users rarely leave reviews, resulting in severe sparsity that undermines the effectiveness of existing models. A natural solution is to impute or generate missing reviews to enrich the data. However, conventional imputation techniques -- such as matrix completion and LLM-based augmentation -- either lose contextualized semantics by embedding texts into vectors, or overlook structural dependencies among user-item interactions. To address these shortcomings, we propose TWISTER (ToWards Imputation on Sparsity with Textual Edge Graph Representation), a unified framework that imputes missing reviews by jointly modeling semantic and structural signals. Specifically, we represent user-item interactions as a Textual-Edge Graph (TEG), treating reviews as edge attributes. To capture relational context, we construct line-graph views and employ a large language model as a graph-aware aggregator. For each interaction lacking a textual review, our model aggregates the neighborhood's natural-language representations to generate a coherent and personalized review. Experiments on the Amazon and Goodreads datasets show that TWISTER consistently outperforms traditional numeric, graph-based, and LLM baselines, delivering higher-quality imputed reviews and, more importantly, enhanced recommendation performance. In summary, TWISTER generates reviews that are more helpful, authentic, and specific, while smoothing structural signals for improved recommendations. 

---
# Addressing Cold Start For next-article Recommendation 

**Authors**: Omar Elgohary, Nathan Jorgenson, Trenton Marple  

**Link**: [PDF](https://arxiv.org/pdf/2508.01036)  

**Abstract**: This replication study modifies ALMM, the Adaptive Linear Mapping Model constructed for the next song recommendation, to the news recommendation problem on the MIND dataset. The original version of ALMM computes latent representations for users, last-time items, and current items in a tensor factorization structure and learns a linear mapping from content features to latent item vectors. Our replication aims to improve recommendation performance in cold-start scenarios by restructuring this model to sequential news click behavior, viewing consecutively read articles as (last news, next news) tuples. Instead of the original audio features, we apply BERT and a TF-IDF (Term Frequency-Inverse Document Frequency) to news titles and abstracts to extract token contextualized representations and align them with triplet-based user reading patterns. We also propose a reproducibly thorough pre-processing pipeline combining news filtering and feature integrity validation. Our implementation of ALMM with TF-IDF shows relatively improved recommendation accuracy and robustness over Forbes and Oord baseline models in the cold-start scenario. We demonstrate that ALMM in a minimally modified state is not suitable for next news recommendation. 

---
# TreeRanker: Fast and Model-agnostic Ranking System for Code Suggestions in IDEs 

**Authors**: Daniele Cipollone, Egor Bogomolov, Arie van Deursen, Maliheh Izadi  

**Link**: [PDF](https://arxiv.org/pdf/2508.02455)  

**Abstract**: Token-level code completion is one of the most critical features in modern Integrated Development Environments (IDEs). It assists developers by suggesting relevant identifiers and APIs during coding. While completions are typically derived from static analysis, their usefulness depends heavily on how they are ranked, as correct predictions buried deep in the list are rarely seen by users. Most current systems rely on hand-crafted heuristics or lightweight machine learning models trained on user logs, which can be further improved to capture context information and generalize across projects and coding styles. In this work, we propose a new scoring approach to ranking static completions using language models in a lightweight and model-agnostic way. Our method organizes all valid completions into a prefix tree and performs a single greedy decoding pass to collect token-level scores across the tree. This enables a precise token-aware ranking without needing beam search, prompt engineering, or model adaptations. The approach is fast, architecture-agnostic, and compatible with already deployed models for code completion. These findings highlight a practical and effective pathway for integrating language models into already existing tools within IDEs, and ultimately providing smarter and more responsive developer assistance. 

---
# Graph Embedding in the Graph Fractional Fourier Transform Domain 

**Authors**: Changjie Sheng, Zhichao Zhang, Wei Yao  

**Link**: [PDF](https://arxiv.org/pdf/2508.02383)  

**Abstract**: Spectral graph embedding plays a critical role in graph representation learning by generating low-dimensional vector representations from graph spectral information. However, the embedding space of traditional spectral embedding methods often exhibit limited expressiveness, failing to exhaustively capture latent structural features across alternative transform domains. To address this issue, we use the graph fractional Fourier transform to extend the existing state-of-the-art generalized frequency filtering embedding (GEFFE) into fractional domains, giving birth to the generalized fractional filtering embedding (GEFRFE), which enhances embedding informativeness via the graph fractional domain. The GEFRFE leverages graph fractional domain filtering and a nonlinear composition of eigenvector components derived from a fractionalized graph Laplacian. To dynamically determine the fractional order, two parallel strategies are introduced: search-based optimization and a ResNet18-based adaptive learning. Extensive experiments on six benchmark datasets demonstrate that the GEFRFE captures richer structural features and significantly enhance classification performance. Notably, the proposed method retains computational complexity comparable to GEFFE approaches. 

---
# Uni-Layout: Integrating Human Feedback in Unified Layout Generation and Evaluation 

**Authors**: Shuo Lu, Yanyin Chen, Wei Feng, Jiahao Fan, Fengheng Li, Zheng Zhang, Jingjing Lv, Junjie Shen, Ching Law, Jian Liang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02374)  

**Abstract**: Layout generation plays a crucial role in enhancing both user experience and design efficiency. However, current approaches suffer from task-specific generation capabilities and perceptually misaligned evaluation metrics, leading to limited applicability and ineffective measurement. In this paper, we propose \textit{Uni-Layout}, a novel framework that achieves unified generation, human-mimicking evaluation and alignment between the two. For universal generation, we incorporate various layout tasks into a single taxonomy and develop a unified generator that handles background or element contents constrained tasks via natural language prompts. To introduce human feedback for the effective evaluation of layouts, we build \textit{Layout-HF100k}, the first large-scale human feedback dataset with 100,000 expertly annotated layouts. Based on \textit{Layout-HF100k}, we introduce a human-mimicking evaluator that integrates visual and geometric information, employing a Chain-of-Thought mechanism to conduct qualitative assessments alongside a confidence estimation module to yield quantitative measurements. For better alignment between the generator and the evaluator, we integrate them into a cohesive system by adopting Dynamic-Margin Preference Optimization (DMPO), which dynamically adjusts margins based on preference strength to better align with human judgments. Extensive experiments show that \textit{Uni-Layout} significantly outperforms both task-specific and general-purpose methods. Our code is publicly available at this https URL. 

---
# Learning Partially-Decorrelated Common Spaces for Ad-hoc Video Search 

**Authors**: Fan Hu, Zijie Xin, Xirong Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.02340)  

**Abstract**: Ad-hoc Video Search (AVS) involves using a textual query to search for multiple relevant videos in a large collection of unlabeled short videos. The main challenge of AVS is the visual diversity of relevant videos. A simple query such as "Find shots of a man and a woman dancing together indoors" can span a multitude of environments, from brightly lit halls and shadowy bars to dance scenes in black-and-white animations. It is therefore essential to retrieve relevant videos as comprehensively as possible. Current solutions for the AVS task primarily fuse multiple features into one or more common spaces, yet overlook the need for diverse spaces. To fully exploit the expressive capability of individual features, we propose LPD, short for Learning Partially Decorrelated common spaces. LPD incorporates two key innovations: feature-specific common space construction and the de-correlation loss. Specifically, LPD learns a separate common space for each video and text feature, and employs de-correlation loss to diversify the ordering of negative samples across different spaces. To enhance the consistency of multi-space convergence, we designed an entropy-based fair multi-space triplet ranking loss. Extensive experiments on the TRECVID AVS benchmarks (2016-2023) justify the effectiveness of LPD. Moreover, diversity visualizations of LPD's spaces highlight its ability to enhance result diversity. 

---
# Understanding User Preferences for Interaction Styles in Conversational Recommender Systems: The Predictive Role of System Qualities, User Experience, and Traits 

**Authors**: Raj Mahmud, Shlomo Berkovsky, Mukesh Prasad, A. Baki Kocaballi  

**Link**: [PDF](https://arxiv.org/pdf/2508.02328)  

**Abstract**: Conversational Recommender Systems (CRSs) deliver personalised recommendations through multi-turn natural language dialogue and increasingly support both task-oriented and exploratory interactions. Yet, the factors shaping user interaction preferences remain underexplored. In this within-subjects study (\(N = 139\)), participants experienced two scripted CRS dialogues, rated their experiences, and indicated the importance of eight system qualities. Logistic regression revealed that preference for the exploratory interaction was predicted by enjoyment, usefulness, novelty, and conversational quality. Unexpectedly, perceived effectiveness was also associated with exploratory preference. Clustering uncovered five latent user profiles with distinct dialogue style preferences. Moderation analyses indicated that age, gender, and control preference significantly influenced these choices. These findings integrate affective, cognitive, and trait-level predictors into CRS user modelling and inform autonomy-sensitive, value-adaptive dialogue design. The proposed predictive and adaptive framework applies broadly to conversational AI systems seeking to align dynamically with evolving user needs. 

---
# I2CR: Intra- and Inter-modal Collaborative Reflections for Multimodal Entity Linking 

**Authors**: Ziyan Liu, Junwen Li, Kaiwen Li, Tong Ruan, Chao Wang, Xinyan He, Zongyu Wang, Xuezhi Cao, Jingping Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02243)  

**Abstract**: Multimodal entity linking plays a crucial role in a wide range of applications. Recent advances in large language model-based methods have become the dominant paradigm for this task, effectively leveraging both textual and visual modalities to enhance performance. Despite their success, these methods still face two challenges, including unnecessary incorporation of image data in certain scenarios and the reliance only on a one-time extraction of visual features, which can undermine their effectiveness and accuracy. To address these challenges, we propose a novel LLM-based framework for the multimodal entity linking task, called Intra- and Inter-modal Collaborative Reflections. This framework prioritizes leveraging text information to address the task. When text alone is insufficient to link the correct entity through intra- and inter-modality evaluations, it employs a multi-round iterative strategy that integrates key visual clues from various aspects of the image to support reasoning and enhance matching accuracy. Extensive experiments on three widely used public datasets demonstrate that our framework consistently outperforms current state-of-the-art methods in the task, achieving improvements of 3.2%, 5.1%, and 1.6%, respectively. Our code is available at this https URL. 

---
# Controllable and Stealthy Shilling Attacks via Dispersive Latent Diffusion 

**Authors**: Shutong Qiao, Wei Yuan, Junliang Yu, Tong Chen, Quoc Viet Hung Nguyen, Hongzhi Yin  

**Link**: [PDF](https://arxiv.org/pdf/2508.01987)  

**Abstract**: Recommender systems (RSs) are now fundamental to various online platforms, but their dependence on user-contributed data leaves them vulnerable to shilling attacks that can manipulate item rankings by injecting fake users. Although widely studied, most existing attack models fail to meet two critical objectives simultaneously: achieving strong adversarial promotion of target items while maintaining realistic behavior to evade detection. As a result, the true severity of shilling threats that manage to reconcile the two objectives remains underappreciated. To expose this overlooked vulnerability, we present DLDA, a diffusion-based attack framework that can generate highly effective yet indistinguishable fake users by enabling fine-grained control over target promotion. Specifically, DLDA operates in a pre-aligned collaborative embedding space, where it employs a conditional latent diffusion process to iteratively synthesize fake user profiles with precise target item control. To evade detection, DLDA introduces a dispersive regularization mechanism that promotes variability and realism in generated behavioral patterns. Extensive experiments on three real-world datasets and five popular RS models demonstrate that, compared to prior attacks, DLDA consistently achieves stronger item promotion while remaining harder to detect. These results highlight that modern RSs are more vulnerable than previously recognized, underscoring the urgent need for more robust defenses. 

---
# MaRGen: Multi-Agent LLM Approach for Self-Directed Market Research and Analysis 

**Authors**: Roman Koshkin, Pengyu Dai, Nozomi Fujikawa, Masahito Togami, Marco Visentini-Scarzanella  

**Link**: [PDF](https://arxiv.org/pdf/2508.01370)  

**Abstract**: We present an autonomous framework that leverages Large Language Models (LLMs) to automate end-to-end business analysis and market report generation. At its core, the system employs specialized agents - Researcher, Reviewer, Writer, and Retriever - that collaborate to analyze data and produce comprehensive reports. These agents learn from real professional consultants' presentation materials at Amazon through in-context learning to replicate professional analytical methodologies. The framework executes a multi-step process: querying databases, analyzing data, generating insights, creating visualizations, and composing market reports. We also introduce a novel LLM-based evaluation system for assessing report quality, which shows alignment with expert human evaluations. Building on these evaluations, we implement an iterative improvement mechanism that optimizes report quality through automated review cycles. Experimental results show that report quality can be improved by both automated review cycles and consultants' unstructured knowledge. In experimental validation, our framework generates detailed 6-page reports in 7 minutes at a cost of approximately \$1. Our work could be an important step to automatically create affordable market insights. 

---
# BioDisco: Multi-agent hypothesis generation with dual-mode evidence, iterative feedback and temporal evaluation 

**Authors**: Yujing Ke, Kevin George, Kathan Pandya, David Blumenthal, Maximilian Sprang, Gerrit Großmann, Sebastian Vollmer, David Antony Selby  

**Link**: [PDF](https://arxiv.org/pdf/2508.01285)  

**Abstract**: Identifying novel hypotheses is essential to scientific research, yet this process risks being overwhelmed by the sheer volume and complexity of available information. Existing automated methods often struggle to generate novel and evidence-grounded hypotheses, lack robust iterative refinement and rarely undergo rigorous temporal evaluation for future discovery potential. To address this, we propose BioDisco, a multi-agent framework that draws upon language model-based reasoning and a dual-mode evidence system (biomedical knowledge graphs and automated literature retrieval) for grounded novelty, integrates an internal scoring and feedback loop for iterative refinement, and validates performance through pioneering temporal and human evaluations and a Bradley-Terry paired comparison model to provide statistically-grounded assessment. Our evaluations demonstrate superior novelty and significance over ablated configurations representative of existing agentic architectures. Designed for flexibility and modularity, BioDisco allows seamless integration of custom language models or knowledge graphs, and can be run with just a few lines of code. We anticipate researchers using this practical tool as a catalyst for the discovery of new hypotheses. 

---
# DBAIOps: A Reasoning LLM-Enhanced Database Operation and Maintenance System using Knowledge Graphs 

**Authors**: Wei Zhou, Peng Sun, Xuanhe Zhou, Qianglei Zang, Ji Xu, Tieying Zhang, Guoliang Li, Fan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01136)  

**Abstract**: The operation and maintenance (O&M) of database systems is critical to ensuring system availability and performance, typically requiring expert experience (e.g., identifying metric-to-anomaly relations) for effective diagnosis and recovery. However, existing automatic database O&M methods, including commercial products, cannot effectively utilize expert experience. On the one hand, rule-based methods only support basic O&M tasks (e.g., metric-based anomaly detection), which are mostly numerical equations and cannot effectively incorporate literal O&M experience (e.g., troubleshooting guidance in manuals). On the other hand, LLM-based methods, which retrieve fragmented information (e.g., standard documents + RAG), often generate inaccurate or generic results. To address these limitations, we present DBAIOps, a novel hybrid database O&M system that combines reasoning LLMs with knowledge graphs to achieve DBA-style diagnosis. First, DBAIOps introduces a heterogeneous graph model for representing the diagnosis experience, and proposes a semi-automatic graph construction algorithm to build that graph from thousands of documents. Second, DBAIOps develops a collection of (800+) reusable anomaly models that identify both directly alerted metrics and implicitly correlated experience and metrics. Third, for each anomaly, DBAIOps proposes a two-stage graph evolution mechanism to explore relevant diagnosis paths and identify missing relations automatically. It then leverages a reasoning LLM (e.g., DeepSeek-R1) to infer root causes and generate clear diagnosis reports for both DBAs and common users. Our evaluation over four mainstream database systems (Oracle, MySQL, PostgreSQL, and DM8) demonstrates that DBAIOps outperforms state-of-the-art baselines, 34.85% and 47.22% higher in root cause and human evaluation accuracy, respectively. 

---
# Cross-Domain Web Information Extraction at Pinterest 

**Authors**: Michael Farag, Patrick Halina, Andrey Zaytsev, Alekhya Munagala, Imtihan Ahmed, Junhao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01096)  

**Abstract**: The internet offers a massive repository of unstructured information, but it's a significant challenge to convert this into a structured format. At Pinterest, the ability to accurately extract structured product data from e-commerce websites is essential to enhance user experiences and improve content distribution. In this paper, we present Pinterest's system for attribute extraction, which achieves remarkable accuracy and scalability at a manageable cost. Our approach leverages a novel webpage representation that combines structural, visual, and text modalities into a compact form, optimizing it for small model learning. This representation captures each visible HTML node with its text, style and layout information. We show how this allows simple models such as eXtreme Gradient Boosting (XGBoost) to extract attributes more accurately than much more complex Large Language Models (LLMs) such as Generative Pre-trained Transformer (GPT). Our results demonstrate a system that is highly scalable, processing over 1,000 URLs per second, while being 1000 times more cost-effective than the cheapest GPT alternatives. 

---
# MAO-ARAG: Multi-Agent Orchestration for Adaptive Retrieval-Augmented Generation 

**Authors**: Yiqun Chen, Erhan Zhang, Lingyong Yan, Shuaiqiang Wang, Jizhou Huang, Dawei Yin, Jiaxin Mao  

**Link**: [PDF](https://arxiv.org/pdf/2508.01005)  

**Abstract**: In question-answering (QA) systems, Retrieval-Augmented Generation (RAG) has become pivotal in enhancing response accuracy and reducing hallucination issues. The architecture of RAG systems varies significantly, encompassing single-round RAG, iterative RAG, and reasoning RAG, each tailored to address different types of queries. Due to the varying complexity of real-world queries, a fixed RAG pipeline often struggles to balance performance and cost efficiency across different queries. To address this challenge, we propose an adaptive RAG framework called MAO-ARAG, which leverages multi-agent orchestration. Our adaptive RAG is conceived as a multi-turn framework. Specifically, we define multiple executor agents, representing typical RAG modules such as query reformulation agents, document selection agent, and generation agents. A planner agent intelligently selects and integrates the appropriate agents from these executors into a suitable workflow tailored for each query, striving for high-quality answers while maintaining reasonable costs. During each turn, the planner agent is trained using reinforcement learning, guided by an outcome-based reward (F1 score) and a cost-based penalty, continuously improving answer quality while keeping costs within a reasonable range. Experiments conducted on multiple QA datasets demonstrate that our approach, which dynamically plans workflows for each query, not only achieves high answer quality but also maintains both cost and latency within acceptable this http URL code of MAO-ARAG is on this https URL. 

---
# Learning Unified User Quantized Tokenizers for User Representation 

**Authors**: Chuan He, Yang Chen, Wuliang Huang, Tianyi Zheng, Jianhu Chen, Bin Dou, Yice Luo, Yun Zhu, Baokun Wang, Yongchao Liu, Xing Fu, Yu Cheng, Chuntao Hong, Weiqiang Wang, Xin-Wei Yao  

**Link**: [PDF](https://arxiv.org/pdf/2508.00956)  

**Abstract**: Multi-source user representation learning plays a critical role in enabling personalized services on web platforms (e.g., Alipay). While prior works have adopted late-fusion strategies to combine heterogeneous data sources, they suffer from three key limitations: lack of unified representation frameworks, scalability and storage issues in data compression, and inflexible cross-task generalization. To address these challenges, we propose U^2QT (Unified User Quantized Tokenizers), a novel framework that integrates cross-domain knowledge transfer with early fusion of heterogeneous domains. Our framework employs a two-stage architecture: first, a causal Q-Former projects domain-specific features into a shared causal representation space to preserve inter-modality dependencies; second, a multi-view RQ-VAE discretizes causal embeddings into compact tokens through shared and source-specific codebooks, enabling efficient storage while maintaining semantic coherence. Experimental results showcase U^2QT's advantages across diverse downstream tasks, outperforming task-specific baselines in future behavior prediction and recommendation tasks while achieving efficiency gains in storage and computation. The unified tokenization framework enables seamless integration with language models and supports industrial-scale applications. 

---
# From Generator to Embedder: Harnessing Innate Abilities of Multimodal LLMs via Building Zero-Shot Discriminative Embedding Model 

**Authors**: Yeong-Joon Ju, Seong-Whan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.00955)  

**Abstract**: Multimodal Large Language Models (MLLMs) have emerged as a promising solution for universal embedding tasks, yet adapting their generative nature for discriminative representation learning remains a significant challenge. The dominant paradigm of large-scale contrastive pre-training suffers from critical inefficiencies, including prohibitive computational costs and a failure to leverage the intrinsic, instruction-following capabilities of MLLMs. To overcome these limitations, we propose an efficient framework for universal multimodal embeddings, which bridges this gap by centering on two synergistic components. First, our hierarchical embedding prompt template employs a two-level instruction architecture that forces the model to produce discriminative representations. Building on this strong foundation, our second component, self-aware hard negative sampling, redefines the fine-tuning process by leveraging the model's own understanding to efficiently mine challenging negatives while actively filtering out potential false negatives. Our comprehensive experiments show that our hierarchical prompt achieves zero-shot performance competitive with contrastively trained baselines and enhances the fine-tuning process by lifting a simple in-batch negative baseline by 4.8 points on the MMEB benchmark. We further boost the performance via our self-aware hard negative sampling, achieving the state-of-the-art performance without the contrative pre-training. Our work presents an effective and efficient pathway to adapt MLLMs for universal embedding tasks, significantly reducing training time. 

---
# Better Recommendations: Validating AI-generated Subject Terms Through LOC Linked Data Service 

**Authors**: Kwok Leong Tang, Yi Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.00867)  

**Abstract**: This article explores the integration of AI-generated subject terms into library cataloging, focusing on validation through the Library of Congress Linked Data Service. It examines the challenges of traditional subject cataloging under the Library of Congress Subject Headings system, including inefficiencies and cataloging backlogs. While generative AI shows promise in expediting cataloging workflows, studies reveal significant limitations in the accuracy of AI-assigned subject headings. The article proposes a hybrid approach combining AI technology with human validation through LOC Linked Data Service, aiming to enhance the precision, efficiency, and overall quality of metadata creation in library cataloging practices. 

---
# A Schema.org Mapping for Brazilian Legal Norms: Toward Interoperable Legal Graphs and Open Government Data 

**Authors**: Hudson de Martim  

**Link**: [PDF](https://arxiv.org/pdf/2508.00827)  

**Abstract**: Open Government Data (OGD) initiatives aim to enhance transparency and public participation by making government data openly accessible. However, structuring legal norms for machine readability remains a critical challenge for advancing Legal Tech applications such as Legal Knowledge Graphs (LKGs). Focusing on the this http URL portal initiative by the Brazilian National Congress, we propose a unified mapping of Brazilian legislation to the this http URL vocabulary via JSON-LD and Linked Data. Our approach covers both the conceptual "Norm" entity (mapped to sdo:Legislation) and its digital publications or manifestations (mapped to sdo:LegislationObject). We detail key properties for each type, providing concrete examples and considering URN identifiers (per the LexML standard), multilingual support, versioning in the Official Journal, and inter-norm relationships (e.g., citations and references). Our structured schema improves the quality and interoperability of Brazilian legal data, fosters integration within the global OGD ecosystem, and facilitates the creation of a wor 

---
