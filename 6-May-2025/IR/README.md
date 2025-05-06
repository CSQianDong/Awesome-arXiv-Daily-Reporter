# Predicting Movie Hits Before They Happen with LLMs 

**Authors**: Shaghayegh Agah, Yejin Kim, Neeraj Sharma, Mayur Nankani, Kevin Foley, H. Howie Huang, Sardar Hamidian  

**Link**: [PDF](https://arxiv.org/pdf/2505.02693)  

**Abstract**: Addressing the cold-start issue in content recommendation remains a critical ongoing challenge. In this work, we focus on tackling the cold-start problem for movies on a large entertainment platform. Our primary goal is to forecast the popularity of cold-start movies using Large Language Models (LLMs) leveraging movie metadata. This method could be integrated into retrieval systems within the personalization pipeline or could be adopted as a tool for editorial teams to ensure fair promotion of potentially overlooked movies that may be missed by traditional or algorithmic solutions. Our study validates the effectiveness of this approach compared to established baselines and those we developed. 

---
# Evaluating Contrastive Feedback for Effective User Simulations 

**Authors**: Andreas Konstantin Kruff, Timo Breuer, Philipp Schaer  

**Link**: [PDF](https://arxiv.org/pdf/2505.02560)  

**Abstract**: The use of Large Language Models (LLMs) for simulating user behavior in the domain of Interactive Information Retrieval has recently gained significant popularity. However, their application and capabilities remain highly debated and understudied. This study explores whether the underlying principles of contrastive training techniques, which have been effective for fine-tuning LLMs, can also be applied beneficially in the area of prompt engineering for user simulations.
Previous research has shown that LLMs possess comprehensive world knowledge, which can be leveraged to provide accurate estimates of relevant documents. This study attempts to simulate a knowledge state by enhancing the model with additional implicit contextual information gained during the simulation. This approach enables the model to refine the scope of desired documents further. The primary objective of this study is to analyze how different modalities of contextual information influence the effectiveness of user simulations.
Various user configurations were tested, where models are provided with summaries of already judged relevant, irrelevant, or both types of documents in a contrastive manner. The focus of this study is the assessment of the impact of the prompting techniques on the simulated user agent performance. We hereby lay the foundations for leveraging LLMs as part of more realistic simulated users. 

---
# Uncertainty in Repeated Implicit Feedback as a Measure of Reliability 

**Authors**: Bruno Sguerra, Viet-Anh Tran, Romain Hennequin, Manuel Moussallam  

**Link**: [PDF](https://arxiv.org/pdf/2505.02492)  

**Abstract**: Recommender systems rely heavily on user feedback to learn effective user and item representations. Despite their widespread adoption, limited attention has been given to the uncertainty inherent in the feedback used to train these systems. Both implicit and explicit feedback are prone to noise due to the variability in human interactions, with implicit feedback being particularly challenging. In collaborative filtering, the reliability of interaction signals is critical, as these signals determine user and item similarities. Thus, deriving accurate confidence measures from implicit feedback is essential for ensuring the reliability of these signals.
A common assumption in academia and industry is that repeated interactions indicate stronger user interest, increasing confidence in preference estimates. However, in domains such as music streaming, repeated consumption can shift user preferences over time due to factors like satiation and exposure. While literature on repeated consumption acknowledges these dynamics, they are often overlooked when deriving confidence scores for implicit feedback.
This paper addresses this gap by focusing on music streaming, where repeated interactions are frequent and quantifiable. We analyze how repetition patterns intersect with key factors influencing user interest and develop methods to quantify the associated uncertainty. These uncertainty measures are then integrated as consistency metrics in a recommendation task. Our empirical results show that incorporating uncertainty into user preference models yields more accurate and relevant recommendations. Key contributions include a comprehensive analysis of uncertainty in repeated consumption patterns, the release of a novel dataset, and a Bayesian model for implicit listening feedback. 

---
# Tevatron 2.0: Unified Document Retrieval Toolkit across Scale, Language, and Modality 

**Authors**: Xueguang Ma, Luyu Gao, Shengyao Zhuang, Jiaqi Samantha Zhan, Jamie Callan, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.02466)  

**Abstract**: Recent advancements in large language models (LLMs) have driven interest in billion-scale retrieval models with strong generalization across retrieval tasks and languages. Additionally, progress in large vision-language models has created new opportunities for multimodal retrieval. In response, we have updated the Tevatron toolkit, introducing a unified pipeline that enables researchers to explore retriever models at different scales, across multiple languages, and with various modalities. This demo paper highlights the toolkit's key features, bridging academia and industry by supporting efficient training, inference, and evaluation of neural retrievers. We showcase a unified dense retriever achieving strong multilingual and multimodal effectiveness, and conduct a cross-modality zero-shot study to demonstrate its research potential. Alongside, we release OmniEmbed, to the best of our knowledge, the first embedding model that unifies text, image document, video, and audio retrieval, serving as a baseline for future research. 

---
# SymbioticRAG: Enhancing Document Intelligence Through Human-LLM Symbiotic Collaboration 

**Authors**: Qiang Sun, Tingting Bi, Sirui Li, Eun-Jung Holden, Paul Duuring, Kai Niu, Wei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.02418)  

**Abstract**: We present \textbf{SymbioticRAG}, a novel framework that fundamentally reimagines Retrieval-Augmented Generation~(RAG) systems by establishing a bidirectional learning relationship between humans and machines. Our approach addresses two critical challenges in current RAG systems: the inherently human-centered nature of relevance determination and users' progression from "unconscious incompetence" in query formulation. SymbioticRAG introduces a two-tier solution where Level 1 enables direct human curation of retrieved content through interactive source document exploration, while Level 2 aims to build personalized retrieval models based on captured user interactions. We implement Level 1 through three key components: (1)~a comprehensive document processing pipeline with specialized models for layout detection, OCR, and extraction of tables, formulas, and figures; (2)~an extensible retriever module supporting multiple retrieval strategies; and (3)~an interactive interface that facilitates both user engagement and interaction data logging. We experiment Level 2 implementation via a retriever strategy incorporated LLM summarized user intention from user interaction logs. To maintain high-quality data preparation, we develop a human-on-the-loop validation interface that improves pipeline output while advancing research in specialized extraction tasks. Evaluation across three scenarios (literature review, geological exploration, and education) demonstrates significant improvements in retrieval relevance and user satisfaction compared to traditional RAG approaches. To facilitate broader research and further advancement of SymbioticRAG Level 2 implementation, we will make our system openly accessible to the research community. 

---
# Social Biases in Knowledge Representations of Wikidata separates Global North from Global South 

**Authors**: Paramita Das, Sai Keerthana Karnam, Aditya Soni, Animesh Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2505.02352)  

**Abstract**: Knowledge Graphs have become increasingly popular due to their wide usage in various downstream applications, including information retrieval, chatbot development, language model construction, and many others. Link prediction (LP) is a crucial downstream task for knowledge graphs, as it helps to address the problem of the incompleteness of the knowledge graphs. However, previous research has shown that knowledge graphs, often created in a (semi) automatic manner, are not free from social biases. These biases can have harmful effects on downstream applications, especially by leading to unfair behavior toward minority groups. To understand this issue in detail, we develop a framework -- AuditLP -- deploying fairness metrics to identify biased outcomes in LP, specifically how occupations are classified as either male or female-dominated based on gender as a sensitive attribute. We have experimented with the sensitive attribute of age and observed that occupations are categorized as young-biased, old-biased, and age-neutral. We conduct our experiments on a large number of knowledge triples that belong to 21 different geographies extracted from the open-sourced knowledge graph, Wikidata. Our study shows that the variance in the biased outcomes across geographies neatly mirrors the socio-economic and cultural division of the world, resulting in a transparent partition of the Global North from the Global South. 

---
# Minimally Supervised Hierarchical Domain Intent Learning for CRS 

**Authors**: Safikureshi Mondal, Subhasis Dasgupta, Amarnath Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2505.02209)  

**Abstract**: Modeling domain intent within an evolving domain structure presents a significant challenge for domain-specific conversational recommendation systems (CRS). The conventional approach involves training an intent model using utterance-intent pairs. However, as new intents and patterns emerge, the model must be continuously updated while preserving existing relationships and maintaining efficient retrieval. This process leads to substantial growth in utterance-intent pairs, making manual labeling increasingly costly and impractical. In this paper, we propose an efficient solution for constructing a dynamic hierarchical structure that minimizes the number of user utterances required to achieve adequate domain knowledge coverage. To this end, we introduce a neural network-based attention-driven hierarchical clustering algorithm designed to optimize intent grouping using minimal data. The proposed method builds upon and integrates concepts from two existing flat clustering algorithms DEC and NAM, both of which utilize neural attention mechanisms. We apply our approach to a curated subset of 44,000 questions from the business food domain. Experimental results demonstrate that constructing the hierarchy using a stratified sampling strategy significantly reduces the number of questions needed to represent the evolving intent structure. Our findings indicate that this approach enables efficient coverage of dynamic domain knowledge without frequent retraining, thereby enhancing scalability and adaptability in domain-specific CSRs. 

---
# Exploring new Approaches for Information Retrieval through Natural Language Processing 

**Authors**: Manak Raj, Nidhi Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2505.02199)  

**Abstract**: This review paper explores recent advancements and emerging approaches in Information Retrieval (IR) applied to Natural Language Processing (NLP). We examine traditional IR models such as Boolean, vector space, probabilistic, and inference network models, and highlight modern techniques including deep learning, reinforcement learning, and pretrained transformer models like BERT. We discuss key tools and libraries - Lucene, Anserini, and Pyserini - for efficient text indexing and search. A comparative analysis of sparse, dense, and hybrid retrieval methods is presented, along with applications in web search engines, cross-language IR, argument mining, private information retrieval, and hate speech detection. Finally, we identify open challenges and future research directions to enhance retrieval accuracy, scalability, and ethical considerations. 

---
# Interpreting Multilingual and Document-Length Sensitive Relevance Computations in Neural Retrieval Models through Axiomatic Causal Interventions 

**Authors**: Oliver Savolainen, Dur e Najaf Amjad, Roxana Petcu  

**Link**: [PDF](https://arxiv.org/pdf/2505.02154)  

**Abstract**: This reproducibility study analyzes and extends the paper "Axiomatic Causal Interventions for Reverse Engineering Relevance Computation in Neural Retrieval Models," which investigates how neural retrieval models encode task-relevant properties such as term frequency. We reproduce key experiments from the original paper, confirming that information on query terms is captured in the model encoding. We extend this work by applying activation patching to Spanish and Chinese datasets and by exploring whether document-length information is encoded in the model as well. Our results confirm that the designed activation patching method can isolate the behavior to specific components and tokens in neural retrieval models. Moreover, our findings indicate that the location of term frequency generalizes across languages and that in later layers, the information for sequence-level tasks is represented in the CLS token. The results highlight the need for further research into interpretability in information retrieval and reproducibility in machine learning research. Our code is available at this https URL. 

---
# Tricolore: Multi-Behavior User Profiling for Enhanced Candidate Generation in Recommender Systems 

**Authors**: Xiao Zhou, Zhongxiang Zhao, Hanze Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.02120)  

**Abstract**: Online platforms aggregate extensive user feedback across diverse behaviors, providing a rich source for enhancing user engagement. Traditional recommender systems, however, typically optimize for a single target behavior and represent user preferences with a single vector, limiting their ability to handle multiple important behaviors or optimization objectives. This conventional approach also struggles to capture the full spectrum of user interests, resulting in a narrow item pool during candidate generation. To address these limitations, we present Tricolore, a versatile multi-vector learning framework that uncovers connections between different behavior types for more robust candidate generation. Tricolore's adaptive multi-task structure is also customizable to specific platform needs. To manage the variability in sparsity across behavior types, we incorporate a behavior-wise multi-view fusion module that dynamically enhances learning. Moreover, a popularity-balanced strategy ensures the recommendation list balances accuracy with item popularity, fostering diversity and improving overall performance. Extensive experiments on public datasets demonstrate Tricolore's effectiveness across various recommendation scenarios, from short video platforms to e-commerce. By leveraging a shared base embedding strategy, Tricolore also significantly improves the performance for cold-start users. The source code is publicly available at: this https URL. 

---
# Embedding based retrieval for long tail search queries in ecommerce 

**Authors**: Akshay Kekuda, Yuyang Zhang, Arun Udayashankar  

**Link**: [PDF](https://arxiv.org/pdf/2505.01946)  

**Abstract**: In this abstract we present a series of optimizations we performed on the two-tower model architecture [14], training and evaluation datasets to implement semantic product search at Best Buy. Search queries on this http URL follow the pareto distribution whereby a minority of them account for most searches. This leaves us with a long tail of search queries that have low frequency of issuance. The queries in the long tail suffer from very spare interaction signals. Our current work focuses on building a model to serve the long tail queries. We present a series of optimizations we have done to this model to maximize conversion for the purpose of retrieval from the catalog. The first optimization we present is using a large language model to improve the sparsity of conversion signals. The second optimization is pretraining an off-the-shelf transformer-based model on the Best Buy catalog data. The third optimization we present is on the finetuning front. We use query-to-query pairs in addition to query-to-product pairs and combining the above strategies for finetuning the model. We also demonstrate how merging the weights of these finetuned models improves the evaluation metrics. Finally, we provide a recipe for curating an evaluation dataset for continuous monitoring of model performance with human-in-the-loop evaluation. We found that adding this recall mechanism to our current term match-based recall improved conversion by 3% in an online A/B test. 

---
# A Generalised and Adaptable Reinforcement Learning Stopping Method 

**Authors**: Reem Bin-Hezam, Mark Stevenson  

**Link**: [PDF](https://arxiv.org/pdf/2505.01907)  

**Abstract**: This paper presents a Technology Assisted Review (TAR) stopping approach based on Reinforcement Learning (RL). Previous such approaches offered limited control over stopping behaviour, such as fixing the target recall and tradeoff between preferring to maximise recall or cost. These limitations are overcome by introducing a novel RL environment, GRLStop, that allows a single model to be applied to multiple target recalls, balances the recall/cost tradeoff and integrates a classifier. Experiments were carried out on six benchmark datasets (CLEF e-Health datasets 2017-9, TREC Total Recall, TREC Legal and Reuters RCV1) at multiple target recall levels. Results showed that the proposed approach to be effective compared to multiple baselines in addition to offering greater flexibility. 

---
# Exploring the Role of Diversity in Example Selection for In-Context Learning 

**Authors**: Janak Kapuriya, Manit Kaushik, Debasis Ganguly, Sumit Bhatia  

**Link**: [PDF](https://arxiv.org/pdf/2505.01842)  

**Abstract**: In-Context Learning (ICL) has gained prominence due to its ability to perform tasks without requiring extensive training data and its robustness to noisy labels. A typical ICL workflow involves selecting localized examples relevant to a given input using sparse or dense embedding-based similarity functions. However, relying solely on similarity-based selection may introduce topical biases in the retrieved contexts, potentially leading to suboptimal downstream performance. We posit that reranking the retrieved context to enhance topical diversity can improve downstream task performance. To achieve this, we leverage maximum marginal relevance (MMR) which balances topical similarity with inter-example diversity. Our experimental results demonstrate that diversifying the selected examples leads to consistent improvements in downstream performance across various context sizes and similarity functions. The implementation of our approach is made available at this https URL. 

---
# SimAug: Enhancing Recommendation with Pretrained Language Models for Dense and Balanced Data Augmentation 

**Authors**: Yuying Zhao, Xiaodong Yang, Huiyuan Chen, Xiran Fan, Yu Wang, Yiwei Cai, Tyler Derr  

**Link**: [PDF](https://arxiv.org/pdf/2505.01695)  

**Abstract**: Deep Neural Networks (DNNs) are extensively used in collaborative filtering due to their impressive effectiveness. These systems depend on interaction data to learn user and item embeddings that are crucial for recommendations. However, the data often suffers from sparsity and imbalance issues: limited observations of user-item interactions can result in sub-optimal performance, and a predominance of interactions with popular items may introduce recommendation bias. To address these challenges, we employ Pretrained Language Models (PLMs) to enhance the interaction data with textual information, leading to a denser and more balanced dataset. Specifically, we propose a simple yet effective data augmentation method (SimAug) based on the textual similarity from PLMs, which can be seamlessly integrated to any systems as a lightweight, plug-and-play component in the pre-processing stage. Our experiments across nine datasets consistently demonstrate improvements in both utility and fairness when training with the augmented data generated by SimAug. The code is available at this https URL. 

---
# RAGAR: Retrieval Augment Personalized Image Generation Guided by Recommendation 

**Authors**: Run Ling, Wenji Wang, Yuting Liu, Guibing Guo, Linying Jiang, Xingwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.01657)  

**Abstract**: Personalized image generation is crucial for improving the user experience, as it renders reference images into preferred ones according to user visual preferences. Although effective, existing methods face two main issues. First, existing methods treat all items in the user historical sequence equally when extracting user preferences, overlooking the varying semantic similarities between historical items and the reference item. Disproportionately high weights for low-similarity items distort users' visual preferences for the reference item. Second, existing methods heavily rely on consistency between generated and reference images to optimize the generation, which leads to underfitting user preferences and hinders personalization. To address these issues, we propose Retrieval Augment Personalized Image GenerAtion guided by Recommendation (RAGAR). Our approach uses a retrieval mechanism to assign different weights to historical items according to their similarities to the reference item, thereby extracting more refined users' visual preferences for the reference item. Then we introduce a novel rank task based on the multi-modal ranking model to optimize the personalization of the generated images instead of forcing depend on consistency. Extensive experiments and human evaluations on three real-world datasets demonstrate that RAGAR achieves significant improvements in both personalization and semantic metrics compared to five baselines. 

---
# A Multi-Granularity Multimodal Retrieval Framework for Multimodal Document Tasks 

**Authors**: Mingjun Xu, Zehui Wang, Hengxing Cai, Renxin Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2505.01457)  

**Abstract**: Retrieval-augmented generation (RAG) systems have predominantly focused on text-based retrieval, limiting their effectiveness in handling visually-rich documents that encompass text, images, tables, and charts. To bridge this gap, we propose a unified multi-granularity multimodal retrieval framework tailored for two benchmark tasks: MMDocIR and M2KR. Our approach integrates hierarchical encoding strategies, modality-aware retrieval mechanisms, and reranking modules to effectively capture and utilize the complex interdependencies between textual and visual modalities. By leveraging off-the-shelf vision-language models and implementing a training-free hybridretrieval strategy, our framework demonstrates robust performance without the need for task-specific fine-tuning. Experimental evaluations reveal that incorporating layout-aware search and reranking modules significantly enhances retrieval accuracy, achieving a top performance score of 65.56. This work underscores the potential of scalable and reproducible solutions in advancing multimodal document retrieval systems. 

---
# Effective Inference-Free Retrieval for Learned Sparse Representations 

**Authors**: Franco Maria Nardini, Thong Nguyen, Cosimo Rulli, Rossano Venturini, Andrew Yates  

**Link**: [PDF](https://arxiv.org/pdf/2505.01452)  

**Abstract**: Learned Sparse Retrieval (LSR) is an effective IR approach that exploits pre-trained language models for encoding text into a learned bag of words. Several efforts in the literature have shown that sparsity is key to enabling a good trade-off between the efficiency and effectiveness of the query processor. To induce the right degree of sparsity, researchers typically use regularization techniques when training LSR models. Recently, new efficient -- inverted index-based -- retrieval engines have been proposed, leading to a natural question: has the role of regularization changed in training LSR models? In this paper, we conduct an extended evaluation of regularization approaches for LSR where we discuss their effectiveness, efficiency, and out-of-domain generalization capabilities. We first show that regularization can be relaxed to produce more effective LSR encoders. We also show that query encoding is now the bottleneck limiting the overall query processor performance. To remove this bottleneck, we advance the state-of-the-art of inference-free LSR by proposing Learned Inference-free Retrieval (Li-LSR). At training time, Li-LSR learns a score for each token, casting the query encoding step into a seamless table lookup. Our approach yields state-of-the-art effectiveness for both in-domain and out-of-domain evaluation, surpassing Splade-v3-Doc by 1 point of mRR@10 on MS MARCO and 1.8 points of nDCG@10 on BEIR. 

---
# AdSight: Scalable and Accurate Quantification of User Attention in Multi-Slot Sponsored Search 

**Authors**: Mario Villaizán-Vallelado, Matteo Salvatori, Kayhan Latifzadeh, Antonio Penta, Luis A. Leiva, Ioannis Arapakis  

**Link**: [PDF](https://arxiv.org/pdf/2505.01451)  

**Abstract**: Modern Search Engine Results Pages (SERPs) present complex layouts where multiple elements compete for visibility. Attention modelling is crucial for optimising web design and computational advertising, whereas attention metrics can inform ad placement and revenue strategies. We introduce AdSight, a method leveraging mouse cursor trajectories to quantify in a scalable and accurate manner user attention in multi-slot environments like SERPs. AdSight uses a novel Transformer-based sequence-to-sequence architecture where the encoder processes cursor trajectory embeddings, and the decoder incorporates slot-specific features, enabling robust attention prediction across various SERP layouts. We evaluate our approach on two Machine Learning tasks: (1)~\emph{regression}, to predict fixation times and counts; and (2)~\emph{classification}, to determine some slot types were noticed. Our findings demonstrate the model's ability to predict attention with unprecedented precision, offering actionable insights for researchers and practitioners. 

---
# LLM-Enabled EV Charging Stations Recommendation 

**Authors**: Zeinab Teimoori  

**Link**: [PDF](https://arxiv.org/pdf/2505.01447)  

**Abstract**: Charging infrastructure is not expanding quickly enough to accommodate the increasing usage of Electric Vehicles (EVs). For this reason, EV owners experience extended waiting periods, range anxiety, and overall dissatisfaction. Challenges, such as fragmented data and the complexity of integrating factors like location, energy pricing, and user preferences, make the current recommendation systems ineffective. To overcome these limitations, we propose RecomBot, which is a Large Language Model (LLM)-powered prompt-based recommender system that dynamically suggests optimal Charging Stations (CSs) using real-time heterogeneous data. By leveraging natural language reasoning and fine-tuning EV-specific datasets, RecomBot enhances personalization, improves charging efficiency, and adapts to various EV types, offering a scalable solution for intelligent EV recommendation systems. Through testing across various prompt engineering scenarios, the results obtained underline the capability and efficiency of the proposed model. 

---
# Algorithm Performance Spaces for Strategic Dataset Selection 

**Authors**: Steffen Schulz  

**Link**: [PDF](https://arxiv.org/pdf/2505.01442)  

**Abstract**: The evaluation of new algorithms in recommender systems frequently depends on publicly available datasets, such as those from MovieLens or Amazon. Some of these datasets are being disproportionately utilized primarily due to their historical popularity as baselines rather than their suitability for specific research contexts. This thesis addresses this issue by introducing the Algorithm Performance Space, a novel framework designed to differentiate datasets based on the measured performance of algorithms applied to them. An experimental study proposes three metrics to quantify and justify dataset selection to evaluate new algorithms. These metrics also validate assumptions about datasets, such as the similarity between MovieLens datasets of varying sizes. By creating an Algorithm Performance Space and using the proposed metrics, differentiating datasets was made possible, and diverse dataset selections could be found. While the results demonstrate the framework's potential, further research proposals and implications are discussed to develop Algorithm Performance Spaces tailored to diverse use cases. 

---
# AdaParse: An Adaptive Parallel PDF Parsing and Resource Scaling Engine 

**Authors**: Carlo Siebenschuh, Kyle Hippe, Ozan Gokdemir, Alexander Brace, Arham Khan, Khalid Hossain, Yadu Babuji, Nicholas Chia, Venkatram Vishwanath, Rick Stevens, Arvind Ramanathan, Ian Foster, Robert Underwood  

**Link**: [PDF](https://arxiv.org/pdf/2505.01435)  

**Abstract**: Language models for scientific tasks are trained on text from scientific publications, most distributed as PDFs that require parsing. PDF parsing approaches range from inexpensive heuristics (for simple documents) to computationally intensive ML-driven systems (for complex or degraded ones). The choice of the "best" parser for a particular document depends on its computational cost and the accuracy of its output. To address these issues, we introduce an Adaptive Parallel PDF Parsing and Resource Scaling Engine (AdaParse), a data-driven strategy for assigning an appropriate parser to each document. We enlist scientists to select preferred parser outputs and incorporate this information through direct preference optimization (DPO) into AdaParse, thereby aligning its selection process with human judgment. AdaParse then incorporates hardware requirements and predicted accuracy of each parser to orchestrate computational resources efficiently for large-scale parsing campaigns. We demonstrate that AdaParse, when compared to state-of-the-art parsers, improves throughput by $17\times$ while still achieving comparable accuracy (0.2 percent better) on a benchmark set of 1000 scientific documents. AdaParse's combination of high accuracy and parallel scalability makes it feasible to parse large-scale scientific document corpora to support the development of high-quality, trillion-token-scale text datasets. The implementation is available at this https URL 

---
# Knowing You Don't Know: Learning When to Continue Search in Multi-round RAG through Self-Practicing 

**Authors**: Diji Yang, Linda Zeng, Jinmeng Rao, Yi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.02811)  

**Abstract**: Retrieval Augmented Generation (RAG) has shown strong capability in enhancing language models' knowledge and reducing AI generative hallucinations, driving its widespread use. However, complex tasks requiring multi-round retrieval remain challenging, and early attempts tend to be overly optimistic without a good sense of self-skepticism. Current multi-round RAG systems may continue searching even when enough information has already been retrieved, or they may provide incorrect answers without having sufficient information or knowledge. Existing solutions either require large amounts of expensive human-labeled process supervision data or lead to subpar performance.
This paper aims to address these limitations by introducing a new framework, \textbf{SIM-RAG}, to explicitly enhance RAG systems' self-awareness and multi-round retrieval capabilities. To train SIM-RAG, we first let a RAG system self-practice multi-round retrieval, augmenting existing question-answer pairs with intermediate inner monologue reasoning steps to generate synthetic training data. For each pair, the system may explore multiple retrieval paths, which are labeled as successful if they reach the correct answer and unsuccessful otherwise. Using this data, we train a lightweight information sufficiency Critic. At inference time, the Critic evaluates whether the RAG system has retrieved sufficient information at each round, guiding retrieval decisions and improving system-level self-awareness through in-context reinforcement learning.
Experiments across multiple prominent RAG benchmarks show that SIM-RAG is an effective multi-round RAG solution. Furthermore, this framework is system-efficient, adding a lightweight component to RAG without requiring modifications to existing LLMs or search engines, and data-efficient, eliminating the need for costly human-annotated mid-step retrieval process supervision data. 

---
# Using Knowledge Graphs to harvest datasets for efficient CLIP model training 

**Authors**: Simon Ging, Sebastian Walter, Jelena Bratulić, Johannes Dienert, Hannah Bast, Thomas Brox  

**Link**: [PDF](https://arxiv.org/pdf/2505.02746)  

**Abstract**: Training high-quality CLIP models typically requires enormous datasets, which limits the development of domain-specific models -- especially in areas that even the largest CLIP models do not cover well -- and drives up training costs. This poses challenges for scientific research that needs fine-grained control over the training procedure of CLIP models. In this work, we show that by employing smart web search strategies enhanced with knowledge graphs, a robust CLIP model can be trained from scratch with considerably less data. Specifically, we demonstrate that an expert foundation model for living organisms can be built using just 10M images. Moreover, we introduce EntityNet, a dataset comprising 33M images paired with 46M text descriptions, which enables the training of a generic CLIP model in significantly reduced time. 

---
# Unraveling Media Perspectives: A Comprehensive Methodology Combining Large Language Models, Topic Modeling, Sentiment Analysis, and Ontology Learning to Analyse Media Bias 

**Authors**: Orlando Jähde, Thorsten Weber, Rüdiger Buchkremer  

**Link**: [PDF](https://arxiv.org/pdf/2505.01754)  

**Abstract**: Biased news reporting poses a significant threat to informed decision-making and the functioning of democracies. This study introduces a novel methodology for scalable, minimally biased analysis of media bias in political news. The proposed approach examines event selection, labeling, word choice, and commission and omission biases across news sources by leveraging natural language processing techniques, including hierarchical topic modeling, sentiment analysis, and ontology learning with large language models. Through three case studies related to current political events, we demonstrate the methodology's effectiveness in identifying biases across news sources at various levels of granularity. This work represents a significant step towards scalable, minimally biased media bias analysis, laying the groundwork for tools to help news consumers navigate an increasingly complex media landscape. 

---
# HoneyBee: Efficient Role-based Access Control for Vector Databases via Dynamic Partitioning 

**Authors**: Hongbin Zhong, Matthew Lentz, Nina Narodytska, Adriana Szekeres, Kexin Rong  

**Link**: [PDF](https://arxiv.org/pdf/2505.01538)  

**Abstract**: As vector databases gain traction in enterprise applications, robust access control has become critical to safeguard sensitive data. Access control in these systems is often implemented through hybrid vector queries, which combine nearest neighbor search on vector data with relational predicates based on user permissions. However, existing approaches face significant trade-offs: creating dedicated indexes for each user minimizes query latency but introduces excessive storage redundancy, while building a single index and applying access control after vector search reduces storage overhead but suffers from poor recall and increased query latency. This paper introduces HoneyBee, a dynamic partitioning framework that bridges the gap between these approaches by leveraging the structure of Role-Based Access Control (RBAC) policies. RBAC, widely adopted in enterprise settings, groups users into roles and assigns permissions to those roles, creating a natural "thin waist" in the permission structure that is ideal for partitioning decisions. Specifically, HoneyBee produces overlapping partitions where vectors can be strategically replicated across different partitions to reduce query latency while controlling storage overhead. By introducing analytical models for the performance and recall of the vector search, HoneyBee formulates the partitioning strategy as a constrained optimization problem to dynamically balance storage, query efficiency, and recall. Evaluations on RBAC workloads demonstrate that HoneyBee reduces storage redundancy compared to role partitioning and achieves up to 6x faster query speeds than row-level security (RLS) with only 1.4x storage increase, offering a practical middle ground for secure and efficient vector search. 

---
