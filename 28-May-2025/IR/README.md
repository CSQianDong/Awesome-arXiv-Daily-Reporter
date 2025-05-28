# Something's Fishy In The Data Lake: A Critical Re-evaluation of Table Union Search Benchmarks 

**Authors**: Allaa Boutaleb, Bernd Amann, Hubert Naacke, Rafael Angarita  

**Link**: [PDF](https://arxiv.org/pdf/2505.21329)  

**Abstract**: Recent table representation learning and data discovery methods tackle table union search (TUS) within data lakes, which involves identifying tables that can be unioned with a given query table to enrich its content. These methods are commonly evaluated using benchmarks that aim to assess semantic understanding in real-world TUS tasks. However, our analysis of prominent TUS benchmarks reveals several limitations that allow simple baselines to perform surprisingly well, often outperforming more sophisticated approaches. This suggests that current benchmark scores are heavily influenced by dataset-specific characteristics and fail to effectively isolate the gains from semantic understanding. To address this, we propose essential criteria for future benchmarks to enable a more realistic and reliable evaluation of progress in semantic table union search. 

---
# Counterfactual Multi-player Bandits for Explainable Recommendation Diversification 

**Authors**: Yansen Zhang, Bowei He, Xiaokun Zhang, Haolun Wu, Zexu Sun, Chen Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.21165)  

**Abstract**: Existing recommender systems tend to prioritize items closely aligned with users' historical interactions, inevitably trapping users in the dilemma of ``filter bubble''. Recent efforts are dedicated to improving the diversity of recommendations. However, they mainly suffer from two major issues: 1) a lack of explainability, making it difficult for the system designers to understand how diverse recommendations are generated, and 2) limitations to specific metrics, with difficulty in enhancing non-differentiable diversity metrics. To this end, we propose a \textbf{C}ounterfactual \textbf{M}ulti-player \textbf{B}andits (CMB) method to deliver explainable recommendation diversification across a wide range of diversity metrics. Leveraging a counterfactual framework, our method identifies the factors influencing diversity outcomes. Meanwhile, we adopt the multi-player bandits to optimize the counterfactual optimization objective, making it adaptable to both differentiable and non-differentiable diversity metrics. Extensive experiments conducted on three real-world datasets demonstrate the applicability, effectiveness, and explainability of the proposed CMB. 

---
# Disentangling Locality and Entropy in Ranking Distillation 

**Authors**: Andrew Parry, Debasis Ganguly, Sean MacAvaney  

**Link**: [PDF](https://arxiv.org/pdf/2505.21058)  

**Abstract**: The training process of ranking models involves two key data selection decisions: a sampling strategy, and a labeling strategy. Modern ranking systems, especially those for performing semantic search, typically use a ``hard negative'' sampling strategy to identify challenging items using heuristics and a distillation labeling strategy to transfer ranking "knowledge" from a more capable model. In practice, these approaches have grown increasingly expensive and complex, for instance, popular pretrained rankers from SentenceTransformers involve 12 models in an ensemble with data provenance hampering reproducibility. Despite their complexity, modern sampling and labeling strategies have not been fully ablated, leaving the underlying source of effectiveness gains unclear. Thus, to better understand why models improve and potentially reduce the expense of training effective models, we conduct a broad ablation of sampling and distillation processes in neural ranking. We frame and theoretically derive the orthogonal nature of model geometry affected by example selection and the effect of teacher ranking entropy on ranking model optimization, establishing conditions in which data augmentation can effectively improve bias in a ranking model. Empirically, our investigation on established benchmarks and common architectures shows that sampling processes that were once highly effective in contrastive objectives may be spurious or harmful under distillation. We further investigate how data augmentation, in terms of inputs and targets, can affect effectiveness and the intrinsic behavior of models in ranking. Through this work, we aim to encourage more computationally efficient approaches that reduce focus on contrastive pairs and instead directly understand training dynamics under rankings, which better represent real-world settings. 

---
# A Reduction-Driven Local Search for the Generalized Independent Set Problem 

**Authors**: Yiping Liu, Yi Zhou, Zhenxiang Xu, Mingyu Xiao, Jin-Kao Hao  

**Link**: [PDF](https://arxiv.org/pdf/2505.21052)  

**Abstract**: The Generalized Independent Set (GIS) problem extends the classical maximum independent set problem by incorporating profits for vertices and penalties for edges. This generalized problem has been identified in diverse applications in fields such as forest harvest planning, competitive facility location, social network analysis, and even machine learning. However, solving the GIS problem in large-scale, real-world networks remains computationally challenging. In this paper, we explore data reduction techniques to address this challenge. We first propose 14 reduction rules that can reduce the input graph with rigorous optimality guarantees. We then present a reduction-driven local search (RLS) algorithm that integrates these reduction rules into the pre-processing, the initial solution generation, and the local search components in a computationally efficient way. The RLS is empirically evaluated on 278 graphs arising from different application scenarios. The results indicates that the RLS is highly competitive -- For most graphs, it achieves significantly superior solutions compared to other known solvers, and it effectively provides solutions for graphs exceeding 260 million edges, a task at which every other known method fails. Analysis also reveals that the data reduction plays a key role in achieving such a competitive performance. 

---
# LifeIR at the NTCIR-18 Lifelog-6 Task 

**Authors**: Jiahan Chen, Da Li, Keping Bi  

**Link**: [PDF](https://arxiv.org/pdf/2505.20987)  

**Abstract**: In recent years, sharing lifelogs recorded through wearable devices such as sports watches and GoPros, has gained significant popularity. Lifelogs involve various types of information, including images, videos, and GPS data, revealing users' lifestyles, dietary patterns, and physical activities. The Lifelog Semantic Access Task(LSAT) in the NTCIR-18 Lifelog-6 Challenge focuses on retrieving relevant images from a large scale of users' lifelogs based on textual queries describing an action or event. It serves users' need to find images about a scenario in the historical moments of their lifelogs. We propose a multi-stage pipeline for this task of searching images with texts, addressing various challenges in lifelog retrieval. Our pipeline includes: filtering blurred images, rewriting queries to make intents clearer, extending the candidate set based on events to include images with temporal connections, and reranking results using a multimodal large language model(MLLM) with stronger relevance judgment capabilities. The evaluation results of our submissions have shown the effectiveness of each stage and the entire pipeline. 

---
# Embed Progressive Implicit Preference in Unified Space for Deep Collaborative Filtering 

**Authors**: Zhongjin Zhang, Yu Liang, Cong Fu, Yuxuan Zhu, Kun Wang, Yabo Ni, Anxiang Zeng, Jiazhi Xia  

**Link**: [PDF](https://arxiv.org/pdf/2505.20900)  

**Abstract**: Embedding-based collaborative filtering, often coupled with nearest neighbor search, is widely deployed in large-scale recommender systems for personalized content selection. Modern systems leverage multiple implicit feedback signals (e.g., clicks, add to cart, purchases) to model user preferences comprehensively. However, prevailing approaches adopt a feedback-wise modeling paradigm, which (1) fails to capture the structured progression of user engagement entailed among different feedback and (2) embeds feedback-specific information into disjoint spaces, making representations incommensurable, increasing system complexity, and leading to suboptimal retrieval performance. A promising alternative is Ordinal Logistic Regression (OLR), which explicitly models discrete ordered relations. However, existing OLR-based recommendation models mainly focus on explicit feedback (e.g., movie ratings) and struggle with implicit, correlated feedback, where ordering is vague and non-linear. Moreover, standard OLR lacks flexibility in handling feedback-dependent covariates, resulting in suboptimal performance in real-world systems. To address these limitations, we propose Generalized Neural Ordinal Logistic Regression (GNOLR), which encodes multiple feature-feedback dependencies into a unified, structured embedding space and enforces feedback-specific dependency learning through a nested optimization framework. Thus, GNOLR enhances predictive accuracy, captures the progression of user engagement, and simplifies the retrieval process. We establish a theoretical comparison with existing paradigms, demonstrating how GNOLR avoids disjoint spaces while maintaining effectiveness. Extensive experiments on ten real-world datasets show that GNOLR significantly outperforms state-of-the-art methods in efficiency and adaptability. 

---
# Cold-Start Recommendation with Knowledge-Guided Retrieval-Augmented Generation 

**Authors**: Wooseong Yang, Weizhi Zhang, Yuqing Liu, Yuwei Han, Yu Wang, Junhyun Lee, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.20773)  

**Abstract**: Cold-start items remain a persistent challenge in recommender systems due to their lack of historical user interactions, which collaborative models rely on. While recent zero-shot methods leverage large language models (LLMs) to address this, they often struggle with sparse metadata and hallucinated or incomplete knowledge. We propose ColdRAG, a retrieval-augmented generation approach that builds a domain-specific knowledge graph dynamically to enhance LLM-based recommendation in cold-start scenarios, without requiring task-specific fine-tuning. ColdRAG begins by converting structured item attributes into rich natural-language profiles, from which it extracts entities and relationships to construct a unified knowledge graph capturing item semantics. Given a user's interaction history, it scores edges in the graph using an LLM, retrieves candidate items with supporting evidence, and prompts the LLM to rank them. By enabling multi-hop reasoning over this graph, ColdRAG grounds recommendations in verifiable evidence, reducing hallucinations and strengthening semantic connections. Experiments on three public benchmarks demonstrate that ColdRAG surpasses existing zero-shot baselines in both Recall and NDCG. This framework offers a practical solution to cold-start recommendation by combining knowledge-graph reasoning with retrieval-augmented LLM generation. 

---
# Bridging the Gap: Self-Optimized Fine-Tuning for LLM-based Recommender Systems 

**Authors**: Heng Tang, Feng Liu, Xinbo Chen, Jiawei Chen, Bohao Wang, Changwang Zhang, Jun Wang, Yuegang Sun, Bingde Hu, Can Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20771)  

**Abstract**: Recent years have witnessed extensive exploration of Large Language Models (LLMs) on the field of Recommender Systems (RS). There are currently two commonly used strategies to enable LLMs to have recommendation capabilities: 1) The "Guidance-Only" strategy uses in-context learning to exploit and amplify the inherent semantic understanding and item recommendation capabilities of LLMs; 2) The "Tuning-Only" strategy uses supervised fine-tuning (SFT) to fine-tune LLMs with the aim of fitting them to real recommendation data. However, neither of these strategies can effectively bridge the gap between the knowledge space of LLMs and recommendation, and their performance do not meet our expectations.
To better enable LLMs to learn recommendation knowledge, we combine the advantages of the above two strategies and proposed a novel "Guidance+Tuning" method called Self-Optimized Fine-Tuning (SOFT), which adopts the idea of curriculum learning. It first employs self-distillation to construct an auxiliary easy-to-learn but meaningful dataset from a fine-tuned LLM. Then it further utilizes a self-adaptive curriculum scheduler to enable LLMs to gradually learn from simpler data (self-distilled data) to more challenging data (real RS data). Extensive experiments demonstrate that SOFT significantly enhances the recommendation accuracy (37.59\% on average) of LLM-based methods. The code is available via this https URL 

---
# UQLegalAI@COLIEE2025: Advancing Legal Case Retrieval with Large Language Models and Graph Neural Networks 

**Authors**: Yanran Tang, Ruihong Qiu, Zi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20743)  

**Abstract**: Legal case retrieval plays a pivotal role in the legal domain by facilitating the efficient identification of relevant cases, supporting legal professionals and researchers to propose legal arguments and make informed decision-making. To improve retrieval accuracy, the Competition on Legal Information Extraction and Entailment (COLIEE) is held annually, offering updated benchmark datasets for evaluation. This paper presents a detailed description of CaseLink, the method employed by UQLegalAI, the second highest team in Task 1 of COLIEE 2025. The CaseLink model utilises inductive graph learning and Global Case Graphs to capture the intrinsic case connectivity to improve the accuracy of legal case retrieval. Specifically, a large language model specialized in text embedding is employed to transform legal texts into embeddings, which serve as the feature representations of the nodes in the constructed case graph. A new contrastive objective, incorporating a regularization on the degree of case nodes, is proposed to leverage the information within the case reference relationship for model optimization. The main codebase used in our method is based on an open-sourced repo of CaseLink: this https URL. 

---
# What LLMs Miss in Recommendations: Bridging the Gap with Retrieval-Augmented Collaborative Signals 

**Authors**: Shahrooz Pouryousef  

**Link**: [PDF](https://arxiv.org/pdf/2505.20730)  

**Abstract**: User-item interactions contain rich collaborative signals that form the backbone of many successful recommender systems. While recent work has explored the use of large language models (LLMs) for recommendation, it remains unclear whether LLMs can effectively reason over this type of collaborative information. In this paper, we conduct a systematic comparison between LLMs and classical matrix factorization (MF) models to assess LLMs' ability to leverage user-item interaction data. We further introduce a simple retrieval-augmented generation (RAG) method that enhances LLMs by grounding their predictions in structured interaction data. Our experiments reveal that current LLMs often fall short in capturing collaborative patterns inherent to MF models, but that our RAG-based approach substantially improves recommendation quality-highlighting a promising direction for future LLM-based recommenders. 

---
# How Do Experts Make Sense of Integrated Process Models? 

**Authors**: Tianwa Chen, Barbara Weber, Graeme Shanks, Gianluca Demartini, Marta Indulska, Shazia Sadiq  

**Link**: [PDF](https://arxiv.org/pdf/2505.20667)  

**Abstract**: A range of integrated modeling approaches have been developed to enable a holistic representation of business process logic together with all relevant business rules. These approaches address inherent problems with separate documentation of business process models and business rules. In this study, we explore how expert process workers make sense of the information provided through such integrated modeling approaches. To do so, we complement verbal protocol analysis with eye-tracking metrics to reveal nuanced user behaviours involved in the main phases of sensemaking, namely information foraging and information processing. By studying expert process workers engaged in tasks based on integrated modeling of business processes and rules, we provide insights that pave the way for a better understanding of sensemaking practices and improved development of business process and business rule integration approaches. Our research underscores the importance of offering personalized support mechanisms that increase the efficacy and efficiency of sensemaking practices for process knowledge workers. 

---
# TeroSeek: An AI-Powered Knowledge Base and Retrieval Generation Platform for Terpenoid Research 

**Authors**: Xu Kang, Siqi Jiang, Kangwei Xu, Jiahao Li, Ruibo Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.20663)  

**Abstract**: Terpenoids are a crucial class of natural products that have been studied for over 150 years, but their interdisciplinary nature (spanning chemistry, pharmacology, and biology) complicates knowledge integration. To address this, the authors developed TeroSeek, a curated knowledge base (KB) built from two decades of terpenoid literature, coupled with an AI-powered question-answering chatbot and web service. Leveraging a retrieval-augmented generation (RAG) framework, TeroSeek provides structured, high-quality information and outperforms general-purpose large language models (LLMs) in terpenoid-related queries. It serves as a domain-specific expert tool for multidisciplinary research and is publicly available at this http URL. 

---
# WikiTermBase: An AI-Augmented Term Base to Standardize Arabic Translation on Wikipedia 

**Authors**: Michel Bakni, Abbad Diraneyya, Wael Tellat  

**Link**: [PDF](https://arxiv.org/pdf/2505.20369)  

**Abstract**: Term bases are recognized as one of the most effective components of translation software in time saving and consistency. In spite of the many recent advances in natural language processing (NLP) and large language models (LLMs), major translation platforms have yet to take advantage of these tools to improve their term bases and support scalable content for underrepresented languages, which often struggle with localizing technical terminology. Language academies in the Arab World, for example, have struggled since the 1940s to unify the way new scientific terms enter the Arabic language at scale. This abstract introduces an open source tool, WikiTermBase, with a systematic approach for building a lexicographical database with over 900K terms, which were collected and mapped from a multitude of sources on a semantic and morphological basis. The tool was successfully implemented on Arabic Wikipedia to standardize translated English and French terms. 

---
# Hierarchical Retrieval with Evidence Curation for Open-Domain Financial Question Answering on Standardized Documents 

**Authors**: Jaeyoung Choe, Jihoon Kim, Woohwan Jung  

**Link**: [PDF](https://arxiv.org/pdf/2505.20368)  

**Abstract**: Retrieval-augmented generation (RAG) based large language models (LLMs) are widely used in finance for their excellent performance on knowledge-intensive tasks. However, standardized documents (e.g., SEC filing) share similar formats such as repetitive boilerplate texts, and similar table structures. This similarity forces traditional RAG methods to misidentify near-duplicate text, leading to duplicate retrieval that undermines accuracy and completeness. To address these issues, we propose the Hierarchical Retrieval with Evidence Curation (HiREC) framework. Our approach first performs hierarchical retrieval to reduce confusion among similar texts. It first retrieve related documents and then selects the most relevant passages from the documents. The evidence curation process removes irrelevant passages. When necessary, it automatically generates complementary queries to collect missing information. To evaluate our approach, we construct and release a Large-scale Open-domain Financial (LOFin) question answering benchmark that includes 145,897 SEC documents and 1,595 question-answer pairs. Our code and data are available at this https URL. 

---
# VSCBench: Bridging the Gap in Vision-Language Model Safety Calibration 

**Authors**: Jiahui Geng, Qing Li, Zongxiong Chen, Yuxia Wang, Derui Zhu, Zhuohan Xie, Chenyang Lyu, Xiuying Chen, Preslav Nakov, Fakhri Karray  

**Link**: [PDF](https://arxiv.org/pdf/2505.20362)  

**Abstract**: The rapid advancement of vision-language models (VLMs) has brought a lot of attention to their safety alignment. However, existing methods have primarily focused on model undersafety, where the model responds to hazardous queries, while neglecting oversafety, where the model refuses to answer safe queries. In this paper, we introduce the concept of $\textit{safety calibration}$, which systematically addresses both undersafety and oversafety. Specifically, we present $\textbf{VSCBench}$, a novel dataset of 3,600 image-text pairs that are visually or textually similar but differ in terms of safety, which is designed to evaluate safety calibration across image-centric and text-centric scenarios. Based on our benchmark, we evaluate safety calibration across eleven widely used VLMs. Our extensive experiments revealed major issues with both undersafety and oversafety. We further investigated four approaches to improve the model's safety calibration. We found that even though some methods effectively calibrated the models' safety problems, these methods also lead to the degradation of models' utility. This trade-off underscores the urgent need for advanced calibration methods, and our benchmark provides a valuable tool for evaluating future approaches. Our code and data are available at this https URL. 

---
# Large Language Model-Powered Decision Support for a Metal Additive Manufacturing Knowledge Graph 

**Authors**: Muhammad Tayyab Khan, Lequn Chen, Wenhe Feng, Seung Ki Moon  

**Link**: [PDF](https://arxiv.org/pdf/2505.20308)  

**Abstract**: Metal additive manufacturing (AM) involves complex interdependencies among processes, materials, feedstock, and post-processing steps. However, the underlying relationships and domain knowledge remain fragmented across literature and static databases that often demand expert-level queries, limiting their applicability in design and planning. To address these gaps, we develop a novel and queryable knowledge graph (KG) in Neo4j, encoding 53 distinct metals and alloys across seven material families, nine AM processes, four feedstock types, and associated post-processing requirements. A large language model (LLM) interface, guided by a few-shot prompting strategy, enables natural language querying without the need for formal query syntax. The system supports a range of tasks, including compatibility checks, multi-constraint filtering, and design for AM (DfAM) guidance. User natural language queries are normalized, translated into Cypher, and executed over the KG, with results reformatted into structured responses. This work presents the first real-time, interactive system that integrates a domain-specific metal AM KG with an LLM interface, offering accessible, explainable decision support for engineers and advancing human-centric tools in manufacturing intelligence. 

---
# LazyVLM: Neuro-Symbolic Approach to Video Analytics 

**Authors**: Xiangru Jian, Wei Pang, Zhengyuan Dong, Chao Zhang, M. Tamer Özsu  

**Link**: [PDF](https://arxiv.org/pdf/2505.21459)  

**Abstract**: Current video analytics approaches face a fundamental trade-off between flexibility and efficiency. End-to-end Vision Language Models (VLMs) often struggle with long-context processing and incur high computational costs, while neural-symbolic methods depend heavily on manual labeling and rigid rule design. In this paper, we introduce LazyVLM, a neuro-symbolic video analytics system that provides a user-friendly query interface similar to VLMs, while addressing their scalability limitation. LazyVLM enables users to effortlessly drop in video data and specify complex multi-frame video queries using a semi-structured text interface for video analytics. To address the scalability limitations of VLMs, LazyVLM decomposes multi-frame video queries into fine-grained operations and offloads the bulk of the processing to efficient relational query execution and vector similarity search. We demonstrate that LazyVLM provides a robust, efficient, and user-friendly solution for querying open-domain video data at scale. 

---
# Towards Better Instruction Following Retrieval Models 

**Authors**: Yuchen Zhuang, Aaron Trinh, Rushi Qiang, Haotian Sun, Chao Zhang, Hanjun Dai, Bo Dai  

**Link**: [PDF](https://arxiv.org/pdf/2505.21439)  

**Abstract**: Modern information retrieval (IR) models, trained exclusively on standard <query, passage> pairs, struggle to effectively interpret and follow explicit user instructions. We introduce InF-IR, a large-scale, high-quality training corpus tailored for enhancing retrieval models in Instruction-Following IR. InF-IR expands traditional training pairs into over 38,000 expressive <instruction, query, passage> triplets as positive samples. In particular, for each positive triplet, we generate two additional hard negative examples by poisoning both instructions and queries, then rigorously validated by an advanced reasoning model (o3-mini) to ensure semantic plausibility while maintaining instructional incorrectness. Unlike existing corpora that primarily support computationally intensive reranking tasks for decoder-only language models, the highly contrastive positive-negative triplets in InF-IR further enable efficient representation learning for smaller encoder-only models, facilitating direct embedding-based retrieval. Using this corpus, we train InF-Embed, an instruction-aware Embedding model optimized through contrastive learning and instruction-query attention mechanisms to align retrieval outcomes precisely with user intents. Extensive experiments across five instruction-based retrieval benchmarks demonstrate that InF-Embed significantly surpasses competitive baselines by 8.1% in p-MRR, measuring the instruction-following capabilities. 

---
# A Lightweight Multi-Expert Generative Language Model System for Engineering Information and Knowledge Extraction 

**Authors**: Bogdan Bogachov, Yaoyao Fiona Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.21109)  

**Abstract**: Despite recent advancements in domain adaptation techniques for large language models, these methods remain computationally intensive, and the resulting models can still exhibit hallucination issues. Most existing adaptation methods do not prioritize reducing the computational resources required for fine-tuning and inference of language models. Hallucination issues have gradually decreased with each new model release. However, they remain prevalent in engineering contexts, where generating well-structured text with minimal errors and inconsistencies is critical. This work introduces a novel approach called the Small Language Graph (SLG), which is a lightweight adaptation solution designed to address the two key challenges outlined above. The system is structured in the form of a graph, where each node represents a lightweight expert - a small language model fine-tuned on specific and concise texts. The results of this study have shown that SLG was able to surpass conventional fine-tuning methods on the Exact Match metric by 3 times. Additionally, the fine-tuning process was 1.7 times faster compared to that of a larger stand-alone language model. These findings introduce a potential for small to medium-sized engineering companies to confidently use generative AI technologies, such as LLMs, without the necessity to invest in expensive computational resources. Also, the graph architecture and the small size of expert nodes offer a possible opportunity for distributed AI systems, thus potentially diverting the global need for expensive centralized compute clusters. 

---
# Personalized Query Auto-Completion for Long and Short-Term Interests with Adaptive Detoxification Generation 

**Authors**: Zhibo Wang, Xiaoze Jiang, Zhiheng Qin, Enyun Yu, Han Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.20966)  

**Abstract**: Query auto-completion (QAC) plays a crucial role in modern search systems. However, in real-world applications, there are two pressing challenges that still need to be addressed. First, there is a need for hierarchical personalized representations for users. Previous approaches have typically used users' search behavior as a single, overall representation, which proves inadequate in more nuanced generative scenarios. Additionally, query prefixes are typically short and may contain typos or sensitive information, increasing the likelihood of generating toxic content compared to traditional text generation tasks. Such toxic content can degrade user experience and lead to public relations issues. Therefore, the second critical challenge is detoxifying QAC systems.
To address these two limitations, we propose a novel model (LaD) that captures personalized information from both long-term and short-term interests, incorporating adaptive detoxification. In LaD, personalized information is captured hierarchically at both coarse-grained and fine-grained levels. This approach preserves as much personalized information as possible while enabling online generation within time constraints. To move a futher step, we propose an online training method based on Reject Preference Optimization (RPO). By incorporating a special token [Reject] during both the training and inference processes, the model achieves adaptive detoxification. Consequently, the generated text presented to users is both non-toxic and relevant to the given prefix. We conduct comprehensive experiments on industrial-scale datasets and perform online A/B tests, delivering the largest single-experiment metric improvement in nearly two years of our product. Our model has been deployed on Kuaishou search, driving the primary traffic for hundreds of millions of active users. The code is available at this https URL. 

---
# Beyond Prompt Engineering: Robust Behavior Control in LLMs via Steering Target Atoms 

**Authors**: Mengru Wang, Ziwen Xu, Shengyu Mao, Shumin Deng, Zhaopeng Tu, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20322)  

**Abstract**: Precise control over language model generation is vital for ensuring both safety and reliability. Although prompt engineering and steering are commonly used to intervene in model behaviors, the vast number of parameters in models often results in highly intertwined internal representations. This interdependency can limit control precision and sometimes lead to unintended side effects. Recent research has explored the use of sparse autoencoders (SAE) to disentangle knowledge in high-dimensional spaces for steering. However, these applications have been limited to toy tasks owing to the nontrivial issue of locating atomic knowledge components. In this paper, we propose Steering Target Atoms (STA), a novel method that isolates and manipulates disentangled knowledge components to enhance safety. Comprehensive experiments demonstrate the effectiveness of our approach. Further analysis reveals that steering exhibits superior robustness and flexibility, particularly in adversarial scenarios. We also apply the steering strategy to the large reasoning model, confirming its effectiveness in precise reasoning control. 

---
# CAMEF: Causal-Augmented Multi-Modality Event-Driven Financial Forecasting by Integrating Time Series Patterns and Salient Macroeconomic Announcements 

**Authors**: Yang Zhang, Wenbo Yang, Jun Wang, Qiang Ma, Jie Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2502.04592)  

**Abstract**: Accurately forecasting the impact of macroeconomic events is critical for investors and policymakers. Salient events like monetary policy decisions and employment reports often trigger market movements by shaping expectations of economic growth and risk, thereby establishing causal relationships between events and market behavior. Existing forecasting methods typically focus either on textual analysis or time-series modeling, but fail to capture the multi-modal nature of financial markets and the causal relationship between events and price movements. To address these gaps, we propose CAMEF (Causal-Augmented Multi-Modality Event-Driven Financial Forecasting), a multi-modality framework that effectively integrates textual and time-series data with a causal learning mechanism and an LLM-based counterfactual event augmentation technique for causal-enhanced financial forecasting. Our contributions include: (1) a multi-modal framework that captures causal relationships between policy texts and historical price data; (2) a new financial dataset with six types of macroeconomic releases from 2008 to April 2024, and high-frequency real trading data for five key U.S. financial assets; and (3) an LLM-based counterfactual event augmentation strategy. We compare CAMEF to state-of-the-art transformer-based time-series and multi-modal baselines, and perform ablation studies to validate the effectiveness of the causal learning mechanism and event types. 

---
