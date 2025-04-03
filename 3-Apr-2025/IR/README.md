# Is Less Really More? Fake News Detection with Limited Information 

**Authors**: Zhaoyang Cao, John Nguyen, Reza Zafarani  

**Link**: [PDF](https://arxiv.org/pdf/2504.01922)  

**Abstract**: The threat that online fake news and misinformation pose to democracy, justice, public confidence, and especially to vulnerable populations, has led to a sharp increase in the need for fake news detection and intervention. Whether multi-modal or pure text-based, most fake news detection methods depend on textual analysis of entire articles. However, these fake news detection methods come with certain limitations. For instance, fake news detection methods that rely on full text can be computationally inefficient, demand large amounts of training data to achieve competitive accuracy, and may lack robustness across different datasets. This is because fake news datasets have strong variations in terms of the level and types of information they provide; where some can include large paragraphs of text with images and metadata, others can be a few short sentences. Perhaps if one could only use minimal information to detect fake news, fake news detection methods could become more robust and resilient to the lack of information. We aim to overcome these limitations by detecting fake news using systematically selected, limited information that is both effective and capable of delivering robust, promising performance. We propose a framework called SLIM Systematically-selected Limited Information) for fake news detection. In SLIM, we quantify the amount of information by introducing information-theoretic measures. SLIM leverages limited information to achieve performance in fake news detection comparable to that of state-of-the-art obtained using the full text. Furthermore, by combining various types of limited information, SLIM can perform even better while significantly reducing the quantity of information required for training compared to state-of-the-art language model-based fake news detection techniques. 

---
# Extending MovieLens-32M to Provide New Evaluation Objectives 

**Authors**: Mark D. Smucker, Houmaan Chamani  

**Link**: [PDF](https://arxiv.org/pdf/2504.01863)  

**Abstract**: Offline evaluation of recommender systems has traditionally treated the problem as a machine learning problem. In the classic case of recommending movies, where the user has provided explicit ratings of which movies they like and don't like, each user's ratings are split into test and train sets, and the evaluation task becomes to predict the held out test data using the training data. This machine learning style of evaluation makes the objective to recommend the movies that a user has watched and rated highly, which is not the same task as helping the user find movies that they would enjoy if they watched them. This mismatch in objective between evaluation and task is a compromise to avoid the cost of asking a user to evaluate recommendations by watching each movie. As a resource available for download, we offer an extension to the MovieLens-32M dataset that provides for new evaluation objectives. Our primary objective is to predict the movies that a user would be interested in watching, i.e. predict their watchlist. To construct this extension, we recruited MovieLens users, collected their profiles, made recommendations with a diverse set of algorithms, pooled the recommendations, and had the users assess the pools. Notably, we found that the traditional machine learning style of evaluation ranks the Popular algorithm, which recommends movies based on total number of ratings in the system, in the middle of the twenty-two recommendation runs we used to build the pools. In contrast, when we rank the runs by users' interest in watching movies, we find that recommending popular movies as a recommendation algorithm becomes one of the worst performing runs. It appears that by asking users to assess their personal recommendations, we can alleviate the popularity bias issues created by using information retrieval effectiveness measures for the evaluation of recommender systems. 

---
# Efficient Constant-Space Multi-Vector Retrieval 

**Authors**: Sean MacAvaney, Antonio Mallia, Nicola Tonellotto  

**Link**: [PDF](https://arxiv.org/pdf/2504.01818)  

**Abstract**: Multi-vector retrieval methods, exemplified by the ColBERT architecture, have shown substantial promise for retrieval by providing strong trade-offs in terms of retrieval latency and effectiveness. However, they come at a high cost in terms of storage since a (potentially compressed) vector needs to be stored for every token in the input collection. To overcome this issue, we propose encoding documents to a fixed number of vectors, which are no longer necessarily tied to the input tokens. Beyond reducing the storage costs, our approach has the advantage that document representations become of a fixed size on disk, allowing for better OS paging management. Through experiments using the MSMARCO passage corpus and BEIR with the ColBERT-v2 architecture, a representative multi-vector ranking model architecture, we find that passages can be effectively encoded into a fixed number of vectors while retaining most of the original effectiveness. 

---
# Horizon Scans can be accelerated using novel information retrieval and artificial intelligence tools 

**Authors**: Lena Schmidt, Oshin Sharma, Chris Marshall, Sonia Garcia Gonzalez Moral  

**Link**: [PDF](https://arxiv.org/pdf/2504.01627)  

**Abstract**: Introduction: Horizon scanning in healthcare assesses early signals of innovation, crucial for timely adoption. Current horizon scanning faces challenges in efficient information retrieval and analysis, especially from unstructured sources like news, presenting a need for innovative tools. Methodology: The study introduces SCANAR and AIDOC, open-source Python-based tools designed to improve horizon scanning. SCANAR automates the retrieval and processing of news articles, offering functionalities such as de-duplication and unsupervised relevancy ranking. AIDOC aids filtration by leveraging AI to reorder textual data based on relevancy, employing neural networks for semantic similarity, and subsequently prioritizing likely relevant entries for human review. Results: Twelve internal datasets from horizon scans and four external benchmarking datasets were used. SCANAR improved retrieval efficiency by automating processes previously dependent on manual labour. AIDOC displayed work-saving potential, achieving around 62% reduction in manual review efforts at 95% recall. Comparative analysis with benchmarking data showed AIDOC's performance was similar to existing systematic review automation tools, though performance varied depending on dataset characteristics. A smaller case-study on our news datasets shows the potential of ensembling large language models within the active-learning process for faster detection of relevant articles across news datasets. Conclusion: The validation indicates that SCANAR and AIDOC show potential to enhance horizon scanning efficiency by streamlining data retrieval and prioritisation. These tools may alleviate methodological limitations and allow broader, swifter horizon scans. Further studies are suggested to optimize these models and to design new workflows and validation processes that integrate large language models. 

---
# Comment Staytime Prediction with LLM-enhanced Comment Understanding 

**Authors**: Changshuo Zhang, Zihan Lin, Shukai Liu, Yongqi Liu, Han Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.01602)  

**Abstract**: In modern online streaming platforms, the comments section plays a critical role in enhancing the overall user experience. Understanding user behavior within the comments section is essential for comprehensive user interest modeling. A key factor of user engagement is staytime, which refers to the amount of time that users browse and post comments. Existing watchtime prediction methods struggle to adapt to staytime prediction, overlooking interactions with individual comments and their interrelation. In this paper, we present a micro-video recommendation dataset with video comments (named as KuaiComt) which is collected from Kuaishou platform. correspondingly, we propose a practical framework for comment staytime prediction with LLM-enhanced Comment Understanding (LCU). Our framework leverages the strong text comprehension capabilities of large language models (LLMs) to understand textual information of comments, while also incorporating fine-grained comment ranking signals as auxiliary tasks. The framework is two-staged: first, the LLM is fine-tuned using domain-specific tasks to bridge the video and the comments; second, we incorporate the LLM outputs into the prediction model and design two comment ranking auxiliary tasks to better understand user preference. Extensive offline experiments demonstrate the effectiveness of our framework, showing significant improvements on the task of comment staytime prediction. Additionally, online A/B testing further validates the practical benefits on industrial scenario. Our dataset KuaiComt (this https URL) and code for LCU (this https URL) are fully released. 

---
# Hyperbolic Diffusion Recommender Model 

**Authors**: Meng Yuan, Yutian Xiao, Wei Chen, Chu Zhao, Deqing Wang, Fuzhen Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2504.01541)  

**Abstract**: Diffusion models (DMs) have emerged as the new state-of-the-art family of deep generative models. To gain deeper insights into the limitations of diffusion models in recommender systems, we investigate the fundamental structural disparities between images and items. Consequently, items often exhibit distinct anisotropic and directional structures that are less prevalent in images. However, the traditional forward diffusion process continuously adds isotropic Gaussian noise, causing anisotropic signals to degrade into noise, which impairs the semantically meaningful representations in recommender systems.
Inspired by the advancements in hyperbolic spaces, we propose a novel \textit{\textbf{H}yperbolic} \textit{\textbf{D}iffusion} \textit{\textbf{R}ecommender} \textit{\textbf{M}odel} (named HDRM). Unlike existing directional diffusion methods based on Euclidean space, the intrinsic non-Euclidean structure of hyperbolic space makes it particularly well-adapted for handling anisotropic diffusion processes. In particular, we begin by formulating concepts to characterize latent directed diffusion processes within a geometrically grounded hyperbolic space. Subsequently, we propose a novel hyperbolic latent diffusion process specifically tailored for users and items. Drawing upon the natural geometric attributes of hyperbolic spaces, we impose structural restrictions on the space to enhance hyperbolic diffusion propagation, thereby ensuring the preservation of the intrinsic topology of user-item graphs. Extensive experiments on three benchmark datasets demonstrate the effectiveness of HDRM. 

---
# Test-Time Alignment for Tracking User Interest Shifts in Sequential Recommendation 

**Authors**: Changshuo Zhang, Xiao Zhang, Teng Shi, Jun Xu, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2504.01489)  

**Abstract**: Sequential recommendation is essential in modern recommender systems, aiming to predict the next item a user may interact with based on their historical behaviors. However, real-world scenarios are often dynamic and subject to shifts in user interests. Conventional sequential recommendation models are typically trained on static historical data, limiting their ability to adapt to such shifts and resulting in significant performance degradation during testing. Recently, Test-Time Training (TTT) has emerged as a promising paradigm, enabling pre-trained models to dynamically adapt to test data by leveraging unlabeled examples during testing. However, applying TTT to effectively track and address user interest shifts in recommender systems remains an open and challenging problem. Key challenges include how to capture temporal information effectively and explicitly identifying shifts in user interests during the testing phase. To address these issues, we propose T$^2$ARec, a novel model leveraging state space model for TTT by introducing two Test-Time Alignment modules tailored for sequential recommendation, effectively capturing the distribution shifts in user interest patterns over time. Specifically, T$^2$ARec aligns absolute time intervals with model-adaptive learning intervals to capture temporal dynamics and introduce an interest state alignment mechanism to effectively and explicitly identify the user interest shifts with theoretical guarantees. These two alignment modules enable efficient and incremental updates to model parameters in a self-supervised manner during testing, enhancing predictions for online recommendation. Extensive evaluations on three benchmark datasets demonstrate that T$^2$ARec achieves state-of-the-art performance and robustly mitigates the challenges posed by user interest shifts. 

---
# GeoRAG: A Question-Answering Approach from a Geographical Perspective 

**Authors**: Jian Wang, Zhuo Zhao, Zheng Jie Wang, Bo Da Cheng, Lei Nie, Wen Luo, Zhao Yuan Yu, Ling Wang Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2504.01458)  

**Abstract**: Geographic Question Answering (GeoQA) addresses natural language queries in geographic domains to fulfill complex user demands and improve information retrieval efficiency. Traditional QA systems, however, suffer from limited comprehension, low retrieval accuracy, weak interactivity, and inadequate handling of complex tasks, hindering precise information acquisition. This study presents GeoRAG, a knowledge-enhanced QA framework integrating domain-specific fine-tuning and prompt engineering with Retrieval-Augmented Generation (RAG) technology to enhance geographic knowledge retrieval accuracy and user interaction. The methodology involves four components: (1) A structured geographic knowledge base constructed from 3267 corpora (research papers, monographs, and technical reports), categorized via a multi-agent approach into seven dimensions: semantic understanding, spatial location, geometric morphology, attribute characteristics, feature relationships, evolutionary processes, and operational mechanisms. This yielded 145234 classified entries and 875432 multi-dimensional QA pairs. (2) A multi-label text classifier based on BERT-Base-Chinese, trained to analyze query types through geographic dimension classification. (3) A retrieval evaluator leveraging QA pair data to assess query-document relevance, optimizing retrieval precision. (4) GeoPrompt templates engineered to dynamically integrate user queries with retrieved information, enhancing response quality through dimension-specific prompting. Comparative experiments demonstrate GeoRAG's superior performance over conventional RAG across multiple base models, validating its generalizability. This work advances geographic AI by proposing a novel paradigm for deploying large language models in domain-specific contexts, with implications for improving GeoQA systems scalability and accuracy in real-world applications. 

---
# LLM-VPRF: Large Language Model Based Vector Pseudo Relevance Feedback 

**Authors**: Hang Li, Shengyao Zhuang, Bevan Koopman, Guido Zuccon  

**Link**: [PDF](https://arxiv.org/pdf/2504.01448)  

**Abstract**: Vector Pseudo Relevance Feedback (VPRF) has shown promising results in improving BERT-based dense retrieval systems through iterative refinement of query representations. This paper investigates the generalizability of VPRF to Large Language Model (LLM) based dense retrievers. We introduce LLM-VPRF and evaluate its effectiveness across multiple benchmark datasets, analyzing how different LLMs impact the feedback mechanism. Our results demonstrate that VPRF's benefits successfully extend to LLM architectures, establishing it as a robust technique for enhancing dense retrieval performance regardless of the underlying models. This work bridges the gap between VPRF with traditional BERT-based dense retrievers and modern LLMs, while providing insights into their future directions. 

---
# Generative Retrieval and Alignment Model: A New Paradigm for E-commerce Retrieval 

**Authors**: Ming Pang, Chunyuan Yuan, Xiaoyu He, Zheng Fang, Donghao Xie, Fanyi Qu, Xue Jiang, Changping Peng, Zhangang Lin, Zheng Luo, Jingping Shao  

**Link**: [PDF](https://arxiv.org/pdf/2504.01403)  

**Abstract**: Traditional sparse and dense retrieval methods struggle to leverage general world knowledge and often fail to capture the nuanced features of queries and products. With the advent of large language models (LLMs), industrial search systems have started to employ LLMs to generate identifiers for product retrieval. Commonly used identifiers include (1) static/semantic IDs and (2) product term sets. The first approach requires creating a product ID system from scratch, missing out on the world knowledge embedded within LLMs. While the second approach leverages this general knowledge, the significant difference in word distribution between queries and products means that product-based identifiers often do not align well with user search queries, leading to missed product recalls. Furthermore, when queries contain numerous attributes, these algorithms generate a large number of identifiers, making it difficult to assess their quality, which results in low overall recall efficiency.
To address these challenges, this paper introduces a novel e-commerce retrieval paradigm: the Generative Retrieval and Alignment Model (GRAM). GRAM employs joint training on text information from both queries and products to generate shared text identifier codes, effectively bridging the gap between queries and products. This approach not only enhances the connection between queries and products but also improves inference efficiency. The model uses a co-alignment strategy to generate codes optimized for maximizing retrieval efficiency. Additionally, it introduces a query-product scoring mechanism to compare product values across different codes, further boosting retrieval efficiency. Extensive offline and online A/B testing demonstrates that GRAM significantly outperforms traditional models and the latest generative retrieval models, confirming its effectiveness and practicality. 

---
# Real-time Ad retrieval via LLM-generative Commercial Intention for Sponsored Search Advertising 

**Authors**: Tongtong Liu, Zhaohui Wang, Meiyue Qin, Zenghui Lu, Xudong Chen, Yuekui Yang, Peng Shu  

**Link**: [PDF](https://arxiv.org/pdf/2504.01304)  

**Abstract**: The integration of Large Language Models (LLMs) with retrieval systems has shown promising potential in retrieving documents (docs) or advertisements (ads) for a given query. Existing LLM-based retrieval methods generate numeric or content-based DocIDs to retrieve docs/ads. However, the one-to-few mapping between numeric IDs and docs, along with the time-consuming content extraction, leads to semantic inefficiency and limits scalability in large-scale corpora. In this paper, we propose the Real-time Ad REtrieval (RARE) framework, which leverages LLM-generated text called Commercial Intentions (CIs) as an intermediate semantic representation to directly retrieve ads for queries in real-time. These CIs are generated by a customized LLM injected with commercial knowledge, enhancing its domain relevance. Each CI corresponds to multiple ads, yielding a lightweight and scalable set of CIs. RARE has been implemented in a real-world online system, handling daily search volumes in the hundreds of millions. The online implementation has yielded significant benefits: a 5.04% increase in consumption, a 6.37% rise in Gross Merchandise Volume (GMV), a 1.28% enhancement in click-through rate (CTR) and a 5.29% increase in shallow conversions. Extensive offline experiments show RARE's superiority over ten competitive baselines in four major categories. 

---
# Migrating a Job Search Relevance Function 

**Authors**: Bennett Mountain, Gabriel Womark, Ritvik Kharkar  

**Link**: [PDF](https://arxiv.org/pdf/2504.01284)  

**Abstract**: In this paper, we describe the migration of a homebrewed C++ search engine to OpenSearch, aimed at preserving and improving search performance with minimal impact on business metrics. To facilitate the migration, we froze our job corpus and executed queries in low inventory locations to capture a representative mixture of high- and low-quality search results. These query-job pairs were labeled by crowd-sourced annotators using a custom rubric designed to reflect relevance and user satisfaction. Leveraging Bayesian optimization, we fine-tuned a new retrieval algorithm on OpenSearch, replicating key components of the original engine's logic while introducing new functionality where necessary. Through extensive online testing, we demonstrated that the new system performed on par with the original, showing improvements in specific engagement metrics, with negligible effects on revenue. 

---
# Information Retrieval for Climate Impact 

**Authors**: Maarten de Rijke, Bart van den Hurk, Flora Salim, Alaa Al Khourdajie, Nan Bai, Renato Calzone, Declan Curran, Getnet Demil, Lesley Frew, Noah Gießing, Mukesh Kumar Gupta, Maria Heuss, Sanaa Hobeichi, David Huard, Jingwei Kang, Ana Lucic, Tanwi Mallick, Shruti Nath, Andrew Okem, Barbara Pernici, Thilina Rajapakse, Hira Saleem, Harry Scells, Nicole Schneider, Damiano Spina, Yuanyuan Tian, Edmund Totin, Andrew Trotman, Ramamurthy Valavandan, Dereje Workneh, Yangxinyu Xie  

**Link**: [PDF](https://arxiv.org/pdf/2504.01162)  

**Abstract**: The purpose of the MANILA24 Workshop on information retrieval for climate impact was to bring together researchers from academia, industry, governments, and NGOs to identify and discuss core research problems in information retrieval to assess climate change impacts. The workshop aimed to foster collaboration by bringing communities together that have so far not been very well connected -- information retrieval, natural language processing, systematic reviews, impact assessments, and climate science. The workshop brought together a diverse set of researchers and practitioners interested in contributing to the development of a technical research agenda for information retrieval to assess climate change impacts. 

---
# Uncovering the Limitations of Query Performance Prediction: Failures, Insights, and Implications for Selective Query Processing 

**Authors**: Adrian-Gabriel Chifu, Sébastien Déjean, Josiane Mothe, Moncef Garouani, Diego Ortiz, Md Zia Ullah  

**Link**: [PDF](https://arxiv.org/pdf/2504.01101)  

**Abstract**: Query Performance Prediction (QPP) estimates retrieval systems effectiveness for a given query, offering valuable insights for search effectiveness and query processing. Despite extensive research, QPPs face critical challenges in generalizing across diverse retrieval paradigms and collections. This paper provides a comprehensive evaluation of state-of-the-art QPPs (e.g. NQC, UQC), LETOR-based features, and newly explored dense-based predictors. Using diverse sparse rankers (BM25, DFree without and with query expansion) and hybrid or dense (SPLADE and ColBert) rankers and diverse test collections ROBUST, GOV2, WT10G, and MS MARCO; we investigate the relationships between predicted and actual performance, with a focus on generalization and robustness. Results show significant variability in predictors accuracy, with collections as the main factor and rankers next. Some sparse predictors perform somehow on some collections (TREC ROBUST and GOV2) but do not generalise to other collections (WT10G and MS-MARCO). While some predictors show promise in specific scenarios, their overall limitations constrain their utility for applications. We show that QPP-driven selective query processing offers only marginal gains, emphasizing the need for improved predictors that generalize across collections, align with dense retrieval architectures and are useful for downstream applications. 

---
# CoRAG: Collaborative Retrieval-Augmented Generation 

**Authors**: Aashiq Muhamed, Mona Diab, Virginia Smith  

**Link**: [PDF](https://arxiv.org/pdf/2504.01883)  

**Abstract**: Retrieval-Augmented Generation (RAG) models excel in knowledge-intensive tasks, especially under few-shot learning constraints. We introduce CoRAG, a framework extending RAG to collaborative settings, where clients jointly train a shared model using a collaborative passage store. To evaluate CoRAG, we introduce CRAB, a benchmark for collaborative homogeneous open-domain question answering. Our experiments demonstrate that CoRAG consistently outperforms both parametric collaborative learning methods and locally trained RAG models in low-resource scenarios. Further analysis reveals the critical importance of relevant passages within the shared store, the surprising benefits of incorporating irrelevant passages, and the potential for hard negatives to negatively impact performance. This introduces a novel consideration in collaborative RAG: the trade-off between leveraging a collectively enriched knowledge base and the potential risk of incorporating detrimental passages from other clients. Our findings underscore the viability of CoRAG, while also highlighting key design challenges and promising avenues for future research. 

---
# TransientTables: Evaluating LLMs' Reasoning on Temporally Evolving Semi-structured Tables 

**Authors**: Abhilash Shankarampeta, Harsh Mahajan, Tushar Kataria, Dan Roth, Vivek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2504.01879)  

**Abstract**: Humans continuously make new discoveries, and understanding temporal sequence of events leading to these breakthroughs is essential for advancing science and society. This ability to reason over time allows us to identify future steps and understand the effects of financial and political decisions on our lives. However, large language models (LLMs) are typically trained on static datasets, limiting their ability to perform effective temporal reasoning. To assess the temporal reasoning capabilities of LLMs, we present the TRANSIENTTABLES dataset, which comprises 3,971 questions derived from over 14,000 tables, spanning 1,238 entities across multiple time periods. We introduce a template-based question-generation pipeline that harnesses LLMs to refine both templates and questions. Additionally, we establish baseline results using state-of-the-art LLMs to create a benchmark. We also introduce novel modeling strategies centered around task decomposition, enhancing LLM performance. 

---
# Prompt-Guided Attention Head Selection for Focus-Oriented Image Retrieval 

**Authors**: Yuji Nozawa, Yu-Chieh Lin, Kazumoto Nakamura, Youyang Ng  

**Link**: [PDF](https://arxiv.org/pdf/2504.01348)  

**Abstract**: The goal of this paper is to enhance pretrained Vision Transformer (ViT) models for focus-oriented image retrieval with visual prompting. In real-world image retrieval scenarios, both query and database images often exhibit complexity, with multiple objects and intricate backgrounds. Users often want to retrieve images with specific object, which we define as the Focus-Oriented Image Retrieval (FOIR) task. While a standard image encoder can be employed to extract image features for similarity matching, it may not perform optimally in the multi-object-based FOIR task. This is because each image is represented by a single global feature vector. To overcome this, a prompt-based image retrieval solution is required. We propose an approach called Prompt-guided attention Head Selection (PHS) to leverage the head-wise potential of the multi-head attention mechanism in ViT in a promptable manner. PHS selects specific attention heads by matching their attention maps with user's visual prompts, such as a point, box, or segmentation. This empowers the model to focus on specific object of interest while preserving the surrounding visual context. Notably, PHS does not necessitate model re-training and avoids any image alteration. Experimental results show that PHS substantially improves performance on multiple datasets, offering a practical and training-free solution to enhance model performance in the FOIR task. 

---
# GTR: Graph-Table-RAG for Cross-Table Question Answering 

**Authors**: Jiaru Zou, Dongqi Fu, Sirui Chen, Xinrui He, Zihao Li, Yada Zhu, Jiawei Han, Jingrui He  

**Link**: [PDF](https://arxiv.org/pdf/2504.01346)  

**Abstract**: Beyond pure text, a substantial amount of knowledge is stored in tables. In real-world scenarios, user questions often require retrieving answers that are distributed across multiple tables. GraphRAG has recently attracted much attention for enhancing LLMs' reasoning capabilities by organizing external knowledge to address ad-hoc and complex questions, exemplifying a promising direction for cross-table question answering. In this paper, to address the current gap in available data, we first introduce a multi-table benchmark, MutliTableQA, comprising 60k tables and 25k user queries collected from real-world sources. Then, we propose the first Graph-Table-RAG framework, namely GTR, which reorganizes table corpora into a heterogeneous graph, employs a hierarchical coarse-to-fine retrieval process to extract the most relevant tables, and integrates graph-aware prompting for downstream LLMs' tabular reasoning. Extensive experiments show that GTR exhibits superior cross-table question-answering performance while maintaining high deployment efficiency, demonstrating its real-world practical applicability. 

---
# Biomedical Question Answering via Multi-Level Summarization on a Local Knowledge Graph 

**Authors**: Lingxiao Guan, Yuanhao Huang, Jie Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.01309)  

**Abstract**: In Question Answering (QA), Retrieval Augmented Generation (RAG) has revolutionized performance in various domains. However, how to effectively capture multi-document relationships, particularly critical for biomedical tasks, remains an open question. In this work, we propose a novel method that utilizes propositional claims to construct a local knowledge graph from retrieved documents. Summaries are then derived via layerwise summarization from the knowledge graph to contextualize a small language model to perform QA. We achieved comparable or superior performance with our method over RAG baselines on several biomedical QA benchmarks. We also evaluated each individual step of our methodology over a targeted set of metrics, demonstrating its effectiveness. 

---
# Scaling Test-Time Inference with Policy-Optimized, Dynamic Retrieval-Augmented Generation via KV Caching and Decoding 

**Authors**: Sakhinana Sagar Srinivas, Venkataramana Runkana  

**Link**: [PDF](https://arxiv.org/pdf/2504.01281)  

**Abstract**: We present a comprehensive framework for enhancing Retrieval-Augmented Generation (RAG) systems through dynamic retrieval strategies and reinforcement fine-tuning. This approach significantly improves large language models on knowledge-intensive tasks, including opendomain question answering and complex reasoning. Our framework integrates two complementary techniques: Policy-Optimized RetrievalAugmented Generation (PORAG), which optimizes the use of retrieved information, and Adaptive Token-Layer Attention Scoring (ATLAS), which dynamically determines retrieval timing and content based on contextual needs. Together, these techniques enhance both the utilization and relevance of retrieved content, improving factual accuracy and response quality. Designed as a lightweight solution compatible with any Transformer-based LLM without requiring additional training, our framework excels in knowledge-intensive tasks, boosting output accuracy in RAG settings. We further propose CRITIC, a novel method to selectively compress key-value caches by token importance, mitigating memory bottlenecks in long-context applications. The framework also incorporates test-time scaling techniques to dynamically balance reasoning depth and computational resources, alongside optimized decoding strategies for faster inference. Experiments on benchmark datasets show that our framework reduces hallucinations, strengthens domain-specific reasoning, and achieves significant efficiency and scalability gains over traditional RAG systems. This integrated approach advances the development of robust, efficient, and scalable RAG systems across diverse applications. 

---
# Beyond Quacking: Deep Integration of Language Models and RAG into DuckDB 

**Authors**: Anas Dorbani, Sunny Yasser, Jimmy Lin, Amine Mhedhbi  

**Link**: [PDF](https://arxiv.org/pdf/2504.01157)  

**Abstract**: Knowledge-intensive analytical applications retrieve context from both structured tabular data and unstructured, text-free documents for effective decision-making. Large language models (LLMs) have made it significantly easier to prototype such retrieval and reasoning data pipelines. However, implementing these pipelines efficiently still demands significant effort and has several challenges. This often involves orchestrating heterogeneous data systems, managing data movement, and handling low-level implementation details, e.g., LLM context management.
To address these challenges, we introduce FlockMTL: an extension for DBMSs that deeply integrates LLM capabilities and retrieval-augmented generation (RAG). FlockMTL includes model-driven scalar and aggregate functions, enabling chained predictions through tuple-level mappings and reductions. Drawing inspiration from the relational model, FlockMTL incorporates: (i) cost-based optimizations, which seamlessly apply techniques such as batching and caching; and (ii) resource independence, enabled through novel SQL DDL abstractions: PROMPT and MODEL, introduced as first-class schema objects alongside TABLE. FlockMTL streamlines the development of knowledge-intensive analytical applications, and its optimizations ease the implementation burden. 

---
