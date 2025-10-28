# Think before Recommendation: Autonomous Reasoning-enhanced Recommender 

**Authors**: Xiaoyu Kong, Junguang Jiang, Bin Liu, Ziru Xu, Han Zhu, Jian Xu, Bo Zheng, Jiancan Wu, Xiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.23077)  

**Abstract**: The core task of recommender systems is to learn user preferences from historical user-item interactions. With the rapid development of large language models (LLMs), recent research has explored leveraging the reasoning capabilities of LLMs to enhance rating prediction tasks. However, existing distillation-based methods suffer from limitations such as the teacher model's insufficient recommendation capability, costly and static supervision, and superficial transfer of reasoning ability. To address these issues, this paper proposes RecZero, a reinforcement learning (RL)-based recommendation paradigm that abandons the traditional multi-model and multi-stage distillation approach. Instead, RecZero trains a single LLM through pure RL to autonomously develop reasoning capabilities for rating prediction. RecZero consists of two key components: (1) "Think-before-Recommendation" prompt construction, which employs a structured reasoning template to guide the model in step-wise analysis of user interests, item features, and user-item compatibility; and (2) rule-based reward modeling, which adopts group relative policy optimization (GRPO) to compute rewards for reasoning trajectories and optimize the LLM. Additionally, the paper explores a hybrid paradigm, RecOne, which combines supervised fine-tuning with RL, initializing the model with cold-start reasoning samples and further optimizing it with RL. Experimental results demonstrate that RecZero and RecOne significantly outperform existing baseline methods on multiple benchmark datasets, validating the superiority of the RL paradigm in achieving autonomous reasoning-enhanced recommender systems. 

---
# Multi-Stage Field Extraction of Financial Documents with OCR and Compact Vision-Language Models 

**Authors**: Yichao Jin, Yushuo Wang, Qishuai Zhong, Kent Chiu Jin-Chun, Kenneth Zhu Ke, Donald MacDonald  

**Link**: [PDF](https://arxiv.org/pdf/2510.23066)  

**Abstract**: Financial documents are essential sources of information for regulators, auditors, and financial institutions, particularly for assessing the wealth and compliance of Small and Medium-sized Businesses. However, SMB documents are often difficult to parse. They are rarely born digital and instead are distributed as scanned images that are none machine readable. The scans themselves are low in resolution, affected by skew or rotation, and often contain noisy backgrounds. These documents also tend to be heterogeneous, mixing narratives, tables, figures, and multilingual content within the same report. Such characteristics pose major challenges for automated information extraction, especially when relying on end to end large Vision Language Models, which are computationally expensive, sensitive to noise, and slow when applied to files with hundreds of pages.
We propose a multistage pipeline that leverages traditional image processing models and OCR extraction, together with compact VLMs for structured field extraction of large-scale financial documents. Our approach begins with image pre-processing, including segmentation, orientation detection, and size normalization. Multilingual OCR is then applied to recover page-level text. Upon analyzing the text information, pages are retrieved for coherent sections. Finally, compact VLMs are operated within these narrowed-down scopes to extract structured financial indicators.
Our approach is evaluated using an internal corpus of multi-lingual, scanned financial documents. The results demonstrate that compact VLMs, together with a multistage pipeline, achieves 8.8 times higher field level accuracy relative to directly feeding the whole document into large VLMs, only at 0.7 percent of the GPU cost and 92.6 percent less end-to-end service latency. 

---
# Improving Product Search Relevance with EAR-MP: A Solution for the CIKM 2025 AnalytiCup 

**Authors**: JaeEun Lim, Soomin Kim, Jaeyong Seo, Iori Ono, Qimu Ran, Jae-woong Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.23018)  

**Abstract**: Multilingual e-commerce search is challenging due to linguistic diversity and the noise inherent in user-generated queries. This paper documents the solution employed by our team (EAR-MP) for the CIKM 2025 AnalytiCup, which addresses two core tasks: Query-Category (QC) relevance and Query-Item (QI) relevance. Our approach first normalizes the multilingual dataset by translating all text into English, then mitigates noise through extensive data cleaning and normalization. For model training, we build on DeBERTa-v3-large and improve performance with label smoothing, self-distillation, and dropout. In addition, we introduce task-specific upgrades, including hierarchical token injection for QC and a hybrid scoring mechanism for QI. Under constrained compute, our method achieves competitive results, attaining an F1 score of 0.8796 on QC and 0.8744 on QI. These findings underscore the importance of systematic data preprocessing and tailored training strategies for building robust, resource-efficient multilingual relevance systems. 

---
# MGFRec: Towards Reinforced Reasoning Recommendation with Multiple Groundings and Feedback 

**Authors**: Shihao Cai, Chongming Gao, Haoyan Liu, Wentao Shi, Jianshan Sun, Ruiming Tang, Fuli Feng  

**Link**: [PDF](https://arxiv.org/pdf/2510.22888)  

**Abstract**: The powerful reasoning and generative capabilities of large language models (LLMs) have inspired researchers to apply them to reasoning-based recommendation tasks, which require in-depth reasoning about user interests and the generation of recommended items. However, previous reasoning-based recommendation methods have typically performed inference within the language space alone, without incorporating the actual item space. This has led to over-interpreting user interests and deviating from real items. Towards this research gap, we propose performing multiple rounds of grounding during inference to help the LLM better understand the actual item space, which could ensure that its reasoning remains aligned with real items. Furthermore, we introduce a user agent that provides feedback during each grounding step, enabling the LLM to better recognize and adapt to user interests. Comprehensive experiments conducted on three Amazon review datasets demonstrate the effectiveness of incorporating multiple groundings and feedback. These findings underscore the critical importance of reasoning within the actual item space, rather than being confined to the language space, for recommendation tasks. 

---
# Civic Ground Truth in News Recommenders: A Method for Public Value Scoring 

**Authors**: James Meese, Kyle Herbertson  

**Link**: [PDF](https://arxiv.org/pdf/2510.22865)  

**Abstract**: Research in news recommendation systems (NRS) continues to explore the best ways to integrate normative goals such as editorial objectives and public service values into existing systems. Prior efforts have incorporated expert input or audience feedback to quantify these values, laying the groundwork for more civic-minded recommender systems. This paper contributes to that trajectory, introducing a method for embedding civic values into NRS through large-scale, structured audience evaluations. The proposed civic ground truth approach aims to generate value-based labels through a nationally representative survey that are generalisable across a wider news corpus, using automated metadata enrichment. 

---
# REVISION:Reflective Intent Mining and Online Reasoning Auxiliary for E-commerce Visual Search System Optimization 

**Authors**: Yiwen Tang, Qiuyu Zhao, Zenghui Sun, Jinsong Lan, Xiaoyong Zhu, Bo Zheng, Kaifu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.22739)  

**Abstract**: In Taobao e-commerce visual search, user behavior analysis reveals a large proportion of no-click requests, suggesting diverse and implicit user intents. These intents are expressed in various forms and are difficult to mine and discover, thereby leading to the limited adaptability and lag in platform strategies. This greatly restricts users' ability to express diverse intents and hinders the scalability of the visual search system. This mismatch between user implicit intent expression and system response defines the User-SearchSys Intent Discrepancy. To alleviate the issue, we propose a novel framework REVISION. This framework integrates offline reasoning mining with online decision-making and execution, enabling adaptive strategies to solve implicit user demands. In the offline stage, we construct a periodic pipeline to mine discrepancies from historical no-click requests. Leveraging large models, we analyze implicit intent factors and infer optimal suggestions by jointly reasoning over query and product metadata. These inferred suggestions serve as actionable insights for refining platform strategies. In the online stage, REVISION-R1-3B, trained on the curated offline data, performs holistic analysis over query images and associated historical products to generate optimization plans and adaptively schedule strategies across the search pipeline. Our framework offers a streamlined paradigm for integrating large models with traditional search systems, enabling end-to-end intelligent optimization across information aggregation and user interaction. Experimental results demonstrate that our approach improves the efficiency of implicit intent mining from large-scale search logs and significantly reduces the no-click rate. 

---
# Diversification as Risk Minimization 

**Authors**: Rikiya Takehi, Fernando Diaz, Tetsuya Sakai  

**Link**: [PDF](https://arxiv.org/pdf/2510.22681)  

**Abstract**: Users tend to remember failures of a search session more than its many successes. This observation has led to work on search robustness, where systems are penalized if they perform very poorly on some queries. However, this principle of robustness has been overlooked within a single query. An ambiguous or underspecified query (e.g., ``jaguar'') can have several user intents, where popular intents often dominate the ranking, leaving users with minority intents unsatisfied. Although the diversification literature has long recognized this issue, existing metrics only model the average relevance across intents and provide no robustness guarantees. More surprisingly, we show theoretically and empirically that many well-known diversification algorithms are no more robust than a naive, non-diversified algorithm. To address this critical gap, we propose to frame diversification as a risk-minimization problem. We introduce VRisk, which measures the expected risk faced by the least-served fraction of intents in a query. Optimizing VRisk produces a robust ranking, reducing the likelihood of poor user experiences. We then propose VRisker, a fast greedy re-ranker with provable approximation guarantees. Finally, experiments on NTCIR INTENT-2, TREC Web 2012, and MovieLens show the vulnerability of existing methods. VRisker reduces worst-case intent failures by up to 33% with a minimal 2% drop in average performance. 

---
# Tools are under-documented: Simple Document Expansion Boosts Tool Retrieval 

**Authors**: Xuan Lu, Haohang Huang, Rui Meng, Yaohui Jin, Wenjun Zeng, Xiaoyu Shen  

**Link**: [PDF](https://arxiv.org/pdf/2510.22670)  

**Abstract**: Large Language Models (LLMs) have recently demonstrated strong capabilities in tool use, yet progress in tool retrieval remains hindered by incomplete and heterogeneous tool documentation. To address this challenge, we introduce Tool-DE, a new benchmark and framework that systematically enriches tool documentation with structured fields to enable more effective tool retrieval, together with two dedicated models, Tool-Embed and Tool-Rank. We design a scalable document expansion pipeline that leverages both open- and closed-source LLMs to generate, validate, and refine enriched tool profiles at low cost, producing large-scale corpora with 50k instances for embedding-based retrievers and 200k for rerankers. On top of this data, we develop two models specifically tailored for tool retrieval: Tool-Embed, a dense retriever, and Tool-Rank, an LLM-based reranker. Extensive experiments on ToolRet and Tool-DE demonstrate that document expansion substantially improves retrieval performance, with Tool-Embed and Tool-Rank achieving new state-of-the-art results on both benchmarks. We further analyze the contribution of individual fields to retrieval effectiveness, as well as the broader impact of document expansion on both training and evaluation. Overall, our findings highlight both the promise and limitations of LLM-driven document expansion, positioning Tool-DE, along with the proposed Tool-Embed and Tool-Rank, as a foundation for future research in tool retrieval. 

---
# PaperAsk: A Benchmark for Reliability Evaluation of LLMs in Paper Search and Reading 

**Authors**: Yutao Wu, Xiao Liu, Yunhao Feng, Jiale Ding, Xingjun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2510.22242)  

**Abstract**: Large Language Models (LLMs) increasingly serve as research assistants, yet their reliability in scholarly tasks remains under-evaluated. In this work, we introduce PaperAsk, a benchmark that systematically evaluates LLMs across four key research tasks: citation retrieval, content extraction, paper discovery, and claim verification. We evaluate GPT-4o, GPT-5, and Gemini-2.5-Flash under realistic usage conditions-via web interfaces where search operations are opaque to the user. Through controlled experiments, we find consistent reliability failures: citation retrieval fails in 48-98% of multi-reference queries, section-specific content extraction fails in 72-91% of cases, and topical paper discovery yields F1 scores below 0.32, missing over 60% of relevant literature. Further human analysis attributes these failures to the uncontrolled expansion of retrieved context and the tendency of LLMs to prioritize semantically relevant text over task instructions. Across basic tasks, the LLMs display distinct failure behaviors: ChatGPT often withholds responses rather than risk errors, whereas Gemini produces fluent but fabricated answers. To address these issues, we develop lightweight reliability classifiers trained on PaperAsk data to identify unreliable outputs. PaperAsk provides a reproducible and diagnostic framework for advancing the reliability evaluation of LLM-based scholarly assistance systems. 

---
# Hybrid-Vector Retrieval for Visually Rich Documents: Combining Single-Vector Efficiency and Multi-Vector Accuracy 

**Authors**: Juyeon Kim, Geon Lee, Dongwon Choi, Taeuk Kim, Kijung Shin  

**Link**: [PDF](https://arxiv.org/pdf/2510.22215)  

**Abstract**: Retrieval over visually rich documents is essential for tasks such as legal discovery, scientific search, and enterprise knowledge management. Existing approaches fall into two paradigms: single-vector retrieval, which is efficient but coarse, and multi-vector retrieval, which is accurate but computationally expensive. To address this trade-off, we propose HEAVEN, a two-stage hybrid-vector framework. In the first stage, HEAVEN efficiently retrieves candidate pages using a single-vector method over Visually-Summarized Pages (VS-Pages), which assemble representative visual layouts from multiple pages. In the second stage, it reranks candidates with a multi-vector method while filtering query tokens by linguistic importance to reduce redundant computations. To evaluate retrieval systems under realistic conditions, we also introduce ViMDOC, the first benchmark for visually rich, multi-document, and long-document retrieval. Across four benchmarks, HEAVEN attains 99.87% of the Recall@1 performance of multi-vector models on average while reducing per-query computation by 99.82%, achieving efficiency and accuracy. Our code and datasets are available at: this https URL 

---
# Scaling Up Efficient Small Language Models Serving and Deployment for Semantic Job Search 

**Authors**: Kayhan Behdin, Qingquan Song, Sriram Vasudevan, Jian Sheng, Xiaojing Ma, Z Zhou, Chuanrui Zhu, Guoyao Li, Chanh Nguyen, Sayan Ghosh, Hejian Sang, Ata Fatahi Baarzi, Sundara Raman Ramachandran, Xiaoqing Wang, Qing Lan, Vinay Y S, Qi Guo, Caleb Johnson, Zhipeng Wang, Fedor Borisyuk  

**Link**: [PDF](https://arxiv.org/pdf/2510.22101)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive quality when applied to predictive tasks such as relevance ranking and semantic search. However, deployment of such LLMs remains prohibitively expensive for industry applications with strict latency and throughput requirements. In this work, we present lessons and efficiency insights from developing a purely text-based decoder-only Small Language Model (SLM) for a semantic search application at LinkedIn. Particularly, we discuss model compression techniques such as pruning that allow us to reduce the model size by up to $40\%$ while maintaining the accuracy. Additionally, we present context compression techniques that allow us to reduce the input context length by up to $10$x with minimal loss of accuracy. Finally, we present practical lessons from optimizing the serving infrastructure for deploying such a system on GPUs at scale, serving millions of requests per second. Taken together, this allows us to increase our system's throughput by $10$x in a real-world deployment, while meeting our quality bar. 

---
# A Benchmark for Open-Domain Numerical Fact-Checking Enhanced by Claim Decomposition 

**Authors**: V Venktesh, Deepali Prabhu, Avishek Anand  

**Link**: [PDF](https://arxiv.org/pdf/2510.22055)  

**Abstract**: Fact-checking numerical claims is critical as the presence of numbers provide mirage of veracity despite being fake potentially causing catastrophic impacts on society. The prior works in automatic fact verification do not primarily focus on natural numerical claims. A typical human fact-checker first retrieves relevant evidence addressing the different numerical aspects of the claim and then reasons about them to predict the veracity of the claim. Hence, the search process of a human fact-checker is a crucial skill that forms the foundation of the verification process. Emulating a real-world setting is essential to aid in the development of automated methods that encompass such skills. However, existing benchmarks employ heuristic claim decomposition approaches augmented with weakly supervised web search to collect evidences for verifying claims. This sometimes results in less relevant evidences and noisy sources with temporal leakage rendering a less realistic retrieval setting for claim verification. Hence, we introduce QuanTemp++: a dataset consisting of natural numerical claims, an open domain corpus, with the corresponding relevant evidence for each claim. The evidences are collected through a claim decomposition process approximately emulating the approach of human fact-checker and veracity labels ensuring there is no temporal leakage. Given this dataset, we also characterize the retrieval performance of key claim decomposition paradigms. Finally, we observe their effect on the outcome of the verification pipeline and draw insights. The code for data pipeline along with link to data can be found at this https URL 

---
# Massive Memorization with Hundreds of Trillions of Parameters for Sequential Transducer Generative Recommenders 

**Authors**: Zhimin Chen, Chenyu Zhao, Ka Chun Mo, Yunjiang Jiang, Jane H. Lee, Shouwei Chen, Khushhall Chandra Mahajan, Ning Jiang, Kai Ren, Jinhui Li, Wen-Yun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.22049)  

**Abstract**: Modern large-scale recommendation systems rely heavily on user interaction history sequences to enhance the model performance. The advent of large language models and sequential modeling techniques, particularly transformer-like architectures, has led to significant advancements recently (e.g., HSTU, SIM, and TWIN models). While scaling to ultra-long user histories (10k to 100k items) generally improves model performance, it also creates significant challenges on latency, queries per second (QPS) and GPU cost in industry-scale recommendation systems. Existing models do not adequately address these industrial scalability issues. In this paper, we propose a novel two-stage modeling framework, namely VIrtual Sequential Target Attention (VISTA), which decomposes traditional target attention from a candidate item to user history items into two distinct stages: (1) user history summarization into a few hundred tokens; followed by (2) candidate item attention to those tokens. These summarization token embeddings are then cached in storage system and then utilized as sequence features for downstream model training and inference. This novel design for scalability enables VISTA to scale to lifelong user histories (up to one million items) while keeping downstream training and inference costs fixed, which is essential in industry. Our approach achieves significant improvements in offline and online metrics and has been successfully deployed on an industry leading recommendation platform serving billions of users. 

---
# Multimodal Item Scoring for Natural Language Recommendation via Gaussian Process Regression with LLM Relevance Judgments 

**Authors**: Yifan Liu, Qianfeng Wen, Jiazhou Liang, Mark Zhao, Justin Cui, Anton Korikov, Armin Torogh, Junyoung Kim, Scott Sanner  

**Link**: [PDF](https://arxiv.org/pdf/2510.22023)  

**Abstract**: Natural Language Recommendation (NLRec) generates item suggestions based on the relevance between user-issued NL requests and NL item description passages. Existing NLRec approaches often use Dense Retrieval (DR) to compute item relevance scores from aggregation of inner products between user request embeddings and relevant passage embeddings. However, DR views the request as the sole relevance label, thus leading to a unimodal scoring function centered on the query embedding that is often a weak proxy for query relevance. To better capture the potential multimodal distribution of the relevance scoring function that may arise from complex NLRec data, we propose GPR-LLM that uses Gaussian Process Regression (GPR) with LLM relevance judgments for a subset of candidate passages. Experiments on four NLRec datasets and two LLM backbones demonstrate that GPR-LLM with an RBF kernel, capable of modeling multimodal relevance scoring functions, consistently outperforms simpler unimodal kernels (dot product, cosine similarity), as well as baseline methods including DR, cross-encoder, and pointwise LLM-based relevance scoring by up to 65%. Overall, GPR-LLM provides an efficient and effective approach to NLRec within a minimal LLM labeling budget. 

---
# Temporal Graph Theoretic Analysis of Geopolitical Dynamics in the U.S. Entity List 

**Authors**: Yunsen Lei, Kexin Bai, Quan Li, H. Howie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.21962)  

**Abstract**: Export controls have become one of America's most prominent tools of economic statecraft. They aim to block rival countries' access to sensitive technologies, safeguard U.S. supply chains, protect national security, and shape geopolitical competition. Among various instruments, the U.S. Entity List has emerged as the most salient, yet its dynamics remain underexplored. This paper introduces a novel temporal graph framework that transforms the Entity List documents from a static registry of foreign entities of concern into a dynamic representation of geopolitical strategy. We construct the first event-based dataset of U.S. government foreign entity designations and model them as a temporal bipartite graph. Building on this representation, we develop a multi-level analytical approach that reveals shifting roles, enforcement strategy, and broader sanction ecosystems. Applied to 25 years of data, the framework uncovers dynamic patterns of escalation, persistence, and coordination that static views cannot capture. More broadly, our study demonstrates how temporal graph analysis offers systematic computational insights into the geopolitical dynamics of export controls. 

---
# Development of an Automated Web Application for Efficient Web Scraping: Design and Implementation 

**Authors**: Alok Dutta, Nilanjana Roy, Rhythm Sen, Sougata Dutta, Prabhat Das  

**Link**: [PDF](https://arxiv.org/pdf/2510.21831)  

**Abstract**: This paper presents the design and implementation of a user-friendly, automated web application that simplifies and optimizes the web scraping process for non-technical users. The application breaks down the complex task of web scraping into three main stages: fetching, extraction, and execution. In the fetching stage, the application accesses target websites using the HTTP protocol, leveraging the requests library to retrieve HTML content. The extraction stage utilizes powerful parsing libraries like BeautifulSoup and regular expressions to extract relevant data from the HTML. Finally, the execution stage structures the data into accessible formats, such as CSV, ensuring the scraped content is organized for easy use. To provide personalized and secure experiences, the application includes user registration and login functionalities, supported by MongoDB, which stores user data and scraping history. Deployed using the Flask framework, the tool offers a scalable, robust environment for web scraping. Users can easily input website URLs, define data extraction parameters, and download the data in a simplified format, without needing technical expertise. This automated tool not only enhances the efficiency of web scraping but also democratizes access to data extraction by empowering users of all technical levels to gather and manage data tailored to their needs. The methodology detailed in this paper represents a significant advancement in making web scraping tools accessible, efficient, and easy to use for a broader audience. 

---
# Unifying Inductive, Cross-Domain, and Multimodal Learning for Robust and Generalizable Recommendation 

**Authors**: Chanyoung Chung, Kyeongryul Lee, Sunbin Park, Joyce Jiyoung Whang  

**Link**: [PDF](https://arxiv.org/pdf/2510.21812)  

**Abstract**: Recommender systems have long been built upon the modeling of interactions between users and items, while recent studies have sought to broaden this paradigm by generalizing to new users and items, incorporating diverse information sources, and transferring knowledge across domains. Nevertheless, these efforts have largely focused on individual aspects, hindering their ability to tackle the complex recommendation scenarios that arise in daily consumptions across diverse domains. In this paper, we present MICRec, a unified framework that fuses inductive modeling, multimodal guidance, and cross-domain transfer to capture user contexts and latent preferences in heterogeneous and incomplete real-world data. Moving beyond the inductive backbone of INMO, our model refines expressive representations through modality-based aggregation and alleviates data sparsity by leveraging overlapping users as anchors across domains, thereby enabling robust and generalizable recommendation. Experiments show that MICRec outperforms 12 baselines, with notable gains in domains with limited training data. 

---
# DiffGRM: Diffusion-based Generative Recommendation Model 

**Authors**: Zhao Liu, Yichen Zhu, Yiqing Yang, Guoping Tang, Rui Huang, Qiang Luo, Xiao Lv, Ruiming Tang, Kun Gai, Guorui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.21805)  

**Abstract**: Generative recommendation (GR) is an emerging paradigm that represents each item via a tokenizer as an n-digit semantic ID (SID) and predicts the next item by autoregressively generating its SID conditioned on the user's history. However, two structural properties of SIDs make ARMs ill-suited. First, intra-item consistency: the n digits jointly specify one item, yet the left-to-right causality trains each digit only under its prefix and blocks bidirectional cross-digit evidence, collapsing supervision to a single causal path. Second, inter-digit heterogeneity: digits differ in semantic granularity and predictability, while the uniform next-token objective assigns equal weight to all digits, overtraining easy digits and undertraining hard digits. To address these two issues, we propose DiffGRM, a diffusion-based GR model that replaces the autoregressive decoder with a masked discrete diffusion model (MDM), thereby enabling bidirectional context and any-order parallel generation of SID digits for recommendation. Specifically, we tailor DiffGRM in three aspects: (1) tokenization with Parallel Semantic Encoding (PSE) to decouple digits and balance per-digit information; (2) training with On-policy Coherent Noising (OCN) that prioritizes uncertain digits via coherent masking to concentrate supervision on high-value signals; and (3) inference with Confidence-guided Parallel Denoising (CPD) that fills higher-confidence digits first and generates diverse Top-K candidates. Experiments show consistent gains over strong generative and discriminative recommendation baselines on multiple datasets, improving NDCG@10 by 6.9%-15.5%. Code is available at this https URL. 

---
# From Factoid Questions to Data Product Requests: Benchmarking Data Product Discovery over Tables and Text 

**Authors**: Liangliang Zhang, Nandana Mihindukulasooriya, Niharika S. D'Souza, Sola Shirai, Sarthak Dash, Yao Ma, Horst Samulowitz  

**Link**: [PDF](https://arxiv.org/pdf/2510.21737)  

**Abstract**: Data products are reusable, self-contained assets designed for specific business use cases. Automating their discovery and generation is of great industry interest, as it enables discovery in large data lakes and supports analytical Data Product Requests (DPRs). Currently, there is no benchmark established specifically for data product discovery. Existing datasets focus on answering single factoid questions over individual tables rather than collecting multiple data assets for broader, coherent products. To address this gap, we introduce DPBench, the first user-request-driven data product benchmark over hybrid table-text corpora. Our framework systematically repurposes existing table-text QA datasets by clustering related tables and passages into coherent data products, generating professional-level analytical requests that span both data sources, and validating benchmark quality through multi-LLM evaluation. DPBench preserves full provenance while producing actionable, analyst-like data product requests. Baseline experiments with hybrid retrieval methods establish the feasibility of DPR evaluation, reveal current limitations, and point to new opportunities for automatic data product discovery research.
Code and datasets are available at: this https URL 

---
# Augmenting Researchy Questions with Sub-question Judgments 

**Authors**: Jia-Huei Ju, Eugene Yang, Trevor Adriaanse, Andrew Yates  

**Link**: [PDF](https://arxiv.org/pdf/2510.21733)  

**Abstract**: The Researchy Questions dataset provides about 100k question queries with complex information needs that require retrieving information about several aspects of a topic. Each query in ResearchyQuestions is associated with sub-questions that were produced by prompting GPT-4. While ResearchyQuestions contains labels indicating what documents were clicked after issuing the query, there are no associations in the dataset between sub-questions and relevant documents. In this work, we augment the Researchy Questions dataset with LLM-judged labels for each sub-question using a Llama3.3 70B model. We intend these sub-question labels to serve as a resource for training retrieval models that better support complex information needs. 

---
# TriMat: Context-aware Recommendation by Tri-Matrix Factorization 

**Authors**: Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.21730)  

**Abstract**: Search engine is the symbolic technology of Web 2.0, and many people used to believe recommender systems is the new frontier of Web 3.0. In the past 10 years, with the advent of TikTok and similar apps, recommender systems has materialized the vision of the machine learning pioneers. However, many research topics of the field remain unfixed until today. One such topic is CARS (Context-aware Recommender Systems) , which is largely a theoretical topic without much advance in real-world applications. In this paper, we utilize tri-matrix factorization technique to incorporate contextual information into our matrix factorization framework, and prove that our technique is effective in improving both the accuracy and fairness metrics in our experiments. 

---
# CustomIR: Unsupervised Fine-Tuning of Dense Embeddings for Known Document Corpora 

**Authors**: Nathan Paull  

**Link**: [PDF](https://arxiv.org/pdf/2510.21729)  

**Abstract**: Dense embedding models have become critical for modern information retrieval, particularly in RAG pipelines, but their performance often degrades when applied to specialized corpora outside their pre-training distribution. To address thi we introduce \textbf{CustomIR}, a framework for unsupervised adaptation of pre-trained language embedding models to domain-specific corpora using synthetically generated query-document pairs. CustomIR leverages large language models (LLMs) to create diverse queries grounded in a known target corpus, paired with LLM-verified hard negatives, eliminating the need for costly human annotation. Experiments on enterprise email and messaging datasets show that CustomIR consistently improves retrieval effectiveness with small models gaining up to 2.3 points in Recall@10. This performance increase allows these small models to rival the performance of much larger alternatives, allowing for cheaper RAG deployments. These results highlight that targeted synthetic fine-tuning offers a scalable and cost-efficient strategy for increasing domain-specific performance. 

---
# Modeling Bias Evolution in Fashion Recommender Systems: A System Dynamics Approach 

**Authors**: Mahsa Goodarzi, M. Abdullah Canbaz  

**Link**: [PDF](https://arxiv.org/pdf/2510.21728)  

**Abstract**: Bias in recommender systems not only distorts user experience but also perpetuates and amplifies existing societal stereotypes, particularly in sectors like fashion e-commerce. This study employs a dynamic modeling approach to scrutinize the mechanisms of bias activation and reinforcement within Fashion Recommender Systems (FRS). By leveraging system dynamics modeling and experimental simulations, we dissect the temporal evolution of bias and its multifaceted impacts on system performance. Our analysis reveals that inductive biases exert a more substantial influence on system outcomes than user biases, suggesting critical areas for intervention. We demonstrate that while current debiasing strategies, including data rebalancing and algorithmic regularization, are effective to an extent, they require further enhancement to comprehensively mitigate biases. This research underscores the necessity for advancing these strategies and extending system boundaries to incorporate broader contextual factors such as user demographics and item diversity, aiming to foster inclusivity and fairness in FRS. The findings advocate for a proactive approach in recommender system design to counteract bias propagation and ensure equitable user experiences. 

---
# Your Dense Retriever is Secretly an Expeditious Reasoner 

**Authors**: Yichi Zhang, Jun Bai, Zhixin Cai, Shuhan Qin, Zhuofan Chen, Jinghua Guan, Wenge Rong  

**Link**: [PDF](https://arxiv.org/pdf/2510.21727)  

**Abstract**: Dense retrievers enhance retrieval by encoding queries and documents into continuous vectors, but they often struggle with reasoning-intensive queries. Although Large Language Models (LLMs) can reformulate queries to capture complex reasoning, applying them universally incurs significant computational cost. In this work, we propose Adaptive Query Reasoning (AdaQR), a hybrid query rewriting framework. Within this framework, a Reasoner Router dynamically directs each query to either fast dense reasoning or deep LLM reasoning. The dense reasoning is achieved by the Dense Reasoner, which performs LLM-style reasoning directly in the embedding space, enabling a controllable trade-off between efficiency and accuracy. Experiments on large-scale retrieval benchmarks BRIGHT show that AdaQR reduces reasoning cost by 28% while preserving-or even improving-retrieval performance by 7%. 

---
# From Authors to Reviewers: Leveraging Rankings to Improve Peer Review 

**Authors**: Weichen Wang, Chengchun Shi  

**Link**: [PDF](https://arxiv.org/pdf/2510.21726)  

**Abstract**: This paper is a discussion of the 2025 JASA discussion paper by Su et al. (2025). We would like to congratulate the authors on conducting a comprehensive and insightful empirical investigation of the 2023 ICML ranking data. The review quality of machine learning (ML) conferences has become a big concern in recent years, due to the rapidly growing number of submitted manuscripts. In this discussion, we propose an approach alternative to Su et al. (2025) that leverages ranking information from reviewers rather than authors. We simulate review data that closely mimics the 2023 ICML conference submissions. Our results show that (i) incorporating ranking information from reviewers can significantly improve the evaluation of each paper's quality, often outperforming the use of ranking information from authors alone; and (ii) combining ranking information from both reviewers and authors yields the most accurate evaluation of submitted papers in most scenarios. 

---
# Words to Waves: Emotion-Adaptive Music Recommendation System 

**Authors**: Apoorva Chavali, Reeve Menezes  

**Link**: [PDF](https://arxiv.org/pdf/2510.21724)  

**Abstract**: Current recommendation systems often tend to overlook emotional context and rely on historical listening patterns or static mood tags. This paper introduces a novel music recommendation framework employing a variant of Wide and Deep Learning architecture that takes in real-time emotional states inferred directly from natural language as inputs and recommends songs that closely portray the mood. The system captures emotional contexts from user-provided textual descriptions by using transformer-based embeddings, which were finetuned to predict the emotional dimensions of valence-arousal. The deep component of the architecture utilizes these embeddings to generalize unseen emotional patterns, while the wide component effectively memorizes user-emotion and emotion-genre associations through cross-product features. Experimental results show that personalized music selections positively influence the user's emotions and lead to a significant improvement in emotional relevance. 

---
# Practice on Long Behavior Sequence Modeling in Tencent Advertising 

**Authors**: Xian Hu, Ming Yue, Zhixiang Feng, Junwei Pan, Junjie Zhai, Ximei Wang, Xinrui Miao, Qian Li, Xun Liu, Shangyu Zhang, Letian Wang, Hua Lu, Zijian Zeng, Chen Cai, Wei Wang, Fei Xiong, Pengfei Xiong, Jintao Zhang, Zhiyuan Wu, Chunhui Zhang, Anan Liu, Jiulong You, Chao Deng, Yuekui Yang, Shudong Huang, Dapeng Liu, Haijie Gu  

**Link**: [PDF](https://arxiv.org/pdf/2510.21714)  

**Abstract**: Long-sequence modeling has become an indispensable frontier in recommendation systems for capturing users' long-term preferences. However, user behaviors within advertising domains are inherently sparse, posing a significant barrier to constructing long behavioral sequences using data from a single advertising domain alone. This motivates us to collect users' behaviors not only across diverse advertising scenarios, but also beyond the boundaries of the advertising domain into content domains-thereby constructing unified commercial behavior trajectories. This cross-domain or cross-scenario integration gives rise to the following challenges: (1) feature taxonomy gaps between distinct scenarios and domains, (2) inter-field interference arising from irrelevant feature field pairs, and (3) target-wise interference in temporal and semantic patterns when optimizing for different advertising targets. To address these challenges, we propose several practical approaches within the two-stage framework for long-sequence modeling. In the first (search) stage, we design a hierarchical hard search method for handling complex feature taxonomy hierarchies, alongside a decoupled embedding-based soft search to alleviate conflicts between attention mechanisms and feature representation. In the second (sequence modeling) stage, we introduce: (a) Decoupled Side Information Temporal Interest Networks (TIN) to mitigate inter-field conflicts; (b) Target-Decoupled Positional Encoding and Target-Decoupled SASRec to address target-wise interference; and (c) Stacked TIN to model high-order behavioral correlations. Deployed in production on Tencent's large-scale advertising platforms, our innovations delivered significant performance gains: an overall 4.22% GMV lift in WeChat Channels and an overall 1.96% GMV increase in WeChat Moments. 

---
# asLLR: LLM based Leads Ranking in Auto Sales 

**Authors**: Yin Sun, Yiwen Liu, Junjie Song, Chenyu Zhang, Xinyuan Zhang, Lingjie Liu, Siqi Chen, Yuji Cao  

**Link**: [PDF](https://arxiv.org/pdf/2510.21713)  

**Abstract**: In the area of commercial auto sales system, high-quality lead score sequencing determines the priority of a sale's work and is essential for optimizing the efficiency of the sales system. Since CRM (Customer Relationship Management) system contains plenty of textual interaction features between sales and customers, traditional techniques such as Click Through Rate (CTR) prediction struggle with processing the complex information inherent in natural language features, which limits their effectiveness in sales lead ranking. Bridging this gap is critical for enhancing business intelligence and decision-making. Recently, the emergence of large language models (LLMs) has opened new avenues for improving recommendation systems, this study introduces asLLR (LLM-based Leads Ranking in Auto Sales), which integrates CTR loss and Question Answering (QA) loss within a decoder-only large language model architecture. This integration enables the simultaneous modeling of both tabular and natural language features. To verify the efficacy of asLLR, we constructed an innovative dataset derived from the customer lead pool of a prominent new energy vehicle brand, with 300,000 training samples and 40,000 testing samples. Our experimental results demonstrate that asLLR effectively models intricate patterns in commercial datasets, achieving the AUC of 0.8127, surpassing traditional CTR estimation methods by 0.0231. Moreover, asLLR enhances CTR models when used for extracting text features by 0.0058. In real-world sales scenarios, after rigorous online A/B testing, asLLR increased the sales volume by about 9.5% compared to the traditional method, providing a valuable tool for business intelligence and operational decision-making. 

---
# DecoupleSearch: Decouple Planning and Search via Hierarchical Reward Modeling 

**Authors**: Hao Sun, Zile Qiao, Bo Wang, Guoxin Chen, Yingyan Hou, Yong Jiang, Pengjun Xie, Fei Huang, Yan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.21712)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems have emerged as a pivotal methodology for enhancing Large Language Models (LLMs) through the dynamic integration of external knowledge. To further improve RAG's flexibility, Agentic RAG introduces autonomous agents into the workflow. However, Agentic RAG faces several challenges: (1) the success of each step depends on both high-quality planning and accurate search, (2) the lack of supervision for intermediate reasoning steps, and (3) the exponentially large candidate space for planning and searching. To address these challenges, we propose DecoupleSearch, a novel framework that decouples planning and search processes using dual value models, enabling independent optimization of plan reasoning and search grounding. Our approach constructs a reasoning tree, where each node represents planning and search steps. We leverage Monte Carlo Tree Search to assess the quality of each step. During inference, Hierarchical Beam Search iteratively refines planning and search candidates with dual value models. Extensive experiments across policy models of varying parameter sizes, demonstrate the effectiveness of our method. 

---
# Improving E-commerce Search with Category-Aligned Retrieval 

**Authors**: Rauf Aliev  

**Link**: [PDF](https://arxiv.org/pdf/2510.21711)  

**Abstract**: Traditional e-commerce search systems often struggle with the semantic gap between user queries and product catalogs. In this paper, we propose a Category-Aligned Retrieval System (CARS) that improves search relevance by first predicting the product category from a user's query and then boosting products within that category. We introduce a novel method for creating "Trainable Category Prototypes" from query embeddings. We evaluate this method with two models: a lightweight all-MiniLM-L6-v2 and OpenAI's text-embedding-ada-002. Our offline evaluation shows this method is highly effective, with the OpenAI model increasing Top-3 category prediction accuracy from a zero-shot baseline of 43.8% to 83.2% after training. The end-to-end simulation, however, highlights the limitations of blindly applying category boosts in a complex retrieval pipeline: while accuracy is high, naive integration can negatively affect search relevance metrics such as nDCG@10. We argue that this is partly due to dataset-specific ambiguities (e.g., polysemous queries in the Amazon ESCI corpus) and partly due to the sensitivity of retrieval systems to over-constraining filters. Crucially, these results do not diminish the value of the approach; rather, they emphasize the need for confidence-aware and adaptive integration strategies. 

---
# LimRank: Less is More for Reasoning-Intensive Information Reranking 

**Authors**: Tingyu Song, Yilun Zhao, Siyue Zhang, Chen Zhao, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2510.23544)  

**Abstract**: Existing approaches typically rely on large-scale fine-tuning to adapt LLMs for information reranking tasks, which is computationally expensive. In this work, we demonstrate that modern LLMs can be effectively adapted using only minimal, high-quality supervision. To enable this, we design LIMRANK-SYNTHESIZER, a reusable and open-source pipeline for generating diverse, challenging, and realistic reranking examples. Using this synthetic data, we fine-tune our reranker model, LIMRANK. We evaluate LIMRANK on two challenging benchmarks, i.e., BRIGHT for reasoning-intensive retrieval and FollowIR for instruction-following retrieval. Our experiments demonstrate that LIMRANK achieves competitive performance, while being trained on less than 5% of the data typically used in prior work. Further ablation studies demonstrate the effectiveness of LIMRANK-SYNTHESIZER and the strong generalization capabilities of LIMRANK across downstream tasks, including scientific literature search and retrieval-augmented generation for knowledge-intensive problem solving. 

---
# Accurate and Scalable Multimodal Pathology Retrieval via Attentive Vision-Language Alignment 

**Authors**: Hongyi Wang, Zhengjie Zhu, Jiabo Ma, Fang Wang, Yue Shi, Bo Luo, Jili Wang, Qiuyu Cai, Xiuming Zhang, Yen-Wei Chen, Lanfen Lin, Hao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.23224)  

**Abstract**: The rapid digitization of histopathology slides has opened up new possibilities for computational tools in clinical and research workflows. Among these, content-based slide retrieval stands out, enabling pathologists to identify morphologically and semantically similar cases, thereby supporting precise diagnoses, enhancing consistency across observers, and assisting example-based education. However, effective retrieval of whole slide images (WSIs) remains challenging due to their gigapixel scale and the difficulty of capturing subtle semantic differences amid abundant irrelevant content. To overcome these challenges, we present PathSearch, a retrieval framework that unifies fine-grained attentive mosaic representations with global-wise slide embeddings aligned through vision-language contrastive learning. Trained on a corpus of 6,926 slide-report pairs, PathSearch captures both fine-grained morphological cues and high-level semantic patterns to enable accurate and flexible retrieval. The framework supports two key functionalities: (1) mosaic-based image-to-image retrieval, ensuring accurate and efficient slide research; and (2) multi-modal retrieval, where text queries can directly retrieve relevant slides. PathSearch was rigorously evaluated on four public pathology datasets and three in-house cohorts, covering tasks including anatomical site retrieval, tumor subtyping, tumor vs. non-tumor discrimination, and grading across diverse organs such as breast, lung, kidney, liver, and stomach. External results show that PathSearch outperforms traditional image-to-image retrieval frameworks. A multi-center reader study further demonstrates that PathSearch improves diagnostic accuracy, boosts confidence, and enhances inter-observer agreement among pathologists in real clinical scenarios. These results establish PathSearch as a scalable and generalizable retrieval solution for digital pathology. 

---
# Leveraging Hierarchical Organization for Medical Multi-document Summarization 

**Authors**: Yi-Li Hsu, Katelyn X. Mei, Lucy Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.23104)  

**Abstract**: Medical multi-document summarization (MDS) is a complex task that requires effectively managing cross-document relationships. This paper investigates whether incorporating hierarchical structures in the inputs of MDS can improve a model's ability to organize and contextualize information across documents compared to traditional flat summarization methods. We investigate two ways of incorporating hierarchical organization across three large language models (LLMs), and conduct comprehensive evaluations of the resulting summaries using automated metrics, model-based metrics, and domain expert evaluation of preference, understandability, clarity, complexity, relevance, coverage, factuality, and coherence. Our results show that human experts prefer model-generated summaries over human-written summaries. Hierarchical approaches generally preserve factuality, coverage, and coherence of information, while also increasing human preference for summaries. Additionally, we examine whether simulated judgments from GPT-4 align with human judgments, finding higher agreement along more objective evaluation facets. Our findings demonstrate that hierarchical structures can improve the clarity of medical summaries generated by models while maintaining content coverage, providing a practical way to improve human preference for generated summaries. 

---
# Tagging-Augmented Generation: Assisting Language Models in Finding Intricate Knowledge In Long Contexts 

**Authors**: Anwesan Pal, Karen Hovsepian, Tinghao Guo, Mengnan Zhao, Somendra Tripathi, Nikos Kanakaris, George Mihaila, Sumit Nigam  

**Link**: [PDF](https://arxiv.org/pdf/2510.22956)  

**Abstract**: Recent investigations into effective context lengths of modern flagship large language models (LLMs) have revealed major limitations in effective question answering (QA) and reasoning over long and complex contexts for even the largest and most impressive cadre of models. While approaches like retrieval-augmented generation (RAG) and chunk-based re-ranking attempt to mitigate this issue, they are sensitive to chunking, embedding and retrieval strategies and models, and furthermore, rely on extensive pre-processing, knowledge acquisition and indexing steps. In this paper, we propose Tagging-Augmented Generation (TAG), a lightweight data augmentation strategy that boosts LLM performance in long-context scenarios, without degrading and altering the integrity and composition of retrieved documents. We validate our hypothesis by augmenting two challenging and directly relevant question-answering benchmarks -- NoLima and NovelQA -- and show that tagging the context or even just adding tag definitions into QA prompts leads to consistent performance gains over the baseline -- up to 17% for 32K token contexts, and 2.9% in complex reasoning question-answering for multi-hop queries requiring knowledge across a wide span of text. Additional details are available at this https URL. 

---
# GTR-Mamba: Geometry-to-Tangent Routing for Hyperbolic POI Recommendation 

**Authors**: Zhuoxuan Li, Jieyuan Pei, Tangwei Ye, Zhongyuan Lai, Zihan Liu, Fengyuan Xu, Qi Zhang, Liang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.22942)  

**Abstract**: Next Point-of-Interest (POI) recommendation is a critical task in modern Location-Based Social Networks (LBSNs), aiming to model the complex decision-making process of human mobility to provide personalized recommendations for a user's next check-in location. Existing POI recommendation models, predominantly based on Graph Neural Networks and sequential models, have been extensively studied. However, these models face a fundamental limitation: they struggle to simultaneously capture the inherent hierarchical structure of spatial choices and the dynamics and irregular shifts of user-specific temporal contexts. To overcome this limitation, we propose GTR-Mamba, a novel framework for cross-manifold conditioning and routing. GTR-Mamba leverages the distinct advantages of different mathematical spaces for different tasks: it models the static, tree-like preference hierarchies in hyperbolic geometry, while routing the dynamic sequence updates to a novel Mamba layer in the computationally stable and efficient Euclidean tangent space. This process is coordinated by a cross-manifold channel that fuses spatio-temporal information to explicitly steer the State Space Model (SSM), enabling flexible adaptation to contextual changes. Extensive experiments on three real-world datasets demonstrate that GTR-Mamba consistently outperforms state-of-the-art baseline models in next POI recommendation. 

---
# $\text{E}^2\text{Rank}$: Your Text Embedding can Also be an Effective and Efficient Listwise Reranker 

**Authors**: Qi Liu, Yanzhao Zhang, Mingxin Li, Dingkun Long, Pengjun Xie, Jiaxin Mao  

**Link**: [PDF](https://arxiv.org/pdf/2510.22733)  

**Abstract**: Text embedding models serve as a fundamental component in real-world search applications. By mapping queries and documents into a shared embedding space, they deliver competitive retrieval performance with high efficiency. However, their ranking fidelity remains limited compared to dedicated rerankers, especially recent LLM-based listwise rerankers, which capture fine-grained query-document and document-document interactions. In this paper, we propose a simple yet effective unified framework $\text{E}^2\text{Rank}$, means Efficient Embedding-based Ranking (also means Embedding-to-Rank), which extends a single text embedding model to perform both high-quality retrieval and listwise reranking through continued training under a listwise ranking objective, thereby achieving strong effectiveness with remarkable efficiency. By applying cosine similarity between the query and document embeddings as a unified ranking function, the listwise ranking prompt, which is constructed from the original query and its candidate documents, serves as an enhanced query enriched with signals from the top-K documents, akin to pseudo-relevance feedback (PRF) in traditional retrieval models. This design preserves the efficiency and representational quality of the base embedding model while significantly improving its reranking performance. Empirically, $\textrm{E}^2\text{Rank}$ achieves state-of-the-art results on the BEIR reranking benchmark and demonstrates competitive performance on the reasoning-intensive BRIGHT benchmark, with very low reranking latency. We also show that the ranking training process improves embedding performance on the MTEB benchmark. Our findings indicate that a single embedding model can effectively unify retrieval and reranking, offering both computational efficiency and competitive ranking accuracy. 

---
# ATLAS: Actor-Critic Task-Completion with Look-ahead Action Simulation 

**Authors**: Jiali Cheng, Anjishnu Kumar, Roshan Lal, Rishi Rajasekaran, Hani Ramezani, Omar Zia Khan, Oleg Rokhlenko, Sunny Chiu-Webster, Gang Hua, Hadi Amiri  

**Link**: [PDF](https://arxiv.org/pdf/2510.22732)  

**Abstract**: We observe that current state-of-the-art web-agents are unable to effectively adapt to new environments without neural network fine-tuning, without which they produce inefficient execution plans due to a lack of awareness of the structure and dynamics of the new environment. To address this limitation, we introduce ATLAS (Actor-Critic Task-completion with Look-ahead Action Simulation), a memory-augmented agent that is able to make plans grounded in a model of the environment by simulating the consequences of those actions in cognitive space. Our agent starts by building a "cognitive map" by performing a lightweight curiosity driven exploration of the environment. The planner proposes candidate actions; the simulator predicts their consequences in cognitive space; a critic analyzes the options to select the best roll-out and update the original plan; and a browser executor performs the chosen action. On the WebArena-Lite Benchmark, we achieve a 63% success rate compared to 53.9% success rate for the previously published state-of-the-art. Unlike previous systems, our modular architecture requires no website-specific LLM fine-tuning. Ablations show sizable drops without the world-model, hierarchical planner, and look-ahead-based replanner confirming their complementary roles within the design of our system 

---
# Windsock is Dancing: Adaptive Multimodal Retrieval-Augmented Generation 

**Authors**: Shu Zhao, Tianyi Shen, Nilesh Ahuja, Omesh Tickoo, Vijaykrishnan Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2510.22694)  

**Abstract**: Multimodal Retrieval-Augmented Generation (MRAG) has emerged as a promising method to generate factual and up-to-date responses of Multimodal Large Language Models (MLLMs) by incorporating non-parametric knowledge from external knowledge bases. However, existing MRAG approaches suffer from static retrieval strategies, inflexible modality selection, and suboptimal utilization of retrieved information, leading to three critical challenges: determining when to retrieve, what modality to incorporate, and how to utilize retrieved information effectively. To address these challenges, we introduce Windsock, a query-dependent module making decisions on retrieval necessity and modality selection, effectively reducing computational overhead and improving response quality. Additionally, we propose Dynamic Noise-Resistance (DANCE) Instruction Tuning, an adaptive training strategy that enhances MLLMs' ability to utilize retrieved information while maintaining robustness against noise. Moreover, we adopt a self-assessment approach leveraging knowledge within MLLMs to convert question-answering datasets to MRAG training datasets. Extensive experiments demonstrate that our proposed method significantly improves the generation quality by 17.07% while reducing 8.95% retrieval times. 

---
# ATOM: AdapTive and OptiMized dynamic temporal knowledge graph construction using LLMs 

**Authors**: Yassir Lairgi, Ludovic Moncla, Khalid Benabdeslem, Rémy Cazabet, Pierre Cléau  

**Link**: [PDF](https://arxiv.org/pdf/2510.22590)  

**Abstract**: In today's rapidly expanding data landscape, knowledge extraction from unstructured text is vital for real-time analytics, temporal inference, and dynamic memory frameworks. However, traditional static knowledge graph (KG) construction often overlooks the dynamic and time-sensitive nature of real-world data, limiting adaptability to continuous changes. Moreover, recent zero- or few-shot approaches that avoid domain-specific fine-tuning or reliance on prebuilt ontologies often suffer from instability across multiple runs, as well as incomplete coverage of key facts. To address these challenges, we introduce ATOM (AdapTive and OptiMized), a few-shot and scalable approach that builds and continuously updates Temporal Knowledge Graphs (TKGs) from unstructured texts. ATOM splits input documents into minimal, self-contained "atomic" facts, improving extraction exhaustivity and stability. Then, it constructs atomic TKGs from these facts while employing a dual-time modeling that distinguishes when information is observed from when it is valid. The resulting atomic TKGs are subsequently merged in parallel. Empirical evaluations demonstrate that ATOM achieves ~18% higher exhaustivity, ~17% better stability, and over 90% latency reduction compared to baseline methods, demonstrating a strong scalability potential for dynamic TKG construction. 

---
# Open Multimodal Retrieval-Augmented Factual Image Generation 

**Authors**: Yang Tian, Fan Liu, Jingyuan Zhang, Wei Bi, Yupeng Hu, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2510.22521)  

**Abstract**: Large Multimodal Models (LMMs) have achieved remarkable progress in generating photorealistic and prompt-aligned images, but they often produce outputs that contradict verifiable knowledge, especially when prompts involve fine-grained attributes or time-sensitive events. Conventional retrieval-augmented approaches attempt to address this issue by introducing external information, yet they are fundamentally incapable of grounding generation in accurate and evolving knowledge due to their reliance on static sources and shallow evidence integration. To bridge this gap, we introduce ORIG, an agentic open multimodal retrieval-augmented framework for Factual Image Generation (FIG), a new task that requires both visual realism and factual grounding. ORIG iteratively retrieves and filters multimodal evidence from the web and incrementally integrates the refined knowledge into enriched prompts to guide generation. To support systematic evaluation, we build FIG-Eval, a benchmark spanning ten categories across perceptual, compositional, and temporal dimensions. Experiments demonstrate that ORIG substantially improves factual consistency and overall image quality over strong baselines, highlighting the potential of open multimodal retrieval for factual image generation. 

---
# FAIR-RAG: Faithful Adaptive Iterative Refinement for Retrieval-Augmented Generation 

**Authors**: Mohammad Aghajani Asl, Majid Asgari-Bidhendi, Behrooz Minaei-Bidgoli  

**Link**: [PDF](https://arxiv.org/pdf/2510.22344)  

**Abstract**: While Retrieval-Augmented Generation (RAG) mitigates hallucination and knowledge staleness in Large Language Models (LLMs), existing frameworks often falter on complex, multi-hop queries that require synthesizing information from disparate sources. Current advanced RAG methods, employing iterative or adaptive strategies, lack a robust mechanism to systematically identify and fill evidence gaps, often propagating noise or failing to gather a comprehensive context. We introduce FAIR-RAG, a novel agentic framework that transforms the standard RAG pipeline into a dynamic, evidence-driven reasoning process. At its core is an Iterative Refinement Cycle governed by a module we term Structured Evidence Assessment (SEA). The SEA acts as an analytical gating mechanism: it deconstructs the initial query into a checklist of required findings and audits the aggregated evidence to identify confirmed facts and, critically, explicit informational gaps. These gaps provide a precise signal to an Adaptive Query Refinement agent, which generates new, targeted sub-queries to retrieve missing information. This cycle repeats until the evidence is verified as sufficient, ensuring a comprehensive context for a final, strictly faithful generation. We conducted experiments on challenging multi-hop QA benchmarks, including HotpotQA, 2WikiMultiHopQA, and MusiQue. In a unified experimental setup, FAIR-RAG significantly outperforms strong baselines. On HotpotQA, it achieves an F1-score of 0.453 -- an absolute improvement of 8.3 points over the strongest iterative baseline -- establishing a new state-of-the-art for this class of methods on these benchmarks. Our work demonstrates that a structured, evidence-driven refinement process with explicit gap analysis is crucial for unlocking reliable and accurate reasoning in advanced RAG systems for complex, knowledge-intensive tasks. 

---
# PatenTEB: A Comprehensive Benchmark and Model Family for Patent Text Embedding 

**Authors**: Iliass Ayaou, Denis Cavallucci  

**Link**: [PDF](https://arxiv.org/pdf/2510.22264)  

**Abstract**: Patent text embeddings enable prior art search, technology landscaping, and patent analysis, yet existing benchmarks inadequately capture patent-specific challenges. We introduce PatenTEB, a comprehensive benchmark comprising 15 tasks across retrieval, classification, paraphrase, and clustering, with 2.06 million examples. PatenTEB employs domain-stratified splits, domain specific hard negative mining, and systematic coverage of asymmetric fragment-to-document matching scenarios absent from general embedding benchmarks. We develop the patembed model family through multi-task training, spanning 67M to 344M parameters with context lengths up to 4096 tokens. External validation shows strong generalization: patembed-base achieves state-of-the-art on MTEB BigPatentClustering.v2 (0.494 V-measure vs. 0.445 previous best), while patembed-large achieves 0.377 NDCG@100 on DAPFAM. Systematic ablations reveal that multi-task training improves external generalization despite minor benchmark costs, and that domain-pretrained initialization provides consistent advantages across task families. All resources will be made available at this https URL. Keywords: patent retrieval, sentence embeddings, multi-task learning, asymmetric retrieval, benchmark evaluation, contrastive learning. 

---
# A Multi-Stage Hybrid Framework for Automated Interpretation of Multi-View Engineering Drawings Using Vision Language Model 

**Authors**: Muhammad Tayyab Khan, Zane Yong, Lequn Chen, Wenhe Feng, Nicholas Yew Jin Tan, Seung Ki Moon  

**Link**: [PDF](https://arxiv.org/pdf/2510.21862)  

**Abstract**: Engineering drawings are fundamental to manufacturing communication, serving as the primary medium for conveying design intent, tolerances, and production details. However, interpreting complex multi-view drawings with dense annotations remains challenging using manual methods, generic optical character recognition (OCR) systems, or traditional deep learning approaches, due to varied layouts, orientations, and mixed symbolic-textual content. To address these challenges, this paper proposes a three-stage hybrid framework for the automated interpretation of 2D multi-view engineering drawings using modern detection and vision language models (VLMs). In the first stage, YOLOv11-det performs layout segmentation to localize key regions such as views, title blocks, and notes. The second stage uses YOLOv11-obb for orientation-aware, fine-grained detection of annotations, including measures, GD&T symbols, and surface roughness indicators. The third stage employs two Donut-based, OCR-free VLMs for semantic content parsing: the Alphabetical VLM extracts textual and categorical information from title blocks and notes, while the Numerical VLM interprets quantitative data such as measures, GD&T frames, and surface roughness. Two specialized datasets were developed to ensure robustness and generalization: 1,000 drawings for layout detection and 1,406 for annotation-level training. The Alphabetical VLM achieved an overall F1 score of 0.672, while the Numerical VLM reached 0.963, demonstrating strong performance in textual and quantitative interpretation, respectively. The unified JSON output enables seamless integration with CAD and manufacturing databases, providing a scalable solution for intelligent engineering drawing analysis. 

---
# 10 Simple Rules for Improving Your Standardized Fields and Terms 

**Authors**: Rhiannon Cameron, Emma Griffiths, Damion Dooley, William Hsiao  

**Link**: [PDF](https://arxiv.org/pdf/2510.21825)  

**Abstract**: Contextual metadata is the unsung hero of research data. When done right, standardized and structured vocabularies make your data findable, shareable, and reusable. When done wrong, they turn a well intended effort into data cleanup and curation nightmares. In this paper we tackle the surprisingly tricky process of vocabulary standardization with a mix of practical advice and grounded examples. Drawing from real-world experience in contextual data harmonization, we highlight common challenges (e.g., semantic noise and concept bombs) and provide actionable strategies to address them. Our rules emphasize alignment with Findability, Accessibility, Interoperability, and Reusability (FAIR) principles while remaining adaptable to evolving user and research needs. Whether you are curating datasets, designing a schema, or contributing to a standards body, these rules aim to help you create metadata that is not only technically sound but also meaningful to users. 

---
