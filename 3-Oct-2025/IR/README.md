# Study on LLMs for Promptagator-Style Dense Retriever Training 

**Authors**: Daniel Gwon, Nour Jedidi, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.02241)  

**Abstract**: Promptagator demonstrated that Large Language Models (LLMs) with few-shot prompts can be used as task-specific query generators for fine-tuning domain-specialized dense retrieval models. However, the original Promptagator approach relied on proprietary and large-scale LLMs which users may not have access to or may be prohibited from using with sensitive data. In this work, we study the impact of open-source LLMs at accessible scales ($\leq$14B parameters) as an alternative. Our results demonstrate that open-source LLMs as small as 3B parameters can serve as effective Promptagator-style query generators. We hope our work will inform practitioners with reliable alternatives for synthetic data generation and give insights to maximize fine-tuning results for domain-specific applications. 

---
# Contrastive Retrieval Heads Improve Attention-Based Re-Ranking 

**Authors**: Linh Tran, Yulong Li, Radu Florian, Wei Sun  

**Link**: [PDF](https://arxiv.org/pdf/2510.02219)  

**Abstract**: The strong zero-shot and long-context capabilities of recent Large Language Models (LLMs) have paved the way for highly effective re-ranking systems. Attention-based re-rankers leverage attention weights from transformer heads to produce relevance scores, but not all heads are created equally: many contribute noise and redundancy, thus limiting performance. To address this, we introduce CoRe heads, a small set of retrieval heads identified via a contrastive scoring metric that explicitly rewards high attention heads that correlate with relevant documents, while downplaying nodes with higher attention that correlate with irrelevant documents. This relative ranking criterion isolates the most discriminative heads for re-ranking and yields a state-of-the-art list-wise re-ranker. Extensive experiments with three LLMs show that aggregated signals from CoRe heads, constituting less than 1% of all heads, substantially improve re-ranking accuracy over strong baselines. We further find that CoRe heads are concentrated in middle layers, and pruning the computation of final 50% of model layers preserves accuracy while significantly reducing inference time and memory usage. 

---
# Ranking Items from Discrete Ratings: The Cost of Unknown User Thresholds 

**Authors**: Oscar Villemaud, Suryanarayana Sankagiri, Matthias Grossglauser  

**Link**: [PDF](https://arxiv.org/pdf/2510.01871)  

**Abstract**: Ranking items is a central task in many information retrieval and recommender systems. User input for the ranking task often comes in the form of ratings on a coarse discrete scale. We ask whether it is possible to recover a fine-grained item ranking from such coarse-grained ratings. We model items as having scores and users as having thresholds; a user rates an item positively if the item's score exceeds the user's threshold. Although all users agree on the total item order, estimating that order is challenging when both the scores and the thresholds are latent. Under our model, any ranking method naturally partitions the $n$ items into bins; the bins are ordered, but the items inside each bin are still unordered. Users arrive sequentially, and every new user can be queried to refine the current ranking. We prove that achieving a near-perfect ranking, measured by Spearman distance, requires $\Theta(n^2)$ users (and therefore $\Omega(n^2)$ queries). This is significantly worse than the $O(n\log n)$ queries needed to rank from comparisons; the gap reflects the additional queries needed to identify the users who have the appropriate thresholds. Our bound also quantifies the impact of a mismatch between score and threshold distributions via a quadratic divergence factor. To show the tightness of our results, we provide a ranking algorithm whose query complexity matches our bound up to a logarithmic factor. Our work reveals a tension in online ranking: diversity in thresholds is necessary to merge coarse ratings from many users into a fine-grained ranking, but this diversity has a cost if the thresholds are a priori unknown. 

---
# TalkPlay-Tools: Conversational Music Recommendation with LLM Tool Calling 

**Authors**: Seungheon Doh, Keunwoo Choi, Juhan Nam  

**Link**: [PDF](https://arxiv.org/pdf/2510.01698)  

**Abstract**: While the recent developments in large language models (LLMs) have successfully enabled generative recommenders with natural language interactions, their recommendation behavior is limited, leaving other simpler yet crucial components such as metadata or attribute filtering underutilized in the system. We propose an LLM-based music recommendation system with tool calling to serve as a unified retrieval-reranking pipeline. Our system positions an LLM as an end-to-end recommendation system that interprets user intent, plans tool invocations, and orchestrates specialized components: boolean filters (SQL), sparse retrieval (BM25), dense retrieval (embedding similarity), and generative retrieval (semantic IDs). Through tool planning, the system predicts which types of tools to use, their execution order, and the arguments needed to find music matching user preferences, supporting diverse modalities while seamlessly integrating multiple database filtering methods. We demonstrate that this unified tool-calling framework achieves competitive performance across diverse recommendation scenarios by selectively employing appropriate retrieval methods based on user queries, envisioning a new paradigm for conversational music recommendation systems. 

---
# LLM4Rec: Large Language Models for Multimodal Generative Recommendation with Causal Debiasing 

**Authors**: Bo Ma, Hang Li, ZeHua Hu, XiaoFan Gui, LuYao Liu, Simon Lau  

**Link**: [PDF](https://arxiv.org/pdf/2510.01622)  

**Abstract**: Contemporary generative recommendation systems face significant challenges in handling multimodal data, eliminating algorithmic biases, and providing transparent decision-making processes. This paper introduces an enhanced generative recommendation framework that addresses these limitations through five key innovations: multimodal fusion architecture, retrieval-augmented generation mechanisms, causal inference-based debiasing, explainable recommendation generation, and real-time adaptive learning capabilities. Our framework leverages advanced large language models as the backbone while incorporating specialized modules for cross-modal understanding, contextual knowledge integration, bias mitigation, explanation synthesis, and continuous model adaptation. Extensive experiments on three benchmark datasets (MovieLens-25M, Amazon-Electronics, Yelp-2023) demonstrate consistent improvements in recommendation accuracy, fairness, and diversity compared to existing approaches. The proposed framework achieves up to 2.3% improvement in NDCG@10 and 1.4% enhancement in diversity metrics while maintaining computational efficiency through optimized inference strategies. 

---
# Bridging Collaborative Filtering and Large Language Models with Dynamic Alignment, Multimodal Fusion and Evidence-grounded Explanations 

**Authors**: Bo Ma, LuYao Liu, Simon Lau, Chandler Yuan, and XueY Cui, Rosie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.01606)  

**Abstract**: Recent research has explored using Large Language Models for recommendation tasks by transforming user interaction histories and item metadata into text prompts, then having the LLM produce rankings or recommendations. A promising approach involves connecting collaborative filtering knowledge to LLM representations through compact adapter networks, which avoids expensive fine-tuning while preserving the strengths of both components. Yet several challenges persist in practice: collaborative filtering models often use static snapshots that miss rapidly changing user preferences; many real-world items contain rich visual and audio content beyond textual descriptions; and current systems struggle to provide trustworthy explanations backed by concrete evidence. Our work introduces \model{}, a framework that tackles these limitations through three key innovations. We develop an online adaptation mechanism that continuously incorporates new user interactions through lightweight modules, avoiding the need to retrain large models. We create a unified representation that seamlessly combines collaborative signals with visual and audio features, handling cases where some modalities may be unavailable. Finally, we design an explanation system that grounds recommendations in specific collaborative patterns and item attributes, producing natural language rationales users can verify. Our approach maintains the efficiency of frozen base models while adding minimal computational overhead, making it practical for real-world deployment. 

---
# Synthetic Prefixes to Mitigate Bias in Real-Time Neural Query Autocomplete 

**Authors**: Adithya Rajan, Xiaoyu Liu, Prateek Verma, Vibhu Arora  

**Link**: [PDF](https://arxiv.org/pdf/2510.01574)  

**Abstract**: We introduce a data-centric approach for mitigating presentation bias in real-time neural query autocomplete systems through the use of synthetic prefixes. These prefixes are generated from complete user queries collected during regular search sessions where autocomplete was not active. This allows us to enrich the training data for learning to rank models with more diverse and less biased examples. This method addresses the inherent bias in engagement signals collected from live query autocomplete interactions, where model suggestions influence user behavior. Our neural ranker is optimized for real-time deployment under strict latency constraints and incorporates a rich set of features, including query popularity, seasonality, fuzzy match scores, and contextual signals such as department affinity, device type, and vertical alignment with previous user queries. To support efficient training, we introduce a task-specific simplification of the listwise loss, reducing computational complexity from $O(n^2)$ to $O(n)$ by leveraging the query autocomplete structure of having only one ground-truth selection per prefix. Deployed in a large-scale e-commerce setting, our system demonstrates statistically significant improvements in user engagement, as measured by mean reciprocal rank and related metrics. Our findings show that synthetic prefixes not only improve generalization but also provide a scalable path toward bias mitigation in other low-latency ranking tasks, including related searches and query recommendations. 

---
# IoDResearch: Deep Research on Private Heterogeneous Data via the Internet of Data 

**Authors**: Zhuofan Shi, Zijie Guo, Xinjian Ma, Gang Huang, Yun Ma, Xiang Jing  

**Link**: [PDF](https://arxiv.org/pdf/2510.01553)  

**Abstract**: The rapid growth of multi-source, heterogeneous, and multimodal scientific data has increasingly exposed the limitations of traditional data management. Most existing DeepResearch (DR) efforts focus primarily on web search while overlooking local private data. Consequently, these frameworks exhibit low retrieval efficiency for private data and fail to comply with the FAIR principles, ultimately resulting in inefficiency and limited reusability. To this end, we propose IoDResearch (Internet of Data Research), a private data-centric Deep Research framework that operationalizes the Internet of Data paradigm. IoDResearch encapsulates heterogeneous resources as FAIR-compliant digital objects, and further refines them into atomic knowledge units and knowledge graphs, forming a heterogeneous graph index for multi-granularity retrieval. On top of this representation, a multi-agent system supports both reliable question answering and structured scientific report generation. Furthermore, we establish the IoD DeepResearch Benchmark to systematically evaluate both data representation and Deep Research capabilities in IoD scenarios. Experimental results on retrieval, QA, and report-writing tasks show that IoDResearch consistently surpasses representative RAG and Deep Research baselines. Overall, IoDResearch demonstrates the feasibility of private-data-centric Deep Research under the IoD paradigm, paving the way toward more trustworthy, reusable, and automated scientific discovery. 

---
# MetaSynth: Multi-Agent Metadata Generation from Implicit Feedback in Black-Box Systems 

**Authors**: Shreeranjani Srirangamsridharan, Ali Abavisani, Reza Yousefi Maragheh, Ramin Giahi, Kai Zhao, Jason Cho, Sushant Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2510.01523)  

**Abstract**: Meta titles and descriptions strongly shape engagement in search and recommendation platforms, yet optimizing them remains challenging. Search engine ranking models are black box environments, explicit labels are unavailable, and feedback such as click-through rate (CTR) arrives only post-deployment. Existing template, LLM, and retrieval-augmented approaches either lack diversity, hallucinate attributes, or ignore whether candidate phrasing has historically succeeded in ranking. This leaves a gap in directly leveraging implicit signals from observable outcomes. We introduce MetaSynth, a multi-agent retrieval-augmented generation framework that learns from implicit search feedback. MetaSynth builds an exemplar library from top-ranked results, generates candidate snippets conditioned on both product content and exemplars, and iteratively refines outputs via evaluator-generator loops that enforce relevance, promotional strength, and compliance. On both proprietary e-commerce data and the Amazon Reviews corpus, MetaSynth outperforms strong baselines across NDCG, MRR, and rank metrics. Large-scale A/B tests further demonstrate 10.26% CTR and 7.51% clicks. Beyond metadata, this work contributes a general paradigm for optimizing content in black-box systems using implicit signals. 

---
# Optimal signals assignment for eBay View Item page 

**Authors**: Matan Mandelbrod, Biwei Jiang, Giald Wagner, Tal Franji, Guy Feigenblat  

**Link**: [PDF](https://arxiv.org/pdf/2510.01198)  

**Abstract**: Signals are short textual or visual snippets displayed on the eBay View-Item (VI) page, providing additional, contextual information for users about the viewed item. The aim in displaying the signals is to facilitate intelligent purchase and to incentivise engagement. In this paper, we present two approaches for developing statistical models that optimally populate the VI page with signals. Both approaches were A/B tested, and yielded significant increase in business metrics. 

---
# Are LLMs ready to help non-expert users to make charts of official statistics data? 

**Authors**: Gadir Suleymanli, Alexander Rogiers, Lucas Lageweg, Jefrey Lijffijt  

**Link**: [PDF](https://arxiv.org/pdf/2510.01197)  

**Abstract**: In this time when biased information, deep fakes, and propaganda proliferate, the accessibility of reliable data sources is more important than ever. National statistical institutes provide curated data that contain quantitative information on a wide range of topics. However, that information is typically spread across many tables and the plain numbers may be arduous to process. Hence, this open data may be practically inaccessible. We ask the question "Are current Generative AI models capable of facilitating the identification of the right data and the fully-automatic creation of charts to provide information in visual form, corresponding to user queries?". We present a structured evaluation of recent large language models' (LLMs) capabilities to generate charts from complex data in response to user queries. Working with diverse public data from Statistics Netherlands, we assessed multiple LLMs on their ability to identify relevant data tables, perform necessary manipulations, and generate appropriate visualizations autonomously. We propose a new evaluation framework spanning three dimensions: data retrieval & pre-processing, code quality, and visual representation. Results indicate that locating and processing the correct data represents the most significant challenge. Additionally, LLMs rarely implement visualization best practices without explicit guidance. When supplemented with information about effective chart design, models showed marked improvement in representation scores. Furthermore, an agentic approach with iterative self-evaluation led to excellent performance across all evaluation dimensions. These findings suggest that LLMs' effectiveness for automated chart generation can be enhanced through appropriate scaffolding and feedback mechanisms, and that systems can already reach the necessary accuracy across the three evaluation dimensions. 

---
# Location Matters: Leveraging Multi-Resolution Geo-Embeddings for Housing Search 

**Authors**: Ivo Silva, Pedro Nogueira, Guilherme Bonaldo  

**Link**: [PDF](https://arxiv.org/pdf/2510.01196)  

**Abstract**: QuintoAndar Group is Latin America's largest housing platform, revolutionizing property rentals and sales. Headquartered in Brazil, it simplifies the housing process by eliminating paperwork and enhancing accessibility for tenants, buyers, and landlords. With thousands of houses available for each city, users struggle to find the ideal home. In this context, location plays a pivotal role, as it significantly influences property value, access to amenities, and life quality. A great location can make even a modest home highly desirable. Therefore, incorporating location into recommendations is essential for their effectiveness. We propose a geo-aware embedding framework to address sparsity and spatial nuances in housing recommendations on digital rental platforms. Our approach integrates an hierarchical H3 grid at multiple levels into a two-tower neural architecture. We compare our method with a traditional matrix factorization baseline and a single-resolution variant using interaction data from our platform. Embedding specific evaluation reveals richer and more balanced embedding representations, while offline ranking simulations demonstrate a substantial uplift in recommendation quality. 

---
# Comparison of Unsupervised Metrics for Evaluating Judicial Decision Extraction 

**Authors**: Ivan Leonidovich Litvak, Anton Kostin, Fedor Lashkin, Tatiana Maksiyan, Sergey Lagutin  

**Link**: [PDF](https://arxiv.org/pdf/2510.01792)  

**Abstract**: The rapid advancement of artificial intelligence in legal natural language processing demands scalable methods for evaluating text extraction from judicial decisions. This study evaluates 16 unsupervised metrics, including novel formulations, to assess the quality of extracting seven semantic blocks from 1,000 anonymized Russian judicial decisions, validated against 7,168 expert reviews on a 1--5 Likert scale. These metrics, spanning document-based, semantic, structural, pseudo-ground truth, and legal-specific categories, operate without pre-annotated ground truth. Bootstrapped correlations, Lin's concordance correlation coefficient (CCC), and mean absolute error (MAE) reveal that Term Frequency Coherence (Pearson $r = 0.540$, Lin CCC = 0.512, MAE = 0.127) and Coverage Ratio/Block Completeness (Pearson $r = 0.513$, Lin CCC = 0.443, MAE = 0.139) best align with expert ratings, while Legal Term Density (Pearson $r = -0.479$, Lin CCC = -0.079, MAE = 0.394) show strong negative correlations. The LLM Evaluation Score (mean = 0.849, Pearson $r = 0.382$, Lin CCC = 0.325, MAE = 0.197) showed moderate alignment, but its performance, using gpt-4.1-mini via g4f, suggests limited specialization for legal textse. These findings highlight that unsupervised metrics, including LLM-based approaches, enable scalable screening but, with moderate correlations and low CCC values, cannot fully replace human judgment in high-stakes legal contexts. This work advances legal NLP by providing annotation-free evaluation tools, with implications for judicial analytics and ethical AI deployment. 

---
# From Videos to Indexed Knowledge Graphs -- Framework to Marry Methods for Multimodal Content Analysis and Understanding 

**Authors**: Basem Rizk, Joel Walsh, Mark Core, Benjamin Nye  

**Link**: [PDF](https://arxiv.org/pdf/2510.01513)  

**Abstract**: Analysis of multi-modal content can be tricky, computationally expensive, and require a significant amount of engineering efforts. Lots of work with pre-trained models on static data is out there, yet fusing these opensource models and methods with complex data such as videos is relatively challenging. In this paper, we present a framework that enables efficiently prototyping pipelines for multi-modal content analysis. We craft a candidate recipe for a pipeline, marrying a set of pre-trained models, to convert videos into a temporal semi-structured data format. We translate this structure further to a frame-level indexed knowledge graph representation that is query-able and supports continual learning, enabling the dynamic incorporation of new domain-specific knowledge through an interactive medium. 

---
# LLM-based Multi-Agent Blackboard System for Information Discovery in Data Science 

**Authors**: Alireza Salemi, Mihir Parmar, Palash Goyal, Yiwen Song, Jinsung Yoon, Hamed Zamani, Hamid Palangi, Tomas Pfister  

**Link**: [PDF](https://arxiv.org/pdf/2510.01285)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) has opened new opportunities in data science, yet their practical deployment is often constrained by the challenge of discovering relevant data within large heterogeneous data lakes. Existing methods struggle with this: single-agent systems are quickly overwhelmed by large, heterogeneous files in the large data lakes, while multi-agent systems designed based on a master-slave paradigm depend on a rigid central controller for task allocation that requires precise knowledge of each sub-agent's capabilities. To address these limitations, we propose a novel multi-agent communication paradigm inspired by the blackboard architecture for traditional AI models. In this framework, a central agent posts requests to a shared blackboard, and autonomous subordinate agents -- either responsible for a partition of the data lake or general information retrieval -- volunteer to respond based on their capabilities. This design improves scalability and flexibility by eliminating the need for a central coordinator to have prior knowledge of all sub-agents' expertise. We evaluate our method on three benchmarks that require explicit data discovery: KramaBench and modified versions of DS-Bench and DA-Code to incorporate data discovery. Experimental results demonstrate that the blackboard architecture substantially outperforms baselines, including RAG and the master-slave multi-agent paradigm, achieving between 13% to 57% relative improvement in end-to-end task success and up to a 9% relative gain in F1 score for data discovery over the best-performing baselines across both proprietary and open-source LLMs. Our findings establish the blackboard paradigm as a scalable and generalizable communication framework for multi-agent systems. 

---
