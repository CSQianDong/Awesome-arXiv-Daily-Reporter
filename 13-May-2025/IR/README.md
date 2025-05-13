# Reproducibility, Replicability, and Insights into Visual Document Retrieval with Late Interaction 

**Authors**: Jingfen Qiao, Jia-Huei Ju, Xinyu Ma, Evangelos Kanoulas, Andrew Yates  

**Link**: [PDF](https://arxiv.org/pdf/2505.07730)  

**Abstract**: Visual Document Retrieval (VDR) is an emerging research area that focuses on encoding and retrieving document images directly, bypassing the dependence on Optical Character Recognition (OCR) for document search. A recent advance in VDR was introduced by ColPali, which significantly improved retrieval effectiveness through a late interaction mechanism. ColPali's approach demonstrated substantial performance gains over existing baselines that do not use late interaction on an established benchmark. In this study, we investigate the reproducibility and replicability of VDR methods with and without late interaction mechanisms by systematically evaluating their performance across multiple pre-trained vision-language models. Our findings confirm that late interaction yields considerable improvements in retrieval effectiveness; however, it also introduces computational inefficiencies during inference. Additionally, we examine the adaptability of VDR models to textual inputs and assess their robustness across text-intensive datasets within the proposed benchmark, particularly when scaling the indexing mechanism. Furthermore, our research investigates the specific contributions of late interaction by looking into query-patch matching in the context of visual document retrieval. We find that although query tokens cannot explicitly match image patches as in the text retrieval scenario, they tend to match the patch contains visually similar tokens or their surrounding patches. 

---
# KAQG: A Knowledge-Graph-Enhanced RAG for Difficulty-Controlled Question Generation 

**Authors**: Ching Han Chen, Ming Fang Shiu  

**Link**: [PDF](https://arxiv.org/pdf/2505.07618)  

**Abstract**: KAQG introduces a decisive breakthrough for Retrieval-Augmented Generation (RAG) by explicitly tackling the two chronic weaknesses of current pipelines: transparent multi-step reasoning and fine-grained cognitive difficulty control. This transforms RAG from a passive retriever into an accountable generator of calibrated exam items. Technically, the framework fuses knowledge graphs, RAG retrieval, and educational assessment theory into a single pipeline. Domain passages are parsed into a structured graph; graph-aware retrieval feeds fact chains to an LLM; and an assessment layer governed by Bloom's Taxonomy levels and Item Response Theory (IRT) transforms those chains into psychometrically sound questions. This cross-disciplinary marriage yields two scholarly contributions: it shows how semantic graph contexts guide LLM reasoning paths, and it operationalizes difficulty metrics within the generation process, producing items whose IRT parameters match expert benchmarks. Every module, from KG construction scripts to the multi-agent reasoning scheduler and the automatic IRT validator, is openly released on GitHub. This enables peer laboratories to replicate experiments, benchmark against baselines, and extend individual components without licensing barriers. Its reproducible design paves the way for rigorous ablation studies, cross-domain transfer experiments, and shared leaderboards on multi-step reasoning benchmarks. 

---
# GRADA: Graph-based Reranker against Adversarial Documents Attack 

**Authors**: Jingjie Zheng, Aryo Pradipta Gema, Giwon Hong, Xuanli He, Pasquale Minervini, Youcheng Sun, Qiongkai Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.07546)  

**Abstract**: Retrieval Augmented Generation (RAG) frameworks improve the accuracy of large language models (LLMs) by integrating external knowledge from retrieved documents, thereby overcoming the limitations of models' static intrinsic knowledge. However, these systems are susceptible to adversarial attacks that manipulate the retrieval process by introducing documents that are adversarial yet semantically similar to the query. Notably, while these adversarial documents resemble the query, they exhibit weak similarity to benign documents in the retrieval set. Thus, we propose a simple yet effective Graph-based Reranking against Adversarial Document Attacks (GRADA) framework aiming at preserving retrieval quality while significantly reducing the success of adversaries. Our study evaluates the effectiveness of our approach through experiments conducted on five LLMs: GPT-3.5-Turbo, GPT-4o, Llama3.1-8b, Llama3.1-70b, and Qwen2.5-7b. We use three datasets to assess performance, with results from the Natural Questions dataset demonstrating up to an 80% reduction in attack success rates while maintaining minimal loss in accuracy. 

---
# Why Uncertainty Estimation Methods Fall Short in RAG: An Axiomatic Analysis 

**Authors**: Heydar Soudani, Evangelos Kanoulas, Faegheh Hasibi  

**Link**: [PDF](https://arxiv.org/pdf/2505.07459)  

**Abstract**: Large Language Models (LLMs) are valued for their strong performance across various tasks, but they also produce inaccurate or misleading outputs. Uncertainty Estimation (UE) quantifies the model's confidence and helps users assess response reliability. However, existing UE methods have not been thoroughly examined in scenarios like Retrieval-Augmented Generation (RAG), where the input prompt includes non-parametric knowledge. This paper shows that current UE methods cannot reliably assess correctness in the RAG setting. We further propose an axiomatic framework to identify deficiencies in existing methods and guide the development of improved approaches. Our framework introduces five constraints that an effective UE method should meet after incorporating retrieved documents into the LLM's prompt. Experimental results reveal that no existing UE method fully satisfies all the axioms, explaining their suboptimal performance in RAG. We further introduce a simple yet effective calibration function based on our framework, which not only satisfies more axioms than baseline methods but also improves the correlation between uncertainty estimates and correctness. 

---
# Diffusion-driven SpatioTemporal Graph KANsformer for Medical Examination Recommendation 

**Authors**: Jianan Li, Yangtao Zhou, Zhifu Zhao, Qinglan Huang, Jian Qi, Xiao He, Hua Chu, Fu Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.07431)  

**Abstract**: Recommendation systems in AI-based medical diagnostics and treatment constitute a critical component of AI in healthcare. Although some studies have explored this area and made notable progress, healthcare recommendation systems remain in their nascent stage. And these researches mainly target the treatment process such as drug or disease recommendations. In addition to the treatment process, the diagnostic process, particularly determining which medical examinations are necessary to evaluate the condition, also urgently requires intelligent decision support. To bridge this gap, we first formalize the task of medical examination recommendations. Compared to traditional recommendations, the medical examination recommendation involves more complex interactions. This complexity arises from two folds: 1) The historical medical records for examination recommendations are heterogeneous and redundant, which makes the recommendation results susceptible to noise. 2) The correlation between the medical history of patients is often irregular, making it challenging to model spatiotemporal dependencies. Motivated by the above observation, we propose a novel Diffusion-driven SpatioTemporal Graph KANsformer for Medical Examination Recommendation (DST-GKAN) with a two-stage learning paradigm to solve the above challenges. In the first stage, we exploit a task-adaptive diffusion model to distill recommendation-oriented information by reducing the noises in heterogeneous medical data. In the second stage, a spatiotemporal graph KANsformer is proposed to simultaneously model the complex spatial and temporal relationships. Moreover, to facilitate the medical examination recommendation research, we introduce a comprehensive dataset. The experimental results demonstrate the state-of-the-art performance of the proposed method compared to various competitive baselines. 

---
# DARLR: Dual-Agent Offline Reinforcement Learning for Recommender Systems with Dynamic Reward 

**Authors**: Yi Zhang, Ruihong Qiu, Xuwei Xu, Jiajun Liu, Sen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07257)  

**Abstract**: Model-based offline reinforcement learning (RL) has emerged as a promising approach for recommender systems, enabling effective policy learning by interacting with frozen world models. However, the reward functions in these world models, trained on sparse offline logs, often suffer from inaccuracies. Specifically, existing methods face two major limitations in addressing this challenge: (1) deterministic use of reward functions as static look-up tables, which propagates inaccuracies during policy learning, and (2) static uncertainty designs that fail to effectively capture decision risks and mitigate the impact of these inaccuracies. In this work, a dual-agent framework, DARLR, is proposed to dynamically update world models to enhance recommendation policies. To achieve this, a \textbf{\textit{selector}} is introduced to identify reference users by balancing similarity and diversity so that the \textbf{\textit{recommender}} can aggregate information from these users and iteratively refine reward estimations for dynamic reward shaping. Further, the statistical features of the selected users guide the dynamic adaptation of an uncertainty penalty to better align with evolving recommendation requirements. Extensive experiments on four benchmark datasets demonstrate the superior performance of DARLR, validating its effectiveness. The code is available at this https URL. 

---
# A Generative Re-ranking Model for List-level Multi-objective Optimization at Taobao 

**Authors**: Yue Meng, Cheng Guo, Yi Cao, Tong Liu, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.07197)  

**Abstract**: E-commerce recommendation systems aim to generate ordered lists of items for customers, optimizing multiple business objectives, such as clicks, conversions and Gross Merchandise Volume (GMV). Traditional multi-objective optimization methods like formulas or Learning-to-rank (LTR) models take effect at item-level, neglecting dynamic user intent and contextual item interactions. List-level multi-objective optimization in the re-ranking stage can overcome this limitation, but most current re-ranking models focus more on accuracy improvement with context. In addition, re-ranking is faced with the challenges of time complexity and diversity. In light of this, we propose a novel end-to-end generative re-ranking model named Sequential Ordered Regression Transformer-Generator (SORT-Gen) for the less-studied list-level multi-objective optimization problem. Specifically, SORT-Gen is divided into two parts: 1)Sequential Ordered Regression Transformer innovatively uses Transformer and ordered regression to accurately estimate multi-objective values for variable-length sub-lists. 2)Mask-Driven Fast Generation Algorithm combines multi-objective candidate queues, efficient item selection and diversity mechanism into model inference, providing a fast online list generation method. Comprehensive online experiments demonstrate that SORT-Gen brings +4.13% CLCK and +8.10% GMV for Baiyibutie, a notable Mini-app of Taobao. Currently, SORT-Gen has been successfully deployed in multiple scenarios of Taobao App, serving for a vast number of users. 

---
# Pre-training vs. Fine-tuning: A Reproducibility Study on Dense Retrieval Knowledge Acquisition 

**Authors**: Zheng Yao, Shuai Wang, Guido Zuccon  

**Link**: [PDF](https://arxiv.org/pdf/2505.07166)  

**Abstract**: Dense retrievers utilize pre-trained backbone language models (e.g., BERT, LLaMA) that are fine-tuned via contrastive learning to perform the task of encoding text into sense representations that can be then compared via a shallow similarity operation, e.g. inner product. Recent research has questioned the role of fine-tuning vs. that of pre-training within dense retrievers, specifically arguing that retrieval knowledge is primarily gained during pre-training, meaning knowledge not acquired during pre-training cannot be sub-sequentially acquired via fine-tuning. We revisit this idea here as the claim was only studied in the context of a BERT-based encoder using DPR as representative dense retriever. We extend the previous analysis by testing other representation approaches (comparing the use of CLS tokens with that of mean pooling), backbone architectures (encoder-only BERT vs. decoder-only LLaMA), and additional datasets (MSMARCO in addition to Natural Questions). Our study confirms that in DPR tuning, pre-trained knowledge underpins retrieval performance, with fine-tuning primarily adjusting neuron activation rather than reorganizing knowledge. However, this pattern does not hold universally, such as in mean-pooled (Contriever) and decoder-based (LLaMA) models. We ensure full reproducibility and make our implementation publicly available at this https URL. 

---
# Reassessing Large Language Model Boolean Query Generation for Systematic Reviews 

**Authors**: Shuai Wang, Harrisen Scells, Bevan Koopman, Guido Zuccon  

**Link**: [PDF](https://arxiv.org/pdf/2505.07155)  

**Abstract**: Systematic reviews are comprehensive literature reviews that address highly focused research questions and represent the highest form of evidence in medicine. A critical step in this process is the development of complex Boolean queries to retrieve relevant literature. Given the difficulty of manually constructing these queries, recent efforts have explored Large Language Models (LLMs) to assist in their formulation. One of the first studies,Wang et al., investigated ChatGPT for this task, followed by Staudinger et al., which evaluated multiple LLMs in a reproducibility study. However, the latter overlooked several key aspects of the original work, including (i) validation of generated queries, (ii) output formatting constraints, and (iii) selection of examples for chain-of-thought (Guided) prompting. As a result, its findings diverged significantly from the original study. In this work, we systematically reproduce both studies while addressing these overlooked factors. Our results show that query effectiveness varies significantly across models and prompt designs, with guided query formulation benefiting from well-chosen seed studies. Overall, prompt design and model selection are key drivers of successful query formulation. Our findings provide a clearer understanding of LLMs' potential in Boolean query generation and highlight the importance of model- and prompt-specific optimisations. The complex nature of systematic reviews adds to challenges in both developing and reproducing methods but also highlights the importance of reproducibility studies in this domain. 

---
# Knowledge Distillation for Enhancing Walmart E-commerce Search Relevance Using Large Language Models 

**Authors**: Hongwei Shang, Nguyen Vo, Nitin Yadav, Tian Zhang, Ajit Puthenputhussery, Xunfan Cai, Shuyi Chen, Prijith Chandran, Changsung Kang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07105)  

**Abstract**: Ensuring the products displayed in e-commerce search results are relevant to users queries is crucial for improving the user experience. With their advanced semantic understanding, deep learning models have been widely used for relevance matching in search tasks. While large language models (LLMs) offer superior ranking capabilities, it is challenging to deploy LLMs in real-time systems due to the high-latency requirements. To leverage the ranking power of LLMs while meeting the low-latency demands of production systems, we propose a novel framework that distills a high performing LLM into a more efficient, low-latency student model. To help the student model learn more effectively from the teacher model, we first train the teacher LLM as a classification model with soft targets. Then, we train the student model to capture the relevance margin between pairs of products for a given query using mean squared error loss. Instead of using the same training data as the teacher model, we significantly expand the student model dataset by generating unlabeled data and labeling it with the teacher model predictions. Experimental results show that the student model performance continues to improve as the size of the augmented training data increases. In fact, with enough augmented data, the student model can outperform the teacher model. The student model has been successfully deployed in production at this http URL with significantly positive metrics. 

---
# NetSight: Graph Attention Based Traffic Forecasting in Computer Networks 

**Authors**: Jinming Xing, Guoheng Sun, Hui Sun, Linchao Pan, Shakir Mahmood, Xuanhao Luo, Muhammad Shahzad  

**Link**: [PDF](https://arxiv.org/pdf/2505.07034)  

**Abstract**: The traffic in today's networks is increasingly influenced by the interactions among network nodes as well as by the temporal fluctuations in the demands of the nodes. Traditional statistical prediction methods are becoming obsolete due to their inability to address the non-linear and dynamic spatio-temporal dependencies present in today's network traffic. The most promising direction of research today is graph neural networks (GNNs) based prediction approaches that are naturally suited to handle graph-structured data. Unfortunately, the state-of-the-art GNN approaches separate the modeling of spatial and temporal information, resulting in the loss of important information about joint dependencies. These GNN based approaches further do not model information at both local and global scales simultaneously, leaving significant room for improvement. To address these challenges, we propose NetSight. NetSight learns joint spatio-temporal dependencies simultaneously at both global and local scales from the time-series of measurements of any given network metric collected at various nodes in a network. Using the learned information, NetSight can then accurately predict the future values of the given network metric at those nodes in the network. We propose several new concepts and techniques in the design of NetSight, such as spatio-temporal adjacency matrix and node normalization. Through extensive evaluations and comparison with prior approaches using data from two large real-world networks, we show that NetSight significantly outperforms all prior state-of-the-art approaches. We will release the source code and data used in the evaluation of NetSight on the acceptance of this paper. 

---
# Web Page Classification using LLMs for Crawling Support 

**Authors**: Yuichi Sasazawa, Yasuhiro Sogawa  

**Link**: [PDF](https://arxiv.org/pdf/2505.06972)  

**Abstract**: A web crawler is a system designed to collect web pages, and efficient crawling of new pages requires appropriate algorithms. While website features such as XML sitemaps and the frequency of past page updates provide important clues for accessing new pages, their universal application across diverse conditions is challenging. In this study, we propose a method to efficiently collect new pages by classifying web pages into two types, "Index Pages" and "Content Pages," using a large language model (LLM), and leveraging the classification results to select index pages as starting points for accessing new pages. We construct a dataset with automatically annotated web page types and evaluate our approach from two perspectives: the page type classification performance and coverage of new pages. Experimental results demonstrate that the LLM-based method outperformed baseline methods in both evaluation metrics. 

---
# Optimizing Recommendations using Fine-Tuned LLMs 

**Authors**: Prabhdeep Cheema, Erhan Guven  

**Link**: [PDF](https://arxiv.org/pdf/2505.06841)  

**Abstract**: As digital media platforms strive to meet evolving user expectations, delivering highly personalized and intuitive movies and media recommendations has become essential for attracting and retaining audiences. Traditional systems often rely on keyword-based search and recommendation techniques, which limit users to specific keywords and a combination of keywords. This paper proposes an approach that generates synthetic datasets by modeling real-world user interactions, creating complex chat-style data reflective of diverse preferences. This allows users to express more information with complex preferences, such as mood, plot details, and thematic elements, in addition to conventional criteria like genre, title, and actor-based searches. In today's search space, users cannot write queries like ``Looking for a fantasy movie featuring dire wolves, ideally set in a harsh frozen world with themes of loyalty and survival.''
Building on these contributions, we evaluate synthetic datasets for diversity and effectiveness in training and benchmarking models, particularly in areas often absent from traditional datasets. This approach enhances personalization and accuracy by enabling expressive and natural user queries. It establishes a foundation for the next generation of conversational AI-driven search and recommendation systems in digital entertainment. 

---
# Document Attribution: Examining Citation Relationships using Large Language Models 

**Authors**: Vipula Rawte, Ryan A. Rossi, Franck Dernoncourt, Nedim Lipka  

**Link**: [PDF](https://arxiv.org/pdf/2505.06324)  

**Abstract**: As Large Language Models (LLMs) are increasingly applied to document-based tasks - such as document summarization, question answering, and information extraction - where user requirements focus on retrieving information from provided documents rather than relying on the model's parametric knowledge, ensuring the trustworthiness and interpretability of these systems has become a critical concern. A central approach to addressing this challenge is attribution, which involves tracing the generated outputs back to their source documents. However, since LLMs can produce inaccurate or imprecise responses, it is crucial to assess the reliability of these citations.
To tackle this, our work proposes two techniques. (1) A zero-shot approach that frames attribution as a straightforward textual entailment task. Our method using flan-ul2 demonstrates an improvement of 0.27% and 2.4% over the best baseline of ID and OOD sets of AttributionBench, respectively. (2) We also explore the role of the attention mechanism in enhancing the attribution process. Using a smaller LLM, flan-t5-small, the F1 scores outperform the baseline across almost all layers except layer 4 and layers 8 through 11. 

---
# AI Approaches to Qualitative and Quantitative News Analytics on NATO Unity 

**Authors**: Bohdan M. Pavlyshenko  

**Link**: [PDF](https://arxiv.org/pdf/2505.06313)  

**Abstract**: The paper considers the use of GPT models with retrieval-augmented generation (RAG) for qualitative and quantitative analytics on NATO sentiments, NATO unity and NATO Article 5 trust opinion scores in different web sources: news sites found via Google Search API, Youtube videos with comments, and Reddit discussions. A RAG approach using GPT-4.1 model was applied to analyse news where NATO related topics were discussed. Two levels of RAG analytics were used: on the first level, the GPT model generates qualitative news summaries and quantitative opinion scores using zero-shot prompts; on the second level, the GPT model generates the summary of news summaries. Quantitative news opinion scores generated by the GPT model were analysed using Bayesian regression to get trend lines. The distributions found for the regression parameters make it possible to analyse an uncertainty in specified news opinion score trends. Obtained results show a downward trend for analysed scores of opinion related to NATO unity.
This approach does not aim to conduct real political analysis; rather, it consider AI based approaches which can be used for further analytics
as a part of a complex analytical approach. The obtained results demonstrate that the use of GPT models for news analysis can give informative qualitative and quantitative analytics, providing important insights.
The dynamic model based on neural ordinary differential equations was considered for modelling public opinions. This approach makes it possible to analyse different scenarios for evolving public opinions. 

---
# Benchmarking Retrieval-Augmented Generation for Chemistry 

**Authors**: Xianrui Zhong, Bowen Jin, Siru Ouyang, Yanzhen Shen, Qiao Jin, Yin Fang, Zhiyong Lu, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.07671)  

**Abstract**: Retrieval-augmented generation (RAG) has emerged as a powerful framework for enhancing large language models (LLMs) with external knowledge, particularly in scientific domains that demand specialized and dynamic information. Despite its promise, the application of RAG in the chemistry domain remains underexplored, primarily due to the lack of high-quality, domain-specific corpora and well-curated evaluation benchmarks. In this work, we introduce ChemRAG-Bench, a comprehensive benchmark designed to systematically assess the effectiveness of RAG across a diverse set of chemistry-related tasks. The accompanying chemistry corpus integrates heterogeneous knowledge sources, including scientific literature, the PubChem database, PubMed abstracts, textbooks, and Wikipedia entries. In addition, we present ChemRAG-Toolkit, a modular and extensible RAG toolkit that supports five retrieval algorithms and eight LLMs. Using ChemRAG-Toolkit, we demonstrate that RAG yields a substantial performance gain -- achieving an average relative improvement of 17.4% over direct inference methods. We further conduct in-depth analyses on retriever architectures, corpus selection, and the number of retrieved passages, culminating in practical recommendations to guide future research and deployment of RAG systems in the chemistry domain. The code and data is available at this https URL. 

---
# From raw affiliations to organization identifiers 

**Authors**: Myrto Kallipoliti, Serafeim Chatzopoulos, Miriam Baglioni, Eleni Adamidi, Paris Koloveas, Thanasis Vergoulis  

**Link**: [PDF](https://arxiv.org/pdf/2505.07577)  

**Abstract**: Accurate affiliation matching, which links affiliation strings to standardized organization identifiers, is critical for improving research metadata quality, facilitating comprehensive bibliometric analyses, and supporting data interoperability across scholarly knowledge bases. Existing approaches fail to handle the complexity of affiliation strings that often include mentions of multiple organizations or extraneous information. In this paper, we present AffRo, a novel approach designed to address these challenges, leveraging advanced parsing and disambiguation techniques. We also introduce AffRoDB, an expert-curated dataset to systematically evaluate affiliation matching algorithms, ensuring robust benchmarking. Results demonstrate the effectiveness of AffRp in accurately identifying organizations from complex affiliation strings. 

---
# Injecting Knowledge Graphs into Large Language Models 

**Authors**: Erica Coppolillo  

**Link**: [PDF](https://arxiv.org/pdf/2505.07554)  

**Abstract**: Integrating structured knowledge from Knowledge Graphs (KGs) into Large Language Models (LLMs) remains a key challenge for symbolic reasoning. Existing methods mainly rely on prompt engineering or fine-tuning, which lose structural fidelity or incur high computational costs. Building on recent encoding techniques which integrate graph embeddings within the LLM input as tokens, we extend this paradigm to the KG domain by leveraging Knowledge Graph Embedding (KGE) models, thus enabling graph-aware reasoning. Our approach is model-agnostic, resource-efficient, and compatible with any LLMs. Extensive experimentation on synthetic and real-world datasets shows that our method improves reasoning performance over established baselines, further achieving the best trade-off in terms of accuracy and efficiency against state-of-the-art LLMs. 

---
# QUPID: Quantified Understanding for Enhanced Performance, Insights, and Decisions in Korean Search Engines 

**Authors**: Ohjoon Kwon, Changsu Lee, Jihye Back, Lim Sun Suk, Inho Kang, Donghyeon Jeon  

**Link**: [PDF](https://arxiv.org/pdf/2505.07345)  

**Abstract**: Large language models (LLMs) have been widely used for relevance assessment in information retrieval. However, our study demonstrates that combining two distinct small language models (SLMs) with different architectures can outperform LLMs in this task. Our approach -- QUPID -- integrates a generative SLM with an embedding-based SLM, achieving higher relevance judgment accuracy while reducing computational costs compared to state-of-the-art LLM solutions. This computational efficiency makes QUPID highly scalable for real-world search systems processing millions of queries daily. In experiments across diverse document types, our method demonstrated consistent performance improvements (Cohen's Kappa of 0.646 versus 0.387 for leading LLMs) while offering 60x faster inference times. Furthermore, when integrated into production search pipelines, QUPID improved nDCG@5 scores by 1.9%. These findings underscore how architectural diversity in model combinations can significantly enhance both search relevance and operational efficiency in information retrieval systems. 

---
# ReCDAP: Relation-Based Conditional Diffusion with Attention Pooling for Few-Shot Knowledge Graph Completion 

**Authors**: Jeongho Kim, Chanyeong Heo, Jaehee Jung  

**Link**: [PDF](https://arxiv.org/pdf/2505.07171)  

**Abstract**: Knowledge Graphs (KGs), composed of triples in the form of (head, relation, tail) and consisting of entities and relations, play a key role in information retrieval systems such as question answering, entity search, and recommendation. In real-world KGs, although many entities exist, the relations exhibit a long-tail distribution, which can hinder information retrieval performance. Previous few-shot knowledge graph completion studies focused exclusively on the positive triple information that exists in the graph or, when negative triples were incorporated, used them merely as a signal to indicate incorrect triples. To overcome this limitation, we propose Relation-Based Conditional Diffusion with Attention Pooling (ReCDAP). First, negative triples are generated by randomly replacing the tail entity in the support set. By conditionally incorporating positive information in the KG and non-existent negative information into the diffusion process, the model separately estimates the latent distributions for positive and negative relations. Moreover, including an attention pooler enables the model to leverage the differences between positive and negative cases explicitly. Experiments on two widely used datasets demonstrate that our method outperforms existing approaches, achieving state-of-the-art performance. The code is available at this https URL. 

---
# A Reinforcement Learning Framework for Application-Specific TCP Congestion-Control 

**Authors**: Jinming Xing, Muhammad Shahzad  

**Link**: [PDF](https://arxiv.org/pdf/2505.07042)  

**Abstract**: The Congestion Control (CC) module plays a critical role in the Transmission Control Protocol (TCP), ensuring the stability and efficiency of network data transmission. The CC approaches that are commonly used these days employ heuristics-based rules to adjust the sending rate. Due to their heuristics-based nature, these approaches are not only unable to adapt to changing network conditions but are also agnostic to the diverse requirements that different applications often have. Recently, several learning-based CC approaches have been proposed to adapt to changing network conditions. Unfortunately, they are not designed to take application requirements into account. Prior heuristics-based as well as learning-based CC approaches focus on achieving a singular objective, which is often to maximize throughput, even though a lot of applications care more about latency, packet losses, jitter, and different combinations of various network metrics. Motivated by this, we propose a Deep Reinforcement Learning (DRL) based CC framework, namely ASC, which allows any application to specify any arbitrary objectives that the network traffic of that application should achieve and is able to swiftly adapt to the changes in the objectives of the applications as well as to the changes in the network conditions. Our ASC framework further employs a client-server architecture that serves two purposes: 1) it makes ASC highly scalable in terms of the arrival and departure of TCP connections, and 2) it makes ASC very lightweight for the nodes maintaining the TCP connections. We implemented and extensively evaluated ASC in a variety of settings. Our results show that it can not only achieve various objectives but also outperforms prior approaches even in the specific objectives that those approaches were designed to achieve. 

---
# The Distracting Effect: Understanding Irrelevant Passages in RAG 

**Authors**: Chen Amiraz, Florin Cuconasu, Simone Filice, Zohar Karnin  

**Link**: [PDF](https://arxiv.org/pdf/2505.06914)  

**Abstract**: A well-known issue with Retrieval Augmented Generation (RAG) is that retrieved passages that are irrelevant to the query sometimes distract the answer-generating LLM, causing it to provide an incorrect response. In this paper, we shed light on this core issue and formulate the distracting effect of a passage w.r.t. a query (and an LLM). We provide a quantifiable measure of the distracting effect of a passage and demonstrate its robustness across LLMs.
Our research introduces novel methods for identifying and using hard distracting passages to improve RAG systems. By fine-tuning LLMs with these carefully selected distracting passages, we achieve up to a 7.5% increase in answering accuracy compared to counterparts fine-tuned on conventional RAG datasets. Our contribution is two-fold: first, we move beyond the simple binary classification of irrelevant passages as either completely unrelated vs. distracting, and second, we develop and analyze multiple methods for finding hard distracting passages. To our knowledge, no other research has provided such a comprehensive framework for identifying and utilizing hard distracting passages. 

---
# Incremental Analysis of Legacy Applications Using Knowledge Graphs for Application Modernization 

**Authors**: Saravanan Krishnan, Amith Singhee, Keerthi Narayan Raghunath, Alex Mathai, Atul Kumar, David Wenk  

**Link**: [PDF](https://arxiv.org/pdf/2505.06885)  

**Abstract**: Industries such as banking, telecom, and airlines - o6en have large so6ware systems that are several decades old. Many of these systems are written in old programming languages such as COBOL, PL/1, Assembler, etc. In many cases, the documentation is not updated, and those who developed/designed these systems are no longer around. Understanding these systems for either modernization or even regular maintenance has been a challenge. An extensive application may have natural boundaries based on its code dependencies and architecture. There are also other logical boundaries in an enterprise setting driven by business functions, data domains, etc. Due to these complications, the system architects generally plan their modernization across these logical boundaries in parts, thereby adopting an incremental approach for the modernization journey of the entire system. In this work, we present a so6ware system analysis tool that allows a subject ma=er expert (SME) or system architect to analyze a large so6ware system incrementally. We analyze the source code and other artifacts (such as data schema) to create a knowledge graph using a customizable ontology/schema. Entities and relations in our ontology can be defined for any combination of programming languages and platforms. Using this knowledge graph, the analyst can then define logical boundaries around dependent Entities (e.g. Programs, Transactions, Database Tables etc.). Our tool then presents different views showcasing the dependencies from the newly defined boundary to/from the other logical groups of the system. This exercise is repeated interactively to 1) Identify the Entities and groupings of interest for a modernization task and 2) Understand how a change in one part of the system may affect the other parts. To validate the efficacy of our tool, we provide an initial study of our system on two client applications. 

---
# A Split-then-Join Approach to Abstractive Summarization for Very Long Documents in a Low Resource Setting 

**Authors**: Lhuqita Fazry  

**Link**: [PDF](https://arxiv.org/pdf/2505.06862)  

**Abstract**: $\texttt{BIGBIRD-PEGASUS}$ model achieves $\textit{state-of-the-art}$ on abstractive text summarization for long documents. However it's capacity still limited to maximum of $4,096$ tokens, thus caused performance degradation on summarization for very long documents. Common method to deal with the issue is to truncate the documents. In this reasearch, we'll use different approach. We'll use the pretrained $\texttt{BIGBIRD-PEGASUS}$ model by fine tuned the model on other domain dataset. First, we filter out all documents which length less than $20,000$ tokens to focus on very long documents. To prevent domain shifting problem and overfitting on transfer learning due to small dataset, we augment the dataset by splitting document-summary training pair into parts, to fit the document into $4,096$ tokens. Source code available on $\href{this https URL}{this https URL}$. 

---
# Burger: Robust Graph Denoising-augmentation Fusion and Multi-semantic Modeling in Social Recommendation 

**Authors**: Yuqin Lan  

**Link**: [PDF](https://arxiv.org/pdf/2505.06612)  

**Abstract**: In the era of rapid development of social media, social recommendation systems as hybrid recommendation systems have been widely applied. Existing methods capture interest similarity between users to filter out interest-irrelevant relations in social networks that inevitably decrease recommendation accuracy, however, limited research has a focus on the mutual influence of semantic information between the social network and the user-item interaction network for further improving social recommendation. To address these issues, we introduce a social \underline{r}ecommendation model with ro\underline{bu}st g\underline{r}aph denoisin\underline{g}-augmentation fusion and multi-s\underline{e}mantic Modeling(Burger). Specifically, we firstly propose to construct a social tensor in order to smooth the training process of the model. Then, a graph convolutional network and a tensor convolutional network are employed to capture user's item preference and social preference, respectively. Considering the different semantic information in the user-item interaction network and the social network, a bi-semantic coordination loss is proposed to model the mutual influence of semantic information. To alleviate the interference of interest-irrelevant relations on multi-semantic modeling, we further use Bayesian posterior probability to mine potential social relations to replace social noise. Finally, the sliding window mechanism is utilized to update the social tensor as the input for the next iteration. Extensive experiments on three real datasets show Burger has a superior performance compared with the state-of-the-art models. 

---
# MacRAG: Compress, Slice, and Scale-up for Multi-Scale Adaptive Context RAG 

**Authors**: Woosang Lim, Zekun Li, Gyuwan Kim, Sungyoung Ji, HyeonJung Kim, Kyuri Choi, Jin Hyuk Lim, Kyungpyo Park, William Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.06569)  

**Abstract**: Long-context (LC) Large Language Models (LLMs) combined with Retrieval-Augmented Generation (RAG) hold strong potential for complex multi-hop and large-document tasks. However, existing RAG systems often suffer from imprecise retrieval, incomplete context coverage under constrained context windows, and fragmented information caused by suboptimal context construction. We introduce Multi-scale Adaptive Context RAG (MacRAG), a hierarchical retrieval framework that compresses and partitions documents into coarse-to-fine granularities, then adaptively merges relevant contexts through chunk- and document-level expansions in real time. By starting from the finest-level retrieval and progressively incorporating higher-level and broader context, MacRAG constructs effective query-specific long contexts, optimizing both precision and coverage. Evaluations on the challenging LongBench expansions of HotpotQA, 2WikiMultihopQA, and Musique confirm that MacRAG consistently surpasses baseline RAG pipelines on single- and multi-step generation with Llama-3.1-8B, Gemini-1.5-pro, and GPT-4o. Our results establish MacRAG as an efficient, scalable solution for real-world long-context, multi-hop reasoning. Our code is available at this https URL. 

---
# Tweedie Regression for Video Recommendation System 

**Authors**: Yan Zheng, Qiang Chen, Chenglei Niu  

**Link**: [PDF](https://arxiv.org/pdf/2505.06445)  

**Abstract**: Modern recommendation systems aim to increase click-through rates (CTR) for better user experience, through commonly treating ranking as a classification task focused on predicting CTR. However, there is a gap between this method and the actual objectives of businesses across different sectors. In video recommendation services, the objective of video on demand (VOD) extends beyond merely encouraging clicks, but also guiding users to discover their true interests, leading to increased watch time. And longer users watch time will leads to more revenue through increased chances of presenting online display advertisements. This research addresses the issue by redefining the problem from classification to regression, with a focus on maximizing revenue through user viewing time. Due to the lack of positive labels on recommendation, the study introduces Tweedie Loss Function, which is better suited in this scenario than the traditional mean square error loss. The paper also provides insights on how Tweedie process capture users diverse interests. Our offline simulation and online A/B test revealed that we can substantially enhance our core business objectives: user engagement in terms of viewing time and, consequently, revenue. Additionally, we provide a theoretical comparison between the Tweedie Loss and the commonly employed viewing time weighted Logloss, highlighting why Tweedie Regression stands out as an efficient solution. We further outline a framework for designing a loss function that focuses on a singular objective. 

---
# OpenSky Report 2025: Improving Crowdsourced Flight Trajectories with ADS-C Data 

**Authors**: Junzi Sun, Xavier Olive, Martin Strohmeier, Vincent Lenders  

**Link**: [PDF](https://arxiv.org/pdf/2505.06254)  

**Abstract**: The OpenSky Network has been collecting and providing crowdsourced air traffic surveillance data since 2013. The network has primarily focused on Automatic Dependent Surveillance--Broadcast (ADS-B) data, which provides high-frequency position updates over terrestrial areas. However, the ADS-B signals are limited over oceans and remote regions, where ground-based receivers are scarce. To address these coverage gaps, the OpenSky Network has begun incorporating data from the Automatic Dependent Surveillance--Contract (ADS-C) system, which uses satellite communication to track aircraft positions over oceanic regions and remote areas. In this paper, we analyze a dataset of over 720,000 ADS-C messages collected in 2024 from around 2,600 unique aircraft via the Alphasat satellite, covering Europe, Africa, and parts of the Atlantic Ocean. We present our approach to combining ADS-B and ADS-C data to construct detailed long-haul flight paths, particularly for transatlantic and African routes. Our findings demonstrate that this integration significantly improves trajectory reconstruction accuracy, allowing for better fuel consumption and emissions estimates. We illustrate how combined data captures flight patterns across previously underrepresented regions across Africa. Despite coverage limitations, this work marks an important advancement in providing open access to global flight trajectory data, enabling new research opportunities in air traffic management, environmental impact assessment, and aviation safety. 

---
