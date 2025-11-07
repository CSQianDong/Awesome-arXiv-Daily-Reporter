# LLM-as-a-Judge: Toward World Models for Slate Recommendation Systems 

**Authors**: Baptiste Bonin, Maxime Heuillet, Audrey Durand  

**Link**: [PDF](https://arxiv.org/pdf/2511.04541)  

**Abstract**: Modeling user preferences across domains remains a key challenge in slate recommendation (i.e. recommending an ordered sequence of items) research. We investigate how Large Language Models (LLM) can effectively act as world models of user preferences through pairwise reasoning over slates. We conduct an empirical study involving several LLMs on three tasks spanning different datasets. Our results reveal relationships between task performance and properties of the preference function captured by LLMs, hinting towards areas for improvement and highlighting the potential of LLMs as world models in recommender systems. 

---
# Denoised Recommendation Model with Collaborative Signal Decoupling 

**Authors**: Zefeng Li, Ning Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.04237)  

**Abstract**: Although the collaborative filtering (CF) algorithm has achieved remarkable performance in recommendation systems, it suffers from suboptimal recommendation performance due to noise in the user-item interaction matrix. Numerous noise-removal studies have improved recommendation models, but most existing approaches conduct denoising on a single graph. This may cause attenuation of collaborative signals: removing edges between two nodes can interrupt paths between other nodes, weakening path-dependent collaborative information. To address these limitations, this study proposes a novel GNN-based CF model called DRCSD for denoising unstable interactions. DRCSD includes two core modules: a collaborative signal decoupling module (decomposes signals into distinct orders by structural characteristics) and an order-wise denoising module (performs targeted denoising on each order). Additionally, the information aggregation mechanism of traditional GNN-based CF models is modified to avoid cross-order signal interference until the final pooling operation. Extensive experiments on three public real-world datasets show that DRCSD has superior robustness against unstable interactions and achieves statistically significant performance improvements in recommendation accuracy metrics compared to state-of-the-art baseline models. 

---
# Coordination-Free Lane Partitioning for Convergent ANN Search 

**Authors**: Carl Kugblenu, Petri Vuorimaa  

**Link**: [PDF](https://arxiv.org/pdf/2511.04221)  

**Abstract**: Production vector search systems often fan out each query across parallel lanes (threads, replicas, or shards) to meet latency service-level objectives (SLOs). In practice, these lanes rediscover the same candidates, so extra compute does not increase coverage. We present a coordination-free lane partitioner that turns duplication into complementary work at the same cost and deadline. For each query we (1) build a deterministic candidate pool sized to the total top-k budget, (2) apply a per-query pseudorandom permutation, and (3) assign each lane a disjoint slice of positions. Lanes then return different results by construction, with no runtime coordination.
At equal cost with four lanes (total candidate budget 64), on SIFT1M (1M SIFT feature vectors) with Hierarchical Navigable Small World graphs (HNSW) recall@10 rises from 0.249 to 0.999 while lane overlap falls from nearly 100% to 0%. On MS MARCO (8.8M passages) with HNSW, hit@10 improves from 0.200 to 0.601 and Mean Reciprocal Rank at 10 (MRR@10) from 0.133 to 0.330. For inverted file (IVF) indexes we see smaller but consistent gains (for example, +11% on MS MARCO) by de-duplicating list routing. A microbenchmark shows planner overhead of ~37 microseconds per query (mean at the main setting) with linear growth in the number of merged candidates.
These results yield a simple operational guideline: size the per-query pool to the total budget, deterministically partition positions across lanes, and turn redundant fan-out into complementary coverage without changing budget or deadline. 

---
# Transforming Mentorship: An AI Powered Chatbot Approach to University Guidance 

**Authors**: Mashrur Rahman, Mantaqa abedin, Monowar Zamil Abir, Faizul Islam Ansari, Adib Reza, Farig Yousuf Sadeque, Niloy Farhan  

**Link**: [PDF](https://arxiv.org/pdf/2511.04172)  

**Abstract**: University students face immense challenges during their undergraduate lives, often being deprived of personalized on-demand guidance that mentors fail to provide at scale. Digital tools exist, but there is a serious lack of customized coaching for newcomers. This paper presents an AI-powered chatbot that will serve as a mentor for the students of BRAC University. The main component is a data ingestion pipeline that efficiently processes and updates information from diverse sources, such as CSV files and university webpages. The chatbot retrieves information through a hybrid approach, combining BM25 lexical ranking with ChromaDB semantic retrieval, and uses a Large Language Model, LLaMA-3.3-70B, to generate conversational responses. The generated text was found to be semantically highly relevant, with a BERTScore of 0.831 and a METEOR score of 0.809. The data pipeline was also very efficient, taking 106.82 seconds for updates, compared to 368.62 seconds for new data. This chatbot will be able to help students by responding to their queries, helping them to get a better understanding of university life, and assisting them to plan better routines for their semester in the open-credit university. 

---
# E-CARE: An Efficient LLM-based Commonsense-Augmented Framework for E-Commerce 

**Authors**: Ge Zhang, Rohan Deepak Ajwani, Tony Zheng, Hongjian Gu, Yaochen Hu, Wei Guo, Mark Coates, Yingxue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.04087)  

**Abstract**: Finding relevant products given a user query plays a pivotal role in an e-commerce platform, as it can spark shopping behaviors and result in revenue gains. The challenge lies in accurately predicting the correlation between queries and products. Recently, mining the cross-features between queries and products based on the commonsense reasoning capacity of Large Language Models (LLMs) has shown promising performance. However, such methods suffer from high costs due to intensive real-time LLM inference during serving, as well as human annotations and potential Supervised Fine Tuning (SFT). To boost efficiency while leveraging the commonsense reasoning capacity of LLMs for various e-commerce tasks, we propose the Efficient Commonsense-Augmented Recommendation Enhancer (E-CARE). During inference, models augmented with E-CARE can access commonsense reasoning with only a single LLM forward pass per query by utilizing a commonsense reasoning factor graph that encodes most of the reasoning schema from powerful LLMs. The experiments on 2 downstream tasks show an improvement of up to 12.1% on precision@5. 

---
# Caption Injection for Optimization in Generative Search Engine 

**Authors**: Xiaolu Chen, Yong Liao  

**Link**: [PDF](https://arxiv.org/pdf/2511.04080)  

**Abstract**: Generative Search Engines (GSEs) leverage Retrieval-Augmented Generation (RAG) techniques and Large Language Models (LLMs) to integrate multi-source information and provide users with accurate and comprehensive responses. Unlike traditional search engines that present results in ranked lists, GSEs shift users' attention from sequential browsing to content-driven subjective perception, driving a paradigm shift in information retrieval. In this context, enhancing the subjective visibility of content through Generative Search Engine Optimization (G-SEO) methods has emerged as a new research focus. With the rapid advancement of Multimodal Retrieval-Augmented Generation (MRAG) techniques, GSEs can now efficiently integrate text, images, audio, and video, producing richer responses that better satisfy complex information needs. Existing G-SEO methods, however, remain limited to text-based optimization and fail to fully exploit multimodal data. To address this gap, we propose Caption Injection, the first multimodal G-SEO approach, which extracts captions from images and injects them into textual content, integrating visual semantics to enhance the subjective visibility of content in generative search scenarios. We systematically evaluate Caption Injection on MRAMG, a benchmark for MRAG, under both unimodal and multimodal settings. Experimental results show that Caption Injection significantly outperforms text-only G-SEO baselines under the G-Eval metric, demonstrating the necessity and effectiveness of multimodal integration in G-SEO to improve user-perceived content visibility. 

---
# RUST-BENCH: Benchmarking LLM Reasoning on Unstructured Text within Structured Tables 

**Authors**: Nikhil Abhyankar, Purvi Chaurasia, Sanchit Kabra, Ananya Srivastava, Vivek Gupta, Chandan K. Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2511.04491)  

**Abstract**: Existing tabular reasoning benchmarks mostly test models on small, uniform tables, underrepresenting the complexity of real-world data and giving an incomplete view of Large Language Models' (LLMs) reasoning abilities. Real tables are long, heterogeneous, and domain-specific, mixing structured fields with free text and requiring multi-hop reasoning across thousands of tokens. To address this gap, we introduce RUST-BENCH, a benchmark of 7966 questions from 2031 real-world tables spanning two domains: i) RB-Science (NSF grant records) and ii) RB-Sports (NBA statistics). Unlike prior work, RUST-BENCH evaluates LLMs jointly across scale, heterogeneity, domain specificity, and reasoning complexity. Experiments with open-source and proprietary models show that LLMs struggle with heterogeneous schemas and complex multi-hop inference, revealing persistent weaknesses in current architectures and prompting strategies. RUST-BENCH establishes a challenging new testbed for advancing tabular reasoning research. 

---
# Ground-Truth Subgraphs for Better Training and Evaluation of Knowledge Graph Augmented LLMs 

**Authors**: Alberto Cattaneo, Carlo Luschi, Daniel Justus  

**Link**: [PDF](https://arxiv.org/pdf/2511.04473)  

**Abstract**: Retrieval of information from graph-structured knowledge bases represents a promising direction for improving the factuality of LLMs. While various solutions have been proposed, a comparison of methods is difficult due to the lack of challenging QA datasets with ground-truth targets for graph retrieval. We present SynthKGQA, a framework for generating high-quality synthetic Knowledge Graph Question Answering datasets from any Knowledge Graph, providing the full set of ground-truth facts in the KG to reason over each question. We show how, in addition to enabling more informative benchmarking of KG retrievers, the data produced with SynthKGQA also allows us to train better models. We apply SynthKGQA to Wikidata to generate GTSQA, a new dataset designed to test zero-shot generalization abilities of KG retrievers with respect to unseen graph structures and relation types, and benchmark popular solutions for KG-augmented LLMs on it. 

---
# On the Brittleness of CLIP Text Encoders 

**Authors**: Allie Tran, Luca Rossetto  

**Link**: [PDF](https://arxiv.org/pdf/2511.04247)  

**Abstract**: Multimodal co-embedding models, especially CLIP, have advanced the state of the art in zero-shot classification and multimedia information retrieval in recent years by aligning images and text in a shared representation space. However, such modals trained on a contrastive alignment can lack stability towards small input perturbations. Especially when dealing with manually expressed queries, minor variations in the query can cause large differences in the ranking of the best-matching results. In this paper, we present a systematic analysis of the effect of multiple classes of non-semantic query perturbations in an multimedia information retrieval scenario. We evaluate a diverse set of lexical, syntactic, and semantic perturbations across multiple CLIP variants using the TRECVID Ad-Hoc Video Search queries and the V3C1 video collection. Across models, we find that syntactic and semantic perturbations drive the largest instabilities, while brittleness is concentrated in trivial surface edits such as punctuation and case. Our results highlight robustness as a critical dimension for evaluating vision-language models beyond benchmark accuracy. 

---
# Publication Trend in DESIDOC Journal of Library and Information Technology during 2013-2017: A Scientometric Approach 

**Authors**: M Sadik Batcha, S Roselin Jahina, Muneer Ahmad  

**Link**: [PDF](https://arxiv.org/pdf/2511.04082)  

**Abstract**: DESIDOC Journal of Library & Information Technology (DJLIT) formerly known as DESIDOC Bulletin of Information Technology is a peer-reviewed, open access, bimonthly journal. This paper presents a Scientometric analysis of the DESIDOC Journal. The paper analyses the pattern of growth of the research output published in the journal, pattern of authorship, author productivity, and, subjects covered to the papers over the period (2013-2017). It is found that 227 papers were published during the period of study (2001-2012). The maximum numbers of articles were collaborative in nature. The subject concentration of the journal noted is Scientometrics. The maximum numbers of articles (65%) have ranged their thought contents between 6 and 10 pages. The study applied standard formula and statistical tools to bring out the factual result. 

---
# Two Decades of Research at the University of Lagos (2004-2023): A Scientometric Analysis of Productivity, Collaboration, and Impact 

**Authors**: Muneer Ahmad, Samuel Ibor Ubi  

**Link**: [PDF](https://arxiv.org/pdf/2511.04075)  

**Abstract**: This paper presents a scientometric analysis of research output from the University of Lagos, focusing on the two decades spanning 2004 to 2023. Using bibliometric data retrieved from the Web of Science, we examine trends in publication volume, collaboration patterns, citation impact, and the most prolific authors, departments, and research domains at the university. The study reveals a consistent increase in research productivity, with the highest publication output recorded in 2023. Health Sciences, Engineering, and Social Sciences are identified as dominant fields, reflecting the university's interdisciplinary research strengths. Collaborative efforts, both locally and internationally, show a positive correlation with higher citation impact, with the United States and the United Kingdom being the leading international collaborators. Notably, open-access publications account for a significant portion of the university's research output, enhancing visibility and citation rates. The findings offer valuable insights into the university's research performance over the past two decades, providing a foundation for strategic planning and policy formulation to foster research excellence and global impact. 

---
# Learning Filter-Aware Distance Metrics for Nearest Neighbor Search with Multiple Filters 

**Authors**: Ananya Sutradhar, Suryansh Gupta, Ravishankar Krishnaswamy, Haiyang Xu, Aseem Rastogi, Gopal Srinivasa  

**Link**: [PDF](https://arxiv.org/pdf/2511.04073)  

**Abstract**: Filtered Approximate Nearest Neighbor (ANN) search retrieves the closest vectors for a query vector from a dataset. It enforces that a specified set of discrete labels $S$ for the query must be included in the labels of each retrieved vector. Existing graph-based methods typically incorporate filter awareness by assigning fixed penalties or prioritizing nodes based on filter satisfaction. However, since these methods use fixed, data in- dependent penalties, they often fail to generalize across datasets with diverse label and vector distributions. In this work, we propose a principled alternative that learns the optimal trade-off between vector distance and filter match directly from the data, rather than relying on fixed penalties. We formulate this as a constrained linear optimization problem, deriving weights that better reflect the underlying filter distribution and more effectively address the filtered ANN search problem. These learned weights guide both the search process and index construction, leading to graph structures that more effectively capture the underlying filter distribution and filter semantics. Our experiments demonstrate that adapting the distance function to the data significantly im- proves accuracy by 5-10% over fixed-penalty methods, providing a more flexible and generalizable framework for the filtered ANN search problem. 

---
# KnowThyself: An Agentic Assistant for LLM Interpretability 

**Authors**: Suraj Prasai, Mengnan Du, Ying Zhang, Fan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.03878)  

**Abstract**: We develop KnowThyself, an agentic assistant that advances large language model (LLM) interpretability. Existing tools provide useful insights but remain fragmented and code-intensive. KnowThyself consolidates these capabilities into a chat-based interface, where users can upload models, pose natural language questions, and obtain interactive visualizations with guided explanations. At its core, an orchestrator LLM first reformulates user queries, an agent router further directs them to specialized modules, and the outputs are finally contextualized into coherent explanations. This design lowers technical barriers and provides an extensible platform for LLM inspection. By embedding the whole process into a conversational workflow, KnowThyself offers a robust foundation for accessible LLM interpretability. 

---
