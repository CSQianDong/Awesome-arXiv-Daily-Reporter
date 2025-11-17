# SRLF: An Agent-Driven Set-Wise Reflective Learning Framework for Sequential Recommendation 

**Authors**: Jiahao Wang, Bokang Fu, Yu Zhu, Yuli Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.11370)  

**Abstract**: LLM-based agents are emerging as a promising paradigm for simulating user behavior to enhance recommender systems. However, their effectiveness is often limited by existing studies that focus on modeling user ratings for individual items. This point-wise approach leads to prevalent issues such as inaccurate user preference comprehension and rigid item-semantic representations.
To address these limitations, we propose the novel Set-wise Reflective Learning Framework (SRLF). Our framework operationalizes a closed-loop "assess-validate-reflect" cycle that harnesses the powerful in-context learning capabilities of LLMs. SRLF departs from conventional point-wise assessment by formulating a holistic judgment on an entire set of items. It accomplishes this by comprehensively analyzing both the intricate interrelationships among items within the set and their collective alignment with the user's preference profile. This method of set-level contextual understanding allows our model to capture complex relational patterns essential to user behavior, making it significantly more adept for sequential recommendation. Extensive experiments validate our approach, confirming that this set-wise perspective is crucial for achieving state-of-the-art performance in sequential recommendation tasks. 

---
# MOON Embedding: Multimodal Representation Learning for E-commerce Search Advertising 

**Authors**: Chenghan Fu, Daoze Zhang, Yukang Lin, Zhanheng Nie, Xiang Zhang, Jianyu Liu, Yueran Liu, Wanxian Guan, Pengjie Wang, Jian Xu, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2511.11305)  

**Abstract**: We introduce MOON, our comprehensive set of sustainable iterative practices for multimodal representation learning for e-commerce applications. MOON has already been fully deployed across all stages of Taobao search advertising system, including retrieval, relevance, ranking, and so on. The performance gains are particularly significant on click-through rate (CTR) prediction task, which achieves an overall +20.00% online CTR improvement. Over the past three years, this project has delivered the largest improvement on CTR prediction task and undergone five full-scale iterations. Throughout the exploration and iteration of our MOON, we have accumulated valuable insights and practical experience that we believe will benefit the research community. MOON contains a three-stage training paradigm of "Pretraining, Post-training, and Application", allowing effective integration of multimodal representations with downstream tasks. Notably, to bridge the misalignment between the objectives of multimodal representation learning and downstream training, we define the exchange rate to quantify how effectively improvements in an intermediate metric can translate into downstream gains. Through this analysis, we identify the image-based search recall as a critical intermediate metric guiding the optimization of multimodal models. Over three years and five iterations, MOON has evolved along four critical dimensions: data processing, training strategy, model architecture, and downstream application. The lessons and insights gained through the iterative improvements will also be shared. As part of our exploration into scaling effects in the e-commerce field, we further conduct a systematic study of the scaling laws governing multimodal representation learning, examining multiple factors such as the number of training tokens, negative samples, and the length of user behavior sequences. 

---
# Align$^3$GR: Unified Multi-Level Alignment for LLM-based Generative Recommendation 

**Authors**: Wencai Ye, Mingjie Sun, Shuhang Chen, Wenjin Wu, Peng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2511.11255)  

**Abstract**: Large Language Models (LLMs) demonstrate significant advantages in leveraging structured world knowledge and multi-step reasoning capabilities. However, fundamental challenges arise when transforming LLMs into real-world recommender systems due to semantic and behavioral misalignment. To bridge this gap, we propose Align$^3$GR, a novel framework that unifies token-level, behavior modeling-level, and preference-level alignment. Our approach introduces: Dual tokenization fusing user-item semantic and collaborative signals. Enhanced behavior modeling with bidirectional semantic alignment. Progressive DPO strategy combining self-play (SP-DPO) and real-world feedback (RF-DPO) for dynamic preference adaptation. Experiments show Align$^3$GR outperforms the SOTA baseline by +17.8% in Recall@10 and +20.2% in NDCG@10 on the public dataset, with significant gains in online A/B tests and full-scale deployment on an industrial large-scale recommendation platform. 

---
# Enhancing Group Recommendation using Soft Impute Singular Value Decomposition 

**Authors**: Mubaraka Sani Ibrahim, Isah Charles Saidu, Lehel Csato  

**Link**: [PDF](https://arxiv.org/pdf/2511.11172)  

**Abstract**: The growing popularity of group activities increased the need to develop methods for providing recommendations to a group of users based on the collective preferences of the group members. Several group recommender systems have been proposed, but these methods often struggle due to sparsity and high-dimensionality of the available data, common in many real-world applications. In this paper, we propose a group recommender system called Group Soft-Impute SVD, which leverages soft-impute singular value decomposition to enhance group recommendations. This approach addresses the challenge of sparse high-dimensional data using low-rank matrix completion. We compared the performance of Group Soft-Impute SVD with Group MF based approaches and found that our method outperforms the baselines in recall for small user groups while achieving comparable results across all group sizes when tasked on Goodbooks, Movielens, and Synthetic datasets. Furthermore, our method recovers lower matrix ranks than the baselines, demonstrating its effectiveness in handling high-dimensional data. 

---
# GovScape: A Public Multimodal Search System for 70 Million Pages of Government PDFs 

**Authors**: Kyle Deeds, Ying-Hsiang Huang, Claire Gong, Shreya Shaji, Alison Yan, Leslie Harka, Samuel J Klein, Shannon Zejiang Shen, Mark Phillips, Trevor Owens, Benjamin Charles Germain Lee  

**Link**: [PDF](https://arxiv.org/pdf/2511.11010)  

**Abstract**: Efforts over the past three decades have produced web archives containing billions of webpage snapshots and petabytes of data. The End of Term Web Archive alone contains, among other file types, millions of PDFs produced by the federal government. While preservation with web archives has been successful, significant challenges for access and discoverability remain. For example, current affordances for browsing the End of Term PDFs are limited to downloading and browsing individual PDFs, as well as performing basic keyword search across them. In this paper, we introduce GovScape, a public search system that supports multimodal searches across 10,015,993 federal government PDFs from the 2020 End of Term crawl (70,958,487 total PDF pages) - to our knowledge, all renderable PDFs in the 2020 crawl that are 50 pages or under. GovScape supports four primary forms of search over these 10 million PDFs: in addition to providing (1) filter conditions over metadata facets including domain and crawl date and (2) exact text search against the PDF text, we provide (3) semantic text search and (4) visual search against the PDFs across individual pages, enabling users to structure queries such as "redacted documents" or "pie charts." We detail the constituent components of GovScape, including the search affordances, embedding pipeline, system architecture, and open source codebase. Significantly, the total estimated compute cost for GovScape's pre-processing pipeline for 10 million PDFs was approximately $1,500, equivalent to 47,000 PDF pages per dollar spent on compute, demonstrating the potential for immediate scalability. Accordingly, we outline steps that we have already begun pursuing toward multimodal search at the 100+ million PDF scale. GovScape can be found at this https URL. 

---
# LEMUR: Large scale End-to-end MUltimodal Recommendation 

**Authors**: Xintian Han, Honggang Chen, Quan Lin, Jingyue Gao, Xiangyuan Ren, Lifei Zhu, Zhisheng Ye, Shikang Wu, XiongHang Xie, Xiaochu Gan, Bingzheng Wei, Peng Xu, Zhe Wang, Yuchao Zheng, Jingjian Lin, Di Wu, Junfeng Ge  

**Link**: [PDF](https://arxiv.org/pdf/2511.10962)  

**Abstract**: Traditional ID-based recommender systems often struggle with cold-start and generalization challenges. Multimodal recommendation systems, which leverage textual and visual data, offer a promising solution to mitigate these issues. However, existing industrial approaches typically adopt a two-stage training paradigm: first pretraining a multimodal model, then applying its frozen representations to train the recommendation model. This decoupled framework suffers from misalignment between multimodal learning and recommendation objectives, as well as an inability to adapt dynamically to new data. To address these limitations, we propose LEMUR, the first large-scale multimodal recommender system trained end-to-end from raw data. By jointly optimizing both the multimodal and recommendation components, LEMUR ensures tighter alignment with downstream objectives while enabling real-time parameter updates. Constructing multimodal sequential representations from user history often entails prohibitively high computational costs. To alleviate this bottleneck, we propose a novel memory bank mechanism that incrementally accumulates historical multimodal representations throughout the training process. After one month of deployment in Douyin Search, LEMUR has led to a 0.843% reduction in query change rate decay and a 0.81% improvement in QAUC. Additionally, LEMUR has shown significant gains across key offline metrics for Douyin Advertisement. Our results validate the superiority of end-to-end multimodal recommendation in real-world industrial scenarios. 

---
# GRIN Transfer: A production-ready tool for libraries to retrieve digital copies from Google Books 

**Authors**: Liza Daly, Matteo Cargnelutti, Catherine Brobston, John Hess, Greg Leppert, Amanda Watson, Jonathan Zittrain  

**Link**: [PDF](https://arxiv.org/pdf/2511.11447)  

**Abstract**: Publicly launched in 2004, the Google Books project has scanned tens of millions of items in partnership with libraries around the world. As part of this project, Google created the Google Return Interface (GRIN). Through this platform, libraries can access their scanned collections, the associated metadata, and the ongoing OCR and metadata improvements that become available as Google reprocesses these collections using new technologies. When downloading the Harvard Library Google Books collection from GRIN to develop the Institutional Books dataset, we encountered several challenges related to rate-limiting and atomized metadata within the GRIN platform. To overcome these challenges and help other libraries make more robust use of their Google Books collections, this technical report introduces the initial release of GRIN Transfer. This open-source and production-ready Python pipeline allows partner libraries to efficiently retrieve their Google Books collections from GRIN. This report also introduces an updated version of our Institutional Books 1.0 pipeline, initially used to analyze, augment, and assemble the Institutional Books 1.0 dataset. We have revised this pipeline for compatibility with the output format of GRIN Transfer. A library could pair these two tools to create an end-to-end processing pipeline for their Google Books collection to retrieve, structure, and enhance data available from GRIN. This report gives an overview of how GRIN Transfer was designed to optimize for reliability and usability in different environments, as well as guidance on configuration for various use cases. 

---
# Unlocking Advanced Graph Machine Learning Insights through Knowledge Completion on Neo4j Graph Database 

**Authors**: Rosario Napoli, Antonio Celesti, Massimo Villari, Maria Fazio  

**Link**: [PDF](https://arxiv.org/pdf/2511.11399)  

**Abstract**: Graph Machine Learning (GML) with Graph Databases (GDBs) has gained significant relevance in recent years, due to its ability to handle complex interconnected data and apply ML techniques using Graph Data Science (GDS). However, a critical gap exists in the current way GDB-GML applications analyze data, especially in terms of Knowledge Completion (KC) in Knowledge Graphs (KGs). In particular, current architectures ignore KC, working on datasets that appear incomplete or fragmented, despite they actually contain valuable hidden knowledge. This limitation may cause wrong interpretations when these data are used as input for GML models.
This paper proposes an innovative architecture that integrates a KC phase into GDB-GML applications, demonstrating how revealing hidden knowledge can heavily impact datasets' behavior and metrics. For this purpose, we introduce scalable transitive relationships, which are links that propagate information over the network and modelled by a decay function, allowing a deterministic knowledge flows across multiple nodes.
Experimental results demonstrate that our intuition radically reshapes both topology and overall dataset dynamics, underscoring the need for this new GDB-GML architecture to produce better models and unlock the full potential of graph-based data analysis. 

---
# SQuaD: The Software Quality Dataset 

**Authors**: Mikel Robredo, Matteo Esposito, Davide Taibi, Rafael Peñaloza, Valentina Lenarduzzi  

**Link**: [PDF](https://arxiv.org/pdf/2511.11265)  

**Abstract**: Software quality research increasingly relies on large-scale datasets that measure both the product and process aspects of software systems. However, existing resources often focus on limited dimensions, such as code smells, technical debt, or refactoring activity, thereby restricting comprehensive analyses across time and quality dimensions. To address this gap, we present the Software Quality Dataset (SQuaD), a multi-dimensional, time-aware collection of software quality metrics extracted from 450 mature open-source projects across diverse ecosystems, including Apache, Mozilla, FFmpeg, and the Linux kernel. By integrating nine state-of-the-art static analysis tools, i.e., SonarQube, CodeScene, PMD, Understand, CK, JaSoMe, RefactoringMiner, RefactoringMiner++, and PyRef, our dataset unifies over 700 unique metrics at method, class, file, and project levels. Covering a total of 63,586 analyzed project releases, SQuaD also provides version control and issue-tracking histories, software vulnerability data (CVE/CWE), and process metrics proven to enhance Just-In-Time (JIT) defect prediction. The SQuaD enables empirical research on maintainability, technical debt, software evolution, and quality assessment at unprecedented scale. We also outline emerging research directions, including automated dataset updates and cross-project quality modeling to support the continuous evolution of software analytics. The dataset is publicly available on ZENODO (DOI: https://doi.org/10.5281/zenodo.17566690). 

---
# Learn to Select: Exploring Label Distribution Divergence for In-Context Demonstration Selection in Text Classification 

**Authors**: Ye Jiang, Taihang Wang, Youzheng Liu, Yimin Wang, Yuhan Xia, Yunfei Long  

**Link**: [PDF](https://arxiv.org/pdf/2511.10675)  

**Abstract**: In-context learning (ICL) for text classification, which uses a few input-label demonstrations to describe a task, has demonstrated impressive performance on large language models (LLMs). However, the selection of in-context demonstrations plays a crucial role and can significantly affect LLMs' performance. Most existing demonstration selection methods primarily focus on semantic similarity between test inputs and demonstrations, often overlooking the importance of label distribution alignment. To address this limitation, we propose a two-stage demonstration selection method, TopK + Label Distribution Divergence (L2D), which leverages a fine-tuned BERT-like small language model (SLM) to generate label distributions and calculate their divergence for both test inputs and candidate demonstrations. This enables the selection of demonstrations that are not only semantically similar but also aligned in label distribution with the test input. Extensive experiments across seven text classification benchmarks show that our method consistently outperforms previous demonstration selection strategies. Further analysis reveals a positive correlation between the performance of LLMs and the accuracy of the underlying SLMs used for label distribution estimation. 

---
# Information Extraction From Fiscal Documents Using LLMs 

**Authors**: Vikram Aggarwal, Jay Kulkarni, Aditi Mascarenhas, Aakriti Narang, Siddarth Raman, Ajay Shah, Susan Thomas  

**Link**: [PDF](https://arxiv.org/pdf/2511.10659)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in text comprehension, but their ability to process complex, hierarchical tabular data remains underexplored. We present a novel approach to extracting structured data from multi-page government fiscal documents using LLM-based techniques. Applied to annual fiscal documents from the State of Karnataka in India (200+ pages), our method achieves high accuracy through a multi-stage pipeline that leverages domain knowledge, sequential context, and algorithmic validation. A large challenge with traditional OCR methods is the inability to verify the accurate extraction of numbers. When applied to fiscal data, the inherent structure of fiscal tables, with totals at each level of the hierarchy, allows for robust internal validation of the extracted data. We use these hierarchical relationships to create multi-level validation checks. We demonstrate that LLMs can read tables and also process document-specific structural hierarchies, offering a scalable process for converting PDF-based fiscal disclosures into research-ready databases. Our implementation shows promise for broader applications across developing country contexts. 

---
