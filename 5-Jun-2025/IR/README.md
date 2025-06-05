# Quantifying Query Fairness Under Unawareness 

**Authors**: Thomas Jaenich, Alejandro Moreo, Alessandro Fabris, Graham McDonald, Andrea Esuli, Iadh Ounis, Fabrizio Sebastiani  

**Link**: [PDF](https://arxiv.org/pdf/2506.04140)  

**Abstract**: Traditional ranking algorithms are designed to retrieve the most relevant items for a user's query, but they often inherit biases from data that can unfairly disadvantage vulnerable groups. Fairness in information access systems (IAS) is typically assessed by comparing the distribution of groups in a ranking to a target distribution, such as the overall group distribution in the dataset. These fairness metrics depend on knowing the true group labels for each item. However, when groups are defined by demographic or sensitive attributes, these labels are often unknown, leading to a setting known as "fairness under unawareness". To address this, group membership can be inferred using machine-learned classifiers, and group prevalence is estimated by counting the predicted labels. Unfortunately, such an estimation is known to be unreliable under dataset shift, compromising the accuracy of fairness evaluations. In this paper, we introduce a robust fairness estimator based on quantification that effectively handles multiple sensitive attributes beyond binary classifications. Our method outperforms existing baselines across various sensitive attributes and, to the best of our knowledge, is the first to establish a reliable protocol for measuring fairness under unawareness across multiple queries and groups. 

---
# A Generative Adaptive Replay Continual Learning Model for Temporal Knowledge Graph Reasoning 

**Authors**: Zhiyu Zhang, Wei Chen, Youfang Lin, Huaiyu Wan  

**Link**: [PDF](https://arxiv.org/pdf/2506.04083)  

**Abstract**: Recent Continual Learning (CL)-based Temporal Knowledge Graph Reasoning (TKGR) methods focus on significantly reducing computational cost and mitigating catastrophic forgetting caused by fine-tuning models with new data. However, existing CL-based TKGR methods still face two key limitations: (1) They usually one-sidedly reorganize individual historical facts, while overlooking the historical context essential for accurately understanding the historical semantics of these facts; (2) They preserve historical knowledge by simply replaying historical facts, while ignoring the potential conflicts between historical and emerging facts. In this paper, we propose a Deep Generative Adaptive Replay (DGAR) method, which can generate and adaptively replay historical entity distribution representations from the whole historical context. To address the first challenge, historical context prompts as sampling units are built to preserve the whole historical context information. To overcome the second challenge, a pre-trained diffusion model is adopted to generate the historical distribution. During the generation process, the common features between the historical and current distributions are enhanced under the guidance of the TKGR model. In addition, a layer-by-layer adaptive replay mechanism is designed to effectively integrate historical and current distributions. Experimental results demonstrate that DGAR significantly outperforms baselines in reasoning and mitigating forgetting. 

---
# GORACS: Group-level Optimal Transport-guided Coreset Selection for LLM-based Recommender Systems 

**Authors**: Tiehua Mei, Hengrui Chen, Peng Yu, Jiaqing Liang, Deqing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.04015)  

**Abstract**: Although large language models (LLMs) have shown great potential in recommender systems, the prohibitive computational costs for fine-tuning LLMs on entire datasets hinder their successful deployment in real-world scenarios. To develop affordable and effective LLM-based recommender systems, we focus on the task of coreset selection which identifies a small subset of fine-tuning data to optimize the test loss, thereby facilitating efficient LLMs' fine-tuning. Although there exist some intuitive solutions of subset selection, including distribution-based and importance-based approaches, they often lead to suboptimal performance due to the misalignment with downstream fine-tuning objectives or weak generalization ability caused by individual-level sample selection. To overcome these challenges, we propose GORACS, which is a novel Group-level Optimal tRAnsport-guided Coreset Selection framework for LLM-based recommender systems. GORACS is designed based on two key principles for coreset selection: 1) selecting the subsets that minimize the test loss to align with fine-tuning objectives, and 2) enhancing model generalization through group-level data selection. Corresponding to these two principles, GORACS has two key components: 1) a Proxy Optimization Objective (POO) leveraging optimal transport and gradient information to bound the intractable test loss, thus reducing computational costs by avoiding repeated LLM retraining, and 2) a two-stage Initialization-Then-Refinement Algorithm (ITRA) for efficient group-level selection. Our extensive experiments across diverse recommendation datasets and tasks validate that GORACS significantly reduces fine-tuning costs of LLMs while achieving superior performance over the state-of-the-art baselines and full data training. The source code of GORACS are available at this https URL. 

---
# Graph-Embedding Empowered Entity Retrieval 

**Authors**: Emma J. Gerritse, Faegheh Hasibi, Arjen P. de Vries  

**Link**: [PDF](https://arxiv.org/pdf/2506.03895)  

**Abstract**: In this research, we investigate methods for entity retrieval using graph embeddings. While various methods have been proposed over the years, most utilize a single graph embedding and entity linking approach. This hinders our understanding of how different graph embedding and entity linking methods impact entity retrieval. To address this gap, we investigate the effects of three different categories of graph embedding techniques and five different entity linking methods. We perform a reranking of entities using the distance between the embeddings of annotated entities and the entities we wish to rerank. We conclude that the selection of both graph embeddings and entity linkers significantly impacts the effectiveness of entity retrieval. For graph embeddings, methods that incorporate both graph structure and textual descriptions of entities are the most effective. For entity linking, both precision and recall concerning concepts are important for optimal retrieval performance. Additionally, it is essential for the graph to encompass as many entities as possible. 

---
# Scaling Transformers for Discriminative Recommendation via Generative Pretraining 

**Authors**: Chunqi Wang, Bingchao Wu, Zheng Chen, Lei Shen, Bing Wang, Xiaoyi Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2506.03699)  

**Abstract**: Discriminative recommendation tasks, such as CTR (click-through rate) and CVR (conversion rate) prediction, play critical roles in the ranking stage of large-scale industrial recommender systems. However, training a discriminative model encounters a significant overfitting issue induced by data sparsity. Moreover, this overfitting issue worsens with larger models, causing them to underperform smaller ones. To address the overfitting issue and enhance model scalability, we propose a framework named GPSD (\textbf{G}enerative \textbf{P}retraining for \textbf{S}calable \textbf{D}iscriminative Recommendation), drawing inspiration from generative training, which exhibits no evident signs of overfitting. GPSD leverages the parameters learned from a pretrained generative model to initialize a discriminative model, and subsequently applies a sparse parameter freezing strategy. Extensive experiments conducted on both industrial-scale and publicly available datasets demonstrate the superior performance of GPSD. Moreover, it delivers remarkable improvements in online A/B tests. GPSD offers two primary advantages: 1) it substantially narrows the generalization gap in model training, resulting in better test performance; and 2) it leverages the scalability of Transformers, delivering consistent performance gains as models are scaled up. Specifically, we observe consistent performance improvements as the model dense parameters scale from 13K to 0.3B, closely adhering to power laws. These findings pave the way for unifying the architectures of recommendation models and language models, enabling the direct application of techniques well-established in large language models to recommendation models. 

---
# ProRank: Prompt Warmup via Reinforcement Learning for Small Language Models Reranking 

**Authors**: Xianming Li, Aamir Shakir, Rui Huang, Julius Lipp, Jing Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.03487)  

**Abstract**: Reranking is fundamental to information retrieval and retrieval-augmented generation, with recent Large Language Models (LLMs) significantly advancing reranking quality. While recent advances with LLMs have significantly improved document reranking quality, current approaches primarily rely on large-scale LLMs (>7B parameters) through zero-shot prompting, presenting high computational costs. Small Language Models (SLMs) offer a promising alternative because of their efficiency, but our preliminary quantitative analysis reveals they struggle with understanding task prompts without fine-tuning. This limits their effectiveness for document reranking tasks. To address this issue, we introduce a novel two-stage training approach, ProRank, for SLM-based document reranking. First, we propose a prompt warmup stage using reinforcement learning GRPO to steer SLMs to understand task prompts and generate more accurate coarse-grained binary relevance scores for document reranking. Then, we continuously fine-tune the SLMs with a fine-grained score learning stage without introducing additional layers to further improve the reranking quality. Comprehensive experimental results demonstrate that the proposed ProRank consistently outperforms both the most advanced open-source and proprietary reranking models. Notably, our lightweight ProRank-0.5B model even surpasses the powerful 32B LLM reranking model on the BEIR benchmark, establishing that properly trained SLMs can achieve superior document reranking performance while maintaining computational efficiency. 

---
# Quake: Adaptive Indexing for Vector Search 

**Authors**: Jason Mohoney, Devesh Sarda, Mengze Tang, Shihabur Rahman Chowdhury, Anil Pacaci, Ihab F. Ilyas, Theodoros Rekatsinas, Shivaram Venkataraman  

**Link**: [PDF](https://arxiv.org/pdf/2506.03437)  

**Abstract**: Vector search, the task of finding the k-nearest neighbors of high-dimensional vectors, underpins many machine learning applications, including recommendation systems and information retrieval. However, existing approximate nearest neighbor (ANN) methods perform poorly under dynamic, skewed workloads where data distributions evolve. We introduce Quake, an adaptive indexing system that maintains low latency and high recall in such environments. Quake employs a hierarchical partitioning scheme that adjusts to updates and changing access patterns, guided by a cost model that predicts query latency based on partition sizes and access frequencies. Quake also dynamically optimizes query execution parameters to meet recall targets using a novel recall estimation model. Furthermore, Quake utilizes optimized query processing, leveraging NUMA-aware parallelism for improved memory bandwidth utilization. To evaluate Quake, we prepare a Wikipedia vector search workload and develop a workload generator to create vector search workloads with configurable access patterns. Our evaluation shows that on dynamic workloads, Quake achieves query latency reductions of 1.5-22x and update latency reductions of 6-83x compared to state-of-the-art indexes SVS, DiskANN, HNSW, and SCANN. 

---
# Universal Reusability in Recommender Systems: The Case for Dataset- and Task-Independent Frameworks 

**Authors**: Tri Kurniawan Wijaya, Xinyang Shao, Gonzalo Fiz Pontiveros, Edoardo D'Amico  

**Link**: [PDF](https://arxiv.org/pdf/2506.03391)  

**Abstract**: Recommender systems are pivotal in delivering personalized experiences across industries, yet their adoption and scalability remain hindered by the need for extensive dataset- and task-specific configurations. Existing systems often require significant manual intervention, domain expertise, and engineering effort to adapt to new datasets or tasks, creating barriers to entry and limiting reusability. In contrast, recent advancements in large language models (LLMs) have demonstrated the transformative potential of reusable systems, where a single model can handle diverse tasks without significant reconfiguration. Inspired by this paradigm, we propose the Dataset- and Task-Independent Recommender System (DTIRS), a framework aimed at maximizing the reusability of recommender systems while minimizing barriers to entry. Unlike LLMs, which achieve task generalization directly, DTIRS focuses on eliminating the need to rebuild or reconfigure recommendation pipelines for every new dataset or task, even though models may still need retraining on new data. By leveraging the novel Dataset Description Language (DsDL), DTIRS enables standardized dataset descriptions and explicit task definitions, allowing autonomous feature engineering, model selection, and optimization. This paper introduces the concept of DTIRS and establishes a roadmap for transitioning from Level-1 automation (dataset-agnostic but task-specific systems) to Level-2 automation (fully dataset- and task-independent systems). Achieving this paradigm would maximize code reusability and lower barriers to adoption. We discuss key challenges, including the trade-offs between generalization and specialization, computational overhead, and scalability, while presenting DsDL as a foundational tool for this vision. 

---
# Multi-objective Aligned Bidword Generation Model for E-commerce Search Advertising 

**Authors**: Zhenhui Liu, Chunyuan Yuan, Ming Pang, Zheng Fang, Li Yuan, Xue Jiang, Changping Peng, Zhangang Lin, Zheng Luo, Jingping Shao  

**Link**: [PDF](https://arxiv.org/pdf/2506.03827)  

**Abstract**: Retrieval systems primarily address the challenge of matching user queries with the most relevant advertisements, playing a crucial role in e-commerce search advertising. The diversity of user needs and expressions often produces massive long-tail queries that cannot be matched with merchant bidwords or product titles, which results in some advertisements not being recalled, ultimately harming user experience and search efficiency. Existing query rewriting research focuses on various methods such as query log mining, query-bidword vector matching, or generation-based rewriting. However, these methods often fail to simultaneously optimize the relevance and authenticity of the user's original query and rewrite and maximize the revenue potential of recalled ads.
In this paper, we propose a Multi-objective aligned Bidword Generation Model (MoBGM), which is composed of a discriminator, generator, and preference alignment module, to address these challenges. To simultaneously improve the relevance and authenticity of the query and rewrite and maximize the platform revenue, we design a discriminator to optimize these key objectives. Using the feedback signal of the discriminator, we train a multi-objective aligned bidword generator that aims to maximize the combined effect of the three objectives. Extensive offline and online experiments show that our proposed algorithm significantly outperforms the state of the art. After deployment, the algorithm has created huge commercial value for the platform, further verifying its feasibility and robustness. 

---
# CRAWLDoc: A Dataset for Robust Ranking of Bibliographic Documents 

**Authors**: Fabian Karl, Ansgar Scherp  

**Link**: [PDF](https://arxiv.org/pdf/2506.03822)  

**Abstract**: Publication databases rely on accurate metadata extraction from diverse web sources, yet variations in web layouts and data formats present challenges for metadata providers. This paper introduces CRAWLDoc, a new method for contextual ranking of linked web documents. Starting with a publication's URL, such as a digital object identifier, CRAWLDoc retrieves the landing page and all linked web resources, including PDFs, ORCID profiles, and supplementary materials. It embeds these resources, along with anchor texts and the URLs, into a unified representation. For evaluating CRAWLDoc, we have created a new, manually labeled dataset of 600 publications from six top publishers in computer science. Our method CRAWLDoc demonstrates a robust and layout-independent ranking of relevant documents across publishers and data formats. It lays the foundation for improved metadata extraction from web documents with various layouts and formats. Our source code and dataset can be accessed at this https URL. 

---
# Understanding Mental Models of Generative Conversational Search and The Effect of Interface Transparency 

**Authors**: Chadha Degachi, Samuel Kernan Freire, Evangelos Niforatos, Gerd Kortuem  

**Link**: [PDF](https://arxiv.org/pdf/2506.03807)  

**Abstract**: The experience and adoption of conversational search is tied to the accuracy and completeness of users' mental models -- their internal frameworks for understanding and predicting system behaviour. Thus, understanding these models can reveal areas for design interventions. Transparency is one such intervention which can improve system interpretability and enable mental model alignment. While past research has explored mental models of search engines, those of generative conversational search remain underexplored, even while the popularity of these systems soars. To address this, we conducted a study with 16 participants, who performed 4 search tasks using 4 conversational interfaces of varying transparency levels. Our analysis revealed that most user mental models were too abstract to support users in explaining individual search instances. These results suggest that 1) mental models may pose a barrier to appropriate trust in conversational search, and 2) hybrid web-conversational search is a promising novel direction for future search interface design. 

---
# DistRAG: Towards Distance-Based Spatial Reasoning in LLMs 

**Authors**: Nicole R Schneider, Nandini Ramachandran, Kent O'Sullivan, Hanan Samet  

**Link**: [PDF](https://arxiv.org/pdf/2506.03424)  

**Abstract**: Many real world tasks where Large Language Models (LLMs) can be used require spatial reasoning, like Point of Interest (POI) recommendation and itinerary planning. However, on their own LLMs lack reliable spatial reasoning capabilities, especially about distances. To address this problem, we develop a novel approach, DistRAG, that enables an LLM to retrieve relevant spatial information not explicitly learned during training. Our method encodes the geodesic distances between cities and towns in a graph and retrieves a context subgraph relevant to the question. Using this technique, our method enables an LLM to answer distance-based reasoning questions that it otherwise cannot answer. Given the vast array of possible places an LLM could be asked about, DistRAG offers a flexible first step towards providing a rudimentary `world model' to complement the linguistic knowledge held in LLMs. 

---
# Impact of Rankings and Personalized Recommendations in Marketplaces 

**Authors**: Omar Besbes, Yash Kanoria, Akshit Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2506.03369)  

**Abstract**: Individuals often navigate several options with incomplete knowledge of their own preferences. Information provisioning tools such as public rankings and personalized recommendations have become central to helping individuals make choices, yet their value proposition under different marketplace environments remains unexplored. This paper studies a stylized model to explore the impact of these tools in two marketplace settings: uncapacitated supply, where items can be selected by any number of agents, and capacitated supply, where each item is constrained to be matched to a single agent. We model the agents utility as a weighted combination of a common term which depends only on the item, reflecting the item's population level quality, and an idiosyncratic term, which depends on the agent item pair capturing individual specific tastes. Public rankings reveal the common term, while personalized recommendations reveal both terms. In the supply unconstrained settings, both public rankings and personalized recommendations improve welfare, with their relative value determined by the degree of preference heterogeneity. Public rankings are effective when preferences are relatively homogeneous, while personalized recommendations become critical as heterogeneity increases. In contrast, in supply constrained settings, revealing just the common term, as done by public rankings, provides limited benefit since the total common value available is limited by capacity constraints, whereas personalized recommendations, by revealing both common and idiosyncratic terms, significantly enhance welfare by enabling agents to match with items they idiosyncratically value highly. These results illustrate the interplay between supply constraints and preference heterogeneity in determining the effectiveness of information provisioning tools, offering insights for their design and deployment in diverse settings. 

---
