# CRISP: Clustering Multi-Vector Representations for Denoising and Pruning 

**Authors**: João Veneroso, Rajesh Jayaram, Jinmeng Rao, Gustavo Hernández Ábrego, Majid Hadian, Daniel Cer  

**Link**: [PDF](https://arxiv.org/pdf/2505.11471)  

**Abstract**: Multi-vector models, such as ColBERT, are a significant advancement in neural information retrieval (IR), delivering state-of-the-art performance by representing queries and documents by multiple contextualized token-level embeddings. However, this increased representation size introduces considerable storage and computational overheads which have hindered widespread adoption in practice. A common approach to mitigate this overhead is to cluster the model's frozen vectors, but this strategy's effectiveness is fundamentally limited by the intrinsic clusterability of these embeddings. In this work, we introduce CRISP (Clustered Representations with Intrinsic Structure Pruning), a novel multi-vector training method which learns inherently clusterable representations directly within the end-to-end training process. By integrating clustering into the training phase rather than imposing it post-hoc, CRISP significantly outperforms post-hoc clustering at all representation sizes, as well as other token pruning methods. On the BEIR retrieval benchmarks, CRISP achieves a significant rate of ~3x reduction in the number of vectors while outperforming the original unpruned model. This indicates that learned clustering effectively denoises the model by filtering irrelevant information, thereby generating more robust multi-vector representations. With more aggressive clustering, CRISP achieves an 11x reduction in the number of vectors with only a $3.6\%$ quality loss. 

---
# The Future is Sparse: Embedding Compression for Scalable Retrieval in Recommender Systems 

**Authors**: Petr Kasalický, Martin Spišák, Vojtěch Vančura, Daniel Bohuněk, Rodrigo Alves, Pavel Kordík  

**Link**: [PDF](https://arxiv.org/pdf/2505.11388)  

**Abstract**: Industry-scale recommender systems face a core challenge: representing entities with high cardinality, such as users or items, using dense embeddings that must be accessible during both training and inference. However, as embedding sizes grow, memory constraints make storage and access increasingly difficult. We describe a lightweight, learnable embedding compression technique that projects dense embeddings into a high-dimensional, sparsely activated space. Designed for retrieval tasks, our method reduces memory requirements while preserving retrieval performance, enabling scalable deployment under strict resource constraints. Our results demonstrate that leveraging sparsity is a promising approach for improving the efficiency of large-scale recommenders. We release our code at this https URL. 

---
# On the Role of Weight Decay in Collaborative Filtering: A Popularity Perspective 

**Authors**: Donald Loveland, Mingxuan Ju, Tong Zhao, Neil Shah, Danai Koutra  

**Link**: [PDF](https://arxiv.org/pdf/2505.11318)  

**Abstract**: Collaborative filtering (CF) enables large-scale recommendation systems by encoding information from historical user-item interactions into dense ID-embedding tables. However, as embedding tables grow, closed-form solutions become impractical, often necessitating the use of mini-batch gradient descent for training. Despite extensive work on designing loss functions to train CF models, we argue that one core component of these pipelines is heavily overlooked: weight decay. Attaining high-performing models typically requires careful tuning of weight decay, regardless of loss, yet its necessity is not well understood. In this work, we question why weight decay is crucial in CF pipelines and how it impacts training. Through theoretical and empirical analysis, we surprisingly uncover that weight decay's primary function is to encode popularity information into the magnitudes of the embedding vectors. Moreover, we find that tuning weight decay acts as a coarse, non-linear knob to influence preference towards popular or unpopular items. Based on these findings, we propose PRISM (Popularity-awaRe Initialization Strategy for embedding Magnitudes), a straightforward yet effective solution to simplify the training of high-performing CF models. PRISM pre-encodes the popularity information typically learned through weight decay, eliminating its necessity. Our experiments show that PRISM improves performance by up to 4.77% and reduces training times by 38.48%, compared to state-of-the-art training strategies. Additionally, we parameterize PRISM to modulate the initialization strength, offering a cost-effective and meaningful strategy to mitigate popularity bias. 

---
# User-centric Music Recommendations 

**Authors**: Jaime Ramirez Castillo, M. Julia Flores, Ann E. Nicholson  

**Link**: [PDF](https://arxiv.org/pdf/2505.11198)  

**Abstract**: This work presents a user-centric recommendation framework, designed as a pipeline with four distinct, connected, and customizable phases. These phases are intended to improve explainability and boost user engagement.
We have collected the historical this http URL track playback records of a single user over approximately 15 years. The collected dataset includes more than 90,000 playbacks and approximately 14,000 unique tracks.
From track playback records, we have created a dataset of user temporal contexts (each row is a specific moment when the user listened to certain music descriptors). As music descriptors, we have used community-contributed this http URL tags and Spotify audio features. They represent the music that, throughout years, the user has been listening to.
Next, given the most relevant this http URL tags of a moment (e.g. the hour of the day), we predict the Spotify audio features that best fit the user preferences in that particular moment. Finally, we use the predicted audio features to find tracks similar to these features. The final aim is to recommend (and discover) tracks that the user may feel like listening to at a particular moment.
For our initial study case, we have chosen to predict only a single audio feature target: danceability. The framework, however, allows to include more target variables.
The ability to learn the musical habits from a single user can be quite powerful, and this framework could be extended to other users. 

---
# mmRAG: A Modular Benchmark for Retrieval-Augmented Generation over Text, Tables, and Knowledge Graphs 

**Authors**: Chuan Xu, Qiaosheng Chen, Yutong Feng, Gong Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.11180)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a powerful paradigm for enhancing the capabilities of large language models. However, existing RAG evaluation predominantly focuses on text retrieval and relies on opaque, end-to-end assessments of generated outputs. To address these limitations, we introduce mmRAG, a modular benchmark designed for evaluating multi-modal RAG systems. Our benchmark integrates queries from six diverse question-answering datasets spanning text, tables, and knowledge graphs, which we uniformly convert into retrievable documents. To enable direct, granular evaluation of individual RAG components -- such as the accuracy of retrieval and query routing -- beyond end-to-end generation quality, we follow standard information retrieval procedures to annotate document relevance and derive dataset relevance. We establish baseline performance by evaluating a wide range of RAG implementations on mmRAG. 

---
# From Intent Discovery to Recognition with Topic Modeling and Synthetic Data 

**Authors**: Aaron Rodrigues, Mahmood Hegazy, Azzam Naeem  

**Link**: [PDF](https://arxiv.org/pdf/2505.11176)  

**Abstract**: Understanding and recognizing customer intents in AI systems is crucial, particularly in domains characterized by short utterances and the cold start problem, where recommender systems must include new products or services without sufficient real user data. Customer utterances are characterized by infrequent word co-occurences and high term variability, which poses significant challenges for traditional methods in specifying distinct user needs and preparing synthetic queries. To address this, we propose an agentic LLM framework for topic modeling and synthetic query generation, which accelerates the discovery and recognition of customer intents. We first apply hierarchical topic modeling and intent discovery to expand a human-curated taxonomy from 36 generic user intents to 278 granular intents, demonstrating the potential of LLMs to significantly enhance topic specificity and diversity. Next, to support newly discovered intents and address the cold start problem, we generate synthetic user query data, which augments real utterances and reduces dependency on human annotation, especially in low-resource settings. Topic model experiments show substantial improvements in coherence and relevance after topic expansion, while synthetic data experiments indicate that in-class few-shot prompting significantly improves the quality and utility of synthetic queries without compromising diversity. We also show that LLM-generated intent descriptions and keywords can effectively substitute for human-curated versions when used as context for synthetic query generation. Our research underscores the scalability and utility of LLM agents in topic modeling and highlights the strategic use of synthetic utterances to enhance dataset variability and coverage for intent recognition. We present a comprehensive and robust framework for online discovery and recognition of new customer intents in dynamic domains. 

---
# Who You Are Matters: Bridging Topics and Social Roles via LLM-Enhanced Logical Recommendation 

**Authors**: Qing Yu, Xiaobei Wang, Shuchang Liu, Yandong Bai, Xiaoyu Yang, Xueliang Wang, Chang Meng, Shanshan Wu, Hailan Yang, Huihui Xiao, Xiang Li, Fan Yang, Xiaoqiang Feng, Lantao Hu, Han Li, Kun Gai, Lixin Zou  

**Link**: [PDF](https://arxiv.org/pdf/2505.10940)  

**Abstract**: Recommender systems filter contents/items valuable to users by inferring preferences from user features and historical behaviors. Mainstream approaches follow the learning-to-rank paradigm, which focus on discovering and modeling item topics (e.g., categories), and capturing user preferences on these topics based on historical interactions. However, this paradigm often neglects the modeling of user characteristics and their social roles, which are logical confounders influencing the correlated interest and user preference transition. To bridge this gap, we introduce the user role identification task and the behavioral logic modeling task that aim to explicitly model user roles and learn the logical relations between item topics and user social roles. We show that it is possible to explicitly solve these tasks through an efficient integration framework of Large Language Model (LLM) and recommendation systems, for which we propose TagCF. On the one hand, the exploitation of the LLM's world knowledge and logic inference ability produces a virtual logic graph that reveals dynamic and expressive knowledge of users, augmenting the recommendation performance. On the other hand, the user role aligns the user behavioral logic with the observed user feedback, refining our understanding of user behaviors. Additionally, we also show that the extracted user-item logic graph is empirically a general knowledge that can benefit a wide range of recommendation tasks, and conduct experiments on industrial and several public datasets as verification. 

---
# Explain What You Mean: Intent Augmented Knowledge Graph Recommender Built With LLM 

**Authors**: Wenqing Zheng, Noah Fatsi, Daniel Barcklow, Dmitri Kalaev, Steven Yao, Owen Reinert, C. Bayan Bruss, Daniele Rosa  

**Link**: [PDF](https://arxiv.org/pdf/2505.10900)  

**Abstract**: Interaction sparsity is the primary obstacle for recommendation systems. Sparsity manifests in environments with disproportional cardinality of groupings of entities, such as users and products in an online marketplace. It also is found for newly introduced entities, described as the cold-start problem. Recent efforts to mitigate this sparsity issue shifts the performance bottleneck to other areas in the computational pipeline. Those that focus on enriching sparse representations with connectivity data from other external sources propose methods that are resource demanding and require careful domain expert aided addition of this newly introduced data. Others that turn to Large Language Model (LLM) based recommenders will quickly encounter limitations surrounding data quality and availability. In this work, we propose LLM-based Intent Knowledge Graph Recommender (IKGR), a novel framework that leverages retrieval-augmented generation and an encoding approach to construct and densify a knowledge graph. IKGR learns latent user-item affinities from an interaction knowledge graph and further densifies it through mutual intent connectivity. This addresses sparsity issues and allows the model to make intent-grounded recommendations with an interpretable embedding translation layer. Through extensive experiments on real-world datasets, we demonstrate that IKGR overcomes knowledge gaps and achieves substantial gains over state-of-the-art baselines on both publicly available and our internal recommendation datasets. 

---
# Semantic Caching of Contextual Summaries for Efficient Question-Answering with Language Models 

**Authors**: Camille Couturier, Spyros Mastorakis, Haiying Shen, Saravan Rajmohan, Victor Rühle  

**Link**: [PDF](https://arxiv.org/pdf/2505.11271)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed across edge and cloud platforms for real-time question-answering and retrieval-augmented generation. However, processing lengthy contexts in distributed systems incurs high computational overhead, memory usage, and network bandwidth. This paper introduces a novel semantic caching approach for storing and reusing intermediate contextual summaries, enabling efficient information reuse across similar queries in LLM-based QA workflows. Our method reduces redundant computations by up to 50-60% while maintaining answer accuracy comparable to full document processing, as demonstrated on NaturalQuestions, TriviaQA, and a synthetic ArXiv dataset. This approach balances computational cost and response quality, critical for real-time AI assistants. 

---
# Improve Rule Retrieval and Reasoning with Self-Induction and Relevance ReEstimate 

**Authors**: Ziyang Huang, Wangtao Sun, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.10870)  

**Abstract**: This paper systematically addresses the challenges of rule retrieval, a crucial yet underexplored area. Vanilla retrieval methods using sparse or dense retrievers to directly search for relevant rules to support downstream reasoning, often suffer from low accuracy. This is primarily due to a significant semantic gap between the instantiated facts in the queries and the abstract representations of the rules. Such misalignment results in suboptimal retrieval quality, which in turn negatively impacts reasoning performance. To overcome these challenges, we propose Self-Induction Augmented Retrieval (SIAR), a novel approach that utilizes Large Language Models (LLMs) to induce potential inferential rules that might offer benefits for reasoning by abstracting the underlying knowledge and logical structure in queries. These induced rules are then used for query augmentation to improve retrieval effectiveness. Additionally, we introduce Rule Relevance ReEstimate (R$^3$), a method that re-estimates the relevance of retrieved rules by assessing whether the abstract knowledge they contain can be instantiated to align with the facts in the queries and the helpfulness for reasoning. Extensive experiments across various settings demonstrate the effectiveness and versatility of our proposed methods. 

---
# SemEval-2025 Task 7: Multilingual and Crosslingual Fact-Checked Claim Retrieval 

**Authors**: Qiwei Peng, Robert Moro, Michal Gregor, Ivan Srba, Simon Ostermann, Marian Simko, Juraj Podroužek, Matúš Mesarčík, Jaroslav Kopčan, Anders Søgaard  

**Link**: [PDF](https://arxiv.org/pdf/2505.10740)  

**Abstract**: The rapid spread of online disinformation presents a global challenge, and machine learning has been widely explored as a potential solution. However, multilingual settings and low-resource languages are often neglected in this field. To address this gap, we conducted a shared task on multilingual claim retrieval at SemEval 2025, aimed at identifying fact-checked claims that match newly encountered claims expressed in social media posts across different languages. The task includes two subtracks: (1) a monolingual track, where social posts and claims are in the same language, and (2) a crosslingual track, where social posts and claims might be in different languages. A total of 179 participants registered for the task contributing to 52 test submissions. 23 out of 31 teams have submitted their system papers. In this paper, we report the best-performing systems as well as the most common and the most effective approaches across both subtracks. This shared task, along with its dataset and participating systems, provides valuable insights into multilingual claim retrieval and automated fact-checking, supporting future research in this field. 

---
