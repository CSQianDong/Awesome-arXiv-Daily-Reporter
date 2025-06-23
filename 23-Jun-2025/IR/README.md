# RAGentA: Multi-Agent Retrieval-Augmented Generation for Attributed Question Answering 

**Authors**: Ines Besrour, Jingbo He, Tobias Schreieder, Michael Färber  

**Link**: [PDF](https://arxiv.org/pdf/2506.16988)  

**Abstract**: We present RAGentA, a multi-agent retrieval-augmented generation (RAG) framework for attributed question answering (QA). With the goal of trustworthy answer generation, RAGentA focuses on optimizing answer correctness, defined by coverage and relevance to the question and faithfulness, which measures the extent to which answers are grounded in retrieved documents. RAGentA uses a multi-agent architecture that iteratively filters retrieved documents, generates attributed answers with in-line citations, and verifies completeness through dynamic refinement. Central to the framework is a hybrid retrieval strategy that combines sparse and dense methods, improving Recall@20 by 12.5% compared to the best single retrieval model, resulting in more correct and well-supported answers. Evaluated on a synthetic QA dataset derived from the FineWeb index, RAGentA outperforms standard RAG baselines, achieving gains of 1.09% in correctness and 10.72% in faithfulness. These results demonstrate the effectiveness of the multi-agent architecture and hybrid retrieval in advancing trustworthy QA. 

---
# Pyramid Mixer: Multi-dimensional Multi-period Interest Modeling for Sequential Recommendation 

**Authors**: Zhen Gong, Zhifang Fan, Hui Lu, Qiwei Chen, Chenbin Zhang, Lin Guan, Yuchao Zheng, Feng Zhang, Xiao Yang, Zuotao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.16942)  

**Abstract**: Sequential recommendation, a critical task in recommendation systems, predicts the next user action based on the understanding of the user's historical behaviors. Conventional studies mainly focus on cross-behavior modeling with self-attention based methods while neglecting comprehensive user interest modeling for more dimensions. In this study, we propose a novel sequential recommendation model, Pyramid Mixer, which leverages the MLP-Mixer architecture to achieve efficient and complete modeling of user interests. Our method learns comprehensive user interests via cross-behavior and cross-feature user sequence modeling. The mixer layers are stacked in a pyramid way for cross-period user temporal interest learning. Through extensive offline and online experiments, we demonstrate the effectiveness and efficiency of our method, and we obtain a +0.106% improvement in user stay duration and a +0.0113% increase in user active days in the online A/B test. The Pyramid Mixer has been successfully deployed on the industrial platform, demonstrating its scalability and impact in real-world applications. 

---
# Multi-Objective Recommendation in the Era of Generative AI: A Survey of Recent Progress and Future Prospects 

**Authors**: Zihan Hong, Yushi Wu, Zhiting Zhao, Shanshan Feng, Jianghong Ma, Jiao Liu, Tianjun Wei  

**Link**: [PDF](https://arxiv.org/pdf/2506.16893)  

**Abstract**: With the recent progress in generative artificial intelligence (Generative AI), particularly in the development of large language models, recommendation systems are evolving to become more versatile. Unlike traditional techniques, generative AI not only learns patterns and representations from complex data but also enables content generation, data synthesis, and personalized experiences. This generative capability plays a crucial role in the field of recommendation systems, helping to address the issue of data sparsity and improving the overall performance of recommendation systems. Numerous studies on generative AI have already emerged in the field of recommendation systems. Meanwhile, the current requirements for recommendation systems have surpassed the single utility of accuracy, leading to a proliferation of multi-objective research that considers various goals in recommendation systems. However, to the best of our knowledge, there remains a lack of comprehensive studies on multi-objective recommendation systems based on generative AI technologies, leaving a significant gap in the literature. Therefore, we investigate the existing research on multi-objective recommendation systems involving generative AI to bridge this gap. We compile current research on multi-objective recommendation systems based on generative techniques, categorizing them by objectives. Additionally, we summarize relevant evaluation metrics and commonly used datasets, concluding with an analysis of the challenges and future directions in this domain. 

---
# eSapiens: A Real-World NLP Framework for Multimodal Document Understanding and Enterprise Knowledge Processing 

**Authors**: Isaac Shi, Zeyuan Li, Wenli Wang, Lewei He, Yang Yang, Tianyu Shi  

**Link**: [PDF](https://arxiv.org/pdf/2506.16768)  

**Abstract**: We introduce eSapiens, a unified question-answering system designed for enterprise settings, which bridges structured databases and unstructured textual corpora via a dual-module architecture. The system combines a Text-to-SQL planner with a hybrid Retrieval-Augmented Generation (RAG) pipeline, enabling natural language access to both relational data and free-form documents. To enhance answer faithfulness, the RAG module integrates dense and sparse retrieval, commercial reranking, and a citation verification loop that ensures grounding consistency. We evaluate eSapiens on the RAGTruth benchmark across five leading large language models (LLMs), analyzing performance across key dimensions such as completeness, hallucination, and context utilization. Results demonstrate that eSapiens outperforms a FAISS baseline in contextual relevance and generation quality, with optional strict-grounding controls for high-stakes scenarios. This work provides a deployable framework for robust, citation-aware question answering in real-world enterprise applications. 

---
# A Simple Contrastive Framework Of Item Tokenization For Generative Recommendation 

**Authors**: Penglong Zhai, Yifang Yuan, Fanyi Di, Jie Li, Yue Liu, Chen Li, Jie Huang, Sicong Wang, Yao Xu, Xin Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.16683)  

**Abstract**: Generative retrieval-based recommendation has emerged as a promising paradigm aiming at directly generating the identifiers of the target candidates. However, in large-scale recommendation systems, this approach becomes increasingly cumbersome due to the redundancy and sheer scale of the token space. To overcome these limitations, recent research has explored the use of semantic tokens as an alternative to ID tokens, which typically leveraged reconstruction-based strategies, like RQ-VAE, to quantize content embeddings and significantly reduce the embedding size. However, reconstructive quantization aims for the precise reconstruction of each item embedding independently, which conflicts with the goal of generative retrieval tasks focusing more on differentiating among items. Moreover, multi-modal side information of items, such as descriptive text and images, geographical knowledge in location-based recommendation services, has been shown to be effective in improving recommendations by providing richer contexts for interactions. Nevertheless, effectively integrating such complementary knowledge into existing generative recommendation frameworks remains challenging. To overcome these challenges, we propose a novel unsupervised deep quantization exclusively based on contrastive learning, named SimCIT (a Simple Contrastive Item Tokenization framework). Specifically, different from existing reconstruction-based strategies, SimCIT propose to use a learnable residual quantization module to align with the signals from different modalities of the items, which combines multi-modal knowledge alignment and semantic tokenization in a mutually beneficial contrastive learning framework. Extensive experiments across public datasets and a large-scale industrial dataset from various domains demonstrate SimCIT's effectiveness in LLM-based generative recommendation. 

---
# Revela: Dense Retriever Learning via Language Modeling 

**Authors**: Fengyu Cai, Tong Chen, Xinran Zhao, Sihao Chen, Hongming Zhang, Sherry Tongshuang Wu, Iryna Gurevych, Heinz Koeppl  

**Link**: [PDF](https://arxiv.org/pdf/2506.16552)  

**Abstract**: Dense retrievers play a vital role in accessing external and specialized knowledge to augment language models (LMs). Training dense retrievers typically requires annotated query-document pairs, which are costly and hard to obtain in specialized domains such as code-motivating growing interest in self-supervised retriever learning. Since LMs are trained to capture token-level dependencies through a self-supervised learning objective (i.e., next-token prediction), we can analogously cast retrieval as learning dependencies among chunks of tokens. This analogy naturally leads to the question: How can we adapt self-supervised learning objectives in the spirit of language modeling to train retrievers?
To answer this question, we introduce Revela, a unified and scalable training framework for self-supervised retriever learning via language modeling. Revela models semantic dependencies among documents by conditioning next-token prediction on both local and cross-document context through an in-batch attention mechanism. This attention is weighted by retriever-computed similarity scores, enabling the retriever to be optimized as part of language modeling. We evaluate Revela on both general-domain (BEIR) and domain-specific (CoIR) benchmarks across various retriever backbones. At a comparable parameter scale, Revela outperforms the previous best method with absolute improvements of 5.2 % (18.3 % relative) and 5.6 % (14.4 % relative) on NDCG@10, respectively, underscoring its effectiveness. Performance increases with model size, highlighting both the scalability of our approach and its promise for self-supervised retriever learning. 

---
# Neural Prioritisation for Web Crawling 

**Authors**: Francesza Pezzuti, Sean MacAvaney, Nicola Tonellotto  

**Link**: [PDF](https://arxiv.org/pdf/2506.16146)  

**Abstract**: Given the vast scale of the Web, crawling prioritisation techniques based on link graph traversal, popularity, link analysis, and textual content are frequently applied to surface documents that are most likely to be valuable. While existing techniques are effective for keyword-based search, both retrieval methods and user search behaviours are shifting from keyword-based matching to natural language semantic matching. The remarkable success of applying semantic matching and quality signals during ranking leads us to hypothesize that crawling could be improved by prioritizing Web pages with high semantic quality. To investigate this, we propose a semantic quality-driven prioritisation technique to enhance the effectiveness of crawling and align the crawler behaviour with recent shift towards natural language search. We embed semantic understanding directly into the crawling process -- leveraging recent neural semantic quality estimators to prioritise the crawling frontier -- with the goal of surfacing content that is semantically rich and valuable for modern search needs. Our experiments on the English subset of ClueWeb22-B and the Researchy Questions query set show that, compared to existing crawling techniques, neural crawling policies significantly improve harvest rate, maxNDCG, and search effectiveness during the early stages of crawling. Meanwhile, crawlers based on our proposed neural policies maintain comparable search performance on keyword queries from the MS MARCO Web Search query set. While this work does not propose a definitive and complete solution, it presents a forward-looking perspective on Web crawling and opens the door to a new line of research on leveraging semantic analysis to effectively align crawlers with the ongoing shift toward natural language search. 

---
# GFlowGR: Fine-tuning Generative Recommendation Frameworks with Generative Flow Networks 

**Authors**: Yejing Wang, Shengyu Zhou, Jinyu Lu, Qidong Liu, Xinhang Li, Wenlin Zhang, Feng Li, Pengjie Wang, Jian Xu, Bo Zheng, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.16114)  

**Abstract**: Generative recommendations (GR), which usually include item tokenizers and generative Large Language Models (LLMs), have demonstrated remarkable success across a wide range of scenarios. The majority of existing research efforts primarily concentrate on developing powerful item tokenizers or advancing LLM decoding strategies to attain superior performance. However, the critical fine-tuning step in GR frameworks, which is essential for adapting LLMs to recommendation data, remains largely unexplored. Current approaches predominantly rely on either the next-token prediction loss of supervised fine-tuning (SFT) or recommendationspecific direct preference optimization (DPO) strategies. Both methods ignore the exploration of possible positive unobserved samples, which is commonly referred to as the exposure bias problem. To mitigate this problem, this paper treats the GR as a multi-step generation task and constructs a GFlowNets-based fine-tuning framework (GFlowGR). The proposed framework integrates collaborative knowledge from traditional recommender systems to create an adaptive trajectory sampler and a comprehensive reward model. Leveraging the diverse generation property of GFlowNets, along with sampling and heuristic weighting techniques, GFlowGR emerges as a promising approach to mitigate the exposure bias problem. Extensive empirical results on two real-world datasets and with two different GR backbones highlight the effectiveness and robustness of GFlowGR. 

---
# SEP-GCN: Leveraging Similar Edge Pairs with Temporal and Spatial Contexts for Location-Based Recommender Systems 

**Authors**: Tan Loc Nguyen, Tin T. Tran  

**Link**: [PDF](https://arxiv.org/pdf/2506.16003)  

**Abstract**: Recommender systems play a crucial role in enabling personalized content delivery amidst the challenges of information overload and human mobility. Although conventional methods often rely on interaction matrices or graph-based retrieval, recent approaches have sought to exploit contextual signals such as time and location. However, most existing models focus on node-level representation or isolated edge attributes, underutilizing the relational structure between interactions. We propose SEP-GCN, a novel graph-based recommendation framework that learns from pairs of contextually similar interaction edges, each representing a user-item check-in event. By identifying edge pairs that occur within similar temporal windows or geographic proximity, SEP-GCN augments the user-item graph with contextual similarity links. These links bridge distant but semantically related interactions, enabling improved long-range information propagation. The enriched graph is processed via an edge-aware convolutional mechanism that integrates contextual similarity into the message-passing process. This allows SEP-GCN to model user preferences more accurately and robustly, especially in sparse or dynamic environments. Experiments on benchmark data sets show that SEP-GCN consistently outperforms strong baselines in both predictive accuracy and robustness. 

---
# MoR: Better Handling Diverse Queries with a Mixture of Sparse, Dense, and Human Retrievers 

**Authors**: Jushaan Singh Kalra, Xinran Zhao, To Eun Kim, Fengyu Cai, Fernando Diaz, Tongshuang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15862)  

**Abstract**: Retrieval-augmented Generation (RAG) is powerful, but its effectiveness hinges on which retrievers we use and how. Different retrievers offer distinct, often complementary signals: BM25 captures lexical matches; dense retrievers, semantic similarity. Yet in practice, we typically fix a single retriever based on heuristics, which fails to generalize across diverse information needs. Can we dynamically select and integrate multiple retrievers for each individual query, without the need for manual selection? In our work, we validate this intuition with quantitative analysis and introduce mixture of retrievers: a zero-shot, weighted combination of heterogeneous retrievers. Extensive experiments show that such mixtures are effective and efficient: Despite totaling just 0.8B parameters, this mixture outperforms every individual retriever and even larger 7B models by +10.8% and +3.9% on average, respectively. Further analysis also shows that this mixture framework can help incorporate specialized non-oracle human information sources as retrievers to achieve good collaboration, with a 58.9% relative performance improvement over simulated humans alone. 

---
# Architecture is All You Need: Improving LLM Recommenders by Dropping the Text 

**Authors**: Kevin Foley, Shaghayegh Agah, Kavya Priyanka Kakinada  

**Link**: [PDF](https://arxiv.org/pdf/2506.15833)  

**Abstract**: In recent years, there has been an explosion of interest in the applications of large pre-trained language models (PLMs) to recommender systems, with many studies showing strong performance of PLMs on common benchmark datasets. PLM-based recommender models benefit from flexible and customizable prompting, an unlimited vocabulary of recommendable items, and general ``world knowledge'' acquired through pre-training on massive text corpora. While PLM-based recommenders show promise in settings where data is limited, they are hard to implement in practice due to their large size and computational cost. Additionally, fine-tuning PLMs to improve performance on collaborative signals may degrade the model's capacity for world knowledge and generalizability. We propose a recommender model that uses the architecture of large language models (LLMs) while reducing layer count and dimensions and replacing the text-based subword tokenization of a typical LLM with discrete tokens that uniquely represent individual content items. We find that this simplified approach substantially outperforms both traditional sequential recommender models and PLM-based recommender models at a tiny fraction of the size and computational complexity of PLM-based models. Our results suggest that the principal benefit of LLMs in recommender systems is their architecture, rather than the world knowledge acquired during extensive pre-training. 

---
# Towards AI Search Paradigm 

**Authors**: Yuchen Li, Hengyi Cai, Rui Kong, Xinran Chen, Jiamin Chen, Jun Yang, Haojie Zhang, Jiayi Li, Jiayi Wu, Yiqun Chen, Changle Qu, Keyi Kong, Wenwen Ye, Lixin Su, Xinyu Ma, Long Xia, Daiting Shi, Jiashu Zhao, Haoyi Xiong, Shuaiqiang Wang, Dawei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2506.17188)  

**Abstract**: In this paper, we introduce the AI Search Paradigm, a comprehensive blueprint for next-generation search systems capable of emulating human information processing and decision-making. The paradigm employs a modular architecture of four LLM-powered agents (Master, Planner, Executor and Writer) that dynamically adapt to the full spectrum of information needs, from simple factual queries to complex multi-stage reasoning tasks. These agents collaborate dynamically through coordinated workflows to evaluate query complexity, decompose problems into executable plans, and orchestrate tool usage, task execution, and content synthesis. We systematically present key methodologies for realizing this paradigm, including task planning and tool integration, execution strategies, aligned and robust retrieval-augmented generation, and efficient LLM inference, spanning both algorithmic techniques and infrastructure-level optimizations. By providing an in-depth guide to these foundational components, this work aims to inform the development of trustworthy, adaptive, and scalable AI search systems. 

---
# Universal Music Representations? Evaluating Foundation Models on World Music Corpora 

**Authors**: Charilaos Papaioannou, Emmanouil Benetos, Alexandros Potamianos  

**Link**: [PDF](https://arxiv.org/pdf/2506.17055)  

**Abstract**: Foundation models have revolutionized music information retrieval, but questions remain about their ability to generalize across diverse musical traditions. This paper presents a comprehensive evaluation of five state-of-the-art audio foundation models across six musical corpora spanning Western popular, Greek, Turkish, and Indian classical traditions. We employ three complementary methodologies to investigate these models' cross-cultural capabilities: probing to assess inherent representations, targeted supervised fine-tuning of 1-2 layers, and multi-label few-shot learning for low-resource scenarios. Our analysis shows varying cross-cultural generalization, with larger models typically outperforming on non-Western music, though results decline for culturally distant traditions. Notably, our approaches achieve state-of-the-art performance on five out of six evaluated datasets, demonstrating the effectiveness of foundation models for world music understanding. We also find that our targeted fine-tuning approach does not consistently outperform probing across all settings, suggesting foundation models already encode substantial musical knowledge. Our evaluation framework and benchmarking results contribute to understanding how far current models are from achieving universal music representations while establishing metrics for future progress. 

---
# PersonalAI: Towards digital twins in the graph form 

**Authors**: Mikhail Menschikov, Dmitry Evseev, Ruslan Kostoev, Ilya Perepechkin, Ilnaz Salimov, Victoria Dochkina, Petr Anokhin, Evgeny Burnaev, Nikita Semenov  

**Link**: [PDF](https://arxiv.org/pdf/2506.17001)  

**Abstract**: The challenge of personalizing language models, specifically the ability to account for a user's history during interactions, is of significant interest. Despite recent advancements in large language models (LLMs) and Retrieval Augmented Generation that have enhanced the factual base of LLMs, the task of retaining extensive personal information and using it to generate personalized responses remains pertinent. To address this, we propose utilizing external memory in the form of knowledge graphs, which are constructed and updated by the LLM itself. We have expanded upon ideas of AriGraph architecture and for the first time introduced a combined graph featuring both standard edges and two types of hyperedges. Experiments conducted on the TriviaQA, HotpotQA and DiaASQ benchmarks indicates that this approach aids in making the process of graph construction and knowledge extraction unified and robust. Furthermore, we augmented the DiaASQ benchmark by incorporating parameters such as time into dialogues and introducing contradictory statements made by the same speaker at different times. Despite these modifications, the performance of the question-answering system remained robust, demonstrating the proposed architecture's ability to maintain and utilize temporal dependencies. 

---
# Semantic Outlier Removal with Embedding Models and LLMs 

**Authors**: Eren Akbiyik, João Almeida, Rik Melis, Ritu Sriram, Viviana Petrescu, Vilhjálmur Vilhjálmsson  

**Link**: [PDF](https://arxiv.org/pdf/2506.16644)  

**Abstract**: Modern text processing pipelines demand robust methods to remove extraneous content while preserving a document's core message. Traditional approaches such as HTML boilerplate extraction or keyword filters often fail in multilingual settings and struggle with context-sensitive nuances, whereas Large Language Models (LLMs) offer improved quality at high computational cost. We introduce SORE (Semantic Outlier Removal), a cost-effective, transparent method that leverages multilingual sentence embeddings and approximate nearest-neighbor search to identify and excise unwanted text segments. By first identifying core content via metadata embedding and then flagging segments that either closely match predefined outlier groups or deviate significantly from the core, SORE achieves near-LLM extraction precision at a fraction of the cost. Experiments on HTML datasets demonstrate that SORE outperforms structural methods and yield high precision in diverse scenarios. Our system is currently deployed in production, processing millions of documents daily across multiple languages while maintaining both efficiency and accuracy. To facilitate reproducibility and further research, we release our implementation and evaluation datasets. 

---
# Agentic Personalisation of Cross-Channel Marketing Experiences 

**Authors**: Sami Abboud, Eleanor Hanna, Olivier Jeunen, Vineesha Raheja, Schaun Wheeler  

**Link**: [PDF](https://arxiv.org/pdf/2506.16429)  

**Abstract**: Consumer applications provide ample opportunities to surface and communicate various forms of content to users. From promotional campaigns for new features or subscriptions, to evergreen nudges for engagement, or personalised recommendations; across e-mails, push notifications, and in-app surfaces. The conventional approach to orchestration for communication relies heavily on labour-intensive manual marketer work, and inhibits effective personalisation of content, timing, frequency, and copy-writing. We formulate this task under a sequential decision-making framework, where we aim to optimise a modular decision-making policy that maximises incremental engagement for any funnel event. Our approach leverages a Difference-in-Differences design for Individual Treatment Effect estimation, and Thompson sampling to balance the explore-exploit trade-off. We present results from a multi-service application, where our methodology has resulted in significant increases to a variety of goal events across several product features, and is currently deployed across 150 million users. 

---
# Analyzing the Influence of Knowledge Graph Information on Relation Extraction 

**Authors**: Cedric Möller, Ricardo Usbeck  

**Link**: [PDF](https://arxiv.org/pdf/2506.16343)  

**Abstract**: We examine the impact of incorporating knowledge graph information on the performance of relation extraction models across a range of datasets. Our hypothesis is that the positions of entities within a knowledge graph provide important insights for relation extraction tasks. We conduct experiments on multiple datasets, each varying in the number of relations, training examples, and underlying knowledge graphs. Our results demonstrate that integrating knowledge graph information significantly enhances performance, especially when dealing with an imbalance in the number of training examples for each relation. We evaluate the contribution of knowledge graph-based features by combining established relation extraction methods with graph-aware Neural Bellman-Ford networks. These features are tested in both supervised and zero-shot settings, demonstrating consistent performance improvements across various datasets. 

---
# Vision-Guided Chunking Is All You Need: Enhancing RAG with Multimodal Document Understanding 

**Authors**: Vishesh Tripathi, Tanmay Odapally, Indraneel Das, Uday Allu, Biddwan Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2506.16035)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems have revolutionized information retrieval and question answering, but traditional text-based chunking methods struggle with complex document structures, multi-page tables, embedded figures, and contextual dependencies across page boundaries. We present a novel multimodal document chunking approach that leverages Large Multimodal Models (LMMs) to process PDF documents in batches while maintaining semantic coherence and structural integrity. Our method processes documents in configurable page batches with cross-batch context preservation, enabling accurate handling of tables spanning multiple pages, embedded visual elements, and procedural content. We evaluate our approach on a curated dataset of PDF documents with manually crafted queries, demonstrating improvements in chunk quality and downstream RAG performance. Our vision-guided approach achieves better accuracy compared to traditional vanilla RAG systems, with qualitative analysis showing superior preservation of document structure and semantic coherence. 

---
# Empowering Graph-based Approximate Nearest Neighbor Search with Adaptive Awareness Capabilities 

**Authors**: Jiancheng Ruan, Tingyang Chen, Renchi Yang, Xiangyu Ke, Yunjun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.15986)  

**Abstract**: Approximate Nearest Neighbor Search (ANNS) in high-dimensional spaces finds extensive applications in databases, information retrieval, recommender systems, etc. While graph-based methods have emerged as the leading solution for ANNS due to their superior query performance, they still face several challenges, such as struggling with local optima and redundant computations. These issues arise because existing methods (i) fail to fully exploit the topological information underlying the proximity graph G, and (ii) suffer from severe distribution mismatches between the base data and queries in practice.
To this end, this paper proposes GATE, high-tier proximity Graph with Adaptive Topology and Query AwarEness, as a lightweight and adaptive module atop the graph-based indexes to accelerate ANNS. Specifically, GATE formulates the critical problem to identify an optimal entry point in the proximity graph for a given query, facilitating faster online search. By leveraging the inherent clusterability of high-dimensional data, GATE first extracts a small set of hub nodes V as candidate entry points. Then, resorting to a contrastive learning-based two-tower model, GATE encodes both the structural semantics underlying G and the query-relevant features into the latent representations of these hub nodes V. A navigation graph index on V is further constructed to minimize the model inference overhead. Extensive experiments demonstrate that GATE achieves a 1.2-2.0X speed-up in query performance compared to state-of-the-art graph-based indexes. 

---
# MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents 

**Authors**: Zijian Zhou, Ao Qu, Zhaoxuan Wu, Sunghwan Kim, Alok Prakash, Daniela Rus, Jinhua Zhao, Bryan Kian Hsiang Low, Paul Pu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2506.15841)  

**Abstract**: Modern language agents must operate over long-horizon, multi-turn interactions, where they retrieve external information, adapt to observations, and answer interdependent queries. Yet, most LLM systems rely on full-context prompting, appending all past turns regardless of their relevance. This leads to unbounded memory growth, increased computational costs, and degraded reasoning performance on out-of-distribution input lengths. We introduce MEM1, an end-to-end reinforcement learning framework that enables agents to operate with constant memory across long multi-turn tasks. At each turn, MEM1 updates a compact shared internal state that jointly supports memory consolidation and reasoning. This state integrates prior memory with new observations from the environment while strategically discarding irrelevant or redundant information. To support training in more realistic and compositional settings, we propose a simple yet effective and scalable approach to constructing multi-turn environments by composing existing datasets into arbitrarily complex task sequences. Experiments across three domains, including internal retrieval QA, open-domain web QA, and multi-turn web shopping, show that MEM1-7B improves performance by 3.5x while reducing memory usage by 3.7x compared to Qwen2.5-14B-Instruct on a 16-objective multi-hop QA task, and generalizes beyond the training horizon. Our results demonstrate the promise of reasoning-driven memory consolidation as a scalable alternative to existing solutions for training long-horizon interactive agents, where both efficiency and performance are optimized. 

---
# cAST: Enhancing Code Retrieval-Augmented Generation with Structural Chunking via Abstract Syntax Tree 

**Authors**: Yilin Zhang, Xinran Zhao, Zora Zhiruo Wang, Chenyang Yang, Jiayi Wei, Tongshuang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15655)  

**Abstract**: Retrieval-Augmented Generation (RAG) has become essential for large-scale code generation, grounding predictions in external code corpora to improve actuality. However, a critical yet underexplored aspect of RAG pipelines is chunking -- the process of dividing documents into retrievable units. Existing line-based chunking heuristics often break semantic structures, splitting functions or merging unrelated code, which can degrade generation quality. We propose chunking via Abstract Syntax Trees (\ourwork), a structure-aware method that recursively breaks large AST nodes into smaller chunks and merges sibling nodes while respecting size limits. This approach generates self-contained, semantically coherent units across programming languages and tasks, improving performance on diverse code generation tasks, e.g., boosting Recall@5 by 4.3 points on RepoEval retrieval and Pass@1 by 2.67 points on SWE-bench generation. Our work highlights the importance of structure-aware chunking for scaling retrieval-enhanced code intelligence. 

---
