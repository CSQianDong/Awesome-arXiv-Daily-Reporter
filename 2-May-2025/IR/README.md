# Investigating Task Arithmetic for Zero-Shot Information Retrieval 

**Authors**: Marco Braga, Pranav Kasela, Alessandro Raganato, Gabriella Pasi  

**Link**: [PDF](https://arxiv.org/pdf/2505.00649)  

**Abstract**: Large Language Models (LLMs) have shown impressive zero-shot performance across a variety of Natural Language Processing tasks, including document re-ranking. However, their effectiveness degrades on unseen tasks and domains, largely due to shifts in vocabulary and word distributions. In this paper, we investigate Task Arithmetic, a technique that combines the weights of LLMs pre-trained on different tasks or domains via simple mathematical operations, such as addition or subtraction, to adapt retrieval models without requiring additional fine-tuning. Our method is able to synthesize diverse tasks and domain knowledge into a single model, enabling effective zero-shot adaptation in different retrieval contexts. Extensive experiments on publicly available scientific, biomedical, and multilingual datasets show that our method improves state-of-the-art re-ranking performance by up to 18% in NDCG@10 and 15% in P@10. In addition to these empirical gains, our analysis provides insights into the strengths and limitations of Task Arithmetic as a practical strategy for zero-shot learning and model adaptation. We make our code publicly available at this https URL. 

---
# Efficient Recommendation with Millions of Items by Dynamic Pruning of Sub-Item Embeddings 

**Authors**: Aleksandr V. Petrov, Craig Macdonald, Nicola Tonellotto  

**Link**: [PDF](https://arxiv.org/pdf/2505.00560)  

**Abstract**: A large item catalogue is a major challenge for deploying modern sequential recommender models, since it makes the memory footprint of the model large and increases inference latency. One promising approach to address this is RecJPQ, which replaces item embeddings with sub-item embeddings. However, slow inference remains problematic because finding the top highest-scored items usually requires scoring all items in the catalogue, which may not be feasible for large catalogues. By adapting dynamic pruning concepts from document retrieval, we propose the RecJPQPrune dynamic pruning algorithm to efficiently find the top highest-scored items without computing the scores of all items in the catalogue. Our RecJPQPrune algorithm is safe-up-to-rank K since it theoretically guarantees that no potentially high-scored item is excluded from the final top K recommendation list, thereby ensuring no impact on effectiveness. Our experiments on two large datasets and three recommendation models demonstrate the efficiency achievable using RecJPQPrune: for instance, on the Tmall dataset with 2.2M items, we can reduce the median model scoring time by 64 times compared to the Transformer Default baseline, and 5.3 times compared to a recent scoring approach called PQTopK. Overall, this paper demonstrates the effective and efficient inference of Transformer-based recommendation models at catalogue scales not previously reported in the literature. Indeed, our RecJPQPrune algorithm can score 2 million items in under 10 milliseconds without GPUs, and without relying on Approximate Nearest Neighbour (ANN) techniques. 

---
# Graph Spectral Filtering with Chebyshev Interpolation for Recommendation 

**Authors**: Chanwoo Kim, Jinkyu Sung, Yebonn Han, Joonseok Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.00552)  

**Abstract**: Graph convolutional networks have recently gained prominence in collaborative filtering (CF) for recommendations. However, we identify potential bottlenecks in two foundational components. First, the embedding layer leads to a latent space with limited capacity, overlooking locally observed but potentially valuable preference patterns. Also, the widely-used neighborhood aggregation is limited in its ability to leverage diverse preference patterns in a fine-grained manner. Building on spectral graph theory, we reveal that these limitations stem from graph filtering with a cut-off in the frequency spectrum and a restricted linear form. To address these issues, we introduce ChebyCF, a CF framework based on graph spectral filtering. Instead of a learned embedding, it takes a user's raw interaction history to utilize the full spectrum of signals contained in it. Also, it adopts Chebyshev interpolation to effectively approximate a flexible non-linear graph filter, and further enhances it by using an additional ideal pass filter and degree-based normalization. Through extensive experiments, we verify that ChebyCF overcomes the aforementioned bottlenecks and achieves state-of-the-art performance across multiple benchmarks and reasonably fast inference. Our code is available at this https URL. 

---
# EnronQA: Towards Personalized RAG over Private Documents 

**Authors**: Michael J. Ryan, Danmei Xu, Chris Nivera, Daniel Campos  

**Link**: [PDF](https://arxiv.org/pdf/2505.00263)  

**Abstract**: Retrieval Augmented Generation (RAG) has become one of the most popular methods for bringing knowledge-intensive context to large language models (LLM) because of its ability to bring local context at inference time without the cost or data leakage risks associated with fine-tuning. A clear separation of private information from the LLM training has made RAG the basis for many enterprise LLM workloads as it allows the company to augment LLM's understanding using customers' private documents. Despite its popularity for private documents in enterprise deployments, current RAG benchmarks for validating and optimizing RAG pipelines draw their corpora from public data such as Wikipedia or generic web pages and offer little to no personal context. Seeking to empower more personal and private RAG we release the EnronQA benchmark, a dataset of 103,638 emails with 528,304 question-answer pairs across 150 different user inboxes. EnronQA enables better benchmarking of RAG pipelines over private data and allows for experimentation on the introduction of personalized retrieval settings over realistic data. Finally, we use EnronQA to explore the tradeoff in memorization and retrieval when reasoning over private documents. 

---
# Optimization of embeddings storage for RAG systems using quantization and dimensionality reduction techniques 

**Authors**: Naamán Huerga-Pérez, Rubén Álvarez, Rubén Ferrero-Guillén, Alberto Martínez-Gutiérrez, Javier Díez-González  

**Link**: [PDF](https://arxiv.org/pdf/2505.00105)  

**Abstract**: Retrieval-Augmented Generation enhances language models by retrieving relevant information from external knowledge bases, relying on high-dimensional vector embeddings typically stored in float32 precision. However, storing these embeddings at scale presents significant memory challenges. To address this issue, we systematically investigate on MTEB benchmark two complementary optimization strategies: quantization, evaluating standard formats (float16, int8, binary) and low-bit floating-point types (float8), and dimensionality reduction, assessing methods like PCA, Kernel PCA, UMAP, Random Projections and Autoencoders. Our results show that float8 quantization achieves a 4x storage reduction with minimal performance degradation (<0.3%), significantly outperforming int8 quantization at the same compression level, being simpler to implement. PCA emerges as the most effective dimensionality reduction technique. Crucially, combining moderate PCA (e.g., retaining 50% dimensions) with float8 quantization offers an excellent trade-off, achieving 8x total compression with less performance impact than using int8 alone (which provides only 4x compression). To facilitate practical application, we propose a methodology based on visualizing the performance-storage trade-off space to identify the optimal configuration that maximizes performance within their specific memory constraints. 

---
# Clustering Internet Memes Through Template Matching and Multi-Dimensional Similarity 

**Authors**: Tygo Bloem, Filip Ilievski  

**Link**: [PDF](https://arxiv.org/pdf/2505.00056)  

**Abstract**: Meme clustering is critical for toxicity detection, virality modeling, and typing, but it has received little attention in previous research. Clustering similar Internet memes is challenging due to their multimodality, cultural context, and adaptability. Existing approaches rely on databases, overlook semantics, and struggle to handle diverse dimensions of similarity. This paper introduces a novel method that uses template-based matching with multi-dimensional similarity features, thus eliminating the need for predefined databases and supporting adaptive matching. Memes are clustered using local and global features across similarity categories such as form, visual content, text, and identity. Our combined approach outperforms existing clustering methods, producing more consistent and coherent clusters, while similarity-based feature sets enable adaptability and align with human intuition. We make all supporting code publicly available to support subsequent research. Code: this https URL 

---
# Graph RAG for Legal Norms: A Hierarchical and Temporal Approach 

**Authors**: Hudson de Martim  

**Link**: [PDF](https://arxiv.org/pdf/2505.00039)  

**Abstract**: This article proposes an adaptation of Graph Retrieval Augmented Generation (Graph RAG) specifically designed for the analysis and comprehension of legal norms, which are characterized by their predefined hierarchical structure, extensive network of internal and external references and multiple temporal versions. By combining structured knowledge graphs with contextually enriched text segments, Graph RAG offers a promising solution to address the inherent complexity and vast volume of legal data. The integration of hierarchical structure and temporal evolution into knowledge graphs - along with the concept of comprehensive Text Units - facilitates the construction of richer, interconnected representations of legal knowledge. Through a detailed analysis of Graph RAG and its application to legal norm datasets, this article aims to significantly advance the field of Artificial Intelligence applied to Law, creating opportunities for more effective systems in legal research, legislative analysis, and decision support. 

---
# Enhancing Speech-to-Speech Dialogue Modeling with End-to-End Retrieval-Augmented Generation 

**Authors**: Pengchao Feng, Ziyang Ma, Wenxi Chen, Yao Li, Sheng Wang, Kai Yu, Xie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.00028)  

**Abstract**: In recent years, end-to-end speech-to-speech (S2S) dialogue systems have garnered increasing research attention due to their advantages over traditional cascaded systems, including achieving lower latency and more natural integration of nonverbal cues such as emotion and speaker identity. However, these end-to-end systems face key challenges, particularly in incorporating external knowledge, a capability commonly addressed by Retrieval-Augmented Generation (RAG) in text-based large language models (LLMs). The core difficulty lies in the modality gap between input speech and retrieved textual knowledge, which hinders effective integration. To address this issue, we propose a novel end-to-end RAG framework that directly retrieves relevant textual knowledge from speech queries, eliminating the need for intermediate speech-to-text conversion via techniques like ASR. Experimental results demonstrate that our method significantly improves the performance of end-to-end S2S dialogue systems while achieving higher retrieval efficiency. Although the overall performance still lags behind cascaded models, our framework offers a promising direction for enhancing knowledge integration in end-to-end S2S systems. We will release the code and dataset to support reproducibility and promote further research in this area. 

---
