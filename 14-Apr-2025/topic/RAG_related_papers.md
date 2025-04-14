# PCA-RAG: Principal Component Analysis for Efficient Retrieval-Augmented Generation 

**Authors**: Arman Khaledian, Amirreza Ghadiridehkordi, Nariman Khaledian  

**Link**: [PDF](https://arxiv.org/pdf/2504.08386)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a powerful paradigm for grounding large language models in external knowledge sources, improving the precision of agents responses. However, high-dimensional language model embeddings, often in the range of hundreds to thousands of dimensions, can present scalability challenges in terms of storage and latency, especially when processing massive financial text corpora. This paper investigates the use of Principal Component Analysis (PCA) to reduce embedding dimensionality, thereby mitigating computational bottlenecks without incurring large accuracy losses. We experiment with a real-world dataset and compare different similarity and distance metrics under both full-dimensional and PCA-compressed embeddings. Our results show that reducing vectors from 3,072 to 110 dimensions provides a sizeable (up to $60\times$) speedup in retrieval operations and a $\sim 28.6\times$ reduction in index size, with only moderate declines in correlation metrics relative to human-annotated similarity scores. These findings demonstrate that PCA-based compression offers a viable balance between retrieval fidelity and resource efficiency, essential for real-time systems such as Zanista AI's \textit{Newswitch} platform. Ultimately, our study underscores the practicality of leveraging classical dimensionality reduction techniques to scale RAG architectures for knowledge-intensive applications in finance and trading, where speed, memory efficiency, and accuracy must jointly be optimized. 

---
# RAG-VR: Leveraging Retrieval-Augmented Generation for 3D Question Answering in VR Environments 

**Authors**: Shiyi Ding, Ying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.08256)  

**Abstract**: Recent advances in large language models (LLMs) provide new opportunities for context understanding in virtual reality (VR). However, VR contexts are often highly localized and personalized, limiting the effectiveness of general-purpose LLMs. To address this challenge, we present RAG-VR, the first 3D question-answering system for VR that incorporates retrieval-augmented generation (RAG), which augments an LLM with external knowledge retrieved from a localized knowledge database to improve the answer quality. RAG-VR includes a pipeline for extracting comprehensive knowledge about virtual environments and user conditions for accurate answer generation. To ensure efficient retrieval, RAG-VR offloads the retrieval process to a nearby edge server and uses only essential information during retrieval. Moreover, we train the retriever to effectively distinguish among relevant, irrelevant, and hard-to-differentiate information in relation to questions. RAG-VR improves answer accuracy by 17.9%-41.8% and reduces end-to-end latency by 34.5%-47.3% compared with two baseline systems. 

---
# TP-RAG: Benchmarking Retrieval-Augmented Large Language Model Agents for Spatiotemporal-Aware Travel Planning 

**Authors**: Hang Ni, Fan Liu, Xinyu Ma, Lixin Su, Shuaiqiang Wang, Dawei Yin, Hui Xiong, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.08694)  

**Abstract**: Large language models (LLMs) have shown promise in automating travel planning, yet they often fall short in addressing nuanced spatiotemporal rationality. While existing benchmarks focus on basic plan validity, they neglect critical aspects such as route efficiency, POI appeal, and real-time adaptability. This paper introduces TP-RAG, the first benchmark tailored for retrieval-augmented, spatiotemporal-aware travel planning. Our dataset includes 2,348 real-world travel queries, 85,575 fine-grain annotated POIs, and 18,784 high-quality travel trajectory references sourced from online tourist documents, enabling dynamic and context-aware planning. Through extensive experiments, we reveal that integrating reference trajectories significantly improves spatial efficiency and POI rationality of the travel plan, while challenges persist in universality and robustness due to conflicting references and noisy data. To address these issues, we propose EvoRAG, an evolutionary framework that potently synergizes diverse retrieved trajectories with LLMs' intrinsic reasoning. EvoRAG achieves state-of-the-art performance, improving spatiotemporal compliance and reducing commonsense violation compared to ground-up and retrieval-augmented baselines. Our work underscores the potential of hybridizing Web knowledge with LLM-driven optimization, paving the way for more reliable and adaptive travel planning agents. 

---
# Out of Style: RAG's Fragility to Linguistic Variation 

**Authors**: Tianyu Cao, Neel Bhandari, Akhila Yerukola, Akari Asai, Maarten Sap  

**Link**: [PDF](https://arxiv.org/pdf/2504.08231)  

**Abstract**: Despite the impressive performance of Retrieval-augmented Generation (RAG) systems across various NLP benchmarks, their robustness in handling real-world user-LLM interaction queries remains largely underexplored. This presents a critical gap for practical deployment, where user queries exhibit greater linguistic variations and can trigger cascading errors across interdependent RAG components. In this work, we systematically analyze how varying four linguistic dimensions (formality, readability, politeness, and grammatical correctness) impact RAG performance. We evaluate two retrieval models and nine LLMs, ranging from 3 to 72 billion parameters, across four information-seeking Question Answering (QA) datasets. Our results reveal that linguistic reformulations significantly impact both retrieval and generation stages, leading to a relative performance drop of up to 40.41% in Recall@5 scores for less formal queries and 38.86% in answer match scores for queries containing grammatical errors. Notably, RAG systems exhibit greater sensitivity to such variations compared to LLM-only generations, highlighting their vulnerability to error propagation due to linguistic shifts. These findings highlight the need for improved robustness techniques to enhance reliability in diverse user interactions. 

---
