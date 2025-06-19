# SciVer: Evaluating Foundation Models for Multimodal Scientific Claim Verification 

**Authors**: Chengye Wang, Yifei Shen, Zexi Kuang, Arman Cohan, Yilun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.15569)  

**Abstract**: We introduce SciVer, the first benchmark specifically designed to evaluate the ability of foundation models to verify claims within a multimodal scientific context. SciVer consists of 3,000 expert-annotated examples over 1,113 scientific papers, covering four subsets, each representing a common reasoning type in multimodal scientific claim verification. To enable fine-grained evaluation, each example includes expert-annotated supporting evidence. We assess the performance of 21 state-of-the-art multimodal foundation models, including o4-mini, Gemini-2.5-Flash, Llama-3.2-Vision, and Qwen2.5-VL. Our experiment reveals a substantial performance gap between these models and human experts on SciVer. Through an in-depth analysis of retrieval-augmented generation (RAG), and human-conducted error evaluations, we identify critical limitations in current open-source models, offering key insights to advance models' comprehension and reasoning in multimodal scientific literature tasks. 

---
# Research on Graph-Retrieval Augmented Generation Based on Historical Text Knowledge Graphs 

**Authors**: Yang Fan, Zhang Qi, Xing Wenqian, Liu Chang, Liu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15241)  

**Abstract**: This article addresses domain knowledge gaps in general large language models for historical text analysis in the context of computational humanities and AIGC technology. We propose the Graph RAG framework, combining chain-of-thought prompting, self-instruction generation, and process supervision to create a The First Four Histories character relationship dataset with minimal manual annotation. This dataset supports automated historical knowledge extraction, reducing labor costs. In the graph-augmented generation phase, we introduce a collaborative mechanism between knowledge graphs and retrieval-augmented generation, improving the alignment of general models with historical knowledge. Experiments show that the domain-specific model Xunzi-Qwen1.5-14B, with Simplified Chinese input and chain-of-thought prompting, achieves optimal performance in relation extraction (F1 = 0.68). The DeepSeek model integrated with GraphRAG improves F1 by 11% (0.08-0.19) on the open-domain C-CLUE relation extraction dataset, surpassing the F1 value of Xunzi-Qwen1.5-14B (0.12), effectively alleviating hallucinations phenomenon, and improving interpretability. This framework offers a low-resource solution for classical text knowledge extraction, advancing historical knowledge services and humanities research. 

---
# TopClustRAG at SIGIR 2025 LiveRAG Challenge 

**Authors**: Juli Bakagianni, John Pavlopoulos, Aristidis Likas  

**Link**: [PDF](https://arxiv.org/pdf/2506.15246)  

**Abstract**: We present TopClustRAG, a retrieval-augmented generation (RAG) system developed for the LiveRAG Challenge, which evaluates end-to-end question answering over large-scale web corpora. Our system employs a hybrid retrieval strategy combining sparse and dense indices, followed by K-Means clustering to group semantically similar passages. Representative passages from each cluster are used to construct cluster-specific prompts for a large language model (LLM), generating intermediate answers that are filtered, reranked, and finally synthesized into a single, comprehensive response. This multi-stage pipeline enhances answer diversity, relevance, and faithfulness to retrieved evidence. Evaluated on the FineWeb Sample-10BT dataset, TopClustRAG ranked 2nd in faithfulness and 7th in correctness on the official leaderboard, demonstrating the effectiveness of clustering-based context filtering and prompt aggregation in large-scale RAG systems. 

---
# RePCS: Diagnosing Data Memorization in LLM-Powered Retrieval-Augmented Generation 

**Authors**: Le Vu Anh, Nguyen Viet Anh, Mehmet Dik, Luong Van Nghia  

**Link**: [PDF](https://arxiv.org/pdf/2506.15513)  

**Abstract**: Retrieval-augmented generation (RAG) has become a common strategy for updating large language model (LLM) responses with current, external information. However, models may still rely on memorized training data, bypass the retrieved evidence, and produce contaminated outputs. We introduce Retrieval-Path Contamination Scoring (RePCS), a diagnostic method that detects such behavior without requiring model access or retraining. RePCS compares two inference paths: (i) a parametric path using only the query, and (ii) a retrieval-augmented path using both the query and retrieved context by computing the Kullback-Leibler (KL) divergence between their output distributions. A low divergence suggests that the retrieved context had minimal impact, indicating potential memorization. This procedure is model-agnostic, requires no gradient or internal state access, and adds only a single additional forward pass. We further derive PAC-style guarantees that link the KL threshold to user-defined false positive and false negative rates. On the Prompt-WNQA benchmark, RePCS achieves a ROC-AUC of 0.918. This result outperforms the strongest prior method by 6.5 percentage points while keeping latency overhead below 4.7% on an NVIDIA T4 GPU. RePCS offers a lightweight, black-box safeguard to verify whether a RAG system meaningfully leverages retrieval, making it especially valuable in safety-critical applications. 

---
