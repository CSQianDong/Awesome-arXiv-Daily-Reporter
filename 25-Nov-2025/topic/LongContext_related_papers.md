# Concept than Document: Context Compression via AMR-based Conceptual Entropy 

**Authors**: Kaize Shi, Xueyao Sun, Xiaohui Tao, Lin Li, Qika Lin, Guandong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2511.18832)  

**Abstract**: Large Language Models (LLMs) face information overload when handling long contexts, particularly in Retrieval-Augmented Generation (RAG) where extensive supporting documents often introduce redundant content. This issue not only weakens reasoning accuracy but also increases computational overhead. We propose an unsupervised context compression framework that exploits Abstract Meaning Representation (AMR) graphs to preserve semantically essential information while filtering out irrelevant text. By quantifying node-level entropy within AMR graphs, our method estimates the conceptual importance of each node, enabling the retention of core semantics. Specifically, we construct AMR graphs from raw contexts, compute the conceptual entropy of each node, and screen significant informative nodes to form a condensed and semantically focused context than raw documents. Experiments on the PopQA and EntityQuestions datasets show that our method outperforms vanilla and other baselines, achieving higher accuracy while substantially reducing context length. To the best of our knowledge, this is the first work introducing AMR-based conceptual entropy for context compression, demonstrating the potential of stable linguistic features in context engineering. 

---
# Large Language Models for the Summarization of Czech Documents: From History to the Present 

**Authors**: Václav Tran, Jakub Šmíd, Ladislav Lenc, Jean-Pierre Salmon, Pavel Král  

**Link**: [PDF](https://arxiv.org/pdf/2511.18848)  

**Abstract**: Text summarization is the task of automatically condensing longer texts into shorter, coherent summaries while preserving the original meaning and key information. Although this task has been extensively studied in English and other high-resource languages, Czech summarization, particularly in the context of historical documents, remains underexplored. This is largely due to the inherent linguistic complexity of Czech and the lack of high-quality annotated datasets.
In this work, we address this gap by leveraging the capabilities of Large Language Models (LLMs), specifically Mistral and mT5, which have demonstrated strong performance across a wide range of natural language processing tasks and multilingual settings. In addition, we also propose a translation-based approach that first translates Czech texts into English, summarizes them using an English-language model, and then translates the summaries back into Czech. Our study makes the following main contributions: We demonstrate that LLMs achieve new state-of-the-art results on the SumeCzech dataset, a benchmark for modern Czech text summarization, showing the effectiveness of multilingual LLMs even for morphologically rich, medium-resource languages like Czech. We introduce a new dataset, Posel od Čerchova, designed for the summarization of historical Czech texts. This dataset is derived from digitized 19th-century publications and annotated for abstractive summarization. We provide initial baselines using modern LLMs to facilitate further research in this underrepresented area.
By combining cutting-edge models with both modern and historical Czech datasets, our work lays the foundation for further progress in Czech summarization and contributes valuable resources for future research in Czech historical document processing and low-resource summarization more broadly. 

---
# CLaRa: Bridging Retrieval and Generation with Continuous Latent Reasoning 

**Authors**: Jie He, Richard He Bai, Sinead Williamson, Jeff Z. Pan, Navdeep Jaitly, Yizhe Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.18659)  

**Abstract**: Retrieval-augmented generation (RAG) enhances large language models (LLMs) with external knowledge but still suffers from long contexts and disjoint retrieval-generation optimization. In this work, we propose CLaRa (Continuous Latent Reasoning), a unified framework that performs embedding-based compression and joint optimization in a shared continuous space. To obtain semantically rich and retrievable compressed vectors, we introduce SCP, a key-preserving data synthesis framework using QA and paraphrase supervision. CLaRa then trains the reranker and generator end-to-end via a single language modeling loss, with gradients flowing through both modules using a differentiable top-k estimator. Theoretically, this unified optimization aligns retrieval relevance with answer quality. Experiments across multiple QA benchmarks show that CLaRa achieves state-of-the-art compression and reranking performance, often surpassing text-based fine-tuned baselines. 

---
# SWAN: Sparse Winnowed Attention for Reduced Inference Memory via Decompression-Free KV-Cache Compression 

**Authors**: Santhosh G S, Saurav Prakash, Balaraman Ravindran  

**Link**: [PDF](https://arxiv.org/pdf/2511.18936)  

**Abstract**: Large Language Models (LLMs) face a significant bottleneck during autoregressive inference due to the massive memory footprint of the Key-Value (KV) cache. Existing compression techniques like token eviction, quantization, or other low-rank methods often risk information loss, have fixed limits, or introduce significant computational overhead from explicit decompression steps. In this work, we introduce SWAN, a novel, fine-tuning-free framework that eliminates this overhead. Our method uses an offline orthogonal matrix to rotate and prune the KV-cache, which is then used directly in the attention computation without any reconstruction. Our extensive experiments demonstrate that SWAN, augmented with a small dense buffer, offers a robust trade-off, maintaining performance close to the uncompressed baseline even at aggressive 50-60% memory savings per-token on KV-cache. A key advantage is its runtime-tunable compression level, allowing operators to dynamically adjust the memory footprint, a flexibility absent in methods requiring fixed offline configurations. This combination of a decompression-free design, high performance under compression, and adaptability makes SWAN a practical and efficient solution for serving LLMs with long contexts. 

---
# Llamazip: Leveraging LLaMA for Lossless Text Compression and Training Dataset Detection 

**Authors**: Sören Dréano, Derek Molloy, Noel Murphy  

**Link**: [PDF](https://arxiv.org/pdf/2511.17589)  

**Abstract**: This work introduces Llamazip, a novel lossless text compression algorithm based on the predictive capabilities of the LLaMA3 language model. Llamazip achieves significant data reduction by only storing tokens that the model fails to predict, optimizing storage efficiency without compromising data integrity. Key factors affecting its performance, including quantization and context window size, are analyzed, revealing their impact on compression ratios and computational requirements. Beyond compression, Llamazip demonstrates the potential to identify whether a document was part of the training dataset of a language model. This capability addresses critical concerns about data provenance, intellectual property, and transparency in language model training. 

---
