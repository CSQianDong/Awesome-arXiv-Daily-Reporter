# Knowledge Elicitation with Large Language Models for Interpretable Cancer Stage Identification from Pathology Reports 

**Authors**: Yeawon Lee, Christopher C. Yang, Chia-Hsuan Chang, Grace Lu-Yao  

**Link**: [PDF](https://arxiv.org/pdf/2511.01052)  

**Abstract**: Cancer staging is critical for patient prognosis and treatment planning, yet extracting pathologic TNM staging from unstructured pathology reports poses a persistent challenge. Existing natural language processing (NLP) and machine learning (ML) strategies often depend on large annotated datasets, limiting their scalability and adaptability. In this study, we introduce two Knowledge Elicitation methods designed to overcome these limitations by enabling large language models (LLMs) to induce and apply domain-specific rules for cancer staging. The first, Knowledge Elicitation with Long-Term Memory (KEwLTM), uses an iterative prompting strategy to derive staging rules directly from unannotated pathology reports, without requiring ground-truth labels. The second, Knowledge Elicitation with Retrieval-Augmented Generation (KEwRAG), employs a variation of RAG where rules are pre-extracted from relevant guidelines in a single step and then applied, enhancing interpretability and avoiding repeated retrieval overhead. We leverage the ability of LLMs to apply broad knowledge learned during pre-training to new tasks. Using breast cancer pathology reports from the TCGA dataset, we evaluate their performance in identifying T and N stages, comparing them against various baseline approaches on two open-source LLMs. Our results indicate that KEwLTM outperforms KEwRAG when Zero-Shot Chain-of-Thought (ZSCOT) inference is effective, whereas KEwRAG achieves better performance when ZSCOT inference is less effective. Both methods offer transparent, interpretable interfaces by making the induced rules explicit. These findings highlight the promise of our Knowledge Elicitation methods as scalable, high-performing solutions for automated cancer staging with enhanced interpretability, particularly in clinical settings with limited annotated data. 

---
# KV Cache Transform Coding for Compact Storage in LLM Inference 

**Authors**: Konrad Staniszewski, Adrian Łańcucki  

**Link**: [PDF](https://arxiv.org/pdf/2511.01815)  

**Abstract**: Serving large language models (LLMs) at scale necessitates efficient key-value (KV) cache management. KV caches can be reused across conversation turns via shared-prefix prompts that are common in iterative code editing and chat. However, stale caches consume scarce GPU memory, require offloading, or force recomputation. We present KVTC, a lightweight transform coder that compresses KV caches for compact on-GPU and off-GPU storage. Drawing on classical media compression, KVTC combines PCA-based feature decorrelation, adaptive quantization, and entropy coding. It requires only a brief initial calibration and leaves model parameters unchanged. By exploiting redundancies in KV caches, KVTC achieves up to 20$\times$ compression while maintaining reasoning and long-context accuracy, and 40$\times$ or higher for specific use cases. We test KVTC with Llama 3, Mistral NeMo, and R1-Qwen 2.5 models across benchmarks including AIME25, LiveCodeBench, GSM8K, MMLU, Qasper, RULER, and MATH-500. It consistently outperforms inference-time baselines such as token eviction, quantization, and SVD-based methods, while achieving higher compression ratios. These results support KVTC as a practical building block for memory-efficient LLM serving with reusable KV caches. 

---
# Accumulating Context Changes the Beliefs of Language Models 

**Authors**: Jiayi Geng, Howard Chen, Ryan Liu, Manoel Horta Ribeiro, Robb Willer, Graham Neubig, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2511.01805)  

**Abstract**: Language model (LM) assistants are increasingly used in applications such as brainstorming and research. Improvements in memory and context size have allowed these models to become more autonomous, which has also resulted in more text accumulation in their context windows without explicit user intervention. This comes with a latent risk: the belief profiles of models -- their understanding of the world as manifested in their responses or actions -- may silently change as context accumulates. This can lead to subtly inconsistent user experiences, or shifts in behavior that deviate from the original alignment of the models. In this paper, we explore how accumulating context by engaging in interactions and processing text -- talking and reading -- can change the beliefs of language models, as manifested in their responses and this http URL results reveal that models' belief profiles are highly malleable: GPT-5 exhibits a 54.7% shift in its stated beliefs after 10 rounds of discussion about moral dilemmas and queries about safety, while Grok 4 shows a 27.2% shift on political issues after reading texts from the opposing position. We also examine models' behavioral changes by designing tasks that require tool use, where each tool selection corresponds to an implicit belief. We find that these changes align with stated belief shifts, suggesting that belief shifts will be reflected in actual behavior in agentic systems. Our analysis exposes the hidden risk of belief shift as models undergo extended sessions of talking or reading, rendering their opinions and actions unreliable. 

---
# Scalable Processing-Near-Memory for 1M-Token LLM Inference: CXL-Enabled KV-Cache Management Beyond GPU Limits 

**Authors**: Dowon Kim, MinJae Lee, Janghyeon Kim, HyuckSung Kwon, Hyeonggyu Jeong, Sang-Soo Park, Minyong Yoon, Si-Dong Roh, Yongsuk Kwon, Jinin So, Jungwook Choi  

**Link**: [PDF](https://arxiv.org/pdf/2511.00321)  

**Abstract**: The expansion of context windows in large language models (LLMs) to multi-million tokens introduces severe memory and compute bottlenecks, particularly in managing the growing Key-Value (KV) cache. While Compute Express Link (CXL) enables non-eviction frameworks that offload the full KV-cache to scalable external memory, these frameworks still suffer from costly data transfers when recalling non-resident KV tokens to limited GPU memory as context lengths increase. This work proposes scalable Processing-Near-Memory (PNM) for 1M-Token LLM Inference, a CXL-enabled KV-cache management system that coordinates memory and computation beyond GPU limits. Our design offloads token page selection to a PNM accelerator within CXL memory, eliminating costly recalls and enabling larger GPU batch sizes. We further introduce a hybrid parallelization strategy and a steady-token selection mechanism to enhance compute efficiency and scalability. Implemented atop a state-of-the-art CXL-PNM system, our solution delivers consistent performance gains for LLMs with up to 405B parameters and 1M-token contexts. Our PNM-only offloading scheme (PNM-KV) and GPU-PNM hybrid with steady-token execution (PnG-KV) achieve up to 21.9x throughput improvement, up to 60x lower energy per token, and up to 7.3x better total cost efficiency than the baseline, demonstrating that CXL-enabled multi-PNM architectures can serve as a scalable backbone for future long-context LLM inference. 

---
# Optimizing Native Sparse Attention with Latent Attention and Local Global Alternating Strategies 

**Authors**: Yuxuan Hu, Jianchao Tan, Jiaqi Zhang, Wen Zan, Pingwei Sun, Yifan Lu, Yerui Sun, Yuchen Xie, Xunliang Cai, Jing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.00819)  

**Abstract**: In this work, we conduct a systematic analysis of Native Sparse Attention (NSA) and propose targeted improvements that enhance long-context modeling. A key insight is that alternating between local (sliding-window) and global (compression, selective) attention across layers, rather than using fixed patterns, enables more effective propagation of long-range dependencies and substantially boosts performance on long-sequence tasks. Meanwhile, we further refine NSA's branches with Latent Attention that the sliding-window branch is enhanced with Multi-head Latent Attention (MLA) while compression and selective branches adopt Group-head Latent Attention (GLA). These changes reduce KV-cache memory by 50\% versus NSA while improving the model's common-sense reasoning and long-text understanding capabilities. Experiments on models from 340M to 1.3B parameters (trained on 15B and 100B tokens) show our method matches or exceeds full attention and native sparse attention in both common-sense reasoning and long-context understanding tasks. 

---
# ToM: Leveraging Tree-oriented MapReduce for Long-Context Reasoning in Large Language Models 

**Authors**: Jiani Guo, Zuchao Li, Jie Wu, Qianren Wang, Yun Li, Lefei Zhang, Hai Zhao, Yujiu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.00489)  

**Abstract**: Large Language Models (LLMs), constrained by limited context windows, often face significant performance degradation when reasoning over long contexts. To address this, Retrieval-Augmented Generation (RAG) retrieves and reasons over chunks but frequently sacrifices logical coherence due to its reliance on similarity-based rankings. Similarly, divide-and-conquer frameworks (DCF) split documents into small chunks for independent reasoning and aggregation. While effective for local reasoning, DCF struggles to capture long-range dependencies and risks inducing conflicts by processing chunks in isolation. To overcome these limitations, we propose ToM, a novel Tree-oriented MapReduce framework for long-context reasoning. ToM leverages the inherent hierarchical structure of long documents (e.g., main headings and subheadings) by constructing a DocTree through hierarchical semantic parsing and performing bottom-up aggregation. Using a Tree MapReduce approach, ToM enables recursive reasoning: in the Map step, rationales are generated at child nodes; in the Reduce step, these rationales are aggregated across sibling nodes to resolve conflicts or reach consensus at parent nodes. Experimental results on 70B+ LLMs show that ToM significantly outperforms existing divide-and-conquer frameworks and retrieval-augmented generation methods, achieving better logical coherence and long-context reasoning. Our code is available at this https URL . 

---
# LiCoMemory: Lightweight and Cognitive Agentic Memory for Efficient Long-Term Reasoning 

**Authors**: Zhengjun Huang, Zhoujin Tian, Qintian Guo, Fangyuan Zhang, Yingli Zhou, Di Jiang, Xiaofang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2511.01448)  

**Abstract**: Large Language Model (LLM) agents exhibit remarkable conversational and reasoning capabilities but remain constrained by limited context windows and the lack of persistent memory. Recent efforts address these limitations via external memory architectures, often employing graph-based representations, yet most adopt flat, entangled structures that intertwine semantics with topology, leading to redundant representations, unstructured retrieval, and degraded efficiency and accuracy. To resolve these issues, we propose LiCoMemory, an end-to-end agentic memory framework for real-time updating and retrieval, which introduces CogniGraph, a lightweight hierarchical graph that utilizes entities and relations as semantic indexing layers, and employs temporal and hierarchy-aware search with integrated reranking for adaptive and coherent knowledge retrieval. Experiments on long-term dialogue benchmarks, LoCoMo and LongMemEval, show that LiCoMemory not only outperforms established baselines in temporal reasoning, multi-session consistency, and retrieval efficiency, but also notably reduces update latency. Our official code and data are available at this https URL. 

---
