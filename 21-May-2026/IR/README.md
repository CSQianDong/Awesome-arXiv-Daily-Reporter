# SG-LegalCite: A Principle-Augmented Benchmark for Legal Citation Retrieval in Singapore Law 

**Authors**: Shannon Lee Yueh Ern, Kaidong Feng, Yingpeng Du, Chloe Lee En Jia, Zhu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2605.21057)  

**Abstract**: Legal citation in common-law systems depends not only on factual similarity, but also on the legal principle for which a precedent is invoked. However, existing benchmarks for legal citation retrieval use case facts, citation context, or full judgments as inputs, where the governing legal principle is often missing or only implicitly expressed and entangled with broader context. As a result, models may retrieve precedents that are factually similar yet doctrinally irrelevant. This limitation is particularly consequential in Singapore, where the legal system has evolved independently: only domestic precedents are binding, while foreign authorities serve merely as persuasive references. Thus, we propose a new retrieval paradigm that ranks cited cases based on queries integrating case facts and explicit legal principles, inspired by real-world legal reasoning workflows. To support this paradigm, we introduce SG-LegalCite, a dataset of 100,890 case-principle pairs extracted from 8,523 Singapore Supreme Court judgments spanning from 2000 to 2025. Experiments across 11 baselines demonstrate the effectiveness of our principle-augmented retrieval paradigm, showing that explicit legal principles provide strong discriminative signals for legal citation retrieval. 

---
# MemConflict: Evaluating Long-Term Memory Systems Under Memory Conflicts 

**Authors**: Zhen Tao, Jinxiang Zhao, Peng Liu, Dinghao Xi, Yanfang Chen, Wei Xu, Zhiyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.20926)  

**Abstract**: Long-term memory systems enable conversational agents based on large language models (LLMs) to retain, retrieve, and apply user-specific information across multi-session interactions. However, existing evaluations mainly assess outcome-level performance or temporal updating, providing limited insight into how systems retrieve and rank temporally valid, factually correct, and contextually applicable memory evidence under conflicting alternatives. To address this gap, we propose MemConflict, a diagnostic framework that treats memory validity as a query-conditioned fitness-for-use problem. MemConflict formalizes dynamic, static, and conditional conflicts over temporal validity, factual correctness, and contextual applicability. It simulates controlled long-horizon histories from structured user profiles, introduces cross-session conflicts, and injects semantically similar distractors to create competition among memory candidates. The resulting multi-session dialogue benchmark supports black-box evaluation of final answers and white-box analysis of supporting-memory retrieval and ranking. Experiments on six representative long-term memory systems show uneven strengths across conflict types, with answer correctness often diverging from memory retrieval and ranking. Sensitivity analyses reveal that longer histories, distractors, implicit queries, and larger conflict distances degrade performance. Diagnostics show failures from missing supporting memories and ineffective use of retrieved memories. Collectively, MemConflict advances principled long-term memory governance through retrieval-aware, conflict-aware reliability assessment. 

---
# CALMem : Application-Layer Dual Memory for Conversational AI 

**Authors**: Rajendra Narayan Jena, Rajan Padmanabhan, Sankar Arumugam  

**Link**: [PDF](https://arxiv.org/pdf/2605.20724)  

**Abstract**: Large language models (LLMs) operate within fixed context windows that fundamentally limit conversational continuity. When context fills, compaction discards history irreversibly; when sessions end, all memory resets to zero. Existing solutions-larger context windows, retrieval-augmented generation for knowledge bases, and memory-augmented architectures such as MemGPT-either require model modification, impose provider lock-in, or do not address the compaction continuity problem. We present CALMem (Conversational Application-Layer Memory), an application-layer dual memory architecture that gives LLM-based conversational assistants virtually unbounded effective context without any modification to the underlying model. CALMem combines two complementary memory subsystems: an episodic memory layer built on sliding-window vector embeddings of conversation history, and a semantic memory layer of agent-writable structured facts. A token-budget-adaptive injection mechanism, called the MOIM (Message of Injected Memory), automatically retrieves and injects relevant past context each turn, scaling injection depth inversely with context pressure. A key contribution is intra-session retrieval: compacted away turns from the current session remain searchable, closing a gap unaddressed by prior work. The system is implemented as a pure application layer in a production Rust codebase, is provider-agnostic, and degrades to original LLM behaviour with zero overhead when disabled. We describe the architecture, design decisions, and performance characteristics, and analyse the trade-offs that guided each implementation choice. 

---
# Layer-wise Token Compression for Efficient Document Reranking 

**Authors**: Shengyao Zhuang, zhichao Xu, Ivano Lauriola  

**Link**: [PDF](https://arxiv.org/pdf/2605.20683)  

**Abstract**: Transformer-based document cross-encoder rerankers are a central component of modern information retrieval systems. Despite their success, these models suffer from high computational costs due to processing long query-document sequences at inference time. A known approach to improve efficiency is token compression, which consists of aggregating groups of tokens together in the initial embedding layer, reducing the effective number of tokens, and making the computation faster. While token compression has proven to be successful for bi-encoder retrievers, we empirically observed that this approach may be ineffective for cross-encoder rerankers. In this paper, we propose Layer-wise Token Compression (LTC), which applies adaptive token pooling at intermediate transformer layers. Through extensive ablation studies on MS MARCO passage and document ranking tasks, we demonstrate that compression at middle layers preserves ranking quality while increasing inference QPS by up to 25% for passage ranking and up to 116% for document ranking. We also extend LTC to listwise LLM rerankers and show that the same approach can be easily applied to long-context listwise reranking, where the QPS improvements are even greater. More surprisingly, when applying rerankers trained on short passages to long-document ranking tasks, models trained with compression outperform their uncompressed counterparts, suggesting that compression may act as a beneficial regularizer that encourages length-invariant representations. 

---
# Efficient Table QA via TableGrid Navigation and Progressive Inference Prompting 

**Authors**: Amritansh Maurya, Navjot Singh, Mohammed Javed, Omar Moured  

**Link**: [PDF](https://arxiv.org/pdf/2605.20254)  

**Abstract**: Large Language Models (LLMs) have shown promising results on NLP tasks, however, their performance on tabular data still needs research attention, because Table Question-Answering (TQA) requires precise cell retrieval and multi-step structured reasoning. Existing work improves TQA either by fine-tuning or training LLMs on task-specific tabular data, but often lacks verifiable control over how the model navigates tables and derives answers. In this work, we propose a training-free TQA approach with two structured prompting frameworks: TableGrid Navigation (TGN), which iteratively navigates rows and columns via a three-module loop to locate evidence and refine answers, and Progressive Inference Prompting (PIP), which enforces columns identification for explicit progressive row selection constraint according to the query. We evaluate 17 LLMs against 6 baselines on TableBench and FeTaQa dataset. On TableBench, TGN improves over the strongest baseline by 3.8 points, and on FeTaQa, PIP achieves SOTA performance over ReAct and Chain-of-Thought. Beyond inference-time gains, PIP and TGN can also serve as supervision templates to fine-tune small models, narrowing the performance gap to much larger architectures in resource-constrained settings, offering versatile and cost-efficient solution for TQA. 

---
# GraphRAG on Consumer Hardware: Benchmarking Local LLMs for Healthcare EHR Schema Retrieval 

**Authors**: Peter Fernandes, Ria Kanjilal  

**Link**: [PDF](https://arxiv.org/pdf/2605.20815)  

**Abstract**: Graph-based Retrieval Augmented Generation (GraphRAG) extends retrieval-augmented generation to support structured reasoning over complex corpora, but its reliability under resource-constrained, privacy-sensitive deployments remains unclear. In healthcare, where Electronic Health Record (EHR) data is complex and strictly regulated, reliance on cloud-based large language models (LLMs) introduces challenges in cost, latency, and compliance. In this work, we present a systematic evaluation of GraphRAG for EHR schema retrieval using locally deployed open-source LLMs. We implement the Microsoft GraphRAG pipeline on real-world EHR schema documentation and benchmark four models, including Llama 3.1 (8B), Mistral (7B), Qwen 2.5 (7B), and Phi-4-mini (3.8B), each deployed via Ollama on a single consumer GPU (8 GB VRAM). We evaluate indexing efficiency, knowledge graph construction, query latency, answer quality, and hallucination under both global and local retrieval modes. Our results reveal substantial differences: Llama 3.1 produces the richest knowledge graph (1,172 entities), Qwen 2.5 achieves the best answer quality (3.3/5), Phi-4-mini fails to complete the pipeline due to structured-output errors, and Mistral exhibits degenerate repetition behavior. We further show that GraphRAG exhibits a practical capacity threshold, where models below approximately 7B parameters fail to reliably produce valid structured outputs and cannot complete the pipeline. In addition, indexing and answer quality are decoupled across models, and local retrieval consistently outperforms global summarization in both latency and factual grounding, with reduced hallucination. These findings demonstrate that GraphRAG is feasible on consumer hardware while highlighting the importance of model selection and retrieval design for robust deployment in regulated settings. 

---
# DIVE: Embedding Compression via Self-Limiting Gradient Updates 

**Authors**: Dongfang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.20689)  

**Abstract**: High-dimensional embeddings from large language models impose significant storage and computational costs on vector search systems. Recent embedding compression methods, including Matryoshka-Adaptor (EMNLP 2024), Search-Adaptor (ACL 2024), and SMEC (EMNLP 2025), enable dimensionality reduction through lightweight residual adapters, but their training objectives cause severe overfitting when labeled data is scarce, degrading retrieval performance below the frozen baseline. We propose \textsc{DIVE} (\textbf{D}imensionality reduction with \textbf{I}mplicit \textbf{V}iew \textbf{E}nsembles), a compression adapter that addresses this failure through two mechanisms. First, a self-limiting hinge-based triplet loss produces zero gradient once a triplet satisfies the margin constraint, bounding the total perturbation applied to the pretrained embedding space. Second, a head-wise NT-Xent contrastive loss treats multiple learned projections of each embedding as implicit views, providing dense self-supervised gradients that compensate for the sparsity of the triplet signal on small datasets. Across six BEIR datasets, \textsc{DIVE} outperforms all three baseline adapters on every dataset and at every evaluated compression ratio, with a 14M-parameter open-source implementation. 

---
# Advanced Scientific Methodology Plays Rossini 

**Authors**: Silvia Licciardi, Daniela Macchione, Emmanuel Caronna, Elisa Francomano  

**Link**: [PDF](https://arxiv.org/pdf/2605.20220)  

**Abstract**: A musical score provides the essential instructions for its performance while containing indications - at times implicit - regarding the composer's intentions. The presence of authorial variants, and even more so complex series of revisions associated with a single text, presents a challenging path for analytical study. This research, situated within the application of Scientific Methodologies to Music Philology, proposes a methodological approach oriented toward the structural analysis of one of the many settings composed by Gioachino Rossini on the same Metastasio arietta ``Mi lagnerò tacendo''. Through Computational Analysis - incorporating parsing, data mining, and graph theory - the melodic, harmonic, and textual compositional choices have been rigorously explored. The results constitute a significant unicum in the field, laying the foundation for a systematic study that supports philological research and paves the way for the use of generative models to investigate the creative process. 

---
