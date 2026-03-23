# The $\mathbf{Y}$-Combinator for LLMs: Solving Long-Context Rot with $λ$-Calculus 

**Authors**: Amartya Roy, Rasul Tutunov, Xiaotong Ji, Matthieu Zimmer, Haitham Bou-Ammar  

**Link**: [PDF](https://arxiv.org/pdf/2603.20105)  

**Abstract**: LLMs are increasingly used as general-purpose reasoners, but long inputs remain bottlenecked by a fixed context window. Recursive Language Models (RLMs) address this by externalising the prompt and recursively solving subproblems. Yet existing RLMs depend on an open-ended read-eval-print loop (REPL) in which the model generates arbitrary control code, making execution difficult to verify, predict, and analyse.
We introduce $\lambda$-RLM, a framework for long-context reasoning that replaces free-form recursive code generation with a typed functional runtime grounded in $\lambda$-calculus. It executes a compact library of pre-verified combinators and uses neural inference only on bounded leaf subproblems, turning recursive reasoning into a structured functional program with explicit control flow. We show that $\lambda$-RLM admits formal guarantees absent from standard RLMs, including termination, closed-form cost bounds, controlled accuracy scaling with recursion depth, and an optimal partition rule under a simple cost model. Empirically, across four long-context reasoning tasks and nine base models, $\lambda$-RLM outperforms standard RLM in 29 of 36 model-task comparisons, improves average accuracy by up to +21.9 points across model tiers, and reduces latency by up to 4.1x. These results show that typed symbolic control yields a more reliable and efficient foundation for long-context reasoning than open-ended recursive code generation. The complete implementation of $\lambda$-RLM, is open-sourced for the community at: this https URL. 

---
# MOSS-TTSD: Text to Spoken Dialogue Generation 

**Authors**: Yuqian Zhang, Donghua Yu, Zhengyuan Lin, Botian Jiang, Mingshu Chen, Yaozhou Jiang, Yiwei Zhao, Yiyang Zhang, Yucheng Yuan, Hanfu Chen, Kexin Huang, Jun Zhan, Cheng Chang, Zhaoye Fei, Shimin Li, Xiaogui Yang, Qinyuan Cheng, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2603.19739)  

**Abstract**: Spoken dialogue generation is crucial for applications like podcasts, dynamic commentary, and entertainment content, but poses significant challenges compared to single-utterance text-to-speech (TTS). Key requirements include accurate turn-taking, cross-turn acoustic consistency, and long-form stability, which current models often fail to address due to a lack of dialogue context modeling. To bridge this gap, we present MOSS-TTSD, a spoken dialogue synthesis model designed for expressive, multi-party conversational speech across multiple languages. With enhanced long-context modeling, MOSS-TTSD generates long-form spoken conversations from dialogue scripts with explicit speaker tags, supporting up to 60 minutes of single-pass synthesis, multi-party dialogue with up to 5 speakers, and zero-shot voice cloning from a short reference audio clip. The model supports various mainstream languages, including English and Chinese, and is adapted to several long-form scenarios. Additionally, to address limitations of existing evaluation methods, we propose TTSD-eval, an objective evaluation framework based on forced alignment that measures speaker attribution accuracy and speaker similarity without relying on speaker diarization tools. Both objective and subjective evaluation results show that MOSS-TTSD surpasses strong open-source and proprietary baselines in dialogue synthesis. 

---
# FDARxBench: Benchmarking Regulatory and Clinical Reasoning on FDA Generic Drug Assessment 

**Authors**: Betty Xiong, Jillian Fisher, Benjamin Newman, Meng Hu, Shivangi Gupta, Yejin Choi, Lanyan Fang, Russ B Altman  

**Link**: [PDF](https://arxiv.org/pdf/2603.19539)  

**Abstract**: We introduce an expert curated, real-world benchmark for evaluating document-grounded question-answering (QA) motivated by generic drug assessment, using the U.S. Food and Drug Administration (FDA) drug label documents. Drug labels contain rich but heterogeneous clinical and regulatory information, making accurate question answering difficult for current language models. In collaboration with FDA regulatory assessors, we introduce FDARxBench, and construct a multi-stage pipeline for generating high-quality, expert curated, QA examples spanning factual, multi-hop, and refusal tasks, and design evaluation protocols to assess both open-book and closed-book reasoning. Experiments across proprietary and open-weight models reveal substantial gaps in factual grounding, long-context retrieval, and safe refusal behavior. While motivated by FDA generic drug assessment needs, this benchmark also provides a substantial foundation for challenging regulatory-grade evaluation of label comprehension. The benchmark is designed to support evaluation of LLM behavior on drug-label questions. 

---
# OmniDiT: Extending Diffusion Transformer to Omni-VTON Framework 

**Authors**: Weixuan Zeng, Pengcheng Wei, Huaiqing Wang, Boheng Zhang, Jia Sun, Dewen Fan, Lin HE, Long Chen, Qianqian Gan, Fan Yang, Tingting Gao  

**Link**: [PDF](https://arxiv.org/pdf/2603.19643)  

**Abstract**: Despite the rapid advancement of Virtual Try-On (VTON) and Try-Off (VTOFF) technologies, existing VTON methods face challenges with fine-grained detail preservation, generalization to complex scenes, complicated pipeline, and efficient inference. To tackle these problems, we propose OmniDiT, an omni Virtual Try-On framework based on the Diffusion Transformer, which combines try-on and try-off tasks into one unified model. Specifically, we first establish a self-evolving data curation pipeline to continuously produce data, and construct a large VTON dataset Omni-TryOn, which contains over 380k diverse and high-quality garment-model-tryon image pairs and detailed text prompts. Then, we employ the token concatenation and design an adaptive position encoding to effectively incorporate multiple reference conditions. To relieve the bottleneck of long sequence computation, we are the first to introduce Shifted Window Attention into the diffusion model, thus achieving a linear complexity. To remedy the performance degradation caused by local window attention, we utilize multiple timestep prediction and an alignment loss to improve generation fidelity. Experiments reveal that, under various complex scenes, our method achieves the best performance in both the model-free VTON and VTOFF tasks and a performance comparable to current SOTA methods in the model-based VTON task. 

---
# BEAVER: A Training-Free Hierarchical Prompt Compression Method via Structure-Aware Page Selection 

**Authors**: Zhengpei Hu, Kai Li, Dapeng Fu, Chang Zeng, Yue Li, Yuanhao Tang, Jianqiang Huang  

**Link**: [PDF](https://arxiv.org/pdf/2603.19635)  

**Abstract**: The exponential expansion of context windows in LLMs has unlocked capabilities for long-document understanding but introduced severe bottlenecks in inference latency and information utilization. Existing compression methods often suffer from high training costs or semantic fragmentation due to aggressive token pruning. In this paper, we propose BEAVER, a novel training-free framework that shifts compression from linear token removal to structure-aware hierarchical selection. BEAVER maximizes hardware parallelism by mapping variable-length contexts into dense page-level tensors via dual-path pooling, and preserves discourse integrity through a hybrid planner combining semantic and lexical dual-branch selection with sentence smoothing. Extensive evaluations on four long-context benchmarks demonstrate that BEAVER achieves comparable performance to state-of-the-art (SOTA) methods like LongLLMLingua. Notably, on the RULER benchmark, BEAVER maintains high fidelity in multi-needle retrieval where baselines deteriorate. Regarding efficiency, BEAVER reduces latency by 26.4x on 128k contexts, offering a scalable solution for high-throughput applications. Our code is available at this https URL. 

---
# Enhancing Legal LLMs through Metadata-Enriched RAG Pipelines and Direct Preference Optimization 

**Authors**: Suyash Maniyar, Deepali Singh, Rohith Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2603.19251)  

**Abstract**: Large Language Models (LLMs) perform well in short contexts but degrade on long legal documents, often producing hallucinations such as incorrect clauses or precedents. In the legal domain, where precision is critical, such errors undermine reliability and trust.
Retrieval Augmented Generation (RAG) helps ground outputs but remains limited in legal settings, especially with small, locally deployed models required for data privacy. We identify two failure modes: retrieval errors due to lexical redundancy in legal corpora, and decoding errors where models generate answers despite insufficient context.
To address this, we propose Metadata Enriched Hybrid RAG to improve document level retrieval, and apply Direct Preference Optimization (DPO) to enforce safe refusal when context is inadequate. Together, these methods improve grounding, reliability, and safety in legal language models. 

---
# Can Structural Cues Save LLMs? Evaluating Language Models in Massive Document Streams 

**Authors**: Yukyung Lee, Yebin Lim, Woojun Jung, Wonjun Choi, Susik Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2603.19250)  

**Abstract**: Evaluating language models in streaming environments is critical, yet underexplored. Existing benchmarks either focus on single complex events or provide curated inputs for each query, and do not evaluate models under the conflicts that arise when multiple concurrent events are mixed within the same document stream. We introduce StreamBench, a benchmark built from major news stories in 2016 and 2025, comprising 605 events and 15,354 documents across three tasks: Topic Clustering, Temporal Question Answering, and Summarization. To diagnose how models fail, we compare performance with and without structural cues, which organize key facts by event. We find that structural cues improve performance on clustering (up to +4.37%) and temporal QA (up to +9.63%), helping models locate relevant information and separate distinct events. While temporal reasoning remains an open challenge inherent to current LLMs, consistent gains across tasks show that structural cues are a promising direction for future work in massive document streams. 

---
