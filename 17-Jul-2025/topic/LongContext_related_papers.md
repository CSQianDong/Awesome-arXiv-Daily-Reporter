# Characterizing State Space Model (SSM) and SSM-Transformer Hybrid Language Model Performance with Long Context Length 

**Authors**: Saptarshi Mitra, Rachid Karami, Haocheng Xu, Sitao Huang, Hyoukjun Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2507.12442)  

**Abstract**: The demand for machine intelligence capable of processing continuous, long-context inputs on local devices is growing rapidly. However, the quadratic complexity and memory requirements of traditional Transformer architectures make them inefficient and often unusable for these tasks. This has spurred a paradigm shift towards new architectures like State Space Models (SSMs) and hybrids, which promise near-linear scaling. While most current research focuses on the accuracy and theoretical throughput of these models, a systematic performance characterization on practical consumer hardware is critically needed to guide system-level optimization and unlock new applications.
To address this gap, we present a comprehensive, comparative benchmarking of carefully selected Transformer, SSM, and hybrid models specifically for long-context inference on consumer and embedded GPUs. Our analysis reveals that SSMs are not only viable but superior for this domain, capable of processing sequences up to 220K tokens on a 24GB consumer GPU-approximately 4x longer than comparable Transformers. While Transformers may be up to 1.8x faster at short sequences, SSMs demonstrate a dramatic performance inversion, becoming up to 4x faster at very long contexts (~57K tokens). Our operator-level analysis reveals that custom, hardware-aware SSM kernels dominate the inference runtime, accounting for over 55% of latency on edge platforms, identifying them as a primary target for future hardware acceleration. We also provide detailed, device-specific characterization results to guide system co-design for the edge. To foster further research, we will open-source our characterization framework. 

---
# Advancing Retrieval-Augmented Generation for Structured Enterprise and Internal Data 

**Authors**: Chandana Cheerla  

**Link**: [PDF](https://arxiv.org/pdf/2507.12425)  

**Abstract**: Organizations increasingly rely on proprietary enterprise data, including HR records, structured reports, and tabular documents, for critical decision-making. While Large Language Models (LLMs) have strong generative capabilities, they are limited by static pretraining, short context windows, and challenges in processing heterogeneous data formats. Conventional Retrieval-Augmented Generation (RAG) frameworks address some of these gaps but often struggle with structured and semi-structured data.
This work proposes an advanced RAG framework that combines hybrid retrieval strategies using dense embeddings (all-mpnet-base-v2) and BM25, enhanced by metadata-aware filtering with SpaCy NER and cross-encoder reranking. The framework applies semantic chunking to maintain textual coherence and retains tabular data structures to preserve row-column integrity. Quantized indexing optimizes retrieval efficiency, while human-in-the-loop feedback and conversation memory improve adaptability.
Experiments on enterprise datasets show notable improvements: Precision@5 increased by 15 percent (90 versus 75), Recall@5 by 13 percent (87 versus 74), and Mean Reciprocal Rank by 16 percent (0.85 versus 0.69). Qualitative evaluations show higher scores in Faithfulness (4.6 versus 3.0), Completeness (4.2 versus 2.5), and Relevance (4.5 versus 3.2) on a 5-point Likert scale. These results demonstrate the framework's effectiveness in delivering accurate, comprehensive, and contextually relevant responses for enterprise tasks. Future work includes extending to multimodal data and integrating agent-based retrieval. The source code will be released at this https URL 

---
# Infherno: End-to-end Agent-based FHIR Resource Synthesis from Free-form Clinical Notes 

**Authors**: Johann Frei, Nils Feldhus, Lisa Raithel, Roland Roller, Alexander Meyer, Frank Kramer  

**Link**: [PDF](https://arxiv.org/pdf/2507.12261)  

**Abstract**: For clinical data integration and healthcare services, the HL7 FHIR standard has established itself as a desirable format for interoperability between complex health data. Previous attempts at automating the translation from free-form clinical notes into structured FHIR resources rely on modular, rule-based systems or LLMs with instruction tuning and constrained decoding. Since they frequently suffer from limited generalizability and structural inconformity, we propose an end-to-end framework powered by LLM agents, code execution, and healthcare terminology database tools to address these issues. Our solution, called Infherno, is designed to adhere to the FHIR document schema and competes well with a human baseline in predicting FHIR resources from unstructured text. The implementation features a front end for custom and synthetic data and both local and proprietary models, supporting clinical data integration processes and interoperability across institutions. 

---
# BOOKCOREF: Coreference Resolution at Book Scale 

**Authors**: Giuliano Martinelli, Tommaso Bonomo, Pere-Lluís Huguet Cabot, Roberto Navigli  

**Link**: [PDF](https://arxiv.org/pdf/2507.12075)  

**Abstract**: Coreference Resolution systems are typically evaluated on benchmarks containing small- to medium-scale documents. When it comes to evaluating long texts, however, existing benchmarks, such as LitBank, remain limited in length and do not adequately assess system capabilities at the book scale, i.e., when co-referring mentions span hundreds of thousands of tokens. To fill this gap, we first put forward a novel automatic pipeline that produces high-quality Coreference Resolution annotations on full narrative texts. Then, we adopt this pipeline to create the first book-scale coreference benchmark, BOOKCOREF, with an average document length of more than 200,000 tokens. We carry out a series of experiments showing the robustness of our automatic procedure and demonstrating the value of our resource, which enables current long-document coreference systems to gain up to +20 CoNLL-F1 points when evaluated on full books. Moreover, we report on the new challenges introduced by this unprecedented book-scale setting, highlighting that current models fail to deliver the same performance they achieve on smaller documents. We release our data and code to encourage research and development of new book-scale Coreference Resolution systems at this https URL. 

---
# Graph Representations for Reading Comprehension Analysis using Large Language Model and Eye-Tracking Biomarker 

**Authors**: Yuhong Zhang, Jialu Li, Shilai Yang, Yuchen Xu, Gert Cauwenberghs, Tzyy-Ping Jung  

**Link**: [PDF](https://arxiv.org/pdf/2507.11972)  

**Abstract**: Reading comprehension is a fundamental skill in human cognitive development. With the advancement of Large Language Models (LLMs), there is a growing need to compare how humans and LLMs understand language across different contexts and apply this understanding to functional tasks such as inference, emotion interpretation, and information retrieval. Our previous work used LLMs and human biomarkers to study the reading comprehension process. The results showed that the biomarkers corresponding to words with high and low relevance to the inference target, as labeled by the LLMs, exhibited distinct patterns, particularly when validated using eye-tracking data. However, focusing solely on individual words limited the depth of understanding, which made the conclusions somewhat simplistic despite their potential significance. This study used an LLM-based AI agent to group words from a reading passage into nodes and edges, forming a graph-based text representation based on semantic meaning and question-oriented prompts. We then compare the distribution of eye fixations on important nodes and edges. Our findings indicate that LLMs exhibit high consistency in language understanding at the level of graph topological structure. These results build on our previous findings and offer insights into effective human-AI co-learning strategies. 

---
# IAM: Efficient Inference through Attention Mapping between Different-scale LLMs 

**Authors**: Yi Zhao, Zuchao Li, Hai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2507.11953)  

**Abstract**: LLMs encounter significant challenges in resource consumption nowadays, especially with long contexts. Despite extensive efforts dedicate to enhancing inference efficiency, these methods primarily exploit internal sparsity within the models, without leveraging external information for optimization. We identify the high similarity of attention matrices across different-scale LLMs, which offers a novel perspective for optimization. We first conduct a comprehensive analysis of how to measure similarity, how to select mapping Layers and whether mapping is consistency. Based on these insights, we introduce the IAM framework, which achieves dual benefits of accelerated attention computation and reduced KV cache usage by performing attention mapping between small and large LLMs. Our experimental results demonstrate that IAM can accelerate prefill by 15% and reduce KV cache usage by 22.1% without appreciably sacrificing performance. Experiments on different series of models show the generalizability of IAM. Importantly, it is also orthogonal to many existing KV cache optimization methods, making it a versatile addition to the current toolkit for enhancing LLM efficiency. 

---
# DAC: A Dynamic Attention-aware Approach for Task-Agnostic Prompt Compression 

**Authors**: Yi Zhao, Zuchao Li, Hai Zhao, Baoyuan Qi, Guoming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.11942)  

**Abstract**: Task-agnostic prompt compression leverages the redundancy in natural language to reduce computational overhead and enhance information density within prompts, especially in long-context scenarios. Existing methods predominantly rely on information entropy as the metric to compress lexical units, aiming to achieve minimal information loss. However, these approaches overlook two critical aspects: (i) the importance of attention-critical tokens at the algorithmic level, and (ii) shifts in information entropy during the compression process. Motivated by these challenges, we propose a dynamic attention-aware approach for task-agnostic prompt compression (DAC). This approach effectively integrates entropy and attention information, dynamically sensing entropy shifts during compression to achieve fine-grained prompt compression. Extensive experiments across various domains, including LongBench, GSM8K, and BBH, show that DAC consistently yields robust and substantial improvements across a diverse range of tasks and LLMs, offering compelling evidence of its efficacy. 

---
