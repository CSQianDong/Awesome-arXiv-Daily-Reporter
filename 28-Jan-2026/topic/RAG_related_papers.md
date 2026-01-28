# When Iterative RAG Beats Ideal Evidence: A Diagnostic Study in Scientific Multi-hop Question Answering 

**Authors**: Mahdi Astaraki, Mohammad Arshi Saloot, Ali Shiraee Kasmaee, Hamidreza Mahyar, Soheila Samiee  

**Link**: [PDF](https://arxiv.org/pdf/2601.19827)  

**Abstract**: Retrieval-Augmented Generation (RAG) extends large language models (LLMs) beyond parametric knowledge, yet it is unclear when iterative retrieval-reasoning loops meaningfully outperform static RAG, particularly in scientific domains with multi-hop reasoning, sparse domain knowledge, and heterogeneous evidence. We provide the first controlled, mechanism-level diagnostic study of whether synchronized iterative retrieval and reasoning can surpass an idealized static upper bound (Gold Context) RAG. We benchmark eleven state-of-the-art LLMs under three regimes: (i) No Context, measuring reliance on parametric memory; (ii) Gold Context, where all oracle evidence is supplied at once; and (iii) Iterative RAG, a training-free controller that alternates retrieval, hypothesis refinement, and evidence-aware stopping. Using the chemistry-focused ChemKGMultiHopQA dataset, we isolate questions requiring genuine retrieval and analyze behavior with diagnostics spanning retrieval coverage gaps, anchor-carry drop, query quality, composition fidelity, and control calibration. Across models, Iterative RAG consistently outperforms Gold Context, with gains up to 25.6 percentage points, especially for non-reasoning fine-tuned models. Staged retrieval reduces late-hop failures, mitigates context overload, and enables dynamic correction of early hypothesis drift, but remaining failure modes include incomplete hop coverage, distractor latch trajectories, early stopping miscalibration, and high composition failure rates even with perfect retrieval. Overall, staged retrieval is often more influential than the mere presence of ideal evidence; we provide practical guidance for deploying and diagnosing RAG systems in specialized scientific settings and a foundation for more reliable, controllable iterative retrieval-reasoning frameworks. 

---
# RPO-RAG: Aligning Small LLMs with Relation-aware Preference Optimization for Knowledge Graph Question Answering 

**Authors**: Kaehyun Um, KyuHwan Yeom, Haerim Yang, Minyoung Choi, Hyeongjun Yang, Kyong-Ho Lee  

**Link**: [PDF](https://arxiv.org/pdf/2601.19225)  

**Abstract**: Large Language Models (LLMs) have recently demonstrated remarkable reasoning abilities, yet hallucinate on knowledge-intensive tasks. Retrieval-augmented generation (RAG) mitigates this issue by grounding answers in external sources, e.g., knowledge graphs (KGs). However, existing KG-based RAG approaches rely on semantics-unaware path sampling and are weakly aligned with KG reasoning objectives, which limits further accuracy gains. They also feed retrieved paths directly into the reasoner without organizing them into answer-centered reasoning paths, hindering small LLMs' ability to leverage the retrieved knowledge. Furthermore, prior works predominantly rely on large LLMs (e.g., ChatGPT/GPT-4) or assume backbones above 7B parameters, leaving sub-7B models underexplored. We address this gap with RPO-RAG, the first KG-based RAG framework specifically designed for small LLMs, to the best of our knowledge. RPO-RAG introduces three key innovations: (1) a query-path semantic sampling strategy that provides informative supervisory signals; (2) a relation-aware preference optimization that aligns training with intermediate KG reasoning signals (e.g., relation); and (3) an answer-centered prompt design that organizes entities and reasoning paths in an interpretable format. Extensive experiments on two benchmark Knowledge Graph Question Answering (KGQA) datasets, WebQSP and CWQ, demonstrate that RPO-RAG effectively bridges the performance gap between small and large language models. On WebQSP, it improves F1 by up to 8.8%, reflecting enhanced answer precision, while on CWQ it achieves new state-of-the-art results among models under 8B parameters in both Hit and F1. Overall, RPO-RAG substantially improves the reasoning capability of small LLMs, even under 3B parameters-highlighting their potential for resource-efficient and practical on-device KGQA applications. 

---
# MulVul: Retrieval-augmented Multi-Agent Code Vulnerability Detection via Cross-Model Prompt Evolution 

**Authors**: Zihan Wu, Jie Xu, Yun Peng, Chun Yong Chong, Xiaohua Jia  

**Link**: [PDF](https://arxiv.org/pdf/2601.18847)  

**Abstract**: Large Language Models (LLMs) struggle to automate real-world vulnerability detection due to two key limitations: the heterogeneity of vulnerability patterns undermines the effectiveness of a single unified model, and manual prompt engineering for massive weakness categories is unscalable.
To address these challenges, we propose \textbf{MulVul}, a retrieval-augmented multi-agent framework designed for precise and broad-coverage vulnerability detection. MulVul adopts a coarse-to-fine strategy: a \emph{Router} agent first predicts the top-$k$ coarse categories and then forwards the input to specialized \emph{Detector} agents, which identify the exact vulnerability types. Both agents are equipped with retrieval tools to actively source evidence from vulnerability knowledge bases to mitigate hallucinations.
Crucially, to automate the generation of specialized prompts, we design \emph{Cross-Model Prompt Evolution}, a prompt optimization mechanism where a generator LLM iteratively refines candidate prompts while a distinct executor LLM validates their effectiveness. This decoupling mitigates the self-correction bias inherent in single-model optimization.
Evaluated on 130 CWE types, MulVul achieves 34.79\% Macro-F1, outperforming the best baseline by 41.5\%. Ablation studies validate cross-model prompt evolution, which boosts performance by 51.6\% over manual prompts by effectively handling diverse vulnerability patterns. 

---
# Evaluation of Oncotimia: An LLM based system for supporting tumour boards 

**Authors**: Luis Lorenzo, Marcos Montana-Mendez, Sergio Figueiras, Miguel Boubeta, Cristobal Bernardo-Castineira  

**Link**: [PDF](https://arxiv.org/pdf/2601.19899)  

**Abstract**: Multidisciplinary tumour boards (MDTBs) play a central role in oncology decision-making but require manual processes and structuring large volumes of heterogeneous clinical information, resulting in a substantial documentation burden. In this work, we present ONCOTIMIA, a modular and secure clinical tool designed to integrate generative artificial intelligence (GenAI) into oncology workflows and evaluate its application to the automatic completion of lung cancer tumour board forms using large language models (LLMs). The system combines a multi-layer data lake, hybrid relational and vector storage, retrieval-augmented generation (RAG) and a rule-driven adaptive form model to transform unstructured clinical documentation into structured and standardised tumour board records. We assess the performance of six LLMs deployed through AWS Bedrock on ten lung cancer cases, measuring both completion form accuracy and end-to-end latency. The results demonstrate high performance across models, with the best performing configuration achieving an 80% of correct field completion and clinically acceptable response time for most LLMs. Larger and more recent models exhibit best accuracies without incurring prohibitive latency. These findings provide empirical evidence that LLM- assisted autocompletion form is technically feasible and operationally viable in multidisciplinary lung cancer workflows and support its potential to significantly reduce documentation burden while preserving data quality. 

---
# XProvence: Zero-Cost Multilingual Context Pruning for Retrieval-Augmented Generation 

**Authors**: Youssef Mohamed, Mohamed Elhoseiny, Thibault Formal, Nadezhda Chirkova  

**Link**: [PDF](https://arxiv.org/pdf/2601.18886)  

**Abstract**: This paper introduces XProvence, a multilingual zero-cost context pruning model for retrieval-augmented generation (RAG), trained on 16 languages and supporting 100+ languages through effective cross-lingual transfer. Motivated by the growing use of RAG systems across diverse languages, we explore several strategies to generalize the Provence framework-which first integrated efficient zero-cost context pruning directly into the re-ranking model-beyond English. Across four multilingual question answering benchmarks, we show how XProvence can prune RAG contexts with minimal-to-no performance degradation and outperforms strong baselines. Our model is available at this https URL. 

---
# LURE-RAG: Lightweight Utility-driven Reranking for Efficient RAG 

**Authors**: Manish Chandra, Debasis Ganguly, Iadh Ounis  

**Link**: [PDF](https://arxiv.org/pdf/2601.19535)  

**Abstract**: Most conventional Retrieval-Augmented Generation (RAG) pipelines rely on relevance-based retrieval, which often misaligns with utility -- that is, whether the retrieved passages actually improve the quality of the generated text specific to a downstream task such as question answering or query-based summarization. The limitations of existing utility-driven retrieval approaches for RAG are that, firstly, they are resource-intensive typically requiring query encoding, and that secondly, they do not involve listwise ranking loss during training. The latter limitation is particularly critical, as the relative order between documents directly affects generation in RAG. To address this gap, we propose Lightweight Utility-driven Reranking for Efficient RAG (LURE-RAG), a framework that augments any black-box retriever with an efficient LambdaMART-based reranker. Unlike prior methods, LURE-RAG trains the reranker with a listwise ranking loss guided by LLM utility, thereby directly optimizing the ordering of retrieved documents. Experiments on two standard datasets demonstrate that LURE-RAG achieves competitive performance, reaching 97-98% of the state-of-the-art dense neural baseline, while remaining efficient in both training and inference. Moreover, its dense variant, UR-RAG, significantly outperforms the best existing baseline by up to 3%. 

---
