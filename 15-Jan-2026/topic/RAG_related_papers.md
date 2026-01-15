# Private LLM Inference on Consumer Blackwell GPUs: A Practical Guide for Cost-Effective Local Deployment in SMEs 

**Authors**: Jonathan Knoop, Hendrik Holtmann  

**Link**: [PDF](https://arxiv.org/pdf/2601.09527)  

**Abstract**: SMEs increasingly seek alternatives to cloud LLM APIs, which raise data privacy concerns. Dedicated cloud GPU instances offer improved privacy but with limited guarantees and ongoing costs, while professional on-premise hardware (A100, H100) remains prohibitively expensive. We present a systematic evaluation of NVIDIA's Blackwell consumer GPUs (RTX 5060 Ti, 5070 Ti, 5090) for production LLM inference, benchmarking four open-weight models (Qwen3-8B, Gemma3-12B, Gemma3-27B, GPT-OSS-20B) across 79 configurations spanning quantization formats (BF16, W4A16, NVFP4, MXFP4), context lengths (8k-64k), and three workloads: RAG, multi-LoRA agentic serving, and high-concurrency APIs. The RTX 5090 delivers 3.5-4.6x higher throughput than the 5060 Ti with 21x lower latency for RAG, but budget GPUs achieve the highest throughput-per-dollar for API workloads with sub-second latency. NVFP4 quantization provides 1.6x throughput over BF16 with 41% energy reduction and only 2-4% quality loss. Self-hosted inference costs $0.001-0.04 per million tokens (electricity only), which is 40-200x cheaper than budget-tier cloud APIs, with hardware breaking even in under four months at moderate volume (30M tokens/day). Our results show that consumer GPUs can reliably replace cloud inference for most SME workloads, except latency-critical long-context RAG, where high-end GPUs remain essential. We provide deployment guidance and release all benchmark data for reproducible SME-scale deployments. 

---
# OpenDecoder: Open Large Language Model Decoding to Incorporate Document Quality in RAG 

**Authors**: Fengran Mo, Zhan Su, Yuchen Hui, Jinghan Zhang, Jia Ao Sun, Zheyuan Liu, Chao Zhang, Tetsuya Sakai, Jian-Yun Nie  

**Link**: [PDF](https://arxiv.org/pdf/2601.09028)  

**Abstract**: The development of large language models (LLMs) has achieved superior performance in a range of downstream tasks, including LLM-based retrieval-augmented generation (RAG). The quality of generated content heavily relies on the usefulness of the retrieved information and the capacity of LLMs' internal information processing mechanism to incorporate it in answer generation. It is generally assumed that the retrieved information is relevant to the question. However, the retrieved information may have a variable degree of relevance and usefulness, depending on the question and the document collection. It is important to take into account the relevance of the retrieved information in answer generation. In this paper, we propose OpenDecoder, a new approach that leverages explicit evaluation of the retrieved information as quality indicator features for generation. We aim to build a RAG model that is more robust to varying levels of noisy context. Three types of explicit evaluation information are considered: relevance score, ranking score, and QPP (query performance prediction) score. The experimental results on five benchmark datasets demonstrate the effectiveness and better robustness of OpenDecoder by outperforming various baseline methods. Importantly, this paradigm is flexible to be integrated with the post-training of LLMs for any purposes and incorporated with any type of external indicators. 

---
# Más contexto no es mejor. Paradoja de la dilución vectorial en RAG corporativos 

**Authors**: Alex Dantart  

**Link**: [PDF](https://arxiv.org/pdf/2601.08851)  

**Abstract**: Técnicas recientes de "Contextualized Chunking" inyectan resúmenes para mejorar el contexto en RAG, pero introducen una "dilución vectorial" que opaca el contenido local. Evaluando distintos ratios de inyección, demostramos una curva en "U invertida": una inyección moderada mejora el "Recall" (+18%), pero superar un umbral crítico (CIR > 0.4) reduce la precisión en un 22% para consultas específicas. Proponemos un marco teórico para calcular el ratio óptimo de inyección. --
Recent "Contextualized Chunking" techniques inject summaries to improve RAG context but introduce "vector dilution" drowning out local content. Evaluating various injection ratios, we demonstrate an "inverted U" curve: moderate injection boosts Recall (+18%), but exceeding a critical threshold (CIR > 0.4) drops precision by 22% for specific queries. We propose a theoretical framework to calculate the optimal injection ratio. 

---
# SpectraQuery: A Hybrid Retrieval-Augmented Conversational Assistant for Battery Science 

**Authors**: Sreya Vangara, Jagjit Nanda, Yan-Kai Tzeng, Eric Darve  

**Link**: [PDF](https://arxiv.org/pdf/2601.09036)  

**Abstract**: Scientific reasoning increasingly requires linking structured experimental data with the unstructured literature that explains it, yet most large language model (LLM) assistants cannot reason jointly across these modalities. We introduce SpectraQuery, a hybrid natural-language query framework that integrates a relational Raman spectroscopy database with a vector-indexed scientific literature corpus using a Structured and Unstructured Query Language (SUQL)-inspired design. By combining semantic parsing with retrieval-augmented generation, SpectraQuery translates open-ended questions into coordinated SQL and literature retrieval operations, producing cited answers that unify numerical evidence with mechanistic explanation. Across SQL correctness, answer groundedness, retrieval effectiveness, and expert evaluation, SpectraQuery demonstrates strong performance: approximately 80 percent of generated SQL queries are fully correct, synthesized answers reach 93-97 percent groundedness with 10-15 retrieved passages, and battery scientists rate responses highly across accuracy, relevance, grounding, and clarity (4.1-4.6/5). These results show that hybrid retrieval architectures can meaningfully support scientific workflows by bridging data and discourse for high-volume experimental datasets. 

---
# Structured Knowledge Representation through Contextual Pages for Retrieval-Augmented Generation 

**Authors**: Xinze Li, Zhenghao Liu, Haidong Xin, Yukun Yan, Shuo Wang, Zheni Zeng, Sen Mei, Ge Yu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2601.09402)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by incorporating external knowledge. Recently, some works have incorporated iterative knowledge accumulation processes into RAG models to progressively accumulate and refine query-related knowledge, thereby constructing more comprehensive knowledge representations. However, these iterative processes often lack a coherent organizational structure, which limits the construction of more comprehensive and cohesive knowledge representations. To address this, we propose PAGER, a page-driven autonomous knowledge representation framework for RAG. PAGER first prompts an LLM to construct a structured cognitive outline for a given question, which consists of multiple slots representing a distinct knowledge aspect. Then, PAGER iteratively retrieves and refines relevant documents to populate each slot, ultimately constructing a coherent page that serves as contextual input for guiding answer generation. Experiments on multiple knowledge-intensive benchmarks and backbone models show that PAGER consistently outperforms all RAG baselines. Further analyses demonstrate that PAGER constructs higher-quality and information-dense knowledge representations, better mitigates knowledge conflicts, and enables LLMs to leverage external knowledge more effectively. All code is available at this https URL. 

---
# When to Trust: A Causality-Aware Calibration Framework for Accurate Knowledge Graph Retrieval-Augmented Generation 

**Authors**: Jing Ren, Bowen Li, Ziqi Xu, Xinkun Zhang, Haytham Fayek, Xiaodong Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.09241)  

**Abstract**: Knowledge Graph Retrieval-Augmented Generation (KG-RAG) extends the RAG paradigm by incorporating structured knowledge from knowledge graphs, enabling Large Language Models (LLMs) to perform more precise and explainable reasoning. While KG-RAG improves factual accuracy in complex tasks, existing KG-RAG models are often severely overconfident, producing high-confidence predictions even when retrieved sub-graphs are incomplete or unreliable, which raises concerns for deployment in high-stakes domains. To address this issue, we propose Ca2KG, a Causality-aware Calibration framework for KG-RAG. Ca2KG integrates counterfactual prompting, which exposes retrieval-dependent uncertainties in knowledge quality and reasoning reliability, with a panel-based re-scoring mechanism that stabilises predictions across interventions. Extensive experiments on two complex QA datasets demonstrate that Ca2KG consistently improves calibration while maintaining or even enhancing predictive accuracy. 

---
