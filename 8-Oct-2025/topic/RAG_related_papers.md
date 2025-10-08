# Deterministic Legal Retrieval: An Action API for Querying the SAT-Graph RAG 

**Authors**: Hudson de Martim  

**Link**: [PDF](https://arxiv.org/pdf/2510.06002)  

**Abstract**: The Structure-Aware Temporal Graph RAG (SAT-Graph RAG) addresses core limitations of standard Retrieval-Augmented Generation in the legal domain by providing a verifiable knowledge graph that models hierarchical structure, temporal evolution, and causal events of legal norms. However, a critical gap remains: how to reliably query this structured knowledge without sacrificing its deterministic properties. This paper introduces the SAT-Graph API, a formal query execution layer centered on canonical actions-atomic, composable, and auditable primitives that isolate probabilistic discovery from deterministic retrieval. These actions enable: (i) high-precision hybrid search; (ii) robust reference resolution; (iii) point-in-time version retrieval; and (iv) auditable causal tracing. We demonstrate how planner-guided agents can decompose complex queries into Directed Acyclic Graphs (DAGs) of these actions. This two-layer architecture transforms retrieval from an opaque black box to a transparent, auditable process, directly addressing Explainable AI (XAI) requirements for high-stakes domains. 

---
# MHA-RAG: Improving Efficiency, Accuracy, and Consistency by Encoding Exemplars as Soft Prompts 

**Authors**: Abhinav Jain, Xinyu Yao, Thomas Reps, Christopher Jermaine  

**Link**: [PDF](https://arxiv.org/pdf/2510.05363)  

**Abstract**: Adapting Foundation Models to new domains with limited training data is challenging and computationally expensive. While prior work has demonstrated the effectiveness of using domain-specific exemplars as in-context demonstrations, we investigate whether representing exemplars purely as text is the most efficient, effective, and stable approach. We explore an alternative: representing exemplars as soft prompts with an exemplar order invariant model architecture. To this end, we introduce Multi-Head Attention Retrieval-Augmented Generation (MHA-RAG), a framework with the number of attention heads serving as a simple hyperparameter to control soft prompt-generation across different tasks. Across multiple question-answering benchmarks and model scales, MHA-RAG achieves a 20-point performance gain over standard RAG, while cutting inference costs by a factor of 10X GFLOPs-delivering both higher accuracy and greater efficiency, invariant to exemplar order. 

---
# RAG Makes Guardrails Unsafe? Investigating Robustness of Guardrails under RAG-style Contexts 

**Authors**: Yining She, Daniel W. Peterson, Marianne Menglin Liu, Vikas Upadhyay, Mohammad Hossein Chaghazardi, Eunsuk Kang, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2510.05310)  

**Abstract**: With the increasing adoption of large language models (LLMs), ensuring the safety of LLM systems has become a pressing concern. External LLM-based guardrail models have emerged as a popular solution to screen unsafe inputs and outputs, but they are themselves fine-tuned or prompt-engineered LLMs that are vulnerable to data distribution shifts. In this paper, taking Retrieval Augmentation Generation (RAG) as a case study, we investigated how robust LLM-based guardrails are against additional information embedded in the context. Through a systematic evaluation of 3 Llama Guards and 2 GPT-oss models, we confirmed that inserting benign documents into the guardrail context alters the judgments of input and output guardrails in around 11% and 8% of cases, making them unreliable. We separately analyzed the effect of each component in the augmented context: retrieved documents, user query, and LLM-generated response. The two mitigation methods we tested only bring minor improvements. These results expose a context-robustness gap in current guardrails and motivate training and evaluation protocols that are robust to retrieval and query composition. 

---
# DecEx-RAG: Boosting Agentic Retrieval-Augmented Generation with Decision and Execution Optimization via Process Supervision 

**Authors**: Yongqi Leng, Yikun Lei, Xikai Liu, Meizhi Zhong, Bojian Xiong, Yurong Zhang, Yan Gao, Yi Wu, Yao Hu, Deyi Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2510.05691)  

**Abstract**: Agentic Retrieval-Augmented Generation (Agentic RAG) enhances the processing capability for complex tasks through dynamic retrieval and adaptive workflows. Recent advances (e.g., Search-R1) have shown that outcome-supervised reinforcement learning demonstrate strong performance. However, this approach still suffers from inefficient exploration, sparse reward signals, and ambiguous global reward feedback. To address these challenges, we propose DecEx-RAG, which models RAG as a Markov Decision Process (MDP) incorporating decision-making and execution, while introducing an efficient pruning strategy to optimize data expansion. Through comprehensive process-level policy optimization, DecEx-RAG significantly enhances the autonomous task decomposition, dynamic retrieval, and high-quality answer generation capabilities of large language models (LLMs). Experiments show that DecEx-RAG achieves an average absolute performance improvement of $6.2\%$ across six datasets, significantly outperforming existing baselines. Moreover, the pruning strategy improves data construction efficiency by nearly $6 \times$, providing an efficient solution for process-supervised RAG training. The code is available at this https URL. 

---
# KEO: Knowledge Extraction on OMIn via Knowledge Graphs and RAG for Safety-Critical Aviation Maintenance 

**Authors**: Kuangshi Ai, Jonathan A. Karr Jr, Meng Jiang, Nitesh V. Chawla, Chaoli Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.05524)  

**Abstract**: We present Knowledge Extraction on OMIn (KEO), a domain-specific knowledge extraction and reasoning framework with large language models (LLMs) in safety-critical contexts. Using the Operations and Maintenance Intelligence (OMIn) dataset, we construct a QA benchmark spanning global sensemaking and actionable maintenance tasks. KEO builds a structured Knowledge Graph (KG) and integrates it into a retrieval-augmented generation (RAG) pipeline, enabling more coherent, dataset-wide reasoning than traditional text-chunk RAG. We evaluate locally deployable LLMs (Gemma-3, Phi-4, Mistral-Nemo) and employ stronger models (GPT-4o, Llama-3.3) as judges. Experiments show that KEO markedly improves global sensemaking by revealing patterns and system-level insights, while text-chunk RAG remains effective for fine-grained procedural tasks requiring localized retrieval. These findings underscore the promise of KG-augmented LLMs for secure, domain-specific QA and their potential in high-stakes reasoning. 

---
# WeatherArchive-Bench: Benchmarking Retrieval-Augmented Reasoning for Historical Weather Archives 

**Authors**: Yongan Yu, Xianda Du, Qingchen Hu, Jiahao Liang, Jingwei Ni, Dan Qiang, Kaiyu Huang, Grant McKenzie, Renee Sieber, Fengran Mo  

**Link**: [PDF](https://arxiv.org/pdf/2510.05336)  

**Abstract**: Historical archives on weather events are collections of enduring primary source records that offer rich, untapped narratives of how societies have experienced and responded to extreme weather events. These qualitative accounts provide insights into societal vulnerability and resilience that are largely absent from meteorological records, making them valuable for climate scientists to understand societal responses. However, their vast scale, noisy digitized quality, and archaic language make it difficult to transform them into structured knowledge for climate research. To address this challenge, we introduce WeatherArchive-Bench, the first benchmark for evaluating retrieval-augmented generation (RAG) systems on historical weather archives. WeatherArchive-Bench comprises two tasks: WeatherArchive-Retrieval, which measures a system's ability to locate historically relevant passages from over one million archival news segments, and WeatherArchive-Assessment, which evaluates whether Large Language Models (LLMs) can classify societal vulnerability and resilience indicators from extreme weather narratives. Extensive experiments across sparse, dense, and re-ranking retrievers, as well as a diverse set of LLMs, reveal that dense retrievers often fail on historical terminology, while LLMs frequently misinterpret vulnerability and resilience concepts. These findings highlight key limitations in reasoning about complex societal indicators and provide insights for designing more robust climate-focused RAG systems from archival contexts. The constructed dataset and evaluation framework are publicly available at this https URL. 

---
# Demystifying deep search: a holistic evaluation with hint-free multi-hop questions and factorised metrics 

**Authors**: Maojia Song, Renhang Liu, Xinyu Wang, Yong Jiang, Pengjun Xie, Fei Huang, Soujanya Poria, Jingren Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.05137)  

**Abstract**: RAG (Retrieval-Augmented Generation) systems and web agents are increasingly evaluated on multi-hop deep search tasks, yet current practice suffers from two major limitations. First, most benchmarks leak the reasoning path in the question text, allowing models to follow surface cues rather than discover reasoning chains autonomously. Second, evaluation is typically reduced to a single pass rate, which collapses diverse behaviours into one score and obscures whether failures stem from inadequate search, poor knowledge use, or inappropriate refusal. To address these issues, we present WebDetective, a benchmark of hint-free multi-hop questions paired with a controlled Wikipedia sandbox that ensures full traceability of model actions, and a holistic evaluation framework that separates search sufficiency, knowledge utilisation, and refusal behaviour. Our evaluation of 25 state-of-the-art models reveals systematic weaknesses across all architectures: models struggle with knowledge utilisation despite having sufficient evidence and demonstrate near-absent appropriate refusal when evidence is lacking. These patterns expose a fundamental gap: today's systems excel at executing given reasoning paths but fail when required to discover them. We develop an agentic workflow, EvidenceLoop, that explicitly targets the challenges our benchmark identifies, incorporating verification loops and systematic evidence tracking that improve both search and synthesis capabilities. This baseline demonstrates that WebDetective's diagnostic framework can guide concrete architectural improvements, establishing our benchmark as a critical tool for developing genuinely autonomous reasoning systems rather than pattern-following agents. 

---
