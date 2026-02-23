# Enhancing Scientific Literature Chatbots with Retrieval-Augmented Generation: A Performance Evaluation of Vector and Graph-Based Systems 

**Authors**: Hamideh Ghanadian, Amin Kamali, Mohammad Hossein Tekieh  

**Link**: [PDF](https://arxiv.org/pdf/2602.17856)  

**Abstract**: This paper investigates the enhancement of scientific literature chatbots through retrieval-augmented generation (RAG), with a focus on evaluating vector- and graph-based retrieval systems. The proposed chatbot leverages both structured (graph) and unstructured (vector) databases to access scientific articles and gray literature, enabling efficient triage of sources according to research objectives. To systematically assess performance, we examine two use-case scenarios: retrieval from a single uploaded document and retrieval from a large-scale corpus. Benchmark test sets were generated using a GPT model, with selected outputs annotated for evaluation. The comparative analysis emphasizes retrieval accuracy and response relevance, providing insight into the strengths and limitations of each approach. The findings demonstrate the potential of hybrid RAG systems to improve accessibility to scientific knowledge and to support evidence-based decision making. 

---
# Decomposing Retrieval Failures in RAG for Long-Document Financial Question Answering 

**Authors**: Amine Kobeissi, Philippe Langlais  

**Link**: [PDF](https://arxiv.org/pdf/2602.17981)  

**Abstract**: Retrieval-augmented generation is increasingly used for financial question answering over long regulatory filings, yet reliability depends on retrieving the exact context needed to justify answers in high stakes settings. We study a frequent failure mode in which the correct document is retrieved but the page or chunk that contains the answer is missed, leading the generator to extrapolate from incomplete context. Despite its practical significance, this within-document retrieval failure mode has received limited systematic attention in the Financial Question Answering (QA) literature. We evaluate retrieval at multiple levels of granularity, document, page, and chunk level, and introduce an oracle based analysis to provide empirical upper bounds on retrieval and generative performance. On a 150 question subset of FinanceBench, we reproduce and compare diverse retrieval strategies including dense, sparse, hybrid, and hierarchical methods with reranking and query reformulation. Across methods, gains in document discovery tend to translate into stronger page recall, yet oracle performance still suggests headroom for page and chunk level retrieval. To target this gap, we introduce a domain fine-tuned page scorer that treats pages as an intermediate retrieval unit between documents and chunks. Unlike prior passage-based hierarchical retrieval, we fine-tune a bi-encoder specifically for page-level relevance on financial filings, exploiting the semantic coherence of pages. Overall, our results demonstrate a significant improvement in page recall and chunk retrieval. 

---
# Aurora: Neuro-Symbolic AI Driven Advising Agent 

**Authors**: Lorena Amanda Quincoso Lugones, Christopher Kverne, Nityam Sharadkumar Bhimani, Ana Carolina Oliveira, Agoritsa Polyzou, Christine Lisetti, Janki Bhimani  

**Link**: [PDF](https://arxiv.org/pdf/2602.17999)  

**Abstract**: Academic advising in higher education is under severe strain, with advisor-to-student ratios commonly exceeding 300:1. These structural bottlenecks limit timely access to guidance, increase the risk of delayed graduation, and contribute to inequities in student support. We introduce Aurora, a modular neuro-symbolic advising agent that unifies retrieval-augmented generation (RAG), symbolic reasoning, and normalized curricular databases to deliver policy-compliant, verifiable recommendations at scale. Aurora integrates three components: (i) a Boyce-Codd Normal Form (BCNF) catalog schema for consistent program rules, (ii) a Prolog engine for prerequisite and credit enforcement, and (iii) an instruction-tuned large language model for natural-language explanations of its recommendations. To assess performance, we design a structured evaluation suite spanning common and edge-case advising scenarios, including short-term scheduling, long-term roadmapping, skill-aligned pathways, and out-of-scope requests. Across this diverse set, Aurora improves semantic alignment with expert-crafted answers from 0.68 (Raw LLM baseline) to 0.93 (+36%), achieves perfect precision and recall in nearly half of in-scope cases, and consistently produces correct fallbacks for unanswerable prompts. On commodity hardware, Aurora delivers sub-second mean latency (0.71s across 20 queries), approximately 83X faster than a Raw LLM baseline (59.2s). By combining symbolic rigor with neural fluency, Aurora advances a paradigm for accurate, explainable, and scalable AI-driven advising. 

---
# CUICurate: A GraphRAG-based Framework for Automated Clinical Concept Curation for NLP applications 

**Authors**: Victoria Blake, Mathew Miller, Jamie Novak, Sze-yuan Ooi, Blanca Gallego  

**Link**: [PDF](https://arxiv.org/pdf/2602.17949)  

**Abstract**: Background: Clinical named entity recognition tools commonly map free text to Unified Medical Language System (UMLS) Concept Unique Identifiers (CUIs). For many downstream tasks, however, the clinically meaningful unit is not a single CUI but a concept set comprising related synonyms, subtypes, and supertypes. Constructing such concept sets is labour-intensive, inconsistently performed, and poorly supported by existing tools, particularly for NLP pipelines that operate directly on UMLS CUIs. Methods We present CUICurate, a Graph-based retrieval-augmented generation (GraphRAG) framework for automated UMLS concept set curation. A UMLS knowledge graph (KG) was constructed and embedded for semantic retrieval. For each target concept, candidate CUIs were retrieved from the KG, followed by large language model (LLM) filtering and classification steps comparing two LLMs (GPT-5 and GPT-5-mini). The framework was evaluated on five lexically heterogeneous clinical concepts against a manually curated benchmark and gold-standard concept sets. Results Across all concepts, CUICurate produced substantially larger and more complete concept sets than the manual benchmarks whilst matching human precision. Comparisons between the two LLMs found that GPT-5-mini achieved higher recall during filtering, while GPT-5 produced classifications that more closely aligned with clinician judgements. Outputs were stable across repeated runs and computationally inexpensive. Conclusions CUICurate offers a scalable and reproducible approach to support UMLS concept set curation that substantially reduces manual effort. By integrating graph-based retrieval with LLM reasoning, the framework produces focused candidate concept sets that can be adapted to clinical NLP pipelines for different phenotyping and analytic requirements. 

---
