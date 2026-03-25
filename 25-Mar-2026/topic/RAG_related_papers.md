# Leveraging Large Language Models to Extract and Translate Medical Information in Doctors' Notes for Health Records and Diagnostic Billing Codes 

**Authors**: Peter Hartnett, Chung-Chi Huang, Sarah Hartnett, David Hartnett  

**Link**: [PDF](https://arxiv.org/pdf/2603.22625)  

**Abstract**: Physician burnout in the United States has reached critical levels, driven in part by the administrative burden of Electronic Health Record (EHR) documentation and complex diagnostic codes. To relieve this strain and maintain strict patient privacy, this thesis explores an on-device, offline automatic medical coding system. The work focuses on using open-weight Large Language Models (LLMs) to extract clinical information from physician notes and translate it into ICD-10-CM diagnostic codes without reliance on cloud-based services.
A privacy-focused pipeline was developed using Ollama, LangChain, and containerized environments to evaluate multiple open-weight models, including Llama 3.2, Mistral, Phi, and DeepSeek, on consumer-grade hardware. Model performance was assessed for zero-shot, few-shot, and retrieval-augmented generation (RAG) prompting strategies using a novel benchmark of synthetic medical notes.
Results show that strict JSON schema enforcement achieved near 100% formatting compliance, but accurate generation of specific diagnostic codes remains challenging for smaller local models (7B-20B parameters). Contrary to common prompt-engineering guidance, few-shot prompting degraded performance through overfitting and hallucinations. While RAG enabled limited discovery of unseen codes, it frequently saturated context windows, reducing overall accuracy. The findings suggest that fully automated unsupervised coding with local open-source models is not yet reliable; instead, a human-in-the-loop assisted coding approach is currently the most practical path forward. This work contributes a reproducible local LLM architecture and benchmark dataset for privacy-preserving medical information extraction and coding. 

---
# ProGRank: Probe-Gradient Reranking to Defend Dense-Retriever RAG from Corpus Poisoning 

**Authors**: Xiangyu Yin, Yi Qi, Chih-hong Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2603.22934)  

**Abstract**: Retrieval-Augmented Generation (RAG) improves the reliability of large language model applications by grounding generation in retrieved evidence, but it also introduces a new attack surface: corpus poisoning. In this setting, an adversary injects or edits passages so that they are ranked into the Top-$K$ results for target queries and then affect downstream generation. Existing defences against corpus poisoning often rely on content filtering, auxiliary models, or generator-side reasoning, which can make deployment more difficult. We propose ProGRank, a post hoc, training-free retriever-side defence for dense-retriever RAG. ProGRank stress-tests each query--passage pair under mild randomized perturbations and extracts probe gradients from a small fixed parameter subset of the retriever. From these signals, it derives two instability signals, representational consistency and dispersion risk, and combines them with a score gate in a reranking step. ProGRank preserves the original passage content, requires no retraining, and also supports a surrogate-based variant when the deployed retriever is unavailable. Extensive experiments across three datasets, three dense retriever backbones, representative corpus poisoning attacks, and both retrieval-stage and end-to-end settings show that ProGRank provides stronger defence performance and a favorable robustness--utility trade-off. It also remains competitive under adaptive evasive attacks. 

---
# Graph-Aware Late Chunking for Retrieval-Augmented Generation in Biomedical Literature 

**Authors**: Pouria Mortezaagha, Arya Rahgozar  

**Link**: [PDF](https://arxiv.org/pdf/2603.22633)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems for biomedical literature are typically evaluated using ranking metrics like Mean Reciprocal Rank (MRR), which measure how well the system identifies the single most relevant chunk. We argue that for full-text scientific documents, this paradigm is incomplete: it rewards retrieval precision while ignoring retrieval breadth -- the ability to surface evidence from across a document's structural sections. We propose GraLC-RAG, a framework that unifies late chunking with graph-aware structural intelligence, introducing structure-aware chunk boundary detection, UMLS knowledge graph infusion, and graph-guided hybrid retrieval. We evaluate six strategies on 2,359 IMRaD-filtered PubMed Central articles using 2,033 cross-section questions and two metric families: standard ranking metrics (MRR, Recall@k) and structural coverage metrics (SecCov@k, CS Recall). Our results expose a sharp divergence: content-similarity methods achieve the highest MRR (0.517) but always retrieve from a single section, while structure-aware methods retrieve from up to 15.6x more sections. Generation experiments show that KG-infused retrieval narrows the answer-quality gap to delta-F1 = 0.009 while maintaining 4.6x section diversity. These findings demonstrate that standard metrics systematically undervalue structural retrieval and that closing the multi-section synthesis gap is a key open problem for biomedical RAG. 

---
# GraphRAG for Engineering Diagrams: ChatP&ID Enables LLM Interaction with P&IDs 

**Authors**: Achmad Anggawirya Alimin, Artur M. Schweidtmann  

**Link**: [PDF](https://arxiv.org/pdf/2603.22528)  

**Abstract**: Large Language Models (LLMs) combined with Retrieval-Augmented Generation (RAG) and knowledge graphs offer new opportunities for interacting with engineering diagrams such as Piping and Instrumentation Diagrams (P&IDs). However, directly processing raw images or smart P&ID files with LLMs is often costly, inefficient, and prone to hallucinations. This work introduces ChatP&ID, an agentic framework that enables grounded and cost-effective natural-language interaction with P&IDs using Graph Retrieval-Augmented Generation (GraphRAG), a paradigm we refer to as GraphRAG for engineering diagrams. Smart P&IDs encoded in the DEXPI standard are transformed into structured knowledge graphs, which serve as the basis for graph-based retrieval and reasoning by LLM agents. This approach enables reliable querying of engineering diagrams while significantly reducing computational cost. Benchmarking across commercial LLM APIs (OpenAI, Anthropic) demonstrates that graph-based representations improve accuracy by 18% over raw image inputs and reduce token costs by 85% compared to directly ingesting smart P&ID files. While small open-source models still struggle to interpret knowledge graph formats and structured engineering data, integrating them with VectorRAG and PathRAG improves response accuracy by up to 40%. Notably, GPT-5-mini combined with ContextRAG achieves 91% accuracy at a cost of only $0.004 per task. The resulting ChatP&ID interface enables intuitive natural-language interaction with complex engineering diagrams and lays the groundwork for AI-assisted process engineering tasks such as Hazard and Operability Studies (HAZOP) and multi-agent analysis. 

---
# Parametric Knowledge and Retrieval Behavior in RAG Fine-Tuning for Electronic Design Automation 

**Authors**: Julian Oestreich, Maximilian Bley, Frank Binder, Lydia Müller, Maksym Sydorenko, André Alcalde  

**Link**: [PDF](https://arxiv.org/pdf/2603.23047)  

**Abstract**: Retrieval-Augmented Generation (RAG) fine-tuning has shown substantial improvements over vanilla RAG, yet most studies target document question answering and often rely on standard NLP metrics that can obscure factual differences. We evaluate RAG fine-tuning for long-form text generation in electronic design automation, adapting a 7B model under five context augmentation strategies with varying retrieval conditions. We introduce TriFEX, a human-validated, triple-based evaluation pipeline that attributes generated claims to their origin-user query, context and reference-and propose Parametric Knowledge Precision (PKP), which isolates internalized knowledge by filtering out claims leaked in the prompt. We show that ROUGE and BERTScore fail to detect factual differences that our triple-based evaluation reveals. Additionally, we demonstrate that an existing metric for knowledge internalization is retrieva-sensitive, with about 75% of its cross-condition variance driven by changes in the rate at which internal knowledge is expressed (PR), rather than by changes in its actual correctness (PKP). The fine-tuned 7B variants outperform a 72B baseline on most metrics, further showing generalization across conditions and on a related benchmark. These results underscore the limitations of available metrics in RAG evaluation and show that smaller models could be reasonably well adapted to specialized tasks for cost-efficient, on-premises deployment. 

---
# Graphs RAG at Scale: Beyond Retrieval-Augmented Generation With Labeled Property Graphs and Resource Description Framework for Complex and Unknown Search Spaces 

**Authors**: Manie Tadayon, Mayank Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2603.22340)  

**Abstract**: Recent advances in Retrieval-Augmented Generation (RAG) have revolutionized knowledge-intensive tasks, yet traditional RAG methods struggle when the search space is unknown or when documents are semi-structured or structured. We introduce a novel end-to-end Graph RAG framework that leverages both Labeled Property Graph (LPG) and Resource Description Framework (RDF) architectures to overcome these limitations. Our approach enables dynamic document retrieval without the need to pre-specify the number of documents and eliminates inefficient reranking. We propose an innovative method for converting documents into RDF triplets using JSON key-value pairs, facilitating seamless integration of semi-structured data. Additionally, we present a text to Cypher framework for LPG, achieving over 90% accuracy in real-time translation of text queries to Cypher, enabling fast and reliable query generation suitable for online applications. Our empirical evaluation demonstrates that Graph RAG significantly outperforms traditional embedding-based RAG in accuracy, response quality, and reasoning, especially for complex, semi-structured tasks. These findings establish Graph RAG as a transformative solution for next-generation retrieval-augmented systems. 

---
