# ArcAligner: Adaptive Recursive Aligner for Compressed Context Embeddings in RAG 

**Authors**: Jianbo Li, Yi Jiang, Sendong Zhao, Bairui Hu, Haochun Wang, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2601.05038)  

**Abstract**: Retrieval-Augmented Generation (RAG) helps LLMs stay accurate, but feeding long documents into a prompt makes the model slow and expensive. This has motivated context compression, ranging from token pruning and summarization to embedding-based compression. While researchers have tried ''compressing'' these documents into smaller summaries or mathematical embeddings, there is a catch: the more you compress the data, the more the LLM struggles to understand it. To address this challenge, we propose ArcAligner (Adaptive recursive context *Aligner*), a lightweight module integrated into the language model layers to help the model better utilize highly compressed context representations for downstream generation. It uses an adaptive ''gating'' system that only adds extra processing power when the information is complex, keeping the system fast. Across knowledge-intensive QA benchmarks, ArcAligner consistently beats compression baselines at comparable compression rates, especially on multi-hop and long-tail settings. The source code is publicly available. 

---
# A Navigational Approach for Comprehensive RAG via Traversal over Proposition Graphs 

**Authors**: Maxime Delmas, Lei Xu, André Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2601.04859)  

**Abstract**: Standard RAG pipelines based on chunking excel at simple factual retrieval but fail on complex multi-hop queries due to a lack of structural connectivity. Conversely, initial strategies that interleave retrieval with reasoning often lack global corpus awareness, while Knowledge Graph (KG)-based RAG performs strongly on complex multi-hop tasks but suffers on fact-oriented single-hop queries. To bridge this gap, we propose a novel RAG framework: ToPG (Traversal over Proposition Graphs). ToPG models its knowledge base as a heterogeneous graph of propositions, entities, and passages, effectively combining the granular fact density of propositions with graph connectivity. We leverage this structure using iterative Suggestion-Selection cycles, where the Suggestion phase enables a query-aware traversal of the graph, and the Selection phase provides LLM feedback to prune irrelevant propositions and seed the next iteration. Evaluated on three distinct QA tasks (Simple, Complex, and Abstract QA), ToPG demonstrates strong performance across both accuracy- and quality-based metrics. Overall, ToPG shows that query-aware graph traversal combined with factual granularity is a critical component for efficient structured RAG systems. ToPG is available at this https URL. 

---
# Leveraging Language Models and RAG for Efficient Knowledge Discovery in Clinical Environments 

**Authors**: Seokhwan Ko, Donghyeon Lee, Jaewoo Chun, Hyungsoo Han, Junghwan Cho  

**Link**: [PDF](https://arxiv.org/pdf/2601.04209)  

**Abstract**: Large language models (LLMs) are increasingly recognized as valuable tools across the medical environment, supporting clinical, research, and administrative workflows. However, strict privacy and network security regulations in hospital settings require that sensitive data be processed within fully local infrastructures. Within this context, we developed and evaluated a retrieval-augmented generation (RAG) system designed to recommend research collaborators based on PubMed publications authored by members of a medical institution. The system utilizes PubMedBERT for domain-specific embedding generation and a locally deployed LLaMA3 model for generative synthesis. This study demonstrates the feasibility and utility of integrating domain-specialized encoders with lightweight LLMs to support biomedical knowledge discovery under local deployment constraints. 

---
# Disco-RAG: Discourse-Aware Retrieval-Augmented Generation 

**Authors**: Dongqi Liu, Hang Ding, Qiming Feng, Jian Li, Xurong Xie, Zhucun Xue, Chengjie Wang, Jiangning Zhang, Yabiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.04377)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as an important means of enhancing the performance of large language models (LLMs) in knowledge-intensive tasks. However, most existing RAG strategies treat retrieved passages in a flat and unstructured way, which prevents the model from capturing structural cues and constrains its ability to synthesize knowledge from dispersed evidence across documents. To overcome these limitations, we propose Disco-RAG, a discourse-aware framework that explicitly injects discourse signals into the generation process. Our method constructs intra-chunk discourse trees to capture local hierarchies and builds inter-chunk rhetorical graphs to model cross-passage coherence. These structures are jointly integrated into a planning blueprint that conditions the generation. Experiments on question answering and long-document summarization benchmarks show the efficacy of our approach. Disco-RAG achieves state-of-the-art results on the benchmarks without fine-tuning. These findings underscore the important role of discourse structure in advancing RAG systems. 

---
# Enhancing Admission Inquiry Responses with Fine-Tuned Models and Retrieval-Augmented Generation 

**Authors**: Aram Virabyan  

**Link**: [PDF](https://arxiv.org/pdf/2601.04206)  

**Abstract**: University admissions offices face the significant challenge of managing high volumes of inquiries efficiently while maintaining response quality, which critically impacts prospective students' perceptions. This paper addresses the issues of response time and information accuracy by proposing an AI system integrating a fine-tuned language model with Retrieval-Augmented Generation (RAG). While RAG effectively retrieves relevant information from large datasets, its performance in narrow, complex domains like university admissions can be limited without adaptation, potentially leading to contextually inadequate responses due to the intricate rules and specific details involved. To overcome this, we fine-tuned the model on a curated dataset specific to admissions processes, enhancing its ability to interpret RAG-provided data accurately and generate domain-relevant outputs. This hybrid approach leverages RAG's ability to access up-to-date information and fine-tuning's capacity to embed nuanced domain understanding. We further explored optimization strategies for the response generation logic, experimenting with settings to balance response quality and speed, aiming for consistently high-quality outputs that meet the specific requirements of admissions communications. 

---
# Neurosymbolic Retrievers for Retrieval-augmented Generation 

**Authors**: Yash Saxena, Manas Gaur  

**Link**: [PDF](https://arxiv.org/pdf/2601.04568)  

**Abstract**: Retrieval Augmented Generation (RAG) has made significant strides in overcoming key limitations of large language models, such as hallucination, lack of contextual grounding, and issues with transparency. However, traditional RAG systems consist of three interconnected neural components - the retriever, re-ranker, and generator - whose internal reasoning processes remain opaque. This lack of transparency complicates interpretability, hinders debugging efforts, and erodes trust, especially in high-stakes domains where clear decision-making is essential. To address these challenges, we introduce the concept of Neurosymbolic RAG, which integrates symbolic reasoning using a knowledge graph with neural retrieval techniques. This new framework aims to answer two primary questions: (a) Can retrievers provide a clear and interpretable basis for document selection? (b) Can symbolic knowledge enhance the clarity of the retrieval process? We propose three methods to improve this integration. First is MAR (Knowledge Modulation Aligned Retrieval) that employs modulation networks to refine query embeddings using interpretable symbolic features, thereby making document matching more explicit. Second, KG-Path RAG enhances queries by traversing knowledge graphs to improve overall retrieval quality and interpretability. Lastly, Process Knowledge-infused RAG utilizes domain-specific tools to reorder retrieved content based on validated workflows. Preliminary results from mental health risk assessment tasks indicate that this neurosymbolic approach enhances both transparency and overall performance 

---
# RAGVUE: A Diagnostic View for Explainable and Automated Evaluation of Retrieval-Augmented Generation 

**Authors**: Keerthana Murugaraj, Salima Lamsiyah, Martin Theobald  

**Link**: [PDF](https://arxiv.org/pdf/2601.04196)  

**Abstract**: Evaluating Retrieval-Augmented Generation (RAG) systems remains a challenging task: existing metrics often collapse heterogeneous behaviors into single scores and provide little insight into whether errors arise from retrieval,reasoning, or grounding. In this paper, we introduce RAGVUE, a diagnostic and explainable framework for automated, reference-free evaluation of RAG pipelines. RAGVUE decomposes RAG behavior into retrieval quality, answer relevance and completeness, strict claim-level faithfulness, and judge calibration. Each metric includes a structured explanation, making the evaluation process transparent. Our framework supports both manual metric selection and fully automated agentic evaluation. It also provides a Python API, CLI, and a local Streamlit interface for interactive usage. In comparative experiments, RAGVUE surfaces fine-grained failures that existing tools such as RAGAS often overlook. We showcase the full RAGVUE workflow and illustrate how it can be integrated into research pipelines and practical RAG development. The source code and detailed instructions on usage are publicly available on GitHub 

---
# ArtCognition: A Multimodal AI Framework for Affective State Sensing from Visual and Kinematic Drawing Cues 

**Authors**: Behrad Binaei-Haghighi, Nafiseh Sadat Sajadi, Mehrad Liviyan, Reyhane Akhavan Kharazi, Fatemeh Amirkhani, Behnam Bahrak  

**Link**: [PDF](https://arxiv.org/pdf/2601.04297)  

**Abstract**: The objective assessment of human affective and psychological states presents a significant challenge, particularly through non-verbal channels. This paper introduces digital drawing as a rich and underexplored modality for affective sensing. We present a novel multimodal framework, named ArtCognition, for the automated analysis of the House-Tree-Person (HTP) test, a widely used psychological instrument. ArtCognition uniquely fuses two distinct data streams: static visual features from the final artwork, captured by computer vision models, and dynamic behavioral kinematic cues derived from the drawing process itself, such as stroke speed, pauses, and smoothness. To bridge the gap between low-level features and high-level psychological interpretation, we employ a Retrieval-Augmented Generation (RAG) architecture. This grounds the analysis in established psychological knowledge, enhancing explainability and reducing the potential for model hallucination. Our results demonstrate that the fusion of visual and behavioral kinematic cues provides a more nuanced assessment than either modality alone. We show significant correlations between the extracted multimodal features and standardized psychological metrics, validating the framework's potential as a scalable tool to support clinicians. This work contributes a new methodology for non-intrusive affective state assessment and opens new avenues for technology-assisted mental healthcare. 

---
# Adversarial Yet Cooperative: Multi-Perspective Reasoning in Retrieved-Augmented Language Models 

**Authors**: Can Xu, Lingyong Yan, Jiayi Wu, Haosen Wang, Shuaiqiang Wang, Yuchen Li, Jizhou Huang, Dawei Yin, Xiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.04651)  

**Abstract**: Recent advances in synergizing large reasoning models (LRMs) with retrieval-augmented generation (RAG) have shown promising results, yet two critical challenges remain: (1) reasoning models typically operate from a single, unchallenged perspective, limiting their ability to conduct deep, self-correcting reasoning over external documents, and (2) existing training paradigms rely excessively on outcome-oriented rewards, which provide insufficient signal for shaping the complex, multi-step reasoning process. To address these issues, we propose an Reasoner-Verifier framework named Adversarial Reasoning RAG (ARR). The Reasoner and Verifier engage in reasoning on retrieved evidence and critiquing each other's logic while being guided by process-aware advantage that requires no external scoring model. This reward combines explicit observational signals with internal model uncertainty to jointly optimize reasoning fidelity and verification rigor. Experiments on multiple benchmarks demonstrate the effectiveness of our method. 

---
# Self-MedRAG: a Self-Reflective Hybrid Retrieval-Augmented Generation Framework for Reliable Medical Question Answering 

**Authors**: Jessica Ryan, Alexander I. Gumilang, Robert Wiliam, Derwin Suhartono  

**Link**: [PDF](https://arxiv.org/pdf/2601.04531)  

**Abstract**: Large Language Models (LLMs) have demonstrated significant potential in medical Question Answering (QA), yet they remain prone to hallucinations and ungrounded reasoning, limiting their reliability in high-stakes clinical scenarios. While Retrieval-Augmented Generation (RAG) mitigates these issues by incorporating external knowledge, conventional single-shot retrieval often fails to resolve complex biomedical queries requiring multi-step inference. To address this, we propose Self-MedRAG, a self-reflective hybrid framework designed to mimic the iterative hypothesis-verification process of clinical reasoning. Self-MedRAG integrates a hybrid retrieval strategy, combining sparse (BM25) and dense (Contriever) retrievers via Reciprocal Rank Fusion (RRF) to maximize evidence coverage. It employs a generator to produce answers with supporting rationales, which are then assessed by a lightweight self-reflection module using Natural Language Inference (NLI) or LLM-based verification. If the rationale lacks sufficient evidentiary support, the system autonomously reformulates the query and iterates to refine the context. We evaluated Self-MedRAG on the MedQA and PubMedQA benchmarks. The results demonstrate that our hybrid retrieval approach significantly outperforms single-retriever baselines. Furthermore, the inclusion of the self-reflective loop yielded substantial gains, increasing accuracy on MedQA from 80.00% to 83.33% and on PubMedQA from 69.10% to 79.82%. These findings confirm that integrating hybrid retrieval with iterative, evidence-based self-reflection effectively reduces unsupported claims and enhances the clinical reliability of LLM-based systems. 

---
# OptiSet: Unified Optimizing Set Selection and Ranking for Retrieval-Augmented Generation 

**Authors**: Yi Jiang, Sendong Zhao, Jianbo Li, Bairui Hu, Yanrui Du, Haochun Wang, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2601.05027)  

**Abstract**: Retrieval-Augmented Generation (RAG) improves generation quality by incorporating evidence retrieved from large external corpora. However, most existing methods rely on statically selecting top-k passages based on individual relevance, which fails to exploit combinatorial gains among passages and often introduces substantial redundancy. To address this limitation, we propose OptiSet, a set-centric framework that unifies set selection and set-level ranking for RAG. OptiSet adopts an "Expand-then-Refine" paradigm: it first expands a query into multiple perspectives to enable a diverse candidate pool and then refines the candidate pool via re-selection to form a compact evidence set. We then devise a self-synthesis strategy without strong LLM supervision to derive preference labels from the set conditional utility changes of the generator, thereby identifying complementary and redundant evidence. Finally, we introduce a set-list wise training strategy that jointly optimizes set selection and set-level ranking, enabling the model to favor compact, high-gain evidence sets. Extensive experiments demonstrate that OptiSet improves performance on complex combinatorial problems and makes generation more efficient. The source code is publicly available. 

---
# T-Retriever: Tree-based Hierarchical Retrieval Augmented Generation for Textual Graphs 

**Authors**: Chunyu Wei, Huaiyu Qin, Siyuan He, Yunhai Wang, Yueguo Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.04945)  

**Abstract**: Retrieval-Augmented Generation (RAG) has significantly enhanced Large Language Models' ability to access external knowledge, yet current graph-based RAG approaches face two critical limitations in managing hierarchical information: they impose rigid layer-specific compression quotas that damage local graph structures, and they prioritize topological structure while neglecting semantic content. We introduce T-Retriever, a novel framework that reformulates attributed graph retrieval as tree-based retrieval using a semantic and structure-guided encoding tree. Our approach features two key innovations: (1) Adaptive Compression Encoding, which replaces artificial compression quotas with a global optimization strategy that preserves the graph's natural hierarchical organization, and (2) Semantic-Structural Entropy ($S^2$-Entropy), which jointly optimizes for both structural cohesion and semantic consistency when creating hierarchical partitions. Experiments across diverse graph reasoning benchmarks demonstrate that T-Retriever significantly outperforms state-of-the-art RAG methods, providing more coherent and contextually relevant responses to complex queries. 

---
# Orion-RAG: Path-Aligned Hybrid Retrieval for Graphless Data 

**Authors**: Zhen Chen, Weihao Xie, Peilin Chen, Shiqi Wang, Jianping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.04764)  

**Abstract**: Retrieval-Augmented Generation (RAG) has proven effective for knowledge synthesis, yet it encounters significant challenges in practical scenarios where data is inherently discrete and fragmented. In most environments, information is distributed across isolated files like reports and logs that lack explicit links. Standard search engines process files independently, ignoring the connections between them. Furthermore, manually building Knowledge Graphs is impractical for such vast data. To bridge this gap, we present Orion-RAG. Our core insight is simple yet effective: we do not need heavy algorithms to organize this data. Instead, we use a low-complexity strategy to extract lightweight paths that naturally link related concepts. We demonstrate that this streamlined approach suffices to transform fragmented documents into semi-structured data, enabling the system to link information across different files effectively. Extensive experiments demonstrate that Orion-RAG consistently outperforms mainstream frameworks across diverse domains, supporting real-time updates and explicit Human-in-the-Loop verification with high cost-efficiency. Experiments on FinanceBench demonstrate superior precision with a 25.2% relative improvement over strong baselines. 

---
# TrueBrief: Faithful Summarization through Small Language Models 

**Authors**: Kumud Lakara, Ruibo Shi, Fran Silavong  

**Link**: [PDF](https://arxiv.org/pdf/2601.04212)  

**Abstract**: Large language models (LLMs) have exhibited remarkable proficiency in generating high-quality text; however, their propensity for producing hallucinations poses a significant challenge for their deployment in security-critical domains. In this work, we present TrueBrief, an end-to-end framework specifically designed to enhance the faithfulness of small LLMs (SLMs) primarily for the task of text summarization through a preference-optimization paradigm. Central to our framework is a data generation module that facilitates controlled hallucination injection to generate synthetic preference data. Our work provides insights into the impact of data quality and model size on preference-based optimization, highlighting the conditions under which these methods are most effective. 

---
