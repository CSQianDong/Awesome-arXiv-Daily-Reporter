# Unifying Ranking and Generation in Query Auto-Completion via Retrieval-Augmented Generation and Multi-Objective Alignment 

**Authors**: Kai Yuan, Anthony Zheng, Jia Hu, Divyanshu Sheth, Hemanth Velaga, Kylee Kim, Matteo Guarrera, Besim Avci, Xuetao Yin, Rajyashree Mukherjee, Sean Suchter  

**Link**: [PDF](https://arxiv.org/pdf/2602.01023)  

**Abstract**: Query Auto-Completion (QAC) suggests query completions as users type, helping them articulate intent and reach results more efficiently. Existing approaches face fundamental challenges: traditional retrieve-and-rank pipelines have limited long-tail coverage and require extensive feature engineering, while recent generative methods suffer from hallucination and safety risks. We present a unified framework that reformulates QAC as end-to-end list generation through Retrieval-Augmented Generation (RAG) and multi-objective Direct Preference Optimization (DPO). Our approach combines three key innovations: (1) reformulating QAC as end-to-end list generation with multi-objective optimization; (2) defining and deploying a suite of rule-based, model-based, and LLM-as-judge verifiers for QAC, and using them in a comprehensive methodology that combines RAG, multi-objective DPO, and iterative critique-revision for high-quality synthetic data; (3) a hybrid serving architecture enabling efficient production deployment under strict latency constraints. Evaluation on a large-scale commercial search platform demonstrates substantial improvements: offline metrics show gains across all dimensions, human evaluation yields +0.40 to +0.69 preference scores, and a controlled online experiment achieves 5.44\% reduction in keystrokes and 3.46\% increase in suggestion adoption, validating that unified generation with RAG and multi-objective alignment provides an effective solution for production QAC. This work represents a paradigm shift to end-to-end generation powered by large language models, RAG, and multi-objective alignment, establishing a production-validated framework that can benefit the broader search and recommendation industry. 

---
# AI-assisted Protocol Information Extraction For Improved Accuracy and Efficiency in Clinical Trial Workflows 

**Authors**: Ramtin Babaeipour, François Charest, Madison Wright  

**Link**: [PDF](https://arxiv.org/pdf/2602.00052)  

**Abstract**: Increasing clinical trial protocol complexity, amendments, and challenges around knowledge management create significant burden for trial teams. Structuring protocol content into standard formats has the potential to improve efficiency, support documentation quality, and strengthen compliance. We evaluate an Artificial Intelligence (AI) system using generative LLMs with Retrieval-Augmented Generation (RAG) for automated clinical trial protocol information extraction. We compare the extraction accuracy of our clinical-trial-specific RAG process against that of publicly available (standalone) LLMs. We also assess the operational impact of AI-assistance on simulated extraction CRC workflows. Our RAG process was measured as more accurate (87.8%) than standalone LLMs with fine-tuned prompts (62.6%) against expert-supported reference annotations. In the simulated extraction workflows, AI-assisted tasks were completed 40% faster, rated as less cognitively demanding and strongly preferred by users. While expert oversight remains essential, this suggests that AI-assisted extraction can enable protocol intelligence at scale, motivating the integration of similar methodologies into real world clinical workflows to further validate its impact on feasibility, study start-up, and post-activation monitoring. 

---
# RAGRouter-Bench: A Dataset and Benchmark for Adaptive RAG Routing 

**Authors**: Ziqi Wang, Xi Zhu, Shuhang Lin, Haochen Xue, Minghao Guo, Yongfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.00296)  

**Abstract**: Retrieval-Augmented Generation (RAG) has become a core paradigm for grounding large language models with external knowledge. Despite extensive efforts exploring diverse retrieval strategies, existing studies predominantly focus on query-side complexity or isolated method improvements, lacking a systematic understanding of how RAG paradigms behave across different query-corpus contexts and effectiveness-efficiency trade-offs. In this work, we introduce RAGRouter-Bench, the first dataset and benchmark designed for adaptive RAG routing. RAGRouter-Bench revisits retrieval from a query-corpus compatibility perspective and standardizes five representative RAG paradigms for systematic evaluation across 7,727 queries and 21,460 documents spanning diverse domains. The benchmark incorporates three canonical query types together with fine-grained semantic and structural corpus metrics, as well as a unified evaluation for both generation quality and resource consumption. Experiments with DeepSeek-V3 and LLaMA-3.1-8B demonstrate that no single RAG paradigm is universally optimal, that paradigm applicability is strongly shaped by query-corpus interactions, and that increased advanced mechanism does not necessarily yield better effectiveness-efficiency trade-offs. These findings underscore the necessity of routing-aware evaluation and establish a foundation for adaptive, interpretable, and generalizable next-generation RAG systems. 

---
# SPARC-RAG: Adaptive Sequential-Parallel Scaling with Context Management for Retrieval-Augmented Generation 

**Authors**: Yuxin Yang, Gangda Deng, Ömer Faruk Akgül, Nima Chitsazan, Yash Govilkar, Akasha Tigalappanavara, Shi-Xiong Zhang, Sambit Sahu, Viktor Prasanna  

**Link**: [PDF](https://arxiv.org/pdf/2602.00083)  

**Abstract**: Retrieval-Augmented Generation (RAG) grounds large language model outputs in external evidence, but remains challenged on multi-hop question answering that requires long reasoning. Recent works scale RAG at inference time along two complementary dimensions: sequential depth for iterative refinement and parallel width for coverage expansion. However, naive scaling causes context contamination and scaling inefficiency, leading to diminishing or negative returns despite increased computation. To address these limitations, we propose SPARC-RAG, a multi-agent framework that coordinates sequential and parallel inference-time scaling under a unified context management mechanism. SPARC-RAG employs specialized agents that maintain a shared global context and provide explicit control over the scaling process. It generates targeted, complementary sub-queries for each branch to enable diverse parallel exploration, and explicitly regulates exiting decisions based on answer correctness and evidence grounding. To optimize scaling behavior, we further introduce a lightweight fine-tuning method with process-level verifiable preferences, which improves the efficiency of sequential scaling and effectiveness of parallel scaling. Across single- and multi-hop QA benchmarks, SPARC-RAG consistently outperforms previous RAG baselines, yielding an average +6.2 F1 improvement under lower inference cost. 

---
# Towards AI Evaluation in Domain-Specific RAG Systems: The AgriHubi Case Study 

**Authors**: Md. Toufique Hasan, Ayman Asad Khan, Mika Saari, Vaishnavi Bankhele, Pekka Abrahamsson  

**Link**: [PDF](https://arxiv.org/pdf/2602.02208)  

**Abstract**: Large language models show promise for knowledge-intensive domains, yet their use in agriculture is constrained by weak grounding, English-centric training data, and limited real-world evaluation. These issues are amplified for low-resource languages, where high-quality domain documentation exists but remains difficult to access through general-purpose models. This paper presents AgriHubi, a domain-adapted retrieval-augmented generation (RAG) system for Finnish-language agricultural decision support. AgriHubi integrates Finnish agricultural documents with open PORO family models and combines explicit source grounding with user feedback to support iterative refinement. Developed over eight iterations and evaluated through two user studies, the system shows clear gains in answer completeness, linguistic accuracy, and perceived reliability. The results also reveal practical trade-offs between response quality and latency when deploying larger models. This study provides empirical guidance for designing and evaluating domain-specific RAG systems in low-resource language settings. 

---
# Culinary Crossroads: A RAG Framework for Enhancing Diversity in Cross-Cultural Recipe Adaptation 

**Authors**: Tianyi Hu, Andrea Morales-Garzón, Jingyi Zheng, Maria Maistro, Daniel Hershcovich  

**Link**: [PDF](https://arxiv.org/pdf/2507.21934)  

**Abstract**: In cross-cultural recipe adaptation, the goal is not only to ensure cultural appropriateness and retain the original dish's essence, but also to provide diverse options for various dietary needs and preferences. Retrieval Augmented Generation (RAG) is a promising approach, combining the retrieval of real recipes from the target cuisine for cultural adaptability with large language models (LLMs) for relevance. However, it remains unclear whether RAG can generate diverse adaptation results. Our analysis shows that RAG tends to overly rely on a limited portion of the context across generations, failing to produce diverse outputs even when provided with varied contextual inputs. This reveals a key limitation of RAG in creative tasks with multiple valid answers: it fails to leverage contextual diversity for generating varied responses. To address this issue, we propose CARRIAGE, a plug-and-play RAG framework for cross-cultural recipe adaptation that enhances diversity in both retrieval and context organization. To our knowledge, this is the first RAG framework that explicitly aims to generate highly diverse outputs to accommodate multiple user preferences. Our experiments show that CARRIAGE achieves Pareto efficiency in terms of diversity and quality of recipe adaptation compared to closed-book LLMs. 

---
# Unlocking Electronic Health Records: A Hybrid Graph RAG Approach to Safe Clinical AI for Patient QA 

**Authors**: Samuel Thio, Matthew Lewis, Spiros Denaxas, Richard JB Dobson  

**Link**: [PDF](https://arxiv.org/pdf/2602.00009)  

**Abstract**: Electronic health record (EHR) systems present clinicians with vast repositories of clinical information, creating a significant cognitive burden where critical details are easily overlooked. While Large Language Models (LLMs) offer transformative potential for data processing, they face significant limitations in clinical settings, particularly regarding context grounding and hallucinations. Current solutions typically isolate retrieval methods focusing either on structured data (SQL/Cypher) or unstructured semantic search but fail to integrate both simultaneously. This work presents MediGRAF (Medical Graph Retrieval Augmented Framework), a novel hybrid Graph RAG system that bridges this gap. By uniquely combining Neo4j Text2Cypher capabilities for structured relationship traversal with vector embeddings for unstructured narrative retrieval, MediGRAF enables natural language querying of the complete patient journey. Using 10 patients from the MIMIC-IV dataset (generating 5,973 nodes and 5,963 relationships), we generated enough nodes and data for patient level question answering (QA), and we evaluated this architecture across varying query complexities. The system demonstrated 100\% recall for factual queries which means all relevant information was retrieved and in the output, while complex inference tasks achieved a mean expert quality score of 4.25/5 with zero safety violations. These results demonstrate that hybrid graph-grounding significantly advances clinical information retrieval, offering a safer, more comprehensive alternative to standard LLM deployments. 

---
# dziribot: rag based intelligent conversational agent for algerian arabic dialect 

**Authors**: El Batoul Bechiri, Dihia Lanasri  

**Link**: [PDF](https://arxiv.org/pdf/2602.02270)  

**Abstract**: The rapid digitalization of customer service has intensified the demand for conversational agents capable of providing accurate and natural interactions. In the Algerian context, this is complicated by the linguistic complexity of Darja, a dialect characterized by non-standardized orthography, extensive code-switching with French, and the simultaneous use of Arabic and Latin (Arabizi) scripts. This paper introduces DziriBOT, a hybrid intelligent conversational agent specifically engineered to overcome these challenges. We propose a multi-layered architecture that integrates specialized Natural Language Understanding (NLU) with Retrieval-Augmented Generation (RAG), allowing for both structured service flows and dynamic, knowledge-intensive responses grounded in curated enterprise documentation. To address the low-resource nature of Darja, we systematically evaluate three distinct approaches: a sparse-feature Rasa pipeline, classical machine learning baselines, and transformer-based fine-tuning. Our experimental results demonstrate that the fine-tuned DziriBERT model achieves state-of-the-art performance. These results significantly outperform traditional baselines, particularly in handling orthographic noise and rare intents. Ultimately, DziriBOT provides a robust, scalable solution that bridges the gap between formal language models and the linguistic realities of Algerian users, offering a blueprint for dialect-aware automation in the regional market. 

---
# Breaking the Static Graph: Context-Aware Traversal for Robust Retrieval-Augmented Generation 

**Authors**: Kwun Hang Lau, Fangyuan Zhang, Boyu Ruan, Yingli Zhou, Qintian Guo, Ruiyuan Zhang, Xiaofang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2602.01965)  

**Abstract**: Recent advances in Retrieval-Augmented Generation (RAG) have shifted from simple vector similarity to structure-aware approaches like HippoRAG, which leverage Knowledge Graphs (KGs) and Personalized PageRank (PPR) to capture multi-hop dependencies. However, these methods suffer from a "Static Graph Fallacy": they rely on fixed transition probabilities determined during indexing. This rigidity ignores the query-dependent nature of edge relevance, causing semantic drift where random walks are diverted into high-degree "hub" nodes before reaching critical downstream evidence. Consequently, models often achieve high partial recall but fail to retrieve the complete evidence chain required for multi-hop queries. To address this, we propose CatRAG, Context-Aware Traversal for robust RAG, a framework that builds on the HippoRAG 2 architecture and transforms the static KG into a query-adaptive navigation structure. We introduce a multi-faceted framework to steer the random walk: (1) Symbolic Anchoring, which injects weak entity constraints to regularize the random walk; (2) Query-Aware Dynamic Edge Weighting, which dynamically modulates graph structure, to prune irrelevant paths while amplifying those aligned with the query's intent; and (3) Key-Fact Passage Weight Enhancement, a cost-efficient bias that structurally anchors the random walk to likely evidence. Experiments across four multi-hop benchmarks demonstrate that CatRAG consistently outperforms state of the art baselines. Our analysis reveals that while standard Recall metrics show modest gains, CatRAG achieves substantial improvements in reasoning completeness, the capacity to recover the entire evidence path without gaps. These results reveal that our approach effectively bridges the gap between retrieving partial context and enabling fully grounded reasoning. Resources are available at this https URL. 

---
# WildGraphBench: Benchmarking GraphRAG with Wild-Source Corpora 

**Authors**: Pengyu Wang, Benfeng Xu, Licheng Zhang, Shaohan Wang, Mingxuan Du, Chiwei Zhu, Zhendong Mao  

**Link**: [PDF](https://arxiv.org/pdf/2602.02053)  

**Abstract**: Graph-based Retrieval-Augmented Generation (GraphRAG) organizes external knowledge as a hierarchical graph, enabling efficient retrieval and aggregation of scattered evidence across multiple documents. However, many existing benchmarks for GraphRAG rely on short, curated passages as external knowledge, failing to adequately evaluate systems in realistic settings involving long contexts and large-scale heterogeneous documents. To bridge this gap, we introduce WildGraphBench, a benchmark designed to assess GraphRAG performance in the wild. We leverage Wikipedia's unique structure, where cohesive narratives are grounded in long and heterogeneous external reference documents, to construct a benchmark reflecting real-word scenarios. Specifically, we sample articles across 12 top-level topics, using their external references as the retrieval corpus and citation-linked statements as ground truth, resulting in 1,100 questions spanning three levels of complexity: single-fact QA, multi-fact QA, and section-level summarization. Experiments across multiple baselines reveal that current GraphRAG pipelines help on multi-fact aggregation when evidence comes from a moderate number of sources, but this aggregation paradigm may overemphasize high-level statements at the expense of fine-grained details, leading to weaker performance on summarization tasks. Project page:this https URL. 

---
# DIVERGE: Diversity-Enhanced RAG for Open-Ended Information Seeking 

**Authors**: Tianyi Hu, Niket Tandon, Akhil Arora  

**Link**: [PDF](https://arxiv.org/pdf/2602.00238)  

**Abstract**: Existing retrieval-augmented generation (RAG) systems are primarily designed under the assumption that each query has a single correct answer. This overlooks common information-seeking scenarios with multiple plausible answers, where diversity is essential to avoid collapsing to a single dominant response, thereby constraining creativity and compromising fair and inclusive information access. Our analysis reveals a commonly overlooked limitation of standard RAG systems: they underutilize retrieved context diversity, such that increasing retrieval diversity alone does not yield diverse generations. To address this limitation, we propose DIVERGE, a plug-and-play agentic RAG framework with novel reflection-guided generation and memory-augmented iterative refinement, which promotes diverse viewpoints while preserving answer quality. We introduce novel metrics tailored to evaluating the diversity-quality trade-off in open-ended questions, and show that they correlate well with human judgments. We demonstrate that DIVERGE achieves the best diversity-quality trade-off compared to competitive baselines and previous state-of-the-art methods on the real-world Infinity-Chat dataset, substantially improving diversity while maintaining quality. More broadly, our results reveal a systematic limitation of current LLM-based systems for open-ended information-seeking and show that explicitly modeling diversity can mitigate it. Our code is available at: this https URL 

---
# When RAG Hurts: Diagnosing and Mitigating Attention Distraction in Retrieval-Augmented LVLMs 

**Authors**: Beidi Zhao, Wenlong Deng, Xinting Liao, Yushu Li, Nazim Shaikh, Yao Nie, Xiaoxiao Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.00344)  

**Abstract**: While Retrieval-Augmented Generation (RAG) is one of the dominant paradigms for enhancing Large Vision-Language Models (LVLMs) on knowledge-based VQA tasks, recent work attributes RAG failures to insufficient attention towards the retrieved context, proposing to reduce the attention allocated to image tokens. In this work, we identify a distinct failure mode that previous study overlooked: Attention Distraction (AD). When the retrieved context is sufficient (highly relevant or including the correct answer), the retrieved text suppresses the visual attention globally, and the attention on image tokens shifts away from question-relevant regions. This leads to failures on questions the model could originally answer correctly without the retrieved text. To mitigate this issue, we propose MAD-RAG, a training-free intervention that decouples visual grounding from context integration through a dual-question formulation, combined with attention mixing to preserve image-conditioned evidence. Extensive experiments on OK-VQA, E-VQA, and InfoSeek demonstrate that MAD-RAG consistently outperforms existing baselines across different model families, yielding absolute gains of up to 4.76%, 9.20%, and 6.18% over the vanilla RAG baseline. Notably, MAD-RAG rectifies up to 74.68% of failure cases with negligible computational overhead. 

---
# SOPRAG: Multi-view Graph Experts Retrieval for Industrial Standard Operating Procedures 

**Authors**: Liangtao Lin, Zhaomeng Zhu, Tianwei Zhang, Yonggang Wen  

**Link**: [PDF](https://arxiv.org/pdf/2602.01858)  

**Abstract**: Standard Operating Procedures (SOPs) are essential for ensuring operational safety and consistency in industrial environments. However, retrieving and following these procedures presents unique challenges, such as rigid proprietary structures, condition-dependent relevance, and actionable execution requirement, which standard semantic-driven Retrieval-Augmented Generation (RAG) paradigms fail to address. Inspired by the Mixture-of-Experts (MoE) paradigm, we propose SOPRAG, a novel framework specifically designed to address the above pain points in SOP retrieval. SOPRAG replaces flat chunking with specialized Entity, Causal, and Flow graph experts to resolve industrial structural and logical complexities. To optimize and coordinate these experts, we propose a Procedure Card layer that prunes the search space to eliminate computational noise, and an LLM-Guided gating mechanism that dynamically weights these experts to align retrieval with operator intent. To address the scarcity of domain-specific data, we also introduce an automated, multi-agent workflow for benchmark construction. Extensive experiments across four industrial domains demonstrate that SOPRAG significantly outperforms strong lexical, dense, and graph-based RAG baselines in both retrieval accuracy and response utility, achieving perfect execution scores in real-world critical tasks. 

---
# Aggregation Queries over Unstructured Text: Benchmark and Agentic Method 

**Authors**: Haojia Zhu, Qinyuan Xu, Haoyu Li, Yuxi Liu, Hanchen Qiu, Jiaoyan Chen, Jiahui Jin  

**Link**: [PDF](https://arxiv.org/pdf/2602.01355)  

**Abstract**: Aggregation query over free text is a long-standing yet underexplored problem. Unlike ordinary question answering, aggregate queries require exhaustive evidence collection and systems are required to "find all," not merely "find one." Existing paradigms such as Text-to-SQL and Retrieval-Augmented Generation fail to achieve this completeness. In this work, we formalize entity-level aggregation querying over text in a corpus-bounded setting with strict completeness requirement. To enable principled evaluation, we introduce AGGBench, a benchmark designed to evaluate completeness-oriented aggregation under realistic large-scale corpus. To accompany the benchmark, we propose DFA (Disambiguation--Filtering--Aggregation), a modular agentic baseline that decomposes aggregation querying into interpretable stages and exposes key failure modes related to ambiguity, filtering, and aggregation. Empirical results show that DFA consistently improves aggregation evidence coverage over strong RAG and agentic baselines. The data and code are available in this https URL. 

---
# RAG-GNN: Integrating Retrieved Knowledge with Graph Neural Networks for Precision Medicine 

**Authors**: Hasi Hays, William J. Richardson  

**Link**: [PDF](https://arxiv.org/pdf/2602.00586)  

**Abstract**: Network topology excels at structural predictions but fails to capture functional semantics encoded in biomedical literature. We present a retrieval-augmented generation (RAG) embedding framework that integrates graph neural network representations with dynamically retrieved literature-derived knowledge through contrastive learning. Benchmarking against ten embedding methods reveals task-specific complementarity: topology-focused methods achieve near-perfect link prediction (GCN: 0.983 AUROC), while RAG-GNN is the only method achieving positive silhouette scores for functional clustering (0.001 vs. negative scores for all baselines). Information-theoretic decomposition shows network topology contributes 77.3% of predictive information, while retrieved documents provide 8.6% unique information. Applied to cancer signaling networks (379 proteins, 3,498 interactions), the framework identifies DDR1 as a therapeutic target based on retrieved evidence of synthetic lethality with KRAS mutations. These results establish that topology-only and retrieval-augmented approaches serve complementary purposes: structural prediction tasks are solved by network topology alone, while functional interpretation uniquely benefits from retrieved knowledge. 

---
# Beyond Static Question Banks: Dynamic Knowledge Expansion via LLM-Automated Graph Construction and Adaptive Generation 

**Authors**: Yingquan Wang, Tianyu Wei, Qinsi Li, Li Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2602.00020)  

**Abstract**: Personalized education systems increasingly rely on structured knowledge representations to support adaptive learning and question generation. However, existing approaches face two fundamental limitations. First, constructing and maintaining knowledge graphs for educational content largely depends on manual curation, resulting in high cost and poor scalability. Second, most personalized education systems lack effective support for state-aware and systematic reasoning over learners' knowledge, and therefore rely on static question banks with limited adaptability. To address these challenges, this paper proposes a Generative GraphRAG framework for automated knowledge modeling and personalized exercise generation. It consists of two core modules. The first module, Automated Hierarchical Knowledge Graph Constructor (Auto-HKG), leverages LLMs to automatically construct hierarchical knowledge graphs that capture structured concepts and their semantic relations from educational resources. The second module, Cognitive GraphRAG (CG-RAG), performs graph-based reasoning over a learner mastery graph and combines it with retrieval-augmented generation to produce personalized exercises that adapt to individual learning states. The proposed framework has been deployed in real-world educational scenarios, where it receives favorable user feedback, suggesting its potential to support practical personalized education systems. 

---
