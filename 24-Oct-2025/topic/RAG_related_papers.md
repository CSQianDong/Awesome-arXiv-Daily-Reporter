# Automating Iconclass: LLMs and RAG for Large-Scale Classification of Religious Woodcuts 

**Authors**: Drew B. Thomas  

**Link**: [PDF](https://arxiv.org/pdf/2510.19986)  

**Abstract**: This paper presents a novel methodology for classifying early modern religious images by using Large Language Models (LLMs) and vector databases in combination with Retrieval-Augmented Generation (RAG). The approach leverages the full-page context of book illustrations from the Holy Roman Empire, allowing the LLM to generate detailed descriptions that incorporate both visual and textual elements. These descriptions are then matched to relevant Iconclass codes through a hybrid vector search. This method achieves 87% and 92% precision at five and four levels of classification, significantly outperforming traditional image and keyword-based searches. By employing full-page descriptions and RAG, the system enhances classification accuracy, offering a powerful tool for large-scale analysis of early modern visual archives. This interdisciplinary approach demonstrates the growing potential of LLMs and RAG in advancing research within art history and digital humanities. 

---
# Balancing Fine-tuning and RAG: A Hybrid Strategy for Dynamic LLM Recommendation Updates 

**Authors**: Changping Meng, Hongyi Ling, Jianling Wang, Yifan Liu, Shuzhou Zhang, Dapeng Hong, Mingyan Gao, Onkar Dalal, Ed Chi, Lichan Hong, Haokai Lu, Ningren Han  

**Link**: [PDF](https://arxiv.org/pdf/2510.20260)  

**Abstract**: Large Language Models (LLMs) empower recommendation systems through their advanced reasoning and planning capabilities. However, the dynamic nature of user interests and content poses a significant challenge: While initial fine-tuning aligns LLMs with domain knowledge and user preferences, it fails to capture such real-time changes, necessitating robust update mechanisms. This paper investigates strategies for updating LLM-powered recommenders, focusing on the trade-offs between ongoing fine-tuning and Retrieval-Augmented Generation (RAG). Using an LLM-powered user interest exploration system as a case study, we perform a comparative analysis of these methods across dimensions like cost, agility, and knowledge incorporation. We propose a hybrid update strategy that leverages the long-term knowledge adaptation of periodic fine-tuning with the agility of low-cost RAG. We demonstrate through live A/B experiments on a billion-user platform that this hybrid approach yields statistically significant improvements in user satisfaction, offering a practical and cost-effective framework for maintaining high-quality LLM-powered recommender systems. 

---
# RAGRank: Using PageRank to Counter Poisoning in CTI LLM Pipelines 

**Authors**: Austin Jia, Avaneesh Ramesh, Zain Shamsi, Daniel Zhang, Alex Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.20768)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as the dominant architectural pattern to operationalize Large Language Model (LLM) usage in Cyber Threat Intelligence (CTI) systems. However, this design is susceptible to poisoning attacks, and previously proposed defenses can fail for CTI contexts as cyber threat information is often completely new for emerging attacks, and sophisticated threat actors can mimic legitimate formats, terminology, and stylistic conventions. To address this issue, we propose that the robustness of modern RAG defenses can be accelerated by applying source credibility algorithms on corpora, using PageRank as an example. In our experiments, we demonstrate quantitatively that our algorithm applies a lower authority score to malicious documents while promoting trusted content, using the standardized MS MARCO dataset. We also demonstrate proof-of-concept performance of our algorithm on CTI documents and feeds. 

---
# Practical Code RAG at Scale: Task-Aware Retrieval Design Choices under Compute Budgets 

**Authors**: Timur Galimzyanov, Olga Kolomyttseva, Egor Bogomolov  

**Link**: [PDF](https://arxiv.org/pdf/2510.20609)  

**Abstract**: We study retrieval design for code-focused generation tasks under realistic compute budgets. Using two complementary tasks from Long Code Arena -- code completion and bug localization -- we systematically compare retrieval configurations across various context window sizes along three axes: (i) chunking strategy, (ii) similarity scoring, and (iii) splitting granularity. (1) For PL-PL, sparse BM25 with word-level splitting is the most effective and practical, significantly outperforming dense alternatives while being an order of magnitude faster. (2) For NL-PL, proprietary dense encoders (Voyager-3 family) consistently beat sparse retrievers, however requiring 100x larger latency. (3) Optimal chunk size scales with available context: 32-64 line chunks work best at small budgets, and whole-file retrieval becomes competitive at 16000 tokens. (4) Simple line-based chunking matches syntax-aware splitting across budgets. (5) Retrieval latency varies by up to 200x across configurations; BPE-based splitting is needlessly slow, and BM25 + word splitting offers the best quality-latency trade-off. Thus, we provide evidence-based recommendations for implementing effective code-oriented RAG systems based on task requirements, model constraints, and computational efficiency. 

---
# Simple Context Compression: Mean-Pooling and Multi-Ratio Training 

**Authors**: Yair Feldman, Yoav Artzi  

**Link**: [PDF](https://arxiv.org/pdf/2510.20797)  

**Abstract**: A common strategy to reduce the computational costs of using long contexts in retrieval-augmented generation (RAG) with large language models (LLMs) is soft context compression, where the input sequence is transformed into a shorter continuous representation. We develop a lightweight and simple mean-pooling approach that consistently outperforms the widely used compression-tokens architecture, and study training the same compressor to output multiple compression ratios. We conduct extensive experiments across in-domain and out-of-domain QA datasets, as well as across model families, scales, and compression ratios. Overall, our simple mean-pooling approach achieves the strongest performance, with a relatively small drop when training for multiple compression ratios. More broadly though, across architectures and training regimes the trade-offs are more nuanced, illustrating the complex landscape of compression methods. 

---
# GlobalRAG: Enhancing Global Reasoning in Multi-hop Question Answering via Reinforcement Learning 

**Authors**: Jinchang Luo, Mingquan Cheng, Fan Wan, Ni Li, Xiaoling Xia, Shuangshuang Tian, Tingcheng Bian, Haiwei Wang, Haohuan Fu, Yan Tao  

**Link**: [PDF](https://arxiv.org/pdf/2510.20548)  

**Abstract**: Reinforcement learning has recently shown promise in improving retrieval-augmented generation (RAG). Despite these advances, its effectiveness in multi-hop question answering (QA) remains limited by two fundamental limitations: (i) global planning absence to structure multi-step reasoning, and (ii) unfaithful execution, which hinders effective query formulation and consistent use of retrieved evidence. We propose GlobalRAG, a reinforcement learning framework designed to enhance global reasoning in multi-hop QA. GlobalRAG decomposes questions into subgoals, coordinates retrieval with reasoning, and refines evidence iteratively. To guide this process, we introduce Planning Quality Reward and SubGoal Completion Reward, which encourage coherent planning and reliable subgoal execution. In addition, a progressive weight annealing strategy balances process-oriented and outcome-based objectives. Extensive experiments on both in-domain and out-of-domain benchmarks demonstrate that GlobalRAG significantly outperforms strong baselines while using only 8k training data (42% of the training data used by strong baselines), achieving average improvements of 14.2% in both EM and F1. 

---
# Hierarchical Sequence Iteration for Heterogeneous Question Answering 

**Authors**: Ruiyi Yang, Hao Xue, Imran Razzak, Hakim Hacid, Flora D. Salim  

**Link**: [PDF](https://arxiv.org/pdf/2510.20505)  

**Abstract**: Retrieval-augmented generation (RAG) remains brittle on multi-step questions and heterogeneous evidence sources, trading accuracy against latency and token/tool budgets. This paper introducesHierarchical Sequence (HSEQ) Iteration for Heterogeneous Question Answering, a unified framework that (i) linearize documents, tables, and knowledge graphs into a reversible hierarchical sequence with lightweight structural tags, and (ii) perform structure-aware iteration to collect just-enough evidence before answer synthesis. A Head Agent provides guidance that leads retrieval, while an Iteration Agent selects and expands HSeq via structure-respecting actions (e.g., parent/child hops, table row/column neighbors, KG relations); Finally the head agent composes canonicalized evidence to genearte the final answer, with an optional refinement loop to resolve detected contradictions. Experiments on HotpotQA (text), HybridQA/TAT-QA (table+text), and MetaQA (KG) show consistent EM/F1 gains over strong single-pass, multi-hop, and agentic RAG baselines with high efficiency. Besides, HSEQ exhibits three key advantages: (1) a format-agnostic unification that enables a single policy to operate across text, tables, and KGs without per-dataset specialization; (2) guided, budget-aware iteration that reduces unnecessary hops, tool calls, and tokens while preserving accuracy; and (3) evidence canonicalization for reliable QA, improving answers consistency and auditability. 

---
# RAG-Stack: Co-Optimizing RAG Quality and Performance From the Vector Database Perspective 

**Authors**: Wenqi Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2510.20296)  

**Abstract**: Retrieval-augmented generation (RAG) has emerged as one of the most prominent applications of vector databases. By integrating documents retrieved from a database into the prompt of a large language model (LLM), RAG enables more reliable and informative content generation. While there has been extensive research on vector databases, many open research problems remain once they are considered in the wider context of end-to-end RAG pipelines. One practical yet challenging problem is how to jointly optimize both system performance and generation quality in RAG, which is significantly more complex than it appears due to the numerous knobs on both the algorithmic side (spanning models and databases) and the systems side (from software to hardware). In this paper, we present RAG-Stack, a three-pillar blueprint for quality-performance co-optimization in RAG systems. RAG-Stack comprises: (1) RAG-IR, an intermediate representation that serves as an abstraction layer to decouple quality and performance aspects; (2) RAG-CM, a cost model for estimating system performance given an RAG-IR; and (3) RAG-PE, a plan exploration algorithm that searches for high-quality, high-performance RAG configurations. We believe this three-pillar blueprint will become the de facto paradigm for RAG quality-performance co-optimization in the years to come. 

---
