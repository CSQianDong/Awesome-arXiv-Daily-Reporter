# Document-as-Image Representations Fall Short for Scientific Retrieval 

**Authors**: Ghazal Khalighinejad, Raghuveer Thirukovalluru, Alexander H. Oh, Bhuwan Dhingra  

**Link**: [PDF](https://arxiv.org/pdf/2604.18508)  

**Abstract**: Many recent document embedding models are trained on document-as-image representations, embedding rendered pages as images rather than the underlying source. Meanwhile, existing benchmarks for scientific document retrieval, such as ArXivQA and ViDoRe, treat documents as images of pages, implicitly favoring such representations. In this work, we argue that this paradigm is not well-suited for text-rich multimodal scientific documents, where critical evidence is distributed across structured sources, including text, tables, and figures. To study this setting, we introduce ArXivDoc, a new benchmark constructed from the underlying LaTeX sources of scientific papers. Unlike PDF or image-based representations, LaTeX provides direct access to structured elements (e.g., sections, tables, figures, equations), enabling controlled query construction grounded in specific evidence types. We systematically compare text-only, image-based, and multimodal representations across both single-vector and multi-vector retrieval models. Our results show that: (1) document-as-image representations are consistently suboptimal, especially as document length increases; (2) text-based representations are most effective, even for figure-based queries, by leveraging captions and surrounding context; and (3) interleaved text+image representations outperform document-as-image approaches without requiring specialized training. 

---
# Context-Aware Search and Retrieval Under Token Erasure 

**Authors**: Sara Ghasvarianjahromi, Joshua Barr, Yauhen Yakimenka, Jörg Kliewer  

**Link**: [PDF](https://arxiv.org/pdf/2604.18424)  

**Abstract**: This paper introduces and analyzes a search and retrieval model for RAG-like systems under {token} erasures. We provide an information-theoretic analysis of remote document retrieval when query representations are only partially preserved. The query is represented using term-frequency-based features, and semantically adaptive redundancy is assigned according to feature importance. Retrieval is performed using TF-IDF-weighted similarity. We characterize the retrieval error probability by showing that the vector of similarity margins converges to a multivariate Gaussian distribution, yielding an explicit approximation and computable upper bounds. Numerical results support the analysis, while a separate data-driven evaluation using embedding-based retrieval on real-world data shows that the same importance-aware redundancy principles extend to modern retrieval pipelines. Overall, the results show that assigning higher redundancy to semantically important query features improves retrieval reliability. 

---
# Balanced Co-Clustering of Users and Items for Embedding Table Compression in Recommender Systems 

**Authors**: Runhao Jiang, Renchi Yang, Donghao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.18351)  

**Abstract**: Recommender systems have advanced markedly over the past decade by transforming each user/item into a dense embedding vector with deep learning models. At industrial scale, embedding tables constituted by such vectors of all users/items demand a vast amount of parameters and impose heavy compute and memory overhead during training and inference, hindering model deployment under resource constraints. Existing solutions towards embedding compression either suffer from severely compromised recommendation accuracy or incur considerable computational costs.
To mitigate these issues, this paper presents BACO, a fast and effective framework for compressing embedding tables. Unlike traditional ID hashing, BACO is built on the idea of exploiting collaborative signals in user-item interactions for user and item groupings, such that similar users/items share the same embeddings in the codebook. Specifically, we formulate a balanced co-clustering objective that maximizes intra-cluster connectivity while enforcing cluster-volume balance, and unify canonical graph clustering techniques into the framework through rigorous theoretical analyses. To produce effective groupings while averting codebook collapse, BACO instantiates this framework with a principled weighting scheme for users and items, an efficient label propagation solver, as well as secondary user clusters. Our extensive experiments comparing BACO against full models and 18 baselines over benchmark datasets demonstrate that BACO cuts embedding parameters by over 75% with a drop of at most 1.85% in recall, while surpassing the strongest baselines by being up to 346X faster. 

---
# DocQAC: Adaptive Trie-Guided Decoding for Effective In-Document Query Auto-Completion 

**Authors**: Rahul Mehta, Kavin R V, Indrajit Pal, Tushar Abhishek, Pawan Goyal, Manish Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2604.18257)  

**Abstract**: Query auto-completion (QAC) has been widely studied in the context of web search, yet remains underexplored for in-document search, which we term DocQAC. DocQAC aims to enhance search productivity within long documents by helping users craft faster, more precise queries, even for complex or hard-to-spell terms. While global historical queries are available to both WebQAC and DocQAC, DocQAC uniquely accesses document-specific context, including the current document's content and its specific history of user query interactions.
To address this setting, we propose a novel adaptive trie-guided decoding framework that uses user query prefixes to softly steer language models toward high-quality completions. Our approach introduces an adaptive penalty mechanism with tunable hyperparameters, enabling a principled trade-off between model confidence and trie-based guidance. To efficiently incorporate document context, we explore retrieval-augmented generation (RAG) and lightweight contextual document signals such as titles, keyphrases, and summaries.
When applied to encoder-decoder models like T5 and BART, our trie-guided framework outperforms strong baselines and even surpasses much larger instruction-tuned models such as LLaMA-3 and Phi-3 on seen queries across both seen and unseen documents. This demonstrates its practicality for real-world DocQAC deployments, where efficiency and scalability are critical. We evaluate our method on a newly introduced DocQAC benchmark derived from ORCAS, enriched with query-document pairs. We make both the DocQAC dataset (this https URL) and code (this https URL) publicly available. 

---
# Evaluating Multi-Hop Reasoning in RAG Systems: A Comparison of LLM-Based Retriever Evaluation Strategies 

**Authors**: Lorenz Brehme, Thomas Ströhle, Ruth Breu  

**Link**: [PDF](https://arxiv.org/pdf/2604.18234)  

**Abstract**: Retrieval-augmented generation (RAG) enhances large language models (LLMs) with external knowledge to answer questions more accurately.
However, research on evaluating RAG systems-particularly the retriever component-remains limited, as most existing work focuses on single-context retrieval rather than multi-hop queries, where individual contexts may appear irrelevant in isolation but are essential when combined. In this research, we use the HotPotQA, MuSiQue, and SQuAD datasets to simulate a RAG system and compare three LLM-as-judge evaluation strategies, including our proposed Context-Aware Retriever Evaluation (CARE). Our goal is to better understand how multi-hop reasoning can be most effectively evaluated in RAG systems.
Experiments with LLMs from OpenAI, Meta, and Google demonstrate that CARE consistently outperforms existing methods for evaluating multi-hop reasoning in RAG systems. The performance gains are most pronounced in models with larger parameter counts and longer context windows, while single-hop queries show minimal sensitivity to context-aware evaluation. Overall, the results highlight the critical role of context-aware evaluation in improving the reliability and accuracy of retrieval-augmented generation systems, particularly in complex query scenarios. To ensure reproducibility, we provide the complete data of our experiments at this https URL. 

---
# Multi-LLM Token Filtering and Routing for Sequential Recommendation 

**Authors**: Wuhan Chen, Min Gao, Xin Xia, Zongwei Wang, Wentao Li, Shane Culpepper  

**Link**: [PDF](https://arxiv.org/pdf/2604.18200)  

**Abstract**: Large language models (LLMs) have recently shown promise in recommendation by providing rich semantic knowledge. While most existing approaches rely on external textual corpora to align LLMs with recommender systems, we revisit a more fundamental yet underexplored question: Can recommendation benefit from LLM token embeddings alone without textual input? Through a systematic empirical study, we show that directly injecting token embeddings from a single LLM into sequential recommenders leads to unstable or limited gains, due to semantic misalignment, insufficient task adaptation, and the restricted coverage of individual LLMs. To address these challenges, we propose MLTFR, a Multi-LLM Token Filtering and Routing framework for corpus-free sequential recommendation. MLTFR follows an interaction-guided LLM knowledge integration paradigm, where task-relevant token embeddings are selected via user-guided token filtering to suppress noisy and irrelevant vocabulary signals. To overcome the limitations of single-LLM representations, MLTFR integrates multiple LLM token spaces through a Mixture-of-Experts architecture, with a Fisher-weighted semantic consensus expert to balance heterogeneous experts and prevent domination during training. By jointly filtering informative tokens and aggregating complementary semantic knowledge across multiple LLMs, MLTFR enables stable and effective utilization of LLM token embeddings without textual inputs or backbone modification. Extensive experiments demonstrate that MLTFR consistently outperforms state-of-the-art sequential recommendation baselines and existing alignment methods. Our code is available at: this https URL. 

---
# Modular Representation Compression: Adapting LLMs for Efficient and Effective Recommendations 

**Authors**: Yunjia Xi, Menghui Zhu, Jianghao Lin, Bo Chen, Ruiming Tang, Yong Yu, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.18146)  

**Abstract**: Recently, large language models (LLMs) have advanced recommendation systems (RSs), and recent works have begun to explore how to integrate LLMs into industrial RSs. While most approaches deploy LLMs offline to generate and pre-cache augmented representations for RSs, high-dimensional representations from LLMs introduce substantial storage and computational costs. Thus, it is crucial to compress LLM representations effectively. However, we identify a counterintuitive phenomenon during representation compression: Mid-layer Representation Advantage (MRA), where representations from middle layers of LLMs outperform those from final layers in recommendation tasks. This degraded final layer renders existing compression methods, which typically compress on the final layer, suboptimal. We interpret this based on modularity theory that LLMs develop spontaneous internal functional modularity and force the final layer to specialize in the proxy training task. Thus, we propose \underline{M}odul\underline{a}r \underline{R}epresentation \underline{C}ompression (MARC) to explicitly control the modularity of LLMs. First, Modular Adjustment explicitly introduces compression and task adaptation modules, enabling the LLM to operate strictly as a representation-learning module. Next, to ground each module to its specific task, Modular Task Decoupling uses information constraints and different network structures to decouple tasks. Extensive experiments validate that MARC addresses MRA and produces efficient representations. Notably, MARC achieved a 2.82% eCPM lift in an online A/B test within a large-scale commercial search advertising scenario. 

---
# Architecture Matters More Than Scale: A Comparative Study of Retrieval and Memory Augmentation for Financial QA Under SME Compute Constraints 

**Authors**: Jianan Liu, Jing Yang, Xianyou Li, Weiran Yan, Yichao Wu, Penghao Liang, Mengwei Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2604.17979)  

**Abstract**: The rapid adoption of artificial intelligence (AI) and large language models (LLMs) is transforming financial analytics by enabling natural language interfaces for reporting, decision support, and automated reasoning. However, limited empirical understanding exists regarding how different LLM-based reasoning architectures perform across realistic financial workflows, particularly under the cost, accuracy, and compliance constraints faced by small and medium-sized enterprises (SMEs). SMEs typically operate within severe infrastructure constraints, lacking cloud GPU budgets, dedicated AI teams, and API-scale inference capacity, making architectural efficiency a first-class concern. To ensure practical relevance, we introduce an explicit SME-constrained evaluation setting in which all experiments are conducted using a locally hosted 8B-parameter instruction-tuned model without cloud-scale infrastructure. This design isolates the impact of architectural choices within a realistic deployment environment. We systematically compare four reasoning architectures: baseline LLM, retrieval-augmented generation (RAG), structured long-term memory, and memory-augmented conversational reasoning across both FinQA and ConvFinQA benchmarks. Results reveal a consistent architectural inversion: structured memory improves precision in deterministic, operand-explicit tasks, while retrieval-based approaches outperform memory-centric methods in conversational, reference-implicit settings. Based on these findings, we propose a hybrid deployment framework that dynamically selects reasoning strategies to balance numerical accuracy, auditability, and infrastructure efficiency, providing a practical pathway for financial AI adoption in resource-constrained environments. 

---
# Bayesian Active Learning with Gaussian Processes Guided by LLM Relevance Scoring for Dense Passage Retrieval 

**Authors**: Junyoung Kim, Anton Korikov, Jiazhou Liang, Justin Cui, Yifan Simon Liu, Qianfeng Wen, Mark Zhao, Scott Sanner  

**Link**: [PDF](https://arxiv.org/pdf/2604.17906)  

**Abstract**: While Large Language Models (LLMs) exhibit exceptional zero-shot relevance modeling, their high computational cost necessitates framing passage retrieval as a budget-constrained global optimization problem. Existing approaches passively rely on first-stage dense retrievers, which leads to two limitations: (1) failing to retrieve relevant passages in semantically distinct clusters, and (2) failing to propagate relevance signals to the broader corpus. To address these limitations, we propose Bayesian Active Learning with Gaussian Processes guided by LLM relevance scoring (BAGEL), a novel framework that propagates sparse LLM relevance signals across the embedding space to guide global exploration. BAGEL models the multimodal relevance distribution across the entire embedding space with a query-specific Gaussian Process (GP) based on LLM relevance scores. Subsequently, it iteratively selects passages for scoring by strategically balancing the exploitation of high-confidence regions with the exploration of uncertain areas. Extensive experiments across four benchmark datasets and two LLM backbones demonstrate that BAGEL effectively explores and captures complex relevance distributions and outperforms LLM reranking methods under the same LLM budget on all four datasets. 

---
# RankUp: Towards High-rank Representations for Large Scale Advertising Recommender Systems 

**Authors**: Jin Chen, Shangyu Zhang, Bin Hu, Chao Zhou, Junwei Pan, Gengsheng Xue, Wentao Ning, Gengyu Weng, Wang Zheng, Shaohua Liu, Zeen Xu, Chengyuan Mai, Tingyu Jiang, Lifeng Wang, Shudong Huang, Chengguo Yin, Haijie Gu, Jie Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2604.17878)  

**Abstract**: The scaling laws for recommender systems have been increasingly validated, where MetaFormer-based architectures consistently benefit from increased model depth, hidden dimensionality, and user behavior sequence length. However, whether representation capacity scales proportionally with parameter growth remains largely unexplored. Prior studies on RankMixer reveal that the effective rank of token representations exhibits a damped oscillatory trajectory across layers, failing to increase consistently with depth and even degrading in deeper layers. Motivated by this observation, we propose \textbf{RankUp}, an architecture designed to mitigate representation collapse and enhance expressive capacity through randomized permutation splitting over sparse features, a multi-embedding paradigm, global token integration, crossed pretrained embedding tokens and task-specific token decoupling. RankUp has been fully deployed in large-scale production across Weixin Video Accounts, Official Accounts and Moments, yielding GMV improvements of 3.41\%, 4.81\% and 2.21\%, respectively. 

---
# FedCRF: A Federated Cross-domain Recommendation Method with Semantic-driven Deep Knowledge Fusion 

**Authors**: Lei Guo, Ting Yang, Hui Liu, Xu Yu, Xiaohui Han, Xinhua Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.17681)  

**Abstract**: As user behavior data becomes increasingly scattered across different platforms, achieving cross-domain knowledge fusion while preserving privacy has become a critical issue in recommender systems. Existing PPCDR methods usually rely on overlapping users or items as a bridge, making them inapplicable to non-overlapping scenarios. They also suffer from limitations in the collaborative modeling of global and local semantics. To this end, this paper proposes a Federated Cross-domain Recommendation method with deep knowledge Fusion (FedCRF). Using textual semantics as a cross-domain bridge, FedCRF achieves cross-domain knowledge transfer via federated semantic learning under the non-overlapping scenario. Specifically, FedCRF constructs global semantic clusters on the server side to extract shared semantic information, and designs a FGSAT module on the client side to dynamically adapt to local data distributions and alleviate cross-domain distribution shift. Meanwhile, it builds a semantic graph based on textual features to learn representations that integrate both structural and semantic information, and introduces contrastive learning constraints between global and local semantic representations to enhance semantic consistency and promote deep knowledge fusion. In this framework, only item semantic representations are shared, while user interaction data remains locally stored, effectively mitigating privacy leakage risks. Experimental results on multiple real-world datasets show that FedCRF significantly outperforms existing methods in terms of Recall@20 and NDCG@20, validating its effectiveness and superiority in non-overlapping cross-domain recommendation scenarios. 

---
# MasterSet: A Large-Scale Benchmark for Must-Cite Citation Recommendation in the AI/ML Literature 

**Authors**: Md Toyaha Rahman Ratul, Zhiqian Chen, Kaiqun Fu, Taoran Ji, Lei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.17680)  

**Abstract**: The explosive growth of AI and machine learning literature -- with venues like NeurIPS and ICLR now accepting thousands of papers annually -- has made comprehensive citation coverage increasingly difficult for researchers. While citation recommendation has been studied for over a decade, existing systems primarily focus on broad relevance rather than identifying the critical set of ``must-cite'' papers: direct experimental baselines, foundational methods, and core dependencies whose omission would misrepresent a contribution's novelty or undermine reproducibility. We introduce MasterSet, a large-scale benchmark specifically designed to evaluate must-cite recommendation in the AI/ML domain. MasterSet incorporates over 150,000 papers collected from official conference proceedings/websites of 15 leading venues, serving as a comprehensive candidate pool for retrieval. We annotate citations with a three-tier labeling scheme: (I) experimental baseline status, (II) core relevance (1--5 scale), and (III) intra-paper mention frequency. Our annotation pipeline leverages an LLM-based judge, validated by human experts on a stratified sample. The benchmark task requires retrieving must-cite papers from the candidate pool given only a query paper's title and abstract, evaluated by Recall@$K$. We establish baselines using sparse retrieval, dense scientific embeddings, and graph-based methods, demonstrating that must-cite retrieval remains a challenging open problem. 

---
# Code-Switching Information Retrieval: Benchmarks, Analysis, and the Limits of Current Retrievers 

**Authors**: Qingcheng Zeng, Yuheng Lu, Zeqi Zhou, Heli Qi, Puxuan Yu, Fuheng Zhao, Hitomi Yanaka, Weihao Xuan, Naoto Yokoya  

**Link**: [PDF](https://arxiv.org/pdf/2604.17632)  

**Abstract**: Code-switching is a pervasive linguistic phenomenon in global communication, yet modern information retrieval systems remain predominantly designed for, and evaluated within, monolingual contexts. To bridge this critical disconnect, we present a holistic study dedicated to code-switching IR. We introduce CSR-L (Code-Switching Retrieval benchmark-Lite), constructing a dataset via human annotation to capture the authentic naturalness of mixed-language queries. Our evaluation across statistical, dense, and late-interaction paradigms reveals that code-switching acts as a fundamental performance bottleneck, degrading the effectiveness of even robust multilingual models. We demonstrate that this failure stems from substantial divergence in the embedding space between pure and code-switched text. Scaling this investigation, we propose CS-MTEB, a comprehensive benchmark covering 11 diverse tasks, where we observe performance declines of up to 27%. Finally, we show that standard multilingual techniques like vocabulary expansion are insufficient to resolve these deficits completely. These findings underscore the fragility of current systems and establish code-switching as a crucial frontier for future IR optimization. 

---
# Matlas: A Semantic Search Engine for Mathematics 

**Authors**: Haocheng Ju, Leheng Chen, Peihao Wu, Bryan Dai, Bin Dong  

**Link**: [PDF](https://arxiv.org/pdf/2604.17484)  

**Abstract**: Retrieving mathematical knowledge is a central task in both human-driven research, such as determining whether a result already exists, finding related results, and identifying historical origins, and in emerging AI systems for mathematics, where reliable grounding is essential. However, the scale and structure of the mathematical literature pose significant challenges: results are distributed across millions of documents, and individual statements are often difficult to interpret in isolation due to their dependence on prior definitions and theorems. In this paper, we introduce Matlas, a semantic search engine for mathematical statements. Matlas is built on a large-scale corpus of 8.07 million statements extracted from 435K peer-reviewed papers spanning 1826 to 2025, drawn from a curated set of 180 journals selected using an ICM citation-based criterion, together with 1.9K textbooks. From these sources, we extract mathematical statements together with their dependencies, construct document-level dependency graphs, and recursively unfold statements in topological order to produce more self-contained representations. On top of this corpus, we develop a semantic retrieval system that enables efficient search for mathematical results using natural language queries. We hope that Matlas can improve the efficiency of theorem retrieval for mathematicians and provide a structured source of grounding for AI systems tackling research-level mathematical problems, and serve as part of the infrastructure for mathematical knowledge retrieval. 

---
# Transparent and Controllable Recommendation Filtering via Multimodal Multi-Agent Collaboration 

**Authors**: Chi Zhang, Zhipeng Xu, Jiahao Liu, Dongsheng Li, Hansu Gu, Peng Zhang, Ning Gu, Tun Lu  

**Link**: [PDF](https://arxiv.org/pdf/2604.17459)  

**Abstract**: While personalized recommender systems excel at content discovery, they frequently expose users to undesirable or discomforting information, highlighting the critical need for user-centric filtering tools. Current methods leveraging Large Language Models (LLMs) struggle with two major bottlenecks: they lack multimodal awareness to identify visually inappropriate content, and they are highly prone to "over-association" -- incorrectly generalizing a user's specific dislike (e.g., anxiety-inducing marketing) to block benign, educational materials. These unconstrained hallucinations lead to a high volume of false positives, ultimately undermining user agency. To overcome these challenges, we introduce a novel framework that integrates end-to-cloud collaboration, multimodal perception, and multi-agent orchestration. Our system employs a fact-grounded adjudication pipeline to eliminate inferential hallucinations. Furthermore, it constructs a dynamic, two-tier preference graph that allows for explicit, human-in-the-loop modifications (via Delta-adjustments), explicitly preventing the algorithm from catastrophically forgetting fine-grained user intents. Evaluated on an adversarial dataset comprising 473 highly confusing samples, the proposed architecture effectively curbed over-association, decreasing the false positive rate by 74.3% and achieving nearly twice the F1-Score of traditional text-only baselines. Additionally, a 7-day longitudinal field study with 19 participants demonstrated robust intent alignment and enhanced governance efficiency. User feedback confirmed that the framework drastically improves algorithmic transparency, rebuilds user control, and alleviates the fear of missing out (FOMO), paving the way for transparent human-AI co-governance in personalized feeds. 

---
# MemSearch-o1: Empowering Large Language Models with Reasoning-Aligned Memory Growth in Agentic Search 

**Authors**: Sheng Zhang, Junyi Li, Yingyi Zhang, Pengyue Jia, Yichao Wang, Xiaowei Qian, Wenlin Zhang, Maolin Wang, Yong Liu, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.17265)  

**Abstract**: Recent advances in large language models (LLMs) have scaled the potential for reasoning and agentic search, wherein models autonomously plan, retrieve, and reason over external knowledge to answer complex queries. However, the iterative think-search loop accumulates long system memories, leading to memory dilution problem. In addition, existing memory management methods struggle to capture fine-grained semantic relations between queries and documents and often lose substantial information. Therefore, we propose MemSearch-o1, an agentic search framework built on reasoning-aligned memory growth and retracing. MemSearch-o1 dynamically grows fine-grained memory fragments from memory seed tokens from the queries, then retraces and deeply refines the memory via a contribution function, and finally reorganizes a globally connected memory path. This shifts memory management from stream-like concatenation to structured, token-level growth with path-based reasoning. Experiments on eight benchmark datasets show that MemSearch-o1 substantially mitigates memory dilution, and more effectively activates the reasoning potential of diverse LLMs, establishing a solid foundation for memory-aware agentic intelligence. 

---
# HORIZON: A Benchmark for In-the-wild User Behaviour Modeling 

**Authors**: Arnav Goel, Pranjal A Chitale, Bhawna Paliwal, Bishal Santra, Amit Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2604.17259)  

**Abstract**: User behavior in the real world is diverse, cross-domain, and spans long time horizons. Existing user modeling benchmarks however remain narrow, focusing mainly on short sessions and next-item prediction within a single domain. Such limitations hinder progress toward robust and generalizable user models. We present HORIZON, a new benchmark that reformulates user modeling along three axes i.e. dataset, task, and evaluation. Built from a large-scale, cross-domain reformulation of Amazon Reviews, HORIZON covers 54M users and 35M items, enabling both pretraining and realistic evaluation of models in heterogeneous environments. Unlike prior benchmarks, it challenges models to generalize across domains, users, and time, moving beyond standard missing-positive prediction in the same domain. We propose new tasks and evaluation setups that better reflect real-world deployment scenarios. These include temporal generalization, sequence-length variation, and modeling unseen users, with metrics designed to assess general user behavior understanding rather than isolated next-item prediction. We benchmark popular sequential recommendation architectures alongside LLM-based baselines that leverage long-term interaction histories. Our results highlight the gap between current methods and the demands of real-world user modeling, while establishing HORIZON as a foundation for research on temporally robust, cross-domain, and general-purpose user models. 

---
# HeadRank: Decoding-Free Passage Reranking via Preference-Aligned Attention Heads 

**Authors**: Juyuan Wang, Chenxing Wang, Yuchen Fang, Huiyun Hu, Junwu Du, Aolin Li, Haijun Wu, Jin Xu, Ligang Liu, Dongliang Liao  

**Link**: [PDF](https://arxiv.org/pdf/2604.17237)  

**Abstract**: Decoding-free reranking methods that read relevance signals directly from LLM attention weights offer significant latency advantages over autoregressive approaches, yet suffer from attention score homogenization: middle-context documents receive near-identical scores, destroying the fine-grained distinctions required for ranking. We propose HeadRank, a framework that lifts preference optimization from discrete token space into the continuous attention domain through entropy-regularized head selection, hard adjacent-level preference pairs, and a distribution regularizer that jointly sharpen discriminability in the homogenized middle zone. Depth truncation at the deepest selected layer further reduces inference to $\mathcal{O}(1)$ forward passes. Across 14 benchmarks on three Qwen3 scales (0.6B--4B) using only 211 training queries, HeadRank consistently outperforms generative and decoding-free baselines with 100\% formatting success. At 4B, 57.4\% of relevant middle-zone documents reach the top quartile versus 14.2\% for irrelevant ones -- a 43-percentage-point selectivity gap that demonstrates the effectiveness of attention-space preference alignment for listwise reranking. 

---
# RLM-on-KG: Heuristics First, LLMs When Needed: Adaptive Retrieval Control over Mention Graphs for Scattered Evidence 

**Authors**: Andrea Volpini, Elie Raad  

**Link**: [PDF](https://arxiv.org/pdf/2604.17056)  

**Abstract**: When does an LLM controller outperform rule-based traversal for knowledge graph exploration? We study this question through RLM-on-KG, a retrieval system that treats an LLM as an autonomous navigator over an RDF-encoded mention graph for grounded question answering. Unlike GraphRAG pipelines that rely on offline LLM indexing, RLM-on-KG performs entity-first, multi-hop exploration at query time using deterministic graph construction and a fixed tool set. Our central finding is a conditional advantage: the value of LLM control depends on evidence scatter and tool-calling sophistication. The paper's core claim is LLM control versus heuristic traversal, not a generic win over GraphRAG. On GraphRAG-Bench Novel (519 questions), Gemini 2.0 Flash achieves +2.47 pp F1 over a rule-based heuristic baseline (p < 0.0001), but only +0.16 pp over a GraphRAG-local variant (not significant). With a stronger controller, Claude Haiku 4.5, the gain over heuristic grows to +4.37 pp (p < 0.001) and extends to a +2.42 pp significant improvement over GraphRAG-local (p < 0.001). The gain is largest when gold evidence is scattered across 6-10 chunks (+3.21 pp) and smallest for concentrated evidence (+1.85 pp). Cross-scale validation on MuSiQue confirms that the LLM-over-heuristic advantage transfers, with expected attenuation on smaller per-question graphs. The core architectural insight is the separation of candidate discovery from ranking: the LLM adds value through exploration breadth, while final evidence selection is best handled by pure vector re-ranking. Beyond retrieval, exploration traces provide a proposed stress-test harness for structured data quality, yielding diagnostics for coverage, connectivity, provenance, and queryability. 

---
# On the Robustness of LLM-Based Dense Retrievers: A Systematic Analysis of Generalizability and Stability 

**Authors**: Yongkang Li, Panagiotis Eustratiadis, Yixing Fan, Evangelos Kanoulas  

**Link**: [PDF](https://arxiv.org/pdf/2604.16576)  

**Abstract**: Decoder-only large language models (LLMs) are increasingly replacing BERT-style architectures as the backbone for dense retrieval, achieving substantial performance gains and broad adoption. However, the robustness of these LLM-based retrievers remains underexplored. In this paper, we present the first systematic study of the robustness of state-of-the-art open-source LLM-based dense retrievers from two complementary perspectives: generalizability and stability. For generalizability, we evaluate retrieval effectiveness across four benchmarks spanning 30 datasets, using linear mixed-effects models to estimate marginal mean performance and disentangle intrinsic model capability from dataset heterogeneity. Our analysis reveals that while instruction-tuned models generally excel, those optimized for complex reasoning often suffer a ``specialization tax,'' exhibiting limited generalizability in broader contexts. For stability, we assess model resilience against both unintentional query variations~(e.g., paraphrasing, typos) and malicious adversarial attacks~(e.g., corpus poisoning). We find that LLM-based retrievers show improved robustness against typos and corpus poisoning compared to encoder-only baselines, yet remain vulnerable to semantic perturbations like synonymizing. Further analysis shows that embedding geometry (e.g., angular uniformity) provides predictive signals for lexical stability and suggests that scaling model size generally improves robustness. These findings inform future robustness-aware retriever design and principled benchmarking. Our code is publicly available at this https URL. 

---
# Modeling User Exploration Saturation: When Recommender Systems Should Stop Pushing Novelty 

**Authors**: Enock O. Ayiku, Evelyn Osei, Emebo Onyeka  

**Link**: [PDF](https://arxiv.org/pdf/2604.16419)  

**Abstract**: Fairness-aware recommender systems often mitigate bias by increasing exposure to under-represented or long-tail content, commonly through mechanisms that promote novelty and diversity. In practice, the strength of such interventions is typically controlled using global hyperparameters, fixed regularization weights, heuristic caps, or offline tuning strategies. These approaches implicitly assume that a single level of exploration is appropriate across users, contexts, and stages of interaction. In this work, we study exploration saturation as a user-dependent phenomenon arising from fairness- and novelty-driven recommendation strategies. We define exploration saturation as the point at which further increases in exploration no longer improve user utility and may instead reduce engagement or perceived relevance. Rather than proposing a new fairness-aware algorithm or optimizing a specific objective, we empirically analyze how increasing exploration affects users across varied recommendation models. Through longitudinal experiments using MovieLens-1M and this http URL datasets, our results indicate that fairness-induced exploration exhibits diminishing or non-monotonic returns and varies substantially across users. In particular, users with limited interaction histories tend to reach saturation earlier, suggesting that uniform fairness or novelty pressure can disproportionately disadvantage certain users. These findings reveal a trade-off between fairness and user experience, suggesting that recommendation systems should adapt not only to relevance but also to the amount of fairness-driven exploration applied to individual users. 

---
# Tensor Manifold-Based Graph-Vector Fusion for AI-Native Academic Literature Retrieval 

**Authors**: Xing Wei, Yang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.16416)  

**Abstract**: The rapid development of large language models and AI agents has triggered a paradigm shift in academic literature retrieval, putting forward new demands for fine-grained, time-aware, and programmable retrieval. Existing graph-vector fusion methods still face bottlenecks such as matrix dependence, storage explosion, semantic dilution, and lack of AI-native support. This paper proposes a geometry-unified graph-vector fusion framework based on tensor manifold theory, which formally proves that an academic literature graph is a discrete projection of a tensor manifold, realizing the native unification of graph topology and vector geometric embedding. Based on this theoretical conclusion, we design four core modules: matrix-independent temporal diffusion signature update, hierarchical temporal manifold encoding, temporal Riemannian manifold indexing, and AI-agent programmable retrieval. Theoretical analysis and complexity proof show that all core algorithms have linear time and space complexity, which can adapt to large-scale dynamic academic literature graphs. This research provides a new theoretical framework and engineering solution for AI-native academic literature retrieval, promoting the industrial application of graph-vector fusion technology in the academic field. 

---
# GraphRAG-Router: Learning Cost-Efficient Routing over GraphRAGs and LLMs with Reinforcement Learning 

**Authors**: Dongzhe Fan, Chuanhao Ji, Zimu Wang, Tong Chen, Qiaoyu Tan  

**Link**: [PDF](https://arxiv.org/pdf/2604.16401)  

**Abstract**: Graph-based retrieval-augmented generation (GraphRAG) has recently emerged as a powerful paradigm for knowledge-intensive question answering, especially for tasks that require structured evidence organization and multi-hop reasoning. However, existing GraphRAG systems are typically built in a one-size-fits-all manner, relying on a fixed retrieval framework and a single, often large and costly, generator LLM for all queries. This static design limits their ability to adapt to the complexity of varying questions and often incurs unnecessary computational cost. To fill in the gap, we propose GraphRAG-Router, a cost-efficient framework that adopts a hierarchical routing strategy to coordinate heterogeneous GraphRAGs and generator LLMs. Specifically, GraphRAG-Router is first warmed up through supervised fine-tuning and then optimized with a two-stage reinforcement learning procedure, whose second stage introduces a curriculum cost-aware reward to encourage difficulty-aware and economical generator allocation. Extensive experiments on six general-domain and multi-hop QA benchmarks show that GraphRAG-Router consistently outperforms state-of-the-art baselines, reducing the overuse of large LLMs by nearly 30% while maintaining strong generalization capability. 

---
# A Reference Architecture for Agentic Hybrid Retrieval in Dataset Search 

**Authors**: Riccardo Terrenzi, Phongsakon Mark Konrad, Tim Lukas Adam, Serkan Ayvaz  

**Link**: [PDF](https://arxiv.org/pdf/2604.16394)  

**Abstract**: Ad hoc dataset search requires matching underspecified natural-language queries against sparse, heterogeneous metadata records, a task where typical lexical or dense retrieval alone falls short. We reposition dataset search as a software-architecture problem and propose a bounded, auditable reference architecture for agentic hybrid retrieval that combines BM25 lexical search with dense-embedding retrieval via reciprocal rank fusion (RRF), orchestrated by a large language model (LLM) agent that repeatedly plans queries, evaluates the sufficiency of results, and reranks candidates. To reduce the vocabulary mismatch between user intent and provider-authored metadata, we introduce an offline metadata augmentation step in which an LLM generates pseudo-queries for each dataset record, augmenting both retrieval indexes before query time. Two architectural styles are examined: a single ReAct agent and a multi-agent horizontal architecture with Feedback Control. Their quality-attribute tradeoffs are analyzed with respect to modifiability, observability, performance, and governance. An evaluation framework comprising seven system variants is defined to isolate the contribution of each architectural decision. The architecture is presented as an extensible reference design for the software architecture community, incorporating explicit governance tactics to bound and audit nondeterministic LLM components. 

---
# Large language models for post-publication research evaluation: Evidence from expert recommendations and citation indicators 

**Authors**: Mengjia Wu, Yi Zhang, Robin Haunschild, Lutz Bornmann  

**Link**: [PDF](https://arxiv.org/pdf/2604.16387)  

**Abstract**: Assessing the quality of scientific research is essential for scholarly communication, yet widely used approaches face limitations in scalability, subjectivity, and time delay. Recent advances in large language models (LLMs) offer new opportunities for automated research evaluation based on textual content. This study examines whether LLMs can support post-publication peer review tasks by benchmarking their outputs against expert judgments and citation-based indicators. Two evaluation tasks are constructed using articles from the H1 Connect platform: identifying high-quality articles and performing finer-grained evaluation including article rating, merit classification, and expert style commenting. Multiple model families, including BERT models, general-purpose LLMs, and reasoning oriented LLMs, are evaluated under multiple learning strategies. Results show that LLMs perform well in coarse grained evaluation tasks, achieving accuracy above 0.8 in identifying highly recommended articles. However, performance decreases substantially in fine-grained rating tasks. Few-shot prompting improves performance over zero-shot settings, while supervised fine-tuning produces the strongest and most balanced results. Retrieval augmented prompting improves classification accuracy in some cases but does not consistently strengthen alignment with citation indicators. The overall correlations between model outputs and citation indicators remain positive but moderate. 

---
# LLMAR: A Tuning-Free Recommendation Framework for Sparse and Text-Rich Industrial Domains 

**Authors**: Ryogo Hishikawa, Ichiro Kataoka, Shinya Yuda  

**Link**: [PDF](https://arxiv.org/pdf/2604.16379)  

**Abstract**: Industrial B2B applications (e.g., construction site risk prediction, material procurement) face extreme data sparsity yet feature rich textual interactions. In such environments, traditional ID-based collaborative filtering fails lacking co-occurrence signals, while fine-tuning standard Large Language Models (LLMs) incurs high operational costs and struggles with frequent data drift.
We propose LLMAR (LLM-Annotated Recommendation), a tuning-free framework. Moving beyond simple embeddings, LLMAR systematically integrates LLM reasoning to capture user "latent motives" without any training process. We introduce three core contributions: (1) Inference-Driven Annotation: uses LLMs to transform behavioral history into structured semantic motives, enabling reasoning-based matching unattainable by ID-based methods; (2) Reflection Loop: a self-correction mechanism that refines generated queries to mitigate hallucinations and resolve "context competition" between past history and current instructions; and (3) Cost-Effective Architecture: relies on tuning-free components and asynchronous batch processing to minimize maintenance costs.
Evaluations on public benchmarks (MovieLens-1M, Amazon Prime Pantry) and a sparse industrial dataset (construction risk prediction) demonstrate that LLMAR outperforms state-of-the-art learning-based models (SASRecF), achieving up to a 54.6% nDCG@10 improvement on the industrial dataset. Inference costs remain highly practical (~$1 per 1,000 users). For B2B domains where strict real-time latency is not critical, combining LLM reasoning with self-verification offers a superior alternative to training-based approaches across accuracy, explainability, and operational cost. 

---
# AgriIR: A Scalable Framework for Domain-Specific Knowledge Retrieval 

**Authors**: Shuvam Banerji Seal, Aheli Poddar, Alok Mishra, Dwaipayan Roy  

**Link**: [PDF](https://arxiv.org/pdf/2604.16353)  

**Abstract**: This paper introduces AgriIR, a configurable retrieval augmented generation (RAG) framework designed to deliver grounded, domain-specific answers while maintaining flexibility and low computational cost. Instead of relying on large, monolithic models, AgriIR decomposes the information access process into declarative modular stages -- query refinement, sub-query planning, retrieval, synthesis, and evaluation. This design allows practitioners to adapt the framework to new knowledge verticals without modifying the architecture. Our reference implementation targets Indian agricultural information access, integrating 1B-parameter language models with adaptive retrievers and domain-aware agent catalogues. The system enforces deterministic citation, integrates telemetry for transparency, and includes automated deployment assets to ensure auditable, reproducible operation. By emphasizing architectural design and modular control, AgriIR demonstrates that well-engineered pipelines can achieve domain-accurate, trustworthy retrieval even under constrained resources. We argue that this approach exemplifies ``AI for Agriculture'' by promoting accessibility, sustainability, and accountability in retrieval-augmented generation systems. 

---
# Training for Compositional Sensitivity Reduces Dense Retrieval Generalization 

**Authors**: Radoslav Ralev, Aditeya Baral, Iliya Zhechev, Jen Agarwal, Srijith Rajamohan  

**Link**: [PDF](https://arxiv.org/pdf/2604.16351)  

**Abstract**: Dense retrieval compresses texts into single embeddings ranked by cosine similarity. While efficient for recall, this interface is brittle for identity-level matching: minimal compositional edits (negation, role swaps) flip meaning yet retain high similarity. Motivated by geometric results for unit-sphere cosine spaces (Kang et al., 2025), we test this retrieval-composition tension in text-only retrieval. Across four dual-encoder backbones, adding structure-targeted negatives consistently reduces zero-shot NanoBEIR retrieval (8-9% mean nDCG@10 drop on small backbones; up to 40% on medium ones), while only partially improving pooled-space separation. Treating pooled cosine as a recall interface, we then benchmark verifiers scoring token--token cosine maps. MaxSim (late interaction) excels at reranking but fails to reject structural near-misses, whereas a small Transformer over similarity maps reliably separates near-misses under end-to-end training. 

---
# LiteSemRAG: Lightweight LLM-Free Semantic-Aware Graph Retrieval for Robust RAG 

**Authors**: Xiao Yue, Guangzhi Qu, Lige Gan  

**Link**: [PDF](https://arxiv.org/pdf/2604.16350)  

**Abstract**: Graph-based Retrieval-Augmented Generation (RAG) has shown great potential for improving multi-level reasoning and structured evidence aggregation. However, existing graph-based RAG frameworks heavily rely on exploiting large language models (LLMs) during indexing and querying, leading to high token consumption, computational cost and latency overhead. In this paper, we propose LiteSemRAG, a lightweight, fully LLM-free, semantic-aware graph retrieval framework. LiteSemRAG constructs a heterogeneous semantic graph by exploiting contextual token-level embeddings, explicitly separating surface lexical representations from context-dependent semantic meanings. To robustly model polysemy, we introduce a dynamic semantic node construction mechanism with chunk-level context aggregation and adaptive anomaly handling. At query stage, LiteSemRAG performs a two-step semantic-aware retrieval process that integrates co-occurrence graph weighting with an isolated semantic recovery mechanism, enabling balanced structural reasoning and semantic coverage. We evaluate LiteSemRAG on three benchmark datasets and experimental results show that LiteSemRAG achieves the best mean reciprocal rank (MRR@10) across all datasets and competitive or superior recall rate (Recall@10) compared to state-of-the-art LLM-based graph RAG systems. Meanwhile, LiteSemRAG consumes zero LLM tokens and achieves substantial efficiency improvements in both indexing and querying due to the elimination of LLM usage. These results demonstrate the effectiveness of LiteSemRAG, indicating that a strong semantic-aware graph retrieval framework can be achieved without relying on LLM-based approaches. 

---
# Benchmarking Real-Time Question Answering via Executable Code Workflows 

**Authors**: Wenjie Zhou, Yuan Gao, Xin Zhou, Hao Fu, Zhongjian Miao, Wei Chen, Bo Chen, Xiaobing Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.16349)  

**Abstract**: Retrieving real-time information is a fundamental capability for search-integrated agents in real-world applications. However, existing benchmarks are predominantly static and therefore fail to capture the temporal dynamics of information and the continuously evolving nature of real-world knowledge. To address this limitation, we propose RT-QA, a dynamic evaluation framework that leverages executable code workflows to retrieve up-to-date answers at evaluation time. Specifically, we construct an agent-driven pipeline that autonomously generates code for web crawling and DOM-based answer extraction to produce real-time ground truth. To ensure robust evaluation over time, the pipeline further incorporates a self-repair mechanism to adapt to changes in web page structures. RT-QA spans 12 domains (e.g., Finance, Sports) with 320 Chinese questions categorized into three difficulty levels. Extensive evaluations of state-of-the-art models (e.g., GPT-5.2, GLM-4.7) reveal significant limitations in real-time adaptability: even the best models achieve only 46% accuracy. Our analysis highlights two primary failure modes: (1) Lazy Retrieval, where agents rely on search snippets instead of deeply scanning specific websites for information (20% of failures); and (2) Temporal Confusion, a cognitive error where agents retrieve a historical date (e.g., an event in 2024) and fail to re-anchor to the current time (2026) for subsequent reasoning. These findings suggest that future agents require not just better retrieval strategies, but robust temporal state management. 

---
# HR-Agents: Using Multiple LLM-based Agents to Improve Q&A about Brazilian Labor Legislation 

**Authors**: Abriel K. Moraes, Gabriel S. M. Dias, Vitor L. Fabris, Lucas D. Gessoni, Leonardo R. do Nascimento, Charles S. Oliveira, Vitor G. C. B. de Farias, Fabiana C. Q. de O. Marucci, Matheus H. R. Vicente, Gabriel U. Talasso, Erik Soares, Amparo Munoz, Sildolfo Gomes, Maria L. A. de S. Cruvinel, Leonardo T. dos Santos, Renata De Paris, Wandemberg Gibaut  

**Link**: [PDF](https://arxiv.org/pdf/2604.16337)  

**Abstract**: The Consolidation of Labor Laws (CLT) serves as the primary legal framework governing labor relations in Brazil, ensuring essential protections for workers. However, its complexity creates challenges for Human Resources (HR) professionals in navigating regulations and ensuring compliance. Traditional methods for addressing labor law inquiries often lead to inefficiencies, delays, and inconsistencies. To enhance the accuracy and efficiency of legal question-answering (Q&A), a multi-agent system powered by Large Language Models (LLMs) is introduced. This approach employs specialized agents to address distinct aspects of employment law while integrating Retrieval-Augmented Generation (RAG) to enhance contextual relevance. Implemented using CrewAI, the system enables cooperative agent interactions, ensuring response validation and reducing misinformation. The effectiveness of this framework is evaluated through a comparison with a baseline RAG pipeline utilizing a single LLM, using automated metrics such as BLEU, LLM-as-judge evaluations, and expert human assessments. Results indicate that the multi-agent approach improves response coherence and correctness, providing a more reliable and efficient solution for HR professionals. This study contributes to AI-driven legal assistance by demonstrating the potential of multi-agent LLM architectures in improving labor law compliance and streamlining HR operations. 

---
# A Collection of Systematic Reviews in Computer Science 

**Authors**: Pierre Achkar, Tim Gollub amd Martin Potthast  

**Link**: [PDF](https://arxiv.org/pdf/2604.16330)  

**Abstract**: Systematic reviews are the standard method for synthesizing scientific evidence, but their creation requires substantial manual effort, particularly during retrieval and screening. While recent work has explored automating these steps, evaluation resources remain largely confined to the biomedical domain, limiting reproducible experimentation in other domains. This paper introduces SR4CS, a large-scale collection of systematic reviews in computer science, designed to support reproducible research on Boolean query generation, retrieval, and screening. The corpus comprises 1,212 systematic reviews with their original expert-designed Boolean search queries, 104,316 resolved references, and structured methodological metadata. For controlled evaluation, the original Boolean queries are additionally provided in a normalized, approximated form operating over titles and abstracts. To illustrate the intended use of the collection, baseline experiments compare the approximated expert Boolean queries with zero-shot LLM-generated Boolean queries, BM25, and dense retrieval under a unified evaluation setting. The results highlight systematic differences in precision, recall, and ranking behavior across retrieval paradigms and expose limitations of naive zero-shot Boolean generation. SR4CS is released under an open license on Zenodo (this https URL), together with documentation and code (this https URL), to enable reproducible evaluation and future research on scaling systematic review automation. 

---
# Beyond Single-Score Ranking: Facet-Aware Reranking for Controllable Diversity in Paper Recommendation 

**Authors**: Duan Ming Tao  

**Link**: [PDF](https://arxiv.org/pdf/2604.16329)  

**Abstract**: Current paper recommendation systems output a single similarity score that mixes different notions of relatedness, so users cannot specify why papers should be similar. We present SciFACE (Scientific Faceted Cross-Encoder), a reranking framework that models two independent facets: Background (what problem is studied) and Method (how it is solved). SciFACE trains two separate cross-encoders on 5,891 real seed-candidate paper pairs labeled by GPT-4o-mini with facet-specific criteria and validated against human judgments. On CSFCube, SciFACE reaches 70.63 NDCG@20 on Background (5.9 points above SPECTER) and 49.06 NDCG@20 on Method (31.1 points above SPECTER), competitive with state-of-the-art results. Compared with FaBLE without citation pre-training, SciFACE improves Method NDCG@20 by 4.1 points while using 5,891 labeled pairs versus 40K synthetic augmentations. These results show that high-quality grounded facet labels can be more data-efficient than large-scale synthetic augmentation for learning fine-grained scientific similarity. 

---
# Diagnosing LLM-based Rerankers in Cold-Start Recommender Systems: Coverage, Exposure and Practical Mitigations 

**Authors**: Ekaterina Lemdiasova, Nikita Zmanovskii  

**Link**: [PDF](https://arxiv.org/pdf/2604.16318)  

**Abstract**: Large language models (LLMs) and cross-encoder rerankers have gained attention for improving recommender systems, particularly in cold-start scenarios where user interaction history is limited. However, practical deployment reveals significant performance gaps between LLM-based approaches and simple baselines. This paper presents a systematic diagnostic study of cross-encoder rerankers in cold-start movie recommendation using the Serendipity-2018 dataset. Through controlled experiments with 500 users across multiple random seeds, we identify three critical failure modes: (1) low retrieval coverage in candidate generation (recall@200 = 0.109 vs. 0.609 for baselines), (2) severe exposure bias with rerankers concentrating recommendations on 3 unique items versus 497 for random baseline, and (3) minimal score discrimination between relevant and irrelevant items (mean difference = 0.098, Cohen's d = 0.13). We demonstrate that popularity-based ranking substantially outperforms LLM reranking (HR@10: 0.268 vs. 0.008, p < 0.001), with the performance gap primarily attributable to retrieval stage limitations rather than reranker capacity. Based on these findings, we provide actionable recommendations including hybrid retrieval strategies, candidate pool size optimization, and score calibration techniques. All code, configurations, and experimental results are made available for reproducibility. 

---
# Paper2Data: Large-Scale LLM Extraction and Metadata Structuring of Global Urban Data from Scientific Literature 

**Authors**: Runwen You, Tong Xia, Jingzhi Wang, Jiankun Zhang, Tengyao Tu, Jinghua Piao, Yi Chang, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.16317)  

**Abstract**: Urban data support a wide range of applications across multiple disciplines. However, at the global scale, there is no unified platform for urban data discovery. As a result, researchers often have to manually search through websites or scientific literature to identify relevant datasets. To address this problem, we curate an open urban data discovery portal, \textit{UrbanDataMiner}, which supports dataset-level search and filtering over more than 60{,}000 urban datasets extracted from over 15{,}000 Nature-affiliated publications. \textit{UrbanDataMiner} is enabled by \textit{Paper2Data}, a novel large-scale LLM-driven pipeline that automatically identifies dataset mentions in scientific papers and structures them using a unified urban data metadata schema. Human-annotated evaluation demonstrates that \textit{Paper2Data} achieves high recall (approximately 90\%) in dataset identification and high field-level precision (above 80\%). In addition, \textit{UrbanDataMiner} can retrieve over 9\% of datasets that are not easily discoverable through general-purpose search engines such as Google. Overall, our work provides the first large-scale, literature-derived infrastructure for urban data discovery and enables more systematic and reusable data-driven research across disciplines. Our code and data are publicly available\footnote{this https URL}. 

---
# MARA: A Multimodal Adaptive Retrieval-Augmented Framework for Document Question Answering 

**Authors**: Hui Wu, Haoquan Zhai, Yuchen Li, Hengyi Cai, Peirong Zhang, Yidan Zhang, Lei Wang, Chunle Wang, Yingyan Hou, Shuaiqiang Wang, Dawei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2604.16313)  

**Abstract**: Retrieval-based multimodal document QA aims to identify and integrate relevant information from visually rich documents with complex multimodal structures. While retrieval-augmented generation (RAG) has shown strong performance in text-based QA, its extensions to multimodal documents remain underexplored and face significant limitations. Specifically, current approaches rely on query-agnostic document representations that overlook salient content and use static top-k evidence selection, which fails to adapt to the uncertain distribution of relevant information. To address these limitations, we propose the Multimodal Adaptive Retrieval-Augmented (MARA) framework, which introduces query-adaptive mechanisms to both retrieval and generation. MARA consists of two components: a Query-Aligned Region Encoder that builds multi-level document representations and reweights them based on query relevance to improve retrieval precision; and a Self-Reflective Evidence Controller that monitors evidence sufficiency during generation and adaptively incorporates content from lower-ranked sources using a sliding-window strategy. Experiments on six multimodal QA benchmarks demonstrate that MARA consistently improves retrieval relevance and answer quality over existing SOTA method. 

---
# FlexStructRAG: Flexible Structure-Aware Multi-Granular Relational Retrieval for RAG 

**Authors**: Mengzhu Chen, Haodong Yang, Jia Cai, Xiaolin Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.16312)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems critically depend on how external knowledge is segmented, structured, and retrieved. Most existing approaches either retrieve fixed-length text chunks, which fragments discourse context, or commit to a single structured index (e.g., a knowledge graph or hypergraph), which hard-codes one relational granularity. This often yields brittle retrieval when queries require different forms of evidence, such as local binary relations, higher-order interactions, or broader document-grounded context. We propose \textbf{FlexStructRAG}, a flexible structure-aware RAG framework that supports \emph{multi-granular, query-adaptive retrieval} over heterogeneous knowledge representations. FlexStructRAG jointly constructs (i) a knowledge graph for binary relations, (ii) a knowledge hypergraph for n-ary relations, and (iii) structure-aware semantic clusters that aggregate relational evidence into document-grounded context units. To reduce semantic fragmentation induced by uniform chunking, we introduce dynamic partitioning and a truncated sliding-window extraction mechanism that incorporates bounded contextual dependencies during knowledge construction. At inference time, FlexStructRAG enables entity-, edge-, hyperedge-, and cluster-level retrieval, which can be flexibly combined to supply generation with relationally and contextually aligned evidence. Experiments on the UltraDomain benchmark across four domains show that FlexStructRAG improves semantic evaluation over strong RAG baselines. Ablation and sensitivity analysis further demonstrate the necessity of multi-granular relational retrieval and structure-aware clustering. 

---
# RAG-DIVE: A Dynamic Approach for Multi-Turn Dialogue Evaluation in Retrieval-Augmented Generation 

**Authors**: Lorenz Brehme, Benedikt Dornauer, Jan-Henrik Böttcher, Klaus Schmid, Mircea-Cristian Racasan, Ruth Breu  

**Link**: [PDF](https://arxiv.org/pdf/2604.16310)  

**Abstract**: Evaluating Retrieval-Augmented Generation (RAG) systems using static multi-turn datasets fails to capture the dynamic nature of real-world dialogues. Existing evaluation methods rely on predefined datasets, which restrict them to static, one-directional queries and limit their ability to capture the adaptive, context-dependent performance of RAG systems in interactive, multi-turn settings. Thus, we introduce the RAG-DIVE, a Dynamic Interactive Validation and Evaluation approach, that simulates user interactions with RAG systems. RAG-DIVE leverages an LLM to generate multi-turn conversations dynamically and is organized into three components. The dialogue generation stage consists of the (1) Conversation Generator, which simulates a user by creating multi-turn queries, and the (2) Conversation Validator, which filters and corrects invalid or low-quality outputs to ensure coherent conversations. The evaluation stage is handled by the (3) Conversation Evaluator, which assesses the RAG system's performance across the entire dialogue and generates both per-turn and multi-turn metrics that provide an aggregated view of system behavior. We validated RAG-DIVE through two experimental setups. First, we tested a sample RAG system, including human evaluation of dialogue quality, repeated trials to assess consistency, and an ablation study showing that RAG-DIVE detects performance changes caused by system modifications. Second, we compared RAG-DIVE with a traditional static dataset evaluation on an industrial RAG system under different configurations to verify whether both approaches reveal similar performance trends. Our findings demonstrate that RAG-DIVE facilitates dynamic, interaction-driven evaluation for multi-turn conversations, thereby advancing the assessment of RAG systems. 

---
# Domain-Specific Query Understanding for Automotive Applications: A Modular and Scalable Approach 

**Authors**: Isha Motiyani, Abhishek Kumar, Tilak Kasturi  

**Link**: [PDF](https://arxiv.org/pdf/2604.16301)  

**Abstract**: Despite the growing prevalence of large language models (LLMs) in domain-specific applications, the challenge of query understanding in the automotive sector still remains underexplored. This domain presents unique complexities due to its specialized vocabulary and the diverse range of user intents it encompasses. Unlike general-purpose assistants, automotive systems must precisely interpret user queries and route them to appropriate underlying tool, each designed to fulfill a distinct task such as part recommendations, repair procedures, or regulatory lookups. Moreover, these systems must extract structured inputs precisely aligned with the schema required by each tool. In this study, we present a novel two-step system for domain-specific query interpretation in the automotive context that achieves an effective balance between responsiveness, reliability, and scalability. Our initial single-step approach, which jointly performed classification and entity extraction, exhibited moderate performance and higher latency. By decomposing the task into a lightweight classification stage followed by targeted entity extraction using smaller, specialized prompts, our system achieves substantial gains in both efficiency and accuracy. Due to the niche nature of the automotive domain, we also curated a high-quality dataset by combining manually annotated and synthetically generated samples, all reviewed by domain experts. Overall, our findings demonstrate that decomposing query understanding into modular subtasks leads to a scalable, accurate, and latency-efficient solution. This approach establishes a strong ground for practical deployment in real-world automotive query understanding systems. 

---
# MathNet: a Global Multimodal Benchmark for Mathematical Reasoning and Retrieval 

**Authors**: Shaden Alshammari, Kevin Wen, Abrar Zainal, Mark Hamilton, Navid Safaei, Sultan Albarakati, William T. Freeman, Antonio Torralba  

**Link**: [PDF](https://arxiv.org/pdf/2604.18584)  

**Abstract**: Mathematical problem solving remains a challenging test of reasoning for large language and multimodal models, yet existing benchmarks are limited in size, language coverage, and task diversity. We introduce MathNet, a high-quality, large-scale, multimodal, and multilingual dataset of Olympiad-level math problems together with a benchmark for evaluating mathematical reasoning in generative models and mathematical retrieval in embedding-based systems. MathNet spans 47 countries, 17 languages, and two decades of competitions, comprising 30,676 expert-authored problems with solutions across diverse domains. In addition to the core dataset, we construct a retrieval benchmark consisting of mathematically equivalent and structurally similar problem pairs curated by human experts.
MathNet supports three tasks: (i) Problem Solving, (ii) Math-Aware Retrieval, and (iii) Retrieval-Augmented Problem Solving. Experimental results show that even state-of-the-art reasoning models (78.4% for Gemini-3.1-Pro and 69.3% for GPT-5) remain challenged, while embedding models struggle to retrieve equivalent problems. We further show that retrieval-augmented generation performance is highly sensitive to retrieval quality; for example, DeepSeek-V3.2-Speciale achieves gains of up to 12%, obtaining the highest scores on the benchmark. MathNet provides the largest high-quality Olympiad dataset together with the first benchmark for evaluating mathematical problem retrieval, and we publicly release both the dataset and benchmark at this https URL. 

---
# ArbGraph: Conflict-Aware Evidence Arbitration for Reliable Long-Form Retrieval-Augmented Generation 

**Authors**: Qingying Niu, Yuhao Wang, Ruiyang Ren, Bohui Fang, Wayne Xin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.18362)  

**Abstract**: Retrieval-augmented generation (RAG) remains unreliable in long-form settings, where retrieved evidence is noisy or contradictory, making it difficult for RAG pipelines to maintain factual consistency. Existing approaches focus on retrieval expansion or verification during generation, leaving conflict resolution entangled with generation. To address this limitation, we propose ArbGraph, a framework for pre-generation evidence arbitration in long-form RAG that explicitly resolves factual conflicts. ArbGraph decomposes retrieved documents into atomic claims and organizes them into a conflict-aware evidence graph with explicit support and contradiction relations. On top of this graph, we introduce an intensity-driven iterative arbitration mechanism that propagates credibility signals through evidence interactions, enabling the system to suppress unreliable and inconsistent claims before final generation. In this way, ArbGraph separates evidence validation from text generation and provides a coherent evidence foundation for downstream long-form generation. We evaluate ArbGraph on two widely used long-form RAG benchmarks, LongFact and RAGChecker, using multiple large language model backbones. Experimental results show that ArbGraph consistently improves factual recall and information density while reducing hallucinations and sensitivity to retrieval noise. Additional analyses show that these gains are evident under conflicting or ambiguous evidence, highlighting the effectiveness of evidence-level conflict resolution for improving the reliability of long-form RAG. The implementation is publicly available at this https URL. 

---
# The Collaboration Gap in Human-AI Work 

**Authors**: Varad Vishwarupe, Marina Jirotka, Nigel Shadbolt, Ivan Flechais  

**Link**: [PDF](https://arxiv.org/pdf/2604.18096)  

**Abstract**: LLMs are increasingly presented as collaborators in programming, design, writing, and analysis. Yet the practical experience of working with them often falls short of this promise. In many settings, users must diagnose misunderstandings, reconstruct missing assumptions, and repeatedly repair misaligned responses. This poster introduces a conceptual framework for understanding why such collaboration remains fragile. Drawing on a constructivist grounded theory analysis of 16 interviews with designers, developers, and applied AI practitioners working on LLM-enabled systems, and informed by literature on human-AI collaboration, we argue that stable collaboration depends not only on model capability but on the interaction's grounding conditions. We distinguish three recurrent structures of human-AI work: one-shot assistance, weak collaboration with asymmetric repair, and grounded collaboration. We propose that collaboration breaks down when the appearance of partnership outpaces the grounding capacity of the interaction and contribute a framework for discussing grounding, repair, and interaction structure in LLM-enabled work. 

---
# Peerispect: Claim Verification in Scientific Peer Reviews 

**Authors**: Ali Ghorbanpour, Soroush Sadeghian, Alireza Daghighfarsoodeh, Sajad Ebrahimi, Negar Arabzadeh, Seyed Mohammad Hosseini, Ebrahim Bagheri  

**Link**: [PDF](https://arxiv.org/pdf/2604.17667)  

**Abstract**: Peer review is central to scientific publishing, yet reviewers frequently include claims that are subjective, rhetorical, or misaligned with the submitted work. Assessing whether review statements are factual and verifiable is crucial for fairness and accountability. At the scale of modern conferences and journals, manually inspecting the grounding of such claims is infeasible. We present Peerispect, an interactive system that operationalizes claim-level verification in peer reviews by extracting check-worthy claims from peer reviews, retrieving relevant evidence from the manuscript, and verifying the claims through natural language inference. Results are presented through a visual interface that highlights evidence directly in the paper, enabling rapid inspection and interpretation. Peerispect is designed as a modular Information Retrieval (IR) pipeline, supporting alternative retrievers, rerankers, and verifiers, and is intended for use by reviewers, authors, and program committees. We demonstrate Peerispect through a live, publicly available demo (this https URL) and API services (this https URL), accompanied by a video tutorial (this https URL). 

---
# COSEARCH: Joint Training of Reasoning and Document Ranking via Reinforcement Learning for Agentic Search 

**Authors**: Hansi Zeng, Liam Collins, Bhuvesh Kumar, Neil Shah, Hamed Zamani  

**Link**: [PDF](https://arxiv.org/pdf/2604.17555)  

**Abstract**: Agentic search -- the task of training agents that iteratively reason, issue queries, and synthesize retrieved information to answer complex questions -- has achieved remarkable progress through reinforcement learning (RL). However, existing approaches such as Search-R1, treat the retrieval system as a fixed tool, optimizing only the reasoning agent while the retrieval component remains unchanged. A preliminary experiment reveals that the gap between an oracle and a fixed retrieval system reaches up to +26.8% relative F1 improvement across seven QA benchmarks, suggesting that the retrieval system is a key bottleneck in scaling agentic search performance. Motivated by this finding, we propose CoSearch, a framework that jointly trains a multi-step reasoning agent and a generative document ranking model via Group Relative Policy Optimization (GRPO). To enable effective GRPO training for the ranker -- whose inputs vary across reasoning trajectories -- we introduce a semantic grouping strategy that clusters sub-queries by token-level similarity, forming valid optimization groups without additional rollouts. We further design a composite reward combining ranking quality signals with trajectory-level outcome feedback, providing the ranker with both immediate and long-term learning signals. Experiments on seven single-hop and multi-hop QA benchmarks demonstrate consistent improvements over strong baselines, with ablation studies validating each design choice. Our results show that joint training of the reasoning agent and retrieval system is both feasible and strongly performant, pointing to a key ingredient for future search agents. 

---
# RoTRAG: Rule of Thumb Reasoning for Conversation Harm Detection with Retrieval-Augmented Generation 

**Authors**: Juhyeon Lee, Wonduk Seo, Junseo Koh, Seunghyun Lee, Haihua Chen, Yi Bu  

**Link**: [PDF](https://arxiv.org/pdf/2604.17301)  

**Abstract**: Detecting harmful content in multi turn dialogue requires reasoning over the full conversational context rather than isolated utterances. However, most existing methods rely mainly on models internal parametric knowledge, without explicit grounding in external normative principles. This often leads to inconsistent judgments in socially nuanced contexts, limited interpretability, and redundant reasoning across turns. To address this, we propose RoTRAG, a retrieval augmented framework that incorporates concise human written moral norms, called Rules of Thumb (RoTs), into LLM based harm assessment. For each turn, RoTRAG retrieves relevant RoTs from an external corpus and uses them as explicit normative evidence for turn level reasoning and final severity classification. To improve efficiency, we further introduce a lightweight binary routing classifier that decides whether a new turn requires retrieval grounded reasoning or can reuse existing context. Experiments on ProsocialDialog and Safety Reasoning Multi Turn Dialogue show that RoTRAG consistently improves both harm classification and severity estimation over competitive baselines, with an average relative gain of around 40% in F1 across benchmark datasets and an average relative reduction of 8.4% in distributional error, while reducing redundant computation without sacrificing performance. 

---
# Detecting Alarming Student Verbal Responses using Text and Audio Classifier 

**Authors**: Christopher Ormerod, Gitit Kehat  

**Link**: [PDF](https://arxiv.org/pdf/2604.16717)  

**Abstract**: This paper addresses a critical safety gap in the use Automated Verbal Response Scoring (AVRS). We present a novel hybrid framework for troubled student detection that combines a text classifier, trained to detect responses based on their content, and an audio classifier, trained to detect responses using prosodic markers. This approach overcomes key limitations of traditional AVRS systems by considering both content and prosody of responses, achieving enhanced performance in identifying potentially concerning responses. This system can expedite the review process by humans, which can be life-saving particularly when timely intervention may be crucial. 

---
# GRAB-ANNS: High-Throughput Indexing and Hybrid Search via GPU-Native Bucketing 

**Authors**: Xinkui Zhao, Hengxuan Lou, Yifan Zhang, Junjie Dai, Shuiguang Deng, Jianwei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2604.16402)  

**Abstract**: Hybrid search, which jointly optimizes vector similarity and structured predicate filtering, has become a fundamental building block for modern AI-driven systems. While recent predicate-aware ANN indices improve filtering efficiency on CPUs, their performance is increasingly constrained by limited memory bandwidth and parallelism. Although GPUs offer massive parallelism and superior memory bandwidth, directly porting CPU-centric hybrid search algorithms to GPUs leads to severe performance degradation due to architectural mismatches, including irregular memory access, branch divergence, and excessive CPU-GPU synchronization. In this paper, we present GRAB-ANNS, a high-throughput, GPU-native graph index for dynamic hybrid search. Our key insight is to rethink hybrid indexing from a hardware-first perspective. We introduce a bucket-based memory layout that transforms range predicates into lightweight bucket selection, enabling coalesced memory accesses and efficient SIMT execution. To preserve global navigability under arbitrary filters, we design a hybrid graph topology that combines dense intra-bucket local edges with sparse inter-bucket remote edges. We further develop an append-only update pipeline that supports efficient batched insertions and parallel graph maintenance on GPUs. Extensive experiments on large-scale datasets show that GRAB-ANNS achieves up to 240.1 times higher query throughput and 12.6 times faster index construction than state-of-the-art CPU-based systems, and up to 10 times higher throughput compared to optimized GPU-native reimplementations, while maintaining high recall. 

---
# CrossTraffic: An Open-Source Framework for Reproducible and Executable Transportation Analysis and Knowledge Management 

**Authors**: Rei Tamaru, Bin Ran  

**Link**: [PDF](https://arxiv.org/pdf/2604.16316)  

**Abstract**: Transportation engineering often relies on technical manuals and analytical tools for planning, design, and operations. However, the dissemination and management of these methodologies, such as those defined in the Highway Capacity Manual (HCM), remain fragmented. Computational procedures are often embedded within proprietary tools, updates are inconsistently propagated across platforms, and knowledge transfer is limited. These challenges hinder reproducibility, interoperability, and collaborative advancement in transportation analysis.
This paper introduces CrossTraffic, an open-source framework that treats transportation methodologies and regulatory knowledge as continuously deployable and verifiable software infrastructure. CrossTraffic provides an executable computational core for transportation analysis with cross-platform access through standardized interfaces. An ontology-driven knowledge graph encodes engineering rules and provenance and serves as a semantic validation layer for analytical workflows. A conversational interface further connects large language models to this validated execution environment through structured tool invocation, enabling natural-language access while preventing procedurally invalid analyses.
Experimental results show that knowledge-graph-constrained execution substantially improves numerical accuracy and methodological fidelity compared with context-only approaches, achieving near-zero numerical error (MAE<0.50) across multiple large language models and perfect detection of invalid analytical inputs in stress testing (F1~=~1.0). Its modular architecture supports the integration of additional transportation manuals and research models, providing a foundation for an open and collaborative transportation science ecosystem with a reproducible computational core. The system implementation is publicly available at this https URL. 

---
