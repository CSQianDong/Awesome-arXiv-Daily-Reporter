# GRASP: Plan-Guided Graph Retrieval with Adaptive Fusion and Reranking on Semi-Structured Knowledge Bases 

**Authors**: Yicheng Tao, Yiqun Wang, Xiangchen Song, Xin Luo, Kai Liu, Jie Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.30237)  

**Abstract**: Semi-structured knowledge bases (SKBs) embed textual documents in a typed graph of entities and relations, and underpin applications such as product search, academic paper search, and precision-medicine inquiries. Existing hybrid retrieval systems on SKBs either use the graph only for query expansion, mix textual and structural branches under a global weighting, or rely on fine-tuned graph-traversal generators. We present GRASP, a three-stage SKB retrieval framework unifying plan-based graph retrieval, plan-conditioned fusion with a dense retriever, and a fine-tuned reranker over the fused candidates. GRASP substantially advances the state of the art on every metric across the three STaRK benchmarks, lifting average Hit@1 from 62.0 to 73.9. Ablation and sensitivity studies further confirm the effectiveness and robustness of GRASP. 

---
# LexPath: A domain-oriented multi-path framework for legal article retrieval 

**Authors**: Weixuan Liu, Qingfeng Zhuge, Xuyang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.30205)  

**Abstract**: Legal article retrieval is critical for building traceable and reliable legal AI systems, where conclusions must be grounded in specific legal articles. However, existing open-domain retrieval methods rely heavily on surface-level lexical or semantic similarity, making it difficult for them to distinguish legally relevant articles from those that are textually similar but legally inapplicable or misaligned with the user's underlying intent. To bridge this gap, we propose \textsc{LexPath}, a domain-oriented multi-path framework comprising a multi-path retrieval module and an intent-aware reranking module. The retrieval module combines two complementary legal-specific paths to collect candidate articles: an IRAC-guided sparse path that expands queries with legally informative keywords, and a structure-guided dense path trained with hard negatives derived from legal hierarchy and citation relations. Then, the reranking module further refines the candidate ranking by incorporating the intent consistency score between queries and legal articles. We evaluate \textsc{LexPath} on two publicly available benchmarks focusing on general-public queries and a self-constructed benchmark targeting domain-professional scenarios. Experimental results demonstrate that \textsc{LexPath} consistently outperforms lexical, dense, hybrid, and adaptive retrieval-augmented generation (RAG) baselines. Ablation studies further verify the effectiveness of each component. 

---
# No More K-means:Single-Stage Sparse Coding for Efficient Multi-Vector Retrieval 

**Authors**: Lixuan Guo, Yifei Wang, Tiansheng Wen, Aosong Feng, Stefanie Jegelka, Chenyu You  

**Link**: [PDF](https://arxiv.org/pdf/2605.30120)  

**Abstract**: Multi-vector retrieval (MVR) models, exemplified by ColBERT, have established new benchmarks in retrieval accuracy by preserving fine-grained token-level interactions. However, this granularity imposes prohibitive storage and retrieval efficiency bottlenecks: to manage the immense memory footprint and computational overhead of billion-scale token vectors, state-of-the-art systems are forced to rely on aggressive dimension reduction and complex clustering (e.g., K-means). This compromise introduces two critical limitations: excessive indexing latency of clustering large-scale corpora and semantic information loss inherent to compression. In this paper, we propose Single-stage Sparse Retrieval (SSR}, a paradigm shift that replaces expensive clustering with efficient sparse coding. Instead of compressing features into low-dimensional dense vectors, we utilize Sparse Autoencoder (SAE) to project token embeddings into a high-dimensional but highly sparse representation. This transformation enables us to bypass vector clustering entirely and leverage inverted indexing for precise, high-throughput retrieval. Extensive experiments on the BEIR benchmark demonstrate that SSR achieves a "trifecta" of improvements: it reduces indexing time by 15x compared to ColBERTv2, halves retrieval latency, and simultaneously improves retrieval performance over leading baselines. 

---
# Uncertainty Quantification for Multimodal Retrieval Augmented Generation 

**Authors**: Simon Binz, Heydar Soudani, Faegheh Hasibi  

**Link**: [PDF](https://arxiv.org/pdf/2605.29956)  

**Abstract**: Retrieval Augmented Generation (RAG) improves the question answering capabilities of Large Language Models (LLMs) by incorporating external knowledge and has recently been extended to multimodal settings through Vision-Language Models (VLMs) that integrate visual and textual information. Despite these advances, generated answers can still be incorrect or misleading. Uncertainty Quantification (UQ) methods aim to estimate the reliability of model outputs, but most existing approaches are designed for text-only models and perform poorly in multimodal RAG scenarios. A key challenge is capturing uncertainty arising from multiple stages of the pipeline, including retrieval, visual understanding, and generation. In this work, we show that modeling uncertainty using multimodal and retrieval-aware probability signals improves estimation in multimodal RAG systems. We introduce LeMUQ, a Learnable Multimodal UQ method that analyzes token probabilities under input modifications, such as removing modalities or retrieved context. By encoding these signals as probability tokens and processing them with a finetuned model, our approach captures interactions between modalities and retrieval. Experiments across datasets, retrievers, and VLMs show consistent improvements over baseline and finetuned UQ methods. Our proposed LeMUQ increases the AUROC metric by 3.8% on average. Additionally, our method shows strong generalization performance across different retrieval setups and datasets with mixed results when transferring across different VLMs. Our findings highlight the importance of modeling multimodal uncertainty and provide a step toward more reliable and safer multimodal RAG systems. Code is available on GitHub. 

---
# Rec-Distill: An Industrial Distillation Pipeline for Large-Scale Recommendation Models 

**Authors**: Haoran Ding, Wenlin Zhao, Yuchen Jiang, Juren Li, Jie Zhu, Xinchun Li, Yishujie Zhao, Yi Zhang, Ao Qiao, Jianhui Dong, Cheng Chen, Ziyan Gong, Deping Xie, Peng Xu, Zikai Wang, Yuwei Wang, Huizhi Yang, Zhe Chen, Yuchao Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2605.29755)  

**Abstract**: Large recommendation models have demonstrated substantial potential gains under scaling laws, yet these gains are difficult to realize in industrial recommendation systems because real-world deployment requires lightweight models with strict serving efficiency and latency guarantees. This creates a fundamental gap between offline model scaling and online deployment. In this work, we present Rec-Distill, an industrial distillation pipeline that transfers the performance gains of large-scale recommendation modeling to efficient serving models. Rec-Distill combines large-teacher scaling with student-side transfer optimization through decoupled training, black-box distillation, debiasing mechanism, and a hybrid batch-streaming pipeline for dynamic recommendation environments. Across multiple recommendation and advertising scenarios on real-world platforms, our framework scales teacher models up to 24B dense parameters and 20K behavior sequence length, while enabling lightweight students to recover a substantial portion of teacher gains, with distillation transferability exceeding 60% in the best setting. Extensive offline and online experiments further show that these transferred gains consistently translate into measurable business improvements under industrial constraints. These results demonstrate that Rec-Distill provides a practical framework for distilling large-scale recommendation models into deployable, cost-efficient serving systems, while also establishing a reliable path toward scaling recommendation models to even larger regimes in the future. 

---
# FLASH-MAXSIM: IO-Aware Fused Kernels for Late-Interaction Scoring 

**Authors**: Roi Pony, Adi Raz Goldfarb, Idan Friedman, Daniel Ezer, Udi Barzelay  

**Link**: [PDF](https://arxiv.org/pdf/2605.29517)  

**Abstract**: Late-interaction retrieval (ColBERT, ColPali) scores a query against a document with the MaxSim operator: for every query token, the maximum similarity over the document tokens, summed over
query tokens. The standard implementation materializes the full query-token x document-token similarity tensor in GPU memory; for visual ColPali at 10K documents this tensor alone is 21 GB in
FP16, created only to be reduced to one score per document and discarded. It exhausts a 40 GB GPU and bounds the achievable batch size in both inference and training. We present
Flash-MaxSim, an IO-aware fused GPU kernel that computes exactly the same scores without ever materializing the tensor, by streaming query and document tiles through on-chip SRAM and folding
the row-maximum reduction into the same pass. We extend the IO-aware principle through the training backward pass, an inverse-grid CSR construction that reuses the forward argmax for an
atomic-free, destination-owned gradient reduction, and through INT8xINT8 quantization and variable-length (padding-free) scoring. Flash-MaxSim is up to 3.9x faster on an A100 (4.7x on an
H100) than naive PyTorch at matched precision, uses up to 16x less inference memory and ~28x less training memory, unlocks corpus and batch sizes that exhaust PyTorch entirely, preserves the
exact ranking (100% top-20 agreement with an FP32 reference) 

---
# Latent Terms: Dense Retrievers Contain Trivially Extractable BM25-ready Zipfian Vocabularies 

**Authors**: Benjamin Clavié, Sean Lee, Aamir Shakir, Makoto P. Kato  

**Link**: [PDF](https://arxiv.org/pdf/2605.29384)  

**Abstract**: We propose Latent Terms, a method revealing that models trained for dense retrieval, whether single- or multi-vector, learn representations that can trivially be decomposed into retrieval-ready sparse features. When trained on frozen retrievers, Sparse Autoencoders without any retrieval-specific adjustments extract a latent vocabulary with approximately Zipfian collection statistics, directly suitable for classical sparse retrieval scoring via BM25. This approach enables sparse retrieval while requiring no learned expansion objective or sparse retrieval supervision whatsoever, and can be readily applied to any dense retriever. Latent Terms is able to match or outperform single-vector scoring methods from its own base model as well as comparable SPLADE variants. In addition, it substantially outperforms its base model on LIMIT, a task specifically designed to highlight the failures of single-vector retrieval. Overall, our results highlight that neural retrievers contain more expressive and indexable structure than their default scoring functions expose, but that other methods can nonetheless be leveraged. 

---
# ACE: Anisotropy-Controllable Embedding for LLM-enhanced Sequential Recommendation 

**Authors**: Dongcheol Lee, Hye-young Kim, Jongwuk Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.29322)  

**Abstract**: Recent advances in the LLM-as-Extractor paradigm leverage large language models (LLMs) to transfer semantically rich item embeddings into sequential recommendation (SR) backbones. However, LLM-generated embeddings often suffer from strong anisotropy. Most vectors are concentrated in similar directions, resulting in a geometric imbalance that makes it difficult to adapt to collaborative signals during fine-tuning. To address this challenge, we propose Anisotropy-Controllable Embedding (ACE), which explicitly controls the anisotropy of LLM-generated embeddings. Specifically, ACE utilizes a linear autoencoder (LAE) to reshape the embedding distribution while preserving its semantic structure. In this process, the L2-regularization term mitigates the anisotropy by controlling the dispersion of embedding dimensions, while the reconstruction loss maintains semantic relationships among items. That is, ACE balances geometric uniformity and semantic embedding preservation for more stable learning. Extensive experiments demonstrate that ACE consistently outperforms existing LLM-enhanced SR models, yielding improvements of up to 12.4% and 11.8% in Recall@20 and NDCG@20, respectively. 

---
# UniNote: A Unified Embedding Model for Multimodal Representation and Ranking 

**Authors**: Jinghan Zhao, Wenwei Jin, Anqi Li, Jintao Tong, Luya Mo, Jiawei Li, Bin Li, Yao Hu  

**Link**: [PDF](https://arxiv.org/pdf/2605.29287)  

**Abstract**: Item-to-Item (I2I) retrieval is a fundamental part of modern content platforms, supporting critical industrial workflows from recommendation engines to content auditing. While multimodal embedding methods have advanced general retrieval, they often falter in I2I scenarios due to the challenges of balancing global content representation with fine-grained local retrieval, the systemic inefficiency of decoupled embedding-and-ranking pipelines, and the inherent trade-offs between model precision and serving latency. To solve these issues, we propose \textbf{UniNote}, a unified embedding model designed for industrial I2I retrieval. Tailored retrieval strategies are introduced to support representation learning over complex, multimodal content at varying granularities. To operationalize these strategies, UniNote employs a two-stage training paradigm: the first stage leverages contrastive SFT to establish robust base embeddings, while the second stage refines ranking quality through a reinforcement learning (RL) process that aligns the model with content relevance. Our results show that UniNote achieves SOTA performance across diverse I2I tasks. Deployed at Xiaohongshu and integrated with Matryoshka Representation Learning (MRL), UniNote achieved significant improvements in retrieval quality and cost efficiency in large-scale applications. 

---
# CrossAlpha: An Annual-Report Benchmark for Cross-Market Factor Research 

**Authors**: Qian Wang, Zhongyi Tong, Nuo Chen, Zhaomin Wu, Bingsheng He  

**Link**: [PDF](https://arxiv.org/pdf/2605.29286)  

**Abstract**: Cross-market factor research studies whether firm-level signals from one or more markets can predict returns in a target market, but existing public benchmarks do not support cross-market disclosure-to-return evaluation. Building such a benchmark is challenging because filings differ across languages and regulatory systems, disclosure-derived similarity can be biased by common reporting components, and cross-market signals must be evaluated under feasible trading-time alignment. We introduce \textbf{CrossAlpha}, a public annual-report benchmark for cross-market factor research. CrossAlpha addresses these challenges through three corresponding components: \emph{Disclosure Distillation}, which standardises heterogeneous filings into ten-category English business descriptions; \emph{Residual Schema Graph Construction}, which builds PCA-whitened cross-market firm-pair scores from schema-level disclosures; and \emph{Timing-Aligned Evaluation}, which pairs the graph with 11 years of daily OHLCV data to construct forward-return labels under feasible cross-market execution protocols. CrossAlpha covers about 3,600 firms and 10,700 firm-year reports from the United States, Japan, Taiwan, South Korea, and Hong Kong, and releases about 19M directed firm-pair scores. In experiments, disclosure-derived cross-market peers outperform domestic text, industry-code, and return-correlation peers in the US-to-Japan setting (ICIR 0.39 versus 0.07--0.18), and cross-market sources beat the domestic text baseline in most target markets. CrossAlpha offers an open-sourced, reusable, return-grounded benchmark for cross-market financial NLP. 

---
# On the Practice of Scaling Search Conversion Rate Prediction 

**Authors**: James Pak, Jyun-Yu Jiang, Fan Zhang, Sen Wang, Taekmin Kim, Henry Tsai, Vijay Rajaram, Juexin Lin, Mohitdeep Singh, Alessandro Magnani, Johnny Chen, Qian Zhao, Rao Fu, Zhirong Liang, Jordan Gilliland, Winter Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2605.29232)  

**Abstract**: Scaling a Search Conversion Rate (CVR) prediction model, especially in high-traffic environments, presents a challenge: superior model quality needs to be balanced with strict constraints on training cost and serving latency. This paper details an effective approach for scaling modern search CVR prediction models. We begin with an empirical study to understand the scaling performance of search CVR models, analyzing how quality improves as we scale three key factors of model backbone computation, the size of embedding parameters, and the volume of training data.
We use a large-scale production dataset, comprising over a year of customer interaction logs from a high-traffic e-commerce platform, to evaluate the scalability of several state-of-the-art architectures and their ensembles. Our key findings are: (1) selecting the right backbone and scaling factors is crucial; (2) the impact of scaling backbone, embedding, and data is largely independent and additive, which has implications for more efficient scaling exploration; (3) a streamlined warmstart strategy can accelerate training iterations while simplifying new updates; (4) inference optimization strategies such as decoupled graph execution and dynamic batching can enable low-latency GPU serving even for high-capacity models. Compared to a baseline of a pre-scaling production model, we ultimately deployed a model trained on 2.5x larger training data with 8x more inference compute while having minimal latency impact. Online A/B tests also demonstrate that our launches achieved a combined +2.6% gain in a key metric of search conversion rate. 

---
# Toward User Preference Alignment in LLM Recommendation via Explicit Context Feedback 

**Authors**: Weizhi Zhang, Wooseong Yang, Yuxin Cui, Zhaohui Guo, Hins Hu, Liangwei Yang, Henry Peng Zou, Qifei Wang, Hanqing Zeng, Jiayi Liu, Yinglong Xia, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.29141)  

**Abstract**: Traditional recommender systems (RecSys) primarily infer user preferences from implicit signals (such as clicks, watches, and purchases), often neglecting the rich explicit contextual feedback users provide through verbal text, like comments and reviews. This explicit context feedback captures the nuanced reasons behind user decisions regarding their preferences. In addition, it offers critical heterogeneous information for user preference alignment and more explainable recommendations. Overlooking such signals can lead to misaligned user preferences and further reinforce filter bubbles, as algorithms fail to understand the "semantic context" behind user choices. Recent advances in Large Language Models (LLMs) present new opportunities to harness user-generated content for more accurate and diverse recommendations, yet current LLM-based recommendations still focus on using item meta-data and underutilize this resource. In this paper, we advocate for prioritizing explicit context feedback in the next generation of LLM-based RecSys. We review the evolution of recommendation paradigms, highlight the value of context-rich feedback, call for new benchmarks and metrics, and introduce frameworks for integrating explicit user signals into scalable LLM-driven RecSys. Centering on user-preference modeling, we aim to foster more personalized, transparent, and explainable RecSys online platforms. 

---
# Generative Spatiotemporal Intent Sequence Recommendation via Implicit Reasoning in Amap 

**Authors**: Sicong Wang, Ruiting Dong, Yue Liu, Bowen Zheng, Jun Meng, Jie Li, Shuaijun Guo, Yu Gu, Fanyi Di, Xin Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.28888)  

**Abstract**: Real-world user behavior rarely consists of isolated actions; instead, it often forms intent flows governed by spatiotemporal dependencies. To provide integrated service recommendations, we focus on the task of Generative Spatiotemporal Intent Sequence Recommendation (GSISR), which aims to generate intent sequences that are logically coherent and physically executable within complex spatiotemporal contexts. While LLMs offer strong reasoning potential for GSISR, direct industrial deployment is limited by high inference latency and context-mismatched or physically infeasible plans. To address these challenges, we propose a generative framework, GPlan, that internalizes LLM reasoning into lightweight models through two components. First, to enable reasoning under strict latency constraints, we introduce Progressive Implicit CoT Distillation, which compresses explicit reasoning processes into reserved latent tokens, allowing small models to inherit complex planning logic without generating long reasoning text. Second, to address the disconnect between general knowledge and real-world constraints, we design Spatiotemporal Counterfactual DPO. By aligning the model with counterfactual context-plan pairs, we improve sensitivity to spatiotemporal context and reduce context-mismatched plans. Offline experiments and online A/B testing demonstrate that our approach improves sequence coherence and context responsiveness. Our implementation and the anonymized GSISR dataset are available at this https URL. 

---
# DocRetriever: A Plug-and-Play Framework for Multimodal Document Retrieval with Comprehensive Benchmark 

**Authors**: Ruofan Hu, Menghui Zhu, Jieming Zhu, Bo Chen, Shengyang Xu, Minjie Hong, Xiaoda Yang, Sashuai Zhou, Li Tang, Tao Jin, Zhou Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.30027)  

**Abstract**: Multimodal documents contain diverse elements, such as tables, figures, and layouts, which can complicate retrieval tasks. While current approaches typically combine dense visual embedding models with supervised rerankers to achieve high-precision retrieval, they face inherent limitations. First, the coarse-grained nature of dense embeddings tends to obfuscate explicit semantics, failing to leverage structurally salient information. Second, supervised reranking models suffer from generalization bottlenecks, as their performance heavily relies on domain-specific training data. Furthermore, existing benchmarks often lack diverse assessment dimensions and comprehensive relevance annotations, limiting reliable evaluation. To address these challenges, we propose DocRetriever, a plug-and-play framework. It enhances visual retrieval via a layout-aware sparse embedding technique, enabling effective hybrid encoding without the overhead of optical character recognition (OCR). We also introduce a generalizable reranker that leverages reasoning-augmented demonstrations and optimized sampling to improve accuracy in few-shot settings. Finally, we construct a new benchmark, MultiDocR, to enable more rigorous evaluation. Experiments across diverse benchmarks validate DocRetriever's superiority over state-of-the-art methods. 

---
# From Prompts to Context: An Ontology-Driven Framework for Human-Generative AI Collaboration 

**Authors**: Ngoc Luyen Le, Marie-Hélène Abel, Bertrand Laforge  

**Link**: [PDF](https://arxiv.org/pdf/2605.29675)  

**Abstract**: Collaborations with Generative AI often begin with a short prompt and end with an opaque output, leaving implicit who was involved, what task was being pursued, which resources were used, and which constraints should have shaped the process. This limited contextual explicitness hinders trust, traceability, and accountability, particularly when Generative AI is embedded in information-intensive workflows such as search, querying, and profile management. This paper introduces From Prompts to Context, an ontology-driven framework for representing Human-Generative AI collaboration. Its core component, the Contextual Collaboration AI Ontology (CCAI), models key elements of collaboration - including tasks, agent roles, resources, and constraints - as a shared machine-interpretable vocabulary. By combining populated CCAI instances with SPARQL-based context retrieval in operational workflows, the framework turns otherwise ephemeral prompt-response interactions into structured and queryable collaboration traces linking prompts, outputs, and their surrounding context. The approach is illustrated through a case study involving a software development team building a competency-based education feature for viewing and updating learner competency profiles. The case study shows how the framework can support the representation and documentation of collaboration episodes across requirements analysis, design, implementation, and testing. Within this setting, the results indicate that explicit collaboration modelling helps make task context more explicit, improves the traceability of AI-generated contributions, and supports more transparent and accountable Human-Generative AI practices. We conclude by outlining design principles for future Human-Generative AI systems that emphasise not only output quality, but also the explicit representation of the collaborative context in which outputs are produced. 

---
# Entity-Collision: A Stratified Protocol for Attributing Retrieval Lift in Agent Memory 

**Authors**: Youwang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2605.29630)  

**Abstract**: End-to-end agent-memory benchmarks report a single hit@k per retriever, confounding lexical leakage (uncontrolled query/gold/distractor entity overlap) with tag-mixing (preferences, services, tools averaged together). We propose entity-collision, a system-agnostic protocol that pins the BM25 floor by construction -- every distractor shares the answer's entity tokens -- and stratifies queries by discriminator tag, so any lift over BM25 is attributable to the embedder. Applied to an open-source agent-memory testbed across 5 tags x 3 embedders x 5 collision degrees with paired-bootstrap 95% CIs, the protocol reveals a two-axis pattern: a 256-d hash trigram helps only on closed-vocabulary lexical tags at deep collision; MiniLM-384 dominates both axes; and a 2.7x-parameter BGE-large does not uniformly improve on MiniLM -- it wins on intent-style queries but loses on lexical ones. Encoder capacity alone is not the binding constraint. The synthetic intent-tag null replicates on LongMemEval (n=500) as a single-session-preference recall cliff. Adaptive vector-weight routing on LoCoMo is a measured null: 11.7pp of oracle headroom exists, but no signal we tested recovers it. All 26 result tables and 37 reproduce scripts are version-controlled and verified by a public registry; the protocol is exercised on a deterministically governed memory testbed (event-sourced decision log, DAG-state-machine schema lifecycle) so every reported CI is reproducible byte-for-byte from the ingest stream. 

---
# HiKEY: Hierarchical Multimodal Retrieval for Open-Domain Document Question Answering 

**Authors**: Joongmin Shin, Gyuho Shim, Jeongbae Park, Jaehyung Seo, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2605.29606)  

**Abstract**: Retrieval-augmented generation (RAG) for document-based Open-domain Question Answering (ODQA) on large-scale industrial corpora faces two critical bottlenecks: routing failure in locating the correct document and evidence fragmentation in integrating scattered information. Existing approaches relying on flat text chunks or page-level images inherently struggle to (i) precisely pinpoint the target document among thousands of candidates and (ii) organically connect multimodal evidence, such as tables and figures, within a limited token budget. To address these challenges, we propose HiKEY, a hierarchical tree-based multimodal retrieval framework that elevates document hierarchy to a first-class retrieval signal. Instead of simple chunking, HiKEY reconstructs a logical heterogeneous graph via Document Hierarchical Parsing (DHP), explicitly encoding parent-child relationships. Adopting a hierarchical coarse-to-fine strategy, the framework (1) performs global routing to rapidly prune the search space using hierarchical indexing, and (2) conducts fine-grained retrieval to rank sections by employing a multimodal fusion strategy that captures the most discriminative evidence. Finally, HiKEY assembles a token-efficient evidence subgraph via a hybrid structural-semantic packing strategy. Experiments on ODQA benchmarks demonstrate that HiKEY significantly outperforms page- and chunk-based baselines, improving retrieval recall by up to 12.9% and end-to-end QA performance by up to 6.8%. 

---
# SCOPE: A Lightweight-training LLM Framework for Air Traffic Control Readback Monitoring 

**Authors**: Qihan Deng, Minghua Zhang, Yang Yang, Zhenyu Gao  

**Link**: [PDF](https://arxiv.org/pdf/2605.29543)  

**Abstract**: Pilot readback of Air Traffic Control (ATC) voice instructions is a primary safeguard against miscommunication in air transportation. However, readback anomalies remain implicated in approximately 80% of aviation incidents. This vulnerability is further exacerbated by rising traffic volume and elevated cognitive workload, thereby motivating automated readback monitoring by machine. Traditional rule-based and machine learning approaches struggle to generalize across the highly variable and evolving phraseology of air traffic controller-pilot communications. While Large Language Models (LLMs) have opened a new avenue through their strong reasoning and generalization capabilities, existing approaches still face deployment and computational barriers in practice. In this work, we propose Semantic reasoning for Communication via Open-set Plug-in with Examples (SCOPE), a novel lightweight-training LLM framework that advances both the efficiency and accuracy of machine-based ATC readback monitoring. The core idea is to couple a plug-in open-set classifier with a carefully designed in-context learning mechanism on top of a frozen LLM. Extensive experiments on the semi-synthetic communication dataset show that SCOPE attains superior accuracy while delivering the low-latency response required for operational environments. Under a few-shot setting, SCOPE achieves 91.05% accuracy in open-set detection and corrects 96.63% of anomalous readbacks, thereby outperforming the strongest available baselines while providing explanations for its decisions. These findings demonstrate the potential of our framework as a practical pathway toward interpretable and controllable ATC readback monitoring. 

---
# Xetrieval: Mechanistically Explaining Dense Retrieval 

**Authors**: Zhixin Cai, Jun Bai, Yang Liu, Jiaqi Li, Yichi Zhang, Taichuan Li, Zhuofan Chen, Zixia Jia, Zilong Zheng, Wenge Rong  

**Link**: [PDF](https://arxiv.org/pdf/2605.29507)  

**Abstract**: Explaining why dense retrievers assign high relevance scores remains challenging because retrieval decisions are made through opaque high-dimensional embeddings. Existing explanations often focus on surface signals, such as lexical matches, token alignments, or post-hoc textual rationales, and thus provide limited insight into the latent factors that shape dense retrieval behavior at the embedding level. We propose \textit{Xetrieval}, an embedding-level mechanistic framework for explaining dense retrieval. \textit{Xetrieval} first introduces a lightweight reasoning internalizer that approximates Chain-of-Thought reasoning directly in the embedding space with a single forward pass, enriching sentence embeddings with reasoning-oriented information while avoiding expensive autoregressive generation. It then decomposes these reasoning-enhanced embeddings into sparse, human-interpretable features, each associated with a coherent natural language description. By aggregating sparse feature overlaps across multiple document-side views, \textit{Xetrieval} provides feature-level explanations of individual retrieval decisions. Experiments on diverse retrievers and benchmarks show that \textit{Xetrieval} uncovers coherent interpretable features, yields stronger pair-level intervention effects, and supports task-level feature steering. The project page and source code are available at this https URL . 

---
# SkillBrew: Multi-Objective Curation of Skill Banks for LLM Agents 

**Authors**: Wentao Hu, Zhendong Chu, Yiming Zhang, Junda Wu, Ming Jin, Xiangyu Zhao, Yilei Shao, Yanfeng Wang, Qingsong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2605.29440)  

**Abstract**: Retrieval-augmented LLM agents increasingly rely on curated skill banks: collections of reusable textual principles that guide decision making on complex tasks. Existing approaches typically expand these banks in an append-only fashion, continuously adding new skills without removing redundant, outdated, or harmful ones, resulting in inefficient and poorly curated repositories. In this paper, we formulate the skill bank curation as a constrained multi-objective problem: a desirable bank must be useful for the agent, diverse in its content, and provide good coverage of the query distribution. To this end, we introduce SkillBrew, a multi-objective curation framework that formalizes skill bank curation as Pareto-aware optimization under a utility constraint, and solves it via a bi-level propose-then-verify loop. We evaluate our approach on two public benchmarks. Our findings suggest that treating skill banks as objects of principled curation, rather than ever-growing append-only logs, is an important step toward building self-improving LLM agents. 

---
# GrepSeek: Training Search Agents for Direct Corpus Interaction 

**Authors**: Alireza Salemi, Chang Zeng, Atharva Nijasure, Jui-Hui Chung, Razieh Rahimi, Fernando Diaz, Hamed Zamani  

**Link**: [PDF](https://arxiv.org/pdf/2605.29307)  

**Abstract**: Large Language Model (LLM) search agents have shown strong promise for knowledge-intensive language tasks through multiple rounds of reasoning and information retrieval. Most existing systems access information using a retriever that takes a keyword or natural language query and returns a ranked list of documents using an index of pre-computed document representations. In this work, we explore a complementary perspective in which the search agent treats the corpus itself as the search environment and finds evidence by issuing executable shell commands. We introduce GrepSeek, an optimized direct corpus interaction (DCI) search agent that trains a compact search agent to find, filter, and compose evidence from large text corpora. To address the instability of learning behavior directly with reinforcement learning on large corpora, we propose a two-stage training pipeline. First, we construct a cold-start dataset using an answer-aware Tutor and answer-blind Planner to generate verified, causally grounded search trajectories. Second, we refine the initialized policy with Group Relative Policy Optimization (GRPO), allowing the agent to improve its task-oriented search behavior through direct interaction with the corpus. To make DCI practical at scale, we further use a semantics-preserving sharded-parallel execution engine that accelerates shell-based retrieval by up to $7.6\times$ while preserving byte-exact equivalence with sequential execution of the shell command. Experiments across seven open-domain question answering benchmarks show that GrepSeek achieves the strongest overall token-level $F_1$ and Exact Match. Our analysis also highlights the limitations of purely lexical interaction on queries with substantial surface-form variation, suggesting DCI as a practical and competitive method for search agents that can complement existing retrieval paradigms in the real world. 

---
# LoopFM: Learning frOm HistOrical RePresentations of Foundation Model for Recommendation 

**Authors**: Shali Jiang, Hua Zheng, Boyang Liu, Laming Chen, Kenny Lov, Chuanqi Xu, Lisang Ding, Qinghai Zhou, Can Cui, Xiaolong Liu, Xiaoyi Liu, Yasmine Badr, Xin Xu, Jiyan Yang, Ellie Dingqiao Wen, Gerard Jonathan Mugisha Akkerhuis, Chenxiao Guan, Rong Jin, Ruichao Qiu, Xian Chen, Shifu Xu, Zhehui Zhou, Ping Chen, Rui Yang, Haicheng Chen, Xiangge Meng, Song Zhou, Dharak Kharod, Shuyu Xu, Qiang Jin, Qiao Yang, Wankun Zhu, Qin Huang, Yuzhen Huang, Darren Liu, Parish Aggarwal, Hui Zhou, Erzhuo Wang, Shuo Chang, Xiaorui Gan, Wenlin Chen, Santanu Kolay, Huayu Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.29280)  

**Abstract**: Knowledge distillation (KD) transfers a single scalar prediction from a large foundation model (FM) to compact vertical models (VMs), suffering from diminishing transfer ratio -- the fraction of FM improvement captured by the VM -- as a single scalar cannot convey the rich intermediate knowledge that larger FMs learn. To address this bottleneck, we propose LoopFM (Learning frOm HistOrical ReP*resentations of FM), a framework that opens a high-bandwidth transfer channel by structuring FM intermediate embeddings as input features (e.g., user history sequence) for downstream VMs, without requiring real-time FM inference at serving and architectural coupling between FM and VM. We provide a theoretical framework for LoopFM with a gain decomposition and transfer-ratio analysis. On three public benchmarks, LoopFM demonstrates strong AUC improvements (e.g., 6\%+ on TaobaoAd) and complementary knowledge transfer capability with KD. On industrial-scale systems (billions of examples, trillion-parameter FMs), LoopFM approximately doubles the knowledge transfer ratio on top of KD, delivering a +0.5\% conversion improvement in Y1H1, and a +1.03\% and +1.22\% conversion improvement from two individual launches respectively in Y1H2. 

---
# CoHyDE: Iterative Co-Training of LLM Rewriter & Dense Encoder for Tool Retrieval 

**Authors**: Vaishali Senthil, Ashutosh Hathidara, Sebastian Schreiber  

**Link**: [PDF](https://arxiv.org/pdf/2605.29271)  

**Abstract**: Tool retrieval over large API catalogs is a core bottleneck for LLM agents: user queries arrive in colloquial, often underspecified language, while the catalog uses technical API vocabulary that no fixed encoder can bridge on its own. The two dominant training approaches, contrastive encoder fine-tuning and HyDE-style query expansion with a frozen LLM, address this problem from opposite ends and fail in complementary directions: the fine-tuned encoder excels when the query's surface form already matches the catalog but collapses when it does not, while zero-shot HyDE is more robust to underspecified queries yet generates catalog-unaware hypothetical descriptions that degrade retrieval when queries are well-formed. We introduce CoHyDE, an iterative procedure that trains the dense encoder and the LLM rewriter as a single co-evolving system: the encoder is retrained with InfoNCE on catalog-style hypothetical descriptions produced by the rewriter, and the rewriter is preference-aligned via DPO against the encoder's retrieval scores, with both sides warm-started on the tool catalog before the loop begins. On a ~10k tool subset of the ToolBench catalog, three rounds of CoHyDE improve over the strongest single-component baseline by +2.5 pp NDCG@5 on standard queries and +6.3 pp on held-out vague queries, with gains as large as +8 pp on the hardest vague tier. Ablations confirm that co-training is the key ingredient: using either component in isolation fails to match CoHyDE on both well-formed and vague queries, with losses of up to -8 pp on vague queries. 

---
# OmniRetrieval: Unified Retrieval across Heterogeneous Knowledge Sources 

**Authors**: Jinheon Baek, Soyeong Jeong, Sangwoo Park, Woongyeong Yeo, Minki Kang, Patara Trirat, Heejun Lee, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2605.29250)  

**Abstract**: Real-world information needs require access to structurally diverse knowledge sources, from unstructured text and relational tables to knowledge graphs and property graphs. Existing retrievers, however, operate over one source at a time under a fixed query language, leaving the broader landscape of available knowledge fragmented behind incompatible interfaces. A natural attempt at unification would collapse these sources into a shared space, but this erases the structural affordances (such as schemas, ontologies, compositional operators) that give each source its expressive power. Effective retrieval over diverse knowledge, therefore, requires not homogenization but an overarching layer that meets each source on its own terms. To achieve this, we present OmniRetrieval, a framework that takes any natural-language query, identifies appropriate knowledge sources, and dispatches source-native queries to their native execution engines. Across an extensive benchmark spanning 13 datasets and 309 distinct knowledge bases over text, relational, and graph-structured sources, OmniRetrieval exceeds single-source baselines, demonstrating that it can serve as a general-purpose interface to the heterogeneous sources while preserving the structural distinctions that make each source valuable. 

---
# Surfacing Isolated Learners with Outcome-Independent Mediation of Feedback between Teachers and Students Using AI 

**Authors**: Junsoo Park, Youssef Medhat, Htet Phyo Wai, Ploy Thajchayapong, Ashok K. Goel  

**Link**: [PDF](https://arxiv.org/pdf/2605.29240)  

**Abstract**: AI-augmented classrooms generate rich teacher and student feedback before graded outcomes become available, yet these signals can be difficult to translate into timely instructional decisions. We propose an interpretable decision layer: a transparent mechanism that ranks course topics requiring attention without using grades or post-hoc outcome labels. The approach combines three signals: student learning difficulty prevalence, disagreement between learner self-reports and observed difficulties, and unresolved teacher concerns. The output is a ranked set of topic priorities with per-topic decision records explaining each ranking. In one graduate CS course offering ($n=5$ instructor interviews; $n=279$ survey responses), prioritized topics aligned with instructor concerns (top-5 overlap 3/5; Spearman $\rho=0.80$) and student-reported topic difficulty ($\rho=0.46$, $p=.048$). Multi-signal integration also surfaced learners not identified through individual signal sources alone (AUC $=0.96$ vs. $0.91$ for gap prevalence alone). Reflective thinking, help-seeking, and self-efficacy provided additional evidence that student behavioral signals align with learning-related constructs. While preliminary, these findings suggest that transparent coordination mechanisms may help support human-AI co-agency when feedback is incomplete. 

---
# Rethinking Literature Search Evaluation: Deep Research Helps, and Human Citation Lists Are Not a Ground Truth 

**Authors**: Gaurav Sahu, Laurent Charlin, Christopher Pal  

**Link**: [PDF](https://arxiv.org/pdf/2605.29234)  

**Abstract**: We study large-scale literature search from two complementary angles: improving the retrieval pipeline, and stress-testing the human reference list as an evaluation target. First, we implement a Deep Research pipeline that processes the full query paper and expands the retrieved results breadth-first along their bibliographies, and show that it substantially outperforms vanilla API-only search, raising recall on RollingEval-Jun25 (a 250-paper literature-search benchmark) from below 20% to above 80%. Second, we use a neutral LLM-as-a-judge to determine if human references are sound ground truth for the task. We find significant limitations: only 51% of human citations are judged moderately relevant or higher, against 86--88% for the strongest AI-based re-rankers. We study this gap on the OpenAlex co-authorship graph, finding that humans are 2.5x more likely than the best AI re-rankers to cite a direct collaborator. Together, our results argue against single-axis literature-search evaluation: recall, topical-relevance scoring, ranked-list diversity, and a co-authorship-distance diagnostic each measure complementary properties of citation quality and should be reported jointly. 

---
# PROTOCOL: Late Interaction Retrieval for Protein Homolog Search 

**Authors**: Gabrielle Cohn, Rohan Gumaste, Minh Hoang, Vihan Lakshman  

**Link**: [PDF](https://arxiv.org/pdf/2605.29158)  

**Abstract**: Protein homology search underlies function annotation, structure prediction, and evolutionary analysis, but remains challenging in the "twilight zone," where global sequence similarity is weak and classical alignment methods lose sensitivity. Protein language models provide context-aware representations that could improve alignment sensitivity in this regime. However, prior protein embedding-based retrieval pipelines often pool these representations into a single vector, potentially obscuring local motifs, domains, or conserved residues that reveal remote homology. We introduce ProtoCol, a model which represents proteins as sets of residue embeddings and uses ColBERT-style late interaction to test whether residue-level comparison improves homolog retrieval. ProtoCol encodes proteins independently, keeps candidate representations pre-computable, and scores candidates with MaxSim over residue embeddings. On SCOPe superfamily and Pfam clan benchmarks, ProtoCol outperforms sequence-composition, alignment-based, pooled PLM, and trained single-vector baselines, supporting late interaction as an effective retrieval layer for remote homology search. 

---
# Same Question, Different Source, Different Answer: Auditing Source-Dependence in Medical Multi-Source RAG 

**Authors**: Yubo Li, Rema Padman, Ramayya Krishnan  

**Link**: [PDF](https://arxiv.org/pdf/2605.29084)  

**Abstract**: A retrieval-augmented generation (RAG) system deployed over a multi-author institutional corpus can give a different answer to the same question depending on which source it retrieves -- a failure mode the dominant single-gold-answer paradigm cannot diagnose. We argue that source-dependence is a missing axis of NLP evaluation, and that auditing it means shifting the unit of evaluation from answer correctness to the inter-source relationship. We make this concrete in transplant patient education, where institutional sources demonstrably disagree, releasing three artefacts: TransplantQA, a benchmark of real patient questions, each answered by grounding generation in multiple institutional handbooks as candidate sources; HERO-QA, a hierarchical retrieval strategy that grounds and audits each answer; and a structured-output judge that scores inter-source relationships on a validated 5-label taxonomy. At scale, better retrieval reveals far more disagreement than prior estimates suggested -- understating its prevalence, not its intensity. The framework is domain-agnostic and transfers to legal and educational RAG: measuring source-dependence is a responsibility for deployed multi-source NLP generally. 

---
# When LLM Reward Design Fails: Diagnostic-Driven Refinement for Sparse Structured RL 

**Authors**: Youting Wang, Yuan Tang, Bowen Liu, Xuan Liu, Dingyan Shang  

**Link**: [PDF](https://arxiv.org/pdf/2605.28918)  

**Abstract**: For sparse, structured reinforcement-learning tasks with semantic reward-function interfaces, LLM-generated reward shaping is better framed as debugging than one-shot generation. We study PPO-trained agents using MiniGrid as core evaluation and MuJoCo as boundary stress test. Our audit finds two dominant one-shot failure modes -- reward flooding and semantic/API misunderstanding -- plus a rarer weak-shaping case. We propose diagnostic-driven iterative refinement, where training diagnostics and a failure-mode taxonomy guide targeted reward-function revision. Refinement improves DoorKey-8x8 from 2.3% to 97.6% and KeyCorridor from 31.2% to 86.7% with high seed-to-seed variance. Controls show these gains are not from retrying or extra training: metrics-only re-prompting yields large drops, while a static-vocabulary control recovers much of the gap (87.6%; 70.7%), showing the taxonomy prompt is a major mechanism and dynamic labels provide only partially isolated incremental evidence. Budget-matched and Best-of-3 comparisons separate refinement from selection and training-time effects. Component-removal tests, sensitivity analyses, and an audit against author labels provide converging evidence for the debugging interpretation while revealing calibration limits. Continuous-control results show the boundary: success-based diagnostics can misfire in dense-reward locomotion, and return-trend feedback removes one false-positive mechanism without robust gains. The low-call protocol is a cost contrast with population-based reward search, not a benchmark comparison. In four crossed-variance-design environments, point estimates suggest larger gains when LLM reward-function variance dominates but bootstrap intervals are wide. The method is bounded to sparse structured tasks with reliable interfaces under PPO; fields like event_text may help, hurt, or be neutral. 

---
