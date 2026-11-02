# Recommendation Feedback Loops Shape Urban Income Segregation and Venue Visitation 

**Authors**: Guangyuan Weng, Lin Chen, Ciro Beneduce, Giovanni Mauro, Massimiliano Luca, Takahiro Yabe, Bruno Lepri, Luca Pappalardo, Esteban Moro  

**Link**: [PDF](https://arxiv.org/pdf/2610.41911)  

**Abstract**: Location-based recommender systems shape where people go in cities and learn from the visits they steer, forming a human-AI feedback loop. This feedback loop shapes whom people share places with across income groups and which venues keep attracting customers. Yet platforms judge these systems by their offline accuracy on past visits, a test in which the suggestions never change where people go and which therefore cannot capture long-term effects. Studying these effects in cities is challenging because, unlike online resources, travel constraints and limited venue capacity shape where people can go, while each recommended visit also changes whom other visitors encounter. To capture such features, we build a long-term simulation framework based on fine-grained human mobility data from four U.S. metropolitan areas, with distance and capacity constraints. Focusing on food venues, we study two urban consequences: how unevenly people's exposure is distributed across income groups (experienced income segregation) and the share of venues receiving no visits from simulated residents (venue disappearance). We find that, under every algorithm and in every metropolitan area, at least a fifth of venues stop receiving visits, and that more than half of this loss arises from retraining. More importantly, more accurate algorithms leave people more segregated, and the most accurate algorithm leaves people 15-21% more segregated than they actually are. We then test a re-ranking intervention that favours less popular venues and runs on top of a platform's existing algorithm. It lowers both venue disappearance and segregation across algorithms, largely without directing people to venues where they may feel less comfortable. Our results suggest that impact assessments of recommender systems should account for the feedback loop and report segregation and venue disappearance beyond accuracy. 

---
# Completeness and Cost of Retrieval Structures for Multi-Tool Requests in LLM Agents 

**Authors**: Shuyang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.41905)  

**Abstract**: An agent that must call several tools to fulfill a request fails when the tool retriever omits any one of them, yet tool retrievers are mostly built and evaluated on queries requiring a single tool. We compare representatives of six retrieval structures (flat lexical and dense retrieval, LLM query decomposition, LLM-navigated hierarchies, server-then-tool matching, dependency graphs, and trained set-level retrievers) across two multi-tool testbeds: 345 AppWorld queries with expert dependency labels and 103 MCP-Bench tasks over 257 MCP tools. We measure completeness@k, the share of queries whose entire required tool set appears in the top k, alongside the LLM calls, tokens, and latency that each structure incurs at retrieval time. A motivating analysis on all 7,961 ToolRet queries shows that aggregate completeness mixes two regimes: completeness at 10 is .58 to .67 on single-target queries and .08 to .17 on queries with four or more targets, and the ordering of retrievers on single-target queries reverses on multi-target ones. Across the two testbeds, none of the compared structures that is untrained on the testbed and makes at most one LLM call per query completes more than 10% of AppWorld queries at k = 16. The two structures that do better incur different costs: a set-level retriever trained on 75 to 90 in-domain queries reaches .33 on AppWorld and .58 on MCP-Bench with no LLM calls, while an LLM-navigated hierarchy reaches .30 and .40 with a median of 31 to 38 LLM calls and about five minutes per query. Their failure modes also differ: the trained retriever mostly ranks the missing tools just below the cutoff, whereas the hierarchy finds the target API and stops before collecting its prerequisites. In a downstream check with the official AppWorld agent, tasks with a complete retrieved set succeeded 21 of 21 times, whereas incomplete ones succeeded 2 of 39 times. 

---
# Search When Needed, Proceed When Reliable: Metacognitive Reliability Supervision for Efficient Agentic Search 

**Authors**: Minhao Sun, Zhijing Wu, Xian-Ling Mao  

**Link**: [PDF](https://arxiv.org/pdf/2610.41738)  

**Abstract**: Efficient agentic search requires large language models (LLMs) to determine whether they can reliably proceed with their current knowledge or need additional external evidence. Supervision based on final-answer correctness or trajectory-level feedback evaluates realized trajectory performance. However, observed trajectory performance is jointly shaped by stochastic generation, retrieval quality, and downstream reasoning. Such signals therefore provide only indirect evidence of how reliable the current decision state was and how retrieved evidence changed its reliability. An effective reward should encourage search when external information is needed and favor searches that contribute to the answer. To this end, we propose a metacognitive framework for training LLMs to make search decisions based on the reliability of their intermediate reasoning states. Our framework estimates the reliability of local conclusions to distinguish whether the current information supports proceeding without search from how much retrieved evidence improves that reliability. The current-state reliability score assesses whether the model can proceed without search. The search-action score measures how retrieved evidence changes estimated reliability. Together, these scores define a reliability-aware process reward for Group Relative Policy Optimization (GRPO), providing finer-grained supervision for search and no-search behavior. Experiments across seven question-answering benchmarks demonstrate that our method achieves a better accuracy-efficiency trade-off than strong agentic-search baselines. Our code and model checkpoints will be open-sourced. 

---
# Towards Trustworthy Multi-stakeholder Recommender Systems 

**Authors**: Katarina Marcincinova, Adrian Gavornik, Matus Mesarcik, Michal Kompan  

**Link**: [PDF](https://arxiv.org/pdf/2610.41674)  

**Abstract**: Recommender system (RS) nowadays has to reflect, combine, and solve often contrasting requirements and expectations from dozens of stakeholders. In fact, the problem is even more complicated as each of the stakeholders can simultaneously have various objectives creating a diverse and complicated environment with several stakeholders and their objectives in which a RS operates. To address this problems the Multi-objective and Multi-stakeholder RS have been studied in the literature. However, a limited attention has been paid to trustworthiness aspects of such RS, which slowly becomes a new standard. In this paper, we highlight open challenges and important questions which need to be addressed on the way to trustworthy Multi-stakeholder RS. 

---
# KITE: Retrieval as a Decision Procedure for Long-Term Conversational Memory 

**Authors**: Feng Li, Kailun Tan, Zhibo Zhou, Youxiang Wang, Lejun Liu, Huaigu Li, Fei Du, Huilong An  

**Link**: [PDF](https://arxiv.org/pdf/2610.41521)  

**Abstract**: Many long-term memory systems for LLM agents are built around embedding-based retrieval. Al- though effective for semantic matching, similarity ranking alone does not establish whether re- trieved memories satisfy the conditions implied by a question. In this paper, we introduce KITE (Knowledge-Indexed Temporal Evidence), a vector-free framework that formulates memory re- trieval as an explicit decision procedure. KITE organizes multi-session conversations as structured knowledge indexed by a governed topic vocabulary, which evolves through controlled expansion and consolidation. At query time, questions are compiled into inspectable retrieval plans over the same knowledge index that guides memory construction. The plans are then executed through fixed lookup and ranking operations, while sparse lexical retrieval recovers surface details omitted from the structured representation. Experiments on LoCoMo and LongMemEval-S show that KITE achieves 93.5% and 86.0% accuracy while using only about 1.6k reader-context tokens per ques- tion. These results demonstrate that conversational memory can move beyond similarity ranking by unifying memory construction and query compilation around a governed index, turning retrieval into an inspectable decision procedure for determining which evidence satisfies the question. 

---
# How BERT-based Cross-Encoders Model Relevance in Information Retrieval? 

**Authors**: Mathias Vast, Basile Van Cooten, Laure Soulier, Benjamin Piwowarski  

**Link**: [PDF](https://arxiv.org/pdf/2610.41425)  

**Abstract**: Cross-encoders are the most effective but also the least efficient models in Information Retrieval. Despite their success, the mechanisms underlying their relevance predictions remain poorly understood, particularly how they process to match queries and documents. This lack of interpretability limits their optimization, can lead to suboptimal retrieval, especially out-of-domain, and raises efficiency concerns for over-parameterized models. To address these challenges, this paper investigates two under-explored aspects of cross-encoder behavior. First, to understand the relevance prediction process, we trace the flow of information at the different stages of the process and between different input segments. We study the nature of the signals that compose this flow and that allow cross-encoders to predict the relevance of a passage to a query. Our work particularly focuses on the matching performed by the cross-encoders between queries and documents. Second, we explore how cross-encoders behave with out-of-domain documents and/or queries. Through a series of interpretability experiments, we evaluate three BERT-based cross-encoder models, varying in size and pre-training strategies, across both in-domain and out-of-domain datasets. Overall, this study both consolidates what is known about how cross-encoders model relevance and extends it by exploring the role of matching signals. 

---
# ProSearch: Benchmarking Multi-Constraint Protocol Retrieval in Experimental Science 

**Authors**: Wenliang Liang, Zhiyuan Ning, Haolin Chen, Yuanchun Zhou, Wenjuan Cui, Yi Du  

**Link**: [PDF](https://arxiv.org/pdf/2610.41396)  

**Abstract**: In AI for Science scenarios, experimental design and method reproduction require retrieving protocols that satisfy multiple interdependent constraints, yet existing benchmarks lack systematic evaluation of this capability. To address this gap, we introduce ProSearch, a benchmark for multi-constraint protocol retrieval in experimental science. ProSearch has three key features: (1) realism, with queries grounded in real experimental records and expert-defined templates; (2) complexity, with 60 experimental fields and their dependencies; and (3) diagnosticity, with 3-5 explicit constraints per query, including numerical and negation constraints. We construct ProSearch through LLM-based extraction, data-driven constraint selection, controlled query generation, and evidence-based relevance validation. The resulting benchmark contains 1,515 queries across six experimental science domains. We conduct extensive experiments on ProSearch with 19 representative retrievers and complementary analyses. The best-performing retriever achieves an average nDCG@10 of only 0.290, which improves to 0.474 with reranking. Structured diagnostic analysis shows that explicit constraint matching provides a strong ranking signal, while manual error analysis identifies joint constraint binding as a key bottleneck. ProSearch provides a testbed for systematically evaluating scientific retrieval systems. 

---
# BrowseFlow: Sparse Event Flow Matching for Generative User Simulation 

**Authors**: Yitian Pang, Jiayi Zhang, Zhenjia Ma, Xiaowei Yue, Dagui Chen, Jian Xu  

**Link**: [PDF](https://arxiv.org/pdf/2610.41374)  

**Abstract**: User simulation aims to reproduce how users interact with content and products across online platforms. Recent generative approaches have enriched user simulation with behavioral history and contextual information. However, realistic browsing simulation still faces two challenges. (i) Stochastic Responses: Even with the same recommendation feed, the same user may produce different browsing responses, yielding distinct behavioral trajectories. A logged trajectory records only one such realization. (ii) Sparse Key Events: Clicks and likes provide important feedback but occur sparsely within browsing trajectories dominated by progression actions such as scrolling. Many stepwise simulators predict these actions in a dense sequence, where key events contribute relatively few training targets. To address these challenges, we propose BrowseFlow, a conditional discrete flow matching method for generative user simulation. For Sparse Key Events, BrowseFlow represents each response by its browsing extent and an ordered sequence of key events, directly modeling their counts and locations under exposure constraints. BrowseFlow learns the conditional distribution of complete browsing responses through probabilistic extent prediction and discrete flow matching over key events. Random event retention defines probability paths over partial event sequences, while transition rate matching learns stochastic insertion dynamics. For Stochastic Responses, we use the energy score, a proper scoring rule, to evaluate response fidelity and generated variability, complemented by the Wasserstein distance between key event count distributions. We further introduce TaobaoBrowse, a benchmark pairing real recommendation contexts with fine-grained browsing trajectories. Experiments across three datasets demonstrate that BrowseFlow consistently achieves the best distributional performance. 

---
# Defining Page-Centric Visual Document Retrieval and Evaluating Existing Benchmarks 

**Authors**: Chongzhi Zhang, Yuqi Zuo, Aixin Sun  

**Link**: [PDF](https://arxiv.org/pdf/2610.41360)  

**Abstract**: In modern information retrieval (IR), documents are increasingly encountered as structured sequences of visually rich, paginated units, rather than as flat text. Example documents encountered in our daily life include Word files, PDFs, and slide decks. Free text-centric retrieval systems either ignore this structure entirely or destroy it through arbitrary passage chunking. In this paper, we formulate page-centric Visual Document Retrieval (VDR) with pages as the fundamental indexing units and document context available for assessing page relevance. The formulation distinguishes two retrieval granularities, page and document, and three retrieval paradigms: independent-item, evidence-set, and iterative retrieval. We then assess 42 benchmarks using eight criteria covering corpus characteristics, query construction, relevance annotation, and ranking support. Our analysis shows two main findings on current VDR research. First, evaluation remains concentrated on page-level independent-item retrieval, with fewer resources supporting document-level, evidence-set, and iterative settings. Second, queries originating from real user information needs independently of target corpus content and graded relevance judgments are uncommon, while the coverage and verification of relevance judgments vary substantially. Both findings reveal gaps between the capabilities applications require and those existing evaluation protocols can establish. We provide benchmark selection guidance and identify priorities for future benchmark design, emphasizing alignment between task requirements, relevance semantics, and evaluation protocols. 

---
# Pix2Evidence: A Multimodal RAG System for Text and Visual Evidence 

**Authors**: Ziniu Liu, Xiaoyang Chen, Dezhi Ye, Jie Liu, Jin Ma, Hao Deng  

**Link**: [PDF](https://arxiv.org/pdf/2610.41335)  

**Abstract**: Retrieval-augmented generation (RAG) over visually rich documents must find both relevant content and the evidence that supports an answer. Screenshot retrieval preserves images and page structure, but written facts within screenshots also need to be searchable. Moreover, topical relevance alone does not identify which retrieved tile contains evidence for the answer. We introduce Pix2Evidence, a multimodal RAG system that encodes each screenshot tile together with its available transcription into a single fused image--text representation, so that layout and readable text are considered jointly during retrieval. To guide evidence selection, a vision-language teacher describes the visible content that supports each training question. We use these descriptions to adapt the retriever, then rerank the retrieved candidates before answer generation. Extensive experiments demonstrate that Pix2Evidence outperforms state-of-the-art methods. Across five widely-used Wikipedia QA benchmarks, the non-reranked Pix2Evidence leads the main comparison in answer accuracy on four. On MMSearch, it reaches 26.0\%, 4.4 percentage points above PixelRAG. We will release the code and share derived data where licensing and privacy reviews permit. 

---
# MILES-Rec: Uncovering Implicit Intents from Mobility Behaviors for Cross-Domain POI Recommendation 

**Authors**: Yi Zou, Wei Luo, Xuewen Zhang, Zhuang Yan, Pengjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.41319)  

**Abstract**: Map applications bridge everyday mobility with local services, creating opportunities to recommend places by uncovering users' implicit intents. However, utility-driven map interactions provide only indirect and incomplete evidence of consumption demands, making it challenging to infer personalized service needs from observed behaviors. We propose MILES-Rec, a cross-domain POI recommendation framework that infers latent service intents from heterogeneous map behaviors and expresses them as prioritized natural language queries. It first learns from progressive reasoning demonstrations that synthesize user profiles and map histories for situation understanding, intent expansion, and query prioritization. We then introduce intent-aligned group relative policy optimization, using subsequent user interactions to reward query relevance and ranked service coverage while penalizing redundancy and excessive response length. A two-level credit assignment mechanism integrates shared list-level advantages with query-specific advantages, coordinating individual query quality with the complementary coverage of the complete query list. We further construct MILES-Bench from real user interaction logs on a large-scale mapping platform, supporting training and evaluation across diverse services under rich and sparse behavioral histories. Extensive experiments against general-purpose LLMs and specialized recommendation baselines demonstrate superior POI retrieval utility and strong service and semantic alignment, with consistent gains from behavioral policy optimization across four backbone scales. 

---
# Context as Infrastructure: Knowledge Compilation for Retrieval in Agentic Systems with Harnesses 

**Authors**: Zhichang Yu, Baining Cao, Sbaaoui Idriss, Qi Wang, Gangqiang Xu, Ling Qin, Xiaoling Jin, An Liu, Shaohao Cai, Xiaoyang Bao, Shengle Chen, Changchang Ding, Pengfei Zhu, Jingxiang Xu, Yifeng Mao, Lu Liu, Hai Jin, Yingfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.41172)  

**Abstract**: Retrieval-augmented generation is becoming agentic. A search harness writes queries, reads results, reformulates, and decides when to stop. This largely solves accuracy: a harness with only BM25/grep and MiniMax M3 answers 85% of BrowseComp-Plus and 91% on multi-hop FRAMES in our runs, without compiled or vector indexes. It exposes a context problem: an undisciplined loop repeats queries, rereads documents, and fills the window with unused passages. The missing layer is knowledge compilation: a configurable ingestion pipeline where LLMs enrich documents with context no chunk contains alone--navigation hierarchies with claim-evidence leaves, entity graphs, timelines, mind maps, ontologies, wikis, and memory. Compilation is not a better retriever: under a competent harness, structural artifacts gave no detectable accuracy gain on FRAMES, and a PageIndex-style outline degraded at scale. It is a supply layer. A navigation tree whose leaves are claim-evidence pairs--the mechanism we measure--cuts query tokens by 91% when evidence suffices and adds 1.9% when it does not, a 38% average reduction; this implies a 2% break-even sufficiency rate and a measured rate near 43%. A decision gate cuts latency from 140 s to 30 s, though substitution's token saving is bought with retrieval rounds and carries its own latency cost. Other artifacts--ontology, graph, wiki, timeline, mind map, memory--pay off in deployment (governance, integration, onboarding, temporal orientation, continuity) where accuracy benchmarks cannot see them. We describe RAGFlow's pipeline and harness (open source at this https URL), which turns the loop into five disciplined stations and keeps the context entering the model the smallest sufficient set; together they form an agent-native database: a store that hands the agent compiled, governed context instead of a ranked list of passages. 

---
# ProdCrawl: A Replayable Test Collection for Focused Crawling 

**Authors**: Manuel Rey-Escobar, David Otero, Javier Parapar  

**Link**: [PDF](https://arxiv.org/pdf/2610.40855)  

**Abstract**: A focused crawler searches for web pages on a specific topic under a constrained download budget. Since it can only fetch a limited number of pages, its success depends on choosing which links to follow. Focused crawlers are usually evaluated on live websites by harvest rate, the fraction of downloaded pages that are relevant. This has two problems: on the one hand, websites change over time, so a crawl cannot be repeated under the same conditions; on the other hand, there is no ground truth for which pages are relevant. As a result, comparisons between crawlers are unreliable. We present ProdCrawl, a replayable test collection for focused crawling. It comprises static mirrors of 621 company and organization websites, about 1.1 million pages, with each page labeled as a product page or non-product page based on a majority vote among three large language models. Any crawler can be run on exactly the same sites and scored against the same labels. We use ProdCrawl to compare simple baselines (breadth-first, depth-first and text classifiers) with an approach that scores each link by searching for similar link texts whose target pages have already been labeled. Our results show that text classifiers and keyword matching do not significantly outperform breadth-first crawling, whereas neural embeddings, which compare link texts by semantic meaning, outperform all simple strategies. Because every run evaluates over identical pages and labels, these results are exactly reproducible. The collection, codebase, and models are publicly released to support experimental replication and fair crawler comparison. 

---
# Retrieve-then-Disambiguate: Scalable Reviewer Discovery 

**Authors**: Vicente Amado Olivo, Joe Petrecca, Wolfgang E. Kerzendorf  

**Link**: [PDF](https://arxiv.org/pdf/2610.40768)  

**Abstract**: The growing number of scientific submissions is straining the peer review process, while a large pool of relevant researchers remains unidentified and is therefore not considered as potential reviewers. We formulate the discovery of these researchers as a retrieval problem: given a submission, we retrieve and rank researchers based on domain similarity. We present retrieve-then-disambiguate, an open-source algorithm that identifies domain relevant researchers for a query submission by retrieving similar publications from the scientific literature and resolving ambiguous author names into unique researcher identities. We evaluate the algorithm on 500 astronomy and physics publications. We use ORCID-linked first authors, coauthors, and cited researchers as proxy labels to evaluate the recovery of researchers with known connections to each query publication. When retrieving the top 500 most similar papers, the algorithm recovers roughly 68\% of first authors, 60\% of coauthors, and 45\% of cited researchers. Our code is available at this https URL. 

---
# Explain, Then Forget: Recommendation Unlearning under Asymmetric Access Constraint 

**Authors**: Haichao Zhang, Chong Zhang, Wen Wang, Shi Qiu, Peiyu Hu, Jia Wang  

**Link**: [PDF](https://arxiv.org/pdf/2610.40658)  

**Abstract**: Recommendation unlearning aims to remove the influence of specific historical interactions that should no longer affect future recommendations, such as noisy clicks or outdated user preferences. However, performing recommendation unlearning becomes much harder in modern Machine-Learning-as-a-Service (MLaaS) architectures, where the recommendation model is remotely hosted by a model provider (MP) and accessed by service operators (SO) only through APIs. In this setting, a service operator serves end users but cannot access or modify internal model parameters. As a result, existing unlearning methods are difficult to apply because they require access to model internals for retraining, fine-tuning, or other parameter updates. To address these challenges, we propose Causal Explanation-based Unlearning (CEU), a novel unlearning framework tailored for MLaaS environments. Instead of modifying internal model weights, CEU operates through a two-step output-level score intervention. First, it estimates how the forgotten interaction causally contributes to the current recommendation. Second, guided by this causal signal, the system removes the influence of the forgotten interaction by adjusting the final ranking scores of affected items. Because CEU leaves model parameters untouched, it naturally supports continuous unlearning requests while avoiding the error accumulation that can result from repeated parameter updates. Extensive experiments on three public datasets show that CEU achieves unlearning effectiveness comparable to full retraining while preserving recommendation utility and significantly reducing computational cost. The source code is available at: this https URL. 

---
# Peeling the Matryoshka: Extracting Ontological Structure from Retrieval Embeddings 

**Authors**: Eunchan Kim, Inwhee Joe  

**Link**: [PDF](https://arxiv.org/pdf/2610.41785)  

**Abstract**: Taxonomy construction is essential for knowledge-intensive NLP yet remains costly: clustering-based methods (i) need careful per-corpus tuning and yield unstable hierarchies, while LLM-based pipelines (ii) depend on expensive task-specific training that does not transfer across domains. We report an empirical finding that bears on both: in Matryoshka Representation Learning (MRL) embeddings, where each leading prefix is itself a usable, coarser vector, prefix length aligns with the coarse-to-fine granularity of gold taxonomies; shuffled- and reverse-dimension controls isolate the alignment to the learned early prefix dimensions. The pattern holds across three architecturally distinct MRL models. To test whether this regularity is usable, we introduce PEEL (Peeling Embeddings to Extract Latent ontologies): a minimal probe over these frozen embeddings, trained once with an order-embedding objective that represents hierarchy as vector containment; a minimum spanning arborescence then decodes taxonomic DAGs. Using only a concept vocabulary (no documents), PEEL recovers taxonomic DAGs with high structural fidelity on two OLLM benchmarks, transferring to a new domain with no target-edge supervision. The end-to-end OLLM pipeline, which jointly discovers concepts and induces structure from document corpora, is reported as a cross-task reference. Pre-trained retrieval embeddings encode a decodable granularity gradient; a minimal probe renders it as explicit parent-child (is-a) structure. 

---
# Enhancing Knowledge Entity Relation Extraction in Abstracts and Full-text Content of Academic Articles Using Large Language Models 

**Authors**: Yingyi Zhang, Chengzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2610.41552)  

**Abstract**: This study investigates the effectiveness of large pre-trained language models (LLMs) for knowledge entity relation extraction (KERE) from academic papers. Existing studies have examined individual types of LLM-based strategies and evaluated model performance using metrics such as precision and recall. To provide a broader evaluation, this study investigates multiple strategies, including closed-source LLM-based CoT prompting, LoRA-adapted open-source LLMs, LLM-enhanced small pre-trained language models (SLMs), and SLM-enhanced closed-source LLMs. In addition, this study introduces two strategies, namely SLM-enhanced open-source LLMs and closed-source LLM-enhanced open-source LLMs. These strategies are evaluated from multiple perspectives, including metrics, their ability to mitigate errors observed in SLMs, and potential directions for model optimization. This study manually constructs two annotated datasets in the NLP domain, covering Abstract and Full-Text datasets. Experimental results show that the LoRA-adapted T5 model achieves the highest macro F1 scores of 0.7408 and 0.7013 on the Abstract and Full-Text datasets, respectively. SLMs also benefit from LLM-generated entity descriptions under the LLM-enhanced SLM strategy. Error analysis further reveals that the effectiveness of LLM-based strategies in mitigating SLM errors varies across error types. Based on the identified limitations and error patterns of these strategies, several optimization directions are proposed. In addition, combining strategies with different error patterns further improves KERE performance. 

---
# Fidelity Synopses: Pre-Execution Plan Selection for Approximate Vector Search 

**Authors**: Adithya Ramanatha, Sanskar S. Singh, Maya Ramanath  

**Link**: [PDF](https://arxiv.org/pdf/2610.41407)  

**Abstract**: Vector search has become a database operator with many physical implementations: different index families, build-time and search-time parameters, and compressed or disk-resident variants. Each trades cost against fidelity, the fraction of a query's exact top-k nearest neighbors that a configuration returns. Deployments nevertheless choose one configuration offline and use it for every query, so they are sized for their hardest queries and overpay for the rest. On a 5.8M-item LAION benchmark with four embedding models, running each query on the cheapest configuration that suffices for it would roughly halve the cost in a typical setting. Exploiting this requires predicting each configuration's fidelity before running it, which is hard because fidelity is defined by the very exact search that approximate indexes avoid. Relational optimizers solve the analogous problem for cardinality with offline synopses; we propose one for vector search. A fidelity synopsis divides the embedding space into cells around a set of landmarks and records, for every cell and configuration, how often that configuration met a fidelity target on a sample of queries. At query time, one nearest-landmark lookup yields a calibrated estimate for every configuration, and an optimizer chooses configurations so that the workload misses the target no more often than its requirement allows. We test the two premises the synopsis relies on, that fidelity varies smoothly across the embedding space and that its loss is specific to each configuration. Held to the same requirement a static deployment is built to meet, synopsis-guided selection lowers cost in all 16 combinations of model and query direction, to a median of 0.72x the static deployment, with a batched lookup of 21-32 microseconds per query. The saving persists against the best static choice among the same indexes, and within a single index at no extra memory. 

---
# Hierarchical Dual-Graph Retrieval for Long-form Audio Reasoning 

**Authors**: Eunsang Lee, Taein Ju, Jiyoung Lee  

**Link**: [PDF](https://arxiv.org/pdf/2610.41390)  

**Abstract**: Large audio language models (LALMs) struggle with long-form audio reasoning, as processing entire recordings often limits to capture query-relevant content distributed across distant segments while incurring substantial computational cost. We introduce AudioHelix, a hierarchical DualGraph framework tailored for LALMs' retrieval-augmented generation (RAG) systems. Unlike existing text-only or non-graph RAG approaches, AudioHelix connects related textual and acoustic evidence across temporally distant segments for multi-hop reasoning. Extensive experiments on TraceAV-Bench, MMAU-Pro, and AudioMarathon demonstrate that AudioHelix consistently outperforms full-audio processing and existing RAG methods. The project site is this https URL. 

---
# WebOverlay: High-Throughput Vision-Only Distillation for Extreme Aspect-Ratio Web Overlay Detection 

**Authors**: Jiannan Huang, Jerry Ding, Li Yu, Zhenting Li  

**Link**: [PDF](https://arxiv.org/pdf/2610.41037)  

**Abstract**: Web overlays, such as cookie consent banners, newsletter prompts, and age gates, obstruct automated web crawlers and degrade downstream document representations. While proprietary multimodal Large Language Models (MLLMs) can accurately identify overlays, their latency and inference cost preclude web-scale ingestion. Conversely, lightweight vision models struggle with full-page renderings characterized by extreme vertical aspect ratios, where standard isotropic resizing collapses critical text features and causes false positives. In this work, we present WebOverlay, a high-throughput vision-only detector trained via distillation from a frontier vision-language teacher. We establish a dynamic tiling strategy that partitions extreme aspect-ratio web renders into aspect-preserving patches with spatial overlap, coupled with an aspect-ratio-aware boundary adjudication mechanism to eliminate boundary artifacts. To eliminate cross-modal latency overheads, our student model relies purely on visual cues, bypassing DOM tree parsing. WebOverlay significantly outperforms DOM-based heuristics and achieves competitive precision and recall against heavy proprietary MLLM baselines while delivering multiple orders of magnitude higher throughput and substantially reduced deployment costs. 

---
# FieldGuideRAG: Image-Grounded GraphRAG for Fine-Grained Recognition with Vision-Language Models 

**Authors**: Jiacheng Ding, Yanliang Qi, Cong Guo, Peihua Mai  

**Link**: [PDF](https://arxiv.org/pdf/2610.40971)  

**Abstract**: We present FieldGuideRAG, a training-free GraphRAG method in which a vision-language model, or VLM, recognizes fine-grained species by comparing the query with real photographs stored on the nodes of a knowledge graph. Multimodal GraphRAG systems describe each entity in text, and text conveys little of the visual detail that separates look-alike species. Our graph stores one photograph on every species node and links each species to its relatives, its geographic range and its ecological facts. A frozen VLM verifies the query against the photograph of each candidate node. FieldGuideRAG fuses the verification scores with the range edge and answers questions about the organism from the fact edges of the verified species. On CUB-200, a Wikidata graph and four iNaturalist domains, the node photograph improves verification over the species name on every graph, while text descriptions add little. Verification of the candidate pool exceeds a SigLIP retriever by up to 16 points, the range edge adds 7 to 19 points on iNaturalist, and answering from fact edges is correct on up to 89.5% of 28,820 ecological questions. The verification scores also detect queries whose species is missing from the graph. Graphs, prompts and code will be released. 

---
# Prevented is Not Protected: What Attack Success Rate Hides about RAG 

**Authors**: Chahana Dahal, Zuobin Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2610.40742)  

**Abstract**: Retrieval-Augmented Generation (RAG) is vulnerable to knowledge poisoning, where adversaries manipulate retrieved evidence to influence model outputs. Existing defenses are primarily evaluated using attack success rate (ASR), which obscures defense-induced failures and the mechanisms behind apparent attack prevention. To address this issue, this work introduces \textbf{Aligned RAG Evaluation}, a matched-query protocol that jointly examines clean and poisoned responses, with and without defense. By combining matched conditions analysis, LLM-judge endorsement, and retrieval dependence controls, the proposed framework distinguishes prevented from introduced failures, target mentions from LLM endorsement, and single-passage influence from evidence composition. Experiments on HotpotQA and Natural Questions reveal that existing defenses can introduce unexpected target-positive answers, although they can prevent some attacks, while evaluation criteria can also change defense rankings. Our findings expose critical blind spots in ASR-based evaluation and motivate a more comprehensive assessment of RAG poisoning defenses. The code and the evaluation artifact are available at this https URL. 

---
