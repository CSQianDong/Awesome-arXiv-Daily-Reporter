# Cost-Aware Agentic Feature Optimization System for Fleet-Scale Ranking Models 

**Authors**: Chang Yang, Yao Xuan, Wendan Yan, Nhat Le, Jiang Liu, Longhao Jin, Zhen Peggy Yao, Alireza Vahdatpour, Djordje Gligorijevic  

**Link**: [PDF](https://arxiv.org/pdf/2611.26531)  

**Abstract**: Large Web platforms operate large fleets of ranking models, each selecting features from a shared pool of tens of thousands. We present an end-to-end agentic system that optimizes features across the fleet under resource constraints shared across models and those set per model, such as feature pipeline cost and serving throughput (QPS), respectively. In this system, LLM agents propose hypotheses, and deterministic tools execute experiments and verify them. For each model, an LLM generates and shortlists cross features from a combinatorial candidate space, and all features are then ranked by several importance methods. Feature blending mixes these rankings into a single selection, using Pareto fronts across tasks for multi-task models. An LLM Researcher agent then tunes the feature selection in 10--25 asynchronous trials under resource caps, and a trimming step keeps the pipeline cost within budget. Across production models, the system improves normalized entropy by 0.31% on average within each model's budget, and the trimming step cuts pipeline cost by 2.3% with no regression beyond measurement noise. The optimized models were launched, and in online A/B tests they improved a primary online metric by 0.20%. 

---
# Democratizing Cross-Project Knowledge for Transforming Water Governance and Climate Resilience: an Agentic AI Ecosystem 

**Authors**: Oriol Alàs, Ian Palacín, Iván Cester, Edgar Rubión  

**Link**: [PDF](https://arxiv.org/pdf/2611.26398)  

**Abstract**: EU-funded projects produce extensive evidence in the form of deliverables, reports, and case studies. However, this knowledge is typically scattered across projects and document structures, making it difficult to locate relevant, contextual information when making decisions. This paper introduces an Agentic AI Ecosystem that consolidates heterogeneous project documentation into a domain-aware knowledge base and provides adaptive access through an agentic Retrieval-Augmented Generation pipeline. Documents from three EU-funded projects (IMPETUS, NATALIE, and InnWater) are reconstructed into hierarchical section trees and indexed with structured metadata. Retrieval is supported by two domain-specific embedding models fine-tuned on domain data (water and climate). On top of this knowledge base, a ReAct-style controller iteratively orchestrates retrieval actions by allocating retrieval budget across domain collections and accumulating evidence until a stopping criterion is met. In a retrieval benchmark over the corpus, the proposed approach improves Mean Reciprocal Rank by 26.7% (relative) compared to a single-domain embedding baseline, indicating more consistent retrieval of relevant evidence to support cross-project knowledge synthesis and climate-resilient decision-making. 

---
# From Sessions to Serving: Integrating LLM-Based Embeddings into Retrieval for Production Content Recommendation 

**Authors**: Zhixiong Hu, Meng Li, Maggie Zhang, Deborshi Goswami, David Bailey, James Bradley  

**Link**: [PDF](https://arxiv.org/pdf/2611.26096)  

**Abstract**: Personalized content recommendation on large marketplace websites must reconcile heterogeneous candidate content (e.g., customer stories, product pages, industry pages, and learning content) with sparse, fast-changing user behavior, placing increasing demands on the retrieval stage to represent user interest well, not just the downstream ranking stage. We present LSER (LLM Session Embedding Retrieval), a production content recommendation system for this http URL that derives session- and history-level embeddings from a pretrained large language model (LLM) and adds them as auxiliary user-side features in the retrieval-stage user tower alongside existing learned sequence encoders. To keep user-side embeddings fresh under real-time behavior, we build a nearline serving path that asynchronously refreshes sequential and LLM-derived user vectors for our main content recommendation server. We validate the incremental value of the LLM features through an ablation study across our full multi-content-type catalog. Our results indicate that integrating LLM-derived embeddings as additional retrieval signals, together with production-grade nearline serving, is technically feasible and can improve model performances. 

---
# A Task-Level Bias Taxonomy for Historical Image Retrieval 

**Authors**: Omkar Gavali, Ratan Sebastian, Christoph Rippe, Ralph Ewerth, Anett Hoppe  

**Link**: [PDF](https://arxiv.org/pdf/2611.26019)  

**Abstract**: In recent decades, galleries, libraries, archives, and museums (GLAM institutions) have digitised vast amounts of cultural heritage material, including historical imagery. A major challenge in making these collections publicly accessible is enabling effective search and discovery. Many historical images contain incomplete, sparse, or no descriptive metadata. Consequently, machine learning (ML) methods are increasingly being used to generate or enrich metadata to improve the searchability of such collections. However, artificial intelligence (AI)-based systems can reproduce or amplify existing biases, particularly when applied to historical imagery. Factors such as differences between the data used to train the models and the historical images on which they are applied, degraded image quality, monochrome photographs, and historically biased or incomplete descriptions can reduce model performance and introduce bias into search and retrieval results. In this paper, we examine different forms of bias in historical image collections and review current methods to mitigate them. We investigate how these biases arise and how they influence AI-supported search and retrieval systems. 

---
# LINEAGE: A Benchmark for Empirical Lineage Retrieval in AI Scientific Literature 

**Authors**: Pangjing Wu, Xu Yuan, Wenqi Fan, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.25845)  

**Abstract**: Current research agents are moving beyond academic search toward literature synthesis and experiment planning. These workflows increasingly rely on literature retrieval systems, which serve as standalone systems or as components of research agents. However, existing scientific literature retrieval benchmarks mainly evaluate these methods using topical relevance or citation links as supervision, which cannot capture which prior methods are used for empirical comparison. To address this gap, we introduce Lineage, a benchmark for empirical lineage retrieval in AI scientific literature. An empirical lineage relation links a prior paper to a source paper that explicitly uses its method as an empirical comparison baseline. These relations form lineage graphs whose multi-hop neighborhoods provide retrieval ground truth for structural evaluation. To construct these graphs at scale, we develop Lineage-Constructor, which converts heterogeneous paper records and full-paper empirical baseline evidence into calibrated lineage graphs. Built from 121,048 papers across 17 top AI venues from 2018 to 2025, Lineage contains 3,950 graph components, 67,502 paper nodes, 164,948 lineage edges, and 80,000 multi-facet retrieval queries. Experiments with 19 representative retrieval systems show that empirical lineage retrieval remains challenging. Performance drops sharply with lineage distance, while structural metrics reveal that isolated retrieval hits often fail to recover closely connected empirical comparison neighborhoods. 

---
# CACO: Coverage-Aware Coordinated Optimization for Continuous Generative Retrieval 

**Authors**: Pan Zhang, Nan Zhuang, Yongkang Zhang, Kun Xú, Zhongtao Jiang, Zhiwei Wang, Kun Xǔ, Youhuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.25738)  

**Abstract**: Continuous generative retrievers powered by flow matching and diffusion sample multiple queries in item embedding space for parallel approximate nearest-neighbor (ANN) search under fixed retrieval budgets. However, optimizing distribution diversity or target-level interest separation ignores set-level retrieval geometry: sampled queries collapse onto overlapping ANN neighborhoods, while discrete top-K lookup and single-target supervision block end-to-end coverage optimization. We introduce CACO (Coverage-Aware Coordinated Optimization), a lightweight self-supervised post-training framework that coordinates queries from frozen flow-matching and diffusion retrievers toward complementary item coverage with negligible additional inference latency. CACO uses empirical retrieval frequencies across sampled reference queries as a relevance proxy, formulating a label-free coverage objective that improves recall across all 30 backbone-dataset settings in offline greedy selection experiments. To avoid the inference overhead of generating a large query pool, CACO employs a coordinated query adjustment network that combines inter-query bidirectional attention with gated relative directions, finally updating queries by bounded tangent-space projection and spherical retraction. During training, CACO applies tangential antithetic perturbations and measures each query's marginal gain via counterfactual single-query replacement, yielding an unbiased gradient estimator of smoothed set coverage across non-differentiable ANN search. On eight public benchmarks and an in-house 24-partition streaming industrial dataset, CACO improves Recall@100 by up to 23.7% and Recall@400 by up to 23.9% over the strongest baseline, cuts inter-query overlap by 53.3%-64.9%, and adds only 0.5%-3.0% inference latency. 

---
# Can LLMs Resolve Entity Conflicts in Domain Specific NLU Pipelines? 

**Authors**: Akansha Bhardwaj, Mingbo Cui, Victor Guimaraes, Dominik Kohler, Diana Nicoleta Popa, Meghana M. Reddy, Paolo Rosso  

**Link**: [PDF](https://arxiv.org/pdf/2611.25733)  

**Abstract**: Resolving entity conflicts is a critical step in Pipeline-based NLU architectures, especially when multiple extractors provide overlapping or conflicting outputs. Traditional rule-based entity conflict resolution methods often lack flexibility and are unscalable. In this work, we aim to study if the broad knowledge of LLMs can be leveraged to resolve entity conflicts in domain specific NLU pipelines. We present an experimental comparison between a traditional rule-based conflict resolution method, and two LLM-based approaches: a multi-shot prompting method using the Llama3.1-8B-Instruct model and a LoRA fine-tuned Llama3.1-8B-Instruct model. The evaluation was conducted under two setups: (1) a broad assessment of LLM's capability to match existing NLU pipeline output, and (2) a more controlled scenario assessing the LLM accuracy on human-labeled data. Results show that while multi-shot prompting offers ease of adaptation and rapid behavioral changes, the fine-tuned model delivers superior performance on complex, ambiguous cases. Ultimately, the best approach appears to be a combination of both strategies, leveraging multi-shot prompting's flexibility alongside the broad accuracy gains of LoRA fine-tuning for a more comprehensive performance. 

---
# Exploring Semantic Table Search in Data Lakes 

**Authors**: Martin Pekár Christensen, Matteo Lissandrini, Katja Hose  

**Link**: [PDF](https://arxiv.org/pdf/2611.25691)  

**Abstract**: Discovery of relevant datasets within data lakes remains a key challenge. Union and join search are effective techniques for discovering data when augmenting an existing table. In exploratory search, instead, keyword search is commonly adopted to find a relatively small set of highly relevant tables matching some search terms. In this context, semantic search is a new technique that exploits the presence of named entities in exemplar tuples used as queries and in target tables. Hence, keyword search and entity search are able to find a diverse set of tables. In this work, we propose \texttt{Thetis++}, which combines keyword and semantic table search to discover relevant tables that combine the semantic signal with the presence of exact keyword matches. Incorporating multiple paradigms and sources of information allow us to combine the best of multiple worlds. Therefore, we study many alternative techniques for combining rankings and evaluate these for the table search task in data lakes. We explore both unsupervised and supervised techniques, including TOPSIS. Finally, alternative to Thetis++ combining keyword and semantic relevance signals, we study a variant of learning-to-rank for the table search task in data lakes. Our evaluation highlights both the limitations of many standard approaches along with some promising directions offered by learned approaches. Furthermore, while indexes for keyword search are quite efficient, here we provide insights into how to improve the scalability of prefiltering techniques for semantic search, comparing LSH indexes with Hierarchical Navigable Small Worlds. Our experiments show that HNSW drastically improves the runtime of semantic table search by 4.7X on average whilst retaining the high-quality tables. Together, our findings plot the line for effective and efficient data discovery techniques for semantic data lakes. 

---
# DNA: Unlocking Depth Scaling in CTR Models 

**Authors**: Caiyuan Li, Wenjie Wu, Yuheng Huang, Xiao Zhou, Silu Zhou, Ke Chen, Zinan Zhao, Xinrui Zeng, Hanmo Xu, Chaoqun Song, Hua Zong, Zongyuan Wu, Hongbin Zhou, Huimin Yi, Zhixuan Zhang, Dan Ou, Haihong Tang  

**Link**: [PDF](https://arxiv.org/pdf/2611.25659)  

**Abstract**: Long user behavior sequences are among the strongest signals for click-through rate (CTR) prediction. Cross-attention makes these sequences practical for industrial ranking by keeping compute linear in sequence length. Yet the gains from stacking cross-attention layers taper off: on Taobao search, growing a cross-attention stack from 16 to 28 layers adds only 0.02 percentage points (pt) of AUC. We argue that the key-value side is the bottleneck. Each layer projects the same behavior sequence into keys and values, so adding depth gives the model more reads of a fixed sequence. We propose DNA (Dual-stream Nexus Attention), which lets the sequence itself evolve with depth. At each layer, the request representation (the search query, user features and scene) first reads the sequence. Each behavior token then attends to the updated request representation and uses what it reads to update itself, so the next layer reads a sequence shaped by the current request. The update runs once per request, all candidates share the resulting sequence, and both the reads and the update keep linear complexity in sequence length. At comparable forward FLOPs, DNA improves AUC over cross-attention stacks by 0.08 to 0.26 pt and keeps improving in the depth range where those stacks level off. DNA-4 is deployed in Taobao search. In an online A/B test against the production model, it improves CTR by 0.8% and GMV by 0.5%. 

---
# Generative News Recommendation Reproduces Popularity: A Memorization-Based Analysis 

**Authors**: Shotaro Ishihara  

**Link**: [PDF](https://arxiv.org/pdf/2611.25606)  

**Abstract**: Text-based generative recommenders can go beyond ranking existing items by directly generating item titles or other textual representations. In the news domain, this raises a societal concern that models may repeatedly surface already popular articles and thereby narrow the diversity of information to which users are exposed. We investigate training-data memorization as one mechanism behind this concentration. Using approximately 20 million browsing sessions from a Japanese news service, we continually trained an instruction-tuned Llama 3 8B model to generate the next article title in a session. The generated-title distribution was more concentrated and less diverse than the distribution of titles observed in the browsing logs. Frequently repeated target titles exhibited greater verbatim and approximate memorization, and more strongly memorized titles were generated in response to a wider range of prompts. These findings support a pathway in which popularity in behavioral logs becomes repeated training text, is preferentially memorized, and is subsequently reproduced by a generative recommender. An exploratory target-level deduplication condition reduced both memorization and output concentration, while substantially reducing the training-set size and recommendation accuracy. Our results suggest that popularity bias in generative recommendation may arise not only from user preferences or ranking objectives, but also from the model's tendency to reproduce frequently encountered strings, with important implications for information diversity on news platforms. 

---
# Learning Generative Ranking Policies for Whole-Page Optimization 

**Authors**: Chen Qiu, Zhigang Hua, Qi Xu, Jizhe Zhang, Sha Hua, Ren Mao, Luke Simon  

**Link**: [PDF](https://arxiv.org/pdf/2611.25527)  

**Abstract**: A recommendation page is consumed as a whole: the items on it compete for attention and complement one another, so the best page is not simply the individually best items placed side by side. We study whole-page optimization, which selects and orders the items on a page jointly to maximize the score of the page. The number of possible pages grows combinatorially, so good pages must be generated rather than enumerated. Some generative re-ranking methods use the same page-level training signal for every choice in a sampled page. Others imitate target pages built before training, which remain largely fixed as the generator improves. We address both limitations with one local page-level objective, which repeatedly moves the current generator toward pages with higher score. This objective splits exactly into one training target per position, and each target compares candidates at the same partial page by the scores of the complete pages they lead to. Every position therefore learns to choose the item that is better for the whole page, rather than the item that looks best on its own. We call this procedure whole-page policy improvement. On two public datasets, KuaiRand-1K and RL4RS-SeqSlate, which cover multi-objective item selection and constrained ordering, and on an industrial dataset of logged pages, our method outperforms the evaluated generative re-ranking baselines. On a synthetic benchmark with a known optimal page, it also comes closest to the optimum, showing that its improvements come from better whole-page decisions. 

---
# LQ-Rec: Late Quantization for Sequential Recommendation 

**Authors**: Satoshi Koide, Yueqi Wang, Julian McAuley  

**Link**: [PDF](https://arxiv.org/pdf/2611.25256)  

**Abstract**: Generative retrieval (GR) for sequential recommendation compresses each item into a short semantic ID and trains an autoregressive decoder to generate next-item codes from a user history. While appealing for cold-start handling and storage, au- toregressive GR suffers a structural scaling limit: typically, longer semantic IDs reduce quantization error in principle but degrade ranking quality and quadratic memory usage against semantic ID length. We propose Late Quantization (LQ), a non-autoregressive paradigm that confines discretization to the output (item) side of the pipeline. The encoder and scoring rule operate in continuous space, while items are stored and matched as semantic IDs produced by an end-to-end-trained residual quantizer. This architectural shift permits an additive error decomposition and a pairwise ordering theorem that jointly bound the quantization- induced ranking drop, a structural guarantee absent in prior semantic-ID recommenders. Across four public benchmarks, the resulting recommender, LQ-Rec, delivers three capabilities simultaneously: ranking quality improves monotonically with the semantic-ID length and outperforms existing baselines; a single trained model supports anytime compression, tuning ranking quality against inference cost without retraining; and LQ-Rec shows superior generalization on a memorization/generalization partition. 

---
# Retrieving Patristic References and Modelling Topics in Bullinger's Reformation Correspondence 

**Authors**: Lena Pickartz, Philipp Schaer, Phillip B. Ströbel  

**Link**: [PDF](https://arxiv.org/pdf/2611.25247)  

**Abstract**: The correspondence of Heinrich Bullinger (1504-1575), Swiss Reformer and Antistes of Zurich, documents the political and religious debates of the Reformation, including extensive appeals to the Church Fathers as authoritative sources. Detecting such references computationally remains largely unexplored. We investigate dense and sparse retrieval for patristic reference detection, and metadata-informed topic modelling as a proxy for the topics and ideas discussed. Prior work on Latin intertextuality has fine-tuned dense retrievers for their respective target domain, and we test how well these models transfer to Bullinger's correspondence. We compare lexical (BM25), dense (bi-encoders), and hybrid methods on explicit and implicit references. A Latin encoder fine-tuned on biblical reuse performs best among the bi-encoders (82%, Recall@20 on explicit and 16% on implicit references), while BM25 outperforms dense approaches on explicit citations (94%, R@20). Because BM25 and dense retrieval recover disjoint sets of implicit references, fusing their rankings raises implicit R@20 to 22%, close to cross-encoder reranking of the same candidates (26%) but without its inference cost. For topic modelling, we extend FASTopic with a seed-guided loss function and editorial abstract embeddings. Measurable coherence gains over the baseline remain minimal, but qualitative and temporal analysis shows that some topics track documented historical events. However, dominant topic weights barely exceed a uniform distribution, making topics unusable to inform retrieval. A pilot application of LLM-based concept induction (LLooM) further points towards a more human-interpretable alternative. Together, these results present a first approach to retrieving patristic references and modelling fine-grained topics in Bullinger's correspondence. 

---
# LSP-RAG: Latent Scoring Perspectives for Knowledge Graph Retrieval-Augmented Generation 

**Authors**: Haochen Liu, Guanghui Min, Song Wang, Yada Zhu, Chen Chen, Jundong Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.25229)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong capabilities in open-domain question answering, but often struggle with factual accuracy and multi-hop reasoning due to the incompleteness of the training corpora. A promising solution is Knowledge Graph Retrieval-Augmented Generation (KG-RAG), which supplements LLMs with structured knowledge retrieved from external knowledge graphs (KGs). However, existing KG-RAG methods either rely on large-scale language models (e.g., ChatGPT) to guide the retrieval process, which leads to high computational costs, or suffer from limited retrieval quality when using lightweight language models, particularly under multi-hop scenarios. We propose LSP-RAG (Latent Scoring Perspectives for Retrieval-Augmented Generation over Knowledge Graphs), a novel KG-RAG framework that enhances hop-wise KG knowledge retrieval through multi-perspective prompted scoring. Each latent scoring perspective is guided by a combination of two types of soft prompts: a perspective-specific soft prompt induces a distinct prompt-conditioned scoring function over candidate triplets, while a contextual soft prompt evolves with each reasoning hop by encoding the query and previously explored KG triplets, enabling the model to preserve consistency and relevance across multi-hop retrieval. Rather than assuming manually defined semantic roles, these latent perspectives are adaptively aggregated to capture heterogeneous relevance signals during retrieval. This design allows LSP-RAG to perform accurate and robust retrieval using lightweight language models. LSP-RAG achieves superior performance across multiple KG-based Question Answering benchmarks compared to existing retrieval systems, including those that rely on much larger language models, demonstrating its effectiveness under limited computational budgets. 

---
# Variants of Beam Search for DiskANN 

**Authors**: Viraj Agrawal, Kirankumar Shiragur  

**Link**: [PDF](https://arxiv.org/pdf/2611.26609)  

**Abstract**: Graph-based data structures such as DiskANN, HNSW, and NSG constitute the state of the art for large-scale approximate nearest-neighbor (ANN) search. Indyk and Xu introduced the notion of $\alpha$-reachable graphs, which include graphs built by DiskANN's slow-preprocessing construction, and proved that greedy search achieves an approximation factor of $\tfrac{\alpha+1}{\alpha-1}$. Independently, Al-Jazzazi et al. proposed adaptive beam search (ABS) with a relaxation parameter $\gamma > 0$, showing a $\tfrac{2}{\gamma}$ approximation on navigable graphs ($\alpha \approx 1$), where greedy search admits no guarantees. In this work, we unify and generalize these results by proving that ABS with parameter $\gamma$ on $\alpha$-reachable graphs achieves an approximation of $\tfrac{\alpha+1}{\alpha(\gamma+1)-1}$ for any metric. Moreover, on sorted $\alpha$-reachable graphs (introduced by Gollapudi et al.) under the $\ell_2$ metric, we obtain the tighter bound $\max\bigl(\tfrac{\alpha}{\alpha(\gamma+1)-1}, \tfrac{\alpha^{2}+1}{\alpha^{2}(\gamma+1)-1}\bigr)$. Beyond these theoretical results, we design a beam predictor using features inspired by fine-grained analysis of ABS, trained via a constrained objective that minimizes mean distance computations subject to a target recall. On three 100-million-point benchmarks, using graphs built by DiskANN's practical construction, beam predictor lowers mean query latency on SSD-resident indices at matched recall by up to $29.6\%$ relative to classical beam search (CBS). Relative to ABS, it is faster in $11$ of $15$ settings, by up to $6.5\%$, and at most $1.4\%$ slower in the other $4$; at $95\%$ recall, its advantage over ABS grows from $0.8$--$4.2\%$ for $100$ neighbors to $4.6$--$5.6\%$ for $500$ neighbors. In memory, it needs $1.3$--$23.7\%$ fewer distance computations than CBS and $0.0$--$3.9\%$ fewer than ABS. 

---
# The Window Lever: Longer Observation Windows Outweigh the Observed Effects of Data, Capacity, and Architecture in Customer Behavior Forecasting 

**Authors**: Tara Sowrirajan, Yanqiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.26521)  

**Abstract**: In our experiments on long-horizon customer-behavior forecasting, a longer observation window outweighed the observed effects of more data, more capacity and newer architectures. The window is the weeks of history a model sees before the week it forecasts from. Going from an 8- to a 16-week window adds a raw +0.070 to 0.107 log-space $R^2$ in every family at 1M. Matched test cohorts, which score the same customer-weeks at every window, show this raw comparison understates the gain in every family they cover. The direction holds on five public datasets, four spanning over a year (ridge and our transformer on all five). On our internal data, longer windows admit up to 10% more and colder training customers, which matching does not fix. Our design does not separate window length from calendar coverage, because every test anchor (the week a forecast is made from) falls in one eight-week band. On training-period anchors, the weekly ridge probe's gain persists outside the holiday season, keeping about 72% of it. We therefore report a robust positive association with longer windows and do not identify the magnitude of a calendar-independent history effect. Of four internally registered predictions, the only fully blind one failed. An activity-mixture model predicted +0.014 from 24 to 32 weeks; the weekly-aggregate probe returned -0.0004. For the sequence families, the other three levers are about an order of magnitude smaller, bounded and unresolved. Their observed effects are at most 0.006 between the nominal 1M and 10M tiers and 0.0052 among five sequence families at 16 weeks with clocks as built, both single-seed on the full test split, and 0.006 for capacity on three-seed means. These are observed intervals, not equivalence bounds. Our claim is the order-of-magnitude contrast between the window and those bounds. A team building on this class of data should start by lengthening the window. 

---
# STARK: Moving towards Personalized Agents That Truly Learn and Remember 

**Authors**: Aryan Jain, Abhishek Jha, Anish Nediyanchath, Promod Yenigalla  

**Link**: [PDF](https://arxiv.org/pdf/2611.25705)  

**Abstract**: Long-term memory is essential for personalized agentic systems that interact with users over long time periods, yet current architectures only capture what happened while leaving the reasoning of why it happened to be derived later by underlying LLMs.%, and silo knowledge within individual users. These events can, however, convey not only what a user experienced but also how that experience shaped their behavioral patterns. We introduce STARK, a dual-graph memory architecture which treats these events not just as nodes but also as edges between behavioral insights. STARK maintains an Episodic Graph which records interactions as temporally, causally, and semantically linked episodes; and an Insight Graph which represents LLM-extracted behavioral signals scored by how many events support them, connected by the very events that led to them. At inference, behavioral knowledge is retrieved directly from insight nodes grounded by event-as-edges, eliminating repeated episodic traversal. The event-as-edge structure also naturally enables projecting the knowledge graph into a relational embedding space where cross-user transfer becomes possible under shared contextual circumstances, allowing the system to infer missing insights for cold-start users. Evaluations show that STARK improves by 1.48 judge score on LongMemEval information extraction and 0.93 on LoCoMo open domain questions over state of the art. On the public LongLaMP abstract-generation benchmark, cross-user transfer improves open-ended judge scores by 0.7 points and ROUGE-L by 3 at 5% graph sparsity. 

---
# Uncontrolled source titles: When Nature, Science, and PNAS meet Scopus 

**Authors**: Weishu Liu, Ruifeng Zhang, Haifeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.25627)  

**Abstract**: Bibliographic databases such as Scopus and Web of Science play a critical role in literature discovery, systematic reviews, bibliometric analyses, and research evaluation, yet their reliability depends on effective control of the quality of the bibliographic data. This study examines the problem of uncontrolled source titles in Scopus by comparing five possibly adopted journal-level retrieval strategies across three flagship journals-Nature, Science, and PNAS. Substantial discrepancies across retrieval results indicate persistent fragmentation in the source title field. A comparative analysis using the Web of Science Core Collection shows that no comparable degree of uncontrolled source-title variation is observed for the same journals. The study further explores possible mechanisms, including inheritance of different title forms from (different) upstream providers, automated parsing and OCR-related artifacts, and insufficient cross-validation. Based on these findings, the study proposes recommendations at both the user and database-provider levels to improve source-title normalization and strengthen source-level metadata governance, with implications for the accuracy and reproducibility of journal-level retrieval and evaluative uses. 

---
# AutoRAGBench: A Benchmark for LLM Agents Automating RAG System Development 

**Authors**: Hang Zhang, Yuzhi Pan, Ziyao Zhong, Chaokun Wang, Shuo Cao, Zeyu Huang, Yue Xue, Fang Niu, Bofan Xie, Guanchen Ge, Zecheng Li, Leqi Zheng, Ziyang Liu, Wei Wang, Miao Gu  

**Link**: [PDF](https://arxiv.org/pdf/2611.25582)  

**Abstract**: Retrieval-augmented generation (RAG) is a well-established paradigm for augmenting large language models (LLMs) with information retrieved from external corpora. However, developing effective RAG systems requires substantial engineering effort and iterative experimentation, as effective design choices vary across tasks and domains. With rapid advances in LLM agent capabilities, we ask how well these agents can autonomously build RAG systems and how effectively they can improve them through iteration. In this paper, we propose AutoRAGBench, the first benchmark to evaluate how well LLM agents develop RAG systems from scratch in an open code space. It comprises nine RAG system development tasks. For each task, agents build RAG systems from scratch without starter code or prescribed strategies, under a six-hour budget on a single H800 GPU. They analyze task data, implement RAG pipelines, and iteratively refine their designs based on experimental feedback. We evaluate nine agent configurations and analyze both the effectiveness of the resulting RAG systems and their development trajectories. GPT-6.1 Sol with Codex CLI achieves the highest average score of 56.1. The composite Human-SOTA baseline averages 44.4. These results show that LLM agents can automate effective RAG system development. When starting from a human-designed system on each of seven tasks, GPT-6.1 Sol with Codex CLI improves the score by 16.6 points on average. 

---
# VideoTrail: Interactive Video Foraging Through Hierarchical Visualization 

**Authors**: Catherine Yeh, Bryan Wang, Fernanda Viégas, Martin Wattenberg  

**Link**: [PDF](https://arxiv.org/pdf/2611.25377)  

**Abstract**: People often watch videos to find information, but locating relevant moments can be difficult and time-consuming. Existing strategies like scrubbing rely primarily on visual scanning and linear search, providing limited support for goal-directed navigation. We present VideoTrail, an AI-powered browser extension that supports information seeking in videos through a hierarchical visualization. Inspired by information foraging theory, VideoTrail surfaces semantic and structural cues as information scent, creating an explorable roadmap of segments. The system is multimodal and adaptive, prioritizing visual or textual scent based on the video's content. In a user study (N=16), VideoTrail helped participants efficiently locate key moments and enabled new navigation strategies compared to traditional timelines, highlighting how exposing an adaptive video structure can enhance goal-directed viewing. 

---
# FinVeriQA: Financial Research Agents for Question Answering: Routing, Retrieval, MCP Tool Use, and Provenance-Aware Evidence Verification 

**Authors**: Meenakshi Rajpurohit, Anna Phan, Ravikumar Komandur Narayanan, Gunanidhi Ramakrishnan, Nathan Howland  

**Link**: [PDF](https://arxiv.org/pdf/2611.25071)  

**Abstract**: Large language models (LLMs) write fluent financial analyses but miscompute ratios, recall stale prices, and route questions to the wrong tool with high confidence. We ask which component should do each job in a financial question-answering assistant, and study it with FinVeriQA, an open-source multi-agent research prototype over SEC filings and live market data: a compact typed-decision model (Laya) routes each question and screens out advice requests, Python computes every number, retrieval and tools fetch evidence, Mistral-7B writes a cited answer, a code validator checks that cited IDs exist and that numbers match computed notes, and a Model Context Protocol (MCP) server exposes the pipeline. In live experiments, Laya routed better than Mistral-7B (0.85 vs. 0.52; overall 0.86 vs. 0.68, McNemar p = 0.0003); a 12B Mistral decided no better, and DeepSeek-R1-8B decided better (0.79 vs. 0.68, p = 0.0192) but about 9.9 times more slowly. Mistral-7B computed 29% of 100 financial calculations correctly against 100% for Python, a live quote cut price error from 210% to 0.00%, and hybrid retrieval reached 76% Hit@5 on 50 filing questions. Yet in a live case study, both agent answers passed validation while describing AMD sample data under NVIDIA's and Alphabet's names. A controlled upper-bound study on the FinRank benchmark explains why: when one figure in the shared evidence is corrupted, self-consistency stays confident (agreement 0.55 to 0.59 while accuracy falls from 0.50 to 0.00), and a shared-evidence check (the kind our validator performs) catches 35% of the errors, whereas a check against the original passage catches 77% ($p = 5.96 \times 10^{-8}$); for DeepSeek-R1-8B, 68% vs. 90%. Verification must be independent of the evidence the writer saw. 

---
