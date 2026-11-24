# CO-GRAFT: Attributing Reinforcement Learning Gains in Search Agents via Role-Level Policy Grafting 

**Authors**: Ramraj Chandradevan, Sayontan Ghosh, Vinoth Selvendran  

**Link**: [PDF](https://arxiv.org/pdf/2611.31185)  

**Abstract**: Reinforcement learning (RL) improves large language model search agents by up to 32 exact-match (EM) points over their pre-RL initializations, yet overall accuracy gains do not identify which underlying functional capabilities improve or interact during training. We introduce CO-GRAFT, a reward-agnostic interventional attribution framework in which an RL-trained agent and its pre-RL initialization jointly generate a single trajectory under closed-loop execution with live retrieval. Each policy controls a mutually exclusive subset of four functional roles: intermediate reasoning, search-or-answer decision, search query generation, and final answer extraction. Evaluating all 2^4=16 role assignments without modifying weights or retraining defines a cooperative game whose Shapley values decompose the total RL gain across roles, while M"obius interaction terms distinguish standalone role improvements from multi-role coupling. Applying CO-GRAFT to Search-R1 agents across Qwen2.5 and Llama-3.2 models (spanning two sizes, base and instruction-tuned initializations, GRPO and PPO optimizers, and NQ and HotpotQA datasets), we show that query generation never exceeds 15% of the total gain; instead, gains are governed by pre-RL answer validity, search retention, model scale, and fixed task-level search budgets, shifting credit from reasoning toward answer extraction in larger or well-formatted answer response checkpoints while smaller base models rely on reasoning--decision coupling. 

---
# From Head to Tail: LLM-Augmented Sequential Recommendation and Trade-off Optimization 

**Authors**: Adeep Hande, Shaghayegh Agah, Neeraj Sharma, Karina Halevy, Sardar Hamidian  

**Link**: [PDF](https://arxiv.org/pdf/2611.31052)  

**Abstract**: Recommender systems exhibit popularity bias, often under-serving long-tail and cold-start items with sparse interaction data. We propose a group-specific data augmentation approach that leverages LLMs to generate synthetic interactions, formulated as a constrained optimization problem, maximizing group-specific NDCG@20 and coverage while bounding overall NDCG@20 degradation within an acceptable threshold. Items are partitioned by interaction frequency to find the optimal augmentation strategy for each group. We further study combining the augmentation across groups simultaneously and find that while individual optimal strategies navigate the multi-group augmentation design, they are not simply additive. We further ablate on the LLM-based augmentation itself, comparing LLM scales, augmentation configuration under token constraints, finding that augmentation quality scales with contextual richness, albeit at a proportionally higher cost. Experiments with SASRec demonstrate the effectiveness of the proposed approach, achieving over 100% and 60% improvement in group-specific NDCG@20 for extreme and regular long-tail items, respectively, on our industrial dataset, and 40%, 25% for MovieLens-32M, while increasing their coverage without significant overall degradation. 

---
# Ask Again: Answer-List Instability in Generative Engines 

**Authors**: Björn Busch-Geertsema, David Antony Selby, Sebastian Vollmer  

**Link**: [PDF](https://arxiv.org/pdf/2611.30819)  

**Abstract**: Visibility in generative search engines is measured by which entities appear in responses, and in what order. Ask an engine the same question twice and the answer list can change: a different ordering, or different items altogether. Across 60,000 responses, we find that answers to the same prompt name a different first item 32% of the time, and even two answers with the same items disagree on which comes first. The first position is a measurement target of its own. A "clear leader" emerges after seven or eight runs, which are the budgets recommended for brand and source monitoring, but a "close leader" only about half the time. After 25 runs, a close leader is identified only 60% of the time. Variability between engines is larger than within the same engine. For open-ended prompts, the gap is more than double that for determinate ones. Search queries and cited sites observed leave most of the disagreement unexplained. Visibility measurement should report, for each engine, which items appear and how often each comes first, each with its own quantification of uncertainty. 

---
# SuSi-RAG: A Hybrid Graph RAG for Equipment Maintenance 

**Authors**: Hansi Zhang+, Wilma Johanna Schmidt+, Wanlin Yang, Xiaozhi Shen, Adrian Paschke, Evgeny Kharlamov  

**Link**: [PDF](https://arxiv.org/pdf/2611.30588)  

**Abstract**: In the manufacturing domain, equipment maintenance tasks depend on expert knowledge. Their documentation is often stored in heterogeneous formats and sources. The enhancement of LLMs with Retrieval-Augmented Generation (RAG) is promising for bridging the scattered knowledge, but traditional vector-based RAG shows weaknesses with two characteristics which are common in real maintenance queries: complex causal chains and cross-domain synthesis. To address this, we propose SuSi-RAG, a hybrid Graph RAG solution, that matches a user query and each maintenance record using subgraph similarity. This work shows an in-use system: SuSi-RAG has run on an active Bosch production line for more than eight months, handling 10,664 question-answer interactions and demonstrates upper bound answer accuracy of 96.25% (confirmed error rate 3.75%). In a proof-of-concept, SuSi-RAG improves top-1 retrieval accuracy from 61.5% to 96.2% and expert-rated answer quality from 44.9% to 84.1% against a vector-based baseline. We present an integrated operational feedback loop to update SuSi-RAG. 

---
# Can Search Agents Correct Their Own Errors? Evaluating Retrieval, Evidence Delivery, and Answer Completion 

**Authors**: Venkata Pagadala, K. Rajasekhara Rao, Pratyay Ganguly  

**Link**: [PDF](https://arxiv.org/pdf/2611.30243)  

**Abstract**: A search agent can leave an earlier error unresolved by missing the right source, overlooking a relevant passage, or failing to revise and deliver the answer after receiving the evidence. We evaluate these possibilities on BrowseComp-Plus by replaying each model's own wrong closed-book answer in conversation and intervening on what evidence reaches the model. In a supplied-evidence study spanning four open-weight checkpoints, 709 questions and 26,665 test conditions, a confident assistant prior reduced Nemotron's paired accuracy by 16.1 percentage points despite answer-bearing passages in context. When the user insisted on the prior, accuracy fell by 32.9 points for Nemotron and 12.6 for gpt-oss. Among gpt-oss responses that lost correctness in that condition, abstention was more common than repeating the prior. In separate search interventions with Qwen3.5 and Nemotron, placing a gold page first often left the decisive source unopened. Delivering a decisive passage in context improved accuracy over neutral delivery by 51.0 percentage points for Qwen3.5 and 22.4 for Nemotron. Even with the passage in context, Nemotron exhausted its search budget and returned no answer in 19 of 51 episodes. The tests distinguish failures of retrieval, passage delivery, correction and answer completion, giving system builders a way to test which changes resolve a recurring error. 

---
# Beyond Knowledge Transfer: How User Overlap Shapes Diversity in Cross-Domain Recommendation 

**Authors**: Yuhan Zhao, Weixin Chen, Shu Chen, Li Chen  

**Link**: [PDF](https://arxiv.org/pdf/2611.30156)  

**Abstract**: A common paradigm in cross-domain recommendation (CDR) transfers knowledge through users shared by source and target domains. Although this overlap bridge is effective in improving accuracy, its effect on diversity remains unclear. We identify two bottlenecks: the bridge can be too narrow when overlapping users cover limited target-domain tastes, and too loud when shared signals dominate the sparse personal evidence of non-overlapping users. To address both, we propose DivCDR, a model-agnostic Push--Pull framework. Pull constructs confidence-gated virtual positives for overlapping users and optimizes their exposure-entropy gain, broadening the preferences carried by the bridge. Push uses overlap-density-guided negative sampling to move non-overlapping users away from overlap-dense regions. Experiments on three datasets and five representative CDR backbones show that DivCDR improves recommendation diversity while preserving or improving accuracy in most settings. 

---
# Structure-Exploiting Hierarchical Retrieval for Long Narrative Documents: A Precision-Oriented Alternative to Learned Abstraction Trees 

**Authors**: Rohit Singh, Nathaniel Weems  

**Link**: [PDF](https://arxiv.org/pdf/2611.29966)  

**Abstract**: Retrieval-augmented generation over book-length documents must locate a small, coherent set of passages inside tens of thousands of tokens. Flat dense retrieval ignores document structure; RAPTOR instead builds a hierarchy by recursively clustering and summarizing with a language model. We study a complementary design that exploits the native structure of a document: a two-level retriever that (i) selects chapters with a hybrid BM25+dense match over chapter recaps, (ii) runs a chapter-filtered dense search, and (iii) expands top passages into contiguous, section-aware spans under a token budget, plus content-hash-gated delta ingestion to keep the index cheap to update. We evaluate on five public-domain books (113 chapter-localization queries) and on QuALITY and NovelQA (200K+ token novels). In-house, we find a seed-stable precision/ranking tradeoff: with section expansion off, hierarchical retrieval returns a higher fraction of chunks from the correct chapter (+3.3pp, Wilcoxon p=0.014); with expansion on, ranking improves (MRR, p=0.040). Accuracy differences are not significant. On QuALITY, hierarchical retrieval beats flat top-k (81.0% vs 70.4%, p=0.0019) and matches the full article (p=0.74). On NovelQA, hierarchical and flat retrieval are statistically tied (55.7% vs 56.8%, p=0.24); hierarchy helps aggregation questions while flat wins on localized facts. Against a faithful RAPTOR baseline we find a complementary tradeoff, not superiority. Delta ingestion reuses 94 to 99.7% of embeddings on incremental edits. We make no claim of semantic chapter-boundary inference and scope claims to English narrative prose. 

---
# Weight Coding: Evolving Web-Scale Recommenders through Direct Weight Edits with LLM Agents 

**Authors**: Junpeng Hou, Jiaxing Qu, Derek Cheng, Jaewon Yang, Abhinav Naikawadi, Karthik Subbian  

**Link**: [PDF](https://arxiv.org/pdf/2611.29613)  

**Abstract**: Web-scale recommender models evolve primarily through retraining or fine-tuning. We propose \emph{weight coding}, a methodology for model evolution through direct, executable weight edits. A large language model (LLM) agent inspects model structure and parameters, formulates hypotheses, and implements edits using evaluation feedback. The loop progresses from localized changes to targeted searches and composition, allowing edited components to adapt to one another. A cached engine enables rapid forward evaluation without backpropagation. On Pinterest's 147M-parameter Image Search two-tower retriever, weight coding improves held-out recall@10 of production model by up to 15.0 percentage points (+24\% relative) with few edits in about 1/4 GPU hours of a full post-training finetune, while a common launch only improves around 1\%. After reinitializing the entire 13.6M-parameter interaction module randomly while retaining the surrounding trained model, searching 2,448 mixing and normalization parameters yields 9.8 points above production. Offline gains span the full funnel including dense retrieval, generative retrieval and ranking, with up to 1.37B learnable parameters and different edits for each architecture. Steering edits provide explicit control of model performance over segments like freshness and product. Weight coding requires no serving change and online A/B tests increase saves per user from the edited retrieval source by 8.4\% to 10.7\%, aligned with the target business objectives. Our results establish weight coding as a practical and useful tool for model development and and behavioral control in a real web-scale production system, with potential to extend to both pre-training and post-training stages across the funnel. 

---
# MoR4Rec: Mixture-of-Recursion for Sequential Recommendation 

**Authors**: Tiwari Ankit Satyaprakash, Adamya Shyam, Vikas Kumar, Bharti Rana, Venkateswara Rao Kagita  

**Link**: [PDF](https://arxiv.org/pdf/2611.29462)  

**Abstract**: Sequential recommender systems aim to predict a user's next interaction by modeling temporal dependencies in historical behavior sequences. To capture such dependencies, recent approaches increasingly rely on attention-based architectures, particularly transformers. While effective at modeling long-range sequential patterns, these models apply a fixed computation depth to all interaction items, treating them as equally complex. However, user behavior is inherently heterogeneous, with many interactions following simple repetitive patterns, whereas others require modeling higher-order dependencies. This mismatch leads to inefficient utilization of model capacity and unnecessary computational overhead, particularly in deep transformer-based architectures. To address this issue, we propose MoR4Rec, a sequential recommendation framework based on a Mixture-of-Recursion paradigm, which enables item-wise adaptive computation through a shared transformer backbone. A learned routing mechanism dynamically assigns each item to an appropriate recursion depth, allocating deeper computation to complex interactions while processing simpler patterns more efficiently. To improve computational efficiency, we develop a cascade-based strategy with key-value sharing across recursion steps. We further introduce a gated attention mechanism that adaptively regulates contextual information during recursive sequence modeling. Extensive experiments on three benchmark datasets show that MoR4Rec consistently outperforms strong transformer-based baselines, achieving average gains of 2.28% in Recall@5, 2.50% in NDCG@5, and 2.0% in MRR@5 over the strongest baseline. The improvements remain consistent across top-10 and top-20 recommendation cutoffs, while comparisons with vanilla transformers demonstrate the computational efficiency of adaptive recursive processing. 

---
# Who Gets Re-ranked When a Clinical Model Gains New Data? 

**Authors**: Priyanka Siddappa, Muntaqim Ahmed Raju, Shifat Haider Al Amin, Ruizhe Ma  

**Link**: [PDF](https://arxiv.org/pdf/2611.28968)  

**Abstract**: Hospitals often update a deployed risk model by adding a data source, such as laboratory results, that only some patients have. Work on safe model updates assumes fixed inputs and so ignores patients who lack the new data. The update has no new measurements for them, yet their rank can change, and a sick patient not yet tested can drop off an alert list. We split rank-based compatibility by data availability and pair it with error correction and alert-list effects. We compared each update with ordinary retraining on the old inputs, predicting ICU admission or death and hospital admission, on 78,475 emergency department visits from MC-MED and, in a pre-registered replication, 100,000 stays from MIMIC-IV-ED at a second hospital. Radiograph report text gave no detectable gain and almost no extra re-ranking. Laboratory results improved prediction but, at both hospitals and with linear or tree models, pushed patients without labs down alert lists; linear updates also reshuffled them more than retraining did, without ranking them better. A top-20% admission list caught 4.2% of admitted patients without labs after the update, against 18.8% after retraining (second hospital: 4.9% against 22.6%), while catching more patients overall. This held for list sizes from 5% to 50% and in every patient group we checked, despite better measured calibration. Separate alert thresholds for patients with and without labs restored most or all of their catch rate at both hospitals at almost no cost. Teams should check these patients before deploying such an update. 

---
# ChatGPT Can't Sell (And That's Probably a Good Thing): Buyer-Aligned AI Advice, the Sales-Representative Gap, and Evidence from Commercial Conversations 

**Authors**: Benjamin Tannenbaum  

**Link**: [PDF](https://arxiv.org/pdf/2611.28946)  

**Abstract**: AI assistants and merchant sales agents have different objectives: assistants should help buyers compare options and preserve the choice not to purchase, while sales agents also guide qualified buyers toward an authorized next step. We study this sales-representative gap with three evidence streams. In 12,178 historical English GPT-family conversations screened for commercial or transactional intent, a 279-request sample contained recommendation language in 63 first replies (22.6%) and no narrow transactional action wording. In a 56-conversation explicit-buying sample, 24 replies (42.9%) contained recommendation markers, 32 (57.1%) qualification phrases, and none transaction-specific phrases. These lexical measures do not estimate successful selling. A 2026 brand-monitoring cohort of 1,050 GPT-5.4--Gemini answer pairs over 383 purchase-related prompts showed transaction-specific phrases in 0.0% of recommendation-worded prompts, 0.21% of price/comparison prompts, and 9.52% of explicitly transaction-worded prompts. The prompts are selected, not organic shopping sessions. In a prospective pilot across five matched hypothetical product scenarios, sales instructions produced more next-step language than general-assistant instructions (5/5 vs. 2/5), but no increase in clean product choice (3/5 in both); unsupported offers and fit errors also occurred. This pilot cannot establish a population effect. A separate company-affiliated randomized field study reports higher engaged-visitor conversion for Javo, a commercially trained sales agent, than for a frontier-model agent in shared tooling, but does not identify the mechanism. We argue that apparent reluctance to sell can be appropriate for buyer-aligned assistants. Merchant agents should optimize qualified decision progress through catalog grounding, truthful authorization, and safeguards against unsuitable sales. 

---
# PROSPER: LLMs as Sparse Retrievers for First-Stage Product Search 

**Authors**: Hongru Song, Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Sen Li, Wenjun Peng, Fuyu Lv, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2611.28645)  

**Abstract**: Product search is a crucial component of large-scale e-commerce platforms, where first-stage retrieval should achieve high recall while ensuring efficient online deployment. Sparse retrieval remains attractive for this setting due to its interpretability and deployability, yet it suffers from severe vocabulary mismatch challenges, especially for short queries and product titles. Large language models (LLMs) offer strong semantic capabilities, but directly applying them to sparse retrieval in product search introduces new challenges: (i) semantic expansion may distort critical literal signals such as brand names and model identifiers; and (ii) learning meaningful sparse representations is difficult in the high-dimensional vocabulary space induced by this http URL address these challenges, we propose PROSPER, a framework that uses LLMs as sparse retrievers for product search while explicitly preserving lexical fidelity. PROSPER combines (i) a literal residual network that alleviates lexical expansion hallucination by reinforcing underweighted literal terms through a residual compensation mechanism; and (ii) a lexical focusing window that uses top-k selection for rapid early sparsification and effective training initialization. Extensive offline evaluations and large-scale online A/B tests demonstrate that PROSPER consistently improves over existing sparse retrievers, and provides revenue gains for industrial hybrid retrieval systems. 

---
# MADRec: Reliable Multi-Agent Reasoning with Evidence-Grounded Debate for Recommendation 

**Authors**: Zunlong Liu, Weijie Feng, Jingzhi Li, Lei Zhu, Zhiyong Cheng, Meng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.28604)  

**Abstract**: Large Language Models (LLMs) are transforming recommender systems from conventional preference matching toward reasoning-driven and explainable decision-making. Multi-agent collaboration further advances this paradigm by distributing recommendation reasoning across specialized agents. However, existing frameworks rarely assess the reliability of intermediate reasoning during collaboration, allowing erroneous judgments to propagate across agents and ultimately degrade recommendation quality. To address this issue, we propose Multi-Agent Debate-based Recommendation (MADRec), an evidence-grounded multi-agent framework that enhances the reliability of multi-agent recommendation reasoning. MADRec consists of three coordinated modules. First, the Multi-Agent Debate (MAD) module enables an Interaction Agent and a Semantic Agent to analyze interaction signals and content-level semantic signals from complementary perspectives, generating diverse recommendation judgments through structured debate. Second, the Evidence-Grounded Retrieval-Augmented Reasoning (EG-RAR) module grounds the Semantic Agent's reasoning in item-level evidence and refines candidate scores by assessing how well the evidence supports the corresponding semantic claims. Third, the Judge-based Ranking Calibration (JRC) module leverages debate-derived reliability cues to adaptively calibrate interaction-based and semantic ranking results for the final recommendation. Extensive experiments on three real-world datasets demonstrate that MADRec consistently outperforms competitive baselines and enhances the reliability of multi-agent recommendation reasoning. 

---
# The abc index: Exact Retrieval for Learned Sparse Representations 

**Authors**: Rossano Venturini  

**Link**: [PDF](https://arxiv.org/pdf/2611.28555)  

**Abstract**: In learned sparse retrieval (LSR), queries have tens of terms with learned weights, on which classic dynamic pruning is believed to fail, and approximate indexes such as Seismic and HNSW are considered faster than exact ones. We challenge this view with abc, an exact index built only from standard IR techniques: term-at-a-time (TAAT) evaluation, Block-Max MaxScore, binary packing and doc-id reordering. They are combined in a new design and engineered for modern CPUs. On MS MARCO encoded with four LSR models, the abc index is the fastest exact index at every k: 6.5-84x faster than the exact search of Qdrant, 13-55x faster than that of Lucene, and 2.9-18x faster than BMP with an index 5-7x smaller. Against approximate indexes, it takes 1.5-3.6x less space, and its latency grows much more slowly with k: they are up to 4.3x faster at k=10, but at k=1000 it overtakes them, by up to 7.4x. Exactness comes with an extra benefit: there are no parameters to tune. 

---
# Beyond Relevance: An Empirical Study of Multimodal LLMs for Diversity-Aware Recommendation Reranking 

**Authors**: Chengkai Liu, Yan Fu, Huiyuan Chen, Sheikh Sarwar, Qilin Qi  

**Link**: [PDF](https://arxiv.org/pdf/2611.28379)  

**Abstract**: Large language models (LLMs) have emerged as promising rerankers for recommendation, yet most existing approaches rely on textual item descriptions and primarily focus on relevance. In visually oriented domains such as video streaming, item images provide complementary signals that may also inform recommendation diversity. This raises two questions: when does visual information improve recommendation quality, and can multimodal LLMs (MLLMs) enhance relevance and diversity simultaneously? We investigate these questions through a training-free, diversity-aware multimodal reranking framework that jointly incorporates item images, textual metadata, and user interaction histories. We conduct an empirical study across three public Amazon Review datasets and an industrial Amazon Prime Video dataset, using both open-source and proprietary MLLMs. Our results reveal substantial domain-dependent effects: adding visual information yields up to 121.9% relative improvement in NDCG@5 on Prime Video, while producing limited or inconsistent gains on product recommendation datasets. Furthermore, MLLM-based reranking improves both relevance and category-diversity-sensitive ranking quality over a sequential recommendation baseline, achieving up to 71.0% improvement in NDCG@5 and 14.1% in $\alpha$-NDCG@5 on Prime Video. These findings highlight the conditional value of visual information and the potential of training-free MLLMs for relevance- and diversity-aware recommendation reranking. 

---
# DM-Lab at TREC 2026 RAG: Multi-Query Retrieval, Ensemble Judging, and Claim-Pack Generation 

**Authors**: Elyas Irankhah, Muhammad Arif, Sreeraj Ramachandran, Samah Fodeh-Jarad  

**Link**: [PDF](https://arxiv.org/pdf/2611.28233)  

**Abstract**: We present DM-Lab-RAG-2026, a framework developed for the TREC 2026 Retrieval-Augmented Generation (RAG) track. The framework integrates large-scale evidence retrieval and citation-grounded answer generation over the 553-million-document ClimbMix-400B corpus. The retrieval architecture combines multi-query BM25, reciprocal rank fusion, ensemble relevance assessment, listwise reranking, and adaptive retrieval depth. The generation pipeline employs a claim-first strategy to extract evidence-linked atomic claims, construct cited responses, and verify source support. Across 22 development topics, the retrieval system achieved a mean nDCG@30 of 0.604. In the official evaluation, it attained a set precision of 0.455 and a manual nugget gain of 0.910. Additional experiments examine the influence of reranking, evidence selection, and citation verification on system performance. The evaluation demonstrates the potential of diversified lexical retrieval and language-model-based relevance assessment for large-scale evidence discovery. It further identifies evidence preservation and structured answer construction as key considerations for reliable RAG. This work provides a reproducible framework for investigating retrieval effectiveness and evidence-grounded generation. 

---
# Evaluating Memanto on DolphinBench for AI Agents 

**Authors**: Suleman Khan, Rasa Rahnema, Tara Khani, Majid Fekri, Fattane Zarrinkalam  

**Link**: [PDF](https://arxiv.org/pdf/2611.28231)  

**Abstract**: Memory benchmarks built on question answering tell an agent which fact to recall. Workplace requests often leave that need implicit: to post an update in the team's channel, an agent must work out which channel from earlier interactions and use it in a tool call. We evaluate Memanto, a memory agent that ranks memories with an information-theoretic relevance score, on DolphinBench, which tests memory through 600 simulated workplace tasks, each validated as requiring memory. Running in the Hermes agent harness with GPT-6-Luna and at most 6 iterations per conversation, Memanto completes 503 tasks (83.83%). The strongest configuration reported by the benchmark's authors, Mem0 with GPT-5.6-Luna and up to 60 iterations, completes 424 (70.67%). Memanto's run also has the lowest total cost ($10.86, against $61.48 for the cheapest reported run) and the shortest median task latency (15.32s). Grading uses the benchmark's own judge, but the agent model differs, so these results compare complete systems. The memory system affects both whether an agent completes such tasks and what completion costs. 

---
# Orthogonally Mixed Sign Coding with Progressive Residual Refinement for Transformer Weight Compression 

**Authors**: Anindya Roy Chowdhury, Atanu Barai, Nandakishore Santhi  

**Link**: [PDF](https://arxiv.org/pdf/2611.31089)  

**Abstract**: LLM quantization reduces storage but often degrades predictive fidelity at low bit rates. We propose a training-free, CDF-free, Gaussian-CLT-motivated progressive residual-sign quantization framework that uses randomized orthogonal mixing to promote Gaussian-like residual distributions and improve sign based reconstruction of Transformer weights. Each stage independently normalizes residuals and analytically calibrates reconstruction, progressively recovering quantization errors. The framework enables flexible storage-fidelity trade-offs without learned codebooks, empirical CDFs, or trellis optimization. We evaluate GPT-2 Large and Llama-3-8B across attention and MLP projections. At five residual stages, joint quantization preserves near-baseline predictive fidelity, achieving perplexities of 32.16 and 12.73, compared with full-precision values of 29.89 and 11.07, respectively. Ablations further demonstrate that orthogonal mixing substantially improves reconstruction over identity-based sign coding, while activation-aware scaling improves output fidelity and reduced-precision statistics lower storage overhead with negligible perplexity degradation. Comparisons with conventional quantizers establish the method's progressive fidelity recovery and characterize its storage-quality and runtime trade-offs. These results demonstrate the effectiveness of orthogonally mixed residual-sign coding for flexible, training-free Transformer weight compression. 

---
# Trallie: from Free-Form Text to Structured Data with LLMs 

**Authors**: Vijayasri Iyer, Cristiano De Nobili, Mubashir Hussain Shah, Sébastien Bratières  

**Link**: [PDF](https://arxiv.org/pdf/2611.30773)  

**Abstract**: Trallie is an open-source framework that leverages large language models (LLMs) to perform information extraction (IE) from natural language sources with or without user guidance. Unlike conventional approaches that rely on extensive labeled datasets or domain-specific training, Trallie enables the conversion of unstructured data into structured formats with minimal or no representative examples. The framework is designed with three key objectives: (i) reducing manual annotation by harnessing prompt-based zero-shot and few-shot capabilities of modern LLMs, (ii) flexibly parsing diverse document formats (i.e., HTML, PDF, TXT, and JSON), and (iii) improving accessibility for non-expert users by automatically inducing schemas from document collections, which can then be refined for extraction. We evaluate Trallie with open-weight LLMs on FDA 510(k) documents and on movie and university websites from SWDE. It outperforms previously reported systems when the target attributes are provided for text extraction and is competitive when it must discover attributes for the schema automatically. The project is available at this https URL 

---
# Fetch, Read, Reuse: How AI Search Agents Bend the Web's Rules 

**Authors**: Yixi Chen, Zafar A. Qazi, Marco Canini, Suhaib A. Fahmy  

**Link**: [PDF](https://arxiv.org/pdf/2611.30546)  

**Abstract**: AI search agents form a new and fast-growing class of web traffic. Like a browser, they fetch pages on demand for one user; like a transcoding proxy, they rewrite what they fetch; and like a shared cache, they answer other users' questions from the stored copy. The rules publishers rely on, such as this http URL, Retry-After, and Cache-Control, were written for more familiar clients, so the agent's vendor decides which of them to follow. We present a black-box measurement study of how these agents fetch, read, and reuse pages, covering the search APIs of OpenAI, Anthropic, xAI, and Google and the consumer apps of ten services. We find that OpenAI and xAI do not honor this http URL, and no service acts on Retry-After. Services silently read less than the origin served: OpenAI drops some formats, no service reads JavaScript-only content, and xAI reads only the first 0.5 MB of a page. OpenAI, Anthropic, and xAI share fetched pages across users and largely ignore Cache-Control, so answers can reflect outdated pages for 30 minutes to a day or more. Publishers therefore lack effective control: directives are advisory and followed selectively, describe the served page rather than the rewritten copy, and bound how long a copy is reused but not how many answers it serves. We release our code and data. 

---
# Ontology-Compiled 2D Clinical State Fields with Bidirectional Diffusion Encoders for Zero-Training Cross-Ontology Medical Coding 

**Authors**: Zeqiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.30218)  

**Abstract**: Healthcare systems continually revise clinical taxonomies, update reimbursement schedules, and harmonize cross-border registries. Existing automated coding models bind clinical text understanding to fixed vocabulary-specific heads, requiring manual relabeling and retraining whenever code sets change. We introduce Ontology-Compiled 2D Clinical State Fields (OCSF), a vocabulary-free framework that decouples clinical document reading from target ontology indexing so one trained encoder can switch across arbitrary medical vocabularies in seconds without target training. A single-pass bidirectional diffusion-pretrained encoder and vocabulary-free decoder map each clinical note into a fixed-dimension two-dimensional state matrix with zero code-specific parameters. Any target ontology is compiled offline on CPU from public text descriptions into a scale-invariant Tikhonov deconvolution operator that suppresses sibling-code interference while preserving source decision thresholds. Trained solely on English MIMIC-IV ICD-9 discharge summaries and evaluated without target tuning across 58,706 held-out test notes and five unseen ontologies spanning English ICD-10, MS-DRG, Spanish, Greek, and Chinese (115-185,458 codes), OCSF reaches 63.15% Micro-F1 on in-domain ICD-9 (outperforming a matched closed-vocabulary classifier and eliminating late-epoch tail overfitting) and transfers directly to 26,085 English ICD-10 codes (29.23% recall at gold cardinality), MS-DRG groups, and multilingual terminologies, compiling up to 185,458 codes in under 10 seconds on CPU with sub-millisecond GPU readout. 

---
# Pixel as the Common Language: Unified Multimodal Embedding via Visual Rendering 

**Authors**: Xu Yuan, Pangjing Wu, Wenqi Fan, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2611.30069)  

**Abstract**: Multimodal embedding models map heterogeneous inputs into a shared representation space, yet text-only, image-only, and composed inputs are still presented in distinct formats and follow different routes through the model. This fragmentation prevents heterogeneous tasks from jointly optimizing the same end-to-end encoding process, limiting supervision transfer across input configurations. To address this issue, we propose unification before encoding and instantiate this paradigm in RenderEmbed. The framework uses Visual Substrate Rendering to convert heterogeneous inputs into a common visual substrate for encoding with a VLM. From the resulting visual tokens, Progressive Hierarchical Anchor Encoding produces compact retrieval representations by interleaving learnable anchors and fusing their states across layers, retaining broad context and fine-grained cues. Interventional Rendering Learning trains these representations to remain sensitive to content changes and stable across rendering variations through complementary semantic and rendering interventions. Extensive experiments on MMEB and ViDoRe v2, spanning diverse retrieval tasks, show that RenderEmbed outperforms prior state-of-the-art models at the same scale and exhibits strong out-of-domain generalization. Compared with MetaEmbed, RenderEmbed improves MMEB Overall Precision@1 by 1.7 points, with a larger 2.1-point gain on out-of-domain tasks. Transfer and scaling analyses further show that a shared visual pathway enables more effective learning from heterogeneous supervision. 

---
# STUN-Bench: Truly Unseen Tools for Evaluating Shared Tool Experience 

**Authors**: Abhijit Chakraborty, Ni Trieu, Vivek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2611.30034)  

**Abstract**: A tool router retrieves a few candidate tools per request and lets a language model choose. Methods that share experience across organizations add past requests about each tool to what the router retrieves. StableToolBench calls some of its test groups tests on "unseen" tools, yet the ToolBench instruction pool, which retrievers and experience-sharing methods learn from, contains every test query and labels every gold tool of the two single-tool groups among them. We release STUN-Bench (StableToolBench, Truly UNseen), a split whose experience pool has no item that labels or names 172 held-out tools. Unseen here means absent from shared experience and retriever training data, not from pretraining. On the split we find crowding: experience pooled from five simulated clients, added to an index of tool descriptions, outranks the descriptions of tools no client has used, which then often drop out of the candidate list. Under the Jina-v2 retriever, routing accuracy on held-out tools falls from 0.534 to 0.278. Retrievers trained for tool retrieval lose much less, and keeping experience in a separate index recovers most of the loss under Jina-v2. The split also reverses a conclusion: with every test tool labeled in the pool and our default leak filter applied, shared experience beats descriptions alone; on STUN-Bench it falls below them. The split, leak filter and evaluation harness are available at this https URL. 

---
# SHAKTI: Evidence-Grounded Conversational AI for Nutrition and Health 

**Authors**: Chaitanya Pallerla, Dongyi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.29988)  

**Abstract**: Large language models (LLMs) answer many diet and health questions but can state unsupported claims and cite sources that do not support them. We present SHAKTI (Smart Health Advisor for Knowledge-based Tailored Interventions), a conversational assistant that answers only from retrieved scholarly papers and exact food-composition data, and evaluate it with retrieval, generation, grounding, efficiency and contextawareness metrics. We indexed 484,368,086 OpenAlex works, searched exactly on a GPU for evaluation (10,065 NFCorpus, BioASQ, SciFact and PubMedQA questions) and through a compressed 230-million-abstract index in deployment. On 1,000 stratified questions, we compared single-shot, multi-hop, corrective and deep-research retrieval-augmented generation (RAG) with Claude Sonnet. SHAKTI adds conversational query rewriting, USDA FoodData Central lookups, source-link checks and online verification of quotes and numbers. It was tested in 36 scripted conversational turns, a 248-question memory-depth test and a 38-question load test. Claims were checked by a Claude Opus judge against exactly the text the system saw. MedCPT reranking of hybrid retrieval gave the best MRR (0.21) and MAP@100 (0.14). Across existing RAG methods, faithfulness was 0.95-0.96, claim hallucination 0.5-0.8% and context precision up to 0.98; the deep-research agent answered 73.2% of BioASQ questions correctly. In conversation, SHAKTI reached a context-awareness score of 97.1%, faithfulness 95.5%, 1.9% unsupported claims, no contradicted claims and 229/229 working source links; it resolved references to the first question after 16 unrelated questions. An answer cost 14,716 tokens (US$0.06) with a median latency of 24 s. Large-scale retrieval with agentic evidence selection and mechanical verification yields auditable, low-hallucination nutrition answers. 

---
# Look Before You Leap: Re-Architecting Disk-Resident Vector Search with TriVS 

**Authors**: Yizou Chen, Baotong Lu, Peiqi Yin, Zhisheng Hu, Qi Chen, Ming-Chang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2611.29948)  

**Abstract**: Disk-resident approximate nearest-neighbor search (ANNS) scales vector search beyond DRAM, yet state-of-the-art systems remain I/O-bound, even with fast SSDs and sophisticated I/O execution. To understand this bottleneck, we conduct an I/O-centric analysis that breaks each query down into an approach phase that locates a query-near region and a converge phase that resolves the final top-k. This analysis reveals complementary I/O tradeoffs: cluster-based search circumvents approach-phase I/O through in-memory routing but reads coarse-grained, full clusters during convergence, whereas graph-based search employs fine-grained, vector-level I/O throughout, enabling efficient convergence at the cost of inevitable I/O during the approach phase.
To combine the complementary I/O benefits of the two granularities, we present TriVS, a new paradigm that re-architects disk-resident vector search as a trilogy: approach coarsely, distill progressively, and converge exactly. Its key innovations come in two parts. First, it decouples the approach and converge phases from a single search granularity, employing cluster-based search to avoid approach-phase I/O and leveraging vector-level I/O for efficient convergence. Second, to bridge the two granularities, it introduces a new in-memory distill phase between the original two, progressively reducing routed clusters to a vector-level shortlist using compact quantized vectors before disk access. Moreover, coarser clustering shrinks the centroid-routing index, allowing TriVS to reallocate existing memory to distillation rather than expand its memory budget. Our evaluation on five datasets spanning diverse vector types and scales shows that, at Recall@10=0.9, TriVS achieves on average 3.42X PipeANN's throughput, 51.2% lower latency, and 63.7% less disk I/O, while using less memory and storage than PipeANN. 

---
# Where Do Fact-Checking Errors Arise? Separating Retrieval, Search Control and Judgement in Evidence-Grounded Verification 

**Authors**: Martin Pavlíček, Martin Kotyrba, Eva Volná, Tomáš Filip, Petr Sosík  

**Link**: [PDF](https://arxiv.org/pdf/2611.29697)  

**Abstract**: An evidence-grounded fact-checker can fail because it does not find the evidence or because it does not use it; end-to-end accuracy does not tell the two apart. We present a protocol that follows the reference evidence from the store to the verdict and separates retrieval, search control, judgement and verdict validation. We assess 293 claims from FEVER, FEVER 2.0 and AVeriTeC with one open-weights 9B model under two evidence conditions: small stores of annotated evidence, where discovery is easy, and the full source collections. In the control condition a single hybrid retrieval pass delivers complete reference evidence for 226 of 228 evidence-bearing claims, and adaptive controllers and a tool agent spend two to nine times as many model calls without a consistent gain. Of 34 decisive claims labelled as lacking evidence, 33 have the complete evidence in the packet: in 15 the judge abstains, and in 18 validation rules overturn a decisive verdict. On AVeriTeC these rules lower single-pass accuracy by six points, to 65.0% (balanced accuracy 57.6%). Ten judges from eight model families also show abstentions and validation conversions on fixed packets. Against the full stores hybrid single-pass accuracy falls by 17 to 29 points. Wikipedia evidence recovery declines; AVeriTeC recovery changes from annotated answers to cited pages and is measured separately. No controller recovers the accuracy loss. The agent also shows no resolved gain on 714 further Wikipedia claims. Evidence recovery, judgement and validation should be measured separately before an error is attributed to search. 

---
# VistaTree: Question Answering over Long Multimodal Documents with Image-Derived Summary Trees 

**Authors**: Yexiang Cheng, Xiaochen Zheng, David Tokar, Tatyana Y. Doktorova  

**Link**: [PDF](https://arxiv.org/pdf/2611.29419)  

**Abstract**: Answering questions over long multimodal documents requires locating relevant evidence across pages containing text, tables, and figures. When the document exceeds the model's context window, selective retrieval becomes necessary. We introduce VistaTree, a document question-answering agent that builds a summary tree directly from rendered page images. VistaTree segments documents into contiguous page spans, recursively groups them into a topic hierarchy, and refines sibling summaries to distinguish neighboring sections. At query time, the agent uses node summaries to select relevant nodes and reads the corresponding pages, using hierarchical navigation when the full outline exceeds the context budget. Experiments on MMLongBench-Doc and LongDocURL evaluate answer quality and retrieval cost using a locally hosted vision-language model, and quantify the performance gap between global and hierarchical access to the index. 

---
# LLM-Based Assistant for Dog Care Guidance 

**Authors**: Abubakar Sadiq Shittu, Venkat Gopu, Arshita Sharma, Jacob Kennedy  

**Link**: [PDF](https://arxiv.org/pdf/2611.29286)  

**Abstract**: This study investigates whether domain grounding through Retrieval-Augmented Generation (RAG) and parameter-efficient fine-tuning using Low-Rank Adaptation (LoRA) can enhance the accuracy and reliability of pet care advice generated by large language models (LLMs). We curated a dataset of breed-specific dog care information and used it to develop two systems: a RAG pipeline based on TinyLlama-1.1B-Chat-v1.0, and a fine-tuned LLaMA-3.2-1B model adapted using LoRA. Both systems were evaluated on 15 dog care questions. The RAG model achieved a perfect average score of 2.0, outperforming the fine-tuned model (mean = 1.47). Statistical analyses, including Welch's t-test, Mann-Whitney U, and Wilcoxon signed-rank tests (all p < 0.05), confirmed the significance of the performance gap. Our findings suggest that retrieval-based domain grounding leads to more accurate, consistent, and trustworthy responses than standalone fine-tuning in specialized LLM applications. 

---
# NoeVera: Can Knowledge Guide Experiments and Experiments Refine Knowledge? 

**Authors**: Pinaki Prasad Guha Neogi, Spyros Mastorakis, Ankur Mallick, Saravan Rajmohan, Victor Rühle  

**Link**: [PDF](https://arxiv.org/pdf/2611.29255)  

**Abstract**: With the growing demand for large language model (LLM) services, optimizing LLM serving systems is becoming increasingly complex, and the bottleneck is no longer predicting how a given configuration will perform. The harder problem is deciding what to investigate: which mechanisms matter for a workload, which configurations are worth testing, and what evidence a decision still needs. In this work, we present NoeVera, an agentic Second Brain that goes beyond just one-shot retrieval by turning papers, code, benchmarks, and experimental results into a persistent, provenance-preserving knowledge base with three planes: source evidence, reusable synthesis, and agent memory. Decision-aware retrieval assembles compact evidence packs, role-specific agents reason over them and invoke a simulator (Vidur or GAUSS) whenever quantitative support is needed, and every finding is written back as a scoped evidence record, so knowledge guides experiments and experiments refine knowledge. In configuration search over four serving scenarios, NoeVera comes within 5% of the optimum in a median of 6-7 simulator calls, versus 9 for Bayesian optimization and 15 for a knowledge-free agent with the same tools; on A100 GPUs, its representative configurations deliver ~1.30x vLLM-default throughput and meet every scenario's latency objective. When reused, provenance-preserving evidence records over-generalize on only 8% of applicability traps, versus 28% for a general-purpose agent memory. On the held-out 48-task split of a 72-task benchmark, NoeVera scores 29% higher than an equally equipped agent and 36% higher than its single-call retrieval counterpart (0.80 vs. 0.62 and 0.59), and even when baselines receive NoeVera's own prompts, a statistically significant part of its evidence-evolution advantage remains. 

---
# Zarc: Exact Nearest Neighbors over a Billion Vectors in Seconds 

**Authors**: Dongfang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2611.29252)  

**Abstract**: Exact k-nearest-neighbor vector search lies on the critical path of machine learning systems, from sparse attention in the latest large language models (LLMs) to long-context memories and training-data deduplication. Exhaustive GPU search scans the whole dataset for every query, while approximate indexes return inexact neighbors. Recent methods skip almost every vector with a cheap half-precision proof that it lies too far away, but only on one GPU that holds the whole dataset. Our insight is to keep only the short rows of these proofs in GPU memory and to fetch the few remaining full vectors from host memory. Our system, Zarc, answers Euclidean, cosine, inner-product, and filtered queries exactly under updates. On four H100 GPUs, it answers 10,000 exact queries over one billion vectors in 1.1 seconds instead of five minutes. On eleven open LLMs, it retrieves the exact top-10 attention keys 2.2x to 13.1x faster than exhaustive search. In sparse attention over one million tokens, it runs 3.1x faster than bf16 selection with FlashInfer, which misses the exact top-64 for most queries. 

---
# DataRecycle: Trustworthy Distillation of Structured Data from Scattered Web Evidence 

**Authors**: Can Liu, Tengteng Zhu, Xinyu Jiang, Xiaolin Wen, Yong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2611.29206)  

**Abstract**: The web contains vast amounts of scattered quantitative evidence that is difficult to organize into comparable and verifiable structures. Yet its unstructured, sparse, and sometimes conflicting nature makes a systematic distillation costly and error-prone. Existing approaches force users to make a trade-off: search engines return document lists that require labor-intensive extraction and reconciliation, while generative AI systems produce fluent summaries that obscure data provenance, introduce reporting bias, and hide inconsistencies. We present DataRecycle, an interactive system that helps transform scattered web evidence into structured tables of comparable numerical facts. Given a user's query or statement, DataRecycle decomposes it into entities and attributes, retrieves candidate evidence, and organizes scattered, potentially conflicting records in a table-based evidence view. To address key challenges in automated data distillation (i.e., incomplete retrieval, untraceable generation, sparse data distribution, and multi-source inconsistency), DataRecycle adopts an ontology-guided, top-down data acquisition strategy, visualizes the recorded lineage from retrieved snippets to extracted values, and aligns heterogeneous data across dimensions such as time, definition, and scale. We evaluate DataRecycle through module-level experiments, a usage scenario, and a user study, showing that the system helps users organize numerical web evidence into provenance-linked tables and inspect uncertainty and conflicting sources. 

---
# Bridging Blind Spots With Words: Decentralized Multi-View Spatial Reasoning via Language-Constrained Communication 

**Authors**: Soumyabrata Chaudhuri, Joydeep Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2611.29112)  

**Abstract**: As multimodal LLMs (MLLMs) get increasingly used as collaborative partners across embodied robotics, AR/VR, and multi-user systems, spatial reasoning is now often distributed across agents with partial environment visibility, necessitating spatial reasoning integration. Prior work in language-mediated spatial integration is however strictly limited to simplified two-agent scenarios. To address this gap, we first formalize the general problem of decentralized multi-view spatial reasoning, where N agents observe distinct viewpoints, exchange budget-constrained natural language messages, and collaborate to enable a listener to answer spatial queries. We then introduce a novel multi-view spatial benchmark to isolate queries that demand non-trivial multi-view integration. Through an extensive empirical study, we discover that the penalty of restricting decentralized spatial reasoning to natural language with limited budget is far smaller than suggested by prior two-agent literature. Nevertheless, we observe that substantial headroom still remains for multi-agent spatial reasoning in this setting, as evidenced by a significant gap between current centralized performance and an oracle best-of-K ceiling. Moreover, we show that reinforcement learning with credit assignment substantially bolsters the decentralized model, effectively closing the gap with the centralized baseline. 

---
# HME-PFD: Heterogeneous Mixture-of-Experts Privileged Feature Distillation 

**Authors**: Wen Xu, Xiaomin Zhang, Yucheng Yang, Danny Liu, Xunlong Gui, Fei Huang, Sha Meng  

**Link**: [PDF](https://arxiv.org/pdf/2611.29094)  

**Abstract**: Knowledge distillation (KD) is widely used to transfer knowledge from large teacher models to smaller student models, facilitating efficient serving of machine learning systems. In the context of ads recommendation, privileged features distillation (PFD) extends KD by allowing the teacher model to leverage privileged features, i.e., features available only during training, to enhance distillation performance. However, in real-world scenarios, privileged features often have limited coverage and exhibit heterogeneous distributions, resulting in training examples of varying importance for distillation and model learning. We propose Heterogeneous Mixture-of-Experts Privileged Feature Distillation (HME-PFD), a novel approach that incorporates privileged features into the routing module of a Mixture-of-Experts-based student model. Our method introduces both hard-coded and learnable routing modules, enabling the student to dynamically determine expert assignment based on the availability and values of privileged features. This design allows the student model to better handle the heterogeneous data scenario, where privileged features are partially observed, by differentiating between high- and low-signal samples. Extensive experiments on two post-click conversion prediction tasks demonstrate that HME-PFD consistently outperforms baseline methods. Our results show that leveraging privileged features for expert routing in the student model leads to improved model performance, even when privileged features are incomplete or sparse. 

---
# GERF: A Token-Efficient Multimodal Retrieval Framework for Schema-Guided Scientific Information Extraction 

**Authors**: Yiyu Qian, Xing Wu  

**Link**: [PDF](https://arxiv.org/pdf/2611.28295)  

**Abstract**: Scientific information extraction (IE) is a fundamental component of AI for Science, transforming experimental knowledge embedded in scientific literature into structured and traceable data for evidence synthesis, scientific database construction, and data-driven discovery. However, critical experimental details are often distributed across text, tables, figures, captions, and surrounding discussion, making them difficult to recover using conventional text-only extraction. Existing LLM-based and multimodal RAG pipelines typically rely on full-document context or coarse page-level retrieval, introducing substantial irrelevant evidence and high textual and visual token costs. We propose GERF, a Graph-based Evidence Retrieval Framework for token-efficient, schema-guided multimodal scientific IE. GERF represents each article as a heterogeneous evidence graph that connects schema fields, text chunks, figures, and figure-derived content through semantic and structural relations. Using schema fields as retrieval anchors, GERF combines direct semantic retrieval, Random Walk with Restart, and null-aware evidence expansion to construct compact, field-specific multimodal evidence packages. Visual interpretation is applied only to graph-selected figures, avoiding indiscriminate processing of the entire document. We evaluate GERF from two complementary perspectives: evidence retrieval quality and pipeline-level input-token consumption. On the expert-annotated SOFC-Exp corpus, GERF retrieves gold-supported evidence for 94.55% of the evaluated field instances, providing broad field-level evidence coverage for downstream extraction. In the token comparison, GERF reduces average input-token usage by 73.40% relative to full-context processing and uses fewer tokens than the evaluated page-level pipelines. 

---
# Close the Loop on the Retrievability Gap 

**Authors**: Guangyuan Dong, Wenhao Li, Zhen Tian, Jiaqi Zhang, Haoyu Wang, Junhao Xu, Mingzhe Liu, Pan Wang, Fei Shen, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2611.28146)  

**Abstract**: A frozen vision-language encoder now guides image restoration and later indexes what it produced. Over a fixed set of eighteen pansharpening systems on one WorldView-III test set, the fidelity index that ranks them, $Q2^n$ with a reference and HQNR without one, picks the less retrievable of two systems one time in three and one in two; the disagreement lies between the pixel-supervised and the language-conditioned family, and even the best system falls short of the reference, small objects first. We call this the retrievability gap and trace it to the area weighting of position-averaged losses. LEDGER closes the loop around the encoder: it indexes the input into a set of named concepts, generates, re-encodes the output alone with the same frozen encoder, and minimises a set-level soft-Jaccard loss on the coverage of that set, a label-free target available at test time, so that every present concept of a scale receives a gradient share independent of its area. Measured outside the loop by label-grounded probes, held-out retrievers, language-free evaluators, detectors and human raters, LEDGER matches or exceeds the strongest baselines in fidelity on four sensors, raises retrieval by seven to nine points within its vocabulary and two to five outside it, closes two thirds of the reduced-resolution gap to the reference, lifts extraction on raw SpaceNet imagery above the vendor product in each of four cities, and transfers to existing generators as a training-only loss at no inference cost. 

---
# Monte Carlo Power Analysis for Small-System A/B Trials: Detecting Differences in Newsletter Engagement 

**Authors**: Michael D. Ekstrand, Daniel Kluver, Karl Higley, Bart P. Knijnenburg  

**Link**: [PDF](https://arxiv.org/pdf/2611.28018)  

**Abstract**: Good experimental practice and predicting the scientific usefulness of an experiment or experimental platform require power analysis: estimating, *a priori*, how likely the experiment is to detect the intended effect if indeed it exists. While simple experimental designs admit well-understood power analysis methods, more sophisticated experimental designs and settings often require bespoke techniques to avoid either over- or under-estimating experimental power. We present the Monte Carlo method we use to estimate the power of experiments intended to increase user engagement with personalized e-mail newsletters of recommended news articles. 

---
