# IUU+DB: Tracking Illegal, Unreported, and Unregulated Fishing, Seafood Fraud, and Labor Abuse through LLM-driven Information Extraction 

**Authors**: Henry Bodwell, Hong Yang, John C. Simeone, Kelvin Gorospe, Bella Sullivan, Lana Huang, Jessica Gephart, Sandy Aylesworth, Molly Masterton, Naren Ramakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2606.18181)  

**Abstract**: Illegal, unreported, and unregulated fishing (IUU) traditionally refers to fishing activities that violate applicable laws or occur in areas that lack applicable laws. We propose the term IUU+ to capture a broader suite of fisheries sector environmental and associated supply chain trade-related crimes and behaviors. Although IUU+ activity is widely recognized as a serious threat to marine ecosystems, markets, and livelihoods, a quantitative understanding of these incidents, e.g., their frequency, geography, species, actors, and patterns in the type of illicit activity, remains difficult to obtain. We propose IUU+DB, a large language model driven system for building a global incident database of IUU+ activity. The system ingests heterogeneous documents, classifies whether they describe relevant incidents, extracts key data elements such as actors, locations, species, vessels, violations, and enforcement outcomes, and supports deduplication and trend analysis. Case studies and validation results show that IUU+DB can help organize fragmented evidence, surface geographic and behavioral hotspots, support fisheries-domain specific research in academia and non-government organizations, assist source and species risk assessments for industry, and provide support for policy implementation and targeted enforcement efforts to government agencies. 

---
# Non-negative Elastic Net Decoding for Information Retrieval 

**Authors**: Koki Okajima, Yasutoshi Ida, Tsukasa Yoshida, Yasuaki Nakamura  

**Link**: [PDF](https://arxiv.org/pdf/2606.17910)  

**Abstract**: Dense retrieval has become the dominant paradigm in information retrieval, in which each document is scored against a query by the inner product of their vector embeddings, and the top-$k$ documents by score are retrieved for this query. However, since each document's score depends solely on the embedding of the query and itself, the retrieval process is oblivious to the content of the entire corpus. Therefore, dense retrieval cannot avoid selecting semantically similar documents from the corpus, which may result in a non-diverse, redundant set of retrieved documents. To this end, we approach retrieval as a joint decoding problem, in which documents are selected as a set with regard to the context of the rest of the corpus. To achieve this, we propose Non-Negative elastic Net (NNN) decoding, which selects documents whose embeddings jointly reconstruct the query embedding as a sparse non-negative linear combination.
Our main theoretical result establishes a strict separation between dense retrieval and NNN decoding. For any corpus, every query correctly handled by dense retrieval is also handled by NNN decoding, while on corpora containing correlated documents, NNN decoding additionally handles queries that dense retrieval cannot. Experimental results indicate that applying NNN decoding to frozen embeddings trained for inner-product scoring yields consistent improvements across several benchmarks. Moreover, we introduce an end-to-end training procedure which optimizes the embeddings for NNN decoding, producing significant performance gains surpassing in all metrics and benchmarks compared to dense retrieval. Our work establishes a new paradigm for leveraging dense embeddings in information retrieval, beyond the standard practice of inner-product scoring. 

---
# Understanding and Debugging Failures in N-Gram-Based Generative Retrieval 

**Authors**: Richard Takacs, Adrian Bracher, Svitlana Vakulenko  

**Link**: [PDF](https://arxiv.org/pdf/2606.17721)  

**Abstract**: Generative Retrieval (GR) is an emerging Information Retrieval (IR) paradigm that is motivated by increasingly capable language models. In GR, a model directly generates identifiers for relevant documents. While these systems offer unique advantages, they also introduce distinct failure mechanisms. We explore these failure modes in three contributions: (1) We present a taxonomy of GR failure modes based on GR literature. (2) We empirically investigate failure in a subset of GR: ngram-based methods, more specifically, SEAL and MINDER. Our analysis reveals common issues, such as ambiguous docids, low identifier diversity, and the disproportionate impact of specific identifiers. (3) We introduce a new web-based tool that helps the IR community analyze generated ngrams and their respective contribution to the final ranking, providing an intuitive interface to identify where such GR methods go wrong. 

---
# Do Generative Recommenders Deepen the Information Cocoon? A Closed-Loop Simulation with LLM-powered User Simulators 

**Authors**: Jiyuan Yang, Gengxin Sun, Mengqi Zhang, Lingjie Wang, Yuanzi Li, Hongxi Cui, Xin Xin, Pengjie Ren  

**Link**: [PDF](https://arxiv.org/pdf/2606.17707)  

**Abstract**: Recommender systems alleviate information overload, yet repeated feedback between recommendations and user interactions can reinforce existing preferences and narrow users' exposure, forming information cocoons. While this phenomenon has been widely studied in traditional sequential recommendation, its impact on generative recommendation remains unclear. By replacing atomic item IDs with Semantic ID (SID) sequences, generative recommenders introduce a different recommendation mechanism whose role in information cocoon formation is not yet understood. To investigate whether generative recommenders deepen information cocoons, we propose \textsc{RecLoop}, a closed-loop simulation framework with LLM-driven user agents. We compare two generative recommenders and two traditional sequential baselines on two Amazon datasets across multiple feedback cycles. In addition to standard exposure-level metrics, we introduce \emph{Code-Space Structural Cocoon}, a model-level metric that measures concentration in the generated SID space. Experimental results show that generative recommenders are generally less prone to exposure-level cocoon formation than traditional baselines, preserving broader exposure diversity and slowing cross-user homogenization. However, feedback loops can still induce concentration within the generated SID space. We further find that cocoon severity depends strongly on tokenization strategy and model scale: collaborative-signal tokenization produces stronger cocoon effects than semantic tokenization, whereas larger models maintain greater code-space diversity and better retain access to niche content. These findings suggest that information cocoons in generative recommendation are shaped not only by recommendation behavior, but also by item tokenization and model capacity. Our code is available at this https URL. 

---
# Temporal Preference Optimization for Unsupervised Retrieval 

**Authors**: HyunJin Kim, Jaejun Shim, Young Jin Kim, JinYeong Bak  

**Link**: [PDF](https://arxiv.org/pdf/2606.17664)  

**Abstract**: Unsupervised dense retrievers offer scalability by learning semantic similarity from unlabeled documents via contrastive learning, but they struggle to capture the temporal relevance, retrieving semantically related but temporally misaligned documents-an important aspect when a document collection spans multiple time periods (e.g., retrieving documents from 2018-2025 for "Who is the president in 2019?" introduces temporal ambiguity). Existing methods rely on supervised training with explicit timestamps, which are not always feasible. We propose TPOUR (Temporal Preference Optimization for Unsupervised Retriever), which uses our novel training method Temporal Retrieval Preference Optimization (TRPO). TRPO reinterprets preference learning in the temporal dimension, guiding the retriever to favor temporally aligned documents. TPOUR further generalizes to unseen time periods via interpolation in a learned time embedding, enabling continuous temporal alignment. Experiments on temporal information retrieval (T-IR), TPOUR outperforms both unsupervised and supervised baselines. Compared to Qwen-Embedding-8B, despite being about 72.7x smaller, TPOUR Contriever improves average nDCG@5 by +4.04 (+12.15%) on explicit and +4.98 (+15.21%) on implicit queries. We provide our code at this https URL. 

---
# RSRank: Learning Relevance from Representational Shifts 

**Authors**: Archit Gupta, Sai Sundaresan, Debabrata Mahapatra  

**Link**: [PDF](https://arxiv.org/pdf/2606.17468)  

**Abstract**: As enterprises deploy RAG-based systems to provide grounded responses to user queries, reranking has become a critical component for the final filtering step that separates relevant from distracting or irrelevant documents. Existing rerankers often rely on heuristic thresholds to achieve optimal filtering. Moreover, for relevance scoring, state-of-the-art methods use a language model's logit signals, which are designed for next-token prediction, not for assessing relevance. To address these limitations, we identify a principled signal for relevance: the representational shift (RS) induced in a query's internal state when conditioned on a document. We observe that the alignment between (a) RS induced by a candidate document and (b) RS induced by an oracle document-set provides a robust indicator of relevance. Building on this insight, we introduce a lightweight training framework that learns projections mapping RS to calibrated relevance scores. Our training objectives naturally filter irrelevant content at a zero threshold, reducing dependence on heuristic tuning. Across diverse retrieval datasets, our method delivers gains over SOTA rerankers. 

---
# On the Memorization Behavior of LLMs in Generative Recommendation: Observations, Implications, and Training Strategies 

**Authors**: Sunwoo Kim, Sunkyung Lee, Clark Mingxuan Ju, Donald Loveland, Bhuvesh Kumar, Kijung Shin, Neil Shah, Liam Collins  

**Link**: [PDF](https://arxiv.org/pdf/2606.17276)  

**Abstract**: Generative recommendation (GR) has emerged as a promising direction for recommender systems. Recently, large language models (LLMs) have been increasingly adopted for GR, as their rich pretrained knowledge is expected to help them generalize beyond common user behavior patterns that traditional memorization-oriented baselines can capture. However, existing LLM-based GR works largely ignore LLMs' well-known tendency to memorize, which, if present in LLMs fine-tuned for GR, would restrict their utilization of pretrained knowledge. In this work, we investigate this concern by examining one-hop memorization, where a model recommends items that are direct successors of items in the training data. We show that LLMs do this more than non-LLM-based GR models-in fact, the vast majority of their gains over GR baselines are actually on users whose target items can be predicted through one-hop memorization. We intuit that improving performance on the remaining users requires LLMs to learn richer item-item relations beyond one-hop transitions. To achieve this, we propose IIRG, a novel training strategy that teaches LLMs to capture: (1) collaborative relations derived from item co-occurrences across multiple hops in user sequences, and (2) semantic relations among items with similar themes, both of which can serve as useful recommendation signals. We show that IIRG significantly improves over LLMs trained solely with standard next-item prediction, with especially large gains for users whose test items are not covered by train-time one-hop transitions. 

---
# HistoRAG: Embedding Historical Methodology in Retrieval-Augmented Generation Through Critical Technical Practice 

**Authors**: Noah J. Kim-Baumann, Torsten Hiltmann  

**Link**: [PDF](https://arxiv.org/pdf/2606.18103)  

**Abstract**: Retrieval-Augmented Generation (RAG) is the prevailing architecture for grounding language model outputs in external evidence, yet its dominant evaluation paradigms and default configurations remain oriented toward factual question-answering. For interpretive disciplines such as historical studies, RAG embeds assumptions that conflict with scholarly practice. We introduce HistoRAG, a framework that translates historiographical principles into concrete architectural interventions. Separated retrieval and generation decouples source discovery from interpretation, temporal windowing enforces balanced source representation across the research period as a methodological requirement of historical inquiry, and LLM-as-judge evaluation makes relevance judgments transparent and contestable. We evaluate these interventions using SPIEGELragged, applied to 102,189 articles from Der Spiegel (1950-1979). Each intervention addresses a measurable deficiency in standard RAG: era-specific vocabulary retrieves zero chunks from the 1950s when using 1970s terminology, evidence of the temporal skew that motivates windowing; vector similarity and LLM-assessed relevance correlate only weakly (Spearman rho = 0.275), motivating post-retrieval evaluation; and keyword-based and semantic retrieval surface largely disjoint source pools, motivating an architecture in which both operate as complementary retrieval layers under a shared LLM evaluation filter. We also introduce the concept of Zwischentexte (intermediate texts that function as interpretive proposals rather than findings) as a framework for responsible integration of LLM-generated text into scholarly practice. The architecture offers a model for how domain-specific epistemological commitments can be translated into RAG design decisions, and may transfer to other interpretive disciplines working with large corpora. 

---
# Designing Recommendation Exposure and Favorite Lists: A Field Experiment in a Spot-Work Platform 

**Authors**: Kazuki Sekiya, Suguru Otani, Yuki Komatsu, Shunsuke Ozeki, Shunya Noda  

**Link**: [PDF](https://arxiv.org/pdf/2606.17397)  

**Abstract**: How should recommender systems be designed when recommendations shape access to scarce, short-lived opportunities? We study this question in a production setting: Timee, Japan's largest platform for spot work, where workers favorite job templates and receive notifications when firms post shifts from those templates. Maximizing predicted favoriting can generate misdirected concentration: recommendations accumulate on popular templates that create few viable job openings, while templates with unmet labor demand receive too little exposure. We design exposure-control mechanisms for favorite-list management, reallocating template exposure based on posting activity and unfilled capacity. The proposed recommender, thresholded eligibility control (TEC), is fully parallelizable and suitable for large-scale digital platforms. In simulations calibrated to Timee data, TEC raises the per-round job-finding rate from 57.6\% to 70.0\%. A prefecture-level randomized field experiment increases realized matches and exposure per active template, reduces the share of low-exposure templates, and improves impression-level favoriting and downstream matching. 

---
# Beyond Parallel Sampling: Diverse Query Initialization for Agentic Search 

**Authors**: Sidhaarth Murali, João Coelho, Jingjie Ning, João Magalhães, Bruno Martins, Chenyan Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2606.17209)  

**Abstract**: Test-time scaling for agentic search typically increases depth (i.e., more turns and tokens per trajectory) or breadth (i.e., more parallel rollouts). Here we focus on breadth scaling, showing that standard parallel sampling yields diminishing returns, tracing this to query redundancy at the first turn. When models issue similar first queries across rollouts, the threads retrieve overlapping evidence, and subsequent turns are conditioned on this shared retrieval. We address this limitation with DivInit, a training-free intervention at the first turn. Rather than sampling k independent first queries, DivInit draws n candidates from a single call, picks k < n diverse seeds, and runs them as parallel trajectories. Across five open-weight models and eight benchmarks, DivInit consistently improves over standard parallel sampling, with average gains of five to seven points on multi-hop QA at matched compute. Code available at this https URL 

---
