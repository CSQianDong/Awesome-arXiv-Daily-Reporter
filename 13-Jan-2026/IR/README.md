# AptaFind: A lightweight local interface for automated aptamer curation from scientific literature 

**Authors**: Geoffrey Taghon  

**Link**: [PDF](https://arxiv.org/pdf/2601.07684)  

**Abstract**: Aptamer researchers face a literature landscape scattered across publications, supplements, and databases, with each search consuming hours that could be spent at the bench. AptaFind transforms this navigation problem through a three-tier intelligence architecture that recognizes research mining is a spectrum, not a binary success or failure. The system delivers direct sequence extraction when possible, curated research leads when extraction fails, and exhaustive literature discovery for additional confidence. By combining local language models for semantic understanding with deterministic algorithms for reliability, AptaFind operates without cloud dependencies or subscription barriers. Validation across 300 University of Texas Aptamer Database targets demonstrates 84 % with some literature found, 84 % with curated research leads, and 79 % with a direct sequence extraction, at a laptop-compute rate of over 900 targets an hour. The platform proves that even when direct sequence extraction fails, automation can still deliver the actionable intelligence researchers need by rapidly narrowing the search to high quality references. 

---
# GAP-Net: Calibrating User Intent via Gated Adaptive Progressive Learning for CTR Prediction 

**Authors**: Ke Shenqiang, Wei Jianxiong, Hua Qingsong  

**Link**: [PDF](https://arxiv.org/pdf/2601.07613)  

**Abstract**: Sequential user behavior modeling is pivotal for Click-Through Rate (CTR) prediction yet is hindered by three intrinsic bottlenecks: (1) the "Attention Sink" phenomenon, where standard Softmax compels the model to allocate probability mass to noisy behaviors; (2) the Static Query Assumption, which overlooks dynamic shifts in user intent driven by real-time contexts; and (3) Rigid View Aggregation, which fails to adaptively weight heterogeneous temporal signals according to the decision context. To bridge these gaps, we propose GAP-Net (Gated Adaptive Progressive Network), a unified framework establishing a "Triple Gating" architecture to progressively refine information from micro-level features to macro-level views. GAP-Net operates through three integrated mechanisms: (1) Adaptive Sparse-Gated Attention (ASGA) employs micro-level gating to enforce sparsity, effectively suppressing massive noise activations; (2) Gated Cascading Query Calibration (GCQC) dynamically aligns user intent by bridging real-time triggers and long-term memories via a meso-level cascading channel; and (3) Context-Gated Denoising Fusion (CGDF) performs macro-level modulation to orchestrate the aggregation of multi-view sequences. Extensive experiments on industrial datasets demonstrate that GAP-Net achieves substantial improvements over state-of-the-art baselines, exhibiting superior robustness against interaction noise and intent drift. 

---
# Loci Similes: A Benchmark for Extracting Intertextualities in Latin Literature 

**Authors**: Julian Schelb, Michael Wittweiler, Marie Revellio, Barbara Feichtinger, Andreas Spitz  

**Link**: [PDF](https://arxiv.org/pdf/2601.07533)  

**Abstract**: Tracing connections between historical texts is an important part of intertextual research, enabling scholars to reconstruct the virtual library of a writer and identify the sources influencing their creative process. These intertextual links manifest in diverse forms, ranging from direct verbatim quotations to subtle allusions and paraphrases disguised by morphological variation. Language models offer a promising path forward due to their capability of capturing semantic similarity beyond lexical overlap. However, the development of new methods for this task is held back by the scarcity of standardized benchmarks and easy-to-use datasets. We address this gap by introducing Loci Similes, a benchmark for Latin intertextuality detection comprising of a curated dataset of ~172k text segments containing 545 expert-verified parallels linking Late Antique authors to a corpus of classical authors. Using this data, we establish baselines for retrieval and classification of intertextualities with state-of-the-art LLMs. 

---
# RLPO: Residual Listwise Preference Optimization for Long-Context Review Ranking 

**Authors**: Hao Jiang, Zhi Yang, Annan Wang, Yichi Zhang, Weisi Lin  

**Link**: [PDF](https://arxiv.org/pdf/2601.07449)  

**Abstract**: Review ranking is pivotal in e-commerce for prioritizing diagnostic and authentic feedback from the deluge of user-generated content. While large language models have improved semantic assessment, existing ranking paradigms face a persistent trade-off in long-context settings. Pointwise scoring is efficient but often fails to account for list-level interactions, leading to miscalibrated top-$k$ rankings. Listwise approaches can leverage global context, yet they are computationally expensive and become unstable as candidate lists grow. To address this, we propose Residual Listwise Preference Optimization (RLPO), which formulates ranking as listwise representation-level residual correction over a strong pointwise LLM scorer. RLPO first produces calibrated pointwise scores and item representations, then applies a lightweight encoder over the representations to predict listwise score residuals, avoiding full token-level listwise processing. We also introduce a large-scale benchmark for long-context review ranking with human verification. Experiments show RLPO improves NDCG@k over strong pointwise and listwise baselines and remains robust as list length increases. 

---
# Towards Multi-Behavior Multi-Task Recommendation via Behavior-informed Graph Embedding Learning 

**Authors**: Wenhao Lai, Weike Pan, Zhong Ming  

**Link**: [PDF](https://arxiv.org/pdf/2601.07294)  

**Abstract**: Multi-behavior recommendation (MBR) aims to improve the performance w.r.t. the target behavior (i.e., purchase) by leveraging auxiliary behaviors (e.g., click, favourite). However, in real-world scenarios, a recommendation method often needs to process different types of behaviors and generate personalized lists for each task (i.e., each behavior type). Such a new recommendation problem is referred to as multi-behavior multi-task recommendation (MMR). So far, the most powerful MBR methods usually model multi-behavior interactions using a cascading graph paradigm. Although significant progress has been made in optimizing the performance of the target behavior, it often neglects the performance of auxiliary behaviors. To compensate for the deficiencies of the cascading paradigm, we propose a novel solution for MMR, i.e., behavior-informed graph embedding learning (BiGEL). Specifically, we first obtain a set of behavior-aware embeddings by using a cascading graph paradigm. Subsequently, we introduce three key modules to improve the performance of the model. The cascading gated feedback (CGF) module enables a feedback-driven optimization process by integrating feedback from the target behavior to refine the auxiliary behaviors preferences. The global context enhancement (GCE) module integrates the global context to maintain the user's overall preferences, preventing the loss of key preferences due to individual behavior graph modeling. Finally, the contrastive preference alignment (CPA) module addresses the potential changes in user preferences during the cascading process by aligning the preferences of the target behaviors with the global preferences through contrastive learning. Extensive experiments on two real-world datasets demonstrate the effectiveness of our BiGEL compared with ten very competitive methods. 

---
# ReinPool: Reinforcement Learning Pooling Multi-Vector Embeddings for Retrieval System 

**Authors**: Sungguk Cha, DongWook Kim, Mintae Kim, Youngsub Han, Byoung-Ki Jeon, Sangyeob Lee  

**Link**: [PDF](https://arxiv.org/pdf/2601.07125)  

**Abstract**: Multi-vector embedding models have emerged as a powerful paradigm for document retrieval, preserving fine-grained visual and textual details through token-level representations. However, this expressiveness comes at a staggering cost: storing embeddings for every token inflates index sizes by over $1000\times$ compared to single-vector approaches, severely limiting scalability. We introduce \textbf{ReinPool}, a reinforcement learning framework that learns to dynamically filter and pool multi-vector embeddings into compact, retrieval-optimized representations. By training with an inverse retrieval objective and NDCG-based rewards, ReinPool identifies and retains only the most discriminative vectors without requiring manual importance annotations. On the Vidore V2 benchmark across three vision-language embedding models, ReinPool compresses multi-vector representations by $746$--$1249\times$ into single vectors while recovering 76--81\% of full multi-vector retrieval performance. Compared to static mean pooling baselines, ReinPool achieves 22--33\% absolute NDCG@3 improvement, demonstrating that learned selection significantly outperforms heuristic aggregation. 

---
# FinCARDS: Card-Based Analyst Reranking for Financial Document Question Answering 

**Authors**: Yixi Zhou, Fan Zhang, Yu Chen, Haipeng Zhang, Preslav Nakov, Zhuohan Xie  

**Link**: [PDF](https://arxiv.org/pdf/2601.06992)  

**Abstract**: Financial question answering (QA) over long corporate filings requires evidence to satisfy strict constraints on entities, financial metrics, fiscal periods, and numeric values. However, existing LLM-based rerankers primarily optimize semantic relevance, leading to unstable rankings and opaque decisions on long documents. We propose FinCards, a structured reranking framework that reframes financial evidence selection as constraint satisfaction under a finance-aware schema. FinCards represents filing chunks and questions using aligned schema fields (entities, metrics, periods, and numeric spans), enabling deterministic field-level matching. Evidence is selected via a multi-stage tournament reranking with stability-aware aggregation, producing auditable decision traces. Across two corporate filing QA benchmarks, FinCards substantially improves early-rank retrieval over both lexical and LLM-based reranking baselines, while reducing ranking variance, without requiring model fine-tuning or unpredictable inference budgets. Our code is available at this https URL. 

---
# Applying Embedding-Based Retrieval to Airbnb Search 

**Authors**: Mustafa Abdool, Soumyadip Banerjee, Moutupsi Paul, Do-kyum Kim, Xioawei Liu, Bin Xu, Tracy Yu, Hui Gao, Karen Ouyang, Huiji Gao, Liwei He, Stephanie Moyerman, Sanjeev Katariya  

**Link**: [PDF](https://arxiv.org/pdf/2601.06873)  

**Abstract**: The goal of Airbnb search is to match guests with the ideal accommodation that fits their travel needs. This is a challenging problem, as popular search locations can have around a hundred thousand available homes, and guests themselves have a wide variety of preferences. Furthermore, the launch of new product features, such as \textit{flexible date search,} significantly increased the number of eligible homes per search query. As such, there is a need for a sophisticated retrieval system which can provide high-quality candidates with low latency in a way that integrates with the overall ranking stack.
This paper details our journey to build an efficient and high-quality retrieval system for Airbnb search. We describe the key unique challenges we encountered when implementing an Embedding-Based Retrieval (EBR) system for a two sided marketplace like Airbnb -- such as the dynamic nature of the inventory, a lengthy user funnel with multiple stages, and a variety of product surfaces. We cover unique insights when modeling the retrieval problem, how to build robust evaluation systems, and design choices for online serving. The EBR system was launched to production and powers several use-cases such as regular search, flexible date and promotional emails for marketing campaigns. The system demonstrated statistically-significant improvements in key metrics, such as booking conversion, via A/B testing. 

---
# Unleashing the Native Recommendation Potential: LLM-Based Generative Recommendation via Structured Term Identifiers 

**Authors**: Zhiyang Zhang, Junda She, Kuo Cai, Bo Chen, Shiyao Wang, Xinchen Luo, Qiang Luo, Ruiming Tang, Han Li, Kun Gai, Guorui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2601.06798)  

**Abstract**: Leveraging the vast open-world knowledge and understanding capabilities of Large Language Models (LLMs) to develop general-purpose, semantically-aware recommender systems has emerged as a pivotal research direction in generative recommendation. However, existing methods face bottlenecks in constructing item identifiers. Text-based methods introduce LLMs' vast output space, leading to hallucination, while methods based on Semantic IDs (SIDs) encounter a semantic gap between SIDs and LLMs' native vocabulary, requiring costly vocabulary expansion and alignment training. To address this, this paper introduces Term IDs (TIDs), defined as a set of semantically rich and standardized textual keywords, to serve as robust item identifiers. We propose GRLM, a novel framework centered on TIDs, employs Context-aware Term Generation to convert item's metadata into standardized TIDs and utilizes Integrative Instruction Fine-tuning to collaboratively optimize term internalization and sequential recommendation. Additionally, Elastic Identifier Grounding is designed for robust item mapping. Extensive experiments on real-world datasets demonstrate that GRLM significantly outperforms baselines across multiple scenarios, pointing a promising direction for generalizable and high-performance generative recommendation systems. 

---
# Industrial Semantics-Aware Digital Twins: A Hybrid Graph Matching Approach for Asset Administration Shells 

**Authors**: Ariana Metović, Nicolai Maisch, Samed Ajdinović, Armin Lechler, Andreas Wortmann, Oliver Riedel  

**Link**: [PDF](https://arxiv.org/pdf/2601.06613)  

**Abstract**: Although the Asset Administration Shell (AAS) standard provides a structured and machine-readable representation of industrial assets, their semantic comparability remains a major challenge, particularly when different vocabularies and modeling practices are used. Engineering would benefit from retrieving existing AAS models that are similar to the target in order to reuse submodels, parameters, and metadata. In practice, however, heterogeneous vocabularies and divergent modeling conventions hinder automated, content-level comparison across AAS. This paper proposes a hybrid graph matching approach to enable semantics-aware comparison of Digital Twin representations. The method combines rule-based pre-filtering using SPARQL with embedding-based similarity calculation leveraging RDF2vec to capture both structural and semantic relationships between AAS models. This contribution provides a foundation for enhanced discovery, reuse, and automated configuration in Digital Twin networks. 

---
# L-RAG: Balancing Context and Retrieval with Entropy-Based Lazy Loading 

**Authors**: Sergii Voloshyn  

**Link**: [PDF](https://arxiv.org/pdf/2601.06551)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as the predominant paradigm for grounding Large Language Model outputs in factual knowledge, effectively mitigating hallucinations. However, conventional RAG systems operate under a "retrieve-always" assumption, querying vector databases for every input regardless of query complexity. This static approach incurs substantial computational overhead and inference latency, particularly problematic for high-throughput production deployments. We introduce L-RAG (Lazy Retrieval-Augmented Generation), an adaptive framework that implements hierarchical context management through entropy-based gating. L-RAG employs a two-tier architecture: queries are first processed with a compact document summary, and expensive chunk retrieval is triggered only when the model's predictive entropy exceeds a calibrated threshold, signaling genuine uncertainty. Through experiments on SQuAD 2.0 (N=500) using the Phi-2 model, we demonstrate that L-RAG provides a tunable accuracy-efficiency trade-off: at a conservative threshold (tau=0.5), L-RAG achieves 78.2% accuracy, matching Standard RAG (77.8%), with 8% retrieval reduction; at a balanced threshold (tau=1.0), retrieval reduction increases to 26% with modest accuracy trade-off (76.0%). Latency analysis shows that L-RAG saves 80-210ms per query when retrieval latency exceeds 500ms. Analysis of entropy distributions reveals statistically significant separation (p < 0.001) between correct predictions (H=1.72) and errors (H=2.20), validating entropy as a reliable uncertainty signal. L-RAG offers a practical, training-free approach toward more efficient RAG deployment, providing system architects with a configurable knob to balance accuracy and throughput requirements. 

---
# PixRec: Leveraging Visual Context for Next-Item Prediction in Sequential Recommendation 

**Authors**: Sayak Chakrabarty, Souradip Pal  

**Link**: [PDF](https://arxiv.org/pdf/2601.06458)  

**Abstract**: Large Language Models (LLMs) have recently shown strong potential for usage in sequential recommendation tasks through text-only models, which combine advanced prompt design, contrastive alignment, and fine-tuning on downstream domain-specific data. While effective, these approaches overlook the rich visual information present in many real-world recommendation scenarios, particularly in e-commerce. This paper proposes PixRec - a vision-language framework that incorporates both textual attributes and product images into the recommendation pipeline. Our architecture leverages a vision-language model backbone capable of jointly processing image-text sequences, maintaining a dual-tower structure and mixed training objective while aligning multi-modal feature projections for both item-item and user-item interactions. Using the Amazon Reviews dataset augmented with product images, our experiments demonstrate $3\times$ and 40% improvements in top-rank and top-10 rank accuracy over text-only recommenders respectively, indicating that visual features can help distinguish items with similar textual descriptions. Our work outlines future directions for scaling multi-modal recommenders training, enhancing visual-text feature fusion, and evaluating inference-time performance. This work takes a step toward building software systems utilizing visual information in sequential recommendation for real-world applications like e-commerce. 

---
# Towards Building efficient Routed systems for Retrieval 

**Authors**: Ramnath Kumar, Prateek Jain, Cho-Jui Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2601.06389)  

**Abstract**: Late-interaction retrieval models like ColBERT achieve superior accuracy by enabling token-level interactions, but their computational cost hinders scalability and integration with Approximate Nearest Neighbor Search (ANNS). We introduce FastLane, a novel retrieval framework that dynamically routes queries to their most informative representations, eliminating redundant token comparisons. FastLane employs a learnable routing mechanism optimized alongside the embedding model, leveraging self-attention and differentiable selection to maximize efficiency. Our approach reduces computational complexity by up to 30x while maintaining competitive retrieval performance. By bridging late-interaction models with ANNS, FastLane enables scalable, low-latency retrieval, making it feasible for large-scale applications such as search engines, recommendation systems, and question-answering platforms. This work opens pathways for multi-lingual, multi-modal, and long-context retrieval, pushing the frontier of efficient and adaptive information retrieval. 

---
# Beyond Single-Shot: Multi-step Tool Retrieval via Query Planning 

**Authors**: Wei Fang, James Glass  

**Link**: [PDF](https://arxiv.org/pdf/2601.07782)  

**Abstract**: LLM agents operating over massive, dynamic tool libraries rely on effective retrieval, yet standard single-shot dense retrievers struggle with complex requests. These failures primarily stem from the disconnect between abstract user goals and technical documentation, and the limited capacity of fixed-size embeddings to model combinatorial tool compositions. To address these challenges, we propose TOOLQP, a lightweight framework that models retrieval as iterative query planning. Instead of single-shot matching, TOOLQP decomposes instructions into sub-tasks and dynamically generates queries to interact with the retriever, effectively bridging the semantic gap by targeting the specific sub-tasks required for composition. We train TOOLQP using synthetic query trajectories followed by optimization via Reinforcement Learning with Verifiable Rewards (RLVR). Experiments demonstrate that TOOLQP achieves state-of-the-art performance, exhibiting superior zero-shot generalization, robustness across diverse retrievers, and significant improvements in downstream agentic execution. 

---
# Making Absence Visible: The Roles of Reference and Prompting in Recognizing Missing Information 

**Authors**: Hagit Ben Shoshan, Joel Lanir, Pavel Goldstein, Osnat Mokryn  

**Link**: [PDF](https://arxiv.org/pdf/2601.07234)  

**Abstract**: Interactive systems that explain data, or support decision making often emphasize what is present while overlooking what is expected but missing. This presence bias limits users' ability to form complete mental models of a dataset or situation. Detecting absence depends on expectations about what should be there, yet interfaces rarely help users form such expectations. We present an experimental study examining how reference framing and prompting influence people's ability to recognize expected but missing categories in datasets. Participants compared distributions across three domains (energy, wealth, and regime) under two reference conditions: Global, presenting a unified population baseline, and Partial, showing several concrete exemplars. Results indicate that absence detection was higher with Partial reference than with Global reference, suggesting that partial, samples-based framing can support expectation formation and absence detection. When participants were prompted to look for what was missing, absence detection rose sharply. We discuss implications for interactive user interfaces and expectation-based visualization design, while considering cognitive trade-offs of reference structures and guided attention. 

---
# DiSCo: Making Absence Visible in Intelligent Summarization Interfaces 

**Authors**: Eran Fainman, Hagit Ben Shoshan, Adir Solomon, Osnat Mokryn  

**Link**: [PDF](https://arxiv.org/pdf/2601.07229)  

**Abstract**: Intelligent interfaces increasingly use large language models to summarize user-generated content, yet these summaries emphasize what is mentioned while overlooking what is missing. This presence bias can mislead users who rely on summaries to make decisions. We present Domain Informed Summarization through Contrast (DiSCo), an expectation-based computational approach that makes absences visible by comparing each entity's content with domain topical expectations captured in reference distributions of aspects typically discussed in comparable accommodations. This comparison identifies aspects that are either unusually emphasized or missing relative to domain norms and integrates them into the generated text. In a user study across three accommodation domains, namely ski, beach, and city center, DiSCo summaries were rated as more detailed and useful for decision making than baseline large language model summaries, although slightly harder to read. The findings show that modeling expectations reduces presence bias and improves both transparency and decision support in intelligent summarization interfaces. 

---
# RAIRS: Optimizing Redundant Assignment and List Layout for IVF-Based ANN Search 

**Authors**: Zehai Yang, Shimin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.07183)  

**Abstract**: IVF is one of the most widely used ANNS (Approximate Nearest Neighbors Search) methods in vector databases. The idea of redundant assignment is to assign a data vector to more than one IVF lists for reducing the chance of missing true neighbors in IVF search. However, the naive strategy, which selects the second IVF list based on the distance between a data vector and the list centroids, performs poorly. Previous work focuses only on the inner product distance, while there is no optimized list selection study for the most popular Euclidean space. Moreover, the IVF search may access the same vector in more than one lists, resulting in redundant distance computation and decreasing query throughput. In this paper, we present RAIRS to address the above two challenges. For the challenge of the list selection, we propose an optimized AIR metric for the Euclidean space. AIR takes not only distances but also directions into consideration in order to support queries that are closer to the data vector but father away from the first chosen list's centroid. For the challenge of redundant distance computation, we propose SEIL, an optimized list layout that exploits shared cells to reduce repeated distance computations for IVF search. Our experimental results using representative real-world data sets show that RAIRS out-performs existing redundant assignment solutions and achieves up to 1.33x improvement over the best-performing IVF method, IVF-PQ Fast Scan with refinement. 

---
# AI-Assisted Authoring for Transparent, Data-Driven Documents 

**Authors**: Alfonso Piscitelli, Cristina David, Mattia De Rosa, Ali Mohammed, Federico Nanni, Jacob Pake, Roly Perera, Jessy Sodimu, Chenyiqiu Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2601.06027)  

**Abstract**: We introduce _transparent documents_, interactive web-based scholarly articles which allow readers to explore the relationship to the underlying data by hovering over fragments of text, and present an LLM-based tool for authoring transparent documents, building on recent developments in data provenance for general-purpose programming languages. As a target platform, our implementation uses Fluid, an open source programming language with a provenance-tracking runtime. Our agent-based tool supports a human author during the creation of transparent documents, identifying fragments of text which can be computed from data, such as numerical values selected from records or computed by aggregations like sum and mean, comparatives and superlatives like _better than_ and _largest_, trend-adjectives like _growing_, and similar quantitative or semi-quantitative phrases, and then attempts to synthesise a suitable Fluid query over the data which generates the target string. The resulting expression is inserted into the article's web page, turning the static text fragment into an interactable data-driven element able to reveal the data that underwrites the natural language claim. We evaluate our approach on a subset of SciGen, an open source dataset consisting of tables from scientific articles and their corresponding descriptions, which we extend with hand-generated counterfactual test cases to evaluate how well machine-generated expressions generalise. Our results show that gpt4o is often able to synthesise compound expressions extensionally compatible with our gold solutions. 

---
# Data-Driven Framework Development for Public Space Quality Assessment 

**Authors**: Sherzod Turaev, Mary John  

**Link**: [PDF](https://arxiv.org/pdf/2601.06026)  

**Abstract**: Public space quality assessment lacks systematic methodologies that integrate factors across diverse spatial typologies while maintaining context-specific relevance. Current approaches remain fragmented within disciplinary boundaries, limiting comprehensive evaluation and comparative analysis across different space types. This study develops a systematic, data-driven framework for assessing public space quality through the algorithmic integration of empirical research findings. Using a 7-phase methodology, we transform 1,207 quality factors extracted from 157 peer-reviewed studies into a validated hierarchical taxonomy spanning six public space typologies: urban spaces, open spaces, green spaces, parks and waterfronts, streets and squares, and public facilities. The methodology combines semantic analysis, cross-typology distribution analysis, and domain knowledge integration to address terminological variations and functional relationships across space types. The resulting framework organizes 1,029 unique quality factors across 14 main categories and 66 subcategories, identifying 278 universal factors applicable across all space types, 397 space-specific factors unique to particular typologies, and 124 cross-cutting factors serving multiple functions. Framework validation demonstrates systematic consistency in factor organization and theoretical alignment with established research on public spaces. This research provides a systematic methodology for transforming empirical public space research into practical assessment frameworks, supporting evidence-based policy development, design quality evaluation, and comparative analysis across diverse urban contexts. 

---
