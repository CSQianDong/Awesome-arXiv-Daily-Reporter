# PULSAR: Pooled Unified Late-Interaction Search and Retrieval for Enterprise Visual Document RAG 

**Authors**: Benjamin Constable, Anup Roy, Vishal Sharma, Rishabh Upadhyay, Robin Mills, Aidan Millar  

**Link**: [PDF](https://arxiv.org/pdf/2608.28572)  

**Abstract**: Institutional investors search visually dense pitch decks, board packs, and diligence materials that change hourly near deal closing. OCR followed by figure verbalisation is costly to refresh at this scale and can lose chart detail. We present PULSAR, a production vision-first retrieval system deployed at Mubadala Investment Company. PULSAR indexes page images with a frozen ColPali-style backbone and uses a pooled two-stage late-interaction index: compact page summaries support initial retrieval, followed by exact MaxSim rescoring over a finer pooled representation. On ViDoRe V3, this design reduces median vector-search latency by 15.1 times against an unpooled configuration with less than 0.01 absolute NDCG@10 and Recall@10 loss; production median vector-search latency is 156 ms. Under concurrent load, the pooled index sustains approximately 88 times higher QPS than an unpooled index. The event-driven ingestion path is estimated to be approximately 20 times cheaper per page than the OCR+verbalisation baseline it replaced. Since March 2026, PULSAR has served 78 thousand documents and approximately 2.4 million pages across more than 3,000 deals. At the production top K, it more than doubles answer-fact recall over the OCR+verbalisation baseline. 

---
# QUEST: A Query and Extraction System for Topics in Asylum Law Application Decisions 

**Authors**: Maria Vlachou, Anna Murphy Høgenhaug, Mohammad N. S. Jahromi, Galadrielle Humblot-Renaux, Thomas Gammeltoft-Hansen, Thomas B. Moeslund, Desmond Elliott  

**Link**: [PDF](https://arxiv.org/pdf/2608.28555)  

**Abstract**: Legal decisions on asylum applications consist of long, complex, and heterogeneous documents, covering narrative applicant interviews, original decisions, and additional supporting materials. If an application is rejected, a critical question in processing an appeal is whether the credibility of the information in the original application was a factor that determined the original decision. In this paper, we present the QUEST system (Query and Extraction System for Topics) to extract and identify factors relating to credibility assessments in two datasets of Danish asylum application appeals. QUEST frames this problem as an information retrieval task, combining synthetic query generation, topic extraction, and relevance assessment to identify information related to credibility indicators in appeals board application materials. In addition to standard retrieval evaluation metrics, we propose a new type of domain-specific assessments distinct from the traditional relevance to evaluate the performance of the tested systems with respect to credibility factors. In this way, we obtain insights about how well automatic methods can return answers for different types of indicators appearing in asylum appeals. Our results indicate that there is an increased challenge when estimating performance using credibility-based relevance assessments, thus pointing to the difficulty of the task. 

---
# SG-UMP: Sequence-Guided Universal Multimodal Prioritization Calculation Framework 

**Authors**: Xinyi Zhang, Yutong Li, Peijie Sun  

**Link**: [PDF](https://arxiv.org/pdf/2608.28503)  

**Abstract**: Multimodal sequential recommendation (MSR) improves recommendation by incorporating heterogeneous information such as text, images, and user interactions. However, existing MSR methods often fail to capture user-level preference heterogeneity and dataset-level modality bias, limiting their adaptability across users and datasets. To address this issue, we propose \textbf{S}equence-\textbf{G}uided \textbf{U}niversal \textbf{M}ultimodal \textbf{P}rioritization Calculation Framework (\textbf{SG-UMP}), a plug-and-play plugin for enhancing multimodal information processing in MSR. SG-UMP includes a Module Combiner for flexible multimodal processing and a Module Router for dynamic module ordering, enabling adaptation to both user preferences and dataset characteristics. Experiments on four real-world datasets show that SG-UMP consistently improves recommendation performance across different backbones and multimodal settings. The code is available at this https URL . 

---
# Every Article Deserves a Video: Contextual Video Matching for Digital Publishers 

**Authors**: Arnaud Corone, Brice Pierre de la Briere, Gladys Roch, Samuel Leonardo Gracio, Yassine Bouher, Parvati Chauchaix  

**Link**: [PDF](https://arxiv.org/pdf/2608.28359)  

**Abstract**: As digital publishers face the challenge of managing massive content catalogs, the ability to effectively embed relevant video within text-based articles has become critical for both monetization and user retention. However, manual selection is impractical for large scale publishers, especially when navigating their own extensive video libraries or the entire global Dailymotion catalog. In this paper, we present the "Contextual Video Matching" system, a solution that automatically matches relevant videos with text-heavy web pages and articles. By leveraging Large Language Models (LLMs) and textual embeddings, we provide a scalable solution for publishers to efficiently combine video content with their articles. We discuss in detail the motivations, architecture, evaluations, and deployment of this system within Dailymotion's production environment. Since its launch, the system has been adopted by hundreds of publishers, significantly increasing user engagement and enriching user experiences with highly relevant video content. 

---
# HubMixer: Progressive Latent Hub Mixing for Parameter-Efficient Feature Interaction in Recommendation 

**Authors**: Jie Zhou, Zixian Gong, Wenhao Li, Chang Liu, Enzhao Shen, Bo Liu, Xu Guo, Fei Pan, Peng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2608.27991)  

**Abstract**: Learning effective feature interactions is central to industrial recommendation and advertising ranking systems. Recent token-mixing architectures simplify self-attention with lightweight mixing operators, improving hardware efficiency and enabling large-scale deployment. However, recommendation tokens are fundamentally heterogeneous: user profiles, item attributes, behavioral sequences, context features, statistical signals, and business-side features live in different semantic spaces and interact in sparse, sample-specific patterns. Directly mixing all tokens in the raw heterogeneous token space may therefore be parameter-inefficient, as the model must implicitly discover which feature groups should interact and how such interactions should be routed. In the paper, we propose HubMixer, a parameter-efficient latent hub mixing architecture for feature interaction in recommendation. Instead of directly mixing raw feature tokens, HubMixer introduces a small set of learnable latent hubs to organize feature interactions through an `induction--interaction--readout` paradigm. First, hub induction summarizes heterogeneous tokens into compact latent hubs, where latent hubs query input tokens through cross-attention. Second, hub interaction performs high-order interaction in the cleaner latent hub space. Third, token-conditioned readout lets each original token selectively read from the interacted hubs, injecting global interaction semantics while preserving token-level field identity. Extensive offline experiments on industrial recommendation tasks show that HubMixer outperforms the SOTA models. Online A/B testing in the Kuaishou short-video recruitment business further shows a statistically significant 5.48% improvement in resume submission conversion rate, and HubMixer has been fully deployed in production. 

---
# Information-Guided Selective Modality-Interest Alignment for Multimodal Recommendation 

**Authors**: Wenze Ma, Chenyu Sun, Yanmin Zhu, Qiwen Gu, Xuhao Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2608.27950)  

**Abstract**: Multimodal recommendation (MMRec) aims to enhance recommendation performance by leveraging rich item content from multiple modalities. However, directly incorporating all modality information does not necessarily lead to better preference modeling, since user interests are often more related to a subset of modality signals, while other signals may be weakly aligned with user preferences or even introduce noise. Although recent MMRec methods improve modality utilization through invariant learning, attention mechanisms, graph refinement, or contrastive learning, their alignment processes are often implicit or heuristic and lack a clear objective for selecting modality signals that better match user interests.
In this paper, we propose AMUR, an information-guided selective modality-interest alignment framework for multimodal recommendation. Inspired by an information-theoretic view, AMUR aims to enhance modality information that is more related to user interests while reducing the influence of less aligned signals. Specifically, AMUR first refines modality graph structures towards user behavior, and then selectively aligns shared interest-related semantics across modalities. This enables AMUR to improve modality-interest alignment while preserving useful modality-specific complementary information. Extensive experiments on three real-world datasets demonstrate the effectiveness of AMUR over competitive baselines. The code is available at this https URL. 

---
# ITER: Interaction-Aware Retrieval for Agentic Search 

**Authors**: Haodong Chen, Shuai Wang, Yu Yin, Shengyao Zhuang, Guido Zuccon, Teerapong Leelanupab  

**Link**: [PDF](https://arxiv.org/pdf/2608.27912)  

**Abstract**: Deep-research agents answer complex user questions through an iterative sequence of search steps, where the agent autonomously formulates sub-queries to retrieve the evidence needed at each stage. However, existing retriever training typically relies only on the sub-query and its corresponding search results at the current step as training signals, leaving the information accumulated from previous interactions largely underutilized.
We introduce iter, an agent interaction-aware dense retriever trained using agent trajectory learning signals. iter represents each query by incorporating not only the current sub-query, but also the main question and preceding sub-queries, and is trained using trajectory-relative learning signals derived from the agent's interactions.
Across six agent backbones from three model families, iter consistently outperforms the existing agent-trajectory-trained dense retriever, LRAT, achieving an average improvement of 7.5% on InfoSeek-Eval and 13.5% on BrowseComp-Plus. iter also demonstrates stronger cross-agent robustness than AgentIR, a deep-research retriever that relies on external LLM-judge signals and the agent's pre-search reasoning. Ablations further show that the main question and previous sub-queries provide the most robust query representation, while previously visited and useful documents, used as redundancy negatives in subsequent searches, provide the strongest trajectory-relative supervision. Code is available at this https URL. 

---
# Personalized and Multi-View Representation for Federated Cold-Start Recommendation 

**Authors**: Jaehyung Lim, Wonbin Kweon, Woojoo Kim, Junyoung Kim, Dongha Kim, Hwanjo Yu  

**Link**: [PDF](https://arxiv.org/pdf/2608.27826)  

**Abstract**: Federated recommendation (FedRec) enables personalized modeling without centralizing users' interaction histories, but most existing methods assume a fixed item pool and thus overlook the practical cold-item setting where new items continuously arrive. Under the dual-sided constraint, where the server cannot access clients' interactions while clients cannot access the server's proprietary item attribute features, prior federated cold-start recommendation approaches suffer from three structural limitations: a lack of personalization, compositionality failure caused by forcing heterogeneous semantics into a single embedding space, and training- and communication-inefficiency arising from explicit alignment between separate collaborative and attribute representations. To address these challenges, we propose Personalized and Multi-view Representation for Federated Cold-Start Recommendation (PMFRec). PMFRec learns a personalized representation generator to produce user-specific item representations from attribute features, and introduces a global multi-view encoder with item-adaptive gating and an orthogonality objective to capture complementary semantic views while reducing cross-view redundancy. In addition, PMFRec fuses collaborative and attribute knowledge into a single exchanged item representation, eliminating the need for an explicit client-side regularizer and reducing communication overhead. Extensive experiments on real-world datasets show that PMFRec consistently outperforms strong baselines in cold-item recommendation and further improves user-level fairness, warm-scenario adaptability, and robustness under Local Differential Privacy (LDP). 

---
# LINE Conversation History Retrieval for Personal Memory RAG: Evaluating Search Representations and Hybrid Retrieval 

**Authors**: Akito Hattori  

**Link**: [PDF](https://arxiv.org/pdf/2608.27809)  

**Abstract**: As an initial step toward personal memory retrieval-augmented generation (RAG) for large language models (LLMs), this study presents a retrieval-only case study over one user's LINE conversation history. We segmented 358,896 messages into 22,329 temporally coherent chunks and constructed three search representations: raw_text, a generated summary, and embedding_text, which combines a summary with a raw-text excerpt and other fixed text. We compared BM25, dense vector retrieval, and linear hybrid retrieval on 100 evaluation questions verified by a single annotator. Among individual retrievers, embedding_text_bm25 achieved the highest point estimate, with Recall@5 of 0.584. We then explored six retriever pairings and 21 weights, for 126 configurations on the same evaluation set. The selected combination of embedding_text_bm25 and embedding_text_vector at beta = 0.45 achieved Recall@5 = 0.697, MRR@5 = 0.595, and nDCG@5 = 0.575. Its Recall@5 exceeded that of embedding_text_bm25 by 0.113, with a question-level paired percentile-bootstrap 95% confidence interval of [0.048, 0.184]. This interval is conditional on fixing the configuration selected on the same 100 questions and does not account for uncertainty from configuration selection or weight search. The difference from a summary-based hybrid at beta = 0.50 was 0.050, with a 95% confidence interval of [-0.013, 0.115], so no clear difference could be established. The 17 aggregate questions also yielded lower point estimates than the other question types, suggesting that flat chunk-level retrieval struggles when evidence is distributed across multiple times and conversations. This evaluation is an exploratory single-user, single-annotator study conducted on the same question set used for configuration search; it does not evaluate final answer generation or generalization to unseen questions. 

---
# NormasTCU --- A Brazilian Portuguese IR Dataset and an Evaluation of LLM-as-a-Judge for Relevance Assessment 

**Authors**: Leandro Carísio Fernandes, Marcus Vinícius Borela de Castro, Leandro dos Santos Ribeiro, Leonardo Augusto da Silva Pacheco, Edans Flávius de Oliveira Sandes  

**Link**: [PDF](https://arxiv.org/pdf/2608.27746)  

**Abstract**: Portuguese Information Retrieval (IR) lacks public datasets, and relevance assessment for specialized collections remains costly. While Large Language Models (LLMs) increasingly support relevance assessment, their reliability in non-English specialized domains remains unclear. We introduce NormasTCU (this https URL), a Brazilian Portuguese IR dataset with 14,469 legal documents, 46 queries, and 3,048 human judgments over 812 query-document pairs. Using NormasTCU, we evaluated LLM-as-a-judge for relevance assessment by prompting three models with two prompt techniques to grade these pairs. We then compared the rankings of 15 IR systems derived from LLM-generated and human reference qrels. LLMs consistently showed a positive scoring bias (mean absolute error: 0.46--0.66 on a 0-2 scale). Furthermore, pair-level agreement with human judgments achieved only fair to moderate levels, with Cohen's kappa ranging from 0.32 to 0.53. Despite this bias, LLM-generated judgments often yielded highly similar system rankings for nDCG@10 and MRR (observed Kendall's tau greater than or equal 0.90, although the bootstrap confidence intervals did not always remain above this threshold), but were less reliable for P@10 and R@10. Notably, LLM-based rankings were sometimes more strongly correlated with the reference ranking than individual human annotations were. As a practical implication, our results suggest that LLMs could effectively support scalable relevance assessment in specialized Portuguese corpora when evaluated using nDCG or MRR (rank-aware metrics), but they should be avoided when relying on precision or recall. 

---
# A Versioned Unified Graph Index for Dynamic Timestamp-Aware Nearest Neighbor Search 

**Authors**: Jun Woo Chung, Weijie Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2608.27663)  

**Abstract**: We present TiGER (Time-Integrated Graph for Efficient Retrieval), a novel approach for performing fast time-aware approximate nearest neighbor searches on dynamic vector datasets with flexibility over any possible time range. Our proposed algorithm builds and maintains a unified graph for all vectors by leveraging an index structure based on integrated versioned connectivity, allowing arbitrary time intervals to be queried directly on the unified graph without having to traverse invalid vectors. This forgoes the need for post-search filtering or merging, or separate graphs for each possible composite range. Empirical evaluations show that our method attains up to a 5x improvement in queries per second (QPS) without compromising accuracy over baselines based on filtering or per-time-segment sub-graphs. We believe that this method will enable efficient temporal analysis across evolving datasets in real-time recommendation systems, log analysis, and any scenario requiring fast similarity search over dynamic, time-segmented data. 

---
# Beyond the Vacuum: Combinatorial Strategy Selection for Competitor-Aware Generative Engine Optimization 

**Authors**: Vaibhav Sourirajan, Yao Zhang, Himanshu Kumar, Sahil Wadhwa, Mann Patel, Amirfarrokh Iranitalab  

**Link**: [PDF](https://arxiv.org/pdf/2608.27631)  

**Abstract**: Generative Engine Optimization (GEO) has emerged as a novel paradigm for transforming content to increase visibility in Large Language Model (LLM) responses. Traditional GEO methods, however, select rewriting strategies in isolation, ignoring a critical externality: as adoption of content optimization grows, optimal strategies for rewriting content change. We formalize GEO as a competitor-aware strategy selection problem and propose a two-phase pipeline to solve it: (1) We use Bayesian Optimization of Combinatorial Structures (BOCS) to efficiently search the space of rewriting strategies, (2) We generate preference pairs and grounded reasoning traces from the BOCS black-box observations to fine-tune a language model to analyze a document corpus and propose optimal rewriting strategy combinations. We achieve state-of-the-art performance across several impression metrics over existing agentic and single-heuristic methods on both geo-bench and our synthetically augmented competitive dataset geo-bench_comp. Our method also transfers to multiple out-of-distribution datasets, proving effective across domains, queries, and document types. 

---
# LitCurate: A Configuration-Driven AI-Assisted Framework for Scientific Database Construction with an Application to Lower-Mantle Equation-of-State Data 

**Authors**: Abin Shakya, Wilson Samuels, Dominica Wilson, Gioia A. Marchi, Israa Draz, Chenxing Luo, Renata M. Wentzcovitch  

**Link**: [PDF](https://arxiv.org/pdf/2608.27629)  

**Abstract**: The growing scientific literature contains decades of experimental and computational results that could support data-driven and physics-based modeling, yet much of this infor- mation remains locked in publications and is not readily usable for large-scale analysis or sci- entific software. Building structured databases from the literature is particularly challenging whenrelevantstudiesmustfirstbediscoveredamonglargecollectionsofpapersandreported quantities must be extracted with enough scientific context to remain usable. We present LitCurate, an open-source framework for building scientific databases from the literature using large language models within an auditable, stage-wise curation workflow. LitCurate integratesliteraturediscovery, relevancescreening, full-textprocessing, andstructuredinfor- mation extraction while retaining intermediate results and provenance, allowing researchers to inspect and revise individual stages rather than treating automated curation as a black- box process. We apply LitCurate to construct an equation-of-state database of lower-mantle and lower-mantle-relevant high-pressure mineral phases from experimental and theoretical studies, comprising 1,334 entries from 205 papers. The resulting dataset links reported equation-of-state parameters to mineral phases, compositions, equation formulations, meth- ods, and parameter constraints, and labels values as source-reported or citation-reported when provenance can be determined. The records are available through a searchable web application. By connecting scientific literature to traceable, machine-readable data, LitCu- rate provides a reusable approach for transforming accumulated literature into resources for scientific analysis and computational modeling. 

---
# Nested Byte-Level Vocabularies Are Cheap to Deploy and Expensive to Share: A Pre-Registered Negative Result 

**Authors**: Christos Koutsiaris  

**Link**: [PDF](https://arxiv.org/pdf/2608.28151)  

**Abstract**: A byte-level BPE tokenizer is an ordered list of merge rules, so applying only a prefix yields a vocabulary whose token identifiers are the first rows of the full vocabulary. This prefix nesting allows one language model to operate at several vocabulary sizes, use a control token to indicate the active size, and be deployed at any trained size by slicing its embedding and output head. We pre-registered five claims, including margins, seeds, contrasts, and a stop rule, and trained 30 models with 3.1M- and 10.6M-parameter bodies on 200M tokens each. Slicing is numerically exact: across 76 checks, a sliced model reproduces the restricted full model's logits bit for bit and removes 66% of deployed weights without changing latency. However, the shared model trails a fixed-cap specialist by 3.64% bits per byte at 32k against a 1% margin, and by 2.96% at 8k against a 2% margin. A 2x2 ablation separating the control token from output restriction finds that the token changes performance by +0.07% to +0.13%, with all intervals crossing zero, while output restriction costs +0.47% to +1.19%; the factors are substitutes rather than complements. Multi-cap training nevertheless improves robustness: under typographical noise, the same checkpoint degrades 12.5--15.4 points less in its fine mode and outperforms each fixed-cap specialist at that specialist's vocabulary size. A control with neither cap token nor output restriction is equally robust, attributing this benefit to multi-granularity training rather than conditioning. The per-cap penalty tracks each cap's share of training rows, yielding a falsifiable prediction for future work. 

---
# An Empirical Evaluation of Cross-City POI Recommendation on a Large-Scale Benchmark 

**Authors**: Peibo Li, Yang Song, Hao Xue, Maarten de Rijke, Flora D. Salim  

**Link**: [PDF](https://arxiv.org/pdf/2608.27840)  

**Abstract**: Cross-city point-of-interest (POI) recommendation is crucial for navigating unfamiliar urban environments, yet its progress has historically been constrained by data limitations. Using the recently proposed large-scale benchmark Trip World, we empirically re-examine whether conclusions drawn on small prior benchmarks still hold under worldwide coverage, low home-destination region overlap, and large, semantically rich POI inventories. Our evaluation surfaces three bottlenecks of representative state-of-the-art methods: (1) hometown-aware models appear to rely more on destination-region priors than on user-specific preference transfer; (2) their accuracy-efficiency trade-off degrades at this scale, where the simplest model is among the strongest; and (3) existing mechanisms for integrating semantic metadata yield little benefit. We further include a diagnostic pilot on agentic methods adapted from next-POI recommendation, finding that naive adaptation trails a simple popularity prior even though the relevant semantic signal is present in the data. These results highlight the need for task-specific designs that support cross-city preference transfer, semantic grounding, and scalable reasoning over unseen destination inventories. 

---
# CareGraph: An Auditable Hybrid AI Framework for Evidence-Grounded Personalized Longitudinal Health Intelligence 

**Authors**: Pratik Ghawate, Tanvi Patil  

**Link**: [PDF](https://arxiv.org/pdf/2608.27484)  

**Abstract**: Artificial intelligence is transforming personalized healthcare, yet fragmented clinical, self reported, and wearable evidence remains difficult to interpret and trace. We present CareGraph, an auditable hybrid AI framework that converts heterogeneous records into prioritized trends, missing context indicators, bounded next steps, discussion questions, and provenance linked explanations. CareGraph organizes evidence without diagnosing, predicting outcomes, selecting treatment, or making autonomous clinical decisions. Its pipeline covers deterministic analysis, context detection, graph construction, constrained language model synthesis, evidence validation, safety controls, and release gating. Tests used synthetic cohorts of 400 patients each for development, validation, and holdout. On holdout data, a frozen ordinary least squares trend rule with a sufficiency gate achieved 0.827 accuracy, 0.837 macro F1 with a 95 percent confidence interval of 0.819 to 0.854, and 0.974 insufficient data F1. Missing context detection achieved 0.815 strict micro F1 versus 0.318 for the legacy detector. On an authored holdout benchmark, safety ruleset version 1.2 achieved 1.000 precision, 0.950 recall, and 0.974 F1. An audit requiring graph retrieval across 80 patients yielded 79 syntheses and 78 presentations without fallback; one output was blocked and one failed closed because of an invalid evidence key. Against monolithic GPT 5.6 on 56 matched patients, CareGraph was faster at 40.15 versus 49.62 seconds, shorter at 661 versus 1,163 words, and showed better exploratory lexical alignment with longitudinal targets; the baseline used fewer tokens and cited more raw evidence. Graph auditing verified provenance and deterministic retrieval; incremental graph effects on generation require paired evaluation. CareGraph offers a safety bounded foundation for intelligent personalized health systems. 

---
