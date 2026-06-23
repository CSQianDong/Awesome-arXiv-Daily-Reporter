# Improving Long-Context Retrieval with Multi-Prefix Embedding 

**Authors**: Zhenglin Yu, Xueguang Ma, Shengyao Zhuang, Zhichao Xu, Luyu Gao, Crystina Zhang, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2606.23642)  

**Abstract**: Long-context retrieval exposes a tension: single-vector embeddings lose fine-grained detail, while token-level multi-vector methods incur prohibitive storage. We propose Multi-Prefix Embedding (MPE), which partitions a document into chunks separated by EOS tokens, encodes the full sequence in a single causal forward pass, and extracts one embedding at each prefix boundary. MPE retains cross-chunk context, enables chunk-level MaxSim matching, and trains with only document-level relevance labels. Experiments on MLDR-en, BrowseComp-Plus, and LongEmbed show that MPE is competitive with or outperforms single-vector, independent-chunk, and multi-vector baselines, while providing a natural source attribution mechanism for locating evidence chunks. 

---
# Analysis of Autonomic Regulation in Cancer Survivors During Daily Physical Activity: A Real-World Wearable ECG Study 

**Authors**: Sajad Farrokhiørcidicon, Lerick Sequeira, Shanna L. Burke, Waltenegus Dargie, Christian Poellabauer  

**Link**: [PDF](https://arxiv.org/pdf/2606.23461)  

**Abstract**: This study investigates heart rate (HR) and heart rate variability (HRV) responses to physical activity in breast cancer survivors using wearable electrocardiogram (ECG) data collected in real-world settings. Reliable HRV analysis in such environments is challenging due to motion artifacts and activity-related signal degradation. To address this, we use an approach that combines accelerometer and gyroscope data for activity intensity segmentation (light, moderate, vigorous) with a robust ECG processing pipeline incorporating R-peak detection and annotation-free signal quality assessment. Because vigorous activity produced unreliable HRV estimates, analyses focused on light and moderate activity levels. Using 30~s, 1~min, and 2~min windows, HR and HRV metrics were computed and compared between breast cancer survivors and healthy controls. Cancer survivors consistently exhibited elevated HR and reduced HRV across activity levels. During light activity, HR increased from 95.7~bpm in controls to 103.4~bpm in cancer survivors. Differences became more pronounced during moderate activity, where RMSSD decreased from 39.7~ms to 22.1~ms and SDNN from 42.6~ms to 25.1~ms. Statistical analyses showed significant group differences with strong and consistent effects across observations. In addition, the proposed ECG quality assessment framework reliably identified high-quality signal segments, achieving near-perfect valid RR ratios (0.99) without manual annotations. Overall, these findings demonstrate impaired and activity-dependent autonomic regulation in cancer survivors and highlight the importance of motion-aware activity segmentation and robust ECG quality control for accurate physiological monitoring in real-world wearable settings. 

---
# URecJPQ: Memory-efficient Multimodal Recommendation Models through RecJPQ in Large-Scale Scenarios 

**Authors**: Giuseppe Spillo, Zixuan Yi, Aleksandr Petrov, Cataldo Musto, Craig Macdonald, Iadh Ounis  

**Link**: [PDF](https://arxiv.org/pdf/2606.23291)  

**Abstract**: Training state-of-the-art recommendation models on large-scale industrial datasets can be a challenging task due to the high number of users and items which are typically represented through ID embeddings. Such embeddings typically require a large amount of memory resources, which are not always available. This problem is further exacerbated in multimodal recommendation, in which multimodal item features generally improve recommendation performance, but require more resources to encode. In this paper, we introduce URecJPQ, a Joint Product Quantization method specifically designed for large-scale and multimodal top-k recommendation tasks, in which the vast number of users and items, combined with the available modalities, further increases the memory demands for the computation. The core idea is to represent each user/item not as a fully learned, unique embedding, but rather as a concatenation of shared learned sub-embeddings, thereby significantly reducing the total number of trainable parameters. Our experiments on three widely-used datasets across different domains (movies, baby and sports products) show that URecJPQ can be effectively applied to multimodal recommendation settings. In large scale scenarios, we observe a substantial reduction in checkpoint sizes and the number of trainable parameters (ranging from 86% to 98%, and 98% to 99%, respectively), with only a marginal decrease in accuracy (8.5% on recall and 16% on NDCG, on average), and, in some cases, even performance improvements (up to 85%), as in the baby products domain. Our codebase is available at this https URL. 

---
# The Language Blind Spot: How Query Language and Brand Recognition Tier Shape AI-Constructed Brand Reputation Across Twelve European Languages 

**Authors**: Dmitrij Żatuchin  

**Link**: [PDF](https://arxiv.org/pdf/2606.23165)  

**Abstract**: Large language models (LLMs) increasingly mediate how people form impressions of organisations, yet most monitoring is done in English, assuming an English query returns a representative picture. We measure how far that holds. We queried three grounded LLMs (GPT-5.4, Gemini 3.1 Pro, Perplexity Sonar Pro) about 66 brands from eleven Northern, Baltic, and Central European markets, in twelve languages across four families (Germanic, Uralic, Baltic, Slavic), generating 35,640 responses. Multilingual embeddings (BGE-M3) allow cross-language comparison without translation. Three results emerge. First, AI-constructed reputation is language-bound: mean cross-language cosine similarity is 0.825, same-family responses are more similar than cross-family (0.844 vs 0.820; d = 0.31), and sentiment varies by language (F = 268.5, eta^2 = 0.077), with Uralic and Baltic languages most positive and Germanic, including English, most critical; clustering recovers the Slavic and Baltic families (cophenetic 0.915). Second, query language shifts which brands are recommended far more than how they are described: moving from an English query to a brand's home language raises recommendation share by 0.80 for local champions but only 0.15 for global multinationals (t = -8.84, p < 0.001), with no comparable reversal in sentiment. An English-only audit therefore understates a local champion's AI visibility. Third, response stability varies more with model choice than with language (eta^2_model = 0.32 vs eta^2_language = 0.01, on a five-iteration replication over a 20-brand subset). These results indicate that English-only AI reputation monitoring leaves a measurable language blind spot, concentrated in the visibility of locally headquartered brands. 

---
# Who Owns the AI Recommendation? A Multi-Industry Empirical Map of Brand Category Ownership Across Large Language Models 

**Authors**: Dmitrij Żatuchin  

**Link**: [PDF](https://arxiv.org/pdf/2606.23057)  

**Abstract**: Large language models now mediate how buyers discover products and services, making the competitive structure of AI-generated recommendations a strategic concern for brands. A basic question has lacked large-scale empirical answers: in a given category, which brand does a model recommend, and how concentrated is that ownership? Across 3,750 responses spanning 50 brands, five industries, and 250 brand-free category queries on three models (GPT-5.2, Google Gemini 3 Flash, and Perplexity sonar-pro), each query repeated five times under a dice-roll stability protocol, we propose three exploratory metrics: the Category Ownership Index (COI), a brand's share of mentions within a category; the Competitive Vacuum Index (CVI), flagging categories with no single leader; and the Displacement Score (DS), quantifying asymmetric substitution between brand pairs. In this sample, recommendation concentration was moderate: the mean Gini coefficient was 0.28 (95% CI [0.16, 0.41]), below the 0.60 power-law threshold we set. Competitive vacuums were rare, appearing in 8.0% of queries, so the models named at least one sampled brand in most cases. Cross-model agreement on the top-recommended brand was 41.6%: a top position on one model did not reliably hold on another. Displacement was industry-dependent, from co-recommendation in consulting (0.4:1) to one-directional substitution up to 4.3:1, with an unweighted mean of 2.4:1 across the five industries. A BERTopic check placed only 4.2% of discovered topic clusters outside the original categories. Within the scope studied, these results sit in tension with a strong winner-takes-all narrative around AI recommendation, and the three metrics offer a candidate, reproducible procedure for competitive-intelligence analysis that future work can validate. 

---
# LLM-as-a-Judge for Reliable and Explainable Offline Evaluation in Top-K Recommendation 

**Authors**: Yue Que, Junyi Zhou, Xiaokun Zhang, Haiming Jin, Qiao Xiang, Chen Ma  

**Link**: [PDF](https://arxiv.org/pdf/2606.22961)  

**Abstract**: Recommendation evaluation plays a crucial role in guiding the refinement and deployment of recommender systems. Most existing trials rely on offline evaluation using Top-K metrics computed over holdout user behaviors. However, we identify two fundamental limitations that undermine their ability to deliver reliable and explainable evaluations. Regarding reliability, offline evaluation treats observed user feedback as a proxy of true preferences and enforces rigid ID matching between the proxy and recommendation. In practice, feedback collections are inherently shaped by incomplete and biased item exposure, leading to distorted and unreliable assessments. Regarding explainability, Top-K metrics only establish numerical scores without offering meaningful insights to support them, thereby reinforcing the black-box nature of offline evaluation.
In this paper, we propose a reliable and explainable LLM-as-a-Judge framework for offline recommendation evaluation. To enhance reliability, we introduce a semantic proxy from user textual behaviors to represent their true preferences. This proxy allows for more flexible matching between preferences and recommendations in the semantic space, rather than depending on the holdout feedback. To ensure explainability, the LLM Judge adopts a reasoning-then-scoring process to generate relevance judgments along with explicit rationale. Finally, we aggregate the individual scores into global Top-K metrics to quantify overall recommendation quality, and provide justification for each preference hit or miss. Extensive experiments demonstrate that the LLM Judge achieves solid reliability, explainability, and robustness in evaluation. 

---
# Trajectory-Based Recommender Systems as Control Systems 

**Authors**: Eriam Schaffter, Ahmed Bounekkar, Elsa Negre  

**Link**: [PDF](https://arxiv.org/pdf/2606.22957)  

**Abstract**: Recommender Systems (RS) are a key research domain and play an increasing role in our content-overwhelmed lives. In this paper, we explore Trajectory-Based Recommender Systems (TBRS), a subfield for which many related studies exist, yet still lacking a common framework. We argue that Control Theory provides an appropriate foundation for formalizing and solving TBRS problems. TBRS, sometimes named Long Term goal Recommender Systems, share core principles with classical RS, but at their core lies the concept of a trajectory, a defining element that makes these systems a singular category. To date, most RSs that include a notion of goal or long-term objective, when this goal is explicit, have not been recognized as having specific characteristics that make them worth regrouping under a dedicated field of research. We review related work, observe how they differ from already conceptualized RSs, and sketch the foundations of a possible theoretical framework based on control theory. Finally, we show how Educational Recommender Systems (ERS), intrinsically long-term and goal-driven, can be modeled within the proposed TBRS framework. 

---
# Towards Fast Domain Adaptation and Fine-Grained User Simulation for Evaluating Conversational Recommender Systems 

**Authors**: Yuanzi Li, Quanyu Dai, Xueyang Feng, Zihang Tian, Junhao Wang, Xu Chen, Zhenhua Dong, Huifeng Guo  

**Link**: [PDF](https://arxiv.org/pdf/2606.22803)  

**Abstract**: Conversational Recommender Systems (CRSs) enhance user experience through multi-turn interactions, yet evaluating their performance remains challenging. While Large Language Model (LLM) based user simulators are effective, they suffer from three key limitations: (1) Lack of Domain Adaptability: Reliance on fixed prompts and predefined action spaces hinders transfer to novel domains; (2) Limited User Modeling: Inability to accurately replicate subtle linguistic styles and dynamic preferences; (3) Insufficient Evaluation Validity: Existing simulators fail to adequately assess fundamental capabilities and system robustness. To overcome these, we propose AdaptSim, an Adaptive domain and automatic prompt tuning User Simulator. AdaptSim offers an efficient framework for evaluating CRSs by enabling realistic behavior modeling and diverse style generation. It leverages automatic prompt generation and an open action mechanism to reduce manual effort and improve cross-domain flexibility. For response generation, we employ controlled text generation with a "think-then-respond" strategy for fine-grained control over language style. For CRS evaluation, AdaptSim incorporates a novel Breadth-First Search (BFS)-based, turn-level pairwise comparison framework for comprehensive assessment. Extensive experiments across three domains and four LLMs demonstrate that AdaptSim generates realistic dialogues, enabling a highly effective and reliable evaluation of CRS capabilities and robustness. 

---
# Breaking the Evaluation Paradox: Evaluating High-Entropy Search with Computationally Irreducible Constraints 

**Authors**: Juntao Wu, Wei Wen, Xianting Huang, Shuai Pang, Ruizhi Qiao, Xing Sun, Ke Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.22783)  

**Abstract**: Evaluating the exhaustive search capabilities of large language models (LLMs) is plagued by a fundamental paradox: verifying completeness requires complete ground truth, yet high-entropy enumeration tasks make such ground truth impossible for humans to create. This causes benchmarks to systematically penalize models for outperforming their human annotators. Despite rapid progress in web-search and deep research agents -- which now issue hundreds of queries, traverse diverse sites, and synthesize long reports -- evaluation still largely relies on partially annotated answer sets, LLM-based judges, or single-answer questions that avoid genuinely exhaustive search scenarios. We break this paradox by shifting the evaluation paradigm from simulating a messy reality to constructing computationally pure challenges. We introduce VERITAS (Verifiable Traversal Assessment for Search), a framework built on the principle of computationally irreducible constraints. By introducing novel, non-optimizable constraints, we create verifiable, sparse-answer search tasks that are computationally equivalent to exhaustive enumeration. These constraints are easy to verify but impossible for LLMs or search engines to optimize, forcing agents to genuinely traverse the entire search space. VERITAS can automatically generate a virtually infinite number of test cases with perfect ground truth and precise difficulty control, with marginal instance cost dominated by hash computations. This provides not only a robust benchmark for evaluating systematic exploration under uncertainty but also a scalable method for generating training data to improve these crucial, yet underdeveloped, capabilities. 

---
# HAKARI-Bench: A Lightweight Benchmark for Comparing Retrieval Architectures and Efficiency Settings under Unified Conditions 

**Authors**: Yuichi Tateno  

**Link**: [PDF](https://arxiv.org/pdf/2606.22778)  

**Abstract**: With the rapid spread of retrieval-augmented generation and semantic search, choosing the right embedding and retrieval configuration is increasingly hard. Large retrieval benchmarks are comprehensive but too heavy to rerun during development, and there is little infrastructure for comparing production settings--dimensionality reduction, quantization, reranking--across many models under identical conditions. We present HAKARI-Bench, a lightweight benchmark that reconstructs existing retrieval suites into small datasets (Nano-sets): 35 benchmarks and 551 tasks across 43 languages in a unified format, enabling same-condition, model-agnostic comparison of five retrieval families (BM25, dense, sparse, late interaction, rerankers) and their efficiency variants. Across 55 models, its overall ranking reproduces the official MTEB retrieval v2, MMTEB v2 retrieval, and English BEIR (full) at Spearman >0.97. HAKARI-Bench does not replace full evaluation; it enables rapid model selection, regression detection, and reading the quality-efficiency Pareto frontier. Code, data, and leaderboard are released under the MIT license. 

---
# PA-User: Simulating Trust and Verification under AI-Generated Content 

**Authors**: Saber Zerhoudi  

**Link**: [PDF](https://arxiv.org/pdf/2606.22738)  

**Abstract**: Most users of online information now assume that some of what they read has been written, edited, or selected by an AI model. Hybrid cases are the hardest to tell apart: human prose rewritten by a language model, AI-curated lists presented as editorial, retrieval-augmented answers composed on the fly from human sources. Users cannot reliably distinguish these cases, and the ongoing cost of checking what is genuine has become part of how they search. Current user simulators in information retrieval do not model this. We propose PA-User, a user simulator with three new components: a detection-effort budget that is spent on verification and recovers between sessions; a trust component that holds a separate Beta belief over the factuality of each source class (domain by provenance) and updates from observed outcomes; and a decision rule that picks accept, verify, or discard for each result, conditional on current trust, current effort, and per-domain stakes. We state two verification-and-validation (V\&V) properties of the framework. The trust posterior converges to the true class factuality (face validity). Each component's contribution to any observable can be isolated by ablation (structural validity). On the HC3 corpus (85,449 paired human and ChatGPT answers in five domains), PA-User reaches a trust-calibration error of $0.162$, against $0.356$ for any configuration without the trust component. PA-User reduces high-stakes regret from $0.171$ to $0.122$ ($29\%$ relative) against an always-accept ablation, and verifies $34.5\%$ of results, half the rate of an ablation with no effort budget. Each single-mechanism ablation isolates one component, which makes the framework individually diagnosable. 

---
# All Relations Lead to Rome: Automated Knowledge Graph Creation and Question Generation 

**Authors**: Matthijs Jansen op de Haar, Tobias Stähle, Lorenzo Gatti  

**Link**: [PDF](https://arxiv.org/pdf/2606.22645)  

**Abstract**: Large language models have substantially improved information retrieval and question answering; however, existing datasets generally support either vector-based retrieval over unstructured text or reasoning over knowledge graphs, without providing a unified representation that combines both paradigms. Moreover, current benchmarks rarely provide ground-truth entities, relations, and fact-grounded question-answer pairs aligned with the underlying corpus. To address this gap, we introduce All Relations Lead to Rome (ARLtR), a unified framework for automated knowledge graph construction and fact-grounded question-answer generation. ARLtR jointly constructs a knowledge graph, embeddings, and question-answer pairs that are explicitly grounded in extracted entities, relations, and supporting textual evidence. We further instantiate the framework as a historical dataset centered on the Roman Empire, comprising over 19,000 entities, 16,000 chunks, and 8,400 question-answer pairs (this https URL). By tightly coupling symbolic graph representations with dense retrieval representations, ARLtR facilitates the evaluation and development of hybrid retrieval systems and semantic steering approaches within a single coherent resource. 

---
# Music Playlist Captioning at Scale with Large Language Models 

**Authors**: Mathieu Delcluze, Léa Briand, Benjamin Chapus, Deniz Mekik, Guillaume Salha-Galvan  

**Link**: [PDF](https://arxiv.org/pdf/2606.22460)  

**Abstract**: Music streaming services such as Deezer often recommend personalized playlists to users. Playlist captioning, which involves describing these playlists in natural language, is essential for helping users understand the content behind each recommendation, yet remains challenging at scale. This paper presents the automatic playlist captioning system deployed on Deezer in 2025 to address this challenge. Leveraging recent advances in large language models (LLMs) to generate descriptive captions from diverse data sources in a controlled manner, this system now powers the Daily Mix feature, used by millions of users. This deployment has led to significant improvements in user engagement, highlighting how the semantic framing of an unchanged recommendation shapes user perception in online personalized experiences. 

---
# Novelty-Aware Agentic Retrieval: Comparing Research Contributions Through Structured Multi-Step Reasoning 

**Authors**: Shou-Tzu Han  

**Link**: [PDF](https://arxiv.org/pdf/2606.22151)  

**Abstract**: Scientific literature search is an information retrieval (IR) task in which ranked lists are insufficient: a researcher entering a new area needs to know not only which papers are relevant, but how they relate, where they overlap, how they differ, and what problem-method combinations are absent. Standard retrieval-augmented generation (RAG) summarizes documents independently, discarding this comparative signal. We present the Novelty-Aware Research Agent, a prototype agentic retrieval system that layers structured multi-step reasoning on a RAG pipeline through six typed-contract components: query analysis, a ReAct-style retrieval loop, relevance ranking, schema-guided contribution extraction, a three-pass comparison agent, and answer generation. Beyond returning relevant papers, it produces structured comparison artifacts: per-paper contribution records, paper-level overlaps, and a problem x method gap matrix. On a 100-paper corpus, the system supports five structured comparison capabilities that a standard RAG baseline supports none of, while remaining query-sensitive: across three main queries no paper appears in all three top-5 sets (mean pairwise Jaccard 0.12), and an extended seven-query evaluation holds the pattern across ten queries (mean Jaccard 0.115, 18 of 29 retrieved papers query-exclusive). Under author-assigned graded relevance the ranker attains mean Precision@5 1.000 and nDCG@5 0.752 on the main queries, ahead of BM25, dense, and hybrid retrieval; over ten queries Precision@5 is non-saturated at 0.980 with nDCG@5 0.739. Schema compliance is 86.7% on the main queries and 84.0% over the ten-query set, and validating 20 sampled empty gap-matrix cells yields a gap precision of 0.600. We discuss the latency-structure trade-off in agentic retrieval and identify corpus scale, author-assigned labels, and limited independent evaluation as the main limitations. 

---
# A feasibility study on filtering low-accessibility web pages considering color vision deficiency 

**Authors**: Ryota Mizutani, Shiori Nakayama, Masateru Tsunoda  

**Link**: [PDF](https://arxiv.org/pdf/2606.22095)  

**Abstract**: Recently, the importance of universal design has increased. Color universal design (CUD) is one type of universal design that takes people with color vision deficiency (CVD) into consideration. Websites are important media for providing various types of information and functions. Therefore, it is essential to enhance the accessibility of web pages by incorporating CUD principles. The goal of our study is to help improve the accessibility of web pages. Our approach is to automatically filter low-accessibility web pages. To evaluate the feasibility of this approach, we conducted an experiment using 21 web pages. The prediction model identified low-accessibility pages with reasonable accuracy, achieving a maximum AUC of 0.76. 

---
# The Pitfall of Scaling Up: Uncovering and Mitigating Popularity Bias Amplification in Scaling Transformer-based Recommenders 

**Authors**: Weiqin Yang, Yue Pan, Chongming Gao, Sheng Zhou, Xiang Wang, Can Wang, Jiawei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.21911)  

**Abstract**: We identify a critical pitfall in scaling transformer-based sequential recommenders: while increasing model size improves recommendation accuracy, it simultaneously amplifies popularity bias. This bias drives systems to over-recommend popular items at the expense of niche ones, which not only undermines fairness but also degrades the broader ecosystem by reinforcing the Matthew effect and filter bubbles. Consequently, this bias amplification emerges as a fundamental obstacle to sustainable model scaling.
Through comprehensive theoretical and empirical analyses, we uncover the root cause of this amplification. Our findings reveal that as model depth increases, the two core components of the transformer architecture, i.e., attention aggregation and feed-forward projections, synergistically induce severe spectral collapse in model predictions, which directly translates to the amplification of popularity bias. To address this challenge, we propose SPRINT (Scalable Popularity Regularization IN Transformers), which mitigates spectral collapse during scaling by constraining (i) the maximum column-sums of the attention score matrices and (ii) the spectral norms of the feed-forward parameters. Extensive experiments demonstrate that SPRINT significantly improves both accuracy and long-tail fairness. Crucially, it yields more favorable scaling behaviors when expanding model sizes from 0.05M to 0.34B parameters. The code is available at this https URL. 

---
# CRAwLeR -- Cross-Reference Aware Legal Retrieval 

**Authors**: Maciej Jalocha, William Michelsen  

**Link**: [PDF](https://arxiv.org/pdf/2606.21676)  

**Abstract**: Existing benchmarks for context-aware chunk retrieval rely heavily on repurposed task items and rarely demonstrate that their queries genuinely require context, making score interpretation difficult. We focus on a specific kind of context dependence, legal cross-references, and introduce CRAwLeR, an operationalization of a narrow, well-defined phenomenon: cross-reference-aware context utilization for chunk retrieval in legal documents. Our pipeline detects legal cross-references, identifies query candidates, links target chunks to their relevant context, generates context-demanding queries with an LLM, and filters them through both an adversarial non-contextual baseline and an assurance prompt. We release CRAwLeR-DK and CRAwLeR-PL, Danish and Polish datasets built with this pipeline, alongside a strong Anthropic-style contextualization baseline. Manual analysis finds that approximately 80% of randomly sampled queries genuinely target the labelled target chunk and require context, with failures following systematic and named patterns. The benchmarks are hard but not solved: best Recall@10 reaches 55% on CRAwLeR-DK and 59% on CRAwLeR-PL. Ablation and failure analysis attribute the remaining gap to the contextualising LLM, not the retriever. Even when the target is retrieved in the top ten, labelled context chunks routinely outrank it. We are the first dataset for context-aware chunk retrieval to carefully consider construct validity and inspect our results in the light of such a narrow, well-defined phenomenon. 

---
# From Embedding Geometry to Spectral Search: Energy Dispersion Networks For Vector Retrieval 

**Authors**: Lorenzo Moriondo, Ilias Azizi  

**Link**: [PDF](https://arxiv.org/pdf/2606.21535)  

**Abstract**: Vector spaces, such as embedding spaces that encode dense semantic information, need not be analyzed solely through pointwise geometry. They can also be interpreted as energy networks through the spectral graph induced by the topology of their column vectors, i.e., their feature-space structure. Building on this perspective, we introduce Graph Wiring, a general framework for exploiting feature-space spectral structure, together with Spectral Indexing, its task-specific instantiation for vector search. By coupling geometric similarity with spectral information, the proposed method improves head-tail coherence and semantic alignment relative to purely geometric retrieval methods. It further supports adaptive search behavior through tau-modulation, providing the flexibility increasingly required by modern Retrieval-Augmented Generation (RAG) pipelines. We present the complete algorithmic pipeline, establish its theoretical foundation through epiplexity, and evaluate the approach across benchmark and industrial settings using the open-source arrowspace library. 

---
# A Rank-One Popularity Component in Dot-Product Recommender Scores:Population Theory and Prior-Separation Evidence 

**Authors**: Yang Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2606.21275)  

**Abstract**: Representation anisotropy in recommender systems is often attributed to Transformer architectures. We identify a more general source in the conditional training distribution. For any encoder using a dot-product softmax decoder, the population-optimal score decomposes into pointwise mutual information, an item-marginal term log p(i), and a context-dependent offset. After centering, the item marginal produces a context-shared rank-one score component, while time-varying marginals induce a low-rank popularity subspace. This score-level result does not imply universal embedding collapse because its transfer to embeddings depends on factorization geometry. Experiments on synthetic data and public Alibaba and Tianchi interaction logs support the proposed mechanism. Separating log p(i) from the learned dot product reduces the measured popularity-aligned score energy by 98.6 percent in a matched intervention. Permutation tests confirm that this reduction is specific to the empirical popularity direction. These results explain a class of apparent representation degeneration as a decoder-level consequence of long-tailed item marginals rather than a property unique to Transformer encoders. 

---
# The Token Tax of Epistemic Accuracy: Comparing RAG and Long-Context Architectures for Document-Grounded Generative AI Applications 

**Authors**: Austin Hamilton, Ryan Singh, Michael Wise, Ibrahim Yousif, Arthur Carvalho, Zhe Shan, Mohammad Mayyas, Lora A. Cavuoto, Fadel M. Megahed  

**Link**: [PDF](https://arxiv.org/pdf/2606.20898)  

**Abstract**: Document-grounded assistants built on large language models are increasingly used in high-stakes, knowledge-intensive work. Their usefulness, however, may depend on how evidence is allocated before generation. We investigate such a claim by comparing two grounding architectures: (a) retrieval-augmented generation (RAG) that retrieves a few relevant passages, and (b) long-context prompting, which loads the whole document collection in context. We view these as two regimes of "epistemic access" on an accuracy--cost frontier. We use "epistemic accuracy" to capture model correctness that depends on having the right evidence. We posit that broader access (via long context) can increase it, but with a "token tax" (i.e., a substantial increase in cost due to larger input token consumption). We probe this framing with a case study in manufacturing safety training. Using an expert-validated benchmark, we evaluate 972 answers across three machines, two small language models, and three retrieval/in-context prompting approaches. Long-context prompting achieved the highest correctness (73.1% vs. 65.4% for semantic RAG), but at 26 times the per-query token cost. We interpret this gap as the token tax of broader evidentiary access. We carefully discuss the implications of our findings for resource-constrained organizations. 

---
# Multi-Vector Embeddings are Provably More Expressive than Single Vector Embeddings 

**Authors**: Rajesh Jayaram  

**Link**: [PDF](https://arxiv.org/pdf/2606.23475)  

**Abstract**: Multi-vector (MV) embeddings have become a powerful paradigm in neural information retrieval (IR), achieving high retrieval accuracy by representing data with multiple vectors and scoring them via the non-linear Chamfer similarity. Despite their widely perceived superiority over single-vector (SV) embeddings which use inner product similarity, to date there is no formal proof that SV similarities cannot approximate MV similarities with the same representation size. Specifically, we ask the following: for any bounded dataset size $n \leq 2^{poly(m)}$, what is the smallest dimension $D$ so that given any collection of MV embeddings $Q_1,\dots,Q_n,X_1,\dots,X_n \subset \mathbb{R}^d$ containing at most $m$ vectors each, there always exist $q_1,\dots,q_n$, $d_1,\dots,d_n \in \mathbb{R}^{D}$ satisfying $|\langle q_i, d_j \rangle - \texttt{Chamfer}(Q_i,X_j)| \leq \epsilon$ for all $i,j$? Recently, the MUVERA algorithm demonstrated that $D = m^{O(1/\epsilon^2)}$ is possible. If improved to $D = md$, this would imply that MV embeddings are no more expressive than SV embeddings.
In this paper, we rule out this scenario. Specifically, we prove the existence of a collection of MV embeddings in $\mathbb{R}^d$, each containing at most $m$ vectors, which require single-vector dimension of $D =(\epsilon^2 m)^{\Omega(1/\epsilon)}$ to approximate, establishing a strong separation in representation size between MV and SV embeddings. Our proof leverages the Pattern Matrix Method by constructing a hard instance whose Chamfer similarity matrix encodes the $NAND_k$ boolean function. Our results confirm a long-held belief in the IR community: at a fixed representation size, multi-vector embeddings can express similarities which cannot even be approximately represented by single vector embeddings. 

---
# Ranking Companion: A Visual Analytics Approach to Item-Based Ranking with Hybrid Item Selection 

**Authors**: Aman Kumar, Maximilian Tornow, Michaela Benk, Ibrahim Al-Hazwani, Jürgen Bernard  

**Link**: [PDF](https://arxiv.org/pdf/2606.23263)  

**Abstract**: Personalizing item ranking creation is a challenging task, especially when users lack knowledge of data attributes or the ability to express and formalize their attribute preferences. Item-based ranking creation is an approach allowing users to directly externalize preferences through known-item judgments rather than attribute-based scoring. However, a core challenge of item-based ranking is identifying and selecting representative candidate items for externalizing preferences. Existing approaches rely on singular item-selection methods, limiting flexibility and user control. To address this challenge, we present Ranking Companion, a visual analytics approach for item-based ranking that combines model-driven active learning with human-driven item-selection methods. By drawing from six complementary item-selection methods, users can externalize listwise preferences based on selected candidate items, while an iterative machine learning process with a ranking model calculates ranking results, presented to users alongside explanations for interpretation. We evaluated Ranking Companion in a formative user study with 10 participants, in which participants used each item-selection method across three iterations, revealing tradeoffs in perceived ranking quality across accuracy, diversity, novelty, transparency, control, and satisfaction. Ranking Companion contributes a unified interactive item selection space and provides preliminary empirical guidance toward the hybrid use of multiple complementary item-selection methods in personalized item-based ranking creation. 

---
# The Correct Answer Trap: Pedagogically-Grounded Detection and Feedback for Hidden Misconceptions 

**Authors**: Moiz Imran, Sahan Bulathwela  

**Link**: [PDF](https://arxiv.org/pdf/2606.23205)  

**Abstract**: Automated feedback systems that rely on answer correctness will reinforce, rather than address, misconceptions when students reach the correct answer through flawed reasoning. We investigate automatic detection of these hidden misconceptions using 20,964 real student responses from the Eedi mathematics platform. Fine-tuned classifiers detect only 57% of these hidden misconceptions, and standard ML interventions do not improve on this. An open-weight reasoning model detects 84%, but at realistic prevalence, false alarms outnumber genuine detections roughly 8 to 1. We present a graduated assessment rubric that separates answer correctness from method validity, and propose a detect-verify-escalate pipeline that routes uncertain cases to diagnostic follow-up questions rather than directly to teachers. Two deployment modes adapt the pipeline: a teacher dashboard where the system filters a review queue, and an autonomous tutor where flags trigger low-cost formative follow-up. 

---
# Graph-Enhanced Large Language Models for Spatial Search 

**Authors**: Nicole R. Schneider, Kent O'Sullivan, Hanan Samet  

**Link**: [PDF](https://arxiv.org/pdf/2606.22909)  

**Abstract**: There have been many recent improvements in the ability of Large Language Models (LLMs) to perform complex tasks and answer domain-specific questions through techniques like Retrieval Augmented Generation (RAG). However, reasoning abilities of LLMs, including spatial reasoning abilities, are still lacking. Spatial reasoning is a key component required to answer questions in a variety of domains that are grounded in the physical world, including urban planning, civil engineering, travel, and many others. To advance the development of LLMs and facilitate an impact in these domains, new research techniques must be developed to enable LLMs to reason over spatial data, which is commonly stored in the form of a graph. In this paper we outline the challenges associated with spatial reasoning through LLMs and envision a future in which search engines integrate with LLMs to answer complex spatial questions through graph-enhanced reasoning. 

---
# VISTA Architect: A graph database-oriented health AI system demonstrated in multidisciplinary tumor boards 

**Authors**: Tuomo Kiiskinen, Jason Fries, Philip Adamson, David Wu, Timothy John Ellis-Caleo, Aaron Fanous, Balasubramanian Narasimhan, Joel Neal, Sylvia Plevritis, Manuel A. Rivas  

**Link**: [PDF](https://arxiv.org/pdf/2606.22692)  

**Abstract**: We introduce VISTA Architect, a database-oriented AI architecture for integrating large language models (LLMs) with longitudinal electronic health records (EHRs). At ingestion, it transforms complex clinical documentation into a persistent, provenance-linked knowledge graph, eliminating repeated reprocessing of raw records at query time. The architecture has two layers: a source-faithful MEDS Graph preserving granular EHR structure with full provenance, and a clinically abstracted Timeline Object Architecture (TOA) that uses graph-guided LLM extraction to synthesize a concise timeline of deduplicated, temporally coherent clinical events. This addresses key limitations of direct long-context prompting and retrieval-augmented generation (RAG), which often miss temporal relationships and incur high cost and latency from repeated raw-text processing. By precomputing clinical synthesis once, downstream queries access an organized patient state and traverse to source documentation only when detailed verification is needed.
We demonstrate the system in multidisciplinary thoracic oncology tumor boards at Stanford Medicine, where precise reconstruction of patient histories is critical. Across 1,180 patients, VISTA Architect achieved 96.4% accuracy (mean 9.75/10) on 15 tumor board-salient variables (17,700 evaluations; 95% CI 96.1-96.7%), surpassing a matched BM25 RAG baseline and recent benchmarks for LLM-based clinical extraction. An agentic interface reduced preparation for a 30-patient held-out cohort to about 2.2 minutes without sacrificing accuracy. While configured here for thoracic oncology, the modular design adapts to other specialties through customizable event definitions, episode structures, and agentic tools; validation beyond thoracic oncology remains future work. 

---
# ARIA: A Causal-Aware Framework for Rescuing LLM Reasoning in Trustworthy Materials Discovery 

**Authors**: Yi Cao, Liaoyaqi Wang, Jieneng Chen, Benjamin Van Durme, Alan Yuille, Paulette Clancy  

**Link**: [PDF](https://arxiv.org/pdf/2606.22375)  

**Abstract**: Generative models have revolutionized the process of materials discovery, yet they often fail to satisfy underlying physical causality. Through an analysis of Large Language Models (LLMs) augmented with knowledge graphs derived from current literature, we uncover a phenomenon termed contextual tunneling, where models "over-anchor" on narrow, retrieved evidence while suppressing global physical reasoning. To address this problem, we introduce ARIA, a causal-aware framework that conditions knowledge use on mechanistic completeness. ARIA routes each query through a three-tier cascade: (i) direct causal reasoning when complete evidence chains of Process-Structure-Property (PSP) are available, (ii) physics-informed analogical transfer for sparse or novel material systems, and (iii) explicit parametric fallback when external evidence is incomplete. As a proof of concept, we construct a Knowledge Graph (KG) containing 2,839 extracted PSP relations from peer-reviewed articles in the materials literature and evaluate ARIA on forward prediction and inverse design tasks for two-dimensional (2D) materials. ARIA mitigates contextual tunneling, improves over unaugmented and naive KG-augmented baselines, and provides further gains when an online literature search is used for evidence enrichment. Crucially, ARIA produces auditable causal traces, enabling physically grounded and trustworthy AI-assisted materials discovery. 

---
# SHACR: A Graph-Augmented Semi-Autonomous Framework for Multi-Class Conflict Resolution in Smart Home IoT Automation 

**Authors**: Leena Marghalani, Walid Aljoby, Suayb S. Arslan  

**Link**: [PDF](https://arxiv.org/pdf/2606.22312)  

**Abstract**: Smart home automation increasingly relies on user-defined rules across heterogeneous IoT devices. While these rules appear harmless in isolation, their concurrent execution creates hidden, cross-rule interactions via shared devices, environmental variables, and physical topology. These interactions result in unsafe, wasteful, or privacy-threatening behaviors that are completely invisible to text-only analysis. Existing conflict detectors remain siloed, catching either static syntactic conflicts or specific environment-mediated interactions without unifying the two or providing actionable repairs for non-expert users.
This paper presents SHACR, a smart home conflict resolution framework that anchors Large Language Model (LLM) unpredictability by grounding its reasoning in a formal, directed knowledge graph. SHACR encodes devices, capabilities, physical states, and Trigger-Condition-Action rules as typed, traversable entities. By elevating physical cause-effect relationships to first-class graph edges, SHACR transforms conflict detection from fragile text inference into deterministic multi-hop graph traversal, unifying logical, semantic, and physical conflict classes. It drives a closed-loop Scan-Explain-Repair-Validate workflow that uses the graph to bound the LLM's action space. We evaluated SHACR on a testbed of 203 rules deployed across 70 apartments within a smart building. By holding the underlying LLM fixed and introducing SHACR's knowledge graph, classification errors drop by 36.7\%, F1 rises from 0.59 to 0.79, and few-shot calibration further lifts F1 to 0.95, whereas the same calibration barely helps a graph-free LLM. Ultimately, this work challenges the current AI paradigm, establishing that structured knowledge representation is a far more critical factor for dependable IoT automation management than prompt engineering or underlying model architecture. 

---
# Nous: A Predictive World Model for Long-Term Agent Memory 

**Authors**: Pranav Singh  

**Link**: [PDF](https://arxiv.org/pdf/2606.22030)  

**Abstract**: We present Nous, a novel agent memory architecture grounded in the principle that knowledge is prediction, not storage. Rather than persisting facts as database records, vector embeddings, or knowledge-graph triples, Nous maintains a predictive world model: a collection of categorical probability distributions, called dimensions, one per entity-attribute pair observed in conversation. Each incoming observation is scored by its information-theoretic surprise S = -log2 P(obs | D), and the distribution is updated via a closed-form Bayesian posterior. The primary stored artifact is the delta, a record of the shift from prior to posterior belief, rather than the fact itself. Forgetting emerges naturally as entropy decay toward the uniform distribution, and identity resolution is handled through mutual information between entity dimension sets. Evaluated on the LoCoMo long-term conversational memory benchmark across ten conversations (1,540 questions) using GPT-4o-mini as backbone, Nous achieves F1 of 63.50 (single-hop), 55.32 (multi-hop), 58.57 (temporal), and 62.50 (open-domain). Against A-MEM's self-reported GPT-4o-mini numbers, Nous shows substantial gains in three of four categories, though we note that independent citations of A-MEM's results disagree with each other on category assignment, a reproducibility issue we discuss openly rather than resolve unilaterally. We additionally compare against BeliefMem, a concurrently developed system built on the same core premise of belief-based rather than deterministic memory; on the same benchmark and backbone, Nous's self-reported numbers exceed BeliefMem's self-reported numbers on all four categories, though we flag several uncontrolled differences between the two evaluation pipelines that prevent this from being a fully controlled comparison. Nous requires no external vector database or graph engine. 

---
# Gender Differences in Research Topic and Method Convergence among Collaborating Scholars in Library and Information Science 

**Authors**: Chengzhi Zhang, Linlei Xie, Siqi Wei  

**Link**: [PDF](https://arxiv.org/pdf/2606.21908)  

**Abstract**: This study explores gender differences in research topic choice and methodology among collaborating scholars. Previous studies have often focused on gender differences in research topics or methods at the individual level of scholars, without considering collaborating groups, lacking depth and practical guidance. This study takes Library and Information Science (LIS) as an example, employing the Top2Vec method for topic identification and the CogFT model for research method classification. It systematically analyzes 25,204 papers published between 1990 and 2022 to investigate gender differences in the convergence of research topics and method choices among collaborating scholars in this field. The results of the study found that female scholars showed lower convergence in their research methods and topic choices compared to male scholars. This study uses a relatively systematic methodology to address the difficulty of studying gender differences in academic publishing, and is expected to serve as a reference for other disciplines and research questions. This study also emphasizes the manifestation of gender differences in collaborative research and provides insights into the convergence and diversity of research topics and methods chosen by scholars. 

---
# Which Review Aspect Has a Greater Impact on the Duration of Open Peer Review in Multiple Rounds? -- Evidence from Nature Communications 

**Authors**: Haomin Zhou, Ruxue Han, Jiangtao Zhong, Chengzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.21904)  

**Abstract**: Purpose: Peer review is essential to scientific publishing, but increasing submission volumes have placed growing pressure on reviewers and editors. This study examines the relationship between sentiment toward specific review aspects and peer review duration. It also investigates how this relationship varies across disciplines and review rounds, with the aim of supporting targeted manuscript revision and improving review efficiency.
Design/methodology/approach: We adopt a two-stage approach. First, fine-grained aspects are extracted from peer review reports, and a sentiment classification model is used to determine the sentiment associated with each aspect. Second, correlations between aspect-level sentiment and peer review duration are analyzed. Sentiment scores are also calculated for different review rounds to determine whether these relationships change over successive rounds.
Findings: Review sentiment has a weak but statistically significant negative correlation with peer review duration, indicating that more positive reviews tend to be associated with shorter review periods. Aspects concerning Evaluation and Results and Impact and Research Value show relatively stronger correlations with review duration. The relationships between aspect-level sentiment and review duration also differ significantly across review rounds.
Originality/value: This study connects the textual content of peer review reports with the temporal characteristics of the review process. By identifying review aspects that are more closely associated with review duration, it provides evidence that may help authors prioritize revisions and assist reviewers and editors in improving review efficiency. The findings contribute to reducing the burden of peer review and accelerating scholarly communication and knowledge dissemination. 

---
# Research Method Usage across Academic Ages in Library and Information Science: An Empirical Study (1990-2023) 

**Authors**: Chengzhi Zhang, Jiayi Hao, Yi Mao  

**Link**: [PDF](https://arxiv.org/pdf/2606.21862)  

**Abstract**: Academic age critically shapes career development, influencing research behavior, output volume, and methodological choices. Analyzing method variation across academic ages offers a new theoretical lens on scholarly evolution and provides early-career researchers with practical guidance for method selection. A corpus of 26,677 articles published 1990-2023 in 14 authoritative Library and Information Science journals was compiled. The CogFT model automatically classified the research methods embedded in these articles, and Top2Vec generated the topic model. This process resulted in a comprehensive dataset linking research methods with topics. Author-name disambiguation enabled calculation of each scholar's academic age. Popularity and Shannon diversity indices for methods, together with topic diversity, were compared across academic age groups. Results reveal dynamic methodological trends: the share of theoretical approaches declined gradually, whereas experimental and bibliometric methods gained ground. Method popularity differs significantly among cohorts. Mid-career scholars exhibit the highest method diversity; late-career scholars the lowest. 

---
# PrivacyAlign: Contextual Privacy Alignment for LLM Agents 

**Authors**: Manveer Singh Tamber, Abhay Puri, Marc-Etienne Brunet, Perouz Taslakian, Jimmy Lin, Spandana Gella  

**Link**: [PDF](https://arxiv.org/pdf/2606.21710)  

**Abstract**: AI agents acting on behalf of users are constantly making decisions, and for users to trust their agents, those decisions must align with what they actually want. Privacy is an important alignment problem for agents: every message, post, or tool call an agent makes is a contextual judgment about what is appropriate to share, with whom, and under which conditions. Because such judgments depend on social expectations and norms, human judgment does not merely label privacy violations but also helps define them. While existing work relies on unreliable proxies for both training and evaluation, we place human judgment at the center of agentic privacy alignment. We introduce PrivacyAlign, a dataset of 1,350 samples with 3,516 detailed annotations from 599 unique annotators across diverse scenarios where current LLMs actually leak, and use it to ground both alignment training and automated evaluation in human privacy norms. Building on these annotations, we first show that conditioning LLM judges on human annotations and explanations for reference responses to the same prompt makes their judgments more reliable. We then introduce annotation-conditioned reward modeling, which uses these annotations to score new responses during RL, and show that small open-weight agents trained with this reward better align with human privacy norms, with strong gains on PrivacyAlign and existing privacy benchmarks for agents. 

---
# ATLAS: Agentic Taxonomy of Large-Scale Software Ecosystems 

**Authors**: Junyi Lu, Mengyao Lyu, Jiahui Wu, Lei Yu, Chengwei Liu, Fengjun Zhang, Li Yang, Chun Zuo, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.21597)  

**Abstract**: The open-source ecosystem on GitHub lacks a systematic hierarchical taxonomy of software repositories. GitHub Topics, the dominant organizational mechanism, is flat, inconsistent, and covers only 67% of projects. We present ATLAS, the first framework that automatically constructs a hierarchical taxonomy for software repositories and classifies projects into it end-to-end. By combining LLM global knowledge with real repository distributions, ATLAS proposes meaningful splitting dimensions and iteratively corrects those that fail to accommodate real projects. A Designer Agent proposes splitting dimensions while a Classifier Agent assigns repositories; a self-corrective refinement loop uses classification failures to drive dimension revision through escalating strategies. We evaluate ATLAS on 54,387 GitHub repositories against six baselines spanning four paradigms, two downstream tasks, and three model families. On a stratified 2,001-repository benchmark, ATLAS achieves a Taxonomy Quality F-score (TQF) of 83.13%, outperforming the best baseline by 15 percentage points (on the full 54k corpus the approximate TQF is 73.0%, a gap driven by Path Granularity's all-or-nothing scoring on longer paths rather than lower classification accuracy). It is the only method to simultaneously achieve high structural quality and high practical applicability. On downstream tasks, ATLAS enables alternative discovery with P@1 = 85.71%, surpassing even human-curated lists (62.34%), and achieves the highest P@1 for repository retrieval. The taxonomy further reveals structural ecosystem trends that are difficult to obtain from flat tags or similarity methods: the shift from libraries to AI/ML applications (now 61% of newly community-adopted projects) becomes visible only through hierarchical, type-based categorization. An interactive taxonomy explorer is available at this https URL 

---
# Per-Entity Bias Mapping for AI Visibility: Why Brand Mentions Require Entity-Specific Calibration 

**Authors**: Zoltan Varga  

**Link**: [PDF](https://arxiv.org/pdf/2606.21595)  

**Abstract**: AI-mediated answer systems increasingly determine how brands and organizations are represented to users. Existing approaches reduce visibility to mention rate or citation frequency. This paper argues that aggregate metrics are insufficient because entities exhibit systematically different AI visibility error profiles.
We introduce Per-Entity Bias Mapping (PEBM): a ten-dimensional framework distinguishing raw from verified mentions. Three failure modes are identified: (1) underrepresented entities suffer invisibility due to weak knowledge graph presence; (2) large entities suffer the Brand Hallucination Paradox -- model familiarity creates stronger surfaces for plausible but incorrect completions; (3) CEE entities face a structural infrastructure gap across knowledge graphs, NER, and entity linking. A fourth dimension, Parametric-Retrieval Lag Asymmetry, describes divergence between retrieval-augmented and parametric memory update cycles.
A full-scale empirical study (n=100 Hungarian B2B entities, 1,400 probe runs, 2,062 sources) finds Tier 1 brands produce 52.69% fabricated citations versus 37.87% for Tier 3 entities (+14.82 pp; p=1.67e-11), supporting the Brand Hallucination Paradox. Regulatory-framed queries elevate fabrication to 56.77% versus 37.59% baseline (+19.2 pp). We identify rejection-induced confabulation escalation: agentic quality filters function as hallucination accelerators in compliance contexts. We introduce ghost cartography as a unifying mechanism: entities in sparse latent regions produce confident output interpolated from neighboring dense regions, yielding a two-dimensional confabulation space (fabricated presence vs. frozen representation). 

---
# Dissecting Agentic RAG: A Component Ablation for Multi-Hop QA with a Local 7B Model 

**Authors**: Sheroz Shaikh  

**Link**: [PDF](https://arxiv.org/pdf/2606.21553)  

**Abstract**: Agentic retrieval-augmented generation (RAG) systems combine iterative reasoning loops, query decomposition, and adaptive retrieval to tackle multi-hop question answering. However, the contribution of each component remains poorly understood, particularly under resource-constrained settings using only local language models. Many agentic designs add adaptive retrieval routing and deeper retrieval loops on the assumption that the added complexity helps. To test whether it does, we run a controlled ablation study of a full agentic RAG pipeline evaluated on 5,000 questions from the HotpotQA distractor development set using a local 7B parameter model (Qwen2.5-7B-Instruct). Our full pipeline achieves EM=53.2% and F1=61.6%, compared to a single-pass dense-retrieval baseline of EM=43.1% and F1=54.0%. Across eight ablation conditions, we find that: (1) fixed hybrid retrieval via reciprocal rank fusion consistently outperforms rule-based adaptive routing (+1.8 EM, +1.9 F1), as the routing heuristic over-routes to BM25 by firing on named entities present in nearly all multi-hop sub-questions; (2) two retrieval iterations over the decomposed sub-questions capture 95% of the gains of five, with no meaningful benefit from deeper loops; and (3) query decomposition and cross-encoder reranking each contribute statistically significant but smaller gains (p<0.01 and p<0.001 respectively). Taken together, on a fixed local-model budget, the simpler and fixed choices turn out to be competitive with or better than their adaptive versions: most of the gain comes from running a short retrieval loop, not from adaptive routing or from many iterations. We use no proprietary APIs or large-scale compute. 

---
# Memory Is No Longer a Bottleneck: Memory-Efficient Graph Filtering for Scalable Collaborative Filtering 

**Authors**: Jin-Duk Park, Won-Yong Shin  

**Link**: [PDF](https://arxiv.org/pdf/2606.21540)  

**Abstract**: Graph convolutional networks (GCNs) have demonstrated significant success in capturing complex user-item relationships for collaborative filtering (CF). However, due to their reliance on extensive model training, training-free graph filtering (GF)-based CF methods have emerged as a promising alternative, offering computational efficiency by smoothing graph signals via matrix operations. In particular, polynomial GF-based approaches demonstrate improved accuracy through their ability to design more expressive and flexible filtering functions. Despite these advantages, existing GF methods suffer from a critical memory bottleneck: they necessitate storing the full item similarity graph, incurring prohibitive memory costs for large-scale datasets, which limits their practical applicability. To tackle this challenge, we propose Mem-GF (Memory-efficient GF), a new GF-based CF method that departs from conventional designs by principally leveraging the structure of Krylov subspaces as a core mechanism for approximating polynomial graph filters without explicitly storing the item similarity graph. We theoretically analyze the minimum Krylov subspace size that guarantees lossless approximation. Through extensive experiments, we demonstrate that Mem-GF achieves up to 5.74$\times$ lower memory usage and 4.38$\times$ speedup in runtime, while consistently exceeding the recommendation accuracy of state-of-the-art GF and GCN-based methods. Mem-GF robustly scales to datasets with tens of millions of interactions, establishing itself as a practically viable and theoretically grounded solution for efficient CF. 

---
# Dual-Attention Convolution Experts for Sparse Tensor Completion 

**Authors**: Yanlei Liu, Zhenyu Liao  

**Link**: [PDF](https://arxiv.org/pdf/2606.21427)  

**Abstract**: Tensor factorization (TF) has been widely adopted for high-dimensional sparse data completion tasks. Despite significant progress, neural TF methods often struggle to capture complex cross-mode interactions and remain vulnerable to (extreme) data sparsity. To address these challenges, we propose a novel neural tensor factorization approach, termed Dual-Attention Convolution Expert Networks with Group-Level Contrastive Learning (DCGC). For the first problem, DCGC generates diverse non-linear alignment patterns of latent factors via a multi-channel convolution network, and leverages the gated dual-attention mechanism to drive the model to focus on more important output channels (i.e., convolution experts) and the aligned features. Furthermore, DCGC introduces a group-level contrastive learning strategy that aggregates positive samples with identical feedback levels while separating negative samples across different levels. This strategy injects high-quality self-supervised signals to mitigate data sparsity. Extensive experiments conducted on five datasets demonstrate that our DCGC outperforms the state-of-the-art methods in sparse tensor completion for traffic and recommendation applications. Code to reproduce the experimental results in the paper is available at this https URL. 

---
# PulseCX: Breaking the Closed-World Assumption in Real-Time CX 

**Authors**: Rajat Agarwal, Suvidha Tripathi, Shubham Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2606.21124)  

**Abstract**: Conversational AI agents in Customer Experience (CX) typically suffer from a Closed-World Constraint, ignoring high-velocity external shifts like viral trends or outages. Ad-hoc web search attempts to bridge this gap but often introduce prohibitive latency and context poisoning. We introduce PulseCX, a framework that decouples knowledge acquisition from consumption. Adopting a structure-first paradigm, PulseCX employs an asynchronous agent to linearize signals into a Decay-Aware Temporal Knowledge Graph (DA-TKG) governed by reinforcement--decay dynamics to actively manage information lifecycles. By coupling this self-evolving memory with hierarchical intent gating, PulseCX removes synchronous search bottlenecks (<10ms overhead) and drives significant gains in Intent Resolution (IRR) and Customer Satisfaction (s-CSAT) in dynamic environments. 

---
# Topic-to-Timestamp Alignment by Constrained Evidence Selection 

**Authors**: Zeynep Yılbırt, Marina Litvak, Michael Färber  

**Link**: [PDF](https://arxiv.org/pdf/2606.20890)  

**Abstract**: Meeting archives are difficult to search when users remember what was discussed but not when. We study topic-to-timestamp alignment: given a natural-language topic and a timestamped meeting transcript, the goal is to return the time at which the topic is discussed. A standard RAG setup can retrieve relevant transcript excerpts, but still asks the language model to generate a timestamp, which can produce unsupported or invalid timecodes. We therefore recast timestamp prediction as constrained temporal candidate selection: the system retrieves timestamped transcript chunks, and the model selects the candidate that best grounds the topic instead of generating a timecode. On 420 topic-timestamp queries from 200 municipal meeting transcripts, this increases Recall@5 from 31.9% to 50.0%, reduces MAE from 837.0 seconds to 761.0 seconds with Mistral-7B-Instruct, and increases the number of parseable outputs from 373 to 419 of 420 queries. The results suggest that temporal grounding in long transcripts depends strongly on retrieval quality and output design, not only on the choice of the language model. 

---
