# Auditing Return Conditioning as a Control Knob: An Offline Diagnostic for Decision Transformer Recommendation 

**Authors**: Jingyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2608.24815)  

**Abstract**: Offline return-to-go (RTG) sweeps can test whether a recommender conditioned on return is controllable, but the intervention is rarely audited. Rewriting every historical RTG token creates an increasingly synthetic context, while rewriting only the current token is more local. We test this distinction in an offline setting with a fixed window. On MovieLens 25M and MyAnimeList 2020 (MAL), we evaluate a Decision Transformer using an RTG locality ladder, a control without RTG, a logged match and score reward check, and a within-trajectory shuffled RTG ablation. On MovieLens, a $K=20$ intervention that covers the full context, applied only to real context positions, shifts the share of Crime predictions by $+23.61 \pm 2.96$ percentage points from the validation 5th to 95th percentile, whereas changing only the current slot shifts it by $+1.77 \pm 1.17$ points. The shuffled RTG model largely removes this response ($+2.08 \pm 1.20$ points at $K=20$). On MAL, the same protocol does not produce a Drama response: $K=20$ changes Drama by $-0.03 \pm 0.07$ points, and $K=1$ by $-0.01 \pm 0.01$. Genre prediction accuracy is numerically close across real RTG, no RTG, and shuffled RTG, and at $K=1$ logged match rates and matched ratings change little. Because dataset and focus-genre selection were exploratory, these magnitudes are descriptive; the cross-diagnostic pattern across locality, shuffled RTG, and the null result on MAL does not establish reward control. We propose four checks: intervention locality, a no-RTG baseline, a reward check, and RTG-content ablation. 

---
# EviGraph: Towards Verifiable Evidence Construction for Information-Seeking Agents 

**Authors**: Jiashun Chen, Yirong Mao, Wenhui Que  

**Link**: [PDF](https://arxiv.org/pdf/2608.24667)  

**Abstract**: Agentic Web search can retrieve relevant information without establishing that the retrieved content actually supports the claims used in an answer. Existing agents typically keep search and evidence recording in a linear interaction trace and optimize primarily for final-answer correctness, providing limited supervision for intermediate grounding. We present EviGraph, a deep-search framework that separates search execution from evidence recording while using a shared policy for the trainable roles. An executor plans concise queries, a frozen evidence verifier inspects source pages and returns verbatim evidence items with an explicit polarity, and the policy maps those items to add/support graph requests that are checked by a deterministic structural validator. The resulting graph serves both as persistent working memory and as a source of dense process rewards, enabling reinforcement learning to directly supervise evidence construction rather than only the final answer. On BrowseComp-Plus, a Qwen3-8B EviGraph agent achieves 35.9% accuracy under a matched interaction budget, compared with 26.9% for the same dual-role architecture without reinforcement learning and 2.7% for a monolithic agent, while generating fewer tokens per rollout. Consistent gains on BrowseComp, GAIA, and XBench indicate that explicitly structuring and rewarding evidence recording improves agentic search 

---
# Rethinking Semantic Alignment in LLM-Enhanced Collaborative Filtering: A Spectral Decoupling Approach 

**Authors**: Yedong Jin, Shaowen Peng, Tsunenori Mine, Shoko Wakamiya, Eiji Aramaki  

**Link**: [PDF](https://arxiv.org/pdf/2608.24363)  

**Abstract**: Recent advances in LLM-enhanced recommendation commonly align semantic representations with collaborative embeddings in a shared space, yet how alignment affects LLM-encoded information remains unclear. In this work, we revisit LLM-enhanced recommendation from a spectral perspective and show that collaborative and semantic signals benefit from different spectral parts. While collaborative representations are dominated by smooth low-frequency components due to user-item homophily, semantic embeddings contain useful non-principal singular components. Through component-wise evaluation and training-dynamics analysis, we find that alignment increasingly concentrates learned representations in dominant collaborative and principal semantic subspaces, reducing overlap with non-principal semantic components. Controlled comparisons show that non-principal components provide inconsistent gains under alignment but consistently improve performance through component-level decoupling, while full prediction-level decoupling achieves the best overall performance. These results indicate that alignment fails to effectively exploit complementary non-principal semantic information. Motivated by these findings, we propose UniSpecRec (Unifying Spectral Signals for Recommendation), which applies signal-specific spectral filtering while preserving collaborative and semantic representations in their respective spaces. UniSpecRec combines their predictions without cross-space alignment or additional trainable parameters. Extensive experiments demonstrate its effectiveness, efficiency, and generalizability. 

---
# RecGPT-Mobile-V2 Technical Report 

**Authors**: Lingqing Zhang, Bin Zhang, Weipeng Huang, Chengfei Lv, Chengyu Lai, Chuxin Chen, Dimin Wang, Han Zhu, Hongtao Cheng, Jialin Zhu, Jian Wang, Jiuning Lin, Junqing Wu, Li Chen, Qichao Ma, Ruiquan Lan, Shuai Zhong, Tao Wang, Xiaodong Zhu, Yinjiang Cai, Yinnan Song, Yipeng Yu, Yuan Liu, Yuning Jiang, Zhaode Wang, Zhibo Xiao, Zhixin Ma, Zihong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2608.24295)  

**Abstract**: Personalized Query prediction maps implicit behavioral signals---clicks, favorites, purchases, and post-purchase exploration---to explicit retrieval intent. On-device deployment makes this task particularly challenging: behavioral trajectories are noisy and multi-scale, multiple Queries may be valid for a single trajectory, and a uniform reasoning policy either expends unnecessary computation on simple instances or allocates insufficient capacity to complex ones. We introduce RecGPT-Mobile-V2, an end-to-end framework that treats intent quality and execution efficiency as coupled objectives within a staged design. The framework transforms heterogeneous interactions into an evidence-preserving trajectory, establishes a recommendation-native foundation through domain adaptation and supervised alignment, and applies reasoning-cost optimization only after grouped rollouts meet grounding and utility criteria. The resulting teacher is distilled into a compact student deployed with low-bit execution, structured compression, and budget-aware device--cloud routing. In an aligned CoT ablation, an evidence-focused short rationale increases ROUGE-L from 0.228 to 0.315 and Jaccard from 0.174 to 0.248, while slightly outperforming the full five-stage rationale. In the controlled RL comparison, the complete reward formulation improves Query quality from 73.2% under quality-only RL to 78.6%, lowers the hard-failure rate from 3.6% to 1.6%, and reduces the median CoT length from 62 to 14 tokens. Online retrieval analysis further indicates that the Query recall channel retrieves inventory complementary to that surfaced by established recall channels. Collectively, these findings support sufficiency-oriented rather than uniformly short reasoning: retain decision-relevant evidence and allocate additional computation only when it is likely to improve the predicted Query. 

---
# Tlow: Flow-based Item Tokenizer for Recommendation 

**Authors**: Nian Li, Chonggang Song, Jingtao Ding, Lingling Yi, Yong Li, Qingmin Liao  

**Link**: [PDF](https://arxiv.org/pdf/2608.24176)  

**Abstract**: Item tokenizer encodes semantic embeddings into token IDs to replace the randomly assigned item IDs used in traditional recommendation models, fundamentally addressing the problems of excessive parameters and cold starts. However, the most common tokenizer, RQ-VAE, suffers from low decoding efficiency due to the inherent dependencies among its codebooks. Meanwhile, efficient independent tokenizers such as optimized product quantization (OPQ) still struggle with dimensional correlations and distribution complexity of semantic embeddings. In this work, we propose a f\underline{low}-based item \underline{T}okenizer (Tlow) to transform raw semantic embeddings into a latent space where embeddings conform to a unified standard normal distribution, achieving dual advantages of dimensional independence and distributional simplicity. Independent tokenization performed on these latent embeddings yields semantically clear token IDs. Additionally, we introduce a novel codebook guidance to align the codebook space with the token embedding space, further aiding the learning of more semantically distinct token embeddings. Offline experiments on four public datasets demonstrate that Tlow's tokenization and codebook guidance significantly improve recommendation performance. The improvement on cross-domain and multi-modal recommendations also proves the effectiveness of item tokenization in a simplified embedding space. Online experiments for a multi-modal retrieval task on China's largest social media platform WeChat validate Tlow's powerful distribution transformation capability. The retrieval model based on token IDs improves user CTR by 10.32\% globally and by 11.64\% for new items. Our codes are available at this https URL. 

---
# Native Multimodal Representation Learning for Click-Through Rate Prediction in E-Commerce Scenarios 

**Authors**: Chao Yi, Feifan Yang, Jiawei Feng, Sishuo Chen, Zhangming Chan, Xiang-Rong Sheng, Han Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2608.24091)  

**Abstract**: Multimodal representations have been widely adopted in industrial e-commerce recommendation systems. Due to their strong semantic understanding and generalization capabilities, they enhance the performance of traditional sparse ID-based Click-Through Rate (CTR) prediction models. Current multimodal application frameworks in the CTR prediction task typically follow a two-stage paradigm: first, pre-training a multimodal encoder on data from specific recommendation scenarios; second, extracting items' multimodal representations using this pre-trained multimodal encoder and integrating them into the CTR prediction model. However, the training objectives and data distribution of multimodal pre-training tasks often differ from those of the CTR prediction task, which limits the effectiveness of multimodal representation on downstream tasks. In this paper, we focus on how to learn Native Multimodal Representation for the CTR prediction task. One intuitive solution is to jointly train the multimodal encoder and CTR model end-to-end on the CTR task, with the expectation that the encoder can automatically learn downstream-relevant knowledge. However, we find that the end-to-end training does not bring performance improvements to existing multimodal application paradigms. Our analysis reveals that user behaviors in raw CTR data are driven by both multimodal semantics and non-multimodal factors, leading to ambiguous supervision and inconsistent encoder updates. To address this, we propose a Mine-Then-Train method that mines high-quality, multimodally interpretable training samples from CTR data and uses them to fine-tune the multimodal encoder for better alignment with user click preferences. Offline and online experiments demonstrate the effectiveness of our approach. 

---
# CodeHID: Learning an Addressable Hierarchical Code Index for Generative Code Retrieval 

**Authors**: Zhen Li, Yuhong Chen, Wenhao Xu, Xiaodong Li, Hui Li  

**Link**: [PDF](https://arxiv.org/pdf/2608.24089)  

**Abstract**: Code retrieval models have predominantly relied on a flat matching paradigm that treats code snippets as independent candidates, making them less capable of distinguishing similar code candidates. Generative retrieval offers a solution by constructing a learnable index over the code corpus, guiding the retriever to better understand how code candidates are semantically organized and addressed. However, naively applying generative retrieval in the code retrieval task may result in operating over an identifier space whose prefixes do not correspond to meaningful code-semantic regions. In this paper, we propose CodeHID, a generative code retrieval framework that reformulates the code retrieval task from flat candidate matching to coarse-to-fine semantic address generation. CodeHID relies on two core components. First, Pseudo-Neighbor Guided DocID Learning constructs a globally static hierarchical index by applying multi-level residual quantization and $k$-nearest-neighbor pseudo-supervision, ensuring that semantically related code snippets share prefixes while preserving target-level separability. Second, Dual-Phase DocID Generation Guidance reliably navigates this fixed index by combining training-side ranking enhancements, using hard negatives and rank distillation, with inference-side candidate constraints and prefix-aware decoding. Extensive experiments on CoSQA and ProCQA benchmarks demonstrate that CodeHID outperforms existing sparse retrieval, pre-trained code models, dense code retrieval, and generative retrieval baselines by a large margin in most cases, achieving particularly strong improvements in rank-one retrieval metrics. 

---
# RetrievalFormer: A Dual-Encoder Transformer for Efficient Approximate Nearest Neighbor Retrieval and Cold-Item Recommendation 

**Authors**: Theodore Rogers, Joe Standerfer, Dmitrii Timoshenko, Haoxue Li, Zuhaib Akhtar, Soyoung Yang  

**Link**: [PDF](https://arxiv.org/pdf/2608.24079)  

**Abstract**: A shared search-and-recommendation index must score new items from features alone because search has no exploration slot. In a public log covering both surfaces over one catalog, $38.6\%$ of held-out query-search impressions show an item never previously shown or visited. For user-cold engagements, the feature-based tower serves this demand without measurable loss against $99$ sampled negatives ($0.9595$ Recall@20 versus $0.9510$ warm). A lexical baseline reaches similar parity, while a full-catalog check remains statistically undecided. Dual-encoder retrieval therefore keeps the index \emph{open} to new items, unlike an ID-softmax recommender that requires retraining. We price this openness on recommendation against six sequential baselines, each retrained and tuned through five rounds on corrected targets. A float32 timestamp bug had reordered leave-one-out targets for $19.7\%$ of users. On MovieLens-1M, warm accuracy trails the strongest retrained baseline by $5.2\%$ Recall@20 and $11.4\%$ NDCG@20. On MIND, the gap narrows to $0.8$--$3.6\%$ relative to the five strongest baselines, though the model ranks sixth of seven. Under strict zero-leakage cold-start evaluation, the content tower achieves $0.172 \pm 0.006$ Recall@20, $1.4\times$ the strongest retrained dedicated method ($0.124 \pm 0.007$) and $3\times$ a training-free floor, without cold-specific training. Exact full-softmax training raises Recall@20 by $54\%$ on MIND-small and $6.9\%$ on MovieLens-1M over sampled InfoNCE, but recomputes the full catalog each step and exhausts accelerator memory at $240$K items. Approximate nearest-neighbor search explains none of the remaining gap, serving cost does not regress against ID-softmax retrieval, and a history-window sweep explains half the post-recipe remainder. Exact-quality training at catalog scale remains the open problem. 

---
# SQLite is Enough. Lexical, Semantic, and Hybrid Search with scrydb 

**Authors**: Timo Breuer  

**Link**: [PDF](https://arxiv.org/pdf/2608.24060)  

**Abstract**: This work introduces scrydb, a Python library that enables lexical, semantic, and hybrid search within SQLite. For lexical search, scrydb leverages SQLite's full-text search extension FTS5. Semantic search builds on sqlite-vec, a SQLite extension for vector search. Furthermore, the library allows users to rerank and fuse retrieval results to combine both lexical and semantic approaches, providing a lightweight solution for downstream tasks in information retrieval (IR) or agentic search. We evaluate scrydb on various IR benchmark datasets and demonstrate its effectiveness in text retrieval based on keyword matching, semantic similarity, and rank fusion. In addition, we provide insights into query latency and the trade-off between efficiency and effectiveness. scrydb is available under the MIT license. 

---
# TAGR: Temporally Adaptive Generative Recommendation for Industrial Live-Streaming Advertising 

**Authors**: Wencai Ye, Guangyi Liu, Chaoyi Wang, Wenbin Luo, Shengyu Wang, Mingjie Sun, Peng Wang, Quanming Yao, Wenjin Wu, Peng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2608.24034)  

**Abstract**: Live-streaming advertising is an important monetization channel on short-video and e-commerce platforms, where rapidly changing live content, promoted products, and user feedback impose strong freshness requirements on recommendation models. Existing generative recommenders designed for static domains fail at three levels: static semantic IDs (SID) cannot track evolving live ads; single-scale behavior modeling misses shifting intent; preference optimization conflicts between fresh on-policy feedback and training stability. We propose TAGR, a generative recommendation framework with temporal adaptation at three levels: live-ad tokenization, user intent modeling, and preference alignment. At the token level, Live Semantic-Collaborative ID (LSID) periodically refreshes each active ad's SID based on its current live scene and promoted products, while retaining a stable hierarchical token vocabulary for autoregressive generation. At the intent level, Intent-Aware Generation (IAG) models live-room entry histories at multiple temporal granularities as the primary intent sequence, keeps auxiliary behaviors as separate inputs, and weights next-token prediction (NTP) using post-request intent evidence and business value. At the alignment level, Intermittent On-Policy Preference Optimization (IOPO) periodically samples fresh candidate groups from the current policy and performs behavior- and value-aligned preference updates interleaved with supervised NTP maintenance to preserve learned behavior distribution. Deployed on a large-scale e-commerce live-stream advertising platform, TAGR improves live-room entry and shopping-cart click rates by 8.5% and 7.4%, respectively, and achieves a 16.1% revenue lift over the production baseline. These results demonstrate the effectiveness and industrial viability of temporally adaptive generative recommendation for live-stream advertising. 

---
# Hybrid Semantic Tool Discovery for Enterprise MCP Gateway: Architecture and Implementation 

**Authors**: Olympia Saha, Amy Wang, Srinivasan Manoharan  

**Link**: [PDF](https://arxiv.org/pdf/2608.23992)  

**Abstract**: Large language model (LLM) agents invoke external tools to retrieve and reason over information beyond pretrained knowledge. The Model Context Protocol (MCP) standardizes how such tools are surfaced, and a proxy MCP server aggregates many backend servers behind a single endpoint providing a secure, governable chokepoint for authentication, policy enforcement, and observability. This architecture creates two compounding challenges: a context-engineering bottleneck where full tool schemas saturate the model context window before any user query, and a tool discoverability barrier where users and agents cannot identify the best tool among 2,000+ indexed tools across 200+ MCP servers. Prompt caching reduces reprocessing cost but neither frees context capacity nor improves accuracy. We present SCOUT (Selective Context Optimization for Universal Tooling), which reframes tool exposure as a context-selection problem, injecting only tools relevant to the current step. SCOUT surfaces two MCP meta-tools -- tool_search and execute_tool -- where tool_search performs hybrid retrieval, fusing BM25 sparse matching with dense vector search via Reciprocal Rank Fusion to return the top-k relevant tools. Backed by zero-downtime catalog update pipelines, SCOUT resolves both context saturation and tool discovery challenges. In production at PayPal, SCOUT reduces MCP tool-token consumption from 140.2k tokens (70.1% of context) to 1.3k tokens (0.8%), a 99% reduction, cutting per-query inference cost at enterprise scale. Because SCOUT is surfaced as standard MCP tools, it is model-agnostic and requires no client-side modifications. 

---
# Wontopos Tablet 2: Measuring Multilingual and Multimodal Memory Retrieval Without Lexical Matching 

**Authors**: Sunwoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2608.23920)  

**Abstract**: We measure tablet-2, a production long-term memory engine for language models, on the text benchmarks the field already uses and on cross-lingual retrieval of photographs stored with no text at all. Its retrieval path contains no lexical matching, no keyword scoring, and no language model of its own.
On LongMemEval-S (500 questions) it scores 95.7% [93.4, 97.1]; on BEAM-1M (700 questions, 2.21M stored memories) 67.5% [64.8, 70.2]. Those are question-sampling intervals, not the run-to-run spread, which is an order of magnitude narrower.
Most of the paper is about how little they mean alone. Holding engine, corpus, settings and judge fixed, changing only the reader moves LongMemEval-S by 2.0 points; changing only the re-ask budget moves BEAM-1M by 8.9. Neither is stated in the reports we compare against, and the second exceeds most gaps there, so we give that table as a placement and not a ranking.
For the multimodal axis we run two controls. Against BM25, configured as strongly as we could, we reach 95.2% mean recall@5 over 70 store-and-query language cells where BM25 reaches 19.0% and is exactly zero in 54. On captionless photographs a lexical method has no document to score at all. Open dense baselines on 300 Crossmodal-3600 photographs in 14 languages show that density confers no language independence: one scores 91.0% on English and 4.7% on Russian from identical image vectors, and a multilingual variant collapses on Telugu and Swahili. Our spread across languages is 14.0 against their 27.5 and 27.7.
Three results run against us and are reported at equal weight: low-resource languages degrade sharply (Swahili 53.0%, Telugu 64.0%), attaching captions lowers cross-lingual retrieval by 11.4 points, and one setting omitted into one stage of our own retrieval cost 37 points of Korean top-1 accuracy while leaving nine languages untouched. 

---
# AdaWidth: Query-Adaptive Embedding Width for Dense Retrieval 

**Authors**: Shubing Yang, Dongfang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2608.23862)  

**Abstract**: High-dimensional embeddings are central to dense retrieval, but not all of these dimensions need to be evaluated at retrieval time. Existing methods reduce dimensions in two ways: truncating the same leading dimensions for every query, or masking a different subset for each query while still storing and accessing the full embedding. Yet queries within a single task differ widely in the number of dimensions they need for their rankings to stabilize.
We introduce AdaWidth, which adapts the number of evaluated dimensions to each query within a shared prefix representation. An orthogonal prefix adapter applies a single learned rotation to queries and documents alike, concentrating discriminative signal in leading coordinates while leaving every full width inner product unchanged. A lightweight router then reads order statistics off the ranking a query has already produced, and evaluates more dimensions only for the queries whose top results would change. We further derive a prefix sufficiency analysis showing that the required number of dimensions is set by the competing documents at the retrieval cutoff: it grows logarithmically with corpus size, decreases logarithmically with retrieval depth, and remains heavy-tailed across queries. Across six retrieval tasks and five frozen encoders, AdaWidth matches the NDCG@10 of state-of-the-art dimensionality reduction using 55% to 84% fewer dimensions per query. 

---
# Structurally-bounded Agentic Graph Exploration for Evidence-Grounded Scholarly DeepSearch 

**Authors**: Rima Hazra, Sayan Layek, Somnath Banerjee, Soumen Chakrabarti, Animesh Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2608.24809)  

**Abstract**: We present Crase, a bounded and inspectable alternative to deep research agents for scholarly search. Instead of an open-ended search loop, Crase queries a search engine once for seed papers, expands them along their 1.5-hop citation neighborhood, prunes citation edges whose claims lack entailment support, and ranks the remaining papers with a recency-aware random walk. This makes the candidate set, the reason each paper is kept, and the stopping condition explicit and fixed before inference. On LitSearch and one further benchmarks over a 500K-paper arXiv corpus, Crase outperforms deep research agents built on proprietary models by up to 3$\times$ recall@50 at roughly a third of the cost. 

---
# Matched Excess-Outranker Regularization for Candidate-Set Interference in Continual Knowledge Graph Embedding 

**Authors**: Hao Ren, Junbin Gao, Jiaojiao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2608.24273)  

**Abstract**: Continual knowledge graph embedding updates entity and relation representations as a graph grows. Existing methods primarily address catastrophic forgetting, but entity admission also changes the candidate universe of every compatible query. A historical answer can therefore lose rank even when its score and its ordering among old entities are preserved. We formalize this effect as candidate-set interference and introduce Matched Excess-Outranker Regularization (MEOR), a host-level objective that compares smooth answer-relative newcomer pressure with score-blind, structurally matched old references. Its one-sided penalty acts only when newcomer competition exceeds the matched reference, preserving the host learner's signal for legitimate new entities. Across eight paired runs on ENTITY-ComplEx, MEOR improves historical current-universe mean reciprocal rank (MRR) by 0.0057 over replay and reduces candidate-set interference by 0.0055, with one-sided 95% lower bounds of 0.0052 and 0.0051, respectively. It satisfies the preservation criteria for old-universe ranking and newcomer acquisition and improves historical current-universe MRR over persistent calibration, matched maximum regularizer (MMR), and unmatched old regularizer (UOR). Direct ablations support each component of its reference construction and aggregation. Adding MEOR also improves historical ranking in all ten reported FBInc-S and FBInc-L host and backbone settings, with every paired 95% confidence interval excluding zero. These results establish candidate admission as a distinct source of continual rank loss and show that it can be controlled without replacing the underlying embedding architecture or continual learner. 

---
# PlaceSeek: Human-Centered Geospatial Retrieval of Urban Outdoor Places via Semantic Grounding and Affective Alignment 

**Authors**: Ziqi Cui, Shangyu Lou  

**Link**: [PDF](https://arxiv.org/pdf/2608.24133)  

**Abstract**: People search for urban outdoor places not only by category or function, but also by what activities a place can support and how it is perceived. Existing geospatial retrieval remains largely POIcentric and metadata-driven, making it difficult to satisfy openended, affective, or activity-oriented needs. We present PlaceSeek, a human-centered outdoor place retrieval framework that maps natural-language queries to geolocated street-view imagery. PlaceSeek introduces an intent-aware retrieval mechanism that decomposes user queries into functional and affective sub-intents. A Semantic Grounding Module verifies whether candidate street-view results contain the physical evidence needed to support the intended activity, while an Affective Alignment Module re-ranks physically valid candidates using a LoRA-adapted vision-language model trained on human urban perception judgments. We evaluate PlaceSeek on 31,956 street-view locations in Milan across 10 naturallanguage queries annotated by five human evaluators. PlaceSeek achieves 88.0% Precision@5, a mean match score of 3.39/4.0, and 0.920 nDCG@5, outperforming CLIP, fine-tuned CLIP, SigLIP, and a VQA-based baseline. Ablation results show that physical grounding is essential for retrieval validity, while affective alignment improves ranking quality among physically valid candidates. These findings highlight that complex urban spatial queries require modeling both verifiable visual evidence and human perceptual preferences. PlaceSeek provides a potential framework for human-centered nextgeneration geospatial retrieval systems. 

---
# PARTAB: Partition-Aware Reasoning with Structured Evidence for Scalable Table Understanding 

**Authors**: Md Mahadi Hasan Nahid, Davood Rafiei  

**Link**: [PDF](https://arxiv.org/pdf/2608.24082)  

**Abstract**: Large Language Models (LLMs) have shown strong capabilities in table reasoning, but their effectiveness degrades as tables grow in size and complexity due to irrelevant context and difficulty localizing the evidence required for reasoning. Existing approaches typically reason over either the full table or a single reduced view, which can still obscure important row-column relationships. We introducePARTAB (Partition-Aware Reasoning overTables), a framework that constructs a structured evidence interface between the LLM and the table. PARTAB represents query-relevant evidence as semantically coherent, row-linked table regions and performs hierarchical selection over column groups and row-level partitions before composing the selected evidence for answer generation. We evaluate PARTAB on multiple table reasoning benchmarks, covering question answering, fact verification, and numerical reasoning. PARTAB consistently improves over full-table prompting and several recent table reasoning methods, achieving strong performance on WikiTableQuestions and TabFact while remaining competitive on numerical reasoning. Additional analyses show that semantic partitioning and targeted evidence selection improve evidence localization, substantially reduce the reasoning context, and provide larger benefits on complex tables. These results demonstrate the value of structured, partition aware evidence construction for scalable table reasoning. 

---
# WeMM-Embedding: WeChat Multi-Modal Embedding Technical Report 

**Authors**: Junjie Zhou, Ke Mei, Lei Li, Tianyi Wang, Fengyun Rao, Jing Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2608.24053)  

**Abstract**: Universal multimodal embeddings are becoming a core component of modern AI systems, enabling heterogeneous content to be represented in a shared space for applications such as retrieval, recommendation, classification, and agentic systems. In this report, we present WeMM-Embedding, a family of universal multimodal embedding models supporting text, images, videos, visual documents, and arbitrarily interleaved multimodal inputs with flexible output dimensions. The family comprises 2B, 4B, and 9B variants and is trained in two stages: a large-scale multimodal alignment stage, followed by a refinement stage using curated data, fine-grained relevance supervision, and cross-scale knowledge transfer. Across extensive evaluations, WeMM-Embedding achieves leading performance on multiple public benchmarks. Notably, the 2B variant already surpasses the previously leading 8B open-source baseline on MMEB-v2, while the 9B variant further achieves a new state-of-the-art overall score of 80.6. WeMM-Embedding also demonstrates strong practical performance across WeChat applications, with substantial gains on a 26-task in-house benchmark and consistent improvements across 14 online A/B tests. It has been deployed at scale across recommendation and search applications, including WeChat Channels, Official Accounts, Moments, and e-commerce services. We have released the model weights and code to facilitate future research at this https URL. 

---
# RAGSentinel: Certifiable Geometric Consensus for Robust Retrieval-Augmented Generation 

**Authors**: Yueyang Quan, Anjun Gao, Yufei Xia, Minghong Fang, Zhuqing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2608.23965)  

**Abstract**: Retrieval-augmented generation (RAG) improves the factuality of large language models by grounding responses in external documents, but it also exposes a critical security vulnerability: adversarial documents injected into the knowledge database can enter the context window and steer the model toward targeted incorrect answers. Existing post-retrieval defenses rely on instruction following, parametric knowledge, or text-level consistency, all of which can be imitated or optimized against by adaptive attackers. We propose RAGSentinel, a training-free, label-free defense for black-box RAG systems. RAGSentinel uses a surrogate encoder to measure query-conditioned hidden-state shifts induced by retrieved documents, removes shared topic directions, and filters poisoned documents as geometric outliers from a robust majority consensus. We prove that, under an honest-majority assumption and a representation-level separation condition, RAGSentinel exactly recovers a poison-free majority-sized context. Experiments across three question-answering datasets, three LLM families, and multiple poisoning attacks show that RAGSentinel consistently achieves low attack success rates while preserving competitive accuracy and remaining effective against adaptive attacks with full pipeline knowledge. 

---
# NeuronGuard: Robust LLM Safety Alignment via Ablation-Aware Safety Signal Redistribution 

**Authors**: Anjun Gao, Yueyang Quan, Yufei Xia, Zhuqing Liu, Minghong Fang  

**Link**: [PDF](https://arxiv.org/pdf/2608.23959)  

**Abstract**: Safety alignment in large language models (LLMs) remains brittle against a growing spectrum of attacks. Jailbreak attacks bypass safety mechanisms through crafted prompts, while neuron-level attacks directly prune safety-critical neurons post-deployment. Both exploit a common weakness: safety-relevant information concentrates in a sparse neuron subset. We present NeuronGuard, a fine-tuning-stage defense that simultaneously hardens LLMs against both attack classes by redistributing safety signals across a broader set of neurons. NeuronGuard dynamically identifies safety-critical neurons via periodically refreshed per-layer linear classifiers, forces refusal behavior under deliberate neuron ablation, and applies KL-divergence regularization for distributional consistency. A randomized gradient projection strategy preserves downstream task utility by resolving conflicts between the defense and task objectives. We provide a formal guarantee that NeuronGuard strictly reduces the attack success rate (ASR) upper bound, and experiments across three LLMs, six state-of-the-art attack strategies, and multimodal settings confirm near-zero ASR while maintaining task accuracy, including against white-box adaptive adversaries. 

---
# Degree Centrality Algorithms for Weighted Multilayer Networks (or w-MLNs) 

**Authors**: Ayomide Ayowole-Obi, Abhishek Santra, Sharma Chakravarthy  

**Link**: [PDF](https://arxiv.org/pdf/2608.23876)  

**Abstract**: Centrality measures are defined for simple graphs -- directed, undirected, weighted or unweighted. Attributed graphs have to be reduced to simple graphs for computing centrality measures. However, when applications with multiple types of relationships are modeled using multilayer networks (MLNs), simple graph algorithms cannot be directly used. Existing approaches typically analyze MLNs by aggregating layers of an MLN into a single graph, which results in the loss of structural and semantic information. The semantic information loss can be more pronounced particularly, in weighted networks.
This work focuses on computing degree centrality in weighted homogeneous multilayer networks (HoMLNs) using a decoupling-based framework. The framework performs independent layer-wise analysis on MLNs without reducing them to simple graphs. The decoupling approach allows use of exiting algorithms for each layer and uses minimal information from individual layers for computing degree centrality of HoMLNs. We propose heuristic-based algorithms that strike a balance between accuracy and efficiency.
The proposed methods are evaluated against ground truth (GT) results obtained using Boolean OR aggregation and naive baselines. Experimental results on both synthetic and real-world HoMLN datasets demonstrate that the heuristics achieve accuracy comparable to the ground truth while significantly improving computational efficiency, thereby establishing the scalability and effectiveness of the HoMLN algorithms developed using the decoupling approach. 

---
