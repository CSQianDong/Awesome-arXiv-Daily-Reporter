# Towards Realistic Evaluation of Uncertainty Quantification for LLM-Based Recommender Systems 

**Authors**: Hao Wu, Sipeng Chen, Weikai Li, Tianyi Chen, Qiancheng Yang, Hang Zhou, Yizhou Sun, Wei Wang, Xiao Luo  

**Link**: [PDF](https://arxiv.org/pdf/2611.33036)  

**Abstract**: Large language models (LLMs) provide a flexible interface for recommendation systems. However, their issued rankings require reliable instance-level uncertainty monitoring. In this scenario, comparing uncertainty quantification (UQ) methods becomes challenging because LLM rankers lack a standard item-classification head. Their scores may depend on option labels, prompts, generation procedures, and the available user and item evidence. Existing methods derive uncertainty from different sources and apply it at different stages of the recommendation process. Some methods evaluate an issued ranking, whereas others use uncertainty to change the ranking itself, making direct comparison difficult. Here we introduce a fixed-ranking benchmark which evaluates eight heterogeneous UQ signals as record-level monitors under a shared output contract. We evaluate these signals across four recommendation datasets and two LLM backbones under a shared fixed-ranking protocol. From the results, we draw three main observations. First, UQ method ordering changes across backbones even on the same reference instances. Second, ranking-aligned Top10Mass provides the strongest scalar Error@10 detection in several evaluation conditions, reflecting effective cutoff-aligned monitoring, while its advantage narrows for popularity-tail targets. Third, scalar UQ signals are weak at identifying which control has the larger clipped NDCG@10 difference, whereas source-specific signals, including Perturbation Rank Instability and simple listwise distances, provide stronger paired-control agreement, reflecting their sensitivity to the two controlled evidence sources. These findings motivate us to evaluate ranking-failure detection, robustness across recommendation conditions, and responses to controlled evidence changes as distinct properties of UQ. 

---
# SOARS: System One Assisted Reranking for Agentic Search 

**Authors**: Amin Bigdeli, Liye Fu, Fattane Zarrinkalam, Joel Stremmel, Elizabeth Lingg  

**Link**: [PDF](https://arxiv.org/pdf/2611.32917)  

**Abstract**: Deep search agents address complex information needs through iterative retrieval and reasoning, where the relevance of retrieved evidence influences downstream performance. Existing rerankers improve retrieval effectiveness but introduce substantial latency and cost when invoked throughout the search trajectory. We introduce SOARS, a training-free framework leveraging a System One model for fast, calibrated relevance estimation. At each search step, SOARS selects query-relevant passages from long documents and reranks candidates in a single call, jointly considering the original question and agent-generated sub-query. Across six LLM backbones on BrowseComp-Plus and WebShaper, SOARS consistently improves answer accuracy and evidence recall while reducing search calls, end-to-end latency, and inference cost. On BrowseComp-Plus, it improves GPT-5.4 Mini's accuracy from 73.9% to 84.7% while reducing search calls from 23 to 14, demonstrating the potential of System One reranking for more effective and efficient deep search. 

---
# Speculative Decoding for Semantic-ID-Based Generative Recommendations 

**Authors**: Harshal A. Chaudhari  

**Link**: [PDF](https://arxiv.org/pdf/2611.32855)  

**Abstract**: Generative recommender systems with Semantic IDs (SIDs) represent each item as a short sequence of $L$ discrete tokens and use a sequence model to predict the next item's tokens from a user's history. Industry-scale deployments typically use these models to retrieve multiple candidates per request with constrained beam search. However, this incurs a fixed cost of $L$ sequential model calls per request, and accuracy degrades as the SID generator becomes less sharp, without a guarantee on how close the output is to the exact top-$K$. Production systems therefore cannot detect degraded retrieval for individual requests.
We propose SIDECAR, which uses approximate draft candidates and a tree-attention pass to score the entire draft in a single model call, memory permitting. This pass bounds the probability of every unexplored SID prefix and certifies the exact top-$K$ if the draft contains them. Otherwise, a branch-and-bound fallback expands only the parts of the trie that can still change the output, reaching certification within at most $L$ calls for any draft. Moreover, under a budget of $M$ tokens per call and $Y$ calls, which for $Y = L$ matches the resources of a beam search of width $M$, SIDECAR announces a factor $1 + \delta$ within which its list will be exact at every rank. Depending on model sharpness, SIDECAR returns the exact top-$K$ using a fraction of beam search's calls or memory. Even when the budget is too small to recover the exact top-$K$, SIDECAR, unlike beam search, reports the announced factor, allowing the serving system to stop early once a quality target or latency limit is reached. 

---
# LazUnirec: Unifying Retrieval and Ranking in Multi-scenario Industrial Recommendation 

**Authors**: Xiaodong Li, Junfeng Zhang, Shichao Nie, Ziyue Guo, Yiming Liu, Xinxin Xu, Xurong Li, Jinyuan Dang, Zhanming Ou, Jianan Yan, Tao Luo, Xiaoyi Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2611.32590)  

**Abstract**: Modern industrial recommendation systems typically divide retrieval and ranking into two separate cascaded stages, leading to inconsistent optimization objectives, redundant encoding of user sequences, and a waste of computational resources. Previous efforts have investigated using Transformers to model both retrieval and ranking simultaneously. However, these studies still face two major limitations: (1) Since retrieval and ranking share a single Transformer backbone, the gradient conflict arises when the gradient directions of the two tasks are inconsistent. (2) When gradient conflict occurs, the model cannot determine a reliable direction for gradient updates. To this end, we propose a Unified Retrieval and Ranking approach for Multi-scenario Industrial Recommendation, termed as LazUnirec. Specifically, we construct task-shared and task-private parameters on the Transformer backbone to mitigate the gradient conflict. Wherein, for the attention components, we propose LoRA-modulated attention, enabling retrieval and ranking to share attention parameters while preserving task-private parameters. For the feed-forward networks (FFN), we present task-adaptive sparse mixture-of-experts (MoE) to mitigate gradient conflict by constructing task-shared and task-private experts. Moreover, we propose an uncertainty estimation that evaluates the gradient reliability of the retrieval and ranking, thereby guiding the model toward a more reliable gradient update direction when gradient conflict occurs. In addition, knowledge distillation from ranking to retrieval further enables cross-task knowledge transfer. Extensive offline experiments and online A/B testing demonstrate the effectiveness and superiority of LazUnirec. LazUnirec has been deployed in online multi-scenario industrial recommendation system and currently delivers significant gains for the platform. 

---
# OmniEVIE: Evidence-Vector-Informed Embedding for Multi-modal Retrieval 

**Authors**: Zifei Wang, Wei Wen, Qiang Ji, Qian-Wen Zhang, Ruizhi Qiao, He Liu, Xing Sun, Qiang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2611.32203)  

**Abstract**: Multimodal retrieval must match dense pages, fine-grained entities, ordered video, and agent tools and memory. Single-vector embeddings lose local evidence, cross-encoders reread every recalled item online, and uncompressed late interaction is too large to serve. We present OmniEVIE, a 4.5B multi-vector embedding model covering text, images, visual documents, video, and tools with one compressed index and a three-stage retriever. Each record stores at most 128 int8 vectors of dimension 64 (9,472 bytes, or 8.82 GiB per million records), providing a mean vector, 1-bit signs, and int8 multi-vectors for coarse, middle, and final search while localizing each hit to a page region or video interval. Training, merging, indexing, and retrieval are co-designed: all tasks share one retrieval format, a three-grade evidence check removes false negatives, dynamic sampling follows the baseline-teacher gap, candidates are compressed with ST-HAC during training to match the index, stage-wise and specialist models are merged by parameter block, and a saliency gate with a noise floor refines query scoring. Trained on public datasets, OmniEVIE scores 60.68 (full d=2048 sequence) and 60.06 (K128 x d64 int8 index) on MMEB-V3 (190 tasks), 77.96 on MMEB-V2 (78 tasks), and 62.96 on ViDoRe V3 (48 tasks). On 1M and 5M records, a single-thread query takes 54.2 ms and 120.2 ms (11.1x and 25.3x faster than exact 4096-d single-vector search; 236x and 532x faster than full int8 MaxSim). Video is encoded directly from frames without a parsing model: on QVHighlights (1,083 queries), the peak falls inside the labeled segment for 78.0% of queries and the returned span (18% of video length) intersects it for 94.3% vs. 44.9% random. Weights and code: this https URL and this https URL. 

---
# LARF: Learning-aligned Reasoning Framework for Search Relevance 

**Authors**: Tian Tang, Guo Chen, Reed Li, Hao Wang, Yu Lu, Haijin Liang, Haibo Shi  

**Link**: [PDF](https://arxiv.org/pdf/2611.32125)  

**Abstract**: Query-document relevance estimation is a core component of AI-native search systems, where a model must assess whether retrieved documents satisfy users' information needs under complex, multi-dimensional criteria. However, directly applying large language models (LLMs) to this task remains challenging because conventional supervised fine-tuning (SFT) provides limited supervision for domain-specific judgment rationales, while standard trajectory-level reinforcement learning (RL) objectives do not explicitly enforce margin separation among adjacent ordinal relevance labels. Moreover, although such reasoning can be useful during training, chain-of-thought (CoT) reasoning is impractical for latency-sensitive online ranking. These challenges make it difficult to resolve ambiguous decision boundaries. We propose a three-stage framework LARF (Learning-Aligned Reasoning Framework for Search Relevance): it first performs cold-start SFT on strategically selected samples with multi-agent guided CoT rationales, then applies discriminative GRPO with attribute-wise, list-wise, and rule-consistency rewards together with a margin loss on the terminal relevance token, and finally distills the teacher into a dual-head student whose classification heads are retained for non-autoregressive inference. On a human-annotated web search relevance benchmark with approximately 300K training and 16K test instances, LARF-8B improves six-class classification accuracy by 5.78 percentage points and NDCG by 5.05 points, while the classification-only serving path reduces inference latency by 82.9\%. These results indicate that aligning structured reasoning, terminal decision optimization, and classification-head distillation can provide an effective accuracy--efficiency trade-off for industrial search relevance modeling. 

---
# User-Adaptive Fusion of Multiple Foundation Models for Recommender Systems 

**Authors**: Sunwoo Kim, Hyunjin Hwang, Geon Lee, Kyungho Kim, Kijung Shin  

**Link**: [PDF](https://arxiv.org/pdf/2611.31951)  

**Abstract**: Can recommender systems benefit from combining a collection of foundation models within and across modalities? While different foundation models may capture complementary aspects of item metadata, we find that naively combining their representations yields only marginal gains over using a single foundation model. We make a key observation: the users who benefit vary across foundation models. Motivated by this observation, we introduce UF2O, a user-adaptive approach for fusing multiple foundation models for recommendation. In a nutshell, UF2O models the contribution of each foundation model as user-specific by maintaining separate recommendation views and adaptively combining their scores for each user. By performing fusion at the score level, UF2O can also fuse heterogeneous foundation models without directly aligning their representation spaces. Extensive experiments that jointly combine four foundation models on four datasets show that UF2O outperforms existing fusion approaches in both static and sequential recommendation settings, with improvements of up to 7% over the strongest fusion baseline. The code, datasets, and supplementary materials for our work are available at this https URL. 

---
# MELODI: Similarity-Identity Decoupling for Lightweight Ranking 

**Authors**: Quoc Phong Nguyen, Vuong Le, Volodymyr Vaskovych, Julien Monteil  

**Link**: [PDF](https://arxiv.org/pdf/2611.31940)  

**Abstract**: For latency-constrained candidate ranking, we propose a similarity-identity decoupling framework that separates target-history compatibility from target-only information. We study two orders for computing pooled similarity, similarity-then-pool and pool-then-similarity, and identify linearity of the similarity operator in its history argument, for each fixed target, as sufficient for their equivalence, justifying the cheaper pool-then-similarity form. We instantiate the framework as multi-head efficient lightweight operator for decoupling identity and similarity (MELODI), whose similarity branch combines target-conditioned multi-head history pooling with a single residual-free cross layer that is linear in its pooled-history input and whose identity branch is problem-dependent. On public benchmark datasets, MELODI matches or surpasses the evaluated target-aware and target-agnostic baselines. On two large-scale industrial datasets, its area under the precision-recall curve (AUPRC), normalised discounted cumulative gain (NDCG), and recall values are equal to or higher than those of the evaluated target-aware baselines. Processor measurements show lower latency than the evaluated candidate-conditioned baselines at large candidate counts, further reduced by request-local history reuse. 

---
# Query Obfuscation with Embedding-Optimized Noise (EON): A Representational Blindspot in Agentic Retrieval 

**Authors**: Ankit Gupta, Christoph Adami, Emily Dolson  

**Link**: [PDF](https://arxiv.org/pdf/2611.31344)  

**Abstract**: In modern agentic and semantic retrieval systems, dense embedding models increasingly determine which intent, tool, document, memory, or image to retrieve for a query. We identify a representational failure mode in this pipeline: a meaningful query can be replaced by visibly nonsemantic content while preserving much of its retrieval outcome, even when the candidate bank is hidden from the optimizer. We introduce Embedding-Optimized Noise (EON), constructed using only the natural query representation, without access to candidate embeddings, relevance labels, rankings, retrieval feedback, or guardrail decisions. Despite this corpus-blind objective, EONs frequently preserve intent and tool routing, benchmark document search, episodic agent-memory retrieval, and cross-modal search across multiple dense embedding families. Crucially, this disconnect carries serious security implications: for malicious queries, an EON can reliably evade input guardrails that block the natural query, executing the intended safety-sensitive retrieval undetected. Through the geometry of the embedding space, we then characterize how query-level representation matching can reliably maintain subsequent retrieval outcomes over candidates unseen during optimization. Together, these findings expose a fundamental phenomenon across dual encoders that we call BLINDSPOT, where systems remain operationally blind to nonsemantic artifacts while reliably retrieving real, consequential content. 

---
# SAG-Mem-1.0 Technical Report 

**Authors**: Yuchao Wu, XingCheng Liang, Xuanze Wang, Yini Huang, Yue Wu  

**Link**: [PDF](https://arxiv.org/pdf/2611.33124)  

**Abstract**: Long-term conversational memory requires retrieving complementary facts scattered across dialogue sessions, including intermediate evidence with limited similarity to the original question. Existing approaches often rely on derived memory representations or persistent relational structures, introducing construction costs and potentially losing source details. We present SAG-Mem, a conversational memory framework that preserves complete dialogue messages in non-overlapping chunks and discovers complementary evidence through query-time lexical expansion. Memory construction requires no model inference, using a lexical index without LLM-based extraction, summarization, or semantic relation construction. At query time, BM25 retrieves initial candidates, an evidence evaluator identifies useful chunks, and lexical cues from selected evidence guide an additional retrieval round. Retrieved dialogue is further annotated with calendar interpretations of supported relative-time expressions, preserving the original wording and temporal granularity. We also introduce SAG-Mem-Lite, an alternative configuration that replaces evidence scoring with embedding-based candidate scoring, achieving lower operational cost than Full Context and all memory-based baselines across both benchmarks with low query latency. On LoCoMo and LongMemEval, SAG-Mem achieves overall scores of 83.08 and 80.20, respectively, the highest among the evaluated methods on both benchmarks. On LongMemEval, it exceeds the strongest compared baseline by 7.20 points. SAG-Mem-Lite achieves scores of 77.09 and 77.20, respectively. These results support dialogue preservation and evidence-guided retrieval expansion as an effective design for long-term conversational memory. 

---
# Delegent: A Multi-Agent Harness That Reduces Cost without Sacrificing Performance 

**Authors**: Lingwei Gu, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2611.33098)  

**Abstract**: Agents built on frontier language models are expensive. At every step, the frontier model re-reads its entire context and executes every subtask regardless of its difficulty. This paper explores the simple thesis that much of this cost can be removed by delegating easy subtasks to a small language model, or SLM, that runs on hardware one already owns. To test it, we build Delegent, a simple and effective agent harness built on Claude Code, in which the frontier model stays in charge of every task but hands focused subtasks, such as investigating code, running tests, and searching a corpus, to SLM subagents. Existing approaches such as cascades and model routers usually choose which model executes the entire task, whereas in Delegent, the frontier model always leads the execution and delegates easy subtasks to the SLM. No router or model is trained, so either model can be swapped freely. With the help of a 27B FP8 SLM, Delegent reduces the frontier cost of long-horizon coding by 67% and of agentic search by 59%, with no loss in task performance, and even a 2B SLM that fits on a laptop reduces the coding cost by 66%. The savings grow with the legwork that the frontier model would otherwise do itself, and a short instruction, which leaves the main coding tasks to the frontier model, accounts for most of the saving and keeps the 2B SLM from losing tasks. Code: this https URL 

---
# A Text-to-SQL Benchmark for Official Statistics 

**Authors**: Lucas Lageweg, Karen Goes, Tom van den Broek, Frank Pijpers, Paul Groth, Jan-Christoph Kalo  

**Link**: [PDF](https://arxiv.org/pdf/2611.32646)  

**Abstract**: Official statistics are critical datasets for a wide variety of actors, including policymakers and businesses. For providers of official statistics, improving accessibility, governance, and disclosure of their data is of key importance. This can be improved by providing question-answering systems based on text-to-SQL approaches. However, existing text-to-SQL benchmarks do not resemble this setting. Official statistics are published as open-domain catalogs of thousands of wide tables whose columns and values are codes, and whose meaning is recorded in separate metadata rather than in a relational schema. To address this gap, we introduce LOCuST, a text-to-SQL benchmark that pairs over two thousand English and Dutch tables from Statistics Netherlands with a semantic knowledge graph. The graph describes the tables, measures, and dimension codes through labels, definitions, hierarchies, and units, and is the external knowledge a system needs to find the right tables and ground a question in the right codes. The benchmark includes expert-annotated questions for different query and aggregation types, including raw value retrieval, summations, proportion calculations, and joins. Alongside standard evaluation measures, we introduce an observation F1 metric that measures how close a query is to the ground truth even when the answers are different. We evaluate a variety of state-of-the-art LLMs and find that they struggle to write SQL that answers the questions correctly. 

---
# See the Answer, Learn to Embed: Learning Universal Multimodal Embeddings from Hindsight 

**Authors**: Yoonki Cho, Youngju Na, Taegeun Yang, Jinhwan Seo, Sung-Eui Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2611.32439)  

**Abstract**: Universal Multimodal Embedding (UME) models map heterogeneous multimodal inputs into a shared embedding space, where pairwise similarity supports diverse downstream tasks. Recent reasoning-driven UME approaches use intermediate reasoning to generate additional context for embedding, but often incur extra computation at inference. In this paper, we explore the matched candidate (i.e., the paired target) as an alternative source of context, asking whether a model can "see the answer" during training and "learn to embed" without it at inference. However, direct conditioning on this privileged context can introduce candidate-specific information unavailable from the query alone or cause the candidate-conditioned query representation to collapse toward the candidate embedding. To address this, we introduce Hindsight-Enhanced Multimodal Embeddings (HEME), which learns query-grounded hindsight representations from such candidate context. HEME first trains a hindsight teacher with a hindsight attention mask, mismatched-candidate conditioning, and query-token alignment to limit excessive reliance on the candidate. It then freezes the teacher and distills its hindsight representation and similarity distribution into a standalone student embedder. At inference, the student independently encodes queries and candidates without candidate conditioning or the hindsight teacher. Experiments on MMEB-V2 and the agent and text tasks of MMEB-V3 demonstrate the effectiveness of HEME and show that it enables more efficient inference than explicit reasoning-driven methods. 

---
# Unjamming Deferrals: Supporting Students Through JAMN Bot, a Mobile RAG-Grounded Policy Assistant 

**Authors**: Mark Farinas, Alesandra Garcia, Maria Reyes, Jethro Nueva, Nawaf Qureshi, Mahmoud Elsaadany, Shokry Shamseldin  

**Link**: [PDF](https://arxiv.org/pdf/2611.32266)  

**Abstract**: In some educational institutions, workflows such as the exam deferral process can be slow, inconvenient, and inconsistent, which creates the need to implement novel digital systems. This paper presents Schedulater, a mobile app for managing university exam deferrals. JAMN Bot is a Large Language Model (LLM)-powered chatbot integrated into Schedulater that provides context-aware support to help users understand the deferral process. The system features a Retrieval-Augmented Generation (RAG) pipeline using Ollama (Llama 3.2 3B) for self-hosted inference which enables policy-grounded responses without sacrificing privacy. We evaluate the system using a 43-item benchmark spanning Exam Deferral, Privacy, Scheduling, Cross-Policy, and Adversarial & Out-of-Scope questions, and report inter-rater agreement among our human evaluators alongside retrieval hit-rate metrics. Subsequently, we utilize an LLM-as-a-judge approach with GPT-5.4, revealing slight-to-moderate alignment with our raters that improves with answer quality, and check the ranking against a panel including Gemini 3.5 Flash and Qwen3.5 397B. The results of our study display trade-offs between open-weight and frontier models, as well as LLM-human evaluation alignment with answer quality. Schedulater demonstrates the feasibility of integrating lightweight LLMs into university administrative workflows with structured retrieval. 

---
# RELAY: Learning Compatible Representations One Encoder at a Time 

**Authors**: Yu Wang, Chuan Qin, Qiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2611.31882)  

**Abstract**: Old and new document embeddings coexist during incremental gallery re-encoding. RELAY trains a new query encoder with a learned projection against the original gallery, then freezes both to train the new document encoder. The same query vector scores both document versions throughout a single re-encoding pass. After backfill, optional query refinement drops the projection and reuses the stored document vectors. Across three text-to-image retrieval upgrades on COCO and Flickr30K, RELAY improves Recall@1 on untouched galleries by 8.66-10.23 percentage points over the original systems. For direct re-encoding, RELAY ranks first in both Qwen-to-Qwen upgrades and second in CLIP-to-Qwen by area under the Recall@1 curve, under both native scoring and matched calibration. With embeddings held fixed, calibrating scores separately for each document version can reduce drops in Recall@1 during backfill and change which method achieves the highest backfill AUC. After query refinement, final Recall@1 is within 0.97 percentage points of conventional joint training in all three upgrades. 

---
# Provenance Loss and Repair in Multilingual Compaction 

**Authors**: Ian Miao, Fangrui Zhu, Bowen Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2611.31668)  

**Abstract**: Conversational assistants compact older messages into short notes and evict the originals. When the note's language requires a distinction that English leaves open, such as Spanish adjective gender (agotado vs. agotada), messages that omit or explicitly state that attribute can yield the same note. After eviction, later turns cannot tell them apart: taking the inflected form literally asserts an attribute the user never gave, whereas discounting it as routine grammar discards one the user did give. We propose a training-free repair across writing, pre-eviction verification and reading. The writer marks unstated attributes with paired forms (e.g., agotado/a), a likelihood check before eviction rewrites unsupported single forms into paired ones, and the reader trusts remaining single forms as stated facts. On five Qwen3.5 and Gemma-4 models, eviction cuts recall of stated attributes from 90% to 34% in question answering (QA); our repair raises F1 (which penalizes both errors) on every model, by 9-53 points in QA and 12-40 points in agent handovers. 

---
# Provably Near-Optimal PQ in the Extended Zador Model 

**Authors**: Edo Liberty, Ashwin Padaki, Amir Ingber  

**Link**: [PDF](https://arxiv.org/pdf/2611.31579)  

**Abstract**: This paper investigates Product Quantization (PQ) Optimization with Gaussian data. It offers three key contributions (i) a proof that SVD is the optimal preconditioning rotation for PQ; (ii) a new Extended Zador (EZ) estimator of the $k$-means quantization cost that is accurate in low bit rates and low dimensions; and (iii) a novel segmentation algorithm based on 1-d bin packing of global bit allocations. The result is a simple algorithm (EZPQ) with strong approximation guarantees. It matches OPQ's empirical accuracy while providing a per-instance distortion lower bound certificate under the Gaussian EZ cost model. On typical real embedding spectra, EZPQ with 2 bits per coordinate is often certifiably within 1% of the optimum in this model. It also consistently achieves a lower reconstruction error than all random-rotation-based algorithms such as EDEN, E-RaBitQ, and TurboQuant. 

---
# ModernELECTRA: Replaced-Token Detection with Boundary-Window Reconstruction for Compact Encoders that Classify and Embed 

**Authors**: Mykhailo Shtopko, Ihor Stepanov, Dmytro Vodianytskyi, Oleksandr Lukashov  

**Link**: [PDF](https://arxiv.org/pdf/2611.31495)  

**Abstract**: ELECTRA-style replaced-token detection (RTD) is a sample-efficient objective for pre-training text encoders, and RTD-trained models such as DeBERTaV3 are strong on classification benchmarks. The same models, however, are known to be poor backbones for embedding tasks such as retrieval and semantic similarity: the binary "original vs. replaced" signal does not ask the final layer to retain the token identity and contextual content that dense representations rely on. We introduce ModernELECTRA, a pre-training recipe that keeps RTD as the main objective and adds a lightweight auxiliary task, boundary-window reconstruction. For every token, we sample a window of 3-12 tokens that strictly contains it and reconstruct the token's original identity only from the discriminator's hidden states at the two window boundaries. The task costs one small FFN at training time and is discarded afterwards, yet it forces every final-layer state to carry information about its neighbourhood. Combined with a frozen, separately pre-trained generator, a temperature curriculum for hard negatives and a 512-to-8192 length curriculum, we train two compact encoders: EdgeCoder-31M, an 8-layer English model, and EdgeCoder-Multilingual-61M, a 16-layer model covering 46 languages, each on about 1T tokens. They reach 86.0 and 85.1 on GLUE. The English model outperforms Ettin-Enc-32m (83.5) and MiniLM-L12 (84.6) on GLUE while matching them on MTEB English v2 (51.1), and the multilingual model outperforms the 2.3x larger mmBERT-small on GLUE and Multilingual MiniLM and mDistilBERT on XTREME over the languages it covers. We also describe FlashDeBERTa, our open-source fused implementation of DeBERTa's disentangled attention that stores no $n \times n$ score matrix: at 4096 tokens it is 10x faster and uses 93% less memory than the reference implementation, and it loads existing DeBERTa checkpoints unchanged. 

---
# A dataset of clinically similar image pairs from citation-linked case reports for medical image retrieval 

**Authors**: Guy Shmueli, Keno K. Bressem, Tom Hope  

**Link**: [PDF](https://arxiv.org/pdf/2611.31465)  

**Abstract**: Retrieving images of patients clinically similar to a given patient can support diagnosis, teaching and research. Training and evaluating retrieval systems requires knowing which images of different patients show the same finding, and existing datasets are small or narrow in scope. We construct PMC-PaIR, a dataset of 86,144 pairs of clinically similar images from 56,013 open-access PubMed Central case reports, which often document rare or challenging presentations. We develop an approach that follows citations, which often link case reports of similar patients. To determine what two linked cases share clinically, it prompts large language models to generate and answer hierarchical questions, from modality to diagnosis and findings, and to match images by caption using the answers. Experts find the pairs highly accurate, and this question-guided approach finds twice as many correct pairs as direct matching. Fine-tuning BiomedCLIP on PMC-PaIR improves retrieval in all modality, anatomy and diagnosis categories tested, for image-only queries and image queries with a short text naming the finding. We release the dataset and the automated pipeline, which can extend PMC-PaIR as new case reports are published. 

---
# Small Learned Projections for Semantic-Reversal-Aware Retrieval with Spectral Acceleration: A Four-Domain Empirical Study 

**Authors**: Sayed Amir Karim  

**Link**: [PDF](https://arxiv.org/pdf/2611.31442)  

**Abstract**: Dense-embedding retrievers can fail adversarially: when a query is answered by a document A and reversed by a near-copy B that differs from A by a small meaning-changing edit (a "trap"), raw cosine over E5-Large-v2 places B in the top-10 for 82-100% of queries at a pool of 5,000. We study two things. First, a small learned projection W: R^1024 -> R^d (d in {11, 32}), trained by a deterministic recipe on a declared set of cases, that suppresses the trap: for the declared cases, HN@10 = 0.0000 (no trap in the top-10) in all twelve (domain, pool-size) settings across four independent domains (legal, medical, patent, financial) and pools up to 100,000, while the correct paraphrase is retained for 93-99% of cases, under exact nearest-neighbour search. Second, a boundary-conditioned accelerator that serves the projection at that scale: an acceleration of spectral graph retrieval (SpectRAG) that replaces a dense eigendecomposition with a boundary-value solve, holding HN@10 = 0.0000 at n = 100,000 in every domain and retrieving the ~300-query suite in 0.6-1.1 s (~2-4 ms per query on average). Because the lens's output is the stored representation (a few floats per document), a hardened, specialised index per domain or tenant is ~90x smaller and dimension-proportionally faster than the full embedding, making many small declared indexes a practical alternative to one large vector store. The claim is scoped to the declared cases on the measured pools, with no probabilistic bound; the correct-answer rate drops for a reworded query against an unchanged corpus (about 0.79, trap suppression still holding). A projection transfers only partly to other domains, which we report in full. The recipe, lenses, trap sets and evaluation harness are released under AGPL-3.0; the accelerator's implementation is a reserved commercial component, and every HN number is reproducible under exact search. 

---
# Reliability of Language-Model Judges for Quantitative Scientific Question Answering: Evidence from Perovskite Photovoltaics 

**Authors**: Qiwei Han, Chi Zhou, Gaoxiang Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2611.31269)  

**Abstract**: Evaluating retrieval-augmented language models increasingly relies on automated model judges, yet their reliability on quantitative scientific synthesis remains largely unmeasured. Here, we evaluate automated grading on a quantitative benchmark for perovskite photovoltaics, where all 755 ground-truth answers are deterministically computed from 45,441 experimental device records. We evaluate a frontier judge (Claude Opus 5.5) alongside two local open-weight judges (Qwen3-14B and Qwen3-8B) across 4,530 free-text answers from six retrieval architectures under three regimes: reference-augmented (with the computed key), evidence-augmented (with retrieved passages and query logs), and unassisted (question and answer alone). Equipped with the reference key, Claude Opus reproduces ground truth near-perfectly (Cohen's $\kappa = 0.982 \pm 0.003$) and preserves exact architectural rankings ($\tau = 1.00$). Withholding the reference causes evaluation fidelity to collapse: frontier agreement drops to $\kappa = 0.557 \pm 0.022$, and providing retrieved literature passages yields virtually no improvement ($\kappa = 0.576 \pm 0.024$). Without reference keys, local judges degrade to near-random agreement ($\kappa = 0.087$ and $0.112$), inverting the leaderboard to rank a hallucinating closed-book baseline first ($\tau \le 0.20$). Furthermore, SQL-augmented generations induce severe evaluative blind spots: absent a reference, the frontier judge accepts 46.6% of erroneous SQL answers, overestimating architecture accuracy as 66.4% against its true 52.6%. These findings demonstrate that model-based evaluation in the physical sciences cannot rely on unstructured text inspection; rigorous benchmarking requires reference ground truth precomputed from structured measurement records. 

---
