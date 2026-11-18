# CARE: Conversational Affective Response Retrieval for Internet Memes 

**Authors**: Jiajun Li, Leihan Zhang, Teng Zhang, Le Zhang, Qiang Yan  

**Link**: [PDF](https://arxiv.org/pdf/2611.23005)  

**Abstract**: Conversational meme retrieval supports response selection in online communication, where relevance depends on both contextual meaning and affective expression. We study this task using two language-specific collections from Douban and Imgflip, containing 98,506 meme records across 5,583 discussion topics. Thread membership supplies weak relevance labels, while likes provide implicit ranking feedback. Automatically extracted sentiment, facial-expression indicators, and cross-modal consistency features support an exploratory analysis of engagement. We propose CARE, a Conversational Affective Response Retrieval framework that combines an emotion-enhanced CLIP retriever with a Jina-based reranker. The retriever incorporates free-form emotion descriptors through gated fusion and learns from multiple positive examples per query. The reranker combines query--image interactions with emotion representations and is trained using engagement-based ranking supervision. Under the reported topic-held-out protocol, CARE achieves Hit@10 of 24.03\% on Douban and 49.33\% on Imgflip, compared with 23.16\% and 41.07\%, respectively, for fine-tuned CLIP. These results support the utility of the complete two-stage system for recovering thread-associated memes. They do not independently establish perceived social resonance or isolate the contribution of affect from reranker training. The study provides a basis for further evaluation of affect-aware retrieval with human relevance judgments and controlled comparisons. 

---
# Music Similarity Learning in Sparse Latin Scenarios 

**Authors**: Gabrielly Xavier dos Santos, Lucas Dias Cler, Olga Goussevskaia  

**Link**: [PDF](https://arxiv.org/pdf/2611.22846)  

**Abstract**: In this work we propose a lightweight, interpretable approach that estimates item-item music similarity from publicly available metadata. Our hybrid regression model combines content-based filtering with the classic item-based collaborative filtering. We test generalization via a zero-shot evaluation on Latin American music, a genre for which co-occurrence data is scarce. Accurately predicting similarity for these unseen, sparse pairs shows that our approach is robust to the regional gap affecting many existing systems, without the audio signal processing or the computational budget that deep multimodal approaches would require. 

---
# Better Rankers, Worse Retrievers: The Hidden Cost of RL Post-Training in Generative Recommendation 

**Authors**: Yuheng Chen, Fan Zhang, Yixuan Xu, Chao Yang, Yang Xiang, Can Yang  

**Link**: [PDF](https://arxiv.org/pdf/2611.22838)  

**Abstract**: Generative recommenders decode the semantic ID of a user's next item with a language model, and post-training them with reinforcement learning (RL) after supervised fine-tuning (SFT) has become standard practice. The gains are mostly reported on the first ten results, yet in deployment such models hand hundreds of candidates to a downstream ranker or reward model. We show that RL post-training has a hidden cost: it improves accuracy at the top of the list (hit@K for small K) but, on all five datasets we study, lowers the recall of the candidate list (hit@K for K in the hundreds). The model becomes a better ranker and a worse retriever. Controlled experiments trace the loss to the negative advantages that RL assigns to the candidates it samples, the rollout. The root cause is that only one next item is logged per user, so RL treats every other candidate as wrong. Yet these candidates are the model's most plausible alternatives, and often the next items of other users with similar histories. Penalising them pushes down exactly what similar users choose next, and these items drop out of those users' candidate lists. This diagnosis points directly to a remedy: out-of-rollout anchoring, a single additional loss term that preserves the SFT model's ordering of the items outside the rollout. It keeps the gain at the top and restores recall to the SFT level on all five datasets, and it is effective under other RL configurations as well. To our knowledge, this is the first systematic study of this recall loss: we measure it, identify its cause, and show that, once the cause is known, a simple remedy suffices. 

---
# IntentGR: Enhancing Intent for End-to-End Personalized Generative Retrieval in Search 

**Authors**: Gefan Ye, Yanjing Wu, Yinfu Feng, Zhiqiang Wang, Liang Li, Jun Xiao, Rong Xiao, Xiaoyi Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2611.22788)  

**Abstract**: Personalized generative retrieval in search should jointly satisfy the user's immediate search intent and long-term preference. Recent end-to-end approaches, primarily explored in recommendation, unify item tokenization and sequence generation within a single model, allowing semantic identifiers (SIDs) to be optimized toward the downstream generation objective. However, these methods often treat the query merely as an input feature, without explicitly leveraging it to guide either identifier learning or sequence generation. This leaves intent guidance weak in both the tokenization and the generation stage, causing generated semantic identifiers to drift toward high-frequency token patterns rather than the current search intent. In this work, we find that incorporating additional query-related evidence into the framework of end-to-end generative retrieval is important. Based on this finding, we propose IntentGR, an intent-enhanced end-to-end framework that strengthens query semantics at four complementary stages of the retrieval pipeline. Query-Aware Semantic Behavior Representation (QBR) introduces query-level behavioral evidence as a new input signal; Query-Semantic Alignment (QSA), Long-Tail Query Enrichment (LQE), and Query-conditioned Attention Modulation (QAM) then serve as integration mechanisms at the optimization, training-distribution, and decoding stages to fully exploit this evidence. Experiments on the public AOL4PS benchmark and AE, a large-scale industrial e-commerce dataset, show that IntentGR improves Recall@10 by 8.11% and 11.96% over the strongest baseline, with strong gains on long-tail queries. These results suggest that enriched query-related input evidence, together with effective integration mechanisms, is beneficial for building intent-aware end-to-end generative search systems. 

---
# Learning Cross-Domain Media Similarity from Amazon Reviews 

**Authors**: Gabrielly Xavier dos Santos, Lucas Dias Cler, Luciana Fujii Pontello, Olga Goussevskaia  

**Link**: [PDF](https://arxiv.org/pdf/2611.22756)  

**Abstract**: Collaborative filtering is a basic foundation of many recommendation algorithms. It works by analyzing a user's history and then suggesting items similar to those in that history. This method typically relies on user data to compute item similarity, data that is widely available to commercial systems but is usually sparse or inaccessible for independent systems and research, especially across multiple media domains. In this work, we propose a method to generalize similarity information between items using supervised learning. As input to our model, we use semantic and temporal features extracted from Amazon product reviews, which are publicly available. As the training target, we use item (product) co-occurrence in the reviewing history of Amazon customers. We evaluate the quality of the predicted similarities across two media domains: books and movies. Experimental results demonstrate the robustness of our methodology, yielding positive outcomes in zero-shot cross-domain transfer tasks. Furthermore, feature attribution analysis reveals a structural shift in how feature groups are utilized across setups: intra-domain models rely heavily on linguistic semantics, whereas cross-domain environments pivot toward behavioral popularity cues. Incorporating even limited target-domain context during training bridges this domain sparsity gap, significantly enhancing framework generalization. 

---
# MIPR: A Benchmark for Reasoning-Intensive Multimodal Intent-to-Product Retrieval with Multi-Category Solutions 

**Authors**: Qiang Li, Xuan Lu, Jian Ruan, Manxi Lin, Jiacheng Chen, Wen Shen, Zhihua Wei, Xiaoyu Shen, Dan Ou, Xiangheng Kong, Junjun Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2611.22509)  

**Abstract**: Real-world e-commerce queries often express user intent without naming a target product, and a single intent may be fulfilled by products from multiple categories. However, existing multimodal retrieval benchmarks typically assume a relatively well-defined target, leaving retrieval for intents with multi-category solutions less explored. To fill this gap, we introduce MIPR, a benchmark for Multimodal Intent-to-Product Retrieval comprising 3,522 multimodal queries and a shared corpus of 30,731 products. MIPR features reasoning-intensive retrieval, multi-category solutions, and fine-grained relevance labels. Experiments show that the best embedding-based retriever achieves 49.23% nDCG@10 on the full query set, and 55.86% solution-category recall@10 on queries with multiple solution categories. Reranking improves these scores to 60.11% and 67.00%, respectively, yet substantial gaps to the candidate-pool oracle remain. These results establish MIPR as a challenging and diagnostic benchmark for evaluating multimodal intent-to-product retrieval. The data and code are publicly available at this https URL. 

---
# From Relevance to Utility: Rubric Alignment in Yuanbao Search 

**Authors**: Yu Lu, Lixin Zou, Yangqi Chen, Hui Xu, Qiaolin Xia, Hao Wang, Qianwen Ma, Reed Li, Shuang Zeng, Haijin Liang, Haibo Shi  

**Link**: [PDF](https://arxiv.org/pdf/2611.22503)  

**Abstract**: Agentic search demands that a search engine move beyond per document relevance ranking to assemble document sets optimized for downstream LLM reasoning. Prior work addresses this either by improving individual pipeline stages in isolation or by applying RL based selection on top of relevance trained upstream modules, leaving the full stack unaligned to utility for reasoning. We propose rubric aligned search, a framework that unifies the traditional search pipeline under a single supervision surface. For each query, we derive a rubric from coarse to fine, specifying weighted positive criteria that the document set must satisfy and negative criteria that it must avoid. From this rubric we derive two training signals. First, a lightweight labeler distills rubric grounded judgments into per document utility labels, which supervise a contrastive dual encoder for retrieval and a multi task ranker trained with four rubric aligned scoring heads. Second, a rubric conditioned evaluator produces a set level reward that drives a dual branch RL selector, combining GRPO with best rollout distillation to resolve credit assignment in subset optimization. Deployed in Yuanbao Search, the pipeline improves end to end answer quality by 8.52% over the online production baseline. 

---
# More Than a Score: Reinforcement Learning with Verbalized Feedback for Generative Recommendation 

**Authors**: Zipeng Chen, Xinyu Lin, Yuliang Sun, Sheng Zhang, Zeyu Wang, Xiaowen Shi, Junwei Pan, Chengguo Yin  

**Link**: [PDF](https://arxiv.org/pdf/2611.22410)  

**Abstract**: Large language models (LLMs) are increasingly used for generative recommendation, where recommended items are generated directly from users' interaction histories. To further improve recommendation quality, reinforcement learning (RL) has been applied to align these models with recommendation objectives. However, existing RL methods rely on scalar rewards to implicitly guide exploration. These rewards do not specify which aspects of a candidate conflict with the user's preferences or how the next candidate should differ. To address this gap, we propose VerRec, a verbalized feedback-guided RL framework for generative recommendation. Specifically, a user simulator generates personalized verbalized feedback based on the user's historical preferences and the candidate item profile. The policy uses this feedback to produce refined recommendations. Original and refined candidates form an expanded group and are jointly optimized with group relative policy optimization. The user simulator is only required during training and is removed at inference time. Experiments on three Amazon review datasets with two generative recommendation backbones demonstrate that VerRec improves recommendation accuracy and ranking quality. Further analyses show that verbalized guidance encourages exploration of more diverse candidates and helps the policy discover more rewarding recommendations during training. 

---
# Deal With It: On Better Schedules for Listwise Reranking 

**Authors**: Benjamin Clavié, Herumb Shandilya, Hanna Lichtenberg, Aamir Shakir, Makoto P. Kato  

**Link**: [PDF](https://arxiv.org/pdf/2611.22392)  

**Abstract**: Listwise LLM rerankers cannot reliably rank hundreds of documents in a single call, and rely on schedules to orchestrate their calls. Parallel schedules split candidates into groups and advance a fixed quota from each. When a group holds more of the top passages than its quota, some are cut, and expensive repair calls become required. We show that the way candidates are split matters where such a quota can bind, and that, in expected cuts, the best split there deals them round-robin by retriever rank, like cards. Although not conceptually new, dealing has remained unanalyzed. We prove that, under a simple retriever model, no split with equal quotas cuts fewer of the true top ten on average, and that contiguous chunks cut more as the retriever improves. DealRank, the simplest schedule our analysis allows, is on average within 0.1 nDCG@10 points of the best schedules over six LLMs, in two rounds. It is consistently 1.1 to 1.4 times faster than BlitzRank with concurrent first calls up to a thousand candidates. There, it needs three rounds and is eleven times faster than the sliding window. In existing schedules such as BlitzRank, dealing reduces repair calls from 59% of queries to 14%, improving latency with no significant change in quality. Random partitioning appears equivalent at a quota of four, as predicted, but ties results to the seed and trails by 1.3 nDCG@10 points at a quota of one over a dense retriever. 

---
# NONSENSE: Non-Semantic Strings Expose Semanticity Gaps in Dense Retrieval 

**Authors**: Ankit Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2611.22353)  

**Abstract**: Dense retrieval is central to modern search, question answering, and retrieval-augmented generation, where high similarity is often treated as evidence of semantic relevance. We examine how far this assumption can be stressed. NONSENSE is a black-box sparse-edit probe that constructs visibly non-semantic strings while forbidding query-term copying, optimizing only the dense similarity between a query and the string. Across five BEIR-style datasets and three popular single-vector retrievers, these strings frequently outscore the retriever's own best judged query-relevant passage (QRP), and often rank first in the full corpus. The result indicates a semanticity gap in the evaluated dense retrievers: nearest-neighbor status can be separated from passage-like meaning. These findings highlight the importance of evaluating dense retrievers not only by whether they retrieve relevant documents, but also by how reliably their similarity scores reflect semantic evidence. 

---
# GuardAgent: A Memory-Augmented Agentic Framework for Adaptive Detection of Adversarial Promotions in LLM-Based Recommendation 

**Authors**: Shiyi Yang, Xinshu Li, Chen Wang, Xiwei Xu, Liming Zhu, Lina Yao  

**Link**: [PDF](https://arxiv.org/pdf/2611.22274)  

**Abstract**: In traditional recommender systems (RSs), textual information typically serves as auxiliary signals for representation learning. In contrast, in LLM-based RSs, textual information becomes the primary medium for recommendation generation through semantic understanding and reasoning. This shift fundamentally changes the attack surface, making promotion-driven manipulations more feasible, effective, and harder to detect. However, effectively detecting such attacks remains largely unexplored. To fill this gap, we present the first systematic study on detecting promotion-driven attacks against LLM-based RSs, aiming to safeguard next-generation recommendation platforms. Specifically, we propose GuardAgent, a novel LLM-powered agentic detection framework that operates through a loop of Perception, Action, Reflection and Memory. In contrast to conventional LLM-based detectors that function in a static, one-off, and opaque fashion, GuardAgent continually learns and self-evolves to adapt to unseen threat patterns while generating explainable reasoning traces and outcomes. Extensive experiments on real-world datasets demonstrate that GuardAgent achieves state-of-the-art generalization performance, laying the foundation for robust and trustworthy recommendation in practical settings. 

---
# Teaching Language Models to Narrow Boolean Patent Searches from Search Feedback 

**Authors**: Seiya Kawano, Koji Marusaki, Hirofumi Nonaka, Koichiro Yoshino  

**Link**: [PDF](https://arxiv.org/pdf/2611.22270)  

**Abstract**: Patent prior-art searchers issue Boolean queries line by line, broadening to find relevant documents and narrowing until candidates can be screened. We distill 38,397 Japan Patent Office search reports into executable expert strategies, re-execute them, and evaluate generated strategies by citation recall and the smallest set containing each citation. Prompted models write valid, high-recall strategies but leave citations in sets too large to screen. We adapt a generator offline from scored rollouts in two forms: strategy-level learning from a screening-aware strategy score, and line-level learning from citation-preserving narrowing credit, which upweights the query lines keeping a found citation in a smaller set. On shared rollouts, line-level credit outperforms strategy-level scores on screening measures; iterated, it cuts the median citation-containing set from 768 to under 50 and places more citations in sets under 100 documents than experts, at lower recall. Strategy-level preference learning reaches near-complete recall; the two forms are complementary. 

---
# Cluster-Guided Decoding with Trust-Region Constraints for Personalized Recommendation 

**Authors**: Gezheng Wen, Chao Shen, Raviteja Anantha, Tan Shen Cliff Chung, Layne C. Price  

**Link**: [PDF](https://arxiv.org/pdf/2611.22163)  

**Abstract**: Most recommenders score each item in the catalog separately, and do not explicitly model which semantic region of the catalog a recommendation belongs to. Models expressive enough to offer that control are often too slow to run over a full catalog for online inference. In this paper, we present Cluster-Guided Generative Reconstruction (CG2R), which learns a hierarchy of semantic clusters and uses a cluster-level prior to steer item ranking for personalized recommendations. Unlike existing systems that only use clusters to pre-select candidates, CG2R adds the cluster prior directly to the item scoring rule, so that the learned cluster structure shapes the ranking itself. Because the prior is assembled from cluster centroids rather than per-item statistics, it remains defined even for cold-start users when no history is available. Across the three benchmark datasets evaluated, CG2R outperforms classical, neural, and LLM-based baselines in terms of Recall and nDCG, with its performance gains widening as the catalog grows sparser. In addition, CG2R serves recommendations in milliseconds, suggesting potential as low-latency first-stage retrieval. 

---
# SIMQ: A Single Graph-based Index for Multi-Vector Query 

**Authors**: Yifan Lu, Jiabao Jin, Xiaoyao Zhong, Peng Cheng, Xiangyu Wang, Haotian Li, Hua Chen, Junqing Chen, Wei Xi, Heng Tao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2611.22079)  

**Abstract**: Multi-vector retrieval models such as ColBERT achieve state-of-the-art effectiveness by representing documents as sets of token-level vectors and scoring via MaxSim, but their high computational cost limits scalability to large corpora. Existing approximate methods face fundamental bottlenecks: centroid interaction matrix approaches require computing a full matrix at every query and rely on GPU-accelerated clustering, while data-oblivious compression methods yield unstable recall that requires repeated compensation. We propose SIMQ (Single graph Index for Multi-vector Query), a framework that bridges single-vector graph indices and multi-vector retrieval through data-aware sampling-based clustering and per-token graph navigation, simultaneously bypassing all aforementioned bottlenecks. SIMQ organizes all token vectors into fine-grained clusters via data-aware sampling, builds a single graph index over cluster centroids independently of PLAID's residual representation, and approximates the MaxSim score through per-token graph search without materializing any interaction matrix or relying on GPU acceleration. We provide theoretical analysis showing that, under a manifold assumption and successful cell retrieval, data-aware partitioning has a smaller expected approximation error than a data-oblivious partition. Building on this foundation, SIMQ+ further introduces count-based scoring and a dynamic candidate set to improve robustness across varying query conditions. Experiments on five benchmarks compare recall, throughput, and construction time against several baselines, primarily on CPU; large-scale PLAID k-means centroids are built on a separate GPU machine. SIMQ has been integrated into VSAG, an Ant Group's open-source vector search library. 

---
# Beyond Bitwidth: Deployment-Constrained Mixed-Precision Quantization for Multilingual LLM Inference on NVIDIA Blackwell 

**Authors**: Hyojung Han  

**Link**: [PDF](https://arxiv.org/pdf/2611.21974)  

**Abstract**: Low-bit LLM quantization is useful only if a model preserves service-relevant accuracy, executes through supported GPU kernels, and meets measured memory and latency limits. We study Qwen3-8B on NVIDIA B200 with vLLM, including INT2/INT3, GPTQ, NVFP4, and mixed-precision configurations. We verify native packed execution and precision on all 144 fused linear modules and evaluate Korean, English, numerical, and reasoning quality under preregistered gates. Packed INT2 and INT3 execute, but all seven tested INT2 grids degenerate on generation. Isolated module-restoration effects are strongly non-additive; Korean and reasoning sensitivity rankings disagree, and a promising INT4/INT8 development candidate fails independent final evaluation. A configuration with 16-bit activations, 98 NVFP4 weight-only modules, 46 INT8 modules, and an INT8 output head passes all gates on the untouched FINAL-81 set (5,168 items): 5.917 GiB of weight-loading memory (limit 5.98), Korean teacher-correct retention 0.904 (gate 0.90), and TPOT ratio 0.939 relative to BF16. It improves paired accuracy over NVFP4 W4A4 by +1.4 percentage points (Bonferroni-adjusted interval [+0.2, +2.6]), compared with +3.4 points on the selection set. This comparison changes both activation precision and INT8 protection; exploratory analyses suggest that the activation change contributes substantially, but the isolated allocation effect is not established. Final checkpoints were not retained, and bitwise rebuild reproducibility remains unverified. 

---
# When the tail is earned by its companions: measuring label dependence in long-tailed multi-label classification 

**Authors**: Christoffer Löffler  

**Link**: [PDF](https://arxiv.org/pdf/2611.22859)  

**Abstract**: A rare label seldom occurs alone, and a classifier can recover it from the frequent labels it co-occurs with, its companions, without having learned the label itself. Existing diagnostics of this need a trained model: none says before training whether a corpus admits the failure, or what the standard aggregate hides. Macro-F1, like any measure built from per-label confusion counts, is unchanged by how a label's recall divides between occurrences with and without a companion. We introduce a within-label estimand of that division, a reading rule under which a fall in dependence counts only if companion-free recall rises, and a statistic of training labels that bounds, before training, how far reported recall can exceed companion-free recall. Across 23 corpora, 7 training objectives and five families of label-correlation mechanisms, companion-free recall falls short on all six corpora where the estimand is defined. Up to 18.4% of the tail true positives macro-F1 credits would vanish at the companion-free rate, within the training-label bound on every corpus. No long-tail objective reduces the dependence, but architecture moves it. Models whose tail score is computed from other labels' predictions are strongly dependent. 

---
# RICoT: Recurrent Interleaved Chain-of-Thought for Page Stream Segmentation 

**Authors**: Prashant Khatri, Bhavana Malagavalli, Saiful Haq, Niyati Chhaya  

**Link**: [PDF](https://arxiv.org/pdf/2611.22382)  

**Abstract**: Large language models (LLMs) have renewed interest in Page Stream Segmentation (PSS), splitting a multi-page file into its constituent documents. Yet published LLM methods still frame the task as boundary classification over one or a few neighboring pages, and commercial document-splitting APIs do not disclose their methods. We reimagine PSS for LLMs as a sequential process over the whole stream and propose RICoT (Recurrent Interleaved Chain-of-Thought), a prompting framework that works through the pages as a reader would: within a sliding window it reasons about each page and commits its boundary decision before moving on, and across windows it carries forward a compact record of what it has read, keeping per-call cost bounded. We also introduce FinancePSS-205, a benchmark of 205 real-world, de-identified financial page streams, to be released publicly. On FinancePSS-205, RICoT matches or outperforms pairwise and windowed prompting with the same backbone for each of three open-weight LLMs; on a TABME test subset, RICoT with GLM-5 reaches document F1 0.847 and boundary F1 0.915, above published zero-shot results. A 2B-parameter student, trained on a larger teacher's decisions with and without the page-level reasoning, performs comparably in both modes and comes within 0.05 document F1 of the strongest prompted model on FinancePSS-205. 

---
# Complexity and precision in quantum associative retrieval 

**Authors**: Carlo A. Trugenberger  

**Link**: [PDF](https://arxiv.org/pdf/2611.21941)  

**Abstract**: We show that efficient and high-precision quantum associative retrieval is possible with an exponential complexity advantage with respect to the classical brute force approach and with a polynomial capacity advantage with respect to classical neural network implementations for any number of patterns scaling polynomially in the number of qubits. Both classical neural and quantum associative memories store the patterns in a Hamiltonian; in the classical case the dynamics is irreversible relaxation toward attractors, in the quantum case it is unitary transformations generating an effective thermal distribution, followed by a measurement. 

---
# Urban Representation Learning for Identifying Potential Disparities in Crime Reporting 

**Authors**: Jorge Valverde-Rebaza, Carlos Paez De La Cruz, Pedro Pérez-Collado, Daniel Sánchez-Cataño, Eashley Martínez-Vergara  

**Link**: [PDF](https://arxiv.org/pdf/2611.21825)  

**Abstract**: Open crime data support urban safety analysis but reflect reported incidents rather than the complete distribution of criminal activity, potentially introducing spatial distortions in downstream analysis. We propose a self-supervised urban representation learning framework for identifying unusual reporting patterns relative to representation-similar peers using only georeferenced crime records. The method constructs complementary hourly, weekly, and static reporting profiles, learns scale-specific representations under geographic self-supervision, and integrates them into a unified latent space. Neighbor Profile Divergence (NPD) is then used to quantify disagreement between each region's reporting profile and those of its retrieved peers. Experiments on more than 1.8 million crime records from Mexico City, aggregated into 1,058 H3 regions, show that the proposed representation achieves a mean Silhouette score of 0.4052, compared with 0.2863 for Hex2Vec. NPD rankings remain highly stable across neighborhood sizes, with Spearman correlations from 0.9720 to 0.9951, while most high-divergence candidates persist across configurations. The retrieved peers also extend substantially beyond immediate geographic neighborhoods, indicating that the learned representation captures structure beyond local spatial proximity. High-divergence regions exhibit spatially organized patterns and, in several cases, differences in incident volume relative to peer expectations. These results support NPD as an internal auditing and prioritization mechanism for identifying regions that warrant further investigation, rather than as direct evidence of underreporting. 

---
# Evaluating AST-Derived Knowledge Graph Retrieval for AIOps Incident Diagnosis in Apache Kafka 

**Authors**: Shashank Mishra, Shweta Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2611.21625)  

**Abstract**: Incident diagnosis requires retrieving source-linked operational evidence from large codebases. We evaluate graph-augmented retrieval for Apache Kafka using an abstract-syntax-tree (AST)-derived ownership graph over 1,600 source-linked log records and an audited benchmark of 315 incident-style queries covering 103 unique targets. Vector-only retrieval is a strong baseline at Recall@20 of 0.956, and neither static nor confidence-guided staged AST expansion significantly improves Recall@20 over it. As an external control we evaluate HippoRAG, a published text-derived graph-retrieval system, against its own ColBERTv2 retrieval backbone. Against ColBERTv2-only (Recall@20 0.803), HippoRAG attained 0.308; a post-hoc document-ensemble sensitivity raised Recall@20 to 0.648 but did not eliminate the deficit. Relative to static expansion, staging reduces cross-encoder pair workload and measured reranking time by about 52%. A manually evaluated 100-query downstream study finds no significant root-cause-correctness advantage for AST-derived structural evidence. 

---
