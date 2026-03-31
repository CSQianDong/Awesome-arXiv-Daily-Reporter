# KVSculpt: KV Cache Compression as Distillation 

**Authors**: Bo Jiang, Sian Jin  

**Link**: [PDF](https://arxiv.org/pdf/2603.27819)  

**Abstract**: KV cache compression is critical for efficient long-context LLM inference. Approaches that reduce the per-pair footprint -- quantization and low-rank decomposition -- are orthogonal to those that reduce the sequence length of the cache. Along the sequence-length dimension, existing methods range from pure eviction -- selecting which KV pairs to keep -- to merging, which combines similar pairs into fewer ones. Both remain anchored to the original cache entries. We propose KVSculpt, which moves to the other end of this spectrum: instead of selecting or combining original pairs, we optimize a smaller set of unconstrained KV pairs in continuous embedding space to preserve each layer's attention behavior. Keys are optimized via L-BFGS and values are solved in closed form via least squares, alternating every few steps. On top of this, we introduce adaptive budget allocation, which uses a cheap pilot compression run to redistribute the compression budget across layers and KV heads based on per-component difficulty. On Qwen2.5-1.5B-Instruct with 2048-token contexts, KVSculpt reduces KL divergence by 3.5-4.1x compared to Select+Fit -- attention-score eviction with least-squares value fitting -- across compression ratios r in {0.3, 0.5, 0.7}. Adaptive allocation provides an additional 1.3x KL reduction at no extra inference cost. Analysis reveals that compression difficulty is highly non-uniform: per-layer pilot MSE varies by up to 100x across layers, and the two KV heads within a single layer can differ by up to 467x -- demonstrating that fine-grained budget allocation is essential. 

---
# AdaptToken: Entropy-based Adaptive Token Selection for MLLM Long Video Understanding 

**Authors**: Haozhe Qi, Kevin Qu, Mahdi Rad, Rui Wang, Alexander Mathis, Marc Pollefeys  

**Link**: [PDF](https://arxiv.org/pdf/2603.28696)  

**Abstract**: Long video understanding remains challenging for Multi-modal Large Language Models (MLLMs) due to high memory costs and context-length limits. Prior approaches mitigate this by scoring and selecting frames/tokens within short clips, but they lack a principled mechanism to (i) compare relevance across distant video clips and (ii) stop processing once sufficient evidence has been gathered. We propose AdaptToken, a training-free framework that turns an MLLM's self-uncertainty into a global control signal for long-video token selection. AdaptToken splits a video into groups, extracts cross-modal attention to rank tokens within each group, and uses the model's response entropy to estimate each group's prompt relevance. This entropy signal enables a global token budget allocation across groups and further supports early stopping (AdaptToken-Lite), skipping the remaining groups when the model becomes sufficiently certain. Across four long-video benchmarks (VideoMME, LongVideoBench, LVBench, and MLVU) and multiple base MLLMs (7B-72B), AdaptToken consistently improves accuracy (e.g., +6.7 on average over Qwen2.5-VL 7B) and continues to benefit from extremely long inputs (up to 10K frames), while AdaptToken-Lite reduces inference time by about half with comparable performance. Project page: this https URL 

---
# HISA: Efficient Hierarchical Indexing for Fine-Grained Sparse Attention 

**Authors**: Yufei Xu, Fanxu Meng, Fan Jiang, Yuxuan Wang, Ruijie Zhou, Jiexi Wu, Zhixin Pan, Zhaohui Wang, Xiaojuan Tang, Wenjie Pei, Tongxuan Liu, Di yin, Xing Sun, Muhan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.28458)  

**Abstract**: Token-level sparse attention mechanisms, exemplified by DeepSeek Sparse Attention (DSA), achieve fine-grained key selection by scoring every historical token for each query using a lightweight indexer, and then computing attention only over the selected subset. While the downstream sparse attention scales efficiently, the indexer still scans the entire prefix for every query, introducing an O($L^2$) per-layer bottleneck that becomes prohibitive as context length grows. We propose HISA (Hierarchical Indexed Sparse Attention), a drop-in replacement for the indexer that transforms the search process from a flat token scan into a two-stage hierarchical procedure. First, a block-level coarse filter scores pooled block representatives to prune irrelevant regions. Then, a token-level refinement applies the original indexer only within the remaining candidate blocks. HISA preserves the exact token-level top-k sparsity pattern required by the downstream Sparse MLA operator and requires no additional training. On kernel-level benchmarks, HISA achieves a 2$\times$ speedup at 32K context length and 4$\times$ at 128K. On Needle-in-a-Haystack and LongBench, we directly replace the indexer in DeepSeek-V3.2 with HISA, without any fine-tuning. HISA closely matches the original DSA in quality while significantly outperforming block-sparse baselines. Moreover, the token selection sets produced by HISA and the original DSA exhibit a mean IoU greater than 99%, indicating that the efficiency gains come with virtually no impact on selection fidelity. 

---
# Improving Attributed Long-form Question Answering with Intent Awareness 

**Authors**: Xinran Zhao, Aakanksha Naik, Jay DeYoung, Joseph Chee Chang, Jena D. Hwang, Tongshuang Wu, Varsha Kishore  

**Link**: [PDF](https://arxiv.org/pdf/2603.27435)  

**Abstract**: Large language models (LLMs) are increasingly being used to generate comprehensive, knowledge-intensive reports. However, while these models are trained on diverse academic papers and reports, they are not exposed to the reasoning processes and intents that guide authors in crafting these documents. We hypothesize that enhancing a model's intent awareness can significantly improve the quality of generated long-form reports. We develop and employ structured, tag-based schemes to better elicit underlying implicit intents to write or cite. We demonstrate that these extracted intents enhance both zero-shot generation capabilities in LLMs and enable the creation of high-quality synthetic data for fine-tuning smaller models. Our experiments reveal improved performance across various challenging scientific report generation tasks, with an average improvement of +2.9 and +12.3 absolute points for large and small models over baselines, respectively. Furthermore, our analysis illuminates how intent awareness enhances model citation usage and substantially improves report readability. 

---
# M-RAG: Making RAG Faster, Stronger, and More Efficient 

**Authors**: Sun Xu, Tongkai Xu, Baiheng Xie, Li Huang, Qiang Gao, Kunpeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.26667)  

**Abstract**: Retrieval-Augmented Generation (RAG) has become a widely adopted paradigm for enhancing the reliability of large language models (LLMs). However, RAG systems are sensitive to retrieval strategies that rely on text chunking to construct retrieval units, which often introduce information fragmentation, retrieval noise, and reduced efficiency. Recent work has even questioned the necessity of RAG, arguing that long-context LLMs may eliminate multi-stage retrieval pipelines by directly processing full documents. Nevertheless, expanded context capacity alone does not resolve the challenges of relevance filtering, evidence prioritization, and isolating answer-bearing information. To this end, we proposed M-RAG, a novel Chunk-free retrieval strategy. Instead of retrieving coarse-grained textual chunks, M-RAG extracts structured, k-v decomposition meta-markers, with a lightweight, intent-aligned retrieval key for retrieval and a context-rich information value for generation. Under this setting, M-RAG enables efficient and stable query-key similarity matching without sacrificing expressive ability. Experimental results on the LongBench subtasks demonstrate that M-RAG outperforms chunk-based RAG baselines across varying token budgets, particularly under low-resource settings. Extensive analysis further reveals that M-RAG retrieves more answer-friendly evidence with high efficiency, validating the effectiveness of decoupling retrieval representation from generation and highlighting the proposed strategy as a scalable and robust alternative to existing chunk-based methods. 

---
