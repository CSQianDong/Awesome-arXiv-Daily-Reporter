# MemRouter: Memory-as-Embedding Routing for Long-Term Conversational Agents 

**Authors**: Tianyu Hu, Weikai Lin, Weizhi Zhang, Jing Ma, Song Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.00356)  

**Abstract**: Long-term conversational agents must decide which turns to store in external memory, yet recent systems rely on autoregressive LLM generation at every turn to make that decision. We present MemRouter, a write-side memory router that decouples memory admission from the downstream answer backbone and replaces per-turn memory-management decoding with an embedding-based routing policy. MemRouter encodes each turn together with recent context, projects the resulting embeddings through a frozen LLM backbone, and predicts whether the turn should be stored using lightweight classification heads while training only 12M parameters. Under a controlled matched-harness comparison on LoCoMo, where the retrieval pipeline, answer prompts, and QA backbone (Qwen2.5-7B) are held identical, MemRouter outperforms an LLM-based memory manager on every question category (overall F1 52.0 vs 45.6, non-overlapping 95% CIs) while reducing memory-management p50 latency from 970ms to 58ms. Descriptive factorial averaging further shows that learned admission improves mean F1 by +10.3 over random storage, category-specific prompting adds +5.2 over a generic prompt, and retrieval contributes +0.7. These results suggest that write-side memory admission can be learned by a small supervised router, while answer generation remains a separate downstream component in long-horizon conversational QA. 

---
# NorBERTo: A ModernBERT Model Trained for Portuguese with 331 Billion Tokens Corpus 

**Authors**: Enzo S. N. Silva, Pablo B. Costa, Raphael C. Vlasman, Rosimeire P. Costa, Henrique L. P. Silva, Lucas F. A. O. Pellicer, Guilherme Rinaldo, Renato A. Almeida, Darian S. R. Rabbani, Cinthya O. Oestreich, Vinicius F. Caridá  

**Link**: [PDF](https://arxiv.org/pdf/2605.00086)  

**Abstract**: High-quality corpora are essential for advancing Natural Language Processing (NLP) in Portuguese. Building on previous encoder-only models such as BERTimbau and Albertina PT-BR, we introduce NorBERTo, a modern encoder based on the ModernBERT architecture, featuring long-context support and efficient attention mechanisms. NorBERTo is trained on Aurora-PT, a newly curated Brazilian Portuguese corpus comprising 331 billion GPT-2 tokens collected from diverse web sources and existing multilingual datasets. We systematically benchmark NorBERTo against Strong baselines on semantic similarity, textual entailment and classification tasks using standardized datasets such as ASSIN 2 and PLUE. On PLUE, NorBERTo-large achieves the best results among the encoder models we evaluated, notably reaching 0.9191 F1 on MRPC and 0.7689 accuracy on RTE. On ASSIN 2, NorBERTo-large attains the highest entailment F1 (~0.904) among all encoders considered, although Albertina-900M and BERTimbau-large still hold an advantage. To the best of our knowledge, Aurora-PT is currently the largest openly available monolingual Portuguese corpus, surpassing previous resources. NorBERTo provides a modern, mid-sized encoder designed for realistic deployment scenarios: it is straight-forward to fine-tune, efficient to serve, and well suited as a backbone for retrieval-augmented generation and other downstream Portuguese NLP systems. 

---
# Persistent Visual Memory: Sustaining Perception for Deep Generation in LVLMs 

**Authors**: Siyuan Huang, Xiaoye Qu, Yafu Li, Tong Zhu, Zefeng He, Muxin Fu, Daizong Liu, Wei-Long Zheng, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2605.00814)  

**Abstract**: While autoregressive Large Vision-Language Models (LVLMs) demonstrate remarkable proficiency in multimodal tasks, they face a "Visual Signal Dilution" phenomenon, where the accumulation of textual history expands the attention partition function, causing visual attention to decay inversely with generated sequence length. To counteract this, we propose Persistent Visual Memory (PVM), a lightweight learnable module designed to ensure sustained, on-demand visual perception. Integrated as a parallel branch alongside the Feed-Forward Network (FFN) in LVLMs, PVM establishes a distance-agnostic retrieval pathway that directly provides visual embeddings for precise visual perception, thereby structurally mitigating the signal suppression inherent to deep generation. Extensive experiments on Qwen3-VL models demonstrate that PVM brings notable improvements with negligible parameter overhead, delivering consistent average accuracy gains across both 4B and 8B scales, particularly in complex reasoning tasks that demand persistent visual perception. Furthermore, in-depth analysis reveals that PVM can resist length-induced signal decay and accelerate internal prediction convergence. 

---
# Hierarchical Abstract Tree for Cross-Document Retrieval-Augmented Generation 

**Authors**: Ziwen Zhao, Menglin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.00529)  

**Abstract**: Retrieval-augmented generation (RAG) enhances large language models with external knowledge, and tree-based RAG organizes documents into hierarchical indexes to support queries at multiple granularities. However, existing Tree-RAG methods designed for single-document retrieval face critical challenges in scaling to cross-document multi-hop questions: (1) poor distribution adaptability, where $k$-means clustering introduces noise due to rigid distribution assumptions; (2) structural isolation, as tree indexes lack explicit cross-document connections; and (3) coarse abstraction, which obscures fine-grained details. To address these limitations, we propose $\Psi$-RAG, a tree-RAG framework with two key components. First, a hierarchical abstract tree index built through an iterative "merging and collapse" process that adapts to data distributions without a priori assumption. Second, a multi-granular retrieval agent that intelligently interacts with the knowledge base with reorganized queries and an agent-powered hybrid retriever. $\Psi$-RAG supports diverse tasks from token-level question answering to document-level summarization. On cross-document multi-hop QA benchmarks, it outperforms RAPTOR by 25.9% and HippoRAG 2 by 7.4% in average F1 score. Code is available at this https URL. 

---
# Budget-Aware Routing for Long Clinical Text 

**Authors**: Khizar Qureshi, Geoffrey Martin, Yifan Peng  

**Link**: [PDF](https://arxiv.org/pdf/2605.00336)  

**Abstract**: A key challenge for large language models is token cost per query and overall deployment cost. Clinical inputs are long, heterogeneous, and often redundant, while downstream tasks are short and high stakes. We study budgeted context selection, where a subset of document units is chosen under a strict token budget so an off-the-shelf generator can meet fixed cost and latency constraints. We cast this as a knapsack-constrained subset selection problem with two design choices, unitization that defines document segmentation and selection that determines which units are kept.
We propose \textbf{RCD}, a monotone submodular objective that balances relevance, coverage, and diversity. We compare sentence, section, window, and cluster-based unitization, and introduce a routing heuristic that adapts to the budget regime. Experiments on MIMIC discharge notes, Cochrane abstracts, and L-Eval show that optimal strategies depend on the evaluation setting. Positional heuristics perform best at low budgets in extractive tasks, while diversity-aware methods such as MMR improve LLM generation. Selector choice matters more than unitization, with cluster-based grouping reducing performance and other schemes behaving similarly. ROUGE saturates for LLM summaries, while BERTScore better reflects quality differences. We release our code at this https URL. 

---
# Caracal: Causal Architecture via Spectral Mixing 

**Authors**: Bingzheng Gan, Tianyi Zhang, Yusu Li, Jing Huang, Wei Shi, Yangkai Ding, Tao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.00292)  

**Abstract**: The scalability of Large Language Models to long sequences is hindered by the quadratic cost of attention and the limitations of positional encodings. To address these, we introduce Caracal, a novel architecture that replaces attention with a parameter-efficient, $\mathcal{O}(L \log L)$ Multi-Head Fourier (MHF) module. Our contributions are threefold: (1) We leverage the Fast Fourier Transform (FFT) for sequence mixing, inherently addressing both bottlenecks mentioned above. (2) We apply a frequency-domain causal masking technique that enforces autoregressive capabilities via asymmetric padding and truncation, overcoming a critical barrier for Fourier-based generative models. (3) Unlike efficient models relying on hardware-specific implementations (e.g., Mamba), we uses standard library operators. This ensures robust portability, eliminating common deployment barriers. Evaluations demonstrate that Caracal performs competitively with Transformer and SSM baselines, offering a scalable and simple pathway for efficient long-sequence modeling. Code is available in Appendix. 

---
