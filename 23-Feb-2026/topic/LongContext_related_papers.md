# HyTRec: A Hybrid Temporal-Aware Attention Architecture for Long Behavior Sequential Recommendation 

**Authors**: Lei Xin, Yuhao Zheng, Ke Cheng, Changjiang Jiang, Zifan Zhang, Fanhu Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2602.18283)  

**Abstract**: Modeling long sequences of user behaviors has emerged as a critical frontier in generative recommendation. However, existing solutions face a dilemma: linear attention mechanisms achieve efficiency at the cost of retrieval precision due to limited state capacity, while softmax attention suffers from prohibitive computational overhead. To address this challenge, we propose HyTRec, a model featuring a Hybrid Attention architecture that explicitly decouples long-term stable preferences from short-term intent spikes. By assigning massive historical sequences to a linear attention branch and reserving a specialized softmax attention branch for recent interactions, our approach restores precise retrieval capabilities within industrial-scale contexts involving ten thousand interactions. To mitigate the lag in capturing rapid interest drifts within the linear layers, we furthermore design Temporal-Aware Delta Network (TADN) to dynamically upweight fresh behavioral signals while effectively suppressing historical noise. Empirical results on industrial-scale datasets confirm the superiority that our model maintains linear inference speed and outperforms strong baselines, notably delivering over 8% improvement in Hit Rate for users with ultra-long sequences with great efficiency. 

---
# Decomposing Retrieval Failures in RAG for Long-Document Financial Question Answering 

**Authors**: Amine Kobeissi, Philippe Langlais  

**Link**: [PDF](https://arxiv.org/pdf/2602.17981)  

**Abstract**: Retrieval-augmented generation is increasingly used for financial question answering over long regulatory filings, yet reliability depends on retrieving the exact context needed to justify answers in high stakes settings. We study a frequent failure mode in which the correct document is retrieved but the page or chunk that contains the answer is missed, leading the generator to extrapolate from incomplete context. Despite its practical significance, this within-document retrieval failure mode has received limited systematic attention in the Financial Question Answering (QA) literature. We evaluate retrieval at multiple levels of granularity, document, page, and chunk level, and introduce an oracle based analysis to provide empirical upper bounds on retrieval and generative performance. On a 150 question subset of FinanceBench, we reproduce and compare diverse retrieval strategies including dense, sparse, hybrid, and hierarchical methods with reranking and query reformulation. Across methods, gains in document discovery tend to translate into stronger page recall, yet oracle performance still suggests headroom for page and chunk level retrieval. To target this gap, we introduce a domain fine-tuned page scorer that treats pages as an intermediate retrieval unit between documents and chunks. Unlike prior passage-based hierarchical retrieval, we fine-tune a bi-encoder specifically for page-level relevance on financial filings, exploiting the semantic coherence of pages. Overall, our results demonstrate a significant improvement in page recall and chunk retrieval. 

---
# GeneZip: Region-Aware Compression for Long Context DNA Modeling 

**Authors**: Jianan Zhao, Xixian Liu, Zhihao Zhan, Xinyu Yuan, Hongyu Guo, Jian Tang  

**Link**: [PDF](https://arxiv.org/pdf/2602.17739)  

**Abstract**: Genomic sequences span billions of base pairs (bp), posing a fundamental challenge for genome-scale foundation models. Existing approaches largely sidestep this barrier by either scaling relatively small models to long contexts or relying on heavy multi-GPU parallelism. Here we introduce GeneZip, a DNA compression model that leverages a key biological prior: genomic information is highly imbalanced. Coding regions comprise only a small fraction (about 2 percent) yet are information-dense, whereas most non-coding sequence is comparatively information-sparse. GeneZip couples HNet-style dynamic routing with a region-aware compression-ratio objective, enabling adaptive allocation of representation budget across genomic regions. As a result, GeneZip learns region-aware compression and achieves 137.6x compression with only 0.31 perplexity increase. On downstream long-context benchmarks, GeneZip achieves comparable or better performance on contact map prediction, expression quantitative trait loci prediction, and enhancer-target gene prediction. By reducing effective sequence length, GeneZip unlocks simultaneous scaling of context and capacity: compared to the prior state-of-the-art model JanusDNA, it enables training models 82.6x larger at 1M-bp context, supporting a 636M-parameter GeneZip model at 1M-bp context. All experiments in this paper can be trained on a single A100 80GB GPU. 

---
