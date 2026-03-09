# Beyond Rows to Reasoning: Agentic Retrieval for Multimodal Spreadsheet Understanding and Editing 

**Authors**: Anmol Gulati, Sahil Sen, Waqar Sarguroh, Kevin Paul  

**Link**: [PDF](https://arxiv.org/pdf/2603.06503)  

**Abstract**: Recent advances in multimodal Retrieval-Augmented Generation (RAG) enable Large Language Models (LLMs) to analyze enterprise spreadsheet workbooks containing millions of cells, cross-sheet dependencies, and embedded visual artifacts. However, state-of-the-art approaches exclude critical context through single-pass retrieval, lose data resolution through compression, and exceed LLM context windows through naive full-context injection, preventing reliable multi-step reasoning over complex enterprise workbooks. We introduce Beyond Rows to Reasoning (BRTR), a multimodal agentic framework for spreadsheet understanding that replaces single-pass retrieval with an iterative tool-calling loop, supporting end-to-end Excel workflows from complex analysis to structured editing. Supported by over 200 hours of expert human evaluation, BRTR achieves state-of-the-art performance across three frontier spreadsheet understanding benchmarks, surpassing prior methods by 25 percentage points on FRTR-Bench, 7 points on SpreadsheetLLM, and 32 points on FINCH. We evaluate five multimodal embedding models, identifying NVIDIA NeMo Retriever 1B as the top performer for mixed tabular and visual data, and vary nine LLMs. Ablation experiments confirm that the planner, retrieval, and iterative reasoning each contribute substantially, and cost analysis shows GPT-5.2 achieves the best efficiency-accuracy trade-off. Throughout all evaluations, BRTR maintains full auditability through explicit tool-call traces. 

---
# FlashPrefill: Instantaneous Pattern Discovery and Thresholding for Ultra-Fast Long-Context Prefilling 

**Authors**: Qihang Fan, Huaibo Huang, Zhiying Wu, Juqiu Wang, Bingning Wang, Ran He  

**Link**: [PDF](https://arxiv.org/pdf/2603.06199)  

**Abstract**: Long-context modeling is a pivotal capability for Large Language Models, yet the quadratic complexity of attention remains a critical bottleneck, particularly during the compute-intensive prefilling phase. While various sparse attention mechanisms have been explored, they typically suffer from either significant search latency or insufficient sparsity. In this paper, we propose FlashPrefill, a framework enabling ultra-fast prefilling via instantaneous pattern discovery and thresholding. FlashPrefill leverages a fast block-searching technique to simultaneously locate dynamic vertical, slash, and block-sparse attention patterns. Crucially, it introduces a dynamic thresholding mechanism that bypasses the prohibitive overhead of sorting or accumulating attention scores while effectively eliminating the long-tail distribution to enhance sparsity. Extensive evaluations demonstrate that FlashPrefill achieves a substantial leap in efficiency, delivering an unprecedented 27.78x speedup on 256K sequences. Notably, unlike existing methods that incur efficiency degradation on shorter contexts, FlashPrefill maintains a 1.71x speedup even at a 4K context length, demonstrating its robustness and practical utility across varying sequence scales. 

---
# Learning Next Action Predictors from Human-Computer Interaction 

**Authors**: Omar Shaikh, Valentin Teutschbein, Kanishk Gandhi, Yikun Chi, Nick Haber, Thomas Robinson, Nilam Ram, Byron Reeves, Sherry Yang, Michael S. Bernstein, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.05923)  

**Abstract**: Truly proactive AI systems must anticipate what we will do next. This foresight demands far richer information than the sparse signals we type into our prompts -- it demands reasoning over the entire context of what we see and do. We formalize this as next action prediction (NAP): given a sequence of a user's multimodal interactions with a computer (screenshots, clicks, sensor data), predict that user's next action. Progress on this task requires both new data and modeling approaches. To scale data, we annotate longitudinal, naturalistic computer use with vision-language models. We release an open-source pipeline for performing this labeling on private infrastructure, and label over 360K actions across one month of continuous phone usage from 20 users, amounting to 1,800 hours of screen time. We then introduce LongNAP, a user model that combines parametric and in-context learning to reason over long interaction histories. LongNAP is trained via policy gradient methods to generate user-specific reasoning traces given some context; retrieve relevant traces from a library of past traces; and then apply retrieved traces in-context to predict future actions. Using an LLM-as-judge evaluation metric (0-1 similarity to ground truth), LongNAP significantly outperforms supervised finetuning and prompted baselines on held-out data (by 79% and 39% respectively). Additionally, LongNAP generalizes to held out users when trained across individuals. The space of next actions a user might take at any moment is unbounded, spanning thousands of possible outcomes. Despite this, 17.1% of LongNAP's predicted trajectories are well-aligned with what a user does next (LLM-judge score $\geq$ 0.5). This rises to 26% when we filter to highly confident predictions. In sum, we argue that learning from the full context of user behavior to anticipate user needs is now a viable task with substantial opportunity. 

---
# Stem: Rethinking Causal Information Flow in Sparse Attention 

**Authors**: Lin Niu, Xin Luo, Linchuan Xie, Yifu Sun, Guanghua Yu, Jianchen Zhu, S Kevin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2603.06274)  

**Abstract**: The quadratic computational complexity of self-attention remains a fundamental bottleneck for scaling Large Language Models (LLMs) to long contexts, particularly during the pre-filling phase. In this paper, we rethink the causal attention mechanism from the perspective of information flow. Due to causal constraints, tokens at initial positions participate in the aggregation of every subsequent token. However, existing sparse methods typically apply a uniform top-k selection across all token positions within a layer, ignoring the cumulative dependency of token information inherent in causal architectures. To address this, we propose Stem, a novel, plug-and-play sparsity module aligned with information flow. First, Stem employs the Token Position-Decay strategy, applying position-dependent top-k within each layer to retain initial tokens for recursive dependencies. Second, to preserve information-rich tokens, Stem utilizes the Output-Aware Metric. It prioritizes high-impact tokens based on approximate output magnitude. Extensive evaluations demonstrate that Stem achieves superior accuracy with reduced computation and pre-filling latency. 

---
