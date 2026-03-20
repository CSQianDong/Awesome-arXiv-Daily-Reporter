# Mi:dm K 2.5 Pro 

**Authors**: KT Tech innovation Group  

**Link**: [PDF](https://arxiv.org/pdf/2603.18788)  

**Abstract**: The evolving LLM landscape requires capabilities beyond simple text generation, prioritizing multi-step reasoning, long-context understanding, and agentic workflows. This shift challenges existing models in enterprise environments, especially in Korean-language and domain-specific scenarios where scaling is insufficient. We introduce Mi:dm K 2.5 Pro, a 32B parameter flagship LLM designed to address enterprise-grade complexity through reasoning-focused optimization.
Our methodology builds a robust data foundation via a quality-centric curation pipeline utilizing abstract syntax tree (AST) analysis for code, gap-filling synthesis for mathematics, and an LLM-based quality evaluator. Pre-training scales the model via layer-predictor-based Depth Upscaling (DuS) and a progressive strategy supporting a 128K token context window. Post-training introduces a specialized multi-stage pipeline, including Reasoning SFT, model merging, and asynchronous reinforcement learning (RL), to develop complex problem-solving skills. "Fusion Training" then rebalances these capabilities with conversational fluency, consistent response styling, and reliable tool-use.
The evaluations show that Mi:dm K 2.5 Pro achieves competitive performance against leading global and domestic models. In addition, it sets state-of-the-art results on Korean-specific benchmarks, showcasing deep linguistic and cultural understanding. Finally, Responsible AI evaluations validate safety against attacks, ensuring a secure profile for deployment with a balance of harmlessness and responsiveness. 

---
# UT-ACA: Uncertainty-Triggered Adaptive Context Allocation for Long-Context Inference 

**Authors**: Lang Zhou, Shuxuan Li, Zhuohao Li, Shi Liu, Zhilin Zhao, Wei-Shi Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2603.18446)  

**Abstract**: Long-context inference remains challenging for large language models due to attention dilution and out-of-distribution degradation. Context selection mitigates this limitation by attending to a subset of key-value cache entries, yet most methods allocate a fixed context budget throughout decoding despite highly non-uniform token-level contextual demands. To address this issue, we propose Uncertainty-Triggered Adaptive Context Allocation (UT-ACA), an inference-time framework that dynamically adjusts the context window based on token-wise uncertainty. UT-ACA learns an uncertainty detector that combines semantic embeddings with logit-based confidence while accounting for uncertainty accumulation across decoding steps. When insufficient evidence is indicated, UT-ACA selectively rolls back, expands the context window, and regenerates the token with additional support. Experiments show that UT-ACA substantially reduces average context usage while preserving generation quality in long-context settings. 

---
# Frayed RoPE and Long Inputs: A Geometric Perspective 

**Authors**: Davis Wertheimer, Aozhong Zhang, Derrick Liu, Penghang Yin, Naigang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.18017)  

**Abstract**: Rotary Positional Embedding (RoPE) is a widely adopted technique for encoding position in language models, which, while effective, causes performance breakdown when input length exceeds training length. Prior analyses assert (rightly) that long inputs cause channels to rotate ``out of distribution,'' but it is not clear how extra rotation relates to or causes pathological behavior. Through empirical and theoretical analysis we advance a unified geometric understanding of attention behavior with RoPE. We find that attention induces tight clustering of separated key and query latent point clouds, allowing for creation of sink tokens: placeholders that allow attention heads to avoid token mixing when not required. RoPE applied to longer inputs damages this key/query cluster separation, producing pathological behavior by inhibiting sink token functionality. From this geometric perspective, we propose RoPE-ID (In Distribution), a straightforward modification that allows attention layers to generalize to longer inputs out of the box: apply RoPE with high frequency to a subset of channels. We demonstrate the effectiveness of RoPE-ID for extended inputs using 1B and 3B parameter Transformers on the LongBench and RULER information retrieval benchmarks. 

---
# MOSS-TTS Technical Report 

**Authors**: Yitian Gong, Botian Jiang, Yiwei Zhao, Yucheng Yuan, Kuangwei Chen, Yaozhou Jiang, Cheng Chang, Dong Hong, Mingshu Chen, Ruixiao Li, Yiyang Zhang, Yang Gao, Hanfu Chen, Ke Chen, Songlin Wang, Xiaogui Yang, Yuqian Zhang, Kexin Huang, ZhengYuan Lin, Kang Yu, Ziqi Chen, Jin Wang, Zhaoye Fei, Qinyuan Cheng, Shimin Li, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2603.18090)  

**Abstract**: This technical report presents MOSS-TTS, a speech generation foundation model built on a scalable recipe: discrete audio tokens, autoregressive modeling, and large-scale pretraining. Built on MOSS-Audio-Tokenizer, a causal Transformer tokenizer that compresses 24 kHz audio to 12.5 fps with variable-bitrate RVQ and unified semantic-acoustic representations, we release two complementary generators: MOSS-TTS, which emphasizes structural simplicity, scalability, and long-context/control-oriented deployment, and MOSS-TTS-Local-Transformer, which introduces a frame-local autoregressive module for higher modeling efficiency, stronger speaker preservation, and a shorter time to first audio. Across multilingual and open-domain settings, MOSS-TTS supports zero-shot voice cloning, token-level duration control, phoneme-/pinyin-level pronunciation control, smooth code-switching, and stable long-form generation. This report summarizes the design, training recipe, and empirical characteristics of the released models. 

---
# NeuroGame Transformer: Gibbs-Inspired Attention Driven by Game Theory and Statistical Physics 

**Authors**: Djamel Bouchaffra, Fayçal Ykhlef, Hanene Azzag, Mustapha Lebbah, Bilal Faye  

**Link**: [PDF](https://arxiv.org/pdf/2603.18761)  

**Abstract**: Standard attention mechanisms in transformers are limited by their pairwise formulation, which hinders the modeling of higher-order dependencies among tokens. We introduce the NeuroGame Transformer (NGT) to overcome this by reconceptualizing attention through a dual perspective: tokens are treated simultaneously as players in a cooperative game and as interacting spins in a statistical physics system. Token importance is quantified using two complementary game-theoretic concepts -- Shapley values for global, permutation-based attribution and Banzhaf indices for local, coalition-level influence. These are combined via a learnable gating parameter to form an external magnetic field, while pairwise interaction potentials capture synergistic relationships. The system's energy follows an Ising Hamiltonian, with attention weights emerging as marginal probabilities under the Gibbs distribution, efficiently computed via mean-field equations. To ensure scalability despite the exponential coalition space, we develop importance-weighted Monte Carlo estimators with Gibbs-distributed weights. This approach avoids explicit exponential factors, ensuring numerical stability for long sequences. We provide theoretical convergence guarantees and characterize the fairness-sensitivity trade-off governed by the interpolation parameter. Experimental results demonstrate that the NeuroGame Transformer achieves strong performance across SNLI, and MNLI-matched, outperforming some major efficient transformer baselines. On SNLI, it attains a test accuracy of 86.4\% (with a peak validation accuracy of 86.6\%), surpassing ALBERT-Base and remaining highly competitive with RoBERTa-Base. Code is available at this https URL. 

---
# HiMu: Hierarchical Multimodal Frame Selection for Long Video Question Answering 

**Authors**: Dan Ben-Ami, Gabriele Serussi, Kobi Cohen, Chaim Baskin  

**Link**: [PDF](https://arxiv.org/pdf/2603.18558)  

**Abstract**: Long-form video question answering requires reasoning over extended temporal contexts, making frame selection critical for large vision-language models (LVLMs) bound by finite context windows. Existing methods face a sharp trade-off: similarity-based selectors are fast but collapse compositional queries into a single dense vector, losing sub-event ordering and cross-modal bindings; agent-based methods recover this structure through iterative LVLM inference, but at prohibitive cost. We introduce HiMu, a training-free framework that bridges this gap. A single text-only LLM call decomposes the query into a hierarchical logic tree whose leaves are atomic predicates, each routed to a lightweight expert spanning vision (CLIP, open-vocabulary detection, OCR) and audio (ASR, CLAP). The resulting signals are normalized, temporally smoothed to align different modalities, and composed bottom-up through fuzzy-logic operators that enforce temporal sequencing and adjacency, producing a continuous satisfaction curve. Evaluations on Video-MME, LongVideoBench and HERBench-Lite show that HiMu advances the efficiency-accuracy Pareto front: at 16 frames with Qwen3-VL 8B it outperforms all competing selectors, and with GPT-4o it surpasses agentic systems operating at 32-512 frames while requiring roughly 10x fewer FLOPs. 

---
# Self-Tuning Sparse Attention: Multi-Fidelity Hyperparameter Optimization for Transformer Acceleration 

**Authors**: Arundhathi Dev, Justin Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2603.18417)  

**Abstract**: Sparse attention mechanisms promise to break the quadratic bottleneck of long-context transformers, yet production adoption remains limited by a critical usability gap: optimal hyperparameters vary substantially across layers and models, and current methods (e.g., SpargeAttn) rely on manual grid search to identify them. We propose AFBS-BO (Adaptive Fidelity Binary Search with Bayesian Optimization), a fully automated framework that discovers optimal layer- and head-specific hyperparameters without human intervention. Our hybrid algorithm combines Bayesian Optimization for global exploration with binary search for local refinement, leveraging multi-fidelity evaluation across sequence lengths to reduce tuning cost. On Llama-2-7B, AFBS-BO accelerates hyperparameter discovery by 3.4x with 8.8x fewer evaluations than grid search, and identifies high-sparsity configurations that outperform existing sparse attention baselines while closely matching dense attention quality. By transforming sparse attention from a manually tuned heuristic into a self-optimizing primitive, AFBS-BO enables plug-and-play acceleration across diverse transformer architectures and domains. 

---
# Insight-V++: Towards Advanced Long-Chain Visual Reasoning with Multimodal Large Language Models 

**Authors**: Yuhao Dong, Zuyan Liu, Shulin Tian, Yongming Rao, Ziwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.18118)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable reliability and advanced capabilities through extended test-time reasoning. However, extending these capabilities to Multi-modal Large Language Models (MLLMs) remains a significant challenge due to a critical scarcity of high-quality, long-chain reasoning data and optimized training pipelines. To bridge this gap, we present a unified multi-agent visual reasoning framework that systematically evolves from our foundational image-centric model, Insight-V, into a generalized spatial-temporal architecture, Insight-V++. We first propose a scalable data generation pipeline equipped with multi-granularity assessment that autonomously synthesizes structured, complex reasoning trajectories across image and video domains without human intervention. Recognizing that directly supervising MLLMs with such intricate data yields sub-optimal results, we design a dual-agent architecture comprising a reasoning agent to execute extensive analytical chains, and a summary agent to critically evaluate and distill final outcomes. While our initial framework utilized Direct Preference Optimization (DPO), its off-policy nature fundamentally constrained reinforcement learning potential. To overcome these limitations, particularly for long-horizon video understanding, Insight-V++ introduces two novel algorithms, ST-GRPO and J-GRPO, which enhance spatial-temporal reasoning and improve evaluative robustness. Crucially, by leveraging reliable feedback from the summary agent, we guide an iterative reasoning path generation process, retraining the entire multi-agent system in a continuous, self-improving loop. Extensive experiments on base models like LLaVA-NeXT and Qwen2.5-VL demonstrate significant performance gains across challenging image and video reasoning benchmarks while preserving strong capabilities on traditional perception-focused tasks. 

---
