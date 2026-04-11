# KV Cache Offloading for Context-Intensive Tasks 

**Authors**: Andrey Bocharnikov, Ivan Ermakov, Denis Kuznedelev, Vyacheslav Zhdanovskiy, Yegor Yershov  

**Link**: [PDF](https://arxiv.org/pdf/2604.08426)  

**Abstract**: With the growing demand for long-context LLMs across a wide range of applications, the key-value (KV) cache has become a critical bottleneck for both latency and memory usage. Recently, KV-cache offloading has emerged as a promising approach to reduce memory footprint and inference latency while preserving accuracy. Prior evaluations have largely focused on tasks that do not require extracting large amounts of information from the context. In this work, we study KV-cache offloading on context-intensive tasks: problems where the solution requires looking up a lot of information from the input prompt. We create and release the Text2JSON benchmark, a highly context-intensive task that requires extracting structured knowledge from raw text. We evaluate modern KV offloading on Text2JSON and other context-intensive tasks and find significant performance degradation on both Llama 3 and Qwen 3 models. Our analysis identifies two key reasons for poor accuracy: low-rank projection of keys and unreliable landmarks, and proposes a simpler alternative strategy that significantly improves accuracy across multiple LLM families and benchmarks. These findings highlight the need for a comprehensive and rigorous evaluation of long-context compression techniques. 

---
# Small Vision-Language Models are Smart Compressors for Long Video Understanding 

**Authors**: Junjie Fei, Jun Chen, Zechun Liu, Yunyang Xiong, Chong Zhou, Wei Wen, Junlin Han, Mingchen Zhuge, Saksham Suri, Qi Qian, Shuming Liu, Lemeng Wu, Raghuraman Krishnamoorthi, Vikas Chandra, Mohamed Elhoseiny, Chenchen Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2604.08120)  

**Abstract**: Adapting Multimodal Large Language Models (MLLMs) for hour-long videos is bottlenecked by context limits. Dense visual streams saturate token budgets and exacerbate the lost-in-the-middle phenomenon. Existing heuristics, like sparse sampling or uniform pooling, blindly sacrifice fidelity by discarding decisive moments and wasting bandwidth on irrelevant backgrounds. We propose Tempo, an efficient query-aware framework compressing long videos for downstream understanding. Tempo leverages a Small Vision-Language Model (SVLM) as a local temporal compressor, casting token reduction as an early cross-modal distillation process to generate compact, intent-aligned representations in a single forward pass. To enforce strict budgets without breaking causality, we introduce Adaptive Token Allocation (ATA). Exploiting the SVLM's zero-shot relevance prior and semantic front-loading, ATA acts as a training-free $O(1)$ dynamic router. It allocates dense bandwidth to query-critical segments while compressing redundancies into minimal temporal anchors to maintain the global storyline. Extensive experiments show our 6B architecture achieves state-of-the-art performance with aggressive dynamic compression (0.5-16 tokens/frame). On the extreme-long LVBench (4101s), Tempo scores 52.3 under a strict 8K visual budget, outperforming GPT-4o and Gemini 1.5 Pro. Scaling to 2048 frames reaches 53.7. Crucially, Tempo compresses hour-long videos substantially below theoretical limits, proving true long-form video understanding relies on intent-driven efficiency rather than greedily padded context windows. 

---
# A Decomposition Perspective to Long-context Reasoning for LLMs 

**Authors**: Yanling Xiao, Huaibing Xie, Guoliang Zhao, Shihan Dou, Shaolei Wang, Yiting Liu, Nantao Zheng, Cheng Zhang, Pluto Zhou, Zhisong Zhang, Lemao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.07981)  

**Abstract**: Long-context reasoning is essential for complex real-world applications, yet remains a significant challenge for Large Language Models (LLMs). Despite the rapid evolution in long-context reasoning, current research often overlooks the internal complexity of the long-context reasoning task itself. In this paper, we move beyond this holistic view and decompose long-context reasoning into a set of fundamental atomic skills, and we then automatically synthesize a suite of pseudo datasets, each explicitly targeting a specific atomic skill. Our empirical analysis confirms that proficiency in these atomic skills is strongly correlated with general long-text reasoning performance. Building on this insight, we employ reinforcement learning on these pseudo datasets to sharpen the model's atomic skills, in the hope of boosting its general long-context reasoning ability. Extensive experiments across multiple benchmarks demonstrate the effectiveness of our approach: it outperforms a strong baseline by an average margin of 7.7\% (improving from 46.3\% to 54.0\%) across Loogle, Loong, LongBench-v2, BrowscompLong, Ruler-qa2, and MRCR. 

---
# PolicyLong: Towards On-Policy Context Extension 

**Authors**: Junlong Jia, Ziyang Chen, Xing Wu, Chaochen Gao, TingHao Yu, Feng Zhang, Songlin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2604.07809)  

**Abstract**: Extending LLM context windows is hindered by scarce high-quality long-context data. Recent methods synthesize data with genuine long-range dependencies via information-theoretic verification, selecting contexts that reduce a base model's predictive entropy. However, their single-pass offline construction with a fixed model creates a fundamental off-policy gap: the static screening landscape misaligns with the model's evolving capabilities, causing the training distribution to drift. We propose PolicyLong, shifting data construction towards a dynamic on-policy paradigm. By iteratively re-executing data screening (entropy computation, retrieval, and verification) using the current model, PolicyLong ensures the training distribution tracks evolving capabilities, yielding an emergent self-curriculum. Crucially, both positive and hard negative contexts derive from the current model's entropy landscape, co-evolving what the model learns to exploit and resist. Experiments on RULER, HELMET, and LongBench-v2 (Qwen2.5-3B) show PolicyLong consistently outperforms EntropyLong and NExtLong, with gains growing at longer contexts (e.g., +2.54 at 128K on RULER), confirming the value of on-policy data evolution. 

---
# TSUBASA: Improving Long-Horizon Personalization via Evolving Memory and Self-Learning with Context Distillation 

**Authors**: Xinliang Frederick Zhang, Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.07894)  

**Abstract**: Personalized large language models (PLLMs) have garnered significant attention for their ability to align outputs with individual's needs and preferences. However, they still struggle with long-horizon tasks, such as tracking a user's extensive history of conversations or activities. Existing memory mechanisms often fail to capture evolving behaviors, and RAG paradigms are trapped by a quality-efficiency tradeoff. Meanwhile, parametric adaptation is bottlenecked by train-inference gap due to the scarcity of labeled data. To enhance the long-horizon capabilities of PLLMs, we introduce TSUBASA, a two-pronged approach designed to improve memory writing via dynamic memory evolution, and memory reading via self-learning with a context distillation objective to internalize user experiences. Extensive evaluations on long-horizon benchmarks using the Qwen-3 model family (4B to 32B) validate the effectiveness of TSUBASA, surpassing competitive memory-augmented systems that rely primarily on memory writing, such as Mem0 and Memory-R1. Our analyses further confirms that TSUBASA breaks the quality-efficiency barrier to achieve Pareto improvements, delivering robust, high-fidelity personalization with a reduced token budget. 

---
# Optimal Decay Spectra for Linear Recurrences 

**Authors**: Yang Cao  

**Link**: [PDF](https://arxiv.org/pdf/2604.07658)  

**Abstract**: Linear recurrent models offer linear-time sequence processing but often suffer from suboptimal long-range memory. We trace this to the decay spectrum: for $N$ channels, random initialization collapses the minimum spectral gap to $O(N^{-2})$, yielding sub-exponential error $\exp(-\Omega(N/\log N))$; linear spacing avoids collapse but degrades to $\exp(-O(N/\sqrt{T}))$, practically algebraic over long contexts. We introduce Position-Adaptive Spectral Tapering (PoST), an architecture-agnostic framework combining two mechanisms: (1) Spectral Reparameterization, which structurally enforces geometrically spaced log-decay rates, proven minimax optimal at rate $O(\exp(-cN/\log T))$; and (2) Position-Adaptive Scaling, the provably unique mechanism that eliminates the scale mismatch of static spectra (where only $N\log t/\log T$ of $N$ channels are effective at position $t$) by stretching the spectrum to the actual dependency range, sharpening the rate to $O(\exp(-cN/\log t))$. This scaling natively induces fractional invariance: the impulse response becomes scale-free, with channels interpolating between relative and absolute temporal coordinates. PoST integrates into any diagonal linear recurrence without overhead. We instantiate it across Mamba-2, RWKV-7, Gated DeltaNet, Gated Linear Attention, and RetNet. Pre-training at 180M-440M scales shows consistent zero-shot language modeling improvements, significant long-context retrieval gains for Mamba-2 (MQAR and NIAH), and competitive or improved performance across other architectures. Code: this https URL. 

---
# Beyond Human-Readable: Rethinking Software Engineering Conventions for the Agentic Development Era 

**Authors**: Dmytro Ustynov  

**Link**: [PDF](https://arxiv.org/pdf/2604.07502)  

**Abstract**: For six decades, software engineering principles have been optimized for a single consumer: the human developer. The rise of agentic AI development, where LLM-based agents autonomously read, write, navigate, and debug codebases, introduces a new primary consumer with fundamentally different constraints. This paper presents a systematic analysis of human-centric conventions under agentic pressure and proposes a key design principle: semantic density optimization, eliminating tokens that carry zero information while preserving tokens that carry high semantic value. We validate this principle through a controlled experiment on log format token economy across four conditions (human-readable, structured, compressed, and tool-assisted compressed), demonstrating a counterintuitive finding: aggressive compression increased total session cost by 67% despite reducing input tokens by 17%, because it shifted interpretive burden to the model's reasoning phase. We extend this principle to propose the rehabilitation of classical anti-patterns, introduce the program skeleton concept for agentic code navigation, and argue for a fundamental decoupling of semantic intent from human-readable representation. 

---
# Task-Adaptive Retrieval over Agentic Multi-Modal Web Histories via Learned Graph Memory 

**Authors**: Saman Forouzandeh, Kamal Berahmand, Mahdi Jalili  

**Link**: [PDF](https://arxiv.org/pdf/2604.07863)  

**Abstract**: Retrieving relevant observations from long multi-modal web interaction histories is challenging because relevance depends on the evolving task state, modality (screenshots, HTML text, structured signals), and temporal distance. Prior approaches typically rely on static similarity thresholds or fixed-capacity buffers, which fail to adapt relevance to the current task context. We propose \textbf{ACGM}, a learned graph-memory retriever that constructs \emph{task-adaptive} relevance graphs over agent histories using policy-gradient optimization from downstream task success. ACGM captures heterogeneous temporal dynamics with modality-specific decay (visual decays $4.3\times$ faster than text: $\lambda_v{=}0.47$ vs.\ $\lambda_x{=}0.11$) and learns sparse connectivity (3.2 edges/node), enabling efficient $O(\log T)$ retrieval. Across WebShop, VisualWebArena, and Mind2Web, ACGM improves retrieval quality to \textbf{82.7 nDCG@10} (+9.3 over GPT-4o, $p{<}0.001$) and \textbf{89.2\% Precision@10} (+7.7), outperforming 19 strong dense, re-ranking, multi-modal, and graph-based baselines. Code to reproduce our results is available at{\color{blue}\href{this https URL}{Saman Forouzandeh}}. 

---
# HyperMem: Hypergraph Memory for Long-Term Conversations 

**Authors**: Juwei Yue, Chuanrui Hu, Jiawei Sheng, Zuyi Zhou, Wenyuan Zhang, Tingwen Liu, Li Guo, Yafeng Deng  

**Link**: [PDF](https://arxiv.org/pdf/2604.08256)  

**Abstract**: Long-term memory is essential for conversational agents to maintain coherence, track persistent tasks, and provide personalized interactions across extended dialogues. However, existing approaches as Retrieval-Augmented Generation (RAG) and graph-based memory mostly rely on pairwise relations, which can hardly capture high-order associations, i.e., joint dependencies among multiple elements, causing fragmented retrieval. To this end, we propose HyperMem, a hypergraph-based hierarchical memory architecture that explicitly models such associations using hyperedges. Particularly, HyperMem structures memory into three levels: topics, episodes, and facts, and groups related episodes and their facts via hyperedges, unifying scattered content into coherent units. Leveraging this structure, we design a hybrid lexical-semantic index and a coarse-to-fine retrieval strategy, supporting accurate and efficient retrieval of high-order associations. Experiments on the LoCoMo benchmark show that HyperMem achieves state-of-the-art performance with 92.73% LLM-as-a-judge accuracy, demonstrating the effectiveness of HyperMem for long-term conversations. 

---
# AsyncTLS: Efficient Generative LLM Inference with Asynchronous Two-level Sparse Attention 

**Authors**: Yuxuan Hu, Jianchao Tan, Jiaqi Zhang, Wen Zan, Pingwei Sun, Yifan Lu, Yerui Sun, Yuchen Xie, Xunliang Cai, Jing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.07815)  

**Abstract**: Long-context inference in LLMs faces the dual challenges of quadratic attention complexity and prohibitive KV cache memory. While token-level sparse attention offers superior accuracy, its indexing overhead is costly; block-level methods improve efficiency but sacrifice precision. We propose AsyncTLS, a hierarchical sparse attention system that combines coarse-grained block filtering with fine-grained token selection to balance accuracy and efficiency, coupled with an asynchronous offloading engine that overlaps KV cache transfers with computation via temporal locality exploitation. Evaluated on Qwen3 and GLM-4.7-Flash across GQA, and MLA architectures, AsyncTLS achieves accuracy comparable to full attention while delivering 1.2x - 10.0x operator speedups and 1.3x - 4.7x end-to-end throughput improvements on 48k - 96k contexts. 

---
# SepSeq: A Training-Free Framework for Long Numerical Sequence Processing in LLMs 

**Authors**: Jie Sun, Yu Liu, Lu Han, Qiwen Deng, Xiang Shu, Yang Xiao, Xingyu Lu, Jun Zhou, Pengfei Liu, Lintao Ma, Jiancan Wu, Xiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.07737)  

**Abstract**: While transformer-based Large Language Models (LLMs) theoretically support massive context windows, they suffer from severe performance degradation when processing long numerical sequences. We attribute this failure to the attention dispersion in the Softmax mechanism, which prevents the model from concentrating attention. To overcome this, we propose Separate Sequence (SepSeq), a training-free, plug-and-play framework to mitigate dispersion by strategically inserting separator tokens. Mechanistically, we demonstrate that separator tokens act as an attention sink, recalibrating attention to focus on local segments while preserving global context. Extensive evaluations on 9 widely-adopted LLMs confirm the effectiveness of our approach: SepSeq yields an average relative accuracy improvement of 35.6% across diverse domains while reducing total inference token consumption by 16.4% on average. 

---
# Flux Attention: Context-Aware Hybrid Attention for Efficient LLMs Inference 

**Authors**: Quantong Qiu, Zhiyi Hong, Yi Yang, Haitian Wang, Kebin Liu, Qingqing Dang, Juntao Li, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.07394)  

**Abstract**: The quadratic computational complexity of standard attention mechanisms presents a severe scalability bottleneck for LLMs in long-context scenarios. While hybrid attention mechanisms combining Full Attention (FA) and Sparse Attention (SA) offer a potential solution, existing methods typically rely on static allocation ratios that fail to accommodate the variable retrieval demands of different tasks. Furthermore, head-level dynamic sparsity often introduces severe computational load imbalance and synchronization long-tails, which hinder hardware acceleration during autoregressive decoding. To bridge this gap, we introduce Flux Attention, a context-aware framework that dynamically optimizes attention computation at the layer level. By integrating a lightweight Layer Router into frozen pretrained LLMs, the proposed method adaptively routes each layer to FA or SA based on the input context. This layer-wise routing preserves high-fidelity information retrieval while ensuring contiguous memory access, translating theoretical computational reductions into practical wall-clock speedups. As a parameter-efficient approach, our framework requires only 12 hours of training on 8$\times$A800 GPUs. Extensive experiments across multiple long-context and mathematical reasoning benchmarks demonstrate that Flux Attention achieves a superior trade-off between performance and inference speed compared with baseline models, with speed improvements of up to $2.8\times$ and $2.0\times$ in the prefill and decode stages. 

---
