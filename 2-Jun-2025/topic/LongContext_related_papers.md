# Conversational Exploration of Literature Landscape with LitChat 

**Authors**: Mingyu Huang, Shasha Zhou, Yuxuan Chen, Ke Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.23789)  

**Abstract**: We are living in an era of "big literature", where the volume of digital scientific publications is growing exponentially. While offering new opportunities, this also poses challenges for understanding literature landscapes, as traditional manual reviewing is no longer feasible. Recent large language models (LLMs) have shown strong capabilities for literature comprehension, yet they are incapable of offering "comprehensive, objective, open and transparent" views desired by systematic reviews due to their limited context windows and trust issues like hallucinations. Here we present LitChat, an end-to-end, interactive and conversational literature agent that augments LLM agents with data-driven discovery tools to facilitate literature exploration. LitChat automatically interprets user queries, retrieves relevant sources, constructs knowledge graphs, and employs diverse data-mining techniques to generate evidence-based insights addressing user needs. We illustrate the effectiveness of LitChat via a case study on AI4Health, highlighting its capacity to quickly navigate the users through large-scale literature landscape with data-based evidence that is otherwise infeasible with traditional means. 

---
# NexusSum: Hierarchical LLM Agents for Long-Form Narrative Summarization 

**Authors**: Hyuntak Kim, Byung-Hak Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.24575)  

**Abstract**: Summarizing long-form narratives--such as books, movies, and TV scripts--requires capturing intricate plotlines, character interactions, and thematic coherence, a task that remains challenging for existing LLMs. We introduce NexusSum, a multi-agent LLM framework for narrative summarization that processes long-form text through a structured, sequential pipeline--without requiring fine-tuning. Our approach introduces two key innovations: (1) Dialogue-to-Description Transformation: A narrative-specific preprocessing method that standardizes character dialogue and descriptive text into a unified format, improving coherence. (2) Hierarchical Multi-LLM Summarization: A structured summarization pipeline that optimizes chunk processing and controls output length for accurate, high-quality summaries. Our method establishes a new state-of-the-art in narrative summarization, achieving up to a 30.0% improvement in BERTScore (F1) across books, movies, and TV scripts. These results demonstrate the effectiveness of multi-agent LLMs in handling long-form content, offering a scalable approach for structured summarization in diverse storytelling domains. 

---
# SALE : Low-bit Estimation for Efficient Sparse Attention in Long-context LLM Prefilling 

**Authors**: Xiaodong Ji, Hailin Zhang, Fangcheng Fu, Bin Cui  

**Link**: [PDF](https://arxiv.org/pdf/2505.24179)  

**Abstract**: Many advanced Large Language Model (LLM) applications require long-context processing, but the self-attention module becomes a bottleneck during the prefilling stage of inference due to its quadratic time complexity with respect to sequence length. Existing sparse attention methods accelerate attention computation by skipping less significant regions of the attention map. However, these approaches typically perform coarse-grained inspection of the attention map, rendering considerable loss in model accuracy. In this paper, we propose SALE, a fine-grained sparse attention method that accelerates the long-context prefilling stage of LLM with negligible loss in model accuracy. SALE achieves fast and accurate fine-grained attention weight estimation through 4-bit quantized query-key products, followed by block-sparse attention to accelerate prefilling computations. For importance evaluation for query-key pairs, we adopt our Relative Attention Score metric, which offers significantly higher efficiency within our framework. We implement a custom CUDA kernel optimized for our approach for hardware efficiency, reducing the additional overhead to approximately 11% of the full attention latency. Notably, SALE requires no parameter training and can be seamlessly integrated into existing systems with trivial code modifications. Experiments on long-context benchmarks demonstrate that our method outperforms existing approaches in accuracy-efficiency trade-offs, achieving at least 3.36x speedups on Llama-3.1-8B for sequences longer than 64K while maintaining model quality. 

---
# MARS-Bench: A Multi-turn Athletic Real-world Scenario Benchmark for Dialogue Evaluation 

**Authors**: Chenghao Yang, Yinbo Luo, Zhoufutu Wen, Qi Chu, Tao Gong, Longxiang Liu, Kaiyuan Zhang, Jianpeng Jiao, Ge Zhang, Wenhao Huang, Nenghai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.23810)  

**Abstract**: Large Language Models (\textbf{LLMs}), e.g. ChatGPT, have been widely adopted in real-world dialogue applications. However, LLMs' robustness, especially in handling long complex dialogue sessions, including frequent motivation transfer, sophisticated cross-turn dependency, is criticized all along. Nevertheless, no existing benchmarks can fully reflect these weaknesses. We present \textbf{MARS-Bench}, a \textbf{M}ulti-turn \textbf{A}thletic \textbf{R}eal-world \textbf{S}cenario Dialogue \textbf{Bench}mark, designed to remedy the gap. MARS-Bench is constructed from play-by-play text commentary so to feature realistic dialogues specifically designed to evaluate three critical aspects of multi-turn conversations: Ultra Multi-turn, Interactive Multi-turn, and Cross-turn Tasks. Extensive experiments on MARS-Bench also reveal that closed-source LLMs significantly outperform open-source alternatives, explicit reasoning significantly boosts LLMs' robustness on handling long complex dialogue sessions, and LLMs indeed face significant challenges when handling motivation transfer and sophisticated cross-turn dependency. Moreover, we provide mechanistic interpretability on how attention sinks due to special tokens lead to LLMs' performance degradation when handling long complex dialogue sessions based on attention visualization experiment in Qwen2.5-7B-Instruction. 

---
# MultiPhishGuard: An LLM-based Multi-Agent System for Phishing Email Detection 

**Authors**: Yinuo Xue, Eric Spero, Yun Sing Koh, Giovanni Russello  

**Link**: [PDF](https://arxiv.org/pdf/2505.23803)  

**Abstract**: Phishing email detection faces critical challenges from evolving adversarial tactics and heterogeneous attack patterns. Traditional detection methods, such as rule-based filters and denylists, often struggle to keep pace with these evolving tactics, leading to false negatives and compromised security. While machine learning approaches have improved detection accuracy, they still face challenges adapting to novel phishing strategies. We present MultiPhishGuard, a dynamic LLM-based multi-agent detection system that synergizes specialized expertise with adversarial-aware reinforcement learning. Our framework employs five cooperative agents (text, URL, metadata, explanation simplifier, and adversarial agents) with automatically adjusted decision weights powered by a Proximal Policy Optimization reinforcement learning algorithm. To address emerging threats, we introduce an adversarial training loop featuring an adversarial agent that generates subtle context-aware email variants, creating a self-improving defense ecosystem and enhancing system robustness. Experimental evaluations on public datasets demonstrate that MultiPhishGuard significantly outperforms Chain-of-Thoughts, single-agent baselines and state-of-the-art detectors, as validated by ablation studies and comparative analyses. Experiments demonstrate that MultiPhishGuard achieves high accuracy (97.89\%) with low false positive (2.73\%) and false negative rates (0.20\%). Additionally, we incorporate an explanation simplifier agent, which provides users with clear and easily understandable explanations for why an email is classified as phishing or legitimate. This work advances phishing defense through dynamic multi-agent collaboration and generative adversarial resilience. 

---
# HiCaM: A Hierarchical-Causal Modification Framework for Long-Form Text Modification 

**Authors**: Yuntao Shi, Yi Luo, Yeyun Gong, Chen Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.24319)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable success in various domains. However, when handling long-form text modification tasks, they still face two major problems: (1) producing undesired modifications by inappropriately altering or summarizing irrelevant content, and (2) missing necessary modifications to implicitly related passages that are crucial for maintaining document coherence. To address these issues, we propose HiCaM, a Hierarchical-Causal Modification framework that operates through a hierarchical summary tree and a causal graph. Furthermore, to evaluate HiCaM, we derive a multi-domain dataset from various benchmarks, providing a resource for assessing its effectiveness. Comprehensive evaluations on the dataset demonstrate significant improvements over strong LLMs, with our method achieving up to a 79.50\% win rate. These results highlight the comprehensiveness of our approach, showing consistent performance improvements across multiple models and domains. 

---
# Beyond Exponential Decay: Rethinking Error Accumulation in Large Language Models 

**Authors**: Mikhail L. Arbuzov, Alexey A. Shvets, Sisong Beir  

**Link**: [PDF](https://arxiv.org/pdf/2505.24187)  

**Abstract**: The prevailing assumption of an exponential decay in large language model (LLM) reliability with sequence length, predicated on independent per-token error probabilities, posits an inherent limitation for long autoregressive outputs. Our research fundamentally challenges this view by synthesizing emerging evidence that LLM errors are not uniformly distributed but are concentrated at sparse "key tokens" ($5-10\%$ of total tokens) representing critical decision junctions. By distinguishing these high-impact tokens from the increasingly predictable majority, we introduce a new reliability formula explaining the sustained coherence of modern LLMs over thousands of tokens. Converging research streams reveal that long-context performance primarily depends on accurately navigating a few crucial semantic decision points rather than on uniform token-level accuracy, enabling targeted strategies that significantly outperform brute-force approaches. We thus propose a framework for next-generation systems centered on selective preservation of semantically vital tokens, dynamic computational allocation at uncertain decision boundaries, multi-path exploration at ambiguities, and architectures aligned with natural semantic domains. This marks a fundamental shift from raw scaling to strategic reasoning, promising breakthrough performance without proportionate computational scaling and offering a more nuanced understanding that supersedes the exponential decay hypothesis, thereby opening pathways toward substantially more powerful and efficient language systems. 

---
# SwingArena: Competitive Programming Arena for Long-context GitHub Issue Solving 

**Authors**: Wendong Xu, Jing Xiong, Chenyang Zhao, Qiujiang Chen, Haoran Wang, Hui Shen, Zhongwei Wan, Jianbo Dai, Taiqiang Wu, He Xiao, Chaofan Tao, Z. Morley Mao, Ying Sheng, Zhijiang Guo, Hongxia Yang, Bei Yu, Lingpeng Kong, Quanquan Gu, Ngai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2505.23932)  

**Abstract**: We present SwingArena, a competitive evaluation framework for Large Language Models (LLMs) that closely mirrors real-world software development workflows. Unlike traditional static benchmarks, SwingArena models the collaborative process of software iteration by pairing LLMs as submitters, who generate patches, and reviewers, who create test cases and verify the patches through continuous integration (CI) pipelines. To support these interactive evaluations, we introduce a retrieval-augmented code generation (RACG) module that efficiently handles long-context challenges by providing syntactically and semantically relevant code snippets from large codebases, supporting multiple programming languages (C++, Python, Rust, and Go). This enables the framework to scale across diverse tasks and contexts while respecting token limitations. Our experiments, using over 400 high-quality real-world GitHub issues selected from a pool of 2,300 issues, show that models like GPT-4o excel at aggressive patch generation, whereas DeepSeek and Gemini prioritize correctness in CI validation. SwingArena presents a scalable and extensible methodology for evaluating LLMs in realistic, CI-driven software development settings. More details are available on our project page: this http URL 

---
# Test-Time Training Done Right 

**Authors**: Tianyuan Zhang, Sai Bi, Yicong Hong, Kai Zhang, Fujun Luan, Songlin Yang, Kalyan Sunkavalli, William T. Freeman, Hao Tan  

**Link**: [PDF](https://arxiv.org/pdf/2505.23884)  

**Abstract**: Test-Time Training (TTT) models context dependencies by adapting part of the model's weights (referred to as fast weights) during inference. This fast weight, akin to recurrent states in RNNs, stores temporary memories of past tokens in the current sequence. Existing TTT methods struggled to show effectiveness in handling long-context data, due to their inefficiency on modern GPUs. The TTT layers in many of these approaches operate with extremely low FLOPs utilization (often <5%) because they deliberately apply small online minibatch sizes (e.g., updating fast weights every 16 or 64 tokens). Moreover, a small minibatch implies fine-grained block-wise causal dependencies in the data, unsuitable for data beyond 1D ordered sequences, like sets or N-dimensional grids such as images or videos. In contrast, we pursue the opposite direction by using an extremely large chunk update, ranging from 2K to 1M tokens across tasks of varying modalities, which we refer to as Large Chunk Test-Time Training (LaCT). It improves hardware utilization by orders of magnitude, and more importantly, facilitates scaling of nonlinear state size (up to 40% of model parameters), hence substantially improving state capacity, all without requiring cumbersome and error-prone kernel implementations. It also allows easy integration of sophisticated optimizers, e.g. Muon for online updates. We validate our approach across diverse modalities and tasks, including novel view synthesis with image set, language models, and auto-regressive video diffusion. Our approach can scale up to 14B-parameter AR video diffusion model on sequences up to 56K tokens. In our longest sequence experiment, we perform novel view synthesis with 1 million context length. We hope this work will inspire and accelerate new research in the field of long-context modeling and test-time training. Website: this https URL 

---
