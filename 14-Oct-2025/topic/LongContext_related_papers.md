# REGENT: Relevance-Guided Attention for Entity-Aware Multi-Vector Neural Re-Ranking 

**Authors**: Shubham Chatterjee  

**Link**: [PDF](https://arxiv.org/pdf/2510.11592)  

**Abstract**: Current neural re-rankers often struggle with complex information needs and long, content-rich documents. The fundamental issue is not computational--it is intelligent content selection: identifying what matters in lengthy, multi-faceted texts. While humans naturally anchor their understanding around key entities and concepts, neural models process text within rigid token windows, treating all interactions as equally important and missing critical semantic signals. We introduce REGENT, a neural re-ranking model that mimics human-like understanding by using entities as a "semantic skeleton" to guide attention. REGENT integrates relevance guidance directly into the attention mechanism, combining fine-grained lexical matching with high-level semantic reasoning. This relevance-guided attention enables the model to focus on conceptually important content while maintaining sensitivity to precise term matches. REGENT achieves new state-of-the-art performance in three challenging datasets, providing up to 108% improvement over BM25 and consistently outperforming strong baselines including ColBERT and RankVicuna. To our knowledge, this is the first work to successfully integrate entity semantics directly into neural attention, establishing a new paradigm for entity-aware information retrieval. 

---
# ImCoref-CeS: An Improved Lightweight Pipeline for Coreference Resolution with LLM-based Checker-Splitter Refinement 

**Authors**: Kangyang Luo, Yuzhuo Bai, Shuzheng Si, Cheng Gao, Zhitong Wang, Yingli Shen, Wenhao Li, Zhu Liu, Yufeng Han, Jiayi Wu, Cunliang Kong, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2510.10241)  

**Abstract**: Coreference Resolution (CR) is a critical task in Natural Language Processing (NLP). Current research faces a key dilemma: whether to further explore the potential of supervised neural methods based on small language models, whose detect-then-cluster pipeline still delivers top performance, or embrace the powerful capabilities of Large Language Models (LLMs). However, effectively combining their strengths remains underexplored. To this end, we propose \textbf{ImCoref-CeS}, a novel framework that integrates an enhanced supervised model with LLM-based reasoning. First, we present an improved CR method (\textbf{ImCoref}) to push the performance boundaries of the supervised neural method by introducing a lightweight bridging module to enhance long-text encoding capability, devising a biaffine scorer to comprehensively capture positional information, and invoking a hybrid mention regularization to improve training efficiency. Importantly, we employ an LLM acting as a multi-role Checker-Splitter agent to validate candidate mentions (filtering out invalid ones) and coreference results (splitting erroneous clusters) predicted by ImCoref. Extensive experiments demonstrate the effectiveness of ImCoref-CeS, which achieves superior performance compared to existing state-of-the-art (SOTA) methods. 

---
# Enhancing Long Chain-of-Thought Reasoning through Multi-Path Plan Aggregation 

**Authors**: Siheng Xiong, Ali Payani, Faramarz Fekri  

**Link**: [PDF](https://arxiv.org/pdf/2510.11620)  

**Abstract**: Inference-time scaling enhances the reasoning ability of a language model (LM) by extending its chain-of-thought (CoT). However, existing approaches typically generate the entire reasoning chain in a single forward pass, which often leads to CoT derailment, i.e., the reasoning trajectory drifting off course due to compounding errors. This problem is particularly severe for smaller LMs with long CoTs due to their limited capacity. To address this, we analyze raw long CoTs and uncover a reasoning hierarchy consisting of planning and execution steps. Our analysis reveals that most reasoning errors stem from incorrect planning. Motivated by this observation, we propose Multi-Path Plan Aggregation (MPPA), a framework that augments single-pass reasoning with plan exploration and aggregation. Following a variable interval schedule based on the token position, MPPA generates multiple candidate plans and aggregates them into a refined planning step. To maintain efficiency, we adopt a minimal design in which the base LM serves as the primary policy, while a lightweight LoRA module implements the plan aggregation policy. We further observe that outcome-reward RL is inefficient for long trajectories (e.g., exceeding 4K tokens). To overcome this, we introduce online Step-DPO, a process-level preference optimization scheme that leverages Twisted Sequential Monte Carlo (TSMC) to provide scalable stepwise supervision using small LMs. This yields more efficient training, improved stability, and higher accuracy. Extensive experiments on challenging math, science, and logical reasoning benchmarks demonstrate that, with only 10% SFT data and 5% of preference pairs, our method outperforms both the DeepSeek-R1 distillation baseline and the outcome-reward RL baseline across multiple base models and tasks. 

---
# Large Language Models for Full-Text Methods Assessment: A Case Study on Mediation Analysis 

**Authors**: Wenqing Zhang, Trang Nguyen, Elizabeth A. Stuart, Yiqun T. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.10762)  

**Abstract**: Systematic reviews are crucial for synthesizing scientific evidence but remain labor-intensive, especially when extracting detailed methodological information. Large language models (LLMs) offer potential for automating methodological assessments, promising to transform evidence synthesis. Here, using causal mediation analysis as a representative methodological domain, we benchmarked state-of-the-art LLMs against expert human reviewers across 180 full-text scientific articles. Model performance closely correlated with human judgments (accuracy correlation 0.71; F1 correlation 0.97), achieving near-human accuracy on straightforward, explicitly stated methodological criteria. However, accuracy sharply declined on complex, inference-intensive assessments, lagging expert reviewers by up to 15%. Errors commonly resulted from superficial linguistic cues -- for instance, models frequently misinterpreted keywords like "longitudinal" or "sensitivity" as automatic evidence of rigorous methodological approache, leading to systematic misclassifications. Longer documents yielded lower model accuracy, whereas publication year showed no significant effect. Our findings highlight an important pattern for practitioners using LLMs for methods review and synthesis from full texts: current LLMs excel at identifying explicit methodological features but require human oversight for nuanced interpretations. Integrating automated information extraction with targeted expert review thus provides a promising approach to enhance efficiency and methodological rigor in evidence synthesis across diverse scientific fields. 

---
# UltraLLaDA: Scaling the Context Length to 128K for Diffusion Large Language Models 

**Authors**: Guangxin He, Shen Nie, Fengqi Zhu, Yuankang Zhao, Tianyi Bai, Ran Yan, Jie Fu, Chongxuan Li, Binhang Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2510.10481)  

**Abstract**: Diffusion LLMs have attracted growing interest, with plenty of recent work emphasizing their great potential in various downstream tasks; yet the long-context behavior of diffusion LLMs remains largely uncharted. We present a case study of post-training techniques for extending the context window of diffusion LLMs (i.e., LLaDA) without retraining from scratch. We show that a simple modification to the standard Rotary Positional Embeddings (RoPE) extension effectively accommodates the probabilistic modeling inherent in the diffusion process, enabling stable scaling to longer context ranges. We further compare masking strategies used during post-training and analyze their impact on optimization stability and long-range recall. Instantiating these insights, we introduce UltraLLaDA, a diffusion LLM with a 128K-token context window that, in our empirical evaluation on long-context tasks, significantly outperforms training-free baselines. Our experimental results highlight the special positional extension as a key lever for scaling diffusion LLMs to extended contexts and offer practical guidance for practitioners seeking 128K-scale context via efficient post-training. 

---
# RECON: Reasoning with Condensation for Efficient Retrieval-Augmented Generation 

**Authors**: Zhichao Xu, Minheng Wang, Yawei Wang, Wenqian Ye, Yuntao Du, Yunpu Ma, Yijun Tian  

**Link**: [PDF](https://arxiv.org/pdf/2510.10448)  

**Abstract**: Retrieval-augmented generation (RAG) systems trained using reinforcement learning (RL) with reasoning are hampered by inefficient context management, where long, noisy retrieved documents increase costs and degrade performance. We introduce RECON (REasoning with CONdensation), a framework that integrates an explicit summarization module to compress evidence within the reasoning loop. Our summarizer is trained via a two-stage process: relevance pretraining on QA datasets, followed by multi-aspect distillation from proprietary LLMs to ensure factuality and clarity. Integrated into the Search-R1 pipeline, RECON reduces total context length by 35\%, leading to improved training speed and inference latency, while simultaneously improving RAG performance on downstream QA benchmarks. Notably, it boosts the average EM score of the 3B model by 14.5\% and the 7B model by 3.0\%, showing particular strength in multi-hop QA. RECON demonstrates that learned context compression is essential for building practical, scalable, and performant RAG systems. Our code implementation is made available at this https URL. 

---
# DELTA: Dynamic Layer-Aware Token Attention for Efficient Long-Context Reasoning 

**Authors**: Hossein Entezari Zarch, Lei Gao, Chaoyi Jiang, Murali Annavarm  

**Link**: [PDF](https://arxiv.org/pdf/2510.09883)  

**Abstract**: Large reasoning models (LRMs) achieve state-of-the-art performance on challenging benchmarks by generating long chains of intermediate steps, but their inference cost is dominated by decoding, where each new token must attend to the entire growing sequence. Existing sparse attention methods reduce computation by pruning the key-value (KV) cache, yet they suffer from severe accuracy degradation on reasoning tasks due to cumulative selection errors and the dynamic importance of tokens over long derivations. We present \textbf{DELTA}, a training-free sparse attention mechanism that achieves computational efficiency without sacrificing model accuracy. DELTA partitions transformer layers into three groups: initial layers that use full attention, a small set of \emph{selection layers} that identify salient tokens via aggregated head-level attention scores, and subsequent \emph{sparse-attention layers} that attend only to the selected subset. This design preserves the full KV cache in GPU memory for accuracy, while avoiding expensive full-attention computation over many layers. On reasoning benchmarks such as AIME and GPQA-Diamond, DELTA matches or surpasses full attention in accuracy, while reducing the number of attended tokens by up to $5\times$ and delivering $1.5\times$ end-to-end speedup. Our results show that selective reuse of intermediate attention maps offers a robust path toward efficient long-context reasoning. 

---
# A Layered Intuition -- Method Model with Scope Extension for LLM Reasoning 

**Authors**: Hong Su  

**Link**: [PDF](https://arxiv.org/pdf/2510.10592)  

**Abstract**: Existing studies have introduced method-based reasoning and scope extension as approaches to enhance Large Language Model (LLM) performance beyond direct matrix mappings. Building on these foundations, this paper summarizes and integrates these ideas into a unified Intuition-Method Layered Model with Scope Extension, designed to address indirected (unseen) issues more systematically. In this framework, intuition-based thinking provides rapid first-reaction answers, while method-based thinking decouples questions and solutions into transferable reasoning units. Scope extension is then applied to broaden applicability, including vertical (cause analysis), horizontal (parallel and generalized issues), and for the first time, temporal and spatial extensions, which expand reasoning across time and contextual dimensions. These extensions are organized into systematic knowledge trees that interconnect into a knowledge network, thereby increasing adaptability. To quantitatively evaluate this process, we propose the entropy of method extension, which measures the independence and diversity of extensions as an indicator of the system's capacity to solve unseen questions. By logically connecting existing approaches with new extensions and introducing an entropy-based evaluation framework, this work advances toward a more robust and extensible reasoning paradigm for LLMs in real-world problem-solving. 

---
# Analyzing and Internalizing Complex Policy Documents for LLM Agents 

**Authors**: Jiateng Liu, Zhenhailong Wang, Xiaojiang Huang, Yingjie Li, Xing Fan, Xiang Li, Chenlei Guo, Ruhi Sarikaya, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2510.11588)  

**Abstract**: Large Language Model (LLM)-based agentic systems rely on in-context policy documents encoding diverse business rules. As requirements grow, these documents expand rapidly, causing high computational overhead. This motivates developing internalization methods that embed policy documents into model priors while preserving performance. Prior prompt compression work targets generic prompts, but agentic policy documents span multiple complexity levels and require deeper reasoning, making internalization harder. We introduce CC-Gen, an agentic benchmark generator with Controllable Complexity across four levels, enabling systematic evaluation of agents' ability to handle complexity and offering a unified framework for assessing policy internalization. Our analysis shows that complex policy specifications governing workflows pose major reasoning challenges. Supporting internalization with gold user agent interaction trajectories containing chain-of-thought (CoT) annotations via supervised fine-tuning (SFT) is data-intensive and degrades sharply as policy complexity increases. To mitigate data and reasoning burdens, we propose Category-Aware Policy Continued Pretraining (CAP-CPT). Our automated pipeline parses policy documents to extract key specifications, grouping them into factual, behavioral, and conditional categories, and isolating complex conditions that drive workflow complexity. This guides targeted data synthesis and enables agents to internalize policy information through an autoregressive pretraining loss. Experiments show CAP-CPT improves SFT baselines in all settings, with up to 41% and 22% gains on Qwen-3-32B, achieving 97.3% prompt length reduction on CC-Gen and further enhancing tau-Bench with minimal SFT data. 

---
# video-SALMONN S: Streaming Audio-Visual LLMs Beyond Length Limits via Memory 

**Authors**: Guangzhi Sun, Yixuan Li, Xiaodong Wu, Yudong Yang, Wei Li, Zejun Ma, Chao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.11129)  

**Abstract**: Continuous, high-frame-rate, high-resolution processing of long video streams is critical for future AI agents, yet current video-understanding LLMs struggle to scale. Offline, fixed-frame-number methods require the stream length to adapt frame rates; streaming methods constrain memory by merging or discarding tokens, losing information. We propose video-SALMONN S, a streaming audio-visual LLM that, to our knowledge, is the first to process 3-hour videos at 1 FPS and 360p resolution under a fixed memory budget. Our model introduces (i) a test-time-training (TTT) memory module that continually updates token representations to capture long-range dependencies by replacing token merging, and (ii) a prompt-dependent memory reader that selectively retrieves context-relevant content from fixed-size memory. The TTT module is optimised with a Hessian-free conjugate-gradient procedure (TTT_HF) for efficient adaptation. On long-video benchmarks (Video-MME, LVBench, VideoEvalPro), video-SALMONN S sustains high-quality understanding on multi-hour videos with 10k frames and 1M tokens. Our 8B-parameter model achieves 74.2% overall and 67.8% on the Video-MME long split, outperforming both offline and streaming baselines. 

---
