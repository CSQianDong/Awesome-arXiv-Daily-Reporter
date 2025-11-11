# Retrieval Quality at Context Limit 

**Authors**: Max McKinnon  

**Link**: [PDF](https://arxiv.org/pdf/2511.05850)  

**Abstract**: The ability of large language models (LLMs) to recall and retrieve information from long contexts is critical for many real-world applications. Prior work (Liu et al., 2023) reported that LLMs suffer significant drops in retrieval accuracy for facts placed in the middle of large contexts, an effect known as "Lost in the Middle" (LITM). We find the model Gemini 2.5 Flash can answer needle-in-a-haystack questions with great accuracy regardless of document position including when the document is nearly at the input context limit. Our results suggest that the "Lost in the Middle" effect is not present for simple factoid Q\&A in Gemini 2.5 Flash, indicating substantial improvements in long-context retrieval. 

---
# Q-RAG: Long Context Multi-step Retrieval via Value-based Embedder Training 

**Authors**: Artyom Sorokin, Nazar Buzun, Alexander Anokhin, Oleg Inozemcev, Egor Vedernikov, Petr Anokhin, Mikhail Burtsev, Trushkov Alexey, Yin Wenshuai, Evgeny Burnaev  

**Link**: [PDF](https://arxiv.org/pdf/2511.07328)  

**Abstract**: Retrieval-Augmented Generation (RAG) methods enhance LLM performance by efficiently filtering relevant context for LLMs, reducing hallucinations and inference cost. However, most existing RAG methods focus on single-step retrieval, which is often insufficient for answering complex questions that require multi-step search. Recently, multi-step retrieval approaches have emerged, typically involving the fine-tuning of small LLMs to perform multi-step retrieval. This type of fine-tuning is highly resource-intensive and does not enable the use of larger LLMs. In this work, we propose Q-RAG, a novel approach that fine-tunes the Embedder model for multi-step retrieval using reinforcement learning (RL). Q-RAG offers a competitive, resource-efficient alternative to existing multi-step retrieval methods for open-domain question answering and achieves state-of-the-art results on the popular long-context benchmarks Babilong and RULER for contexts up to 10M tokens. 

---
# Make It Long, Keep It Fast: End-to-End 10k-Sequence Modeling at Billion Scale on Douyin 

**Authors**: Lin Guan, Jia-Qi Yang, Zhishan Zhao, Beichuan Zhang, Bo Sun, Xuanyuan Luo, Jinan Ni, Xiaowen Li, Yuhang Qi, Zhifang Fan, Hangyu Wang, Qiwei Chen, Yi Cheng, Feng Zhang, Xiao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.06077)  

**Abstract**: Short-video recommenders such as Douyin must exploit extremely long user histories without breaking latency or cost budgets. We present an end-to-end system that scales long-sequence modeling to 10k-length histories in production. First, we introduce Stacked Target-to-History Cross Attention (STCA), which replaces history self-attention with stacked cross-attention from the target to the history, reducing complexity from quadratic to linear in sequence length and enabling efficient end-to-end training. Second, we propose Request Level Batching (RLB), a user-centric batching scheme that aggregates multiple targets for the same user/request to share the user-side encoding, substantially lowering sequence-related storage, communication, and compute without changing the learning objective. Third, we design a length-extrapolative training strategy -- train on shorter windows, infer on much longer ones -- so the model generalizes to 10k histories without additional training cost. Across offline and online experiments, we observe predictable, monotonic gains as we scale history length and model capacity, mirroring the scaling law behavior observed in large language models. Deployed at full traffic on Douyin, our system delivers significant improvements on key engagement metrics while meeting production latency, demonstrating a practical path to scaling end-to-end long-sequence recommendation to the 10k regime. 

---
# Transformers Provably Learn Chain-of-Thought Reasoning with Length Generalization 

**Authors**: Yu Huang, Zixin Wen, Aarti Singh, Yuejie Chi, Yuxin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.07378)  

**Abstract**: The ability to reason lies at the core of artificial intelligence (AI), and challenging problems usually call for deeper and longer reasoning to tackle. A crucial question about AI reasoning is whether models can extrapolate learned reasoning patterns to solve harder tasks with longer chain-of-thought (CoT). In this work, we present a theoretical analysis of transformers learning on synthetic state-tracking tasks with gradient descent. We mathematically prove how the algebraic structure of state-tracking problems governs the degree of extrapolation of the learned CoT. Specifically, our theory characterizes the length generalization of transformers through the mechanism of attention concentration, linking the retrieval robustness of the attention layer to the state-tracking task structure of long-context reasoning. Moreover, for transformers with limited reasoning length, we prove that a recursive self-training scheme can progressively extend the range of solvable problem lengths. To our knowledge, we provide the first optimization guarantee that constant-depth transformers provably learn $\mathsf{NC}^1$-complete problems with CoT, significantly going beyond prior art confined in $\mathsf{TC}^0$, unless the widely held conjecture $\mathsf{TC}^0 \neq \mathsf{NC}^1$ fails. Finally, we present a broad set of experiments supporting our theoretical results, confirming the length generalization behaviors and the mechanism of attention concentration. 

---
# Discourse Graph Guided Document Translation with Large Language Models 

**Authors**: Viet-Thanh Pham, Minghan Wang, Hao-Han Liao, Thuy-Trang Vu  

**Link**: [PDF](https://arxiv.org/pdf/2511.07230)  

**Abstract**: Adapting large language models to full document translation remains challenging due to the difficulty of capturing long-range dependencies and preserving discourse coherence throughout extended texts. While recent agentic machine translation systems mitigate context window constraints through multi-agent orchestration and persistent memory, they require substantial computational resources and are sensitive to memory retrieval strategies. We introduce TransGraph, a discourse-guided framework that explicitly models inter-chunk relationships through structured discourse graphs and selectively conditions each translation segment on relevant graph neighbourhoods rather than relying on sequential or exhaustive context. Across three document-level MT benchmarks spanning six languages and diverse domains, TransGraph consistently surpasses strong baselines in translation quality and terminology consistency while incurring significantly lower token overhead. 

---
# Evaluation of retrieval-based QA on QUEST-LOFT 

**Authors**: Nathan Scales, Nathanael Schärli, Olivier Bousquet  

**Link**: [PDF](https://arxiv.org/pdf/2511.06125)  

**Abstract**: Despite the popularity of retrieval-augmented generation (RAG) as a solution for grounded QA in both academia and industry, current RAG methods struggle with questions where the necessary information is distributed across many documents or where retrieval needs to be combined with complex reasoning. Recently, the LOFT study has shown that this limitation also applies to approaches based on long-context language models, with the QUEST benchmark exhibiting particularly large headroom. In this paper, we provide an in-depth analysis of the factors contributing to the poor performance on QUEST-LOFT, publish updated numbers based on a thorough human evaluation, and demonstrate that RAG can be optimized to significantly outperform long-context approaches when combined with a structured output format containing reasoning and evidence, optionally followed by answer re-verification. 

---
# MoSKA: Mixture of Shared KV Attention for Efficient Long-Sequence LLM Inference 

**Authors**: Myunghyun Rhee, Sookyung Choi, Euiseok Kim, Joonseop Sim, Youngpyo Joo, Hoshik Kim  

**Link**: [PDF](https://arxiv.org/pdf/2511.06010)  

**Abstract**: The escalating context length in Large Language Models (LLMs) creates a severe performance bottleneck around the Key-Value (KV) cache, whose memory-bound nature leads to significant GPU under-utilization. This paper introduces Mixture of Shared KV Attention (MoSKA), an architecture that addresses this challenge by exploiting the heterogeneity of context data. It differentiates between per-request unique and massively reused shared sequences. The core of MoSKA is a novel Shared KV Attention mechanism that transforms the attention on shared data from a series of memory-bound GEMV operations into a single, compute-bound GEMM by batching concurrent requests. This is supported by an MoE-inspired sparse attention strategy that prunes the search space and a tailored Disaggregated Infrastructure that specializes hardware for unique and shared data. This comprehensive approach demonstrates a throughput increase of up to 538.7x over baselines in workloads with high context sharing, offering a clear architectural path toward scalable LLM inference. 

---
# Personalized Chain-of-Thought Summarization of Financial News for Investor Decision Support 

**Authors**: Tianyi Zhang, Mu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.05508)  

**Abstract**: Financial advisors and investors struggle with information overload from financial news, where irrelevant content and noise obscure key market signals and hinder timely investment decisions. To address this, we propose a novel Chain-of-Thought (CoT) summarization framework that condenses financial news into concise, event-driven summaries. The framework integrates user-specified keywords to generate personalized outputs, ensuring that only the most relevant contexts are highlighted. These personalized summaries provide an intermediate layer that supports language models in producing investor-focused narratives, bridging the gap between raw news and actionable insights. 

---
# Learning to Focus: Focal Attention for Selective and Scalable Transformers 

**Authors**: Dhananjay Ram, Wei Xia, Stefano Soatto  

**Link**: [PDF](https://arxiv.org/pdf/2511.06818)  

**Abstract**: Attention is a core component of transformer architecture, whether encoder-only, decoder-only, or encoder-decoder model. However, the standard softmax attention often produces noisy probability distribution, which can impair effective feature selection at every layer of these models, particularly for long contexts. We propose Focal Attention, a simple yet effective modification that sharpens the attention distribution by controlling the softmax temperature, either as a fixed hyperparameter or as a learnable parameter during training. This sharpening enables the model to concentrate on the most relevant tokens while suppressing irrelevant ones. Empirically, Focal Attention scales more favorably than standard transformer with respect to model size, training data, and context length. Across diverse benchmarks, it achieves the same accuracy with up to 42% fewer parameters or 33% less training data. On long-context tasks, it delivers substantial relative improvements ranging from 17% to 82%, demonstrating its effectiveness in real world applications. 

---
# BookAsSumQA: An Evaluation Framework for Aspect-Based Book Summarization via Question Answering 

**Authors**: Ryuhei Miyazato, Ting-Ruen Wei, Xuyang Wu, Hsin-Tai Wu, Kei Harada  

**Link**: [PDF](https://arxiv.org/pdf/2511.06183)  

**Abstract**: Aspect-based summarization aims to generate summaries that highlight specific aspects of a text, enabling more personalized and targeted summaries. However, its application to books remains unexplored due to the difficulty of constructing reference summaries for long text. To address this challenge, we propose BookAsSumQA, a QA-based evaluation framework for aspect-based book summarization. BookAsSumQA automatically generates aspect-specific QA pairs from a narrative knowledge graph to evaluate summary quality based on its question-answering performance. Our experiments using BookAsSumQA revealed that while LLM-based approaches showed higher accuracy on shorter texts, RAG-based methods become more effective as document length increases, making them more efficient and practical for aspect-based book summarization. 

---
