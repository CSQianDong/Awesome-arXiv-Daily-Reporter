# Memoria: A Scalable Agentic Memory Framework for Personalized Conversational AI 

**Authors**: Samarth Sarin, Lovepreet Singh, Bhaskarjit Sarmah, Dhagash Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2512.12686)  

**Abstract**: Agentic memory is emerging as a key enabler for large language models (LLM) to maintain continuity, personalization, and long-term context in extended user interactions, critical capabilities for deploying LLMs as truly interactive and adaptive agents. Agentic memory refers to the memory that provides an LLM with agent-like persistence: the ability to retain and act upon information across conversations, similar to how a human would. We present Memoria, a modular memory framework that augments LLM-based conversational systems with persistent, interpretable, and context-rich memory. Memoria integrates two complementary components: dynamic session-level summarization and a weighted knowledge graph (KG)-based user modelling engine that incrementally captures user traits, preferences, and behavioral patterns as structured entities and relationships. This hybrid architecture enables both short-term dialogue coherence and long-term personalization while operating within the token constraints of modern LLMs. We demonstrate how Memoria enables scalable, personalized conversational artificial intelligence (AI) by bridging the gap between stateless LLM interfaces and agentic memory systems, offering a practical solution for industry applications requiring adaptive and evolving user experiences. 

---
# Uncovering the Role of Initial Saliency in U-Shaped Attention Bias: Scaling Initial Token Weight for Enhanced Long-Text Processing 

**Authors**: Zewen Qiang, Sendong Zhao, Haochun Wang, Bing Qin, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.13109)  

**Abstract**: Large language models (LLMs) have demonstrated strong performance on a variety of natural language processing (NLP) tasks. However, they often struggle with long-text sequences due to the ``lost in the middle'' phenomenon. This issue has been shown to arise from a U-shaped attention bias, where attention is disproportionately focused on the beginning and end of a text, leaving the middle section underrepresented. While previous studies have attributed this bias to position encoding, our research first identifies an additional factor: initial saliency. It means that in the attention computation for each token, tokens with higher attention weights relative to the initial token tend to receive more attention in the prediction of the next token. We further find that utilizing this property by scaling attention weight between the initial token and others improves the model's ability to process long contexts, achieving a maximum improvement of 3.6\% in MDQA dataset. Moreover, combining this approach with existing methods to reduce position encoding bias further enhances performance, achieving a maximum improvement of 3.4\% in KV-Retrieval tasks. 

---
# Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings 

**Authors**: Yoav Gelberg, Koshi Eguchi, Takuya Akiba, Edoardo Cetin  

**Link**: [PDF](https://arxiv.org/pdf/2512.12167)  

**Abstract**: So far, expensive finetuning beyond the pretraining sequence length has been a requirement for effectively extending the context of language models (LM). In this work, we break this key bottleneck by Dropping the Positional Embeddings of LMs after training (DroPE). Our simple method is motivated by three key theoretical and empirical observations. First, positional embeddings (PEs) serve a crucial role during pretraining, providing an important inductive bias that significantly facilitates convergence. Second, over-reliance on this explicit positional information is also precisely what prevents test-time generalization to sequences of unseen length, even when using popular PE-scaling methods. Third, positional embeddings are not an inherent requirement of effective language modeling and can be safely removed after pretraining, following a short recalibration phase. Empirically, DroPE yields seamless zero-shot context extension without any long-context finetuning, quickly adapting pretrained LMs without compromising their capabilities in the original training context. Our findings hold across different models and dataset sizes, far outperforming previous specialized architectures and established rotary positional embedding scaling methods. 

---
# Hold Onto That Thought: Assessing KV Cache Compression On Reasoning 

**Authors**: Minghui Liu, Aadi Palnitkar, Tahseen Rabbani, Hyunwoo Jae, Kyle Rui Sang, Dixi Yao, Shayan Shabihi, Fuheng Zhao, Tian Li, Ce Zhang, Furong Huang, Kunpeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.12008)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable performance on long-context tasks, but are often bottlenecked by memory constraints. Namely, the KV cache, which is used to significantly speed up attention computations, grows linearly with context length. A suite of compression algorithms has been introduced to alleviate cache growth by evicting unimportant tokens. However, several popular strategies are targeted towards the prefill phase, i.e., processing long prompt context, and their performance is rarely assessed on reasoning tasks requiring long decoding. In particular, short but complex prompts, such as those in benchmarks like GSM8K and MATH500, often benefit from multi-step reasoning and self-reflection, resulting in thinking sequences thousands of tokens long. In this work, we benchmark the performance of several popular compression strategies on long-reasoning tasks. For the non-reasoning Llama-3.1-8B-Instruct, we determine that no singular strategy fits all, and that performance is heavily influenced by dataset type. However, we discover that H2O and our decoding-enabled variant of SnapKV are dominant strategies for reasoning models, indicating the utility of heavy-hitter tracking for reasoning traces. We also find that eviction strategies at low budgets can produce longer reasoning traces, revealing a tradeoff between cache size and inference costs. 

---
# KV Cache Recycling to Expand Usable Context Capacity in Low Parameter LLMs 

**Authors**: Prashant Pandey  

**Link**: [PDF](https://arxiv.org/pdf/2512.11851)  

**Abstract**: Whether attention key value (KV) states computed for one prompt for a small LLM can be reused to accelerate inference on a new similar prompt, giving an increase to the space to its context memory using an approach called token recycling. Using a standard Hugging Face setup with DialoGPT-medium (a 345M parameter GPT-2 style decoder trained on 147M Reddit exchanges, 2005 to 2017) as the testbed, we build a cache of past activations and get entries by sentence embeddings, then reuse cached past key values when the cached prompt is an exact prefix of the new input. We compare recycled vs. baseline runs on latency and output fidelity, and log reuse depth in tokens. Reproducibility requires no model modifications, cached KVs are serialized to the CPU, reloaded, and supplied to the generate function to continue decoding from the cached prefix. In tests, we observe consistent speedups when prefix overlap exists, with no material degradation in output semantics, and when overlap is absent, behavior matches baseline. 

---
# AIR: Post-training Data Selection for Reasoning via Attention Head Influence 

**Authors**: Jinrui Liu, Jeff Wu, Xuanguang Pan, Gavin Cheung, Shuai Ma, Chongyang Tao  

**Link**: [PDF](https://arxiv.org/pdf/2512.13279)  

**Abstract**: LLMs achieve remarkable multi-step reasoning capabilities, yet effectively transferring these skills via post-training distillation remains challenging. Existing data selection methods, ranging from manual curation to heuristics based on length, entropy, or overall loss, fail to capture the causal importance of individual reasoning steps, limiting distillation efficiency. To address this, we propose Attention Influence for Reasoning (AIR), a principled, unsupervised and training-free framework that leverages mechanistic insights of the retrieval head to select high-value post-training data. AIR first identifies reasoning-critical attention heads of an off-the-shelf model, then constructs a weakened reference model with disabled head influence, and finally quantifies the resulting loss divergence as the Attention Influence Score. This score enables fine-grained assessment at both the step and sample levels, supporting step-level weighted fine-tuning and global sample selection. Experiments across multiple reasoning benchmarks show that AIR consistently improves reasoning accuracy, surpassing heuristic baselines and effectively isolating the most critical steps and samples. Our work establishes a mechanism-driven, data-efficient approach for reasoning distillation in LLMs. 

---
# What Matters in Evaluating Book-Length Stories? A Systematic Study of Long Story Evaluation 

**Authors**: Dingyi Yang, Qin Jin  

**Link**: [PDF](https://arxiv.org/pdf/2512.12839)  

**Abstract**: In this work, we conduct systematic research in a challenging area: the automatic evaluation of book-length stories (>100K tokens). Our study focuses on two key questions: (1) understanding which evaluation aspects matter most to readers, and (2) exploring effective methods for evaluating lengthy stories. We introduce the first large-scale benchmark, LongStoryEval, comprising 600 newly published books with an average length of 121K tokens (maximum 397K). Each book includes its average rating and multiple reader reviews, presented as critiques organized by evaluation aspects. By analyzing all user-mentioned aspects, we propose an evaluation criteria structure and conduct experiments to identify the most significant aspects among the 8 top-level criteria. For evaluation methods, we compare the effectiveness of three types: aggregation-based, incremental-updated, and summary-based evaluations. Our findings reveal that aggregation- and summary-based evaluations perform better, with the former excelling in detail assessment and the latter offering greater efficiency. Building on these insights, we further propose NovelCritique, an 8B model that leverages the efficient summary-based framework to review and score stories across specified aspects. NovelCritique outperforms commercial models like GPT-4o in aligning with human evaluations. Our datasets and codes are available at this https URL. 

---
# Persistent Personas? Role-Playing, Instruction Following, and Safety in Extended Interactions 

**Authors**: Pedro Henrique Luz de Araujo, Michael A. Hedderich, Ali Modarressi, Hinrich Schuetze, Benjamin Roth  

**Link**: [PDF](https://arxiv.org/pdf/2512.12775)  

**Abstract**: Persona-assigned large language models (LLMs) are used in domains such as education, healthcare, and sociodemographic simulation. Yet, they are typically evaluated only in short, single-round settings that do not reflect real-world usage. We introduce an evaluation protocol that combines long persona dialogues (over 100 rounds) and evaluation datasets to create dialogue-conditioned benchmarks that can robustly measure long-context effects. We then investigate the effects of dialogue length on persona fidelity, instruction-following, and safety of seven state-of-the-art open- and closed-weight LLMs. We find that persona fidelity degrades over the course of dialogues, especially in goal-oriented conversations, where models must sustain both persona fidelity and instruction following. We identify a trade-off between fidelity and instruction following, with non-persona baselines initially outperforming persona-assigned models; as dialogues progress and fidelity fades, persona responses become increasingly similar to baseline responses. Our findings highlight the fragility of persona applications in extended interactions and our work provides a protocol to systematically measure such failures. 

---
# QwenLong-L1.5: Post-Training Recipe for Long-Context Reasoning and Memory Management 

**Authors**: Weizhou Shen, Ziyi Yang, Chenliang Li, Zhiyuan Lu, Miao Peng, Huashan Sun, Yingcheng Shi, Shengyi Liao, Shaopeng Lai, Bo Zhang, Dayiheng Liu, Fei Huang, Jingren Zhou, Ming Yan  

**Link**: [PDF](https://arxiv.org/pdf/2512.12967)  

**Abstract**: We introduce QwenLong-L1.5, a model that achieves superior long-context reasoning capabilities through systematic post-training innovations. The key technical breakthroughs of QwenLong-L1.5 are as follows: (1) Long-Context Data Synthesis Pipeline: We develop a systematic synthesis framework that generates challenging reasoning tasks requiring multi-hop grounding over globally distributed evidence. By deconstructing documents into atomic facts and their underlying relationships, and then programmatically composing verifiable reasoning questions, our approach creates high-quality training data at scale, moving substantially beyond simple retrieval tasks to enable genuine long-range reasoning capabilities. (2) Stabilized Reinforcement Learning for Long-Context Training: To overcome the critical instability in long-context RL, we introduce task-balanced sampling with task-specific advantage estimation to mitigate reward bias, and propose Adaptive Entropy-Controlled Policy Optimization (AEPO) that dynamically regulates exploration-exploitation trade-offs. (3) Memory-Augmented Architecture for Ultra-Long Contexts: Recognizing that even extended context windows cannot accommodate arbitrarily long sequences, we develop a memory management framework with multi-stage fusion RL training that seamlessly integrates single-pass reasoning with iterative memory-based processing for tasks exceeding 4M tokens. Based on Qwen3-30B-A3B-Thinking, QwenLong-L1.5 achieves performance comparable to GPT-5 and Gemini-2.5-Pro on long-context reasoning benchmarks, surpassing its baseline by 9.90 points on average. On ultra-long tasks (1M~4M tokens), QwenLong-L1.5's memory-agent framework yields a 9.48-point gain over the agent baseline. Additionally, the acquired long-context reasoning ability translates to enhanced performance in general domains like scientific reasoning, memory tool using, and extended dialogue. 

---
# BLASST: Dynamic BLocked Attention Sparsity via Softmax Thresholding 

**Authors**: Jiayi Yuan, Cameron Shinn, Kai Xu, Jingze Cui, George Klimiashvili, Guangxuan Xiao, Perkz Zheng, Bo Li, Yuxin Zhou, Zhouhai Ye, Weijie You, Tian Zheng, Dominic Brown, Pengbo Wang, Richard Cai, Julien Demouth, John D. Owens, Xia Hu, Song Han, Timmy Liu, Huizi Mao  

**Link**: [PDF](https://arxiv.org/pdf/2512.12087)  

**Abstract**: The growing demand for long-context inference capabilities in Large Language Models (LLMs) has intensified the computational and memory bottlenecks inherent to the standard attention mechanism. To address this challenge, we introduce BLASST, a drop-in sparse attention method that dynamically prunes the attention matrix without any pre-computation or proxy scores. Our method uses a fixed threshold and existing information from online softmax to identify negligible attention scores, skipping softmax computation, Value block loading, and the subsequent matrix multiplication. This fits seamlessly into existing FlashAttention kernel designs with negligible latency overhead. The approach is applicable to both prefill and decode stages across all attention variants (MHA, GQA, MQA, and MLA), providing a unified solution for accelerating long-context inference. We develop an automated calibration procedure that reveals a simple inverse relationship between optimal threshold and context length, enabling robust deployment across diverse scenarios. Maintaining high accuracy, we demonstrate a 1.62x speedup for prefill at 74.7% sparsity and a 1.48x speedup for decode at 73.2% sparsity on modern GPUs. Furthermore, we explore sparsity-aware training as a natural extension, showing that models can be trained to be inherently more robust to sparse attention patterns, pushing the accuracy-sparsity frontier even further. 

---
# Intelligent Scientific Literature Explorer using Machine Learning (ISLE) 

**Authors**: Sina Jani, Arman Heidari, Amirmohammad Anvari, Zahra Rahimi  

**Link**: [PDF](https://arxiv.org/pdf/2512.12760)  

**Abstract**: The rapid acceleration of scientific publishing has created substantial challenges for researchers attempting to discover, contextualize, and interpret relevant literature. Traditional keyword-based search systems provide limited semantic understanding, while existing AI-driven tools typically focus on isolated tasks such as retrieval, clustering, or bibliometric visualization. This paper presents an integrated system for scientific literature exploration that combines large-scale data acquisition, hybrid retrieval, semantic topic modeling, and heterogeneous knowledge graph construction. The system builds a comprehensive corpus by merging full-text data from arXiv with structured metadata from OpenAlex. A hybrid retrieval architecture fuses BM25 lexical search with embedding-based semantic search using Reciprocal Rank Fusion. Topic modeling is performed on retrieved results using BERTopic or non-negative matrix factorization depending on computational resources. A knowledge graph unifies papers, authors, institutions, countries, and extracted topics into an interpretable structure. The system provides a multi-layered exploration environment that reveals not only relevant publications but also the conceptual and relational landscape surrounding a query. Evaluation across multiple queries demonstrates improvements in retrieval relevance, topic coherence, and interpretability. The proposed framework contributes an extensible foundation for AI-assisted scientific discovery. 

---
# VideoARM: Agentic Reasoning over Hierarchical Memory for Long-Form Video Understanding 

**Authors**: Yufei Yin, Qianke Meng, Minghao Chen, Jiajun Ding, Zhenwei Shao, Zhou Yu  

**Link**: [PDF](https://arxiv.org/pdf/2512.12360)  

**Abstract**: Long-form video understanding remains challenging due to the extended temporal structure and dense multimodal cues. Despite recent progress, many existing approaches still rely on hand-crafted reasoning pipelines or employ token-consuming video preprocessing to guide MLLMs in autonomous reasoning. To overcome these limitations, we introduce VideoARM, an Agentic Reasoning-over-hierarchical-Memory paradigm for long-form video understanding. Instead of static, exhaustive preprocessing, VideoARM performs adaptive, on-the-fly agentic reasoning and memory construction. Specifically, VideoARM performs an adaptive and continuous loop of observing, thinking, acting, and memorizing, where a controller autonomously invokes tools to interpret the video in a coarse-to-fine manner, thereby substantially reducing token consumption. In parallel, a hierarchical multimodal memory continuously captures and updates multi-level clues throughout the operation of the agent, providing precise contextual information to support the controller in decision-making. Experiments on prevalent benchmarks demonstrate that VideoARM outperforms the state-of-the-art method, DVD, while significantly reducing token consumption for long-form videos. 

---
