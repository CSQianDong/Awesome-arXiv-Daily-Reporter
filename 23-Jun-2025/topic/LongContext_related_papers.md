# Cache Me If You Can: How Many KVs Do You Need for Effective Long-Context LMs? 

**Authors**: Adithya Bhaskar, Alexander Wettig, Tianyu Gao, Yihe Dong, Danqi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.17121)  

**Abstract**: Language models handle increasingly long contexts for tasks such as book summarization, but this leads to growing memory costs for the key-value (KV) cache. Many prior works have proposed ways of discarding KVs from memory, but their approaches are tailored to favorable settings, obscuring caveats like high peak memory and performance degradation, and a fair comparison between methods is difficult. In this paper, we propose the *KV footprint* as a unified metric, which accounts for both the amount of KV entries stored and their lifespan in memory. We evaluate methods based on the smallest footprint they attain while preserving performance in both long-context understanding and generation, with context lengths of up to 128K tokens. This metric reveals the high peak memory of prior KV eviction methods. One class of methods -- *post-fill eviction* -- has a high footprint due to being incompatible with eviction during pre-filling. We adapt these methods to be able to evict KVs during pre-filling, achieving substantially lower KV footprints. We then turn to *recency eviction* methods, wherein we propose PruLong, an end-to-end optimization method for learning which attention heads need to retain the full KV cache and which do not. PruLong saves memory while preserving long-context performance, achieving 12% smaller KV footprint than prior methods while retaining performance in challenging recall tasks. Our paper clarifies the complex tangle of long-context inference methods and paves the way for future development to minimize the KV footprint. 

---
# Long-Context Generalization with Sparse Attention 

**Authors**: Pavlo Vasylenko, Marcos Treviso, André F. T. Martins  

**Link**: [PDF](https://arxiv.org/pdf/2506.16640)  

**Abstract**: Transformer-based architectures traditionally employ softmax to compute attention weights, which produces dense distributions over all tokens in a sequence. While effective in many settings, this density has been shown to be detrimental for tasks that demand precise focus on fixed-size patterns: as sequence length increases, non-informative tokens accumulate attention probability mass, leading to dispersion and representational collapse. We show in this paper that sparse attention mechanisms using $\alpha$-entmax can avoid these issues, due to their ability to assign exact zeros to irrelevant tokens. Furthermore, we introduce Adaptive-Scalable Entmax (ASEntmax), which endows $\alpha$-entmax with a learnable temperature parameter, allowing the attention distribution to interpolate between sparse (pattern-focused) and dense (softmax-like) regimes. Finally, we show that the ability to locate and generalize fixed-size patterns can be further improved through a careful design of position encodings, which impacts both dense and sparse attention methods. By integrating ASEntmax into standard transformer layers alongside proper positional encodings, we show that our models greatly outperform softmax, scalable softmax, and fixed-temperature $\alpha$-entmax baselines on long-context generalization. 

---
# LegiGPT: Party Politics and Transport Policy with Large Language Model 

**Authors**: Hyunsoo Yun, Eun Hak Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.16692)  

**Abstract**: Given the significant influence of lawmakers' political ideologies on legislative decision-making, understanding their impact on policymaking is critically important. We introduce a novel framework, LegiGPT, which integrates a large language model (LLM) with explainable artificial intelligence (XAI) to analyze transportation-related legislative proposals. LegiGPT employs a multi-stage filtering and classification pipeline using zero-shot prompting with GPT-4. Using legislative data from South Korea's 21st National Assembly, we identify key factors - including sponsor characteristics, political affiliations, and geographic variables - that significantly influence transportation policymaking. The LLM was used to classify transportation-related bill proposals through a stepwise filtering process based on keywords, phrases, and contextual relevance. XAI techniques were then applied to examine relationships between party affiliation and associated attributes. The results reveal that the number and proportion of conservative and progressive sponsors, along with district size and electoral population, are critical determinants shaping legislative outcomes. These findings suggest that both parties contributed to bipartisan legislation through different forms of engagement, such as initiating or supporting proposals. This integrated approach provides a valuable tool for understanding legislative dynamics and guiding future policy development, with broader implications for infrastructure planning and governance. 

---
# StoryWriter: A Multi-Agent Framework for Long Story Generation 

**Authors**: Haotian Xia, Hao Peng, Yunjia Qi, Xiaozhi Wang, Bin Xu, Lei Hou, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.16445)  

**Abstract**: Long story generation remains a challenge for existing large language models (LLMs), primarily due to two main factors: (1) discourse coherence, which requires plot consistency, logical coherence, and completeness in the long-form generation, and (2) narrative complexity, which requires an interwoven and engaging narrative. To address these challenges, we propose StoryWriter, a multi-agent story generation framework, which consists of three main modules: (1) outline agent, which generates event-based outlines containing rich event plots, character, and event-event relationships. (2) planning agent, which further details events and plans which events should be written in each chapter to maintain an interwoven and engaging story. (3) writing agent, which dynamically compresses the story history based on the current event to generate and reflect new plots, ensuring the coherence of the generated story. We conduct both human and automated evaluation, and StoryWriter significantly outperforms existing story generation baselines in both story quality and length. Furthermore, we use StoryWriter to generate a dataset, which contains about $6,000$ high-quality long stories, with an average length of $8,000$ words. We train the model Llama3.1-8B and GLM4-9B using supervised fine-tuning on LongStory and develop StoryWriter_GLM and StoryWriter_GLM, which demonstrates advanced performance in long story generation. 

---
# When Does Divide and Conquer Work for Long Context LLM? A Noise Decomposition Framework 

**Authors**: Zhen Xu, Shang Zhu, Jue Wang, Junlin Wang, Ben Athiwaratkun, Chi Wang, James Zou, Ce Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.16411)  

**Abstract**: We investigate the challenge of applying Large Language Models (LLMs) to long texts. We propose a theoretical framework that distinguishes the failure modes of long context tasks into three categories: cross-chunk dependence (task noise), confusion that grows with context size (model noise), and the imperfect integration of partial results (aggregator noise). Under this view, we analyze when it is effective to use multi-agent chunking, i.e., dividing a length sequence into smaller chunks and aggregating the processed results of each chunk. Our experiments on tasks such as retrieval, question answering, and summarization confirm both the theoretical analysis and the conditions that favor multi-agent chunking. By exploring superlinear model noise growth with input length, we also explain why, for large inputs, a weaker model configured with chunk-based processing can surpass a more advanced model like GPT4o applied in a single shot. Overall, we present a principled understanding framework and our results highlight a direct pathway to handling long contexts in LLMs with carefully managed chunking and aggregator strategies. 

---
# Enhancing Document-Level Question Answering via Multi-Hop Retrieval-Augmented Generation with LLaMA 3 

**Authors**: Xinyue Huang, Ziqi Lin, Fang Sun, Wenchao Zhang, Kejian Tong, Yunbo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.16037)  

**Abstract**: This paper presents a novel Retrieval-Augmented Generation (RAG) framework tailored for complex question answering tasks, addressing challenges in multi-hop reasoning and contextual understanding across lengthy documents. Built upon LLaMA 3, the framework integrates a dense retrieval module with advanced context fusion and multi-hop reasoning mechanisms, enabling more accurate and coherent response generation. A joint optimization strategy combining retrieval likelihood and generation cross-entropy improves the model's robustness and adaptability. Experimental results show that the proposed system outperforms existing retrieval-augmented and generative baselines, confirming its effectiveness in delivering precise, contextually grounded answers. 

---
# From General to Targeted Rewards: Surpassing GPT-4 in Open-Ended Long-Context Generation 

**Authors**: Zhihan Guo, Jiele Wu, Wenqian Cui, Yifei Zhang, Minda Hu, Yufei Wang, Irwin King  

**Link**: [PDF](https://arxiv.org/pdf/2506.16024)  

**Abstract**: Current research on long-form context in Large Language Models (LLMs) primarily focuses on the understanding of long-contexts, the Open-ended Long Text Generation (Open-LTG) remains insufficiently explored. Training a long-context generation model requires curation of gold standard reference data, which is typically nonexistent for informative Open-LTG tasks. However, previous methods only utilize general assessments as reward signals, which limits accuracy. To bridge this gap, we introduce ProxyReward, an innovative reinforcement learning (RL) based framework, which includes a dataset and a reward signal computation method. Firstly, ProxyReward Dataset generation is accomplished through simple prompts that enables the model to create automatically, obviating extensive labeled data or significant manual effort. Secondly, ProxyReward Signal offers a targeted evaluation of information comprehensiveness and accuracy for specific questions. The experimental results indicate that our method ProxyReward surpasses even GPT-4-Turbo. It can significantly enhance performance by 20% on the Open-LTG task when training widely used open-source models, while also surpassing the LLM-as-a-Judge approach. Our work presents effective methods to enhance the ability of LLMs to address complex open-ended questions posed by human. 

---
# Learn from the Past: Fast Sparse Indexing for Large Language Model Decoding 

**Authors**: Feiyu Yao, Qian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.15704)  

**Abstract**: As large language models (LLMs) continue to support increasingly longer contexts, the memory demand for key-value (KV) caches during decoding grows rapidly, becoming a critical bottleneck in both GPU memory capacity and PCIe bandwidth. Sparse attention mechanisms alleviate this issue by computing attention weights only for selected key-value pairs. However, their indexing computation typically requires traversing all key vectors, resulting in significant computational and data transfer overhead. To reduce the cost of index retrieval, existing methods often treat each decoding step as an independent process, failing to exploit the temporal correlations embedded in historical decoding information. To this end, we propose LFPS(Learn From the Past for Sparse Indexing), an acceleration method that dynamically constructs sparse indexing candidates based on historical attention patterns. LFPS captures two prevalent trends in decoder attention -vertical patterns (attending to fixed positions) and slash patterns (attending to relative positions) -and incorporates a positional expansion strategy to effectively predict the Top-k indices for the current step. We validate LFPS on challenging long-context benchmarks such as LongBench-RULER, using Llama-3.1-8B-Instruct as the base model. Experimental results show that LFPS achieves up to 22.8$\times$ speedup over full attention and 9.6$\times$ speedup over exact Top-k retrieval on an RTX 4090 GPU and a single CPU core of a Xeon Gold 6430, respectively, while preserving generation accuracy. These results demonstrate that LFPS offers a practical and efficient solution for decoding optimization in long-context LLM inference. 

---
# Adaptive Two Sided Laplace Transforms: A Learnable, Interpretable, and Scalable Replacement for Self-Attention 

**Authors**: Andrew Kiruluta  

**Link**: [PDF](https://arxiv.org/pdf/2506.15714)  

**Abstract**: We propose an innovative, learnable two-sided short-time Laplace transform (STLT) mechanism to supplant the traditional self attention in transformer-based LLMs. Our STLT introduces trainable parameters for each Laplace node, enabling end-to-end learning of decay rates , oscillatory frequencies, and window bandwidth T. This flexibility allows the model to dynamically adapt token relevance half lives and frequency responses during training. By selecting S learnable nodes and leveraging fast recursive convolution, we achieve an effective complexity of in time and memory. We further incorporate an efficient FFT-based computation of the relevance matrix and an adaptive node allocation mechanism to dynamically adjust the number of active Laplace nodes. Empirical results on language modeling (WikiText\-103, Project Gutenberg), machine translation (WMT'14 En\-De), and long document question answering (NarrativeQA) demonstrate that our learnable STLT achieves perplexities and scores on par with or better than existing efficient transformers while naturally extending to context lengths exceeding 100k tokens or more limited only by available hardware. Ablation studies confirm the importance of learnable parameters and adaptive node allocation. The proposed approach combines interpretability, through explicit decay and frequency parameters, with scalability and robustness, offering a pathway towards ultra-long-sequence language modeling without the computational bottleneck of self-attention. 

---
# MadaKV: Adaptive Modality-Perception KV Cache Eviction for Efficient Multimodal Long-Context Inference 

**Authors**: Kunxi Li, Zhonghua Jiang, Zhouzhou Shen, Zhaode Wang, Chengfei Lv, Shengyu Zhang, Fan Wu, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15724)  

**Abstract**: This paper introduces MadaKV, a modality-adaptive key-value (KV) cache eviction strategy designed to enhance the efficiency of multimodal large language models (MLLMs) in long-context inference. In multimodal scenarios, attention heads exhibit varying preferences for different modalities, resulting in significant disparities in modality importance across attention heads. Traditional KV cache eviction methods, which are tailored for unimodal settings, fail to capture modality-specific information, thereby yielding suboptimal performance. MadaKV addresses these challenges through two key components: modality preference adaptation and hierarchical compression compensation. By dynamically sensing modality information within attention heads and adaptively retaining critical tokens, MadaKV achieves substantial reductions in KV cache memory footprint and model inference decoding latency (1.3 to 1.5 times improvement) while maintaining high accuracy across various multimodal long-context tasks. Extensive experiments on representative MLLMs and the MileBench benchmark demonstrate the effectiveness of MadaKV compared to existing KV cache eviction methods. 

---
