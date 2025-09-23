# RALLM-POI: Retrieval-Augmented LLM for Zero-shot Next POI Recommendation with Geographical Reranking 

**Authors**: Kunrong Li, Kwan Hui Lim  

**Link**: [PDF](https://arxiv.org/pdf/2509.17066)  

**Abstract**: Next point-of-interest (POI) recommendation predicts a user's next destination from historical movements. Traditional models require intensive training, while LLMs offer flexible and generalizable zero-shot solutions but often generate generic or geographically irrelevant results due to missing trajectory and spatial context. To address these issues, we propose RALLM-POI, a framework that couples LLMs with retrieval-augmented generation and self-rectification. We first propose a Historical Trajectory Retriever (HTR) that retrieves relevant past trajectories to serve as contextual references, which are then reranked by a Geographical Distance Reranker (GDR) for prioritizing spatially relevant trajectories. Lastly, an Agentic LLM Rectifier (ALR) is designed to refine outputs through self-reflection. Without additional training, RALLM-POI achieves substantial accuracy gains across three real-world Foursquare datasets, outperforming both conventional and LLM-based baselines. Code is released at this https URL. 

---
# Dendritic Resonate-and-Fire Neuron for Effective and Efficient Long Sequence Modeling 

**Authors**: Dehao Zhang, Malu Zhang, Shuai Wang, Jingya Wang, Wenjie Wei, Zeyu Ma, Guoqing Wang, Yang Yang, HaiZhou Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.17186)  

**Abstract**: The explosive growth in sequence length has intensified the demand for effective and efficient long sequence modeling. Benefiting from intrinsic oscillatory membrane dynamics, Resonate-and-Fire (RF) neurons can efficiently extract frequency components from input signals and encode them into spatiotemporal spike trains, making them well-suited for long sequence modeling. However, RF neurons exhibit limited effective memory capacity and a trade-off between energy efficiency and training speed on complex temporal tasks. Inspired by the dendritic structure of biological neurons, we propose a Dendritic Resonate-and-Fire (D-RF) model, which explicitly incorporates a multi-dendritic and soma architecture. Each dendritic branch encodes specific frequency bands by utilizing the intrinsic oscillatory dynamics of RF neurons, thereby collectively achieving comprehensive frequency representation. Furthermore, we introduce an adaptive threshold mechanism into the soma structure that adjusts the threshold based on historical spiking activity, reducing redundant spikes while maintaining training efficiency in long sequence tasks. Extensive experiments demonstrate that our method maintains competitive accuracy while substantially ensuring sparse spikes without compromising computational efficiency during training. These results underscore its potential as an effective and efficient solution for long sequence modeling on edge platforms. 

---
# Long document summarization using page specific target text alignment and distilling page importance 

**Authors**: Pushpa Devi, Ayush Agrawal, Ashutosh Dubey, C. Ravindranath Chowdary  

**Link**: [PDF](https://arxiv.org/pdf/2509.16539)  

**Abstract**: The rapid growth of textual data across news, legal, medical, and scientific domains is becoming a challenge for efficiently accessing and understanding large volumes of content. It is increasingly complex for users to consume and extract meaningful information efficiently. Thus, raising the need for summarization. Unlike short document summarization, long document abstractive summarization is resource-intensive, and very little literature is present in this direction. BART is a widely used efficient sequence-to-sequence (seq-to-seq) model. However, when it comes to summarizing long documents, the length of the context window limits its capabilities. We proposed a model called PTS (Page-specific Target-text alignment Summarization) that extends the seq-to-seq method for abstractive summarization by dividing the source document into several pages. PTS aligns each page with the relevant part of the target summary for better supervision. Partial summaries are generated for each page of the document. We proposed another model called PTSPI (Page-specific Target-text alignment Summarization with Page Importance), an extension to PTS where an additional layer is placed before merging the partial summaries into the final summary. This layer provides dynamic page weightage and explicit supervision to focus on the most informative pages. We performed experiments on the benchmark dataset and found that PTSPI outperformed the SOTA by 6.32\% in ROUGE-1 and 8.08\% in ROUGE-2 scores. 

---
# Towards Adaptive Context Management for Intelligent Conversational Question Answering 

**Authors**: Manoj Madushanka Perera, Adnan Mahmood, Kasun Eranda Wijethilake, Quan Z. Sheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.17829)  

**Abstract**: This particular paper introduces an Adaptive Context Management (ACM) framework for the Conversational Question Answering (ConvQA) systems. The key objective of the ACM framework is to optimize the use of the conversation history by dynamically managing context for maximizing the relevant information provided to a ConvQA model within its token limit. Our approach incorporates a Context Manager (CM) Module, a Summarization (SM) Module, and an Entity Extraction (EE) Module in a bid to handle the conversation history efficaciously. The CM Module dynamically adjusts the context size, thereby preserving the most relevant and recent information within a model's token limit. The SM Module summarizes the older parts of the conversation history via a sliding window. When the summarization window exceeds its limit, the EE Module identifies and retains key entities from the oldest conversation turns. Experimental results demonstrate the effectiveness of our envisaged framework in generating accurate and contextually appropriate responses, thereby highlighting the potential of the ACM framework to enhance the robustness and scalability of the ConvQA systems. 

---
# AttnComp: Attention-Guided Adaptive Context Compression for Retrieval-Augmented Generation 

**Authors**: Lvzhou Luo, Yixuan Cao, Ping Luo  

**Link**: [PDF](https://arxiv.org/pdf/2509.17486)  

**Abstract**: Retrieval-augmented generation improves the factual accuracy of Large Language Models (LLMs) by incorporating external context, but often suffers from irrelevant retrieved content that hinders effectiveness. Context compression addresses this issue by filtering out irrelevant information from context before LLM generation. However, existing methods struggle to adaptively adjust compression rates for different context, maintain low latency and integrate information across multiple documents. To overcome these limitations, We introduce AttnComp, an adaptive, efficient and context-aware compression framework. By leveraging the attention mechanism of LLMs to identify relevant information, AttnComp employs a Top-P compression algorithm to retain the minimal set of documents whose cumulative attention weights exceeds a predefined threshold. In addition to compression, AttnComp estimates response confidence by assessing the overall relevance of the retrieved content, enabling users to gauge response reliability. Experiments demonstrate that AttnComp outperforms existing compression methods and uncompressed baselines, achieving higher accuracy with substantial compression rates and lower latency. 

---
# EpiCache: Episodic KV Cache Management for Long Conversational Question Answering 

**Authors**: Minsoo Kim, Arnav Kundu, Han-Byul Kim, Richa Dixit, Minsik Cho  

**Link**: [PDF](https://arxiv.org/pdf/2509.17396)  

**Abstract**: Recent advances in large language models (LLMs) have extended context lengths, enabling assistants to sustain long histories for coherent, personalized responses. This ability, however, hinges on Key-Value (KV) caching, whose memory grows linearly with dialogue length and quickly dominates under strict resource constraints. An active line of research for reducing this overhead is KV cache compression, which seeks to limit cache size while preserving accuracy. Yet existing methods face two major limitations: (i) evicting entries after full-context prefill causes unbounded peak memory, and (ii) query-dependent eviction narrows the cache to a single query, leading to degraded accuracy in multi-turn conversations. We introduce EpiCache, a training-free KV cache management framework for long conversational question answering (LongConvQA) under fixed memory budgets. EpiCache bounds cache growth through block-wise prefill and preserves topic-relevant context via episodic KV compression, which clusters conversation history into coherent episodes and applies episode-specific KV cache eviction. We further design an adaptive layer-wise budget allocation strategy that measures each layer's sensitivity to eviction and distributes the memory budget across layers accordingly. Across three LongConvQA benchmarks, EpiCache improves accuracy by up to 40% over recent baselines, sustains near-full KV accuracy under 4-6x compression, and reduces latency and memory by up to 2.4x and 3.5x, thereby enabling efficient multi-turn interaction under strict resource constraints. 

---
# Extending Automatic Machine Translation Evaluation to Book-Length Documents 

**Authors**: Kuang-Da Wang, Shuoyang Ding, Chao-Han Huck Yang, Ping-Chun Hsieh, Wen-Chih Peng, Vitaly Lavrukhin, Boris Ginsburg  

**Link**: [PDF](https://arxiv.org/pdf/2509.17249)  

**Abstract**: Despite Large Language Models (LLMs) demonstrating superior translation performance and long-context capabilities, evaluation methodologies remain constrained to sentence-level assessment due to dataset limitations, token number restrictions in metrics, and rigid sentence boundary requirements. We introduce SEGALE, an evaluation scheme that extends existing automatic metrics to long-document translation by treating documents as continuous text and applying sentence segmentation and alignment methods. Our approach enables previously unattainable document-level evaluation, handling translations of arbitrary length generated with document-level prompts while accounting for under-/over-translations and varied sentence boundaries. Experiments show our scheme significantly outperforms existing long-form document evaluation schemes, while being comparable to evaluations performed with groundtruth sentence alignments. Additionally, we apply our scheme to book-length texts and newly demonstrate that many open-weight LLMs fail to effectively translate documents at their reported maximum context lengths. 

---
# Language Modeling with Learned Meta-Tokens 

**Authors**: Alok N. Shah, Khush Gupta, Keshav Ramji, Pratik Chaudhari  

**Link**: [PDF](https://arxiv.org/pdf/2509.16278)  

**Abstract**: While modern Transformer-based language models (LMs) have achieved major success in multi-task generalization, they often struggle to capture long-range dependencies within their context window. This work introduces a novel approach using meta-tokens, special tokens injected during pre-training, along with a dedicated meta-attention mechanism to guide LMs to use these tokens. We pre-train a language model with a modified GPT-2 architecture equipped with meta-attention in addition to causal multi-head attention, and study the impact of these tokens on a suite of synthetic tasks. We find that data-efficient language model pre-training on fewer than 100B tokens utilizing meta-tokens and our meta-attention mechanism achieves strong performance on these tasks after fine-tuning. We suggest that these gains arise due to the meta-tokens sharpening the positional encoding. This enables them to operate as trainable, content-based landmarks, implicitly compressing preceding context and "caching" it in the meta-token. At inference-time, the meta-token points to relevant context, facilitating length generalization up to 2$\times$ its context window, even after extension with YaRN. We provide further evidence of these behaviors by visualizing model internals to study the residual stream, and assessing the compression quality by information-theoretic analysis on the rate-distortion tradeoff. Our findings suggest that pre-training LMs with meta-tokens offers a simple, data-efficient method to enhance long-context language modeling performance, while introducing new insights into the nature of their behavior towards length generalization. 

---
