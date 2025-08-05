# ChEmbed: Enhancing Chemical Literature Search Through Domain-Specific Text Embeddings 

**Authors**: Ali Shiraee Kasmaee, Mohammad Khodadad, Mehdi Astaraki, Mohammad Arshi Saloot, Nicholas Sherck, Hamidreza Mahyar, Soheila Samiee  

**Link**: [PDF](https://arxiv.org/pdf/2508.01643)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems in chemistry heavily depend on accurate and relevant retrieval of chemical literature. However, general-purpose text embedding models frequently fail to adequately represent complex chemical terminologies, resulting in suboptimal retrieval quality. Specialized embedding models tailored to chemical literature retrieval have not yet been developed, leaving a substantial performance gap. To address this challenge, we introduce ChEmbed, a domain-adapted family of text embedding models fine-tuned on a dataset comprising chemistry-specific text from the PubChem, Semantic Scholar, and ChemRxiv corpora. To create effective training data, we employ large language models to synthetically generate queries, resulting in approximately 1.7 million high-quality query-passage pairs. Additionally, we augment the tokenizer by adding 900 chemically specialized tokens to previously unused slots, which significantly reduces the fragmentation of chemical entities, such as IUPAC names. ChEmbed also maintains a 8192-token context length, enabling the efficient retrieval of longer passages compared to many other open-source embedding models, which typically have a context length of 512 or 2048 tokens. Evaluated on our newly introduced ChemRxiv Retrieval benchmark, ChEmbed outperforms state-of-the-art general embedding models, raising nDCG@10 from 0.82 to 0.91 (+9 pp). ChEmbed represents a practical, lightweight, and reproducible embedding solution that effectively improves retrieval for chemical literature search. 

---
# CompressKV: Semantic Retrieval Heads Know What Tokens are Not Important Before Generation 

**Authors**: Xiaolin Lin, Jingcun Wang, Olga Kondrateva, Yiyu Shi, Bing Li, Grace Li Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02401)  

**Abstract**: Recent advances in large language models (LLMs) have significantly boosted long-context processing. However, the increasing key-value (KV) cache size poses critical challenges to memory and execution efficiency. Most KV cache compression methods rely on heuristic token eviction using all attention heads in Grouped Query Attention (GQA)-based LLMs. This method ignores the different functionalities of attention heads, leading to the eviction of critical tokens and thus degrades the performance of LLMs.
To address the issue above, instead of using all the attention heads in GQA-based LLMs to determine important tokens as in the previous work, we first identify the attention heads in each layer that are not only capable of retrieving the initial and final tokens of a prompt, but also capable of retrieving important tokens within the text and attending to their surrounding semantic context. Afterwards, we exploit such heads to determine the important tokens and retain their corresponding KV cache pairs. Furthermore, we analyze the cache eviction error of each layer individually and introduce a layer-adaptive KV cache allocation strategy. Experimental results demonstrate the proposed CompressKV consistently outperforms state-of-the-art approaches under various memory budgets on LongBench and Needle-in-a-Haystack benchmarks. Our code is publicly available at: this https URL. 

---
# LaMPE: Length-aware Multi-grained Position Encoding for Adaptive Long-context Scaling Without Training 

**Authors**: Sikui Zhang, Guangze Gao, Ziyun Gan, Chunfeng Yuan, Zefeng Lin, Houwen Peng, Bing Li, Weiming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02308)  

**Abstract**: Large language models (LLMs) experience significant performance degradation when the input exceeds the pretraining context window, primarily due to the out-of-distribution (OOD) behavior of Rotary Position Embedding (RoPE). Recent studies mitigate this problem by remapping OOD positions into the in-distribution range with fixed mapping strategies, ignoring the dynamic relationship between input length and the model's effective context window. To this end, we propose Length-aware Multi-grained Positional Encoding (LaMPE), a training-free method that fully utilizes the model's effective context window for adaptive long-context scaling in LLMs. Motivated by the left-skewed frequency distribution of relative positions, LaMPE establishes a dynamic relationship between mapping length and input length through a parametric scaled sigmoid function to adaptively allocate positional capacity across varying input lengths. Meanwhile, LaMPE devises a novel multi-grained attention mechanism that strategically allocates positional resolution across different sequence regions to capture both fine-grained locality and long-range dependencies. Our method can be seamlessly applied to a wide range of RoPE-based LLMs without training. Extensive experiments on three representative LLMs across five mainstream long-context benchmarks demonstrate that LaMPE achieves significant performance improvements compared to existing length extrapolation methods. The code will be released at this https URL. 

---
# Pointer: Linear-Complexity Long-Range Modeling without Pre-training 

**Authors**: Zixi Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.02631)  

**Abstract**: We introduce Pointer, a novel architecture that achieves linear $O(NK)$ complexity for long-range sequence modeling while maintaining superior performance without requiring pre-training. Unlike standard attention mechanisms that compute $O(N^2)$ pairwise interactions, our approach uses layer-wise pointer chaining where each layer's pointer selection depends on previous layer's pointer positions, creating explicit long-distance connections through pointer chains. We demonstrate that this architecture achieves $2$--$10\times$ speedup on long sequences compared to standard transformers, maintains $>95\%$ accuracy on copy tasks at distances up to 2048 tokens, and learns interpretable pointer patterns that reveal structured dependency modeling. Our experiments on efficiency benchmarks, long-range dependency tasks, and interpretability analysis show that Pointer offers a compelling alternative to attention mechanisms for scenarios requiring efficient long-range modeling without pre-training dependencies. 

---
# SitEmb-v1.5: Improved Context-Aware Dense Retrieval for Semantic Association and Long Story Comprehension 

**Authors**: Junjie Wu, Jiangnan Li, Yuqing Li, Lemao Liu, Liyan Xu, Jiwei Li, Dit-Yan Yeung, Jie Zhou, Mo Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01959)  

**Abstract**: Retrieval-augmented generation (RAG) over long documents typically involves splitting the text into smaller chunks, which serve as the basic units for retrieval. However, due to dependencies across the original document, contextual information is often essential for accurately interpreting each chunk. To address this, prior work has explored encoding longer context windows to produce embeddings for longer chunks. Despite these efforts, gains in retrieval and downstream tasks remain limited. This is because (1) longer chunks strain the capacity of embedding models due to the increased amount of information they must encode, and (2) many real-world applications still require returning localized evidence due to constraints on model or human bandwidth.
We propose an alternative approach to this challenge by representing short chunks in a way that is conditioned on a broader context window to enhance retrieval performance -- i.e., situating a chunk's meaning within its context. We further show that existing embedding models are not well-equipped to encode such situated context effectively, and thus introduce a new training paradigm and develop the situated embedding models (SitEmb). To evaluate our method, we curate a book-plot retrieval dataset specifically designed to assess situated retrieval capabilities. On this benchmark, our SitEmb-v1 model based on BGE-M3 substantially outperforms state-of-the-art embedding models, including several with up to 7-8B parameters, with only 1B parameters. Our 8B SitEmb-v1.5 model further improves performance by over 10% and shows strong results across different languages and several downstream applications. 

---
# Trainable Dynamic Mask Sparse Attention 

**Authors**: Jingze Shi, Yifan Wu, Bingheng Wu, Yiran Peng, Liangdong Wang, Guang Liu, Yuyu Luo  

**Link**: [PDF](https://arxiv.org/pdf/2508.02124)  

**Abstract**: In large language models, the demand for modeling long contexts is constantly increasing, but the quadratic complexity of the standard self-attention mechanism often becomes a bottleneck. Although existing sparse attention mechanisms have improved efficiency, they may still encounter issues such as static patterns or information loss. We introduce a trainable dynamic mask sparse attention mechanism, Dynamic Mask Attention, which effectively utilizes content-aware and position-aware sparsity. DMA achieves this through two key innovations: First, it dynamically generates content-aware sparse masks from value representations, enabling the model to identify and focus on critical information adaptively. Second, it implements position-aware sparse attention computation that effectively skips unnecessary calculation regions. This dual-sparsity design allows the model to significantly reduce the computational complexity of important information while retaining complete information, achieving an excellent balance between information fidelity and computational efficiency. We have verified the performance of DMA through comprehensive experiments. Comparative studies show that DMA outperforms multi-head attention, sliding window attention, multi-head latent attention, and native sparse attention in terms of perplexity under Chinchilla Scaling Law settings. Moreover, in challenging multi-query associative recall tasks, DMA also demonstrates superior performance and efficiency compared to these methods. Crucially, in the evaluation of a 1.7B parameter model, DMA significantly outperforms multi-head attention in both standard benchmark performance and the challenging needle-in-a-haystack task. These experimental results highlight its capability to balance model efficiency and long-context modeling ability effectively. 

---
# ReflecSched: Solving Dynamic Flexible Job-Shop Scheduling via LLM-Powered Hierarchical Reflection 

**Authors**: Shijie Cao, Yuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2508.01724)  

**Abstract**: Dynamic Flexible Job-Shop Scheduling (DFJSP) is an NP-hard problem challenged by real-time event adaptation and complex machine routing. While traditional dispatching rules are efficient but rigid, deep learning approaches are opaque and require intricate feature engineering. Large Language Models (LLMs) promise adaptive reasoning without this engineering overhead, yet we find their direct application is suboptimal. Baseline LLMs suffer from three key pitfalls: the long-context paradox, where crucial data is underutilized; an underutilization of expert heuristics; and myopic decision-making. To address this, we propose ReflecSched, a framework that empowers the LLM beyond a direct scheduler by equipping it with a strategic analysis capability. ReflecSched tasks the LLM to analyze heuristic-driven simulations across multiple planning horizons and distill them into a concise, natural-language summary termed ``Strategic Experience''. This summary is then integrated into the prompt of a final decision-making module, guiding it to produce non-myopic actions. Experiments show that ReflecSched not only statistically significantly outperforms direct LLM baselines, securing a 71.35\% Win Rate and a 2.755\% Relative Percentage Deviation reduction, but also surpasses the performance of all individual heuristics evaluated, all while demonstrably mitigating the three identified pitfalls. Additionally, ReflecSched performs on par with the best heuristic tailored to each instance across all problem cases. 

---
# KCR: Resolving Long-Context Knowledge Conflicts via Reasoning in LLMs 

**Authors**: Xianda Zheng, Zijian Huang, Meng-Fen Chiang, Michael J. Witbrock, Kaiqi Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.01273)  

**Abstract**: Knowledge conflicts commonly arise across diverse sources, and their prevalence has increased with the advent of LLMs. When dealing with conflicts between multiple contexts, also known as \emph{inter-context knowledge conflicts}, LLMs are often confused by lengthy and conflicting contexts. To address this challenge, we propose the Knowledge Conflict Reasoning (KCR) framework, which enhances the ability of LLMs to resolve conflicting knowledge. The key idea of KCR is to train backbone LLMs to establish a correct reasoning process by rewarding them for selecting and adhering to the context with stronger logical consistency when presented with conflicting contexts. Specifically, we first extract reasoning paths, represented by either text or local knowledge graphs, from the conflicting long contexts. Subsequently, we employ Reinforcement Learning to encourage the model to learn the paradigm of reasoning process that follows correct reasoning paths rather than the incorrect counterparts. This enables the backbone models to genuinely acquire the capability to resolve inter-context knowledge conflicts within long contexts. Experimental results demonstrate that our framework significantly improves the ability of various backbone models to resolve knowledge conflicts in long-context scenarios, yielding substantial performance gains. 

---
