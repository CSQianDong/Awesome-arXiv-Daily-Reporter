# Beyond Many-Shot Translation: Scaling In-Context Demonstrations For Low-Resource Machine Translation 

**Authors**: Luis Frentzen Salim, Esteban Carlin, Alexandre Morinvil, Xi Ai, Lun-Wei Ku  

**Link**: [PDF](https://arxiv.org/pdf/2602.04764)  

**Abstract**: Building machine translation (MT) systems for low-resource languages is notably difficult due to the scarcity of high-quality data. Although Large Language Models (LLMs) have improved MT system performance, adapting them to lesser-represented languages remains challenging. In-context learning (ICL) may offer novel ways to adapt LLMs for low-resource MT by conditioning models on demonstration at inference time. In this study, we explore scaling low-resource machine translation ICL beyond the few-shot setting to thousands of examples with long-context models. We scale in-context token budget to 1M tokens and compare three types of training corpora used as in-context supervision: monolingual unsupervised data, instruction-style data, and parallel data (English--target and Indonesian--target). Our experiments on Javanese and Sundanese show that gains from additional context saturate quickly and can degrade near the maximum context window, with scaling behavior strongly dependent on corpus type. Notably, some forms of monolingual supervision can be competitive with parallel data, despite the latter offering additional supervision. Overall, our results characterize the effective limits and corpus-type sensitivity of long-context ICL for low-resource MT, highlighting that larger context windows do not necessarily yield proportional quality gains. 

---
# Beyond Holistic Scores: Automatic Trait-Based Quality Scoring of Argumentative Essays 

**Authors**: Lucile Favero, Juan Antonio Pérez-Ortiz, Tanja Käser, Nuria Oliver  

**Link**: [PDF](https://arxiv.org/pdf/2602.04604)  

**Abstract**: Automated Essay Scoring systems have traditionally focused on holistic scores, limiting their pedagogical usefulness, especially in the case of complex essay genres such as argumentative writing. In educational contexts, teachers and learners require interpretable, trait-level feedback that aligns with instructional goals and established rubrics. In this paper, we study trait-based Automatic Argumentative Essay Scoring using two complementary modeling paradigms designed for realistic educational deployment: (1) structured in-context learning with small open-source LLMs, and (2) a supervised, encoder-based BigBird model with a CORAL-style ordinal regression formulation, optimized for long-sequence understanding. We conduct a systematic evaluation on the ASAP++ dataset, which includes essay scores across five quality traits, offering strong coverage of core argumentation dimensions. LLMs are prompted with designed, rubric-aligned in-context examples, along with feedback and confidence requests, while we explicitly model ordinality in scores with the BigBird model via the rank-consistent CORAL framework. Our results show that explicitly modeling score ordinality substantially improves agreement with human raters across all traits, outperforming LLMs and nominal classification and regression-based baselines. This finding reinforces the importance of aligning model objectives with rubric semantics for educational assessment. At the same time, small open-source LLMs achieve a competitive performance without task-specific fine-tuning, particularly for reasoning-oriented traits, while enabling transparent, privacy-preserving, and locally deployable assessment scenarios. Our findings provide methodological, modeling, and practical insights for the design of AI-based educational systems that aim to deliver interpretable, rubric-aligned feedback for argumentative writing. 

---
# Focus-LIME: Surgical Interpretation of Long-Context Large Language Models via Proxy-Based Neighborhood Selection 

**Authors**: Junhao Liu, Haonan Yu, Zhenyu Yan, Xin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.04607)  

**Abstract**: As Large Language Models (LLMs) scale to handle massive context windows, achieving surgical feature-level interpretation is essential for high-stakes tasks like legal auditing and code debugging. However, existing local model-agnostic explanation methods face a critical dilemma in these scenarios: feature-based methods suffer from attribution dilution due to high feature dimensionality, thus failing to provide faithful explanations. In this paper, we propose Focus-LIME, a coarse-to-fine framework designed to restore the tractability of surgical interpretation. Focus-LIME utilizes a proxy model to curate the perturbation neighborhood, allowing the target model to perform fine-grained attribution exclusively within the optimized context. Empirical evaluations on long-context benchmarks demonstrate that our method makes surgical explanations practicable and provides faithful explanations to users. 

---
# LycheeDecode: Accelerating Long-Context LLM Inference via Hybrid-Head Sparse Decoding 

**Authors**: Gang Lin, Dongfang Li, Zhuoen Chen, Yukun Shi, Xuhui Chen, Baotian Hu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.04541)  

**Abstract**: The proliferation of long-context large language models (LLMs) exposes a key bottleneck: the rapidly expanding key-value cache during decoding, which imposes heavy memory and latency costs. While recent approaches attempt to alleviate this by sharing a single set of crucial tokens across layers, such coarse-grained sharing undermines model performance by neglecting the functional diversity of attention heads. To address this, we propose LycheeDecode, an efficient decoding method centered on a fine-grained hybrid-head attention mechanism that employs a hardware-efficient top-k selection strategy. Specifically, the novel HardKuma-based mechanism partitions attention heads into a small subset of retrieval heads that dynamically identify crucial tokens and a majority of sparse heads that reuse them for efficient computation. Through extensive experiments on leading models like Llama3 and Qwen3 across diverse benchmarks for long-context understanding (e.g., LongBench, RULER) and complex reasoning (e.g., AIME24, OlympiadBench), we demonstrate that LycheeDecode achieves generative quality comparable to, and at times surpassing even the full-attention baseline. Crucially, this is accomplished with up to a 2.7x speedup at a 128K context length. By preserving the functional diversity of attention heads, our fine-grained strategy overcomes the performance bottlenecks of existing methods, providing a powerful and validated pathway to both efficient and high-quality long-context LLM inference. 

---
# SAR-RAG: ATR Visual Question Answering by Semantic Search, Retrieval, and MLLM Generation 

**Authors**: David F. Ramirez, Tim Overman, Kristen Jaskie, Joe Marvin, Andreas Spanias  

**Link**: [PDF](https://arxiv.org/pdf/2602.04712)  

**Abstract**: We present a visual-context image retrieval-augmented generation (ImageRAG) assisted AI agent for automatic target recognition (ATR) of synthetic aperture radar (SAR). SAR is a remote sensing method used in defense and security applications to detect and monitor the positions of military vehicles, which may appear indistinguishable in images. Researchers have extensively studied SAR ATR to improve the differentiation and identification of vehicle types, characteristics, and measurements. Test examples can be compared with known vehicle target types to improve recognition tasks. New methods enhance the capabilities of neural networks, transformer attention, and multimodal large language models. An agentic AI method may be developed to utilize a defined set of tools, such as searching through a library of similar examples. Our proposed method, SAR Retrieval-Augmented Generation (SAR-RAG), combines a multimodal large language model (MLLM) with a vector database of semantic embeddings to support contextual search for image exemplars with known qualities. By recovering past image examples with known true target types, our SAR-RAG system can compare similar vehicle categories, achieving improved ATR prediction accuracy. We evaluate this through search and retrieval metrics, categorical classification accuracy, and numeric regression of vehicle dimensions. These metrics all show improvements when SAR-RAG is added to an MLLM baseline method as an attached ATR memory bank. 

---
