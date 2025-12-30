# TabiBERT: A Large-Scale ModernBERT Foundation Model and Unified Benchmarking Framework for Turkish 

**Authors**: Melikşah Türker, A. Ebrar Kızıloğlu, Onur Güngör, Susan Üsküdarlı  

**Link**: [PDF](https://arxiv.org/pdf/2512.23065)  

**Abstract**: Since the inception of BERT, encoder-only Transformers have evolved significantly in computational efficiency, training stability, and long-context modeling. ModernBERT consolidates these advances by integrating Rotary Positional Embeddings (RoPE), FlashAttention, and refined normalization. Despite these developments, Turkish NLP lacks a monolingual encoder trained from scratch incorporating such modern architectural paradigms. This work introduces TabiBERT, a monolingual Turkish encoder based on ModernBERT architecture trained from scratch on a large, curated corpus. TabiBERT is pre-trained on one trillion tokens sampled from an 84.88B token multi-domain corpus: web text (73%), scientific publications (20%), source code (6%), and mathematical content (0.3%). The model supports 8,192-token context length (16x original BERT), achieves up to 2.65x inference speedup, and reduces GPU memory consumption, enabling larger batch sizes. We introduce TabiBench with 28 datasets across eight task categories with standardized splits and protocols, evaluated using GLUE-style macro-averaging. TabiBERT attains 77.58 on TabiBench, outperforming BERTurk by 1.62 points and establishing state-of-the-art on five of eight categories: question answering (+9.55), code retrieval (+2.41), and document retrieval (+0.60). Compared with task-specific prior best results, including specialized models like TurkishBERTweet, TabiBERT achieves +1.47 average improvement, indicating robust cross-domain generalization. We release model weights, training configurations, and evaluation code for transparent, reproducible Turkish encoder research. 

---
# LENS: LLM-Enabled Narrative Synthesis for Mental Health by Aligning Multimodal Sensing with Language Models 

**Authors**: Wenxuan Xu, Arvind Pillai, Subigya Nepal, Amanda C Collins, Daniel M Mackin, Michael V Heinz, Tess Z Griffin, Nicholas C Jacobson, Andrew Campbell  

**Link**: [PDF](https://arxiv.org/pdf/2512.23025)  

**Abstract**: Multimodal health sensing offers rich behavioral signals for assessing mental health, yet translating these numerical time-series measurements into natural language remains challenging. Current LLMs cannot natively ingest long-duration sensor streams, and paired sensor-text datasets are scarce. To address these challenges, we introduce LENS, a framework that aligns multimodal sensing data with language models to generate clinically grounded mental-health narratives. LENS first constructs a large-scale dataset by transforming Ecological Momentary Assessment (EMA) responses related to depression and anxiety symptoms into natural-language descriptions, yielding over 100,000 sensor-text QA pairs from 258 participants. To enable native time-series integration, we train a patch-level encoder that projects raw sensor signals directly into an LLM's representation space. Our results show that LENS outperforms strong baselines on standard NLP metrics and task-specific measures of symptom-severity accuracy. A user study with 13 mental-health professionals further indicates that LENS-produced narratives are comprehensive and clinically meaningful. Ultimately, our approach advances LLMs as interfaces for health sensing, providing a scalable path toward models that can reason over raw behavioral signals and support downstream clinical decision-making. 

---
# Learning When Not to Attend Globally 

**Authors**: Xuan Luo, Kailai Zhang, Xifeng Yan  

**Link**: [PDF](https://arxiv.org/pdf/2512.22562)  

**Abstract**: When reading books, humans focus primarily on the current page, flipping back to recap prior context only when necessary. Similarly, we demonstrate that Large Language Models (LLMs) can learn to dynamically determine when to attend to global context. We propose All-or-Here Attention (AHA), which utilizes a binary router per attention head to dynamically toggle between full attention and local sliding window attention for each token. Our results indicate that with a window size of 256 tokens, up to 93\% of the original full attention operations can be replaced by sliding window attention without performance loss. Furthermore, by evaluating AHA across various window sizes, we identify a long-tail distribution in context dependency, where the necessity for full attention decays rapidly as the local window expands. By decoupling local processing from global access, AHA reveals that full attention is largely redundant, and that efficient inference requires only on-demand access to the global context. 

---
# The Gaining Paths to Investment Success: Information-Driven LLM Graph Reasoning for Venture Capital Prediction 

**Authors**: Haoyu Pei, Zhongyang Liu, Xiangyi Xiao, Xiaocong Du, Haipeng Zhang, Kunpeng Zhang, Suting Hong  

**Link**: [PDF](https://arxiv.org/pdf/2512.23489)  

**Abstract**: Most venture capital (VC) investments fail, while a few deliver outsized returns. Accurately predicting startup success requires synthesizing complex relational evidence, including company disclosures, investor track records, and investment network structures, through explicit reasoning to form coherent, interpretable investment theses. Traditional machine learning and graph neural networks both lack this reasoning capability. Large language models (LLMs) offer strong reasoning but face a modality mismatch with graphs. Recent graph-LLM methods target in-graph tasks where answers lie within the graph, whereas VC prediction is off-graph: the target exists outside the network. The core challenge is selecting graph paths that maximize predictor performance on an external objective while enabling step-by-step reasoning. We present MIRAGE-VC, a multi-perspective retrieval-augmented generation framework that addresses two obstacles: path explosion (thousands of candidate paths overwhelm LLM context) and heterogeneous evidence fusion (different startups need different analytical emphasis). Our information-gain-driven path retriever iteratively selects high-value neighbors, distilling investment networks into compact chains for explicit reasoning. A multi-agent architecture integrates three evidence streams via a learnable gating mechanism based on company attributes. Under strict anti-leakage controls, MIRAGE-VC achieves +5.0% F1 and +16.6% PrecisionAt5, and sheds light on other off-graph prediction tasks such as recommendation and risk assessment. Code: this https URL. 

---
# Beyond Single Bugs: Benchmarking Large Language Models for Multi-Vulnerability Detection 

**Authors**: Chinmay Pushkar, Sanchit Kabra, Dhruv Kumar, Jagat Sesh Challa  

**Link**: [PDF](https://arxiv.org/pdf/2512.22306)  

**Abstract**: Large Language Models (LLMs) have demonstrated significant potential in automated software security, particularly in vulnerability detection. However, existing benchmarks primarily focus on isolated, single-vulnerability samples or function-level classification, failing to reflect the complexity of real-world software where multiple interacting vulnerabilities often coexist within large files. Recent studies indicate that LLMs suffer from "count bias" and "selection bias" in multi-label tasks, yet this has not been rigorously quantified in the domain of code security. In this work, we introduce a comprehensive benchmark for Multi-Vulnerability Detection across four major languages: C, C++, Python, and JavaScript. We construct a dataset of 40,000 files by systematically injecting controlled counts of vulnerabilities (1, 3, 5, and 9) into long-context code samples (7.5k-10k tokens) sourced from CodeParrot. We evaluate five state-of-the-art LLMs, including GPT-4o-mini, Llama-3.3-70B, and the Qwen-2.5 series. Our results reveal a sharp degradation in performance as vulnerability density increases. While Llama-3.3-70B achieves near-perfect F1 scores (approximately 0.97) on single-vulnerability C tasks, performance drops by up to 40% in high-density settings. Notably, Python and JavaScript show distinct failure modes compared to C/C++, with models exhibiting severe "under-counting" (Recall dropping to less than 0.30) in complex Python files. 

---
