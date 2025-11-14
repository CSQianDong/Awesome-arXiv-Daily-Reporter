# Don't Waste It: Guiding Generative Recommenders with Structured Human Priors via Multi-head Decoding 

**Authors**: Yunkai Zhang, Qiang Zhang, Feng, Ruizhong Qiu, Hanchao Yu, Jason Liu, Yinglong Xia, Zhuoran Yu, Zeyu Zheng, Diji Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.10492)  

**Abstract**: Optimizing recommender systems for objectives beyond accuracy, such as diversity, novelty, and personalization, is crucial for long-term user satisfaction. To this end, industrial practitioners have accumulated vast amounts of structured domain knowledge, which we term human priors (e.g., item taxonomies, temporal patterns). This knowledge is typically applied through post-hoc adjustments during ranking or post-ranking. However, this approach remains decoupled from the core model learning, which is particularly undesirable as the industry shifts to end-to-end generative recommendation foundation models. On the other hand, many methods targeting these beyond-accuracy objectives often require architecture-specific modifications and discard these valuable human priors by learning user intent in a fully unsupervised manner.
Instead of discarding the human priors accumulated over years of practice, we introduce a backbone-agnostic framework that seamlessly integrates these human priors directly into the end-to-end training of generative recommenders. With lightweight, prior-conditioned adapter heads inspired by efficient LLM decoding strategies, our approach guides the model to disentangle user intent along human-understandable axes (e.g., interaction types, long- vs. short-term interests). We also introduce a hierarchical composition strategy for modeling complex interactions across different prior types. Extensive experiments on three large-scale datasets demonstrate that our method significantly enhances both accuracy and beyond-accuracy objectives. We also show that human priors allow the backbone model to more effectively leverage longer context lengths and larger model sizes. 

---
# Instella: Fully Open Language Models with Stellar Performance 

**Authors**: Jiang Liu, Jialian Wu, Xiaodong Yu, Yusheng Su, Prakamya Mishra, Gowtham Ramesh, Sudhanshu Ranjan, Chaitanya Manem, Ximeng Sun, Ze Wang, Pratik Prabhanjan Brahma, Zicheng Liu, Emad Barsoum  

**Link**: [PDF](https://arxiv.org/pdf/2511.10628)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable performance across a wide range of tasks, yet the majority of high-performing models remain closed-source or partially open, limiting transparency and reproducibility. In this work, we introduce Instella, a family of fully open three billion parameter language models trained entirely on openly available data and codebase. Powered by AMD Instinct MI300X GPUs, Instella is developed through large-scale pre-training, general-purpose instruction tuning, and alignment with human preferences. Despite using substantially fewer pre-training tokens than many contemporaries, Instella achieves state-of-the-art results among fully open models and is competitive with leading open-weight models of comparable size. We further release two specialized variants: Instella-Long, capable of handling context lengths up to 128K tokens, and Instella-Math, a reasoning-focused model enhanced through supervised fine-tuning and reinforcement learning on mathematical tasks. Together, these contributions establish Instella as a transparent, performant, and versatile alternative for the community, advancing the goal of open and reproducible language modeling research. 

---
# Convomem Benchmark: Why Your First 150 Conversations Don't Need RAG 

**Authors**: Egor Pakhomov, Erik Nijkamp, Caiming Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2511.10523)  

**Abstract**: We introduce a comprehensive benchmark for conversational memory evaluation containing 75,336 question-answer pairs across diverse categories including user facts, assistant recall, abstention, preferences, temporal changes, and implicit connections. While existing benchmarks have advanced the field, our work addresses fundamental challenges in statistical power, data generation consistency, and evaluation flexibility that limit current memory evaluation frameworks. We examine the relationship between conversational memory and retrieval-augmented generation (RAG). While these systems share fundamental architectural patterns--temporal reasoning, implicit extraction, knowledge updates, and graph representations--memory systems have a unique characteristic: they start from zero and grow progressively with each conversation. This characteristic enables naive approaches that would be impractical for traditional RAG. Consistent with recent findings on long context effectiveness, we observe that simple full-context approaches achieve 70-82% accuracy even on our most challenging multi-message evidence cases, while sophisticated RAG-based memory systems like Mem0 achieve only 30-45% when operating on conversation histories under 150 interactions. Our analysis reveals practical transition points: long context excels for the first 30 conversations, remains viable with manageable trade-offs up to 150 conversations, and typically requires hybrid or RAG approaches beyond that point as costs and latencies become prohibitive. These patterns indicate that the small-corpus advantage of conversational memory--where exhaustive search and complete reranking are feasible--deserves dedicated research attention rather than simply applying general RAG solutions to conversation histories. 

---
# URaG: Unified Retrieval and Generation in Multimodal LLMs for Efficient Long Document Understanding 

**Authors**: Yongxin Shi, Jiapeng Wang, Zeyu Shan, Dezhi Peng, Zening Lin, Lianwen Jin  

**Link**: [PDF](https://arxiv.org/pdf/2511.10552)  

**Abstract**: Recent multimodal large language models (MLLMs) still struggle with long document understanding due to two fundamental challenges: information interference from abundant irrelevant content, and the quadratic computational cost of Transformer-based architectures. Existing approaches primarily fall into two categories: token compression, which sacrifices fine-grained details; and introducing external retrievers, which increase system complexity and prevent end-to-end optimization. To address these issues, we conduct an in-depth analysis and observe that MLLMs exhibit a human-like coarse-to-fine reasoning pattern: early Transformer layers attend broadly across the document, while deeper layers focus on relevant evidence pages. Motivated by this insight, we posit that the inherent evidence localization capabilities of MLLMs can be explicitly leveraged to perform retrieval during the reasoning process, facilitating efficient long document understanding. To this end, we propose URaG, a simple-yet-effective framework that Unifies Retrieval and Generation within a single MLLM. URaG introduces a lightweight cross-modal retrieval module that converts the early Transformer layers into an efficient evidence selector, identifying and preserving the most relevant pages while discarding irrelevant content. This design enables the deeper layers to concentrate computational resources on pertinent information, improving both accuracy and efficiency. Extensive experiments demonstrate that URaG achieves state-of-the-art performance while reducing computational overhead by 44-56%. The code is available at this https URL. 

---
# ScaleFormer: Span Representation Cumulation for Long-Context Transformer 

**Authors**: Jiangshu Du, Wenpeng Yin, Philip Yu  

**Link**: [PDF](https://arxiv.org/pdf/2511.10029)  

**Abstract**: The quadratic complexity of standard self-attention severely limits the application of Transformer-based models to long-context tasks. While efficient Transformer variants exist, they often require architectural changes and costly pre-training from scratch. To circumvent this, we propose ScaleFormer(Span Representation Cumulation for Long-Context Transformer) - a simple and effective plug-and-play framework that adapts off-the-shelf pre-trained encoder-decoder models to process long sequences without requiring architectural modifications. Our approach segments long inputs into overlapping chunks and generates a compressed, context-aware representation for the decoder. The core of our method is a novel, parameter-free fusion mechanism that endows each chunk's representation with structural awareness of its position within the document. It achieves this by enriching each chunk's boundary representations with cumulative context vectors from all preceding and succeeding chunks. This strategy provides the model with a strong signal of the document's narrative flow, achieves linear complexity, and enables pre-trained models to reason effectively over long-form text. Experiments on long-document summarization show that our method is highly competitive with and often outperforms state-of-the-art approaches without requiring architectural modifications or external retrieval mechanisms. 

---
# OIDA-QA: A Multimodal Benchmark for Analyzing the Opioid Industry Documents Archive 

**Authors**: Xuan Shen, Brian Wingenroth, Zichao Wang, Jason Kuen, Wanrong Zhu, Ruiyi Zhang, Yiwei Wang, Lichun Ma, Anqi Liu, Hongfu Liu, Tong Sun, Kevin S. Hawkins, Kate Tasker, G. Caleb Alexander, Jiuxiang Gu  

**Link**: [PDF](https://arxiv.org/pdf/2511.09914)  

**Abstract**: The opioid crisis represents a significant moment in public health that reveals systemic shortcomings across regulatory systems, healthcare practices, corporate governance, and public policy. Analyzing how these interconnected systems simultaneously failed to protect public health requires innovative analytic approaches for exploring the vast amounts of data and documents disclosed in the UCSF-JHU Opioid Industry Documents Archive (OIDA). The complexity, multimodal nature, and specialized characteristics of these healthcare-related legal and corporate documents necessitate more advanced methods and models tailored to specific data types and detailed annotations, ensuring the precision and professionalism in the analysis. In this paper, we tackle this challenge by organizing the original dataset according to document attributes and constructing a benchmark with 400k training documents and 10k for testing. From each document, we extract rich multimodal information-including textual content, visual elements, and layout structures-to capture a comprehensive range of features. Using multiple AI models, we then generate a large-scale dataset comprising 360k training QA pairs and 10k testing QA pairs. Building on this foundation, we develop domain-specific multimodal Large Language Models (LLMs) and explore the impact of multimodal inputs on task performance. To further enhance response accuracy, we incorporate historical QA pairs as contextual grounding for answering current queries. Additionally, we incorporate page references within the answers and introduce an importance-based page classifier, further improving the precision and relevance of the information provided. Preliminary results indicate the improvements with our AI assistant in document information extraction and question-answering tasks. The dataset and models are publicly available at: this https URL 

---
# History Rhymes: Macro-Contextual Retrieval for Robust Financial Forecasting 

**Authors**: Sarthak Khanna, Armin Berger, Muskaan Chopra, Rafet Sifa  

**Link**: [PDF](https://arxiv.org/pdf/2511.09754)  

**Abstract**: Financial markets are inherently non-stationary: structural breaks and macroeconomic regime shifts often cause forecasting models to fail when deployed out of distribution (OOD). Conventional multimodal approaches that simply fuse numerical indicators and textual sentiment rarely adapt to such shifts. We introduce macro-contextual retrieval, a retrieval-augmented forecasting framework that grounds each prediction in historically analogous macroeconomic regimes. The method jointly embeds macro indicators (e.g., CPI, unemployment, yield spread, GDP growth) and financial news sentiment in a shared similarity space, enabling causal retrieval of precedent periods during inference without retraining.
Trained on seventeen years of S&P 500 data (2007-2023) and evaluated OOD on AAPL (2024) and XOM (2024), the framework consistently narrows the CV to OOD performance gap. Macro-conditioned retrieval achieves the only positive out-of-sample trading outcomes (AAPL: PF=1.18, Sharpe=0.95; XOM: PF=1.16, Sharpe=0.61), while static numeric, text-only, and naive multimodal baselines collapse under regime shifts. Beyond metric gains, retrieved neighbors form interpretable evidence chains that correspond to recognizable macro contexts, such as inflationary or yield-curve inversion phases, supporting causal interpretability and transparency. By operationalizing the principle that "financial history may not repeat, but it often rhymes," this work demonstrates that macro-aware retrieval yields robust, explainable forecasts under distributional change.
All datasets, models, and source code are publicly available. 

---
