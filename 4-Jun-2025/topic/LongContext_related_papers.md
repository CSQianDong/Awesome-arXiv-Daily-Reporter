# TransAct V2: Lifelong User Action Sequence Modeling on Pinterest Recommendation 

**Authors**: Xue Xia, Saurabh Vishwas Joshi, Kousik Rajesh, Kangnan Li, Yangyi Lu, Nikil Pancha, Dhruvil Deven Badani, Jiajing Xu, Pong Eksombatchai  

**Link**: [PDF](https://arxiv.org/pdf/2506.02267)  

**Abstract**: Modeling user action sequences has become a popular focus in industrial recommendation system research, particularly for Click-Through Rate (CTR) prediction tasks. However, industry-scale CTR models often rely on short user sequences, limiting their ability to capture long-term behavior. Additionally, these models typically lack an integrated action-prediction task within a point-wise ranking framework, reducing their predictive power. They also rarely address the infrastructure challenges involved in efficiently serving large-scale sequential models. In this paper, we introduce TransAct V2, a production model for Pinterest's Homefeed ranking system, featuring three key innovations: (1) leveraging very long user sequences to improve CTR predictions, (2) integrating a Next Action Loss function for enhanced user action forecasting, and (3) employing scalable, low-latency deployment solutions tailored to handle the computational demands of extended user action sequences. 

---
# Literary Evidence Retrieval via Long-Context Language Models 

**Authors**: Katherine Thai, Mohit Iyyer  

**Link**: [PDF](https://arxiv.org/pdf/2506.03090)  

**Abstract**: How well do modern long-context language models understand literary fiction? We explore this question via the task of literary evidence retrieval, repurposing the RELiC dataset of That et al. (2022) to construct a benchmark where the entire text of a primary source (e.g., The Great Gatsby) is provided to an LLM alongside literary criticism with a missing quotation from that work. This setting, in which the model must generate the missing quotation, mirrors the human process of literary analysis by requiring models to perform both global narrative reasoning and close textual examination. We curate a high-quality subset of 292 examples through extensive filtering and human verification. Our experiments show that recent reasoning models, such as Gemini Pro 2.5 can exceed human expert performance (62.5% vs. 50% accuracy). In contrast, the best open-weight model achieves only 29.1% accuracy, highlighting a wide gap in interpretive reasoning between open and closed-weight models. Despite their speed and apparent accuracy, even the strongest models struggle with nuanced literary signals and overgeneration, signaling open challenges for applying LLMs to literary analysis. We release our dataset and evaluation code to encourage future work in this direction. 

---
# A Controllable Examination for Long-Context Language Models 

**Authors**: Yijun Yang, Zeyu Huang, Wenhao Zhu, Zihan Qiu, Fei Yuan, Jeff Z.Pan, Ivan Titov  

**Link**: [PDF](https://arxiv.org/pdf/2506.02921)  

**Abstract**: Existing frameworks for evaluating long-context language models (LCLM) can be broadly categorized into real-world and synthetic tasks. Despite their utility, both approaches are accompanied by certain intrinsic limitations. Real-world tasks are too complex to interpret or characterize and are susceptible to data contamination. In contrast, synthetic tasks often adopt the needle-in-the-haystack (NIAH) format, wherein a lack of coherence between the "needle" and the "haystack" compromises their validity as proxies for realistic applications. In response to these challenges, we posit that an ideal long-context evaluation framework should be characterized by three essential features: $\textit{seamless context}$, $\textit{controllable setting}$, and $\textit{sound evaluation}$. This study introduces $\textbf{LongBioBench}$, a novel benchmark that utilizes artificially generated biographies as a controlled environment for assessing LCLMs across dimensions of $\textit{understanding}$, $\textit{reasoning}$, and $\textit{trustworthiness}$. Our experimental evaluation, which includes $\textbf{18}$ LCLMs in total, demonstrates that most models still exhibit deficiencies in semantic understanding and elementary reasoning over retrieved results and are less trustworthy as context length increases. Our further analysis indicates some design choices employed by existing synthetic benchmarks, such as contextual non-coherence, numerical needles, and the absence of distractors, rendering them vulnerable to test the model long-context capabilities. Moreover, we also reveal that long-context continual pretraining primarily adjusts RoPE embedding to accommodate extended context lengths. To sum up, compared to previous synthetic benchmarks, LongBioBench achieves a better trade-off between mirroring authentic language tasks and maintaining controllability, and is highly interpretable and configurable. 

---
# RACE-Align: Retrieval-Augmented and Chain-of-Thought Enhanced Preference Alignment for Large Language Models 

**Authors**: Qihang Yan, Xinyu Zhang, Luming Guo, Qi Zhang, Feifan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.02726)  

**Abstract**: Large Language Models (LLMs) struggle with accuracy, domain-specific reasoning, and interpretability in vertical domains. Traditional preference alignment methods like Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO) often overlook the underlying knowledge sources and reasoning logic. This paper introduces RACE-Align (Retrieval-Augmented and Chain-of-Thought Enhanced Alignment), a novel framework designed to address these limitations. RACE-Align systematically constructs a binary preference dataset incorporating external knowledge support and explicit Chain-of-Thought (CoT) reasoning, then aligns LLMs using the DPO algorithm. The core innovation lies in its preference data construction strategy: it integrates AI-driven retrieval for factual grounding, enhancing knowledgeability and accuracy, and emphasizes the optimization of domain-specific CoT, treating the reasoning process itself as a key preference dimension. A multi-stage, AI-driven refinement pipeline cost-effectively generates these preference pairs. Experimental validation in Traditional Chinese Medicine (TCM) using Qwen3-1.7B as the base model demonstrates that RACE-Align significantly outperforms the original base model and a model fine-tuned only with Supervised Fine-Tuning (SFT). Improvements were observed across multiple dimensions, including answer accuracy, information richness, application of TCM thinking patterns, logicality and depth of reasoning, and interpretability. These findings suggest RACE-Align offers an effective pathway to enhance LLMs' knowledge application, reasoning reliability, and process transparency in complex vertical domains. 

---
# Enhancing Large Language Models with Neurosymbolic Reasoning for Multilingual Tasks 

**Authors**: Sina Bagheri Nezhad, Ameeta Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2506.02483)  

**Abstract**: Large language models (LLMs) often struggle to perform multi-target reasoning in long-context scenarios where relevant information is scattered across extensive documents. To address this challenge, we introduce NeuroSymbolic Augmented Reasoning (NSAR), which combines the benefits of neural and symbolic reasoning during inference. NSAR explicitly extracts symbolic facts from text and generates executable Python code to handle complex reasoning steps. Through extensive experiments across seven languages and diverse context lengths, we demonstrate that NSAR significantly outperforms both a vanilla RAG baseline and advanced prompting strategies in accurately identifying and synthesizing multiple pieces of information. Our results highlight the effectiveness of combining explicit symbolic operations with neural inference for robust, interpretable, and scalable reasoning in multilingual settings. 

---
# NovelHopQA: Diagnosing Multi-Hop Reasoning Failures in Long Narrative Contexts 

**Authors**: Abhay Gupta, Michael Lu, Kevin Zhu, Sean O'Brien, Vasu Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2506.02000)  

**Abstract**: Current large language models (LLMs) struggle to answer questions that span tens of thousands of tokens, especially when multi-hop reasoning is involved. While prior benchmarks explore long-context comprehension or multi-hop reasoning in isolation, none jointly vary context length and reasoning depth in natural narrative settings. We introduce NovelHopQA, the first benchmark to evaluate k1-4 hop QA over 64k-128k-token excerpts from 83 full-length public-domain novels. A keyword-guided pipeline builds hop-separated chains grounded in coherent storylines. We evaluate six state-of-the-art (SOTA) models and apply oracle-context filtering to ensure all questions are genuinely answerable. Human annotators validate both alignment and hop depth. We noticed consistent accuracy drops with increased hops and context length, even in frontier models-revealing that sheer scale does not guarantee robust reasoning. Our failure mode analysis highlights common breakdowns, such as missed final-hop integration and long-range drift. NovelHopQA offers a controlled diagnostic setting to stress-test multi-hop reasoning at scale. 

---
# Breaking Quadratic Barriers: A Non-Attention LLM for Ultra-Long Context Horizons 

**Authors**: Andrew Kiruluta, Preethi Raju, Priscilla Burity  

**Link**: [PDF](https://arxiv.org/pdf/2506.01963)  

**Abstract**: We present a novel non attention based architecture for large language models (LLMs) that efficiently handles very long context windows, on the order of hundreds of thousands to potentially millions of tokens. Unlike traditional Transformer designs, which suffer from quadratic memory and computation overload due to the nature of the self attention mechanism, our model avoids token to token attention entirely. Instead, it combines the following complementary components: State Space blocks (inspired by S4) that learn continuous time convolution kernels and scale near linearly with sequence length, Multi Resolution Convolution layers that capture local context at different dilation levels, a lightweight Recurrent Supervisor to maintain a global hidden state across sequential chunks, and Retrieval Augmented External Memory that stores and retrieves high-level chunk embeddings without reintroducing quadratic operations. 

---
