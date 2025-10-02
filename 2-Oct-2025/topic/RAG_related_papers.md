# Judging by Appearances? Auditing and Intervening Vision-Language Models for Bail Prediction 

**Authors**: Sagnik Basu, Shubham Prakash, Ashish Maruti Barge, Siddharth D Jaiswal, Abhisek Dash, Saptarshi Ghosh, Animesh Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2510.00088)  

**Abstract**: Large language models (LLMs) have been extensively used for legal judgment prediction tasks based on case reports and crime history. However, with a surge in the availability of large vision language models (VLMs), legal judgment prediction systems can now be made to leverage the images of the criminals in addition to the textual case reports/crime history. Applications built in this way could lead to inadvertent consequences and be used with malicious intent. In this work, we run an audit to investigate the efficiency of standalone VLMs in the bail decision prediction task. We observe that the performance is poor across multiple intersectional groups and models \textit{wrongly deny bail to deserving individuals with very high confidence}. We design different intervention algorithms by first including legal precedents through a RAG pipeline and then fine-tuning the VLMs using innovative schemes. We demonstrate that these interventions substantially improve the performance of bail prediction. Our work paves the way for the design of smarter interventions on VLMs in the future, before they can be deployed for real-world legal judgment prediction. 

---
# Benchmarking Foundation Models with Retrieval-Augmented Generation in Olympic-Level Physics Problem Solving 

**Authors**: Shunfeng Zheng, Yudi Zhang, Meng Fang, Zihan Zhang, Zhitan Wu, Mykola Pechenizkiy, Ling Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.00919)  

**Abstract**: Retrieval-augmented generation (RAG) with foundation models has achieved strong performance across diverse tasks, but their capacity for expert-level reasoning-such as solving Olympiad-level physics problems-remains largely unexplored. Inspired by the way students prepare for competitions by reviewing past problems, we investigate the potential of RAG to enhance physics reasoning in foundation models. We introduce PhoPile, a high-quality multimodal dataset specifically designed for Olympiad-level physics, enabling systematic study of retrieval-based reasoning. PhoPile includes diagrams, graphs, and equations, capturing the inherently multimodal nature of physics problem solving. Using PhoPile, we benchmark RAG-augmented foundation models, covering both large language models (LLMs) and large multimodal models (LMMs) with multiple retrievers. Our results demonstrate that integrating retrieval with physics corpora can improve model performance, while also highlighting challenges that motivate further research in retrieval-augmented physics reasoning. 

---
# Facilitating Cognitive Accessibility with LLMs: A Multi-Task Approach to Easy-to-Read Text Generation 

**Authors**: François Ledoyen, Gaël Dias, Jeremie Pantin, Alexis Lechervy, Fabrice Maurel, Youssef Chahir  

**Link**: [PDF](https://arxiv.org/pdf/2510.00662)  

**Abstract**: Simplifying complex texts is essential for ensuring equitable access to information, especially for individuals with cognitive impairments. The Easy-to-Read (ETR) initiative offers a framework for making content accessible to the neurodivergent population, but the manual creation of such texts remains time-consuming and resource-intensive. In this work, we investigate the potential of large language models (LLMs) to automate the generation of ETR content. To address the scarcity of aligned corpora and the specificity of ETR constraints, we propose a multi-task learning (MTL) approach that trains models jointly on text summarization, text simplification, and ETR generation. We explore two different strategies: multi-task retrieval-augmented generation (RAG) for in-context learning, and MTL-LoRA for parameter-efficient fine-tuning. Our experiments with Mistral-7B and LLaMA-3-8B, based on ETR-fr, a new high-quality dataset, demonstrate the benefits of multi-task setups over single-task baselines across all configurations. Moreover, results show that the RAG-based strategy enables generalization in out-of-domain settings, while MTL-LoRA outperforms all learning strategies within in-domain configurations. 

---
# Retrieval-Augmented Generation for Electrocardiogram-Language Models 

**Authors**: Xiaoyu Song, William Han, Tony Chen, Chaojing Duan, Michael A. Rosenberg, Emerson Liu, Ding Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.00261)  

**Abstract**: Interest in generative Electrocardiogram-Language Models (ELMs) is growing, as they can produce textual responses conditioned on ECG signals and textual queries. Unlike traditional classifiers that output label probabilities, ELMs are more versatile, supporting domain-specific tasks (e.g., waveform analysis, diagnosis, prognosis) as well as general tasks (e.g., open-ended questions, dialogue). Retrieval-Augmented Generation (RAG), widely used in Large Language Models (LLMs) to ground LLM outputs in retrieved knowledge, helps reduce hallucinations and improve natural language generation (NLG). However, despite its promise, no open-source implementation or systematic study of RAG pipeline design for ELMs currently exists. To address this gap, we present the first open-source RAG pipeline for ELMs, along with baselines and ablation studies for NLG. Experiments on three public datasets show that ELMs with RAG consistently improves performance over non-RAG baselines and highlights key ELM design considerations. Our code is available at: this https URL. 

---
# Optimizing What Matters: AUC-Driven Learning for Robust Neural Retrieval 

**Authors**: Nima Sheikholeslami, Erfan Hosseini, Patrice Bechard, Srivatsava Daruru, Sai Rajeswar  

**Link**: [PDF](https://arxiv.org/pdf/2510.00137)  

**Abstract**: Dual-encoder retrievers depend on the principle that relevant documents should score higher than irrelevant ones for a given query. Yet the dominant Noise Contrastive Estimation (NCE) objective, which underpins Contrastive Loss, optimizes a softened ranking surrogate that we rigorously prove is fundamentally oblivious to score separation quality and unrelated to AUC. This mismatch leads to poor calibration and suboptimal performance in downstream tasks like retrieval-augmented generation (RAG). To address this fundamental limitation, we introduce the MW loss, a new training objective that maximizes the Mann-Whitney U statistic, which is mathematically equivalent to the Area under the ROC Curve (AUC). MW loss encourages each positive-negative pair to be correctly ranked by minimizing binary cross entropy over score differences. We provide theoretical guarantees that MW loss directly upper-bounds the AoC, better aligning optimization with retrieval goals. We further promote ROC curves and AUC as natural threshold free diagnostics for evaluating retriever calibration and ranking quality. Empirically, retrievers trained with MW loss consistently outperform contrastive counterparts in AUC and standard retrieval metrics. Our experiments show that MW loss is an empirically superior alternative to Contrastive Loss, yielding better-calibrated and more discriminative retrievers for high-stakes applications like RAG. 

---
# Methodological Framework for Quantifying Semantic Test Coverage in RAG Systems 

**Authors**: Noah Broestl, Adel Nasser Abdalla, Rajprakash Bale, Hersh Gupta, Max Struever  

**Link**: [PDF](https://arxiv.org/pdf/2510.00001)  

**Abstract**: Reliably determining the performance of Retrieval-Augmented Generation (RAG) systems depends on comprehensive test questions. While a proliferation of evaluation frameworks for LLM-powered applications exists, current practices lack a systematic method to ensure these test sets adequately cover the underlying knowledge base, leaving developers with significant blind spots. To address this, we present a novel, applied methodology to quantify the semantic coverage of RAG test questions against their underlying documents. Our approach leverages existing technologies, including vector embeddings and clustering algorithms, to create a practical framework for validating test comprehensiveness. Our methodology embeds document chunks and test questions into a unified vector space, enabling the calculation of multiple coverage metrics: basic proximity, content-weighted coverage, and multi-topic question coverage. Furthermore, we incorporate outlier detection to filter irrelevant questions, allowing for the refinement of test sets. Experimental evidence from two distinct use cases demonstrates that our framework effectively quantifies test coverage, identifies specific content areas with inadequate representation, and provides concrete recommendations for generating new, high-value test questions. This work provides RAG developers with essential tools to build more robust test suites, thereby improving system reliability and extending to applications such as identifying misaligned documents. 

---
# HalluGuard: Evidence-Grounded Small Reasoning Models to Mitigate Hallucinations in Retrieval-Augmented Generation 

**Authors**: Loris Bergeron, Ioana Buhnila, Jérôme François, Radu State  

**Link**: [PDF](https://arxiv.org/pdf/2510.00880)  

**Abstract**: Large Language Models (LLMs) excel in many NLP tasks but remain prone to hallucinations, limiting trust in real-world applications. We present HalluGuard, a 4B-parameter Small Reasoning Model (SRM) for mitigating hallucinations in Retrieval-Augmented Generation (RAG). HalluGuard classifies document-claim pairs as grounded or hallucinated and produces evidence-grounded justifications for transparency. Our approach combines (i) a domain-agnostic synthetic dataset derived from FineWeb and refined through multi-stage curation and data reformation, (ii) synthetic grounded and hallucinated claims, and (iii) preference-based fine-tuning with Odds Ratio Preference Optimization to distill large-model reasoning into a smaller backbone. On the RAGTruth subset of the LLM-AggreFact benchmark, HalluGuard achieves 84.0% balanced accuracy (BAcc), rivaling specialized models, MiniCheck (7B; 84.0%) and Granite Guardian 3.3 (8B; 82.2%) while using roughly half their parameters. Over the full benchmark it reaches 75.7% BAcc, matching larger general-purpose LLMs such as GPT-4o (75.9%). We will release HalluGuard and datasets under Apache 2.0 upon acceptance. 

---
# Eyes-on-Me: Scalable RAG Poisoning through Transferable Attention-Steering Attractors 

**Authors**: Yen-Shan Chen, Sian-Yao Huang, Cheng-Lin Yang, Yun-Nung Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.00586)  

**Abstract**: Existing data poisoning attacks on retrieval-augmented generation (RAG) systems scale poorly because they require costly optimization of poisoned documents for each target phrase. We introduce Eyes-on-Me, a modular attack that decomposes an adversarial document into reusable Attention Attractors and Focus Regions. Attractors are optimized to direct attention to the Focus Region. Attackers can then insert semantic baits for the retriever or malicious instructions for the generator, adapting to new targets at near zero cost. This is achieved by steering a small subset of attention heads that we empirically identify as strongly correlated with attack success. Across 18 end-to-end RAG settings (3 datasets $\times$ 2 retrievers $\times$ 3 generators), Eyes-on-Me raises average attack success rates from 21.9 to 57.8 (+35.9 points, 2.6$\times$ over prior work). A single optimized attractor transfers to unseen black box retrievers and generators without retraining. Our findings establish a scalable paradigm for RAG data poisoning and show that modular, reusable components pose a practical threat to modern AI systems. They also reveal a strong link between attention concentration and model outputs, informing interpretability research. 

---
