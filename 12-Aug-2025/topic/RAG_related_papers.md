# Careful Queries, Credible Results: Teaching RAG Models Advanced Web Search Tools with Reinforcement Learning 

**Authors**: Yuqin Dai, Shuo Yang, Guoqing Wang, Yong Deng, Zhanwei Zhang, Jun Yin, Pengyu Zeng, Zhenzhe Ying, Changhua Meng, Can Yi, Yuchen Zhou, Weiqiang Wang, Shuai Lu  

**Link**: [PDF](https://arxiv.org/pdf/2508.07956)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by integrating up-to-date external knowledge, yet real-world web environments present unique challenges. These limitations manifest as two key challenges: pervasive misinformation in the web environment, which introduces unreliable or misleading content that can degrade retrieval accuracy, and the underutilization of web tools, which, if effectively employed, could enhance query precision and help mitigate this noise, ultimately improving the retrieval results in RAG systems. To address these issues, we propose WebFilter, a novel RAG framework that generates source-restricted queries and filters out unreliable content. This approach combines a retrieval filtering mechanism with a behavior- and outcome-driven reward strategy, optimizing both query formulation and retrieval outcomes. Extensive experiments demonstrate that WebFilter improves answer quality and retrieval precision, outperforming existing RAG methods on both in-domain and out-of-domain benchmarks. 

---
# PrLM: Learning Explicit Reasoning for Personalized RAG via Contrastive Reward Optimization 

**Authors**: Kepu Zhang, Teng Shi, Weijie Yu, Jun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.07342)  

**Abstract**: Personalized retrieval-augmented generation (RAG) aims to produce user-tailored responses by incorporating retrieved user profiles alongside the input query. Existing methods primarily focus on improving retrieval and rely on large language models (LLMs) to implicitly integrate the retrieved context with the query. However, such models are often sensitive to retrieval quality and may generate responses that are misaligned with user preferences. To address this limitation, we propose PrLM, a reinforcement learning framework that trains LLMs to explicitly reason over retrieved user profiles. Guided by a contrastively trained personalization reward model, PrLM effectively learns from user responses without requiring annotated reasoning paths. Experiments on three personalized text generation datasets show that PrLM outperforms existing methods and remains robust across varying numbers of retrieved profiles and different retrievers. 

---
# REX-RAG: Reasoning Exploration with Policy Correction in Retrieval-Augmented Generation 

**Authors**: Wentao Jiang, Xiang Feng, Zengmao Wang, Yong Luo, Pingbo Xu, Zhe Chen, Bo Du, Jing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.08149)  

**Abstract**: Reinforcement learning (RL) is emerging as a powerful paradigm for enabling large language models (LLMs) to perform complex reasoning tasks. Recent advances indicate that integrating RL with retrieval-augmented generation (RAG) allows LLMs to dynamically incorporate external knowledge, leading to more informed and robust decision making. However, we identify a critical challenge during policy-driven trajectory sampling: LLMs are frequently trapped in unproductive reasoning paths, which we refer to as "dead ends", committing to overconfident yet incorrect conclusions. This severely hampers exploration and undermines effective policy optimization. To address this challenge, we propose REX-RAG (Reasoning Exploration with Policy Correction in Retrieval-Augmented Generation), a novel framework that explores alternative reasoning paths while maintaining rigorous policy learning through principled distributional corrections. Our approach introduces two key innovations: (1) Mixed Sampling Strategy, which combines a novel probe sampling method with exploratory prompts to escape dead ends; and (2) Policy Correction Mechanism, which employs importance sampling to correct distribution shifts induced by mixed sampling, thereby mitigating gradient estimation bias. We evaluate it on seven question-answering benchmarks, and the experimental results show that REX-RAG achieves average performance gains of 5.1% on Qwen2.5-3B and 3.6% on Qwen2.5-7B over strong baselines, demonstrating competitive results across multiple datasets. The code is publicly available at this https URL. 

---
# Retrieval augmented generation based dynamic prompting for few-shot biomedical named entity recognition using large language models 

**Authors**: Yao Ge, Sudeshna Das, Yuting Guo, Abeed Sarker  

**Link**: [PDF](https://arxiv.org/pdf/2508.06504)  

**Abstract**: Biomedical named entity recognition (NER) is a high-utility natural language processing (NLP) task, and large language models (LLMs) show promise particularly in few-shot settings (i.e., limited training data). In this article, we address the performance challenges of LLMs for few-shot biomedical NER by investigating a dynamic prompting strategy involving retrieval-augmented generation (RAG). In our approach, the annotated in-context learning examples are selected based on their similarities with the input texts, and the prompt is dynamically updated for each instance during inference. We implemented and optimized static and dynamic prompt engineering techniques and evaluated them on five biomedical NER datasets. Static prompting with structured components increased average F1-scores by 12% for GPT-4, and 11% for GPT-3.5 and LLaMA 3-70B, relative to basic static prompting. Dynamic prompting further improved performance, with TF-IDF and SBERT retrieval methods yielding the best results, improving average F1-scores by 7.3% and 5.6% in 5-shot and 10-shot settings, respectively. These findings highlight the utility of contextually adaptive prompts via RAG for biomedical NER. 

---
# Hallucination as a Computational Boundary: A Hierarchy of Inevitability and the Oracle Escape 

**Authors**: Quan Shi, Wang Xi, Zenghui Ding, Jianqing Gao, Xianjun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.07334)  

**Abstract**: The illusion phenomenon of large language models (LLMs) is the core obstacle to their reliable deployment. This article formalizes the large language model as a probabilistic Turing machine by constructing a "computational necessity hierarchy", and for the first time proves the illusions are inevitable on diagonalization, incomputability, and information theory boundaries supported by the new "learner pump lemma". However, we propose two "escape routes": one is to model Retrieval Enhanced Generations (RAGs) as oracle machines, proving their absolute escape through "computational jumps", providing the first formal theory for the effectiveness of RAGs; The second is to formalize continuous learning as an "internalized oracle" mechanism and implement this path through a novel neural game theory this http URL, this article proposes a 

---
# Automated Formalization via Conceptual Retrieval-Augmented LLMs 

**Authors**: Wangyue Lu, Lun Du, Sirui Li, Ke Weng, Haozhe Sun, Hengyu Liu, Minghe Yu, Tiancheng Zhang, Ge Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.06931)  

**Abstract**: Interactive theorem provers (ITPs) require manual formalization, which is labor-intensive and demands expert knowledge. While automated formalization offers a potential solution, it faces two major challenges: model hallucination (e.g., undefined predicates, symbol misuse, and version incompatibility) and the semantic gap caused by ambiguous or missing premises in natural language descriptions. To address these issues, we propose CRAMF, a Concept-driven Retrieval-Augmented Mathematical Formalization framework. CRAMF enhances LLM-based autoformalization by retrieving formal definitions of core mathematical concepts, providing contextual grounding during code generation. However, applying retrieval-augmented generation (RAG) in this setting is non-trivial due to the lack of structured knowledge bases, the polymorphic nature of mathematical concepts, and the high precision required in formal retrieval. We introduce a framework for automatically constructing a concept-definition knowledge base from Mathlib4, the standard mathematical library for the Lean 4 theorem prover, indexing over 26,000 formal definitions and 1,000+ core mathematical concepts. To address conceptual polymorphism, we propose contextual query augmentation with domain- and application-level signals. In addition, we design a dual-channel hybrid retrieval strategy with reranking to ensure accurate and relevant definition retrieval. Experiments on miniF2F, ProofNet, and our newly proposed AdvancedMath benchmark show that CRAMF can be seamlessly integrated into LLM-based autoformalizers, yielding consistent improvements in translation accuracy, achieving up to 62.1% and an average of 29.9% relative improvement. 

---
# MuaLLM: A Multimodal Large Language Model Agent for Circuit Design Assistance with Hybrid Contextual Retrieval-Augmented Generation 

**Authors**: Pravallika Abbineni, Saoud Aldowaish, Colin Liechty, Soroosh Noorzad, Ali Ghazizadeh, Morteza Fayazi  

**Link**: [PDF](https://arxiv.org/pdf/2508.08137)  

**Abstract**: Conducting a comprehensive literature review is crucial for advancing circuit design methodologies. However, the rapid influx of state-of-the-art research, inconsistent data representation, and the complexity of optimizing circuit design objectives make this task significantly challenging. In this paper, we propose MuaLLM, an open-source multimodal Large Language Model (LLM) agent for circuit design assistance that integrates a hybrid Retrieval-Augmented Generation (RAG) framework with an adaptive vector database of circuit design research papers. Unlike conventional LLMs, the MuaLLM agent employs a Reason + Act (ReAct) workflow for iterative reasoning, goal-setting, and multi-step information retrieval. It functions as a question-answering design assistant, capable of interpreting complex queries and providing reasoned responses grounded in circuit literature. Its multimodal capabilities enable processing of both textual and visual data, facilitating more efficient and comprehensive analysis. The system dynamically adapts using intelligent search tools, automated document retrieval from the internet, and real-time database updates. Unlike conventional approaches constrained by model context limits, MuaLLM decouples retrieval from inference, enabling scalable reasoning over arbitrarily large corpora. At the maximum context length supported by standard LLMs, MuaLLM remains up to 10x less costly and 1.6x faster while maintaining the same accuracy. This allows rapid, no-human-in-the-loop database generation, overcoming the bottleneck of simulation-based dataset creation for circuits. To evaluate MuaLLM, we introduce two custom datasets: RAG-250, targeting retrieval and citation performance, and Reasoning-100 (Reas-100), focused on multistep reasoning in circuit design. MuaLLM achieves 90.1% recall on RAG-250, and 86.8% accuracy on Reas-100. 

---
# MDK12-Bench: A Comprehensive Evaluation of Multimodal Large Language Models on Multidisciplinary Exams 

**Authors**: Pengfei Zhou, Xiaopeng Peng, Fanrui Zhang, Zhaopan Xu, Jiaxin Ai, Yansheng Qiu, Chuanhao Li, Zhen Li, Ming Li, Yukang Feng, Jianwen Sun, Haoquan Zhang, Zizhen Li, Xiaofeng Mao, Zekai Li, Wangbo Zhao, Kai Wang, Xiaojun Chang, Wenqi Shao, Yang You, Kaipeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.06851)  

**Abstract**: Multimodal large language models (MLLMs), which integrate language and visual cues for problem-solving, are crucial for advancing artificial general intelligence (AGI). However, current benchmarks for measuring the intelligence of MLLMs suffer from limited scale, narrow coverage, and unstructured knowledge, offering only static and undifferentiated evaluations. To bridge this gap, we introduce MDK12-Bench, a large-scale multidisciplinary benchmark built from real-world K-12 exams spanning six disciplines with 141K instances and 6,225 knowledge points organized in a six-layer taxonomy. Covering five question formats with difficulty and year annotations, it enables comprehensive evaluation to capture the extent to which MLLMs perform over four dimensions: 1) difficulty levels, 2) temporal (cross-year) shifts, 3) contextual shifts, and 4) knowledge-driven reasoning. We propose a novel dynamic evaluation framework that introduces unfamiliar visual, textual, and question form shifts to challenge model generalization while improving benchmark objectivity and longevity by mitigating data contamination. We further evaluate knowledge-point reference-augmented generation (KP-RAG) to examine the role of knowledge in problem-solving. Key findings reveal limitations in current MLLMs in multiple aspects and provide guidance for enhancing model robustness, interpretability, and AI-assisted education. 

---
# CarbonScaling: Extending Neural Scaling Laws for Carbon Footprint in Large Language Models 

**Authors**: Lei Jiang, Fan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.06524)  

**Abstract**: Neural scaling laws have driven the development of increasingly large language models (LLMs) by linking accuracy improvements to growth in parameter count, dataset size, and compute. However, these laws overlook the carbon emissions that scale exponentially with LLM size. This paper presents \textit{CarbonScaling}, an analytical framework that extends neural scaling laws to incorporate both operational and embodied carbon in LLM training. By integrating models for neural scaling, GPU hardware evolution, parallelism optimization, and carbon estimation, \textit{CarbonScaling} quantitatively connects model accuracy to carbon footprint. Results show that while a power-law relationship between accuracy and carbon holds, real-world inefficiencies significantly increase the scaling factor. Hardware technology scaling reduces carbon emissions for small to mid-sized models, but offers diminishing returns for extremely large LLMs due to communication overhead and underutilized GPUs. Training optimizations-especially aggressive critical batch size scaling-help alleviate this inefficiency. \textit{CarbonScaling} offers key insights for training more sustainable and carbon-efficient LLMs. 

---
