# AcademicEval: Live Long-Context LLM Benchmark 

**Authors**: Haozhen Zhang, Tao Feng, Pengrui Han, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2510.17725)  

**Abstract**: Large Language Models (LLMs) have recently achieved remarkable performance in long-context understanding. However, current long-context LLM benchmarks are limited by rigid context length, labor-intensive annotation, and the pressing challenge of label leakage issues during LLM training. Therefore, we propose \textsc{AcademicEval}, a live benchmark for evaluating LLMs over long-context generation tasks. \textsc{AcademicEval} adopts papers on arXiv to introduce several academic writing tasks with long-context inputs, \textit{i.e.}, \textsc{Title}, \textsc{Abstract}, \textsc{Introduction}, and \textsc{Related Work}, which cover a wide range of abstraction levels and require no manual labeling. Moreover, \textsc{AcademicEval} integrates high-quality and expert-curated few-shot demonstrations from a collected co-author graph to enable flexible context length. Especially, \textsc{AcademicEval} features an efficient live evaluation, ensuring no label leakage. We conduct a holistic evaluation on \textsc{AcademicEval}, and the results illustrate that LLMs perform poorly on tasks with hierarchical abstraction levels and tend to struggle with long few-shot demonstrations, highlighting the challenge of our benchmark. Through experimental analysis, we also reveal some insights for enhancing LLMs' long-context modeling capabilities. Code is available at this https URL 

---
# Empowering Real-World: A Survey on the Technology, Practice, and Evaluation of LLM-driven Industry Agents 

**Authors**: Yihong Tang, Kehai Chen, Liang Yue, Jinxin Fan, Caishen Zhou, Xiaoguang Li, Yuyang Zhang, Mingming Zhao, Shixiong Kai, Kaiyang Guo, Xingshan Zeng, Wenjing Cun, Lifeng Shang, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.17491)  

**Abstract**: With the rise of large language models (LLMs), LLM agents capable of autonomous reasoning, planning, and executing complex tasks have become a frontier in artificial intelligence. However, how to translate the research on general agents into productivity that drives industry transformations remains a significant challenge. To address this, this paper systematically reviews the technologies, applications, and evaluation methods of industry agents based on LLMs. Using an industry agent capability maturity framework, it outlines the evolution of agents in industry applications, from "process execution systems" to "adaptive social systems." First, we examine the three key technological pillars that support the advancement of agent capabilities: Memory, Planning, and Tool Use. We discuss how these technologies evolve from supporting simple tasks in their early forms to enabling complex autonomous systems and collective intelligence in more advanced forms. Then, we provide an overview of the application of industry agents in real-world domains such as digital engineering, scientific discovery, embodied intelligence, collaborative business execution, and complex system simulation. Additionally, this paper reviews the evaluation benchmarks and methods for both fundamental and specialized capabilities, identifying the challenges existing evaluation systems face regarding authenticity, safety, and industry specificity. Finally, we focus on the practical challenges faced by industry agents, exploring their capability boundaries, developmental potential, and governance issues in various scenarios, while providing insights into future directions. By combining technological evolution with industry practices, this review aims to clarify the current state and offer a clear roadmap and theoretical foundation for understanding and building the next generation of industry agents. 

---
# Understanding and Improving Length Generalization in Hierarchical Sparse Attention Models 

**Authors**: Jiaqi Leng, Xiang Hu, Junxiong Wang, Jianguo Li, Wei Wu, Yucheng Lu  

**Link**: [PDF](https://arxiv.org/pdf/2510.17196)  

**Abstract**: Effectively processing long contexts is a critical challenge for language models. While standard Transformers are limited by quadratic complexity and poor length extrapolation, alternative architectures like sliding window attention and state space models sacrifice the ability to effectively utilize the full context due to their fixed-size memory. Chunk-based sparse attention has emerged as a promising paradigm for extreme length generalization, yet the key architectural principles underpinning its success are not yet fully understood. In this work, we present a systematic dissection of these models to identify the core components driving their performance. Through a unified framework and comprehensive ablation studies, we demonstrate that a combination of three design principles is critical: (1) an expressive, non-linear Chunk Encoder with a dedicated CLS token to produce representations for retrieval; (2) a Bypassing Residual Path to stably integrate retrieved global information without it being overridden by the local residual stream; and (3) enforced selection sparsity during pre-training to bridge the train-test distribution gap. We provide a theoretical motivation for intra-chunk information processing and landmark generation. By combining these principles, we establish a new state-of-the-art for training-free length extrapolation, successfully generalizing models trained on a 4K context to 32 million tokens on RULER and BABILong. Our findings provide a clear and empirically-grounded set of design principles for developing future, highly-capable long-context language models. 

---
# LC-Eval: A Bilingual Multi-Task Evaluation Benchmark for Long-Context Understanding 

**Authors**: Sheikh Jubair, Arwa Omayrah, Amal Alshammari, Alhanoof Althnian, Abdulhamed Alothaimen, Norah A. Alzahrani, Shahad D. Alzaidi, Nora Al-Twairesh, Abdulmohsen Al-Thubaity  

**Link**: [PDF](https://arxiv.org/pdf/2510.16783)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have demonstrated sophisticated capabilities, including the ability to process and comprehend extended contexts. These emergent capabilities necessitate rigorous evaluation methods to effectively assess their performance in long-context understanding. In this paper, we present \textbf{LC-Eval}, a bilingual, multi-task evaluation benchmark designed to evaluate long-context understanding in English and Arabic, targeting context lengths ranging from 4k to over 128k tokens. LC-Eval introduces four novel and challenging tasks: multi-document question answering, bilingual question answering, claim verification within a paragraph, and multiple-choice questions based on long contexts. These tasks are designed to assess LLMs' abilities in deep reasoning, document comprehension, information tracing, and bilingual information extraction and understanding. The benchmark includes datasets in both Arabic and English for each task, allowing for a comparative analysis of their performance across different text genres. Evaluations were conducted on both open-weight and closed LLMs, with results indicating that LC-Eval presents significant challenges. Even high-performing models, such as GPT-4o, struggled with certain tasks, highlighting the complexity and rigor of the benchmark. 

---
# Glyph: Scaling Context Windows via Visual-Text Compression 

**Authors**: Jiale Cheng, Yusen Liu, Xinyu Zhang, Yulin Fei, Wenyi Hong, Ruiliang Lyu, Weihan Wang, Zhe Su, Xiaotao Gu, Xiao Liu, Yushi Bai, Jie Tang, Hongning Wang, Minlie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.17800)  

**Abstract**: Large language models (LLMs) increasingly rely on long-context modeling for tasks such as document understanding, code analysis, and multi-step reasoning. However, scaling context windows to the million-token level brings prohibitive computational and memory costs, limiting the practicality of long-context LLMs. In this work, we take a different perspective-visual context scaling-to tackle this challenge. Instead of extending token-based sequences, we propose Glyph, a framework that renders long texts into images and processes them with vision-language models (VLMs). This approach substantially compresses textual input while preserving semantic information, and we further design an LLM-driven genetic search to identify optimal visual rendering configurations for balancing accuracy and compression. Through extensive experiments, we demonstrate that our method achieves 3-4x token compression while maintaining accuracy comparable to leading LLMs such as Qwen3-8B on various long-context benchmarks. This compression also leads to around 4x faster prefilling and decoding, and approximately 2x faster SFT training. Furthermore, under extreme compression, a 128K-context VLM could scale to handle 1M-token-level text tasks. In addition, the rendered text data benefits real-world multimodal tasks, such as document understanding. Our code and model are released at this https URL. 

---
# U-Codec: Ultra Low Frame-rate Neural Speech Codec for Fast High-fidelity Speech Generation 

**Authors**: Xusheng Yang, Long Zhou, Wenfu Wang, Kai Hu, Shulin Feng, Chenxing Li, Meng Yu, Dong Yu, Yuexian Zou  

**Link**: [PDF](https://arxiv.org/pdf/2510.16718)  

**Abstract**: We propose \textbf{U-Codec}, an \textbf{U}ltra low frame-rate neural speech \textbf{Codec} that achieves high-fidelity reconstruction and fast speech generation at an extremely low frame-rate of 5Hz (5 frames per second). Extreme compression at 5Hz typically leads to severe intelligibility and spectral detail loss, we introduce a Transformer-based inter-frame long-term dependency module and systematically explore residual vector quantization (RVQ) depth and codebook size to identify optimal configurations. Moreover, we apply U-Codec into a large language model (LLM)-based auto-regressive TTS model, which leverages global and local hierarchical architecture to effectively capture dependencies across multi-layer tokens. We extend LLM-based TTS from 3-layer RVQ at 50Hz to 32-layer RVQ at 5Hz. Experimental results demonstrate that U-Codec improves LLM-based TTS inference speed by around 3 $\times$ over high-frame-rate codecs while maintaining similarity and naturalness. These results validate the feasibility of using highly compressed 5Hz discrete tokens for fast and high-fidelity speech synthesis. 

---
# RGMem: Renormalization Group-based Memory Evolution for Language Agent User Profile 

**Authors**: Ao Tian, Yunfeng Lu, Xinxin Fan, Changhao Wang, Lanzhi Zhou, Yeyao Zhang, Yanfang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.16392)  

**Abstract**: Personalized and continuous interactions are the key to enhancing user experience in today's large language model (LLM)-based conversational systems, however, the finite context windows and static parametric memory make it difficult to model the cross-session long-term user states and behavioral consistency. Currently, the existing solutions to this predicament, such as retrieval-augmented generation (RAG) and explicit memory systems, primarily focus on fact-level storage and retrieval, lacking the capability to distill latent preferences and deep traits from the multi-turn dialogues, which limits the long-term and effective user modeling, directly leading to the personalized interactions remaining shallow, and hindering the cross-session continuity. To realize the long-term memory and behavioral consistency for Language Agents in LLM era, we propose a self-evolving memory framework RGMem, inspired by the ideology of classic renormalization group (RG) in physics, this framework enables to organize the dialogue history in multiple scales: it first extracts semantics and user insights from episodic fragments, then through hierarchical coarse-graining and rescaling operations, progressively forms a dynamically-evolved user profile. The core innovation of our work lies in modeling memory evolution as a multi-scale process of information compression and emergence, which accomplishes the high-level and accurate user profiles from noisy and microscopic-level interactions. 

---
