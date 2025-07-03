# BioMARS: A Multi-Agent Robotic System for Autonomous Biological Experiments 

**Authors**: Yibo Qiu, Zan Huang, Zhiyu Wang, Handi Liu, Yiling Qiao, Yifeng Hu, Shu'ang Sun, Hangke Peng, Ronald X Xu, Mingzhai Sun  

**Link**: [PDF](https://arxiv.org/pdf/2507.01485)  

**Abstract**: Large language models (LLMs) and vision-language models (VLMs) have the potential to transform biological research by enabling autonomous experimentation. Yet, their application remains constrained by rigid protocol design, limited adaptability to dynamic lab conditions, inadequate error handling, and high operational complexity. Here we introduce BioMARS (Biological Multi-Agent Robotic System), an intelligent platform that integrates LLMs, VLMs, and modular robotics to autonomously design, plan, and execute biological experiments. BioMARS uses a hierarchical architecture: the Biologist Agent synthesizes protocols via retrieval-augmented generation; the Technician Agent translates them into executable robotic pseudo-code; and the Inspector Agent ensures procedural integrity through multimodal perception and anomaly detection. The system autonomously conducts cell passaging and culture tasks, matching or exceeding manual performance in viability, consistency, and morphological integrity. It also supports context-aware optimization, outperforming conventional strategies in differentiating retinal pigment epithelial cells. A web interface enables real-time human-AI collaboration, while a modular backend allows scalable integration with laboratory hardware. These results highlight the feasibility of generalizable, AI-driven laboratory automation and the transformative role of language-based reasoning in biological research. 

---
# Rethinking All Evidence: Enhancing Trustworthy Retrieval-Augmented Generation via Conflict-Driven Summarization 

**Authors**: Juan Chen, Baolong Bi, Wei Zhang, Jingyan Sui, Xiaofei Zhu, Yuanzhuo Wang, Lingrui Mei, Shenghua Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.01281)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by integrating their parametric knowledge with external retrieved content. However, knowledge conflicts caused by internal inconsistencies or noisy retrieved content can severely undermine the generation reliability of RAG this http URL this work, we argue that LLMs should rethink all evidence, including both retrieved content and internal knowledge, before generating this http URL propose CARE-RAG (Conflict-Aware and Reliable Evidence for RAG), a novel framework that improves trustworthiness through Conflict-Driven Summarization of all available this http URL-RAG first derives parameter-aware evidence by comparing parameter records to identify diverse internal perspectives. It then refines retrieved evidences to produce context-aware evidence, removing irrelevant or misleading content. To detect and summarize conflicts, we distill a 3B LLaMA3.2 model to perform conflict-driven summarization, enabling reliable synthesis across multiple this http URL further ensure evaluation integrity, we introduce a QA Repair step to correct outdated or ambiguous benchmark this http URL on revised QA datasets with retrieval data show that CARE-RAG consistently outperforms strong RAG baselines, especially in scenarios with noisy or conflicting evidence. 

---
# A Data Science Approach to Calcutta High Court Judgments: An Efficient LLM and RAG-powered Framework for Summarization and Similar Cases Retrieval 

**Authors**: Puspendu Banerjee, Aritra Mazumdar, Wazib Ansar, Saptarsi Goswami, Amlan Chakrabarti  

**Link**: [PDF](https://arxiv.org/pdf/2507.01058)  

**Abstract**: The judiciary, as one of democracy's three pillars, is dealing with a rising amount of legal issues, needing careful use of judicial resources. This research presents a complex framework that leverages Data Science methodologies, notably Large Language Models (LLM) and Retrieval-Augmented Generation (RAG) techniques, to improve the efficiency of analyzing Calcutta High Court verdicts. Our framework focuses on two key aspects: first, the creation of a robust summarization mechanism that distills complex legal texts into concise and coherent summaries; and second, the development of an intelligent system for retrieving similar cases, which will assist legal professionals in research and decision making. By fine-tuning the Pegasus model using case head note summaries, we achieve significant improvements in the summarization of legal cases. Our two-step summarizing technique preserves crucial legal contexts, allowing for the production of a comprehensive vector database for RAG. The RAG-powered framework efficiently retrieves similar cases in response to user queries, offering thorough overviews and summaries. This technique not only improves legal research efficiency, but it also helps legal professionals and students easily acquire and grasp key legal information, benefiting the overall legal scenario. 

---
# Frustratingly Simple Retrieval Improves Challenging, Reasoning-Intensive Benchmarks 

**Authors**: Xinxi Lyu, Michael Duan, Rulin Shao, Pang Wei Koh, Sewon Min  

**Link**: [PDF](https://arxiv.org/pdf/2507.01297)  

**Abstract**: Retrieval-augmented Generation (RAG) has primarily been studied in limited settings, such as factoid question answering; more challenging, reasoning-intensive benchmarks have seen limited success from minimal RAG. In this work, we challenge this prevailing view on established, reasoning-intensive benchmarks: MMLU, MMLU Pro, AGI Eval, GPQA, and MATH. We identify a key missing component in prior work: a usable, web-scale datastore aligned with the breadth of pretraining data. To this end, we introduce CompactDS: a diverse, high-quality, web-scale datastore that achieves high retrieval accuracy and subsecond latency on a single-node. The key insights are (1) most web content can be filtered out without sacrificing coverage, and a compact, high-quality subset is sufficient; and (2) combining in-memory approximate nearest neighbor (ANN) retrieval and on-disk exact search balances speed and recall. Using CompactDS, we show that a minimal RAG pipeline achieves consistent accuracy improvements across all benchmarks and model sizes (8B--70B), with relative gains of 10% on MMLU, 33% on MMLU Pro, 14% on GPQA, and 19% on MATH. No single data source suffices alone, highlighting the importance of diversity of sources (web crawls, curated math, academic papers, textbooks). Finally, we show that our carefully designed in-house datastore matches or outperforms web search engines such as Google Search, as well as recently proposed, complex agent-based RAG systems--all while maintaining simplicity, reproducibility, and self-containment. We release CompactDS and our retrieval pipeline, supporting future research exploring retrieval-based AI systems. 

---
