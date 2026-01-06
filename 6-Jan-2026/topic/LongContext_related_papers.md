# Agentic Memory: Learning Unified Long-Term and Short-Term Memory Management for Large Language Model Agents 

**Authors**: Yi Yu, Liuyi Yao, Yuexiang Xie, Qingquan Tan, Jiaqi Feng, Yaliang Li, Libing Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.01885)  

**Abstract**: Large language model (LLM) agents face fundamental limitations in long-horizon reasoning due to finite context windows, making effective memory management critical. Existing methods typically handle long-term memory (LTM) and short-term memory (STM) as separate components, relying on heuristics or auxiliary controllers, which limits adaptability and end-to-end optimization. In this paper, we propose Agentic Memory (AgeMem), a unified framework that integrates LTM and STM management directly into the agent's policy. AgeMem exposes memory operations as tool-based actions, enabling the LLM agent to autonomously decide what and when to store, retrieve, update, summarize, or discard information. To train such unified behaviors, we propose a three-stage progressive reinforcement learning strategy and design a step-wise GRPO to address sparse and discontinuous rewards induced by memory operations. Experiments on five long-horizon benchmarks demonstrate that AgeMem consistently outperforms strong memory-augmented baselines across multiple LLM backbones, achieving improved task performance, higher-quality long-term memory, and more efficient context usage. 

---
# K-EXAONE Technical Report 

**Authors**: Eunbi Choi, Kibong Choi, Seokhee Hong, Junwon Hwang, Hyojin Jeon, Hyunjik Jo, Joonkee Kim, Seonghwan Kim, Soyeon Kim, Sunkyoung Kim, Yireun Kim, Yongil Kim, Haeju Lee, Jinsik Lee, Kyungmin Lee, Sangha Park, Heuiyeen Yeen, Hwan Chang, Stanley Jungkyu Choi, Yejin Choi, Jiwon Ham, Kijeong Jeon, Geunyeong Jeong, Gerrard Jeongwon Jo, Yonghwan Jo, Jiyeon Jung, Naeun Kang, Dohoon Kim, Euisoon Kim, Hayeon Kim, Hyosang Kim, Hyunseo Kim, Jieun Kim, Minu Kim, Myoungshin Kim, Unsol Kim, Youchul Kim, YoungJin Kim, Chaeeun Lee, Chaeyoon Lee, Changhun Lee, Dahm Lee, Edward Hwayoung Lee, Honglak Lee, Jinsang Lee, Jiyoung Lee, Sangeun Lee, Seungwon Lim, Solji Lim, Woohyung Lim, Chanwoo Moon, Jaewoo Park, Jinho Park, Yongmin Park, Hyerin Seo, Wooseok Seo, Yongwoo Song, Sejong Yang, Sihoon Yang, Chang En Yea, Sihyuk Yi, Chansik Yoon, Dongkeun Yoon, Sangyeon Yoon, Hyeongu Yun  

**Link**: [PDF](https://arxiv.org/pdf/2601.01739)  

**Abstract**: This technical report presents K-EXAONE, a large-scale multilingual language model developed by LG AI Research. K-EXAONE is built on a Mixture-of-Experts architecture with 236B total parameters, activating 23B parameters during inference. It supports a 256K-token context window and covers six languages: Korean, English, Spanish, German, Japanese, and Vietnamese. We evaluate K-EXAONE on a comprehensive benchmark suite spanning reasoning, agentic, general, Korean, and multilingual abilities. Across these evaluations, K-EXAONE demonstrates performance comparable to open-weight models of similar size. K-EXAONE, designed to advance AI for a better life, is positioned as a powerful proprietary AI foundation model for a wide range of industrial and research applications. 

---
# CogCanvas: Compression-Resistant Cognitive Artifacts for Long LLM Conversations 

**Authors**: Tao An  

**Link**: [PDF](https://arxiv.org/pdf/2601.00821)  

**Abstract**: Large language models face a fundamental tension between context window limits and information fidelity in long conversations. Existing approaches--truncation and summarization--either discard early information or lose nuanced details. We introduce CogCanvas, a training-free framework that extracts verbatim-grounded cognitive artifacts (decisions, facts, reminders) from conversation turns and organizes them into a temporal-aware graph for compression-resistant retrieval.
On the LoCoMo benchmark, CogCanvas achieves 34.7% overall accuracy, outperforming RAG (25.6%, +9.1pp) and GraphRAG (13.7%, +21.0pp). The advantage is most pronounced on temporal reasoning: 31.5% vs. 9.3% (RAG) and 5.0% (GraphRAG)--a +530% relative improvement. On multi-hop causal reasoning, CogCanvas achieves 81.0% pass rate vs. 40.0% for GraphRAG (+41.0pp). Controlled benchmarks show 97.5% recall (+78.5pp vs. summarization) with 93.0% exact match preservation.
While heavily-optimized approaches achieve higher absolute scores through dedicated training (EverMemOS: approximately 92%), our training-free approach provides practitioners with an immediately-deployable alternative that significantly outperforms standard baselines. Code and data: this https URL. 

---
# ScienceDB AI: An LLM-Driven Agentic Recommender System for Large-Scale Scientific Data Sharing Services 

**Authors**: Qingqing Long, Haotian Chen, Chenyang Zhao, Xiaolei Du, Xuezhi Wang, Pengyao Wang, Chengzan Li, Yuanchun Zhou, Hengshu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2601.01118)  

**Abstract**: The rapid growth of AI for Science (AI4S) has underscored the significance of scientific datasets, leading to the establishment of numerous national scientific data centers and sharing platforms. Despite this progress, efficiently promoting dataset sharing and utilization for scientific research remains challenging. Scientific datasets contain intricate domain-specific knowledge and contexts, rendering traditional collaborative filtering-based recommenders inadequate. Recent advances in Large Language Models (LLMs) offer unprecedented opportunities to build conversational agents capable of deep semantic understanding and personalized recommendations. In response, we present ScienceDB AI, a novel LLM-driven agentic recommender system developed on Science Data Bank (ScienceDB), one of the largest global scientific data-sharing platforms. ScienceDB AI leverages natural language conversations and deep reasoning to accurately recommend datasets aligned with researchers' scientific intents and evolving requirements. The system introduces several innovations: a Scientific Intention Perceptor to extract structured experimental elements from complicated queries, a Structured Memory Compressor to manage multi-turn dialogues effectively, and a Trustworthy Retrieval-Augmented Generation (Trustworthy RAG) framework. The Trustworthy RAG employs a two-stage retrieval mechanism and provides citable dataset references via Citable Scientific Task Record (CSTR) identifiers, enhancing recommendation trustworthiness and reproducibility. Through extensive offline and online experiments using over 10 million real-world datasets, ScienceDB AI has demonstrated significant effectiveness. To our knowledge, ScienceDB AI is the first LLM-driven conversational recommender tailored explicitly for large-scale scientific dataset sharing services. The platform is publicly accessible at: this https URL. 

---
# EverMemOS: A Self-Organizing Memory Operating System for Structured Long-Horizon Reasoning 

**Authors**: Chuanrui Hu, Xingze Gao, Zuyi Zhou, Dannong Xu, Yi Bai, Xintong Li, Hui Zhang, Tong Li, Chong Zhang, Lidong Bing, Yafeng Deng  

**Link**: [PDF](https://arxiv.org/pdf/2601.02163)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed as long-term interactive agents, yet their limited context windows make it difficult to sustain coherent behavior over extended interactions. Existing memory systems often store isolated records and retrieve fragments, limiting their ability to consolidate evolving user states and resolve conflicts. We introduce EverMemOS, a self-organizing memory operating system that implements an engram-inspired lifecycle for computational memory. Episodic Trace Formation converts dialogue streams into MemCells that capture episodic traces, atomic facts, and time-bounded Foresight signals. Semantic Consolidation organizes MemCells into thematic MemScenes, distilling stable semantic structures and updating user profiles. Reconstructive Recollection performs MemScene-guided agentic retrieval to compose the necessary and sufficient context for downstream reasoning. Experiments on LoCoMo and LongMemEval show that EverMemOS achieves state-of-the-art performance on memory-augmented reasoning tasks. We further report a profile study on PersonaMem v2 and qualitative case studies illustrating chat-oriented capabilities such as user profiling and Foresight. Code is available at this https URL. 

---
# Not All Needles Are Found: How Fact Distribution and Don't Make It Up Prompts Shape Literal Extraction, Logical Inference, and Hallucination Risks in Long-Context LLMs 

**Authors**: Amirali Ebrahimzadeh, Seyyed M. Salili  

**Link**: [PDF](https://arxiv.org/pdf/2601.02023)  

**Abstract**: Large language models (LLMs) increasingly support very long input contexts. Yet it remains unclear how reliably they extract and infer information at scale. Performance varies with context length and strongly interacts with how information is distributed in real-world corpora. Motivated by these observations, we study how fact placement, corpus-level fact distributions, and Don't Make It Up prompts influence model behavior. We introduce an extended needle-in-a-haystack benchmark across four production-scale models: Gemini-2.5-flash, ChatGPT-5-mini, Claude-4.5-haiku, and Deepseek-v3.2-chat. Unlike prior work, we separately evaluate literal extraction, logical inference, and hallucination risk. Our study considers both positional effects and realistic distributions of evidence across long contexts, as well as prompts that explicitly discourage fabrication. We find that longer contexts alone do not guarantee better performance and can be detrimental when relevant evidence is diluted or widely dispersed. Performance varies substantially across models: some show severe degradation under realistic conditions, while others remain more robust at longer context lengths. Anti-hallucination (AH) instructions can make some models overly conservative, sharply reducing accuracy in literal extraction and logical inference. While we do not directly compare retrieval-augmented generation (RAG) and cache-augmented generation (CAG), our results suggest many failures stem from ineffective context utilization. Models often struggle to identify and prioritize relevant information even when it is present. These findings have direct practical implications, as enterprise workflows increasingly involve pasting large volumes of unfiltered documents into LLM prompts. Effective context length and model-specific robustness to long contexts are therefore critical for reliable LLM deployment in research and business. 

---
# MOSS Transcribe Diarize: Accurate Transcription with Speaker Diarization 

**Authors**: Donghua Yu, Zhengyuan Lin, Chen Yang, Yiyang Zhang, Zhaoye Fei, Hanfu Chen, Jingqi Chen, Ke Chen, Qinyuan Cheng, Liwei Fan, Yi Jiang, Jie Zhu, Muchen Li, Shimin Li, Wenxuan Wang, Yang Wang, Zhe Xu, Yitian Gong, Yuqian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.01554)  

**Abstract**: Speaker-Attributed, Time-Stamped Transcription (SATS) aims to transcribe what is said and to precisely determine the timing of each speaker, which is particularly valuable for meeting transcription. Existing SATS systems rarely adopt an end-to-end formulation and are further constrained by limited context windows, weak long-range speaker memory, and the inability to output timestamps. To address these limitations, we present MOSS Transcribe Diarize, a unified multimodal large language model that jointly performs Speaker-Attributed, Time-Stamped Transcription in an end-to-end paradigm. Trained on extensive real wild data and equipped with a 128k context window for up to 90-minute inputs, MOSS Transcribe Diarize scales well and generalizes robustly. Across comprehensive evaluations, it outperforms state-of-the-art commercial systems on multiple public and in-house benchmarks. 

---
# Benchmarking the Computational and Representational Efficiency of State Space Models against Transformers on Long-Context Dyadic Sessions 

**Authors**: Abidemi Koledoye, Chinemerem Unachukwu, Gold Nwobu, Hasin Rana  

**Link**: [PDF](https://arxiv.org/pdf/2601.01237)  

**Abstract**: State Space Models (SSMs) have emerged as a promising alternative to Transformers for long-context sequence modeling, offering linear $O(N)$ computational complexity compared to the Transformer's quadratic $O(N^2)$ scaling. This paper presents a comprehensive benchmarking study comparing the Mamba SSM against the LLaMA Transformer on long-context sequences, using dyadic therapy sessions as a representative test case. We evaluate both architectures across two dimensions: (1) computational efficiency, where we measure memory usage and inference speed from 512 to 8,192 tokens, and (2) representational efficiency, where we analyze hidden state dynamics and attention patterns. Our findings provide actionable insights for practitioners working with long-context applications, establishing precise conditions under which SSMs offer advantages over Transformers. 

---
