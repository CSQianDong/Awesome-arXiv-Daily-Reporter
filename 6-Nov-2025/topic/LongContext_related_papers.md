# SnapStream: Efficient Long Sequence Decoding on Dataflow Accelerators 

**Authors**: Jonathan Li, Nasim Farahini, Evgenii Iuliugin, Magnus Vesterlund, Christian Haggstrom, Guangtao Wang, Shubhangi Upasani, Ayush Sachdeva, Rui Li, Faline Fu, Chen Wu, Ayesha Siddiqua, John Long, Tuowen Zhao, Matheen Musaddiq, Hakan Zeffer, Yun Du, Mingran Wang, Qinghua Li, Bo Li, Urmish Thakker, Raghu Prabhakar  

**Link**: [PDF](https://arxiv.org/pdf/2511.03092)  

**Abstract**: The proliferation of 100B+ parameter Large Language Models (LLMs) with 100k+ context length support have resulted in increasing demands for on-chip memory to support large KV caches. Techniques such as StreamingLLM and SnapKV demonstrate how to control KV cache size while maintaining model accuracy. Yet, these techniques are not commonly used within industrial deployments using frameworks like vLLM or SGLang. The reason is twofold: on one hand, the static graphs and continuous batching methodology employed by these frameworks make it difficult to admit modifications to the standard multi-head attention algorithm, while on the other hand, the accuracy implications of such techniques on modern instruction-following and reasoning models are not well understood, obfuscating the need for implementing these techniques. In this paper, we explore these accuracy implications on Llama-3.1-8B-Instruct and DeepSeek-R1, and develop SnapStream, a KV cache compression method that can be deployed at scale. We demonstrate the efficacy of SnapStream in a 16-way tensor-parallel deployment of DeepSeek-671B on SambaNova SN40L accelerators running at 128k context length and up to 1832 tokens per second in a real production setting. SnapStream enables $4\times$ improved on-chip memory usage and introduces minimal accuracy degradation on LongBench-v2, AIME24 and LiveCodeBench. To the best of our knowledge, this is the first implementation of sparse KV attention techniques deployed in a production inference system with static graphs and continuous batching. 

---
# LGM: Enhancing Large Language Models with Conceptual Meta-Relations and Iterative Retrieval 

**Authors**: Wenchang Lei, Ping Zou, Yue Wang, Feng Sun, Lei Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.03214)  

**Abstract**: Large language models (LLMs) exhibit strong semantic understanding, yet struggle when user instructions involve ambiguous or conceptually misaligned terms. We propose the Language Graph Model (LGM) to enhance conceptual clarity by extracting meta-relations-inheritance, alias, and composition-from natural language. The model further employs a reflection mechanism to validate these meta-relations. Leveraging a Concept Iterative Retrieval Algorithm, these relations and related descriptions are dynamically supplied to the LLM, improving its ability to interpret concepts and generate accurate responses. Unlike conventional Retrieval-Augmented Generation (RAG) approaches that rely on extended context windows, our method enables large language models to process texts of any length without the need for truncation. Experiments on standard benchmarks demonstrate that the LGM consistently outperforms existing RAG baselines. 

---
