# Event Extraction in Large Language Model 

**Authors**: Bobo Li, Xudong Han, Jiang Liu, Yuzhe Ding, Liqiang Jing, Zhaoqi Zhang, Jinheng Li, Xinya Du, Fei Li, Meishan Zhang, Min Zhang, Aixin Sun, Philip S. Yu, Hao Fei  

**Link**: [PDF](https://arxiv.org/pdf/2512.19537)  

**Abstract**: Large language models (LLMs) and multimodal LLMs are changing event extraction (EE): prompting and generation can often produce structured outputs in zero shot or few shot settings. Yet LLM based pipelines face deployment gaps, including hallucinations under weak constraints, fragile temporal and causal linking over long contexts and across documents, and limited long horizon knowledge management within a bounded context window. We argue that EE should be viewed as a system component that provides a cognitive scaffold for LLM centered solutions. Event schemas and slot constraints create interfaces for grounding and verification; event centric structures act as controlled intermediate representations for stepwise reasoning; event links support relation aware retrieval with graph based RAG; and event stores offer updatable episodic and agent memory beyond the context window. This survey covers EE in text and multimodal settings, organizing tasks and taxonomy, tracing method evolution from rule based and neural models to instruction driven and generative frameworks, and summarizing formulations, decoding strategies, architectures, representations, datasets, and evaluation. We also review cross lingual, low resource, and domain specific settings, and highlight open challenges and future directions for reliable event centric systems. Finally, we outline open challenges and future directions that are central to the LLM era, aiming to evolve EE from static extraction into a structurally reliable, agent ready perception and memory layer for open world systems. 

---
# KVReviver: Reversible KV Cache Compression with Sketch-Based Token Reconstruction 

**Authors**: Aomufei Yuan, Zhiming Wang, Ruijie Miao, Dayu Wang, Yuxuan Tian, Zihan Wang, Yebo Peng, Yuhan Wu, Bairen Yi, Xin Liu, Tong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2512.17917)  

**Abstract**: As the context length of current large language models (LLMs) rapidly increases, the memory demand for the Key-Value (KV) cache is becoming a bottleneck for LLM deployment and batch processing. Traditional KV cache compression methods typically involve permanently evicting or irreversibly merging "less important" tokens with low attention scores. This approach results in the unrecoverable loss of token information, which we call Contextual Amnesia, significantly degrading the model's information retrieval capability. To address this issue, we propose KVReviver, a reversible KV cache compression method based on the sketch algorithm. This method allows reconstructing compressed tokens from an additional data structure, thus enabling full-scale computation within limited memory. Experiments showed that in 2k-length contexts, it requires only 10% of KV Cache budget while maintaining identical end-to-end inference accuracy. For 32k-length contexts, it achieves equivalent or comparable accuracy ~2% accuracy loss) using merely 25% of KV Cache budget. 

---
