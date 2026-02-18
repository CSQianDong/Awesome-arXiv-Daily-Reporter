# AIC CTU@AVerImaTeC: dual-retriever RAG for image-text fact checking 

**Authors**: Herbert Ullrich, Jan Drchal  

**Link**: [PDF](https://arxiv.org/pdf/2602.15190)  

**Abstract**: In this paper, we present our 3rd place system in the AVerImaTeC shared task, which combines our last year's retrieval-augmented generation (RAG) pipeline with a reverse image search (RIS) module. Despite its simplicity, our system delivers competitive performance with a single multimodal LLM call per fact-check at just $0.013 on average using GPT5.1 via OpenAI Batch API. Our system is also easy to reproduce and tweak, consisting of only three decoupled modules - a textual retrieval module based on similarity search, an image retrieval module based on API-accessed RIS, and a generation module using GPT5.1 - which is why we suggest it as an accesible starting point for further experimentation. We publish its code and prompts, as well as our vector stores and insights into the scheme's running costs and directions for further improvement. 

---
