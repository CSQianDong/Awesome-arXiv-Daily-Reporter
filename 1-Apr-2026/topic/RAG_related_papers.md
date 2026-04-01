# UnWeaving the knots of GraphRAG -- turns out VectorRAG is almost enough 

**Authors**: Ryszard Tuora, Mateusz Galiński, Michał Godziszewski, Michał Karpowicz, Mateusz Czyżnikiewicz, Adam Kozakiewicz, Tomasz Ziętkiewicz  

**Link**: [PDF](https://arxiv.org/pdf/2603.29875)  

**Abstract**: One of the key problems in Retrieval-augmented generation (RAG) systems is that chunk-based retrieval pipelines represent the source chunks as atomic objects, mixing the information contained within such a chunk into a single vector. These vector representations are then fundamentally treated as isolated, independent and self-sufficient, with no attempt to represent possible relations between them. Such an approach has no dedicated mechanisms for handling multi-hop questions. Graph-based RAG systems aimed to ameliorate this problem by modeling information as knowledge-graphs, with entities represented by nodes being connected by robust relations, and forming hierarchical communities. This approach however suffers from its own issues with some of them being: orders of magnitude increased componential complexity in order to create graph-based indices, and reliance on heuristics for performing retrieval. We propose UnWeaver, a novel RAG framework simplifying the idea of GraphRAG. UnWeaver disentangles the contents of the documents into entities which can occur across multiple chunks using an LLM. In the retrieval process entities are used as an intermediate way of recovering original text chunks hence preserving fidelity to the source material. We argue that entity-based decomposition yields a more distilled representation of original information, and additionally serves to reduce noise in the indexing, and generation process. 

---
# UltRAG: a Universal Simple Scalable Recipe for Knowledge Graph RAG 

**Authors**: Dobrik Georgiev, Kheeran Naidu, Alberto Cattaneo, Federico Monti, Carlo Luschi, Daniel Justus  

**Link**: [PDF](https://arxiv.org/pdf/2603.28773)  

**Abstract**: Large language models (LLMs) frequently generate confident yet factually incorrect content when used for language generation (a phenomenon often known as hallucination). Retrieval augmented generation (RAG) tries to reduce factual errors by identifying information in a knowledge corpus and putting it in the context window of the model. While this approach is well-established for document-structured data, it is non-trivial to adapt it for Knowledge Graphs (KGs), especially for queries that require multi-node/multi-hop reasoning on graphs. We introduce ULTRAG, a general framework for retrieving information from Knowledge Graphs that shifts away from classical RAG. By endowing LLMs with off-the-shelf neural query executing modules, we highlight how readily available language models can achieve state-of-the-art results on Knowledge Graph Question Answering (KGQA) tasks without any retraining of the LLM or executor involved. In our experiments, ULTRAG achieves better performance when compared to state-of-the-art KG-RAG solutions, and it enables language models to interface with Wikidata-scale graphs (116M entities, 1.6B relations) at comparable or lower costs. 

---
