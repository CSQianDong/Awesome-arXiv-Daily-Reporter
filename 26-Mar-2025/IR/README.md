# GENIUS: A Generative Framework for Universal Multimodal Search 

**Authors**: Sungyeon Kim, Xinliang Zhu, Xiaofan Lin, Muhammet Bastan, Douglas Gray, Suha Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2503.19868)  

**Abstract**: Generative retrieval is an emerging approach in information retrieval that generates identifiers (IDs) of target data based on a query, providing an efficient alternative to traditional embedding-based retrieval methods. However, existing models are task-specific and fall short of embedding-based retrieval in performance. This paper proposes GENIUS, a universal generative retrieval framework supporting diverse tasks across multiple modalities and domains. At its core, GENIUS introduces modality-decoupled semantic quantization, transforming multimodal data into discrete IDs encoding both modality and semantics. Moreover, to enhance generalization, we propose a query augmentation that interpolates between a query and its target, allowing GENIUS to adapt to varied query forms. Evaluated on the M-BEIR benchmark, it surpasses prior generative methods by a clear margin. Unlike embedding-based retrieval, GENIUS consistently maintains high retrieval speed across database size, with competitive performance across multiple benchmarks. With additional re-ranking, GENIUS often achieves results close to those of embedding-based methods while preserving efficiency. 

---
# How Generative IR Retrieves Documents Mechanistically 

**Authors**: Anja Reusch, Yonatan Belinkov  

**Link**: [PDF](https://arxiv.org/pdf/2503.19715)  

**Abstract**: Generative Information Retrieval (GenIR) is a novel paradigm in which a transformer encoder-decoder model predicts document rankings based on a query in an end-to-end fashion. These GenIR models have received significant attention due to their simple retrieval architecture while maintaining high retrieval effectiveness. However, in contrast to established retrieval architectures like cross-encoders or bi-encoders, their internal computations remain largely unknown. Therefore, this work studies the internal retrieval process of GenIR models by applying methods based on mechanistic interpretability, such as patching and vocabulary projections. By replacing the GenIR encoder with one trained on fewer documents, we demonstrate that the decoder is the primary component responsible for successful retrieval. Our patching experiments reveal that not all components in the decoder are crucial for the retrieval process. More specifically, we find that a pass through the decoder can be divided into three stages: (I) the priming stage, which contributes important information for activating subsequent components in later layers; (II) the bridging stage, where cross-attention is primarily active to transfer query information from the encoder to the decoder; and (III) the interaction stage, where predominantly MLPs are active to predict the document identifier. Our findings indicate that interaction between query and document information occurs only in the last stage. We hope our results promote a better understanding of GenIR models and foster future research to overcome the current challenges associated with these models. 

---
# Beyond Relevance: An Adaptive Exploration-Based Framework for Personalized Recommendations 

**Authors**: Edoardo Bianchi  

**Link**: [PDF](https://arxiv.org/pdf/2503.19525)  

**Abstract**: Recommender systems must balance personalization, diversity, and robustness to cold-start scenarios to remain effective in dynamic content environments. This paper introduces an adaptive, exploration-based recommendation framework that adjusts to evolving user preferences and content distributions to promote diversity and novelty without compromising relevance. The system represents items using sentence-transformer embeddings and organizes them into semantically coherent clusters through an online algorithm with adaptive thresholding. A user-controlled exploration mechanism enhances diversity by selectively sampling from under-explored clusters. Experiments on the MovieLens dataset show that enabling exploration reduces intra-list similarity from 0.34 to 0.26 and increases unexpectedness to 0.73, outperforming collaborative filtering and popularity-based baselines. A/B testing with 300 simulated users reveals a strong link between interaction history and preference for diversity, with 72.7% of long-term users favoring exploratory recommendations. Computational analysis confirms that clustering and recommendation processes scale linearly with the number of clusters. These results demonstrate that adaptive exploration effectively mitigates over-specialization while preserving personalization and efficiency. 

---
# Enhanced Bloom's Educational Taxonomy for Fostering Information Literacy in the Era of Large Language Models 

**Authors**: Yiming Luo, Ting Liu, Patrick Cheong-Iao Pang, Dana McKay, Ziqi Chen, George Buchanan, Shanton Chang  

**Link**: [PDF](https://arxiv.org/pdf/2503.19434)  

**Abstract**: The advent of Large Language Models (LLMs) has profoundly transformed the paradigms of information retrieval and problem-solving, enabling students to access information acquisition more efficiently to support learning. However, there is currently a lack of standardized evaluation frameworks that guide learners in effectively leveraging LLMs. This paper proposes an LLM-driven Bloom's Educational Taxonomy that aims to recognize and evaluate students' information literacy (IL) with LLMs, and to formalize and guide students practice-based activities of using LLMs to solve complex problems. The framework delineates the IL corresponding to the cognitive abilities required to use LLM into two distinct stages: Exploration & Action and Creation & Metacognition. It further subdivides these into seven phases: Perceiving, Searching, Reasoning, Interacting, Evaluating, Organizing, and Curating. Through the case presentation, the analysis demonstrates the framework's applicability and feasibility, supporting its role in fostering IL among students with varying levels of prior knowledge. This framework fills the existing gap in the analysis of LLM usage frameworks and provides theoretical support for guiding learners to improve IL. 

---
# RGL: A Graph-Centric, Modular Framework for Efficient Retrieval-Augmented Generation on Graphs 

**Authors**: Yuan Li, Jun Hu, Jiaxin Jiang, Zemin Liu, Bryan Hooi, Bingsheng He  

**Link**: [PDF](https://arxiv.org/pdf/2503.19314)  

**Abstract**: Recent advances in graph learning have paved the way for innovative retrieval-augmented generation (RAG) systems that leverage the inherent relational structures in graph data. However, many existing approaches suffer from rigid, fixed settings and significant engineering overhead, limiting their adaptability and scalability. Additionally, the RAG community has largely overlooked the decades of research in the graph database community regarding the efficient retrieval of interesting substructures on large-scale graphs. In this work, we introduce the RAG-on-Graphs Library (RGL), a modular framework that seamlessly integrates the complete RAG pipeline-from efficient graph indexing and dynamic node retrieval to subgraph construction, tokenization, and final generation-into a unified system. RGL addresses key challenges by supporting a variety of graph formats and integrating optimized implementations for essential components, achieving speedups of up to 143x compared to conventional methods. Moreover, its flexible utilities, such as dynamic node filtering, allow for rapid extraction of pertinent subgraphs while reducing token consumption. Our extensive evaluations demonstrate that RGL not only accelerates the prototyping process but also enhances the performance and applicability of graph-based RAG systems across a range of tasks. 

---
# Rankers, Judges, and Assistants: Towards Understanding the Interplay of LLMs in Information Retrieval Evaluation 

**Authors**: Krisztian Balog, Donald Metzler, Zhen Qin  

**Link**: [PDF](https://arxiv.org/pdf/2503.19092)  

**Abstract**: Large language models (LLMs) are increasingly integral to information retrieval (IR), powering ranking, evaluation, and AI-assisted content creation. This widespread adoption necessitates a critical examination of potential biases arising from the interplay between these LLM-based components. This paper synthesizes existing research and presents novel experiment designs that explore how LLM-based rankers and assistants influence LLM-based judges. We provide the first empirical evidence of LLM judges exhibiting significant bias towards LLM-based rankers. Furthermore, we observe limitations in LLM judges' ability to discern subtle system performance differences. Contrary to some previous findings, our preliminary study does not find evidence of bias against AI-generated content. These results highlight the need for a more holistic view of the LLM-driven information ecosystem. To this end, we offer initial guidelines and a research agenda to ensure the reliable use of LLMs in IR evaluation. 

---
# CoLLM: A Large Language Model for Composed Image Retrieval 

**Authors**: Chuong Huynh, Jinyu Yang, Ashish Tawari, Mubarak Shah, Son Tran, Raffay Hamid, Trishul Chilimbi, Abhinav Shrivastava  

**Link**: [PDF](https://arxiv.org/pdf/2503.19910)  

**Abstract**: Composed Image Retrieval (CIR) is a complex task that aims to retrieve images based on a multimodal query. Typical training data consists of triplets containing a reference image, a textual description of desired modifications, and the target image, which are expensive and time-consuming to acquire. The scarcity of CIR datasets has led to zero-shot approaches utilizing synthetic triplets or leveraging vision-language models (VLMs) with ubiquitous web-crawled image-caption pairs. However, these methods have significant limitations: synthetic triplets suffer from limited scale, lack of diversity, and unnatural modification text, while image-caption pairs hinder joint embedding learning of the multimodal query due to the absence of triplet data. Moreover, existing approaches struggle with complex and nuanced modification texts that demand sophisticated fusion and understanding of vision and language modalities. We present CoLLM, a one-stop framework that effectively addresses these limitations. Our approach generates triplets on-the-fly from image-caption pairs, enabling supervised training without manual annotation. We leverage Large Language Models (LLMs) to generate joint embeddings of reference images and modification texts, facilitating deeper multimodal fusion. Additionally, we introduce Multi-Text CIR (MTCIR), a large-scale dataset comprising 3.4M samples, and refine existing CIR benchmarks (CIRR and Fashion-IQ) to enhance evaluation reliability. Experimental results demonstrate that CoLLM achieves state-of-the-art performance across multiple CIR benchmarks and settings. MTCIR yields competitive results, with up to 15% performance improvement. Our refined benchmarks provide more reliable evaluation metrics for CIR models, contributing to the advancement of this important field. 

---
# CausalRAG: Integrating Causal Graphs into Retrieval-Augmented Generation 

**Authors**: Nengbo Wang, Xiaotian Han, Jagdip Singh, Jing Ma, Vipin Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2503.19878)  

**Abstract**: Large language models (LLMs) have revolutionized natural language processing (NLP), particularly through Retrieval-Augmented Generation (RAG), which enhances LLM capabilities by integrating external knowledge. However, traditional RAG systems face critical limitations, including disrupted contextual integrity due to text chunking, and over-reliance on semantic similarity for retrieval. To address these issues, we propose CausalRAG, a novel framework that incorporates causal graphs into the retrieval process. By constructing and tracing causal relationships, CausalRAG preserves contextual continuity and improves retrieval precision, leading to more accurate and interpretable responses. We evaluate CausalRAG against regular RAG and graph-based RAG approaches, demonstrating its superiority across several metrics. Our findings suggest that grounding retrieval in causal reasoning provides a promising approach to knowledge-intensive tasks. 

---
# Context-Efficient Retrieval with Factual Decomposition 

**Authors**: Yanhong Li, David Yunis, David McAllester, Jiawei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.19574)  

**Abstract**: There has recently been considerable interest in incorporating information retrieval into large language models (LLMs). Retrieval from a dynamically expanding external corpus of text allows a model to incorporate current events and can be viewed as a form of episodic memory. Here we demonstrate that pre-processing the external corpus into semi-structured ''atomic facts'' makes retrieval more efficient. More specifically, we demonstrate that our particular form of atomic facts improves performance on various question answering tasks when the amount of retrieved text is limited. Limiting the amount of retrieval reduces the size of the context and improves inference efficiency. 

---
# CoMAC: Conversational Agent for Multi-Source Auxiliary Context with Sparse and Symmetric Latent Interactions 

**Authors**: Junfeng Liu, Christopher T. Symons, Ranga Raju Vatsavai  

**Link**: [PDF](https://arxiv.org/pdf/2503.19274)  

**Abstract**: Recent advancements in AI-driven conversational agents have exhibited immense potential of AI applications. Effective response generation is crucial to the success of these agents. While extensive research has focused on leveraging multiple auxiliary data sources (e.g., knowledge bases and personas) to enhance response generation, existing methods often struggle to efficiently extract relevant information from these sources. There are still clear limitations in the ability to combine versatile conversational capabilities with adherence to known facts and adaptation to large variations in user preferences and belief systems, which continues to hinder the wide adoption of conversational AI tools. This paper introduces a novel method, Conversational Agent for Multi-Source Auxiliary Context with Sparse and Symmetric Latent Interactions (CoMAC), for conversation generation, which employs specialized encoding streams and post-fusion grounding networks for multiple data sources to identify relevant persona and knowledge information for the conversation. CoMAC also leverages a novel text similarity metric that allows bi-directional information sharing among multiple sources and focuses on a selective subset of meaningful words. Our experiments show that CoMAC improves the relevant persona and knowledge prediction accuracies and response generation quality significantly over two state-of-the-art methods. 

---
# Browsing Lost Unformed Recollections: A Benchmark for Tip-of-the-Tongue Search and Reasoning 

**Authors**: Sky CH-Wang, Darshan Deshpande, Smaranda Muresan, Anand Kannappan, Rebecca Qian  

**Link**: [PDF](https://arxiv.org/pdf/2503.19193)  

**Abstract**: We introduce Browsing Lost Unformed Recollections, a tip-of-the-tongue known-item search and reasoning benchmark for general AI assistants. BLUR introduces a set of 573 real-world validated questions that demand searching and reasoning across multi-modal and multilingual inputs, as well as proficient tool use, in order to excel on. Humans easily ace these questions (scoring on average 98%), while the best-performing system scores around 56%. To facilitate progress toward addressing this challenging and aspirational use case for general AI assistants, we release 350 questions through a public leaderboard, retain the answers to 250 of them, and have the rest as a private test set. 

---
# Understanding and Improving Information Preservation in Prompt Compression for LLMs 

**Authors**: Weronika Łajewska, Momchil Hardalov, Laura Aina, Neha Anna John, Hang Su, Lluís Màrquez  

**Link**: [PDF](https://arxiv.org/pdf/2503.19114)  

**Abstract**: Recent advancements in large language models (LLMs) have enabled their successful application to a broad range of tasks. However, in information-intensive tasks, the prompt length can grow fast, leading to increased computational requirements, performance degradation, and induced biases from irrelevant or redundant information. Recently, various prompt compression techniques have been introduced to optimize the trade-off between reducing input length and retaining performance. We propose a holistic evaluation framework that allows for in-depth analysis of prompt compression methods. We focus on three key aspects, besides compression ratio: (i) downstream task performance, (ii) grounding in the input context, and (iii) information preservation. Through this framework, we investigate state-of-the-art soft and hard compression methods, showing that they struggle to preserve key details from the original prompt, limiting their performance on complex tasks. We demonstrate that modifying soft prompting methods to control better the granularity of the compressed information can significantly improve their effectiveness -- up to +23\% in downstream task performance, more than +8 BERTScore points in grounding, and 2.7x more entities preserved in compression. 

---
