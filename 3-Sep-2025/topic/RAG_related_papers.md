# Identifying Origins of Place Names via Retrieval Augmented Generation 

**Authors**: Alexis Horde Vo, Matt Duckham, Estrid He, Rafe Benli  

**Link**: [PDF](https://arxiv.org/pdf/2509.01030)  

**Abstract**: Who is the "Batman" behind "Batman Street" in Melbourne? Understanding the historical, cultural, and societal narratives behind place names can reveal the rich context that has shaped a community. Although place names serve as essential spatial references in gazetteers, they often lack information about place name origins. Enriching these place names in today's gazetteers is a time-consuming, manual process that requires extensive exploration of a vast archive of documents and text sources. Recent advances in natural language processing and language models (LMs) hold the promise of significant automation of identifying place name origins due to their powerful capability to exploit the semantics of the stored documents. This chapter presents a retrieval augmented generation pipeline designed to search for place name origins over a broad knowledge base, DBpedia. Given a spatial query, our approach first extracts sub-graphs that may contain knowledge relevant to the query; then ranks the extracted sub-graphs to generate the final answer to the query using fine-tuned LM-based models (i.e., ColBERTv2 and Llama2). Our results highlight the key challenges facing automated retrieval of place name origins, especially the tendency of language models to under-use the spatial information contained in texts as a discriminating factor. Our approach also frames the wider implications for geographic information retrieval using retrieval augmented generation. 

---
# How to Make Museums More Interactive? Case Study of Artistic Chatbot 

**Authors**: Filip J. Kucia, Bartosz Grabek, Szymon D. Trochimiak, Anna Wróblewska  

**Link**: [PDF](https://arxiv.org/pdf/2509.00572)  

**Abstract**: Conversational agents powered by Large Language Models (LLMs) are increasingly utilized in educational settings, in particular in individual closed digital environments, yet their potential adoption in the physical learning environments like cultural heritage sites, museums, and art galleries remains relatively unexplored. In this study, we present Artistic Chatbot, a voice-to-voice RAG-powered chat system to support informal learning and enhance visitor engagement during a live art exhibition celebrating the 15th anniversary of the Faculty of Media Art at the Warsaw Academy of Fine Arts, Poland. The question answering (QA) chatbot responded to free-form spoken questions in Polish using the context retrieved from a curated, domain-specific knowledge base consisting of 226 documents provided by the organizers, including faculty information, art magazines, books, and journals. We describe the key aspects of the system architecture and user interaction design, as well as discuss the practical challenges associated with deploying chatbots at public cultural sites. Our findings, based on interaction analysis, demonstrate that chatbots such as Artistic Chatbot effectively maintain responses grounded in exhibition content (60\% of responses directly relevant), even when faced with unpredictable queries outside the target domain, showing their potential for increasing interactivity in public cultural sites.
GitHub project page: this https URL 

---
# OpinioRAG: Towards Generating User-Centric Opinion Highlights from Large-scale Online Reviews 

**Authors**: Mir Tafseer Nayeem, Davood Rafiei  

**Link**: [PDF](https://arxiv.org/pdf/2509.00285)  

**Abstract**: We study the problem of opinion highlights generation from large volumes of user reviews, often exceeding thousands per entity, where existing methods either fail to scale or produce generic, one-size-fits-all summaries that overlook personalized needs. To tackle this, we introduce OpinioRAG, a scalable, training-free framework that combines RAG-based evidence retrieval with LLMs to efficiently produce tailored summaries. Additionally, we propose novel reference-free verification metrics designed for sentiment-rich domains, where accurately capturing opinions and sentiment alignment is essential. These metrics offer a fine-grained, context-sensitive assessment of factual consistency. To facilitate evaluation, we contribute the first large-scale dataset of long-form user reviews, comprising entities with over a thousand reviews each, paired with unbiased expert summaries and manually annotated queries. Through extensive experiments, we identify key challenges, provide actionable insights into improving systems, pave the way for future research, and position OpinioRAG as a robust framework for generating accurate, relevant, and structured summaries at scale. 

---
# CMRAG: Co-modality-based document retrieval and visual question answering 

**Authors**: Wang Chen, Guanqiang Qi, Weikang Li, Yang Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.02123)  

**Abstract**: Retrieval-Augmented Generation (RAG) has become a core paradigm in document question answering tasks. However, existing methods have limitations when dealing with multimodal documents: one category of methods relies on layout analysis and text extraction, which can only utilize explicit text information and struggle to capture images or unstructured content; the other category treats document segmentation as visual input and directly passes it to visual language models (VLMs) for processing, yet it ignores the semantic advantages of text, leading to suboptimal generation results. This paper proposes co-modality-based RAG (CMRAG), which can simultaneously leverage text and images for efficient retrieval and generation. Specifically, we first perform structured parsing on documents to obtain co-modality representations of text segments and image regions. Subsequently, in response to user queries, we retrieve candidate evidence from text and image channels, respectively, and aggregate the results at the cross-modal retrieval level. Finally, we prompt the VLM to generate the final response based on the co-modality retrieval results. Experiments demonstrate that our method significantly outperforms pure-vision-based RAG in visual document question answering tasks. The findings of this paper show that integrating co-modality information into the RAG framework in a unified manner is an effective approach to improving the performance of complex document visual question-answering (VQA) systems. 

---
# Privacy-Preserving Reasoning with Knowledge-Distilled Parametric Retrieval Augmented Generation 

**Authors**: Jinwen Chen, Hainan Zhang, Liang Pang, Yongxin Tong, Haibo Zhou, Yuan Zhan, Wei Lin, Zhiming Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.01088)  

**Abstract**: The current RAG system requires uploading plaintext documents to the cloud, risking private data leakage. Parametric RAG (PRAG) addresses this by encoding documents as LoRA within LLMs, enabling reasoning without exposing raw content. However, it still faces two issues: (1) PRAG demands synthesizing QA pairs and fine-tuning LLM for each individual document to create its corresponding LoRA, leading to unacceptable inference latency. (2) The performance of PRAG relies solely on synthetic QA data, lacking internal alignment with standard RAG, resulting in poor generalization on out-of-distribution(OOD) inputs. Therefore, achieving high-efficiency parameterization while maintaining RAG-level performance remains a critical challenge for privacy-preserving reasoning. In this paper, we propose DistilledPRAG, a generalizable knowledge-distilled parametric RAG model aligned with standard RAG in document structure and parameter activation. We first synthesize QA pairs from single and multi-documents to enhance cross-document reasoning. Then, we mask the plaintext documents with a special token and translate them to LoRA via a parameter generator, maintaining the standard RAG document structure. Finally, guided by synthetic QA data, we train the parameter generator to match standard RAG's hidden states and output logits, enabling RAG-style reasoning without original documents. Experiments on four QA datasets show that DistilledPRAG outperforms baselines in accuracy and generalizes well on OOD data. 

---
# MeVe: A Modular System for Memory Verification and Effective Context Control in Language Models 

**Authors**: Andreas Ottem  

**Link**: [PDF](https://arxiv.org/pdf/2509.01514)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems typically face constraints because of their inherent mechanism: a simple top-k semantic search [1]. The approach often leads to the incorporation of irrelevant or redundant information in the context, degrading performance and efficiency [10][11]. This paper presents MeVe, a novel modular architecture intended for Memory Verification and smart context composition. MeVe rethinks the RAG paradigm by proposing a five-phase modular design that distinctly breaks down the retrieval and context composition process into distinct, auditable, and independently tunable phases: initial retrieval, relevance verification, fallback retrieval, context prioritization, and token budgeting. This architecture enables fine-grained control of what knowledge is made available to an LLM, enabling task-dependent filtering and adaptation. We release a reference implementation of MeVe as a proof of concept and evaluate its performance on knowledge-heavy QA tasks over a subset of English Wikipedia [22]. Our results demonstrate that by actively verifying information before composition, MeVe significantly improves context efficiency, achieving a 57% reduction on the Wikipedia dataset and a 75% reduction on the more complex HotpotQA dataset compared to standard RAG implementations [25]. This work provides a framework for more scalable and reliable LLM applications. By refining and distilling contextual information, MeVe offers a path toward better grounding and more accurate factual support [16]. 

---
# Speaking at the Right Level: Literacy-Controlled Counterspeech Generation with RAG-RL 

**Authors**: Xiaoying Song, Anirban Saha Anik, Dibakar Barua, Pengcheng Luo, Junhua Ding, Lingzi Hong  

**Link**: [PDF](https://arxiv.org/pdf/2509.01058)  

**Abstract**: Health misinformation spreading online poses a significant threat to public health. Researchers have explored methods for automatically generating counterspeech to health misinformation as a mitigation strategy. Existing approaches often produce uniform responses, ignoring that the health literacy level of the audience could affect the accessibility and effectiveness of counterspeech. We propose a Controlled-Literacy framework using retrieval-augmented generation (RAG) with reinforcement learning (RL) to generate tailored counterspeech adapted to different health literacy levels. In particular, we retrieve knowledge aligned with specific health literacy levels, enabling accessible and factual information to support generation. We design a reward function incorporating subjective user preferences and objective readability-based rewards to optimize counterspeech to the target health literacy level. Experiment results show that Controlled-Literacy outperforms baselines by generating more accessible and user-preferred counterspeech. This research contributes to more equitable and impactful public health communication by improving the accessibility and comprehension of counterspeech to health misinformation. 

---
# REFRAG: Rethinking RAG based Decoding 

**Authors**: Xiaoqiang Lin, Aritra Ghosh, Bryan Kian Hsiang Low, Anshumali Shrivastava, Vijai Mohan  

**Link**: [PDF](https://arxiv.org/pdf/2509.01092)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in leveraging extensive external knowledge to enhance responses in multi-turn and agentic applications, such as retrieval-augmented generation (RAG). However, processing long-context inputs introduces significant system latency and demands substantial memory for the key-value cache, resulting in reduced throughput and a fundamental trade-off between knowledge enrichment and system efficiency. While minimizing latency for long-context inputs is a primary objective for LLMs, we contend that RAG require specialized consideration. In RAG, much of the LLM context consists of concatenated passages from retrieval, with only a small subset directly relevant to the query. These passages often exhibit low semantic similarity due to diversity or deduplication during re-ranking, leading to block-diagonal attention patterns that differ from those in standard LLM generation tasks. Based on this observation, we argue that most computations over the RAG context during decoding are unnecessary and can be eliminated with minimal impact on performance. To this end, we propose REFRAG, an efficient decoding framework that compresses, senses, and expands to improve latency in RAG applications. By exploiting the sparsity structure, we demonstrate a 30.85 the time-to-first-token acceleration (3.75 improvement to previous work) without loss in perplexity. In addition, our optimization framework for large context enables REFRAG to extend the context size of LLMs by 16. We provide rigorous validation of REFRAG across diverse long-context tasks, including RAG, multi-turn conversations, and long document summarization, spanning a wide range of datasets. Experimental results confirm that REFRAG delivers substantial speedup with no loss in accuracy compared to LLaMA models and other state-of-the-art baselines across various context sizes. 

---
# EviNote-RAG: Enhancing RAG Models via Answer-Supportive Evidence Notes 

**Authors**: Yuqin Dai, Guoqing Wang, Yuan Wang, Kairan Dou, Kaichen Zhou, Zhanwei Zhang, Shuo Yang, Fei Tang, Jun Yin, Pengyu Zeng, Zhenzhe Ying, Can Yi, Changhua Meng, Yuchen Zhou, Yongliang Shen, Shuai Lu  

**Link**: [PDF](https://arxiv.org/pdf/2509.00877)  

**Abstract**: Large Language Models (LLMs) empowered with retrieval mechanisms have achieved strong progress in open-domain question answering (QA). Yet, the conventional retrieve--then--answer paradigm often suffers from two key limitations: (1) low signal-to-noise ratio in retrieved evidence, where useful information is buried under irrelevant content, and (2) error accumulation in multi-hop reasoning when incomplete or noisy passages are involved. To address these challenges, we present EviNote-RAG, an agentic RAG framework that introduces a structured retrieve--note--answer pipeline. Instead of directly reasoning over raw retrievals, the model is trained to compose Supportive-Evidence Notes (SENs), concise, human-like notes that preserve only answer-relevant information, highlight uncertainty, and explicitly state when no useful evidence exists. This distillation process is further reinforced by the Evidence Quality Reward (EQR), an entailment-based signal that evaluates whether SENs logically support the final answer. Together, SENs and EQR guide the model toward faithful and robust reasoning, while reducing the impact of noise. Experiments on in-domain and out-of-domain QA benchmarks show that EviNote-RAG consistently outperforms strong baselines in accuracy, generalization, and training stability. In particular, it achieves state-of-the-art results while enhancing robustness and efficiency, yielding relative F1 gains of 20\% on HotpotQA (+0.093), 40\% on Bamboogle (+0.151), and 91\% on 2Wiki (+0.256) via denser rewards and reduced verbosity. 

---
# GOSU: Retrieval-Augmented Generation with Global-Level Optimized Semantic Unit-Centric Framework 

**Authors**: Xuecheng Zou, Ke Liu, Bingbing Wang, Huafei Deng, Li Zhang, Yu Tang  

**Link**: [PDF](https://arxiv.org/pdf/2509.00449)  

**Abstract**: Building upon the standard graph-based Retrieval-Augmented Generation (RAG), the introduction of heterogeneous graphs and hypergraphs aims to enrich retrieval and generation by leveraging the relationships between multiple entities through the concept of semantic units (SUs). But this also raises a key issue: The extraction of high-level SUs limited to local text chunks is prone to ambiguity, complex coupling, and increased retrieval overhead due to the lack of global knowledge or the neglect of fine-grained relationships. To address these issues, we propose GOSU, a semantic unit-centric RAG framework that efficiently performs global disambiguation and utilizes SUs to capture interconnections between different nodes across the global context. In the graph construction phase, GOSU performs global merging on the pre-extracted SUs from local text chunks and guides entity and relationship extraction, reducing the difficulty of coreference resolution while uncovering global semantic objects across text chunks. In the retrieval and generation phase, we introduce hierarchical keyword extraction and semantic unit completion. The former uncovers the fine-grained binary relationships overlooked by the latter, while the latter compensates for the coarse-grained n-ary relationships missing from the former. Evaluation across multiple tasks demonstrates that GOSU outperforms the baseline RAG methods in terms of generation quality. 

---
# Learning to Shop Like Humans: A Review-driven Retrieval-Augmented Recommendation Framework with LLMs 

**Authors**: Kaiwen Wei, Jinpeng Gao, Jiang Zhong, Yuming Yang, Fengmao Lv, Zhenyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.00698)  

**Abstract**: Large language models (LLMs) have shown strong potential in recommendation tasks due to their strengths in language understanding, reasoning and knowledge integration. These capabilities are especially beneficial for review-based recommendation, which relies on semantically rich user-generated texts to reveal fine-grained user preferences and item attributes. However, effectively incorporating reviews into LLM-based recommendation remains challenging due to (1) inefficient to dynamically utilize user reviews under LLMs' constrained context windows, and (2) lacking effective mechanisms to prioritize reviews most relevant to the user's current decision context. To address these challenges, we propose RevBrowse, a review-driven recommendation framework inspired by the "browse-then-decide" decision process commonly observed in online user behavior. RevBrowse integrates user reviews into the LLM-based reranking process to enhance its ability to distinguish between candidate items. To improve the relevance and efficiency of review usage, we introduce PrefRAG, a retrieval-augmented module that disentangles user and item representations into structured forms and adaptively retrieves preference-relevant content conditioned on the target item. Extensive experiments on four Amazon review datasets demonstrate that RevBrowse achieves consistent and significant improvements over strong baselines, highlighting its generalizability and effectiveness in modeling dynamic user preferences. Furthermore, since the retrieval-augmented process is transparent, RevBrowse offers a certain level of interpretability by making visible which reviews influence the final recommendation. 

---
# KG-RAG: Enhancing GUI Agent Decision-Making via Knowledge Graph-Driven Retrieval-Augmented Generation 

**Authors**: Ziyi Guan, Jason Chun Lok Li, Zhijian Hou, Pingping Zhang, Donglai Xu, Yuzhi Zhao, Mengyang Wu, Jinpeng Chen, Thanh-Toan Nguyen, Pengfei Xian, Wenao Ma, Shengchao Qin, Graziano Chesi, Ngai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2509.00366)  

**Abstract**: Despite recent progress, Graphic User Interface (GUI) agents powered by Large Language Models (LLMs) struggle with complex mobile tasks due to limited app-specific knowledge. While UI Transition Graphs (UTGs) offer structured navigation representations, they are underutilized due to poor extraction and inefficient integration. We introduce KG-RAG, a Knowledge Graph-driven Retrieval-Augmented Generation framework that transforms fragmented UTGs into structured vector databases for efficient real-time retrieval. By leveraging an intent-guided LLM search method, KG-RAG generates actionable navigation paths, enhancing agent decision-making. Experiments across diverse mobile apps show that KG-RAG outperforms existing methods, achieving a 75.8% success rate (8.9% improvement over AutoDroid), 84.6% decision accuracy (8.1% improvement), and reducing average task steps from 4.5 to 4.1. Additionally, we present KG-Android-Bench and KG-Harmony-Bench, two benchmarks tailored to the Chinese mobile ecosystem for future research. Finally, KG-RAG transfers to web/desktop (+40% SR on Weibo-web; +20% on QQ Music-desktop), and a UTG cost ablation shows accuracy saturates at ~4h per complex app, enabling practical deployment trade-offs. 

---
# Towards Open-World Retrieval-Augmented Generation on Knowledge Graph: A Multi-Agent Collaboration Framework 

**Authors**: Jiasheng Xu, Mingda Li, Yongqiang Tang, Peijie Wang, Wensheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.01238)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong capabilities in language understanding and reasoning. However, their dependence on static training corpora makes them prone to factual errors and knowledge gaps. Retrieval-Augmented Generation (RAG) addresses this limitation by incorporating external knowledge sources, especially structured Knowledge Graphs (KGs), which provide explicit semantics and efficient retrieval. Existing KG-based RAG approaches, however, generally assume that anchor entities are accessible to initiate graph traversal, which limits their robustness in open world settings where accurate linking between the query and the entity is unreliable. To overcome this limitation, we propose AnchorRAG, a novel multi-agent collaboration framework for open-world RAG without the predefined anchor entities. Specifically, a predictor agent dynamically identifies candidate anchor entities by aligning user query terms with KG nodes and initializes independent retriever agents to conduct parallel multi-hop explorations from each candidate. Then a supervisor agent formulates the iterative retrieval strategy for these retriever agents and synthesizes the resulting knowledge paths to generate the final answer. This multi-agent collaboration framework improves retrieval robustness and mitigates the impact of ambiguous or erroneous anchors. Extensive experiments on four public benchmarks demonstrate that AnchorRAG significantly outperforms existing baselines and establishes new state-of-the-art results on the real-world question answering tasks. 

---
# Error Notebook-Guided, Training-Free Part Retrieval in 3D CAD Assemblies via Vision-Language Models 

**Authors**: Yunqing Liu, Nan Zhang, Zhiming Tan  

**Link**: [PDF](https://arxiv.org/pdf/2509.01350)  

**Abstract**: Effective specification-aware part retrieval within complex CAD assemblies is essential for automated design verification and downstream engineering tasks. However, directly using LLMs/VLMs to this task presents some challenges: the input sequences may exceed model token limits, and even after processing, performance remains unsatisfactory. Moreover, fine-tuning LLMs/VLMs requires significant computational resources, and for many high-performing general-use proprietary models (e.g., GPT or Gemini), fine-tuning access is not available. In this paper, we propose a novel part retrieval framework that requires no extra training, but using Error Notebooks + RAG for refined prompt engineering to help improve the existing general model's retrieval performance. The construction of Error Notebooks consists of two steps: (1) collecting historical erroneous CoTs and their incorrect answers, and (2) connecting these CoTs through reflective corrections until the correct solutions are obtained. As a result, the Error Notebooks serve as a repository of tasks along with their corrected CoTs and final answers. RAG is then employed to retrieve specification-relevant records from the Error Notebooks and incorporate them into the inference process. Another major contribution of our work is a human-in-the-loop CAD dataset, which is used to evaluate our method. In addition, the engineering value of our novel framework lies in its ability to effectively handle 3D models with lengthy, non-natural language metadata. Experiments with proprietary models, including GPT-4o and the Gemini series, show substantial gains, with GPT-4o (Omni) achieving up to a 23.4% absolute accuracy improvement on the human preference dataset. Moreover, ablation studies confirm that CoT reasoning provides benefits especially in challenging cases with higher part counts (>10). 

---
# MODE: Mixture of Document Experts for RAG 

**Authors**: Rahul Anand  

**Link**: [PDF](https://arxiv.org/pdf/2509.00100)  

**Abstract**: Retrieval-Augmented Generation (RAG) often relies on large vector databases and cross-encoders tuned for large-scale corpora, which can be excessive for small, domain-specific collections. We present MODE (Mixture of Document Experts), a lightweight alternative that replaces fine-grained nearest-neighbor search with cluster-and-route retrieval. Documents are embedded, grouped into semantically coherent clusters, and represented by cached centroids. At query time, we route to the top centroid(s) and retrieve context only within those clusters, eliminating external vector-database infrastructure and reranking while keeping latency low. On HotpotQA and SQuAD corpora with 100-500 chunks, MODE matches or exceeds a dense-retrieval baseline in answer quality while reducing end-to-end retrieval time. Ablations show that cluster granularity and multi-cluster routing control the recall/precision trade-off, and that tighter clusters improve downstream accuracy. MODE offers a practical recipe for small and medium corpora where simplicity, speed, and topical focus matter. 

---
# ReCode: Improving LLM-based Code Repair with Fine-Grained Retrieval-Augmented Generation 

**Authors**: Yicong Zhao, Shisong Chen, Jiacheng Zhang, Zhixu Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.02330)  

**Abstract**: Recent advances in large language models (LLMs) have demonstrated impressive capabilities in code-related tasks, such as code generation and automated program repair. Despite their promising performance, most existing approaches for code repair suffer from high training costs or computationally expensive inference. Retrieval-augmented generation (RAG), with its efficient in-context learning paradigm, offers a more scalable alternative. However, conventional retrieval strategies, which are often based on holistic code-text embeddings, fail to capture the structural intricacies of code, resulting in suboptimal retrieval quality. To address the above limitations, we propose ReCode, a fine-grained retrieval-augmented in-context learning framework designed for accurate and efficient code repair. Specifically, ReCode introduces two key innovations: (1) an algorithm-aware retrieval strategy that narrows the search space using preliminary algorithm type predictions; and (2) a modular dual-encoder architecture that separately processes code and textual inputs, enabling fine-grained semantic matching between input and retrieved contexts. Furthermore, we propose RACodeBench, a new benchmark constructed from real-world user-submitted buggy code, which addresses the limitations of synthetic benchmarks and supports realistic evaluation. Experimental results on RACodeBench and competitive programming datasets demonstrate that ReCode achieves higher repair accuracy with significantly reduced inference cost, highlighting its practical value for real-world code repair scenarios. 

---
# Multimodal Iterative RAG for Knowledge Visual Question Answering 

**Authors**: Changin Choi, Wonseok Lee, Jungmin Ko, Wonjong Rhee  

**Link**: [PDF](https://arxiv.org/pdf/2509.00798)  

**Abstract**: While Multimodal Large Language Models (MLLMs) have significantly advanced multimodal understanding, their performance remains limited on knowledge-intensive visual questions that require external knowledge beyond the image. Retrieval-Augmented Generation (RAG) has become a promising solution for providing models with external knowledge, its conventional single-pass framework often fails to gather sufficient knowledge. To overcome this limitation, we propose MI-RAG, a Multimodal Iterative RAG framework that leverages reasoning to enhance retrieval and update reasoning over newly retrieved knowledge across modalities. At each iteration, MI-RAG leverages an accumulated reasoning record to dynamically formulate a multi-query. These queries then drive a joint search across heterogeneous knowledge bases containing both visually-grounded and textual knowledge. The newly acquired knowledge is synthesized into the reasoning record, progressively refining understanding across iterations. Experiments on challenging benchmarks, including Encyclopedic VQA, InfoSeek, and OK-VQA, show that MI-RAG significantly improves both retrieval recall and answer accuracy, establishing a scalable approach for compositional reasoning in knowledge-intensive VQA. 

---
# RAG-PRISM: A Personalized, Rapid, and Immersive Skill Mastery Framework with Adaptive Retrieval-Augmented Tutoring 

**Authors**: Gaurangi Raul, Yu-Zheng Lin, Karan Patel, Bono Po-Jen Shih, Matthew W. Redondo, Banafsheh Saber Latibari, Jesus Pacheco, Soheil Salehi, Pratik Satam  

**Link**: [PDF](https://arxiv.org/pdf/2509.00646)  

**Abstract**: The rapid digital transformation of Fourth Industrial Revolution (4IR) systems is reshaping workforce needs, widening skill gaps, especially for older workers. With growing emphasis on STEM skills such as robotics, automation, artificial intelligence (AI), and security, large-scale re-skilling and up-skilling are required. Training programs must address diverse backgrounds, learning styles, and motivations to improve persistence and success, while ensuring rapid, cost-effective workforce development through experiential learning. To meet these challenges, we present an adaptive tutoring framework that combines generative AI with Retrieval-Augmented Generation (RAG) to deliver personalized training. The framework leverages document hit rate and Mean Reciprocal Rank (MRR) to optimize content for each learner, and is benchmarked against human-generated training for alignment and relevance. We demonstrate the framework in 4IR cybersecurity learning by creating a synthetic QA dataset emulating trainee behavior, while RAG is tuned on curated cybersecurity materials. Evaluation compares its generated training with manually curated queries representing realistic student interactions. Responses are produced using large language models (LLMs) including GPT-3.5 and GPT-4, assessed for faithfulness and content alignment. GPT-4 achieves the best performance with 87% relevancy and 100% alignment. Results show this dual-mode approach enables the adaptive tutor to act as both a personalized topic recommender and content generator, offering a scalable solution for rapid, tailored learning in 4IR education and workforce development. 

---
