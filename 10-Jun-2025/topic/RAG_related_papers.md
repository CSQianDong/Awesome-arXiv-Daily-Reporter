# BRIGHT+: Upgrading the BRIGHT Benchmark with MARCUS, a Multi-Agent RAG Clean-Up Suite 

**Authors**: Liyang Chen, Yujun Cai, Jieqiong Dong, Yiwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07116)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems require corpora that are both structurally clean and semantically coherent. BRIGHT is a recent and influential benchmark designed to evaluate complex multi-hop retrieval across diverse, high-reasoning domains. However, its practical effectiveness is limited by common web-crawled artifacts - such as content redundancy and semantic discontinuity - that impair retrieval accuracy and downstream reasoning. Notably, we find that such issues are concentrated in seven StackExchange-derived subdomains, while other domains (e.g., Coding and Theorem-based content) remain relatively clean.
In this study, we present MARCUS, a multi-agent pipeline that leverages large language models (LLMs) to systematically clean and re-chunk BRIGHT into a higher-quality corpus: BRIGHT-Plus. MARCUS applies dedicated agents for structural noise removal and semantic segmentation, preserving answer-bearing spans while improving contextual integrity. Experimental evaluations demonstrate that BRIGHT-Plus yields consistent and significant improvements in both retrieval accuracy and multi-hop reasoning across a diverse set of retrievers. We release both the BRIGHT-Plus corpus and the MARCUS pipeline to support future research on robust, reasoning-centric retrieval. 

---
# SlideCoder: Layout-aware RAG-enhanced Hierarchical Slide Generation from Design 

**Authors**: Wenxin Tang, Jingyu Xiao, Wenxuan Jiang, Xi Xiao, Yuhang Wang, Xuxin Tang, Qing Li, Yuehe Ma, Junliang Liu, Shisong Tang, Michael R. Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07964)  

**Abstract**: Manual slide creation is labor-intensive and requires expert prior knowledge. Existing natural language-based LLM generation methods struggle to capture the visual and structural nuances of slide designs. To address this, we formalize the Reference Image to Slide Generation task and propose Slide2Code, the first benchmark with difficulty-tiered samples based on a novel Slide Complexity Metric. We introduce SlideCoder, a layout-aware, retrieval-augmented framework for generating editable slides from reference images. SlideCoder integrates a Color Gradient-based Segmentation algorithm and a Hierarchical Retrieval-Augmented Generation method to decompose complex tasks and enhance code generation. We also release SlideMaster, a 7B open-source model fine-tuned with improved reverse-engineered data. Experiments show that SlideCoder outperforms state-of-the-art baselines by up to 40.5 points, demonstrating strong performance across layout fidelity, execution accuracy, and visual consistency. Our code is available at this https URL. 

---
# GaRAGe: A Benchmark with Grounding Annotations for RAG Evaluation 

**Authors**: Ionut-Teodor Sorodoc, Leonardo F. R. Ribeiro, Rexhina Blloshmi, Christopher Davis, Adrià de Gispert  

**Link**: [PDF](https://arxiv.org/pdf/2506.07671)  

**Abstract**: We present GaRAGe, a large RAG benchmark with human-curated long-form answers and annotations of each grounding passage, allowing a fine-grained evaluation of whether LLMs can identify relevant grounding when generating RAG answers. Our benchmark contains 2366 questions of diverse complexity, dynamism, and topics, and includes over 35K annotated passages retrieved from both private document sets and the Web, to reflect real-world RAG use cases. This makes it an ideal test bed to evaluate an LLM's ability to identify only the relevant information necessary to compose a response, or provide a deflective response when there is insufficient information. Evaluations of multiple state-of-the-art LLMs on GaRAGe show that the models tend to over-summarise rather than (a) ground their answers strictly on the annotated relevant passages (reaching at most a Relevance-Aware Factuality Score of 60%), or (b) deflect when no relevant grounding is available (reaching at most 31% true positive rate in deflections). The F1 in attribution to relevant sources is at most 58.9%, and we show that performance is particularly reduced when answering time-sensitive questions and when having to draw knowledge from sparser private grounding sources. 

---
# SceneRAG: Scene-level Retrieval-Augmented Generation for Video Understanding 

**Authors**: Nianbo Zeng, Haowen Hou, Fei Richard Yu, Si Shi, Ying Tiffany He  

**Link**: [PDF](https://arxiv.org/pdf/2506.07600)  

**Abstract**: Despite recent advances in retrieval-augmented generation (RAG) for video understanding, effectively understanding long-form video content remains underexplored due to the vast scale and high complexity of video data. Current RAG approaches typically segment videos into fixed-length chunks, which often disrupts the continuity of contextual information and fails to capture authentic scene boundaries. Inspired by the human ability to naturally organize continuous experiences into coherent scenes, we present SceneRAG, a unified framework that leverages large language models to segment videos into narrative-consistent scenes by processing ASR transcripts alongside temporal metadata. SceneRAG further sharpens these initial boundaries through lightweight heuristics and iterative correction. For each scene, the framework fuses information from both visual and textual modalities to extract entity relations and dynamically builds a knowledge graph, enabling robust multi-hop retrieval and generation that account for long-range dependencies. Experiments on the LongerVideos benchmark, featuring over 134 hours of diverse content, confirm that SceneRAG substantially outperforms prior baselines, achieving a win rate of up to 72.5 percent on generation tasks. 

---
# LlamaRec-LKG-RAG: A Single-Pass, Learnable Knowledge Graph-RAG Framework for LLM-Based Ranking 

**Authors**: Vahid Azizi, Fatemeh Koochaki  

**Link**: [PDF](https://arxiv.org/pdf/2506.07449)  

**Abstract**: Recent advances in Large Language Models (LLMs) have driven their adoption in recommender systems through Retrieval-Augmented Generation (RAG) frameworks. However, existing RAG approaches predominantly rely on flat, similarity-based retrieval that fails to leverage the rich relational structure inherent in user-item interactions. We introduce LlamaRec-LKG-RAG, a novel single-pass, end-to-end trainable framework that integrates personalized knowledge graph context into LLM-based recommendation ranking. Our approach extends the LlamaRec architecture by incorporating a lightweight user preference module that dynamically identifies salient relation paths within a heterogeneous knowledge graph constructed from user behavior and item metadata. These personalized subgraphs are seamlessly integrated into prompts for a fine-tuned Llama-2 model, enabling efficient and interpretable recommendations through a unified inference step. Comprehensive experiments on ML-100K and Amazon Beauty datasets demonstrate consistent and significant improvements over LlamaRec across key ranking metrics (MRR, NDCG, Recall). LlamaRec-LKG-RAG demonstrates the critical value of structured reasoning in LLM-based recommendations and establishes a foundation for scalable, knowledge-aware personalization in next-generation recommender systems. Code is available at~\href{this https URL}{repository}. 

---
# Optimizing RAG Pipelines for Arabic: A Systematic Analysis of Core Components 

**Authors**: Jumana Alsubhi, Mohammad D. Alahmadi, Ahmed Alhusayni, Ibrahim Aldailami, Israa Hamdine, Ahmad Shabana, Yazeed Iskandar, Suhayb Khayyat  

**Link**: [PDF](https://arxiv.org/pdf/2506.06339)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a powerful architecture for combining the precision of retrieval systems with the fluency of large language models. While several studies have investigated RAG pipelines for high-resource languages, the optimization of RAG components for Arabic remains underexplored. This study presents a comprehensive empirical evaluation of state-of-the-art RAG components-including chunking strategies, embedding models, rerankers, and language models-across a diverse set of Arabic datasets. Using the RAGAS framework, we systematically compare performance across four core metrics: context precision, context recall, answer faithfulness, and answer relevancy. Our experiments demonstrate that sentence-aware chunking outperforms all other segmentation methods, while BGE-M3 and Multilingual-E5-large emerge as the most effective embedding models. The inclusion of a reranker (bge-reranker-v2-m3) significantly boosts faithfulness in complex datasets, and Aya-8B surpasses StableLM in generation quality. These findings provide critical insights for building high-quality Arabic RAG pipelines and offer practical guidelines for selecting optimal components across different document types. 

---
# How Significant Are the Real Performance Gains? An Unbiased Evaluation Framework for GraphRAG 

**Authors**: Qiming Zeng, Xiao Yan, Hao Luo, Yuhao Lin, Yuxiang Wang, Fangcheng Fu, Bo Du, Quanqing Xu, Jiawei Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06331)  

**Abstract**: By retrieving contexts from knowledge graphs, graph-based retrieval-augmented generation (GraphRAG) enhances large language models (LLMs) to generate quality answers for user questions. Many GraphRAG methods have been proposed and reported inspiring performance in answer quality. However, we observe that the current answer evaluation framework for GraphRAG has two critical flaws, i.e., unrelated questions and evaluation biases, which may lead to biased or even wrong conclusions on performance. To tackle the two flaws, we propose an unbiased evaluation framework that uses graph-text-grounded question generation to produce questions that are more related to the underlying dataset and an unbiased evaluation procedure to eliminate the biases in LLM-based answer assessment. We apply our unbiased framework to evaluate 3 representative GraphRAG methods and find that their performance gains are much more moderate than reported previously. Although our evaluation framework may still have flaws, it calls for scientific evaluations to lay solid foundations for GraphRAG research. 

---
# Vuyko Mistral: Adapting LLMs for Low-Resource Dialectal Translation 

**Authors**: Roman Kyslyi, Yuliia Maksymiuk, Ihor Pysmennyi  

**Link**: [PDF](https://arxiv.org/pdf/2506.07617)  

**Abstract**: In this paper we introduce the first effort to adapt large language models (LLMs) to the Ukrainian dialect (in our case Hutsul), a low-resource and morphologically complex dialect spoken in the Carpathian Highlands. We created a parallel corpus of 9852 dialect-to-standard Ukrainian sentence pairs and a dictionary of 7320 dialectal word mappings. We also addressed data shortage by proposing an advanced Retrieval-Augmented Generation (RAG) pipeline to generate synthetic parallel translation pairs, expanding the corpus with 52142 examples. We have fine-tuned multiple open-source LLMs using LoRA and evaluated them on a standard-to-dialect translation task, also comparing with few-shot GPT-4o translation. In the absence of human annotators, we adopt a multi-metric evaluation strategy combining BLEU, chrF++, TER, and LLM-based judgment (GPT-4o). The results show that even small(7B) finetuned models outperform zero-shot baselines such as GPT-4o across both automatic and LLM-evaluated metrics. All data, models, and code are publicly released at: this https URL 

---
# Reasoning with RAGged events: RAG-Enhanced Event Knowledge Base Construction and reasoning with proof-assistants 

**Authors**: Stergios Chatzikyriakidis  

**Link**: [PDF](https://arxiv.org/pdf/2506.07042)  

**Abstract**: Extracting structured computational representations of historical events from narrative text remains computationally expensive when constructed manually. While RDF/OWL reasoners enable graph-based reasoning, they are limited to fragments of first-order logic, preventing deeper temporal and semantic analysis. This paper addresses both challenges by developing automatic historical event extraction models using multiple LLMs (GPT-4, Claude, Llama 3.2) with three enhancement strategies: pure base generation, knowledge graph enhancement, and Retrieval-Augmented Generation (RAG). We conducted comprehensive evaluations using historical texts from Thucydides. Our findings reveal that enhancement strategies optimize different performance dimensions rather than providing universal improvements. For coverage and historical breadth, base generation achieves optimal performance with Claude and GPT-4 extracting comprehensive events. However, for precision, RAG enhancement improves coordinate accuracy and metadata completeness. Model architecture fundamentally determines enhancement sensitivity: larger models demonstrate robust baseline performance with incremental RAG improvements, while Llama 3.2 shows extreme variance from competitive performance to complete failure. We then developed an automated translation pipeline converting extracted RDF representations into Coq proof assistant specifications, enabling higher-order reasoning beyond RDF capabilities including multi-step causal verification, temporal arithmetic with BC dates, and formal proofs about historical causation. The Coq formalization validates that RAG-discovered event types represent legitimate domain-specific semantic structures rather than ontological violations. 

---
# KG2QA: Knowledge Graph-enhanced Retrieval-Augmented Generation for Communication Standards Question Answering 

**Authors**: Zhongze Luo, Weixuan Wan, Qizhi Zheng, Yanhong Bai, Jingyun Sun, Jian Wang, Dan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07037)  

**Abstract**: There are many types of standards in the field of communication. The traditional consulting model has a long cycle and relies on the knowledge and experience of experts, making it difficult to meet the rapidly developing technological demands. This paper combines the fine-tuning of large language models with the construction of knowledge graphs to implement an intelligent consultation and question-answering system for communication standards. The experimental results show that after LoRA tuning on the constructed dataset of 6,587 questions and answers in the field of communication standards, Qwen2.5-7B-Instruct demonstrates outstanding professional capabilities in the field of communication standards on the test set. BLEU-4 rose from 18.8564 to 66.8993, and evaluation indicators such as ROUGE also increased significantly, outperforming the fine-tuning effect of the comparison model Llama-3-8B-Instruct. Based on the ontology framework containing 6 entity attributes and 10 relation attributes, a knowledge graph of the communication standard domain containing 13,906 entities and 13,524 relations was constructed, showing a relatively good query accuracy rate. The intelligent consultation and question-answering system enables the fine-tuned model on the server side to access the locally constructed knowledge graph and conduct graphical retrieval of key information first, which is conducive to improving the question-answering effect. The evaluation using DeepSeek as the Judge on the test set shows that our RAG framework enables the fine-tuned model to improve the scores at all five angles, with an average score increase of 2.26%. And combined with web services and API interfaces, it has achieved very good results in terms of interaction experience and back-end access, and has very good practical application value. 

---
# Improving LLM-Powered EDA Assistants with RAFT 

**Authors**: Luyao Shi, Michael Kazda, Charles Schmitter, Hemlata Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2506.06500)  

**Abstract**: Electronic design engineers often struggle to efficiently access relevant information for tasks like design verification and technology development. While large language models (LLMs) can enhance productivity as conversational agents, pre-trained open-source LLMs lack domain-specific knowledge for Electronic Design Automation (EDA). In a Retrieval-Augmented Generation (RAG) context, LLMs rely on external context but may still produce inaccurate responses. Retrieval-Augmented Fine-Tuning (RAFT) improves LLM performance, but acquiring labeled question/answer (Q/A) data in EDA is difficult. To address this, we propose using synthetic Q/A datasets to enhance LLMs with RAFT. Our results show that RAFT with synthetic data significantly boosts LLM performance for RAG-based EDA tasks. We also investigate the impact of using real user questions as Retrieval-Augmented Few-Shot (RAFS) examples for synthetic data generation. Additionally, we implement secure access control to ensure sensitive information is only accessible to authorized personnel. Finally, we assess the risk of data leakage and unintended memorization during fine-tuning with synthetic data, providing practical insights. 

---
