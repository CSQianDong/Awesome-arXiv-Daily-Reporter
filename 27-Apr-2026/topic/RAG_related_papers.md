# Bridging the Long-Tail Gap: Robust Retrieval-Augmented Relation Completion via Multi-Stage Paraphrase Infusion 

**Authors**: Fahmida Alam, Mihai Surdeanu, Ellen Riloff  

**Link**: [PDF](https://arxiv.org/pdf/2604.22261)  

**Abstract**: Large language models (LLMs) struggle with relation completion (RC), both with and without retrieval-augmented generation (RAG), particularly when the required information is rare or sparsely represented. To address this, we propose a novel multi-stage paraphrase-guided relation-completion framework, RC-RAG, that systematically incorporates relation paraphrases across multiple stages. In particular, RC-RAG: (a) integrates paraphrases into retrieval to expand lexical coverage of the relation, (b) uses paraphrases to generate relation-aware summaries, and (c) leverages paraphrases during generation to guide reasoning for relation completion. Importantly, our method does not require any model fine-tuning. Experiments with five LLMs on two benchmark datasets show that RC-RAG consistently outperforms several RAG baselines. In long-tail settings, the best-performing LLM augmented with RC-RAG improves by 40.6 Exact Match (EM) points over its standalone performance and surpasses two strong RAG baselines by 16.0 and 13.8 EM points, respectively, while maintaining low computational overhead. 

---
# STEM: Structure-Tracing Evidence Mining for Knowledge Graphs-Driven Retrieval-Augmented Generation 

**Authors**: Peng Yu, En Xu, Bin Chen, Haibiao Chen, Yinfei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.22282)  

**Abstract**: Knowledge Graph-based Question Answering (KGQA) plays a pivotal role in complex reasoning tasks but remains constrained by two persistent challenges: the structural heterogeneity of Knowledge Graphs(KGs) often leads to semantic mismatch during retrieval, while existing reasoning path retrieval methods lack a global structural perspective. To address these issues, we propose Structure-Tracing Evidence Mining (STEM), a novel framework that reframes multi-hop reasoning as a schema-guided graph search task. First, we design a Semantic-to-Structural Projection pipeline that leverages KG structural priors to decompose queries into atomic relational assertions and construct an adaptive query schema graph. Subsequently, we execute globally-aware node anchoring and subgraph retrieval to obtain the final evidence reasoning graph from KG. To more effectively integrate global structural information during the graph construction process, we design a Triple-Dependent GNN (Triple-GNN) to generate a Global Guidance Subgraph (Guidance Graph) that guides the construction. STEM significantly improves both the accuracy and evidence completeness of multi-hop reasoning graph retrieval, and achieves State-of-the-Art performance on multiple multi-hop benchmarks. 

---
# Navigating Large-Scale Document Collections: MuDABench for Multi-Document Analytical QA 

**Authors**: Zhanli Li, Yixuan Cao, Lvzhou Luo, Ping Luo  

**Link**: [PDF](https://arxiv.org/pdf/2604.22239)  

**Abstract**: This paper introduces the task of analytical question answering over large, semi-structured document collections. We present MuDABench, a benchmark for multi-document analytical QA, where questions require extracting and synthesizing information across numerous documents to perform quantitative analysis. Unlike existing multi-document QA benchmarks that typically require information from only a few documents with limited cross-document reasoning, MuDABench demands extensive inter-document analysis and aggregation. Constructed via distant supervision by leveraging document-level metadata and annotated financial databases, MuDABench comprises over 80,000 pages and 332 analytical QA instances. We also propose an evaluation protocol that measures final answer accuracy and uses intermediate-fact coverage as an auxiliary diagnostic signal for the reasoning process. Experiments reveal that standard RAG systems, which treat all documents as a flat retrieval pool, perform poorly. To address these limitations, we propose a multi-agent workflow that orchestrates planning, extraction, and code generation modules. While this approach substantially improves both process and outcome metrics, a significant gap remains compared to human expert performance. Our analysis identifies two primary bottlenecks: single-document information extraction accuracy and insufficient domain-specific knowledge in current systems. MuDABench is available at this https URL. 

---
# BERAG: Bayesian Ensemble Retrieval-Augmented Generation for Knowledge-based Visual Question Answering 

**Authors**: Jinghong Chen, Jingbiao Mei, Guangyu Yang, Bill Byrne  

**Link**: [PDF](https://arxiv.org/pdf/2604.22678)  

**Abstract**: A common approach to question answering with retrieval-augmented generation (RAG) is to concatenate documents into a single context and pass it to a language model to generate an answer. While simple, this strategy can obscure the contribution of individual documents, making attribution difficult and contributing to the ``lost-in-the-middle'' effect, where relevant information in long contexts is overlooked. Concatenation also scales poorly: computational cost grows quadratically with context length, a problem that becomes especially severe when the context includes visual data, as in visual question answering. Attempts to mitigate these issues by limiting context length can further restrict performance by preventing models from benefiting from the improved recall offered by deeper retrieval. We propose Bayesian Ensemble Retrieval-Augmented Generation (BERAG), along with Bayesian Ensemble Fine-Tuning (BEFT), as a RAG framework in which language models are conditioned on individual retrieved documents rather than a single combined context. BERAG treats document posterior probabilities as ensemble weights and updates them token by token using Bayes' rule during generation. This approach enables probabilistic re-ranking, parallel memory usage, and clear attribution of document contribution, making it well-suited for large document collections. We evaluate BERAG and BEFT primarily on knowledge-based visual question answering tasks, where models must reason over long, imperfect retrieval lists. The results show substantial improvements over standard RAG, including strong gains on Document Visual Question Answering and multimodal needle-in-a-haystack benchmarks. We also demonstrate that BERAG mitigates the ``lost-in-the-middle'' effect. The document posterior can be used to detect insufficient grounding and trigger deflection, while document pruning enables faster decoding than standard RAG. 

---
# How Large Language Models Balance Internal Knowledge with User and Document Assertions 

**Authors**: Shuowei Li, Haoxin Li, Wenda Chu, Yi Fang  

**Link**: [PDF](https://arxiv.org/pdf/2604.22193)  

**Abstract**: Large language models (LLMs) often need to balance their internal parametric knowledge with external information, such as user beliefs and content from retrieved documents, in real-world scenarios like RAG or chat-based systems. A model's ability to reliably process these sources is key to system safety. Previous studies on knowledge conflict and sycophancy are limited to a binary conflict paradigm, primarily exploring conflicts between parametric knowledge and either a document or a user, but ignoring the interactive environment where all three sources exist simultaneously. To fill this gap, we propose a three-source interaction framework and systematically evaluate 27 LLMs from 3 families on 2 datasets. Our findings reveal general patterns: most models rely more on document assertions than user assertions, and this preference is reinforced by post-training. Furthermore, our behavioral analysis shows that most models are impressionable, unable to effectively discriminate between helpful and harmful external information. To address this, we demonstrate that fine-tuning on diverse source interaction data can significantly increase a model's discrimination abilities. In short, our work paves the way for developing trustworthy LLMs that can effectively and reliably integrate multiple sources of information. Code is available at this https URL. 

---
# An End-to-End Ukrainian RAG for Local Deployment. Optimized Hybrid Search and Lightweight Generation 

**Authors**: Mykola Trokhymovych, Yana Oliinyk, Nazarii Nyzhnyk  

**Link**: [PDF](https://arxiv.org/pdf/2604.22095)  

**Abstract**: This paper presents a highly efficient Retrieval-Augmented Generation (RAG) system built specifically for Ukrainian document question answering, which achieved 2nd place in the UNLP 2026 Shared Task. Our solution features a custom two-stage search pipeline that retrieves relevant document pages, paired with a specialized Ukrainian language model fine-tuned on synthetic data to generate accurate, grounded answers. Finally, we compress the model for lightweight deployment. Evaluated under strict computational limits, our architecture demonstrates that high-quality, verifiable AI question answering can be achieved locally on resource-constrained hardware without sacrificing accuracy. 

---
# Can QPP Choose the Right Query Variant? Evaluating Query Variant Selection for RAG Pipelines 

**Authors**: Negar Arabzadeh, Andrew Drozdov, Michael Bendersky, Matei Zaharia  

**Link**: [PDF](https://arxiv.org/pdf/2604.22661)  

**Abstract**: Large Language Models (LLMs) have made query reformulation ubiquitous in modern retrieval and Retrieval-Augmented Generation (RAG) pipelines, enabling the generation of multiple semantically equivalent query variants. However, executing the full pipeline for every reformulation is computationally expensive, motivating selective execution: can we identify the best query variant before incurring downstream retrieval and generation costs? We investigate Query Performance Prediction (QPP) as a mechanism for variant selection across ad-hoc retrieval and end-to-end RAG. Unlike traditional QPP, which estimates query difficulty across topics, we study intra-topic discrimination - selecting the optimal reformulation among competing variants of the same information need. Through large-scale experiments on TREC-RAG using both sparse and dense retrievers, we evaluate pre- and post-retrieval predictors under correlation- and decision-based metrics. Our results reveal a systematic divergence between retrieval and generation objectives: variants that maximize ranking metrics such as nDCG often fail to produce the best generated answers, exposing a "utility gap" between retrieval relevance and generation fidelity. Nevertheless, QPP can reliably identify variants that improve end-to-end quality over the original query. Notably, lightweight pre-retrieval predictors frequently match or outperform more expensive post-retrieval methods, offering a latency-efficient approach to robust RAG. 

---
# Lightweight Retrieval-Augmented Generation and Large Language Model-Based Modeling for Scalable Patient-Trial Matching 

**Authors**: Xiaodi Li, Yang Xiao, Munhwan Lee, Konstantinos Leventakos, Young J. Juhn, David Jones, Terence T. Sio, Wei Liu, Maria Vassilaki, Nansu Zong  

**Link**: [PDF](https://arxiv.org/pdf/2604.22061)  

**Abstract**: Patient-trial matching requires reasoning over long, heterogeneous electronic health records (EHRs) and complex eligibility criteria, posing significant challenges for scalability, generalization, and computational efficiency. Existing approaches either rely on full-document processing with large language models (LLMs), which is computationally expensive, or use traditional machine learning methods that struggle to capture unstructured clinical narratives. In this work, we propose a lightweight framework that combines retrieval-augmented generation and large language model-based modeling for scalable patient-trial matching. The framework explicitly separates two key components: retrieval-augmented generation is used to identify clinically relevant segments from long EHRs, reducing input complexity, while large language models are used to encode these selected segments into informative representations. These representations are further refined through dimensionality reduction and modeled using lightweight predictors, enabling efficient and scalable downstream classification. We evaluate the proposed approach on multiple public benchmarks (n2c2, SIGIR, TREC 2021/2022) and a real-world multimodal dataset from Mayo Clinic (MCPMD). Results show that retrieval-based information selection significantly reduces computational burden while preserving clinically meaningful signals. We further demonstrate that frozen LLMs provide strong representations for structured clinical data, whereas fine-tuning is essential for modeling unstructured clinical narratives. Importantly, the proposed lightweight pipeline achieves performance comparable to end-to-end LLM approaches with substantially lower computational cost. 

---
# Evaluating LLM-Based Goal Extraction in Requirements Engineering: Prompting Strategies and Their Limitations 

**Authors**: Anna Arnaudo, Riccardo Coppola, Maurizio Morisio, Flavio Giobergia, Andrea Bioddo, Angelo Bongiorno, Luca Dadone  

**Link**: [PDF](https://arxiv.org/pdf/2604.22207)  

**Abstract**: Due to the textual and repetitive nature of many Requirements Engineering (RE) artefacts, Large Language Models (LLMs) have proven useful to automate their generation and processing. In this paper, we discuss a possible approach for automating the Goal-Oriented Requirements Engineering (GORE) process by extracting functional goals from software documentation through three phases: actor identification, high and low-level goal extraction. To implement these functionalities, we propose a chain of LLMs fed with engineered prompts. We experimented with different variants of in-context learning and measured the similarities between input data and in-context examples to better investigate their impact. Another key element is the generation-critic mechanism, implemented as a feedback loop involving two LLMs. Although the pipeline achieved 61% accuracy in low-level goal identification, the final stage, these results indicate the approach is best suited as a tool to accelerate manual extraction rather than as a full replacement. The feedback-loop mechanism with Zero-shot outperformed stand-alone Few-shot, with an ablation study suggesting that performance slightly degrades without the feedback cycle. However, we reported that the combination of the feedback mechanism with Few-shot does not deliver any advantage, possibly suggesting that the primary performance ceiling is the prompting strategy applied to the 'critic' LLM. Together with the refinement of both the quantity and quality of the Shot examples, future research will integrate Retrieval-Augmented Generation (RAG) and Chain-of-Thought (CoT) prompting to improve accuracy. 

---
# Aligning Dense Retrievers with LLM Utility via DistillationAligning Dense Retrievers with LLM Utility via Distillation 

**Authors**: Rajinder Sandhu, Di Mu, Cheng Chang, Md Shahriar Tasjid, Himanshu Rai, Maksims Volkovs, Ga Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.22722)  

**Abstract**: Dense vector retrieval is the practical backbone of Retrieval- Augmented Generation (RAG), but similarity search can suffer from precision limitations. Conversely, utility-based approaches leveraging LLM re-ranking often achieve superior performance but are computationally prohibitive and prone to noise inherent in perplexity estimation. We propose Utility-Aligned Embeddings (UAE), a framework designed to merge these advantages into a practical, high-performance retrieval method. We formulate retrieval as a distribution matching problem, training a bi-encoder to imitate a utility distribution derived from perplexity reduction using a Utility-Modulated InfoNCE objective. This approach injects graded utility signals directly into the embedding space without requiring test-time LLM inference. On the QASPER benchmark, UAE improves retrieval Recall@1 by 30.59%, MAP by 30.16% and Token F1 by 17.3% over the strong semantic baseline BGE-Base. Crucially, UAE is over 180x faster than the efficient LLM re-ranking methods preserving competitive performance, demonstrating that aligning retrieval with generative utility yields reliable contexts at scale. 

---
