# L-RAG: Balancing Context and Retrieval with Entropy-Based Lazy Loading 

**Authors**: Sergii Voloshyn  

**Link**: [PDF](https://arxiv.org/pdf/2601.06551)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as the predominant paradigm for grounding Large Language Model outputs in factual knowledge, effectively mitigating hallucinations. However, conventional RAG systems operate under a "retrieve-always" assumption, querying vector databases for every input regardless of query complexity. This static approach incurs substantial computational overhead and inference latency, particularly problematic for high-throughput production deployments. We introduce L-RAG (Lazy Retrieval-Augmented Generation), an adaptive framework that implements hierarchical context management through entropy-based gating. L-RAG employs a two-tier architecture: queries are first processed with a compact document summary, and expensive chunk retrieval is triggered only when the model's predictive entropy exceeds a calibrated threshold, signaling genuine uncertainty. Through experiments on SQuAD 2.0 (N=500) using the Phi-2 model, we demonstrate that L-RAG provides a tunable accuracy-efficiency trade-off: at a conservative threshold (tau=0.5), L-RAG achieves 78.2% accuracy, matching Standard RAG (77.8%), with 8% retrieval reduction; at a balanced threshold (tau=1.0), retrieval reduction increases to 26% with modest accuracy trade-off (76.0%). Latency analysis shows that L-RAG saves 80-210ms per query when retrieval latency exceeds 500ms. Analysis of entropy distributions reveals statistically significant separation (p < 0.001) between correct predictions (H=1.72) and errors (H=2.20), validating entropy as a reliable uncertainty signal. L-RAG offers a practical, training-free approach toward more efficient RAG deployment, providing system architects with a configurable knob to balance accuracy and throughput requirements. 

---
# From RAG to Agentic RAG for Faithful Islamic Question Answering 

**Authors**: Gagan Bhatia, Hamdy Mubarak, Mustafa Jarrar, George Mikros, Fadi Zaraket, Mahmoud Alhirthani, Mutaz Al-Khatib, Logan Cochrane, Kareem Darwish, Rashid Yahiaoui, Firoj Alam  

**Link**: [PDF](https://arxiv.org/pdf/2601.07528)  

**Abstract**: LLMs are increasingly used for Islamic question answering, where ungrounded responses may carry serious religious consequences. Yet standard MCQ/MRC-style evaluations do not capture key real-world failure modes, notably free-form hallucinations and whether models appropriately abstain when evidence is lacking. To shed a light on this aspect we introduce ISLAMICFAITHQA, a 3,810-item bilingual (Arabic/English) generative benchmark with atomic single-gold answers, which enables direct measurement of hallucination and abstention. We additionally developed an end-to-end grounded Islamic modelling suite consisting of (i) 25K Arabic text-grounded SFT reasoning pairs, (ii) 5K bilingual preference samples for reward-guided alignment, and (iii) a verse-level Qur'an retrieval corpus of $\sim$6k atomic verses (ayat). Building on these resources, we develop an agentic Quran-grounding framework (agentic RAG) that uses structured tool calls for iterative evidence seeking and answer revision. Experiments across Arabic-centric and multilingual LLMs show that retrieval improves correctness and that agentic RAG yields the largest gains beyond standard RAG, achieving state-of-the-art performance and stronger Arabic-English robustness even with a small model (i.e., Qwen3 4B). We will make the experimental resources and datasets publicly available for the community. 

---
# BayesRAG: Probabilistic Mutual Evidence Corroboration for Multimodal Retrieval-Augmented Generation 

**Authors**: Xuan Li, Yining Wang, Haocai Luo, Shengping Liu, Jerry Liang, Ying Fu, Weihuang, Jun Yu, Junnan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2601.07329)  

**Abstract**: Retrieval-Augmented Generation (RAG) has become a pivotal paradigm for Large Language Models (LLMs), yet current approaches struggle with visually rich documents by treating text and images as isolated retrieval targets. Existing methods relying solely on cosine similarity often fail to capture the semantic reinforcement provided by cross-modal alignment and layout-induced coherence. To address these limitations, we propose BayesRAG, a novel multimodal retrieval framework grounded in Bayesian inference and Dempster-Shafer evidence theory. Unlike traditional approaches that rank candidates strictly by similarity, BayesRAG models the intrinsic consistency of retrieved candidates across modalities as probabilistic evidence to refine retrieval confidence. Specifically, our method computes the posterior association probability for combinations of multimodal retrieval results, prioritizing text-image pairs that mutually corroborate each other in terms of both semantics and layout. Extensive experiments demonstrate that BayesRAG significantly outperforms state-of-the-art (SOTA) methods on challenging multimodal benchmarks. This study establishes a new paradigm for multimodal retrieval fusion that effectively resolves the isolation of heterogeneous modalities through an evidence fusion mechanism and enhances the robustness of retrieval outcomes. Our code is available at this https URL. 

---
# Relink: Constructing Query-Driven Evidence Graph On-the-Fly for GraphRAG 

**Authors**: Manzong Huang, Chenyang Bu, Yi He, Xingrui Zhuo, Xindong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.07192)  

**Abstract**: Graph-based Retrieval-Augmented Generation (GraphRAG) mitigates hallucinations in Large Language Models (LLMs) by grounding them in structured knowledge. However, current GraphRAG methods are constrained by a prevailing \textit{build-then-reason} paradigm, which relies on a static, pre-constructed Knowledge Graph (KG). This paradigm faces two critical challenges. First, the KG's inherent incompleteness often breaks reasoning paths. Second, the graph's low signal-to-noise ratio introduces distractor facts, presenting query-relevant but misleading knowledge that disrupts the reasoning process.
To address these challenges, we argue for a \textit{reason-and-construct} paradigm and propose Relink, a framework that dynamically builds a query-specific evidence graph. To tackle incompleteness, \textbf{Relink} instantiates required facts from a latent relation pool derived from the original text corpus, repairing broken paths on the fly. To handle misleading or distractor facts, Relink employs a unified, query-aware evaluation strategy that jointly considers candidates from both the KG and latent relations, selecting those most useful for answering the query rather than relying on their pre-existence. This empowers Relink to actively discard distractor facts and construct the most faithful and precise evidence path for each query.
Extensive experiments on five Open-Domain Question Answering benchmarks show that Relink achieves significant average improvements of 5.4\% in EM and 5.2\% in F1 over leading GraphRAG baselines, demonstrating the superiority of our proposed framework. 

---
# Fine-Tuning vs. RAG for Multi-Hop Question Answering with Novel Knowledge 

**Authors**: Zhuoyi Yang, Yurun Song, Iftekhar Ahmed, Ian Harris  

**Link**: [PDF](https://arxiv.org/pdf/2601.07054)  

**Abstract**: Multi-hop question answering is widely used to evaluate the reasoning capabilities of large language models (LLMs), as it requires integrating multiple pieces of supporting knowledge to arrive at a correct answer. While prior work has explored different mechanisms for providing knowledge to LLMs, such as finetuning and retrieval-augmented generation (RAG), their relative effectiveness for multi-hop question answering remains insufficiently understood, particularly when the required knowledge is temporally novel.
In this paper, we systematically compare parametric and non-parametric knowledge injection methods for open-domain multi-hop question answering. We evaluate unsupervised fine-tuning (continual pretraining), supervised fine-tuning, and retrieval-augmented generation across three 7B-parameter open-source LLMs. Experiments are conducted on two benchmarks: QASC, a standard multi-hop science question answering dataset, and a newly constructed dataset of over 10,000 multi-hop questions derived from Wikipedia events in 2024, designed to test knowledge beyond the models' pretraining cutoff.
Our results show that unsupervised fine-tuning provides only limited gains over base models, suggesting that continual pretraining alone is insufficient for improving multi-hop reasoning accuracy. In contrast, retrieval-augmented generation yields substantial and consistent improvements, particularly when answering questions that rely on temporally novel information. Supervised fine-tuning achieves the highest overall accuracy across models and datasets. These findings highlight fundamental differences in how knowledge injection mechanisms support multi-hop question answering and underscore the importance of retrieval-based methods when external or compositional knowledge is required. 

---
# MedTutor: A Retrieval-Augmented LLM System for Case-Based Medical Education 

**Authors**: Dongsuk Jang, Ziyao Shangguan, Kyle Tegtmeyer, Anurag Gupta, Jan Czerminski, Sophie Chheang, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2601.06979)  

**Abstract**: The learning process for medical residents presents significant challenges, demanding both the ability to interpret complex case reports and the rapid acquisition of accurate medical knowledge from reliable sources. Residents typically study case reports and engage in discussions with peers and mentors, but finding relevant educational materials and evidence to support their learning from these cases is often time-consuming and challenging. To address this, we introduce MedTutor, a novel system designed to augment resident training by automatically generating evidence-based educational content and multiple-choice questions from clinical case reports. MedTutor leverages a Retrieval-Augmented Generation (RAG) pipeline that takes clinical case reports as input and produces targeted educational materials. The system's architecture features a hybrid retrieval mechanism that synergistically queries a local knowledge base of medical textbooks and academic literature (using PubMed, Semantic Scholar APIs) for the latest related research, ensuring the generated content is both foundationally sound and current. The retrieved evidence is filtered and ordered using a state-of-the-art reranking model and then an LLM generates the final long-form output describing the main educational content regarding the case-report. We conduct a rigorous evaluation of the system. First, three radiologists assessed the quality of outputs, finding them to be of high clinical and educational value. Second, we perform a large scale evaluation using an LLM-as-a Judge to understand if LLMs can be used to evaluate the output of the system. Our analysis using correlation between LLMs outputs and human expert judgments reveals a moderate alignment and highlights the continued necessity of expert oversight. 

---
# TreePS-RAG: Tree-based Process Supervision for Reinforcement Learning in Agentic RAG 

**Authors**: Tianhua Zhang, Kun Li, Junan Li, Yunxiang Li, Hongyin Luo, Xixin Wu, James Glass, Helen Meng  

**Link**: [PDF](https://arxiv.org/pdf/2601.06922)  

**Abstract**: Agentic retrieval-augmented generation (RAG) formulates question answering as a multi-step interaction between reasoning and information retrieval, and has recently been advanced by reinforcement learning (RL) with outcome-based supervision. While effective, relying solely on sparse final rewards limits step-wise credit assignment and provides weak guidance for intermediate reasoning and actions. Recent efforts explore process-level supervision, but typically depend on offline constructed training data, which risks distribution shift, or require costly intermediate annotations. We present TreePS-RAG, an online, tree-based RL framework for agentic RAG that enables step-wise credit assignment while retaining standard outcome-only rewards. Our key insight is to model agentic RAG reasoning as a rollout tree, where each reasoning step naturally maps to a node. This tree structure allows step utility to be estimated via Monte Carlo estimation over its descendant outcomes, yielding fine-grained process advantages without requiring intermediate labels. To make this paradigm practical, we introduce an efficient online tree construction strategy that preserves exploration diversity under a constrained computational budget. With a rollout cost comparable to strong baselines like Search-R1, experiments on seven multi-hop and general QA benchmarks across multiple model scales show that TreePS-RAG consistently and significantly outperforms both outcome-supervised and leading process-supervised RL methods. 

---
# MedRAGChecker: Claim-Level Verification for Biomedical Retrieval-Augmented Generation 

**Authors**: Yuelyu Ji, Min Gu Kwak, Hang Zhang, Xizhi Wu, Chenyu Li, Yanshan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.06519)  

**Abstract**: Biomedical retrieval-augmented generation (RAG) can ground LLM answers in medical literature, yet long-form outputs often contain isolated unsupported or contradictory claims with safety implications.
We introduce MedRAGChecker, a claim-level verification and diagnostic framework for biomedical RAG.
Given a question, retrieved evidence, and a generated answer, MedRAGChecker decomposes the answer into atomic claims and estimates claim support by combining evidence-grounded natural language inference (NLI) with biomedical knowledge-graph (KG) consistency signals.
Aggregating claim decisions yields answer-level diagnostics that help disentangle retrieval and generation failures, including faithfulness, under-evidence, contradiction, and safety-critical error rates.
To enable scalable evaluation, we distill the pipeline into compact biomedical models and use an ensemble verifier with class-specific reliability weighting.
Experiments on four biomedical QA benchmarks show that MedRAGChecker reliably flags unsupported and contradicted claims and reveals distinct risk profiles across generators, particularly on safety-critical biomedical relations. 

---
# Pragya: An AI-Based Semantic Recommendation System for Sanskrit Subhasitas 

**Authors**: Tanisha Raorane, Prasenjit Kole  

**Link**: [PDF](https://arxiv.org/pdf/2601.06607)  

**Abstract**: Sanskrit Subhasitas encapsulate centuries of cultural and philosophical wisdom, yet remain underutilized in the digital age due to linguistic and contextual barriers. In this work, we present Pragya, a retrieval-augmented generation (RAG) framework for semantic recommendation of Subhasitas. We curate a dataset of 200 verses annotated with thematic tags such as motivation, friendship, and compassion. Using sentence embeddings (IndicBERT), the system retrieves top-k verses relevant to user queries. The retrieved results are then passed to a generative model (Mistral LLM) to produce transliterations, translations, and contextual explanations. Experimental evaluation demonstrates that semantic retrieval significantly outperforms keyword matching in precision and relevance, while user studies highlight improved accessibility through generated summaries. To our knowledge, this is the first attempt at integrating retrieval and generation for Sanskrit Subhasitas, bridging cultural heritage with modern applied AI. 

---
# NC-Bench: An LLM Benchmark for Evaluating Conversational Competence 

**Authors**: Robert J. Moore, Sungeun An, Farhan Ahmed, Jay Pankaj Gala  

**Link**: [PDF](https://arxiv.org/pdf/2601.06426)  

**Abstract**: The Natural Conversation Benchmark (NC-Bench) introduce a new approach to evaluating the general conversational competence of large language models (LLMs). Unlike prior benchmarks that focus on the content of model behavior, NC-Bench focuses on the form and structure of natural conversation. Grounded in the IBM Natural Conversation Framework (NCF), NC-Bench comprises three distinct sets. The Basic Conversation Competence set evaluates fundamental sequence management practices, such as answering inquiries, repairing responses, and closing conversational pairs. The RAG set applies the same sequence management patterns as the first set but incorporates retrieval-augmented generation (RAG). The Complex Request set extends the evaluation to complex requests involving more intricate sequence management patterns. Each benchmark tests a model's ability to produce contextually appropriate conversational actions in response to characteristic interaction patterns. Initial evaluations across 6 open-source models and 14 interaction patterns show that models perform well on basic answering tasks, struggle more with repair tasks (especially repeat), have mixed performance on closing sequences, and find complex multi-turn requests most challenging, with Qwen models excelling on the Basic set and Granite models on the RAG set and the Complex Request set. By operationalizing fundamental principles of human conversation, NC-Bench provides a lightweight, extensible, and theory-grounded framework for assessing and improving the conversational abilities of LLMs beyond topical or task-specific benchmarks. 

---
# TeleMem: Building Long-Term and Multimodal Memory for Agentic AI 

**Authors**: Chunliang Chen, Ming Guan, Xiao Lin, Jiaxu Li, Qiyi Wang, Xiangyu Chen, Jixiang Luo, Changzhi Sun, Dell Zhang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.06037)  

**Abstract**: Large language models (LLMs) excel at many NLP tasks but struggle to sustain long-term interactions due to limited attention over extended dialogue histories. Retrieval-augmented generation (RAG) mitigates this issue but lacks reliable mechanisms for updating or refining stored memories, leading to schema-driven hallucinations, inefficient write operations, and minimal support for multimodal this http URL address these challenges, we propose TeleMem, a unified long-term and multimodal memory system that maintains coherent user profiles through narrative dynamic extraction, ensuring that only dialogue-grounded information is preserved. TeleMem further introduces a structured writing pipeline that batches, retrieves, clusters, and consolidates memory entries, substantially improving storage efficiency, reducing token usage, and accelerating memory operations. Additionally, a multimodal memory module combined with ReAct-style reasoning equips the system with a closed-loop observe, think, and act process that enables accurate understanding of complex video content in long-term contexts. Experimental results show that TeleMem surpasses the state-of-the-art Mem0 baseline with 19% higher accuracy, 43% fewer tokens, and a 2.1x speedup on the ZH-4O long-term role-play gaming benchmark. 

---
# Seeing through the Conflict: Transparent Knowledge Conflict Handling in Retrieval-Augmented Generation 

**Authors**: Hua Ye, Siyuan Chen, Ziqi Zhong, Canran Xiao, Haoliang Zhang, Yuhan Wu, Fei Shen  

**Link**: [PDF](https://arxiv.org/pdf/2601.06842)  

**Abstract**: Large language models (LLMs) equipped with retrieval--the Retrieval-Augmented Generation (RAG) paradigm--should combine their parametric knowledge with external evidence, yet in practice they often hallucinate, over-trust noisy snippets, or ignore vital context. We introduce TCR (Transparent Conflict Resolution), a plug-and-play framework that makes this decision process observable and controllable. TCR (i) disentangles semantic match and factual consistency via dual contrastive encoders, (ii) estimates self-answerability to gauge confidence in internal memory, and (iii) feeds the three scalar signals to the generator through a lightweight soft-prompt with SNR-based weighting. Across seven benchmarks TCR improves conflict detection (+5-18 F1), raises knowledge-gap recovery by +21.4 pp and cuts misleading-context overrides by -29.3 pp, while adding only 0.3% parameters. The signals align with human judgements and expose temporal decision patterns. 

---
# Rational Synthesizers or Heuristic Followers? Analyzing LLMs in RAG-based Question-Answering 

**Authors**: Atharv Naphade  

**Link**: [PDF](https://arxiv.org/pdf/2601.06189)  

**Abstract**: Retrieval-Augmented Generation (RAG) is the prevailing paradigm for grounding Large Language Models (LLMs), yet the mechanisms governing how models integrate groups of conflicting retrieved evidence remain opaque. Does an LLM answer a certain way because the evidence is factually strong, because of a prior belief, or merely because it is repeated frequently? To answer this, we introduce GroupQA, a curated dataset of 1,635 controversial questions paired with 15,058 diversely-sourced evidence documents, annotated for stance and qualitative strength. Through controlled experiments, we characterize group-level evidence aggregation dynamics: Paraphrasing an argument can be more persuasive than providing distinct independent support; Models favor evidence presented first rather than last, and Larger models are increasingly resistant to adapt to presented evidence. Additionally, we find that LLM explanations to group-based answers are unfaithful. Together, we show that LLMs behave consistently as vulnerable heuristic followers, with direct implications for improving RAG system design. 

---
# An Intelligent AI glasses System with Multi-Agent Architecture for Real-Time Voice Processing and Task Execution 

**Authors**: Sheng-Kai Chen, Jyh-Horng Wu, Ching-Yao Lin, Yen-Ting Lin  

**Link**: [PDF](https://arxiv.org/pdf/2601.06235)  

**Abstract**: This paper presents an AI glasses system that integrates real-time voice processing, artificial intelligence(AI) agents, and cross-network streaming capabilities. The system employs dual-agent architecture where Agent 01 handles Automatic Speech Recognition (ASR) and Agent 02 manages AI processing through local Large Language Models (LLMs), Model Context Protocol (MCP) tools, and Retrieval-Augmented Generation (RAG). The system supports real-time RTSP streaming for voice and video data transmission, eye tracking data collection, and remote task execution through RabbitMQ messaging. Implementation demonstrates successful voice command processing with multilingual support and cross-platform task execution capabilities. 

---
# Autonomous QA Agent: A Retrieval-Augmented Framework for Reliable Selenium Script Generation 

**Authors**: Dudekula Kasim Vali  

**Link**: [PDF](https://arxiv.org/pdf/2601.06034)  

**Abstract**: Software testing is critical in the software development lifecycle, yet translating requirements into executable test scripts remains manual and error-prone. While Large Language Models (LLMs) can generate code, they often hallucinate non-existent UI elements. We present the Autonomous QA Agent, a Retrieval-Augmented Generation (RAG) system that grounds Selenium script generation in project-specific documentation and HTML structure. By ingesting diverse formats (Markdown, PDF, HTML) into a vector database, our system retrieves relevant context before generation. Evaluation on 20 e-commerce test scenarios shows our RAG approach achieves 100% (20/20) syntax validity and 90% (18/20, 95% CI: [85%, 95%], p < 0.001) execution success, compared to 30% for standard LLM generation. While our evaluation is limited to a single domain, our method significantly reduces hallucinations by grounding generation in actual DOM structure, demonstrating RAG's potential for automated UI testing. 

---
