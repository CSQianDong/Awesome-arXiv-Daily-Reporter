# LLM-Oriented Information Retrieval: A Denoising-First Perspective 

**Authors**: Lu Dai, Liang Sun, Fanpu Cao, Ziyang Rao, Cehao Yang, Hao Liu, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2605.00505)  

**Abstract**: Modern information retrieval (IR) is no longer consumed primarily by humans but increasingly by large language models (LLMs) via retrieval-augmented generation (RAG) and agentic search. Unlike human users, LLMs are constrained by limited attention budgets and are uniquely vulnerable to noise; misleading or irrelevant information is no longer just a nuisance, but a direct cause of hallucinations and reasoning failures. In this perspective paper, we argue that denoising-maximizing usable evidence density and verifiability within a context window-is becoming the primary bottleneck across the full information access pipeline. We conceptualize this paradigm shift through a four-stage framework of IR challenges: from inaccessible to undiscoverable, to misaligned, and finally to unverifiable. Furthermore, we provide a pipeline-organized taxonomy of signal-to-noise optimization techniques, spanning indexing, retrieval, context engineering, verification, and agentic workflow. We also present research works on information denoising in domains that rely heavily on retrieval such as lifelong assistant, coding agent, deep research, and multimodal understanding. 

---
# Hierarchical Abstract Tree for Cross-Document Retrieval-Augmented Generation 

**Authors**: Ziwen Zhao, Menglin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.00529)  

**Abstract**: Retrieval-augmented generation (RAG) enhances large language models with external knowledge, and tree-based RAG organizes documents into hierarchical indexes to support queries at multiple granularities. However, existing Tree-RAG methods designed for single-document retrieval face critical challenges in scaling to cross-document multi-hop questions: (1) poor distribution adaptability, where $k$-means clustering introduces noise due to rigid distribution assumptions; (2) structural isolation, as tree indexes lack explicit cross-document connections; and (3) coarse abstraction, which obscures fine-grained details. To address these limitations, we propose $\Psi$-RAG, a tree-RAG framework with two key components. First, a hierarchical abstract tree index built through an iterative "merging and collapse" process that adapts to data distributions without a priori assumption. Second, a multi-granular retrieval agent that intelligently interacts with the knowledge base with reorganized queries and an agent-powered hybrid retriever. $\Psi$-RAG supports diverse tasks from token-level question answering to document-level summarization. On cross-document multi-hop QA benchmarks, it outperforms RAPTOR by 25.9% and HippoRAG 2 by 7.4% in average F1 score. Code is available at this https URL. 

---
# Structure-Aware Chunking for Tabular Data in Retrieval-Augmented Generation 

**Authors**: Pooja Guttal, Varun Magotra, Vasudeva Mahavishnu, Natasha Chanto, Sidharth Sivaprasad, Manas Gaur  

**Link**: [PDF](https://arxiv.org/pdf/2605.00318)  

**Abstract**: Tabular documents such as CSV and Excel files are widely used in enterprise data pipelines, yet existing chunking strategies for retrieval-augmented generation (RAG) are primarily designed for unstructured text and do not account for tabular structure. We propose a structure-aware tabular chunking (STC) framework that operates on row-level units by constructing a hierarchical Row Tree representation, where each row is encoded as a key-value block. STC performs token-constrained splitting aligned with structural boundaries and applies overlap-free greedy merging to produce dense, non-overlapping chunks. This design preserves semantic relationships between fields within a row while improving token utilization and reducing fragmentation. Across evaluations on the MAUD dataset, STC reduces chunk count by up to 40% and 56% compared to standard recursive and key-value based baselines, respectively, while improving token utilization and processing efficiency. In retrieval benchmarks, STC improves MRR from 0.3576 to 0.5945 in a hybrid setting and increases Recall@1 from 0.366 to 0.754 in BM25-only retrieval. These results demonstrate that preserving structure during chunking improves retrieval performance, highlighting the importance of structure-aware chunking for RAG over tabular data. 

---
# Retrieval-Augmented Reasoning for Chartered Accountancy 

**Authors**: Jatin Gupta, Akhil Sharma, Saransh Singhania, Ali Imam Abidi  

**Link**: [PDF](https://arxiv.org/pdf/2605.00257)  

**Abstract**: The inception of Large Language Models (LLMs) has catalyzed AI adoption in the finance sector, yet their reliability in complex, jurisdiction-specific tasks like Indian Chartered Accountancy (CA) remains limited. The models display difficulty in executing numerical tasks which require multiple steps while also needing advanced knowledge about legal regulations and the method of scaling their operations is not feasible in settings which have limited access to resources. We present CA-ThinkFlow as a parameter-efficient Retrieval-Augmented Generation (RAG) framework which operates with a 14B, 4-bit-quantized reasoning model, 14B-DeepSeek-R1, and a layout-aware Docling extraction system which maintains document structure during extraction. CA-ThinkFlow uses a basic RAG method which automatically adds retrieved information into the prompt, while it depends on the model's built-in Chain-of-Thought (CoT) functions to create context and produce correct answers. The system we developed system operates at performance levels which match large proprietary models when we tested it on the multi-level CA-Ben benchmark, achieving Scholastic Reliability Coefficient (SRC) results which equal 68.75\% of GPT-4o and Claude 3.5 Sonnet. The framework shows high efficiency and strength in handling parameters, but essential reasoning abilities fail to process complex regulatory texts which exist in fields such as Taxation. 

---
# Exploring LLM biases to manipulate AI search overview 

**Authors**: Roman Smirnov  

**Link**: [PDF](https://arxiv.org/pdf/2605.00012)  

**Abstract**: Modern large language models (LLMs) are used in many business applications in general, and specifically in web search systems and applications that generate overviews of search results - LLM Overview systems. Such systems are using an LLM to select most relevant sources from search results and generate an answer to the user's query. It is known from many studies that LLMs have different biases, in LLM Overview application both the source selection and answer generation stages may be affected by the biases of LLMs (here we are focusing mainly on the selection stage). This research is focused on investigating the presence of the biases in LLM Overview systems and on biases exploitation to manipulate LLM Overview results. Here we train a small language model using reinforcement learning to rewrite search snippets to increase their likelihood of being preferred by an LLM Overview. Our experimental setup intentionally restricts the policy to operate only on snippets and limits reward-hacking strategies, reflecting realistic constraints of web search environments. The results prove that LLM Overview systems have biases and that reinforcement learning in most of the cases can optimize snippet's content to manipulate LLM Overview results. We also prove that LLM Overview selections are driven by comparative rather than absolute advantages among candidate sources. In addition, we examine safety aspects of LLM Overview manipulation possibilities and show that context poisoning attacks can lead to inaccurate or harmful results. 

---
# H-RAG at SemEval-2026 Task 8: Hierarchical Parent-Child Retrieval for Multi-Turn RAG Conversations 

**Authors**: Passant Elchafei, Hossam Emam, Mohamed Alansary, Monorama Swain, Markus Schedl  

**Link**: [PDF](https://arxiv.org/pdf/2605.00631)  

**Abstract**: We present H-RAG, our submission to SemEval-2026 Task 8 (MTRAGEval), addressing both Task A (Retrieval) and Task C (Generation with Retrieved Passages). Task A evaluates standalone retrieval quality, while Task C assesses end-to-end retrieval-augmented generation (RAG) in multi-turn conversational settings, requiring both accurate answer generation and faithful grounding in retrieved evidence. Our approach implements a hierarchical parent-child RAG pipeline that separates fine-grained child-level retrieval from parent-level context reconstruction during generation. Documents are segmented into overlapping sentence-based child chunks, while full documents are preserved as parent units to provide coherent context. Retrieval combines hybrid dense-sparse search, tunable weighting, and embedding-based similarity rescoring over child chunks. Retrieved evidence is aggregated at the parent level and supplied to an instruction-tuned language model for response generation. H-RAG achieves an nDCG@5 score of 0.4271 on Task A and a harmonic mean score of 0.3241 on Task C (RB_agg: 0.2488, RL_F: 0.2703, RB_llm: 0.6508), underscoring the importance of retrieval configuration and parent-level aggregation in multi-turn RAG performance. 

---
# Agentic AI for Substance Use Education: Integrating Regulatory and Scientific Knowledge Sources 

**Authors**: Kosar Haghani, Zahra Kolagar, Mohammed Atiquzzaman  

**Link**: [PDF](https://arxiv.org/pdf/2605.00383)  

**Abstract**: The delivery of traditional substance education has remained problematic due to challenges in scalability, personalization, and the currency of information in a rapidly evolving substance use landscape. While artificial intelligence (AI) offers a promising frontier for enhancing educational delivery, its application in providing real-time, authoritative substance use education remains largely underexplored. We built an agentic-based AI web application that combined Drug Enforcement Administration records with peer-reviewed literature in real-time to provide transparent context-sensitive substance use education. The system uses retrieval-augmented generation with a carefully filtered corpus of 102 documents and dynamic PubMed queries. Document storage was semantically chunked and placed in a vector representation in order to be easily retrieved. We conducted an expert evaluation study in which a panel of five subject matter experts generated 30 domain-specific questions, and two independent raters assessed 90 system interactions (30 primary questions plus two contextual follow-ups each) using a five-point Likert scale across four criteria: factual accuracy, citation quality, contextual coherence, and regulatory appropriateness. Mean ratings ranged from 4.18 to 4.35 across the four criteria (overall category range: 4.05-4.52), with substantial inter-rater agreement (Cohen's kappa = 0.78). These findings suggest that agentic AI architectures integrating authoritative regulatory sources with real-time scientific literature represent a promising direction for scalable, accurate, and verifiable health education delivery, warranting further evaluation through longitudinal user studies. 

---
# When RAG Chatbots Expose Their Backend: An Anonymized Case Study of Privacy and Security Risks in Patient-Facing Medical AI 

**Authors**: Alfredo Madrid-García, Miguel Rujas  

**Link**: [PDF](https://arxiv.org/pdf/2605.00796)  

**Abstract**: Background: Patient-facing medical chatbots based on retrieval-augmented generation (RAG) are increasingly promoted to deliver accessible, grounded health information. AI-assisted development lowers the barrier to building them, but they still demand rigorous security, privacy, and governance controls. Objective: To report an anonymized, non-destructive security assessment of a publicly accessible patient-facing medical RAG chatbot and identify governance lessons for safe deployment of generative AI in health. Methods: We used a two-stage strategy. First, Claude Opus 4.6 supported exploratory prompt-based testing and structured vulnerability hypotheses. Second, candidate findings were manually verified using Chrome Developer Tools, inspecting browser-visible network traffic, payloads, API schemas, configuration objects, and stored interaction data. Results: The LLM-assisted phase identified a critical vulnerability: sensitive system and RAG configuration appeared exposed through client-server communication rather than restricted server-side. Manual verification confirmed that ordinary browser inspection allowed collection of the system prompt, model and embedding configuration, retrieval parameters, backend endpoints, API schema, document and chunk metadata, knowledge-base content, and the 1,000 most recent patient-chatbot conversations. The deployment also contradicted its privacy assurances: full conversation records, including health-related queries, were retrievable without authentication. Conclusions: Serious privacy and security failures in patient-facing RAG chatbots can be identified with standard browser tools, without specialist skills or authentication; independent review should be a prerequisite for deployment. Commercial LLMs accelerated this assessment, including under a false developer persona; assistance available to auditors is equally available to adversaries. 

---
# NorBERTo: A ModernBERT Model Trained for Portuguese with 331 Billion Tokens Corpus 

**Authors**: Enzo S. N. Silva, Pablo B. Costa, Raphael C. Vlasman, Rosimeire P. Costa, Henrique L. P. Silva, Lucas F. A. O. Pellicer, Guilherme Rinaldo, Renato A. Almeida, Darian S. R. Rabbani, Cinthya O. Oestreich, Vinicius F. Caridá  

**Link**: [PDF](https://arxiv.org/pdf/2605.00086)  

**Abstract**: High-quality corpora are essential for advancing Natural Language Processing (NLP) in Portuguese. Building on previous encoder-only models such as BERTimbau and Albertina PT-BR, we introduce NorBERTo, a modern encoder based on the ModernBERT architecture, featuring long-context support and efficient attention mechanisms. NorBERTo is trained on Aurora-PT, a newly curated Brazilian Portuguese corpus comprising 331 billion GPT-2 tokens collected from diverse web sources and existing multilingual datasets. We systematically benchmark NorBERTo against Strong baselines on semantic similarity, textual entailment and classification tasks using standardized datasets such as ASSIN 2 and PLUE. On PLUE, NorBERTo-large achieves the best results among the encoder models we evaluated, notably reaching 0.9191 F1 on MRPC and 0.7689 accuracy on RTE. On ASSIN 2, NorBERTo-large attains the highest entailment F1 (~0.904) among all encoders considered, although Albertina-900M and BERTimbau-large still hold an advantage. To the best of our knowledge, Aurora-PT is currently the largest openly available monolingual Portuguese corpus, surpassing previous resources. NorBERTo provides a modern, mid-sized encoder designed for realistic deployment scenarios: it is straight-forward to fine-tune, efficient to serve, and well suited as a backbone for retrieval-augmented generation and other downstream Portuguese NLP systems. 

---
# Token Arena: A Continuous Benchmark Unifying Energy and Cognition in AI Inference 

**Authors**: Yuxuan Gao, Megan Wang, Yi Ling Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.00300)  

**Abstract**: Public inference benchmarks compare AI systems at the model and provider level, but the unit at which deployment decisions are actually made is the endpoint: the (provider, model, stock-keeping-unit) tuple at which a specific quantization, decoding strategy, region, and serving stack is exposed. We introduce TokenArena, a continuous benchmark that measures inference at endpoint granularity along five core axes (output speed, time to first token, workload-blended price, effective context, and quality on the live endpoint) and synthesizes them, together with a modeled energy estimate, into three headline composites: joules per correct answer, dollars per correct answer, and endpoint fidelity (output-distribution similarity to a first-party reference). The framework's novelty is empirical and methodological. Across 78 endpoints serving 12 model families, the same model on different endpoints differs in mean accuracy by up to 12.5 points on math and code, in fingerprint similarity to first party by up to 12 points, in tail latency by an order of magnitude, and in modeled joules per correct answer by a factor of 6.2. We further show that workload-aware blended pricing reorders the leaderboard substantially: 7 of 10 top-ranked endpoints under the chat preset (3:1 input:output) fall out of the top 10 under the retrieval-augmented preset (20:1), and the reasoning preset (1:5) elevates frontier closed models that the chat preset penalizes on price. We release the framework, schema, probe and eval harness, and a v1.0 leaderboard snapshot under CC BY 4.0. TokenArena is a methodology, not a single ranking; we publish full provenance and limitations and welcome external replication. 

---
# To Call or Not to Call: A Framework to Assess and Optimize LLM Tool Calling 

**Authors**: Qinyuan Wu, Soumi Das, Mahsa Amani, Arijit Nag, Seungeon Lee, Krishna P. Gummadi, Abhilasha Ravichander, Muhammad Bilal Zafar  

**Link**: [PDF](https://arxiv.org/pdf/2605.00737)  

**Abstract**: Agentic AI architectures augment LLMs with external tools, unlocking strong capabilities. However, tool use is not always beneficial; some calls may be redundant or even harmful. Effective tool use, therefore, hinges on a core LLM decision: whether to call or not call a tool, when performing a task. This decision is particularly challenging for web search tools, where the benefits of external information depend on the model's internal knowledge and its ability to integrate potentially noisy tool responses. We introduce a principled framework inspired by decision-making theory to evaluate web search tool-use decisions along three key factors: necessity, utility, and affordability. Our analysis combines two complementary lenses: a normative perspective that infers true need and utility from an optimal allocation of tool calls, and a descriptive perspective that infers the model's self-perceived need and utility from their observed behaviors. We find that models' perceived need and utility of tool calls are often misaligned with their true need and utility. Building on this framework, we train lightweight estimators of need and utility based on models' hidden states. Our estimators enable simple controllers that can improve decision quality and lead to stronger task performance than the self-perceived set up across three tasks and six models. 

---
# Persistent Visual Memory: Sustaining Perception for Deep Generation in LVLMs 

**Authors**: Siyuan Huang, Xiaoye Qu, Yafu Li, Tong Zhu, Zefeng He, Muxin Fu, Daizong Liu, Wei-Long Zheng, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2605.00814)  

**Abstract**: While autoregressive Large Vision-Language Models (LVLMs) demonstrate remarkable proficiency in multimodal tasks, they face a "Visual Signal Dilution" phenomenon, where the accumulation of textual history expands the attention partition function, causing visual attention to decay inversely with generated sequence length. To counteract this, we propose Persistent Visual Memory (PVM), a lightweight learnable module designed to ensure sustained, on-demand visual perception. Integrated as a parallel branch alongside the Feed-Forward Network (FFN) in LVLMs, PVM establishes a distance-agnostic retrieval pathway that directly provides visual embeddings for precise visual perception, thereby structurally mitigating the signal suppression inherent to deep generation. Extensive experiments on Qwen3-VL models demonstrate that PVM brings notable improvements with negligible parameter overhead, delivering consistent average accuracy gains across both 4B and 8B scales, particularly in complex reasoning tasks that demand persistent visual perception. Furthermore, in-depth analysis reveals that PVM can resist length-induced signal decay and accelerate internal prediction convergence. 

---
# BlenderRAG: High-Fidelity 3D Object Generation via Retrieval-Augmented Code Synthesis 

**Authors**: Massimo Rondelli, Francesco Pivi, Maurizio Gabbrielli  

**Link**: [PDF](https://arxiv.org/pdf/2605.00632)  

**Abstract**: Automatic generation of executable Blender code from natural language remains challenging, with state-of-the-art LLMs producing frequent syntactic errors and geometrically inconsistent objects. We present BlenderRAG, a retrieval-augmented generation system that operates on a curated multimodal dataset of 500 expert-validated examples (text, code, image) across 50 object categories. By retrieving semantically similar examples during generation, BlenderRAG improves compilation success rates from 40.8% to 70.0% and semantic normalized alignment from 0.41 to 0.77 (CLIP similarity) across four state-of-the-art LLMs, without requiring fine-tuning or specialized hardware, making it immediately accessible for deployment. The dataset and code will be available at this https URL. 

---
# TADI: Tool-Augmented Drilling Intelligence via Agentic LLM Orchestration over Heterogeneous Wellsite Data 

**Authors**: Rong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2605.00060)  

**Abstract**: We present TADI (Tool-Augmented Drilling Intelligence), an agentic AI system that transforms drilling operational data into evidence-based analytical intelligence. Applied to the Equinor Volve Field dataset, TADI integrates 1,759 daily drilling reports, selected WITSML real-time objects, 15,634 production records, formation tops, and perforations into a dual-store architecture: DuckDB for structured queries over 12 tables with 65,447 rows, and ChromaDB for semantic search over 36,709 embedded documents. Twelve domain-specialized tools, orchestrated by a large language model via iterative function calling, support multi-step evidence gathering that cross-references structured drilling measurements with daily report narratives. The system parses all 1,759 DDR XML files with zero errors, handles three incompatible well naming conventions, and is backed by 95 automated tests plus a 130-question stress-question taxonomy spanning six operational categories. We formalize the agent's behavior as a sequential tool-selection problem and propose the Evidence Grounding Score (EGS) as a simple grounding-compliance proxy based on measurements, attributed DDR quotations, and required answer sections. The complete 6,084-line, framework-free implementation is reproducible given the public Volve download and an API key, and the case studies and qualitative ablation analysis suggest that domain-specialized tool design, rather than model scale alone, is the primary driver of analytical quality in technical operations. 

---
# A Survey of Reasoning-Intensive Retrieval: Progress and Challenges 

**Authors**: Yiyang Wei, Tingyu Song, Siyue Zhang, Yilun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.00063)  

**Abstract**: Reasoning-Intensive Retrieval (RIR) targets retrieval settings where relevance is mediated by latent inferential links between a query and supporting evidence, rather than semantic similarity. Motivated by the emergent reasoning abilities of Large Language Models (LLMs), recent work integrates these capabilities into the IR field, spanning the entire pipeline from benchmarks to retrievers and rerankers. Despite this progress, the field lacks a systematic framework to organize current efforts and articulate a clear path forward. To provide a clear roadmap for this rapidly growing yet fragmented area, this survey (1) systematizes existing RIR benchmarks by knowledge domains and modalities, providing a detailed analysis of the current landscape; (2) introduces a structured taxonomy that categorizes methods based on where and how reasoning is integrated into the retrieval pipeline, alongside an analysis of their trade-offs and practical applications; and (3) summarizes challenges and future directions to guide research in this evolving field. 

---
# SiriusHelper: An LLM Agent-Based Operations Assistant for Big Data Platforms 

**Authors**: Yu Shen, Shiyang Liu, Qihang He, Yihang Cheng, Haining Xie, Zhiming He, Huahua Fan, Xianzhi Tan, Teng Ma, Shaoquan Zhang, Danqing Huang, Fan Jiang, Yang Li, Chongqing Zhao, Peng Chen, Jie Jiang, Bin Cui  

**Link**: [PDF](https://arxiv.org/pdf/2605.00043)  

**Abstract**: Big data platforms are widely used in modern enterprises, and an in-production intelligent assistant is increasingly important to help users quickly find actionable guidance and reduce operational burden. While recent LLM+RAG assistants provide a natural interface, they face practical challenges in real deployments: limited scenario coverage across both general consultation and domain-specific troubleshooting workflows, inefficient knowledge access due to inadequate multi-hop retrieval and flat knowledge organization, and high maintenance cost because escalated tickets are unstructured and hard to convert into assistant improvements and reusable SOPs.
In this paper, we present SiriusHelper, a deployed intelligent assistant for big data platforms. SiriusHelper serves as a unified online assistant that automatically identifies user intent and routes queries to the right handling path, including dedicated expert workflows for specialized scenarios (e.g., SQL execution diagnosis). To support complex troubleshooting, SiriusHelper combines a DeepSearch-driven mechanism with a priority-based hierarchical knowledge base to enable multi-hop retrieval without context overload, thus improving answer reliability and latency. To reduce expert overhead, SiriusHelper further introduces automated ticket understanding and SOP distillation: it diagnoses the assistant failure reason (e.g., missing knowledge or wrong routing) and extracts domain-specific SOPs to continuously enrich the knowledge base. Experiments and online deployment on Tencent Big Data platform show that SiriusHelper outperforms representative alternatives and reduces online ticket volume by 20.8\%. 

---
