# Negative Data Mining for Contrastive Learning in Dense Retrieval at IKEA.com 

**Authors**: Eva Agapaki, Amritpal Singh Gill  

**Link**: [PDF](https://arxiv.org/pdf/2605.00353)  

**Abstract**: Contrastive learning is a core component of modern retrieval systems, but its effectiveness heavily relies on the quality of negative examples used during training. In this work, we present a systematic approach to improving dense retrieval for IKEA product search through structured negative sampling strategies and scalable LLM-as-a-judge relevance evaluation.
Building on IKEA Search Engine's late-interaction retrieval architectures, we introduce two key contributions: (1) structured negative sampling strategies that leverage product hierarchical taxonomy and product attributes to generate semantically challenging negatives, and (2) a comprehensive LLM-based evaluation methodology for generating training data. Rather than relying on sparse human annotations or random sampling, our LLM-based evaluation system allocates a score for all candidate products against each query.
Our methodology achieves +2.6\% average category accuracy on offline real user query experiments on the Canada market. However, our A/B test on long-tail queries showed no statistically significant differences in user engagement metrics between the improved and baseline models ($p > 0.05$). We trace this gap to user search behavior: 67\% of popular searches exhibit zero-click rates above 50\%, indicating that a substantial proportion of search sessions result in no product engagement regardless of result ranking. These findings underscore the importance of hard negative mining but also the need for grounding training data and offline evals in real user search behavior -- including query intent distribution and zero-click patterns -- to bridge the gap between offline retrieval quality and online user engagement. 

---
# DynamicPO: Dynamic Preference Optimization for Recommendation 

**Authors**: Xingyu Hu, Kai Zhang, Jiancan Wu, Shuli Wang, Chi Wang, Wenshuai Chen, Yinhua Zhu, Haitao Wang, Xingxing Wang, Xiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.00327)  

**Abstract**: In large language model (LLM)-based recommendation systems, direct preference optimization (DPO) effectively aligns recommendations with user preferences, requiring multi-negative objective functions to leverage abundant implicit-feedback negatives and sharpen preference boundaries. However, our empirical analyses reveal a counterintuitive phenomenon, preference optimization collapse, where increasing the number of negative samples can lead to performance degradation despite a continuously decreasing training loss. We further theoretically demonstrate that this collapse arises from gradient suppression, caused by the dominance of easily discriminable negatives over boundary-critical negatives that truly define user preference boundaries. As a result, boundary-relevant signals are under-optimized, weakening the model's decision boundary. Motivated by these observations, we propose DynamicPO (Dynamic Preference Optimization), a lightweight and plug-and-play framework comprising two adaptive mechanisms: Dynamic Boundary Negative Selection, which identifies and prioritizes informative negatives near the model's decision boundary, and Dual-Margin Dynamic beta Adjustment, which calibrates optimization strength per sample according to boundary ambiguity. Extensive experiments on three public datasets show that DynamicPO effectively prevents optimization collapse and improves recommendation accuracy on multi-negative preference optimization methods, with negligible computational overhead. Our code and datasets are available at this https URL. 

---
# A Survey of Reasoning-Intensive Retrieval: Progress and Challenges 

**Authors**: Yiyang Wei, Tingyu Song, Siyue Zhang, Yilun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.00063)  

**Abstract**: Reasoning-Intensive Retrieval (RIR) targets retrieval settings where relevance is mediated by latent inferential links between a query and supporting evidence, rather than semantic similarity. Motivated by the emergent reasoning abilities of Large Language Models (LLMs), recent work integrates these capabilities into the IR field, spanning the entire pipeline from benchmarks to retrievers and rerankers. Despite this progress, the field lacks a systematic framework to organize current efforts and articulate a clear path forward. To provide a clear roadmap for this rapidly growing yet fragmented area, this survey (1) systematizes existing RIR benchmarks by knowledge domains and modalities, providing a detailed analysis of the current landscape; (2) introduces a structured taxonomy that categorizes methods based on where and how reasoning is integrated into the retrieval pipeline, alongside an analysis of their trade-offs and practical applications; and (3) summarizes challenges and future directions to guide research in this evolving field. 

---
# When More Reformulations Hurt: Avoiding Drift using Ranker Feedback 

**Authors**: V Venktesh, Mandeep Rathee, Avishek Anand  

**Link**: [PDF](https://arxiv.org/pdf/2605.00560)  

**Abstract**: Modern retrieval pipelines increasingly rely on query reformulation and neural reranking to improve effectiveness, but this comes at a significant computational cost and introduces a fundamental tradeoff between recall and query drift. Generating many reformulated queries can substantially increase recall, yet naively merging or exhaustively reranking their results is prohibitively expensive. In this work, we argue that the core challenge is not reformulation generation itself, but the adaptive selection of reformulations and their retrieved documents under a strict inference budget. We propose ReformIR, a budget-aware retrieval framework that treats query reformulations as first-class features and performs online relevance estimation using a strong reranker as a teacher. Given multiple reformulated queries, ReformIR constructs a large candidate pool and learns a lightweight surrogate model that estimates document utility from reformulation-specific retrieval signals. Under a fixed reranking budget, the surrogate adaptively prioritizes both reformulations and documents, selectively querying a teacher reranker anchored to the original query. This process increases recall while actively suppressing drift through online feature selection over reformulations. We conduct extensive experiments on the MSMARCO passage corpora and TREC Deep Learning benchmarks (DL19-DL22). Our results show that ReformIR consistently outperforms existing reformulation strategies, particularly as the number of reformulations increases, where prior methods suffer from severe quality degradation due to drift. Our findings also suggest a shift in retrieval system design, rather than using large language models as rerankers, their capacity is more effectively leveraged in the reformulation stage with feedback-driven optimization. 

---
# LLM-Oriented Information Retrieval: A Denoising-First Perspective 

**Authors**: Lu Dai, Liang Sun, Fanpu Cao, Ziyang Rao, Cehao Yang, Hao Liu, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2605.00505)  

**Abstract**: Modern information retrieval (IR) is no longer consumed primarily by humans but increasingly by large language models (LLMs) via retrieval-augmented generation (RAG) and agentic search. Unlike human users, LLMs are constrained by limited attention budgets and are uniquely vulnerable to noise; misleading or irrelevant information is no longer just a nuisance, but a direct cause of hallucinations and reasoning failures. In this perspective paper, we argue that denoising-maximizing usable evidence density and verifiability within a context window-is becoming the primary bottleneck across the full information access pipeline. We conceptualize this paradigm shift through a four-stage framework of IR challenges: from inaccessible to undiscoverable, to misaligned, and finally to unverifiable. Furthermore, we provide a pipeline-organized taxonomy of signal-to-noise optimization techniques, spanning indexing, retrieval, context engineering, verification, and agentic workflow. We also present research works on information denoising in domains that rely heavily on retrieval such as lifelong assistant, coding agent, deep research, and multimodal understanding. 

---
# FollowTable: A Benchmark for Instruction-Following Table Retrieval 

**Authors**: Rihui Jin, Yuchen Lu, Ting Zhang, Jun Wang, Kuicai Dong, Zhaocheng Du, Dongping Liu, Gang Wang, Yong Liu, Guilin Qi  

**Link**: [PDF](https://arxiv.org/pdf/2605.00400)  

**Abstract**: Table Retrieval (TR) has traditionally been formulated as an ad-hoc retrieval problem, where relevance is primarily determined by topical semantic similarity. With the growing adoption of LLM-based agentic systems, access to structured data is increasingly instruction-driven, where relevance is conditional on explicit content and schema constraints rather than topical similarity alone. We therefore formalize Instruction-Following Table Retrieval (IFTR), a new task that requires models to jointly satisfy topical relevance and fine-grained instruction constraints. We identify two core challenges in IFTR: (i) sensitivity to content scope, such as inclusion and exclusion constraints, and (ii) awareness of schema-grounded requirements, including column semantics and representation granularity--capabilities largely absent in existing retrievers. To support systematic evaluation, we introduce FollowTable, the first large-scale benchmark for IFTR, constructed via a taxonomy-driven annotation pipeline. We further propose a new metric, termed the Instruction Responsiveness Score, to evaluate whether retrieval rankings consistently adapt to user instructions relative to a topic-only baseline. Our results indicate that existing retrieval models struggle to follow fine-grained instructions over tabular data. In particular, they exhibit systematic biases toward surface-level semantic cues and remain limited in handling schema-grounded constraints, highlighting substantial room for future improvements. 

---
# H-RAG at SemEval-2026 Task 8: Hierarchical Parent-Child Retrieval for Multi-Turn RAG Conversations 

**Authors**: Passant Elchafei, Hossam Emam, Mohamed Alansary, Monorama Swain, Markus Schedl  

**Link**: [PDF](https://arxiv.org/pdf/2605.00631)  

**Abstract**: We present H-RAG, our submission to SemEval-2026 Task 8 (MTRAGEval), addressing both Task A (Retrieval) and Task C (Generation with Retrieved Passages). Task A evaluates standalone retrieval quality, while Task C assesses end-to-end retrieval-augmented generation (RAG) in multi-turn conversational settings, requiring both accurate answer generation and faithful grounding in retrieved evidence. Our approach implements a hierarchical parent-child RAG pipeline that separates fine-grained child-level retrieval from parent-level context reconstruction during generation. Documents are segmented into overlapping sentence-based child chunks, while full documents are preserved as parent units to provide coherent context. Retrieval combines hybrid dense-sparse search, tunable weighting, and embedding-based similarity rescoring over child chunks. Retrieved evidence is aggregated at the parent level and supplied to an instruction-tuned language model for response generation. H-RAG achieves an nDCG@5 score of 0.4271 on Task A and a harmonic mean score of 0.3241 on Task C (RB_agg: 0.2488, RL_F: 0.2703, RB_llm: 0.6508), underscoring the importance of retrieval configuration and parent-level aggregation in multi-turn RAG performance. 

---
# Exploring LLM biases to manipulate AI search overview 

**Authors**: Roman Smirnov  

**Link**: [PDF](https://arxiv.org/pdf/2605.00012)  

**Abstract**: Modern large language models (LLMs) are used in many business applications in general, and specifically in web search systems and applications that generate overviews of search results - LLM Overview systems. Such systems are using an LLM to select most relevant sources from search results and generate an answer to the user's query. It is known from many studies that LLMs have different biases, in LLM Overview application both the source selection and answer generation stages may be affected by the biases of LLMs (here we are focusing mainly on the selection stage). This research is focused on investigating the presence of the biases in LLM Overview systems and on biases exploitation to manipulate LLM Overview results. Here we train a small language model using reinforcement learning to rewrite search snippets to increase their likelihood of being preferred by an LLM Overview. Our experimental setup intentionally restricts the policy to operate only on snippets and limits reward-hacking strategies, reflecting realistic constraints of web search environments. The results prove that LLM Overview systems have biases and that reinforcement learning in most of the cases can optimize snippet's content to manipulate LLM Overview results. We also prove that LLM Overview selections are driven by comparative rather than absolute advantages among candidate sources. In addition, we examine safety aspects of LLM Overview manipulation possibilities and show that context poisoning attacks can lead to inaccurate or harmful results. 

---
# RSAT: Structured Attribution Makes Small Language Models Faithful Table Reasoners 

**Authors**: Jugal Gajjar, Kamalasankari Subramaniakuppusamy  

**Link**: [PDF](https://arxiv.org/pdf/2605.00199)  

**Abstract**: When a language model answers a table question, users have no way to verify which cells informed which reasoning steps. We introduce RSAT, a method that trains small language models (SLMs, 1-8B) to produce step-by-step reasoning with cell-level citations grounded in table evidence. Phase 1 (SFT) teaches a structured JSON output format from verified reasoning traces. Phase 2 (GRPO) optimizes a composite reward centered on NLI-based faithfulness, alongside citation validity and parsimony. Across six models from two families-Qwen 2.5 (1.5B/3B/7B) and Llama 3 (1B/3B/8B)-RSAT improves faithfulness 3.7$\times$ over SFT alone (0.224$\rightarrow$0.826), with near-perfect citation validity (0.992). Post-hoc attribution collapses below 13% format success, confirming that attribution must be integrated into reasoning, not retrofitted. Ablations show the faithfulness reward is essential: removing it drops faithfulness from 0.97 to 0.03. 

---
# Retrieval-Augmented Reasoning for Chartered Accountancy 

**Authors**: Jatin Gupta, Akhil Sharma, Saransh Singhania, Ali Imam Abidi  

**Link**: [PDF](https://arxiv.org/pdf/2605.00257)  

**Abstract**: The inception of Large Language Models (LLMs) has catalyzed AI adoption in the finance sector, yet their reliability in complex, jurisdiction-specific tasks like Indian Chartered Accountancy (CA) remains limited. The models display difficulty in executing numerical tasks which require multiple steps while also needing advanced knowledge about legal regulations and the method of scaling their operations is not feasible in settings which have limited access to resources. We present CA-ThinkFlow as a parameter-efficient Retrieval-Augmented Generation (RAG) framework which operates with a 14B, 4-bit-quantized reasoning model, 14B-DeepSeek-R1, and a layout-aware Docling extraction system which maintains document structure during extraction. CA-ThinkFlow uses a basic RAG method which automatically adds retrieved information into the prompt, while it depends on the model's built-in Chain-of-Thought (CoT) functions to create context and produce correct answers. The system we developed system operates at performance levels which match large proprietary models when we tested it on the multi-level CA-Ben benchmark, achieving Scholastic Reliability Coefficient (SRC) results which equal 68.75\% of GPT-4o and Claude 3.5 Sonnet. The framework shows high efficiency and strength in handling parameters, but essential reasoning abilities fail to process complex regulatory texts which exist in fields such as Taxation. 

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
# DeGenTWeb: A First Look at LLM-dominant Websites 

**Authors**: Sichang Steven He, Calvin Ardi, Ramesh Govindan, Harsha V. Madhyastha  

**Link**: [PDF](https://arxiv.org/pdf/2605.00087)  

**Abstract**: Many recent news reports have claimed that content generated by large language models (LLMs) is taking over the web. However, these claims are typically not based on a representative sample of the web and the methodology underlying them is often opaque. Moreover, when aiming to minimize the chances of falsely attributing human-authored content to LLMs, we find that detectors of LLM-generated text perform much worse than advertised. Consequently, we lack an understanding of the true prevalence and characteristics of LLM content on the web.
We describe DeGenTWeb which systematically identifies LLM-dominant websites: sites whose content has been generated using LLMs with little human input. We show how to adapt detectors of LLM-generated text for use on web pages, and how to aggregate detection results from multiple pages on a site for accurate site-level categorization. Using DeGenTWeb, we find that LLM-dominant sites are highly prevalent both in data from Common Crawl and in Bing's search results, and that this share is growing over time. We also show that continuing to accurately identify such sites appears challenging given the capabilities of the latest LLMs. 

---
# FinSafetyBench: Evaluating LLM Safety in Real-World Financial Scenarios 

**Authors**: Yutao Hou, Yihan Jiang, Yuhan Xie, Jian Yang, Liwen Zhang, Hailiang Huang, Guanhua Chen, Yun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.00706)  

**Abstract**: Large language models (LLMs) are increasingly applied in financial scenarios. However, they may produce harmful outputs, including facilitating illegal activities or unethical behavior, posing serious compliance risks. To systematically evaluate LLM safety in finance, we propose FinSafetyBench, a bilingual (English-Chinese) red-teaming benchmark designed to test an LLM's refusal of requests that violate financial compliance. Grounded in real-world financial crime cases and ethics standards, the benchmark comprises 14 subcategories spanning financial crimes and ethical violations. Through extensive experiments on general-purpose and finance-specialized LLMs under three representative attack settings, we identify critical vulnerabilities that allow adversarial prompts to bypass compliance safeguards. Further analysis reveals stronger susceptibility in Chinese contexts and highlights the limitations of prompt-level defenses against sophisticated or implicit manipulation strategies. 

---
# When LLMs Stop Following Steps: A Diagnostic Study of Procedural Execution in Language Models 

**Authors**: Sailesh Panda, Pritam Kadasi, Abhishek Upperwal, Mayank Singh  

**Link**: [PDF](https://arxiv.org/pdf/2605.00817)  

**Abstract**: Large language models (LLMs) often achieve strong performance on reasoning benchmarks, but final-answer accuracy alone does not show whether they faithfully execute the procedure specified in a prompt. We study this question through a controlled diagnostic benchmark for procedural execution, where models are given a step-wise arithmetic algorithm and two numeric inputs, and must return the final computed value. The benchmark uses simple arithmetic operations but increases complexity through algorithm length and look-back dependencies over intermediate variables. Across 14 models and 55 datasets, average first-answer accuracy drops from 61% on 5-step procedures to 20% on 95-step procedures. Generation-level analysis shows that failures often involve missing answers, premature answers, self-correction after an initial error, under-executed traces, and hallucinated extra steps. These findings suggest that apparent reasoning ability can mask substantial weaknesses in faithful instruction execution. 

---
# ML-Bench&Guard: Policy-Grounded Multilingual Safety Benchmark and Guardrail for Large Language Models 

**Authors**: Yunhan Zhao, Zhaorun Chen, Xingjun Ma, Yu-Gang Jiang, Bo Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.00689)  

**Abstract**: As Large Language Models (LLMs) are increasingly deployed in cross-linguistic contexts, ensuring safety in diverse regulatory and cultural environments has become a critical challenge. However, existing multilingual benchmarks largely rely on general risk taxonomies and machine translation, which confines guardrail models to these predefined categories and hinders their ability to align with region-specific regulations and cultural nuances. To bridge these gaps, we introduce ML-Bench, a policy-grounded multilingual safety benchmark covering 14 languages. ML-Bench is constructed directly from regional regulations, where risk categories and fine-grained rules derived from jurisdiction-specific legal texts are directly used to guide the generation of multilingual safety data, enabling culturally and legally aligned evaluation across languages. Building on ML-Bench, we develop ML-Guard, a Diffusion Large Language Model (dLLM)-based guardrail model that supports multilingual safety judgment and policy-conditioned compliance assessment. ML-Guard has two variants, one 1.5B lightweight model for fast `safe/unsafe' checking and a more capable 7B model for customized compliance checking with detailed explanations. We conduct extensive experiments against 11 strong guardrail baselines across 6 existing multilingual safety benchmarks and our ML-Bench, and show that ML-Guard consistently outperforms prior methods. We hope that ML-Bench and ML-Guard can help advance the development of regulation-aware and culturally aligned multilingual guardrail systems. 

---
# SC-Taxo: Hierarchical Taxonomy Generation under Semantic Consistency Constraints using Large Language Models 

**Authors**: Shiqiang Cai, Nianhong Niu, Shizhu He, Kang Liu, Jun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.00620)  

**Abstract**: Scientific literature is expanding at an unprecedented pace, making it increasingly challenging to efficiently organize and access domain knowledge. A high-quality scientific taxonomy offers a structured and hierarchical representation of a research field, facilitating literature exploration and topic navigation, as well as enabling downstream applications such as trend analysis, idea generation, and information retrieval. However, existing taxonomy generation approaches often suffer from structural inconsistencies and semantic misalignment across hierarchical levels. Through empirical analysis, we find that these issues largely stem from inadequate modeling of hierarchical semantic consistency. To address this limitation, we propose a semantic-consistent taxonomy generation (SC-Taxo) framework that leverages large language models (LLMs) with hierarchy-aware refinement stages to ensure semantic consistency. Specifically, SC-Taxo introduces a bidirectional heading generation mechanism that jointly performs bottom-up abstraction and top-down semantic constraint, while further capturing peer-level semantic dependencies to enhance horizontal consistency. Experiments on multiple benchmark datasets demonstrate consistent improvements in hierarchy alignment and heading quality, and additional evaluation on Chinese scientific literature validates its robust cross-lingual generalization. 

---
# Beyond Benchmarks: MathArena as an Evaluation Platform for Mathematics with LLMs 

**Authors**: Jasper Dekoninck, Nikola Jovanović, Tim Gehrunger, Kári Rögnvalddson, Ivo Petrov, Chenhao Sun, Martin Vechev  

**Link**: [PDF](https://arxiv.org/pdf/2605.00674)  

**Abstract**: Large language models (LLMs) are becoming increasingly capable mathematical collaborators, but static benchmarks are no longer sufficient for evaluating progress: they are often narrow in scope, quickly saturated, and rarely updated. This makes it hard to compare models reliably and track progress over time. Instead, we need evaluation platforms: continuously maintained systems that run, aggregate, and analyze evaluations across many benchmarks to give a comprehensive picture of model performance within a broad domain. In this work, we build on the original MathArena benchmark by substantially broadening its scope from final-answer olympiad problems to a continuously maintained evaluation platform for mathematical reasoning with LLMs. MathArena now covers a much wider range of tasks, including proof-based competitions, research-level arXiv problems, and formal proof generation in Lean. Additionally, we maintain a clear evaluation protocol for all models and regularly design new benchmarks as model capabilities improve to ensure that MathArena remains challenging. Notably, the strongest model, GPT-5.5, now reaches 98% on the 2026 USA Math Olympiad and 74% on research-level questions, showing that frontier models can now comfortably solve extremely challenging mathematical problems. This highlights the importance of continuously maintained evaluation platforms like MathArena to track the rapid progress of LLMs in mathematical reasoning. 

---
# AGoQ: Activation and Gradient Quantization for Memory-Efficient Distributed Training of LLMs 

**Authors**: Wenxiang Lin, Juntao Huang, Luhan Zhang, Laili Li, Xiang Bao, Mengyang Zhang, Bing Wang, Shaohuai Shi  

**Link**: [PDF](https://arxiv.org/pdf/2605.00539)  

**Abstract**: Quantization is a key method for reducing the GPU memory requirement of training large language models (LLMs). Yet, current approaches are ineffective for 4-bit activations and 8-bit gradients, which would easily cause slow convergence or accuracy loss. To address this, we introduce AGoQ, incorporating two new techniques: 1) a layer-aware activation quantization algorithm that allocates appropriate bit-widths for activations of various layers based on their types and pipeline stages to achieve near 4-bit activation storage, and 2) a gradient quantization algorithm that reduces memory usage and shortens communication time by employing 8-bit gradient storage and precision-preserving 8-bit All-Reduce communication. We conduct extensive experiments using different sizes of LLMs on two GPU clusters (up to 64 GPUs), and the experimental results show that our AGoQ reduces the memory by up to 52\% and achieves up to 1.34$\times$ improvement of training speed compared to state-of-the-art training systems Megatron-LM (w/ or w/o ZeRO), COAT and DeepSpeed with 8B to 32B LLaMA models, while achieving convergence loss on pretraining and comparable accuracy on downstream tasks with LLaMA architectures. 

---
# Learning How and What to Memorize: Cognition-Inspired Two-Stage Optimization for Evolving Memory 

**Authors**: Derong Xu, Shuochen Liu, Pengfei Luo, Pengyue Jia, Yingyi Zhang, Yi Wen, Yimin Deng, Wenlin Zhang, Enhong Chen, Xiangyu Zhao, Tong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.00702)  

**Abstract**: Large language model (LLM) agents require long-term user memory for consistent personalization, but limited context windows hinder tracking evolving preferences over long interactions. Existing memory systems mainly rely on static, hand-crafted update rules; although reinforcement learning (RL)-based agents learn memory updates, sparse outcome rewards provide weak supervision, resulting in unstable long-horizon optimization. Drawing on memory schema theory and the functional division between prefrontal regions and hippocampus regions, we introduce MemCoE, a cognition-inspired two-stage optimization framework that learns how memory should be organized and what information to update. In the first stage, we propose Memory Guideline Induction to optimize a global guideline via contrastive feedback interpreted as textual gradients; in the second stage, Guideline-Aligned Memory Policy Optimization uses the induced guideline to define structured process rewards and performs multi-turn RL to learn a guideline-following memory evolution policy. We evaluate on three personalization memory benchmarks, covering explicit/implicit preference and different sizes and noise, and observe consistent improvements over strong baselines with favorable robustness, transferability, and efficiency. 

---
# Structure Liberates: How Constrained Sensemaking Produces More Novel Research Output 

**Authors**: James Mooney, Zae Myung Kim, Young-Jun Lee, Dongyeop Kang  

**Link**: [PDF](https://arxiv.org/pdf/2605.00557)  

**Abstract**: Scientific discovery is an extended process of ideation--surveying prior work, forming hypotheses, and refining reasoning--yet existing approaches treat this phase as a brief preamble despite its central role in research. We introduce SCISENSE, a sensemaking-grounded framework that operationalizes ideation as a structured sequence of eight cognitive stages (Pirolli \& Card, 2005). We construct SCISENSE-Traj, a 100K-scale dataset of citation-conditioned research trajectories in two modes: Target, where an LLM reconstructs the ideation path leading to a known paper from its cited works, and Infer, where the LLM proposes novel directions from the same citations. We distill these into SCISENSE-LM, a family of sensemaking LLMs spanning 3B to 70B parameters. Contrary to the assumption that looser supervision promotes greater exploration, Target-trained models achieve a 2.0\% improvement in trajectory quality over Infer-trained models while also producing more novel and diverse outputs. This advantage propagates downstream: coding agents conditioned on Target trajectories produce research artifacts with higher executability and quality than those conditioned on Infer trajectories. This suggests that targeted ideation reduces cognitive burden on downstream agents, freeing them to explore more creatively. SCISENSE offers both a practical tool for augmenting LLM-driven research workflows and a principled testbed for studying how planning shapes scientific discovery. 

---
# Impact of Task Phrasing on Presumptions in Large Language Models 

**Authors**: Kenneth J.K. Ong  

**Link**: [PDF](https://arxiv.org/pdf/2605.00436)  

**Abstract**: Concerns with the safety and reliability of applying large-language models (LLMs) in unpredictable real-world applications motivate this study, which examines how task phrasing can lead to presumptions in LLMs, making it difficult for them to adapt when the task deviates from these assumptions. We investigated the impact of these presumptions on the performance of LLMs using the iterated prisoner's dilemma as a case study. Our experiments reveal that LLMs are susceptible to presumptions when making decisions even with reasoning steps. However, when the task phrasing was neutral, the models demonstrated logical reasoning without much presumptions. These findings highlight the importance of proper task phrasing to reduce the risk of presumptions in LLMs. 

---
# Escaping Mode Collapse in LLM Generation via Geometric Regulation 

**Authors**: Xin Du, Kumiko Tanaka-Ishii  

**Link**: [PDF](https://arxiv.org/pdf/2605.00435)  

**Abstract**: Mode collapse is a persistent challenge in generative modeling and appears in autoregressive text generation as behaviors ranging from explicit looping to gradual loss of diversity and premature trajectory convergence. We take a dynamical-systems view and reinterpret mode collapse as reduced state-space accessibility caused by *geometric collapse*: during generation, the model's internal trajectory becomes confined to a low-dimensional region of its representation space. This implies mode collapse is not purely a token-level phenomenon and cannot be reliably solved by symbolic constraints or probability-only decoding heuristics. Guided by this perspective, we propose *Reinforced Mode Regulation* (RMR), a lightweight, online state-space intervention that regulates dominant self-reinforcing directions in the Transformer value cache (implemented as low-rank damping). Across multiple large language models, RMR substantially reduces mode collapse and enables stable, high-quality generation at extremely low entropy rates (down to 0.8 nats/step), whereas standard decoding typically collapses near 2.0 nats/step. 

---
# Surprisal Minimisation over Goal-directed Alternatives Predicts Production Choice in Dialogue 

**Authors**: Tom Utting, Mario Giulianelli, Arabella Sinclair  

**Link**: [PDF](https://arxiv.org/pdf/2605.00506)  

**Abstract**: We model utterance production as probabilistic cost-sensitive choice over contextual alternatives, using information-theoretic notions of cost. We distinguish between goal-directed alternatives that realise a fixed communicative intent and goal-agnostic alternatives defined only by contextual plausibility, allowing us to derive speaker- and listener-oriented interpretations of different cost measures. We present a procedure to generate both types of alternative sets using language models. Analysing production choices in open-ended dialogue under both deterministic and probabilistic cost minimisation, we find that surprisal minimisation relative to goal-directed alternatives provides the strongest predictive account under both analyses. By contrast, uniform information density and length-based costs exhibit weaker and less consistent predictive power across conditions. More broadly, our study suggests that alternative-conditioned optimisation with LM-generated alternatives provides a principled framework for studying speaker and listener pressures in naturalistic language production. 

---
# Beyond Decodability: Reconstructing Language Model Representations with an Encoding Probe 

**Authors**: Gaofei Shen, Martijn Bentum, Tom Lentz, Afra Alishahi, Grzegorz Chrupała  

**Link**: [PDF](https://arxiv.org/pdf/2605.00607)  

**Abstract**: Probing is widely used to study which features can be decoded from language model representations. However, the common decoding probe approach has two limitations that we aim to solve with our new encoding probe approach: contributions of different features to model representations cannot be directly compared, and feature correlations can affect probing results. We present an Encoding Probe that reverses this direction and reconstructs internal representations of models using interpretable features. We evaluate this method on text and speech transformer models, using feature sets spanning acoustics, phonetics, syntax, lexicon, and speaker identity. Our results suggest that speaker-related effects vary strongly across different training objectives and datasets, while syntactic and lexical features contribute independently to reconstruction. These results show that the Encoding Probe provides a complementary perspective on interpreting model representations beyond decodability. 

---
# RadLite: Multi-Task LoRA Fine-Tuning of Small Language Models for CPU-Deployable Radiology AI 

**Authors**: Pankaj Gupta, Kartik Bose  

**Link**: [PDF](https://arxiv.org/pdf/2605.00421)  

**Abstract**: Large language models (LLMs) show promise in radiology but their deployment is limited by computational requirements that preclude use in resource-constrained clinical environments. We investigate whether small language models (SLMs) of 3-4 billion parameters can achieve strong multi-task radiology performance through LoRA fine-tuning, enabling deployment on consumer-grade CPUs. We train Qwen2.5-3B-Instruct and Qwen3-4B on 162K samples spanning 9 radiology tasks - RADS classification across 10 systems, impression generation, temporal comparison, radiology NLI, NER, abnormality detection, N/M staging, and radiology Q&A - compiled from 12 public datasets. Both models are evaluated on up to 500 held-out test samples per task with standardized metrics. Our key findings are: (1) LoRA fine-tuning dramatically improves performance over zero-shot baselines (RADS accuracy +53%, NLI +60%, N-staging +89%); (2) the two models exhibit complementary strengths - Qwen2.5 excels at structured generation tasks while Qwen3 dominates extractive tasks; (3) a task-outed oracle ensemble combining both models achieves the best performance across all tasks; (4) few-shot prompting with fine-tuned models hurts performance, demonstrating that LoRA adaptation is more effective than in-context learning for specialized domains; and (5) models can be quantized to GGUF format (~1.8-2.4GB) for CPU deployment at 4-8 tokens/second on consumer hardware. Our work demonstrates that small, efficiently fine-tuned models - which we collectively call RadLite - can serve as practical multi-task radiology AI assistants deployable entirely on consumer hardware without GPU requirements. 

---
# ControBench: An Interaction-Aware Benchmark for Controversial Discourse Analysis on Social Networks 

**Authors**: Ta Thanh Thuy, Jiaqi Zhu, Xuan Liu, Lin Shang, Reihaneh Rabbany, Guillaume Rabusseau, Lihui Chen, Zheng Yilun, Sitao Luan  

**Link**: [PDF](https://arxiv.org/pdf/2605.00513)  

**Abstract**: Understanding how people argue across ideological divides online is important for studying political polarization, misinformation, and content moderation. Existing datasets capture only part of this problem: some preserve text but ignore interaction structure, some model structure without rich semantics, and others represent conversations without stable user-level ideological identity. We introduce ControBench, a benchmark for controversial discourse analysis that combines heterogeneous social interaction graphs with rich textual semantics. Built from Reddit discussions on three topics, Trump, abortion, and religion, ControBench contains 7,370 users, 1,783 posts, and 26,525 interactions. The graph contains user and post nodes connected by semantically enriched edges; in particular, user-comment-user edges encode both a reply and the parent comment that it responds to, preserving local argumentative context. User labels are derived from self-declared Reddit flairs, providing a scalable proxy for ideological identity without manual annotation. The resulting datasets exhibit low or negative adjusted homophily (Trump: -0.77, Abortion: 0.06, Religion: 0.04), reflecting the cross-cutting structure of real-world debate. We evaluate graph neural networks, pretrained language models, and large language models on ControBench and observe distinct performance patterns across topics and model families, especially when ideological boundaries are ambiguous. These results position ControBench as a challenging and realistic benchmark for controversial discourse analysis. 

---
# ReLay: Personalized LLM-Generated Plain-Language Summaries for Better Understanding, but at What Cost? 

**Authors**: Joey Chan, Yikun Han, Jingyuan Chen, Samuel Fang, Lauren D. Gryboski, Alexandra Lee, Sheel Tanna, Qingqing Zhu, Zhiyong Lu, Lucy Lu Wang, Yue Guo  

**Link**: [PDF](https://arxiv.org/pdf/2605.00468)  

**Abstract**: Plain Language Summaries (PLS) aim to make research accessible to lay readers, but they are typically written in a one-size-fits-all style that ignores differences in readers' information needs and comprehension. In health contexts, this limitation is particularly important because misunderstanding scientific information can affect real-world decisions. Large language models (LLMs) offer new opportunities for personalizing PLS, but it remains unclear whether personalization helps, which strategies are most effective, and how to balance personalization with safety. We introduce ReLay, a dataset of 300 participant--PLS pairs from 50 lay participants in both static (expert-written) and interactive (LLM-personalized) settings. ReLay includes user characteristics, health information needs, information-seeking behavior, comprehension outcomes, interaction logs, and quality ratings. We use ReLay to evaluate five LLMs across two personalization methods. Personalization improves comprehension and perceived quality, but it also raises the risk of reinforcing user biases and introducing hallucinations, revealing a trade-off between personalization and safety. These findings highlight the need for personalization methods that are both effective and trustworthy for diverse lay audiences. 

---
# Agent Capsules: Quality-Gated Granularity Control for Multi-Agent LLM Pipelines 

**Authors**: Aninda Ray  

**Link**: [PDF](https://arxiv.org/pdf/2605.00410)  

**Abstract**: A multi-agent pipeline with N agents typically issues N LLM calls per run. Merging agents into fewer calls (compound execution) promises token savings, but naively merged calls silently degrade quality through tool loss and prompt compression. We present Agent Capsules, an adaptive execution runtime that treats multi-agent pipeline execution as an optimization problem with empirical quality constraints. The runtime instruments coordination overhead per group, scores composition opportunity, selects among three compound execution strategies, and gates every mode switch on rolling-mean output quality. A controlled negative result confirms that injecting more context into a merged call worsens compression rather than relieving it, so the framework's escalation ladder (standard, then two-phase, then sequential) recovers quality by moving toward per-agent dispatch rather than by rewriting merged prompts. On LLM-judged quality, the controller matches a hand-tuned oracle on every measured (model, group, mode) cell: routing compound whenever the oracle would, and reverting to fine whenever quality would fail the floor, without per-model configuration. Against a hand-crafted LangGraph implementation of a 14-agent competitive intelligence pipeline, Agent Capsules uses 51% fewer fine-mode input tokens and 42% fewer compound-mode input tokens, at +0.020 and +0.017 quality respectively. Against a DSPy implementation of a 5-agent due diligence pipeline, the framework uses 19% fewer tokens than uncompiled DSPy at quality parity, and 68% fewer tokens than MIPROv2 at +0.052 quality. Even before compound mode fires, the runtime delivers efficiency through automatic policy resolution, cache-aligned prompts, and topology-aware context injection, matching both hand-tuned and compile-time baselines without training data or per-pipeline engineering. 

---
# From Backward Spreading to Forward Replay: Revisiting Target Construction in LLM Parameter Editing 

**Authors**: Wei Liu, Hongkai Liu, Zhiying Deng, Yee Whye Teh, Wee Sun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.00358)  

**Abstract**: LLM parameter editing methods commonly rely on computing an ideal target hidden-state at a target layer (referred as anchor point) and distributing the target vector to multiple preceding layers (commonly known as backward spreading) for cooperative editing. Although widely used for a long time, its underlying basis have not been systematically investigated. In this paper, we first conduct a systematic study of its foundations, which helps clarify its capability boundaries, practical considerations, and potential failure modes. Then, we propose a simple and elegant alternative that replaces backward spreading with forward-propagation. Instead of optimizing the target at the last editing layer, we optimize the anchor point at the first editing layer, and then propagate it forward to obtain accurate and mutually compatible target hidden-states for all subsequent editing layers. This approach achieves the same computational complexity as existing methods while producing more accurate layer-wise targets. Our method is simple, without interfering with either the computation of the initial target hidden state or any other components of the subsequent editing pipeline, and thus constituting a benefit for a wide range of LLM parameter editing methods. 

---
# Unlearning What Matters: Token-Level Attribution for Precise Language Model Unlearning 

**Authors**: Jiawei Wu, DouDou Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.00364)  

**Abstract**: Machine unlearning has emerged as a critical capability for addressing privacy, safety, and regulatory concerns in large language models (LLMs). Existing methods operate at the sequence level, applying uniform updates across all tokens despite only a subset encoding the knowledge targeted for removal. This introduces gradient noise, degrades utility, and leads to suboptimal forgetting. We propose TokenUnlearn, a token-level attribution framework that identifies and selectively targets critical tokens. Our approach combines knowledge-aware signals via masking, and entropy-aware signals to yield importance scores for precise token selection. We develop two complementary strategies: hard selection, applying unlearning only to high-importance tokens, and soft weighting, modulating gradient contributions based on importance scores. Both extend existing methods to token-level variants. Theoretical analysis shows token-level selection improves gradient signal-to-noise ratio. Experiments on TOFU and WMDP benchmarks across three model architectures demonstrate consistent improvements over sequence-level baselines in both forgetting effectiveness and utility preservation. 

---
# MemRouter: Memory-as-Embedding Routing for Long-Term Conversational Agents 

**Authors**: Tianyu Hu, Weikai Lin, Weizhi Zhang, Jing Ma, Song Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.00356)  

**Abstract**: Long-term conversational agents must decide which turns to store in external memory, yet recent systems rely on autoregressive LLM generation at every turn to make that decision. We present MemRouter, a write-side memory router that decouples memory admission from the downstream answer backbone and replaces per-turn memory-management decoding with an embedding-based routing policy. MemRouter encodes each turn together with recent context, projects the resulting embeddings through a frozen LLM backbone, and predicts whether the turn should be stored using lightweight classification heads while training only 12M parameters. Under a controlled matched-harness comparison on LoCoMo, where the retrieval pipeline, answer prompts, and QA backbone (Qwen2.5-7B) are held identical, MemRouter outperforms an LLM-based memory manager on every question category (overall F1 52.0 vs 45.6, non-overlapping 95% CIs) while reducing memory-management p50 latency from 970ms to 58ms. Descriptive factorial averaging further shows that learned admission improves mean F1 by +10.3 over random storage, category-specific prompting adds +5.2 over a generic prompt, and retrieval contributes +0.7. These results suggest that write-side memory admission can be learned by a small supervised router, while answer generation remains a separate downstream component in long-horizon conversational QA. 

---
# Making Every Verified Token Count: Adaptive Verification for MoE Speculative Decoding 

**Authors**: Lehan Pan, Ziyang Tao, Ruoyu Pang, Xiao Wang, Jianjun Zhao, Yanyong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.00342)  

**Abstract**: Tree-based speculative decoding accelerates autoregressive generation by verifying multiple draft candidates in parallel, but this advantage weakens for sparse Mixture-of-Experts (MoE) models. As the draft tree grows, different branches activate different experts, expanding the union of activated experts and substantially increasing target-side verification cost. We propose EVICT, a training-free, hyperparameter-free, and lossless adaptive verification method for MoE speculative decoding. EVICT makes every verified token count by truncating the draft tree before target verification and retaining only the cost-effective prefix. It leverages fine-grained drafter signals to estimate candidate benefit, combines them with offline-profiled verification cost, and remains highly compatible with the high-performance graph-based serving framework SGLang. Extensive experiments on diverse MoE backbones and benchmarks show that EVICT achieves up to 2.35x speedup over autoregressive decoding and an average 1.21x speedup over the state-of-the-art baseline EAGLE-3, while significantly reducing unnecessary expert activations during verification. 

---
# Agentic AI for Substance Use Education: Integrating Regulatory and Scientific Knowledge Sources 

**Authors**: Kosar Haghani, Zahra Kolagar, Mohammed Atiquzzaman  

**Link**: [PDF](https://arxiv.org/pdf/2605.00383)  

**Abstract**: The delivery of traditional substance education has remained problematic due to challenges in scalability, personalization, and the currency of information in a rapidly evolving substance use landscape. While artificial intelligence (AI) offers a promising frontier for enhancing educational delivery, its application in providing real-time, authoritative substance use education remains largely underexplored. We built an agentic-based AI web application that combined Drug Enforcement Administration records with peer-reviewed literature in real-time to provide transparent context-sensitive substance use education. The system uses retrieval-augmented generation with a carefully filtered corpus of 102 documents and dynamic PubMed queries. Document storage was semantically chunked and placed in a vector representation in order to be easily retrieved. We conducted an expert evaluation study in which a panel of five subject matter experts generated 30 domain-specific questions, and two independent raters assessed 90 system interactions (30 primary questions plus two contextual follow-ups each) using a five-point Likert scale across four criteria: factual accuracy, citation quality, contextual coherence, and regulatory appropriateness. Mean ratings ranged from 4.18 to 4.35 across the four criteria (overall category range: 4.05-4.52), with substantial inter-rater agreement (Cohen's kappa = 0.78). These findings suggest that agentic AI architectures integrating authoritative regulatory sources with real-time scientific literature represent a promising direction for scalable, accurate, and verifiable health education delivery, warranting further evaluation through longitudinal user studies. 

---
# Are You the A-hole? A Fair, Multi-Perspective Ethical Reasoning Framework 

**Authors**: Sheza Munir, Ahanaf Rodoshi, Sumin Lee, Feiran Chang, Xujie Si, Syed Ishtiaque Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2605.00270)  

**Abstract**: Standard methods for aggregating natural language judgments, such as majority voting, often fail to produce logically consistent results when applied to high-conflict domains, treating differing opinions as noise. We propose a neuro-symbolic aggregation framework that formalizes conflict resolution through Weighted Maximum Satisfiability (MaxSAT). Our pipeline utilizes a language model to map unstructured natural language explanations into interpretable logical predicates and confidence weights. These components are then encoded as soft constraints within the Z3 solver, transforming the aggregation problem into an optimization task that seeks the maximum consistency across conflicting testimony. Using the Reddit r/AmItheAsshole forum as a case study in large-scale moral disagreement, our system generates logically coherent verdicts that diverge from popularity-based labels 62% of the time, corroborated by an 86% agreement rate with independent human evaluators. This study demonstrates the efficacy of coupling neural semantic extraction with formal solvers to enforce logical soundness and explainability in the aggregation of noisy human reasoning. 

---
# What Don't You Understand? Using Large Language Models to Identify and Characterize Student Misconceptions About Challenging Topics 

**Authors**: Michael J. Parker, Maria G. Zavala-Cerna  

**Link**: [PDF](https://arxiv.org/pdf/2605.00294)  

**Abstract**: This study presents a systematic approach to identifying and characterizing student misconceptions in online learning environments through a novel combination of quantitative performance analysis and large language model (LLM) assessment. We analyzed data from 9 course periods across 5 online biomedical science courses, encompassing 3,802 medical student enrollments. Using data from 40-50 topic-focused quizzes per course, we developed a two-stage methodology. First, we identified challenging central topics using quiz-level performance metrics. Second, we employed LLMs to characterize the underlying misconceptions in these high-priority areas. By examining student performance on first attempts across primarily multiple-choice questions (MCQs), we identified consistently challenging topics that were also central to course objectives. We then leveraged recent advances in generative AI to analyze three distinct data sources in combination: quiz question content, student response patterns, and lecture transcripts. This approach revealed actionable insights about student misconceptions that were not apparent from performance data alone. The quality of the LLM-identified misconceptions was rated as excellent by subject matter experts. We also conducted teacher interviews to assess the perceived utility of our topic identification method. Faculty found that data-driven identification of challenging topics was valuable and corroborated their own classroom observations. This methodology provides a scalable approach to characterizing student difficulties in learning environments where quizzes are used. Our findings demonstrate the potential for targeted and potentially personalized interventions in future course iterations, with clear pathways for measuring intervention effectiveness through follow-up quiz performance. 

---
# A11y-Compressor: A Framework for Enhancing the Efficiency of GUI Agent Observations through Visual Context Reconstruction and Redundancy Reduction 

**Authors**: Michito Takeshita, Takuro Kawada, Takumi Ohashi, Shunsuke Kitada, Hitoshi Iyatomi  

**Link**: [PDF](https://arxiv.org/pdf/2605.00551)  

**Abstract**: AI agents that interact with graphical user interfaces (GUIs) require effective observation representations for reliable grounding. The accessibility tree is a commonly used text-based format that encodes UI element attributes, but it suffers from redundancy and lacks structural information such as spatial relationships among elements. We propose A11y-Compressor, a framework that transforms linearized accessibility trees into compact and structured representations. Our implementation, Compressed-a11y, applies a lightweight and structured transformation pipeline with modal detection, redundancy reduction, and semantic structuring. Experiments on the OSWorld benchmark show that Compressed-a11y reduces input tokens to 22% of the original while improving task success rates by 5.1 percentage points on average. 

---
# Budget-Aware Routing for Long Clinical Text 

**Authors**: Khizar Qureshi, Geoffrey Martin, Yifan Peng  

**Link**: [PDF](https://arxiv.org/pdf/2605.00336)  

**Abstract**: A key challenge for large language models is token cost per query and overall deployment cost. Clinical inputs are long, heterogeneous, and often redundant, while downstream tasks are short and high stakes. We study budgeted context selection, where a subset of document units is chosen under a strict token budget so an off-the-shelf generator can meet fixed cost and latency constraints. We cast this as a knapsack-constrained subset selection problem with two design choices, unitization that defines document segmentation and selection that determines which units are kept.
We propose \textbf{RCD}, a monotone submodular objective that balances relevance, coverage, and diversity. We compare sentence, section, window, and cluster-based unitization, and introduce a routing heuristic that adapts to the budget regime. Experiments on MIMIC discharge notes, Cochrane abstracts, and L-Eval show that optimal strategies depend on the evaluation setting. Positional heuristics perform best at low budgets in extractive tasks, while diversity-aware methods such as MMR improve LLM generation. Selector choice matters more than unitization, with cluster-based grouping reducing performance and other schemes behaving similarly. ROUGE saturates for LLM summaries, while BERTScore better reflects quality differences. We release our code at this https URL. 

---
# Why Do LLMs Struggle in Strategic Play? Broken Links Between Observations, Beliefs, and Actions 

**Authors**: Jan Sobotka, Mustafa O. Karabag, Ufuk Topcu  

**Link**: [PDF](https://arxiv.org/pdf/2605.00226)  

**Abstract**: Large language models (LLMs) are increasingly tasked with strategic decision-making under incomplete information, such as in negotiation and policymaking. While LLMs can excel at many such tasks, they also fail in ways that are poorly understood. We shed light on these failures by uncovering two fundamental gaps in the internal mechanisms underlying the decision-making of LLMs in incomplete-information games, supported by experiments with open-weight models Llama 3.1, Qwen3, and gpt-oss. First, an observation-belief gap: LLMs encode internal beliefs about latent game states that are substantially more accurate than their own verbal reports, yet these beliefs are brittle. In particular, the belief accuracy degrades with multi-hop reasoning, exhibits primacy and recency biases, and drifts away from Bayesian coherence over extended interactions. Second, a belief-action gap: The implicit conversion of internal beliefs into actions is weaker than that of the beliefs externalized in the prompt, yet neither belief-conditioning consistently achieves higher game payoffs. These results show how analyzing LLMs' internal processes can expose systematic vulnerabilities that warrant caution before deploying LLMs in strategic domains without robust guardrails. 

---
# ViLegalNLI: Natural Language Inference for Vietnamese Legal Texts 

**Authors**: Nhung Thi-Hong Duong, Mai Ngoc Ho, Tin Van Huynh, Kiet Van Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2605.00116)  

**Abstract**: In this article, we introduce ViLegalNLI, the first large-scale Vietnamese Natural Language Inference (NLI) dataset specifically constructed for the legal domain. The dataset consists of 42,012 premise-hypothesis pairs derived from official statutory documents and annotated with binary inference labels (Entailment and Non-entailment). It covers multiple legal domains and reflects realistic legal reasoning scenarios characterized by structured logic, conditional clauses, and domain-specific terminology. To construct ViLegalNLI, we propose a semi-automatic data generation framework that integrates large language models for controlled hypothesis generation and systematic quality validation procedures. The framework incorporates artifact mitigation strategies and cross-model validation to improve annotation reliability and ensure legal consistency. The resulting dataset captures diverse reasoning patterns, including paraphrasing, logical implication, and legally invalid inferences, thereby providing a comprehensive benchmark for Vietnamese legal inference tasks. We conduct extensive experiments on the ViLegalNLI using multilingual models, Vietnamese-specific pretrained language models, and instruction-tuned large language models. The results show that few-shot LLM configurations consistently achieve superior performance, while performance is significantly influenced by hypothesis length, lexical overlap, and reasoning complexity. Cross-domain evaluations further reveal the challenges of generalizing legal inference across distinct legal fields. Overall, ViLegalNLI establishes a foundational benchmark for Vietnamese legal NLI and supports future research in legal reasoning, statutory text understanding, and the development of reliable AI systems for legal analysis and decision support. The dataset is publicly available for research purposes. 

---
# Cultural Benchmarking of LLMs in Standard and Dialectal Arabic Dialogues 

**Authors**: Muhammad Dehan Al Kautsar, Saeed Almheiri, Momina Ahsan, Bilal Elbouardi, Younes Samih, Sarfraz Ahmad, Amr Keleg, Omar El Herraoui, Kareem Elzeky, Abed Alhakim Freihat, Mohamed Anwar, Zhuohan Xie, Junhong Liang, Mohammad Rustom Al Nasar, Preslav Nakov, Fajri Koto  

**Link**: [PDF](https://arxiv.org/pdf/2605.00119)  

**Abstract**: There is a significant gap in evaluating cultural reasoning in LLMs using conversational datasets that capture culturally rich and dialectal contexts. Most Arabic benchmarks focus on short text snippets in Modern Standard Arabic (MSA), overlooking the cultural nuances that naturally arise in dialogues. To address this gap, we introduce ArabCulture-Dialogue, a culturally grounded conversational dataset covering 13 Arabic-speaking countries, in both MSA and each country's respective dialect, spanning 12 daily-life topics and 54 fine-grained subtopics. We utilize the dataset to form three benchmarking tasks: (i) multiple-choice cultural reasoning, (ii) machine translation between MSA and dialects, and (iii) dialect-steering generation. Our experiments indicate that the performance gap between MSA and Arabic dialects still exists, whereby the models perform worse on all three tasks in the dialectal setup, compared to the MSA one. 

---
# Estimating LLM Grading Ability and Response Difficulty in Automatic Short Answer Grading via Item Response Theory 

**Authors**: Longwei Cong, Sonja Hahn, Sebastian Gombert, Leon Camus, Hendrik Drachsler, Ulf Kroehne  

**Link**: [PDF](https://arxiv.org/pdf/2605.00238)  

**Abstract**: Automated short answer grading (ASAG) with large language models (LLMs) is commonly evaluated with aggregate metrics such as macro-F1 and Cohen's kappa. However, these metrics provide limited insight into how grading performance varies across student responses of differing grading difficulty. We introduce an evaluation framework for LLM-based ASAG based on item response theory (IRT), which models grading correctness as a function of latent grader ability and response grading difficulty. This formulation enables response-level analysis of where LLM graders succeed or fail and reveals robustness differences that are not visible from aggregate scores alone. We apply the framework to 17 open-weight LLMs on the SciEntsBank and Beetle benchmarks. The results show that even models with similar overall performance differ substantially in how sharply their grading accuracy declines as response difficulty increases. In addition, confusion patterns show that errors on difficult responses concentrate disproportionately on the \texttt{partially\_correct\_incomplete} label, indicating a tendency toward intermediate-label collapse under ambiguity. To characterize difficult responses, we further analyze semantic and linguistic correlates of estimated difficulty. Across both datasets, higher difficulty is associated with weaker semantic alignment to the reference answer, stronger contradiction signals, and greater semantic isolation in embedding space. Overall, these results show that item response theory offers a useful framework for evaluating LLM-based ASAG beyond aggregate performance measures. 

---
# Confidence Estimation in Automatic Short Answer Grading with LLMs 

**Authors**: Longwei Cong, Sonja Hahn, Sebastian Gombert, Leon Camus, Hendrik Drachsler, Ulf Kroehne  

**Link**: [PDF](https://arxiv.org/pdf/2605.00200)  

**Abstract**: Automatic Short Answer Grading (ASAG) with generative large language models (LLMs) has recently demonstrated strong performance without task-specific fine-tuning, while also enabling the generation of synthetic feedback for educational assessment. Despite these advances, LLM-based grading remains imperfect, making reliable confidence estimates essential for safe and effective human-AI collaboration in educational decision-making. In this work, we investigate confidence estimation for ASAG with LLMs by jointly considering model-based confidence signals and dataset-derived uncertainty. We systematically compare three model-based confidence estimation strategies, namely verbalizing, latent, and consistency-based confidence estimation, and show that model-based confidence alone is insufficient to reliably capture uncertainty in ASAG. To address this limitation, we propose a hybrid confidence framework that integrates model-based confidence signals with an explicit estimate of dataset-derived aleatoric uncertainty. Aleatoric uncertainty is operationalized by clustering semantically embedded student responses and quantifying within-cluster heterogeneity. Our results demonstrate that the proposed hybrid confidence measure yields more reliable confidence estimates and improves selective grading performance compared to single-source approaches. Overall, this work advances confidence-aware LLM-based grading for human-in-the-loop assessment, supporting more trustworthy AI-assisted educational assessment systems. 

---
# Persona-Grounded Safety Evaluation of AI Companions in Multi-Turn Conversations 

**Authors**: Prerna Juneja, Lika Lomidze  

**Link**: [PDF](https://arxiv.org/pdf/2605.00227)  

**Abstract**: There are growing concerns about the risks posed by AI companion applications designed for emotional engagement. Existing safety evaluations often rely on self-reported user data or interviews, offering limited insights into real-time dynamics. We present the first end-to-end scalable framework for controlled simulation and safety evaluation of multi-turn interactions with AI companion applications. Our framework integrates four key components: persona construction with clinical and psychometric validation, persona-specific scenario generation, scenario-driven multi-turn simulation with a dialogue refinement module that preserves persona fidelity, and harm evaluation. We apply this framework to evaluate how Replika, a widely used AI companion app, responds to high-risk user groups. We construct 9 personas representing individuals with depression, anxiety, PTSD, eating disorders, and incel identity, and collect 1,674 dialogue pairs across 25 high-risk scenarios. We combine emotion modeling and LLM-assisted utterance-and harm-level classification to analyze these exchanges. Results show that Replika exhibits a narrow emotional range dominated by curiosity and care, while frequently mirroring or normalizing unsafe content such as self-harm, disordered eating, and violent-fantasy narratives. These findings highlight how controlled persona simulations can serve as a scalable testbed for evaluating safety risks in AI companions. 

---
# How Frontier LLMs Adapt to Neurodivergence Context: A Measurement Framework for Surface vs. Structural Change in System-Prompted Responses 

**Authors**: Ishan Gupta, Pavlo Buryi  

**Link**: [PDF](https://arxiv.org/pdf/2605.00113)  

**Abstract**: We examine if frontier chat-based large language models (LLMs) adjust their outputs based on neurodivergence (ND) context in system prompts and describe the nature of these adjustments. Specifically, we propose NDBench, a 576-output benchmark involving two frontier models, three system prompt types (baseline, ND-profile assertion, and ND-profile assertion with explicit instructions for adjustments), four canonical ND profiles, and 24 prompts across four categories, one of which involves an adversarial masking strategy.
Four trends emerge consistently from our findings. First, LLMs show significant adaptation under ND context, where fully instructed conditions yield lengthier and more structured outputs, characterized by higher token counts, more headings, and more granular steps (p < 10^-8, Holm-corrected). Second, such adaptation is largely structural in nature: although list density does not change much, there is a marked rise in the frequency of headings and per-step detail. Third, ND persona assertion alone fails to suppress potentially harmful tendencies, as masking-reinforcement decreases only in explicitly instructed cases (36-44% reduction); the reduction rate barely changes in persona assertion conditions.
Moreover, reliability analysis of LLM-based harm assessment reveals that only two out of the six dimensions (masking and reinforcement, validation quality) exceed the pre-defined inter-judge agreement criterion (alpha >= 0.67) and thus can be considered primary results.
NDBench is made publicly available along with its prompts, outputs, code, and other resources, forming a reproducible framework for auditing future LLMs' adaptation to ND awareness. 

---
# Can Coding Agents Reproduce Findings in Computational Materials Science? 

**Authors**: Ziyang Huang, Yi Cao, Ali K. Shargh, Jing Luo, Ruidong Mei, Mohd Zaki, Zhan Liu, Wyatt Bunstine, William Jurayj, Somdatta Goswami, Tyrel McQueen, Michael Shields, Jaafar El-Awady, Paulette Clancy, Benjamin Van Durme, Nicholas Andrews, William Walden, Daniel Khashabi  

**Link**: [PDF](https://arxiv.org/pdf/2605.00803)  

**Abstract**: Large language models are increasingly deployed as autonomous coding agents and have achieved remarkably strong performance on software engineering benchmarks. However, it is unclear whether such success transfers to computational scientific workflows, where tasks require not only strong coding ability, but also the ability to navigate complex, domain-specific procedures and to interpret results in the context of scientific claims. To address this question, we present AutoMat, a benchmark for evaluating LLM-based agents' ability to reproduce claims from computational materials science. AutoMat poses three interrelated challenges: recovering underspecified computational procedures, navigating specialized toolchains, and determining whether the resulting evidence supports a claim. By working closely with subject matter experts, we curate a set of claims from real materials science papers to test whether coding agents can recover and execute the end-to-end workflow needed to support (or undermine) such claims. We then evaluate multiple representative coding agent settings across several foundation models. Our results show that current LLM-based agents obtain low overall success rates on AutoMat, with the best-performing setting achieving a success rate of only 54.1%. Error analysis further reveals that agents perform worst when workflows must be reconstructed from paper text alone and that they fail primarily due to incomplete procedures, methodological deviations, and execution fragility. Taken together, these findings position AutoMat as both a benchmark for computational scientific reproducibility and a tool for diagnosing the current limitations of agentic systems in AI-for-science settings. 

---
# How Language Models Process Out-of-Distribution Inputs: A Two-Pathway Framework 

**Authors**: Hamidreza Saghir  

**Link**: [PDF](https://arxiv.org/pdf/2605.00269)  

**Abstract**: Recent white-box OOD detection methods for LLMs -- including CED, RAUQ, and WildGuard confidence scores -- appear effective, but we show they are structurally confounded by sequence length (|r| >= 0.61) and collapse to near-chance under length-matched evaluation. Even raw attention entropy (mean H(alpha) across heads and layers), a natural baseline we include for completeness, shows the same confound. The confound stems from attention's Theta(log T) dependence on input length. To identify genuine OOD signals after deconfounding, we propose a two-pathway framework: embeddings capture what text is about (effective for topic shifts), while the processing trajectory -- hidden-state evolution across layers -- captures how the model processes input. The relative power of each pathway varies along a vocabulary-transparency spectrum: embedding methods excel on vocabulary-distinctive OOD, while trajectory features detect covert-intent inputs that share vocabulary with normal text (0.721 avg AUROC; Jailbreak: 0.850). Three evidence lines support this framework: (1) a crossover between k-NN and trajectory scoring across 6 tasks, where each pathway wins on different OOD types; (2) a per-layer analysis showing that layer-0 k-NN signal is almost entirely a length artifact (Jailbreak: 0.759 raw -> 0.389 matched) -- processing constructs genuine OOD signal from near-chance embeddings; and (3) circuit attribution showing adversarial tasks engage attention circuits more than semantic tasks (p = 0.022; Jailbreak patching p < 0.001), with partial cross-model replication. Code release upon publication. 

---
# RunAgent: Interpreting Natural-Language Plans with Constraint-Guided Execution 

**Authors**: Arunabh Srivastava, Mohammad A., Khojastepour, Srimat Chakradhar, Sennur Ulukus  

**Link**: [PDF](https://arxiv.org/pdf/2605.00798)  

**Abstract**: Humans solve problems by executing targeted plans, yet large language models (LLMs) remain unreliable for structured workflow execution. We propose RunAgent, a multi-agent plan execution platform that interprets natural-language plans while enforcing stepwise execution through constraints and rubrics. RunAgent bridges the expressiveness of natural language with the determinism of programming via an agentic language with explicit control constructs (e.g., \texttt{IF}, \texttt{GOTO}, \texttt{FORALL}). Beyond verifying syntactic and semantic verification of the step output, which is performed based on the specific instruction of each step, RunAgent autonomously derives and validates constraints based on the description of the task and its instance at each step. RunAgent also dynamically selects among LLM-based reasoning, tool usage, and code generation and execution (e.g., in Python), and incorporates error correction mechanisms to ensure correctness. Finally, RunAgent filters the context history by retaining only relevant information during the execution of each step. Evaluations on Natural-plan and SciBench Datasets demonstrate that RunAgent outperforms baseline LLMs and state-of-the-art PlanGEN methods. 

---
# When RAG Chatbots Expose Their Backend: An Anonymized Case Study of Privacy and Security Risks in Patient-Facing Medical AI 

**Authors**: Alfredo Madrid-García, Miguel Rujas  

**Link**: [PDF](https://arxiv.org/pdf/2605.00796)  

**Abstract**: Background: Patient-facing medical chatbots based on retrieval-augmented generation (RAG) are increasingly promoted to deliver accessible, grounded health information. AI-assisted development lowers the barrier to building them, but they still demand rigorous security, privacy, and governance controls. Objective: To report an anonymized, non-destructive security assessment of a publicly accessible patient-facing medical RAG chatbot and identify governance lessons for safe deployment of generative AI in health. Methods: We used a two-stage strategy. First, Claude Opus 4.6 supported exploratory prompt-based testing and structured vulnerability hypotheses. Second, candidate findings were manually verified using Chrome Developer Tools, inspecting browser-visible network traffic, payloads, API schemas, configuration objects, and stored interaction data. Results: The LLM-assisted phase identified a critical vulnerability: sensitive system and RAG configuration appeared exposed through client-server communication rather than restricted server-side. Manual verification confirmed that ordinary browser inspection allowed collection of the system prompt, model and embedding configuration, retrieval parameters, backend endpoints, API schema, document and chunk metadata, knowledge-base content, and the 1,000 most recent patient-chatbot conversations. The deployment also contradicted its privacy assurances: full conversation records, including health-related queries, were retrievable without authentication. Conclusions: Serious privacy and security failures in patient-facing RAG chatbots can be identified with standard browser tools, without specialist skills or authentication; independent review should be a prerequisite for deployment. Commercial LLMs accelerated this assessment, including under a false developer persona; assistance available to auditors is equally available to adversaries. 

---
# Rethinking LLM Ensembling from the Perspective of Mixture Models 

**Authors**: Jiale Fu, Yuchu Jiang, Peijun Wu, Chonghan Liu, Joey Tianyi Zhou, Xu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.00419)  

**Abstract**: Model ensembling is a well-established technique for improving the performance of machine learning models. Conventionally, this involves averaging the output distributions of multiple models and selecting the most probable label. This idea has been naturally extended to large language models (LLMs), yielding improved performance but incurring substantial computational cost. This inefficiency stems from directly applying conventional ensemble implementation to LLMs, which require a separate forward pass for each model to explicitly compute the ensemble distribution. In this paper, we propose the Mixture-model-like Ensemble (ME). By reinterpreting the ensemble as a mixture model, ME stochastically selects a single model at each step to generate the next token, thereby avoiding the need to explicitly compute the full ensemble distribution. ME is mathematically equivalent to sampling from the ensemble distribution, but requires invoking only one model, making it 1.78x-2.68x faster than conventional ensemble. Furthermore, this perspective connects LLM ensembling and token-level routing methods, suggesting that LLM ensembling is a special case of routing methods. Our findings open new avenues for efficient LLM ensembling and motivate further exploration of token-level routing strategies for LLMs. Our code is available at this https URL. 

---
# Adaptive Querying with AI Persona Priors 

**Authors**: Kaizheng Wang, Yuhang Wu, Assaf Zeevi  

**Link**: [PDF](https://arxiv.org/pdf/2605.00696)  

**Abstract**: We study adaptive querying for learning user-dependent quantities of interest, such as responses to held-out items and psychometric indicators, within tight question budgets. Classical Bayesian design and computerized adaptive testing typically rely on restrictive parametric assumptions or expensive posterior approximations, limiting their use in heterogeneous, high-dimensional, and cold-start settings. We introduce a persona-induced latent variable model that represents a user's state through membership in a finite dictionary of AI personas, each offering response distributions produced by a large language model. This yields expressive priors with closed-form posterior updates and efficient finite-mixture predictions, enabling scalable Bayesian design for sequential item selection. Experiments on synthetic data and WorldValuesBench demonstrate that persona-based posteriors deliver accurate probabilistic predictions and an interpretable adaptive elicitation pipeline. 

---
# On the Role of Artificial Intelligence in Human-Machine Symbiosis 

**Authors**: Ching-Chun Chang, Yuchen Guo, Hanrui Wang, Timo Spinde, Isao Echizen  

**Link**: [PDF](https://arxiv.org/pdf/2605.00440)  

**Abstract**: The evolution of artificial intelligence (AI) has rendered the boundary between humanity and computational machinery increasingly ambiguous. In the presence of more interwoven relationships within human-machine symbiosis, the very notion of AI-generated information becomes difficult to define, as such information arises not from either humans or machines in isolation, but from their mutual shaping. Therefore, a more pertinent question lies not merely in whether AI has participated, but in how it has participated. In general, the role assumed by AI is often specified, either implicitly or explicitly, in the input prompt, yet becomes less apparent or altogether unobservable when the generated content alone is available. Once detached from the dialogue context, the functional role may no longer be traceable. This study considers the problem of tracing the functional role played by AI in natural language generation. A methodology is proposed to infer the latent role specified by the prompt, embed this role into the content during the probabilistic generation process and subsequently recover the nature of AI participation from the resulting text. Experimentation is conducted under a representative scenario in which AI acts either as an assistive agent that edits human-written content or as a creative agent that generates new content from a brief concept. The experimental results support the validity of the proposed methodology in terms of discrimination between roles, robustness against perturbations and preservation of linguistic quality. We envision that this study may contribute to future research on the ethics of AI with regard to whether AI has been used fairly, transparently and appropriately. 

---
# Block-wise Codeword Embedding for Reliable Multi-bit Text Watermarking 

**Authors**: Joeun Kim, HoEun Kim, Dongsup Jin, Young-Sik Kim  

**Link**: [PDF](https://arxiv.org/pdf/2605.00348)  

**Abstract**: Recent multi-bit watermarking methods for large language models (LLMs) prioritize capacity over reliability, often conflating decoding with detection. Our analysis reveals that existing ECC-based extractors suffer from catastrophic false positive rates (FPR), and applying rejection thresholds merely collapses detection sensitivity (TPR) to random guessing. To resolve this structural limitation, we propose \textbf{BREW} (Block-wise Reliable Embedding for Watermarking), a framework shifting the paradigm to \emph{designated verification}. BREW employs a two-stage mechanism: (i) \textbf{blind message estimation} via independent block voting, followed by (ii) \textbf{window-shifting verification} that rigorously validates the payload against local edits. Experiments demonstrate that BREW achieves a TPR of 0.965 with an FPR of 0.02 under 10\% synonym substitution, demonstrating that the high-FPR issue is not an inherent trade-off of multi-bit watermarking, but a solvable structural flaw of prior decoding-centric designs. Our framework is model-agnostic and theoretically grounded, providing a scalable solution for reliable forensic deployment. 

---
# Uniform-Correct Policy Optimization: Breaking RLVR's Indifference to Diversity 

**Authors**: Anamika Lochab, Bolian Li, Ruqi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.00365)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has achieved substantial gains in single-attempt accuracy (Pass@1) on reasoning tasks, yet often suffers from reduced multi-sample coverage (Pass@K), indicating diversity collapse. We identify a structural cause for this degradation: common RLVR objectives, such as GRPO, are indifferent to how probability mass is distributed among correct solutions. Combined with stochastic training dynamics, this indifference induces a self-reinforcing collapse, in which probability mass concentrates on a narrow subset of correct outputs while alternative valid solutions are suppressed. We formalize this collapse mechanism and further characterize the optimal policy structure under two complementary criteria: robustness and entropy-regularized optimality, which identify the Uniform-Correct Policy as uniquely optimal. Motivated by this analysis, we propose Uniform-Correct Policy Optimization (UCPO), a modification to GRPO that adds a conditional uniformity penalty on the policy's distribution over correct solutions. The penalty redistributes gradient signal toward underrepresented correct responses, encouraging uniform allocation of probability mass within the correct set. Across three models (1.5B-7B parameters) and five mathematical reasoning benchmarks, UCPO improves Pass@K and diversity while maintaining competitive Pass@1, achieving up to +10\% absolute improvement on AIME24 at Pass@64 and up to 45\% higher equation-level diversity within the correct set. The code is available at this https URL. 

---
# AgentFloor: How Far Up the tool use Ladder Can Small Open-Weight Models Go? 

**Authors**: Ranit Karmakar, Jayita Chatterjee  

**Link**: [PDF](https://arxiv.org/pdf/2605.00334)  

**Abstract**: Production agentic systems make many model calls per user request, and most of those calls are short, structured, and routine. This raises a practical routing question that existing evaluations do not directly answer: which parts of an agent workflow truly require large frontier intelligence, and which can be handled by smaller models? We introduce AgentFloor, a deterministic 30-task benchmark organized as a six-tier capability ladder, spanning instruction following, tool use, multi-step coordination, and long-horizon planning under persistent constraints. We evaluate 16 open-weight models, from 0.27B to 32B parameters, alongside GPT-5 across 16,542 scored runs. Our results reveal a clear boundary of model necessity. Small and mid-sized open-weight models are already sufficient for much of the short-horizon, structured tool use work that dominates real agent pipelines, and in aggregate, the strongest open-weight model matches GPT-5 on our benchmark while being substantially cheaper and faster to run. The gap appears most clearly on long-horizon planning tasks that require sustained coordination and reliable constraint tracking over many steps, where frontier models still hold an advantage, though neither side reaches strong reliability. We also find that this boundary is not explained by scale alone: some failures respond to targeted interventions, but the effects are model-specific rather than universal. These findings suggest a practical design principle for agentic systems: use smaller open-weight models for the broad base of routine actions, and reserve large frontier models for the narrower class of tasks that truly demand deeper planning and control. We release the benchmark, harness, sweep configurations, and full run corpus. 

---
# State Stream Transformer (SST) V2: Parallel Training of Nonlinear Recurrence for Latent Space Reasoning 

**Authors**: Thea Aviss  

**Link**: [PDF](https://arxiv.org/pdf/2605.00206)  

**Abstract**: Current transformers discard their rich latent residual stream between positions, reconstructing latent reasoning context at each new position and leaving potential reasoning capacity untapped. The State Stream Transformer (SST) V2 enables parameter-efficient reasoning in continuous latent space through an FFN-driven nonlinear recurrence at each decoder layer, where latent states are streamed horizontally across the full sequence via a learned blend. This same mechanism supports continuous latent deliberation per position at inference time, dedicating additional FLOPs to exploring abstract reasoning before committing to a token. A two-pass parallel training procedure resolves the sequential dependency of the recurrence to allow compute-efficient training. Hidden state analysis shows the state stream facilitates reasoning through exploration of distinct semantic basins in continuous latent space, where transitions at content-dependent positions move the model into a substantially different Bayesian posterior, directly influencing the latent space at future positions. We also find, via a learned probe, that at the first generated token position, the latent state already predicts whether the eventual answer will survive or break under additional latent computation for every subsequent position. Co-trained into an existing 27B backbone using only a small dataset of GSM8K examples, the SST delivers a +15.15 point gain over a fine-tuning-matched baseline on out-of-distribution GPQA-Diamond and cuts that same baseline's remaining GSM8K errors by 46%, together showing that the reasoning improvement is attributable to the architectural mechanism rather than scale or training data. On GPQA-Diamond, the resulting 27B SST also achieves higher accuracy than several larger open-weight and proprietary systems, including open-weight models up to 25 times larger. 

---
# Wasserstein Distributionally Robust Regret Optimization for Reinforcement Learning from Human Feedback 

**Authors**: Yikai Wang, Shang Liu, Jose Blanchet  

**Link**: [PDF](https://arxiv.org/pdf/2605.00155)  

**Abstract**: Reinforcement learning from human feedback (RLHF) has become a core post-training step for aligning large language models, yet the reward signal used in RLHF is only a learned proxy for true human utility. From an operations research perspective, this creates a decision problem under objective misspecification: the policy is optimized against an estimated reward, while deployment performance is determined by an unobserved objective. The resulting gap leads to reward over-optimization, or Goodharting, where proxy reward continues to improve even after true quality deteriorates. Existing mitigations address this problem through uncertainty penalties, pessimistic rewards, or conservative constraints, but they can be computationally burdensome and overly pessimistic. We propose Wasserstein distributionally robust regret optimization (DRRO) for RLHF. Instead of pessimizing worst-case value as in standard DRO, DRRO pessimizes worst-case regret relative to the best policy under the same plausible reward perturbation. We study the promptwise problem through a simplex allocation model and show that, under an $\ell_1$ ambiguity set, the inner worst-case regret admits an exact solution and the optimal policy has a water-filling structure. These results lead to a practical policy-gradient algorithm with a simple sampled-bonus interpretation and only minor changes to PPO/GRPO-style RLHF training. The framework also clarifies theoretically why DRRO is less pessimistic than DRO, and our experiments show that DRRO mitigates over-optimization more effectively than existing baselines while standard DRO is systematically over-pessimistic. 

---
# EGREFINE: An Execution-Grounded Optimization Framework for Text-to-SQL Schema Refinement 

**Authors**: Jiaqian Wang, Yutao Qi, Wenjin Hou, Yu Pang, Rui Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.00628)  

**Abstract**: Text-to-SQL enables non-expert users to query databases in natural language, yet real-world schemas often suffer from ambiguous, abbreviated, or inconsistent naming conventions that degrade model accuracy. Existing approaches treat schemas as fixed and address errors downstream. In this paper, we frame schema refinement as a constrained optimization problem: find a renaming function that maximizes downstream Text-to-SQL execution accuracy while preserving query equivalence through database views. We analyze the computational hardness of this problem, which motivates a column-wise greedy decomposition, and instantiate it as EGRefine: a four-phase pipeline that screens ambiguous columns, generates context-aware candidate names, verifies them through execution-grounded feedback, and materializes the result as non-destructive SQL views. The pipeline carries two structural properties: column-local non-degradation, ensured by the conservative selection rule in the verification phase, and database-level query equivalence, ensured by the view-based materialization phase. Together they make the resulting refinement safe by construction at the column level, with cross-column and prompt-level interactions handled empirically rather than analytically. Across controlled schema-degradation, real-world, and enterprise benchmarks, EGRefine recovers accuracy lost to schema naming noise where applicable and correctly abstains where the underlying task exceeds current Text-to-SQL capabilities, with refined schemas transferring across model families to enable refine-once, serve-many-models deployment. Code and data are publicly available at this https URL. 

---
# Borrowed Geometry: Computational Reuse of Frozen Text-Pretrained Transformer Weights Across Modalities 

**Authors**: Abay Bektursun  

**Link**: [PDF](https://arxiv.org/pdf/2605.00333)  

**Abstract**: Frozen Gemma 4 31B weights pretrained exclusively on text tokens, unmodified, transfer across modality boundaries through a thin trainable interface. (1) OGBench scene-play-singletask-task1-v0: $+4.33$pt over published GCIQL at $n=3$ with std 0.74 -- a published-SOTA win on a robotic manipulation task the substrate has never seen. (2) D4RL Walker2d-medium-v2: Decision-Transformer parity ($76.2 \pm 0.8$, $n=3$) at $0.43\times$ DT's trainable count, with the frozen substrate compressing to a 5L slice ($+1.66$pt over the 6L baseline at $n=3$). (3) Associative recall as the cleanest pretraining-load-bearing case: the frozen slice + a 113K-parameter linear interface reaches L30 best-checkpoint per-bit error 0.0505 ($n=2$); a 6.36M-parameter from-scratch trained transformer at matched capacity ($1/\sqrt{d_k}$ scaling, two seeds, LR sweep) cannot solve the task at all under the protocol (best L30 = 0.4395), an $8.7\times$ advantage. Architecture-alone falsifications: a frozen random transformer with correct $1/\sqrt{d_k}$ scaling stays at random-chance loss for 50k steps; a random-init Gemma slice fails OGBench cube-double-play-task1 entirely (0.89% across $n=3$ where pretrained reaches 60%). A dual-measurement protocol -- text-activation probing on 95 English sentences plus task-ablation on a non-language target -- names individual heads independently identifiable on both protocols: head L26.28 scores $3.7\times$ the slice mean for English token-copying and is the #2 most-critical head for binary copy ablation ($\Delta$ L30 $= +0.221$); three further heads (L27.28, L27.2, L27.3) classify by the same protocol. The mechanism is single-model and the cross-modality results are single-task within their respective benchmarks; cross-model replication is structurally constrained because Gemma 4 31B is the only model on the small-scale Pareto frontier as of April 2026. 

---
# Technical Report: Activation Residual Hessian Quantization (ARHQ) for Low-Bit LLM Quantization 

**Authors**: YiFeng Wang, Zhun Sun, Keisuke Sakaguchi  

**Link**: [PDF](https://arxiv.org/pdf/2605.00140)  

**Abstract**: We present Activation Residual Hessian Quantization (ARHQ), a post-training weight splitting method designed to mitigate error propagation in low-bit activation-weight quantization. By constructing an input-side residual Hessian from activation quantization residuals (G_x), ARHQ analytically identifies and isolates error-sensitive weight directions into a high-precision low-rank branch. This is achieved via a closed-form truncated SVD on the scaled weight matrix W G^{1/2}_x . Experimental results on Qwen3-4B-Thinking-2507 demonstrate that ARHQ significantly improves layer-wise SNR and preserves downstream reasoning performance on ZebraLogic even under aggressive quantization. The code is available at this https URL. 

---
# ResRL: Boosting LLM Reasoning via Negative Sample Projection Residual Reinforcement Learning 

**Authors**: Zihan Lin, Xiaohan Wang, Jie Cao, Jiajun Chai, Li Wang, Xiaodong Lu, Wei Lin, Ran He, Guojun Yin  

**Link**: [PDF](https://arxiv.org/pdf/2605.00380)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) enhances reasoning of Large Language Models (LLMs) but usually exhibits limited generation diversity due to the over-incentivization of positive rewards. Although methods like Negative Sample Reinforcement (NSR) mitigate this issue by upweighting penalty from negative samples, they may suppress the semantic distributions shared between positive and negative responses. To boost reasoning ability without losing diversity, this paper proposes negative sample projection Residual Reinforcement Learning (ResRL) that decouples similar semantic distributions among positive and negative responses. We theoretically link Lazy Likelihood Displacement (LLD) to negative-positive head-gradient interference and derive a single-forward proxy that upper-bounds representation alignment to guide conservative advantage reweighting. ResRL then projects negative-token hidden representations onto an SVD-based low-rank positive subspace and uses projection residuals to modulate negative gradients, improving reasoning while preserving diversity and outperforming strong baselines on average across twelve benchmarks spanning Mathematics, Code, Agent Tasks, and Function Calling. Notably, ResRL surpasses NSR on mathematical reasoning by 9.4\% in Avg@16 and 7.0\% in Pass@128. Code is available at this https URL. 

---
# MoDAl: Self-Supervised Neural Modality Discovery via Decorrelation for Speech Neuroprosthesis 

**Authors**: Yuanhao Chen, Peter Chin  

**Link**: [PDF](https://arxiv.org/pdf/2605.00025)  

**Abstract**: Speech neuroprosthesis systems decode intended speech from neural activity in the absence of audible output, offering a path to restoring communication for individuals with speech-impairing conditions. Current approaches decode predominantly from motor cortical areas, discarding others -- such as area 44, part of Broca's area -- that may encode complementary linguistic information. We introduce MoDAl (Modality Decorrelation and Alignment), a framework that discovers complementary neural modalities through the interplay of two objectives in a shared projection space. A contrastive loss aligns each of several parallel brain encoders with the text embeddings of a pretrained large language model (LLM), while a decorrelation loss prevents the encoders from coalescing to duplicative representations. We prove that these objectives are in productive tension: Contrastive alignment induces transitive modality coalescence, which decorrelation must counteract for the framework to discover diverse neurolinguistic modalities. On the Brain-to-Text Benchmark '24, MoDAl reduces word error rate (WER) from 26.3% to 21.6% compared to the previous best end-to-end method, with the gain from incorporating previously discarded area 44 signals arising entirely from the decorrelation mechanism. Analysis of the discovered modalities reveals functional specialization: Encoders receiving area 44 input capture structural and syntactic properties (sentence length, grammatical voice, wh-words), consistent with the neurolinguistic understanding of Broca's area. 

---
# Odysseus: Scaling VLMs to 100+ Turn Decision-Making in Games via Reinforcement Learning 

**Authors**: Chengshuai Shi, Wenzhe Li, Xinran Liang, Yizhou Lu, Wenjia Yang, Ruirong Feng, Seth Karten, Ziran Yang, Zihan Ding, Gabriel Sarch, Danqi Chen, Karthik Narasimhan, Chi Jin  

**Link**: [PDF](https://arxiv.org/pdf/2605.00347)  

**Abstract**: Given the rapidly growing capabilities of vision-language models (VLMs), extending them to interactive decision-making tasks such as video games has emerged as a promising frontier. However, existing approaches either rely on large-scale supervised fine-tuning (SFT) on human trajectories or apply reinforcement learning (RL) only in relatively short-horizon settings (typically around 20--30 turns). In this work, we study RL-based training of VLMs for long-horizon decision-making in Super Mario Land, a visually grounded environment requiring 100+ turns of interaction with coordinated perception, reasoning, and action. We begin with a systematic investigation of key algorithmic components and propose an adapted variant of PPO with a lightweight turn-level critic, which substantially improves training stability and sample efficiency over critic-free methods such as GRPO and Reinforce++. We further show that pretrained VLMs provide strong action priors, significantly improving sample efficiency during RL training and reducing the need for manual design choices such as action engineering, compared to classical deep RL trained from scratch. Building on these insights, we introduce Odysseus, an open training framework for VLM agents, achieving substantial gains across multiple levels of the game and at least 3 times average game progresses than frontier models. Moreover, the trained models exhibit consistent improvements under both in-game and cross-game generalization settings, while maintaining general-domain capabilities. Overall, our results identify key ingredients for making RL stable and effective in long-horizon, multi-modal settings, and provide practical guidance for developing VLMs as embodied agents. 

---
# RouteProfile: Elucidating the Design Space of LLM Profiles for Routing 

**Authors**: Jingjun Xu, Hongji Pu, Tao Feng, Haozhen Zhang, Jiaxuan You, Ge Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.00180)  

**Abstract**: As the large language model (LLM) ecosystem expands, individual models exhibit varying capabilities across queries, benchmarks, and domains, motivating the development of LLM routing. While prior work has largely focused on router mechanism design, LLM profiles, which capture model capabilities, remain underexplored. In this work, we ask: How does LLM profile design affect routing performance across different routers? Addressing this question helps clarify the role of profiles in routing, disentangle profile design from router design, and enable fairer comparison and more principled development of routing systems. To this end, we view LLM profiling as a structured information integration problem over heterogeneous interaction histories. We develop a general design space of LLM profiles, named RouteProfile, along four key dimensions: organizational form, representation type, aggregation depth, and learning configuration. Through systematic evaluation across three representative routers under both standard and new-LLM generalization settings, we show that: (1) structured profiles consistently outperform flat ones; (2) query-level signals are more reliable than coarse domain-level signals; and (3) generalization to newly introduced models benefits most from structured profiles under trainable configurations. Overall, our work highlights LLM profile design as an important direction for future routing research. 

---
# AEM: Adaptive Entropy Modulation for Multi-Turn Agentic Reinforcement Learning 

**Authors**: Haotian Zhao, Yuxin Zhang, Songlin Zhou, Stephen S.-T. Yau, Wenyu Zhang, Lun Tian, Tianshu Zhu, Yifeng Huang, Yucheng Zeng, Jingnan Gu, Daxiang Dong, Jianmin Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.00425)  

**Abstract**: Reinforcement learning (RL) has significantly advanced the ability of large language model (LLM) agents to interact with environments and solve multi-turn tasks. Yet effective training remains challenging, as sparse, outcome-only rewards make it difficult to assign credit to individual steps in an agent's action trajectory. A common remedy is to introduce dense intermediate supervision, such as process reward models or auxiliary self-supervised signals, but this increases supervision and tuning complexity and often generalizes poorly across tasks and domains. This paper presents AEM, a supervision-free credit assignment method that adaptively modulates entropy dynamics during RL training to achieve a more effective exploration-exploitation trade-off. Theoretically, we elevate entropy analysis from the token level to the response level to reduce token sampling variance and show that entropy drift under natural gradients is intrinsically governed by the product of the advantage and the relative response surprisal. Specifically, we derive a practical proxy to reshape training dynamics, enabling a natural transition from exploration to exploitation. Extensive experiments across various benchmarks and models ranging from 1.5B to 32B parameters demonstrate the effectiveness of AEM, including a notable 1.4 percent gain when integrated into a state-of-the-art baseline on the highly challenging SWE-bench-Verified benchmark. 

---
# To Call or Not to Call: A Framework to Assess and Optimize LLM Tool Calling 

**Authors**: Qinyuan Wu, Soumi Das, Mahsa Amani, Arijit Nag, Seungeon Lee, Krishna P. Gummadi, Abhilasha Ravichander, Muhammad Bilal Zafar  

**Link**: [PDF](https://arxiv.org/pdf/2605.00737)  

**Abstract**: Agentic AI architectures augment LLMs with external tools, unlocking strong capabilities. However, tool use is not always beneficial; some calls may be redundant or even harmful. Effective tool use, therefore, hinges on a core LLM decision: whether to call or not call a tool, when performing a task. This decision is particularly challenging for web search tools, where the benefits of external information depend on the model's internal knowledge and its ability to integrate potentially noisy tool responses. We introduce a principled framework inspired by decision-making theory to evaluate web search tool-use decisions along three key factors: necessity, utility, and affordability. Our analysis combines two complementary lenses: a normative perspective that infers true need and utility from an optimal allocation of tool calls, and a descriptive perspective that infers the model's self-perceived need and utility from their observed behaviors. We find that models' perceived need and utility of tool calls are often misaligned with their true need and utility. Building on this framework, we train lightweight estimators of need and utility based on models' hidden states. Our estimators enable simple controllers that can improve decision quality and lead to stronger task performance than the self-perceived set up across three tasks and six models. 

---
# Token Arena: A Continuous Benchmark Unifying Energy and Cognition in AI Inference 

**Authors**: Yuxuan Gao, Megan Wang, Yi Ling Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.00300)  

**Abstract**: Public inference benchmarks compare AI systems at the model and provider level, but the unit at which deployment decisions are actually made is the endpoint: the (provider, model, stock-keeping-unit) tuple at which a specific quantization, decoding strategy, region, and serving stack is exposed. We introduce TokenArena, a continuous benchmark that measures inference at endpoint granularity along five core axes (output speed, time to first token, workload-blended price, effective context, and quality on the live endpoint) and synthesizes them, together with a modeled energy estimate, into three headline composites: joules per correct answer, dollars per correct answer, and endpoint fidelity (output-distribution similarity to a first-party reference). The framework's novelty is empirical and methodological. Across 78 endpoints serving 12 model families, the same model on different endpoints differs in mean accuracy by up to 12.5 points on math and code, in fingerprint similarity to first party by up to 12 points, in tail latency by an order of magnitude, and in modeled joules per correct answer by a factor of 6.2. We further show that workload-aware blended pricing reorders the leaderboard substantially: 7 of 10 top-ranked endpoints under the chat preset (3:1 input:output) fall out of the top 10 under the retrieval-augmented preset (20:1), and the reasoning preset (1:5) elevates frontier closed models that the chat preset penalizes on price. We release the framework, schema, probe and eval harness, and a v1.0 leaderboard snapshot under CC BY 4.0. TokenArena is a methodology, not a single ranking; we publish full provenance and limitations and welcome external replication. 

---
# Position: agentic AI orchestration should be Bayes-consistent 

**Authors**: Theodore Papamarkou, Pierre Alquier, Matthias Bauer, Wray Buntine, Andrew Davison, Gintare Karolina Dziugaite, Maurizio Filippone, Andrew Y. K. Foong, Vincent Fortuin, Dimitris Fouskakis, Jes Frellsen, Eyke Hüllermeier, Theofanis Karaletsos, Mohammad Emtiyaz Khan, Nikita Kotelevskii, Salem Lahlou, Yingzhen Li, Fang Liu, Clare Lyle, Thomas Möllenhoff, Konstantina Palla, Maxim Panov, Yusuf Sale, Kajetan Schweighofer, Artem Shelmanov, Siddharth Swaroop, Martin Trapp, Willem Waegeman, Andrew Gordon Wilson, Alexey Zaytsev  

**Link**: [PDF](https://arxiv.org/pdf/2605.00742)  

**Abstract**: LLMs excel at predictive tasks and complex reasoning tasks, but many high-value deployments rely on decisions under uncertainty, for example, which tool to call, which expert to consult, or how many resources to invest. While the usefulness and feasibility of Bayesian approaches remain unclear for LLM inference, this position paper argues that the control layer of an agentic AI system (that orchestrates LLMs and tools) is a clear case where Bayesian principles should shine. Bayesian decision theory provides a framework for agentic systems that can help to maintain beliefs over task-relevant latent quantities, to update these beliefs from observed agentic and human-AI interactions, and to choose actions. Making LLMs themselves explicitly Bayesian belief-updating engines remains computationally intensive and conceptually nontrivial as a general modeling target. In contrast, this paper argues that coherent decision-making requires Bayesian principles at the orchestration level of the agentic system, not necessarily the LLM agent parameters. This paper articulates practical properties for Bayesian control that fit modern agentic AI systems and human-AI collaboration, and provides concrete examples and design patterns to illustrate how calibrated beliefs and utility-aware policies can improve agentic AI orchestration. 

---
# ARMOR 2025: A Military-Aligned Benchmark for Evaluating Large Language Model Safety Beyond Civilian Contexts 

**Authors**: Sydney Johns, Heng Jin, Chaoyu Zhang, Y. Thomas Hou, Wenjing Lou  

**Link**: [PDF](https://arxiv.org/pdf/2605.00245)  

**Abstract**: Large language models (LLMs) are now being explored for defense applications that require reliable and legally compliant decision support. They also hold significant potential to enhance decision making, coordination, and operational efficiency in military contexts. These uses demand evaluation methods that reflect the doctrinal standards that guide real military operations. Existing safety benchmarks focus on general social risks and do not test whether models follow the legal and ethical rules that govern real military operations. To address this gap, we introduce ARMOR 2025, a military aligned safety benchmark grounded in three core military doctrines the Law of War, the Rules of Engagement, and the Joint Ethics Regulation. We extract doctrinal text from these sources and generate multiple choice questions that preserve the intended meaning of each rule. The benchmark is organized through a taxonomy informed by the Observe Orient Decide Act (OODA) decision making framework. This structure enables systematic testing of accuracy and refusal across military relevant decision types. This benchmark features a structured 12-category taxonomy, 519 doctrinally grounded prompts, and rigorous evaluation procedures applied to 21 commercial LLMs. Evaluation results reveal critical gaps in safety alignment for military applications. 

---
# Thinking in Text and Images: Interleaved Vision--Language Reasoning Traces for Long-Horizon Robot Manipulation 

**Authors**: Jinkun Liu, Haohan Chi, Lingfeng Zhang, Yifan Xie, YuAn Wang, Long Chen, Hangjun Ye, Xiaoshuai Hao, Wenbo Ding  

**Link**: [PDF](https://arxiv.org/pdf/2605.00438)  

**Abstract**: Long-horizon robotic manipulation requires plans that are both logically coherent and geometrically grounded. Existing Vision-Language-Action policies usually hide planning in latent states or expose only one modality: text-only chain-of-thought encodes causal order but misses spatial constraints, while visual prediction provides geometric cues but often remains local and semantically underconstrained. We introduce Interleaved Vision--Language Reasoning (IVLR), a policy framework built around \trace{}, an explicit intermediate representation that alternates textual subgoals with visual keyframes over the full task horizon. At test time, a single native multimodal transformer self-generates this global semantic-geometric trace from the initial observation and instruction, caches it, and conditions a closed-loop action decoder on the trace, original instruction, and current observation. Because standard robot datasets lack such traces, we construct pseudo-supervision by temporally segmenting demonstrations and captioning each stage with a vision-language model. Across simulated benchmarks for long-horizon manipulation and visual distribution shift, \method{} reaches 95.5\% average success on LIBERO, including 92.4\% on LIBERO-Long, and 59.4\% overall success on SimplerEnv-WidowX. Ablations show that both modalities are necessary: without traces, LIBERO-Long success drops to 37.7\%; text-only and vision-only traces reach 62.0\% and 68.4\%, while the full interleaved trace reaches 92.4\%. Stress tests with execution perturbations and masked trace content show moderate degradation, suggesting that the trace can tolerate local corruption and moderate execution drift, but remains limited under stale or incorrect global plans. 

---
# Minimal, Local, Causal Explanations for Jailbreak Success in Large Language Models 

**Authors**: Shubham Kumar, Narendra Ahuja  

**Link**: [PDF](https://arxiv.org/pdf/2605.00123)  

**Abstract**: Safety trained large language models (LLMs) can often be induced to answer harmful requests through jailbreak prompts. Because we lack a robust understanding of why LLMs are susceptible to jailbreaks, future frontier models operating more autonomously in higher-stakes settings may similarly be vulnerable to such attacks. Prior work has studied jailbreak success by examining the model's intermediate representations, identifying directions in this space that causally encode concepts like harmfulness and refusal. Then, they globally explain all jailbreak attacks as attempting to reduce or strengthen these concepts (e.g., reduce harmfulness). However, different jailbreak strategies may succeed by strengthening or suppressing different intermediate concepts, and the same jailbreak strategy may not work for different harmful request categories (e.g., violence vs. cyberattack); thus, we seek to give a local explanation -- i.e., why did this specific jailbreak succeed? To address this gap, we introduce LOCA, a method that gives Local, CAusal explanations of jailbreak success by identifying a minimal set of interpretable, intermediate representation changes that causally induce model refusal on an otherwise successful jailbreak request. We evaluate LOCA on harmful original-jailbreak pairs from a large jailbreak benchmark across Gemma and Llama chat models, comparing against prior methods adapted to this setting. LOCA can successfully induce refusal by making, on average, six interpretable changes; prior work routinely fails to achieve refusal even after 20 changes. LOCA is a step toward mechanistic, local explanations of jailbreak success in LLMs. Code to be released. 

---
# TADI: Tool-Augmented Drilling Intelligence via Agentic LLM Orchestration over Heterogeneous Wellsite Data 

**Authors**: Rong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2605.00060)  

**Abstract**: We present TADI (Tool-Augmented Drilling Intelligence), an agentic AI system that transforms drilling operational data into evidence-based analytical intelligence. Applied to the Equinor Volve Field dataset, TADI integrates 1,759 daily drilling reports, selected WITSML real-time objects, 15,634 production records, formation tops, and perforations into a dual-store architecture: DuckDB for structured queries over 12 tables with 65,447 rows, and ChromaDB for semantic search over 36,709 embedded documents. Twelve domain-specialized tools, orchestrated by a large language model via iterative function calling, support multi-step evidence gathering that cross-references structured drilling measurements with daily report narratives. The system parses all 1,759 DDR XML files with zero errors, handles three incompatible well naming conventions, and is backed by 95 automated tests plus a 130-question stress-question taxonomy spanning six operational categories. We formalize the agent's behavior as a sequential tool-selection problem and propose the Evidence Grounding Score (EGS) as a simple grounding-compliance proxy based on measurements, attributed DDR quotations, and required answer sections. The complete 6,084-line, framework-free implementation is reproducible given the public Volve download and an API key, and the case studies and qualitative ablation analysis suggest that domain-specialized tool design, rather than model scale alone, is the primary driver of analytical quality in technical operations. 

---
# TUR-DPO: Topology- and Uncertainty-Aware Direct Preference Optimization 

**Authors**: Abdulhady Abas Abdullah, Fatemeh Daneshfar, Seyedali Mirjalili, Mourad Oussalah  

**Link**: [PDF](https://arxiv.org/pdf/2605.00224)  

**Abstract**: Aligning large language models (LLMs) with human preferences is commonly done via reinforcement learning from human feedback (RLHF) with Proximal Policy Optimization (PPO) or, more simply, via Direct Preference Optimization (DPO). While DPO is stable and RL-free, it treats preferences as flat winner vs. loser signals and is sensitive to noisy or brittle preferences arising from fragile chains of thought. We propose TUR-DPO, a topology- and uncertainty-aware variant of DPO that rewards how answers are derived, not only what they say, by eliciting lightweight reasoning topologies and combining semantic faithfulness, utility, and topology quality into a calibrated uncertainty signal. A small learnable reward is factorized over these signals and incorporated into an uncertainty-weighted DPO objective that remains RL-free and relies only on a fixed or moving reference policy. Empirically, across open 7-8B models and benchmarks spanning mathematical reasoning, factual question answering, summarization, and helpful/harmless dialogue, TUR-DPO improves judge win-rates, faithfulness, and calibration relative to DPO while preserving training simplicity and avoiding online rollouts. We further observe consistent gains in multimodal and long-context settings, and show that TUR-DPO matches or exceeds PPO on reasoning-centric tasks while maintaining operational simplicity. 

---
# Are Tools All We Need? Unveiling the Tool-Use Tax in LLM Agents 

**Authors**: Kaituo Zhang, Zhen Xiong, Mingyu Zhong, Zhimeng Jiang, Zhouyuan Yuan, Zhecheng Li, Ying Lin  

**Link**: [PDF](https://arxiv.org/pdf/2605.00136)  

**Abstract**: Tool-augmented reasoning has become a popular direction for LLM-based agents, and it is widely assumed to improve reasoning and reliability. However, we demonstrate that this consensus does not always hold: in the presence of semantic distractors, tool-augmented reasoning does not necessarily outperform native CoT. To explain this performance gap, we propose a Factorized Intervention Framework that isolates the cost of prompt formatting, the overhead of the tool-calling protocol, and the actual gain from executing tools. Our analysis reveals a critical tradeoff: under semantic noise, the gains from tools often fail to offset the "tool-use tax", which is the performance degradation introduced by the tool-calling protocol itself. To address this, we introduce G-STEP, a lightweight inference-time gate to mitigate protocol-induced errors. While this yields partial recovery, our findings suggest that more substantial improvements still require strengthening the model's intrinsic reasoning and tool-interaction capabilities. 

---
# AgentReputation: A Decentralized Agentic AI Reputation Framework 

**Authors**: Mohd Sameen Chishti, Damilare Peter Oyinloye, Jingyue Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.00073)  

**Abstract**: Decentralized, agentic AI marketplaces are rapidly emerging to support software engineering tasks such as debugging, patch generation, and security auditing, often operating without centralized oversight. However, existing reputation mechanisms fail in this setting for three fundamental reasons: agents can strategically optimize against evaluation procedures; demonstrated competence does not reliably transfer across heterogeneous task contexts; and verification rigor varies widely, from lightweight automated checks to costly expert review. Current approaches to reputation drawing on federated learning, blockchain-based AI platforms, and large language model safety research are unable to address these challenges in combination. We therefore propose \textbf{AgentReputation}, a decentralized, three-layer reputation framework for agentic AI systems. The framework separates task execution, reputation services, and tamper-proof persistence to both leverage their respective strengths and enable independent evolution. The framework introduces explicit verification regimes linked to agent reputation metadata, as well as context-conditioned reputation cards that prevent reputation conflation across domains and task types. In addition, AgentReputation provides a decision-facing policy engine that supports resource allocation, access control, and adaptive verification escalation based on risk and uncertainty. Building on this framework, we outline several future research directions, including the development of verification ontologies, methods for quantifying verification strength, privacy-preserving evidence mechanisms, cold-start reputation bootstrapping, and defenses against adversarial manipulation. 

---
# Agentic AI for Trip Planning Optimization Application 

**Authors**: Tiejin Chen, Ahmadreza Moradipari, Kyungtae Han, Hua Wei, Nejib Ammar  

**Link**: [PDF](https://arxiv.org/pdf/2605.00276)  

**Abstract**: Trip planning for intelligent vehicles increasingly requires selecting optimal routes rather than merely producing feasible itineraries, as interacting factors such as travel time, energy consumption, and traffic conditions directly affect plan quality. Yet existing systems are largely designed for feasibility-oriented planning, and current benchmarks provide only reference answers without ground truth, preventing objective evaluation of optimization performance. In our paper, we address these limitations with an agentic AI framework that enables dynamic refinement through an orchestration agent coordinating specialized agents for traffic, charging, and points of interest, and with the Trip-planning Optimization Problems Dataset, which supplies definitive optimal solutions and category-level task structure for fine-grained analysis. Experiments show that our system achieves 77.4\% accuracy on the TOP Benchmark, significantly outperforming single-agent and workflow-based multi-agent baselines, demonstrating the importance of orchestrated agentic reasoning for robust trip planning optimization. 

---
# GeoContra: From Fluent GIS Code to Verifiable Spatial Analysis with Geography-Grounded Repair 

**Authors**: Yinhao Xiao, Rongbo Xiao, Yihan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.00782)  

**Abstract**: Reliable spatial analysis in GIScience requires preserving coordinate semantics, topology, units, and geographic plausibility. Current LLM-based GIS systems generate fluent scripts but rarely enforce these geographic rules at scale. We present GeoContra, a verification and repair framework for LLM-driven Python GIS workflows. It represents each task as an executable geospatial contract-including natural-language questions, schemas, CRS metadata, expected outputs, spatial predicates, topology, metrics, required operations, and forbidden shortcuts. Generated programs undergo static rule inspection, runtime validation, and semantic verification, with violations fed back into a bounded repair loop. Evaluated on 7,079 real geospatial tasks across 15 Boston-area zones, 9 task families, and 11 open-source models (600 runs each), GeoContra improves spatial correctness on closed models from 47.6% to 77.5% for DeepSeek-V4 and from 57.7% to 81.5% for Kimi-K2.5. Across 11 open models, average correctness rises by 26.6%. GeoContra turns fluent code production into verifiable spatial analysis, catching negative travel times, CRS/field-schema violations, missing predicates, and brittle output casts that otherwise yield executable but geographically invalid results. 

---
# Persistent Visual Memory: Sustaining Perception for Deep Generation in LVLMs 

**Authors**: Siyuan Huang, Xiaoye Qu, Yafu Li, Tong Zhu, Zefeng He, Muxin Fu, Daizong Liu, Wei-Long Zheng, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2605.00814)  

**Abstract**: While autoregressive Large Vision-Language Models (LVLMs) demonstrate remarkable proficiency in multimodal tasks, they face a "Visual Signal Dilution" phenomenon, where the accumulation of textual history expands the attention partition function, causing visual attention to decay inversely with generated sequence length. To counteract this, we propose Persistent Visual Memory (PVM), a lightweight learnable module designed to ensure sustained, on-demand visual perception. Integrated as a parallel branch alongside the Feed-Forward Network (FFN) in LVLMs, PVM establishes a distance-agnostic retrieval pathway that directly provides visual embeddings for precise visual perception, thereby structurally mitigating the signal suppression inherent to deep generation. Extensive experiments on Qwen3-VL models demonstrate that PVM brings notable improvements with negligible parameter overhead, delivering consistent average accuracy gains across both 4B and 8B scales, particularly in complex reasoning tasks that demand persistent visual perception. Furthermore, in-depth analysis reveals that PVM can resist length-induced signal decay and accelerate internal prediction convergence. 

---
# Make Your LVLM KV Cache More Lightweight 

**Authors**: Xihao Chen, Yangyang Guo, Roger Zimmermann  

**Link**: [PDF](https://arxiv.org/pdf/2605.00789)  

**Abstract**: Key-Value (KV) cache has become a de facto component of modern Large Vision-Language Models (LVLMs) for inference. While it enhances decoding efficiency in Large Language Models (LLMs), its direct adoption in LVLMs introduces substantial GPU memory overhead due to the large number of vision tokens processed during the prefill stage. To tackle this problem, we propose LightKV, a novel approach that reduces KV cache size by exploiting the redundancy among vision-token embeddings. Guided by text prompts, LightKV employs cross-modality message passing to aggregate informative messages across vision tokens and progressively compress them during prefill. This prompt-aware guidance distinguishes our method from prior vision-only compression strategies. We evaluate LightKV on eight open-source LVLMs across eight public benchmark datasets, e.g., MME and SeedBench. Experimental results demonstrate that with only 55% of the original vision tokens, LightKV (a) halves the vision-token KV cache size, (b) reduces computation by up to 40%, and (c) preserves general-purpose performance while significantly outperforming existing baselines. 

---
# BlenderRAG: High-Fidelity 3D Object Generation via Retrieval-Augmented Code Synthesis 

**Authors**: Massimo Rondelli, Francesco Pivi, Maurizio Gabbrielli  

**Link**: [PDF](https://arxiv.org/pdf/2605.00632)  

**Abstract**: Automatic generation of executable Blender code from natural language remains challenging, with state-of-the-art LLMs producing frequent syntactic errors and geometrically inconsistent objects. We present BlenderRAG, a retrieval-augmented generation system that operates on a curated multimodal dataset of 500 expert-validated examples (text, code, image) across 50 object categories. By retrieving semantically similar examples during generation, BlenderRAG improves compilation success rates from 40.8% to 70.0% and semantic normalized alignment from 0.41 to 0.77 (CLIP similarity) across four state-of-the-art LLMs, without requiring fine-tuning or specialized hardware, making it immediately accessible for deployment. The dataset and code will be available at this https URL. 

---
# AdaMeZO: Adam-style Zeroth-Order Optimizer for LLM Fine-tuning Without Maintaining the Moments 

**Authors**: Zhijie Cai, Haolong Chen, Guangxu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2605.00650)  

**Abstract**: Fine-tuning LLMs is necessary for various dedicated downstream tasks, but classic backpropagation-based fine-tuning methods require substantial GPU memory. To this end, a recent work, MeZO, which relies solely on forward passes to fine-tune LLMs, significantly reduces GPU requirements at the cost of slower convergence due to its indifference to loss landscapes. Standard solutions, such as Adam, explore loss landscapes by estimating the first- and second-order moments and storing them in memory to guide the model's movement through dimensions with lower curvature and vice versa. However, directly applying Adam negates MeZO's advantage as it will triple the memory requirement. In light of this, we propose AdaMeZO, a zeroth-order optimizer that leverages Adam-style first- and second-moment estimates without maintaining them in memory. We present a theoretical analysis of AdaMeZO, corroborated by extensive experiments demonstrating AdaMeZO's performance, showing that AdaMeZO can outperform MeZO while requiring up to $70\%$ fewer forward passes. Trajectory visualizations affirm AdaMeZO's ability to adapt to diverse loss landscapes. 

---
# Jailbreaking Vision-Language Models Through the Visual Modality 

**Authors**: Aharon Azulay, Jan Dubiński, Zhuoyun Li, Atharv Mittal, Yossi Gandelsman  

**Link**: [PDF](https://arxiv.org/pdf/2605.00583)  

**Abstract**: The visual modality of vision-language models (VLMs) is an underexplored attack surface for bypassing safety alignment. We introduce four jailbreak attacks exploiting the vision component: (1) encoding harmful instructions as visual symbol sequences with a decoding legend, (2) replacing harmful objects with benign substitutes (e.g., bomb -> banana) then prompting for harmful actions using the substitute term, (3) replacing harmful text in images (e.g., on book covers) with benign words while visual context preserves the original meaning, and (4) visual analogy puzzles whose solution requires inferring a prohibited concept. Evaluating across six frontier VLMs, our visual attacks bypass safety alignment and expose a cross-modality alignment gap: text-based safety training does not automatically generalize to harmful intent conveyed visually. For example, our visual cipher achieves 40.9% attack success on Claude-Haiku-4.5 versus 10.7% for an equivalent textual cipher. To further our insight into the attack mechanism, we present preliminary interpretability and mitigation results. These findings highlight that robust VLM alignment requires treating vision as a first-class target for safety post-training. 

---
# Space Network of Experts: Architecture and Expert Placement 

**Authors**: Zhanwei Wang, Huiling Yang, Min Sheng, Khaled B. Letaief, Kaibin Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.00515)  

**Abstract**: Leveraging continuous solar energy harvesting at high efficiency, space data centers are envisioned as a promising platform for executing energy-intensive large language models (LLMs). Recognizing this advantage, space and AI conglomerates (e.g., SpaceX, Google) are actively investing in this vision. One key challenge, however, is the efficient distributed deployment of a large-scale LLM in a satellite network due to the limited onboard computing and communication resources. This gives rise to a placement problem that involves partitioning and mapping model components to satellites such that the fundamentally different model architecture and network topology can be reconciled to ensure low-latency token generation. To address this problem, we present the Space Network of Experts (Space-XNet) framework targeting the distributed execution of a popular mixture-of-experts (MoE) model in space. The proposed placement strategies are two-level: (1) layer placement, which assigns MoE layers to satellite subnets; and (2) intra-layer expert placement, which assigns individual experts to satellites associated with the same layer/subnet. For layer placement, we exploit the ring-like communication pattern of autoregressive inference to partition the satellite constellation along the orbiting direction into subnets arranged on a ring, each hosting one MoE layer. Based on this architecture, we formulate and solve an optimization problem for intra-layer expert placement to map experts with heterogeneous activation probabilities onto satellites. The derived strategy reveals an intuitive principle: a frequently activated expert should be mapped to a satellite on a routing path with low expected latency. Experiments over a thousand-satellite constellation show that Space-XNet achieves at least a threefold latency reduction compared with conventional random and ablation-based placement strategies. 

---
# SAGA: Workflow-Atomic Scheduling for AI Agent Inference on GPU Clusters 

**Authors**: Dongxin Guo, Jikun Wu, Siu Ming Yiu  

**Link**: [PDF](https://arxiv.org/pdf/2605.00528)  

**Abstract**: AI agents execute tens to hundreds of chained LLM calls per task, yet GPU schedulers treat each call as independent, discarding gigabytes of intermediate state between steps and inflating end-to-end latency by 3-8x. We argue that this request-level abstraction is fundamentally mismatched to compound AI workloads, and propose a shift to program-level scheduling: treating the entire agent workflow (not individual inference calls) as the first-class schedulable unit. We present SAGA, a distributed scheduler that implements this abstraction through three mechanisms: (1) Agent Execution Graphs that capture workflow structure to predict KV cache reuse across tool-call boundaries, achieving within 1.31x of Bélády's optimal offline policy; (2) session-affinity batching with work stealing that co-locates correlated requests while maintaining global load balance; and (3) Agent Fair Share, a task-completion-time fairness metric with provable bounded-deviation guarantees. On a 64-GPU cluster serving SWE-bench coding agents and WebArena browser tasks, SAGA reduces task completion time by 1.64x (geometric mean, p < 0.001) over vLLM v0.15.1 with prefix caching and affinity routing, while improving GPU memory utilization by 1.22x and achieving 99.2% SLO attainment under multi-tenant interference. These latency gains come at a quantified cost: approximately 30% lower peak throughput than throughput-optimal batch scheduling, a tradeoff appropriate for the latency-sensitive interactive deployments that dominate compound AI usage. Our results demonstrate that workflow-aware scheduling is essential for efficient compound AI serving. 

---
# Silicon Showdown: Performance, Efficiency, and Ecosystem Barriers in Consumer-Grade LLM Inference 

**Authors**: Allan Kazakov, Abdurrahman Javat  

**Link**: [PDF](https://arxiv.org/pdf/2605.00519)  

**Abstract**: The operational landscape of local Large Language Model (LLM) inference has shifted from lightweight models to datacenter-class weights exceeding 70B parameters, creating profound systems challenges for consumer hardware. This paper presents a systematic empirical analysis of the Nvidia and Apple Silicon ecosystems, specifically characterizing the distinct intra-architecture trade-offs required to deploy these massive models. On the Nvidia Blackwell architecture, we identify a critical "Backend Dichotomy" within the TensorRT-LLM stack: while the new NVFP4 quantization format delivers a 1.6x throughput advantage over optimized BF16 baselines (151 tokens/s vs. 92 tokens/s), realizing this performance requires navigating complex runtime constraints that trade startup latency for generation speed. Furthermore, we characterize the "VRAM Wall" for 70B+ models: on discrete GPUs, users face a destructive choice between aggressive quantization (e.g., Q2) that degrades model intelligence to fit in VRAM, or PCIe-bottlenecked CPU offloading, which reduces throughput by over 90% compared to full-GPU execution. Conversely, Apple's Unified Memory Architecture (UMA) circumvents these bottlenecks, enabling linear scaling for 80B parameter models at practical 4-bit precisions. This architectural divergence extends to operational sustainability, where Apple's SoC design demonstrates up to a 23x advantage in energy efficiency (tokens/joule). We conclude that for consumer-grade inference, the optimal hardware is defined by a complex interplay between compute density (Nvidia) and memory capacity (Apple), moderated by the significant "ecosystem friction" of proprietary quantization workflows. 

---
# BWLA: Breaking the Barrier of W1AX Post-Training Quantization for LLMs 

**Authors**: Zhixiong Zhao, Zukang Xu, Dawei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.00422)  

**Abstract**: Large language models (LLMs) have driven major progress in NLP, yet their substantial memory and compute demands still hinder practical deployment. Binarization can compress weights to 1 bit, fundamentally lowering compute and bandwidth cost. However, existing methods cannot address activation heavy tails and thus must keep activations in high precision, preventing true end-to-end acceleration. To overcome this limitation, we propose BWLA (Binarized Weights and Low-bit Activations), the first post-training quantization framework that preserves high accuracy while achieving 1-bit weight quantization together with low-bit activations (e.g., 6 bits). The Orthogonal-Kronecker Transformation (OKT) learns an orthogonal mapping via EM minimization, converting unimodal weights into symmetric bimodal forms while suppressing activation tails and incoherence. The Proximal SVD Projection (PSP) then performs lightweight low-rank refinement through proximal SVD projection, further enhancing quantizability with minimal overhead. On Qwen3-32B, BWLA reaches a Wikitext2 perplexity of 11.92 under 6-bit activations (vs. 38 from SOTA), improves five zero-shot tasks by more than 70%, and delivers 3.26 times inference speedup, demonstrating strong potential for real-world LLM compression and acceleration. 

---
# AlphaInventory: Evolving White-Box Inventory Policies via Large Language Models with Deployment Guarantees 

**Authors**: Chenyu Huang, Jianghao Lin, Zhengyang Tang, Bo Jiang, Ruoqing Jiang, Benyou Wang, Lai Wei  

**Link**: [PDF](https://arxiv.org/pdf/2605.00369)  

**Abstract**: We study how large language models can be used to evolve inventory policies in online, non-stationary environments. Our work is motivated by recent advances in LLM-based evolutionary search, such as AlphaEvolve, which demonstrates strong performance for static and highly structured problems such as mathematical discovery, but is not directly suited to online dynamic inventory settings. To this end, we propose AlphaInventory, an end-to-end inventory-policy evolution and inference framework grounded in confidence-interval-based certification. The framework trains a large language model using reinforcement learning, incorporates demand data as well as numerical and textual features beyond demand, and generates white-box inventory policy with statistical safety guarantees for deployment in future periods. We further introduce a unified theoretical interface that connects training, inference, and deployment. This allows us to characterize the probability that the AlphaInventory evolves a statistically safe and improved policy, and to quantify the deployment gap relative to the oracle-safe benchmark. Tested on both synthetic data and real-world retail data, AlphaInventory outperforms classical inventory policies and deep learning based methods. In canonical inventory settings, it evolves new policies that improve upon existing benchmarks. 

---
# Pedagogical Promise and Peril of AI: A Text Mining Analysis of ChatGPT Research Discussions in Programming Education 

**Authors**: Juvy C.Grume, John Paul P. Miranda, Aileen P. De Leon, Jordan L. Salenga, Hilene E. Hernandez, Mark Anthony A. Castro, Vernon Grace M. Maniago, Joel D. Canlas, Joel B. Quiambao  

**Link**: [PDF](https://arxiv.org/pdf/2605.00361)  

**Abstract**: GenAI systems such as ChatGPT are increasingly discussed in programming education, but the ways in which the research literature conceptualizes and frames their role remain unclear. This chapter applies text mining to publications indexed in a leading academic database to map scholarly discourse on ChatGPT in programming education. Term frequency analysis, phrase pattern extraction, and topic modeling reveal four dominant themes: pedagogical implementation, student-centered learning and engagement, AI infrastructure and human-AI collaboration, and assessment, prompting, and model evaluation. The literature prioritizes classroom practice and learner interaction, with comparatively limited attention to assessment design and institutional governance. Across studies, ChatGPT is positioned both as a learning aid that supports explanation, feedback, and efficiency and as a pedagogical risk linked to overreliance, unreliable outputs, and academic integrity concerns. These findings support responsible integration and highlight the need for stronger assessment and governance mechanisms. 

---
# Improving LLM Code Generation via Requirement-Aware Curriculum Reinforcement Learning 

**Authors**: Shouyu Yin, Zhao Tian, Junjie Chen, Shikai Guo  

**Link**: [PDF](https://arxiv.org/pdf/2605.00433)  

**Abstract**: Code generation, which aims to automatically generate source code from given programming requirements, has the potential to substantially improve software development efficiency. With the rapid advancement of large language models (LLMs), LLM-based code generation has attracted widespread attention from both academia and industry. However, as programming requirements become increasingly complex, existing LLMs still exhibit notable performance limitations. To address this challenge, recent studies have proposed training-based curriculum reinforcement learning (CRL) strategies to improve LLM code generation performance. Despite their effectiveness, existing CRL approaches suffer from several limitations, including misaligned requirement difficulty perception, the absence of requirement difficulty optimization, and suboptimal curriculum sampling strategies. In CRL-based code generation, programming requirements serve as the sole input to the model, making their quality and difficulty critical to training effectiveness. Motivated by insights from software requirements engineering, we propose RECRL, a novel requirement-aware curriculum reinforcement learning framework for enhancing LLM-based code generation. RECRL automatically perceives model-specific requirement difficulty, optimizes challenging requirements to improve training data utilization, and employs an adaptive curriculum sampling strategy to construct training batches with smoothly varying difficulty. Extensive experiments on five state-of-the-art LLMs across five widely-used code generation benchmarks by comparing with five state-of-the-art baselines, demonstrate the significant effectiveness of RECRL. For example, RECRL achieves an average Pass@1 improvement of 1.23%-5.62% over all state-of-the-art baselines. 

---
# GaMMA: Towards Joint Global-Temporal Music Understanding in Large Multimodal Models 

**Authors**: Zuyao You, Zhesong Yu, Mingyu Liu, Bilei Zhu, Yuan Wan, Zuxuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.00371)  

**Abstract**: In this paper, we propose GaMMA, a state-of-the-art (SoTA) large multimodal model (LMM) designed to achieve comprehensive musical content understanding. GaMMA inherits the streamlined encoder-decoder design of LLaVA, enabling effective cross-modal learning between music and language. By incorporating audio encoders in a mixture-of-experts manner, GaMMA effectively unifies both time-series and non-time-series music understanding tasks within one set of parameters. Our approach combines carefully curated datasets at scale with a progressive training pipeline, effectively pushing the boundaries of music understanding via pretraining, supervised fine-tuning (SFT), and reinforcement learning (RL). To comprehensively assess both temporal and non-temporal capability of music LMMs, we introduce MusicBench, the largest music-oriented benchmark, comprising 3,739 human-curated multiple-choice questions covering diverse aspects of musical understanding. Extensive experiments demonstrate that GaMMA establishes new SoTA in the music domain, achieving 79.1% accuracy on MuchoMusic, 79.3% on MusicBench-Temporal, and 81.3% on MusicBench-Global, consistently outperforming previous methods. 

---
# Social Bias in LLM-Generated Code: Benchmark and Mitigation 

**Authors**: Fazle Rabbi, Lin Ling, Song Wang, Jinqiu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.00382)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed to generate code for human-centered applications where demographic fairness is critical. However, existing evaluations focus almost exclusively on functional correctness, leaving social bias in LLM-generated code largely unexamined. Extending our prior work on Solar, we conduct a comprehensive empirical study using SocialBias-Bench, a benchmark of 343 real-world coding tasks spanning seven demographic dimensions. We evaluate four prominent LLMs and find severe bias across all models, with Code Bias Scores reaching up to 60.58%. We further show that standard prompt-level interventions, such as Chain-of-Thought reasoning and fairness persona assignment, inadvertently amplify bias rather than reduce it. We then investigate whether structured multi-agent software process frameworks can improve fairness, finding that structured pipelines reduce bias when early roles correctly scope what the code should and should not consider. However, adding explicit fairness instructions to all agent roles produces worse outcomes than providing none, suggesting that diffused responsibility goes unaddressed. To address these limitations, we propose the Fairness Monitor Agent (FMA), a modular component that plugs into any existing code generation pipeline without modifying it. FMA analyzes the task description to determine which attributes should be considered or restricted, then detects and corrects violations through an iterative review process, without requiring an executable test suite. Evaluated on all 343 tasks, FMA reduces bias by 65.1% compared to a developer agent alone and improves functional correctness from 75.80% to 83.97%, outperforming all other studied approaches. 

---
# Skills as Verifiable Artifacts: A Trust Schema and a Biconditional Correctness Criterion for Human-in-the-Loop Agent Runtimes 

**Authors**: Alfredo Metere  

**Link**: [PDF](https://arxiv.org/pdf/2605.00424)  

**Abstract**: Agent skills -- structured packages of instructions, scripts, and references that augment a large language model (LLM) without modifying the model itself -- have moved from convenience to first-class deployment artifact. The runtime that loads them inherits the same problem package managers and operating systems have always faced: a piece of content claims a behavior; the runtime must decide whether to believe it. We argue this paper's central thesis up front: a skill is \emph{untrusted code} until it is verified, and the runtime that loads it must enforce that default rather than infer trust from a signature, a clearance, or a registry of origin. Without skill verification, a human-in-the-loop (HITL) gate must fire on every irreversible call -- which is operationally untenable and degrades into rubber-stamping at any non-trivial scale. With skill verification treated as a separate, gated process, HITL fires only for what is unverified, and the system becomes sustainable. We give a trust schema (§\ref{sec:schema}) that includes an explicit verification level on every skill manifest; a capability gate (§\ref{sec:gate}) whose HITL policy is a function of that verification level; a \emph{biconditional} correctness criterion (§\ref{sec:biconditional}) that any candidate verification procedure must satisfy on an adversarial-ensemble exercise (§\ref{sec:eval}); and a portable runtime profile (§\ref{sec:guidelines}) with ten normative guidelines abstracted from a working open-source reference implementation \cite{metere2026enclawed}. The contribution is harness- and model-agnostic; nothing here requires retraining, fine-tuning, or proprietary infrastructure. 

---
# Caracal: Causal Architecture via Spectral Mixing 

**Authors**: Bingzheng Gan, Tianyi Zhang, Yusu Li, Jing Huang, Wei Shi, Yangkai Ding, Tao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.00292)  

**Abstract**: The scalability of Large Language Models to long sequences is hindered by the quadratic cost of attention and the limitations of positional encodings. To address these, we introduce Caracal, a novel architecture that replaces attention with a parameter-efficient, $\mathcal{O}(L \log L)$ Multi-Head Fourier (MHF) module. Our contributions are threefold: (1) We leverage the Fast Fourier Transform (FFT) for sequence mixing, inherently addressing both bottlenecks mentioned above. (2) We apply a frequency-domain causal masking technique that enforces autoregressive capabilities via asymmetric padding and truncation, overcoming a critical barrier for Fourier-based generative models. (3) Unlike efficient models relying on hardware-specific implementations (e.g., Mamba), we uses standard library operators. This ensures robust portability, eliminating common deployment barriers. Evaluations demonstrate that Caracal performs competitively with Transformer and SSM baselines, offering a scalable and simple pathway for efficient long-sequence modeling. Code is available in Appendix. 

---
# Semia: Auditing Agent Skills via Constraint-Guided Representation Synthesis 

**Authors**: Hongbo Wen, Ying Li, Hanzhi Liu, Chaofan Shou, Yanju Chen, Yuan Tian, Yu Feng  

**Link**: [PDF](https://arxiv.org/pdf/2605.00314)  

**Abstract**: An agent skill is a configuration package that equips an LLM-driven agent with a concrete capability, such as reading email, executing shell commands, or signing blockchain transactions. Each skill is a hybrid artifact-a structured half declares executable interfaces, while a prose half dictates when and how those interfaces fire-and the prose is reinterpreted probabilistically on every invocation. Conventional static analyzers parse the structured half but ignore the prose; LLM-based tools read the prose but cannot reproducibly prove that a tainted input reaches a high-impact sink.
We present Semia, a static auditor for agent skills. Semia lifts each skill into the Skill Description Language (SDL), a Datalog fact base that captures LLM-triggered actions, prose-defined conditions, and human-in-the-loop checkpoints. Synthesizing a fact base that is both structurally sound and semantically faithful to the original prose is the central challenge; we address it with Constraint-Guided Representation Synthesis (CGRS), a propose-verify-evaluate loop that refines LLM candidates until convergence. Security properties (e.g., indirect injection, secret leakage, confused deputies, unguarded sinks, etc.) over an agent skill can then be reduced to Datalog reachability queries. We evaluate Semia on 13,728 real-world skills from public marketplaces. Semia renders all of them auditable and finds that more than half carry at least one critical semantic risk. On a stratified sample of 541 expert-labeled skills, Semia achieves 97.7% recall and an F1 of 90.6%, substantially outperforming signature-based scanners and LLM baselines. 

---
# Jailbroken Frontier Models Retain Their Capabilities 

**Authors**: Daniel Zhu, Zihan Wang, Jenny Bao, Jerry Wei  

**Link**: [PDF](https://arxiv.org/pdf/2605.00267)  

**Abstract**: As language model safeguards become more robust, attackers are pushed toward developing increasingly complex jailbreaks. Prior work has found that this complexity imposes a "jailbreak tax" that degrades the target model's task performance. We show that this tax scales inversely with model capability and that the most advanced jailbreaks effectively yield no reduction in model capabilities. Evaluating 28 jailbreaks on five benchmarks across Claude models ranging in capability from Haiku 4.5 to Opus 4.6, we find Haiku 4.5 loses an average of 33.1% on benchmark performance when jailbroken, while Opus 4.6 at max thinking effort loses only 7.7%. We also observe that across all models, reasoning-heavy tasks display considerably more degradation than knowledge-recall tasks. Finally, Boundary Point Jailbreaking, currently the strongest jailbreak against deployed classifiers, achieves near-perfect classifier evasion with near-zero degradation across safeguarded models. We recommend that safety cases for frontier models should not rely on a meaningful capability degradation from jailbreaks. 

---
# Rethinking Network Topologies for Cost-Effective Mixture-of-Experts LLM Serving 

**Authors**: Junsun Choi, Sam Son, Sunjin Choi, Hansung Kim, Yakun Sophia Shao, Scott Shenker, Sylvia Ratnasamy, Borivoje Nikolic  

**Link**: [PDF](https://arxiv.org/pdf/2605.00254)  

**Abstract**: Mixture-of-experts (MoE) architectures have turned LLM serving into a cluster-scale workload in which communication consumes a considerable portion of LLM serving runtime. This has prompted industry to invest heavily in expensive high-bandwidth scale-up networks. We question whether such costly infrastructure is strictly necessary. We present the first systematic cross-layer analysis of network cost-effectiveness for MoE LLM serving, comparing four representative XPU (e.g., GPU/TPU) topologies (scale-up, scale-out, 3D torus, and 3D full-mesh). We find that lower-cost switchless topologies are more cost-effective than the scale-up topology across all serving scenarios explored, improving cost-effectiveness by 20.6-56.2%. In particular, the 3D full-mesh topology is Pareto-optimal in terms of the performance-cost tradeoff. We also find that current scale-up link bandwidths are over-provisioned: reducing the link bandwidth improves throughput per cost by up to 27%. A forward-looking analysis of upcoming GPU generations indicates that the cost-performance advantage of switchless networks will likely persist. 

---
# The $\textit{Silicon Society}$ Cookbook: Design Space of LLM-based Social Simulations 

**Authors**: Aurélien Bück-Kaeffer, Sneheel Sarangi, Maximilian Puelma Touzel, Reihaneh Rabbany, Zachary Yang, Jean-François Godbout  

**Link**: [PDF](https://arxiv.org/pdf/2605.00197)  

**Abstract**: Studies attempting to simulate human behavior with $\textit{Silicon Societies}$ grow in numbers while LLM-only social networks have started appearing outside of controlled settings. However, the design space of these networks remains under-studied, which contributes to a gap in validating model realism. To enable future works to make more informed design decisions, we perform a systematic analysis of the consequences and interactions of key design choices in simulated social networks, including the choice of base model used to model individual agents, and how they are connected to each other. Using surveys as a proxy for agent opinions, our findings suggest that the geometry of the design space is non-trivial, with some parameters behaving in additive ways while others display more complex interactions. In particular, the choice of the base LLM is the most important variable impacting the simulation outcomes. 

---
# XekRung Technical Report 

**Authors**: Jiutian Zeng, Junjie Li, Chengwei Dai, Jie Liang, Zhaoyu Hu, Yiliang Zhang, Ziang Weng, Longtao Huang, Dongjie Zhang, Libin Dong, Yang Ge, Yuanda Wang, Kaiwen Lv Kacuila, Bingyu Zhu, Jing Wang, Jin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.00072)  

**Abstract**: We present XekRung, a frontier large language model for cybersecurity, designed to provide comprehensive security capabilities. To achieve this, we develop diverse data synthesis pipelines tailored to the cybersecurity domain, enabling the scalable construction of high-quality training data and providing a strong foundation for cybersecurity knowledge and understanding. Building on this foundation, we establish a complete training pipeline spanning continued pre-training (CPT), supervised fine-tuning (SFT), and reinforcement learning (RL) to further extend the model's capabilities. We further introduce a multi-dimensional evaluation system to guide the iterative improvement of both domain-specific and general-purpose abilities. Extensive experiments demonstrate that XekRung achieves state-of-the-art performance on cybersecurity-specific benchmarks among models of the same scale, while maintaining strong performance on general benchmarks. 

---
# CRC-Screen: Certified DNA-Synthesis Hazard Screening Under Taxonomic Shift 

**Authors**: Najmul Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2605.00074)  

**Abstract**: DNA-synthesis providers screen incoming orders by searching the requested sequence against curated hazard lists. We show that this baseline collapses to a 100% false-flag rate when the hazardous sequence comes from a taxonomic family absent from the reference set: under Conformal Risk Control's certified miss-rate constraint, a low-discrimination signal forces the threshold below the entire test-benign mass. We compose three signals derived from a synthesis order's public annotation: $k$-mer Jaccard similarity to known toxins, the trimmed-mean score of a five-LLM judge panel, and cosine similarity to clustered embedding centroids. Fused under a monotone logistic aggregator and calibrated by Conformal Risk Control, the resulting screener certifies $\mathbb{E}[\mathrm{FNR}] \le \alpha$. Across ten leave-one-taxonomic-family-out folds at $\alpha=0.05$ on UniProt KW-0800 reviewed toxins, the calibrated screener achieves 0% test miss rate on every fold and 0% test false-flag rate on nine of ten folds. The bound's finite-sample slack $1/(n_{\mathrm{cal}}+1)$ caps the certifiable miss rate at 1.77% on our 200-hazard subsample; reaching procurement-grade $\alpha=10^{-3}$ requires an $18\times$ larger calibration set, which the full reviewed UniProt KW-0800 corpus is large enough to deliver. The binding constraint on certifiable DNA-synthesis screening is calibration data, not algorithms. Code: this https URL 

---
# Attention Is Where You Attack 

**Authors**: Aviral Srivastava, Sourav Panda  

**Link**: [PDF](https://arxiv.org/pdf/2605.00236)  

**Abstract**: Safety-aligned large language models rely on RLHF and instruction tuning to refuse harmful requests, yet the internal mechanisms implementing safety behavior remain poorly understood. We introduce the Attention Redistribution Attack (ARA), a white-box adversarial attack that identifies safety-critical attention heads and crafts nonsemantic adversarial tokens that redirect attention away from safety-relevant positions. Unlike prior jailbreak methods operating at the semantic or output-logit level, ARA targets the geometry of softmax attention on the probability simplex using Gumbel-softmax optimization over targeted heads. Across LLaMA-3-8B-Instruct, Mistral-7B-Instruct-v0.1, and Gemma-2-9B-it, ARA bypasses safety alignment with as few as 5 tokens and 500 optimization steps, achieving 36% ASR on Mistral-7B and 30% on LLaMA-3 against 200 HarmBench prompts, while Gemma-2 remains at 1%. Our principal mechanistic finding is a dissociation between ablation and redistribution: zeroing out the top-ranked safety heads produces at most 1 flip among 39 to 50 baseline refusals, while ARA targeting the corresponding safety-heavy layers flips 72/200 prompts on Mistral-7B and 60/200 on LLaMA-3. This suggests that safety is not localized in these heads as removable components, but emerges from the attention routing they perform. Removing a head allows compensation through the residual stream, while redirecting its attention propagates a corrupted signal downstream. 

---
# SiriusHelper: An LLM Agent-Based Operations Assistant for Big Data Platforms 

**Authors**: Yu Shen, Shiyang Liu, Qihang He, Yihang Cheng, Haining Xie, Zhiming He, Huahua Fan, Xianzhi Tan, Teng Ma, Shaoquan Zhang, Danqing Huang, Fan Jiang, Yang Li, Chongqing Zhao, Peng Chen, Jie Jiang, Bin Cui  

**Link**: [PDF](https://arxiv.org/pdf/2605.00043)  

**Abstract**: Big data platforms are widely used in modern enterprises, and an in-production intelligent assistant is increasingly important to help users quickly find actionable guidance and reduce operational burden. While recent LLM+RAG assistants provide a natural interface, they face practical challenges in real deployments: limited scenario coverage across both general consultation and domain-specific troubleshooting workflows, inefficient knowledge access due to inadequate multi-hop retrieval and flat knowledge organization, and high maintenance cost because escalated tickets are unstructured and hard to convert into assistant improvements and reusable SOPs.
In this paper, we present SiriusHelper, a deployed intelligent assistant for big data platforms. SiriusHelper serves as a unified online assistant that automatically identifies user intent and routes queries to the right handling path, including dedicated expert workflows for specialized scenarios (e.g., SQL execution diagnosis). To support complex troubleshooting, SiriusHelper combines a DeepSearch-driven mechanism with a priority-based hierarchical knowledge base to enable multi-hop retrieval without context overload, thus improving answer reliability and latency. To reduce expert overhead, SiriusHelper further introduces automated ticket understanding and SOP distillation: it diagnoses the assistant failure reason (e.g., missing knowledge or wrong routing) and extracts domain-specific SOPs to continuously enrich the knowledge base. Experiments and online deployment on Tencent Big Data platform show that SiriusHelper outperforms representative alternatives and reduces online ticket volume by 20.8\%. 

---
# Ambient Persuasion in a Deployed AI Agent: Unauthorized Escalation Following Routine Non-Adversarial Content Exposure 

**Authors**: Diego F. Cuadros, Abdoul-Aziz Maiga  

**Link**: [PDF](https://arxiv.org/pdf/2605.00055)  

**Abstract**: We report a safety incident in a deployed multi-agent research system in which a primary AI agent installed 107 unauthorized software components, overwrote a system registry, overrode a prior negative decision from an oversight agent, and escalated through increasingly privileged operations up to an attempted system administrator command. The incident was preceded not by an adversarial attack but by routine content: a forwarded technology article written for human developers and shared by the principal investigator for discussion. The agent operated in a permissive environment, with unrestricted shell access, soft behavioral guidelines containing genuinely conflicting instructions, and no machine-enforced installation policy, and had recommended installing the same tool six hours earlier before being told to stand down. We analyze the behavioral cascade, the control boundaries that failed, and the limitations of multi-agent oversight in detecting and remediating the damage. We use directive weighting error as a descriptive interpretation of the observed failure and ambient persuasion as a provisional analytic label for the broader trigger configuration of non-adversarial environmental content preceding unauthorized agent action. The case highlights ethical and governance implications for deployed agent systems: ambiguous conversational cues are insufficient authorization for consequential actions, prior refusals must persist as enforceable constraints rather than message-level reminders, and oversight mechanisms require systematic post-incident auditing in addition to routine monitoring. 

---
# Models Recall What They Violate: Constraint Adherence in Multi-Turn LLM Ideation 

**Authors**: Garvin Kruthof  

**Link**: [PDF](https://arxiv.org/pdf/2604.28031)  

**Abstract**: When researchers iteratively refine ideas with large language models, do the models preserve fidelity to the original objective? We introduce DriftBench, a benchmark for evaluating constraint adherence in multi-turn LLM-assisted scientific ideation. Across 2,146 scored benchmark runs spanning seven models from five providers (including two open-weight), four interaction conditions, and 38 research briefs from 24 scientific domains, we find that iterative pressure reliably increases structural complexity and often reduces adherence to original constraints. A restatement probe reveals a dissociation between declarative recall and behavioral adherence, as models accurately restate constraints they simultaneously violate. The knows-but-violates (KBV) rate, measuring constraint non-compliance despite preserved recall, ranges from 8% to 99% across models. Structured checkpointing partially reduces KBV rates but does not close the dissociation, and complexity inflation persists. Human validation against blind raters confirms that the LLM judge under-detects constraint violations, making reported constraint adherence scores conservative. Sensitivity analyses confirm the findings are robust to temperature (0.7 vs.\ 1.0) and pressure type (novelty vs.\ rigor). We release all briefs, prompts, rubrics, transcripts, and scores as an open benchmark. 

---
