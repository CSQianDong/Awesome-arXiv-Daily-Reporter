# Reasoning over Semantic IDs Enhances Generative Recommendation 

**Authors**: Yingzhi He, Yan Sun, Junfei Tan, Yuxin Chen, Xiaoyu Kong, Chunxu Shen, Xiang Wang, An Zhang, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2603.23183)  

**Abstract**: Recent advances in generative recommendation have leveraged pretrained LLMs by formulating sequential recommendation as autoregressive generation over a unified token space comprising language tokens and itemic identifiers, where each item is represented by a compact sequence of discrete tokens, namely Semantic IDs (SIDs). This SID-based formulation enables efficient decoding over large-scale item corpora and provides a natural interface for LLM-based recommenders to leverage rich world knowledge. Meanwhile, breakthroughs in LLM reasoning motivate reasoning-enhanced recommendation, yet effective reasoning over SIDs remains underexplored and challenging. Itemic tokens are not natively meaningful to LLMs; moreover, recommendation-oriented SID reasoning is hard to evaluate, making high-quality supervision scarce.
To address these challenges, we propose SIDReasoner, a two-stage framework that elicits reasoning over SIDs by strengthening SID--language alignment to unlock transferable LLM reasoning, rather than relying on large amounts of recommendation-specific reasoning traces. Concretely, SIDReasoner first enhances SID-language alignment via multi-task training on an enriched SID-centered corpus synthesized by a stronger teacher model, grounding itemic tokens in diverse semantic and behavioral contexts. Building on this enhanced alignment, SIDReasoner further improves recommendation reasoning through outcome-driven reinforced optimization, which guides the model toward effective reasoning trajectories without requiring explicit reasoning annotations. Extensive experiments on three real-world datasets demonstrate the effectiveness of our reasoning-augmented SID-based generative recommendation. Beyond accuracy, the results highlight the broader potential of large reasoning models for generative recommendation, including improved interpretability and cross-domain generalization. 

---
# From Questions to Trust Reports: A LLM-IR Framework for the TREC 2025 DRAGUN Track 

**Authors**: Ignacy Alwasiak, Kene Nnolim, Jaclyn Thi, Samy Ateia, Markus Bink, Gregor Donabauer, David Elsweiler, Udo Kruschwitz  

**Link**: [PDF](https://arxiv.org/pdf/2603.23125)  

**Abstract**: The DRAGUN Track at TREC 2025 targets the growing need for effective support tools that help users evaluate the trustworthiness of online news. We describe the UR_Trecking system submitted for both Task 1 (critical question generation) and Task 2 (retrieval-augmented trustworthiness reporting). Our approach combines LLM-based question generation with semantic filtering, diversity enforcement using clustering, and several query expansion strategies (including reasoning-based Chain-of-Thought expansion) to retrieve relevant evidence from the MS MARCO V2.1 segmented corpus. Retrieved documents are re-ranked using a monoT5 model and filtered using an LLM relevance judge together with a domain-level trustworthiness dataset. For Task 2, selected evidence is synthesized by an LLM into concise trustworthiness reports with citations. Results from the official evaluation indicate that Chain-of-Thought query expansion and re-ranking substantially improve both relevance and domain trust compared to baseline retrieval, while question-generation performance shows moderate quality with room for improvement. We conclude by outlining key challenges encountered and suggesting directions for enhancing robustness and trustworthiness assessment in future iterations of the system. 

---
# GateSID: Adaptive Gating for Semantic-Collaborative Alignment in Cold-Start Recommendation 

**Authors**: Hai Zhu, Yantao Yu, Lei Shen, Bing Wang, Xiaoyi Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2603.22916)  

**Abstract**: In cold-start scenarios, the scarcity of collaborative signals for new items exacerbates the Matthew effect, which undermines platform diversity and remains a persistent challenge in real-world recommender systems. Existing methods typically enhance collaborative signals with semantic information, but they often suffer from a collaborative-semantic tradeoff: collaborative signals are effective for popular items but unreliable for cold-start items, whereas over-reliance on semantic information may obscure meaningful collaborative differences. To address this issue, we propose GateSID, a framework that uses an adaptive gating network to dynamically balance semantic and collaborative signals according to item maturity. Specifically, we first discretize multimodal features into hierarchical Semantic IDs using Residual Quantized VAE. Building on this representation, we design two key components: (1) Gating-Fused Shared Attention, which fuses intra-modal attention distributions with item-level gating weights derived from embeddings and statistical features; and (2) Gate-Regulated Contrastive Alignment, which adaptively calibrates cross-modal alignment, enforcing stronger semantic-behavior consistency for cold-start items while relaxing the constraint for popular items to preserve reliable collaborative signals. Extensive offline experiments on large-scale industrial datasets demonstrate that GateSID consistently outperforms strong baselines. Online A/B tests further confirm its practical value, yielding +2.6% GMV, +1.1% CTR, and +1.6% orders with less than 5 ms additional latency. 

---
# KARMA: Knowledge-Action Regularized Multimodal Alignment for Personalized Search at Taobao 

**Authors**: Zhi Sun, Wenming Zhang, Yi Wei, Liren Yu, Zhixuan Zhang, Dan Ou, Haihong Tang  

**Link**: [PDF](https://arxiv.org/pdf/2603.22779)  

**Abstract**: Large Language Models (LLMs) are equipped with profound semantic knowledge, making them a natural choice for injecting semantic generalization into personalized search systems. However, in practice we find that directly fine-tuning LLMs on industrial personalized tasks (e.g. next item prediction) often yields suboptimal results. We attribute this bottleneck to a critical Knowledge--Action Gap: the inherent conflict between preserving pre-trained semantic knowledge and aligning with specific personalized actions by discriminative objectives. Empirically, action-only training objectives induce Semantic Collapse, such as attention ``sinks''. This degradation severely cripples the LLM's generalization, failing to bring improvements to personalized search systems.
We propose KARMA (Knowledge--Action Regularized Multimodal Alignment), a unified framework that treats semantic reconstruction as a train-only regularizer. KARMA optimizes a next-interest embedding for retrieval (Action) while enforcing semantic decodability (Knowledge) through two complementary objectives: (i) history-conditioned semantic generation, which anchors optimization to the LLM's native next-token distribution, and (ii) embedding-conditioned semantic reconstruction, which constrains the interest embedding to remain semantically recoverable.
On Taobao search system, KARMA mitigates semantic collapse (attention-sink analysis) and improves both action metrics and semantic fidelity. In ablations, semantic decodability yields up to +22.5 HR@200. With KARMA, we achieve +0.25 CTR AUC in ranking, +1.86 HR in pre-ranking and +2.51 HR in recalling. Deployed online with low inference overhead at ranking stage, KARMA drives +0.5% increase in Item Click. 

---
# Leveraging Large Language Models to Extract and Translate Medical Information in Doctors' Notes for Health Records and Diagnostic Billing Codes 

**Authors**: Peter Hartnett, Chung-Chi Huang, Sarah Hartnett, David Hartnett  

**Link**: [PDF](https://arxiv.org/pdf/2603.22625)  

**Abstract**: Physician burnout in the United States has reached critical levels, driven in part by the administrative burden of Electronic Health Record (EHR) documentation and complex diagnostic codes. To relieve this strain and maintain strict patient privacy, this thesis explores an on-device, offline automatic medical coding system. The work focuses on using open-weight Large Language Models (LLMs) to extract clinical information from physician notes and translate it into ICD-10-CM diagnostic codes without reliance on cloud-based services.
A privacy-focused pipeline was developed using Ollama, LangChain, and containerized environments to evaluate multiple open-weight models, including Llama 3.2, Mistral, Phi, and DeepSeek, on consumer-grade hardware. Model performance was assessed for zero-shot, few-shot, and retrieval-augmented generation (RAG) prompting strategies using a novel benchmark of synthetic medical notes.
Results show that strict JSON schema enforcement achieved near 100% formatting compliance, but accurate generation of specific diagnostic codes remains challenging for smaller local models (7B-20B parameters). Contrary to common prompt-engineering guidance, few-shot prompting degraded performance through overfitting and hallucinations. While RAG enabled limited discovery of unseen codes, it frequently saturated context windows, reducing overall accuracy. The findings suggest that fully automated unsupervised coding with local open-source models is not yet reliable; instead, a human-in-the-loop assisted coding approach is currently the most practical path forward. This work contributes a reproducible local LLM architecture and benchmark dataset for privacy-preserving medical information extraction and coding. 

---
# flexvec: SQL Vector Retrieval with Programmatic Embedding Modulation 

**Authors**: Damian Delmas  

**Link**: [PDF](https://arxiv.org/pdf/2603.22587)  

**Abstract**: As AI agents become the primary consumers of retrieval APIs, there is an opportunity to expose more of the retrieval pipeline to the caller. flexvec is a retrieval kernel that exposes the embedding matrix and score array as a programmable surface, allowing arithmetic operations on both before selection. We refer to composing operations on this surface at query time as Programmatic Embedding Modulation (PEM). This paper describes a set of such operations and integrates them into a SQL interface via a query materializer that facilitates composable query primitives. On a production corpus of 240,000 chunks, three composed modulations execute in 19 ms end-to-end on a desktop CPU without approximate indexing. At one million chunks, the same operations execute in 82 ms. 

---
# GraphRAG for Engineering Diagrams: ChatP&ID Enables LLM Interaction with P&IDs 

**Authors**: Achmad Anggawirya Alimin, Artur M. Schweidtmann  

**Link**: [PDF](https://arxiv.org/pdf/2603.22528)  

**Abstract**: Large Language Models (LLMs) combined with Retrieval-Augmented Generation (RAG) and knowledge graphs offer new opportunities for interacting with engineering diagrams such as Piping and Instrumentation Diagrams (P&IDs). However, directly processing raw images or smart P&ID files with LLMs is often costly, inefficient, and prone to hallucinations. This work introduces ChatP&ID, an agentic framework that enables grounded and cost-effective natural-language interaction with P&IDs using Graph Retrieval-Augmented Generation (GraphRAG), a paradigm we refer to as GraphRAG for engineering diagrams. Smart P&IDs encoded in the DEXPI standard are transformed into structured knowledge graphs, which serve as the basis for graph-based retrieval and reasoning by LLM agents. This approach enables reliable querying of engineering diagrams while significantly reducing computational cost. Benchmarking across commercial LLM APIs (OpenAI, Anthropic) demonstrates that graph-based representations improve accuracy by 18% over raw image inputs and reduce token costs by 85% compared to directly ingesting smart P&ID files. While small open-source models still struggle to interpret knowledge graph formats and structured engineering data, integrating them with VectorRAG and PathRAG improves response accuracy by up to 40%. Notably, GPT-5-mini combined with ContextRAG achieves 91% accuracy at a cost of only $0.004 per task. The resulting ChatP&ID interface enables intuitive natural-language interaction with complex engineering diagrams and lays the groundwork for AI-assisted process engineering tasks such as Hazard and Operability Studies (HAZOP) and multi-agent analysis. 

---
# A Brief Comparison of Training-Free Multi-Vector Sequence Compression Methods 

**Authors**: Rohan Jha, Chunsheng Zuo, Reno Kriz, Benjamin Van Durme  

**Link**: [PDF](https://arxiv.org/pdf/2603.22434)  

**Abstract**: While multi-vector retrieval models outperform single-vector models of comparable size in retrieval quality, their practicality is limited by substantially larger index sizes, driven by the additional sequence-length dimension in their document embeddings. Because document embedding size dictates both memory overhead and query latency, compression is essential for deployment. In this work, we present an evaluation of training-free methods targeting the token sequence length, a dimension unique to multi-vector retrieval. Our findings suggest that token merging is strictly superior to token pruning for reducing index size while maintaining retrieval effectiveness. 

---
# AI Co-Scientist for Ranking: Discovering Novel Search Ranking Models alongside LLM-based AI Agents with Cloud Computing Access 

**Authors**: Liwei Wu, Cho-Jui Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2603.22376)  

**Abstract**: Recent advances in AI agents for software engineering and scientific discovery have demonstrated remarkable capabilities, yet their application to developing novel ranking models in commercial search engines remains unexplored. In this paper, we present an AI Co-Scientist framework that automates the full search ranking research pipeline: from idea generation to code implementation and GPU training job scheduling with expert in the loop. Our approach strategically employs single-LLM agents for routine tasks while leveraging multi-LLM consensus agents (GPT 5.2, Gemini Pro 3, and Claude Opus 4.5) for challenging phases such as results analysis and idea generation. To our knowledge, this is the first study in the ranking community to utilize an AI Co-Scientist framework for algorithmic research. We demonstrate that this framework discovered a novel technique for handling sequence features, with all model enhancements produced automatically, yielding substantial offline performance improvements. Our findings suggest that AI systems can discover ranking architectures comparable to those developed by human experts while significantly reducing routine research workloads. 

---
# Reasoner-Executor-Synthesizer: Scalable Agentic Architecture with Static O(1) Context Window 

**Authors**: Ivan Dobrovolskyi  

**Link**: [PDF](https://arxiv.org/pdf/2603.22367)  

**Abstract**: Large Language Models (LLMs) deployed as autonomous agents commonly use Retrieval-Augmented Generation (RAG), feeding retrieved documents into the context window, which creates two problems: the risk of hallucination grows with context length, and token cost scales linearly with dataset size. We propose the Reasoner-Executor-Synthesizer (RES) architecture, a three-layer design that strictly separates intent parsing (Reasoner), deterministic data retrieval and aggregation (Executor), and narrative generation (Synthesizer). The Executor uses zero LLM tokens and passes only fixed-size statistical summaries to the Synthesizer. We formally prove that RES achieves O(1) token complexity with respect to dataset size, and validate this on ScholarSearch, a scholarly research assistant backed by the Crossref API (130M+ articles). Across 100 benchmark runs, RES achieves a mean token cost of 1,574 tokens regardless of whether the dataset contains 42,000 or 16.3 million articles. The architecture eliminates data hallucination by construction: the LLM never sees raw records. KEYWORDS LLM agents; agentic architecture; hallucination elimination; token optimization; context window; retrieval-augmented generation; deterministic execution; scholarly metadata; Crossref API; O(1) complexity. 

---
# Personalized Federated Sequential Recommender 

**Authors**: Yicheng Di  

**Link**: [PDF](https://arxiv.org/pdf/2603.22349)  

**Abstract**: In the domain of consumer electronics, personalized sequential recommendation has emerged as a central task. Current methodologies in this field are largely centered on modeling user behavior and have achieved notable performance. Nevertheless, the inherent quadratic computational complexity typical of most existing approaches often leads to inefficiencies that hinder real-time recommendation. Moreover, these methods face challenges in being effectively adapted to the personalized requirements of users across diverse scenarios. To tackle these issues, we propose the Personalized Federated Sequential Recommender (PFSR). In this framework, an Associative Mamba Block is introduced to capture user profiles from a global perspective while improving prediction efficiency. In addition, a Variable Response Mechanism is developed to enable fine-tuning of parameters in accordance with individual user needs. A Dynamic Magnitude Loss is further devised to preserve greater amounts of localized personalized information throughout the training process. 

---
# Errors in AI-Assisted Retrieval of Medical Literature: A Comparative Study 

**Authors**: Jenny Gao, Yongfeng Zhang, Mary L Disis, Lanjing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.22344)  

**Abstract**: Large language models (LLMs) assisted literature retrieval may lead to erroneous references, but these errors have not been rigorously quantified. Therefore, we quantitatively assess errors in reference retrieval of widely used free-version LLM platforms and identify the factors associated with retrieval errors. We evaluated 2,000 references retrieved by 5 LLMs (Grok-2, ChatGPT GPT-4.1, Google Gemini Flash 2.5, Perplexity AI, and DeepSeek GPT-4) for 40 randomly-selected original articles (10 per journal) published Jan. 2024 to July 2025 from British Medical Journal (BMJ), Journal of the American Medical Association, and The New England Journal of Medicine (NEJM). Primary outcomes were a multimetric score ratio combining validity of digital object identifier, PubMed ID, Google-Scholar link, and relevance; and complete miss rate (proportion of references failing all applicable metrics). Multivariable regression was used to examine independent associations. LLM platforms completely failed to retrieve correct reference data 47.8% of the time. The average score ratio of the 5 LLM platforms was 0.29 (standard deviation, 0.35; range, 0-1.25), with a higher score ratio indicating a higher accuracy in retrieving relevant references and correct bibliographic data. The highest and lowest accuracies were achieved by Grok (0.57) and Genimi (0.11), respectively. Compared with BMJ, NEJM articles had lower score ratios and higher complete miss rates. Multivariable analysis shows LLM platforms and journals were independently associated with score ratios and complete miss rate, respectively. We show modest overall performance of LLMs and significant variability in retrieval accuracy across platforms and journals. LLM platforms and journals are associated with LLM's performance in retrieving medical literature. Bibliographic data should be carefully reviewed when using LLM-assisted literature retrieval. 

---
# Graphs RAG at Scale: Beyond Retrieval-Augmented Generation With Labeled Property Graphs and Resource Description Framework for Complex and Unknown Search Spaces 

**Authors**: Manie Tadayon, Mayank Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2603.22340)  

**Abstract**: Recent advances in Retrieval-Augmented Generation (RAG) have revolutionized knowledge-intensive tasks, yet traditional RAG methods struggle when the search space is unknown or when documents are semi-structured or structured. We introduce a novel end-to-end Graph RAG framework that leverages both Labeled Property Graph (LPG) and Resource Description Framework (RDF) architectures to overcome these limitations. Our approach enables dynamic document retrieval without the need to pre-specify the number of documents and eliminates inefficient reranking. We propose an innovative method for converting documents into RDF triplets using JSON key-value pairs, facilitating seamless integration of semi-structured data. Additionally, we present a text to Cypher framework for LPG, achieving over 90% accuracy in real-time translation of text queries to Cypher, enabling fast and reliable query generation suitable for online applications. Our empirical evaluation demonstrates that Graph RAG significantly outperforms traditional embedding-based RAG in accuracy, response quality, and reasoning, especially for complex, semi-structured tasks. These findings establish Graph RAG as a transformative solution for next-generation retrieval-augmented systems. 

---
# Causal Direct Preference Optimization for Distributionally Robust Generative Recommendation 

**Authors**: Chu Zhao, Enneng Yang, Jianzhe Zhao, Guibing Guo  

**Link**: [PDF](https://arxiv.org/pdf/2603.22335)  

**Abstract**: Direct Preference Optimization (DPO) guides large language models (LLMs) to generate recommendations aligned with user historical behavior distributions by minimizing preference alignment loss. However, our systematic empirical research and theoretical analysis reveal that DPO tends to amplify spurious correlations caused by environmental confounders during the alignment process, significantly undermining the generalization capability of LLM-based generative recommendation methods in out of distribution (OOD) scenarios. To mitigate this issue, we propose CausalDPO, an extension of DPO that incorporates a causal invariance learning mechanism. This method introduces a backdoor adjustment strategy during the preference alignment phase to eliminate interference from environmental confounders, explicitly models the latent environmental distribution using a soft clustering approach, and enhances robust consistency across diverse environments through invariance constraints. Theoretical analysis demonstrates that CausalDPO can effectively capture users stable preference structures across multiple environments, thereby improving the OOD generalization performance of LLM-based recommendation models. We conduct extensive experiments under four representative distribution shift settings to validate the effectiveness of CausalDPO, achieving an average performance improvement of 17.17% across four evaluation metrics. 

---
# AgentSLR: Automating Systematic Literature Reviews in Epidemiology with Agentic AI 

**Authors**: Shreyansh Padarha, Ryan Othniel Kearns, Tristan Naidoo, Lingyi Yang, Łukasz Borchmann, Piotr BŁaszczyk, Christian Morgenstern, Ruth McCabe, Sangeeta Bhatia, Philip H. Torr, Jakob Foerster, Scott A. Hale, Thomas Rawson, Anne Cori, Elizaveta Semenova, Adam Mahdi  

**Link**: [PDF](https://arxiv.org/pdf/2603.22327)  

**Abstract**: Systematic literature reviews are essential for synthesizing scientific evidence but are costly, difficult to scale and time-intensive, creating bottlenecks for evidence-based policy. We study whether large language models can automate the complete systematic review workflow, from article retrieval, article screening, data extraction to report synthesis. Applied to epidemiological reviews of nine WHO-designated priority pathogens and validated against expert-curated ground truth, our open-source agentic pipeline (AgentSLR) achieves performance comparable to human researchers while reducing review time from approximately 7 weeks to 20 hours (a 58x speed-up). Our comparison of five frontier models reveals that performance on SLR is driven less by model size or inference cost than by each model's distinctive capabilities. Through human-in-the-loop validation, we identify key failure modes. Our results demonstrate that agentic AI can substantially accelerate scientific evidence synthesis in specialised domains. 

---
# DALDALL: Data Augmentation for Lexical and Semantic Diverse in Legal Domain by leveraging LLM-Persona 

**Authors**: Janghyeok Choi, Jaewon Lee, Sungzoon Cho  

**Link**: [PDF](https://arxiv.org/pdf/2603.22765)  

**Abstract**: Data scarcity remains a persistent challenge in low-resource domains. While existing data augmentation methods leverage the generative capabilities of large language models (LLMs) to produce large volumes of synthetic data, these approaches often prioritize quantity over quality and lack domain-specific strategies. In this work, we introduce DALDALL, a persona-based data augmentation framework tailored for legal information retrieval (IR). Our method employs domain-specific professional personas--such as attorneys, prosecutors, and judges--to generate synthetic queries that exhibit substantially greater lexical and semantic diversity than vanilla prompting approaches. Experiments on the CLERC and COLIEE benchmarks demonstrate that persona-based augmentation achieves improvement in lexical diversity as measured by Self-BLEU scores, while preserving semantic fidelity to the original queries. Furthermore, dense retrievers fine-tuned on persona-augmented data consistently achieve competitive or superior recall performance compared to those trained on original data or generic augmentations. These findings establish persona-based prompting as an effective strategy for generating high-quality training data in specialized, low-resource domains. 

---
# Graph-Aware Late Chunking for Retrieval-Augmented Generation in Biomedical Literature 

**Authors**: Pouria Mortezaagha, Arya Rahgozar  

**Link**: [PDF](https://arxiv.org/pdf/2603.22633)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems for biomedical literature are typically evaluated using ranking metrics like Mean Reciprocal Rank (MRR), which measure how well the system identifies the single most relevant chunk. We argue that for full-text scientific documents, this paradigm is incomplete: it rewards retrieval precision while ignoring retrieval breadth -- the ability to surface evidence from across a document's structural sections. We propose GraLC-RAG, a framework that unifies late chunking with graph-aware structural intelligence, introducing structure-aware chunk boundary detection, UMLS knowledge graph infusion, and graph-guided hybrid retrieval. We evaluate six strategies on 2,359 IMRaD-filtered PubMed Central articles using 2,033 cross-section questions and two metric families: standard ranking metrics (MRR, Recall@k) and structural coverage metrics (SecCov@k, CS Recall). Our results expose a sharp divergence: content-similarity methods achieve the highest MRR (0.517) but always retrieve from a single section, while structure-aware methods retrieve from up to 15.6x more sections. Generation experiments show that KG-infused retrieval narrows the answer-quality gap to delta-F1 = 0.009 while maintaining 4.6x section diversity. These findings demonstrate that standard metrics systematically undervalue structural retrieval and that closing the multi-section synthesis gap is a key open problem for biomedical RAG. 

---
# Do Large Language Models Reduce Research Novelty? Evidence from Information Systems Journals 

**Authors**: Ali Safari  

**Link**: [PDF](https://arxiv.org/pdf/2603.22510)  

**Abstract**: Large language models such as ChatGPT have increased scholarly output, but whether this productivity boost produces genuine intellectual advancement remains untested. I address this gap by measuring the semantic novelty of 13,847 articles published between 2020 and 2025 in 44 Information Systems journals. Using SPECTER2 embeddings, I operationalize novelty as the cosine distance between each paper and its nearest prior neighbors. A difference-in-differences design with the November 2022 release of ChatGPT as the treatment break reveals a heterogeneous pattern: authors affiliated with institutions in non-English-dominant countries show a 0.18 standard deviation decline in relative novelty compared to authors in English-dominant countries (beta = -0.176, p < 0.001), equivalent to a 7-percentile-point drop in the novelty distribution. This finding is robust across alternative novelty specifications, treatment break dates, and sub-samples, and survives a placebo test at a pre-treatment break. I interpret these results through the lens of construal level theory, proposing that LLMs function as proximity tools that shift researchers from abstract, exploratory thinking toward concrete, convention-following execution. The paper contributes to the growing debate on whether LLM-driven productivity gains come at the cost of intellectual diversity. 

---
# Less is More: Adapting Text Embeddings for Low-Resource Languages with Small Scale Noisy Synthetic Data 

**Authors**: Zaruhi Navasardyan, Spartak Bughdaryan, Bagrat Minasyan, Hrant Davtyan  

**Link**: [PDF](https://arxiv.org/pdf/2603.22290)  

**Abstract**: Low-resource languages (LRLs) often lack high-quality, large-scale datasets for training effective text embedding models, hindering their application in tasks like retrieval-augmented generation (RAG) and semantic search. In this work, we challenge the prevailing assumption that effective semantic alignment requires massive datasets or pristine, human-verified translations. Focusing on Armenian (an LRL with a unique script), we introduce a cost-effective adaptation strategy using small scale noisy synthetic data generated by translating English Reddit title-body pairs with open-weights models. We establish a comprehensive evaluation benchmark comprising existing datasets, translated data, and a manually curated dataset. Our experiments reveal a surprising "Less is More" phenomenon: fine-tuning a multilingual encoder (mE5) on just 10,000 noisy synthetic pairs yields 11-12\% average improvements across the benchmark with a 20\%+ relative improvement in retrieval performance, matching the performance of models trained on ~1 million examples. Furthermore, we demonstrate that neither increasing data scale, improving translation quality via state-of-the-art LLMs, nor diversifying data domains yields significant gains over this minimal baseline. We validate the generalizability of these findings on another LRL with a unique script. Our results suggest that semantic alignment for LRLs saturates early and is highly robust to noise, democratizing high-performance embedding creation for resource-constrained communities. We release the model, data, and the benchmark at this https URL to facilitate further research. 

---
