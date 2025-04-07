# Learning Sparse Disentangled Representations for Multimodal Exclusion Retrieval 

**Authors**: Prachi, Sumit Bhatia, Srikanta Bedathur  

**Link**: [PDF](https://arxiv.org/pdf/2504.03184)  

**Abstract**: Multimodal representations are essential for cross-modal retrieval, but they often lack interpretability, making it difficult to understand the reasoning behind retrieved results. Sparse disentangled representations offer a promising solution; however, existing methods rely heavily on text tokens, resulting in high-dimensional embeddings. In this work, we propose a novel approach that generates compact, fixed-size embeddings that maintain disentanglement while providing greater control over retrieval tasks. We evaluate our method on challenging exclusion queries using the MSCOCO and Conceptual Captions benchmarks, demonstrating notable improvements over dense models like CLIP, BLIP, and VISTA (with gains of up to 11% in AP@10), as well as over sparse disentangled models like VDR (achieving up to 21% gains in AP@10). Furthermore, we present qualitative results that emphasize the enhanced interpretability of our disentangled representations. 

---
# Exploiting Fine-Grained Skip Behaviors for Micro-Video Recommendation 

**Authors**: Sanghyuck Lee, Sangkeun Park, Jaesung Lee  

**Link**: [PDF](https://arxiv.org/pdf/2504.03107)  

**Abstract**: The growing trend of sharing short videos on social media platforms, where users capture and share moments from their daily lives, has led to an increase in research efforts focused on micro-video recommendations. However, conventional methods oversimplify the modeling of skip behavior, categorizing interactions solely as positive or negative based on whether skipping occurs. This study was motivated by the importance of the first few seconds of micro-videos, leading to a refinement of signals into three distinct categories: highly positive, less positive, and negative. Specifically, we classify skip interactions occurring within a short time as negatives, while those occurring after a delay are categorized as less positive. The proposed dual-level graph and hierarchical ranking loss are designed to effectively learn these fine-grained interactions. Our experiments demonstrated that the proposed method outperformed three conventional methods across eight evaluation measures on two public datasets. 

---
# EnrichIndex: Using LLMs to Enrich Retrieval Indices Offline 

**Authors**: Peter Baile Chen, Tomer Wolfson, Michael Cafarella, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2504.03598)  

**Abstract**: Existing information retrieval systems excel in cases where the language of target documents closely matches that of the user query. However, real-world retrieval systems are often required to implicitly reason whether a document is relevant. For example, when retrieving technical texts or tables, their relevance to the user query may be implied through a particular jargon or structure, rather than explicitly expressed in their content. Large language models (LLMs) hold great potential in identifying such implied relevance by leveraging their reasoning skills. Nevertheless, current LLM-augmented retrieval is hindered by high latency and computation cost, as the LLM typically computes the query-document relevance online, for every query anew. To tackle this issue we introduce EnrichIndex, a retrieval approach which instead uses the LLM offline to build semantically-enriched retrieval indices, by performing a single pass over all documents in the retrieval corpus once during ingestion time. Furthermore, the semantically-enriched indices can complement existing online retrieval approaches, boosting the performance of LLM re-rankers. We evaluated EnrichIndex on five retrieval tasks, involving passages and tables, and found that it outperforms strong online LLM-based retrieval systems, with an average improvement of 11.7 points in recall @ 10 and 10.6 points in NDCG @ 10 compared to strong baselines. In terms of online calls to the LLM, it processes 293.3 times fewer tokens which greatly reduces the online latency and cost. Overall, EnrichIndex is an effective way to build better retrieval indices offline by leveraging the strong reasoning skills of LLMs. 

---
# AutoSSVH: Exploring Automated Frame Sampling for Efficient Self-Supervised Video Hashing 

**Authors**: Niu Lian, Jun Li, Jinpeng Wang, Ruisheng Luo, Yaowei Wang, Shu-Tao Xia, Bin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.03587)  

**Abstract**: Self-Supervised Video Hashing (SSVH) compresses videos into hash codes for efficient indexing and retrieval using unlabeled training videos. Existing approaches rely on random frame sampling to learn video features and treat all frames equally. This results in suboptimal hash codes, as it ignores frame-specific information density and reconstruction difficulty. To address this limitation, we propose a new framework, termed AutoSSVH, that employs adversarial frame sampling with hash-based contrastive learning. Our adversarial sampling strategy automatically identifies and selects challenging frames with richer information for reconstruction, enhancing encoding capability. Additionally, we introduce a hash component voting strategy and a point-to-set (P2Set) hash-based contrastive objective, which help capture complex inter-video semantic relationships in the Hamming space and improve the discriminability of learned hash codes. Extensive experiments demonstrate that AutoSSVH achieves superior retrieval efficacy and efficiency compared to state-of-the-art approaches. Code is available at this https URL. 

---
# RANa: Retrieval-Augmented Navigation 

**Authors**: Gianluca Monaci, Rafael S. Rezende, Romain Deffayet, Gabriela Csurka, Guillaume Bono, Hervé Déjean, Stéphane Clinchant, Christian Wolf  

**Link**: [PDF](https://arxiv.org/pdf/2504.03524)  

**Abstract**: Methods for navigation based on large-scale learning typically treat each episode as a new problem, where the agent is spawned with a clean memory in an unknown environment. While these generalization capabilities to an unknown environment are extremely important, we claim that, in a realistic setting, an agent should have the capacity of exploiting information collected during earlier robot operations. We address this by introducing a new retrieval-augmented agent, trained with RL, capable of querying a database collected from previous episodes in the same environment and learning how to integrate this additional context information. We introduce a unique agent architecture for the general navigation task, evaluated on ObjectNav, ImageNav and Instance-ImageNav. Our retrieval and context encoding methods are data-driven and heavily employ vision foundation models (FM) for both semantic and geometric understanding. We propose new benchmarks for these settings and we show that retrieval allows zero-shot transfer across tasks and environments while significantly improving performance. 

---
# Structured Legal Document Generation in India: A Model-Agnostic Wrapper Approach with VidhikDastaavej 

**Authors**: Shubham Kumar Nigam, Balaramamahanthi Deepak Patnaik, Ajay Varghese Thomas, Noel Shallum, Kripabandhu Ghosh, Arnab Bhattacharya  

**Link**: [PDF](https://arxiv.org/pdf/2504.03486)  

**Abstract**: Automating legal document drafting can significantly enhance efficiency, reduce manual effort, and streamline legal workflows. While prior research has explored tasks such as judgment prediction and case summarization, the structured generation of private legal documents in the Indian legal domain remains largely unaddressed. To bridge this gap, we introduce VidhikDastaavej, a novel, anonymized dataset of private legal documents, and develop NyayaShilp, a fine-tuned legal document generation model specifically adapted to Indian legal texts. We propose a Model-Agnostic Wrapper (MAW), a two-step framework that first generates structured section titles and then iteratively produces content while leveraging retrieval-based mechanisms to ensure coherence and factual accuracy. We benchmark multiple open-source LLMs, including instruction-tuned and domain-adapted versions, alongside proprietary models for comparison. Our findings indicate that while direct fine-tuning on small datasets does not always yield improvements, our structured wrapper significantly enhances coherence, factual adherence, and overall document quality while mitigating hallucinations. To ensure real-world applicability, we developed a Human-in-the-Loop (HITL) Document Generation System, an interactive user interface that enables users to specify document types, refine section details, and generate structured legal drafts. This tool allows legal professionals and researchers to generate, validate, and refine AI-generated legal documents efficiently. Extensive evaluations, including expert assessments, confirm that our framework achieves high reliability in structured legal drafting. This research establishes a scalable and adaptable foundation for AI-assisted legal drafting in India, offering an effective approach to structured legal document generation. 

---
# Talk2X -- An Open-Source Toolkit Facilitating Deployment of LLM-Powered Chatbots on the Web 

**Authors**: Lars Krupp, Daniel Geißler, Peter Hevesi, Marco Hirsch, Paul Lukowicz, Jakob Karolus  

**Link**: [PDF](https://arxiv.org/pdf/2504.03343)  

**Abstract**: Integrated into websites, LLM-powered chatbots offer alternative means of navigation and information retrieval, leading to a shift in how users access information on the web. Yet, predominantly closed-sourced solutions limit proliferation among web hosts and suffer from a lack of transparency with regard to implementation details and energy efficiency. In this work, we propose our openly available agent Talk2X leveraging an adapted retrieval-augmented generation approach (RAG) combined with an automatically generated vector database, benefiting energy efficiency. Talk2X's architecture is generalizable to arbitrary websites offering developers a ready to use tool for integration. Using a mixed-methods approach, we evaluated Talk2X's usability by tasking users to acquire specific assets from an open science repository. Talk2X significantly improved task completion time, correctness, and user experience supporting users in quickly pinpointing specific information as compared to standard user-website interaction. Our findings contribute technical advancements to an ongoing paradigm shift of how we access information on the web. 

---
# Optimal Embedding Guided Negative Sample Generation for Knowledge Graph Link Prediction 

**Authors**: Makoto Takamoto, Daniel Oñoro-Rubio, Wiem Ben Rim, Takashi Maruyama, Bhushan Kotnis  

**Link**: [PDF](https://arxiv.org/pdf/2504.03327)  

**Abstract**: Knowledge graph embedding (KGE) models encode the structural information of knowledge graphs to predicting new links. Effective training of these models requires distinguishing between positive and negative samples with high precision. Although prior research has shown that improving the quality of negative samples can significantly enhance model accuracy, identifying high-quality negative samples remains a challenging problem. This paper theoretically investigates the condition under which negative samples lead to optimal KG embedding and identifies a sufficient condition for an effective negative sample distribution. Based on this theoretical foundation, we propose \textbf{E}mbedding \textbf{MU}tation (\textsc{EMU}), a novel framework that \emph{generates} negative samples satisfying this condition, in contrast to conventional methods that focus on \emph{identifying} challenging negative samples within the training data. Importantly, the simplicity of \textsc{EMU} ensures seamless integration with existing KGE models and negative sampling methods. To evaluate its efficacy, we conducted comprehensive experiments across multiple datasets. The results consistently demonstrate significant improvements in link prediction performance across various KGE models and negative sampling methods. Notably, \textsc{EMU} enables performance improvements comparable to those achieved by models with embedding dimension five times larger. An implementation of the method and experiments are available at this https URL. 

---
# Scraping the Shadows: Deep Learning Breakthroughs in Dark Web Intelligence 

**Authors**: Ingmar Bakermans, Daniel De Pascale, Gonçalo Marcelino, Giuseppe Cascavilla, Zeno Geradts  

**Link**: [PDF](https://arxiv.org/pdf/2504.02872)  

**Abstract**: Darknet markets (DNMs) facilitate the trade of illegal goods on a global scale. Gathering data on DNMs is critical to ensuring law enforcement agencies can effectively combat crime. Manually extracting data from DNMs is an error-prone and time-consuming task. Aiming to automate this process we develop a framework for extracting data from DNMs and evaluate the application of three state-of-the-art Named Entity Recognition (NER) models, ELMo-BiLSTM \citep{ShahEtAl2022}, UniversalNER \citep{ZhouEtAl2024}, and GLiNER \citep{ZaratianaEtAl2023}, at the task of extracting complex entities from DNM product listing pages. We propose a new annotated dataset, which we use to train, fine-tune, and evaluate the models. Our findings show that state-of-the-art NER models perform well in information extraction from DNMs, achieving 91% Precision, 96% Recall, and an F1 score of 94%. In addition, fine-tuning enhances model performance, with UniversalNER achieving the best performance. 

---
# Synthesized Annotation Guidelines are Knowledge-Lite Boosters for Clinical Information Extraction 

**Authors**: Enshuo Hsu, Martin Ugbala, Krishna Kumar Kookal, Zouaidi Kawtar, Nicholas L. Rider, Muhammad F. Walji, Kirk Roberts  

**Link**: [PDF](https://arxiv.org/pdf/2504.02871)  

**Abstract**: Generative information extraction using large language models, particularly through few-shot learning, has become a popular method. Recent studies indicate that providing a detailed, human-readable guideline-similar to the annotation guidelines traditionally used for training human annotators can significantly improve performance. However, constructing these guidelines is both labor- and knowledge-intensive. Additionally, the definitions are often tailored to meet specific needs, making them highly task-specific and often non-reusable. Handling these subtle differences requires considerable effort and attention to detail. In this study, we propose a self-improving method that harvests the knowledge summarization and text generation capacity of LLMs to synthesize annotation guidelines while requiring virtually no human input. Our zero-shot experiments on the clinical named entity recognition benchmarks, 2012 i2b2 EVENT, 2012 i2b2 TIMEX, 2014 i2b2, and 2018 n2c2 showed 25.86%, 4.36%, 0.20%, and 7.75% improvements in strict F1 scores from the no-guideline baseline. The LLM-synthesized guidelines showed equivalent or better performance compared to human-written guidelines by 1.15% to 4.14% in most tasks. In conclusion, this study proposes a novel LLM self-improving method that requires minimal knowledge and human input and is applicable to multiple biomedical domains. 

---
# Integrating Notch Filtering and Statistical Methods for Improved Cardiac Diagnostics Using MATLAB 

**Authors**: Lohit Bibar, Samali bose, Tribeni Prasad Banerjee  

**Link**: [PDF](https://arxiv.org/pdf/2504.02847)  

**Abstract**: A Notch Filter is essential in ECG signal processing to eliminate narrowband noise, especially powerline interference at 50 Hz or 60 Hz. This interference overlaps with vital ECG signal features, affecting the accuracy of downstream classification tasks (e.g., arrhythmia detection). A properly designed notch filter enhances signal quality, preserves essential ECG components (P, QRS, T waves), and improves the performance of machine learning or deep learning models used for ECG classification. 

---
