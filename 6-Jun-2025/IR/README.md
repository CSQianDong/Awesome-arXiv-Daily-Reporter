# On the Comprehensibility of Multi-structured Financial Documents using LLMs and Pre-processing Tools 

**Authors**: Shivani Upadhyay, Messiah Ataey, Shariyar Murtuza, Yifan Nie, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2506.05182)  

**Abstract**: The proliferation of complex structured data in hybrid sources, such as PDF documents and web pages, presents unique challenges for current Large Language Models (LLMs) and Multi-modal Large Language Models (MLLMs) in providing accurate answers. Despite the recent advancements of MLLMs, they still often falter when interpreting intricately structured information, such as nested tables and multi-dimensional plots, leading to hallucinations and erroneous outputs. This paper explores the capabilities of LLMs and MLLMs in understanding and answering questions from complex data structures found in PDF documents by leveraging industrial and open-source tools as part of a pre-processing pipeline. Our findings indicate that GPT-4o, a popular MLLM, achieves an accuracy of 56% on multi-structured documents when fed documents directly, and that integrating pre-processing tools raises the accuracy of LLMs to 61.3% for GPT-4o and 76% for GPT-4, and with lower overall cost. The code is publicly available at this https URL. 

---
# Reason-to-Recommend: Using Interaction-of-Thought Reasoning to Enhance LLM Recommendation 

**Authors**: Keyu Zhao, Fengli Xu, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.05069)  

**Abstract**: Driven by advances in Large Language Models (LLMs), integrating them into recommendation tasks has gained interest due to their strong semantic understanding and prompt flexibility. Prior work encoded user-item interactions or metadata into prompts for recommendations. In parallel, LLM reasoning, boosted by test-time scaling and reinforcement learning, has excelled in fields like mathematics and code, where reasoning traces and correctness signals are clear, enabling high performance and interpretability. However, directly applying these reasoning methods to recommendation is ineffective because user feedback is implicit and lacks reasoning supervision. To address this, we propose $\textbf{R2Rec}$, a reasoning-enhanced recommendation framework that samples interaction chains from the user-item graph and converts them into structured interaction-of-thoughts via a progressive masked prompting strategy, with each thought representing stepwise reasoning grounded in interaction context. This allows LLMs to simulate step-by-step decision-making based on implicit patterns. We design a two-stage training pipeline: supervised fine-tuning teaches basic reasoning from high-quality traces, and reinforcement learning refines reasoning via reward signals, alleviating sparse explicit supervision. Experiments on three real-world datasets show R2Rec outperforms classical and LLM-based baselines with an average $\textbf{10.48%}$ improvement in HitRatio@1 and $\textbf{131.81%}$ gain over the original LLM. Furthermore, the explicit reasoning chains enhance interpretability by revealing the decision process. Our code is available at: this https URL. 

---
# Rethinking Contrastive Learning in Session-based Recommendation 

**Authors**: Xiaokun Zhang, Bo Xu, Fenglong Ma, Zhizheng Wang, Liang Yang, Hongfei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2506.05044)  

**Abstract**: Session-based recommendation aims to predict intents of anonymous users based on limited behaviors. With the ability in alleviating data sparsity, contrastive learning is prevailing in the task. However, we spot that existing contrastive learning based methods still suffer from three obstacles: (1) they overlook item-level sparsity and primarily focus on session-level sparsity; (2) they typically augment sessions using item IDs like crop, mask and reorder, failing to ensure the semantic consistency of augmented views; (3) they treat all positive-negative signals equally, without considering their varying utility. To this end, we propose a novel multi-modal adaptive contrastive learning framework called MACL for session-based recommendation. In MACL, a multi-modal augmentation is devised to generate semantically consistent views at both item and session levels by leveraging item multi-modal features. Besides, we present an adaptive contrastive loss that distinguishes varying contributions of positive-negative signals to improve self-supervised learning. Extensive experiments on three real-world datasets demonstrate the superiority of MACL over state-of-the-art methods. 

---
# Towards Storage-Efficient Visual Document Retrieval: An Empirical Study on Reducing Patch-Level Embeddings 

**Authors**: Yubo Ma, Jinsong Li, Yuhang Zang, Xiaobao Wu, Xiaoyi Dong, Pan Zhang, Yuhang Cao, Haodong Duan, Jiaqi Wang, Yixin Cao, Aixin Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.04997)  

**Abstract**: Despite the strong performance of ColPali/ColQwen2 in Visualized Document Retrieval (VDR), it encodes each page into multiple patch-level embeddings and leads to excessive memory usage. This empirical study investigates methods to reduce patch embeddings per page at minimum performance degradation. We evaluate two token-reduction strategies: token pruning and token merging. Regarding token pruning, we surprisingly observe that a simple random strategy outperforms other sophisticated pruning methods, though still far from satisfactory. Further analysis reveals that pruning is inherently unsuitable for VDR as it requires removing certain page embeddings without query-specific information. Turning to token merging (more suitable for VDR), we search for the optimal combinations of merging strategy across three dimensions and develop Light-ColPali/ColQwen2. It maintains 98.2% of retrieval performance with only 11.8% of original memory usage, and preserves 94.6% effectiveness at 2.8% memory footprint. We expect our empirical findings and resulting Light-ColPali/ColQwen2 offer valuable insights and establish a competitive baseline for future research towards efficient VDR. 

---
# GOLFer: Smaller LM-Generated Documents Hallucination Filter & Combiner for Query Expansion in Information Retrieval 

**Authors**: Lingyuan Liu, Mengxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.04762)  

**Abstract**: Large language models (LLMs)-based query expansion for information retrieval augments queries with generated hypothetical documents with LLMs. However, its performance relies heavily on the scale of the language models (LMs), necessitating larger, more advanced LLMs. This approach is costly, computationally intensive, and often has limited accessibility. To address these limitations, we introduce GOLFer - Smaller LMs-Generated Documents Hallucination Filter & Combiner - a novel method leveraging smaller open-source LMs for query expansion. GOLFer comprises two modules: a hallucination filter and a documents combiner. The former detects and removes non-factual and inconsistent sentences in generated documents, a common issue with smaller LMs, while the latter combines the filtered content with the query using a weight vector to balance their influence. We evaluate GOLFer alongside dominant LLM-based query expansion methods on three web search and ten low-resource datasets. Experimental results demonstrate that GOLFer consistently outperforms other methods using smaller LMs, and maintains competitive performance against methods using large-size LLMs, demonstrating its effectiveness. 

---
# Exp4Fuse: A Rank Fusion Framework for Enhanced Sparse Retrieval using Large Language Model-based Query Expansion 

**Authors**: Lingyuan Liu, Mengxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.04760)  

**Abstract**: Large Language Models (LLMs) have shown potential in generating hypothetical documents for query expansion, thereby enhancing information retrieval performance. However, the efficacy of this method is highly dependent on the quality of the generated documents, which often requires complex prompt strategies and the integration of advanced dense retrieval techniques. This can be both costly and computationally intensive. To mitigate these limitations, we explore the use of zero-shot LLM-based query expansion to improve sparse retrieval, particularly for learned sparse retrievers. We introduce a novel fusion ranking framework, Exp4Fuse, which enhances the performance of sparse retrievers through an indirect application of zero-shot LLM-based query expansion. Exp4Fuse operates by simultaneously considering two retrieval routes-one based on the original query and the other on the LLM-augmented query. It then generates two ranked lists using a sparse retriever and fuses them using a modified reciprocal rank fusion method. We conduct extensive evaluations of Exp4Fuse against leading LLM-based query expansion methods and advanced retrieval techniques on three MS MARCO-related datasets and seven low-resource datasets. Experimental results reveal that Exp4Fuse not only surpasses existing LLM-based query expansion methods in enhancing sparse retrievers but also, when combined with advanced sparse retrievers, achieves SOTA results on several benchmarks. This highlights the superior performance and effectiveness of Exp4Fuse in improving query expansion for sparse retrieval. 

---
# PUB: An LLM-Enhanced Personality-Driven User Behaviour Simulator for Recommender System Evaluation 

**Authors**: Chenglong Ma, Ziqi Xu, Yongli Ren, Danula Hettiachchi, Jeffrey Chan  

**Link**: [PDF](https://arxiv.org/pdf/2506.04551)  

**Abstract**: Traditional offline evaluation methods for recommender systems struggle to capture the complexity of modern platforms due to sparse behavioural signals, noisy data, and limited modelling of user personality traits. While simulation frameworks can generate synthetic data to address these gaps, existing methods fail to replicate behavioural diversity, limiting their effectiveness. To overcome these challenges, we propose the Personality-driven User Behaviour Simulator (PUB), an LLM-based simulation framework that integrates the Big Five personality traits to model personalised user behaviour. PUB dynamically infers user personality from behavioural logs (e.g., ratings, reviews) and item metadata, then generates synthetic interactions that preserve statistical fidelity to real-world data. Experiments on the Amazon review datasets show that logs generated by PUB closely align with real user behaviour and reveal meaningful associations between personality traits and recommendation outcomes. These results highlight the potential of the personality-driven simulator to advance recommender system evaluation, offering scalable, controllable, high-fidelity alternatives to resource-intensive real-world experiments. 

---
# I'm Sorry Dave, I'm Afraid I Can't Return That: On YouTube Search API Use in Research 

**Authors**: Alexandros Efstratiou  

**Link**: [PDF](https://arxiv.org/pdf/2506.04422)  

**Abstract**: YouTube is among the most widely-used platforms worldwide, and has seen a lot of recent academic attention. Despite its popularity and the number of studies conducted on it, much less is understood about the way in which YouTube's Data API, and especially the Search endpoint, operates. In this paper, we analyze the API's behavior by running identical queries across a period of 12 weeks. Our findings suggest that the search endpoint returns highly inconsistent results between queries in ways that are not officially documented. Specifically, the API seems to randomize returned videos based on the relative popularity of the respective topic during the query period, making it nearly impossible to obtain representative historical video samples, especially during non-peak topical periods. Our results also suggest that the API may prioritize shorter, more popular videos, although the role of channel popularity is not as clear. We conclude with suggested strategies for researchers using the API for data collection, as well as future research directions on expanding the API's use-cases. 

---
# Search Arena: Analyzing Search-Augmented LLMs 

**Authors**: Mihran Miroyan, Tsung-Han Wu, Logan King, Tianle Li, Jiayi Pan, Xinyan Hu, Wei-Lin Chiang, Anastasios N. Angelopoulos, Trevor Darrell, Narges Norouzi, Joseph E. Gonzalez  

**Link**: [PDF](https://arxiv.org/pdf/2506.05334)  

**Abstract**: Search-augmented language models combine web search with Large Language Models (LLMs) to improve response groundedness and freshness. However, analyzing these systems remains challenging: existing datasets are limited in scale and narrow in scope, often constrained to static, single-turn, fact-checking questions. In this work, we introduce Search Arena, a crowd-sourced, large-scale, human-preference dataset of over 24,000 paired multi-turn user interactions with search-augmented LLMs. The dataset spans diverse intents and languages, and contains full system traces with around 12,000 human preference votes. Our analysis reveals that user preferences are influenced by the number of citations, even when the cited content does not directly support the attributed claims, uncovering a gap between perceived and actual credibility. Furthermore, user preferences vary across cited sources, revealing that community-driven platforms are generally preferred and static encyclopedic sources are not always appropriate and reliable. To assess performance across different settings, we conduct cross-arena analyses by testing search-augmented LLMs in a general-purpose chat environment and conventional LLMs in search-intensive settings. We find that web search does not degrade and may even improve performance in non-search settings; however, the quality in search settings is significantly affected if solely relying on the model's parametric knowledge. We open-sourced the dataset to support future research in this direction. Our dataset and code are available at: this https URL. 

---
# ECoRAG: Evidentiality-guided Compression for Long Context RAG 

**Authors**: Yeonseok Jeong, Jinsu Kim, Dohyeon Lee, Seung-won Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05167)  

**Abstract**: Large Language Models (LLMs) have shown remarkable performance in Open-Domain Question Answering (ODQA) by leveraging external documents through Retrieval-Augmented Generation (RAG). To reduce RAG overhead, from longer context, context compression is necessary. However, prior compression methods do not focus on filtering out non-evidential information, which limit the performance in LLM-based RAG. We thus propose Evidentiality-guided RAG, or \textbf{ECoRAG} framework. ECoRAG improves LLM performance by compressing retrieved documents based on evidentiality, ensuring whether answer generation is supported by the correct evidence. As an additional step, ECoRAG reflects whether the compressed content provides sufficient evidence, and if not, retrieves more until sufficient. Experiments show that ECoRAG improves LLM performance on ODQA tasks, outperforming existing compression methods. Furthermore, ECoRAG is highly cost-efficient, as it not only reduces latency but also minimizes token usage by retaining only the necessary information to generate the correct answer. Code is available at this https URL. 

---
# Knowledgeable-r1: Policy Optimization for Knowledge Exploration in Retrieval-Augmented Generation 

**Authors**: Chenyu Lin, Yilin Wen, Du Su, Fei Sun, Muhan Chen, Chenfu Bao, Zhonghou Lv  

**Link**: [PDF](https://arxiv.org/pdf/2506.05154)  

**Abstract**: Retrieval-augmented generation (RAG) is a mainstream method for improving performance on knowledge-intensive tasks. However,current RAG systems often place too much emphasis on retrieved contexts. This can lead to reliance on inaccurate sources and overlook the model's inherent knowledge, especially when dealing with misleading or excessive information. To resolve this imbalance, we propose Knowledgeable-r1 that using joint sampling and define multi policy distributions in knowledge capability exploration to stimulate large language models'self-integrated utilization of parametric and contextual knowledge. Experiments show that Knowledgeable-r1 significantly enhances robustness and reasoning accuracy in both parameters and contextual conflict tasks and general RAG tasks, especially outperforming baselines by 17.07% in counterfactual scenarios and demonstrating consistent gains across RAG tasks. Our code are available at this https URL knowledgeable-r1. 

---
# Verbose ListOps (VLO): Beyond Long Context -- Unmasking LLM's Reasoning Blind Spots 

**Authors**: Alex Pan, Mary-Anne Williams  

**Link**: [PDF](https://arxiv.org/pdf/2506.04907)  

**Abstract**: Large Language Models (LLMs), whilst great at extracting facts from text, struggle with nested narrative reasoning. Existing long context and multi-hop QA benchmarks inadequately test this, lacking realistic distractors or failing to decouple context length from reasoning complexity, masking a fundamental LLM limitation. We introduce Verbose ListOps, a novel benchmark that programmatically transposes ListOps computations into lengthy, coherent stories. This uniquely forces internal computation and state management of nested reasoning problems by withholding intermediate results, and offers fine-grained controls for both narrative size \emph{and} reasoning difficulty. Whilst benchmarks like LongReason (2025) advance approaches for synthetically expanding the context size of multi-hop QA problems, Verbose ListOps pinpoints a specific LLM vulnerability: difficulty in state management for nested sub-reasoning amongst semantically-relevant, distracting narrative. Our experiments show that leading LLMs (e.g., OpenAI o4, Gemini 2.5 Pro) collapse in performance on Verbose ListOps at modest (~10k token) narrative lengths, despite effortlessly solving raw ListOps equations. Addressing this failure is paramount for real-world text interpretation which requires identifying key reasoning points, tracking conceptual intermediate results, and filtering irrelevant information. Verbose ListOps, and its extensible generation framework thus enables targeted reasoning enhancements beyond mere context-window expansion; a critical step to automating the world's knowledge work. 

---
# LotusFilter: Fast Diverse Nearest Neighbor Search via a Learned Cutoff Table 

**Authors**: Yusuke Matsui  

**Link**: [PDF](https://arxiv.org/pdf/2506.04790)  

**Abstract**: Approximate nearest neighbor search (ANNS) is an essential building block for applications like RAG but can sometimes yield results that are overly similar to each other. In certain scenarios, search results should be similar to the query and yet diverse. We propose LotusFilter, a post-processing module to diversify ANNS results. We precompute a cutoff table summarizing vectors that are close to each other. During the filtering, LotusFilter greedily looks up the table to delete redundant vectors from the candidates. We demonstrated that the LotusFilter operates fast (0.02 [ms/query]) in settings resembling real-world RAG applications, utilizing features such as OpenAI embeddings. Our code is publicly available at this https URL. 

---
# CoRe-MMRAG: Cross-Source Knowledge Reconciliation for Multimodal RAG 

**Authors**: Yang Tian, Fan Liu, Jingyuan Zhang, Victoria W., Yupeng Hu, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2506.02544)  

**Abstract**: Multimodal Retrieval-Augmented Generation (MMRAG) has been introduced to enhance Multimodal Large Language Models by incorporating externally retrieved multimodal knowledge, but it introduces two challenges: Parametric-Retrieved Knowledge Inconsistency (PRKI), where discrepancies between parametric and retrieved knowledge create uncertainty in determining reliability, and Visual-Textual Knowledge Inconsistency (VTKI), where misalignment between visual and textual sources disrupts entity representation. To address these challenges, we propose Cross-source knowledge \textbf{Re}conciliation for Multimodal RAG (CoRe-MMRAG), a novel end-to-end framework that effectively reconciles inconsistencies across knowledge sources. CoRe-MMRAG follows a four-stage pipeline: it first generates an internal response from parametric knowledge, then selects the most relevant multimodal evidence via joint similarity assessment, generates an external response, and finally integrates both to produce a reliable answer. Additionally, a specialized training paradigm enhances knowledge source discrimination, multimodal integration, and unified answer generation. Experiments on KB-VQA benchmarks show that CoRe-MMRAG achieves substantial improvements over baseline methods, achieving 5.6% and 9.3% performance gains on InfoSeek and Encyclopedic-VQA, respectively. 

---
# CMIE: Combining MLLM Insights with External Evidence for Explainable Out-of-Context Misinformation Detection 

**Authors**: Fanxiao Li, Jiaying Wu, Canyuan He, Wei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.23449)  

**Abstract**: Multimodal large language models (MLLMs) have demonstrated impressive capabilities in visual reasoning and text generation. While previous studies have explored the application of MLLM for detecting out-of-context (OOC) misinformation, our empirical analysis reveals two persisting challenges of this paradigm. Evaluating the representative GPT-4o model on direct reasoning and evidence augmented reasoning, results indicate that MLLM struggle to capture the deeper relationships-specifically, cases in which the image and text are not directly connected but are associated through underlying semantic links. Moreover, noise in the evidence further impairs detection accuracy. To address these challenges, we propose CMIE, a novel OOC misinformation detection framework that incorporates a Coexistence Relationship Generation (CRG) strategy and an Association Scoring (AS) mechanism. CMIE identifies the underlying coexistence relationships between images and text, and selectively utilizes relevant evidence to enhance misinformation detection. Experimental results demonstrate that our approach outperforms existing methods. 

---
# Prosthetics of the Indian State: The e-Shram Portal for Unorganized Workers in India 

**Authors**: Rozin Hasin  

**Link**: [PDF](https://arxiv.org/pdf/2503.05714)  

**Abstract**: This research paper examines the digital portal/database for unorganized workers in the informal sector economy of India today: e-Shram. Using affordance theory, I criticize the operationalization of this database for the labourers, alongside problems of accessibility and perception. 

---
