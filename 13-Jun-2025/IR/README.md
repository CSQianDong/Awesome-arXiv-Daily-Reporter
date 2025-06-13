# Precise Zero-Shot Pointwise Ranking with LLMs through Post-Aggregated Global Context Information 

**Authors**: Kehan Long, Shasha Li, Chen Xu, Jintao Tang, Ting Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10859)  

**Abstract**: Recent advancements have successfully harnessed the power of Large Language Models (LLMs) for zero-shot document ranking, exploring a variety of prompting strategies. Comparative approaches like pairwise and listwise achieve high effectiveness but are computationally intensive and thus less practical for larger-scale applications. Scoring-based pointwise approaches exhibit superior efficiency by independently and simultaneously generating the relevance scores for each candidate document. However, this independence ignores critical comparative insights between documents, resulting in inconsistent scoring and suboptimal performance. In this paper, we aim to improve the effectiveness of pointwise methods while preserving their efficiency through two key innovations: (1) We propose a novel Global-Consistent Comparative Pointwise Ranking (GCCP) strategy that incorporates global reference comparisons between each candidate and an anchor document to generate contrastive relevance scores. We strategically design the anchor document as a query-focused summary of pseudo-relevant candidates, which serves as an effective reference point by capturing the global context for document comparison. (2) These contrastive relevance scores can be efficiently Post-Aggregated with existing pointwise methods, seamlessly integrating essential Global Context information in a training-free manner (PAGC). Extensive experiments on the TREC DL and BEIR benchmark demonstrate that our approach significantly outperforms previous pointwise methods while maintaining comparable efficiency. Our method also achieves competitive performance against comparative methods that require substantially more computational resources. More analyses further validate the efficacy of our anchor construction strategy. 

---
# Constructing and Evaluating Declarative RAG Pipelines in PyTerrier 

**Authors**: Craig Macdonald, Jinyuan Fang, Andrew Parry, Zaiqiao Meng  

**Link**: [PDF](https://arxiv.org/pdf/2506.10802)  

**Abstract**: Search engines often follow a pipeline architecture, where complex but effective reranking components are used to refine the results of an initial retrieval. Retrieval augmented generation (RAG) is an exciting application of the pipeline architecture, where the final component generates a coherent answer for the users from the retrieved documents. In this demo paper, we describe how such RAG pipelines can be formulated in the declarative PyTerrier architecture, and the advantages of doing so. Our PyTerrier-RAG extension for PyTerrier provides easy access to standard RAG datasets and evaluation measures, state-of-the-art LLM readers, and using PyTerrier's unique operator notation, easy-to-build pipelines. We demonstrate the succinctness of indexing and RAG pipelines on standard datasets (including Natural Questions) and how to build on the larger PyTerrier ecosystem with state-of-the-art sparse, learned-sparse, and dense retrievers, and other neural rankers. 

---
# Contrastive Matrix Completion with Denoising and Augmented Graph Views for Robust Recommendation 

**Authors**: Narges Nemati, Mostafa Haghir Chehreghani  

**Link**: [PDF](https://arxiv.org/pdf/2506.10658)  

**Abstract**: Matrix completion is a widely adopted framework in recommender systems, as predicting the missing entries in the user-item rating matrix enables a comprehensive understanding of user preferences. However, current graph neural network (GNN)-based approaches are highly sensitive to noisy or irrelevant edges--due to their inherent message-passing mechanisms--and are prone to overfitting, which limits their generalizability. To overcome these challenges, we propose a novel method called Matrix Completion using Contrastive Learning (MCCL). Our approach begins by extracting local neighborhood subgraphs for each interaction and subsequently generates two distinct graph representations. The first representation emphasizes denoising by integrating GNN layers with an attention mechanism, while the second is obtained via a graph variational autoencoder that aligns the feature distribution with a standard prior. A mutual learning loss function is employed during training to gradually harmonize these representations, enabling the model to capture common patterns and significantly enhance its generalizability. Extensive experiments on several real-world datasets demonstrate that our approach not only improves the numerical accuracy of the predicted scores--achieving up to a 0.8% improvement in RMSE--but also produces superior rankings with improvements of up to 36% in ranking metrics. 

---
# Conversational Search: From Fundamentals to Frontiers in the LLM Era 

**Authors**: Fengran Mo, Chuan Meng, Mohammad Aliannejadi, Jian-Yun Nie  

**Link**: [PDF](https://arxiv.org/pdf/2506.10635)  

**Abstract**: Conversational search enables multi-turn interactions between users and systems to fulfill users' complex information needs. During this interaction, the system should understand the users' search intent within the conversational context and then return the relevant information through a flexible, dialogue-based interface. The recent powerful large language models (LLMs) with capacities of instruction following, content generation, and reasoning, attract significant attention and advancements, providing new opportunities and challenges for building up intelligent conversational search systems. This tutorial aims to introduce the connection between fundamentals and the emerging topics revolutionized by LLMs in the context of conversational search. It is designed for students, researchers, and practitioners from both academia and industry. Participants will gain a comprehensive understanding of both the core principles and cutting-edge developments driven by LLMs in conversational search, equipping them with the knowledge needed to contribute to the development of next-generation conversational search systems. 

---
# Macro Graph of Experts for Billion-Scale Multi-Task Recommendation 

**Authors**: Hongyu Yao, Zijin Hong, Hao Chen, Yuanchen Bei, Zhiqing Li, Qijie Shen, Zuobin Ying, Huan Gong, Feiran Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.10520)  

**Abstract**: Graph-based multi-task learning at billion-scale presents a significant challenge, as different tasks correspond to distinct billion-scale graphs. Traditional multi-task learning methods often neglect these graph structures, relying solely on individual user and item embeddings. However, disregarding graph structures overlooks substantial potential for improving performance. In this paper, we introduce the Macro Graph of Expert (MGOE) framework, the first approach capable of leveraging macro graph embeddings to capture task-specific macro features while modeling the correlations between task-specific experts. Specifically, we propose the concept of a Macro Graph Bottom, which, for the first time, enables multi-task learning models to incorporate graph information effectively. We design the Macro Prediction Tower to dynamically integrate macro knowledge across tasks. MGOE has been deployed at scale, powering multi-task learning for the homepage of a leading billion-scale recommender system. Extensive offline experiments conducted on three public benchmark datasets demonstrate its superiority over state-of-the-art multi-task learning methods, establishing MGOE as a breakthrough in multi-task graph-based recommendation. Furthermore, online A/B tests confirm the superiority of MGOE in billion-scale recommender systems. 

---
# SHORE: A Long-term User Lifetime Value Prediction Model in Digital Games 

**Authors**: Shuaiqi Sun, Congde Yuan, Haoqiang Yang, Mengzhuo Guo, Guiying Wei, Jiangbo Tian  

**Link**: [PDF](https://arxiv.org/pdf/2506.10487)  

**Abstract**: In digital gaming, long-term user lifetime value (LTV) prediction is essential for monetization strategy, yet presents major challenges due to delayed payment behavior, sparse early user data, and the presence of high-value outliers. While existing models typically rely on either short-cycle observations or strong distributional assumptions, such approaches often underestimate long-term value or suffer from poor robustness. To address these issues, we propose SHort-cycle auxiliary with Order-preserving REgression (SHORE), a novel LTV prediction framework that integrates short-horizon predictions (e.g., LTV-15 and LTV-30) as auxiliary tasks to enhance long-cycle targets (e.g., LTV-60). SHORE also introduces a hybrid loss function combining order-preserving multi-class classification and a dynamic Huber loss to mitigate the influence of zero-inflation and outlier payment behavior. Extensive offline and online experiments on real-world datasets demonstrate that SHORE significantly outperforms existing baselines, achieving a 47.91\% relative reduction in prediction error in online deployment. These results highlight SHORE's practical effectiveness and robustness in industrial-scale LTV prediction for digital games. 

---
# LightKG: Efficient Knowledge-Aware Recommendations with Simplified GNN Architecture 

**Authors**: Yanhui Li, Dongxia Wang, Zhu Sun, Haonan Zhang, Huizhong Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.10347)  

**Abstract**: Recently, Graph Neural Networks (GNNs) have become the dominant approach for Knowledge Graph-aware Recommender Systems (KGRSs) due to their proven effectiveness. Building upon GNN-based KGRSs, Self-Supervised Learning (SSL) has been incorporated to address the sparity issue, leading to longer training time. However, through extensive experiments, we reveal that: (1)compared to other KGRSs, the existing GNN-based KGRSs fail to keep their superior performance under sparse interactions even with SSL. (2) More complex models tend to perform worse in sparse interaction scenarios and complex mechanisms, like attention mechanism, can be detrimental as they often increase learning difficulty. Inspired by these findings, we propose LightKG, a simple yet powerful GNN-based KGRS to address sparsity issues. LightKG includes a simplified GNN layer that encodes directed relations as scalar pairs rather than dense embeddings and employs a linear aggregation framework, greatly reducing the complexity of GNNs. Additionally, LightKG incorporates an efficient contrastive layer to implement SSL. It directly minimizes the node similarity in original graph, avoiding the time-consuming subgraph generation and comparison required in previous SSL methods. Experiments on four benchmark datasets show that LightKG outperforms 12 competitive KGRSs in both sparse and dense scenarios while significantly reducing training time. Specifically, it surpasses the best baselines by an average of 5.8\% in recommendation accuracy and saves 84.3\% of training time compared to KGRSs with SSL. Our code is available at this https URL. 

---
# An Analysis of Datasets, Metrics and Models in Keyphrase Generation 

**Authors**: Florian Boudin, Akiko Aizawa  

**Link**: [PDF](https://arxiv.org/pdf/2506.10346)  

**Abstract**: Keyphrase generation refers to the task of producing a set of words or phrases that summarises the content of a document. Continuous efforts have been dedicated to this task over the past few years, spreading across multiple lines of research, such as model architectures, data resources, and use-case scenarios. Yet, the current state of keyphrase generation remains unknown as there has been no attempt to review and analyse previous work. In this paper, we bridge this gap by presenting an analysis of over 50 research papers on keyphrase generation, offering a comprehensive overview of recent progress, limitations, and open challenges. Our findings highlight several critical issues in current evaluation practices, such as the concerning similarity among commonly-used benchmark datasets and inconsistencies in metric calculations leading to overestimated performances. Additionally, we address the limited availability of pre-trained models by releasing a strong PLM-based model for keyphrase generation as an effort to facilitate future research. 

---
# Context-Adaptive Graph Neural Networks for Next POI Recommendation 

**Authors**: Yu Lei, Limin Shen, Zhu Sun, Tiantian He, Yew-Soon Ong  

**Link**: [PDF](https://arxiv.org/pdf/2506.10329)  

**Abstract**: Next Point-of-Interest (POI) recommendation is a critical task in location-based services, aiming to predict users' next visits based on their check-in histories. While many existing methods leverage Graph Neural Networks (GNNs) to incorporate collaborative information and improve recommendation accuracy, most of them model each type of context using separate graphs, treating different factors in isolation. This limits their ability to model the co-influence of multiple contextual factors on user transitions during message propagation, resulting in suboptimal attention weights and recommendation performance. Furthermore, they often prioritize sequential components as the primary predictor, potentially undermining the semantic and structural information encoded in the POI embeddings learned by GNNs. To address these limitations, we propose a Context-Adaptive Graph Neural Networks (CAGNN) for next POI recommendation, which dynamically adjusts attention weights using edge-specific contextual factors and enables mutual enhancement between graph-based and sequential components. Specifically, CAGNN introduces (1) a context-adaptive attention mechanism that jointly incorporates different types of contextual factors into the attention computation during graph propagation, enabling the model to dynamically capture collaborative and context-dependent transition patterns; (2) a graph-sequential mutual enhancement module, which aligns the outputs of the graph- and sequential-based modules via the KL divergence, enabling mutual enhancement of both components. Experimental results on three real-world datasets demonstrate that CAGNN consistently outperforms state-of-the-art methods. Meanwhile, theoretical guarantees are provided that our context-adaptive attention mechanism improves the expressiveness of POI representations. 

---
# Towards Understanding Bias in Synthetic Data for Evaluation 

**Authors**: Hossein A. Rahmani, Varsha Ramineni, Nick Craswell, Bhaskar Mitra, Emine Yilmaz  

**Link**: [PDF](https://arxiv.org/pdf/2506.10301)  

**Abstract**: Test collections are crucial for evaluating Information Retrieval (IR) systems. Creating a diverse set of user queries for these collections can be challenging, and obtaining relevance judgments, which indicate how well retrieved documents match a query, is often costly and resource-intensive. Recently, generating synthetic datasets using Large Language Models (LLMs) has gained attention in various applications. While previous work has used LLMs to generate synthetic queries or documents to improve ranking models, using LLMs to create synthetic test collections is still relatively unexplored. Previous work~\cite{rahmani2024synthetic} showed that synthetic test collections have the potential to be used for system evaluation, however, more analysis is needed to validate this claim. In this paper, we thoroughly investigate the reliability of synthetic test collections constructed using LLMs, where LLMs are used to generate synthetic queries, labels, or both. In particular, we examine the potential biases that might occur when such test collections are used for evaluation. We first empirically show the presence of such bias in evaluation results and analyse the effects it might have on system evaluation. We further validate the presence of such bias using a linear mixed-effects model. Our analysis shows that while the effect of bias present in evaluation results obtained using synthetic test collections could be significant, for e.g.~computing absolute system performance, its effect may not be as significant in comparing relative system performance. Codes and data are available at: this https URL. 

---
# ChineseHarm-Bench: A Chinese Harmful Content Detection Benchmark 

**Authors**: Kangwei Liu, Siyuan Cheng, Bozhong Tian, Xiaozhuan Liang, Yuyang Yin, Meng Han, Ningyu Zhang, Bryan Hooi, Xi Chen, Shumin Deng  

**Link**: [PDF](https://arxiv.org/pdf/2506.10960)  

**Abstract**: Large language models (LLMs) have been increasingly applied to automated harmful content detection tasks, assisting moderators in identifying policy violations and improving the overall efficiency and accuracy of content review. However, existing resources for harmful content detection are predominantly focused on English, with Chinese datasets remaining scarce and often limited in scope. We present a comprehensive, professionally annotated benchmark for Chinese content harm detection, which covers six representative categories and is constructed entirely from real-world data. Our annotation process further yields a knowledge rule base that provides explicit expert knowledge to assist LLMs in Chinese harmful content detection. In addition, we propose a knowledge-augmented baseline that integrates both human-annotated knowledge rules and implicit knowledge from large language models, enabling smaller models to achieve performance comparable to state-of-the-art LLMs. Code and data are available at this https URL. 

---
# CIIR@LiveRAG 2025: Optimizing Multi-Agent Retrieval Augmented Generation through Self-Training 

**Authors**: Alireza Salemi, Mukta Maddipatla, Hamed Zamani  

**Link**: [PDF](https://arxiv.org/pdf/2506.10844)  

**Abstract**: This paper presents mRAG, a multi-agent retrieval-augmented generation (RAG) framework composed of specialized agents for subtasks such as planning, searching, reasoning, and coordination. Our system uses a self-training paradigm with reward-guided trajectory sampling to optimize inter-agent collaboration and enhance response generation. Evaluated on DataMorgana-derived datasets during the SIGIR 2025 LiveRAG competition, mRAG outperforms conventional RAG baselines. We further analyze competition outcomes and showcase the framework's strengths with case studies, demonstrating its efficacy for complex, real-world RAG tasks. 

---
# TaxoAdapt: Aligning LLM-Based Multidimensional Taxonomy Construction to Evolving Research Corpora 

**Authors**: Priyanka Kargupta, Nan Zhang, Yunyi Zhang, Rui Zhang, Prasenjit Mitra, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.10737)  

**Abstract**: The rapid evolution of scientific fields introduces challenges in organizing and retrieving scientific literature. While expert-curated taxonomies have traditionally addressed this need, the process is time-consuming and expensive. Furthermore, recent automatic taxonomy construction methods either (1) over-rely on a specific corpus, sacrificing generalizability, or (2) depend heavily on the general knowledge of large language models (LLMs) contained within their pre-training datasets, often overlooking the dynamic nature of evolving scientific domains. Additionally, these approaches fail to account for the multi-faceted nature of scientific literature, where a single research paper may contribute to multiple dimensions (e.g., methodology, new tasks, evaluation metrics, benchmarks). To address these gaps, we propose TaxoAdapt, a framework that dynamically adapts an LLM-generated taxonomy to a given corpus across multiple dimensions. TaxoAdapt performs iterative hierarchical classification, expanding both the taxonomy width and depth based on corpus' topical distribution. We demonstrate its state-of-the-art performance across a diverse set of computer science conferences over the years to showcase its ability to structure and capture the evolution of scientific fields. As a multidimensional method, TaxoAdapt generates taxonomies that are 26.51% more granularity-preserving and 50.41% more coherent than the most competitive baselines judged by LLMs. 

---
# Beyond True or False: Retrieval-Augmented Hierarchical Analysis of Nuanced Claims 

**Authors**: Priyanka Kargupta, Runchu Tian, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.10728)  

**Abstract**: Claims made by individuals or entities are oftentimes nuanced and cannot be clearly labeled as entirely "true" or "false" -- as is frequently the case with scientific and political claims. However, a claim (e.g., "vaccine A is better than vaccine B") can be dissected into its integral aspects and sub-aspects (e.g., efficacy, safety, distribution), which are individually easier to validate. This enables a more comprehensive, structured response that provides a well-rounded perspective on a given problem while also allowing the reader to prioritize specific angles of interest within the claim (e.g., safety towards children). Thus, we propose ClaimSpect, a retrieval-augmented generation-based framework for automatically constructing a hierarchy of aspects typically considered when addressing a claim and enriching them with corpus-specific perspectives. This structure hierarchically partitions an input corpus to retrieve relevant segments, which assist in discovering new sub-aspects. Moreover, these segments enable the discovery of varying perspectives towards an aspect of the claim (e.g., support, neutral, or oppose) and their respective prevalence (e.g., "how many biomedical papers believe vaccine A is more transportable than B?"). We apply ClaimSpect to a wide variety of real-world scientific and political claims featured in our constructed dataset, showcasing its robustness and accuracy in deconstructing a nuanced claim and representing perspectives within a corpus. Through real-world case studies and human evaluation, we validate its effectiveness over multiple baselines. 

---
# Sheet Music Benchmark: Standardized Optical Music Recognition Evaluation 

**Authors**: Juan C. Martinez-Sevilla, Joan Cerveto-Serrano, Noelia Luna, Greg Chapman, Craig Sapp, David Rizo, Jorge Calvo-Zaragoza  

**Link**: [PDF](https://arxiv.org/pdf/2506.10488)  

**Abstract**: In this work, we introduce the Sheet Music Benchmark (SMB), a dataset of six hundred and eighty-five pages specifically designed to benchmark Optical Music Recognition (OMR) research. SMB encompasses a diverse array of musical textures, including monophony, pianoform, quartet, and others, all encoded in Common Western Modern Notation using the Humdrum **kern format. Alongside SMB, we introduce the OMR Normalized Edit Distance (OMR-NED), a new metric tailored explicitly for evaluating OMR performance. OMR-NED builds upon the widely-used Symbol Error Rate (SER), offering a fine-grained and detailed error analysis that covers individual musical elements such as note heads, beams, pitches, accidentals, and other critical notation features. The resulting numeric score provided by OMR-NED facilitates clear comparisons, enabling researchers and end-users alike to identify optimal OMR approaches. Our work thus addresses a long-standing gap in OMR evaluation, and we support our contributions with baseline experiments using standardized SMB dataset splits for training and assessing state-of-the-art methods. 

---
# Reasoning RAG via System 1 or System 2: A Survey on Reasoning Agentic Retrieval-Augmented Generation for Industry Challenges 

**Authors**: Jintao Liang, Gang Su, Huifeng Lin, You Wu, Rui Zhao, Ziyue Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.10408)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a powerful framework to overcome the knowledge limitations of Large Language Models (LLMs) by integrating external retrieval with language generation. While early RAG systems based on static pipelines have shown effectiveness in well-structured tasks, they struggle in real-world scenarios requiring complex reasoning, dynamic retrieval, and multi-modal integration. To address these challenges, the field has shifted toward Reasoning Agentic RAG, a paradigm that embeds decision-making and adaptive tool use directly into the retrieval process. In this paper, we present a comprehensive review of Reasoning Agentic RAG methods, categorizing them into two primary systems: predefined reasoning, which follows fixed modular pipelines to boost reasoning, and agentic reasoning, where the model autonomously orchestrates tool interaction during inference. We analyze representative techniques under both paradigms, covering architectural design, reasoning strategies, and tool coordination. Finally, we discuss key research challenges and propose future directions to advance the flexibility, robustness, and applicability of reasoning agentic RAG systems. Our collection of the relevant research has been organized into a this https URL. 

---
# TableRAG: A Retrieval Augmented Generation Framework for Heterogeneous Document Reasoning 

**Authors**: Xiaohan Yu, Pu Jian, Chong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.10380)  

**Abstract**: Retrieval-Augmented Generation (RAG) has demonstrated considerable effectiveness in open-domain question answering. However, when applied to heterogeneous documents, comprising both textual and tabular components, existing RAG approaches exhibit critical limitations. The prevailing practice of flattening tables and chunking strategies disrupts the intrinsic tabular structure, leads to information loss, and undermines the reasoning capabilities of LLMs in multi-hop, global queries. To address these challenges, we propose TableRAG, an hybrid framework that unifies textual understanding and complex manipulations over tabular data. TableRAG iteratively operates in four steps: context-sensitive query decomposition, text retrieval, SQL programming and execution, and compositional intermediate answer generation. We also develop HeteQA, a novel benchmark designed to evaluate the multi-hop heterogeneous reasoning capabilities. Experimental results demonstrate that TableRAG consistently outperforms existing baselines on both public datasets and our HeteQA, establishing a new state-of-the-art for heterogeneous document question answering. We release TableRAG at this https URL. 

---
# A quantum semantic framework for natural language processing 

**Authors**: Christopher J. Agostino, Quan Le Thien, Molly Apsel, Denizhan Pak, Elina Lesyk, Ashabari Majumdar  

**Link**: [PDF](https://arxiv.org/pdf/2506.10077)  

**Abstract**: Semantic degeneracy represents a fundamental property of natural language that extends beyond simple polysemy to encompass the combinatorial explosion of potential interpretations that emerges as semantic expressions increase in complexity. Large Language Models (LLMs) and other modern NLP systems face inherent limitations precisely because they operate within natural language itself, making them subject to the same interpretive constraints imposed by semantic degeneracy. In this work, we argue using Kolmogorov complexity that as an expression's complexity grows, the likelihood of any interpreting agent (human or LLM-powered AI) recovering the single intended meaning vanishes. This computational intractability suggests the classical view that linguistic forms possess meaning in and of themselves is flawed. We alternatively posit that meaning is instead actualized through an observer-dependent interpretive act. To test this, we conducted a semantic Bell inequality test using diverse LLM agents as ``computational cognitive systems'' to interpret ambiguous word pairs under varied contextual settings. Across several independent experiments, we found average CHSH expectation values ranging from 1.2 to 2.8, with several runs yielding values (e.g., 2.3-2.4) that significantly violate the classical boundary ($|S|\leq2$). This demonstrates that linguistic interpretation under ambiguity can exhibit non-classical contextuality, consistent with results from human cognition experiments. These results inherently imply that classical frequentist-based analytical approaches for natural language are necessarily lossy. Instead, we propose that Bayesian-style repeated sampling approaches can provide more practically useful and appropriate characterizations of linguistic meaning in context. 

---
