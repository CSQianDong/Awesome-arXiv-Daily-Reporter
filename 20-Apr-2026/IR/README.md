# Beyond One-Size-Fits-All: Adaptive Test-Time Augmentation for Sequential Recommendation 

**Authors**: Xibo Li, Liang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.16121)  

**Abstract**: Test-time augmentation (TTA) has become a promising approach for mitigating data sparsity in sequential recommendation by improving inference accuracy without requiring costly model retraining. However, existing TTA methods typically rely on uniform, user-agnostic augmentation strategies. We show that this "one-size-fits-all" design is inherently suboptimal, as it neglects substantial behavioral heterogeneity across users, and empirically demonstrate that the optimal augmentation operators vary significantly across user sequences with different characteristics for the first time. To address this limitation, we propose AdaTTA, a plug-and-play reinforcement learning-based adaptive inference framework that learns to select sequence-specific augmentation operators on a per-sequence basis. We formulate augmentation selection as a Markov Decision Process and introduce an Actor-Critic policy network with hybrid state representations and a joint macro-rank reward design to dynamically determine the optimal operator for each input user sequence. Extensive experiments on four real-world datasets and two recommendation backbones demonstrate that AdaTTA consistently outperforms the best fixed-strategy baselines, achieving up to 26.31% relative improvement on the Home dataset while incurring only moderate computational overhead 

---
# JFinTEB: Japanese Financial Text Embedding Benchmark 

**Authors**: Masahiro Suzuki, Hiroki Sakaji  

**Link**: [PDF](https://arxiv.org/pdf/2604.15882)  

**Abstract**: We introduce JFinTEB, the first comprehensive benchmark specifically designed for evaluating Japanese financial text embeddings. Existing embedding benchmarks provide limited coverage of language-specific and domain-specific aspects found in Japanese financial texts. Our benchmark encompasses diverse task categories including retrieval and classification tasks that reflect realistic and well-defined financial text processing scenarios. The retrieval tasks leverage instruction-following datasets and financial text generation queries, while classification tasks cover sentiment analysis, document categorization, and domain-specific classification challenges derived from economic survey data. We conduct extensive evaluations across a wide range of embedding models, including Japanese-specific models of various sizes, multilingual models, and commercial embedding services. We publicly release JFinTEB datasets and evaluation framework at this https URL to facilitate future research and provide a standardized evaluation protocol for the Japanese financial text mining community. This work addresses a critical gap in Japanese financial text processing resources and establishes a foundation for advancing domain-specific embedding research. 

---
# UsefulBench: Towards Decision-Useful Information as a Target for Information Retrieval 

**Authors**: Tobias Schimanski, Stefanie Lewandowski, Christian Woerle, Nicola Reichenau, Yauheni Huryn, Markus Leippold  

**Link**: [PDF](https://arxiv.org/pdf/2604.15827)  

**Abstract**: Conventional information retrieval is concerned with identifying the relevance of texts for a given query. Yet, the conventional definition of relevance is dominated by aspects of similarity in texts, leaving unobserved whether the text is truly useful for addressing the query. For instance, when answering whether Paris is larger than Berlin, texts about Paris being in France are relevant (lexical/semantic similarity), but not useful. In this paper, we introduce UsefulBench, a domain-specific dataset curated by three professional analysts labeling whether a text is connected to a query (relevance) or holds practical value in responding to it (usefulness). We show that classic similarity-based information retrieval aligns more strongly with relevance. While LLM-based systems can counteract this bias, we find that domain-specific problems require a high degree of expertise, which current LLMs do not fully incorporate. We explore approaches to (partially) overcome this challenge. However, UsefulBench presents a dataset challenge for targeted information retrieval systems. 

---
# Scattered Hypothesis Generation for Open-Ended Event Forecasting 

**Authors**: He Chang, Zhulin Tao, Lifang Yang, Xianglin Huang, Yunshan Ma  

**Link**: [PDF](https://arxiv.org/pdf/2604.15788)  

**Abstract**: Despite the importance of open-ended event forecasting for risk management, current LLM-based methods predominantly target only the most probable outcomes, neglecting the intrinsic uncertainty of real-world events. To bridge this gap, we advance open-ended event forecasting from pinpoint forecasting to scatter forecasting by introducing the proxy task of hypothesis generation. This paradigm aims to generate an inclusive and diverse set of hypotheses that broadly cover the space of plausible future events. To this end, we propose SCATTER, a reinforcement learning framework that jointly optimizes inclusiveness and diversity of the hypothesis. Specifically, we design a novel hybrid reward that consists of three components: 1) a validity reward that measures semantic alignment with observed events, 2) an intra-group diversity reward to encourage variation within sampled responses, and 3) an inter-group diversity reward to promote exploration across distinct modes. By integrating the validity-gated score into the overall objective, we confine the exploration of wildly diversified outcomes to contextually plausible futures, preventing the mode collapse issue. Experiments on two real-world benchmark datasets, i.e., OpenForecast and OpenEP, demonstrate that SCATTER significantly outperforms strong baselines. Our code is available at this https URL. 

---
# On the Equivalence Between Auto-Regressive Next Token Prediction and Full-Item-Vocabulary Maximum Likelihood Estimation in Generative Recommendation--A Short Note 

**Authors**: Yusheng Huang, Shuang Yang, Zhaojie Liu, Han Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.15739)  

**Abstract**: Generative recommendation (GR) has emerged as a widely adopted paradigm in industrial sequential recommendation. Current GR systems follow a similar pipeline: tokenization for item indexing, next-token prediction as the training objective and auto-regressive decoding for next-item generation. However, existing GR research mainly focuses on architecture design and empirical performance optimization, with few rigorous theoretical explanations for the working mechanism of auto-regressive next-token prediction in recommendation scenarios.
In this work, we formally prove that \textbf{the k-token auto-regressive next-token prediction (AR-NTP) paradigm is strictly mathematically equivalent to full-item-vocabulary maximum likelihood estimation (FV-MLE)}, under the core premise of a bijective mapping between items and their corresponding k-token sequences. We further show that this equivalence holds for both cascaded and parallel tokenizations, the two most widely used schemes in industrial GR systems. Our result provides the first formal theoretical foundation for the dominant industrial GR paradigm, and offers principled guidance for future GR system optimization. 

---
# Intent Propagation Contrastive Collaborative Filtering 

**Authors**: Haojie Li, Junwei Du, Guanfeng Liu, Feng Jiang, Yan Wang, Xiaofang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.15704)  

**Abstract**: Disentanglement techniques used in collaborative filtering uncover interaction intents between nodes, improving the interpretability of node representations and enhancing recommendation performance. However, existing disentanglement methods still face two problems. First, they focus on local structural features derived from direct node interactions and overlook the comprehensive graph structure, which limits disentanglement accuracy. Second, the disentanglement process depends on backpropagation signals derived from recommendation tasks and lacks direct supervision, which may lead to biases and overfitting. To address these issues, we propose the Intent Propagation Contrastive Collaborative Filtering (IPCCF) algorithm. Specifically, we design a double helix message propagation framework to more effectively extract the deep semantic information of nodes, thereby improving the model's understanding of interactions between nodes. We also develop an intent message propagation method that incorporates graph structure information into the disentanglement process, thereby expanding the consideration scope of disentanglement. In addition, contrastive learning techniques are employed to align node representations derived from structure and intents, providing direct supervision for the disentanglement process, mitigating biases, and enhancing the model's robustness to overfitting. Experiments on three real data graphs illustrate the superiority of the proposed approach. 

---
# Sample Is Feature: Beyond Item-Level, Toward Sample-Level Tokens for Unified Large Recommender Models 

**Authors**: Shuli Wang, Junwei Yin, Changhao Li, Senjie Kou, Chi Wang, Yinqiu Huang, Yinhua Zhu, Haitao Wang, Xingxing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.15650)  

**Abstract**: Scaling industrial recommender models has followed two parallel paradigms: \textbf{sample information scaling} -- enriching the information content of each training sample through deeper and longer behavior sequences -- and \textbf{model capacity scaling} -- unifying sequence modeling and feature interaction within a single Transformer backbone. However, these two paradigms still face two structural limitations. Firstly, sample information scaling methods encode only a subset of each historical interaction into the sequence token, leaving the majority of the original sample context unexploited and precluding the modeling of sample-level, time-varying features. Secondly, model capacity scaling methods are inherently constrained by the structural heterogeneity between sequential and non-sequential features, preventing the model from fully realizing its representational capacity.
To address these issues, we propose \textbf{SIF} (\emph{Sample Is Feature}), which encodes each historical Raw Sample directly into the sequence token -- maximally preserving sample information while simultaneously resolving the heterogeneity between sequential and non-sequential features. SIF consists of two key components. The \textbf{Sample Tokenizer} quantizes each historical Raw Sample into a Token Sample via hierarchical group-adaptive quantization (HGAQ), enabling full sample-level context to be incorporated into the sequence efficiently. The \textbf{SIF-Mixer} then performs deep feature interaction over the homogeneous sample representations via token-level and sample-level mixing, fully unleashing the model's representational capacity. Extensive experiments on a large-scale industrial dataset validate SIF's effectiveness, and we have successfully deployed SIF on the Meituan food delivery platform. 

---
# Rethinking the Necessity of Adaptive Retrieval-Augmented Generation through the Lens of Adaptive Listwise Ranking 

**Authors**: Jun Feng, Jiahui Tang, Zhicheng He, Hang Lv, Hongchao Gu, Hao Wang, Xuezhi Yang, Shuai Fang  

**Link**: [PDF](https://arxiv.org/pdf/2604.15621)  

**Abstract**: Adaptive Retrieval-Augmented Generation aims to mitigate the interference of extraneous noise by dynamically determining the necessity of retrieving supplementary passages. However, as Large Language Models evolve with increasing robustness to noise, the necessity of adaptive retrieval warrants re-evaluation. In this paper, we rethink this necessity and propose AdaRankLLM, a novel adaptive retrieval framework. To effectively verify the necessity of adaptive listwise reranking, we first develop an adaptive ranker employing a zero-shot prompt with a passage dropout mechanism, and compare its generation outcomes against static fixed-depth retrieval strategies. Furthermore, to endow smaller open-source LLMs with this precise listwise ranking and adaptive filtering capability, we introduce a two-stage progressive distillation paradigm enhanced by data sampling and augmentation techniques. Extensive experiments across three datasets and eight LLMs demonstrate that AdaRankLLM consistently achieves optimal performance in most scenarios with significantly reduced context overhead. Crucially, our analysis reveals a role shift in adaptive retrieval: it functions as a critical noise filter for weaker models to overcome their limitations, while serving as a cost-effective efficiency optimizer for stronger reasoning models. 

---
# BioHiCL: Hierarchical Multi-Label Contrastive Learning for Biomedical Retrieval with MeSH Labels 

**Authors**: Mengfei Lan, Lecheng Zheng, Halil Kilicoglu  

**Link**: [PDF](https://arxiv.org/pdf/2604.15591)  

**Abstract**: Effective biomedical information retrieval requires modeling domain semantics and hierarchical relationships among biomedical texts. Existing biomedical generative retrievers build on coarse binary relevance signals, limiting their ability to capture semantic overlap. We propose BioHiCL (Biomedical Retrieval with Hierarchical Multi-Label Contrastive Learning), which leverages hierarchical MeSH annotations to provide structured supervision for multi-label contrastive learning. Our models, BioHiCL-Base (0.1B) and BioHiCL-Large (0.3B), achieve promising performance on biomedical retrieval, sentence similarity, and question answering tasks, while remaining computationally efficient for deployment. 

---
# Learning Behaviorally Grounded Item Embeddings via Personalized Temporal Contexts 

**Authors**: Rafael T. Sereicikas, Pedro R. Pires, Gregorio F. Azevedo, Tiago A. Almeida  

**Link**: [PDF](https://arxiv.org/pdf/2604.15581)  

**Abstract**: Effective user modeling requires distinguishing between short-term and long-term preference evolution. While item embeddings have become a key component of recommender systems, standard approaches like Item2Vec treat user histories as unordered sets (bag-of-items), implicitly assuming that interactions separated by minutes are as semantically related as those separated by months. This simplification flattens the rich temporal structure of user behavior, obscuring the distinction between coherent consumption sessions and gradual interest drifts. In this work, we introduce TAI2Vec (Time-Aware Item-to-Vector), a family of lightweight embedding models that integrates temporal proximity directly into the representation learning process. Unlike approaches that apply global time constraints, TAI2Vec is user-adaptive, tailoring its temporal definitions to individual interaction paces. We propose two complementary strategies: TAI2Vec-Disc, which utilizes personalized anomaly detection to dynamically segment interactions into semantic sessions, and TAI2Vec-Cont, which employs continuous, user-specific decay functions to weigh item relationships based on their relative temporal distance. Experimental results across eight diverse datasets demonstrate that TAI2Vec consistently produces more accurate and behaviorally grounded representations than static baselines, achieving competitive or superior performance in over 80% of the datasets, with improvements of up to 135%. The source code is publicly available at this https URL. 

---
# Collaborative Filtering Through Weighted Similarities of User and Item Embeddings 

**Authors**: Pedro R. Pires, Rafael T. Sereicikas, Gregorio F. Azevedo, Tiago A. Almeida  

**Link**: [PDF](https://arxiv.org/pdf/2604.15573)  

**Abstract**: In recent years, neural networks and other complex models have dominated recommender systems, often setting new benchmarks for state-of-the-art performance. Yet, despite these advancements, award-winning research has demonstrated that traditional matrix factorization methods can remain competitive, offering simplicity and reduced computational overhead. Hybrid models, which combine matrix factorization with newer techniques, are increasingly employed to harness the strengths of multiple approaches. This paper proposes a novel ensemble method that unifies user-item and item-item recommendations through a weighted similarity framework to deliver top-N recommendations. Our approach is distinctive in its use of shared user and item embeddings for both recommendation strategies, simplifying the architecture and enhancing computational efficiency. Extensive experiments across multiple datasets show that our method achieves competitive performance and is robust in varying scenarios that favor either user-item or item-item recommendations. Additionally, by eliminating the need for embedding-specific fine-tuning, our model allows for the seamless reuse of hyperparameters from the base algorithm without sacrificing performance. This results in a method that is both efficient and easy to implement. Our open-source implementation is available at this https URL. 

---
# vstash: Local-First Hybrid Retrieval with Adaptive Fusion for LLM Agents 

**Authors**: Jayson Steffens  

**Link**: [PDF](https://arxiv.org/pdf/2604.15484)  

**Abstract**: We present **vstash**, a local-first document memory system that combines vector similarity search with full-text keyword matching via Reciprocal Rank Fusion (RRF) and adaptive per-query IDF weighting. All data resides in a single SQLite file using sqlite-vec for approximate nearest neighbor search and FTS5 for keyword matching.
We make four primary contributions. **(1)** Self-supervised embedding refinement via hybrid retrieval disagreement: across 753 BEIR queries on SciFact, NFCorpus, and FiQA, 74.5% produce top-10 disagreement between vector-heavy (vec=0.95, fts=0.05) and FTS-heavy (vec=0.05, fts=0.95) search (per-dataset rates 63.4% / 73.4% / 86.7%, Section 5.2), providing a free training signal without human labels. Fine-tuning BGE-small (33M params) with MultipleNegativesRankingLoss on 76K disagreement triples improves NDCG@10 on all 5 BEIR datasets (up to +19.5% on NFCorpus vs. BGE-small base RRF, Table 6). On 3 of 5 datasets, under different preprocessing, the tuned 33M-parameter pipeline matches or exceeds published ColBERTv2 results (110M params) and an untrained BGE-base (110M); on FiQA and ArguAna it underperforms ColBERTv2 (Section 5.5). **(2)** Adaptive RRF with per-query IDF weighting improves NDCG@10 on all 5 BEIR datasets versus fixed weights (up to +21.4% on ArguAna), achieving 0.7263 on SciFact with BGE-small. **(3)** A negative result on post-RRF scoring: frequency+decay, history-augmented recall, and cross-encoder reranking all failed to improve NDCG. **(4)** A production-grade substrate with integrity checking, schema versioning, ranking diagnostics, and a distance-based relevance signal validated on 50,425 relevance-judged queries across the 5 BEIR datasets.
Search latency remains 20.9 ms median at 50K chunks with stable NDCG. The fine-tuned model is published as `Stffens/bge-small-rrf-v2` on HuggingFace. All code, data, and experiments are open-source. 

---
# SIMMER: Cross-Modal Food Image--Recipe Retrieval via MLLM-Based Embedding 

**Authors**: Keisuke Gomi, Keiji Yanai  

**Link**: [PDF](https://arxiv.org/pdf/2604.15628)  

**Abstract**: Cross-modal retrieval between food images and recipe texts is an important task with applications in nutritional management, dietary logging, and cooking assistance. Existing methods predominantly rely on dual-encoder architectures with separate image and text encoders, requiring complex alignment strategies and task-specific network designs to bridge the semantic gap between modalities. In this work, we propose SIMMER (Single Integrated Multimodal Model for Embedding Recipes), which applies Multimodal Large Language Model (MLLM)-based embedding models, specifically VLM2Vec, to this task, replacing the conventional dual-encoder paradigm with a single unified encoder that processes both food images and recipe texts. We design prompt templates tailored to the structured nature of recipes, which consist of a title, ingredients, and cooking instructions, enabling effective embedding generation by the MLLM. We further introduce a component-aware data augmentation strategy that trains the model on both complete and partial recipes, improving robustness to incomplete inputs. Experiments on the Recipe1M dataset demonstrate that SIMMER achieves state-of-the-art performance across both the 1k and 10k evaluation settings, substantially outperforming all prior methods. In particular, our best model improves the 1k image-to-recipe R@1 from 81.8\% to 87.5\% and the 10k image-to-recipe R@1 from 56.5\% to 65.5\% compared to the previous best method. 

---
# OverCite: Add citations in LaTeX without leaving the editor 

**Authors**: Cheyanne Shariat  

**Link**: [PDF](https://arxiv.org/pdf/2604.15366)  

**Abstract**: Adding citations while drafting in LaTeX often requires leaving the editor, searching for a paper in mind, copying its BibTeX entry into the project bibliography, renaming the cite key, and then returning to the sentence. \texttt{OverCite} is an open-source, lightweight tool that lets authors find, select, and insert citations without leaving the writing environment. In Overleaf, \texttt{OverCite} uses rough citation placeholders (e.g., $\texttt{\textbackslash citep\{Perlmutter1999\}}$) and local sentence context to query ADS/SciX-indexed literature, rank likely matches, and insert the selected reference, without leaving the editor. A companion \texttt{VS Code} extension provides the same functionality for local LaTeX projects. The ADS/SciX database includes astronomy, physics, computer science, mathematics, biology, and \emph{all} indexed arXiv e-prints, making \texttt{OverCite} useful across a broad range of scientific disciplines. 

---
# SocialWise: LLM-Agentic Conversation Therapy for Individuals with Autism Spectrum Disorder to Enhance Communication Skills 

**Authors**: Albert Tang  

**Link**: [PDF](https://arxiv.org/pdf/2604.15347)  

**Abstract**: Autism Spectrum Disorder (ASD) affects more than 75 million people worldwide. However, scalable support for practicing everyday conversation is scarce: Low-cost activities such as story reading yield limited improvement. At the same time, effective role-play therapy demands expensive, in-person sessions with specialists. SocialWise bridges this gap through a browser-based application that pairs LLM conversational agents with a therapeutic retrieval augmented generation (RAG) knowledge base. Users select a scenario (e.g., ordering food, joining a group), interact by text or voice, and receive instant, structured feedback on tone, engagement, and alternative phrasing. The SocialWise prototype, implemented with Streamlit, LangChain, and ChromaDB, runs on any computer with internet access, and demonstrates how recent advances in LLM can provide evidence-based, on-demand communication coaching for individuals with ASD. 

---
# To LLM, or Not to LLM: How Designers and Developers Navigate LLMs as Tools or Teammates 

**Authors**: Varad Vishwarupe, Ivan Flechais, Nigel Shadbolt, Marina Jirotka  

**Link**: [PDF](https://arxiv.org/pdf/2604.15344)  

**Abstract**: Large language models (LLMs) are increasingly integrated into design and development workflows, yet decisions about their use are rarely binary or purely technical. We report findings from a constructivist grounded theory study based on interviews with 33 designers and developers across three large technology organisations. Rather than evaluating LLMs solely by capability, participants reasoned about the role an LLM could occupy within a workflow and how that role would interact with existing structures of responsibility and organisational accountability. When LLMs were framed as tools under clear human control, their use was typically acceptable and could be integrated within existing governance structures. When framed as teammates with shared or ambiguous agency, practitioners expressed hesitation, particularly when responsibility for outcomes could not be clearly justified. At the same time, participants also described productive teammate configurations in which LLMs supported collaborative reasoning while remaining embedded within explicit oversight structures. We identify tool and teammate framings as recurring ways in which designers and developers position LLMs relative to human work and present an analytic rubric describing how role framing shapes decision authority, accountability ownership, oversight strategies, and organisational acceptability. By foregrounding design-time reasoning, this work reframes To LLM or Not to LLM as a sociotechnical positioning problem that emerges during system design rather than during post-deployment evaluation. 

---
