# NExT-Search: Rebuilding User Feedback Ecosystem for Generative AI Search 

**Authors**: Sunhao Dai, Wenjie Wang, Liang Pang, Jun Xu, See-Kiong Ng, Ji-Rong Wen, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2505.14680)  

**Abstract**: Generative AI search is reshaping information retrieval by offering end-to-end answers to complex queries, reducing users' reliance on manually browsing and summarizing multiple web pages. However, while this paradigm enhances convenience, it disrupts the feedback-driven improvement loop that has historically powered the evolution of traditional Web search. Web search can continuously improve their ranking models by collecting large-scale, fine-grained user feedback (e.g., clicks, dwell time) at the document level. In contrast, generative AI search operates through a much longer search pipeline, spanning query decomposition, document retrieval, and answer generation, yet typically receives only coarse-grained feedback on the final answer. This introduces a feedback loop disconnect, where user feedback for the final output cannot be effectively mapped back to specific system components, making it difficult to improve each intermediate stage and sustain the feedback loop. In this paper, we envision NExT-Search, a next-generation paradigm designed to reintroduce fine-grained, process-level feedback into generative AI search. NExT-Search integrates two complementary modes: User Debug Mode, which allows engaged users to intervene at key stages; and Shadow User Mode, where a personalized user agent simulates user preferences and provides AI-assisted feedback for less interactive users. Furthermore, we envision how these feedback signals can be leveraged through online adaptation, which refines current search outputs in real-time, and offline update, which aggregates interaction logs to periodically fine-tune query decomposition, retrieval, and generation models. By restoring human control over key stages of the generative AI search pipeline, we believe NExT-Search offers a promising direction for building feedback-rich AI search systems that can evolve continuously alongside human feedback. 

---
# R2MED: A Benchmark for Reasoning-Driven Medical Retrieval 

**Authors**: Lei Li, Xiao Zhou, Zheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.14558)  

**Abstract**: Current medical retrieval benchmarks primarily emphasize lexical or shallow semantic similarity, overlooking the reasoning-intensive demands that are central to clinical decision-making. In practice, physicians often retrieve authoritative medical evidence to support diagnostic hypotheses. Such evidence typically aligns with an inferred diagnosis rather than the surface form of a patient's symptoms, leading to low lexical or semantic overlap between queries and relevant documents. To address this gap, we introduce R2MED, the first benchmark explicitly designed for reasoning-driven medical retrieval. It comprises 876 queries spanning three tasks: Q&A reference retrieval, clinical evidence retrieval, and clinical case retrieval. These tasks are drawn from five representative medical scenarios and twelve body systems, capturing the complexity and diversity of real-world medical information needs. We evaluate 15 widely-used retrieval systems on R2MED and find that even the best model achieves only 31.4 nDCG@10, demonstrating the benchmark's difficulty. Classical re-ranking and generation-augmented retrieval methods offer only modest improvements. Although large reasoning models improve performance via intermediate inference generation, the best results still peak at 41.4 nDCG@10. These findings underscore a substantial gap between current retrieval techniques and the reasoning demands of real clinical tasks. We release R2MED as a challenging benchmark to foster the development of next-generation medical retrieval systems with enhanced reasoning capabilities. Data and code are available at this https URL 

---
# Rank-K: Test-Time Reasoning for Listwise Reranking 

**Authors**: Eugene Yang, Andrew Yates, Kathryn Ricci, Orion Weller, Vivek Chari, Benjamin Van Durme, Dawn Lawrie  

**Link**: [PDF](https://arxiv.org/pdf/2505.14432)  

**Abstract**: Retrieve-and-rerank is a popular retrieval pipeline because of its ability to make slow but effective rerankers efficient enough at query time by reducing the number of comparisons. Recent works in neural rerankers take advantage of large language models for their capability in reasoning between queries and passages and have achieved state-of-the-art retrieval effectiveness. However, such rerankers are resource-intensive, even after heavy optimization. In this work, we introduce Rank-K, a listwise passage reranking model that leverages the reasoning capability of the reasoning language model at query time that provides test time scalability to serve hard queries. We show that Rank-K improves retrieval effectiveness by 23\% over the RankZephyr, the state-of-the-art listwise reranker, when reranking a BM25 initial ranked list and 19\% when reranking strong retrieval results by SPLADE-v3. Since Rank-K is inherently a multilingual model, we found that it ranks passages based on queries in different languages as effectively as it does in monolingual retrieval. 

---
# Taming Recommendation Bias with Causal Intervention on Evolving Personal Popularity 

**Authors**: Shiyin Tan, Dongyuan Li, Renhe Jiang, Zhen Wang, Xingtong Yu, Manabu Okumura  

**Link**: [PDF](https://arxiv.org/pdf/2505.14310)  

**Abstract**: Popularity bias occurs when popular items are recommended far more frequently than they should be, negatively impacting both user experience and recommendation accuracy. Existing debiasing methods mitigate popularity bias often uniformly across all users and only partially consider the time evolution of users or items. However, users have different levels of preference for item popularity, and this preference is evolving over time. To address these issues, we propose a novel method called CausalEPP (Causal Intervention on Evolving Personal Popularity) for taming recommendation bias, which accounts for the evolving personal popularity of users. Specifically, we first introduce a metric called {Evolving Personal Popularity} to quantify each user's preference for popular items. Then, we design a causal graph that integrates evolving personal popularity into the conformity effect, and apply deconfounded training to mitigate the popularity bias of the causal graph. During inference, we consider the evolution consistency between users and items to achieve a better recommendation. Empirical studies demonstrate that CausalEPP outperforms baseline methods in reducing popularity bias while improving recommendation accuracy. 

---
# The Limits of Graph Samplers for Training Inductive Recommender Systems: Extended results 

**Authors**: Theis E. Jendal, Matteo Lissandrini, Peter Dolog, Katja Hose  

**Link**: [PDF](https://arxiv.org/pdf/2505.14241)  

**Abstract**: Inductive Recommender Systems are capable of recommending for new users and with new items thus avoiding the need to retrain after new data reaches the system. However, these methods are still trained on all the data available, requiring multiple days to train a single model, without counting hyperparameter tuning. In this work we focus on graph-based recommender systems, i.e., systems that model the data as a heterogeneous network. In other applications, graph sampling allows to study a subgraph and generalize the findings to the original graph. Thus, we investigate the applicability of sampling techniques for this task. We test on three real world datasets, with three state-of-the-art inductive methods, and using six different sampling methods. We find that its possible to maintain performance using only 50% of the training data with up to 86% percent decrease in training time; however, using less training data leads to far worse performance. Further, we find that when it comes to data for recommendations, graph sampling should also account for the temporal dimension. Therefore, we find that if higher data reduction is needed, new graph based sampling techniques should be studied and new inductive methods should be designed. 

---
# Bridge the Gap between Past and Future: Siamese Model Optimization for Context-Aware Document Ranking 

**Authors**: Songhao Wu, Quan Tu, Mingjie Zhong, Hong Liu, Jia Xu, Jinjie Gu, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2505.14180)  

**Abstract**: In the realm of information retrieval, users often engage in multi-turn interactions with search engines to acquire information, leading to the formation of sequences of user feedback behaviors. Leveraging the session context has proven to be beneficial for inferring user search intent and document ranking. A multitude of approaches have been proposed to exploit in-session context for improved document ranking. Despite these advances, the limitation of historical session data for capturing evolving user intent remains a challenge. In this work, we explore the integration of future contextual information into the session context to enhance document ranking. We present the siamese model optimization framework, comprising a history-conditioned model and a future-aware model. The former processes only the historical behavior sequence, while the latter integrates both historical and anticipated future behaviors. Both models are trained collaboratively using the supervised labels and pseudo labels predicted by the other. The history-conditioned model, referred to as ForeRanker, progressively learns future-relevant information to enhance ranking, while it singly uses historical session at inference time. To mitigate inconsistencies during training, we introduce the peer knowledge distillation method with a dynamic gating mechanism, allowing models to selectively incorporate contextual information. Experimental results on benchmark datasets demonstrate the effectiveness of our ForeRanker, showcasing its superior performance compared to existing methods. 

---
# Process vs. Outcome Reward: Which is Better for Agentic RAG Reinforcement Learning 

**Authors**: Wenlin Zhang, Xiangyang Li, Kuicai Dong, Yichao Wang, Pengyue Jia, Xiaopeng Li, Yingyi Zhang, Derong Xu, Zhaocheng Du, Huifeng Guo, Ruiming Tang, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.14069)  

**Abstract**: Retrieval-augmented generation (RAG) enhances the text generation capabilities of large language models (LLMs) by integrating external knowledge and up-to-date information. However, traditional RAG systems are limited by static workflows and lack the adaptability required for multistep reasoning and complex task management. To address these limitations, agentic RAG systems (e.g., DeepResearch) have been proposed, enabling dynamic retrieval strategies, iterative context refinement, and adaptive workflows for handling complex search queries beyond the capabilities of conventional RAG. Recent advances, such as Search-R1, have demonstrated promising gains using outcome-based reinforcement learning, where the correctness of the final answer serves as the reward signal. Nevertheless, such outcome-supervised agentic RAG methods face challenges including low exploration efficiency, gradient conflict, and sparse reward signals. To overcome these challenges, we propose to utilize fine-grained, process-level rewards to improve training stability, reduce computational costs, and enhance efficiency. Specifically, we introduce a novel method ReasonRAG that automatically constructs RAG-ProGuide, a high-quality dataset providing process-level rewards for (i) query generation, (ii) evidence extraction, and (iii) answer generation, thereby enhancing model inherent capabilities via process-supervised reinforcement learning. With the process-level policy optimization, the proposed framework empowers LLMs to autonomously invoke search, generate queries, extract relevant evidence, and produce final answers. Compared to existing approaches such as Search-R1 and traditional RAG systems, ReasonRAG, leveraging RAG-ProGuide, achieves superior performance on five benchmark datasets using only 5k training instances, significantly fewer than the 90k training instances required by Search-R1. 

---
# Field Matters: A lightweight LLM-enhanced Method for CTR Prediction 

**Authors**: Yu Cui, Feng Liu, Jiawei Chen, Xingyu Lou, Changwang Zhang, Jun Wang, Yuegang Sun, Xiaohu Yang, Can Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14057)  

**Abstract**: Click-through rate (CTR) prediction is a fundamental task in modern recommender systems. In recent years, the integration of large language models (LLMs) has been shown to effectively enhance the performance of traditional CTR methods. However, existing LLM-enhanced methods often require extensive processing of detailed textual descriptions for large-scale instances or user/item entities, leading to substantial computational overhead. To address this challenge, this work introduces LLaCTR, a novel and lightweight LLM-enhanced CTR method that employs a field-level enhancement paradigm. Specifically, LLaCTR first utilizes LLMs to distill crucial and lightweight semantic knowledge from small-scale feature fields through self-supervised field-feature fine-tuning. Subsequently, it leverages this field-level semantic knowledge to enhance both feature representation and feature interactions. In our experiments, we integrate LLaCTR with six representative CTR models across four datasets, demonstrating its superior performance in terms of both effectiveness and efficiency compared to existing LLM-enhanced methods. Our code is available at this https URL. 

---
# DIFF: Dual Side-Information Filtering and Fusion for Sequential Recommendation 

**Authors**: Hye-young Kim, Minjin Choi, Sunkyung Lee, Ilwoong Baek, Jongwuk Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.13974)  

**Abstract**: Side-information Integrated Sequential Recommendation (SISR) benefits from auxiliary item information to infer hidden user preferences, which is particularly effective for sparse interactions and cold-start scenarios. However, existing studies face two main challenges. (i) They fail to remove noisy signals in item sequence and (ii) they underutilize the potential of side-information integration. To tackle these issues, we propose a novel SISR model, Dual Side-Information Filtering and Fusion (DIFF), which employs frequency-based noise filtering and dual multi-sequence fusion. Specifically, we convert the item sequence to the frequency domain to filter out noisy short-term fluctuations in user interests. We then combine early and intermediate fusion to capture diverse relationships across item IDs and attributes. Thanks to our innovative filtering and fusion strategy, DIFF is more robust in learning subtle and complex item correlations in the sequence. DIFF outperforms state-of-the-art SISR models, achieving improvements of up to 14.1% and 12.5% in Recall@20 and NDCG@20 across four benchmark datasets. 

---
# Benchmarking the Myopic Trap: Positional Bias in Information Retrieval 

**Authors**: Ziyang Zeng, Dun Zhang, Jiacheng Li, Panxiang Zou, Yuqing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13950)  

**Abstract**: This study investigates a specific form of positional bias, termed the Myopic Trap, where retrieval models disproportionately attend to the early parts of documents while overlooking relevant information that appears later. To systematically quantify this phenomenon, we propose a semantics-preserving evaluation framework that repurposes the existing NLP datasets into position-aware retrieval benchmarks. By evaluating the SOTA models of full retrieval pipeline, including BM25, embedding models, ColBERT-style late-interaction models, and reranker models, we offer a broader empirical perspective on positional bias than prior work. Experimental results show that embedding models and ColBERT-style models exhibit significant performance degradation when query-related content is shifted toward later positions, indicating a pronounced head bias. Notably, under the same training configuration, ColBERT-style approach show greater potential for mitigating positional bias compared to the traditional single-vector approach. In contrast, BM25 and reranker models remain largely unaffected by such perturbations, underscoring their robustness to positional bias. Code and data are publicly available at: this http URL. 

---
# TranSUN: A Preemptive Paradigm to Eradicate Retransformation Bias Intrinsically from Regression Models in Recommender Systems 

**Authors**: Jiahao Yu, Haozhuang Liu, Yeqiu Yang, Lu Chen, Wu Jian, Yuning Jiang, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.13881)  

**Abstract**: Regression models are crucial in recommender systems. However, retransformation bias problem has been conspicuously neglected within the community. While many works in other fields have devised effective bias correction methods, all of them are post-hoc cures externally to the model, facing practical challenges when applied to real-world recommender systems. Hence, we propose a preemptive paradigm to eradicate the bias intrinsically from the models via minor model refinement. Specifically, a novel TranSUN method is proposed with a joint bias learning manner to offer theoretically guaranteed unbiasedness under empirical superior convergence. It is further generalized into a novel generic regression model family, termed Generalized TranSUN (GTS), which not only offers more theoretical insights but also serves as a generic framework for flexibly developing various bias-free models. Comprehensive experimental results demonstrate the superiority of our methods across data from various domains, which have been successfully deployed in two real-world industrial recommendation scenarios, i.e. product and short video recommendation scenarios in Guess What You Like business domain in the homepage of Taobao App (a leading e-commerce platform), to serve the major online traffic. Codes will be released after this paper is published. 

---
# LLM-Based Compact Reranking with Document Features for Scientific Retrieval 

**Authors**: Runchu Tian, Xueqiang Xu, Bowen Jin, SeongKu Kang, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.13757)  

**Abstract**: Scientific retrieval is essential for advancing academic discovery. Within this process, document reranking plays a critical role by refining first-stage retrieval results. However, large language model (LLM) listwise reranking faces unique challenges in the scientific domain. First-stage retrieval is often suboptimal in the scientific domain, so relevant documents are ranked lower. Moreover, conventional listwise reranking uses the full text of candidate documents in the context window, limiting the number of candidates that can be considered. As a result, many relevant documents are excluded before reranking, which constrains overall retrieval performance. To address these challenges, we explore compact document representations based on semantic features such as categories, sections, and keywords, and propose a training-free, model-agnostic reranking framework for scientific retrieval called CoRank. The framework involves three stages: (i) offline extraction of document-level features, (ii) coarse reranking using these compact representations, and (iii) fine-grained reranking on full texts of the top candidates from stage (ii). This hybrid design provides a high-level abstraction of document semantics, expands candidate coverage, and retains critical details required for precise ranking. Experiments on LitSearch and CSFCube show that CoRank significantly improves reranking performance across different LLM backbones, increasing nDCG@10 from 32.0 to 39.7. Overall, these results highlight the value of information extraction for reranking in scientific retrieval. 

---
# RAR: Setting Knowledge Tripwires for Retrieval Augmented Rejection 

**Authors**: Tommaso Mario Buonocore, Enea Parimbelli  

**Link**: [PDF](https://arxiv.org/pdf/2505.13581)  

**Abstract**: Content moderation for large language models (LLMs) remains a significant challenge, requiring flexible and adaptable solutions that can quickly respond to emerging threats. This paper introduces Retrieval Augmented Rejection (RAR), a novel approach that leverages a retrieval-augmented generation (RAG) architecture to dynamically reject unsafe user queries without model retraining. By strategically inserting and marking malicious documents into the vector database, the system can identify and reject harmful requests when these documents are retrieved. Our preliminary results show that RAR achieves comparable performance to embedded moderation in LLMs like Claude 3.5 Sonnet, while offering superior flexibility and real-time customization capabilities, a fundamental feature to timely address critical vulnerabilities. This approach introduces no architectural changes to existing RAG systems, requiring only the addition of specially crafted documents and a simple rejection mechanism based on retrieval results. 

---
# AMAQA: A Metadata-based QA Dataset for RAG Systems 

**Authors**: Davide Bruni, Marco Avvenuti, Nicola Tonellotto, Maurizio Tesconi  

**Link**: [PDF](https://arxiv.org/pdf/2505.13557)  

**Abstract**: Retrieval-augmented generation (RAG) systems are widely used in question-answering (QA) tasks, but current benchmarks lack metadata integration, hindering evaluation in scenarios requiring both textual data and external information. To address this, we present AMAQA, a new open-access QA dataset designed to evaluate tasks combining text and metadata. The integration of metadata is especially important in fields that require rapid analysis of large volumes of data, such as cybersecurity and intelligence, where timely access to relevant information is critical. AMAQA includes about 1.1 million English messages collected from 26 public Telegram groups, enriched with metadata such as timestamps, topics, emotional tones, and toxicity indicators, which enable precise and contextualized queries by filtering documents based on specific criteria. It also includes 450 high-quality QA pairs, making it a valuable resource for advancing research on metadata-driven QA and RAG systems. To the best of our knowledge, AMAQA is the first single-hop QA benchmark to incorporate metadata and labels such as topics covered in the messages. We conduct extensive tests on the benchmark, establishing a new standard for future research. We show that leveraging metadata boosts accuracy from 0.12 to 0.61, highlighting the value of structured context. Building on this, we explore several strategies to refine the LLM input by iterating over provided context and enriching it with noisy documents, achieving a further 3-point gain over the best baseline and a 14-point improvement over simple metadata filtering. The dataset is available at this https URL 

---
# JIR-Arena: The First Benchmark Dataset for Just-in-time Information Recommendation 

**Authors**: Ke Yang, Kevin Ros, Shankar Kumar Senthil Kumar, ChengXiang Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2505.13550)  

**Abstract**: Just-in-time Information Recommendation (JIR) is a service designed to deliver the most relevant information precisely when users need it, , addressing their knowledge gaps with minimal effort and boosting decision-making and efficiency in daily life. Advances in device-efficient deployment of foundation models and the growing use of intelligent wearable devices have made always-on JIR assistants feasible. However, there has been no systematic effort to formally define JIR tasks or establish evaluation frameworks. To bridge this gap, we present the first mathematical definition of JIR tasks and associated evaluation metrics. Additionally, we introduce JIR-Arena, a multimodal benchmark dataset featuring diverse, information-request-intensive scenarios to evaluate JIR systems across critical dimensions: i) accurately inferring user information needs, ii) delivering timely and relevant recommendations, and iii) avoiding irrelevant content that may distract users.
Developing a JIR benchmark dataset poses challenges due to subjectivity in estimating user information needs and uncontrollable system variables affecting reproducibility. To address these, JIR-Arena: i) combines input from multiple humans and large AI models to approximate information need distributions; ii) assesses JIR quality through information retrieval outcomes using static knowledge base snapshots; and iii) employs a multi-turn, multi-entity validation framework to improve objectivity and generality. Furthermore, we implement a baseline JIR system capable of processing real-time information streams aligned with user inputs. Our evaluation of this baseline system on JIR-Arena indicates that while foundation model-based JIR systems simulate user needs with reasonable precision, they face challenges in recall and effective content retrieval. To support future research in this new area, we fully release our code and data. 

---
# Know Or Not: a library for evaluating out-of-knowledge base robustness 

**Authors**: Jessica Foo, Pradyumna Shyama Prasad, Shaun Khoo  

**Link**: [PDF](https://arxiv.org/pdf/2505.13545)  

**Abstract**: While the capabilities of large language models (LLMs) have progressed significantly, their use in high-stakes applications have been limited due to risks of hallucination. One key approach in reducing hallucination is retrieval-augmented generation (RAG), but even in such setups, LLMs may still hallucinate when presented with questions outside of the knowledge base. Such behavior is unacceptable in high-stake applications where LLMs are expected to abstain from answering queries it does not have sufficient context on. In this work, we present a novel methodology for systematically evaluating out-of-knowledge base (OOKB) robustness of LLMs (whether LLMs know or do not know) in the RAG setting, without the need for manual annotation of gold standard answers. We implement our methodology in knowornot, an open-source library that enables users to develop their own customized evaluation data and pipelines for OOKB robustness. knowornot comprises four main features. Firstly, it provides a unified, high-level API that streamlines the process of setting up and running robustness benchmarks. Secondly, its modular architecture emphasizes extensibility and flexibility, allowing users to easily integrate their own LLM clients and RAG settings. Thirdly, its rigorous data modeling design ensures experiment reproducibility, reliability and traceability. Lastly, it implements a comprehensive suite of tools for users to customize their pipelines. We demonstrate the utility of knowornot by developing a challenging benchmark, PolicyBench, which spans four Question-Answer (QA) chatbots on government policies, and analyze its OOKB robustness. The source code of knowornot is available this https URL. 

---
# RAGXplain: From Explainable Evaluation to Actionable Guidance of RAG Pipelines 

**Authors**: Dvir Cohen, Lin Burg, Gilad Barkan  

**Link**: [PDF](https://arxiv.org/pdf/2505.13538)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems show promise by coupling large language models with external knowledge, yet traditional RAG evaluation methods primarily report quantitative scores while offering limited actionable guidance for refining these complex pipelines. In this paper, we introduce RAGXplain, an evaluation framework that quantifies RAG performance and translates these assessments into clear insights that clarify the workings of its complex, multi-stage pipeline and offer actionable recommendations. Using LLM reasoning, RAGXplain converts raw scores into coherent narratives identifying performance gaps and suggesting targeted improvements. By providing transparent explanations for AI decision-making, our framework fosters user trust-a key challenge in AI adoption. Our LLM-based metric assessments show strong alignment with human judgments, and experiments on public question-answering datasets confirm that applying RAGXplain's actionable recommendations measurably improves system performance. RAGXplain thus bridges quantitative evaluation and practical optimization, empowering users to understand, trust, and enhance their AI systems. 

---
# Information Extraction from Visually Rich Documents using LLM-based Organization of Documents into Independent Textual Segments 

**Authors**: Aniket Bhattacharyya, Anurag Tripathi, Ujjal Das, Archan Karmakar, Amit Pathak, Maneesh Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2505.13535)  

**Abstract**: Information extraction (IE) from Visually Rich Documents (VRDs) containing layout features along with text is a critical and well-studied task. Specialized non-LLM NLP-based solutions typically involve training models using both textual and geometric information to label sequences/tokens as named entities or answers to specific questions. However, these approaches lack reasoning, are not able to infer values not explicitly present in documents, and do not generalize well to new formats. Generative LLM-based approaches proposed recently are capable of reasoning, but struggle to comprehend clues from document layout especially in previously unseen document formats, and do not show competitive performance in heterogeneous VRD benchmark datasets. In this paper, we propose BLOCKIE, a novel LLM-based approach that organizes VRDs into localized, reusable semantic textual segments called $\textit{semantic blocks}$, which are processed independently. Through focused and more generalizable reasoning,our approach outperforms the state-of-the-art on public VRD benchmarks by 1-3% in F1 scores, is resilient to document formats previously not encountered and shows abilities to correctly extract information not explicitly present in documents. 

---
# LLM-Based User Simulation for Low-Knowledge Shilling Attacks on Recommender Systems 

**Authors**: Shengkang Gu, Jiahao Liu, Dongsheng Li, Guangping Zhang, Mingzhe Han, Hansu Gu, Peng Zhang, Ning Gu, Li Shang, Tun Lu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13528)  

**Abstract**: Recommender systems (RS) are increasingly vulnerable to shilling attacks, where adversaries inject fake user profiles to manipulate system outputs. Traditional attack strategies often rely on simplistic heuristics, require access to internal RS data, and overlook the manipulation potential of textual reviews. In this work, we introduce Agent4SR, a novel framework that leverages Large Language Model (LLM)-based agents to perform low-knowledge, high-impact shilling attacks through both rating and review generation. Agent4SR simulates realistic user behavior by orchestrating adversarial interactions, selecting items, assigning ratings, and crafting reviews, while maintaining behavioral plausibility. Our design includes targeted profile construction, hybrid memory retrieval, and a review attack strategy that propagates target item features across unrelated reviews to amplify manipulation. Extensive experiments on multiple datasets and RS architectures demonstrate that Agent4SR outperforms existing low-knowledge baselines in both effectiveness and stealth. Our findings reveal a new class of emergent threats posed by LLM-driven agents, underscoring the urgent need for enhanced defenses in modern recommender systems. 

---
# Geography-Aware Large Language Models for Next POI Recommendation 

**Authors**: Zhao Liu, Wei Liu, Huajie Zhu, Jianxing Yu, Jian Yin, Wang-Chien Lee, Shun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13526)  

**Abstract**: The next Point-of-Interest (POI) recommendation task aims to predict users' next destinations based on their historical movement data and plays a key role in location-based services and personalized applications. Accurate next POI recommendation depends on effectively modeling geographic information and POI transition relations, which are crucial for capturing spatial dependencies and user movement patterns. While Large Language Models (LLMs) exhibit strong capabilities in semantic understanding and contextual reasoning, applying them to spatial tasks like next POI recommendation remains challenging. First, the infrequent nature of specific GPS coordinates makes it difficult for LLMs to model precise spatial contexts. Second, the lack of knowledge about POI transitions limits their ability to capture potential POI-POI relationships. To address these issues, we propose GA-LLM (Geography-Aware Large Language Model), a novel framework that enhances LLMs with two specialized components. The Geographic Coordinate Injection Module (GCIM) transforms GPS coordinates into spatial representations using hierarchical and Fourier-based positional encoding, enabling the model to understand geographic features from multiple perspectives. The POI Alignment Module (PAM) incorporates POI transition relations into the LLM's semantic space, allowing it to infer global POI relationships and generalize to unseen POIs. Experiments on three real-world datasets demonstrate the state-of-the-art performance of GA-LLM. 

---
# Beyond Retrieval: Joint Supervision and Multimodal Document Ranking for Textbook Question Answering 

**Authors**: Hessa Alawwad, Usman Naseem, Areej Alhothali, Ali Alkhathlan, Amani Jamal  

**Link**: [PDF](https://arxiv.org/pdf/2505.13520)  

**Abstract**: Textbook question answering (TQA) is a complex task, requiring the interpretation of complex multimodal context. Although recent advances have improved overall performance, they often encounter difficulties in educational settings where accurate semantic alignment and task-specific document retrieval are essential. In this paper, we propose a novel approach to multimodal textbook question answering by introducing a mechanism for enhancing semantic representations through multi-objective joint training. Our model, Joint Embedding Training With Ranking Supervision for Textbook Question Answering (JETRTQA), is a multimodal learning framework built on a retriever--generator architecture that uses a retrieval-augmented generation setup, in which a multimodal large language model generates answers. JETRTQA is designed to improve the relevance of retrieved documents in complex educational contexts. Unlike traditional direct scoring approaches, JETRTQA learns to refine the semantic representations of questions and documents through a supervised signal that combines pairwise ranking and implicit supervision derived from answers. We evaluate our method on the CK12-QA dataset and demonstrate that it significantly improves the discrimination between informative and irrelevant documents, even when they are long, complex, and multimodal. JETRTQA outperforms the previous state of the art, achieving a 2.4\% gain in accuracy on the validation set and 11.1\% on the test set. 

---
# An agentic system with reinforcement-learned subsystem improvements for parsing form-like documents 

**Authors**: Ayesha Amjad, Saurav Sthapit, Tahir Qasim Syed  

**Link**: [PDF](https://arxiv.org/pdf/2505.13504)  

**Abstract**: Extracting alphanumeric data from form-like documents such as invoices, purchase orders, bills, and financial documents is often performed via vision (OCR) and learning algorithms or monolithic pipelines with limited potential for systemic improvements. We propose an agentic AI system that leverages Large Language Model (LLM) agents and a reinforcement learning (RL) driver agent to automate consistent, self-improving extraction under LLM inference uncertainty. Our work highlights the limitations of monolithic LLM-based extraction and introduces a modular, multi-agent framework with task-specific prompts and an RL policy of rewards and penalties to guide a meta-prompting agent to learn from past errors and improve prompt-based actor agents. This self-corrective adaptive system handles diverse documents, file formats, layouts, and LLMs, aiming to automate accurate information extraction without the need for human intervention. Results as reported on two benchmark datasets of SOIRE, and CORD, are promising for the agentic AI framework. 

---
# MedEIR: A Specialized Medical Embedding Model for Enhanced Information Retrieval 

**Authors**: Anand Selvadurai, Jasheen Shaik, Girish Chandrasekar, ShriRadhaKrishnan Balamurugan, Eswara Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2505.13482)  

**Abstract**: Embedding models have become essential for retrieval-augmented generation (RAG) tasks, semantic clustering, and text re-ranking. But despite their growing use, many of these come with notable limitations. For example, Jina fails to capture the semantic content of medical documents, while models such as MiniLM often perform poorly on long-form documents. Domain-adapted models, while specialized, often underperform in general-purpose tasks, reducing their overall applicability. General-domain tokenizers often misinterpret medical vocabulary. The limitations of current embedding models, whether in tokenization accuracy, domain comprehension, or handling long sequences, highlight the need for more versatile solutions. In this work, we present MedEIR, a novel embedding model and tokenizer jointly optimized for both medical and general NLP tasks, incorporating ALiBi-based long-context processing to support sequences of up to 8,192 tokens. MedEIR was pre-trained on only 6 billion tokens, significantly fewer than Jina's, followed by fine-tuning on 3 million sentence pairs. MedEIR consistently outperforms Jina V2 and MiniLM across MTEB benchmarks, achieving top scores on ArguAna (55.24), NFCorpus (38.44), MedicalQARetrieval (74.25), SciFact (72.04), and TRECCOVID (79.56). These results highlight the potential of MedEIR as a highly effective embedding model, demonstrating strong performance across both general-purpose and domain-specific tasks and outperforming existing models on multiple benchmarks. 

---
# Two Experts Are All You Need for Steering Thinking: Reinforcing Cognitive Effort in MoE Reasoning Models Without Additional Training 

**Authors**: Mengru Wang, Xingyu Chen, Yue Wang, Zhiwei He, Jiahao Xu, Tian Liang, Qiuzhi Liu, Yunzhi Yao, Wenxuan Wang, Ruotian Ma, Haitao Mi, Ningyu Zhang, Zhaopeng Tu, Xiaolong Li, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.14681)  

**Abstract**: Mixture-of-Experts (MoE) architectures within Large Reasoning Models (LRMs) have achieved impressive reasoning capabilities by selectively activating experts to facilitate structured cognitive processes. Despite notable advances, existing reasoning models often suffer from cognitive inefficiencies like overthinking and underthinking. To address these limitations, we introduce a novel inference-time steering methodology called Reinforcing Cognitive Experts (RICE), designed to improve reasoning performance without additional training or complex heuristics. Leveraging normalized Pointwise Mutual Information (nPMI), we systematically identify specialized experts, termed ''cognitive experts'' that orchestrate meta-level reasoning operations characterized by tokens like ''<think>''. Empirical evaluations with leading MoE-based LRMs (DeepSeek-R1 and Qwen3-235B) on rigorous quantitative and scientific reasoning benchmarks demonstrate noticeable and consistent improvements in reasoning accuracy, cognitive efficiency, and cross-domain generalization. Crucially, our lightweight approach substantially outperforms prevalent reasoning-steering techniques, such as prompt design and decoding constraints, while preserving the model's general instruction-following skills. These results highlight reinforcing cognitive experts as a promising, practical, and interpretable direction to enhance cognitive efficiency within advanced reasoning models. 

---
# Technical Report on classification of literature related to children speech disorder 

**Authors**: Ziang Wang, Amir Aryani  

**Link**: [PDF](https://arxiv.org/pdf/2505.14242)  

**Abstract**: This technical report presents a natural language processing (NLP)-based approach for systematically classifying scientific literature on childhood speech disorders. We retrieved and filtered 4,804 relevant articles published after 2015 from the PubMed database using domain-specific keywords. After cleaning and pre-processing the abstracts, we applied two topic modeling techniques - Latent Dirichlet Allocation (LDA) and BERTopic - to identify latent thematic structures in the corpus. Our models uncovered 14 clinically meaningful clusters, such as infantile hyperactivity and abnormal epileptic behavior. To improve relevance and precision, we incorporated a custom stop word list tailored to speech pathology. Evaluation results showed that the LDA model achieved a coherence score of 0.42 and a perplexity of -7.5, indicating strong topic coherence and predictive performance. The BERTopic model exhibited a low proportion of outlier topics (less than 20%), demonstrating its capacity to classify heterogeneous literature effectively. These results provide a foundation for automating literature reviews in speech-language pathology. 

---
# Enhancing Abstractive Summarization of Scientific Papers Using Structure Information 

**Authors**: Tong Bao, Heng Zhang, Chengzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14179)  

**Abstract**: Abstractive summarization of scientific papers has always been a research focus, yet existing methods face two main challenges. First, most summarization models rely on Encoder-Decoder architectures that treat papers as sequences of words, thus fail to fully capture the structured information inherent in scientific papers. Second, existing research often use keyword mapping or feature engineering to identify the structural information, but these methods struggle with the structural flexibility of scientific papers and lack robustness across different disciplines. To address these challenges, we propose a two-stage abstractive summarization framework that leverages automatic recognition of structural functions within scientific papers. In the first stage, we standardize chapter titles from numerous scientific papers and construct a large-scale dataset for structural function recognition. A classifier is then trained to automatically identify the key structural components (e.g., Background, Methods, Results, Discussion), which provides a foundation for generating more balanced summaries. In the second stage, we employ Longformer to capture rich contextual relationships across sections and generating context-aware summaries. Experiments conducted on two domain-specific scientific paper summarization datasets demonstrate that our method outperforms advanced baselines, and generates more comprehensive summaries. The code and dataset can be accessed at this https URL. 

---
# Unify Graph Learning with Text: Unleashing LLM Potentials for Session Search 

**Authors**: Songhao Wu, Quan Tu, Hong Liu, Jia Xu, Zhongyi Liu, Guannan Zhang, Ran Wang, Xiuying Chen, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2505.14156)  

**Abstract**: Session search involves a series of interactive queries and actions to fulfill user's complex information need. Current strategies typically prioritize sequential modeling for deep semantic understanding, overlooking the graph structure in interactions. While some approaches focus on capturing structural information, they use a generalized representation for documents, neglecting the word-level semantic modeling. In this paper, we propose Symbolic Graph Ranker (SGR), which aims to take advantage of both text-based and graph-based approaches by leveraging the power of recent Large Language Models (LLMs). Concretely, we first introduce a set of symbolic grammar rules to convert session graph into text. This allows integrating session history, interaction process, and task instruction seamlessly as inputs for the LLM. Moreover, given the natural discrepancy between LLMs pre-trained on textual corpora, and the symbolic language we produce using our graph-to-text grammar, our objective is to enhance LLMs' ability to capture graph structures within a textual format. To achieve this, we introduce a set of self-supervised symbolic learning tasks including link prediction, node content generation, and generative contrastive learning, to enable LLMs to capture the topological information from coarse-grained to fine-grained. Experiment results and comprehensive analysis on two benchmark datasets, AOL and Tiangong-ST, confirm the superiority of our approach. Our paradigm also offers a novel and effective methodology that bridges the gap between traditional search strategies and modern LLMs. 

---
# Enhancing Keyphrase Extraction from Academic Articles Using Section Structure Information 

**Authors**: Chengzhi Zhang, Xinyi Yan, Lei Zhao, Yingyi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14149)  

**Abstract**: The exponential increase in academic papers has significantly increased the time required for researchers to access relevant literature. Keyphrase Extraction (KPE) offers a solution to this situation by enabling researchers to efficiently retrieve relevant literature. The current study on KPE from academic articles aims to improve the performance of extraction models through innovative approaches using Title and Abstract as input corpora. However, the semantic richness of keywords is significantly constrained by the length of the abstract. While full-text-based KPE can address this issue, it simultaneously introduces noise, which significantly diminishes KPE performance. To address this issue, this paper utilized the structural features and section texts obtained from the section structure information of academic articles to extract keyphrase from academic papers. The approach consists of two main parts: (1) exploring the effect of seven structural features on KPE models, and (2) integrating the extraction results from all section texts used as input corpora for KPE models via a keyphrase integration algorithm to obtain the keyphrase integration result. Furthermore, this paper also examined the effect of the classification quality of section structure on the KPE performance. The results show that incorporating structural features improves KPE performance, though different features have varying effects on model efficacy. The keyphrase integration approach yields the best performance, and the classification quality of section structure can affect KPE performance. These findings indicate that using the section structure information of academic articles contributes to effective KPE from academic articles. The code and dataset supporting this study are available at this https URL. 

---
# Beyond Chains: Bridging Large Language Models and Knowledge Bases in Complex Question Answering 

**Authors**: Yihua Zhu, Qianying Liu, Akiko Aizawa, Hidetoshi Shimodaira  

**Link**: [PDF](https://arxiv.org/pdf/2505.14099)  

**Abstract**: Knowledge Base Question Answering (KBQA) aims to answer natural language questions using structured knowledge from KBs. While LLM-only approaches offer generalization, they suffer from outdated knowledge, hallucinations, and lack of transparency. Chain-based KG-RAG methods address these issues by incorporating external KBs, but are limited to simple chain-structured questions due to the absence of planning and logical structuring. Inspired by semantic parsing methods, we propose PDRR: a four-stage framework consisting of Predict, Decompose, Retrieve, and Reason. Our method first predicts the question type and decomposes the question into structured triples. Then retrieves relevant information from KBs and guides the LLM as an agent to reason over and complete the decomposed triples. Experimental results demonstrate that PDRR consistently outperforms existing methods across various LLM backbones and achieves superior performance on both chain-structured and non-chain complex questions. 

---
# Disentangled Multi-span Evolutionary Network against Temporal Knowledge Graph Reasoning 

**Authors**: Hao Dong, Ziyue Qiao, Zhiyuan Ning, Qi Hao, Yi Du, Pengyang Wang, Yuanchun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.14020)  

**Abstract**: Temporal Knowledge Graphs (TKGs), as an extension of static Knowledge Graphs (KGs), incorporate the temporal feature to express the transience of knowledge by describing when facts occur. TKG extrapolation aims to infer possible future facts based on known history, which has garnered significant attention in recent years. Some existing methods treat TKG as a sequence of independent subgraphs to model temporal evolution patterns, demonstrating impressive reasoning performance. However, they still have limitations: 1) In modeling subgraph semantic evolution, they usually neglect the internal structural interactions between subgraphs, which are actually crucial for encoding TKGs. 2) They overlook the potential smooth features that do not lead to semantic changes, which should be distinguished from the semantic evolution process. Therefore, we propose a novel Disentangled Multi-span Evolutionary Network (DiMNet) for TKG reasoning. Specifically, we design a multi-span evolution strategy that captures local neighbor features while perceiving historical neighbor semantic information, thus enabling internal interactions between subgraphs during the evolution process. To maximize the capture of semantic change patterns, we design a disentangle component that adaptively separates nodes' active and stable features, used to dynamically control the influence of historical semantics on future evolution. Extensive experiments conducted on four real-world TKG datasets show that DiMNet demonstrates substantial performance in TKG reasoning, and outperforms the state-of-the-art up to 22.7% in MRR. 

---
# Divide by Question, Conquer by Agent: SPLIT-RAG with Question-Driven Graph Partitioning 

**Authors**: Ruiyi Yang, Hao Xue, Imran Razzak, Hakim Hacid, Flora D. Salim  

**Link**: [PDF](https://arxiv.org/pdf/2505.13994)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems empower large language models (LLMs) with external knowledge, yet struggle with efficiency-accuracy trade-offs when scaling to large knowledge graphs. Existing approaches often rely on monolithic graph retrieval, incurring unnecessary latency for simple queries and fragmented reasoning for complex multi-hop questions. To address these challenges, this paper propose SPLIT-RAG, a multi-agent RAG framework that addresses these limitations with question-driven semantic graph partitioning and collaborative subgraph retrieval. The innovative framework first create Semantic Partitioning of Linked Information, then use the Type-Specialized knowledge base to achieve Multi-Agent RAG. The attribute-aware graph segmentation manages to divide knowledge graphs into semantically coherent subgraphs, ensuring subgraphs align with different query types, while lightweight LLM agents are assigned to partitioned subgraphs, and only relevant partitions are activated during retrieval, thus reduce search space while enhancing efficiency. Finally, a hierarchical merging module resolves inconsistencies across subgraph-derived answers through logical verifications. Extensive experimental validation demonstrates considerable improvements compared to existing approaches. 

---
# LoVR: A Benchmark for Long Video Retrieval in Multimodal Contexts 

**Authors**: Qifeng Cai, Hao Liang, Hejun Dong, Meiyi Qiang, Ruichuan An, Zhaoyang Han, Zhengzhou Zhu, Bin Cui, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13928)  

**Abstract**: Long videos contain a vast amount of information, making video-text retrieval an essential and challenging task in multimodal learning. However, existing benchmarks suffer from limited video duration, low-quality captions, and coarse annotation granularity, which hinder the evaluation of advanced video-text retrieval methods. To address these limitations, we introduce LoVR, a benchmark specifically designed for long video-text retrieval. LoVR contains 467 long videos and over 40,804 fine-grained clips with high-quality captions. To overcome the issue of poor machine-generated annotations, we propose an efficient caption generation framework that integrates VLM automatic generation, caption quality scoring, and dynamic refinement. This pipeline improves annotation accuracy while maintaining scalability. Furthermore, we introduce a semantic fusion method to generate coherent full-video captions without losing important contextual information. Our benchmark introduces longer videos, more detailed captions, and a larger-scale dataset, presenting new challenges for video understanding and retrieval. Extensive experiments on various advanced embedding models demonstrate that LoVR is a challenging benchmark, revealing the limitations of current approaches and providing valuable insights for future research. We release the code and dataset link at this https URL 

---
# VulCPE: Context-Aware Cybersecurity Vulnerability Retrieval and Management 

**Authors**: Yuning Jiang, Feiyang Shang, Freedy Tan Wei You, Huilin Wang, Chia Ren Cong, Qiaoran Meng, Nay Oo, Hoon Wei Lim, Biplab Sikdar  

**Link**: [PDF](https://arxiv.org/pdf/2505.13895)  

**Abstract**: The dynamic landscape of cybersecurity demands precise and scalable solutions for vulnerability management in heterogeneous systems, where configuration-specific vulnerabilities are often misidentified due to inconsistent data in databases like the National Vulnerability Database (NVD). Inaccurate Common Platform Enumeration (CPE) data in NVD further leads to false positives and incomplete vulnerability retrieval. Informed by our systematic analysis of CPE and CVEdeails data, revealing more than 50% vendor name inconsistencies, we propose VulCPE, a framework that standardizes data and models configuration dependencies using a unified CPE schema (uCPE), entity recognition, relation extraction, and graph-based modeling. VulCPE achieves superior retrieval precision (0.766) and coverage (0.926) over existing tools. VulCPE ensures precise, context-aware vulnerability management, enhancing cyber resilience. 

---
# Q${}^2$Forge: Minting Competency Questions and SPARQL Queries for Question-Answering Over Knowledge Graphs 

**Authors**: Yousouf Taghzouti, Franck Michel, Tao Jiang, Louis-Félix Nothias, Fabien Gandon  

**Link**: [PDF](https://arxiv.org/pdf/2505.13572)  

**Abstract**: The SPARQL query language is the standard method to access knowledge graphs (KGs). However, formulating SPARQL queries is a significant challenge for non-expert users, and remains time-consuming for the experienced ones. Best practices recommend to document KGs with competency questions and example queries to contextualise the knowledge they contain and illustrate their potential applications. In practice, however, this is either not the case or the examples are provided in limited numbers. Large Language Models (LLMs) are being used in conversational agents and are proving to be an attractive solution with a wide range of applications, from simple question-answering about common knowledge to generating code in a targeted programming language. However, training and testing these models to produce high quality SPARQL queries from natural language questions requires substantial datasets of question-query pairs. In this paper, we present Q${}^2$Forge that addresses the challenge of generating new competency questions for a KG and corresponding SPARQL queries. It iteratively validates those queries with human feedback and LLM as a judge. Q${}^2$Forge is open source, generic, extensible and modular, meaning that the different modules of the application (CQ generation, query generation and query refinement) can be used separately, as an integrated pipeline, or replaced by alternative services. The result is a complete pipeline from competency question formulation to query evaluation, supporting the creation of reference query sets for any target KG. 

---
# InterFeat: An Automated Pipeline for Finding Interesting Hypotheses in Structured Biomedical Data 

**Authors**: Dan Ofer, Michal Linial, Dafna Shahaf  

**Link**: [PDF](https://arxiv.org/pdf/2505.13534)  

**Abstract**: Finding interesting phenomena is the core of scientific discovery, but it is a manual, ill-defined concept. We present an integrative pipeline for automating the discovery of interesting simple hypotheses (feature-target relations with effect direction and a potential underlying mechanism) in structured biomedical data. The pipeline combines machine learning, knowledge graphs, literature search and Large Language Models. We formalize "interestingness" as a combination of novelty, utility and plausibility. On 8 major diseases from the UK Biobank, our pipeline consistently recovers risk factors years before their appearance in the literature. 40--53% of our top candidates were validated as interesting, compared to 0--7% for a SHAP-based baseline. Overall, 28% of 109 candidates were interesting to medical experts. The pipeline addresses the challenge of operationalizing "interestingness" scalably and for any target. We release data and code: this https URL 

---
