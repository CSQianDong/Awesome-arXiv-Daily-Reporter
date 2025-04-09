# StealthRank: LLM Ranking Manipulation via Stealthy Prompt Optimization 

**Authors**: Yiming Tang, Yi Fan, Chenxiao Yu, Tiankai Yang, Yue Zhao, Xiyang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2504.05804)  

**Abstract**: The integration of large language models (LLMs) into information retrieval systems introduces new attack surfaces, particularly for adversarial ranking manipulations. We present StealthRank, a novel adversarial ranking attack that manipulates LLM-driven product recommendation systems while maintaining textual fluency and stealth. Unlike existing methods that often introduce detectable anomalies, StealthRank employs an energy-based optimization framework combined with Langevin dynamics to generate StealthRank Prompts (SRPs)-adversarial text sequences embedded within product descriptions that subtly yet effectively influence LLM ranking mechanisms. We evaluate StealthRank across multiple LLMs, demonstrating its ability to covertly boost the ranking of target products while avoiding explicit manipulation traces that can be easily detected. Our results show that StealthRank consistently outperforms state-of-the-art adversarial ranking baselines in both effectiveness and stealth, highlighting critical vulnerabilities in LLM-driven recommendation systems. 

---
# Retrieval Augmented Generation with Collaborative Filtering for Personalized Text Generation 

**Authors**: Teng Shi, Jun Xu, Xiao Zhang, Xiaoxue Zang, Kai Zheng, Yang Song, Han Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.05731)  

**Abstract**: Recently, the personalization of Large Language Models (LLMs) to generate content that aligns with individual user preferences has garnered widespread attention. Personalized Retrieval-Augmented Generation (RAG), which retrieves relevant documents from the user's history to reflect their preferences and enhance LLM generation, is one commonly used approach for personalization. However, existing personalized RAG methods do not consider that the histories of similar users can also assist in personalized generation for the current user, meaning that collaborative information between users can also benefit personalized generation. Inspired by the application of collaborative filtering in recommender systems, we propose a method called CFRAG, which adapts Collaborative Filtering to RAG for personalized text generation. However, this presents two challenges: (1)~how to incorporate collaborative information without explicit user similarity labels? (2)~how to retrieve documents that support personalized LLM generation? For Challenge 1, we use contrastive learning to train user embeddings to retrieve similar users and introduce collaborative information. For Challenge 2, we design a personalized retriever and reranker to retrieve the top-$k$ documents from these users' histories. We take into account the user's preference during retrieval and reranking. Then we leverage feedback from the LLM to fine-tune the personalized retriever and reranker, enabling them to retrieve documents that meet the personalized generation needs of the LLM. Experimental results on the Language Model Personalization (LaMP) benchmark validate the effectiveness of CFRAG. Further analysis confirms the importance of incorporating collaborative information. 

---
# PathGPT: Leveraging Large Language Models for Personalized Route Generation 

**Authors**: Steeve Cuthbert Marcelyn, Yucen Gao, Yuzhe Zhang, Xiaofeng Gao, Guihai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.05846)  

**Abstract**: The proliferation of GPS enabled devices has led to the accumulation of a substantial corpus of historical trajectory data. By leveraging these data for training machine learning models,researchers have devised novel data-driven methodologies that address the personalized route recommendation (PRR) problem. In contrast to conventional algorithms such as Dijkstra shortest path algorithm,these novel algorithms possess the capacity to discern and learn patterns within the data,thereby facilitating the generation of more personalized paths. However,once these models have been trained,their application is constrained to the generation of routes that align with their training patterns. This limitation renders them less adaptable to novel scenarios and the deployment of multiple machine learning models might be necessary to address new possible scenarios,which can be costly as each model must be trained separately. Inspired by recent advances in the field of Large Language Models (LLMs),we leveraged their natural language understanding capabilities to develop a unified model to solve the PRR problem while being seamlessly adaptable to new scenarios without additional training. To accomplish this,we combined the extensive knowledge LLMs acquired during training with further access to external hand-crafted context information,similar to RAG (Retrieved Augmented Generation) systems,to enhance their ability to generate paths according to user-defined requirements. Extensive experiments on different datasets show a considerable uplift in LLM performance on the PRR problem. 

---
# Hybrid Retrieval for Hallucination Mitigation in Large Language Models: A Comparative Analysis 

**Authors**: Chandana Sree Mala, Gizem Gezici, Fosca Giannotti  

**Link**: [PDF](https://arxiv.org/pdf/2504.05324)  

**Abstract**: Large Language Models (LLMs) excel in language comprehension and generation but are prone to hallucinations, producing factually incorrect or unsupported outputs. Retrieval Augmented Generation (RAG) systems address this issue by grounding LLM responses with external knowledge. This study evaluates the relationship between retriever effectiveness and hallucination reduction in LLMs using three retrieval approaches: sparse retrieval based on BM25 keyword search, dense retrieval using semantic search with Sentence Transformers, and a proposed hybrid retrieval module. The hybrid module incorporates query expansion and combines the results of sparse and dense retrievers through a dynamically weighted Reciprocal Rank Fusion score. Using the HaluBench dataset, a benchmark for hallucinations in question answering tasks, we assess retrieval performance with metrics such as mean average precision and normalised discounted cumulative gain, focusing on the relevance of the top three retrieved documents. Results show that the hybrid retriever achieves better relevance scores, outperforming both sparse and dense retrievers. Further evaluation of LLM-generated answers against ground truth using metrics such as accuracy, hallucination rate, and rejection rate reveals that the hybrid retriever achieves the highest accuracy on fails, the lowest hallucination rate, and the lowest rejection rate. These findings highlight the hybrid retriever's ability to enhance retrieval relevance, reduce hallucination rates, and improve LLM reliability, emphasising the importance of advanced retrieval techniques in mitigating hallucinations and improving response accuracy. 

---
# User Feedback Alignment for LLM-powered Exploration in Large-scale Recommendation Systems 

**Authors**: Jianling Wang, Yifan Liu, Yinghao Sun, Xuejian Ma, Yueqi Wang, He Ma, Steven Su, Ed H. Chi, Minmin Chen, Lichan Hong, Ningren Han, Haokai Lu  

**Link**: [PDF](https://arxiv.org/pdf/2504.05522)  

**Abstract**: Exploration, the act of broadening user experiences beyond their established preferences, is challenging in large-scale recommendation systems due to feedback loops and limited signals on user exploration patterns. Large Language Models (LLMs) offer potential by leveraging their world knowledge to recommend novel content outside these loops. A key challenge is aligning LLMs with user preferences while preserving their knowledge and reasoning. While using LLMs to plan for the next novel user interest, this paper introduces a novel approach combining hierarchical planning with LLM inference-time scaling to improve recommendation relevancy without compromising novelty. We decouple novelty and user-alignment, training separate LLMs for each objective. We then scale up the novelty-focused LLM's inference and select the best-of-n predictions using the user-aligned LLM. Live experiments demonstrate efficacy, showing significant gains in both user satisfaction (measured by watch activity and active user counts) and exploration diversity. 

---
# Large Language Models Enhanced Hyperbolic Space Recommender Systems 

**Authors**: Wentao Cheng, Zhida Qin, Zexue Wu, Pengzhan Zhou, Tianyu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.05694)  

**Abstract**: Large Language Models (LLMs) have attracted significant attention in recommender systems for their excellent world knowledge capabilities. However, existing methods that rely on Euclidean space struggle to capture the rich hierarchical information inherent in textual and semantic data, which is essential for capturing user preferences. The geometric properties of hyperbolic space offer a promising solution to address this issue. Nevertheless, integrating LLMs-based methods with hyperbolic space to effectively extract and incorporate diverse hierarchical information is non-trivial. To this end, we propose a model-agnostic framework, named HyperLLM, which extracts and integrates hierarchical information from both structural and semantic perspectives. Structurally, HyperLLM uses LLMs to generate multi-level classification tags with hierarchical parent-child relationships for each item. Then, tag-item and user-item interactions are jointly learned and aligned through contrastive learning, thereby providing the model with clear hierarchical information. Semantically, HyperLLM introduces a novel meta-optimized strategy to extract hierarchical information from semantic embeddings and bridge the gap between the semantic and collaborative spaces for seamless integration. Extensive experiments show that HyperLLM significantly outperforms recommender systems based on hyperbolic space and LLMs, achieving performance improvements of over 40%. Furthermore, HyperLLM not only improves recommender performance but also enhances training stability, highlighting the critical role of hierarchical information in recommender systems. 

---
# Efficient Multi-Task Learning via Generalist Recommender 

**Authors**: Luyang Wang, Cangcheng Tang, Chongyang Zhang, Jun Ruan, Kai Huang, Jason Dai  

**Link**: [PDF](https://arxiv.org/pdf/2504.05318)  

**Abstract**: Multi-task learning (MTL) is a common machine learning technique that allows the model to share information across different tasks and improve the accuracy of recommendations for all of them. Many existing MTL implementations suffer from scalability issues as the training and inference performance can degrade with the increasing number of tasks, which can limit production use case scenarios for MTL-based recommender systems. Inspired by the recent advances of large language models, we developed an end-to-end efficient and scalable Generalist Recommender (GRec). GRec takes comprehensive data signals by utilizing NLP heads, parallel Transformers, as well as a wide and deep structure to process multi-modal inputs. These inputs are then combined and fed through a newly proposed task-sentence level routing mechanism to scale the model capabilities on multiple tasks without compromising performance. Offline evaluations and online experiments show that GRec significantly outperforms our previous recommender solutions. GRec has been successfully deployed on one of the largest telecom websites and apps, effectively managing high volumes of online traffic every day. 

---
# Unified Generative Search and Recommendation 

**Authors**: Teng Shi, Jun Xu, Xiao Zhang, Xiaoxue Zang, Kai Zheng, Yang Song, Enyun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.05730)  

**Abstract**: Modern commercial platforms typically offer both search and recommendation functionalities to serve diverse user needs, making joint modeling of these tasks an appealing direction. While prior work has shown that integrating search and recommendation can be mutually beneficial, it also reveals a performance trade-off: enhancements in one task often come at the expense of the other. This challenge arises from their distinct information requirements: search emphasizes semantic relevance between queries and items, whereas recommendation depends more on collaborative signals among users and items. Effectively addressing this trade-off requires tackling two key problems: (1) integrating both semantic and collaborative signals into item representations, and (2) guiding the model to distinguish and adapt to the unique demands of search and recommendation. The emergence of generative retrieval with Large Language Models (LLMs) presents new possibilities. This paradigm encodes items as identifiers and frames both search and recommendation as sequential generation tasks, offering the flexibility to leverage multiple identifiers and task-specific prompts. In light of this, we introduce GenSAR, a unified generative framework for balanced search and recommendation. Our approach designs dual-purpose identifiers and tailored training strategies to incorporate complementary signals and align with task-specific objectives. Experiments on both public and commercial datasets demonstrate that GenSAR effectively reduces the trade-off and achieves state-of-the-art performance on both tasks. 

---
# VALUE: Value-Aware Large Language Model for Query Rewriting via Weighted Trie in Sponsored Search 

**Authors**: Boyang Zuo, Xiao Zhang, Feng Li, Pengjie Wang, Jian Xu, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.05321)  

**Abstract**: In the realm of sponsored search advertising, matching advertisements with the search intent of a user's query is crucial. Query-to-bidwords(i.e. bidding keywords) rewriting is a vital technique that has garnered significant attention. Recently, with the prevalence of LLMs, generative retrieval methods have proven effective in producing high-relevance rewrites. However, we have identified a significant limitation in existing approaches: While fine-tuning LLMs for specific domains enhances semantic relevance, these models have no perception of the intrinsic value of their generated outputs, such as commercial value. Therefore, after SFT, a RLHF phase is often employed to address this issue. Nevertheless, traditional preference alignment methods often face challenges in aligning fine-grained values and are susceptible to overfitting, which diminishes the effectiveness and quality of the generated results. To address these challenges, we propose VALUE(Value-Aware Large language model for qUery rewriting via wEighted trie), the first framework that ensures the generation of high-value and highly relevant bidwords. Our approach utilizes weighted trie, an innovative modification of the traditional trie data structure. By modulating the LLM's output probability distribution with value information from the trie during decoding process, we constrain the generation space and guide the trajectory of text production. Offline experiments demonstrate the effectiveness of our method in semantic matching and preference alignment, showing a remarkable improvement in the value attribute by more than fivefold. Online A/B tests further revealed that our Revenue Per Mille (RPM) metric increased by 1.64%. VALUE has been deployed on our advertising system since October 2024 and served the Double Eleven promotions, the biggest shopping carnival in China. 

---
# Coherency Improved Explainable Recommendation via Large Language Model 

**Authors**: Shijie Liu, Ruixing Ding, Weihai Lu, Jun Wang, Mo Yu, Xiaoming Shi, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.05315)  

**Abstract**: Explainable recommender systems are designed to elucidate the explanation behind each recommendation, enabling users to comprehend the underlying logic. Previous works perform rating prediction and explanation generation in a multi-task manner. However, these works suffer from incoherence between predicted ratings and explanations. To address the issue, we propose a novel framework that employs a large language model (LLM) to generate a rating, transforms it into a rating vector, and finally generates an explanation based on the rating vector and user-item information. Moreover, we propose utilizing publicly available LLMs and pre-trained sentiment analysis models to automatically evaluate the coherence without human annotations. Extensive experimental results on three datasets of explainable recommendation show that the proposed framework is effective, outperforming state-of-the-art baselines with improvements of 7.3\% in explainability and 4.4\% in text quality. 

---
# IterQR: An Iterative Framework for LLM-based Query Rewrite in e-Commercial Search System 

**Authors**: Shangyu Chen, Xinyu Jia, Yingfei Zhang, Shuai Zhang, Xiang Li, Wei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.05309)  

**Abstract**: The essence of modern e-Commercial search system lies in matching user's intent and available candidates depending on user's query, providing personalized and precise service. However, user's query may be incorrect due to ambiguous input and typo, leading to inaccurate search. These cases may be released by query rewrite: modify query to other representation or expansion. However, traditional query rewrite replies on static rewrite vocabulary, which is manually established meanwhile lacks interaction with both domain knowledge in e-Commercial system and common knowledge in the real world. In this paper, with the ability to generate text content of Large Language Models (LLMs), we provide an iterative framework to generate query rewrite. The framework incorporates a 3-stage procedure in each iteration: Rewrite Generation with domain knowledge by Retrieval-Augmented Generation (RAG) and query understanding by Chain-of-Thoughts (CoT); Online Signal Collection with automatic positive rewrite update; Post-training of LLM with multi task objective to generate new rewrites. Our work (named as IterQR) provides a comprehensive framework to generate \textbf{Q}uery \textbf{R}ewrite with both domain / real-world knowledge. It automatically update and self-correct the rewrites during \textbf{iter}ations. \method{} has been deployed in Meituan Delivery's search system (China's leading food delivery platform), providing service for users with significant improvement. 

---
# Are Generative AI Agents Effective Personalized Financial Advisors? 

**Authors**: Takehiro Takayanagi, Kiyoshi Izumi, Javier Sanz-Cruzado, Richard McCreadie, Iadh Ounis  

**Link**: [PDF](https://arxiv.org/pdf/2504.05862)  

**Abstract**: Large language model-based agents are becoming increasingly popular as a low-cost mechanism to provide personalized, conversational advice, and have demonstrated impressive capabilities in relatively simple scenarios, such as movie recommendations. But how do these agents perform in complex high-stakes domains, where domain expertise is essential and mistakes carry substantial risk? This paper investigates the effectiveness of LLM-advisors in the finance domain, focusing on three distinct challenges: (1) eliciting user preferences when users themselves may be unsure of their needs, (2) providing personalized guidance for diverse investment preferences, and (3) leveraging advisor personality to build relationships and foster trust. Via a lab-based user study with 64 participants, we show that LLM-advisors often match human advisor performance when eliciting preferences, although they can struggle to resolve conflicting user needs. When providing personalized advice, the LLM was able to positively influence user behavior, but demonstrated clear failure modes. Our results show that accurate preference elicitation is key, otherwise, the LLM-advisor has little impact, or can even direct the investor toward unsuitable assets. More worryingly, users appear insensitive to the quality of advice being given, or worse these can have an inverse relationship. Indeed, users reported a preference for and increased satisfaction as well as emotional trust with LLMs adopting an extroverted persona, even though those agents provided worse advice. 

---
# Automated Archival Descriptions with Federated Intelligence of LLMs 

**Authors**: Jinghua Groppe, Andreas Marquet, Annabel Walz, Sven Groppe  

**Link**: [PDF](https://arxiv.org/pdf/2504.05711)  

**Abstract**: Enforcing archival standards requires specialized expertise, and manually creating metadata descriptions for archival materials is a tedious and error-prone task. This work aims at exploring the potential of agentic AI and large language models (LLMs) in addressing the challenges of implementing a standardized archival description process. To this end, we introduce an agentic AI-driven system for automated generation of high-quality metadata descriptions of archival materials. We develop a federated optimization approach that unites the intelligence of multiple LLMs to construct optimal archival metadata. We also suggest methods to overcome the challenges associated with using LLMs for consistent metadata generation. To evaluate the feasibility and effectiveness of our techniques, we conducted extensive experiments using a real-world dataset of archival materials, which covers a variety of document types and data formats. The evaluation results demonstrate the feasibility of our techniques and highlight the superior performance of the federated optimization approach compared to single-model solutions in metadata quality and reliability. 

---
# GraphRAFT: Retrieval Augmented Fine-Tuning for Knowledge Graphs on Graph Databases 

**Authors**: Alfred Clemedtson, Borun Shi  

**Link**: [PDF](https://arxiv.org/pdf/2504.05478)  

**Abstract**: Large language models have shown remarkable language processing and reasoning ability but are prone to hallucinate when asked about private data. Retrieval-augmented generation (RAG) retrieves relevant data that fit into an LLM's context window and prompts the LLM for an answer. GraphRAG extends this approach to structured Knowledge Graphs (KGs) and questions regarding entities multiple hops away. The majority of recent GraphRAG methods either overlook the retrieval step or have ad hoc retrieval processes that are abstract or inefficient. This prevents them from being adopted when the KGs are stored in graph databases supporting graph query languages. In this work, we present GraphRAFT, a retrieve-and-reason framework that finetunes LLMs to generate provably correct Cypher queries to retrieve high-quality subgraph contexts and produce accurate answers. Our method is the first such solution that can be taken off-the-shelf and used on KGs stored in native graph DBs. Benchmarks suggest that our method is sample-efficient and scales with the availability of training data. Our method achieves significantly better results than all state-of-the-art models across all four standard metrics on two challenging Q\&As on large text-attributed KGs. 

---
# QGen Studio: An Adaptive Question-Answer Generation, Training and Evaluation Platform 

**Authors**: Movina Moses, Mohab Elkaref, James Barry, Shinnosuke Tanaka, Vishnudev Kuruvanthodi, Nathan Herr, Campbell D Watson, Geeth De Mel  

**Link**: [PDF](https://arxiv.org/pdf/2504.06136)  

**Abstract**: We present QGen Studio: an adaptive question-answer generation, training, and evaluation platform. QGen Studio enables users to leverage large language models (LLMs) to create custom question-answer datasets and fine-tune models on this synthetic data. It features a dataset viewer and model explorer to streamline this process. The dataset viewer provides key metrics and visualizes the context from which the QA pairs are generated, offering insights into data quality. The model explorer supports model comparison, allowing users to contrast the performance of their trained LLMs against other models, supporting performance benchmarking and refinement. QGen Studio delivers an interactive, end-to-end solution for generating QA datasets and training scalable, domain-adaptable models. The studio will be open-sourced soon, allowing users to deploy it locally. 

---
# From 128K to 4M: Efficient Training of Ultra-Long Context Large Language Models 

**Authors**: Chejian Xu, Wei Ping, Peng Xu, Zihan Liu, Boxin Wang, Mohammad Shoeybi, Bo Li, Bryan Catanzaro  

**Link**: [PDF](https://arxiv.org/pdf/2504.06214)  

**Abstract**: Long-context capabilities are essential for a wide range of applications, including document and video understanding, in-context learning, and inference-time scaling, all of which require models to process and reason over long sequences of text and multimodal data. In this work, we introduce a efficient training recipe for building ultra-long context LLMs from aligned instruct model, pushing the boundaries of context lengths from 128K to 1M, 2M, and 4M tokens. Our approach leverages efficient continued pretraining strategies to extend the context window and employs effective instruction tuning to maintain the instruction-following and reasoning abilities. Our UltraLong-8B, built on Llama3.1-Instruct with our recipe, achieves state-of-the-art performance across a diverse set of long-context benchmarks. Importantly, models trained with our approach maintain competitive performance on standard benchmarks, demonstrating balanced improvements for both long and short context tasks. We further provide an in-depth analysis of key design choices, highlighting the impacts of scaling strategies and data composition. Our findings establish a robust framework for efficiently scaling context lengths while preserving general model capabilities. We release all model weights at: this https URL. 

---
# Multi-Sense Embeddings for Language Models and Knowledge Distillation 

**Authors**: Qitong Wang, Mohammed J. Zaki, Georgios Kollias, Vasileios Kalantzis  

**Link**: [PDF](https://arxiv.org/pdf/2504.06036)  

**Abstract**: Transformer-based large language models (LLMs) rely on contextual embeddings which generate different (continuous) representations for the same token depending on its surrounding context. Nonetheless, words and tokens typically have a limited number of senses (or meanings). We propose multi-sense embeddings as a drop-in replacement for each token in order to capture the range of their uses in a language. To construct a sense embedding dictionary, we apply a clustering algorithm to embeddings generated by an LLM and consider the cluster centers as representative sense embeddings. In addition, we propose a novel knowledge distillation method that leverages the sense dictionary to learn a smaller student model that mimics the senses from the much larger base LLM model, offering significant space and inference time savings, while maintaining competitive performance. Via thorough experiments on various benchmarks, we showcase the effectiveness of our sense embeddings and knowledge distillation approach. We share our code at this https URL 

---
# LExT: Towards Evaluating Trustworthiness of Natural Language Explanations 

**Authors**: Krithi Shailya, Shreya Rajpal, Gokul S Krishnan, Balaraman Ravindran  

**Link**: [PDF](https://arxiv.org/pdf/2504.06227)  

**Abstract**: As Large Language Models (LLMs) become increasingly integrated into high-stakes domains, there have been several approaches proposed toward generating natural language explanations. These explanations are crucial for enhancing the interpretability of a model, especially in sensitive domains like healthcare, where transparency and reliability are key. In light of such explanations being generated by LLMs and its known concerns, there is a growing need for robust evaluation frameworks to assess model-generated explanations. Natural Language Generation metrics like BLEU and ROUGE capture syntactic and semantic accuracies but overlook other crucial aspects such as factual accuracy, consistency, and faithfulness. To address this gap, we propose a general framework for quantifying trustworthiness of natural language explanations, balancing Plausibility and Faithfulness, to derive a comprehensive Language Explanation Trustworthiness Score (LExT) (The code and set up to reproduce our experiments are publicly available at this https URL). Applying our domain-agnostic framework to the healthcare domain using public medical datasets, we evaluate six models, including domain-specific and general-purpose models. Our findings demonstrate significant differences in their ability to generate trustworthy explanations. On comparing these explanations, we make interesting observations such as inconsistencies in Faithfulness demonstrated by general-purpose models and their tendency to outperform domain-specific fine-tuned models. This work further highlights the importance of using a tailored evaluation framework to assess natural language explanations in sensitive fields, providing a foundation for improving the trustworthiness and transparency of language models in healthcare and beyond. 

---
# Llama-3-Nanda-10B-Chat: An Open Generative Large Language Model for Hindi 

**Authors**: Monojit Choudhury, Shivam Chauhan, Rocktim Jyoti Das, Dhruv Sahnan, Xudong Han, Haonan Li, Aaryamonvikram Singh, Alok Anil Jadhav, Utkarsh Agarwal, Mukund Choudhary, Debopriyo Banerjee, Fajri Koto, Junaid Bhat, Awantika Shukla, Samujjwal Ghosh, Samta Kamboj, Onkar Pandit, Lalit Pradhan, Rahul Pal, Sunil Sahu, Soundar Doraiswamy, Parvez Mullah, Ali El Filali, Neha Sengupta, Gokul Ramakrishnan, Rituraj Joshi, Gurpreet Gosal, Avraham Sheinin, Natalia Vassilieva, Preslav Nakov  

**Link**: [PDF](https://arxiv.org/pdf/2504.06011)  

**Abstract**: Developing high-quality large language models (LLMs) for moderately resourced languages presents unique challenges in data availability, model adaptation, and evaluation. We introduce Llama-3-Nanda-10B-Chat, or Nanda for short, a state-of-the-art Hindi-centric instruction-tuned generative LLM, designed to push the boundaries of open-source Hindi language models. Built upon Llama-3-8B, Nanda incorporates continuous pre-training with expanded transformer blocks, leveraging the Llama Pro methodology. A key challenge was the limited availability of high-quality Hindi text data; we addressed this through rigorous data curation, augmentation, and strategic bilingual training, balancing Hindi and English corpora to optimize cross-linguistic knowledge transfer. With 10 billion parameters, Nanda stands among the top-performing open-source Hindi and multilingual models of similar scale, demonstrating significant advantages over many existing models. We provide an in-depth discussion of training strategies, fine-tuning techniques, safety alignment, and evaluation metrics, demonstrating how these approaches enabled Nanda to achieve state-of-the-art results. By open-sourcing Nanda, we aim to advance research in Hindi LLMs and support a wide range of real-world applications across academia, industry, and public services. 

---
# Assessing how hyperparameters impact Large Language Models' sarcasm detection performance 

**Authors**: Montgomery Gole, Andriy Miranskyy  

**Link**: [PDF](https://arxiv.org/pdf/2504.06166)  

**Abstract**: Sarcasm detection is challenging for both humans and machines. This work explores how model characteristics impact sarcasm detection in OpenAI's GPT, and Meta's Llama-2 models, given their strong natural language understanding, and popularity. We evaluate fine-tuned and zero-shot models across various sizes, releases, and hyperparameters. Experiments were conducted on the political and balanced (pol-bal) portion of the popular Self-Annotated Reddit Corpus (SARC2.0) sarcasm dataset. Fine-tuned performance improves monotonically with model size within a model family, while hyperparameter tuning also impacts performance. In the fine-tuning scenario, full precision Llama-2-13b achieves state-of-the-art accuracy and $F_1$-score, both measured at 0.83, comparable to average human performance. In the zero-shot setting, one GPT-4 model achieves competitive performance to prior attempts, yielding an accuracy of 0.70 and an $F_1$-score of 0.75. Furthermore, a model's performance may increase or decline with each release, highlighting the need to reassess performance after each release. 

---
# NativQA Framework: Enabling LLMs with Native, Local, and Everyday Knowledge 

**Authors**: Firoj Alam, Md Arid Hasan, Sahinur Rahman Laskar, Mucahid Kutlu, Shammur Absar Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2504.05995)  

**Abstract**: The rapid advancement of large language models (LLMs) has raised concerns about cultural bias, fairness, and their applicability in diverse linguistic and underrepresented regional contexts. To enhance and benchmark the capabilities of LLMs, there is a need to develop large-scale resources focused on multilingual, local, and cultural contexts. In this study, we propose a framework, NativQA, that can seamlessly construct large-scale, culturally and regionally aligned QA datasets in native languages. The framework utilizes user-defined seed queries and leverages search engines to collect location-specific, everyday information. It has been evaluated across 39 locations in 24 countries and in 7 languages, ranging from extremely low-resource to high-resource languages, which resulted over 300K Question Answer (QA) pairs. The developed resources can be used for LLM benchmarking and further fine-tuning. The framework has been made publicly available for the community (this https URL). 

---
# Leveraging Robust Optimization for LLM Alignment under Distribution Shifts 

**Authors**: Mingye Zhu, Yi Liu, Junbo Guo, Quan Wang, Yongdong Zhang, Zhendong Mao  

**Link**: [PDF](https://arxiv.org/pdf/2504.05831)  

**Abstract**: Large language models (LLMs) increasingly rely on preference alignment methods to steer outputs toward human values, yet these methods are often constrained by the scarcity of high-quality human-annotated data. To tackle this, recent approaches have turned to synthetic data generated by LLMs as a scalable alternative. However, synthetic data can introduce distribution shifts, compromising the nuanced human preferences that are essential for desirable outputs. In this paper, we propose a novel distribution-aware optimization framework that improves preference alignment in the presence of such shifts. Our approach first estimates the likelihood ratios between the target and training distributions leveraging a learned classifier, then it minimizes the worst-case loss over data regions that reflect the target human-preferred distribution. By explicitly prioritizing the target distribution during optimization, our method mitigates the adverse effects of distributional variation and enhances the generation of responses that faithfully reflect human values. 

---
# Encoder-Decoder Gemma: Improving the Quality-Efficiency Trade-Off via Adaptation 

**Authors**: Biao Zhang, Fedor Moiseev, Joshua Ainslie, Paul Suganthan, Min Ma, Surya Bhupatiraju, Fede Lebron, Orhan Firat, Armand Joulin, Zhe Dong  

**Link**: [PDF](https://arxiv.org/pdf/2504.06225)  

**Abstract**: While decoder-only large language models (LLMs) have shown impressive results, encoder-decoder models are still widely adopted in real-world applications for their inference efficiency and richer encoder representation. In this paper, we study a novel problem: adapting pretrained decoder-only LLMs to encoder-decoder, with the goal of leveraging the strengths of both approaches to achieve a more favorable quality-efficiency trade-off. We argue that adaptation not only enables inheriting the capability of decoder-only LLMs but also reduces the demand for computation compared to pretraining from scratch. We rigorously explore different pretraining objectives and parameter initialization/optimization techniques. Through extensive experiments based on Gemma 2 (2B and 9B) and a suite of newly pretrained mT5-sized models (up to 1.6B), we demonstrate the effectiveness of adaptation and the advantage of encoder-decoder LLMs. Under similar inference budget, encoder-decoder LLMs achieve comparable (often better) pretraining performance but substantially better finetuning performance than their decoder-only counterpart. For example, Gemma 2B-2B outperforms Gemma 2B by $\sim$7\% after instruction tuning. Encoder-decoder adaptation also allows for flexible combination of different-sized models, where Gemma 9B-2B significantly surpasses Gemma 2B-2B by $>$3\%. The adapted encoder representation also yields better results on SuperGLUE. We will release our checkpoints to facilitate future research. 

---
# LLM$\times$MapReduce-V2: Entropy-Driven Convolutional Test-Time Scaling for Generating Long-Form Articles from Extremely Long Resources 

**Authors**: Haoyu Wang, Yujia Fu, Zhu Zhang, Shuo Wang, Zirui Ren, Xiaorong Wang, Zhili Li, Chaoqun He, Bo An, Zhiyuan Liu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.05732)  

**Abstract**: Long-form generation is crucial for a wide range of practical applications, typically categorized into short-to-long and long-to-long generation. While short-to-long generations have received considerable attention, generating long texts from extremely long resources remains relatively underexplored. The primary challenge in long-to-long generation lies in effectively integrating and analyzing relevant information from extensive inputs, which remains difficult for current large language models (LLMs). In this paper, we propose LLM$\times$MapReduce-V2, a novel test-time scaling strategy designed to enhance the ability of LLMs to process extremely long inputs. Drawing inspiration from convolutional neural networks, which iteratively integrate local features into higher-level global representations, LLM$\times$MapReduce-V2 utilizes stacked convolutional scaling layers to progressively expand the understanding of input materials. Both quantitative and qualitative experimental results demonstrate that our approach substantially enhances the ability of LLMs to process long inputs and generate coherent, informative long-form articles, outperforming several representative baselines. 

---
# STRIVE: A Think & Improve Approach with Iterative Refinement for Enhancing Question Quality Estimation 

**Authors**: Aniket Deroy, Subhankar Maity  

**Link**: [PDF](https://arxiv.org/pdf/2504.05693)  

**Abstract**: Automatically assessing question quality is crucial for educators as it saves time, ensures consistency, and provides immediate feedback for refining teaching materials. We propose a novel methodology called STRIVE (Structured Thinking and Refinement with multiLLMs for Improving Verified Question Estimation) using a series of Large Language Models (LLMs) for automatic question evaluation. This approach aims to improve the accuracy and depth of question quality assessment, ultimately supporting diverse learners and enhancing educational practices. The method estimates question quality in an automated manner by generating multiple evaluations based on the strengths and weaknesses of the provided question and then choosing the best solution generated by the LLM. Then the process is improved by iterative review and response with another LLM until the evaluation metric values converge. This sophisticated method of evaluating question quality improves the estimation of question quality by automating the task of question quality evaluation. Correlation scores show that using this proposed method helps to improve correlation with human judgments compared to the baseline method. Error analysis shows that metrics like relevance and appropriateness improve significantly relative to human judgments by using STRIVE. 

---
# Towards Smarter Hiring: Are Zero-Shot and Few-Shot Pre-trained LLMs Ready for HR Spoken Interview Transcript Analysis? 

**Authors**: Subhankar Maity, Aniket Deroy, Sudeshna Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2504.05683)  

**Abstract**: This research paper presents a comprehensive analysis of the performance of prominent pre-trained large language models (LLMs), including GPT-4 Turbo, GPT-3.5 Turbo, text-davinci-003, text-babbage-001, text-curie-001, text-ada-001, llama-2-7b-chat, llama-2-13b-chat, and llama-2-70b-chat, in comparison to expert human evaluators in providing scores, identifying errors, and offering feedback and improvement suggestions to candidates during mock HR (Human Resources) interviews. We introduce a dataset called HURIT (Human Resource Interview Transcripts), which comprises 3,890 HR interview transcripts sourced from real-world HR interview scenarios. Our findings reveal that pre-trained LLMs, particularly GPT-4 Turbo and GPT-3.5 Turbo, exhibit commendable performance and are capable of producing evaluations comparable to those of expert human evaluators. Although these LLMs demonstrate proficiency in providing scores comparable to human experts in terms of human evaluation metrics, they frequently fail to identify errors and offer specific actionable advice for candidate performance improvement in HR interviews. Our research suggests that the current state-of-the-art pre-trained LLMs are not fully conducive for automatic deployment in an HR interview assessment. Instead, our findings advocate for a human-in-the-loop approach, to incorporate manual checks for inconsistencies and provisions for improving feedback quality as a more suitable strategy. 

---
# Layer-Aware Embedding Fusion for LLMs in Text Classifications 

**Authors**: Jiho Gwak, Yuchul Jung  

**Link**: [PDF](https://arxiv.org/pdf/2504.05764)  

**Abstract**: Embedding fusion has emerged as an effective approach for enhancing performance across various NLP tasks. However, systematic guidelines for selecting optimal layers and developing effective fusion strategies for the integration of LLMs remain underexplored. In this study, we propose a layer-aware embedding selection method and investigate how to quantitatively evaluate different layers to identify the most important ones for downstream NLP tasks, showing that the critical layers vary depending on the dataset. We also explore how combining embeddings from multiple LLMs, without requiring model fine-tuning, can improve performance. Experiments on four English text classification datasets (SST-2, MR, R8, and R52) demonstrate that different layers in LLMs exhibit varying degrees of representational strength for classification, and that combining embeddings from different models can enhance performance if the models exhibit complementary characteristics. Additionally, we discuss resources overhead (memory and inference time) to provide a balanced perspective on the real world feasibility of embedding fusion. Future work will explore multilingual and domain specific datasets, as well as techniques for automating layer selection, to improve both performance and scalability. 

---
# Leveraging Prompt-Tuning for Bengali Grammatical Error Explanation Using Large Language Models 

**Authors**: Subhankar Maity, Aniket Deroy  

**Link**: [PDF](https://arxiv.org/pdf/2504.05642)  

**Abstract**: We propose a novel three-step prompt-tuning method for Bengali Grammatical Error Explanation (BGEE) using state-of-the-art large language models (LLMs) such as GPT-4, GPT-3.5 Turbo, and Llama-2-70b. Our approach involves identifying and categorizing grammatical errors in Bengali sentences, generating corrected versions of the sentences, and providing natural language explanations for each identified error. We evaluate the performance of our BGEE system using both automated evaluation metrics and human evaluation conducted by experienced Bengali language experts. Our proposed prompt-tuning approach shows that GPT-4, the best performing LLM, surpasses the baseline model in automated evaluation metrics, with a 5.26% improvement in F1 score and a 6.95% improvement in exact match. Furthermore, compared to the previous baseline, GPT-4 demonstrates a decrease of 25.51% in wrong error type and a decrease of 26.27% in wrong error explanation. However, the results still lag behind the human baseline. 

---
# Two Intermediate Translations Are Better Than One: Fine-tuning LLMs for Document-level Translation Refinement 

**Authors**: Yichen Dong, Xinglin Lyu, Junhui Li, Daimeng Wei, Min Zhang, Shimin Tao, Hao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.05614)  

**Abstract**: Recent research has shown that large language models (LLMs) can enhance translation quality through self-refinement. In this paper, we build on this idea by extending the refinement from sentence-level to document-level translation, specifically focusing on document-to-document (Doc2Doc) translation refinement. Since sentence-to-sentence (Sent2Sent) and Doc2Doc translation address different aspects of the translation process, we propose fine-tuning LLMs for translation refinement using two intermediate translations, combining the strengths of both Sent2Sent and Doc2Doc. Additionally, recognizing that the quality of intermediate translations varies, we introduce an enhanced fine-tuning method with quality awareness that assigns lower weights to easier translations and higher weights to more difficult ones, enabling the model to focus on challenging translation cases. Experimental results across ten translation tasks with LLaMA-3-8B-Instruct and Mistral-Nemo-Instruct demonstrate the effectiveness of our approach. 

---
# Rank-Then-Score: Enhancing Large Language Models for Automated Essay Scoring 

**Authors**: Yida Cai, Kun Liang, Sanwoo Lee, Qinghan Wang, Yunfang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.05736)  

**Abstract**: In recent years, large language models (LLMs) achieve remarkable success across a variety of tasks. However, their potential in the domain of Automated Essay Scoring (AES) remains largely underexplored. Moreover, compared to English data, the methods for Chinese AES is not well developed. In this paper, we propose Rank-Then-Score (RTS), a fine-tuning framework based on large language models to enhance their essay scoring capabilities. Specifically, we fine-tune the ranking model (Ranker) with feature-enriched data, and then feed the output of the ranking model, in the form of a candidate score set, with the essay content into the scoring model (Scorer) to produce the final score. Experimental results on two benchmark datasets, HSK and ASAP, demonstrate that RTS consistently outperforms the direct prompting (Vanilla) method in terms of average QWK across all LLMs and datasets, and achieves the best performance on Chinese essay scoring using the HSK dataset. 

---
# On the Impact of Language Nuances on Sentiment Analysis with Large Language Models: Paraphrasing, Sarcasm, and Emojis 

**Authors**: Naman Bhargava, Mohammed I. Radaideh, O Hwang Kwon, Aditi Verma, Majdi I. Radaideh  

**Link**: [PDF](https://arxiv.org/pdf/2504.05603)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive performance across various tasks, including sentiment analysis. However, data quality--particularly when sourced from social media--can significantly impact their accuracy. This research explores how textual nuances, including emojis and sarcasm, affect sentiment analysis, with a particular focus on improving data quality through text paraphrasing techniques. To address the lack of labeled sarcasm data, the authors created a human-labeled dataset of 5929 tweets that enabled the assessment of LLM in various sarcasm contexts. The results show that when topic-specific datasets, such as those related to nuclear power, are used to finetune LLMs these models are not able to comprehend accurate sentiment in presence of sarcasm due to less diverse text, requiring external interventions like sarcasm removal to boost model accuracy. Sarcasm removal led to up to 21% improvement in sentiment accuracy, as LLMs trained on nuclear power-related content struggled with sarcastic tweets, achieving only 30% accuracy. In contrast, LLMs trained on general tweet datasets, covering a broader range of topics, showed considerable improvements in predicting sentiment for sarcastic tweets (60% accuracy), indicating that incorporating general text data can enhance sarcasm detection. The study also utilized adversarial text augmentation, showing that creating synthetic text variants by making minor changes significantly increased model robustness and accuracy for sarcastic tweets (approximately 85%). Additionally, text paraphrasing of tweets with fragmented language transformed around 40% of the tweets with low-confidence labels into high-confidence ones, improving LLMs sentiment analysis accuracy by 6%. 

---
# FactGuard: Leveraging Multi-Agent Systems to Generate Answerable and Unanswerable Questions for Enhanced Long-Context LLM Extraction 

**Authors**: Qian-Wen Zhang, Fang Li, Jie Wang, Lingfeng Qiao, Yifei Yu, Di Yin, Xing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.05607)  

**Abstract**: Extractive reading comprehension systems are designed to locate the correct answer to a question within a given text. However, a persistent challenge lies in ensuring these models maintain high accuracy in answering questions while reliably recognizing unanswerable queries. Despite significant advances in large language models (LLMs) for reading comprehension, this issue remains critical, particularly as the length of supported contexts continues to expand. To address this challenge, we propose an innovative data augmentation methodology grounded in a multi-agent collaborative framework. Unlike traditional methods, such as the costly human annotation process required for datasets like SQuAD 2.0, our method autonomously generates evidence-based question-answer pairs and systematically constructs unanswerable questions. Using this methodology, we developed the FactGuard-Bench dataset, which comprises 25,220 examples of both answerable and unanswerable question scenarios, with context lengths ranging from 8K to 128K. Experimental evaluations conducted on seven popular LLMs reveal that even the most advanced models achieve only 61.79% overall accuracy. Furthermore, we emphasize the importance of a model's ability to reason about unanswerable questions to avoid generating plausible but incorrect answers. By implementing efficient data selection and generation within the multi-agent collaborative framework, our method significantly reduces the traditionally high costs associated with manual annotation and provides valuable insights for the training and optimization of LLMs. 

---
# Reasoning Towards Fairness: Mitigating Bias in Language Models through Reasoning-Guided Fine-Tuning 

**Authors**: Sanchit Kabra, Akshita Jha, Chandan Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2504.05632)  

**Abstract**: Recent advances in large-scale generative language models have shown that reasoning capabilities can significantly improve model performance across a variety of tasks. However, the impact of reasoning on a model's ability to mitigate stereotypical responses remains largely underexplored. In this work, we investigate the crucial relationship between a model's reasoning ability and fairness, and ask whether improved reasoning capabilities can mitigate harmful stereotypical responses, especially those arising due to shallow or flawed reasoning. We conduct a comprehensive evaluation of multiple open-source LLMs, and find that larger models with stronger reasoning abilities exhibit substantially lower stereotypical bias on existing fairness benchmarks. Building on this insight, we introduce ReGiFT -- Reasoning Guided Fine-Tuning, a novel approach that extracts structured reasoning traces from advanced reasoning models and infuses them into models that lack such capabilities. We use only general-purpose reasoning and do not require any fairness-specific supervision for bias mitigation. Notably, we see that models fine-tuned using ReGiFT not only improve fairness relative to their non-reasoning counterparts but also outperform advanced reasoning models on fairness benchmarks. We also analyze how variations in the correctness of the reasoning traces and their length influence model fairness and their overall performance. Our findings highlight that enhancing reasoning capabilities is an effective, fairness-agnostic strategy for mitigating stereotypical bias caused by reasoning flaws. 

---
# Can Large Language Models Match Tutoring System Adaptivity? A Benchmarking Study 

**Authors**: Conrad Borchers, Tianze Shou  

**Link**: [PDF](https://arxiv.org/pdf/2504.05570)  

**Abstract**: Large Language Models (LLMs) hold promise as dynamic instructional aids. Yet, it remains unclear whether LLMs can replicate the adaptivity of intelligent tutoring systems (ITS)--where student knowledge and pedagogical strategies are explicitly modeled. We propose a prompt variation framework to assess LLM-generated instructional moves' adaptivity and pedagogical soundness across 75 real-world tutoring scenarios from an ITS. We systematically remove key context components (e.g., student errors and knowledge components) from prompts to create variations of each scenario. Three representative LLMs (Llama3-8B, Llama3-70B, and GPT-4o) generate 1,350 instructional moves. We use text embeddings and randomization tests to measure how the omission of each context feature impacts the LLMs' outputs (adaptivity) and a validated tutor-training classifier to evaluate response quality (pedagogical soundness). Surprisingly, even the best-performing model only marginally mimics the adaptivity of ITS. Specifically, Llama3-70B demonstrates statistically significant adaptivity to student errors. Although Llama3-8B's recommendations receive higher pedagogical soundness scores than the other models, it struggles with instruction-following behaviors, including output formatting. By contrast, GPT-4o reliably adheres to instructions but tends to provide overly direct feedback that diverges from effective tutoring, prompting learners with open-ended questions to gauge knowledge. Given these results, we discuss how current LLM-based tutoring is unlikely to produce learning benefits rivaling known-to-be-effective ITS tutoring. Through our open-source benchmarking code, we contribute a reproducible method for evaluating LLMs' instructional adaptivity and fidelity. 

---
# Bridging Industrial Expertise and XR with LLM-Powered Conversational Agents 

**Authors**: Despina Tomkou, George Fatouros, Andreas Andreou, Georgios Makridis, Fotis Liarokapis, Dimitrios Dardanis, Athanasios Kiourtis, John Soldatos, Dimosthenis Kyriazis  

**Link**: [PDF](https://arxiv.org/pdf/2504.05527)  

**Abstract**: This paper introduces a novel integration of Retrieval-Augmented Generation (RAG) enhanced Large Language Models (LLMs) with Extended Reality (XR) technologies to address knowledge transfer challenges in industrial environments. The proposed system embeds domain-specific industrial knowledge into XR environments through a natural language interface, enabling hands-free, context-aware expert guidance for workers. We present the architecture of the proposed system consisting of an LLM Chat Engine with dynamic tool orchestration and an XR application featuring voice-driven interaction. Performance evaluation of various chunking strategies, embedding models, and vector databases reveals that semantic chunking, balanced embedding models, and efficient vector stores deliver optimal performance for industrial knowledge retrieval. The system's potential is demonstrated through early implementation in multiple industrial use cases, including robotic assembly, smart infrastructure maintenance, and aerospace component servicing. Results indicate potential for enhancing training efficiency, remote assistance capabilities, and operational guidance in alignment with Industry 5.0's human-centric and resilient approach to industrial development. 

---
# Pretraining Language Models for Diachronic Linguistic Change Discovery 

**Authors**: Elisabeth Fittschen, Sabrina Li, Tom Lippincott, Leshem Choshsem, Craig Messner  

**Link**: [PDF](https://arxiv.org/pdf/2504.05523)  

**Abstract**: Large language models (LLMs) have shown potential as tools for scientific discovery. This has engendered growing interest in their use in humanistic disciplines, such as historical linguistics and literary studies. These fields often construct arguments on the basis of delineations like genre, or more inflexibly, time period. Although efforts have been made to restrict inference to specific domains via fine-tuning or model editing, we posit that the only true guarantee is domain-restricted pretraining -- typically, a data- and compute-expensive proposition.
We show that efficient pretraining techniques can produce useful models over corpora too large for easy manual inspection but too small for "typical" LLM approaches. We employ a novel date-attribution pipeline in order to obtain a temporally-segmented dataset of five 10-million-word slices. We train two corresponding five-model batteries over these corpus segments, efficient pretraining and Llama3-8B parameter efficiently finetuned.
We find that the pretrained models are faster to train than the finetuned baselines and that they better respect the historical divisions of our corpus. Emphasizing speed and precision over a-historical comprehensiveness enables a number of novel approaches to hypothesis discovery and testing in our target fields. Taking up diachronic linguistics as a testbed, we show that our method enables the detection of a diverse set of phenomena, including en masse lexical change, non-lexical (grammatical and morphological) change, and word sense introduction/obsolescence. We provide a ready-to-use pipeline that allows extension of our approach to other target fields with only minimal adaptation. 

---
# A Survey on Hypothesis Generation for Scientific Discovery in the Era of Large Language Models 

**Authors**: Atilla Kaan Alkan, Shashwat Sourav, Maja Jablonska, Simone Astarita, Rishabh Chakrabarty, Nikhil Garuda, Pranav Khetarpal, Maciej Pióro, Dimitrios Tanoglidis, Kartheik G. Iyer, Mugdha S. Polimera, Michael J. Smith, Tirthankar Ghosal, Marc Huertas-Company, Sandor Kruk, Kevin Schawinski, Ioana Ciucă  

**Link**: [PDF](https://arxiv.org/pdf/2504.05496)  

**Abstract**: Hypothesis generation is a fundamental step in scientific discovery, yet it is increasingly challenged by information overload and disciplinary fragmentation. Recent advances in Large Language Models (LLMs) have sparked growing interest in their potential to enhance and automate this process. This paper presents a comprehensive survey of hypothesis generation with LLMs by (i) reviewing existing methods, from simple prompting techniques to more complex frameworks, and proposing a taxonomy that categorizes these approaches; (ii) analyzing techniques for improving hypothesis quality, such as novelty boosting and structured reasoning; (iii) providing an overview of evaluation strategies; and (iv) discussing key challenges and future directions, including multimodal integration and human-AI collaboration. Our survey aims to serve as a reference for researchers exploring LLMs for hypothesis generation. 

---
# Less but Better: Parameter-Efficient Fine-Tuning of Large Language Models for Personality Detection 

**Authors**: Lingzhi Shen, Yunfei Long, Xiaohao Cai, Guanming Chen, Imran Razzak, Shoaib Jameel  

**Link**: [PDF](https://arxiv.org/pdf/2504.05411)  

**Abstract**: Personality detection automatically identifies an individual's personality from various data sources, such as social media texts. However, as the parameter scale of language models continues to grow, the computational cost becomes increasingly difficult to manage. Fine-tuning also grows more complex, making it harder to justify the effort and reliably predict outcomes. We introduce a novel parameter-efficient fine-tuning framework, PersLLM, to address these challenges. In PersLLM, a large language model (LLM) extracts high-dimensional representations from raw data and stores them in a dynamic memory layer. PersLLM then updates the downstream layers with a replaceable output network, enabling flexible adaptation to various personality detection scenarios. By storing the features in the memory layer, we eliminate the need for repeated complex computations by the LLM. Meanwhile, the lightweight output network serves as a proxy for evaluating the overall effectiveness of the framework, improving the predictability of results. Experimental results on key benchmark datasets like Kaggle and Pandora show that PersLLM significantly reduces computational cost while maintaining competitive performance and strong adaptability. 

---
# Fast Controlled Generation from Language Models with Adaptive Weighted Rejection Sampling 

**Authors**: Benjamin Lipkin, Benjamin LeBrun, Jacob Hoover Vigly, João Loula, David R. MacIver, Li Du, Jason Eisner, Ryan Cotterell, Vikash Mansinghka, Timothy J. O'Donnell, Alexander K. Lew, Tim Vieira  

**Link**: [PDF](https://arxiv.org/pdf/2504.05410)  

**Abstract**: The dominant approach to generating from language models subject to some constraint is locally constrained decoding (LCD), incrementally sampling tokens at each time step such that the constraint is never violated. Typically, this is achieved through token masking: looping over the vocabulary and excluding non-conforming tokens. There are two important problems with this approach. (i) Evaluating the constraint on every token can be prohibitively expensive -- LM vocabularies often exceed $100,000$ tokens. (ii) LCD can distort the global distribution over strings, sampling tokens based only on local information, even if they lead down dead-end paths. This work introduces a new algorithm that addresses both these problems. First, to avoid evaluating a constraint on the full vocabulary at each step of generation, we propose an adaptive rejection sampling algorithm that typically requires orders of magnitude fewer constraint evaluations. Second, we show how this algorithm can be extended to produce low-variance, unbiased estimates of importance weights at a very small additional cost -- estimates that can be soundly used within previously proposed sequential Monte Carlo algorithms to correct for the myopic behavior of local constraint enforcement. Through extensive empirical evaluation in text-to-SQL, molecular synthesis, goal inference, pattern matching, and JSON domains, we show that our approach is superior to state-of-the-art baselines, supporting a broader class of constraints and improving both runtime and performance. Additional theoretical and empirical analyses show that our method's runtime efficiency is driven by its dynamic use of computation, scaling with the divergence between the unconstrained and constrained LM, and as a consequence, runtime improvements are greater for better models. 

---
# FEABench: Evaluating Language Models on Multiphysics Reasoning Ability 

**Authors**: Nayantara Mudur, Hao Cui, Subhashini Venugopalan, Paul Raccuglia, Michael P. Brenner, Peter Norgaard  

**Link**: [PDF](https://arxiv.org/pdf/2504.06260)  

**Abstract**: Building precise simulations of the real world and invoking numerical solvers to answer quantitative problems is an essential requirement in engineering and science. We present FEABench, a benchmark to evaluate the ability of large language models (LLMs) and LLM agents to simulate and solve physics, mathematics and engineering problems using finite element analysis (FEA). We introduce a comprehensive evaluation scheme to investigate the ability of LLMs to solve these problems end-to-end by reasoning over natural language problem descriptions and operating COMSOL Multiphysics$^\circledR$, an FEA software, to compute the answers. We additionally design a language model agent equipped with the ability to interact with the software through its Application Programming Interface (API), examine its outputs and use tools to improve its solutions over multiple iterations. Our best performing strategy generates executable API calls 88% of the time. LLMs that can successfully interact with and operate FEA software to solve problems such as those in our benchmark would push the frontiers of automation in engineering. Acquiring this capability would augment LLMs' reasoning skills with the precision of numerical solvers and advance the development of autonomous systems that can tackle complex problems in the real world. The code is available at this https URL 

---
# Hogwild! Inference: Parallel LLM Generation via Concurrent Attention 

**Authors**: Gleb Rodionov, Roman Garipov, Alina Shutova, George Yakushev, Vage Egiazarian, Anton Sinitsin, Denis Kuznedelev, Dan Alistarh  

**Link**: [PDF](https://arxiv.org/pdf/2504.06261)  

**Abstract**: Large Language Models (LLMs) have demonstrated the ability to tackle increasingly complex tasks through advanced reasoning, long-form content generation, and tool use. Solving these tasks often involves long inference-time computations. In human problem solving, a common strategy to expedite work is collaboration: by dividing the problem into sub-tasks, exploring different strategies concurrently, etc. Recent research has shown that LLMs can also operate in parallel by implementing explicit cooperation frameworks, such as voting mechanisms or the explicit creation of independent sub-tasks that can be executed in parallel. However, each of these frameworks may not be suitable for all types of tasks, which can hinder their applicability. In this work, we propose a different design approach: we run LLM "workers" in parallel , allowing them to synchronize via a concurrently-updated attention cache and prompt these workers to decide how best to collaborate. Our approach allows the instances to come up with their own collaboration strategy for the problem at hand, all the while "seeing" each other's partial progress in the concurrent cache. We implement this approach via Hogwild! Inference: a parallel LLM inference engine where multiple instances of the same LLM run in parallel with the same attention cache, with "instant" access to each other's generated tokens. Hogwild! inference takes advantage of Rotary Position Embeddings (RoPE) to avoid recomputation while improving parallel hardware utilization. We find that modern reasoning-capable LLMs can perform inference with shared Key-Value cache out of the box, without additional fine-tuning. 

---
# Knowledge-Instruct: Effective Continual Pre-training from Limited Data using Instructions 

**Authors**: Oded Ovadia, Meni Brief, Rachel Lemberg, Eitam Sheetrit  

**Link**: [PDF](https://arxiv.org/pdf/2504.05571)  

**Abstract**: While Large Language Models (LLMs) acquire vast knowledge during pre-training, they often lack domain-specific, new, or niche information. Continual pre-training (CPT) attempts to address this gap but suffers from catastrophic forgetting and inefficiencies in low-data regimes. We introduce Knowledge-Instruct, a novel approach to efficiently inject knowledge from limited corpora through pure instruction-tuning. By generating information-dense synthetic instruction data, it effectively integrates new knowledge while preserving general reasoning and instruction-following abilities. Knowledge-Instruct demonstrates superior factual memorization, minimizes catastrophic forgetting, and remains scalable by leveraging synthetic data from relatively small language models. Additionally, it enhances contextual understanding, including complex multi-hop reasoning, facilitating integration with retrieval systems. We validate its effectiveness across diverse benchmarks, including Companies, a new dataset that we release to measure knowledge injection capabilities. 

---
# ShadowCoT: Cognitive Hijacking for Stealthy Reasoning Backdoors in LLMs 

**Authors**: Gejian Zhao, Hanzhou Wu, Xinpeng Zhang, Athanasios V. Vasilakos  

**Link**: [PDF](https://arxiv.org/pdf/2504.05605)  

**Abstract**: Chain-of-Thought (CoT) enhances an LLM's ability to perform complex reasoning tasks, but it also introduces new security issues. In this work, we present ShadowCoT, a novel backdoor attack framework that targets the internal reasoning mechanism of LLMs. Unlike prior token-level or prompt-based attacks, ShadowCoT directly manipulates the model's cognitive reasoning path, enabling it to hijack multi-step reasoning chains and produce logically coherent but adversarial outcomes. By conditioning on internal reasoning states, ShadowCoT learns to recognize and selectively disrupt key reasoning steps, effectively mounting a self-reflective cognitive attack within the target model. Our approach introduces a lightweight yet effective multi-stage injection pipeline, which selectively rewires attention pathways and perturbs intermediate representations with minimal parameter overhead (only 0.15% updated). ShadowCoT further leverages reinforcement learning and reasoning chain pollution (RCP) to autonomously synthesize stealthy adversarial CoTs that remain undetectable to advanced defenses. Extensive experiments across diverse reasoning benchmarks and LLMs show that ShadowCoT consistently achieves high Attack Success Rate (94.4%) and Hijacking Success Rate (88.4%) while preserving benign performance. These results reveal an emergent class of cognition-level threats and highlight the urgent need for defenses beyond shallow surface-level consistency. 

---
# Skywork R1V: Pioneering Multimodal Reasoning with Chain-of-Thought 

**Authors**: Yi Peng, Chris, Xiaokun Wang, Yichen Wei, Jiangbo Pei, Weijie Qiu, Ai Jian, Yunzhuo Hao, Jiachun Pan, Tianyidan Xie, Li Ge, Rongxian Zhuang, Xuchen Song, Yang Liu, Yahui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.05599)  

**Abstract**: We introduce Skywork R1V, a multimodal reasoning model extending the an R1-series Large language models (LLM) to visual modalities via an efficient multimodal transfer method. Leveraging a lightweight visual projector, Skywork R1V facilitates seamless multimodal adaptation without necessitating retraining of either the foundational language model or the vision encoder. To strengthen visual-text alignment, we propose a hybrid optimization strategy that combines Iterative Supervised Fine-Tuning (SFT) with Group Relative Policy Optimization (GRPO), significantly enhancing cross-modal integration efficiency. Additionally, we introduce an adaptive-length Chain-of-Thought distillation approach for reasoning data generation. This approach dynamically optimizes reasoning chain lengths, thereby enhancing inference efficiency and preventing excessive reasoning overthinking. Empirical evaluations demonstrate that Skywork R1V, with only 38B parameters, delivers competitive performance, achieving a score of 69.0 on the MMMU benchmark and 67.5 on MathVista. Meanwhile, it maintains robust textual reasoning performance, evidenced by impressive scores of 72.0 on AIME and 94.0 on MATH500. The Skywork R1V model weights have been publicly released to promote openness and reproducibility. 

---
# Efficient Reinforcement Finetuning via Adaptive Curriculum Learning 

**Authors**: Taiwei Shi, Yiyang Wu, Linxin Song, Tianyi Zhou, Jieyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.05520)  

**Abstract**: Reinforcement finetuning (RFT) has shown great potential for enhancing the mathematical reasoning capabilities of large language models (LLMs), but it is often sample- and compute-inefficient, requiring extensive training. In this work, we introduce AdaRFT (Adaptive Curriculum Reinforcement Finetuning), a method that significantly improves both the efficiency and final accuracy of RFT through adaptive curriculum learning. AdaRFT dynamically adjusts the difficulty of training problems based on the model's recent reward signals, ensuring that the model consistently trains on tasks that are challenging but solvable. This adaptive sampling strategy accelerates learning by maintaining an optimal difficulty range, avoiding wasted computation on problems that are too easy or too hard. AdaRFT requires only a lightweight extension to standard RFT algorithms like Proximal Policy Optimization (PPO), without modifying the reward function or model architecture. Experiments on competition-level math datasets-including AMC, AIME, and IMO-style problems-demonstrate that AdaRFT significantly improves both training efficiency and reasoning performance. We evaluate AdaRFT across multiple data distributions and model sizes, showing that it reduces the number of training steps by up to 2x and improves accuracy by a considerable margin, offering a more scalable and effective RFT framework. 

---
# Evaluating the Generalization Capabilities of Large Language Models on Code Reasoning 

**Authors**: Rem Yang, Julian Dai, Nikos Vasilakis, Martin Rinard  

**Link**: [PDF](https://arxiv.org/pdf/2504.05518)  

**Abstract**: We assess how the code reasoning abilities of large language models (LLMs) generalize to different kinds of programs. We present techniques for obtaining in- and out-of-distribution programs with different characteristics: code sampled from a domain-specific language, code automatically generated by an LLM, code collected from competitive programming contests, and mutated versions of these programs. We also present an experimental methodology for evaluating LLM generalization by comparing their performance on these programs. We perform an extensive evaluation across 10 state-of-the-art models from the past year, obtaining insights into their generalization capabilities over time and across different classes of programs. Our results highlight that while earlier models exhibit behavior consistent with pattern matching, the latest models exhibit strong generalization abilities on code reasoning. 

---
# TxGemma: Efficient and Agentic LLMs for Therapeutics 

**Authors**: Eric Wang, Samuel Schmidgall, Paul F. Jaeger, Fan Zhang, Rory Pilgrim, Yossi Matias, Joelle Barral, David Fleet, Shekoofeh Azizi  

**Link**: [PDF](https://arxiv.org/pdf/2504.06196)  

**Abstract**: Therapeutic development is a costly and high-risk endeavor that is often plagued by high failure rates. To address this, we introduce TxGemma, a suite of efficient, generalist large language models (LLMs) capable of therapeutic property prediction as well as interactive reasoning and explainability. Unlike task-specific models, TxGemma synthesizes information from diverse sources, enabling broad application across the therapeutic development pipeline. The suite includes 2B, 9B, and 27B parameter models, fine-tuned from Gemma-2 on a comprehensive dataset of small molecules, proteins, nucleic acids, diseases, and cell lines. Across 66 therapeutic development tasks, TxGemma achieved superior or comparable performance to the state-of-the-art generalist model on 64 (superior on 45), and against state-of-the-art specialist models on 50 (superior on 26). Fine-tuning TxGemma models on therapeutic downstream tasks, such as clinical trial adverse event prediction, requires less training data than fine-tuning base LLMs, making TxGemma suitable for data-limited applications. Beyond these predictive capabilities, TxGemma features conversational models that bridge the gap between general LLMs and specialized property predictors. These allow scientists to interact in natural language, provide mechanistic reasoning for predictions based on molecular structure, and engage in scientific discussions. Building on this, we further introduce Agentic-Tx, a generalist therapeutic agentic system powered by Gemini 2.5 that reasons, acts, manages diverse workflows, and acquires external domain knowledge. Agentic-Tx surpasses prior leading models on the Humanity's Last Exam benchmark (Chemistry & Biology) with 52.3% relative improvement over o3-mini (high) and 26.7% over o3-mini (high) on GPQA (Chemistry) and excels with improvements of 6.3% (ChemBench-Preference) and 2.4% (ChemBench-Mini) over o3-mini (high). 

---
# Sugar-Coated Poison: Benign Generation Unlocks LLM Jailbreaking 

**Authors**: Yu-Hang Wu, Yu-Jie Xiong, Jie-Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.05652)  

**Abstract**: Large Language Models (LLMs) have become increasingly integral to a wide range of applications. However, they still remain the threat of jailbreak attacks, where attackers manipulate designed prompts to make the models elicit malicious outputs. Analyzing jailbreak methods can help us delve into the weakness of LLMs and improve it. In this paper, We reveal a vulnerability in large language models (LLMs), which we term Defense Threshold Decay (DTD), by analyzing the attention weights of the model's output on input and subsequent output on prior output: as the model generates substantial benign content, its attention weights shift from the input to prior output, making it more susceptible to jailbreak attacks. To demonstrate the exploitability of DTD, we propose a novel jailbreak attack method, Sugar-Coated Poison (SCP), which induces the model to generate substantial benign content through benign input and adversarial reasoning, subsequently producing malicious content. To mitigate such attacks, we introduce a simple yet effective defense strategy, POSD, which significantly reduces jailbreak success rates while preserving the model's generalization capabilities. 

---
# Reasoning Models Know When They're Right: Probing Hidden States for Self-Verification 

**Authors**: Anqi Zhang, Yulin Chen, Jane Pan, Chen Zhao, Aurojit Panda, Jinyang Li, He He  

**Link**: [PDF](https://arxiv.org/pdf/2504.05419)  

**Abstract**: Reasoning models have achieved remarkable performance on tasks like math and logical reasoning thanks to their ability to search during reasoning. However, they still suffer from overthinking, often performing unnecessary reasoning steps even after reaching the correct answer. This raises the question: can models evaluate the correctness of their intermediate answers during reasoning? In this work, we study whether reasoning models encode information about answer correctness through probing the model's hidden states. The resulting probe can verify intermediate answers with high accuracy and produces highly calibrated scores. Additionally, we find models' hidden states encode correctness of future answers, enabling early prediction of the correctness before the intermediate answer is fully formulated. We then use the probe as a verifier to decide whether to exit reasoning at intermediate answers during inference, reducing the number of inference tokens by 24\% without compromising performance. These findings confirm that reasoning models do encode a notion of correctness yet fail to exploit it, revealing substantial untapped potential to enhance their efficiency. 

---
# Agent Guide: A Simple Agent Behavioral Watermarking Framework 

**Authors**: Kaibo Huang, Zhongliang Yang, Linna Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.05871)  

**Abstract**: The increasing deployment of intelligent agents in digital ecosystems, such as social media platforms, has raised significant concerns about traceability and accountability, particularly in cybersecurity and digital content protection. Traditional large language model (LLM) watermarking techniques, which rely on token-level manipulations, are ill-suited for agents due to the challenges of behavior tokenization and information loss during behavior-to-action translation. To address these issues, we propose Agent Guide, a novel behavioral watermarking framework that embeds watermarks by guiding the agent's high-level decisions (behavior) through probability biases, while preserving the naturalness of specific executions (action). Our approach decouples agent behavior into two levels, behavior (e.g., choosing to bookmark) and action (e.g., bookmarking with specific tags), and applies watermark-guided biases to the behavior probability distribution. We employ a z-statistic-based statistical analysis to detect the watermark, ensuring reliable extraction over multiple rounds. Experiments in a social media scenario with diverse agent profiles demonstrate that Agent Guide achieves effective watermark detection with a low false positive rate. Our framework provides a practical and robust solution for agent watermarking, with applications in identifying malicious agents and protecting proprietary agent systems. 

---
# SciSciGPT: Advancing Human-AI Collaboration in the Science of Science 

**Authors**: Erzhuo Shao, Yifang Wang, Yifan Qian, Zhenyu Pan, Han Liu, Dashun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.05559)  

**Abstract**: The increasing availability of large-scale datasets has fueled rapid progress across many scientific fields, creating unprecedented opportunities for research and discovery while posing significant analytical challenges. Recent advances in large language models (LLMs) and AI agents have opened new possibilities for human-AI collaboration, offering powerful tools to navigate this complex research landscape. In this paper, we introduce SciSciGPT, an open-source, prototype AI collaborator that uses the science of science as a testbed to explore the potential of LLM-powered research tools. SciSciGPT automates complex workflows, supports diverse analytical approaches, accelerates research prototyping and iteration, and facilitates reproducibility. Through case studies, we demonstrate its ability to streamline a wide range of empirical and analytical research tasks while highlighting its broader potential to advance research. We further propose an LLM Agent capability maturity model for human-AI collaboration, envisioning a roadmap to further improve and expand upon frameworks like SciSciGPT. As AI capabilities continue to evolve, frameworks like SciSciGPT may play increasingly pivotal roles in scientific research and discovery, unlocking further opportunities. At the same time, these new advances also raise critical challenges, from ensuring transparency and ethical use to balancing human and AI contributions. Addressing these issues may shape the future of scientific inquiry and inform how we train the next generation of scientists to thrive in an increasingly AI-integrated research ecosystem. 

---
# Prism: Dynamic and Flexible Benchmarking of LLMs Code Generation with Monte Carlo Tree Search 

**Authors**: Vahid Majdinasab, Amin Nikanjam, Foutse Khomh  

**Link**: [PDF](https://arxiv.org/pdf/2504.05500)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) has outpaced traditional evaluation methods. Static benchmarks fail to capture the depth and breadth of LLM capabilities and eventually become obsolete, while most dynamic approaches either rely too heavily on LLM-based evaluation or remain constrained by predefined test sets. We introduce Prism, a flexible, dynamic benchmarking framework designed for comprehensive LLM assessment. Prism builds on three key components: (1) a tree-based state representation that models evaluation as a Markov Decision Process, (2) a Monte Carlo Tree Search algorithm adapted to uncover challenging evaluation scenarios, and (3) a multi-agent evaluation pipeline that enables simultaneous assessment of diverse capabilities. To ensure robust evaluation, Prism integrates structural measurements of tree exploration patterns with performance metrics across difficulty levels, providing detailed diagnostics of error patterns, test coverage, and solution approaches. Through extensive experiments on five state-of-the-art LLMs, we analyze how model architecture and scale influence code generation performance across varying task difficulties. Our results demonstrate Prism's effectiveness as a dynamic benchmark that evolves with model advancements while offering deeper insights into their limitations. 

---
# EduPlanner: LLM-Based Multi-Agent Systems for Customized and Intelligent Instructional Design 

**Authors**: Xueqiao Zhang, Chao Zhang, Jianwen Sun, Jun Xiao, Yi Yang, Yawei Luo  

**Link**: [PDF](https://arxiv.org/pdf/2504.05370)  

**Abstract**: Large Language Models (LLMs) have significantly advanced smart education in the Artificial General Intelligence (AGI) era. A promising application lies in the automatic generalization of instructional design for curriculum and learning activities, focusing on two key aspects: (1) Customized Generation: generating niche-targeted teaching content based on students' varying learning abilities and states, and (2) Intelligent Optimization: iteratively optimizing content based on feedback from learning effectiveness or test scores. Currently, a single large LLM cannot effectively manage the entire process, posing a challenge for designing intelligent teaching plans. To address these issues, we developed EduPlanner, an LLM-based multi-agent system comprising an evaluator agent, an optimizer agent, and a question analyst, working in adversarial collaboration to generate customized and intelligent instructional design for curriculum and learning activities. Taking mathematics lessons as our example, EduPlanner employs a novel Skill-Tree structure to accurately model the background mathematics knowledge of student groups, personalizing instructional design for curriculum and learning activities according to students' knowledge levels and learning abilities. Additionally, we introduce the CIDDP, an LLM-based five-dimensional evaluation module encompassing clarity, Integrity, Depth, Practicality, and Pertinence, to comprehensively assess mathematics lesson plan quality and bootstrap intelligent optimization. Experiments conducted on the GSM8K and Algebra datasets demonstrate that EduPlanner excels in evaluating and optimizing instructional design for curriculum and learning activities. Ablation studies further validate the significance and effectiveness of each component within the framework. Our code is publicly available at this https URL 

---
# GOLLuM: Gaussian Process Optimized LLMs -- Reframing LLM Finetuning through Bayesian Optimization 

**Authors**: Bojana Ranković, Philippe Schwaller  

**Link**: [PDF](https://arxiv.org/pdf/2504.06265)  

**Abstract**: Large Language Models (LLMs) can encode complex relationships in their latent spaces, yet harnessing them for optimization under uncertainty remains challenging. We address this gap with a novel architecture that reframes LLM finetuning as Gaussian process (GP) marginal likelihood optimization via deep kernel methods. We introduce LLM-based deep kernels, jointly optimized with GPs to preserve the benefits of both - LLMs to provide a rich and flexible input space for Bayesian optimization and - GPs to model this space with predictive uncertainty for more efficient sampling. Applied to Buchwald-Hartwig reaction optimization, our method nearly doubles the discovery rate of high-performing reactions compared to static LLM embeddings (from 24% to 43% coverage of the top 5% reactions in just 50 optimization iterations). We also observe a 14% improvement over domain-specific representations without requiring specialized features. Extensive empirical evaluation across 19 benchmarks - ranging from general chemistry to reaction and molecular property optimization - demonstrates our method's robustness, generality, and consistent improvements across: (1) tasks, (2) LLM architectures (encoder, decoder, encoder-decoder), (3) pretraining domains (chemistry-related or general-purpose) and (4) hyperparameter settings (tuned once on a single dataset). Finally, we explain these improvements: joint LLM-GP optimization through marginal likelihood implicitly performs contrastive learning, aligning representations to produce (1) better-structured embedding spaces, (2) improved uncertainty calibration, and (3) more efficient sampling - without requiring any external loss. This work provides both practical advances in sample-efficient optimization and insights into what makes effective Bayesian optimization. 

---
# From Superficial to Deep: Integrating External Knowledge for Follow-up Question Generation Using Knowledge Graph and LLM 

**Authors**: Jianyu Liu, Yi Huang, Sheng Bi, Junlan Feng, Guilin Qi  

**Link**: [PDF](https://arxiv.org/pdf/2504.05801)  

**Abstract**: In a conversational system, dynamically generating follow-up questions based on context can help users explore information and provide a better user experience. Humans are usually able to ask questions that involve some general life knowledge and demonstrate higher order cognitive skills. However, the questions generated by existing methods are often limited to shallow contextual questions that are uninspiring and have a large gap to the human level. In this paper, we propose a three-stage external knowledge-enhanced follow-up question generation method, which generates questions by identifying contextual topics, constructing a knowledge graph (KG) online, and finally combining these with a large language model to generate the final question. The model generates information-rich and exploratory follow-up questions by introducing external common sense knowledge and performing a knowledge fusion operation. Experiments show that compared to baseline models, our method generates questions that are more informative and closer to human questioning levels while maintaining contextual relevance. 

---
# ARLO: A Tailorable Approach for Transforming Natural Language Software Requirements into Architecture using LLMs 

**Authors**: Tooraj Helmi  

**Link**: [PDF](https://arxiv.org/pdf/2504.06143)  

**Abstract**: Software requirements expressed in natural language (NL) frequently suffer from verbosity, ambiguity, and inconsistency. This creates a range of challenges, including selecting an appropriate architecture for a system and assessing different architectural alternatives. Relying on human expertise to accomplish the task of mapping NL requirements to architecture is time-consuming and error-prone. This paper proposes ARLO, an approach that automates this task by leveraging (1) a set of NL requirements for a system, (2) an existing standard that specifies architecturally relevant software quality attributes, and (3) a readily available Large Language Model (LLM). Specifically, ARLO determines the subset of NL requirements for a given system that is architecturally relevant and maps that subset to a tailorable matrix of architectural choices. ARLO applies integer linear programming on the architectural-choice matrix to determine the optimal architecture for the current requirements. We demonstrate ARLO's efficacy using a set of real-world examples. We highlight ARLO's ability (1) to trace the selected architectural choices to the requirements and (2) to isolate NL requirements that exert a particular influence on a system's architecture. This allows the identification, comparative assessment, and exploration of alternative architectural choices based on the requirements and constraints expressed therein. 

---
# Optuna vs Code Llama: Are LLMs a New Paradigm for Hyperparameter Tuning? 

**Authors**: Roman Kochnev, Arash Torabi Goodarzi, Zofia Antonina Bentyn, Dmitry Ignatov, Radu Timofte  

**Link**: [PDF](https://arxiv.org/pdf/2504.06006)  

**Abstract**: Optimal hyperparameter selection is critical for maximizing neural network performance, especially as models grow in complexity. This work investigates the viability of using large language models (LLMs) for hyperparameter optimization by employing a fine-tuned version of Code Llama. Through parameter-efficient fine-tuning using LoRA, we adapt the LLM to generate accurate and efficient hyperparameter recommendations tailored to diverse neural network architectures. Unlike traditional methods such as Optuna, which rely on exhaustive trials, the proposed approach achieves competitive or superior results in terms of Root Mean Square Error (RMSE) while significantly reducing computational overhead. Our approach highlights that LLM-based optimization not only matches state-of-the-art methods like Tree-structured Parzen Estimators but also accelerates the tuning process. This positions LLMs as a promising alternative to conventional optimization techniques, particularly for rapid experimentation. Furthermore, the ability to generate hyperparameters in a single inference step makes this method particularly well-suited for resource-constrained environments such as edge devices and mobile applications, where computational efficiency is paramount. The results confirm that LLMs, beyond their efficiency, offer substantial time savings and comparable stability, underscoring their value in advancing machine learning workflows. All generated hyperparameters are included in the LEMUR Neural Network (NN) Dataset, which is publicly available and serves as an open-source benchmark for hyperparameter optimization research. 

---
# MDK12-Bench: A Multi-Discipline Benchmark for Evaluating Reasoning in Multimodal Large Language Models 

**Authors**: Pengfei Zhou, Fanrui Zhang, Xiaopeng Peng, Zhaopan Xu, Jiaxin Ai, Yansheng Qiu, Chuanhao Li, Zhen Li, Ming Li, Yukang Feng, Jianwen Sun, Haoquan Zhang, Zizhen Li, Xiaofeng Mao, Wangbo Zhao, Kai Wang, Xiaojun Chang, Wenqi Shao, Yang You, Kaipeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.05782)  

**Abstract**: Multimodal reasoning, which integrates language and visual cues into problem solving and decision making, is a fundamental aspect of human intelligence and a crucial step toward artificial general intelligence. However, the evaluation of multimodal reasoning capabilities in Multimodal Large Language Models (MLLMs) remains inadequate. Most existing reasoning benchmarks are constrained by limited data size, narrow domain coverage, and unstructured knowledge distribution. To close these gaps, we introduce MDK12-Bench, a multi-disciplinary benchmark assessing the reasoning capabilities of MLLMs via real-world K-12 examinations. Spanning six disciplines (math, physics, chemistry, biology, geography, and information science), our benchmark comprises 140K reasoning instances across diverse difficulty levels from primary school to 12th grade. It features 6,827 instance-level knowledge point annotations based on a well-organized knowledge structure, detailed answer explanations, difficulty labels and cross-year partitions, providing a robust platform for comprehensive evaluation. Additionally, we present a novel dynamic evaluation framework to mitigate data contamination issues by bootstrapping question forms, question types, and image styles during evaluation. Extensive experiment on MDK12-Bench reveals the significant limitation of current MLLMs in multimodal reasoning. The findings on our benchmark provide insights into the development of the next-generation models. Our data and codes are available at this https URL. 

---
# How to Enable LLM with 3D Capacity? A Survey of Spatial Reasoning in LLM 

**Authors**: Jirong Zha, Yuxuan Fan, Xiao Yang, Chen Gao, Xinlei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.05786)  

**Abstract**: 3D spatial understanding is essential in real-world applications such as robotics, autonomous vehicles, virtual reality, and medical imaging. Recently, Large Language Models (LLMs), having demonstrated remarkable success across various domains, have been leveraged to enhance 3D understanding tasks, showing potential to surpass traditional computer vision methods. In this survey, we present a comprehensive review of methods integrating LLMs with 3D spatial understanding. We propose a taxonomy that categorizes existing methods into three branches: image-based methods deriving 3D understanding from 2D visual data, point cloud-based methods working directly with 3D representations, and hybrid modality-based methods combining multiple data streams. We systematically review representative methods along these categories, covering data representations, architectural modifications, and training strategies that bridge textual and 3D modalities. Finally, we discuss current limitations, including dataset scarcity and computational challenges, while highlighting promising research directions in spatial perception, multi-modal fusion, and real-world applications. 

---
# Debate-Feedback: A Multi-Agent Framework for Efficient Legal Judgment Prediction 

**Authors**: Xi Chen, Mao Mao, Shuo Li, Haotian Shangguan  

**Link**: [PDF](https://arxiv.org/pdf/2504.05358)  

**Abstract**: The use of AI in legal analysis and prediction (LegalAI) has gained widespread attention, with past research focusing on retrieval-based methods and fine-tuning large models. However, these approaches often require large datasets and underutilize the capabilities of modern large language models (LLMs). In this paper, inspired by the debate phase of real courtroom trials, we propose a novel legal judgment prediction model based on the Debate-Feedback architecture, which integrates LLM multi-agent debate and reliability evaluation models. Unlike traditional methods, our model achieves significant improvements in efficiency by minimizing the need for large historical datasets, thus offering a lightweight yet robust solution. Comparative experiments show that it outperforms several general-purpose and domain-specific legal models, offering a dynamic reasoning process and a promising direction for future LegalAI research. 

---
# When Reasoning Meets Compression: Benchmarking Compressed Large Reasoning Models on Complex Reasoning Tasks 

**Authors**: Nan Zhang, Yusen Zhang, Prasenjit Mitra, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.02010)  

**Abstract**: Recent open-source large reasoning models (LRMs) exhibit strong performance on complex reasoning tasks, but their large parameter count makes them prohibitively expensive for individuals. The compression of large language models (LLMs) offers an effective solution to reduce cost of computational resources. However, systematic studies on the performance of compressed LLMs in complex reasoning tasks, especially for LRMs, are lacking. Most works on quantization and pruning focus on preserving language modeling performance, while existing distillation works do not comprehensively benchmark student models based on reasoning difficulty or compression impact on knowledge and reasoning. In this paper, we benchmark compressed DeepSeek-R1 models on four different reasoning datasets (AIME 2024, FOLIO, Temporal Sequences of BIG-Bench Hard, and MuSiQue), ranging from mathematical to multihop reasoning, using quantization, distillation, and pruning methods. We benchmark 2.51-, 1.73-, and 1.58-bit R1 models that adopt dynamic quantization. We also benchmark distilled R1 models that are based on LLaMA or Qwen and run SparseGPT on them to obtain various sparsity levels. Studying the performance and behavior of compressed LRMs, we report their performance scores and test-time compute (number of tokens spent on each question). Notably, using MuSiQue, we find that parameter count has a much greater impact on LRMs' knowledge memorization than on their reasoning capability, which can inform the choice of compression techniques. Through our empirical analysis of test-time compute, we find that shorter model outputs generally achieve better performance than longer ones across several benchmarks for both R1 and its compressed variants, highlighting the need for more concise reasoning chains. 

---
