# SAIL-Embedding Technical Report: Omni-modal Embedding Foundation Model 

**Authors**: Lin Lin, Jiefeng Long, Zhihe Wan, Yuchi Wang, Dingkang Yang, Shuang Yang, Yueyang Yao, Xu Chen, Zirui Guo, Shengqiang Li, Weiran Li, Hanyu Li, Yaling Mou, Yan Qiu, Haiyang Yu, Xiao Liang, Hongsheng Li, Chao Feng  

**Link**: [PDF](https://arxiv.org/pdf/2510.12709)  

**Abstract**: Multimodal embedding models aim to yield informative unified representations that empower diverse cross-modal tasks. Despite promising developments in the evolution from CLIP-based dual-tower architectures to large vision-language models, prior works still face unavoidable challenges in real-world applications and business scenarios, such as the limited modality support, unstable training mechanisms, and industrial domain gaps. In this work, we introduce SAIL-Embedding, an omni-modal embedding foundation model that addresses these issues through tailored training strategies and architectural design. In the optimization procedure, we propose a multi-stage training scheme to boost the multifaceted effectiveness of representation learning. Specifically, the content-aware progressive training aims to enhance the model's adaptability to diverse downstream tasks and master enriched cross-modal proficiency. The collaboration-aware recommendation enhancement training further adapts multimodal representations for recommendation scenarios by distilling knowledge from sequence-to-item and ID-to-item embeddings while mining user historical interests. Concurrently, we develop the stochastic specialization and dataset-driven pattern matching to strengthen model training flexibility and generalizability. Experimental results show that SAIL-Embedding achieves SOTA performance compared to other methods in different retrieval tasks. In online experiments across various real-world scenarios integrated with our model, we observe a significant increase in Lifetime (LT), which is a crucial indicator for the recommendation experience. For instance, the model delivers the 7-day LT gain of +0.158% and the 14-day LT gain of +0.144% in the Douyin-Selected scenario. For the Douyin feed rank model, the match features produced by SAIL-Embedding yield a +0.08% AUC gain. 

---
# The Role of Parametric Injection-A Systematic Study of Parametric Retrieval-Augmented Generation 

**Authors**: Minghao Tang, Shiyu Ni, Jingtong Wu, Zengxin Han, Keping Bi  

**Link**: [PDF](https://arxiv.org/pdf/2510.12668)  

**Abstract**: Retrieval-augmented generation (RAG) enhances large language models (LLMs) by retrieving external documents. As an emerging form of RAG, parametric retrieval-augmented generation (PRAG) encodes documents as model parameters (i.e., LoRA modules) and injects these representations into the model during inference, enabling interaction between the LLM and documents at parametric level. Compared with directly placing documents in the input context, PRAG is more efficient and has the potential to offer deeper model-document interaction. Despite its growing attention, the mechanism underlying parametric injection remains poorly understood. In this work, we present a systematic study of PRAG to clarify the role of parametric injection, showing that parameterized documents capture only partial semantic information of documents, and relying on them alone yields inferior performance compared to interaction at text level. However, these parametric representations encode high-level document information that can enhance the model's understanding of documents within the input context. When combined parameterized documents with textual documents, the model can leverage relevant information more effectively and become more robust to noisy inputs, achieving better performance than either source alone. We recommend jointly using parameterized and textual documents and advocate for increasing the information content of parametric representations to advance PRAG. 

---
# SMILE: SeMantic Ids Enhanced CoLd Item Representation for Click-through Rate Prediction in E-commerce SEarch 

**Authors**: Qihang Zhao, Zhongbo Sun, Xiaoyang Zheng, Xian Guo, Siyuan Wang, Zihan Liang, Mingcan Peng, Ben Chen, Chenyi Lei  

**Link**: [PDF](https://arxiv.org/pdf/2510.12604)  

**Abstract**: With the rise of modern search and recommendation platforms, insufficient collaborative information of cold-start items exacerbates the Matthew effect of existing platform items, challenging platform diversity and becoming a longstanding issue. Existing methods align items' side content with collaborative information to transfer collaborative signals from high-popularity items to cold-start items. However, these methods fail to account for the asymmetry between collaboration and content, nor the fine-grained differences among items. To address these issues, we propose SMILE, an item representation enhancement approach based on fused alignment of semantic IDs. Specifically, we use RQ-OPQ encoding to quantize item content and collaborative information, followed by a two-step alignment: RQ encoding transfers shared collaborative signals across items, while OPQ encoding learns differentiated information of items. Comprehensive offline experiments on large-scale industrial datasets demonstrate superiority of SMILE, and rigorous online A/B tests confirm statistically significant improvements: item CTR +1.66%, buyers +1.57%, and order volume +2.17%. 

---
# Leveraging Language Semantics for Collaborative Filtering with TextGCN and TextGCN-MLP: Zero-Shot vs In-Domain Performance 

**Authors**: Andrei Chernov, Haroon Wahab, Oleg Novitskij  

**Link**: [PDF](https://arxiv.org/pdf/2510.12461)  

**Abstract**: In recent years, various approaches have been proposed to leverage large language models (LLMs) for incorporating textual information about items into recommender systems. Existing methods primarily focus on either fine-tuning LLMs to generate recommendations or integrating LLM-based embeddings into downstream models. In this work, we follow the latter direction and propose \textbf{TextGCN}, which applies parameter-free graph convolution layers directly over LLM-based item-title embeddings, instead of learning ID-based embeddings as in traditional methods. By combining language semantics with graph message passing, this architecture achieves state-of-the-art zero-shot performance, significantly outperforming prior approaches. Furthermore, we introduce \textbf{TextGCN-MLP}, which extends TextGCN with a trainable multilayer perceptron trained using a contrastive loss, achieving state-of-the-art in-domain performance on recommendation benchmarks. However, the zero-shot performance of TextGCN-MLP remains lower than that of TextGCN, highlighting the trade-off between in-domain specialization and zero-shot generalization. We release our code on github at \href{this https URL}{this http URL}. 

---
# A Hierarchical Quantized Tokenization Framework for Task-Adaptive Graph Representation Learning 

**Authors**: Yang Xiang, Li Fan, Chenke Yin, Chengtao Ji  

**Link**: [PDF](https://arxiv.org/pdf/2510.12369)  

**Abstract**: Recent progress in language and vision foundation models demonstrates the importance of discrete token interfaces that transform complex inputs into compact sequences for large-scale modeling. Extending this paradigm to graphs requires a tokenization scheme that handles non-Euclidean structures and multi-scale dependencies efficiently. Existing approaches to graph tokenization, linearized, continuous, and quantized, remain limited in adaptability and efficiency. In particular, most current quantization-based tokenizers organize hierarchical information in fixed or task-agnostic ways, which may either over-represent or under-utilize structural cues, and lack the ability to dynamically reweight contributions from different levels without retraining the encoder. This work presents a hierarchical quantization framework that introduces a self-weighted mechanism for task-adaptive aggregation across multiple scales. The proposed method maintains a frozen encoder while modulating information flow through a lightweight gating process, enabling parameter-efficient adaptation to diverse downstream tasks. Experiments on benchmark datasets for node classification and link prediction demonstrate consistent improvements over strong baselines under comparable computational budgets. 

---
# Simple Projection Variants Improve ColBERT Performance 

**Authors**: Benjamin Clavié, Sean Lee, Rikiya Takehi, Aamir Shakir, Makoto P. Kato  

**Link**: [PDF](https://arxiv.org/pdf/2510.12327)  

**Abstract**: Multi-vector dense retrieval methods like ColBERT systematically use a single-layer linear projection to reduce the dimensionality of individual vectors. In this study, we explore the implications of the MaxSim operator on the gradient flows of the training of multi-vector models and show that such a simple linear projection has inherent, if non-critical, limitations in this setting. We then discuss the theoretical improvements that could result from replacing this single-layer projection with well-studied alternative feedforward linear networks (FFN), such as deeper, non-linear FFN blocks, GLU blocks, and skip-connections, could alleviate these limitations. Through the design and systematic evaluation of alternate projection blocks, we show that better-designed final projections positively impact the downstream performance of ColBERT models. We highlight that many projection variants outperform the original linear projections, with the best-performing variants increasing average performance on a range of retrieval benchmarks across domains by over 2 NDCG@10 points. We then conduct further exploration on the individual parameters of these projections block in order to understand what drives this empirical performance, highlighting the particular importance of upscaled intermediate projections and residual connections. As part of these ablation studies, we show that numerous suboptimal projection variants still outperform the traditional single-layer projection across multiple benchmarks, confirming our hypothesis. Finally, we observe that this effect is consistent across random seeds, further confirming that replacing the linear layer of ColBERT models is a robust, drop-in upgrade. 

---
# Causal Inspired Multi Modal Recommendation 

**Authors**: Jie Yang, Chenyang Gu, Zixuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.12325)  

**Abstract**: Multimodal recommender systems enhance personalized recommendations in e-commerce and online advertising by integrating visual, textual, and user-item interaction data. However, existing methods often overlook two critical biases: (i) modal confounding, where latent factors (e.g., brand style or product category) simultaneously drive multiple modalities and influence user preference, leading to spurious feature-preference associations; (ii) interaction bias, where genuine user preferences are mixed with noise from exposure effects and accidental clicks. To address these challenges, we propose a Causal-inspired multimodal Recommendation framework. Specifically, we introduce a dual-channel cross-modal diffusion module to identify hidden modal confounders, utilize back-door adjustment with hierarchical matching and vector-quantized codebooks to block confounding paths, and apply front-door adjustment combined with causal topology reconstruction to build a deconfounded causal subgraph. Extensive experiments on three real-world e-commerce datasets demonstrate that our method significantly outperforms state-of-the-art baselines while maintaining strong interpretability. 

---
# An Empirical Study for Representations of Videos in Video Question Answering via MLLMs 

**Authors**: Zhi Li, Yanan Wang, Hao Niu, Julio Vizcarra, Masato Taya  

**Link**: [PDF](https://arxiv.org/pdf/2510.12299)  

**Abstract**: Multimodal large language models have recently achieved remarkable progress in video question answering (VideoQA) by jointly processing visual, textual, and audio information. However, it remains unclear which video representations are most effective for MLLMs, and how different modalities balance task accuracy against computational efficiency. In this work, we present a comprehensive empirical study of video representation methods for VideoQA with MLLMs. We systematically evaluate single modality inputs question only, subtitles, visual frames, and audio signals as well as multimodal combinations, on two widely used benchmarks: VideoMME and LongVideoBench. Our results show that visual frames substantially enhance accuracy but impose heavy costs in GPU memory and inference latency, while subtitles provide a lightweight yet effective alternative, particularly for long videos. These findings highlight clear trade-offs between effectiveness and efficiency and provide practical insights for designing resource-aware MLLM-based VideoQA systems. 

---
# Reinforced Preference Optimization for Recommendation 

**Authors**: Junfei Tan, Yuxin Chen, An Zhang, Junguang Jiang, Bin Liu, Ziru Xu, Han Zhu, Jian Xu, Bo Zheng, Xiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.12211)  

**Abstract**: Recent breakthroughs in large language models (LLMs) have fundamentally shifted recommender systems from discriminative to generative paradigms, where user behavior modeling is achieved by generating target items conditioned on historical interactions. Yet current generative recommenders still suffer from two core limitations: the lack of high-quality negative modeling and the reliance on implicit rewards. Reinforcement learning with verifiable rewards (RLVR) offers a natural solution by enabling on-policy sampling of harder negatives and grounding optimization in explicit reward signals. However, applying RLVR to generative recommenders remains non-trivial. Its unique generation space often leads to invalid or repetitive items that undermine sampling efficiency, and ranking supervision is sparse since most items receive identical zero rewards. To address these challenges, we propose Reinforced Preference Optimization for Recommendation (ReRe), a reinforcement-based paradigm tailored to LLM-based recommenders, an important direction in generative recommendation. ReRe incorporates constrained beam search to improve sampling efficiency and diversify hard negatives, while augmenting rule-based accuracy rewards with auxiliary ranking rewards for finer-grained supervision. Extensive experiments on three real-world datasets demonstrate that ReRe consistently outperforms both traditional and LLM-based recommenders in ranking performance. Further analysis shows that ReRe not only enhances performance across both base and SFT-initialized models but also generalizes robustly across different backbone families and scales. Beyond empirical gains, we systematically investigate the design space of RLVR in recommendation across generation, sampling strategy, reward modeling, and optimization algorithm, offering insights for future research. 

---
# MIARec: Mutual-influence-aware Heterogeneous Network Embedding for Scientific Paper Recommendation 

**Authors**: Wenjin Xie, Tao Jia  

**Link**: [PDF](https://arxiv.org/pdf/2510.12054)  

**Abstract**: With the rapid expansion of scientific literature, scholars increasingly demand precise and high-quality paper recommendations. Among various recommendation methodologies, graph-based approaches have garnered attention by effectively exploiting the structural characteristics inherent in scholarly networks. However, these methods often overlook the asymmetric academic influence that is prevalent in scholarly networks when learning graph representations. To address this limitation, this study proposes the Mutual-Influence-Aware Recommendation (MIARec) model, which employs a gravity-based approach to measure the mutual academic influence between scholars and incorporates this influence into the feature aggregation process during message propagation in graph representation learning. Additionally, the model utilizes a multi-channel aggregation method to capture both individual embeddings of distinct single relational sub-networks and their interdependent embeddings, thereby enabling a more comprehensive understanding of the heterogeneous scholarly network. Extensive experiments conducted on real-world datasets demonstrate that the MIARec model outperforms baseline models across three primary evaluation metrics, indicating its effectiveness in scientific paper recommendation tasks. 

---
# Embedding the Teacher: Distilling vLLM Preferences for Scalable Image Retrieval 

**Authors**: Eric He, Akash Gupta, Adian Liusie, Vatsal Raina, Piotr Molenda, Shirom Chabra, Vyas Raina  

**Link**: [PDF](https://arxiv.org/pdf/2510.12014)  

**Abstract**: Text--image retrieval is necessary for applications such as product recommendation. Embedding-based approaches like CLIP enable efficient large-scale retrieval via vector similarity search, but they are primarily trained on literal caption-like text--image pairs and often fail to capture abstract or persona-driven attributes common in product recommendation applications (e.g., ``a gift for a mother who loves gardening''). In contrast, state-of-the-art vision--language models (vLLMs) can align text with images in a flexible manner, but their limited context window prevents them from directly handling retrieval over large catalogs. We propose a framework that distills the preference rankings of a powerful vLLM into an embedding-based system, transferring its nuanced alignment abilities while maintaining the inference-time scalability of an embedding-based approach. Experiments on persona-driven product recommendation tasks demonstrate that our method significantly outperforms existing embedding-based baselines, providing an efficient solution for personalized text--image retrieval. 

---
# DeepMMSearch-R1: Empowering Multimodal LLMs in Multimodal Web Search 

**Authors**: Kartik Narayan, Yang Xu, Tian Cao, Kavya Nerella, Vishal M. Patel, Navid Shiee, Peter Grasch, Chao Jia, Yinfei Yang, Zhe Gan  

**Link**: [PDF](https://arxiv.org/pdf/2510.12801)  

**Abstract**: Multimodal Large Language Models (MLLMs) in real-world applications require access to external knowledge sources and must remain responsive to the dynamic and ever-changing real-world information in order to address information-seeking and knowledge-intensive user queries. Existing approaches, such as retrieval augmented generation (RAG) methods, search agents, and search equipped MLLMs, often suffer from rigid pipelines, excessive search calls, and poorly constructed search queries, which result in inefficiencies and suboptimal outcomes. To address these limitations, we present DeepMMSearch-R1, the first multimodal LLM capable of performing on-demand, multi-turn web searches and dynamically crafting queries for both image and text search tools. Specifically, DeepMMSearch-R1 can initiate web searches based on relevant crops of the input image making the image search more effective, and can iteratively adapt text search queries based on retrieved information, thereby enabling self-reflection and self-correction. Our approach relies on a two-stage training pipeline: a cold start supervised finetuning phase followed by an online reinforcement learning optimization. For training, we introduce DeepMMSearchVQA, a novel multimodal VQA dataset created through an automated pipeline intermixed with real-world information from web search tools. This dataset contains diverse, multi-hop queries that integrate textual and visual information, teaching the model when to search, what to search for, which search tool to use and how to reason over the retrieved information. We conduct extensive experiments across a range of knowledge-intensive benchmarks to demonstrate the superiority of our approach. Finally, we analyze the results and provide insights that are valuable for advancing multimodal web-search. 

---
# CTRL-Rec: Controlling Recommender Systems With Natural Language 

**Authors**: Micah Carroll, Adeline Foote, Kevin Feng, Marcus Williams, Anca Dragan, W. Bradley Knox, Smitha Milli  

**Link**: [PDF](https://arxiv.org/pdf/2510.12742)  

**Abstract**: When users are dissatisfied with recommendations from a recommender system, they often lack fine-grained controls for changing them. Large language models (LLMs) offer a solution by allowing users to guide their recommendations through natural language requests (e.g., "I want to see respectful posts with a different perspective than mine"). We propose a method, CTRL-Rec, that allows for natural language control of traditional recommender systems in real-time with computational efficiency. Specifically, at training time, we use an LLM to simulate whether users would approve of items based on their language requests, and we train embedding models that approximate such simulated judgments. We then integrate these user-request-based predictions into the standard weighting of signals that traditional recommender systems optimize. At deployment time, we require only a single LLM embedding computation per user request, allowing for real-time control of recommendations. In experiments with the MovieLens dataset, our method consistently allows for fine-grained control across a diversity of requests. In a study with 19 Letterboxd users, we find that CTRL-Rec was positively received by users and significantly enhanced users' sense of control and satisfaction with recommendations compared to traditional controls. 

---
# Evaluating Retrieval-Augmented Generation Systems on Unanswerable, Uncheatable, Realistic, Multi-hop Queries 

**Authors**: Gabrielle Kaili-May Liu, Bryan Li, Arman Cohan, William Gantt Walden, Eugene Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.11956)  

**Abstract**: Real-world use cases often present RAG systems with complex queries for which relevant information is missing from the corpus or is incomplete. In these settings, RAG systems must be able to reject unanswerable, out-of-scope queries and identify failures of retrieval and multi-hop reasoning. Despite this, existing RAG benchmarks rarely reflect realistic task complexity for multi-hop or out-of-scope questions, which often can be cheated via disconnected reasoning (i.e., solved without genuine multi-hop inference) or require only simple factual recall. This limits the ability for such benchmarks to uncover limitations of existing RAG systems. To address this gap, we present the first pipeline for automatic, difficulty-controlled creation of un$\underline{c}$heatable, $\underline{r}$ealistic, $\underline{u}$nanswerable, and $\underline{m}$ulti-hop $\underline{q}$uerie$\underline{s}$ (CRUMQs), adaptable to any corpus and domain. We use our pipeline to create CRUMQs over two popular RAG datasets and demonstrate its effectiveness via benchmark experiments on leading retrieval-augmented LLMs. Results show that compared to prior RAG benchmarks, CRUMQs are highly challenging for RAG systems and achieve up to 81.0\% reduction in cheatability scores. More broadly, our pipeline offers a simple way to enhance benchmark difficulty and realism and drive development of more capable RAG systems. 

---
