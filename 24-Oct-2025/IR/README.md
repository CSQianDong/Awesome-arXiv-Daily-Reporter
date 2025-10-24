# Generative Reasoning Recommendation via LLMs 

**Authors**: Minjie Hong, Zetong Zhou, Zirun Guo, Ziang Zhang, Ruofan Hu, Weinan Gan, Jieming Zhu, Zhou Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.20815)  

**Abstract**: Despite their remarkable reasoning capabilities across diverse domains, large language models (LLMs) face fundamental challenges in natively functioning as generative reasoning recommendation models (GRRMs), where the intrinsic modeling gap between textual semantics and collaborative filtering signals, combined with the sparsity and stochasticity of user feedback, presents significant obstacles. This work explores how to build GRRMs by adapting pre-trained LLMs, which achieves a unified understanding-reasoning-prediction manner for recommendation tasks. We propose GREAM, an end-to-end framework that integrates three components: (i) Collaborative-Semantic Alignment, which fuses heterogeneous textual evidence to construct semantically consistent, discrete item indices and auxiliary alignment tasks that ground linguistic representations in interaction semantics; (ii) Reasoning Curriculum Activation, which builds a synthetic dataset with explicit Chain-of-Thought supervision and a curriculum that progresses through behavioral evidence extraction, latent preference modeling, intent inference, recommendation formulation, and denoised sequence rewriting; and (iii) Sparse-Regularized Group Policy Optimization (SRPO), which stabilizes post-training via Residual-Sensitive Verifiable Reward and Bonus-Calibrated Group Advantage Estimation, enabling end-to-end optimization under verifiable signals despite sparse successes. GREAM natively supports two complementary inference modes: Direct Sequence Recommendation for high-throughput, low-latency deployment, and Sequential Reasoning Recommendation that first emits an interpretable reasoning chain for causal transparency. Experiments on three datasets demonstrate consistent gains over strong baselines, providing a practical path toward verifiable-RL-driven LLM recommenders. 

---
# Analyticup E-commerce Product Search Competition Technical Report from Team Tredence_AICOE 

**Authors**: Rakshith R, Shubham Sharma, Mohammed Sameer Khan, Ankush Chopra  

**Link**: [PDF](https://arxiv.org/pdf/2510.20674)  

**Abstract**: This study presents the multilingual e-commerce search system developed by the Tredence_AICOE team. The competition features two multilingual relevance tasks: Query-Category (QC) Relevance, which evaluates how well a user's search query aligns with a product category, and Query-Item (QI) Relevance, which measures the match between a multilingual search query and an individual product listing. To ensure full language coverage, we performed data augmentation by translating existing datasets into languages missing from the development set, enabling training across all target languages. We fine-tuned Gemma-3 12B and Qwen-2.5 14B model for both tasks using multiple strategies. The Gemma-3 12B (4-bit) model achieved the best QC performance using original and translated data, and the best QI performance using original, translated, and minority class data creation. These approaches secured 4th place on the final leaderboard, with an average F1-score of 0.8857 on the private test set. 

---
# Rotate Both Ways: Time-and-Order RoPE for Generative Recommendation 

**Authors**: Xiaokai Wei, Jiajun Wu, Daiyao Yi, Reza Shirkavand, Michelle Gong  

**Link**: [PDF](https://arxiv.org/pdf/2510.20455)  

**Abstract**: Generative recommenders, typically transformer-based autoregressive models, predict the next item or action from a user's interaction history. Their effectiveness depends on how the model represents where an interaction event occurs in the sequence (discrete index) and when it occurred in wall-clock time. Prevailing approaches inject time via learned embeddings or relative attention biases. In this paper, we argue that RoPE-based approaches, if designed properly, can be a stronger alternative for jointly modeling temporal and sequential information in user behavior sequences. While vanilla RoPE in LLMs considers only token order, generative recommendation requires incorporating both event time and token index. To address this, we propose Time-and-Order RoPE (TO-RoPE), a family of rotary position embedding designs that treat index and time as angle sources shaping the query-key geometry directly. We present three instantiations: early fusion, split-by-dim, and split-by-head. Extensive experiments on both publicly available datasets and a proprietary industrial dataset show that TO-RoPE variants consistently improve accuracy over existing methods for encoding time and index. These results position rotary embeddings as a simple, principled, and deployment-friendly foundation for generative recommendation. 

---
# From Generation to Attribution: Music AI Agent Architectures for the Post-Streaming Era 

**Authors**: Wonil Kim, Hyeongseok Wi, Seungsoon Park, Taejun Kim, Sangeun Keum, Keunhyoung Kim, Taewan Kim, Jongmin Jung, Taehyoung Kim, Gaetan Guerrero, Mael Le Goff, Julie Po, Dongjoo Moon, Juhan Nam, Jongpil Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.20276)  

**Abstract**: Generative AI is reshaping music creation, but its rapid growth exposes structural gaps in attribution, rights management, and economic models. Unlike past media shifts, from live performance to recordings, downloads, and streaming, AI transforms the entire lifecycle of music, collapsing boundaries between creation, distribution, and monetization. However, existing streaming systems, with opaque and concentrated royalty flows, are ill-equipped to handle the scale and complexity of AI-driven production. We propose a content-based Music AI Agent architecture that embeds attribution directly into the creative workflow through block-level retrieval and agentic orchestration. Designed for iterative, session-based interaction, the system organizes music into granular components (Blocks) stored in BlockDB; each use triggers an Attribution Layer event for transparent provenance and real-time settlement. This framework reframes AI from a generative tool into infrastructure for a Fair AI Media Platform. By enabling fine-grained attribution, equitable compensation, and participatory engagement, it points toward a post-streaming paradigm where music functions not as a static catalog but as a collaborative and adaptive ecosystem. 

---
# Balancing Fine-tuning and RAG: A Hybrid Strategy for Dynamic LLM Recommendation Updates 

**Authors**: Changping Meng, Hongyi Ling, Jianling Wang, Yifan Liu, Shuzhou Zhang, Dapeng Hong, Mingyan Gao, Onkar Dalal, Ed Chi, Lichan Hong, Haokai Lu, Ningren Han  

**Link**: [PDF](https://arxiv.org/pdf/2510.20260)  

**Abstract**: Large Language Models (LLMs) empower recommendation systems through their advanced reasoning and planning capabilities. However, the dynamic nature of user interests and content poses a significant challenge: While initial fine-tuning aligns LLMs with domain knowledge and user preferences, it fails to capture such real-time changes, necessitating robust update mechanisms. This paper investigates strategies for updating LLM-powered recommenders, focusing on the trade-offs between ongoing fine-tuning and Retrieval-Augmented Generation (RAG). Using an LLM-powered user interest exploration system as a case study, we perform a comparative analysis of these methods across dimensions like cost, agility, and knowledge incorporation. We propose a hybrid update strategy that leverages the long-term knowledge adaptation of periodic fine-tuning with the agility of low-cost RAG. We demonstrate through live A/B experiments on a billion-user platform that this hybrid approach yields statistically significant improvements in user satisfaction, offering a practical and cost-effective framework for maintaining high-quality LLM-powered recommender systems. 

---
# Multimedia-Aware Question Answering: A Review of Retrieval and Cross-Modal Reasoning Architectures 

**Authors**: Rahul Raja, Arpita Vats  

**Link**: [PDF](https://arxiv.org/pdf/2510.20193)  

**Abstract**: Question Answering (QA) systems have traditionally relied on structured text data, but the rapid growth of multimedia content (images, audio, video, and structured metadata) has introduced new challenges and opportunities for retrieval-augmented QA. In this survey, we review recent advancements in QA systems that integrate multimedia retrieval pipelines, focusing on architectures that align vision, language, and audio modalities with user queries. We categorize approaches based on retrieval methods, fusion techniques, and answer generation strategies, and analyze benchmark datasets, evaluation protocols, and performance tradeoffs. Furthermore, we highlight key challenges such as cross-modal alignment, latency-accuracy tradeoffs, and semantic grounding, and outline open problems and future research directions for building more robust and context-aware QA systems leveraging multimedia data. 

---
# Rank-GRPO: Training LLM-based Conversational Recommender Systems with Reinforcement Learning 

**Authors**: Yaochen Zhu, Harald Steck, Dawen Liang, Yinhan He, Jundong Li, Nathan Kallus  

**Link**: [PDF](https://arxiv.org/pdf/2510.20150)  

**Abstract**: Large language models (LLMs) are reshaping the recommender system paradigm by enabling users to express preferences and receive recommendations through conversations. Yet, aligning LLMs to the recommendation task remains challenging: pretrained LLMs often generate out-of-catalog items, violate required output formats, and their ranking quality degrades sharply toward the end of the generated list. To this end, we propose ConvRec-R1, a two-stage framework for end-to-end training of LLM-based conversational recommender systems. In Stage 1, we construct a behavioral-cloning dataset with a Remap-Reflect-Adjust pipeline, which produces high-quality, catalog-grounded demonstrations from powerful blackbox LLMs to warm-start the RL training. In Stage 2, we propose Rank-GRPO, a principled extension of group relative policy optimization (GRPO) tailored to tasks with rank-style outputs. Rank-GRPO treats each rank in the recommendation list as the unit instead of token (too fine-grained) or sequence (too coarse), redefining rewards to remove non-causal credit assignment and introducing a rank-level importance ratio based on the geometric mean of rank-wise token probabilities to stabilize policy updates. Experiments on the public Reddit-v2 dataset show that ConvRec-R1 converges faster and achieves higher Recall and NDCG than GRPO-style baselines. Code and datasets are released at this https URL. 

---
# Automating Iconclass: LLMs and RAG for Large-Scale Classification of Religious Woodcuts 

**Authors**: Drew B. Thomas  

**Link**: [PDF](https://arxiv.org/pdf/2510.19986)  

**Abstract**: This paper presents a novel methodology for classifying early modern religious images by using Large Language Models (LLMs) and vector databases in combination with Retrieval-Augmented Generation (RAG). The approach leverages the full-page context of book illustrations from the Holy Roman Empire, allowing the LLM to generate detailed descriptions that incorporate both visual and textual elements. These descriptions are then matched to relevant Iconclass codes through a hybrid vector search. This method achieves 87% and 92% precision at five and four levels of classification, significantly outperforming traditional image and keyword-based searches. By employing full-page descriptions and RAG, the system enhances classification accuracy, offering a powerful tool for large-scale analysis of early modern visual archives. This interdisciplinary approach demonstrates the growing potential of LLMs and RAG in advancing research within art history and digital humanities. 

---
# RAGRank: Using PageRank to Counter Poisoning in CTI LLM Pipelines 

**Authors**: Austin Jia, Avaneesh Ramesh, Zain Shamsi, Daniel Zhang, Alex Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.20768)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as the dominant architectural pattern to operationalize Large Language Model (LLM) usage in Cyber Threat Intelligence (CTI) systems. However, this design is susceptible to poisoning attacks, and previously proposed defenses can fail for CTI contexts as cyber threat information is often completely new for emerging attacks, and sophisticated threat actors can mimic legitimate formats, terminology, and stylistic conventions. To address this issue, we propose that the robustness of modern RAG defenses can be accelerated by applying source credibility algorithms on corpora, using PageRank as an example. In our experiments, we demonstrate quantitatively that our algorithm applies a lower authority score to malicious documents while promoting trusted content, using the standardized MS MARCO dataset. We also demonstrate proof-of-concept performance of our algorithm on CTI documents and feeds. 

---
# Practical Code RAG at Scale: Task-Aware Retrieval Design Choices under Compute Budgets 

**Authors**: Timur Galimzyanov, Olga Kolomyttseva, Egor Bogomolov  

**Link**: [PDF](https://arxiv.org/pdf/2510.20609)  

**Abstract**: We study retrieval design for code-focused generation tasks under realistic compute budgets. Using two complementary tasks from Long Code Arena -- code completion and bug localization -- we systematically compare retrieval configurations across various context window sizes along three axes: (i) chunking strategy, (ii) similarity scoring, and (iii) splitting granularity. (1) For PL-PL, sparse BM25 with word-level splitting is the most effective and practical, significantly outperforming dense alternatives while being an order of magnitude faster. (2) For NL-PL, proprietary dense encoders (Voyager-3 family) consistently beat sparse retrievers, however requiring 100x larger latency. (3) Optimal chunk size scales with available context: 32-64 line chunks work best at small budgets, and whole-file retrieval becomes competitive at 16000 tokens. (4) Simple line-based chunking matches syntax-aware splitting across budgets. (5) Retrieval latency varies by up to 200x across configurations; BPE-based splitting is needlessly slow, and BM25 + word splitting offers the best quality-latency trade-off. Thus, we provide evidence-based recommendations for implementing effective code-oriented RAG systems based on task requirements, model constraints, and computational efficiency. 

---
