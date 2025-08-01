# Biases in LLM-Generated Musical Taste Profiles for Recommendation 

**Authors**: Bruno Sguerra, Elena V. Epure, Harin Lee, Manuel Moussallam  

**Link**: [PDF](https://arxiv.org/pdf/2507.16708)  

**Abstract**: One particularly promising use case of Large Language Models (LLMs) for recommendation is the automatic generation of Natural Language (NL) user taste profiles from consumption data. These profiles offer interpretable and editable alternatives to opaque collaborative filtering representations, enabling greater transparency and user control. However, it remains unclear whether users consider these profiles to be an accurate representation of their taste, which is crucial for trust and usability. Moreover, because LLMs inherit societal and data-driven biases, profile quality may systematically vary across user and item characteristics. In this paper, we study this issue in the context of music streaming, where personalization is challenged by a large and culturally diverse catalog. We conduct a user study in which participants rate NL profiles generated from their own listening histories. We analyze whether identification with the profiles is biased by user attributes (e.g., mainstreamness, taste diversity) and item features (e.g., genre, country of origin). We also compare these patterns to those observed when using the profiles in a downstream recommendation task. Our findings highlight both the potential and limitations of scrutable, LLM-based profiling in personalized systems. 

---
# Generating Search Explanations using Large Language Models 

**Authors**: Arif Laksito, Mark Stevenson  

**Link**: [PDF](https://arxiv.org/pdf/2507.16692)  

**Abstract**: Aspect-oriented explanations in search results are typically concise text snippets placed alongside retrieved documents to serve as explanations that assist users in efficiently locating relevant information. While Large Language Models (LLMs) have demonstrated exceptional performance for a range of problems, their potential to generate explanations for search results has not been explored. This study addresses that gap by leveraging both encoder-decoder and decoder-only LLMs to generate explanations for search results. The explanations generated are consistently more accurate and plausible explanations than those produced by a range of baseline models. 

---
# Enhancing patent retrieval using automated patent summarization 

**Authors**: Eleni Kamateri, Renukswamy Chikkamath, Michail Salampasis, Linda Andersson, Markus Endres  

**Link**: [PDF](https://arxiv.org/pdf/2507.16371)  

**Abstract**: Effective query formulation is a key challenge in long-document Information Retrieval (IR). This challenge is particularly acute in domain-specific contexts like patent retrieval, where documents are lengthy, linguistically complex, and encompass multiple interrelated technical topics. In this work, we present the application of recent extractive and abstractive summarization methods for generating concise, purpose-specific summaries of patent documents. We further assess the utility of these automatically generated summaries as surrogate queries across three benchmark patent datasets and compare their retrieval performance against conventional approaches that use entire patent sections. Experimental results show that summarization-based queries significantly improve prior-art retrieval effectiveness, highlighting their potential as an efficient alternative to traditional query formulation techniques. 

---
# Time to Split: Exploring Data Splitting Strategies for Offline Evaluation of Sequential Recommenders 

**Authors**: Danil Gusak, Anna Volodkevich, Anton Klenitskiy, Alexey Vasilev, Evgeny Frolov  

**Link**: [PDF](https://arxiv.org/pdf/2507.16289)  

**Abstract**: Modern sequential recommender systems, ranging from lightweight transformer-based variants to large language models, have become increasingly prominent in academia and industry due to their strong performance in the next-item prediction task. Yet common evaluation protocols for sequential recommendations remain insufficiently developed: they often fail to reflect the corresponding recommendation task accurately, or are not aligned with real-world scenarios.
Although the widely used leave-one-out split matches next-item prediction, it permits the overlap between training and test periods, which leads to temporal leakage and unrealistically long test horizon, limiting real-world relevance. Global temporal splitting addresses these issues by evaluating on distinct future periods. However, its applications to sequential recommendations remain loosely defined, particularly in terms of selecting target interactions and constructing a validation subset that provides necessary consistency between validation and test metrics.
In this paper, we demonstrate that evaluation outcomes can vary significantly across splitting strategies, influencing model rankings and practical deployment decisions. To improve reproducibility in both academic and industrial settings, we systematically compare different splitting strategies for sequential recommendations across multiple datasets and established baselines. Our findings show that prevalent splits, such as leave-one-out, may be insufficiently aligned with more realistic evaluation strategies. Code: this https URL 

---
# Reinforce Lifelong Interaction Value of User-Author Pairs for Large-Scale Recommendation Systems 

**Authors**: Yisha Li, Lexi Gao, Jingxin Liu, Xiang Gao, Xin Li, Haiyang Lu, Liyin Hong  

**Link**: [PDF](https://arxiv.org/pdf/2507.16253)  

**Abstract**: Recommendation systems (RS) help users find interested content and connect authors with their target audience. Most research in RS tends to focus either on predicting users' immediate feedback (like click-through rate) accurately or improving users' long-term engagement. However, they ignore the influence for authors and the lifelong interaction value (LIV) of user-author pairs, which is particularly crucial for improving the prosperity of social community in short-video platforms. Currently, reinforcement learning (RL) can optimize long-term benefits and has been widely applied in RS. In this paper, we introduce RL to Reinforce Lifelong Interaction Value of User-Author pairs (RLIV-UA) based on each interaction of UA pairs. To address the long intervals between UA interactions and the large scale of the UA space, we propose a novel Sparse Cross-Request Interaction Markov Decision Process (SCRI-MDP) and introduce an Adjacent State Approximation (ASA) method to construct RL training samples. Additionally, we introduce Multi-Task Critic Learning (MTCL) to capture the progressive nature of UA interactions (click -> follow -> gift), where denser interaction signals are leveraged to compensate for the learning of sparse labels. Finally, an auxiliary supervised learning task is designed to enhance the convergence of the RLIV-UA model. In offline experiments and online A/B tests, the RLIV-UA model achieves both higher user satisfaction and higher platform profits than compared methods. 

---
# LLM-Enhanced Reranking for Complementary Product Recommendation 

**Authors**: Zekun Xu, Yudi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.16237)  

**Abstract**: Complementary product recommendation, which aims to suggest items that are used together to enhance customer value, is a crucial yet challenging task in e-commerce. While existing graph neural network (GNN) approaches have made significant progress in capturing complex product relationships, they often struggle with the accuracy-diversity tradeoff, particularly for long-tail items. This paper introduces a model-agnostic approach that leverages Large Language Models (LLMs) to enhance the reranking of complementary product recommendations. Unlike previous works that use LLMs primarily for data preprocessing and graph augmentation, our method applies LLM-based prompting strategies directly to rerank candidate items retrieved from existing recommendation models, eliminating the need for model retraining. Through extensive experiments on public datasets, we demonstrate that our approach effectively balances accuracy and diversity in complementary product recommendations, with at least 50% lift in accuracy metrics and 2% lift in diversity metrics on average for the top recommended items across datasets. 

---
# Scaling Recommender Transformers to One Billion Parameters 

**Authors**: Kirill Khrylchenko, Artem Matveev, Sergei Makeev, Vladimir Baikalov  

**Link**: [PDF](https://arxiv.org/pdf/2507.15994)  

**Abstract**: While large transformer models have been successfully used in many real-world applications such as natural language processing, computer vision, and speech processing, scaling transformers for recommender systems remains a challenging problem. Recently, Generative Recommenders framework was proposed to scale beyond typical Deep Learning Recommendation Models (DLRMs). Reformulation of recommendation as sequential transduction task led to improvement of scaling properties in terms of compute. Nevertheless, the largest encoder configuration reported by the HSTU authors amounts only to ~176 million parameters, which is considerably smaller than the hundreds of billions or even trillions of parameters common in modern language models.
In this work, we present a recipe for training large transformer recommenders with up to a billion parameters. We show that autoregressive learning on user histories naturally decomposes into two subtasks, feedback prediction and next-item prediction, and demonstrate that such a decomposition scales effectively across a wide range of transformer sizes. Furthermore, we report a successful deployment of our proposed architecture on a large-scale music platform serving millions of users. According to our online A/B tests, this new model increases total listening time by +2.26% and raises the likelihood of user likes by +6.37%, constituting (to our knowledge) the largest improvement in recommendation quality reported for any deep learning-based system in the platform's history. 

---
# RAVine: Reality-Aligned Evaluation for Agentic Search 

**Authors**: Yilong Xu, Xiang Long, Zhi Zheng, Jinhua Gao  

**Link**: [PDF](https://arxiv.org/pdf/2507.16725)  

**Abstract**: Agentic search, as a more autonomous and adaptive paradigm of retrieval augmentation, is driving the evolution of intelligent search systems. However, existing evaluation frameworks fail to align well with the goals of agentic search. First, the complex queries commonly used in current benchmarks often deviate from realistic user search scenarios. Second, prior approaches tend to introduce noise when extracting ground truth for end-to-end evaluations, leading to distorted assessments at a fine-grained level. Third, most current frameworks focus solely on the quality of final answers, neglecting the evaluation of the iterative process inherent to agentic search. To address these limitations, we propose RAVine -- a Reality-Aligned eValuation framework for agentic LLMs with search. RAVine targets multi-point queries and long-form answers that better reflect user intents, and introduces an attributable ground truth construction strategy to enhance the accuracy of fine-grained evaluation. Moreover, RAVine examines model's interaction with search tools throughout the iterative process, and accounts for factors of efficiency. We benchmark a series of models using RAVine and derive several insights, which we hope will contribute to advancing the development of agentic search systems. The code and datasets are available at this https URL. 

---
# Agentic RAG with Knowledge Graphs for Complex Multi-Hop Reasoning in Real-World Applications 

**Authors**: Jean Lelong, Adnane Errazine, Annabelle Blangero  

**Link**: [PDF](https://arxiv.org/pdf/2507.16507)  

**Abstract**: Conventional Retrieval-Augmented Generation (RAG) systems enhance Large Language Models (LLMs) but often fall short on complex queries, delivering limited, extractive answers and struggling with multiple targeted retrievals or navigating intricate entity relationships. This is a critical gap in knowledge-intensive domains. We introduce INRAExplorer, an agentic RAG system for exploring the scientific data of INRAE (France's National Research Institute for Agriculture, Food and Environment). INRAExplorer employs an LLM-based agent with a multi-tool architecture to dynamically engage a rich knowledge base, through a comprehensive knowledge graph derived from open access INRAE publications. This design empowers INRAExplorer to conduct iterative, targeted queries, retrieve exhaustive datasets (e.g., all publications by an author), perform multi-hop reasoning, and deliver structured, comprehensive answers. INRAExplorer serves as a concrete illustration of enhancing knowledge interaction in specialized fields. 

---
# EBaReT: Expert-guided Bag Reward Transformer for Auto Bidding 

**Authors**: Kaiyuan Li, Pengyu Wang, Yunshan Peng, Pengjia Yuan, Yanxiang Zeng, Rui Xiang, Yanhua Cheng, Xialong Liu, Peng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2507.16186)  

**Abstract**: Reinforcement learning has been widely applied in automated bidding. Traditional approaches model bidding as a Markov Decision Process (MDP). Recently, some studies have explored using generative reinforcement learning methods to address long-term dependency issues in bidding environments. Although effective, these methods typically rely on supervised learning approaches, which are vulnerable to low data quality due to the amount of sub-optimal bids and low probability rewards resulting from the low click and conversion rates. Unfortunately, few studies have addressed these challenges.
In this paper, we formalize the automated bidding as a sequence decision-making problem and propose a novel Expert-guided Bag Reward Transformer (EBaReT) to address concerns related to data quality and uncertainty rewards. Specifically, to tackle data quality issues, we generate a set of expert trajectories to serve as supplementary data in the training process and employ a Positive-Unlabeled (PU) learning-based discriminator to identify expert transitions. To ensure the decision also meets the expert level, we further design a novel expert-guided inference strategy. Moreover, to mitigate the uncertainty of rewards, we consider the transitions within a certain period as a "bag" and carefully design a reward function that leads to a smoother acquisition of rewards. Extensive experiments demonstrate that our model achieves superior performance compared to state-of-the-art bidding methods. 

---
