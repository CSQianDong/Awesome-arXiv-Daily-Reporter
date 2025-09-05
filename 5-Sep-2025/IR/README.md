# Global-to-Local or Local-to-Global? Enhancing Image Retrieval with Efficient Local Search and Effective Global Re-ranking 

**Authors**: Dror Aiger, Bingyi Cao, Kaifeng Chen, Andre Araujo  

**Link**: [PDF](https://arxiv.org/pdf/2509.04351)  

**Abstract**: The dominant paradigm in image retrieval systems today is to search large databases using global image features, and re-rank those initial results with local image feature matching techniques. This design, dubbed global-to-local, stems from the computational cost of local matching approaches, which can only be afforded for a small number of retrieved images. However, emerging efficient local feature search approaches have opened up new possibilities, in particular enabling detailed retrieval at large scale, to find partial matches which are often missed by global feature search. In parallel, global feature-based re-ranking has shown promising results with high computational efficiency. In this work, we leverage these building blocks to introduce a local-to-global retrieval paradigm, where efficient local feature search meets effective global feature re-ranking. Critically, we propose a re-ranking method where global features are computed on-the-fly, based on the local feature retrieval similarities. Such re-ranking-only global features leverage multidimensional scaling techniques to create embeddings which respect the local similarities obtained during search, enabling a significant re-ranking boost. Experimentally, we demonstrate solid retrieval performance, setting new state-of-the-art results on the Revisited Oxford and Paris datasets. 

---
# Decoupled Entity Representation Learning for Pinterest Ads Ranking 

**Authors**: Jie Liu, Yinrui Li, Jiankai Sun, Kungang Li, Han Sun, Sihan Wang, Huasen Wu, Siyuan Gao, Paulo Soares, Nan Li, Zhifang Liu, Haoyang Li, Siping Ji, Ling Leng, Prathibha Deshikachar  

**Link**: [PDF](https://arxiv.org/pdf/2509.04337)  

**Abstract**: In this paper, we introduce a novel framework following an upstream-downstream paradigm to construct user and item (Pin) embeddings from diverse data sources, which are essential for Pinterest to deliver personalized Pins and ads effectively. Our upstream models are trained on extensive data sources featuring varied signals, utilizing complex architectures to capture intricate relationships between users and Pins on Pinterest. To ensure scalability of the upstream models, entity embeddings are learned, and regularly refreshed, rather than real-time computation, allowing for asynchronous interaction between the upstream and downstream models. These embeddings are then integrated as input features in numerous downstream tasks, including ad retrieval and ranking models for CTR and CVR predictions. We demonstrate that our framework achieves notable performance improvements in both offline and online settings across various downstream tasks. This framework has been deployed in Pinterest's production ad ranking systems, resulting in significant gains in online metrics. 

---
# Temporal Interest-Driven Multimodal Personalized Content Generation 

**Authors**: Tian Miao  

**Link**: [PDF](https://arxiv.org/pdf/2509.04330)  

**Abstract**: With the dynamic evolution of user interests and the increasing multimodal demands in internet applications, personalized content generation strategies based on static interest preferences struggle to meet practical application requirements. The proposed TIMGen (Temporal Interest-driven Multimodal Generation) model addresses this challenge by modeling the long-term temporal evolution of users' interests and capturing dynamic interest representations with strong temporal dependencies. This model also supports the fusion of multimodal features, such as text, images, video, and audio, and delivers customized content based on multimodal preferences. TIMGen jointly learns temporal dependencies and modal preferences to obtain a unified interest representation, which it then generates to meet users' personalized content needs. TIMGen overcomes the shortcomings of personalized content recommendation methods based on static preferences, enabling flexible and dynamic modeling of users' multimodal interests, better understanding and capturing their interests and preferences. It can be extended to a variety of practical application scenarios, including e-commerce, advertising, online education, and precision medicine, providing insights for future research. 

---
# Enhancing Technical Documents Retrieval for RAG 

**Authors**: Songjiang Lai, Tsun-Hin Cheung, Ka-Chun Fung, Kaiwen Xue, Kwan-Ho Lin, Yan-Ming Choi, Vincent Ng, Kin-Man Lam  

**Link**: [PDF](https://arxiv.org/pdf/2509.04139)  

**Abstract**: In this paper, we introduce Technical-Embeddings, a novel framework designed to optimize semantic retrieval in technical documentation, with applications in both hardware and software development. Our approach addresses the challenges of understanding and retrieving complex technical content by leveraging the capabilities of Large Language Models (LLMs). First, we enhance user queries by generating expanded representations that better capture user intent and improve dataset diversity, thereby enriching the fine-tuning process for embedding models. Second, we apply summary extraction techniques to encode essential contextual information, refining the representation of technical documents. To further enhance retrieval performance, we fine-tune a bi-encoder BERT model using soft prompting, incorporating separate learning parameters for queries and document context to capture fine-grained semantic nuances. We evaluate our approach on two public datasets, RAG-EDA and Rust-Docs-QA, demonstrating that Technical-Embeddings significantly outperforms baseline models in both precision and recall. Our findings highlight the effectiveness of integrating query expansion and contextual summarization to enhance information access and comprehension in technical domains. This work advances the state of Retrieval-Augmented Generation (RAG) systems, offering new avenues for efficient and accurate technical document retrieval in engineering and product development workflows. 

---
# Safeguarding Patient Trust in the Age of AI: Tackling Health Misinformation with Explainable AI 

**Authors**: Sueun Hong, Shuojie Fu, Ovidiu Serban, Brianna Bao, James Kinross, Francesa Toni, Guy Martin, Uddhav Vaghela  

**Link**: [PDF](https://arxiv.org/pdf/2509.04052)  

**Abstract**: AI-generated health misinformation poses unprecedented threats to patient safety and healthcare system trust globally. This white paper presents an explainable AI framework developed through the EPSRC INDICATE project to combat medical misinformation while enhancing evidence-based healthcare delivery. Our systematic review of 17 studies reveals the urgent need for transparent AI systems in healthcare. The proposed solution demonstrates 95% recall in clinical evidence retrieval and integrates novel trustworthiness classifiers achieving 76% F1 score in detecting biomedical misinformation. Results show that explainable AI can transform traditional 6-month expert review processes into real-time, automated evidence synthesis while maintaining clinical rigor. This approach offers a critical intervention to preserve healthcare integrity in the AI era. 

---
# NER Retriever: Zero-Shot Named Entity Retrieval with Type-Aware Embeddings 

**Authors**: Or Shachar, Uri Katz, Yoav Goldberg, Oren Glickman  

**Link**: [PDF](https://arxiv.org/pdf/2509.04011)  

**Abstract**: We present NER Retriever, a zero-shot retrieval framework for ad-hoc Named Entity Retrieval, a variant of Named Entity Recognition (NER), where the types of interest are not provided in advance, and a user-defined type description is used to retrieve documents mentioning entities of that type. Instead of relying on fixed schemas or fine-tuned models, our method builds on internal representations of large language models (LLMs) to embed both entity mentions and user-provided open-ended type descriptions into a shared semantic space. We show that internal representations, specifically the value vectors from mid-layer transformer blocks, encode fine-grained type information more effectively than commonly used top-layer embeddings. To refine these representations, we train a lightweight contrastive projection network that aligns type-compatible entities while separating unrelated types. The resulting entity embeddings are compact, type-aware, and well-suited for nearest-neighbor search. Evaluated on three benchmarks, NER Retriever significantly outperforms both lexical and dense sentence-level retrieval baselines. Our findings provide empirical support for representation selection within LLMs and demonstrate a practical solution for scalable, schema-free entity retrieval. The NER Retriever Codebase is publicly available at this https URL 

---
# Evaluating the Robustness of Retrieval-Augmented Generation to Adversarial Evidence in the Health Domain 

**Authors**: Shakiba Amirshahi, Amin Bigdeli, Charles L. A. Clarke, Amira Ghenai  

**Link**: [PDF](https://arxiv.org/pdf/2509.03787)  

**Abstract**: Retrieval augmented generation (RAG) systems provide a method for factually grounding the responses of a Large Language Model (LLM) by providing retrieved evidence, or context, as support. Guided by this context, RAG systems can reduce hallucinations and expand the ability of LLMs to accurately answer questions outside the scope of their training data. Unfortunately, this design introduces a critical vulnerability: LLMs may absorb and reproduce misinformation present in retrieved evidence. This problem is magnified if retrieved evidence contains adversarial material explicitly intended to promulgate misinformation. This paper presents a systematic evaluation of RAG robustness in the health domain and examines alignment between model outputs and ground-truth answers. We focus on the health domain due to the potential for harm caused by incorrect responses, as well as the availability of evidence-based ground truth for many common health-related questions. We conduct controlled experiments using common health questions, varying both the type and composition of the retrieved documents (helpful, harmful, and adversarial) as well as the framing of the question by the user (consistent, neutral, and inconsistent). Our findings reveal that adversarial documents substantially degrade alignment, but robustness can be preserved when helpful evidence is also present in the retrieval pool. These findings offer actionable insights for designing safer RAG systems in high-stakes domains by highlighting the need for retrieval safeguards. To enable reproducibility and facilitate future research, all experimental results are publicly available in our github repository.
this https URL 

---
# LLM-based Relevance Assessment for Web-Scale Search Evaluation at Pinterest 

**Authors**: Han Wang, Alex Whitworth, Pak Ming Cheung, Zhenjie Zhang, Krishna Kamath  

**Link**: [PDF](https://arxiv.org/pdf/2509.03764)  

**Abstract**: Relevance evaluation plays a crucial role in personalized search systems to ensure that search results align with a user's queries and intent. While human annotation is the traditional method for relevance evaluation, its high cost and long turnaround time limit its scalability. In this work, we present our approach at Pinterest Search to automate relevance evaluation for online experiments using fine-tuned LLMs. We rigorously validate the alignment between LLM-generated judgments and human annotations, demonstrating that LLMs can provide reliable relevance measurement for experiments while greatly improving the evaluation efficiency. Leveraging LLM-based labeling further unlocks the opportunities to expand the query set, optimize sampling design, and efficiently assess a wider range of search experiences at scale. This approach leads to higher-quality relevance metrics and significantly reduces the Minimum Detectable Effect (MDE) in online experiment measurements. 

---
# Efficient Item ID Generation for Large-Scale LLM-based Recommendation 

**Authors**: Anushya Subbiah, Vikram Aggarwal, James Pine, Steffen Rendle, Krishna Sayana, Kun Su  

**Link**: [PDF](https://arxiv.org/pdf/2509.03746)  

**Abstract**: Integrating product catalogs and user behavior into LLMs can enhance recommendations with broad world knowledge, but the scale of real-world item catalogs, often containing millions of discrete item identifiers (Item IDs), poses a significant challenge. This contrasts with the smaller, tokenized text vocabularies typically used in LLMs. The predominant view within the LLM-based recommendation literature is that it is infeasible to treat item ids as a first class citizen in the LLM and instead some sort of tokenization of an item into multiple tokens is required. However, this creates a key practical bottleneck in serving these models for real-time low-latency applications.
Our paper challenges this predominant practice and integrates item ids as first class citizens into the LLM. We provide simple, yet highly effective, novel training and inference modifications that enable single-token representations of items and single-step decoding. Our method shows improvements in recommendation quality (Recall and NDCG) over existing techniques on the Amazon shopping datasets while significantly improving inference efficiency by 5x-14x. Our work offers an efficiency perspective distinct from that of other popular approaches within LLM-based recommendation, potentially inspiring further research and opening up a new direction for integrating IDs into LLMs. Our code is available here this https URL 

---
# LLMs for estimating positional bias in logged interaction data 

**Authors**: Aleksandr V. Petrov, Michael Murtagh, Karthik Nagesh  

**Link**: [PDF](https://arxiv.org/pdf/2509.03696)  

**Abstract**: Recommender and search systems commonly rely on Learning To Rank models trained on logged user interactions to order items by predicted relevance. However, such interaction data is often subject to position bias, as users are more likely to click on items that appear higher in the ranking, regardless of their actual relevance. As a result, newly trained models may inherit and reinforce the biases of prior ranking models rather than genuinely improving relevance. A standard approach to mitigate position bias is Inverse Propensity Scoring (IPS), where the model's loss is weighted by the inverse of a propensity function, an estimate of the probability that an item at a given position is examined. However, accurate propensity estimation is challenging, especially in interfaces with complex non-linear layouts. In this paper, we propose a novel method for estimating position bias using Large Language Models (LLMs) applied to logged user interaction data. This approach offers a cost-effective alternative to online experimentation. Our experiments show that propensities estimated with our LLM-as-a-judge approach are stable across score buckets and reveal the row-column effects of Viator's grid layout that simpler heuristics overlook. An IPS-weighted reranker trained with these propensities matches the production model on standard NDCG@10 while improving weighted NDCG@10 by roughly 2%. We will verify these offline gains in forthcoming live-traffic experiments. 

---
# lifeXplore at the Lifelog Search Challenge 2021 

**Authors**: Andreas Leibetseder, Klaus Schoeffmann  

**Link**: [PDF](https://arxiv.org/pdf/2509.03692)  

**Abstract**: Since its first iteration in 2018, the Lifelog Search Challenge (LSC) continues to rise in popularity as an interactive lifelog data retrieval competition, co-located at the ACM International Conference on Multimedia Retrieval (ICMR). The goal of this annual live event is to search a large corpus of lifelogging data for specifically announced memories using a purposefully developed tool within a limited amount of time. As long-standing participants, we present our improved lifeXplore - a retrieval system combining chronologic day summary browsing with interactive combinable concept filtering. Compared to previous versions, the tool is improved by incorporating temporal queries, advanced day summary features as well as usability improvements. 

---
# ACT: Automated Constraint Targeting for Multi-Objective Recommender Systems 

**Authors**: Daryl Chang, Yi Wu, Jennifer She, Li Wei, Lukasz Heldt  

**Link**: [PDF](https://arxiv.org/pdf/2509.03661)  

**Abstract**: Recommender systems often must maximize a primary objective while ensuring secondary ones satisfy minimum thresholds, or "guardrails." This is critical for maintaining a consistent user experience and platform ecosystem, but enforcing these guardrails despite orthogonal system changes is challenging and often requires manual hyperparameter tuning. We introduce the Automated Constraint Targeting (ACT) framework, which automatically finds the minimal set of hyperparameter changes needed to satisfy these guardrails. ACT uses an offline pairwise evaluation on unbiased data to find solutions and continuously retrains to adapt to system and user behavior changes. We empirically demonstrate its efficacy and describe its deployment in a large-scale production environment. 

---
# Delta Activations: A Representation for Finetuned Large Language Models 

**Authors**: Zhiqiu Xu, Amish Sethi, Mayur Naik, Ser-Nam Lim  

**Link**: [PDF](https://arxiv.org/pdf/2509.04442)  

**Abstract**: The success of powerful open source Large Language Models (LLMs) has enabled the community to create a vast collection of post-trained models adapted to specific tasks and domains. However, navigating and understanding these models remains challenging due to inconsistent metadata and unstructured repositories. We introduce Delta Activations, a method to represent finetuned models as vector embeddings by measuring shifts in their internal activations relative to a base model. This representation allows for effective clustering by domain and task, revealing structure in the model landscape. Delta Activations also demonstrate desirable properties: it is robust across finetuning settings and exhibits an additive property when finetuning datasets are mixed. In addition, we show that Delta Activations can embed tasks via few-shot finetuning, and further explore its use for model selection and merging. We hope Delta Activations can facilitate the practice of reusing publicly available models. Code is available at this https URL. 

---
# How many patients could we save with LLM priors? 

**Authors**: Shota Arai, David Selby, Andrew Vargo, Sebastian Vollmer  

**Link**: [PDF](https://arxiv.org/pdf/2509.04250)  

**Abstract**: Imagine a world where clinical trials need far fewer patients to achieve the same statistical power, thanks to the knowledge encoded in large language models (LLMs). We present a novel framework for hierarchical Bayesian modeling of adverse events in multi-center clinical trials, leveraging LLM-informed prior distributions. Unlike data augmentation approaches that generate synthetic data points, our methodology directly obtains parametric priors from the model. Our approach systematically elicits informative priors for hyperparameters in hierarchical Bayesian models using a pre-trained LLM, enabling the incorporation of external clinical expertise directly into Bayesian safety modeling. Through comprehensive temperature sensitivity analysis and rigorous cross-validation on real-world clinical trial data, we demonstrate that LLM-derived priors consistently improve predictive performance compared to traditional meta-analytical approaches. This methodology paves the way for more efficient and expert-informed clinical trial design, enabling substantial reductions in the number of patients required to achieve robust safety assessment and with the potential to transform drug safety monitoring and regulatory decision making. 

---
# PianoBind: A Multimodal Joint Embedding Model for Pop-piano Music 

**Authors**: Hayeon Bang, Eunjin Choi, Seungheon Doh, Juhan Nam  

**Link**: [PDF](https://arxiv.org/pdf/2509.04215)  

**Abstract**: Solo piano music, despite being a single-instrument medium, possesses significant expressive capabilities, conveying rich semantic information across genres, moods, and styles. However, current general-purpose music representation models, predominantly trained on large-scale datasets, often struggle to captures subtle semantic distinctions within homogeneous solo piano music. Furthermore, existing piano-specific representation models are typically unimodal, failing to capture the inherently multimodal nature of piano music, expressed through audio, symbolic, and textual modalities. To address these limitations, we propose PianoBind, a piano-specific multimodal joint embedding model. We systematically investigate strategies for multi-source training and modality utilization within a joint embedding framework optimized for capturing fine-grained semantic distinctions in (1) small-scale and (2) homogeneous piano datasets. Our experimental results demonstrate that PianoBind learns multimodal representations that effectively capture subtle nuances of piano music, achieving superior text-to-music retrieval performance on in-domain and out-of-domain piano datasets compared to general-purpose music joint embedding models. Moreover, our design choices offer reusable insights for multimodal representation learning with homogeneous datasets beyond piano music. 

---
