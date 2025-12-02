# Structured Spectral Reasoning for Frequency-Adaptive Multimodal Recommendation 

**Authors**: Wei Yang, Rui Zhong, Yiqun Chen, Chi Lu, Peng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2512.01372)  

**Abstract**: Multimodal recommendation aims to integrate collaborative signals with heterogeneous content such as visual and textual information, but remains challenged by modality-specific noise, semantic inconsistency, and unstable propagation over user-item graphs. These issues are often exacerbated by naive fusion or shallow modeling strategies, leading to degraded generalization and poor robustness. While recent work has explored the frequency domain as a lens to separate stable from noisy signals, most methods rely on static filtering or reweighting, lacking the ability to reason over spectral structure or adapt to modality-specific reliability. To address these challenges, we propose a Structured Spectral Reasoning (SSR) framework for frequency-aware multimodal recommendation. Our method follows a four-stage pipeline: (i) Decompose graph-based multimodal signals into spectral bands via graph-guided transformations to isolate semantic granularity; (ii) Modulate band-level reliability with spectral band masking, a training-time masking with a prediction-consistency objective that suppresses brittle frequency components; (iii) Fuse complementary frequency cues using hyperspectral reasoning with low-rank cross-band interaction; and (iv) Align modality-specific spectral features via contrastive regularization to promote semantic and structural consistency. Experiments on three real-world benchmarks show consistent gains over strong baselines, particularly under sparse and cold-start settings. Additional analyses indicate that structured spectral modeling improves robustness and provides clearer diagnostics of how different bands contribute to performance. 

---
# Toward a benchmark for CTR prediction in online advertising: datasets, evaluation protocols and perspectives 

**Authors**: Shan Gao, Yanwu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2512.01179)  

**Abstract**: This research designs a unified architecture of CTR prediction benchmark (Bench-CTR) platform that offers flexible interfaces with datasets and components of a wide range of CTR prediction models. Moreover, we construct a comprehensive system of evaluation protocols encompassing real-world and synthetic datasets, a taxonomy of metrics, standardized procedures and experimental guidelines for calibrating the performance of CTR prediction models. Furthermore, we implement the proposed benchmark platform and conduct a comparative study to evaluate a wide range of state-of-the-art models from traditional multivariate statistical to modern large language model (LLM)-based approaches on three public datasets and two synthetic datasets. Experimental results reveal that, (1) high-order models largely outperform low-order models, though such advantage varies in terms of metrics and on different datasets; (2) LLM-based models demonstrate a remarkable data efficiency, i.e., achieving the comparable performance to other models while using only 2% of the training data; (3) the performance of CTR prediction models has achieved significant improvements from 2015 to 2016, then reached a stage with slow progress, which is consistent across various datasets. This benchmark is expected to facilitate model development and evaluation and enhance practitioners' understanding of the underlying mechanisms of models in the area of CTR prediction. Code is available at this https URL. 

---
# Conversion rate prediction in online advertising: modeling techniques, performance evaluation and future directions 

**Authors**: Tao Xue, Yanwu Yang, Panyu Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2512.01171)  

**Abstract**: Conversion and conversion rate (CVR) prediction play a critical role in efficient advertising decision-making. In past decades, although researchers have developed plenty of models for CVR prediction, the methodological evolution and relationships between different techniques have been precluded. In this paper, we conduct a comprehensive literature review on CVR prediction in online advertising, and classify state-of-the-art CVR prediction models into six categories with respect to the underlying techniques and elaborate on connections between these techniques. For each category of models, we present the framework of underlying techniques, their advantages and disadvantages, and discuss how they are utilized for CVR prediction. Moreover, we summarize the performance of various CVR prediction models on public and proprietary datasets. Finally, we identify research trends, major challenges, and promising future directions. We observe that results of performance evaluation reported in prior studies are not unanimous; semantics-enriched, attribution-enhanced, debiased CVR prediction and jointly modeling CTR and CVR prediction would be promising directions to explore in the future. This review is expected to provide valuable references and insights for future researchers and practitioners in this area. 

---
# Optimizing Generative Ranking Relevance via Reinforcement Learning in Xiaohongshu Search 

**Authors**: Ziyang Zeng, Heming Jing, Jindong Chen, Xiangli Li, Hongyu Liu, Yixuan He, Zhengyu Li, Yige Sun, Zheyong Xie, Yuqing Yang, Shaosheng Cao, Jun Fan, Yi Wu, Yao Hu  

**Link**: [PDF](https://arxiv.org/pdf/2512.00968)  

**Abstract**: Ranking relevance is a fundamental task in search engines, aiming to identify the items most relevant to a given user query. Traditional relevance models typically produce scalar scores or directly predict relevance labels, limiting both interpretability and the modeling of complex relevance signals. Inspired by recent advances in Chain-of-Thought (CoT) reasoning for complex tasks, we investigate whether explicit reasoning can enhance both interpretability and performance in relevance modeling. However, existing reasoning-based Generative Relevance Models (GRMs) primarily rely on supervised fine-tuning on large amounts of human-annotated or synthetic CoT data, which often leads to limited generalization. Moreover, domain-agnostic, free-form reasoning tends to be overly generic and insufficiently grounded, limiting its potential to handle the diverse and ambiguous cases prevalent in open-domain search. In this work, we formulate relevance modeling in Xiaohongshu search as a reasoning task and introduce a Reinforcement Learning (RL)-based training framework to enhance the grounded reasoning capabilities of GRMs. Specifically, we incorporate practical business-specific relevance criteria into the multi-step reasoning prompt design and propose Stepwise Advantage Masking (SAM), a lightweight process-supervision strategy which facilitates effective learning of these criteria through improved credit assignment. To enable industrial deployment, we further distill the large-scale RL-tuned model to a lightweight version suitable for real-world search systems. Extensive experiments on industrial datasets, along with online A/B tests, demonstrate the effectiveness of our approach. 

---
# SHRAG: AFrameworkfor Combining Human-Inspired Search with RAG 

**Authors**: Hyunseok Ryu, Wonjune Shin, Hyun Park  

**Link**: [PDF](https://arxiv.org/pdf/2512.00772)  

**Abstract**: Retrieval-Augmented Generation (RAG) is gaining recognition as one of the key technological axes for next generation information retrieval, owing to its ability to mitigate the hallucination phenomenon in Large Language
Models (LLMs)and effectively incorporate up-to-date information. However, specialized expertise is necessary to
construct ahigh-quality retrieval system independently; moreover, RAGdemonstratesrelativelyslowerprocessing
speeds compared to conventional pure retrieval systems because it involves both retrieval and generation stages.
Accordingly, this study proposes SHRAG, a novel framework designed to facilitate the seamless integration of
Information Retrieval and RAG while simultaneously securing precise retrieval performance. SHRAG utilizes a
Large Language Model as a Query Strategist to automatically transform unstructured natural language queries
into logically structured search queries, subsequently performing Boolean retrieval to emulate the search process
of an expert human searcher. Furthermore, it incorporates multilingual query expansion and a multilingual
embedding model, enabling it to perform efficient cross-lingual question answering within the multilingual
dataset environment of the ScienceON Challenge. Experimental results demonstrate that the proposed method,
combining logical retrieval capabilities and generative reasoning, can significantly enhance the accuracy and
reliability of RAG systems. Furthermore, SHRAG movesbeyondconventionaldocument-centric retrieval methods,
presenting the potential for a new search paradigm capable of providing direct and reliable responses to queries. 

---
# ProEx: A Unified Framework Leveraging Large Language Model with Profile Extrapolation for Recommendation 

**Authors**: Yi Zhang, Yiwen Zhang, Yu Wang, Tong Chen, Hongzhi Yin  

**Link**: [PDF](https://arxiv.org/pdf/2512.00679)  

**Abstract**: The powerful text understanding and generation capabilities of large language models (LLMs) have brought new vitality to general recommendation with implicit feedback. One possible strategy involves generating a unique user (or item) profile from historical interaction data, which is then mapped to a semantic representation in the language space. However, a single-instance profile may be insufficient to comprehensively capture the complex intentions behind a user's interacted items. Moreover, due to the inherent instability of LLMs, a biased or misinterpreted profile could even undermine the original recommendation performance. Consequently, an intuitive solution is to generate multiple profiles for each user (or item), each reflecting a distinct aspect of their characteristics. In light of this, we propose a unified recommendation framework with multi-faceted profile extrapolation (ProEx) in this paper. By leveraging chain-of-thought reasoning, we construct multiple distinct profiles for each user and item. These new profiles are subsequently mapped into semantic vectors, extrapolating from the position of the original profile to explore a broader region of the language space. Subsequently, we introduce the concept of environments, where each environment represents a possible linear combination of all profiles. The differences across environments are minimized to reveal the inherent invariance of user preferences. We apply ProEx to three discriminative methods and three generative methods, and conduct extensive experiments on three datasets. The experimental results demonstrate that ProEx significantly enhances the performance of these base recommendation models. 

---
# DLRREC: Denoising Latent Representations via Multi-Modal Knowledge Fusion in Deep Recommender Systems 

**Authors**: Jiahao Tian, Zhenkai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.00596)  

**Abstract**: Modern recommender systems struggle to effectively utilize the rich, yet high-dimensional and noisy, multi-modal features generated by Large Language Models (LLMs). Treating these features as static inputs decouples them from the core recommendation task. We address this limitation with a novel framework built on a key insight: deeply fusing multi-modal and collaborative knowledge for representation denoising. Our unified architecture introduces two primary technical innovations. First, we integrate dimensionality reduction directly into the recommendation model, enabling end-to-end co-training that makes the reduction process aware of the final ranking objective. Second, we introduce a contrastive learning objective that explicitly incorporates the collaborative filtering signal into the latent space. This synergistic process refines raw LLM embeddings, filtering noise while amplifying task-relevant signals. Extensive experiments confirm our method's superior discriminative power, proving that this integrated fusion and denoising strategy is critical for achieving state-of-the-art performance. Our work provides a foundational paradigm for effectively harnessing LLMs in recommender systems. 

---
# PEOAT: Personalization-Guided Evolutionary Question Assembly for One-Shot Adaptive Testing 

**Authors**: Xiaoshan Yu, Ziwei Huang, Shangshang Yang, Ziwen Wang, Haiping Ma, Xingyi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.00439)  

**Abstract**: With the rapid advancement of intelligent education, Computerized Adaptive Testing (CAT) has attracted increasing attention by integrating educational psychology with deep learning technologies. Unlike traditional paper-and-pencil testing, CAT aims to efficiently and accurately assess examinee abilities by adaptively selecting the most suitable items during the assessment process. However, its real-time and sequential nature presents limitations in practical scenarios, particularly in large-scale assessments where interaction costs are high, or in sensitive domains such as psychological evaluations where minimizing noise and interference is essential. These challenges constrain the applicability of conventional CAT methods in time-sensitive or resourceconstrained environments. To this end, we first introduce a novel task called one-shot adaptive testing (OAT), which aims to select a fixed set of optimal items for each test-taker in a one-time selection. Meanwhile, we propose PEOAT, a Personalization-guided Evolutionary question assembly framework for One-shot Adaptive Testing from the perspective of combinatorial optimization. Specifically, we began by designing a personalization-aware initialization strategy that integrates differences between examinee ability and exercise difficulty, using multi-strategy sampling to construct a diverse and informative initial population. Building on this, we proposed a cognitive-enhanced evolutionary framework incorporating schema-preserving crossover and cognitively guided mutation to enable efficient exploration through informative signals. To maintain diversity without compromising fitness, we further introduced a diversity-aware environmental selection mechanism. The effectiveness of PEOAT is validated through extensive experiments on two datasets, complemented by case studies that uncovered valuable insights. 

---
# Breaking It Down: Domain-Aware Semantic Segmentation for Retrieval Augmented Generation 

**Authors**: Aparajitha Allamraju, Maitreya Prafulla Chitale, Hiranmai Sri Adibhatla, Rahul Mishra, Manish Shrivastava  

**Link**: [PDF](https://arxiv.org/pdf/2512.00367)  

**Abstract**: Document chunking is a crucial component of Retrieval-Augmented Generation (RAG), as it directly affects the retrieval of relevant and precise context. Conventional fixed-length and recursive splitters often produce arbitrary, incoherent segments that fail to preserve semantic structure. Although semantic chunking has gained traction, its influence on generation quality remains underexplored. This paper introduces two efficient semantic chunking methods, Projected Similarity Chunking (PSC) and Metric Fusion Chunking (MFC), trained on PubMed data using three different embedding models. We further present an evaluation framework that measures the effect of chunking on both retrieval and generation by augmenting PubMedQA with full-text PubMed Central articles. Our results show substantial retrieval improvements (24x with PSC) in MRR and higher Hits@k on PubMedQA. We provide a comprehensive analysis, including statistical significance and response-time comparisons with common chunking libraries. Despite being trained on a single domain, PSC and MFC also generalize well, achieving strong out-of-domain generation performance across multiple datasets. Overall, our findings confirm that our semantic chunkers, especially PSC, consistently deliver superior performance. 

---
# Evolving Paradigms in Task-Based Search and Learning: A Comparative Analysis of Traditional Search Engine with LLM-Enhanced Conversational Search System 

**Authors**: Zhitong Guan, Yi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.00313)  

**Abstract**: Large Language Models (LLMs) are rapidly reshaping information retrieval by enabling interactive, generative, and inference-driven search. While traditional keyword-based search remains central to web and academic information access, it often struggles to support multi-step reasoning and exploratory learning tasks. LLM-powered search interfaces, such as ChatGPT and Claude, introduce new capabilities that may influence how users formulate queries, navigate information, and construct knowledge. However, empirical understanding of these effects is still limited. This study compares search behavior and learning outcomes in two environments: a standard search engine and an LLM-powered search system. We investigate (1) how search strategies, query formulation, and evaluation behaviors differ across systems, and (2) how LLM use affects comprehension, knowledge integration, and critical thinking during search-based learning tasks. Findings offer insight into how generative AI shapes information-seeking processes and contribute to ongoing discussions in information retrieval, human-AI interaction, and technology-supported learning. 

---
# Use of Retrieval-Augmented Large Language Model Agent for Long-Form COVID-19 Fact-Checking 

**Authors**: Jingyi Huang, Yuyi Yang, Mengmeng Ji, Charles Alba, Sheng Zhang, Ruopeng An  

**Link**: [PDF](https://arxiv.org/pdf/2512.00007)  

**Abstract**: The COVID-19 infodemic calls for scalable fact-checking solutions that handle long-form misinformation with accuracy and reliability. This study presents SAFE (system for accurate fact extraction and evaluation), an agent system that combines large language models with retrieval-augmented generation (RAG) to improve automated fact-checking of long-form COVID-19 misinformation. SAFE includes two agents - one for claim extraction and another for claim verification using LOTR-RAG, which leverages a 130,000-document COVID-19 research corpus. An enhanced variant, SAFE (LOTR-RAG + SRAG), incorporates Self-RAG to refine retrieval via query rewriting. We evaluated both systems on 50 fake news articles (2-17 pages) containing 246 annotated claims (M = 4.922, SD = 3.186), labeled as true (14.1%), partly true (14.4%), false (27.0%), partly false (2.2%), and misleading (21.0%) by public health professionals. SAFE systems significantly outperformed baseline LLMs in all metrics (p < 0.001). For consistency (0-1 scale), SAFE (LOTR-RAG) scored 0.629, exceeding both SAFE (+SRAG) (0.577) and the baseline (0.279). In subjective evaluations (0-4 Likert scale), SAFE (LOTR-RAG) also achieved the highest average ratings in usefulness (3.640), clearness (3.800), and authenticity (3.526). Adding SRAG slightly reduced overall performance, except for a minor gain in clearness. SAFE demonstrates robust improvements in long-form COVID-19 fact-checking by addressing LLM limitations in consistency and explainability. The core LOTR-RAG design proved more effective than its SRAG-augmented variant, offering a strong foundation for scalable misinformation mitigation. 

---
# Enhancing Talent Search Ranking with Role-Aware Expert Mixtures and LLM-based Fine-Grained Job Descriptions 

**Authors**: Jihang Li, Bing Xu, Zulong Chen, Chuanfei Xu, Minping Chen, Suyu Liu, Ying Zhou, Zeyi Wen  

**Link**: [PDF](https://arxiv.org/pdf/2512.00004)  

**Abstract**: Talent search is a cornerstone of modern recruitment systems, yet existing approaches often struggle to capture nuanced job-specific preferences, model recruiter behavior at a fine-grained level, and mitigate noise from subjective human judgments. We present a novel framework that enhances talent search effectiveness and delivers substantial business value through two key innovations: (i) leveraging LLMs to extract fine-grained recruitment signals from job descriptions and historical hiring data, and (ii) employing a role-aware multi-gate MoE network to capture behavioral differences across recruiter roles. To further reduce noise, we introduce a multi-task learning module that jointly optimizes click-through rate (CTR), conversion rate (CVR), and resume matching relevance. Experiments on real-world recruitment data and online A/B testing show relative AUC gains of 1.70% (CTR) and 5.97% (CVR), and a 17.29% lift in click-through conversion rate. These improvements reduce dependence on external sourcing channels, enabling an estimated annual cost saving of millions of CNY. 

---
# MMAG: Mixed Memory-Augmented Generation for Large Language Models Applications 

**Authors**: Stefano Zeppieri  

**Link**: [PDF](https://arxiv.org/pdf/2512.01710)  

**Abstract**: Large Language Models (LLMs) excel at generating coherent text within a single prompt but fall short in sustaining relevance, personalization, and continuity across extended interactions. Human communication, however, relies on multiple forms of memory, from recalling past conversations to adapting to personal traits and situational context. This paper introduces the Mixed Memory-Augmented Generation (MMAG) pattern, a framework that organizes memory for LLM-based agents into five interacting layers: conversational, long-term user, episodic and event-linked, sensory and context-aware, and short-term working memory. Drawing inspiration from cognitive psychology, we map these layers to technical components and outline strategies for coordination, prioritization, and conflict resolution. We demonstrate the approach through its implementation in the Heero conversational agent, where encrypted long-term bios and conversational history already improve engagement and retention. We further discuss implementation concerns around storage, retrieval, privacy, and latency, and highlight open challenges. MMAG provides a foundation for building memory-rich language agents that are more coherent, proactive, and aligned with human needs. 

---
# Text Mining Analysis of Symptom Patterns in Medical Chatbot Conversations 

**Authors**: Hamed Razavi  

**Link**: [PDF](https://arxiv.org/pdf/2512.00768)  

**Abstract**: The fast growth of digital health systems has led to a need to better comprehend how they interpret and represent patient-reported symptoms. Chatbots have been used in healthcare to provide clinical support and enhance the user experience, making it possible to provide meaningful clinical patterns from text-based data through chatbots. The proposed research utilises several different natural language processing methods to study the occurrences of symptom descriptions in medicine as well as analyse the patterns that emerge through these conversations within medical bots. Through the use of the Medical Conversations to Disease Dataset which contains 960 multi-turn dialogues divided into 24 Clinical Conditions, a standardised representation of conversations between patient and bot is created for further analysis by computational means. The multi-method approach uses a variety of tools, including Latent Dirichlet Allocation (LDA) to identify latent symptom themes, K-Means to group symptom descriptions by similarity, Transformer-based Named Entity Recognition (NER) to extract medical concepts, and the Apriori algorithm to discover frequent symptom pairs. Findings from the analysis indicate a coherent structure of clinically relevant topics, moderate levels of clustering cohesiveness and several high confidence rates on the relationships between symptoms like fever headache and rash itchiness. The results support the notion that conversational medical data can be a valuable diagnostic signal for early symptom interpretation, assist in strengthening decision support and improve how users interact with tele-health technology. By demonstrating a method for converting unstructured free-flowing dialogue into actionable knowledge regarding symptoms this work provides an extensible framework to further enhance future performance, dependability and clinical utility of selecting medical chatbots. 

---
# Upcycled and Merged MoE Reward Model for Mitigating Reward Hacking 

**Authors**: Lingling Fu  

**Link**: [PDF](https://arxiv.org/pdf/2512.00724)  

**Abstract**: Reward models play a critical role in Reinforcement Learning from Human Feedback (RLHF) by assessing the consistency between generated outputs and human preferences. However, conventional reward models are prone to reward hacking or over-optimization, where the policy exploits shortcut patterns to obtain high reward scores that do not reflect true human preference. Although Mixture-of-Experts (MoE)-based reward models can enhance discriminative capability, they typically introduce substantial computational overhead. To address these challenges, we propose an upcycle and merge MoE reward modeling approach. We first upcycle a dense reward model into a MoE architecture, where a shared expert captures general knowledge, while normal experts specialize in instruction-specific patterns. We then apply routing-weight normalization and merge experts back into a dense model through a learnable weight-averaging mechanism, preserving performance gains while significantly reducing inference cost. Experimental results demonstrate that our method effectively mitigates reward hacking across various model scales. Our work highlights the potential of upcycle and merge MoE structures for improving both robustness and efficiency of RLHF reward models. 

---
# Cross-Domain Federated Semantic Communication with Global Representation Alignment and Domain-Aware Aggregation 

**Authors**: Loc X. Nguyen, Ji Su Yoon, Huy Q. Le, Yu Qiao, Avi Deb Raha, Eui-Nam Huh, Walid Saad, Dusit Niyato, Zhu Han, Choong Seon Hong  

**Link**: [PDF](https://arxiv.org/pdf/2512.00711)  

**Abstract**: Semantic communication can significantly improve bandwidth utilization in wireless systems by exploiting the meaning behind raw data. However, the advancements achieved through semantic communication are closely dependent on the development of deep learning (DL) models for joint source-channel coding (JSCC) encoder/decoder techniques, which require a large amount of data for training. To address this data-intensive nature of DL models, federated learning (FL) has been proposed to train a model in a distributed manner, where the server broadcasts the DL model to clients in the network for training with their local data. However, the conventional FL approaches suffer from catastrophic degradation when client data are from different domains. In contrast, in this paper, a novel FL framework is proposed to address this domain shift by constructing the global representation, which aligns with the local features of the clients to preserve the semantics of different data domains. In addition, the dominance problem of client domains with a large number of samples is identified and, then, addressed with a domain-aware aggregation approach. This work is the first to consider the domain shift in training the semantic communication system for the image reconstruction task. Finally, simulation results demonstrate that the proposed approach outperforms the model-contrastive FL (MOON) framework by 0.5 for PSNR values under three domains at an SNR of 1 dB, and this gap continues to widen as the channel quality improves. 

---
# Mitigating the Threshold Priming Effect in Large Language Model-Based Relevance Judgments via Personality Infusing 

**Authors**: Nuo Chen, Hanpei Fang, Jiqun Liu, Wilson Wei, Tetsuya Sakai, Xiao-Ming Wu  

**Link**: [PDF](https://arxiv.org/pdf/2512.00390)  

**Abstract**: Recent research has explored LLMs as scalable tools for relevance labeling, but studies indicate they are susceptible to priming effects, where prior relevance judgments influence later ones. Although psychological theories link personality traits to such biases, it is unclear whether simulated personalities in LLMs exhibit similar effects. We investigate how Big Five personality profiles in LLMs influence priming in relevance labeling, using multiple LLMs on TREC 2021 and 2022 Deep Learning Track datasets. Our results show that certain profiles, such as High Openness and Low Neuroticism, consistently reduce priming susceptibility. Additionally, the most effective personality in mitigating priming may vary across models and task types. Based on these findings, we propose personality prompting as a method to mitigate threshold priming, connecting psychological evidence with LLM-based evaluation practices. 

---
# The Information Theory of Similarity 

**Authors**: Nikit Phadke  

**Link**: [PDF](https://arxiv.org/pdf/2512.00378)  

**Abstract**: We establish a precise mathematical equivalence between witness-based similarity systems (REWA) and Shannon's information theory. We prove that witness overlap is mutual information, that REWA bit complexity bounds arise from channel capacity limitations, and that ranking-preserving encodings obey rate-distortion constraints. This unification reveals that fifty years of similarity search research -- from Bloom filters to locality-sensitive hashing to neural retrieval -- implicitly developed information theory for relational data. We derive fundamental lower bounds showing that REWA's $O(\Delta^{-2} \log N)$ complexity is optimal: no encoding scheme can preserve similarity rankings with fewer bits. The framework establishes that semantic similarity has physical units (bits of mutual information), search is communication (query transmission over a noisy channel), and retrieval systems face fundamental capacity limits analogous to Shannon's channel coding theorem. 

---
# Evidence-Guided Schema Normalization for Temporal Tabular Reasoning 

**Authors**: Ashish Thanga, Vibhu Dixit, Abhilash Shankarampeta, Vivek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2512.00329)  

**Abstract**: Temporal reasoning over evolving semi-structured tables poses a challenge to current QA systems. We propose a SQL-based approach that involves (1) generating a 3NF schema from Wikipedia infoboxes, (2) generating SQL queries, and (3) query execution. Our central finding challenges model scaling assumptions: the quality of schema design has a greater impact on QA precision than model capacity. We establish three evidence-based principles: normalization that preserves context, semantic naming that reduces ambiguity, and consistent temporal anchoring. Our best configuration (Gemini 2.5 Flash schema + Gemini-2.0-Flash queries) achieves 80.39 EM, a 16.8\% improvement over the baseline (68.89 EM). 

---
