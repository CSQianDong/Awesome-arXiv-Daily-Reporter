# CroPS: Improving Dense Retrieval with Cross-Perspective Positive Samples in Short-Video Search 

**Authors**: Ao Xie, Jiahui Chen, Quanzhi Zhu, Xiaoze Jiang, Zhiheng Qin, Enyun Yu, Han Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.15443)  

**Abstract**: Dense retrieval has become a foundational paradigm in modern search systems, especially on short-video platforms. However, most industrial systems adopt a self-reinforcing training pipeline that relies on historically exposed user interactions for supervision. This paradigm inevitably leads to a filter bubble effect, where potentially relevant but previously unseen content is excluded from the training signal, biasing the model toward narrow and conservative retrieval. In this paper, we present CroPS (Cross-Perspective Positive Samples), a novel retrieval data engine designed to alleviate this problem by introducing diverse and semantically meaningful positive examples from multiple perspectives. CroPS enhances training with positive signals derived from user query reformulation behavior (query-level), engagement data in recommendation streams (system-level), and world knowledge synthesized by large language models (knowledge-level). To effectively utilize these heterogeneous signals, we introduce a Hierarchical Label Assignment (HLA) strategy and a corresponding H-InfoNCE loss that together enable fine-grained, relevance-aware optimization. Extensive experiments conducted on Kuaishou Search, a large-scale commercial short-video search platform, demonstrate that CroPS significantly outperforms strong baselines both offline and in live A/B tests, achieving superior retrieval performance and reducing query reformulation rates. CroPS is now fully deployed in Kuaishou Search, serving hundreds of millions of users daily. 

---
# Unveiling Inference Scaling for Difference-Aware User Modeling in LLM Personalization 

**Authors**: Suyu Chen, Yimeng Bai, Yulong Huang, Xiaoyan Zhao, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.15389)  

**Abstract**: Large Language Models (LLMs) are increasingly integrated into users' daily lives, driving a growing demand for personalized outputs. Prior work has primarily leveraged a user's own history, often overlooking inter-user differences that are critical for effective personalization. While recent methods have attempted to model such differences, their feature extraction processes typically rely on fixed dimensions and quick, intuitive inference (System-1 thinking), limiting both the coverage and granularity of captured user differences. To address these limitations, we propose Difference-aware Reasoning Personalization (DRP), a framework that reconstructs the difference extraction mechanism by leveraging inference scaling to enhance LLM personalization. DRP autonomously identifies relevant difference feature dimensions and generates structured definitions and descriptions, enabling slow, deliberate reasoning (System-2 thinking) over user differences. Experiments on personalized review generation demonstrate that DRP consistently outperforms baseline methods across multiple metrics. 

---
# Selective Mixup for Debiasing Question Selection in Computerized Adaptive Testing 

**Authors**: Mi Tian, Kun Zhang, Fei Liu, Jinglong Li, Yuxin Liao, Chenxi Bai, Zhengtao Tan, Le Wu, Richang Hong  

**Link**: [PDF](https://arxiv.org/pdf/2511.15241)  

**Abstract**: Computerized Adaptive Testing (CAT) is a widely used technology for evaluating learners' proficiency in online education platforms. By leveraging prior estimates of proficiency to select questions and updating the estimates iteratively based on responses, CAT enables personalized learner modeling and has attracted substantial attention. Despite this progress, most existing works focus primarily on improving diagnostic accuracy, while overlooking the selection bias inherent in the adaptive process. Selection Bias arises because the question selection is strongly influenced by the estimated proficiency, such as assigning easier questions to learners with lower proficiency and harder ones to learners with higher proficiency. Since the selection depends on prior estimation, this bias propagates into the diagnosis model, which is further amplified during iterative updates, leading to misalignment and biased predictions. Moreover, the imbalanced nature of learners' historical interactions often exacerbates the bias in diagnosis models. To address this issue, we propose a debiasing framework consisting of two key modules: Cross-Attribute Examinee Retrieval and Selective Mixup-based Regularization. First, we retrieve balanced examinees with relatively even distributions of correct and incorrect responses and use them as neutral references for biased examinees. Then, mixup is applied between each biased examinee and its matched balanced counterpart under label consistency. This augmentation enriches the diversity of bias-conflicting samples and smooths selection boundaries. Finally, extensive experiments on two benchmark datasets with multiple advanced diagnosis models demonstrate that our method substantially improves both the generalization ability and fairness of question selection in CAT. 

---
# ItemRAG: Item-Based Retrieval-Augmented Generation for LLM-Based Recommendation 

**Authors**: Sunwoo Kim, Geon Lee, Kyungho Kim, Jaemin Yoo, Kijung Shin  

**Link**: [PDF](https://arxiv.org/pdf/2511.15141)  

**Abstract**: Recently, large language models (LLMs) have been widely used as recommender systems, owing to their strong reasoning capability and their effectiveness in handling cold-start items. To better adapt LLMs for recommendation, retrieval-augmented generation (RAG) has been incorporated. Most existing RAG methods are user-based, retrieving purchase patterns of users similar to the target user and providing them to the LLM. In this work, we propose ItemRAG, an item-based RAG method for LLM-based recommendation that retrieves relevant items (rather than users) from item-item co-purchase histories. ItemRAG helps LLMs capture co-purchase patterns among items, which are beneficial for recommendations. Especially, our retrieval strategy incorporates semantically similar items to better handle cold-start items and uses co-purchase frequencies to improve the relevance of the retrieved items. Through extensive experiments, we demonstrate that ItemRAG consistently (1) improves the zero-shot LLM-based recommender by up to 43% in Hit-Ratio-1 and (2) outperforms user-based RAG baselines under both standard and cold-start item recommendation settings. 

---
# Multi-Aspect Cross-modal Quantization for Generative Recommendation 

**Authors**: Fuwei Zhang, Xiaoyu Liu, Dongbo Xi, Jishen Yin, Huan Chen, Peng Yan, Fuzhen Zhuang, Zhao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.15122)  

**Abstract**: Generative Recommendation (GR) has emerged as a new paradigm in recommender systems. This approach relies on quantized representations to discretize item features, modeling users' historical interactions as sequences of discrete tokens. Based on these tokenized sequences, GR predicts the next item by employing next-token prediction methods. The challenges of GR lie in constructing high-quality semantic identifiers (IDs) that are hierarchically organized, minimally conflicting, and conducive to effective generative model training. However, current approaches remain limited in their ability to harness multimodal information and to capture the deep and intricate interactions among diverse modalities, both of which are essential for learning high-quality semantic IDs and for effectively training GR models. To address this, we propose Multi-Aspect Cross-modal quantization for generative Recommendation (MACRec), which introduces multimodal information and incorporates it into both semantic ID learning and generative model training from different aspects. Specifically, we first introduce cross-modal quantization during the ID learning process, which effectively reduces conflict rates and thus improves codebook usability through the complementary integration of multimodal information. In addition, to further enhance the generative ability of our GR model, we incorporate multi-aspect cross-modal alignments, including the implicit and explicit alignments. Finally, we conduct extensive experiments on three well-known recommendation datasets to demonstrate the effectiveness of our proposed method. 

---
# SilverTorch: A Unified Model-based System to Democratize Large-Scale Recommendation on GPUs 

**Authors**: Bi Xue, Hong Wu, Lei Chen, Chao Yang, Yiming Ma, Fei Ding, Zhen Wang, Liang Wang, Xiaoheng Mao, Ke Huang, Xialu Li, Peng Xia, Rui Jian, Yanli Zhao, Yanzun Huang, Yijie Deng, Harry Tran, Ryan Chang, Min Yu, Eric Dong, Jiazhou Wang, Qianqian Zhang, Keke Zhai, Hongzhang Yin, Pawel Garbacki, Zheng Fang, Yiyi Pan, Min Ni, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.14881)  

**Abstract**: Serving deep learning based recommendation models (DLRM) at scale is challenging. Existing systems rely on CPU-based ANN indexing and filtering services, suffering from non-negligible costs and forgoing joint optimization opportunities. Such inefficiency makes them difficult to support more complex model architectures, such as learned similarities and multi-task retrieval.
In this paper, we propose SilverTorch, a model-based system for serving recommendation models on GPUs. SilverTorch unifies model serving by replacing standalone indexing and filtering services with layers of served models. We propose a Bloom index algorithm on GPUs for feature filtering and a tensor-native fused Int8 ANN kernel on GPUs for nearest neighbor search. We further co-design the ANN search index and filtering index to reduce GPU memory utilization and eliminate unnecessary computation. Benefit from SilverTorch's serving paradigm, we introduce a OverArch scoring layer and a Value Model to aggregate results across multi-tasks. These advancements improve the accuracy for retrieval and enable future studies for serving more complex models. For ranking, SilverTorch's design accelerates item embedding calculation by caching the pre-calculated embeddings inside the serving model.
Our evaluation on the industry-scale datasets show that SilverTorch achieves up to 5.6x lower latency and 23.7x higher throughput compared to the state-of-the-art approaches. We also demonstrate that SilverTorch's solution is 13.35x more cost-efficient than CPU-based solution while improving accuracy via serving more complex models. SilverTorch serves over hundreds of models online across major products and recommends contents for billions of daily active users. 

---
# ExplainRec: Towards Explainable Multi-Modal Zero-Shot Recommendation with Preference Attribution and Large Language Models 

**Authors**: Bo Ma, LuYao Liu, ZeHua Hu, Simon Lau  

**Link**: [PDF](https://arxiv.org/pdf/2511.14770)  

**Abstract**: Recent advances in Large Language Models (LLMs) have opened new possibilities for recommendation systems, though current approaches such as TALLRec face challenges in explainability and cold-start scenarios. We present ExplainRec, a framework that extends LLM-based recommendation capabilities through preference attribution, multi-modal fusion, and zero-shot transfer learning. The framework incorporates four technical contributions: preference attribution tuning for explainable recommendations, zero-shot preference transfer for cold-start users and items, multi-modal enhancement leveraging visual and textual content, and multi-task collaborative optimization. Experimental evaluation on MovieLens-25M and Amazon datasets shows that ExplainRec outperforms existing methods, achieving AUC improvements of 0.7\% on movie recommendation and 0.9\% on cross-domain tasks, while generating interpretable explanations and handling cold-start scenarios effectively. 

---
# Cluster-based Adaptive Retrieval: Dynamic Context Selection for RAG Applications 

**Authors**: Yifan Xu, Vipul Gupta, Rohit Aggarwal, Varsha Mahadevan, Bhaskar Krishnamachari  

**Link**: [PDF](https://arxiv.org/pdf/2511.14769)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by pulling in external material, document, code, manuals, from vast and ever-growing corpora, to effectively answer user queries. The effectiveness of RAG depends significantly on aligning the number of retrieved documents with query characteristics: narrowly focused queries typically require fewer, highly relevant documents, whereas broader or ambiguous queries benefit from retrieving more extensive supporting information. However, the common static top-k retrieval approach fails to adapt to this variability, resulting in either insufficient context from too few documents or redundant information from too many. Motivated by these challenges, we introduce Cluster-based Adaptive Retrieval (CAR), an algorithm that dynamically determines the optimal number of documents by analyzing the clustering patterns of ordered query-document similarity distances. CAR detects the transition point within similarity distances, where tightly clustered, highly relevant documents shift toward less pertinent candidates, establishing an adaptive cut-off that scales with query complexity. On Coinbase's CDP corpus and the public MultiHop-RAG benchmark, CAR consistently picks the optimal retrieval depth and achieves the highest TES score, outperforming every fixed top-k baseline. In downstream RAG evaluations, CAR cuts LLM token usage by 60%, trims end-to-end latency by 22%, and reduces hallucinations by 10% while fully preserving answer relevance. Since integrating CAR into Coinbase's virtual assistant, we've seen user engagement jump by 200%. 

---
# Causally-Informed Reinforcement Learning for Adaptive Emotion-Aware Social Media Recommendation 

**Authors**: Bhavika Jain, Robert Pitsko, Ananya Drishti, Mahfuza Farooque  

**Link**: [PDF](https://arxiv.org/pdf/2511.14768)  

**Abstract**: Social media recommendation systems play a central role in shaping users' emotional experiences. However, most systems are optimized solely for engagement metrics, such as click rate, viewing time, or scrolling, without accounting for users' emotional states. Repeated exposure to emotionally charged content has been shown to negatively affect users' emotional well-being over time. We propose an Emotion-aware Social Media Recommendation (ESMR) framework that personalizes content based on users' evolving emotional trajectories. ESMR integrates a Transformer-based emotion predictor with a hybrid recommendation policy: a LightGBM model for engagement during stable periods and a reinforcement learning agent with causally informed rewards when negative emotional states persist. Through behaviorally grounded evaluation over 30-day interaction traces, ESMR demonstrates improved emotional recovery, reduced volatility, and strong engagement retention. ESMR offers a path toward emotionally aware recommendations without compromising engagement performance. 

---
# An LLM-Powered Agent for Real-Time Analysis of the Vietnamese IT Job Market 

**Authors**: Minh-Thuan Nguyen, Thien Vo-Thanh, Thai-Duy Dinh, Xuan-Quang Phan, Tan-Ha Mai, Lam-Son Lê  

**Link**: [PDF](https://arxiv.org/pdf/2511.14767)  

**Abstract**: Individuals entering Vietnam's dynamic Information Technology (IT) job market face a critical gap in reliable career guidance. Existing market reports are often outdated, while the manual analysis of thousands of job postings is impractical for most. To address this challenge, we present the AI Job Market Consultant, a novel conversational agent that delivers deep, data-driven insights directly from the labor market in real-time. The foundation of our system is a custom-built dataset created via an automated pipeline that crawls job portals using Playwright and leverages the Large Language Model (LLM) to intelligently structure unstructured posting data. The core of our system is a tool-augmented AI agent, based on the ReAct agentic framework, which enables the ability of autonomously reasoning, planning, and executing actions through a specialized toolbox for SQL queries, semantic search, and data visualization. Our prototype successfully collected and analyzed 3,745 job postings, demonstrating its ability to answer complex, multi-step queries, generate on-demand visualizations, and provide personalized career advice grounded in real-world data. This work introduces a new paradigm for labor market analysis, showcasing how specialized agentic AI systems can democratize access to timely, trustworthy career intelligence for the next generation of professionals. 

---
# OTCR: Optimal Transmission, Compression and Representation for Multimodal Information Extraction 

**Authors**: Yang Li, Yajiao Wang, Wenhao Hu, Zhixiong Zhang, Mengting Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.14766)  

**Abstract**: Multimodal Information Extraction (MIE) requires fusing text and visual cues from visually rich documents. While recent methods have advanced multimodal representation learning, most implicitly assume modality equivalence or treat modalities in a largely uniform manner, still relying on generic fusion paradigms. This often results in indiscriminate incorporation of multimodal signals and insufficient control over task-irrelevant redundancy, which may in turn limit generalization. We revisit MIE from a task-centric view: text should dominate, vision should selectively support. We present OTCR, a two-stage framework. First, Cross-modal Optimal Transport (OT) yields sparse, probabilistic alignments between text tokens and visual patches, with a context-aware gate controlling visual injection. Second, a Variational Information Bottleneck (VIB) compresses fused features, filtering task-irrelevant noise to produce compact, task-adaptive representations. On FUNSD, OTCR achieves 91.95% SER and 91.13% RE, while on XFUND (ZH), it reaches 91.09% SER and 94.20% RE, demonstrating competitive performance across datasets. Feature-level analyses further confirm reduced modality redundancy and strengthened task signals. This work offers an interpretable, information-theoretic paradigm for controllable multimodal fusion in document AI. 

---
# Optimizing Agricultural Research: A RAG-Based Approach to Mycorrhizal Fungi Information 

**Authors**: Mohammad Usman Altam, Md Imtiaz Habib, Tuan Hoang  

**Link**: [PDF](https://arxiv.org/pdf/2511.14765)  

**Abstract**: Retrieval-Augmented Generation (RAG) represents a transformative approach within natural language processing (NLP), combining neural information retrieval with generative language modeling to enhance both contextual accuracy and factual reliability of responses. Unlike conventional Large Language Models (LLMs), which are constrained by static training corpora, RAG-powered systems dynamically integrate domain-specific external knowledge sources, thereby overcoming temporal and disciplinary limitations. In this study, we present the design and evaluation of a RAG-enabled system tailored for Mycophyto, with a focus on advancing agricultural applications related to arbuscular mycorrhizal fungi (AMF). These fungi play a critical role in sustainable agriculture by enhancing nutrient acquisition, improving plant resilience under abiotic and biotic stresses, and contributing to soil health. Our system operationalizes a dual-layered strategy: (i) semantic retrieval and augmentation of domain-specific content from agronomy and biotechnology corpora using vector embeddings, and (ii) structured data extraction to capture predefined experimental metadata such as inoculation methods, spore densities, soil parameters, and yield outcomes. This hybrid approach ensures that generated responses are not only semantically aligned but also supported by structured experimental evidence. To support scalability, embeddings are stored in a high-performance vector database, allowing near real-time retrieval from an evolving literature base. Empirical evaluation demonstrates that the proposed pipeline retrieves and synthesizes highly relevant information regarding AMF interactions with crop systems, such as tomato (Solanum lycopersicum). The framework underscores the potential of AI-driven knowledge discovery to accelerate agroecological innovation and enhance decision-making in sustainable farming systems. 

---
# Image-Seeking Intent Prediction for Cross-Device Product Search 

**Authors**: Mariya Hendriksen, Svitlana Vakulenko, Jordan Massiah, Gabriella Kazai, Emine Yilmaz  

**Link**: [PDF](https://arxiv.org/pdf/2511.14764)  

**Abstract**: Large Language Models (LLMs) are transforming personalized search, recommendations, and customer interaction in e-commerce. Customers increasingly shop across multiple devices, from voice-only assistants to multimodal displays, each offering different input and output capabilities. A proactive suggestion to switch devices can greatly improve the user experience, but it must be offered with high precision to avoid unnecessary friction. We address the challenge of predicting when a query requires visual augmentation and a cross-device switch to improve product discovery. We introduce Image-Seeking Intent Prediction, a novel task for LLM-driven e-commerce assistants that anticipates when a spoken product query should proactively trigger a visual on a screen-enabled device. Using large-scale production data from a multi-device retail assistant, including 900K voice queries, associated product retrievals, and behavioral signals such as image carousel engagement, we train IRP (Image Request Predictor), a model that leverages user input query and corresponding retrieved product metadata to anticipate visual intent. Our experiments show that combining query semantics with product data, particularly when improved through lightweight summarization, consistently improves prediction accuracy. Incorporating a differentiable precision-oriented loss further reduces false positives. These results highlight the potential of LLMs to power intelligent, cross-device shopping assistants that anticipate and adapt to user needs, enabling more seamless and personalized e-commerce experiences. 

---
# Membership Inference Attack against Large Language Model-based Recommendation Systems: A New Distillation-based Paradigm 

**Authors**: Li Cuihong, Huang Xiaowen, Yin Chuanhuan, Sang Jitao  

**Link**: [PDF](https://arxiv.org/pdf/2511.14763)  

**Abstract**: Membership Inference Attack (MIA) aims to determine if a data sample is used in the training dataset of a target model. Traditional MIA obtains feature of target model via shadow models and uses the feature to train attack model, but the scale and complexity of training or fine-tuning data for large language model (LLM)-based recommendation systems make shadow models difficult to construct. Knowledge distillation as a method for extracting knowledge contributes to construct a stronger reference model. Knowledge distillation enables separate distillation for member and non-member data during the distillation process, enhancing the model's discriminative capability between the two in MIA. This paper propose a knowledge distillation-based MIA paradigm to improve the performance of membership inference attacks on LLM-based recommendation systems. Our paradigm introduces knowledge distillation to obtain a reference model, which enhances the reference model's ability to distinguish between member and non-member data. We obtain individual features from the reference model and train our attack model with fused feature. Our paradigm improves the attack performance of MIA compared to shadow model-based attack. 

---
# HV-Attack: Hierarchical Visual Attack for Multimodal Retrieval Augmented Generation 

**Authors**: Linyin Luo, Yujuan Ding, Yunshan Ma, Wenqi Fan, Hanjiang Lai  

**Link**: [PDF](https://arxiv.org/pdf/2511.15435)  

**Abstract**: Advanced multimodal Retrieval-Augmented Generation (MRAG) techniques have been widely applied to enhance the capabilities of Large Multimodal Models (LMMs), but they also bring along novel safety issues. Existing adversarial research has revealed the vulnerability of MRAG systems to knowledge poisoning attacks, which fool the retriever into recalling injected poisoned contents. However, our work considers a different setting: visual attack of MRAG by solely adding imperceptible perturbations at the image inputs of users, without manipulating any other components. This is challenging due to the robustness of fine-tuned retrievers and large-scale generators, and the effect of visual perturbation may be further weakened by propagation through the RAG chain. We propose a novel Hierarchical Visual Attack that misaligns and disrupts the two inputs (the multimodal query and the augmented knowledge) of MRAG's generator to confuse its generation. We further design a hierarchical two-stage strategy to obtain misaligned augmented knowledge. We disrupt the image input of the retriever to make it recall irrelevant knowledge from the original database, by optimizing the perturbation which first breaks the cross-modal alignment and then disrupts the multimodal semantic alignment. We conduct extensive experiments on two widely-used MRAG datasets: OK-VQA and InfoSeek. We use CLIP-based retrievers and two LMMs BLIP-2 and LLaVA as generators. Results demonstrate the effectiveness of our visual attack on MRAG through the significant decrease in both retrieval and generation performance. 

---
# NAMeGEn: Creative Name Generation via A Novel Agent-based Multiple Personalized Goal Enhancement Framework 

**Authors**: Shanlin Zhou, Xinpeng Wang, Jianxun Lian, Zhenghao Liu, Laks V.S. Lakshmanan, Xiaoyuan Yi, Yongtao Hao  

**Link**: [PDF](https://arxiv.org/pdf/2511.15408)  

**Abstract**: Trained on diverse human-authored texts, Large Language Models (LLMs) unlocked the potential for Creative Natural Language Generation (CNLG), benefiting various applications like advertising and storytelling. Nevertheless, CNLG still remains difficult due to two main challenges. (1) Multi-objective flexibility: user requirements are often personalized, fine-grained, and pluralistic, which LLMs struggle to satisfy simultaneously; (2) Interpretive complexity: beyond generation, creativity also involves understanding and interpreting implicit meaning to enhance users' perception. These challenges significantly limit current methods, especially in short-form text generation, in generating creative and insightful content. To address this, we focus on Chinese baby naming, a representative short-form CNLG task requiring adherence to explicit user constraints (e.g., length, semantics, anthroponymy) while offering meaningful aesthetic explanations. We propose NAMeGEn, a novel multi-agent optimization framework that iteratively alternates between objective extraction, name generation, and evaluation to meet diverse requirements and generate accurate explanations. To support this task, we further construct a classical Chinese poetry corpus with 17k+ poems to enhance aesthetics, and introduce CBNames, a new benchmark with tailored metrics. Extensive experiments demonstrate that NAMeGEn effectively generates creative names that meet diverse, personalized requirements while providing meaningful explanations, outperforming six baseline methods spanning various LLM backbones without any training. 

---
# A Compliance-Preserving Retrieval System for Aircraft MRO Task Search 

**Authors**: Byungho Jo  

**Link**: [PDF](https://arxiv.org/pdf/2511.15383)  

**Abstract**: Aircraft Maintenance Technicians (AMTs) spend up to 30% of work time searching manuals, a documented efficiency bottleneck in MRO operations where every procedure must be traceable to certified sources. We present a compliance-preserving retrieval system that adapts LLM reranking and semantic search to aviation MRO environments by operating alongside, rather than replacing, certified legacy viewers. The system constructs revision-robust embeddings from ATA chapter hierarchies and uses vision-language parsing to structure certified content, allowing technicians to preview ranked tasks and access verified procedures in existing viewers. Evaluation on 49k synthetic queries achieves >90% retrieval accuracy, while bilingual controlled studies with 10 licensed AMTs demonstrate 90.9% top-10 success rate and 95% reduction in lookup time, from 6-15 minutes to 18 seconds per task. These gains provide concrete evidence that semantic retrieval can operate within strict regulatory constraints and meaningfully reduce operational workload in real-world multilingual MRO workflows. 

---
# Opinion Dynamics Models for Sentiment Evolution in Weibo Blogs 

**Authors**: Yulong He, Anton V. Proskurnikov, Artem Sedakov  

**Link**: [PDF](https://arxiv.org/pdf/2511.15303)  

**Abstract**: Online social media platforms enable influencers to distribute content and quickly capture audience reactions, significantly shaping their promotional strategies and advertising agreements. Understanding how sentiment dynamics and emotional contagion unfold among followers is vital for influencers and marketers, as these processes shape engagement, brand perception, and purchasing behavior. While sentiment analysis tools effectively track sentiment fluctuations, dynamical models explaining their evolution remain limited, often neglecting network structures and interactions both among blogs and between their topic-focused follower groups. In this study, we tracked influential tech-focused Weibo bloggers over six months, quantifying follower sentiment from text-mined feedback. By treating each blogger's audience as a single "macro-agent", we find that sentiment trajectories follow the principle of iterative averaging -- a foundational mechanism in many dynamical models of opinion formation, a theoretical framework at the intersection of social network analysis and dynamical systems theory. The sentiment evolution aligns closely with opinion-dynamics models, particularly modified versions of the classical French-DeGroot model that incorporate delayed perception and distinguish between expressed and private opinions. The inferred influence structures reveal interdependencies among blogs that may arise from homophily, whereby emotionally similar users subscribe to the same blogs and collectively shape the shared sentiment expressed within these communities. 

---
# Beyond GeneGPT: A Multi-Agent Architecture with Open-Source LLMs for Enhanced Genomic Question Answering 

**Authors**: Haodong Chen, Guido Zuccon, Teerapong Leelanupab  

**Link**: [PDF](https://arxiv.org/pdf/2511.15061)  

**Abstract**: Genomic question answering often requires complex reasoning and integration across diverse biomedical sources. GeneGPT addressed this challenge by combining domain-specific APIs with OpenAI's code-davinci-002 large language model to enable natural language interaction with genomic databases. However, its reliance on a proprietary model limits scalability, increases operational costs, and raises concerns about data privacy and generalization.
In this work, we revisit and reproduce GeneGPT in a pilot study using open source models, including Llama 3.1, Qwen2.5, and Qwen2.5 Coder, within a monolithic architecture; this allows us to identify the limitations of this approach. Building on this foundation, we then develop OpenBioLLM, a modular multi-agent framework that extends GeneGPT by introducing agent specialization for tool routing, query generation, and response validation. This enables coordinated reasoning and role-based task execution.
OpenBioLLM matches or outperforms GeneGPT on over 90% of the benchmark tasks, achieving average scores of 0.849 on Gene-Turing and 0.830 on GeneHop, while using smaller open-source models without additional fine-tuning or tool-specific pretraining. OpenBioLLM's modular multi-agent design reduces latency by 40-50% across benchmark tasks, significantly improving efficiency without compromising model capability. The results of our comprehensive evaluation highlight the potential of open-source multi-agent systems for genomic question answering. Code and resources are available at this https URL. 

---
