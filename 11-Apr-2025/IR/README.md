# How do Large Language Models Understand Relevance? A Mechanistic Interpretability Perspective 

**Authors**: Qi Liu, Jiaxin Mao, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2504.07898)  

**Abstract**: Recent studies have shown that large language models (LLMs) can assess relevance and support information retrieval (IR) tasks such as document ranking and relevance judgment generation. However, the internal mechanisms by which off-the-shelf LLMs understand and operationalize relevance remain largely unexplored. In this paper, we systematically investigate how different LLM modules contribute to relevance judgment through the lens of mechanistic interpretability. Using activation patching techniques, we analyze the roles of various model components and identify a multi-stage, progressive process in generating either pointwise or pairwise relevance judgment. Specifically, LLMs first extract query and document information in the early layers, then process relevance information according to instructions in the middle layers, and finally utilize specific attention heads in the later layers to generate relevance judgments in the required format. Our findings provide insights into the mechanisms underlying relevance assessment in LLMs, offering valuable implications for future research on leveraging LLMs for IR tasks. 

---
# Siren Federate: Bridging document, relational, and graph models for exploratory graph analysis 

**Authors**: Georgeta Bordea, Stephane Campinas, Matteo Catena, Renaud Delbru  

**Link**: [PDF](https://arxiv.org/pdf/2504.07815)  

**Abstract**: Investigative workflows require interactive exploratory analysis on large heterogeneous knowledge graphs. Current databases show limitations in enabling such task. This paper discusses the architecture of Siren Federate, a system that efficiently supports exploratory graph analysis by bridging document-oriented, relational and graph models. Technical contributions include distributed join algorithms, adaptive query planning, query plan folding, semantic caching, and semi-join decomposition for path query. Semi-join decomposition addresses the exponential growth of intermediate results in path-based queries. Experiments show that Siren Federate exhibits low latency and scales well with the amount of data, the number of users, and the number of computing nodes. 

---
# FairEval: Evaluating Fairness in LLM-Based Recommendations with Personality Awareness 

**Authors**: Chandan Kumar Sah, Xiaoli Lian, Tony Xu, Li Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.07801)  

**Abstract**: Recent advances in Large Language Models (LLMs) have enabled their application to recommender systems (RecLLMs), yet concerns remain regarding fairness across demographic and psychological user dimensions. We introduce FairEval, a novel evaluation framework to systematically assess fairness in LLM-based recommendations. FairEval integrates personality traits with eight sensitive demographic attributes,including gender, race, and age, enabling a comprehensive assessment of user-level bias. We evaluate models, including ChatGPT 4o and Gemini 1.5 Flash, on music and movie recommendations. FairEval's fairness metric, PAFS, achieves scores up to 0.9969 for ChatGPT 4o and 0.9997 for Gemini 1.5 Flash, with disparities reaching 34.79 percent. These results highlight the importance of robustness in prompt sensitivity and support more inclusive recommendation systems. 

---
# CollEX -- A Multimodal Agentic RAG System Enabling Interactive Exploration of Scientific Collections 

**Authors**: Florian Schneider, Narges Baba Ahmadi, Niloufar Baba Ahmadi, Iris Vogel, Martin Semmann, Chris Biemann  

**Link**: [PDF](https://arxiv.org/pdf/2504.07643)  

**Abstract**: In this paper, we introduce CollEx, an innovative multimodal agentic Retrieval-Augmented Generation (RAG) system designed to enhance interactive exploration of extensive scientific collections. Given the overwhelming volume and inherent complexity of scientific collections, conventional search systems often lack necessary intuitiveness and interactivity, presenting substantial barriers for learners, educators, and researchers. CollEx addresses these limitations by employing state-of-the-art Large Vision-Language Models (LVLMs) as multimodal agents accessible through an intuitive chat interface. By abstracting complex interactions via specialized agents equipped with advanced tools, CollEx facilitates curiosity-driven exploration, significantly simplifying access to diverse scientific collections and records therein. Our system integrates textual and visual modalities, supporting educational scenarios that are helpful for teachers, pupils, students, and researchers by fostering independent exploration as well as scientific excitement and curiosity. Furthermore, CollEx serves the research community by discovering interdisciplinary connections and complementing visual data. We illustrate the effectiveness of our system through a proof-of-concept application containing over 64,000 unique records across 32 collections from a local scientific collection from a public university. 

---
# REANIMATOR: Reanimate Retrieval Test Collections with Extracted and Synthetic Resources 

**Authors**: Björn Engelmann, Fabian Haak, Philipp Schaer, Mani Erfanian Abdoust, Linus Netze, Meik Bittkowski  

**Link**: [PDF](https://arxiv.org/pdf/2504.07584)  

**Abstract**: Retrieval test collections are essential for evaluating information retrieval systems, yet they often lack generalizability across tasks. To overcome this limitation, we introduce REANIMATOR, a versatile framework designed to enable the repurposing of existing test collections by enriching them with extracted and synthetic resources. REANIMATOR enhances test collections from PDF files by parsing full texts and machine-readable tables, as well as related contextual information. It then employs state-of-the-art large language models to produce synthetic relevance labels. Including an optional human-in-the-loop step can help validate the resources that have been extracted and generated. We demonstrate its potential with a revitalized version of the TREC-COVID test collection, showcasing the development of a retrieval-augmented generation system and evaluating the impact of tables on retrieval-augmented generation. REANIMATOR enables the reuse of test collections for new applications, lowering costs and broadening the utility of legacy resources. 

---
# Explicit Uncertainty Modeling for Video Watch Time Prediction 

**Authors**: Shanshan Wu, Shuchang Liu, Shuai Zhang, Xiaoyu Yang, Xiang Li, Lantao Hu, Han Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.07575)  

**Abstract**: In video recommendation, a critical component that determines the system's recommendation accuracy is the watch-time prediction module, since how long a user watches a video directly reflects personalized preferences. One of the key challenges of this problem is the user's stochastic watch-time behavior. To improve the prediction accuracy for such an uncertain behavior, existing approaches show that one can either reduce the noise through duration bias modeling or formulate a distribution modeling task to capture the uncertainty. However, the uncontrolled uncertainty is not always equally distributed across users and videos, inducing a balancing paradox between the model accuracy and the ability to capture out-of-distribution samples. In practice, we find that the uncertainty of the watch-time prediction model also provides key information about user behavior, which, in turn, could benefit the prediction task itself. Following this notion, we derive an explicit uncertainty modeling strategy for the prediction model and propose an adversarial optimization framework that can better exploit the user watch-time behavior. This framework has been deployed online on an industrial video sharing platform that serves hundreds of millions of daily active users, which obtains a significant increase in users' video watch time by 0.31% through the online A/B test. Furthermore, extended offline experiments on two public datasets verify the effectiveness of the proposed framework across various watch-time prediction backbones. 

---
# Exploring Human-Like Thinking in Search Simulations with Large Language Models 

**Authors**: Erhan Zhang, Xingzhu Wang, Peiyuan Gong, Zixuan Yang, Jiaxin Mao  

**Link**: [PDF](https://arxiv.org/pdf/2504.07570)  

**Abstract**: Simulating user search behavior is a critical task in information retrieval, which can be employed for user behavior modeling, data augmentation, and system evaluation. Recent advancements in large language models (LLMs) have opened up new possibilities for generating human-like actions including querying, browsing, and clicking. In this work, we explore the integration of human-like thinking into search simulations by leveraging LLMs to simulate users' hidden cognitive processes. Specifically, given a search task and context, we prompt LLMs to first think like a human before executing the corresponding action. As existing search datasets do not include users' thought processes, we conducted a user study to collect a new dataset enriched with users' explicit thinking. We investigate the impact of incorporating such human-like thinking on simulation performance and apply supervised fine-tuning (SFT) to teach LLMs to emulate both human thinking and actions. Our experiments span two dimensions in leveraging LLMs for user simulation: (1) with or without explicit thinking, and (2) with or without fine-tuning on the thinking-augmented dataset. The results demonstrate the feasibility and potential of incorporating human-like thinking in user simulations, though performance improvements on some metrics remain modest. We believe this exploration provides new avenues and inspirations for advancing user behavior modeling in search simulations. 

---
# LLM4Ranking: An Easy-to-use Framework of Utilizing Large Language Models for Document Reranking 

**Authors**: Qi Liu, Haozhe Duan, Yiqun Chen, Quanfeng Lu, Weiwei Sun, Jiaxin Mao  

**Link**: [PDF](https://arxiv.org/pdf/2504.07439)  

**Abstract**: Utilizing large language models (LLMs) for document reranking has been a popular and promising research direction in recent years, many studies are dedicated to improving the performance and efficiency of using LLMs for reranking. Besides, it can also be applied in many real-world applications, such as search engines or retrieval-augmented generation. In response to the growing demand for research and application in practice, we introduce a unified framework, \textbf{LLM4Ranking}, which enables users to adopt different ranking methods using open-source or closed-source API-based LLMs. Our framework provides a simple and extensible interface for document reranking with LLMs, as well as easy-to-use evaluation and fine-tuning scripts for this task. We conducted experiments based on this framework and evaluated various models and methods on several widely used datasets, providing reproducibility results on utilizing LLMs for document reranking. Our code is publicly available at this https URL. 

---
# Emergency Communication: OTFS-Based Semantic Transmission with Diffusion Noise Suppression 

**Authors**: Kexin Zhang, Xin Zhang, Lixin Li, Wensheng Lin, Wenchi Cheng, Qinghe Du  

**Link**: [PDF](https://arxiv.org/pdf/2504.07420)  

**Abstract**: Due to their flexibility and dynamic coverage capabilities, Unmanned Aerial Vehicles (UAVs) have emerged as vital platforms for emergency communication in disaster-stricken areas. However, the complex channel conditions in high-speed mobile scenarios significantly impact the reliability and efficiency of traditional communication systems. This paper presents an intelligent emergency communication framework that integrates Orthogonal Time Frequency Space (OTFS) modulation, semantic communication, and a diffusion-based denoising module to address these challenges. OTFS ensures robust communication under dynamic channel conditions due to its superior anti-fading characteristics and adaptability to rapidly changing environments. Semantic communication further enhances transmission efficiency by focusing on key information extraction and reducing data redundancy. Moreover, a diffusion-based channel denoising module is proposed to leverage the gradual noise reduction process and statistical noise modeling, optimizing the accuracy of semantic information recovery. Experimental results demonstrate that the proposed solution significantly improves link stability and transmission performance in high-mobility UAV scenarios, achieving at least a 3dB SNR gain over existing methods. 

---
# A Novel Mamba-based Sequential Recommendation Method 

**Authors**: Jun Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2504.07398)  

**Abstract**: Sequential recommendation (SR), which encodes user activity to predict the next action, has emerged as a widely adopted strategy in developing commercial personalized recommendation systems. Although Transformer-based models have proven effective for sequential recommendation, the complexity of the self-attention module in Transformers scales quadratically with the sequence length. Controlling model complexity is essential for large-scale recommendation systems, as these systems may need to handle billion-scale vocabularies that evolve continuously, as well as user behavior sequences that can exceed tens of thousands in length. In this paper, we propose a novel multi-head latent Mamba architecture, which employs multiple low-dimensional Mamba layers and fully connected layers coupled with positional encoding to simultaneously capture historical and item information within each latent subspace. Our proposed method not only enables scaling up to large-scale parameters but also extends to multi-domain recommendation by integrating and fine-tuning LLMs. Through extensive experiments on public datasets, we demonstrate how Hydra effectively addresses the effectiveness-efficiency dilemma, outperforming state-of-the-art sequential recommendation baselines with significantly fewer parameters and reduced training time. 

---
# Towards Distribution Matching between Collaborative and Language Spaces for Generative Recommendation 

**Authors**: Yi Zhang, Yiwen Zhang, Yu Wang, Tong Chen, Hongzhi Yin  

**Link**: [PDF](https://arxiv.org/pdf/2504.07363)  

**Abstract**: Generative recommendation aims to learn the underlying generative process over the entire item set to produce recommendations for users. Although it leverages non-linear probabilistic models to surpass the limited modeling capacity of linear factor models, it is often constrained by a trade-off between representation ability and tractability. With the rise of a new generation of generative methods based on pre-trained language models (LMs), incorporating LMs into general recommendation with implicit feedback has gained considerable attention. However, adapting them to generative recommendation remains challenging. The core reason lies in the mismatch between the input-output formats and semantics of generative models and LMs, making it challenging to achieve optimal alignment in the feature space. This work addresses this issue by proposing a model-agnostic generative recommendation framework called DMRec, which introduces a probabilistic meta-network to bridge the outputs of LMs with user interactions, thereby enabling an equivalent probabilistic modeling process. Subsequently, we design three cross-space distribution matching processes aimed at maximizing shared information while preserving the unique semantics of each space and filtering out irrelevant information. We apply DMRec to three different types of generative recommendation methods and conduct extensive experiments on three public datasets. The experimental results demonstrate that DMRec can effectively enhance the recommendation performance of these generative models, and it shows significant advantages over mainstream LM-enhanced recommendation methods. 

---
# Are AI Agents interacting with Online Ads? 

**Authors**: Andreas Stöckl, Joel Nitu  

**Link**: [PDF](https://arxiv.org/pdf/2504.07112)  

**Abstract**: As AI-driven agents become increasingly integrated into the digital ecosystem, they reshape how online advertising is perceived and processed. Particularly in the travel and hotel booking sector, these autonomous systems influence the effectiveness of traditional advertising formats. While visual cues and emotional appeals sway human users, AI agents prioritize structured data such as price, availability, and specifications. This study examines how different AI agents interact with online advertising, whether they incorporate ads into their decision-making processes, and which ad formats prove most effective. We analyze interaction patterns, click behavior, and decision-making strategies through experiments with multimodal language models such as OpenAI GPT-4o, Anthropic Claude, and Google Gemini 2.0 Flash. Our findings reveal that AI agents neither ignore nor systematically avoid advertisements but instead favor certain features-particularly keywords and structured data. These insights have significant implications for the future design of advertising strategies in AI-dominated digital environments. 

---
# DashCLIP: Leveraging multimodal models for generating semantic embeddings for DoorDash 

**Authors**: Omkar Gurjar, Kin Sum Liu, Praveen Kolli, Utsaw Kumar, Mandar Rahurkar  

**Link**: [PDF](https://arxiv.org/pdf/2504.07110)  

**Abstract**: Despite the success of vision-language models in various generative tasks, obtaining high-quality semantic representations for products and user intents is still challenging due to the inability of off-the-shelf models to capture nuanced relationships between the entities. In this paper, we introduce a joint training framework for product and user queries by aligning uni-modal and multi-modal encoders through contrastive learning on image-text data. Our novel approach trains a query encoder with an LLM-curated relevance dataset, eliminating the reliance on engagement history. These embeddings demonstrate strong generalization capabilities and improve performance across applications, including product categorization and relevance prediction. For personalized ads recommendation, a significant uplift in the click-through rate and conversion rate after the deployment further confirms the impact on key business metrics. We believe that the flexibility of our framework makes it a promising solution toward enriching the user experience across the e-commerce landscape. 

---
# OSCAR: Online Soft Compression And Reranking 

**Authors**: Maxime Louis, Thibault Formal, Hervé Dejean, Stéphane Clinchant  

**Link**: [PDF](https://arxiv.org/pdf/2504.07109)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by integrating external knowledge, leading to improved accuracy and relevance. However, scaling RAG pipelines remains computationally expensive as retrieval sizes grow. To address this, we introduce OSCAR, a novel query-dependent online soft compression method that reduces computational overhead while preserving performance. Unlike traditional hard compression methods, which shorten retrieved texts, or soft compression approaches, which map documents to continuous embeddings offline, OSCAR dynamically compresses retrieved information at inference time, eliminating storage overhead and enabling higher compression rates. Additionally, we extend OSCAR to simultaneously perform reranking, further optimizing the efficiency of the RAG pipeline. Our experiments demonstrate state-of-the-art performance with a 2-5x speed-up in inference and minimal to no loss in accuracy for LLMs ranging from 1B to 24B parameters. The models are available at: this https URL. 

---
# OKRA: an Explainable, Heterogeneous, Multi-Stakeholder Job Recommender System 

**Authors**: Roan Schellingerhout, Francesco Barile, Nava Tintarev  

**Link**: [PDF](https://arxiv.org/pdf/2504.07108)  

**Abstract**: The use of recommender systems in the recruitment domain has been labeled as 'high-risk' in recent legislation. As a result, strict requirements regarding explainability and fairness have been put in place to ensure proper treatment of all involved stakeholders. To allow for stakeholder-specific explainability, while also handling highly heterogeneous recruitment data, we propose a novel explainable multi-stakeholder job recommender system using graph neural networks: the Occupational Knowledge-based Recommender using Attention (OKRA). The proposed method is capable of providing both candidate- and company-side recommendations and explanations. We find that OKRA performs substantially better than six baselines in terms of nDCG for two datasets. Furthermore, we find that the tested models show a bias toward candidates and vacancies located in urban areas. Overall, our findings suggest that OKRA provides a balance between accuracy, explainability, and fairness. 

---
# Guarding Digital Privacy: Exploring User Profiling and Security Enhancements 

**Authors**: Rishika Kohli, Shaifu Gupta, Manoj Singh Gaur  

**Link**: [PDF](https://arxiv.org/pdf/2504.07107)  

**Abstract**: User profiling, the practice of collecting user information for personalized recommendations, has become widespread, driving progress in technology. However, this growth poses a threat to user privacy, as devices often collect sensitive data without their owners' awareness. This article aims to consolidate knowledge on user profiling, exploring various approaches and associated challenges. Through the lens of two companies sharing user data and an analysis of 18 popular Android applications in India across various categories, including $\textit{Social, Education, Entertainment, Travel, Shopping and Others}$, the article unveils privacy vulnerabilities. Further, the article propose an enhanced machine learning framework, employing decision trees and neural networks, that improves state-of-the-art classifiers in detecting personal information exposure. Leveraging the XAI (explainable artificial intelligence) algorithm LIME (Local Interpretable Model-agnostic Explanations), it enhances interpretability, crucial for reliably identifying sensitive data. Results demonstrate a noteworthy performance boost, achieving a $75.01\%$ accuracy with a reduced training time of $3.62$ seconds for neural networks. Concluding, the paper suggests research directions to strengthen digital security measures. 

---
# Business Entity Entropy 

**Authors**: Adam McCabe, Matthew H. Chequers  

**Link**: [PDF](https://arxiv.org/pdf/2504.07106)  

**Abstract**: Organizations generate vast amounts of interconnected content across various platforms. While language models enable sophisticated reasoning for use in business applications, retrieving and contextualizing information from organizational memory remains challenging. We explore this challenge through the lens of entropy, proposing a measure of entity entropy to quantify the distribution of an entity's knowledge across documents as well as a novel generative model inspired by diffusion models in order to provide an explanation for observed behaviours. Empirical analysis on a large-scale enterprise corpus reveals heavy-tailed entropy distributions, a correlation between entity size and entropy, and category-specific entropy patterns. These findings suggest that not all entities are equally retrievable, motivating the need for entity-centric retrieval or pre-processing strategies for a subset of, but not all, entities. We discuss practical implications and theoretical models to guide the design of more efficient knowledge retrieval systems. 

---
# The Feedback Loop Between Recommendation Systems and Reactive Users 

**Authors**: Atefeh Mollabagher, Parinaz Naghizadeh  

**Link**: [PDF](https://arxiv.org/pdf/2504.07105)  

**Abstract**: Recommendation systems underlie a variety of online platforms. These recommendation systems and their users form a feedback loop, wherein the former aims to maximize user engagement through personalization and the promotion of popular content, while the recommendations shape users' opinions or behaviors, potentially influencing future recommendations. These dynamics have been shown to lead to shifts in users' opinions. In this paper, we ask whether reactive users, who are cognizant of the influence of the content they consume, can prevent such changes by actively choosing whether to engage with recommended content. We first model the feedback loop between reactive users' opinion dynamics and a recommendation system. We study these dynamics under three different policies - fixed content consumption (a passive policy), and decreasing or adaptive decreasing content consumption (reactive policies). We analytically show how reactive policies can help users effectively prevent or restrict undesirable opinion shifts, while still deriving utility from consuming content on the platform. We validate and illustrate our theoretical findings through numerical experiments. 

---
# Relevance Isn't All You Need: Scaling RAG Systems With Inference-Time Compute Via Multi-Criteria Reranking 

**Authors**: Will LeVine, Bijan Varjavand  

**Link**: [PDF](https://arxiv.org/pdf/2504.07104)  

**Abstract**: Modern Large Language Model (LLM) systems typically rely on Retrieval Augmented Generation (RAG) which aims to gather context that is useful for response generation. These RAG systems typically optimize strictly towards retrieving context that is maximally relevant to the query. However, conventional theory suggests that retrieval systems which seek to maximize context relevance without any additional explicit criteria can create information bottlenecks. We reaffirm this finding in the modern age of LLM's by showing that in standard RAG pipelines, maximizing for context relevance alone can degrade downstream response quality. In response, we show evaluations of existing RAG methods which account for both context relevance and answer quality. These evaluations introduce a novel finding that existing RAG systems scale poorly with inference time compute usage when considering our combined metric. We introduce "RErank BEyond reLevance (REBEL)", which enables RAG systems to scale with inference-time compute via injection of multi-criteria optimization using Chain-of-Thought prompting (and optionally Multi-Turn dialogue). Ultimately, this enables a new performance/speed tradeoff curve, where RAG systems are able to achieve both higher relevance of retrieved contexts and superior answer quality as inference time increases. Code for the implementation of our method in llama-index can be found at the following PR: this https URL. Code for running experiments using this llama-index implementation can be found at this https URL. 

---
# FG-RAG: Enhancing Query-Focused Summarization with Context-Aware Fine-Grained Graph RAG 

**Authors**: Yubin Hong, Chaofan Li, Jingyi Zhang, Yingxia Shao  

**Link**: [PDF](https://arxiv.org/pdf/2504.07103)  

**Abstract**: Retrieval-Augmented Generation (RAG) enables large language models to provide more precise and pertinent responses by incorporating external knowledge. In the Query-Focused Summarization (QFS) task, GraphRAG-based approaches have notably enhanced the comprehensiveness and diversity of generated responses. However, existing GraphRAG-based approaches predominantly focus on coarse-grained information summarization without being aware of the specific query, and the retrieved content lacks sufficient contextual information to generate comprehensive responses. To address the deficiencies of current RAG systems, we propose Context-Aware Fine-Grained Graph RAG (FG-RAG) to enhance the performance of the QFS task. FG-RAG employs Context-Aware Entity Expansion in graph retrieval to expand the coverage of retrieved entities in the graph, thus providing enough contextual information for the retrieved content. Furthermore, FG-RAG utilizes Query-Level Fine-Grained Summarization to incorporate fine-grained details during response generation, enhancing query awareness for the generated summarization. Our evaluation demonstrates that FG-RAG outperforms other RAG systems in multiple metrics of comprehensiveness, diversity, and empowerment when handling the QFS task. Our implementation is available at this https URL. 

---
# Behavior Importance-Aware Graph Neural Architecture Search for Cross-Domain Recommendation 

**Authors**: Chendi Ge, Xin Wang, Ziwei Zhang, Yijian Qin, Hong Chen, Haiyang Wu, Yang Zhang, Yuekui Yang, Wenwu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2504.07102)  

**Abstract**: Cross-domain recommendation (CDR) mitigates data sparsity and cold-start issues in recommendation systems. While recent CDR approaches using graph neural networks (GNNs) capture complex user-item interactions, they rely on manually designed architectures that are often suboptimal and labor-intensive. Additionally, extracting valuable behavioral information from source domains to improve target domain recommendations remains challenging. To address these challenges, we propose Behavior importance-aware Graph Neural Architecture Search (BiGNAS), a framework that jointly optimizes GNN architecture and data importance for CDR. BiGNAS introduces two key components: a Cross-Domain Customized Supernetwork and a Graph-Based Behavior Importance Perceptron. The supernetwork, as a one-shot, retrain-free module, automatically searches the optimal GNN architecture for each domain without the need for retraining. The perceptron uses auxiliary learning to dynamically assess the importance of source domain behaviors, thereby improving target domain recommendations. Extensive experiments on benchmark CDR datasets and a large-scale industry advertising dataset demonstrate that BiGNAS consistently outperforms state-of-the-art baselines. To the best of our knowledge, this is the first work to jointly optimize GNN architecture and behavior data importance for cross-domain recommendation. 

---
# Personalized Recommendation Models in Federated Settings: A Survey 

**Authors**: Chunxu Zhang, Guodong Long, Zijian Zhang, Zhiwei Li, Honglei Zhang, Qiang Yang, Bo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.07101)  

**Abstract**: Federated recommender systems (FedRecSys) have emerged as a pivotal solution for privacy-aware recommendations, balancing growing demands for data security and personalized experiences. Current research efforts predominantly concentrate on adapting traditional recommendation architectures to federated environments, optimizing communication efficiency, and mitigating security vulnerabilities. However, user personalization modeling, which is essential for capturing heterogeneous preferences in this decentralized and non-IID data setting, remains underexplored. This survey addresses this gap by systematically exploring personalization in FedRecSys, charting its evolution from centralized paradigms to federated-specific innovations. We establish a foundational definition of personalization in a federated setting, emphasizing personalized models as a critical solution for capturing fine-grained user preferences. The work critically examines the technical hurdles of building personalized FedRecSys and synthesizes promising methodologies to meet these challenges. As the first consolidated study in this domain, this survey serves as both a technical reference and a catalyst for advancing personalized FedRecSys research. 

---
# Plan-and-Refine: Diverse and Comprehensive Retrieval-Augmented Generation 

**Authors**: Alireza Salemi, Chris Samarinas, Hamed Zamani  

**Link**: [PDF](https://arxiv.org/pdf/2504.07794)  

**Abstract**: This paper studies the limitations of (retrieval-augmented) large language models (LLMs) in generating diverse and comprehensive responses, and introduces the Plan-and-Refine (P&R) framework based on a two phase system design. In the global exploration phase, P&R generates a diverse set of plans for the given input, where each plan consists of a list of diverse query aspects with corresponding additional descriptions. This phase is followed by a local exploitation phase that generates a response proposal for the input query conditioned on each plan and iteratively refines the proposal for improving the proposal quality. Finally, a reward model is employed to select the proposal with the highest factuality and coverage. We conduct our experiments based on the ICAT evaluation methodology--a recent approach for answer factuality and comprehensiveness evaluation. Experiments on the two diverse information seeking benchmarks adopted from non-factoid question answering and TREC search result diversification tasks demonstrate that P&R significantly outperforms baselines, achieving up to a 13.1% improvement on the ANTIQUE dataset and a 15.41% improvement on the TREC dataset. Furthermore, a smaller scale user study confirms the substantial efficacy of the P&R framework. 

---
# ConceptFormer: Towards Efficient Use of Knowledge-Graph Embeddings in Large Language Models 

**Authors**: Joel Barmettler, Abraham Bernstein, Luca Rossetto  

**Link**: [PDF](https://arxiv.org/pdf/2504.07624)  

**Abstract**: Retrieval Augmented Generation (RAG) has enjoyed increased attention in the recent past and recent advancements in Large Language Models (LLMs) have highlighted the importance of integrating world knowledge into these systems. Current RAG methodologies often modify the internal architecture of pre-trained language models (PLMs) or rely on textifying knowledge graphs (KGs), which is inefficient in terms of token usage. This paper introduces ConceptFormer, a new approach to augment LLMs with structured knowledge from KGs, such as Wikidata, without altering their internal structure or relying on textual input of KGs. ConceptFormer operates in the LLM embedding vector space, creating and injecting \emph{concept vectors} that encapsulate the information of the KG nodes directly. Trained in conjunction with a frozen LLM, ConceptFormer generates a comprehensive lookup table that maps KG nodes to their respective concept vectors. The approach aims to enhance the factual recall capabilities of LLMs by enabling them to process these concept vectors natively, thus enriching them with structured world knowledge in an efficient and scalable manner. Our experiments demonstrate that the addition of concept vectors to GPT-2 0.1B substantially increases its factual recall ability (Hit@10) by up to 272\% when tested on sentences from Wikipedia and up to 348\% on synthetically generated sentences. Even injecting only a single concept vector into the prompt increases factual recall ability (Hit@10) by up to 213\% on Wikipedia sentences, significantly outperforming RAG with graph textification while consuming 130x fewer input tokens. 

---
# Benchmarking Image Embeddings for E-Commerce: Evaluating Off-the Shelf Foundation Models, Fine-Tuning Strategies and Practical Trade-offs 

**Authors**: Urszula Czerwinska, Cenk Bircanoglu, Jeremy Chamoux  

**Link**: [PDF](https://arxiv.org/pdf/2504.07567)  

**Abstract**: We benchmark foundation models image embeddings for classification and retrieval in e-Commerce, evaluating their suitability for real-world applications. Our study spans embeddings from pre-trained convolutional and transformer models trained via supervised, self-supervised, and text-image contrastive learning. We assess full fine-tuning and transfer learning (top-tuning) on six diverse e-Commerce datasets: fashion, consumer goods, cars, food, and retail. Results show full fine-tuning consistently performs well, while text-image and self-supervised embeddings can match its performance with less training. While supervised embeddings remain stable across architectures, SSL and contrastive embeddings vary significantly, often benefiting from top-tuning. Top-tuning emerges as an efficient alternative to full fine-tuning, reducing computational costs. We also explore cross-tuning, noting its impact depends on dataset characteristics. Our findings offer practical guidelines for embedding selection and fine-tuning strategies, balancing efficiency and performance. 

---
