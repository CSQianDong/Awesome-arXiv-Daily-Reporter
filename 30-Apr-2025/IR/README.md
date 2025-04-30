# X-Cross: Dynamic Integration of Language Models for Cross-Domain Sequential Recommendation 

**Authors**: Guy Hadad, Haggai Roitman, Yotam Eshel, Bracha Shapira, Lior Rokach  

**Link**: [PDF](https://arxiv.org/pdf/2504.20859)  

**Abstract**: As new products are emerging daily, recommendation systems are required to quickly adapt to possible new domains without needing extensive retraining. This work presents ``X-Cross'' -- a novel cross-domain sequential-recommendation model that recommends products in new domains by integrating several domain-specific language models; each model is fine-tuned with low-rank adapters (LoRA). Given a recommendation prompt, operating layer by layer, X-Cross dynamically refines the representation of each source language model by integrating knowledge from all other models. These refined representations are propagated from one layer to the next, leveraging the activations from each domain adapter to ensure domain-specific nuances are preserved while enabling adaptability across domains. Using Amazon datasets for sequential recommendation, X-Cross achieves performance comparable to a model that is fine-tuned with LoRA, while using only 25% of the additional parameters. In cross-domain tasks, such as adapting from Toys domain to Tools, Electronics or Sports, X-Cross demonstrates robust performance, while requiring about 50%-75% less fine-tuning data than LoRA to make fine-tuning effective. Furthermore, X-Cross achieves significant improvement in accuracy over alternative cross-domain baselines. Overall, X-Cross enables scalable and adaptive cross-domain recommendations, reducing computational overhead and providing an efficient solution for data-constrained environments. 

---
# RecGaze: The First Eye Tracking and User Interaction Dataset for Carousel Interfaces 

**Authors**: Santiago de Leon-Martinez, Jingwei Kang, Robert Moro, Maarten de Rijke, Branislav Kveton, Harrie Oosterhuis, Maria Bielikova  

**Link**: [PDF](https://arxiv.org/pdf/2504.20792)  

**Abstract**: Carousel interfaces are widely used in e-commerce and streaming services, but little research has been devoted to them. Previous studies of interfaces for presenting search and recommendation results have focused on single ranked lists, but it appears their results cannot be extrapolated to carousels due to the added complexity. Eye tracking is a highly informative approach to understanding how users click, yet there are no eye tracking studies concerning carousels. There are very few interaction datasets on recommenders with carousel interfaces and none that contain gaze data.
We introduce the RecGaze dataset: the first comprehensive feedback dataset on carousels that includes eye tracking results, clicks, cursor movements, and selection explanations. The dataset comprises of interactions from 3 movie selection tasks with 40 different carousel interfaces per user. In total, 87 users and 3,477 interactions are logged. In addition to the dataset, its description and possible use cases, we provide results of a survey on carousel design and the first analysis of gaze data on carousels, which reveals a golden triangle or F-pattern browsing behavior.
Our work seeks to advance the field of carousel interfaces by providing the first dataset with eye tracking results on carousels. In this manner, we provide and encourage an empirical understanding of interactions with carousel interfaces, for building better recommender systems through gaze information, and also encourage the development of gaze-based recommenders. 

---
# Information Retrieval in the Age of Generative AI: The RGB Model 

**Authors**: Michele Garetto, Alessandro Cornacchia, Franco Galante, Emilio Leonardi, Alessandro Nordio, Alberto Tarable  

**Link**: [PDF](https://arxiv.org/pdf/2504.20610)  

**Abstract**: The advent of Large Language Models (LLMs) and generative AI is fundamentally transforming information retrieval and processing on the Internet, bringing both great potential and significant concerns regarding content authenticity and reliability. This paper presents a novel quantitative approach to shed light on the complex information dynamics arising from the growing use of generative AI tools. Despite their significant impact on the digital ecosystem, these dynamics remain largely uncharted and poorly understood. We propose a stochastic model to characterize the generation, indexing, and dissemination of information in response to new topics. This scenario particularly challenges current LLMs, which often rely on real-time Retrieval-Augmented Generation (RAG) techniques to overcome their static knowledge limitations. Our findings suggest that the rapid pace of generative AI adoption, combined with increasing user reliance, can outpace human verification, escalating the risk of inaccurate information proliferation across digital resources. An in-depth analysis of Stack Exchange data confirms that high-quality answers inevitably require substantial time and human effort to emerge. This underscores the considerable risks associated with generating persuasive text in response to new questions and highlights the critical need for responsible development and deployment of future generative AI tools. 

---
# Natural Language Processing tools for Pharmaceutical Manufacturing Information Extraction from Patents Natural Language Processing (NLP) tools for Pharmaceutical Manufacturing Information Extraction from Patents 

**Authors**: Diego Alvarado-Maldonado, Blair Johnston, Cameron J. Brown  

**Link**: [PDF](https://arxiv.org/pdf/2504.20598)  

**Abstract**: Abundant and diverse data on medicines manufacturing and other lifecycle components has been made easily accessible in the last decades. However, a significant proportion of this information is characterised by not being tabulated and usable for machine learning purposes. Thus, natural language processing tools have been used to build databases in domains such as biomedical and chemical to address this limitation. This has allowed the development of artificial intelligence applications, which have improved drug discovery and treatments. In the pharmaceutical manufacturing context, some initiatives and datasets for primary processing can be found, but the manufacturing of drug products is an area which is still lacking, to the best of our knowledge. This works aims to explore and adapt NLP tools used in other domains to extract information on both primary and secondary manufacturing, employing patents as the main source of data. Thus, two independent, but complementary, models were developed comprising a method to select fragments of text that contain manufacturing data, and a named entity recognition system that enables extracting information on operations, materials, and conditions of a process. For the first model, the identification of relevant sections was achieved using an unsupervised approach combining Latent Dirichlet Allocation and k-Means clustering. The performance of this model measured as a Cohen's kappa between model output and manual revision was higher than 90%. NER model consisted of a deep neural network, and an f1-score micro average of 84.2% was obtained which is comparable to other works. Some considerations for these tools to be used in data extraction are discussed throughout this document. 

---
# Search-Based Interaction For Conversation Recommendation via Generative Reward Model Based Simulated User 

**Authors**: Xiaolei Wang, Chunxuan Xia, Junyi Li, Fanzhe Meng, Lei Huang, Jinpeng Wang, Wayne Xin Zhao, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2504.20458)  

**Abstract**: Conversational recommendation systems (CRSs) use multi-turn interaction to capture user preferences and provide personalized recommendations. A fundamental challenge in CRSs lies in effectively understanding user preferences from conversations. User preferences can be multifaceted and complex, posing significant challenges for accurate recommendations even with access to abundant external knowledge. While interaction with users can clarify their true preferences, frequent user involvement can lead to a degraded user experience.
To address this problem, we propose a generative reward model based simulated user, named GRSU, for automatic interaction with CRSs. The simulated user provides feedback to the items recommended by CRSs, enabling them to better capture intricate user preferences through multi-turn interaction. Inspired by generative reward models, we design two types of feedback actions for the simulated user: i.e., generative item scoring, which offers coarse-grained feedback, and attribute-based item critique, which provides fine-grained feedback. To ensure seamless integration, these feedback actions are unified into an instruction-based format, allowing the development of a unified simulated user via instruction tuning on synthesized data. With this simulated user, automatic multi-turn interaction with CRSs can be effectively conducted. Furthermore, to strike a balance between effectiveness and efficiency, we draw inspiration from the paradigm of reward-guided search in complex reasoning tasks and employ beam search for the interaction process. On top of this, we propose an efficient candidate ranking method to improve the recommendation results derived from interaction. Extensive experiments on public datasets demonstrate the effectiveness, efficiency, and transferability of our approach. 

---
# Enhancing News Recommendation with Hierarchical LLM Prompting 

**Authors**: Hai-Dang Kieu, Delvin Ce Zhang, Minh Duc Nguyen, Min Xu, Qiang Wu, Dung D. Le  

**Link**: [PDF](https://arxiv.org/pdf/2504.20452)  

**Abstract**: Personalized news recommendation systems often struggle to effectively capture the complexity of user preferences, as they rely heavily on shallow representations, such as article titles and abstracts. To address this problem, we introduce a novel method, namely PNR-LLM, for Large Language Models for Personalized News Recommendation. Specifically, PNR-LLM harnesses the generation capabilities of LLMs to enrich news titles and abstracts, and consequently improves recommendation quality. PNR-LLM contains a novel module, News Enrichment via LLMs, which generates deeper semantic information and relevant entities from articles, transforming shallow contents into richer representations. We further propose an attention mechanism to aggregate enriched semantic- and entity-level data, forming unified user and news embeddings that reveal a more accurate user-news match. Extensive experiments on MIND datasets show that PNR-LLM outperforms state-of-the-art baselines. Moreover, the proposed data enrichment module is model-agnostic, and we empirically show that applying our proposed module to multiple existing models can further improve their performance, verifying the advantage of our design. 

---
# Can LLMs Be Trusted for Evaluating RAG Systems? A Survey of Methods and Datasets 

**Authors**: Lorenz Brehme, Thomas Ströhle, Ruth Breu  

**Link**: [PDF](https://arxiv.org/pdf/2504.20119)  

**Abstract**: Retrieval-Augmented Generation (RAG) has advanced significantly in recent years. The complexity of RAG systems, which involve multiple components-such as indexing, retrieval, and generation-along with numerous other parameters, poses substantial challenges for systematic evaluation and quality enhancement. Previous research highlights that evaluating RAG systems is essential for documenting advancements, comparing configurations, and identifying effective approaches for domain-specific applications. This study systematically reviews 63 academic articles to provide a comprehensive overview of state-of-the-art RAG evaluation methodologies, focusing on four key areas: datasets, retrievers, indexing and databases, and the generator component. We observe the feasibility of an automated evaluation approach for each component of a RAG system, leveraging an LLM capable of both generating evaluation datasets and conducting evaluations. In addition, we found that further practical research is essential to provide companies with clear guidance on the do's and don'ts of implementing and evaluating RAG systems. By synthesizing evaluation approaches for key RAG components and emphasizing the creation and adaptation of domain-specific datasets for benchmarking, we contribute to the advancement of systematic evaluation methods and the improvement of evaluation rigor for RAG systems. Furthermore, by examining the interplay between automated approaches leveraging LLMs and human judgment, we contribute to the ongoing discourse on balancing automation and human input, clarifying their respective contributions, limitations, and challenges in achieving robust and reliable evaluations. 

---
# OpenTCM: A GraphRAG-Empowered LLM-based System for Traditional Chinese Medicine Knowledge Retrieval and Diagnosis 

**Authors**: Jinglin He, Yunqi Guo, Lai Kwan Lam, Waikei Leung, Lixing He, Yuanan Jiang, Chi Chiu Wang, Guoliang Xing, Hongkai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.20118)  

**Abstract**: Traditional Chinese Medicine (TCM) represents a rich repository of ancient medical knowledge that continues to play an important role in modern healthcare. Due to the complexity and breadth of the TCM literature, the integration of AI technologies is critical for its modernization and broader accessibility. However, this integration poses considerable challenges, including the interpretation of obscure classical Chinese texts and the modeling of intricate semantic relationships among TCM concepts. In this paper, we develop OpenTCM, an LLM-based system that combines a domain-specific TCM knowledge graph and Graph-based Retrieval-Augmented Generation (GraphRAG). First, we extract more than 3.73 million classical Chinese characters from 68 gynecological books in the Chinese Medical Classics Database, with the help of TCM and gynecology experts. Second, we construct a comprehensive multi-relational knowledge graph comprising more than 48,000 entities and 152,000 interrelationships, using customized prompts and Chinese-oriented LLMs such as DeepSeek and Kimi to ensure high-fidelity semantic understanding. Last, we integrate OpenTCM with this knowledge graph, enabling high-fidelity ingredient knowledge retrieval and diagnostic question-answering without model fine-tuning. Experimental evaluations demonstrate that our prompt design and model selection significantly improve knowledge graph quality, achieving a precision of 98. 55% and an F1 score of 99. 55%. In addition, OpenTCM achieves mean expert scores of 4.5 in ingredient information retrieval and 3.8 in diagnostic question-answering tasks, outperforming state-of-the-art solutions in real-world TCM use cases. 

---
# TreeHop: Generate and Filter Next Query Embeddings Efficiently for Multi-hop Question Answering 

**Authors**: Zhonghao Li, Kunpeng Zhang, Jinghuai Ou, Shuliang Liu, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2504.20114)  

**Abstract**: Retrieval-augmented generation (RAG) systems face significant challenges in multi-hop question answering (MHQA), where complex queries require synthesizing information across multiple document chunks. Existing approaches typically rely on iterative LLM-based query rewriting and routing, resulting in high computational costs due to repeated LLM invocations and multi-stage processes. To address these limitations, we propose TreeHop, an embedding-level framework without the need for LLMs in query refinement. TreeHop dynamically updates query embeddings by fusing semantic information from prior queries and retrieved documents, enabling iterative retrieval through embedding-space operations alone. This method replaces the traditional "Retrieve-Rewrite-Vectorize-Retrieve" cycle with a streamlined "Retrieve-Embed-Retrieve" loop, significantly reducing computational overhead. Moreover, a rule-based stop criterion is introduced to further prune redundant retrievals, balancing efficiency and recall rate. Experimental results show that TreeHop rivals advanced RAG methods across three open-domain MHQA datasets, achieving comparable performance with only 5\%-0.4\% of the model parameter size and reducing the query latency by approximately 99\% compared to concurrent approaches. This makes TreeHop a faster and more cost-effective solution for deployment in a range of knowledge-intensive applications. For reproducibility purposes, codes and data are available here: this https URL. 

---
# MATCHA: Can Multi-Agent Collaboration Build a Trustworthy Conversational Recommender? 

**Authors**: Zheng Hui, Xiaokai Wei, Yexi Jiang, Kevin Gao, Chen Wang, Frank Ong, Se-eun Yoon, Rachit Pareek, Michelle Gong  

**Link**: [PDF](https://arxiv.org/pdf/2504.20094)  

**Abstract**: In this paper, we propose a multi-agent collaboration framework called MATCHA for conversational recommendation system, leveraging large language models (LLMs) to enhance personalization and user engagement. Users can request recommendations via free-form text and receive curated lists aligned with their interests, preferences, and constraints. Our system introduces specialized agents for intent analysis, candidate generation, ranking, re-ranking, explainability, and safeguards. These agents collaboratively improve recommendations accuracy, diversity, and safety. On eight metrics, our model achieves superior or comparable performance to the current state-of-the-art. Through comparisons with six baseline models, our approach addresses key challenges in conversational recommendation systems for game recommendations, including: (1) handling complex, user-specific requests, (2) enhancing personalization through multi-agent collaboration, (3) empirical evaluation and deployment, and (4) ensuring safe and trustworthy interactions. 

---
# An Integrated Framework for Contextual Personalized LLM-Based Food Recommendation 

**Authors**: Ali Rostami  

**Link**: [PDF](https://arxiv.org/pdf/2504.20092)  

**Abstract**: Personalized food recommendation systems (Food-RecSys) critically underperform due to fragmented component understanding and the failure of conventional machine learning with vast, imbalanced food data. While Large Language Models (LLMs) offer promise, current generic Recommendation as Language Processing (RLP) strategies lack the necessary specialization for the food domain's complexity. This thesis tackles these deficiencies by first identifying and analyzing the essential components for effective Food-RecSys. We introduce two key innovations: a multimedia food logging platform for rich contextual data acquisition and the World Food Atlas, enabling unique geolocation-based food analysis previously unavailable. Building on this foundation, we pioneer the Food Recommendation as Language Processing (F-RLP) framework - a novel, integrated approach specifically architected for the food domain. F-RLP leverages LLMs in a tailored manner, overcoming the limitations of generic models and providing a robust infrastructure for effective, contextual, and truly personalized food recommendations. 

---
# A model and package for German ColBERT 

**Authors**: Thuong Dang, Qiqi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.20083)  

**Abstract**: In this work, we introduce a German version for ColBERT, a late interaction multi-dense vector retrieval method, with a focus on RAG applications. We also present the main features of our package for ColBERT models, supporting both retrieval and fine-tuning workflows. 

---
# Against Opacity: Explainable AI and Large Language Models for Effective Digital Advertising 

**Authors**: Qi Yang, Marlo Ongpin, Sergey Nikolenko, Alfred Huang, Aleksandr Farseev  

**Link**: [PDF](https://arxiv.org/pdf/2504.20064)  

**Abstract**: The opaqueness of modern digital advertising, exemplified by platforms such as Meta Ads, raises concerns regarding their autonomous control over audience targeting, pricing structures, and ad relevancy assessments. Locked in their leading positions by network effects, ``Metas and Googles of the world'' attract countless advertisers who rely on intuition, with billions of dollars lost on ineffective social media ads. The platforms' algorithms use huge amounts of data unavailable to advertisers, and the algorithms themselves are opaque as well. This lack of transparency hinders the advertisers' ability to make informed decisions and necessitates efforts to promote transparency, standardize industry metrics, and strengthen regulatory frameworks. In this work, we propose novel ways to assist marketers in optimizing their advertising strategies via machine learning techniques designed to analyze and evaluate content, in particular, predict the click-through rates (CTR) of novel advertising content. Another important problem is that large volumes of data available in the competitive landscape, e.g., competitors' ads, impede the ability of marketers to derive meaningful insights. This leads to a pressing need for a novel approach that would allow us to summarize and comprehend complex data. Inspired by the success of ChatGPT in bridging the gap between large language models (LLMs) and a broader non-technical audience, we propose a novel system that facilitates marketers in data interpretation, called SODA, that merges LLMs with explainable AI, enabling better human-AI collaboration with an emphasis on the domain of digital marketing and advertising. By combining LLMs and explainability features, in particular modern text-image models, we aim to improve the synergy between human marketers and AI systems. 

---
# Recommending Clinical Trials for Online Patient Cases using Artificial Intelligence 

**Authors**: Joey Chan, Qiao Jin, Nicholas Wan, Charalampos S. Floudas, Elisabetta Xue, Zhiyong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2504.20059)  

**Abstract**: Clinical trials are crucial for assessing new treatments; however, recruitment challenges - such as limited awareness, complex eligibility criteria, and referral barriers - hinder their success. With the growth of online platforms, patients increasingly turn to social media and health communities for support, research, and advocacy, expanding recruitment pools and established enrollment pathways. Recognizing this potential, we utilized TrialGPT, a framework that leverages a large language model (LLM) as its backbone, to match 50 online patient cases (collected from published case reports and a social media website) to clinical trials and evaluate performance against traditional keyword-based searches. Our results show that TrialGPT outperforms traditional methods by 46% in identifying eligible trials, with each patient, on average, being eligible for around 7 trials. Additionally, our outreach efforts to case authors and trial organizers regarding these patient-trial matches yielded highly positive feedback, which we present from both perspectives. 

---
# HCT-QA: A Benchmark for Question Answering on Human-Centric Tables 

**Authors**: Mohammad S. Ahmad, Zan A. Naeem, Michaël Aupetit, Ahmed Elmagarmid, Mohamed Eltabakh, Xiasong Ma, Mourad Ouzzani, Chaoyi Ruan  

**Link**: [PDF](https://arxiv.org/pdf/2504.20047)  

**Abstract**: Tabular data embedded within PDF files, web pages, and other document formats are prevalent across numerous sectors such as government, engineering, science, and business. These human-centric tables (HCTs) possess a unique combination of high business value, intricate layouts, limited operational power at scale, and sometimes serve as the only data source for critical insights. However, their complexity poses significant challenges to traditional data extraction, processing, and querying methods. While current solutions focus on transforming these tables into relational formats for SQL queries, they fall short in handling the diverse and complex layouts of HCTs and hence being amenable to querying. This paper describes HCT-QA, an extensive benchmark of HCTs, natural language queries, and related answers on thousands of tables. Our dataset includes 2,188 real-world HCTs with 9,835 QA pairs and 4,679 synthetic tables with 67.5K QA pairs. While HCTs can be potentially processed by different type of query engines, in this paper, we focus on Large Language Models as potential engines and assess their ability in processing and querying such tables. 

---
# CBM-RAG: Demonstrating Enhanced Interpretability in Radiology Report Generation with Multi-Agent RAG and Concept Bottleneck Models 

**Authors**: Hasan Md Tusfiqur Alam, Devansh Srivastav, Abdulrahman Mohamed Selim, Md Abdul Kadir, Md Moktadiurl Hoque Shuvo, Daniel Sonntag  

**Link**: [PDF](https://arxiv.org/pdf/2504.20898)  

**Abstract**: Advancements in generative Artificial Intelligence (AI) hold great promise for automating radiology workflows, yet challenges in interpretability and reliability hinder clinical adoption. This paper presents an automated radiology report generation framework that combines Concept Bottleneck Models (CBMs) with a Multi-Agent Retrieval-Augmented Generation (RAG) system to bridge AI performance with clinical explainability. CBMs map chest X-ray features to human-understandable clinical concepts, enabling transparent disease classification. Meanwhile, the RAG system integrates multi-agent collaboration and external knowledge to produce contextually rich, evidence-based reports. Our demonstration showcases the system's ability to deliver interpretable predictions, mitigate hallucinations, and generate high-quality, tailored reports with an interactive interface addressing accuracy, trust, and usability challenges. This framework provides a pathway to improving diagnostic consistency and empowering radiologists with actionable insights. 

---
# UniversalRAG: Retrieval-Augmented Generation over Multiple Corpora with Diverse Modalities and Granularities 

**Authors**: Woongyeong Yeo, Kangsan Kim, Soyeong Jeong, Jinheon Baek, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2504.20734)  

**Abstract**: Retrieval-Augmented Generation (RAG) has shown substantial promise in improving factual accuracy by grounding model responses with external knowledge relevant to queries. However, most existing RAG approaches are limited to a text-only corpus, and while recent efforts have extended RAG to other modalities such as images and videos, they typically operate over a single modality-specific corpus. In contrast, real-world queries vary widely in the type of knowledge they require, which a single type of knowledge source cannot address. To address this, we introduce UniversalRAG, a novel RAG framework designed to retrieve and integrate knowledge from heterogeneous sources with diverse modalities and granularities. Specifically, motivated by the observation that forcing all modalities into a unified representation space derived from a single combined corpus causes a modality gap, where the retrieval tends to favor items from the same modality as the query, we propose a modality-aware routing mechanism that dynamically identifies the most appropriate modality-specific corpus and performs targeted retrieval within it. Also, beyond modality, we organize each modality into multiple granularity levels, enabling fine-tuned retrieval tailored to the complexity and scope of the query. We validate UniversalRAG on 8 benchmarks spanning multiple modalities, showing its superiority over modality-specific and unified baselines. 

---
# Are Information Retrieval Approaches Good at Harmonising Longitudinal Survey Questions in Social Science? 

**Authors**: Wing Yan Li, Zeqiang Wang, Jon Johnson, Suparna De  

**Link**: [PDF](https://arxiv.org/pdf/2504.20679)  

**Abstract**: Automated detection of semantically equivalent questions in longitudinal social science surveys is crucial for long-term studies informing empirical research in the social, economic, and health sciences. Retrieving equivalent questions faces dual challenges: inconsistent representation of theoretical constructs (i.e. concept/sub-concept) across studies as well as between question and response options, and the evolution of vocabulary and structure in longitudinal text. To address these challenges, our multi-disciplinary collaboration of computer scientists and survey specialists presents a new information retrieval (IR) task of identifying concept (e.g. Housing, Job, etc.) equivalence across question and response options to harmonise longitudinal population studies. This paper investigates multiple unsupervised approaches on a survey dataset spanning 1946-2020, including probabilistic models, linear probing of language models, and pre-trained neural networks specialised for IR. We show that IR-specialised neural models achieve the highest overall performance with other approaches performing comparably. Additionally, the re-ranking of the probabilistic model's results with neural models only introduces modest improvements of 0.07 at most in F1-score. Qualitative post-hoc evaluation by survey specialists shows that models generally have a low sensitivity to questions with high lexical overlap, particularly in cases where sub-concepts are mismatched. Altogether, our analysis serves to further research on harmonising longitudinal studies in social science. 

---
# ReasonIR: Training Retrievers for Reasoning Tasks 

**Authors**: Rulin Shao, Rui Qiao, Varsha Kishore, Niklas Muennighoff, Xi Victoria Lin, Daniela Rus, Bryan Kian Hsiang Low, Sewon Min, Wen-tau Yih, Pang Wei Koh, Luke Zettlemoyer  

**Link**: [PDF](https://arxiv.org/pdf/2504.20595)  

**Abstract**: We present ReasonIR-8B, the first retriever specifically trained for general reasoning tasks. Existing retrievers have shown limited gains on reasoning tasks, in part because existing training datasets focus on short factual queries tied to documents that straightforwardly answer them. We develop a synthetic data generation pipeline that, for each document, our pipeline creates a challenging and relevant query, along with a plausibly related but ultimately unhelpful hard negative. By training on a mixture of our synthetic data and existing public data, ReasonIR-8B achieves a new state-of-the-art of 29.9 nDCG@10 without reranker and 36.9 nDCG@10 with reranker on BRIGHT, a widely-used reasoning-intensive information retrieval (IR) benchmark. When applied to RAG tasks, ReasonIR-8B improves MMLU and GPQA performance by 6.4% and 22.6% respectively, relative to the closed-book baseline, outperforming other retrievers and search engines. In addition, ReasonIR-8B uses test-time compute more effectively: on BRIGHT, its performance consistently increases with longer and more information-rich rewritten queries; it continues to outperform other retrievers when combined with an LLM reranker. Our training recipe is general and can be easily extended to future LLMs; to this end, we open-source our code, data, and model. 

---
# Team ACK at SemEval-2025 Task 2: Beyond Word-for-Word Machine Translation for English-Korean Pairs 

**Authors**: Daniel Lee, Harsh Sharma, Jieun Han, Sunny Jeong, Alice Oh, Vered Shwartz  

**Link**: [PDF](https://arxiv.org/pdf/2504.20451)  

**Abstract**: Translating knowledge-intensive and entity-rich text between English and Korean requires transcreation to preserve language-specific and cultural nuances beyond literal, phonetic or word-for-word conversion. We evaluate 13 models (LLMs and MT models) using automatic metrics and human assessment by bilingual annotators. Our findings show LLMs outperform traditional MT systems but struggle with entity translation requiring cultural adaptation. By constructing an error taxonomy, we identify incorrect responses and entity name errors as key issues, with performance varying by entity type and popularity level. This work exposes gaps in automatic evaluation metrics and hope to enable future work in completing culturally-nuanced machine translation. 

---
# Labeling Case Similarity based on Co-Citation of Legal Articles in Judgment Documents with Empirical Dispute-Based Evaluation 

**Authors**: Chao-Lin Liu, Po-Hsien Wu, Yi-Ting Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.20323)  

**Abstract**: This report addresses the challenge of limited labeled datasets for developing legal recommender systems, particularly in specialized domains like labor disputes. We propose a new approach leveraging the co-citation of legal articles within cases to establish similarity and enable algorithmic annotation. This method draws a parallel to the concept of case co-citation, utilizing cited precedents as indicators of shared legal issues. To evaluate the labeled results, we employ a system that recommends similar cases based on plaintiffs' accusations, defendants' rebuttals, and points of disputes. The evaluation demonstrates that the recommender, with finetuned text embedding models and a reasonable BiLSTM module can recommend labor cases whose similarity was measured by the co-citation of the legal articles. This research contributes to the development of automated annotation techniques for legal documents, particularly in areas with limited access to comprehensive legal databases. 

---
# Spark: A System for Scientifically Creative Idea Generation 

**Authors**: Aishik Sanyal, Samuel Schapiro, Sumuk Shashidhar, Royce Moon, Lav R. Varshney, Dilek Hakkani-Tur  

**Link**: [PDF](https://arxiv.org/pdf/2504.20090)  

**Abstract**: Recently, large language models (LLMs) have shown promising abilities to generate novel research ideas in science, a direction which coincides with many foundational principles in computational creativity (CC). In light of these developments, we present an idea generation system named Spark that couples retrieval-augmented idea generation using LLMs with a reviewer model named Judge trained on 600K scientific reviews from OpenReview. Our work is both a system demonstration and intended to inspire other CC researchers to explore grounding the generation and evaluation of scientific ideas within foundational CC principles. To this end, we release the annotated dataset used to train Judge, inviting other researchers to explore the use of LLMs for idea generation and creative evaluations. 

---
# Refiner: Restructure Retrieval Content Efficiently to Advance Question-Answering Capabilities 

**Authors**: Zhonghao Li, Xuming Hu, Aiwei Liu, Kening Zheng, Sirui Huang, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2406.11357)  

**Abstract**: Large Language Models (LLMs) are limited by their parametric knowledge, leading to hallucinations in knowledge-extensive tasks. To address this, Retrieval-Augmented Generation (RAG) incorporates external document chunks to expand LLM knowledge. Furthermore, compressing information from document chunks through extraction or summarization can improve LLM performance. Nonetheless, LLMs still struggle to notice and utilize scattered key information, a problem known as the "lost-in-the-middle" syndrome. Therefore, we typically need to restructure the content for LLM to recognize the key information. We propose $\textit{Refiner}$, an end-to-end extract-and-restructure paradigm that operates in the post-retrieval process of RAG. $\textit{Refiner}$ leverages a single decoder-only LLM to adaptively extract query-relevant contents verbatim along with the necessary context, and section them based on their interconnectedness, thereby highlights information distinction, and aligns downstream LLMs with the original context effectively. Experiments show that a trained $\textit{Refiner}$ (with 7B parameters) exhibits significant gain to downstream LLM in improving answer accuracy, and outperforms other state-of-the-art advanced RAG and concurrent compressing approaches in various single-hop and multi-hop QA tasks. Notably, $\textit{Refiner}$ achieves a 80.5% tokens reduction and a 1.6-7.0% improvement margin in multi-hop tasks compared to the next best solution. $\textit{Refiner}$ is a plug-and-play solution that can be seamlessly integrated with RAG systems, facilitating its application across diverse open-source frameworks. 

---
