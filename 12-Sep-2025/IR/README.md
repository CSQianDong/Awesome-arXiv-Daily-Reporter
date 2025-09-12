# Retrieval-Augmented Generation for Reliable Interpretation of Radio Regulations 

**Authors**: Zakaria El Kassimi, Fares Fourati, Mohamed-Slim Alouini  

**Link**: [PDF](https://arxiv.org/pdf/2509.09651)  

**Abstract**: We study question answering in the domain of radio regulations, a legally sensitive and high-stakes area. We propose a telecom-specific Retrieval-Augmented Generation (RAG) pipeline and introduce, to our knowledge, the first multiple-choice evaluation set for this domain, constructed from authoritative sources using automated filtering and human validation. To assess retrieval quality, we define a domain-specific retrieval metric, under which our retriever achieves approximately 97% accuracy. Beyond retrieval, our approach consistently improves generation accuracy across all tested models. In particular, while naively inserting documents without structured retrieval yields only marginal gains for GPT-4o (less than 1%), applying our pipeline results in nearly a 12% relative improvement. These findings demonstrate that carefully targeted grounding provides a simple yet strong baseline and an effective domain-specific solution for regulatory question answering. All code and evaluation scripts, along with our derived question-answer dataset, are available at this https URL. 

---
# AskDoc -- Identifying Hidden Healthcare Disparities 

**Authors**: Shashank Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2509.09622)  

**Abstract**: The objective of this study is to understand the online Ask the Doctor services medical advice on internet platforms via AskDoc, a Reddit community that serves as a public AtD platform and study if platforms mirror existing hurdles and partiality in healthcare across various demographic groups. We downloaded data from January 2020 to May 2022 from AskDoc -- a subreddit, and created regular expressions to identify self-reported demographics (Gender, Race, and Age) from the posts, and performed statistical analysis to understand the interaction between peers and physicians with the posters. Half of the posts did not receive comments from peers or physicians. At least 90% of the people disclose their gender and age, and 80% of the people do not disclose their race. It was observed that the subreddit is dominated by adult (age group 20-39) white males. Some disparities were observed in the engagement between the users and the posters with certain demographics. Beyond the confines of clinics and hospitals, social media could bring patients and providers closer together, however, as observed, current physicians participation is low compared to posters. 

---
# Boosting Data Utilization for Multilingual Dense Retrieval 

**Authors**: Chao Huang, Fengran Mo, Yufeng Chen, Changhao Guan, Zhenrui Yue, Xinyu Wang, Jinan Xu, Kaiyu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09459)  

**Abstract**: Multilingual dense retrieval aims to retrieve relevant documents across different languages based on a unified retriever model. The challenge lies in aligning representations of different languages in a shared vector space. The common practice is to fine-tune the dense retriever via contrastive learning, whose effectiveness highly relies on the quality of the negative sample and the efficacy of mini-batch data. Different from the existing studies that focus on developing sophisticated model architecture, we propose a method to boost data utilization for multilingual dense retrieval by obtaining high-quality hard negative samples and effective mini-batch data. The extensive experimental results on a multilingual retrieval benchmark, MIRACL, with 16 languages demonstrate the effectiveness of our method by outperforming several existing strong baselines. 

---
# We're Still Doing It (All) Wrong: Recommender Systems, Fifteen Years Later 

**Authors**: Alan Said, Maria Soledad Pera, Michael D. Ekstrand  

**Link**: [PDF](https://arxiv.org/pdf/2509.09414)  

**Abstract**: In 2011, Xavier Amatriain sounded the alarm: recommender systems research was "doing it all wrong" [1]. His critique, rooted in statistical misinterpretation and methodological shortcuts, remains as relevant today as it was then. But rather than correcting course, we added new layers of sophistication on top of the same broken foundations. This paper revisits Amatriain's diagnosis and argues that many of the conceptual, epistemological, and infrastructural failures he identified still persist, in more subtle or systemic forms. Drawing on recent work in reproducibility, evaluation methodology, environmental impact, and participatory design, we showcase how the field's accelerating complexity has outpaced its introspection. We highlight ongoing community-led initiatives that attempt to shift the paradigm, including workshops, evaluation frameworks, and calls for value-sensitive and participatory research. At the same time, we contend that meaningful change will require not only new metrics or better tooling, but a fundamental reframing of what recommender systems research is for, who it serves, and how knowledge is produced and validated. Our call is not just for technical reform, but for a recommender systems research agenda grounded in epistemic humility, human impact, and sustainable practice. 

---
# CESRec: Constructing Pseudo Interactions for Sequential Recommendation via Conversational Feedback 

**Authors**: Yifan Wang, Shen Gao, Jiabao Fang, Rui Yan, Billy Chiu, Shuo Shang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09342)  

**Abstract**: Sequential Recommendation Systems (SRS) have become essential in many real-world applications. However, existing SRS methods often rely on collaborative filtering signals and fail to capture real-time user preferences, while Conversational Recommendation Systems (CRS) excel at eliciting immediate interests through natural language interactions but neglect historical behavior. To bridge this gap, we propose CESRec, a novel framework that integrates the long-term preference modeling of SRS with the real-time preference elicitation of CRS. We introduce semantic-based pseudo interaction construction, which dynamically updates users'historical interaction sequences by analyzing conversational feedback, generating a pseudo-interaction sequence that seamlessly combines long-term and real-time preferences. Additionally, we reduce the impact of outliers in historical items that deviate from users'core preferences by proposing dual alignment outlier items masking, which identifies and masks such items using semantic-collaborative aligned representations. Extensive experiments demonstrate that CESRec achieves state-of-the-art performance by boosting strong SRS models, validating its effectiveness in integrating conversational feedback into SRS. 

---
# Modality Alignment with Multi-scale Bilateral Attention for Multimodal Recommendation 

**Authors**: Kelin Ren, Chan-Yang Ju, Dong-Ho Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.09114)  

**Abstract**: Multimodal recommendation systems are increasingly becoming foundational technologies for e-commerce and content platforms, enabling personalized services by jointly modeling users' historical behaviors and the multimodal features of items (e.g., visual and textual). However, most existing methods rely on either static fusion strategies or graph-based local interaction modeling, facing two critical limitations: (1) insufficient ability to model fine-grained cross-modal associations, leading to suboptimal fusion quality; and (2) a lack of global distribution-level consistency, causing representational bias. To address these, we propose MambaRec, a novel framework that integrates local feature alignment and global distribution regularization via attention-guided learning. At its core, we introduce the Dilated Refinement Attention Module (DREAM), which uses multi-scale dilated convolutions with channel-wise and spatial attention to align fine-grained semantic patterns between visual and textual modalities. This module captures hierarchical relationships and context-aware associations, improving cross-modal semantic modeling. Additionally, we apply Maximum Mean Discrepancy (MMD) and contrastive loss functions to constrain global modality alignment, enhancing semantic consistency. This dual regularization reduces mode-specific deviations and boosts robustness. To improve scalability, MambaRec employs a dimensionality reduction strategy to lower the computational cost of high-dimensional multimodal features. Extensive experiments on real-world e-commerce datasets show that MambaRec outperforms existing methods in fusion quality, generalization, and efficiency. Our code has been made publicly available at this https URL. 

---
# Envy-Free but Still Unfair: Envy-Freeness Up To One Item (EF-1) in Personalized Recommendation 

**Authors**: Amanda Aird, Ben Armstrong, Nicholas Mattei, Robin Burke  

**Link**: [PDF](https://arxiv.org/pdf/2509.09037)  

**Abstract**: Envy-freeness and the relaxation to Envy-freeness up to one item (EF-1) have been used as fairness concepts in the economics, game theory, and social choice literatures since the 1960s, and have recently gained popularity within the recommendation systems communities. In this short position paper we will give an overview of envy-freeness and its use in economics and recommendation systems; and illustrate why envy is not appropriate to measure fairness for use in settings where personalization plays a role. 

---
# Generative Engine Optimization: How to Dominate AI Search 

**Authors**: Mahe Chen, Xiaoxuan Wang, Kaiwen Chen, Nick Koudas  

**Link**: [PDF](https://arxiv.org/pdf/2509.08919)  

**Abstract**: The rapid adoption of generative AI-powered search engines like ChatGPT, Perplexity, and Gemini is fundamentally reshaping information retrieval, moving from traditional ranked lists to synthesized, citation-backed answers. This shift challenges established Search Engine Optimization (SEO) practices and necessitates a new paradigm, which we term Generative Engine Optimization (GEO).
This paper presents a comprehensive comparative analysis of AI Search and traditional web search (Google). Through a series of large-scale, controlled experiments across multiple verticals, languages, and query paraphrases, we quantify critical differences in how these systems source information. Our key findings reveal that AI Search exhibit a systematic and overwhelming bias towards Earned media (third-party, authoritative sources) over Brand-owned and Social content, a stark contrast to Google's more balanced mix. We further demonstrate that AI Search services differ significantly from each other in their domain diversity, freshness, cross-language stability, and sensitivity to phrasing.
Based on these empirical results, we formulate a strategic GEO agenda. We provide actionable guidance for practitioners, emphasizing the critical need to: (1) engineer content for machine scannability and justification, (2) dominate earned media to build AI-perceived authority, (3) adopt engine-specific and language-aware strategies, and (4) overcome the inherent "big brand bias" for niche players. Our work provides the foundational empirical analysis and a strategic framework for achieving visibility in the new generative search landscape. 

---
# Constructing a Question-Answering Simulator through the Distillation of LLMs 

**Authors**: Haipeng Liu, Ting Long, Jing Fu  

**Link**: [PDF](https://arxiv.org/pdf/2509.09226)  

**Abstract**: The question-answering (QA) simulator is a model that mimics real student learning behaviors and predicts their correctness of their responses to questions. QA simulators enable educational recommender systems (ERS) to collect large amounts of training data without interacting with real students, thereby preventing harmful recommendations made by an undertrained ERS from undermining actual student learning. Given the QA history, there are two categories of solutions to predict the correctness, conducting the simulation: (1) LLM-free methods, which apply a traditional sequential model to transfer the QA history into a vector representation first, and make predictions based on the representation; (2) LLM-based methods, which leverage the domain knowledge and reasoning capability of LLM to enhence the prediction. LLM-free methods offer fast inference but generally yield suboptimal performance. In contrast, most LLM-based methods achieve better results, but at the cost of slower inference speed and higher GPU memory consumption. In this paper, we propose a method named LLM Distillation based Simulator (LDSim), which distills domain knowledge and reasoning capability from an LLM to better assist prediction, thereby improving simulation performance. Extensive experiments demonstrate that our LDSim achieves strong results on both the simulation task and the knowledge tracing (KT) task. Our code is publicly available at this https URL. 

---
# PerFairX: Is There a Balance Between Fairness and Personality in Large Language Model Recommendations? 

**Authors**: Chandan Kumar Sah  

**Link**: [PDF](https://arxiv.org/pdf/2509.08829)  

**Abstract**: The integration of Large Language Models (LLMs) into recommender systems has enabled zero-shot, personality-based personalization through prompt-based interactions, offering a new paradigm for user-centric recommendations. However, incorporating user personality traits via the OCEAN model highlights a critical tension between achieving psychological alignment and ensuring demographic fairness. To address this, we propose PerFairX, a unified evaluation framework designed to quantify the trade-offs between personalization and demographic equity in LLM-generated recommendations. Using neutral and personality-sensitive prompts across diverse user profiles, we benchmark two state-of-the-art LLMs, ChatGPT and DeepSeek, on movie (MovieLens 10M) and music (this http URL 360K) datasets. Our results reveal that personality-aware prompting significantly improves alignment with individual traits but can exacerbate fairness disparities across demographic groups. Specifically, DeepSeek achieves stronger psychological fit but exhibits higher sensitivity to prompt variations, while ChatGPT delivers stable yet less personalized outputs. PerFairX provides a principled benchmark to guide the development of LLM-based recommender systems that are both equitable and psychologically informed, contributing to the creation of inclusive, user-centric AI applications in continual learning contexts. 

---
