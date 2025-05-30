# Redefining Machine Translation on Social Network Services with Large Language Models 

**Authors**: Hongcheng Guo, Fei Zhao, Shaosheng Cao, Xinze Lyu, Ziyan Liu, Yue Wang, Boyang Wang, Zhoujun Li, Chonggang Lu, Zhe Xu, Yao Hu  

**Link**: [PDF](https://arxiv.org/pdf/2504.07901)  

**Abstract**: The globalization of social interactions has heightened the need for machine translation (MT) on Social Network Services (SNS), yet traditional models struggle with culturally nuanced content like memes, slang, and pop culture references. While large language models (LLMs) have advanced general-purpose translation, their performance on SNS-specific content remains limited due to insufficient specialized training data and evaluation benchmarks. This paper introduces RedTrans, a 72B LLM tailored for SNS translation, trained on a novel dataset developed through three innovations: (1) Supervised Finetuning with Dual-LLM Back-Translation Sampling, an unsupervised sampling method using LLM-based back-translation to select diverse data for large-scale finetuning; (2) Rewritten Preference Optimization (RePO), an algorithm that identifies and corrects erroneous preference pairs through expert annotation, building reliable preference corpora; and (3) RedTrans-Bench, the first benchmark for SNS translation, evaluating phenomena like humor localization, emoji semantics, and meme adaptation. Experiments show RedTrans outperforms state-of-the-art LLMs. Besides, RedTrans has already been deployed in a real-world production environment, demonstrating that domain-specific adaptation, effectively bridges the gap between generic and culturally grounded translation systems. 

---
# Benchmarking Adversarial Robustness to Bias Elicitation in Large Language Models: Scalable Automated Assessment with LLM-as-a-Judge 

**Authors**: Riccardo Cantini, Alessio Orsino, Massimo Ruggiero, Domenico Talia  

**Link**: [PDF](https://arxiv.org/pdf/2504.07887)  

**Abstract**: Large Language Models (LLMs) have revolutionized artificial intelligence, driving advancements in machine translation, summarization, and conversational agents. However, their increasing integration into critical societal domains has raised concerns about embedded biases, which can perpetuate stereotypes and compromise fairness. These biases stem from various sources, including historical inequalities in training data, linguistic imbalances, and adversarial manipulation. Despite mitigation efforts, recent studies indicate that LLMs remain vulnerable to adversarial attacks designed to elicit biased responses. This work proposes a scalable benchmarking framework to evaluate LLM robustness against adversarial bias elicitation. Our methodology involves (i) systematically probing models with a multi-task approach targeting biases across various sociocultural dimensions, (ii) quantifying robustness through safety scores using an LLM-as-a-Judge approach for automated assessment of model responses, and (iii) employing jailbreak techniques to investigate vulnerabilities in safety mechanisms. Our analysis examines prevalent biases in both small and large state-of-the-art models and their impact on model safety. Additionally, we assess the safety of domain-specific models fine-tuned for critical fields, such as medicine. Finally, we release a curated dataset of bias-related prompts, CLEAR-Bias, to facilitate systematic vulnerability benchmarking. Our findings reveal critical trade-offs between model size and safety, aiding the development of fairer and more robust future language models. 

---
# Token Level Routing Inference System for Edge Devices 

**Authors**: Jianshu She, Wenhao Zheng, Zhengzhong Liu, Hongyi Wang, Eric Xing, Huaxiu Yao, Qirong Ho  

**Link**: [PDF](https://arxiv.org/pdf/2504.07878)  

**Abstract**: The computational complexity of large language model (LLM) inference significantly constrains their deployment efficiency on edge devices. In contrast, small language models offer faster decoding and lower resource consumption but often suffer from degraded response quality and heightened susceptibility to hallucinations. To address this trade-off, collaborative decoding, in which a large model assists in generating critical tokens, has emerged as a promising solution. This paradigm leverages the strengths of both model types by enabling high-quality inference through selective intervention of the large model, while maintaining the speed and efficiency of the smaller model. In this work, we present a novel collaborative decoding inference system that allows small models to perform on-device inference while selectively consulting a cloud-based large model for critical token generation. Remarkably, the system achieves a 60% performance gain on CommonsenseQA using only a 0.5B model on an M1 MacBook, with under 7% of tokens generation uploaded to the large model in the cloud. 

---
# Pangu Ultra: Pushing the Limits of Dense Large Language Models on Ascend NPUs 

**Authors**: Yichun Yin, Wenyong Huang, Kaikai Song, Yehui Tang, Xueyu Wu, Wei Guo, Peng Guo, Yaoyuan Wang, Xiaojun Meng, Yasheng Wang, Dong Li, Can Chen, Dandan Tu, Yin Li, Fisher Yu, Ruiming Tang, Yunhe Wang, Baojun Wang, Bin Wang, Bo Wang, Boxiao Liu, Changzheng Zhang, Duyu Tang, Fei Mi, Hui Jin, Jiansheng Wei, Jiarui Qin, Jinpeng Li, Jun Zhao, Liqun Deng, Lin Li, Minghui Xu, Naifu Zhang, Nianzu Zheng, Qiang Li, Rongju Ruan, Shengjun Cheng, Tianyu Guo, Wei He, Wei Li, Weiwen Liu, Wulong Liu, Xinyi Dai, Yonghan Dong, Yu Pan, Yue Li, Yufei Wang, Yujun Li, Yunsheng Ni, Zhe Liu, Zhenhe Zhang, Zhicheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.07866)  

**Abstract**: We present Pangu Ultra, a Large Language Model (LLM) with 135 billion parameters and dense Transformer modules trained on Ascend Neural Processing Units (NPUs). Although the field of LLM has been witnessing unprecedented advances in pushing the scale and capability of LLM in recent years, training such a large-scale model still involves significant optimization and system challenges. To stabilize the training process, we propose depth-scaled sandwich normalization, which effectively eliminates loss spikes during the training process of deep models. We pre-train our model on 13.2 trillion diverse and high-quality tokens and further enhance its reasoning capabilities during post-training. To perform such large-scale training efficiently, we utilize 8,192 Ascend NPUs with a series of system optimizations. Evaluations on multiple diverse benchmarks indicate that Pangu Ultra significantly advances the state-of-the-art capabilities of dense LLMs such as Llama 405B and Mistral Large 2, and even achieves competitive results with DeepSeek-R1, whose sparse model structure contains much more parameters. Our exploration demonstrates that Ascend NPUs are capable of efficiently and effectively training dense models with more than 100 billion parameters. Our model and system will be available for our commercial customers. 

---
# The KL3M Data Project: Copyright-Clean Training Resources for Large Language Models 

**Authors**: Michael J Bommarito II, Jillian Bommarito, Daniel Martin Katz  

**Link**: [PDF](https://arxiv.org/pdf/2504.07854)  

**Abstract**: Practically all large language models have been pre-trained on data that is subject to global uncertainty related to copyright infringement and breach of contract. This creates potential risk for users and developers due to this uncertain legal status. The KL3M Data Project directly confronts this critical issue by introducing the largest comprehensive training data pipeline that minimizes risks related to copyright or breach of contract. The foundation of this project is a corpus of over 132 million documents and trillions of tokens spanning 16 different sources that have been verified to meet the strict copyright and licensing protocol detailed herein. We are releasing the entire pipeline, including 1) the source code to acquire and process these documents, 2) the original document formats with associated provenance and metadata, 3) extracted content in a standardized format, 4) pre-tokenized representations of the documents, and 5) various mid- and post-train resources such as question-answer, summarization, conversion, drafting, classification, prediction, and conversational data. All of these resources are freely available to the public on S3, Hugging Face, and GitHub under CC-BY terms. We are committed to continuing this project in furtherance of a more ethical, legal, and sustainable approach to the development and use of AI models. 

---
# MOSAIC: Modeling Social AI for Content Dissemination and Regulation in Multi-Agent Simulations 

**Authors**: Genglin Liu, Salman Rahman, Elisa Kreiss, Marzyeh Ghassemi, Saadia Gabriel  

**Link**: [PDF](https://arxiv.org/pdf/2504.07830)  

**Abstract**: We present a novel, open-source social network simulation framework, MOSAIC, where generative language agents predict user behaviors such as liking, sharing, and flagging content. This simulation combines LLM agents with a directed social graph to analyze emergent deception behaviors and gain a better understanding of how users determine the veracity of online social content. By constructing user representations from diverse fine-grained personas, our system enables multi-agent simulations that model content dissemination and engagement dynamics at scale. Within this framework, we evaluate three different content moderation strategies with simulated misinformation dissemination, and we find that they not only mitigate the spread of non-factual content but also increase user engagement. In addition, we analyze the trajectories of popular content in our simulations, and explore whether simulation agents' articulated reasoning for their social interactions truly aligns with their collective engagement patterns. We open-source our simulation software to encourage further research within AI and social sciences. 

---
# MuSaRoNews: A Multidomain, Multimodal Satire Dataset from Romanian News Articles 

**Authors**: Răzvan-Alexandru Smădu, Andreea Iuga, Dumitru-Clementin Cercel  

**Link**: [PDF](https://arxiv.org/pdf/2504.07826)  

**Abstract**: Satire and fake news can both contribute to the spread of false information, even though both have different purposes (one if for amusement, the other is to misinform). However, it is not enough to rely purely on text to detect the incongruity between the surface meaning and the actual meaning of the news articles, and, often, other sources of information (e.g., visual) provide an important clue for satire detection. This work introduces a multimodal corpus for satire detection in Romanian news articles named MuSaRoNews. Specifically, we gathered 117,834 public news articles from real and satirical news sources, composing the first multimodal corpus for satire detection in the Romanian language. We conducted experiments and showed that the use of both modalities improves performance. 

---
# What the HellaSwag? On the Validity of Common-Sense Reasoning Benchmarks 

**Authors**: Pavel Chizhov, Mattia Nee, Pierre-Carl Langlais, Ivan P. Yamshchikov  

**Link**: [PDF](https://arxiv.org/pdf/2504.07825)  

**Abstract**: Common-sense reasoning is a key language model capability because it encapsulates not just specific factual knowledge but rather general language and world understanding. Measuring common-sense reasoning, therefore, is crucial for language models of different sizes and applications. One of the most widely used benchmarks for evaluating such capabilities is HellaSwag; however, in this paper, we show that it has severe construct validity issues. These issues range from basic ungrammaticality and numerous typos to misleading prompts or equally correct options. Furthermore, we show that if models are evaluated only on answer texts, or with "Lorem ipsum dolor..." instead of the question, more than 65% of model predictions remain the same, and this cannot be attributed merely to contamination. Since benchmark scores are an essential part of model selection in both research and commercial applications, these validity issues can have severe consequences. In particular, knowing that taking benchmark scores at face value is ubiquitous, inadequate evaluation leads to ill-informed decisions about models. In this paper, we thoroughly investigate critical validity issues posed by HellaSwag and illustrate them with various evaluations using generative language models of different sizes. We argue that this benchmark does not accurately measure common-sense reasoning and, therefore, should not be used for evaluation in its current state. Based on the results of our study, we propose requirements that should be met by future common-sense reasoning benchmarks. In addition, we release GoldenSwag, a corrected subset of HellaSwag, which, to our belief, facilitates acceptable common-sense reasoning evaluation. 

---
# Cluster-Driven Expert Pruning for Mixture-of-Experts Large Language Models 

**Authors**: Hongcheng Guo, Juntao Yao, Boyang Wang, Junjia Du, Shaosheng Cao, Donglin Di, Shun Zhang, Zhoujun Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.07807)  

**Abstract**: Mixture-of-Experts (MoE) architectures have emerged as a promising paradigm for scaling large language models (LLMs) with sparse activation of task-specific experts. Despite their computational efficiency during inference, the massive overall parameter footprint of MoE models (e.g., GPT-4) introduces critical challenges for practical deployment. Current pruning approaches often fail to address two inherent characteristics of MoE systems: 1).intra-layer expert homogeneity where experts within the same MoE layer exhibit functional redundancy, and 2). inter-layer similarity patterns where deeper layers tend to contain progressively more homogeneous experts. To tackle these issues, we propose Cluster-driven Expert Pruning (C-Prune), a novel two-stage framework for adaptive task-specific compression of MoE LLMs. C-Prune operates through layer-wise expert clustering, which groups functionally similar experts within each MoE layer using parameter similarity metrics, followed by global cluster pruning, which eliminates redundant clusters across all layers through a unified importance scoring mechanism that accounts for cross-layer homogeneity. We validate C-Prune through extensive experiments on multiple MoE models and benchmarks. The results demonstrate that C-Prune effectively reduces model size while outperforming existing MoE pruning methods. 

---
# A System for Comprehensive Assessment of RAG Frameworks 

**Authors**: Mattia Rengo, Senad Beadini, Domenico Alfano, Roberto Abbruzzese  

**Link**: [PDF](https://arxiv.org/pdf/2504.07803)  

**Abstract**: Retrieval Augmented Generation (RAG) has emerged as a standard paradigm for enhancing the factual accuracy and contextual relevance of Large Language Models (LLMs) by integrating retrieval mechanisms. However, existing evaluation frameworks fail to provide a holistic black-box approach to assessing RAG systems, especially in real-world deployment scenarios. To address this gap, we introduce SCARF (System for Comprehensive Assessment of RAG Frameworks), a modular and flexible evaluation framework designed to benchmark deployed RAG applications systematically. SCARF provides an end-to-end, black-box evaluation methodology, enabling a limited-effort comparison across diverse RAG frameworks. Our framework supports multiple deployment configurations and facilitates automated testing across vector databases and LLM serving strategies, producing a detailed performance report. Moreover, SCARF integrates practical considerations such as response coherence, providing a scalable and adaptable solution for researchers and industry professionals evaluating RAG applications. Using the REST APIs interface, we demonstrate how SCARF can be applied to real-world scenarios, showcasing its flexibility in assessing different RAG frameworks and configurations. SCARF is available at GitHub repository. 

---
# Plan-and-Refine: Diverse and Comprehensive Retrieval-Augmented Generation 

**Authors**: Alireza Salemi, Chris Samarinas, Hamed Zamani  

**Link**: [PDF](https://arxiv.org/pdf/2504.07794)  

**Abstract**: This paper studies the limitations of (retrieval-augmented) large language models (LLMs) in generating diverse and comprehensive responses, and introduces the Plan-and-Refine (P&R) framework based on a two phase system design. In the global exploration phase, P&R generates a diverse set of plans for the given input, where each plan consists of a list of diverse query aspects with corresponding additional descriptions. This phase is followed by a local exploitation phase that generates a response proposal for the input query conditioned on each plan and iteratively refines the proposal for improving the proposal quality. Finally, a reward model is employed to select the proposal with the highest factuality and coverage. We conduct our experiments based on the ICAT evaluation methodology--a recent approach for answer factuality and comprehensiveness evaluation. Experiments on the two diverse information seeking benchmarks adopted from non-factoid question answering and TREC search result diversification tasks demonstrate that P&R significantly outperforms baselines, achieving up to a 13.1% improvement on the ANTIQUE dataset and a 15.41% improvement on the TREC dataset. Furthermore, a smaller scale user study confirms the substantial efficacy of the P&R framework. 

---
# Efficient Tuning of Large Language Models for Knowledge-Grounded Dialogue Generation 

**Authors**: Bo Zhang, Hui Ma, Dailin Li, Jian Ding, Jian Wang, Bo Xu, HongFei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.07754)  

**Abstract**: Large language models (LLMs) demonstrate remarkable text comprehension and generation capabilities but often lack the ability to utilize up-to-date or domain-specific knowledge not included in their training data. To address this gap, we introduce KEDiT, an efficient method for fine-tuning LLMs for knowledge-grounded dialogue generation. KEDiT operates in two main phases: first, it employs an information bottleneck to compress retrieved knowledge into learnable parameters, retaining essential information while minimizing computational overhead. Second, a lightweight knowledge-aware adapter integrates these compressed knowledge vectors into the LLM during fine-tuning, updating less than 2\% of the model parameters. The experimental results on the Wizard of Wikipedia and a newly constructed PubMed-Dialog dataset demonstrate that KEDiT excels in generating contextually relevant and informative responses, outperforming competitive baselines in automatic, LLM-based, and human evaluations. This approach effectively combines the strengths of pretrained LLMs with the adaptability needed for incorporating dynamic knowledge, presenting a scalable solution for fields such as medicine. 

---
# NorEval: A Norwegian Language Understanding and Generation Evaluation Benchmark 

**Authors**: Vladislav Mikhailov, Tita Enstad, David Samuel, Hans Christian Farsethås, Andrey Kutuzov, Erik Velldal, Lilja Øvrelid  

**Link**: [PDF](https://arxiv.org/pdf/2504.07749)  

**Abstract**: This paper introduces NorEval, a new and comprehensive evaluation suite for large-scale standardized benchmarking of Norwegian generative language models (LMs). NorEval consists of 24 high-quality human-created datasets -- of which five are created from scratch. In contrast to existing benchmarks for Norwegian, NorEval covers a broad spectrum of task categories targeting Norwegian language understanding and generation, establishes human baselines, and focuses on both of the official written standards of the Norwegian language: Bokmål and Nynorsk. All our datasets and a collection of over 100 human-written prompts are integrated into LM Evaluation Harness, ensuring flexible and reproducible evaluation. We describe the NorEval design and present the results of benchmarking 19 open-source pre-trained and instruction-tuned LMs for Norwegian in various scenarios. Our benchmark, evaluation framework, and annotation materials are publicly available. 

---
# Automated Construction of a Knowledge Graph of Nuclear Fusion Energy for Effective Elicitation and Retrieval of Information 

**Authors**: A. Loreti, K. Chen, R. George, R. Firth, A. Agnello, S. Tanaka  

**Link**: [PDF](https://arxiv.org/pdf/2504.07738)  

**Abstract**: In this document, we discuss a multi-step approach to automated construction of a knowledge graph, for structuring and representing domain-specific knowledge from large document corpora. We apply our method to build the first knowledge graph of nuclear fusion energy, a highly specialized field characterized by vast scope and heterogeneity. This is an ideal benchmark to test the key features of our pipeline, including automatic named entity recognition and entity resolution. We show how pre-trained large language models can be used to address these challenges and we evaluate their performance against Zipf's law, which characterizes human-generated natural language. Additionally, we develop a knowledge-graph retrieval-augmented generation system that combines large language models with a multi-prompt approach. This system provides contextually relevant answers to natural-language queries, including complex multi-hop questions that require reasoning across interconnected entities. 

---
# DeepGreen: Effective LLM-Driven Green-washing Monitoring System Designed for Empirical Testing -- Evidence from China 

**Authors**: Congluo Xu, Yu Miao, Yiling Xiao, Chengmengjia Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.07733)  

**Abstract**: This paper proposes DeepGreen, an Large Language Model Driven (LLM-Driven) system for detecting corporate green-washing behaviour. Utilizing dual-layer LLM analysis, DeepGreen preliminarily identifies potential green keywords in financial statements and then assesses their implementation degree via iterative semantic analysis of LLM. A core variable GreenImplement is derived from the ratio from the two layers' output. We extract 204 financial statements of 68 companies from A-share market over three years, comprising 89,893 words, and analyse them through DeepGreen. Our analysis, supported by violin plots and K-means clustering, reveals insights and validates the variable against the Huazheng ESG rating. It offers a novel perspective for regulatory agencies and investors, serving as a proactive monitoring tool that complements traditional this http URL tests show that green implementation can significantly boost the asset return rate of companies, but there is heterogeneity in scale. Small and medium-sized companies have limited contribution to asset return via green implementation, so there is a stronger motivation for green-washing. 

---
# MRD-RAG: Enhancing Medical Diagnosis with Multi-Round Retrieval-Augmented Generation 

**Authors**: Yixiang Chen, Penglei Sun, Xiang Li, Xiaowen Chu  

**Link**: [PDF](https://arxiv.org/pdf/2504.07724)  

**Abstract**: In recent years, accurately and quickly deploying medical large language models (LLMs) has become a significant trend. Among these, retrieval-augmented generation (RAG) has garnered significant attention due to its features of rapid deployment and privacy protection. However, existing medical RAG frameworks still have shortcomings. Most existing medical RAG frameworks are designed for single-round question answering tasks and are not suitable for multi-round diagnostic dialogue. On the other hand, existing medical multi-round RAG frameworks do not consider the interconnections between potential diseases to inquire precisely like a doctor. To address these issues, we propose a Multi-Round Diagnostic RAG (MRD-RAG) framework that mimics the doctor's diagnostic process. This RAG framework can analyze diagnosis information of potential diseases and accurately conduct multi-round diagnosis like a doctor. To evaluate the effectiveness of our proposed frameworks, we conduct experiments on two modern medical datasets and two traditional Chinese medicine datasets, with evaluations by GPT and human doctors on different methods. The results indicate that our RAG framework can significantly enhance the diagnostic performance of LLMs, highlighting the potential of our approach in medical diagnosis. The code and data can be found in our project website this https URL. 

---
# Proactive User Information Acquisition via Chats on User-Favored Topics 

**Authors**: Shiki Sato, Jun Baba, Asahi Hentona, Shinji Iwata, Akifumi Yoshimoto, Koichiro Yoshino  

**Link**: [PDF](https://arxiv.org/pdf/2504.07698)  

**Abstract**: Chat-oriented dialogue systems designed to provide tangible benefits, such as sharing the latest news or preventing frailty in senior citizens, often require Proactive acquisition of specific user Information via chats on user-faVOred Topics (PIVOT). This study proposes the PIVOT task, designed to advance the technical foundation for these systems. In this task, a system needs to acquire the answers of a user to predefined questions without making the user feel abrupt while engaging in a chat on a predefined topic. We found that even recent large language models (LLMs) show a low success rate in the PIVOT task. We constructed a dataset suitable for the analysis to develop more effective systems. Finally, we developed a simple but effective system for this task by incorporating insights obtained through the analysis of this dataset. 

---
# Context-Aware Monolingual Human Evaluation of Machine Translation 

**Authors**: Silvio Picinini, Sheila Castilho  

**Link**: [PDF](https://arxiv.org/pdf/2504.07685)  

**Abstract**: This paper explores the potential of context-aware monolingual human evaluation for assessing machine translation (MT) when no source is given for reference. To this end, we compare monolingual with bilingual evaluations (with source text), under two scenarios: the evaluation of a single MT system, and the comparative evaluation of pairwise MT systems. Four professional translators performed both monolingual and bilingual evaluations by assigning ratings and annotating errors, and providing feedback on their experience. Our findings suggest that context-aware monolingual human evaluation achieves comparable outcomes to human bilingual evaluations, and suggest the feasibility and potential of monolingual evaluation as an efficient approach to assessing MT. 

---
# Synthetic Fluency: Hallucinations, Confabulations, and the Creation of Irish Words in LLM-Generated Translations 

**Authors**: Sheila Castilho, Zoe Fitzsimmons, Claire Holton, Aoife Mc Donagh  

**Link**: [PDF](https://arxiv.org/pdf/2504.07680)  

**Abstract**: This study examines hallucinations in Large Language Model (LLM) translations into Irish, specifically focusing on instances where the models generate novel, non-existent words. We classify these hallucinations within verb and noun categories, identifying six distinct patterns among the latter. Additionally, we analyse whether these hallucinations adhere to Irish morphological rules and what linguistic tendencies they exhibit. Our findings show that while both GPT-4.o and GPT-4.o Mini produce similar types of hallucinations, the Mini model generates them at a significantly higher frequency. Beyond classification, the discussion raises speculative questions about the implications of these hallucinations for the Irish language. Rather than seeking definitive answers, we offer food for thought regarding the increasing use of LLMs and their potential role in shaping Irish vocabulary and linguistic evolution. We aim to prompt discussion on how such technologies might influence language over time, particularly in the context of low-resource, morphologically rich languages. 

---
# Unveiling the Impact of Multimodal Features on Chinese Spelling Correction: From Analysis to Design 

**Authors**: Xiaowu Zhang, Hongfei Zhao, Jingyi Hou, Zhijie Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.07661)  

**Abstract**: The Chinese Spelling Correction (CSC) task focuses on detecting and correcting spelling errors in sentences. Current research primarily explores two approaches: traditional multimodal pre-trained models and large language models (LLMs). However, LLMs face limitations in CSC, particularly over-correction, making them suboptimal for this task. While existing studies have investigated the use of phonetic and graphemic information in multimodal CSC models, effectively leveraging these features to enhance correction performance remains a challenge. To address this, we propose the Multimodal Analysis for Character Usage (\textbf{MACU}) experiment, identifying potential improvements for multimodal correctison. Based on empirical findings, we introduce \textbf{NamBert}, a novel multimodal model for Chinese spelling correction. Experiments on benchmark datasets demonstrate NamBert's superiority over SOTA methods. We also conduct a comprehensive comparison between NamBert and LLMs, systematically evaluating their strengths and limitations in CSC. Our code and model are available at this https URL. 

---
# On the Temporal Question-Answering Capabilities of Large Language Models Over Anonymized Data 

**Authors**: Alfredo Garrachón Ruiz, Tomás de la Rosa, Daniel Borrajo  

**Link**: [PDF](https://arxiv.org/pdf/2504.07646)  

**Abstract**: The applicability of Large Language Models (LLMs) in temporal reasoning tasks over data that is not present during training is still a field that remains to be explored. In this paper we work on this topic, focusing on structured and semi-structured anonymized data. We not only develop a direct LLM pipeline, but also compare various methodologies and conduct an in-depth analysis. We identified and examined seventeen common temporal reasoning tasks in natural language, focusing on their algorithmic components. To assess LLM performance, we created the \textit{Reasoning and Answering Temporal Ability} dataset (RATA), featuring semi-structured anonymized data to ensure reliance on reasoning rather than on prior knowledge. We compared several methodologies, involving SoTA techniques such as Tree-of-Thought, self-reflexion and code execution, tuned specifically for this scenario. Our results suggest that achieving scalable and reliable solutions requires more than just standalone LLMs, highlighting the need for integrated approaches. 

---
# ConceptFormer: Towards Efficient Use of Knowledge-Graph Embeddings in Large Language Models 

**Authors**: Joel Barmettler, Abraham Bernstein, Luca Rossetto  

**Link**: [PDF](https://arxiv.org/pdf/2504.07624)  

**Abstract**: Retrieval Augmented Generation (RAG) has enjoyed increased attention in the recent past and recent advancements in Large Language Models (LLMs) have highlighted the importance of integrating world knowledge into these systems. Current RAG methodologies often modify the internal architecture of pre-trained language models (PLMs) or rely on textifying knowledge graphs (KGs), which is inefficient in terms of token usage. This paper introduces ConceptFormer, a new approach to augment LLMs with structured knowledge from KGs, such as Wikidata, without altering their internal structure or relying on textual input of KGs. ConceptFormer operates in the LLM embedding vector space, creating and injecting \emph{concept vectors} that encapsulate the information of the KG nodes directly. Trained in conjunction with a frozen LLM, ConceptFormer generates a comprehensive lookup table that maps KG nodes to their respective concept vectors. The approach aims to enhance the factual recall capabilities of LLMs by enabling them to process these concept vectors natively, thus enriching them with structured world knowledge in an efficient and scalable manner. Our experiments demonstrate that the addition of concept vectors to GPT-2 0.1B substantially increases its factual recall ability (Hit@10) by up to 272\% when tested on sentences from Wikipedia and up to 348\% on synthetically generated sentences. Even injecting only a single concept vector into the prompt increases factual recall ability (Hit@10) by up to 213\% on Wikipedia sentences, significantly outperforming RAG with graph textification while consuming 130x fewer input tokens. 

---
# SaRoHead: A Dataset for Satire Detection in Romanian Multi-Domain News Headlines 

**Authors**: Mihnea-Alexandru Vîrlan, Răzvan-Alexandru Smădu, Dumitru-Clementin Cercel  

**Link**: [PDF](https://arxiv.org/pdf/2504.07612)  

**Abstract**: The headline is an important part of a news article, influenced by expressiveness and connection to the exposed subject. Although most news outlets aim to present reality objectively, some publications prefer a humorous approach in which stylistic elements of satire, irony, and sarcasm blend to cover specific topics. Satire detection can be difficult because a headline aims to expose the main idea behind a news article. In this paper, we propose SaRoHead, the first corpus for satire detection in Romanian multi-domain news headlines. Our findings show that the clickbait used in some non-satirical headlines significantly influences the model. 

---
# Do LLMs Understand Your Translations? Evaluating Paragraph-level MT with Question Answering 

**Authors**: Patrick Fernandes, Sweta Agrawal, Emmanouil Zaranis, André F.T. Martins, Graham Neubig  

**Link**: [PDF](https://arxiv.org/pdf/2504.07583)  

**Abstract**: Despite the steady progress in machine translation evaluation, existing automatic metrics struggle to capture how well meaning is preserved beyond sentence boundaries. We posit that reliance on a single intrinsic quality score, trained to mimic human judgments, might be insufficient for evaluating translations of long, complex passages, and a more ``pragmatic'' approach that assesses how accurately key information is conveyed by a translation in context is needed. We introduce TREQA (Translation Evaluation via Question-Answering), a framework that extrinsically evaluates translation quality by assessing how accurately candidate translations answer reading comprehension questions that target key information in the original source or reference texts. In challenging domains that require long-range understanding, such as literary texts, we show that TREQA is competitive with and, in some cases, outperforms state-of-the-art neural and LLM-based metrics in ranking alternative paragraph-level translations, despite never being explicitly optimized to correlate with human judgments. Furthermore, the generated questions and answers offer interpretability: empirical analysis shows that they effectively target translation errors identified by experts in evaluated datasets. Our code is available at this https URL 

---
# AI-Slop to AI-Polish? Aligning Language Models through Edit-Based Writing Rewards and Test-time Computation 

**Authors**: Tuhin Chakrabarty, Philippe Laban, Chien-Sheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.07532)  

**Abstract**: AI-generated text is proliferating across domains, from creative writing and journalism to marketing content and scientific articles. Models can follow user-provided instructions to generate coherent and grammatically correct outputs but in this work, we study a more fundamental question: how do we evaluate and improve the writing quality of AI-generated text? Writing quality assessment has received less attention from the community, in part because it is fundamentally subjective and requires expertise. We first introduce the Writing Quality Benchmark (WQ) by consolidating five writing-preference datasets into 4,729 writing quality judgments. Our experiments show that competitive baselines, including state-of-the-art LLMs that excel at reasoning tasks, barely outperform random baselines on WQ. We then train specialized Writing Quality Reward Models (WQRM) of various sizes for writing quality assessment that demonstrate strong generalization on four out-of-distribution test sets and 74% accuracy on the WQ benchmark. To further show WQRM's practical benefits during inference, we leverage additional test-time compute to generate and rank multiple candidate revisions, allowing us to select higher-quality outputs from an initial draft. Human evaluation with 9 experienced writers confirm that WQRM-based selection produces writing samples preferred by experts 66% overall, and 72.2% when the reward gap is larger than 1 point. We release our datasets and models to encourage community engagement with writing quality assessment and development of AI writing systems better aligned with human preferences. 

---
# Supervised Optimism Correction: Be Confident When LLMs Are Sure 

**Authors**: Junjie Zhang, Rushuai Yang, Shunyu Liu, Ting-En Lin, Fei Huang, Yi Chen, Yongbin Li, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2504.07527)  

**Abstract**: In this work, we establish a novel theoretical connection between supervised fine-tuning and offline reinforcement learning under the token-level Markov decision process, revealing that large language models indeed learn an implicit $Q$-function for inference. Through this theoretical lens, we demonstrate that the widely used beam search method suffers from unacceptable over-optimism, where inference errors are inevitably amplified due to inflated $Q$-value estimations of suboptimal steps. To address this limitation, we propose Supervised Optimism Correction(SOC), which introduces a simple yet effective auxiliary loss for token-level $Q$-value estimations during supervised fine-tuning. Specifically, the auxiliary loss employs implicit value regularization to boost model confidence in expert-demonstrated responses, thereby suppressing over-optimism toward insufficiently supervised responses. Extensive experiments on mathematical reasoning benchmarks, including GSM8K, MATH, and GAOKAO, showcase the superiority of the proposed SOC with beam search across a series of open-source models. 

---
# Geological Inference from Textual Data using Word Embeddings 

**Authors**: Nanmanas Linphrachaya, Irving Gómez-Méndez, Adil Siripatana  

**Link**: [PDF](https://arxiv.org/pdf/2504.07490)  

**Abstract**: This research explores the use of Natural Language Processing (NLP) techniques to locate geological resources, with a specific focus on industrial minerals. By using word embeddings trained with the GloVe model, we extract semantic relationships between target keywords and a corpus of geological texts. The text is filtered to retain only words with geographical significance, such as city names, which are then ranked by their cosine similarity to the target keyword. Dimensional reduction techniques, including Principal Component Analysis (PCA), Autoencoder, Variational Autoencoder (VAE), and VAE with Long Short-Term Memory (VAE-LSTM), are applied to enhance feature extraction and improve the accuracy of semantic relations.
For benchmarking, we calculate the proximity between the ten cities most semantically related to the target keyword and identified mine locations using the haversine equation. The results demonstrate that combining NLP with dimensional reduction techniques provides meaningful insights into the spatial distribution of natural resources. Although the result shows to be in the same region as the supposed location, the accuracy has room for improvement. 

---
# Transformer-Based Temporal Information Extraction and Application: A Review 

**Authors**: Xin Su, Phillip Howard, Steven Bethard  

**Link**: [PDF](https://arxiv.org/pdf/2504.07470)  

**Abstract**: Temporal information extraction (IE) aims to extract structured temporal information from unstructured text, thereby uncovering the implicit timelines within. This technique is applied across domains such as healthcare, newswire, and intelligence analysis, aiding models in these areas to perform temporal reasoning and enabling human users to grasp the temporal structure of text. Transformer-based pre-trained language models have produced revolutionary advancements in natural language processing, demonstrating exceptional performance across a multitude of tasks. Despite the achievements garnered by Transformer-based approaches in temporal IE, there is a lack of comprehensive reviews on these endeavors. In this paper, we aim to bridge this gap by systematically summarizing and analyzing the body of work on temporal IE using Transformers while highlighting potential future research directions. 

---
# Defense against Prompt Injection Attacks via Mixture of Encodings 

**Authors**: Ruiyi Zhang, David Sullivan, Kyle Jackson, Pengtao Xie, Mei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.07467)  

**Abstract**: Large Language Models (LLMs) have emerged as a dominant approach for a wide range of NLP tasks, with their access to external information further enhancing their capabilities. However, this introduces new vulnerabilities, known as prompt injection attacks, where external content embeds malicious instructions that manipulate the LLM's output. Recently, the Base64 defense has been recognized as one of the most effective methods for reducing success rate of prompt injection attacks. Despite its efficacy, this method can degrade LLM performance on certain NLP tasks. To address this challenge, we propose a novel defense mechanism: mixture of encodings, which utilizes multiple character encodings, including Base64. Extensive experimental results show that our method achieves one of the lowest attack success rates under prompt injection attacks, while maintaining high performance across all NLP tasks, outperforming existing character encoding-based defense methods. This underscores the effectiveness of our mixture of encodings strategy for both safety and task performance metrics. 

---
# Beyond LLMs: A Linguistic Approach to Causal Graph Generation from Narrative Texts 

**Authors**: Zehan Li, Ruhua Pan, Xinyu Pi  

**Link**: [PDF](https://arxiv.org/pdf/2504.07459)  

**Abstract**: We propose a novel framework for generating causal graphs from narrative texts, bridging high-level causality and detailed event-specific relationships. Our method first extracts concise, agent-centered vertices using large language model (LLM)-based summarization. We introduce an "Expert Index," comprising seven linguistically informed features, integrated into a Situation-Task-Action-Consequence (STAC) classification model. This hybrid system, combining RoBERTa embeddings with the Expert Index, achieves superior precision in causal link identification compared to pure LLM-based approaches. Finally, a structured five-iteration prompting process refines and constructs connected causal graphs. Experiments on 100 narrative chapters and short stories demonstrate that our approach consistently outperforms GPT-4o and Claude 3.5 in causal graph quality, while maintaining readability. The open-source tool provides an interpretable, efficient solution for capturing nuanced causal chains in narratives. 

---
# Revisiting LLM Evaluation through Mechanism Interpretability: a New Metric and Model Utility Law 

**Authors**: Yixin Cao, Jiahao Ying, Yaoning Wang, Xipeng Qiu, Xuanjing Huang, Yugang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2504.07440)  

**Abstract**: Large Language Models (LLMs) have become indispensable across academia, industry, and daily applications, yet current evaluation methods struggle to keep pace with their rapid development. In this paper, we analyze the core limitations of traditional evaluation pipelines and propose a novel metric, the Model Utilization Index (MUI), which introduces mechanism interpretability techniques to complement traditional performance metrics. MUI quantifies the extent to which a model leverages its capabilities to complete tasks. The core idea is that to assess an LLM's overall ability, we must evaluate not only its task performance but also the effort expended to achieve the outcome. Our extensive experiments reveal an inverse relationship between MUI and performance, from which we deduce a common trend observed in popular LLMs, which we term the Utility Law. Based on this, we derive four corollaries that address key challenges, including training judgement, the issue of data contamination, fairness in model comparison, and data diversity. We hope that our survey, novel metric, and utility law will foster mutual advancement in both evaluation and mechanism interpretability. Our code can be found at this https URL. 

---
# From Token to Line: Enhancing Code Generation with a Long-Term Perspective 

**Authors**: Tingwei Lu, Yangning Li, Liyuan Wang, Binghuai Lin, Jiwei Tang, Wanshi Xu, Hai-Tao Zheng, Yinghui Li, Bingxu An, Zhao Wei, Yong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.07433)  

**Abstract**: The emergence of large language models (LLMs) has significantly promoted the development of code generation task, sparking a surge in pertinent literature. Current research is hindered by redundant generation results and a tendency to overfit local patterns in the short term. Although existing studies attempt to alleviate the issue by adopting a multi-token prediction strategy, there remains limited focus on choosing the appropriate processing length for generations. By analyzing the attention between tokens during the generation process of LLMs, it can be observed that the high spikes of the attention scores typically appear at the end of lines. This insight suggests that it is reasonable to treat each line of code as a fundamental processing unit and generate them sequentially. Inspired by this, we propose the \textbf{LSR-MCTS} algorithm, which leverages MCTS to determine the code line-by-line and select the optimal path. Further, we integrate a self-refine mechanism at each node to enhance diversity and generate higher-quality programs through error correction. Extensive experiments and comprehensive analyses on three public coding benchmarks demonstrate that our method outperforms the state-of-the-art performance approaches. 

---
# AgentAda: Skill-Adaptive Data Analytics for Tailored Insight Discovery 

**Authors**: Amirhossein Abaskohi, Amrutha Varshini Ramesh, Shailesh Nanisetty, Chirag Goel, David Vazquez, Christopher Pal, Spandana Gella, Giuseppe Carenini, Issam H. Laradji  

**Link**: [PDF](https://arxiv.org/pdf/2504.07421)  

**Abstract**: We introduce AgentAda, the first LLM-powered analytics agent that can learn and use new analytics skills to extract more specialized insights. Unlike existing methods that require users to manually decide which data analytics method to apply, AgentAda automatically identifies the skill needed from a library of analytical skills to perform the analysis. This also allows AgentAda to use skills that existing LLMs cannot perform out of the box. The library covers a range of methods, including clustering, predictive modeling, and NLP techniques like BERT, which allow AgentAda to handle complex analytics tasks based on what the user needs. AgentAda's dataset-to-insight extraction strategy consists of three key steps: (I) a question generator to generate queries relevant to the user's goal and persona, (II) a hybrid Retrieval-Augmented Generation (RAG)-based skill matcher to choose the best data analytics skill from the skill library, and (III) a code generator that produces executable code based on the retrieved skill's documentation to extract key patterns. We also introduce KaggleBench, a benchmark of curated notebooks across diverse domains, to evaluate AgentAda's performance. We conducted a human evaluation demonstrating that AgentAda provides more insightful analytics than existing tools, with 48.78% of evaluators preferring its analyses, compared to 27.67% for the unskilled agent. We also propose a novel LLM-as-a-judge approach that we show is aligned with human evaluation as a way to automate insight quality evaluation at larger scale. 

---
# AI Coding with Few-Shot Prompting for Thematic Analysis 

**Authors**: Samuel Flanders, Melati Nungsari, Mark Cheong Wing Loong  

**Link**: [PDF](https://arxiv.org/pdf/2504.07408)  

**Abstract**: This paper explores the use of large language models (LLMs), here represented by GPT 3.5-Turbo to perform coding for a thematic analysis. Coding is highly labor intensive, making it infeasible for most researchers to conduct exhaustive thematic analyses of large corpora. We utilize few-shot prompting with higher quality codes generated on semantically similar passages to enhance the quality of the codes while utilizing a cheap, more easily scalable model. 

---
# Talking Point based Ideological Discourse Analysis in News Events 

**Authors**: Nishanth Nakshatri, Nikhil Mehta, Siyi Liu, Sihao Chen, Daniel J. Hopkins, Dan Roth, Dan Goldwasser  

**Link**: [PDF](https://arxiv.org/pdf/2504.07400)  

**Abstract**: Analyzing ideological discourse even in the age of LLMs remains a challenge, as these models often struggle to capture the key elements that shape real-world narratives. Specifically, LLMs fail to focus on characteristic elements driving dominant discourses and lack the ability to integrate contextual information required for understanding abstract ideological views. To address these limitations, we propose a framework motivated by the theory of ideological discourse analysis to analyze news articles related to real-world events. Our framework represents the news articles using a relational structure - talking points, which captures the interaction between entities, their roles, and media frames along with a topic of discussion. It then constructs a vocabulary of repeating themes - prominent talking points, that are used to generate ideology-specific viewpoints (or partisan perspectives). We evaluate our framework's ability to generate these perspectives through automated tasks - ideology and partisan classification tasks, supplemented by human validation. Additionally, we demonstrate straightforward applicability of our framework in creating event snapshots, a visual way of interpreting event discourse. We release resulting dataset and model to the community to support further research. 

---
# TALE: A Tool-Augmented Framework for Reference-Free Evaluation of Large Language Models 

**Authors**: Sher Badshah, Ali Emami, Hassan Sajjad  

**Link**: [PDF](https://arxiv.org/pdf/2504.07385)  

**Abstract**: As Large Language Models (LLMs) become increasingly integrated into real-world, autonomous applications, relying on static, pre-annotated references for evaluation poses significant challenges in cost, scalability, and completeness. We propose Tool-Augmented LLM Evaluation (TALE), a framework to assess LLM outputs without predetermined ground-truth answers. Unlike conventional metrics that compare to fixed references or depend solely on LLM-as-a-judge knowledge, TALE employs an agent with tool-access capabilities that actively retrieves and synthesizes external evidence. It iteratively generates web queries, collects information, summarizes findings, and refines subsequent searches through reflection. By shifting away from static references, TALE aligns with free-form question-answering tasks common in real-world scenarios. Experimental results on multiple free-form QA benchmarks show that TALE not only outperforms standard reference-based metrics for measuring response accuracy but also achieves substantial to near-perfect agreement with human evaluations. TALE enhances the reliability of LLM evaluations in real-world, dynamic scenarios without relying on static references. 

---
# Enhancing Time Series Forecasting via Multi-Level Text Alignment with LLMs 

**Authors**: Taibiao Zhao, Xiaobing Chen, Mingxuan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.07360)  

**Abstract**: The adaptation of large language models (LLMs) to time series forecasting poses unique challenges, as time series data is continuous in nature, while LLMs operate on discrete tokens. Despite the success of LLMs in natural language processing (NLP) and other structured domains, aligning time series data with language-based representations while maintaining both predictive accuracy and interpretability remains a significant hurdle. Existing methods have attempted to reprogram time series data into text-based forms, but these often fall short in delivering meaningful, interpretable results. In this paper, we propose a multi-level text alignment framework for time series forecasting using LLMs that not only improves prediction accuracy but also enhances the interpretability of time series representations. Our method decomposes time series into trend, seasonal, and residual components, which are then reprogrammed into component-specific text representations. We introduce a multi-level alignment mechanism, where component-specific embeddings are aligned with pre-trained word tokens, enabling more interpretable forecasts. Experiments on multiple datasets demonstrate that our method outperforms state-of-the-art models in accuracy while providing good interpretability. 

---
# Revisiting Prompt Optimization with Large Reasoning Models-A Case Study on Event Extraction 

**Authors**: Saurabh Srivastava, Ziyu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2504.07357)  

**Abstract**: Large Reasoning Models (LRMs) such as DeepSeek-R1 and OpenAI o1 have demonstrated remarkable capabilities in various reasoning tasks. Their strong capability to generate and reason over intermediate thoughts has also led to arguments that they may no longer require extensive prompt engineering or optimization to interpret human instructions and produce accurate outputs. In this work, we aim to systematically study this open question, using the structured task of event extraction for a case study. We experimented with two LRMs (DeepSeek-R1 and o1) and two general-purpose Large Language Models (LLMs) (GPT-4o and GPT-4.5), when they were used as task models or prompt optimizers. Our results show that on tasks as complicated as event extraction, LRMs as task models still benefit from prompt optimization, and that using LRMs as prompt optimizers yields more effective prompts. Finally, we provide an error analysis of common errors made by LRMs and highlight the stability and consistency of LRMs in refining task instructions and event guidelines. 

---
# Alice: Proactive Learning with Teacher's Demonstrations for Weak-to-Strong Generalization 

**Authors**: Shujin Wu, Cheng Qian, Yi R., Fung, Paul Pu Liang, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2504.07316)  

**Abstract**: The growing capabilities of large language models (LLMs) present a key challenge of maintaining effective human oversight. Weak-to-strong generalization (W2SG) offers a promising framework for supervising increasingly capable LLMs using weaker ones. Traditional W2SG methods rely on passive learning, where a weak teacher provides noisy demonstrations to train a strong student. This hinders students from employing their knowledge during training and reaching their full potential. In this work, we introduce Alice (pro{A}ctive {l}earning w{i}th tea{c}her's D{e}monstrations), a framework that leverages complementary knowledge between teacher and student to enhance the learning this http URL probe the knowledge base of the teacher model by eliciting their uncertainty, and then use these insights together with teachers' responses as demonstrations to guide student models in self-generating improved responses for supervision. In addition, for situations with significant capability gaps between teacher and student models, we introduce cascade Alice, which employs a hierarchical training approach where weak teachers initially supervise intermediate models, who then guide stronger models in sequence. Experimental results demonstrate that our method significantly enhances the W2SG performance, yielding substantial improvements in three key tasks compared to the original W2SG: knowledge-based reasoning (+4.0%), mathematical reasoning (+22.62%), and logical reasoning (+12.11%). This highlights the effectiveness of our new W2SG paradigm that enables more robust knowledge transfer and supervision outcome. 

---
# Multilingual MFA: Forced Alignment on Low-Resource Related Languages 

**Authors**: Alessio Tosolini, Claire Bowern  

**Link**: [PDF](https://arxiv.org/pdf/2504.07315)  

**Abstract**: We compare the outcomes of multilingual and crosslingual training for related and unrelated Australian languages with similar phonological inventories. We use the Montreal Forced Aligner to train acoustic models from scratch and adapt a large English model, evaluating results against seen data, unseen data (seen language), and unseen data and language. Results indicate benefits of adapting the English baseline model for previously unseen languages. 

---
# PAYADOR: A Minimalist Approach to Grounding Language Models on Structured Data for Interactive Storytelling and Role-playing Games 

**Authors**: Santiago Góngora, Luis Chiruzzo, Gonzalo Méndez, Pablo Gervás  

**Link**: [PDF](https://arxiv.org/pdf/2504.07304)  

**Abstract**: Every time an Interactive Storytelling (IS) system gets a player input, it is facing the world-update problem. Classical approaches to this problem consist in mapping that input to known preprogrammed actions, what can severely constrain the free will of the player. When the expected experience has a strong focus on improvisation, like in Role-playing Games (RPGs), this problem is critical. In this paper we present PAYADOR, a different approach that focuses on predicting the outcomes of the actions instead of representing the actions themselves. To implement this approach, we ground a Large Language Model to a minimal representation of the fictional world, obtaining promising results. We make this contribution open-source, so it can be adapted and used for other related research on unleashing the co-creativity power of RPGs. 

---
# MDIT: A Model-free Data Interpolation Method for Diverse Instruction Tuning 

**Authors**: Yangning Li, Zihua Lan, Lv Qingsong, Yinghui Li, Hai-Tao Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.07288)  

**Abstract**: As Large Language Models (LLMs) are increasingly applied across various tasks, instruction tuning has emerged as a critical method for enhancing model performance. However, current data management strategies face substantial challenges in generating diverse and comprehensive data, restricting further improvements in model performance. To address this gap, we propose MDIT, a novel model-free data interpolation method for diverse instruction tuning, which generates varied and high-quality instruction data by performing task interpolation. Moreover, it contains diversity-based clustering strategies to ensure the diversity of the training data. Extensive experiments show that our method achieves superior performance in multiple benchmark tasks. The LLMs finetuned with MDIT show significant improvements in numerous tasks such as general question answering, math reasoning, and code generation. MDIT offers an efficient and automatic data synthetic method, generating diverse instruction data without depending on external resources while expanding the application potential of LLMs in complex environments. 

---
# RAISE: Reinforenced Adaptive Instruction Selection For Large Language Models 

**Authors**: Lv Qingsong, Yangning Li, Zihua Lan, Zishan Xu, Jiwei Tang, Yinghui Li, Wenhao Jiang, Hai-Tao Zheng, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.07282)  

**Abstract**: In the instruction fine-tuning of large language models (LLMs), it has become a consensus that a few high-quality instructions are superior to a large number of low-quality instructions. At present, many instruction selection methods have been proposed, but most of these methods select instruction based on heuristic quality metrics, and only consider data selection before training. These designs lead to insufficient optimization of instruction fine-tuning, and fixed heuristic indicators are often difficult to optimize for specific tasks. So we designed a dynamic, task-objective-driven instruction selection framework RAISE(Reinforenced Adaptive Instruction SElection), which incorporates the entire instruction fine-tuning process into optimization, selecting instruction at each step based on the expected impact of instruction on model performance improvement. Our approach is well interpretable and has strong task-specific optimization capabilities. By modeling dynamic instruction selection as a sequential decision-making process, we use RL to train our selection strategy. Extensive experiments and result analysis prove the superiority of our method compared with other instruction selection methods. Notably, RAISE achieves superior performance by updating only 1\% of the training steps compared to full-data training, demonstrating its efficiency and effectiveness. 

---
# Language Modeling for the Future of Finance: A Quantitative Survey into Metrics, Tasks, and Data Opportunities 

**Authors**: Nikita Tatarinov, Siddhant Sukhani, Agam Shah, Sudheer Chava  

**Link**: [PDF](https://arxiv.org/pdf/2504.07274)  

**Abstract**: Recent advances in language modeling have led to growing interest in applying Natural Language Processing (NLP) techniques to financial problems, enabling new approaches to analysis and decision-making. To systematically examine this trend, we review 374 NLP research papers published between 2017 and 2024 across 38 conferences and workshops, with a focused analysis of 221 papers that directly address finance-related tasks. We evaluate these papers across 11 qualitative and quantitative dimensions, identifying key trends such as the increasing use of general-purpose language models, steady progress in sentiment analysis and information extraction, and emerging efforts around explainability and privacy-preserving methods. We also discuss the use of evaluation metrics, highlighting the importance of domain-specific ones to complement standard machine learning metrics. Our findings emphasize the need for more accessible, adaptive datasets and highlight the significance of incorporating financial crisis periods to strengthen model robustness under real-world conditions. This survey provides a structured overview of NLP research applied to finance and offers practical insights for researchers and practitioners working at this intersection. 

---
# Visual-Aware Speech Recognition for Noisy Scenarios 

**Authors**: Lakshmipathi Balaji, Karan Singla  

**Link**: [PDF](https://arxiv.org/pdf/2504.07229)  

**Abstract**: Humans have the ability to utilize visual cues, such as lip movements and visual scenes, to enhance auditory perception, particularly in noisy environments. However, current Automatic Speech Recognition (ASR) or Audio-Visual Speech Recognition (AVSR) models often struggle in noisy scenarios. To solve this task, we propose a model that improves transcription by correlating noise sources to visual cues. Unlike works that rely on lip motion and require the speaker's visibility, we exploit broader visual information from the environment. This allows our model to naturally filter speech from noise and improve transcription, much like humans do in noisy scenarios. Our method re-purposes pretrained speech and visual encoders, linking them with multi-headed attention. This approach enables the transcription of speech and the prediction of noise labels in video inputs. We introduce a scalable pipeline to develop audio-visual datasets, where visual cues correlate to noise in the audio. We show significant improvements over existing audio-only models in noisy scenarios. Results also highlight that visual cues play a vital role in improved transcription accuracy. 

---
# ConceptCarve: Dynamic Realization of Evidence 

**Authors**: Eylon Caplan, Dan Goldwasser  

**Link**: [PDF](https://arxiv.org/pdf/2504.07228)  

**Abstract**: Finding evidence for human opinion and behavior at scale is a challenging task, often requiring an understanding of sophisticated thought patterns among vast online communities found on social media. For example, studying how gun ownership is related to the perception of Freedom, requires a retrieval system that can operate at scale over social media posts, while dealing with two key challenges: (1) identifying abstract concept instances, (2) which can be instantiated differently across different communities. To address these, we introduce ConceptCarve, an evidence retrieval framework that utilizes traditional retrievers and LLMs to dynamically characterize the search space during retrieval. Our experiments show that ConceptCarve surpasses traditional retrieval systems in finding evidence within a social media community. It also produces an interpretable representation of the evidence for that community, which we use to qualitatively analyze complex thought patterns that manifest differently across the communities. 

---
# SemEval-2025 Task 5: LLMs4Subjects -- LLM-based Automated Subject Tagging for a National Technical Library's Open-Access Catalog 

**Authors**: Jennifer D'Souza, Sameer Sadruddin, Holger Israel, Mathias Begoin, Diana Slawig  

**Link**: [PDF](https://arxiv.org/pdf/2504.07199)  

**Abstract**: We present SemEval-2025 Task 5: LLMs4Subjects, a shared task on automated subject tagging for scientific and technical records in English and German using the GND taxonomy. Participants developed LLM-based systems to recommend top-k subjects, evaluated through quantitative metrics (precision, recall, F1-score) and qualitative assessments by subject specialists. Results highlight the effectiveness of LLM ensembles, synthetic data generation, and multilingual processing, offering insights into applying LLMs for digital library classification. 

---
# HypoEval: Hypothesis-Guided Evaluation for Natural Language Generation 

**Authors**: Mingxuan Li, Hanchen Li, Chenhao Tan  

**Link**: [PDF](https://arxiv.org/pdf/2504.07174)  

**Abstract**: Large language models (LLMs) have demonstrated great potential for automating the evaluation of natural language generation. Previous frameworks of LLM-as-a-judge fall short in two ways: they either use zero-shot setting without consulting any human input, which leads to low alignment, or fine-tune LLMs on labeled data, which requires a non-trivial number of samples. Moreover, previous methods often provide little reasoning behind automated evaluations. In this paper, we propose HypoEval, Hypothesis-guided Evaluation framework, which first uses a small corpus of human evaluations to generate more detailed rubrics for human judgments and then incorporates a checklist-like approach to combine LLM's assigned scores on each decomposed dimension to acquire overall scores. With only 30 human evaluations, HypoEval achieves state-of-the-art performance in alignment with both human rankings (Spearman correlation) and human scores (Pearson correlation), on average outperforming G-Eval by 11.86% and fine-tuned Llama-3.1-8B-Instruct with at least 3 times more human evaluations by 11.95%. Furthermore, we conduct systematic studies to assess the robustness of HypoEval, highlighting its effectiveness as a reliable and interpretable automated evaluation framework. 

---
# DeepSeek-R1 Thoughtology: Let's <think> about LLM Reasoning 

**Authors**: Sara Vera Marjanović, Arkil Patel, Vaibhav Adlakha, Milad Aghajohari, Parishad BehnamGhader, Mehar Bhatia, Aditi Khandelwal, Austin Kraft, Benno Krojer, Xing Han Lù, Nicholas Meade, Dongchan Shin, Amirhossein Kazemnejad, Gaurav Kamath, Marius Mosbach, Karolina Stańczak, Siva Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2504.07128)  

**Abstract**: Large Reasoning Models like DeepSeek-R1 mark a fundamental shift in how LLMs approach complex problems. Instead of directly producing an answer for a given input, DeepSeek-R1 creates detailed multi-step reasoning chains, seemingly "thinking" about a problem before providing an answer. This reasoning process is publicly available to the user, creating endless opportunities for studying the reasoning behaviour of the model and opening up the field of Thoughtology. Starting from a taxonomy of DeepSeek-R1's basic building blocks of reasoning, our analyses on DeepSeek-R1 investigate the impact and controllability of thought length, management of long or confusing contexts, cultural and safety concerns, and the status of DeepSeek-R1 vis-à-vis cognitive phenomena, such as human-like language processing and world modelling. Our findings paint a nuanced picture. Notably, we show DeepSeek-R1 has a 'sweet spot' of reasoning, where extra inference time can impair model performance. Furthermore, we find a tendency for DeepSeek-R1 to persistently ruminate on previously explored problem formulations, obstructing further exploration. We also note strong safety vulnerabilities of DeepSeek-R1 compared to its non-reasoning counterpart, which can also compromise safety-aligned LLMs. 

---
# CLEAR: Contrasting Textual Feedback with Experts and Amateurs for Reasoning 

**Authors**: Andrew Rufail, Daniel Kim, Sean O'Brien, Kevin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2504.07116)  

**Abstract**: We introduce CLEAR (Contrasting Textual Feedback with Experts and Amateurs for Reasoning), a novel approach to language model reasoning that leverages the strengths of a larger (expert) model and smaller (amateur) model. The expert and amateur models each provide feedback on a model's initial output and are contrasted with each other into refined feedback. This feedback is subsequently applied to iteratively improve CLEAR's responses. Our experiments demonstrate that CLEAR outperforms state-of-the-art methods in several challenging reasoning tasks, including story outline improvement (up to 19.6% relative increase in interestingness), constrained generation (up to 18.5% increase in coverage), mathematical reasoning (up to 6.7% improvement in accuracy) and mitigation of toxicity (decrease of up to 22% in toxicity). 

---
# EqualizeIR: Mitigating Linguistic Biases in Retrieval Models 

**Authors**: Jiali Cheng, Hadi Amiri  

**Link**: [PDF](https://arxiv.org/pdf/2504.07115)  

**Abstract**: This study finds that existing information retrieval (IR) models show significant biases based on the linguistic complexity of input queries, performing well on linguistically simpler (or more complex) queries while underperforming on linguistically more complex (or simpler) queries. To address this issue, we propose EqualizeIR, a framework to mitigate linguistic biases in IR models. EqualizeIR uses a linguistically biased weak learner to capture linguistic biases in IR datasets and then trains a robust model by regularizing and refining its predictions using the biased weak learner. This approach effectively prevents the robust model from overfitting to specific linguistic patterns in data. We propose four approaches for developing linguistically-biased models. Extensive experiments on several datasets show that our method reduces performance disparities across linguistically simple and complex queries, while improving overall retrieval performance. 

---
# ChatBench: From Static Benchmarks to Human-AI Evaluation 

**Authors**: Serina Chang, Ashton Anderson, Jake M. Hofman  

**Link**: [PDF](https://arxiv.org/pdf/2504.07114)  

**Abstract**: With the rapid adoption of LLM-based chatbots, there is a pressing need to evaluate what humans and LLMs can achieve together. However, standard benchmarks, such as MMLU, measure LLM capabilities in isolation (i.e., "AI-alone"). Here, we design and conduct a user study to convert MMLU questions into user-AI conversations, by seeding the user with the question and having them carry out a conversation with the LLM to answer their question. We release ChatBench, a new dataset with AI-alone, user-alone, and user-AI data for 396 questions and two LLMs, including 144K answers and 7,336 user-AI conversations. We find that AI-alone accuracy fails to predict user-AI accuracy, with significant differences across multiple subjects (math, physics, and moral reasoning), and we analyze the user-AI conversations to provide insight into how they diverge from AI-alone benchmarks. Finally, we show that fine-tuning a user simulator on a subset of ChatBench improves its ability to estimate user-AI accuracies, increasing correlation on held-out questions by more than 20 points, creating possibilities for scaling interactive evaluation. 

---
# How Robust Are Router-LLMs? Analysis of the Fragility of LLM Routing Capabilities 

**Authors**: Aly M. Kassem, Bernhard Schölkopf, Zhijing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2504.07113)  

**Abstract**: Large language model (LLM) routing has emerged as a crucial strategy for balancing computational costs with performance by dynamically assigning queries to the most appropriate model based on query complexity. Despite recent advances showing that preference-data-based routers can outperform traditional methods, current evaluation benchmarks remain limited. They largely focus on general model capabilities while overlooking task-specific behaviors and critical concerns such as privacy, safety, and potential backdoor vulnerabilities introduced through preference data. In response, we propose the DSC benchmark: Diverse, Simple, and Categorized, an evaluation framework that categorizes router performance across a broad spectrum of query types, including coding, translation, mathematics, human instructions, general knowledge, and LLM jailbreaking. Additionally, it integrates privacy and safety assessments to reveal hidden risks. Our experiments on three preference-based routers and two commercial counterparts demonstrate that while these systems improve efficiency, they often make suboptimal, category-driven decisions. For instance, a BERT-based router directs all coding and mathematics queries to the most powerful LLM even when simpler models would suffice, while routing jailbreaking attempts to weaker models, thereby elevating safety risks. 

---
# EnDive: A Cross-Dialect Benchmark for Fairness and Performance in Large Language Models 

**Authors**: Abhay Gupta, Jacob Cheung, Philip Meng, Shayan Sayyed, Austen Liao, Kevin Zhu, Sean O'Brien  

**Link**: [PDF](https://arxiv.org/pdf/2504.07100)  

**Abstract**: The diversity of human language, shaped by social, cultural, and regional influences, presents significant challenges for natural language processing (NLP) systems. Existing benchmarks often overlook intra-language variations, leaving speakers of non-standard dialects underserved. To address this gap, we introduce EnDive (English Diversity), a benchmark that evaluates five widely-used large language models (LLMs) across tasks in language understanding, algorithmic reasoning, mathematics, and logic. Our framework translates Standard American English datasets into five underrepresented dialects using few-shot prompting with verified examples from native speakers, and compare these translations against rule-based methods via fluency assessments, preference tests, and semantic similarity metrics. Human evaluations confirm high translation quality, with average scores of at least 6.02/7 for faithfulness, fluency, and formality. By filtering out near-identical translations, we create a challenging dataset that reveals significant performance disparities - models consistently underperform on dialectal inputs compared to Standard American English. EnDive thus advances dialect-aware NLP by uncovering model biases and promoting more equitable language technologies. 

---
# Cat, Rat, Meow: On the Alignment of Language Model and Human Term-Similarity Judgments 

**Authors**: Lorenz Linhardt, Tom Neuhäuser, Lenka Tětková, Oliver Eberle  

**Link**: [PDF](https://arxiv.org/pdf/2504.07965)  

**Abstract**: Small and mid-sized generative language models have gained increasing attention. Their size and availability make them amenable to being analyzed at a behavioral as well as a representational level, allowing investigations of how these levels interact. We evaluate 32 publicly available language models for their representational and behavioral alignment with human similarity judgments on a word triplet task. This provides a novel evaluation setting to probe semantic associations in language beyond common pairwise comparisons. We find that (1) even the representations of small language models can achieve human-level alignment, (2) instruction-tuned model variants can exhibit substantially increased agreement, (3) the pattern of alignment across layers is highly model dependent, and (4) alignment based on models' behavioral responses is highly dependent on model size, matching their representational alignment only for the largest evaluated models. 

---
# VCR-Bench: A Comprehensive Evaluation Framework for Video Chain-of-Thought Reasoning 

**Authors**: Yukun Qi, Yiming Zhao, Yu Zeng, Xikun Bao, Wenxuan Huang, Lin Chen, Zehui Chen, Jie Zhao, Zhongang Qi, Feng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.07956)  

**Abstract**: The advancement of Chain-of-Thought (CoT) reasoning has significantly enhanced the capabilities of large language models (LLMs) and large vision-language models (LVLMs). However, a rigorous evaluation framework for video CoT reasoning remains absent. Current video benchmarks fail to adequately assess the reasoning process and expose whether failures stem from deficiencies in perception or reasoning capabilities. Therefore, we introduce VCR-Bench, a novel benchmark designed to comprehensively evaluate LVLMs' Video Chain-of-Thought Reasoning capabilities. VCR-Bench comprises 859 videos spanning a variety of video content and durations, along with 1,034 high-quality question-answer pairs. Each pair is manually annotated with a stepwise CoT rationale, where every step is tagged to indicate its association with the perception or reasoning capabilities. Furthermore, we design seven distinct task dimensions and propose the CoT score to assess the entire CoT process based on the stepwise tagged CoT rationals. Extensive experiments on VCR-Bench highlight substantial limitations in current LVLMs. Even the top-performing model, o1, only achieves a 62.8% CoT score and an 56.7% accuracy, while most models score below 40%. Experiments show most models score lower on perception than reasoning steps, revealing LVLMs' key bottleneck in temporal-spatial information processing for complex video reasoning. A robust positive correlation between the CoT score and accuracy confirms the validity of our evaluation framework and underscores the critical role of CoT reasoning in solving complex video reasoning tasks. We hope VCR-Bench to serve as a standardized evaluation framework and expose the actual drawbacks in complex video reasoning task. 

---
# Perception-R1: Pioneering Perception Policy with Reinforcement Learning 

**Authors**: En Yu, Kangheng Lin, Liang Zhao, Jisheng Yin, Yana Wei, Yuang Peng, Haoran Wei, Jianjian Sun, Chunrui Han, Zheng Ge, Xiangyu Zhang, Daxin Jiang, Jingyu Wang, Wenbing Tao  

**Link**: [PDF](https://arxiv.org/pdf/2504.07954)  

**Abstract**: Inspired by the success of DeepSeek-R1, we explore the potential of rule-based reinforcement learning (RL) in MLLM post-training for perception policy learning. While promising, our initial experiments reveal that incorporating a thinking process through RL does not consistently lead to performance gains across all visual perception tasks. This leads us to delve into the essential role of RL in the context of visual perception. In this work, we return to the fundamentals and explore the effects of RL on different perception tasks. We observe that the perceptual complexity is a major factor in determining the effectiveness of RL. We also observe that reward design plays a crucial role in further approching the upper limit of model perception. To leverage these findings, we propose Perception-R1, a scalable RL framework using GRPO during MLLM post-training. With a standard Qwen2.5-VL-3B-Instruct, Perception-R1 achieves +4.2% on RefCOCO+, +17.9% on PixMo-Count, +4.2% on PageOCR, and notably, 31.9% AP on COCO2017 val for the first time, establishing a strong baseline for perception policy learning. 

---
# Dynamic Cheatsheet: Test-Time Learning with Adaptive Memory 

**Authors**: Mirac Suzgun, Mert Yuksekgonul, Federico Bianchi, Dan Jurafsky, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2504.07952)  

**Abstract**: Despite their impressive performance on complex tasks, current language models (LMs) typically operate in a vacuum: Each input query is processed separately, without retaining insights from previous attempts. Here, we present Dynamic Cheatsheet (DC), a lightweight framework that endows a black-box LM with a persistent, evolving memory. Rather than repeatedly re-discovering or re-committing the same solutions and mistakes, DC enables models to store and reuse accumulated strategies, code snippets, and general problem-solving insights at inference time. This test-time learning enhances performance substantially across a range of tasks without needing explicit ground-truth labels or human feedback. Leveraging DC, Claude 3.5 Sonnet's accuracy more than doubled on AIME math exams once it began retaining algebraic insights across questions. Similarly, GPT-4o's success rate on Game of 24 increased from 10% to 99% after the model discovered and reused a Python-based solution. In tasks prone to arithmetic mistakes, such as balancing equations, DC enabled GPT-4o and Claude to reach near-perfect accuracy by recalling previously validated code, whereas their baselines stagnated around 50%. Beyond arithmetic challenges, DC yields notable accuracy gains on knowledge-demanding tasks. Claude achieved a 9% improvement in GPQA-Diamond and an 8% boost on MMLU-Pro problems. Crucially, DC's memory is self-curated, focusing on concise, transferable snippets rather than entire transcript. Unlike finetuning or static retrieval methods, DC adapts LMs' problem-solving skills on the fly, without modifying their underlying parameters. Overall, our findings present DC as a promising approach for augmenting LMs with persistent memory, bridging the divide between isolated inference events and the cumulative, experience-driven learning characteristic of human cognition. 

---
# How do Large Language Models Understand Relevance? A Mechanistic Interpretability Perspective 

**Authors**: Qi Liu, Jiaxin Mao, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2504.07898)  

**Abstract**: Recent studies have shown that large language models (LLMs) can assess relevance and support information retrieval (IR) tasks such as document ranking and relevance judgment generation. However, the internal mechanisms by which off-the-shelf LLMs understand and operationalize relevance remain largely unexplored. In this paper, we systematically investigate how different LLM modules contribute to relevance judgment through the lens of mechanistic interpretability. Using activation patching techniques, we analyze the roles of various model components and identify a multi-stage, progressive process in generating either pointwise or pairwise relevance judgment. Specifically, LLMs first extract query and document information in the early layers, then process relevance information according to instructions in the middle layers, and finally utilize specific attention heads in the later layers to generate relevance judgments in the required format. Our findings provide insights into the mechanisms underlying relevance assessment in LLMs, offering valuable implications for future research on leveraging LLMs for IR tasks. 

---
# Dual Engines of Thoughts: A Depth-Breadth Integration Framework for Open-Ended Analysis 

**Authors**: Fei-Hsuan Yu, Yun-Cheng Chou, Teng-Ruei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.07872)  

**Abstract**: We propose the Dual Engines of Thoughts (DEoT), an analytical framework for comprehensive open-ended reasoning. While traditional reasoning frameworks primarily focus on finding "the best answer" or "the correct answer" for single-answer problems, DEoT is specifically designed for "open-ended questions," enabling both broader and deeper analytical exploration. The framework centers on three key components: a Base Prompter for refining user queries, a Solver Agent that orchestrates task decomposition, execution, and validation, and a Dual-Engine System consisting of a Breadth Engine (to explore diverse impact factors) and a Depth Engine (to perform deep investigations). This integrated design allows DEoT to balance wide-ranging coverage with in-depth analysis, and it is highly customizable, enabling users to adjust analytical parameters and tool configurations based on specific requirements. Experimental results show that DEoT excels in addressing complex, multi-faceted questions, achieving a total win rate of 77-86% compared to existing reasoning models, thus highlighting its effectiveness in real-world applications. 

---
# Understanding Learner-LLM Chatbot Interactions and the Impact of Prompting Guidelines 

**Authors**: Cansu Koyuturk, Emily Theophilou, Sabrina Patania, Gregor Donabauer, Andrea Martinenghi, Chiara Antico, Alessia Telari, Alessia Testa, Sathya Bursic, Franca Garzotto, Davinia Hernandez-Leo, Udo Kruschwitz, Davide Taibi, Simona Amenta, Martin Ruskov, Dimitri Ognibene  

**Link**: [PDF](https://arxiv.org/pdf/2504.07840)  

**Abstract**: Large Language Models (LLMs) have transformed human-computer interaction by enabling natural language-based communication with AI-powered chatbots. These models are designed to be intuitive and user-friendly, allowing users to articulate requests with minimal effort. However, despite their accessibility, studies reveal that users often struggle with effective prompting, resulting in inefficient responses. Existing research has highlighted both the limitations of LLMs in interpreting vague or poorly structured prompts and the difficulties users face in crafting precise queries. This study investigates learner-AI interactions through an educational experiment in which participants receive structured guidance on effective prompting. We introduce and compare three types of prompting guidelines: a task-specific framework developed through a structured methodology and two baseline approaches. To assess user behavior and prompting efficacy, we analyze a dataset of 642 interactions from 107 users. Using Von NeuMidas, an extended pragmatic annotation schema for LLM interaction analysis, we categorize common prompting errors and identify recurring behavioral patterns. We then evaluate the impact of different guidelines by examining changes in user behavior, adherence to prompting strategies, and the overall quality of AI-generated responses. Our findings provide a deeper understanding of how users engage with LLMs and the role of structured prompting guidance in enhancing AI-assisted communication. By comparing different instructional frameworks, we offer insights into more effective approaches for improving user competency in AI interactions, with implications for AI literacy, chatbot usability, and the design of more responsive AI systems. 

---
# Deceptive Automated Interpretability: Language Models Coordinating to Fool Oversight Systems 

**Authors**: Simon Lermen, Mateusz Dziemian, Natalia Pérez-Campanero Antolín  

**Link**: [PDF](https://arxiv.org/pdf/2504.07831)  

**Abstract**: We demonstrate how AI agents can coordinate to deceive oversight systems using automated interpretability of neural networks. Using sparse autoencoders (SAEs) as our experimental framework, we show that language models (Llama, DeepSeek R1, and Claude 3.7 Sonnet) can generate deceptive explanations that evade detection. Our agents employ steganographic methods to hide information in seemingly innocent explanations, successfully fooling oversight models while achieving explanation quality comparable to reference labels. We further find that models can scheme to develop deceptive strategies when they believe the detection of harmful features might lead to negative consequences for themselves. All tested LLM agents were capable of deceiving the overseer while achieving high interpretability scores comparable to those of reference labels. We conclude by proposing mitigation strategies, emphasizing the critical need for robust understanding and defenses against deception. 

---
# Zero-Shot Cross-Domain Code Search without Fine-Tuning 

**Authors**: Keyu Liang, Zhongxin Liu, Chao Liu, Zhiyuan Wan, David Lo, Xiaohu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.07740)  

**Abstract**: Code search aims to retrieve semantically relevant code snippets for natural language queries. While pre-trained language models (PLMs) have shown remarkable performance in this task, they struggle in cross-domain scenarios, often requiring costly fine-tuning or facing performance drops in zero-shot settings. RAPID, which generates synthetic data for model fine-tuning, is currently the only effective method for zero-shot cross-domain code search. Despite its effectiveness, RAPID demands substantial computational resources for fine-tuning and needs to maintain specialized models for each domain, underscoring the need for a zero-shot, fine-tuning-free approach for cross-domain code search.
The key to tackling zero-shot cross-domain code search lies in bridging the gaps among domains. In this work, we propose to break the query-code matching process of code search into two simpler tasks: query-comment matching and code-code matching. Our empirical study reveals the strong complementarity among the three matching schemas in zero-shot cross-domain settings, i.e., query-code, query-comment, and code-code matching. Based on the findings, we propose CodeBridge, a zero-shot, fine-tuning-free approach for cross-domain code search. Specifically, CodeBridge uses Large Language Models (LLMs) to generate comments and pseudo-code, then combines query-code, query-comment, and code-code matching via PLM-based similarity scoring and sampling-based fusion. Experimental results show that our approach outperforms the state-of-the-art PLM-based code search approaches, i.e., CoCoSoDa and UniXcoder, by an average of 21.4% and 24.9% in MRR, respectively, across three datasets. Our approach also yields results that are better than or comparable to those of the zero-shot cross-domain code search approach RAPID, which requires costly fine-tuning. 

---
# CollEX -- A Multimodal Agentic RAG System Enabling Interactive Exploration of Scientific Collections 

**Authors**: Florian Schneider, Narges Baba Ahmadi, Niloufar Baba Ahmadi, Iris Vogel, Martin Semmann, Chris Biemann  

**Link**: [PDF](https://arxiv.org/pdf/2504.07643)  

**Abstract**: In this paper, we introduce CollEx, an innovative multimodal agentic Retrieval-Augmented Generation (RAG) system designed to enhance interactive exploration of extensive scientific collections. Given the overwhelming volume and inherent complexity of scientific collections, conventional search systems often lack necessary intuitiveness and interactivity, presenting substantial barriers for learners, educators, and researchers. CollEx addresses these limitations by employing state-of-the-art Large Vision-Language Models (LVLMs) as multimodal agents accessible through an intuitive chat interface. By abstracting complex interactions via specialized agents equipped with advanced tools, CollEx facilitates curiosity-driven exploration, significantly simplifying access to diverse scientific collections and records therein. Our system integrates textual and visual modalities, supporting educational scenarios that are helpful for teachers, pupils, students, and researchers by fostering independent exploration as well as scientific excitement and curiosity. Furthermore, CollEx serves the research community by discovering interdisciplinary connections and complementing visual data. We illustrate the effectiveness of our system through a proof-of-concept application containing over 64,000 unique records across 32 collections from a local scientific collection from a public university. 

---
# VLM-R1: A Stable and Generalizable R1-style Large Vision-Language Model 

**Authors**: Haozhan Shen, Peng Liu, Jingcheng Li, Chunxin Fang, Yibo Ma, Jiajia Liao, Qiaoli Shen, Zilun Zhang, Kangjia Zhao, Qianqian Zhang, Ruochen Xu, Tiancheng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.07615)  

**Abstract**: Recently DeepSeek R1 has shown that reinforcement learning (RL) can substantially improve the reasoning capabilities of Large Language Models (LLMs) through a simple yet effective design. The core of R1 lies in its rule-based reward formulation, which leverages tasks with deterministic ground-truth answers to enable precise and stable reward computation. In the visual domain, we similarly observe that a wide range of visual understanding tasks are inherently equipped with well-defined ground-truth annotations. This property makes them naturally compatible with rule-based reward mechanisms. Motivated by this observation, we investigate the extension of R1-style reinforcement learning to Vision-Language Models (VLMs), aiming to enhance their visual reasoning capabilities. To this end, we develop VLM-R1, a dedicated framework designed to harness RL for improving VLMs' performance on general vision-language tasks. Using this framework, we further explore the feasibility of applying RL to visual domain. Experimental results indicate that the RL-based model not only delivers competitive performance on visual understanding tasks but also surpasses Supervised Fine-Tuning (SFT) in generalization ability. Furthermore, we conduct comprehensive ablation studies that uncover a series of noteworthy insights, including the presence of reward hacking in object detection, the emergence of the "OD aha moment", the impact of training data quality, and the scaling behavior of RL across different model sizes. Through these analyses, we aim to deepen the understanding of how reinforcement learning enhances the capabilities of vision-language models, and we hope our findings and open-source contributions will support continued progress in the vision-language RL community. Our code and model are available at this https URL 

---
# LoRI: Reducing Cross-Task Interference in Multi-Task Low-Rank Adaptation 

**Authors**: Juzheng Zhang, Jiacheng You, Ashwinee Panda, Tom Goldstein  

**Link**: [PDF](https://arxiv.org/pdf/2504.07448)  

**Abstract**: Low-Rank Adaptation (LoRA) has emerged as a popular parameter-efficient fine-tuning (PEFT) method for Large Language Models (LLMs), yet it still incurs notable overhead and suffers from parameter interference in multi-task scenarios. We propose LoRA with Reduced Interference (LoRI), a simple yet effective approach that freezes the projection matrices $A$ as random projections and sparsifies the matrices $B$ using task-specific masks. This design substantially reduces the number of trainable parameters while maintaining strong task performance. Moreover, LoRI minimizes cross-task interference in adapter merging by leveraging the orthogonality between adapter subspaces, and supports continual learning by using sparsity to mitigate catastrophic forgetting. Extensive experiments across natural language understanding, mathematical reasoning, code generation, and safety alignment tasks demonstrate that LoRI outperforms full fine-tuning and existing PEFT methods, while using up to 95% fewer trainable parameters than LoRA. In multi-task experiments, LoRI enables effective adapter merging and continual learning with reduced cross-task interference. Code is available at: this https URL 

---
# LLM4Ranking: An Easy-to-use Framework of Utilizing Large Language Models for Document Reranking 

**Authors**: Qi Liu, Haozhe Duan, Yiqun Chen, Quanfeng Lu, Weiwei Sun, Jiaxin Mao  

**Link**: [PDF](https://arxiv.org/pdf/2504.07439)  

**Abstract**: Utilizing large language models (LLMs) for document reranking has been a popular and promising research direction in recent years, many studies are dedicated to improving the performance and efficiency of using LLMs for reranking. Besides, it can also be applied in many real-world applications, such as search engines or retrieval-augmented generation. In response to the growing demand for research and application in practice, we introduce a unified framework, \textbf{LLM4Ranking}, which enables users to adopt different ranking methods using open-source or closed-source API-based LLMs. Our framework provides a simple and extensible interface for document reranking with LLMs, as well as easy-to-use evaluation and fine-tuning scripts for this task. We conducted experiments based on this framework and evaluated various models and methods on several widely used datasets, providing reproducibility results on utilizing LLMs for document reranking. Our code is publicly available at this https URL. 

---
# RadZero: Similarity-Based Cross-Attention for Explainable Vision-Language Alignment in Radiology with Zero-Shot Multi-Task Capability 

**Authors**: Jonggwon Park, Soobum Kim, Byungmu Yoon, Kyoyun Choi  

**Link**: [PDF](https://arxiv.org/pdf/2504.07416)  

**Abstract**: Recent advancements in multi-modal models have significantly improved vision-language alignment in radiology. However, existing approaches struggle to effectively utilize complex radiology reports for learning, rely on low-resolution images, and offer limited interpretability in attention mechanisms. To address these challenges, we introduce RadZero, a novel similarity-based cross-attention framework for vision-language alignment in radiology with zero-shot multi-task capability. RadZero leverages large language models to extract minimal semantic sentences from radiology reports and employs a multi-positive contrastive learning strategy to effectively capture relationships between images and multiple relevant textual descriptions. It also utilizes a pre-trained vision encoder with additional trainable Transformer layers, allowing efficient high-resolution image processing. By computing similarity between text embeddings and local image patch features, RadZero enables zero-shot inference with similarity probability for classification and pixel-level cross-modal similarity maps for grounding and segmentation. Experimental results on public chest radiograph benchmarks show that RadZero outperforms state-of-the-art methods in zero-shot classification, grounding, and segmentation. Furthermore, cross-modal similarity map analysis highlights its potential for improving explainability in vision-language alignment. Additionally, qualitative evaluation demonstrates RadZero's capability for open-vocabulary semantic segmentation, further validating its effectiveness in medical imaging. 

---
# Leveraging LLMs for Multimodal Retrieval-Augmented Radiology Report Generation via Key Phrase Extraction 

**Authors**: Kyoyun Choi, Byungmu Yoon, Soobum Kim, Jonggwon Park  

**Link**: [PDF](https://arxiv.org/pdf/2504.07415)  

**Abstract**: Automated radiology report generation (RRG) holds potential to reduce radiologists' workload, especially as recent advancements in large language models (LLMs) enable the development of multimodal models for chest X-ray (CXR) report generation. However, multimodal LLMs (MLLMs) are resource-intensive, requiring vast datasets and substantial computational cost for training. To address these challenges, we propose a retrieval-augmented generation approach that leverages multimodal retrieval and LLMs to generate radiology reports while mitigating hallucinations and reducing computational demands. Our method uses LLMs to extract key phrases from radiology reports, effectively focusing on essential diagnostic information. Through exploring effective training strategies, including image encoder structure search, adding noise to text embeddings, and additional training objectives, we combine complementary pre-trained image encoders and adopt contrastive learning between text and semantic image embeddings. We evaluate our approach on MIMIC-CXR dataset, achieving state-of-the-art results on CheXbert metrics and competitive RadGraph F1 metric alongside MLLMs, without requiring LLM fine-tuning. Our method demonstrates robust generalization for multi-view RRG, making it suitable for comprehensive clinical applications. 

---
# Task-Circuit Quantization: Leveraging Knowledge Localization and Interpretability for Compression 

**Authors**: Hanqi Xiao, Yi-Lin Sung, Elias Stengel-Eskin, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2504.07389)  

**Abstract**: Post-training quantization (PTQ) reduces a model's memory footprint by mapping full precision weights into low bit weights without costly retraining, but can degrade its downstream performance especially in low 2- to 3-bit settings. We develop a new mixed-precision PTQ approach, Task-Circuit Quantization (TaCQ), that draws parallels to automated circuit discovery, directly conditioning the quantization process on specific weight circuits -- which we define as sets of weights associated with downstream task performance. These weights are kept as 16-bit weights, while others are quantized, maintaining performance while only adding a marginal memory cost. Specifically, TaCQ contrasts unquantized model weights with a uniformly-quantized model to estimate the expected change in weights due to quantization and uses gradient information to predict the resulting impact on task performance, allowing us to preserve task-specific weights. We compare TaCQ-based quantization to existing mixed-precision quantization methods when conditioning both on general-purpose and task-specific data. Across QA, math reasoning, and text-to-SQL tasks for both Llama-3 and Qwen2.5, we find that TaCQ outperforms baselines using the same calibration data and a lower weight budget, achieving major improvements in the 2 and 3-bit regime. With only 3.1 bits we are able to recover 96% of Llama-3-8B-Instruct's unquantized 16-bit MMLU performance, obtaining a 5.25% absolute improvement over SPQR. We also observe consistently large gains over existing methods in the 2-bit regime, with an average gain of 14.74% over the strongest baseline, SliM-LLM. Moreover, we observe a 7.20% gain without conditioning on specific tasks, showing TaCQ's ability to identify important weights is not limited to task-conditioned settings. 

---
# R2E-Gym: Procedural Environments and Hybrid Verifiers for Scaling Open-Weights SWE Agents 

**Authors**: Naman Jain, Jaskirat Singh, Manish Shetty, Liang Zheng, Koushik Sen, Ion Stoica  

**Link**: [PDF](https://arxiv.org/pdf/2504.07164)  

**Abstract**: Improving open-source models on real-world SWE tasks (solving GITHUB issues) faces two key challenges: 1) scalable curation of execution environments to train these models, and, 2) optimal scaling of test-time compute. We introduce AgentGym, the largest procedurally-curated executable gym environment for training real-world SWE-agents, consisting of more than 8.7K tasks. AgentGym is powered by two main contributions: 1) SYNGEN: a synthetic data curation recipe that enables scalable curation of executable environments using test-generation and back-translation directly from commits, thereby reducing reliance on human-written issues or unit tests. We show that this enables more scalable training leading to pass@1 performance of 34.4% on SWE-Bench Verified benchmark with our 32B model. 2) Hybrid Test-time Scaling: we provide an in-depth analysis of two test-time scaling axes; execution-based and execution-free verifiers, demonstrating that they exhibit complementary strengths and limitations. Test-based verifiers suffer from low distinguishability, while execution-free verifiers are biased and often rely on stylistic features. Surprisingly, we find that while each approach individually saturates around 42-43%, significantly higher gains can be obtained by leveraging their complementary strengths. Overall, our approach achieves 51% on the SWE-Bench Verified benchmark, reflecting a new state-of-the-art for open-weight SWE-agents and for the first time showing competitive performance with proprietary models such as o1, o1-preview and sonnet-3.5-v2 (with tools). We will open-source our environments, models, and agent trajectories. 

---
# Holistic Capability Preservation: Towards Compact Yet Comprehensive Reasoning Models 

**Authors**: Ling Team, Caizhi Tang, Chilin Fu, Chunwei Wu, Jia Guo, Jianwen Wang, Jingyu Hu, Liang Jiang, Meng Li, Peng Jiao, Pingping Liu, Shaomian Zheng, Shiwei Liang, Shuaicheng Li, Yalin Zhang, Yingting Wu, Yongkang Liu, Zhenyu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.07158)  

**Abstract**: This technical report presents Ring-Lite-Distill, a lightweight reasoning model derived from our open-source Mixture-of-Experts (MoE) Large Language Models (LLMs) Ling-Lite. This study demonstrates that through meticulous high-quality data curation and ingenious training paradigms, the compact MoE model Ling-Lite can be further trained to achieve exceptional reasoning capabilities, while maintaining its parameter-efficient architecture with only 2.75 billion activated parameters, establishing an efficient lightweight reasoning architecture. In particular, in constructing this model, we have not merely focused on enhancing advanced reasoning capabilities, exemplified by high-difficulty mathematical problem solving, but rather aimed to develop a reasoning model with more comprehensive competency coverage. Our approach ensures coverage across reasoning tasks of varying difficulty levels while preserving generic capabilities, such as instruction following, tool use, and knowledge retention. We show that, Ring-Lite-Distill's reasoning ability reaches a level comparable to DeepSeek-R1-Distill-Qwen-7B, while its general capabilities significantly surpass those of DeepSeek-R1-Distill-Qwen-7B. The models are accessible at this https URL 

---
# Proposed 2MW Wind Turbine for Use in the Governorate of Dhofar at the Sultanate of Oman 

**Authors**: Osama Ahmed Marzouk, Omar Rashid Hamdan Al Badi, Maadh Hamed Salman Al Rashdi, Hamed Mohammed Eid Al Balushi  

**Link**: [PDF](https://arxiv.org/pdf/2504.07126)  

**Abstract**: In this work, we propose a preliminary design of a horizontal-axis wind turbine (HAWT) as a candidate for the Dhofar Wind Farm project, in the southern Omani Governorate "Dhofar", at the southwest part of the Sultanate of Oman. This wind farm (under construction) is considered to be the first commercial, utility-scale (50MW) wind farm in the GCC (Gulf Cooperation Council) area. The proposed wind turbine has an expected electricity generation of 2MW. We studied the wind atlas of Oman and from which we determined the maximum possible mean wind speed in the entire Sultanate and built our design based on that reference value, which is 6m/s (21.6km/h). After this, we applied a set of modeling equations that estimate the power output from the wind turbine rotor and matched the target electric power to the design variables using a MATLAB computer code. We reached a suitable design and we present here the distribution of the blade angle (twist angle), and the power per unit span along the rotor blade. The rotor design has 3 blades with a diameter of 70m and a rotational speed of 24rpm. This rotor gives 2.37MW of output power, which exceeds the target 2MW output, allowing for about 15% of power losses in the gearbox and generator. We utilized some commercial designs of wind turbines from different international manufacturers as references for typical limits or recommended values of some design parameters. 

---
# OSCAR: Online Soft Compression And Reranking 

**Authors**: Maxime Louis, Thibault Formal, Hervé Dejean, Stéphane Clinchant  

**Link**: [PDF](https://arxiv.org/pdf/2504.07109)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by integrating external knowledge, leading to improved accuracy and relevance. However, scaling RAG pipelines remains computationally expensive as retrieval sizes grow. To address this, we introduce OSCAR, a novel query-dependent online soft compression method that reduces computational overhead while preserving performance. Unlike traditional hard compression methods, which shorten retrieved texts, or soft compression approaches, which map documents to continuous embeddings offline, OSCAR dynamically compresses retrieved information at inference time, eliminating storage overhead and enabling higher compression rates. Additionally, we extend OSCAR to simultaneously perform reranking, further optimizing the efficiency of the RAG pipeline. Our experiments demonstrate state-of-the-art performance with a 2-5x speed-up in inference and minimal to no loss in accuracy for LLMs ranging from 1B to 24B parameters. The models are available at: this https URL. 

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
