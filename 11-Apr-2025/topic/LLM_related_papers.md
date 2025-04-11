# LLM4Ranking: An Easy-to-use Framework of Utilizing Large Language Models for Document Reranking 

**Authors**: Qi Liu, Haozhe Duan, Yiqun Chen, Quanfeng Lu, Weiwei Sun, Jiaxin Mao  

**Link**: [PDF](https://arxiv.org/pdf/2504.07439)  

**Abstract**: Utilizing large language models (LLMs) for document reranking has been a popular and promising research direction in recent years, many studies are dedicated to improving the performance and efficiency of using LLMs for reranking. Besides, it can also be applied in many real-world applications, such as search engines or retrieval-augmented generation. In response to the growing demand for research and application in practice, we introduce a unified framework, \textbf{LLM4Ranking}, which enables users to adopt different ranking methods using open-source or closed-source API-based LLMs. Our framework provides a simple and extensible interface for document reranking with LLMs, as well as easy-to-use evaluation and fine-tuning scripts for this task. We conducted experiments based on this framework and evaluated various models and methods on several widely used datasets, providing reproducibility results on utilizing LLMs for document reranking. Our code is publicly available at this https URL. 

---
# Exploring Human-Like Thinking in Search Simulations with Large Language Models 

**Authors**: Erhan Zhang, Xingzhu Wang, Peiyuan Gong, Zixuan Yang, Jiaxin Mao  

**Link**: [PDF](https://arxiv.org/pdf/2504.07570)  

**Abstract**: Simulating user search behavior is a critical task in information retrieval, which can be employed for user behavior modeling, data augmentation, and system evaluation. Recent advancements in large language models (LLMs) have opened up new possibilities for generating human-like actions including querying, browsing, and clicking. In this work, we explore the integration of human-like thinking into search simulations by leveraging LLMs to simulate users' hidden cognitive processes. Specifically, given a search task and context, we prompt LLMs to first think like a human before executing the corresponding action. As existing search datasets do not include users' thought processes, we conducted a user study to collect a new dataset enriched with users' explicit thinking. We investigate the impact of incorporating such human-like thinking on simulation performance and apply supervised fine-tuning (SFT) to teach LLMs to emulate both human thinking and actions. Our experiments span two dimensions in leveraging LLMs for user simulation: (1) with or without explicit thinking, and (2) with or without fine-tuning on the thinking-augmented dataset. The results demonstrate the feasibility and potential of incorporating human-like thinking in user simulations, though performance improvements on some metrics remain modest. We believe this exploration provides new avenues and inspirations for advancing user behavior modeling in search simulations. 

---
# How do Large Language Models Understand Relevance? A Mechanistic Interpretability Perspective 

**Authors**: Qi Liu, Jiaxin Mao, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2504.07898)  

**Abstract**: Recent studies have shown that large language models (LLMs) can assess relevance and support information retrieval (IR) tasks such as document ranking and relevance judgment generation. However, the internal mechanisms by which off-the-shelf LLMs understand and operationalize relevance remain largely unexplored. In this paper, we systematically investigate how different LLM modules contribute to relevance judgment through the lens of mechanistic interpretability. Using activation patching techniques, we analyze the roles of various model components and identify a multi-stage, progressive process in generating either pointwise or pairwise relevance judgment. Specifically, LLMs first extract query and document information in the early layers, then process relevance information according to instructions in the middle layers, and finally utilize specific attention heads in the later layers to generate relevance judgments in the required format. Our findings provide insights into the mechanisms underlying relevance assessment in LLMs, offering valuable implications for future research on leveraging LLMs for IR tasks. 

---
# FairEval: Evaluating Fairness in LLM-Based Recommendations with Personality Awareness 

**Authors**: Chandan Kumar Sah, Xiaoli Lian, Tony Xu, Li Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.07801)  

**Abstract**: Recent advances in Large Language Models (LLMs) have enabled their application to recommender systems (RecLLMs), yet concerns remain regarding fairness across demographic and psychological user dimensions. We introduce FairEval, a novel evaluation framework to systematically assess fairness in LLM-based recommendations. FairEval integrates personality traits with eight sensitive demographic attributes,including gender, race, and age, enabling a comprehensive assessment of user-level bias. We evaluate models, including ChatGPT 4o and Gemini 1.5 Flash, on music and movie recommendations. FairEval's fairness metric, PAFS, achieves scores up to 0.9969 for ChatGPT 4o and 0.9997 for Gemini 1.5 Flash, with disparities reaching 34.79 percent. These results highlight the importance of robustness in prompt sensitivity and support more inclusive recommendation systems. 

---
# OSCAR: Online Soft Compression And Reranking 

**Authors**: Maxime Louis, Thibault Formal, Hervé Dejean, Stéphane Clinchant  

**Link**: [PDF](https://arxiv.org/pdf/2504.07109)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by integrating external knowledge, leading to improved accuracy and relevance. However, scaling RAG pipelines remains computationally expensive as retrieval sizes grow. To address this, we introduce OSCAR, a novel query-dependent online soft compression method that reduces computational overhead while preserving performance. Unlike traditional hard compression methods, which shorten retrieved texts, or soft compression approaches, which map documents to continuous embeddings offline, OSCAR dynamically compresses retrieved information at inference time, eliminating storage overhead and enabling higher compression rates. Additionally, we extend OSCAR to simultaneously perform reranking, further optimizing the efficiency of the RAG pipeline. Our experiments demonstrate state-of-the-art performance with a 2-5x speed-up in inference and minimal to no loss in accuracy for LLMs ranging from 1B to 24B parameters. The models are available at: this https URL. 

---
# REANIMATOR: Reanimate Retrieval Test Collections with Extracted and Synthetic Resources 

**Authors**: Björn Engelmann, Fabian Haak, Philipp Schaer, Mani Erfanian Abdoust, Linus Netze, Meik Bittkowski  

**Link**: [PDF](https://arxiv.org/pdf/2504.07584)  

**Abstract**: Retrieval test collections are essential for evaluating information retrieval systems, yet they often lack generalizability across tasks. To overcome this limitation, we introduce REANIMATOR, a versatile framework designed to enable the repurposing of existing test collections by enriching them with extracted and synthetic resources. REANIMATOR enhances test collections from PDF files by parsing full texts and machine-readable tables, as well as related contextual information. It then employs state-of-the-art large language models to produce synthetic relevance labels. Including an optional human-in-the-loop step can help validate the resources that have been extracted and generated. We demonstrate its potential with a revitalized version of the TREC-COVID test collection, showcasing the development of a retrieval-augmented generation system and evaluating the impact of tables on retrieval-augmented generation. REANIMATOR enables the reuse of test collections for new applications, lowering costs and broadening the utility of legacy resources. 

---
# Relevance Isn't All You Need: Scaling RAG Systems With Inference-Time Compute Via Multi-Criteria Reranking 

**Authors**: Will LeVine, Bijan Varjavand  

**Link**: [PDF](https://arxiv.org/pdf/2504.07104)  

**Abstract**: Modern Large Language Model (LLM) systems typically rely on Retrieval Augmented Generation (RAG) which aims to gather context that is useful for response generation. These RAG systems typically optimize strictly towards retrieving context that is maximally relevant to the query. However, conventional theory suggests that retrieval systems which seek to maximize context relevance without any additional explicit criteria can create information bottlenecks. We reaffirm this finding in the modern age of LLM's by showing that in standard RAG pipelines, maximizing for context relevance alone can degrade downstream response quality. In response, we show evaluations of existing RAG methods which account for both context relevance and answer quality. These evaluations introduce a novel finding that existing RAG systems scale poorly with inference time compute usage when considering our combined metric. We introduce "RErank BEyond reLevance (REBEL)", which enables RAG systems to scale with inference-time compute via injection of multi-criteria optimization using Chain-of-Thought prompting (and optionally Multi-Turn dialogue). Ultimately, this enables a new performance/speed tradeoff curve, where RAG systems are able to achieve both higher relevance of retrieved contexts and superior answer quality as inference time increases. Code for the implementation of our method in llama-index can be found at the following PR: this https URL. Code for running experiments using this llama-index implementation can be found at this https URL. 

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
# Pangu Ultra: Pushing the Limits of Dense Large Language Models on Ascend NPUs 

**Authors**: Yichun Yin, Wenyong Huang, Kaikai Song, Yehui Tang, Xueyu Wu, Wei Guo, Peng Guo, Yaoyuan Wang, Xiaojun Meng, Yasheng Wang, Dong Li, Can Chen, Dandan Tu, Yin Li, Fisher Yu, Ruiming Tang, Yunhe Wang, Baojun Wang, Bin Wang, Bo Wang, Boxiao Liu, Changzheng Zhang, Duyu Tang, Fei Mi, Hui Jin, Jiansheng Wei, Jiarui Qin, Jinpeng Li, Jun Zhao, Liqun Deng, Lin Li, Minghui Xu, Naifu Zhang, Nianzu Zheng, Qiang Li, Rongju Ruan, Shengjun Cheng, Tianyu Guo, Wei He, Wei Li, Weiwen Liu, Wulong Liu, Xinyi Dai, Yonghan Dong, Yu Pan, Yue Li, Yufei Wang, Yujun Li, Yunsheng Ni, Zhe Liu, Zhenhe Zhang, Zhicheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.07866)  

**Abstract**: We present Pangu Ultra, a Large Language Model (LLM) with 135 billion parameters and dense Transformer modules trained on Ascend Neural Processing Units (NPUs). Although the field of LLM has been witnessing unprecedented advances in pushing the scale and capability of LLM in recent years, training such a large-scale model still involves significant optimization and system challenges. To stabilize the training process, we propose depth-scaled sandwich normalization, which effectively eliminates loss spikes during the training process of deep models. We pre-train our model on 13.2 trillion diverse and high-quality tokens and further enhance its reasoning capabilities during post-training. To perform such large-scale training efficiently, we utilize 8,192 Ascend NPUs with a series of system optimizations. Evaluations on multiple diverse benchmarks indicate that Pangu Ultra significantly advances the state-of-the-art capabilities of dense LLMs such as Llama 405B and Mistral Large 2, and even achieves competitive results with DeepSeek-R1, whose sparse model structure contains much more parameters. Our exploration demonstrates that Ascend NPUs are capable of efficiently and effectively training dense models with more than 100 billion parameters. Our model and system will be available for our commercial customers. 

---
# A System for Comprehensive Assessment of RAG Frameworks 

**Authors**: Mattia Rengo, Senad Beadini, Domenico Alfano, Roberto Abbruzzese  

**Link**: [PDF](https://arxiv.org/pdf/2504.07803)  

**Abstract**: Retrieval Augmented Generation (RAG) has emerged as a standard paradigm for enhancing the factual accuracy and contextual relevance of Large Language Models (LLMs) by integrating retrieval mechanisms. However, existing evaluation frameworks fail to provide a holistic black-box approach to assessing RAG systems, especially in real-world deployment scenarios. To address this gap, we introduce SCARF (System for Comprehensive Assessment of RAG Frameworks), a modular and flexible evaluation framework designed to benchmark deployed RAG applications systematically. SCARF provides an end-to-end, black-box evaluation methodology, enabling a limited-effort comparison across diverse RAG frameworks. Our framework supports multiple deployment configurations and facilitates automated testing across vector databases and LLM serving strategies, producing a detailed performance report. Moreover, SCARF integrates practical considerations such as response coherence, providing a scalable and adaptable solution for researchers and industry professionals evaluating RAG applications. Using the REST APIs interface, we demonstrate how SCARF can be applied to real-world scenarios, showcasing its flexibility in assessing different RAG frameworks and configurations. SCARF is available at GitHub repository. 

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
# Redefining Machine Translation on Social Network Services with Large Language Models 

**Authors**: Hongcheng Guo, Fei Zhao, Shaosheng Cao, Xinze Lyu, Ziyan Liu, Yue Wang, Boyang Wang, Zhoujun Li, Chonggang Lu, Zhe Xu, Yao Hu  

**Link**: [PDF](https://arxiv.org/pdf/2504.07901)  

**Abstract**: The globalization of social interactions has heightened the need for machine translation (MT) on Social Network Services (SNS), yet traditional models struggle with culturally nuanced content like memes, slang, and pop culture references. While large language models (LLMs) have advanced general-purpose translation, their performance on SNS-specific content remains limited due to insufficient specialized training data and evaluation benchmarks. This paper introduces RedTrans, a 72B LLM tailored for SNS translation, trained on a novel dataset developed through three innovations: (1) Supervised Finetuning with Dual-LLM Back-Translation Sampling, an unsupervised sampling method using LLM-based back-translation to select diverse data for large-scale finetuning; (2) Rewritten Preference Optimization (RePO), an algorithm that identifies and corrects erroneous preference pairs through expert annotation, building reliable preference corpora; and (3) RedTrans-Bench, the first benchmark for SNS translation, evaluating phenomena like humor localization, emoji semantics, and meme adaptation. Experiments show RedTrans outperforms state-of-the-art LLMs. Besides, RedTrans has already been deployed in a real-world production environment, demonstrating that domain-specific adaptation, effectively bridges the gap between generic and culturally grounded translation systems. 

---
# Automated Construction of a Knowledge Graph of Nuclear Fusion Energy for Effective Elicitation and Retrieval of Information 

**Authors**: A. Loreti, K. Chen, R. George, R. Firth, A. Agnello, S. Tanaka  

**Link**: [PDF](https://arxiv.org/pdf/2504.07738)  

**Abstract**: In this document, we discuss a multi-step approach to automated construction of a knowledge graph, for structuring and representing domain-specific knowledge from large document corpora. We apply our method to build the first knowledge graph of nuclear fusion energy, a highly specialized field characterized by vast scope and heterogeneity. This is an ideal benchmark to test the key features of our pipeline, including automatic named entity recognition and entity resolution. We show how pre-trained large language models can be used to address these challenges and we evaluate their performance against Zipf's law, which characterizes human-generated natural language. Additionally, we develop a knowledge-graph retrieval-augmented generation system that combines large language models with a multi-prompt approach. This system provides contextually relevant answers to natural-language queries, including complex multi-hop questions that require reasoning across interconnected entities. 

---
# Efficient Tuning of Large Language Models for Knowledge-Grounded Dialogue Generation 

**Authors**: Bo Zhang, Hui Ma, Dailin Li, Jian Ding, Jian Wang, Bo Xu, HongFei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.07754)  

**Abstract**: Large language models (LLMs) demonstrate remarkable text comprehension and generation capabilities but often lack the ability to utilize up-to-date or domain-specific knowledge not included in their training data. To address this gap, we introduce KEDiT, an efficient method for fine-tuning LLMs for knowledge-grounded dialogue generation. KEDiT operates in two main phases: first, it employs an information bottleneck to compress retrieved knowledge into learnable parameters, retaining essential information while minimizing computational overhead. Second, a lightweight knowledge-aware adapter integrates these compressed knowledge vectors into the LLM during fine-tuning, updating less than 2\% of the model parameters. The experimental results on the Wizard of Wikipedia and a newly constructed PubMed-Dialog dataset demonstrate that KEDiT excels in generating contextually relevant and informative responses, outperforming competitive baselines in automatic, LLM-based, and human evaluations. This approach effectively combines the strengths of pretrained LLMs with the adaptability needed for incorporating dynamic knowledge, presenting a scalable solution for fields such as medicine. 

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
# MOSAIC: Modeling Social AI for Content Dissemination and Regulation in Multi-Agent Simulations 

**Authors**: Genglin Liu, Salman Rahman, Elisa Kreiss, Marzyeh Ghassemi, Saadia Gabriel  

**Link**: [PDF](https://arxiv.org/pdf/2504.07830)  

**Abstract**: We present a novel, open-source social network simulation framework, MOSAIC, where generative language agents predict user behaviors such as liking, sharing, and flagging content. This simulation combines LLM agents with a directed social graph to analyze emergent deception behaviors and gain a better understanding of how users determine the veracity of online social content. By constructing user representations from diverse fine-grained personas, our system enables multi-agent simulations that model content dissemination and engagement dynamics at scale. Within this framework, we evaluate three different content moderation strategies with simulated misinformation dissemination, and we find that they not only mitigate the spread of non-factual content but also increase user engagement. In addition, we analyze the trajectories of popular content in our simulations, and explore whether simulation agents' articulated reasoning for their social interactions truly aligns with their collective engagement patterns. We open-source our simulation software to encourage further research within AI and social sciences. 

---
# Synthetic Fluency: Hallucinations, Confabulations, and the Creation of Irish Words in LLM-Generated Translations 

**Authors**: Sheila Castilho, Zoe Fitzsimmons, Claire Holton, Aoife Mc Donagh  

**Link**: [PDF](https://arxiv.org/pdf/2504.07680)  

**Abstract**: This study examines hallucinations in Large Language Model (LLM) translations into Irish, specifically focusing on instances where the models generate novel, non-existent words. We classify these hallucinations within verb and noun categories, identifying six distinct patterns among the latter. Additionally, we analyse whether these hallucinations adhere to Irish morphological rules and what linguistic tendencies they exhibit. Our findings show that while both GPT-4.o and GPT-4.o Mini produce similar types of hallucinations, the Mini model generates them at a significantly higher frequency. Beyond classification, the discussion raises speculative questions about the implications of these hallucinations for the Irish language. Rather than seeking definitive answers, we offer food for thought regarding the increasing use of LLMs and their potential role in shaping Irish vocabulary and linguistic evolution. We aim to prompt discussion on how such technologies might influence language over time, particularly in the context of low-resource, morphologically rich languages. 

---
# AI-Slop to AI-Polish? Aligning Language Models through Edit-Based Writing Rewards and Test-time Computation 

**Authors**: Tuhin Chakrabarty, Philippe Laban, Chien-Sheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.07532)  

**Abstract**: AI-generated text is proliferating across domains, from creative writing and journalism to marketing content and scientific articles. Models can follow user-provided instructions to generate coherent and grammatically correct outputs but in this work, we study a more fundamental question: how do we evaluate and improve the writing quality of AI-generated text? Writing quality assessment has received less attention from the community, in part because it is fundamentally subjective and requires expertise. We first introduce the Writing Quality Benchmark (WQ) by consolidating five writing-preference datasets into 4,729 writing quality judgments. Our experiments show that competitive baselines, including state-of-the-art LLMs that excel at reasoning tasks, barely outperform random baselines on WQ. We then train specialized Writing Quality Reward Models (WQRM) of various sizes for writing quality assessment that demonstrate strong generalization on four out-of-distribution test sets and 74% accuracy on the WQ benchmark. To further show WQRM's practical benefits during inference, we leverage additional test-time compute to generate and rank multiple candidate revisions, allowing us to select higher-quality outputs from an initial draft. Human evaluation with 9 experienced writers confirm that WQRM-based selection produces writing samples preferred by experts 66% overall, and 72.2% when the reward gap is larger than 1 point. We release our datasets and models to encourage community engagement with writing quality assessment and development of AI writing systems better aligned with human preferences. 

---
# On the Temporal Question-Answering Capabilities of Large Language Models Over Anonymized Data 

**Authors**: Alfredo Garrachón Ruiz, Tomás de la Rosa, Daniel Borrajo  

**Link**: [PDF](https://arxiv.org/pdf/2504.07646)  

**Abstract**: The applicability of Large Language Models (LLMs) in temporal reasoning tasks over data that is not present during training is still a field that remains to be explored. In this paper we work on this topic, focusing on structured and semi-structured anonymized data. We not only develop a direct LLM pipeline, but also compare various methodologies and conduct an in-depth analysis. We identified and examined seventeen common temporal reasoning tasks in natural language, focusing on their algorithmic components. To assess LLM performance, we created the \textit{Reasoning and Answering Temporal Ability} dataset (RATA), featuring semi-structured anonymized data to ensure reliance on reasoning rather than on prior knowledge. We compared several methodologies, involving SoTA techniques such as Tree-of-Thought, self-reflexion and code execution, tuned specifically for this scenario. Our results suggest that achieving scalable and reliable solutions requires more than just standalone LLMs, highlighting the need for integrated approaches. 

---
# Supervised Optimism Correction: Be Confident When LLMs Are Sure 

**Authors**: Junjie Zhang, Rushuai Yang, Shunyu Liu, Ting-En Lin, Fei Huang, Yi Chen, Yongbin Li, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2504.07527)  

**Abstract**: In this work, we establish a novel theoretical connection between supervised fine-tuning and offline reinforcement learning under the token-level Markov decision process, revealing that large language models indeed learn an implicit $Q$-function for inference. Through this theoretical lens, we demonstrate that the widely used beam search method suffers from unacceptable over-optimism, where inference errors are inevitably amplified due to inflated $Q$-value estimations of suboptimal steps. To address this limitation, we propose Supervised Optimism Correction(SOC), which introduces a simple yet effective auxiliary loss for token-level $Q$-value estimations during supervised fine-tuning. Specifically, the auxiliary loss employs implicit value regularization to boost model confidence in expert-demonstrated responses, thereby suppressing over-optimism toward insufficiently supervised responses. Extensive experiments on mathematical reasoning benchmarks, including GSM8K, MATH, and GAOKAO, showcase the superiority of the proposed SOC with beam search across a series of open-source models. 

---
# Proactive User Information Acquisition via Chats on User-Favored Topics 

**Authors**: Shiki Sato, Jun Baba, Asahi Hentona, Shinji Iwata, Akifumi Yoshimoto, Koichiro Yoshino  

**Link**: [PDF](https://arxiv.org/pdf/2504.07698)  

**Abstract**: Chat-oriented dialogue systems designed to provide tangible benefits, such as sharing the latest news or preventing frailty in senior citizens, often require Proactive acquisition of specific user Information via chats on user-faVOred Topics (PIVOT). This study proposes the PIVOT task, designed to advance the technical foundation for these systems. In this task, a system needs to acquire the answers of a user to predefined questions without making the user feel abrupt while engaging in a chat on a predefined topic. We found that even recent large language models (LLMs) show a low success rate in the PIVOT task. We constructed a dataset suitable for the analysis to develop more effective systems. Finally, we developed a simple but effective system for this task by incorporating insights obtained through the analysis of this dataset. 

---
# Revisiting LLM Evaluation through Mechanism Interpretability: a New Metric and Model Utility Law 

**Authors**: Yixin Cao, Jiahao Ying, Yaoning Wang, Xipeng Qiu, Xuanjing Huang, Yugang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2504.07440)  

**Abstract**: Large Language Models (LLMs) have become indispensable across academia, industry, and daily applications, yet current evaluation methods struggle to keep pace with their rapid development. In this paper, we analyze the core limitations of traditional evaluation pipelines and propose a novel metric, the Model Utilization Index (MUI), which introduces mechanism interpretability techniques to complement traditional performance metrics. MUI quantifies the extent to which a model leverages its capabilities to complete tasks. The core idea is that to assess an LLM's overall ability, we must evaluate not only its task performance but also the effort expended to achieve the outcome. Our extensive experiments reveal an inverse relationship between MUI and performance, from which we deduce a common trend observed in popular LLMs, which we term the Utility Law. Based on this, we derive four corollaries that address key challenges, including training judgement, the issue of data contamination, fairness in model comparison, and data diversity. We hope that our survey, novel metric, and utility law will foster mutual advancement in both evaluation and mechanism interpretability. Our code can be found at this https URL. 

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
# Do LLMs Understand Your Translations? Evaluating Paragraph-level MT with Question Answering 

**Authors**: Patrick Fernandes, Sweta Agrawal, Emmanouil Zaranis, André F.T. Martins, Graham Neubig  

**Link**: [PDF](https://arxiv.org/pdf/2504.07583)  

**Abstract**: Despite the steady progress in machine translation evaluation, existing automatic metrics struggle to capture how well meaning is preserved beyond sentence boundaries. We posit that reliance on a single intrinsic quality score, trained to mimic human judgments, might be insufficient for evaluating translations of long, complex passages, and a more ``pragmatic'' approach that assesses how accurately key information is conveyed by a translation in context is needed. We introduce TREQA (Translation Evaluation via Question-Answering), a framework that extrinsically evaluates translation quality by assessing how accurately candidate translations answer reading comprehension questions that target key information in the original source or reference texts. In challenging domains that require long-range understanding, such as literary texts, we show that TREQA is competitive with and, in some cases, outperforms state-of-the-art neural and LLM-based metrics in ranking alternative paragraph-level translations, despite never being explicitly optimized to correlate with human judgments. Furthermore, the generated questions and answers offer interpretability: empirical analysis shows that they effectively target translation errors identified by experts in evaluated datasets. Our code is available at this https URL 

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
# HypoEval: Hypothesis-Guided Evaluation for Natural Language Generation 

**Authors**: Mingxuan Li, Hanchen Li, Chenhao Tan  

**Link**: [PDF](https://arxiv.org/pdf/2504.07174)  

**Abstract**: Large language models (LLMs) have demonstrated great potential for automating the evaluation of natural language generation. Previous frameworks of LLM-as-a-judge fall short in two ways: they either use zero-shot setting without consulting any human input, which leads to low alignment, or fine-tune LLMs on labeled data, which requires a non-trivial number of samples. Moreover, previous methods often provide little reasoning behind automated evaluations. In this paper, we propose HypoEval, Hypothesis-guided Evaluation framework, which first uses a small corpus of human evaluations to generate more detailed rubrics for human judgments and then incorporates a checklist-like approach to combine LLM's assigned scores on each decomposed dimension to acquire overall scores. With only 30 human evaluations, HypoEval achieves state-of-the-art performance in alignment with both human rankings (Spearman correlation) and human scores (Pearson correlation), on average outperforming G-Eval by 11.86% and fine-tuned Llama-3.1-8B-Instruct with at least 3 times more human evaluations by 11.95%. Furthermore, we conduct systematic studies to assess the robustness of HypoEval, highlighting its effectiveness as a reliable and interpretable automated evaluation framework. 

---
# SemEval-2025 Task 5: LLMs4Subjects -- LLM-based Automated Subject Tagging for a National Technical Library's Open-Access Catalog 

**Authors**: Jennifer D'Souza, Sameer Sadruddin, Holger Israel, Mathias Begoin, Diana Slawig  

**Link**: [PDF](https://arxiv.org/pdf/2504.07199)  

**Abstract**: We present SemEval-2025 Task 5: LLMs4Subjects, a shared task on automated subject tagging for scientific and technical records in English and German using the GND taxonomy. Participants developed LLM-based systems to recommend top-k subjects, evaluated through quantitative metrics (precision, recall, F1-score) and qualitative assessments by subject specialists. Results highlight the effectiveness of LLM ensembles, synthetic data generation, and multilingual processing, offering insights into applying LLMs for digital library classification. 

---
# DeepSeek-R1 Thoughtology: Let's <think> about LLM Reasoning 

**Authors**: Sara Vera Marjanović, Arkil Patel, Vaibhav Adlakha, Milad Aghajohari, Parishad BehnamGhader, Mehar Bhatia, Aditi Khandelwal, Austin Kraft, Benno Krojer, Xing Han Lù, Nicholas Meade, Dongchan Shin, Amirhossein Kazemnejad, Gaurav Kamath, Marius Mosbach, Karolina Stańczak, Siva Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2504.07128)  

**Abstract**: Large Reasoning Models like DeepSeek-R1 mark a fundamental shift in how LLMs approach complex problems. Instead of directly producing an answer for a given input, DeepSeek-R1 creates detailed multi-step reasoning chains, seemingly "thinking" about a problem before providing an answer. This reasoning process is publicly available to the user, creating endless opportunities for studying the reasoning behaviour of the model and opening up the field of Thoughtology. Starting from a taxonomy of DeepSeek-R1's basic building blocks of reasoning, our analyses on DeepSeek-R1 investigate the impact and controllability of thought length, management of long or confusing contexts, cultural and safety concerns, and the status of DeepSeek-R1 vis-à-vis cognitive phenomena, such as human-like language processing and world modelling. Our findings paint a nuanced picture. Notably, we show DeepSeek-R1 has a 'sweet spot' of reasoning, where extra inference time can impair model performance. Furthermore, we find a tendency for DeepSeek-R1 to persistently ruminate on previously explored problem formulations, obstructing further exploration. We also note strong safety vulnerabilities of DeepSeek-R1 compared to its non-reasoning counterpart, which can also compromise safety-aligned LLMs. 

---
# Beyond LLMs: A Linguistic Approach to Causal Graph Generation from Narrative Texts 

**Authors**: Zehan Li, Ruhua Pan, Xinyu Pi  

**Link**: [PDF](https://arxiv.org/pdf/2504.07459)  

**Abstract**: We propose a novel framework for generating causal graphs from narrative texts, bridging high-level causality and detailed event-specific relationships. Our method first extracts concise, agent-centered vertices using large language model (LLM)-based summarization. We introduce an "Expert Index," comprising seven linguistically informed features, integrated into a Situation-Task-Action-Consequence (STAC) classification model. This hybrid system, combining RoBERTa embeddings with the Expert Index, achieves superior precision in causal link identification compared to pure LLM-based approaches. Finally, a structured five-iteration prompting process refines and constructs connected causal graphs. Experiments on 100 narrative chapters and short stories demonstrate that our approach consistently outperforms GPT-4o and Claude 3.5 in causal graph quality, while maintaining readability. The open-source tool provides an interpretable, efficient solution for capturing nuanced causal chains in narratives. 

---
# How Robust Are Router-LLMs? Analysis of the Fragility of LLM Routing Capabilities 

**Authors**: Aly M. Kassem, Bernhard Schölkopf, Zhijing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2504.07113)  

**Abstract**: Large language model (LLM) routing has emerged as a crucial strategy for balancing computational costs with performance by dynamically assigning queries to the most appropriate model based on query complexity. Despite recent advances showing that preference-data-based routers can outperform traditional methods, current evaluation benchmarks remain limited. They largely focus on general model capabilities while overlooking task-specific behaviors and critical concerns such as privacy, safety, and potential backdoor vulnerabilities introduced through preference data. In response, we propose the DSC benchmark: Diverse, Simple, and Categorized, an evaluation framework that categorizes router performance across a broad spectrum of query types, including coding, translation, mathematics, human instructions, general knowledge, and LLM jailbreaking. Additionally, it integrates privacy and safety assessments to reveal hidden risks. Our experiments on three preference-based routers and two commercial counterparts demonstrate that while these systems improve efficiency, they often make suboptimal, category-driven decisions. For instance, a BERT-based router directs all coding and mathematics queries to the most powerful LLM even when simpler models would suffice, while routing jailbreaking attempts to weaker models, thereby elevating safety risks. 

---
# ChatBench: From Static Benchmarks to Human-AI Evaluation 

**Authors**: Serina Chang, Ashton Anderson, Jake M. Hofman  

**Link**: [PDF](https://arxiv.org/pdf/2504.07114)  

**Abstract**: With the rapid adoption of LLM-based chatbots, there is a pressing need to evaluate what humans and LLMs can achieve together. However, standard benchmarks, such as MMLU, measure LLM capabilities in isolation (i.e., "AI-alone"). Here, we design and conduct a user study to convert MMLU questions into user-AI conversations, by seeding the user with the question and having them carry out a conversation with the LLM to answer their question. We release ChatBench, a new dataset with AI-alone, user-alone, and user-AI data for 396 questions and two LLMs, including 144K answers and 7,336 user-AI conversations. We find that AI-alone accuracy fails to predict user-AI accuracy, with significant differences across multiple subjects (math, physics, and moral reasoning), and we analyze the user-AI conversations to provide insight into how they diverge from AI-alone benchmarks. Finally, we show that fine-tuning a user simulator on a subset of ChatBench improves its ability to estimate user-AI accuracies, increasing correlation on held-out questions by more than 20 points, creating possibilities for scaling interactive evaluation. 

---
# PAYADOR: A Minimalist Approach to Grounding Language Models on Structured Data for Interactive Storytelling and Role-playing Games 

**Authors**: Santiago Góngora, Luis Chiruzzo, Gonzalo Méndez, Pablo Gervás  

**Link**: [PDF](https://arxiv.org/pdf/2504.07304)  

**Abstract**: Every time an Interactive Storytelling (IS) system gets a player input, it is facing the world-update problem. Classical approaches to this problem consist in mapping that input to known preprogrammed actions, what can severely constrain the free will of the player. When the expected experience has a strong focus on improvisation, like in Role-playing Games (RPGs), this problem is critical. In this paper we present PAYADOR, a different approach that focuses on predicting the outcomes of the actions instead of representing the actions themselves. To implement this approach, we ground a Large Language Model to a minimal representation of the fictional world, obtaining promising results. We make this contribution open-source, so it can be adapted and used for other related research on unleashing the co-creativity power of RPGs. 

---
# Understanding Learner-LLM Chatbot Interactions and the Impact of Prompting Guidelines 

**Authors**: Cansu Koyuturk, Emily Theophilou, Sabrina Patania, Gregor Donabauer, Andrea Martinenghi, Chiara Antico, Alessia Telari, Alessia Testa, Sathya Bursic, Franca Garzotto, Davinia Hernandez-Leo, Udo Kruschwitz, Davide Taibi, Simona Amenta, Martin Ruskov, Dimitri Ognibene  

**Link**: [PDF](https://arxiv.org/pdf/2504.07840)  

**Abstract**: Large Language Models (LLMs) have transformed human-computer interaction by enabling natural language-based communication with AI-powered chatbots. These models are designed to be intuitive and user-friendly, allowing users to articulate requests with minimal effort. However, despite their accessibility, studies reveal that users often struggle with effective prompting, resulting in inefficient responses. Existing research has highlighted both the limitations of LLMs in interpreting vague or poorly structured prompts and the difficulties users face in crafting precise queries. This study investigates learner-AI interactions through an educational experiment in which participants receive structured guidance on effective prompting. We introduce and compare three types of prompting guidelines: a task-specific framework developed through a structured methodology and two baseline approaches. To assess user behavior and prompting efficacy, we analyze a dataset of 642 interactions from 107 users. Using Von NeuMidas, an extended pragmatic annotation schema for LLM interaction analysis, we categorize common prompting errors and identify recurring behavioral patterns. We then evaluate the impact of different guidelines by examining changes in user behavior, adherence to prompting strategies, and the overall quality of AI-generated responses. Our findings provide a deeper understanding of how users engage with LLMs and the role of structured prompting guidance in enhancing AI-assisted communication. By comparing different instructional frameworks, we offer insights into more effective approaches for improving user competency in AI interactions, with implications for AI literacy, chatbot usability, and the design of more responsive AI systems. 

---
# EnDive: A Cross-Dialect Benchmark for Fairness and Performance in Large Language Models 

**Authors**: Abhay Gupta, Jacob Cheung, Philip Meng, Shayan Sayyed, Austen Liao, Kevin Zhu, Sean O'Brien  

**Link**: [PDF](https://arxiv.org/pdf/2504.07100)  

**Abstract**: The diversity of human language, shaped by social, cultural, and regional influences, presents significant challenges for natural language processing (NLP) systems. Existing benchmarks often overlook intra-language variations, leaving speakers of non-standard dialects underserved. To address this gap, we introduce EnDive (English Diversity), a benchmark that evaluates five widely-used large language models (LLMs) across tasks in language understanding, algorithmic reasoning, mathematics, and logic. Our framework translates Standard American English datasets into five underrepresented dialects using few-shot prompting with verified examples from native speakers, and compare these translations against rule-based methods via fluency assessments, preference tests, and semantic similarity metrics. Human evaluations confirm high translation quality, with average scores of at least 6.02/7 for faithfulness, fluency, and formality. By filtering out near-identical translations, we create a challenging dataset that reveals significant performance disparities - models consistently underperform on dialectal inputs compared to Standard American English. EnDive thus advances dialect-aware NLP by uncovering model biases and promoting more equitable language technologies. 

---
# Deceptive Automated Interpretability: Language Models Coordinating to Fool Oversight Systems 

**Authors**: Simon Lermen, Mateusz Dziemian, Natalia Pérez-Campanero Antolín  

**Link**: [PDF](https://arxiv.org/pdf/2504.07831)  

**Abstract**: We demonstrate how AI agents can coordinate to deceive oversight systems using automated interpretability of neural networks. Using sparse autoencoders (SAEs) as our experimental framework, we show that language models (Llama, DeepSeek R1, and Claude 3.7 Sonnet) can generate deceptive explanations that evade detection. Our agents employ steganographic methods to hide information in seemingly innocent explanations, successfully fooling oversight models while achieving explanation quality comparable to reference labels. We further find that models can scheme to develop deceptive strategies when they believe the detection of harmful features might lead to negative consequences for themselves. All tested LLM agents were capable of deceiving the overseer while achieving high interpretability scores comparable to those of reference labels. We conclude by proposing mitigation strategies, emphasizing the critical need for robust understanding and defenses against deception. 

---
# CLEAR: Contrasting Textual Feedback with Experts and Amateurs for Reasoning 

**Authors**: Andrew Rufail, Daniel Kim, Sean O'Brien, Kevin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2504.07116)  

**Abstract**: We introduce CLEAR (Contrasting Textual Feedback with Experts and Amateurs for Reasoning), a novel approach to language model reasoning that leverages the strengths of a larger (expert) model and smaller (amateur) model. The expert and amateur models each provide feedback on a model's initial output and are contrasted with each other into refined feedback. This feedback is subsequently applied to iteratively improve CLEAR's responses. Our experiments demonstrate that CLEAR outperforms state-of-the-art methods in several challenging reasoning tasks, including story outline improvement (up to 19.6% relative increase in interestingness), constrained generation (up to 18.5% increase in coverage), mathematical reasoning (up to 6.7% improvement in accuracy) and mitigation of toxicity (decrease of up to 22% in toxicity). 

---
# Leveraging LLMs for Multimodal Retrieval-Augmented Radiology Report Generation via Key Phrase Extraction 

**Authors**: Kyoyun Choi, Byungmu Yoon, Soobum Kim, Jonggwon Park  

**Link**: [PDF](https://arxiv.org/pdf/2504.07415)  

**Abstract**: Automated radiology report generation (RRG) holds potential to reduce radiologists' workload, especially as recent advancements in large language models (LLMs) enable the development of multimodal models for chest X-ray (CXR) report generation. However, multimodal LLMs (MLLMs) are resource-intensive, requiring vast datasets and substantial computational cost for training. To address these challenges, we propose a retrieval-augmented generation approach that leverages multimodal retrieval and LLMs to generate radiology reports while mitigating hallucinations and reducing computational demands. Our method uses LLMs to extract key phrases from radiology reports, effectively focusing on essential diagnostic information. Through exploring effective training strategies, including image encoder structure search, adding noise to text embeddings, and additional training objectives, we combine complementary pre-trained image encoders and adopt contrastive learning between text and semantic image embeddings. We evaluate our approach on MIMIC-CXR dataset, achieving state-of-the-art results on CheXbert metrics and competitive RadGraph F1 metric alongside MLLMs, without requiring LLM fine-tuning. Our method demonstrates robust generalization for multi-view RRG, making it suitable for comprehensive clinical applications. 

---
# VLM-R1: A Stable and Generalizable R1-style Large Vision-Language Model 

**Authors**: Haozhan Shen, Peng Liu, Jingcheng Li, Chunxin Fang, Yibo Ma, Jiajia Liao, Qiaoli Shen, Zilun Zhang, Kangjia Zhao, Qianqian Zhang, Ruochen Xu, Tiancheng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.07615)  

**Abstract**: Recently DeepSeek R1 has shown that reinforcement learning (RL) can substantially improve the reasoning capabilities of Large Language Models (LLMs) through a simple yet effective design. The core of R1 lies in its rule-based reward formulation, which leverages tasks with deterministic ground-truth answers to enable precise and stable reward computation. In the visual domain, we similarly observe that a wide range of visual understanding tasks are inherently equipped with well-defined ground-truth annotations. This property makes them naturally compatible with rule-based reward mechanisms. Motivated by this observation, we investigate the extension of R1-style reinforcement learning to Vision-Language Models (VLMs), aiming to enhance their visual reasoning capabilities. To this end, we develop VLM-R1, a dedicated framework designed to harness RL for improving VLMs' performance on general vision-language tasks. Using this framework, we further explore the feasibility of applying RL to visual domain. Experimental results indicate that the RL-based model not only delivers competitive performance on visual understanding tasks but also surpasses Supervised Fine-Tuning (SFT) in generalization ability. Furthermore, we conduct comprehensive ablation studies that uncover a series of noteworthy insights, including the presence of reward hacking in object detection, the emergence of the "OD aha moment", the impact of training data quality, and the scaling behavior of RL across different model sizes. Through these analyses, we aim to deepen the understanding of how reinforcement learning enhances the capabilities of vision-language models, and we hope our findings and open-source contributions will support continued progress in the vision-language RL community. Our code and model are available at this https URL 

---
# Zero-Shot Cross-Domain Code Search without Fine-Tuning 

**Authors**: Keyu Liang, Zhongxin Liu, Chao Liu, Zhiyuan Wan, David Lo, Xiaohu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.07740)  

**Abstract**: Code search aims to retrieve semantically relevant code snippets for natural language queries. While pre-trained language models (PLMs) have shown remarkable performance in this task, they struggle in cross-domain scenarios, often requiring costly fine-tuning or facing performance drops in zero-shot settings. RAPID, which generates synthetic data for model fine-tuning, is currently the only effective method for zero-shot cross-domain code search. Despite its effectiveness, RAPID demands substantial computational resources for fine-tuning and needs to maintain specialized models for each domain, underscoring the need for a zero-shot, fine-tuning-free approach for cross-domain code search.
The key to tackling zero-shot cross-domain code search lies in bridging the gaps among domains. In this work, we propose to break the query-code matching process of code search into two simpler tasks: query-comment matching and code-code matching. Our empirical study reveals the strong complementarity among the three matching schemas in zero-shot cross-domain settings, i.e., query-code, query-comment, and code-code matching. Based on the findings, we propose CodeBridge, a zero-shot, fine-tuning-free approach for cross-domain code search. Specifically, CodeBridge uses Large Language Models (LLMs) to generate comments and pseudo-code, then combines query-code, query-comment, and code-code matching via PLM-based similarity scoring and sampling-based fusion. Experimental results show that our approach outperforms the state-of-the-art PLM-based code search approaches, i.e., CoCoSoDa and UniXcoder, by an average of 21.4% and 24.9% in MRR, respectively, across three datasets. Our approach also yields results that are better than or comparable to those of the zero-shot cross-domain code search approach RAPID, which requires costly fine-tuning. 

---
# Holistic Capability Preservation: Towards Compact Yet Comprehensive Reasoning Models 

**Authors**: Ling Team, Caizhi Tang, Chilin Fu, Chunwei Wu, Jia Guo, Jianwen Wang, Jingyu Hu, Liang Jiang, Meng Li, Peng Jiao, Pingping Liu, Shaomian Zheng, Shiwei Liang, Shuaicheng Li, Yalin Zhang, Yingting Wu, Yongkang Liu, Zhenyu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.07158)  

**Abstract**: This technical report presents Ring-Lite-Distill, a lightweight reasoning model derived from our open-source Mixture-of-Experts (MoE) Large Language Models (LLMs) Ling-Lite. This study demonstrates that through meticulous high-quality data curation and ingenious training paradigms, the compact MoE model Ling-Lite can be further trained to achieve exceptional reasoning capabilities, while maintaining its parameter-efficient architecture with only 2.75 billion activated parameters, establishing an efficient lightweight reasoning architecture. In particular, in constructing this model, we have not merely focused on enhancing advanced reasoning capabilities, exemplified by high-difficulty mathematical problem solving, but rather aimed to develop a reasoning model with more comprehensive competency coverage. Our approach ensures coverage across reasoning tasks of varying difficulty levels while preserving generic capabilities, such as instruction following, tool use, and knowledge retention. We show that, Ring-Lite-Distill's reasoning ability reaches a level comparable to DeepSeek-R1-Distill-Qwen-7B, while its general capabilities significantly surpass those of DeepSeek-R1-Distill-Qwen-7B. The models are accessible at this https URL 

---
# Enhancing Large Language Models through Neuro-Symbolic Integration and Ontological Reasoning 

**Authors**: Ruslan Idelfonso Magana Vsevolodovna, Marco Monti  

**Link**: [PDF](https://arxiv.org/pdf/2504.07640)  

**Abstract**: Large Language Models (LLMs) demonstrate impressive capabilities in natural language processing but suffer from inaccuracies and logical inconsistencies known as hallucinations. This compromises their reliability, especially in domains requiring factual accuracy. We propose a neuro-symbolic approach integrating symbolic ontological reasoning and machine learning methods to enhance the consistency and reliability of LLM outputs. Our workflow utilizes OWL ontologies, a symbolic reasoner (e.g., HermiT) for consistency checking, and a lightweight machine learning model (logistic regression) for mapping natural language statements into logical forms compatible with the ontology. When inconsistencies between LLM outputs and the ontology are detected, the system generates explanatory feedback to guide the LLM towards a corrected, logically coherent response in an iterative refinement loop. We present a working Python prototype demonstrating this pipeline. Experimental results in a defined domain suggest significant improvements in semantic coherence and factual accuracy of LLM outputs, showcasing the potential of combining LLM fluency with the rigor of formal semantics. 

---
# A taxonomy of epistemic injustice in the context of AI and the case for generative hermeneutical erasure 

**Authors**: Warmhold Jan Thomas Mollema  

**Link**: [PDF](https://arxiv.org/pdf/2504.07531)  

**Abstract**: Whether related to machine learning models' epistemic opacity, algorithmic classification systems' discriminatory automation of testimonial prejudice, the distortion of human beliefs via the 'hallucinations' of generative AI, the inclusion of the global South in global AI governance, the execution of bureaucratic violence via algorithmic systems, or located in the interaction with conversational artificial agents epistemic injustice related to AI is a growing concern. Based on a proposed general taxonomy of epistemic injustice, this paper first sketches a taxonomy of the types of epistemic injustice in the context of AI, relying on the work of scholars from the fields of philosophy of technology, political philosophy and social epistemology. Secondly, an additional perspective on epistemic injustice in the context of AI: generative hermeneutical erasure. I argue that this injustice that can come about through the application of Large Language Models (LLMs) and contend that generative AI, when being deployed outside of its Western space of conception, can have effects of conceptual erasure, particularly in the epistemic domain, followed by forms of conceptual disruption caused by a mismatch between AI system and the interlocutor in terms of conceptual frameworks. AI systems' 'view from nowhere' epistemically inferiorizes non-Western epistemologies and thereby contributes to the erosion of their epistemic particulars, gradually contributing to hermeneutical erasure. This work's relevance lies in proposal of a taxonomy that allows epistemic injustices to be mapped in the AI domain and the proposal of a novel form of AI-related epistemic injustice. 

---
# Boosting Universal LLM Reward Design through the Heuristic Reward Observation Space Evolution 

**Authors**: Zen Kit Heng, Zimeng Zhao, Tianhao Wu, Yuanfei Wang, Mingdong Wu, Yangang Wang, Hao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2504.07596)  

**Abstract**: Large Language Models (LLMs) are emerging as promising tools for automated reinforcement learning (RL) reward design, owing to their robust capabilities in commonsense reasoning and code generation. By engaging in dialogues with RL agents, LLMs construct a Reward Observation Space (ROS) by selecting relevant environment states and defining their internal operations. However, existing frameworks have not effectively leveraged historical exploration data or manual task descriptions to iteratively evolve this space. In this paper, we propose a novel heuristic framework that enhances LLM-driven reward design by evolving the ROS through a table-based exploration caching mechanism and a text-code reconciliation strategy. Our framework introduces a state execution table, which tracks the historical usage and success rates of environment states, overcoming the Markovian constraint typically found in LLM dialogues and facilitating more effective exploration. Furthermore, we reconcile user-provided task descriptions with expert-defined success criteria using structured prompts, ensuring alignment in reward design objectives. Comprehensive evaluations on benchmark RL tasks demonstrate the effectiveness and stability of the proposed framework. Code and video demos are available at this http URL. 

---
# Enhanced Question-Answering for Skill-based learning using Knowledge-based AI and Generative AI 

**Authors**: Rahul K. Dass, Rochan H. Madhusudhana, Erin C. Deye, Shashank Verma, Timothy A. Bydlon, Grace Brazil, Ashok K. Goel  

**Link**: [PDF](https://arxiv.org/pdf/2504.07463)  

**Abstract**: Supporting learners' understanding of taught skills in online settings is a longstanding challenge. While exercises and chat-based agents can evaluate understanding in limited contexts, this challenge is magnified when learners seek explanations that delve into procedural knowledge (how things are done) and reasoning (why things happen). We hypothesize that an intelligent agent's ability to understand and explain learners' questions about skills can be significantly enhanced using the TMK (Task-Method-Knowledge) model, a Knowledge-based AI framework. We introduce Ivy, an intelligent agent that leverages an LLM and iterative refinement techniques to generate explanations that embody teleological, causal, and compositional principles. Our initial evaluation demonstrates that this approach goes beyond the typical shallow responses produced by an agent with access to unstructured text, thereby substantially improving the depth and relevance of feedback. This can potentially ensure learners develop a comprehensive understanding of skills crucial for effective problem-solving in online environments. 

---
# Better Decisions through the Right Causal World Model 

**Authors**: Elisabeth Dillies, Quentin Delfosse, Jannis Blüml, Raban Emunds, Florian Peter Busch, Kristian Kersting  

**Link**: [PDF](https://arxiv.org/pdf/2504.07257)  

**Abstract**: Reinforcement learning (RL) agents have shown remarkable performances in various environments, where they can discover effective policies directly from sensory inputs. However, these agents often exploit spurious correlations in the training data, resulting in brittle behaviours that fail to generalize to new or slightly modified environments. To address this, we introduce the Causal Object-centric Model Extraction Tool (COMET), a novel algorithm designed to learn the exact interpretable causal world models (CWMs). COMET first extracts object-centric state descriptions from observations and identifies the environment's internal states related to the depicted objects' properties. Using symbolic regression, it models object-centric transitions and derives causal relationships governing object dynamics. COMET further incorporates large language models (LLMs) for semantic inference, annotating causal variables to enhance interpretability.
By leveraging these capabilities, COMET constructs CWMs that align with the true causal structure of the environment, enabling agents to focus on task-relevant features. The extracted CWMs mitigate the danger of shortcuts, permitting the development of RL systems capable of better planning and decision-making across dynamic scenarios. Our results, validated in Atari environments such as Pong and Freeway, demonstrate the accuracy and robustness of COMET, highlighting its potential to bridge the gap between object-centric reasoning and causal inference in reinforcement learning. 

---
# SpecReason: Fast and Accurate Inference-Time Compute via Speculative Reasoning 

**Authors**: Rui Pan, Yinwei Dai, Zhihao Zhang, Gabriele Oliaro, Zhihao Jia, Ravi Netravali  

**Link**: [PDF](https://arxiv.org/pdf/2504.07891)  

**Abstract**: Recent advances in inference-time compute have significantly improved performance on complex tasks by generating long chains of thought (CoTs) using Large Reasoning Models (LRMs). However, this improved accuracy comes at the cost of high inference latency due to the length of generated reasoning sequences and the autoregressive nature of decoding. Our key insight in tackling these overheads is that LRM inference, and the reasoning that it embeds, is highly tolerant of approximations: complex tasks are typically broken down into simpler steps, each of which brings utility based on the semantic insight it provides for downstream steps rather than the exact tokens it generates. Accordingly, we introduce SpecReason, a system that automatically accelerates LRM inference by using a lightweight model to (speculatively) carry out simpler intermediate reasoning steps and reserving the costly base model only to assess (and potentially correct) the speculated outputs. Importantly, SpecReason's focus on exploiting the semantic flexibility of thinking tokens in preserving final-answer accuracy is complementary to prior speculation techniques, most notably speculative decoding, which demands token-level equivalence at each step. Across a variety of reasoning benchmarks, SpecReason achieves 1.5-2.5$\times$ speedup over vanilla LRM inference while improving accuracy by 1.0-9.9\%. Compared to speculative decoding without SpecReason, their combination yields an additional 19.4-44.2\% latency reduction. We open-source SpecReason at this https URL. 

---
# Why We Feel: Breaking Boundaries in Emotional Reasoning with Multimodal Large Language Models 

**Authors**: Yuxiang Lin, Jingdong Sun, Zhi-Qi Cheng, Jue Wang, Haomin Liang, Zebang Cheng, Yifei Dong, Jun-Yan He, Xiaojiang Peng, Xian-Sheng Hua  

**Link**: [PDF](https://arxiv.org/pdf/2504.07521)  

**Abstract**: Most existing emotion analysis emphasizes which emotion arises (e.g., happy, sad, angry) but neglects the deeper why. We propose Emotion Interpretation (EI), focusing on causal factors-whether explicit (e.g., observable objects, interpersonal interactions) or implicit (e.g., cultural context, off-screen events)-that drive emotional responses. Unlike traditional emotion recognition, EI tasks require reasoning about triggers instead of mere labeling. To facilitate EI research, we present EIBench, a large-scale benchmark encompassing 1,615 basic EI samples and 50 complex EI samples featuring multifaceted emotions. Each instance demands rationale-based explanations rather than straightforward categorization. We further propose a Coarse-to-Fine Self-Ask (CFSA) annotation pipeline, which guides Vision-Language Models (VLLMs) through iterative question-answer rounds to yield high-quality labels at scale. Extensive evaluations on open-source and proprietary large language models under four experimental settings reveal consistent performance gaps-especially for more intricate scenarios-underscoring EI's potential to enrich empathetic, context-aware AI applications. Our benchmark and methods are publicly available at: this https URL, offering a foundation for advanced multimodal causal analysis and next-generation affective computing. 

---
# SF2T: Self-supervised Fragment Finetuning of Video-LLMs for Fine-Grained Understanding 

**Authors**: Yangliu Hu, Zikai Song, Na Feng, Yawei Luo, Junqing Yu, Yi-Ping Phoebe Chen, Wei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.07745)  

**Abstract**: Video-based Large Language Models (Video-LLMs) have witnessed substantial advancements in recent years, propelled by the advancement in multi-modal LLMs. Although these models have demonstrated proficiency in providing the overall description of videos, they struggle with fine-grained understanding, particularly in aspects such as visual dynamics and video details inquiries. To tackle these shortcomings, we find that fine-tuning Video-LLMs on self-supervised fragment tasks, greatly improve their fine-grained video understanding abilities. Hence we propose two key contributions:(1) Self-Supervised Fragment Fine-Tuning (SF$^2$T), a novel effortless fine-tuning method, employs the rich inherent characteristics of videos for training, while unlocking more fine-grained understanding ability of Video-LLMs. Moreover, it relieves researchers from labor-intensive annotations and smartly circumvents the limitations of natural language, which often fails to capture the complex spatiotemporal variations in videos; (2) A novel benchmark dataset, namely FineVidBench, for rigorously assessing Video-LLMs' performance at both the scene and fragment levels, offering a comprehensive evaluation of their capabilities. We assessed multiple models and validated the effectiveness of SF$^2$T on them. Experimental results reveal that our approach improves their ability to capture and interpret spatiotemporal details. 

---
# GPT Carry-On: Training Foundation Model for Customization Could Be Simple, Scalable and Affordable 

**Authors**: Jianqiao Wangni  

**Link**: [PDF](https://arxiv.org/pdf/2504.07513)  

**Abstract**: Modern large language foundation models (LLM) have now entered the daily lives of millions of users. We ask a natural question whether it is possible to customize LLM for every user or every task. From system and industrial economy consideration, general continue-training or fine-tuning still require substantial computation and memory of training GPU nodes, whereas most inference nodes under deployment, possibly with lower-end GPUs, are configured to make forward pass fastest possible. We propose a framework to take full advantages of existing LLMs and systems of online service. We train an additional branch of transformer blocks on the final-layer embedding of pretrained LLMs, which is the base, then a carry-on module merge the base models to compose a customized LLM. We can mix multiple layers, or multiple LLMs specialized in different domains such as chat, coding, math, to form a new mixture of LLM that best fit a new task. As the base model don't need to update parameters, we are able to outsource most computation of the training job on inference nodes, and only train a lightweight carry-on on training nodes, where we consume less than 1GB GPU memory to train a 100M carry-on layer on 30B LLM. We tested Qwen and DeepSeek opensourced models for continue-pretraining and got faster loss convergence. We use it to improve solving math questions with extremely small computation and model size, with 1000 data samples of chain-of-thoughts, and as small as 1 MB parameters of two layer layer carry-on, and the results are promising. 

---
# Enhancements for Developing a Comprehensive AI Fairness Assessment Standard 

**Authors**: Avinash Agarwal, Mayashankar Kumar, Manisha J. Nene  

**Link**: [PDF](https://arxiv.org/pdf/2504.07516)  

**Abstract**: As AI systems increasingly influence critical sectors like telecommunications, finance, healthcare, and public services, ensuring fairness in decision-making is essential to prevent biased or unjust outcomes that disproportionately affect vulnerable entities or result in adverse impacts. This need is particularly pressing as the industry approaches the 6G era, where AI will drive complex functions like autonomous network management and hyper-personalized services. The TEC Standard for Fairness Assessment and Rating of AI Systems provides guidelines for evaluating fairness in AI, focusing primarily on tabular data and supervised learning models. However, as AI applications diversify, this standard requires enhancement to strengthen its impact and broaden its applicability. This paper proposes an expansion of the TEC Standard to include fairness assessments for images, unstructured text, and generative AI, including large language models, ensuring a more comprehensive approach that keeps pace with evolving AI technologies. By incorporating these dimensions, the enhanced framework will promote responsible and trustworthy AI deployment across various sectors. 

---
# ReXCL: A Tool for Requirement Document Extraction and Classification 

**Authors**: Paheli Bhattacharya, Manojit Chakraborty, Santhosh Kumar Arumugam, Rishabh Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2504.07562)  

**Abstract**: This paper presents the ReXCL tool, which automates the extraction and classification processes in requirement engineering, enhancing the software development lifecycle. The tool features two main modules: Extraction, which processes raw requirement documents into a predefined schema using heuristics and predictive modeling, and Classification, which assigns class labels to requirements using adaptive fine-tuning of encoder-based models. The final output can be exported to external requirement engineering tools. Performance evaluations indicate that ReXCL significantly improves efficiency and accuracy in managing requirements, marking a novel approach to automating the schematization of semi-structured requirement documents. 

---
# PR-Attack: Coordinated Prompt-RAG Attacks on Retrieval-Augmented Generation in Large Language Models via Bilevel Optimization 

**Authors**: Yang Jiao, Xiaodong Wang, Kai Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.07717)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across a wide range of applications, e.g., medical question-answering, mathematical sciences, and code generation. However, they also exhibit inherent limitations, such as outdated knowledge and susceptibility to hallucinations. Retrieval-Augmented Generation (RAG) has emerged as a promising paradigm to address these issues, but it also introduces new vulnerabilities. Recent efforts have focused on the security of RAG-based LLMs, yet existing attack methods face three critical challenges: (1) their effectiveness declines sharply when only a limited number of poisoned texts can be injected into the knowledge database, (2) they lack sufficient stealth, as the attacks are often detectable by anomaly detection systems, which compromises their effectiveness, and (3) they rely on heuristic approaches to generate poisoned texts, lacking formal optimization frameworks and theoretic guarantees, which limits their effectiveness and applicability. To address these issues, we propose coordinated Prompt-RAG attack (PR-attack), a novel optimization-driven attack that introduces a small number of poisoned texts into the knowledge database while embedding a backdoor trigger within the prompt. When activated, the trigger causes the LLM to generate pre-designed responses to targeted queries, while maintaining normal behavior in other contexts. This ensures both high effectiveness and stealth. We formulate the attack generation process as a bilevel optimization problem leveraging a principled optimization framework to develop optimal poisoned texts and triggers. Extensive experiments across diverse LLMs and datasets demonstrate the effectiveness of PR-Attack, achieving a high attack success rate even with a limited number of poisoned texts and significantly improved stealth compared to existing methods. 

---
# Automating quantum feature map design via large language models 

**Authors**: Kenya Sakka, Kosuke Mitarai, Keisuke Fujii  

**Link**: [PDF](https://arxiv.org/pdf/2504.07396)  

**Abstract**: Quantum feature maps are a key component of quantum machine learning, encoding classical data into quantum states to exploit the expressive power of high-dimensional Hilbert spaces. Despite their theoretical promise, designing quantum feature maps that offer practical advantages over classical methods remains an open challenge. In this work, we propose an agentic system that autonomously generates, evaluates, and refines quantum feature maps using large language models. The system consists of five component: Generation, Storage, Validation, Evaluation, and Review. Using these components, it iteratively improves quantum feature maps. Experiments on the MNIST dataset show that it can successfully discover and refine feature maps without human intervention. The best feature map generated outperforms existing quantum baselines and achieves competitive accuracy compared to classical kernels across MNIST, Fashion-MNIST, and CIFAR-10. Our approach provides a framework for exploring dataset-adaptive quantum features and highlights the potential of LLM-driven automation in quantum algorithm design. 

---
# A Multi-Phase Analysis of Blood Culture Stewardship: Machine Learning Prediction, Expert Recommendation Assessment, and LLM Automation 

**Authors**: Fatemeh Amrollahi, Nicholas Marshall, Fateme Nateghi Haredasht, Kameron C Black, Aydin Zahedivash, Manoj V Maddali, Stephen P. Ma, Amy Chang, MD Phar Stanley C Deresinski, Mary Kane Goldstein, Steven M. Asch, Niaz Banaei, Jonathan H Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.07278)  

**Abstract**: Blood cultures are often over ordered without clear justification, straining healthcare resources and contributing to inappropriate antibiotic use pressures worsened by the global shortage. In study of 135483 emergency department (ED) blood culture orders, we developed machine learning (ML) models to predict the risk of bacteremia using structured electronic health record (EHR) data and provider notes via a large language model (LLM). The structured models AUC improved from 0.76 to 0.79 with note embeddings and reached 0.81 with added diagnosis codes. Compared to an expert recommendation framework applied by human reviewers and an LLM-based pipeline, our ML approach offered higher specificity without compromising sensitivity. The recommendation framework achieved sensitivity 86%, specificity 57%, while the LLM maintained high sensitivity (96%) but over classified negatives, reducing specificity (16%). These findings demonstrate that ML models integrating structured and unstructured data can outperform consensus recommendations, enhancing diagnostic stewardship beyond existing standards of care. 

---
# Large Language Model (LLM) for Software Security: Code Analysis, Malware Analysis, Reverse Engineering 

**Authors**: Hamed Jelodar, Samita Bai, Parisa Hamedi, Hesamodin Mohammadian, Roozbeh Razavi-Far, Ali Ghorbani  

**Link**: [PDF](https://arxiv.org/pdf/2504.07137)  

**Abstract**: Large Language Models (LLMs) have recently emerged as powerful tools in cybersecurity, offering advanced capabilities in malware detection, generation, and real-time monitoring. Numerous studies have explored their application in cybersecurity, demonstrating their effectiveness in identifying novel malware variants, analyzing malicious code structures, and enhancing automated threat analysis. Several transformer-based architectures and LLM-driven models have been proposed to improve malware analysis, leveraging semantic and structural insights to recognize malicious intent more accurately. This study presents a comprehensive review of LLM-based approaches in malware code analysis, summarizing recent advancements, trends, and methodologies. We examine notable scholarly works to map the research landscape, identify key challenges, and highlight emerging innovations in LLM-driven cybersecurity. Additionally, we emphasize the role of static analysis in malware detection, introduce notable datasets and specialized LLM models, and discuss essential datasets supporting automated malware research. This study serves as a valuable resource for researchers and cybersecurity professionals, offering insights into LLM-powered malware detection and defence strategies while outlining future directions for strengthening cybersecurity resilience. 

---
# Zeus: Zero-shot LLM Instruction for Union Segmentation in Multimodal Medical Imaging 

**Authors**: Siyuan Dai, Kai Ye, Guodong Liu, Haoteng Tang, Liang Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2504.07336)  

**Abstract**: Medical image segmentation has achieved remarkable success through the continuous advancement of UNet-based and Transformer-based foundation backbones. However, clinical diagnosis in the real world often requires integrating domain knowledge, especially textual information. Conducting multimodal learning involves visual and text modalities shown as a solution, but collecting paired vision-language datasets is expensive and time-consuming, posing significant challenges. Inspired by the superior ability in numerous cross-modal tasks for Large Language Models (LLMs), we proposed a novel Vision-LLM union framework to address the issues. Specifically, we introduce frozen LLMs for zero-shot instruction generation based on corresponding medical images, imitating the radiology scanning and report generation process. {To better approximate real-world diagnostic processes}, we generate more precise text instruction from multimodal radiology images (e.g., T1-w or T2-w MRI and CT). Based on the impressive ability of semantic understanding and rich knowledge of LLMs. This process emphasizes extracting special features from different modalities and reunion the information for the ultimate clinical diagnostic. With generated text instruction, our proposed union segmentation framework can handle multimodal segmentation without prior collected vision-language datasets. To evaluate our proposed method, we conduct comprehensive experiments with influential baselines, the statistical results and the visualized case study demonstrate the superiority of our novel method.} 

---
