# $\text{R}^2\text{ec}$: Towards Large Recommender Models with Reasoning 

**Authors**: Runyang You, Yongqi Li, Xinyu Lin, Xin Zhang, Wenjie Wang, Wenjie Li, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2505.16994)  

**Abstract**: Large recommender models have extended LLMs as powerful recommenders via encoding or item generation, and recent breakthroughs in LLM reasoning synchronously motivate the exploration of reasoning in recommendation. Current studies usually position LLMs as external reasoning modules to yield auxiliary thought for augmenting conventional recommendation pipelines. However, such decoupled designs are limited in significant resource cost and suboptimal joint optimization. To address these issues, we propose \name, a unified large recommender model with intrinsic reasoning capabilities. Initially, we reconceptualize the model architecture to facilitate interleaved reasoning and recommendation in the autoregressive process. Subsequently, we propose RecPO, a corresponding reinforcement learning framework that optimizes \name\ both the reasoning and recommendation capabilities simultaneously in a single policy update; RecPO introduces a fused reward scheme that solely leverages recommendation labels to simulate the reasoning capability, eliminating dependency on specialized reasoning annotations. Experiments on three datasets with various baselines verify the effectiveness of \name, showing relative improvements of 68.67\% in Hit@5 and 45.21\% in NDCG@20. Code available at this https URL. 

---
# Fixing Data That Hurts Performance: Cascading LLMs to Relabel Hard Negatives for Robust Information Retrieval 

**Authors**: Nandan Thakur, Crystina Zhang, Xueguang Ma, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.16967)  

**Abstract**: Training robust retrieval and reranker models typically relies on large-scale retrieval datasets; for example, the BGE collection contains 1.6 million query-passage pairs sourced from various data sources. However, we find that certain datasets can negatively impact model effectiveness -- pruning 8 out of 15 datasets from the BGE collection reduces the training set size by 2.35$\times$ and increases nDCG@10 on BEIR by 1.0 point. This motivates a deeper examination of training data quality, with a particular focus on "false negatives", where relevant passages are incorrectly labeled as irrelevant. We propose a simple, cost-effective approach using cascading LLM prompts to identify and relabel hard negatives. Experimental results show that relabeling false negatives with true positives improves both E5 (base) and Qwen2.5-7B retrieval models by 0.7-1.4 nDCG@10 on BEIR and by 1.7-1.8 nDCG@10 on zero-shot AIR-Bench evaluation. Similar gains are observed for rerankers fine-tuned on the relabeled data, such as Qwen2.5-3B on BEIR. The reliability of the cascading design is further supported by human annotation results, where we find judgment by GPT-4o shows much higher agreement with humans than GPT-4o-mini. 

---
# Don't "Overthink" Passage Reranking: Is Reasoning Truly Necessary? 

**Authors**: Nour Jedidi, Yung-Sung Chuang, James Glass, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.16886)  

**Abstract**: With the growing success of reasoning models across complex natural language tasks, researchers in the Information Retrieval (IR) community have begun exploring how similar reasoning capabilities can be integrated into passage rerankers built on Large Language Models (LLMs). These methods typically employ an LLM to produce an explicit, step-by-step reasoning process before arriving at a final relevance prediction. But, does reasoning actually improve reranking accuracy? In this paper, we dive deeper into this question, studying the impact of the reasoning process by comparing reasoning-based pointwise rerankers (ReasonRR) to standard, non-reasoning pointwise rerankers (StandardRR) under identical training conditions, and observe that StandardRR generally outperforms ReasonRR. Building on this observation, we then study the importance of reasoning to ReasonRR by disabling its reasoning process (ReasonRR-NoReason), and find that ReasonRR-NoReason is surprisingly more effective than ReasonRR. Examining the cause of this result, our findings reveal that reasoning-based rerankers are limited by the LLM's reasoning process, which pushes it toward polarized relevance scores and thus fails to consider the partial relevance of passages, a key factor for the accuracy of pointwise rerankers. 

---
# LARES: Latent Reasoning for Sequential Recommendation 

**Authors**: Enze Liu, Bowen Zheng, Xiaolei Wang, Wayne Xin Zhao, Jinpeng Wang, Sheng Chen, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2505.16865)  

**Abstract**: Sequential recommender systems have become increasingly important in real-world applications that model user behavior sequences to predict their preferences. However, existing sequential recommendation methods predominantly rely on non-reasoning paradigms, which may limit the model's computational capacity and result in suboptimal recommendation performance. To address these limitations, we present LARES, a novel and scalable LAtent REasoning framework for Sequential recommendation that enhances model's representation capabilities through increasing the computation density of parameters by depth-recurrent latent reasoning. Our proposed approach employs a recurrent architecture that allows flexible expansion of reasoning depth without increasing parameter complexity, thereby effectively capturing dynamic and intricate user interest patterns. A key difference of LARES lies in refining all input tokens at each implicit reasoning step to improve the computation utilization. To fully unlock the model's reasoning potential, we design a two-phase training strategy: (1) Self-supervised pre-training (SPT) with dual alignment objectives; (2) Reinforcement post-training (RPT). During the first phase, we introduce trajectory-level alignment and step-level alignment objectives, which enable the model to learn recommendation-oriented latent reasoning patterns without requiring supplementary annotated data. The subsequent phase utilizes reinforcement learning (RL) to harness the model's exploratory ability, further refining its reasoning capabilities. Comprehensive experiments on real-world benchmarks demonstrate our framework's superior performance. Notably, LARES exhibits seamless compatibility with existing advanced models, further improving their recommendation performance. 

---
# Walk&Retrieve: Simple Yet Effective Zero-shot Retrieval-Augmented Generation via Knowledge Graph Walks 

**Authors**: Martin Böckling, Heiko Paulheim, Andreea Iana  

**Link**: [PDF](https://arxiv.org/pdf/2505.16849)  

**Abstract**: Large Language Models (LLMs) have showcased impressive reasoning abilities, but often suffer from hallucinations or outdated knowledge. Knowledge Graph (KG)-based Retrieval-Augmented Generation (RAG) remedies these shortcomings by grounding LLM responses in structured external information from a knowledge base. However, many KG-based RAG approaches struggle with (i) aligning KG and textual representations, (ii) balancing retrieval accuracy and efficiency, and (iii) adapting to dynamically updated KGs. In this work, we introduce Walk&Retrieve, a simple yet effective KG-based framework that leverages walk-based graph traversal and knowledge verbalization for corpus generation for zero-shot RAG. Built around efficient KG walks, our method does not require fine-tuning on domain-specific data, enabling seamless adaptation to KG updates, reducing computational overhead, and allowing integration with any off-the-shelf backbone LLM. Despite its simplicity, Walk&Retrieve performs competitively, often outperforming existing RAG systems in response accuracy and hallucination reduction. Moreover, it demonstrates lower query latency and robust scalability to large KGs, highlighting the potential of lightweight retrieval strategies as strong baselines for future RAG research. 

---
# DeepRec: Towards a Deep Dive Into the Item Space with Large Language Model Based Recommendation 

**Authors**: Bowen Zheng, Xiaolei Wang, Enze Liu, Xi Wang, Lu Hongyu, Yu Chen, Wayne Xin Zhao, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2505.16810)  

**Abstract**: Recently, large language models (LLMs) have been introduced into recommender systems (RSs), either to enhance traditional recommendation models (TRMs) or serve as recommendation backbones. However, existing LLM-based RSs often do not fully exploit the complementary advantages of LLMs (e.g., world knowledge and reasoning) and TRMs (e.g., recommendation-specific knowledge and efficiency) to fully explore the item space. To address this, we propose DeepRec, a novel LLM-based RS that enables autonomous multi-turn interactions between LLMs and TRMs for deep exploration of the item space. In each interaction turn, LLMs reason over user preferences and interact with TRMs to retrieve candidate items. After multi-turn interactions, LLMs rank the retrieved items to generate the final recommendations. We adopt reinforcement learning(RL) based optimization and propose novel designs from three aspects: recommendation model based data rollout, recommendation-oriented hierarchical rewards, and a two-stage RL training strategy. For data rollout, we introduce a preference-aware TRM, with which LLMs interact to construct trajectory data. For rewards, we design a hierarchical reward function that involves both process-level and outcome-level rewards to optimize the interaction process and recommendation performance, respectively. For RL training, we develop a two-stage training strategy, where the first stage aims to guide LLMs to interact with TRMs and the second stage focuses on performance improvement. Experiments on public datasets demonstrate that DeepRec significantly outperforms both traditional and LLM-based baselines, offering a new paradigm for deep exploration in recommendation systems. 

---
# Action is All You Need: Dual-Flow Generative Ranking Network for Recommendation 

**Authors**: Hao Guo, Erpeng Xue, Lei Huang, Shichao Wang, Xiaolei Wang, Lei Wang, Jinpeng Wang, Sheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.16752)  

**Abstract**: We introduce the Dual-Flow Generative Ranking Network (DFGR), a two-stream architecture designed for recommendation systems. DFGR integrates innovative interaction patterns between real and fake flows within the QKV modules of the self-attention mechanism, enhancing both training and inference efficiency. This approach effectively addresses a key limitation observed in Meta's proposed HSTU generative recommendation approach, where heterogeneous information volumes are mapped into identical vector spaces, leading to training instability. Unlike traditional recommendation models, DFGR only relies on user history behavior sequences and minimal attribute information, eliminating the need for extensive manual feature engineering. Comprehensive evaluations on open-source and industrial datasets reveal DFGR's superior performance compared to established baselines such as DIN, DCN, DIEN, and DeepFM. We also investigate optimal parameter allocation strategies under computational constraints, establishing DFGR as an efficient and effective next-generation generate ranking paradigm. 

---
# A Novel Generative Model with Causality Constraint for Mitigating Biases in Recommender Systems 

**Authors**: Jianfeng Deng, Qingfeng Chen, Debo Cheng, Jiuyong Li, Lin Liu, Shichao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16708)  

**Abstract**: Accurately predicting counterfactual user feedback is essential for building effective recommender systems. However, latent confounding bias can obscure the true causal relationship between user feedback and item exposure, ultimately degrading recommendation performance. Existing causal debiasing approaches often rely on strong assumptions-such as the availability of instrumental variables (IVs) or strong correlations between latent confounders and proxy variables-that are rarely satisfied in real-world scenarios. To address these limitations, we propose a novel generative framework called Latent Causality Constraints for Debiasing representation learning in Recommender Systems (LCDR). Specifically, LCDR leverages an identifiable Variational Autoencoder (iVAE) as a causal constraint to align the latent representations learned by a standard Variational Autoencoder (VAE) through a unified loss function. This alignment allows the model to leverage even weak or noisy proxy variables to recover latent confounders effectively. The resulting representations are then used to improve recommendation performance. Extensive experiments on three real-world datasets demonstrate that LCDR consistently outperforms existing methods in both mitigating bias and improving recommendation accuracy. 

---
# MDVT: Enhancing Multimodal Recommendation with Model-Agnostic Multimodal-Driven Virtual Triplets 

**Authors**: Jinfeng Xu, Zheyu Chen, Jinze Li, Shuo Yang, Hewei Wang, Yijie Li, Mengran Li, Puzhen Wu, Edith C. H. Ngai  

**Link**: [PDF](https://arxiv.org/pdf/2505.16665)  

**Abstract**: The data sparsity problem significantly hinders the performance of recommender systems, as traditional models rely on limited historical interactions to learn user preferences and item properties. While incorporating multimodal information can explicitly represent these preferences and properties, existing works often use it only as side information, failing to fully leverage its potential. In this paper, we propose MDVT, a model-agnostic approach that constructs multimodal-driven virtual triplets to provide valuable supervision signals, effectively mitigating the data sparsity problem in multimodal recommendation systems. To ensure high-quality virtual triplets, we introduce three tailored warm-up threshold strategies: static, dynamic, and hybrid. The static warm-up threshold strategy exhaustively searches for the optimal number of warm-up epochs but is time-consuming and computationally intensive. The dynamic warm-up threshold strategy adjusts the warm-up period based on loss trends, improving efficiency but potentially missing optimal performance. The hybrid strategy combines both, using the dynamic strategy to find the approximate optimal number of warm-up epochs and then refining it with the static strategy in a narrow hyper-parameter space. Once the warm-up threshold is satisfied, the virtual triplets are used for joint model optimization by our enhanced pair-wise loss function without causing significant gradient skew. Extensive experiments on multiple real-world datasets demonstrate that integrating MDVT into advanced multimodal recommendation models effectively alleviates the data sparsity problem and improves recommendation performance, particularly in sparse data scenarios. 

---
# MiLQ: Benchmarking IR Models for Bilingual Web Search with Mixed Language Queries 

**Authors**: Jonghwi Kim, Deokhyung Kang, Seonjeong Hwang, Yunsu Kim, Jungseul Ok, Gary Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.16631)  

**Abstract**: Despite bilingual speakers frequently using mixed-language queries in web searches, Information Retrieval (IR) research on them remains scarce. To address this, we introduce MiLQ,Mixed-Language Query test set, the first public benchmark of mixed-language queries, confirmed as realistic and highly preferred. Experiments show that multilingual IR models perform moderately on MiLQ and inconsistently across native, English, and mixed-language queries, also suggesting code-switched training data's potential for robust IR models handling such queries. Meanwhile, intentional English mixing in queries proves an effective strategy for bilinguals searching English documents, which our analysis attributes to enhanced token matching compared to native queries. 

---
# Causal-Invariant Cross-Domain Out-of-Distribution Recommendation 

**Authors**: Jiajie Zhu, Yan Wang, Feng Zhu, Pengfei Ding, Hongyang Liu, Zhu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.16532)  

**Abstract**: Cross-Domain Recommendation (CDR) aims to leverage knowledge from a relatively data-richer source domain to address the data sparsity problem in a relatively data-sparser target domain. While CDR methods need to address the distribution shifts between different domains, i.e., cross-domain distribution shifts (CDDS), they typically assume independent and identical distribution (IID) between training and testing data within the target domain. However, this IID assumption rarely holds in real-world scenarios due to single-domain distribution shift (SDDS). The above two co-existing distribution shifts lead to out-of-distribution (OOD) environments that hinder effective knowledge transfer and generalization, ultimately degrading recommendation performance in CDR. To address these co-existing distribution shifts, we propose a novel Causal-Invariant Cross-Domain Out-of-distribution Recommendation framework, called CICDOR. In CICDOR, we first learn dual-level causal structures to infer domain-specific and domain-shared causal-invariant user preferences for tackling both CDDS and SDDS under OOD environments in CDR. Then, we propose an LLM-guided confounder discovery module that seamlessly integrates LLMs with a conventional causal discovery method to extract observed confounders for effective deconfounding, thereby enabling accurate causal-invariant preference inference. Extensive experiments on two real-world datasets demonstrate the superior recommendation accuracy of CICDOR over state-of-the-art methods across various OOD scenarios. 

---
# Utilizing citation index and synthetic quality measure to compare Wikipedia languages across various topics 

**Authors**: Włodzimierz Lewoniewski, Krzysztof Węcel, Witold Abramowicz  

**Link**: [PDF](https://arxiv.org/pdf/2505.16506)  

**Abstract**: This study presents a comparative analysis of 55 Wikipedia language editions employing a citation index alongside a synthetic quality measure. Specifically, we identified the most significant Wikipedia articles within distinct topical areas, selecting the top 10, top 25, and top 100 most cited articles in each topic and language version. This index was built on the basis of wikilinks between Wikipedia articles in each language version and in order to do that we processed 6.6 billion page-to-page link records. Next, we used a quality score for each Wikipedia article - a synthetic measure scaled from 0 to 100. This approach enabled quality comparison of Wikipedia articles even between language versions with different quality grading schemes. Our results highlight disparities among Wikipedia language editions, revealing strengths and gaps in content coverage and quality across topics. 

---
# Benchmarking Retrieval-Augmented Multimomal Generation for Document Question Answering 

**Authors**: Kuicai Dong, Yujing Chang, Shijie Huang, Yasheng Wang, Ruiming Tang, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.16470)  

**Abstract**: Document Visual Question Answering (DocVQA) faces dual challenges in processing lengthy multimodal documents (text, images, tables) and performing cross-modal reasoning. Current document retrieval-augmented generation (DocRAG) methods remain limited by their text-centric approaches, frequently missing critical visual information. The field also lacks robust benchmarks for assessing multimodal evidence selection and integration. We introduce MMDocRAG, a comprehensive benchmark featuring 4,055 expert-annotated QA pairs with multi-page, cross-modal evidence chains. Our framework introduces innovative metrics for evaluating multimodal quote selection and enables answers that interleave text with relevant visual elements. Through large-scale experiments with 60 VLM/LLM models and 14 retrieval systems, we identify persistent challenges in multimodal evidence retrieval, selection, and this http URL findings reveal advanced proprietary LVMs show superior performance than open-sourced alternatives. Also, they show moderate advantages using multimodal inputs over text-only inputs, while open-source alternatives show significant performance degradation. Notably, fine-tuned LLMs achieve substantial improvements when using detailed image descriptions. MMDocRAG establishes a rigorous testing ground and provides actionable insights for developing more robust multimodal DocVQA systems. Our benchmark and code are available at this https URL. 

---
# Conf-GNNRec: Quantifying and Calibrating the Prediction Confidence for GNN-based Recommendation Methods 

**Authors**: Meng Yan, Cai Xu, Xujing Wang, Ziyu Guan, Wei Zhao, Yuhang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.16466)  

**Abstract**: Recommender systems based on graph neural networks perform well in tasks such as rating and ranking. However, in real-world recommendation scenarios, noise such as user misuse and malicious advertisement gradually accumulates through the message propagation mechanism. Even if existing studies mitigate their effects by reducing the noise propagation weights, the severe sparsity of the recommender system still leads to the low-weighted noisy neighbors being mistaken as meaningful information, and the prediction result obtained based on the polluted nodes is not entirely trustworthy. Therefore, it is crucial to measure the confidence of the prediction results in this highly noisy framework. Furthermore, our evaluation of the existing representative GNN-based recommendation shows that it suffers from overconfidence. Based on the above considerations, we propose a new method to quantify and calibrate the prediction confidence of GNN-based recommendations (Conf-GNNRec). Specifically, we propose a rating calibration method that dynamically adjusts excessive ratings to mitigate overconfidence based on user personalization. We also design a confidence loss function to reduce the overconfidence of negative samples and effectively improve recommendation performance. Experiments on public datasets demonstrate the validity of Conf-GNNRec in prediction confidence and recommendation performance. 

---
# Chain-of-Thought Poisoning Attacks against R1-based Retrieval-Augmented Generation Systems 

**Authors**: Hongru Song, Yu-an Liu, Ruqing Zhang, Jiafeng Guo, Yixing Fan  

**Link**: [PDF](https://arxiv.org/pdf/2505.16367)  

**Abstract**: Retrieval-augmented generation (RAG) systems can effectively mitigate the hallucination problem of large language models (LLMs),but they also possess inherent vulnerabilities. Identifying these weaknesses before the large-scale real-world deployment of RAG systems is of great importance, as it lays the foundation for building more secure and robust RAG systems in the future. Existing adversarial attack methods typically exploit knowledge base poisoning to probe the vulnerabilities of RAG systems, which can effectively deceive standard RAG models. However, with the rapid advancement of deep reasoning capabilities in modern LLMs, previous approaches that merely inject incorrect knowledge are inadequate when attacking RAG systems equipped with deep reasoning abilities. Inspired by the deep thinking capabilities of LLMs, this paper extracts reasoning process templates from R1-based RAG systems, uses these templates to wrap erroneous knowledge into adversarial documents, and injects them into the knowledge base to attack RAG systems. The key idea of our approach is that adversarial documents, by simulating the chain-of-thought patterns aligned with the model's training signals, may be misinterpreted by the model as authentic historical reasoning processes, thus increasing their likelihood of being referenced. Experiments conducted on the MS MARCO passage ranking dataset demonstrate the effectiveness of our proposed method. 

---
# Flow Matching based Sequential Recommender Model 

**Authors**: Feng Liu, Lixin Zou, Xiangyu Zhao, Min Tang, Liming Dong, Dan Luo, Xiangyang Luo, Chenliang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.16298)  

**Abstract**: Generative models, particularly diffusion model, have emerged as powerful tools for sequential recommendation. However, accurately modeling user preferences remains challenging due to the noise perturbations inherent in the forward and reverse processes of diffusion-based methods. Towards this end, this study introduces FMRec, a Flow Matching based model that employs a straight flow trajectory and a modified loss tailored for the recommendation task. Additionally, from the diffusion-model perspective, we integrate a reconstruction loss to improve robustness against noise perturbations, thereby retaining user preferences during the forward process. In the reverse process, we employ a deterministic reverse sampler, specifically an ODE-based updating function, to eliminate unnecessary randomness, thereby ensuring that the generated recommendations closely align with user needs. Extensive evaluations on four benchmark datasets reveal that FMRec achieves an average improvement of 6.53% over state-of-the-art methods. The replication code is available at this https URL. 

---
# HASH-RAG: Bridging Deep Hashing with Retriever for Efficient, Fine Retrieval and Augmented Generation 

**Authors**: Jinyu Guo, Xunlei Chen, Qiyang Xia, Zhaokun Wang, Jie Ou, Libo Qin, Shunyu Yao, Wenhong Tian  

**Link**: [PDF](https://arxiv.org/pdf/2505.16133)  

**Abstract**: Retrieval-Augmented Generation (RAG) encounters efficiency challenges when scaling to massive knowledge bases while preserving contextual relevance. We propose Hash-RAG, a framework that integrates deep hashing techniques with systematic optimizations to address these limitations. Our queries directly learn binary hash codes from knowledgebase code, eliminating intermediate feature extraction steps, and significantly reducing storage and computational overhead. Building upon this hash-based efficient retrieval framework, we establish the foundation for fine-grained chunking. Consequently, we design a Prompt-Guided Chunk-to-Context (PGCC) module that leverages retrieved hash-indexed propositions and their original document segments through prompt engineering to enhance the LLM's contextual awareness. Experimental evaluations on NQ, TriviaQA, and HotpotQA datasets demonstrate that our approach achieves a 90% reduction in retrieval time compared to conventional methods while maintaining considerate recall performance. Additionally, The proposed system outperforms retrieval/non-retrieval baselines by 1.4-4.3% in EM scores. 

---
# Emotion-based Recommender System 

**Authors**: Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16121)  

**Abstract**: Recommender system is one of the most critical technologies for large internet companies such as Amazon and TikTok. Although millions of users use recommender systems globally everyday, and indeed, much data analysis work has been done to improve the technical accuracy of the system, to our limited knowledge, there has been little attention paid to analysis of users' emotion in recommender systems. In this paper, we create a new theory and metrics that could capture users' emotion when they are interacting with recommender systems. We also provide effective and efficient visualization techniques for visualization of users' emotion and its change in the customers' lifetime cycle. In the end, we design a framework for emotion-based recommendation algorithms, illustrated in a straightforward example with experimental results to demonstrate the effectiveness of our new theory. 

---
# Aug2Search: Enhancing Facebook Marketplace Search with LLM-Generated Synthetic Data Augmentation 

**Authors**: Ruijie Xi, He Ba, Hao Yuan, Rishu Agrawal, Arul Prakash  

**Link**: [PDF](https://arxiv.org/pdf/2505.16065)  

**Abstract**: Embedding-Based Retrieval (EBR) is an important technique in modern search engines, enabling semantic match between search queries and relevant results. However, search logging data on platforms like Facebook Marketplace lacks the diversity and details needed for effective EBR model training, limiting the models' ability to capture nuanced search patterns. To address this challenge, we propose Aug2Search, an EBR-based framework leveraging synthetic data generated by Generative AI (GenAI) models, in a multimodal and multitask approach to optimize query-product relevance. This paper investigates the capabilities of GenAI, particularly Large Language Models (LLMs), in generating high-quality synthetic data, and analyzing its impact on enhancing EBR models. We conducted experiments using eight Llama models and 100 million data points from Facebook Marketplace logs. Our synthetic data generation follows three strategies: (1) generate queries, (2) enhance product listings, and (3) generate queries from enhanced listings. We train EBR models on three different datasets: sampled engagement data or original data ((e.g., "Click" and "Listing Interactions")), synthetic data, and a mixture of both engagement and synthetic data to assess their performance across various training sets. Our findings underscore the robustness of Llama models in producing synthetic queries and listings with high coherence, relevance, and diversity, while maintaining low levels of hallucination. Aug2Search achieves an improvement of up to 4% in ROC_AUC with 100 million synthetic data samples, demonstrating the effectiveness of our approach. Moreover, our experiments reveal that with the same volume of training data, models trained exclusively on synthetic data often outperform those trained on original data only or a mixture of original and synthetic data. 

---
# Text-to-Pipeline: Bridging Natural Language and Data Preparation Pipelines 

**Authors**: Yuhang Ge, Yachuan Liu, Yuren Mao, Yunjun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2505.15874)  

**Abstract**: Data preparation (DP) transforms raw data into a form suitable for downstream applications, typically by composing operations into executable pipelines. Building such pipelines is time-consuming and requires sophisticated programming skills. If we can build the pipelines with natural language (NL), the technical barrier of DP will be significantly reduced. However, constructing DP pipelines from NL instructions remains underexplored. To fill the gap, we introduce Text-to-Pipeline, a new task that translates NL data preparation instructions into DP pipelines. Furthermore, we develop a benchmark named PARROT to support systematic evaluation. To simulate realistic DP scenarios, we mined transformation patterns from production pipelines and instantiated them on 23,009 real-world tables collected from six public sources. The resulting benchmark comprises ~18,000 pipelines covering 16 core DP operators. We evaluated cutting-edge large language models on PARROTand observed that they only solved 72.86% of the cases, revealing notable limitations in instruction understanding and multi-step reasoning. To address this, we propose Pipeline-Agent, a stronger baseline that iteratively predicts and executes operations with intermediate table feedback, achieving the best performance of 76.17%. Despite this improvement, there remains substantial room for progress on Text-to-Pipeline. Our data, codes, and evaluation tools are available at this https URL. 

---
# InfoDeepSeek: Benchmarking Agentic Information Seeking for Retrieval-Augmented Generation 

**Authors**: Yunjia Xi, Jianghao Lin, Menghui Zhu, Yongzhao Xiao, Zhuoying Ou, Jiaqi Liu, Tong Wan, Bo Chen, Weiwen Liu, Yasheng Wang, Ruiming Tang, Weinan Zhang, Yong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.15872)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by grounding responses with retrieved information. As an emerging paradigm, Agentic RAG further enhances this process by introducing autonomous LLM agents into the information seeking process. However, existing benchmarks fall short in evaluating such systems, as they are confined to a static retrieval environment with a fixed, limited corpus} and simple queries that fail to elicit agentic behavior. Moreover, their evaluation protocols assess information seeking effectiveness by pre-defined gold sets of documents, making them unsuitable for the open-ended and dynamic nature of real-world web environments. To bridge this gap, we present InfoDeepSeek, a new benchmark with challenging questions designed for assessing agentic information seeking in real-world, dynamic web environments. We propose a systematic methodology for constructing challenging queries satisfying the criteria of determinacy, difficulty, and diversity. Based on this, we develop the first evaluation framework tailored to dynamic agentic information seeking, including fine-grained metrics about the accuracy, utility, and compactness of information seeking outcomes. Through extensive experiments across LLMs, search engines, and question types, InfoDeepSeek reveals nuanced agent behaviors and offers actionable insights for future research. 

---
# AutoData: A Multi-Agent System for Open Web Data Collection 

**Authors**: Tianyi Ma, Yiyue Qian, Zheyuan Zhang, Zehong Wang, Xiaoye Qian, Feifan Bai, Yifan Ding, Xuwei Luo, Shinan Zhang, Keerthiram Murugesan, Chuxu Zhang, Yanfang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2505.15859)  

**Abstract**: The exponential growth of data-driven systems and AI technologies has intensified the demand for high-quality web-sourced datasets. While existing datasets have proven valuable, conventional web data collection approaches face significant limitations in terms of human effort and scalability. Current data-collecting solutions fall into two categories: wrapper-based methods that struggle with adaptability and reproducibility, and large language model (LLM)-based approaches that incur substantial computational and financial costs. To address these challenges, we propose AutoData, a novel multi-agent system for Automated web Data collection, that requires minimal human intervention, i.e., only necessitating a natural language instruction specifying the desired dataset. In addition, AutoData is designed with a robust multi-agent architecture, featuring a novel oriented message hypergraph coordinated by a central task manager, to efficiently organize agents across research and development squads. Besides, we introduce a novel hypergraph cache system to advance the multi-agent collaboration process that enables efficient automated data collection and mitigates the token cost issues prevalent in existing LLM-based systems. Moreover, we introduce Instruct2DS, a new benchmark dataset supporting live data collection from web sources across three domains: academic, finance, and sports. Comprehensive evaluations over Instruct2DS and three existing benchmark datasets demonstrate AutoData's superior performance compared to baseline methods. Case studies on challenging tasks such as picture book collection and paper extraction from surveys further validate its applicability. Our source code and dataset are available at this https URL. 

---
# DisastIR: A Comprehensive Information Retrieval Benchmark for Disaster Management 

**Authors**: Kai Yin, Xiangjue Dong, Chengkai Liu, Lipai Huang, Yiming Xiao, Zhewei Liu, Ali Mostafavi, James Caverlee  

**Link**: [PDF](https://arxiv.org/pdf/2505.15856)  

**Abstract**: Effective disaster management requires timely access to accurate and contextually relevant information. Existing Information Retrieval (IR) benchmarks, however, focus primarily on general or specialized domains, such as medicine or finance, neglecting the unique linguistic complexity and diverse information needs encountered in disaster management scenarios. To bridge this gap, we introduce DisastIR, the first comprehensive IR evaluation benchmark specifically tailored for disaster management. DisastIR comprises 9,600 diverse user queries and more than 1.3 million labeled query-passage pairs, covering 48 distinct retrieval tasks derived from six search intents and eight general disaster categories that include 301 specific event types. Our evaluations of 30 state-of-the-art retrieval models demonstrate significant performance variances across tasks, with no single model excelling universally. Furthermore, comparative analyses reveal significant performance gaps between general-domain and disaster management-specific tasks, highlighting the necessity of disaster management-specific benchmarks for guiding IR model selection to support effective decision-making in disaster management scenarios. All source codes and DisastIR are available at this https URL. 

---
# R1-Searcher++: Incentivizing the Dynamic Knowledge Acquisition of LLMs via Reinforcement Learning 

**Authors**: Huatong Song, Jinhao Jiang, Wenqing Tian, Zhipeng Chen, Yuhuan Wu, Jiahao Zhao, Yingqian Min, Wayne Xin Zhao, Lei Fang, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2505.17005)  

**Abstract**: Large Language Models (LLMs) are powerful but prone to hallucinations due to static knowledge. Retrieval-Augmented Generation (RAG) helps by injecting external information, but current methods often are costly, generalize poorly, or ignore the internal knowledge of the model. In this paper, we introduce R1-Searcher++, a novel framework designed to train LLMs to adaptively leverage both internal and external knowledge sources. R1-Searcher++ employs a two-stage training strategy: an initial SFT Cold-start phase for preliminary format learning, followed by RL for Dynamic Knowledge Acquisition. The RL stage uses outcome-supervision to encourage exploration, incorporates a reward mechanism for internal knowledge utilization, and integrates a memorization mechanism to continuously assimilate retrieved information, thereby enriching the model's internal knowledge. By leveraging internal knowledge and external search engine, the model continuously improves its capabilities, enabling efficient retrieval-augmented reasoning. Our experiments demonstrate that R1-Searcher++ outperforms previous RAG and reasoning methods and achieves efficient retrieval. The code is available at this https URL. 

---
# SimpleDeepSearcher: Deep Information Seeking via Web-Powered Reasoning Trajectory Synthesis 

**Authors**: Shuang Sun, Huatong Song, Yuhao Wang, Ruiyang Ren, Jinhao Jiang, Junjie Zhang, Fei Bai, Jia Deng, Wayne Xin Zhao, Zheng Liu, Lei Fang, Zhongyuan Wang, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2505.16834)  

**Abstract**: Retrieval-augmented generation (RAG) systems have advanced large language models (LLMs) in complex deep search scenarios requiring multi-step reasoning and iterative information retrieval. However, existing approaches face critical limitations that lack high-quality training trajectories or suffer from the distributional mismatches in simulated environments and prohibitive computational costs for real-world deployment. This paper introduces SimpleDeepSearcher, a lightweight yet effective framework that bridges this gap through strategic data engineering rather than complex training paradigms. Our approach synthesizes high-quality training data by simulating realistic user interactions in live web search environments, coupled with a multi-criteria curation strategy that optimizes the diversity and quality of input and output side. Experiments on five benchmarks across diverse domains demonstrate that SFT on only 871 curated samples yields significant improvements over RL-based baselines. Our work establishes SFT as a viable pathway by systematically addressing the data-scarce bottleneck, offering practical insights for efficient deep search systems. Our code is available at this https URL. 

---
# Two-way Evidence self-Alignment based Dual-Gated Reasoning Enhancement 

**Authors**: Kexin Zhang, Junlan Chen, Daifeng Li, Yuxuan Zhang, Yangyang Feng, Bowen Deng, Weixu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.16806)  

**Abstract**: Large language models (LLMs) encounter difficulties in knowledge-intensive multi-step reasoning (KIMSR) tasks. One challenge is how to effectively extract and represent rationale evidence. The current methods often extract semantically relevant but logically irrelevant evidence, resulting in flawed reasoning and inaccurate responses. We propose a two-way evidence self-alignment (TW-ESA) module, which utilizes the mutual alignment between strict reasoning and LLM reasoning to enhance its understanding of the causal logic of evidence, thereby addressing the first challenge. Another challenge is how to utilize the rationale evidence and LLM's intrinsic knowledge for accurate reasoning when the evidence contains uncertainty. We propose a dual-gated reasoning enhancement (DGR) module to gradually fuse useful knowledge of LLM within strict reasoning, which can enable the model to perform accurate reasoning by focusing on causal elements in the evidence and exhibit greater robustness. The two modules are collaboratively trained in a unified framework ESA-DGR. Extensive experiments on three diverse and challenging KIMSR datasets reveal that ESA-DGR significantly surpasses state-of-the-art LLM-based fine-tuning methods, with remarkable average improvements of 4% in exact match (EM) and 5% in F1 score. The implementation code is available at this https URL. 

---
# Representation Discrepancy Bridging Method for Remote Sensing Image-Text Retrieval 

**Authors**: Hailong Ning, Siying Wang, Tao Lei, Xiaopeng Cao, Huanmin Dou, Bin Zhao, Asoke K. Nandi, Petia Radeva  

**Link**: [PDF](https://arxiv.org/pdf/2505.16756)  

**Abstract**: Remote Sensing Image-Text Retrieval (RSITR) plays a critical role in geographic information interpretation, disaster monitoring, and urban planning by establishing semantic associations between image and textual descriptions. Existing Parameter-Efficient Fine-Tuning (PEFT) methods for Vision-and-Language Pre-training (VLP) models typically adopt symmetric adapter structures for exploring cross-modal correlations. However, the strong discriminative nature of text modality may dominate the optimization process and inhibits image representation learning. The nonnegligible imbalanced cross-modal optimization remains a bottleneck to enhancing the model performance. To address this issue, this study proposes a Representation Discrepancy Bridging (RDB) method for the RSITR task. On the one hand, a Cross-Modal Asymmetric Adapter (CMAA) is designed to enable modality-specific optimization and improve feature alignment. The CMAA comprises a Visual Enhancement Adapter (VEA) and a Text Semantic Adapter (TSA). VEA mines fine-grained image features by Differential Attention (DA) mechanism, while TSA identifies key textual semantics through Hierarchical Attention (HA) mechanism. On the other hand, this study extends the traditional single-task retrieval framework to a dual-task optimization framework and develops a Dual-Task Consistency Loss (DTCL). The DTCL improves cross-modal alignment robustness through an adaptive weighted combination of cross-modal, classification, and exponential moving average consistency constraints. Experiments on RSICD and RSITMD datasets show that the proposed RDB method achieves a 6%-11% improvement in mR metrics compared to state-of-the-art PEFT methods and a 1.15%-2% improvement over the full fine-tuned GeoRSCLIP model. 

---
