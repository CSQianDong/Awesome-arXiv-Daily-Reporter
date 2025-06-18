# XGraphRAG: Interactive Visual Analysis for Graph-based Retrieval-Augmented Generation 

**Authors**: Ke Wang, Bo Pan, Yingchaojie Feng, Yuwei Wu, Jieyi Chen, Minfeng Zhu, Wei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.13782)  

**Abstract**: Graph-based Retrieval-Augmented Generation (RAG) has shown great capability in enhancing Large Language Model (LLM)'s answer with an external knowledge base. Compared to traditional RAG, it introduces a graph as an intermediate representation to capture better structured relational knowledge in the corpus, elevating the precision and comprehensiveness of generation results. However, developers usually face challenges in analyzing the effectiveness of GraphRAG on their dataset due to GraphRAG's complex information processing pipeline and the overwhelming amount of LLM invocations involved during graph construction and query, which limits GraphRAG interpretability and accessibility. This research proposes a visual analysis framework that helps RAG developers identify critical recalls of GraphRAG and trace these recalls through the GraphRAG pipeline. Based on this framework, we develop XGraphRAG, a prototype system incorporating a set of interactive visualizations to facilitate users' analysis process, boosting failure cases collection and improvement opportunities identification. Our evaluation demonstrates the effectiveness and usability of our approach. Our work is open-sourced and available at this https URL. 

---
# InsertRank: LLMs can reason over BM25 scores to Improve Listwise Reranking 

**Authors**: Rahul Seetharaman, Kaustubh D. Dhole, Aman Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2506.14086)  

**Abstract**: Large Language Models (LLMs) have demonstrated significant strides across various information retrieval tasks, particularly as rerankers, owing to their strong generalization and knowledge-transfer capabilities acquired from extensive pretraining. In parallel, the rise of LLM-based chat interfaces has raised user expectations, encouraging users to pose more complex queries that necessitate retrieval by ``reasoning'' over documents rather than through simple keyword matching or semantic similarity. While some recent efforts have exploited reasoning abilities of LLMs for reranking such queries, considerable potential for improvement remains. In that regards, we introduce InsertRank, an LLM-based reranker that leverages lexical signals like BM25 scores during reranking to further improve retrieval performance. InsertRank demonstrates improved retrieval effectiveness on -- BRIGHT, a reasoning benchmark spanning 12 diverse domains, and R2MED, a specialized medical reasoning retrieval benchmark spanning 8 different tasks. We conduct an exhaustive evaluation and several ablation studies and demonstrate that InsertRank consistently improves retrieval effectiveness across multiple families of LLMs, including GPT, Gemini, and Deepseek models. %In addition, we also conduct ablation studies on normalization by varying the scale of the BM25 scores, and positional bias by shuffling the order of the documents. With Deepseek-R1, InsertRank achieves a score of 37.5 on the BRIGHT benchmark. and 51.1 on the R2MED benchmark, surpassing previous methods. 

---
# AcademicBrowse: Benchmarking Academic Browse Ability of LLMs 

**Authors**: Junting Zhou, Wang Li, Yiyan Liao, Nengyuan Zhang, Tingjia Miaoand Zhihui Qi, Yuhan Wu, Tong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.13784)  

**Abstract**: Large Language Models (LLMs)' search capabilities have garnered significant attention. Existing benchmarks, such as OpenAI's BrowseComp, primarily focus on general search scenarios and fail to adequately address the specific demands of academic search. These demands include deeper literature tracing and organization, professional support for academic databases, the ability to navigate long-tail academic knowledge, and ensuring academic rigor. Here, we proposed AcademicBrowse, the first dataset specifically designed to evaluate the complex information retrieval capabilities of Large Language Models (LLMs) in academic research. AcademicBrowse possesses the following key characteristics: Academic Practicality, where question content closely mirrors real academic learning and research environments, avoiding deliberately misleading models; High Difficulty, with answers that are challenging for single models (e.g., Grok DeepSearch or Gemini Deep Research) to provide directly, often requiring at least three deep searches to derive; Concise Evaluation, where limiting conditions ensure answers are as unique as possible, accompanied by clear sources and brief solution explanations, greatly facilitating subsequent audit and verification, surpassing the current lack of analyzed search datasets both domestically and internationally; and Broad Coverage, as the dataset spans at least 15 different academic disciplines. Through AcademicBrowse, we expect to more precisely measure and promote the performance improvement of LLMs in complex academic information retrieval tasks. The data is available at: this https URL 

---
# RAGtifier: Evaluating RAG Generation Approaches of State-of-the-Art RAG Systems for the SIGIR LiveRAG Competition 

**Authors**: Tim Cofala, Oleh Astappiev, William Xion, Hailay Teklehaymanot  

**Link**: [PDF](https://arxiv.org/pdf/2506.14412)  

**Abstract**: Retrieval-Augmented Generation (RAG) enriches Large Language Models (LLMs) by combining their internal, parametric knowledge with external, non-parametric sources, with the goal of improving factual correctness and minimizing hallucinations. The LiveRAG 2025 challenge explores RAG solutions to maximize accuracy on DataMorgana's QA pairs, which are composed of single-hop and multi-hop questions. The challenge provides access to sparse OpenSearch and dense Pinecone indices of the Fineweb 10BT dataset. It restricts model use to LLMs with up to 10B parameters and final answer generation with Falcon-3-10B. A judge-LLM assesses the submitted answers along with human evaluators. By exploring distinct retriever combinations and RAG solutions under the challenge conditions, our final solution emerged using InstructRAG in combination with a Pinecone retriever and a BGE reranker. Our solution achieved a correctness score of 1.13 and a faithfulness score of 0.55, placing fourth in the SIGIR 2025 LiveRAG Challenge. 

---
# LLM-Driven Data Generation and a Novel Soft Metric for Evaluating Text-to-SQL in Aviation MRO 

**Authors**: Patrick Sutanto, Jonathan Kenrick, Max Lorenz, Joan Santoso  

**Link**: [PDF](https://arxiv.org/pdf/2506.13785)  

**Abstract**: The application of Large Language Models (LLMs) to text-to-SQL tasks promises to democratize data access, particularly in critical industries like aviation Maintenance, Repair, and Operation (MRO). However, progress is hindered by two key challenges: the rigidity of conventional evaluation metrics such as execution accuracy, which offer coarse, binary feedback, and the scarcity of domain-specific evaluation datasets. This paper addresses these gaps. To enable more nuanced assessment, we introduce a novel F1-score-based 'soft' metric that quantifies the informational overlap between generated and ground-truth SQL results. To address data scarcity, we propose an LLM-driven pipeline that synthesizes realistic question-SQL pairs from database schemas. We demonstrate our contributions through an empirical evaluation on an authentic MRO database. Our experiments show that the proposed soft metric provides more insightful performance analysis than strict accuracy, and our data generation technique is effective in creating a domain-specific benchmark. Together, these contributions offer a robust framework for evaluating and advancing text-to-SQL systems in specialized environments. 

---
# ImpReSS: Implicit Recommender System for Support Conversations 

**Authors**: Omri Haller, Yair Meidan, Dudu Mimran, Yuval Elovici, Asaf Shabtai  

**Link**: [PDF](https://arxiv.org/pdf/2506.14231)  

**Abstract**: Following recent advancements in large language models (LLMs), LLM-based chatbots have transformed customer support by automating interactions and providing consistent, scalable service. While LLM-based conversational recommender systems (CRSs) have attracted attention for their ability to enhance the quality of recommendations, limited research has addressed the implicit integration of recommendations within customer support interactions. In this work, we introduce ImpReSS, an implicit recommender system designed for customer support conversations. ImpReSS operates alongside existing support chatbots, where users report issues and chatbots provide solutions. Based on a customer support conversation, ImpReSS identifies opportunities to recommend relevant solution product categories (SPCs) that help resolve the issue or prevent its recurrence -- thereby also supporting business growth. Unlike traditional CRSs, ImpReSS functions entirely implicitly and does not rely on any assumption of a user's purchasing intent. Our empirical evaluation of ImpReSS's ability to recommend relevant SPCs that can help address issues raised in support conversations shows promising results, including an MRR@1 (and recall@3) of 0.72 (0.89) for general problem solving, 0.82 (0.83) for information security support, and 0.85 (0.67) for cybersecurity troubleshooting. To support future research, our data and code will be shared upon request. 

---
# Vela: Scalable Embeddings with Voice Large Language Models for Multimodal Retrieval 

**Authors**: Ruofan Hu, Yan Xia, Minjie Hong, Jieming Zhu, Bo Chen, Xiaoda Yang, Minghui Fang, Tao Jin  

**Link**: [PDF](https://arxiv.org/pdf/2506.14445)  

**Abstract**: Multimodal large language models (MLLMs) have seen substantial progress in recent years. However, their ability to represent multimodal information in the acoustic domain remains underexplored. In this work, we introduce Vela, a novel framework designed to adapt MLLMs for the generation of universal multimodal embeddings. By leveraging MLLMs with specially crafted prompts and selected in-context learning examples, Vela effectively bridges the modality gap across various modalities. We then propose a single-modality training approach, where the model is trained exclusively on text pairs. Our experiments show that Vela outperforms traditional CLAP models in standard text-audio retrieval tasks. Furthermore, we introduce new benchmarks that expose CLAP models' limitations in handling long texts and complex retrieval tasks. In contrast, Vela, by harnessing the capabilities of MLLMs, demonstrates robust performance in these scenarios. Our code will soon be available. 

---
# RMIT-ADM+S at the SIGIR 2025 LiveRAG Challenge 

**Authors**: Kun Ran, Shuoqi Sun, Khoi Nguyen Dinh Anh, Damiano Spina, Oleg Zendel  

**Link**: [PDF](https://arxiv.org/pdf/2506.14516)  

**Abstract**: This paper presents the RMIT--ADM+S participation in the SIGIR 2025 LiveRAG Challenge. Our Generation-Retrieval-Augmented Generation (GRAG) approach relies on generating a hypothetical answer that is used in the retrieval phase, alongside the original question. GRAG also incorporates a pointwise large language model (LLM)-based re-ranking step prior to final answer generation. We describe the system architecture and the rationale behind our design choices. In particular, a systematic evaluation using the Grid of Points (GoP) framework and N-way ANOVA enabled comparison across multiple configurations, including query variant generation, question decomposition, rank fusion strategies, and prompting techniques for answer generation. Our system achieved a Relevance score of 1.199 and a Faithfulness score of 0.477 on the private leaderboard, placing among the top four finalists in the LiveRAG 2025 Challenge. 

---
# Revisiting Chain-of-Thought Prompting: Zero-shot Can Be Stronger than Few-shot 

**Authors**: Xiang Cheng, Chengyan Pan, Minjun Zhao, Deyang Li, Fangchao Liu, Xinyu Zhang, Xiao Zhang, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.14641)  

**Abstract**: In-Context Learning (ICL) is an essential emergent ability of Large Language Models (LLMs), and recent studies introduce Chain-of-Thought (CoT) to exemplars of ICL to enhance the reasoning capability, especially in mathematics tasks. However, given the continuous advancement of model capabilities, it remains unclear whether CoT exemplars still benefit recent, stronger models in such tasks. Through systematic experiments, we find that for recent strong models such as the Qwen2.5 series, adding traditional CoT exemplars does not improve reasoning performance compared to Zero-Shot CoT. Instead, their primary function is to align the output format with human expectations. We further investigate the effectiveness of enhanced CoT exemplars, constructed using answers from advanced models such as \texttt{Qwen2.5-Max} and \texttt{DeepSeek-R1}. Experimental results indicate that these enhanced exemplars still fail to improve the model's reasoning performance. Further analysis reveals that models tend to ignore the exemplars and focus primarily on the instructions, leading to no observable gain in reasoning ability. Overall, our findings highlight the limitations of the current ICL+CoT framework in mathematical reasoning, calling for a re-examination of the ICL paradigm and the definition of exemplars. 

---
# Ring-lite: Scalable Reasoning via C3PO-Stabilized Reinforcement Learning for LLMs 

**Authors**: Ring Team, Bin Hu, Cai Chen, Deng Zhao, Ding Liu, Dingnan Jin, Feng Zhu, Hao Dai, Hongzhi Luan, Jia Guo, Jiaming Liu, Jiewei Wu, Jun Mei, Jun Zhou, Junbo Zhao, Junwu Xiong, Kaihong Zhang, Kuan Xu, Lei Liang, Liang Jiang, Liangcheng Fu, Longfei Zheng, Qiang Gao, Qing Cui, Quan Wan, Shaomian Zheng, Shuaicheng Li, Tongkai Yang, Wang Ren, Xiaodong Yan, Xiaopei Wan, Xiaoyun Feng, Xin Zhao, Xinxing Yang, Xinyu Kong, Xuemin Yang, Yang Li, Yingting Wu, Yongkang Liu, Zhankai Xu, Zhenduo Zhang, Zhenglei Zhou, Zhenyu Huang, Zhiqiang Zhang, Zihao Wang, Zujie Wen  

**Link**: [PDF](https://arxiv.org/pdf/2506.14731)  

**Abstract**: We present Ring-lite, a Mixture-of-Experts (MoE)-based large language model optimized via reinforcement learning (RL) to achieve efficient and robust reasoning capabilities. Built upon the publicly available Ling-lite model, a 16.8 billion parameter model with 2.75 billion activated parameters, our approach matches the performance of state-of-the-art (SOTA) small-scale reasoning models on challenging benchmarks (e.g., AIME, LiveCodeBench, GPQA-Diamond) while activating only one-third of the parameters required by comparable models. To accomplish this, we introduce a joint training pipeline integrating distillation with RL, revealing undocumented challenges in MoE RL training. First, we identify optimization instability during RL training, and we propose Constrained Contextual Computation Policy Optimization(C3PO), a novel approach that enhances training stability and improves computational throughput via algorithm-system co-design methodology. Second, we empirically demonstrate that selecting distillation checkpoints based on entropy loss for RL training, rather than validation metrics, yields superior performance-efficiency trade-offs in subsequent RL training. Finally, we develop a two-stage training paradigm to harmonize multi-domain data integration, addressing domain conflicts that arise in training with mixed dataset. We will release the model, dataset, and code. 

---
# Probabilistic Aggregation and Targeted Embedding Optimization for Collective Moral Reasoning in Large Language Models 

**Authors**: Chenchen Yuan, Zheyu Zhang, Shuo Yang, Bardh Prenkaj, Gjergji Kasneci  

**Link**: [PDF](https://arxiv.org/pdf/2506.14625)  

**Abstract**: Large Language Models (LLMs) have shown impressive moral reasoning abilities. Yet they often diverge when confronted with complex, multi-factor moral dilemmas. To address these discrepancies, we propose a framework that synthesizes multiple LLMs' moral judgments into a collectively formulated moral judgment, realigning models that deviate significantly from this consensus. Our aggregation mechanism fuses continuous moral acceptability scores (beyond binary labels) into a collective probability, weighting contributions by model reliability. For misaligned models, a targeted embedding-optimization procedure fine-tunes token embeddings for moral philosophical theories, minimizing JS divergence to the consensus while preserving semantic integrity. Experiments on a large-scale social moral dilemma dataset show our approach builds robust consensus and improves individual model fidelity. These findings highlight the value of data-driven moral alignment across multiple models and its potential for safer, more consistent AI systems. 

---
# Massive Supervised Fine-tuning Experiments Reveal How Data, Layer, and Training Factors Shape LLM Alignment Quality 

**Authors**: Yuto Harada, Yusuke Yamauchi, Yusuke Oda, Yohei Oseki, Yusuke Miyao, Yu Takagi  

**Link**: [PDF](https://arxiv.org/pdf/2506.14681)  

**Abstract**: Supervised fine-tuning (SFT) is a critical step in aligning large language models (LLMs) with human instructions and values, yet many aspects of SFT remain poorly understood. We trained a wide range of base models on a variety of datasets including code generation, mathematical reasoning, and general-domain tasks, resulting in 1,000+ SFT models under controlled conditions. We then identified the dataset properties that matter most and examined the layer-wise modifications introduced by SFT. Our findings reveal that some training-task synergies persist across all models while others vary substantially, emphasizing the importance of model-specific strategies. Moreover, we demonstrate that perplexity consistently predicts SFT effectiveness--often surpassing superficial similarity between trained data and benchmark--and that mid-layer weight changes correlate most strongly with performance gains. We will release these 1,000+ SFT models and benchmark results to accelerate further research. 

---
# Guaranteed Guess: A Language Modeling Approach for CISC-to-RISC Transpilation with Testing Guarantees 

**Authors**: Ahmed Heakl, Sarim Hashmi, Chaimaa Abi, Celine Lee, Abdulrahman Mahmoud  

**Link**: [PDF](https://arxiv.org/pdf/2506.14606)  

**Abstract**: The hardware ecosystem is rapidly evolving, with increasing interest in translating low-level programs across different instruction set architectures (ISAs) in a quick, flexible, and correct way to enhance the portability and longevity of existing code. A particularly challenging class of this transpilation problem is translating between complex- (CISC) and reduced- (RISC) hardware architectures, due to fundamental differences in instruction complexity, memory models, and execution paradigms. In this work, we introduce GG (Guaranteed Guess), an ISA-centric transpilation pipeline that combines the translation power of pre-trained large language models (LLMs) with the rigor of established software testing constructs. Our method generates candidate translations using an LLM from one ISA to another, and embeds such translations within a software-testing framework to build quantifiable confidence in the translation. We evaluate our GG approach over two diverse datasets, enforce high code coverage (>98%) across unit tests, and achieve functional/semantic correctness of 99% on HumanEval programs and 49% on BringupBench programs, respectively. Further, we compare our approach to the state-of-the-art Rosetta 2 framework on Apple Silicon, showcasing 1.73x faster runtime performance, 1.47x better energy efficiency, and 2.41x better memory usage for our transpiled code, demonstrating the effectiveness of GG for real-world CISC-to-RISC translation tasks. We will open-source our codes, data, models, and benchmarks to establish a common foundation for ISA-level code translation research. 

---
# GenerationPrograms: Fine-grained Attribution with Executable Programs 

**Authors**: David Wan, Eran Hirsch, Elias Stengel-Eskin, Ido Dagan, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2506.14580)  

**Abstract**: Recent large language models (LLMs) achieve impressive performance in source-conditioned text generation but often fail to correctly provide fine-grained attributions for their outputs, undermining verifiability and trust. Moreover, existing attribution methods do not explain how and why models leverage the provided source documents to generate their final responses, limiting interpretability. To overcome these challenges, we introduce a modular generation framework, GenerationPrograms, inspired by recent advancements in executable "code agent" architectures. Unlike conventional generation methods that simultaneously generate outputs and attributions or rely on post-hoc attribution, GenerationPrograms decomposes the process into two distinct stages: first, creating an executable program plan composed of modular text operations (such as paraphrasing, compression, and fusion) explicitly tailored to the query, and second, executing these operations following the program's specified instructions to produce the final response. Empirical evaluations demonstrate that GenerationPrograms significantly improves attribution quality at both the document level and sentence level across two long-form question-answering tasks and a multi-document summarization task. We further demonstrate that GenerationPrograms can effectively function as a post-hoc attribution method, outperforming traditional techniques in recovering accurate attributions. In addition, the interpretable programs generated by GenerationPrograms enable localized refinement through modular-level improvements that further enhance overall attribution quality. 

---
# M2BeamLLM: Multimodal Sensing-empowered mmWave Beam Prediction with Large Language Models 

**Authors**: Can Zheng, Jiguang He, Chung G. Kang, Guofa Cai, Zitong Yu, Merouane Debbah  

**Link**: [PDF](https://arxiv.org/pdf/2506.14532)  

**Abstract**: This paper introduces a novel neural network framework called M2BeamLLM for beam prediction in millimeter-wave (mmWave) massive multi-input multi-output (mMIMO) communication systems. M2BeamLLM integrates multi-modal sensor data, including images, radar, LiDAR, and GPS, leveraging the powerful reasoning capabilities of large language models (LLMs) such as GPT-2 for beam prediction. By combining sensing data encoding, multimodal alignment and fusion, and supervised fine-tuning (SFT), M2BeamLLM achieves significantly higher beam prediction accuracy and robustness, demonstrably outperforming traditional deep learning (DL) models in both standard and few-shot scenarios. Furthermore, its prediction performance consistently improves with increased diversity in sensing modalities. Our study provides an efficient and intelligent beam prediction solution for vehicle-to-infrastructure (V2I) mmWave communication systems. 

---
# Passing the Turing Test in Political Discourse: Fine-Tuning LLMs to Mimic Polarized Social Media Comments 

**Authors**: . Pazzaglia, V. Vendetti, L. D. Comencini, F. Deriu, V. Modugno  

**Link**: [PDF](https://arxiv.org/pdf/2506.14645)  

**Abstract**: The increasing sophistication of large language models (LLMs) has sparked growing concerns regarding their potential role in exacerbating ideological polarization through the automated generation of persuasive and biased content. This study explores the extent to which fine-tuned LLMs can replicate and amplify polarizing discourse within online environments. Using a curated dataset of politically charged discussions extracted from Reddit, we fine-tune an open-source LLM to produce context-aware and ideologically aligned responses. The model's outputs are evaluated through linguistic analysis, sentiment scoring, and human annotation, with particular attention to credibility and rhetorical alignment with the original discourse. The results indicate that, when trained on partisan data, LLMs are capable of producing highly plausible and provocative comments, often indistinguishable from those written by humans. These findings raise significant ethical questions about the use of AI in political discourse, disinformation, and manipulation campaigns. The paper concludes with a discussion of the broader implications for AI governance, platform regulation, and the development of detection tools to mitigate adversarial fine-tuning risks. 

---
# AIn't Nothing But a Survey? Using Large Language Models for Coding German Open-Ended Survey Responses on Survey Motivation 

**Authors**: Leah von der Heyde, Anna-Carolina Haensch, Bernd Weiß, Jessika Daikeler  

**Link**: [PDF](https://arxiv.org/pdf/2506.14634)  

**Abstract**: The recent development and wider accessibility of LLMs have spurred discussions about how they can be used in survey research, including classifying open-ended survey responses. Due to their linguistic capacities, it is possible that LLMs are an efficient alternative to time-consuming manual coding and the pre-training of supervised machine learning models. As most existing research on this topic has focused on English-language responses relating to non-complex topics or on single LLMs, it is unclear whether its findings generalize and how the quality of these classifications compares to established methods. In this study, we investigate to what extent different LLMs can be used to code open-ended survey responses in other contexts, using German data on reasons for survey participation as an example. We compare several state-of-the-art LLMs and several prompting approaches, and evaluate the LLMs' performance by using human expert codings. Overall performance differs greatly between LLMs, and only a fine-tuned LLM achieves satisfactory levels of predictive performance. Performance differences between prompting approaches are conditional on the LLM used. Finally, LLMs' unequal classification performance across different categories of reasons for survey participation results in different categorical distributions when not using fine-tuning. We discuss the implications of these findings, both for methodological research on coding open-ended responses and for their substantive analysis, and for practitioners processing or substantively analyzing such data. Finally, we highlight the many trade-offs researchers need to consider when choosing automated methods for open-ended response classification in the age of LLMs. In doing so, our study contributes to the growing body of research about the conditions under which LLMs can be efficiently, accurately, and reliably leveraged in survey research. 

---
# Thunder-NUBench: A Benchmark for LLMs' Sentence-Level Negation Understanding 

**Authors**: Yeonkyoung So, Gyuseong Lee, Sungmok Jung, Joonhak Lee, JiA Kang, Sangho Kim, Jaejin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.14397)  

**Abstract**: Negation is a fundamental linguistic phenomenon that poses persistent challenges for Large Language Models (LLMs), particularly in tasks requiring deep semantic understanding. Existing benchmarks often treat negation as a side case within broader tasks like natural language inference, resulting in a lack of benchmarks that exclusively target negation understanding. In this work, we introduce \textbf{Thunder-NUBench}, a novel benchmark explicitly designed to assess sentence-level negation understanding in LLMs. Thunder-NUBench goes beyond surface-level cue detection by contrasting standard negation with structurally diverse alternatives such as local negation, contradiction, and paraphrase. The benchmark consists of manually curated sentence-negation pairs and a multiple-choice dataset that enables in-depth evaluation of models' negation understanding. 

---
# Reasoning with Exploration: An Entropy Perspective 

**Authors**: Daixuan Cheng, Shaohan Huang, Xuekai Zhu, Bo Dai, Wayne Xin Zhao, Zhenliang Zhang, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2506.14758)  

**Abstract**: Balancing exploration and exploitation is a central goal in reinforcement learning (RL). Despite recent advances in enhancing language model (LM) reasoning, most methods lean toward exploitation, and increasingly encounter performance plateaus. In this work, we revisit entropy -- a signal of exploration in RL -- and examine its relationship to exploratory reasoning in LMs. Through empirical analysis, we uncover strong positive correlations between high-entropy regions and three types of exploratory reasoning actions: (1) pivotal tokens that determine or connect logical steps, (2) reflective actions such as self-verification and correction, and (3) rare behaviors under-explored by the base LMs. Motivated by this, we introduce a minimal modification to standard RL with only one line of code: augmenting the advantage function with an entropy-based term. Unlike traditional maximum-entropy methods which encourage exploration by promoting uncertainty, we encourage exploration by promoting longer and deeper reasoning chains. Notably, our method achieves significant gains on the Pass@K metric -- an upper-bound estimator of LM reasoning capabilities -- even when evaluated with extremely large K values, pushing the boundaries of LM reasoning. 

---
# LongLLaDA: Unlocking Long Context Capabilities in Diffusion LLMs 

**Authors**: Xiaoran Liu, Zhigeng Liu, Zengfeng Huang, Qipeng Guo, Ziwei He, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2506.14429)  

**Abstract**: Large Language Diffusion Models, or diffusion LLMs, have emerged as a significant focus in NLP research, with substantial effort directed toward understanding their scalability and downstream task performance. However, their long-context capabilities remain unexplored, lacking systematic analysis or methods for context extension. In this work, we present the first systematic investigation comparing the long-context performance of diffusion LLMs and traditional auto-regressive LLMs. We first identify a unique characteristic of diffusion LLMs, unlike auto-regressive LLMs, they maintain remarkably \textbf{\textit{stable perplexity}} during direct context extrapolation. Furthermore, where auto-regressive models fail outright during the Needle-In-A-Haystack task with context exceeding their pretrained length, we discover diffusion LLMs exhibit a distinct \textbf{\textit{local perception}} phenomenon, enabling successful retrieval from recent context segments. We explain both phenomena through the lens of Rotary Position Embedding (RoPE) scaling theory. Building on these observations, we propose LongLLaDA, a training-free method that integrates LLaDA with the NTK-based RoPE extrapolation. Our results validate that established extrapolation scaling laws remain effective for extending the context windows of diffusion LLMs. Furthermore, we identify long-context tasks where diffusion LLMs outperform auto-regressive LLMs and others where they fall short. Consequently, this study establishes the first context extrapolation method for diffusion LLMs while providing essential theoretical insights and empirical benchmarks critical for advancing future research on long-context diffusion LLMs. 

---
# ELLIS Alicante at CQs-Gen 2025: Winning the critical thinking questions shared task: LLM-based question generation and selection 

**Authors**: Lucile Favero, Daniel Frases, Juan Antonio Pérez-Ortiz, Tanja Käser, Nuria Oliver  

**Link**: [PDF](https://arxiv.org/pdf/2506.14371)  

**Abstract**: The widespread adoption of chat interfaces based on Large Language Models (LLMs) raises concerns about promoting superficial learning and undermining the development of critical thinking skills. Instead of relying on LLMs purely for retrieving factual information, this work explores their potential to foster deeper reasoning by generating critical questions that challenge unsupported or vague claims in debate interventions. This study is part of a shared task of the 12th Workshop on Argument Mining, co-located with ACL 2025, focused on automatic critical question generation. We propose a two-step framework involving two small-scale open source language models: a Questioner that generates multiple candidate questions and a Judge that selects the most relevant ones. Our system ranked first in the shared task competition, demonstrating the potential of the proposed LLM-based approach to encourage critical engagement with argumentative texts. 

---
# How Far Can LLMs Improve from Experience? Measuring Test-Time Learning Ability in LLMs with Human Comparison 

**Authors**: Jiayin Wang, Zhiquang Guo, Weizhi Ma, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.14448)  

**Abstract**: As evaluation designs of large language models may shape our trajectory toward artificial general intelligence, comprehensive and forward-looking assessment is essential. Existing benchmarks primarily assess static knowledge, while intelligence also entails the ability to rapidly learn from experience. To this end, we advocate for the evaluation of Test-time Learning, the capacity to improve performance in experience-based, reasoning-intensive tasks during test time. In this work, we propose semantic games as effective testbeds for evaluating test-time learning, due to their resistance to saturation and inherent demand for strategic reasoning. We introduce an objective evaluation framework that compares model performance under both limited and cumulative experience settings, and contains four forms of experience representation. To provide a comparative baseline, we recruit eight human participants to complete the same task. Results show that LLMs exhibit measurable test-time learning capabilities; however, their improvements are less stable under cumulative experience and progress more slowly than those observed in humans. These findings underscore the potential of LLMs as general-purpose learning machines, while also revealing a substantial intellectual gap between models and humans, irrespective of how well LLMs perform on static benchmarks. 

---
# Re-Initialization Token Learning for Tool-Augmented Large Language Models 

**Authors**: Chenghao Li, Liu Liu, Baosheng Yu, Jiayan Qiu, Yibing Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2506.14248)  

**Abstract**: Large language models have demonstrated exceptional performance, yet struggle with complex tasks such as numerical reasoning, plan generation. Integrating external tools, such as calculators and databases, into large language models (LLMs) is crucial for enhancing problem-solving capabilities. Current methods assign a unique token to each tool, enabling LLMs to call tools through token prediction-similar to word generation. However, this approach fails to account for the relationship between tool and word tokens, limiting adaptability within pre-trained LLMs. To address this issue, we propose a novel token learning method that aligns tool tokens with the existing word embedding space from the perspective of initialization, thereby enhancing model performance. We begin by constructing prior token embeddings for each tool based on the tool's name or description, which are used to initialize and regularize the learnable tool token embeddings. This ensures the learned embeddings are well-aligned with the word token space, improving tool call accuracy. We evaluate the method on tasks such as numerical reasoning, knowledge-based question answering, and embodied plan generation using GSM8K-XL, FuncQA, KAMEL, and VirtualHome datasets. The results demonstrate clear improvements over recent baselines, including CoT, REACT, ICL, and ToolkenGPT, indicating that our approach effectively augments LLMs with tools through relevant tokens across diverse domains. 

---
# Expectation Confirmation Preference Optimization for Multi-Turn Conversational Recommendation Agent 

**Authors**: Xueyang Feng, Jingsen Zhang, Jiakai Tang, Wei Li, Guohao Cai, Xu Chen, Quanyu Dai, Yue Zhu, Zhenhua Dong  

**Link**: [PDF](https://arxiv.org/pdf/2506.14302)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have significantly propelled the development of Conversational Recommendation Agents (CRAs). However, these agents often generate short-sighted responses that fail to sustain user guidance and meet expectations. Although preference optimization has proven effective in aligning LLMs with user expectations, it remains costly and performs poorly in multi-turn dialogue. To address this challenge, we introduce a novel multi-turn preference optimization (MTPO) paradigm ECPO, which leverages Expectation Confirmation Theory to explicitly model the evolution of user satisfaction throughout multi-turn dialogues, uncovering the underlying causes of dissatisfaction. These causes can be utilized to support targeted optimization of unsatisfactory responses, thereby achieving turn-level preference optimization. ECPO ingeniously eliminates the significant sampling overhead of existing MTPO methods while ensuring the optimization process drives meaningful improvements. To support ECPO, we introduce an LLM-based user simulator, AILO, to simulate user feedback and perform expectation confirmation during conversational recommendations. Experimental results show that ECPO significantly enhances CRA's interaction capabilities, delivering notable improvements in both efficiency and effectiveness over existing MTPO methods. 

---
# Xolver: Multi-Agent Reasoning with Holistic Experience Learning Just Like an Olympiad Team 

**Authors**: Md Tanzib Hosain, Salman Rahman, Md Kishor Morol, Md Rizwan Parvez  

**Link**: [PDF](https://arxiv.org/pdf/2506.14234)  

**Abstract**: Despite impressive progress on complex reasoning, current large language models (LLMs) typically operate in isolation - treating each problem as an independent attempt, without accumulating or integrating experiential knowledge. In contrast, expert problem solvers - such as Olympiad or programming contest teams - leverage a rich tapestry of experiences: absorbing mentorship from coaches, developing intuition from past problems, leveraging knowledge of tool usage and library functionality, adapting strategies based on the expertise and experiences of peers, continuously refining their reasoning through trial and error, and learning from other related problems even during competition. We introduce Xolver, a training-free multi-agent reasoning framework that equips a black-box LLM with a persistent, evolving memory of holistic experience. Xolver integrates diverse experience modalities, including external and self-retrieval, tool use, collaborative interactions, agent-driven evaluation, and iterative refinement. By learning from relevant strategies, code fragments, and abstract reasoning patterns at inference time, Xolver avoids generating solutions from scratch - marking a transition from isolated inference toward experience-aware language agents. Built on both open-weight and proprietary models, Xolver consistently outperforms specialized reasoning agents. Even with lightweight backbones (e.g., QWQ-32B), it often surpasses advanced models including Qwen3-235B, Gemini 2.5 Pro, o3, and o4-mini-high. With o3-mini-high, it achieves new best results on GSM8K (98.1%), AIME'24 (94.4%), AIME'25 (93.7%), Math-500 (99.8%), and LiveCodeBench-V5 (91.6%) - highlighting holistic experience learning as a key step toward generalist agents capable of expert-level reasoning. Code and data are available at this https URL. 

---
# AgentSynth: Scalable Task Generation for Generalist Computer-Use Agents 

**Authors**: Jingxu Xie, Dylan Xu, Xuandong Zhao, Dawn Song  

**Link**: [PDF](https://arxiv.org/pdf/2506.14205)  

**Abstract**: We introduce AgentSynth, a scalable and cost-efficient pipeline for automatically synthesizing high-quality tasks and trajectory datasets for generalist computer-use agents. Leveraging information asymmetry, AgentSynth constructs subtasks that are simple during generation but significantly more challenging when composed into long-horizon tasks, enabling the creation of over 6,000 diverse and realistic tasks. Our pipeline begins with an LLM-based task proposer guided by a persona, followed by an execution agent that completes the task and logs the trajectory. This process is repeated iteratively to form a sequence of subtasks, which are then summarized by a separate agent into a composite task of controllable difficulty. A key strength of AgentSynth is its ability to precisely modulate task complexity by varying the number of subtasks. Empirical evaluations show that state-of-the-art LLM agents suffer a steep performance drop, from 18% success at difficulty level 1 to just 4% at level 6, highlighting the benchmark's difficulty and discriminative power. Moreover, our pipeline achieves a low average cost of \$0.60 per trajectory, orders of magnitude cheaper than human annotations. Our code and data are publicly available at this https URL 

---
# LingoLoop Attack: Trapping MLLMs via Linguistic Context and State Entrapment into Endless Loops 

**Authors**: Jiyuan Fu, Kaixun Jiang, Lingyi Hong, Jinglun Li, Haijing Guo, Dingkang Yang, Zhaoyu Chen, Wenqiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.14493)  

**Abstract**: Multimodal Large Language Models (MLLMs) have shown great promise but require substantial computational resources during inference. Attackers can exploit this by inducing excessive output, leading to resource exhaustion and service degradation. Prior energy-latency attacks aim to increase generation time by broadly shifting the output token distribution away from the EOS token, but they neglect the influence of token-level Part-of-Speech (POS) characteristics on EOS and sentence-level structural patterns on output counts, limiting their efficacy. To address this, we propose LingoLoop, an attack designed to induce MLLMs to generate excessively verbose and repetitive sequences. First, we find that the POS tag of a token strongly affects the likelihood of generating an EOS token. Based on this insight, we propose a POS-Aware Delay Mechanism to postpone EOS token generation by adjusting attention weights guided by POS information. Second, we identify that constraining output diversity to induce repetitive loops is effective for sustained generation. We introduce a Generative Path Pruning Mechanism that limits the magnitude of hidden states, encouraging the model to produce persistent loops. Extensive experiments demonstrate LingoLoop can increase generated tokens by up to 30 times and energy consumption by a comparable factor on models like Qwen2.5-VL-3B, consistently driving MLLMs towards their maximum generation limits. These findings expose significant MLLMs' vulnerabilities, posing challenges for their reliable deployment. The code will be released publicly following the paper's acceptance. 

---
# From What to Respond to When to Respond: Timely Response Generation for Open-domain Dialogue Agents 

**Authors**: Seongbo Jang, Minjin Jeon, Jaehoon Lee, Seonghyeon Lee, Dongha Lee, Hwanjo Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.14285)  

**Abstract**: While research on dialogue response generation has primarily focused on generating coherent responses conditioning on textual context, the critical question of when to respond grounded on the temporal context remains underexplored. To bridge this gap, we propose a novel task called timely dialogue response generation and introduce the TimelyChat benchmark, which evaluates the capabilities of language models to predict appropriate time intervals and generate time-conditioned responses. Additionally, we construct a large-scale training dataset by leveraging unlabeled event knowledge from a temporal commonsense knowledge graph and employing a large language model (LLM) to synthesize 55K event-driven dialogues. We then train Timer, a dialogue agent designed to proactively predict time intervals and generate timely responses that align with those intervals. Experimental results show that Timer outperforms prompting-based LLMs and other fine-tuned baselines in both turn-level and dialogue-level evaluations. We publicly release our data, model, and code. 

---
# MAS-LitEval : Multi-Agent System for Literary Translation Quality Assessment 

**Authors**: Junghwan Kim, Kieun Park, Sohee Park, Hyunggug Kim, Bongwon Suh  

**Link**: [PDF](https://arxiv.org/pdf/2506.14199)  

**Abstract**: Literary translation requires preserving cultural nuances and stylistic elements, which traditional metrics like BLEU and METEOR fail to assess due to their focus on lexical overlap. This oversight neglects the narrative consistency and stylistic fidelity that are crucial for literary works. To address this, we propose MAS-LitEval, a multi-agent system using Large Language Models (LLMs) to evaluate translations based on terminology, narrative, and style. We tested MAS-LitEval on translations of The Little Prince and A Connecticut Yankee in King Arthur's Court, generated by various LLMs, and compared it to traditional metrics. \textbf{MAS-LitEval} outperformed these metrics, with top models scoring up to 0.890 in capturing literary nuances. This work introduces a scalable, nuanced framework for Translation Quality Assessment (TQA), offering a practical tool for translators and researchers. 

---
# S$^4$C: Speculative Sampling with Syntactic and Semantic Coherence for Efficient Inference of Large Language Models 

**Authors**: Tao He, Guang Huang, Yu Yang, Tianshi Xu, Sicheng Zhao, Guiguang Ding, Pengyang Wang, Feng Tian  

**Link**: [PDF](https://arxiv.org/pdf/2506.14158)  

**Abstract**: Large language models (LLMs) exhibit remarkable reasoning capabilities across diverse downstream tasks. However, their autoregressive nature leads to substantial inference latency, posing challenges for real-time applications. Speculative sampling mitigates this issue by introducing a drafting phase followed by a parallel validation phase, enabling faster token generation and verification. Existing approaches, however, overlook the inherent coherence in text generation, limiting their efficiency. To address this gap, we propose a Speculative Sampling with Syntactic and Semantic Coherence (S$^4$C) framework, which extends speculative sampling by leveraging multi-head drafting for rapid token generation and a continuous verification tree for efficient candidate validation and feature reuse. Experimental results demonstrate that S$^4$C surpasses baseline methods across mainstream tasks, offering enhanced efficiency, parallelism, and the ability to generate more valid tokens with fewer computational resources. On Spec-bench benchmarks, S$^4$C achieves an acceleration ratio of 2.26x-2.60x, outperforming state-of-the-art methods. 

---
# GRAM: A Generative Foundation Reward Model for Reward Generalization 

**Authors**: Chenglong Wang, Yang Gan, Yifu Huo, Yongyu Mu, Qiaozhi He, Murun Yang, Bei Li, Tong Xiao, Chunliang Zhang, Tongran Liu, Jingbo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.14175)  

**Abstract**: In aligning large language models (LLMs), reward models have played an important role, but are standardly trained as discriminative models and rely only on labeled human preference data. In this paper, we explore methods that train reward models using both unlabeled and labeled data. Building on the generative models in LLMs, we develop a generative reward model that is first trained via large-scale unsupervised learning and then fine-tuned via supervised learning. We also show that by using label smoothing, we are in fact optimizing a regularized pairwise ranking loss. This result, in turn, provides a new view of training reward models, which links generative models and discriminative models under the same class of training objectives. The outcome of these techniques is a foundation reward model, which can be applied to a wide range of tasks with little or no further fine-tuning effort. Extensive experiments show that this model generalizes well across several tasks, including response ranking, reinforcement learning from human feedback, and task adaptation with fine-tuning, achieving significant performance improvements over several strong baseline models. 

---
# Ace-CEFR -- A Dataset for Automated Evaluation of the Linguistic Difficulty of Conversational Texts for LLM Applications 

**Authors**: David Kogan, Max Schumacher, Sam Nguyen, Masanori Suzuki, Melissa Smith, Chloe Sophia Bellows, Jared Bernstein  

**Link**: [PDF](https://arxiv.org/pdf/2506.14046)  

**Abstract**: There is an unmet need to evaluate the language difficulty of short, conversational passages of text, particularly for training and filtering Large Language Models (LLMs). We introduce Ace-CEFR, a dataset of English conversational text passages expert-annotated with their corresponding level of text difficulty. We experiment with several models on Ace-CEFR, including Transformer-based models and LLMs. We show that models trained on Ace-CEFR can measure text difficulty more accurately than human experts and have latency appropriate to production environments. Finally, we release the Ace-CEFR dataset to the public for research and development. 

---
# MultiFinBen: A Multilingual, Multimodal, and Difficulty-Aware Benchmark for Financial LLM Evaluation 

**Authors**: Xueqing Peng, Lingfei Qian, Yan Wang, Ruoyu Xiang, Yueru He, Yang Ren, Mingyang Jiang, Jeff Zhao, Huan He, Yi Han, Yun Feng, Yuechen Jiang, Yupeng Cao, Haohang Li, Yangyang Yu, Xiaoyu Wang, Penglei Gao, Shengyuan Lin, Keyi Wang, Shanshan Yang, Yilun Zhao, Zhiwei Liu, Peng Lu, Jerry Huang, Suyuchen Wang, Triantafillos Papadopoulos, Polydoros Giannouris, Efstathia Soufleri, Nuo Chen, Guojun Xiong, Zhiyang Deng, Yijia Zhao, Mingquan Lin, Meikang Qiu, Kaleb E Smith, Arman Cohan, Xiao-Yang Liu, Jimin Huang, Alejandro Lopez-Lira, Xi Chen, Junichi Tsujii, Jian-Yun Nie, Sophia Ananiadou, Qianqian Xie  

**Link**: [PDF](https://arxiv.org/pdf/2506.14028)  

**Abstract**: Recent advances in large language models (LLMs) have accelerated progress in financial NLP and applications, yet existing benchmarks remain limited to monolingual and unimodal settings, often over-relying on simple tasks and failing to reflect the complexity of real-world financial communication. We introduce MultiFinBen, the first multilingual and multimodal benchmark tailored to the global financial domain, evaluating LLMs across modalities (text, vision, audio) and linguistic settings (monolingual, bilingual, multilingual) on domain-specific tasks. We introduce two novel tasks, including PolyFiQA-Easy and PolyFiQA-Expert, the first multilingual financial benchmarks requiring models to perform complex reasoning over mixed-language inputs; and EnglishOCR and SpanishOCR, the first OCR-embedded financial QA tasks challenging models to extract and reason over information from visual-text financial documents. Moreover, we propose a dynamic, difficulty-aware selection mechanism and curate a compact, balanced benchmark rather than simple aggregation existing datasets. Extensive evaluation of 22 state-of-the-art models reveals that even the strongest models, despite their general multimodal and multilingual capabilities, struggle dramatically when faced with complex cross-lingual and multimodal tasks in financial domain. MultiFinBen is publicly released to foster transparent, reproducible, and inclusive progress in financial studies and applications. 

---
# Lost in the Mix: Evaluating LLM Understanding of Code-Switched Text 

**Authors**: Amr Mohamed, Yang Zhang, Michalis Vazirgiannis, Guokan Shang  

**Link**: [PDF](https://arxiv.org/pdf/2506.14012)  

**Abstract**: Code-switching (CSW) is the act of alternating between two or more languages within a single discourse. This phenomenon is widespread in multilingual communities, and increasingly prevalent in online content, where users naturally mix languages in everyday communication. As a result, Large Language Models (LLMs), now central to content processing and generation, are frequently exposed to code-switched inputs. Given their widespread use, it is crucial to understand how LLMs process and reason about such mixed-language text. This paper presents a systematic evaluation of LLM comprehension under code-switching by generating CSW variants of established reasoning and comprehension benchmarks. While degradation is evident when foreign tokens disrupt English text$\unicode{x2013}$even under linguistic constraints$\unicode{x2013}$embedding English into other languages often improves comprehension. Though prompting yields mixed results, fine-tuning offers a more stable path to degradation mitigation. 

---
# AI shares emotion with humans across languages and cultures 

**Authors**: Xiuwen Wu, Hao Wang, Zhiang Yan, Xiaohan Tang, Pengfei Xu, Wai-Ting Siok, Ping Li, Jia-Hong Gao, Bingjiang Lyu, Lang Qin  

**Link**: [PDF](https://arxiv.org/pdf/2506.13978)  

**Abstract**: Effective and safe human-machine collaboration requires the regulated and meaningful exchange of emotions between humans and artificial intelligence (AI). Current AI systems based on large language models (LLMs) can provide feedback that makes people feel heard. Yet it remains unclear whether LLMs represent emotion in language as humans do, or whether and how the emotional tone of their output can be controlled. We assess human-AI emotional alignment across linguistic-cultural groups and model-families, using interpretable LLM features translated from concept-sets for over twenty nuanced emotion categories (including six basic emotions). Our analyses reveal that LLM-derived emotion spaces are structurally congruent with human perception, underpinned by the fundamental affective dimensions of valence and arousal. Furthermore, these emotion-related features also accurately predict large-scale behavioural data on word ratings along these two core dimensions, reflecting both universal and language-specific patterns. Finally, by leveraging steering vectors derived solely from human-centric emotion concepts, we show that model expressions can be stably and naturally modulated across distinct emotion categories, which provides causal evidence that human emotion concepts can be used to systematically induce LLMs to produce corresponding affective states when conveying content. These findings suggest AI not only shares emotional representations with humans but its affective outputs can be precisely guided using psychologically grounded emotion concepts. 

---
# Alignment Quality Index (AQI) : Beyond Refusals: AQI as an Intrinsic Alignment Diagnostic via Latent Geometry, Cluster Divergence, and Layer wise Pooled Representations 

**Authors**: Abhilekh Borah, Chhavi Sharma, Danush Khanna, Utkarsh Bhatt, Gurpreet Singh, Hasnat Md Abdullah, Raghav Kaushik Ravi, Vinija Jain, Jyoti Patel, Shubham Singh, Vasu Sharma, Arpita Vats, Rahul Raja, Aman Chadha, Amitava Das  

**Link**: [PDF](https://arxiv.org/pdf/2506.13901)  

**Abstract**: Alignment is no longer a luxury, it is a necessity. As large language models (LLMs) enter high-stakes domains like education, healthcare, governance, and law, their behavior must reliably reflect human-aligned values and safety constraints. Yet current evaluations rely heavily on behavioral proxies such as refusal rates, G-Eval scores, and toxicity classifiers, all of which have critical blind spots. Aligned models are often vulnerable to jailbreaking, stochasticity of generation, and alignment faking.
To address this issue, we introduce the Alignment Quality Index (AQI). This novel geometric and prompt-invariant metric empirically assesses LLM alignment by analyzing the separation of safe and unsafe activations in latent space. By combining measures such as the Davies-Bouldin Score (DBS), Dunn Index (DI), Xie-Beni Index (XBI), and Calinski-Harabasz Index (CHI) across various formulations, AQI captures clustering quality to detect hidden misalignments and jailbreak risks, even when outputs appear compliant. AQI also serves as an early warning signal for alignment faking, offering a robust, decoding invariant tool for behavior agnostic safety auditing.
Additionally, we propose the LITMUS dataset to facilitate robust evaluation under these challenging conditions. Empirical tests on LITMUS across different models trained under DPO, GRPO, and RLHF conditions demonstrate AQI's correlation with external judges and ability to reveal vulnerabilities missed by refusal metrics. We make our implementation publicly available to foster future research in this area. 

---
# EmoNews: A Spoken Dialogue System for Expressive News Conversations 

**Authors**: Ryuki Matsuura, Shikhar Bharadwaj, Jiarui Liu, Dhatchi Kunde Govindarajan  

**Link**: [PDF](https://arxiv.org/pdf/2506.13894)  

**Abstract**: We develop a task-oriented spoken dialogue system (SDS) that regulates emotional speech based on contextual cues to enable more empathetic news conversations. Despite advancements in emotional text-to-speech (TTS) techniques, task-oriented emotional SDSs remain underexplored due to the compartmentalized nature of SDS and emotional TTS research, as well as the lack of standardized evaluation metrics for social goals. We address these challenges by developing an emotional SDS for news conversations that utilizes a large language model (LLM)-based sentiment analyzer to identify appropriate emotions and PromptTTS to synthesize context-appropriate emotional speech. We also propose subjective evaluation scale for emotional SDSs and judge the emotion regulation performance of the proposed and baseline systems. Experiments showed that our emotional SDS outperformed a baseline system in terms of the emotion regulation and engagement. These results suggest the critical role of speech emotion for more engaging conversations. All our source code is open-sourced at this https URL 

---
# ASMR: Augmenting Life Scenario using Large Generative Models for Robotic Action Reflection 

**Authors**: Shang-Chi Tsai, Seiya Kawano, Angel Garcia Contreras, Koichiro Yoshino, Yun-Nung Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.13956)  

**Abstract**: When designing robots to assist in everyday human activities, it is crucial to enhance user requests with visual cues from their surroundings for improved intent understanding. This process is defined as a multimodal classification task. However, gathering a large-scale dataset encompassing both visual and linguistic elements for model training is challenging and time-consuming. To address this issue, our paper introduces a novel framework focusing on data augmentation in robotic assistance scenarios, encompassing both dialogues and related environmental imagery. This approach involves leveraging a sophisticated large language model to simulate potential conversations and environmental contexts, followed by the use of a stable diffusion model to create images depicting these environments. The additionally generated data serves to refine the latest multimodal models, enabling them to more accurately determine appropriate actions in response to user interactions with the limited target data. Our experimental results, based on a dataset collected from real-world scenarios, demonstrate that our methodology significantly enhances the robot's action selection capabilities, achieving the state-of-the-art performance. 

---
# Investigating the interaction of linguistic and mathematical reasoning in language models using multilingual number puzzles 

**Authors**: Antara Raaghavi Bhattacharya, Isabel Papadimitriou, Kathryn Davidson, David Alvarez-Melis  

**Link**: [PDF](https://arxiv.org/pdf/2506.13886)  

**Abstract**: Across languages, numeral systems vary widely in how they construct and combine numbers. While humans consistently learn to navigate this diversity, large language models (LLMs) struggle with linguistic-mathematical puzzles involving cross-linguistic numeral systems, which humans can learn to solve successfully. We investigate why this task is difficult for LLMs through a series of experiments that untangle the linguistic and mathematical aspects of numbers in language. Our experiments establish that models cannot consistently solve such problems unless the mathematical operations in the problems are explicitly marked using known symbols ($+$, $\times$, etc, as in "twenty + three"). In further ablation studies, we probe how individual parameters of numeral construction and combination affect performance. While humans use their linguistic understanding of numbers to make inferences about the implicit compositional structure of numerals, LLMs seem to lack this notion of implicit numeral structure. We conclude that the ability to flexibly infer compositional rules from implicit patterns in human-scale data remains an open challenge for current reasoning models. 

---
# ClimateChat: Designing Data and Methods for Instruction Tuning LLMs to Answer Climate Change Queries 

**Authors**: Zhou Chen, Xiao Wang, Yuanhong Liao, Ming Lin, Yuqi Bai  

**Link**: [PDF](https://arxiv.org/pdf/2506.13796)  

**Abstract**: As the issue of global climate change becomes increasingly severe, the demand for research in climate science continues to grow. Natural language processing technologies, represented by Large Language Models (LLMs), have been widely applied to climate change-specific research, providing essential information support for decision-makers and the public. Some studies have improved model performance on relevant tasks by constructing climate change-related instruction data and instruction-tuning LLMs. However, current research remains inadequate in efficiently producing large volumes of high-precision instruction data for climate change, which limits further development of climate change LLMs. This study introduces an automated method for constructing instruction data. The method generates instructions using facts and background knowledge from documents and enhances the diversity of the instruction data through web scraping and the collection of seed instructions. Using this method, we constructed a climate change instruction dataset, named ClimateChat-Corpus, which was used to fine-tune open-source LLMs, resulting in an LLM named ClimateChat. Evaluation results show that ClimateChat significantly improves performance on climate change question-and-answer tasks. Additionally, we evaluated the impact of different base models and instruction data on LLM performance and demonstrated its capability to adapt to a wide range of climate change scientific discovery tasks, emphasizing the importance of selecting an appropriate base model for instruction tuning. This research provides valuable references and empirical support for constructing climate change instruction data and training climate change-specific LLMs. 

---
# Are manual annotations necessary for statutory interpretations retrieval? 

**Authors**: Aleksander Smywiński-Pohl, Tomer Libal, Adam Kaczmarczyk, Magdalena Król  

**Link**: [PDF](https://arxiv.org/pdf/2506.13965)  

**Abstract**: One of the elements of legal research is looking for cases where judges have extended the meaning of a legal concept by providing interpretations of what a concept means or does not mean. This allow legal professionals to use such interpretations as precedents as well as laymen to better understand the legal concept. The state-of-the-art approach for retrieving the most relevant interpretations for these concepts currently depends on the ranking of sentences and the training of language models over annotated examples. That manual annotation process can be quite expensive and need to be repeated for each such concept, which prompted recent research in trying to automate this process. In this paper, we highlight the results of various experiments conducted to determine the volume, scope and even the need for manual annotation. First of all, we check what is the optimal number of annotations per a legal concept. Second, we check if we can draw the sentences for annotation randomly or there is a gain in the performance of the model, when only the best candidates are annotated. As the last question we check what is the outcome of automating the annotation process with the help of an LLM. 

---
# ASCD: Attention-Steerable Contrastive Decoding for Reducing Hallucination in MLLM 

**Authors**: Yujun Wang, Jinhe Bi, Yunpu Ma, Soeren Pirk  

**Link**: [PDF](https://arxiv.org/pdf/2506.14766)  

**Abstract**: Multimodal Large Language Model (MLLM) often suffer from hallucinations. They over-rely on partial cues and generate incorrect responses. Recently, methods like Visual Contrastive Decoding (VCD) and Instruction Contrastive Decoding (ICD) have been proposed to mitigate hallucinations by contrasting predictions from perturbed or negatively prefixed inputs against original outputs. In this work, we uncover that methods like VCD and ICD fundamentally influence internal attention dynamics of the model. This observation suggests that their effectiveness may not stem merely from surface-level modifications to logits but from deeper shifts in attention distribution. Inspired by this insight, we propose an attention-steerable contrastive decoding framework that directly intervenes in attention mechanisms of the model to offer a more principled approach to mitigating hallucinations. Our experiments across multiple MLLM architectures and diverse decoding methods demonstrate that our approach significantly reduces hallucinations and improves the performance on benchmarks such as POPE, CHAIR, and MMHal-Bench, while simultaneously enhancing performance on standard VQA benchmarks. 

---
# Optimizing Length Compression in Large Reasoning Models 

**Authors**: Zhengxiang Cheng, Dongping Chen, Mingyang Fu, Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.14755)  

**Abstract**: Large Reasoning Models (LRMs) have achieved remarkable success, yet they often suffer from producing unnecessary and verbose reasoning chains. We identify a core aspect of this issue as "invalid thinking" -- models tend to repeatedly double-check their work after having derived the correct answer. To address this specific inefficiency, we move beyond the general principles of Efficacy and Efficiency to propose two new, fine-grained principles: Brevity, which advocates for eliminating redundancy, and Sufficiency, which ensures critical reasoning steps are preserved. Guided by these principles, we introduce LC-R1, a post-training method based on Group Relative Policy Optimization (GRPO). LC-R1 employs a novel combination of a Length Reward for overall conciseness and a Compress Reward that is specifically designed to remove the invalid portion of the thinking process. Extensive experiments on multiple reasoning benchmarks demonstrate that LC-R1 achieves a significant reduction in sequence length (~50%) with only a marginal (~2%) drop in accuracy, achieving a favorable trade-off point on the Pareto frontier that prioritizes high compression. Our analysis further validates the robustness of LC-R1 and provides valuable insights for developing more powerful yet computationally efficient LRMs. Our code is released at this https URL. 

---
# MIST: Towards Multi-dimensional Implicit Bias and Stereotype Evaluation of LLMs via Theory of Mind 

**Authors**: Yanlin Li, Hao Liu, Huimin Liu, Yinwei Wei, Yupeng Hu  

**Link**: [PDF](https://arxiv.org/pdf/2506.14161)  

**Abstract**: Theory of Mind (ToM) in Large Language Models (LLMs) refers to their capacity for reasoning about mental states, yet failures in this capacity often manifest as systematic implicit bias. Evaluating this bias is challenging, as conventional direct-query methods are susceptible to social desirability effects and fail to capture its subtle, multi-dimensional nature. To this end, we propose an evaluation framework that leverages the Stereotype Content Model (SCM) to reconceptualize bias as a multi-dimensional failure in ToM across Competence, Sociability, and Morality. The framework introduces two indirect tasks: the Word Association Bias Test (WABT) to assess implicit lexical associations and the Affective Attribution Test (AAT) to measure covert affective leanings, both designed to probe latent stereotypes without triggering model avoidance. Extensive experiments on 8 State-of-the-Art LLMs demonstrate our framework's capacity to reveal complex bias structures, including pervasive sociability bias, multi-dimensional divergence, and asymmetric stereotype amplification, thereby providing a more robust methodology for identifying the structural nature of implicit bias. 

---
# Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs 

**Authors**: Xumeng Wen, Zihan Liu, Shun Zheng, Zhijian Xu, Shengyu Ye, Zhirong Wu, Xiao Liang, Yang Wang, Junjie Li, Ziming Miao, Jiang Bian, Mao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.14245)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a promising paradigm for advancing the reasoning capabilities of Large Language Models (LLMs). However, a critical paradox clouds its efficacy: RLVR-tuned models often underperform their base models on the $Pass@K$ metric for solution-finding, leading to the hypothesis that RLVR merely re-weights existing reasoning paths at the cost of reasoning diversity. In this work, we resolve this contradiction by identifying the source of the problem: the $Pass@K$ metric itself is a flawed measure of reasoning, as it credits correct final answers that probably arise from inaccurate or incomplete chains of thought (CoTs). To address this, we introduce a more precise evaluation metric, $CoT$-$Pass@K$, which mandates that both the reasoning path and the final answer be correct. We provide a new theoretical foundation that formalizes how RLVR, unlike traditional RL, is uniquely structured to incentivize logical integrity. Our empirical results are supportive: using $CoT$-$Pass@K$, we observe that RLVR can incentivize the generalization of correct reasoning for all values of $K$. Furthermore, by analyzing the training dynamics, we find that this enhanced reasoning capability emerges early in the training process and smoothly generalizes. Our work provides a clear perspective on the role of RLVR, offers a more reliable method for its evaluation, and confirms its potential to genuinely advance machine reasoning. 

---
# CRITICTOOL: Evaluating Self-Critique Capabilities of Large Language Models in Tool-Calling Error Scenarios 

**Authors**: Shiting Huang, Zhen Fang, Zehui Chen, Siyu Yuan, Junjie Ye, Yu Zeng, Lin Chen, Qi Mao, Feng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.13977)  

**Abstract**: The ability of large language models (LLMs) to utilize external tools has enabled them to tackle an increasingly diverse range of tasks. However, as the tasks become more complex and long-horizon, the intricate tool utilization process may trigger various unexpected errors. Therefore, how to effectively handle such errors, including identifying, diagnosing, and recovering from them, has emerged as a key research direction for advancing tool learning. In this work, we first extensively analyze the types of errors encountered during the function-calling process on several competitive tool evaluation benchmarks. Based on it, we introduce CRITICTOOL, a comprehensive critique evaluation benchmark specialized for tool learning. Building upon a novel evolutionary strategy for dataset construction, CRITICTOOL holds diverse tool-use errors with varying complexities, which better reflects real-world scenarios. We conduct extensive experiments on CRITICTOOL, and validate the generalization and effectiveness of our constructed benchmark strategy. We also provide an in-depth analysis of the tool reflection ability on various LLMs, offering a new perspective on the field of tool learning in LLMs. The code is available at \href{this https URL}{this https URL}. 

---
# AssistedDS: Benchmarking How External Domain Knowledge Assists LLMs in Automated Data Science 

**Authors**: An Luo, Xun Xian, Jin Du, Fangqiao Tian, Ganghua Wang, Ming Zhong, Shengchun Zhao, Xuan Bi, Zirui Liu, Jiawei Zhou, Jayanth Srinivasa, Ashish Kundu, Charles Fleming, Mingyi Hong, Jie Ding  

**Link**: [PDF](https://arxiv.org/pdf/2506.13992)  

**Abstract**: Large language models (LLMs) have advanced the automation of data science workflows. Yet it remains unclear whether they can critically leverage external domain knowledge as human data scientists do in practice. To answer this question, we introduce AssistedDS (Assisted Data Science), a benchmark designed to systematically evaluate how LLMs handle domain knowledge in tabular prediction tasks. AssistedDS features both synthetic datasets with explicitly known generative mechanisms and real-world Kaggle competitions, each accompanied by curated bundles of helpful and adversarial documents. These documents provide domain-specific insights into data cleaning, feature engineering, and model selection. We assess state-of-the-art LLMs on their ability to discern and apply beneficial versus harmful domain knowledge, evaluating submission validity, information recall, and predictive performance. Our results demonstrate three key findings: (1) LLMs frequently exhibit an uncritical adoption of provided information, significantly impairing their predictive performance when adversarial content is introduced, (2) helpful guidance is often insufficient to counteract the negative influence of adversarial information, and (3) in Kaggle datasets, LLMs often make errors in handling time-series data, applying consistent feature engineering across different folds, and interpreting categorical variables correctly. These findings highlight a substantial gap in current models' ability to critically evaluate and leverage expert knowledge, underscoring an essential research direction for developing more robust, knowledge-aware automated data science systems. 

---
# LittleBit: Ultra Low-Bit Quantization via Latent Factorization 

**Authors**: Banseok Lee, Dongkyu Kim, Youngcheon You, Youngmin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.13771)  

**Abstract**: Deploying large language models (LLMs) often faces challenges from substantial memory and computational costs. Quantization offers a solution, yet performance degradation in the sub-1-bit regime remains particularly difficult. This paper introduces LittleBit, a novel method for extreme LLM compression. It targets levels like 0.1 bits per weight (BPW), achieving nearly 31$\times$ memory reduction, e.g., Llama2-13B to under 0.9 GB. LittleBit represents weights in a low-rank form using latent matrix factorization, subsequently binarizing these factors. To counteract information loss from this extreme precision, it integrates a multi-scale compensation mechanism. This includes row, column, and an additional latent dimension that learns per-rank importance. Two key contributions enable effective training: Dual Sign-Value-Independent Decomposition (Dual-SVID) for stable quantization-aware training (QAT) initialization, and integrated Residual Compensation to mitigate errors. Extensive experiments confirm LittleBit's superiority in sub-1-bit quantization: e.g., its 0.1 BPW performance on Llama2-7B surpasses the leading method's 0.7 BPW. This establishes a superior size-performance trade-off, with kernel-level benchmarks indicating potential for a 5$\times$ speedup compared to FP16. LittleBit paves the way for deploying powerful LLMs in resource-constrained environments. 

---
# LLM-Powered Swarms: A New Frontier or a Conceptual Stretch? 

**Authors**: Muhammad Atta Ur Rahman, Melanie Schranz  

**Link**: [PDF](https://arxiv.org/pdf/2506.14496)  

**Abstract**: Swarm intelligence traditionally refers to systems of simple, decentralized agents whose local interactions lead to emergent, collective behavior. Recently, the term 'swarm' has been extended to describe AI systems like OpenAI's Swarm, where large language models (LLMs) act as collaborative agents. This paper contrasts traditional swarm algorithms with LLM-driven swarms exploring how decentralization, scalability, and emergence are redefined in modern artificial intelligence (AI). We implement and compare both paradigms using Boids and Ant Colony Optimization (ACO), evaluating latency, resource usage, and behavioral accuracy. The suitability of both cloud-based and local LLMs is assessed for the agent-based use in swarms. Although LLMs offer powerful reasoning and abstraction capabilities, they introduce new constraints in computation and coordination that challenge traditional notions of swarm design. This study highlights the opportunities and limitations of integrating LLMs into swarm systems and discusses the evolving definition of 'swarm' in modern AI research. 

---
# GUI-Robust: A Comprehensive Dataset for Testing GUI Agent Robustness in Real-World Anomalies 

**Authors**: Jingqi Yang, Zhilong Song, Jiawei Chen, Mingli Song, Sheng Zhou, linjun sun, Xiaogang Ouyang, Chun Chen, Can Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.14477)  

**Abstract**: The development of high-quality datasets is crucial for benchmarking and advancing research in Graphical User Interface (GUI) agents. Despite their importance, existing datasets are often constructed under idealized conditions, overlooking the diverse anomalies frequently encountered in real-world deployments. To address this limitation, we introduce GUI-Robust, a novel dataset designed for comprehensive GUI agent evaluation, explicitly incorporating seven common types of anomalies observed in everyday GUI interactions. Furthermore, we propose a semi-automated dataset construction paradigm that collects user action sequences from natural interactions via RPA tools and then generate corresponding step and task descriptions for these actions with the assistance of MLLMs. This paradigm significantly reduces annotation time cost by a factor of over 19 times. Finally, we assess state-of-the-art GUI agents using the GUI-Robust dataset, revealing their substantial performance degradation in abnormal scenarios. We anticipate that our work will highlight the importance of robustness in GUI agents and inspires more future research in this direction. The dataset and code are available at this https URL.. 

---
# Don't Make It Up: Preserving Ignorance Awareness in LLM Fine-Tuning 

**Authors**: William F. Shen, Xinchi Qiu, Nicola Cancedda, Nicholas D. Lane  

**Link**: [PDF](https://arxiv.org/pdf/2506.14387)  

**Abstract**: Existing work on mitigating catastrophic forgetting in large language model (LLM) fine-tuning has primarily focused on preserving specific data or tasks, while critically overlooking the degradation of essential capabilities instilled through safety alignment, particularly the model's ability to faithfully express ignorance. In this work, we show that this capability is significantly degraded during conventional fine-tuning, leading to undesired behaviors such as hallucinations. To address this novel but highly practical problem, we propose SEAT, a simple and effective fine-tuning approach that preserves both fine-tuning performance and the model's inherent ability to acknowledge its ignorance. SEAT integrates two key components: (1) sparse training that constrains activation drift, and (2) a novel entity perturbation method with KL-divergence regularization, designed to counter knowledge entanglement. Experimental results demonstrate that SEAT significantly outperforms baselines in preserving ignorance awareness while retaining fine-tuning performance, offering a more robust solution for LLM fine-tuning. 

---
# ADRD: LLM-Driven Autonomous Driving Based on Rule-based Decision Systems 

**Authors**: Fanzhi Zeng, Siqi Wang, Chuzhao Zhu, Li Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.14299)  

**Abstract**: How to construct an interpretable autonomous driving decision-making system has become a focal point in academic research. In this study, we propose a novel approach that leverages large language models (LLMs) to generate executable, rule-based decision systems to address this challenge. Specifically, harnessing the strong reasoning and programming capabilities of LLMs, we introduce the ADRD(LLM-Driven Autonomous Driving Based on Rule-based Decision Systems) framework, which integrates three core modules: the Information Module, the Agents Module, and the Testing Module. The framework operates by first aggregating contextual driving scenario information through the Information Module, then utilizing the Agents Module to generate rule-based driving tactics. These tactics are iteratively refined through continuous interaction with the Testing Module. Extensive experimental evaluations demonstrate that ADRD exhibits superior performance in autonomous driving decision tasks. Compared to traditional reinforcement learning approaches and the most advanced LLM-based methods, ADRD shows significant advantages in terms of interpretability, response speed, and driving performance. These results highlight the framework's ability to achieve comprehensive and accurate understanding of complex driving scenarios, and underscore the promising future of transparent, rule-based decision systems that are easily modifiable and broadly applicable. To the best of our knowledge, this is the first work that integrates large language models with rule-based systems for autonomous driving decision-making, and our findings validate its potential for real-world deployment. 

---
# Causes in neuron diagrams, and testing causal reasoning in Large Language Models. A glimpse of the future of philosophy? 

**Authors**: Louis Vervoort, Vitaly Nikolaev  

**Link**: [PDF](https://arxiv.org/pdf/2506.14239)  

**Abstract**: We propose a test for abstract causal reasoning in AI, based on scholarship in the philosophy of causation, in particular on the neuron diagrams popularized by D. Lewis. We illustrate the test on advanced Large Language Models (ChatGPT, DeepSeek and Gemini). Remarkably, these chatbots are already capable of correctly identifying causes in cases that are hotly debated in the literature. In order to assess the results of these LLMs and future dedicated AI, we propose a definition of cause in neuron diagrams with a wider validity than published hitherto, which challenges the widespread view that such a definition is elusive. We submit that these results are an illustration of how future philosophical research might evolve: as an interplay between human and artificial expertise. 

---
# Fragile Preferences: A Deep Dive Into Order Effects in Large Language Models 

**Authors**: Haonan Yin, Shai Vardi, Vidyanand Choudhary  

**Link**: [PDF](https://arxiv.org/pdf/2506.14092)  

**Abstract**: Large language models (LLMs) are increasingly used in decision-support systems across high-stakes domains such as hiring and university admissions, where decisions often involve selecting among competing alternatives. While prior work has noted positional order biases in LLM-driven comparisons, these biases have not been systematically dissected or linked to underlying preference structures. We provide the first comprehensive investigation of positional biases across multiple LLM architectures and domains, uncovering strong and consistent order effects, including a novel centrality bias not previously documented in human or machine decision-making. We also find a quality-dependent shift: when options are high quality, models exhibit primacy bias, but favor latter options when option quality is low. We further identify a previously undocumented bias favoring certain names over others. To distinguish superficial tie-breaking from true distortions of judgment, we introduce a framework that classifies pairwise preferences as robust, fragile, or indifferent. We show that order effects can lead models to select strictly inferior options, and that positional biases are typically stronger than gender biases. These findings suggest that LLMs are not merely inheriting human-like biases, but exhibit distinct failure modes not seen in human decision-making. We propose targeted mitigation strategies, including a novel use of the temperature parameter, to reduce order-driven distortions. 

---
# From Black Boxes to Transparent Minds: Evaluating and Enhancing the Theory of Mind in Multimodal Large Language Models 

**Authors**: Xinyang Li, Siqi Liu, Bochao Zou, Jiansheng Chen, Huimin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.14224)  

**Abstract**: As large language models evolve, there is growing anticipation that they will emulate human-like Theory of Mind (ToM) to assist with routine tasks. However, existing methods for evaluating machine ToM focus primarily on unimodal models and largely treat these models as black boxes, lacking an interpretative exploration of their internal mechanisms. In response, this study adopts an approach based on internal mechanisms to provide an interpretability-driven assessment of ToM in multimodal large language models (MLLMs). Specifically, we first construct a multimodal ToM test dataset, GridToM, which incorporates diverse belief testing tasks and perceptual information from multiple perspectives. Next, our analysis shows that attention heads in multimodal large models can distinguish cognitive information across perspectives, providing evidence of ToM capabilities. Furthermore, we present a lightweight, training-free approach that significantly enhances the model's exhibited ToM by adjusting in the direction of the attention head. 

---
# Lightweight Relevance Grader in RAG 

**Authors**: Taehee Jeong  

**Link**: [PDF](https://arxiv.org/pdf/2506.14084)  

**Abstract**: Retrieval-Augmented Generation (RAG) addresses limitations of large language models (LLMs) by leveraging a vector database to provide more accurate and up-to-date information. When a user submits a query, RAG executes a vector search to find relevant documents, which are then used to generate a response. However, ensuring the relevance of retrieved documents with a query would be a big challenge. To address this, a secondary model, known as a relevant grader, can be served to verify its relevance. To reduce computational requirements of a relevant grader, a lightweight small language model is preferred. In this work, we finetuned llama-3.2-1b as a relevant grader and achieved a significant increase in precision from 0.1301 to 0.7750. Its precision is comparable to that of llama-3.1-70b. Our code is available at this https URL. 

---
# ProfiLLM: An LLM-Based Framework for Implicit Profiling of Chatbot Users 

**Authors**: Shahaf David, Yair Meidan, Ido Hersko, Daniel Varnovitzky, Dudu Mimran, Yuval Elovici, Asaf Shabtai  

**Link**: [PDF](https://arxiv.org/pdf/2506.13980)  

**Abstract**: Despite significant advancements in conversational AI, large language model (LLM)-powered chatbots often struggle with personalizing their responses according to individual user characteristics, such as technical expertise, learning style, and communication preferences. This lack of personalization is particularly problematic in specialized knowledge-intense domains like IT/cybersecurity (ITSec), where user knowledge levels vary widely. Existing approaches for chatbot personalization primarily rely on static user categories or explicit self-reported information, limiting their adaptability to an evolving perception of the user's proficiency, obtained in the course of ongoing interactions. In this paper, we propose ProfiLLM, a novel framework for implicit and dynamic user profiling through chatbot interactions. This framework consists of a taxonomy that can be adapted for use in diverse domains and an LLM-based method for user profiling in terms of the taxonomy. To demonstrate ProfiLLM's effectiveness, we apply it in the ITSec domain where troubleshooting interactions are used to infer chatbot users' technical proficiency. Specifically, we developed ProfiLLM[ITSec], an ITSec-adapted variant of ProfiLLM, and evaluated its performance on 1,760 human-like chatbot conversations from 263 synthetic users. Results show that ProfiLLM[ITSec] rapidly and accurately infers ITSec profiles, reducing the gap between actual and predicted scores by up to 55--65\% after a single prompt, followed by minor fluctuations and further refinement. In addition to evaluating our new implicit and dynamic profiling framework, we also propose an LLM-based persona simulation methodology, a structured taxonomy for ITSec proficiency, our codebase, and a dataset of chatbot interactions to support future research. 

---
# SANGAM: SystemVerilog Assertion Generation via Monte Carlo Tree Self-Refine 

**Authors**: Adarsh Gupta, Bhabesh Mali, Chandan Karfa  

**Link**: [PDF](https://arxiv.org/pdf/2506.13983)  

**Abstract**: Recent advancements in the field of reasoning using Large Language Models (LLMs) have created new possibilities for more complex and automatic Hardware Assertion Generation techniques. This paper introduces SANGAM, a SystemVerilog Assertion Generation framework using LLM-guided Monte Carlo Tree Search for the automatic generation of SVAs from industry-level specifications. The proposed framework utilizes a three-stage approach: Stage 1 consists of multi-modal Specification Processing using Signal Mapper, SPEC Analyzer, and Waveform Analyzer LLM Agents. Stage 2 consists of using the Monte Carlo Tree Self-Refine (MCTSr) algorithm for automatic reasoning about SVAs for each signal, and finally, Stage 3 combines the MCTSr-generated reasoning traces to generate SVA assertions for each signal. The results demonstrated that our framework, SANGAM, can generate a robust set of SVAs, performing better in the evaluation process in comparison to the recent methods. 

---
# Don't throw the baby out with the bathwater: How and why deep learning for ARC 

**Authors**: Jack Cole, Mohamed Osman  

**Link**: [PDF](https://arxiv.org/pdf/2506.14276)  

**Abstract**: The Abstraction and Reasoning Corpus (ARC-AGI) presents a formidable challenge for AI systems. Despite the typically low performance on ARC, the deep learning paradigm remains the most effective known strategy for generating skillful (state-of-the-art) neural networks (NN) across varied modalities and tasks in vision, language etc. The deep learning paradigm has proven to be able to train these skillful neural networks and learn the abstractions needed in these diverse domains. Our work doubles down on that and continues to leverage this paradigm by incorporating on-the-fly NN training at test time. We demonstrate that fully committing to deep learning's capacity to acquire novel abstractions yields state-of-the-art performance on ARC. Specifically, we treat both the neural network and the optimizer (rather than just a pre-trained network) as integral components of the inference process, fostering generalization to unseen tasks. Concretely, we propose a methodology for training on ARC, starting from pretrained LLMs, and enhancing their ARC reasoning. We also propose Test-Time Fine-Tuning (TTFT) and the Augment Inference Reverse-Augmentation and Vote (AIRV) as effective test-time techniques. We are the first to propose and show deep learning can be used effectively for ARC, showing boosts of up to 260% in accuracy with AIRV and a further 300% boost with TTFT. An early version of this approach secured first place in the 2023 ARCathon competition, while the final version achieved the current best score on the ARC private test-set (58%). Our findings highlight the key ingredients of a robust reasoning system in unfamiliar domains, underscoring the central mechanisms that improve broad perceptual reasoning. 

---
# AviationLLM: An LLM-based Knowledge System for Aviation Training 

**Authors**: Jia'ang Wan, Feng Shen, Fujuan Li, Yanjin Sun, Yan Li, Shiwen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.14336)  

**Abstract**: Aviation training is a core link in ensuring flight safety, improving industry efficiency and promoting sustainable development. It not only involves flight simulation but also requires the learning of a great deal of professional aviation theory knowledge. In the existing training system, the knowledge is mainly imparted by the the instructors. However, the number of instructors is limited and the professional answers obtained from the Internet are not accurate enough, resulting in low training efficiency. To address this, we introduced LLM, but the basic pre-trained model cannot provide accurate answers to professional fields, so we fine-tuned it. Traditional Supervised Fine-Tuning (SFT) risk generating superficially plausible but factually incorrect responses due to insufficient data coverage. To address this, we employ Direct Preference Optimization(DPO). This paper proposes Retrieval-Augmented LLM Alignment via Direct Preference Optimization(RALA-DPO). We select open source pre-trained LLM Qwen and adapt it to aviation theory training through DPO-based domain alignment. Simultaneously, to mitigate hallucinations caused by training data biases, knowledge obsolescence, or domain knowledge gaps, we implement Retrieval-Augmented Generation(RAG) technology that combines generative and retrieval models. RALA-DPO effectively retrieves relevant information from external knowledge bases and delivers precise and high-quality responses through the generative model. Experimental results demonstrate that RALA-DPO can improve accuracy in response to professional aviation knowledge. With integrated RAG mechanisms, this system can further improve the accuracy of answers and achieve zero-cost knowledge updates simultaneously. 

---
# The NordDRG AI Benchmark for Large Language Models 

**Authors**: Tapio Pitkäranta  

**Link**: [PDF](https://arxiv.org/pdf/2506.13790)  

**Abstract**: Large language models (LLMs) are already being piloted for clinical coding and decision support. However, until now, no open benchmark has targeted the hospital funding layer where Diagnosis-Related Groups (DRG) determine reimbursement across many countries. We release NordDRG-AI-Benchmark, the first public test-bed that captures a complete DRG rule set and evaluates an LLM's ability to reason over multilingual diagnosis, procedure, and tariff logic.
The benchmark bundles three classes of artefacts: (i) definition tables with 20 interlinked tables covering DRG logic, ICD and NCSP codes, age/sex splits, and country flags; (ii) expert manuals and changelog templates describing real governance workflows; and (iii) a prompt pack of 14 CaseMix tasks that span code lookup, cross-table inference, multilingual terminology, and quality-assurance audits.
All artefacts are available at: this https URL
A baseline demonstration shows that five state-of-the-art LLMs perform very differently on the nine automatically verifiable tasks: o3 (OpenAI) scores 9 out of 9, GPT-4o and o4-mini-high score 7 out of 9, while Gemini 2.5 Pro and Gemini 2.5 Flash solve only 5 out of 9 and 3 out of 9, respectively. These results confirm that NordDRG-AI-Benchmark highlights domain-specific strengths and weaknesses that remain hidden in generic LLM benchmarks, offering a reproducible baseline for research on trustworthy automation in hospital funding. 

---
# Collaborative Editable Model 

**Authors**: Kaiwen Tang, Aitong Wu, Yao Lu, Guangda Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.14146)  

**Abstract**: Vertical-domain large language models (LLMs) play a crucial role in specialized scenarios such as finance, healthcare, and law; however, their training often relies on large-scale annotated data and substantial computational resources, impeding rapid development and continuous iteration. To address these challenges, we introduce the Collaborative Editable Model (CoEM), which constructs a candidate knowledge pool from user-contributed domain snippets, leverages interactive user-model dialogues combined with user ratings and attribution analysis to pinpoint high-value knowledge fragments, and injects these fragments via in-context prompts for lightweight domain adaptation. With high-value knowledge, the LLM can generate more accurate and domain-specific content. In a financial information scenario, we collect 15k feedback from about 120 users and validate CoEM with user ratings to assess the quality of generated insights, demonstrating significant improvements in domain-specific generation while avoiding the time and compute overhead of traditional fine-tuning workflows. 

---
# LocationReasoner: Evaluating LLMs on Real-World Site Selection Reasoning 

**Authors**: Miho Koda, Yu Zheng, Ruixian Ma, Mingyang Sun, Devesh Pansare, Fabio Duarte, Paolo Santi  

**Link**: [PDF](https://arxiv.org/pdf/2506.13841)  

**Abstract**: Recent advances in large language models (LLMs), particularly those enhanced through reinforced post-training, have demonstrated impressive reasoning capabilities, as exemplified by models such as OpenAI o1 and DeepSeek-R1. However, these capabilities are predominantly benchmarked on domains like mathematical problem solving and code generation -- leaving open the question of whether such reasoning skills generalize to complex, real-world scenarios. In this paper, we introduce LocationReasoner, a benchmark designed to evaluate LLMs' reasoning abilities in the context of real-world site selection, where models must identify feasible locations by reasoning over diverse and complicated spatial, environmental, and logistical constraints. The benchmark comprises over 300 carefully crafted queries of varying difficulty levels, supported by a sandbox environment with in-house tools for constraint-based location search. Extensive evaluations reveal that state-of-the-art reasoning models offer limited improvement over their non-reasoning predecessors in real-world contexts, with even the latest OpenAI o4 model failing on 30% of site selection tasks. Moreover, agentic strategies such as ReAct and Reflexion often suffer from over-reasoning, leading to worse outcomes than direct code-generation prompting. With key limitations of LLMs in holistic and non-linear reasoning highlighted, we release LocationReasoner to foster the development of LLMs and agents capable of robust, grounded reasoning in real-world decision-making tasks. Codes and data for our benchmark are available at this https URL. 

---
# Automatic Qiskit Code Refactoring Using Large Language Models 

**Authors**: José Manuel Suárez, Luis Mariano Bibbó, Joaquin Bogado, Alejandro Fernandez  

**Link**: [PDF](https://arxiv.org/pdf/2506.14535)  

**Abstract**: As quantum software frameworks evolve, developers face increasing challenges in maintaining compatibility with rapidly changing APIs. In this work, we present a novel methodology for refactoring Qiskit code using large language models (LLMs). We begin by extracting a taxonomy of migration scenarios from the different sources of official Qiskit documentation (such as release notes), capturing common patterns such as migration of functionality to different modules and deprecated usage. This taxonomy, along with the original Python source code, is provided as input to an LLM, which is then tasked with identifying instances of migration scenarios in the code and suggesting appropriate refactoring solutions. Our approach is designed to address the context length limitations of current LLMs by structuring the input and reasoning process in a targeted, efficient manner. The results demonstrate that LLMs, when guided by domain-specific migration knowledge, can effectively assist in automating Qiskit code migration. This work contributes both a set of proven prompts and taxonomy for Qiskit code migration from earlier versions to version 0.46 and a methodology to asses the capabilities of LLMs to assist in the migration of quantum code. 

---
# LLM-Powered Intent-Based Categorization of Phishing Emails 

**Authors**: Even Eilertsen, Vasileios Mavroeidis, Gudmund Grov  

**Link**: [PDF](https://arxiv.org/pdf/2506.14337)  

**Abstract**: Phishing attacks remain a significant threat to modern cybersecurity, as they successfully deceive both humans and the defense mechanisms intended to protect them. Traditional detection systems primarily focus on email metadata that users cannot see in their inboxes. Additionally, these systems struggle with phishing emails, which experienced users can often identify empirically by the text alone. This paper investigates the practical potential of Large Language Models (LLMs) to detect these emails by focusing on their intent. In addition to the binary classification of phishing emails, the paper introduces an intent-type taxonomy, which is operationalized by the LLMs to classify emails into distinct categories and, therefore, generate actionable threat information. To facilitate our work, we have curated publicly available datasets into a custom dataset containing a mix of legitimate and phishing emails. Our results demonstrate that existing LLMs are capable of detecting and categorizing phishing emails, underscoring their potential in this domain. 

---
# Image Segmentation with Large Language Models: A Survey with Perspectives for Intelligent Transportation Systems 

**Authors**: Sanjeda Akter, Ibne Farabi Shihab, Anuj Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2506.14096)  

**Abstract**: The integration of Large Language Models (LLMs) with computer vision is profoundly transforming perception tasks like image segmentation. For intelligent transportation systems (ITS), where accurate scene understanding is critical for safety and efficiency, this new paradigm offers unprecedented capabilities. This survey systematically reviews the emerging field of LLM-augmented image segmentation, focusing on its applications, challenges, and future directions within ITS. We provide a taxonomy of current approaches based on their prompting mechanisms and core architectures, and we highlight how these innovations can enhance road scene understanding for autonomous driving, traffic monitoring, and infrastructure maintenance. Finally, we identify key challenges, including real-time performance and safety-critical reliability, and outline a perspective centered on explainable, human-centric AI as a prerequisite for the successful deployment of this technology in next-generation transportation systems. 

---
# How Does LLM Reasoning Work for Code? A Survey and a Call to Action 

**Authors**: Ira Ceka, Saurabh Pujar, Irene Manotas, Gail Kaiser, Baishakhi Ray, Shyam Ramji  

**Link**: [PDF](https://arxiv.org/pdf/2506.13932)  

**Abstract**: The rise of large language models (LLMs) has led to dramatic improvements across a wide range of natural language tasks. These advancements have extended into the domain of code, facilitating complex tasks such as code generation, translation, summarization, and repair. However, their utility for real-world deployment in-the-wild has only recently been studied, particularly on software engineering (SWE) tasks such as GitHub issue resolution. In this study, we examine the code reasoning techniques that underlie the ability to perform such tasks, and examine the paradigms used to drive their performance. Our contributions in this paper are: (1) the first dedicated survey on code reasoning for code tasks, highlighting overarching strategies, hybrid and agentic approaches; (2) a taxonomy of various techniques used to drive code reasoning; (3) a comprehensive overview of performance on common benchmarks and a showcase of new, under-explored benchmarks with high potential in SWE; (4) an exploration on how core properties of code can be used to explain different reasoning techniques; and (5) gaps and potentially under-explored areas for future research. 

---
# Med-REFL: Medical Reasoning Enhancement via Self-Corrected Fine-grained Reflection 

**Authors**: Zongxian Yang, Jiayu Qian, Zegao Peng, Haoyu Zhang, Zhi-An Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.13793)  

**Abstract**: Large reasoning models have recently made significant strides in mathematical and code reasoning, yet their success has not transferred smoothly to the medical domain. While multiple factors contribute to this disparity, a critical issue is the inadequate focus on the quality of intermediate reflection steps, which is particularly crucial in high-stakes medical scenarios. To address this challenge, we propose Med-REFL, a \underline{\textbf{Med}}ical \underline{\textbf{R}}easoning \underline{\textbf{E}}nhancement via self-corrected \underline{\textbf{F}}ine-grained ref\underline{\textbf{L}}ection. Our method leverages a tree-of-thought approach to decompose medical questions into fine-grained reasoning paths, quantitatively evaluating each step and its subsequent reflections. These assessments enable automatic construction of direct preference optimization data, reducing reliance on expensive expert annotations while guiding models to identify and correct reasoning errors. Experimental results on the MedQA-USMLE benchmark demonstrate Med-REFL achieves consistent improvements, with average gains up to 4.11\%. Notably, it further boosts the state-of-the-art performance of 7B/8B models by an additional 4.13\%. Furthermore, Med-REFL exhibits strong generalization capabilities and robustness across several challenging medical question-answering datasets. Our work illustrates that prioritizing reflection quality leads to more accurate and trustworthy reasoning in medical AI applications. Checkpoints, code, and data can be found \href{this https URL}{here}. 

---
# FrontendBench: A Benchmark for Evaluating LLMs on Front-End Development via Automatic Evaluation 

**Authors**: Hongda Zhu, Yiwen Zhang, Bing Zhao, Jingzhe Ding, Siyao Liu, Tong Liu, Dandan Wang, Yanan Liu, Zhaojian Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.13832)  

**Abstract**: Large Language Models (LLMs) have made significant strides in front-end code generation. However, existing benchmarks exhibit several critical limitations: many tasks are overly simplistic, test cases often lack rigor, and end-to-end validation is absent. These issues hinder the accurate assessment of model performance. To address these challenges, we present FrontendBench, a benchmark co-developed by humans and LLMs. FrontendBench categorizes tasks based on code functionality and incorporates interactive test scenarios, enabling a more comprehensive and practical evaluation of front-end code generation capabilities. The benchmark comprises 148 meticulously crafted prompt-test case pairs spanning five levels of web components, from basic UI elements to complex interactive features. Each task reflects realistic front-end development challenges. Furthermore, we introduce an automatic evaluation framework that executes generated code within a sandbox environment and assesses outcomes using predefined test scripts. This framework achieves a 90.54% agreement rate with expert human evaluations, demonstrating high reliability. We benchmark several state-of-the-art LLMs on FrontendBench and observe substantial performance disparities in handling real-world front-end tasks. These results highlight FrontendBench as a reliable and scalable benchmark, supporting consistent multimodal evaluation and providing a robust foundation for future research in front-end code generation. Our data and code will be released soon. 

---
# MLDebugging: Towards Benchmarking Code Debugging Across Multi-Library Scenarios 

**Authors**: Jinyang Huang, Xiachong Feng, Qiguang Chen, Hanjie Zhao, Zihui Cheng, Jiesong Bai, Jingxuan Zhou, Min Li, Libo Qin  

**Link**: [PDF](https://arxiv.org/pdf/2506.13824)  

**Abstract**: Code debugging is a crucial task in software engineering, which attracts increasing attention. While remarkable success has been made in the era of large language models (LLMs), current research still focuses on the simple no-library or single-library setting, ignoring the complex multi-library scenario in real-world applications. To address this limitation, we make the first attempt to introduce MLDebugging (Multi-Library Debugging), a comprehensive benchmark designed to assess debugging challenges within multi-library Python code. Specifically, MLDebugging encompasses 126 distinct Python libraries, covering a wide range of multi-library code issues, categorized into seven distinct types. Furthermore, we conduct a thorough evaluation of MLDebugging using both mainstream open-source and closed-source LLMs and highlight that current LLMs still struggle to correctly perform code debugging across multi-library scenarios. We hope this work can uncover the potential of LLMs in multi-library debugging scenario and offer insights for future research. 

---
# Structured Program Synthesis using LLMs: Results and Insights from the IPARC Challenge 

**Authors**: Shraddha Surana, Ashwin Srinivasan, Michael Bain  

**Link**: [PDF](https://arxiv.org/pdf/2506.13820)  

**Abstract**: The IPARC Challenge, inspired by ARC, provides controlled program synthesis tasks over synthetic images to evaluate automatic program construction, focusing on sequence, selection, and iteration. This set of 600 tasks has resisted automated solutions. This paper presents a structured inductive programming approach with LLMs that successfully solves tasks across all IPARC categories. The controlled nature of IPARC reveals insights into LLM-based code generation, including the importance of prior structuring, LLMs' ability to aid structuring (requiring human refinement), the need to freeze correct code, the efficiency of code reuse, and how LLM-generated code can spark human creativity. These findings suggest valuable mechanisms for human-LLM collaboration in tackling complex program synthesis. 

---
# Dr. GPT Will See You Now, but Should It? Exploring the Benefits and Harms of Large Language Models in Medical Diagnosis using Crowdsourced Clinical Cases 

**Authors**: Bonam Mingole, Aditya Majumdar, Firdaus Ahmed Choudhury, Jennifer L. Kraschnewski, Shyam S. Sundar, Amulya Yadav  

**Link**: [PDF](https://arxiv.org/pdf/2506.13805)  

**Abstract**: The proliferation of Large Language Models (LLMs) in high-stakes applications such as medical (self-)diagnosis and preliminary triage raises significant ethical and practical concerns about the effectiveness, appropriateness, and possible harmfulness of the use of these technologies for health-related concerns and queries. Some prior work has considered the effectiveness of LLMs in answering expert-written health queries/prompts, questions from medical examination banks, or queries based on pre-existing clinical cases. Unfortunately, these existing studies completely ignore an in-the-wild evaluation of the effectiveness of LLMs in answering everyday health concerns and queries typically asked by general users, which corresponds to the more prevalent use case for LLMs. To address this research gap, this paper presents the findings from a university-level competition that leveraged a novel, crowdsourced approach for evaluating the effectiveness of LLMs in answering everyday health queries. Over the course of a week, a total of 34 participants prompted four publicly accessible LLMs with 212 real (or imagined) health concerns, and the LLM generated responses were evaluated by a team of nine board-certified physicians. At a high level, our findings indicate that on average, 76% of the 212 LLM responses were deemed to be accurate by physicians. Further, with the help of medical professionals, we investigated whether RAG versions of these LLMs (powered with a comprehensive medical knowledge base) can improve the quality of responses generated by LLMs. Finally, we also derive qualitative insights to explain our quantitative findings by conducting interviews with seven medical professionals who were shown all the prompts in our competition. This paper aims to provide a more grounded understanding of how LLMs perform in real-world everyday health communication. 

---
# Enhancing Clinical Decision Support and EHR Insights through LLMs and the Model Context Protocol: An Open-Source MCP-FHIR Framework 

**Authors**: Abul Ehtesham, Aditi Singh, Saket Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2506.13800)  

**Abstract**: Enhancing clinical decision support (CDS), reducing documentation burdens, and improving patient health literacy remain persistent challenges in digital health. This paper presents an open-source, agent-based framework that integrates Large Language Models (LLMs) with HL7 FHIR data via the Model Context Protocol (MCP) for dynamic extraction and reasoning over electronic health records (EHRs). Built on the established MCP-FHIR implementation, the framework enables declarative access to diverse FHIR resources through JSON-based configurations, supporting real-time summarization, interpretation, and personalized communication across multiple user personas, including clinicians, caregivers, and patients. To ensure privacy and reproducibility, the framework is evaluated using synthetic EHR data from the SMART Health IT sandbox (this https URL), which conforms to the FHIR R4 standard. Unlike traditional approaches that rely on hardcoded retrieval and static workflows, the proposed method delivers scalable, explainable, and interoperable AI-powered EHR applications. The agentic architecture further supports multiple FHIR formats, laying a robust foundation for advancing personalized digital health solutions. 

---
# Taming Polysemanticity in LLMs: Provable Feature Recovery via Sparse Autoencoders 

**Authors**: Siyu Chen, Heejune Sheen, Xuyuan Xiong, Tianhao Wang, Zhuoran Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.14002)  

**Abstract**: We study the challenge of achieving theoretically grounded feature recovery using Sparse Autoencoders (SAEs) for the interpretation of Large Language Models. Existing SAE training algorithms often lack rigorous mathematical guarantees and suffer from practical limitations such as hyperparameter sensitivity and instability. To address these issues, we first propose a novel statistical framework for the feature recovery problem, which includes a new notion of feature identifiability by modeling polysemantic features as sparse mixtures of underlying monosemantic concepts. Building on this framework, we introduce a new SAE training algorithm based on ``bias adaptation'', a technique that adaptively adjusts neural network bias parameters to ensure appropriate activation sparsity. We theoretically \highlight{prove that this algorithm correctly recovers all monosemantic features} when input data is sampled from our proposed statistical model. Furthermore, we develop an improved empirical variant, Group Bias Adaptation (GBA), and \highlight{demonstrate its superior performance against benchmark methods when applied to LLMs with up to 1.5 billion parameters}. This work represents a foundational step in demystifying SAE training by providing the first SAE algorithm with theoretical recovery guarantees, thereby advancing the development of more transparent and trustworthy AI systems through enhanced mechanistic interpretability. 

---
# MobiEdit: Resource-efficient Knowledge Editing for Personalized On-device LLMs 

**Authors**: Zhenyan Lu, Daliang Xu, Dongqi Cai, Zexi Li, Wei Liu, Fangming Liu, Shangguang Wang, Mengwei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.13772)  

**Abstract**: Large language models (LLMs) are deployed on mobile devices to power killer applications such as intelligent assistants. LLMs pre-trained on general corpora often hallucinate when handling personalized or unseen queries, leading to incorrect or outdated responses. Knowledge editing addresses this by identifying and adjusting a small crucial portion of model weights, without compromising the general knowledge. However, prior knowledge editing methods are impractical to run on local devices due to the resource-heavy backpropagation (BP) needed for updates. We present MobiEdit, the first mobile knowledge editing framework that enables efficient LLM personalization on commercial off-the-shelf (COTS) mobile devices. MobiEdit replaces full-precision BP with quantized forward-only gradient estimation, thus compatible with the energy-efficient mobile neural processing units (NPUs). MobiEdit replaces full-precision backpropagation with quantized forward-only gradient estimation, making it compatible with energy-efficient mobile NPUs. To further improve gradient estimation efficiency, we introduce two optimizations: an early stoping mechanism that adaptively terminates editing upon success and a prefix cache that reuses computation across steps. Our approach enables real-time editing of a 3B-parameter model (Qwen2.5-3B-Instruct) on COTS mobile devices with 7.6$\times$ less memory, 14.7 $\times$ less energy and 3.6$\times$ less latency compared to previous knowledge editing methods. 

---
