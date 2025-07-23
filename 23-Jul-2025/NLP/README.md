# MegaScience: Pushing the Frontiers of Post-Training Datasets for Science Reasoning 

**Authors**: Run-Ze Fan, Zengzhi Wang, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.16812)  

**Abstract**: Scientific reasoning is critical for developing AI scientists and supporting human researchers in advancing the frontiers of natural science discovery. However, the open-source community has primarily focused on mathematics and coding while neglecting the scientific domain, largely due to the absence of open, large-scale, high-quality, verifiable scientific reasoning datasets. To bridge this gap, we first present TextbookReasoning, an open dataset featuring truthful reference answers extracted from 12k university-level scientific textbooks, comprising 650k reasoning questions spanning 7 scientific disciplines. We further introduce MegaScience, a large-scale mixture of high-quality open-source datasets totaling 1.25 million instances, developed through systematic ablation studies that evaluate various data selection methodologies to identify the optimal subset for each publicly available scientific dataset. Meanwhile, we build a comprehensive evaluation system covering diverse subjects and question types across 15 benchmarks, incorporating comprehensive answer extraction strategies to ensure accurate evaluation metrics. Our experiments demonstrate that our datasets achieve superior performance and training efficiency with more concise response lengths compared to existing open-source scientific datasets. Furthermore, we train Llama3.1, Qwen2.5, and Qwen3 series base models on MegaScience, which significantly outperform the corresponding official instruct models in average performance. In addition, MegaScience exhibits greater effectiveness for larger and stronger models, suggesting a scaling benefit for scientific tuning. We release our data curation pipeline, evaluation system, datasets, and seven trained models to the community to advance scientific reasoning research. 

---
# LingBench++: A Linguistically-Informed Benchmark and Reasoning Framework for Multi-Step and Cross-Cultural Inference with LLMs 

**Authors**: Da-Chen Lian, Ri-Sheng Huang, Pin-Er Chen, Chunki Lim, You-Kuan Lin, Guan-Yu Tseng, Zi-Cheng Yang, Shu-Kai Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2507.16809)  

**Abstract**: We propose LingBench++, a linguistically-informed benchmark and reasoning framework designed to evaluate large language models (LLMs) on complex linguistic tasks inspired by the International Linguistics Olympiad (IOL). Unlike prior benchmarks that focus solely on final answer accuracy, LingBench++ provides structured reasoning traces, stepwise evaluation protocols, and rich typological metadata across over 90 low-resource and cross-cultural languages. We further develop a multi-agent architecture integrating grammatical knowledge retrieval, tool-augmented reasoning, and deliberate hypothesis testing. Through systematic comparisons of baseline and our proposed agentic models, we demonstrate that models equipped with external knowledge sources and iterative reasoning outperform single-pass approaches in both accuracy and interpretability. LingBench++ offers a comprehensive foundation for advancing linguistically grounded, culturally informed, and cognitively plausible reasoning in LLMs. 

---
# Agentar-Fin-R1: Enhancing Financial Intelligence through Domain Expertise, Training Efficiency, and Advanced Reasoning 

**Authors**: Yanjun Zheng, Xiyang Du, Longfei Liao, Xiaoke Zhao, Zhaowen Zhou, Bo Zhang, Jiawei Liu, Xiang Qi, Zhe Li, Zhiqiang Zhang, Wang Wei, Peng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.16802)  

**Abstract**: Large Language Models (LLMs) demonstrate tremendous potential in the financial domain, yet existing models often fall short in scenarios demanding robust reasoning capabilities, stringent trustworthiness requirements, and efficient adaptation to task-specific needs. We introduce the Agentar-Fin-R1 series of financial large language models (8B and 32B parameters), specifically engineered based on the Qwen3 foundation model to enhance reasoning capabilities, reliability, and domain specialization for financial applications. Our optimization approach integrates a high-quality, systematic financial task taxonomy with a comprehensive multi-layered trustworthiness assurance framework. This framework encompasses high-quality trustworthy knowledge engineering, multi-agent trustworthy data synthesis, and rigorous data validation governance. Through label-guided automated difficulty-aware optimization, tow-stage learning processes, and detailed attribution systems, we achieve substantial improvements in training efficiency. Our models undergo comprehensive evaluation on mainstream financial benchmarks including FinEva, FinEval, and FinanceIQ, as well as general reasoning datasets such as MATH-500 and GPQA. To thoroughly assess real-world deployment capabilities, we innovatively propose the Finova evaluation benchmark, which focuses on agent-level financial reasoning and compliance verification. Experimental results demonstrate that Agentar-Fin-R1 not only achieves state-of-the-art performance on financial tasks but also exhibits exceptional general reasoning capabilities, validating its effectiveness as a trustworthy solution for high-stakes financial applications. 

---
# Test-Time-Matching: Decouple Personality, Memory, and Linguistic Style in LLM-based Role-Playing Language Agent 

**Authors**: Xiaoyu Zhan, Xinyu Fu, Hao Sun, Yuanqi Li, Jie Guo, Yanwen Guo  

**Link**: [PDF](https://arxiv.org/pdf/2507.16799)  

**Abstract**: The rapid advancement of large language models (LLMs) has enabled role-playing language agents to demonstrate significant potential in various applications. However, relying solely on prompts and contextual inputs often proves insufficient for achieving deep immersion in specific roles, particularly well-known fictional or public figures. On the other hand, fine-tuning-based approaches face limitations due to the challenges associated with data collection and the computational resources required for training, thereby restricting their broader applicability. To address these issues, we propose Test-Time-Matching (TTM), a training-free role-playing framework through test-time scaling and context engineering. TTM uses LLM agents to automatically decouple a character's features into personality, memory, and linguistic style. Our framework involves a structured, three-stage generation pipeline that utilizes these features for controlled role-playing. It achieves high-fidelity role-playing performance, also enables seamless combinations across diverse linguistic styles and even variations in personality and memory. We evaluate our framework through human assessment, and the results demonstrate that our method achieves the outstanding performance in generating expressive and stylistically consistent character dialogues. 

---
# Beyond Context Limits: Subconscious Threads for Long-Horizon Reasoning 

**Authors**: Hongyin Luo, Nathaniel Morgan, Tina Li, Derek Zhao, Ai Vy Ngo, Philip Schroeder, Lijie Yang, Assaf Ben-Kish, Jack O'Brien, James Glass  

**Link**: [PDF](https://arxiv.org/pdf/2507.16784)  

**Abstract**: To break the context limits of large language models (LLMs) that bottleneck reasoning accuracy and efficiency, we propose the Thread Inference Model (TIM), a family of LLMs trained for recursive and decompositional problem solving, and TIMRUN, an inference runtime enabling long-horizon structured reasoning beyond context limits. Together, TIM hosted on TIMRUN supports virtually unlimited working memory and multi-hop tool calls within a single language model inference, overcoming output limits, positional-embedding constraints, and GPU-memory bottlenecks. Performance is achieved by modeling natural language as reasoning trees measured by both length and depth instead of linear sequences. The reasoning trees consist of tasks with thoughts, recursive subtasks, and conclusions based on the concept we proposed in Schroeder et al, 2025. During generation, we maintain a working memory that retains only the key-value states of the most relevant context tokens, selected by a rule-based subtask-pruning mechanism, enabling reuse of positional embeddings and GPU memory pages throughout reasoning. Experimental results show that our system sustains high inference throughput, even when manipulating up to 90% of the KV cache in GPU memory. It also delivers accurate reasoning on mathematical tasks and handles information retrieval challenges that require long-horizon reasoning and multi-hop tool use. 

---
# Unpacking Ambiguity: The Interaction of Polysemous Discourse Markers and Non-DM Signals 

**Authors**: Jingni Wu, Amir Zeldes  

**Link**: [PDF](https://arxiv.org/pdf/2507.16748)  

**Abstract**: Discourse markers (DMs) like 'but' or 'then' are crucial for creating coherence in discourse, yet they are often replaced by or co-occur with non-DMs ('in the morning' can mean the same as 'then'), and both can be ambiguous ('since' can refer to time or cause). The interaction mechanism between such signals remains unclear but pivotal for their disambiguation. In this paper we investigate the relationship between DM polysemy and co-occurrence of non-DM signals in English, as well as the influence of genre on these patterns.
Using the framework of eRST, we propose a graded definition of DM polysemy, and conduct correlation and regression analyses to examine whether polysemous DMs are accompanied by more numerous and diverse non-DM signals. Our findings reveal that while polysemous DMs do co-occur with more diverse non-DMs, the total number of co-occurring signals does not necessarily increase. Moreover, genre plays a significant role in shaping DM-signal interactions. 

---
# RAVine: Reality-Aligned Evaluation for Agentic Search 

**Authors**: Yilong Xu, Xiang Long, Zhi Zheng, Jinhua Gao  

**Link**: [PDF](https://arxiv.org/pdf/2507.16725)  

**Abstract**: Agentic search, as a more autonomous and adaptive paradigm of retrieval augmentation, is driving the evolution of intelligent search systems. However, existing evaluation frameworks fail to align well with the goals of agentic search. First, the complex queries commonly used in current benchmarks often deviate from realistic user search scenarios. Second, prior approaches tend to introduce noise when extracting ground truth for end-to-end evaluations, leading to distorted assessments at a fine-grained level. Third, most current frameworks focus solely on the quality of final answers, neglecting the evaluation of the iterative process inherent to agentic search. To address these limitations, we propose RAVine -- a Reality-Aligned eValuation framework for agentic LLMs with search. RAVine targets multi-point queries and long-form answers that better reflect user intents, and introduces an attributable ground truth construction strategy to enhance the accuracy of fine-grained evaluation. Moreover, RAVine examines model's interaction with search tools throughout the iterative process, and accounts for factors of efficiency. We benchmark a series of models using RAVine and derive several insights, which we hope will contribute to advancing the development of agentic search systems. The code and datasets are available at this https URL. 

---
# Advancing Risk and Quality Assurance: A RAG Chatbot for Improved Regulatory Compliance 

**Authors**: Lars Hillebrand, Armin Berger, Daniel Uedelhoven, David Berghaus, Ulrich Warning, Tim Dilmaghani, Bernd Kliem, Thomas Schmid, Rüdiger Loitz, Rafet Sifa  

**Link**: [PDF](https://arxiv.org/pdf/2507.16711)  

**Abstract**: Risk and Quality (R&Q) assurance in highly regulated industries requires constant navigation of complex regulatory frameworks, with employees handling numerous daily queries demanding accurate policy interpretation. Traditional methods relying on specialized experts create operational bottlenecks and limit scalability. We present a novel Retrieval Augmented Generation (RAG) system leveraging Large Language Models (LLMs), hybrid search and relevance boosting to enhance R&Q query processing. Evaluated on 124 expert-annotated real-world queries, our actively deployed system demonstrates substantial improvements over traditional RAG approaches. Additionally, we perform an extensive hyperparameter analysis to compare and evaluate multiple configuration setups, delivering valuable insights to practitioners. 

---
# Interpretable Topic Extraction and Word Embedding Learning using row-stochastic DEDICOM 

**Authors**: Lars Hillebrand, David Biesner, Christian Bauckhage, Rafet Sifa  

**Link**: [PDF](https://arxiv.org/pdf/2507.16695)  

**Abstract**: The DEDICOM algorithm provides a uniquely interpretable matrix factorization method for symmetric and asymmetric square matrices. We employ a new row-stochastic variation of DEDICOM on the pointwise mutual information matrices of text corpora to identify latent topic clusters within the vocabulary and simultaneously learn interpretable word embeddings. We introduce a method to efficiently train a constrained DEDICOM algorithm and a qualitative evaluation of its topic modeling and word embedding performance. 

---
# PICACO: Pluralistic In-Context Value Alignment of LLMs via Total Correlation Optimization 

**Authors**: Han Jiang, Dongyao Zhu, Zhihua Wei, Xiaoyuan Yi, Ziang Xiao, Xing Xie  

**Link**: [PDF](https://arxiv.org/pdf/2507.16679)  

**Abstract**: In-Context Learning has shown great potential for aligning Large Language Models (LLMs) with human values, helping reduce harmful outputs and accommodate diverse preferences without costly post-training, known as In-Context Alignment (ICA). However, LLMs' comprehension of input prompts remains agnostic, limiting ICA's ability to address value tensions--human values are inherently pluralistic, often imposing conflicting demands, e.g., stimulation vs. tradition. Current ICA methods therefore face the Instruction Bottleneck challenge, where LLMs struggle to reconcile multiple intended values within a single prompt, leading to incomplete or biased alignment. To address this, we propose PICACO, a novel pluralistic ICA method. Without fine-tuning, PICACO optimizes a meta-instruction that navigates multiple values to better elicit LLMs' understanding of them and improve their alignment. This is achieved by maximizing the total correlation between specified values and LLM responses, theoretically reinforcing value correlation while reducing distractive noise, resulting in effective value instructions. Extensive experiments on five value sets show that PICACO works well with both black-box and open-source LLMs, outperforms several recent strong baselines, and achieves a better balance across up to 8 distinct values. 

---
# Self-Contradiction as Self-Improvement: Mitigating the Generation-Understanding Gap in MLLMs 

**Authors**: Yujin Han, Hao Chen, Andi Han, Zhiheng Wang, Xinyu Lin, Yingya Zhang, Shiwei Zhang, Difan Zou  

**Link**: [PDF](https://arxiv.org/pdf/2507.16663)  

**Abstract**: Despite efforts to unify multimodal generation and understanding tasks in a single model, we show these MLLMs exhibit self-contradiction where generation produces images deemed misaligned with input prompts based on the model's own understanding. We define a Nonunified score that quantifies such self-contradiction. Our empirical results reveal that the self-contradiction mainly arises from weak generation that fails to align with prompts, rather than misunderstanding. This capability asymmetry indicates the potential of leveraging self-contradiction for self-improvement, where the stronger model understanding guides the weaker generation to mitigate the generation-understanding gap. Applying standard post-training methods (e.g., SFT, DPO) with such internal supervision successfully improves both generation and unification. We discover a co-improvement effect on both generation and understanding when only fine-tuning the generation branch, a phenomenon known in pre-training but underexplored in post-training. Our analysis shows improvements stem from better detection of false positives that are previously incorrectly identified as prompt-aligned. Theoretically, we show the aligned training dynamics between generation and understanding allow reduced prompt-misaligned generations to also improve mismatch detection in the understanding branch. Additionally, the framework reveals a potential risk of co-degradation under poor supervision-an overlooked phenomenon that is empirically validated in our experiments. Notably, we find intrinsic metrics like Nonunified score cannot distinguish co-degradation from co-improvement, which highlights the necessity of data quality check. Finally, we propose a curriculum-based strategy based on our findings that gradually introduces harder samples as the model improves, leading to better unification and improved MLLM generation and understanding. 

---
# P-CoT: A Pedagogically-motivated Participatory Chain-of-Thought Prompting for Phonological Reasoning in LLMs 

**Authors**: Dongjun Jang, Youngchae Ahn, Hyopil Shin  

**Link**: [PDF](https://arxiv.org/pdf/2507.16656)  

**Abstract**: This study explores the potential of phonological reasoning within text-based large language models (LLMs). Utilizing the PhonologyBench benchmark, we assess tasks like rhyme word generation, g2p conversion, and syllable counting. Our evaluations across 12 LLMs reveal that while few-shot learning offers inconsistent gains, the introduction of a novel Pedagogically-motivated Participatory Chain-of-Thought (P-CoT) prompt, which is anchored in educational theories like scaffolding and discovery learning, consistently enhances performance. This method leverages structured guidance to activate latent phonological abilities, achieving up to 52% improvement and even surpassing human baselines in certain tasks. Future work could aim to optimize P-CoT prompts for specific models or explore their application across different linguistic domains. 

---
# Towards Automated Regulatory Compliance Verification in Financial Auditing with Large Language Models 

**Authors**: Armin Berger, Lars Hillebrand, David Leonhard, Tobias Deußer, Thiago Bell Felix de Oliveira, Tim Dilmaghani, Mohamed Khaled, Bernd Kliem, Rüdiger Loitz, Christian Bauckhage, Rafet Sifa  

**Link**: [PDF](https://arxiv.org/pdf/2507.16642)  

**Abstract**: The auditing of financial documents, historically a labor-intensive process, stands on the precipice of transformation. AI-driven solutions have made inroads into streamlining this process by recommending pertinent text passages from financial reports to align with the legal requirements of accounting standards. However, a glaring limitation remains: these systems commonly fall short in verifying if the recommended excerpts indeed comply with the specific legal mandates. Hence, in this paper, we probe the efficiency of publicly available Large Language Models (LLMs) in the realm of regulatory compliance across different model configurations. We place particular emphasis on comparing cutting-edge open-source LLMs, such as Llama-2, with their proprietary counterparts like OpenAI's GPT models. This comparative analysis leverages two custom datasets provided by our partner PricewaterhouseCoopers (PwC) Germany. We find that the open-source Llama-2 70 billion model demonstrates outstanding performance in detecting non-compliance or true negative occurrences, beating all their proprietary counterparts. Nevertheless, proprietary models such as GPT-4 perform the best in a broad variety of scenarios, particularly in non-English contexts. 

---
# Step-Audio 2 Technical Report 

**Authors**: Boyong Wu, Chao Yan, Chen Hu, Cheng Yi, Chengli Feng, Fei Tian, Feiyu Shen, Gang Yu, Haoyang Zhang, Jingbei Li, Mingrui Chen, Peng Liu, Wang You, Xiangyu Tony Zhang, Xingyuan Li, Xuerui Yang, Yayue Deng, Yechang Huang, Yuxin Li, Yuxin Zhang, Zhao You, Brian Li, Changyi Wan, Hanpeng Hu, Jiangjie Zhen, Siyu Chen, Song Yuan, Xuelin Zhang, Yimin Jiang, Yu Zhou, Yuxiang Yang, Bingxin Li, Buyun Ma, Changhe Song, Dongqing Pang, Guoqiang Hu, Haiyang Sun, Kang An, Na Wang, Shuli Gao, Wei Ji, Wen Li, Wen Sun, Xuan Wen, Yong Ren, Yuankai Ma, Yufan Lu, Bin Wang, Bo Li, Changxin Miao, Che Liu, Chen Xu, Dapeng Shi, Dingyuan Hu, Donghang Wu, Enle Liu, Guanzhe Huang, Gulin Yan, Han Zhang, Hao Nie, Haonan Jia, Hongyu Zhou, Jianjian Sun, Jiaoren Wu, Jie Wu, Jie Yang, Jin Yang, Junzhe Lin, Kaixiang Li, Lei Yang, Liying Shi, Li Zhou, Longlong Gu, Ming Li, Mingliang Li, Mingxiao Li, Nan Wu, Qi Han, Qinyuan Tan, Shaoliang Pang, Shengjie Fan, Siqi Liu, Tiancheng Cao, Wanying Lu, Wenqing He, Wuxun Xie, Xu Zhao, Xueqi Li, Yanbo Yu, Yang Yang, Yi Liu, Yifan Lu, Yilei Wang, Yuanhao Ding, Yuanwei Liang, Yuanwei Lu, Yuchu Luo, Yuhe Yin, Yumeng Zhan, Yuxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.16632)  

**Abstract**: This paper presents Step-Audio~2, an end-to-end multi-modal large language model designed for industry-strength audio understanding and speech conversation. By integrating a latent audio encoder and reasoning-centric reinforcement learning (RL), Step-Audio 2 achieves promising performance in automatic speech recognition (ASR) and audio understanding. To facilitate genuine end-to-end speech conversation, Step-Audio 2 incorporates the generation of discrete audio tokens into language modeling, significantly enhancing its responsiveness to paralinguistic information such as speaking styles and emotions. To effectively leverage the rich textual and acoustic knowledge in real-world data, Step-Audio 2 integrates retrieval-augmented generation (RAG) and is able to call external tools such as web search to mitigate hallucination and audio search to switch timbres. Trained on millions of hours of speech and audio data, Step-Audio 2 delivers intelligence and expressiveness across diverse conversational scenarios. Evaluation results demonstrate that Step-Audio 2 achieves state-of-the-art performance on various audio understanding and conversational benchmarks compared to other open-source and commercial solutions. Please visit this https URL for more information. 

---
# Pixels to Principles: Probing Intuitive Physics Understanding in Multimodal Language Models 

**Authors**: Mohamad Ballout, Serwan Jassim, Elia Bruni  

**Link**: [PDF](https://arxiv.org/pdf/2507.16572)  

**Abstract**: This paper presents a systematic evaluation of state-of-the-art multimodal large language models (MLLMs) on intuitive physics tasks using the GRASP and IntPhys 2 datasets. We assess the open-source models InternVL 2.5, Qwen 2.5 VL, LLaVA-OneVision, and the proprietary Gemini 2.0 Flash Thinking, finding that even the latest models struggle to reliably distinguish physically plausible from implausible scenarios. To go beyond performance metrics, we conduct a probing analysis of model embeddings, extracting intermediate representations at key processing stages to examine how well task-relevant information is preserved. Our results show that, depending on task difficulty, a critical vision-language misalignment can emerge: vision encoders successfully capture physical plausibility cues, but this information is not effectively utilized by the language model, leading to failures in reasoning. This misalignment suggests that the primary limitation of MLLMs in intuitive physics tasks is not the vision component but the ineffective integration of visual and linguistic information. Our findings highlight vision-language alignment as a key area for improvement, offering insights for future MLLMs development. 

---
# Exploring Gender Bias in Large Language Models: An In-depth Dive into the German Language 

**Authors**: Kristin Gnadt, David Thulke, Simone Kopeinik, Ralf Schlüter  

**Link**: [PDF](https://arxiv.org/pdf/2507.16557)  

**Abstract**: In recent years, various methods have been proposed to evaluate gender bias in large language models (LLMs). A key challenge lies in the transferability of bias measurement methods initially developed for the English language when applied to other languages. This work aims to contribute to this research strand by presenting five German datasets for gender bias evaluation in LLMs. The datasets are grounded in well-established concepts of gender bias and are accessible through multiple methodologies. Our findings, reported for eight multilingual LLM models, reveal unique challenges associated with gender bias in German, including the ambiguous interpretation of male occupational terms and the influence of seemingly neutral nouns on gender perception. This work contributes to the understanding of gender bias in LLMs across languages and underscores the necessity for tailored evaluation frameworks. 

---
# Learning Text Styles: A Study on Transfer, Attribution, and Verification 

**Authors**: Zhiqiang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2507.16530)  

**Abstract**: This thesis advances the computational understanding and manipulation of text styles through three interconnected pillars: (1) Text Style Transfer (TST), which alters stylistic properties (e.g., sentiment, formality) while preserving content; (2)Authorship Attribution (AA), identifying the author of a text via stylistic fingerprints; and (3) Authorship Verification (AV), determining whether two texts share the same authorship. We address critical challenges in these areas by leveraging parameter-efficient adaptation of large language models (LLMs), contrastive disentanglement of stylistic features, and instruction-based fine-tuning for explainable verification. 

---
# Introducing Quality Estimation to Machine Translation Post-editing Workflow: An Empirical Study on Its Usefulness 

**Authors**: Siqi Liu, Guangrong Dai, Dechao Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.16515)  

**Abstract**: This preliminary study investigates the usefulness of sentence-level Quality Estimation (QE) in English-Chinese Machine Translation Post-Editing (MTPE), focusing on its impact on post-editing speed and student translators' perceptions. It also explores the interaction effects between QE and MT quality, as well as between QE and translation expertise. The findings reveal that QE significantly reduces post-editing time. The examined interaction effects were not significant, suggesting that QE consistently improves MTPE efficiency across medium- and high-quality MT outputs and among student translators with varying levels of expertise. In addition to indicating potentially problematic segments, QE serves multiple functions in MTPE, such as validating translators' evaluations of MT quality and enabling them to double-check translation outputs. However, interview data suggest that inaccurate QE may hinder post-editing processes. This research provides new insights into the strengths and limitations of QE, facilitating its more effective integration into MTPE workflows to enhance translators' productivity. 

---
# The Ever-Evolving Science Exam 

**Authors**: Junying Wang, Zicheng Zhang, Yijin Guo, Farong Wen, Ye Shen, Yingji Liang, Yalun Wu, Wenzhe Li, Chunyi Li, Zijian Chen, Qi Jia, Guangtao Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2507.16514)  

**Abstract**: As foundation models grow rapidly in capability and deployment, evaluating their scientific understanding becomes increasingly critical. Existing science benchmarks have made progress towards broad **Range**, wide **Reach**, and high **Rigor**, yet they often face two major challenges: **data leakage risks** that compromise benchmarking validity, and **evaluation inefficiency** due to large-scale testing. To address these issues, we introduce the **Ever-Evolving Science Exam (EESE)**, a dynamic benchmark designed to reliably assess scientific capabilities in foundation models. Our approach consists of two components: 1) a non-public **EESE-Pool** with over 100K expertly constructed science instances (question-answer pairs) across 5 disciplines and 500+ subfields, built through a multi-stage pipeline ensuring **Range**, **Reach**, and **Rigor**, 2) a periodically updated 500-instance subset **EESE**, sampled and validated to enable leakage-resilient, low-overhead evaluations. Experiments on 32 open- and closed-source models demonstrate that EESE effectively differentiates the strengths and weaknesses of models in scientific fields and cognitive dimensions. Overall, EESE provides a robust, scalable, and forward-compatible solution for science benchmark design, offering a realistic measure of how well foundation models handle science questions. The project page is at: this https URL. 

---
# Combining Language and Topic Models for Hierarchical Text Classification 

**Authors**: Jaco du Toit, Marcel Dunaiski  

**Link**: [PDF](https://arxiv.org/pdf/2507.16490)  

**Abstract**: Hierarchical text classification (HTC) is a natural language processing task which has the objective of categorising text documents into a set of classes from a predefined structured class hierarchy. Recent HTC approaches use various techniques to incorporate the hierarchical class structure information with the natural language understanding capabilities of pre-trained language models (PLMs) to improve classification performance. Furthermore, using topic models along with PLMs to extract features from text documents has been shown to be an effective approach for multi-label text classification tasks. The rationale behind the combination of these feature extractor models is that the PLM captures the finer-grained contextual and semantic information while the topic model obtains high-level representations which consider the corpus of documents as a whole. In this paper, we use a HTC approach which uses a PLM and a topic model to extract features from text documents which are used to train a classification model. Our objective is to determine whether the combination of the features extracted from the two models is beneficial to HTC performance in general. In our approach, the extracted features are passed through separate convolutional layers whose outputs are combined and passed to a label-wise attention mechanisms which obtains label-specific document representations by weighing the most important features for each class separately. We perform comprehensive experiments on three HTC benchmark datasets and show that using the features extracted from the topic model generally decreases classification performance compared to only using the features obtained by the PLM. In contrast to previous work, this shows that the incorporation of features extracted from topic models for text classification tasks should not be assumed beneficial. 

---
# ICR Probe: Tracking Hidden State Dynamics for Reliable Hallucination Detection in LLMs 

**Authors**: Zhenliang Zhang, Xinyu Hu, Huixuan Zhang, Junzhe Zhang, Xiaojun Wan  

**Link**: [PDF](https://arxiv.org/pdf/2507.16488)  

**Abstract**: Large language models (LLMs) excel at various natural language processing tasks, but their tendency to generate hallucinations undermines their reliability. Existing hallucination detection methods leveraging hidden states predominantly focus on static and isolated representations, overlooking their dynamic evolution across layers, which limits efficacy. To address this limitation, we shift the focus to the hidden state update process and introduce a novel metric, the ICR Score (Information Contribution to Residual Stream), which quantifies the contribution of modules to the hidden states' update. We empirically validate that the ICR Score is effective and reliable in distinguishing hallucinations. Building on these insights, we propose a hallucination detection method, the ICR Probe, which captures the cross-layer evolution of hidden states. Experimental results show that the ICR Probe achieves superior performance with significantly fewer parameters. Furthermore, ablation studies and case analyses offer deeper insights into the underlying mechanism of this method, improving its interpretability. 

---
# Towards Enforcing Company Policy Adherence in Agentic Workflows 

**Authors**: Naama Zwerdling, David Boaz, Ella Rabinovich, Guy Uziel, David Amid, Ateret Anaby-Tavor  

**Link**: [PDF](https://arxiv.org/pdf/2507.16459)  

**Abstract**: Large Language Model (LLM) agents hold promise for a flexible and scalable alternative to traditional business process automation, but struggle to reliably follow complex company policies. In this study we introduce a deterministic, transparent, and modular framework for enforcing business policy adherence in agentic workflows. Our method operates in two phases: (1) an offline buildtime stage that compiles policy documents into verifiable guard code associated with tool use, and (2) a runtime integration where these guards ensure compliance before each agent action. We demonstrate our approach on the challenging $\tau$-bench Airlines domain, showing encouraging preliminary results in policy enforcement, and further outline key challenges for real-world deployments. 

---
# Dutch CrowS-Pairs: Adapting a Challenge Dataset for Measuring Social Biases in Language Models for Dutch 

**Authors**: Elza Strazda, Gerasimos Spanakis  

**Link**: [PDF](https://arxiv.org/pdf/2507.16442)  

**Abstract**: Warning: This paper contains explicit statements of offensive stereotypes which might be upsetting.
Language models are prone to exhibiting biases, further amplifying unfair and harmful stereotypes. Given the fast-growing popularity and wide application of these models, it is necessary to ensure safe and fair language models. As of recent considerable attention has been paid to measuring bias in language models, yet the majority of studies have focused only on English language. A Dutch version of the US-specific CrowS-Pairs dataset for measuring bias in Dutch language models is introduced. The resulting dataset consists of 1463 sentence pairs that cover bias in 9 categories, such as Sexual orientation, Gender and Disability. The sentence pairs are composed of contrasting sentences, where one of the sentences concerns disadvantaged groups and the other advantaged groups. Using the Dutch CrowS-Pairs dataset, we show that various language models, BERTje, RobBERT, multilingual BERT, GEITje and Mistral-7B exhibit substantial bias across the various bias categories. Using the English and French versions of the CrowS-Pairs dataset, bias was evaluated in English (BERT and RoBERTa) and French (FlauBERT and CamemBERT) language models, and it was shown that English models exhibit the most bias, whereas Dutch models the least amount of bias. Additionally, results also indicate that assigning a persona to a language model changes the level of bias it exhibits. These findings highlight the variability of bias across languages and contexts, suggesting that cultural and linguistic factors play a significant role in shaping model biases. 

---
# PromptAL: Sample-Aware Dynamic Soft Prompts for Few-Shot Active Learning 

**Authors**: Hui Xiang, Jinqiao Shi, Ting Zhang, Xiaojie Zhao, Yong Liu, Yong Ma  

**Link**: [PDF](https://arxiv.org/pdf/2507.16424)  

**Abstract**: Active learning (AL) aims to optimize model training and reduce annotation costs by selecting the most informative samples for labeling. Typically, AL methods rely on the empirical distribution of labeled data to define the decision boundary and perform uncertainty or diversity estimation, subsequently identifying potential high-quality samples. In few-shot scenarios, the empirical distribution often diverges significantly from the target distribution, causing the decision boundary to shift away from its optimal position. However, existing methods overlook the role of unlabeled samples in enhancing the empirical distribution to better align with the target distribution, resulting in a suboptimal decision boundary and the selection of samples that inadequately represent the target distribution. To address this, we propose a hybrid AL framework, termed \textbf{PromptAL} (Sample-Aware Dynamic Soft \textbf{Prompts} for Few-Shot \textbf{A}ctive \textbf{L}earning). This framework accounts for the contribution of each unlabeled data point in aligning the current empirical distribution with the target distribution, thereby optimizing the decision boundary. Specifically, PromptAL first leverages unlabeled data to construct sample-aware dynamic soft prompts that adjust the model's predictive distribution and decision boundary. Subsequently, based on the adjusted decision boundary, it integrates uncertainty estimation with both global and local diversity to select high-quality samples that more accurately represent the target distribution. Experimental results on six in-domain and three out-of-domain datasets show that PromptAL achieves superior performance over nine baselines. Our codebase is openly accessible. 

---
# GG-BBQ: German Gender Bias Benchmark for Question Answering 

**Authors**: Shalaka Satheesh, Katrin Klug, Katharina Beckh, Héctor Allende-Cid, Sebastian Houben, Teena Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2507.16410)  

**Abstract**: Within the context of Natural Language Processing (NLP), fairness evaluation is often associated with the assessment of bias and reduction of associated harm. In this regard, the evaluation is usually carried out by using a benchmark dataset, for a task such as Question Answering, created for the measurement of bias in the model's predictions along various dimensions, including gender identity. In our work, we evaluate gender bias in German Large Language Models (LLMs) using the Bias Benchmark for Question Answering by Parrish et al. (2022) as a reference. Specifically, the templates in the gender identity subset of this English dataset were machine translated into German. The errors in the machine translated templates were then manually reviewed and corrected with the help of a language expert. We find that manual revision of the translation is crucial when creating datasets for gender bias evaluation because of the limitations of machine translation from English to a language such as German with grammatical gender. Our final dataset is comprised of two subsets: Subset-I, which consists of group terms related to gender identity, and Subset-II, where group terms are replaced with proper names. We evaluate several LLMs used for German NLP on this newly created dataset and report the accuracy and bias scores. The results show that all models exhibit bias, both along and against existing social stereotypes. 

---
# Re:Form -- Reducing Human Priors in Scalable Formal Software Verification with RL in LLMs: A Preliminary Study on Dafny 

**Authors**: Chuanhao Yan, Fengdi Che, Xuhan Huang, Xu Xu, Xin Li, Yizhi Li, Xingwei Qu, Jingzhe Shi, Zhuangzhuang He, Chenghua Lin, Yaodong Yang, Binhang Yuan, Hang Zhao, Yu Qiao, Bowen Zhou, Jie Fu  

**Link**: [PDF](https://arxiv.org/pdf/2507.16331)  

**Abstract**: Existing informal language-based (e.g., human language) Large Language Models (LLMs) trained with Reinforcement Learning (RL) face a significant challenge: their verification processes, which provide crucial training signals, are neither reliable nor scalable. In fact, the prevalent large proprietary models could hardly generate verifiable programs. A promising yet largely uncharted alternative is formal language-based reasoning. Grounding LLMs in rigorous formal systems where generative models operate in formal language spaces (e.g., Dafny) enables the automatic and mathematically provable verification of their reasoning processes and outcomes. This capability is pivotal for achieving large-scale, reliable formal software verification. It is a common practice to employ human-annotated chain-of-thought and other human priors to induce the reasoning and coding capabilities of LLMs. Unfortunately, it becomes unacceptably all-consuming to provide such priors for supervising complex programming tasks. In this work, we systematically explore ways to reduce human priors with the formal language, Dafny, as the main environment for our pilot study. Our pipeline mainly relies on introducing an automatic and scalable data curation pipeline, and careful RL designs integrated with feedback from the formal language verifier. We introduce DafnyComp, a benchmark of compositional formal programs with auto-formalized specifications for specification reasoning. Our supervised fine-tuning (SFT) stage enables even small models (e.g., 0.5B) to generate syntactically valid and verifiable Dafny code, surpassing proprietary models. RL with regularization further improves performance, achieving stronger generalization to out-of-domain tasks and outperforming all strong baselines on the challenging DafnyComp benchmark. 

---
# SpeLLM: Character-Level Multi-Head Decoding 

**Authors**: Amit Ben-Artzy, Roy Schwartz  

**Link**: [PDF](https://arxiv.org/pdf/2507.16323)  

**Abstract**: Scaling LLM vocabulary is often used to reduce input sequence length and alleviate attention's quadratic cost. Yet, current LLM architectures impose a critical bottleneck to this procedure: the output projection layer scales linearly with vocabulary size, rendering substantial expansion impractical. We propose SpeLLM, a method that decouples input and output vocabularies by predicting character-level strings through multiple output heads. In SpeLLM, each of the $k$ linear heads predicts a single character simultaneously, enabling the model to represent a much larger output space using smaller, independent linear heads. We present a self-distillation approach for converting a standard LLM to a SpeLLM. Our experiments with four pre-trained LLMs show their SpeLLM variants achieve competitive performance on downstream tasks while reducing runtime by 5.1% on average across models. Our approach provides a potential avenue for reducing LLM costs, while increasing support for underrepresented languages and domains. 

---
# Language Detection by Means of the Minkowski Norm: Identification Through Character Bigrams and Frequency Analysis 

**Authors**: Paul-Andrei Pogăcean, Sanda-Maria Avram  

**Link**: [PDF](https://arxiv.org/pdf/2507.16284)  

**Abstract**: The debate surrounding language identification has gained renewed attention in recent years, especially with the rapid evolution of AI-powered language models. However, the non-AI-based approaches to language identification have been overshadowed. This research explores a mathematical implementation of an algorithm for language determinism by leveraging monograms and bigrams frequency rankings derived from established linguistic research. The datasets used comprise texts varying in length, historical period, and genre, including short stories, fairy tales, and poems. Despite these variations, the method achieves over 80\% accuracy on texts shorter than 150 characters and reaches 100\% accuracy for longer texts and older writings. These results demonstrate that classical frequency-based approaches remain effective and scalable alternatives to AI-driven models for language detection. 

---
# Beyond Isolated Dots: Benchmarking Structured Table Construction as Deep Knowledge Extraction 

**Authors**: Tianyun Zhong, Guozhao Mo, Yanjiang Liu, Yihan Chen, Lingdi Kong, Xuanang Chen, Yaojie Lu, Hongyu Lin, Ben He, Le Sun  

**Link**: [PDF](https://arxiv.org/pdf/2507.16271)  

**Abstract**: With the emergence of large language models (LLMs), there is an expectation that LLMs can effectively extract explicit information from complex real-world documents (e.g., papers, reports). However, most LLMs generate paragraph-style answers that are chaotic, disorganized, and untraceable. To bridge this gap, we introduce the Arranged and Organized Extraction Benchmark (AOE), a new bilingual benchmark with data and documents of varying lengths designed to systematically evaluate the ability of LLMs to comprehend fragmented documents and reconstruct isolated information into one organized table. Unlike conventional text-to-table tasks, which rely on fixed schema and narrow task domains, AOE includes 11 carefully crafted tasks across three diverse domains, requiring models to generate context-specific schema tailored to varied input queries. In the experiment, we evaluated both open-source and closed-source state-of-the-art LLMs. The results show that even the most advanced models struggled significantly. The benchmark is available at this https URL. 

---
# iShumei-Chinchunmei at SemEval-2025 Task 4: A balanced forgetting and retention multi-task framework using effective unlearning loss 

**Authors**: Yujian Sun, Tian Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.16263)  

**Abstract**: As the Large Language Model (LLM) gains widespread adoption, increasing attention has been given to the challenge of making LLM forget non-compliant data memorized during its pre-training. Machine Unlearning focuses on efficiently erasing sensitive information from LLM under limited computational resources. To advance research in this area, SemEval 2025 Task 4: "Unlearning Sensitive Content from Large Language Models" introduces three unlearning datasets and establishes a benchmark by evaluating both forgetting effectiveness and the preservation of standard capabilities. In this work, we propose a more controllable forgetting loss, Effective Unlearning Loss, and explore its integration with various techniques to achieve more efficient and controlled unlearning. Our system ultimately ranked 5th on the competition leaderboard. 

---
# Efficient RL for optimizing conversation level outcomes with an LLM-based tutor 

**Authors**: Hyunji Nam, Omer Gottesman, Amy Zhang, Dean Foster, Emma Brunskill, Lyle Ungar  

**Link**: [PDF](https://arxiv.org/pdf/2507.16252)  

**Abstract**: Large language models (LLMs) built on existing reinforcement learning with human feedback (RLHF) frameworks typically optimize responses based on immediate turn-level human preferences. However, this approach falls short in multi-turn dialogue settings, such as online math tutoring. We propose a method to enhance LLM-based tutors by representing the dialogue history with a lower-dimensional latent state representation of a student and optimizing a long-term policy to determine high-level actions based on the latent state. The goal is to better align the tutor's behavior with the long-term objective of guiding the student towards solving a target math problem on their own. Our model is lightweight, requiring less computational resources than prior work of training the tutor policy end-to-end to directly output the tutor's next utterance. Our experiment results demonstrate that these modifications lead to improved long-term outcomes compared to prompting in LLM-simulated tutoring tasks. 

---
# FinResearchBench: A Logic Tree based Agent-as-a-Judge Evaluation Framework for Financial Research Agents 

**Authors**: Run Sun, Zuo Bai, Wentao Zhang, Yuxiang Zhang, Li Zhao, Shan Sun, Zhengwen Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2507.16248)  

**Abstract**: Recently, AI agents are rapidly evolving in intelligence and widely used in professional research applications, such as STEM, software development, finance, etc. Among these AI agents, deep research agent is a key category as it can perform long-horizon tasks and solve problems of greater complexity. However, there are few evaluation frameworks and benchmarks that systematically and automatically investigate the capabilities of these research agents. Furthermore, financial research problems have distinct complexity and subtlety. To fill in the gap, we propose FinResearchBench, which is a logic tree based Agent-as-a-Judge and targets specifically for the financial research agents. It provides a comprehensive and automatic assessment of the research agents across 7 key types of tasks in the financial research domain. The contributions of this work are two-folded: (1) the first and innovative Agent-as-a-Judge system that extracts the logic tree of the research outcome and uses it as the intermediate information to present a comprehensive, reliable and robust evaluation; (2) finance oriented that it covers 70 typical financial research questions, spreading across 7 frequently encountered types of tasks in the domain. 

---
# Towards Compute-Optimal Many-Shot In-Context Learning 

**Authors**: Shahriar Golchin, Yanfei Chen, Rujun Han, Manan Gandhi, Tianli Yu, Swaroop Mishra, Mihai Surdeanu, Rishabh Agarwal, Chen-Yu Lee, Tomas Pfister  

**Link**: [PDF](https://arxiv.org/pdf/2507.16217)  

**Abstract**: Long-context large language models (LLMs) are able to process inputs containing up to several million tokens. In the scope of in-context learning (ICL), this translates into using hundreds/thousands of demonstrations in the input prompt, enabling many-shot ICL. In practice, a fixed set of demonstrations is often selected at random in many-shot settings due to (1) high inference costs, (2) the benefits of caching and reusing computations, and (3) the similar performance offered by this strategy compared to others when scaled. In this work, we propose two straightforward strategies for demonstration selection in many-shot ICL that improve performance with minimal computational overhead. Our first method combines a small number of demonstrations, selected based on their similarity to each test sample, with a disproportionately larger set of random demonstrations that are cached. The second strategy improves the first by replacing random demonstrations with those selected using centroids derived from test sample representations via k-means clustering. Our experiments with Gemini Pro and Flash across several datasets indicate that our strategies consistently outperform random selection and surpass or match the most performant selection approach while supporting caching and reducing inference cost by up to an order of magnitude. We also show that adjusting the proportion of demonstrations selected based on different criteria can balance performance and inference cost in many-shot ICL. 

---
# WakenLLM: A Fine-Grained Benchmark for Evaluating LLM Reasoning Potential and Reasoning Process Stability 

**Authors**: Zipeng Ling, Yuehao Tang, Shuliang Liu, Junqi Yang, Shenghong Fu, Yao Wan, Kejia Huang, Zhichao Hou, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2507.16199)  

**Abstract**: Large Language Models (LLMs) frequently output the label \emph{Unknown}, yet current evaluations focus almost exclusively on whether such answers are \emph{honest} rather than why they arise. This blurs two distinct cases: (i) an input that is genuinely indeterminate and (ii) a solvable problem that the model fails to resolve. We call this phenomenon \emph{Vague Perception}. And thus we introduce a framework that quantifies the proportion of \emph{Unknown} responses attributable to model incapacity and tests whether guided stimulation can convert them into either correct (\emph{Known}) or intrinsically indeterminate outcomes. By separating these sources of uncertainty, our method provides a clearer picture of LLM reasoning limits and their potential for improvement. As we get a theoretical accuracy of reasoning task on different LLMs, we apply different methods to test whether the model can reach the accuracy given a baseline framework. Our work is meaningful in exploring the true reasoning ability of LLMs and providing a new perspective on solving the \emph{Vague Perception} phenomenon. 

---
# Do Large Language Models Have a Planning Theory of Mind? Evidence from MindGames: a Multi-Step Persuasion Task 

**Authors**: Jared Moore, Ned Cooper, Rasmus Overmark, Beba Cibralic, Nick Haber, Cameron R. Jones  

**Link**: [PDF](https://arxiv.org/pdf/2507.16196)  

**Abstract**: Recent evidence suggests Large Language Models (LLMs) display Theory of Mind (ToM) abilities. Most ToM experiments place participants in a spectatorial role, wherein they predict and interpret other agents' behavior. However, human ToM also contributes to dynamically planning action and strategically intervening on others' mental states. We present MindGames: a novel `planning theory of mind' (PToM) task which requires agents to infer an interlocutor's beliefs and desires to persuade them to alter their behavior. Unlike previous evaluations, we explicitly evaluate use cases of ToM. We find that humans significantly outperform o1-preview (an LLM) at our PToM task (11% higher; $p=0.006$). We hypothesize this is because humans have an implicit causal model of other agents (e.g., they know, as our task requires, to ask about people's preferences). In contrast, o1-preview outperforms humans in a baseline condition which requires a similar amount of planning but minimal mental state inferences (e.g., o1-preview is better than humans at planning when already given someone's preferences). These results suggest a significant gap between human-like social reasoning and LLM abilities. 

---
# BIDWESH: A Bangla Regional Based Hate Speech Detection Dataset 

**Authors**: Azizul Hakim Fayaz, MD. Shorif Uddin, Rayhan Uddin Bhuiyan, Zakia Sultana, Md. Samiul Islam, Bidyarthi Paul, Tashreef Muhammad, Shahriar Manzoor  

**Link**: [PDF](https://arxiv.org/pdf/2507.16183)  

**Abstract**: Hate speech on digital platforms has become a growing concern globally, especially in linguistically diverse countries like Bangladesh, where regional dialects play a major role in everyday communication. Despite progress in hate speech detection for standard Bangla, Existing datasets and systems fail to address the informal and culturally rich expressions found in dialects such as Barishal, Noakhali, and Chittagong. This oversight results in limited detection capability and biased moderation, leaving large sections of harmful content unaccounted for. To address this gap, this study introduces BIDWESH, the first multi-dialectal Bangla hate speech dataset, constructed by translating and annotating 9,183 instances from the BD-SHS corpus into three major regional dialects. Each entry was manually verified and labeled for hate presence, type (slander, gender, religion, call to violence), and target group (individual, male, female, group), ensuring linguistic and contextual accuracy. The resulting dataset provides a linguistically rich, balanced, and inclusive resource for advancing hate speech detection in Bangla. BIDWESH lays the groundwork for the development of dialect-sensitive NLP tools and contributes significantly to equitable and context-aware content moderation in low-resource language settings. 

---
# Efficient Compositional Multi-tasking for On-device Large Language Models 

**Authors**: Ondrej Bohdal, Mete Ozay, Jijoong Moon, Kyeng-Hun Lee, Hyeonmok Ko, Umberto Michieli  

**Link**: [PDF](https://arxiv.org/pdf/2507.16083)  

**Abstract**: Adapter parameters provide a mechanism to modify the behavior of machine learning models and have gained significant popularity in the context of large language models (LLMs) and generative AI. These parameters can be merged to support multiple tasks via a process known as task merging. However, prior work on merging in LLMs, particularly in natural language processing, has been limited to scenarios where each test example addresses only a single task. In this paper, we focus on on-device settings and study the problem of text-based compositional multi-tasking, where each test example involves the simultaneous execution of multiple tasks. For instance, generating a translated summary of a long text requires solving both translation and summarization tasks concurrently. To facilitate research in this setting, we propose a benchmark comprising four practically relevant compositional tasks. We also present an efficient method (Learnable Calibration) tailored for on-device applications, where computational resources are limited, emphasizing the need for solutions that are both resource-efficient and high-performing. Our contributions lay the groundwork for advancing the capabilities of LLMs in real-world multi-tasking scenarios, expanding their applicability to complex, resource-constrained use cases. 

---
# The Prompt Makes the Person(a): A Systematic Evaluation of Sociodemographic Persona Prompting for Large Language Models 

**Authors**: Marlene Lutz, Indira Sen, Georg Ahnert, Elisa Rogers, Markus Strohmaier  

**Link**: [PDF](https://arxiv.org/pdf/2507.16076)  

**Abstract**: Persona prompting is increasingly used in large language models (LLMs) to simulate views of various sociodemographic groups. However, how a persona prompt is formulated can significantly affect outcomes, raising concerns about the fidelity of such simulations. Using five open-source LLMs, we systematically examine how different persona prompt strategies, specifically role adoption formats and demographic priming strategies, influence LLM simulations across 15 intersectional demographic groups in both open- and closed-ended tasks. Our findings show that LLMs struggle to simulate marginalized groups, particularly nonbinary, Hispanic, and Middle Eastern identities, but that the choice of demographic priming and role adoption strategy significantly impacts their portrayal. Specifically, we find that prompting in an interview-style format and name-based priming can help reduce stereotyping and improve alignment. Surprisingly, smaller models like OLMo-2-7B outperform larger ones such as Llama-3.3-70B. Our findings offer actionable guidance for designing sociodemographic persona prompts in LLM-based simulation studies. 

---
# Deep Researcher with Test-Time Diffusion 

**Authors**: Rujun Han, Yanfei Chen, Zoey CuiZhu, Lesly Miculicich, Guan Sun, Yuanjun Bi, Weiming Wen, Hui Wan, Chunfeng Wen, Solène Maître, George Lee, Vishy Tirumalashetty, Emily Xue, Zizhao Zhang, Salem Haykal, Burak Gokturk, Tomas Pfister, Chen-Yu Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.16075)  

**Abstract**: Deep research agents, powered by Large Language Models (LLMs), are rapidly advancing; yet, their performance often plateaus when generating complex, long-form research reports using generic test-time scaling algorithms. Drawing inspiration from the iterative nature of human research, which involves cycles of searching, reasoning, and revision, we propose the Test-Time Diffusion Deep Researcher (TTD-DR). This novel framework conceptualizes research report generation as a diffusion process. TTD-DR initiates this process with a preliminary draft, an updatable skeleton that serves as an evolving foundation to guide the research direction. The draft is then iteratively refined through a "denoising" process, which is dynamically informed by a retrieval mechanism that incorporates external information at each step. The core process is further enhanced by a self-evolutionary algorithm applied to each component of the agentic workflow, ensuring the generation of high-quality context for the diffusion process. This draft-centric design makes the report writing process more timely and coherent while reducing information loss during the iterative search process. We demonstrate that our TTD-DR achieves state-of-the-art results on a wide array of benchmarks that require intensive search and multi-hop reasoning, significantly outperforming existing deep research agents. 

---
# AutoMeet: a proof-of-concept study of genAI to automate meetings in automotive engineering 

**Authors**: Simon Baeuerle, Max Radyschevski, Ulrike Pado  

**Link**: [PDF](https://arxiv.org/pdf/2507.16054)  

**Abstract**: In large organisations, knowledge is mainly shared in meetings, which takes up significant amounts of work time. Additionally, frequent in-person meetings produce inconsistent documentation -- official minutes, personal notes, presentations may or may not exist. Shared information therefore becomes hard to retrieve outside of the meeting, necessitating lengthy updates and high-frequency meeting schedules.
Generative Artificial Intelligence (genAI) models like Large Language Models (LLMs) exhibit an impressive performance on spoken and written language processing. This motivates a practical usage of genAI for knowledge management in engineering departments: using genAI for transcribing meetings and integrating heterogeneous additional information sources into an easily usable format for ad-hoc searches.
We implement an end-to-end pipeline to automate the entire meeting documentation workflow in a proof-of-concept state: meetings are recorded and minutes are created by genAI. These are further made easily searchable through a chatbot interface. The core of our work is to test this genAI-based software tooling in a real-world engineering department and collect extensive survey data on both ethical and technical aspects. Direct feedback from this real-world setup points out both opportunities and risks: a) users agree that the effort for meetings could be significantly reduced with the help of genAI models, b) technical aspects are largely solved already, c) organizational aspects are crucial for a successful ethical usage of such a system. 

---
# mRAKL: Multilingual Retrieval-Augmented Knowledge Graph Construction for Low-Resourced Languages 

**Authors**: Hellina Hailu Nigatu, Min Li, Maartje ter Hoeve, Saloni Potdar, Sarah Chasins  

**Link**: [PDF](https://arxiv.org/pdf/2507.16011)  

**Abstract**: Knowledge Graphs represent real-world entities and the relationships between them. Multilingual Knowledge Graph Construction (mKGC) refers to the task of automatically constructing or predicting missing entities and links for knowledge graphs in a multilingual setting. In this work, we reformulate the mKGC task as a Question Answering (QA) task and introduce mRAKL: a Retrieval-Augmented Generation (RAG) based system to perform mKGC. We achieve this by using the head entity and linking relation in a question, and having our model predict the tail entity as an answer. Our experiments focus primarily on two low-resourced languages: Tigrinya and Amharic. We experiment with using higher-resourced languages Arabic and English for cross-lingual transfer. With a BM25 retriever, we find that the RAG-based approach improves performance over a no-context setting. Further, our ablation studies show that with an idealized retrieval system, mRAKL improves accuracy by 4.92 and 8.79 percentage points for Tigrinya and Amharic, respectively. 

---
# Help Me Write a Story: Evaluating LLMs' Ability to Generate Writing Feedback 

**Authors**: Hannah Rashkin, Elizabeth Clark, Fantine Huot, Mirella Lapata  

**Link**: [PDF](https://arxiv.org/pdf/2507.16007)  

**Abstract**: Can LLMs provide support to creative writers by giving meaningful writing feedback? In this paper, we explore the challenges and limitations of model-generated writing feedback by defining a new task, dataset, and evaluation frameworks. To study model performance in a controlled manner, we present a novel test set of 1,300 stories that we corrupted to intentionally introduce writing issues. We study the performance of commonly used LLMs in this task with both automatic and human evaluation metrics. Our analysis shows that current models have strong out-of-the-box behavior in many respects -- providing specific and mostly accurate writing feedback. However, models often fail to identify the biggest writing issue in the story and to correctly decide when to offer critical vs. positive feedback. 

---
# Learning without training: The implicit dynamics of in-context learning 

**Authors**: Benoit Dherin, Michael Munn, Hanna Mazzawi, Michael Wunder, Javier Gonzalvo  

**Link**: [PDF](https://arxiv.org/pdf/2507.16003)  

**Abstract**: One of the most striking features of Large Language Models (LLM) is their ability to learn in context. Namely at inference time an LLM is able to learn new patterns without any additional weight update when these patterns are presented in the form of examples in the prompt, even if these patterns were not seen during training. The mechanisms through which this can happen are still largely unknown. In this work, we show that the stacking of a self-attention layer with an MLP, allows the transformer block to implicitly modify the weights of the MLP layer according to the context. We argue through theory and experimentation that this simple mechanism may be the reason why LLMs can learn in context and not only during training. Specifically, we show under mild simplifying assumptions how a transformer block implicitly transforms a context into a low-rank weight-update of the MLP layer. 

---
# Enhancing Hindi NER in Low Context: A Comparative study of Transformer-based models with vs. without Retrieval Augmentation 

**Authors**: Sumit Singh, Rohit Mishra, Uma Shanker Tiwary  

**Link**: [PDF](https://arxiv.org/pdf/2507.16002)  

**Abstract**: One major challenge in natural language processing is named entity recognition (NER), which identifies and categorises named entities in textual input. In order to improve NER, this study investigates a Hindi NER technique that makes use of Hindi-specific pretrained encoders (MuRIL and XLM-R) and Generative Models ( Llama-2-7B-chat-hf (Llama2-7B), Llama-2-70B-chat-hf (Llama2-70B), Llama-3-70B-Instruct (Llama3-70B) and GPT3.5-turbo), and augments the data with retrieved data from external relevant contexts, notably from Wikipedia. We have fine-tuned MuRIL, XLM-R and Llama2-7B with and without RA. However, Llama2-70B, lama3-70B and GPT3.5-turbo are utilised for few-shot NER generation. Our investigation shows that the mentioned language models (LMs) with Retrieval Augmentation (RA) outperform baseline methods that don't incorporate RA in most cases. The macro F1 scores for MuRIL and XLM-R are 0.69 and 0.495, respectively, without RA and increase to 0.70 and 0.71, respectively, in the presence of RA. Fine-tuned Llama2-7B outperforms Llama2-7B by a significant margin. On the other hand the generative models which are not fine-tuned also perform better with augmented data. GPT3.5-turbo adopted RA well; however, Llama2-70B and llama3-70B did not adopt RA with our retrieval context. The findings show that RA significantly improves performance, especially for low-context data. This study adds significant knowledge about how best to use data augmentation methods and pretrained models to enhance NER performance, particularly in languages with limited resources. 

---
# Small Edits, Big Consequences: Telling Good from Bad Robustness in Large Language Models 

**Authors**: Altynbek Ismailov, Salia Asanova  

**Link**: [PDF](https://arxiv.org/pdf/2507.15868)  

**Abstract**: Large language models (LLMs) now write code in settings where misreading a single word can break safety or cost money, yet we still expect them to overlook stray typos. To probe where useful robustness ends and harmful insensitivity begins, we compile 50 LeetCode problems and craft three minimal prompt perturbations that should vary in importance: (i) progressive underspecification deleting 10 % of words per step; (ii) lexical flip swapping a pivotal quantifier ("max" to "min"); and (iii) jargon inflation replacing a common noun with an obscure technical synonym. Six frontier models, including three "reasoning-tuned" versions, solve each mutated prompt, and their Python outputs are checked against the original test suites to reveal whether they reused the baseline solution or adapted. Among 11 853 generations we observe a sharp double asymmetry. Models remain correct in 85 % of cases even after 90 % of the prompt is missing, showing over-robustness to underspecification, yet only 54 % react to a single quantifier flip that reverses the task, with reasoning-tuned variants even less sensitive than their bases. Jargon edits lie in between, passing through 56 %. Current LLMs thus blur the line between harmless noise and meaning - changing edits, often treating both as ignorable. Masking salient anchors such as function names can force re - evaluation. We advocate evaluation and training protocols that reward differential sensitivity: stay steady under benign noise but adapt - or refuse - when semantics truly change. 

---
# Adversarial Demonstration Learning for Low-resource NER Using Dual Similarity 

**Authors**: Guowen Yuan, Tien-Hsuan Wu, Lianghao Xia, Ben Kao  

**Link**: [PDF](https://arxiv.org/pdf/2507.15864)  

**Abstract**: We study the problem of named entity recognition (NER) based on demonstration learning in low-resource scenarios. We identify two issues in demonstration construction and model training. Firstly, existing methods for selecting demonstration examples primarily rely on semantic similarity; We show that feature similarity can provide significant performance improvement. Secondly, we show that the NER tagger's ability to reference demonstration examples is generally inadequate. We propose a demonstration and training approach that effectively addresses these issues. For the first issue, we propose to select examples by dual similarity, which comprises both semantic similarity and feature similarity. For the second issue, we propose to train an NER model with adversarial demonstration such that the model is forced to refer to the demonstrations when performing the tagging task. We conduct comprehensive experiments in low-resource NER tasks, and the results demonstrate that our method outperforms a range of methods. 

---
# eSapiens's DEREK Module: Deep Extraction & Reasoning Engine for Knowledge with LLMs 

**Authors**: Isaac Shi, Zeyuan Li, Fan Liu, Wenli Wang, Lewei He, Yang Yang, Tianyu Shi  

**Link**: [PDF](https://arxiv.org/pdf/2507.15863)  

**Abstract**: We present the DEREK (Deep Extraction & Reasoning Engine for Knowledge) Module, a secure and scalable Retrieval-Augmented Generation pipeline designed specifically for enterprise document question answering. Designed and implemented by eSapiens, the system ingests heterogeneous content (PDF, Office, web), splits it into 1,000-token overlapping chunks, and indexes them in a hybrid HNSW+BM25 store. User queries are refined by GPT-4o, retrieved via combined vector+BM25 search, reranked with Cohere, and answered by an LLM using CO-STAR prompt engineering. A LangGraph verifier enforces citation overlap, regenerating answers until every claim is grounded. On four LegalBench subsets, 1000-token chunks improve Recall@50 by approximately 1 pp and hybrid+rerank boosts Precision@10 by approximately 7 pp; the verifier raises TRACe Utilization above 0.50 and limits unsupported statements to less than 3%. All components run in containers, enforce end-to-end TLS 1.3 and AES-256. These results demonstrate that the DEREK module delivers accurate, traceable, and production-ready document QA with minimal operational overhead. The module is designed to meet enterprise demands for secure, auditable, and context-faithful retrieval, providing a reliable baseline for high-stakes domains such as legal and finance. 

---
# Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty 

**Authors**: Mehul Damani, Isha Puri, Stewart Slocum, Idan Shenfeld, Leshem Choshen, Yoon Kim, Jacob Andreas  

**Link**: [PDF](https://arxiv.org/pdf/2507.16806)  

**Abstract**: When language models (LMs) are trained via reinforcement learning (RL) to generate natural language "reasoning chains", their performance improves on a variety of difficult question answering tasks. Today, almost all successful applications of RL for reasoning use binary reward functions that evaluate the correctness of LM outputs. Because such reward functions do not penalize guessing or low-confidence outputs, they often have the unintended side-effect of degrading calibration and increasing the rate at which LMs generate incorrect responses (or "hallucinate") in other problem domains. This paper describes RLCR (Reinforcement Learning with Calibration Rewards), an approach to training reasoning models that jointly improves accuracy and calibrated confidence estimation. During RLCR, LMs generate both predictions and numerical confidence estimates after reasoning. They are trained to optimize a reward function that augments a binary correctness score with a Brier score -- a scoring rule for confidence estimates that incentivizes calibrated prediction. We first prove that this reward function (or any analogous reward function that uses a bounded, proper scoring rule) yields models whose predictions are both accurate and well-calibrated. We next show that across diverse datasets, RLCR substantially improves calibration with no loss in accuracy, on both in-domain and out-of-domain evaluations -- outperforming both ordinary RL training and classifiers trained to assign post-hoc confidence scores. While ordinary RL hurts calibration, RLCR improves it. Finally, we demonstrate that verbalized confidence can be leveraged at test time to improve accuracy and calibration via confidence-weighted scaling methods. Our results show that explicitly optimizing for calibration can produce more generally reliable reasoning models. 

---
# Steering Out-of-Distribution Generalization with Concept Ablation Fine-Tuning 

**Authors**: Helena Casademunt, Caden Juang, Adam Karvonen, Samuel Marks, Senthooran Rajamanoharan, Neel Nanda  

**Link**: [PDF](https://arxiv.org/pdf/2507.16795)  

**Abstract**: Fine-tuning large language models (LLMs) can lead to unintended out-of-distribution generalization. Standard approaches to this problem rely on modifying training data, for example by adding data that better specify the intended generalization. However, this is not always practical. We introduce Concept Ablation Fine-Tuning (CAFT), a technique that leverages interpretability tools to control how LLMs generalize from fine-tuning, without needing to modify the training data or otherwise use data from the target distribution. Given a set of directions in an LLM's latent space corresponding to undesired concepts, CAFT works by ablating these concepts with linear projections during fine-tuning, steering the model away from unintended generalizations. We successfully apply CAFT to three fine-tuning tasks, including emergent misalignment, a phenomenon where LLMs fine-tuned on a narrow task generalize to give egregiously misaligned responses to general questions. Without any changes to the fine-tuning data, CAFT reduces misaligned responses by 10x without degrading performance on the training distribution. Overall, CAFT represents a novel approach for steering LLM generalization without modifying training data. 

---
# Zebra-CoT: A Dataset for Interleaved Vision Language Reasoning 

**Authors**: Ang Li, Charles Wang, Kaiyu Yue, Zikui Cai, Ollie Liu, Deqing Fu, Peng Guo, Wang Bill Zhu, Vatsal Sharan, Robin Jia, Willie Neiswanger, Furong Huang, Tom Goldstein, Micah Goldblum  

**Link**: [PDF](https://arxiv.org/pdf/2507.16746)  

**Abstract**: Humans often use visual aids, for example diagrams or sketches, when solving complex problems. Training multimodal models to do the same, known as Visual Chain of Thought (Visual CoT), is challenging due to: (1) poor off-the-shelf visual CoT performance, which hinders reinforcement learning, and (2) the lack of high-quality visual CoT training data. We introduce $\textbf{Zebra-CoT}$, a diverse large-scale dataset with 182,384 samples, containing logically coherent interleaved text-image reasoning traces. We focus on four categories of tasks where sketching or visual reasoning is especially natural, spanning scientific questions such as geometry, physics, and algorithms; 2D visual reasoning tasks like visual search and jigsaw puzzles; 3D reasoning tasks including 3D multi-hop inference, embodied and robot planning; visual logic problems and strategic games like chess. Fine-tuning the Anole-7B model on the Zebra-CoT training corpus results in an improvement of +12% in our test-set accuracy and yields up to +13% performance gain on standard VLM benchmark evaluations. Fine-tuning Bagel-7B yields a model that generates high-quality interleaved visual reasoning chains, underscoring Zebra-CoT's effectiveness for developing multimodal reasoning abilities. We open-source our dataset and models to support development and evaluation of visual CoT. 

---
# Experience is the Best Teacher: Grounding VLMs for Robotics through Self-Generated Memory 

**Authors**: Guowei Lan, Kaixian Qu, René Zurbrügg, Changan Chen, Christopher E. Mower, Haitham Bou-Ammar, Marco Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2507.16713)  

**Abstract**: Vision-language models (VLMs) have been widely adopted in robotics to enable autonomous planning. However, grounding VLMs, originally trained on internet data, to diverse real-world robots remains a challenge. This paper presents ExpTeach, a framework that grounds VLMs to physical robots by building a self-generated memory of real-world experiences. In ExpTeach, the VLM autonomously plans actions, verifies outcomes, reflects on failures, and adapts robot behaviors in a closed loop. The self-generated experiences during this process are then summarized into a long-term memory, enabling retrieval of learned knowledge to guide future tasks via retrieval-augmented generation (RAG). Additionally, ExpTeach enhances the spatial understanding of VLMs with an on-demand image annotation module. In experiments, we show that reflection improves success rates from 36% to 84% on four challenging robotic tasks and observe the emergence of intelligent object interactions, including creative tool use. Across extensive tests on 12 real-world scenarios (including eight unseen ones), we find that grounding with long-term memory boosts single-trial success rates from 22% to 80%, demonstrating the effectiveness and generalizability of ExpTeach. 

---
# Scaling Linear Attention with Sparse State Expansion 

**Authors**: Yuqi Pan, Yongqi An, Zheng Li, Yuhong Chou, Ruijie Zhu, Xiaohui Wang, Mingxuan Wang, Jinqiao Wang, Guoqi Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.16577)  

**Abstract**: The Transformer architecture, despite its widespread success, struggles with long-context scenarios due to quadratic computation and linear memory growth. While various linear attention variants mitigate these efficiency constraints by compressing context into fixed-size states, they often degrade performance in tasks such as in-context retrieval and reasoning. To address this limitation and achieve more effective context compression, we propose two key innovations. First, we introduce a row-sparse update formulation for linear attention by conceptualizing state updating as information classification. This enables sparse state updates via softmax-based top-$k$ hard classification, thereby extending receptive fields and reducing inter-class interference. Second, we present Sparse State Expansion (SSE) within the sparse framework, which expands the contextual state into multiple partitions, effectively decoupling parameter size from state capacity while maintaining the sparse classification paradigm. Our design, supported by efficient parallelized implementations, yields effective classification and discriminative state representations. We extensively validate SSE in both pure linear and hybrid (SSE-H) architectures across language modeling, in-context retrieval, and mathematical reasoning benchmarks. SSE demonstrates strong retrieval performance and scales favorably with state size. Moreover, after reinforcement learning (RL) training, our 2B SSE-H model achieves state-of-the-art mathematical reasoning performance among small reasoning models, scoring 64.7 on AIME24 and 51.3 on AIME25, significantly outperforming similarly sized open-source Transformers. These results highlight SSE as a promising and efficient architecture for long-context modeling. 

---
# Frontier AI Risk Management Framework in Practice: A Risk Analysis Technical Report 

**Authors**: Shanghai AI Lab, Xiaoyang Chen, Yunhao Chen, Zeren Chen, Zhiyun Chen, Hanyun Cui, Yawen Duan, Jiaxuan Guo, Qi Guo, Xuhao Hu, Hong Huang, Lige Huang, Chunxiao Li, Juncheng Li, Qihao Lin, Dongrui Liu, Xinmin Liu, Zicheng Liu, Chaochao Lu, Xiaoya Lu, Jingjing Qu, Qibing Ren, Jing Shao, Jingwei Shi, Jingwei Sun, Peng Wang, Weibing Wang, Jia Xu, Lewen Yan, Xiao Yu, Yi Yu, Boxuan Zhang, Jie Zhang, Weichen Zhang, Zhijie Zheng, Tianyi Zhou, Bowen Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.16534)  

**Abstract**: To understand and identify the unprecedented risks posed by rapidly advancing artificial intelligence (AI) models, this report presents a comprehensive assessment of their frontier risks. Drawing on the E-T-C analysis (deployment environment, threat source, enabling capability) from the Frontier AI Risk Management Framework (v1.0) (SafeWork-F1-Framework), we identify critical risks in seven areas: cyber offense, biological and chemical risks, persuasion and manipulation, uncontrolled autonomous AI R\&D, strategic deception and scheming, self-replication, and collusion. Guided by the "AI-$45^\circ$ Law," we evaluate these risks using "red lines" (intolerable thresholds) and "yellow lines" (early warning indicators) to define risk zones: green (manageable risk for routine deployment and continuous monitoring), yellow (requiring strengthened mitigations and controlled deployment), and red (necessitating suspension of development and/or deployment). Experimental results show that all recent frontier AI models reside in green and yellow zones, without crossing red lines. Specifically, no evaluated models cross the yellow line for cyber offense or uncontrolled AI R\&D risks. For self-replication, and strategic deception and scheming, most models remain in the green zone, except for certain reasoning models in the yellow zone. In persuasion and manipulation, most models are in the yellow zone due to their effective influence on humans. For biological and chemical risks, we are unable to rule out the possibility of most models residing in the yellow zone, although detailed threat modeling and in-depth assessment are required to make further claims. This work reflects our current understanding of AI frontier risks and urges collective action to mitigate these challenges. 

---
# C2-Evo: Co-Evolving Multimodal Data and Model for Self-Improving Reasoning 

**Authors**: Xiuwei Chen, Wentao Hu, Hanhui Li, Jun Zhou, Zisheng Chen, Meng Cao, Yihan Zeng, Kui Zhang, Yu-Jie Yuan, Jianhua Han, Hang Xu, Xiaodan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2507.16518)  

**Abstract**: Recent advances in multimodal large language models (MLLMs) have shown impressive reasoning capabilities. However, further enhancing existing MLLMs necessitates high-quality vision-language datasets with carefully curated task complexities, which are both costly and challenging to scale. Although recent self-improving models that iteratively refine themselves offer a feasible solution, they still suffer from two core challenges: (i) most existing methods augment visual or textual data separately, resulting in discrepancies in data complexity (e.g., over-simplified diagrams paired with redundant textual descriptions); and (ii) the evolution of data and models is also separated, leading to scenarios where models are exposed to tasks with mismatched difficulty levels. To address these issues, we propose C2-Evo, an automatic, closed-loop self-improving framework that jointly evolves both training data and model capabilities. Specifically, given a base dataset and a base model, C2-Evo enhances them by a cross-modal data evolution loop and a data-model evolution loop. The former loop expands the base dataset by generating complex multimodal problems that combine structured textual sub-problems with iteratively specified geometric diagrams, while the latter loop adaptively selects the generated problems based on the performance of the base model, to conduct supervised fine-tuning and reinforcement learning alternately. Consequently, our method continuously refines its model and training data, and consistently obtains considerable performance gains across multiple mathematical reasoning benchmarks. Our code, models, and datasets will be released. 

---
# MMS Player: an open source software for parametric data-driven animation of Sign Language avatars 

**Authors**: Fabrizio Nunnari, Shailesh Mishra, Patrick Gebhard  

**Link**: [PDF](https://arxiv.org/pdf/2507.16463)  

**Abstract**: This paper describes the MMS-Player, an open source software able to synthesise sign language animations from a novel sign language representation format called MMS (MultiModal Signstream). The MMS enhances gloss-based representations by adding information on parallel execution of signs, timing, and inflections. The implementation consists of Python scripts for the popular Blender 3D authoring tool and can be invoked via command line or HTTP API. Animations can be rendered as videos or exported in other popular 3D animation exchange formats. The software is freely available under GPL-3.0 license at this https URL. 

---
# WhatsApp Tiplines and Multilingual Claims in the 2021 Indian Assembly Elections 

**Authors**: Gautam Kishore Shahi, Scot A. Hale  

**Link**: [PDF](https://arxiv.org/pdf/2507.16298)  

**Abstract**: WhatsApp tiplines, first launched in 2019 to combat misinformation, enable users to interact with fact-checkers to verify misleading content. This study analyzes 580 unique claims (tips) from 451 users, covering both high-resource languages (English, Hindi) and a low-resource language (Telugu) during the 2021 Indian assembly elections using a mixed-method approach. We categorize the claims into three categories, election, COVID-19, and others, and observe variations across languages. We compare content similarity through frequent word analysis and clustering of neural sentence embeddings. We also investigate user overlap across languages and fact-checking organizations. We measure the average time required to debunk claims and inform tipline users. Results reveal similarities in claims across languages, with some users submitting tips in multiple languages to the same fact-checkers. Fact-checkers generally require a couple of days to debunk a new claim and share the results with users. Notably, no user submits claims to multiple fact-checking organizations, indicating that each organization maintains a unique audience. We provide practical recommendations for using tiplines during elections with ethical consideration of users' information. 

---
# Characterizing Online Activities Contributing to Suicide Mortality among Youth 

**Authors**: Aparna Ananthasubramaniam, Elyse J. Thulin, Viktoryia Kalesnikava, Silas Falde, Jonathan Kertawidjaja, Lily Johns, Alejandro Rodríguez-Putnam, Emma Spring, Kara Zivin, Briana Mezuk  

**Link**: [PDF](https://arxiv.org/pdf/2507.16185)  

**Abstract**: The recent rise in youth suicide highlights the urgent need to understand how online experiences contribute to this public health issue. Our mixed-methods approach responds to this challenge by developing a set of themes focused on risk factors for suicide mortality in online spaces among youth ages 10-24, and a framework to model these themes at scale. Using 29,124 open text summaries of death investigations between 2013-2022, we conducted a thematic analysis to identify 12 types of online activities that were considered by investigators or next of kin to be relevant in contextualizing a given suicide death. We then develop a zero-shot learning framework to model these 12 themes at scale, and analyze variation in these themes by decedent characteristics and over time. Our work uncovers several online activities related to harm to self, harm to others, interpersonal interactions, activity levels online, and life events, which correspond to different phases of suicide risk from two prominent suicide theories. We find an association between these themes and decedent characteristics like age, means of death, and interpersonal problems, and many themes became more prevalent during the 2020 COVID-19 lockdowns. While digital spaces have taken some steps to address expressions of suicidality online, our work illustrates the opportunities for developing interventions related to less explicit indicators of suicide risk by combining suicide theories with computational research. 

---
# SpiroLLM: Finetuning Pretrained LLMs to Understand Spirogram Time Series with Clinical Validation in COPD Reporting 

**Authors**: Shuhao Mei, Yongchao Long, Shan Cao, Xiaobo Han, Shijia Geng, Jinbo Sun, Yuxi Zhou, Shenda Hong  

**Link**: [PDF](https://arxiv.org/pdf/2507.16145)  

**Abstract**: Chronic Obstructive Pulmonary Disease (COPD), a major chronic respiratory disease with persistent airflow limitation, is a leading global cause of disability and mortality. Respiratory spirogram time series, routinely collected during pulmonary function tests (PFTs), play a critical role in the early detection of repsiratory diseases and in monitoring lung function over time. However, most current AI models for COPD diagnosis are limited to outputting classification results without providing a rationale for their diagnostic process, while current Large Language Models (LLMs) cannot understand spirograms yet, which severely limits their clinical trust and adoption. To tackle this challenge, we leverage a cohort of 234,028 individuals from the UK Biobank (UKB) to propose SpiroLLM, the first multimodal large language model that can understand spirogram. The model extracts morphological features from respiratory curves via a SpiroEncoder and aligns them with PFT numerical values in a unified latent space using a SpiroProjector, ultimately empowering a large language model to generate a comprehensive diagnostic report. Experimental results confirm that SpiroLLM achieved a diagnostic AUROC of 0.8980 (95% CI: 0.8820-0.9132). In a robustness test with missing core data, it maintained a 100% valid response rate, far surpassing the 13.4% of a text-only model and showcasing the superiority of its multimodal design. This work demonstrates the substantial potential of deeply fusing physiological signals with large language models, establishing a new paradigm for the next generation of interpretable and reliable clinical decision support tools. 

---
# AlgoTune: Can Language Models Speed Up General-Purpose Numerical Programs? 

**Authors**: Ori Press, Brandon Amos, Haoyu Zhao, Yikai Wu, Samuel K. Ainsworth, Dominik Krupke, Patrick Kidger, Touqir Sajed, Bartolomeo Stellato, Jisun Park, Nathanael Bosch, Eli Meril, Albert Steppi, Arman Zharmagambetov, Fangzhao Zhang, David Perez-Pineiro, Alberto Mercurio, Ni Zhan, Talor Abramovich, Kilian Lieret, Hanlin Zhang, Shirley Huang, Matthias Bethge, Ofir Press  

**Link**: [PDF](https://arxiv.org/pdf/2507.15887)  

**Abstract**: Despite progress in language model (LM) capabilities, evaluations have thus far focused on models' performance on tasks that humans have previously solved, including in programming (Jimenez et al., 2024) and mathematics (Glazer et al., 2024). We therefore propose testing models' ability to design and implement algorithms in an open-ended benchmark: We task LMs with writing code that efficiently solves computationally challenging problems in computer science, physics, and mathematics. Our AlgoTune benchmark consists of 155 coding tasks collected from domain experts and a framework for validating and timing LM-synthesized solution code, which is compared to reference implementations from popular open-source packages. In addition, we develop a baseline LM agent, AlgoTuner, and evaluate its performance across a suite of frontier models. AlgoTuner achieves an average 1.72x speedup against our reference solvers, which use libraries such as SciPy, sk-learn and CVXPY. However, we find that current models fail to discover algorithmic innovations, instead preferring surface-level optimizations. We hope that AlgoTune catalyzes the development of LM agents exhibiting creative problem solving beyond state-of-the-art human performance. 

---
# Document Haystack: A Long Context Multimodal Image/Document Understanding Vision LLM Benchmark 

**Authors**: Goeric Huybrechts, Srikanth Ronanki, Sai Muralidhar Jayanthi, Jack Fitzgerald, Srinivasan Veeravanallur  

**Link**: [PDF](https://arxiv.org/pdf/2507.15882)  

**Abstract**: The proliferation of multimodal Large Language Models has significantly advanced the ability to analyze and understand complex data inputs from different modalities. However, the processing of long documents remains under-explored, largely due to a lack of suitable benchmarks. To address this, we introduce Document Haystack, a comprehensive benchmark designed to evaluate the performance of Vision Language Models (VLMs) on long, visually complex documents. Document Haystack features documents ranging from 5 to 200 pages and strategically inserts pure text or multimodal text+image "needles" at various depths within the documents to challenge VLMs' retrieval capabilities. Comprising 400 document variants and a total of 8,250 questions, it is supported by an objective, automated evaluation framework. We detail the construction and characteristics of the Document Haystack dataset, present results from prominent VLMs and discuss potential research avenues in this area. 

---
# Why Braking? Scenario Extraction and Reasoning Utilizing LLM 

**Authors**: Yin Wu, Daniel Slieter, Vivek Subramanian, Ahmed Abouelazm, Robin Bohn, J. Marius Zöllner  

**Link**: [PDF](https://arxiv.org/pdf/2507.15874)  

**Abstract**: The growing number of ADAS-equipped vehicles has led to a dramatic increase in driving data, yet most of them capture routine driving behavior. Identifying and understanding safety-critical corner cases within this vast dataset remains a significant challenge. Braking events are particularly indicative of potentially hazardous situations, motivating the central question of our research: Why does a vehicle brake? Existing approaches primarily rely on rule-based heuristics to retrieve target scenarios using predefined condition filters. While effective in simple environments such as highways, these methods lack generalization in complex urban settings. In this paper, we propose a novel framework that leverages Large Language Model (LLM) for scenario understanding and reasoning. Our method bridges the gap between low-level numerical signals and natural language descriptions, enabling LLM to interpret and classify driving scenarios. We propose a dual-path scenario retrieval that supports both category-based search for known scenarios and embedding-based retrieval for unknown Out-of-Distribution (OOD) scenarios. To facilitate evaluation, we curate scenario annotations on the Argoverse 2 Sensor Dataset. Experimental results show that our method outperforms rule-based baselines and generalizes well to OOD scenarios. 

---
# RDMA: Cost Effective Agent-Driven Rare Disease Discovery within Electronic Health Record Systems 

**Authors**: John Wu, Adam Cross, Jimeng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2507.15867)  

**Abstract**: Rare diseases affect 1 in 10 Americans, yet standard ICD coding systems fail to capture these conditions in electronic health records (EHR), leaving crucial information buried in clinical notes. Current approaches struggle with medical abbreviations, miss implicit disease mentions, raise privacy concerns with cloud processing, and lack clinical reasoning abilities. We present Rare Disease Mining Agents (RDMA), a framework that mirrors how medical experts identify rare disease patterns in EHR. RDMA connects scattered clinical observations that together suggest specific rare conditions. By handling clinical abbreviations, recognizing implicit disease patterns, and applying contextual reasoning locally on standard hardware, RDMA reduces privacy risks while improving F1 performance by upwards of 30\% and decreasing inferences costs 10-fold. This approach helps clinicians avoid the privacy risk of using cloud services while accessing key rare disease information from EHR systems, supporting earlier diagnosis for rare disease patients. Available at this https URL. 

---
