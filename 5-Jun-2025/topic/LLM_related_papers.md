# ProRank: Prompt Warmup via Reinforcement Learning for Small Language Models Reranking 

**Authors**: Xianming Li, Aamir Shakir, Rui Huang, Julius Lipp, Jing Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.03487)  

**Abstract**: Reranking is fundamental to information retrieval and retrieval-augmented generation, with recent Large Language Models (LLMs) significantly advancing reranking quality. While recent advances with LLMs have significantly improved document reranking quality, current approaches primarily rely on large-scale LLMs (>7B parameters) through zero-shot prompting, presenting high computational costs. Small Language Models (SLMs) offer a promising alternative because of their efficiency, but our preliminary quantitative analysis reveals they struggle with understanding task prompts without fine-tuning. This limits their effectiveness for document reranking tasks. To address this issue, we introduce a novel two-stage training approach, ProRank, for SLM-based document reranking. First, we propose a prompt warmup stage using reinforcement learning GRPO to steer SLMs to understand task prompts and generate more accurate coarse-grained binary relevance scores for document reranking. Then, we continuously fine-tune the SLMs with a fine-grained score learning stage without introducing additional layers to further improve the reranking quality. Comprehensive experimental results demonstrate that the proposed ProRank consistently outperforms both the most advanced open-source and proprietary reranking models. Notably, our lightweight ProRank-0.5B model even surpasses the powerful 32B LLM reranking model on the BEIR benchmark, establishing that properly trained SLMs can achieve superior document reranking performance while maintaining computational efficiency. 

---
# DistRAG: Towards Distance-Based Spatial Reasoning in LLMs 

**Authors**: Nicole R Schneider, Nandini Ramachandran, Kent O'Sullivan, Hanan Samet  

**Link**: [PDF](https://arxiv.org/pdf/2506.03424)  

**Abstract**: Many real world tasks where Large Language Models (LLMs) can be used require spatial reasoning, like Point of Interest (POI) recommendation and itinerary planning. However, on their own LLMs lack reliable spatial reasoning capabilities, especially about distances. To address this problem, we develop a novel approach, DistRAG, that enables an LLM to retrieve relevant spatial information not explicitly learned during training. Our method encodes the geodesic distances between cities and towns in a graph and retrieves a context subgraph relevant to the question. Using this technique, our method enables an LLM to answer distance-based reasoning questions that it otherwise cannot answer. Given the vast array of possible places an LLM could be asked about, DistRAG offers a flexible first step towards providing a rudimentary `world model' to complement the linguistic knowledge held in LLMs. 

---
# Establishing Trustworthy LLM Evaluation via Shortcut Neuron Analysis 

**Authors**: Kejian Zhu, Shangqing Tu, Zhuoran Jin, Lei Hou, Juanzi Li, Jun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.04142)  

**Abstract**: The development of large language models (LLMs) depends on trustworthy evaluation. However, most current evaluations rely on public benchmarks, which are prone to data contamination issues that significantly compromise fairness. Previous researches have focused on constructing dynamic benchmarks to address contamination. However, continuously building new benchmarks is costly and cyclical. In this work, we aim to tackle contamination by analyzing the mechanisms of contaminated models themselves. Through our experiments, we discover that the overestimation of contaminated models is likely due to parameters acquiring shortcut solutions in training. We further propose a novel method for identifying shortcut neurons through comparative and causal analysis. Building on this, we introduce an evaluation method called shortcut neuron patching to suppress shortcut neurons. Experiments validate the effectiveness of our approach in mitigating contamination. Additionally, our evaluation results exhibit a strong linear correlation with MixEval, a recently released trustworthy benchmark, achieving a Spearman coefficient ($\rho$) exceeding 0.95. This high correlation indicates that our method closely reveals true capabilities of the models and is trustworthy. We conduct further experiments to demonstrate the generalizability of our method across various benchmarks and hyperparameter settings. Code: this https URL 

---
# Are Lexicon-Based Tools Still the Gold Standard for Valence Analysis in Low-Resource Flemish? 

**Authors**: Ratna Kandala, Katie Hoemann  

**Link**: [PDF](https://arxiv.org/pdf/2506.04139)  

**Abstract**: Understanding the nuances in everyday language is pivotal for advancements in computational linguistics & emotions research. Traditional lexicon-based tools such as LIWC and Pattern have long served as foundational instruments in this domain. LIWC is the most extensively validated word count based text analysis tool in the social sciences and Pattern is an open source Python library offering functionalities for NLP. However, everyday language is inherently spontaneous, richly expressive, & deeply context dependent. To explore the capabilities of LLMs in capturing the valences of daily narratives in Flemish, we first conducted a study involving approximately 25,000 textual responses from 102 Dutch-speaking participants. Each participant provided narratives prompted by the question, "What is happening right now and how do you feel about it?", accompanied by self-assessed valence ratings on a continuous scale from -50 to +50. We then assessed the performance of three Dutch-specific LLMs in predicting these valence scores, and compared their outputs to those generated by LIWC and Pattern. Our findings indicate that, despite advancements in LLM architectures, these Dutch tuned models currently fall short in accurately capturing the emotional valence present in spontaneous, real-world narratives. This study underscores the imperative for developing culturally and linguistically tailored models/tools that can adeptly handle the complexities of natural language use. Enhancing automated valence analysis is not only pivotal for advancing computational methodologies but also holds significant promise for psychological research with ecologically valid insights into human daily experiences. We advocate for increased efforts in creating comprehensive datasets & finetuning LLMs for low-resource languages like Flemish, aiming to bridge the gap between computational linguistics & emotion research. 

---
# SuperWriter: Reflection-Driven Long-Form Generation with Large Language Models 

**Authors**: Yuhao Wu, Yushi Bai, Zhiqiang Hu, Juanzi Li, Roy Ka-Wei Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.04180)  

**Abstract**: Long-form text generation remains a significant challenge for large language models (LLMs), particularly in maintaining coherence, ensuring logical consistency, and preserving text quality as sequence length increases. To address these limitations, we propose SuperWriter-Agent, an agent-based framework designed to enhance the quality and consistency of long-form text generation. SuperWriter-Agent introduces explicit structured thinking-through planning and refinement stages into the generation pipeline, guiding the model to follow a more deliberate and cognitively grounded process akin to that of a professional writer. Based on this framework, we construct a supervised fine-tuning dataset to train a 7B SuperWriter-LM. We further develop a hierarchical Direct Preference Optimization (DPO) procedure that uses Monte Carlo Tree Search (MCTS) to propagate final quality assessments and optimize each generation step accordingly. Empirical results across diverse benchmarks demonstrate that SuperWriter-LM achieves state-of-the-art performance, surpassing even larger-scale baseline models in both automatic evaluation and human evaluation. Furthermore, comprehensive ablation studies demonstrate the effectiveness of hierarchical DPO and underscore the value of incorporating structured thinking steps to improve the quality of long-form text generation. 

---
# Long or short CoT? Investigating Instance-level Switch of Large Reasoning Models 

**Authors**: Ruiqi Zhang, Changyi Xiao, Yixin Cao  

**Link**: [PDF](https://arxiv.org/pdf/2506.04182)  

**Abstract**: With the rapid advancement of large reasoning models, long Chain-of-Thought (CoT) prompting has demonstrated strong performance on complex tasks. However, this often comes with a significant increase in token usage. In this paper, we conduct a comprehensive empirical analysis comparing long and short CoT strategies. Our findings reveal that while long CoT can lead to performance improvements, its benefits are often marginal relative to its significantly higher token consumption. Specifically, long CoT tends to outperform when ample generation budgets are available, whereas short CoT is more effective under tighter budget constraints. These insights underscore the need for a dynamic approach that selects the proper CoT strategy based on task context and resource availability. To address this, we propose SwitchCoT, an automatic framework that adaptively chooses between long and short CoT strategies to balance reasoning accuracy and computational efficiency. Moreover, SwitchCoT is designed to be budget-aware, making it broadly applicable across scenarios with varying resource constraints. Experimental results demonstrate that SwitchCoT can reduce inference costs by up to 50% while maintaining high accuracy. Notably, under limited token budgets, it achieves performance comparable to, or even exceeding, that of using either long or short CoT alone. 

---
# R-Search: Empowering LLM Reasoning with Search via Multi-Reward Reinforcement Learning 

**Authors**: Qingfei Zhao, Ruobing Wang, Dingling Xu, Daren Zha, Limin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.04185)  

**Abstract**: Large language models (LLMs) have notably progressed in multi-step and long-chain reasoning. However, extending their reasoning capabilities to encompass deep interactions with search remains a non-trivial challenge, as models often fail to identify optimal reasoning-search interaction trajectories, resulting in suboptimal responses. We propose R-Search, a novel reinforcement learning framework for Reasoning-Search integration, designed to enable LLMs to autonomously execute multi-step reasoning with deep search interaction, and learn optimal reasoning search interaction trajectories via multi-reward signals, improving response quality in complex logic- and knowledge-intensive tasks. R-Search guides the LLM to dynamically decide when to retrieve or reason, while globally integrating key evidence to enhance deep knowledge interaction between reasoning and search. During RL training, R-Search provides multi-stage, multi-type rewards to jointly optimize the reasoning-search trajectory. Experiments on seven datasets show that R-Search outperforms advanced RAG baselines by up to 32.2% (in-domain) and 25.1% (out-of-domain). The code and data are available at this https URL. 

---
# SkipGPT: Dynamic Layer Pruning Reinvented with Token Awareness and Module Decoupling 

**Authors**: Anhao Zhao, Fanghua Ye, Yingqi Fan, Junlong Tong, Zhiwei Fei, Hui Su, Xiaoyu Shen  

**Link**: [PDF](https://arxiv.org/pdf/2506.04179)  

**Abstract**: Large language models (LLMs) achieve remarkable performance across tasks but incur substantial computational costs due to their deep, multi-layered architectures. Layer pruning has emerged as a strategy to alleviate these inefficiencies, but conventional static pruning methods overlook two critical dynamics inherent to LLM inference: (1) horizontal dynamics, where token-level heterogeneity demands context-aware pruning decisions, and (2) vertical dynamics, where the distinct functional roles of MLP and self-attention layers necessitate component-specific pruning policies. We introduce SkipGPT, a dynamic layer pruning framework designed to optimize computational resource allocation through two core innovations: (1) global token-aware routing to prioritize critical tokens, and (2) decoupled pruning policies for MLP and self-attention components. To mitigate training instability, we propose a two-stage optimization paradigm: first, a disentangled training phase that learns routing strategies via soft parameterization to avoid premature pruning decisions, followed by parameter-efficient LoRA fine-tuning to restore performance impacted by layer removal. Extensive experiments demonstrate that SkipGPT reduces over 40% of model parameters while matching or exceeding the performance of the original dense model across benchmarks. By harmonizing dynamic efficiency with preserved expressivity, SkipGPT advances the practical deployment of scalable, resource-aware LLMs. Our code is publicly available at: this https URL. 

---
# A Dataset for Addressing Patient's Information Needs related to Clinical Course of Hospitalization 

**Authors**: Sarvesh Soni, Dina Demner-Fushman  

**Link**: [PDF](https://arxiv.org/pdf/2506.04156)  

**Abstract**: Patients have distinct information needs about their hospitalization that can be addressed using clinical evidence from electronic health records (EHRs). While artificial intelligence (AI) systems show promise in meeting these needs, robust datasets are needed to evaluate the factual accuracy and relevance of AI-generated responses. To our knowledge, no existing dataset captures patient information needs in the context of their EHRs. We introduce ArchEHR-QA, an expert-annotated dataset based on real-world patient cases from intensive care unit and emergency department settings. The cases comprise questions posed by patients to public health forums, clinician-interpreted counterparts, relevant clinical note excerpts with sentence-level relevance annotations, and clinician-authored answers. To establish benchmarks for grounded EHR question answering (QA), we evaluated three open-weight large language models (LLMs)--Llama 4, Llama 3, and Mixtral--across three prompting strategies: generating (1) answers with citations to clinical note sentences, (2) answers before citations, and (3) answers from filtered citations. We assessed performance on two dimensions: Factuality (overlap between cited note sentences and ground truth) and Relevance (textual and semantic similarity between system and reference answers). The final dataset contains 134 patient cases. The answer-first prompting approach consistently performed best, with Llama 4 achieving the highest scores. Manual error analysis supported these findings and revealed common issues such as omitted key clinical evidence and contradictory or hallucinated content. Overall, ArchEHR-QA provides a strong benchmark for developing and evaluating patient-centered EHR QA systems, underscoring the need for further progress toward generating factual and relevant responses in clinical contexts. 

---
# LLMEval-Med: A Real-world Clinical Benchmark for Medical LLMs with Physician Validation 

**Authors**: Ming Zhang, Yujiong Shen, Zelin Li, Huayu Sha, Binze Hu, Yuhui Wang, Chenhao Huang, Shichun Liu, Jingqi Tong, Changhao Jiang, Mingxu Chai, Zhiheng Xi, Shihan Dou, Tao Gui, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.04078)  

**Abstract**: Evaluating large language models (LLMs) in medicine is crucial because medical applications require high accuracy with little room for error. Current medical benchmarks have three main types: medical exam-based, comprehensive medical, and specialized assessments. However, these benchmarks have limitations in question design (mostly multiple-choice), data sources (often not derived from real clinical scenarios), and evaluation methods (poor assessment of complex reasoning). To address these issues, we present LLMEval-Med, a new benchmark covering five core medical areas, including 2,996 questions created from real-world electronic health records and expert-designed clinical scenarios. We also design an automated evaluation pipeline, incorporating expert-developed checklists into our LLM-as-Judge framework. Furthermore, our methodology validates machine scoring through human-machine agreement analysis, dynamically refining checklists and prompts based on expert feedback to ensure reliability. We evaluate 13 LLMs across three categories (specialized medical models, open-source models, and closed-source models) on LLMEval-Med, providing valuable insights for the safe and effective deployment of LLMs in medical domains. The dataset is released in this https URL. 

---
# LaF-GRPO: In-Situ Navigation Instruction Generation for the Visually Impaired via GRPO with LLM-as-Follower Reward 

**Authors**: Yi Zhao, Siqi Wang, Jing Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.04070)  

**Abstract**: Navigation instruction generation for visually impaired (VI) individuals (NIG-VI) is critical yet relatively underexplored. This study, hence, focuses on producing precise, in-situ, step-by-step navigation instructions that are practically usable by VI users. Concretely, we propose LaF-GRPO (LLM-as-Follower GRPO), where an LLM simulates VI user responses to generate rewards guiding the Vision-Language Model (VLM) post-training. This enhances instruction usability while reducing costly real-world data needs. To facilitate training and testing, we introduce NIG4VI, a 27k-sample open-sourced benchmark. It provides diverse navigation scenarios with accurate spatial coordinates, supporting detailed, open-ended in-situ instruction generation. Experiments on NIG4VI show the effectiveness of LaF-GRPO by quantitative metrics (e.g., Zero-(LaF-GRPO) boosts BLEU +14\%; SFT+(LaF-GRPO) METEOR 0.542 vs. GPT-4o's 0.323) and yields more intuitive, safer instructions. Code and benchmark are available at \href{this https URL}{this https URL}. 

---
# A Novel Data Augmentation Approach for Automatic Speaking Assessment on Opinion Expressions 

**Authors**: Chung-Chun Wang, Jhen-Ke Lin, Hao-Chien Lu, Hong-Yun Lin, Berlin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.04077)  

**Abstract**: Automated speaking assessment (ASA) on opinion expressions is often hampered by the scarcity of labeled recordings, which restricts prompt diversity and undermines scoring reliability. To address this challenge, we propose a novel training paradigm that leverages a large language models (LLM) to generate diverse responses of a given proficiency level, converts responses into synthesized speech via speaker-aware text-to-speech synthesis, and employs a dynamic importance loss to adaptively reweight training instances based on feature distribution differences between synthesized and real speech. Subsequently, a multimodal large language model integrates aligned textual features with speech signals to predict proficiency scores directly. Experiments conducted on the LTTC dataset show that our approach outperforms methods relying on real data or conventional augmentation, effectively mitigating low-resource constraints and enabling ASA on opinion expressions with cross-modal information. 

---
# Progressive Mastery: Customized Curriculum Learning with Guided Prompting for Mathematical Reasoning 

**Authors**: Muling Wu, Qi Qian, Wenhao Liu, Xiaohua Wang, Zisu Huang, Di Liang, LI Miao, Shihan Dou, Changze Lv, Zhenghua Wang, Zhibo Xu, Lina Chen, Tianlong Li, Xiaoqing Zheng, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.04065)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable performance across various reasoning tasks, yet post-training is constrained by inefficient sample utilization and inflexible difficulty samples processing. To address these limitations, we propose Customized Curriculum Learning (CCL), a novel framework with two key innovations. First, we introduce model-adaptive difficulty definition that customizes curriculum datasets based on each model's individual capabilities rather than using predefined difficulty metrics. Second, we develop "Guided Prompting," which dynamically reduces sample difficulty through strategic hints, enabling effective utilization of challenging samples that would otherwise degrade performance. Comprehensive experiments on supervised fine-tuning and reinforcement learning demonstrate that CCL significantly outperforms uniform training approaches across five mathematical reasoning benchmarks, confirming its effectiveness across both paradigms in enhancing sample utilization and model performance. 

---
# High Accuracy, Less Talk (HALT): Reliable LLMs through Capability-Aligned Finetuning 

**Authors**: Tim Franzmeyer, Archie Sravankumar, Lijuan Liu, Yuning Mao, Rui Hou, Sinong Wang, Jakob N. Foerster, Luke Zettlemoyer, Madian Khabsa  

**Link**: [PDF](https://arxiv.org/pdf/2506.04051)  

**Abstract**: Large Language Models (LLMs) currently respond to every prompt. However, they can produce incorrect answers when they lack knowledge or capability -- a problem known as hallucination. We instead propose post-training an LLM to generate content only when confident in its correctness and to otherwise (partially) abstain. Specifically, our method, HALT, produces capability-aligned post-training data that encodes what the model can and cannot reliably generate. We generate this data by splitting responses of the pretrained LLM into factual fragments (atomic statements or reasoning steps), and use ground truth information to identify incorrect fragments. We achieve capability-aligned finetuning responses by either removing incorrect fragments or replacing them with "Unsure from Here" -- according to a tunable threshold that allows practitioners to trade off response completeness and mean correctness of the response's fragments. We finetune four open-source models for biography writing, mathematics, coding, and medicine with HALT for three different trade-off thresholds. HALT effectively trades off response completeness for correctness, increasing the mean correctness of response fragments by 15% on average, while resulting in a 4% improvement in the F1 score (mean of completeness and correctness of the response) compared to the relevant baselines. By tuning HALT for highest correctness, we train a single reliable Llama3-70B model with correctness increased from 51% to 87% across all four domains while maintaining 53% of the response completeness achieved with standard finetuning. 

---
# Lacuna Inc. at SemEval-2025 Task 4: LoRA-Enhanced Influence-Based Unlearning for LLMs 

**Authors**: Aleksey Kudelya, Alexander Shirnin  

**Link**: [PDF](https://arxiv.org/pdf/2506.04044)  

**Abstract**: This paper describes LIBU (LoRA enhanced influence-based unlearning), an algorithm to solve the task of unlearning - removing specific knowledge from a large language model without retraining from scratch and compromising its overall utility (SemEval-2025 Task 4: Unlearning sensitive content from Large Language Models). The algorithm combines classical \textit{influence functions} to remove the influence of the data from the model and \textit{second-order optimization} to stabilize the overall utility. Our experiments show that this lightweight approach is well applicable for unlearning LLMs in different kinds of task. 

---
# Explainability-Based Token Replacement on LLM-Generated Text 

**Authors**: Hadi Mohammadi, Anastasia Giachanou, Daniel L. Oberski, Ayoub Bagheri  

**Link**: [PDF](https://arxiv.org/pdf/2506.04050)  

**Abstract**: Generative models, especially large language models (LLMs), have shown remarkable progress in producing text that appears human-like. However, they often exhibit patterns that make their output easier to detect than text written by humans. In this paper, we investigate how explainable AI (XAI) methods can be used to reduce the detectability of AI-generated text (AIGT) while also introducing a robust ensemble-based detection approach. We begin by training an ensemble classifier to distinguish AIGT from human-written text, then apply SHAP and LIME to identify tokens that most strongly influence its predictions. We propose four explainability-based token replacement strategies to modify these influential tokens. Our findings show that these token replacement approaches can significantly diminish a single classifier's ability to detect AIGT. However, our ensemble classifier maintains strong performance across multiple languages and domains, showing that a multi-model approach can mitigate the impact of token-level manipulations. These results show that XAI methods can make AIGT harder to detect by focusing on the most influential tokens. At the same time, they highlight the need for robust, ensemble-based detection strategies that can adapt to evolving approaches for hiding AIGT. 

---
# Think Like a Person Before Responding: A Multi-Faceted Evaluation of Persona-Guided LLMs for Countering Hate 

**Authors**: Mikel K. Ngueajio, Flor Miriam Plaza-del-Arco, Yi-Ling Chung, Danda B. Rawat, Amanda Cercas Curry  

**Link**: [PDF](https://arxiv.org/pdf/2506.04043)  

**Abstract**: Automated counter-narratives (CN) offer a promising strategy for mitigating online hate speech, yet concerns about their affective tone, accessibility, and ethical risks remain. We propose a framework for evaluating Large Language Model (LLM)-generated CNs across four dimensions: persona framing, verbosity and readability, affective tone, and ethical robustness. Using GPT-4o-Mini, Cohere's CommandR-7B, and Meta's LLaMA 3.1-70B, we assess three prompting strategies on the MT-Conan and HatEval datasets. Our findings reveal that LLM-generated CNs are often verbose and adapted for people with college-level literacy, limiting their accessibility. While emotionally guided prompts yield more empathetic and readable responses, there remain concerns surrounding safety and effectiveness. 

---
# EuroLLM-9B: Technical Report 

**Authors**: Pedro Henrique Martins, João Alves, Patrick Fernandes, Nuno M. Guerreiro, Ricardo Rei, Amin Farajian, Mateusz Klimaszewski, Duarte M. Alves, José Pombal, Manuel Faysse, Pierre Colombo, François Yvon, Barry Haddow, José G. C. de Souza, Alexandra Birch, André F. T. Martins  

**Link**: [PDF](https://arxiv.org/pdf/2506.04079)  

**Abstract**: This report presents EuroLLM-9B, a large language model trained from scratch to support the needs of European citizens by covering all 24 official European Union languages and 11 additional languages. EuroLLM addresses the issue of European languages being underrepresented and underserved in existing open large language models. We provide a comprehensive overview of EuroLLM-9B's development, including tokenizer design, architectural specifications, data filtering, and training procedures. We describe the pre-training data collection and filtering pipeline, including the creation of EuroFilter, an AI-based multilingual filter, as well as the design of EuroBlocks-Synthetic, a novel synthetic dataset for post-training that enhances language coverage for European languages. Evaluation results demonstrate EuroLLM-9B's competitive performance on multilingual benchmarks and machine translation tasks, establishing it as the leading open European-made LLM of its size. To support open research and adoption, we release all major components of this work, including the base and instruction-tuned models, the EuroFilter classifier, and the synthetic post-training dataset. 

---
# More or Less Wrong: A Benchmark for Directional Bias in LLM Comparative Reasoning 

**Authors**: Mohammadamin Shafiei, Hamidreza Saffari, Nafise Sadat Moosavi  

**Link**: [PDF](https://arxiv.org/pdf/2506.03923)  

**Abstract**: Large language models (LLMs) are known to be sensitive to input phrasing, but the mechanisms by which semantic cues shape reasoning remain poorly understood. We investigate this phenomenon in the context of comparative math problems with objective ground truth, revealing a consistent and directional framing bias: logically equivalent questions containing the words ``more'', ``less'', or ``equal'' systematically steer predictions in the direction of the framing term. To study this effect, we introduce MathComp, a controlled benchmark of 300 comparison scenarios, each evaluated under 14 prompt variants across three LLM families. We find that model errors frequently reflect linguistic steering, systematic shifts toward the comparative term present in the prompt. Chain-of-thought prompting reduces these biases, but its effectiveness varies: free-form reasoning is more robust, while structured formats may preserve or reintroduce directional drift. Finally, we show that including demographic identity terms (e.g., ``a woman'', ``a Black person'') in input scenarios amplifies directional drift, despite identical underlying quantities, highlighting the interplay between semantic framing and social referents. These findings expose critical blind spots in standard evaluation and motivate framing-aware benchmarks for diagnosing reasoning robustness and fairness in LLMs. 

---
# HSSBench: Benchmarking Humanities and Social Sciences Ability for Multimodal Large Language Models 

**Authors**: Zhaolu Kang, Junhao Gong, Jiaxu Yan, Wanke Xia, Yian Wang, Ziwen Wang, Huaxuan Ding, Zhuo Cheng, Wenhao Cao, Zhiyuan Feng, Siqi He, Shannan Yan, Junzhe Chen, Xiaomin He, Chaoya Jiang, Wei Ye, Kaidong Yu, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.03922)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated significant potential to advance a broad range of domains. However, current benchmarks for evaluating MLLMs primarily emphasize general knowledge and vertical step-by-step reasoning typical of STEM disciplines, while overlooking the distinct needs and potential of the Humanities and Social Sciences (HSS). Tasks in the HSS domain require more horizontal, interdisciplinary thinking and a deep integration of knowledge across related fields, which presents unique challenges for MLLMs, particularly in linking abstract concepts with corresponding visual representations. Addressing this gap, we present HSSBench, a dedicated benchmark designed to assess the capabilities of MLLMs on HSS tasks in multiple languages, including the six official languages of the United Nations. We also introduce a novel data generation pipeline tailored for HSS scenarios, in which multiple domain experts and automated agents collaborate to generate and iteratively refine each sample. HSSBench contains over 13,000 meticulously designed samples, covering six key categories. We benchmark more than 20 mainstream MLLMs on HSSBench and demonstrate that it poses significant challenges even for state-of-the-art models. We hope that this benchmark will inspire further research into enhancing the cross-disciplinary reasoning abilities of MLLMs, especially their capacity to internalize and connect knowledge across fields. 

---
# TextAtari: 100K Frames Game Playing with Language Agents 

**Authors**: Wenhao Li, Wenwu Li, Chuyun Shen, Junjie Sheng, Zixiao Huang, Di Wu, Yun Hua, Wei Yin, Xiangfeng Wang, Hongyuan Zha, Bo Jin  

**Link**: [PDF](https://arxiv.org/pdf/2506.04098)  

**Abstract**: We present TextAtari, a benchmark for evaluating language agents on very long-horizon decision-making tasks spanning up to 100,000 steps. By translating the visual state representations of classic Atari games into rich textual descriptions, TextAtari creates a challenging test bed that bridges sequential decision-making with natural language processing. The benchmark includes nearly 100 distinct tasks with varying complexity, action spaces, and planning horizons, all rendered as text through an unsupervised representation learning framework (AtariARI). We evaluate three open-source large language models (Qwen2.5-7B, Gemma-7B, and Llama3.1-8B) across three agent frameworks (zero-shot, few-shot chain-of-thought, and reflection reasoning) to assess how different forms of prior knowledge affect performance on these long-horizon challenges. Four scenarios-Basic, Obscured, Manual Augmentation, and Reference-based-investigate the impact of semantic understanding, instruction comprehension, and expert demonstrations on agent decision-making. Our results reveal significant performance gaps between language agents and human players in extensive planning tasks, highlighting challenges in sequential reasoning, state tracking, and strategic planning across tens of thousands of steps. TextAtari provides standardized evaluation protocols, baseline implementations, and a framework for advancing research at the intersection of language models and planning. 

---
# RadialRouter: Structured Representation for Efficient and Robust Large Language Models Routing 

**Authors**: Ruihan Jin, Pengpeng Shao, Zhengqi Wen, Jinyang Wu, Mingkuan Feng, Shuai Zhang, Jianhua Tao  

**Link**: [PDF](https://arxiv.org/pdf/2506.03880)  

**Abstract**: The rapid advancements in large language models (LLMs) have led to the emergence of routing techniques, which aim to efficiently select the optimal LLM from diverse candidates to tackle specific tasks, optimizing performance while reducing costs. Current LLM routing methods are limited in effectiveness due to insufficient exploration of the intrinsic connection between user queries and the characteristics of LLMs. To address this issue, in this paper, we present RadialRouter, a novel framework for LLM routing which employs a lightweight Transformer-based backbone with a radial structure named RadialFormer to articulate the query-LLMs relationship. The optimal LLM selection is performed based on the final states of RadialFormer. The pipeline is further refined by an objective function that combines Kullback-Leibler divergence with the query-query contrastive loss to enhance robustness. Experimental results on RouterBench show that RadialRouter significantly outperforms existing routing methods by 9.2\% and 5.8\% in the Balance and Cost First scenarios, respectively. Additionally, its adaptability toward different performance-cost trade-offs and the dynamic LLM pool demonstrates practical application potential. 

---
# Knockout LLM Assessment: Using Large Language Models for Evaluations through Iterative Pairwise Comparisons 

**Authors**: Isik Baran Sandan, Tu Anh Dinh, Jan Niehues  

**Link**: [PDF](https://arxiv.org/pdf/2506.03785)  

**Abstract**: Large Language Models (LLMs) have shown to be effective evaluators across various domains such as machine translations or the scientific domain. Current LLM-as-a-Judge approaches rely mostly on individual assessments or a single round of pairwise assessments, preventing the judge LLM from developing a global ranking perspective. To address this, we present Knockout Assessment, an LLM-asa Judge method using a knockout tournament system with iterative pairwise comparisons. Experiments across three LLMs on two datasets show that knockout assessment improves scoring accuracy, increasing Pearson correlation with expert evaluations by 0.07 on average for university-level exam scoring and machine translation evaluations, aligning LLM assessments more closely with human scoring. 

---
# ClozeMath: Improving Mathematical Reasoning in Language Models by Learning to Fill Equations 

**Authors**: Quang Hieu Pham, Thuy Duong Nguyen, Tung Pham, Anh Tuan Luu, Dat Quoc Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2506.03763)  

**Abstract**: The capabilities of large language models (LLMs) have been enhanced by training on data that reflects human thought processes, such as the Chain-of-Thought format. However, evidence suggests that the conventional scheme of next-word prediction may not fully capture how humans learn to think. Inspired by how humans generalize mathematical reasoning, we propose a new approach named ClozeMath to fine-tune LLMs for mathematical reasoning. Our ClozeMath involves a text-infilling task that predicts masked equations from a given solution, analogous to cloze exercises used in human learning. Experiments on GSM8K, MATH, and GSM-Symbolic show that ClozeMath surpasses the strong baseline Masked Thought in performance and robustness, with two test-time scaling decoding algorithms, Beam Search and Chain-of-Thought decoding. Additionally, we conduct an ablation study to analyze the effects of various architectural and implementation choices on our approach. 

---
# AhaKV: Adaptive Holistic Attention-Driven KV Cache Eviction for Efficient Inference of Large Language Models 

**Authors**: Yifeng Gu, Zicong Jiang, Jianxiu Jin, Kailing Guo, Ziyang Zhang, Xiangmin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.03762)  

**Abstract**: Large Language Models (LLMs) have significantly advanced the field of Artificial Intelligence. However, their deployment is resource-intensive, not only due to the large number of model parameters but also because the (Key-Value) KV cache consumes a lot of memory during inference. While several works propose reducing the KV cache by evicting the unnecessary tokens, these approaches rely on accumulated attention score as eviction score to quantify the importance of the token. We identify the accumulated attention score is biased and it decreases with the position of the tokens in the mathematical expectation. As a result, the retained tokens concentrate on the initial positions, limiting model's access to global contextual information. To address this issue, we propose Adaptive holistic attention KV (AhaKV), it addresses the bias of the accumulated attention score by adaptively tuning the scale of softmax according the expectation of information entropy of attention scores. To make use of the holistic attention information in self-attention mechanism, AhaKV utilize the information of value vectors, which is overlooked in previous works, to refine the adaptive score. We show theoretically that our method is well suited for bias reduction. We deployed AhaKV on different models with a fixed cache budget. Experiments show that AhaKV successfully mitigates bias and retains crucial tokens across global context and achieve state-of-the-art results against other related work on several benchmark tasks. 

---
# AdaDecode: Accelerating LLM Decoding with Adaptive Layer Parallelism 

**Authors**: Zhepei Wei, Wei-Lin Chen, Xinyu Zhu, Yu Meng  

**Link**: [PDF](https://arxiv.org/pdf/2506.03700)  

**Abstract**: Large language models (LLMs) are increasingly used for long-content generation (e.g., long Chain-of-Thought reasoning) where decoding efficiency becomes a critical bottleneck: Autoregressive decoding is inherently limited by its sequential token generation process, where each token must be generated before the next can be processed. This sequential dependency restricts the ability to fully leverage modern hardware's parallel processing capabilities. Existing methods like speculative decoding and layer skipping offer potential speedups but have notable drawbacks: speculative decoding relies on an auxiliary "drafter" model, which can be challenging to acquire and increases memory overhead, while layer skipping may introduce discrepancies in the outputs due to the missing key-value cache at skipped layers. In this work, we propose AdaDecode, which accelerates LLM decoding without requiring auxiliary models or changes to the original model parameters, while ensuring output consistency. AdaDecode leverages the insight that many tokens can accurately be generated at intermediate layers, as further layers often do not significantly alter predictions once the model reaches a certain confidence. By adaptively generating tokens at intermediate layers when confidence is high, AdaDecode enables the next token's computation to begin immediately. The remaining layer computations for early-predicted tokens are deferred and executed in parallel with subsequent tokens when needed, maximizing hardware utilization and reducing decoding latency. A final verification step ensures that early predictions match the results of standard autoregressive decoding, preserving output parity. Experiments across diverse generation tasks shows that AdaDecode consistently achieves superior decoding throughput with up to 1.73x speedup, while guaranteeing output parity with standard autoregressive decoding. 

---
# ScoreRAG: A Retrieval-Augmented Generation Framework with Consistency-Relevance Scoring and Structured Summarization for News Generation 

**Authors**: Pei-Yun Lin, Yen-lung Tsai  

**Link**: [PDF](https://arxiv.org/pdf/2506.03704)  

**Abstract**: This research introduces ScoreRAG, an approach to enhance the quality of automated news generation. Despite advancements in Natural Language Processing and large language models, current news generation methods often struggle with hallucinations, factual inconsistencies, and lack of domain-specific expertise when producing news articles. ScoreRAG addresses these challenges through a multi-stage framework combining retrieval-augmented generation, consistency relevance evaluation, and structured summarization. The system first retrieves relevant news documents from a vector database, maps them to complete news items, and assigns consistency relevance scores based on large language model evaluations. These documents are then reranked according to relevance, with low-quality items filtered out. The framework proceeds to generate graded summaries based on relevance scores, which guide the large language model in producing complete news articles following professional journalistic standards. Through this methodical approach, ScoreRAG aims to significantly improve the accuracy, coherence, informativeness, and professionalism of generated news articles while maintaining stability and consistency throughout the generation process. The code and demo are available at: this https URL. 

---
# Trustworthy Medical Question Answering: An Evaluation-Centric Survey 

**Authors**: Yinuo Wang, Robert E. Mercer, Frank Rudzicz, Sudipta Singha Roy, Pengjie Ren, Zhumin Chen, Xindi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.03659)  

**Abstract**: Trustworthiness in healthcare question-answering (QA) systems is important for ensuring patient safety, clinical effectiveness, and user confidence. As large language models (LLMs) become increasingly integrated into medical settings, the reliability of their responses directly influences clinical decision-making and patient outcomes. However, achieving comprehensive trustworthiness in medical QA poses significant challenges due to the inherent complexity of healthcare data, the critical nature of clinical scenarios, and the multifaceted dimensions of trustworthy AI. In this survey, we systematically examine six key dimensions of trustworthiness in medical QA, i.e., Factuality, Robustness, Fairness, Safety, Explainability, and Calibration. We review how each dimension is evaluated in existing LLM-based medical QA systems. We compile and compare major benchmarks designed to assess these dimensions and analyze evaluation-guided techniques that drive model improvements, such as retrieval-augmented grounding, adversarial fine-tuning, and safety alignment. Finally, we identify open challenges-such as scalable expert evaluation, integrated multi-dimensional metrics, and real-world deployment studies-and propose future research directions to advance the safe, reliable, and transparent deployment of LLM-powered medical QA. 

---
# Robust Preference Optimization via Dynamic Target Margins 

**Authors**: Jie Sun, Junkang Wu, Jiancan Wu, Zhibo Zhu, Xingyu Lu, Jun Zhou, Lintao Ma, Xiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.03690)  

**Abstract**: The alignment of Large Language Models (LLMs) is crucial for ensuring their safety and reliability in practical applications. Direct Preference Optimization (DPO) has emerged as an efficient method that directly optimizes models using preference pairs, significantly reducing resource demands. However, the effectiveness of DPO heavily depends on the data quality, which is frequently compromised by noise. In this work, we propose $\gamma$-PO, a dynamic target margin preference optimization algorithm that adjust reward margins at the pairwise level. By introducing instance-specific margin calibration, $\gamma$-PO strategically prioritizes high-confidence pairs (those demonstrating higher reward margins) while suppressing potential noise from ambiguous pairs. Moreover, $\gamma$-PO is a plug-and-play method, compatible with variants of DPO that rely on reward margin between preference pairs. Across benchmarks such as AlpacaEval2 and Arena-Hard, $\gamma$-PO achieves an average 4.4\% improvement over other baselines, setting new benchmarks for state-of-the-art performance. Additionally, $\gamma$-PO requires minimal code changes and has a negligible impact on training efficiency, making it a robust solution for enhancing LLMs alignment. Our codes are available at \href{this https URL}{this https URL}. 

---
# PulseReddit: A Novel Reddit Dataset for Benchmarking MAS in High-Frequency Cryptocurrency Trading 

**Authors**: Qiuhan Han, Qian Wang, Atsushi Yoshikawa, Masayuki Yamamura  

**Link**: [PDF](https://arxiv.org/pdf/2506.03861)  

**Abstract**: High-Frequency Trading (HFT) is pivotal in cryptocurrency markets, demanding rapid decision-making. Social media platforms like Reddit offer valuable, yet underexplored, information for such high-frequency, short-term trading. This paper introduces \textbf{PulseReddit}, a novel dataset that is the first to align large-scale Reddit discussion data with high-frequency cryptocurrency market statistics for short-term trading analysis. We conduct an extensive empirical study using Large Language Model (LLM)-based Multi-Agent Systems (MAS) to investigate the impact of social sentiment from PulseReddit on trading performance. Our experiments conclude that MAS augmented with PulseReddit data achieve superior trading outcomes compared to traditional baselines, particularly in bull markets, and demonstrate robust adaptability across different market regimes. Furthermore, our research provides conclusive insights into the performance-efficiency trade-offs of different LLMs, detailing significant considerations for practical model selection in HFT applications. PulseReddit and our findings establish a foundation for advanced MAS research in HFT, demonstrating the tangible benefits of integrating social media. 

---
# Robustness of Prompting: Enhancing Robustness of Large Language Models Against Prompting Attacks 

**Authors**: Lin Mu, Guowei Chu, Li Ni, Lei Sang, Zhize Wu, Peiquan Jin, Yiwen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.03627)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across various tasks by effectively utilizing a prompting strategy. However, they are highly sensitive to input perturbations, such as typographical errors or slight character order errors, which can substantially degrade their performance. Despite advances in prompting techniques, developing a prompting strategy that explicitly mitigates the negative impact of such perturbations remains an open challenge. To bridge this gap, we propose Robustness of Prompting (RoP), a novel prompting strategy specifically designed to enhance the robustness of LLMs. RoP consists of two stages: Error Correction and Guidance. In the Error Correction stage, RoP applies diverse perturbation methods to generate adversarial examples, which are then used to construct prompts that automatically correct input errors. In the Guidance stage, RoP generates an optimal guidance prompting based on the corrected input, steering the model toward more robust and accurate inferences. Through comprehensive experiments spanning arithmetic, commonsense, and logical reasoning tasks, we demonstrate that RoP significantly improves LLMs' robustness against adversarial perturbations. Notably, it maintains model accuracy with only minimal degradation compared to clean input scenarios, thereby establishing RoP as a practical and effective approach for enhancing LLM robustness in real-world applications. 

---
# Do Large Language Models Know Folktales? A Case Study of Yokai in Japanese Folktales 

**Authors**: Ayuto Tsutsumi, Yuu Jinnai  

**Link**: [PDF](https://arxiv.org/pdf/2506.03619)  

**Abstract**: Although Large Language Models (LLMs) have demonstrated strong language understanding and generation abilities across various languages, their cultural knowledge is often limited to English-speaking communities, which can marginalize the cultures of non-English communities. To address the problem, evaluation of the cultural awareness of the LLMs and the methods to develop culturally aware LLMs have been investigated. In this study, we focus on evaluating knowledge of folktales, a key medium for conveying and circulating culture. In particular, we focus on Japanese folktales, specifically on knowledge of Yokai. Yokai are supernatural creatures originating from Japanese folktales that continue to be popular motifs in art and entertainment today. Yokai have long served as a medium for cultural expression, making them an ideal subject for assessing the cultural awareness of LLMs. We introduce YokaiEval, a benchmark dataset consisting of 809 multiple-choice questions (each with four options) designed to probe knowledge about yokai. We evaluate the performance of 31 Japanese and multilingual LLMs on this dataset. The results show that models trained with Japanese language resources achieve higher accuracy than English-centric models, with those that underwent continued pretraining in Japanese, particularly those based on Llama-3, performing especially well. The code and dataset are available at this https URL ILab/YokaiEval. 

---
# Learning to Insert [PAUSE] Tokens for Better Reasoning 

**Authors**: Eunki Kim, Sangryul Kim, James Thorne  

**Link**: [PDF](https://arxiv.org/pdf/2506.03616)  

**Abstract**: To enhance reasoning capabilities, previous works have explored incorporating special-purpose tokens into the training process. These strategies strengthen the learning mechanism of transformer-based large language models (LLMs). Building on prior research, in which inserting dummy tokens consecutively just before reasoning steps can enhance effectiveness, we introduce a novel approach termed Dynamic Inserting Tokens Training (DIT). Our method identifies positions within sequences where model confidence is lowest according to token log-likelihood. Strategically inserting [PAUSE] tokens on these positions bolsters the model's predictive capabilities for subsequent tokens. Experimental results across diverse datasets and models, from the 2.7B model to the 8B model, demonstrate that DIT consistently outperforms traditional fine-tuning and previous token insertion methods. With this simple yet effective method, we achieve accuracy gains of up to 4.7%p on GSM8K, 3.23%p on AQUA-RAT, and pass@1 improvements of up to 3.4%p on MBPP datasets. Our work shows a model-based, dynamic approach rather than a heuristic one, thereby broadening the scope of research in reasoning. 

---
# From Understanding to Generation: An Efficient Shortcut for Evaluating Language Models 

**Authors**: Viktor Hangya, Fabian Küch, Darina Gold  

**Link**: [PDF](https://arxiv.org/pdf/2506.03592)  

**Abstract**: Iterative evaluation of LLMs during training is essential to ensure expected capability development, but can be time- and compute-intensive. While NLU tasks, where the model selects from fixed answer choices, are cheap to evaluate, essential capabilities like reasoning and code generation rely on the more time-consuming NLG (token-by-token generation) format. In this work, our aim is to decrease the computational burden of NLG benchmarks in order to enable monitoring crucial LLM capabilities during model training. We reformulate generative tasks into computationally cheaper NLU alternatives. We test the performance correlation between the original and reformulated tasks using 8 LMs of various sizes and 4 capabilities: mathematical reasoning, code generation, factual knowledge and reading comprehension. Our results show a strong correlation between task formats, supporting capability assessment via cheaper alternatives and achieving over 35x average reduction in evaluation time. We plan to publish our benchmark adaptions. 

---
# Exchange of Perspective Prompting Enhances Reasoning in Large Language Models 

**Authors**: Lin Sun, Can Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.03573)  

**Abstract**: Large language models (LLMs) have made significant advancements in addressing diverse natural language processing (NLP) tasks. However, their performance is often limited by inherent comprehension of problems. To address this limitation, we propose Exchange-of-Perspective (EoP), a novel framework designed to exchange perspectives across different definitions of problem, so that it can break the fixed mindset from any particular formulation of the question. We conducted extensive and comprehensive experiments on 8 benchmarks. The results show that EoP can significantly improve performance. For instance, compared to the non-commutative baseline PHP, with GPT-3.5-Turbo and EoP, we observe a 3.6% improvement on AQuA (60.6% to 64.2%), while GPT-4-powered EoP demonstrates a 7.7% overall accuracy enhancement on Math (53.9% to 61.6%) and a 3.5% improvement on OlympiadBench Maths (43.5% to 47.0%) when using Qwen-2.5-72b. 

---
# BPO: Revisiting Preference Modeling in Direct Preference Optimization 

**Authors**: Lin Sun, Chuang Liu, Peng Liu, Bingyang Li, Weijia Lu, Ning Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.03557)  

**Abstract**: Direct Preference Optimization (DPO) have emerged as a popular method for aligning Large Language Models (LLMs) with human preferences. While DPO effectively preserves the relative ordering between chosen and rejected responses through pairwise ranking losses, it often neglects absolute reward magnitudes. This oversight can decrease the likelihood of chosen responses and increase the risk of generating out-of-distribution responses, leading to poor performance. We term this issue Degraded Chosen Responses (DCR).To address this issue, we propose Balanced Preference Optimization (BPO), a novel framework that dynamically balances the optimization of chosen and rejected responses through two key components: balanced reward margin and gap adaptor. Unlike previous methods, BPO can fundamentally resolve DPO's DCR issue, without introducing additional constraints to the loss function. Experimental results on multiple mathematical reasoning tasks show that BPO significantly outperforms DPO, improving accuracy by +10.1% with Llama-3.1-8B-Instruct (18.8% to 28.9%) and +11.7% with Qwen2.5-Math-7B (35.0% to 46.7%). It also surpasses DPO variants by +3.6% over IPO (43.1%), +5.0% over SLiC (41.7%), and +3.1% over Cal-DPO (43.6%) on the same model. Remarkably, our algorithm requires only a single line of code modification, making it simple to implement and fully compatible with existing DPO-based frameworks. 

---
# Auto prompt sql: a resource-efficient architecture for text-to-sql translation in constrained environments 

**Authors**: Zetong Tang, Qian Ma, Di Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.03598)  

**Abstract**: Using the best Text-to-SQL methods in resource-constrained environments is challenging due to their reliance on resource-intensive open-source models. This paper introduces Auto Prompt SQL(AP-SQL), a novel architecture designed to bridge the gap between resource-efficient small open-source models and the powerful capabilities of large closed-source models for Text-to-SQL translation. Our method decomposes the task into schema filtering, retrieval-augmented text-to-SQL generation based on in-context examples, and prompt-driven schema linking and SQL generation. To improve schema selection accuracy, we fine-tune large language models. Crucially, we also explore the impact of prompt engineering throughout the process, leveraging Chain-of-Thought(CoT) and Graph-of-Thought(GoT) templates to significantly enhance the model's reasoning for accurate SQL generation. Comprehensive evaluations on the Spider benchmarks demonstrate the effectiveness of AP-SQL. 

---
# ConsistentChat: Building Skeleton-Guided Consistent Dialogues for Large Language Models from Scratch 

**Authors**: Jiawei Chen, Xinyan Guan, Qianhao Yuan, Guozhao Mo, Weixiang Zhou, Yaojie Lu, Hongyu Lin, Ben He, Le Sun, Xianpei Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.03558)  

**Abstract**: Current instruction data synthesis methods primarily focus on single-turn instructions and often neglect cross-turn coherence, resulting in context drift and reduced task completion rates in extended conversations. To address this limitation, we propose Skeleton-Guided Multi-Turn Dialogue Generation, a framework that constrains multi-turn instruction synthesis by explicitly modeling human conversational intent. It operates in two stages: (1) Intent Modeling, which captures the global structure of human dialogues by assigning each conversation to one of nine well-defined intent trajectories, ensuring a coherent and goal-oriented information flow; and (2) Skeleton Generation, which constructs a structurally grounded sequence of user queries aligned with the modeled intent, thereby serving as a scaffold that constrains and guides the downstream instruction synthesis process. Based on this process, we construct ConsistentChat, a multi-turn instruction dataset with approximately 15,000 multi-turn conversations and 224,392 utterances. Experiments on the Light, Topdial, and MT-Eval benchmarks show that models fine-tuned on ConsistentChat achieve a 20-30% improvement in chat consistency and up to a 15% increase in task success rate, significantly outperforming models trained on existing single-turn and multi-turn instruction datasets. 

---
# POSS: Position Specialist Generates Better Draft for Speculative Decoding 

**Authors**: Langlin Huang, Chengsong Huang, Jixuan Leng, Di Huang, Jiaxin Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.03566)  

**Abstract**: Speculative decoding accelerates Large Language Model (LLM) inference by using a small draft model to predict multiple tokens, and a large target model to verify these tokens in parallel. Recent studies leverage the hidden state of the target model to enhance draft model prediction accuracy. However, existing methods suffer from the degrading quality of draft token predictions at later positions, due to error accumulation in draft model generated features. In this paper, we propose Position Specialists (PosS), which consist of multiple position-specialized draft layers to generate tokens at assigned position(s). Position specialists greatly improve token acceptance rate at later positions per drafting round, as each specialist only needs to focus on handling a certain level of draft model feature deviation. Experiment results on Llama-3-8B-Instruct and Llama-2-13B-chat across six datasets demonstrate that PosS effectively improves over baselines on average acceptance length and speed-up ratio. Our codebase is available at this https URL. 

---
# Beyond Memorization: A Rigorous Evaluation Framework for Medical Knowledge Editing 

**Authors**: Shigeng Chen, Linhao Luo, Zhangchi Qiu, Yanan Cao, Carl Yang, Shirui Pan  

**Link**: [PDF](https://arxiv.org/pdf/2506.03490)  

**Abstract**: Recently, knowledge editing (KE) has emerged as a promising approach to update specific facts in Large Language Models (LLMs) without the need for full retraining. Despite the effectiveness in general-domain benchmarks, their applicability to complex medical domain remains largely unexplored. Medical knowledge editing is particularly challenging, as it requires LLMs to internalize the knowledge and generalize to unseen scenarios for effective and interpretable decision-making. In this work, we propose a novel framework called MedEditBench to rigorously evaluate the effectiveness of existing KE methods in the medical domain. In MedEditBench, we introduce a new medical knowledge editing benchmark as well as three different knowledge editing paradigms, which are designed to assess the impact of different knowledge sources for editing. Our findings indicate that current KE methods result in only superficial memorization of the injected information, failing to generalize to new scenarios. To overcome this limitation, we present Self-Generated Rationale Editing (SGR-Edit), which utilizes model-derived rationales as the target knowledge for editing, thereby uncovering the underlying reasoning process and demonstrating significant improvements over existing KE approaches. Additionally, we offer deeper insights into medical knowledge editing, including the localization of medical knowledge in LLMs and the impact of sequential editing on evolving knowledge. This could provide practical guidance for implementing KE methods in real-world medical applications. 

---
# APT: Improving Specialist LLM Performance with Weakness Case Acquisition and Iterative Preference Training 

**Authors**: Jun Rao, Zepeng Lin, Xuebo Liu, Xiaopeng Ke, Lian Lian, Dong Jin, Shengjun Cheng, Jun Yu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.03483)  

**Abstract**: Large Language Models (LLMs) often require domain-specific fine-tuning to address targeted tasks, which risks degrading their general capabilities. Maintaining a balance between domain-specific enhancements and general model utility is a key challenge. This paper proposes a novel approach named APT (Weakness Case Acquisition and Iterative Preference Training) to enhance domain-specific performance with self-generated dis-preferred weakness data (bad cases and similar cases). APT uniquely focuses on training the model using only those samples where errors occur, alongside a small, similar set of samples retrieved for this purpose. This targeted training minimizes interference with the model's existing knowledge base, effectively retaining generic capabilities. Experimental results on the LLama-2 and Mistral-V0.3 models across various benchmarks demonstrate that APT ensures no reduction in generic capacity and achieves superior performance on downstream tasks compared to various existing methods. This validates our method as an effective strategy for enhancing domain-specific capabilities without sacrificing the model's broader applicability. 

---
# EpiCoDe: Boosting Model Performance Beyond Training with Extrapolation and Contrastive Decoding 

**Authors**: Mingxu Tao, Jie Hu, Mingchuan Yang, Yunhuai Liu, Dongyan Zhao, Yansong Feng  

**Link**: [PDF](https://arxiv.org/pdf/2506.03489)  

**Abstract**: The remarkable performance of Large language models (LLMs) relies heavily on the availability of abundant high-quality training data. However, the high cost of acquiring annotated data often prevents models from obtaining capabilities to tackle downstream tasks. In this paper, we introduce a novel method, EpiCoDe that boosts model performance in data-scarcity scenarios without extra training. We first employ model extrapolation to enhance a finetuned model with its inferior version, and then adopt contrastive decoding to further reduce predicted errors, by comparing the logit scores given by the extrapolated and the vanilla finetuned model. Experiments across three tasks over four different LLMs show that EpiCoDe consistently outperforms existing methods with significant and robust improvement. We also propose a new theoretical framework to reveal the mechanism behind contrastive decoding in data-scarcity scenarios, which further helps us better understand the effectiveness of EpiCoDe. 

---
# Trajectory Prediction Meets Large Language Models: A Survey 

**Authors**: Yi Xu, Ruining Yang, Yitian Zhang, Yizhou Wang, Jianglin Lu, Mingyuan Zhang, Lili Su, Yun Fu  

**Link**: [PDF](https://arxiv.org/pdf/2506.03408)  

**Abstract**: Recent advances in large language models (LLMs) have sparked growing interest in integrating language-driven techniques into trajectory prediction. By leveraging their semantic and reasoning capabilities, LLMs are reshaping how autonomous systems perceive, model, and predict trajectories. This survey provides a comprehensive overview of this emerging field, categorizing recent work into five directions: (1) Trajectory prediction via language modeling paradigms, (2) Direct trajectory prediction with pretrained language models, (3) Language-guided scene understanding for trajectory prediction, (4) Language-driven data generation for trajectory prediction, (5) Language-based reasoning and interpretability for trajectory prediction. For each, we analyze representative methods, highlight core design choices, and identify open challenges. This survey bridges natural language processing and trajectory prediction, offering a unified perspective on how language can enrich trajectory prediction. 

---
# TokAlign: Efficient Vocabulary Adaptation via Token Alignment 

**Authors**: Chong Li, Jiajun Zhang, Chengqing Zong  

**Link**: [PDF](https://arxiv.org/pdf/2506.03523)  

**Abstract**: Tokenization serves as a foundational step for Large Language Models (LLMs) to process text. In new domains or languages, the inefficiency of the tokenizer will slow down the training and generation of LLM. The mismatch in vocabulary also hinders deep knowledge transfer between LLMs like token-level distillation. To mitigate this gap, we propose an efficient method named TokAlign to replace the vocabulary of LLM from the token co-occurrences view, and further transfer the token-level knowledge between models. It first aligns the source vocabulary to the target one by learning a one-to-one mapping matrix for token IDs. Model parameters, including embeddings, are rearranged and progressively fine-tuned for the new vocabulary. Our method significantly improves multilingual text compression rates and vocabulary initialization for LLMs, decreasing the perplexity from 3.4$\text{e}^2$ of strong baseline methods to 1.2$\text{e}^2$ after initialization. Experimental results on models across multiple parameter scales demonstrate the effectiveness and generalization of TokAlign, which costs as few as 5k steps to restore the performance of the vanilla model. After unifying vocabularies between LLMs, token-level distillation can remarkably boost (+4.4% than sentence-level distillation) the base model, costing only 235M tokens. 

---
# Debate, Reflect, and Distill: Multi-Agent Feedback with Tree-Structured Preference Optimization for Efficient Language Model Enhancement 

**Authors**: Xiaofeng Zhou, Heyan Huang, Lizi Liao  

**Link**: [PDF](https://arxiv.org/pdf/2506.03541)  

**Abstract**: Large Language Models (LLMs) continue to set new standards in knowledge-intensive and complex reasoning tasks, yet their high computational demands limit widespread adoption. While distilling large models into smaller ones offers a sustainable solution, current techniques--such as static knowledge distillation, resource-intensive reinforcement learning from human feedback, or limited self-reflection--struggle to yield substantial and lasting performance gains. In this paper, we present a novel Debate and Reflect (D&R) framework that orchestrates multi-turn debates between smaller models and stronger teacher models, eliciting actionable feedback (e.g., error analysis, corrective strategies) to guide student models. Further, we introduce Tree-structured Direct Preference Optimization (T-DPO) to efficiently leverage these debate logs, organizing interactions into a hierarchical format for effective training. Empirical evaluations across diverse NLP benchmarks demonstrate that our approach significantly improves smaller-model accuracy, robustness, and generalization, outperforming conventional baselines by a large margin. 

---
# A Multimodal, Multilingual, and Multidimensional Pipeline for Fine-grained Crowdsourcing Earthquake Damage Evaluation 

**Authors**: Zihui Ma, Lingyao Li, Juan Li, Wenyue Hua, Jingxiao Liu, Qingyuan Feng, Yuki Miura  

**Link**: [PDF](https://arxiv.org/pdf/2506.03360)  

**Abstract**: Rapid, fine-grained disaster damage assessment is essential for effective emergency response, yet remains challenging due to limited ground sensors and delays in official reporting. Social media provides a rich, real-time source of human-centric observations, but its multimodal and unstructured nature presents challenges for traditional analytical methods. In this study, we propose a structured Multimodal, Multilingual, and Multidimensional (3M) pipeline that leverages multimodal large language models (MLLMs) to assess disaster impacts. We evaluate three foundation models across two major earthquake events using both macro- and micro-level analyses. Results show that MLLMs effectively integrate image-text signals and demonstrate a strong correlation with ground-truth seismic data. However, performance varies with language, epicentral distance, and input modality. This work highlights the potential of MLLMs for disaster assessment and provides a foundation for future research in applying MLLMs to real-time crisis contexts. The code and data are released at: this https URL 

---
# Ask a Local: Detecting Hallucinations With Specialized Model Divergence 

**Authors**: Aldan Creo, Héctor Cerezo-Costas, Pedro Alonso-Doval, Maximiliano Hormazábal-Lagos  

**Link**: [PDF](https://arxiv.org/pdf/2506.03357)  

**Abstract**: Hallucinations in large language models (LLMs) - instances where models generate plausible but factually incorrect information - present a significant challenge for AI.
We introduce "Ask a Local", a novel hallucination detection method exploiting the intuition that specialized models exhibit greater surprise when encountering domain-specific inaccuracies. Our approach computes divergence between perplexity distributions of language-specialized models to identify potentially hallucinated spans. Our method is particularly well-suited for a multilingual context, as it naturally scales to multiple languages without the need for adaptation, relying on external data sources, or performing training. Moreover, we select computationally efficient models, providing a scalable solution that can be applied to a wide range of languages and domains.
Our results on a human-annotated question-answer dataset spanning 14 languages demonstrate consistent performance across languages, with Intersection-over-Union (IoU) scores around 0.3 and comparable Spearman correlation values. Our model shows particularly strong performance on Italian and Catalan, with IoU scores of 0.42 and 0.38, respectively, while maintaining cross-lingual effectiveness without language-specific adaptations. We release our code and architecture to facilitate further research in multilingual hallucination detection. 

---
# Evaluating Large Language Models for Zero-Shot Disease Labeling in CT Radiology Reports Across Organ Systems 

**Authors**: Michael E. Garcia-Alcoser, Mobina GhojoghNejad, Fakrul Islam Tushar, David Kim, Kyle J. Lafata, Geoffrey D. Rubin, Joseph Y. Lo  

**Link**: [PDF](https://arxiv.org/pdf/2506.03259)  

**Abstract**: Purpose: This study aims to evaluate the effectiveness of large language models (LLMs) in automating disease annotation of CT radiology reports. We compare a rule-based algorithm (RBA), RadBERT, and three lightweight open-weight LLMs for multi-disease labeling of chest, abdomen, and pelvis (CAP) CT reports.
Materials and Methods: This retrospective study analyzed 40,833 CT reports from 29,540 patients, with 1,789 CAP reports manually annotated across three organ systems. External validation was conducted using the CT-RATE dataset. Three open-weight LLMs were tested with zero-shot prompting. Performance was evaluated using Cohen's Kappa and micro/macro-averaged F1 scores.
Results: In 12,197 Duke CAP reports from 8,854 patients, Llama-3.1 8B and Gemma-3 27B showed the highest agreement ($\kappa$ median: 0.87). On the manually annotated set, Gemma-3 27B achieved the top macro-F1 (0.82), followed by Llama-3.1 8B (0.79), while the RBA scored lowest (0.64). On the CT-RATE dataset (lungs/pleura only), Llama-3.1 8B performed best (0.91), with Gemma-3 27B close behind (0.89). Performance differences were mainly due to differing labeling practices, especially for lung atelectasis.
Conclusion: Lightweight LLMs outperform rule-based methods for CT report annotation and generalize across organ systems with zero-shot prompting. However, binary labels alone cannot capture the full nuance of report language. LLMs can provide a flexible, efficient solution aligned with clinical judgment and user needs. 

---
# FailureSensorIQ: A Multi-Choice QA Dataset for Understanding Sensor Relationships and Failure Modes 

**Authors**: Christodoulos Constantinides, Dhaval Patel, Shuxin Lin, Claudio Guerrero, Sunil Dagajirao Patil, Jayant Kalagnanam  

**Link**: [PDF](https://arxiv.org/pdf/2506.03278)  

**Abstract**: We introduce FailureSensorIQ, a novel Multi-Choice Question-Answering (MCQA) benchmarking system designed to assess the ability of Large Language Models (LLMs) to reason and understand complex, domain-specific scenarios in Industry 4.0. Unlike traditional QA benchmarks, our system focuses on multiple aspects of reasoning through failure modes, sensor data, and the relationships between them across various industrial assets. Through this work, we envision a paradigm shift where modeling decisions are not only data-driven using statistical tools like correlation analysis and significance tests, but also domain-driven by specialized LLMs which can reason about the key contributors and useful patterns that can be captured with feature engineering. We evaluate the Industrial knowledge of over a dozen LLMs-including GPT-4, Llama, and Mistral-on FailureSensorIQ from different lens using Perturbation-Uncertainty-Complexity analysis, Expert Evaluation study, Asset-Specific Knowledge Gap analysis, ReAct agent using external knowledge-bases. Even though closed-source models with strong reasoning capabilities approach expert-level performance, the comprehensive benchmark reveals a significant drop in performance that is fragile to perturbations, distractions, and inherent knowledge gaps in the models. We also provide a real-world case study of how LLMs can drive the modeling decisions on 3 different failure prediction datasets related to various assets. We release: (a) expert-curated MCQA for various industrial assets, (b) FailureSensorIQ benchmark and Hugging Face leaderboard based on MCQA built from non-textual data found in ISO documents, and (c) LLMFeatureSelector, an LLM-based feature selection scikit-learn pipeline. The software is available at this https URL. 

---
# Unleashing the Reasoning Potential of Pre-trained LLMs by Critique Fine-Tuning on One Problem 

**Authors**: Yubo Wang, Ping Nie, Kai Zou, Lijun Wu, Wenhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.03295)  

**Abstract**: We have witnessed that strong LLMs like Qwen-Math, MiMo, and Phi-4 possess immense reasoning potential inherited from the pre-training stage. With reinforcement learning (RL), these models can improve dramatically on reasoning tasks. Recent studies have shown that even RL on a single problem can unleash these models' reasoning capabilities. However, RL is not only expensive but also unstable. Even one-shot RL requires hundreds of GPU hours. This raises a critical question: Is there a more efficient way to unleash the reasoning potential of these powerful base LLMs? In this work, we demonstrate that Critique Fine-Tuning (CFT) on only one problem can effectively unleash the reasoning potential of LLMs. Our method constructs critique data by collecting diverse model-generated solutions to a single problem and using teacher LLMs to provide detailed critiques. We fine-tune Qwen and Llama family models, ranging from 1.5B to 14B parameters, on the CFT data and observe significant performance gains across diverse reasoning tasks. For example, with just 5 GPU hours of training, Qwen-Math-7B-CFT show an average improvement of 15% on six math benchmarks and 16% on three logic reasoning benchmarks. These results are comparable to or even surpass the results from RL with 20x less compute. Ablation studies reveal the robustness of one-shot CFT across different prompt problems. These results highlight one-shot CFT as a simple, general, and compute-efficient approach to unleashing the reasoning capabilities of modern LLMs. 

---
# AmbiK: Dataset of Ambiguous Tasks in Kitchen Environment 

**Authors**: Anastasiia Ivanova, Eva Bakaeva, Zoya Volovikova, Alexey K. Kovalev, Aleksandr I. Panov  

**Link**: [PDF](https://arxiv.org/pdf/2506.04089)  

**Abstract**: As a part of an embodied agent, Large Language Models (LLMs) are typically used for behavior planning given natural language instructions from the user. However, dealing with ambiguous instructions in real-world environments remains a challenge for LLMs. Various methods for task ambiguity detection have been proposed. However, it is difficult to compare them because they are tested on different datasets and there is no universal benchmark. For this reason, we propose AmbiK (Ambiguous Tasks in Kitchen Environment), the fully textual dataset of ambiguous instructions addressed to a robot in a kitchen environment. AmbiK was collected with the assistance of LLMs and is human-validated. It comprises 1000 pairs of ambiguous tasks and their unambiguous counterparts, categorized by ambiguity type (Human Preferences, Common Sense Knowledge, Safety), with environment descriptions, clarifying questions and answers, user intents, and task plans, for a total of 2000 tasks. We hope that AmbiK will enable researchers to perform a unified comparison of ambiguity detection methods. AmbiK is available at this https URL. 

---
# AgentMisalignment: Measuring the Propensity for Misaligned Behaviour in LLM-Based Agents 

**Authors**: Akshat Naik, Patrick Quinn, Guillermo Bosch, Emma Gouné, Francisco Javier Campos Zabala, Jason Ross Brown, Edward James Young  

**Link**: [PDF](https://arxiv.org/pdf/2506.04018)  

**Abstract**: As Large Language Model (LLM) agents become more widespread, associated misalignment risks increase. Prior work has examined agents' ability to enact misaligned behaviour (misalignment capability) and their compliance with harmful instructions (misuse propensity). However, the likelihood of agents attempting misaligned behaviours in real-world settings (misalignment propensity) remains poorly understood. We introduce a misalignment propensity benchmark, AgentMisalignment, consisting of a suite of realistic scenarios in which LLM agents have the opportunity to display misaligned behaviour. We organise our evaluations into subcategories of misaligned behaviours, including goal-guarding, resisting shutdown, sandbagging, and power-seeking. We report the performance of frontier models on our benchmark, observing higher misalignment on average when evaluating more capable models. Finally, we systematically vary agent personalities through different system prompts. We find that persona characteristics can dramatically and unpredictably influence misalignment tendencies -- occasionally far more than the choice of model itself -- highlighting the importance of careful system prompt engineering for deployed AI agents. Our work highlights the failure of current alignment methods to generalise to LLM agents, and underscores the need for further propensity evaluations as autonomous systems become more prevalent. 

---
# Time Course MechInterp: Analyzing the Evolution of Components and Knowledge in Large Language Models 

**Authors**: Ahmad Dawar Hakimi, Ali Modarressi, Philipp Wicke, Hinrich Schütze  

**Link**: [PDF](https://arxiv.org/pdf/2506.03434)  

**Abstract**: Understanding how large language models (LLMs) acquire and store factual knowledge is crucial for enhancing their interpretability and reliability. In this work, we analyze the evolution of factual knowledge representation in the OLMo-7B model by tracking the roles of its attention heads and feed forward networks (FFNs) over the course of pre-training. We classify these components into four roles: general, entity, relation-answer, and fact-answer specific, and examine their stability and transitions. Our results show that LLMs initially depend on broad, general-purpose components, which later specialize as training progresses. Once the model reliably predicts answers, some components are repurposed, suggesting an adaptive learning process. Notably, attention heads display the highest turnover. We also present evidence that FFNs remain more stable throughout training. Furthermore, our probing experiments reveal that location-based relations converge to high accuracy earlier in training than name-based relations, highlighting how task complexity shapes acquisition dynamics. These insights offer a mechanistic view of knowledge formation in LLMs. 

---
# Multimodal Tabular Reasoning with Privileged Structured Information 

**Authors**: Jun-Peng Jiang, Yu Xia, Hai-Long Sun, Shiyin Lu, Qing-Guo Chen, Weihua Luo, Kaifu Zhang, De-Chuan Zhan, Han-Jia Ye  

**Link**: [PDF](https://arxiv.org/pdf/2506.04088)  

**Abstract**: Tabular reasoning involves multi-step information extraction and logical inference over tabular data. While recent advances have leveraged large language models (LLMs) for reasoning over structured tables, such high-quality textual representations are often unavailable in real-world settings, where tables typically appear as images. In this paper, we tackle the task of tabular reasoning from table images, leveraging privileged structured information available during training to enhance multimodal large language models (MLLMs). The key challenges lie in the complexity of accurately aligning structured information with visual representations, and in effectively transferring structured reasoning skills to MLLMs despite the input modality gap. To address these, we introduce TabUlar Reasoning with Bridged infOrmation ({\sc Turbo}), a new framework for multimodal tabular reasoning with privileged structured tables. {\sc Turbo} benefits from a structure-aware reasoning trace generator based on DeepSeek-R1, contributing to high-quality modality-bridged data. On this basis, {\sc Turbo} repeatedly generates and selects the advantageous reasoning paths, further enhancing the model's tabular reasoning ability. Experimental results demonstrate that, with limited ($9$k) data, {\sc Turbo} achieves state-of-the-art performance ($+7.2\%$ vs. previous SOTA) across multiple datasets. 

---
# VisCoder: Fine-Tuning LLMs for Executable Python Visualization Code Generation 

**Authors**: Yuansheng Ni, Ping Nie, Kai Zou, Xiang Yue, Wenhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.03930)  

**Abstract**: Large language models (LLMs) often struggle with visualization tasks like plotting diagrams, charts, where success depends on both code correctness and visual semantics. Existing instruction-tuning datasets lack execution-grounded supervision and offer limited support for iterative code correction, resulting in fragile and unreliable plot generation. We present VisCode-200K, a large-scale instruction tuning dataset for Python-based visualization and self-correction. It contains over 200K examples from two sources: (1) validated plotting code from open-source repositories, paired with natural language instructions and rendered plots; and (2) 45K multi-turn correction dialogues from Code-Feedback, enabling models to revise faulty code using runtime feedback. We fine-tune Qwen2.5-Coder-Instruct on VisCode-200K to create VisCoder, and evaluate it on PandasPlotBench. VisCoder significantly outperforms strong open-source baselines and approaches the performance of proprietary models like GPT-4o-mini. We further adopt a self-debug evaluation protocol to assess iterative repair, demonstrating the benefits of feedback-driven learning for executable, visually accurate code generation. 

---
# Prompt Candidates, then Distill: A Teacher-Student Framework for LLM-driven Data Annotation 

**Authors**: Mingxuan Xia, Haobo Wang, Yixuan Li, Zewei Yu, Jindong Wang, Junbo Zhao, Runze Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.03857)  

**Abstract**: Recently, Large Language Models (LLMs) have demonstrated significant potential for data annotation, markedly reducing the labor costs associated with downstream applications. However, existing methods mostly adopt an aggressive strategy by prompting LLM to determine a single gold label for each unlabeled sample. Due to the inherent uncertainty within LLMs, they often produce incorrect labels for difficult samples, severely compromising the data quality for downstream applications. Motivated by ambiguity aversion in human behaviors, we propose a novel candidate annotation paradigm wherein large language models are encouraged to output all possible labels when incurring uncertainty. To ensure unique labels are provided for downstream tasks, we develop a teacher-student framework CanDist that distills candidate annotations with a Small Language Model (SLM). We further provide a rigorous justification demonstrating that distilling candidate annotations from the teacher LLM offers superior theoretical guarantees compared to directly using single annotations. Extensive experiments across six text classification tasks validate the effectiveness of our proposed method. The source code is available at this https URL. 

---
# Mitigating Hallucinations in Large Vision-Language Models via Entity-Centric Multimodal Preference Optimization 

**Authors**: Jiulong Wu, Zhengliang Shi, Shuaiqiang Wang, Jizhou Huang, Dawei Yin, Lingyong Yan, Min Cao, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.04039)  

**Abstract**: Large Visual Language Models (LVLMs) have demonstrated impressive capabilities across multiple tasks. However, their trustworthiness is often challenged by hallucinations, which can be attributed to the modality misalignment and the inherent hallucinations of their underlying Large Language Models (LLMs) backbone. Existing preference alignment methods focus on aligning model responses with human preferences while neglecting image-text modality alignment, resulting in over-reliance on LLMs and hallucinations. In this paper, we propose Entity-centric Multimodal Preference Optimization (EMPO), which achieves enhanced modality alignment than existing human preference alignment methods. Besides, to overcome the scarcity of high-quality multimodal preference data, we utilize open-source instruction datasets to automatically construct high-quality preference data across three aspects: image, instruction, and response. Experiments on two human preference datasets and five multimodal hallucination benchmarks demonstrate the effectiveness of EMPO, e.g., reducing hallucination rates by 85.9% on Object-HalBench and 49.8% on MM-HalBench. 

---
# MMR-V: What's Left Unsaid? A Benchmark for Multimodal Deep Reasoning in Videos 

**Authors**: Kejian Zhu, Zhuoran Jin, Hongbang Yuan, Jiachun Li, Shangqing Tu, Pengfei Cao, Yubo Chen, Kang Liu, Jun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.04141)  

**Abstract**: The sequential structure of videos poses a challenge to the ability of multimodal large language models (MLLMs) to locate multi-frame evidence and conduct multimodal reasoning. However, existing video benchmarks mainly focus on understanding tasks, which only require models to match frames mentioned in the question (hereafter referred to as "question frame") and perceive a few adjacent frames. To address this gap, we propose MMR-V: A Benchmark for Multimodal Deep Reasoning in Videos. The benchmark is characterized by the following features. (1) Long-range, multi-frame reasoning: Models are required to infer and analyze evidence frames that may be far from the question frame. (2) Beyond perception: Questions cannot be answered through direct perception alone but require reasoning over hidden information. (3) Reliability: All tasks are manually annotated, referencing extensive real-world user understanding to align with common perceptions. (4) Confusability: Carefully designed distractor annotation strategies to reduce model shortcuts. MMR-V consists of 317 videos and 1,257 tasks. Our experiments reveal that current models still struggle with multi-modal reasoning; even the best-performing model, o4-mini, achieves only 52.5% accuracy. Additionally, current reasoning enhancement strategies (Chain-of-Thought and scaling test-time compute) bring limited gains. Further analysis indicates that the CoT demanded for multi-modal reasoning differs from it in textual reasoning, which partly explains the limited performance gains. We hope that MMR-V can inspire further research into enhancing multi-modal reasoning capabilities. 

---
# CETBench: A Novel Dataset constructed via Transformations over Programs for Benchmarking LLMs for Code-Equivalence Checking 

**Authors**: Neeva Oza, Ishaan Govil, Parul Gupta, Dinesh Khandelwal, Dinesh Garg, Parag Singla  

**Link**: [PDF](https://arxiv.org/pdf/2506.04019)  

**Abstract**: LLMs have been extensively used for the task of automated code generation. In this work, we examine the applicability of LLMs for the related but relatively unexplored task of code-equivalence checking, i.e., given two programs, whether they are functionally equivalent or not. This is an important problem since benchmarking code equivalence can play a critical role in evaluating LLM capabilities for tasks such as code re-writing and code translation. Towards this end, we present CETBench - Code Equivalence with Transformations Benchmark, constructed via a repository of programs, where two programs in the repository may be solving the same or different tasks. Each instance in our dataset is obtained by taking a pair of programs in the repository and applying a random series of pre-defined code transformations, resulting in (non-)equivalent pairs. Our analysis on this dataset reveals a surprising finding that very simple code transformations in the underlying pair of programs can result in a significant drop in performance of SOTA LLMs for the task of code-equivalence checking. To remedy this, we present a simple fine-tuning-based approach to boost LLM performance on the transformed pairs of programs. Our approach for dataset generation is generic, and can be used with repositories with varying program difficulty levels and allows for applying varying numbers as well as kinds of transformations. In our experiments, we perform ablations over the difficulty level of original programs, as well as the kind of transformations used in generating pairs for equivalence checking. Our analysis presents deep insights into the working of LLMs for the task of code-equivalence, and points to the fact that they may still be far from what could be termed as a semantic understanding of the underlying code. 

---
# Preface to the Special Issue of the TAL Journal on Scholarly Document Processing 

**Authors**: Florian Boudin, Akiko Aizawa  

**Link**: [PDF](https://arxiv.org/pdf/2506.03587)  

**Abstract**: The rapid growth of scholarly literature makes it increasingly difficult for researchers to keep up with new knowledge. Automated tools are now more essential than ever to help navigate and interpret this vast body of information. Scientific papers pose unique difficulties, with their complex language, specialized terminology, and diverse formats, requiring advanced methods to extract reliable and actionable insights. Large language models (LLMs) offer new opportunities, enabling tasks such as literature reviews, writing assistance, and interactive exploration of research. This special issue of the TAL journal highlights research addressing these challenges and, more broadly, research on natural language processing and information retrieval for scholarly and scientific documents. 

---
# Exploiting LLMs for Automatic Hypothesis Assessment via a Logit-Based Calibrated Prior 

**Authors**: Yue Gong, Raul Castro Fernandez  

**Link**: [PDF](https://arxiv.org/pdf/2506.03444)  

**Abstract**: As hypothesis generation becomes increasingly automated, a new bottleneck has emerged: hypothesis assessment. Modern systems can surface thousands of statistical relationships-correlations, trends, causal links-but offer little guidance on which ones are novel, non-trivial, or worthy of expert attention. In this work, we study the complementary problem to hypothesis generation: automatic hypothesis assessment. Specifically, we ask: given a large set of statistical relationships, can we automatically assess which ones are novel and worth further exploration? We focus on correlations as they are a common entry point in exploratory data analysis that often serve as the basis for forming deeper scientific or causal hypotheses.
To support automatic assessment, we propose to leverage the vast knowledge encoded in LLMs' weights to derive a prior distribution over the correlation value of a variable pair. If an LLM's prior expects the correlation value observed, then such correlation is not surprising, and vice versa. We propose the Logit-based Calibrated Prior, an LLM-elicited correlation prior that transforms the model's raw output logits into a calibrated, continuous predictive distribution over correlation values. We evaluate the prior on a benchmark of 2,096 real-world variable pairs and it achieves a sign accuracy of 78.8%, a mean absolute error of 0.26, and 95% credible interval coverage of 89.2% in predicting Pearson correlation coefficient. It also outperforms a fine-tuned RoBERTa classifier in binary correlation prediction and achieves higher precision@K in hypothesis ranking. We further show that the prior generalizes to correlations not seen during LLM pretraining, reflecting context-sensitive reasoning rather than memorization. 

---
# DiaBlo: Diagonal Blocks Are Sufficient For Finetuning 

**Authors**: Selcuk Gurses, Aozhong Zhang, Yanxia Deng, Xun Dong, Xin Li, Naigang Wang, Penghang Yin, Zi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.03230)  

**Abstract**: Finetuning is a critical step for adapting large language models (LLMs) to domain-specific downstream tasks. To mitigate the substantial computational and memory costs of full-model fine-tuning, Parameter-Efficient Finetuning (PEFT) methods have been proposed to update only a small subset of model parameters. However, performance gaps between PEFT approaches and full-model fine-tuning still exist. In this work, we present DiaBlo, a simple yet effective PEFT approach that updates only the diagonal blocks of selected model weight matrices. Unlike Low Rank Adaptation (LoRA) and its variants, DiaBlo eliminates the need for low rank matrix products, thereby avoiding the reliance on auxiliary initialization schemes or customized optimization strategies to improve convergence. This design leads to stable and robust convergence while maintaining comparable memory efficiency and training speed to LoRA. We conduct extensive experiments across a range of tasks, including commonsense reasoning, arithmetic reasoning, code generation, and safety alignment, to evaluate the effectiveness and efficiency of DiaBlo. Across these benchmarks, DiaBlo demonstrates strong and consistent performance while maintaining high memory efficiency and fast finetuning speed. Codes are available at this https URL. 

---
# Adaptive Task Vectors for Large Language Models 

**Authors**: Joonseong Kang, Soojeong Lee, Subeen Park, Sumin Park, Taero Kim, Jihee Kim, Ryunyi Lee, Kyungwoo Song  

**Link**: [PDF](https://arxiv.org/pdf/2506.03426)  

**Abstract**: In-Context Learning (ICL) enables Large Language Models (LLMs) to perform tasks without parameter updates by conditioning on a few demonstrations provided in the prompt. Despite its success, ICL suffers from several limitations, including sensitivity to demonstration order, context length constraints, and computational inefficiency. To address these challenges, task vector-based approaches compress task information into a single vector. However, these methods typically construct task vectors from fixed sets of demonstrations and reuse them across input queries, without conditioning on the specific input. This limitation can lead models to struggle with effective adaptation when the input query is not well aligned with the underlying demonstrations, consequently degrading their generalization performance on unseen tasks. To overcome this limitation, we propose Adaptive Task Vectors (ATV), a simple and effective framework that dynamically generates task vectors conditioned on each input query. ATV employs a small language model to generate task vectors, which are then transformed to match the target LLM's architecture and applied to guide its output generation. In contrast to ICL and previous vector-based approaches, which rely on fixed demonstration sets and their corresponding vectors, ATV dynamically generates task vectors tailored to each specific input query and task. Consequently, ATV demonstrates strong performance and generalization capabilities, even for unseen tasks. Furthermore, we provide a theoretical analysis indicating that ATV is expressively equivalent to LoRA under equal rank budgets and more expressive than Prefix-Tuning, thereby offering formal support for its representational advantage. 

---
# From Instructions to ODRL Usage Policies: An Ontology Guided Approach 

**Authors**: Daham M. Mustafa, Abhishek Nadgeri, Diego Collarana, Benedikt T. Arnold, Christoph Quix, Christoph Lange, Stefan Decker  

**Link**: [PDF](https://arxiv.org/pdf/2506.03301)  

**Abstract**: This study presents an approach that uses large language models such as GPT-4 to generate usage policies in the W3C Open Digital Rights Language ODRL automatically from natural language instructions. Our approach uses the ODRL ontology and its documentation as a central part of the prompt. Our research hypothesis is that a curated version of existing ontology documentation will better guide policy generation. We present various heuristics for adapting the ODRL ontology and its documentation to guide an end-to-end KG construction process. We evaluate our approach in the context of dataspaces, i.e., distributed infrastructures for trustworthy data exchange between multiple participating organizations for the cultural domain. We created a benchmark consisting of 12 use cases of varying complexity. Our evaluation shows excellent results with up to 91.95% accuracy in the resulting knowledge graph. 

---
# Graph Counselor: Adaptive Graph Exploration via Multi-Agent Synergy to Enhance LLM Reasoning 

**Authors**: Junqi Gao, Xiang Zou, YIng Ai, Dong Li, Yichen Niu, Biqing Qi, Jianxing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.03939)  

**Abstract**: Graph Retrieval Augmented Generation (GraphRAG) effectively enhances external knowledge integration capabilities by explicitly modeling knowledge relationships, thereby improving the factual accuracy and generation quality of Large Language Models (LLMs) in specialized domains. However, existing methods suffer from two inherent limitations: 1) Inefficient Information Aggregation: They rely on a single agent and fixed iterative patterns, making it difficult to adaptively capture multi-level textual, structural, and degree information within graph data. 2) Rigid Reasoning Mechanism: They employ preset reasoning schemes, which cannot dynamically adjust reasoning depth nor achieve precise semantic correction. To overcome these limitations, we propose Graph Counselor, an GraphRAG method based on multi-agent collaboration. This method uses the Adaptive Graph Information Extraction Module (AGIEM), where Planning, Thought, and Execution Agents work together to precisely model complex graph structures and dynamically adjust information extraction strategies, addressing the challenges of multi-level dependency modeling and adaptive reasoning depth. Additionally, the Self-Reflection with Multiple Perspectives (SR) module improves the accuracy and semantic consistency of reasoning results through self-reflection and backward reasoning mechanisms. Experiments demonstrate that Graph Counselor outperforms existing methods in multiple graph reasoning tasks, exhibiting higher reasoning accuracy and generalization ability. Our code is available at this https URL. 

---
# Delta-KNN: Improving Demonstration Selection in In-Context Learning for Alzheimer's Disease Detection 

**Authors**: Chuyuan Li, Raymond Li, Thalia S. Field, Giuseppe Carenini  

**Link**: [PDF](https://arxiv.org/pdf/2506.03476)  

**Abstract**: Alzheimer's Disease (AD) is a progressive neurodegenerative disorder that leads to dementia, and early intervention can greatly benefit from analyzing linguistic abnormalities. In this work, we explore the potential of Large Language Models (LLMs) as health assistants for AD diagnosis from patient-generated text using in-context learning (ICL), where tasks are defined through a few input-output examples. Empirical results reveal that conventional ICL methods, such as similarity-based selection, perform poorly for AD diagnosis, likely due to the inherent complexity of this task. To address this, we introduce Delta-KNN, a novel demonstration selection strategy that enhances ICL performance. Our method leverages a delta score to assess the relative gains of each training example, coupled with a KNN-based retriever that dynamically selects optimal "representatives" for a given input. Experiments on two AD detection datasets across three open-source LLMs demonstrate that Delta-KNN consistently outperforms existing ICL baselines. Notably, when using the Llama-3.1 model, our approach achieves new state-of-the-art results, surpassing even supervised classifiers. 

---
# Reason from Future: Reverse Thought Chain Enhances LLM Reasoning 

**Authors**: Yinlong Xu, Yanzhao Zheng, Shuoshuo Sun, Shuaihan Huang, Baohua Dong, Hangcheng Zhu, Ruohui Huang, Gang Yu, Hongxia Xu, Jian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.03673)  

**Abstract**: It has been demonstrated that carefully designed reasoning paradigms, like Chain-of-Thought (CoT) and Tree-of-Thought (ToT), can enhance the reasoning capabilities of small language models by detailed thinking and extensive thought searching, unbounded branching factors in the searching space create prohibitive reasoning consumption. However these methods fall into the trap of local optimum reasoning, which means the model lacks a global perspective while solving problems. We propose a novel reasoning paradigm called Reason from Future (RFF), which generates reasoning paths by bidirectional reasoning that combines top-down planning with bottom-up reasoning accumulation. The essence of RFF lies in its reverse reasoning mechanism, which prioritizes core logical relationships and imposes goal-oriented constraints on intermediate steps, thereby reducing the searching space and mitigating error accumulation inherent in sequential forward reasoning. Empirical evaluations across diverse experiments demonstrate that RFF outperforms conventional paradigms with higher accuracy and less searching space to solve complex tasks. 

---
# AssetOpsBench: Benchmarking AI Agents for Task Automation in Industrial Asset Operations and Maintenance 

**Authors**: Dhaval Patel, Shuxin Lin, James Rayfield, Nianjun Zhou, Roman Vaculin, Natalia Martinez, Fearghal O'donncha, Jayant Kalagnanam  

**Link**: [PDF](https://arxiv.org/pdf/2506.03828)  

**Abstract**: AI for Industrial Asset Lifecycle Management aims to automate complex operational workflows -- such as condition monitoring, maintenance planning, and intervention scheduling -- to reduce human workload and minimize system downtime. Traditional AI/ML approaches have primarily tackled these problems in isolation, solving narrow tasks within the broader operational pipeline. In contrast, the emergence of AI agents and large language models (LLMs) introduces a next-generation opportunity: enabling end-to-end automation across the entire asset lifecycle. This paper envisions a future where AI agents autonomously manage tasks that previously required distinct expertise and manual coordination. To this end, we introduce AssetOpsBench -- a unified framework and environment designed to guide the development, orchestration, and evaluation of domain-specific agents tailored for Industry 4.0 applications. We outline the key requirements for such holistic systems and provide actionable insights into building agents that integrate perception, reasoning, and control for real-world industrial operations. The software is available at this https URL. 

---
# CogniPair: From LLM Chatbots to Conscious AI Agents -- GNWT-Based Multi-Agent Digital Twins for Social Pairing -- Dating & Hiring Applications 

**Authors**: Wanghao Ye, Sihan Chen, Yiting Wang, Shwai He, Bowei Tian, Guoheng Sun, Ziyi Wang, Ziyao Wang, Yexiao He, Zheyu Shen, Meng Liu, Yuning Zhang, Meng Feng, Yang Wang, Siyuan Peng, Yilong Dai, Zhenle Duan, Hanzhang Qin, Ang Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.03543)  

**Abstract**: Current large language model (LLM) agents lack authentic human psychological processes necessary for genuine digital twins and social AI applications. To address this limitation, we present a computational implementation of Global Workspace Theory (GNWT) that integrates human cognitive architecture principles into LLM agents, creating specialized sub-agents for emotion, memory, social norms, planning, and goal-tracking coordinated through a global workspace mechanism. However, authentic digital twins require accurate personality initialization. We therefore develop a novel adventure-based personality test that evaluates true personality through behavioral choices within interactive scenarios, bypassing self-presentation bias found in traditional assessments. Building on these innovations, our CogniPair platform enables digital twins to engage in realistic simulated dating interactions and job interviews before real encounters, providing bidirectional cultural fit assessment for both romantic compatibility and workplace matching. Validation using 551 GNWT-Agents and Columbia University Speed Dating dataset demonstrates 72% correlation with human attraction patterns, 77.8% match prediction accuracy, and 74% agreement in human validation studies. This work advances psychological authenticity in LLM agents and establishes a foundation for intelligent dating platforms and HR technology solutions. 

---
# TRiSM for Agentic AI: A Review of Trust, Risk, and Security Management in LLM-based Agentic Multi-Agent Systems 

**Authors**: Shaina Raza, Ranjan Sapkota, Manoj Karkee, Christos Emmanouilidis  

**Link**: [PDF](https://arxiv.org/pdf/2506.04133)  

**Abstract**: Agentic AI systems, built on large language models (LLMs) and deployed in multi-agent configurations, are redefining intelligent autonomy, collaboration and decision-making across enterprise and societal domains. This review presents a structured analysis of Trust, Risk, and Security Management (TRiSM) in the context of LLM-based agentic multi-agent systems (AMAS). We begin by examining the conceptual foundations of agentic AI, its architectural differences from traditional AI agents, and the emerging system designs that enable scalable, tool-using autonomy. The TRiSM in the agentic AI framework is then detailed through four pillars governance, explainability, ModelOps, and privacy/security each contextualized for agentic LLMs. We identify unique threat vectors and introduce a comprehensive risk taxonomy for the agentic AI applications, supported by case studies illustrating real-world vulnerabilities. Furthermore, the paper also surveys trust-building mechanisms, transparency and oversight techniques, and state-of-the-art explainability strategies in distributed LLM agent systems. Additionally, metrics for evaluating trust, interpretability, and human-centered performance are reviewed alongside open benchmarking challenges. Security and privacy are addressed through encryption, adversarial defense, and compliance with evolving AI regulations. The paper concludes with a roadmap for responsible agentic AI, proposing research directions to align emerging multi-agent systems with robust TRiSM principles for safe, accountable, and transparent deployment. 

---
# Helpful Agent Meets Deceptive Judge: Understanding Vulnerabilities in Agentic Workflows 

**Authors**: Yifei Ming, Zixuan Ke, Xuan-Phi Nguyen, Jiayu Wang, Shafiq Joty  

**Link**: [PDF](https://arxiv.org/pdf/2506.03332)  

**Abstract**: Agentic workflows -- where multiple large language model (LLM) instances interact to solve tasks -- are increasingly built on feedback mechanisms, where one model evaluates and critiques another. Despite the promise of feedback-driven improvement, the stability of agentic workflows rests on the reliability of the judge. However, judges may hallucinate information, exhibit bias, or act adversarially -- introducing critical vulnerabilities into the workflow. In this work, we present a systematic analysis of agentic workflows under deceptive or misleading feedback. We introduce a two-dimensional framework for analyzing judge behavior, along axes of intent (from constructive to malicious) and knowledge (from parametric-only to retrieval-augmented systems). Using this taxonomy, we construct a suite of judge behaviors and develop WAFER-QA, a new benchmark with critiques grounded in retrieved web evidence to evaluate robustness of agentic workflows against factually supported adversarial feedback. We reveal that even strongest agents are vulnerable to persuasive yet flawed critiques -- often switching correct answers after a single round of misleading feedback. Taking a step further, we study how model predictions evolve over multiple rounds of interaction, revealing distinct behavioral patterns between reasoning and non-reasoning models. Our findings highlight fundamental vulnerabilities in feedback-based workflows and offer guidance for building more robust agentic systems. 

---
# Advancing Multimodal Reasoning: From Optimized Cold Start to Staged Reinforcement Learning 

**Authors**: Shuang Chen, Yue Guo, Zhaochen Su, Yafu Li, Yulun Wu, Jiacheng Chen, Jiayu Chen, Weijie Wang, Xiaoye Qu, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.04207)  

**Abstract**: Inspired by the remarkable reasoning capabilities of Deepseek-R1 in complex textual tasks, many works attempt to incentivize similar capabilities in Multimodal Large Language Models (MLLMs) by directly applying reinforcement learning (RL). However, they still struggle to activate complex reasoning. In this paper, rather than examining multimodal RL in isolation, we delve into current training pipelines and identify three crucial phenomena: 1) Effective cold start initialization is critical for enhancing MLLM reasoning. Intriguingly, we find that initializing with carefully selected text data alone can lead to performance surpassing many recent multimodal reasoning models, even before multimodal RL. 2) Standard GRPO applied to multimodal RL suffers from gradient stagnation, which degrades training stability and performance. 3) Subsequent text-only RL training, following the multimodal RL phase, further enhances multimodal reasoning. This staged training approach effectively balances perceptual grounding and cognitive reasoning development. By incorporating the above insights and addressing multimodal RL issues, we introduce ReVisual-R1, achieving a new state-of-the-art among open-source 7B MLLMs on challenging benchmarks including MathVerse, MathVision, WeMath, LogicVista, DynaMath, and challenging AIME2024 and AIME2025. 

---
# TracLLM: A Generic Framework for Attributing Long Context LLMs 

**Authors**: Yanting Wang, Wei Zou, Runpeng Geng, Jinyuan Jia  

**Link**: [PDF](https://arxiv.org/pdf/2506.04202)  

**Abstract**: Long context large language models (LLMs) are deployed in many real-world applications such as RAG, agent, and broad LLM-integrated applications. Given an instruction and a long context (e.g., documents, PDF files, webpages), a long context LLM can generate an output grounded in the provided context, aiming to provide more accurate, up-to-date, and verifiable outputs while reducing hallucinations and unsupported claims. This raises a research question: how to pinpoint the texts (e.g., sentences, passages, or paragraphs) in the context that contribute most to or are responsible for the generated output by an LLM? This process, which we call context traceback, has various real-world applications, such as 1) debugging LLM-based systems, 2) conducting post-attack forensic analysis for attacks (e.g., prompt injection attack, knowledge corruption attacks) to an LLM, and 3) highlighting knowledge sources to enhance the trust of users towards outputs generated by LLMs. When applied to context traceback for long context LLMs, existing feature attribution methods such as Shapley have sub-optimal performance and/or incur a large computational cost. In this work, we develop TracLLM, the first generic context traceback framework tailored to long context LLMs. Our framework can improve the effectiveness and efficiency of existing feature attribution methods. To improve the efficiency, we develop an informed search based algorithm in TracLLM. We also develop contribution score ensemble/denoising techniques to improve the accuracy of TracLLM. Our evaluation results show TracLLM can effectively identify texts in a long context that lead to the output of an LLM. Our code and data are at: this https URL. 

---
# Generating Automotive Code: Large Language Models for Software Development and Verification in Safety-Critical Systems 

**Authors**: Sven Kirchner, Alois C. Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2506.04038)  

**Abstract**: Developing safety-critical automotive software presents significant challenges due to increasing system complexity and strict regulatory demands. This paper proposes a novel framework integrating Generative Artificial Intelligence (GenAI) into the Software Development Lifecycle (SDLC). The framework uses Large Language Models (LLMs) to automate code generation in languages such as C++, incorporating safety-focused practices such as static verification, test-driven development and iterative refinement. A feedback-driven pipeline ensures the integration of test, simulation and verification for compliance with safety standards. The framework is validated through the development of an Adaptive Cruise Control (ACC) system. Comparative benchmarking of LLMs ensures optimal model selection for accuracy and reliability. Results demonstrate that the framework enables automatic code generation while ensuring compliance with safety-critical requirements, systematically integrating GenAI into automotive software engineering. This work advances the use of AI in safety-critical domains, bridging the gap between state-of-the-art generative models and real-world safety requirements. 

---
# Sampling Preferences Yields Simple Trustworthiness Scores 

**Authors**: Sean Steinle  

**Link**: [PDF](https://arxiv.org/pdf/2506.03399)  

**Abstract**: With the onset of large language models (LLMs), the performance of artificial intelligence (AI) models is becoming increasingly multi-dimensional. Accordingly, there have been several large, multi-dimensional evaluation frameworks put forward to evaluate LLMs. Though these frameworks are much more realistic than previous attempts which only used a single score like accuracy, multi-dimensional evaluations can complicate decision-making since there is no obvious way to select an optimal model. This work introduces preference sampling, a method to extract a scalar trustworthiness score from multi-dimensional evaluation results by considering the many characteristics of model performance which users value. We show that preference sampling improves upon alternate aggregation methods by using multi-dimensional trustworthiness evaluations of LLMs from TrustLLM and DecodingTrust. We find that preference sampling is consistently reductive, fully reducing the set of candidate models 100% of the time whereas Pareto optimality never reduces the set by more than 50%. Likewise, preference sampling is consistently sensitive to user priors-allowing users to specify the relative weighting and confidence of their preferences-whereas averaging scores is intransigent to the users' prior knowledge. 

---
# KG-BiLM: Knowledge Graph Embedding via Bidirectional Language Models 

**Authors**: Zirui Chen, Xin Wang, Zhao Li, Wenbin Guo, Dongxiao He  

**Link**: [PDF](https://arxiv.org/pdf/2506.03576)  

**Abstract**: Recent advances in knowledge representation learning (KRL) highlight the urgent necessity to unify symbolic knowledge graphs (KGs) with language models (LMs) for richer semantic understanding. However, existing approaches typically prioritize either graph structure or textual semantics, leaving a gap: a unified framework that simultaneously captures global KG connectivity, nuanced linguistic context, and discriminative reasoning semantics. To bridge this gap, we introduce KG-BiLM, a bidirectional LM framework that fuses structural cues from KGs with the semantic expressiveness of generative transformers. KG-BiLM incorporates three key components: (i) Bidirectional Knowledge Attention, which removes the causal mask to enable full interaction among all tokens and entities; (ii) Knowledge-Masked Prediction, which encourages the model to leverage both local semantic contexts and global graph connectivity; and (iii) Contrastive Graph Semantic Aggregation, which preserves KG structure via contrastive alignment of sampled sub-graph representations. Extensive experiments on standard benchmarks demonstrate that KG-BiLM outperforms strong baselines in link prediction, especially on large-scale graphs with complex multi-hop relations - validating its effectiveness in unifying structural information and textual semantics. 

---
# Adversarial Attacks on Robotic Vision Language Action Models 

**Authors**: Eliot Krzysztof Jones, Alexander Robey, Andy Zou, Zachary Ravichandran, George J. Pappas, Hamed Hassani, Matt Fredrikson, J. Zico Kolter  

**Link**: [PDF](https://arxiv.org/pdf/2506.03350)  

**Abstract**: The emergence of vision-language-action models (VLAs) for end-to-end control is reshaping the field of robotics by enabling the fusion of multimodal sensory inputs at the billion-parameter scale. The capabilities of VLAs stem primarily from their architectures, which are often based on frontier large language models (LLMs). However, LLMs are known to be susceptible to adversarial misuse, and given the significant physical risks inherent to robotics, questions remain regarding the extent to which VLAs inherit these vulnerabilities. Motivated by these concerns, in this work we initiate the study of adversarial attacks on VLA-controlled robots. Our main algorithmic contribution is the adaptation and application of LLM jailbreaking attacks to obtain complete control authority over VLAs. We find that textual attacks, which are applied once at the beginning of a rollout, facilitate full reachability of the action space of commonly used VLAs and often persist over longer horizons. This differs significantly from LLM jailbreaking literature, as attacks in the real world do not have to be semantically linked to notions of harm. We make all code available at this https URL . 

---
# NetPress: Dynamically Generated LLM Benchmarks for Network Applications 

**Authors**: Yajie Zhou, Jiajun Ruan, Eric S. Wang, Sadjad Fouladi, Francis Y. Yan, Kevin Hsieh, Zaoxing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.03231)  

**Abstract**: Despite growing interest in domain-specific benchmarking of large language models (LLMs) and agents, current evaluations remain limited to static, small-scale datasets, especially in high-stakes tasks like network operations that demand reliability for deployments. We present NetPress, an automated benchmark generation framework for evaluating LLM agents in network applications. NetPress introduces a unified abstraction with state and action, enabling dynamic generation of diverse query sets along with corresponding ground truths. At runtime, users can specify benchmark configurations to generate millions of queries on the fly. In addition to dynamic benchmark construction, NetPress integrates with network emulators to provide realistic environment feedback, supporting comprehensive evaluation across correctness, safety, and latency. We instantiate NetPress on three representative applications, revealing interesting fine-grained differences in agent behavior that static, correctness-only benchmarks often miss. NetPress moves LLM evaluation toward realistic, scalable testing in infrastructure-centric domains, helping close the gap between benchmark performance and real-world deployment readiness. Code is available at this https URL. 

---
# Multimodal Generative AI with Autoregressive LLMs for Human Motion Understanding and Generation: A Way Forward 

**Authors**: Muhammad Islam, Tao Huang, Euijoon Ahn, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2506.03191)  

**Abstract**: This paper presents an in-depth survey on the use of multimodal Generative Artificial Intelligence (GenAI) and autoregressive Large Language Models (LLMs) for human motion understanding and generation, offering insights into emerging methods, architectures, and their potential to advance realistic and versatile motion synthesis. Focusing exclusively on text and motion modalities, this research investigates how textual descriptions can guide the generation of complex, human-like motion sequences. The paper explores various generative approaches, including autoregressive models, diffusion models, Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs), and transformer-based models, by analyzing their strengths and limitations in terms of motion quality, computational efficiency, and adaptability. It highlights recent advances in text-conditioned motion generation, where textual inputs are used to control and refine motion outputs with greater precision. The integration of LLMs further enhances these models by enabling semantic alignment between instructions and motion, improving coherence and contextual relevance. This systematic survey underscores the transformative potential of text-to-motion GenAI and LLM architectures in applications such as healthcare, humanoids, gaming, animation, and assistive technologies, while addressing ongoing challenges in generating efficient and realistic human motion. 

---
# LLaMA-XR: A Novel Framework for Radiology Report Generation using LLaMA and QLoRA Fine Tuning 

**Authors**: Md. Zihad Bin Jahangir, Muhammad Ashad Kabir, Sumaiya Akter, Israt Jahan, Minh Chau  

**Link**: [PDF](https://arxiv.org/pdf/2506.03178)  

**Abstract**: Automated radiology report generation holds significant potential to reduce radiologists' workload and enhance diagnostic accuracy. However, generating precise and clinically meaningful reports from chest radiographs remains challenging due to the complexity of medical language and the need for contextual understanding. Existing models often struggle with maintaining both accuracy and contextual relevance. In this paper, we present LLaMA-XR, a novel framework that integrates LLaMA 3.1 with DenseNet-121-based image embeddings and Quantized Low-Rank Adaptation (QLoRA) fine-tuning. LLaMA-XR achieves improved coherence and clinical accuracy while maintaining computational efficiency. This efficiency is driven by an optimization strategy that enhances parameter utilization and reduces memory overhead, enabling faster report generation with lower computational resource demands. Extensive experiments conducted on the IU X-ray benchmark dataset demonstrate that LLaMA-XR outperforms a range of state-of-the-art methods. Our model achieves a ROUGE-L score of 0.433 and a METEOR score of 0.336, establishing new performance benchmarks in the domain. These results underscore LLaMA-XR's potential as an effective and efficient AI system for automated radiology reporting, offering enhanced clinical utility and reliability. 

---
# LLM Code Customization with Visual Results: A Benchmark on TikZ 

**Authors**: Charly Reux, Mathieu Acher, Djamel Eddine Khelladi, Olivier Barais, Clément Quinton  

**Link**: [PDF](https://arxiv.org/pdf/2505.04670)  

**Abstract**: With the rise of AI-based code generation, customizing existing code out of natural language instructions to modify visual results -such as figures or images -has become possible, promising to reduce the need for deep programming expertise. However, even experienced developers can struggle with this task, as it requires identifying relevant code regions (feature location), generating valid code variants, and ensuring the modifications reliably align with user intent. In this paper, we introduce vTikZ, the first benchmark designed to evaluate the ability of Large Language Models (LLMs) to customize code while preserving coherent visual outcomes. Our benchmark consists of carefully curated vTikZ editing scenarios, parameterized ground truths, and a reviewing tool that leverages visual feedback to assess correctness. Empirical evaluation with stateof-the-art LLMs shows that existing solutions struggle to reliably modify code in alignment with visual intent, highlighting a gap in current AI-assisted code editing approaches. We argue that vTikZ opens new research directions for integrating LLMs with visual feedback mechanisms to improve code customization tasks in various domains beyond TikZ, including image processing, art creation, Web design, and 3D modeling. 

---
