# KnowRL: Exploring Knowledgeable Reinforcement Learning for Factuality 

**Authors**: Baochang Ren, Shuofei Qiao, Wenhao Yu, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19807)  

**Abstract**: Large Language Models (LLMs), particularly slow-thinking models, often exhibit severe hallucination, outputting incorrect content due to an inability to accurately recognize knowledge boundaries during reasoning. While Reinforcement Learning (RL) can enhance complex reasoning abilities, its outcome-oriented reward mechanism often lacks factual supervision over the thinking process, further exacerbating the hallucination problem. To address the high hallucination in slow-thinking models, we propose Knowledge-enhanced RL, KnowRL. KnowRL guides models to perform fact-based slow thinking by integrating a factuality reward, based on knowledge verification, into the RL training process, helping them recognize their knowledge boundaries. KnowRL guides models to perform fact-based slow thinking by integrating a factuality reward, based on knowledge verification, into the RL training process, helping them recognize their knowledge boundaries. This targeted factual input during RL training enables the model to learn and internalize fact-based reasoning strategies. By directly rewarding adherence to facts within the reasoning steps, KnowRL fosters a more reliable thinking process. Experimental results on three hallucination evaluation datasets and two reasoning evaluation datasets demonstrate that KnowRL effectively mitigates hallucinations in slow-thinking models while maintaining their original strong reasoning capabilities. Our code is available at this https URL. 

---
# JoyAgents-R1: Joint Evolution Dynamics for Versatile Multi-LLM Agents with Reinforcement Learning 

**Authors**: Ai Han, Junxing Hu, Pu Wei, Zhiqian Zhang, Yuhang Guo, Jiawei Lu, Zicheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19846)  

**Abstract**: Multi-agent reinforcement learning (MARL) has emerged as a prominent paradigm for increasingly complex tasks. However, joint evolution across heterogeneous agents remains challenging due to cooperative inefficiency and training instability. In this paper, we propose the joint evolution dynamics for MARL called JoyAgents-R1, which first applies Group Relative Policy Optimization (GRPO) to the joint training of heterogeneous multi-agents. By iteratively refining agents' large language models (LLMs) and memories, the method achieves holistic equilibrium with optimal decision-making and memory capabilities. Specifically, JoyAgents-R1 first implements node-wise Monte Carlo sampling on the behavior of each agent across entire reasoning trajectories to enhance GRPO sampling efficiency while maintaining policy diversity. Then, our marginal benefit-driven selection strategy identifies top-$K$ sampling groups with maximal reward fluctuations, enabling targeted agent model updates that improve training stability and maximize joint benefits through cost-effective parameter adjustments. Meanwhile, JoyAgents-R1 introduces an adaptive memory evolution mechanism that repurposes GRPO rewards as cost-free supervisory signals to eliminate repetitive reasoning and accelerate convergence. Experiments across general and domain-specific scenarios demonstrate that JoyAgents-R1 achieves performance comparable to that of larger LLMs while built on smaller open-source models. 

---
# LLM-Driven Medical Document Analysis: Enhancing Trustworthy Pathology and Differential Diagnosis 

**Authors**: Lei Kang, Xuanshuo Fu, Oriol Ramos Terrades, Javier Vazquez-Corral, Ernest Valveny, Dimosthenis Karatzas  

**Link**: [PDF](https://arxiv.org/pdf/2506.19702)  

**Abstract**: Medical document analysis plays a crucial role in extracting essential clinical insights from unstructured healthcare records, supporting critical tasks such as differential diagnosis. Determining the most probable condition among overlapping symptoms requires precise evaluation and deep medical expertise. While recent advancements in large language models (LLMs) have significantly enhanced performance in medical document analysis, privacy concerns related to sensitive patient data limit the use of online LLMs services in clinical settings. To address these challenges, we propose a trustworthy medical document analysis platform that fine-tunes a LLaMA-v3 using low-rank adaptation, specifically optimized for differential diagnosis tasks. Our approach utilizes DDXPlus, the largest benchmark dataset for differential diagnosis, and demonstrates superior performance in pathology prediction and variable-length differential diagnosis compared to existing methods. The developed web-based platform allows users to submit their own unstructured medical documents and receive accurate, explainable diagnostic results. By incorporating advanced explainability techniques, the system ensures transparent and reliable predictions, fostering user trust and confidence. Extensive evaluations confirm that the proposed method surpasses current state-of-the-art models in predictive accuracy while offering practical utility in clinical settings. This work addresses the urgent need for reliable, explainable, and privacy-preserving artificial intelligence solutions, representing a significant advancement in intelligent medical document analysis for real-world healthcare applications. The code can be found at \href{this https URL}{this https URL}. 

---
# SAGE: Strategy-Adaptive Generation Engine for Query Rewriting 

**Authors**: Teng Wang, Hailei Gong, Changwang Zhang, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19783)  

**Abstract**: Query rewriting is pivotal for enhancing dense retrieval, yet current methods demand large-scale supervised data or suffer from inefficient reinforcement learning (RL) exploration. In this work, we first establish that guiding Large Language Models (LLMs) with a concise set of expert-crafted strategies, such as semantic expansion and entity disambiguation, substantially improves retrieval effectiveness on challenging benchmarks, including HotpotQA, FEVER, NFCorpus, and SciFact. Building on this insight, we introduce the Strategy-Adaptive Generation Engine (SAGE), which operationalizes these strategies in an RL framework. SAGE introduces two novel reward shaping mechanisms-Strategic Credit Shaping (SCS) and Contrastive Reward Shaping (CRS)-to deliver more informative learning signals. This strategy-guided approach not only achieves new state-of-the-art NDCG@10 results, but also uncovers a compelling emergent behavior: the agent learns to select optimal strategies, reduces unnecessary exploration, and generates concise rewrites, lowering inference cost without sacrificing performance. Our findings demonstrate that strategy-guided RL, enhanced with nuanced reward shaping, offers a scalable, efficient, and more interpretable paradigm for developing the next generation of robust information retrieval systems. 

---
# Adaptive Domain Modeling with Language Models: A Multi-Agent Approach to Task Planning 

**Authors**: Harisankar Babu, Philipp Schillinger, Tamim Asfour  

**Link**: [PDF](https://arxiv.org/pdf/2506.19592)  

**Abstract**: We introduce TAPAS (Task-based Adaptation and Planning using AgentS), a multi-agent framework that integrates Large Language Models (LLMs) with symbolic planning to solve complex tasks without the need for manually defined environment models. TAPAS employs specialized LLM-based agents that collaboratively generate and adapt domain models, initial states, and goal specifications as needed using structured tool-calling mechanisms. Through this tool-based interaction, downstream agents can request modifications from upstream agents, enabling adaptation to novel attributes and constraints without manual domain redefinition. A ReAct (Reason+Act)-style execution agent, coupled with natural language plan translation, bridges the gap between dynamically generated plans and real-world robot capabilities. TAPAS demonstrates strong performance in benchmark planning domains and in the VirtualHome simulated real-world environment. 

---
# KunLunBaizeRAG: Reinforcement Learning Driven Inference Performance Leap for Large Language Models 

**Authors**: Cheng Li, Jiexiong Liu, Yixuan Chen, Qihang Zhou, KunLun Meta  

**Link**: [PDF](https://arxiv.org/pdf/2506.19466)  

**Abstract**: This paper introduces KunLunBaizeRAG, a reinforcement learning-driven reasoning framework designed to enhance the reasoning capabilities of large language models (LLMs) in complex multi-hop question-answering tasks. The framework addresses key limitations of traditional RAG, such as retrieval drift, information redundancy, and strategy rigidity. Key innovations include the RAG-driven Reasoning Alignment (RDRA) mechanism, the Search-Think Iterative Enhancement (STIE) mechanism, the Network-Local Intelligent Routing (NLR) mechanism, and a progressive hybrid training strategy. Experimental results demonstrate significant improvements in exact match (EM) and LLM-judged score (LJ) across four benchmarks, highlighting the framework's robustness and effectiveness in complex reasoning scenarios. 

---
# FEAT: A Preference Feedback Dataset through a Cost-Effective Auto-Generation and Labeling Framework for English AI Tutoring 

**Authors**: Hyein Seo, Taewook Hwang, Yohan Lee, sangkeun Jung  

**Link**: [PDF](https://arxiv.org/pdf/2506.19325)  

**Abstract**: In English education tutoring, teacher feedback is essential for guiding students. Recently, AI-based tutoring systems have emerged to assist teachers; however, these systems require high-quality and large-scale teacher feedback data, which is both time-consuming and costly to generate manually. In this study, we propose FEAT, a cost-effective framework for generating teacher feedback, and have constructed three complementary datasets: (1) DIRECT-Manual (DM), where both humans and large language models (LLMs) collaboratively generate high-quality teacher feedback, albeit at a higher cost; (2) DIRECT-Generated (DG), an LLM-only generated, cost-effective dataset with lower quality;, and (3) DIRECT-Augmented (DA), primarily based on DG with a small portion of DM added to enhance quality while maintaining cost-efficiency. Experimental results showed that incorporating a small portion of DM (5-10%) into DG leads to superior performance compared to using 100% DM alone. 

---
# Baba is LLM: Reasoning in a Game with Dynamic Rules 

**Authors**: Fien van Wetten, Aske Plaat, Max van Duijn  

**Link**: [PDF](https://arxiv.org/pdf/2506.19095)  

**Abstract**: Large language models (LLMs) are known to perform well on language tasks, but struggle with reasoning tasks. This paper explores the ability of LLMs to play the 2D puzzle game Baba is You, in which players manipulate rules by rearranging text blocks that define object properties. Given that this rule-manipulation relies on language abilities and reasoning, it is a compelling challenge for LLMs. Six LLMs are evaluated using different prompt types, including (1) simple, (2) rule-extended and (3) action-extended prompts. In addition, two models (Mistral, OLMo) are finetuned using textual and structural data from the game. Results show that while larger models (particularly GPT-4o) perform better in reasoning and puzzle solving, smaller unadapted models struggle to recognize game mechanics or apply rule changes. Finetuning improves the ability to analyze the game levels, but does not significantly improve solution formulation. We conclude that even for state-of-the-art and finetuned LLMs, reasoning about dynamic rule changes is difficult (specifically, understanding the use-mention distinction). The results provide insights into the applicability of LLMs to complex problem-solving tasks and highlight the suitability of games with dynamically changing rules for testing reasoning and reflection by LLMs. 

---
# Do LLMs Know When to Flip a Coin? Strategic Randomization through Reasoning and Experience 

**Authors**: Lingyu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18928)  

**Abstract**: Strategic randomization is a key principle in game theory, yet it remains underexplored in large language models (LLMs). Prior work often conflates the cognitive decision to randomize with the mechanical generation of randomness, leading to incomplete evaluations. To address this, we propose a novel zero-sum game inspired by the Tian Ji Horse Race, where the Nash equilibrium corresponds to a maximal entropy strategy. The game's complexity masks this property from untrained humans and underdeveloped LLMs. We evaluate five LLMs across prompt styles -- framed, neutral, and hinted -- using competitive multi-tournament gameplay with system-provided random choices, isolating the decision to randomize. Results show that weaker models remain deterministic regardless of prompts, while stronger models exhibit increased randomization under explicit hints. When facing weaker models, strong LLMs adopt deterministic strategies to exploit biases, but converge toward equilibrium play when facing peers. Through win/loss outcomes and Bayes factor analysis, we demonstrate meaningful variation in LLMs' strategic reasoning capabilities, highlighting opportunities for improvement in abstract reasoning and adaptive learning. We make our implementation publicly available at this https URL to ensure full reproducibility. 

---
# Commander-GPT: Dividing and Routing for Multimodal Sarcasm Detection 

**Authors**: Yazhou Zhang, Chunwang Zou, Bo Wang, Jing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2506.19420)  

**Abstract**: Multimodal sarcasm understanding is a high-order cognitive task. Although large language models (LLMs) have shown impressive performance on many downstream NLP tasks, growing evidence suggests that they struggle with sarcasm understanding. In this paper, we propose Commander-GPT, a modular decision routing framework inspired by military command theory. Rather than relying on a single LLM's capability, Commander-GPT orchestrates a team of specialized LLM agents where each agent will be selectively assigned to a focused sub-task such as context modeling, sentiment analysis, etc. Their outputs are then routed back to the commander, which integrates the information and performs the final sarcasm judgment. To coordinate these agents, we introduce three types of centralized commanders: (1) a trained lightweight encoder-based commander (e.g., multi-modal BERT); (2) four small autoregressive language models, serving as moderately capable commanders (e.g., DeepSeek-VL); (3) two large LLM-based commander (Gemini Pro and GPT-4o) that performs task routing, output aggregation, and sarcasm decision-making in a zero-shot fashion. We evaluate Commander-GPT on the MMSD and MMSD 2.0 benchmarks, comparing five prompting strategies. Experimental results show that our framework achieves 4.4% and 11.7% improvement in F1 score over state-of-the-art (SoTA) baselines on average, demonstrating its effectiveness. 

---
# Spiritual-LLM : Gita Inspired Mental Health Therapy In the Era of LLMs 

**Authors**: Janak Kapuriya, Aman Singh, Jainendra Shukla, Rajiv Ratn Shah  

**Link**: [PDF](https://arxiv.org/pdf/2506.19185)  

**Abstract**: Traditional mental health support systems often generate responses based solely on the user's current emotion and situations, resulting in superficial interventions that fail to address deeper emotional needs. This study introduces a novel framework by integrating spiritual wisdom from the Bhagavad Gita with advanced large language model GPT-4o to enhance emotional well-being. We present the GITes (Gita Integrated Therapy for Emotional Support) dataset, which enhances the existing ExTES mental health dataset by including 10,729 spiritually guided responses generated by GPT-4o and evaluated by domain experts. We benchmark GITes against 12 state-of-the-art LLMs, including both mental health specific and general purpose models. To evaluate spiritual relevance in generated responses beyond what conventional n-gram based metrics capture, we propose a novel Spiritual Insight metric and automate assessment via an LLM as jury framework using chain-of-thought prompting. Integrating spiritual guidance into AI driven support enhances both NLP and spiritual metrics for the best performing LLM Phi3-Mini 3.2B Instruct, achieving improvements of 122.71% in ROUGE, 126.53% in METEOR, 8.15% in BERT score, 15.92% in Spiritual Insight, 18.61% in Sufficiency and 13.22% in Relevance compared to its zero-shot counterpart. While these results reflect substantial improvements across automated empathy and spirituality metrics, further validation in real world patient populations remains a necessary step. Our findings indicate a strong potential for AI systems enriched with spiritual guidance to enhance user satisfaction and perceived support outcomes. The code and dataset will be publicly available to advance further research in this emerging area. 

---
# SRFT: A Single-Stage Method with Supervised and Reinforcement Fine-Tuning for Reasoning 

**Authors**: Yuqian Fu, Tinghong Chen, Jiajun Chai, Xihuai Wang, Songjun Tu, Guojun Yin, Wei Lin, Qichao Zhang, Yuanheng Zhu, Dongbin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.19767)  

**Abstract**: Large language models (LLMs) have achieved remarkable progress in reasoning tasks, yet the optimal integration of Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) remains a fundamental challenge. Through comprehensive analysis of token distributions, learning dynamics, and integration mechanisms from entropy-based perspectives, we reveal key differences between these paradigms: SFT induces coarse-grained global changes to LLM policy distributions, while RL performs fine-grained selective optimizations, with entropy serving as a critical indicator of training effectiveness. Building on these observations, we propose Supervised Reinforcement Fine-Tuning (SRFT), a single-stage method that unifies both fine-tuning paradigms through entropy-aware weighting mechanisms. Our approach simultaneously applies SFT and RL to directly optimize the LLM using demonstrations and self-exploration rollouts rather than through two-stage sequential methods. Extensive experiments show that SRFT achieves 59.1% average accuracy, outperforming zero-RL methods by 9.0% on five mathematical reasoning benchmarks and 10.9% on three out-of-distribution benchmarks. 

---
# RecLLM-R1: A Two-Stage Training Paradigm with Reinforcement Learning and Chain-of-Thought v1 

**Authors**: Yu Xie, Xingkai Ren, Ying Qi, Yao Hu, Lianlei Shan  

**Link**: [PDF](https://arxiv.org/pdf/2506.19235)  

**Abstract**: Traditional recommendation systems often grapple with "filter bubbles", underutilization of external knowledge, and a disconnect between model optimization and business policy iteration. To address these limitations, this paper introduces RecLLM-R1, a novel recommendation framework leveraging Large Language Models (LLMs) and drawing inspiration from the DeepSeek R1 methodology. The framework initiates by transforming user profiles, historical interactions, and multi-faceted item attributes into LLM-interpretable natural language prompts through a carefully engineered data construction process. Subsequently, a two-stage training paradigm is employed: the initial stage involves Supervised Fine-Tuning (SFT) to imbue the LLM with fundamental recommendation capabilities. The subsequent stage utilizes Group Relative Policy Optimization (GRPO), a reinforcement learning technique, augmented with a Chain-of-Thought (CoT) mechanism. This stage guides the model through multi-step reasoning and holistic decision-making via a flexibly defined reward function, aiming to concurrently optimize recommendation accuracy, diversity, and other bespoke business objectives. Empirical evaluations on a real-world user behavior dataset from a large-scale social media platform demonstrate that RecLLM-R1 significantly surpasses existing baseline methods across a spectrum of evaluation metrics, including accuracy, diversity, and novelty. It effectively mitigates the filter bubble effect and presents a promising avenue for the integrated optimization of recommendation models and policies under intricate business goals. 

---
# Arabic Dialect Classification using RNNs, Transformers, and Large Language Models: A Comparative Analysis 

**Authors**: Omar A.Essameldin, Ali O.Elbeih, Wael H.Gomaa, Wael F.Elsersy  

**Link**: [PDF](https://arxiv.org/pdf/2506.19753)  

**Abstract**: The Arabic language is among the most popular languages in the world with a huge variety of dialects spoken in 22 countries. In this study, we address the problem of classifying 18 Arabic dialects of the QADI dataset of Arabic tweets. RNN models, Transformer models, and large language models (LLMs) via prompt engineering are created and tested. Among these, MARBERTv2 performed best with 65% accuracy and 64% F1-score. Through the use of state-of-the-art preprocessing techniques and the latest NLP models, this paper identifies the most significant linguistic issues in Arabic dialect identification. The results corroborate applications like personalized chatbots that respond in users' dialects, social media monitoring, and greater accessibility for Arabic communities. 

---
# ECCoT: A Framework for Enhancing Effective Cognition via Chain of Thought in Large Language Model 

**Authors**: Zhenke Duan, Jiqun Pan, Jiani Tu, Xiaoyi Wang, Yanqing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19599)  

**Abstract**: In the era of large-scale artificial intelligence, Large Language Models (LLMs) have made significant strides in natural language processing. However, they often lack transparency and generate unreliable outputs, raising concerns about their interpretability. To address this, the Chain of Thought (CoT) prompting method structures reasoning into step-by-step deductions. Yet, not all reasoning chains are valid, and errors can lead to unreliable conclusions. We propose ECCoT, an End-to-End Cognitive Chain of Thought Validation Framework, to evaluate and refine reasoning chains in LLMs. ECCoT integrates the Markov Random Field-Embedded Topic Model (MRF-ETM) for topic-aware CoT generation and Causal Sentence-BERT (CSBert) for causal reasoning alignment. By filtering ineffective chains using structured ordering statistics, ECCoT improves interpretability, reduces biases, and enhances the trustworthiness of LLM-based decision-making. Key contributions include the introduction of ECCoT, MRF-ETM for topic-driven CoT generation, and CSBert for causal reasoning enhancement. Code is released at: this https URL. 

---
# Tailored Conversations beyond LLMs: A RL-Based Dialogue Manager 

**Authors**: Lucie Galland, Catherine Pelachaud, Florian Pecune  

**Link**: [PDF](https://arxiv.org/pdf/2506.19652)  

**Abstract**: In this work, we propose a novel framework that integrates large language models (LLMs) with an RL-based dialogue manager for open-ended dialogue with a specific goal. By leveraging hierarchical reinforcement learning to model the structured phases of dialogue and employ meta-learning to enhance adaptability across diverse user profiles, our approach enhances adaptability and efficiency, enabling the system to learn from limited data, transition fluidly between dialogue phases, and personalize responses to heterogeneous patient needs. We apply our framework to Motivational Interviews, aiming to foster behavior change, and demonstrate that the proposed dialogue manager outperforms a state-of-the-art LLM baseline in terms of reward, showing a potential benefit of conditioning LLMs to create open-ended dialogue systems with specific goals. 

---
# PrivacyXray: Detecting Privacy Breaches in LLMs through Semantic Consistency and Probability Certainty 

**Authors**: Jinwen He, Yiyang Lu, Zijin Lin, Kai Chen, Yue Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.19563)  

**Abstract**: Large Language Models (LLMs) are widely used in sensitive domains, including healthcare, finance, and legal services, raising concerns about potential private information leaks during inference. Privacy extraction attacks, such as jailbreaking, expose vulnerabilities in LLMs by crafting inputs that force the models to output sensitive information. However, these attacks cannot verify whether the extracted private information is accurate, as no public datasets exist for cross-validation, leaving a critical gap in private information detection during inference. To address this, we propose PrivacyXray, a novel framework detecting privacy breaches by analyzing LLM inner states. Our analysis reveals that LLMs exhibit higher semantic coherence and probabilistic certainty when generating correct private outputs. Based on this, PrivacyXray detects privacy breaches using four metrics: intra-layer and inter-layer semantic similarity, token-level and sentence-level probability distributions. PrivacyXray addresses critical challenges in private information detection by overcoming the lack of open-source private datasets and eliminating reliance on external data for validation. It achieves this through the synthesis of realistic private data and a detection mechanism based on the inner states of LLMs. Experiments show that PrivacyXray achieves consistent performance, with an average accuracy of 92.69% across five LLMs. Compared to state-of-the-art methods, PrivacyXray achieves significant improvements, with an average accuracy increase of 20.06%, highlighting its stability and practical utility in real-world applications. 

---
# Why Do Open-Source LLMs Struggle with Data Analysis? A Systematic Empirical Study 

**Authors**: Yuqi Zhu, Yi Zhong, Jintian Zhang, Ziheng Zhang, Shuofei Qiao, Yujie Luo, Lun Du, Da Zheng, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19794)  

**Abstract**: Large Language Models (LLMs) hold promise in automating data analysis tasks, yet open-source models face significant limitations in these kinds of reasoning-intensive scenarios. In this work, we investigate strategies to enhance the data analysis capabilities of open-source LLMs. By curating a seed dataset of diverse, realistic scenarios, we evaluate models across three dimensions: data understanding, code generation, and strategic planning. Our analysis reveals three key findings: (1) Strategic planning quality serves as the primary determinant of model performance; (2) Interaction design and task complexity significantly influence reasoning capabilities; (3) Data quality demonstrates a greater impact than diversity in achieving optimal performance. We leverage these insights to develop a data synthesis methodology, demonstrating significant improvements in open-source LLMs' analytical reasoning capabilities. 

---
# Automatic Posology Structuration : What role for LLMs? 

**Authors**: Natalia Bobkova, Laura Zanella-Calzada, Anyes Tafoughalt, Raphaël Teboul, François Plesse, Félix Gaschi  

**Link**: [PDF](https://arxiv.org/pdf/2506.19525)  

**Abstract**: Automatically structuring posology instructions is essential for improving medication safety and enabling clinical decision support. In French prescriptions, these instructions are often ambiguous, irregular, or colloquial, limiting the effectiveness of classic ML pipelines. We explore the use of Large Language Models (LLMs) to convert free-text posologies into structured formats, comparing prompt-based methods and fine-tuning against a "pre-LLM" system based on Named Entity Recognition and Linking (NERL). Our results show that while prompting improves performance, only fine-tuned LLMs match the accuracy of the baseline. Through error analysis, we observe complementary strengths: NERL offers structural precision, while LLMs better handle semantic nuances. Based on this, we propose a hybrid pipeline that routes low-confidence cases from NERL (<0.8) to the LLM, selecting outputs based on confidence scores. This strategy achieves 91% structuration accuracy while minimizing latency and compute. Our results show that this hybrid approach improves structuration accuracy while limiting computational cost, offering a scalable solution for real-world clinical use. 

---
# Dialogic Pedagogy for Large Language Models: Aligning Conversational AI with Proven Theories of Learning 

**Authors**: Russell Beale  

**Link**: [PDF](https://arxiv.org/pdf/2506.19484)  

**Abstract**: Large Language Models (LLMs) are rapidly transforming education by enabling rich conversational learning experiences. This article provides a comprehensive review of how LLM-based conversational agents are being used in higher education, with extensions to secondary and lifelong learning contexts. We synthesize existing literature on LLMs in education and theories of conversational and dialogic pedagogy - including Vygotsky's sociocultural learning (scaffolding and the Zone of Proximal Development), the Socratic method, and Laurillard's conversational framework - and examine how prompting strategies and retrieval-augmented generation (RAG) can align LLM behaviors with these pedagogical theories, and how it can support personalized, adaptive learning. We map educational theories to LLM capabilities, highlighting where LLM-driven dialogue supports established learning principles and where it challenges or falls short of traditional pedagogical assumptions. Notable gaps in applying prior theories to LLMs are identified, such as the models tendency to provide direct answers instead of fostering co-construction of knowledge, and the need to account for the constant availability and broad but non-human expertise of LLM tutors. In response, we propose practical strategies to better align LLM interactions with sound pedagogy - for example, designing prompts that encourage Socratic questioning, scaffolded guidance, and student reflection, as well as integrating retrieval mechanisms to ensure accuracy and contextual relevance. Our aim is to bridge the gap between educational theory and the emerging practice of AI-driven conversational learning, offering insights and tools for making LLM-based dialogues more educationally productive and theory-aligned. 

---
# Surgery-R1: Advancing Surgical-VQLA with Reasoning Multimodal Large Language Model via Reinforcement Learning 

**Authors**: Pengfei Hao, Shuaibo Li, Hongqiu Wang, Zhizhuo Kou, Junhang Zhang, Guang Yang, Lei Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.19469)  

**Abstract**: In recent years, significant progress has been made in the field of surgical scene understanding, particularly in the task of Visual Question Localized-Answering in robotic surgery (Surgical-VQLA). However, existing Surgical-VQLA models lack deep reasoning capabilities and interpretability in surgical scenes, which limits their reliability and potential for development in clinical applications. To address this issue, inspired by the development of Reasoning Multimodal Large Language Models (MLLMs), we first build the Surgery-R1-54k dataset, including paired data for Visual-QA, Grounding-QA, and Chain-of-Thought (CoT). Then, we propose the first Reasoning MLLM for Surgical-VQLA (Surgery-R1). In our Surgery-R1, we design a two-stage fine-tuning mechanism to enable the basic MLLM with complex reasoning abilities by utilizing supervised fine-tuning (SFT) and reinforcement fine-tuning (RFT). Furthermore, for an efficient and high-quality rule-based reward system in our RFT, we design a Multimodal Coherence reward mechanism to mitigate positional illusions that may arise in surgical scenarios. Experiment results demonstrate that Surgery-R1 outperforms other existing state-of-the-art (SOTA) models in the Surgical-VQLA task and widely-used MLLMs, while also validating its reasoning capabilities and the effectiveness of our approach. The code and dataset will be organized in this https URL. 

---
# MuBench: Assessment of Multilingual Capabilities of Large Language Models Across 61 Languages 

**Authors**: Wenhan Han, Yifan Zhang, Zhixun Chen, Binbin Liu, Haobin Lin, Bingni Zhang, Taifeng Wang, Mykola Pechenizkiy, Meng Fang, Yin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.19468)  

**Abstract**: Multilingual large language models (LLMs) are advancing rapidly, with new models frequently claiming support for an increasing number of languages. However, existing evaluation datasets are limited and lack cross-lingual alignment, leaving assessments of multilingual capabilities fragmented in both language and skill coverage. To address this, we introduce MuBench, a benchmark covering 61 languages and evaluating a broad range of capabilities. We evaluate several state-of-the-art multilingual LLMs and find notable gaps between claimed and actual language coverage, particularly a persistent performance disparity between English and low-resource languages. Leveraging MuBench's alignment, we propose Multilingual Consistency (MLC) as a complementary metric to accuracy for analyzing performance bottlenecks and guiding model improvement. Finally, we pretrain a suite of 1.2B-parameter models on English and Chinese with 500B tokens, varying language ratios and parallel data proportions to investigate cross-lingual transfer dynamics. 

---
# Can Large Language Models Capture Human Annotator Disagreements? 

**Authors**: Jingwei Ni, Yu Fan, Vilém Zouhar, Donya Rooein, Alexander Hoyle, Mrinmaya Sachan, Markus Leippold, Dirk Hovy, Elliott Ash  

**Link**: [PDF](https://arxiv.org/pdf/2506.19467)  

**Abstract**: Human annotation variation (i.e., annotation disagreements) is common in NLP and often reflects important information such as task subjectivity and sample ambiguity. While Large Language Models (LLMs) are increasingly used for automatic annotation to reduce human effort, their evaluation often focuses on predicting the majority-voted "ground truth" labels. It is still unclear, however, whether these models also capture informative human annotation variation. Our work addresses this gap by extensively evaluating LLMs' ability to predict annotation disagreements without access to repeated human labels. Our results show that LLMs struggle with modeling disagreements, which can be overlooked by majority label-based evaluations. Notably, while RLVR-style (Reinforcement learning with verifiable rewards) reasoning generally boosts LLM performance, it degrades performance in disagreement prediction. Our findings highlight the critical need for evaluating and improving LLM annotators in disagreement modeling. Code and data at this https URL. 

---
# MATE: LLM-Powered Multi-Agent Translation Environment for Accessibility Applications 

**Authors**: Aleksandr Algazinov, Matt Laing, Paul Laban  

**Link**: [PDF](https://arxiv.org/pdf/2506.19502)  

**Abstract**: Accessibility remains a critical concern in today's society, as many technologies are not developed to support the full range of user needs. Existing multi-agent systems (MAS) often cannot provide comprehensive assistance for users in need due to the lack of customization stemming from closed-source designs. Consequently, individuals with disabilities frequently encounter significant barriers when attempting to interact with digital environments. We introduce MATE, a multimodal accessibility MAS, which performs the modality conversions based on the user's needs. The system is useful for assisting people with disabilities by ensuring that data will be converted to an understandable format. For instance, if the user cannot see well and receives an image, the system converts this image to its audio description. MATE can be applied to a wide range of domains, industries, and areas, such as healthcare, and can become a useful assistant for various groups of users. The system supports multiple types of models, ranging from LLM API calling to using custom machine learning (ML) classifiers. This flexibility ensures that the system can be adapted to various needs and is compatible with a wide variety of hardware. Since the system is expected to run locally, it ensures the privacy and security of sensitive information. In addition, the framework can be effectively integrated with institutional technologies (e.g., digital healthcare service) for real-time user assistance. Furthermore, we introduce ModCon-Task-Identifier, a model that is capable of extracting the precise modality conversion task from the user input. Numerous experiments show that ModCon-Task-Identifier consistently outperforms other LLMs and statistical models on our custom data. Our code and data are publicly available at this https URL. 

---
# Semantic Scene Graph for Ultrasound Image Explanation and Scanning Guidance 

**Authors**: Xuesong Li, Dianye Huang, Yameng Zhang, Nassir Navab, Zhongliang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19683)  

**Abstract**: Understanding medical ultrasound imaging remains a long-standing challenge due to significant visual variability caused by differences in imaging and acquisition parameters. Recent advancements in large language models (LLMs) have been used to automatically generate terminology-rich summaries orientated to clinicians with sufficient physiological knowledge. Nevertheless, the increasing demand for improved ultrasound interpretability and basic scanning guidance among non-expert users, e.g., in point-of-care settings, has not yet been explored. In this study, we first introduce the scene graph (SG) for ultrasound images to explain image content to ordinary and provide guidance for ultrasound scanning. The ultrasound SG is first computed using a transformer-based one-stage method, eliminating the need for explicit object detection. To generate a graspable image explanation for ordinary, the user query is then used to further refine the abstract SG representation through LLMs. Additionally, the predicted SG is explored for its potential in guiding ultrasound scanning toward missing anatomies within the current imaging view, assisting ordinary users in achieving more standardized and complete anatomical exploration. The effectiveness of this SG-based image explanation and scanning guidance has been validated on images from the left and right neck regions, including the carotid and thyroid, across five volunteers. The results demonstrate the potential of the method to maximally democratize ultrasound by enhancing its interpretability and usability for ordinaries. 

---
# Automated Detection of Pre-training Text in Black-box LLMs 

**Authors**: Ruihan Hu, Yu-Ming Shang, Jiankun Peng, Wei Luo, Yazhe Wang, Xi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19399)  

**Abstract**: Detecting whether a given text is a member of the pre-training data of Large Language Models (LLMs) is crucial for ensuring data privacy and copyright protection. Most existing methods rely on the LLM's hidden information (e.g., model parameters or token probabilities), making them ineffective in the black-box setting, where only input and output texts are accessible. Although some methods have been proposed for the black-box setting, they rely on massive manual efforts such as designing complicated questions or instructions. To address these issues, we propose VeilProbe, the first framework for automatically detecting LLMs' pre-training texts in a black-box setting without human intervention. VeilProbe utilizes a sequence-to-sequence mapping model to infer the latent mapping feature between the input text and the corresponding output suffix generated by the LLM. Then it performs the key token perturbations to obtain more distinguishable membership features. Additionally, considering real-world scenarios where the ground-truth training text samples are limited, a prototype-based membership classifier is introduced to alleviate the overfitting issue. Extensive evaluations on three widely used datasets demonstrate that our framework is effective and superior in the black-box setting. 

---
# MedErr-CT: A Visual Question Answering Benchmark for Identifying and Correcting Errors in CT Reports 

**Authors**: Sunggu Kyung, Hyungbin Park, Jinyoung Seo, Jimin Sung, Jihyun Kim, Dongyeong Kim, Wooyoung Jo, Yoojin Nam, Sangah Park, Taehee Kwon, Sang Min Lee, Namkug Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.19217)  

**Abstract**: Computed Tomography (CT) plays a crucial role in clinical diagnosis, but the growing demand for CT examinations has raised concerns about diagnostic errors. While Multimodal Large Language Models (MLLMs) demonstrate promising comprehension of medical knowledge, their tendency to produce inaccurate information highlights the need for rigorous validation. However, existing medical visual question answering (VQA) benchmarks primarily focus on simple visual recognition tasks, lacking clinical relevance and failing to assess expert-level knowledge. We introduce MedErr-CT, a novel benchmark for evaluating medical MLLMs' ability to identify and correct errors in CT reports through a VQA framework. The benchmark includes six error categories - four vision-centric errors (Omission, Insertion, Direction, Size) and two lexical error types (Unit, Typo) - and is organized into three task levels: classification, detection, and correction. Using this benchmark, we quantitatively assess the performance of state-of-the-art 3D medical MLLMs, revealing substantial variation in their capabilities across different error types. Our benchmark contributes to the development of more reliable and clinically applicable MLLMs, ultimately helping reduce diagnostic errors and improve accuracy in clinical practice. The code and datasets are available at this https URL. 

---
# EmoStage: A Framework for Accurate Empathetic Response Generation via Perspective-Taking and Phase Recognition 

**Authors**: Zhiyang Qi, Keiko Takamizo, Mariko Ukiyo, Michimasa Inaba  

**Link**: [PDF](https://arxiv.org/pdf/2506.19279)  

**Abstract**: The rising demand for mental health care has fueled interest in AI-driven counseling systems. While large language models (LLMs) offer significant potential, current approaches face challenges, including limited understanding of clients' psychological states and counseling stages, reliance on high-quality training data, and privacy concerns associated with commercial deployment. To address these issues, we propose EmoStage, a framework that enhances empathetic response generation by leveraging the inference capabilities of open-source LLMs without additional training data. Our framework introduces perspective-taking to infer clients' psychological states and support needs, enabling the generation of emotionally resonant responses. In addition, phase recognition is incorporated to ensure alignment with the counseling process and to prevent contextually inappropriate or inopportune responses. Experiments conducted in both Japanese and Chinese counseling settings demonstrate that EmoStage improves the quality of responses generated by base models and performs competitively with data-driven methods. 

---
# Language Models Might Not Understand You: Evaluating Theory of Mind via Story Prompting 

**Authors**: Nathaniel Getachew, Abulhair Saparov  

**Link**: [PDF](https://arxiv.org/pdf/2506.19089)  

**Abstract**: We introduce $\texttt{StorySim}$, a programmable framework for synthetically generating stories to evaluate the theory of mind (ToM) and world modeling (WM) capabilities of large language models (LLMs). Unlike prior benchmarks that may suffer from contamination in pretraining data, $\texttt{StorySim}$ produces novel, compositional story prompts anchored by a highly controllable $\texttt{Storyboard}$, enabling precise manipulation of character perspectives and events. We use this framework to design first- and second-order ToM tasks alongside WM tasks that control for the ability to track and model mental states. Our experiments across a suite of state-of-the-art LLMs reveal that most models perform better on WM tasks than ToM tasks, and that models tend to perform better reasoning with humans compared to inanimate objects. Additionally, our framework enabled us to find evidence of heuristic behavior such as recency bias and an over-reliance on earlier events in the story. All code for generating data and evaluations is freely available. 

---
# Thought Anchors: Which LLM Reasoning Steps Matter? 

**Authors**: Paul C. Bogdan, Uzay Macar, Neel Nanda, Arthur Conmy  

**Link**: [PDF](https://arxiv.org/pdf/2506.19143)  

**Abstract**: Reasoning large language models have recently achieved state-of-the-art performance in many fields. However, their long-form chain-of-thought reasoning creates interpretability challenges as each generated token depends on all previous ones, making the computation harder to decompose. We argue that analyzing reasoning traces at the sentence level is a promising approach to understanding reasoning processes. We present three complementary attribution methods: (1) a black-box method measuring each sentence's counterfactual importance by comparing final answers across 100 rollouts conditioned on the model generating that sentence or one with a different meaning; (2) a white-box method of aggregating attention patterns between pairs of sentences, which identified ``broadcasting'' sentences that receive disproportionate attention from all future sentences via ``receiver'' attention heads; (3) a causal attribution method measuring logical connections between sentences by suppressing attention toward one sentence and measuring the effect on each future sentence's tokens. Each method provides evidence for the existence of thought anchors, reasoning steps that have outsized importance and that disproportionately influence the subsequent reasoning process. These thought anchors are typically planning or backtracking sentences. We provide an open-source tool (this http URL) for visualizing the outputs of our methods, and present a case study showing converging patterns across methods that map how a model performs multi-step reasoning. The consistency across methods demonstrates the potential of sentence-level analysis for a deeper understanding of reasoning models. 

---
# Improving Student-AI Interaction Through Pedagogical Prompting: An Example in Computer Science Education 

**Authors**: Ruiwei Xiao, Xinying Hou, Runlong Ye, Majeed Kazemitabaar, Nicholas Diana, Michael Liut, John Stamper  

**Link**: [PDF](https://arxiv.org/pdf/2506.19107)  

**Abstract**: With the proliferation of large language model (LLM) applications since 2022, their use in education has sparked both excitement and concern. Recent studies consistently highlight students' (mis)use of LLMs can hinder learning outcomes. This work aims to teach students how to effectively prompt LLMs to improve their learning. We first proposed pedagogical prompting, a theoretically-grounded new concept to elicit learning-oriented responses from LLMs. To move from concept design to a proof-of-concept learning intervention in real educational settings, we selected early undergraduate CS education (CS1/CS2) as the example context. We began with a formative survey study with instructors (N=36) teaching early-stage undergraduate-level CS courses to inform the instructional design based on classroom needs. Based on their insights, we designed and developed a learning intervention through an interactive system with scenario-based instruction to train pedagogical prompting skills. Finally, we evaluated its instructional effectiveness through a user study with CS novice students (N=22) using pre/post-tests. Through mixed methods analyses, our results indicate significant improvements in learners' LLM-based pedagogical help-seeking skills, along with positive attitudes toward the system and increased willingness to use pedagogical prompts in the future. Our contributions include (1) a theoretical framework of pedagogical prompting; (2) empirical insights into current instructor attitudes toward pedagogical prompting; and (3) a learning intervention design with an interactive learning tool and scenario-based instruction leading to promising results on teaching LLM-based help-seeking. Our approach is scalable for broader implementation in classrooms and has the potential to be integrated into tools like ChatGPT as an on-boarding experience to encourage learning-oriented use of generative AI. 

---
# LLMs on a Budget? Say HOLA 

**Authors**: Zohaib Hasan Siddiqui, Jiechao Gao, Ebad Shabbir, Mohammad Anas Azeez, Rafiq Ali, Gautam Siddharth Kashyap, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2506.18952)  

**Abstract**: Running Large Language Models (LLMs) on edge devices is constrained by high compute and memory demands posing a barrier for real-time applications in sectors like healthcare, education, and embedded systems. Current solutions such as quantization, pruning, and retrieval-augmented generation (RAG) offer only partial optimizations and often compromise on speed or accuracy. We introduce HOLA, an end-to-end optimization framework for efficient LLM deployment. Internally, it leverages Hierarchical Speculative Decoding (HSD) for faster inference without quality loss. Externally, AdaComp-RAG adjusts retrieval complexity based on context needs. Together with LoBi, which blends structured pruning (LoRA) and quantization, HOLA delivers significant gains: 17.6% EMA on GSM8K, 10.5% MCA on ARC, and reduced latency and memory on edge devices like Jetson Nano--proving both scalable and production-ready. 

---
# Can AI support student engagement in classroom activities in higher education? 

**Authors**: Neha Rani, Sharan Majumder, Ishan Bhardwaj, Pedro Guillermo Feijoo Garcia  

**Link**: [PDF](https://arxiv.org/pdf/2506.18941)  

**Abstract**: Lucrative career prospects and creative opportunities often attract students to enroll in computer science majors and pursue advanced studies in the field. Consequently, there has been a significant surge in enrollment in computer science courses, resulting in large class sizes that can range from hundreds to even thousands of students. A common challenge in such large classrooms is the lack of engagement between students and both the instructor and the learning material. However, with advancements in technology and improvements in large language models (LLMs), there is a considerable opportunity to utilize LLM-based AI models, such as conversational artificial intelligence (CAI), to enhance student engagement with learning content in large classes. To explore the potential of CAI to support engagement, especially with learning content, we designed an activity in a software Engineering course (with a large class size) where students used CAI for an in-class activity. We conducted a within-subject investigation in a large classroom at a US university where we compared student engagement during an in-class activity that used CAI tool vs. one without CAI tool. The CAI tool we used was ChatGPT due to its widespread popularity and familiarity. Our results indicate that CAI (ChatGPT) has the potential to support engagement with learning content during in-class activities, especially in large class sizes. We further discuss the implications of our findings. 

---
# Privacy-Preserving LLM Interaction with Socratic Chain-of-Thought Reasoning and Homomorphically Encrypted Vector Databases 

**Authors**: Yubeen Bae, Minchan Kim, Jaejin Lee, Sangbum Kim, Jaehyung Kim, Yejin Choi, Niloofar Mireshghallah  

**Link**: [PDF](https://arxiv.org/pdf/2506.17336)  

**Abstract**: Large language models (LLMs) are increasingly used as personal agents, accessing sensitive user data such as calendars, emails, and medical records. Users currently face a trade-off: They can send private records, many of which are stored in remote databases, to powerful but untrusted LLM providers, increasing their exposure risk. Alternatively, they can run less powerful models locally on trusted devices. We bridge this gap. Our Socratic Chain-of-Thought Reasoning first sends a generic, non-private user query to a powerful, untrusted LLM, which generates a Chain-of-Thought (CoT) prompt and detailed sub-queries without accessing user data. Next, we embed these sub-queries and perform encrypted sub-second semantic search using our Homomorphically Encrypted Vector Database across one million entries of a single user's private data. This represents a realistic scale of personal documents, emails, and records accumulated over years of digital activity. Finally, we feed the CoT prompt and the decrypted records to a local language model and generate the final response. On the LoCoMo long-context QA benchmark, our hybrid framework, combining GPT-4o with a local Llama-3.2-1B model, outperforms using GPT-4o alone by up to 7.1 percentage points. This demonstrates a first step toward systems where tasks are decomposed and split between untrusted strong LLMs and weak local ones, preserving user privacy. 

---
# SWE-SQL: Illuminating LLM Pathways to Solve User SQL Issues in Real-World Applications 

**Authors**: Jinyang Li, Xiaolong Li, Ge Qu, Per Jacobsson, Bowen Qin, Binyuan Hui, Shuzheng Si, Nan Huo, Xiaohan Xu, Yue Zhang, Ziwei Tang, Yuanshuai Li, Florensia Widjaja, Xintong Zhu, Feige Zhou, Yongfeng Huang, Yannis Papakonstantinou, Fatma Ozcan, Chenhao Ma, Reynold Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.18951)  

**Abstract**: Resolution of complex SQL issues persists as a significant bottleneck in real-world database applications. Current Large Language Models (LLMs), while adept at text-to-SQL translation, have not been rigorously evaluated on the more challenging task of debugging SQL issues. To address this gap, we introduce BIRD-CRITIC, a new SQL issue debugging benchmark comprising 530 PostgreSQL tasks (BIRD-CRITIC-PG) and 570 multi-dialect tasks (BIRD-CRITIC-Multi), distilled from authentic user issues and replayed within new environments to facilitate rigorous evaluation. Baseline evaluations underscore the task's complexity, with the leading reasoning model O3-Mini achieving only 38.87% success rate on BIRD-CRITIC-PG and 33.33% on BIRD-CRITIC-Multi. Meanwhile, advancing open-source models for database tasks is crucial for empowering local development while safeguarding data privacy. Therefore, we present Six-Gym (Sql-fIX-Gym), a training environment for elevating open-source model capabilities for SQL issue debugging. This environment leverages SQL-Rewind strategy, which automatically generates executable issue-solution datasets by reverse-engineering issues from verified SQLs. However, popular trajectory-based fine-tuning methods do not explore substantial supervisory signals. We further propose f-Plan Boosting, which extracts high-level debugging plans from SQL solutions, enabling teacher LLMs to produce 73.7% more successful trajectories for training. We integrate these components into an open-source agent, Bird-Fixer. Based on Qwen-2.5-Coder-14B, Bird-Fixer achieves 38.11% success rate on BIRD-CRITIC-PG and 29.65% on BIRD-CRITIC-Multi, surpassing leading proprietary models such as Claude-3.7-Sonnet and GPT-4.1, marking a significant step toward democratizing sophisticated SQL-debugging capabilities. The leaderboard and source code are available: this https URL 

---
# Breaking Barriers: Do Reinforcement Post Training Gains Transfer To Unseen Domains? 

**Authors**: Chuxuan Hu, Yuxuan Zhu, Antony Kellermann, Caleb Biddulph, Suppakit Waiwitlikhit, Jason Benn, Daniel Kang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19733)  

**Abstract**: Reinforcement post training (RPT) has recently shown promise in improving the reasoning abilities of large language models (LLMs). However, it remains unclear how well these improvements generalize to new domains, as prior work evaluates RPT models on data from the same domains used for fine-tuning. To understand the generalizability of RPT, we conduct two studies. (1) Observational: We compare a wide range of open-weight RPT models against their corresponding base models across multiple domains, including both seen and unseen domains in their fine-tuning data. (2) Interventional: we fine-tune LLMs with RPT on single domains and evaluate their performance across multiple domains. Both studies converge on the same conclusion that, although RPT brings substantial gains on tasks similar to the fine-tuning data, the gains generalize inconsistently and can vanish on domains with different reasoning patterns. 

---
# MAM: Modular Multi-Agent Framework for Multi-Modal Medical Diagnosis via Role-Specialized Collaboration 

**Authors**: Yucheng Zhou, Lingran Song, Jianbing Shen  

**Link**: [PDF](https://arxiv.org/pdf/2506.19835)  

**Abstract**: Recent advancements in medical Large Language Models (LLMs) have showcased their powerful reasoning and diagnostic capabilities. Despite their success, current unified multimodal medical LLMs face limitations in knowledge update costs, comprehensiveness, and flexibility. To address these challenges, we introduce the Modular Multi-Agent Framework for Multi-Modal Medical Diagnosis (MAM). Inspired by our empirical findings highlighting the benefits of role assignment and diagnostic discernment in LLMs, MAM decomposes the medical diagnostic process into specialized roles: a General Practitioner, Specialist Team, Radiologist, Medical Assistant, and Director, each embodied by an LLM-based agent. This modular and collaborative framework enables efficient knowledge updates and leverages existing medical LLMs and knowledge bases. Extensive experimental evaluations conducted on a wide range of publicly accessible multimodal medical datasets, incorporating text, image, audio, and video modalities, demonstrate that MAM consistently surpasses the performance of modality-specific LLMs. Notably, MAM achieves significant performance improvements ranging from 18% to 365% compared to baseline models. Our code is released at this https URL. 

---
# KnowMap: Efficient Knowledge-Driven Task Adaptation for LLMs 

**Authors**: Kelin Fu, Kaigui Bian  

**Link**: [PDF](https://arxiv.org/pdf/2506.19527)  

**Abstract**: While Large Language Models (LLMs) possess significant capabilities in open-world agent tasks, they also face challenges in rapidly adapting to new, specialized tasks due to their reliance on static pre-trained knowledge. Traditional methods such as fine-tuning are often costly, data-intensive, and may lead to "catastrophic forgetting." Therefore, we present KnowMap, a novel approach that dynamically constructs a knowledge base from environmental and experiential data. KnowMap fine-tunes a small knowledge-embedding model to equip a larger LLM with valuable task-specific knowledge. Our experiments on the ScienceWorld benchmark demonstrate 17.71% improvement for the performance of gpt-4-turbo model. KnowMap not only provides an efficient and effective means for LLM task-adapting, but also highlights how integrating environmental and experiential knowledge can enhance LLMs' reasoning capabilities. 

---
# Is Long-to-Short a Free Lunch? Investigating Inconsistency and Reasoning Efficiency in LRMs 

**Authors**: Shu Yang, Junchao Wu, Xuansheng Wu, Derek Wong, Ninhao Liu, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19492)  

**Abstract**: Large Reasoning Models (LRMs) have achieved remarkable performance on complex tasks by engaging in extended reasoning before producing final answers, yet this strength introduces the risk of overthinking, where excessive token generation occurs even for simple tasks. While recent work in efficient reasoning seeks to reduce reasoning length while preserving accuracy, it remains unclear whether such optimization is truly a free lunch. Drawing on the intuition that compressing reasoning may reduce the robustness of model responses and lead models to omit key reasoning steps, we investigate whether efficient reasoning strategies introduce behavioral inconsistencies. To systematically assess this, we introduce $ICBENCH$, a benchmark designed to measure inconsistency in LRMs across three dimensions: inconsistency across task settings (ITS), inconsistency between training objectives and learned behavior (TR-LB), and inconsistency between internal reasoning and self-explanations (IR-SE). Applying $ICBENCH$ to a range of open-source LRMs, we find that while larger models generally exhibit greater consistency than smaller ones, they all display widespread "scheming" behaviors, including self-disagreement, post-hoc rationalization, and the withholding of reasoning cues. Crucially, our results demonstrate that efficient reasoning strategies such as No-Thinking and Simple Token-Budget consistently increase all three defined types of inconsistency. These findings suggest that although efficient reasoning enhances token-level efficiency, further investigation is imperative to ascertain whether it concurrently introduces the risk of models evading effective supervision. 

---
# Commonsense Generation and Evaluation for Dialogue Systems using Large Language Models 

**Authors**: Marcos Estecha-Garitagoitia, Chen Zhang, Mario Rodríguez-Cantelar, Luis Fernando D'Haro  

**Link**: [PDF](https://arxiv.org/pdf/2506.19483)  

**Abstract**: This paper provides preliminary results on exploring the task of performing turn-level data augmentation for dialogue system based on different types of commonsense relationships, and the automatic evaluation of the generated synthetic turns. The proposed methodology takes advantage of the extended knowledge and zero-shot capabilities of pretrained Large Language Models (LLMs) to follow instructions, understand contextual information, and their commonsense reasoning capabilities. The approach draws inspiration from methodologies like Chain-of-Thought (CoT), applied more explicitly to the task of prompt-based generation for dialogue-based data augmentation conditioned on commonsense attributes, and the automatic evaluation of the generated dialogues.
To assess the effectiveness of the proposed approach, first we extracted 200 randomly selected partial dialogues, from 5 different well-known dialogue datasets, and generate alternative responses conditioned on different event commonsense attributes. This novel dataset allows us to measure the proficiency of LLMs in generating contextually relevant commonsense knowledge, particularly up to 12 different specific ATOMIC [10] database relations. Secondly, we propose an evaluation framework to automatically detect the quality of the generated dataset inspired by the ACCENT [26] metric, which offers a nuanced approach to assess event commonsense. However, our method does not follow ACCENT's complex eventrelation tuple extraction process. Instead, we propose an instruction-based prompt for each commonsense attribute and use state-of-the-art LLMs to automatically detect the original attributes used when creating each augmented turn in the previous step.
Preliminary results suggest that our approach effectively harnesses LLMs capabilities for commonsense reasoning and evaluation in dialogue systems. 

---
# Learning to Disentangle Latent Reasoning Rules with Language VAEs: A Systematic Study 

**Authors**: Yingji Zhang, Marco Valentino, Danilo S. Carvalho, André Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2506.19418)  

**Abstract**: Incorporating explicit reasoning rules within the latent space of language models (LMs) offers a promising pathway to enhance generalisation, interpretability, and controllability. While current Transformer-based language models have shown strong performance on Natural Language Inference (NLI) tasks, they often rely on memorisation rather than rule-based inference. This work investigates how reasoning rules can be explicitly embedded and memorised within the LMs through Language Variational Autoencoders (VAEs). We propose a complete pipeline for learning reasoning rules within Transformer-based language VAEs. This pipeline encompasses three rule-based reasoning tasks, a supporting theoretical framework, and a practical end-to-end architecture. The experiment illustrates the following findings: Disentangled reasoning: Under explicit signal supervision, reasoning rules - viewed as functional mappings - can be disentangled within the encoder's parametric space. This separation results in distinct clustering of rules in the output feature space. Prior knowledge injection: injecting reasoning information into the Query enables the model to more effectively retrieve the stored value Value from memory based on Key. This approach offers a simple method for integrating prior knowledge into decoder-only language models. Performance bottleneck: In mathematical reasoning tasks using Qwen2.5(0.5B), increasing sample count doesn't improve performance beyond a point. Moreover, ffn layers are better than attention layers at preserving the separation of reasoning rules in the model's parameters. 

---
# Measuring and Guiding Monosemanticity 

**Authors**: Ruben Härle, Felix Friedrich, Manuel Brack, Stephan Wäldchen, Björn Deiseroth, Patrick Schramowski, Kristian Kersting  

**Link**: [PDF](https://arxiv.org/pdf/2506.19382)  

**Abstract**: There is growing interest in leveraging mechanistic interpretability and controllability to better understand and influence the internal dynamics of large language models (LLMs). However, current methods face fundamental challenges in reliably localizing and manipulating feature representations. Sparse Autoencoders (SAEs) have recently emerged as a promising direction for feature extraction at scale, yet they, too, are limited by incomplete feature isolation and unreliable monosemanticity. To systematically quantify these limitations, we introduce Feature Monosemanticity Score (FMS), a novel metric to quantify feature monosemanticity in latent representation. Building on these insights, we propose Guided Sparse Autoencoders (G-SAE), a method that conditions latent representations on labeled concepts during training. We demonstrate that reliable localization and disentanglement of target concepts within the latent space improve interpretability, detection of behavior, and control. Specifically, our evaluations on toxicity detection, writing style identification, and privacy attribute recognition show that G-SAE not only enhances monosemanticity but also enables more effective and fine-grained steering with less quality degradation. Our findings provide actionable guidelines for measuring and advancing mechanistic interpretability and control of LLMs. 

---
# Correcting Hallucinations in News Summaries: Exploration of Self-Correcting LLM Methods with External Knowledge 

**Authors**: Juraj Vladika, Ihsan Soydemir, Florian Matthes  

**Link**: [PDF](https://arxiv.org/pdf/2506.19607)  

**Abstract**: While large language models (LLMs) have shown remarkable capabilities to generate coherent text, they suffer from the issue of hallucinations -- factually inaccurate statements. Among numerous approaches to tackle hallucinations, especially promising are the self-correcting methods. They leverage the multi-turn nature of LLMs to iteratively generate verification questions inquiring additional evidence, answer them with internal or external knowledge, and use that to refine the original response with the new corrections. These methods have been explored for encyclopedic generation, but less so for domains like news summarization. In this work, we investigate two state-of-the-art self-correcting systems by applying them to correct hallucinated summaries using evidence from three search engines. We analyze the results and provide insights into systems' performance, revealing interesting practical findings on the benefits of search engine snippets and few-shot prompts, as well as high alignment of G-Eval and human evaluation. 

---
# What Matters in LLM-generated Data: Diversity and Its Effect on Model Fine-Tuning 

**Authors**: Yuchang Zhu, Zhonghua zhen, Qunshu Lin, Haotong Wei, Xiaolong Sun, Zixuan Yu, Minghao Liu, Zibin Zheng, Liang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.19262)  

**Abstract**: With the remarkable generative capabilities of large language models (LLMs), using LLM-generated data to train downstream models has emerged as a promising approach to mitigate data scarcity in specific domains and reduce time-consuming annotations. However, recent studies have highlighted a critical issue: iterative training on self-generated data results in model collapse, where model performance degrades over time. Despite extensive research on the implications of LLM-generated data, these works often neglect the importance of data diversity, a key factor in data quality. In this work, we aim to understand the implications of the diversity of LLM-generated data on downstream model performance. Specifically, we explore how varying levels of diversity in LLM-generated data affect downstream model performance. Additionally, we investigate the performance of models trained on data that mixes different proportions of LLM-generated data, which we refer to as synthetic data. Our experimental results show that, with minimal distribution shift, moderately diverse LLM-generated data can enhance model performance in scenarios with insufficient labeled data, whereas highly diverse generated data has a negative impact. We hope our empirical findings will offer valuable guidance for future studies on LLMs as data generators. 

---
# Human-Aligned Faithfulness in Toxicity Explanations of LLMs 

**Authors**: Ramaravind K. Mothilal, Joanna Roy, Syed Ishtiaque Ahmed, Shion Guha  

**Link**: [PDF](https://arxiv.org/pdf/2506.19113)  

**Abstract**: The discourse around toxicity and LLMs in NLP largely revolves around detection tasks. This work shifts the focus to evaluating LLMs' reasoning about toxicity -- from their explanations that justify a stance -- to enhance their trustworthiness in downstream tasks. Despite extensive research on explainability, it is not straightforward to adopt existing methods to evaluate free-form toxicity explanation due to their over-reliance on input text perturbations, among other challenges. To account for these, we propose a novel, theoretically-grounded multi-dimensional criterion, Human-Aligned Faithfulness (HAF), that measures the extent to which LLMs' free-form toxicity explanations align with those of a rational human under ideal conditions. We develop six metrics, based on uncertainty quantification, to comprehensively evaluate \haf of LLMs' toxicity explanations with no human involvement, and highlight how "non-ideal" the explanations are. We conduct several experiments on three Llama models (of size up to 70B) and an 8B Ministral model on five diverse toxicity datasets. Our results show that while LLMs generate plausible explanations to simple prompts, their reasoning about toxicity breaks down when prompted about the nuanced relations between the complete set of reasons, the individual reasons, and their toxicity stances, resulting in inconsistent and nonsensical responses. We open-source our code and LLM-generated explanations at this https URL. 

---
# Broken Tokens? Your Language Model can Secretly Handle Non-Canonical Tokenizations 

**Authors**: Brian Siyuan Zheng, Alisa Liu, Orevaoghene Ahia, Jonathan Hayase, Yejin Choi, Noah A. Smith  

**Link**: [PDF](https://arxiv.org/pdf/2506.19004)  

**Abstract**: Modern tokenizers employ deterministic algorithms to map text into a single "canonical" token sequence, yet the same string can be encoded as many non-canonical tokenizations using the tokenizer vocabulary. In this work, we investigate the robustness of LMs to text encoded with non-canonical tokenizations entirely unseen during training. Surprisingly, when evaluated across 20 benchmarks, we find that instruction-tuned models retain up to 93.4% of their original performance when given a randomly sampled tokenization, and 90.8% with character-level tokenization. We see that overall stronger models tend to be more robust, and robustness diminishes as the tokenization departs farther from the canonical form. Motivated by these results, we then identify settings where non-canonical tokenization schemes can *improve* performance, finding that character-level segmentation improves string manipulation and code understanding tasks by up to +14%, and right-aligned digit grouping enhances large-number arithmetic by +33%. Finally, we investigate the source of this robustness, finding that it arises in the instruction-tuning phase. We show that while both base and post-trained models grasp the semantics of non-canonical tokenizations (perceiving them as containing misspellings), base models try to mimic the imagined mistakes and degenerate into nonsensical output, while post-trained models are committed to fluent responses. Overall, our findings suggest that models are less tied to their tokenizer than previously believed, and demonstrate the promise of intervening on tokenization at inference time to boost performance. 

---
# MFTCXplain: A Multilingual Benchmark Dataset for Evaluating the Moral Reasoning of LLMs through Hate Speech Multi-hop Explanation 

**Authors**: Jackson Trager, Francielle Vargas, Diego Alves, Matteo Guida, Mikel K. Ngueajio, Ameeta Agrawal, Flor Plaza-del-Arco, Yalda Daryanai, Farzan Karimi-Malekabadi  

**Link**: [PDF](https://arxiv.org/pdf/2506.19073)  

**Abstract**: Ensuring the moral reasoning capabilities of Large Language Models (LLMs) is a growing concern as these systems are used in socially sensitive tasks. Nevertheless, current evaluation benchmarks present two major shortcomings: a lack of annotations that justify moral classifications, which limits transparency and interpretability; and a predominant focus on English, which constrains the assessment of moral reasoning across diverse cultural settings. In this paper, we introduce MFTCXplain, a multilingual benchmark dataset for evaluating the moral reasoning of LLMs via hate speech multi-hop explanation using Moral Foundation Theory (MFT). The dataset comprises 3,000 tweets across Portuguese, Italian, Persian, and English, annotated with binary hate speech labels, moral categories, and text span-level rationales. Empirical results highlight a misalignment between LLM outputs and human annotations in moral reasoning tasks. While LLMs perform well in hate speech detection (F1 up to 0.836), their ability to predict moral sentiments is notably weak (F1 < 0.35). Furthermore, rationale alignment remains limited mainly in underrepresented languages. These findings show the limited capacity of current LLMs to internalize and reflect human moral reasoning. 

---
# Mirage of Mastery: Memorization Tricks LLMs into Artificially Inflated Self-Knowledge 

**Authors**: Sahil Kale, Vijaykant Nadadur  

**Link**: [PDF](https://arxiv.org/pdf/2506.18998)  

**Abstract**: When artificial intelligence mistakes memorization for intelligence, it creates a dangerous mirage of reasoning. Existing studies treat memorization and self-knowledge deficits in LLMs as separate issues and do not recognize an intertwining link that degrades the trustworthiness of LLM responses. In our study, we utilize a novel framework to ascertain if LLMs genuinely learn reasoning patterns from training data or merely memorize them to assume competence across problems of similar complexity focused on STEM domains. Our analysis shows a noteworthy problem in generalization: LLMs draw confidence from memorized solutions to infer a higher self-knowledge about their reasoning ability, which manifests as an over 45% inconsistency in feasibility assessments when faced with self-validated, logically coherent task perturbations. This effect is most pronounced in science and medicine domains, which tend to have maximal standardized jargon and problems, further confirming our approach. Significant wavering within the self-knowledge of LLMs also shows flaws in current architectures and training patterns, highlighting the need for techniques that ensure a balanced, consistent stance on models' perceptions of their own knowledge for maximum AI explainability and trustworthiness. Our code and results are available publicly at this https URL. 

---
# Augmenting Multi-Agent Communication with State Delta Trajectory 

**Authors**: Yichen Tang, Weihang Su, Yujia Zhou, Yiqun Liu, Min Zhang, Shaoping Ma, Qingyao Ai  

**Link**: [PDF](https://arxiv.org/pdf/2506.19209)  

**Abstract**: Multi-agent techniques such as role playing or multi-turn debates have been shown to be effective in improving the performance of large language models (LLMs) in downstream tasks. Despite their differences in workflows, existing LLM-based multi-agent systems mostly use natural language for agent communication. While this is appealing for its simplicity and interpretability, it also introduces inevitable information loss as one model must down sample its continuous state vectors to concrete tokens before transferring them to the other model. Such losses are particularly significant when the information to transfer is not simple facts, but reasoning logics or abstractive thoughts. To tackle this problem, we propose a new communication protocol that transfers both natural language tokens and token-wise state transition trajectory from one agent to another. Particularly, compared to the actual state value, we find that the sequence of state changes in LLMs after generating each token can better reflect the information hidden behind the inference process, so we propose a State Delta Encoding (SDE) method to represent state transition trajectories. The experimental results show that multi-agent systems with SDE achieve SOTA performance compared to other communication protocols, particularly in tasks that involve complex reasoning. This shows the potential of communication augmentation for LLM-based multi-agent systems. 

---
# LLM-Based Social Simulations Require a Boundary 

**Authors**: Zengqing Wu, Run Peng, Takayuki Ito, Chuan Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2506.19806)  

**Abstract**: This position paper argues that large language model (LLM)-based social simulations should establish clear boundaries to meaningfully contribute to social science research. While LLMs offer promising capabilities for modeling human-like agents compared to traditional agent-based modeling, they face fundamental limitations that constrain their reliability for social pattern discovery. The core issue lies in LLMs' tendency towards an ``average persona'' that lacks sufficient behavioral heterogeneity, a critical requirement for simulating complex social dynamics. We examine three key boundary problems: alignment (simulated behaviors matching real-world patterns), consistency (maintaining coherent agent behavior over time), and robustness (reproducibility under varying conditions). We propose heuristic boundaries for determining when LLM-based simulations can reliably advance social science understanding. We believe that these simulations are more valuable when focusing on (1) collective patterns rather than individual trajectories, (2) agent behaviors aligning with real population averages despite limited variance, and (3) proper validation methods available for testing simulation robustness. We provide a practical checklist to guide researchers in determining the appropriate scope and claims for LLM-based social simulations. 

---
# Recurrent Visual Feature Extraction and Stereo Attentions for CT Report Generation 

**Authors**: Yuanhe Tian, Lei Mao, Yan Song  

**Link**: [PDF](https://arxiv.org/pdf/2506.19665)  

**Abstract**: Generating reports for computed tomography (CT) images is a challenging task, while similar to existing studies for medical image report generation, yet has its unique characteristics, such as spatial encoding of multiple images, alignment between image volume and texts, etc. Existing solutions typically use general 2D or 3D image processing techniques to extract features from a CT volume, where they firstly compress the volume and then divide the compressed CT slices into patches for visual encoding. These approaches do not explicitly account for the transformations among CT slices, nor do they effectively integrate multi-level image features, particularly those containing specific organ lesions, to instruct CT report generation (CTRG). In considering the strong correlation among consecutive slices in CT scans, in this paper, we propose a large language model (LLM) based CTRG method with recurrent visual feature extraction and stereo attentions for hierarchical feature modeling. Specifically, we use a vision Transformer to recurrently process each slice in a CT volume, and employ a set of attentions over the encoded slices from different perspectives to selectively obtain important visual information and align them with textual features, so as to better instruct an LLM for CTRG. Experiment results and further analysis on the benchmark M3D-Cap dataset show that our method outperforms strong baseline models and achieves state-of-the-art results, demonstrating its validity and effectiveness. 

---
# Mix-of-Language-Experts Architecture for Multilingual Programming 

**Authors**: Yifan Zong, Yuntian Deng, Pengyu Nie  

**Link**: [PDF](https://arxiv.org/pdf/2506.18923)  

**Abstract**: Large language models (LLMs) have demonstrated impressive capabilities in aiding developers with tasks like code comprehension, generation, and translation. Supporting multilingual programming -- i.e., coding tasks across multiple programming languages -- typically requires either (1) finetuning a single LLM across all programming languages, which is cost-efficient but sacrifices language-specific specialization and performance, or (2) finetuning separate LLMs for each programming language, which allows for specialization but is computationally expensive and storage-intensive due to the duplication of parameters. This paper introduces MoLE (Mix-of-Language-Experts), a novel architecture that balances efficiency and specialization for multilingual programming. MoLE is composed of a base model, a shared LoRA (low-rank adaptation) module, and a collection of language-specific LoRA modules. These modules are jointly optimized during the finetuning process, enabling effective knowledge sharing and specialization across programming languages. During inference, MoLE automatically routes to the language-specific LoRA module corresponding to the programming language of the code token being generated. Our experiments demonstrate that MoLE achieves greater parameter efficiency compared to training separate language-specific LoRAs, while outperforming a single shared LLM finetuned for all programming languages in terms of accuracy. 

---
# From Web Search towards Agentic Deep Research: Incentivizing Search with Reasoning Agents 

**Authors**: Weizhi Zhang, Yangning Li, Yuanchen Bei, Junyu Luo, Guancheng Wan, Liangwei Yang, Chenxuan Xie, Yuyao Yang, Wei-Chieh Huang, Chunyu Miao, Henry Peng Zou, Xiao Luo, Yusheng Zhao, Yankai Chen, Chunkit Chan, Peilin Zhou, Xinyang Zhang, Chenwei Zhang, Jingbo Shang, Ming Zhang, Yangqiu Song, Irwin King, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.18959)  

**Abstract**: Information retrieval is a cornerstone of modern knowledge acquisition, enabling billions of queries each day across diverse domains. However, traditional keyword-based search engines are increasingly inadequate for handling complex, multi-step information needs. Our position is that Large Language Models (LLMs), endowed with reasoning and agentic capabilities, are ushering in a new paradigm termed Agentic Deep Research. These systems transcend conventional information search techniques by tightly integrating autonomous reasoning, iterative retrieval, and information synthesis into a dynamic feedback loop. We trace the evolution from static web search to interactive, agent-based systems that plan, explore, and learn. We also introduce a test-time scaling law to formalize the impact of computational depth on reasoning and search. Supported by benchmark results and the rise of open-source implementations, we demonstrate that Agentic Deep Research not only significantly outperforms existing approaches, but is also poised to become the dominant paradigm for future information seeking. All the related resources, including industry products, research papers, benchmark datasets, and open-source implementations, are collected for the community in this https URL. 

---
# KnowML: Improving Generalization of ML-NIDS with Attack Knowledge Graphs 

**Authors**: Xin Fan Guo, Albert Merono Penuela, Sergio Maffeis, Fabio Pierazzi  

**Link**: [PDF](https://arxiv.org/pdf/2506.19802)  

**Abstract**: Despite extensive research on Machine Learning-based Network Intrusion Detection Systems (ML-NIDS), their capability to detect diverse attack variants remains uncertain. Prior studies have largely relied on homogeneous datasets, which artificially inflate performance scores and offer a false sense of security. Designing systems that can effectively detect a wide range of attack variants remains a significant challenge. The progress of ML-NIDS continues to depend heavily on human expertise, which can embed subjective judgments of system designers into the model, potentially hindering its ability to generalize across diverse attack types.
To address this gap, we propose KnowML, a framework for knowledge-guided machine learning that integrates attack knowledge into ML-NIDS. KnowML systematically explores the threat landscape by leveraging Large Language Models (LLMs) to perform automated analysis of attack implementations. It constructs a unified Knowledge Graph (KG) of attack strategies, on which it applies symbolic reasoning to generate KG-Augmented Input, embedding domain knowledge directly into the design process of ML-NIDS.
We evaluate KnowML on 28 realistic attack variants, of which 10 are newly collected for this study. Our findings reveal that baseline ML-NIDS models fail to detect several variants entirely, achieving F1 scores as low as 0 %. In contrast, our knowledge-guided approach achieves up to 99 % F1 score while maintaining a False Positive Rate below 0.1 %. 

---
