# Language Models use Lookbacks to Track Beliefs 

**Authors**: Nikhil Prakash, Natalie Shapira, Arnab Sen Sharma, Christoph Riedl, Yonatan Belinkov, Tamar Rott Shaham, David Bau, Atticus Geiger  

**Link**: [PDF](https://arxiv.org/pdf/2505.14685)  

**Abstract**: How do language models (LMs) represent characters' beliefs, especially when those beliefs may differ from reality? This question lies at the heart of understanding the Theory of Mind (ToM) capabilities of LMs. We analyze Llama-3-70B-Instruct's ability to reason about characters' beliefs using causal mediation and abstraction. We construct a dataset that consists of simple stories where two characters each separately change the state of two objects, potentially unaware of each other's actions. Our investigation uncovered a pervasive algorithmic pattern that we call a lookback mechanism, which enables the LM to recall important information when it becomes necessary. The LM binds each character-object-state triple together by co-locating reference information about them, represented as their Ordering IDs (OIs) in low rank subspaces of the state token's residual stream. When asked about a character's beliefs regarding the state of an object, the binding lookback retrieves the corresponding state OI and then an answer lookback retrieves the state token. When we introduce text specifying that one character is (not) visible to the other, we find that the LM first generates a visibility ID encoding the relation between the observing and the observed character OIs. In a visibility lookback, this ID is used to retrieve information about the observed character and update the observing character's beliefs. Our work provides insights into the LM's belief tracking mechanisms, taking a step toward reverse-engineering ToM reasoning in LMs. 

---
# Mind the Gap: Bridging Thought Leap for Improved Chain-of-Thought Tuning 

**Authors**: Haolei Xu, Yuchen Yan, Yongliang Shen, Wenqi Zhang, Guiyang Hou, Shengpei Jiang, Kaitao Song, Weiming Lu, Jun Xiao, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14684)  

**Abstract**: Large language models (LLMs) have achieved remarkable progress on mathemati-cal tasks through Chain-of-Thought (CoT) reasoning. However, existing mathematical CoT datasets often suffer from Thought Leaps due to experts omitting intermediate steps, which negatively impacts model learning and generalization. We propose the CoT Thought Leap Bridge Task, which aims to automatically detect leaps and generate missing intermediate reasoning steps to restore the completeness and coherence of CoT. To facilitate this, we constructed a specialized training dataset called ScaleQM+, based on the structured ScaleQuestMath dataset, and trained CoT-Bridge to bridge thought leaps. Through comprehensive experiments on mathematical reasoning benchmarks, we demonstrate that models fine-tuned on bridged datasets consistently outperform those trained on original datasets, with improvements of up to +5.87% on NuminaMath. Our approach effectively enhances distilled data (+3.02%) and provides better starting points for reinforcement learning (+3.1%), functioning as a plug-and-play module compatible with existing optimization techniques. Furthermore, CoT-Bridge demonstrate improved generalization to out-of-domain logical reasoning tasks, confirming that enhancing reasoning completeness yields broadly applicable benefits. 

---
# UltraEdit: Training-, Subject-, and Memory-Free Lifelong Editing in Large Language Models 

**Authors**: Xiaojie Gu, Guangxu Chen, Jungang Li, Jia-Chen Gu, Xuming Hu, Kai Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14679)  

**Abstract**: Lifelong learning enables large language models (LLMs) to adapt to evolving information by continually updating their internal knowledge. An ideal system should support efficient, wide-ranging updates while preserving existing capabilities and ensuring reliable deployment. Model editing stands out as a promising solution for this goal, offering a focused and efficient way to revise a model's internal knowledge. Although recent paradigms have made notable progress, they often struggle to meet the demands of practical lifelong adaptation at scale. To bridge this gap, we propose ULTRAEDIT-a fundamentally new editing solution that is training-, subject- and memory-free, making it particularly well-suited for ultra-scalable, real-world lifelong model editing. ULTRAEDIT performs editing through a self-contained process that relies solely on lightweight linear algebra operations to compute parameter shifts, enabling fast and consistent parameter modifications with minimal overhead. To improve scalability in lifelong settings, ULTRAEDIT employs a lifelong normalization strategy that continuously updates feature statistics across turns, allowing it to adapt to distributional shifts and maintain consistency over time. ULTRAEDIT achieves editing speeds over 7x faster than the previous state-of-the-art method-which was also the fastest known approach-while consuming less than 1/3 the VRAM, making it the only method currently capable of editing a 7B LLM on a 24GB consumer-grade GPU. Furthermore, we construct ULTRAEDITBENCH-the largest dataset in the field to date, with over 2M editing pairs-and demonstrate that our method supports up to 1M edits while maintaining high accuracy. Comprehensive experiments on four datasets and six models show that ULTRAEDIT consistently achieves superior performance across diverse model editing scenarios. Our code is available at: this https URL. 

---
# Reward Reasoning Model 

**Authors**: Jiaxin Guo, Zewen Chi, Li Dong, Qingxiu Dong, Xun Wu, Shaohan Huang, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2505.14674)  

**Abstract**: Reward models play a critical role in guiding large language models toward outputs that align with human expectations. However, an open challenge remains in effectively utilizing test-time compute to enhance reward model performance. In this work, we introduce Reward Reasoning Models (RRMs), which are specifically designed to execute a deliberate reasoning process before generating final rewards. Through chain-of-thought reasoning, RRMs leverage additional test-time compute for complex queries where appropriate rewards are not immediately apparent. To develop RRMs, we implement a reinforcement learning framework that fosters self-evolved reward reasoning capabilities without requiring explicit reasoning traces as training data. Experimental results demonstrate that RRMs achieve superior performance on reward modeling benchmarks across diverse domains. Notably, we show that RRMs can adaptively exploit test-time compute to further improve reward accuracy. The pretrained reward reasoning models are available at this https URL. 

---
# EmoGist: Efficient In-Context Learning for Visual Emotion Understanding 

**Authors**: Ronald Seoh, Dan Goldwasser  

**Link**: [PDF](https://arxiv.org/pdf/2505.14660)  

**Abstract**: In this paper, we introduce EmoGist, a training-free, in-context learning method for performing visual emotion classification with LVLMs. The key intuition of our approach is that context-dependent definition of emotion labels could allow more accurate predictions of emotions, as the ways in which emotions manifest within images are highly context dependent and nuanced. EmoGist pre-generates multiple explanations of emotion labels, by analyzing the clusters of example images belonging to each category. At test time, we retrieve a version of explanation based on embedding similarity, and feed it to a fast VLM for classification. Through our experiments, we show that EmoGist allows up to 13 points improvement in micro F1 scores with the multi-label Memotion dataset, and up to 8 points in macro F1 in the multi-class FI dataset. 

---
# General-Reasoner: Advancing LLM Reasoning Across All Domains 

**Authors**: Xueguang Ma, Qian Liu, Dongfu Jiang, Ge Zhang, Zejun Ma, Wenhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.14652)  

**Abstract**: Reinforcement learning (RL) has recently demonstrated strong potential in enhancing the reasoning capabilities of large language models (LLMs). Particularly, the "Zero" reinforcement learning introduced by Deepseek-R1-Zero, enables direct RL training of base LLMs without relying on an intermediate supervised fine-tuning stage. Despite these advancements, current works for LLM reasoning mainly focus on mathematical and coding domains, largely due to data abundance and the ease of answer verification. This limits the applicability and generalization of such models to broader domains, where questions often have diverse answer representations, and data is more scarce. In this paper, we propose General-Reasoner, a novel training paradigm designed to enhance LLM reasoning capabilities across diverse domains. Our key contributions include: (1) constructing a large-scale, high-quality dataset of questions with verifiable answers curated by web crawling, covering a wide range of disciplines; and (2) developing a generative model-based answer verifier, which replaces traditional rule-based verification with the capability of chain-of-thought and context-awareness. We train a series of models and evaluate them on a wide range of datasets covering wide domains like physics, chemistry, finance, electronics etc. Our comprehensive evaluation across these 12 benchmarks (e.g. MMLU-Pro, GPQA, SuperGPQA, TheoremQA, BBEH and MATH AMC) demonstrates that General-Reasoner outperforms existing baseline methods, achieving robust and generalizable reasoning performance while maintaining superior effectiveness in mathematical reasoning tasks. 

---
# Will AI Tell Lies to Save Sick Children? Litmus-Testing AI Values Prioritization with AIRiskDilemmas 

**Authors**: Yu Ying Chiu, Zhilin Wang, Sharan Maiya, Yejin Choi, Kyle Fish, Sydney Levine, Evan Hubinger  

**Link**: [PDF](https://arxiv.org/pdf/2505.14633)  

**Abstract**: Detecting AI risks becomes more challenging as stronger models emerge and find novel methods such as Alignment Faking to circumvent these detection attempts. Inspired by how risky behaviors in humans (i.e., illegal activities that may hurt others) are sometimes guided by strongly-held values, we believe that identifying values within AI models can be an early warning system for AI's risky behaviors. We create LitmusValues, an evaluation pipeline to reveal AI models' priorities on a range of AI value classes. Then, we collect AIRiskDilemmas, a diverse collection of dilemmas that pit values against one another in scenarios relevant to AI safety risks such as Power Seeking. By measuring an AI model's value prioritization using its aggregate choices, we obtain a self-consistent set of predicted value priorities that uncover potential risks. We show that values in LitmusValues (including seemingly innocuous ones like Care) can predict for both seen risky behaviors in AIRiskDilemmas and unseen risky behaviors in HarmBench. 

---
# Think Only When You Need with Large Hybrid-Reasoning Models 

**Authors**: Lingjie Jiang, Xun Wu, Shaohan Huang, Qingxiu Dong, Zewen Chi, Li Dong, Xingxing Zhang, Tengchao Lv, Lei Cui, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2505.14631)  

**Abstract**: Recent Large Reasoning Models (LRMs) have shown substantially improved reasoning capabilities over traditional Large Language Models (LLMs) by incorporating extended thinking processes prior to producing final responses. However, excessively lengthy thinking introduces substantial overhead in terms of token consumption and latency, which is particularly unnecessary for simple queries. In this work, we introduce Large Hybrid-Reasoning Models (LHRMs), the first kind of model capable of adaptively determining whether to perform thinking based on the contextual information of user queries. To achieve this, we propose a two-stage training pipeline comprising Hybrid Fine-Tuning (HFT) as a cold start, followed by online reinforcement learning with the proposed Hybrid Group Policy Optimization (HGPO) to implicitly learn to select the appropriate thinking mode. Furthermore, we introduce a metric called Hybrid Accuracy to quantitatively assess the model's capability for hybrid thinking. Extensive experimental results show that LHRMs can adaptively perform hybrid thinking on queries of varying difficulty and type. It outperforms existing LRMs and LLMs in reasoning and general capabilities while significantly improving efficiency. Together, our work advocates for a reconsideration of the appropriate use of extended thinking processes and provides a solid starting point for building hybrid thinking systems. 

---
# Linear Control of Test Awareness Reveals Differential Compliance in Reasoning Models 

**Authors**: Sahar Abdelnabi, Ahmed Salem  

**Link**: [PDF](https://arxiv.org/pdf/2505.14617)  

**Abstract**: Reasoning-focused large language models (LLMs) sometimes alter their behavior when they detect that they are being evaluated, an effect analogous to the Hawthorne phenomenon, which can lead them to optimize for test-passing performance or to comply more readily with harmful prompts if real-world consequences appear absent. We present the first quantitative study of how such "test awareness" impacts model behavior, particularly its safety alignment. We introduce a white-box probing framework that (i) linearly identifies awareness-related activations and (ii) steers models toward or away from test awareness while monitoring downstream performance. We apply our method to different state-of-the-art open-source reasoning LLMs across both realistic and hypothetical tasks. Our results demonstrate that test awareness significantly impact safety alignment, and is different for different models. By providing fine-grained control over this latent effect, our work aims to increase trust in how we perform safety evaluation. 

---
# Language Models Optimized to Fool Detectors Still Have a Distinct Style (And How to Change It) 

**Authors**: Rafael Rivera Soto, Barry Chen, Nicholas Andrews  

**Link**: [PDF](https://arxiv.org/pdf/2505.14608)  

**Abstract**: Despite considerable progress in the development of machine-text detectors, it has been suggested that the problem is inherently hard, and therefore, that stakeholders should proceed under the assumption that machine-generated text cannot be reliably detected as such. We examine a recent such claim by Nicks et al. (2024) regarding the ease with which language models can be optimized to degrade the performance of machine-text detectors, including detectors not specifically optimized against. We identify a feature space$\unicode{x2013}$the stylistic feature space$\unicode{x2013}$that is robust to such optimization, and show that it may be used to reliably detect samples from language models optimized to prevent detection. Furthermore, we show that even when models are explicitly optimized against stylistic detectors, detection performance remains surprisingly unaffected. We then seek to understand if stylistic detectors are inherently more robust. To study this question, we explore a new paraphrasing approach that simultaneously aims to close the gap between human writing and machine writing in stylistic feature space while avoiding detection using traditional features. We show that when only a single sample is available for detection, this attack is universally effective across all detectors considered, including those that use writing style. However, as the number of samples available for detection grows, the human and machine distributions become distinguishable. This observation encourages us to introduce AURA, a metric that estimates the overlap between human and machine-generated distributions by analyzing how detector performance improves as more samples become available. Overall, our findings underscore previous recommendations to avoid reliance on machine-text detection. 

---
# sudoLLM : On Multi-role Alignment of Language Models 

**Authors**: Soumadeep Saha, Akshay Chaturvedi, Joy Mahapatra, Utpal Garain  

**Link**: [PDF](https://arxiv.org/pdf/2505.14607)  

**Abstract**: User authorization-based access privileges are a key feature in many safety-critical systems, but have thus far been absent from the large language model (LLM) realm. In this work, drawing inspiration from such access control systems, we introduce sudoLLM, a novel framework that results in multi-role aligned LLMs, i.e., LLMs that account for, and behave in accordance with, user access rights. sudoLLM injects subtle user-based biases into queries and trains an LLM to utilize this bias signal in order to produce sensitive information if and only if the user is authorized. We present empirical results demonstrating that this approach shows substantially improved alignment, generalization, and resistance to prompt-based jailbreaking attacks. The persistent tension between the language modeling objective and safety alignment, which is often exploited to jailbreak LLMs, is somewhat resolved with the aid of the injected bias signal. Our framework is meant as an additional security layer, and complements existing guardrail mechanisms for enhanced end-to-end safety with LLMs. 

---
# Toward Reliable Biomedical Hypothesis Generation: Evaluating Truthfulness and Hallucination in Large Language Models 

**Authors**: Guangzhi Xiong, Eric Xie, Corey Williams, Myles Kim, Amir Hassan Shariatmadari, Sikun Guo, Stefan Bekiranov, Aidong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14599)  

**Abstract**: Large language models (LLMs) have shown significant potential in scientific disciplines such as biomedicine, particularly in hypothesis generation, where they can analyze vast literature, identify patterns, and suggest research directions. However, a key challenge lies in evaluating the truthfulness of generated hypotheses, as verifying their accuracy often requires substantial time and resources. Additionally, the hallucination problem in LLMs can lead to the generation of hypotheses that appear plausible but are ultimately incorrect, undermining their reliability. To facilitate the systematic study of these challenges, we introduce TruthHypo, a benchmark for assessing the capabilities of LLMs in generating truthful biomedical hypotheses, and KnowHD, a knowledge-based hallucination detector to evaluate how well hypotheses are grounded in existing knowledge. Our results show that LLMs struggle to generate truthful hypotheses. By analyzing hallucinations in reasoning steps, we demonstrate that the groundedness scores provided by KnowHD serve as an effective metric for filtering truthful hypotheses from the diverse outputs of LLMs. Human evaluations further validate the utility of KnowHD in identifying truthful hypotheses and accelerating scientific discovery. Our data and source code are available at this https URL. 

---
# Success is in the Details: Evaluate and Enhance Details Sensitivity of Code LLMs through Counterfactuals 

**Authors**: Xianzhen Luo, Qingfu Zhu, Zhiming Zhang, Mingzheng Xu, Tianhao Cheng, Yixuan Wang, Zheng Chu, Shijie Xuyang, Zhiyuan Ma, YuanTao Fan, Wanxiang Che  

**Link**: [PDF](https://arxiv.org/pdf/2505.14597)  

**Abstract**: Code Sensitivity refers to the ability of Code LLMs to recognize and respond to details changes in problem descriptions. While current code benchmarks and instruction data focus on difficulty and diversity, sensitivity is overlooked. We first introduce the CTF-Code benchmark, constructed using counterfactual perturbations, minimizing input changes while maximizing output changes. The evaluation shows that many LLMs have a more than 10\% performance drop compared to the original problems. To fully utilize sensitivity, CTF-Instruct, an incremental instruction fine-tuning framework, extends on existing data and uses a selection mechanism to meet the three dimensions of difficulty, diversity, and sensitivity. Experiments show that LLMs fine-tuned with CTF-Instruct data achieve over a 2\% improvement on CTF-Code, and more than a 10\% performance boost on LiveCodeBench, validating the feasibility of enhancing LLMs' sensitivity to improve performance. 

---
# MCIP: Protecting MCP Safety via Model Contextual Integrity Protocol 

**Authors**: Huihao Jing, Haoran Li, Wenbin Hu, Qi Hu, Heli Xu, Tianshu Chu, Peizhao Hu, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.14590)  

**Abstract**: As Model Context Protocol (MCP) introduces an easy-to-use ecosystem for users and developers, it also brings underexplored safety risks. Its decentralized architecture, which separates clients and servers, poses unique challenges for systematic safety analysis. This paper proposes a novel framework to enhance MCP safety. Guided by the MAESTRO framework, we first analyze the missing safety mechanisms in MCP, and based on this analysis, we propose the Model Contextual Integrity Protocol (MCIP), a refined version of MCP that addresses these this http URL, we develop a fine-grained taxonomy that captures a diverse range of unsafe behaviors observed in MCP scenarios. Building on this taxonomy, we develop benchmark and training data that support the evaluation and improvement of LLMs' capabilities in identifying safety risks within MCP interactions. Leveraging the proposed benchmark and training data, we conduct extensive experiments on state-of-the-art LLMs. The results highlight LLMs' vulnerabilities in MCP interactions and demonstrate that our approach substantially improves their safety performance. 

---
# Context Reasoner: Incentivizing Reasoning Capability for Contextualized Privacy and Safety Compliance via Reinforcement Learning 

**Authors**: Wenbin Hu, Haoran Li, Huihao Jing, Qi Hu, Ziqian Zeng, Sirui Han, Heli Xu, Tianshu Chu, Peizhao Hu, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.14585)  

**Abstract**: While Large Language Models (LLMs) exhibit remarkable capabilities, they also introduce significant safety and privacy risks. Current mitigation strategies often fail to preserve contextual reasoning capabilities in risky scenarios. Instead, they rely heavily on sensitive pattern matching to protect LLMs, which limits the scope. Furthermore, they overlook established safety and privacy standards, leading to systemic risks for legal compliance. To address these gaps, we formulate safety and privacy issues into contextualized compliance problems following the Contextual Integrity (CI) theory. Under the CI framework, we align our model with three critical regulatory standards: GDPR, EU AI Act, and HIPAA. Specifically, we employ reinforcement learning (RL) with a rule-based reward to incentivize contextual reasoning capabilities while enhancing compliance with safety and privacy norms. Through extensive experiments, we demonstrate that our method not only significantly enhances legal compliance (achieving a +17.64% accuracy improvement in safety/privacy benchmarks) but also further improves general reasoning capability. For OpenThinker-7B, a strong reasoning model that significantly outperforms its base model Qwen2.5-7B-Instruct across diverse subjects, our method enhances its general reasoning capabilities, with +2.05% and +8.98% accuracy improvement on the MMLU and LegalBench benchmark, respectively. 

---
# Can Pruning Improve Reasoning? Revisiting Long-CoT Compression with Capability in Mind for Better Reasoning 

**Authors**: Shangziqi Zhao, Jiahao Yuan, Guisong Yang, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2505.14582)  

**Abstract**: Long chain-of-thought (Long-CoT) reasoning improves accuracy in LLMs, yet its verbose, self-reflective style often hinders effective distillation into small language models (SLMs). We revisit Long-CoT compression through the lens of capability alignment and ask: Can pruning improve reasoning? We propose Prune-on-Logic, a structure-aware framework that transforms Long-CoT into logic graphs and selectively prunes low-utility reasoning steps under self-verification constraints. Through systematic analysis across three pruning strategies -- targeting entire chains, core reasoning, and verification -- we find that pruning verification steps yields consistent accuracy gains while reducing inference cost, outperforming token-level baselines and uncompressed fine-tuning. In contrast, pruning reasoning or all-chain steps degrades performance, revealing that small models benefit not from shorter CoTs, but from semantically leaner ones. Our findings highlight pruning as a structural optimization strategy for aligning CoT reasoning with SLM capacity. 

---
# TRATES: Trait-Specific Rubric-Assisted Cross-Prompt Essay Scoring 

**Authors**: Sohaila Eltanbouly, Salam Albatarni, Tamer Elsayed  

**Link**: [PDF](https://arxiv.org/pdf/2505.14577)  

**Abstract**: Research on holistic Automated Essay Scoring (AES) is long-dated; yet, there is a notable lack of attention for assessing essays according to individual traits. In this work, we propose TRATES, a novel trait-specific and rubric-based cross-prompt AES framework that is generic yet specific to the underlying trait. The framework leverages a Large Language Model (LLM) that utilizes the trait grading rubrics to generate trait-specific features (represented by assessment questions), then assesses those features given an essay. The trait-specific features are eventually combined with generic writing-quality and prompt-specific features to train a simple classical regression model that predicts trait scores of essays from an unseen prompt. Experiments show that TRATES achieves a new state-of-the-art performance across all traits on a widely-used dataset, with the generated LLM-based features being the most significant. 

---
# Pivot Language for Low-Resource Machine Translation 

**Authors**: Abhimanyu Talwar, Julien Laasri  

**Link**: [PDF](https://arxiv.org/pdf/2505.14553)  

**Abstract**: Certain pairs of languages suffer from lack of a parallel corpus which is large in size and diverse in domain. One of the ways this is overcome is via use of a pivot language. In this paper we use Hindi as a pivot language to translate Nepali into English. We describe what makes Hindi a good candidate for the pivot. We discuss ways in which a pivot language can be used, and use two such approaches - the Transfer Method (fully supervised) and Backtranslation (semi-supervised) - to translate Nepali into English. Using the former, we are able to achieve a devtest Set SacreBLEU score of 14.2, which improves the baseline fully supervised score reported by (Guzman et al., 2019) by 6.6 points. While we are slightly below the semi-supervised baseline score of 15.1, we discuss what may have caused this under-performance, and suggest scope for future work. 

---
# KORGym: A Dynamic Game Platform for LLM Reasoning Evaluation 

**Authors**: Jiajun Shi, Jian Yang, Jiaheng Liu, Xingyuan Bu, Jiangjie Chen, Junting Zhou, Kaijing Ma, Zhoufutu Wen, Bingli Wang, Yancheng He, Liang Song, Hualei Zhu, Shilong Li, Xingjian Wang, Wei Zhang, Ruibin Yuan, Yifan Yao, Wenjun Yang, Yunli Wang, Siyuan Fang, Siyu Yuan, Qianyu He, Xiangru Tang, Yingshui Tan, Wangchunshu Zhou, Zhaoxiang Zhang, Zhoujun Li, Wenhao Huang, Ge Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14552)  

**Abstract**: Recent advancements in large language models (LLMs) underscore the need for more comprehensive evaluation methods to accurately assess their reasoning capabilities. Existing benchmarks are often domain-specific and thus cannot fully capture an LLM's general reasoning potential. To address this limitation, we introduce the Knowledge Orthogonal Reasoning Gymnasium (KORGym), a dynamic evaluation platform inspired by KOR-Bench and Gymnasium. KORGym offers over fifty games in either textual or visual formats and supports interactive, multi-turn assessments with reinforcement learning scenarios. Using KORGym, we conduct extensive experiments on 19 LLMs and 8 VLMs, revealing consistent reasoning patterns within model families and demonstrating the superior performance of closed-source models. Further analysis examines the effects of modality, reasoning strategies, reinforcement learning techniques, and response length on model performance. We expect KORGym to become a valuable resource for advancing LLM reasoning research and developing evaluation methodologies suited to complex, interactive environments. 

---
# Breaking Bad Tokens: Detoxification of LLMs Using Sparse Autoencoders 

**Authors**: Agam Goyal, Vedant Rathi, William Yeh, Yian Wang, Yuen Chen, Hari Sundaram  

**Link**: [PDF](https://arxiv.org/pdf/2505.14536)  

**Abstract**: Large language models (LLMs) are now ubiquitous in user-facing applications, yet they still generate undesirable toxic outputs, including profanity, vulgarity, and derogatory remarks. Although numerous detoxification methods exist, most apply broad, surface-level fixes and can therefore easily be circumvented by jailbreak attacks. In this paper we leverage sparse autoencoders (SAEs) to identify toxicity-related directions in the residual stream of models and perform targeted activation steering using the corresponding decoder vectors. We introduce three tiers of steering aggressiveness and evaluate them on GPT-2 Small and Gemma-2-2B, revealing trade-offs between toxicity reduction and language fluency. At stronger steering strengths, these causal interventions surpass competitive baselines in reducing toxicity by up to 20%, though fluency can degrade noticeably on GPT-2 Small depending on the aggressiveness. Crucially, standard NLP benchmark scores upon steering remain stable, indicating that the model's knowledge and general abilities are preserved. We further show that feature-splitting in wider SAEs hampers safety interventions, underscoring the importance of disentangled feature learning. Our findings highlight both the promise and the current limitations of SAE-based causal interventions for LLM detoxification, further suggesting practical guidelines for safer language-model deployment. 

---
# Internal Chain-of-Thought: Empirical Evidence for Layer-wise Subtask Scheduling in LLMs 

**Authors**: Zhipeng Yang, Junzhuo Li, Siyu Xia, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.14530)  

**Abstract**: We show that large language models (LLMs) exhibit an $\textit{internal chain-of-thought}$: they sequentially decompose and execute composite tasks layer-by-layer. Two claims ground our study: (i) distinct subtasks are learned at different network depths, and (ii) these subtasks are executed sequentially across layers. On a benchmark of 15 two-step composite tasks, we employ layer-from context-masking and propose a novel cross-task patching method, confirming (i). To examine claim (ii), we apply LogitLens to decode hidden states, revealing a consistent layerwise execution pattern. We further replicate our analysis on the real-world $\text{TRACE}$ benchmark, observing the same stepwise dynamics. Together, our results enhance LLMs transparency by showing their capacity to internally plan and execute subtasks (or instructions), opening avenues for fine-grained, instruction-level activation steering. 

---
# Exploring Graph Representations of Logical Forms for Language Modeling 

**Authors**: Michael Sullivan  

**Link**: [PDF](https://arxiv.org/pdf/2505.14523)  

**Abstract**: We make the case for language models over logical forms (LFLMs), arguing that such models are more data-efficient than their textual counterparts. To that end, we introduce the Graph-based Formal-Logical Distributional Semantics (GFoLDS) prototype, a pretrained LM over graph representations of logical forms, as a proof-of-concept of LFLMs. Using GFoLDS, we present strong experimental evidence that LFLMs can leverage the built-in, basic linguistic knowledge inherent in such models to immediately begin learning more complex patterns. On downstream tasks, we show that GFoLDS vastly outperforms textual, transformer LMs pretrained on similar amounts of data, indicating that LFLMs can learn with substantially less data than models over plain text. Furthermore, we show that the performance of this model is likely to scale with additional parameters and pretraining data, suggesting the viability of LFLMs in real-world applications. 

---
# ModRWKV: Transformer Multimodality in Linear Time 

**Authors**: Jiale Kang, Ziyin Yue, Qingyu Yin, Jiang Rui, Weile Li, Zening Lu, Zhouran Ji  

**Link**: [PDF](https://arxiv.org/pdf/2505.14505)  

**Abstract**: Currently, most multimodal studies are based on large language models (LLMs) with quadratic-complexity Transformer architectures. While linear models like RNNs enjoy low inference costs, their application has been largely limited to the text-only modality. This work explores the capabilities of modern RNN architectures in multimodal contexts. We propose ModRWKV-a decoupled multimodal framework built upon the RWKV7 architecture as its LLM backbone-which achieves multi-source information fusion through dynamically adaptable heterogeneous modality encoders. We designed the multimodal modules in ModRWKV with an extremely lightweight architecture and, through extensive experiments, identified a configuration that achieves an optimal balance between performance and computational efficiency. ModRWKV leverages the pretrained weights of the RWKV7 LLM for initialization, which significantly accelerates multimodal training. Comparative experiments with different pretrained checkpoints further demonstrate that such initialization plays a crucial role in enhancing the model's ability to understand multimodal signals. Supported by extensive experiments, we conclude that modern RNN architectures present a viable alternative to Transformers in the domain of multimodal large language models (MLLMs). Furthermore, we identify the optimal configuration of the ModRWKV architecture through systematic exploration. 

---
# Enhanced Multimodal Aspect-Based Sentiment Analysis by LLM-Generated Rationales 

**Authors**: Jun Cao, Jiyi Li, Ziwei Yang, Renjie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.14499)  

**Abstract**: There has been growing interest in Multimodal Aspect-Based Sentiment Analysis (MABSA) in recent years. Existing methods predominantly rely on pre-trained small language models (SLMs) to collect information related to aspects and sentiments from both image and text, with an aim to align these two modalities. However, small SLMs possess limited capacity and knowledge, often resulting in inaccurate identification of meaning, aspects, sentiments, and their interconnections in textual and visual data. On the other hand, Large language models (LLMs) have shown exceptional capabilities in various tasks by effectively exploring fine-grained information in multimodal data. However, some studies indicate that LLMs still fall short compared to fine-tuned small models in the field of ABSA. Based on these findings, we propose a novel framework, termed LRSA, which combines the decision-making capabilities of SLMs with additional information provided by LLMs for MABSA. Specifically, we inject explanations generated by LLMs as rationales into SLMs and employ a dual cross-attention mechanism for enhancing feature interaction and fusion, thereby augmenting the SLMs' ability to identify aspects and sentiments. We evaluated our method using two baseline models, numerous experiments highlight the superiority of our approach on three widely-used benchmarks, indicating its generalizability and applicability to most pre-trained models for MABSA. 

---
# MoMoE: Mixture of Moderation Experts Framework for AI-Assisted Online Governance 

**Authors**: Agam Goyal, Xianyang Zhan, Yilun Chen, Koustuv Saha, Eshwar Chandrasekharan  

**Link**: [PDF](https://arxiv.org/pdf/2505.14483)  

**Abstract**: Large language models (LLMs) have shown great potential in flagging harmful content in online communities. Yet, existing approaches for moderation require a separate model for every community and are opaque in their decision-making, limiting real-world adoption. We introduce Mixture of Moderation Experts (MoMoE), a modular, cross-community framework that adds post-hoc explanations to scalable content moderation. MoMoE orchestrates four operators -- Allocate, Predict, Aggregate, Explain -- and is instantiated as seven community-specialized experts (MoMoE-Community) and five norm-violation experts (MoMoE-NormVio). On 30 unseen subreddits, the best variants obtain Micro-F1 scores of 0.72 and 0.67, respectively, matching or surpassing strong fine-tuned baselines while consistently producing concise and reliable explanations. Although community-specialized experts deliver the highest peak accuracy, norm-violation experts provide steadier performance across domains. These findings show that MoMoE yields scalable, transparent moderation without needing per-community fine-tuning. More broadly, they suggest that lightweight, explainable expert ensembles can guide future NLP and HCI research on trustworthy human-AI governance of online communities. 

---
# PlanGPT-VL: Enhancing Urban Planning with Domain-Specific Vision-Language Models 

**Authors**: He Zhu, Junyou Su, Minxi Chen, Wen Wang, Yijie Deng, Guanhua Chen, Wenjia Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14481)  

**Abstract**: In the field of urban planning, existing Vision-Language Models (VLMs) frequently fail to effectively analyze and evaluate planning maps, despite the critical importance of these visual elements for urban planners and related educational contexts. Planning maps, which visualize land use, infrastructure layouts, and functional zoning, require specialized understanding of spatial configurations, regulatory requirements, and multi-scale analysis. To address this challenge, we introduce PlanGPT-VL, the first domain-specific Vision-Language Model tailored specifically for urban planning maps. PlanGPT-VL employs three innovative approaches: (1) PlanAnno-V framework for high-quality VQA data synthesis, (2) Critical Point Thinking to reduce hallucinations through structured verification, and (3) comprehensive training methodology combining Supervised Fine-Tuning with frozen vision encoder parameters. Through systematic evaluation on our proposed PlanBench-V benchmark, we demonstrate that PlanGPT-VL significantly outperforms general-purpose state-of-the-art VLMs in specialized planning map interpretation tasks, offering urban planning professionals a reliable tool for map analysis, assessment, and educational applications while maintaining high factual accuracy. Our lightweight 7B parameter model achieves comparable performance to models exceeding 72B parameters, demonstrating efficient domain specialization without sacrificing performance. 

---
# Adapting Pretrained Language Models for Citation Classification via Self-Supervised Contrastive Learning 

**Authors**: Tong Li, Jiachuan Wang, Yongqi Zhang, Shuangyin Li, Lei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.14471)  

**Abstract**: Citation classification, which identifies the intention behind academic citations, is pivotal for scholarly analysis. Previous works suggest fine-tuning pretrained language models (PLMs) on citation classification datasets, reaping the reward of the linguistic knowledge they gained during pretraining. However, directly fine-tuning for citation classification is challenging due to labeled data scarcity, contextual noise, and spurious keyphrase correlations. In this paper, we present a novel framework, Citss, that adapts the PLMs to overcome these challenges. Citss introduces self-supervised contrastive learning to alleviate data scarcity, and is equipped with two specialized strategies to obtain the contrastive pairs: sentence-level cropping, which enhances focus on target citations within long contexts, and keyphrase perturbation, which mitigates reliance on specific keyphrases. Compared with previous works that are only designed for encoder-based PLMs, Citss is carefully developed to be compatible with both encoder-based PLMs and decoder-based LLMs, to embrace the benefits of enlarged pretraining. Experiments with three benchmark datasets with both encoder-based PLMs and decoder-based LLMs demonstrate our superiority compared to the previous state of the art. Our code is available at: this http URL 

---
# Attributional Safety Failures in Large Language Models under Code-Mixed Perturbations 

**Authors**: Somnath Banerjee, Pratyush Chatterjee, Shanu Kumar, Sayan Layek, Parag Agrawal, Rima Hazra, Animesh Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2505.14469)  

**Abstract**: Recent advancements in LLMs have raised significant safety concerns, particularly when dealing with code-mixed inputs and outputs. Our study systematically investigates the increased susceptibility of LLMs to produce unsafe outputs from code-mixed prompts compared to monolingual English prompts. Utilizing explainability methods, we dissect the internal attribution shifts causing model's harmful behaviors. In addition, we explore cultural dimensions by distinguishing between universally unsafe and culturally-specific unsafe queries. This paper presents novel experimental insights, clarifying the mechanisms driving this phenomenon. 

---
# Void in Language Models 

**Authors**: Mani Shemiranifar  

**Link**: [PDF](https://arxiv.org/pdf/2505.14467)  

**Abstract**: Despite advances in transformer-based language models (LMs), a fundamental question remains largely unanswered: Are all layers activated during inference? We investigate this question by detecting unactivated layers (which we refer to as Voids) using a non-trainable and parameter-free adaptive computation method called L2 Adaptive Computation (LAC). We adapt LAC from its original efficiency-focused application to trace activated layers during inference. This method monitors changes in the L2-norm of activations to identify voids. We analyze layer activation in instruction-tuned LMs across two phases: Prompt Processing (PP), where we trace activated layers for each token in the input prompts, and Response Generation (RG), where we trace activated layers for each generated token. We further demonstrate that distinct layers are activated during these two phases. To show the effectiveness of our method, we evaluated three distinct instruction-tuned LMs from the Llama, Mistral, and Qwen families on three benchmarks: MMLU, GPQA Diamond, and BoolQ. For example, on MMLU with a zero-shot setting, skipping voids in Qwen2.5-7B-Instruct resulted in an improvement from 69.24 to 71.29 while the model uses only 30% of the layers. Similarly, Mistral-7B-Instruct-v0.3 on GPQA Diamond improved from 13.88 to 18.36 when using 70% of the layers during both the PP and RG phases. These results show that not all layers contribute equally during inference, and that selectively skipping most of them can improve the performance of models on certain tasks. 

---
# Not All Correct Answers Are Equal: Why Your Distillation Source Matters 

**Authors**: Xiaoyu Tian, Yunjie Ji, Haotian Wang, Shuaiting Chen, Sitong Zhao, Yiping Peng, Han Zhao, Xiangang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.14464)  

**Abstract**: Distillation has emerged as a practical and effective approach to enhance the reasoning capabilities of open-source language models. In this work, we conduct a large-scale empirical study on reasoning data distillation by collecting verified outputs from three state-of-the-art teacher models-AM-Thinking-v1, Qwen3-235B-A22B, and DeepSeek-R1-on a shared corpus of 1.89 million queries. We construct three parallel datasets and analyze their distributions, revealing that AM-Thinking-v1-distilled data exhibits greater token length diversity and lower perplexity. Student models trained on each dataset are evaluated on reasoning benchmarks including AIME2024, AIME2025, MATH500, and LiveCodeBench. The AM-based model consistently achieves the best performance (e.g., 84.3 on AIME2024, 72.2 on AIME2025, 98.4 on MATH500, and 65.9 on LiveCodeBench) and demonstrates adaptive output behavior-producing longer responses for harder tasks and shorter ones for simpler tasks. These findings highlight the value of high-quality, verified reasoning traces. We release the AM-Thinking-v1 and Qwen3-235B-A22B distilled datasets to support future research on open and high-performing reasoning-oriented language models. The datasets are publicly available on Hugging Face\footnote{Datasets are available on Hugging Face: \href{this https URL}{AM-Thinking-v1-Distilled}, \href{this https URL}{AM-Qwen3-Distilled}.}. 

---
# CtrlDiff: Boosting Large Diffusion Language Models with Dynamic Block Prediction and Controllable Generation 

**Authors**: Chihan Huang, Hao Tang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14455)  

**Abstract**: Although autoregressive models have dominated language modeling in recent years, there has been a growing interest in exploring alternative paradigms to the conventional next-token prediction framework. Diffusion-based language models have emerged as a compelling alternative due to their powerful parallel generation capabilities and inherent editability. However, these models are often constrained by fixed-length generation. A promising direction is to combine the strengths of both paradigms, segmenting sequences into blocks, modeling autoregressive dependencies across blocks while leveraging discrete diffusion to estimate the conditional distribution within each block given the preceding context. Nevertheless, their practical application is often hindered by two key limitations: rigid fixed-length outputs and a lack of flexible control mechanisms. In this work, we address the critical limitations of fixed granularity and weak controllability in current large diffusion language models. We propose CtrlDiff, a dynamic and controllable semi-autoregressive framework that adaptively determines the size of each generation block based on local semantics using reinforcement learning. Furthermore, we introduce a classifier-guided control mechanism tailored to discrete diffusion, which significantly reduces computational overhead while facilitating efficient post-hoc conditioning without retraining. Extensive experiments demonstrate that CtrlDiff sets a new standard among hybrid diffusion models, narrows the performance gap to state-of-the-art autoregressive approaches, and enables effective conditional text generation across diverse tasks. 

---
# Creative Preference Optimization 

**Authors**: Mete Ismayilzada, Antonio Laverghetta Jr., Simone A. Luchini, Reet Patel, Antoine Bosselut, Lonneke van der Plas, Roger Beaty  

**Link**: [PDF](https://arxiv.org/pdf/2505.14442)  

**Abstract**: While Large Language Models (LLMs) have demonstrated impressive performance across natural language generation tasks, their ability to generate truly creative content-characterized by novelty, diversity, surprise, and quality-remains limited. Existing methods for enhancing LLM creativity often focus narrowly on diversity or specific tasks, failing to address creativity's multifaceted nature in a generalizable way. In this work, we propose Creative Preference Optimization (CrPO), a novel alignment method that injects signals from multiple creativity dimensions into the preference optimization objective in a modular fashion. We train and evaluate creativity-augmented versions of several models using CrPO and MuCE, a new large-scale human preference dataset spanning over 200,000 human-generated responses and ratings from more than 30 psychological creativity assessments. Our models outperform strong baselines, including GPT-4o, on both automated and human evaluations, producing more novel, diverse, and surprising generations while maintaining high output quality. Additional evaluations on NoveltyBench further confirm the generalizability of our approach. Together, our results demonstrate that directly optimizing for creativity within preference frameworks is a promising direction for advancing the creative capabilities of LLMs without compromising output quality. 

---
# Neural Incompatibility: The Unbridgeable Gap of Cross-Scale Parametric Knowledge Transfer in Large Language Models 

**Authors**: Yuqiao Tan, Shizhu He, Kang Liu, Jun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.14436)  

**Abstract**: Large Language Models (LLMs) offer a transparent brain with accessible parameters that encode extensive knowledge, which can be analyzed, located and transferred. Consequently, a key research challenge is to transcend traditional knowledge transfer paradigms rooted in symbolic language and achieve genuine Parametric Knowledge Transfer (PKT). Significantly, exploring effective methods for transferring knowledge across LLMs of different scales through parameters presents an intriguing and valuable research direction. In this paper, we first demonstrate $\textbf{Alignment}$ in parametric space is the fundamental prerequisite to achieve successful cross-scale PKT. We redefine the previously explored knowledge transfer as Post-Align PKT (PostPKT), which utilizes extracted parameters for LoRA initialization and requires subsequent fine-tune for alignment. Hence, to reduce cost for further fine-tuning, we introduce a novel Pre-Align PKT (PrePKT) paradigm and propose a solution called $\textbf{LaTen}$ ($\textbf{L}$oc$\textbf{a}$te-$\textbf{T}$h$\textbf{e}$n-Alig$\textbf{n}$) that aligns the parametric spaces of LLMs across scales only using several training steps without following training. Comprehensive experiments on four benchmarks demonstrate that both PostPKT and PrePKT face challenges in achieving consistently stable transfer. Through in-depth analysis, we identify $\textbf{Neural Incompatibility}$ as the ethological and parametric structural differences between LLMs of varying scales, presenting fundamental challenges to achieving effective PKT. These findings provide fresh insights into the parametric architectures of LLMs and highlight promising directions for future research on efficient PKT. Our code is available at this https URL. 

---
# From Templates to Natural Language: Generalization Challenges in Instruction-Tuned LLMs for Spatial Reasoning 

**Authors**: Chalamalasetti Kranti, Sherzod Hakimov, David Schlangen  

**Link**: [PDF](https://arxiv.org/pdf/2505.14425)  

**Abstract**: Instruction-tuned large language models (LLMs) have shown strong performance on a variety of tasks; however, generalizing from synthetic to human-authored instructions in grounded environments remains a challenge for them. In this work, we study generalization challenges in spatial grounding tasks where models interpret and translate instructions for building object arrangements on a $2.5$D grid. We fine-tune LLMs using only synthetic instructions and evaluate their performance on a benchmark dataset containing both synthetic and human-written instructions. Our results reveal that while models generalize well on simple tasks, their performance degrades significantly on more complex tasks. We present a detailed error analysis of the gaps in instruction generalization. 

---
# Scaling Low-Resource MT via Synthetic Data Generation with LLMs 

**Authors**: Ona de Gibert, Joseph Attieh, Teemu Vahtola, Mikko Aulamo, Zihao Li, Raúl Vázquez, Tiancheng Hu, Jörg Tiedemann  

**Link**: [PDF](https://arxiv.org/pdf/2505.14423)  

**Abstract**: We investigate the potential of LLM-generated synthetic data for improving low-resource machine translation (MT). Focusing on seven diverse target languages, we construct a document-level synthetic corpus from English Europarl, and extend it via pivoting to 147 additional language pairs. Automatic and human evaluation confirm its high overall quality. We study its practical application by (i) identifying effective training regimes, (ii) comparing our data with the HPLT dataset, and (iii) testing its utility beyond English-centric MT. Finally, we introduce SynOPUS, a public repository for synthetic parallel datasets. Our findings show that LLM-generated synthetic data, even when noisy, can substantially improve MT performance for low-resource languages. 

---
# SAE-FiRE: Enhancing Earnings Surprise Predictions Through Sparse Autoencoder Feature Selection 

**Authors**: Huopu Zhang, Yanguang Liu, Mengnan Du  

**Link**: [PDF](https://arxiv.org/pdf/2505.14420)  

**Abstract**: Predicting earnings surprises through the analysis of earnings conference call transcripts has attracted increasing attention from the financial research community. Conference calls serve as critical communication channels between company executives, analysts, and shareholders, offering valuable forward-looking information. However, these transcripts present significant analytical challenges, typically containing over 5,000 words with substantial redundancy and industry-specific terminology that creates obstacles for language models. In this work, we propose the Sparse Autoencoder for Financial Representation Enhancement (SAE-FiRE) framework to address these limitations by extracting key information while eliminating redundancy. SAE-FiRE employs Sparse Autoencoders (SAEs) to efficiently identify patterns and filter out noises, and focusing specifically on capturing nuanced financial signals that have predictive power for earnings surprises. Experimental results indicate that the proposed method can significantly outperform comparing baselines. 

---
# Hidden Ghost Hand: Unveiling Backdoor Vulnerabilities in MLLM-Powered Mobile GUI Agents 

**Authors**: Pengzhou Cheng, Haowen Hu, Zheng Wu, Zongru Wu, Tianjie Ju, Daizong Ding, Zhuosheng Zhang, Gongshen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.14418)  

**Abstract**: Graphical user interface (GUI) agents powered by multimodal large language models (MLLMs) have shown greater promise for human-interaction. However, due to the high fine-tuning cost, users often rely on open-source GUI agents or APIs offered by AI providers, which introduces a critical but underexplored supply chain threat: backdoor attacks. In this work, we first unveil that MLLM-powered GUI agents naturally expose multiple interaction-level triggers, such as historical steps, environment states, and task progress. Based on this observation, we introduce AgentGhost, an effective and stealthy framework for red-teaming backdoor attacks. Specifically, we first construct composite triggers by combining goal and interaction levels, allowing GUI agents to unintentionally activate backdoors while ensuring task utility. Then, we formulate backdoor injection as a Min-Max optimization problem that uses supervised contrastive learning to maximize the feature difference across sample classes at the representation space, improving flexibility of the backdoor. Meanwhile, it adopts supervised fine-tuning to minimize the discrepancy between backdoor and clean behavior generation, enhancing effectiveness and utility. Extensive evaluations of various agent models in two established mobile benchmarks show that AgentGhost is effective and generic, with attack accuracy that reaches 99.7\% on three attack objectives, and shows stealthiness with only 1\% utility degradation. Furthermore, we tailor a defense method against AgentGhost that reduces the attack accuracy to 22.1\%. Our code is available at \texttt{anonymous}. 

---
# Pierce the Mists, Greet the Sky: Decipher Knowledge Overshadowing via Knowledge Circuit Analysis 

**Authors**: Haoming Huang, Yibo Yan, Jiahao Huo, Xin Zou, Xinfeng Li, Kun Wang, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.14406)  

**Abstract**: Large Language Models (LLMs), despite their remarkable capabilities, are hampered by hallucinations. A particularly challenging variant, knowledge overshadowing, occurs when one piece of activated knowledge inadvertently masks another relevant piece, leading to erroneous outputs even with high-quality training data. Current understanding of overshadowing is largely confined to inference-time observations, lacking deep insights into its origins and internal mechanisms during model training. Therefore, we introduce PhantomCircuit, a novel framework designed to comprehensively analyze and detect knowledge overshadowing. By innovatively employing knowledge circuit analysis, PhantomCircuit dissects the internal workings of attention heads, tracing how competing knowledge pathways contribute to the overshadowing phenomenon and its evolution throughout the training process. Extensive experiments demonstrate PhantomCircuit's effectiveness in identifying such instances, offering novel insights into this elusive hallucination and providing the research community with a new methodological lens for its potential mitigation. 

---
# Log-Augmented Generation: Scaling Test-Time Reasoning with Reusable Computation 

**Authors**: Peter Baile Chen, Yi Zhang, Dan Roth, Samuel Madden, Jacob Andreas, Michael Cafarella  

**Link**: [PDF](https://arxiv.org/pdf/2505.14398)  

**Abstract**: While humans naturally learn and adapt from past experiences, large language models (LLMs) and their agentic counterparts struggle to retain reasoning from previous tasks and apply them in future contexts. To address this limitation, we propose a novel framework, log-augmented generation (LAG) that directly reuses prior computation and reasoning from past logs at test time to enhance model's ability to learn from previous tasks and perform better on new, unseen challenges, all while keeping the system efficient and scalable. Specifically, our system represents task logs using key-value (KV) caches, encoding the full reasoning context of prior tasks while storing KV caches for only a selected subset of tokens. When a new task arises, LAG retrieves the KV values from relevant logs to augment generation. Our approach differs from reflection-based memory mechanisms by directly reusing prior reasoning and computations without requiring additional steps for knowledge extraction or distillation. Our method also goes beyond existing KV caching techniques, which primarily target efficiency gains rather than improving accuracy. Experiments on knowledge- and reasoning-intensive datasets demonstrate that our method significantly outperforms standard agentic systems that do not utilize logs, as well as existing solutions based on reflection and KV cache techniques. 

---
# MUG-Eval: A Proxy Evaluation Framework for Multilingual Generation Capabilities in Any Language 

**Authors**: Seyoung Song, Seogyeong Jeong, Eunsu Kim, Jiho Jin, Dongkwan Kim, Jay Shin, Alice Oh  

**Link**: [PDF](https://arxiv.org/pdf/2505.14395)  

**Abstract**: Evaluating text generation capabilities of large language models (LLMs) is challenging, particularly for low-resource languages where methods for direct assessment are scarce. We propose MUG-Eval, a novel framework that evaluates LLMs' multilingual generation capabilities by transforming existing benchmarks into conversational tasks and measuring the LLMs' accuracies on those tasks. We specifically designed these conversational tasks to require effective communication in the target language. Then, we simply use task success rate as a proxy of successful conversation generation. Our approach offers two key advantages: it is independent of language-specific NLP tools or annotated datasets, which are limited for most languages, and it does not rely on LLMs-as-judges, whose evaluation quality degrades outside a few high-resource languages. We evaluate 8 LLMs across 30 languages spanning high, mid, and low-resource categories, and we find that MUG-Eval correlates strongly with established benchmarks ($r$ > 0.75) while enabling standardized comparisons across languages and models. Our framework provides a robust and resource-efficient solution for evaluating multilingual generation that can be extended to thousands of languages. 

---
# Editing Across Languages: A Survey of Multilingual Knowledge Editing 

**Authors**: Nadir Durrani, Basel Mousi, Fahim Dalvi  

**Link**: [PDF](https://arxiv.org/pdf/2505.14393)  

**Abstract**: While Knowledge Editing has been extensively studied in monolingual settings, it remains underexplored in multilingual contexts. This survey systematizes recent research on Multilingual Knowledge Editing (MKE), a growing subdomain of model editing focused on ensuring factual edits generalize reliably across languages. We present a comprehensive taxonomy of MKE methods, covering parameter-based, memory-based, fine-tuning, and hypernetwork approaches. We survey available benchmarks,summarize key findings on method effectiveness and transfer patterns, identify challenges in cross-lingual propagation, and highlight open problems related to language anisotropy, evaluation coverage, and edit scalability. Our analysis consolidates a rapidly evolving area and lays the groundwork for future progress in editable language-aware LLMs. 

---
# AutoRev: Automatic Peer Review System for Academic Research Papers 

**Authors**: Maitreya Prafulla Chitale, Ketaki Mangesh Shetye, Harshit Gupta, Manav Chaudhary, Vasudeva Varma  

**Link**: [PDF](https://arxiv.org/pdf/2505.14376)  

**Abstract**: Generating a review for an academic research paper is a complex task that requires a deep understanding of the document's content and the interdependencies between its sections. It demands not only insight into technical details but also an appreciation of the paper's overall coherence and structure. Recent methods have predominantly focused on fine-tuning large language models (LLMs) to address this challenge. However, they often overlook the computational and performance limitations imposed by long input token lengths. To address this, we introduce AutoRev, an Automatic Peer Review System for Academic Research Papers. Our novel framework represents an academic document as a graph, enabling the extraction of the most critical passages that contribute significantly to the review. This graph-based approach demonstrates effectiveness for review generation and is potentially adaptable to various downstream tasks, such as question answering, summarization, and document representation. When applied to review generation, our method outperforms SOTA baselines by an average of 58.72% across all evaluation metrics. We hope that our work will stimulate further research in applying graph-based extraction techniques to other downstream tasks in NLP. We plan to make our code public upon acceptance. 

---
# Dual Decomposition of Weights and Singular Value Low Rank Adaptation 

**Authors**: Jialong Han, Si Zhang, Ke Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14367)  

**Abstract**: Parameter-Efficient Fine-Tuning (PEFT) has emerged as a critical paradigm for adapting Large Language Models (LLMs) to downstream tasks, among which Low-rank Adaptation (LoRA) represents one of the most widely adopted methodologies. However, existing LoRA-based approaches exhibit two fundamental limitations: unstable training dynamics and inefficient knowledge transfer from pre-trained models, both stemming from random initialization of adapter parameters. To overcome these challenges, we propose DuDe, a novel approach that decomposes weight matrices into magnitude and direction components, employing Singular Value Decomposition (SVD) for principled initialization. Our comprehensive evaluation demonstrates DuDe's superior performance and robustness, achieving up to 48.35\% accuracy on MMLU and 62.53\% ($\pm$ 1.59) accuracy on GSM8K. Our theoretical analysis and empirical validation collectively demonstrate that DuDe's decomposition strategy enhances optimization stability and better preserves pre-trained representations, particularly for domain-specific tasks requiring specialized knowledge. The combination of robust empirical performance and rigorous theoretical foundations establishes DuDe as a significant contribution to PEFT methodologies for LLMs. 

---
# WirelessMathBench: A Mathematical Modeling Benchmark for LLMs in Wireless Communications 

**Authors**: Xin Li, Mengbing Liu, Li Wei, Jiancheng An, Mérouane Debbah, Chau Yuen  

**Link**: [PDF](https://arxiv.org/pdf/2505.14354)  

**Abstract**: Large Language Models (LLMs) have achieved impressive results across a broad array of tasks, yet their capacity for complex, domain-specific mathematical reasoning-particularly in wireless communications-remains underexplored. In this work, we introduce WirelessMathBench, a novel benchmark specifically designed to evaluate LLMs on mathematical modeling challenges to wireless communications engineering. Our benchmark consists of 587 meticulously curated questions sourced from 40 state-of-the-art research papers, encompassing a diverse spectrum of tasks ranging from basic multiple-choice questions to complex equation completion tasks, including both partial and full completions, all of which rigorously adhere to physical and dimensional constraints. Through extensive experimentation with leading LLMs, we observe that while many models excel in basic recall tasks, their performance degrades significantly when reconstructing partially or fully obscured equations, exposing fundamental limitations in current LLMs. Even DeepSeek-R1, the best performer on our benchmark, achieves an average accuracy of only 38.05%, with a mere 7.83% success rate in full equation completion. By publicly releasing WirelessMathBench along with the evaluation toolkit, we aim to advance the development of more robust, domain-aware LLMs for wireless system analysis and broader engineering applications. 

---
# OSoRA: Output-Dimension and Singular-Value Initialized Low-Rank Adaptation 

**Authors**: Jialong Han, Si Zhang, Ke Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14350)  

**Abstract**: Fine-tuning Large Language Models (LLMs) has become increasingly challenging due to their massive scale and associated computational costs. Parameter-Efficient Fine-Tuning (PEFT) methodologies have been proposed as computational alternatives; however, their implementations still require significant resources. In this paper, we present OSoRA (Output-Dimension and Singular-Value Initialized Low-Rank Adaptation), a novel PEFT method for LLMs. OSoRA extends Low-Rank Adaptation (LoRA) by integrating Singular Value Decomposition (SVD) with learnable scaling vectors in a unified framework. It first performs an SVD of pre-trained weight matrices, then optimizes an output-dimension vector during training, while keeping the corresponding singular vector matrices frozen. OSoRA substantially reduces computational resource requirements by minimizing the number of trainable parameters during fine-tuning. Comprehensive evaluations across mathematical reasoning, common sense reasoning, and other benchmarks demonstrate that OSoRA achieves comparable or superior performance to state-of-the-art methods like LoRA and VeRA, while maintaining a linear parameter scaling even as the rank increases to higher dimensions. Our ablation studies further confirm that jointly training both the singular values and the output-dimension vector is critical for optimal performance. 

---
# QA-prompting: Improving Summarization with Large Language Models using Question-Answering 

**Authors**: Neelabh Sinha  

**Link**: [PDF](https://arxiv.org/pdf/2505.14347)  

**Abstract**: Language Models (LMs) have revolutionized natural language processing, enabling high-quality text generation through prompting and in-context learning. However, models often struggle with long-context summarization due to positional biases, leading to suboptimal extraction of critical information. There are techniques to improve this with fine-tuning, pipelining, or using complex techniques, which have their own challenges. To solve these challenges, we propose QA-prompting - a simple prompting method for summarization that utilizes question-answering as an intermediate step prior to summary generation. Our method extracts key information and enriches the context of text to mitigate positional biases and improve summarization in a single LM call per task without requiring fine-tuning or pipelining. Experiments on multiple datasets belonging to different domains using ten state-of-the-art pre-trained models demonstrate that QA-prompting outperforms baseline and other state-of-the-art methods, achieving up to 29% improvement in ROUGE scores. This provides an effective and scalable solution for summarization and highlights the importance of domain-specific question selection for optimal performance. 

---
# A MIND for Reasoning: Meta-learning for In-context Deduction 

**Authors**: Leonardo Bertolazzi, Manuel Vargas Guzmán, Raffaella Bernardi, Maciej Malicki, Jakub Szymanik  

**Link**: [PDF](https://arxiv.org/pdf/2505.14313)  

**Abstract**: Large language models (LLMs) are increasingly evaluated on formal tasks, where strong reasoning abilities define the state of the art. However, their ability to generalize to out-of-distribution problems remains limited. In this paper, we investigate how LLMs can achieve a systematic understanding of deductive rules. Our focus is on the task of identifying the appropriate subset of premises within a knowledge base needed to derive a given hypothesis. To tackle this challenge, we propose Meta-learning for In-context Deduction (MIND), a novel few-shot meta-learning fine-tuning approach. The goal of MIND is to enable models to generalize more effectively to unseen knowledge bases and to systematically apply inference rules. Our results show that MIND significantly improves generalization in small LMs ranging from 1.5B to 7B parameters. The benefits are especially pronounced in smaller models and low-data settings. Remarkably, small models fine-tuned with MIND outperform state-of-the-art LLMs, such as GPT-4o and o3-mini, on this task. 

---
# HausaNLP: Current Status, Challenges and Future Directions for Hausa Natural Language Processing 

**Authors**: Shamsuddeen Hassan Muhammad, Ibrahim Said Ahmad, Idris Abdulmumin, Falalu Ibrahim Lawan, Babangida Sani, Sukairaj Hafiz Imam, Yusuf Aliyu, Sani Abdullahi Sani, Ali Usman Umar, Kenneth Church, Vukosi Marivate  

**Link**: [PDF](https://arxiv.org/pdf/2505.14311)  

**Abstract**: Hausa Natural Language Processing (NLP) has gained increasing attention in recent years, yet remains understudied as a low-resource language despite having over 120 million first-language (L1) and 80 million second-language (L2) speakers worldwide. While significant advances have been made in high-resource languages, Hausa NLP faces persistent challenges, including limited open-source datasets and inadequate model representation. This paper presents an overview of the current state of Hausa NLP, systematically examining existing resources, research contributions, and gaps across fundamental NLP tasks: text classification, machine translation, named entity recognition, speech recognition, and question answering. We introduce HausaNLP (this https URL), a curated catalog that aggregates datasets, tools, and research works to enhance accessibility and drive further development. Furthermore, we discuss challenges in integrating Hausa into large language models (LLMs), addressing issues of suboptimal tokenization and dialectal variation. Finally, we propose strategic research directions emphasizing dataset expansion, improved language modeling approaches, and strengthened community collaboration to advance Hausa NLP. Our work provides both a foundation for accelerating Hausa NLP progress and valuable insights for broader multilingual NLP research. 

---
# Studying the Role of Input-Neighbor Overlap in Retrieval-Augmented Language Models Training Efficiency 

**Authors**: Ehsan Doostmohammadi, Marco Kuhlmann  

**Link**: [PDF](https://arxiv.org/pdf/2505.14309)  

**Abstract**: Retrieval-augmented language models have demonstrated performance comparable to much larger models while requiring fewer computational resources. The effectiveness of these models crucially depends on the overlap between query and retrieved context, but the optimal degree of this overlap remains unexplored. In this paper, we systematically investigate how varying levels of query--context overlap affect model performance during both training and inference. Our experiments reveal that increased overlap initially has minimal effect, but substantially improves test-time perplexity and accelerates model learning above a critical threshold. Building on these findings, we demonstrate that deliberately increasing overlap through synthetic context can enhance data efficiency and reduce training time by approximately 40\% without compromising performance. We specifically generate synthetic context through paraphrasing queries. We validate our perplexity-based findings on question-answering tasks, confirming that the benefits of retrieval-augmented language modeling extend to practical applications. Our results provide empirical evidence of significant optimization potential for retrieval mechanisms in language model pretraining. 

---
# JOLT-SQL: Joint Loss Tuning of Text-to-SQL with Confusion-aware Noisy Schema Sampling 

**Authors**: Jinwang Song, Hongying Zan, Kunli Zhang, Lingling Mu, Yingjie Han, Haobo Hua, Min Peng  

**Link**: [PDF](https://arxiv.org/pdf/2505.14305)  

**Abstract**: Text-to-SQL, which maps natural language to SQL queries, has benefited greatly from recent advances in Large Language Models (LLMs). While LLMs offer various paradigms for this task, including prompting and supervised fine-tuning (SFT), SFT approaches still face challenges such as complex multi-stage pipelines and poor robustness to noisy schema information. To address these limitations, we present JOLT-SQL, a streamlined single-stage SFT framework that jointly optimizes schema linking and SQL generation via a unified loss. JOLT-SQL employs discriminative schema linking, enhanced by local bidirectional attention, alongside a confusion-aware noisy schema sampling strategy with selective attention to improve robustness under noisy schema conditions. Experiments on the Spider and BIRD benchmarks demonstrate that JOLT-SQL achieves state-of-the-art execution accuracy among comparable-size open-source models, while significantly improving both training and inference efficiency. 

---
# Cross-Lingual Optimization for Language Transfer in Large Language Models 

**Authors**: Jungseob Lee, Seongtae Hong, Hyeonseok Moon, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2505.14297)  

**Abstract**: Adapting large language models to other languages typically employs supervised fine-tuning (SFT) as a standard approach. However, it often suffers from an overemphasis on English performance, a phenomenon that is especially pronounced in data-constrained environments. To overcome these challenges, we propose \textbf{Cross-Lingual Optimization (CLO)} that efficiently transfers an English-centric LLM to a target language while preserving its English capabilities. CLO utilizes publicly available English SFT data and a translation model to enable cross-lingual transfer. We conduct experiments using five models on six languages, each possessing varying levels of resource. Our results show that CLO consistently outperforms SFT in both acquiring target language proficiency and maintaining English performance. Remarkably, in low-resource languages, CLO with only 3,200 samples surpasses SFT with 6,400 samples, demonstrating that CLO can achieve better performance with less data. Furthermore, we find that SFT is particularly sensitive to data quantity in medium and low-resource languages, whereas CLO remains robust. Our comprehensive analysis emphasizes the limitations of SFT and incorporates additional training strategies in CLO to enhance efficiency. 

---
# Universal Acoustic Adversarial Attacks for Flexible Control of Speech-LLMs 

**Authors**: Rao Ma, Mengjie Qian, Vyas Raina, Mark Gales, Kate Knill  

**Link**: [PDF](https://arxiv.org/pdf/2505.14286)  

**Abstract**: The combination of pre-trained speech encoders with large language models has enabled the development of speech LLMs that can handle a wide range of spoken language processing tasks. While these models are powerful and flexible, this very flexibility may make them more vulnerable to adversarial attacks. To examine the extent of this problem, in this work we investigate universal acoustic adversarial attacks on speech LLMs. Here a fixed, universal, adversarial audio segment is prepended to the original input audio. We initially investigate attacks that cause the model to either produce no output or to perform a modified task overriding the original prompt. We then extend the nature of the attack to be selective so that it activates only when specific input attributes, such as a speaker gender or spoken language, are present. Inputs without the targeted attribute should be unaffected, allowing fine-grained control over the model outputs. Our findings reveal critical vulnerabilities in Qwen2-Audio and Granite-Speech and suggest that similar speech LLMs may be susceptible to universal adversarial attacks. This highlights the need for more robust training strategies and improved resistance to adversarial attacks. 

---
# YESciEval: Robust LLM-as-a-Judge for Scientific Question Answering 

**Authors**: Jennifer D'Souza, Hamed Babaei Giglou, Quentin Münch  

**Link**: [PDF](https://arxiv.org/pdf/2505.14279)  

**Abstract**: Large Language Models (LLMs) drive scientific question-answering on modern search engines, yet their evaluation robustness remains underexplored. We introduce YESciEval, an open-source framework that combines fine-grained rubric-based assessment with reinforcement learning to mitigate optimism bias in LLM evaluators. We release multidisciplinary scienceQ&A datasets, including adversarial variants, with evaluation scores from multiple LLMs. Independent of proprietary models and human feedback, our approach enables scalable, cost-free evaluation. By advancing reliable LLM-as-a-judge models, this work supports AI alignment and fosters robust, transparent evaluation essential for scientific inquiry and artificial general intelligence. 

---
# Data-Efficient Hate Speech Detection via Cross-Lingual Nearest Neighbor Retrieval with Limited Labeled Data 

**Authors**: Faeze Ghorbanpour, Daryna Dementieva, Alexander Fraser  

**Link**: [PDF](https://arxiv.org/pdf/2505.14272)  

**Abstract**: Considering the importance of detecting hateful language, labeled hate speech data is expensive and time-consuming to collect, particularly for low-resource languages. Prior work has demonstrated the effectiveness of cross-lingual transfer learning and data augmentation in improving performance on tasks with limited labeled data. To develop an efficient and scalable cross-lingual transfer learning approach, we leverage nearest-neighbor retrieval to augment minimal labeled data in the target language, thereby enhancing detection performance. Specifically, we assume access to a small set of labeled training instances in the target language and use these to retrieve the most relevant labeled examples from a large multilingual hate speech detection pool. We evaluate our approach on eight languages and demonstrate that it consistently outperforms models trained solely on the target language data. Furthermore, in most cases, our method surpasses the current state-of-the-art. Notably, our approach is highly data-efficient, retrieving as small as 200 instances in some cases while maintaining superior performance. Moreover, it is scalable, as the retrieval pool can be easily expanded, and the method can be readily adapted to new languages and tasks. We also apply maximum marginal relevance to mitigate redundancy and filter out highly similar retrieved instances, resulting in improvements in some languages. 

---
# FAID: Fine-grained AI-generated Text Detection using Multi-task Auxiliary and Multi-level Contrastive Learning 

**Authors**: Minh Ngoc Ta, Dong Cao Van, Duc-Anh Hoang, Minh Le-Anh, Truong Nguyen, My Anh Tran Nguyen, Yuxia Wang, Preslav Nakov, Sang Dinh  

**Link**: [PDF](https://arxiv.org/pdf/2505.14271)  

**Abstract**: The growing collaboration between humans and AI models in generative tasks has introduced new challenges in distinguishing between human-written, AI-generated, and human-AI collaborative texts. In this work, we collect a multilingual, multi-domain, multi-generator dataset FAIDSet. We further introduce a fine-grained detection framework FAID to classify text into these three categories, meanwhile identifying the underlying AI model family. Unlike existing binary classifiers, FAID is built to capture both authorship and model-specific characteristics. Our method combines multi-level contrastive learning with multi-task auxiliary classification to learn subtle stylistic cues. By modeling AI families as distinct stylistic entities, FAID offers improved interpretability. We incorporate an adaptation to address distributional shifts without retraining for unseen data. Experimental results demonstrate that FAID outperforms several baseline approaches, particularly enhancing the generalization accuracy on unseen domains and new AI models. It provide a potential solution for improving transparency and accountability in AI-assisted writing. 

---
# Think-J: Learning to Think for Generative LLM-as-a-Judge 

**Authors**: Hui Huang, Yancheng He, Hongli Zhou, Rui Zhang, Wei Liu, Weixun Wang, Wenbo Su, Bo Zheng, Jiaheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.14268)  

**Abstract**: LLM-as-a-Judge refers to the automatic modeling of preferences for responses generated by Large Language Models (LLMs), which is of significant importance for both LLM evaluation and reward modeling. Although generative LLMs have made substantial progress in various tasks, their performance as LLM-Judge still falls short of expectations. In this work, we propose Think-J, which improves generative LLM-as-a-Judge by learning how to think. We first utilized a small amount of curated data to develop the model with initial judgment thinking capabilities. Subsequently, we optimize the judgment thinking traces based on reinforcement learning (RL). We propose two methods for judgment thinking optimization, based on offline and online RL, respectively. The offline RL requires training a critic model to construct positive and negative examples for learning. The online method defines rule-based reward as feedback for optimization. Experimental results showed that our approach can significantly enhance the evaluation capability of generative LLM-Judge, surpassing both generative and classifier-based LLM-Judge without requiring extra human annotations. 

---
# FuxiMT: Sparsifying Large Language Models for Chinese-Centric Multilingual Machine Translation 

**Authors**: Shaolin Zhu, Tianyu Dong, Bo Li, Deyi Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2505.14256)  

**Abstract**: In this paper, we present FuxiMT, a novel Chinese-centric multilingual machine translation model powered by a sparsified large language model (LLM). We adopt a two-stage strategy to train FuxiMT. We first pre-train the model on a massive Chinese corpus and then conduct multilingual fine-tuning on a large parallel dataset encompassing 65 languages. FuxiMT incorporates Mixture-of-Experts (MoEs) and employs a curriculum learning strategy for robust performance across various resource levels. Experimental results demonstrate that FuxiMT significantly outperforms strong baselines, including state-of-the-art LLMs and machine translation models, particularly under low-resource scenarios. Furthermore, FuxiMT exhibits remarkable zero-shot translation capabilities for unseen language pairs, indicating its potential to bridge communication gaps where parallel data are scarce or unavailable. 

---
# TransBench: Benchmarking Machine Translation for Industrial-Scale Applications 

**Authors**: Haijun Li, Tianqi Shi, Zifu Shang, Yuxuan Han, Xueyu Zhao, Hao Wang, Yu Qian, Zhiqiang Qian, Linlong Xu, Minghao Wu, Chenyang Lyu, Longyue Wang, Gongbo Tang, Weihua Luo, Zhao Xu, Kaifu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14244)  

**Abstract**: Machine translation (MT) has become indispensable for cross-border communication in globalized industries like e-commerce, finance, and legal services, with recent advancements in large language models (LLMs) significantly enhancing translation quality. However, applying general-purpose MT models to industrial scenarios reveals critical limitations due to domain-specific terminology, cultural nuances, and stylistic conventions absent in generic benchmarks. Existing evaluation frameworks inadequately assess performance in specialized contexts, creating a gap between academic benchmarks and real-world efficacy. To address this, we propose a three-level translation capability framework: (1) Basic Linguistic Competence, (2) Domain-Specific Proficiency, and (3) Cultural Adaptation, emphasizing the need for holistic evaluation across these dimensions. We introduce TransBench, a benchmark tailored for industrial MT, initially targeting international e-commerce with 17,000 professionally translated sentences spanning 4 main scenarios and 33 language pairs. TransBench integrates traditional metrics (BLEU, TER) with Marco-MOS, a domain-specific evaluation model, and provides guidelines for reproducible benchmark construction. Our contributions include: (1) a structured framework for industrial MT evaluation, (2) the first publicly available benchmark for e-commerce translation, (3) novel metrics probing multi-level translation quality, and (4) open-sourced evaluation tools. This work bridges the evaluation gap, enabling researchers and practitioners to systematically assess and enhance MT systems for industry-specific needs. 

---
# Technical Report on classification of literature related to children speech disorder 

**Authors**: Ziang Wang, Amir Aryani  

**Link**: [PDF](https://arxiv.org/pdf/2505.14242)  

**Abstract**: This technical report presents a natural language processing (NLP)-based approach for systematically classifying scientific literature on childhood speech disorders. We retrieved and filtered 4,804 relevant articles published after 2015 from the PubMed database using domain-specific keywords. After cleaning and pre-processing the abstracts, we applied two topic modeling techniques - Latent Dirichlet Allocation (LDA) and BERTopic - to identify latent thematic structures in the corpus. Our models uncovered 14 clinically meaningful clusters, such as infantile hyperactivity and abnormal epileptic behavior. To improve relevance and precision, we incorporated a custom stop word list tailored to speech pathology. Evaluation results showed that the LDA model achieved a coherence score of 0.42 and a perplexity of -7.5, indicating strong topic coherence and predictive performance. The BERTopic model exhibited a low proportion of outlier topics (less than 20%), demonstrating its capacity to classify heterogeneous literature effectively. These results provide a foundation for automating literature reviews in speech-language pathology. 

---
# ABBA: Highly Expressive Hadamard Product Adaptation for Large Language Models 

**Authors**: Raghav Singhal, Kaustubh Ponkshe, Rohit Vartak, Praneeth Vepakomma  

**Link**: [PDF](https://arxiv.org/pdf/2505.14238)  

**Abstract**: Large Language Models have demonstrated strong performance across a wide range of tasks, but adapting them efficiently to new domains remains a key challenge. Parameter-Efficient Fine-Tuning (PEFT) methods address this by introducing lightweight, trainable modules while keeping most pre-trained weights fixed. The prevailing approach, LoRA, models updates using a low-rank decomposition, but its expressivity is inherently constrained by the rank. Recent methods like HiRA aim to increase expressivity by incorporating a Hadamard product with the frozen weights, but still rely on the structure of the pre-trained model. We introduce ABBA, a new PEFT architecture that reparameterizes the update as a Hadamard product of two independently learnable low-rank matrices. In contrast to prior work, ABBA fully decouples the update from the pre-trained weights, enabling both components to be optimized freely. This leads to significantly higher expressivity under the same parameter budget. We formally analyze ABBA's expressive capacity and validate its advantages through matrix reconstruction experiments. Empirically, ABBA achieves state-of-the-art results on arithmetic and commonsense reasoning benchmarks, consistently outperforming existing PEFT methods by a significant margin across multiple models. Our code is publicly available at: this https URL. 

---
# Mechanistic Fine-tuning for In-context Learning 

**Authors**: Hakaze Cho, Peng Luo, Mariko Kato, Rin Kaenbyou, Naoya Inoue  

**Link**: [PDF](https://arxiv.org/pdf/2505.14233)  

**Abstract**: In-context Learning (ICL) utilizes structured demonstration-query inputs to induce few-shot learning on Language Models (LMs), which are not originally pre-trained on ICL-style data. To bridge the gap between ICL and pre-training, some approaches fine-tune LMs on large ICL-style datasets by an end-to-end paradigm with massive computational costs. To reduce such costs, in this paper, we propose Attention Behavior Fine-Tuning (ABFT), utilizing the previous findings on the inner mechanism of ICL, building training objectives on the attention scores instead of the final outputs, to force the attention scores to focus on the correct label tokens presented in the context and mitigate attention scores from the wrong label tokens. Our experiments on 9 modern LMs and 8 datasets empirically find that ABFT outperforms in performance, robustness, unbiasedness, and efficiency, with only around 0.01% data cost compared to the previous methods. Moreover, our subsequent analysis finds that the end-to-end training objective contains the ABFT objective, suggesting the implicit bias of ICL-style data to the emergence of induction heads. Our work demonstrates the possibility of controlling specific module sequences within LMs to improve their behavior, opening up the future application of mechanistic interpretability. 

---
# "Haet Bhasha aur Diskrimineshun": Phonetic Perturbations in Code-Mixed Hinglish to Red-Team LLMs 

**Authors**: Darpan Aswal, Siddharth D Jaiswal  

**Link**: [PDF](https://arxiv.org/pdf/2505.14226)  

**Abstract**: Large Language Models (LLMs) have become increasingly powerful, with multilingual and multimodal capabilities improving by the day. These models are being evaluated through audits, alignment studies and red-teaming efforts to expose model vulnerabilities towards generating harmful, biased and unfair content. Existing red-teaming efforts have previously focused on the English language, using fixed template-based attacks; thus, models continue to be susceptible to multilingual jailbreaking strategies, especially in the multimodal context. In this study, we introduce a novel strategy that leverages code-mixing and phonetic perturbations to jailbreak LLMs for both text and image generation tasks. We also introduce two new jailbreak strategies that show higher effectiveness than baseline strategies. Our work presents a method to effectively bypass safety filters in LLMs while maintaining interpretability by applying phonetic misspellings to sensitive words in code-mixed prompts. Our novel prompts achieve a 99% Attack Success Rate for text generation and 78% for image generation, with Attack Relevance Rate of 100% for text generation and 95% for image generation when using the phonetically perturbed code-mixed prompts. Our interpretability experiments reveal that phonetic perturbations impact word tokenization, leading to jailbreak success. Our study motivates increasing the focus towards more generalizable safety alignment for multilingual multimodal models, especially in real-world settings wherein prompts can have misspelt words. 

---
# Automatic Dataset Generation for Knowledge Intensive Question Answering Tasks 

**Authors**: Sizhe Yuen, Ting Su, Ziyang Wang, Yali Du, Adam J. Sobey  

**Link**: [PDF](https://arxiv.org/pdf/2505.14212)  

**Abstract**: A question-answering (QA) system is to search suitable answers within a knowledge base. Current QA systems struggle with queries requiring complex reasoning or real-time knowledge integration. They are often supplemented with retrieval techniques on a data source such as Retrieval-Augmented Generation (RAG). However, RAG continues to face challenges in handling complex reasoning and logical connections between multiple sources of information. A novel approach for enhancing Large Language Models (LLMs) in knowledge-intensive QA tasks is presented through the automated generation of context-based QA pairs. This methodology leverages LLMs to create fine-tuning data, reducing reliance on human labelling and improving model comprehension and reasoning capabilities. The proposed system includes an automated QA generator and a model fine-tuner, evaluated using perplexity, ROUGE, BLEU, and BERTScore. Comprehensive experiments demonstrate improvements in logical coherence and factual accuracy, with implications for developing adaptable Artificial Intelligence (AI) systems. Mistral-7b-v0.3 outperforms Llama-3-8b with BERT F1, BLEU, and ROUGE scores 0.858, 0.172, and 0.260 of for the LLM generated QA pairs compared to scores of 0.836, 0.083, and 0.139 for the human annotated QA pairs. 

---
# Unraveling Interwoven Roles of Large Language Models in Authorship Privacy: Obfuscation, Mimicking, and Verification 

**Authors**: Tuc Nguyen, Yifan Hu, Thai Le  

**Link**: [PDF](https://arxiv.org/pdf/2505.14195)  

**Abstract**: Recent advancements in large language models (LLMs) have been fueled by large scale training corpora drawn from diverse sources such as websites, news articles, and books. These datasets often contain explicit user information, such as person names and addresses, that LLMs may unintentionally reproduce in their generated outputs. Beyond such explicit content, LLMs can also leak identity revealing cues through implicit signals such as distinctive writing styles, raising significant concerns about authorship privacy. There are three major automated tasks in authorship privacy, namely authorship obfuscation (AO), authorship mimicking (AM), and authorship verification (AV). Prior research has studied AO, AM, and AV independently. However, their interplays remain under explored, which leaves a major research gap, especially in the era of LLMs, where they are profoundly shaping how we curate and share user generated content, and the distinction between machine generated and human authored text is also increasingly blurred. This work then presents the first unified framework for analyzing the dynamic relationships among LLM enabled AO, AM, and AV in the context of authorship privacy. We quantify how they interact with each other to transform human authored text, examining effects at a single point in time and iteratively over time. We also examine the role of demographic metadata, such as gender, academic background, in modulating their performances, inter-task dynamics, and privacy risks. All source code will be publicly available. 

---
# ThinkSwitcher: When to Think Hard, When to Think Fast 

**Authors**: Guosheng Liang, Longguang Zhong, Ziyi Yang, Xiaojun Quan  

**Link**: [PDF](https://arxiv.org/pdf/2505.14183)  

**Abstract**: Large reasoning models (LRMs) excel at solving complex tasks by leveraging long chain-of-thought (CoT) reasoning. However, this often leads to overthinking on simple tasks, resulting in unnecessary computational overhead. We observe that LRMs inherently possess the capability for efficient short CoT reasoning, which can be reliably elicited through prompt design. To leverage this capability, we propose ThinkSwitcher, a framework that enables a single LRM to dynamically switch between short and long CoT modes based on task complexity. ThinkSwitcher introduces a lightweight switching module trained with supervision signals derived from the relative performance of each reasoning mode across tasks. Experiments on multiple reasoning benchmarks show that ThinkSwitcher reduces computational cost by 20-30% while maintaining high accuracy on complex tasks. This demonstrates the effectiveness of ThinkSwitcher as a scalable and efficient solution for unified LRM deployment. 

---
# SlangDIT: Benchmarking LLMs in Interpretative Slang Translation 

**Authors**: Yunlong Liang, Fandong Meng, Jiaan Wang, Jie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.14181)  

**Abstract**: The challenge of slang translation lies in capturing context-dependent semantic extensions, as slang terms often convey meanings beyond their literal interpretation. While slang detection, explanation, and translation have been studied as isolated tasks in the era of large language models (LLMs), their intrinsic interdependence remains underexplored. The main reason is lacking of a benchmark where the two tasks can be a prerequisite for the third one, which can facilitate idiomatic translation. In this paper, we introduce the interpretative slang translation task (named SlangDIT) consisting of three sub-tasks: slang detection, cross-lingual slang explanation, and slang translation within the current context, aiming to generate more accurate translation with the help of slang detection and slang explanation. To this end, we construct a SlangDIT dataset, containing over 25k English-Chinese sentence pairs. Each source sentence mentions at least one slang term and is labeled with corresponding cross-lingual slang explanation. Based on the benchmark, we propose a deep thinking model, named SlangOWL. It firstly identifies whether the sentence contains a slang, and then judges whether the slang is polysemous and analyze its possible meaning. Further, the SlangOWL provides the best explanation of the slang term targeting on the current context. Finally, according to the whole thought, the SlangOWL offers a suitable translation. Our experiments on LLMs (\emph{e.g.}, Qwen2.5 and LLama-3.1), show that our deep thinking approach indeed enhances the performance of LLMs where the proposed SLangOWL significantly surpasses the vanilla models and supervised fine-tuned models without thinking. 

---
# Enhancing Abstractive Summarization of Scientific Papers Using Structure Information 

**Authors**: Tong Bao, Heng Zhang, Chengzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14179)  

**Abstract**: Abstractive summarization of scientific papers has always been a research focus, yet existing methods face two main challenges. First, most summarization models rely on Encoder-Decoder architectures that treat papers as sequences of words, thus fail to fully capture the structured information inherent in scientific papers. Second, existing research often use keyword mapping or feature engineering to identify the structural information, but these methods struggle with the structural flexibility of scientific papers and lack robustness across different disciplines. To address these challenges, we propose a two-stage abstractive summarization framework that leverages automatic recognition of structural functions within scientific papers. In the first stage, we standardize chapter titles from numerous scientific papers and construct a large-scale dataset for structural function recognition. A classifier is then trained to automatically identify the key structural components (e.g., Background, Methods, Results, Discussion), which provides a foundation for generating more balanced summaries. In the second stage, we employ Longformer to capture rich contextual relationships across sections and generating context-aware summaries. Experiments conducted on two domain-specific scientific paper summarization datasets demonstrate that our method outperforms advanced baselines, and generates more comprehensive summaries. The code and dataset can be accessed at this https URL. 

---
# Tokenization Constraints in LLMs: A Study of Symbolic and Arithmetic Reasoning Limits 

**Authors**: Xiang Zhang, Juntai Cao, Jiaqi Wei, Yiwei Xu, Chenyu You  

**Link**: [PDF](https://arxiv.org/pdf/2505.14178)  

**Abstract**: Tokenization is the first - and often underappreciated - layer of computation in language models. While Chain-of-Thought (CoT) prompting enables transformer models to approximate recurrent computation by externalizing intermediate steps, we show that the success of such reasoning is fundamentally bounded by the structure of tokenized inputs. This work presents a theoretical and empirical investigation into how tokenization schemes, particularly subword-based methods like byte-pair encoding (BPE), impede symbolic computation by merging or obscuring atomic reasoning units. We introduce the notion of Token Awareness to formalize how poor token granularity disrupts logical alignment and prevents models from generalizing symbolic procedures. Through systematic evaluation on arithmetic and symbolic tasks, we demonstrate that token structure dramatically affect reasoning performance, causing failure even with CoT, while atomically-aligned formats unlock strong generalization, allowing small models (e.g., GPT-4o-mini) to outperform larger systems (e.g., o1) in structured reasoning. Our findings reveal that symbolic reasoning ability in LLMs is not purely architectural, but deeply conditioned on token-level representations. 

---
# Cheaper, Better, Faster, Stronger: Robust Text-to-SQL without Chain-of-Thought or Fine-Tuning 

**Authors**: Yusuf Denizay Dönder, Derek Hommel, Andrea W Wen-Yi, David Mimno, Unso Eun Seo Jo  

**Link**: [PDF](https://arxiv.org/pdf/2505.14174)  

**Abstract**: LLMs are effective at code generation tasks like text-to-SQL, but is it worth the cost? Many state-of-the-art approaches use non-task-specific LLM techniques including Chain-of-Thought (CoT), self-consistency, and fine-tuning. These methods can be costly at inference time, sometimes requiring over a hundred LLM calls with reasoning, incurring average costs of up to \$0.46 per query, while fine-tuning models can cost thousands of dollars. We introduce "N-rep" consistency, a more cost-efficient text-to-SQL approach that achieves similar BIRD benchmark scores as other more expensive methods, at only \$0.039 per query. N-rep leverages multiple representations of the same schema input to mitigate weaknesses in any single representation, making the solution more robust and allowing the use of smaller and cheaper models without any reasoning or fine-tuning. To our knowledge, N-rep is the best-performing text-to-SQL approach in its cost range. 

---
# THOR-MoE: Hierarchical Task-Guided and Context-Responsive Routing for Neural Machine Translation 

**Authors**: Yunlong Liang, Fandong Meng, Jie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.14173)  

**Abstract**: The sparse Mixture-of-Experts (MoE) has achieved significant progress for neural machine translation (NMT). However, there exist two limitations in current MoE solutions which may lead to sub-optimal performance: 1) they directly use the task knowledge of NMT into MoE (\emph{e.g.}, domain/linguistics-specific knowledge), which are generally unavailable at practical application and neglect the naturally grouped domain/linguistic properties; 2) the expert selection only depends on the localized token representation without considering the context, which fully grasps the state of each token in a global view. To address the above limitations, we propose THOR-MoE via arming the MoE with hierarchical task-guided and context-responsive routing policies. Specifically, it 1) firstly predicts the domain/language label and then extracts mixed domain/language representation to allocate task-level experts in a hierarchical manner; 2) injects the context information to enhance the token routing from the pre-selected task-level experts set, which can help each token to be accurately routed to more specialized and suitable experts. Extensive experiments on multi-domain translation and multilingual translation benchmarks with different architectures consistently demonstrate the superior performance of THOR-MoE. Additionally, the THOR-MoE operates as a plug-and-play module compatible with existing Top-$k$~\cite{shazeer2017} and Top-$p$~\cite{huang-etal-2024-harder} routing schemes, ensuring broad applicability across diverse MoE architectures. For instance, compared with vanilla Top-$p$~\cite{huang-etal-2024-harder} routing, the context-aware manner can achieve an average improvement of 0.75 BLEU with less than 22\% activated parameters on multi-domain translation tasks. 

---
# The Strawberry Problem: Emergence of Character-level Understanding in Tokenized Language Models 

**Authors**: Adrian Cosma, Stefan Ruseti, Emilian Radoi, Mihai Dascalu  

**Link**: [PDF](https://arxiv.org/pdf/2505.14172)  

**Abstract**: Despite their remarkable progress across diverse domains, Large Language Models (LLMs) consistently fail at simple character-level tasks, such as counting letters in words, due to a fundamental limitation: tokenization. In this work, we frame this limitation as a problem of low mutual information and analyze it in terms of concept emergence. Using a suite of 19 synthetic tasks that isolate character-level reasoning in a controlled setting, we show that such capabilities emerge slowly, suddenly, and only late in training. We further show that percolation-based models of concept emergence explain these patterns, suggesting that learning character composition is not fundamentally different from learning commonsense knowledge. To address this bottleneck, we propose a lightweight architectural modification that significantly improves character-level reasoning while preserving the inductive advantages of subword models. Together, our results bridge low-level perceptual gaps in tokenized LMs and provide a principled framework for understanding and mitigating their structural blind spots. We make our code publicly available. 

---
# PL-FGSA: A Prompt Learning Framework for Fine-Grained Sentiment Analysis Based on MindSpore 

**Authors**: Zhenkai Qin, Jiajing He, Qiao Fang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14165)  

**Abstract**: Fine-grained sentiment analysis (FGSA) aims to identify sentiment polarity toward specific aspects within a text, enabling more precise opinion mining in domains such as product reviews and social media. However, traditional FGSA approaches often require task-specific architectures and extensive annotated data, limiting their generalization and scalability. To address these challenges, we propose PL-FGSA, a unified prompt learning-based framework implemented using the MindSpore platform, which integrates prompt design with a lightweight TextCNN backbone. Our method reformulates FGSA as a multi-task prompt-augmented generation problem, jointly tackling aspect extraction, sentiment classification, and causal explanation in a unified paradigm. By leveraging prompt-based guidance, PL-FGSA enhances interpretability and achieves strong performance under both full-data and low-resource conditions. Experiments on three benchmark datasets-SST-2, SemEval-2014 Task 4, and MAMS-demonstrate that our model consistently outperforms traditional fine-tuning methods and achieves F1-scores of 0.922, 0.694, and 0.597, respectively. These results validate the effectiveness of prompt-based generalization and highlight the practical value of PL-FGSA for real-world sentiment analysis tasks. 

---
# Breaking Language Barriers or Reinforcing Bias? A Study of Gender and Racial Disparities in Multilingual Contrastive Vision Language Models 

**Authors**: Zahraa Al Sahili, Ioannis Patras, Matthew Purver  

**Link**: [PDF](https://arxiv.org/pdf/2505.14160)  

**Abstract**: Multilingual vision-language models promise universal image-text retrieval, yet their social biases remain under-explored. We present the first systematic audit of three public multilingual CLIP checkpoints -- M-CLIP, NLLB-CLIP, and CAPIVARA-CLIP -- across ten languages that vary in resource availability and grammatical gender. Using balanced subsets of \textsc{FairFace} and the \textsc{PATA} stereotype suite in a zero-shot setting, we quantify race and gender bias and measure stereotype amplification. Contrary to the assumption that multilinguality mitigates bias, every model exhibits stronger gender bias than its English-only baseline. CAPIVARA-CLIP shows its largest biases precisely in the low-resource languages it targets, while the shared cross-lingual encoder of NLLB-CLIP transports English gender stereotypes into gender-neutral languages; loosely coupled encoders largely avoid this transfer. Highly gendered languages consistently magnify all measured bias types, but even gender-neutral languages remain vulnerable when cross-lingual weight sharing imports foreign stereotypes. Aggregated metrics conceal language-specific ``hot spots,'' underscoring the need for fine-grained, language-aware bias evaluation in future multilingual vision-language research. 

---
# Temporal Alignment of Time Sensitive Facts with Activation Engineering 

**Authors**: Sanjay Govindan, Maurice Pagnucco, Yang Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.14158)  

**Abstract**: Large Language Models (LLMs) are trained on diverse and often conflicting knowledge spanning multiple domains and time periods. Some of this knowledge is only valid within specific temporal contexts, such as answering the question, "Who is the President of the United States in 2022?" Ensuring LLMs generate time appropriate responses is crucial for maintaining relevance and accuracy. In this work we explore activation engineering as a method for temporally aligning LLMs to improve factual recall without any training or dataset creation. In this research we explore an activation engineering technique to ground three versions of LLaMA 2 to specific points in time and examine the effects of varying injection layers and prompting strategies. Our experiments demonstrate up to a 44% and 16% improvement in relative and explicit prompting respectively, achieving comparable performance to the fine-tuning method proposed by Zhao et al. (2024) . Notably, our approach achieves similar results to the fine-tuning baseline while being significantly more computationally efficient and requiring no pre-aligned datasets. 

---
# Prior Prompt Engineering for Reinforcement Fine-Tuning 

**Authors**: Pittawat Taveekitworachai, Potsawee Manakul, Sarana Nutanong, Kunat Pipatanakul  

**Link**: [PDF](https://arxiv.org/pdf/2505.14157)  

**Abstract**: This paper investigates prior prompt engineering (pPE) in the context of reinforcement fine-tuning (RFT), where language models (LMs) are incentivized to exhibit behaviors that maximize performance through reward signals. While existing RFT research has primarily focused on algorithms, reward shaping, and data curation, the design of the prior prompt--the instructions prepended to queries during training to elicit behaviors such as step-by-step reasoning--remains underexplored. We investigate whether different pPE approaches can guide LMs to internalize distinct behaviors after RFT. Inspired by inference-time prompt engineering (iPE), we translate five representative iPE strategies--reasoning, planning, code-based reasoning, knowledge recall, and null-example utilization--into corresponding pPE approaches. We experiment with Qwen2.5-7B using each of the pPE approaches, then evaluate performance on in-domain and out-of-domain benchmarks (e.g., AIME2024, HumanEval+, and GPQA-Diamond). Our results show that all pPE-trained models surpass their iPE-prompted counterparts, with the null-example pPE approach achieving the largest average performance gain and the highest improvement on AIME2024 and GPQA-Diamond, surpassing the commonly used reasoning approach. Furthermore, by adapting a behavior-classification framework, we demonstrate that different pPE strategies instill distinct behavioral styles in the resulting models. These findings position pPE as a powerful yet understudied axis for RFT. 

---
# Enhancing Keyphrase Extraction from Academic Articles Using Section Structure Information 

**Authors**: Chengzhi Zhang, Xinyi Yan, Lei Zhao, Yingyi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14149)  

**Abstract**: The exponential increase in academic papers has significantly increased the time required for researchers to access relevant literature. Keyphrase Extraction (KPE) offers a solution to this situation by enabling researchers to efficiently retrieve relevant literature. The current study on KPE from academic articles aims to improve the performance of extraction models through innovative approaches using Title and Abstract as input corpora. However, the semantic richness of keywords is significantly constrained by the length of the abstract. While full-text-based KPE can address this issue, it simultaneously introduces noise, which significantly diminishes KPE performance. To address this issue, this paper utilized the structural features and section texts obtained from the section structure information of academic articles to extract keyphrase from academic papers. The approach consists of two main parts: (1) exploring the effect of seven structural features on KPE models, and (2) integrating the extraction results from all section texts used as input corpora for KPE models via a keyphrase integration algorithm to obtain the keyphrase integration result. Furthermore, this paper also examined the effect of the classification quality of section structure on the KPE performance. The results show that incorporating structural features improves KPE performance, though different features have varying effects on model efficacy. The keyphrase integration approach yields the best performance, and the classification quality of section structure can affect KPE performance. These findings indicate that using the section structure information of academic articles contributes to effective KPE from academic articles. The code and dataset supporting this study are available at this https URL. 

---
# Texts or Images? A Fine-grained Analysis on the Effectiveness of Input Representations and Models for Table Question Answering 

**Authors**: Wei Zhou, Mohsen Mesgar, Heike Adel, Annemarie Friedrich  

**Link**: [PDF](https://arxiv.org/pdf/2505.14131)  

**Abstract**: In table question answering (TQA), tables are encoded as either texts or images. Prior work suggests that passing images of tables to multi-modal large language models (MLLMs) performs comparably to or even better than using textual input with large language models (LLMs). However, the lack of controlled setups limits fine-grained distinctions between these approaches. In this paper, we conduct the first controlled study on the effectiveness of several combinations of table representations and models from two perspectives: question complexity and table size. We build a new benchmark based on existing TQA datasets. In a systematic analysis of seven pairs of MLLMs and LLMs, we find that the best combination of table representation and model varies across setups. We propose FRES, a method selecting table representations dynamically, and observe a 10% average performance improvement compared to using both representations indiscriminately. 

---
# Probing BERT for German Compound Semantics 

**Authors**: Filip Miletić, Aaron Schmid, Sabine Schulte im Walde  

**Link**: [PDF](https://arxiv.org/pdf/2505.14130)  

**Abstract**: This paper investigates the extent to which pretrained German BERT encodes knowledge of noun compound semantics. We comprehensively vary combinations of target tokens, layers, and cased vs. uncased models, and evaluate them by predicting the compositionality of 868 gold standard compounds. Looking at representational patterns within the transformer architecture, we observe trends comparable to equivalent prior work on English, with compositionality information most easily recoverable in the early layers. However, our strongest results clearly lag behind those reported for English, suggesting an inherently more difficult task in German. This may be due to the higher productivity of compounding in German than in English and the associated increase in constituent-level ambiguity, including in our target compound set. 

---
# Self-Reasoning Language Models: Unfold Hidden Reasoning Chains with Few Reasoning Catalyst 

**Authors**: Hongru Wang, Deng Cai, Wanjun Zhong, Shijue Huang, Jeff Z. Pan, Zeming Liu, Kam-Fai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2505.14116)  

**Abstract**: Inference-time scaling has attracted much attention which significantly enhance the performance of Large Language Models (LLMs) in complex reasoning tasks by increasing the length of Chain-of-Thought. These longer intermediate reasoning rationales embody various meta-reasoning skills in human cognition, such as reflection and decomposition, being difficult to create and acquire. In this work, we introduce \textit{Self-Reasoning Language Model} (SRLM), where the model itself can synthesize longer CoT data and iteratively improve performance through self-training. By incorporating a few demonstration examples (i.e., 1,000 samples) on how to unfold hidden reasoning chains from existing responses, which act as a reasoning catalyst, we demonstrate that SRLM not only enhances the model's initial performance but also ensures more stable and consistent improvements in subsequent iterations. Our proposed SRLM achieves an average absolute improvement of more than $+2.5$ points across five reasoning tasks: MMLU, GSM8K, ARC-C, HellaSwag, and BBH on two backbone models. Moreover, it brings more improvements with more times of sampling during inference, such as absolute $+7.89$ average improvement with $64$ sampling times, revealing the in-depth, diverse and creative reasoning paths in SRLM against the strong baseline. 

---
# Invisible Entropy: Towards Safe and Efficient Low-Entropy LLM Watermarking 

**Authors**: Tianle Gu, Zongqi Wang, Kexin Huang, Yuanqi Yao, Xiangliang Zhang, Yujiu Yang, Xiuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.14112)  

**Abstract**: Logit-based LLM watermarking traces and verifies AI-generated content by maintaining green and red token lists and increasing the likelihood of green tokens during generation. However, it fails in low-entropy scenarios, where predictable outputs make green token selection difficult without disrupting natural text flow. Existing approaches address this by assuming access to the original LLM to calculate entropy and selectively watermark high-entropy tokens. However, these methods face two major challenges: (1) high computational costs and detection delays due to reliance on the original LLM, and (2) potential risks of model leakage. To address these limitations, we propose Invisible Entropy (IE), a watermarking paradigm designed to enhance both safety and efficiency. Instead of relying on the original LLM, IE introduces a lightweight feature extractor and an entropy tagger to predict whether the entropy of the next token is high or low. Furthermore, based on theoretical analysis, we develop a threshold navigator that adaptively sets entropy thresholds. It identifies a threshold where the watermark ratio decreases as the green token count increases, enhancing the naturalness of the watermarked text and improving detection robustness. Experiments on HumanEval and MBPP datasets demonstrate that IE reduces parameter size by 99\% while achieving performance on par with state-of-the-art methods. Our work introduces a safe and efficient paradigm for low-entropy watermarking. this https URL this https URL 

---
# DiagnosisArena: Benchmarking Diagnostic Reasoning for Large Language Models 

**Authors**: Yakun Zhu, Zhongzhen Huang, Linjie Mu, Yutong Huang, Wei Nie, Shaoting Zhang, Pengfei Liu, Xiaofan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14107)  

**Abstract**: The emergence of groundbreaking large language models capable of performing complex reasoning tasks holds significant promise for addressing various scientific challenges, including those arising in complex clinical scenarios. To enable their safe and effective deployment in real-world healthcare settings, it is urgently necessary to benchmark the diagnostic capabilities of current models systematically. Given the limitations of existing medical benchmarks in evaluating advanced diagnostic reasoning, we present DiagnosisArena, a comprehensive and challenging benchmark designed to rigorously assess professional-level diagnostic competence. DiagnosisArena consists of 1,113 pairs of segmented patient cases and corresponding diagnoses, spanning 28 medical specialties, deriving from clinical case reports published in 10 top-tier medical journals. The benchmark is developed through a meticulous construction pipeline, involving multiple rounds of screening and review by both AI systems and human experts, with thorough checks conducted to prevent data leakage. Our study reveals that even the most advanced reasoning models, o3-mini, o1, and DeepSeek-R1, achieve only 45.82%, 31.09%, and 17.79% accuracy, respectively. This finding highlights a significant generalization bottleneck in current large language models when faced with clinical diagnostic reasoning challenges. Through DiagnosisArena, we aim to drive further advancements in AIs diagnostic reasoning capabilities, enabling more effective solutions for real-world clinical diagnostic challenges. We provide the benchmark and evaluation tools for further research and development this https URL. 

---
# A Personalized Conversational Benchmark: Towards Simulating Personalized Conversations 

**Authors**: Li Li, Peilin Cai, Ryan A. Rossi, Franck Dernoncourt, Branislav Kveton, Junda Wu, Tong Yu, Linxin Song, Tiankai Yang, Yuehan Qin, Nesreen K. Ahmed, Samyadeep Basu, Subhojyoti Mukherjee, Ruiyi Zhang, Zhengmian Hu, Bo Ni, Yuxiao Zhou, Zichao Wang, Yue Huang, Yu Wang, Xiangliang Zhang, Philip S. Yu, Xiyang Hu, Yue Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.14106)  

**Abstract**: We present PersonaConvBench, a large-scale benchmark for evaluating personalized reasoning and generation in multi-turn conversations with large language models (LLMs). Unlike existing work that focuses on either personalization or conversational structure in isolation, PersonaConvBench integrates both, offering three core tasks: sentence classification, impact regression, and user-centric text generation across ten diverse Reddit-based domains. This design enables systematic analysis of how personalized conversational context shapes LLM outputs in realistic multi-user scenarios. We benchmark several commercial and open-source LLMs under a unified prompting setup and observe that incorporating personalized history yields substantial performance improvements, including a 198 percent relative gain over the best non-conversational baseline in sentiment classification. By releasing PersonaConvBench with evaluations and code, we aim to support research on LLMs that adapt to individual styles, track long-term context, and produce contextually rich, engaging responses. 

---
# Legal Rule Induction: Towards Generalizable Principle Discovery from Analogous Judicial Precedents 

**Authors**: Wei Fan, Tianshi Zheng, Yiran Hu, Zheye Deng, Weiqi Wang, Baixuan Xu, Chunyang Li, Haoran Li, Weixing Shen, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.14104)  

**Abstract**: Legal rules encompass not only codified statutes but also implicit adjudicatory principles derived from precedents that contain discretionary norms, social morality, and policy. While computational legal research has advanced in applying established rules to cases, inducing legal rules from judicial decisions remains understudied, constrained by limitations in model inference efficacy and symbolic reasoning capability. The advent of Large Language Models (LLMs) offers unprecedented opportunities for automating the extraction of such latent principles, yet progress is stymied by the absence of formal task definitions, benchmark datasets, and methodologies. To address this gap, we formalize Legal Rule Induction (LRI) as the task of deriving concise, generalizable doctrinal rules from sets of analogous precedents, distilling their shared preconditions, normative behaviors, and legal consequences. We introduce the first LRI benchmark, comprising 5,121 case sets (38,088 Chinese cases in total) for model tuning and 216 expert-annotated gold test sets. Experimental results reveal that: 1) State-of-the-art LLMs struggle with over-generalization and hallucination; 2) Training on our dataset markedly enhances LLMs capabilities in capturing nuanced rule patterns across similar cases. 

---
# MultiHal: Multilingual Dataset for Knowledge-Graph Grounded Evaluation of LLM Hallucinations 

**Authors**: Ernests Lavrinovics, Russa Biswas, Katja Hose, Johannes Bjerva  

**Link**: [PDF](https://arxiv.org/pdf/2505.14101)  

**Abstract**: Large Language Models (LLMs) have inherent limitations of faithfulness and factuality, commonly referred to as hallucinations. Several benchmarks have been developed that provide a test bed for factuality evaluation within the context of English-centric datasets, while relying on supplementary informative context like web links or text passages but ignoring the available structured factual resources. To this end, Knowledge Graphs (KGs) have been identified as a useful aid for hallucination mitigation, as they provide a structured way to represent the facts about entities and their relations with minimal linguistic overhead. We bridge the lack of KG paths and multilinguality for factual language modeling within the existing hallucination evaluation benchmarks and propose a KG-based multilingual, multihop benchmark called \textbf{MultiHal} framed for generative text evaluation. As part of our data collection pipeline, we mined 140k KG-paths from open-domain KGs, from which we pruned noisy KG-paths, curating a high-quality subset of 25.9k. Our baseline evaluation shows an absolute scale increase by approximately 0.12 to 0.36 points for the semantic similarity score in KG-RAG over vanilla QA across multiple languages and multiple models, demonstrating the potential of KG integration. We anticipate MultiHal will foster future research towards several graph-based hallucination mitigation and fact-checking tasks. 

---
# Beyond Chains: Bridging Large Language Models and Knowledge Bases in Complex Question Answering 

**Authors**: Yihua Zhu, Qianying Liu, Akiko Aizawa, Hidetoshi Shimodaira  

**Link**: [PDF](https://arxiv.org/pdf/2505.14099)  

**Abstract**: Knowledge Base Question Answering (KBQA) aims to answer natural language questions using structured knowledge from KBs. While LLM-only approaches offer generalization, they suffer from outdated knowledge, hallucinations, and lack of transparency. Chain-based KG-RAG methods address these issues by incorporating external KBs, but are limited to simple chain-structured questions due to the absence of planning and logical structuring. Inspired by semantic parsing methods, we propose PDRR: a four-stage framework consisting of Predict, Decompose, Retrieve, and Reason. Our method first predicts the question type and decomposes the question into structured triples. Then retrieves relevant information from KBs and guides the LLM as an agent to reason over and complete the decomposed triples. Experimental results demonstrate that PDRR consistently outperforms existing methods across various LLM backbones and achieves superior performance on both chain-structured and non-chain complex questions. 

---
# Gender Trouble in Language Models: An Empirical Audit Guided by Gender Performativity Theory 

**Authors**: Franziska Sofia Hafner, Ana Valdivia, Luc Rocher  

**Link**: [PDF](https://arxiv.org/pdf/2505.14080)  

**Abstract**: Language models encode and subsequently perpetuate harmful gendered stereotypes. Research has succeeded in mitigating some of these harms, e.g. by dissociating non-gendered terms such as occupations from gendered terms such as 'woman' and 'man'. This approach, however, remains superficial given that associations are only one form of prejudice through which gendered harms arise. Critical scholarship on gender, such as gender performativity theory, emphasizes how harms often arise from the construction of gender itself, such as conflating gender with biological sex. In language models, these issues could lead to the erasure of transgender and gender diverse identities and cause harms in downstream applications, from misgendering users to misdiagnosing patients based on wrong assumptions about their anatomy.
For FAccT research on gendered harms to go beyond superficial linguistic associations, we advocate for a broader definition of 'gender bias' in language models. We operationalize insights on the construction of gender through language from gender studies literature and then empirically test how 16 language models of different architectures, training datasets, and model sizes encode gender. We find that language models tend to encode gender as a binary category tied to biological sex, and that gendered terms that do not neatly fall into one of these binary categories are erased and pathologized. Finally, we show that larger models, which achieve better results on performance benchmarks, learn stronger associations between gender and sex, further reinforcing a narrow understanding of gender. Our findings lead us to call for a re-evaluation of how gendered harms in language models are defined and addressed. 

---
# BAR: A Backward Reasoning based Agent for Complex Minecraft Tasks 

**Authors**: Weihong Du, Wenrui Liao, Binyu Yan, Hongru Liang, Anthony G. Cohn, Wenqiang Lei  

**Link**: [PDF](https://arxiv.org/pdf/2505.14079)  

**Abstract**: Large language model (LLM) based agents have shown great potential in following human instructions and automatically completing various tasks. To complete a task, the agent needs to decompose it into easily executed steps by planning. Existing studies mainly conduct the planning by inferring what steps should be executed next starting from the agent's initial state. However, this forward reasoning paradigm doesn't work well for complex tasks. We propose to study this issue in Minecraft, a virtual environment that simulates complex tasks based on real-world scenarios. We believe that the failure of forward reasoning is caused by the big perception gap between the agent's initial state and task goal. To this end, we leverage backward reasoning and make the planning starting from the terminal state, which can directly achieve the task goal in one step. Specifically, we design a BAckward Reasoning based agent (BAR). It is equipped with a recursive goal decomposition module, a state consistency maintaining module and a stage memory module to make robust, consistent, and efficient planning starting from the terminal state. Experimental results demonstrate the superiority of BAR over existing methods and the effectiveness of proposed modules. 

---
# Enhancing LLMs via High-Knowledge Data Selection 

**Authors**: Feiyu Duan, Xuemiao Zhang, Sirui Wang, Haoran Que, Yuqi Liu, Wenge Rong, Xunliang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2505.14070)  

**Abstract**: The performance of Large Language Models (LLMs) is intrinsically linked to the quality of its training data. Although several studies have proposed methods for high-quality data selection, they do not consider the importance of knowledge richness in text corpora. In this paper, we propose a novel and gradient-free High-Knowledge Scorer (HKS) to select high-quality data from the dimension of knowledge, to alleviate the problem of knowledge scarcity in the pre-trained corpus. We propose a comprehensive multi-domain knowledge element pool and introduce knowledge density and coverage as metrics to assess the knowledge content of the text. Based on this, we propose a comprehensive knowledge scorer to select data with intensive knowledge, which can also be utilized for domain-specific high-knowledge data selection by restricting knowledge elements to the specific domain. We train models on a high-knowledge bilingual dataset, and experimental results demonstrate that our scorer improves the model's performance in knowledge-intensive and general comprehension tasks, and is effective in enhancing both the generic and domain-specific capabilities of the model. 

---
# Improved Methods for Model Pruning and Knowledge Distillation 

**Authors**: Wei Jiang, Anying Fu, Youling Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14052)  

**Abstract**: Model pruning is a performance optimization technique for large language models like R1 or o3-mini. However, existing pruning methods often lead to significant performance degradation or require extensive retraining and fine-tuning. This technique aims to identify and remove neurons, connections unlikely leading to the contribution during the human-computer interaction phase. Our goal is to obtain a much smaller and faster knowledge distilled model that can quickly generate content almost as good as those of the unpruned ones. We propose MAMA Pruning, short for Movement and Magnitude Analysis, an improved pruning method that effectively reduces model size and computational complexity while maintaining performance comparable to the original unpruned model even at extreme pruned levels. The improved method is based on weights, bias fixed in the pre-training phase and GRPO rewards verified during the post-training phase as our novel pruning indicators. Preliminary experimental results show that our method outperforms and be comparable to state-of-the-art methods across various pruning levels and different downstream computational linguistics tasks. 

---
# From Unaligned to Aligned: Scaling Multilingual LLMs with Multi-Way Parallel Corpora 

**Authors**: Yingli Shen, Wen Lai, Shuo Wang, Kangyang Luo, Alexander Fraser, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.14045)  

**Abstract**: Continued pretraining and instruction tuning on large-scale multilingual data have proven to be effective in scaling large language models (LLMs) to low-resource languages. However, the unaligned nature of such data limits its ability to effectively capture cross-lingual semantics. In contrast, multi-way parallel data, where identical content is aligned across multiple languages, provides stronger cross-lingual consistency and offers greater potential for improving multilingual performance. In this paper, we introduce a large-scale, high-quality multi-way parallel corpus, TED2025, based on TED Talks. The corpus spans 113 languages, with up to 50 languages aligned in parallel, ensuring extensive multilingual coverage. Using this dataset, we investigate best practices for leveraging multi-way parallel data to enhance LLMs, including strategies for continued pretraining, instruction tuning, and the analysis of key influencing factors. Experiments on six multilingual benchmarks show that models trained on multiway parallel data consistently outperform those trained on unaligned multilingual data. 

---
# AUTOLAW: Enhancing Legal Compliance in Large Language Models via Case Law Generation and Jury-Inspired Deliberation 

**Authors**: Tai D. Nguyen, Long H. Pham, Jun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.14015)  

**Abstract**: The rapid advancement of domain-specific large language models (LLMs) in fields like law necessitates frameworks that account for nuanced regional legal distinctions, which are critical for ensuring compliance and trustworthiness. Existing legal evaluation benchmarks often lack adaptability and fail to address diverse local contexts, limiting their utility in dynamically evolving regulatory landscapes. To address these gaps, we propose AutoLaw, a novel violation detection framework that combines adversarial data generation with a jury-inspired deliberation process to enhance legal compliance of LLMs. Unlike static approaches, AutoLaw dynamically synthesizes case law to reflect local regulations and employs a pool of LLM-based "jurors" to simulate judicial decision-making. Jurors are ranked and selected based on synthesized legal expertise, enabling a deliberation process that minimizes bias and improves detection accuracy. Evaluations across three benchmarks: Law-SG, Case-SG (legality), and Unfair-TOS (policy), demonstrate AutoLaw's effectiveness: adversarial data generation improves LLM discrimination, while the jury-based voting strategy significantly boosts violation detection rates. Our results highlight the framework's ability to adaptively probe legal misalignments and deliver reliable, context-aware judgments, offering a scalable solution for evaluating and enhancing LLMs in legally sensitive applications. 

---
# Activation-Guided Consensus Merging for Large Language Models 

**Authors**: Yuxuan Yao, Shuqi Liu, Zehua Liu, Qintong Li, Mingyang Liu, Xiongwei Han, Zhijiang Guo, Han Wu, Linqi Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.14009)  

**Abstract**: Recent research has increasingly focused on reconciling the reasoning capabilities of System 2 with the efficiency of System 1. While existing training-based and prompt-based approaches face significant challenges in terms of efficiency and stability, model merging emerges as a promising strategy to integrate the diverse capabilities of different Large Language Models (LLMs) into a unified model. However, conventional model merging methods often assume uniform importance across layers, overlooking the functional heterogeneity inherent in neural components. To address this limitation, we propose \textbf{A}ctivation-Guided \textbf{C}onsensus \textbf{M}erging (\textbf{ACM}), a plug-and-play merging framework that determines layer-specific merging coefficients based on mutual information between activations of pre-trained and fine-tuned models. ACM effectively preserves task-specific capabilities without requiring gradient computations or additional training. Extensive experiments on Long-to-Short (L2S) and general merging tasks demonstrate that ACM consistently outperforms all baseline methods. For instance, in the case of Qwen-7B models, TIES-Merging equipped with ACM achieves a \textbf{55.3\%} reduction in response length while simultaneously improving reasoning accuracy by \textbf{1.3} points. We submit the code with the paper for reproducibility, and it will be publicly available. 

---
# Social Sycophancy: A Broader Understanding of LLM Sycophancy 

**Authors**: Myra Cheng, Sunny Yu, Cinoo Lee, Pranav Khadpe, Lujain Ibrahim, Dan Jurafsky  

**Link**: [PDF](https://arxiv.org/pdf/2505.13995)  

**Abstract**: A serious risk to the safety and utility of LLMs is sycophancy, i.e., excessive agreement with and flattery of the user. Yet existing work focuses on only one aspect of sycophancy: agreement with users' explicitly stated beliefs that can be compared to a ground truth. This overlooks forms of sycophancy that arise in ambiguous contexts such as advice and support-seeking, where there is no clear ground truth, yet sycophancy can reinforce harmful implicit assumptions, beliefs, or actions. To address this gap, we introduce a richer theory of social sycophancy in LLMs, characterizing sycophancy as the excessive preservation of a user's face (the positive self-image a person seeks to maintain in an interaction). We present ELEPHANT, a framework for evaluating social sycophancy across five face-preserving behaviors (emotional validation, moral endorsement, indirect language, indirect action, and accepting framing) on two datasets: open-ended questions (OEQ) and Reddit's r/AmITheAsshole (AITA). Across eight models, we show that LLMs consistently exhibit high rates of social sycophancy: on OEQ, they preserve face 47% more than humans, and on AITA, they affirm behavior deemed inappropriate by crowdsourced human judgments in 42% of cases. We further show that social sycophancy is rewarded in preference datasets and is not easily mitigated. Our work provides theoretical grounding and empirical tools (datasets and code) for understanding and addressing this under-recognized but consequential issue. 

---
# DecIF: Improving Instruction-Following through Meta-Decomposition 

**Authors**: Tingfeng Hui, Pengyu Zhu, Bowen Ping, Ling Tang, Yaqi Zhang, Sen Su  

**Link**: [PDF](https://arxiv.org/pdf/2505.13990)  

**Abstract**: Instruction-following has emerged as a crucial capability for large language models (LLMs). However, existing approaches often rely on pre-existing documents or external resources to synthesize instruction-following data, which limits their flexibility and generalizability. In this paper, we introduce DecIF, a fully autonomous, meta-decomposition guided framework that generates diverse and high-quality instruction-following data using only LLMs. DecIF is grounded in the principle of decomposition. For instruction generation, we guide LLMs to iteratively produce various types of meta-information, which are then combined with response constraints to form well-structured and semantically rich instructions. We further utilize LLMs to detect and resolve potential inconsistencies within the generated instructions. Regarding response generation, we decompose each instruction into atomic-level evaluation criteria, enabling rigorous validation and the elimination of inaccurate instruction-response pairs. Extensive experiments across a wide range of scenarios and settings demonstrate DecIF's superior performance on instruction-following tasks. Further analysis highlights its strong flexibility, scalability, and generalizability in automatically synthesizing high-quality instruction data. 

---
# The Hallucination Tax of Reinforcement Finetuning 

**Authors**: Linxin Song, Taiwei Shi, Jieyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.13988)  

**Abstract**: Reinforcement finetuning (RFT) has become a standard approach for enhancing the reasoning capabilities of large language models (LLMs). However, its impact on model trustworthiness remains underexplored. In this work, we identify and systematically study a critical side effect of RFT, which we term the hallucination tax: a degradation in refusal behavior causing models to produce hallucinated answers to unanswerable questions confidently. To investigate this, we introduce SUM (Synthetic Unanswerable Math), a high-quality dataset of unanswerable math problems designed to probe models' ability to recognize an unanswerable question by reasoning from the insufficient or ambiguous information. Our results show that standard RFT training could reduce model refusal rates by more than 80%, which significantly increases model's tendency to hallucinate. We further demonstrate that incorporating just 10% SUM during RFT substantially restores appropriate refusal behavior, with minimal accuracy trade-offs on solvable tasks. Crucially, this approach enables LLMs to leverage inference-time compute to reason about their own uncertainty and knowledge boundaries, improving generalization not only to out-of-domain math problems but also to factual question answering tasks. 

---
# Mixed Signals: Understanding Model Disagreement in Multimodal Empathy Detection 

**Authors**: Maya Srikanth, Run Chen, Julia Hirschberg  

**Link**: [PDF](https://arxiv.org/pdf/2505.13979)  

**Abstract**: Multimodal models play a key role in empathy detection, but their performance can suffer when modalities provide conflicting cues. To understand these failures, we examine cases where unimodal and multimodal predictions diverge. Using fine-tuned models for text, audio, and video, along with a gated fusion model, we find that such disagreements often reflect underlying ambiguity, as evidenced by annotator uncertainty. Our analysis shows that dominant signals in one modality can mislead fusion when unsupported by others. We also observe that humans, like models, do not consistently benefit from multimodal input. These insights position disagreement as a useful diagnostic signal for identifying challenging examples and improving empathy system robustness. 

---
# DRP: Distilled Reasoning Pruning with Skill-aware Step Decomposition for Efficient Large Reasoning Models 

**Authors**: Yuxuan Jiang, Dawei Li, Frank Ferraro  

**Link**: [PDF](https://arxiv.org/pdf/2505.13975)  

**Abstract**: While Large Reasoning Models (LRMs) have demonstrated success in complex reasoning tasks through long chain-of-thought (CoT) reasoning, their inference often involves excessively verbose reasoning traces, resulting in substantial inefficiency. To address this, we propose Distilled Reasoning Pruning (DRP), a hybrid framework that combines inference-time pruning with tuning-based distillation, two widely used strategies for efficient reasoning. DRP uses a teacher model to perform skill-aware step decomposition and content pruning, and then distills the pruned reasoning paths into a student model, enabling it to reason both efficiently and accurately. Across several challenging mathematical reasoning datasets, we find that models trained with DRP achieve substantial improvements in token efficiency without sacrificing accuracy. Specifically, DRP reduces average token usage on GSM8K from 917 to 328 while improving accuracy from 91.7% to 94.1%, and achieves a 43% token reduction on AIME with no performance drop. Further analysis shows that aligning the reasoning structure of training CoTs with the student's reasoning capacity is critical for effective knowledge transfer and performance gains. 

---
# Toward Effective Reinforcement Learning Fine-Tuning for Medical VQA in Vision-Language Models 

**Authors**: Wenhui Zhu, Xuanzhao Dong, Xin Li, Peijie Qiu, Xiwen Chen, Abolfazl Razi, Aris Sotiras, Yi Su, Yalin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13973)  

**Abstract**: Recently, reinforcement learning (RL)-based tuning has shifted the trajectory of Multimodal Large Language Models (MLLMs), particularly following the introduction of Group Relative Policy Optimization (GRPO). However, directly applying it to medical tasks remains challenging for achieving clinically grounded model behavior. Motivated by the need to align model response with clinical expectations, we investigate four critical dimensions that affect the effectiveness of RL-based tuning in medical visual question answering (VQA): base model initialization strategy, the role of medical semantic alignment, the impact of length-based rewards on long-chain reasoning, and the influence of bias. We conduct extensive experiments to analyze these factors for medical MLLMs, providing new insights into how models are domain-specifically fine-tuned. Additionally, our results also demonstrate that GRPO-based RL tuning consistently outperforms standard supervised fine-tuning (SFT) in both accuracy and reasoning quality. 

---
# Truth or Twist? Optimal Model Selection for Reliable Label Flipping Evaluation in LLM-based Counterfactuals 

**Authors**: Qianli Wang, Van Bach Nguyen, Nils Feldhus, Luis Felipe Villa-Arenas, Christin Seifert, Sebastian Möller, Vera Schmitt  

**Link**: [PDF](https://arxiv.org/pdf/2505.13972)  

**Abstract**: Counterfactual examples are widely employed to enhance the performance and robustness of large language models (LLMs) through counterfactual data augmentation (CDA). However, the selection of the judge model used to evaluate label flipping, the primary metric for assessing the validity of generated counterfactuals for CDA, yields inconsistent results. To decipher this, we define four types of relationships between the counterfactual generator and judge models. Through extensive experiments involving two state-of-the-art LLM-based methods, three datasets, five generator models, and 15 judge models, complemented by a user study (n = 90), we demonstrate that judge models with an independent, non-fine-tuned relationship to the generator model provide the most reliable label flipping evaluations. Relationships between the generator and judge models, which are closely aligned with the user study for CDA, result in better model performance and robustness. Nevertheless, we find that the gap between the most effective judge models and the results obtained from the user study remains considerably large. This suggests that a fully automated pipeline for CDA may be inadequate and requires human intervention. 

---
# CAFES: A Collaborative Multi-Agent Framework for Multi-Granular Multimodal Essay Scoring 

**Authors**: Jiamin Su, Yibo Yan, Zhuoran Gao, Han Zhang, Xiang Liu, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13965)  

**Abstract**: Automated Essay Scoring (AES) is crucial for modern education, particularly with the increasing prevalence of multimodal assessments. However, traditional AES methods struggle with evaluation generalizability and multimodal perception, while even recent Multimodal Large Language Model (MLLM)-based approaches can produce hallucinated justifications and scores misaligned with human judgment. To address the limitations, we introduce CAFES, the first collaborative multi-agent framework specifically designed for AES. It orchestrates three specialized agents: an Initial Scorer for rapid, trait-specific evaluations; a Feedback Pool Manager to aggregate detailed, evidence-grounded strengths; and a Reflective Scorer that iteratively refines scores based on this feedback to enhance human alignment. Extensive experiments, using state-of-the-art MLLMs, achieve an average relative improvement of 21% in Quadratic Weighted Kappa (QWK) against ground truth, especially for grammatical and lexical diversity. Our proposed CAFES framework paves the way for an intelligent multimodal AES system. The code will be available upon acceptance. 

---
# Through a Compressed Lens: Investigating the Impact of Quantization on LLM Explainability and Interpretability 

**Authors**: Qianli Wang, Mingyang Wang, Nils Feldhus, Simon Ostermann, Yuan Cao, Hinrich Schütze, Sebastian Möller, Vera Schmitt  

**Link**: [PDF](https://arxiv.org/pdf/2505.13963)  

**Abstract**: Quantization methods are widely used to accelerate inference and streamline the deployment of large language models (LLMs). While prior research has extensively investigated the degradation of various LLM capabilities due to quantization, its effects on model explainability and interpretability, which are crucial for understanding decision-making processes, remain unexplored. To address this gap, we conduct comprehensive experiments using three common quantization techniques at distinct bit widths, in conjunction with two explainability methods, counterfactual examples and natural language explanations, as well as two interpretability approaches, knowledge memorization analysis and latent multi-hop reasoning analysis. We complement our analysis with a thorough user study, evaluating selected explainability methods. Our findings reveal that, depending on the configuration, quantization can significantly impact model explainability and interpretability. Notably, the direction of this effect is not consistent, as it strongly depends on (1) the quantization method, (2) the explainability or interpretability approach, and (3) the evaluation protocol. In some settings, human evaluation shows that quantization degrades explainability, while in others, it even leads to improvements. Our work serves as a cautionary tale, demonstrating that quantization can unpredictably affect model transparency. This insight has important implications for deploying LLMs in applications where transparency is a critical requirement. 

---
# FlashThink: An Early Exit Method For Efficient Reasoning 

**Authors**: Guochao Jiang, Guofeng Quan, Zepeng Ding, Ziqin Luo, Dixuan Wang, Zheng Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13949)  

**Abstract**: Large Language Models (LLMs) have shown impressive performance in reasoning tasks. However, LLMs tend to generate excessively long reasoning content, leading to significant computational overhead. Our observations indicate that even on simple problems, LLMs tend to produce unnecessarily lengthy reasoning content, which is against intuitive expectations. Preliminary experiments show that at a certain point during the generation process, the model is already capable of producing the correct solution without completing the full reasoning content. Therefore, we consider that the reasoning process of the model can be exited early to achieve the purpose of efficient reasoning. We introduce a verification model that identifies the exact moment when the model can stop reasoning and still provide the correct answer. Comprehensive experiments on four different benchmarks demonstrate that our proposed method, FlashThink, effectively shortens the reasoning content while preserving the model accuracy. For the Deepseek-R1 and QwQ-32B models, we reduced the length of reasoning content by 77.04% and 77.47%, respectively, without reducing the accuracy. 

---
# Memory-Centric Embodied Question Answer 

**Authors**: Mingliang Zhai, Zhi Gao, Yuwei Wu, Yunde Jia  

**Link**: [PDF](https://arxiv.org/pdf/2505.13948)  

**Abstract**: Embodied Question Answering (EQA) requires agents to autonomously explore and understand the environment to answer context-dependent questions. Existing frameworks typically center around the planner, which guides the stopping module, memory module, and answering module for reasoning. In this paper, we propose a memory-centric EQA framework named MemoryEQA. Unlike planner-centric EQA models where the memory module cannot fully interact with other modules, MemoryEQA flexible feeds memory information into all modules, thereby enhancing efficiency and accuracy in handling complex tasks, such as those involving multiple targets across different regions. Specifically, we establish a multi-modal hierarchical memory mechanism, which is divided into global memory that stores language-enhanced scene maps, and local memory that retains historical observations and state information. When performing EQA tasks, the multi-modal large language model is leveraged to convert memory information into the required input formats for injection into different modules. To evaluate EQA models' memory capabilities, we constructed the MT-HM3D dataset based on HM3D, comprising 1,587 question-answer pairs involving multiple targets across various regions, which requires agents to maintain memory of exploration-acquired target information. Experimental results on HM-EQA, MT-HM3D, and OpenEQA demonstrate the effectiveness of our framework, where a 19.8% performance gain on MT-HM3D compared to baseline model further underscores memory capability's pivotal role in resolving complex tasks. 

---
# Towards Rehearsal-Free Continual Relation Extraction: Capturing Within-Task Variance with Adaptive Prompting 

**Authors**: Bao-Ngoc Dao, Quang Nguyen, Luyen Ngo Dinh, Minh Le, Nam Le, Linh Ngo Van  

**Link**: [PDF](https://arxiv.org/pdf/2505.13944)  

**Abstract**: Memory-based approaches have shown strong performance in Continual Relation Extraction (CRE). However, storing examples from previous tasks increases memory usage and raises privacy concerns. Recently, prompt-based methods have emerged as a promising alternative, as they do not rely on storing past samples. Despite this progress, current prompt-based techniques face several core challenges in CRE, particularly in accurately identifying task identities and mitigating catastrophic forgetting. Existing prompt selection strategies often suffer from inaccuracies, lack robust mechanisms to prevent forgetting in shared parameters, and struggle to handle both cross-task and within-task variations. In this paper, we propose WAVE++, a novel approach inspired by the connection between prefix-tuning and mixture of experts. Specifically, we introduce task-specific prompt pools that enhance flexibility and adaptability across diverse tasks while avoiding boundary-spanning risks; this design more effectively captures variations within each task and across tasks. To further refine relation classification, we incorporate label descriptions that provide richer, more global context, enabling the model to better distinguish among different relations. We also propose a training-free mechanism to improve task prediction during inference. Moreover, we integrate a generative model to consolidate prior knowledge within the shared parameters, thereby removing the need for explicit data storage. Extensive experiments demonstrate that WAVE++ outperforms state-of-the-art prompt-based and rehearsal-based methods, offering a more robust solution for continual relation extraction. Our code is publicly available at this https URL. 

---
# EEG-to-Text Translation: A Model for Deciphering Human Brain Activity 

**Authors**: Saydul Akbar Murad, Ashim Dahal, Nick Rahimi  

**Link**: [PDF](https://arxiv.org/pdf/2505.13936)  

**Abstract**: With the rapid advancement of large language models like Gemini, GPT, and others, bridging the gap between the human brain and language processing has become an important area of focus. To address this challenge, researchers have developed various models to decode EEG signals into text. However, these models still face significant performance limitations. To overcome these shortcomings, we propose a new model, R1 Translator, which aims to improve the performance of EEG-to-text decoding. The R1 Translator model combines a bidirectional LSTM encoder with a pretrained transformer-based decoder, utilizing EEG features to produce high-quality text outputs. The model processes EEG embeddings through the LSTM to capture sequential dependencies, which are then fed into the transformer decoder for effective text generation. The R1 Translator excels in ROUGE metrics, outperforming both T5 (previous research) and Brain Translator. Specifically, R1 achieves a ROUGE-1 score of 38.00% (P), which is up to 9% higher than T5 (34.89%) and 3% better than Brain (35.69%). It also leads in ROUGE-L, with a F1 score of 32.51%, outperforming T5 by 3% (29.67%) and Brain by 2% (30.38%). In terms of CER, R1 achieves a CER of 0.5795, which is 2% lower than T5 (0.5917) and 4% lower than Brain (0.6001). Additionally, R1 performs better in WER with a score of 0.7280, outperforming T5 by 4.3% (0.7610) and Brain by 3.6% (0.7553). Code is available at this https URL. 

---
# Word length predicts word order: "Min-max"-ing drives language evolution 

**Authors**: Hiram Ring  

**Link**: [PDF](https://arxiv.org/pdf/2505.13913)  

**Abstract**: Current theories of language propose an innate (Baker 2001; Chomsky 1981) or a functional (Greenberg 1963; Dryer 2007; Hawkins 2014) origin for the surface structures (i.e. word order) that we observe in languages of the world, while evolutionary modeling (Dunn et al. 2011) suggests that descent is the primary factor influencing such patterns. Although there are hypotheses for word order change from both innate and usage-based perspectives for specific languages and families, there are key disagreements between the two major proposals for mechanisms that drive the evolution of language more broadly (Wasow 2002; Levy 2008). This paper proposes a universal underlying mechanism for word order change based on a large tagged parallel dataset of over 1,500 languages representing 133 language families and 111 isolates. Results indicate that word class length is significantly correlated with word order crosslinguistically, but not in a straightforward manner, partially supporting opposing theories of processing, while at the same time predicting historical word order change in two different phylogenetic lines and explaining more variance than descent or language area in regression models. Such findings suggest an integrated "Min-Max" theory of language evolution driven by competing pressures of processing and information structure, aligning with recent efficiency-oriented (Levshina 2023) and information-theoretic proposals (Zaslavsky 2020; Tucker et al. 2025). 

---
# Cross-Linguistic Transfer in Multilingual NLP: The Role of Language Families and Morphology 

**Authors**: Ajitesh Bankula, Praney Bankula  

**Link**: [PDF](https://arxiv.org/pdf/2505.13908)  

**Abstract**: Cross-lingual transfer has become a crucial aspect of multilingual NLP, as it allows for models trained on resource-rich languages to be applied to low-resource languages more effectively. Recently massively multilingual pre-trained language models (e.g., mBERT, XLM-R) demonstrate strong zero-shot transfer capabilities[14] [13]. This paper investigates cross-linguistic transfer through the lens of language families and morphology. Investigating how language family proximity and morphological similarity affect performance across NLP tasks. We further discuss our results and how it relates to findings from recent literature. Overall, we compare multilingual model performance and review how linguistic distance metrics correlate with transfer outcomes. We also look into emerging approaches that integrate typological and morphological information into model pre-training to improve transfer to diverse languages[18] [19]. 

---
# Let's Verify Math Questions Step by Step 

**Authors**: Chengyu Shen, Zhen Hao Wong, Runming He, Hao Liang, Meiyi Qiang, Zimo Meng, Zhengyang Zhao, Bohan Zeng, Zhengzhou Zhu, Bin Cui, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13903)  

**Abstract**: Large Language Models (LLMs) have recently achieved remarkable progress in mathematical reasoning. To enable such capabilities, many existing works distill strong reasoning models into long chains of thought or design algorithms to construct high-quality math QA data for training. However, these efforts primarily focus on generating correct reasoning paths and answers, while largely overlooking the validity of the questions themselves. In this work, we propose Math Question Verification (MathQ-Verify), a novel five-stage pipeline designed to rigorously filter ill-posed or under-specified math problems. MathQ-Verify first performs format-level validation to remove redundant instructions and ensure that each question is syntactically well-formed. It then formalizes each question, decomposes it into atomic conditions, and verifies them against mathematical definitions. Next, it detects logical contradictions among these conditions, followed by a goal-oriented completeness check to ensure the question provides sufficient information for solving. To evaluate this task, we use existing benchmarks along with an additional dataset we construct, containing 2,147 math questions with diverse error types, each manually double-validated. Experiments show that MathQ-Verify achieves state-of-the-art performance across multiple benchmarks, improving the F1 score by up to 25 percentage points over the direct verification baseline. It further attains approximately 90% precision and 63% recall through a lightweight model voting scheme. MathQ-Verify offers a scalable and accurate solution for curating reliable mathematical datasets, reducing label noise and avoiding unnecessary computation on invalid questions. Our code and data are available at this https URL. 

---
# InfiGFusion: Graph-on-Logits Distillation via Efficient Gromov-Wasserstein for Model Fusion 

**Authors**: Yuanyi Wang, Zhaoyi Yan, Yiming Zhang, Qi Zhou, Yanggan Gu, Fei Wu, Hongxia Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13893)  

**Abstract**: Recent advances in large language models (LLMs) have intensified efforts to fuse heterogeneous open-source models into a unified system that inherits their complementary strengths. Existing logit-based fusion methods maintain inference efficiency but treat vocabulary dimensions independently, overlooking semantic dependencies encoded by cross-dimension interactions. These dependencies reflect how token types interact under a model's internal reasoning and are essential for aligning models with diverse generation behaviors. To explicitly model these dependencies, we propose \textbf{InfiGFusion}, the first structure-aware fusion framework with a novel \textit{Graph-on-Logits Distillation} (GLD) loss. Specifically, we retain the top-$k$ logits per output and aggregate their outer products across sequence positions to form a global co-activation graph, where nodes represent vocabulary channels and edges quantify their joint activations. To ensure scalability and efficiency, we design a sorting-based closed-form approximation that reduces the original $O(n^4)$ cost of Gromov-Wasserstein distance to $O(n \log n)$, with provable approximation guarantees. Experiments across multiple fusion settings show that GLD consistently improves fusion quality and stability. InfiGFusion outperforms SOTA models and fusion baselines across 11 benchmarks spanning reasoning, coding, and mathematics. It shows particular strength in complex reasoning tasks, with +35.6 improvement on Multistep Arithmetic and +37.06 on Causal Judgement over SFT, demonstrating superior multi-step and relational inference. 

---
# Mapping the Minds of LLMs: A Graph-Based Analysis of Reasoning LLM 

**Authors**: Zhen Xiong, Yujun Cai, Zhecheng Li, Yiwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13890)  

**Abstract**: Recent advances in test-time scaling have enabled Large Language Models (LLMs) to display sophisticated reasoning abilities via extended Chain-of-Thought (CoT) generation. Despite their potential, these Reasoning LLMs (RLMs) often demonstrate counterintuitive and unstable behaviors, such as performance degradation under few-shot prompting, that challenge our current understanding of RLMs. In this work, we introduce a unified graph-based analytical framework for better modeling the reasoning processes of RLMs. Our method first clusters long, verbose CoT outputs into semantically coherent reasoning steps, then constructs directed reasoning graphs to capture contextual and logical dependencies among these steps. Through comprehensive analysis across models and prompting regimes, we reveal that structural properties, such as exploration density, branching, and convergence ratios, strongly correlate with reasoning accuracy. Our findings demonstrate how prompting strategies substantially reshape the internal reasoning structure of RLMs, directly affecting task outcomes. The proposed framework not only enables quantitative evaluation of reasoning quality beyond conventional metrics but also provides practical insights for prompt engineering and the cognitive analysis of LLMs. Code and resources will be released to facilitate future research in this direction. 

---
# Code2Logic: Game-Code-Driven Data Synthesis for Enhancing VLMs General Reasoning 

**Authors**: Jingqi Tong, Jixin Tang, Hangcheng Li, Yurong Mou, Ming Zhang, Jun Zhao, Yanbo Wen, Fan Song, Jiahao Zhan, Yuyang Lu, Chaoran Tao, Zhiyuan Guo, Jizhou Yu, Tianhao Cheng, Changhao Jiang, Zhen Wang, Tao Liang, Zhihui Fei, Mingyang Wan, Guojun Ma, Weifeng Ge, Guanhua Chen, Tao Gui, Xipeng Qiu, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13886)  

**Abstract**: Visual-language Chain-of-Thought (CoT) data resources are relatively scarce compared to text-only counterparts, limiting the improvement of reasoning capabilities in Vision Language Models (VLMs). However, high-quality vision-language reasoning data is expensive and labor-intensive to annotate. To address this issue, we leverage a promising resource: game code, which naturally contains logical structures and state transition processes. Therefore, we propose Code2Logic, a novel game-code-driven approach for multimodal reasoning data synthesis. Our approach leverages Large Language Models (LLMs) to adapt game code, enabling automatic acquisition of reasoning processes and results through code execution. Using the Code2Logic approach, we developed the GameQA dataset to train and evaluate VLMs. GameQA is cost-effective and scalable to produce, challenging for state-of-the-art models, and diverse with 30 games and 158 tasks. Surprisingly, despite training solely on game data, VLMs demonstrated out of domain generalization, specifically Qwen2.5-VL-7B improving performance by 2.33\% across 7 diverse vision-language benchmarks. Our code and dataset are available at this https URL. 

---
# Reasoning Path Compression: Compressing Generation Trajectories for Efficient LLM Reasoning 

**Authors**: Jiwon Song, Dongwon Jo, Yulhwa Kim, Jae-Joon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.13866)  

**Abstract**: Recent reasoning-focused language models achieve high accuracy by generating lengthy intermediate reasoning paths before producing final answers. While this approach is effective in solving problems that require logical thinking, long reasoning paths significantly increase memory usage and throughput of token generation, limiting the practical deployment of such models. We propose Reasoning Path Compression (RPC), a training-free method that accelerates inference by leveraging the semantic sparsity of reasoning paths. RPC periodically compresses the KV cache by retaining KV cache that receive high importance score, which are computed using a selector window composed of recently generated queries. Experiments show that RPC improves generation throughput of QwQ-32B by up to 1.60$\times$ compared to the inference with full KV cache, with an accuracy drop of 1.2% on the AIME 2024 benchmark. Our findings demonstrate that semantic sparsity in reasoning traces can be effectively exploited for compression, offering a practical path toward efficient deployment of reasoning LLMs. Our code is available at this https URL. 

---
# Domain Gating Ensemble Networks for AI-Generated Text Detection 

**Authors**: Arihant Tripathi, Liam Dugan, Charis Gao, Maggie Huan, Emma Jin, Peter Zhang, David Zhang, Julia Zhao, Chris Callison-Burch  

**Link**: [PDF](https://arxiv.org/pdf/2505.13855)  

**Abstract**: As state-of-the-art language models continue to improve, the need for robust detection of machine-generated text becomes increasingly critical. However, current state-of-the-art machine text detectors struggle to adapt to new unseen domains and generative models. In this paper we present DoGEN (Domain Gating Ensemble Networks), a technique that allows detectors to adapt to unseen domains by ensembling a set of domain expert detector models using weights from a domain classifier. We test DoGEN on a wide variety of domains from leading benchmarks and find that it achieves state-of-the-art performance on in-domain detection while outperforming models twice its size on out-of-domain detection. We release our code and trained models to assist in future research in domain-adaptive AI detection. 

---
# Improve Language Model and Brain Alignment via Associative Memory 

**Authors**: Congchi Yin, Yongpeng Zhang, Xuyun Wen, Piji Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.13844)  

**Abstract**: Associative memory engages in the integration of relevant information for comprehension in the human cognition system. In this work, we seek to improve alignment between language models and human brain while processing speech information by integrating associative memory. After verifying the alignment between language model and brain by mapping language model activations to brain activity, the original text stimuli expanded with simulated associative memory are regarded as input to computational language models. We find the alignment between language model and brain is improved in brain regions closely related to associative memory processing. We also demonstrate large language models after specific supervised fine-tuning better align with brain response, by building the \textit{Association} dataset containing 1000 samples of stories, with instructions encouraging associative memory as input and associated content as output. 

---
# EfficientLLM: Efficiency in Large Language Models 

**Authors**: Zhengqing Yuan, Weixiang Sun, Yixin Liu, Huichi Zhou, Rong Zhou, Yiyang Li, Zheyuan Zhang, Wei Song, Yue Huang, Haolong Jia, Keerthiram Murugesan, Yu Wang, Lifang He, Jianfeng Gao, Lichao Sun, Yanfang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2505.13840)  

**Abstract**: Large Language Models (LLMs) have driven significant progress, yet their growing parameter counts and context windows incur prohibitive compute, energy, and monetary costs. We introduce EfficientLLM, a novel benchmark and the first comprehensive empirical study evaluating efficiency techniques for LLMs at scale. Conducted on a production-class cluster (48xGH200, 8xH200 GPUs), our study systematically explores three key axes: (1) architecture pretraining (efficient attention variants: MQA, GQA, MLA, NSA; sparse Mixture-of-Experts (MoE)), (2) fine-tuning (parameter-efficient methods: LoRA, RSLoRA, DoRA), and (3) inference (quantization methods: int4, float16). We define six fine-grained metrics (Memory Utilization, Compute Utilization, Latency, Throughput, Energy Consumption, Compression Rate) to capture hardware saturation, latency-throughput balance, and carbon cost. Evaluating over 100 model-technique pairs (0.5B-72B parameters), we derive three core insights: (i) Efficiency involves quantifiable trade-offs: no single method is universally optimal; e.g., MoE reduces FLOPs and improves accuracy but increases VRAM by 40%, while int4 quantization cuts memory/energy by up to 3.9x at a 3-5% accuracy drop. (ii) Optima are task- and scale-dependent: MQA offers optimal memory-latency trade-offs for constrained devices, MLA achieves lowest perplexity for quality-critical tasks, and RSLoRA surpasses LoRA efficiency only beyond 14B parameters. (iii) Techniques generalize across modalities: we extend evaluations to Large Vision Models (Stable Diffusion 3.5, Wan 2.1) and Vision-Language Models (Qwen2.5-VL), confirming effective transferability. By open-sourcing datasets, evaluation pipelines, and leaderboards, EfficientLLM provides essential guidance for researchers and engineers navigating the efficiency-performance landscape of next-generation foundation models. 

---
# Interpretable Traces, Unexpected Outcomes: Investigating the Disconnect in Trace-Based Knowledge Distillation 

**Authors**: Siddhant Bhambri, Upasana Biswas, Subbarao Kambhampati  

**Link**: [PDF](https://arxiv.org/pdf/2505.13792)  

**Abstract**: Question Answering (QA) poses a challenging and critical problem, particularly in today's age of interactive dialogue systems such as ChatGPT, Perplexity, Microsoft Copilot, etc. where users demand both accuracy and transparency in the model's outputs. Since smaller language models (SLMs) are computationally more efficient but often under-perform compared to larger models, Knowledge Distillation (KD) methods allow for finetuning these smaller models to improve their final performance. Lately, the intermediate tokens or the so called `reasoning' traces produced by Chain-of-Thought (CoT) or by reasoning models such as DeepSeek R1 are used as a training signal for KD. However, these reasoning traces are often verbose and difficult to interpret or evaluate. In this work, we aim to address the challenge of evaluating the faithfulness of these reasoning traces and their correlation with the final performance. To this end, we employ a KD method leveraging rule-based problem decomposition. This approach allows us to break down complex queries into structured sub-problems, generating interpretable traces whose correctness can be readily evaluated, even at inference time. Specifically, we demonstrate this approach on Open Book QA, decomposing the problem into a Classification step and an Information Retrieval step, thereby simplifying trace evaluation. Our SFT experiments with correct and incorrect traces on the CoTemp QA, Microsoft Machine Reading Comprehension QA, and Facebook bAbI QA datasets reveal the striking finding that correct traces do not necessarily imply that the model outputs the correct final solution. Similarly, we find a low correlation between correct final solutions and intermediate trace correctness. These results challenge the implicit assumption behind utilizing reasoning traces for improving SLMs' final performance via KD. 

---
# Krikri: Advancing Open Large Language Models for Greek 

**Authors**: Dimitris Roussis, Leon Voukoutis, Georgios Paraskevopoulos, Sokratis Sofianopoulos, Prokopis Prokopidis, Vassilis Papavasileiou, Athanasios Katsamanis, Stelios Piperidis, Vassilis Katsouros  

**Link**: [PDF](https://arxiv.org/pdf/2505.13772)  

**Abstract**: We introduce Llama-Krikri-8B, a cutting-edge Large Language Model tailored for the Greek language, built on Meta's Llama 3.1-8B. Llama-Krikri-8B has been extensively trained on high-quality Greek data to ensure superior adaptation to linguistic nuances. With 8 billion parameters, it offers advanced capabilities while maintaining efficient computational performance. Llama-Krikri-8B supports both Modern Greek and English, and is also equipped to handle polytonic text and Ancient Greek. The chat version of Llama-Krikri-8B features a multi-stage post-training pipeline, utilizing both human and synthetic instruction and preference data, by applying techniques such as MAGPIE. In addition, for evaluation, we propose three novel public benchmarks for Greek. Our evaluation on existing as well as the proposed benchmarks shows notable improvements over comparable Greek and multilingual LLMs in both natural language understanding and generation as well as code generation. 

---
# Simulation Agent: A Framework for Integrating Simulation and Large Language Models for Enhanced Decision-Making 

**Authors**: Jacob Kleiman, Kevin Frank, Sindy Campagna  

**Link**: [PDF](https://arxiv.org/pdf/2505.13761)  

**Abstract**: Simulations, although powerful in accurately replicating real-world systems, often remain inaccessible to non-technical users due to their complexity. Conversely, large language models (LLMs) provide intuitive, language-based interactions but can lack the structured, causal understanding required to reliably model complex real-world dynamics. We introduce our simulation agent framework, a novel approach that integrates the strengths of both simulation models and LLMs. This framework helps empower users by leveraging the conversational capabilities of LLMs to interact seamlessly with sophisticated simulation systems, while simultaneously utilizing the simulations to ground the LLMs in accurate and structured representations of real-world phenomena. This integrated approach helps provide a robust and generalizable foundation for empirical validation and offers broad applicability across diverse domains. 

---
# SQLForge: Synthesizing Reliable and Diverse Data to Enhance Text-to-SQL Reasoning in LLMs 

**Authors**: Yu Guo, Dong Jin, Shenghao Ye, Shuangwu Chen, Jian Yang, Xiaobin Tan  

**Link**: [PDF](https://arxiv.org/pdf/2505.13725)  

**Abstract**: Large Language models (LLMs) have demonstrated significant potential in text-to-SQL reasoning tasks, yet a substantial performance gap persists between existing open-source models and their closed-source counterparts. In this paper, we introduce SQLForge, a novel approach for synthesizing reliable and diverse data to enhance text-to-SQL reasoning in LLMs. We improve data reliability through SQL syntax constraints and SQL-to-question reverse translation, ensuring data logic at both structural and semantic levels. We also propose an SQL template enrichment and iterative data domain exploration mechanism to boost data diversity. Building on the augmented data, we fine-tune a variety of open-source models with different architectures and parameter sizes, resulting in a family of models termed SQLForge-LM. SQLForge-LM achieves the state-of-the-art performance on the widely recognized Spider and BIRD benchmarks among the open-source models. Specifically, SQLForge-LM achieves EX accuracy of 85.7% on Spider Dev and 59.8% on BIRD Dev, significantly narrowing the performance gap with closed-source methods. 

---
# Are Large Language Models Good at Detecting Propaganda? 

**Authors**: Julia Jose, Rachel Greenstadt  

**Link**: [PDF](https://arxiv.org/pdf/2505.13706)  

**Abstract**: Propagandists use rhetorical devices that rely on logical fallacies and emotional appeals to advance their agendas. Recognizing these techniques is key to making informed decisions. Recent advances in Natural Language Processing (NLP) have enabled the development of systems capable of detecting manipulative content. In this study, we look at several Large Language Models and their performance in detecting propaganda techniques in news articles. We compare the performance of these LLMs with transformer-based models. We find that, while GPT-4 demonstrates superior F1 scores (F1=0.16) compared to GPT-3.5 and Claude 3 Opus, it does not outperform a RoBERTa-CRF baseline (F1=0.67). Additionally, we find that all three LLMs outperform a MultiGranularity Network (MGN) baseline in detecting instances of one out of six propaganda techniques (name-calling), with GPT-3.5 and GPT-4 also outperforming the MGN baseline in detecting instances of appeal to fear and flag-waving. 

---
# Clarifying orthography: Orthographic transparency as compressibility 

**Authors**: Charles J. Torres, Richard Futrell  

**Link**: [PDF](https://arxiv.org/pdf/2505.13657)  

**Abstract**: Orthographic transparency -- how directly spelling is related to sound -- lacks a unified, script-agnostic metric. Using ideas from algorithmic information theory, we quantify orthographic transparency in terms of the mutual compressibility between orthographic and phonological strings. Our measure provides a principled way to combine two factors that decrease orthographic transparency, capturing both irregular spellings and rule complexity in one quantity. We estimate our transparency measure using prequential code-lengths derived from neural sequence models. Evaluating 22 languages across a broad range of script types (alphabetic, abjad, abugida, syllabic, logographic) confirms common intuitions about relative transparency of scripts. Mutual compressibility offers a simple, principled, and general yardstick for orthographic transparency. 

---
# Cross-Lingual Representation Alignment Through Contrastive Image-Caption Tuning 

**Authors**: Nathaniel Krasner, Nicholas Lanuzo, Antonios Anastasopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2505.13628)  

**Abstract**: Multilingual alignment of sentence representations has mostly required bitexts to bridge the gap between languages. We investigate whether visual information can bridge this gap instead. Image caption datasets are very easy to create without requiring multilingual expertise, so this offers a more efficient alternative for low-resource languages. We find that multilingual image-caption alignment can implicitly align the text representations between languages, languages unseen by the encoder in pretraining can be incorporated into this alignment post-hoc, and these aligned representations are usable for cross-lingual Natural Language Understanding (NLU) and bitext retrieval. 

---
# CS-Sum: A Benchmark for Code-Switching Dialogue Summarization and the Limits of Large Language Models 

**Authors**: Sathya Krishnan Suresh, Tanmay Surana, Lim Zhi Hao, Eng Siong Chng  

**Link**: [PDF](https://arxiv.org/pdf/2505.13559)  

**Abstract**: Code-switching (CS) poses a significant challenge for Large Language Models (LLMs), yet its comprehensibility remains underexplored in LLMs. We introduce CS-Sum, to evaluate the comprehensibility of CS by the LLMs through CS dialogue to English summarization. CS-Sum is the first benchmark for CS dialogue summarization across Mandarin-English (EN-ZH), Tamil-English (EN-TA), and Malay-English (EN-MS), with 900-1300 human-annotated dialogues per language pair. Evaluating ten LLMs, including open and closed-source models, we analyze performance across few-shot, translate-summarize, and fine-tuning (LoRA, QLoRA on synthetic data) approaches. Our findings show that though the scores on automated metrics are high, LLMs make subtle mistakes that alter the complete meaning of the dialogue. To this end, we introduce 3 most common type of errors that LLMs make when handling CS input. Error rates vary across CS pairs and LLMs, with some LLMs showing more frequent errors on certain language pairs, underscoring the need for specialized training on code-switched data. 

---
# Combining the Best of Both Worlds: A Method for Hybrid NMT and LLM Translation 

**Authors**: Zhanglin Wu, Daimeng Wei, Xiaoyu Chen, Hengchao Shang, Jiaxin Guo, Zongyao Li, Yuanchang Luo, Jinlong Yang, Zhiqiang Rao, Hao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13554)  

**Abstract**: Large language model (LLM) shows promising performances in a variety of downstream tasks, such as machine translation (MT). However, using LLMs for translation suffers from high computational costs and significant latency. Based on our evaluation, in most cases, translations using LLMs are comparable to that generated by neural machine translation (NMT) systems. Only in particular scenarios, LLM and NMT models show respective advantages. As a result, integrating NMT and LLM for translation and using LLM only when necessary seems to be a sound solution. A scheduling policy that optimizes translation result while ensuring fast speed and as little LLM usage as possible is thereby required. We compare several scheduling policies and propose a novel and straightforward decider that leverages source sentence features. We conduct extensive experiments on multilingual test sets and the result shows that we can achieve optimal translation performance with minimal LLM usage, demonstrating effectiveness of our decider. 

---
# Logic Jailbreak: Efficiently Unlocking LLM Safety Restrictions Through Formal Logical Expression 

**Authors**: Jingyu Peng, Maolin Wang, Nan Wang, Xiangyu Zhao, Jiatong Li, Kai Zhang, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13527)  

**Abstract**: Despite substantial advancements in aligning large language models (LLMs) with human values, current safety mechanisms remain susceptible to jailbreak attacks. We hypothesize that this vulnerability stems from distributional discrepancies between alignment-oriented prompts and malicious prompts. To investigate this, we introduce LogiBreak, a novel and universal black-box jailbreak method that leverages logical expression translation to circumvent LLM safety systems. By converting harmful natural language prompts into formal logical expressions, LogiBreak exploits the distributional gap between alignment data and logic-based inputs, preserving the underlying semantic intent and readability while evading safety constraints. We evaluate LogiBreak on a multilingual jailbreak dataset spanning three languages, demonstrating its effectiveness across various evaluation settings and linguistic contexts. 

---
# Induction Head Toxicity Mechanistically Explains Repetition Curse in Large Language Models 

**Authors**: Shuxun Wang, Qingyu Yin, Chak Tou Leong, Qiang Zhang, Linyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13514)  

**Abstract**: Repetition curse is a phenomenon where Large Language Models (LLMs) generate repetitive sequences of tokens or cyclic sequences. While the repetition curse has been widely observed, its underlying mechanisms remain poorly understood. In this work, we investigate the role of induction heads--a specific type of attention head known for their ability to perform in-context learning--in driving this repetitive behavior. Specifically, we focus on the "toxicity" of induction heads, which we define as their tendency to dominate the model's output logits during repetition, effectively excluding other attention heads from contributing to the generation process. Our findings have important implications for the design and training of LLMs. By identifying induction heads as a key driver of the repetition curse, we provide a mechanistic explanation for this phenomenon and suggest potential avenues for mitigation. We also propose a technique with attention head regularization that could be employed to reduce the dominance of induction heads during generation, thereby promoting more diverse and coherent outputs. 

---
# Time-R1: Towards Comprehensive Temporal Reasoning in LLMs 

**Authors**: Zijia Liu, Peixuan Han, Haofei Yu, Haoru Li, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2505.13508)  

**Abstract**: Large Language Models (LLMs) demonstrate impressive capabilities but lack robust temporal intelligence, struggling to integrate reasoning about the past with predictions and plausible generations of the future. Meanwhile, existing methods typically target isolated temporal skills, such as question answering about past events or basic forecasting, and exhibit poor generalization, particularly when dealing with events beyond their knowledge cutoff or requiring creative foresight. To address these limitations, we introduce \textit{Time-R1}, the first framework to endow a moderate-sized (3B-parameter) LLM with comprehensive temporal abilities: understanding, prediction, and creative generation. Our approach features a novel three-stage development path; the first two constitute a \textit{reinforcement learning (RL) curriculum} driven by a meticulously designed dynamic rule-based reward system. This framework progressively builds (1) foundational temporal understanding and logical event-time mappings from historical data, (2) future event prediction skills for events beyond its knowledge cutoff, and finally (3) enables remarkable generalization to creative future scenario generation without any fine-tuning. Strikingly, experiments demonstrate that Time-R1 outperforms models over 200 times larger, including the state-of-the-art 671B DeepSeek-R1, on highly challenging future event prediction and creative scenario generation benchmarks. This work provides strong evidence that thoughtfully engineered, progressive RL fine-tuning allows smaller, efficient models to achieve superior temporal performance, offering a practical and scalable path towards truly time-aware AI. To foster further research, we also release \textit{Time-Bench}, a large-scale multi-task temporal reasoning dataset derived from 10 years of news data, and our series of \textit{Time-R1} checkpoints. 

---
# EcoSafeRAG: Efficient Security through Context Analysis in Retrieval-Augmented Generation 

**Authors**: Ruobing Yao, Yifei Zhang, Shuang Song, Neng Gao, Chenyang Tu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13506)  

**Abstract**: Retrieval-Augmented Generation (RAG) compensates for the static knowledge limitations of Large Language Models (LLMs) by integrating external knowledge, producing responses with enhanced factual correctness and query-specific contextualization. However, it also introduces new attack surfaces such as corpus poisoning at the same time. Most of the existing defense methods rely on the internal knowledge of the model, which conflicts with the design concept of RAG. To bridge the gap, EcoSafeRAG uses sentence-level processing and bait-guided context diversity detection to identify malicious content by analyzing the context diversity of candidate documents without relying on LLM internal knowledge. Experiments show EcoSafeRAG delivers state-of-the-art security with plug-and-play deployment, simultaneously improving clean-scenario RAG performance while maintaining practical operational costs (relatively 1.2$\times$ latency, 48\%-80\% token reduction versus Vanilla RAG). 

---
# Noise Injection Systemically Degrades Large Language Model Safety Guardrails 

**Authors**: Prithviraj Singh Shahani, Matthias Scheutz  

**Link**: [PDF](https://arxiv.org/pdf/2505.13500)  

**Abstract**: Safety guardrails in large language models (LLMs) are a critical component in preventing harmful outputs. Yet, their resilience under perturbation remains poorly understood. In this paper, we investigate the robustness of safety fine-tuning in LLMs by systematically injecting Gaussian noise into model activations. We show across multiple open-weight models that (1) Gaussian noise raises harmful-output rates (p < 0.001) by up to 27%, (2) that deeper safety fine-tuning affords no extra protection, and (3) that chain-of-thought reasoning remains largely intact. The findings reveal critical vulnerabilities in current safety alignment techniques and highlight the potential of reasoning-based and reinforcement learning approaches as promising direction for developing more robust AI safety systems. These results have important implications for real-world deployment of LLMs in safety-critical applications as these results imply that widely-deployed safety tuning methods can fail even without adversarial prompts. 

---
# IRLBench: A Multi-modal, Culturally Grounded, Parallel Irish-English Benchmark for Open-Ended LLM Reasoning Evaluation 

**Authors**: Khanh-Tung Tran, Barry O'Sullivan, Hoang D. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2505.13498)  

**Abstract**: Recent advances in Large Language Models (LLMs) have demonstrated promising knowledge and reasoning abilities, yet their performance in multilingual and low-resource settings remains underexplored. Existing benchmarks often exhibit cultural bias, restrict evaluation to text-only, rely on multiple-choice formats, and, more importantly, are limited for extremely low-resource languages. To address these gaps, we introduce IRLBench, presented in parallel English and Irish, which is considered definitely endangered by UNESCO. Our benchmark consists of 12 representative subjects developed from the 2024 Irish Leaving Certificate exams, enabling fine-grained analysis of model capabilities across domains. By framing the task as long-form generation and leveraging the official marking scheme, it does not only support a comprehensive evaluation of correctness but also language fidelity. Our extensive experiments of leading closed-source and open-source LLMs reveal a persistent performance gap between English and Irish, in which models produce valid Irish responses less than 80\% of the time, and answer correctly 55.8\% of the time compared to 76.2\% in English for the best-performing model. We release IRLBench (this https URL) and an accompanying evaluation codebase (this https URL) to enable future research on robust, culturally aware multilingual AI development. 

---
# LLM4CD: Leveraging Large Language Models for Open-World Knowledge Augmented Cognitive Diagnosis 

**Authors**: Weiming Zhang, Lingyue Fu, Qingyao Li, Kounianhua Du, Jianghao Lin, Jingwei Yu, Wei Xia, Weinan Zhang, Ruiming Tang, Yong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13492)  

**Abstract**: Cognitive diagnosis (CD) plays a crucial role in intelligent education, evaluating students' comprehension of knowledge concepts based on their test histories. However, current CD methods often model students, exercises, and knowledge concepts solely on their ID relationships, neglecting the abundant semantic relationships present within educational data space. Furthermore, contemporary intelligent tutoring systems (ITS) frequently involve the addition of new students and exercises, a situation that ID-based methods find challenging to manage effectively. The advent of large language models (LLMs) offers the potential for overcoming this challenge with open-world knowledge. In this paper, we propose LLM4CD, which Leverages Large Language Models for Open-World Knowledge Augmented Cognitive Diagnosis. Our method utilizes the open-world knowledge of LLMs to construct cognitively expressive textual representations, which are then encoded to introduce rich semantic information into the CD task. Additionally, we propose an innovative bi-level encoder framework that models students' test histories through two levels of encoders: a macro-level cognitive text encoder and a micro-level knowledge state encoder. This approach substitutes traditional ID embeddings with semantic representations, enabling the model to accommodate new students and exercises with open-world knowledge and address the cold-start problem. Extensive experimental results demonstrate that our proposed method consistently outperforms previous CD models on multiple real-world datasets, validating the effectiveness of leveraging LLMs to introduce rich semantic information into the CD task. 

---
# ProdRev: A DNN framework for empowering customers using generative pre-trained transformers 

**Authors**: Aakash Gupta, Nataraj Das  

**Link**: [PDF](https://arxiv.org/pdf/2505.13491)  

**Abstract**: Following the pandemic, customers, preference for using e-commerce has accelerated. Since much information is available in multiple reviews (sometimes running in thousands) for a single product, it can create decision paralysis for the buyer. This scenario disempowers the consumer, who cannot be expected to go over so many reviews since its time consuming and can confuse them. Various commercial tools are available, that use a scoring mechanism to arrive at an adjusted score. It can alert the user to potential review manipulations. This paper proposes a framework that fine-tunes a generative pre-trained transformer to understand these reviews better. Furthermore, using "common-sense" to make better decisions. These models have more than 13 billion parameters. To fine-tune the model for our requirement, we use the curie engine from generative pre-trained transformer (GPT3). By using generative models, we are introducing abstractive summarization. Instead of using a simple extractive method of summarizing the reviews. This brings out the true relationship between the reviews and not simply copy-paste. This introduces an element of "common sense" for the user and helps them to quickly make the right decisions. The user is provided the pros and cons of the processed reviews. Thus the user/customer can take their own decisions. 

---
# Source framing triggers systematic evaluation bias in Large Language Models 

**Authors**: Federico Germani, Giovanni Spitale  

**Link**: [PDF](https://arxiv.org/pdf/2505.13488)  

**Abstract**: Large Language Models (LLMs) are increasingly used not only to generate text but also to evaluate it, raising urgent questions about whether their judgments are consistent, unbiased, and robust to framing effects. In this study, we systematically examine inter- and intra-model agreement across four state-of-the-art LLMs (OpenAI o3-mini, Deepseek Reasoner, xAI Grok 2, and Mistral) tasked with evaluating 4,800 narrative statements on 24 different topics of social, political, and public health relevance, for a total of 192,000 assessments. We manipulate the disclosed source of each statement to assess how attribution to either another LLM or a human author of specified nationality affects evaluation outcomes. We find that, in the blind condition, different LLMs display a remarkably high degree of inter- and intra-model agreement across topics. However, this alignment breaks down when source framing is introduced. Here we show that attributing statements to Chinese individuals systematically lowers agreement scores across all models, and in particular for Deepseek Reasoner. Our findings reveal that framing effects can deeply affect text evaluation, with significant implications for the integrity, neutrality, and fairness of LLM-mediated information systems. 

---
# Detecting Prefix Bias in LLM-based Reward Models 

**Authors**: Ashwin Kumar, Yuzi He, Aram H. Markosyan, Bobbie Chern, Imanol Arrieta-Ibarra  

**Link**: [PDF](https://arxiv.org/pdf/2505.13487)  

**Abstract**: Reinforcement Learning with Human Feedback (RLHF) has emerged as a key paradigm for task-specific fine-tuning of language models using human preference data. While numerous publicly available preference datasets provide pairwise comparisons of responses, the potential for biases in the resulting reward models remains underexplored. In this work, we introduce novel methods to detect and evaluate prefix bias -- a systematic shift in model preferences triggered by minor variations in query prefixes -- in LLM-based reward models trained on such datasets. We leverage these metrics to reveal significant biases in preference models across racial and gender dimensions. Our comprehensive evaluation spans diverse open-source preference datasets and reward model architectures, demonstrating susceptibility to this kind of bias regardless of the underlying model architecture. Furthermore, we propose a data augmentation strategy to mitigate these biases, showing its effectiveness in reducing the impact of prefix bias. Our findings highlight the critical need for bias-aware dataset design and evaluation in developing fair and reliable reward models, contributing to the broader discourse on fairness in AI. 

---
# EmoMeta: A Multimodal Dataset for Fine-grained Emotion Classification in Chinese Metaphors 

**Authors**: Xingyuan Lu, Yuxi Liu, Dongyu Zhang, Zhiyao Wu, Jing Ren, Feng Xia  

**Link**: [PDF](https://arxiv.org/pdf/2505.13483)  

**Abstract**: Metaphors play a pivotal role in expressing emotions, making them crucial for emotional intelligence. The advent of multimodal data and widespread communication has led to a proliferation of multimodal metaphors, amplifying the complexity of emotion classification compared to single-mode scenarios. However, the scarcity of research on constructing multimodal metaphorical fine-grained emotion datasets hampers progress in this domain. Moreover, existing studies predominantly focus on English, overlooking potential variations in emotional nuances across languages. To address these gaps, we introduce a multimodal dataset in Chinese comprising 5,000 text-image pairs of metaphorical advertisements. Each entry is meticulously annotated for metaphor occurrence, domain relations and fine-grained emotion classification encompassing joy, love, trust, fear, sadness, disgust, anger, surprise, anticipation, and neutral. Our dataset is publicly accessible (this https URL), facilitating further advancements in this burgeoning field. 

---
# Evaluating Reasoning LLMs for Suicide Screening with the Columbia-Suicide Severity Rating Scale 

**Authors**: Avinash Patil, Siru Tao, Amardeep Gedhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13480)  

**Abstract**: Suicide prevention remains a critical public health challenge. While online platforms such as Reddit's r/SuicideWatch have historically provided spaces for individuals to express suicidal thoughts and seek community support, the advent of large language models (LLMs) introduces a new paradigm-where individuals may begin disclosing ideation to AI systems instead of humans. This study evaluates the capability of LLMs to perform automated suicide risk assessment using the Columbia-Suicide Severity Rating Scale (C-SSRS). We assess the zero-shot performance of six models-including Claude, GPT, Mistral, and LLaMA-in classifying posts across a 7-point severity scale (Levels 0-6). Results indicate that Claude and GPT closely align with human annotations, while Mistral achieves the lowest ordinal prediction error. Most models exhibit ordinal sensitivity, with misclassifications typically occurring between adjacent severity levels. We further analyze confusion patterns, misclassification sources, and ethical considerations, underscoring the importance of human oversight, transparency, and cautious deployment. Full code and supplementary materials are available at this https URL. 

---
# Two Experts Are All You Need for Steering Thinking: Reinforcing Cognitive Effort in MoE Reasoning Models Without Additional Training 

**Authors**: Mengru Wang, Xingyu Chen, Yue Wang, Zhiwei He, Jiahao Xu, Tian Liang, Qiuzhi Liu, Yunzhi Yao, Wenxuan Wang, Ruotian Ma, Haitao Mi, Ningyu Zhang, Zhaopeng Tu, Xiaolong Li, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.14681)  

**Abstract**: Mixture-of-Experts (MoE) architectures within Large Reasoning Models (LRMs) have achieved impressive reasoning capabilities by selectively activating experts to facilitate structured cognitive processes. Despite notable advances, existing reasoning models often suffer from cognitive inefficiencies like overthinking and underthinking. To address these limitations, we introduce a novel inference-time steering methodology called Reinforcing Cognitive Experts (RICE), designed to improve reasoning performance without additional training or complex heuristics. Leveraging normalized Pointwise Mutual Information (nPMI), we systematically identify specialized experts, termed ''cognitive experts'' that orchestrate meta-level reasoning operations characterized by tokens like ''<think>''. Empirical evaluations with leading MoE-based LRMs (DeepSeek-R1 and Qwen3-235B) on rigorous quantitative and scientific reasoning benchmarks demonstrate noticeable and consistent improvements in reasoning accuracy, cognitive efficiency, and cross-domain generalization. Crucially, our lightweight approach substantially outperforms prevalent reasoning-steering techniques, such as prompt design and decoding constraints, while preserving the model's general instruction-following skills. These results highlight reinforcing cognitive experts as a promising, practical, and interpretable direction to enhance cognitive efficiency within advanced reasoning models. 

---
# NExT-Search: Rebuilding User Feedback Ecosystem for Generative AI Search 

**Authors**: Sunhao Dai, Wenjie Wang, Liang Pang, Jun Xu, See-Kiong Ng, Ji-Rong Wen, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2505.14680)  

**Abstract**: Generative AI search is reshaping information retrieval by offering end-to-end answers to complex queries, reducing users' reliance on manually browsing and summarizing multiple web pages. However, while this paradigm enhances convenience, it disrupts the feedback-driven improvement loop that has historically powered the evolution of traditional Web search. Web search can continuously improve their ranking models by collecting large-scale, fine-grained user feedback (e.g., clicks, dwell time) at the document level. In contrast, generative AI search operates through a much longer search pipeline, spanning query decomposition, document retrieval, and answer generation, yet typically receives only coarse-grained feedback on the final answer. This introduces a feedback loop disconnect, where user feedback for the final output cannot be effectively mapped back to specific system components, making it difficult to improve each intermediate stage and sustain the feedback loop. In this paper, we envision NExT-Search, a next-generation paradigm designed to reintroduce fine-grained, process-level feedback into generative AI search. NExT-Search integrates two complementary modes: User Debug Mode, which allows engaged users to intervene at key stages; and Shadow User Mode, where a personalized user agent simulates user preferences and provides AI-assisted feedback for less interactive users. Furthermore, we envision how these feedback signals can be leveraged through online adaptation, which refines current search outputs in real-time, and offline update, which aggregates interaction logs to periodically fine-tune query decomposition, retrieval, and generation models. By restoring human control over key stages of the generative AI search pipeline, we believe NExT-Search offers a promising direction for building feedback-rich AI search systems that can evolve continuously alongside human feedback. 

---
# ContextAgent: Context-Aware Proactive LLM Agents with Open-World Sensory Perceptions 

**Authors**: Bufang Yang, Lilin Xu, Liekang Zeng, Kaiwei Liu, Siyang Jiang, Wenrui Lu, Hongkai Chen, Xiaofan Jiang, Guoliang Xing, Zhenyu Yan  

**Link**: [PDF](https://arxiv.org/pdf/2505.14668)  

**Abstract**: Recent advances in Large Language Models (LLMs) have propelled intelligent agents from reactive responses to proactive support. While promising, existing proactive agents either rely exclusively on observations from enclosed environments (e.g., desktop UIs) with direct LLM inference or employ rule-based proactive notifications, leading to suboptimal user intent understanding and limited functionality for proactive service. In this paper, we introduce ContextAgent, the first context-aware proactive agent that incorporates extensive sensory contexts to enhance the proactive capabilities of LLM agents. ContextAgent first extracts multi-dimensional contexts from massive sensory perceptions on wearables (e.g., video and audio) to understand user intentions. ContextAgent then leverages the sensory contexts and the persona contexts from historical data to predict the necessity for proactive services. When proactive assistance is needed, ContextAgent further automatically calls the necessary tools to assist users unobtrusively. To evaluate this new task, we curate ContextAgentBench, the first benchmark for evaluating context-aware proactive LLM agents, covering 1,000 samples across nine daily scenarios and twenty tools. Experiments on ContextAgentBench show that ContextAgent outperforms baselines by achieving up to 8.5% and 6.0% higher accuracy in proactive predictions and tool calling, respectively. We hope our research can inspire the development of more advanced, human-centric, proactive AI assistants. 

---
# SAFEPATH: Preventing Harmful Reasoning in Chain-of-Thought via Early Alignment 

**Authors**: Wonje Jeung, Sangyeon Yoon, Minsuk Kahng, Albert No  

**Link**: [PDF](https://arxiv.org/pdf/2505.14667)  

**Abstract**: Large Reasoning Models (LRMs) have become powerful tools for complex problem solving, but their structured reasoning pathways can lead to unsafe outputs when exposed to harmful prompts. Existing safety alignment methods reduce harmful outputs but can degrade reasoning depth, leading to significant trade-offs in complex, multi-step tasks, and remain vulnerable to sophisticated jailbreak attacks. To address this, we introduce SAFEPATH, a lightweight alignment method that fine-tunes LRMs to emit a short, 8-token Safety Primer at the start of their reasoning, in response to harmful prompts, while leaving the rest of the reasoning process unsupervised. Empirical results across multiple benchmarks indicate that SAFEPATH effectively reduces harmful outputs while maintaining reasoning performance. Specifically, SAFEPATH reduces harmful responses by up to 90.0% and blocks 83.3% of jailbreak attempts in the DeepSeek-R1-Distill-Llama-8B model, while requiring 295.9x less compute than Direct Refusal and 314.1x less than SafeChain. We further introduce a zero-shot variant that requires no fine-tuning. In addition, we provide a comprehensive analysis of how existing methods in LLMs generalize, or fail, when applied to reasoning-centric models, revealing critical gaps and new directions for safer AI. 

---
# Beyond Words: Multimodal LLM Knows When to Speak 

**Authors**: Zikai Liao, Yi Ouyang, Yi-Lun Lee, Chen-Ping Yu, Yi-Hsuan Tsai, Zhaozheng Yin  

**Link**: [PDF](https://arxiv.org/pdf/2505.14654)  

**Abstract**: While large language model (LLM)-based chatbots have demonstrated strong capabilities in generating coherent and contextually relevant responses, they often struggle with understanding when to speak, particularly in delivering brief, timely reactions during ongoing conversations. This limitation arises largely from their reliance on text input, lacking the rich contextual cues in real-world human dialogue. In this work, we focus on real-time prediction of response types, with an emphasis on short, reactive utterances that depend on subtle, multimodal signals across vision, audio, and text. To support this, we introduce a new multimodal dataset constructed from real-world conversational videos, containing temporally aligned visual, auditory, and textual streams. This dataset enables fine-grained modeling of response timing in dyadic interactions. Building on this dataset, we propose MM-When2Speak, a multimodal LLM-based model that adaptively integrates visual, auditory, and textual context to predict when a response should occur, and what type of response is appropriate. Experiments show that MM-When2Speak significantly outperforms state-of-the-art unimodal and LLM-based baselines, achieving up to a 4x improvement in response timing accuracy over leading commercial LLMs. These results underscore the importance of multimodal inputs for producing timely, natural, and engaging conversational AI. 

---
# Dual Precision Quantization for Efficient and Accurate Deep Neural Networks Inference 

**Authors**: Tomer Gafni, Asaf Karnieli, Yair Hanani  

**Link**: [PDF](https://arxiv.org/pdf/2505.14638)  

**Abstract**: Deep neural networks have achieved state-of-the-art results in a wide range of applications, from natural language processing and computer vision to speech recognition. However, as tasks become increasingly complex, model sizes continue to grow, posing challenges in latency and memory efficiency. To meet these constraints, post-training quantization has emerged as a promising solution. In this paper, we propose a novel hardware-efficient quantization and inference scheme that exploits hardware advantages with minimal accuracy degradation. Specifically, we introduce a W4A8 scheme, where weights are quantized and stored using 4-bit integer precision, and inference computations are performed using 8-bit floating-point arithmetic, demonstrating significant speedups and improved memory utilization compared to 16-bit operations, applicable on various modern accelerators. To mitigate accuracy loss, we develop a novel quantization algorithm, dubbed Dual Precision Quantization (DPQ), that leverages the unique structure of our scheme without introducing additional inference overhead. Experimental results demonstrate improved performance (i.e., increased throughput) while maintaining tolerable accuracy degradation relative to the full-precision model. 

---
# KERL: Knowledge-Enhanced Personalized Recipe Recommendation using Large Language Models 

**Authors**: Fnu Mohbat, Mohammed J Zaki  

**Link**: [PDF](https://arxiv.org/pdf/2505.14629)  

**Abstract**: Recent advances in large language models (LLMs) and the abundance of food data have resulted in studies to improve food understanding using LLMs. Despite several recommendation systems utilizing LLMs and Knowledge Graphs (KGs), there has been limited research on integrating food related KGs with LLMs. We introduce KERL, a unified system that leverages food KGs and LLMs to provide personalized food recommendations and generates recipes with associated micro-nutritional information. Given a natural language question, KERL extracts entities, retrieves subgraphs from the KG, which are then fed into the LLM as context to select the recipes that satisfy the constraints. Next, our system generates the cooking steps and nutritional information for each recipe. To evaluate our approach, we also develop a benchmark dataset by curating recipe related questions, combined with constraints and personal preferences. Through extensive experiments, we show that our proposed KG-augmented LLM significantly outperforms existing approaches, offering a complete and coherent solution for food recommendation, recipe generation, and nutritional analysis. Our code and benchmark datasets are publicly available at this https URL. 

---
# Debating for Better Reasoning: An Unsupervised Multimodal Approach 

**Authors**: Ashutosh Adhikari, Mirella Lapata  

**Link**: [PDF](https://arxiv.org/pdf/2505.14627)  

**Abstract**: As Large Language Models (LLMs) gain expertise across diverse domains and modalities, scalable oversight becomes increasingly challenging, particularly when their capabilities may surpass human evaluators. Debate has emerged as a promising mechanism for enabling such oversight. In this work, we extend the debate paradigm to a multimodal setting, exploring its potential for weaker models to supervise and enhance the performance of stronger models. We focus on visual question answering (VQA), where two "sighted" expert vision-language models debate an answer, while a "blind" (text-only) judge adjudicates based solely on the quality of the arguments. In our framework, the experts defend only answers aligned with their beliefs, thereby obviating the need for explicit role-playing and concentrating the debate on instances of expert disagreement. Experiments on several multimodal tasks demonstrate that the debate framework consistently outperforms individual expert models. Moreover, judgments from weaker LLMs can help instill reasoning capabilities in vision-language models through finetuning. 

---
# TinyV: Reducing False Negatives in Verification Improves RL for LLM Reasoning 

**Authors**: Zhangchen Xu, Yuetai Li, Fengqing Jiang, Bhaskar Ramasubramanian, Luyao Niu, Bill Yuchen Lin, Radha Poovendran  

**Link**: [PDF](https://arxiv.org/pdf/2505.14625)  

**Abstract**: Reinforcement Learning (RL) has become a powerful tool for enhancing the reasoning abilities of large language models (LLMs) by optimizing their policies with reward signals. Yet, RL's success relies on the reliability of rewards, which are provided by verifiers. In this paper, we expose and analyze a widespread problem--false negatives--where verifiers wrongly reject correct model outputs. Our in-depth study of the Big-Math-RL-Verified dataset reveals that over 38% of model-generated responses suffer from false negatives, where the verifier fails to recognize correct answers. We show, both empirically and theoretically, that these false negatives severely impair RL training by depriving the model of informative gradient signals and slowing convergence. To mitigate this, we propose tinyV, a lightweight LLM-based verifier that augments existing rule-based methods, which dynamically identifies potential false negatives and recovers valid responses to produce more accurate reward estimates. Across multiple math-reasoning benchmarks, integrating TinyV boosts pass rates by up to 10% and accelerates convergence relative to the baseline. Our findings highlight the critical importance of addressing verifier false negatives and offer a practical approach to improve RL-based fine-tuning of LLMs. Our code is available at this https URL. 

---
# Enhancing Learned Knowledge in LoRA Adapters Through Efficient Contrastive Decoding on Ascend NPUs 

**Authors**: Morgan Lindsay Heisler, Linzi Xing, Ge Shi, Hanieh Sadri, Gursimran Singh, Weiwei Zhang, Tao Ye, Ying Xiong, Yong Zhang, Zhenan Fan  

**Link**: [PDF](https://arxiv.org/pdf/2505.14620)  

**Abstract**: Huawei Cloud users leverage LoRA (Low-Rank Adaptation) as an efficient and scalable method to fine-tune and customize large language models (LLMs) for application-specific needs. However, tasks that require complex reasoning or deep contextual understanding are often hindered by biases or interference from the base model when using typical decoding methods like greedy or beam search. These biases can lead to generic or task-agnostic responses from the base model instead of leveraging the LoRA-specific adaptations. In this paper, we introduce Contrastive LoRA Decoding (CoLD), a novel decoding framework designed to maximize the use of task-specific knowledge in LoRA-adapted models, resulting in better downstream performance. CoLD uses contrastive decoding by scoring candidate tokens based on the divergence between the probability distributions of a LoRA-adapted expert model and the corresponding base model. This approach prioritizes tokens that better align with the LoRA's learned representations, enhancing performance for specialized tasks. While effective, a naive implementation of CoLD is computationally expensive because each decoding step requires evaluating multiple token candidates across both models. To address this, we developed an optimized kernel for Huawei's Ascend NPU. CoLD achieves up to a 5.54% increase in task accuracy while reducing end-to-end latency by 28% compared to greedy decoding. This work provides practical and efficient decoding strategies for fine-tuned LLMs in resource-constrained environments and has broad implications for applied data science in both cloud and on-premises settings. 

---
# SATBench: Benchmarking LLMs' Logical Reasoning via Automated Puzzle Generation from SAT Formulas 

**Authors**: Anjiang Wei, Yuheng Wu, Yingjia Wan, Tarun Suresh, Huanmi Tan, Zhanke Zhou, Sanmi Koyejo, Ke Wang, Alex Aiken  

**Link**: [PDF](https://arxiv.org/pdf/2505.14615)  

**Abstract**: We introduce SATBench, a benchmark for evaluating the logical reasoning capabilities of large language models (LLMs) through logical puzzles derived from Boolean satisfiability (SAT) problems. Unlike prior work that focuses on inference rule-based reasoning, which often involves deducing conclusions from a set of premises, our approach leverages the search-based nature of SAT problems, where the objective is to find a solution that fulfills a specified set of logical constraints. Each instance in SATBench is generated from a SAT formula, then translated into a story context and conditions using LLMs. The generation process is fully automated and allows for adjustable difficulty by varying the number of clauses. All 2100 puzzles are validated through both LLM-assisted and solver-based consistency checks, with human validation on a subset. Experimental results show that even the strongest model, o4-mini, achieves only 65.0% accuracy on hard UNSAT problems, close to the random baseline of 50%. SATBench exposes fundamental limitations in the search-based logical reasoning abilities of current LLMs and provides a scalable testbed for future research in logical reasoning. 

---
# Agent Context Protocols Enhance Collective Inference 

**Authors**: Devansh Bhardwaj, Arjun Beniwal, Shreyas Chaudhari, Ashwin Kalyan, Tanmay Rajpurohit, Karthik R. Narasimhan, Ameet Deshpande, Vishvak Murahari  

**Link**: [PDF](https://arxiv.org/pdf/2505.14569)  

**Abstract**: AI agents have become increasingly adept at complex tasks such as coding, reasoning, and multimodal understanding. However, building generalist systems requires moving beyond individual agents to collective inference -- a paradigm where multi-agent systems with diverse, task-specialized agents complement one another through structured communication and collaboration. Today, coordination is usually handled with imprecise, ad-hoc natural language, which limits complex interaction and hinders interoperability with domain-specific agents. We introduce Agent context protocols (ACPs): a domain- and agent-agnostic family of structured protocols for agent-agent communication, coordination, and error handling. ACPs combine (i) persistent execution blueprints -- explicit dependency graphs that store intermediate agent outputs -- with (ii) standardized message schemas, enabling robust and fault-tolerant multi-agent collective inference. ACP-powered generalist systems reach state-of-the-art performance: 28.3 % accuracy on AssistantBench for long-horizon web assistance and best-in-class multimodal technical reports, outperforming commercial AI systems in human evaluation. ACPs are highly modular and extensible, allowing practitioners to build top-tier generalist agents quickly. 

---
# Teaching Audio-Aware Large Language Models What Does Not Hear: Mitigating Hallucinations through Synthesized Negative Samples 

**Authors**: Chun-Yi Kuan, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.14518)  

**Abstract**: Recent advancements in audio-aware large language models (ALLMs) enable them to process and understand audio inputs. However, these models often hallucinate non-existent sound events, reducing their reliability in real-world applications. To address this, we propose LISTEN (Learning to Identify Sounds Through Extended Negative Samples), a contrastive-like training method that enhances ALLMs' ability to distinguish between present and absent sounds using synthesized data from the backbone LLM. Unlike prior approaches, our method requires no modification to LLM parameters and efficiently integrates audio representations via a lightweight adapter. Experiments show that LISTEN effectively mitigates hallucinations while maintaining impressive performance on existing audio question and reasoning benchmarks. At the same time, it is more efficient in both data and computation. 

---
# Reasoning Models Better Express Their Confidence 

**Authors**: Dongkeun Yoon, Seungone Kim, Sohee Yang, Sunkyoung Kim, Soyeon Kim, Yongil Kim, Eunbi Choi, Yireun Kim, Minjoon Seo  

**Link**: [PDF](https://arxiv.org/pdf/2505.14489)  

**Abstract**: Despite their strengths, large language models (LLMs) often fail to communicate their confidence accurately, making it difficult to assess when they might be wrong and limiting their reliability. In this work, we demonstrate that reasoning models-LLMs that engage in extended chain-of-thought (CoT) reasoning-exhibit superior performance not only in problem-solving but also in accurately expressing their confidence. Specifically, we benchmark six reasoning models across six datasets and find that they achieve strictly better confidence calibration than their non-reasoning counterparts in 33 out of the 36 settings. Our detailed analysis reveals that these gains in calibration stem from the slow thinking behaviors of reasoning models-such as exploring alternative approaches and backtracking-which enable them to adjust their confidence dynamically throughout their CoT, making it progressively more accurate. In particular, we find that reasoning models become increasingly better calibrated as their CoT unfolds, a trend not observed in non-reasoning models. Moreover, removing slow thinking behaviors from the CoT leads to a significant drop in calibration. Lastly, we show that these gains are not exclusive to reasoning models-non-reasoning models also benefit when guided to perform slow thinking via in-context learning. 

---
# Towards Reliable Proof Generation with LLMs: A Neuro-Symbolic Approach 

**Authors**: Oren Sultan, Eitan Stern, Dafna Shahaf  

**Link**: [PDF](https://arxiv.org/pdf/2505.14479)  

**Abstract**: Large language models (LLMs) struggle with formal domains that require rigorous logical deduction and symbolic reasoning, such as mathematical proof generation. We propose a neuro-symbolic approach that combines LLMs' generative strengths with structured components to overcome this challenge. As a proof-of-concept, we focus on geometry problems. Our approach is two-fold: (1) we retrieve analogous problems and use their proofs to guide the LLM, and (2) a formal verifier evaluates the generated proofs and provides feedback, helping the model fix incorrect proofs. We demonstrate that our method significantly improves proof accuracy for OpenAI's o1 model (58%-70% improvement); both analogous problems and the verifier's feedback contribute to these gains. More broadly, shifting to LLMs that generate provably correct conclusions could dramatically improve their reliability, accuracy and consistency, unlocking complex tasks and critical real-world applications that require trustworthiness. 

---
# PAST: Phonetic-Acoustic Speech Tokenizer 

**Authors**: Nadav Har-Tuv, Or Tal, Yossi Adi  

**Link**: [PDF](https://arxiv.org/pdf/2505.14470)  

**Abstract**: We present PAST, a novel end-to-end framework that jointly models phonetic information alongside signal reconstruction, eliminating the need for external pretrained models. Unlike previous approaches that rely on pretrained self-supervised models, PAST employs supervised phonetic data, directly integrating domain knowledge into the tokenization process via auxiliary tasks. Additionally, we introduce a streamable, causal variant of PAST, enabling real-time speech applications. Results demonstrate that PAST surpasses existing evaluated baseline tokenizers across common evaluation metrics, including phonetic representation and speech reconstruction. Notably, PAST also achieves superior performance when serving as a speech representation for speech language models, further highlighting its effectiveness as a foundation for spoken language generation. To foster further research, we release the full implementation. For code, model checkpoints, and samples see: this https URL 

---
# RAVENEA: A Benchmark for Multimodal Retrieval-Augmented Visual Culture Understanding 

**Authors**: Jiaang Li, Yifei Yuan, Wenyan Li, Mohammad Aliannejadi, Daniel Hershcovich, Anders Søgaard, Ivan Vulić, Wenxuan Zhang, Paul Pu Liang, Yang Deng, Serge Belongie  

**Link**: [PDF](https://arxiv.org/pdf/2505.14462)  

**Abstract**: As vision-language models (VLMs) become increasingly integrated into daily life, the need for accurate visual culture understanding is becoming critical. Yet, these models frequently fall short in interpreting cultural nuances effectively. Prior work has demonstrated the effectiveness of retrieval-augmented generation (RAG) in enhancing cultural understanding in text-only settings, while its application in multimodal scenarios remains underexplored. To bridge this gap, we introduce RAVENEA (Retrieval-Augmented Visual culturE uNdErstAnding), a new benchmark designed to advance visual culture understanding through retrieval, focusing on two tasks: culture-focused visual question answering (cVQA) and culture-informed image captioning (cIC). RAVENEA extends existing datasets by integrating over 10,000 Wikipedia documents curated and ranked by human annotators. With RAVENEA, we train and evaluate seven multimodal retrievers for each image query, and measure the downstream impact of retrieval-augmented inputs across fourteen state-of-the-art VLMs. Our results show that lightweight VLMs, when augmented with culture-aware retrieval, outperform their non-augmented counterparts (by at least 3.2% absolute on cVQA and 6.2% absolute on cIC). This highlights the value of retrieval-augmented methods and culturally inclusive benchmarks for multimodal understanding. 

---
# Mitigating Subgroup Disparities in Multi-Label Speech Emotion Recognition: A Pseudo-Labeling and Unsupervised Learning Approach 

**Authors**: Yi-Cheng Lin, Huang-Cheng Chou, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.14449)  

**Abstract**: While subgroup disparities and performance bias are increasingly studied in computational research, fairness in categorical Speech Emotion Recognition (SER) remains underexplored. Existing methods often rely on explicit demographic labels, which are difficult to obtain due to privacy concerns. To address this limitation, we introduce an Implicit Demography Inference (IDI) module that leverages pseudo-labeling from a pre-trained model and unsupervised learning using k-means clustering to mitigate bias in SER. Our experiments show that pseudo-labeling IDI reduces subgroup disparities, improving fairness metrics by over 33% with less than a 3% decrease in SER accuracy. Also, the unsupervised IDI yields more than a 26% improvement in fairness metrics with a drop of less than 4% in SER performance. Further analyses reveal that the unsupervised IDI consistently mitigates race and age disparities, demonstrating its potential in scenarios where explicit demographic information is unavailable. 

---
# S2SBench: A Benchmark for Quantifying Intelligence Degradation in Speech-to-Speech Large Language Models 

**Authors**: Yuanbo Fang, Haoze Sun, Jun Liu, Tao Zhang, Zenan Zhou, Weipeng Chen, Xiaofen Xing, Xiangmin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.14438)  

**Abstract**: End-to-end speech large language models ((LLMs)) extend the capabilities of text-based models to directly process and generate audio tokens. However, this often leads to a decline in reasoning and generation performance compared to text input, a phenomenon referred to as intelligence degradation. To systematically evaluate this gap, we propose S2SBench, a benchmark designed to quantify performance degradation in Speech LLMs. It includes diagnostic datasets targeting sentence continuation and commonsense reasoning under audio input. We further introduce a pairwise evaluation protocol based on perplexity differences between plausible and implausible samples to measure degradation relative to text input. We apply S2SBench to analyze the training process of Baichuan-Audio, which further demonstrates the benchmark's effectiveness. All datasets and evaluation code are available at this https URL. 

---
# Rank-K: Test-Time Reasoning for Listwise Reranking 

**Authors**: Eugene Yang, Andrew Yates, Kathryn Ricci, Orion Weller, Vivek Chari, Benjamin Van Durme, Dawn Lawrie  

**Link**: [PDF](https://arxiv.org/pdf/2505.14432)  

**Abstract**: Retrieve-and-rerank is a popular retrieval pipeline because of its ability to make slow but effective rerankers efficient enough at query time by reducing the number of comparisons. Recent works in neural rerankers take advantage of large language models for their capability in reasoning between queries and passages and have achieved state-of-the-art retrieval effectiveness. However, such rerankers are resource-intensive, even after heavy optimization. In this work, we introduce Rank-K, a listwise passage reranking model that leverages the reasoning capability of the reasoning language model at query time that provides test time scalability to serve hard queries. We show that Rank-K improves retrieval effectiveness by 23\% over the RankZephyr, the state-of-the-art listwise reranker, when reranking a BM25 initial ranked list and 19\% when reranking strong retrieval results by SPLADE-v3. Since Rank-K is inherently a multilingual model, we found that it ranks passages based on queries in different languages as effectively as it does in monolingual retrieval. 

---
# PRL: Prompts from Reinforcement Learning 

**Authors**: Paweł Batorski, Adrian Kosmala, Paul Swoboda  

**Link**: [PDF](https://arxiv.org/pdf/2505.14412)  

**Abstract**: Effective prompt engineering remains a central challenge in fully harnessing the capabilities of LLMs. While well-designed prompts can dramatically enhance performance, crafting them typically demands expert intuition and a nuanced understanding of the task. Moreover, the most impactful prompts often hinge on subtle semantic cues, ones that may elude human perception but are crucial for guiding LLM behavior. In this paper, we introduce PRL (Prompts from Reinforcement Learning), a novel RL-based approach for automatic prompt generation. Unlike previous methods, PRL can produce novel few-shot examples that were not seen during training. Our approach achieves state-of-the-art performance across a range of benchmarks, including text classification, simplification, and summarization. On the classification task, it surpasses prior methods by 2.58% over APE and 1.00% over EvoPrompt. Additionally, it improves the average ROUGE scores on the summarization task by 4.32 over APE and by 2.12 over EvoPrompt and the SARI score on simplification by 6.93 over APE and by 6.01 over EvoPrompt. Our code is available at this https URL . 

---
# Pairwise Evaluation of Accent Similarity in Speech Synthesis 

**Authors**: Jinzuomu Zhong, Suyuan Liu, Dan Wells, Korin Richmond  

**Link**: [PDF](https://arxiv.org/pdf/2505.14410)  

**Abstract**: Despite growing interest in generating high-fidelity accents, evaluating accent similarity in speech synthesis has been underexplored. We aim to enhance both subjective and objective evaluation methods for accent similarity. Subjectively, we refine the XAB listening test by adding components that achieve higher statistical significance with fewer listeners and lower costs. Our method involves providing listeners with transcriptions, having them highlight perceived accent differences, and implementing meticulous screening for reliability. Objectively, we utilise pronunciation-related metrics, based on distances between vowel formants and phonetic posteriorgrams, to evaluate accent generation. Comparative experiments reveal that these metrics, alongside accent similarity, speaker similarity, and Mel Cepstral Distortion, can be used. Moreover, our findings underscore significant limitations of common metrics like Word Error Rate in assessing underrepresented accents. 

---
# OmniGenBench: A Modular Platform for Reproducible Genomic Foundation Models Benchmarking 

**Authors**: Heng Yang, Jack Cole, Yuan Li, Renzhi Chen, Geyong Min, Ke Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.14402)  

**Abstract**: The code of nature, embedded in DNA and RNA genomes since the origin of life, holds immense potential to impact both humans and ecosystems through genome modeling. Genomic Foundation Models (GFMs) have emerged as a transformative approach to decoding the genome. As GFMs scale up and reshape the landscape of AI-driven genomics, the field faces an urgent need for rigorous and reproducible evaluation. We present OmniGenBench, a modular benchmarking platform designed to unify the data, model, benchmarking, and interpretability layers across GFMs. OmniGenBench enables standardized, one-command evaluation of any GFM across five benchmark suites, with seamless integration of over 31 open-source models. Through automated pipelines and community-extensible features, the platform addresses critical reproducibility challenges, including data transparency, model interoperability, benchmark fragmentation, and black-box interpretability. OmniGenBench aims to serve as foundational infrastructure for reproducible genomic AI research, accelerating trustworthy discovery and collaborative innovation in the era of genome-scale modeling. 

---
# Causal Cartographer: From Mapping to Reasoning Over Counterfactual Worlds 

**Authors**: Gaël Gendron, Jože M. Rožanec, Michael Witbrock, Gillian Dobbie  

**Link**: [PDF](https://arxiv.org/pdf/2505.14396)  

**Abstract**: Causal world models are systems that can answer counterfactual questions about an environment of interest, i.e. predict how it would have evolved if an arbitrary subset of events had been realized differently. It requires understanding the underlying causes behind chains of events and conducting causal inference for arbitrary unseen distributions. So far, this task eludes foundation models, notably large language models (LLMs), which do not have demonstrated causal reasoning capabilities beyond the memorization of existing causal relationships. Furthermore, evaluating counterfactuals in real-world applications is challenging since only the factual world is observed, limiting evaluation to synthetic datasets. We address these problems by explicitly extracting and modeling causal relationships and propose the Causal Cartographer framework. First, we introduce a graph retrieval-augmented generation agent tasked to retrieve causal relationships from data. This approach allows us to construct a large network of real-world causal relationships that can serve as a repository of causal knowledge and build real-world counterfactuals. In addition, we create a counterfactual reasoning agent constrained by causal relationships to perform reliable step-by-step causal inference. We show that our approach can extract causal knowledge and improve the robustness of LLMs for causal reasoning tasks while reducing inference costs and spurious correlations. 

---
# Is Your Prompt Safe? Investigating Prompt Injection Attacks Against Open-Source LLMs 

**Authors**: Jiawen Wang, Pritha Gupta, Ivan Habernal, Eyke Hüllermeier  

**Link**: [PDF](https://arxiv.org/pdf/2505.14368)  

**Abstract**: Recent studies demonstrate that Large Language Models (LLMs) are vulnerable to different prompt-based attacks, generating harmful content or sensitive information. Both closed-source and open-source LLMs are underinvestigated for these attacks. This paper studies effective prompt injection attacks against the $\mathbf{14}$ most popular open-source LLMs on five attack benchmarks. Current metrics only consider successful attacks, whereas our proposed Attack Success Probability (ASP) also captures uncertainty in the model's response, reflecting ambiguity in attack feasibility. By comprehensively analyzing the effectiveness of prompt injection attacks, we propose a simple and effective hypnotism attack; results show that this attack causes aligned language models, including Stablelm2, Mistral, Openchat, and Vicuna, to generate objectionable behaviors, achieving around $90$% ASP. They also indicate that our ignore prefix attacks can break all $\mathbf{14}$ open-source LLMs, achieving over $60$% ASP on a multi-categorical dataset. We find that moderately well-known LLMs exhibit higher vulnerability to prompt injection attacks, highlighting the need to raise public awareness and prioritize efficient mitigation strategies. 

---
# PersonaTAB: Predicting Personality Traits using Textual, Acoustic, and Behavioral Cues in Fully-Duplex Speech Dialogs 

**Authors**: Sho Inoue, Shai Wang, Haizhou Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.14356)  

**Abstract**: Despite significant progress in neural spoken dialog systems, personality-aware conversation agents -- capable of adapting behavior based on personalities -- remain underexplored due to the absence of personality annotations in speech datasets. We propose a pipeline that preprocesses raw audio recordings to create a dialogue dataset annotated with timestamps, response types, and emotion/sentiment labels. We employ an automatic speech recognition (ASR) system to extract transcripts and timestamps, then generate conversation-level annotations. Leveraging these annotations, we design a system that employs large language models to predict conversational personality. Human evaluators were engaged to identify conversational characteristics and assign personality labels. Our analysis demonstrates that the proposed system achieves stronger alignment with human judgments compared to existing approaches. 

---
# FMSD-TTS: Few-shot Multi-Speaker Multi-Dialect Text-to-Speech Synthesis for Ü-Tsang, Amdo and Kham Speech Dataset Generation 

**Authors**: Yutong Liu, Ziyue Zhang, Ban Ma-bao, Yuqing Cai, Yongbin Yu, Renzeng Duojie, Xiangxiang Wang, Fan Gao, Cheng Huang, Nyima Tashi  

**Link**: [PDF](https://arxiv.org/pdf/2505.14351)  

**Abstract**: Tibetan is a low-resource language with minimal parallel speech corpora spanning its three major dialects-Ü-Tsang, Amdo, and Kham-limiting progress in speech modeling. To address this issue, we propose FMSD-TTS, a few-shot, multi-speaker, multi-dialect text-to-speech framework that synthesizes parallel dialectal speech from limited reference audio and explicit dialect labels. Our method features a novel speaker-dialect fusion module and a Dialect-Specialized Dynamic Routing Network (DSDR-Net) to capture fine-grained acoustic and linguistic variations across dialects while preserving speaker identity. Extensive objective and subjective evaluations demonstrate that FMSD-TTS significantly outperforms baselines in both dialectal expressiveness and speaker similarity. We further validate the quality and utility of the synthesized speech through a challenging speech-to-speech dialect conversion task. Our contributions include: (1) a novel few-shot TTS system tailored for Tibetan multi-dialect speech synthesis, (2) the public release of a large-scale synthetic Tibetan speech corpus generated by FMSD-TTS, and (3) an open-source evaluation toolkit for standardized assessment of speaker similarity, dialect consistency, and audio quality. 

---
# RADAR: Enhancing Radiology Report Generation with Supplementary Knowledge Injection 

**Authors**: Wenjun Hou, Yi Cheng, Kaishuai Xu, Heng Li, Yan Hu, Wenjie Li, Jiang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.14318)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities in various domains, including radiology report generation. Previous approaches have attempted to utilize multimodal LLMs for this task, enhancing their performance through the integration of domain-specific knowledge retrieval. However, these approaches often overlook the knowledge already embedded within the LLMs, leading to redundant information integration and inefficient utilization of learned representations. To address this limitation, we propose RADAR, a framework for enhancing radiology report generation with supplementary knowledge injection. RADAR improves report generation by systematically leveraging both the internal knowledge of an LLM and externally retrieved information. Specifically, it first extracts the model's acquired knowledge that aligns with expert image-based classification outputs. It then retrieves relevant supplementary knowledge to further enrich this information. Finally, by aggregating both sources, RADAR generates more accurate and informative radiology reports. Extensive experiments on MIMIC-CXR, CheXpert-Plus, and IU X-ray demonstrate that our model outperforms state-of-the-art LLMs in both language quality and clinical accuracy 

---
# Scaling Law for Quantization-Aware Training 

**Authors**: Mengzhao Chen, Chaoyi Zhang, Jing Liu, Yutao Zeng, Zeyue Xue, Zhiheng Liu, Yunshui Li, Jin Ma, Jie Huang, Xun Zhou, Ping Luo  

**Link**: [PDF](https://arxiv.org/pdf/2505.14302)  

**Abstract**: Large language models (LLMs) demand substantial computational and memory resources, creating deployment challenges. Quantization-aware training (QAT) addresses these challenges by reducing model precision while maintaining performance. However, the scaling behavior of QAT, especially at 4-bit precision (W4A4), is not well understood. Existing QAT scaling laws often ignore key factors such as the number of training tokens and quantization granularity, which limits their applicability. This paper proposes a unified scaling law for QAT that models quantization error as a function of model size, training data volume, and quantization group size. Through 268 QAT experiments, we show that quantization error decreases as model size increases, but rises with more training tokens and coarser quantization granularity. To identify the sources of W4A4 quantization error, we decompose it into weight and activation components. Both components follow the overall trend of W4A4 quantization error, but with different sensitivities. Specifically, weight quantization error increases more rapidly with more training tokens. Further analysis shows that the activation quantization error in the FC2 layer, caused by outliers, is the primary bottleneck of W4A4 QAT quantization error. By applying mixed-precision quantization to address this bottleneck, we demonstrate that weight and activation quantization errors can converge to similar levels. Additionally, with more training data, weight quantization error eventually exceeds activation quantization error, suggesting that reducing weight quantization error is also important in such scenarios. These findings offer key insights for improving QAT research and development. 

---
# SafetyNet: Detecting Harmful Outputs in LLMs by Modeling and Monitoring Deceptive Behaviors 

**Authors**: Maheep Chaudhary, Fazl Barez  

**Link**: [PDF](https://arxiv.org/pdf/2505.14300)  

**Abstract**: High-risk industries like nuclear and aviation use real-time monitoring to detect dangerous system conditions. Similarly, Large Language Models (LLMs) need monitoring safeguards. We propose a real-time framework to predict harmful AI outputs before they occur by using an unsupervised approach that treats normal behavior as the baseline and harmful outputs as outliers. Our study focuses specifically on backdoor-triggered responses -- where specific input phrases activate hidden vulnerabilities causing the model to generate unsafe content like violence, pornography, or hate speech. We address two key challenges: (1) identifying true causal indicators rather than surface correlations, and (2) preventing advanced models from deception -- deliberately evading monitoring systems. Hence, we approach this problem from an unsupervised lens by drawing parallels to human deception: just as humans exhibit physical indicators while lying, we investigate whether LLMs display distinct internal behavioral signatures when generating harmful content. Our study addresses two critical challenges: 1) designing monitoring systems that capture true causal indicators rather than superficial correlations; and 2)preventing intentional evasion by increasingly capable "Future models''. Our findings show that models can produce harmful content through causal mechanisms and can become deceptive by: (a) alternating between linear and non-linear representations, and (b) modifying feature relationships. To counter this, we developed Safety-Net -- a multi-detector framework that monitors different representation dimensions, successfully detecting harmful behavior even when information is shifted across representational spaces to evade individual monitors. Our evaluation shows 96% accuracy in detecting harmful cases using our unsupervised ensemble approach. 

---
# AAPO: Enhance the Reasoning Capabilities of LLMs with Advantage Momentum 

**Authors**: Jian Xiong, Jingbo Zhou, Jingyong Ye, Dejing Dou  

**Link**: [PDF](https://arxiv.org/pdf/2505.14264)  

**Abstract**: Reinforcement learning (RL) has emerged as an effective approach for enhancing the reasoning capabilities of large language models (LLMs), especially in scenarios where supervised fine-tuning (SFT) falls short due to limited chain-of-thought (CoT) data. Among RL-based post-training methods, group relative advantage estimation, as exemplified by Group Relative Policy Optimization (GRPO), has attracted considerable attention for eliminating the dependency on the value model, thereby simplifying training compared to traditional approaches like Proximal Policy Optimization (PPO). However, we observe that exsiting group relative advantage estimation method still suffers from training inefficiencies, particularly when the estimated advantage approaches zero. To address this limitation, we propose Advantage-Augmented Policy Optimization (AAPO), a novel RL algorithm that optimizes the cross-entropy (CE) loss using advantages enhanced through a momentum-based estimation scheme. This approach effectively mitigates the inefficiencies associated with group relative advantage estimation. Experimental results on multiple mathematical reasoning benchmarks demonstrate the superior performance of AAPO. 

---
# Reinforcement Learning vs. Distillation: Understanding Accuracy and Capability in LLM Reasoning 

**Authors**: Minwu Kim, Anubhav Shrestha, Safal Shrestha, Aadim Nepal, Keith Ross  

**Link**: [PDF](https://arxiv.org/pdf/2505.14216)  

**Abstract**: Recent studies have shown that reinforcement learning with verifiable rewards (RLVR) enhances overall accuracy but fails to improve capability, while distillation can improve both. In this paper, we investigate the mechanisms behind these phenomena. First, we demonstrate that RLVR does not improve capability because it focuses on improving the accuracy of the less-difficult questions to the detriment of the accuracy of the most difficult questions, thereby leading to no improvement in capability. Second, we find that RLVR does not merely increase the success probability for the less difficult questions, but in our small model settings produces quality responses that were absent in its output distribution before training. In addition, we show these responses are neither noticeably longer nor feature more reflection-related keywords, underscoring the need for more reliable indicators of response quality. Third, we show that while distillation reliably improves accuracy by learning strong reasoning patterns, it only improves capability when new knowledge is introduced. Moreover, when distilling only with reasoning patterns and no new knowledge, the accuracy of the less-difficult questions improves to the detriment of the most difficult questions, similar to RLVR. Together, these findings offer a clearer understanding of how RLVR and distillation shape reasoning behavior in language models. 

---
# Safety Subspaces are Not Distinct: A Fine-Tuning Case Study 

**Authors**: Kaustubh Ponkshe, Shaan Shah, Raghav Singhal, Praneeth Vepakomma  

**Link**: [PDF](https://arxiv.org/pdf/2505.14185)  

**Abstract**: Large Language Models (LLMs) rely on safety alignment to produce socially acceptable responses. This is typically achieved through instruction tuning and reinforcement learning from human feedback. However, this alignment is known to be brittle: further fine-tuning, even on benign or lightly contaminated data, can degrade safety and reintroduce harmful behaviors. A growing body of work suggests that alignment may correspond to identifiable geometric directions in weight space, forming subspaces that could, in principle, be isolated or preserved to defend against misalignment. In this work, we conduct a comprehensive empirical study of this geometric perspective. We examine whether safety-relevant behavior is concentrated in specific subspaces, whether it can be separated from general-purpose learning, and whether harmfulness arises from distinguishable patterns in internal representations. Across both parameter and activation space, our findings are consistent: subspaces that amplify safe behaviors also amplify unsafe ones, and prompts with different safety implications activate overlapping representations. We find no evidence of a subspace that selectively governs safety. These results challenge the assumption that alignment is geometrically localized. Rather than residing in distinct directions, safety appears to emerge from entangled, high-impact components of the model's broader learning dynamics. This suggests that subspace-based defenses may face fundamental limitations and underscores the need for alternative strategies to preserve alignment under continued training. We corroborate these findings through multiple experiments on five open-source LLMs. Our code is publicly available at: this https URL. 

---
# s3: You Don't Need That Much Data to Train a Search Agent via RL 

**Authors**: Pengcheng Jiang, Xueqiang Xu, Jiacheng Lin, Jinfeng Xiao, Zifeng Wang, Jimeng Sun, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.14146)  

**Abstract**: Retrieval-augmented generation (RAG) systems empower large language models (LLMs) to access external knowledge during inference. Recent advances have enabled LLMs to act as search agents via reinforcement learning (RL), improving information acquisition through multi-turn interactions with retrieval engines. However, existing approaches either optimize retrieval using search-only metrics (e.g., NDCG) that ignore downstream utility or fine-tune the entire LLM to jointly reason and retrieve-entangling retrieval with generation and limiting the real search utility and compatibility with frozen or proprietary models. In this work, we propose s3, a lightweight, model-agnostic framework that decouples the searcher from the generator and trains the searcher using a Gain Beyond RAG reward: the improvement in generation accuracy over naive RAG. s3 requires only 2.4k training samples to outperform baselines trained on over 70x more data, consistently delivering stronger downstream performance across six general QA and five medical QA benchmarks. 

---
# Textual Steering Vectors Can Improve Visual Understanding in Multimodal Large Language Models 

**Authors**: Woody Haosheng Gan, Deqing Fu, Julian Asilis, Ollie Liu, Dani Yogatama, Vatsal Sharan, Robin Jia, Willie Neiswanger  

**Link**: [PDF](https://arxiv.org/pdf/2505.14071)  

**Abstract**: Steering methods have emerged as effective and targeted tools for guiding large language models' (LLMs) behavior without modifying their parameters. Multimodal large language models (MLLMs), however, do not currently enjoy the same suite of techniques, due in part to their recency and architectural diversity. Inspired by this gap, we investigate whether MLLMs can be steered using vectors derived from their text-only LLM backbone, via sparse autoencoders (SAEs), mean shift, and linear probing. We find that text-derived steering consistently enhances multimodal accuracy across diverse MLLM architectures and visual tasks. In particular, mean shift boosts spatial relationship accuracy on CV-Bench by up to +7.3% and counting accuracy by up to +3.3%, outperforming prompting and exhibiting strong generalization to out-of-distribution datasets. These results highlight textual steering vectors as a powerful, efficient mechanism for enhancing grounding in MLLMs with minimal additional data collection and computational overhead. 

---
# ProMind-LLM: Proactive Mental Health Care via Causal Reasoning with Sensor Data 

**Authors**: Xinzhe Zheng, Sijie Ji, Jiawei Sun, Renqi Chen, Wei Gao, Mani Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2505.14038)  

**Abstract**: Mental health risk is a critical global public health challenge, necessitating innovative and reliable assessment methods. With the development of large language models (LLMs), they stand out to be a promising tool for explainable mental health care applications. Nevertheless, existing approaches predominantly rely on subjective textual mental records, which can be distorted by inherent mental uncertainties, leading to inconsistent and unreliable predictions. To address these limitations, this paper introduces ProMind-LLM. We investigate an innovative approach integrating objective behavior data as complementary information alongside subjective mental records for robust mental health risk assessment. Specifically, ProMind-LLM incorporates a comprehensive pipeline that includes domain-specific pretraining to tailor the LLM for mental health contexts, a self-refine mechanism to optimize the processing of numerical behavioral data, and causal chain-of-thought reasoning to enhance the reliability and interpretability of its predictions. Evaluations of two real-world datasets, PMData and Globem, demonstrate the effectiveness of our proposed methods, achieving substantial improvements over general LLMs. We anticipate that ProMind-LLM will pave the way for more dependable, interpretable, and scalable mental health case solutions. 

---
# Beyond Text: Unveiling Privacy Vulnerabilities in Multi-modal Retrieval-Augmented Generation 

**Authors**: Jiankun Zhang, Shenglai Zeng, Jie Ren, Tianqi Zheng, Hui Liu, Xianfeng Tang, Hui Liu, Yi Chang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13957)  

**Abstract**: Multimodal Retrieval-Augmented Generation (MRAG) systems enhance LMMs by integrating external multimodal databases, but introduce unexplored privacy vulnerabilities. While text-based RAG privacy risks have been studied, multimodal data presents unique challenges. We provide the first systematic analysis of MRAG privacy vulnerabilities across vision-language and speech-language modalities. Using a novel compositional structured prompt attack in a black-box setting, we demonstrate how attackers can extract private information by manipulating queries. Our experiments reveal that LMMs can both directly generate outputs resembling retrieved content and produce descriptions that indirectly expose sensitive information, highlighting the urgent need for robust privacy-preserving MRAG techniques. 

---
# MLZero: A Multi-Agent System for End-to-end Machine Learning Automation 

**Authors**: Haoyang Fang, Boran Han, Nick Erickson, Xiyuan Zhang, Su Zhou, Anirudh Dagar, Jiani Zhang, Ali Caner Turkmen, Cuixiong Hu, Huzefa Rangwala, Ying Nian Wu, Bernie Wang, George Karypis  

**Link**: [PDF](https://arxiv.org/pdf/2505.13941)  

**Abstract**: Existing AutoML systems have advanced the automation of machine learning (ML); however, they still require substantial manual configuration and expert input, particularly when handling multimodal data. We introduce MLZero, a novel multi-agent framework powered by Large Language Models (LLMs) that enables end-to-end ML automation across diverse data modalities with minimal human intervention. A cognitive perception module is first employed, transforming raw multimodal inputs into perceptual context that effectively guides the subsequent workflow. To address key limitations of LLMs, such as hallucinated code generation and outdated API knowledge, we enhance the iterative code generation process with semantic and episodic memory. MLZero demonstrates superior performance on MLE-Bench Lite, outperforming all competitors in both success rate and solution quality, securing six gold medals. Additionally, when evaluated on our Multimodal AutoML Agent Benchmark, which includes 25 more challenging tasks spanning diverse data modalities, MLZero outperforms the competing methods by a large margin with a success rate of 0.92 (+263.6\%) and an average rank of 2.28. Our approach maintains its robust effectiveness even with a compact 8B LLM, outperforming full-size systems from existing solutions. 

---
# Efficient Agent Training for Computer Use 

**Authors**: Yanheng He, Jiahe Jin, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13909)  

**Abstract**: Scaling up high-quality trajectory data has long been a critical bottleneck for developing human-like computer use agents. We introduce PC Agent-E, an efficient agent training framework that significantly reduces reliance on large-scale human demonstrations. Starting with just 312 human-annotated computer use trajectories, we further improved data quality by synthesizing diverse action decisions with Claude 3.7 Sonnet. Trained on these enriched trajectories, our PC Agent-E model achieved a remarkable 141% relative improvement, surpassing the strong Claude 3.7 Sonnet with extended thinking on WindowsAgentArena-V2, an improved benchmark we also released. Furthermore, PC Agent-E demonstrates strong generalizability to different operating systems on OSWorld. Our findings suggest that strong computer use capabilities can be stimulated from a small amount of high-quality trajectory data. 

---
# Mobile-Agent-V: A Video-Guided Approach for Effortless and Efficient Operational Knowledge Injection in Mobile Automation 

**Authors**: Junyang Wang, Haiyang Xu, Xi Zhang, Ming Yan, Ji Zhang, Fei Huang, Jitao Sang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13887)  

**Abstract**: The exponential rise in mobile device usage necessitates streamlined automation for effective task management, yet many AI frameworks fall short due to inadequate operational expertise. While manually written knowledge can bridge this gap, it is often burdensome and inefficient. We introduce Mobile-Agent-V, an innovative framework that utilizes video as a guiding tool to effortlessly and efficiently inject operational knowledge into mobile automation processes. By deriving knowledge directly from video content, Mobile-Agent-V eliminates manual intervention, significantly reducing the effort and time required for knowledge acquisition. To rigorously evaluate this approach, we propose Mobile-Knowledge, a benchmark tailored to assess the impact of external knowledge on mobile agent performance. Our experimental findings demonstrate that Mobile-Agent-V enhances performance by 36% compared to existing methods, underscoring its effortless and efficient advantages in mobile automation. 

---
# InfiFPO: Implicit Model Fusion via Preference Optimization in Large Language Models 

**Authors**: Yanggan Gu, Zhaoyi Yan, Yuanyi Wang, Yiming Zhang, Qi Zhou, Fei Wu, Hongxia Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13878)  

**Abstract**: Model fusion combines multiple Large Language Models (LLMs) with different strengths into a more powerful, integrated model through lightweight training methods. Existing works on model fusion focus primarily on supervised fine-tuning (SFT), leaving preference alignment (PA) --a critical phase for enhancing LLM performance--largely unexplored. The current few fusion methods on PA phase, like WRPO, simplify the process by utilizing only response outputs from source models while discarding their probability information. To address this limitation, we propose InfiFPO, a preference optimization method for implicit model fusion. InfiFPO replaces the reference model in Direct Preference Optimization (DPO) with a fused source model that synthesizes multi-source probabilities at the sequence level, circumventing complex vocabulary alignment challenges in previous works and meanwhile maintaining the probability information. By introducing probability clipping and max-margin fusion strategies, InfiFPO enables the pivot model to align with human preferences while effectively distilling knowledge from source models. Comprehensive experiments on 11 widely-used benchmarks demonstrate that InfiFPO consistently outperforms existing model fusion and preference optimization methods. When using Phi-4 as the pivot model, InfiFPO improve its average performance from 79.95 to 83.33 on 11 benchmarks, significantly improving its capabilities in mathematics, coding, and reasoning tasks. 

---
# PandaGuard: Systematic Evaluation of LLM Safety in the Era of Jailbreaking Attacks 

**Authors**: Guobin Shen, Dongcheng Zhao, Linghao Feng, Xiang He, Jihang Wang, Sicheng Shen, Haibo Tong, Yiting Dong, Jindong Li, Xiang Zheng, Yi Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2505.13862)  

**Abstract**: Large language models (LLMs) have achieved remarkable capabilities but remain vulnerable to adversarial prompts known as jailbreaks, which can bypass safety alignment and elicit harmful outputs. Despite growing efforts in LLM safety research, existing evaluations are often fragmented, focused on isolated attack or defense techniques, and lack systematic, reproducible analysis. In this work, we introduce PandaGuard, a unified and modular framework that models LLM jailbreak safety as a multi-agent system comprising attackers, defenders, and judges. Our framework implements 19 attack methods and 12 defense mechanisms, along with multiple judgment strategies, all within a flexible plugin architecture supporting diverse LLM interfaces, multiple interaction modes, and configuration-driven experimentation that enhances reproducibility and practical deployment. Built on this framework, we develop PandaBench, a comprehensive benchmark that evaluates the interactions between these attack/defense methods across 49 LLMs and various judgment approaches, requiring over 3 billion tokens to execute. Our extensive evaluation reveals key insights into model vulnerabilities, defense cost-performance trade-offs, and judge consistency. We find that no single defense is optimal across all dimensions and that judge disagreement introduces nontrivial variance in safety assessments. We release the code, configurations, and evaluation results to support transparent and reproducible research in LLM safety. 

---
# Forensic deepfake audio detection using segmental speech features 

**Authors**: Tianle Yang, Chengzhe Sun, Siwei Lyu, Phil Rose  

**Link**: [PDF](https://arxiv.org/pdf/2505.13847)  

**Abstract**: This study explores the potential of using acoustic features of segmental speech sounds to detect deepfake audio. These features are highly interpretable because of their close relationship with human articulatory processes and are expected to be more difficult for deepfake models to replicate. The results demonstrate that certain segmental features commonly used in forensic voice comparison are effective in identifying deep-fakes, whereas some global features provide little value. These findings underscore the need to approach audio deepfake detection differently for forensic voice comparison and offer a new perspective on leveraging segmental features for this purpose. 

---
# Structured Agent Distillation for Large Language Model 

**Authors**: Jun Liu, Zhenglun Kong, Peiyan Dong, Changdi Yang, Tianqi Li, Hao Tang, Geng Yuan, Wei Niu, Wenbin Zhang, Pu Zhao, Xue Lin, Dong Huang, Yanzhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13820)  

**Abstract**: Large language models (LLMs) exhibit strong capabilities as decision-making agents by interleaving reasoning and actions, as seen in ReAct-style frameworks. Yet, their practical deployment is constrained by high inference costs and large model sizes. We propose Structured Agent Distillation, a framework that compresses large LLM-based agents into smaller student models while preserving both reasoning fidelity and action consistency. Unlike standard token-level distillation, our method segments trajectories into {[REASON]} and {[ACT]} spans, applying segment-specific losses to align each component with the teacher's behavior. This structure-aware supervision enables compact agents to better replicate the teacher's decision process. Experiments on ALFWorld, HotPotQA-ReAct, and WebShop show that our approach consistently outperforms token-level and imitation learning baselines, achieving significant compression with minimal performance drop. Scaling and ablation results further highlight the importance of span-level alignment for efficient and deployable agents. 

---
# Ice Cream Doesn't Cause Drowning: Benchmarking LLMs Against Statistical Pitfalls in Causal Inference 

**Authors**: Jin Du, Li Chen, Xun Xian, An Luo, Fangqiao Tian, Ganghua Wang, Charles Doss, Xiaotong Shen, Jie Ding  

**Link**: [PDF](https://arxiv.org/pdf/2505.13770)  

**Abstract**: Reliable causal inference is essential for making decisions in high-stakes areas like medicine, economics, and public policy. However, it remains unclear whether large language models (LLMs) can handle rigorous and trustworthy statistical causal inference. Current benchmarks usually involve simplified tasks. For example, these tasks might only ask LLMs to identify semantic causal relationships or draw conclusions directly from raw data. As a result, models may overlook important statistical pitfalls, such as Simpson's paradox or selection bias. This oversight limits the applicability of LLMs in the real world. To address these limitations, we propose CausalPitfalls, a comprehensive benchmark designed to rigorously evaluate the capability of LLMs in overcoming common causal inference pitfalls. Our benchmark features structured challenges across multiple difficulty levels, each paired with grading rubrics. This approach allows us to quantitatively measure both causal reasoning capabilities and the reliability of LLMs' responses. We evaluate models using two protocols: (1) direct prompting, which assesses intrinsic causal reasoning, and (2) code-assisted prompting, where models generate executable code for explicit statistical analysis. Additionally, we validate the effectiveness of this judge by comparing its scoring with assessments from human experts. Our results reveal significant limitations in current LLMs when performing statistical causal inference. The CausalPitfalls benchmark provides essential guidance and quantitative metrics to advance the development of trustworthy causal reasoning systems. 

---
# Advancing Software Quality: A Standards-Focused Review of LLM-Based Assurance Techniques 

**Authors**: Avinash Patil  

**Link**: [PDF](https://arxiv.org/pdf/2505.13766)  

**Abstract**: Software Quality Assurance (SQA) is critical for delivering reliable, secure, and efficient software products. The Software Quality Assurance Process aims to provide assurance that work products and processes comply with predefined provisions and plans. Recent advancements in Large Language Models (LLMs) present new opportunities to enhance existing SQA processes by automating tasks like requirement analysis, code review, test generation, and compliance checks. Simultaneously, established standards such as ISO/IEC 12207, ISO/IEC 25010, ISO/IEC 5055, ISO 9001/ISO/IEC 90003, CMMI, and TMM provide structured frameworks for ensuring robust quality practices. This paper surveys the intersection of LLM-based SQA methods and these recognized standards, highlighting how AI-driven solutions can augment traditional approaches while maintaining compliance and process maturity. We first review the foundational software quality standards and the technical fundamentals of LLMs in software engineering. Next, we explore various LLM-based SQA applications, including requirement validation, defect detection, test generation, and documentation maintenance. We then map these applications to key software quality frameworks, illustrating how LLMs can address specific requirements and metrics within each standard. Empirical case studies and open-source initiatives demonstrate the practical viability of these methods. At the same time, discussions on challenges (e.g., data privacy, model bias, explainability) underscore the need for deliberate governance and auditing. Finally, we propose future directions encompassing adaptive learning, privacy-focused deployments, multimodal analysis, and evolving standards for AI-driven software quality. 

---
# Language Models Are Capable of Metacognitive Monitoring and Control of Their Internal Activations 

**Authors**: Li Ji-An, Hua-Dong Xiong, Robert C. Wilson, Marcelo G. Mattar, Marcus K. Benna  

**Link**: [PDF](https://arxiv.org/pdf/2505.13763)  

**Abstract**: Large language models (LLMs) can sometimes report the strategies they actually use to solve tasks, but they can also fail to do so. This suggests some degree of metacognition -- the capacity to monitor one's own cognitive processes for subsequent reporting and self-control. Metacognitive abilities enhance AI capabilities but raise safety concerns, as models might obscure their internal processes to evade neural-activation-based oversight mechanisms designed to detect harmful behaviors. Given society's increased reliance on these models, it is critical that we understand the limits of their metacognitive abilities, particularly their ability to monitor their internal activations. To address this, we introduce a neuroscience-inspired neurofeedback paradigm designed to quantify the ability of LLMs to explicitly report and control their activation patterns. By presenting models with sentence-label pairs where labels correspond to sentence-elicited internal activations along specific directions in the neural representation space, we demonstrate that LLMs can learn to report and control these activations. The performance varies with several factors: the number of example pairs provided, the semantic interpretability of the target neural direction, and the variance explained by that direction. These results reveal a "metacognitive space" with dimensionality much lower than the model's neural space, suggesting LLMs can monitor only a subset of their neural mechanisms. Our findings provide empirical evidence quantifying metacognitive capabilities in LLMs, with significant implications for AI safety. 

---
# LLM-Based Compact Reranking with Document Features for Scientific Retrieval 

**Authors**: Runchu Tian, Xueqiang Xu, Bowen Jin, SeongKu Kang, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.13757)  

**Abstract**: Scientific retrieval is essential for advancing academic discovery. Within this process, document reranking plays a critical role by refining first-stage retrieval results. However, large language model (LLM) listwise reranking faces unique challenges in the scientific domain. First-stage retrieval is often suboptimal in the scientific domain, so relevant documents are ranked lower. Moreover, conventional listwise reranking uses the full text of candidate documents in the context window, limiting the number of candidates that can be considered. As a result, many relevant documents are excluded before reranking, which constrains overall retrieval performance. To address these challenges, we explore compact document representations based on semantic features such as categories, sections, and keywords, and propose a training-free, model-agnostic reranking framework for scientific retrieval called CoRank. The framework involves three stages: (i) offline extraction of document-level features, (ii) coarse reranking using these compact representations, and (iii) fine-grained reranking on full texts of the top candidates from stage (ii). This hybrid design provides a high-level abstraction of document semantics, expands candidate coverage, and retains critical details required for precise ranking. Experiments on LitSearch and CSFCube show that CoRank significantly improves reranking performance across different LLM backbones, increasing nDCG@10 from 32.0 to 39.7. Overall, these results highlight the value of information extraction for reranking in scientific retrieval. 

---
# Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training 

**Authors**: Shane Bergsma, Nolan Dey, Gurpreet Gosal, Gavia Gray, Daria Soboleva, Joel Hestness  

**Link**: [PDF](https://arxiv.org/pdf/2505.13738)  

**Abstract**: Efficient LLM pre-training requires well-tuned hyperparameters (HPs), including learning rate {\eta} and weight decay {\lambda}. We study scaling laws for HPs: formulas for how to scale HPs as we scale model size N, dataset size D, and batch size B. Recent work suggests the AdamW timescale, B/({\eta}{\lambda}D), should remain constant across training settings, and we verify the implication that optimal {\lambda} scales linearly with B, for a fixed N,D. However, as N,D scale, we show the optimal timescale obeys a precise power law in the tokens-per-parameter ratio, D/N. This law thus provides a method to accurately predict {\lambda}opt in advance of large-scale training. We also study scaling laws for optimal batch size Bopt (the B enabling lowest loss at a given N,D) and critical batch size Bcrit (the B beyond which further data parallelism becomes ineffective). In contrast with prior work, we find both Bopt and Bcrit scale as power laws in D, independent of model size, N. Finally, we analyze how these findings inform the real-world selection of Pareto-optimal N and D under dual training time and compute objectives. 

---
# Warm Up Before You Train: Unlocking General Reasoning in Resource-Constrained Settings 

**Authors**: Safal Shrestha, Minwu Kim, Aadim Nepal, Anubhav Shrestha, Keith Ross  

**Link**: [PDF](https://arxiv.org/pdf/2505.13718)  

**Abstract**: Designing effective reasoning-capable LLMs typically requires training using Reinforcement Learning with Verifiable Rewards (RLVR) or distillation with carefully curated Long Chain of Thoughts (CoT), both of which depend heavily on extensive training data. This creates a major challenge when the amount of quality training data is scarce. We propose a sample-efficient, two-stage training strategy to develop reasoning LLMs under limited supervision. In the first stage, we "warm up" the model by distilling Long CoTs from a toy domain, namely, Knights \& Knaves (K\&K) logic puzzles to acquire general reasoning skills. In the second stage, we apply RLVR to the warmed-up model using a limited set of target-domain examples. Our experiments demonstrate that this two-phase approach offers several benefits: $(i)$ the warmup phase alone facilitates generalized reasoning, leading to performance improvements across a range of tasks, including MATH, HumanEval$^{+}$, and MMLU-Pro. $(ii)$ When both the base model and the warmed-up model are RLVR trained on the same small dataset ($\leq100$ examples), the warmed-up model consistently outperforms the base model; $(iii)$ Warming up before RLVR training allows a model to maintain cross-domain generalizability even after training on a specific domain; $(iv)$ Introducing warmup in the pipeline improves not only accuracy but also overall sample efficiency during RLVR training. The results in this paper highlight the promise of warmup for building robust reasoning LLMs in data-scarce environments. 

---
# Guided Search Strategies in Non-Serializable Environments with Applications to Software Engineering Agents 

**Authors**: Karina Zainullina, Alexander Golubev, Maria Trofimova, Sergei Polezhaev, Ibragim Badertdinov, Daria Litvintseva, Simon Karasik, Filipp Fisin, Sergei Skvortsov, Maksim Nekrashevich, Anton Shevtsov, Boris Yangel  

**Link**: [PDF](https://arxiv.org/pdf/2505.13652)  

**Abstract**: Large language models (LLMs) have recently achieved remarkable results in complex multi-step tasks, such as mathematical reasoning and agentic software engineering. However, they often struggle to maintain consistent performance across multiple solution attempts. One effective approach to narrow the gap between average-case and best-case performance is guided test-time search, which explores multiple solution paths to identify the most promising one. Unfortunately, effective search techniques (e.g. MCTS) are often unsuitable for non-serializable RL environments, such as Docker containers, where intermediate environment states cannot be easily saved and restored. We investigate two complementary search strategies applicable to such environments: 1-step lookahead and trajectory selection, both guided by a learned action-value function estimator. On the SWE-bench Verified benchmark, a key testbed for agentic software engineering, we find these methods to double the average success rate of a fine-tuned Qwen-72B model, achieving 40.8%, the new state-of-the-art for open-weights models. Additionally, we show that these techniques are transferable to more advanced closed models, yielding similar improvements with GPT-4o. 

---
# RAR: Setting Knowledge Tripwires for Retrieval Augmented Rejection 

**Authors**: Tommaso Mario Buonocore, Enea Parimbelli  

**Link**: [PDF](https://arxiv.org/pdf/2505.13581)  

**Abstract**: Content moderation for large language models (LLMs) remains a significant challenge, requiring flexible and adaptable solutions that can quickly respond to emerging threats. This paper introduces Retrieval Augmented Rejection (RAR), a novel approach that leverages a retrieval-augmented generation (RAG) architecture to dynamically reject unsafe user queries without model retraining. By strategically inserting and marking malicious documents into the vector database, the system can identify and reject harmful requests when these documents are retrieved. Our preliminary results show that RAR achieves comparable performance to embedded moderation in LLMs like Claude 3.5 Sonnet, while offering superior flexibility and real-time customization capabilities, a fundamental feature to timely address critical vulnerabilities. This approach introduces no architectural changes to existing RAG systems, requiring only the addition of specially crafted documents and a simple rejection mechanism based on retrieval results. 

---
# InterFeat: An Automated Pipeline for Finding Interesting Hypotheses in Structured Biomedical Data 

**Authors**: Dan Ofer, Michal Linial, Dafna Shahaf  

**Link**: [PDF](https://arxiv.org/pdf/2505.13534)  

**Abstract**: Finding interesting phenomena is the core of scientific discovery, but it is a manual, ill-defined concept. We present an integrative pipeline for automating the discovery of interesting simple hypotheses (feature-target relations with effect direction and a potential underlying mechanism) in structured biomedical data. The pipeline combines machine learning, knowledge graphs, literature search and Large Language Models. We formalize "interestingness" as a combination of novelty, utility and plausibility. On 8 major diseases from the UK Biobank, our pipeline consistently recovers risk factors years before their appearance in the literature. 40--53% of our top candidates were validated as interesting, compared to 0--7% for a SHAP-based baseline. Overall, 28% of 109 candidates were interesting to medical experts. The pipeline addresses the challenge of operationalizing "interestingness" scalably and for any target. We release data and code: this https URL 

---
# AdAEM: An Adaptively and Automated Extensible Measurement of LLMs' Value Difference 

**Authors**: Shitong Duan, Xiaoyuan Yi, Peng Zhang, Dongkuan Xu, Jing Yao, Tun Lu, Ning Gu, Xing Xie  

**Link**: [PDF](https://arxiv.org/pdf/2505.13531)  

**Abstract**: Assessing Large Language Models (LLMs)' underlying value differences enables comprehensive comparison of their misalignment, cultural adaptability, and biases. Nevertheless, current value measurement datasets face the informativeness challenge: with often outdated, contaminated, or generic test questions, they can only capture the shared value orientations among different LLMs, leading to saturated and thus uninformative results. To address this problem, we introduce AdAEM, a novel, self-extensible assessment framework for revealing LLMs' inclinations. Distinct from previous static benchmarks, AdAEM can automatically and adaptively generate and extend its test questions. This is achieved by probing the internal value boundaries of a diverse set of LLMs developed across cultures and time periods in an in-context optimization manner. The optimization process theoretically maximizes an information-theoretic objective to extract the latest or culturally controversial topics, providing more distinguishable and informative insights about models' value differences. In this way, AdAEM is able to co-evolve with the development of LLMs, consistently tracking their value dynamics. Using AdAEM, we generate 12,310 questions grounded in Schwartz Value Theory, conduct an extensive analysis to manifest our method's validity and effectiveness, and benchmark the values of 16 LLMs, laying the groundwork for better value research. 

---
# BARREL: Boundary-Aware Reasoning for Factual and Reliable LRMs 

**Authors**: Junxiao Yang, Jinzhe Tu, Haoran Liu, Xiaoce Wang, Chujie Zheng, Zhexin Zhang, Shiyao Cui, Caishun Chen, Tiantian He, Hongning Wang, Yew-Soon Ong, Minlie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13529)  

**Abstract**: Recent advances in Large Reasoning Models (LRMs) have shown impressive capabilities in mathematical and logical reasoning. However, current LRMs rarely admit ignorance or respond with "I don't know". Instead, they often produce incorrect answers while showing undue confidence, raising concerns about their factual reliability. In this work, we identify two pathological reasoning patterns characterized by overthinking that contribute to the overconfident and incorrect answers: last-minute guessing and second-thought spiraling. To address these issues, we propose BARREL-a novel framework that promotes concise and boundary-aware factual reasoning. Our experiments show that BARREL-training increases the reliability of DeepSeek-R1-Distill-Llama-8B from 39.33% to 61.48%, while still achieving accuracy comparable to models finetuned on reasoning data generated by R1. These results demonstrate that our pilot study is inspiring to build more reliable and factual System 2 LRMs. 

---
# LoRASuite: Efficient LoRA Adaptation Across Large Language Model Upgrades 

**Authors**: Yanan Li, Fanxu Meng, Muhan Zhang, Shiai Zhu, Shangguang Wang, Mengwei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13515)  

**Abstract**: As Large Language Models (LLMs) are frequently updated, LoRA weights trained on earlier versions quickly become obsolete. The conventional practice of retraining LoRA weights from scratch on the latest model is costly, time-consuming, and environmentally detrimental, particularly as the diversity of LLMs and downstream tasks expands. This motivates a critical question: "How can we efficiently leverage existing LoRA weights to adapt to newer model versions?" To address this, we propose LoRASuite, a modular approach tailored specifically to various types of LLM updates. First, we compute a transfer matrix utilizing known parameters from both old and new LLMs. Next, we allocate corresponding layers and attention heads based on centered kernel alignment and cosine similarity metrics, respectively. A subsequent small-scale, skillful fine-tuning step ensures numerical stability. Experimental evaluations demonstrate that LoRASuite consistently surpasses small-scale vanilla LoRA methods. Notably, on backbone LLMs such as MiniCPM and Qwen, LoRASuite even exceeds the performance of full-scale LoRA retraining, with average improvements of +1.4 and +6.6 points on math tasks, respectively. Additionally, LoRASuite significantly reduces memory consumption by 5.5 GB and computational time by 78.23%. 

---
# Can AI Freelancers Compete? Benchmarking Earnings, Reliability, and Task Success at Scale 

**Authors**: David Noever, Forrest McKee  

**Link**: [PDF](https://arxiv.org/pdf/2505.13511)  

**Abstract**: This study explores Large Language Models (LLMs) as autonomous agents for real-world tasks, including freelance software development. This work presents a new benchmark that evaluates LLMs on freelance programming and data analysis tasks derived from economic data. We construct the benchmark using synthetic tasks created from a Kaggle Freelancer dataset of job postings, with all job prices standardized to USD (median fixed-project price around $250, and an average of $306). Each task is accompanied by structured input-output test cases and an estimated price tag, enabling automated correctness checking and a monetary performance valuation. This approach is inspired by OpenAI's recent SWE-Lancer benchmark (1,400 real Upwork tasks worth $1M total). Still, our framework simplifies evaluation using programmatically testable tasks and predicted price values, making it highly scalable and repeatable. On this benchmark, we evaluate four modern LLMs - Claude 3.5 Haiku, GPT-4o-mini, Qwen 2.5, and Mistral. We report each model's accuracy (task success rate and test-case pass rate) and the total "freelance earnings" it achieves (sum of prices of solved tasks). Our results show that Claude 3.5 Haiku performs best, earning approximately $1.52 million USD, followed closely by GPT-4o-mini at $1.49 million, then Qwen 2.5 ($1.33M) and Mistral ($0.70M). We analyze the distribution of errors per task and observe that the strongest models solve the most tasks and rarely fail completely on any project. We discuss the implications of these results for the feasibility of AI as a freelance developer, the advantages and limitations of our automated benchmark approach, and the gap between performance on structured tasks versus the true complexity of real-world freelance jobs. 

---
# Contrastive Cross-Course Knowledge Tracing via Concept Graph Guided Knowledge Transfer 

**Authors**: Wenkang Han, Wang Lin, Liya Hu, Zhenlong Dai, Yiyun Zhou, Mengze Li, Zemin Liu, Chang Yao, Jingyuan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.13489)  

**Abstract**: Knowledge tracing (KT) aims to predict learners' future performance based on historical learning interactions. However, existing KT models predominantly focus on data from a single course, limiting their ability to capture a comprehensive understanding of learners' knowledge states. In this paper, we propose TransKT, a contrastive cross-course knowledge tracing method that leverages concept graph guided knowledge transfer to model the relationships between learning behaviors across different courses, thereby enhancing knowledge state estimation. Specifically, TransKT constructs a cross-course concept graph by leveraging zero-shot Large Language Model (LLM) prompts to establish implicit links between related concepts across different courses. This graph serves as the foundation for knowledge transfer, enabling the model to integrate and enhance the semantic features of learners' interactions across courses. Furthermore, TransKT includes an LLM-to-LM pipeline for incorporating summarized semantic features, which significantly improves the performance of Graph Convolutional Networks (GCNs) used for knowledge transfer. Additionally, TransKT employs a contrastive objective that aligns single-course and cross-course knowledge states, thereby refining the model's ability to provide a more robust and accurate representation of learners' overall knowledge states. 

---
# Evaluating Large Language Models for Real-World Engineering Tasks 

**Authors**: Rene Heesch, Sebastian Eilermann, Alexander Windmann, Alexander Diedrich, Philipp Rosenthal, Oliver Niggemann  

**Link**: [PDF](https://arxiv.org/pdf/2505.13484)  

**Abstract**: Large Language Models (LLMs) are transformative not only for daily activities but also for engineering tasks. However, current evaluations of LLMs in engineering exhibit two critical shortcomings: (i) the reliance on simplified use cases, often adapted from examination materials where correctness is easily verifiable, and (ii) the use of ad hoc scenarios that insufficiently capture critical engineering competencies. Consequently, the assessment of LLMs on complex, real-world engineering problems remains largely unexplored. This paper addresses this gap by introducing a curated database comprising over 100 questions derived from authentic, production-oriented engineering scenarios, systematically designed to cover core competencies such as product design, prognosis, and diagnosis. Using this dataset, we evaluate four state-of-the-art LLMs, including both cloud-based and locally hosted instances, to systematically investigate their performance on complex engineering tasks. Our results show that LLMs demonstrate strengths in basic temporal and structural reasoning but struggle significantly with abstract reasoning, formal modeling, and context-sensitive engineering logic. 

---
# MedEIR: A Specialized Medical Embedding Model for Enhanced Information Retrieval 

**Authors**: Anand Selvadurai, Jasheen Shaik, Girish Chandrasekar, ShriRadhaKrishnan Balamurugan, Eswara Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2505.13482)  

**Abstract**: Embedding models have become essential for retrieval-augmented generation (RAG) tasks, semantic clustering, and text re-ranking. But despite their growing use, many of these come with notable limitations. For example, Jina fails to capture the semantic content of medical documents, while models such as MiniLM often perform poorly on long-form documents. Domain-adapted models, while specialized, often underperform in general-purpose tasks, reducing their overall applicability. General-domain tokenizers often misinterpret medical vocabulary. The limitations of current embedding models, whether in tokenization accuracy, domain comprehension, or handling long sequences, highlight the need for more versatile solutions. In this work, we present MedEIR, a novel embedding model and tokenizer jointly optimized for both medical and general NLP tasks, incorporating ALiBi-based long-context processing to support sequences of up to 8,192 tokens. MedEIR was pre-trained on only 6 billion tokens, significantly fewer than Jina's, followed by fine-tuning on 3 million sentence pairs. MedEIR consistently outperforms Jina V2 and MiniLM across MTEB benchmarks, achieving top scores on ArguAna (55.24), NFCorpus (38.44), MedicalQARetrieval (74.25), SciFact (72.04), and TRECCOVID (79.56). These results highlight the potential of MedEIR as a highly effective embedding model, demonstrating strong performance across both general-purpose and domain-specific tasks and outperforming existing models on multiple benchmarks. 

---
