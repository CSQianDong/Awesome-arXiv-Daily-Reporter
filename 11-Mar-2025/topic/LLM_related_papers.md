# LMM-R1: Empowering 3B LMMs with Strong Reasoning Abilities Through Two-Stage Rule-Based RL 

**Authors**: Yingzhe Peng, Gongrui Zhang, Miaosen Zhang, Zhiyuan You, Jie Liu, Qipeng Zhu, Kai Yang, Xingzhong Xu, Xin Geng, Xu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.07536)  

**Abstract**: Enhancing reasoning in Large Multimodal Models (LMMs) faces unique challenges from the complex interplay between visual perception and logical reasoning, particularly in compact 3B-parameter architectures where architectural constraints limit reasoning capacity and modality alignment.
While rule-based reinforcement learning (RL) excels in text-only domains, its multimodal extension confronts two critical barriers: (1) data limitations due to ambiguous answers and scarce complex reasoning examples, and (2) degraded foundational reasoning induced by multimodal pretraining.
To address these challenges, we propose \textbf{\method}, a two-stage framework adapting rule-based RL for multimodal reasoning through \textbf{Foundational Reasoning Enhancement (FRE)} followed by \textbf{Multimodal Generalization Training (MGT)}. The FRE stage first strengthens reasoning abilities using text-only data with rule-based RL, then the MGT stage generalizes these reasoning capabilities to multimodal domains.
Experiments on Qwen2.5-VL-Instruct-3B demonstrate that \method achieves 4.83\% and 4.5\% average improvements over baselines in multimodal and text-only benchmarks, respectively, with a 3.63\% gain in complex Football Game tasks. These results validate that text-based reasoning enhancement enables effective multimodal generalization, offering a data-efficient paradigm that bypasses costly high-quality multimodal training data. 

---
# Implicit Reasoning in Transformers is Reasoning through Shortcuts 

**Authors**: Tianhe Lin, Jian Xie, Siyu Yuan, Deqing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.07604)  

**Abstract**: Test-time compute is emerging as a new paradigm for enhancing language models' complex multi-step reasoning capabilities, as demonstrated by the success of OpenAI's o1 and o3, as well as DeepSeek's R1. Compared to explicit reasoning in test-time compute, implicit reasoning is more inference-efficient, requiring fewer generated tokens. However, why does the advanced reasoning capability fail to emerge in the implicit reasoning style? In this work, we train GPT-2 from scratch on a curated multi-step mathematical reasoning dataset and conduct analytical experiments to investigate how language models perform implicit reasoning in multi-step tasks. Our findings reveal: 1) Language models can perform step-by-step reasoning and achieve high accuracy in both in-domain and out-of-domain tests via implicit reasoning. However, this capability only emerges when trained on fixed-pattern data. 2) Conversely, implicit reasoning abilities emerging from training on unfixed-pattern data tend to overfit a specific pattern and fail to generalize further. Notably, this limitation is also observed in state-of-the-art large language models. These findings suggest that language models acquire implicit reasoning through shortcut learning, enabling strong performance on tasks with similar patterns while lacking generalization. 

---
# KSOD: Knowledge Supplement for LLMs On Demand 

**Authors**: Haoran Li, Junfeng Hu  

**Link**: [PDF](https://arxiv.org/pdf/2503.07550)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in various tasks, yet still produce errors in domain-specific tasks. To further improve their performance, we propose KSOD (Knowledge Supplement for LLMs On Demand), a novel framework that empowers LLMs to improve their capabilities with knowledge-based supervised fine-tuning (SFT). KSOD analyzes the causes of errors from the perspective of knowledge deficiency by identifying potential missing knowledge in LLM that may lead to the errors. Subsequently, KSOD tunes a knowledge module on knowledge dataset and verifies whether the LLM lacks the identified knowledge based on it. If the knowledge is verified, KSOD supplements the LLM with the identified knowledge using the knowledge module. Tuning LLMs on specific knowledge instead of specific task decouples task and knowledge and our experiments on two domain-specific benchmarks and four general benchmarks empirically demonstrate that KSOD enhances the performance of LLMs on tasks requiring the supplemented knowledge while preserving their performance on other tasks. Our findings shed light on the potential of improving the capabilities of LLMs with knowledge-based SFT. 

---
# XIFBench: Evaluating Large Language Models on Multilingual Instruction Following 

**Authors**: Zhenyu Li, Kehai Chen, Yunfei Long, Xuefeng Bai, Yaoyin Zhang, Xuchen Wei, Juntao Li, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.07539)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable instruction-following capabilities across various applications. However, their performance in multilingual settings remains poorly understood, as existing evaluations lack fine-grained constraint analysis. We introduce XIFBench, a comprehensive constraint-based benchmark for assessing multilingual instruction-following abilities of LLMs, featuring a novel taxonomy of five constraint categories and 465 parallel instructions across six languages spanning different resource levels. To ensure consistent cross-lingual evaluation, we develop a requirement-based protocol that leverages English requirements as semantic anchors. These requirements are then used to validate the translations across languages. Extensive experiments with various LLMs reveal notable variations in instruction-following performance across resource levels, identifying key influencing factors such as constraint categories, instruction complexity, and cultural specificity. 

---
# TokenButler: Token Importance is Predictable 

**Authors**: Yash Akhauri, Ahmed F AbouElhamayed, Yifei Gao, Chi-Chih Chang, Nilesh Jain, Mohamed S. Abdelfattah  

**Link**: [PDF](https://arxiv.org/pdf/2503.07518)  

**Abstract**: Large Language Models (LLMs) rely on the Key-Value (KV) Cache to store token history, enabling efficient decoding of tokens. As the KV-Cache grows, it becomes a major memory and computation bottleneck, however, there is an opportunity to alleviate this bottleneck, especially because prior research has shown that only a small subset of tokens contribute meaningfully to each decoding step. A key challenge in finding these critical tokens is that they are dynamic, and heavily input query-dependent. Existing methods either risk quality by evicting tokens permanently, or retain the full KV-Cache but rely on retrieving chunks (pages) of tokens at generation, failing at dense, context-rich tasks. Additionally, many existing KV-Cache sparsity methods rely on inaccurate proxies for token importance. To address these limitations, we introduce TokenButler, a high-granularity, query-aware predictor that learns to identify these critical tokens. By training a light-weight predictor with less than 1.2% parameter overhead, TokenButler prioritizes tokens based on their contextual, predicted importance. This improves perplexity & downstream accuracy by over 8% relative to SoTA methods for estimating token importance. We evaluate TokenButler on a novel synthetic small-context co-referential retrieval task, demonstrating near-oracle accuracy. Code, models and benchmarks: this https URL 

---
# Language Models Fail to Introspect About Their Knowledge of Language 

**Authors**: Siyuan Song, Jennifer Hu, Kyle Mahowald  

**Link**: [PDF](https://arxiv.org/pdf/2503.07513)  

**Abstract**: There has been recent interest in whether large language models (LLMs) can introspect about their own internal states. Such abilities would make LLMs more interpretable, and also validate the use of standard introspective methods in linguistics to evaluate grammatical knowledge in models (e.g., asking "Is this sentence grammatical?"). We systematically investigate emergent introspection across 21 open-source LLMs, in two domains where introspection is of theoretical interest: grammatical knowledge and word prediction. Crucially, in both domains, a model's internal linguistic knowledge can be theoretically grounded in direct measurements of string probability. We then evaluate whether models' responses to metalinguistic prompts faithfully reflect their internal knowledge. We propose a new measure of introspection: the degree to which a model's prompted responses predict its own string probabilities, beyond what would be predicted by another model with nearly identical internal knowledge. While both metalinguistic prompting and probability comparisons lead to high task accuracy, we do not find evidence that LLMs have privileged "self-access". Our findings complicate recent results suggesting that models can introspect, and add new evidence to the argument that prompted responses should not be conflated with models' linguistic generalizations. 

---
# LLMs syntactically adapt their language use to their conversational partner 

**Authors**: Florian Kandra, Vera Demberg, Alexander Koller  

**Link**: [PDF](https://arxiv.org/pdf/2503.07457)  

**Abstract**: It has been frequently observed that human speakers align their language use with each other during conversations. In this paper, we study empirically whether large language models (LLMs) exhibit the same behavior of conversational adaptation. We construct a corpus of conversations between LLMs and find that two LLM agents end up making more similar syntactic choices as conversations go on, confirming that modern LLMs adapt their language use to their conversational partners in at least a rudimentary way. 

---
# Is My Text in Your AI Model? Gradient-based Membership Inference Test applied to LLMs 

**Authors**: Gonzalo Mancera, Daniel de Alcala, Julian Fierrez, Ruben Tolosana, Aythami Morales  

**Link**: [PDF](https://arxiv.org/pdf/2503.07384)  

**Abstract**: This work adapts and studies the gradient-based Membership Inference Test (gMINT) to the classification of text based on LLMs. MINT is a general approach intended to determine if given data was used for training machine learning models, and this work focuses on its application to the domain of Natural Language Processing. Using gradient-based analysis, the MINT model identifies whether particular data samples were included during the language model training phase, addressing growing concerns about data privacy in machine learning. The method was evaluated in seven Transformer-based models and six datasets comprising over 2.5 million sentences, focusing on text classification tasks. Experimental results demonstrate MINTs robustness, achieving AUC scores between 85% and 99%, depending on data size and model architecture. These findings highlight MINTs potential as a scalable and reliable tool for auditing machine learning models, ensuring transparency, safeguarding sensitive data, and fostering ethical compliance in the deployment of AI/NLP technologies. 

---
# SEAP: Training-free Sparse Expert Activation Pruning Unlock the Brainpower of Large Language Models 

**Authors**: Xun Liang, Hanyu Wang, Huayi Lai, Simin Niu, Shichao Song, Jiawei Yang, Jihao Zhao, Feiyu Xiong, Bo Tang, Zhiyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.07605)  

**Abstract**: Large Language Models have achieved remarkable success across various natural language processing tasks, yet their high computational cost during inference remains a major bottleneck. This paper introduces Sparse Expert Activation Pruning (SEAP), a training-free pruning method that selectively retains task-relevant parameters to reduce inference overhead. Inspired by the clustering patterns of hidden states and activations in LLMs, SEAP identifies task-specific expert activation patterns and prunes the model while preserving task performance and enhancing computational efficiency. Experimental results demonstrate that SEAP significantly reduces computational overhead while maintaining competitive accuracy. Notably, at 50% pruning, SEAP surpasses both WandA and FLAP by over 20%, and at 20% pruning, it incurs only a 2.2% performance drop compared to the dense model. These findings highlight SEAP's scalability and effectiveness, making it a promising approach for optimizing large-scale LLMs. 

---
# Contextual Cues in Machine Translation: Investigating the Potential of Multi-Source Input Strategies in LLMs and NMT Systems 

**Authors**: Lia Shahnazaryan, Patrick Simianer, Joern Wuebker  

**Link**: [PDF](https://arxiv.org/pdf/2503.07195)  

**Abstract**: We explore the impact of multi-source input strategies on machine translation (MT) quality, comparing GPT-4o, a large language model (LLM), with a traditional multilingual neural machine translation (NMT) system. Using intermediate language translations as contextual cues, we evaluate their effectiveness in enhancing English and Chinese translations into Portuguese. Results suggest that contextual information significantly improves translation quality for domain-specific datasets and potentially for linguistically distant language pairs, with diminishing returns observed in benchmarks with high linguistic variability. Additionally, we demonstrate that shallow fusion, a multi-source approach we apply within the NMT system, shows improved results when using high-resource languages as context for other translation pairs, highlighting the importance of strategic context language selection. 

---
# Benchmarking Chinese Medical LLMs: A Medbench-based Analysis of Performance Gaps and Hierarchical Optimization Strategies 

**Authors**: Luyi Jiang, Jiayuan Chen, Lu Lu, Xinwei Peng, Lihao Liu, Junjun He, Jie Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.07306)  

**Abstract**: The evaluation and improvement of medical large language models (LLMs) are critical for their real-world deployment, particularly in ensuring accuracy, safety, and ethical alignment. Existing frameworks inadequately dissect domain-specific error patterns or address cross-modal challenges. This study introduces a granular error taxonomy through systematic analysis of top 10 models on MedBench, categorizing incorrect responses into eight types: Omissions, Hallucination, Format Mismatch, Causal Reasoning Deficiency, Contextual Inconsistency, Unanswered, Output Error, and Deficiency in Medical Language Generation. Evaluation of 10 leading models reveals vulnerabilities: despite achieving 0.86 accuracy in medical knowledge recall, critical reasoning tasks show 96.3% omission, while safety ethics evaluations expose alarming inconsistency (robustness score: 0.79) under option shuffled. Our analysis uncovers systemic weaknesses in knowledge boundary enforcement and multi-step reasoning. To address these, we propose a tiered optimization strategy spanning four levels, from prompt engineering and knowledge-augmented retrieval to hybrid neuro-symbolic architectures and causal reasoning frameworks. This work establishes an actionable roadmap for developing clinically robust LLMs while redefining evaluation paradigms through error-driven insights, ultimately advancing the safety and trustworthiness of AI in high-stakes medical environments. 

---
# MRCEval: A Comprehensive, Challenging and Accessible Machine Reading Comprehension Benchmark 

**Authors**: Shengkun Ma, Hao Peng, Lei Hou, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.07144)  

**Abstract**: Machine Reading Comprehension (MRC) is an essential task in evaluating natural language understanding. Existing MRC datasets primarily assess specific aspects of reading comprehension (RC), lacking a comprehensive MRC benchmark. To fill this gap, we first introduce a novel taxonomy that categorizes the key capabilities required for RC. Based on this taxonomy, we construct MRCEval, an MRC benchmark that leverages advanced Large Language Models (LLMs) as both sample generators and selection judges. MRCEval is a comprehensive, challenging and accessible benchmark designed to assess the RC capabilities of LLMs thoroughly, covering 13 distinct RC skills with a total of 2.1K high-quality multi-choice questions. We perform an extensive evaluation of 28 widely used open-source and proprietary models, highlighting that MRC continues to present significant challenges even in the era of LLMs. 

---
# A Graph-based Verification Framework for Fact-Checking 

**Authors**: Yani Huang, Richong Zhang, Zhijie Nie, Junfan Chen, Xuefeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.07282)  

**Abstract**: Fact-checking plays a crucial role in combating misinformation. Existing methods using large language models (LLMs) for claim decomposition face two key limitations: (1) insufficient decomposition, introducing unnecessary complexity to the verification process, and (2) ambiguity of mentions, leading to incorrect verification results. To address these challenges, we suggest introducing a claim graph consisting of triplets to address the insufficient decomposition problem and reduce mention ambiguity through graph structure. Based on this core idea, we propose a graph-based framework, GraphFC, for fact-checking. The framework features three key components: graph construction, which builds both claim and evidence graphs; graph-guided planning, which prioritizes the triplet verification order; and graph-guided checking, which verifies the triples one by one between claim and evidence graphs. Extensive experiments show that GraphFC enables fine-grained decomposition while resolving referential ambiguities through relational constraints, achieving state-of-the-art performance across three datasets. 

---
# Application of Multiple Chain-of-Thought in Contrastive Reasoning for Implicit Sentiment Analysis 

**Authors**: Liwei Yang, Xinying Wang, Xiaotang Zhou, Zhengchao Wu, Ningning Tan  

**Link**: [PDF](https://arxiv.org/pdf/2503.07140)  

**Abstract**: Implicit sentiment analysis aims to uncover emotions that are subtly expressed, often obscured by ambiguity and figurative language. To accomplish this task, large language models and multi-step reasoning are needed to identify those sentiments that are not explicitly stated. In this study, we propose a novel Dual Reverse Chain Reasoning (DRCR) framework to enhance the performance of implicit sentiment analysis. Inspired by deductive reasoning, the framework consists of three key steps: 1) hypothesize an emotional polarity and derive a reasoning process, 2) negate the initial hypothesis and derive a new reasoning process, and 3) contrast the two reasoning paths to deduce the final sentiment polarity. Building on this, we also introduce a Triple Reverse Chain Reasoning (TRCR) framework to address the limitations of random hypotheses. Both methods combine contrastive mechanisms and multi-step reasoning, significantly improving the accuracy of implicit sentiment classification. Experimental results demonstrate that both approaches outperform existing methods across various model scales, achieving state-of-the-art performance. This validates the effectiveness of combining contrastive reasoning and multi-step reasoning for implicit sentiment analysis. 

---
# DistiLLM-2: A Contrastive Approach Boosts the Distillation of LLMs 

**Authors**: Jongwoo Ko, Tianyi Chen, Sungnyun Kim, Tianyu Ding, Luming Liang, Ilya Zharkov, Se-Young Yun  

**Link**: [PDF](https://arxiv.org/pdf/2503.07067)  

**Abstract**: Despite the success of distillation in large language models (LLMs), most prior work applies identical loss functions to both teacher- and student-generated data. These strategies overlook the synergy between loss formulations and data types, leading to a suboptimal performance boost in student models. To address this, we propose DistiLLM-2, a contrastive approach that simultaneously increases the likelihood of teacher responses and decreases that of student responses by harnessing this synergy. Our extensive experiments show that DistiLLM-2 not only builds high-performing student models across a wide range of tasks, including instruction-following and code generation, but also supports diverse applications, such as preference alignment and vision-language extensions. These findings highlight the potential of a contrastive approach to enhance the efficacy of LLM distillation by effectively aligning teacher and student models across varied data types. 

---
# LLM-C3MOD: A Human-LLM Collaborative System for Cross-Cultural Hate Speech Moderation 

**Authors**: Junyeong Park, Seogyeong Jeong, Seyoung Song, Yohan Lee, Alice Oh  

**Link**: [PDF](https://arxiv.org/pdf/2503.07237)  

**Abstract**: Content moderation is a global challenge, yet major tech platforms prioritize high-resource languages, leaving low-resource languages with scarce native moderators. Since effective moderation depends on understanding contextual cues, this imbalance increases the risk of improper moderation due to non-native moderators' limited cultural understanding. Through a user study, we identify that non-native moderators struggle with interpreting culturally-specific knowledge, sentiment, and internet culture in the hate speech moderation. To assist them, we present LLM-C3MOD, a human-LLM collaborative pipeline with three steps: (1) RAG-enhanced cultural context annotations; (2) initial LLM-based moderation; and (3) targeted human moderation for cases lacking LLM consensus. Evaluated on a Korean hate speech dataset with Indonesian and German participants, our system achieves 78% accuracy (surpassing GPT-4o's 71% baseline), while reducing human workload by 83.6%. Notably, human moderators excel at nuanced contents where LLMs struggle. Our findings suggest that non-native moderators, when properly supported by LLMs, can effectively contribute to cross-cultural hate speech moderation. 

---
# MedAgentsBench: Benchmarking Thinking Models and Agent Frameworks for Complex Medical Reasoning 

**Authors**: Xiangru Tang, Daniel Shao, Jiwoong Sohn, Jiapeng Chen, Jiayi Zhang, Jinyu Xiang, Fang Wu, Yilun Zhao, Chenglin Wu, Wenqi Shi, Arman Cohan, Mark Gerstein  

**Link**: [PDF](https://arxiv.org/pdf/2503.07459)  

**Abstract**: Large Language Models (LLMs) have shown impressive performance on existing medical question-answering benchmarks. This high performance makes it increasingly difficult to meaningfully evaluate and differentiate advanced methods. We present MedAgentsBench, a benchmark that focuses on challenging medical questions requiring multi-step clinical reasoning, diagnosis formulation, and treatment planning-scenarios where current models still struggle despite their strong performance on standard tests. Drawing from seven established medical datasets, our benchmark addresses three key limitations in existing evaluations: (1) the prevalence of straightforward questions where even base models achieve high performance, (2) inconsistent sampling and evaluation protocols across studies, and (3) lack of systematic analysis of the interplay between performance, cost, and inference time. Through experiments with various base models and reasoning methods, we demonstrate that the latest thinking models, DeepSeek R1 and OpenAI o3, exhibit exceptional performance in complex medical reasoning tasks. Additionally, advanced search-based agent methods offer promising performance-to-cost ratios compared to traditional approaches. Our analysis reveals substantial performance gaps between model families on complex questions and identifies optimal model selections for different computational constraints. Our benchmark and evaluation framework are publicly available at this https URL. 

---
# TCM-3CEval: A Triaxial Benchmark for Assessing Responses from Large Language Models in Traditional Chinese Medicine 

**Authors**: Tianai Huang, Lu Lu, Jiayuan Chen, Lihao Liu, Junjun He, Yuping Zhao, Wenchao Tang, Jie Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.07041)  

**Abstract**: Large language models (LLMs) excel in various NLP tasks and modern medicine, but their evaluation in traditional Chinese medicine (TCM) is underexplored. To address this, we introduce TCM3CEval, a benchmark assessing LLMs in TCM across three dimensions: core knowledge mastery, classical text understanding, and clinical decision-making. We evaluate diverse models, including international (e.g., GPT-4o), Chinese (e.g., InternLM), and medical-specific (e.g., PLUSE). Results show a performance hierarchy: all models have limitations in specialized subdomains like Meridian & Acupoint theory and Various TCM Schools, revealing gaps between current capabilities and clinical needs. Models with Chinese linguistic and cultural priors perform better in classical text interpretation and clinical reasoning. TCM-3CEval sets a standard for AI evaluation in TCM, offering insights for optimizing LLMs in culturally grounded medical domains. The benchmark is available on Medbench's TCM track, aiming to assess LLMs' TCM capabilities in basic knowledge, classic texts, and clinical decision-making through multidimensional questions and real cases. 

---
# Bot Wars Evolved: Orchestrating Competing LLMs in a Counterstrike Against Phone Scams 

**Authors**: Nardine Basta, Conor Atkins, Dali Kaafar  

**Link**: [PDF](https://arxiv.org/pdf/2503.07036)  

**Abstract**: We present "Bot Wars," a framework using Large Language Models (LLMs) scam-baiters to counter phone scams through simulated adversarial dialogues. Our key contribution is a formal foundation for strategy emergence through chain-of-thought reasoning without explicit optimization. Through a novel two-layer prompt architecture, our framework enables LLMs to craft demographically authentic victim personas while maintaining strategic coherence. We evaluate our approach using a dataset of 3,200 scam dialogues validated against 179 hours of human scam-baiting interactions, demonstrating its effectiveness in capturing complex adversarial dynamics. Our systematic evaluation through cognitive, quantitative, and content-specific metrics shows that GPT-4 excels in dialogue naturalness and persona authenticity, while Deepseek demonstrates superior engagement sustainability. 

---
# Large Language Models Often Say One Thing and Do Another 

**Authors**: Ruoxi Xu, Hongyu Lin, Xianpei Han, Jia Zheng, Weixiang Zhou, Le Sun, Yingfei Sun  

**Link**: [PDF](https://arxiv.org/pdf/2503.07003)  

**Abstract**: As large language models (LLMs) increasingly become central to various applications and interact with diverse user populations, ensuring their reliable and consistent performance is becoming more important. This paper explores a critical issue in assessing the reliability of LLMs: the consistency between their words and deeds. To quantitatively explore this consistency, we developed a novel evaluation benchmark called the Words and Deeds Consistency Test (WDCT). The benchmark establishes a strict correspondence between word-based and deed-based questions across different domains, including opinion vs. action, non-ethical value vs. action, ethical value vs. action, and theory vs. application. The evaluation results reveal a widespread inconsistency between words and deeds across different LLMs and domains. Subsequently, we conducted experiments with either word alignment or deed alignment to observe their impact on the other aspect. The experimental results indicate that alignment only on words or deeds poorly and unpredictably influences the other aspect. This supports our hypothesis that the underlying knowledge guiding LLMs' word or deed choices is not contained within a unified space. 

---
# Toward Multi-Session Personalized Conversation: A Large-Scale Dataset and Hierarchical Tree Framework for Implicit Reasoning 

**Authors**: Xintong Li, Jalend Bantupalli, Ria Dharmani, Yuwei Zhang, Jingbo Shang  

**Link**: [PDF](https://arxiv.org/pdf/2503.07018)  

**Abstract**: There has been a surge in the use of large language models (LLM) conversational agents to generate responses based on long-term history from multiple sessions. However, existing long-term open-domain dialogue datasets lack complex, real-world personalization and fail to capture implicit reasoning-where relevant information is embedded in subtle, syntactic, or semantically distant connections rather than explicit statements. In such cases, traditional retrieval methods fail to capture relevant context, and long-context modeling also becomes inefficient due to numerous complicated persona-related details. To address this gap, we introduce ImplexConv, a large-scale long-term dataset with 2,500 examples, each containing approximately 100 conversation sessions, designed to study implicit reasoning in personalized dialogues. Additionally, we propose TaciTree, a novel hierarchical tree framework that structures conversation history into multiple levels of summarization. Instead of brute-force searching all data, TaciTree enables an efficient, level-based retrieval process where models refine their search by progressively selecting relevant details. Our experiments demonstrate that TaciTree significantly improves the ability of LLMs to reason over long-term conversations with implicit contextual dependencies. 

---
# Social Bias Benchmark for Generation: A Comparison of Generation and QA-Based Evaluations 

**Authors**: Jiho Jin, Woosung Kang, Junho Myung, Alice Oh  

**Link**: [PDF](https://arxiv.org/pdf/2503.06987)  

**Abstract**: Measuring social bias in large language models (LLMs) is crucial, but existing bias evaluation methods struggle to assess bias in long-form generation. We propose a Bias Benchmark for Generation (BBG), an adaptation of the Bias Benchmark for QA (BBQ), designed to evaluate social bias in long-form generation by having LLMs generate continuations of story prompts. Building our benchmark in English and Korean, we measure the probability of neutral and biased generations across ten LLMs. We also compare our long-form story generation evaluation results with multiple-choice BBQ evaluation, showing that the two approaches produce inconsistent results. 

---
# CtrlRAG: Black-box Adversarial Attacks Based on Masked Language Models in Retrieval-Augmented Language Generation 

**Authors**: Runqi Sui  

**Link**: [PDF](https://arxiv.org/pdf/2503.06950)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems enhance Large Language Models (LLMs) by integrating external knowledge bases. However, this integration introduces a new security threat: adversaries can exploit the retrieval mechanism to inject malicious content into the knowledge base, thereby influencing the generated responses. Based on this attack vector, we propose CtrlRAG, a novel attack method designed for RAG system in the black-box setting, which aligns with real-world scenarios. Unlike existing attack methods, CtrlRAG introduces a perturbation mechanism using Masked Language Model (MLM) to dynamically optimize malicious content in response to changes in the retrieved context. Experimental results demonstrate that CtrlRAG outperforms three baseline methods in both Emotional Manipulation and Hallucination Amplification objectives. Furthermore, we evaluate three existing defense mechanisms, revealing their limited effectiveness against CtrlRAG and underscoring the urgent need for more robust defenses. 

---
# Multimodal Human-AI Synergy for Medical Imaging Quality Control: A Hybrid Intelligence Framework with Adaptive Dataset Curation and Closed-Loop Evaluation 

**Authors**: Zhi Qin, Qianhui Gui, Mouxiao Bian, Rui Wang, Hong Ge, Dandan Yao, Ziying Sun, Yuan Zhao, Yu Zhang, Hui Shi, Dongdong Wang, Chenxin Song, Shenghong Ju, Lihao Liu, Junjun He, Jie Xu, Yuan-Cheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.07032)  

**Abstract**: Medical imaging quality control (QC) is essential for accurate diagnosis, yet traditional QC methods remain labor-intensive and subjective. To address this challenge, in this study, we establish a standardized dataset and evaluation framework for medical imaging QC, systematically assessing large language models (LLMs) in image quality assessment and report standardization. Specifically, we first constructed and anonymized a dataset of 161 chest X-ray (CXR) radiographs and 219 CT reports for evaluation. Then, multiple LLMs, including Gemini 2.0-Flash, GPT-4o, and DeepSeek-R1, were evaluated based on recall, precision, and F1 score to detect technical errors and inconsistencies. Experimental results show that Gemini 2.0-Flash achieved a Macro F1 score of 90 in CXR tasks, demonstrating strong generalization but limited fine-grained performance. DeepSeek-R1 excelled in CT report auditing with a 62.23\% recall rate, outperforming other models. However, its distilled variants performed poorly, while InternLM2.5-7B-chat exhibited the highest additional discovery rate, indicating broader but less precise error detection. These findings highlight the potential of LLMs in medical imaging QC, with DeepSeek-R1 and Gemini 2.0-Flash demonstrating superior performance. 

---
# Linguistic Knowledge Transfer Learning for Speech Enhancement 

**Authors**: Kuo-Hsuan Hung, Xugang Lu, Szu-Wei Fu, Huan-Hsin Tseng, Hsin-Yi Lin, Chii-Wann Lin, Yu Tsao  

**Link**: [PDF](https://arxiv.org/pdf/2503.07078)  

**Abstract**: Linguistic knowledge plays a crucial role in spoken language comprehension. It provides essential semantic and syntactic context for speech perception in noisy environments. However, most speech enhancement (SE) methods predominantly rely on acoustic features to learn the mapping relationship between noisy and clean speech, with limited exploration of linguistic integration. While text-informed SE approaches have been investigated, they often require explicit speech-text alignment or externally provided textual data, constraining their practicality in real-world scenarios. Additionally, using text as input poses challenges in aligning linguistic and acoustic representations due to their inherent differences. In this study, we propose the Cross-Modality Knowledge Transfer (CMKT) learning framework, which leverages pre-trained large language models (LLMs) to infuse linguistic knowledge into SE models without requiring text input or LLMs during inference. Furthermore, we introduce a misalignment strategy to improve knowledge transfer. This strategy applies controlled temporal shifts, encouraging the model to learn more robust representations. Experimental evaluations demonstrate that CMKT consistently outperforms baseline models across various SE architectures and LLM embeddings, highlighting its adaptability to different configurations. Additionally, results on Mandarin and English datasets confirm its effectiveness across diverse linguistic conditions, further validating its robustness. Moreover, CMKT remains effective even in scenarios without textual data, underscoring its practicality for real-world applications. By bridging the gap between linguistic and acoustic modalities, CMKT offers a scalable and innovative solution for integrating linguistic knowledge into SE models, leading to substantial improvements in both intelligibility and enhancement performance. 

---
# Dr Genre: Reinforcement Learning from Decoupled LLM Feedback for Generic Text Rewriting 

**Authors**: Yufei Li, John Nham, Ganesh Jawahar, Lei Shu, David Uthus, Yun-Hsuan Sung, Chengrun Yang, Itai Rolnick, Yi Qiao, Cong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.06781)  

**Abstract**: Generic text rewriting is a prevalent large language model (LLM) application that covers diverse real-world tasks, such as style transfer, fact correction, and email editing. These tasks vary in rewriting objectives (e.g., factual consistency vs. semantic preservation), making it challenging to develop a unified model that excels across all dimensions. Existing methods often specialize in either a single task or a specific objective, limiting their generalizability. In this work, we introduce a generic model proficient in factuality, stylistic, and conversational rewriting tasks. To simulate real-world user rewrite requests, we construct a conversational rewrite dataset, ChatRewrite, that presents ``natural''-sounding instructions, from raw emails using LLMs. Combined with other popular rewrite datasets, including LongFact for the factuality rewrite task and RewriteLM for the stylistic rewrite task, this forms a broad benchmark for training and evaluating generic rewrite models. To align with task-specific objectives, we propose Dr Genre, a Decoupled-reward learning framework for Generic rewriting, that utilizes objective-oriented reward models with a task-specific weighting. Evaluation shows that \approach delivers higher-quality rewrites across all targeted tasks, improving objectives including instruction following (agreement), internal consistency (coherence), and minimal unnecessary edits (conciseness). 

---
# Large Language Models Are Effective Human Annotation Assistants, But Not Good Independent Annotators 

**Authors**: Feng Gu, Zongxia Li, Carlos Rafael Colon, Benjamin Evans, Ishani Mondal, Jordan Lee Boyd-Graber  

**Link**: [PDF](https://arxiv.org/pdf/2503.06778)  

**Abstract**: Event annotation is important for identifying market changes, monitoring breaking news, and understanding sociological trends. Although expert annotators set the gold standards, human coding is expensive and inefficient. Unlike information extraction experiments that focus on single contexts, we evaluate a holistic workflow that removes irrelevant documents, merges documents about the same event, and annotates the events. Although LLM-based automated annotations are better than traditional TF-IDF-based methods or Event Set Curation, they are still not reliable annotators compared to human experts. However, adding LLMs to assist experts for Event Set Curation can reduce the time and mental effort required for Variable Annotation. When using LLMs to extract event variables to assist expert annotators, they agree more with the extracted variables than fully automated LLMs for annotation. 

---
# Effectiveness of Zero-shot-CoT in Japanese Prompts 

**Authors**: Shusuke Takayama, Ian Frank  

**Link**: [PDF](https://arxiv.org/pdf/2503.06765)  

**Abstract**: We compare the effectiveness of zero-shot Chain-of-Thought (CoT) prompting in Japanese and English using ChatGPT-3.5 and 4o-mini. The technique of zero-shot CoT, which involves appending a phrase such as "Let's think step by step" to a prompt to encourage reasoning before answering, has been shown to offer LLM performance improvements in mathematical and reasoning tasks, particularly in English. We investigate how these effects transfer to Japanese using the Japanese Multi-task Language Understanding Benchmark (JMMLU) and the Multi-task Language Understanding Benchmark (MMLU). Our results show that while zero-shot CoT prompting can lead to notable performance gains for some prompt categories in GPT-3.5, its impact in GPT-4o-mini is associated with significant performance declines. However, for Japanese prompts there remain certain categories, such as college mathematics and abstract algebra, that still exhibit improvements, despite the broader trend of diminishing effectiveness in more advanced models. 

---
# Delusions of Large Language Models 

**Authors**: Hongshen Xu, Zixv yang, Zichen Zhu, Kunyao Lan, Zihan Wang, Mengyue Wu, Ziwei Ji, Lu Chen, Pascale Fung, Kai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2503.06709)  

**Abstract**: Large Language Models often generate factually incorrect but plausible outputs, known as hallucinations. We identify a more insidious phenomenon, LLM delusion, defined as high belief hallucinations, incorrect outputs with abnormally high confidence, making them harder to detect and mitigate. Unlike ordinary hallucinations, delusions persist with low uncertainty, posing significant challenges to model reliability. Through empirical analysis across different model families and sizes on several Question Answering tasks, we show that delusions are prevalent and distinct from hallucinations. LLMs exhibit lower honesty with delusions, which are harder to override via finetuning or self reflection. We link delusion formation with training dynamics and dataset noise and explore mitigation strategies such as retrieval augmented generation and multi agent debating to mitigate delusions. By systematically investigating the nature, prevalence, and mitigation of LLM delusions, our study provides insights into the underlying causes of this phenomenon and outlines future directions for improving model reliability. 

---
# A Novel Ophthalmic Benchmark for Evaluating Multimodal Large Language Models with Fundus Photographs and OCT Images 

**Authors**: Xiaoyi Liang, Mouxiao Bian, Moxin Chen, Lihao Liu, Junjun He, Jie Xu, Lin Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.07094)  

**Abstract**: In recent years, large language models (LLMs) have demonstrated remarkable potential across various medical applications. Building on this foundation, multimodal large language models (MLLMs) integrate LLMs with visual models to process diverse inputs, including clinical data and medical images. In ophthalmology, LLMs have been explored for analyzing optical coherence tomography (OCT) reports, assisting in disease classification, and even predicting treatment outcomes. However, existing MLLM benchmarks often fail to capture the complexities of real-world clinical practice, particularly in the analysis of OCT images. Many suffer from limitations such as small sample sizes, a lack of diverse OCT datasets, and insufficient expert validation. These shortcomings hinder the accurate assessment of MLLMs' ability to interpret OCT scans and their broader applicability in ophthalmology. Our dataset, curated through rigorous quality control and expert annotation, consists of 439 fundus images and 75 OCT images. Using a standardized API-based framework, we assessed seven mainstream MLLMs and observed significant variability in diagnostic accuracy across different diseases. While some models performed well in diagnosing conditions such as diabetic retinopathy and age-related macular degeneration, they struggled with others, including choroidal neovascularization and myopia, highlighting inconsistencies in performance and the need for further refinement. Our findings emphasize the importance of developing clinically relevant benchmarks to provide a more accurate assessment of MLLMs' capabilities. By refining these models and expanding their scope, we can enhance their potential to transform ophthalmic diagnosis and treatment. 

---
# Alignment for Efficient Tool Calling of Large Language Models 

**Authors**: Hongshen Xu, Zihan Wang, Zichen Zhu, Lei Pan, Xingyu Chen, Lu Chen, Kai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2503.06708)  

**Abstract**: Recent advancements in tool learning have enabled large language models (LLMs) to integrate external tools, enhancing their task performance by expanding their knowledge boundaries. However, relying on tools often introduces tradeoffs between performance, speed, and cost, with LLMs sometimes exhibiting overreliance and overconfidence in tool usage. This paper addresses the challenge of aligning LLMs with their knowledge boundaries to make more intelligent decisions about tool invocation. We propose a multi objective alignment framework that combines probabilistic knowledge boundary estimation with dynamic decision making, allowing LLMs to better assess when to invoke tools based on their confidence. Our framework includes two methods for knowledge boundary estimation, consistency based and absolute estimation, and two training strategies for integrating these estimates into the model decision making process. Experimental results on various tool invocation scenarios demonstrate the effectiveness of our framework, showing significant improvements in tool efficiency by reducing unnecessary tool usage. 

---
# Enhancing NLP Robustness and Generalization through LLM-Generated Contrast Sets: A Scalable Framework for Systematic Evaluation and Adversarial Training 

**Authors**: Hender Lin  

**Link**: [PDF](https://arxiv.org/pdf/2503.06648)  

**Abstract**: Standard NLP benchmarks often fail to capture vulnerabilities stemming from dataset artifacts and spurious correlations. Contrast sets address this gap by challenging models near decision boundaries but are traditionally labor-intensive to create and limited in diversity. This study leverages large language models to automate the generation of diverse contrast sets. Using the SNLI dataset, we created a 3,000-example contrast set to evaluate and improve model robustness. Fine-tuning on these contrast sets enhanced performance on systematically perturbed examples, maintained standard test accuracy, and modestly improved generalization to novel perturbations. This automated approach offers a scalable solution for evaluating and improving NLP models, addressing systematic generalization challenges, and advancing robustness in real-world applications. 

---
# Beyond Decoder-only: Large Language Models Can be Good Encoders for Machine Translation 

**Authors**: Yingfeng Luo, Tong Zheng, Yongyu Mu, Bei Li, Qinghong Zhang, Yongqi Gao, Ziqiang Xu, Peinan Feng, Xiaoqian Liu, Tong Xiao, Jingbo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2503.06594)  

**Abstract**: The field of neural machine translation (NMT) has changed with the advent of large language models (LLMs). Much of the recent emphasis in natural language processing (NLP) has been on modeling machine translation and many other problems using a single pre-trained Transformer decoder, while encoder-decoder architectures, which were the standard in earlier NMT models, have received relatively less attention. In this paper, we explore translation models that are universal, efficient, and easy to optimize, by marrying the world of LLMs with the world of NMT. We apply LLMs to NMT encoding and leave the NMT decoder unchanged. We also develop methods for adapting LLMs to work better with the NMT decoder. Furthermore, we construct a new dataset involving multiple tasks to assess how well the machine translation system generalizes across various tasks. Evaluations on the WMT and our datasets show that results using our method match or surpass a range of baselines in terms of translation quality, but achieve $2.4 \sim 6.5 \times$ inference speedups and a $75\%$ reduction in the memory footprint of the KV cache. It also demonstrates strong generalization across a variety of translation-related tasks. 

---
# WildIFEval: Instruction Following in the Wild 

**Authors**: Gili Lior, Asaf Yehudai, Ariel Gera, Liat Ein-Dor  

**Link**: [PDF](https://arxiv.org/pdf/2503.06573)  

**Abstract**: Recent LLMs have shown remarkable success in following user instructions, yet handling instructions with multiple constraints remains a significant challenge. In this work, we introduce WildIFEval - a large-scale dataset of 12K real user instructions with diverse, multi-constraint conditions. Unlike prior datasets, our collection spans a broad lexical and topical spectrum of constraints, in natural user prompts. We categorize these constraints into eight high-level classes to capture their distribution and dynamics in real-world scenarios. Leveraging WildIFEval, we conduct extensive experiments to benchmark the instruction-following capabilities of leading LLMs. Our findings reveal that all evaluated models experience performance degradation with an increasing number of constraints. Thus, we show that all models have a large room for improvement on such tasks. Moreover, we observe that the specific type of constraint plays a critical role in model performance. We release our dataset to promote further research on instruction-following under complex, realistic conditions. 

---
# InftyThink: Breaking the Length Limits of Long-Context Reasoning in Large Language Models 

**Authors**: Yuchen Yan, Yongliang Shen, Yang Liu, Jin Jiang, Mengdi Zhang, Jian Shao, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2503.06692)  

**Abstract**: Advanced reasoning in large language models has achieved remarkable performance on challenging tasks, but the prevailing long-context reasoning paradigm faces critical limitations: quadratic computational scaling with sequence length, reasoning constrained by maximum context boundaries, and performance degradation beyond pre-training context windows. Existing approaches primarily compress reasoning chains without addressing the fundamental scaling problem. To overcome these challenges, we introduce InftyThink, a paradigm that transforms monolithic reasoning into an iterative process with intermediate summarization. By interleaving short reasoning segments with concise progress summaries, our approach enables unbounded reasoning depth while maintaining bounded computational costs. This creates a characteristic sawtooth memory pattern that significantly reduces computational complexity compared to traditional approaches. Furthermore, we develop a methodology for reconstructing long-context reasoning datasets into our iterative format, transforming OpenR1-Math into 333K training instances. Experiments across multiple model architectures demonstrate that our approach reduces computational costs while improving performance, with Qwen2.5-Math-7B showing 3-13% improvements across MATH500, AIME24, and GPQA_diamond benchmarks. Our work challenges the assumed trade-off between reasoning depth and computational efficiency, providing a more scalable approach to complex reasoning without architectural modifications. 

---
# Training LLM-based Tutors to Improve Student Learning Outcomes in Dialogues 

**Authors**: Alexander Scarlatos, Naiming Liu, Jaewook Lee, Richard Baraniuk, Andrew Lan  

**Link**: [PDF](https://arxiv.org/pdf/2503.06424)  

**Abstract**: Generative artificial intelligence (AI) has the potential to scale up personalized tutoring through large language models (LLMs). Recent AI tutors are adapted for the tutoring task by training or prompting LLMs to follow effective pedagogical principles, though they are not trained to maximize student learning throughout the course of a dialogue. Therefore, they may engage with students in a suboptimal way. We address this limitation by introducing an approach to train LLMs to generate tutor utterances that maximize the likelihood of student correctness, while still encouraging the model to follow good pedagogical practice. Specifically, we generate a set of candidate tutor utterances and score them using (1) an LLM-based student model to predict the chance of correct student responses and (2) a pedagogical rubric evaluated by GPT-4o. We then use the resulting data to train an open-source LLM, Llama 3.1 8B, using direct preference optimization. We show that tutor utterances generated by our model lead to significantly higher chances of correct student responses while maintaining the pedagogical quality of GPT-4o. We also conduct qualitative analyses and a human evaluation to demonstrate that our model generates high quality tutor utterances. 

---
# SafeSpeech: A Comprehensive and Interactive Tool for Analysing Sexist and Abusive Language in Conversations 

**Authors**: Xingwei Tan, Chen Lyu, Hafiz Muhammad Umer, Sahrish Khan, Mahathi Parvatham, Lois Arthurs, Simon Cullen, Shelley Wilson, Arshad Jhumka, Gabriele Pergola  

**Link**: [PDF](https://arxiv.org/pdf/2503.06534)  

**Abstract**: Detecting toxic language including sexism, harassment and abusive behaviour, remains a critical challenge, particularly in its subtle and context-dependent forms. Existing approaches largely focus on isolated message-level classification, overlooking toxicity that emerges across conversational contexts. To promote and enable future research in this direction, we introduce SafeSpeech, a comprehensive platform for toxic content detection and analysis that bridges message-level and conversation-level insights. The platform integrates fine-tuned classifiers and large language models (LLMs) to enable multi-granularity detection, toxic-aware conversation summarization, and persona profiling. SafeSpeech also incorporates explainability mechanisms, such as perplexity gain analysis, to highlight the linguistic elements driving predictions. Evaluations on benchmark datasets, including EDOS, OffensEval, and HatEval, demonstrate the reproduction of state-of-the-art performance across multiple tasks, including fine-grained sexism detection. 

---
# Exploring Multimodal Perception in Large Language Models Through Perceptual Strength Ratings 

**Authors**: Jonghyun Lee, Dojun Park, Jiwoo Lee, Hoekeon Choi, Sung-Eun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.06980)  

**Abstract**: This study investigated the multimodal perception of large language models (LLMs), focusing on their ability to capture human-like perceptual strength ratings across sensory modalities. Utilizing perceptual strength ratings as a benchmark, the research compared GPT-3.5, GPT-4, GPT-4o, and GPT-4o-mini, highlighting the influence of multimodal inputs on grounding and linguistic reasoning. While GPT-4 and GPT-4o demonstrated strong alignment with human evaluations and significant advancements over smaller models, qualitative analyses revealed distinct differences in processing patterns, such as multisensory overrating and reliance on loose semantic associations. Despite integrating multimodal capabilities, GPT-4o did not exhibit superior grounding compared to GPT-4, raising questions about their role in improving human-like grounding. These findings underscore how LLMs' reliance on linguistic patterns can both approximate and diverge from human embodied cognition, revealing limitations in replicating sensory experiences. 

---
# BingoGuard: LLM Content Moderation Tools with Risk Levels 

**Authors**: Fan Yin, Philippe Laban, Xiangyu Peng, Yilun Zhou, Yixin Mao, Vaibhav Vats, Linnea Ross, Divyansh Agarwal, Caiming Xiong, Chien-Sheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.06550)  

**Abstract**: Malicious content generated by large language models (LLMs) can pose varying degrees of harm. Although existing LLM-based moderators can detect harmful content, they struggle to assess risk levels and may miss lower-risk outputs. Accurate risk assessment allows platforms with different safety thresholds to tailor content filtering and rejection. In this paper, we introduce per-topic severity rubrics for 11 harmful topics and build BingoGuard, an LLM-based moderation system designed to predict both binary safety labels and severity levels. To address the lack of annotations on levels of severity, we propose a scalable generate-then-filter framework that first generates responses across different severity levels and then filters out low-quality responses. Using this framework, we create BingoGuardTrain, a training dataset with 54,897 examples covering a variety of topics, response severity, styles, and BingoGuardTest, a test set with 988 examples explicitly labeled based on our severity rubrics that enables fine-grained analysis on model behaviors on different severity levels. Our BingoGuard-8B, trained on BingoGuardTrain, achieves the state-of-the-art performance on several moderation benchmarks, including WildGuardTest and HarmBench, as well as BingoGuardTest, outperforming best public models, WildGuard, by 4.3\%. Our analysis demonstrates that incorporating severity levels into training significantly enhances detection performance and enables the model to effectively gauge the severity of harmful responses. 

---
# IteRABRe: Iterative Recovery-Aided Block Reduction 

**Authors**: Haryo Akbarianto Wibowo, Haiyue Song, Hideki Tanaka, Masao Utiyama, Alham Fikri Aji, Raj Dabre  

**Link**: [PDF](https://arxiv.org/pdf/2503.06291)  

**Abstract**: Large Language Models (LLMs) have grown increasingly expensive to deploy, driving the need for effective model compression techniques. While block pruning offers a straightforward approach to reducing model size, existing methods often struggle to maintain performance or require substantial computational resources for recovery. We present IteRABRe, a simple yet effective iterative pruning method that achieves superior compression results while requiring minimal computational resources. Using only 2.5M tokens for recovery, our method outperforms baseline approaches by ~3% on average when compressing the Llama3.1-8B and Qwen2.5-7B models. IteRABRe demonstrates particular strength in the preservation of linguistic capabilities, showing an improvement 5% over the baselines in language-related tasks. Our analysis reveals distinct pruning characteristics between these models, while also demonstrating preservation of multilingual capabilities. 

---
# Integrating Chain-of-Thought for Multimodal Alignment: A Study on 3D Vision-Language Learning 

**Authors**: Yanjun Chen, Yirong Sun, Xinghao Chen, Jian Wang, Xiaoyu Shen, Wenjie Li, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.06232)  

**Abstract**: Chain-of-Thought (CoT) reasoning has proven effective in natural language tasks but remains underexplored in multimodal alignment. This study investigates its integration into 3D vision-language learning by embedding structured reasoning into alignment training. We introduce the 3D-CoT Benchmark, a dataset with hierarchical CoT annotations covering shape recognition, functional inference, and causal reasoning. Through controlled experiments, we compare CoT-structured and standard textual annotations across large reasoning models (LRMs) and large language models (LLMs). Our evaluation employs a dual-layer framework assessing both intermediate reasoning and final inference quality. Extensive experiments demonstrate that CoT significantly improves 3D semantic grounding, with LRMs leveraging CoT more effectively than LLMs. Furthermore, we highlight that annotation structure influences performance-explicit reasoning markers aid LLMs, while unmarked CoT better aligns with LRM inference patterns. Our analyses suggest that CoT is crucial for enhancing multimodal reasoning, with implications beyond 3D tasks. 

---
# Graph Retrieval-Augmented LLM for Conversational Recommendation Systems 

**Authors**: Zhangchi Qiu, Linhao Luo, Zicheng Zhao, Shirui Pan, Alan Wee-Chung Liew  

**Link**: [PDF](https://arxiv.org/pdf/2503.06430)  

**Abstract**: Conversational Recommender Systems (CRSs) have emerged as a transformative paradigm for offering personalized recommendations through natural language dialogue. However, they face challenges with knowledge sparsity, as users often provide brief, incomplete preference statements. While recent methods have integrated external knowledge sources to mitigate this, they still struggle with semantic understanding and complex preference reasoning. Recent Large Language Models (LLMs) demonstrate promising capabilities in natural language understanding and reasoning, showing significant potential for CRSs. Nevertheless, due to the lack of domain knowledge, existing LLM-based CRSs either produce hallucinated recommendations or demand expensive domain-specific training, which largely limits their applicability. In this work, we present G-CRS (Graph Retrieval-Augmented Large Language Model for Conversational Recommender Systems), a novel training-free framework that combines graph retrieval-augmented generation and in-context learning to enhance LLMs' recommendation capabilities. Specifically, G-CRS employs a two-stage retrieve-and-recommend architecture, where a GNN-based graph reasoner first identifies candidate items, followed by Personalized PageRank exploration to jointly discover potential items and similar user interactions. These retrieved contexts are then transformed into structured prompts for LLM reasoning, enabling contextually grounded recommendations without task-specific training. Extensive experiments on two public datasets show that G-CRS achieves superior recommendation performance compared to existing methods without requiring task-specific training. 

---
# CUPCase: Clinically Uncommon Patient Cases and Diagnoses Dataset 

**Authors**: Oriel Perets, Ofir Ben Shoham, Nir Grinberg, Nadav Rappoport  

**Link**: [PDF](https://arxiv.org/pdf/2503.06204)  

**Abstract**: Medical benchmark datasets significantly contribute to developing Large Language Models (LLMs) for medical knowledge extraction, diagnosis, summarization, and other uses. Yet, current benchmarks are mainly derived from exam questions given to medical students or cases described in the medical literature, lacking the complexity of real-world patient cases that deviate from classic textbook abstractions. These include rare diseases, uncommon presentations of common diseases, and unexpected treatment responses. Here, we construct Clinically Uncommon Patient Cases and Diagnosis Dataset (CUPCase) based on 3,562 real-world case reports from BMC, including diagnoses in open-ended textual format and as multiple-choice options with distractors. Using this dataset, we evaluate the ability of state-of-the-art LLMs, including both general-purpose and Clinical LLMs, to identify and correctly diagnose a patient case, and test models' performance when only partial information about cases is available. Our findings show that general-purpose GPT-4o attains the best performance in both the multiple-choice task (average accuracy of 87.9%) and the open-ended task (BERTScore F1 of 0.764), outperforming several LLMs with a focus on the medical domain such as Meditron-70B and MedLM-Large. Moreover, GPT-4o was able to maintain 87% and 88% of its performance with only the first 20% of tokens of the case presentation in multiple-choice and free text, respectively, highlighting the potential of LLMs to aid in early diagnosis in real-world cases. CUPCase expands our ability to evaluate LLMs for clinical decision support in an open and reproducible manner. 

---
# GRP: Goal-Reversed Prompting for Zero-Shot Evaluation with LLMs 

**Authors**: Mingyang Song, Mao Zheng, Xuan Luo  

**Link**: [PDF](https://arxiv.org/pdf/2503.06139)  

**Abstract**: Using Large Language Models (LLMs) to evaluate and compare two answers from different models typically involves having LLM-based judges select the better answer. However, humans often approach problem-solving from a reverse perspective, for instance, by choosing the worse option instead of the better one in a pairwise comparison. Generally, this kind of reverse thinking plays a crucial role in human reasoning and decision-making and can further test the difference between original and reverse thought processes simultaneously. To address the above issue, in this paper, we propose a Goal-Reversed Prompting (GRP) approach for pairwise evaluation that shifts the original task from selecting the better answer to choosing the worse one. We encourage LLMs to think in reverse by prompting LLMs to identify the worse response. Experiments on closed-source models demonstrate that GRP significantly enhances evaluation capabilities, outperforming the prompt template with the original goal. 

---
# How LLMs Learn: Tracing Internal Representations with Sparse Autoencoders 

**Authors**: Tatsuro Inaba, Kentaro Inui, Yusuke Miyao, Yohei Oseki, Benjamin Heinzerling, Yu Takagi  

**Link**: [PDF](https://arxiv.org/pdf/2503.06394)  

**Abstract**: Large Language Models (LLMs) demonstrate remarkable multilingual capabilities and broad knowledge. However, the internal mechanisms underlying the development of these capabilities remain poorly understood. To investigate this, we analyze how the information encoded in LLMs' internal representations evolves during the training process. Specifically, we train sparse autoencoders at multiple checkpoints of the model and systematically compare the interpretative results across these stages. Our findings suggest that LLMs initially acquire language-specific knowledge independently, followed by cross-linguistic correspondences. Moreover, we observe that after mastering token-level knowledge, the model transitions to learning higher-level, abstract concepts, indicating the development of more conceptual understanding. 

---
# A Survey on Post-training of Large Language Models 

**Authors**: Guiyao Tie, Zeli Zhao, Dingjie Song, Fuyang Wei, Rong Zhou, Yurou Dai, Wen Yin, Zhejian Yang, Jiangyue Yan, Yao Su, Zhenhan Dai, Yifeng Xie, Yihan Cao, Lichao Sun, Pan Zhou, Lifang He, Hechang Chen, Yu Zhang, Qingsong Wen, Tianming Liu, Neil Zhenqiang Gong, Jiliang Tang, Caiming Xiong, Heng Ji, Philip S. Yu, Jianfeng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2503.06072)  

**Abstract**: The emergence of Large Language Models (LLMs) has fundamentally transformed natural language processing, making them indispensable across domains ranging from conversational systems to scientific exploration. However, their pre-trained architectures often reveal limitations in specialized contexts, including restricted reasoning capacities, ethical uncertainties, and suboptimal domain-specific performance. These challenges necessitate advanced post-training language models (PoLMs) to address these shortcomings, such as OpenAI-o1/o3 and DeepSeek-R1 (collectively known as Large Reasoning Models, or LRMs). This paper presents the first comprehensive survey of PoLMs, systematically tracing their evolution across five core paradigms: Fine-tuning, which enhances task-specific accuracy; Alignment, which ensures alignment with human preferences; Reasoning, which advances multi-step inference despite challenges in reward design; Efficiency, which optimizes resource utilization amidst increasing complexity; and Integration and Adaptation, which extend capabilities across diverse modalities while addressing coherence issues. Charting progress from ChatGPT's foundational alignment strategies to DeepSeek-R1's innovative reasoning advancements, we illustrate how PoLMs leverage datasets to mitigate biases, deepen reasoning capabilities, and enhance domain adaptability. Our contributions include a pioneering synthesis of PoLM evolution, a structured taxonomy categorizing techniques and datasets, and a strategic agenda emphasizing the role of LRMs in improving reasoning proficiency and domain flexibility. As the first survey of its scope, this work consolidates recent PoLM advancements and establishes a rigorous intellectual framework for future research, fostering the development of LLMs that excel in precision, ethical robustness, and versatility across scientific and societal applications. 

---
# GEM: Empowering MLLM for Grounded ECG Understanding with Time Series and Images 

**Authors**: Xiang Lan, Feng Wu, Kai He, Qinghao Zhao, Shenda Hong, Mengling Feng  

**Link**: [PDF](https://arxiv.org/pdf/2503.06073)  

**Abstract**: While recent multimodal large language models (MLLMs) have advanced automated ECG interpretation, they still face two key limitations: (1) insufficient multimodal synergy between time series signals and visual ECG representations, and (2) limited explainability in linking diagnoses to granular waveform evidence. We introduce GEM, the first MLLM unifying ECG time series, 12-lead ECG images and text for grounded and clinician-aligned ECG interpretation. GEM enables feature-grounded analysis, evidence-driven reasoning, and a clinician-like diagnostic process through three core innovations: a dual-encoder framework extracting complementary time series and image features, cross-modal alignment for effective multimodal understanding, and knowledge-guided instruction generation for generating high-granularity grounding data (ECG-Grounding) linking diagnoses to measurable parameters ($e.g.$, QRS/PR Intervals). Additionally, we propose the Grounded ECG Understanding task, a clinically motivated benchmark designed to comprehensively assess the MLLM's capability in grounded ECG understanding. Experimental results on both existing and our proposed benchmarks show GEM significantly improves predictive performance (CSN $7.4\% \uparrow$), explainability ($22.7\% \uparrow$), and grounding ($24.8\% \uparrow$), making it more suitable for real-world clinical applications. GitHub repository: this https URL 

---
# Fine-Grained Bias Detection in LLM: Enhancing detection mechanisms for nuanced biases 

**Authors**: Suvendu Mohanty  

**Link**: [PDF](https://arxiv.org/pdf/2503.06054)  

**Abstract**: Recent advancements in Artificial Intelligence, particularly in Large Language Models (LLMs), have transformed natural language processing by improving generative capabilities. However, detecting biases embedded within these models remains a challenge. Subtle biases can propagate misinformation, influence decision-making, and reinforce stereotypes, raising ethical concerns. This study presents a detection framework to identify nuanced biases in LLMs. The approach integrates contextual analysis, interpretability via attention mechanisms, and counterfactual data augmentation to capture hidden biases across linguistic contexts. The methodology employs contrastive prompts and synthetic datasets to analyze model behaviour across cultural, ideological, and demographic scenarios.
Quantitative analysis using benchmark datasets and qualitative assessments through expert reviews validate the effectiveness of the framework. Results show improvements in detecting subtle biases compared to conventional methods, which often fail to highlight disparities in model responses to race, gender, and socio-political contexts. The framework also identifies biases arising from imbalances in training data and model architectures. Continuous user feedback ensures adaptability and refinement. This research underscores the importance of proactive bias mitigation strategies and calls for collaboration between policymakers, AI developers, and regulators. The proposed detection mechanisms enhance model transparency and support responsible LLM deployment in sensitive applications such as education, legal systems, and healthcare. Future work will focus on real-time bias monitoring and cross-linguistic generalization to improve fairness and inclusivity in AI-driven communication tools. 

---
# SmartBench: Is Your LLM Truly a Good Chinese Smartphone Assistant? 

**Authors**: Xudong Lu, Haohao Gao, Renshou Wu, Shuai Ren, Xiaoxin Chen, Hongsheng Li, Fangyuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.06029)  

**Abstract**: Large Language Models (LLMs) have become integral to daily life, especially advancing as intelligent assistants through on-device deployment on smartphones. However, existing LLM evaluation benchmarks predominantly focus on objective tasks like mathematics and coding in English, which do not necessarily reflect the practical use cases of on-device LLMs in real-world mobile scenarios, especially for Chinese users. To address these gaps, we introduce SmartBench, the first benchmark designed to evaluate the capabilities of on-device LLMs in Chinese mobile contexts. We analyze functionalities provided by representative smartphone manufacturers and divide them into five categories: text summarization, text Q\&A, information extraction, content creation, and notification management, further detailed into 20 specific tasks. For each task, we construct high-quality datasets comprising 50 to 200 question-answer pairs that reflect everyday mobile interactions, and we develop automated evaluation criteria tailored for these tasks. We conduct comprehensive evaluations of on-device LLMs and MLLMs using SmartBench and also assess their performance after quantized deployment on real smartphone NPUs. Our contributions provide a standardized framework for evaluating on-device LLMs in Chinese, promoting further development and optimization in this critical area. Code and data will be available at this https URL. 

---
# Intent-Aware Self-Correction for Mitigating Social Biases in Large Language Models 

**Authors**: Panatchakorn Anantaprayoon, Masahiro Kaneko, Naoaki Okazaki  

**Link**: [PDF](https://arxiv.org/pdf/2503.06011)  

**Abstract**: Self-Correction based on feedback improves the output quality of Large Language Models (LLMs). Moreover, as Self-Correction functions like the slow and conscious System-2 thinking from cognitive psychology's perspective, it can potentially reduce LLMs' social biases. LLMs are sensitive to contextual ambiguities and inconsistencies; therefore, explicitly communicating their intentions during interactions when applying Self-Correction for debiasing is crucial. In this study, we demonstrate that clarifying intentions is essential for effectively reducing biases in LLMs through Self-Correction. We divide the components needed for Self-Correction into three parts: instruction, response, and feedback, and clarify intentions at each component. We incorporate an explicit debiasing prompt to convey the intention of bias mitigation from the instruction for response generation. In the response, we use Chain-of-Thought (CoT) to clarify the reasoning process. In the feedback, we define evaluation aspects necessary for debiasing and propose clear feedback through multi-aspect critiques and scoring. Through experiments, we demonstrate that self-correcting CoT responses obtained from a debiasing prompt based on multi-aspect feedback can reduce biased responses more robustly and consistently than the baselines. We also find the variation in debiasing efficacy when using models with different bias levels or separating models for response and feedback generation. 

---
# Mitigating Memorization in LLMs using Activation Steering 

**Authors**: Manan Suri, Nishit Anand, Amisha Bhaskar  

**Link**: [PDF](https://arxiv.org/pdf/2503.06040)  

**Abstract**: The memorization of training data by Large Language Models (LLMs) poses significant risks, including privacy leaks and the regurgitation of copyrighted content. Activation steering, a technique that directly intervenes in model activations, has emerged as a promising approach for manipulating LLMs. In this work, we explore the effectiveness of activation steering in reducing memorization while preserving generalization capabilities. We conduct empirical evaluations using a controlled memorization benchmark of literary material and demonstrate that our method successfully suppresses memorized content with minimal degradation in model performance in Gemma. Additionally, we analyze the trade-offs between suppression effectiveness and linguistic fluency, highlighting the advantages and limitations of activation-based interventions. Our findings contribute to ongoing efforts in developing safer and more privacy-preserving LLMs by providing a practical and efficient mechanism to mitigate unintended memorization. 

---
# SINdex: Semantic INconsistency Index for Hallucination Detection in LLMs 

**Authors**: Samir Abdaljalil, Hasan Kurban, Parichit Sharma, Erchin Serpedin, Rachad Atat  

**Link**: [PDF](https://arxiv.org/pdf/2503.05980)  

**Abstract**: Large language models (LLMs) are increasingly deployed across diverse domains, yet they are prone to generating factually incorrect outputs - commonly known as "hallucinations." Among existing mitigation strategies, uncertainty-based methods are particularly attractive due to their ease of implementation, independence from external data, and compatibility with standard LLMs. In this work, we introduce a novel and scalable uncertainty-based semantic clustering framework for automated hallucination detection. Our approach leverages sentence embeddings and hierarchical clustering alongside a newly proposed inconsistency measure, SINdex, to yield more homogeneous clusters and more accurate detection of hallucination phenomena across various LLMs. Evaluations on prominent open- and closed-book QA datasets demonstrate that our method achieves AUROC improvements of up to 9.3% over state-of-the-art techniques. Extensive ablation studies further validate the effectiveness of each component in our framework. 

---
# SANDWiCH: Semantical Analysis of Neighbours for Disambiguating Words in Context ad Hoc 

**Authors**: Daniel Guzman-Olivares, Lara Quijano-Sanchez, Federico Liberatore  

**Link**: [PDF](https://arxiv.org/pdf/2503.05958)  

**Abstract**: The rise of generative chat-based Large Language Models (LLMs) over the past two years has spurred a race to develop systems that promise near-human conversational and reasoning experiences. However, recent studies indicate that the language understanding offered by these models remains limited and far from human-like performance, particularly in grasping the contextual meanings of words, an essential aspect of reasoning. In this paper, we present a simple yet computationally efficient framework for multilingual Word Sense Disambiguation (WSD). Our approach reframes the WSD task as a cluster discrimination analysis over a semantic network refined from BabelNet using group algebra. We validate our methodology across multiple WSD benchmarks, achieving a new state of the art for all languages and tasks, as well as in individual assessments by part of speech. Notably, our model significantly surpasses the performance of current alternatives, even in low-resource languages, while reducing the parameter count by 72%. 

---
# QG-SMS: Enhancing Test Item Analysis via Student Modeling and Simulation 

**Authors**: Bang Nguyen, Tingting Du, Mengxia Yu, Lawrence Angrave, Meng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.05888)  

**Abstract**: While the Question Generation (QG) task has been increasingly adopted in educational assessments, its evaluation remains limited by approaches that lack a clear connection to the educational values of test items. In this work, we introduce test item analysis, a method frequently used by educators to assess test question quality, into QG evaluation. Specifically, we construct pairs of candidate questions that differ in quality across dimensions such as topic coverage, item difficulty, item discrimination, and distractor efficiency. We then examine whether existing QG evaluation approaches can effectively distinguish these differences. Our findings reveal significant shortcomings in these approaches with respect to accurately assessing test item quality in relation to student performance. To address this gap, we propose a novel QG evaluation framework, QG-SMS, which leverages Large Language Model for Student Modeling and Simulation to perform test item analysis. As demonstrated in our extensive experiments and human evaluation study, the additional perspectives introduced by the simulated student profiles lead to a more effective and robust assessment of test items. 

---
# Towards Conversational AI for Disease Management 

**Authors**: Anil Palepu, Valentin Liévin, Wei-Hung Weng, Khaled Saab, David Stutz, Yong Cheng, Kavita Kulkarni, S. Sara Mahdavi, Joëlle Barral, Dale R. Webster, Katherine Chou, Avinatan Hassidim, Yossi Matias, James Manyika, Ryutaro Tanno, Vivek Natarajan, Adam Rodman, Tao Tu, Alan Karthikesalingam, Mike Schaekermann  

**Link**: [PDF](https://arxiv.org/pdf/2503.06074)  

**Abstract**: While large language models (LLMs) have shown promise in diagnostic dialogue, their capabilities for effective management reasoning - including disease progression, therapeutic response, and safe medication prescription - remain under-explored. We advance the previously demonstrated diagnostic capabilities of the Articulate Medical Intelligence Explorer (AMIE) through a new LLM-based agentic system optimised for clinical management and dialogue, incorporating reasoning over the evolution of disease and multiple patient visit encounters, response to therapy, and professional competence in medication prescription. To ground its reasoning in authoritative clinical knowledge, AMIE leverages Gemini's long-context capabilities, combining in-context retrieval with structured reasoning to align its output with relevant and up-to-date clinical practice guidelines and drug formularies. In a randomized, blinded virtual Objective Structured Clinical Examination (OSCE) study, AMIE was compared to 21 primary care physicians (PCPs) across 100 multi-visit case scenarios designed to reflect UK NICE Guidance and BMJ Best Practice guidelines. AMIE was non-inferior to PCPs in management reasoning as assessed by specialist physicians and scored better in both preciseness of treatments and investigations, and in its alignment with and grounding of management plans in clinical guidelines. To benchmark medication reasoning, we developed RxQA, a multiple-choice question benchmark derived from two national drug formularies (US, UK) and validated by board-certified pharmacists. While AMIE and PCPs both benefited from the ability to access external drug information, AMIE outperformed PCPs on higher difficulty questions. While further research would be needed before real-world translation, AMIE's strong performance across evaluations marks a significant step towards conversational AI as a tool in disease management. 

---
# Extracting and Emulsifying Cultural Explanation to Improve Multilingual Capability of LLMs 

**Authors**: Hamin Koo, Jaehyung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.05846)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable success, but their English-centric training data limits performance in non-English languages, highlighting the need for enhancements in their multilingual capabilities. While some work on multilingual prompting methods handles non-English queries by utilizing English translations or restructuring them to more closely align with LLM reasoning patterns, these works often overlook the importance of cultural context, limiting their effectiveness. To address this limitation, we propose EMCEI, a simple yet effective approach that improves LLMs' multilingual capabilities by incorporating cultural context for more accurate and appropriate responses. Specifically, EMCEI follows a two-step process that first extracts relevant cultural context from the LLM's parametric knowledge via prompting. Then, EMCEI employs an LLM-as-Judge mechanism to select the most appropriate response by balancing cultural relevance and reasoning ability. Experiments on diverse multilingual benchmarks show that EMCEI outperforms existing baselines, demonstrating its effectiveness in handling multilingual queries with LLMs. 

---
# MastermindEval: A Simple But Scalable Reasoning Benchmark 

**Authors**: Jonas Golde, Patrick Haller, Fabio Barth, Alan Akbik  

**Link**: [PDF](https://arxiv.org/pdf/2503.05891)  

**Abstract**: Recent advancements in large language models (LLMs) have led to remarkable performance across a wide range of language understanding and mathematical tasks. As a result, increasing attention has been given to assessing the true reasoning capabilities of LLMs, driving research into commonsense, numerical, logical, and qualitative reasoning. However, with the rapid progress of reasoning-focused models such as OpenAI's o1 and DeepSeek's R1, there has been a growing demand for reasoning benchmarks that can keep pace with ongoing model developments. In this paper, we introduce MastermindEval, a simple, scalable, and interpretable deductive reasoning benchmark inspired by the board game Mastermind. Our benchmark supports two evaluation paradigms: (1) agentic evaluation, in which the model autonomously plays the game, and (2) deductive reasoning evaluation, in which the model is given a pre-played game state with only one possible valid code to infer. In our experimental results we (1) find that even easy Mastermind instances are difficult for current models and (2) demonstrate that the benchmark is scalable to possibly more advanced models in the future Furthermore, we investigate possible reasons why models cannot deduce the final solution and find that current models are limited in deducing the concealed code as the number of statement to combine information from is increasing. 

---
# This Is Your Doge, If It Please You: Exploring Deception and Robustness in Mixture of LLMs 

**Authors**: Lorenz Wolf, Sangwoong Yoon, Ilija Bogunovic  

**Link**: [PDF](https://arxiv.org/pdf/2503.05856)  

**Abstract**: Mixture of large language model (LLMs) Agents (MoA) architectures achieve state-of-the-art performance on prominent benchmarks like AlpacaEval 2.0 by leveraging the collaboration of multiple LLMs at inference time. Despite these successes, an evaluation of the safety and reliability of MoA is missing. We present the first comprehensive study of MoA's robustness against deceptive LLM agents that deliberately provide misleading responses. We examine factors like the propagation of deceptive information, model size, and information availability, and uncover critical vulnerabilities. On AlpacaEval 2.0, the popular LLaMA 3.1-70B model achieves a length-controlled Win Rate (LC WR) of 49.2% when coupled with 3-layer MoA (6 LLM agents). However, we demonstrate that introducing only a $\textit{single}$ carefully-instructed deceptive agent into the MoA can reduce performance to 37.9%, effectively nullifying all MoA gains. On QuALITY, a multiple-choice comprehension task, the impact is also severe, with accuracy plummeting by a staggering 48.5%. Inspired in part by the historical Doge of Venice voting process, designed to minimize influence and deception, we propose a range of unsupervised defense mechanisms that recover most of the lost performance. 

---
# FedMentalCare: Towards Privacy-Preserving Fine-Tuned LLMs to Analyze Mental Health Status Using Federated Learning Framework 

**Authors**: S M Sarwar  

**Link**: [PDF](https://arxiv.org/pdf/2503.05786)  

**Abstract**: With the increasing prevalence of mental health conditions worldwide, AI-powered chatbots and conversational agents have emerged as accessible tools to support mental health. However, deploying Large Language Models (LLMs) in mental healthcare applications raises significant privacy concerns, especially regarding regulations like HIPAA and GDPR. In this work, we propose FedMentalCare, a privacy-preserving framework that leverages Federated Learning (FL) combined with Low-Rank Adaptation (LoRA) to fine-tune LLMs for mental health analysis. We investigate the performance impact of varying client data volumes and model architectures (e.g., MobileBERT and MiniLM) in FL environments. Our framework demonstrates a scalable, privacy-aware approach for deploying LLMs in real-world mental healthcare scenarios, addressing data security and computational efficiency challenges. 

---
# Optimizing Test-Time Compute via Meta Reinforcement Fine-Tuning 

**Authors**: Yuxiao Qu, Matthew Y. R. Yang, Amrith Setlur, Lewis Tunstall, Edward Emanuel Beeching, Ruslan Salakhutdinov, Aviral Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2503.07572)  

**Abstract**: Training models to effectively use test-time compute is crucial for improving the reasoning performance of LLMs. Current methods mostly do so via fine-tuning on search traces or running RL with 0/1 outcome reward, but do these approaches efficiently utilize test-time compute? Would these approaches continue to scale as the budget improves? In this paper, we try to answer these questions. We formalize the problem of optimizing test-time compute as a meta-reinforcement learning (RL) problem, which provides a principled perspective on spending test-time compute. This perspective enables us to view the long output stream from the LLM as consisting of several episodes run at test time and leads us to use a notion of cumulative regret over output tokens as a way to measure the efficacy of test-time compute. Akin to how RL algorithms can best tradeoff exploration and exploitation over training, minimizing cumulative regret would also provide the best balance between exploration and exploitation in the token stream. While we show that state-of-the-art models do not minimize regret, one can do so by maximizing a dense reward bonus in conjunction with the outcome 0/1 reward RL. This bonus is the ''progress'' made by each subsequent block in the output stream, quantified by the change in the likelihood of eventual success. Using these insights, we develop Meta Reinforcement Fine-Tuning, or MRT, a new class of fine-tuning methods for optimizing test-time compute. MRT leads to a 2-3x relative gain in performance and roughly a 1.5x gain in token efficiency for math reasoning compared to outcome-reward RL. 

---
# DependEval: Benchmarking LLMs for Repository Dependency Understanding 

**Authors**: Junjia Du, Yadi Liu, Hongcheng Guo, Jiawei Wang, Haojian Huang, Yunyi Ni, Zhoujun Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.06689)  

**Abstract**: While large language models (LLMs) have shown considerable promise in code generation, real-world software development demands advanced repository-level reasoning. This includes understanding dependencies, project structures, and managing multi-file changes. However, the ability of LLMs to effectively comprehend and handle complex code repositories has yet to be fully explored. To address challenges, we introduce a hierarchical benchmark designed to evaluate repository dependency understanding (DependEval). Benchmark is based on 15,576 repositories collected from real-world websites. It evaluates models on three core tasks: Dependency Recognition, Repository Construction, and Multi-file Editing, across 8 programming languages from actual code repositories. Our evaluation of over 25 LLMs reveals substantial performance gaps and provides valuable insights into repository-level code understanding. 

---
# Evaluating and Aligning Human Economic Risk Preferences in LLMs 

**Authors**: Jiaxin Liu, Yi Yang, Kar Yan Tam  

**Link**: [PDF](https://arxiv.org/pdf/2503.06646)  

**Abstract**: Large Language Models (LLMs) are increasingly used in decision-making scenarios that involve risk assessment, yet their alignment with human economic rationality remains unclear. In this study, we investigate whether LLMs exhibit risk preferences consistent with human expectations across different personas. Specifically, we assess whether LLM-generated responses reflect appropriate levels of risk aversion or risk-seeking behavior based on individual's persona. Our results reveal that while LLMs make reasonable decisions in simplified, personalized risk contexts, their performance declines in more complex economic decision-making tasks. To address this, we propose an alignment method designed to enhance LLM adherence to persona-specific risk preferences. Our approach improves the economic rationality of LLMs in risk-related applications, offering a step toward more human-aligned AI decision-making. 

---
# HuixiangDou2: A Robustly Optimized GraphRAG Approach 

**Authors**: Huanjun Kong, Zhefan Wang, Chenyang Wang, Zhe Ma, Nanqing Dong  

**Link**: [PDF](https://arxiv.org/pdf/2503.06474)  

**Abstract**: Large Language Models (LLMs) perform well on familiar queries but struggle with specialized or emerging topics. Graph-based Retrieval-Augmented Generation (GraphRAG) addresses this by structuring domain knowledge as a graph for dynamic retrieval. However, existing pipelines involve complex engineering workflows, making it difficult to isolate the impact of individual components. Evaluating retrieval effectiveness is also challenging due to dataset overlap with LLM pretraining data. In this work, we introduce HuixiangDou2, a robustly optimized GraphRAG framework. Specifically, we leverage the effectiveness of dual-level retrieval and optimize its performance in a 32k context for maximum precision, and compare logic-based retrieval and dual-level retrieval to enhance overall functionality. Our implementation includes comparative experiments on a test set, where Qwen2.5-7B-Instruct initially underperformed. With our approach, the score improved significantly from 60 to 74.5, as illustrated in the Figure. Experiments on domain-specific datasets reveal that dual-level retrieval enhances fuzzy matching, while logic-form retrieval improves structured reasoning. Furthermore, we propose a multi-stage verification mechanism to improve retrieval robustness without increasing computational cost. Empirical results show significant accuracy gains over baselines, highlighting the importance of adaptive retrieval. To support research and adoption, we release HuixiangDou2 as an open-source resource this https URL. 

---
# Multimodal Programming in Computer Science with Interactive Assistance Powered by Large Language Model 

**Authors**: Rajan Das Gupta, Md. Tanzib Hosain, M. F. Mridha, Salah Uddin Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2503.06552)  

**Abstract**: LLM chatbot interfaces allow students to get instant, interactive assistance with homework, but doing so carelessly may not advance educational objectives. In this study, an interactive homework help system based on DeepSeek R1 is developed and first implemented for students enrolled in a large computer science beginning programming course. In addition to an assist button in a well-known code editor, our assistant also has a feedback option in our command-line automatic evaluator. It wraps student work in a personalized prompt that advances our educational objectives without offering answers straight away. We have discovered that our assistant can recognize students' conceptual difficulties and provide ideas, plans, and template code in pedagogically appropriate ways. However, among other mistakes, it occasionally incorrectly labels the correct student code as incorrect or encourages students to use correct-but-lesson-inappropriate approaches, which can lead to long and frustrating journeys for the students. After discussing many development and deployment issues, we provide our conclusions and future actions. 

---
# General Scales Unlock AI Evaluation with Explanatory and Predictive Power 

**Authors**: Lexin Zhou, Lorenzo Pacchiardi, Fernando Martínez-Plumed, Katherine M. Collins, Yael Moros-Daval, Seraphina Zhang, Qinlin Zhao, Yitian Huang, Luning Sun, Jonathan E. Prunty, Zongqian Li, Pablo Sánchez-García, Kexin Jiang Chen, Pablo A. M. Casares, Jiyun Zu, John Burden, Behzad Mehrbakhsh, David Stillwell, Manuel Cebrian, Jindong Wang, Peter Henderson, Sherry Tongshuang Wu, Patrick C. Kyllonen, Lucy Cheke, Xing Xie, José Hernández-Orallo  

**Link**: [PDF](https://arxiv.org/pdf/2503.06378)  

**Abstract**: Ensuring safe and effective use of AI requires understanding and anticipating its performance on novel tasks, from advanced scientific challenges to transformed workplace activities. So far, benchmarking has guided progress in AI, but it has offered limited explanatory and predictive power for general-purpose AI systems, given the low transferability across diverse tasks. In this paper, we introduce general scales for AI evaluation that can explain what common AI benchmarks really measure, extract ability profiles of AI systems, and predict their performance for new task instances, in- and out-of-distribution. Our fully-automated methodology builds on 18 newly-crafted rubrics that place instance demands on general scales that do not saturate. Illustrated for 15 large language models and 63 tasks, high explanatory power is unleashed from inspecting the demand and ability profiles, bringing insights on the sensitivity and specificity exhibited by different benchmarks, and how knowledge, metacognition and reasoning are affected by model size, chain-of-thought and distillation. Surprisingly, high predictive power at the instance level becomes possible using these demand levels, providing superior estimates over black-box baseline predictors based on embeddings or finetuning, especially in out-of-distribution settings (new tasks and new benchmarks). The scales, rubrics, battery, techniques and results presented here represent a major step for AI evaluation, underpinning the reliable deployment of AI in the years ahead. 

---
# SKG-LLM: Developing a Mathematical Model for Stroke Knowledge Graph Construction Using Large Language Models 

**Authors**: Ali Sarabadani, Kheirolah Rahsepar Fard, Hamid Dalvand  

**Link**: [PDF](https://arxiv.org/pdf/2503.06475)  

**Abstract**: The purpose of this study is to introduce SKG-LLM. A knowledge graph (KG) is constructed from stroke-related articles using mathematical and large language models (LLMs). SKG-LLM extracts and organizes complex relationships from the biomedical literature, using it to increase the accuracy and depth of KG in stroke research. In the proposed method, GPT-4 was used for data pre-processing, and the extraction of embeddings was also done by GPT-4 in the whole KG construction process. The performance of the proposed model was tested with two evaluation criteria: Precision and Recall. For further validation of the proposed model, GPT-4 was used. Compared with Wikidata and WN18RR, the proposed KG-LLM approach performs better, especially in precision and recall. By including GPT-4 in the preprocessing process, the SKG-LLM model achieved a precision score of 0.906 and a recall score of 0.923. Expert reviews further improved the results and increased precision to 0.923 and recall to 0.918. The knowledge graph constructed by SKG-LLM contains 2692 nodes and 5012 edges, which are 13 distinct types of nodes and 24 types of edges. 

---
# Critical Foreign Policy Decisions (CFPD)-Benchmark: Measuring Diplomatic Preferences in Large Language Models 

**Authors**: Benjamin Jensen, Ian Reynolds, Yasir Atalan, Michael Garcia, Austin Woo, Anthony Chen, Trevor Howarth  

**Link**: [PDF](https://arxiv.org/pdf/2503.06263)  

**Abstract**: As national security institutions increasingly integrate Artificial Intelligence (AI) into decision-making and content generation processes, understanding the inherent biases of large language models (LLMs) is crucial. This study presents a novel benchmark designed to evaluate the biases and preferences of seven prominent foundation models-Llama 3.1 8B Instruct, Llama 3.1 70B Instruct, GPT-4o, Gemini 1.5 Pro-002, Mixtral 8x22B, Claude 3.5 Sonnet, and Qwen2 72B-in the context of international relations (IR). We designed a bias discovery study around core topics in IR using 400-expert crafted scenarios to analyze results from our selected models. These scenarios focused on four topical domains including: military escalation, military and humanitarian intervention, cooperative behavior in the international system, and alliance dynamics. Our analysis reveals noteworthy variation among model recommendations based on scenarios designed for the four tested domains. Particularly, Qwen2 72B, Gemini 1.5 Pro-002 and Llama 3.1 8B Instruct models offered significantly more escalatory recommendations than Claude 3.5 Sonnet and GPT-4o models. All models exhibit some degree of country-specific biases, often recommending less escalatory and interventionist actions for China and Russia compared to the United States and the United Kingdom. These findings highlight the necessity for controlled deployment of LLMs in high-stakes environments, emphasizing the need for domain-specific evaluations and model fine-tuning to align with institutional objectives. 

---
# Rank-R1: Enhancing Reasoning in LLM-based Document Rerankers via Reinforcement Learning 

**Authors**: Shengyao Zhuang, Xueguang Ma, Bevan Koopman, Jimmy Lin, Guido Zuccon  

**Link**: [PDF](https://arxiv.org/pdf/2503.06034)  

**Abstract**: In this paper, we introduce Rank-R1, a novel LLM-based reranker that performs reasoning over both the user query and candidate documents before performing the ranking task. Existing document reranking methods based on large language models (LLMs) typically rely on prompting or fine-tuning LLMs to order or label candidate documents according to their relevance to a query. For Rank-R1, we use a reinforcement learning algorithm along with only a small set of relevance labels (without any reasoning supervision) to enhance the reasoning ability of LLM-based rerankers. Our hypothesis is that adding reasoning capabilities to the rerankers can improve their relevance assessement and ranking capabilities. Our experiments on the TREC DL and BRIGHT datasets show that Rank-R1 is highly effective, especially for complex queries. In particular, we find that Rank-R1 achieves effectiveness on in-domain datasets at par with that of supervised fine-tuning methods, but utilizing only 18\% of the training data used by the fine-tuning methods. We also find that the model largely outperforms zero-shot and supervised fine-tuning when applied to out-of-domain datasets featuring complex queries, especially when a 14B-size model is used. Finally, we qualitatively observe that Rank-R1's reasoning process improves the explainability of the ranking results, opening new opportunities for search engine results presentation and fruition. 

---
# MedSimAI: Simulation and Formative Feedback Generation to Enhance Deliberate Practice in Medical Education 

**Authors**: Yann Hicke, Jadon Geathers, Niroop Rajashekar, Colleen Chan, Anyanate Gwendolyne Jack, Justin Sewell, Mackenzi Preston, Susannah Cornes, Dennis Shung, Rene Kizilcec  

**Link**: [PDF](https://arxiv.org/pdf/2503.05793)  

**Abstract**: Medical education faces challenges in scalability, accessibility, and consistency, particularly in clinical skills training for physician-patient communication. Traditional simulation-based learning, while effective, is resource-intensive, difficult to schedule, and often highly variable in feedback quality. Through a collaboration between AI, learning science, and medical education experts, we co-developed MedSimAI, an AI-powered simulation platform that enables deliberate practice, self-regulated learning (SRL), and automated assessment through interactive patient encounters. Leveraging large language models (LLMs), MedSimAI generates realistic clinical interactions and provides immediate, structured feedback using established medical evaluation frameworks such as the Master Interview Rating Scale (MIRS). In a pilot study with 104 first-year medical students, we examined engagement, conversation patterns, and user perceptions. Students found MedSimAI beneficial for repeated, realistic patient-history practice. Conversation analysis revealed that certain higher-order skills were often overlooked, though students generally performed systematic histories and empathic listening. By integrating unlimited practice opportunities, real-time AI assessment, and SRL principles, MedSimAI addresses key limitations of traditional simulation-based training, making high-quality clinical education more accessible and scalable. 

---
# Emergent Abilities in Large Language Models: A Survey 

**Authors**: Leonardo Berti, Flavio Giorgi, Gjergji Kasneci  

**Link**: [PDF](https://arxiv.org/pdf/2503.05788)  

**Abstract**: Large Language Models (LLMs) are leading a new technological revolution as one of the most promising research streams toward artificial general intelligence. The scaling of these models, accomplished by increasing the number of parameters and the magnitude of the training datasets, has been linked to various so-called emergent abilities that were previously unobserved. These emergent abilities, ranging from advanced reasoning and in-context learning to coding and problem-solving, have sparked an intense scientific debate: Are they truly emergent, or do they simply depend on external factors, such as training dynamics, the type of problems, or the chosen metric? What underlying mechanism causes them? Despite their transformative potential, emergent abilities remain poorly understood, leading to misconceptions about their definition, nature, predictability, and implications. In this work, we shed light on emergent abilities by conducting a comprehensive review of the phenomenon, addressing both its scientific underpinnings and real-world consequences. We first critically analyze existing definitions, exposing inconsistencies in conceptualizing emergent abilities. We then explore the conditions under which these abilities appear, evaluating the role of scaling laws, task complexity, pre-training loss, quantization, and prompting strategies. Our review extends beyond traditional LLMs and includes Large Reasoning Models (LRMs), which leverage reinforcement learning and inference-time search to amplify reasoning and self-reflection. However, emergence is not inherently positive. As AI systems gain autonomous reasoning capabilities, they also develop harmful behaviors, including deception, manipulation, and reward hacking. We highlight growing concerns about safety and governance, emphasizing the need for better evaluation frameworks and regulatory oversight. 

---
# Uncertainty-Aware Fusion: An Ensemble Framework for Mitigating Hallucinations in Large Language Models 

**Authors**: Prasenjit Dey, Srujana Merugu, Sivaramakrishnan Kaveri  

**Link**: [PDF](https://arxiv.org/pdf/2503.05757)  

**Abstract**: Large Language Models (LLMs) are known to hallucinate and generate non-factual outputs which can undermine user trust. Traditional methods to directly mitigate hallucinations, such as representation editing and contrastive decoding, often require additional training data and involve high implementation complexity. While ensemble-based approaches harness multiple LLMs to tap into the "wisdom of crowds", these methods overlook uncertainties in individual model responses. Recent studies reveal that uncertainty estimation can enable LLMs to self-assess the likelihood of generating hallucinations. In this work, we focus on factoid question answering (QA) and observe that LLMs accuracy and self-assessment capabilities vary widely with different models excelling in different scenarios. Leveraging this insight, we propose Uncertainty-Aware Fusion (UAF), an ensemble framework to reduces hallucinations by strategically combining multiple LLM based on their accuracy and self-assessment abilities. Empirical results on several public benchmark datasets show that UAF outperforms state-of-the-art hallucination mitigation methods by $8\%$ in factual accuracy, while either narrowing or surpassing the performance gap with GPT-4. 

---
# Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models 

**Authors**: Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Yao Hu, Shaohui Lin  

**Link**: [PDF](https://arxiv.org/pdf/2503.06749)  

**Abstract**: DeepSeek-R1-Zero has successfully demonstrated the emergence of reasoning capabilities in LLMs purely through Reinforcement Learning (RL). Inspired by this breakthrough, we explore how RL can be utilized to enhance the reasoning capability of MLLMs. However, direct training with RL struggles to activate complex reasoning capabilities such as questioning and reflection in MLLMs, due to the absence of substantial high-quality multimodal reasoning data. To address this issue, we propose the reasoning MLLM, Vision-R1, to improve multimodal reasoning capability. Specifically, we first construct a high-quality multimodal CoT dataset without human annotations by leveraging an existing MLLM and DeepSeek-R1 through modality bridging and data filtering to obtain a 200K multimodal CoT dataset, Vision-R1-cold dataset. It serves as cold-start initialization data for Vision-R1. To mitigate the optimization challenges caused by overthinking after cold start, we propose Progressive Thinking Suppression Training (PTST) strategy and employ Group Relative Policy Optimization (GRPO) with the hard formatting result reward function to gradually refine the model's ability to learn correct and complex reasoning processes on a 10K multimodal math dataset. Comprehensive experiments show our model achieves an average improvement of $\sim$6% across various multimodal math reasoning benchmarks. Vision-R1-7B achieves a 73.5% accuracy on the widely used MathVista benchmark, which is only 0.4% lower than the leading reasoning model, OpenAI O1. The datasets and code will be released in: this https URL . 

---
# DSGBench: A Diverse Strategic Game Benchmark for Evaluating LLM-based Agents in Complex Decision-Making Environments 

**Authors**: Wenjie Tang, Yuan Zhou, Erqiang Xu, Keyan Cheng, Minne Li, Liquan Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2503.06047)  

**Abstract**: Large Language Model~(LLM) based agents have been increasingly popular in solving complex and dynamic tasks, which requires proper evaluation systems to assess their capabilities. Nevertheless, existing benchmarks usually either focus on single-objective tasks or use overly broad assessing metrics, failing to provide a comprehensive inspection of the actual capabilities of LLM-based agents in complicated decision-making tasks. To address these issues, we introduce DSGBench, a more rigorous evaluation platform for strategic decision-making. Firstly, it incorporates six complex strategic games which serve as ideal testbeds due to their long-term and multi-dimensional decision-making demands and flexibility in customizing tasks of various difficulty levels or multiple targets. Secondly, DSGBench employs a fine-grained evaluation scoring system which examines the decision-making capabilities by looking into the performance in five specific dimensions and offering a comprehensive assessment in a well-designed way. Furthermore, DSGBench also incorporates an automated decision-tracking mechanism which enables in-depth analysis of agent behaviour patterns and the changes in their strategies. We demonstrate the advances of DSGBench by applying it to multiple popular LLM-based agents and our results suggest that DSGBench provides valuable insights in choosing LLM-based agents as well as improving their future development. DSGBench is available at this https URL. 

---
# ChatWise: AI-Powered Engaging Conversations for Enhancing Senior Cognitive Wellbeing 

**Authors**: Zhengbang Yang, Zhuangdi Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2503.05740)  

**Abstract**: Cognitive health in older adults presents a growing challenge. While conversational interventions show feasibility in improving cognitive wellness, human caregiver resources remain overburdened. AI-based methods have shown promise in providing conversational support, yet existing work is limited to implicit strategy while lacking multi-turn support tailored to seniors. We improve prior art with an LLM-driven chatbot named ChatWise for older adults. It follows dual-level conversation reasoning at the inference phase to provide engaging companionship. ChatWise thrives in long-turn conversations, in contrast to conventional LLMs that primarily excel in short-turn exchanges. Grounded experiments show that ChatWise significantly enhances simulated users' cognitive and emotional status, including those with Mild Cognitive Impairment. 

---
# Talking to GDELT Through Knowledge Graphs 

**Authors**: Audun Myers, Max Vargas, Sinan G. Aksoy, Cliff Joslyn, Benjamin Wilson, Tom Grimes  

**Link**: [PDF](https://arxiv.org/pdf/2503.07584)  

**Abstract**: In this work we study various Retrieval Augmented Regeneration (RAG) approaches to gain an understanding of the strengths and weaknesses of each approach in a question-answering analysis. To gain this understanding we use a case-study subset of the Global Database of Events, Language, and Tone (GDELT) dataset as well as a corpus of raw text scraped from the online news articles. To retrieve information from the text corpus we implement a traditional vector store RAG as well as state-of-the-art large language model (LLM) based approaches for automatically constructing KGs and retrieving the relevant subgraphs. In addition to these corpus approaches, we develop a novel ontology-based framework for constructing knowledge graphs (KGs) from GDELT directly which leverages the underlying schema of GDELT to create structured representations of global events. For retrieving relevant information from the ontology-based KGs we implement both direct graph queries and state-of-the-art graph retrieval approaches. We compare the performance of each method in a question-answering task. We find that while our ontology-based KGs are valuable for question-answering, automated extraction of the relevant subgraphs is challenging. Conversely, LLM-generated KGs, while capturing event summaries, often lack consistency and interpretability. Our findings suggest benefits of a synergistic approach between ontology and LLM-based KG construction, with proposed avenues toward that end. 

---
# Process-Supervised LLM Recommenders via Flow-guided Tuning 

**Authors**: Chongming Gao, Mengyao Gao, Chenxiao Fan, Shuai Yuan, Wentao Shi, Xiangnan He  

**Link**: [PDF](https://arxiv.org/pdf/2503.07377)  

**Abstract**: While large language models (LLMs) are increasingly adapted for recommendation systems via supervised fine-tuning (SFT), this approach amplifies popularity bias due to its likelihood maximization objective, compromising recommendation diversity and fairness. To address this, we present Flow-guided fine-tuning recommender (Flower), which replaces SFT with a Generative Flow Network (GFlowNet) framework that enacts process supervision through token-level reward propagation. Flower's key innovation lies in decomposing item-level rewards into constituent token rewards, enabling direct alignment between token generation probabilities and their reward signals. This mechanism achieves three critical advancements: (1) popularity bias mitigation and fairness enhancement through empirical distribution matching, (2) preservation of diversity through GFlowNet's proportional sampling, and (3) flexible integration of personalized preferences via adaptable token rewards. Experiments demonstrate Flower's superior distribution-fitting capability and its significant advantages over traditional SFT in terms of fairness, diversity, and accuracy, highlighting its potential to improve LLM-based recommendation systems. The implementation is available via this https URL 

---
# Image is All You Need: Towards Efficient and Effective Large Language Model-Based Recommender Systems 

**Authors**: Kibum Kim, Sein Kim, Hongseok Kang, Jiwan Kim, Heewoong Noh, Yeonjun In, Kanghoon Yoon, Jinoh Oh, Chanyoung Park  

**Link**: [PDF](https://arxiv.org/pdf/2503.06238)  

**Abstract**: Large Language Models (LLMs) have recently emerged as a powerful backbone for recommender systems. Existing LLM-based recommender systems take two different approaches for representing items in natural language, i.e., Attribute-based Representation and Description-based Representation. In this work, we aim to address the trade-off between efficiency and effectiveness that these two approaches encounter, when representing items consumed by users. Based on our interesting observation that there is a significant information overlap between images and descriptions associated with items, we propose a novel method, Image is all you need for LLM-based Recommender system (I-LLMRec). Our main idea is to leverage images as an alternative to lengthy textual descriptions for representing items, aiming at reducing token usage while preserving the rich semantic information of item descriptions. Through extensive experiments, we demonstrate that I-LLMRec outperforms existing methods in both efficiency and effectiveness by leveraging images. Moreover, a further appeal of I-LLMRec is its ability to reduce sensitivity to noise in descriptions, leading to more robust recommendations. 

---
# A Zero-shot Learning Method Based on Large Language Models for Multi-modal Knowledge Graph Embedding 

**Authors**: Bingchen Liu, Jingchen Li, Naixing Xu, Xin Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.07202)  

**Abstract**: Zero-shot learning (ZL) is crucial for tasks involving unseen categories, such as natural language processing, image classification, and cross-lingual transfer. Current applications often fail to accurately infer and handle new relations or entities involving unseen categories, severely limiting their scalability and practicality in open-domain scenarios. ZL learning faces the challenge of effectively transferring semantic information of unseen categories in multi-modal knowledge graph (MMKG) embedding representation learning. In this paper, we propose ZSLLM, a framework for zero-shot embedding learning of MMKGs using large language models (LLMs). We leverage textual modality information of unseen categories as prompts to fully utilize the reasoning capabilities of LLMs, enabling semantic information transfer across different modalities for unseen categories. Through model-based learning, the embedding representation of unseen categories in MMKG is enhanced. Extensive experiments conducted on multiple real-world datasets demonstrate the superiority of our approach compared to state-of-the-art methods. 

---
# From Text to Visuals: Using LLMs to Generate Math Diagrams with Vector Graphics 

**Authors**: Jaewook Lee, Jeongah Lee, Wanyong Feng, Andrew Lan  

**Link**: [PDF](https://arxiv.org/pdf/2503.07429)  

**Abstract**: Advances in large language models (LLMs) offer new possibilities for enhancing math education by automating support for both teachers and students. While prior work has focused on generating math problems and high-quality distractors, the role of visualization in math learning remains under-explored. Diagrams are essential for mathematical thinking and problem-solving, yet manually creating them is time-consuming and requires domain-specific expertise, limiting scalability. Recent research on using LLMs to generate Scalable Vector Graphics (SVG) presents a promising approach to automating diagram creation. Unlike pixel-based images, SVGs represent geometric figures using XML, allowing seamless scaling and adaptability. Educational platforms such as Khan Academy and IXL already use SVGs to display math problems and hints. In this paper, we explore the use of LLMs to generate math-related diagrams that accompany textual hints via intermediate SVG representations. We address three research questions: (1) how to automatically generate math diagrams in problem-solving hints and evaluate their quality, (2) whether SVG is an effective intermediate representation for math diagrams, and (3) what prompting strategies and formats are required for LLMs to generate accurate SVG-based diagrams. Our contributions include defining the task of automatically generating SVG-based diagrams for math hints, developing an LLM prompting-based pipeline, and identifying key strategies for improving diagram generation. Additionally, we introduce a Visual Question Answering-based evaluation setup and conduct ablation studies to assess different pipeline variations. By automating the math diagram creation, we aim to provide students and teachers with accurate, conceptually relevant visual aids that enhance problem-solving and learning experiences. 

---
# Queueing, Predictions, and LLMs: Challenges and Open Problems 

**Authors**: Michael Mitzenmacher, Rana Shahout  

**Link**: [PDF](https://arxiv.org/pdf/2503.07545)  

**Abstract**: Queueing systems present many opportunities for applying machine-learning predictions, such as estimated service times, to improve system performance. This integration raises numerous open questions about how predictions can be effectively leveraged to improve scheduling decisions. Recent studies explore queues with predicted service times, typically aiming to minimize job time in the system. We review these works, highlight the effectiveness of predictions, and present open questions on queue performance. We then move to consider an important practical example of using predictions in scheduling, namely Large Language Model (LLM) systems, which presents novel scheduling challenges and highlights the potential for predictions to improve performance. In particular, we consider LLMs performing inference. Inference requests (jobs) in LLM systems are inherently complex; they have variable inference times, dynamic memory footprints that are constrained by key-value (KV) store memory limitations, and multiple possible preemption approaches that affect performance differently. We provide background on the important aspects of scheduling in LLM systems, and introduce new models and open problems that arise from them. We argue that there are significant opportunities for applying insights and analysis from queueing theory to scheduling in LLM systems. 

---
# ReAgent: Reversible Multi-Agent Reasoning for Knowledge-Enhanced Multi-Hop QA 

**Authors**: Zhao Xinjie, Fan Gao, Rui Yang, Yingjian Chen, Yuyang Wang, Ying Zhu, Jiacheng Tang, Irene Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.06951)  

**Abstract**: Recent advances in large language models (LLMs) have significantly improved multi-hop question answering (QA) through direct Chain-of-Thought (CoT) reasoning. However, the irreversible nature of CoT leads to error accumulation, making it challenging to correct mistakes in multi-hop reasoning. This paper introduces ReAgent: a Reversible multi-Agent collaborative framework augmented with explicit backtracking mechanisms, enabling reversible multi-hop reasoning. By incorporating text-based retrieval, information aggregation and validation, our system can detect and correct errors mid-reasoning, leading to more robust and interpretable QA outcomes. The framework and experiments serve as a foundation for future work on error-tolerant QA systems. Empirical evaluations across three benchmarks indicate ReAgent's efficacy, yielding average about 6\% improvements against baseline models. 

---
# ProJudge: A Multi-Modal Multi-Discipline Benchmark and Instruction-Tuning Dataset for MLLM-based Process Judges 

**Authors**: Jiaxin Ai, Pengfei Zhou, Zhaopan Xu, Ming Li, Fanrui Zhang, Zizhen Li, Jianwen Sun, Yukang Feng, Baojin Huang, Zhongyuan Wang, Kaipeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.06553)  

**Abstract**: As multi-modal large language models (MLLMs) frequently exhibit errors when solving scientific problems, evaluating the validity of their reasoning processes is critical for ensuring reliability and uncovering fine-grained model weaknesses. Since human evaluation is laborious and costly, prompting MLLMs as automated process judges has become a common practice. However, the reliability of these model-based judges remains uncertain. To address this, we introduce ProJudgeBench, the first comprehensive benchmark specifically designed for evaluating abilities of MLLM-based process judges. ProJudgeBench comprises 2,400 test cases and 50,118 step-level labels, spanning four scientific disciplines with diverse difficulty levels and multi-modal content. In ProJudgeBench, each step is meticulously annotated by human experts for correctness, error type, and explanation, enabling a systematic evaluation of judges' capabilities to detect, classify and diagnose errors. Evaluation on ProJudgeBench reveals a significant performance gap between open-source and proprietary models. To bridge this gap, we further propose ProJudge-173k, a large-scale instruction-tuning dataset, and a Dynamic Dual-Phase fine-tuning strategy that encourages models to explicitly reason through problem-solving before assessing solutions. Both contributions significantly enhance the process evaluation capabilities of open-source models. All the resources will be released to foster future research of reliable multi-modal process evaluation. 

---
# ExKG-LLM: Leveraging Large Language Models for Automated Expansion of Cognitive Neuroscience Knowledge Graphs 

**Authors**: Ali Sarabadani, Kheirolah Rahsepar Fard, Hamid Dalvand  

**Link**: [PDF](https://arxiv.org/pdf/2503.06479)  

**Abstract**: The paper introduces ExKG-LLM, a framework designed to automate the expansion of cognitive neuroscience knowledge graphs (CNKG) using large language models (LLMs). It addresses limitations in existing tools by enhancing accuracy, completeness, and usefulness in CNKG. The framework leverages a large dataset of scientific papers and clinical reports, applying state-of-the-art LLMs to extract, optimize, and integrate new entities and relationships. Evaluation metrics include precision, recall, and graph density. Results show significant improvements: precision (0.80, +6.67%), recall (0.81, +15.71%), F1 score (0.805, +11.81%), and increased edge nodes (21.13% and 31.92%). Graph density slightly decreased, reflecting a broader but more fragmented structure. Engagement rates rose by 20%, while CNKG diameter increased to 15, indicating a more distributed structure. Time complexity improved to O(n log n), but space complexity rose to O(n2), indicating higher memory usage. ExKG-LLM demonstrates potential for enhancing knowledge generation, semantic search, and clinical decision-making in cognitive neuroscience, adaptable to broader scientific fields. 

---
# Performant LLM Agentic Framework for Conversational AI 

**Authors**: Alex Casella, Wayne Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.06410)  

**Abstract**: The rise of Agentic applications and automation in the Voice AI industry has led to an increased reliance on Large Language Models (LLMs) to navigate graph-based logic workflows composed of nodes and edges. However, existing methods face challenges such as alignment errors in complex workflows and hallucinations caused by excessive context size. To address these limitations, we introduce the Performant Agentic Framework (PAF), a novel system that assists LLMs in selecting appropriate nodes and executing actions in order when traversing complex graphs. PAF combines LLM-based reasoning with a mathematically grounded vector scoring mechanism, achieving both higher accuracy and reduced latency. Our approach dynamically balances strict adherence to predefined paths with flexible node jumps to handle various user inputs efficiently. Experiments demonstrate that PAF significantly outperforms baseline methods, paving the way for scalable, real-time Conversational AI systems in complex business environments. 

---
# Advancing AI Negotiations: New Theory and Evidence from a Large-Scale Autonomous Negotiations Competition 

**Authors**: Michelle Vaccaro, Michael Caoson, Harang Ju, Sinan Aral, Jared R. Curhan  

**Link**: [PDF](https://arxiv.org/pdf/2503.06416)  

**Abstract**: Despite the rapid proliferation of artificial intelligence (AI) negotiation agents, there has been limited integration of computer science research and established negotiation theory to develop new theories of AI negotiation. To bridge this gap, we conducted an International AI Negotiations Competition in which participants iteratively designed and refined prompts for large language model (LLM) negotiation agents. We then facilitated over 120,000 negotiations between these agents across multiple scenarios with diverse characteristics and objectives. Our findings revealed that fundamental principles from established human-human negotiation theory remain crucial in AI-AI negotiations. Specifically, agents exhibiting high warmth fostered higher counterpart subjective value and reached deals more frequently, which enabled them to create and claim more value in integrative settings. However, conditional on reaching a deal, warm agents claimed less value while dominant agents claimed more value. These results align with classic negotiation theory emphasizing relationship-building, assertiveness, and preparation. Our analysis also revealed unique dynamics in AI-AI negotiations not fully explained by negotiation theory, particularly regarding the effectiveness of AI-specific strategies like chain-of-thought reasoning and prompt injection. The agent that won our competition implemented an approach that blended traditional negotiation preparation frameworks with AI-specific methods. Together, these results suggest the importance of establishing a new theory of AI negotiations that integrates established negotiation theory with AI-specific strategies to optimize agent performance. Our research suggests this new theory must account for the unique characteristics of autonomous agents and establish the conditions under which traditional negotiation theory applies in automated settings. 

---
# Agent models: Internalizing Chain-of-Action Generation into Reasoning models 

**Authors**: Yuxiang Zhang, Yuqi Yang, Jiangming Shu, Xinyan Wen, Jitao Sang  

**Link**: [PDF](https://arxiv.org/pdf/2503.06580)  

**Abstract**: Traditional agentic workflows rely on external prompts to manage interactions with tools and the environment, which limits the autonomy of reasoning models. We position \emph{Large Agent Models (LAMs)} that internalize the generation of \emph{Chain-of-Action (CoA)}, enabling the model to autonomously decide when and how to use external tools. Our proposed AutoCoA framework combines supervised fine-tuning (SFT) and reinforcement learning (RL), allowing the model to seamlessly switch between reasoning and action while efficiently managing environment interactions. Main components include step-level action triggering, trajectory-level CoA optimization, and an internal world model to reduce real-environment interaction costs. Evaluations on open-domain QA tasks demonstrate that AutoCoA-trained agent models significantly outperform ReAct-based workflows in task completion, especially in tasks that require long-term reasoning and multi-step actions. Code and dataset are available at this https URL 

---
# Enhancing Reasoning with Collaboration and Memory 

**Authors**: Julie Michelman, Nasrin Baratalipour, Matthew Abueg  

**Link**: [PDF](https://arxiv.org/pdf/2503.05944)  

**Abstract**: We envision a continuous collaborative learning system where groups of LLM agents work together to solve reasoning problems, drawing on memory they collectively build to improve performance as they gain experience. This work establishes the foundations for such a system by studying the interoperability of chain-of-thought reasoning styles, multi-agent collaboration, and memory banks. Extending beyond the identical agents of self-consistency, we introduce varied-context agents with diverse exemplars and a summarizer agent in place of voting. We generate frozen and continuously learned memory banks of exemplars and pair them with fixed, random, and similarity-based retrieval mechanisms. Our systematic study reveals where various methods contribute to reasoning performance of two LLMs on three grounded reasoning tasks, showing that random exemplar selection can often beat more principled approaches, and in some tasks, inclusion of any exemplars serves only to distract both weak and strong models. 

---
# DriveGen: Towards Infinite Diverse Traffic Scenarios with Large Models 

**Authors**: Shenyu Zhang, Jiaguo Tian, Zhengbang Zhu, Shan Huang, Jucheng Yang, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.05808)  

**Abstract**: Microscopic traffic simulation has become an important tool for autonomous driving training and testing. Although recent data-driven approaches advance realistic behavior generation, their learning still relies primarily on a single real-world dataset, which limits their diversity and thereby hinders downstream algorithm optimization. In this paper, we propose DriveGen, a novel traffic simulation framework with large models for more diverse traffic generation that supports further customized designs. DriveGen consists of two internal stages: the initialization stage uses large language model and retrieval technique to generate map and vehicle assets; the rollout stage outputs trajectories with selected waypoint goals from visual language model and a specific designed diffusion planner. Through this two-staged process, DriveGen fully utilizes large models' high-level cognition and reasoning of driving behavior, obtaining greater diversity beyond datasets while maintaining high realism. To support effective downstream optimization, we additionally develop DriveGen-CS, an automatic corner case generation pipeline that uses failures of the driving algorithm as additional prompt knowledge for large models without the need for retraining or fine-tuning. Experiments show that our generated scenarios and corner cases have a superior performance compared to state-of-the-art baselines. Downstream experiments further verify that the synthesized traffic of DriveGen provides better optimization of the performance of typical driving algorithms, demonstrating the effectiveness of our framework. 

---
# Market-based Architectures in RL and Beyond 

**Authors**: Abhimanyu Pallavi Sudhir, Long Tran-Thanh  

**Link**: [PDF](https://arxiv.org/pdf/2503.05828)  

**Abstract**: Market-based agents refer to reinforcement learning agents which determine their actions based on an internal market of sub-agents. We introduce a new type of market-based algorithm where the state itself is factored into several axes called ``goods'', which allows for greater specialization and parallelism than existing market-based RL algorithms. Furthermore, we argue that market-based algorithms have the potential to address many current challenges in AI, such as search, dynamic scaling and complete feedback, and demonstrate that they may be seen to generalize neural networks; finally, we list some novel ways that market algorithms may be applied in conjunction with Large Language Models for immediate practical applicability. 

---
# Breaking Free from MMI: A New Frontier in Rationalization by Probing Input Utilization 

**Authors**: Wei Liu, Zhiying Deng, Zhongyu Niu, Jun Wang, Haozhao Wang, Zhigang Zeng, Ruixuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.06202)  

**Abstract**: Extracting a small subset of crucial rationales from the full input is a key problem in explainability research. The most widely used fundamental criterion for rationale extraction is the maximum mutual information (MMI) criterion. In this paper, we first demonstrate that MMI suffers from diminishing marginal returns. Once part of the rationale has been identified, finding the remaining portions contributes only marginally to increasing the mutual information, making it difficult to use MMI to locate the rest. In contrast to MMI that aims to reproduce the prediction, we seek to identify the parts of the input that the network can actually utilize.
This is achieved by comparing how different rationale candidates match the capability space of the weight matrix. The weight matrix of a neural network is typically low-rank, meaning that the linear combinations of its column vectors can only cover part of the directions in a high-dimensional space (high-dimension: the dimensions of an input vector). If an input is fully utilized by the network, {it generally matches these directions (e.g., a portion of a hypersphere), resulting in a representation with a high norm. Conversely, if an input primarily falls outside (orthogonal to) these directions}, its representation norm will approach zero, behaving like noise that the network cannot effectively utilize. Building on this, we propose using the norms of rationale candidates as an alternative objective to MMI. Through experiments on four text classification datasets and one graph classification dataset using three network architectures (GRUs, BERT, and GCN), we show that our method outperforms MMI and its improved variants in identifying better rationales. We also compare our method with a representative LLM (llama-3.1-8b-instruct) and find that our simple method gets comparable results to it and can sometimes even outperform it. 

---
# VACT: A Video Automatic Causal Testing System and a Benchmark 

**Authors**: Haotong Yang, Qingyuan Zheng, Yunjian Gao, Yongkun Yang, Yangbo He, Zhouchen Lin, Muhan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.06163)  

**Abstract**: With the rapid advancement of text-conditioned Video Generation Models (VGMs), the quality of generated videos has significantly improved, bringing these models closer to functioning as ``*world simulators*'' and making real-world-level video generation more accessible and cost-effective. However, the generated videos often contain factual inaccuracies and lack understanding of fundamental physical laws. While some previous studies have highlighted this issue in limited domains through manual analysis, a comprehensive solution has not yet been established, primarily due to the absence of a generalized, automated approach for modeling and assessing the causal reasoning of these models across diverse scenarios. To address this gap, we propose VACT: an **automated** framework for modeling, evaluating, and measuring the causal understanding of VGMs in real-world scenarios. By combining causal analysis techniques with a carefully designed large language model assistant, our system can assess the causal behavior of models in various contexts without human annotation, which offers strong generalization and scalability. Additionally, we introduce multi-level causal evaluation metrics to provide a detailed analysis of the causal performance of VGMs. As a demonstration, we use our framework to benchmark several prevailing VGMs, offering insight into their causal reasoning capabilities. Our work lays the foundation for systematically addressing the causal understanding deficiencies in VGMs and contributes to advancing their reliability and real-world applicability. 

---
# NeuroChat: A Neuroadaptive AI Chatbot for Customizing Learning Experiences 

**Authors**: Dünya Baradari, Nataliya Kosmyna, Oscar Petrov, Rebecah Kaplun, Pattie Maes  

**Link**: [PDF](https://arxiv.org/pdf/2503.07599)  

**Abstract**: Generative AI is transforming education by enabling personalized, on-demand learning experiences. However, AI tutors lack the ability to assess a learner's cognitive state in real time, limiting their adaptability. Meanwhile, electroencephalography (EEG)-based neuroadaptive systems have successfully enhanced engagement by dynamically adjusting learning content. This paper presents NeuroChat, a proof-of-concept neuroadaptive AI tutor that integrates real-time EEG-based engagement tracking with generative AI. NeuroChat continuously monitors a learner's cognitive engagement and dynamically adjusts content complexity, response style, and pacing using a closed-loop system. We evaluate this approach in a pilot study (n=24), comparing NeuroChat to a standard LLM-based chatbot. Results indicate that NeuroChat enhances cognitive and subjective engagement but does not show an immediate effect on learning outcomes. These findings demonstrate the feasibility of real-time cognitive feedback in LLMs, highlighting new directions for adaptive learning, AI tutoring, and human-AI interaction. 

---
# Artificial Utopia: Simulation and Intelligent Agents for a Democratised Future 

**Authors**: Yannick Oswald  

**Link**: [PDF](https://arxiv.org/pdf/2503.07364)  

**Abstract**: Prevailing top-down systems in politics and economics struggle to keep pace with the pressing challenges of the 21st century, such as climate change, social inequality and conflict. Bottom-up democratisation and participatory approaches in politics and economics are increasingly seen as promising alternatives to confront and overcome these issues, often with utopian overtones, as proponents believe they may dramatically reshape political, social and ecological futures for the better and in contrast to contemporary authoritarian tendencies across various countries. Institutional specifics and the associated collective human behavior or culture remains little understood and debated, however. In this article, I propose a novel research agenda focusing on utopian democratisation efforts with formal and computational methods as well as with artificial intelligence - I call this agenda Artificial Utopia. Artificial Utopias provide safe testing grounds for new political ideas and economic policies in-silico with reduced risk of negative consequences as compared to testing ideas in real-world contexts. An increasing number of advanced simulation and intelligence methods, that aim at representing human cognition and collective decision-making in more realistic ways, could benefit this process. This includes agent-based modelling, reinforcement learning, large language models and more. I clarify what some of these simulation approaches can contribute to the study of Artificial Utopias with the help of two institutional examples: the citizen assembly and the democratic firm. 

---
# V2Flow: Unifying Visual Tokenization and Large Language Model Vocabularies for Autoregressive Image Generation 

**Authors**: Guiwei Zhang, Tianyu Zhang, Mohan Zhou, Yalong Bai, Biye Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.07493)  

**Abstract**: We propose V2Flow, a novel tokenizer that produces discrete visual tokens capable of high-fidelity reconstruction, while ensuring structural and latent distribution alignment with the vocabulary space of large language models (LLMs). Leveraging this tight visual-vocabulary coupling, V2Flow enables autoregressive visual generation on top of existing LLMs. Our approach formulates visual tokenization as a flow-matching problem, aiming to learn a mapping from a standard normal prior to the continuous image distribution, conditioned on token sequences embedded within the LLMs vocabulary space. The effectiveness of V2Flow stems from two core designs. First, we propose a Visual Vocabulary resampler, which compresses visual data into compact token sequences, with each represented as a soft categorical distribution over LLM's vocabulary. This allows seamless integration of visual tokens into existing LLMs for autoregressive visual generation. Second, we present a masked autoregressive Rectified-Flow decoder, employing a masked transformer encoder-decoder to refine visual tokens into contextually enriched embeddings. These embeddings then condition a dedicated velocity field for precise reconstruction. Additionally, an autoregressive rectified-flow sampling strategy is incorporated, ensuring flexible sequence lengths while preserving competitive reconstruction quality. Extensive experiments show that V2Flow outperforms mainstream VQ-based tokenizers and facilitates autoregressive visual generation on top of existing. this https URL 

---
# Dynamic Path Navigation for Motion Agents with LLM Reasoning 

**Authors**: Yubo Zhao, Qi Wu, Yifan Wang, Yu-Wing Tai, Chi-Keung Tang  

**Link**: [PDF](https://arxiv.org/pdf/2503.07323)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong generalizable reasoning and planning capabilities. However, their efficacies in spatial path planning and obstacle-free trajectory generation remain underexplored. Leveraging LLMs for navigation holds significant potential, given LLMs' ability to handle unseen scenarios, support user-agent interactions, and provide global control across complex systems, making them well-suited for agentic planning and humanoid motion generation. As one of the first studies in this domain, we explore the zero-shot navigation and path generation capabilities of LLMs by constructing a dataset and proposing an evaluation protocol. Specifically, we represent paths using anchor points connected by straight lines, enabling movement in various directions. This approach offers greater flexibility and practicality compared to previous methods while remaining simple and intuitive for LLMs. We demonstrate that, when tasks are well-structured in this manner, modern LLMs exhibit substantial planning proficiency in avoiding obstacles while autonomously refining navigation with the generated motion to reach the target. Further, this spatial reasoning ability of a single LLM motion agent interacting in a static environment can be seamlessly generalized in multi-motion agents coordination in dynamic environments. Unlike traditional approaches that rely on single-step planning or local policies, our training-free LLM-based method enables global, dynamic, closed-loop planning, and autonomously resolving collision issues. 

---
# Self-Corrective Task Planning by Inverse Prompting with Large Language Models 

**Authors**: Jiho Lee, Hayun Lee, Jonghyeon Kim, Kyungjae Lee, Eunwoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.07317)  

**Abstract**: In robot task planning, large language models (LLMs) have shown significant promise in generating complex and long-horizon action sequences. However, it is observed that LLMs often produce responses that sound plausible but are not accurate. To address these problems, existing methods typically employ predefined error sets or external knowledge sources, requiring human efforts and computation resources. Recently, self-correction approaches have emerged, where LLM generates and refines plans, identifying errors by itself. Despite their effectiveness, they are more prone to failures in correction due to insufficient reasoning. In this paper, we introduce InversePrompt, a novel self-corrective task planning approach that leverages inverse prompting to enhance interpretability. Our method incorporates reasoning steps to provide clear, interpretable feedback. It generates inverse actions corresponding to the initially generated actions and verifies whether these inverse actions can restore the system to its original state, explicitly validating the logical coherence of the generated this http URL results on benchmark datasets show an average 16.3% higher success rate over existing LLM-based task planning methods. Our approach offers clearer justifications for feedback in real-world environments, resulting in more successful task completion than existing self-correction approaches across various scenarios. 

---
# CoT-Drive: Efficient Motion Forecasting for Autonomous Driving with LLMs and Chain-of-Thought Prompting 

**Authors**: Haicheng Liao, Hanlin Kong, Bonan Wang, Chengyue Wang, Wang Ye, Zhengbing He, Chengzhong Xu, Zhenning Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.07234)  

**Abstract**: Accurate motion forecasting is crucial for safe autonomous driving (AD). This study proposes CoT-Drive, a novel approach that enhances motion forecasting by leveraging large language models (LLMs) and a chain-of-thought (CoT) prompting method. We introduce a teacher-student knowledge distillation strategy to effectively transfer LLMs' advanced scene understanding capabilities to lightweight language models (LMs), ensuring that CoT-Drive operates in real-time on edge devices while maintaining comprehensive scene understanding and generalization capabilities. By leveraging CoT prompting techniques for LLMs without additional training, CoT-Drive generates semantic annotations that significantly improve the understanding of complex traffic environments, thereby boosting the accuracy and robustness of predictions. Additionally, we present two new scene description datasets, Highway-Text and Urban-Text, designed for fine-tuning lightweight LMs to generate context-specific semantic annotations. Comprehensive evaluations of five real-world datasets demonstrate that CoT-Drive outperforms existing models, highlighting its effectiveness and efficiency in handling complex traffic scenarios. Overall, this study is the first to consider the practical application of LLMs in this field. It pioneers the training and use of a lightweight LLM surrogate for motion forecasting, setting a new benchmark and showcasing the potential of integrating LLMs into AD systems. 

---
# Lightweight Multimodal Artificial Intelligence Framework for Maritime Multi-Scene Recognition 

**Authors**: Xinyu Xi, Hua Yang, Shentai Zhang, Yijie Liu, Sijin Sun, Xiuju Fu  

**Link**: [PDF](https://arxiv.org/pdf/2503.06978)  

**Abstract**: Maritime Multi-Scene Recognition is crucial for enhancing the capabilities of intelligent marine robotics, particularly in applications such as marine conservation, environmental monitoring, and disaster response. However, this task presents significant challenges due to environmental interference, where marine conditions degrade image quality, and the complexity of maritime scenes, which requires deeper reasoning for accurate recognition. Pure vision models alone are insufficient to address these issues. To overcome these limitations, we propose a novel multimodal Artificial Intelligence (AI) framework that integrates image data, textual descriptions and classification vectors generated by a Multimodal Large Language Model (MLLM), to provide richer semantic understanding and improve recognition accuracy. Our framework employs an efficient multimodal fusion mechanism to further enhance model robustness and adaptability in complex maritime environments. Experimental results show that our model achieves 98$\%$ accuracy, surpassing previous SOTA models by 3.5$\%$. To optimize deployment on resource-constrained platforms, we adopt activation-aware weight quantization (AWQ) as a lightweight technique, reducing the model size to 68.75MB with only a 0.5$\%$ accuracy drop while significantly lowering computational overhead. This work provides a high-performance solution for real-time maritime scene recognition, enabling Autonomous Surface Vehicles (ASVs) to support environmental monitoring and disaster response in resource-limited settings. 

---
# Large Language Model Guided Progressive Feature Alignment for Multimodal UAV Object Detection 

**Authors**: Wentao Wu, Chenglong Li, Xiao Wang, Bin Luo, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.06948)  

**Abstract**: Existing multimodal UAV object detection methods often overlook the impact of semantic gaps between modalities, which makes it difficult to achieve accurate semantic and spatial alignments, limiting detection performance. To address this problem, we propose a Large Language Model (LLM) guided Progressive feature Alignment Network called LPANet, which leverages the semantic features extracted from a large language model to guide the progressive semantic and spatial alignment between modalities for multimodal UAV object detection. To employ the powerful semantic representation of LLM, we generate the fine-grained text descriptions of each object category by ChatGPT and then extract the semantic features using the large language model MPNet. Based on the semantic features, we guide the semantic and spatial alignments in a progressive manner as follows. First, we design the Semantic Alignment Module (SAM) to pull the semantic features and multimodal visual features of each object closer, alleviating the semantic differences of objects between modalities. Second, we design the Explicit Spatial alignment Module (ESM) by integrating the semantic relations into the estimation of feature-level offsets, alleviating the coarse spatial misalignment between modalities. Finally, we design the Implicit Spatial alignment Module (ISM), which leverages the cross-modal correlations to aggregate key features from neighboring regions to achieve implicit spatial alignment. Comprehensive experiments on two public multimodal UAV object detection datasets demonstrate that our approach outperforms state-of-the-art multimodal UAV object detectors. 

---
# Graphormer-Guided Task Planning: Beyond Static Rules with LLM Safety Perception 

**Authors**: Wanjing Huang, Tongjie Pan, Yalan Ye  

**Link**: [PDF](https://arxiv.org/pdf/2503.06866)  

**Abstract**: Recent advancements in large language models (LLMs) have expanded their role in robotic task planning. However, while LLMs have been explored for generating feasible task sequences, their ability to ensure safe task execution remains underdeveloped. Existing methods struggle with structured risk perception, making them inadequate for safety-critical applications where low-latency hazard adaptation is required. To address this limitation, we propose a Graphormer-enhanced risk-aware task planning framework that combines LLM-based decision-making with structured safety modeling. Our approach constructs a dynamic spatio-semantic safety graph, capturing spatial and contextual risk factors to enable online hazard detection and adaptive task refinement. Unlike existing methods that rely on predefined safety constraints, our framework introduces a context-aware risk perception module that continuously refines safety predictions based on real-time task execution. This enables a more flexible and scalable approach to robotic planning, allowing for adaptive safety compliance beyond static rules. To validate our framework, we conduct experiments in the AI2-THOR environment. The experiments results validates improvements in risk detection accuracy, rising safety notice, and task adaptability of our framework in continuous environments compared to static rule-based and LLM-only baselines. Our project is available at this https URL 

---
# Combating Partial Perception Deficit in Autonomous Driving with Multimodal LLM Commonsense 

**Authors**: Yuting Hu, Chenhui Xu, Ruiyang Qin, Dancheng Liu, Amir Nassereldine, Yiyu Shi, Jinjun Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2503.07020)  

**Abstract**: Partial perception deficits can compromise autonomous vehicle safety by disrupting environmental understanding. Current protocols typically respond with immediate stops or minimal-risk maneuvers, worsening traffic flow and lacking flexibility for rare driving scenarios. In this paper, we propose LLM-RCO, a framework leveraging large language models to integrate human-like driving commonsense into autonomous systems facing perception deficits. LLM-RCO features four key modules: hazard inference, short-term motion planner, action condition verifier, and safety constraint generator. These modules interact with the dynamic driving environment, enabling proactive and context-aware control actions to override the original control policy of autonomous agents. To improve safety in such challenging conditions, we construct DriveLM-Deficit, a dataset of 53,895 video clips featuring deficits of safety-critical objects, complete with annotations for LLM-based hazard inference and motion planning fine-tuning. Extensive experiments in adverse driving conditions with the CARLA simulator demonstrate that systems equipped with LLM-RCO significantly improve driving performance, highlighting its potential for enhancing autonomous driving resilience against adverse perception deficits. Our results also show that LLMs fine-tuned with DriveLM-Deficit can enable more proactive movements instead of conservative stops in the context of perception deficits. 

---
# AutoMisty: A Multi-Agent LLM Framework for Automated Code Generation in the Misty Social Robot 

**Authors**: Xiao Wang, Lu Dong, Sahana Rangasrinivasan, Ifeoma Nwogu, Srirangaraj Setlur, Venugopal Govindaraju  

**Link**: [PDF](https://arxiv.org/pdf/2503.06791)  

**Abstract**: The social robot's open API allows users to customize open-domain interactions. However, it remains inaccessible to those without programming experience. In this work, we introduce AutoMisty, the first multi-agent collaboration framework powered by large language models (LLMs), to enable the seamless generation of executable Misty robot code from natural language instructions. AutoMisty incorporates four specialized agent modules to manage task decomposition, assignment, problem-solving, and result synthesis. Each agent incorporates a two-layer optimization mechanism, with self-reflection for iterative refinement and human-in-the-loop for better alignment with user preferences. AutoMisty ensures a transparent reasoning process, allowing users to iteratively refine tasks through natural language feedback for precise execution. To evaluate AutoMisty's effectiveness, we designed a benchmark task set spanning four levels of complexity and conducted experiments in a real Misty robot environment. Extensive evaluations demonstrate that AutoMisty not only consistently generates high-quality code but also enables precise code control, significantly outperforming direct reasoning with ChatGPT-4o and ChatGPT-o1. All code, optimized APIs, and experimental videos will be publicly released through the webpage: this https URL 

---
# Exploring LLM Agents for Cleaning Tabular Machine Learning Datasets 

**Authors**: Tommaso Bendinelli, Artur Dox, Christian Holz  

**Link**: [PDF](https://arxiv.org/pdf/2503.06664)  

**Abstract**: High-quality, error-free datasets are a key ingredient in building reliable, accurate, and unbiased machine learning (ML) models. However, real world datasets often suffer from errors due to sensor malfunctions, data entry mistakes, or improper data integration across multiple sources that can severely degrade model performance. Detecting and correcting these issues typically require tailor-made solutions and demand extensive domain expertise. Consequently, automation is challenging, rendering the process labor-intensive and tedious. In this study, we investigate whether Large Language Models (LLMs) can help alleviate the burden of manual data cleaning. We set up an experiment in which an LLM, paired with Python, is tasked with cleaning the training dataset to improve the performance of a learning algorithm without having the ability to modify the training pipeline or perform any feature engineering. We run this experiment on multiple Kaggle datasets that have been intentionally corrupted with errors. Our results show that LLMs can identify and correct erroneous entries, such as illogical values or outlier, by leveraging contextual information from other features within the same row, as well as feedback from previous iterations. However, they struggle to detect more complex errors that require understanding data distribution across multiple rows, such as trends and biases. 

---
# Human Cognition Inspired RAG with Knowledge Graph for Complex Problem Solving 

**Authors**: Yao Cheng, Yibo Zhao, Jiapeng Zhu, Yao Liu, Xing Sun, Xiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.06567)  

**Abstract**: Large language models (LLMs) have demonstrated transformative potential across various domains, yet they face significant challenges in knowledge integration and complex problem reasoning, often leading to hallucinations and unreliable outputs. Retrieval-Augmented Generation (RAG) has emerged as a promising solution to enhance LLMs accuracy by incorporating external knowledge. However, traditional RAG systems struggle with processing complex relational information and multi-step reasoning, limiting their effectiveness in advanced problem-solving tasks. To address these limitations, we propose CogGRAG, a cognition inspired graph-based RAG framework, designed to improve LLMs performance in Knowledge Graph Question Answering (KGQA). Inspired by the human cognitive process of decomposing complex problems and performing self-verification, our framework introduces a three-stage methodology: decomposition, retrieval, and reasoning with self-verification. By integrating these components, CogGRAG enhances the accuracy of LLMs in complex problem solving. We conduct systematic experiments with three LLM backbones on four benchmark datasets, where CogGRAG outperforms the baselines. 

---
# From Motion Signals to Insights: A Unified Framework for Student Behavior Analysis and Feedback in Physical Education Classes 

**Authors**: Xian Gao, Jiacheng Ruan, Jingsheng Gao, Mingye Xie, Zongyun Zhang, Ting Liu, Yuzhuo Fu  

**Link**: [PDF](https://arxiv.org/pdf/2503.06525)  

**Abstract**: Analyzing student behavior in educational scenarios is crucial for enhancing teaching quality and student engagement. Existing AI-based models often rely on classroom video footage to identify and analyze student behavior. While these video-based methods can partially capture and analyze student actions, they struggle to accurately track each student's actions in physical education classes, which take place in outdoor, open spaces with diverse activities, and are challenging to generalize to the specialized technical movements involved in these settings. Furthermore, current methods typically lack the ability to integrate specialized pedagogical knowledge, limiting their ability to provide in-depth insights into student behavior and offer feedback for optimizing instructional design. To address these limitations, we propose a unified end-to-end framework that leverages human activity recognition technologies based on motion signals, combined with advanced large language models, to conduct more detailed analyses and feedback of student behavior in physical education classes. Our framework begins with the teacher's instructional designs and the motion signals from students during physical education sessions, ultimately generating automated reports with teaching insights and suggestions for improving both learning and class instructions. This solution provides a motion signal-based approach for analyzing student behavior and optimizing instructional design tailored to physical education classes. Experimental results demonstrate that our framework can accurately identify student behaviors and produce meaningful pedagogical insights. 

---
# PerturboLLaVA: Reducing Multimodal Hallucinations with Perturbative Visual Training 

**Authors**: Cong Chen, Mingyu Liu, Chenchen Jing, Yizhou Zhou, Fengyun Rao, Hao Chen, Bo Zhang, Chunhua Shen  

**Link**: [PDF](https://arxiv.org/pdf/2503.06486)  

**Abstract**: This paper aims to address the challenge of hallucinations in Multimodal Large Language Models (MLLMs) particularly for dense image captioning tasks. To tackle the challenge, we identify the current lack of a metric that finely measures the caption quality in concept level. We hereby introduce HalFscore, a novel metric built upon the language graph and is designed to evaluate both the accuracy and completeness of dense captions at a granular level. Additionally, we identify the root cause of hallucination as the model's over-reliance on its language prior. To address this, we propose PerturboLLaVA, which reduces the model's reliance on the language prior by incorporating adversarially perturbed text during training. This method enhances the model's focus on visual inputs, effectively reducing hallucinations and producing accurate, image-grounded descriptions without incurring additional computational overhead. PerturboLLaVA significantly improves the fidelity of generated captions, outperforming existing approaches in handling multimodal hallucinations and achieving improved performance across general multimodal benchmarks. 

---
# ARMOR v0.1: Empowering Autoregressive Multimodal Understanding Model with Interleaved Multimodal Generation via Asymmetric Synergy 

**Authors**: Jianwen Sun, Yukang Feng, Chuanhao Li, Fanrui Zhang, Zizhen Li, Jiaxin Ai, Sizhuo Zhou, Yu Dai, Shenglin Zhang, Kaipeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.06542)  

**Abstract**: Unified models (UniMs) for multimodal understanding and generation have recently received much attention in the area of vision and language. Existing UniMs are designed to simultaneously learn both multimodal understanding and generation capabilities, demanding substantial computational resources, and often struggle to generate interleaved text-image. We present ARMOR, a resource-efficient and pure autoregressive framework that achieves both understanding and generation by fine-tuning existing multimodal large language models (MLLMs). Specifically, ARMOR extends existing MLLMs from three perspectives: (1) For model architecture, an asymmetric encoder-decoder architecture with a forward-switching mechanism is introduced to unify embedding space integrating textual and visual modalities for enabling natural text-image interleaved generation with minimal computational overhead. (2) For training data, a meticulously curated, high-quality interleaved dataset is collected for fine-tuning MLLMs. (3) For the training algorithm, we propose a ``what or how to generate" algorithm to empower existing MLLMs with multimodal generation capabilities while preserving their multimodal understanding capabilities, through three progressive training stages based on the collected dataset. Experimental results demonstrate that ARMOR upgrades existing MLLMs to UniMs with promising image generation capabilities, using limited training resources. Our code will be released soon at this https URL. 

---
# GenAI for Simulation Model in Model-Based Systems Engineering 

**Authors**: Lin Zhang, Yuteng Zhang, Dusit Niyato, Lei Ren, Pengfei Gu, Zhen Chen, Yuanjun Laili, Wentong Cai, Agostino Bruzzone  

**Link**: [PDF](https://arxiv.org/pdf/2503.06422)  

**Abstract**: Generative AI (GenAI) has demonstrated remarkable capabilities in code generation, and its integration into complex product modeling and simulation code generation can significantly enhance the efficiency of the system design phase in Model-Based Systems Engineering (MBSE). In this study, we introduce a generative system design methodology framework for MBSE, offering a practical approach for the intelligent generation of simulation models for system physical properties. First, we employ inference techniques, generative models, and integrated modeling and simulation languages to construct simulation models for system physical properties based on product design documents. Subsequently, we fine-tune the language model used for simulation model generation on an existing library of simulation models and additional datasets generated through generative modeling. Finally, we introduce evaluation metrics for the generated simulation models for system physical properties. Our proposed approach to simulation model generation presents the innovative concept of scalable templates for simulation models. Using these templates, GenAI generates simulation models for system physical properties through code completion. The experimental results demonstrate that, for mainstream open-source Transformer-based models, the quality of the simulation model is significantly improved using the simulation model generation method proposed in this paper. 

---
# Seesaw: High-throughput LLM Inference via Model Re-sharding 

**Authors**: Qidong Su, Wei Zhao, Xin Li, Muralidhar Andoorveedu, Chenhao Jiang, Zhanda Zhu, Kevin Song, Christina Giannoula, Gennady Pekhimenko  

**Link**: [PDF](https://arxiv.org/pdf/2503.06433)  

**Abstract**: To improve the efficiency of distributed large language model (LLM) inference, various parallelization strategies, such as tensor and pipeline parallelism, have been proposed. However, the distinct computational characteristics inherent in the two stages of LLM inference-prefilling and decoding-render a single static parallelization strategy insufficient for the effective optimization of both stages. In this work, we present Seesaw, an LLM inference engine optimized for throughput-oriented tasks. The key idea behind Seesaw is dynamic model re-sharding, a technique that facilitates the dynamic reconfiguration of parallelization strategies across stages, thereby maximizing throughput at both phases. To mitigate re-sharding overhead and optimize computational efficiency, we employ tiered KV cache buffering and transition-minimizing scheduling. These approaches work synergistically to reduce the overhead caused by frequent stage transitions while ensuring maximum batching efficiency. Our evaluation demonstrates that Seesaw achieves a throughput increase of up to 1.78x (1.36x on average) compared to vLLM, the most widely used state-of-the-art LLM inference engine. 

---
# Using Mechanistic Interpretability to Craft Adversarial Attacks against Large Language Models 

**Authors**: Thomas Winninger, Boussad Addad, Katarzyna Kapusta  

**Link**: [PDF](https://arxiv.org/pdf/2503.06269)  

**Abstract**: Traditional white-box methods for creating adversarial perturbations against LLMs typically rely only on gradient computation from the targeted model, ignoring the internal mechanisms responsible for attack success or failure. Conversely, interpretability studies that analyze these internal mechanisms lack practical applications beyond runtime interventions. We bridge this gap by introducing a novel white-box approach that leverages mechanistic interpretability techniques to craft practical adversarial inputs. Specifically, we first identify acceptance subspaces - sets of feature vectors that do not trigger the model's refusal mechanisms - then use gradient-based optimization to reroute embeddings from refusal subspaces to acceptance subspaces, effectively achieving jailbreaks. This targeted approach significantly reduces computation cost, achieving attack success rates of 80-95\% on state-of-the-art models including Gemma2, Llama3.2, and Qwen2.5 within minutes or even seconds, compared to existing techniques that often fail or require hours of computation. We believe this approach opens a new direction for both attack research and defense development. Furthermore, it showcases a practical application of mechanistic interpretability where other methods are less efficient, which highlights its utility. The code and generated datasets are available at this https URL. 

---
# From Captions to Rewards (CAREVL): Leveraging Large Language Model Experts for Enhanced Reward Modeling in Large Vision-Language Models 

**Authors**: Muzhi Dai, Jiashuo Sun, Zhiyuan Zhao, Shixuan Liu, Rui Li, Junyu Gao, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.06260)  

**Abstract**: Aligning large vision-language models (LVLMs) with human preferences is challenging due to the scarcity of fine-grained, high-quality, and multimodal preference data without human annotations. Existing methods relying on direct distillation often struggle with low-confidence data, leading to suboptimal performance. To address this, we propose CAREVL, a novel method for preference reward modeling by reliably using both high- and low-confidence data. First, a cluster of auxiliary expert models (textual reward models) innovatively leverages image captions as weak supervision signals to filter high-confidence data. The high-confidence data are then used to fine-tune the LVLM. Second, low-confidence data are used to generate diverse preference samples using the fine-tuned LVLM. These samples are then scored and selected to construct reliable chosen-rejected pairs for further training. CAREVL achieves performance improvements over traditional distillation-based methods on VL-RewardBench and MLLM-as-a-Judge benchmark, demonstrating its effectiveness. The code will be released soon. 

---
# Can Atomic Step Decomposition Enhance the Self-structured Reasoning of Multimodal Large Models? 

**Authors**: Kun Xiang, Zhili Liu, Zihao Jiang, Yunshuang Nie, Kaixin Cai, Yiyang Yin, Runhui Huang, Haoxiang Fan, Hanhui Li, Weiran Huang, Yihan Zeng, Yu-Jie Yuan, Jianhua Han, Lanqing Hong, Hang Xu, Xiaodan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2503.06252)  

**Abstract**: In this paper, we address the challenging task of multimodal mathematical reasoning by incorporating the ability of "slow thinking" into multimodal large language models (MLLMs). Our core idea is that different levels of reasoning abilities can be combined dynamically to tackle questions with different complexity. To this end, we propose a paradigm of Self-structured Chain of Thought (SCoT), which is composed of minimal semantic atomic steps. Different from existing methods that rely on structured templates or free-form paradigms, our method can not only generate cognitive CoT structures for various complex tasks but also mitigates the phenomenon of overthinking. To introduce structured reasoning capabilities into visual understanding models, we further design a novel AtomThink framework with four key modules, including (i) a data engine to generate high-quality multimodal reasoning paths; (ii) a supervised fine-tuning process with serialized inference data; (iii) a policy-guided multi-turn inference method; and (iv) an atomic capability metric to evaluate the single step utilization rate. We conduct extensive experiments to show that the proposed AtomThink significantly improves the performance of baseline MLLMs, achieving more than 10\% average accuracy gains on MathVista and MathVerse. Compared to state-of-the-art structured CoT approaches, our method not only achieves higher accuracy but also improves data utilization by 5 times and boosts inference efficiency by 85.3\%. Our code is now public available in this https URL. 

---
# Towards Universal Text-driven CT Image Segmentation 

**Authors**: Yuheng Li, Yuxiang Lai, Maria Thor, Deborah Marshall, Zachary Buchwald, David S. Yu, Xiaofeng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.06030)  

**Abstract**: Computed tomography (CT) is extensively used for accurate visualization and segmentation of organs and lesions. While deep learning models such as convolutional neural networks (CNNs) and vision transformers (ViTs) have significantly improved CT image analysis, their performance often declines when applied to diverse, real-world clinical data. Although foundation models offer a broader and more adaptable solution, their potential is limited due to the challenge of obtaining large-scale, voxel-level annotations for medical images. In response to these challenges, prompting-based models using visual or text prompts have emerged. Visual-prompting methods, such as the Segment Anything Model (SAM), still require significant manual input and can introduce ambiguity when applied to clinical scenarios. Instead, foundation models that use text prompts offer a more versatile and clinically relevant approach. Notably, current text-prompt models, such as the CLIP-Driven Universal Model, are limited to text prompts already encountered during training and struggle to process the complex and diverse scenarios of real-world clinical applications. Instead of fine-tuning models trained from natural imaging, we propose OpenVocabCT, a vision-language model pretrained on large-scale 3D CT images for universal text-driven segmentation. Using the large-scale CT-RATE dataset, we decompose the diagnostic reports into fine-grained, organ-level descriptions using large language models for multi-granular contrastive learning. We evaluate our OpenVocabCT on downstream segmentation tasks across nine public datasets for organ and tumor segmentation, demonstrating the superior performance of our model compared to existing methods. All code, datasets, and models will be publicly released at this https URL. 

---
# TPU-Gen: LLM-Driven Custom Tensor Processing Unit Generator 

**Authors**: Deepak Vungarala, Mohammed E. Elbtity, Sumiya Syed, Sakila Alam, Kartik Pandit, Arnob Ghosh, Ramtin Zand, Shaahin Angizi  

**Link**: [PDF](https://arxiv.org/pdf/2503.05951)  

**Abstract**: The increasing complexity and scale of Deep Neural Networks (DNNs) necessitate specialized tensor accelerators, such as Tensor Processing Units (TPUs), to meet various computational and energy efficiency requirements. Nevertheless, designing optimal TPU remains challenging due to the high domain expertise level, considerable manual design time, and lack of high-quality, domain-specific datasets. This paper introduces TPU-Gen, the first Large Language Model (LLM) based framework designed to automate the exact and approximate TPU generation process, focusing on systolic array architectures. TPU-Gen is supported with a meticulously curated, comprehensive, and open-source dataset that covers a wide range of spatial array designs and approximate multiply-and-accumulate units, enabling design reuse, adaptation, and customization for different DNN workloads. The proposed framework leverages Retrieval-Augmented Generation (RAG) as an effective solution for a data-scare hardware domain in building LLMs, addressing the most intriguing issue, hallucinations. TPU-Gen transforms high-level architectural specifications into optimized low-level implementations through an effective hardware generation pipeline. Our extensive experimental evaluations demonstrate superior performance, power, and area efficiency, with an average reduction in area and power of 92\% and 96\% from the manual optimization reference values. These results set new standards for driving advancements in next-generation design automation tools powered by LLMs. 

---
# Towards Understanding the Use of MLLM-Enabled Applications for Visual Interpretation by Blind and Low Vision People 

**Authors**: Ricardo E. Gonzalez Penuela, Ruiying Hu, Sharon Lin, Tanisha Shende, Shiri Azenkot  

**Link**: [PDF](https://arxiv.org/pdf/2503.05899)  

**Abstract**: Blind and Low Vision (BLV) people have adopted AI-powered visual interpretation applications to address their daily needs. While these applications have been helpful, prior work has found that users remain unsatisfied by their frequent errors. Recently, multimodal large language models (MLLMs) have been integrated into visual interpretation applications, and they show promise for more descriptive visual interpretations. However, it is still unknown how this advancement has changed people's use of these applications. To address this gap, we conducted a two-week diary study in which 20 BLV people used an MLLM-enabled visual interpretation application we developed, and we collected 553 entries. In this paper, we report a preliminary analysis of 60 diary entries from 6 participants. We found that participants considered the application's visual interpretations trustworthy (mean 3.75 out of 5) and satisfying (mean 4.15 out of 5). Moreover, participants trusted our application in high-stakes scenarios, such as receiving medical dosage advice. We discuss our plan to complete our analysis to inform the design of future MLLM-enabled visual interpretation systems. 

---
# Evaluating Large Language Models in Code Generation: INFINITE Methodology for Defining the Inference Index 

**Authors**: Nicholas Christakis, Dimitris Drikakis  

**Link**: [PDF](https://arxiv.org/pdf/2503.05852)  

**Abstract**: This study introduces a new methodology for an Inference Index (InI), called INFerence INdex In Testing model Effectiveness methodology (INFINITE), aiming to evaluate the performance of Large Language Models (LLMs) in code generation tasks. The InI index provides a comprehensive assessment focusing on three key components: efficiency, consistency, and accuracy. This approach encapsulates time-based efficiency, response quality, and the stability of model outputs, offering a thorough understanding of LLM performance beyond traditional accuracy metrics. We applied this methodology to compare OpenAI's GPT-4o (GPT), OpenAI-o1 pro (OAI1), and OpenAI-o3 mini-high (OAI3) in generating Python code for the Long-Short-Term-Memory (LSTM) model to forecast meteorological variables such as temperature, relative humidity and wind velocity. Our findings demonstrate that GPT outperforms OAI1 and performs comparably to OAI3 regarding accuracy and workflow efficiency. The study reveals that LLM-assisted code generation can produce results similar to expert-designed models with effective prompting and refinement. GPT's performance advantage highlights the benefits of widespread use and user feedback. 

---
# Encoding Inequity: Examining Demographic Bias in LLM-Driven Robot Caregiving 

**Authors**: Raj Korpan  

**Link**: [PDF](https://arxiv.org/pdf/2503.05765)  

**Abstract**: As robots take on caregiving roles, ensuring equitable and unbiased interactions with diverse populations is critical. Although Large Language Models (LLMs) serve as key components in shaping robotic behavior, speech, and decision-making, these models may encode and propagate societal biases, leading to disparities in care based on demographic factors. This paper examines how LLM-generated responses shape robot caregiving characteristics and responsibilities when prompted with different demographic information related to sex, gender, sexuality, race, ethnicity, nationality, disability, and age. Findings show simplified descriptions for disability and age, lower sentiment for disability and LGBTQ+ identities, and distinct clustering patterns reinforcing stereotypes in caregiving narratives. These results emphasize the need for ethical and inclusive HRI design. 

---
# The Lazy Student's Dream: ChatGPT Passing an Engineering Course on Its Own 

**Authors**: Gokul Puthumanaillam, Melkior Ornik  

**Link**: [PDF](https://arxiv.org/pdf/2503.05760)  

**Abstract**: This paper presents a comprehensive investigation into the capability of Large Language Models (LLMs) to successfully complete a semester-long undergraduate control systems course. Through evaluation of 115 course deliverables, we assess LLM performance using ChatGPT under a ``minimal effort" protocol that simulates realistic student usage patterns. The investigation employs a rigorous testing methodology across multiple assessment formats, from auto-graded multiple choice questions to complex Python programming tasks and long-form analytical writing. Our analysis provides quantitative insights into AI's strengths and limitations in handling mathematical formulations, coding challenges, and theoretical concepts in control systems engineering. The LLM achieved a B-grade performance (82.24\%), approaching but not exceeding the class average (84.99\%), with strongest results in structured assignments and greatest limitations in open-ended projects. The findings inform discussions about course design adaptation in response to AI advancement, moving beyond simple prohibition towards thoughtful integration of these tools in engineering education. Additional materials including syllabus, examination papers, design projects, and example responses can be found at the project website: this https URL. 

---
# Political Neutrality in AI is Impossible- But Here is How to Approximate it 

**Authors**: Jillian Fisher, Ruth E. Appel, Chan Young Park, Yujin Potter, Liwei Jiang, Taylor Sorensen, Shangbin Feng, Yulia Tsvetkov, Margaret E. Roberts, Jennifer Pan, Dawn Song, Yejin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2503.05728)  

**Abstract**: AI systems often exhibit political bias, influencing users' opinions and decision-making. While political neutrality-defined as the absence of bias-is often seen as an ideal solution for fairness and safety, this position paper argues that true political neutrality is neither feasible nor universally desirable due to its subjective nature and the biases inherent in AI training data, algorithms, and user interactions. However, inspired by Joseph Raz's philosophical insight that "neutrality [...] can be a matter of degree" (Raz, 1986), we argue that striving for some neutrality remains essential for promoting balanced AI interactions and mitigating user manipulation. Therefore, we use the term "approximation" of political neutrality to shift the focus from unattainable absolutes to achievable, practical proxies. We propose eight techniques for approximating neutrality across three levels of conceptualizing AI, examining their trade-offs and implementation strategies. In addition, we explore two concrete applications of these approximations to illustrate their practicality. Finally, we assess our framework on current large language models (LLMs) at the output level, providing a demonstration of how it can be evaluated. This work seeks to advance nuanced discussions of political neutrality in AI and promote the development of responsible, aligned language models. 

---
# Addressing Moral Uncertainty using Large Language Models for Ethical Decision-Making 

**Authors**: Rohit K. Dubey, Damian Dailisan, Sachit Mahajan  

**Link**: [PDF](https://arxiv.org/pdf/2503.05724)  

**Abstract**: We present an ethical decision-making framework that refines a pre-trained reinforcement learning (RL) model using a task-agnostic ethical layer. Following initial training, the RL model undergoes ethical fine-tuning, where human feedback is replaced by feedback generated from a large language model (LLM). The LLM embodies consequentialist, deontological, virtue, social justice, and care ethics as moral principles to assign belief values to recommended actions during ethical decision-making. An ethical layer aggregates belief scores from multiple LLM-derived moral perspectives using Belief Jensen-Shannon Divergence and Dempster-Shafer Theory into probability scores that also serve as the shaping reward, steering the agent toward choices that align with a balanced ethical framework. This integrated learning framework helps the RL agent navigate moral uncertainty in complex environments and enables it to make morally sound decisions across diverse tasks. Our approach, tested across different LLM variants and compared with other belief aggregation techniques, demonstrates improved consistency, adaptability, and reduced reliance on handcrafted ethical rewards. This method is especially effective in dynamic scenarios where ethical challenges arise unexpectedly, making it well-suited for real-world applications. 

---
# ORANSight-2.0: Foundational LLMs for O-RAN 

**Authors**: Pranshav Gajjar, Vijay K. Shah  

**Link**: [PDF](https://arxiv.org/pdf/2503.05200)  

**Abstract**: Despite the transformative impact of Large Language Models (LLMs) across critical domains such as healthcare, customer service, and business marketing, their integration into Open Radio Access Networks (O-RAN) remains limited. This gap is primarily due to the absence of domain-specific foundational models, with existing solutions often relying on general-purpose LLMs that fail to address the unique challenges and technical intricacies of O-RAN. To bridge this gap, we introduce ORANSight-2.0 (O-RAN Insights), a pioneering initiative aimed at developing specialized foundational LLMs tailored for O-RAN. Built on 18 LLMs spanning five open-source LLM frameworks, ORANSight-2.0 fine-tunes models ranging from 1 to 70B parameters, significantly reducing reliance on proprietary, closed-source models while enhancing performance for O-RAN. At the core of ORANSight-2.0 is RANSTRUCT, a novel Retrieval-Augmented Generation (RAG) based instruction-tuning framework that employs two LLM agents to create high-quality instruction-tuning datasets. The generated dataset is then used to fine-tune the 18 pre-trained open-source LLMs via QLoRA. To evaluate ORANSight-2.0, we introduce srsRANBench, a novel benchmark designed for code generation and codebase understanding in the context of srsRAN, a widely used 5G O-RAN stack. We also leverage ORANBench13K, an existing benchmark for assessing O-RAN-specific knowledge. Our comprehensive evaluations demonstrate that ORANSight-2.0 models outperform general-purpose and closed-source models, such as ChatGPT-4o and Gemini, by 5.421% on ORANBench and 18.465% on srsRANBench, achieving superior performance while maintaining lower computational and energy costs. We also experiment with RAG-augmented variants of ORANSight-2.0 LLMs and thoroughly evaluate their energy characteristics, demonstrating costs for training, standard inference, and RAG-augmented inference. 

---
# What I cannot execute, I do not understand: Training and Evaluating LLMs on Program Execution Traces 

**Authors**: Jordi Armengol-Estapé, Quentin Carbonneaux, Tianjun Zhang, Aram H. Markosyan, Volker Seeker, Chris Cummins, Melanie Kambadur, Michael F.P. O'Boyle, Sida Wang, Gabriel Synnaeve, Hugh James Leather  

**Link**: [PDF](https://arxiv.org/pdf/2503.05703)  

**Abstract**: Code generation and understanding are critical capabilities for large language models (LLMs). Thus, most LLMs are pretrained and fine-tuned on code data. However, these datasets typically treat code as static strings and rarely exploit the dynamic information about their execution. Building upon previous work on trace modeling, we study Execution Tuning (E.T.), a training procedure in which we explicitly model real-world program execution traces without requiring manual test annotations. We train and evaluate models on different execution trace granularities (line and instruction-level) and strategies on the task of output prediction, obtaining around 80% accuracy on CruxEval and MBPP, and showing the advantages of dynamic scratchpads (i.e., self-contained intermediate computations updated by the model rather than accumulated as a history of past computations) on long executions (up to 14k steps). Finally, we discuss E.T.'s practical applications. 

---
# AI Mimicry and Human Dignity: Chatbot Use as a Violation of Self-Respect 

**Authors**: Jan-Willem van der Rijt, Dimitri Coelho Mollo, Bram Vaassen  

**Link**: [PDF](https://arxiv.org/pdf/2503.05723)  

**Abstract**: This paper investigates how human interactions with AI-powered chatbots may offend human dignity. Current chatbots, driven by large language models (LLMs), mimic human linguistic behaviour but lack the moral and rational capacities essential for genuine interpersonal respect. Human beings are prone to anthropomorphise chatbots. Indeed, chatbots appear to be deliberately designed to elicit that response. As a result, human beings' behaviour toward chatbots often resembles behaviours typical of interaction between moral agents. Drawing on a second-personal, relational account of dignity, we argue that interacting with chatbots in this way is incompatible with the dignity of users. We show that, since second-personal respect is premised on reciprocal recognition of second-personal authority, behaving towards chatbots in ways that convey second-personal respect is bound to misfire in morally problematic ways, given the lack of reciprocity. Consequently, such chatbot interactions amount to subtle but significant violations of self-respect: the respect we are dutybound to show for our own dignity. We illustrate this by discussing four actual chatbot use cases (information retrieval, customer service, advising, and companionship), and propound that the increasing societal pressure to engage in such interactions with chatbots poses a hitherto underappreciated threat to human dignity. 

---
