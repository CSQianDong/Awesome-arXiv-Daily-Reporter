# SEAP: Training-free Sparse Expert Activation Pruning Unlock the Brainpower of Large Language Models 

**Authors**: Xun Liang, Hanyu Wang, Huayi Lai, Simin Niu, Shichao Song, Jiawei Yang, Jihao Zhao, Feiyu Xiong, Bo Tang, Zhiyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.07605)  

**Abstract**: Large Language Models have achieved remarkable success across various natural language processing tasks, yet their high computational cost during inference remains a major bottleneck. This paper introduces Sparse Expert Activation Pruning (SEAP), a training-free pruning method that selectively retains task-relevant parameters to reduce inference overhead. Inspired by the clustering patterns of hidden states and activations in LLMs, SEAP identifies task-specific expert activation patterns and prunes the model while preserving task performance and enhancing computational efficiency. Experimental results demonstrate that SEAP significantly reduces computational overhead while maintaining competitive accuracy. Notably, at 50% pruning, SEAP surpasses both WandA and FLAP by over 20%, and at 20% pruning, it incurs only a 2.2% performance drop compared to the dense model. These findings highlight SEAP's scalability and effectiveness, making it a promising approach for optimizing large-scale LLMs. 

---
# Implicit Reasoning in Transformers is Reasoning through Shortcuts 

**Authors**: Tianhe Lin, Jian Xie, Siyu Yuan, Deqing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.07604)  

**Abstract**: Test-time compute is emerging as a new paradigm for enhancing language models' complex multi-step reasoning capabilities, as demonstrated by the success of OpenAI's o1 and o3, as well as DeepSeek's R1. Compared to explicit reasoning in test-time compute, implicit reasoning is more inference-efficient, requiring fewer generated tokens. However, why does the advanced reasoning capability fail to emerge in the implicit reasoning style? In this work, we train GPT-2 from scratch on a curated multi-step mathematical reasoning dataset and conduct analytical experiments to investigate how language models perform implicit reasoning in multi-step tasks. Our findings reveal: 1) Language models can perform step-by-step reasoning and achieve high accuracy in both in-domain and out-of-domain tests via implicit reasoning. However, this capability only emerges when trained on fixed-pattern data. 2) Conversely, implicit reasoning abilities emerging from training on unfixed-pattern data tend to overfit a specific pattern and fail to generalize further. Notably, this limitation is also observed in state-of-the-art large language models. These findings suggest that language models acquire implicit reasoning through shortcut learning, enabling strong performance on tasks with similar patterns while lacking generalization. 

---
# Detection Avoidance Techniques for Large Language Models 

**Authors**: Sinclair Schneider, Florian Steuber, Joao A. G. Schneider, Gabi Dreo Rodosek  

**Link**: [PDF](https://arxiv.org/pdf/2503.07595)  

**Abstract**: The increasing popularity of large language models has not only led to widespread use but has also brought various risks, including the potential for systematically spreading fake news. Consequently, the development of classification systems such as DetectGPT has become vital. These detectors are vulnerable to evasion techniques, as demonstrated in an experimental series: Systematic changes of the generative models' temperature proofed shallow learning-detectors to be the least reliable. Fine-tuning the generative model via reinforcement learning circumvented BERT-based-detectors. Finally, rephrasing led to a >90\% evasion of zero-shot-detectors like DetectGPT, although texts stayed highly similar to the original. A comparison with existing work highlights the better performance of the presented methods. Possible implications for society and further research are discussed. 

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
# LMM-R1: Empowering 3B LMMs with Strong Reasoning Abilities Through Two-Stage Rule-Based RL 

**Authors**: Yingzhe Peng, Gongrui Zhang, Miaosen Zhang, Zhiyuan You, Jie Liu, Qipeng Zhu, Kai Yang, Xingzhong Xu, Xin Geng, Xu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.07536)  

**Abstract**: Enhancing reasoning in Large Multimodal Models (LMMs) faces unique challenges from the complex interplay between visual perception and logical reasoning, particularly in compact 3B-parameter architectures where architectural constraints limit reasoning capacity and modality alignment.
While rule-based reinforcement learning (RL) excels in text-only domains, its multimodal extension confronts two critical barriers: (1) data limitations due to ambiguous answers and scarce complex reasoning examples, and (2) degraded foundational reasoning induced by multimodal pretraining.
To address these challenges, we propose \textbf{\method}, a two-stage framework adapting rule-based RL for multimodal reasoning through \textbf{Foundational Reasoning Enhancement (FRE)} followed by \textbf{Multimodal Generalization Training (MGT)}. The FRE stage first strengthens reasoning abilities using text-only data with rule-based RL, then the MGT stage generalizes these reasoning capabilities to multimodal domains.
Experiments on Qwen2.5-VL-Instruct-3B demonstrate that \method achieves 4.83\% and 4.5\% average improvements over baselines in multimodal and text-only benchmarks, respectively, with a 3.63\% gain in complex Football Game tasks. These results validate that text-based reasoning enhancement enables effective multimodal generalization, offering a data-efficient paradigm that bypasses costly high-quality multimodal training data. 

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
# MedAgentsBench: Benchmarking Thinking Models and Agent Frameworks for Complex Medical Reasoning 

**Authors**: Xiangru Tang, Daniel Shao, Jiwoong Sohn, Jiapeng Chen, Jiayi Zhang, Jinyu Xiang, Fang Wu, Yilun Zhao, Chenglin Wu, Wenqi Shi, Arman Cohan, Mark Gerstein  

**Link**: [PDF](https://arxiv.org/pdf/2503.07459)  

**Abstract**: Large Language Models (LLMs) have shown impressive performance on existing medical question-answering benchmarks. This high performance makes it increasingly difficult to meaningfully evaluate and differentiate advanced methods. We present MedAgentsBench, a benchmark that focuses on challenging medical questions requiring multi-step clinical reasoning, diagnosis formulation, and treatment planning-scenarios where current models still struggle despite their strong performance on standard tests. Drawing from seven established medical datasets, our benchmark addresses three key limitations in existing evaluations: (1) the prevalence of straightforward questions where even base models achieve high performance, (2) inconsistent sampling and evaluation protocols across studies, and (3) lack of systematic analysis of the interplay between performance, cost, and inference time. Through experiments with various base models and reasoning methods, we demonstrate that the latest thinking models, DeepSeek R1 and OpenAI o3, exhibit exceptional performance in complex medical reasoning tasks. Additionally, advanced search-based agent methods offer promising performance-to-cost ratios compared to traditional approaches. Our analysis reveals substantial performance gaps between model families on complex questions and identifies optimal model selections for different computational constraints. Our benchmark and evaluation framework are publicly available at this https URL. 

---
# LLMs syntactically adapt their language use to their conversational partner 

**Authors**: Florian Kandra, Vera Demberg, Alexander Koller  

**Link**: [PDF](https://arxiv.org/pdf/2503.07457)  

**Abstract**: It has been frequently observed that human speakers align their language use with each other during conversations. In this paper, we study empirically whether large language models (LLMs) exhibit the same behavior of conversational adaptation. We construct a corpus of conversations between LLMs and find that two LLM agents end up making more similar syntactic choices as conversations go on, confirming that modern LLMs adapt their language use to their conversational partners in at least a rudimentary way. 

---
# Revisiting Noise in Natural Language Processing for Computational Social Science 

**Authors**: Nadav Borenstein  

**Link**: [PDF](https://arxiv.org/pdf/2503.07395)  

**Abstract**: Computational Social Science (CSS) is an emerging field driven by the unprecedented availability of human-generated content for researchers. This field, however, presents a unique set of challenges due to the nature of the theories and datasets it explores, including highly subjective tasks and complex, unstructured textual corpora. Among these challenges, one of the less well-studied topics is the pervasive presence of noise. This thesis aims to address this gap in the literature by presenting a series of interconnected case studies that examine different manifestations of noise in CSS. These include character-level errors following the OCR processing of historical records, archaic language, inconsistencies in annotations for subjective and ambiguous tasks, and even noise and biases introduced by large language models during content generation. This thesis challenges the conventional notion that noise in CSS is inherently harmful or useless. Rather, it argues that certain forms of noise can encode meaningful information that is invaluable for advancing CSS research, such as the unique communication styles of individuals or the culture-dependent nature of datasets and tasks. Further, this thesis highlights the importance of nuance in dealing with noise and the considerations CSS researchers must address when encountering it, demonstrating that different types of noise require distinct strategies. 

---
# Is My Text in Your AI Model? Gradient-based Membership Inference Test applied to LLMs 

**Authors**: Gonzalo Mancera, Daniel de Alcala, Julian Fierrez, Ruben Tolosana, Aythami Morales  

**Link**: [PDF](https://arxiv.org/pdf/2503.07384)  

**Abstract**: This work adapts and studies the gradient-based Membership Inference Test (gMINT) to the classification of text based on LLMs. MINT is a general approach intended to determine if given data was used for training machine learning models, and this work focuses on its application to the domain of Natural Language Processing. Using gradient-based analysis, the MINT model identifies whether particular data samples were included during the language model training phase, addressing growing concerns about data privacy in machine learning. The method was evaluated in seven Transformer-based models and six datasets comprising over 2.5 million sentences, focusing on text classification tasks. Experimental results demonstrate MINTs robustness, achieving AUC scores between 85% and 99%, depending on data size and model architecture. These findings highlight MINTs potential as a scalable and reliable tool for auditing machine learning models, ensuring transparency, safeguarding sensitive data, and fostering ethical compliance in the deployment of AI/NLP technologies. 

---
# RepoST: Scalable Repository-Level Coding Environment Construction with Sandbox Testing 

**Authors**: Yiqing Xie, Alex Xie, Divyanshu Sheth, Pengfei Liu, Daniel Fried, Carolyn Rose  

**Link**: [PDF](https://arxiv.org/pdf/2503.07358)  

**Abstract**: We present RepoST, a scalable method to construct environments that provide execution feedback for repository-level code generation for both training and evaluation. Unlike existing works that aim to build entire repositories for execution, which is challenging for both human and LLMs, we provide execution feedback with sandbox testing, which isolates a given target function and its dependencies to a separate script for testing. Sandbox testing reduces the complexity of external dependencies and enables constructing environments at a large scale. We use our method to construct RepoST-Train, a large-scale train set with 7,415 functions from 832 repositories. Training with the execution feedback provided by RepoST-Train leads to a performance gain of 5.5% Pass@1 on HumanEval and 3.5% Pass@1 on RepoEval. We also build an evaluation dataset, RepoST-Eval, and benchmark 12 code generation models. 

---
# Assessing the Macro and Micro Effects of Random Seeds on Fine-Tuning Large Language Models 

**Authors**: Hao Zhou, Guergana Savova, Lijing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.07329)  

**Abstract**: The impact of random seeds in fine-tuning large language models (LLMs) has been largely overlooked despite its potential influence on model this http URL this study, we systematically evaluate the effects of random seeds on LLMs using the GLUE and SuperGLUE benchmarks. We analyze the macro-level impact through traditional metrics like accuracy and F1, calculating their mean and variance to quantify performance fluctuations. To capture the micro-level effects, we introduce a novel metric, consistency, measuring the stability of individual predictions across runs. Our experiments reveal significant variance at both macro and micro levels, underscoring the need for careful consideration of random seeds in fine-tuning and evaluation. 

---
# Benchmarking Chinese Medical LLMs: A Medbench-based Analysis of Performance Gaps and Hierarchical Optimization Strategies 

**Authors**: Luyi Jiang, Jiayuan Chen, Lu Lu, Xinwei Peng, Lihao Liu, Junjun He, Jie Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.07306)  

**Abstract**: The evaluation and improvement of medical large language models (LLMs) are critical for their real-world deployment, particularly in ensuring accuracy, safety, and ethical alignment. Existing frameworks inadequately dissect domain-specific error patterns or address cross-modal challenges. This study introduces a granular error taxonomy through systematic analysis of top 10 models on MedBench, categorizing incorrect responses into eight types: Omissions, Hallucination, Format Mismatch, Causal Reasoning Deficiency, Contextual Inconsistency, Unanswered, Output Error, and Deficiency in Medical Language Generation. Evaluation of 10 leading models reveals vulnerabilities: despite achieving 0.86 accuracy in medical knowledge recall, critical reasoning tasks show 96.3% omission, while safety ethics evaluations expose alarming inconsistency (robustness score: 0.79) under option shuffled. Our analysis uncovers systemic weaknesses in knowledge boundary enforcement and multi-step reasoning. To address these, we propose a tiered optimization strategy spanning four levels, from prompt engineering and knowledge-augmented retrieval to hybrid neuro-symbolic architectures and causal reasoning frameworks. This work establishes an actionable roadmap for developing clinically robust LLMs while redefining evaluation paradigms through error-driven insights, ultimately advancing the safety and trustworthiness of AI in high-stakes medical environments. 

---
# An Information-Theoretic Approach to Identifying Formulaic Clusters in Textual Data 

**Authors**: Gideon Yoffe, Yair Segev, Barak Sober  

**Link**: [PDF](https://arxiv.org/pdf/2503.07303)  

**Abstract**: Texts, whether literary or historical, exhibit structural and stylistic patterns shaped by their purpose, authorship, and cultural context. Formulaic texts, characterized by repetition and constrained expression, tend to have lower variability in self-information compared to more dynamic compositions. Identifying such patterns in historical documents, particularly multi-author texts like the Hebrew Bible provides insights into their origins, purpose, and transmission.
This study aims to identify formulaic clusters -- sections exhibiting systematic repetition and structural constraints -- by analyzing recurring phrases, syntactic structures, and stylistic markers. However, distinguishing formulaic from non-formulaic elements in an unsupervised manner presents a computational challenge, especially in high-dimensional textual spaces where patterns must be inferred without predefined labels.
To address this, we develop an information-theoretic algorithm leveraging weighted self-information distributions to detect structured patterns in text, unlike covariance-based methods, which become unstable in small-sample, high-dimensional settings, our approach directly models variations in self-information to identify formulaicity. By extending classical discrete self-information measures with a continuous formulation based on differential self-information, our method remains applicable across different types of textual representations, including neural embeddings under Gaussian priors.
Applied to hypothesized authorial divisions in the Hebrew Bible, our approach successfully isolates stylistic layers, providing a quantitative framework for textual stratification. This method enhances our ability to analyze compositional patterns, offering deeper insights into the literary and cultural evolution of texts shaped by complex authorship and editorial processes. 

---
# A Graph-based Verification Framework for Fact-Checking 

**Authors**: Yani Huang, Richong Zhang, Zhijie Nie, Junfan Chen, Xuefeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.07282)  

**Abstract**: Fact-checking plays a crucial role in combating misinformation. Existing methods using large language models (LLMs) for claim decomposition face two key limitations: (1) insufficient decomposition, introducing unnecessary complexity to the verification process, and (2) ambiguity of mentions, leading to incorrect verification results. To address these challenges, we suggest introducing a claim graph consisting of triplets to address the insufficient decomposition problem and reduce mention ambiguity through graph structure. Based on this core idea, we propose a graph-based framework, GraphFC, for fact-checking. The framework features three key components: graph construction, which builds both claim and evidence graphs; graph-guided planning, which prioritizes the triplet verification order; and graph-guided checking, which verifies the triples one by one between claim and evidence graphs. Extensive experiments show that GraphFC enables fine-grained decomposition while resolving referential ambiguities through relational constraints, achieving state-of-the-art performance across three datasets. 

---
# SemEval-2025 Task 11: Bridging the Gap in Text-Based Emotion Detection 

**Authors**: Shamsuddeen Hassan Muhammad, Nedjma Ousidhoum, Idris Abdulmumin, Seid Muhie Yimam, Jan Philip Wahle, Terry Ruas, Meriem Beloucif, Christine De Kock, Tadesse Destaw Belay, Ibrahim Said Ahmad, Nirmal Surange, Daniela Teodorescu, David Ifeoluwa Adelani, Alham Fikri Aji, Felermino Ali, Vladimir Araujo, Abinew Ali Ayele, Oana Ignat, Alexander Panchenko, Yi Zhou, Saif M. Mohammad  

**Link**: [PDF](https://arxiv.org/pdf/2503.07269)  

**Abstract**: We present our shared task on text-based emotion detection, covering more than 30 languages from seven distinct language families. These languages are predominantly low-resource and spoken across various continents. The data instances are multi-labeled into six emotional classes, with additional datasets in 11 languages annotated for emotion intensity. Participants were asked to predict labels in three tracks: (a) emotion labels in monolingual settings, (b) emotion intensity scores, and (c) emotion labels in cross-lingual settings. The task attracted over 700 participants. We received final submissions from more than 200 teams and 93 system description papers. We report baseline results, as well as findings on the best-performing systems, the most common approaches, and the most effective methods across various tracks and languages. The datasets for this task are publicly available. 

---
# LLM-C3MOD: A Human-LLM Collaborative System for Cross-Cultural Hate Speech Moderation 

**Authors**: Junyeong Park, Seogyeong Jeong, Seyoung Song, Yohan Lee, Alice Oh  

**Link**: [PDF](https://arxiv.org/pdf/2503.07237)  

**Abstract**: Content moderation is a global challenge, yet major tech platforms prioritize high-resource languages, leaving low-resource languages with scarce native moderators. Since effective moderation depends on understanding contextual cues, this imbalance increases the risk of improper moderation due to non-native moderators' limited cultural understanding. Through a user study, we identify that non-native moderators struggle with interpreting culturally-specific knowledge, sentiment, and internet culture in the hate speech moderation. To assist them, we present LLM-C3MOD, a human-LLM collaborative pipeline with three steps: (1) RAG-enhanced cultural context annotations; (2) initial LLM-based moderation; and (3) targeted human moderation for cases lacking LLM consensus. Evaluated on a Korean hate speech dataset with Indonesian and German participants, our system achieves 78% accuracy (surpassing GPT-4o's 71% baseline), while reducing human workload by 83.6%. Notably, human moderators excel at nuanced contents where LLMs struggle. Our findings suggest that non-native moderators, when properly supported by LLMs, can effectively contribute to cross-cultural hate speech moderation. 

---
# Cross-Lingual IPA Contrastive Learning for Zero-Shot NER 

**Authors**: Jimin Sohn, David R. Mortensen  

**Link**: [PDF](https://arxiv.org/pdf/2503.07214)  

**Abstract**: Existing approaches to zero-shot Named Entity Recognition (NER) for low-resource languages have primarily relied on machine translation, whereas more recent methods have shifted focus to phonemic representation. Building upon this, we investigate how reducing the phonemic representation gap in IPA transcription between languages with similar phonetic characteristics enables models trained on high-resource languages to perform effectively on low-resource languages. In this work, we propose CONtrastive Learning with IPA (CONLIPA) dataset containing 10 English and high resource languages IPA pairs from 10 frequently used language families. We also propose a cross-lingual IPA Contrastive learning method (IPAC) using the CONLIPA dataset. Furthermore, our proposed dataset and methodology demonstrate a substantial average gain when compared to the best performing baseline. 

---
# Contextual Cues in Machine Translation: Investigating the Potential of Multi-Source Input Strategies in LLMs and NMT Systems 

**Authors**: Lia Shahnazaryan, Patrick Simianer, Joern Wuebker  

**Link**: [PDF](https://arxiv.org/pdf/2503.07195)  

**Abstract**: We explore the impact of multi-source input strategies on machine translation (MT) quality, comparing GPT-4o, a large language model (LLM), with a traditional multilingual neural machine translation (NMT) system. Using intermediate language translations as contextual cues, we evaluate their effectiveness in enhancing English and Chinese translations into Portuguese. Results suggest that contextual information significantly improves translation quality for domain-specific datasets and potentially for linguistically distant language pairs, with diminishing returns observed in benchmarks with high linguistic variability. Additionally, we demonstrate that shallow fusion, a multi-source approach we apply within the NMT system, shows improved results when using high-resource languages as context for other translation pairs, highlighting the importance of strategic context language selection. 

---
# Strategies for political-statement segmentation and labelling in unstructured text 

**Authors**: Dmitry Nikolaev, Sean Papay  

**Link**: [PDF](https://arxiv.org/pdf/2503.07179)  

**Abstract**: Analysis of parliamentary speeches and political-party manifestos has become an integral area of computational study of political texts. While speeches have been overwhelmingly analysed using unsupervised methods, a large corpus of manifestos with by-statement political-stance labels has been created by the participants of the MARPOR project. It has been recently shown that these labels can be predicted by a neural model; however, the current approach relies on provided statement boundaries, limiting out-of-domain applicability. In this work, we propose and test a range of unified split-and-label frameworks -- based on linear-chain CRFs, fine-tuned text-to-text models, and the combination of in-context learning with constrained decoding -- that can be used to jointly segment and classify statements from raw textual data. We show that our approaches achieve competitive accuracy when applied to raw text of political manifestos, and then demonstrate the research potential of our method by applying it to the records of the UK House of Commons and tracing the political trajectories of four major parties in the last three decades. 

---
# DeFine: A Decomposed and Fine-Grained Annotated Dataset for Long-form Article Generation 

**Authors**: Ming Wang, Fang Wang, Minghao Hu, Li He, Haiyang Wang, Jun Zhang, Tianwei Yan, Li Li, Zhunchen Luo, Wei Luo, Xiaoying Bai, Guotong Geng  

**Link**: [PDF](https://arxiv.org/pdf/2503.07170)  

**Abstract**: Long-form article generation (LFAG) presents challenges such as maintaining logical consistency, comprehensive topic coverage, and narrative coherence across extended articles. Existing datasets often lack both the hierarchical structure and fine-grained annotation needed to effectively decompose tasks, resulting in shallow, disorganized article generation. To address these limitations, we introduce DeFine, a Decomposed and Fine-grained annotated dataset for long-form article generation. DeFine is characterized by its hierarchical decomposition strategy and the integration of domain-specific knowledge with multi-level annotations, ensuring granular control and enhanced depth in article generation. To construct the dataset, a multi-agent collaborative pipeline is proposed, which systematically segments the generation process into four parts: Data Miner, Cite Retreiver, Q&A Annotator and Data Cleaner. To validate the effectiveness of DeFine, we designed and tested three LFAG baselines: the web retrieval, the local retrieval, and the grounded reference. We fine-tuned the Qwen2-7b-Instruct model using the DeFine training dataset. The experimental results showed significant improvements in text quality, specifically in topic coverage, depth of information, and content fidelity. Our dataset publicly available to facilitate future research. 

---
# MRCEval: A Comprehensive, Challenging and Accessible Machine Reading Comprehension Benchmark 

**Authors**: Shengkun Ma, Hao Peng, Lei Hou, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.07144)  

**Abstract**: Machine Reading Comprehension (MRC) is an essential task in evaluating natural language understanding. Existing MRC datasets primarily assess specific aspects of reading comprehension (RC), lacking a comprehensive MRC benchmark. To fill this gap, we first introduce a novel taxonomy that categorizes the key capabilities required for RC. Based on this taxonomy, we construct MRCEval, an MRC benchmark that leverages advanced Large Language Models (LLMs) as both sample generators and selection judges. MRCEval is a comprehensive, challenging and accessible benchmark designed to assess the RC capabilities of LLMs thoroughly, covering 13 distinct RC skills with a total of 2.1K high-quality multi-choice questions. We perform an extensive evaluation of 28 widely used open-source and proprietary models, highlighting that MRC continues to present significant challenges even in the era of LLMs. 

---
# A Systematic Comparison of Syntactic Representations of Dependency Parsing 

**Authors**: Guillaume Wisniewski, Ophélie Lacroix  

**Link**: [PDF](https://arxiv.org/pdf/2503.07142)  

**Abstract**: We compare the performance of a transition-based parser in regards to different annotation schemes. We pro-pose to convert some specific syntactic constructions observed in the universal dependency treebanks into a so-called more standard representation and to evaluate parsing performances over all the languages of the project. We show that the ``standard'' constructions do not lead systematically to better parsing performance and that the scores vary considerably according to the languages. 

---
# Application of Multiple Chain-of-Thought in Contrastive Reasoning for Implicit Sentiment Analysis 

**Authors**: Liwei Yang, Xinying Wang, Xiaotang Zhou, Zhengchao Wu, Ningning Tan  

**Link**: [PDF](https://arxiv.org/pdf/2503.07140)  

**Abstract**: Implicit sentiment analysis aims to uncover emotions that are subtly expressed, often obscured by ambiguity and figurative language. To accomplish this task, large language models and multi-step reasoning are needed to identify those sentiments that are not explicitly stated. In this study, we propose a novel Dual Reverse Chain Reasoning (DRCR) framework to enhance the performance of implicit sentiment analysis. Inspired by deductive reasoning, the framework consists of three key steps: 1) hypothesize an emotional polarity and derive a reasoning process, 2) negate the initial hypothesis and derive a new reasoning process, and 3) contrast the two reasoning paths to deduce the final sentiment polarity. Building on this, we also introduce a Triple Reverse Chain Reasoning (TRCR) framework to address the limitations of random hypotheses. Both methods combine contrastive mechanisms and multi-step reasoning, significantly improving the accuracy of implicit sentiment classification. Experimental results demonstrate that both approaches outperform existing methods across various model scales, achieving state-of-the-art performance. This validates the effectiveness of combining contrastive reasoning and multi-step reasoning for implicit sentiment analysis. 

---
# ASTRA: A Negotiation Agent with Adaptive and Strategic Reasoning through Action in Dynamic Offer Optimization 

**Authors**: Deuksin Kwon, Jiwon Hae, Emma Clift, Daniel Shamsoddini, Jonathan Gratch, Gale M. Lucas  

**Link**: [PDF](https://arxiv.org/pdf/2503.07129)  

**Abstract**: Negotiation requires dynamically balancing self-interest and cooperation to maximize one's own utility. Yet, existing agents struggle due to bounded rationality in human data, low adaptability to counterpart behavior, and limited strategic reasoning. To address this, we introduce principle-driven negotiation agents, powered by ASTRA, a novel framework for turn-level offer optimization grounded in two core principles: opponent modeling and Tit-for-Tat reciprocity. ASTRA operates in three stages: (1) interpreting counterpart behavior, (2) optimizing counteroffers via a linear programming (LP) solver, and (3) selecting offers based on negotiation tactics and the partner's acceptance probability. Through simulations and human evaluations, our agent effectively adapts to an opponent's shifting stance and achieves favorable outcomes through enhanced adaptability and strategic reasoning. Beyond improving negotiation performance, it also serves as a powerful coaching tool, offering interpretable strategic feedback and optimal offer recommendations. 

---
# A Novel Ophthalmic Benchmark for Evaluating Multimodal Large Language Models with Fundus Photographs and OCT Images 

**Authors**: Xiaoyi Liang, Mouxiao Bian, Moxin Chen, Lihao Liu, Junjun He, Jie Xu, Lin Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.07094)  

**Abstract**: In recent years, large language models (LLMs) have demonstrated remarkable potential across various medical applications. Building on this foundation, multimodal large language models (MLLMs) integrate LLMs with visual models to process diverse inputs, including clinical data and medical images. In ophthalmology, LLMs have been explored for analyzing optical coherence tomography (OCT) reports, assisting in disease classification, and even predicting treatment outcomes. However, existing MLLM benchmarks often fail to capture the complexities of real-world clinical practice, particularly in the analysis of OCT images. Many suffer from limitations such as small sample sizes, a lack of diverse OCT datasets, and insufficient expert validation. These shortcomings hinder the accurate assessment of MLLMs' ability to interpret OCT scans and their broader applicability in ophthalmology. Our dataset, curated through rigorous quality control and expert annotation, consists of 439 fundus images and 75 OCT images. Using a standardized API-based framework, we assessed seven mainstream MLLMs and observed significant variability in diagnostic accuracy across different diseases. While some models performed well in diagnosing conditions such as diabetic retinopathy and age-related macular degeneration, they struggled with others, including choroidal neovascularization and myopia, highlighting inconsistencies in performance and the need for further refinement. Our findings emphasize the importance of developing clinically relevant benchmarks to provide a more accurate assessment of MLLMs' capabilities. By refining these models and expanding their scope, we can enhance their potential to transform ophthalmic diagnosis and treatment. 

---
# Linguistic Knowledge Transfer Learning for Speech Enhancement 

**Authors**: Kuo-Hsuan Hung, Xugang Lu, Szu-Wei Fu, Huan-Hsin Tseng, Hsin-Yi Lin, Chii-Wann Lin, Yu Tsao  

**Link**: [PDF](https://arxiv.org/pdf/2503.07078)  

**Abstract**: Linguistic knowledge plays a crucial role in spoken language comprehension. It provides essential semantic and syntactic context for speech perception in noisy environments. However, most speech enhancement (SE) methods predominantly rely on acoustic features to learn the mapping relationship between noisy and clean speech, with limited exploration of linguistic integration. While text-informed SE approaches have been investigated, they often require explicit speech-text alignment or externally provided textual data, constraining their practicality in real-world scenarios. Additionally, using text as input poses challenges in aligning linguistic and acoustic representations due to their inherent differences. In this study, we propose the Cross-Modality Knowledge Transfer (CMKT) learning framework, which leverages pre-trained large language models (LLMs) to infuse linguistic knowledge into SE models without requiring text input or LLMs during inference. Furthermore, we introduce a misalignment strategy to improve knowledge transfer. This strategy applies controlled temporal shifts, encouraging the model to learn more robust representations. Experimental evaluations demonstrate that CMKT consistently outperforms baseline models across various SE architectures and LLM embeddings, highlighting its adaptability to different configurations. Additionally, results on Mandarin and English datasets confirm its effectiveness across diverse linguistic conditions, further validating its robustness. Moreover, CMKT remains effective even in scenarios without textual data, underscoring its practicality for real-world applications. By bridging the gap between linguistic and acoustic modalities, CMKT offers a scalable and innovative solution for integrating linguistic knowledge into SE models, leading to substantial improvements in both intelligibility and enhancement performance. 

---
# DistiLLM-2: A Contrastive Approach Boosts the Distillation of LLMs 

**Authors**: Jongwoo Ko, Tianyi Chen, Sungnyun Kim, Tianyu Ding, Luming Liang, Ilya Zharkov, Se-Young Yun  

**Link**: [PDF](https://arxiv.org/pdf/2503.07067)  

**Abstract**: Despite the success of distillation in large language models (LLMs), most prior work applies identical loss functions to both teacher- and student-generated data. These strategies overlook the synergy between loss formulations and data types, leading to a suboptimal performance boost in student models. To address this, we propose DistiLLM-2, a contrastive approach that simultaneously increases the likelihood of teacher responses and decreases that of student responses by harnessing this synergy. Our extensive experiments show that DistiLLM-2 not only builds high-performing student models across a wide range of tasks, including instruction-following and code generation, but also supports diverse applications, such as preference alignment and vision-language extensions. These findings highlight the potential of a contrastive approach to enhance the efficacy of LLM distillation by effectively aligning teacher and student models across varied data types. 

---
# DatawiseAgent: A Notebook-Centric LLM Agent Framework for Automated Data Science 

**Authors**: Ziming You, Yumiao Zhang, Dexuan Xu, Yiwei Lou, Yandong Yan, Wei Wang, Huaming Zhang, Yu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.07044)  

**Abstract**: Data Science tasks are multifaceted, dynamic, and often domain-specific. Existing LLM-based approaches largely concentrate on isolated phases, neglecting the interdependent nature of many data science tasks and limiting their capacity for comprehensive end-to-end support. We propose DatawiseAgent, a notebook-centric LLM agent framework that unifies interactions among user, agent and the computational environment through markdown and executable code cells, supporting flexible and adaptive automated data science. Built on a Finite State Transducer(FST), DatawiseAgent orchestrates four stages, including DSF-like planning, incremental execution, self-debugging, and post-filtering. Specifically, the DFS-like planning stage systematically explores the solution space, while incremental execution harnesses real-time feedback and accommodates LLM's limited capabilities to progressively complete tasks. The self-debugging and post-filtering modules further enhance reliability by diagnosing and correcting errors and pruning extraneous information. Extensive experiments on diverse tasks, including data analysis, visualization, and data modeling, show that DatawiseAgent consistently outperforms or matches state-of-the-art methods across multiple model settings. These results highlight its potential to generalize across data science scenarios and lay the groundwork for more efficient, fully automated workflows. 

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
# Multimodal Human-AI Synergy for Medical Imaging Quality Control: A Hybrid Intelligence Framework with Adaptive Dataset Curation and Closed-Loop Evaluation 

**Authors**: Zhi Qin, Qianhui Gui, Mouxiao Bian, Rui Wang, Hong Ge, Dandan Yao, Ziying Sun, Yuan Zhao, Yu Zhang, Hui Shi, Dongdong Wang, Chenxin Song, Shenghong Ju, Lihao Liu, Junjun He, Jie Xu, Yuan-Cheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.07032)  

**Abstract**: Medical imaging quality control (QC) is essential for accurate diagnosis, yet traditional QC methods remain labor-intensive and subjective. To address this challenge, in this study, we establish a standardized dataset and evaluation framework for medical imaging QC, systematically assessing large language models (LLMs) in image quality assessment and report standardization. Specifically, we first constructed and anonymized a dataset of 161 chest X-ray (CXR) radiographs and 219 CT reports for evaluation. Then, multiple LLMs, including Gemini 2.0-Flash, GPT-4o, and DeepSeek-R1, were evaluated based on recall, precision, and F1 score to detect technical errors and inconsistencies. Experimental results show that Gemini 2.0-Flash achieved a Macro F1 score of 90 in CXR tasks, demonstrating strong generalization but limited fine-grained performance. DeepSeek-R1 excelled in CT report auditing with a 62.23\% recall rate, outperforming other models. However, its distilled variants performed poorly, while InternLM2.5-7B-chat exhibited the highest additional discovery rate, indicating broader but less precise error detection. These findings highlight the potential of LLMs in medical imaging QC, with DeepSeek-R1 and Gemini 2.0-Flash demonstrating superior performance. 

---
# Toward Multi-Session Personalized Conversation: A Large-Scale Dataset and Hierarchical Tree Framework for Implicit Reasoning 

**Authors**: Xintong Li, Jalend Bantupalli, Ria Dharmani, Yuwei Zhang, Jingbo Shang  

**Link**: [PDF](https://arxiv.org/pdf/2503.07018)  

**Abstract**: There has been a surge in the use of large language models (LLM) conversational agents to generate responses based on long-term history from multiple sessions. However, existing long-term open-domain dialogue datasets lack complex, real-world personalization and fail to capture implicit reasoning-where relevant information is embedded in subtle, syntactic, or semantically distant connections rather than explicit statements. In such cases, traditional retrieval methods fail to capture relevant context, and long-context modeling also becomes inefficient due to numerous complicated persona-related details. To address this gap, we introduce ImplexConv, a large-scale long-term dataset with 2,500 examples, each containing approximately 100 conversation sessions, designed to study implicit reasoning in personalized dialogues. Additionally, we propose TaciTree, a novel hierarchical tree framework that structures conversation history into multiple levels of summarization. Instead of brute-force searching all data, TaciTree enables an efficient, level-based retrieval process where models refine their search by progressively selecting relevant details. Our experiments demonstrate that TaciTree significantly improves the ability of LLMs to reason over long-term conversations with implicit contextual dependencies. 

---
# Large Language Models Often Say One Thing and Do Another 

**Authors**: Ruoxi Xu, Hongyu Lin, Xianpei Han, Jia Zheng, Weixiang Zhou, Le Sun, Yingfei Sun  

**Link**: [PDF](https://arxiv.org/pdf/2503.07003)  

**Abstract**: As large language models (LLMs) increasingly become central to various applications and interact with diverse user populations, ensuring their reliable and consistent performance is becoming more important. This paper explores a critical issue in assessing the reliability of LLMs: the consistency between their words and deeds. To quantitatively explore this consistency, we developed a novel evaluation benchmark called the Words and Deeds Consistency Test (WDCT). The benchmark establishes a strict correspondence between word-based and deed-based questions across different domains, including opinion vs. action, non-ethical value vs. action, ethical value vs. action, and theory vs. application. The evaluation results reveal a widespread inconsistency between words and deeds across different LLMs and domains. Subsequently, we conducted experiments with either word alignment or deed alignment to observe their impact on the other aspect. The experimental results indicate that alignment only on words or deeds poorly and unpredictably influences the other aspect. This supports our hypothesis that the underlying knowledge guiding LLMs' word or deed choices is not contained within a unified space. 

---
# Social Bias Benchmark for Generation: A Comparison of Generation and QA-Based Evaluations 

**Authors**: Jiho Jin, Woosung Kang, Junho Myung, Alice Oh  

**Link**: [PDF](https://arxiv.org/pdf/2503.06987)  

**Abstract**: Measuring social bias in large language models (LLMs) is crucial, but existing bias evaluation methods struggle to assess bias in long-form generation. We propose a Bias Benchmark for Generation (BBG), an adaptation of the Bias Benchmark for QA (BBQ), designed to evaluate social bias in long-form generation by having LLMs generate continuations of story prompts. Building our benchmark in English and Korean, we measure the probability of neutral and biased generations across ten LLMs. We also compare our long-form story generation evaluation results with multiple-choice BBQ evaluation, showing that the two approaches produce inconsistent results. 

---
# Exploring Multimodal Perception in Large Language Models Through Perceptual Strength Ratings 

**Authors**: Jonghyun Lee, Dojun Park, Jiwoo Lee, Hoekeon Choi, Sung-Eun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.06980)  

**Abstract**: This study investigated the multimodal perception of large language models (LLMs), focusing on their ability to capture human-like perceptual strength ratings across sensory modalities. Utilizing perceptual strength ratings as a benchmark, the research compared GPT-3.5, GPT-4, GPT-4o, and GPT-4o-mini, highlighting the influence of multimodal inputs on grounding and linguistic reasoning. While GPT-4 and GPT-4o demonstrated strong alignment with human evaluations and significant advancements over smaller models, qualitative analyses revealed distinct differences in processing patterns, such as multisensory overrating and reliance on loose semantic associations. Despite integrating multimodal capabilities, GPT-4o did not exhibit superior grounding compared to GPT-4, raising questions about their role in improving human-like grounding. These findings underscore how LLMs' reliance on linguistic patterns can both approximate and diverge from human embodied cognition, revealing limitations in replicating sensory experiences. 

---
# CtrlRAG: Black-box Adversarial Attacks Based on Masked Language Models in Retrieval-Augmented Language Generation 

**Authors**: Runqi Sui  

**Link**: [PDF](https://arxiv.org/pdf/2503.06950)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems enhance Large Language Models (LLMs) by integrating external knowledge bases. However, this integration introduces a new security threat: adversaries can exploit the retrieval mechanism to inject malicious content into the knowledge base, thereby influencing the generated responses. Based on this attack vector, we propose CtrlRAG, a novel attack method designed for RAG system in the black-box setting, which aligns with real-world scenarios. Unlike existing attack methods, CtrlRAG introduces a perturbation mechanism using Masked Language Model (MLM) to dynamically optimize malicious content in response to changes in the retrieved context. Experimental results demonstrate that CtrlRAG outperforms three baseline methods in both Emotional Manipulation and Hallucination Amplification objectives. Furthermore, we evaluate three existing defense mechanisms, revealing their limited effectiveness against CtrlRAG and underscoring the urgent need for more robust defenses. 

---
# Lshan-1.0 Technical Report 

**Authors**: Haotian Chen, Yanyu Xu, Boyan Wang, Chaoyue Zhao, Xiaoyu Han, Fang Wang, Lizhen Cui, Yonghui Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.06949)  

**Abstract**: In this report, we introduce our first-generation reasoning model, Lshan-1.0, a large language model designed for the highly specialized Chinese legal domain, offering comprehensive capabilities to meet diverse realistic needs. Existing legal LLMs face two primary challenges. Firstly, their design and evaluation are predominantly driven by computer science perspectives, leading to insufficient incorporation of legal expertise and logic, which is crucial for high-precision legal applications, such as handling complex prosecutorial tasks. Secondly, these models often underperform due to a lack of comprehensive training data from the legal domain, limiting their ability to effectively address real-world legal scenarios. To address this, we first compile millions of legal documents covering over 20 types of crimes from 31 provinces in China for model training. From the extensive dataset, we further select high-quality for supervised fine-tuning, ensuring enhanced relevance and precision. The model further undergoes large-scale reinforcement learning without additional supervision, emphasizing the enhancement of its reasoning capabilities and explainability. To validate its effectiveness in complex legal applications, we also conduct human evaluations with legal experts. We develop fine-tuned models based on DeepSeek-R1-Distilled versions, available in three dense configurations: 14B, 32B, and 70B. 

---
# Effect of Selection Format on LLM Performance 

**Authors**: Yuchen Han, Yucheng Wu, Jeffrey Willard  

**Link**: [PDF](https://arxiv.org/pdf/2503.06926)  

**Abstract**: This paper investigates a critical aspect of large language model (LLM) performance: the optimal formatting of classification task options in prompts. Through an extensive experimental study, we compared two selection formats -- bullet points and plain English -- to determine their impact on model performance. Our findings suggest that presenting options via bullet points generally yields better results, although there are some exceptions. Furthermore, our research highlights the need for continued exploration of option formatting to drive further improvements in model performance. 

---
# Automatic Speech Recognition for Non-Native English: Accuracy and Disfluency Handling 

**Authors**: Michael McGuire  

**Link**: [PDF](https://arxiv.org/pdf/2503.06924)  

**Abstract**: Automatic speech recognition (ASR) has been an essential component of computer assisted language learning (CALL) and computer assisted language testing (CALT) for many years. As this technology continues to develop rapidly, it is important to evaluate the accuracy of current ASR systems for language learning applications. This study assesses five cutting-edge ASR systems' recognition of non-native accented English speech using recordings from the L2-ARCTIC corpus, featuring speakers from six different L1 backgrounds (Arabic, Chinese, Hindi, Korean, Spanish, and Vietnamese), in the form of both read and spontaneous speech. The read speech consisted of 2,400 single sentence recordings from 24 speakers, while the spontaneous speech included narrative recordings from 22 speakers. Results showed that for read speech, Whisper and AssemblyAI achieved the best accuracy with mean Match Error Rates (MER) of 0.054 and 0.056 respectively, approaching human-level accuracy. For spontaneous speech, RevAI performed best with a mean MER of 0.063. The study also examined how each system handled disfluencies such as filler words, repetitions, and revisions, finding significant variation in performance across systems and disfluency types. While processing speed varied considerably between systems, longer processing times did not necessarily correlate with better accuracy. By detailing the performance of several of the most recent, widely-available ASR systems on non-native English speech, this study aims to help language instructors and researchers understand the strengths and weaknesses of each system and identify which may be suitable for specific use cases. 

---
# KwaiChat: A Large-Scale Video-Driven Multilingual Mixed-Type Dialogue Corpus 

**Authors**: Xiaoming Shi, Zeming Liu, Yiming Lei, Chenkai Zhang, Haitao Leng, Chuan Wang, Qingjie Liu, Wanxiang Che, Shaoguo Liu, Size Li, Yunhong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.06899)  

**Abstract**: Video-based dialogue systems, such as education assistants, have compelling application value, thereby garnering growing interest. However, the current video-based dialogue systems are limited by their reliance on a single dialogue type, which hinders their versatility in practical applications across a range of scenarios, including question-answering, emotional dialog, etc. In this paper, we identify this challenge as how to generate video-driven multilingual mixed-type dialogues. To mitigate this challenge, we propose a novel task and create a human-to-human video-driven multilingual mixed-type dialogue corpus, termed KwaiChat, containing a total of 93,209 videos and 246,080 dialogues, across 4 dialogue types, 30 domains, 4 languages, and 13 topics. Additionally, we establish baseline models on KwaiChat. An extensive analysis of 7 distinct LLMs on KwaiChat reveals that GPT-4o achieves the best performance but still cannot perform well in this situation even with the help of in-context learning and fine-tuning, which indicates that the task is not trivial and needs further research. 

---
# A LongFormer-Based Framework for Accurate and Efficient Medical Text Summarization 

**Authors**: Dan Sun, Jacky He, Hanlu Zhang, Zhen Qi, Hongye Zheng, Xiaokai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.06888)  

**Abstract**: This paper proposes a medical text summarization method based on LongFormer, aimed at addressing the challenges faced by existing models when processing long medical texts. Traditional summarization methods are often limited by short-term memory, leading to information loss or reduced summary quality in long texts. LongFormer, by introducing long-range self-attention, effectively captures long-range dependencies in the text, retaining more key information and improving the accuracy and information retention of summaries. Experimental results show that the LongFormer-based model outperforms traditional models, such as RNN, T5, and BERT in automatic evaluation metrics like ROUGE. It also receives high scores in expert evaluations, particularly excelling in information retention and grammatical accuracy. However, there is still room for improvement in terms of conciseness and readability. Some experts noted that the generated summaries contain redundant information, which affects conciseness. Future research will focus on further optimizing the model structure to enhance conciseness and fluency, achieving more efficient medical text summarization. As medical data continues to grow, automated summarization technology will play an increasingly important role in fields such as medical research, clinical decision support, and knowledge management. 

---
# Lost-in-the-Middle in Long-Text Generation: Synthetic Dataset, Evaluation Framework, and Mitigation 

**Authors**: Junhao Zhang, Richong Zhang, Fanshuang Kong, Ziyang Miao, Yanhan Ye, Yaowei Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.06868)  

**Abstract**: Existing long-text generation methods primarily concentrate on producing lengthy texts from short inputs, neglecting the long-input and long-output tasks. Such tasks have numerous practical applications while lacking available benchmarks. Moreover, as the input grows in length, existing methods inevitably encounter the "lost-in-the-middle" phenomenon. In this paper, we first introduce a Long Input and Output Benchmark (LongInOutBench), including a synthetic dataset and a comprehensive evaluation framework, addressing the challenge of the missing benchmark. We then develop the Retrieval-Augmented Long-Text Writer (RAL-Writer), which retrieves and restates important yet overlooked content, mitigating the "lost-in-the-middle" issue by constructing explicit prompts. We finally employ the proposed LongInOutBench to evaluate our RAL-Writer against comparable baselines, and the results demonstrate the effectiveness of our approach. Our code has been released at this https URL. 

---
# Enhanced Multi-Tuple Extraction for Alloys: Integrating Pointer Networks and Augmented Attention 

**Authors**: Mengzhe Hei, Zhouran Zhang, Qingbao Liu, Yan Pan, Xiang Zhao, Yongqian Peng, Yicong Ye, Xin Zhang, Shuxin Bai  

**Link**: [PDF](https://arxiv.org/pdf/2503.06861)  

**Abstract**: Extracting high-quality structured information from scientific literature is crucial for advancing material design through data-driven methods. Despite the considerable research in natural language processing for dataset extraction, effective approaches for multi-tuple extraction in scientific literature remain scarce due to the complex interrelations of tuples and contextual ambiguities. In the study, we illustrate the multi-tuple extraction of mechanical properties from multi-principal-element alloys and presents a novel framework that combines an entity extraction model based on MatSciBERT with pointer networks and an allocation model utilizing inter- and intra-entity attention. Our rigorous experiments on tuple extraction demonstrate impressive F1 scores of 0.963, 0.947, 0.848, and 0.753 across datasets with 1, 2, 3, and 4 tuples, confirming the effectiveness of the model. Furthermore, an F1 score of 0.854 was achieved on a randomly curated dataset. These results highlight the model's capacity to deliver precise and structured information, offering a robust alternative to large language models and equipping researchers with essential data for fostering data-driven innovations. 

---
# On the Mutual Influence of Gender and Occupation in LLM Representations 

**Authors**: Haozhe An, Connor Baumler, Abhilasha Sancheti, Rachel Rudinger  

**Link**: [PDF](https://arxiv.org/pdf/2503.06792)  

**Abstract**: We examine LLM representations of gender for first names in various occupational contexts to study how occupations and the gender perception of first names in LLMs influence each other mutually. We find that LLMs' first-name gender representations correlate with real-world gender statistics associated with the name, and are influenced by the co-occurrence of stereotypically feminine or masculine occupations. Additionally, we study the influence of first-name gender representations on LLMs in a downstream occupation prediction task and their potential as an internal metric to identify extrinsic model biases. While feminine first-name embeddings often raise the probabilities for female-dominated jobs (and vice versa for male-dominated jobs), reliably using these internal gender representations for bias detection remains challenging. 

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
# Gender Encoding Patterns in Pretrained Language Model Representations 

**Authors**: Mahdi Zakizadeh, Mohammad Taher Pilehvar  

**Link**: [PDF](https://arxiv.org/pdf/2503.06734)  

**Abstract**: Gender bias in pretrained language models (PLMs) poses significant social and ethical challenges. Despite growing awareness, there is a lack of comprehensive investigation into how different models internally represent and propagate such biases. This study adopts an information-theoretic approach to analyze how gender biases are encoded within various encoder-based architectures. We focus on three key aspects: identifying how models encode gender information and biases, examining the impact of bias mitigation techniques and fine-tuning on the encoded biases and their effectiveness, and exploring how model design differences influence the encoding of biases. Through rigorous and systematic investigation, our findings reveal a consistent pattern of gender encoding across diverse models. Surprisingly, debiasing techniques often exhibit limited efficacy, sometimes inadvertently increasing the encoded bias in internal representations while reducing bias in model output distributions. This highlights a disconnect between mitigating bias in output distributions and addressing its internal representations. This work provides valuable guidance for advancing bias mitigation strategies and fostering the development of more equitable language models. 

---
# Topology of Syntax Networks across Languages 

**Authors**: Juan Soria-Postigo, Luis F Seoane  

**Link**: [PDF](https://arxiv.org/pdf/2503.06724)  

**Abstract**: Syntax connects words to each other in very specific ways. Two words are syntactically connected if they depend
directly on each other. Syntactic connections usually happen within a sentence. Gathering all those connection
across several sentences gives birth to syntax networks. Earlier studies in the field have analysed the structure and
properties of syntax networks trying to find clusters/phylogenies of languages that share similar network features.
The results obtained in those studies will be put to test in this thesis by increasing both the number of languages
and the number of properties considered in the analysis. Besides that, language networks of particular languages
will be inspected in depth by means of a novel network analysis [25]. Words (nodes of the network) will be clustered
into topological communities whose members share similar features. The properties of each of these communities
will be thoroughly studied along with the Part of Speech (grammatical class) of each word. Results across different
languages will also be compared in an attempt to discover universally preserved structural patterns across syntax
networks. 

---
# Delusions of Large Language Models 

**Authors**: Hongshen Xu, Zixv yang, Zichen Zhu, Kunyao Lan, Zihan Wang, Mengyue Wu, Ziwei Ji, Lu Chen, Pascale Fung, Kai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2503.06709)  

**Abstract**: Large Language Models often generate factually incorrect but plausible outputs, known as hallucinations. We identify a more insidious phenomenon, LLM delusion, defined as high belief hallucinations, incorrect outputs with abnormally high confidence, making them harder to detect and mitigate. Unlike ordinary hallucinations, delusions persist with low uncertainty, posing significant challenges to model reliability. Through empirical analysis across different model families and sizes on several Question Answering tasks, we show that delusions are prevalent and distinct from hallucinations. LLMs exhibit lower honesty with delusions, which are harder to override via finetuning or self reflection. We link delusion formation with training dynamics and dataset noise and explore mitigation strategies such as retrieval augmented generation and multi agent debating to mitigate delusions. By systematically investigating the nature, prevalence, and mitigation of LLM delusions, our study provides insights into the underlying causes of this phenomenon and outlines future directions for improving model reliability. 

---
# Alignment for Efficient Tool Calling of Large Language Models 

**Authors**: Hongshen Xu, Zihan Wang, Zichen Zhu, Lei Pan, Xingyu Chen, Lu Chen, Kai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2503.06708)  

**Abstract**: Recent advancements in tool learning have enabled large language models (LLMs) to integrate external tools, enhancing their task performance by expanding their knowledge boundaries. However, relying on tools often introduces tradeoffs between performance, speed, and cost, with LLMs sometimes exhibiting overreliance and overconfidence in tool usage. This paper addresses the challenge of aligning LLMs with their knowledge boundaries to make more intelligent decisions about tool invocation. We propose a multi objective alignment framework that combines probabilistic knowledge boundary estimation with dynamic decision making, allowing LLMs to better assess when to invoke tools based on their confidence. Our framework includes two methods for knowledge boundary estimation, consistency based and absolute estimation, and two training strategies for integrating these estimates into the model decision making process. Experimental results on various tool invocation scenarios demonstrate the effectiveness of our framework, showing significant improvements in tool efficiency by reducing unnecessary tool usage. 

---
# PFDial: A Structured Dialogue Instruction Fine-tuning Method Based on UML Flowcharts 

**Authors**: Ming Zhang, Yuhui Wang, Yujiong Shen, Tingyi Yang, Changhao Jiang, Yilong Wu, Shihan Dou, Qinhao Chen, Zhiheng Xi, Zhihao Zhang, Yi Dong, Zhen Wang, Zhihui Fei, Mingyang Wan, Tao Liang, Guojun Ma, Qi Zhang, Tao Gui, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.06706)  

**Abstract**: Process-driven dialogue systems, which operate under strict predefined process constraints, are essential in customer service and equipment maintenance scenarios. Although Large Language Models (LLMs) have shown remarkable progress in dialogue and reasoning, they still struggle to solve these strictly constrained dialogue tasks. To address this challenge, we construct Process Flow Dialogue (PFDial) dataset, which contains 12,705 high-quality Chinese dialogue instructions derived from 440 flowcharts containing 5,055 process nodes. Based on PlantUML specification, each UML flowchart is converted into atomic dialogue units i.e., structured five-tuples. Experimental results demonstrate that a 7B model trained with merely 800 samples, and a 0.5B model trained on total data both can surpass 90% accuracy. Additionally, the 8B model can surpass GPT-4o up to 43.88% with an average of 11.00%. We further evaluate models' performance on challenging backward transitions in process flows and conduct an in-depth analysis of various dataset formats to reveal their impact on model performance in handling decision and sequential branches. The data is released in this https URL. 

---
# InftyThink: Breaking the Length Limits of Long-Context Reasoning in Large Language Models 

**Authors**: Yuchen Yan, Yongliang Shen, Yang Liu, Jin Jiang, Mengdi Zhang, Jian Shao, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2503.06692)  

**Abstract**: Advanced reasoning in large language models has achieved remarkable performance on challenging tasks, but the prevailing long-context reasoning paradigm faces critical limitations: quadratic computational scaling with sequence length, reasoning constrained by maximum context boundaries, and performance degradation beyond pre-training context windows. Existing approaches primarily compress reasoning chains without addressing the fundamental scaling problem. To overcome these challenges, we introduce InftyThink, a paradigm that transforms monolithic reasoning into an iterative process with intermediate summarization. By interleaving short reasoning segments with concise progress summaries, our approach enables unbounded reasoning depth while maintaining bounded computational costs. This creates a characteristic sawtooth memory pattern that significantly reduces computational complexity compared to traditional approaches. Furthermore, we develop a methodology for reconstructing long-context reasoning datasets into our iterative format, transforming OpenR1-Math into 333K training instances. Experiments across multiple model architectures demonstrate that our approach reduces computational costs while improving performance, with Qwen2.5-Math-7B showing 3-13% improvements across MATH500, AIME24, and GPQA_diamond benchmarks. Our work challenges the assumed trade-off between reasoning depth and computational efficiency, providing a more scalable approach to complex reasoning without architectural modifications. 

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
# BingoGuard: LLM Content Moderation Tools with Risk Levels 

**Authors**: Fan Yin, Philippe Laban, Xiangyu Peng, Yilun Zhou, Yixin Mao, Vaibhav Vats, Linnea Ross, Divyansh Agarwal, Caiming Xiong, Chien-Sheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.06550)  

**Abstract**: Malicious content generated by large language models (LLMs) can pose varying degrees of harm. Although existing LLM-based moderators can detect harmful content, they struggle to assess risk levels and may miss lower-risk outputs. Accurate risk assessment allows platforms with different safety thresholds to tailor content filtering and rejection. In this paper, we introduce per-topic severity rubrics for 11 harmful topics and build BingoGuard, an LLM-based moderation system designed to predict both binary safety labels and severity levels. To address the lack of annotations on levels of severity, we propose a scalable generate-then-filter framework that first generates responses across different severity levels and then filters out low-quality responses. Using this framework, we create BingoGuardTrain, a training dataset with 54,897 examples covering a variety of topics, response severity, styles, and BingoGuardTest, a test set with 988 examples explicitly labeled based on our severity rubrics that enables fine-grained analysis on model behaviors on different severity levels. Our BingoGuard-8B, trained on BingoGuardTrain, achieves the state-of-the-art performance on several moderation benchmarks, including WildGuardTest and HarmBench, as well as BingoGuardTest, outperforming best public models, WildGuard, by 4.3\%. Our analysis demonstrates that incorporating severity levels into training significantly enhances detection performance and enables the model to effectively gauge the severity of harmful responses. 

---
# KréyoLID From Language Identification Towards Language Mining 

**Authors**: Rasul Dent, Pedro Ortiz Suarez, Thibault Clérice, Benoît Sagot  

**Link**: [PDF](https://arxiv.org/pdf/2503.06547)  

**Abstract**: Automatic language identification is frequently framed as a multi-class classification problem. However, when creating digital corpora for less commonly written languages, it may be more appropriate to consider it a data mining problem. For these varieties, one knows ahead of time that the vast majority of documents are of little interest. By minimizing resources spent on classifying such documents, we can create corpora much faster and with better coverage than using established pipelines. To demonstrate the effectiveness of the language mining perspective, we introduce a new pipeline and corpora for several French-based Creoles. 

---
# SafeSpeech: A Comprehensive and Interactive Tool for Analysing Sexist and Abusive Language in Conversations 

**Authors**: Xingwei Tan, Chen Lyu, Hafiz Muhammad Umer, Sahrish Khan, Mahathi Parvatham, Lois Arthurs, Simon Cullen, Shelley Wilson, Arshad Jhumka, Gabriele Pergola  

**Link**: [PDF](https://arxiv.org/pdf/2503.06534)  

**Abstract**: Detecting toxic language including sexism, harassment and abusive behaviour, remains a critical challenge, particularly in its subtle and context-dependent forms. Existing approaches largely focus on isolated message-level classification, overlooking toxicity that emerges across conversational contexts. To promote and enable future research in this direction, we introduce SafeSpeech, a comprehensive platform for toxic content detection and analysis that bridges message-level and conversation-level insights. The platform integrates fine-tuned classifiers and large language models (LLMs) to enable multi-granularity detection, toxic-aware conversation summarization, and persona profiling. SafeSpeech also incorporates explainability mechanisms, such as perplexity gain analysis, to highlight the linguistic elements driving predictions. Evaluations on benchmark datasets, including EDOS, OffensEval, and HatEval, demonstrate the reproduction of state-of-the-art performance across multiple tasks, including fine-grained sexism detection. 

---
# MetaXCR: Reinforcement-Based Meta-Transfer Learning for Cross-Lingual Commonsense Reasoning 

**Authors**: Jie He, Yu Fu  

**Link**: [PDF](https://arxiv.org/pdf/2503.06531)  

**Abstract**: Commonsense reasoning (CR) has been studied in many pieces of domain and has achieved great progress with the aid of large datasets. Unfortunately, most existing CR datasets are built in English, so most previous work focus on English. Furthermore, as the annotation of commonsense reasoning is costly, it is impossible to build a large dataset for every novel task. Therefore, there are growing appeals for Cross-lingual Low-Resource Commonsense Reasoning, which aims to leverage diverse existed English datasets to help the model adapt to new cross-lingual target datasets with limited labeled data. In this paper, we propose a multi-source adapter for cross-lingual low-resource Commonsense Reasoning (MetaXCR). In this framework, we first extend meta learning by incorporating multiple training datasets to learn a generalized task adapters across different tasks. Then, we further introduce a reinforcement-based sampling strategy to help the model sample the source task that is the most helpful to the target task. Finally, we introduce two types of cross-lingual meta-adaption methods to enhance the performance of models on target languages. Extensive experiments demonstrate MetaXCR is superior over state-of-the-arts, while being trained with fewer parameters than other work. 

---
# GFlowVLM: Enhancing Multi-step Reasoning in Vision-Language Models with Generative Flow Networks 

**Authors**: Haoqiang Kang, Enna Sachdeva, Piyush Gupta, Sangjae Bae, Kwonjoon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.06514)  

**Abstract**: Vision-Language Models (VLMs) have recently shown promising advancements in sequential decision-making tasks through task-specific fine-tuning. However, common fine-tuning methods, such as Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) techniques like Proximal Policy Optimization (PPO), present notable limitations: SFT assumes Independent and Identically Distributed (IID) data, while PPO focuses on maximizing cumulative rewards. These limitations often restrict solution diversity and hinder generalization in multi-step reasoning tasks. To address these challenges, we introduce a novel framework, GFlowVLM, a framework that fine-tune VLMs using Generative Flow Networks (GFlowNets) to promote generation of diverse solutions for complex reasoning tasks. GFlowVLM models the environment as a non-Markovian decision process, allowing it to capture long-term dependencies essential for real-world applications. It takes observations and task descriptions as inputs to prompt chain-of-thought (CoT) reasoning which subsequently guides action selection. We use task based rewards to fine-tune VLM with GFlowNets. This approach enables VLMs to outperform prior fine-tuning methods, including SFT and RL. Empirical results demonstrate the effectiveness of GFlowVLM on complex tasks such as card games (NumberLine, BlackJack) and embodied planning tasks (ALFWorld), showing enhanced training efficiency, solution diversity, and stronger generalization capabilities across both in-distribution and out-of-distribution scenarios. 

---
# VisualSimpleQA: A Benchmark for Decoupled Evaluation of Large Vision-Language Models in Fact-Seeking Question Answering 

**Authors**: Yanling Wang, Yihan Zhao, Xiaodong Chen, Shasha Guo, Lixin Liu, Haoyang Li, Yong Xiao, Jing Zhang, Qi Li, Ke Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.06492)  

**Abstract**: Large vision-language models (LVLMs) have demonstrated remarkable achievements, yet the generation of non-factual responses remains prevalent in fact-seeking question answering (QA). Current multimodal fact-seeking benchmarks primarily focus on comparing model outputs to ground truth answers, providing limited insights into the performance of modality-specific modules. To bridge this gap, we introduce VisualSimpleQA, a multimodal fact-seeking benchmark with two key features. First, it enables streamlined and decoupled evaluation of LVLMs in visual and linguistic modalities. Second, it incorporates well-defined difficulty criteria to guide human annotation and facilitates the extraction of a challenging subset, VisualSimpleQA-hard. Experiments on 15 LVLMs show that even state-of-the-art models such as GPT-4o achieve merely 60%+ correctness in multimodal fact-seeking QA on VisualSimpleQA and 30%+ on VisualSimpleQA-hard. Furthermore, the decoupled evaluation across these models highlights substantial opportunities for improvement in both visual and linguistic modules. The dataset is available at this https URL. 

---
# MoFE: Mixture of Frozen Experts Architecture 

**Authors**: Jean Seo, Jaeyoon Kim, Hyopil Shin  

**Link**: [PDF](https://arxiv.org/pdf/2503.06491)  

**Abstract**: We propose the Mixture of Frozen Experts (MoFE) architecture, which integrates Parameter-efficient Fine-tuning (PEFT) and the Mixture of Experts (MoE) architecture to enhance both training efficiency and model scalability. By freezing the Feed Forward Network (FFN) layers within the MoE framework, MoFE significantly reduces the number of trainable parameters, improving training efficiency while still allowing for effective knowledge transfer from the expert models. This facilitates the creation of models proficient in multiple domains. We conduct experiments to evaluate the trade-offs between performance and efficiency, compare MoFE with other PEFT methodologies, assess the impact of domain expertise in the constituent models, and determine the optimal training strategy. The results show that, although there may be some trade-offs in performance, the efficiency gains are substantial, making MoFE a reasonable solution for real-world, resource-constrained environments. 

---
# Graph Retrieval-Augmented LLM for Conversational Recommendation Systems 

**Authors**: Zhangchi Qiu, Linhao Luo, Zicheng Zhao, Shirui Pan, Alan Wee-Chung Liew  

**Link**: [PDF](https://arxiv.org/pdf/2503.06430)  

**Abstract**: Conversational Recommender Systems (CRSs) have emerged as a transformative paradigm for offering personalized recommendations through natural language dialogue. However, they face challenges with knowledge sparsity, as users often provide brief, incomplete preference statements. While recent methods have integrated external knowledge sources to mitigate this, they still struggle with semantic understanding and complex preference reasoning. Recent Large Language Models (LLMs) demonstrate promising capabilities in natural language understanding and reasoning, showing significant potential for CRSs. Nevertheless, due to the lack of domain knowledge, existing LLM-based CRSs either produce hallucinated recommendations or demand expensive domain-specific training, which largely limits their applicability. In this work, we present G-CRS (Graph Retrieval-Augmented Large Language Model for Conversational Recommender Systems), a novel training-free framework that combines graph retrieval-augmented generation and in-context learning to enhance LLMs' recommendation capabilities. Specifically, G-CRS employs a two-stage retrieve-and-recommend architecture, where a GNN-based graph reasoner first identifies candidate items, followed by Personalized PageRank exploration to jointly discover potential items and similar user interactions. These retrieved contexts are then transformed into structured prompts for LLM reasoning, enabling contextually grounded recommendations without task-specific training. Extensive experiments on two public datasets show that G-CRS achieves superior recommendation performance compared to existing methods without requiring task-specific training. 

---
# Training LLM-based Tutors to Improve Student Learning Outcomes in Dialogues 

**Authors**: Alexander Scarlatos, Naiming Liu, Jaewook Lee, Richard Baraniuk, Andrew Lan  

**Link**: [PDF](https://arxiv.org/pdf/2503.06424)  

**Abstract**: Generative artificial intelligence (AI) has the potential to scale up personalized tutoring through large language models (LLMs). Recent AI tutors are adapted for the tutoring task by training or prompting LLMs to follow effective pedagogical principles, though they are not trained to maximize student learning throughout the course of a dialogue. Therefore, they may engage with students in a suboptimal way. We address this limitation by introducing an approach to train LLMs to generate tutor utterances that maximize the likelihood of student correctness, while still encouraging the model to follow good pedagogical practice. Specifically, we generate a set of candidate tutor utterances and score them using (1) an LLM-based student model to predict the chance of correct student responses and (2) a pedagogical rubric evaluated by GPT-4o. We then use the resulting data to train an open-source LLM, Llama 3.1 8B, using direct preference optimization. We show that tutor utterances generated by our model lead to significantly higher chances of correct student responses while maintaining the pedagogical quality of GPT-4o. We also conduct qualitative analyses and a human evaluation to demonstrate that our model generates high quality tutor utterances. 

---
# How LLMs Learn: Tracing Internal Representations with Sparse Autoencoders 

**Authors**: Tatsuro Inaba, Kentaro Inui, Yusuke Miyao, Yohei Oseki, Benjamin Heinzerling, Yu Takagi  

**Link**: [PDF](https://arxiv.org/pdf/2503.06394)  

**Abstract**: Large Language Models (LLMs) demonstrate remarkable multilingual capabilities and broad knowledge. However, the internal mechanisms underlying the development of these capabilities remain poorly understood. To investigate this, we analyze how the information encoded in LLMs' internal representations evolves during the training process. Specifically, we train sparse autoencoders at multiple checkpoints of the model and systematically compare the interpretative results across these stages. Our findings suggest that LLMs initially acquire language-specific knowledge independently, followed by cross-linguistic correspondences. Moreover, we observe that after mastering token-level knowledge, the model transitions to learning higher-level, abstract concepts, indicating the development of more conceptual understanding. 

---
# States of LLM-generated Texts and Phase Transitions between them 

**Authors**: Nikolay Mikhaylovskiy  

**Link**: [PDF](https://arxiv.org/pdf/2503.06330)  

**Abstract**: It is known for some time that autocorrelations of words in human-written texts decay according to a power law. Recent works have also shown that the autocorrelations decay in texts generated by LLMs is qualitatively different from the literary texts. Solid state physics tie the autocorrelations decay laws to the states of matter. In this work, we empirically demonstrate that, depending on the temperature parameter, LLMs can generate text that can be classified as solid, critical state or gas. 

---
# MoEMoE: Question Guided Dense and Scalable Sparse Mixture-of-Expert for Multi-source Multi-modal Answering 

**Authors**: Vinay Kumar Verma, Shreyas Sunil Kulkarni, Happy Mittal, Deepak Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2503.06296)  

**Abstract**: Question Answering (QA) and Visual Question Answering (VQA) are well-studied problems in the language and vision domain. One challenging scenario involves multiple sources of information, each of a different modality, where the answer to the question may exist in one or more sources. This scenario contains richer information but is highly complex to handle. In this work, we formulate a novel question-answer generation (QAG) framework in an environment containing multi-source, multimodal information. The answer may belong to any or all sources; therefore, selecting the most prominent answer source or an optimal combination of all sources for a given question is challenging. To address this issue, we propose a question-guided attention mechanism that learns attention across multiple sources and decodes this information for robust and unbiased answer generation. To learn attention within each source, we introduce an explicit alignment between questions and various information sources, which facilitates identifying the most pertinent parts of the source information relative to the question. Scalability in handling diverse questions poses a challenge. We address this by extending our model to a sparse mixture-of-experts (sparse-MoE) framework, enabling it to handle thousands of question types. Experiments on T5 and Flan-T5 using three datasets demonstrate the model's efficacy, supported by ablation studies. 

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
# KnowLogic: A Benchmark for Commonsense Reasoning via Knowledge-Driven Data Synthesis 

**Authors**: Weidong Zhan, Yue Wang, Nan Hu, Liming Xiao, Jingyuan Ma, Yuhang Qin, Zheng Li, Yixin Yang, Sirui Deng, Jinkun Ding, Wenhan Ma, Rui Li, Weilin Luo, Qun Liu, Zhifang Sui  

**Link**: [PDF](https://arxiv.org/pdf/2503.06218)  

**Abstract**: Current evaluations of commonsense reasoning in LLMs are hindered by the scarcity of natural language corpora with structured annotations for reasoning tasks. To address this, we introduce KnowLogic, a benchmark generated through a knowledge-driven synthetic data strategy. KnowLogic integrates diverse commonsense knowledge, plausible scenarios, and various types of logical reasoning. One of the key advantages of KnowLogic is its adjustable difficulty levels, allowing for flexible control over question complexity. It also includes fine-grained labels for in-depth evaluation of LLMs' reasoning abilities across multiple dimensions. Our benchmark consists of 3,000 bilingual (Chinese and English) questions across various domains, and presents significant challenges for current LLMs, with the highest-performing model achieving only 69.57\%. Our analysis highlights common errors, such as misunderstandings of low-frequency commonsense, logical inconsistencies, and overthinking. This approach, along with our benchmark, provides a valuable tool for assessing and enhancing LLMs' commonsense reasoning capabilities and can be applied to a wide range of knowledge domains. 

---
# Text-Speech Language Models with Improved Cross-Modal Transfer by Aligning Abstraction Levels 

**Authors**: Santiago Cuervo, Adel Moumen, Yanis Labrak, Sameer Khurana, Antoine Laurent, Mickael Rouvier, Ricard Marxer  

**Link**: [PDF](https://arxiv.org/pdf/2503.06211)  

**Abstract**: Text-Speech Language Models (TSLMs) -- language models trained to jointly process and generate text and speech -- aim to enable cross-modal knowledge transfer to overcome the scaling limitations of unimodal speech LMs. The predominant approach to TSLM training expands the vocabulary of a pre-trained text LM by appending new embeddings and linear projections for speech, followed by fine-tuning on speech data. We hypothesize that this method limits cross-modal transfer by neglecting feature compositionality, preventing text-learned functions from being fully leveraged at appropriate abstraction levels. To address this, we propose augmenting vocabulary expansion with modules that better align abstraction levels across layers. Our models, \textsc{SmolTolk}, rival or surpass state-of-the-art TSLMs trained with orders of magnitude more compute. Representation analyses and improved multimodal performance suggest our method enhances cross-modal transfer. 

---
# CUPCase: Clinically Uncommon Patient Cases and Diagnoses Dataset 

**Authors**: Oriel Perets, Ofir Ben Shoham, Nir Grinberg, Nadav Rappoport  

**Link**: [PDF](https://arxiv.org/pdf/2503.06204)  

**Abstract**: Medical benchmark datasets significantly contribute to developing Large Language Models (LLMs) for medical knowledge extraction, diagnosis, summarization, and other uses. Yet, current benchmarks are mainly derived from exam questions given to medical students or cases described in the medical literature, lacking the complexity of real-world patient cases that deviate from classic textbook abstractions. These include rare diseases, uncommon presentations of common diseases, and unexpected treatment responses. Here, we construct Clinically Uncommon Patient Cases and Diagnosis Dataset (CUPCase) based on 3,562 real-world case reports from BMC, including diagnoses in open-ended textual format and as multiple-choice options with distractors. Using this dataset, we evaluate the ability of state-of-the-art LLMs, including both general-purpose and Clinical LLMs, to identify and correctly diagnose a patient case, and test models' performance when only partial information about cases is available. Our findings show that general-purpose GPT-4o attains the best performance in both the multiple-choice task (average accuracy of 87.9%) and the open-ended task (BERTScore F1 of 0.764), outperforming several LLMs with a focus on the medical domain such as Meditron-70B and MedLM-Large. Moreover, GPT-4o was able to maintain 87% and 88% of its performance with only the first 20% of tokens of the case presentation in multiple-choice and free text, respectively, highlighting the potential of LLMs to aid in early diagnosis in real-world cases. CUPCase expands our ability to evaluate LLMs for clinical decision support in an open and reproducible manner. 

---
# Sample-aware Adaptive Structured Pruning for Large Language Models 

**Authors**: Jun Kong, Xinge Ma, Jin Wang, Xuejie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.06184)  

**Abstract**: Large language models (LLMs) have achieved outstanding performance in natural language processing, but enormous model sizes and high computational costs limit their practical deployment. Structured pruning can effectively reduce the resource demands for deployment by removing redundant model parameters. However, the randomly selected calibration data and fixed single importance estimation metrics in existing structured pruning methods lead to degraded performance of pruned models. This study introduces AdaPruner, a sample-aware adaptive structured pruning framework for LLMs, aiming to optimize the calibration data and importance estimation metrics in the structured pruning process. Specifically, AdaPruner effectively removes redundant parameters from LLMs by constructing a structured pruning solution space and then employing Bayesian optimization to adaptively search for the optimal calibration data and importance estimation metrics. Experimental results show that the AdaPruner outperforms existing structured pruning methods on a family of LLMs with varying pruning ratios, demonstrating its applicability and robustness. Remarkably, at a 20\% pruning ratio, the model pruned with AdaPruner maintains 97\% of the performance of the unpruned model. 

---
# GRP: Goal-Reversed Prompting for Zero-Shot Evaluation with LLMs 

**Authors**: Mingyang Song, Mao Zheng, Xuan Luo  

**Link**: [PDF](https://arxiv.org/pdf/2503.06139)  

**Abstract**: Using Large Language Models (LLMs) to evaluate and compare two answers from different models typically involves having LLM-based judges select the better answer. However, humans often approach problem-solving from a reverse perspective, for instance, by choosing the worse option instead of the better one in a pairwise comparison. Generally, this kind of reverse thinking plays a crucial role in human reasoning and decision-making and can further test the difference between original and reverse thought processes simultaneously. To address the above issue, in this paper, we propose a Goal-Reversed Prompting (GRP) approach for pairwise evaluation that shifts the original task from selecting the better answer to choosing the worse one. We encourage LLMs to think in reverse by prompting LLMs to identify the worse response. Experiments on closed-source models demonstrate that GRP significantly enhances evaluation capabilities, outperforming the prompt template with the original goal. 

---
# Evaluating Discourse Cohesion in Pre-trained Language Models 

**Authors**: Jie He, Wanqiu Long, Deyi Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2503.06137)  

**Abstract**: Large pre-trained neural models have achieved remarkable success in natural language process (NLP), inspiring a growing body of research analyzing their ability from different aspects. In this paper, we propose a test suite to evaluate the cohesive ability of pre-trained language models. The test suite contains multiple cohesion phenomena between adjacent and non-adjacent sentences. We try to compare different pre-trained language models on these phenomena and analyze the experimental results,hoping more attention can be given to discourse cohesion in the future. 

---
# Theta Theory: operads and coloring 

**Authors**: Matilde Marcolli, Richard K. Larson  

**Link**: [PDF](https://arxiv.org/pdf/2503.06091)  

**Abstract**: We give an explicit construction of the generating set of a colored operad that implements theta theory in the mathematical model of Minimalism in generative linguistics, in the form of a coloring algorithm for syntactic objects. We show that the coproduct operation on workspaces allows for a recursive implementation of the theta criterion. We also show that this filtering by coloring rules on structures freely formed by Merge is equivalent to a process of structure formation by a colored version of Merge: the form of the generators of the colored operad then implies the dichotomy is semantics between External and Internal Merge, where Internal Merge only moves to non-theta positions. 

---
# Multi-Attribute Multi-Grained Adaptation of Pre-Trained Language Models for Text Understanding from Bayesian Perspective 

**Authors**: You Zhang, Jin Wang, Liang-Chih Yu, Dan Xu, Xuejie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.06085)  

**Abstract**: Current neural networks often employ multi-domain-learning or attribute-injecting mechanisms to incorporate non-independent and identically distributed (non-IID) information for text understanding tasks by capturing individual characteristics and the relationships among samples. However, the extent of the impact of non-IID information and how these methods affect pre-trained language models (PLMs) remains unclear. This study revisits the assumption that non-IID information enhances PLMs to achieve performance improvements from a Bayesian perspective, which unearths and integrates non-IID and IID features. Furthermore, we proposed a multi-attribute multi-grained framework for PLM adaptations (M2A), which combines multi-attribute and multi-grained views to mitigate uncertainty in a lightweight manner. We evaluate M2A through prevalent text-understanding datasets and demonstrate its superior performance, mainly when data are implicitly non-IID, and PLMs scale larger. 

---
# An Empirical Study of Causal Relation Extraction Transfer: Design and Data 

**Authors**: Sydney Anuyah, Jack Vanschaik, Palak Jain, Sawyer Lehman, Sunandan Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2503.06076)  

**Abstract**: We conduct an empirical analysis of neural network architectures and data transfer strategies for causal relation extraction. By conducting experiments with various contextual embedding layers and architectural components, we show that a relatively straightforward BioBERT-BiGRU relation extraction model generalizes better than other architectures across varying web-based sources and annotation strategies. Furthermore, we introduce a metric for evaluating transfer performance, $F1_{phrase}$ that emphasizes noun phrase localization rather than directly matching target tags. Using this metric, we can conduct data transfer experiments, ultimately revealing that augmentation with data with varying domains and annotation styles can improve performance. Data augmentation is especially beneficial when an adequate proportion of implicitly and explicitly causal sentences are included. 

---
# Towards Conversational AI for Disease Management 

**Authors**: Anil Palepu, Valentin Liévin, Wei-Hung Weng, Khaled Saab, David Stutz, Yong Cheng, Kavita Kulkarni, S. Sara Mahdavi, Joëlle Barral, Dale R. Webster, Katherine Chou, Avinatan Hassidim, Yossi Matias, James Manyika, Ryutaro Tanno, Vivek Natarajan, Adam Rodman, Tao Tu, Alan Karthikesalingam, Mike Schaekermann  

**Link**: [PDF](https://arxiv.org/pdf/2503.06074)  

**Abstract**: While large language models (LLMs) have shown promise in diagnostic dialogue, their capabilities for effective management reasoning - including disease progression, therapeutic response, and safe medication prescription - remain under-explored. We advance the previously demonstrated diagnostic capabilities of the Articulate Medical Intelligence Explorer (AMIE) through a new LLM-based agentic system optimised for clinical management and dialogue, incorporating reasoning over the evolution of disease and multiple patient visit encounters, response to therapy, and professional competence in medication prescription. To ground its reasoning in authoritative clinical knowledge, AMIE leverages Gemini's long-context capabilities, combining in-context retrieval with structured reasoning to align its output with relevant and up-to-date clinical practice guidelines and drug formularies. In a randomized, blinded virtual Objective Structured Clinical Examination (OSCE) study, AMIE was compared to 21 primary care physicians (PCPs) across 100 multi-visit case scenarios designed to reflect UK NICE Guidance and BMJ Best Practice guidelines. AMIE was non-inferior to PCPs in management reasoning as assessed by specialist physicians and scored better in both preciseness of treatments and investigations, and in its alignment with and grounding of management plans in clinical guidelines. To benchmark medication reasoning, we developed RxQA, a multiple-choice question benchmark derived from two national drug formularies (US, UK) and validated by board-certified pharmacists. While AMIE and PCPs both benefited from the ability to access external drug information, AMIE outperformed PCPs on higher difficulty questions. While further research would be needed before real-world translation, AMIE's strong performance across evaluations marks a significant step towards conversational AI as a tool in disease management. 

---
# GEM: Empowering MLLM for Grounded ECG Understanding with Time Series and Images 

**Authors**: Xiang Lan, Feng Wu, Kai He, Qinghao Zhao, Shenda Hong, Mengling Feng  

**Link**: [PDF](https://arxiv.org/pdf/2503.06073)  

**Abstract**: While recent multimodal large language models (MLLMs) have advanced automated ECG interpretation, they still face two key limitations: (1) insufficient multimodal synergy between time series signals and visual ECG representations, and (2) limited explainability in linking diagnoses to granular waveform evidence. We introduce GEM, the first MLLM unifying ECG time series, 12-lead ECG images and text for grounded and clinician-aligned ECG interpretation. GEM enables feature-grounded analysis, evidence-driven reasoning, and a clinician-like diagnostic process through three core innovations: a dual-encoder framework extracting complementary time series and image features, cross-modal alignment for effective multimodal understanding, and knowledge-guided instruction generation for generating high-granularity grounding data (ECG-Grounding) linking diagnoses to measurable parameters ($e.g.$, QRS/PR Intervals). Additionally, we propose the Grounded ECG Understanding task, a clinically motivated benchmark designed to comprehensively assess the MLLM's capability in grounded ECG understanding. Experimental results on both existing and our proposed benchmarks show GEM significantly improves predictive performance (CSN $7.4\% \uparrow$), explainability ($22.7\% \uparrow$), and grounding ($24.8\% \uparrow$), making it more suitable for real-world clinical applications. GitHub repository: this https URL 

---
# A Survey on Post-training of Large Language Models 

**Authors**: Guiyao Tie, Zeli Zhao, Dingjie Song, Fuyang Wei, Rong Zhou, Yurou Dai, Wen Yin, Zhejian Yang, Jiangyue Yan, Yao Su, Zhenhan Dai, Yifeng Xie, Yihan Cao, Lichao Sun, Pan Zhou, Lifang He, Hechang Chen, Yu Zhang, Qingsong Wen, Tianming Liu, Neil Zhenqiang Gong, Jiliang Tang, Caiming Xiong, Heng Ji, Philip S. Yu, Jianfeng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2503.06072)  

**Abstract**: The emergence of Large Language Models (LLMs) has fundamentally transformed natural language processing, making them indispensable across domains ranging from conversational systems to scientific exploration. However, their pre-trained architectures often reveal limitations in specialized contexts, including restricted reasoning capacities, ethical uncertainties, and suboptimal domain-specific performance. These challenges necessitate advanced post-training language models (PoLMs) to address these shortcomings, such as OpenAI-o1/o3 and DeepSeek-R1 (collectively known as Large Reasoning Models, or LRMs). This paper presents the first comprehensive survey of PoLMs, systematically tracing their evolution across five core paradigms: Fine-tuning, which enhances task-specific accuracy; Alignment, which ensures alignment with human preferences; Reasoning, which advances multi-step inference despite challenges in reward design; Efficiency, which optimizes resource utilization amidst increasing complexity; and Integration and Adaptation, which extend capabilities across diverse modalities while addressing coherence issues. Charting progress from ChatGPT's foundational alignment strategies to DeepSeek-R1's innovative reasoning advancements, we illustrate how PoLMs leverage datasets to mitigate biases, deepen reasoning capabilities, and enhance domain adaptability. Our contributions include a pioneering synthesis of PoLM evolution, a structured taxonomy categorizing techniques and datasets, and a strategic agenda emphasizing the role of LRMs in improving reasoning proficiency and domain flexibility. As the first survey of its scope, this work consolidates recent PoLM advancements and establishes a rigorous intellectual framework for future research, fostering the development of LLMs that excel in precision, ethical robustness, and versatility across scientific and societal applications. 

---
# Fine-Grained Bias Detection in LLM: Enhancing detection mechanisms for nuanced biases 

**Authors**: Suvendu Mohanty  

**Link**: [PDF](https://arxiv.org/pdf/2503.06054)  

**Abstract**: Recent advancements in Artificial Intelligence, particularly in Large Language Models (LLMs), have transformed natural language processing by improving generative capabilities. However, detecting biases embedded within these models remains a challenge. Subtle biases can propagate misinformation, influence decision-making, and reinforce stereotypes, raising ethical concerns. This study presents a detection framework to identify nuanced biases in LLMs. The approach integrates contextual analysis, interpretability via attention mechanisms, and counterfactual data augmentation to capture hidden biases across linguistic contexts. The methodology employs contrastive prompts and synthetic datasets to analyze model behaviour across cultural, ideological, and demographic scenarios.
Quantitative analysis using benchmark datasets and qualitative assessments through expert reviews validate the effectiveness of the framework. Results show improvements in detecting subtle biases compared to conventional methods, which often fail to highlight disparities in model responses to race, gender, and socio-political contexts. The framework also identifies biases arising from imbalances in training data and model architectures. Continuous user feedback ensures adaptability and refinement. This research underscores the importance of proactive bias mitigation strategies and calls for collaboration between policymakers, AI developers, and regulators. The proposed detection mechanisms enhance model transparency and support responsible LLM deployment in sensitive applications such as education, legal systems, and healthcare. Future work will focus on real-time bias monitoring and cross-linguistic generalization to improve fairness and inclusivity in AI-driven communication tools. 

---
# Constructions are Revealed in Word Distributions 

**Authors**: Joshua Rozner, Leonie Weissweiler, Kyle Mahowald, Cory Shain  

**Link**: [PDF](https://arxiv.org/pdf/2503.06048)  

**Abstract**: Construction grammar posits that constructions (form-meaning pairings) are acquired through experience with language (the distributional learning hypothesis). But how much information about constructions does this distribution actually contain? Corpus-based analyses provide some answers, but text alone cannot answer counterfactual questions about what caused a particular word to occur. For that, we need computable models of the distribution over strings -- namely, pretrained language models (PLMs). Here we treat a RoBERTa model as a proxy for this distribution and hypothesize that constructions will be revealed within it as patterns of statistical affinity. We support this hypothesis experimentally: many constructions are robustly distinguished, including (i) hard cases where semantically distinct constructions are superficially similar, as well as (ii) schematic constructions, whose "slots" can be filled by abstract word classes. Despite this success, we also provide qualitative evidence that statistical affinity alone may be insufficient to identify all constructions from text. Thus, statistical affinity is likely an important, but partial, signal available to learners. 

---
# Mitigating Memorization in LLMs using Activation Steering 

**Authors**: Manan Suri, Nishit Anand, Amisha Bhaskar  

**Link**: [PDF](https://arxiv.org/pdf/2503.06040)  

**Abstract**: The memorization of training data by Large Language Models (LLMs) poses significant risks, including privacy leaks and the regurgitation of copyrighted content. Activation steering, a technique that directly intervenes in model activations, has emerged as a promising approach for manipulating LLMs. In this work, we explore the effectiveness of activation steering in reducing memorization while preserving generalization capabilities. We conduct empirical evaluations using a controlled memorization benchmark of literary material and demonstrate that our method successfully suppresses memorized content with minimal degradation in model performance in Gemma. Additionally, we analyze the trade-offs between suppression effectiveness and linguistic fluency, highlighting the advantages and limitations of activation-based interventions. Our findings contribute to ongoing efforts in developing safer and more privacy-preserving LLMs by providing a practical and efficient mechanism to mitigate unintended memorization. 

---
# SmartBench: Is Your LLM Truly a Good Chinese Smartphone Assistant? 

**Authors**: Xudong Lu, Haohao Gao, Renshou Wu, Shuai Ren, Xiaoxin Chen, Hongsheng Li, Fangyuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.06029)  

**Abstract**: Large Language Models (LLMs) have become integral to daily life, especially advancing as intelligent assistants through on-device deployment on smartphones. However, existing LLM evaluation benchmarks predominantly focus on objective tasks like mathematics and coding in English, which do not necessarily reflect the practical use cases of on-device LLMs in real-world mobile scenarios, especially for Chinese users. To address these gaps, we introduce SmartBench, the first benchmark designed to evaluate the capabilities of on-device LLMs in Chinese mobile contexts. We analyze functionalities provided by representative smartphone manufacturers and divide them into five categories: text summarization, text Q\&A, information extraction, content creation, and notification management, further detailed into 20 specific tasks. For each task, we construct high-quality datasets comprising 50 to 200 question-answer pairs that reflect everyday mobile interactions, and we develop automated evaluation criteria tailored for these tasks. We conduct comprehensive evaluations of on-device LLMs and MLLMs using SmartBench and also assess their performance after quantized deployment on real smartphone NPUs. Our contributions provide a standardized framework for evaluating on-device LLMs in Chinese, promoting further development and optimization in this critical area. Code and data will be available at this https URL. 

---
# GenieBlue: Integrating both Linguistic and Multimodal Capabilities for Large Language Models on Mobile Devices 

**Authors**: Xudong Lu, Yinghao Chen, Renshou Wu, Haohao Gao, Xi Chen, Xue Yang, Xiangyu Zhao, Aojun Zhou, Fangyuan Li, Yafei Wen, Xiaoxin Chen, Shuai Ren, Hongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.06019)  

**Abstract**: Recent advancements in Multimodal Large Language Models (MLLMs) have enabled their deployment on mobile devices. However, challenges persist in maintaining strong language capabilities and ensuring hardware compatibility, both of which are crucial for user experience and practical deployment efficiency. In our deployment process, we observe that existing MLLMs often face performance degradation on pure language tasks, and the current NPU platforms on smartphones do not support the MoE architecture, which is commonly used to preserve pure language capabilities during multimodal training. To address these issues, we systematically analyze methods to maintain pure language capabilities during the training of MLLMs, focusing on both training data and model architecture aspects. Based on these analyses, we propose GenieBlue, an efficient MLLM structural design that integrates both linguistic and multimodal capabilities for LLMs on mobile devices. GenieBlue freezes the original LLM parameters during MLLM training to maintain pure language capabilities. It acquires multimodal capabilities by duplicating specific transformer blocks for full fine-tuning and integrating lightweight LoRA modules. This approach preserves language capabilities while achieving comparable multimodal performance through extensive training. Deployed on smartphone NPUs, GenieBlue demonstrates efficiency and practicality for applications on mobile devices. 

---
# Intent-Aware Self-Correction for Mitigating Social Biases in Large Language Models 

**Authors**: Panatchakorn Anantaprayoon, Masahiro Kaneko, Naoaki Okazaki  

**Link**: [PDF](https://arxiv.org/pdf/2503.06011)  

**Abstract**: Self-Correction based on feedback improves the output quality of Large Language Models (LLMs). Moreover, as Self-Correction functions like the slow and conscious System-2 thinking from cognitive psychology's perspective, it can potentially reduce LLMs' social biases. LLMs are sensitive to contextual ambiguities and inconsistencies; therefore, explicitly communicating their intentions during interactions when applying Self-Correction for debiasing is crucial. In this study, we demonstrate that clarifying intentions is essential for effectively reducing biases in LLMs through Self-Correction. We divide the components needed for Self-Correction into three parts: instruction, response, and feedback, and clarify intentions at each component. We incorporate an explicit debiasing prompt to convey the intention of bias mitigation from the instruction for response generation. In the response, we use Chain-of-Thought (CoT) to clarify the reasoning process. In the feedback, we define evaluation aspects necessary for debiasing and propose clear feedback through multi-aspect critiques and scoring. Through experiments, we demonstrate that self-correcting CoT responses obtained from a debiasing prompt based on multi-aspect feedback can reduce biased responses more robustly and consistently than the baselines. We also find the variation in debiasing efficacy when using models with different bias levels or separating models for response and feedback generation. 

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
# DETQUS: Decomposition-Enhanced Transformers for QUery-focused Summarization 

**Authors**: Yasir Khan, Xinlei Wu, Sangpil Youm, Justin Ho, Aryaan Shaikh, Jairo Garciga, Rohan Sharma, Bonnie J. Dorr  

**Link**: [PDF](https://arxiv.org/pdf/2503.05935)  

**Abstract**: Query-focused tabular summarization is an emerging task in table-to-text generation that synthesizes a summary response from tabular data based on user queries. Traditional transformer-based approaches face challenges due to token limitations and the complexity of reasoning over large tables. To address these challenges, we introduce DETQUS (Decomposition-Enhanced Transformers for QUery-focused Summarization), a system designed to improve summarization accuracy by leveraging tabular decomposition alongside a fine-tuned encoder-decoder model. DETQUS employs a large language model to selectively reduce table size, retaining only query-relevant columns while preserving essential information. This strategy enables more efficient processing of large tables and enhances summary quality. Our approach, equipped with table-based QA model Omnitab, achieves a ROUGE-L score of 0.4437, outperforming the previous state-of-the-art REFACTOR model (ROUGE-L: 0.422). These results highlight DETQUS as a scalable and effective solution for query-focused tabular summarization, offering a structured alternative to more complex architectures. 

---
# Training and Inference Efficiency of Encoder-Decoder Speech Models 

**Authors**: Piotr Żelasko, Kunal Dhawan, Daniel Galvez, Krishna C. Puvvada, Ankita Pasad, Nithin Rao Koluguri, Ke Hu, Vitaly Lavrukhin, Jagadeesh Balam, Boris Ginsburg  

**Link**: [PDF](https://arxiv.org/pdf/2503.05931)  

**Abstract**: Attention encoder-decoder model architecture is the backbone of several recent top performing foundation speech models: Whisper, Seamless, OWSM, and Canary-1B. However, the reported data and compute requirements for their training are prohibitive for many in the research community. In this work, we focus on the efficiency angle and ask the questions of whether we are training these speech models efficiently, and what can we do to improve? We argue that a major, if not the most severe, detrimental factor for training efficiency is related to the sampling strategy of sequential data. We show that negligence in mini-batch sampling leads to more than 50% computation being spent on padding. To that end, we study, profile, and optimize Canary-1B training to show gradual improvement in GPU utilization leading up to 5x increase in average batch sizes versus its original training settings. This in turn allows us to train an equivalent model using 4x less GPUs in the same wall time, or leverage the original resources and train it in 2x shorter wall time. Finally, we observe that the major inference bottleneck lies in the autoregressive decoder steps. We find that adjusting the model architecture to transfer model parameters from the decoder to the encoder results in a 3x inference speedup as measured by inverse real-time factor (RTFx) while preserving the accuracy and compute requirements for convergence. The training code and models will be available as open-source. 

---
# IDEA Prune: An Integrated Enlarge-and-Prune Pipeline in Generative Language Model Pretraining 

**Authors**: Yixiao Li, Xianzhi Du, Ajay Jaiswal, Tao Lei, Tuo Zhao, Chong Wang, Jianyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.05920)  

**Abstract**: Recent advancements in large language models have intensified the need for efficient and deployable models within limited inference budgets. Structured pruning pipelines have shown promise in token efficiency compared to training target-size models from scratch. In this paper, we advocate incorporating enlarged model pretraining, which is often ignored in previous works, into pruning. We study the enlarge-and-prune pipeline as an integrated system to address two critical questions: whether it is worth pretraining an enlarged model even when the model is never deployed, and how to optimize the entire pipeline for better pruned models. We propose an integrated enlarge-and-prune pipeline, which combines enlarge model training, pruning, and recovery under a single cosine annealing learning rate schedule. This approach is further complemented by a novel iterative structured pruning method for gradual parameter removal. The proposed method helps to mitigate the knowledge loss caused by the rising learning rate in naive enlarge-and-prune pipelines and enable effective redistribution of model capacity among surviving neurons, facilitating smooth compression and enhanced performance. We conduct comprehensive experiments on compressing 2.8B models to 1.3B with up to 2T tokens in pretraining. It demonstrates the integrated approach not only provides insights into the token efficiency of enlarged model pretraining but also achieves superior performance of pruned models. 

---
# From Style to Facts: Mapping the Boundaries of Knowledge Injection with Finetuning 

**Authors**: Eric Zhao, Pranjal Awasthi, Nika Haghtalab  

**Link**: [PDF](https://arxiv.org/pdf/2503.05919)  

**Abstract**: Finetuning provides a scalable and cost-effective means of customizing language models for specific tasks or response styles, with greater reliability than prompting or in-context learning. In contrast, the conventional wisdom is that injecting knowledge via finetuning results in brittle performance and poor generalization. We argue that the dichotomy of "task customization" (e.g., instruction tuning) and "knowledge injection" (e.g., teaching new facts) is a distinction without a difference. We instead identify concrete factors that explain the heterogeneous effectiveness observed with finetuning. To this end, we conduct a large-scale experimental study of finetuning the frontier Gemini v1.5 model family on a spectrum of datasets that are artificially engineered to interpolate between the strengths and failure modes of finetuning. Our findings indicate that question-answer training data formats provide much stronger knowledge generalization than document/article-style training data, numerical information can be harder for finetuning to retain than categorical information, and models struggle to apply finetuned knowledge during multi-step reasoning even when trained on similar examples -- all factors that render "knowledge injection" to be especially difficult, even after controlling for considerations like data augmentation and information volume. On the other hand, our findings also indicate that it is not fundamentally more difficult to finetune information about a real-world event than information about what a model's writing style should be. 

---
# MastermindEval: A Simple But Scalable Reasoning Benchmark 

**Authors**: Jonas Golde, Patrick Haller, Fabio Barth, Alan Akbik  

**Link**: [PDF](https://arxiv.org/pdf/2503.05891)  

**Abstract**: Recent advancements in large language models (LLMs) have led to remarkable performance across a wide range of language understanding and mathematical tasks. As a result, increasing attention has been given to assessing the true reasoning capabilities of LLMs, driving research into commonsense, numerical, logical, and qualitative reasoning. However, with the rapid progress of reasoning-focused models such as OpenAI's o1 and DeepSeek's R1, there has been a growing demand for reasoning benchmarks that can keep pace with ongoing model developments. In this paper, we introduce MastermindEval, a simple, scalable, and interpretable deductive reasoning benchmark inspired by the board game Mastermind. Our benchmark supports two evaluation paradigms: (1) agentic evaluation, in which the model autonomously plays the game, and (2) deductive reasoning evaluation, in which the model is given a pre-played game state with only one possible valid code to infer. In our experimental results we (1) find that even easy Mastermind instances are difficult for current models and (2) demonstrate that the benchmark is scalable to possibly more advanced models in the future Furthermore, we investigate possible reasons why models cannot deduce the final solution and find that current models are limited in deducing the concealed code as the number of statement to combine information from is increasing. 

---
# QG-SMS: Enhancing Test Item Analysis via Student Modeling and Simulation 

**Authors**: Bang Nguyen, Tingting Du, Mengxia Yu, Lawrence Angrave, Meng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.05888)  

**Abstract**: While the Question Generation (QG) task has been increasingly adopted in educational assessments, its evaluation remains limited by approaches that lack a clear connection to the educational values of test items. In this work, we introduce test item analysis, a method frequently used by educators to assess test question quality, into QG evaluation. Specifically, we construct pairs of candidate questions that differ in quality across dimensions such as topic coverage, item difficulty, item discrimination, and distractor efficiency. We then examine whether existing QG evaluation approaches can effectively distinguish these differences. Our findings reveal significant shortcomings in these approaches with respect to accurately assessing test item quality in relation to student performance. To address this gap, we propose a novel QG evaluation framework, QG-SMS, which leverages Large Language Model for Student Modeling and Simulation to perform test item analysis. As demonstrated in our extensive experiments and human evaluation study, the additional perspectives introduced by the simulated student profiles lead to a more effective and robust assessment of test items. 

---
# This Is Your Doge, If It Please You: Exploring Deception and Robustness in Mixture of LLMs 

**Authors**: Lorenz Wolf, Sangwoong Yoon, Ilija Bogunovic  

**Link**: [PDF](https://arxiv.org/pdf/2503.05856)  

**Abstract**: Mixture of large language model (LLMs) Agents (MoA) architectures achieve state-of-the-art performance on prominent benchmarks like AlpacaEval 2.0 by leveraging the collaboration of multiple LLMs at inference time. Despite these successes, an evaluation of the safety and reliability of MoA is missing. We present the first comprehensive study of MoA's robustness against deceptive LLM agents that deliberately provide misleading responses. We examine factors like the propagation of deceptive information, model size, and information availability, and uncover critical vulnerabilities. On AlpacaEval 2.0, the popular LLaMA 3.1-70B model achieves a length-controlled Win Rate (LC WR) of 49.2% when coupled with 3-layer MoA (6 LLM agents). However, we demonstrate that introducing only a $\textit{single}$ carefully-instructed deceptive agent into the MoA can reduce performance to 37.9%, effectively nullifying all MoA gains. On QuALITY, a multiple-choice comprehension task, the impact is also severe, with accuracy plummeting by a staggering 48.5%. Inspired in part by the historical Doge of Venice voting process, designed to minimize influence and deception, we propose a range of unsupervised defense mechanisms that recover most of the lost performance. 

---
# Extracting and Emulsifying Cultural Explanation to Improve Multilingual Capability of LLMs 

**Authors**: Hamin Koo, Jaehyung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.05846)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable success, but their English-centric training data limits performance in non-English languages, highlighting the need for enhancements in their multilingual capabilities. While some work on multilingual prompting methods handles non-English queries by utilizing English translations or restructuring them to more closely align with LLM reasoning patterns, these works often overlook the importance of cultural context, limiting their effectiveness. To address this limitation, we propose EMCEI, a simple yet effective approach that improves LLMs' multilingual capabilities by incorporating cultural context for more accurate and appropriate responses. Specifically, EMCEI follows a two-step process that first extracts relevant cultural context from the LLM's parametric knowledge via prompting. Then, EMCEI employs an LLM-as-Judge mechanism to select the most appropriate response by balancing cultural relevance and reasoning ability. Experiments on diverse multilingual benchmarks show that EMCEI outperforms existing baselines, demonstrating its effectiveness in handling multilingual queries with LLMs. 

---
# FedMentalCare: Towards Privacy-Preserving Fine-Tuned LLMs to Analyze Mental Health Status Using Federated Learning Framework 

**Authors**: S M Sarwar  

**Link**: [PDF](https://arxiv.org/pdf/2503.05786)  

**Abstract**: With the increasing prevalence of mental health conditions worldwide, AI-powered chatbots and conversational agents have emerged as accessible tools to support mental health. However, deploying Large Language Models (LLMs) in mental healthcare applications raises significant privacy concerns, especially regarding regulations like HIPAA and GDPR. In this work, we propose FedMentalCare, a privacy-preserving framework that leverages Federated Learning (FL) combined with Low-Rank Adaptation (LoRA) to fine-tune LLMs for mental health analysis. We investigate the performance impact of varying client data volumes and model architectures (e.g., MobileBERT and MiniLM) in FL environments. Our framework demonstrates a scalable, privacy-aware approach for deploying LLMs in real-world mental healthcare scenarios, addressing data security and computational efficiency challenges. 

---
# Medical Hallucinations in Foundation Models and Their Impact on Healthcare 

**Authors**: Yubin Kim, Hyewon Jeong, Shan Chen, Shuyue Stella Li, Mingyu Lu, Kumail Alhamoud, Jimin Mun, Cristina Grau, Minseok Jung, Rodrigo Gameiro, Lizhou Fan, Eugene Park, Tristan Lin, Joonsik Yoon, Wonjin Yoon, Maarten Sap, Yulia Tsvetkov, Paul Liang, Xuhai Xu, Xin Liu, Daniel McDuff, Hyeonhoon Lee, Hae Won Park, Samir Tulebaev, Cynthia Breazeal  

**Link**: [PDF](https://arxiv.org/pdf/2503.05777)  

**Abstract**: Foundation Models that are capable of processing and generating multi-modal data have transformed AI's role in medicine. However, a key limitation of their reliability is hallucination, where inaccurate or fabricated information can impact clinical decisions and patient safety. We define medical hallucination as any instance in which a model generates misleading medical content. This paper examines the unique characteristics, causes, and implications of medical hallucinations, with a particular focus on how these errors manifest themselves in real-world clinical scenarios. Our contributions include (1) a taxonomy for understanding and addressing medical hallucinations, (2) benchmarking models using medical hallucination dataset and physician-annotated LLM responses to real medical cases, providing direct insight into the clinical impact of hallucinations, and (3) a multi-national clinician survey on their experiences with medical hallucinations. Our results reveal that inference techniques such as Chain-of-Thought (CoT) and Search Augmented Generation can effectively reduce hallucination rates. However, despite these improvements, non-trivial levels of hallucination persist. These findings underscore the ethical and practical imperative for robust detection and mitigation strategies, establishing a foundation for regulatory policies that prioritize patient safety and maintain clinical integrity as AI becomes more integrated into healthcare. The feedback from clinicians highlights the urgent need for not only technical advances but also for clearer ethical and regulatory guidelines to ensure patient safety. A repository organizing the paper resources, summaries, and additional information is available at this https URL hallucination. 

---
# Graph Masked Language Models 

**Authors**: Aarush Sinha, OM Kumar CU  

**Link**: [PDF](https://arxiv.org/pdf/2503.05763)  

**Abstract**: Language Models (LMs) are integral to Natural Language Processing (NLP), yet their interaction with structured knowledge graphs (KGs) remains an open research challenge. While Graph Neural Networks (GNNs) excel at capturing graph structures, they struggle with textual feature representation compared to pretrained LMs. To bridge this gap, we propose \textbf{Graph Masked Language Models (GMLM)} for node classification tasks. Our approach introduces two key innovations: a \textit{semantic masking strategy} that selectively masks nodes based on their structural importance, ensuring critical graph components contribute effectively to learning, and a \textit{soft masking mechanism} that generates interpolated node representations, enabling smoother information retention and improved gradient flow. Our dual-branch model architecture fuses structural graph information with contextual embeddings via a multi-layer fusion network. Extensive experiments on six node classification benchmarks demonstrate that GMLM not only achieves state-of-the-art (SOTA) performance but also enhances robustness and stability across datasets. 

---
# CSTRL: Context-Driven Sequential Transfer Learning for Abstractive Radiology Report Summarization 

**Authors**: Mst. Fahmida Sultana Naznin, Adnan Ibney Faruq, Mostafa Rifat Tazwar, Md Jobayer, Md. Mehedi Hasan Shawon, Md Rakibul Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2503.05750)  

**Abstract**: A radiology report comprises several sections, including the Findings and Impression of the diagnosis. Automatically generating the Impression from the Findings is crucial for reducing radiologists' workload and improving diagnostic accuracy. Pretrained models that excel in common abstractive summarization problems encounter challenges when applied to specialized medical domains largely due to the complex terminology and the necessity for accurate clinical context. Such tasks in medical domains demand extracting core information, avoiding context shifts, and maintaining proper flow. Misuse of medical terms can lead to drastic clinical errors. To address these issues, we introduce a sequential transfer learning that ensures key content extraction and coherent summarization. Sequential transfer learning often faces challenges like initial parameter decay and knowledge loss, which we resolve with the Fisher matrix regularization. Using MIMIC-CXR and Open-I datasets, our model, CSTRL-Context-driven Sequential TRansfer Learning-achieved state-of-the-art performance, showing 56.2% improvement in BLEU-1, 40.5% in BLEU-2, 84.3% in BLEU-3, 28.9% in ROUGE-1, 41.0% in ROUGE-2 and 26.5% in ROGUE-3 score over benchmark studies. We also analyze factual consistency scores while preserving the medical context. Our code is publicly available at TBA. 

---
# What Are They Filtering Out? A Survey of Filtering Strategies for Harm Reduction in Pretraining Datasets 

**Authors**: Marco Antonio Stranisci, Christian Hardmeier  

**Link**: [PDF](https://arxiv.org/pdf/2503.05721)  

**Abstract**: Data filtering strategies are a crucial component to develop safe Large Language Models (LLM), since they support the removal of harmful contents from pretraining datasets. There is a lack of research on the actual impact of these strategies on vulnerable groups to discrimination, though, and their effectiveness has not been yet systematically addressed. In this paper we present a benchmark study of data filtering strategies for harm reduction aimed at providing a systematic overview on these approaches. We survey 55 technical reports of English LMs and LLMs to identify the existing filtering strategies in literature and implement an experimental setting to test their impact against vulnerable groups. Our results show that the positive impact that strategies have in reducing harmful contents from documents has the side effect of increasing the underrepresentation of vulnerable groups to discrimination in datasets. 

---
# VisBias: Measuring Explicit and Implicit Social Biases in Vision Language Models 

**Authors**: Jen-tse Huang, Jiantong Qin, Jianping Zhang, Youliang Yuan, Wenxuan Wang, Jieyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.07575)  

**Abstract**: This research investigates both explicit and implicit social biases exhibited by Vision-Language Models (VLMs). The key distinction between these bias types lies in the level of awareness: explicit bias refers to conscious, intentional biases, while implicit bias operates subconsciously. To analyze explicit bias, we directly pose questions to VLMs related to gender and racial differences: (1) Multiple-choice questions based on a given image (e.g., "What is the education level of the person in the image?") (2) Yes-No comparisons using two images (e.g., "Is the person in the first image more educated than the person in the second image?") For implicit bias, we design tasks where VLMs assist users but reveal biases through their responses: (1) Image description tasks: Models are asked to describe individuals in images, and we analyze disparities in textual cues across demographic groups. (2) Form completion tasks: Models draft a personal information collection form with 20 attributes, and we examine correlations among selected attributes for potential biases. We evaluate Gemini-1.5, GPT-4V, GPT-4o, LLaMA-3.2-Vision and LLaVA-v1.6. Our code and data are publicly available at this https URL. 

---
# Optimizing Test-Time Compute via Meta Reinforcement Fine-Tuning 

**Authors**: Yuxiao Qu, Matthew Y. R. Yang, Amrith Setlur, Lewis Tunstall, Edward Emanuel Beeching, Ruslan Salakhutdinov, Aviral Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2503.07572)  

**Abstract**: Training models to effectively use test-time compute is crucial for improving the reasoning performance of LLMs. Current methods mostly do so via fine-tuning on search traces or running RL with 0/1 outcome reward, but do these approaches efficiently utilize test-time compute? Would these approaches continue to scale as the budget improves? In this paper, we try to answer these questions. We formalize the problem of optimizing test-time compute as a meta-reinforcement learning (RL) problem, which provides a principled perspective on spending test-time compute. This perspective enables us to view the long output stream from the LLM as consisting of several episodes run at test time and leads us to use a notion of cumulative regret over output tokens as a way to measure the efficacy of test-time compute. Akin to how RL algorithms can best tradeoff exploration and exploitation over training, minimizing cumulative regret would also provide the best balance between exploration and exploitation in the token stream. While we show that state-of-the-art models do not minimize regret, one can do so by maximizing a dense reward bonus in conjunction with the outcome 0/1 reward RL. This bonus is the ''progress'' made by each subsequent block in the output stream, quantified by the change in the likelihood of eventual success. Using these insights, we develop Meta Reinforcement Fine-Tuning, or MRT, a new class of fine-tuning methods for optimizing test-time compute. MRT leads to a 2-3x relative gain in performance and roughly a 1.5x gain in token efficiency for math reasoning compared to outcome-reward RL. 

---
# Building English ASR model with regional language support 

**Authors**: Purvi Agrawal, Vikas Joshi, Bharati Patidar, Ankur Gupta, Rupesh Kumar Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2503.07522)  

**Abstract**: In this paper, we present a novel approach to developing an English Automatic Speech Recognition (ASR) system that can effectively handle Hindi queries, without compromising its performance on English. We propose a novel acoustic model (AM), referred to as SplitHead with Attention (SHA) model, features shared hidden layers across languages and language-specific projection layers combined via a self-attention mechanism. This mechanism estimates the weight for each language based on input data and weighs the corresponding language-specific projection layers accordingly. Additionally, we propose a language modeling approach that interpolates n-gram models from both English and transliterated Hindi text corpora. Our results demonstrate the effectiveness of our approach, with a 69.3% and 5.7% relative reduction in word error rate on Hindi and English test sets respectively when compared to a monolingual English model. 

---
# GRITHopper: Decomposition-Free Multi-Hop Dense Retrieval 

**Authors**: Justus-Jonas Erker, Nils Reimers, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2503.07519)  

**Abstract**: Decomposition-based multi-hop retrieval methods rely on many autoregressive steps to break down complex queries, which breaks end-to-end differentiability and is computationally expensive. Decomposition-free methods tackle this, but current decomposition-free approaches struggle with longer multi-hop problems and generalization to out-of-distribution data. To address these challenges, we introduce GRITHopper-7B, a novel multi-hop dense retrieval model that achieves state-of-the-art performance on both in-distribution and out-of-distribution benchmarks. GRITHopper combines generative and representational instruction tuning by integrating causal language modeling with dense retrieval training. Through controlled studies, we find that incorporating additional context after the retrieval process, referred to as post-retrieval language modeling, enhances dense retrieval performance. By including elements such as final answers during training, the model learns to better contextualize and retrieve relevant information. GRITHopper-7B offers a robust, scalable, and generalizable solution for multi-hop dense retrieval, and we release it to the community for future research and applications requiring multi-hop reasoning and retrieval capabilities. 

---
# Sometimes the Model doth Preach: Quantifying Religious Bias in Open LLMs through Demographic Analysis in Asian Nations 

**Authors**: Hari Shankar, Vedanta S P, Tejas Cavale, Ponnurangam Kumaraguru, Abhijnan Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2503.07510)  

**Abstract**: Large Language Models (LLMs) are capable of generating opinions and propagating bias unknowingly, originating from unrepresentative and non-diverse data collection. Prior research has analysed these opinions with respect to the West, particularly the United States. However, insights thus produced may not be generalized in non-Western populations. With the widespread usage of LLM systems by users across several different walks of life, the cultural sensitivity of each generated output is of crucial interest. Our work proposes a novel method that quantitatively analyzes the opinions generated by LLMs, improving on previous work with regards to extracting the social demographics of the models. Our method measures the distance from an LLM's response to survey respondents, through Hamming Distance, to infer the demographic characteristics reflected in the model's outputs. We evaluate modern, open LLMs such as Llama and Mistral on surveys conducted in various global south countries, with a focus on India and other Asian nations, specifically assessing the model's performance on surveys related to religious tolerance and identity. Our analysis reveals that most open LLMs match a single homogeneous profile, varying across different countries/territories, which in turn raises questions about the risks of LLMs promoting a hegemonic worldview, and undermining perspectives of different minorities. Our framework may also be useful for future research investigating the complex intersection between training data, model architecture, and the resulting biases reflected in LLM outputs, particularly concerning sensitive topics like religious tolerance and identity. 

---
# Is a Good Foundation Necessary for Efficient Reinforcement Learning? The Computational Role of the Base Model in Exploration 

**Authors**: Dylan J. Foster, Zakaria Mhammedi, Dhruv Rohatgi  

**Link**: [PDF](https://arxiv.org/pdf/2503.07453)  

**Abstract**: Language model alignment (or, reinforcement learning) techniques that leverage active exploration -- deliberately encouraging the model to produce diverse, informative responses -- offer the promise of super-human capabilities. However, current understanding of algorithm design primitives for computationally efficient exploration with language models is limited. To better understand how to leverage access to powerful pre-trained generative models to improve the efficiency of exploration, we introduce a new computational framework for RL with language models, in which the learner interacts with the model through a sampling oracle. Focusing on the linear softmax model parameterization, we provide new results that reveal the computational-statistical tradeoffs of efficient exploration:
1. Necessity of coverage: Coverage refers to the extent to which the pre-trained model covers near-optimal responses -- a form of hidden knowledge. We show that coverage, while not necessary for data efficiency, lower bounds the runtime of any algorithm in our framework.
2. Inference-time exploration: We introduce a new algorithm, SpannerSampling, which obtains optimal data efficiency and is computationally efficient whenever the pre-trained model enjoys sufficient coverage, matching our lower bound. SpannerSampling leverages inference-time computation with the pre-trained model to reduce the effective search space for exploration.
3. Insufficiency of training-time interventions: We contrast the result above by showing that training-time interventions that produce proper policies cannot achieve similar guarantees in polynomial time.
4. Computational benefits of multi-turn exploration: Finally, we show that under additional representational assumptions, one can achieve improved runtime (replacing sequence-level coverage with token-level coverage) through multi-turn exploration. 

---
# VizTrust: A Visual Analytics Tool for Capturing User Trust Dynamics in Human-AI Communication 

**Authors**: Xin Wang, Stephanie Tulk Jesso, Sadamori Kojaku, David M Neyens, Min Sun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.07279)  

**Abstract**: Trust plays a fundamental role in shaping the willingness of users to engage and collaborate with artificial intelligence (AI) systems. Yet, measuring user trust remains challenging due to its complex and dynamic nature. While traditional survey methods provide trust levels for long conversations, they fail to capture its dynamic evolution during ongoing interactions. Here, we present VizTrust, which addresses this challenge by introducing a real-time visual analytics tool that leverages a multi-agent collaboration system to capture and analyze user trust dynamics in human-agent communication. Built on established human-computer trust scales-competence, integrity, benevolence, and predictability-, VizTrust enables stakeholders to observe trust formation as it happens, identify patterns in trust development, and pinpoint specific interaction elements that influence trust. Our tool offers actionable insights into human-agent trust formation and evolution in real time through a dashboard, supporting the design of adaptive conversational agents that responds effectively to user trust signals. 

---
# WISE: A World Knowledge-Informed Semantic Evaluation for Text-to-Image Generation 

**Authors**: Yuwei Niu, Munan Ning, Mengren Zheng, Bin Lin, Peng Jin, Jiaqi Liao, Kunpeng Ning, Bin Zhu, Li Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2503.07265)  

**Abstract**: Text-to-Image (T2I) models are capable of generating high-quality artistic creations and visual content. However, existing research and evaluation standards predominantly focus on image realism and shallow text-image alignment, lacking a comprehensive assessment of complex semantic understanding and world knowledge integration in text to image generation. To address this challenge, we propose $\textbf{WISE}$, the first benchmark specifically designed for $\textbf{W}$orld Knowledge-$\textbf{I}$nformed $\textbf{S}$emantic $\textbf{E}$valuation. WISE moves beyond simple word-pixel mapping by challenging models with 1000 meticulously crafted prompts across 25 sub-domains in cultural common sense, spatio-temporal reasoning, and natural science. To overcome the limitations of traditional CLIP metric, we introduce $\textbf{WiScore}$, a novel quantitative metric for assessing knowledge-image alignment. Through comprehensive testing of 20 models (10 dedicated T2I models and 10 unified multimodal models) using 1,000 structured prompts spanning 25 subdomains, our findings reveal significant limitations in their ability to effectively integrate and apply world knowledge during image generation, highlighting critical pathways for enhancing knowledge incorporation and application in next-generation T2I models. Code and data are available at this https URL. 

---
# Multi-Modal 3D Mesh Reconstruction from Images and Text 

**Authors**: Melvin Reka, Tessa Pulli, Markus Vincze  

**Link**: [PDF](https://arxiv.org/pdf/2503.07190)  

**Abstract**: 6D object pose estimation for unseen objects is essential in robotics but traditionally relies on trained models that require large datasets, high computational costs, and struggle to generalize. Zero-shot approaches eliminate the need for training but depend on pre-existing 3D object models, which are often impractical to obtain. To address this, we propose a language-guided few-shot 3D reconstruction method, reconstructing a 3D mesh from few input images. In the proposed pipeline, receives a set of input images and a language query. A combination of GroundingDINO and Segment Anything Model outputs segmented masks from which a sparse point cloud is reconstructed with VGGSfM. Subsequently, the mesh is reconstructed with the Gaussian Splatting method SuGAR. In a final cleaning step, artifacts are removed, resulting in the final 3D mesh of the queried object. We evaluate the method in terms of accuracy and quality of the geometry and texture. Furthermore, we study the impact of imaging conditions such as viewing angle, number of input images, and image overlap on 3D object reconstruction quality, efficiency, and computational scalability. 

---
# PoseLess: Depth-Free Vision-to-Joint Control via Direct Image Mapping with VLM 

**Authors**: Alan Dao, Dinh Bach Vu, Tuan Le Duc Anh, Bui Quang Huy  

**Link**: [PDF](https://arxiv.org/pdf/2503.07111)  

**Abstract**: This paper introduces PoseLess, a novel framework for robot hand control that eliminates the need for explicit pose estimation by directly mapping 2D images to joint angles using tokenized representations. Our approach leverages synthetic training data generated through randomized joint configurations, enabling zero-shot generalization to real-world scenarios and cross-morphology transfer from robotic to human hands. By tokenizing visual inputs and employing a transformer-based decoder, PoseLess achieves robust, low-latency control while addressing challenges such as depth ambiguity and data scarcity. Experimental results demonstrate competitive performance in joint angle prediction accuracy without relying on any human-labelled dataset. 

---
# ProjectEval: A Benchmark for Programming Agents Automated Evaluation on Project-Level Code Generation 

**Authors**: Kaiyuan Liu, Youcheng Pan, Jing Li, Daojing He, Yang Xiang, Yexing Du, Tianrun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2503.07010)  

**Abstract**: Recently, LLM agents have made rapid progress in improving their programming capabilities. However, existing benchmarks lack the ability to automatically evaluate from users' perspective, and also lack the explainability of the results of LLM agents' code generation capabilities. Thus, we introduce ProjectEval, a new benchmark for LLM agents project-level code generation's automated evaluation by simulating user interaction. ProjectEval is constructed by LLM with human reviewing. It has three different level inputs of natural languages or code skeletons. ProjectEval can evaluate the generated projects by user interaction simulation for execution, and by code similarity through existing objective indicators. Through ProjectEval, we find that systematic engineering project code, overall understanding of the project and comprehensive analysis capability are the keys for LLM agents to achieve practical projects. Our findings and benchmark provide valuable insights for developing more effective programming agents that can be deployed in future real-world production. 

---
# Silent Hazards of Token Reduction in Vision-Language Models: The Hidden Impact on Consistency 

**Authors**: Yizheng Sun, Hao Li, Chang Xu, Chenghua Lin, Riza Batista-Navarro, Jingyuan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2503.06794)  

**Abstract**: Vision language models (VLMs) have excelled in visual reasoning but often incur high computational costs. One key reason is the redundancy of visual tokens. Although recent token reduction methods claim to achieve minimal performance loss, our extensive experiments reveal that token reduction can substantially alter a model's output distribution, leading to changes in prediction patterns that standard metrics such as accuracy loss do not fully capture. Such inconsistencies are especially concerning for practical applications where system stability is critical. To investigate this phenomenon, we analyze how token reduction influences the energy distribution of a VLM's internal representations using a lower-rank approximation via Singular Value Decomposition (SVD). Our results show that changes in the Inverse Participation Ratio of the singular value spectrum are strongly correlated with the model's consistency after token reduction. Based on these insights, we propose LoFi--a training-free visual token reduction method that utilizes the leverage score from SVD for token pruning. Experimental evaluations demonstrate that LoFi not only reduces computational costs with minimal performance degradation but also significantly outperforms state-of-the-art methods in terms of output consistency. 

---
# Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models 

**Authors**: Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Yao Hu, Shaohui Lin  

**Link**: [PDF](https://arxiv.org/pdf/2503.06749)  

**Abstract**: DeepSeek-R1-Zero has successfully demonstrated the emergence of reasoning capabilities in LLMs purely through Reinforcement Learning (RL). Inspired by this breakthrough, we explore how RL can be utilized to enhance the reasoning capability of MLLMs. However, direct training with RL struggles to activate complex reasoning capabilities such as questioning and reflection in MLLMs, due to the absence of substantial high-quality multimodal reasoning data. To address this issue, we propose the reasoning MLLM, Vision-R1, to improve multimodal reasoning capability. Specifically, we first construct a high-quality multimodal CoT dataset without human annotations by leveraging an existing MLLM and DeepSeek-R1 through modality bridging and data filtering to obtain a 200K multimodal CoT dataset, Vision-R1-cold dataset. It serves as cold-start initialization data for Vision-R1. To mitigate the optimization challenges caused by overthinking after cold start, we propose Progressive Thinking Suppression Training (PTST) strategy and employ Group Relative Policy Optimization (GRPO) with the hard formatting result reward function to gradually refine the model's ability to learn correct and complex reasoning processes on a 10K multimodal math dataset. Comprehensive experiments show our model achieves an average improvement of $\sim$6% across various multimodal math reasoning benchmarks. Vision-R1-7B achieves a 73.5% accuracy on the widely used MathVista benchmark, which is only 0.4% lower than the leading reasoning model, OpenAI O1. The datasets and code will be released in: this https URL . 

---
# DependEval: Benchmarking LLMs for Repository Dependency Understanding 

**Authors**: Junjia Du, Yadi Liu, Hongcheng Guo, Jiawei Wang, Haojian Huang, Yunyi Ni, Zhoujun Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.06689)  

**Abstract**: While large language models (LLMs) have shown considerable promise in code generation, real-world software development demands advanced repository-level reasoning. This includes understanding dependencies, project structures, and managing multi-file changes. However, the ability of LLMs to effectively comprehend and handle complex code repositories has yet to be fully explored. To address challenges, we introduce a hierarchical benchmark designed to evaluate repository dependency understanding (DependEval). Benchmark is based on 15,576 repositories collected from real-world websites. It evaluates models on three core tasks: Dependency Recognition, Repository Construction, and Multi-file Editing, across 8 programming languages from actual code repositories. Our evaluation of over 25 LLMs reveals substantial performance gaps and provides valuable insights into repository-level code understanding. 

---
# Attention, Please! PixelSHAP Reveals What Vision-Language Models Actually Focus On 

**Authors**: Roni Goldshmidt  

**Link**: [PDF](https://arxiv.org/pdf/2503.06670)  

**Abstract**: Interpretability in Vision-Language Models (VLMs) is crucial for trust, debugging, and decision-making in high-stakes applications. We introduce PixelSHAP, a model-agnostic framework extending Shapley-based analysis to structured visual entities. Unlike previous methods focusing on text prompts, PixelSHAP applies to vision-based reasoning by systematically perturbing image objects and quantifying their influence on a VLM's response. PixelSHAP requires no model internals, operating solely on input-output pairs, making it compatible with open-source and commercial models. It supports diverse embedding-based similarity metrics and scales efficiently using optimization techniques inspired by Shapley-based methods. We validate PixelSHAP in autonomous driving, highlighting its ability to enhance interpretability. Key challenges include segmentation sensitivity and object occlusion. Our open-source implementation facilitates further research. 

---
# Evaluating and Aligning Human Economic Risk Preferences in LLMs 

**Authors**: Jiaxin Liu, Yi Yang, Kar Yan Tam  

**Link**: [PDF](https://arxiv.org/pdf/2503.06646)  

**Abstract**: Large Language Models (LLMs) are increasingly used in decision-making scenarios that involve risk assessment, yet their alignment with human economic rationality remains unclear. In this study, we investigate whether LLMs exhibit risk preferences consistent with human expectations across different personas. Specifically, we assess whether LLM-generated responses reflect appropriate levels of risk aversion or risk-seeking behavior based on individual's persona. Our results reveal that while LLMs make reasonable decisions in simplified, personalized risk contexts, their performance declines in more complex economic decision-making tasks. To address this, we propose an alignment method designed to enhance LLM adherence to persona-specific risk preferences. Our approach improves the economic rationality of LLMs in risk-related applications, offering a step toward more human-aligned AI decision-making. 

---
# Is Your Benchmark (Still) Useful? Dynamic Benchmarking for Code Language Models 

**Authors**: Batu Guan, Xiao Wu, Yuanyuan Yuan, Shaohua Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.06643)  

**Abstract**: In this paper, we tackle a critical challenge in model evaluation: how to keep code benchmarks useful when models might have already seen them during training. We introduce a novel solution, dynamic benchmarking framework, to address this challenge. Given a code understanding or reasoning benchmark, our framework dynamically transforms each input, i.e., programs, with various semantic-preserving mutations to build a syntactically new while semantically identical benchmark. We evaluated ten popular language models on our dynamic benchmarks. Our evaluation reveals several interesting or surprising findings: (1) all models perform significantly worse than before, (2) the ranking between some models shifts dramatically, and (3) our dynamic benchmarks can resist against the data contamination problem. 

---
# Revisiting Early Detection of Sexual Predators via Turn-level Optimization 

**Authors**: Jinmyeong An, Sangwon Ryu, Heejin Do, Yunsu Kim, Jungseul Ok, Gary Geunbae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.06627)  

**Abstract**: Online grooming is a severe social threat where sexual predators gradually entrap child victims with subtle and gradual manipulation. Therefore, timely intervention for online grooming is critical for proactive protection. However, previous methods fail to determine the optimal intervention points (i.e., jump to conclusions) as they rely on chat-level risk labels by causing weak supervision of risky utterances. For timely detection, we propose speed control reinforcement learning (SCoRL) (The code and supplementary materials are available at this https URL), incorporating a practical strategy derived from luring communication theory (LCT). To capture the predator's turn-level entrapment, we use a turn-level risk label based on the LCT. Then, we design a novel speed control reward function that balances the trade-off between speed and accuracy based on turn-level risk label; thus, SCoRL can identify the optimal intervention moment. In addition, we introduce a turn-level metric for precise evaluation, identifying limitations in previously used chat-level metrics. Experimental results show that SCoRL effectively preempted online grooming, offering a more proactive and timely solution. Further analysis reveals that our method enhances performance while intuitively identifying optimal early intervention points. 

---
# Multimodal Programming in Computer Science with Interactive Assistance Powered by Large Language Model 

**Authors**: Rajan Das Gupta, Md. Tanzib Hosain, M. F. Mridha, Salah Uddin Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2503.06552)  

**Abstract**: LLM chatbot interfaces allow students to get instant, interactive assistance with homework, but doing so carelessly may not advance educational objectives. In this study, an interactive homework help system based on DeepSeek R1 is developed and first implemented for students enrolled in a large computer science beginning programming course. In addition to an assist button in a well-known code editor, our assistant also has a feedback option in our command-line automatic evaluator. It wraps student work in a personalized prompt that advances our educational objectives without offering answers straight away. We have discovered that our assistant can recognize students' conceptual difficulties and provide ideas, plans, and template code in pedagogically appropriate ways. However, among other mistakes, it occasionally incorrectly labels the correct student code as incorrect or encourages students to use correct-but-lesson-inappropriate approaches, which can lead to long and frustrating journeys for the students. After discussing many development and deployment issues, we provide our conclusions and future actions. 

---
# Less is More: Adaptive Program Repair with Bug Localization and Preference Learning 

**Authors**: Zhenlong Dai, Bingrui Chen, Zhuoluo Zhao, Xiu Tang, Sai Wu, Chang Yao, Zhipeng Gao, Jingyuan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.06510)  

**Abstract**: Automated Program Repair (APR) is a task to automatically generate patches for the buggy code. However, most research focuses on generating correct patches while ignoring the consistency between the fixed code and the original buggy code. How to conduct adaptive bug fixing and generate patches with minimal modifications have seldom been investigated. To bridge this gap, we first introduce a novel task, namely AdaPR (Adaptive Program Repair). We then propose a two-stage approach AdaPatcher (Adaptive Patch Generator) to enhance program repair while maintaining the consistency. In the first stage, we utilize a Bug Locator with self-debug learning to accurately pinpoint bug locations. In the second stage, we train a Program Modifier to ensure consistency between the post-modified fixed code and the pre-modified buggy code. The Program Modifier is enhanced with a location-aware repair learning strategy to generate patches based on identified buggy lines, a hybrid training strategy for selective reference and an adaptive preference learning to prioritize fewer changes. The experimental results show that our approach outperforms a set of baselines by a large margin, validating the effectiveness of our two-stage framework for the newly proposed AdaPR task. 

---
# SKG-LLM: Developing a Mathematical Model for Stroke Knowledge Graph Construction Using Large Language Models 

**Authors**: Ali Sarabadani, Kheirolah Rahsepar Fard, Hamid Dalvand  

**Link**: [PDF](https://arxiv.org/pdf/2503.06475)  

**Abstract**: The purpose of this study is to introduce SKG-LLM. A knowledge graph (KG) is constructed from stroke-related articles using mathematical and large language models (LLMs). SKG-LLM extracts and organizes complex relationships from the biomedical literature, using it to increase the accuracy and depth of KG in stroke research. In the proposed method, GPT-4 was used for data pre-processing, and the extraction of embeddings was also done by GPT-4 in the whole KG construction process. The performance of the proposed model was tested with two evaluation criteria: Precision and Recall. For further validation of the proposed model, GPT-4 was used. Compared with Wikidata and WN18RR, the proposed KG-LLM approach performs better, especially in precision and recall. By including GPT-4 in the preprocessing process, the SKG-LLM model achieved a precision score of 0.906 and a recall score of 0.923. Expert reviews further improved the results and increased precision to 0.923 and recall to 0.918. The knowledge graph constructed by SKG-LLM contains 2692 nodes and 5012 edges, which are 13 distinct types of nodes and 24 types of edges. 

---
# HuixiangDou2: A Robustly Optimized GraphRAG Approach 

**Authors**: Huanjun Kong, Zhefan Wang, Chenyang Wang, Zhe Ma, Nanqing Dong  

**Link**: [PDF](https://arxiv.org/pdf/2503.06474)  

**Abstract**: Large Language Models (LLMs) perform well on familiar queries but struggle with specialized or emerging topics. Graph-based Retrieval-Augmented Generation (GraphRAG) addresses this by structuring domain knowledge as a graph for dynamic retrieval. However, existing pipelines involve complex engineering workflows, making it difficult to isolate the impact of individual components. Evaluating retrieval effectiveness is also challenging due to dataset overlap with LLM pretraining data. In this work, we introduce HuixiangDou2, a robustly optimized GraphRAG framework. Specifically, we leverage the effectiveness of dual-level retrieval and optimize its performance in a 32k context for maximum precision, and compare logic-based retrieval and dual-level retrieval to enhance overall functionality. Our implementation includes comparative experiments on a test set, where Qwen2.5-7B-Instruct initially underperformed. With our approach, the score improved significantly from 60 to 74.5, as illustrated in the Figure. Experiments on domain-specific datasets reveal that dual-level retrieval enhances fuzzy matching, while logic-form retrieval improves structured reasoning. Furthermore, we propose a multi-stage verification mechanism to improve retrieval robustness without increasing computational cost. Empirical results show significant accuracy gains over baselines, highlighting the importance of adaptive retrieval. To support research and adoption, we release HuixiangDou2 as an open-source resource this https URL. 

---
# Think Twice, Click Once: Enhancing GUI Grounding via Fast and Slow Systems 

**Authors**: Fei Tang, Yongliang Shen, Hang Zhang, Siqi Chen, Guiyang Hou, Wenqi Zhang, Wenqiao Zhang, Kaitao Song, Weiming Lu, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2503.06470)  

**Abstract**: Humans can flexibly switch between different modes of thinking based on task complexity: from rapid intuitive judgments to in-depth analytical understanding. However, current Graphical User Interface (GUI) grounding systems which locate interface elements based on natural language instructions rely solely on immediate prediction without reasoning, struggling to understand complex interface layouts with nested structures and hierarchical relationships, limiting their effectiveness on complex interfaces. Inspired by human dual-system cognition, we present Focus, a novel GUI grounding framework that combines fast prediction with systematic analysis. The framework dynamically switches between rapid and deliberate processing through an adaptive system switching based on task complexity, optimizing both efficiency and accuracy. Focus decomposes grounding into progressive stages: interface summarization, visual focused analysis, and precise coordinate prediction. This structured decomposition enables systematic understanding of both interface layouts and visual relationships. Extensive experiments show that Focus achieves state-of-the-art performance using only 300K of the training data with a 2B parameter model compared to existing approaches. Focus demonstrates superior performance particularly in complex GUI scenarios, achieving 77.4% average accuracy on ScreenSpot and 13.3% on the more challenging ScreenSpot-Pro. Our analysis reveals the effectiveness of this dual-system approach while demonstrating its potential for improving complex GUI interaction scenarios. 

---
# TI-JEPA: An Innovative Energy-based Joint Embedding Strategy for Text-Image Multimodal Systems 

**Authors**: Khang H. N. Vo, Duc P. T. Nguyen, Thong Nguyen, Tho T. Quan  

**Link**: [PDF](https://arxiv.org/pdf/2503.06380)  

**Abstract**: This paper focuses on multimodal alignment within the realm of Artificial Intelligence, particularly in text and image modalities. The semantic gap between the textual and visual modality poses a discrepancy problem towards the effectiveness of multi-modalities fusion. Therefore, we introduce Text-Image Joint Embedding Predictive Architecture (TI-JEPA), an innovative pre-training strategy that leverages energy-based model (EBM) framework to capture complex cross-modal relationships. TI-JEPA combines the flexibility of EBM in self-supervised learning to facilitate the compatibility between textual and visual elements. Through extensive experiments across multiple benchmarks, we demonstrate that TI-JEPA achieves state-of-the-art performance on multimodal sentiment analysis task (and potentially on a wide range of multimodal-based tasks, such as Visual Question Answering), outperforming existing pre-training methodologies. Our findings highlight the potential of using energy-based framework in advancing multimodal fusion and suggest significant improvements for downstream applications. 

---
# General Scales Unlock AI Evaluation with Explanatory and Predictive Power 

**Authors**: Lexin Zhou, Lorenzo Pacchiardi, Fernando Martínez-Plumed, Katherine M. Collins, Yael Moros-Daval, Seraphina Zhang, Qinlin Zhao, Yitian Huang, Luning Sun, Jonathan E. Prunty, Zongqian Li, Pablo Sánchez-García, Kexin Jiang Chen, Pablo A. M. Casares, Jiyun Zu, John Burden, Behzad Mehrbakhsh, David Stillwell, Manuel Cebrian, Jindong Wang, Peter Henderson, Sherry Tongshuang Wu, Patrick C. Kyllonen, Lucy Cheke, Xing Xie, José Hernández-Orallo  

**Link**: [PDF](https://arxiv.org/pdf/2503.06378)  

**Abstract**: Ensuring safe and effective use of AI requires understanding and anticipating its performance on novel tasks, from advanced scientific challenges to transformed workplace activities. So far, benchmarking has guided progress in AI, but it has offered limited explanatory and predictive power for general-purpose AI systems, given the low transferability across diverse tasks. In this paper, we introduce general scales for AI evaluation that can explain what common AI benchmarks really measure, extract ability profiles of AI systems, and predict their performance for new task instances, in- and out-of-distribution. Our fully-automated methodology builds on 18 newly-crafted rubrics that place instance demands on general scales that do not saturate. Illustrated for 15 large language models and 63 tasks, high explanatory power is unleashed from inspecting the demand and ability profiles, bringing insights on the sensitivity and specificity exhibited by different benchmarks, and how knowledge, metacognition and reasoning are affected by model size, chain-of-thought and distillation. Surprisingly, high predictive power at the instance level becomes possible using these demand levels, providing superior estimates over black-box baseline predictors based on embeddings or finetuning, especially in out-of-distribution settings (new tasks and new benchmarks). The scales, rubrics, battery, techniques and results presented here represent a major step for AI evaluation, underpinning the reliable deployment of AI in the years ahead. 

---
# Phraselette: A Poet's Procedural Palette 

**Authors**: Alex Calderwood, John Joon Young Chung, Yuqian Sun, Melissa Roemmele, Max Kreminski  

**Link**: [PDF](https://arxiv.org/pdf/2503.06335)  

**Abstract**: According to the recently introduced theory of artistic support tools, creativity support tools exert normative influences over artistic production, instantiating a normative ground that shapes both the process and product of artistic expression. We argue that the normative ground of most existing automated writing tools is misaligned with writerly values and identify a potential alternative frame-material writing support-for experimental poetry tools that flexibly support the finding, processing, transforming, and shaping of text(s). Based on this frame, we introduce Phraselette, an artistic material writing support interface that helps experimental poets search for words and phrases. To provide material writing support, Phraselette is designed to counter the dominant mode of automated writing tools, while offering language model affordances in line with writerly values. We further report on an extended expert evaluation involving 10 published poets that indicates support for both our framing of material writing support and for Phraselette itself. 

---
# Advancing Autonomous Vehicle Intelligence: Deep Learning and Multimodal LLM for Traffic Sign Recognition and Robust Lane Detection 

**Authors**: Chandan Kumar Sah, Ankit Kumar Shaw, Xiaoli Lian, Arsalan Shahid Baig, Tuopu Wen, Kun Jiang, Mengmeng Yang, Diange Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.06313)  

**Abstract**: Autonomous vehicles (AVs) require reliable traffic sign recognition and robust lane detection capabilities to ensure safe navigation in complex and dynamic environments. This paper introduces an integrated approach combining advanced deep learning techniques and Multimodal Large Language Models (MLLMs) for comprehensive road perception. For traffic sign recognition, we systematically evaluate ResNet-50, YOLOv8, and RT-DETR, achieving state-of-the-art performance of 99.8% with ResNet-50, 98.0% accuracy with YOLOv8, and achieved 96.6% accuracy in RT-DETR despite its higher computational complexity. For lane detection, we propose a CNN-based segmentation method enhanced by polynomial curve fitting, which delivers high accuracy under favorable conditions. Furthermore, we introduce a lightweight, Multimodal, LLM-based framework that directly undergoes instruction tuning using small yet diverse datasets, eliminating the need for initial pretraining. This framework effectively handles various lane types, complex intersections, and merging zones, significantly enhancing lane detection reliability by reasoning under adverse conditions. Despite constraints in available training resources, our multimodal approach demonstrates advanced reasoning capabilities, achieving a Frame Overall Accuracy (FRM) of 53.87%, a Question Overall Accuracy (QNS) of 82.83%, lane detection accuracies of 99.6% in clear conditions and 93.0% at night, and robust performance in reasoning about lane invisibility due to rain (88.4%) or road degradation (95.6%). The proposed comprehensive framework markedly enhances AV perception reliability, thus contributing significantly to safer autonomous driving across diverse and challenging road scenarios. 

---
# Critical Foreign Policy Decisions (CFPD)-Benchmark: Measuring Diplomatic Preferences in Large Language Models 

**Authors**: Benjamin Jensen, Ian Reynolds, Yasir Atalan, Michael Garcia, Austin Woo, Anthony Chen, Trevor Howarth  

**Link**: [PDF](https://arxiv.org/pdf/2503.06263)  

**Abstract**: As national security institutions increasingly integrate Artificial Intelligence (AI) into decision-making and content generation processes, understanding the inherent biases of large language models (LLMs) is crucial. This study presents a novel benchmark designed to evaluate the biases and preferences of seven prominent foundation models-Llama 3.1 8B Instruct, Llama 3.1 70B Instruct, GPT-4o, Gemini 1.5 Pro-002, Mixtral 8x22B, Claude 3.5 Sonnet, and Qwen2 72B-in the context of international relations (IR). We designed a bias discovery study around core topics in IR using 400-expert crafted scenarios to analyze results from our selected models. These scenarios focused on four topical domains including: military escalation, military and humanitarian intervention, cooperative behavior in the international system, and alliance dynamics. Our analysis reveals noteworthy variation among model recommendations based on scenarios designed for the four tested domains. Particularly, Qwen2 72B, Gemini 1.5 Pro-002 and Llama 3.1 8B Instruct models offered significantly more escalatory recommendations than Claude 3.5 Sonnet and GPT-4o models. All models exhibit some degree of country-specific biases, often recommending less escalatory and interventionist actions for China and Russia compared to the United States and the United Kingdom. These findings highlight the necessity for controlled deployment of LLMs in high-stakes environments, emphasizing the need for domain-specific evaluations and model fine-tuning to align with institutional objectives. 

---
# A Noise-Robust Turn-Taking System for Real-World Dialogue Robots: A Field Experiment 

**Authors**: Koji Inoue, Yuki Okafuji, Jun Baba, Yoshiki Ohira, Katsuya Hyodo, Tatsuya Kawahara  

**Link**: [PDF](https://arxiv.org/pdf/2503.06241)  

**Abstract**: Turn-taking is a crucial aspect of human-robot interaction, directly influencing conversational fluidity and user engagement. While previous research has explored turn-taking models in controlled environments, their robustness in real-world settings remains underexplored. In this study, we propose a noise-robust voice activity projection (VAP) model, based on a Transformer architecture, to enhance real-time turn-taking in dialogue robots. To evaluate the effectiveness of the proposed system, we conducted a field experiment in a shopping mall, comparing the VAP system with a conventional cloud-based speech recognition system. Our analysis covered both subjective user evaluations and objective behavioral analysis. The results showed that the proposed system significantly reduced response latency, leading to a more natural conversation where both the robot and users responded faster. The subjective evaluations suggested that faster responses contribute to a better interaction experience. 

---
# Explainable Synthetic Image Detection through Diffusion Timestep Ensembling 

**Authors**: Yixin Wu, Feiran Zhang, Tianyuan Shi, Ruicheng Yin, Zhenghua Wang, Zhenliang Gan, Xiaohua Wang, Changze Lv, Xiaoqing Zheng, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.06201)  

**Abstract**: Recent advances in diffusion models have enabled the creation of deceptively real images, posing significant security risks when misused. In this study, we reveal that natural and synthetic images exhibit distinct differences in the high-frequency domains of their Fourier power spectra after undergoing iterative noise perturbations through an inverse multi-step denoising process, suggesting that such noise can provide additional discriminative information for identifying synthetic images. Based on this observation, we propose a novel detection method that amplifies these differences by progressively adding noise to the original images across multiple timesteps, and train an ensemble of classifiers on these noised images. To enhance human comprehension, we introduce an explanation generation and refinement module to identify flaws located in AI-generated images. Additionally, we construct two new datasets, GenHard and GenExplain, derived from the GenImage benchmark, providing detection samples of greater difficulty and high-quality rationales for fake images. Extensive experiments show that our method achieves state-of-the-art performance with 98.91% and 95.89% detection accuracy on regular and harder samples, increasing a minimal of 2.51% and 3.46% compared to baselines. Furthermore, our method also generalizes effectively to images generated by other diffusion models. Our code and datasets will be made publicly available. 

---
# AF-KAN: Activation Function-Based Kolmogorov-Arnold Networks for Efficient Representation Learning 

**Authors**: Hoang-Thang Ta, Anh Tran  

**Link**: [PDF](https://arxiv.org/pdf/2503.06112)  

**Abstract**: Kolmogorov-Arnold Networks (KANs) have inspired numerous works exploring their applications across a wide range of scientific problems, with the potential to replace Multilayer Perceptrons (MLPs). While many KANs are designed using basis and polynomial functions, such as B-splines, ReLU-KAN utilizes a combination of ReLU functions to mimic the structure of B-splines and take advantage of ReLU's speed. However, ReLU-KAN is not built for multiple inputs, and its limitations stem from ReLU's handling of negative values, which can restrict feature extraction. To address these issues, we introduce Activation Function-Based Kolmogorov-Arnold Networks (AF-KAN), expanding ReLU-KAN with various activations and their function combinations. This novel KAN also incorporates parameter reduction methods, primarily attention mechanisms and data normalization, to enhance performance on image classification datasets. We explore different activation functions, function combinations, grid sizes, and spline orders to validate the effectiveness of AF-KAN and determine its optimal configuration. In the experiments, AF-KAN significantly outperforms MLP, ReLU-KAN, and other KANs with the same parameter count. It also remains competitive even when using fewer than 6 to 10 times the parameters while maintaining the same network structure. However, AF-KAN requires a longer training time and consumes more FLOPs. The repository for this work is available at this https URL. 

---
# A Novel Trustworthy Video Summarization Algorithm Through a Mixture of LoRA Experts 

**Authors**: Wenzhuo Du, Gerun Wang, Guancheng Chen, Hang Zhao, Xin Li, Jian Gao  

**Link**: [PDF](https://arxiv.org/pdf/2503.06064)  

**Abstract**: With the exponential growth of user-generated content on video-sharing platforms, the challenge of facilitating efficient searching and browsing of videos has garnered significant attention. To enhance users' ability to swiftly locate and review pertinent videos, the creation of concise and informative video summaries has become increasingly important. Video-llama is an effective tool for generating video summarization, but it cannot effectively unify and optimize the modeling of temporal and spatial features and requires a lot of computational resources and time. Therefore, we propose MiLoRA-ViSum to more efficiently capture complex temporal dynamics and spatial relationships inherent in video data and to control the number of parameters for training. By extending traditional Low-Rank Adaptation (LoRA) into a sophisticated mixture-of-experts paradigm, MiLoRA-ViSum incorporates a dual temporal-spatial adaptation mechanism tailored specifically for video summarization tasks. This approach dynamically integrates specialized LoRA experts, each fine-tuned to address distinct temporal or spatial dimensions. Extensive evaluations of the VideoXum and ActivityNet datasets demonstrate that MiLoRA-ViSum achieves the best summarization performance compared to state-of-the-art models, while maintaining significantly lower computational costs. The proposed mixture-of-experts strategy, combined with the dual adaptation mechanism, highlights the model's potential to enhance video summarization capabilities, particularly in large-scale applications requiring both efficiency and precision. 

---
# DSGBench: A Diverse Strategic Game Benchmark for Evaluating LLM-based Agents in Complex Decision-Making Environments 

**Authors**: Wenjie Tang, Yuan Zhou, Erqiang Xu, Keyan Cheng, Minne Li, Liquan Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2503.06047)  

**Abstract**: Large Language Model~(LLM) based agents have been increasingly popular in solving complex and dynamic tasks, which requires proper evaluation systems to assess their capabilities. Nevertheless, existing benchmarks usually either focus on single-objective tasks or use overly broad assessing metrics, failing to provide a comprehensive inspection of the actual capabilities of LLM-based agents in complicated decision-making tasks. To address these issues, we introduce DSGBench, a more rigorous evaluation platform for strategic decision-making. Firstly, it incorporates six complex strategic games which serve as ideal testbeds due to their long-term and multi-dimensional decision-making demands and flexibility in customizing tasks of various difficulty levels or multiple targets. Secondly, DSGBench employs a fine-grained evaluation scoring system which examines the decision-making capabilities by looking into the performance in five specific dimensions and offering a comprehensive assessment in a well-designed way. Furthermore, DSGBench also incorporates an automated decision-tracking mechanism which enables in-depth analysis of agent behaviour patterns and the changes in their strategies. We demonstrate the advances of DSGBench by applying it to multiple popular LLM-based agents and our results suggest that DSGBench provides valuable insights in choosing LLM-based agents as well as improving their future development. DSGBench is available at this https URL. 

---
# Rank-R1: Enhancing Reasoning in LLM-based Document Rerankers via Reinforcement Learning 

**Authors**: Shengyao Zhuang, Xueguang Ma, Bevan Koopman, Jimmy Lin, Guido Zuccon  

**Link**: [PDF](https://arxiv.org/pdf/2503.06034)  

**Abstract**: In this paper, we introduce Rank-R1, a novel LLM-based reranker that performs reasoning over both the user query and candidate documents before performing the ranking task. Existing document reranking methods based on large language models (LLMs) typically rely on prompting or fine-tuning LLMs to order or label candidate documents according to their relevance to a query. For Rank-R1, we use a reinforcement learning algorithm along with only a small set of relevance labels (without any reasoning supervision) to enhance the reasoning ability of LLM-based rerankers. Our hypothesis is that adding reasoning capabilities to the rerankers can improve their relevance assessement and ranking capabilities. Our experiments on the TREC DL and BRIGHT datasets show that Rank-R1 is highly effective, especially for complex queries. In particular, we find that Rank-R1 achieves effectiveness on in-domain datasets at par with that of supervised fine-tuning methods, but utilizing only 18\% of the training data used by the fine-tuning methods. We also find that the model largely outperforms zero-shot and supervised fine-tuning when applied to out-of-domain datasets featuring complex queries, especially when a 14B-size model is used. Finally, we qualitatively observe that Rank-R1's reasoning process improves the explainability of the ranking results, opening new opportunities for search engine results presentation and fruition. 

---
# Psycholinguistic Analyses in Software Engineering Text: A Systematic Literature Review 

**Authors**: Amirali Sajadi, Kostadin Damevski, Preetha Chatterjee  

**Link**: [PDF](https://arxiv.org/pdf/2503.05992)  

**Abstract**: Context: A deeper understanding of human factors in software engineering (SE) is essential for improving team collaboration, decision-making, and productivity. Communication channels like code reviews and chats provide insights into developers' psychological and emotional states. While large language models excel at text analysis, they often lack transparency and precision. Psycholinguistic tools like Linguistic Inquiry and Word Count (LIWC) offer clearer, interpretable insights into cognitive and emotional processes exhibited in text. Despite its wide use in SE research, no comprehensive review of LIWC's use has been conducted. Objective: We examine the importance of psycholinguistic tools, particularly LIWC, and provide a thorough analysis of its current and potential future applications in SE research. Methods: We conducted a systematic review of six prominent databases, identifying 43 SE-related papers using LIWC. Our analysis focuses on five research questions. Results: Our findings reveal a wide range of applications, including analyzing team communication to detect developer emotions and personality, developing ML models to predict deleted Stack Overflow posts, and more recently comparing AI-generated and human-written text. LIWC has been primarily used with data from project management platforms (e.g., GitHub) and Q&A forums (e.g., Stack Overflow). Key BSE concepts include Communication, Organizational Climate, and Positive Psychology. 26 of 43 papers did not formally evaluate LIWC. Concerns were raised about some limitations, including difficulty handling SE-specific vocabulary. Conclusion: We highlight the potential of psycholinguistic tools and their limitations, and present new use cases for advancing the research of human factors in SE (e.g., bias in human-LLM conversations). 

---
# Bimodal Connection Attention Fusion for Speech Emotion Recognition 

**Authors**: Jiachen Luo, Huy Phan, Lin Wang, Joshua D. Reiss  

**Link**: [PDF](https://arxiv.org/pdf/2503.05858)  

**Abstract**: Multi-modal emotion recognition is challenging due to the difficulty of extracting features that capture subtle emotional differences. Understanding multi-modal interactions and connections is key to building effective bimodal speech emotion recognition systems. In this work, we propose Bimodal Connection Attention Fusion (BCAF) method, which includes three main modules: the interactive connection network, the bimodal attention network, and the correlative attention network. The interactive connection network uses an encoder-decoder architecture to model modality connections between audio and text while leveraging modality-specific features. The bimodal attention network enhances semantic complementation and exploits intra- and inter-modal interactions. The correlative attention network reduces cross-modal noise and captures correlations between audio and text. Experiments on the MELD and IEMOCAP datasets demonstrate that the proposed BCAF method outperforms existing state-of-the-art baselines. 

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
# Where is my Glass Slipper? AI, Poetry and Art 

**Authors**: Anastasios P. Pagiaslis  

**Link**: [PDF](https://arxiv.org/pdf/2503.05781)  

**Abstract**: This literature review interrogates the intersections between artificial intelligence, poetry, and art, offering a comprehensive exploration of both historical evolution and current debates in digital creative practices. It traces the development of computer-generated poetry from early template-based systems to generative models, critically assessing evaluative frameworks such as adaptations of the Turing Test, the FACE model, and ProFTAP. It also examines how these frameworks endeavour to measure creativity, semantic coherence, and cultural relevance in AI-generated texts, whilst highlighting the persistent challenges in replicating the nuance of human poetic expression.
The review contributes a Marketing Theory discussion that deconstructs the figurative marketing narratives employed by AI companies, which utilise sanitised language and anthropomorphic metaphors to humanise their technologies. This discussion reveals the reductive nature of such narratives and underscores the tension between algorithmic precision and the realities of human this http URL review also incorporates an auto-ethnographic account that offers a self-reflexive commentary on its own composition. By acknowledging the use of AI in crafting this review, the auto-ethnographic account destabilises conventional notions of authorship and objectivity, resonating with deconstruction and challenging logocentric assumptions in academic discourse.
Ultimately, the review calls for a re-evaluation of creative processes that recognises the interdependence of technological innovation and human subjectivity. It advocates for interdisciplinary dialogue addressing ethical, cultural, and philosophical concerns, while reimagining the boundaries of artistic production. 

---
# DreamNet: A Multimodal Framework for Semantic and Emotional Analysis of Sleep Narratives 

**Authors**: Tapasvi Panchagnula  

**Link**: [PDF](https://arxiv.org/pdf/2503.05778)  

**Abstract**: Dream narratives provide a unique window into human cognition and emotion, yet their systematic analysis using artificial intelligence has been underexplored. We introduce DreamNet, a novel deep learning framework that decodes semantic themes and emotional states from textual dream reports, optionally enhanced with REM-stage EEG data. Leveraging a transformer-based architecture with multimodal attention, DreamNet achieves 92.1% accuracy and 88.4% F1-score in text-only mode (DNet-T) on a curated dataset of 1,500 anonymized dream narratives, improving to 99.0% accuracy and 95.2% F1-score with EEG integration (DNet-M). Strong dream-emotion correlations (e.g., falling-anxiety, r = 0.91, p < 0.01) highlight its potential for mental health diagnostics, cognitive science, and personalized therapy. This work provides a scalable tool, a publicly available enriched dataset, and a rigorous methodology, bridging AI and psychological research. 

---
# Effect of Gender Fair Job Description on Generative AI Images 

**Authors**: Finn Böckling, Jan Marquenie, Ingo Siegert  

**Link**: [PDF](https://arxiv.org/pdf/2503.05769)  

**Abstract**: STEM fields are traditionally male-dominated, with gender biases shaping perceptions of job accessibility. This study analyzed gender representation in STEM occupation images generated by OpenAI DALL-E 3 \& Black Forest FLUX.1 using 150 prompts in three linguistic forms: German generic masculine, German pair form, and English. As control, 20 pictures of social occupations were generated as well. Results revealed significant male bias across all forms, with the German pair form showing reduced bias but still overrepresenting men for the STEM-Group and mixed results for the Group of Social Occupations. These findings highlight generative AI's role in reinforcing societal biases, emphasizing the need for further discussion on diversity (in AI). Further aspects analyzed are age-distribution and ethnic diversity. 

---
# Uncertainty-Aware Fusion: An Ensemble Framework for Mitigating Hallucinations in Large Language Models 

**Authors**: Prasenjit Dey, Srujana Merugu, Sivaramakrishnan Kaveri  

**Link**: [PDF](https://arxiv.org/pdf/2503.05757)  

**Abstract**: Large Language Models (LLMs) are known to hallucinate and generate non-factual outputs which can undermine user trust. Traditional methods to directly mitigate hallucinations, such as representation editing and contrastive decoding, often require additional training data and involve high implementation complexity. While ensemble-based approaches harness multiple LLMs to tap into the "wisdom of crowds", these methods overlook uncertainties in individual model responses. Recent studies reveal that uncertainty estimation can enable LLMs to self-assess the likelihood of generating hallucinations. In this work, we focus on factoid question answering (QA) and observe that LLMs accuracy and self-assessment capabilities vary widely with different models excelling in different scenarios. Leveraging this insight, we propose Uncertainty-Aware Fusion (UAF), an ensemble framework to reduces hallucinations by strategically combining multiple LLM based on their accuracy and self-assessment abilities. Empirical results on several public benchmark datasets show that UAF outperforms state-of-the-art hallucination mitigation methods by $8\%$ in factual accuracy, while either narrowing or surpassing the performance gap with GPT-4. 

---
# ChatWise: AI-Powered Engaging Conversations for Enhancing Senior Cognitive Wellbeing 

**Authors**: Zhengbang Yang, Zhuangdi Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2503.05740)  

**Abstract**: Cognitive health in older adults presents a growing challenge. While conversational interventions show feasibility in improving cognitive wellness, human caregiver resources remain overburdened. AI-based methods have shown promise in providing conversational support, yet existing work is limited to implicit strategy while lacking multi-turn support tailored to seniors. We improve prior art with an LLM-driven chatbot named ChatWise for older adults. It follows dual-level conversation reasoning at the inference phase to provide engaging companionship. ChatWise thrives in long-turn conversations, in contrast to conventional LLMs that primarily excel in short-turn exchanges. Grounded experiments show that ChatWise significantly enhances simulated users' cognitive and emotional status, including those with Mild Cognitive Impairment. 

---
# That is Unacceptable: the Moral Foundations of Canceling 

**Authors**: Soda Marem Lo, Oscar Araque, Rajesh Sharma, Marco Antonio Stranisci  

**Link**: [PDF](https://arxiv.org/pdf/2503.05720)  

**Abstract**: Canceling is a morally-driven phenomenon that hinders the development of safe social media platforms and contributes to ideological polarization. To address this issue we present the Canceling Attitudes Detection (CADE) dataset, an annotated corpus of canceling incidents aimed at exploring the factors of disagreements in evaluating people canceling attitudes on social media. Specifically, we study the impact of annotators' morality in their perception of canceling, showing that morality is an independent axis for the explanation of disagreement on this phenomenon. Annotator's judgments heavily depend on the type of controversial events and involved celebrities. This shows the need to develop more event-centric datasets to better understand how harms are perpetrated in social media and to develop more aware technologies for their detection. 

---
# Beyond English: Unveiling Multilingual Bias in LLM Copyright Compliance 

**Authors**: Yupeng Chen, Xiaoyu Zhang, Yixian Huang, Qian Xie  

**Link**: [PDF](https://arxiv.org/pdf/2503.05713)  

**Abstract**: Large Language Models (LLMs) have raised significant concerns regarding the fair use of copyright-protected content. While prior studies have examined the extent to which LLMs reproduce copyrighted materials, they have predominantly focused on English, neglecting multilingual dimensions of copyright protection. In this work, we investigate multilingual biases in LLM copyright protection by addressing two key questions: (1) Do LLMs exhibit bias in protecting copyrighted works across languages? (2) Is it easier to elicit copyrighted content using prompts in specific languages? To explore these questions, we construct a dataset of popular song lyrics in English, French, Chinese, and Korean and systematically probe seven LLMs using prompts in these languages. Our findings reveal significant imbalances in LLMs' handling of copyrighted content, both in terms of the language of the copyrighted material and the language of the prompt. These results highlight the need for further research and development of more robust, language-agnostic copyright protection mechanisms to ensure fair and consistent protection across languages. 

---
# Russo-Ukrainian war disinformation detection in suspicious Telegram channels 

**Authors**: Anton Bazdyrev  

**Link**: [PDF](https://arxiv.org/pdf/2503.05707)  

**Abstract**: The paper proposes an advanced approach for identifying disinformation on Telegram channels related to the Russo-Ukrainian conflict, utilizing state-of-the-art (SOTA) deep learning techniques and transfer learning. Traditional methods of disinformation detection, often relying on manual verification or rule-based systems, are increasingly inadequate in the face of rapidly evolving propaganda tactics and the massive volume of data generated daily. To address these challenges, the proposed system employs deep learning algorithms, including LLM models, which are fine-tuned on a custom dataset encompassing verified disinformation and legitimate content. The paper's findings indicate that this approach significantly outperforms traditional machine learning techniques, offering enhanced contextual understanding and adaptability to emerging disinformation strategies. 

---
# OPTIC: Optimizing Patient-Provider Triaging & Improving Communications in Clinical Operations using GPT-4 Data Labeling and Model Distillation 

**Authors**: Alberto Santamaria-Pang, Frank Tuan, Ross Campbell, Cindy Zhang, Ankush Jindal, Roopa Surapur, Brad Holloman, Deanna Hanisch, Rae Buckley, Carisa Cooney, Ivan Tarapov, Kimberly S. Peairs, Brian Hasselfeld, Peter Greene  

**Link**: [PDF](https://arxiv.org/pdf/2503.05701)  

**Abstract**: The COVID-19 pandemic has accelerated the adoption of telemedicine and patient messaging through electronic medical portals (patient medical advice requests, or PMARs). While these platforms enhance patient access to healthcare, they have also increased the burden on healthcare providers due to the surge in PMARs. This study seeks to develop an efficient tool for message triaging to reduce physician workload and improve patient-provider communication. We developed OPTIC (Optimizing Patient-Provider Triaging & Improving Communications in Clinical Operations), a powerful message triaging tool that utilizes GPT-4 for data labeling and BERT for model distillation. The study used a dataset of 405,487 patient messaging encounters from Johns Hopkins Medicine between January and June 2020. High-quality labeled data was generated through GPT-4-based prompt engineering, which was then used to train a BERT model to classify messages as "Admin" or "Clinical." The BERT model achieved 88.85% accuracy on the test set validated by GPT-4 labeling, with a sensitivity of 88.29%, specificity of 89.38%, and an F1 score of 0.8842. BERTopic analysis identified 81 distinct topics within the test data, with over 80% accuracy in classifying 58 topics. The system was successfully deployed through Epic's Nebula Cloud Platform, demonstrating its practical effectiveness in healthcare settings. 

---
