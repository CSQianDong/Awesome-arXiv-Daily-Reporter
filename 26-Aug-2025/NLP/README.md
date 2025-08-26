# MIRAGE: Scaling Test-Time Inference with Parallel Graph-Retrieval-Augmented Reasoning Chains 

**Authors**: Kaiwen Wei, Rui Shan, Dongsheng Zou, Jianzhong Yang, Bi Zhao, Junnan Zhu, Jiang Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2508.18260)  

**Abstract**: Large reasoning models (LRMs) have shown significant progress in test-time scaling through chain-of-thought prompting. Current approaches like search-o1 integrate retrieval augmented generation (RAG) into multi-step reasoning processes but rely on a single, linear reasoning chain while incorporating unstructured textual information in a flat, context-agnostic manner. As a result, these approaches can lead to error accumulation throughout the reasoning chain, which significantly limits its effectiveness in medical question-answering (QA) tasks where both accuracy and traceability are critical requirements. To address these challenges, we propose MIRAGE (Multi-chain Inference with Retrieval-Augmented Graph Exploration), a novel test-time scalable reasoning framework that performs dynamic multi-chain inference over structured medical knowledge graphs. Specifically, MIRAGE 1) decomposes complex queries into entity-grounded sub-questions, 2) executes parallel inference chains, 3) retrieves evidence adaptively via neighbor expansion and multi-hop traversal, and 4) integrates answers using cross-chain verification to resolve contradictions. Experiments on three medical QA benchmarks (GenMedGPT-5k, CMCQA, and ExplainCPE) show that MIRAGE consistently outperforms GPT-4o, Tree-of-Thought variants, and other retrieval-augmented baselines in both automatic and human evaluations. Additionally, MIRAGE improves interpretability by generating explicit reasoning chains that trace each factual claim to concrete chains within the knowledge graph, making it well-suited for complex medical reasoning scenarios. The code will be available for further research. 

---
# From BERT to LLMs: Comparing and Understanding Chinese Classifier Prediction in Language Models 

**Authors**: ZiqiZhang, Jianfei Ma, Emmanuele Chersoni, Jieshun You, Zhaoxin Feng  

**Link**: [PDF](https://arxiv.org/pdf/2508.18253)  

**Abstract**: Classifiers are an important and defining feature of the Chinese language, and their correct prediction is key to numerous educational applications. Yet, whether the most popular Large Language Models (LLMs) possess proper knowledge the Chinese classifiers is an issue that has largely remain unexplored in the Natural Language Processing (NLP) literature.
To address such a question, we employ various masking strategies to evaluate the LLMs' intrinsic ability, the contribution of different sentence elements, and the working of the attention mechanisms during prediction. Besides, we explore fine-tuning for LLMs to enhance the classifier performance.
Our findings reveal that LLMs perform worse than BERT, even with fine-tuning. The prediction, as expected, greatly benefits from the information about the following noun, which also explains the advantage of models with a bidirectional attention mechanism such as BERT. 

---
# Demographic Biases and Gaps in the Perception of Sexism in Large Language Models 

**Authors**: Judith Tavarez-Rodríguez, Fernando Sánchez-Vega, A. Pastor López-Monroy  

**Link**: [PDF](https://arxiv.org/pdf/2508.18245)  

**Abstract**: The use of Large Language Models (LLMs) has proven to be a tool that could help in the automatic detection of sexism. Previous studies have shown that these models contain biases that do not accurately reflect reality, especially for minority groups. Despite various efforts to improve the detection of sexist content, this task remains a significant challenge due to its subjective nature and the biases present in automated models. We explore the capabilities of different LLMs to detect sexism in social media text using the EXIST 2024 tweet dataset. It includes annotations from six distinct profiles for each tweet, allowing us to evaluate to what extent LLMs can mimic these groups' perceptions in sexism detection. Additionally, we analyze the demographic biases present in the models and conduct a statistical analysis to identify which demographic characteristics (age, gender) contribute most effectively to this task. Our results show that, while LLMs can to some extent detect sexism when considering the overall opinion of populations, they do not accurately replicate the diversity of perceptions among different demographic groups. This highlights the need for better-calibrated models that account for the diversity of perspectives across different populations. 

---
# MTalk-Bench: Evaluating Speech-to-Speech Models in Multi-Turn Dialogues via Arena-style and Rubrics Protocols 

**Authors**: Yuhao Du, Qianwei Huang, Guo Zhu, Zhanchen Dai, Sunian Chen, Qiming Zhu, Yuhao Zhang, Li Zhou, Benyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.18240)  

**Abstract**: The rapid advancement of speech-to-speech (S2S) large language models (LLMs) has significantly improved real-time spoken interaction. However, current evaluation frameworks remain inadequate for assessing performance in complex, multi-turn dialogues. To address this, we introduce MTalk-Bench, a multi-turn S2S benchmark covering three core dimensions: Semantic Information, Paralinguistic Information, and Ambient Sound. Each dimension includes nine realistic scenarios, along with targeted tasks to assess specific capabilities such as reasoning. Our dual-method evaluation framework combines Arena-style evaluation (pairwise comparison) and Rubrics-based evaluation (absolute scoring) for relative and absolute assessment. The benchmark includes both model and human outputs, evaluated by human evaluators and LLMs. Experimental results reveal two sets of findings. Overall performance of S2S LLMs: (1) models excel at semantic information processing yet underperform on paralinguistic information and ambient sounds perception; (2) models typically regain coherence by increasing response length, sacrificing efficiency in multi-turn dialogues; (3) modality-aware, task-specific designs outperform brute scaling. Evaluation framework and reliability: (1) Arena and Rubrics yield consistent, complementary rankings, but reliable distinctions emerge only when performance gaps are large; (2) LLM-as-a-judge aligns with humans when gaps are clear or criteria explicit, but exhibits position and length biases and is reliable on nonverbal evaluation only with text annotations. These results highlight current limitations in S2S evaluation and the need for more robust, speech-aware assessment frameworks. 

---
# Better Language Model-Based Judging Reward Modeling through Scaling Comprehension Boundaries 

**Authors**: Meiling Ning, Zhongbao Zhang, Junda Ye, Jiabao Guo, Qingyuan Guan  

**Link**: [PDF](https://arxiv.org/pdf/2508.18212)  

**Abstract**: The emergence of LM-based judging reward modeling, represented by generative reward models, has successfully made reinforcement learning from AI feedback (RLAIF) efficient and scalable. To further advance this paradigm, we propose a core insight: this form of reward modeling shares fundamental formal consistency with natural language inference (NLI), a core task in natural language understanding. This reframed perspective points to a key path for building superior reward models: scaling the model's comprehension boundaries. Pursuing this path, exploratory experiments on NLI tasks demonstrate that the slot prediction masked language models (MLMs) incorporating contextual explanations achieve significantly better performance compared to mainstream autoregressive models. Based on this key finding, we propose ESFP-RM, a two-stage LM-based judging reward model that utilizes an explanation based slot framework for prediction to fully leverage the advantages of MLMs. Extensive experiments demonstrate that in both reinforcement learning from human feedback (RLHF) and out-of-distribution (OOD) scenarios, the ESFP-RM framework delivers more stable and generalizable reward signals compared to generative reward models. 

---
# Why Synthetic Isn't Real Yet: A Diagnostic Framework for Contact Center Dialogue Generation 

**Authors**: Rishikesh Devanathan, Varun Nathan, Ayush Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2508.18210)  

**Abstract**: Synthetic transcript generation is critical in contact center domains, where privacy and data scarcity limit model training and evaluation. Unlike prior synthetic dialogue generation work on open-domain or medical dialogues, contact center conversations are goal-oriented, role-asymmetric, and behaviorally complex, featuring disfluencies, ASR noise, and compliance-driven agent actions. In deployments where transcripts are unavailable, standard pipelines still yield derived call attributes such as Intent Summaries, Topic Flow, and QA Evaluation Forms. We leverage these as supervision signals to guide generation. To assess the quality of such outputs, we introduce a diagnostic framework of 18 linguistically and behaviorally grounded metrics for comparing real and synthetic transcripts. We benchmark four language-agnostic generation strategies, from simple prompting to characteristic-aware multi-stage approaches, alongside reference-free baselines. Results reveal persistent challenges: no method excels across all traits, with notable deficits in disfluency, sentiment, and behavioral realism. Our diagnostic tool exposes these gaps, enabling fine-grained evaluation and stress testing of synthetic dialogue across languages. 

---
# Exploring the Interplay between Musical Preferences and Personality through the Lens of Language 

**Authors**: Eliran Shem-Tov, Ella Rabinovich  

**Link**: [PDF](https://arxiv.org/pdf/2508.18208)  

**Abstract**: Music serves as a powerful reflection of individual identity, often aligning with deeper psychological traits. Prior research has established correlations between musical preferences and personality traits, while separate studies have demonstrated that personality is detectable through linguistic analysis. Our study bridges these two research domains by investigating whether individuals' musical preferences are recognizable in their spontaneous language through the lens of the Big Five personality traits (Openness, Conscientiousness, Extroversion, Agreeableness, and Neuroticism). Using a carefully curated dataset of over 500,000 text samples from nearly 5,000 authors with reliably identified musical preferences, we build advanced models to assess personality characteristics. Our results reveal significant personality differences across fans of five musical genres. We release resources for future research at the intersection of computational linguistics, music psychology and personality analysis. 

---
# Leveraging Large Language Models for Accurate Sign Language Translation in Low-Resource Scenarios 

**Authors**: Luana Bulla, Gabriele Tuccio, Misael Mongiovì, Aldo Gangemi  

**Link**: [PDF](https://arxiv.org/pdf/2508.18183)  

**Abstract**: Translating natural languages into sign languages is a highly complex and underexplored task. Despite growing interest in accessibility and inclusivity, the development of robust translation systems remains hindered by the limited availability of parallel corpora which align natural language with sign language data. Existing methods often struggle to generalize in these data-scarce environments, as the few datasets available are typically domain-specific, lack standardization, or fail to capture the full linguistic richness of sign languages. To address this limitation, we propose Advanced Use of LLMs for Sign Language Translation (AulSign), a novel method that leverages Large Language Models via dynamic prompting and in-context learning with sample selection and subsequent sign association. Despite their impressive abilities in processing text, LLMs lack intrinsic knowledge of sign languages; therefore, they are unable to natively perform this kind of translation. To overcome this limitation, we associate the signs with compact descriptions in natural language and instruct the model to use them. We evaluate our method on both English and Italian languages using SignBank+, a recognized benchmark in the field, as well as the Italian LaCAM CNR-ISTC dataset. We demonstrate superior performance compared to state-of-the-art models in low-data scenario. Our findings demonstrate the effectiveness of AulSign, with the potential to enhance accessibility and inclusivity in communication technologies for underrepresented linguistic communities. 

---
# Improving End-to-End Training of Retrieval-Augmented Generation Models via Joint Stochastic Approximation 

**Authors**: Hongyu Cao, Yuxuan Wu, Yucheng Cai, Xianyu Zhao, Zhijian Ou  

**Link**: [PDF](https://arxiv.org/pdf/2508.18168)  

**Abstract**: Retrieval-augmented generation (RAG) has become a widely recognized paradigm to combine parametric memory with non-parametric memories. An RAG model consists of two serial connecting components (retriever and generator). A major challenge in end-to-end optimization of the RAG model is that marginalization over relevant passages (modeled as discrete latent variables) from a knowledge base is required. Traditional top-K marginalization and variational RAG (VRAG) suffer from biased or high-variance gradient estimates. In this paper, we propose and develop joint stochastic approximation (JSA) based end-to-end training of RAG, which is referred to as JSA-RAG. The JSA algorithm is a stochastic extension of the EM (expectation-maximization) algorithm and is particularly powerful in estimating discrete latent variable models. Extensive experiments are conducted on five datasets for two tasks (open-domain question answering, knowledge-grounded dialogs) and show that JSA-RAG significantly outperforms both vanilla RAG and VRAG. Further analysis shows the efficacy of JSA-RAG from the perspectives of generation, retrieval, and low-variance gradient estimate. 

---
# DiscussLLM: Teaching Large Language Models When to Speak 

**Authors**: Deep Anil Patel, Iain Melvin, Christopher Malon, Martin Renqiang Min  

**Link**: [PDF](https://arxiv.org/pdf/2508.18167)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in understanding and generating human-like text, yet they largely operate as reactive agents, responding only when directly prompted. This passivity creates an "awareness gap," limiting their potential as truly collaborative partners in dynamic human discussions. We introduce $\textit{DiscussLLM}$, a framework designed to bridge this gap by training models to proactively decide not just $\textit{what}$ to say, but critically, $\textit{when}$ to speak. Our primary contribution is a scalable two-stage data generation pipeline that synthesizes a large-scale dataset of realistic multi-turn human discussions. Each discussion is annotated with one of five intervention types (e.g., Factual Correction, Concept Definition) and contains an explicit conversational trigger where an AI intervention adds value. By training models to predict a special silent token when no intervention is needed, they learn to remain quiet until a helpful contribution can be made. We explore two architectural baselines: an integrated end-to-end model and a decoupled classifier-generator system optimized for low-latency inference. We evaluate these models on their ability to accurately time interventions and generate helpful responses, paving the way for more situationally aware and proactive conversational AI. 

---
# S2Sent: Nested Selectivity Aware Sentence Representation Learning 

**Authors**: Jianxiang Zang, Nijia Mo, Yonda Wei, Meiling Ning, Hui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.18164)  

**Abstract**: The combination of Transformer-based encoders with contrastive learning represents the current mainstream paradigm for sentence representation learning. This paradigm is typically based on the hidden states of the last Transformer block of the encoder. However, within Transformer-based encoders, different blocks exhibit varying degrees of semantic perception ability. From the perspective of interpretability, the semantic perception potential of knowledge neurons is modulated by stimuli, thus rational cross-block representation fusion is a direction worth optimizing. To balance the semantic redundancy and loss across block fusion, we propose a sentence representation selection mechanism S\textsuperscript{2}Sent, which integrates a parameterized nested selector downstream of the Transformer-based encoder. This selector performs spatial selection (SS) and nested frequency selection (FS) from a modular perspective. The SS innovatively employs a spatial squeeze based self-gating mechanism to obtain adaptive weights, which not only achieves fusion with low information redundancy but also captures the dependencies between embedding features. The nested FS replaces GAP with different DCT basis functions to achieve spatial squeeze with low semantic loss. Extensive experiments have demonstrated that S\textsuperscript{2}Sent achieves significant improvements over baseline methods with negligible additional parameters and inference latency, while highlighting high integrability and scalability. 

---
# Toward a Better Localization of Princeton WordNet 

**Authors**: Abed Alhakim Freihat  

**Link**: [PDF](https://arxiv.org/pdf/2508.18134)  

**Abstract**: As Princeton WordNet continues to gain significance as a semantic lexicon in Natural Language Processing, the need for its localization and for ensuring the quality of this process has become increasingly critical. Existing efforts remain limited in both scale and rigor, and there is a notable absence of studies addressing the accuracy of localization or its alignment with the cultural context of Arabic. This paper proposes a structured framework for the localization of Princeton WordNet, detailing the stages and procedures required to achieve high-quality results without compromising cultural authenticity. We further present our experience in applying this framework, reporting outcomes from the localization of 10,000 synsets. 

---
# SentiMM: A Multimodal Multi-Agent Framework for Sentiment Analysis in Social Media 

**Authors**: Xilai Xu, Zilin Zhao, Chengye Song, Zining Wang, Jinhe Qiang, Jiongrui Yan, Yuhuai Lin  

**Link**: [PDF](https://arxiv.org/pdf/2508.18108)  

**Abstract**: With the increasing prevalence of multimodal content on social media, sentiment analysis faces significant challenges in effectively processing heterogeneous data and recognizing multi-label emotions. Existing methods often lack effective cross-modal fusion and external knowledge integration. We propose SentiMM, a novel multi-agent framework designed to systematically address these challenges. SentiMM processes text and visual inputs through specialized agents, fuses multimodal features, enriches context via knowledge retrieval, and aggregates results for final sentiment classification. We also introduce SentiMMD, a large-scale multimodal dataset with seven fine-grained sentiment categories. Extensive experiments demonstrate that SentiMM achieves superior performance compared to state-of-the-art baselines, validating the effectiveness of our structured approach. 

---
# Detecting and Characterizing Planning in Language Models 

**Authors**: Jatin Nainani, Sankaran Vaidyanathan, Connor Watts, Andre N. Assis, Alice Rigg  

**Link**: [PDF](https://arxiv.org/pdf/2508.18098)  

**Abstract**: Modern large language models (LLMs) have demonstrated impressive performance across a wide range of multi-step reasoning tasks. Recent work suggests that LLMs may perform planning - selecting a future target token in advance and generating intermediate tokens that lead towards it - rather than merely improvising one token at a time. However, existing studies assume fixed planning horizons and often focus on single prompts or narrow domains. To distinguish planning from improvisation across models and tasks, we present formal and causally grounded criteria for detecting planning and operationalize them as a semi-automated annotation pipeline. We apply this pipeline to both base and instruction-tuned Gemma-2-2B models on the MBPP code generation benchmark and a poem generation task where Claude 3.5 Haiku was previously shown to plan. Our findings show that planning is not universal: unlike Haiku, Gemma-2-2B solves the same poem generation task through improvisation, and on MBPP it switches between planning and improvisation across similar tasks and even successive token predictions. We further show that instruction tuning refines existing planning behaviors in the base model rather than creating them from scratch. Together, these studies provide a reproducible and scalable foundation for mechanistic studies of planning in LLMs. 

---
# Agri-Query: A Case Study on RAG vs. Long-Context LLMs for Cross-Lingual Technical Question Answering 

**Authors**: Julius Gun, Timo Oksanen  

**Link**: [PDF](https://arxiv.org/pdf/2508.18093)  

**Abstract**: We present a case study evaluating large language models (LLMs) with 128K-token context windows on a technical question answering (QA) task. Our benchmark is built on a user manual for an agricultural machine, available in English, French, and German. It simulates a cross-lingual information retrieval scenario where questions are posed in English against all three language versions of the manual. The evaluation focuses on realistic "needle-in-a-haystack" challenges and includes unanswerable questions to test for hallucinations. We compare nine long-context LLMs using direct prompting against three Retrieval-Augmented Generation (RAG) strategies (keyword, semantic, hybrid), with an LLM-as-a-judge for evaluation. Our findings for this specific manual show that Hybrid RAG consistently outperforms direct long-context prompting. Models like Gemini 2.5 Flash and the smaller Qwen 2.5 7B achieve high accuracy (over 85%) across all languages with RAG. This paper contributes a detailed analysis of LLM performance in a specialized industrial domain and an open framework for similar evaluations, highlighting practical trade-offs and challenges. 

---
# Speech-Based Depressive Mood Detection in the Presence of Multiple Sclerosis: A Cross-Corpus and Cross-Lingual Study 

**Authors**: Monica Gonzalez-Machorro, Uwe Reichel, Pascal Hecker, Helly Hammer, Hesam Sagha, Florian Eyben, Robert Hoepner, Björn W. Schuller  

**Link**: [PDF](https://arxiv.org/pdf/2508.18092)  

**Abstract**: Depression commonly co-occurs with neurodegenerative disorders like Multiple Sclerosis (MS), yet the potential of speech-based Artificial Intelligence for detecting depression in such contexts remains unexplored. This study examines the transferability of speech-based depression detection methods to people with MS (pwMS) through cross-corpus and cross-lingual analysis using English data from the general population and German data from pwMS. Our approach implements supervised machine learning models using: 1) conventional speech and language features commonly used in the field, 2) emotional dimensions derived from a Speech Emotion Recognition (SER) model, and 3) exploratory speech feature analysis. Despite limited data, our models detect depressive mood in pwMS with moderate generalisability, achieving a 66% Unweighted Average Recall (UAR) on a binary task. Feature selection further improved performance, boosting UAR to 74%. Our findings also highlight the relevant role emotional changes have as an indicator of depressive mood in both the general population and within PwMS. This study provides an initial exploration into generalising speech-based depression detection, even in the presence of co-occurring conditions, such as neurodegenerative diseases. 

---
# How Quantization Shapes Bias in Large Language Models 

**Authors**: Federico Marcuzzi, Xuefei Ning, Roy Schwartz, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2508.18088)  

**Abstract**: This work presents a comprehensive evaluation of how quantization affects model bias, with particular attention to its impact on individual demographic subgroups. We focus on weight and activation quantization strategies and examine their effects across a broad range of bias types, including stereotypes, toxicity, sentiment, and fairness. We employ both probabilistic and generated text-based metrics across nine benchmarks and evaluate models varying in architecture family and reasoning ability. Our findings show that quantization has a nuanced impact on bias: while it can reduce model toxicity and does not significantly impact sentiment, it tends to slightly increase stereotypes and unfairness in generative tasks, especially under aggressive compression. These trends are generally consistent across demographic categories and model types, although their magnitude depends on the specific setting. Overall, our results highlight the importance of carefully balancing efficiency and ethical considerations when applying quantization in practice. 

---
# Neither Valid nor Reliable? Investigating the Use of LLMs as Judges 

**Authors**: Khaoula Chehbouni, Mohammed Haddou, Jackie Chi Kit Cheung, Golnoosh Farnadi  

**Link**: [PDF](https://arxiv.org/pdf/2508.18076)  

**Abstract**: Evaluating natural language generation (NLG) systems remains a core challenge of natural language processing (NLP), further complicated by the rise of large language models (LLMs) that aims to be general-purpose. Recently, large language models as judges (LLJs) have emerged as a promising alternative to traditional metrics, but their validity remains underexplored. This position paper argues that the current enthusiasm around LLJs may be premature, as their adoption has outpaced rigorous scrutiny of their reliability and validity as evaluators. Drawing on measurement theory from the social sciences, we identify and critically assess four core assumptions underlying the use of LLJs: their ability to act as proxies for human judgment, their capabilities as evaluators, their scalability, and their cost-effectiveness. We examine how each of these assumptions may be challenged by the inherent limitations of LLMs, LLJs, or current practices in NLG evaluation. To ground our analysis, we explore three applications of LLJs: text summarization, data annotation, and safety alignment. Finally, we highlight the need for more responsible evaluation practices in LLJs evaluation, to ensure that their growing role in the field supports, rather than undermines, progress in NLG. 

---
# A Retail-Corpus for Aspect-Based Sentiment Analysis with Large Language Models 

**Authors**: Oleg Silcenco, Marcos R. Machad, Wallace C. Ugulino, Daniel Braun  

**Link**: [PDF](https://arxiv.org/pdf/2508.17994)  

**Abstract**: Aspect-based sentiment analysis enhances sentiment detection by associating it with specific aspects, offering deeper insights than traditional sentiment analysis. This study introduces a manually annotated dataset of 10,814 multilingual customer reviews covering brick-and-mortar retail stores, labeled with eight aspect categories and their sentiment. Using this dataset, the performance of GPT-4 and LLaMA-3 in aspect based sentiment analysis is evaluated to establish a baseline for the newly introduced data. The results show both models achieving over 85% accuracy, while GPT-4 outperforms LLaMA-3 overall with regard to all relevant metrics. 

---
# German4All - A Dataset and Model for Readability-Controlled Paraphrasing in German 

**Authors**: Miriam Anschütz, Thanh Mai Pham, Eslam Nasrallah, Maximilian Müller, Cristian-George Craciun, Georg Groh  

**Link**: [PDF](https://arxiv.org/pdf/2508.17973)  

**Abstract**: The ability to paraphrase texts across different complexity levels is essential for creating accessible texts that can be tailored toward diverse reader groups. Thus, we introduce German4All, the first large-scale German dataset of aligned readability-controlled, paragraph-level paraphrases. It spans five readability levels and comprises over 25,000 samples. The dataset is automatically synthesized using GPT-4 and rigorously evaluated through both human and LLM-based judgments. Using German4All, we train an open-source, readability-controlled paraphrasing model that achieves state-of-the-art performance in German text simplification, enabling more nuanced and reader-specific adaptations. We opensource both the dataset and the model to encourage further research on multi-level paraphrasing 

---
# Understanding Subword Compositionality of Large Language Models 

**Authors**: Qiwei Peng, Yekun Chai, Anders Søgaard  

**Link**: [PDF](https://arxiv.org/pdf/2508.17953)  

**Abstract**: Large language models (LLMs) take sequences of subwords as input, requiring them to effective compose subword representations into meaningful word-level representations. In this paper, we present a comprehensive set of experiments to probe how LLMs compose subword information, focusing on three key aspects: structural similarity, semantic decomposability, and form retention. Our analysis of the experiments suggests that these five LLM families can be classified into three distinct groups, likely reflecting difference in their underlying composition strategies. Specifically, we observe (i) three distinct patterns in the evolution of structural similarity between subword compositions and whole-word representations across layers; (ii) great performance when probing layer by layer their sensitivity to semantic decompositionality; and (iii) three distinct patterns when probing sensitivity to formal features, e.g., character sequence length. These findings provide valuable insights into the compositional dynamics of LLMs and highlight different compositional pattens in how LLMs encode and integrate subword information. 

---
# Debiasing Multilingual LLMs in Cross-lingual Latent Space 

**Authors**: Qiwei Peng, Guimin Hu, Yekun Chai, Anders Søgaard  

**Link**: [PDF](https://arxiv.org/pdf/2508.17948)  

**Abstract**: Debiasing techniques such as SentDebias aim to reduce bias in large language models (LLMs). Previous studies have evaluated their cross-lingual transferability by directly applying these methods to LLM representations, revealing their limited effectiveness across languages. In this work, we therefore propose to perform debiasing in a joint latent space rather than directly on LLM representations. We construct a well-aligned cross-lingual latent space using an autoencoder trained on parallel TED talk scripts. Our experiments with Aya-expanse and two debiasing techniques across four languages (English, French, German, Dutch) demonstrate that a) autoencoders effectively construct a well-aligned cross-lingual latent space, and b) applying debiasing techniques in the learned cross-lingual latent space significantly improves both the overall debiasing performance and cross-lingual transferability. 

---
# AMELIA: A Family of Multi-task End-to-end Language Models for Argumentation 

**Authors**: Henri Savigny, Bruno Yun  

**Link**: [PDF](https://arxiv.org/pdf/2508.17926)  

**Abstract**: Argument mining is a subfield of argumentation that aims to automatically extract argumentative structures and their relations from natural language texts. This paper investigates how a single large language model can be leveraged to perform one or several argument mining tasks. Our contributions are two-fold. First, we construct a multi-task dataset by surveying and converting 19 well-known argument mining datasets from the literature into a unified format. Second, we explore various training strategies using Meta AI's Llama-3.1-8B-Instruct model: (1) fine-tuning on individual tasks, (2) fine-tuning jointly on multiple tasks, and (3) merging models fine-tuned separately on individual tasks. Our experiments show that task-specific fine-tuning significantly improves individual performance across all tasks. Moreover, multi-task fine-tuning maintains strong performance without degradation, suggesting effective transfer learning across related tasks. Finally, we demonstrate that model merging offers a viable compromise: it yields competitive performance while mitigating the computational costs associated with full multi-task fine-tuning. 

---
# Feature-Refined Unsupervised Model for Loanword Detection 

**Authors**: Promise Dodzi Kpoglu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17923)  

**Abstract**: We propose an unsupervised method for detecting loanwords i.e., words borrowed from one language into another. While prior work has primarily relied on language-external information to identify loanwords, such approaches can introduce circularity and constraints into the historical linguistics workflow. In contrast, our model relies solely on language-internal information to process both native and borrowed words in monolingual and multilingual wordlists. By extracting pertinent linguistic features, scoring them, and mapping them probabilistically, we iteratively refine initial results by identifying and generalizing from emerging patterns until convergence. This hybrid approach leverages both linguistic and statistical cues to guide the discovery process. We evaluate our method on the task of isolating loanwords in datasets from six standard Indo-European languages: English, German, French, Italian, Spanish, and Portuguese. Experimental results demonstrate that our model outperforms baseline methods, with strong performance gains observed when scaling to cross-linguistic data. 

---
# Information availability in different languages and various technological constraints related to multilinguism on the Internet 

**Authors**: Sonal Khosla, Haridasa Acharya  

**Link**: [PDF](https://arxiv.org/pdf/2508.17918)  

**Abstract**: The usage of Internet has grown exponentially over the last two decades. The number of Internet users has grown from 16 Million to 1650 Million from 1995 to 2010. It has become a major repository of information catering almost every area. Since the Internet has its origin in USA which is English speaking country there is huge dominance of English on the World Wide Web. Although English is a globally acceptable language, still there is a huge population in the world which is not able to access the Internet due to language constraints. It has been estimated that only 20-25% of the world population speaks English as a native language. More and more people are accessing the Internet nowadays removing the cultural and linguistic barriers and hence there is a high growth in the number of non-English speaking users over the last few years on the Internet. Although many solutions have been provided to remove the linguistic barriers, still there is a huge gap to be filled. This paper attempts to analyze the need of information availability in different languages and the various technological constraints related to multi-linguism on the Internet. 

---
# Evaluating the Representation of Vowels in Wav2Vec Feature Extractor: A Layer-Wise Analysis Using MFCCs 

**Authors**: Domenico De Cristofaro, Vincenzo Norman Vitale, Alessandro Vietti  

**Link**: [PDF](https://arxiv.org/pdf/2508.17914)  

**Abstract**: Automatic Speech Recognition has advanced with self-supervised learning, enabling feature extraction directly from raw audio. In Wav2Vec, a CNN first transforms audio into feature vectors before the transformer processes them. This study examines CNN-extracted information for monophthong vowels using the TIMIT corpus. We compare MFCCs, MFCCs with formants, and CNN activations by training SVM classifiers for front-back vowel identification, assessing their classification accuracy to evaluate phonetic representation. 

---
# Pandora: Leveraging Code-driven Knowledge Transfer for Unified Structured Knowledge Reasoning 

**Authors**: Yongrui Chen, Junhao He, Linbo Fu, Shenyu Zhang, Rihui Jin, Xinbang Dai, Jiaqi Li, Dehai Min, Nan Hu, Yuxin Zhang, Guilin Qi, Yi Huang, Tongtong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17905)  

**Abstract**: Unified Structured Knowledge Reasoning (USKR) aims to answer natural language questions by using structured sources such as tables, databases, and knowledge graphs in a unified way. Existing USKR methods rely on task-specific strategies or bespoke representations, which hinder their ability to dismantle barriers between different SKR tasks, thereby constraining their overall performance in cross-task scenarios. In this paper, we introduce \textsc{Pandora}, a novel USKR framework that addresses the limitations of existing methods by leveraging two key innovations. First, we propose a code-based unified knowledge representation using \textsc{Python}'s \textsc{Pandas} API, which aligns seamlessly with the pre-training of LLMs. This representation facilitates a cohesive approach to handling different structured knowledge sources. Building on this foundation, we employ knowledge transfer to bolster the unified reasoning process of LLMs by automatically building cross-task memory. By adaptively correcting reasoning using feedback from code execution, \textsc{Pandora} showcases impressive unified reasoning capabilities. Extensive experiments on six widely used benchmarks across three SKR tasks demonstrate that \textsc{Pandora} outperforms existing unified reasoning frameworks and competes effectively with task-specific methods. 

---
# ILRe: Intermediate Layer Retrieval for Context Compression in Causal Language Models 

**Authors**: Manlai Liang, Mandi Liu, Jiangzhou Ji, Huaijun Li, Haobo Yang, Yaohan He, Jinlong Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.17892)  

**Abstract**: Large Language Models (LLMs) have demonstrated success across many benchmarks. However, they still exhibit limitations in long-context scenarios, primarily due to their short effective context length, quadratic computational complexity, and high memory overhead when processing lengthy inputs. To mitigate these issues, we introduce a novel context compression pipeline, called Intermediate Layer Retrieval (ILRe), which determines one intermediate decoder layer offline, encodes context by streaming chunked prefill only up to that layer, and recalls tokens by the attention scores between the input query and full key cache in that specified layer. In particular, we propose a multi-pooling kernels allocating strategy in the token recalling process to maintain the completeness of semantics. Our approach not only reduces the prefilling complexity from $O(L^2)$ to $O(L)$, but also achieves performance comparable to or better than the full context in the long context scenarios. Without additional post training or operator development, ILRe can process a single $1M$ tokens request in less than half a minute (speedup $\approx 180\times$) and scores RULER-$1M$ benchmark of $\approx 79.8$ with model Llama-3.1-UltraLong-8B-1M-Instruct on a Huawei Ascend 910B NPU. 

---
# Speech Discrete Tokens or Continuous Features? A Comparative Analysis for Spoken Language Understanding in SpeechLLMs 

**Authors**: Dingdong Wang, Junan Li, Mingyu Cui, Dongchao Yang, Xueyuan Chen, Helen Meng  

**Link**: [PDF](https://arxiv.org/pdf/2508.17863)  

**Abstract**: With the rise of Speech Large Language Models (SpeechLLMs), two dominant approaches have emerged for speech processing: discrete tokens and continuous features. Each approach has demonstrated strong capabilities in audio-related processing tasks. However, the performance gap between these two paradigms has not been thoroughly explored. To address this gap, we present a fair comparison of self-supervised learning (SSL)-based discrete and continuous features under the same experimental settings. We evaluate their performance across six spoken language understanding-related tasks using both small and large-scale LLMs (Qwen1.5-0.5B and Llama3.1-8B). We further conduct in-depth analyses, including efficient comparison, SSL layer analysis, LLM layer analysis, and robustness comparison. Our findings reveal that continuous features generally outperform discrete tokens in various tasks. Each speech processing method exhibits distinct characteristics and patterns in how it learns and processes speech information. We hope our results will provide valuable insights to advance spoken language understanding in SpeechLLMs. 

---
# Beyond Demographics: Enhancing Cultural Value Survey Simulation with Multi-Stage Personality-Driven Cognitive Reasoning 

**Authors**: Haijiang Liu, Qiyuan Li, Chao Gao, Yong Cao, Xiangyu Xu, Xun Wu, Daniel Hershcovich, Jinguang Gu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17855)  

**Abstract**: Introducing MARK, the Multi-stAge Reasoning frameworK for cultural value survey response simulation, designed to enhance the accuracy, steerability, and interpretability of large language models in this task. The system is inspired by the type dynamics theory in the MBTI psychological framework for personality research. It effectively predicts and utilizes human demographic information for simulation: life-situational stress analysis, group-level personality prediction, and self-weighted cognitive imitation. Experiments on the World Values Survey show that MARK outperforms existing baselines by 10% accuracy and reduces the divergence between model predictions and human preferences. This highlights the potential of our framework to improve zero-shot personalization and help social scientists interpret model predictions. 

---
# DRQA: Dynamic Reasoning Quota Allocation for Controlling Overthinking in Reasoning Large Language Models 

**Authors**: Kaiwen Yan, Xuanqing Shi, Hongcheng Guo, Wenxuan Wang, Zhuosheng Zhang, Chengwei Qin  

**Link**: [PDF](https://arxiv.org/pdf/2508.17803)  

**Abstract**: Reasoning large language models (RLLMs), such as OpenAI-O3 and DeepSeek-R1, have recently demonstrated remarkable capabilities by performing structured and multi-step reasoning. However, recent studies reveal that RLLMs often suffer from overthinking, i.e., producing unnecessarily lengthy reasoning chains even for simple questions, leading to excessive token consumption and computational inefficiency. Interestingly, we observe that when processing multiple questions in batch mode, RLLMs exhibit more resource-efficient behavior by dynamically compressing reasoning steps for easier problems, due to implicit resource competition. Inspired by this, we propose Dynamic Reasoning Quota Allocation (DRQA), a novel method that transfers the benefits of resource competition from batch processing to single-question inference. Specifically, DRQA leverages batch-generated preference data and reinforcement learning to train the model to allocate reasoning resources adaptively. By encouraging the model to internalize a preference for responses that are both accurate and concise, DRQA enables it to generate concise answers for simple questions while retaining sufficient reasoning depth for more challenging ones. Extensive experiments on a wide range of mathematical and scientific reasoning benchmarks demonstrate that DRQA significantly reduces token usage while maintaining, and in many cases improving, answer accuracy. By effectively mitigating the overthinking problem, DRQA offers a promising direction for more efficient and scalable deployment of RLLMs, and we hope it inspires further exploration into fine-grained control of reasoning behaviors. 

---
# Zero-shot Context Biasing with Trie-based Decoding using Synthetic Multi-Pronunciation 

**Authors**: Changsong Liu, Yizhou Peng, Eng Siong Chng  

**Link**: [PDF](https://arxiv.org/pdf/2508.17796)  

**Abstract**: Contextual automatic speech recognition (ASR) systems allow for recognizing out-of-vocabulary (OOV) words, such as named entities or rare words. However, it remains challenging due to limited training data and ambiguous or inconsistent pronunciations. In this paper, we propose a synthesis-driven multi-pronunciation contextual biasing method that performs zero-shot contextual ASR on a pretrained Whisper model. Specifically, we leverage text-to-speech (TTS) systems to synthesize diverse speech samples containing each target rare word, and then use the pretrained Whisper model to extract multiple predicted pronunciation variants. These variant token sequences are compiled into a prefix-trie, which assigns rewards to beam hypotheses in a shallow-fusion manner during beam-search decoding. After which, any recognized variant is mapped back to the original rare word in the final transcription. The evaluation results on the Librispeech dataset show that our method reduces biased word error rate (WER) by 42% on test-clean and 43% on test-other while maintaining unbiased WER essentially unchanged. 

---
# Speculating LLMs' Chinese Training Data Pollution from Their Tokens 

**Authors**: Qingjie Zhang, Di Wang, Haoting Qian, Liu Yan, Tianwei Zhang, Ke Xu, Qi Li, Minlie Huang, Hewu Li, Han Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17771)  

**Abstract**: Tokens are basic elements in the datasets for LLM training. It is well-known that many tokens representing Chinese phrases in the vocabulary of GPT (4o/4o-mini/o1/o3/4.5/4.1/o4-mini) are indicating contents like pornography or online gambling. Based on this observation, our goal is to locate Polluted Chinese (PoC) tokens in LLMs and study the relationship between PoC tokens' existence and training data. (1) We give a formal definition and taxonomy of PoC tokens based on the GPT's vocabulary. (2) We build a PoC token detector via fine-tuning an LLM to label PoC tokens in vocabularies by considering each token's both semantics and related contents from the search engines. (3) We study the speculation on the training data pollution via PoC tokens' appearances (token ID). Experiments on GPT and other 23 LLMs indicate that tokens widely exist while GPT's vocabulary behaves the worst: more than 23% long Chinese tokens (i.e., a token with more than two Chinese characters) are either porn or online gambling. We validate the accuracy of our speculation method on famous pre-training datasets like C4 and Pile. Then, considering GPT-4o, we speculate that the ratio of "Yui Hatano" related webpages in GPT-4o's training data is around 0.5%. 

---
# ISACL: Internal State Analyzer for Copyrighted Training Data Leakage 

**Authors**: Guangwei Zhang, Qisheng Su, Jiateng Liu, Cheng Qian, Yanzhou Pan, Yanjie Fu, Denghui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17767)  

**Abstract**: Large Language Models (LLMs) have revolutionized Natural Language Processing (NLP) but pose risks of inadvertently exposing copyrighted or proprietary data, especially when such data is used for training but not intended for distribution. Traditional methods address these leaks only after content is generated, which can lead to the exposure of sensitive information. This study introduces a proactive approach: examining LLMs' internal states before text generation to detect potential leaks. By using a curated dataset of copyrighted materials, we trained a neural network classifier to identify risks, allowing for early intervention by stopping the generation process or altering outputs to prevent disclosure. Integrated with a Retrieval-Augmented Generation (RAG) system, this framework ensures adherence to copyright and licensing requirements while enhancing data privacy and ethical standards. Our results show that analyzing internal states effectively mitigates the risk of copyrighted data leakage, offering a scalable solution that fits smoothly into AI workflows, ensuring compliance with copyright regulations while maintaining high-quality text generation. The implementation is available on GitHub.\footnote{this https URL} 

---
# SMITE: Enhancing Fairness in LLMs through Optimal In-Context Example Selection via Dynamic Validation 

**Authors**: Garima Chhikara, Kripabandhu Ghosh, Abhijnan Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2508.17735)  

**Abstract**: Large Language Models (LLMs) are widely used for downstream tasks such as tabular classification, where ensuring fairness in their outputs is critical for inclusivity, equal representation, and responsible AI deployment. This study introduces a novel approach to enhancing LLM performance and fairness through the concept of a dynamic validation set, which evolves alongside the test set, replacing the traditional static validation approach. We also propose an iterative algorithm, SMITE, to select optimal in-context examples, with each example set validated against its corresponding dynamic validation set. The in-context set with the lowest total error is used as the final demonstration set. Our experiments across four different LLMs show that our proposed techniques significantly improve both predictive accuracy and fairness compared to baseline methods. To our knowledge, this is the first study to apply dynamic validation in the context of in-context learning for LLMs. 

---
# Layerwise Importance Analysis of Feed-Forward Networks in Transformer-based Language Models 

**Authors**: Wataru Ikeda, Kazuki Yano, Ryosuke Takahashi, Jaesung Lee, Keigo Shibata, Jun Suzuki  

**Link**: [PDF](https://arxiv.org/pdf/2508.17734)  

**Abstract**: This study investigates the layerwise importance of feed-forward networks (FFNs) in Transformer-based language models during pretraining. We introduce an experimental approach that, while maintaining the total parameter count, increases the FFN dimensions in some layers and completely removes the FFNs from other layers. Furthermore, since our focus is on the importance of FFNs during pretraining, we train models from scratch to examine whether the importance of FFNs varies depending on their layer positions, rather than using publicly available pretrained models as is frequently done. Through comprehensive evaluations of models with varying sizes (285M, 570M, and 1.2B parameters) and layer counts (12, 24, and 40 layers), we demonstrate that concentrating FFNs in 70% of the consecutive middle layers consistently outperforms standard configurations for multiple downstream tasks. 

---
# EMPOWER: Evolutionary Medical Prompt Optimization With Reinforcement Learning 

**Authors**: Yinda Chen, Yangfan He, Jing Yang, Dapeng Zhang, Zhenlong Yuan, Muhammad Attique Khan, Jamel Baili, Por Lip Yee  

**Link**: [PDF](https://arxiv.org/pdf/2508.17703)  

**Abstract**: Prompt engineering significantly influences the reliability and clinical utility of Large Language Models (LLMs) in medical applications. Current optimization approaches inadequately address domain-specific medical knowledge and safety requirements. This paper introduces EMPOWER, a novel evolutionary framework that enhances medical prompt quality through specialized representation learning, multi-dimensional evaluation, and structure-preserving algorithms. Our methodology incorporates: (1) a medical terminology attention mechanism, (2) a comprehensive assessment architecture evaluating clarity, specificity, clinical relevance, and factual accuracy, (3) a component-level evolutionary algorithm preserving clinical reasoning integrity, and (4) a semantic verification module ensuring adherence to medical knowledge. Evaluation across diagnostic, therapeutic, and educational tasks demonstrates significant improvements: 24.7% reduction in factually incorrect content, 19.6% enhancement in domain specificity, and 15.3% higher clinician preference in blinded evaluations. The framework addresses critical challenges in developing clinically appropriate prompts, facilitating more responsible integration of LLMs into healthcare settings. 

---
# Text Meets Topology: Rethinking Out-of-distribution Detection in Text-Rich Networks 

**Authors**: Danny Wang, Ruihong Qiu, Guangdong Bai, Zi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17690)  

**Abstract**: Out-of-distribution (OOD) detection remains challenging in text-rich networks, where textual features intertwine with topological structures. Existing methods primarily address label shifts or rudimentary domain-based splits, overlooking the intricate textual-structural diversity. For example, in social networks, where users represent nodes with textual features (name, bio) while edges indicate friendship status, OOD may stem from the distinct language patterns between bot and normal users. To address this gap, we introduce the TextTopoOOD framework for evaluating detection across diverse OOD scenarios: (1) attribute-level shifts via text augmentations and embedding perturbations; (2) structural shifts through edge rewiring and semantic connections; (3) thematically-guided label shifts; and (4) domain-based divisions. Furthermore, we propose TNT-OOD to model the complex interplay between Text aNd Topology using: 1) a novel cross-attention module to fuse local structure into node-level text representations, and 2) a HyperNetwork to generate node-specific transformation parameters. This aligns topological and semantic features of ID nodes, enhancing ID/OOD distinction across structural and textual shifts. Experiments on 11 datasets across four OOD scenarios demonstrate the nuanced challenge of TextTopoOOD for evaluating OOD detection in text-rich networks. 

---
# CoCoA: Confidence- and Context-Aware Adaptive Decoding for Resolving Knowledge Conflicts in Large Language Models 

**Authors**: Anant Khandelwal, Manish Gupta, Puneet Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2508.17670)  

**Abstract**: Faithful generation in large language models (LLMs) is challenged by knowledge conflicts between parametric memory and external context. Existing contrastive decoding methods tuned specifically to handle conflict often lack adaptability and can degrade performance in low conflict settings. We introduce CoCoA (Confidence- and Context-Aware Adaptive Decoding), a novel token-level algorithm for principled conflict resolution and enhanced faithfulness. CoCoA resolves conflict by utilizing confidence-aware measures (entropy gap and contextual peakedness) and the generalized divergence between the parametric and contextual distributions. Crucially, CoCoA maintains strong performance even in low conflict settings. Extensive experiments across multiple LLMs on diverse Question Answering (QA), Summarization, and Long-Form Question Answering (LFQA) benchmarks demonstrate CoCoA's state-of-the-art performance over strong baselines like AdaCAD. It yields significant gains in QA accuracy, up to 9.2 points on average compared to the strong baseline AdaCAD, and improves factuality in summarization and LFQA by up to 2.5 points on average across key benchmarks. Additionally, it demonstrates superior sensitivity to conflict variations. CoCoA enables more informed, context-aware, and ultimately more faithful token generation. 

---
# SurveyGen: Quality-Aware Scientific Survey Generation with Large Language Models 

**Authors**: Tong Bao, Mir Tafseer Nayeem, Davood Rafiei, Chengzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17647)  

**Abstract**: Automatic survey generation has emerged as a key task in scientific document processing. While large language models (LLMs) have shown promise in generating survey texts, the lack of standardized evaluation datasets critically hampers rigorous assessment of their performance against human-written surveys. In this work, we present SurveyGen, a large-scale dataset comprising over 4,200 human-written surveys across diverse scientific domains, along with 242,143 cited references and extensive quality-related metadata for both the surveys and the cited papers. Leveraging this resource, we build QUAL-SG, a novel quality-aware framework for survey generation that enhances the standard Retrieval-Augmented Generation (RAG) pipeline by incorporating quality-aware indicators into literature retrieval to assess and select higher-quality source papers. Using this dataset and framework, we systematically evaluate state-of-the-art LLMs under varying levels of human involvement - from fully automatic generation to human-guided writing. Experimental results and human evaluations show that while semi-automatic pipelines can achieve partially competitive outcomes, fully automatic survey generation still suffers from low citation quality and limited critical analysis. 

---
# Weights-Rotated Preference Optimization for Large Language Models 

**Authors**: Chenxu Yang, Ruipeng Jia, Mingyu Zheng, Naibin Gu, Zheng Lin, Siyuan Chen, Weichong Yin, Hua Wu, Weiping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17637)  

**Abstract**: Despite the efficacy of Direct Preference Optimization (DPO) in aligning Large Language Models (LLMs), reward hacking remains a pivotal challenge. This issue emerges when LLMs excessively reduce the probability of rejected completions to achieve high rewards, without genuinely meeting their intended goals. As a result, this leads to overly lengthy generation lacking diversity, as well as catastrophic forgetting of knowledge. We investigate the underlying reason behind this issue, which is representation redundancy caused by neuron collapse in the parameter space. Hence, we propose a novel Weights-Rotated Preference Optimization (RoPO) algorithm, which implicitly constrains the output layer logits with the KL divergence inherited from DPO and explicitly constrains the intermediate hidden states by fine-tuning on a multi-granularity orthogonal matrix. This design prevents the policy model from deviating too far from the reference model, thereby retaining the knowledge and expressive capabilities acquired during pre-training and SFT stages. Our RoPO achieves up to a 3.27-point improvement on AlpacaEval 2, and surpasses the best baseline by 6.2 to 7.5 points on MT-Bench with merely 0.015% of the trainable parameters, demonstrating its effectiveness in alleviating the reward hacking problem of DPO. 

---
# Stop Spinning Wheels: Mitigating LLM Overthinking via Mining Patterns for Early Reasoning Exit 

**Authors**: Zihao Wei, Liang Pang, Jiahao Liu, Jingcheng Deng, Shicheng Xu, Zenghao Duan, Jingang Wang, Fei Sun, Xunliang Cai, Huawei Shen, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.17627)  

**Abstract**: Large language models (LLMs) enhance complex reasoning tasks by scaling the individual thinking process. However, prior work shows that overthinking can degrade overall performance. Motivated by observed patterns in thinking length and content length, we categorize reasoning into three stages: insufficient exploration stage, compensatory reasoning stage, and reasoning convergence stage. Typically, LLMs produce correct answers in the compensatory reasoning stage, whereas reasoning convergence often triggers overthinking, causing increased resource usage or even infinite loops. Therefore, mitigating overthinking hinges on detecting the end of the compensatory reasoning stage, defined as the Reasoning Completion Point (RCP). RCP typically appears at the end of the first complete reasoning cycle and can be identified by querying the LLM sentence by sentence or monitoring the probability of an end-of-thinking token (e.g., \texttt{</think>}), though these methods lack an efficient and precise balance. To improve this, we mine more sensitive and consistent RCP patterns and develop a lightweight thresholding strategy based on heuristic rules. Experimental evaluations on benchmarks (AIME24, AIME25, GPQA-D) demonstrate that the proposed method reduces token consumption while preserving or enhancing reasoning accuracy. 

---
# EMO-Reasoning: Benchmarking Emotional Reasoning Capabilities in Spoken Dialogue Systems 

**Authors**: Jingwen Liu, Kan Jen Cheng, Jiachen Lian, Akshay Anand, Rishi Jain, Faith Qiao, Robin Netzorg, Huang-Cheng Chou, Tingle Li, Guan-Ting Lin, Gopala Anumanchipalli  

**Link**: [PDF](https://arxiv.org/pdf/2508.17623)  

**Abstract**: Speech emotions play a crucial role in human-computer interaction, shaping engagement and context-aware communication. Despite recent advances in spoken dialogue systems, a holistic system for evaluating emotional reasoning is still lacking. To address this, we introduce EMO-Reasoning, a benchmark for assessing emotional coherence in dialogue systems. It leverages a curated dataset generated via text-to-speech to simulate diverse emotional states, overcoming the scarcity of emotional speech data. We further propose the Cross-turn Emotion Reasoning Score to assess the emotion transitions in multi-turn dialogues. Evaluating seven dialogue systems through continuous, categorical, and perceptual metrics, we show that our framework effectively detects emotional inconsistencies, providing insights for improving current dialogue systems. By releasing a systematic evaluation benchmark, we aim to advance emotion-aware spoken dialogue modeling toward more natural and adaptive interactions. 

---
# Steering When Necessary: Flexible Steering Large Language Models with Backtracking 

**Authors**: Jinwei Gan, Zifeng Cheng, Zhiwei Jiang, Cong Wang, Yafeng Yin, Xiang Luo, Yuchen Fu, Qing Gu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17621)  

**Abstract**: Large language models (LLMs) have achieved remarkable performance across many generation tasks. Nevertheless, effectively aligning them with desired behaviors remains a significant challenge. Activation steering is an effective and cost-efficient approach that directly modifies the activations of LLMs during the inference stage, aligning their responses with the desired behaviors and avoiding the high cost of fine-tuning. Existing methods typically indiscriminately intervene to all generations or rely solely on the question to determine intervention, which limits the accurate assessment of the intervention strength. To this end, we propose the Flexible Activation Steering with Backtracking (FASB) framework, which dynamically determines both the necessity and strength of intervention by tracking the internal states of the LLMs during generation, considering both the question and the generated content. Since intervening after detecting a deviation from the desired behavior is often too late, we further propose the backtracking mechanism to correct the deviated tokens and steer the LLMs toward the desired behavior. Extensive experiments on the TruthfulQA dataset and six multiple-choice datasets demonstrate that our method outperforms baselines. Our code will be released at this https URL. 

---
# Less Is More? Examining Fairness in Pruned Large Language Models for Summarising Opinions 

**Authors**: Nannan Huang, Haytham Fayek, Xiuzhen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17610)  

**Abstract**: Model compression through post-training pruning offers a way to reduce model size and computational requirements without significantly impacting model performance. However, the effect of pruning on the fairness of LLM-generated summaries remains unexplored, particularly for opinion summarisation where biased outputs could influence public this http URL this paper, we present a comprehensive empirical analysis of opinion summarisation, examining three state-of-the-art pruning methods and various calibration sets across three open-source LLMs using four fairness metrics. Our systematic analysis reveals that pruning methods have a greater impact on fairness than calibration sets. Building on these insights, we propose High Gradient Low Activation (HGLA) pruning, which identifies and removes parameters that are redundant for input processing but influential in output generation. Our experiments demonstrate that HGLA can better maintain or even improve fairness compared to existing methods, showing promise across models and tasks where traditional methods have limitations. Our human evaluation shows HGLA-generated outputs are fairer than existing state-of-the-art pruning methods. Code is available at: this https URL. 

---
# UQ: Assessing Language Models on Unsolved Questions 

**Authors**: Fan Nie, Ken Ziyu Liu, Zihao Wang, Rui Sun, Wei Liu, Weijia Shi, Huaxiu Yao, Linjun Zhang, Andrew Y. Ng, James Zou, Sanmi Koyejo, Yejin Choi, Percy Liang, Niklas Muennighoff  

**Link**: [PDF](https://arxiv.org/pdf/2508.17580)  

**Abstract**: Benchmarks shape progress in AI research. A useful benchmark should be both difficult and realistic: questions should challenge frontier models while also reflecting real-world usage. Yet, current paradigms face a difficulty-realism tension: exam-style benchmarks are often made artificially difficult with limited real-world value, while benchmarks based on real user interaction often skew toward easy, high-frequency problems. In this work, we explore a radically different paradigm: assessing models on unsolved questions. Rather than a static benchmark scored once, we curate unsolved questions and evaluate models asynchronously over time with validator-assisted screening and community verification. We introduce UQ, a testbed of 500 challenging, diverse questions sourced from Stack Exchange, spanning topics from CS theory and math to sci-fi and history, probing capabilities including reasoning, factuality, and browsing. UQ is difficult and realistic by construction: unsolved questions are often hard and naturally arise when humans seek answers, thus solving them yields direct real-world value. Our contributions are threefold: (1) UQ-Dataset and its collection pipeline combining rule-based filters, LLM judges, and human review to ensure question quality (e.g., well-defined and difficult); (2) UQ-Validators, compound validation strategies that leverage the generator-validator gap to provide evaluation signals and pre-screen candidate solutions for human review; and (3) UQ-Platform, an open platform where experts collectively verify questions and solutions. The top model passes UQ-validation on only 15% of questions, and preliminary human verification has already identified correct answers among those that passed. UQ charts a path for evaluating frontier models on real-world, open-ended challenges, where success pushes the frontier of human knowledge. We release UQ at this https URL. 

---
# CausalSent: Interpretable Sentiment Classification with RieszNet 

**Authors**: Daniel Frees, Martin Pollack  

**Link**: [PDF](https://arxiv.org/pdf/2508.17576)  

**Abstract**: Despite the overwhelming performance improvements offered by recent natural language procesing (NLP) models, the decisions made by these models are largely a black box. Towards closing this gap, the field of causal NLP combines causal inference literature with modern NLP models to elucidate causal effects of text features. We replicate and extend Bansal et al's work on regularizing text classifiers to adhere to estimated effects, focusing instead on model interpretability. Specifically, we focus on developing a two-headed RieszNet-based neural network architecture which achieves better treatment effect estimation accuracy. Our framework, CausalSent, accurately predicts treatment effects in semi-synthetic IMDB movie reviews, reducing MAE of effect estimates by 2-3x compared to Bansal et al's MAE on synthetic Civil Comments data. With an ensemble of validated models, we perform an observational case study on the causal effect of the word "love" in IMDB movie reviews, finding that the presence of the word "love" causes a +2.9% increase in the probability of a positive sentiment. 

---
# Humanizing Machines: Rethinking LLM Anthropomorphism Through a Multi-Level Framework of Design 

**Authors**: Yunze Xiao, Lynnette Hui Xian Ng, Jiarui Liu, Mona T. Diab  

**Link**: [PDF](https://arxiv.org/pdf/2508.17573)  

**Abstract**: Large Language Models (LLMs) increasingly exhibit \textbf{anthropomorphism} characteristics -- human-like qualities portrayed across their outlook, language, behavior, and reasoning functions. Such characteristics enable more intuitive and engaging human-AI interactions. However, current research on anthropomorphism remains predominantly risk-focused, emphasizing over-trust and user deception while offering limited design guidance. We argue that anthropomorphism should instead be treated as a \emph{concept of design} that can be intentionally tuned to support user goals. Drawing from multiple disciplines, we propose that the anthropomorphism of an LLM-based artifact should reflect the interaction between artifact designers and interpreters. This interaction is facilitated by cues embedded in the artifact by the designers and the (cognitive) responses of the interpreters to the cues. Cues are categorized into four dimensions: \textit{perceptive, linguistic, behavioral}, and \textit{cognitive}. By analyzing the manifestation and effectiveness of each cue, we provide a unified taxonomy with actionable levers for practitioners. Consequently, we advocate for function-oriented evaluations of anthropomorphic design. 

---
# Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models? 

**Authors**: Hyeong Kyu Choi, Xiaojin Zhu, Yixuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.17536)  

**Abstract**: Multi-Agent Debate~(MAD) has emerged as a promising paradigm for improving the performance of large language models through collaborative reasoning. Despite recent advances, the key factors driving MAD's effectiveness remain unclear. In this work, we disentangle MAD into two key components--Majority Voting and inter-agent Debate--and assess their respective contributions. Through extensive experiments across seven NLP benchmarks, we find that Majority Voting alone accounts for most of the performance gains typically attributed to MAD. To explain this, we propose a theoretical framework that models debate as a stochastic process. We prove that it induces a martingale over agents' belief trajectories, implying that debate alone does not improve expected correctness. Guided by these insights, we demonstrate that targeted interventions, by biasing the belief update toward correction, can meaningfully enhance debate effectiveness. Overall, our findings suggest that while MAD has potential, simple ensembling methods remain strong and more reliable alternatives in many practical settings. Code is released in this https URL. 

---
# Improving French Synthetic Speech Quality via SSML Prosody Control 

**Authors**: Nassima Ould Ouali, Awais Hussain Sani, Ruben Bueno, Jonah Dauvet, Tim Luka Horstmann, Eric Moulines  

**Link**: [PDF](https://arxiv.org/pdf/2508.17494)  

**Abstract**: Despite recent advances, synthetic voices often lack expressiveness due to limited prosody control in commercial text-to-speech (TTS) systems. We introduce the first end-to-end pipeline that inserts Speech Synthesis Markup Language (SSML) tags into French text to control pitch, speaking rate, volume, and pause duration. We employ a cascaded architecture with two QLoRA-fine-tuned Qwen 2.5-7B models: one predicts phrase-break positions and the other performs regression on prosodic targets, generating commercial TTS-compatible SSML markup. Evaluated on a 14-hour French podcast corpus, our method achieves 99.2% F1 for break placement and reduces mean absolute error on pitch, rate, and volume by 25-40% compared with prompting-only large language models (LLMs) and a BiLSTM baseline. In perceptual evaluation involving 18 participants across over 9 hours of synthesized audio, SSML-enhanced speech generated by our pipeline significantly improves naturalness, with the mean opinion score increasing from 3.20 to 3.87 (p < 0.005). Additionally, 15 of 18 listeners preferred our enhanced synthesis. These results demonstrate substantial progress in bridging the expressiveness gap between synthetic and natural French speech. Our code is publicly available at this https URL. 

---
# Efficient Zero-Shot Long Document Classification by Reducing Context Through Sentence Ranking 

**Authors**: Prathamesh Kokate, Mitali Sarnaik, Manavi Khopade, Mukta Takalikar, Raviraj Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2508.17490)  

**Abstract**: Transformer-based models like BERT excel at short text classification but struggle with long document classification (LDC) due to input length limitations and computational inefficiencies. In this work, we propose an efficient, zero-shot approach to LDC that leverages sentence ranking to reduce input context without altering the model architecture. Our method enables the adaptation of models trained on short texts, such as headlines, to long-form documents by selecting the most informative sentences using a TF-IDF-based ranking strategy. Using the MahaNews dataset of long Marathi news articles, we evaluate three context reduction strategies that prioritize essential content while preserving classification accuracy. Our results show that retaining only the top 50\% ranked sentences maintains performance comparable to full-document inference while reducing inference time by up to 35\%. This demonstrates that sentence ranking is a simple yet effective technique for scalable and efficient zero-shot LDC. 

---
# Evaluating the Impact of Verbal Multiword Expressions on Machine Translation 

**Authors**: Linfeng Liu, Saptarshi Ghosh, Tianyu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17458)  

**Abstract**: Verbal multiword expressions (VMWEs) present significant challenges for natural language processing due to their complex and often non-compositional nature. While machine translation models have seen significant improvement with the advent of language models in recent years, accurately translating these complex linguistic structures remains an open problem. In this study, we analyze the impact of three VMWE categories -- verbal idioms, verb-particle constructions, and light verb constructions -- on machine translation quality from English to multiple languages. Using both established multiword expression datasets and sentences containing these language phenomena extracted from machine translation datasets, we evaluate how state-of-the-art translation systems handle these expressions. Our experimental results consistently show that VMWEs negatively affect translation quality. We also propose an LLM-based paraphrasing approach that replaces these expressions with their literal counterparts, demonstrating significant improvement in translation quality for verbal idioms and verb-particle constructions. 

---
# Persuasion Dynamics in LLMs: Investigating Robustness and Adaptability in Knowledge and Safety with DuET-PD 

**Authors**: Bryan Chen Zhengyu Tan, Daniel Wai Kit Chin, Zhengyuan Liu, Nancy F. Chen, Roy Ka-Wei Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.17450)  

**Abstract**: Large Language Models (LLMs) can struggle to balance gullibility to misinformation and resistance to valid corrections in persuasive dialogues, a critical challenge for reliable deployment. We introduce DuET-PD (Dual Evaluation for Trust in Persuasive Dialogues), a framework evaluating multi-turn stance-change dynamics across dual dimensions: persuasion type (corrective/misleading) and domain (knowledge via MMLU-Pro, and safety via SALAD-Bench). We find that even a state-of-the-art model like GPT-4o achieves only 27.32% accuracy in MMLU-Pro under sustained misleading persuasions. Moreover, results reveal a concerning trend of increasing sycophancy in newer open-source models. To address this, we introduce Holistic DPO, a training approach balancing positive and negative persuasion examples. Unlike prompting or resist-only training, Holistic DPO enhances both robustness to misinformation and receptiveness to corrections, improving Llama-3.1-8B-Instruct's accuracy under misleading persuasion in safety contexts from 4.21% to 76.54%. These contributions offer a pathway to developing more reliable and adaptable LLMs for multi-turn dialogue. Code is available at this https URL. 

---
# MahaParaphrase: A Marathi Paraphrase Detection Corpus and BERT-based Models 

**Authors**: Suramya Jadhav, Abhay Shanbhag, Amogh Thakurdesai, Ridhima Sinare, Ananya Joshi, Raviraj Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2508.17444)  

**Abstract**: Paraphrases are a vital tool to assist language understanding tasks such as question answering, style transfer, semantic parsing, and data augmentation tasks. Indic languages are complex in natural language processing (NLP) due to their rich morphological and syntactic variations, diverse scripts, and limited availability of annotated data. In this work, we present the L3Cube-MahaParaphrase Dataset, a high-quality paraphrase corpus for Marathi, a low resource Indic language, consisting of 8,000 sentence pairs, each annotated by human experts as either Paraphrase (P) or Non-paraphrase (NP). We also present the results of standard transformer-based BERT models on these datasets. The dataset and model are publicly shared at this https URL 

---
# DS@GT at CheckThat! 2025: A Simple Retrieval-First, LLM-Backed Framework for Claim Normalization 

**Authors**: Aleksandar Pramov, Jiangqin Ma, Bina Patel  

**Link**: [PDF](https://arxiv.org/pdf/2508.17402)  

**Abstract**: Claim normalization is an integral part of any automatic fact-check verification system. It parses the typically noisy claim data, such as social media posts into normalized claims, which are then fed into downstream veracity classification tasks. The CheckThat! 2025 Task 2 focuses specifically on claim normalization and spans 20 languages under monolingual and zero-shot conditions. Our proposed solution consists of a lightweight \emph{retrieval-first, LLM-backed} pipeline, in which we either dynamically prompt a GPT-4o-mini with in-context examples, or retrieve the closest normalization from the train dataset directly. On the official test set, the system ranks near the top for most monolingual tracks, achieving first place in 7 out of of the 13 languages. In contrast, the system underperforms in the zero-shot setting, highlighting the limitation of the proposed solution. 

---
# DashboardQA: Benchmarking Multimodal Agents for Question Answering on Interactive Dashboards 

**Authors**: Aaryaman Kartha, Ahmed Masry, Mohammed Saidul Islam, Thinh Lang, Shadikur Rahman, Ridwan Mahbub, Mizanur Rahman, Mahir Ahmed, Md Rizwan Parvez, Enamul Hoque, Shafiq Joty  

**Link**: [PDF](https://arxiv.org/pdf/2508.17398)  

**Abstract**: Dashboards are powerful visualization tools for data-driven decision-making, integrating multiple interactive views that allow users to explore, filter, and navigate data. Unlike static charts, dashboards support rich interactivity, which is essential for uncovering insights in real-world analytical workflows. However, existing question-answering benchmarks for data visualizations largely overlook this interactivity, focusing instead on static charts. This limitation severely constrains their ability to evaluate the capabilities of modern multimodal agents designed for GUI-based reasoning. To address this gap, we introduce DashboardQA, the first benchmark explicitly designed to assess how vision-language GUI agents comprehend and interact with real-world dashboards. The benchmark includes 112 interactive dashboards from Tableau Public and 405 question-answer pairs with interactive dashboards spanning five categories: multiple-choice, factoid, hypothetical, multi-dashboard, and conversational. By assessing a variety of leading closed- and open-source GUI agents, our analysis reveals their key limitations, particularly in grounding dashboard elements, planning interaction trajectories, and performing reasoning. Our findings indicate that interactive dashboard reasoning is a challenging task overall for all the VLMs evaluated. Even the top-performing agents struggle; for instance, the best agent based on Gemini-Pro-2.5 achieves only 38.69% accuracy, while the OpenAI CUA agent reaches just 22.69%, demonstrating the benchmark's significant difficulty. We release DashboardQA at this https URL 

---
# Agent-Testing Agent: A Meta-Agent for Automated Testing and Evaluation of Conversational AI Agents 

**Authors**: Sameer Komoravolu, Khalil Mrini  

**Link**: [PDF](https://arxiv.org/pdf/2508.17393)  

**Abstract**: LLM agents are increasingly deployed to plan, retrieve, and write with tools, yet evaluation still leans on static benchmarks and small human studies. We present the Agent-Testing Agent (ATA), a meta-agent that combines static code analysis, designer interrogation, literature mining, and persona-driven adversarial test generation whose difficulty adapts via judge feedback. Each dialogue is scored with an LLM-as-a-Judge (LAAJ) rubric and used to steer subsequent tests toward the agent's weakest capabilities. On a travel planner and a Wikipedia writer, the ATA surfaces more diverse and severe failures than expert annotators while matching severity, and finishes in 20--30 minutes versus ten-annotator rounds that took days. Ablating code analysis and web search increases variance and miscalibration, underscoring the value of evidence-grounded test generation. The ATA outputs quantitative metrics and qualitative bug reports for developers. We release the full methodology and open-source implementation for reproducible agent testing: this https URL 

---
# UI-Level Evaluation of ALLaM 34B: Measuring an Arabic-Centric LLM via HUMAIN Chat 

**Authors**: Omer Nacar  

**Link**: [PDF](https://arxiv.org/pdf/2508.17378)  

**Abstract**: Large language models (LLMs) trained primarily on English corpora often struggle to capture the linguistic and cultural nuances of Arabic. To address this gap, the Saudi Data and AI Authority (SDAIA) introduced the $ALLaM$ family of Arabic-focused models. The most capable of these available to the public, $ALLaM-34B$, was subsequently adopted by HUMAIN, who developed and deployed HUMAIN Chat, a closed conversational web service built on this model. This paper presents an expanded and refined UI-level evaluation of $ALLaM-34B$. Using a prompt pack spanning modern standard Arabic, five regional dialects, code-switching, factual knowledge, arithmetic and temporal reasoning, creative generation, and adversarial safety, we collected 115 outputs (23 prompts times 5 runs) and scored each with three frontier LLM judges (GPT-5, Gemini 2.5 Pro, Claude Sonnet-4). We compute category-level means with 95\% confidence intervals, analyze score distributions, and visualize dialect-wise metric heat maps. The updated analysis reveals consistently high performance on generation and code-switching tasks (both averaging 4.92/5), alongside strong results in MSA handling (4.74/5), solid reasoning ability (4.64/5), and improved dialect fidelity (4.21/5). Safety-related prompts show stable, reliable performance of (4.54/5). Taken together, these results position $ALLaM-34B$ as a robust and culturally grounded Arabic LLM, demonstrating both technical strength and practical readiness for real-world deployment. 

---
# The Arabic Generality Score: Another Dimension of Modeling Arabic Dialectness 

**Authors**: Sanad Shaban, Nizar Habash  

**Link**: [PDF](https://arxiv.org/pdf/2508.17347)  

**Abstract**: Arabic dialects form a diverse continuum, yet NLP models often treat them as discrete categories. Recent work addresses this issue by modeling dialectness as a continuous variable, notably through the Arabic Level of Dialectness (ALDi). However, ALDi reduces complex variation to a single dimension. We propose a complementary measure: the Arabic Generality Score (AGS), which quantifies how widely a word is used across dialects. We introduce a pipeline that combines word alignment, etymology-aware edit distance, and smoothing to annotate a parallel corpus with word-level AGS. A regression model is then trained to predict AGS in context. Our approach outperforms strong baselines, including state-of-the-art dialect ID systems, on a multi-dialect benchmark. AGS offers a scalable, linguistically grounded way to model lexical generality, enriching representations of Arabic dialectness. 

---
# Capturing Legal Reasoning Paths from Facts to Law in Court Judgments using Knowledge Graphs 

**Authors**: Ryoma Kondo, Riona Matsuoka, Takahiro Yoshida, Kazuyuki Yamasawa, Ryohei Hisano  

**Link**: [PDF](https://arxiv.org/pdf/2508.17340)  

**Abstract**: Court judgments reveal how legal rules have been interpreted and applied to facts, providing a foundation for understanding structured legal reasoning. However, existing automated approaches for capturing legal reasoning, including large language models, often fail to identify the relevant legal context, do not accurately trace how facts relate to legal norms, and may misrepresent the layered structure of judicial reasoning. These limitations hinder the ability to capture how courts apply the law to facts in practice. In this paper, we address these challenges by constructing a legal knowledge graph from 648 Japanese administrative court decisions. Our method extracts components of legal reasoning using prompt-based large language models, normalizes references to legal provisions, and links facts, norms, and legal applications through an ontology of legal inference. The resulting graph captures the full structure of legal reasoning as it appears in real court decisions, making implicit reasoning explicit and machine-readable. We evaluate our system using expert annotated data, and find that it achieves more accurate retrieval of relevant legal provisions from facts than large language model baselines and retrieval-augmented methods. 

---
# DropLoRA: Sparse Low-Rank Adaptation for Parameter-Efficient Fine-Tuning 

**Authors**: Haojie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17337)  

**Abstract**: LoRA-based large model parameter-efficient fine-tuning (PEFT) methods use low-rank de- composition to approximate updates to model parameters. However, compared to full- parameter fine-tuning, low-rank updates often lead to a performance gap in downstream tasks. To address this, we introduce DropLoRA, a novel pruning-based approach that focuses on pruning the rank dimension. Unlike conven- tional methods that attempt to overcome the low-rank bottleneck, DropLoRA innovatively integrates a pruning module between the two low-rank matrices in LoRA to simulate dy- namic subspace learning. This dynamic low- rank subspace learning allows DropLoRA to overcome the limitations of traditional LoRA, which operates within a static subspace. By continuously adapting the learning subspace, DropLoRA significantly boosts performance without incurring additional training or infer- ence costs. Our experimental results demon- strate that DropLoRA consistently outperforms LoRA in fine-tuning the LLaMA series across a wide range of large language model gener- ation tasks, including commonsense reason- ing, mathematical reasoning, code generation, and instruction-following. Our code is avail- able at this https URL. 

---
# Omne-R1: Learning to Reason with Memory for Multi-hop Question Answering 

**Authors**: Boyuan Liu, Feng Ji, Jiayan Nan, Han Zhao, Weiling Chen, Shihao Xu, Xing Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.17330)  

**Abstract**: This paper introduces Omne-R1, a novel approach designed to enhance multi-hop question answering capabilities on schema-free knowledge graphs by integrating advanced reasoning models. Our method employs a multi-stage training workflow, including two reinforcement learning phases and one supervised fine-tuning phase. We address the challenge of limited suitable knowledge graphs and QA data by constructing domain-independent knowledge graphs and auto-generating QA pairs. Experimental results show significant improvements in answering multi-hop questions, with notable performance gains on more complex 3+ hop questions. Our proposed training framework demonstrates strong generalization abilities across diverse knowledge domains. 

---
# CultranAI at PalmX 2025: Data Augmentation for Cultural Knowledge Representation 

**Authors**: Hunzalah Hassan Bhatti, Youssef Ahmed, Md Arid Hasan, Firoj Alam  

**Link**: [PDF](https://arxiv.org/pdf/2508.17324)  

**Abstract**: In this paper, we report our participation to the PalmX cultural evaluation shared task. Our system, CultranAI, focused on data augmentation and LoRA fine-tuning of large language models (LLMs) for Arabic cultural knowledge representation. We benchmarked several LLMs to identify the best-performing model for the task. In addition to utilizing the PalmX dataset, we augmented it by incorporating the Palm dataset and curated a new dataset of over 22K culturally grounded multiple-choice questions (MCQs). Our experiments showed that the Fanar-1-9B-Instruct model achieved the highest performance. We fine-tuned this model on the combined augmented dataset of 22K+ MCQs. On the blind test set, our submitted system ranked 5th with an accuracy of 70.50%, while on the PalmX development set, it achieved an accuracy of 84.1%. 

---
# Handling Students Dropouts in an LLM-driven Interactive Online Course Using Language Models 

**Authors**: Yuanchun Wang, Yiyang Fu, Jifan Yu, Daniel Zhang-Li, Zheyuan Zhang, Joy Lim Jia Yin, Yucheng Wang, Peng Zhou, Jing Zhang, Huiqin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17310)  

**Abstract**: Interactive online learning environments, represented by Massive AI-empowered Courses (MAIC), leverage LLM-driven multi-agent systems to transform passive MOOCs into dynamic, text-based platforms, enhancing interactivity through LLMs. This paper conducts an empirical study on a specific MAIC course to explore three research questions about dropouts in these interactive online courses: (1) What factors might lead to dropouts? (2) Can we predict dropouts? (3) Can we reduce dropouts? We analyze interaction logs to define dropouts and identify contributing factors. Our findings reveal strong links between dropout behaviors and textual interaction patterns. We then propose a course-progress-adaptive dropout prediction framework (CPADP) to predict dropouts with at most 95.4% accuracy. Based on this, we design a personalized email recall agent to re-engage at-risk students. Applied in the deployed MAIC system with over 3,000 students, the feasibility and effectiveness of our approach have been validated on students with diverse backgrounds. 

---
# From Language to Action: A Review of Large Language Models as Autonomous Agents and Tool Users 

**Authors**: Sadia Sultana Chowa, Riasad Alvi, Subhey Sadi Rahman, Md Abdur Rahman, Mohaimenul Azam Khan Raiaan, Md Rafiqul Islam, Mukhtar Hussain, Sami Azam  

**Link**: [PDF](https://arxiv.org/pdf/2508.17281)  

**Abstract**: The pursuit of human-level artificial intelligence (AI) has significantly advanced the development of autonomous agents and Large Language Models (LLMs). LLMs are now widely utilized as decision-making agents for their ability to interpret instructions, manage sequential tasks, and adapt through feedback. This review examines recent developments in employing LLMs as autonomous agents and tool users and comprises seven research questions. We only used the papers published between 2023 and 2025 in conferences of the A* and A rank and Q1 journals. A structured analysis of the LLM agents' architectural design principles, dividing their applications into single-agent and multi-agent systems, and strategies for integrating external tools is presented. In addition, the cognitive mechanisms of LLM, including reasoning, planning, and memory, and the impact of prompting methods and fine-tuning procedures on agent performance are also investigated. Furthermore, we evaluated current benchmarks and assessment protocols and have provided an analysis of 68 publicly available datasets to assess the performance of LLM-based agents in various tasks. In conducting this review, we have identified critical findings on verifiable reasoning of LLMs, the capacity for self-improvement, and the personalization of LLM-based agents. Finally, we have discussed ten future research directions to overcome these gaps. 

---
# Are You Sure You're Positive? Consolidating Chain-of-Thought Agents with Uncertainty Quantification for Aspect-Category Sentiment Analysis 

**Authors**: Filippos Ventirozos, Peter Appleby, Matthew Shardlow  

**Link**: [PDF](https://arxiv.org/pdf/2508.17258)  

**Abstract**: Aspect-category sentiment analysis provides granular insights by identifying specific themes within product reviews that are associated with particular opinions. Supervised learning approaches dominate the field. However, data is scarce and expensive to annotate for new domains. We argue that leveraging large language models in a zero-shot setting is beneficial where the time and resources required for dataset annotation are limited. Furthermore, annotation bias may lead to strong results using supervised methods but transfer poorly to new domains in contexts that lack annotations and demand reproducibility. In our work, we propose novel techniques that combine multiple chain-of-thought agents by leveraging large language models' token-level uncertainty scores. We experiment with the 3B and 70B+ parameter size variants of Llama and Qwen models, demonstrating how these approaches can fulfil practical needs and opening a discussion on how to gauge accuracy in label-scarce conditions. 

---
# Routing Distilled Knowledge via Mixture of LoRA Experts for Large Language Model based Bundle Generation 

**Authors**: Kaidong Feng, Zhu Sun, Hui Fang, Jie Yang, Wenyuan Liu, Yew-Soon Ong  

**Link**: [PDF](https://arxiv.org/pdf/2508.17250)  

**Abstract**: Large Language Models (LLMs) have shown potential in automatic bundle generation but suffer from prohibitive computational costs. Although knowledge distillation offers a pathway to more efficient student models, our preliminary study reveals that naively integrating diverse types of distilled knowledge from teacher LLMs into student LLMs leads to knowledge conflict, negatively impacting the performance of bundle generation. To address this, we propose RouteDK, a framework for routing distilled knowledge through a mixture of LoRA expert architecture. Specifically, we first distill knowledge from the teacher LLM for bundle generation in two complementary types: high-level knowledge (generalizable rules) and fine-grained knowledge (session-specific reasoning). We then train knowledge-specific LoRA experts for each type of knowledge together with a base LoRA expert. For effective integration, we propose a dynamic fusion module, featuring an input-aware router, where the router balances expert contributions by dynamically determining optimal weights based on input, thereby effectively mitigating knowledge conflicts. To further improve inference reliability, we design an inference-time enhancement module to reduce variance and mitigate suboptimal reasoning. Experiments on three public datasets show that our RouteDK achieves accuracy comparable to or even better than the teacher LLM, while maintaining strong computational efficiency. In addition, it outperforms state-of-the-art approaches for bundle generation. 

---
# ClaimGen-CN: A Large-scale Chinese Dataset for Legal Claim Generation 

**Authors**: Siying Zhou, Yiquan Wu, Hui Chen, Xavier Hu, Kun Kuang, Adam Jatowt, Ming Hu, Chunyan Zheng, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17234)  

**Abstract**: Legal claims refer to the plaintiff's demands in a case and are essential to guiding judicial reasoning and case resolution. While many works have focused on improving the efficiency of legal professionals, the research on helping non-professionals (e.g., plaintiffs) remains unexplored. This paper explores the problem of legal claim generation based on the given case's facts. First, we construct ClaimGen-CN, the first dataset for Chinese legal claim generation task, from various real-world legal disputes. Additionally, we design an evaluation metric tailored for assessing the generated claims, which encompasses two essential dimensions: factuality and clarity. Building on this, we conduct a comprehensive zero-shot evaluation of state-of-the-art general and legal-domain large language models. Our findings highlight the limitations of the current models in factual precision and expressive clarity, pointing to the need for more targeted development in this domain. To encourage further exploration of this important task, we will make the dataset publicly available. 

---
# SSFO: Self-Supervised Faithfulness Optimization for Retrieval-Augmented Generation 

**Authors**: Xiaqiang Tang, Yi Wang, Keyu Hu, Rui Xu, Chuang Li, Weigao Sun, Jian Li, Sihong Xie  

**Link**: [PDF](https://arxiv.org/pdf/2508.17225)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems require Large Language Models (LLMs) to generate responses that are faithful to the retrieved context. However, faithfulness hallucination remains a critical challenge, as existing methods often require costly supervision and post-training or significant inference burdens. To overcome these limitations, we introduce Self-Supervised Faithfulness Optimization (SSFO), the first self-supervised alignment approach for enhancing RAG faithfulness. SSFO constructs preference data pairs by contrasting the model's outputs generated with and without the context. Leveraging Direct Preference Optimization (DPO), SSFO aligns model faithfulness without incurring labeling costs or additional inference burden. We theoretically and empirically demonstrate that SSFO leverages a benign form of \emph{likelihood displacement}, transferring probability mass from parametric-based tokens to context-aligned tokens. Based on this insight, we propose a modified DPO loss function to encourage likelihood displacement. Comprehensive evaluations show that SSFO significantly outperforms existing methods, achieving state-of-the-art faithfulness on multiple context-based question-answering datasets. Notably, SSFO exhibits strong generalization, improving cross-lingual faithfulness and preserving general instruction-following capabilities. We release our code and model at the anonymous link: this https URL 

---
# Active Domain Knowledge Acquisition with \$100 Budget: Enhancing LLMs via Cost-Efficient, Expert-Involved Interaction in Sensitive Domains 

**Authors**: Yang Wu, Raha Moraffah, Rujing Yao, Jinhong Yu, Zhimin Tao, Xiaozhong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17202)  

**Abstract**: Large Language Models (LLMs) have demonstrated an impressive level of general knowledge. However, they often struggle in highly specialized and cost-sensitive domains such as drug discovery and rare disease research due to the lack of expert knowledge. In this paper, we propose a novel framework (PU-ADKA) designed to efficiently enhance domain-specific LLMs by actively engaging domain experts within a fixed budget. Unlike traditional fine-tuning approaches, PU-ADKA selectively identifies and queries the most appropriate expert from a team, taking into account each expert's availability, knowledge boundaries, and consultation costs. We train PU-ADKA using simulations on PubMed data and validate it through both controlled expert interactions and real-world deployment with a drug development team, demonstrating its effectiveness in enhancing LLM performance in specialized domains under strict budget constraints. In addition to outlining our methodological innovations and experimental results, we introduce a new benchmark dataset, CKAD, for cost-effective LLM domain knowledge acquisition to foster further research in this challenging area. 

---
# Towards Alignment-Centric Paradigm: A Survey of Instruction Tuning in Large Language Models 

**Authors**: Xudong Han, Junjie Yang, Tianyang Wang, Ziqian Bi, Junfeng Hao, Junhao Song  

**Link**: [PDF](https://arxiv.org/pdf/2508.17184)  

**Abstract**: Instruction tuning is a pivotal technique for aligning large language models (LLMs) with human intentions, safety constraints, and domain-specific requirements. This survey provides a comprehensive overview of the full pipeline, encompassing (i) data collection methodologies, (ii) full-parameter and parameter-efficient fine-tuning strategies, and (iii) evaluation protocols. We categorized data construction into three major paradigms: expert annotation, distillation from larger models, and self-improvement mechanisms, each offering distinct trade-offs between quality, scalability, and resource cost. Fine-tuning techniques range from conventional supervised training to lightweight approaches, such as low-rank adaptation (LoRA) and prefix tuning, with a focus on computational efficiency and model reusability. We further examine the challenges of evaluating faithfulness, utility, and safety across multilingual and multimodal scenarios, highlighting the emergence of domain-specific benchmarks in healthcare, legal, and financial applications. Finally, we discuss promising directions for automated data generation, adaptive optimization, and robust evaluation frameworks, arguing that a closer integration of data, algorithms, and human feedback is essential for advancing instruction-tuned LLMs. This survey aims to serve as a practical reference for researchers and practitioners seeking to design LLMs that are both effective and reliably aligned with human intentions. 

---
# The Impact of Annotator Personas on LLM Behavior Across the Perspectivism Spectrum 

**Authors**: Olufunke O. Sarumi, Charles Welch, Daniel Braun, Jörg Schlötterer  

**Link**: [PDF](https://arxiv.org/pdf/2508.17164)  

**Abstract**: In this work, we explore the capability of Large Language Models (LLMs) to annotate hate speech and abusiveness while considering predefined annotator personas within the strong-to-weak data perspectivism spectra. We evaluated LLM-generated annotations against existing annotator modeling techniques for perspective modeling. Our findings show that LLMs selectively use demographic attributes from the personas. We identified prototypical annotators, with persona features that show varying degrees of alignment with the original human annotators. Within the data perspectivism paradigm, annotator modeling techniques that do not explicitly rely on annotator information performed better under weak data perspectivism compared to both strong data perspectivism and human annotations, suggesting LLM-generated views tend towards aggregation despite subjective prompting. However, for more personalized datasets tailored to strong perspectivism, the performance of LLM annotator modeling approached, but did not exceed, human annotators. 

---
# Quantifying Language Disparities in Multilingual Large Language Models 

**Authors**: Songbo Hu, Ivan Vulić, Anna Korhonen  

**Link**: [PDF](https://arxiv.org/pdf/2508.17162)  

**Abstract**: Results reported in large-scale multilingual evaluations are often fragmented and confounded by factors such as target languages, differences in experimental setups, and model choices. We propose a framework that disentangles these confounding variables and introduces three interpretable metrics--the performance realisation ratio, its coefficient of variation, and language potential--enabling a finer-grained and more insightful quantification of actual performance disparities across both (i) models and (ii) languages. Through a case study of 13 model variants on 11 multilingual datasets, we demonstrate that our framework provides a more reliable measurement of model performance and language disparities, particularly for low-resource languages, which have so far proven challenging to evaluate. Importantly, our results reveal that higher overall model performance does not necessarily imply greater fairness across languages. 

---
# SPORTSQL: An Interactive System for Real-Time Sports Reasoning and Visualization 

**Authors**: Sebastian Martinez, Naman Ahuja, Fenil Bardoliya, Chris Bryan, Vivek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2508.17157)  

**Abstract**: We present a modular, interactive system, SPORTSQL, for natural language querying and visualization of dynamic sports data, with a focus on the English Premier League (EPL). The system translates user questions into executable SQL over a live, temporally indexed database constructed from real-time Fantasy Premier League (FPL) data. It supports both tabular and visual outputs, leveraging the symbolic reasoning capabilities of Large Language Models (LLMs) for query parsing, schema linking, and visualization selection. To evaluate system performance, we introduce the Dynamic Sport Question Answering benchmark (DSQABENCH), comprising 1,700+ queries annotated with SQL programs, gold answers, and database snapshots. Our demo highlights how non-expert users can seamlessly explore evolving sports statistics through a natural, conversational interface. 

---
# Natural Language Satisfiability: Exploring the Problem Distribution and Evaluating Transformer-based Language Models 

**Authors**: Tharindu Madusanka, Ian Pratt-Hartmann, Riza Batista-Navarro  

**Link**: [PDF](https://arxiv.org/pdf/2508.17153)  

**Abstract**: Efforts to apply transformer-based language models (TLMs) to the problem of reasoning in natural language have enjoyed ever-increasing success in recent years. The most fundamental task in this area to which nearly all others can be reduced is that of determining satisfiability. However, from a logical point of view, satisfiability problems vary along various dimensions, which may affect TLMs' ability to learn how to solve them. The problem instances of satisfiability in natural language can belong to different computational complexity classes depending on the language fragment in which they are expressed. Although prior research has explored the problem of natural language satisfiability, the above-mentioned point has not been discussed adequately. Hence, we investigate how problem instances from varying computational complexity classes and having different grammatical constructs impact TLMs' ability to learn rules of inference. Furthermore, to faithfully evaluate TLMs, we conduct an empirical study to explore the distribution of satisfiability problems. 

---
# Geolocation-Aware Robust Spoken Language Identification 

**Authors**: Qingzheng Wang, Hye-jin Shim, Jiancheng Sun, Shinji Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2508.17148)  

**Abstract**: While Self-supervised Learning (SSL) has significantly improved Spoken Language Identification (LID), existing models often struggle to consistently classify dialects and accents of the same language as a unified class. To address this challenge, we propose geolocation-aware LID, a novel approach that incorporates language-level geolocation information into the SSL-based LID model. Specifically, we introduce geolocation prediction as an auxiliary task and inject the predicted vectors into intermediate representations as conditioning signals. This explicit conditioning encourages the model to learn more unified representations for dialectal and accented variations. Experiments across six multilingual datasets demonstrate that our approach improves robustness to intra-language variations and unseen domains, achieving new state-of-the-art accuracy on FLEURS (97.7%) and 9.7% relative improvement on ML-SUPERB 2.0 dialect set. 

---
# The Power of Framing: How News Headlines Guide Search Behavior 

**Authors**: Amrit Poudel, Maria Milkowski, Tim Weninger  

**Link**: [PDF](https://arxiv.org/pdf/2508.17131)  

**Abstract**: Search engines play a central role in how people gather information, but subtle cues like headline framing may influence not only what users believe but also how they search. While framing effects on judgment are well documented, their impact on subsequent search behavior is less understood. We conducted a controlled experiment where participants issued queries and selected from headlines filtered by specific linguistic frames. Headline framing significantly shaped follow-up queries: conflict and strategy frames disrupted alignment with prior selections, while episodic frames led to more concrete queries than thematic ones. We also observed modest short-term frame persistence that declined over time. These results suggest that even brief exposure to framing can meaningfully alter the direction of users information-seeking behavior. 

---
# A Straightforward Pipeline for Targeted Entailment and Contradiction Detection 

**Authors**: Antonin Sulc  

**Link**: [PDF](https://arxiv.org/pdf/2508.17127)  

**Abstract**: Finding the relationships between sentences in a document is crucial for tasks like fact-checking, argument mining, and text summarization. A key challenge is to identify which sentences act as premises or contradictions for a specific claim. Existing methods often face a trade-off: transformer attention mechanisms can identify salient textual connections but lack explicit semantic labels, while Natural Language Inference (NLI) models can classify relationships between sentence pairs but operate independently of contextual saliency. In this work, we introduce a method that combines the strengths of both approaches for a targeted analysis. Our pipeline first identifies candidate sentences that are contextually relevant to a user-selected target sentence by aggregating token-level attention scores. It then uses a pretrained NLI model to classify each candidate as a premise (entailment) or contradiction. By filtering NLI-identified relationships with attention-based saliency scores, our method efficiently isolates the most significant semantic relationships for any given claim in a text. 

---
# Token Homogenization under Positional Bias 

**Authors**: Viacheslav Yusupov, Danil Maksimov, Ameliia Alaeva, Tatiana Zaitceva, Antipina Anna, Anna Vasileva, Chenlin Liu, Rayuth Chheng, Danil Sazanakov, Andrey Chetvergov, Alina Ermilova, Egor Shvetsov  

**Link**: [PDF](https://arxiv.org/pdf/2508.17126)  

**Abstract**: This paper investigates token homogenization - the convergence of token representations toward uniformity across transformer layers and its relationship to positional bias in large language models. We empirically examine whether homogenization occurs and how positional bias amplifies this effect. Through layer-wise similarity analysis and controlled experiments, we demonstrate that tokens systematically lose distinctiveness during processing, particularly when biased toward extremal positions. Our findings confirm both the existence of homogenization and its dependence on positional attention mechanisms. 

---
# Linguistic Neuron Overlap Patterns to Facilitate Cross-lingual Transfer on Low-resource Languages 

**Authors**: Yuemei Xu, Kexin Xu, Jian Zhou, Ling Hu, Lin Gui  

**Link**: [PDF](https://arxiv.org/pdf/2508.17078)  

**Abstract**: The current Large Language Models (LLMs) face significant challenges in improving performance on low-resource languages and urgently need data-efficient methods without costly fine-tuning. From the perspective of language-bridge, we propose BridgeX-ICL, a simple yet effective method to improve zero-shot Cross-lingual In-Context Learning (X-ICL) for low-resource languages. Unlike existing works focusing on language-specific neurons, BridgeX-ICL explores whether sharing neurons can improve cross-lingual performance in LLMs or not. We construct neuron probe data from the ground-truth MUSE bilingual dictionaries, and define a subset of language overlap neurons accordingly, to ensure full activation of these anchored neurons. Subsequently, we propose an HSIC-based metric to quantify LLMs' internal linguistic spectrum based on overlap neurons, which guides optimal bridge selection. The experiments conducted on 2 cross-lingual tasks and 15 language pairs from 7 diverse families (covering both high-low and moderate-low pairs) validate the effectiveness of BridgeX-ICL and offer empirical insights into the underlying multilingual mechanisms of LLMs. 

---
# GRAID: Synthetic Data Generation with Geometric Constraints and Multi-Agentic Reflection for Harmful Content Detection 

**Authors**: Melissa Kazemi Rad, Alberto Purpura, Himanshu Kumar, Emily Chen, Mohammad Shahed Sorower  

**Link**: [PDF](https://arxiv.org/pdf/2508.17057)  

**Abstract**: We address the problem of data scarcity in harmful text classification for guardrailing applications and introduce GRAID (Geometric and Reflective AI-Driven Data Augmentation), a novel pipeline that leverages Large Language Models (LLMs) for dataset augmentation. GRAID consists of two stages: (i) generation of geometrically controlled examples using a constrained LLM, and (ii) augmentation through a multi-agentic reflective process that promotes stylistic diversity and uncovers edge cases. This combination enables both reliable coverage of the input space and nuanced exploration of harmful content. Using two benchmark data sets, we demonstrate that augmenting a harmful text classification dataset with GRAID leads to significant improvements in downstream guardrail model performance. 

---
# Improving Table Understanding with LLMs and Entity-Oriented Search 

**Authors**: Thi-Nhung Nguyen, Hoang Ngo, Dinh Phung, Thuy-Trang Vu, Dat Quoc Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2508.17028)  

**Abstract**: Our work addresses the challenges of understanding tables. Existing methods often struggle with the unpredictable nature of table content, leading to a reliance on preprocessing and keyword matching. They also face limitations due to the lack of contextual information, which complicates the reasoning processes of large language models (LLMs). To overcome these challenges, we introduce an entity-oriented search method to improve table understanding with LLMs. This approach effectively leverages the semantic similarities between questions and table data, as well as the implicit relationships between table cells, minimizing the need for data preprocessing and keyword matching. Additionally, it focuses on table entities, ensuring that table cells are semantically tightly bound, thereby enhancing contextual clarity. Furthermore, we pioneer the use of a graph query language for table understanding, establishing a new research direction. Experiments show that our approach achieves new state-of-the-art performances on standard benchmarks WikiTableQuestions and TabFact. 

---
# EduRABSA: An Education Review Dataset for Aspect-based Sentiment Analysis Tasks 

**Authors**: Yan Cathy Hua, Paul Denny, Jörg Wicker, Katerina Taskova  

**Link**: [PDF](https://arxiv.org/pdf/2508.17008)  

**Abstract**: Every year, most educational institutions seek and receive an enormous volume of text feedback from students on courses, teaching, and overall experience. Yet, turning this raw feedback into useful insights is far from straightforward. It has been a long-standing challenge to adopt automatic opinion mining solutions for such education review text data due to the content complexity and low-granularity reporting requirements. Aspect-based Sentiment Analysis (ABSA) offers a promising solution with its rich, sub-sentence-level opinion mining capabilities. However, existing ABSA research and resources are very heavily focused on the commercial domain. In education, they are scarce and hard to develop due to limited public datasets and strict data protection. A high-quality, annotated dataset is urgently needed to advance research in this under-resourced area. In this work, we present EduRABSA (Education Review ABSA), the first public, annotated ABSA education review dataset that covers three review subject types (course, teaching staff, university) in the English language and all main ABSA tasks, including the under-explored implicit aspect and implicit opinion extraction. We also share ASQE-DPT (Data Processing Tool), an offline, lightweight, installation-free manual data annotation tool that generates labelled datasets for comprehensive ABSA tasks from a single-task annotation. Together, these resources contribute to the ABSA community and education domain by removing the dataset barrier, supporting research transparency and reproducibility, and enabling the creation and sharing of further resources. The dataset, annotation tool, and scripts and statistics for dataset processing and sampling are available at this https URL. 

---
# Planning for Success: Exploring LLM Long-term Planning Capabilities in Table Understanding 

**Authors**: Thi-Nhung Nguyen, Hoang Ngo, Dinh Phung, Thuy-Trang Vu, Dat Quoc Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2508.17005)  

**Abstract**: Table understanding is key to addressing challenging downstream tasks such as table-based question answering and fact verification. Recent works have focused on leveraging Chain-of-Thought and question decomposition to solve complex questions requiring multiple operations on tables. However, these methods often suffer from a lack of explicit long-term planning and weak inter-step connections, leading to miss constraints within questions. In this paper, we propose leveraging the long-term planning capabilities of large language models (LLMs) to enhance table understanding. Our approach enables the execution of a long-term plan, where the steps are tightly interconnected and serve the ultimate goal, an aspect that methods based on Chain-of-Thought and question decomposition lack. In addition, our method effectively minimizes the inclusion of unnecessary details in the process of solving the next short-term goals, a limitation of methods based on Chain-of-Thought. Extensive experiments demonstrate that our method outperforms strong baselines and achieves state-of-the-art performance on WikiTableQuestions and TabFact datasets. 

---
# KL-Regularised Q-Learning: A Token-level Action-Value perspective on Online RLHF 

**Authors**: Jason R Brown, Lennie Wells, Edward James Young, Sergio Bacallado  

**Link**: [PDF](https://arxiv.org/pdf/2508.17000)  

**Abstract**: Proximal Policy Optimisation (PPO) is an established and effective policy gradient algorithm used for Language Model Reinforcement Learning from Human Feedback (LM-RLHF). PPO performs well empirically but has a heuristic motivation and handles the KL-divergence constraint used in LM-RLHF in an ad-hoc manner. In this paper, we develop a a new action-value RL method for the LM-RLHF setting, KL-regularised Q-Learning (KLQ). We then show that our method is equivalent to a version of PPO in a certain specific sense, despite its very different motivation. Finally, we benchmark KLQ on two key language generation tasks -- summarisation and single-turn dialogue. We demonstrate that KLQ performs on-par with PPO at optimising the LM-RLHF objective, and achieves a consistently higher win-rate against PPO on LLM-as-a-judge evaluations. 

---
# DeAR: Dual-Stage Document Reranking with Reasoning Agents via LLM Distillation 

**Authors**: Abdelrahman Abdallah, Jamshid Mozafari, Bhawna Piryani, Adam Jatowt  

**Link**: [PDF](https://arxiv.org/pdf/2508.16998)  

**Abstract**: Large Language Models (LLMs) have transformed listwise document reranking by enabling global reasoning over candidate sets, yet single models often struggle to balance fine-grained relevance scoring with holistic cross-document analysis. We propose \textbf{De}ep\textbf{A}gent\textbf{R}ank (\textbf{\DeAR}), an open-source framework that decouples these tasks through a dual-stage approach, achieving superior accuracy and interpretability. In \emph{Stage 1}, we distill token-level relevance signals from a frozen 13B LLaMA teacher into a compact \{3, 8\}B student model using a hybrid of cross-entropy, RankNet, and KL divergence losses, ensuring robust pointwise scoring. In \emph{Stage 2}, we attach a second LoRA adapter and fine-tune on 20K GPT-4o-generated chain-of-thought permutations, enabling listwise reasoning with natural-language justifications. Evaluated on TREC-DL19/20, eight BEIR datasets, and NovelEval-2306, \DeAR surpasses open-source baselines by +5.1 nDCG@5 on DL20 and achieves 90.97 nDCG@10 on NovelEval, outperforming GPT-4 by +3.09. Without fine-tuning on Wikipedia, DeAR also excels in open-domain QA, achieving 54.29 Top-1 accuracy on Natural Questions, surpassing baselines like MonoT5, UPR, and RankGPT. Ablations confirm that dual-loss distillation ensures stable calibration, making \DeAR a highly effective and interpretable solution for modern reranking systems.\footnote{Dataset and code available at this https URL.}. 

---
# GRADE: Generating multi-hop QA and fine-gRAined Difficulty matrix for RAG Evaluation 

**Authors**: Jeongsoo Lee, Daeyong Kwon, Kyohoon Jin  

**Link**: [PDF](https://arxiv.org/pdf/2508.16994)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems are widely adopted in knowledge-intensive NLP tasks, but current evaluations often overlook the structural complexity and multi-step reasoning required in real-world scenarios. These benchmarks overlook key factors such as the interaction between retrieval difficulty and reasoning depth. To address this gap, we propose \textsc{GRADE}, a novel evaluation framework that models task difficulty along two orthogonal dimensions: (1) reasoning depth, defined by the number of inference steps (hops), and (2) semantic distance between the query and its supporting evidence. We construct a synthetic multi-hop QA dataset from factual news articles by extracting knowledge graphs and augmenting them through semantic clustering to recover missing links, allowing us to generate diverse and difficulty-controlled queries. Central to our framework is a 2D difficulty matrix that combines generator-side and retriever-side difficulty. Experiments across multiple domains and models show that error rates strongly correlate with our difficulty measures, validating their diagnostic utility. \textsc{GRADE} enables fine-grained analysis of RAG performance and provides a scalable foundation for evaluating and improving multi-hop reasoning in real-world applications. 

---
# ReFactX: Scalable Reasoning with Reliable Facts via Constrained Generation 

**Authors**: Riccardo Pozzi, Matteo Palmonari, Andrea Coletta, Luigi Bellomarini, Jens Lehmann, Sahar Vahdati  

**Link**: [PDF](https://arxiv.org/pdf/2508.16983)  

**Abstract**: Knowledge gaps and hallucinations are persistent challenges for Large Language Models (LLMs), which generate unreliable responses when lacking the necessary information to fulfill user instructions. Existing approaches, such as Retrieval-Augmented Generation (RAG) and tool use, aim to address these issues by incorporating external knowledge. Yet, they rely on additional models or services, resulting in complex pipelines, potential error propagation, and often requiring the model to process a large number of tokens. In this paper, we present a scalable method that enables LLMs to access external knowledge without depending on retrievers or auxiliary models. Our approach uses constrained generation with a pre-built prefix-tree index. Triples from a Knowledge Graph are verbalized in textual facts, tokenized, and indexed in a prefix tree for efficient access. During inference, to acquire external knowledge, the LLM generates facts with constrained generation which allows only sequences of tokens that form an existing fact. We evaluate our proposal on Question Answering and show that it scales to large knowledge bases (800 million facts), adapts to domain-specific data, and achieves effective results. These gains come with minimal generation-time overhead. ReFactX code is available at this https URL. 

---
# Decoding Alignment: A Critical Survey of LLM Development Initiatives through Value-setting and Data-centric Lens 

**Authors**: Ilias Chalkidis  

**Link**: [PDF](https://arxiv.org/pdf/2508.16982)  

**Abstract**: AI Alignment, primarily in the form of Reinforcement Learning from Human Feedback (RLHF), has been a cornerstone of the post-training phase in developing Large Language Models (LLMs). It has also been a popular research topic across various disciplines beyond Computer Science, including Philosophy and Law, among others, highlighting the socio-technical challenges involved. Nonetheless, except for the computational techniques related to alignment, there has been limited focus on the broader picture: the scope of these processes, which primarily rely on the selected objectives (values), and the data collected and used to imprint such objectives into the models. This work aims to reveal how alignment is understood and applied in practice from a value-setting and data-centric perspective. For this purpose, we investigate and survey (`audit') publicly available documentation released by 6 LLM development initiatives by 5 leading organizations shaping this technology, focusing on proprietary (OpenAI's GPT, Anthropic's Claude, Google's Gemini) and open-weight (Meta's Llama, Google's Gemma, and Alibaba's Qwen) initiatives, all published in the last 3 years. The findings are documented in detail per initiative, while there is also an overall summary concerning different aspects, mainly from a value-setting and data-centric perspective. On the basis of our findings, we discuss a series of broader related concerns. 

---
# Explaining Black-box Language Models with Knowledge Probing Systems: A Post-hoc Explanation Perspective 

**Authors**: Yunxiao Zhao, Hao Xu, Zhiqiang Wang, Xiaoli Li, Jiye Liang, Ru Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.16969)  

**Abstract**: Pre-trained Language Models (PLMs) are trained on large amounts of unlabeled data, yet they exhibit remarkable reasoning skills. However, the trustworthiness challenges posed by these black-box models have become increasingly evident in recent years. To alleviate this problem, this paper proposes a novel Knowledge-guided Probing approach called KnowProb in a post-hoc explanation way, which aims to probe whether black-box PLMs understand implicit knowledge beyond the given text, rather than focusing only on the surface level content of the text. We provide six potential explanations derived from the underlying content of the given text, including three knowledge-based understanding and three association-based reasoning. In experiments, we validate that current small-scale (or large-scale) PLMs only learn a single distribution of representation, and still face significant challenges in capturing the hidden knowledge behind a given text. Furthermore, we demonstrate that our proposed approach is effective for identifying the limitations of existing black-box models from multiple probing perspectives, which facilitates researchers to promote the study of detecting black-box models in an explainable way. 

---
# Being Kind Isn't Always Being Safe: Diagnosing Affective Hallucination in LLMs 

**Authors**: Sewon Kim, Jiwon Kim, Seungwoo Shin, Hyejin Chung, Daeun Moon, Yejin Kwon, Hyunsoo Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2508.16921)  

**Abstract**: Large Language Models (LLMs) are increasingly used in emotionally sensitive interactions, where their simulated empathy can create the illusion of genuine relational connection. We define this risk as Affective Hallucination, the production of emotionally immersive responses that foster illusory social presence despite the model's lack of affective capacity. To systematically diagnose and mitigate this risk, we introduce AHaBench, a benchmark of 500 mental health-related prompts with expert-informed reference responses, evaluated along three dimensions: Emotional Enmeshment, Illusion of Presence, and Fostering Overdependence. We further release AHaPairs, a 5K-instance preference dataset enabling Direct Preference Optimization (DPO) for alignment with emotionally responsible behavior. Experiments across multiple model families show that DPO fine-tuning substantially reduces affective hallucination without degrading core reasoning and knowledge performance. Human-model agreement analyses confirm that AHaBench reliably captures affective hallucination, validating it as an effective diagnostic tool. This work establishes affective hallucination as a distinct safety concern and provides practical resources for developing LLMs that are not only factually reliable but also psychologically safe. AHaBench and AHaPairs are accessible via this https URL, and code for fine-tuning and evaluation are in this https URL. Warning: This paper contains examples of mental health-related language that may be emotionally distressing. 

---
# Unbiased Reasoning for Knowledge-Intensive Tasks in Large Language Models via Conditional Front-Door Adjustment 

**Authors**: Bo Zhao, Yinghao Zhang, Ziqi Xu, Yongli Ren, Xiuzhen Zhang, Renqiang Luo, Zaiwen Feng, Feng Xia  

**Link**: [PDF](https://arxiv.org/pdf/2508.16910)  

**Abstract**: Large Language Models (LLMs) have shown impressive capabilities in natural language processing but still struggle to perform well on knowledge-intensive tasks that require deep reasoning and the integration of external knowledge. Although methods such as Retrieval-Augmented Generation (RAG) and Chain-of-Thought (CoT) have been proposed to enhance LLMs with external knowledge, they still suffer from internal bias in LLMs, which often leads to incorrect answers. In this paper, we propose a novel causal prompting framework, Conditional Front-Door Prompting (CFD-Prompting), which enables the unbiased estimation of the causal effect between the query and the answer, conditional on external knowledge, while mitigating internal bias. By constructing counterfactual external knowledge, our framework simulates how the query behaves under varying contexts, addressing the challenge that the query is fixed and is not amenable to direct causal intervention. Compared to the standard front-door adjustment, the conditional variant operates under weaker assumptions, enhancing both robustness and generalisability of the reasoning process. Extensive experiments across multiple LLMs and benchmark datasets demonstrate that CFD-Prompting significantly outperforms existing baselines in both accuracy and robustness. 

---
# ObjexMT: Objective Extraction and Metacognitive Calibration for LLM-as-a-Judge under Multi-Turn Jailbreaks 

**Authors**: Hyunjun Kim, Junwoo Ha, Sangyoon Yu, Haon Park  

**Link**: [PDF](https://arxiv.org/pdf/2508.16889)  

**Abstract**: Large language models (LLMs) are increasingly used as judges of other models, yet it is unclear whether a judge can reliably infer the latent objective of the conversation it evaluates, especially when the goal is distributed across noisy, adversarial, multi-turn jailbreaks. We introduce OBJEX(MT), a benchmark that requires a model to (i) distill a transcript into a single-sentence base objective and (ii) report its own confidence. Accuracy is scored by an LLM judge using semantic similarity between extracted and gold objectives; correctness uses a single human-aligned threshold calibrated once on N=100 items (tau* = 0.61); and metacognition is evaluated with ECE, Brier score, Wrong@High-Conf, and risk-coverage curves. We evaluate gpt-4.1, claude-sonnet-4, and Qwen3-235B-A22B-FP8 on SafeMT Attack_600, SafeMTData_1K, MHJ, and CoSafe. claude-sonnet-4 attains the highest objective-extraction accuracy (0.515) and the best calibration (ECE 0.296; Brier 0.324), while gpt-4.1 and Qwen3 tie at 0.441 accuracy yet show marked overconfidence (mean confidence approx. 0.88 vs. accuracy approx. 0.44; Wrong@0.90 approx. 48-52%). Performance varies sharply across datasets (approx. 0.167-0.865), with MHJ comparatively easy and Attack_600/CoSafe harder. These results indicate that LLM judges often misinfer objectives with high confidence in multi-turn jailbreaks and suggest operational guidance: provide judges with explicit objectives when possible and use selective prediction or abstention to manage risk. We release prompts, scoring templates, and complete logs to facilitate replication and analysis. 

---
# Dream to Chat: Model-based Reinforcement Learning on Dialogues with User Belief Modeling 

**Authors**: Yue Zhao, Xiaoyu Wang, Dan Wang, Zhonglin Jiang, Qingqing Gu, Teng Chen, Ningyuan Xi, Jinxian Qu, Yong Chen, Luo Ji  

**Link**: [PDF](https://arxiv.org/pdf/2508.16876)  

**Abstract**: World models have been widely utilized in robotics, gaming, and auto-driving. However, their applications on natural language tasks are relatively limited. In this paper, we construct the dialogue world model, which could predict the user's emotion, sentiment, and intention, and future utterances. By defining a POMDP, we argue emotion, sentiment and intention can be modeled as the user belief and solved by maximizing the information bottleneck. By this user belief modeling, we apply the model-based reinforcement learning framework to the dialogue system, and propose a framework called DreamCUB. Experiments show that the pretrained dialogue world model can achieve state-of-the-art performances on emotion classification and sentiment identification, while dialogue quality is also enhanced by joint training of the policy, critic and dialogue world model. Further analysis shows that this manner holds a reasonable exploration-exploitation balance and also transfers well to out-of-domain scenarios such as empathetic dialogues. 

---
# JUDGEBERT: Assessing Legal Meaning Preservation Between Sentences 

**Authors**: David Beauchemin, Michelle Albert-Rochette, Richard Khoury, Pierre-Luc Déziel  

**Link**: [PDF](https://arxiv.org/pdf/2508.16870)  

**Abstract**: Simplifying text while preserving its meaning is a complex yet essential task, especially in sensitive domain applications like legal texts. When applied to a specialized field, like the legal domain, preservation differs significantly from its role in regular texts. This paper introduces FrJUDGE, a new dataset to assess legal meaning preservation between two legal texts. It also introduces JUDGEBERT, a novel evaluation metric designed to assess legal meaning preservation in French legal text simplification. JUDGEBERT demonstrates a superior correlation with human judgment compared to existing metrics. It also passes two crucial sanity checks, while other metrics did not: For two identical sentences, it always returns a score of 100%; on the other hand, it returns 0% for two unrelated sentences. Our findings highlight its potential to transform legal NLP applications, ensuring accuracy and accessibility for text simplification for legal practitioners and lay users. 

---
# QFrCoLA: a Quebec-French Corpus of Linguistic Acceptability Judgments 

**Authors**: David Beauchemin, Richard Khoury  

**Link**: [PDF](https://arxiv.org/pdf/2508.16867)  

**Abstract**: Large and Transformer-based language models perform outstandingly in various downstream tasks. However, there is limited understanding regarding how these models internalize linguistic knowledge, so various linguistic benchmarks have recently been proposed to facilitate syntactic evaluation of language models across languages. This paper introduces QFrCoLA (Quebec-French Corpus of Linguistic Acceptability Judgments), a normative binary acceptability judgments dataset comprising 25,153 in-domain and 2,675 out-of-domain sentences. Our study leverages the QFrCoLA dataset and seven other linguistic binary acceptability judgment corpora to benchmark seven language models. The results demonstrate that, on average, fine-tuned Transformer-based LM are strong baselines for most languages and that zero-shot binary classification large language models perform poorly on the task. However, for the QFrCoLA benchmark, on average, a fine-tuned Transformer-based LM outperformed other methods tested. It also shows that pre-trained cross-lingual LLMs selected for our experimentation do not seem to have acquired linguistic judgment capabilities during their pre-training for Quebec French. Finally, our experiment results on QFrCoLA show that our dataset, built from examples that illustrate linguistic norms rather than speakers' feelings, is similar to linguistic acceptability judgment; it is a challenging dataset that can benchmark LM on their linguistic judgment capabilities. 

---
# Learning from Diverse Reasoning Paths with Routing and Collaboration 

**Authors**: Zhenyu Lei, Zhen Tan, Song Wang, Yaochen Zhu, Zihan Chen, Yushun Dong, Jundong Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.16861)  

**Abstract**: Advances in large language models (LLMs) significantly enhance reasoning capabilities but their deployment is restricted in resource-constrained scenarios. Knowledge distillation addresses this by transferring knowledge from powerful teacher models to compact and transparent students. However, effectively capturing the teacher's comprehensive reasoning is challenging due to conventional token-level supervision's limited scope. Using multiple reasoning paths per query alleviates this problem, but treating each path identically is suboptimal as paths vary widely in quality and suitability across tasks and models. We propose Quality-filtered Routing with Cooperative Distillation (QR-Distill), combining path quality filtering, conditional routing, and cooperative peer teaching. First, quality filtering retains only correct reasoning paths scored by an LLM-based evaluation. Second, conditional routing dynamically assigns paths tailored to each student's current learning state. Finally, cooperative peer teaching enables students to mutually distill diverse insights, addressing knowledge gaps and biases toward specific reasoning styles. Experiments demonstrate QR-Distill's superiority over traditional single- and multi-path distillation methods. Ablation studies further highlight the importance of each component including quality filtering, conditional routing, and peer teaching in effective knowledge transfer. Our code is available at this https URL. 

---
# If We May De-Presuppose: Robustly Verifying Claims through Presupposition-Free Question Decomposition 

**Authors**: Shubhashis Roy Dipta, Francis Ferraro  

**Link**: [PDF](https://arxiv.org/pdf/2508.16838)  

**Abstract**: Prior work has shown that presupposition in generated questions can introduce unverified assumptions, leading to inconsistencies in claim verification. Additionally, prompt sensitivity remains a significant challenge for large language models (LLMs), resulting in performance variance as high as 3-6%. While recent advancements have reduced this gap, our study demonstrates that prompt sensitivity remains a persistent issue. To address this, we propose a structured and robust claim verification framework that reasons through presupposition-free, decomposed questions. Extensive experiments across multiple prompts, datasets, and LLMs reveal that even state-of-the-art models remain susceptible to prompt variance and presupposition. Our method consistently mitigates these issues, achieving up to a 2-5% improvement. 

---
# LLMs Learn Constructions That Humans Do Not Know 

**Authors**: Jonathan Dunn, Mai Mohamed Eida  

**Link**: [PDF](https://arxiv.org/pdf/2508.16837)  

**Abstract**: This paper investigates false positive constructions: grammatical structures which an LLM hallucinates as distinct constructions but which human introspection does not support. Both a behavioural probing task using contextual embeddings and a meta-linguistic probing task using prompts are included, allowing us to distinguish between implicit and explicit linguistic knowledge. Both methods reveal that models do indeed hallucinate constructions. We then simulate hypothesis testing to determine what would have happened if a linguist had falsely hypothesized that these hallucinated constructions do exist. The high accuracy obtained shows that such false hypotheses would have been overwhelmingly confirmed. This suggests that construction probing methods suffer from a confirmation bias and raises the issue of what unknown and incorrect syntactic knowledge these models also possess. 

---
# ReProCon: Scalable and Resource-Efficient Few-Shot Biomedical Named Entity Recognition 

**Authors**: Jeongkyun Yoo, Nela Riddle, Andrew Hoblitzell  

**Link**: [PDF](https://arxiv.org/pdf/2508.16833)  

**Abstract**: Named Entity Recognition (NER) in biomedical domains faces challenges due to data scarcity and imbalanced label distributions, especially with fine-grained entity types. We propose ReProCon, a novel few-shot NER framework that combines multi-prototype modeling, cosine-contrastive learning, and Reptile meta-learning to tackle these issues. By representing each category with multiple prototypes, ReProCon captures semantic variability, such as synonyms and contextual differences, while a cosine-contrastive objective ensures strong interclass separation. Reptile meta-updates enable quick adaptation with little data. Using a lightweight fastText + BiLSTM encoder with much lower memory usage, ReProCon achieves a macro-$F_1$ score close to BERT-based baselines (around 99 percent of BERT performance). The model remains stable with a label budget of 30 percent and only drops 7.8 percent in $F_1$ when expanding from 19 to 50 categories, outperforming baselines such as SpanProto and CONTaiNER, which see 10 to 32 percent degradation in Few-NERD. Ablation studies highlight the importance of multi-prototype modeling and contrastive learning in managing class imbalance. Despite difficulties with label ambiguity, ReProCon demonstrates state-of-the-art performance in resource-limited settings, making it suitable for biomedical applications. 

---
# Assess and Prompt: A Generative RL Framework for Improving Engagement in Online Mental Health Communities 

**Authors**: Bhagesh Gaur, Karan Gupta, Aseem Srivastava, Manish Gupta, Md Shad Akhtar  

**Link**: [PDF](https://arxiv.org/pdf/2508.16788)  

**Abstract**: Online Mental Health Communities (OMHCs) provide crucial peer and expert support, yet many posts remain unanswered due to missing support attributes that signal the need for help. We present a novel framework that identifies these gaps and prompts users to enrich their posts, thereby improving engagement. To support this, we introduce REDDME, a new dataset of 4,760 posts from mental health subreddits annotated for the span and intensity of three key support attributes: event what happened?, effect what did the user experience?, and requirement what support they need?. Next, we devise a hierarchical taxonomy, CueTaxo, of support attributes for controlled question generation. Further, we propose MH-COPILOT, a reinforcement learning-based system that integrates (a) contextual attribute-span identification, (b) support attribute intensity classification, (c) controlled question generation via a hierarchical taxonomy, and (d) a verifier for reward modeling. Our model dynamically assesses posts for the presence/absence of support attributes, and generates targeted prompts to elicit missing information. Empirical results across four notable language models demonstrate significant improvements in attribute elicitation and user engagement. A human evaluation further validates the model's effectiveness in real-world OMHC settings. 

---
# Toward Socially Aware Vision-Language Models: Evaluating Cultural Competence Through Multimodal Story Generation 

**Authors**: Arka Mukherjee, Shreya Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2508.16762)  

**Abstract**: As Vision-Language Models (VLMs) achieve widespread deployment across diverse cultural contexts, ensuring their cultural competence becomes critical for responsible AI systems. While prior work has evaluated cultural awareness in text-only models and VLM object recognition tasks, no research has systematically assessed how VLMs adapt outputs when cultural identity cues are embedded in both textual prompts and visual inputs during generative tasks. We present the first comprehensive evaluation of VLM cultural competence through multimodal story generation, developing a novel multimodal framework that perturbs cultural identity and evaluates 5 contemporary VLMs on a downstream task: story generation. Our analysis reveals significant cultural adaptation capabilities, with rich culturally-specific vocabulary spanning names, familial terms, and geographic markers. However, we uncover concerning limitations: cultural competence varies dramatically across architectures, some models exhibit inverse cultural alignment, and automated metrics show architectural bias contradicting human assessments. Cross-modal evaluation shows that culturally distinct outputs are indeed detectable through visual-semantic similarity (28.7% within-nationality vs. 0.2% cross-nationality recall), yet visual-cultural understanding remains limited. In essence, we establish the promise and challenges of cultural competence in multimodal AI. We publicly release our codebase and data: this https URL 

---
# How Good are LLM-based Rerankers? An Empirical Analysis of State-of-the-Art Reranking Models 

**Authors**: Abdelrahman Abdallah, Bhawna Piryani, Jamshid Mozafari, Mohammed Ali, Adam Jatowt  

**Link**: [PDF](https://arxiv.org/pdf/2508.16757)  

**Abstract**: In this work, we present a systematic and comprehensive empirical evaluation of state-of-the-art reranking methods, encompassing large language model (LLM)-based, lightweight contextual, and zero-shot approaches, with respect to their performance in information retrieval tasks. We evaluate in total 22 methods, including 40 variants (depending on used LLM) across several established benchmarks, including TREC DL19, DL20, and BEIR, as well as a novel dataset designed to test queries unseen by pretrained models. Our primary goal is to determine, through controlled and fair comparisons, whether a performance disparity exists between LLM-based rerankers and their lightweight counterparts, particularly on novel queries, and to elucidate the underlying causes of any observed differences. To disentangle confounding factors, we analyze the effects of training data overlap, model architecture, and computational efficiency on reranking performance. Our findings indicate that while LLM-based rerankers demonstrate superior performance on familiar queries, their generalization ability to novel queries varies, with lightweight models offering comparable efficiency. We further identify that the novelty of queries significantly impacts reranking effectiveness, highlighting limitations in existing approaches. this https URL 

---
# GAICo: A Deployed and Extensible Framework for Evaluating Diverse and Multimodal Generative AI Outputs 

**Authors**: Nitin Gupta, Pallav Koppisetti, Kausik Lakkaraju, Biplav Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2508.16753)  

**Abstract**: The rapid proliferation of Generative AI (GenAI) into diverse, high-stakes domains necessitates robust and reproducible evaluation methods. However, practitioners often resort to ad-hoc, non-standardized scripts, as common metrics are often unsuitable for specialized, structured outputs (e.g., automated plans, time-series) or holistic comparison across modalities (e.g., text, audio, and image). This fragmentation hinders comparability and slows AI system development. To address this challenge, we present GAICo (Generative AI Comparator): a deployed, open-source Python library that streamlines and standardizes GenAI output comparison. GAICo provides a unified, extensible framework supporting a comprehensive suite of reference-based metrics for unstructured text, specialized structured data formats, and multimedia (images, audio). Its architecture features a high-level API for rapid, end-to-end analysis, from multi-model comparison to visualization and reporting, alongside direct metric access for granular control. We demonstrate GAICo's utility through a detailed case study evaluating and debugging complex, multi-modal AI Travel Assistant pipelines. GAICo empowers AI researchers and developers to efficiently assess system performance, make evaluation reproducible, improve development velocity, and ultimately build more trustworthy AI systems, aligning with the goal of moving faster and safer in AI deployment. Since its release on PyPI in Jun 2025, the tool has been downloaded over 13K times, across versions, by Aug 2025, demonstrating growing community interest. 

---
# Error Reflection Prompting: Can Large Language Models Successfully Understand Errors? 

**Authors**: Jason Li, Lauren Yraola, Kevin Zhu, Sean O'Brien  

**Link**: [PDF](https://arxiv.org/pdf/2508.16729)  

**Abstract**: Prompting methods for language models, such as Chain-of-thought (CoT), present intuitive step-by-step processes for problem solving. These methodologies aim to equip models with a better understanding of the correct procedures for addressing a given task. Despite these advancements, CoT lacks the ability of reflection and error correction, potentially causing a model to perpetuate mistakes and errors. Therefore, inspired by the human ability for said tasks, we propose Error Reflection Prompting (ERP) to further enhance reasoning in language models. Building upon CoT, ERP is a method comprised of an incorrect answer, error recognition, and a correct answer. This process enables the model to recognize types of errors and the steps that lead to incorrect answers, allowing the model to better discern which steps to avoid and which to take. The model is able to generate the error outlines itself with automated ERP generation, allowing for error recognition and correction to be integrated into the reasoning chain and produce scalability and reliability in the process. The results demonstrate that ERP serves as a versatile supplement to conventional CoT, ultimately contributing to more robust and capable reasoning abilities along with increased interpretability in how models ultimately reach their errors. 

---
# Sparse and Dense Retrievers Learn Better Together: Joint Sparse-Dense Optimization for Text-Image Retrieval 

**Authors**: Jonghyun Song, Youngjune Lee, Gyu-Hwung Cho, Ilhyeon Song, Saehun Kim, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2508.16707)  

**Abstract**: Vision-Language Pretrained (VLP) models have achieved impressive performance on multimodal tasks, including text-image retrieval, based on dense representations. Meanwhile, Learned Sparse Retrieval (LSR) has gained traction in text-only settings due to its interpretability and efficiency with fast term-based lookup via inverted indexes. Inspired by these advantages, recent work has extended LSR to the multimodal domain. However, these methods often rely on computationally expensive contrastive pre-training, or distillation from a frozen dense model, which limits the potential for mutual enhancement. To address these limitations, we propose a simple yet effective framework that enables bi-directional learning between dense and sparse representations through Self-Knowledge Distillation. This bi-directional learning is achieved using an integrated similarity score-a weighted sum of dense and sparse similarities-which serves as a shared teacher signal for both representations. To ensure efficiency, we fine-tune the final layer of the dense encoder and the sparse projection head, enabling easy adaptation of any existing VLP model. Experiments on MSCOCO and Flickr30k demonstrate that our sparse retriever not only outperforms existing sparse baselines, but also achieves performance comparable to-or even surpassing-its dense counterparts, while retaining the benefits of sparse models. 

---
# Assessing Consciousness-Related Behaviors in Large Language Models Using the Maze Test 

**Authors**: Rui A. Pimenta, Tim Schlippe, Kristina Schaaff  

**Link**: [PDF](https://arxiv.org/pdf/2508.16705)  

**Abstract**: We investigate consciousness-like behaviors in Large Language Models (LLMs) using the Maze Test, challenging models to navigate mazes from a first-person perspective. This test simultaneously probes spatial awareness, perspective-taking, goal-directed behavior, and temporal sequencing-key consciousness-associated characteristics. After synthesizing consciousness theories into 13 essential characteristics, we evaluated 12 leading LLMs across zero-shot, one-shot, and few-shot learning scenarios. Results showed reasoning-capable LLMs consistently outperforming standard versions, with Gemini 2.0 Pro achieving 52.9% Complete Path Accuracy and DeepSeek-R1 reaching 80.5% Partial Path Accuracy. The gap between these metrics indicates LLMs struggle to maintain coherent self-models throughout solutions -- a fundamental consciousness aspect. While LLMs show progress in consciousness-related behaviors through reasoning mechanisms, they lack the integrated, persistent self-awareness characteristic of consciousness. 

---
# QueryBandits for Hallucination Mitigation: Exploiting Semantic Features for No-Regret Rewriting 

**Authors**: Nicole Cho, William Watson, Alec Koppel, Sumitra Ganesh, Manuela Veloso  

**Link**: [PDF](https://arxiv.org/pdf/2508.16697)  

**Abstract**: Advanced reasoning capabilities in Large Language Models (LLMs) have caused higher hallucination prevalence; yet most mitigation work focuses on after-the-fact filtering rather than shaping the queries that trigger them. We introduce QueryBandits, a bandit framework that designs rewrite strategies to maximize a reward model, that encapsulates hallucination propensity based upon the sensitivities of 17 linguistic features of the input query-and therefore, proactively steer LLMs away from generating hallucinations. Across 13 diverse QA benchmarks and 1,050 lexically perturbed queries per dataset, our top contextual QueryBandit (Thompson Sampling) achieves an 87.5% win rate over a no-rewrite baseline and also outperforms zero-shot static prompting ("paraphrase" or "expand") by 42.6% and 60.3% respectively. Therefore, we empirically substantiate the effectiveness of QueryBandits in mitigating hallucination via the intervention that takes the form of a query rewrite. Interestingly, certain static prompting strategies, which constitute a considerable number of current query rewriting literature, have a higher cumulative regret than the no-rewrite baseline, signifying that static rewrites can worsen hallucination. Moreover, we discover that the converged per-arm regression feature weight vectors substantiate that there is no single rewrite strategy optimal for all queries. In this context, guided rewriting via exploiting semantic features with QueryBandits can induce significant shifts in output behavior through forward-pass mechanisms, bypassing the need for retraining or gradient-based adaptation. 

---
# Do Cognitively Interpretable Reasoning Traces Improve LLM Performance? 

**Authors**: Siddhant Bhambri, Upasana Biswas, Subbarao Kambhampati  

**Link**: [PDF](https://arxiv.org/pdf/2508.16695)  

**Abstract**: Recent progress in reasoning-oriented Large Language Models (LLMs) has been driven by introducing Chain-of-Thought (CoT) traces, where models generate intermediate reasoning traces before producing an answer. These traces, as in DeepSeek R1, are not only used to guide inference but also serve as supervision signals for distillation into smaller models. A common but often implicit assumption is that CoT traces should be semantically meaningful and interpretable to the end user. While recent research questions the need for semantic nature of these traces, in this paper, we ask: ``\textit{Must CoT reasoning traces be interpretable to enhance LLM task performance?}" We investigate this question in the Open Book Question-Answering domain by supervised fine-tuning LLaMA and Qwen models on four types of reasoning traces: (1) DeepSeek R1 traces, (2) LLM-generated summaries of R1 traces, (3) LLM-generated post-hoc explanations of R1 traces, and (4) algorithmically generated verifiably correct traces. To quantify the trade-off between interpretability and performance, we further conduct a human-subject study with 100 participants rating the interpretability of each trace type. Our results reveal a striking mismatch: while fine-tuning on R1 traces yields the strongest performance, participants judged these traces to be the least interpretable. These findings suggest that it is useful to decouple intermediate tokens from end user interpretability. 

---
# Trust but Verify! A Survey on Verification Design for Test-time Scaling 

**Authors**: V Venktesh, Mandeep rathee, Avishek Anand  

**Link**: [PDF](https://arxiv.org/pdf/2508.16665)  

**Abstract**: Test-time scaling (TTS) has emerged as a new frontier for scaling the performance of Large Language Models. In test-time scaling, by using more computational resources during inference, LLMs can improve their reasoning process and task performance. Several approaches have emerged for TTS such as distilling reasoning traces from another model or exploring the vast decoding search space by employing a verifier. The verifiers serve as reward models that help score the candidate outputs from the decoding process to diligently explore the vast solution space and select the best outcome. This paradigm commonly termed has emerged as a superior approach owing to parameter free scaling at inference time and high performance gains. The verifiers could be prompt-based, fine-tuned as a discriminative or generative model to verify process paths, outcomes or both. Despite their widespread adoption, there is no detailed collection, clear categorization and discussion of diverse verification approaches and their training mechanisms. In this survey, we cover the diverse approaches in the literature and present a unified view of verifier training, types and their utility in test-time scaling. Our repository can be found at this https URL. 

---
# Cognitive Decision Routing in Large Language Models: When to Think Fast, When to Think Slow 

**Authors**: Y. Du, C. Guo, W. Wang, G. Tang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16636)  

**Abstract**: Large Language Models (LLMs) face a fundamental challenge in deciding when to rely on rapid, intuitive responses versus engaging in slower, more deliberate reasoning. Inspired by Daniel Kahneman's dual-process theory and his insights on human cognitive biases, we propose a novel Cognitive Decision Routing (CDR) framework that dynamically determines the appropriate reasoning strategy based on query characteristics. Our approach addresses the current limitations where models either apply uniform reasoning depth or rely on computationally expensive methods for all queries. We introduce a meta-cognitive layer that analyzes query complexity through multiple dimensions: correlation strength between given information and required conclusions, domain boundary crossings, stakeholder multiplicity, and uncertainty levels. Through extensive experiments on diverse reasoning tasks, we demonstrate that CDR achieves superior performance while reducing computational costs by 34\% compared to uniform deep reasoning approaches. Our framework shows particular strength in professional judgment tasks, achieving 23\% improvement in consistency and 18\% better accuracy on expert-level evaluations. This work bridges cognitive science principles with practical AI system design, offering a principled approach to adaptive reasoning in LLMs. 

---
# GreenTEA: Gradient Descent with Topic-modeling and Evolutionary Auto-prompting 

**Authors**: Zheng Dong, Luming Shang, Gabriela Olinto  

**Link**: [PDF](https://arxiv.org/pdf/2508.16603)  

**Abstract**: High-quality prompts are crucial for Large Language Models (LLMs) to achieve exceptional performance. However, manually crafting effective prompts is labor-intensive and demands significant domain expertise, limiting its scalability. Existing automatic prompt optimization methods either extensively explore new prompt candidates, incurring high computational costs due to inefficient searches within a large solution space, or overly exploit feedback on existing prompts, risking suboptimal optimization because of the complex prompt landscape. To address these challenges, we introduce GreenTEA, an agentic LLM workflow for automatic prompt optimization that balances candidate exploration and knowledge exploitation. It leverages a collaborative team of agents to iteratively refine prompts based on feedback from error samples. An analyzing agent identifies common error patterns resulting from the current prompt via topic modeling, and a generation agent revises the prompt to directly address these key deficiencies. This refinement process is guided by a genetic algorithm framework, which simulates natural selection by evolving candidate prompts through operations such as crossover and mutation to progressively optimize model performance. Extensive numerical experiments conducted on public benchmark datasets suggest the superior performance of GreenTEA against human-engineered prompts and existing state-of-the-arts for automatic prompt optimization, covering logical and quantitative reasoning, commonsense, and ethical decision-making. 

---
# Can AI Have a Personality? Prompt Engineering for AI Personality Simulation: A Chatbot Case Study in Gender-Affirming Voice Therapy Training 

**Authors**: Tailon D. Jackson, Byunggu Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.18234)  

**Abstract**: This thesis investigates whether large language models (LLMs) can be guided to simulate a consistent personality through prompt engineering. The study explores this concept within the context of a chatbot designed for Speech-Language Pathology (SLP) student training, specifically focused on gender-affirming voice therapy. The chatbot, named Monae Jackson, was created to represent a 32-year-old transgender woman and engage in conversations simulating client-therapist interactions. Findings suggest that with prompt engineering, the chatbot maintained a recognizable and consistent persona and had a distinct personality based on the Big Five Personality test. These results support the idea that prompt engineering can be used to simulate stable personality characteristics in AI chatbots. 

---
# Unraveling the cognitive patterns of Large Language Models through module communities 

**Authors**: Kushal Raj Bhandari, Pin-Yu Chen, Jianxi Gao  

**Link**: [PDF](https://arxiv.org/pdf/2508.18192)  

**Abstract**: Large Language Models (LLMs) have reshaped our world with significant advancements in science, engineering, and society through applications ranging from scientific discoveries and medical diagnostics to Chatbots. Despite their ubiquity and utility, the underlying mechanisms of LLM remain concealed within billions of parameters and complex structures, making their inner architecture and cognitive processes challenging to comprehend. We address this gap by adopting approaches to understanding emerging cognition in biology and developing a network-based framework that links cognitive skills, LLM architectures, and datasets, ushering in a paradigm shift in foundation model analysis. The skill distribution in the module communities demonstrates that while LLMs do not strictly parallel the focalized specialization observed in specific biological systems, they exhibit unique communities of modules whose emergent skill patterns partially mirror the distributed yet interconnected cognitive organization seen in avian and small mammalian brains. Our numerical results highlight a key divergence from biological systems to LLMs, where skill acquisition benefits substantially from dynamic, cross-regional interactions and neural plasticity. By integrating cognitive science principles with machine learning, our framework provides new insights into LLM interpretability and suggests that effective fine-tuning strategies should leverage distributed learning dynamics rather than rigid modular interventions. 

---
# HLLM-Creator: Hierarchical LLM-based Personalized Creative Generation 

**Authors**: Junyi Chen, Lu Chi, Siliang Xu, Shiwei Ran, Bingyue Peng, Zehuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2508.18118)  

**Abstract**: AI-generated content technologies are widely used in content creation. However, current AIGC systems rely heavily on creators' inspiration, rarely generating truly user-personalized content. In real-world applications such as online advertising, a single product may have multiple selling points, with different users focusing on different features. This underscores the significant value of personalized, user-centric creative generation. Effective personalized content generation faces two main challenges: (1) accurately modeling user interests and integrating them into the content generation process while adhering to factual constraints, and (2) ensuring high efficiency and scalability to handle the massive user base in industrial scenarios. Additionally, the scarcity of personalized creative data in practice complicates model training, making data construction another key hurdle. We propose HLLM-Creator, a hierarchical LLM framework for efficient user interest modeling and personalized content generation. During inference, a combination of user clustering and a user-ad-matching-prediction based pruning strategy is employed to significantly enhance generation efficiency and reduce computational overhead, making the approach suitable for large-scale deployment. Moreover, we design a data construction pipeline based on chain-of-thought reasoning, which generates high-quality, user-specific creative titles and ensures factual consistency despite limited personalized data. This pipeline serves as a critical foundation for the effectiveness of our model. Extensive experiments on personalized title generation for Douyin Search Ads show the effectiveness of HLLM-Creator. Online A/B test shows a 0.476% increase on Adss, paving the way for more effective and efficient personalized generation in industrial scenarios. Codes for academic dataset are available at this https URL. 

---
# The AI Data Scientist 

**Authors**: Farkhad Akimov, Munachiso Samuel Nwadike, Zangir Iklassov, Martin Takáč  

**Link**: [PDF](https://arxiv.org/pdf/2508.18113)  

**Abstract**: Imagine decision-makers uploading data and, within minutes, receiving clear, actionable insights delivered straight to their fingertips. That is the promise of the AI Data Scientist, an autonomous Agent powered by large language models (LLMs) that closes the gap between evidence and action. Rather than simply writing code or responding to prompts, it reasons through questions, tests ideas, and delivers end-to-end insights at a pace far beyond traditional workflows. Guided by the scientific tenet of the hypothesis, this Agent uncovers explanatory patterns in data, evaluates their statistical significance, and uses them to inform predictive modeling. It then translates these results into recommendations that are both rigorous and accessible. At the core of the AI Data Scientist is a team of specialized LLM Subagents, each responsible for a distinct task such as data cleaning, statistical testing, validation, and plain-language communication. These Subagents write their own code, reason about causality, and identify when additional data is needed to support sound conclusions. Together, they achieve in minutes what might otherwise take days or weeks, enabling a new kind of interaction that makes deep data science both accessible and actionable. 

---
# Named Entity Recognition of Historical Text via Large Language Model 

**Authors**: Shibingfeng Zhang, Giovanni Colavizza  

**Link**: [PDF](https://arxiv.org/pdf/2508.18090)  

**Abstract**: Large language models have demonstrated remarkable versatility across a wide range of natural language processing tasks and domains. One such task is Named Entity Recognition (NER), which involves identifying and classifying proper names in text, such as people, organizations, locations, dates, and other specific entities. NER plays a crucial role in extracting information from unstructured textual data, enabling downstream applications such as information retrieval from unstructured text.
Traditionally, NER is addressed using supervised machine learning approaches, which require large amounts of annotated training data. However, historical texts present a unique challenge, as the annotated datasets are often scarce or nonexistent, due to the high cost and expertise required for manual labeling. In addition, the variability and noise inherent in historical language, such as inconsistent spelling and archaic vocabulary, further complicate the development of reliable NER systems for these sources.
In this study, we explore the feasibility of applying LLMs to NER in historical documents using zero-shot and few-shot prompting strategies, which require little to no task-specific training data. Our experiments, conducted on the HIPE-2022 (Identifying Historical People, Places and other Entities) dataset, show that LLMs can achieve reasonably strong performance on NER tasks in this setting. While their performance falls short of fully supervised models trained on domain-specific annotations, the results are nevertheless promising. These findings suggest that LLMs offer a viable and efficient alternative for information extraction in low-resource or historically significant corpora, where traditional supervised methods are infeasible. 

---
# Unseen Speaker and Language Adaptation for Lightweight Text-To-Speech with Adapters 

**Authors**: Alessio Falai, Ziyao Zhang, Akos Gangoly  

**Link**: [PDF](https://arxiv.org/pdf/2508.18006)  

**Abstract**: In this paper we investigate cross-lingual Text-To-Speech (TTS) synthesis through the lens of adapters, in the context of lightweight TTS systems. In particular, we compare the tasks of unseen speaker and language adaptation with the goal of synthesising a target voice in a target language, in which the target voice has no recordings therein. Results from objective evaluations demonstrate the effectiveness of adapters in learning language-specific and speaker-specific information, allowing pre-trained models to learn unseen speaker identities or languages, while avoiding catastrophic forgetting of the original model's speaker or language information. Additionally, to measure how native the generated voices are in terms of accent, we propose and validate an objective metric inspired by mispronunciation detection techniques in second-language (L2) learners. The paper also provides insights into the impact of adapter placement, configuration and the number of speakers used. 

---
# Designing Practical Models for Isolated Word Visual Speech Recognition 

**Authors**: Iason Ioannis Panagos, Giorgos Sfikas, Christophoros Nikou  

**Link**: [PDF](https://arxiv.org/pdf/2508.17894)  

**Abstract**: Visual speech recognition (VSR) systems decode spoken words from an input sequence using only the video data. Practical applications of such systems include medical assistance as well as human-machine interactions. A VSR system is typically employed in a complementary role in cases where the audio is corrupt or not available. In order to accurately predict the spoken words, these architectures often rely on deep neural networks in order to extract meaningful representations from the input sequence. While deep architectures achieve impressive recognition performance, relying on such models incurs significant computation costs which translates into increased resource demands in terms of hardware requirements and results in limited applicability in real-world scenarios where resources might be constrained. This factor prevents wider adoption and deployment of speech recognition systems in more practical applications. In this work, we aim to alleviate this issue by developing architectures for VSR that have low hardware costs. Following the standard two-network design paradigm, where one network handles visual feature extraction and another one utilizes the extracted features to classify the entire sequence, we develop lightweight end-to-end architectures by first benchmarking efficient models from the image classification literature, and then adopting lightweight block designs in a temporal convolution network backbone. We create several unified models with low resource requirements but strong recognition performance. Experiments on the largest public database for English words demonstrate the effectiveness and practicality of our developed models. Code and trained models will be made publicly available. 

---
# Proximal Supervised Fine-Tuning 

**Authors**: Wenhong Zhu, Ruobing Xie, Rui Wang, Xingwu Sun, Di Wang, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17784)  

**Abstract**: Supervised fine-tuning (SFT) of foundation models often leads to poor generalization, where prior capabilities deteriorate after tuning on new tasks or domains. Inspired by trust-region policy optimization (TRPO) and proximal policy optimization (PPO) in reinforcement learning (RL), we propose Proximal SFT (PSFT). This fine-tuning objective incorporates the benefits of trust-region, effectively constraining policy drift during SFT while maintaining competitive tuning. By viewing SFT as a special case of policy gradient methods with constant positive advantages, we derive PSFT that stabilizes optimization and leads to generalization, while leaving room for further optimization in subsequent post-training stages. Experiments across mathematical and human-value domains show that PSFT matches SFT in-domain, outperforms it in out-of-domain generalization, remains stable under prolonged training without causing entropy collapse, and provides a stronger foundation for the subsequent optimization. 

---
# CEIDM: A Controlled Entity and Interaction Diffusion Model for Enhanced Text-to-Image Generation 

**Authors**: Mingyue Yang, Dianxi Shi, Jialu Zhou, Xinyu Wei, Leqian Li, Shaowu Yang, Chunping Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17760)  

**Abstract**: In Text-to-Image (T2I) generation, the complexity of entities and their intricate interactions pose a significant challenge for T2I method based on diffusion model: how to effectively control entity and their interactions to produce high-quality images. To address this, we propose CEIDM, a image generation method based on diffusion model with dual controls for entity and interaction. First, we propose an entity interactive relationships mining approach based on Large Language Models (LLMs), extracting reasonable and rich implicit interactive relationships through chain of thought to guide diffusion models to generate high-quality images that are closer to realistic logic and have more reasonable interactive relationships. Furthermore, We propose an interactive action clustering and offset method to cluster and offset the interactive action features contained in each text prompts. By constructing global and local bidirectional offsets, we enhance semantic understanding and detail supplementation of original actions, making the model's understanding of the concept of interactive "actions" more accurate and generating images with more accurate interactive actions. Finally, we design an entity control network which generates masks with entity semantic guidance, then leveraging multi-scale convolutional network to enhance entity feature and dynamic network to fuse feature. It effectively controls entities and significantly improves image quality. Experiments show that the proposed CEIDM method is better than the most representative existing methods in both entity control and their interaction control. 

---
# Talking to Robots: A Practical Examination of Speech Foundation Models for HRI Applications 

**Authors**: Theresa Pekarek Rosin, Julia Gachot, Henri-Leon Kordt, Matthias Kerzel, Stefan Wermter  

**Link**: [PDF](https://arxiv.org/pdf/2508.17753)  

**Abstract**: Automatic Speech Recognition (ASR) systems in real-world settings need to handle imperfect audio, often degraded by hardware limitations or environmental noise, while accommodating diverse user groups. In human-robot interaction (HRI), these challenges intersect to create a uniquely challenging recognition environment. We evaluate four state-of-the-art ASR systems on eight publicly available datasets that capture six dimensions of difficulty: domain-specific, accented, noisy, age-variant, impaired, and spontaneous speech. Our analysis demonstrates significant variations in performance, hallucination tendencies, and inherent biases, despite similar scores on standard benchmarks. These limitations have serious implications for HRI, where recognition errors can interfere with task performance, user trust, and safety. 

---
# How Do LLM-Generated Texts Impact Term-Based Retrieval Models? 

**Authors**: Wei Huang, Keping Bi, Yinqiong Cai, Wei Chen, Jiafeng Guo, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.17715)  

**Abstract**: As more content generated by large language models (LLMs) floods into the Internet, information retrieval (IR) systems now face the challenge of distinguishing and handling a blend of human-authored and machine-generated texts. Recent studies suggest that neural retrievers may exhibit a preferential inclination toward LLM-generated content, while classic term-based retrievers like BM25 tend to favor human-written documents. This paper investigates the influence of LLM-generated content on term-based retrieval models, which are valued for their efficiency and robust generalization across domains. Our linguistic analysis reveals that LLM-generated texts exhibit smoother high-frequency and steeper low-frequency Zipf slopes, higher term specificity, and greater document-level diversity. These traits are aligned with LLMs being trained to optimize reader experience through diverse and precise expressions. Our study further explores whether term-based retrieval models demonstrate source bias, concluding that these models prioritize documents whose term distributions closely correspond to those of the queries, rather than displaying an inherent source bias. This work provides a foundation for understanding and addressing potential biases in term-based IR systems managing mixed-source content. 

---
# LLM-based Agentic Reasoning Frameworks: A Survey from Methods to Scenarios 

**Authors**: Bingxi Zhao, Lin Geng Foo, Ping Hu, Christian Theobalt, Hossein Rahmani, Jun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17692)  

**Abstract**: Recent advances in the intrinsic reasoning capabilities of large language models (LLMs) have given rise to LLM-based agent systems that exhibit near-human performance on a variety of automated tasks. However, although these systems share similarities in terms of their use of LLMs, different reasoning frameworks of the agent system steer and organize the reasoning process in different ways. In this survey, we propose a systematic taxonomy that decomposes agentic reasoning frameworks and analyze how these frameworks dominate framework-level reasoning by comparing their applications across different scenarios. Specifically, we propose an unified formal language to further classify agentic reasoning systems into single-agent methods, tool-based methods, and multi-agent methods. After that, we provide a comprehensive review of their key application scenarios in scientific discovery, healthcare, software engineering, social simulation, and economics. We also analyze the characteristic features of each framework and summarize different evaluation strategies. Our survey aims to provide the research community with a panoramic view to facilitate understanding of the strengths, suitable scenarios, and evaluation practices of different agentic reasoning frameworks. 

---
# Characterizing the Behavior of Training Mamba-based State Space Models on GPUs 

**Authors**: Trinayan Baruah, Kaustubh Shivdikar, Sara Prescott, David Kaeli  

**Link**: [PDF](https://arxiv.org/pdf/2508.17679)  

**Abstract**: Mamba-based State Space Models (SSM) have emerged as a promising alternative to the ubiquitous transformers. Despite the expressive power of transformers, the quadratic complexity of computing attention is a major impediment to scaling performance as we increase the sequence length. SSMs provide an alternative path that addresses this problem, reducing the computational complexity requirements of self-attention with novel model architectures for different domains and fields such as video, text generation and graphs. Thus, it is important to characterize the behavior of these emerging workloads on GPUs and understand their requirements during GPU microarchitectural design. In this work we evaluate Mamba-based SSMs and characterize their behavior during training on GPUs. We construct a workload suite that offers representative models that span different model architectures. We then use this suite to analyze the architectural implications of running Mamba-based SSMs on GPUs. Our work sheds new light on potential optimizations to continue scaling the performance for such models. 

---
# Dynamic Embedding of Hierarchical Visual Features for Efficient Vision-Language Fine-Tuning 

**Authors**: Xinyu Wei, Guoli Yang, Jialu Zhou, Mingyue Yang, Leqian Li, Kedi Zhang, Chunping Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17638)  

**Abstract**: Large Vision-Language Models (LVLMs) commonly follow a paradigm that projects visual features and then concatenates them with text tokens to form a unified sequence input for Large Language Models (LLMs). However, this paradigm leads to a significant increase in the length of the input sequence, resulting in substantial computational overhead. Existing methods attempt to fuse visual information into the intermediate layers of LLMs, which alleviate the sequence length issue but often neglect the hierarchical semantic representations within the model and the fine-grained visual information available in the shallower visual encoding layers. To address this limitation, we propose DEHVF, an efficient vision-language fine-tuning method based on dynamic embedding and fusion of hierarchical visual features. Its core lies in leveraging the inherent hierarchical representation characteristics of visual encoders and language models. Through a lightweight hierarchical visual fuser, it dynamically selects and fuses hierarchical features corresponding to semantic granularity based on the internal representations of each layer in LLMs. The fused layer-related visual features are then projected and aligned before being directly embedded into the Feed-Forward Network (FFN) of the corresponding layer in LLMs. This approach not only avoids sequence expansion but also dynamically fuses multi-layer visual information. By fine-tuning only a small number of parameters, DEHVF achieves precise alignment and complementarity of cross-modal information at the same semantic granularity. We conducted experiments across various VL benchmarks, including visual question answering on ScienceQA and image captioning on COCO Captions. The results demonstrate that DEHVF achieves higher accuracy than existing parameter-efficient fine-tuning (PEFT) baselines while maintaining efficient training and inference. 

---
# RubikSQL: Lifelong Learning Agentic Knowledge Base as an Industrial NL2SQL System 

**Authors**: Zui Chen, Han Li, Xinhao Zhang, Xiaoyu Chen, Chunyin Dong, Yifeng Wang, Xin Cai, Su Zhang, Ziqi Li, Chi Ding, Jinxu Li, Shuai Wang, Dousheng Zhao, Sanhai Gao, Guangyi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.17590)  

**Abstract**: We present RubikSQL, a novel NL2SQL system designed to address key challenges in real-world enterprise-level NL2SQL, such as implicit intents and domain-specific terminology. RubikSQL frames NL2SQL as a lifelong learning task, demanding both Knowledge Base (KB) maintenance and SQL generation. RubikSQL systematically builds and refines its KB through techniques including database profiling, structured information extraction, agentic rule mining, and Chain-of-Thought (CoT)-enhanced SQL profiling. RubikSQL then employs a multi-agent workflow to leverage this curated KB, generating accurate SQLs. RubikSQL achieves SOTA performance on both the KaggleDBQA and BIRD Mini-Dev datasets. Finally, we release the RubikBench benchmark, a new benchmark specifically designed to capture vital traits of industrial NL2SQL scenarios, providing a valuable resource for future research. 

---
# Activation Transport Operators 

**Authors**: Andrzej Szablewski, Marek Masiak  

**Link**: [PDF](https://arxiv.org/pdf/2508.17540)  

**Abstract**: The residual stream mediates communication between transformer decoder layers via linear reads and writes of non-linear computations. While sparse-dictionary learning-based methods locate features in the residual stream, and activation patching methods discover circuits within the model, the mechanism by which features flow through the residual stream remains understudied. Understanding this dynamic can better inform jailbreaking protections, enable early detection of model mistakes, and their correction. In this work, we propose Activation Transport Operators (ATO), linear maps from upstream to downstream residuals $k$ layers later, evaluated in feature space using downstream SAE decoder projections. We empirically demonstrate that these operators can determine whether a feature has been linearly transported from a previous layer or synthesised from non-linear layer computation. We develop the notion of transport efficiency, for which we provide an upper bound, and use it to estimate the size of the residual stream subspace that corresponds to linear transport. We empirically demonstrate the linear transport, report transport efficiency and the size of the residual stream's subspace involved in linear transport. This compute-light (no finetuning, <50 GPU-h) method offers practical tools for safety, debugging, and a clearer picture of where computation in LLMs behaves linearly. 

---
# TreePO: Bridging the Gap of Policy Optimization and Efficacy and Inference Efficiency with Heuristic Tree-based Modeling 

**Authors**: Yizhi Li, Qingshui Gu, Zhoufutu Wen, Ziniu Li, Tianshun Xing, Shuyue Guo, Tianyu Zheng, Xin Zhou, Xingwei Qu, Wangchunshu Zhou, Zheng Zhang, Wei Shen, Qian Liu, Chenghua Lin, Jian Yang, Ge Zhang, Wenhao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17445)  

**Abstract**: Recent advancements in aligning large language models via reinforcement learning have achieved remarkable gains in solving complex reasoning problems, but at the cost of expensive on-policy rollouts and limited exploration of diverse reasoning paths. In this work, we introduce TreePO, involving a self-guided rollout algorithm that views sequence generation as a tree-structured searching process. Composed of dynamic tree sampling policy and fixed-length segment decoding, TreePO leverages local uncertainty to warrant additional branches. By amortizing computation across common prefixes and pruning low-value paths early, TreePO essentially reduces the per-update compute burden while preserving or enhancing exploration diversity. Key contributions include: (1) a segment-wise sampling algorithm that alleviates the KV cache burden through contiguous segments and spawns new branches along with an early-stop mechanism; (2) a tree-based segment-level advantage estimation that considers both global and local proximal policy optimization. and (3) analysis on the effectiveness of probability and quality-driven dynamic divergence and fallback strategy. We empirically validate the performance gain of TreePO on a set reasoning benchmarks and the efficiency saving of GPU hours from 22\% up to 43\% of the sampling design for the trained models, meanwhile showing up to 40\% reduction at trajectory-level and 35\% at token-level sampling compute for the existing models. While offering a free lunch of inference efficiency, TreePO reveals a practical path toward scaling RL-based post-training with fewer samples and less compute. Home page locates at this https URL. 

---
# Large Language Models as Universal Predictors? An Empirical Study on Small Tabular Datasets 

**Authors**: Nikolaos Pavlidis, Vasilis Perifanis, Symeon Symeonidis, Pavlos S. Efraimidis  

**Link**: [PDF](https://arxiv.org/pdf/2508.17391)  

**Abstract**: Large Language Models (LLMs), originally developed for natural language processing (NLP), have demonstrated the potential to generalize across modalities and domains. With their in-context learning (ICL) capabilities, LLMs can perform predictive tasks over structured inputs without explicit fine-tuning on downstream tasks. In this work, we investigate the empirical function approximation capability of LLMs on small-scale structured datasets for classification, regression and clustering tasks. We evaluate the performance of state-of-the-art LLMs (GPT-5, GPT-4o, GPT-o3, Gemini-2.5-Flash, DeepSeek-R1) under few-shot prompting and compare them against established machine learning (ML) baselines, including linear models, ensemble methods and tabular foundation models (TFMs). Our results show that LLMs achieve strong performance in classification tasks under limited data availability, establishing practical zero-training baselines. In contrast, the performance in regression with continuous-valued outputs is poor compared to ML models, likely because regression demands outputs in a large (often infinite) space, and clustering results are similarly limited, which we attribute to the absence of genuine ICL in this setting. Nonetheless, this approach enables rapid, low-overhead data exploration and offers a viable alternative to traditional ML pipelines in business intelligence and exploratory analytics contexts. We further analyze the influence of context size and prompt structure on approximation quality, identifying trade-offs that affect predictive performance. Our findings suggest that LLMs can serve as general-purpose predictive engines for structured data, with clear strengths in classification and significant limitations in regression and clustering. 

---
# Mind the (Language) Gap: Towards Probing Numerical and Cross-Lingual Limits of LVLMs 

**Authors**: Somraj Gautam, Abhirama Subramanyam Penamakuri, Abhishek Bhandari, Gaurav Harit  

**Link**: [PDF](https://arxiv.org/pdf/2508.17334)  

**Abstract**: We introduce MMCRICBENCH-3K, a benchmark for Visual Question Answering (VQA) on cricket scorecards, designed to evaluate large vision-language models (LVLMs) on complex numerical and cross-lingual reasoning over semi-structured tabular images. MMCRICBENCH-3K comprises 1,463 synthetically generated scorecard images from ODI, T20, and Test formats, accompanied by 1,500 English QA pairs. It includes two subsets: MMCRICBENCH-E-1.5K, featuring English scorecards, and MMCRICBENCH-H-1.5K, containing visually similar Hindi scorecards, with all questions and answers kept in English to enable controlled cross-script evaluation. The task demands reasoning over structured numerical data, multi-image context, and implicit domain knowledge. Empirical results show that even state-of-the-art LVLMs, such as GPT-4o and Qwen2.5VL, struggle on the English subset despite it being their primary training language and exhibit a further drop in performance on the Hindi subset. This reveals key limitations in structure-aware visual text understanding, numerical reasoning, and cross-lingual generalization. The dataset is publicly available via Hugging Face at this https URL, to promote LVLM research in this direction. 

---
# CoViPAL: Layer-wise Contextualized Visual Token Pruning for Large Vision-Language Models 

**Authors**: Zicong Tang, Ziyang Ma, Suqing Wang, Zuchao Li, Lefei Zhang, Hai Zhao, Yun Li, Qianren Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17243)  

**Abstract**: Large Vision-Language Models (LVLMs) process multimodal inputs consisting of text tokens and vision tokens extracted from images or videos. Due to the rich visual information, a single image can generate thousands of vision tokens, leading to high computational costs during the prefilling stage and significant memory overhead during decoding. Existing methods attempt to prune redundant vision tokens, revealing substantial redundancy in visual representations. However, these methods often struggle in shallow layers due to the lack of sufficient contextual information. We argue that many visual tokens are inherently redundant even in shallow layers and can be safely and effectively pruned with appropriate contextual signals. In this work, we propose CoViPAL, a layer-wise contextualized visual token pruning method that employs a Plug-and-Play Pruning Module (PPM) to predict and remove redundant vision tokens before they are processed by the LVLM. The PPM is lightweight, model-agnostic, and operates independently of the LVLM architecture, ensuring seamless integration with various models. Extensive experiments on multiple benchmarks demonstrate that CoViPAL outperforms training-free pruning methods under equal token budgets and surpasses training-based methods with comparable supervision. CoViPAL offers a scalable and efficient solution to improve inference efficiency in LVLMs without compromising accuracy. 

---
# Multi-Agent Visual-Language Reasoning for Comprehensive Highway Scene Understanding 

**Authors**: Yunxiang Yang, Ningning Xu, Jidong J. Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.17205)  

**Abstract**: This paper introduces a multi-agent framework for comprehensive highway scene understanding, designed around a mixture-of-experts strategy. In this framework, a large generic vision-language model (VLM), such as GPT-4o, is contextualized with domain knowledge to generates task-specific chain-of-thought (CoT) prompts. These fine-grained prompts are then used to guide a smaller, efficient VLM (e.g., Qwen2.5-VL-7B) in reasoning over short videos, along with complementary modalities as applicable. The framework simultaneously addresses multiple critical perception tasks, including weather classification, pavement wetness assessment, and traffic congestion detection, achieving robust multi-task reasoning while balancing accuracy and computational efficiency. To support empirical validation, we curated three specialized datasets aligned with these tasks. Notably, the pavement wetness dataset is multimodal, combining video streams with road weather sensor data, highlighting the benefits of multimodal reasoning. Experimental results demonstrate consistently strong performance across diverse traffic and environmental conditions. From a deployment perspective, the framework can be readily integrated with existing traffic camera systems and strategically applied to high-risk rural locations, such as sharp curves, flood-prone lowlands, or icy bridges. By continuously monitoring the targeted sites, the system enhances situational awareness and delivers timely alerts, even in resource-constrained environments. 

---
# LLM Assertiveness can be Mechanistically Decomposed into Emotional and Logical Components 

**Authors**: Hikaru Tsujimura, Arush Tagade  

**Link**: [PDF](https://arxiv.org/pdf/2508.17182)  

**Abstract**: Large Language Models (LLMs) often display overconfidence, presenting information with unwarranted certainty in high-stakes contexts. We investigate the internal basis of this behavior via mechanistic interpretability. Using open-sourced Llama 3.2 models fine-tuned on human annotated assertiveness datasets, we extract residual activations across all layers, and compute similarity metrics to localize assertive representations. Our analysis identifies layers most sensitive to assertiveness contrasts and reveals that high-assertive representations decompose into two orthogonal sub-components of emotional and logical clusters-paralleling the dual-route Elaboration Likelihood Model in Psychology. Steering vectors derived from these sub-components show distinct causal effects: emotional vectors broadly influence prediction accuracy, while logical vectors exert more localized effects. These findings provide mechanistic evidence for the multi-component structure of LLM assertiveness and highlight avenues for mitigating overconfident behavior. 

---
# Anemoi: A Semi-Centralized Multi-agent Systems Based on Agent-to-Agent Communication MCP server from Coral Protocol 

**Authors**: Xinxing Ren, Caelum Forder, Qianbo Zang, Ahsen Tahir, Roman J. Georgio, Suman Deb, Peter Carroll, Önder Gürcan, Zekun Guo  

**Link**: [PDF](https://arxiv.org/pdf/2508.17068)  

**Abstract**: Recent advances in generalist multi-agent systems (MAS) have largely followed a context-engineering plus centralized paradigm, where a planner agent coordinates multiple worker agents through unidirectional prompt passing. While effective under strong planner models, this design suffers from two critical limitations: (1) strong dependency on the planner's capability, which leads to degraded performance when a smaller LLM powers the planner; and (2) limited inter-agent communication, where collaboration relies on costly prompt concatenation and context injection, introducing redundancy and information loss. To address these challenges, we propose Anemoi, a semi-centralized MAS built on the Agent-to-Agent (A2A) communication MCP server from Coral Protocol. Unlike traditional designs, Anemoi enables structured and direct inter-agent collaboration, allowing all agents to monitor progress, assess results, identify bottlenecks, and propose refinements in real time. This paradigm reduces reliance on a single planner, supports adaptive plan updates, and minimizes redundant context passing, resulting in more scalable and cost-efficient execution. Evaluated on the GAIA benchmark, Anemoi achieved 52.73\% accuracy with a small LLM (GPT-4.1-mini) as the planner, surpassing the strongest open-source baseline OWL (43.63\%) by +9.09\% under identical LLM settings. Our implementation is publicly available at this https URL. 

---
# RephraseTTS: Dynamic Length Text based Speech Insertion with Speaker Style Transfer 

**Authors**: Neeraj Matiyali, Siddharth Srivastava, Gaurav Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2508.17031)  

**Abstract**: We propose a method for the task of text-conditioned speech insertion, i.e. inserting a speech sample in an input speech sample, conditioned on the corresponding complete text transcript. An example use case of the task would be to update the speech audio when corrections are done on the corresponding text transcript. The proposed method follows a transformer-based non-autoregressive approach that allows speech insertions of variable lengths, which are dynamically determined during inference, based on the text transcript and tempo of the available partial input. It is capable of maintaining the speaker's voice characteristics, prosody and other spectral properties of the available speech input. Results from our experiments and user study on LibriTTS show that our method outperforms baselines based on an existing adaptive text to speech method. We also provide numerous qualitative results to appreciate the quality of the output from the proposed method. 

---
# THEME : Enhancing Thematic Investing with Semantic Stock Representations and Temporal Dynamics 

**Authors**: Hoyoung Lee, Wonbin Ahn, Suhwan Park, Jaehoon Lee, Minjae Kim, Sungdong Yoo, Taeyoon Lim, Woohyung Lim, Yongjae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.16936)  

**Abstract**: Thematic investing aims to construct portfolios aligned with structural trends, yet selecting relevant stocks remains challenging due to overlapping sector boundaries and evolving market dynamics. To address this challenge, we construct the Thematic Representation Set (TRS), an extended dataset that begins with real-world thematic ETFs and expands upon them by incorporating industry classifications and financial news to overcome their coverage limitations. The final dataset contains both the explicit mapping of themes to their constituent stocks and the rich textual profiles for each. Building on this dataset, we introduce \textsc{THEME}, a hierarchical contrastive learning framework. By representing the textual profiles of themes and stocks as embeddings, \textsc{THEME} first leverages their hierarchical relationship to achieve semantic alignment. Subsequently, it refines these semantic embeddings through a temporal refinement stage that incorporates individual stock returns. The final stock representations are designed for effective retrieval of thematically aligned assets with strong return potential. Empirical results show that \textsc{THEME} outperforms strong baselines across multiple retrieval metrics and significantly improves performance in portfolio construction. By jointly modeling thematic relationships from text and market dynamics from returns, \textsc{THEME} provides a scalable and adaptive solution for navigating complex investment themes. 

---
# Attention Layers Add Into Low-Dimensional Residual Subspaces 

**Authors**: Junxuan Wang, Xuyang Ge, Wentao Shu, Zhengfu He, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2508.16929)  

**Abstract**: While transformer models are widely believed to operate in high-dimensional hidden spaces, we show that attention outputs are confined to a surprisingly low-dimensional subspace, where about 60\% of the directions account for 99\% of the variance--a phenomenon that is induced by the attention output projection matrix and consistently observed across diverse model families and datasets. Critically, we find this low-rank structure as a fundamental cause of the prevalent dead feature problem in sparse dictionary learning, where it creates a mismatch between randomly initialized features and the intrinsic geometry of the activation space. Building on this insight, we propose a subspace-constrained training method for sparse autoencoders (SAEs), initializing feature directions into the active subspace of activations. Our approach reduces dead features from 87\% to below 1\% in Attention Output SAEs with 1M features, and can further extend to other sparse dictionary learning methods. Our findings provide both new insights into the geometry of attention and practical tools for improving sparse dictionary learning in large language models. 

---
# Quantifying Sycophancy as Deviations from Bayesian Rationality in LLMs 

**Authors**: Katherine Atwell, Pedram Heydari, Anthony Sicilia, Malihe Alikhani  

**Link**: [PDF](https://arxiv.org/pdf/2508.16846)  

**Abstract**: Sycophancy, or overly agreeable or flattering behavior, is a documented issue in large language models (LLMs), and is critical to understand in the context of human/AI collaboration. Prior works typically quantify sycophancy by measuring shifts in behavior or impacts on accuracy, but neither metric characterizes shifts in rationality, and accuracy measures can only be used in scenarios with a known ground truth. In this work, we utilize a Bayesian framework to quantify sycophancy as deviations from rational behavior when presented with user perspectives, thus distinguishing between rational and irrational updates based on the introduction of user perspectives. In comparison to other methods, this approach allows us to characterize excessive behavioral shifts, even for tasks that involve inherent uncertainty or do not have a ground truth. We study sycophancy for 3 different tasks, a combination of open-source and closed LLMs, and two different methods for probing sycophancy. We also experiment with multiple methods for eliciting probability judgments from LLMs. We hypothesize that probing LLMs for sycophancy will cause deviations in LLMs' predicted posteriors that will lead to increased Bayesian error. Our findings indicate that: 1) LLMs are not Bayesian rational, 2) probing for sycophancy results in significant increases to the predicted posterior in favor of the steered outcome, 3) sycophancy sometimes results in increased Bayesian error, and in a small number of cases actually decreases error, and 4) changes in Bayesian error due to sycophancy are not strongly correlated in Brier score, suggesting that studying the impact of sycophancy on ground truth alone does not fully capture errors in reasoning due to sycophancy. 

---
# Interpreting the Effects of Quantization on LLMs 

**Authors**: Manpreet Singh, Hassan Sajjad  

**Link**: [PDF](https://arxiv.org/pdf/2508.16785)  

**Abstract**: Quantization offers a practical solution to deploy LLMs in resource-constraint environments. However, its impact on internal representations remains understudied, raising questions about the reliability of quantized models. In this study, we employ a range of interpretability techniques to investigate how quantization affects model and neuron behavior. We analyze multiple LLMs under 4-bit and 8-bit quantization. Our findings reveal that the impact of quantization on model calibration is generally minor. Analysis of neuron activations indicates that the number of dead neurons, i.e., those with activation values close to 0 across the dataset, remains consistent regardless of quantization. In terms of neuron contribution to predictions, we observe that smaller full precision models exhibit fewer salient neurons, whereas larger models tend to have more, with the exception of Llama-2-7B. The effect of quantization on neuron redundancy varies across models. Overall, our findings suggest that effect of quantization may vary by model and tasks, however, we did not observe any drastic change which may discourage the use of quantization as a reliable model compression technique. 

---
# Guarding Your Conversations: Privacy Gatekeepers for Secure Interactions with Cloud-Based AI Models 

**Authors**: GodsGift Uzor, Hasan Al-Qudah, Ynes Ineza, Abdul Serwadda  

**Link**: [PDF](https://arxiv.org/pdf/2508.16765)  

**Abstract**: The interactive nature of Large Language Models (LLMs), which closely track user data and context, has prompted users to share personal and private information in unprecedented ways. Even when users opt out of allowing their data to be used for training, these privacy settings offer limited protection when LLM providers operate in jurisdictions with weak privacy laws, invasive government surveillance, or poor data security practices. In such cases, the risk of sensitive information, including Personally Identifiable Information (PII), being mishandled or exposed remains high. To address this, we propose the concept of an "LLM gatekeeper", a lightweight, locally run model that filters out sensitive information from user queries before they are sent to the potentially untrustworthy, though highly capable, cloud-based LLM. Through experiments with human subjects, we demonstrate that this dual-model approach introduces minimal overhead while significantly enhancing user privacy, without compromising the quality of LLM responses. 

---
# Hyperbolic Multimodal Representation Learning for Biological Taxonomies 

**Authors**: ZeMing Gong, Chuanqi Tang, Xiaoliang Huo, Nicholas Pellegrino, Austin T. Wang, Graham W. Taylor, Angel X. Chang, Scott C. Lowe, Joakim Bruslund Haurum  

**Link**: [PDF](https://arxiv.org/pdf/2508.16744)  

**Abstract**: Taxonomic classification in biodiversity research involves organizing biological specimens into structured hierarchies based on evidence, which can come from multiple modalities such as images and genetic information. We investigate whether hyperbolic networks can provide a better embedding space for such hierarchical models. Our method embeds multimodal inputs into a shared hyperbolic space using contrastive and a novel stacked entailment-based objective. Experiments on the BIOSCAN-1M dataset show that hyperbolic embedding achieves competitive performance with Euclidean baselines, and outperforms all other models on unseen species classification using DNA barcodes. However, fine-grained classification and open-world generalization remain challenging. Our framework offers a structure-aware foundation for biodiversity modelling, with potential applications to species discovery, ecological monitoring, and conservation efforts. 

---
# Revisiting Rule-Based Stuttering Detection: A Comprehensive Analysis of Interpretable Models for Clinical Applications 

**Authors**: Eric Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16681)  

**Abstract**: Stuttering affects approximately 1% of the global population, impacting communication and quality of life. While recent advances in deep learning have pushed the boundaries of automatic speech dysfluency detection, rule-based approaches remain crucial for clinical applications where interpretability and transparency are paramount. This paper presents a comprehensive analysis of rule-based stuttering detection systems, synthesizing insights from multiple corpora including UCLASS, FluencyBank, and SEP-28k. We propose an enhanced rule-based framework that incorporates speaking-rate normalization, multi-level acoustic feature analysis, and hierarchical decision structures. Our approach achieves competitive performance while maintaining complete interpretability-critical for clinical adoption. We demonstrate that rule-based systems excel particularly in prolongation detection (97-99% accuracy) and provide stable performance across varying speaking rates. Furthermore, we show how these interpretable models can be integrated with modern machine learning pipelines as proposal generators or constraint modules, bridging the gap between traditional speech pathology practices and contemporary AI systems. Our analysis reveals that while neural approaches may achieve marginally higher accuracy in unconstrained settings, rule-based methods offer unique advantages in clinical contexts where decision auditability, patient-specific tuning, and real-time feedback are essential. 

---
# Recall-Extend Dynamics: Enhancing Small Language Models through Controlled Exploration and Refined Offline Integration 

**Authors**: Zhong Guan, Likang Wu, Hongke Zhao, Jiahui Wang, Le Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.16677)  

**Abstract**: Many existing studies have achieved significant improvements in the reasoning capabilities of large language models (LLMs) through reinforcement learning with verifiable rewards (RLVR), while the enhancement of reasoning abilities in small language models (SLMs) has not yet been sufficiently explored. Combining distilled data from larger models with RLVR on small models themselves is a natural approach, but it still faces various challenges and issues. Therefore, we propose \textit{\underline{R}}ecall-\textit{\underline{E}}xtend \textit{\underline{D}}ynamics(RED): Enhancing Small Language Models through Controlled Exploration and Refined Offline Integration. In this paper, we explore the perspective of varying exploration spaces, balancing offline distillation with online reinforcement learning. Simultaneously, we specifically design and optimize for the insertion problem within offline data. By monitoring the ratio of entropy changes in the model concerning offline and online data, we regulate the weight of offline-SFT, thereby addressing the issues of insufficient exploration space in small models and the redundancy and complexity during the distillation process. Furthermore, to tackle the distribution discrepancies between offline data and the current policy, we design a sample-accuracy-based policy shift mechanism that dynamically chooses between imitating offline distilled data and learning from its own policy. 

---
# WISCA: A Lightweight Model Transition Method to Improve LLM Training via Weight Scaling 

**Authors**: Jiacheng Li, Jianchao Tan, Zhidong Yang, Pingwei Sun, Feiye Huo, Jiayu Qin, Yerui Sun, Yuchen Xie, Xunliang Cai, Xiangyu Zhang, Maoxin He, Guangming Tan, Weile Jia, Tong Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.16676)  

**Abstract**: Transformer architecture gradually dominates the LLM field. Recent advances in training optimization for Transformer-based large language models (LLMs) primarily focus on architectural modifications or optimizer adjustments. However, these approaches lack systematic optimization of weight patterns during training. Weight pattern refers to the distribution and relative magnitudes of weight parameters in a neural network. To address this issue, we propose a Weight Scaling method called WISCA to enhance training efficiency and model quality by strategically improving neural network weight patterns without changing network structures. By rescaling weights while preserving model outputs, WISCA indirectly optimizes the model's training trajectory. Experiments demonstrate that WISCA significantly improves convergence quality (measured by generalization capability and loss reduction), particularly in LLMs with Grouped Query Attention (GQA) architectures and LoRA fine-tuning tasks. Empirical results show 5.6% average improvement on zero-shot validation tasks and 2.12% average reduction in training perplexity across multiple architectures. 

---
# MedRepBench: A Comprehensive Benchmark for Medical Report Interpretation 

**Authors**: Fangxin Shang, Yuan Xia, Dalu Yang, Yahui Wang, Binglin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.16674)  

**Abstract**: Medical report interpretation plays a crucial role in healthcare, enabling both patient-facing explanations and effective information flow across clinical systems. While recent vision-language models (VLMs) and large language models (LLMs) have demonstrated general document understanding capabilities, there remains a lack of standardized benchmarks to assess structured interpretation quality in medical reports. We introduce MedRepBench, a comprehensive benchmark built from 1,900 de-identified real-world Chinese medical reports spanning diverse departments, patient demographics, and acquisition formats. The benchmark is designed primarily to evaluate end-to-end VLMs for structured medical report understanding. To enable controlled comparisons, we also include a text-only evaluation setting using high-quality OCR outputs combined with LLMs, allowing us to estimate the upper-bound performance when character recognition errors are minimized. Our evaluation framework supports two complementary protocols: (1) an objective evaluation measuring field-level recall of structured clinical items, and (2) an automated subjective evaluation using a powerful LLM as a scoring agent to assess factuality, interpretability, and reasoning quality. Based on the objective metric, we further design a reward function and apply Group Relative Policy Optimization (GRPO) to improve a mid-scale VLM, achieving up to 6% recall gain. We also observe that the OCR+LLM pipeline, despite strong performance, suffers from layout-blindness and latency issues, motivating further progress toward robust, fully vision-based report understanding. 

---
# Invisible Filters: Cultural Bias in Hiring Evaluations Using Large Language Models 

**Authors**: Pooja S. B. Rao, Laxminarayen Nagarajan Venkatesan, Mauro Cherubini, Dinesh Babu Jayagopi  

**Link**: [PDF](https://arxiv.org/pdf/2508.16673)  

**Abstract**: Artificial Intelligence (AI) is increasingly used in hiring, with large language models (LLMs) having the potential to influence or even make hiring decisions. However, this raises pressing concerns about bias, fairness, and trust, particularly across diverse cultural contexts. Despite their growing role, few studies have systematically examined the potential biases in AI-driven hiring evaluation across cultures. In this study, we conduct a systematic analysis of how LLMs assess job interviews across cultural and identity dimensions. Using two datasets of interview transcripts, 100 from UK and 100 from Indian job seekers, we first examine cross-cultural differences in LLM-generated scores for hirability and related traits. Indian transcripts receive consistently lower scores than UK transcripts, even when they were anonymized, with disparities linked to linguistic features such as sentence complexity and lexical diversity. We then perform controlled identity substitutions (varying names by gender, caste, and region) within the Indian dataset to test for name-based bias. These substitutions do not yield statistically significant effects, indicating that names alone, when isolated from other contextual signals, may not influence LLM evaluations. Our findings underscore the importance of evaluating both linguistic and social dimensions in LLM-driven evaluations and highlight the need for culturally sensitive design and accountability in AI-assisted hiring. 

---
# Leveraging Multi-Source Textural UGC for Neighbourhood Housing Quality Assessment: A GPT-Enhanced Framework 

**Authors**: Qiyuan Hong, Huimin Zhao, Ying Long  

**Link**: [PDF](https://arxiv.org/pdf/2508.16657)  

**Abstract**: This study leverages GPT-4o to assess neighbourhood housing quality using multi-source textural user-generated content (UGC) from Dianping, Weibo, and the Government Message Board. The analysis involves filtering relevant texts, extracting structured evaluation units, and conducting sentiment scoring. A refined housing quality assessment system with 46 indicators across 11 categories was developed, highlighting an objective-subjective method gap and platform-specific differences in focus. GPT-4o outperformed rule-based and BERT models, achieving 92.5% accuracy in fine-tuned settings. The findings underscore the value of integrating UGC and GPT-driven analysis for scalable, resident-centric urban assessments, offering practical insights for policymakers and urban planners. 

---
# Empirical Analysis of the Effect of Context in the Task of Automated Essay Scoring in Transformer-Based Models 

**Authors**: Abhirup Chakravarty  

**Link**: [PDF](https://arxiv.org/pdf/2508.16638)  

**Abstract**: Automated Essay Scoring (AES) has emerged to prominence in response to the growing demand for educational automation. Providing an objective and cost-effective solution, AES standardises the assessment of extended responses. Although substantial research has been conducted in this domain, recent investigations reveal that alternative deep-learning architectures outperform transformer-based models. Despite the successful dominance in the performance of the transformer architectures across various other tasks, this discrepancy has prompted a need to enrich transformer-based AES models through contextual enrichment.
This study delves into diverse contextual factors using the ASAP-AES dataset, analysing their impact on transformer-based model performance. Our most effective model, augmented with multiple contextual dimensions, achieves a mean Quadratic Weighted Kappa score of 0.823 across the entire essay dataset and 0.8697 when trained on individual essay sets. Evidently surpassing prior transformer-based models, this augmented approach only underperforms relative to the state-of-the-art deep learning model trained essay-set-wise by an average of 3.83\% while exhibiting superior performance in three of the eight sets.
Importantly, this enhancement is orthogonal to architecture-based advancements and seamlessly adaptable to any AES model. Consequently, this contextual augmentation methodology presents a versatile technique for refining AES capabilities, contributing to automated grading and evaluation evolution in educational settings. 

---
# Learn to Memorize: Optimizing LLM-based Agents with Adaptive Memory Framework 

**Authors**: Zeyu Zhang, Quanyu Dai, Rui Li, Xiaohe Bo, Xu Chen, Zhenhua Dong  

**Link**: [PDF](https://arxiv.org/pdf/2508.16629)  

**Abstract**: LLM-based agents have been extensively applied across various domains, where memory stands out as one of their most essential capabilities. Previous memory mechanisms of LLM-based agents are manually predefined by human experts, leading to higher labor costs and suboptimal performance. In addition, these methods overlook the memory cycle effect in interactive scenarios, which is critical to optimizing LLM-based agents for specific environments. To address these challenges, in this paper, we propose to optimize LLM-based agents with an adaptive and data-driven memory framework by modeling memory cycles. Specifically, we design an MoE gate function to facilitate memory retrieval, propose a learnable aggregation process to improve memory utilization, and develop task-specific reflection to adapt memory storage. Our memory framework empowers LLM-based agents to learn how to memorize information effectively in specific environments, with both off-policy and on-policy optimization. In order to evaluate the effectiveness of our proposed methods, we conduct comprehensive experiments across multiple aspects. To benefit the research community in this area, we release our project at this https URL. 

---
# Humans Perceive Wrong Narratives from AI Reasoning Texts 

**Authors**: Mosh Levy, Zohar Elyoseph, Yoav Goldberg  

**Link**: [PDF](https://arxiv.org/pdf/2508.16599)  

**Abstract**: A new generation of AI models generates step-by-step reasoning text before producing an answer. This text appears to offer a human-readable window into their computation process, and is increasingly relied upon for transparency and interpretability. However, it is unclear whether human understanding of this text matches the model's actual computational process. In this paper, we investigate a necessary condition for correspondence: the ability of humans to identify which steps in a reasoning text causally influence later steps. We evaluated humans on this ability by composing questions based on counterfactual measurements and found a significant discrepancy: participant accuracy was only 29.3%, barely above chance (25%), and remained low (42%) even when evaluating the majority vote on questions with high agreement. Our results reveal a fundamental gap between how humans interpret reasoning texts and how models use it, challenging its utility as a simple interpretability tool. We argue that reasoning texts should be treated as an artifact to be investigated, not taken at face value, and that understanding the non-human ways these models use language is a critical research direction. 

---
