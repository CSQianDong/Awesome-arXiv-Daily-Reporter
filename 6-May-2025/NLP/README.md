# ReplaceMe: Network Simplification via Layer Pruning and Linear Transformations 

**Authors**: Dmitriy Shopkhoev, Ammar Ali, Magauiya Zhussip, Valentin Malykh, Stamatios Lefkimmiatis, Nikos Komodakis, Sergey Zagoruyko  

**Link**: [PDF](https://arxiv.org/pdf/2505.02819)  

**Abstract**: We introduce ReplaceMe, a generalized training-free depth pruning method that effectively replaces transformer blocks with a linear operation, while maintaining high performance for low compression ratios. In contrast to conventional pruning approaches that require additional training or fine-tuning, our approach requires only a small calibration dataset that is used to estimate a linear transformation to approximate the pruned blocks. This estimated linear mapping can be seamlessly merged with the remaining transformer blocks, eliminating the need for any additional network parameters. Our experiments show that ReplaceMe consistently outperforms other training-free approaches and remains highly competitive with state-of-the-art pruning methods that involve extensive retraining/fine-tuning and architectural modifications. Applied to several large language models (LLMs), ReplaceMe achieves up to 25% pruning while retaining approximately 90% of the original model's performance on open benchmarks - without any training or healing steps, resulting in minimal computational overhead (see Fig.1). We provide an open-source library implementing ReplaceMe alongside several state-of-the-art depth pruning techniques, available at this repository. 

---
# Bye-bye, Bluebook? Automating Legal Procedure with Large Language Models 

**Authors**: Matthew Dahl  

**Link**: [PDF](https://arxiv.org/pdf/2505.02763)  

**Abstract**: Legal practice requires careful adherence to procedural rules. In the United States, few are more complex than those found in The Bluebook: A Uniform System of Citation. Compliance with this system's 500+ pages of byzantine formatting instructions is the raison d'etre of thousands of student law review editors and the bete noire of lawyers everywhere. To evaluate whether large language models (LLMs) are able to adhere to the procedures of such a complicated system, we construct an original dataset of 866 Bluebook tasks and test flagship LLMs from OpenAI, Anthropic, Google, Meta, and DeepSeek. We show (1) that these models produce fully compliant Bluebook citations only 69%-74% of the time and (2) that in-context learning on the Bluebook's underlying system of rules raises accuracy only to 77%. These results caution against using off-the-shelf LLMs to automate aspects of the law where fidelity to procedure is paramount. 

---
# fastabx: A library for efficient computation of ABX discriminability 

**Authors**: Maxime Poli, Emmanuel Chemla, Emmanuel Dupoux  

**Link**: [PDF](https://arxiv.org/pdf/2505.02692)  

**Abstract**: We introduce fastabx, a high-performance Python library for building ABX discrimination tasks. ABX is a measure of the separation between generic categories of interest. It has been used extensively to evaluate phonetic discriminability in self-supervised speech representations. However, its broader adoption has been limited by the absence of adequate tools. fastabx addresses this gap by providing a framework capable of constructing any type of ABX task while delivering the efficiency necessary for rapid development cycles, both in task creation and in calculating distances between representations. We believe that fastabx will serve as a valuable resource for the broader representation learning community, enabling researchers to systematically investigate what information can be directly extracted from learned representations across several domains beyond speech processing. The source code is available at this https URL. 

---
# Sailing AI by the Stars: A Survey of Learning from Rewards in Post-Training and Test-Time Scaling of Large Language Models 

**Authors**: Xiaobao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.02686)  

**Abstract**: Recent developments in Large Language Models (LLMs) have shifted from pre-training scaling to post-training and test-time scaling. Across these developments, a key unified paradigm has arisen: Learning from Rewards, where reward signals act as the guiding stars to steer LLM behavior. It has underpinned a wide range of prevalent techniques, such as reinforcement learning (in RLHF, DPO, and GRPO), reward-guided decoding, and post-hoc correction. Crucially, this paradigm enables the transition from passive learning from static data to active learning from dynamic feedback. This endows LLMs with aligned preferences and deep reasoning capabilities. In this survey, we present a comprehensive overview of the paradigm of learning from rewards. We categorize and analyze the strategies under this paradigm across training, inference, and post-inference stages. We further discuss the benchmarks for reward models and the primary applications. Finally we highlight the challenges and future directions. We maintain a paper collection at this https URL. 

---
# A Survey on Progress in LLM Alignment from the Perspective of Reward Design 

**Authors**: Miaomiao Ji, Yanqiu Wu, Zhibin Wu, Shoujin Wang, Jian Yang, Mark Dras, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2505.02666)  

**Abstract**: The alignment of large language models (LLMs) with human values and intentions represents a core challenge in current AI research, where reward mechanism design has become a critical factor in shaping model behavior. This study conducts a comprehensive investigation of reward mechanisms in LLM alignment through a systematic theoretical framework, categorizing their development into three key phases: (1) feedback (diagnosis), (2) reward design (prescription), and (3) optimization (treatment). Through a four-dimensional analysis encompassing construction basis, format, expression, and granularity, this research establishes a systematic classification framework that reveals evolutionary trends in reward modeling. The field of LLM alignment faces several persistent challenges, while recent advances in reward design are driving significant paradigm shifts. Notable developments include the transition from reinforcement learning-based frameworks to novel optimization paradigms, as well as enhanced capabilities to address complex alignment scenarios involving multimodal integration and concurrent task coordination. Finally, this survey outlines promising future research directions for LLM alignment through innovative reward design strategies. 

---
# Proper Name Diacritization for Arabic Wikipedia: A Benchmark Dataset 

**Authors**: Rawan Bondok, Mayar Nassar, Salam Khalifa, Kurt Micallaf, Nizar Habash  

**Link**: [PDF](https://arxiv.org/pdf/2505.02656)  

**Abstract**: Proper names in Arabic Wikipedia are frequently undiacritized, creating ambiguity in pronunciation and interpretation, especially for transliterated named entities of foreign origin. While transliteration and diacritization have been well-studied separately in Arabic NLP,their intersection remains underexplored. In this paper, we introduce a new manually diacritized dataset of Arabic proper names of various origins with their English Wikipedia equivalent glosses, and present the challenges and guidelines we followed to create it. We benchmark GPT-4o on the task of recovering full diacritization given the undiacritized Arabic and English forms, and analyze its performance. Achieving 73% accuracy, our results underscore both the difficulty of the task and the need for improved models and resources. We release our dataset to facilitate further research on Arabic Wikipedia proper name diacritization. 

---
# LLaMA-Omni2: LLM-based Real-time Spoken Chatbot with Autoregressive Streaming Speech Synthesis 

**Authors**: Qingkai Fang, Yan Zhou, Shoutao Guo, Shaolei Zhang, Yang Feng  

**Link**: [PDF](https://arxiv.org/pdf/2505.02625)  

**Abstract**: Real-time, intelligent, and natural speech interaction is an essential part of the next-generation human-computer interaction. Recent advancements have showcased the potential of building intelligent spoken chatbots based on large language models (LLMs). In this paper, we introduce LLaMA-Omni 2, a series of speech language models (SpeechLMs) ranging from 0.5B to 14B parameters, capable of achieving high-quality real-time speech interaction. LLaMA-Omni 2 is built upon the Qwen2.5 series models, integrating a speech encoder and an autoregressive streaming speech decoder. Despite being trained on only 200K multi-turn speech dialogue samples, LLaMA-Omni 2 demonstrates strong performance on several spoken question answering and speech instruction following benchmarks, surpassing previous state-of-the-art SpeechLMs like GLM-4-Voice, which was trained on millions of hours of speech data. 

---
# Automatic Proficiency Assessment in L2 English Learners 

**Authors**: Armita Mohammadi, Alessandro Lameiras Koerich, Laureano Moro-Velazquez, Patrick Cardinal  

**Link**: [PDF](https://arxiv.org/pdf/2505.02615)  

**Abstract**: Second language proficiency (L2) in English is usually perceptually evaluated by English teachers or expert evaluators, with the inherent intra- and inter-rater variability. This paper explores deep learning techniques for comprehensive L2 proficiency assessment, addressing both the speech signal and its correspondent transcription. We analyze spoken proficiency classification prediction using diverse architectures, including 2D CNN, frequency-based CNN, ResNet, and a pretrained wav2vec 2.0 model. Additionally, we examine text-based proficiency assessment by fine-tuning a BERT language model within resource constraints. Finally, we tackle the complex task of spontaneous dialogue assessment, managing long-form audio and speaker interactions through separate applications of wav2vec 2.0 and BERT models. Results from experiments on EFCamDat and ANGLISH datasets and a private dataset highlight the potential of deep learning, especially the pretrained wav2vec 2.0 model, for robust automated L2 proficiency evaluation. 

---
# Ensemble Kalman filter for uncertainty in human language comprehension 

**Authors**: Diksha Bhandari, Alessandro Lopopolo, Milena Rabovsky, Sebastian Reich  

**Link**: [PDF](https://arxiv.org/pdf/2505.02590)  

**Abstract**: Artificial neural networks (ANNs) are widely used in modeling sentence processing but often exhibit deterministic behavior, contrasting with human sentence comprehension, which manages uncertainty during ambiguous or unexpected inputs. This is exemplified by reversal anomalies-sentences with unexpected role reversals that challenge syntax and semantics-highlighting the limitations of traditional ANN models, such as the Sentence Gestalt (SG) Model. To address these limitations, we propose a Bayesian framework for sentence comprehension, applying an extension of the ensemble Kalman filter (EnKF) for Bayesian inference to quantify uncertainty. By framing language comprehension as a Bayesian inverse problem, this approach enhances the SG model's ability to reflect human sentence processing with respect to the representation of uncertainty. Numerical experiments and comparisons with maximum likelihood estimation (MLE) demonstrate that Bayesian methods improve uncertainty representation, enabling the model to better approximate human cognitive processing when dealing with linguistic ambiguities. 

---
# EMORL: Ensemble Multi-Objective Reinforcement Learning for Efficient and Flexible LLM Fine-Tuning 

**Authors**: Lingxiao Kong, Cong Yang, Susanne Neufang, Oya Deniz Beyan, Zeyd Boukhers  

**Link**: [PDF](https://arxiv.org/pdf/2505.02579)  

**Abstract**: Recent advances in reinforcement learning (RL) for large language model (LLM) fine-tuning show promise in addressing multi-objective tasks but still face significant challenges, including complex objective balancing, low training efficiency, poor scalability, and limited explainability. Leveraging ensemble learning principles, we introduce an Ensemble Multi-Objective RL (EMORL) framework that fine-tunes multiple models with individual objectives while optimizing their aggregation after the training to improve efficiency and flexibility. Our method is the first to aggregate the last hidden states of individual models, incorporating contextual information from multiple objectives. This approach is supported by a hierarchical grid search algorithm that identifies optimal weighted combinations. We evaluate EMORL on counselor reflection generation tasks, using text-scoring LLMs to evaluate the generations and provide rewards during RL fine-tuning. Through comprehensive experiments on the PAIR and Psych8k datasets, we demonstrate the advantages of EMORL against existing baselines: significantly lower and more stable training consumption ($17,529\pm 1,650$ data points and $6,573\pm 147.43$ seconds), improved scalability and explainability, and comparable performance across multiple objectives. 

---
# Bemba Speech Translation: Exploring a Low-Resource African Language 

**Authors**: Muhammad Hazim Al Farouq, Aman Kassahun Wassie, Yasmin Moslem  

**Link**: [PDF](https://arxiv.org/pdf/2505.02518)  

**Abstract**: This paper describes our system submission to the International Conference on Spoken Language Translation (IWSLT 2025), low-resource languages track, namely for Bemba-to-English speech translation. We built cascaded speech translation systems based on Whisper and NLLB-200, and employed data augmentation techniques, such as back-translation. We investigate the effect of using synthetic data and discuss our experimental setup. 

---
# Data Augmentation With Back translation for Low Resource languages: A case of English and Luganda 

**Authors**: Richard Kimera, Dongnyeong Heo, Daniela N. Rim, Heeyoul Choi  

**Link**: [PDF](https://arxiv.org/pdf/2505.02463)  

**Abstract**: In this paper,we explore the application of Back translation (BT) as a semi-supervised technique to enhance Neural Machine Translation(NMT) models for the English-Luganda language pair, specifically addressing the challenges faced by low-resource languages. The purpose of our study is to demonstrate how BT can mitigate the scarcity of bilingual data by generating synthetic data from monolingual corpora. Our methodology involves developing custom NMT models using both publicly available and web-crawled data, and applying Iterative and Incremental Back translation techniques. We strategically select datasets for incremental back translation across multiple small datasets, which is a novel element of our approach. The results of our study show significant improvements, with translation performance for the English-Luganda pair exceeding previous benchmarks by more than 10 BLEU score units across all translation directions. Additionally, our evaluation incorporates comprehensive assessment metrics such as SacreBLEU, ChrF2, and TER, providing a nuanced understanding of translation quality. The conclusion drawn from our research confirms the efficacy of BT when strategically curated datasets are utilized, establishing new performance benchmarks and demonstrating the potential of BT in enhancing NMT models for low-resource languages. 

---
# Colombian Waitresses y Jueces canadienses: Gender and Country Biases in Occupation Recommendations from LLMs 

**Authors**: Elisa Forcada Rodríguez, Olatz Perez-de-Viñaspre, Jon Ander Campos, Dietrich Klakow, Vagrant Gautam  

**Link**: [PDF](https://arxiv.org/pdf/2505.02456)  

**Abstract**: One of the goals of fairness research in NLP is to measure and mitigate stereotypical biases that are propagated by NLP systems. However, such work tends to focus on single axes of bias (most often gender) and the English language. Addressing these limitations, we contribute the first study of multilingual intersecting country and gender biases, with a focus on occupation recommendations generated by large language models. We construct a benchmark of prompts in English, Spanish and German, where we systematically vary country and gender, using 25 countries and four pronoun sets. Then, we evaluate a suite of 5 Llama-based models on this benchmark, finding that LLMs encode significant gender and country biases. Notably, we find that even when models show parity for gender or country individually, intersectional occupational biases based on both country and gender persist. We also show that the prompting language significantly affects bias, and instruction-tuned models consistently demonstrate the lowest and most stable levels of bias. Our findings highlight the need for fairness researchers to use intersectional and multilingual lenses in their work. 

---
# Bielik 11B v2 Technical Report 

**Authors**: Krzysztof Ociepa, Łukasz Flis, Krzysztof Wróbel, Adrian Gwoździej, Remigiusz Kinas  

**Link**: [PDF](https://arxiv.org/pdf/2505.02410)  

**Abstract**: We present Bielik 11B v2, a state-of-the-art language model optimized for Polish text processing. Built on the Mistral 7B v0.2 architecture and scaled to 11B parameters using depth up-scaling, this model demonstrates exceptional performance across Polish language benchmarks while maintaining strong cross-lingual capabilities. We introduce two key technical innovations: Weighted Instruction Cross-Entropy Loss, which optimizes learning across diverse instruction types by assigning quality-based weights to training examples, and Adaptive Learning Rate, which dynamically adjusts based on context length. Comprehensive evaluation across multiple benchmarks demonstrates that Bielik 11B v2 outperforms many larger models, including those with 2-6 times more parameters, and significantly surpasses other specialized Polish language models on tasks ranging from linguistic understanding to complex reasoning. The model's parameter efficiency and extensive quantization options enable deployment across various hardware configurations, advancing Polish language AI capabilities and establishing new benchmarks for resource-efficient language modeling in less-represented languages. 

---
# RM-R1: Reward Modeling as Reasoning 

**Authors**: Xiusi Chen, Gaotang Li, Ziqi Wang, Bowen Jin, Cheng Qian, Yu Wang, Hongru Wang, Yu Zhang, Denghui Zhang, Tong Zhang, Hanghang Tong, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2505.02387)  

**Abstract**: Reward modeling is essential for aligning large language models (LLMs) with human preferences, especially through reinforcement learning from human feedback (RLHF). To provide accurate reward signals, a reward model (RM) should stimulate deep thinking and conduct interpretable reasoning before assigning a score or a judgment. However, existing RMs either produce opaque scalar scores or directly generate the prediction of a preferred answer, making them struggle to integrate natural language critiques, thus lacking interpretability. Inspired by recent advances of long chain-of-thought (CoT) on reasoning-intensive tasks, we hypothesize and validate that integrating reasoning capabilities into reward modeling significantly enhances RM's interpretability and performance. In this work, we introduce a new class of generative reward models -- Reasoning Reward Models (ReasRMs) -- which formulate reward modeling as a reasoning task. We propose a reasoning-oriented training pipeline and train a family of ReasRMs, RM-R1. The training consists of two key stages: (1) distillation of high-quality reasoning chains and (2) reinforcement learning with verifiable rewards. RM-R1 improves LLM rollouts by self-generating reasoning traces or chat-specific rubrics and evaluating candidate responses against them. Empirically, our models achieve state-of-the-art or near state-of-the-art performance of generative RMs across multiple comprehensive reward model benchmarks, outperforming much larger open-weight models (e.g., Llama3.1-405B) and proprietary ones (e.g., GPT-4o) by up to 13.8%. Beyond final performance, we perform thorough empirical analysis to understand the key ingredients of successful ReasRM training. To facilitate future research, we release six ReasRM models along with code and data at this https URL. 

---
# JTCSE: Joint Tensor-Modulus Constraints and Cross-Attention for Unsupervised Contrastive Learning of Sentence Embeddings 

**Authors**: Tianyu Zong, Hongzhu Yi, Bingkang Shi, Yuanxiang Wang, Jungang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.02366)  

**Abstract**: Unsupervised contrastive learning has become a hot research topic in natural language processing. Existing works usually aim at constraining the orientation distribution of the representations of positive and negative samples in the high-dimensional semantic space in contrastive learning, but the semantic representation tensor possesses both modulus and orientation features, and the existing works ignore the modulus feature of the representations and cause insufficient contrastive learning. % Therefore, we firstly propose a training objective that aims at modulus constraints on the semantic representation tensor, to strengthen the alignment between the positive samples in contrastive learning. Therefore, we first propose a training objective that is designed to impose modulus constraints on the semantic representation tensor, to strengthen the alignment between positive samples in contrastive learning. Then, the BERT-like model suffers from the phenomenon of sinking attention, leading to a lack of attention to CLS tokens that aggregate semantic information. In response, we propose a cross-attention structure among the twin-tower ensemble models to enhance the model's attention to CLS token and optimize the quality of CLS Pooling. Combining the above two motivations, we propose a new \textbf{J}oint \textbf{T}ensor representation modulus constraint and \textbf{C}ross-attention unsupervised contrastive learning \textbf{S}entence \textbf{E}mbedding representation framework JTCSE, which we evaluate in seven semantic text similarity computation tasks, and the experimental results show that JTCSE's twin-tower ensemble model and single-tower distillation model outperform the other baselines and become the current SOTA. In addition, we have conducted an extensive zero-shot downstream task evaluation, which shows that JTCSE outperforms other baselines overall on more than 130 tasks. 

---
# SIMPLEMIX: Frustratingly Simple Mixing of Off- and On-policy Data in Language Model Preference Learning 

**Authors**: Tianjian Li, Daniel Khashabi  

**Link**: [PDF](https://arxiv.org/pdf/2505.02363)  

**Abstract**: Aligning language models with human preferences relies on pairwise preference datasets. While some studies suggest that on-policy data consistently outperforms off -policy data for preference learning, others indicate that the advantages of on-policy data may be task-dependent, highlighting the need for a systematic exploration of their interplay.
In this work, we show that on-policy and off-policy data offer complementary strengths in preference optimization: on-policy data is particularly effective for reasoning tasks like math and coding, while off-policy data performs better on open-ended tasks such as creative writing and making personal recommendations. Guided by these findings, we introduce SIMPLEMIX, an approach to combine the complementary strengths of on-policy and off-policy preference learning by simply mixing these two data sources. Our empirical results across diverse tasks and benchmarks demonstrate that SIMPLEMIX substantially improves language model alignment. Specifically, SIMPLEMIX improves upon on-policy DPO and off-policy DPO by an average of 6.03% on Alpaca Eval 2.0. Moreover, it outperforms prior approaches that are much more complex in combining on- and off-policy data, such as HyPO and DPO-Mix-P, by an average of 3.05%. 

---
# Invoke Interfaces Only When Needed: Adaptive Invocation for Large Language Models in Question Answering 

**Authors**: Jihao Zhao, Chunlai Zhou, Biao Qin  

**Link**: [PDF](https://arxiv.org/pdf/2505.02311)  

**Abstract**: The collaborative paradigm of large and small language models (LMs) effectively balances performance and cost, yet its pivotal challenge lies in precisely pinpointing the moment of invocation when hallucinations arise in small LMs. Previous optimization efforts primarily focused on post-processing techniques, which were separate from the reasoning process of LMs, resulting in high computational costs and limited effectiveness. In this paper, we propose a practical invocation evaluation metric called AttenHScore, which calculates the accumulation and propagation of hallucinations during the generation process of small LMs, continuously amplifying potential reasoning errors. By dynamically adjusting the detection threshold, we achieve more accurate real-time invocation of large LMs. Additionally, considering the limited reasoning capacity of small LMs, we leverage uncertainty-aware knowledge reorganization to assist them better capture critical information from different text chunks. Extensive experiments reveal that our AttenHScore outperforms most baseline in enhancing real-time hallucination detection capabilities across multiple QA datasets, especially when addressing complex queries. Moreover, our strategies eliminate the need for additional model training and display flexibility in adapting to various transformer-based LMs. 

---
# Generative Sign-description Prompts with Multi-positive Contrastive Learning for Sign Language Recognition 

**Authors**: Siyu Liang, Yunan Li, Wentian Xin, Huizhou Chen, Xujie Liu, Kang Liu, Qiguang Miao  

**Link**: [PDF](https://arxiv.org/pdf/2505.02304)  

**Abstract**: Sign language recognition (SLR) faces fundamental challenges in creating accurate annotations due to the inherent complexity of simultaneous manual and non-manual signals. To the best of our knowledge, this is the first work to integrate generative large language models (LLMs) into SLR tasks. We propose a novel Generative Sign-description Prompts Multi-positive Contrastive learning (GSP-MC) method that leverages retrieval-augmented generation (RAG) with domain-specific LLMs, incorporating multi-step prompt engineering and expert-validated sign language corpora to produce precise multipart descriptions. The GSP-MC method also employs a dual-encoder architecture to bidirectionally align hierarchical skeleton features with multiple text descriptions (global, synonym, and part level) through probabilistic matching. Our approach combines global and part-level losses, optimizing KL divergence to ensure robust alignment across all relevant text-skeleton pairs while capturing both sign-level semantics and detailed part dynamics. Experiments demonstrate state-of-the-art performance against existing methods on the Chinese SLR500 (reaching 97.1%) and Turkish AUTSL datasets (97.07% accuracy). The method's cross-lingual effectiveness highlight its potential for developing inclusive communication technologies. 

---
# Demystifying optimized prompts in language models 

**Authors**: Rimon Melamed, Lucas H. McCabe, H. Howie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.02273)  

**Abstract**: Modern language models (LMs) are not robust to out-of-distribution inputs. Machine generated (``optimized'') prompts can be used to modulate LM outputs and induce specific behaviors while appearing completely uninterpretable. In this work, we investigate the composition of optimized prompts, as well as the mechanisms by which LMs parse and build predictions from optimized prompts. We find that optimized prompts primarily consist of punctuation and noun tokens which are more rare in the training data. Internally, optimized prompts are clearly distinguishable from natural language counterparts based on sparse subsets of the model's activations. Across various families of instruction-tuned models, optimized prompts follow a similar path in how their representations form through the network. 

---
# Parameter-Efficient Transformer Embeddings 

**Authors**: Henry Ndubuaku, Mouad Talhi  

**Link**: [PDF](https://arxiv.org/pdf/2505.02266)  

**Abstract**: Embedding layers in transformer-based NLP models typically account for the largest share of model parameters, scaling with vocabulary size but not yielding performance gains proportional to scale. We propose an alternative approach in which token embedding vectors are first generated deterministically, directly from the token IDs using a Fourier expansion of their normalized values, followed by a lightweight multilayer perceptron (MLP) that captures higher-order interactions. We train standard transformers and our architecture on natural language inference tasks (SNLI and MNLI), and evaluate zero-shot performance on sentence textual similarity (STS-B). Our results demonstrate that the proposed method achieves competitive performance using significantly fewer parameters, trains faster, and operates effectively without the need for dropout. This proof-of-concept study highlights the potential for scalable, memory-efficient language models and motivates further large-scale experimentation based on our findings. 

---
# Personalisation or Prejudice? Addressing Geographic Bias in Hate Speech Detection using Debias Tuning in Large Language Models 

**Authors**: Paloma Piot, Patricia Martín-Rodilla, Javier Parapar  

**Link**: [PDF](https://arxiv.org/pdf/2505.02252)  

**Abstract**: Commercial Large Language Models (LLMs) have recently incorporated memory features to deliver personalised responses. This memory retains details such as user demographics and individual characteristics, allowing LLMs to adjust their behaviour based on personal information. However, the impact of integrating personalised information into the context has not been thoroughly assessed, leading to questions about its influence on LLM behaviour. Personalisation can be challenging, particularly with sensitive topics. In this paper, we examine various state-of-the-art LLMs to understand their behaviour in different personalisation scenarios, specifically focusing on hate speech. We prompt the models to assume country-specific personas and use different languages for hate speech detection. Our findings reveal that context personalisation significantly influences LLMs' responses in this sensitive area. To mitigate these unwanted biases, we fine-tune the LLMs by penalising inconsistent hate speech classifications made with and without country or language-specific context. The refined models demonstrate improved performance in both personalised contexts and when no context is provided. 

---
# SEval-Ex: A Statement-Level Framework for Explainable Summarization Evaluation 

**Authors**: Tanguy Herserant, Vincent Guigue  

**Link**: [PDF](https://arxiv.org/pdf/2505.02235)  

**Abstract**: Evaluating text summarization quality remains a critical challenge in Natural Language Processing. Current approaches face a trade-off between performance and interpretability. We present SEval-Ex, a framework that bridges this gap by decomposing summarization evaluation into atomic statements, enabling both high performance and explainability. SEval-Ex employs a two-stage pipeline: first extracting atomic statements from text source and summary using LLM, then a matching between generated statements. Unlike existing approaches that provide only summary-level scores, our method generates detailed evidence for its decisions through statement-level alignments. Experiments on the SummEval benchmark demonstrate that SEval-Ex achieves state-of-the-art performance with 0.580 correlation on consistency with human consistency judgments, surpassing GPT-4 based evaluators (0.521) while maintaining interpretability. Finally, our framework shows robustness against hallucination. 

---
# Measuring Hong Kong Massive Multi-Task Language Understanding 

**Authors**: Chuxue Cao, Zhenghao Zhu, Junqi Zhu, Guoying Lu, Siyu Peng, Juntao Dai, Weijie Shi, Sirui Han, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.02177)  

**Abstract**: Multilingual understanding is crucial for the cross-cultural applicability of Large Language Models (LLMs). However, evaluation benchmarks designed for Hong Kong's unique linguistic landscape, which combines Traditional Chinese script with Cantonese as the spoken form and its cultural context, remain underdeveloped. To address this gap, we introduce HKMMLU, a multi-task language understanding benchmark that evaluates Hong Kong's linguistic competence and socio-cultural knowledge. The HKMMLU includes 26,698 multi-choice questions across 66 subjects, organized into four categories: Science, Technology, Engineering, and Mathematics (STEM), Social Sciences, Humanities, and Other. To evaluate the multilingual understanding ability of LLMs, 90,550 Mandarin-Cantonese translation tasks were additionally included. We conduct comprehensive experiments on GPT-4o, Claude 3.7 Sonnet, and 18 open-source LLMs of varying sizes on HKMMLU. The results show that the best-performing model, DeepSeek-V3, struggles to achieve an accuracy of 75\%, significantly lower than that of MMLU and CMMLU. This performance gap highlights the need to improve LLMs' capabilities in Hong Kong-specific language and knowledge domains. Furthermore, we investigate how question language, model size, prompting strategies, and question and reasoning token lengths affect model performance. We anticipate that HKMMLU will significantly advance the development of LLMs in multilingual and cross-cultural contexts, thereby enabling broader and more impactful applications. 

---
# Identifying Legal Holdings with LLMs: A Systematic Study of Performance, Scale, and Memorization 

**Authors**: Chuck Arvin  

**Link**: [PDF](https://arxiv.org/pdf/2505.02172)  

**Abstract**: As large language models (LLMs) continue to advance in capabilities, it is essential to assess how they perform on established benchmarks. In this study, we present a suite of experiments to assess the performance of modern LLMs (ranging from 3B to 90B+ parameters) on CaseHOLD, a legal benchmark dataset for identifying case holdings. Our experiments demonstrate ``scaling effects'' - performance on this task improves with model size, with more capable models like GPT4o and AmazonNovaPro achieving macro F1 scores of 0.744 and 0.720 respectively. These scores are competitive with the best published results on this dataset, and do not require any technically sophisticated model training, fine-tuning or few-shot prompting. To ensure that these strong results are not due to memorization of judicial opinions contained in the training data, we develop and utilize a novel citation anonymization test that preserves semantic meaning while ensuring case names and citations are fictitious. Models maintain strong performance under these conditions (macro F1 of 0.728), suggesting the performance is not due to rote memorization. These findings demonstrate both the promise and current limitations of LLMs for legal tasks with important implications for the development and measurement of automated legal analytics and legal benchmarks. 

---
# A New HOPE: Domain-agnostic Automatic Evaluation of Text Chunking 

**Authors**: Henrik Brådland, Morten Goodwin, Per-Arne Andersen, Alexander S. Nossum, Aditya Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2505.02171)  

**Abstract**: Document chunking fundamentally impacts Retrieval-Augmented Generation (RAG) by determining how source materials are segmented before indexing. Despite evidence that Large Language Models (LLMs) are sensitive to the layout and structure of retrieved data, there is currently no framework to analyze the impact of different chunking methods. In this paper, we introduce a novel methodology that defines essential characteristics of the chunking process at three levels: intrinsic passage properties, extrinsic passage properties, and passages-document coherence. We propose HOPE (Holistic Passage Evaluation), a domain-agnostic, automatic evaluation metric that quantifies and aggregates these characteristics. Our empirical evaluations across seven domains demonstrate that the HOPE metric correlates significantly (p > 0.13) with various RAG performance indicators, revealing contrasts between the importance of extrinsic and intrinsic properties of passages. Semantic independence between passages proves essential for system performance with a performance gain of up to 56.2% in factual correctness and 21.1% in answer correctness. On the contrary, traditional assumptions about maintaining concept unity within passages show minimal impact. These findings provide actionable insights for optimizing chunking strategies, thus improving RAG system design to produce more factually correct responses. 

---
# Incorporating Legal Structure in Retrieval-Augmented Generation: A Case Study on Copyright Fair Use 

**Authors**: Justin Ho, Alexandra Colby, William Fisher  

**Link**: [PDF](https://arxiv.org/pdf/2505.02164)  

**Abstract**: This paper presents a domain-specific implementation of Retrieval-Augmented Generation (RAG) tailored to the Fair Use Doctrine in U.S. copyright law. Motivated by the increasing prevalence of DMCA takedowns and the lack of accessible legal support for content creators, we propose a structured approach that combines semantic search with legal knowledge graphs and court citation networks to improve retrieval quality and reasoning reliability. Our prototype models legal precedents at the statutory factor level (e.g., purpose, nature, amount, market effect) and incorporates citation-weighted graph representations to prioritize doctrinally authoritative sources. We use Chain-of-Thought reasoning and interleaved retrieval steps to better emulate legal reasoning. Preliminary testing suggests this method improves doctrinal relevance in the retrieval process, laying groundwork for future evaluation and deployment of LLM-based legal assistance tools. 

---
# Think on your Feet: Adaptive Thinking via Reinforcement Learning for Social Agents 

**Authors**: Minzheng Wang, Yongbin Li, Haobo Wang, Xinghua Zhang, Nan Xu, Bingli Wu, Fei Huang, Haiyang Yu, Wenji Mao  

**Link**: [PDF](https://arxiv.org/pdf/2505.02156)  

**Abstract**: Effective social intelligence simulation requires language agents to dynamically adjust reasoning depth, a capability notably absent in current approaches. While existing methods either lack this kind of reasoning capability or enforce uniform long chain-of-thought reasoning across all scenarios, resulting in excessive token usage and inappropriate social simulation. In this paper, we propose $\textbf{A}$daptive $\textbf{M}$ode $\textbf{L}$earning ($\textbf{AML}$) that strategically selects from four thinking modes (intuitive reaction $\rightarrow$ deep contemplation) based on real-time context. Our framework's core innovation, the $\textbf{A}$daptive $\textbf{M}$ode $\textbf{P}$olicy $\textbf{O}$ptimization ($\textbf{AMPO}$) algorithm, introduces three key advancements over existing methods: (1) Multi-granular thinking mode design, (2) Context-aware mode switching across social interaction, and (3) Token-efficient reasoning via depth-adaptive processing. Extensive experiments on social intelligence tasks confirm that AML achieves 15.6% higher task performance than state-of-the-art methods. Notably, our method outperforms GRPO by 7.0% with 32.8% shorter reasoning chains. These results demonstrate that context-sensitive thinking mode selection, as implemented in AMPO, enables more human-like adaptive reasoning than GRPO's fixed-depth approach 

---
# QiMeng-Xpiler: Transcompiling Tensor Programs for Deep Learning Systems with a Neural-Symbolic Approach 

**Authors**: Shouyang Dong, Yuanbo Wen, Jun Bi, Di Huang, Jiaming Guo, Jianxing Xu, Ruibai Xu, Xinkai Song, Yifan Hao, Xuehai Zhou, Tianshi Chen, Qi Guo, Yunji Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.02146)  

**Abstract**: Heterogeneous deep learning systems (DLS) such as GPUs and ASICs have been widely deployed in industrial data centers, which requires to develop multiple low-level tensor programs for different platforms. An attractive solution to relieve the programming burden is to transcompile the legacy code of one platform to others. However, current transcompilation techniques struggle with either tremendous manual efforts or functional incorrectness, rendering "Write Once, Run Anywhere" of tensor programs an open question.
We propose a novel transcompiler, i.e., QiMeng-Xpiler, for automatically translating tensor programs across DLS via both large language models (LLMs) and symbolic program synthesis, i.e., neural-symbolic synthesis. The key insight is leveraging the powerful code generation ability of LLM to make costly search-based symbolic synthesis computationally tractable. Concretely, we propose multiple LLM-assisted compilation passes via pre-defined meta-prompts for program transformation. During each program transformation, efficient symbolic program synthesis is employed to repair incorrect code snippets with a limited scale. To attain high performance, we propose a hierarchical auto-tuning approach to systematically explore both the parameters and sequences of transformation passes. Experiments on 4 DLS with distinct programming interfaces, i.e., Intel DL Boost with VNNI, NVIDIA GPU with CUDA, AMD MI with HIP, and Cambricon MLU with BANG, demonstrate that QiMeng-Xpiler correctly translates different tensor programs at the accuracy of 95% on average, and the performance of translated programs achieves up to 2.0x over vendor-provided manually-optimized libraries. As a result, the programming productivity of DLS is improved by up to 96.0x via transcompiling legacy tensor programs. 

---
# Exploring the Potential of Offline RL for Reasoning in LLMs: A Preliminary Study 

**Authors**: Xiaoyu Tian, Sitong Zhao, Haotian Wang, Shuaiting Chen, Yiping Peng, Yunjie Ji, Han Zhao, Xiangang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.02142)  

**Abstract**: Despite significant advances in long-context reasoning by large language models (LLMs), primarily through Online Reinforcement Learning (RL) methods, these approaches incur substantial computational costs and complexity. In contrast, simpler and more economical Offline RL methods remain underexplored. To address this gap, we investigate the effectiveness of Offline RL methods, specifically Direct Preference Optimization (DPO) and its length-desensitized variant LD-DPO, in enhancing the reasoning capabilities of LLMs. Extensive experiments across multiple reasoning benchmarks demonstrate that these simpler Offline RL methods substantially improve model performance, achieving an average enhancement of 3.3\%, with a particularly notable increase of 10.1\% on the challenging Arena-Hard benchmark. Furthermore, we analyze DPO's sensitivity to output length, emphasizing that increasing reasoning length should align with semantic richness, as indiscriminate lengthening may adversely affect model performance. We provide comprehensive descriptions of our data processing and training methodologies, offering empirical evidence and practical insights for developing more cost-effective Offline RL approaches. 

---
# LLM-OptiRA: LLM-Driven Optimization of Resource Allocation for Non-Convex Problems in Wireless Communications 

**Authors**: Xinyue Peng, Yanming Liu, Yihan Cang, Chaoqun Cao, Ming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.02091)  

**Abstract**: Solving non-convex resource allocation problems poses significant challenges in wireless communication systems, often beyond the capability of traditional optimization techniques. To address this issue, we propose LLM-OptiRA, the first framework that leverages large language models (LLMs) to automatically detect and transform non-convex components into solvable forms, enabling fully automated resolution of non-convex resource allocation problems in wireless communication systems. LLM-OptiRA not only simplifies problem-solving by reducing reliance on expert knowledge, but also integrates error correction and feasibility validation mechanisms to ensure robustness. Experimental results show that LLM-OptiRA achieves an execution rate of 96% and a success rate of 80% on GPT-4, significantly outperforming baseline approaches in complex optimization tasks across diverse scenarios. 

---
# LecEval: An Automated Metric for Multimodal Knowledge Acquisition in Multimedia Learning 

**Authors**: Joy Lim Jia Yin, Daniel Zhang-Li, Jifan Yu, Haoxuan Li, Shangqing Tu, Yuanchun Wang, Zhiyuan Liu, Huiqin Liu, Lei Hou, Juanzi Li, Bin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.02078)  

**Abstract**: Evaluating the quality of slide-based multimedia instruction is challenging. Existing methods like manual assessment, reference-based metrics, and large language model evaluators face limitations in scalability, context capture, or bias. In this paper, we introduce LecEval, an automated metric grounded in Mayer's Cognitive Theory of Multimedia Learning, to evaluate multimodal knowledge acquisition in slide-based learning. LecEval assesses effectiveness using four rubrics: Content Relevance (CR), Expressive Clarity (EC), Logical Structure (LS), and Audience Engagement (AE). We curate a large-scale dataset of over 2,000 slides from more than 50 online course videos, annotated with fine-grained human ratings across these rubrics. A model trained on this dataset demonstrates superior accuracy and adaptability compared to existing metrics, bridging the gap between automated and human assessments. We release our dataset and toolkits at this https URL. 

---
# What do Language Model Probabilities Represent? From Distribution Estimation to Response Prediction 

**Authors**: Eitan Wagner, Omri Abend  

**Link**: [PDF](https://arxiv.org/pdf/2505.02072)  

**Abstract**: The notion of language modeling has gradually shifted in recent years from a distribution over finite-length strings to general-purpose prediction models for textual inputs and outputs, following appropriate alignment phases. This paper analyzes the distinction between distribution estimation and response prediction in the context of LLMs, and their often conflicting goals. We examine the training phases of LLMs, which include pretraining, in-context learning, and preference tuning, and also the common use cases for their output probabilities, which include completion probabilities and explicit probabilities as output. We argue that the different settings lead to three distinct intended output distributions. We demonstrate that NLP works often assume that these distributions should be similar, which leads to misinterpretations of their experimental findings. Our work sets firmer formal foundations for the interpretation of LLMs, which will inform ongoing work on the interpretation and use of LLMs' induced distributions. 

---
# An overview of artificial intelligence in computer-assisted language learning 

**Authors**: Anisia Katinskaia  

**Link**: [PDF](https://arxiv.org/pdf/2505.02032)  

**Abstract**: Computer-assisted language learning -- CALL -- is an established research field. We review how artificial intelligence can be applied to support language learning and teaching. The need for intelligent agents that assist language learners and teachers is increasing: the human teacher's time is a scarce and costly resource, which does not scale with growing demand. Further factors contribute to the need for CALL: pandemics and increasing demand for distance learning, migration of large populations, the need for sustainable and affordable support for learning, etc. CALL systems are made up of many components that perform various functions, and AI is applied to many different aspects in CALL, corresponding to their own expansive research areas. Most of what we find in the research literature and in practical use are prototypes or partial implementations -- systems that perform some aspects of the overall desired functionality. Complete solutions -- most of them commercial -- are few, because they require massive resources. Recent advances in AI should result in improvements in CALL, yet there is a lack of surveys that focus on AI in the context of this research field. This paper aims to present a perspective on the AI methods that can be employed for language learning from a position of a developer of a CALL system. We also aim to connect work from different disciplines, to build bridges for interdisciplinary work. 

---
# Towards Safer Pretraining: Analyzing and Filtering Harmful Content in Webscale datasets for Responsible LLMs 

**Authors**: Sai Krishna Mendu, Harish Yenala, Aditi Gulati, Shanu Kumar, Parag Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2505.02009)  

**Abstract**: Large language models (LLMs) have become integral to various real-world applications, leveraging massive, web-sourced datasets like Common Crawl, C4, and FineWeb for pretraining. While these datasets provide linguistic data essential for high-quality natural language generation, they often contain harmful content, such as hate speech, misinformation, and biased narratives. Training LLMs on such unfiltered data risks perpetuating toxic behaviors, spreading misinformation, and amplifying societal biases which can undermine trust in LLM-driven applications and raise ethical concerns about their use. This paper presents a large-scale analysis of inappropriate content across these datasets, offering a comprehensive taxonomy that categorizes harmful webpages into Topical and Toxic based on their intent. We also introduce a prompt evaluation dataset, a high-accuracy Topical and Toxic Prompt (TTP), and a transformer-based model (HarmFormer) for content filtering. Additionally, we create a new multi-harm open-ended toxicity benchmark (HAVOC) and provide crucial insights into how models respond to adversarial toxic inputs. Upon publishing, we will also opensource our model signal on the entire C4 dataset. Our work offers insights into ensuring safer LLM pretraining and serves as a resource for Responsible AI (RAI) compliance. 

---
# LLM-based Text Simplification and its Effect on User Comprehension and Cognitive Load 

**Authors**: Theo Guidroz, Diego Ardila, Jimmy Li, Adam Mansour, Paul Jhun, Nina Gonzalez, Xiang Ji, Mike Sanchez, Sujay Kakarmath, Mathias MJ Bellaiche, Miguel Ángel Garrido, Faruk Ahmed, Divyansh Choudhary, Jay Hartford, Chenwei Xu, Henry Javier Serrano Echeverria, Yifan Wang, Jeff Shaffer, Eric, Yossi Matias, Avinatan Hassidim, Dale R Webster, Yun Liu, Sho Fujiwara, Peggy Bui, Quang Duong  

**Link**: [PDF](https://arxiv.org/pdf/2505.01980)  

**Abstract**: Information on the web, such as scientific publications and Wikipedia, often surpasses users' reading level. To help address this, we used a self-refinement approach to develop a LLM capability for minimally lossy text simplification. To validate our approach, we conducted a randomized study involving 4563 participants and 31 texts spanning 6 broad subject areas: PubMed (biomedical scientific articles), biology, law, finance, literature/philosophy, and aerospace/computer science. Participants were randomized to viewing original or simplified texts in a subject area, and answered multiple-choice questions (MCQs) that tested their comprehension of the text. The participants were also asked to provide qualitative feedback such as task difficulty. Our results indicate that participants who read the simplified text answered more MCQs correctly than their counterparts who read the original text (3.9% absolute increase, p<0.05). This gain was most striking with PubMed (14.6%), while more moderate gains were observed for finance (5.5%), aerospace/computer science (3.8%) domains, and legal (3.5%). Notably, the results were robust to whether participants could refer back to the text while answering MCQs. The absolute accuracy decreased by up to ~9% for both original and simplified setups where participants could not refer back to the text, but the ~4% overall improvement persisted. Finally, participants' self-reported perceived ease based on a simplified NASA Task Load Index was greater for those who read the simplified text (absolute change on a 5-point scale 0.33, p<0.05). This randomized study, involving an order of magnitude more participants than prior works, demonstrates the potential of LLMs to make complex information easier to understand. Our work aims to enable a broader audience to better learn and make use of expert knowledge available on the web, improving information accessibility. 

---
# Analyzing Cognitive Differences Among Large Language Models through the Lens of Social Worldview 

**Authors**: Jiatao Li, Yanheng Li, Xiaojun Wan  

**Link**: [PDF](https://arxiv.org/pdf/2505.01967)  

**Abstract**: Large Language Models (LLMs) have become integral to daily life, widely adopted in communication, decision-making, and information retrieval, raising critical questions about how these systems implicitly form and express socio-cognitive attitudes or "worldviews". While existing research extensively addresses demographic and ethical biases, broader dimensions-such as attitudes toward authority, equality, autonomy, and fate-remain under-explored. In this paper, we introduce the Social Worldview Taxonomy (SWT), a structured framework grounded in Cultural Theory, operationalizing four canonical worldviews (Hierarchy, Egalitarianism, Individualism, Fatalism) into measurable sub-dimensions. Using SWT, we empirically identify distinct and interpretable cognitive profiles across 28 diverse LLMs. Further, inspired by Social Referencing Theory, we experimentally demonstrate that explicit social cues systematically shape these cognitive attitudes, revealing both general response patterns and nuanced model-specific variations. Our findings enhance the interpretability of LLMs by revealing implicit socio-cognitive biases and their responsiveness to social feedback, thus guiding the development of more transparent and socially responsible language technologies. 

---
# CAMOUFLAGE: Exploiting Misinformation Detection Systems Through LLM-driven Adversarial Claim Transformation 

**Authors**: Mazal Bethany, Nishant Vishwamitra, Cho-Yu Jason Chiang, Peyman Najafirad  

**Link**: [PDF](https://arxiv.org/pdf/2505.01900)  

**Abstract**: Automated evidence-based misinformation detection systems, which evaluate the veracity of short claims against evidence, lack comprehensive analysis of their adversarial vulnerabilities. Existing black-box text-based adversarial attacks are ill-suited for evidence-based misinformation detection systems, as these attacks primarily focus on token-level substitutions involving gradient or logit-based optimization strategies, which are incapable of fooling the multi-component nature of these detection systems. These systems incorporate both retrieval and claim-evidence comparison modules, which requires attacks to break the retrieval of evidence and/or the comparison module so that it draws incorrect inferences. We present CAMOUFLAGE, an iterative, LLM-driven approach that employs a two-agent system, a Prompt Optimization Agent and an Attacker Agent, to create adversarial claim rewritings that manipulate evidence retrieval and mislead claim-evidence comparison, effectively bypassing the system without altering the meaning of the claim. The Attacker Agent produces semantically equivalent rewrites that attempt to mislead detectors, while the Prompt Optimization Agent analyzes failed attack attempts and refines the prompt of the Attacker to guide subsequent rewrites. This enables larger structural and stylistic transformations of the text rather than token-level substitutions, adapting the magnitude of changes based on previous outcomes. Unlike existing approaches, CAMOUFLAGE optimizes its attack solely based on binary model decisions to guide its rewriting process, eliminating the need for classifier logits or extensive querying. We evaluate CAMOUFLAGE on four systems, including two recent academic systems and two real-world APIs, with an average attack success rate of 46.92\% while preserving textual coherence and semantic equivalence to the original claims. 

---
# Automated Sentiment Classification and Topic Discovery in Large-Scale Social Media Streams 

**Authors**: Yiwen Lu, Siheng Xiong, Zhaowei Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.01883)  

**Abstract**: We present a framework for large-scale sentiment and topic analysis of Twitter discourse. Our pipeline begins with targeted data collection using conflict-specific keywords, followed by automated sentiment labeling via multiple pre-trained models to improve annotation robustness. We examine the relationship between sentiment and contextual features such as timestamp, geolocation, and lexical content. To identify latent themes, we apply Latent Dirichlet Allocation (LDA) on partitioned subsets grouped by sentiment and metadata attributes. Finally, we develop an interactive visualization interface to support exploration of sentiment trends and topic distributions across time and regions. This work contributes a scalable methodology for social media analysis in dynamic geopolitical contexts. 

---
# Humans can learn to detect AI-generated texts, or at least learn when they can't 

**Authors**: Jiří Milička, Anna Marklová, Ondřej Drobil, Eva Pospíšilová  

**Link**: [PDF](https://arxiv.org/pdf/2505.01877)  

**Abstract**: This study investigates whether individuals can learn to accurately discriminate between human-written and AI-produced texts when provided with immediate feedback, and if they can use this feedback to recalibrate their self-perceived competence. We also explore the specific criteria individuals rely upon when making these decisions, focusing on textual style and perceived readability.
We used GPT-4o to generate several hundred texts across various genres and text types comparable to Koditex, a multi-register corpus of human-written texts. We then presented randomized text pairs to 255 Czech native speakers who identified which text was human-written and which was AI-generated. Participants were randomly assigned to two conditions: one receiving immediate feedback after each trial, the other receiving no feedback until experiment completion. We recorded accuracy in identification, confidence levels, response times, and judgments about text readability along with demographic data and participants' engagement with AI technologies prior to the experiment.
Participants receiving immediate feedback showed significant improvement in accuracy and confidence calibration. Participants initially held incorrect assumptions about AI-generated text features, including expectations about stylistic rigidity and readability. Notably, without feedback, participants made the most errors precisely when feeling most confident -- an issue largely resolved among the feedback group.
The ability to differentiate between human and AI-generated texts can be effectively learned through targeted training with explicit feedback, which helps correct misconceptions about AI stylistic features and readability, as well as potential other variables that were not explored, while facilitating more accurate self-assessment. This finding might be particularly important in educational contexts. 

---
# Positional Attention for Efficient BERT-Based Named Entity Recognition 

**Authors**: Mo Sun, Siheng Xiong, Yuankai Cai, Bowen Zuo  

**Link**: [PDF](https://arxiv.org/pdf/2505.01868)  

**Abstract**: This paper presents a framework for Named Entity Recognition (NER) leveraging the Bidirectional Encoder Representations from Transformers (BERT) model in natural language processing (NLP). NER is a fundamental task in NLP with broad applicability across downstream applications. While BERT has established itself as a state-of-the-art model for entity recognition, fine-tuning it from scratch for each new application is computationally expensive and time-consuming. To address this, we propose a cost-efficient approach that integrates positional attention mechanisms into the entity recognition process and enables effective customization using pre-trained parameters. The framework is evaluated on a Kaggle dataset derived from the Groningen Meaning Bank corpus and achieves strong performance with fewer training epochs. This work contributes to the field by offering a practical solution for reducing the training cost of BERT-based NER systems while maintaining high accuracy. 

---
# Intra-Layer Recurrence in Transformers for Language Modeling 

**Authors**: Anthony Nguyen, Wenjun Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.01855)  

**Abstract**: Transformer models have established new benchmarks in natural language processing; however, their increasing depth results in substantial growth in parameter counts. While existing recurrent transformer methods address this issue by reprocessing layers multiple times, they often apply recurrence indiscriminately across entire blocks of layers. In this work, we investigate Intra-Layer Recurrence (ILR), a more targeted approach that applies recurrence selectively to individual layers within a single forward pass. Our experiments show that allocating more iterations to earlier layers yields optimal results. These findings suggest that ILR offers a promising direction for optimizing recurrent structures in transformer architectures. 

---
# $\textit{New News}$: System-2 Fine-tuning for Robust Integration of New Knowledge 

**Authors**: Core Francisco Park, Zechen Zhang, Hidenori Tanaka  

**Link**: [PDF](https://arxiv.org/pdf/2505.01812)  

**Abstract**: Humans and intelligent animals can effortlessly internalize new information ("news") and accurately extract the implications for performing downstream tasks. While large language models (LLMs) can achieve this through in-context learning (ICL) when the news is explicitly given as context, fine-tuning remains challenging for the models to consolidate learning in weights. In this paper, we introduce $\textit{New News}$, a dataset composed of hypothetical yet plausible news spanning multiple domains (mathematics, coding, discoveries, leaderboards, events), accompanied by downstream evaluation questions whose correct answers critically depend on understanding and internalizing the news. We first demonstrate a substantial gap between naive fine-tuning and in-context learning (FT-ICL gap) on our news dataset. To address this gap, we explore a suite of self-play data generation protocols -- paraphrases, implications and Self-QAs -- designed to distill the knowledge from the model with context into the weights of the model without the context, which we term $\textit{System-2 Fine-tuning}$ (Sys2-FT). We systematically evaluate ICL and Sys2-FT performance across data domains and model scales with the Qwen 2.5 family of models. Our results demonstrate that the self-QA protocol of Sys2-FT significantly improves models' in-weight learning of the news. Furthermore, we discover the $\textit{contexual shadowing effect}$, where training with the news $\textit{in context}$ followed by its rephrases or QAs degrade learning of the news. Finally, we show preliminary evidence of an emerging scaling law of Sys2-FT. 

---
# Distinguishing AI-Generated and Human-Written Text Through Psycholinguistic Analysis 

**Authors**: Chidimma Opara  

**Link**: [PDF](https://arxiv.org/pdf/2505.01800)  

**Abstract**: The increasing sophistication of AI-generated texts highlights the urgent need for accurate and transparent detection tools, especially in educational settings, where verifying authorship is essential. Existing literature has demonstrated that the application of stylometric features with machine learning classifiers can yield excellent results. Building on this foundation, this study proposes a comprehensive framework that integrates stylometric analysis with psycholinguistic theories, offering a clear and interpretable approach to distinguishing between AI-generated and human-written texts. This research specifically maps 31 distinct stylometric features to cognitive processes such as lexical retrieval, discourse planning, cognitive load management, and metacognitive self-monitoring. In doing so, it highlights the unique psycholinguistic patterns found in human writing. Through the intersection of computational linguistics and cognitive science, this framework contributes to the development of reliable tools aimed at preserving academic integrity in the era of generative AI. 

---
# A Multimodal Framework for Explainable Evaluation of Soft Skills in Educational Environments 

**Authors**: Jared D.T. Guerrero-Sosa, Francisco P. Romero, Víctor Hugo Menéndez-Domínguez, Jesus Serrano-Guerrero, Andres Montoro-Montarroso, Jose A. Olivas  

**Link**: [PDF](https://arxiv.org/pdf/2505.01794)  

**Abstract**: In the rapidly evolving educational landscape, the unbiased assessment of soft skills is a significant challenge, particularly in higher education. This paper presents a fuzzy logic approach that employs a Granular Linguistic Model of Phenomena integrated with multimodal analysis to evaluate soft skills in undergraduate students. By leveraging computational perceptions, this approach enables a structured breakdown of complex soft skill expressions, capturing nuanced behaviours with high granularity and addressing their inherent uncertainties, thereby enhancing interpretability and reliability. Experiments were conducted with undergraduate students using a developed tool that assesses soft skills such as decision-making, communication, and creativity. This tool identifies and quantifies subtle aspects of human interaction, such as facial expressions and gesture recognition. The findings reveal that the framework effectively consolidates multiple data inputs to produce meaningful and consistent assessments of soft skills, showing that integrating multiple modalities into the evaluation process significantly improves the quality of soft skills scores, making the assessment work transparent and understandable to educational stakeholders. 

---
# Same evaluation, more tokens: On the effect of input length for machine translation evaluation using Large Language Models 

**Authors**: Tobias Domhan, Dawei Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.01761)  

**Abstract**: Accurately evaluating machine-translated text remains a long-standing challenge, particularly for long documents. Recent work has shown that large language models (LLMs) can serve as reliable and interpretable sentence-level translation evaluators via MQM error span annotations. With modern LLMs supporting larger context windows, a natural question arises: can we feed entire document translations into an LLM for quality assessment? Ideally, evaluation should be invariant to text length, producing consistent error spans regardless of input granularity. However, our analysis shows that text length significantly impacts evaluation: longer texts lead to fewer error spans and reduced system ranking accuracy. To address this limitation, we evaluate several strategies, including granularity-aligned prompting, Focus Sentence Prompting (FSP), and a fine-tuning approach to better align LLMs with the evaluation task. The latter two methods largely mitigate this length bias, making LLMs more reliable for long-form translation evaluation. 

---
# Efficient Shapley Value-based Non-Uniform Pruning of Large Language Models 

**Authors**: Chuan Sun, Han Yu, Lizhen Cui  

**Link**: [PDF](https://arxiv.org/pdf/2505.01731)  

**Abstract**: Pruning large language models (LLMs) is a promising solution for reducing model sizes and computational complexity while preserving performance. Traditional layer-wise pruning methods often adopt a uniform sparsity approach across all layers, which leads to suboptimal performance due to the varying significance of individual transformer layers within the model not being accounted for. To this end, we propose the \underline{S}hapley \underline{V}alue-based \underline{N}on-\underline{U}niform \underline{P}runing (\methodname{}) method for LLMs. This approach quantifies the contribution of each transformer layer to the overall model performance, enabling the assignment of tailored pruning budgets to different layers to retain critical parameters. To further improve efficiency, we design the Sliding Window-based Shapley Value approximation method. It substantially reduces computational overhead compared to exact SV calculation methods. Extensive experiments on various LLMs including LLaMA-v1, LLaMA-v2 and OPT demonstrate the effectiveness of the proposed approach. The results reveal that non-uniform pruning significantly enhances the performance of pruned models. Notably, \methodname{} achieves a reduction in perplexity (PPL) of 18.01\% and 19.55\% on LLaMA-7B and LLaMA-13B, respectively, compared to SparseGPT at 70\% sparsity. 

---
# High-Fidelity Pseudo-label Generation by Large Language Models for Training Robust Radiology Report Classifiers 

**Authors**: Brian Wong, Kaito Tanaka  

**Link**: [PDF](https://arxiv.org/pdf/2505.01693)  

**Abstract**: Automated labeling of chest X-ray reports is essential for enabling downstream tasks such as training image-based diagnostic models, population health studies, and clinical decision support. However, the high variability, complexity, and prevalence of negation and uncertainty in these free-text reports pose significant challenges for traditional Natural Language Processing methods. While large language models (LLMs) demonstrate strong text understanding, their direct application for large-scale, efficient labeling is limited by computational cost and speed. This paper introduces DeBERTa-RAD, a novel two-stage framework that combines the power of state-of-the-art LLM pseudo-labeling with efficient DeBERTa-based knowledge distillation for accurate and fast chest X-ray report labeling. We leverage an advanced LLM to generate high-quality pseudo-labels, including certainty statuses, for a large corpus of reports. Subsequently, a DeBERTa-Base model is trained on this pseudo-labeled data using a tailored knowledge distillation strategy. Evaluated on the expert-annotated MIMIC-500 benchmark, DeBERTa-RAD achieves a state-of-the-art Macro F1 score of 0.9120, significantly outperforming established rule-based systems, fine-tuned transformer models, and direct LLM inference, while maintaining a practical inference speed suitable for high-throughput applications. Our analysis shows particular strength in handling uncertain findings. This work demonstrates a promising path to overcome data annotation bottlenecks and achieve high-performance medical text processing through the strategic combination of LLM capabilities and efficient student models trained via distillation. 

---
# A Survey on Inference Engines for Large Language Models: Perspectives on Optimization and Efficiency 

**Authors**: Sihyeong Park, Sungryeol Jeon, Chaelyn Lee, Seokhun Jeon, Byung-Soo Kim, Jemin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.01658)  

**Abstract**: Large language models (LLMs) are widely applied in chatbots, code generators, and search engines. Workloads such as chain-of-thought, complex reasoning, and agent services significantly increase the inference cost by invoking the model repeatedly. Optimization methods such as parallelism, compression, and caching have been adopted to reduce costs, but the diverse service requirements make it hard to select the right method. Recently, specialized LLM inference engines have emerged as a key component for integrating the optimization methods into service-oriented infrastructures. However, a systematic study on inference engines is still lacking. This paper provides a comprehensive evaluation of 25 open-source and commercial inference engines. We examine each inference engine in terms of ease-of-use, ease-of-deployment, general-purpose support, scalability, and suitability for throughput- and latency-aware computation. Furthermore, we explore the design goals of each inference engine by investigating the optimization techniques it supports. In addition, we assess the ecosystem maturity of open source inference engines and handle the performance and cost policy of commercial solutions. We outline future research directions that include support for complex LLM-based services, support of various hardware, and enhanced security, offering practical guidance to researchers and developers in selecting and designing optimized LLM inference engines. We also provide a public repository to continually track developments in this fast-evolving field: this https URL 

---
# Always Tell Me The Odds: Fine-grained Conditional Probability Estimation 

**Authors**: Liaoyaqi Wang, Zhengping Jiang, Anqi Liu, Benjamin Van Durme  

**Link**: [PDF](https://arxiv.org/pdf/2505.01595)  

**Abstract**: We present a state-of-the-art model for fine-grained probability estimation of propositions conditioned on context. Recent advances in large language models (LLMs) have significantly enhanced their reasoning capabilities, particularly on well-defined tasks with complete information. However, LLMs continue to struggle with making accurate and well-calibrated probabilistic predictions under uncertainty or partial information. While incorporating uncertainty into model predictions often boosts performance, obtaining reliable estimates of that uncertainty remains understudied. In particular, LLM probability estimates tend to be coarse and biased towards more frequent numbers. Through a combination of human and synthetic data creation and assessment, scaling to larger models, and better supervision, we propose a set of strong and precise probability estimation models. We conduct systematic evaluations across tasks that rely on conditional probability estimation and show that our approach consistently outperforms existing fine-tuned and prompting-based methods by a large margin. 

---
# PIPA: A Unified Evaluation Protocol for Diagnosing Interactive Planning Agents 

**Authors**: Takyoung Kim, Janvijay Singh, Shuhaib Mehri, Emre Can Acikgoz, Sagnik Mukherjee, Nimet Beyza Bozdag, Sumuk Shashidhar, Gokhan Tur, Dilek Hakkani-Tür  

**Link**: [PDF](https://arxiv.org/pdf/2505.01592)  

**Abstract**: The growing capabilities of large language models (LLMs) in instruction-following and context-understanding lead to the era of agents with numerous applications. Among these, task planning agents have become especially prominent in realistic scenarios involving complex internal pipelines, such as context understanding, tool management, and response generation. However, existing benchmarks predominantly evaluate agent performance based on task completion as a proxy for overall effectiveness. We hypothesize that merely improving task completion is misaligned with maximizing user satisfaction, as users interact with the entire agentic process and not only the end result. To address this gap, we propose PIPA, a unified evaluation protocol that conceptualizes the behavioral process of interactive task planning agents within a partially observable Markov Decision Process (POMDP) paradigm. The proposed protocol offers a comprehensive assessment of agent performance through a set of atomic evaluation criteria, allowing researchers and practitioners to diagnose specific strengths and weaknesses within the agent's decision-making pipeline. Our analyses show that agents excel in different behavioral stages, with user satisfaction shaped by both outcomes and intermediate behaviors. We also highlight future directions, including systems that leverage multiple agents and the limitations of user simulators in task planning. 

---
# AI agents may be worth the hype but not the resources (yet): An initial exploration of machine translation quality and costs in three language pairs in the legal and news domains 

**Authors**: Vicent Briva Iglesias, Gokhan Dogru  

**Link**: [PDF](https://arxiv.org/pdf/2505.01560)  

**Abstract**: Large language models (LLMs) and multi-agent orchestration are touted as the next leap in machine translation (MT), but their benefits relative to conventional neural MT (NMT) remain unclear. This paper offers an empirical reality check. We benchmark five paradigms, Google Translate (strong NMT baseline), GPT-4o (general-purpose LLM), o1-preview (reasoning-enhanced LLM), and two GPT-4o-powered agentic workflows (sequential three-stage and iterative refinement), on test data drawn from a legal contract and news prose in three English-source pairs: Spanish, Catalan and Turkish. Automatic evaluation is performed with COMET, BLEU, chrF2 and TER; human evaluation is conducted with expert ratings of adequacy and fluency; efficiency with total input-plus-output token counts mapped to April 2025 pricing.
Automatic scores still favour the mature NMT system, which ranks first in seven of twelve metric-language combinations; o1-preview ties or places second in most remaining cases, while both multi-agent workflows trail. Human evaluation reverses part of this narrative: o1-preview produces the most adequate and fluent output in five of six comparisons, and the iterative agent edges ahead once, indicating that reasoning layers capture semantic nuance undervalued by surface metrics. Yet these qualitative gains carry steep costs. The sequential agent consumes roughly five times, and the iterative agent fifteen times, the tokens used by NMT or single-pass LLMs.
We advocate multidimensional, cost-aware evaluation protocols and highlight research directions that could tip the balance: leaner coordination strategies, selective agent activation, and hybrid pipelines combining single-pass LLMs with targeted agent intervention. 

---
# On the effectiveness of Large Language Models in the mechanical design domain 

**Authors**: Daniele Grandi, Fabian Riquelme  

**Link**: [PDF](https://arxiv.org/pdf/2505.01559)  

**Abstract**: In this work, we seek to understand the performance of large language models in the mechanical engineering domain. We leverage the semantic data found in the ABC dataset, specifically the assembly names that designers assigned to the overall assemblies, and the individual semantic part names that were assigned to each part. After pre-processing the data we developed two unsupervised tasks to evaluate how different model architectures perform on domain-specific data: a binary sentence-pair classification task and a zero-shot classification task. We achieved a 0.62 accuracy for the binary sentence-pair classification task with a fine-tuned model that focuses on fighting over-fitting: 1) modifying learning rates, 2) dropout values, 3) Sequence Length, and 4) adding a multi-head attention layer. Our model on the zero-shot classification task outperforms the baselines by a wide margin, and achieves a top-1 classification accuracy of 0.386. The results shed some light on the specific failure modes that arise when learning from language in this domain. 

---
# SymPlanner: Deliberate Planning in Language Models with Symbolic Representation 

**Authors**: Siheng Xiong, Jieyu Zhou, Zhangding Liu, Yusen Su  

**Link**: [PDF](https://arxiv.org/pdf/2505.01479)  

**Abstract**: Planning remains a core challenge for language models (LMs), particularly in domains that require coherent multi-step action sequences grounded in external constraints. We introduce SymPlanner, a novel framework that equips LMs with structured planning capabilities by interfacing them with a symbolic environment that serves as an explicit world model. Rather than relying purely on natural language reasoning, SymPlanner grounds the planning process in a symbolic state space, where a policy model proposes actions and a symbolic environment deterministically executes and verifies their effects. To enhance exploration and improve robustness, we introduce Iterative Correction (IC), which refines previously proposed actions by leveraging feedback from the symbolic environment to eliminate invalid decisions and guide the model toward valid alternatives. Additionally, Contrastive Ranking (CR) enables fine-grained comparison of candidate plans by evaluating them jointly. We evaluate SymPlanner on PlanBench, demonstrating that it produces more coherent, diverse, and verifiable plans than pure natural language baselines. 

---
# MoxE: Mixture of xLSTM Experts with Entropy-Aware Routing for Efficient Language Modeling 

**Authors**: Abdoul Majid O. Thiombiano, Brahim Hnich, Ali Ben Mrad, Mohamed Wiem Mkaouer  

**Link**: [PDF](https://arxiv.org/pdf/2505.01459)  

**Abstract**: This paper introduces MoxE, a novel architecture that synergistically combines the Extended Long Short-Term Memory (xLSTM) with the Mixture of Experts (MoE) framework to address critical scalability and efficiency challenges in large language models (LLMs). The proposed method effectively leverages xLSTM's innovative memory structures while strategically introducing sparsity through MoE to substantially reduce computational overhead. At the heart of our approach is a novel entropy-based routing mechanism, designed to dynamically route tokens to specialized experts, thereby ensuring efficient and balanced resource utilization. This entropy awareness enables the architecture to effectively manage both rare and common tokens, with mLSTM blocks being favored to handle rare tokens. To further enhance generalization, we introduce a suite of auxiliary losses, including entropy-based and group-wise balancing losses, ensuring robust performance and efficient training. Theoretical analysis and empirical evaluations rigorously demonstrate that MoxE achieves significant efficiency gains and enhanced effectiveness compared to existing approaches, marking a notable advancement in scalable LLM architectures. 

---
# Unlearning Sensitive Information in Multimodal LLMs: Benchmark and Attack-Defense Evaluation 

**Authors**: Vaidehi Patil, Yi-Lin Sung, Peter Hase, Jie Peng, Tianlong Chen, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2505.01456)  

**Abstract**: LLMs trained on massive datasets may inadvertently acquire sensitive information such as personal details and potentially harmful content. This risk is further heightened in multimodal LLMs as they integrate information from multiple modalities (image and text). Adversaries can exploit this knowledge through multimodal prompts to extract sensitive details. Evaluating how effectively MLLMs can forget such information (targeted unlearning) necessitates the creation of high-quality, well-annotated image-text pairs. While prior work on unlearning has focused on text, multimodal unlearning remains underexplored. To address this gap, we first introduce a multimodal unlearning benchmark, UnLOK-VQA (Unlearning Outside Knowledge VQA), as well as an attack-and-defense framework to evaluate methods for deleting specific multimodal knowledge from MLLMs. We extend a visual question-answering dataset using an automated pipeline that generates varying-proximity samples for testing generalization and specificity, followed by manual filtering for maintaining high quality. We then evaluate six defense objectives against seven attacks (four whitebox, three blackbox), including a novel whitebox method leveraging interpretability of hidden states. Our results show multimodal attacks outperform text- or image-only ones, and that the most effective defense removes answer information from internal model states. Additionally, larger models exhibit greater post-editing robustness, suggesting that scale enhances safety. UnLOK-VQA provides a rigorous benchmark for advancing unlearning in MLLMs. 

---
# R1-Reward: Training Multimodal Reward Model Through Stable Reinforcement Learning 

**Authors**: Yi-Fan Zhang, Xingyu Lu, Xiao Hu, Chaoyou Fu, Bin Wen, Tianke Zhang, Changyi Liu, Kaiyu Jiang, Kaibing Chen, Kaiyu Tang, Haojie Ding, Jiankang Chen, Fan Yang, Zhang Zhang, Tingting Gao, Liang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.02835)  

**Abstract**: Multimodal Reward Models (MRMs) play a crucial role in enhancing the performance of Multimodal Large Language Models (MLLMs). While recent advancements have primarily focused on improving the model structure and training data of MRMs, there has been limited exploration into the effectiveness of long-term reasoning capabilities for reward modeling and how to activate these capabilities in MRMs. In this paper, we explore how Reinforcement Learning (RL) can be used to improve reward modeling. Specifically, we reformulate the reward modeling problem as a rule-based RL task. However, we observe that directly applying existing RL algorithms, such as Reinforce++, to reward modeling often leads to training instability or even collapse due to the inherent limitations of these algorithms. To address this issue, we propose the StableReinforce algorithm, which refines the training loss, advantage estimation strategy, and reward design of existing RL methods. These refinements result in more stable training dynamics and superior performance. To facilitate MRM training, we collect 200K preference data from diverse datasets. Our reward model, R1-Reward, trained using the StableReinforce algorithm on this dataset, significantly improves performance on multimodal reward modeling benchmarks. Compared to previous SOTA models, R1-Reward achieves a $8.4\%$ improvement on the VL Reward-Bench and a $14.3\%$ improvement on the Multimodal Reward Bench. Moreover, with more inference compute, R1-Reward's performance is further enhanced, highlighting the potential of RL algorithms in optimizing MRMs. 

---
# AOR: Anatomical Ontology-Guided Reasoning for Medical Large Multimodal Model in Chest X-Ray Interpretation 

**Authors**: Qingqiu Li, Zihang Cui, Seongsu Bae, Jilan Xu, Runtian Yuan, Yuejie Zhang, Rui Feng, Quanli Shen, Xiaobo Zhang, Junjun He, Shujun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.02830)  

**Abstract**: Chest X-rays (CXRs) are the most frequently performed imaging examinations in clinical settings. Recent advancements in Large Multimodal Models (LMMs) have enabled automated CXR interpretation, enhancing diagnostic accuracy and efficiency. However, despite their strong visual understanding, current Medical LMMs (MLMMs) still face two major challenges: (1) Insufficient region-level understanding and interaction, and (2) Limited accuracy and interpretability due to single-step reasoning. In this paper, we empower MLMMs with anatomy-centric reasoning capabilities to enhance their interactivity and explainability. Specifically, we first propose an Anatomical Ontology-Guided Reasoning (AOR) framework, which centers on cross-modal region-level information to facilitate multi-step reasoning. Next, under the guidance of expert physicians, we develop AOR-Instruction, a large instruction dataset for MLMMs training. Our experiments demonstrate AOR's superior performance in both VQA and report generation tasks. 

---
# AutoLibra: Agent Metric Induction from Open-Ended Feedback 

**Authors**: Hao Zhu, Phil Cuvin, Xinkai Yu, Charlotte Ka Yee Yan, Jason Zhang, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.02820)  

**Abstract**: Agents are predominantly evaluated and optimized via task success metrics, which are coarse, rely on manual design from experts, and fail to reward intermediate emergent behaviors. We propose AutoLibra, a framework for agent evaluation, that transforms open-ended human feedback, e.g., "If you find that the button is disabled, don't click it again", or "This agent has too much autonomy to decide what to do on its own", into metrics for evaluating fine-grained behaviors in agent trajectories. AutoLibra accomplishes this by grounding feedback to an agent's behavior, clustering similar positive and negative behaviors, and creating concrete metrics with clear definitions and concrete examples, which can be used for prompting LLM-as-a-Judge as evaluators. We further propose two meta-metrics to evaluate the alignment of a set of (induced) metrics with open feedback: "coverage" and "redundancy". Through optimizing these meta-metrics, we experimentally demonstrate AutoLibra's ability to induce more concrete agent evaluation metrics than the ones proposed in previous agent evaluation benchmarks and discover new metrics to analyze agents. We also present two applications of AutoLibra in agent improvement: First, we show that AutoLibra-induced metrics serve as better prompt-engineering targets than the task success rate on a wide range of text game tasks, improving agent performance over baseline by a mean of 20%. Second, we show that AutoLibra can iteratively select high-quality fine-tuning data for web navigation agents. Our results suggest that AutoLibra is a powerful task-agnostic tool for evaluating and improving language agents. 

---
# Knowing You Don't Know: Learning When to Continue Search in Multi-round RAG through Self-Practicing 

**Authors**: Diji Yang, Linda Zeng, Jinmeng Rao, Yi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.02811)  

**Abstract**: Retrieval Augmented Generation (RAG) has shown strong capability in enhancing language models' knowledge and reducing AI generative hallucinations, driving its widespread use. However, complex tasks requiring multi-round retrieval remain challenging, and early attempts tend to be overly optimistic without a good sense of self-skepticism. Current multi-round RAG systems may continue searching even when enough information has already been retrieved, or they may provide incorrect answers without having sufficient information or knowledge. Existing solutions either require large amounts of expensive human-labeled process supervision data or lead to subpar performance.
This paper aims to address these limitations by introducing a new framework, \textbf{SIM-RAG}, to explicitly enhance RAG systems' self-awareness and multi-round retrieval capabilities. To train SIM-RAG, we first let a RAG system self-practice multi-round retrieval, augmenting existing question-answer pairs with intermediate inner monologue reasoning steps to generate synthetic training data. For each pair, the system may explore multiple retrieval paths, which are labeled as successful if they reach the correct answer and unsuccessful otherwise. Using this data, we train a lightweight information sufficiency Critic. At inference time, the Critic evaluates whether the RAG system has retrieved sufficient information at each round, guiding retrieval decisions and improving system-level self-awareness through in-context reinforcement learning.
Experiments across multiple prominent RAG benchmarks show that SIM-RAG is an effective multi-round RAG solution. Furthermore, this framework is system-efficient, adding a lightweight component to RAG without requiring modifications to existing LLMs or search engines, and data-efficient, eliminating the need for costly human-annotated mid-step retrieval process supervision data. 

---
# Using Knowledge Graphs to harvest datasets for efficient CLIP model training 

**Authors**: Simon Ging, Sebastian Walter, Jelena Bratulić, Johannes Dienert, Hannah Bast, Thomas Brox  

**Link**: [PDF](https://arxiv.org/pdf/2505.02746)  

**Abstract**: Training high-quality CLIP models typically requires enormous datasets, which limits the development of domain-specific models -- especially in areas that even the largest CLIP models do not cover well -- and drives up training costs. This poses challenges for scientific research that needs fine-grained control over the training procedure of CLIP models. In this work, we show that by employing smart web search strategies enhanced with knowledge graphs, a robust CLIP model can be trained from scratch with considerably less data. Specifically, we demonstrate that an expert foundation model for living organisms can be built using just 10M images. Moreover, we introduce EntityNet, a dataset comprising 33M images paired with 46M text descriptions, which enables the training of a generic CLIP model in significantly reduced time. 

---
# Voila: Voice-Language Foundation Models for Real-Time Autonomous Interaction and Voice Role-Play 

**Authors**: Yemin Shi, Yu Shu, Siwei Dong, Guangyi Liu, Jaward Sesay, Jingwen Li, Zhiting Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.02707)  

**Abstract**: A voice AI agent that blends seamlessly into daily life would interact with humans in an autonomous, real-time, and emotionally expressive manner. Rather than merely reacting to commands, it would continuously listen, reason, and respond proactively, fostering fluid, dynamic, and emotionally resonant interactions. We introduce Voila, a family of large voice-language foundation models that make a step towards this vision. Voila moves beyond traditional pipeline systems by adopting a new end-to-end architecture that enables full-duplex, low-latency conversations while preserving rich vocal nuances such as tone, rhythm, and emotion. It achieves a response latency of just 195 milliseconds, surpassing the average human response time. Its hierarchical multi-scale Transformer integrates the reasoning capabilities of large language models (LLMs) with powerful acoustic modeling, enabling natural, persona-aware voice generation -- where users can simply write text instructions to define the speaker's identity, tone, and other characteristics. Moreover, Voila supports over one million pre-built voices and efficient customization of new ones from brief audio samples as short as 10 seconds. Beyond spoken dialogue, Voila is designed as a unified model for a wide range of voice-based applications, including automatic speech recognition (ASR), Text-to-Speech (TTS), and, with minimal adaptation, multilingual speech translation. Voila is fully open-sourced to support open research and accelerate progress toward next-generation human-machine interactions. 

---
# Predicting Movie Hits Before They Happen with LLMs 

**Authors**: Shaghayegh Agah, Yejin Kim, Neeraj Sharma, Mayur Nankani, Kevin Foley, H. Howie Huang, Sardar Hamidian  

**Link**: [PDF](https://arxiv.org/pdf/2505.02693)  

**Abstract**: Addressing the cold-start issue in content recommendation remains a critical ongoing challenge. In this work, we focus on tackling the cold-start problem for movies on a large entertainment platform. Our primary goal is to forecast the popularity of cold-start movies using Large Language Models (LLMs) leveraging movie metadata. This method could be integrated into retrieval systems within the personalization pipeline or could be adopted as a tool for editorial teams to ensure fair promotion of potentially overlooked movies that may be missed by traditional or algorithmic solutions. Our study validates the effectiveness of this approach compared to established baselines and those we developed. 

---
# Enhancing Chemical Reaction and Retrosynthesis Prediction with Large Language Model and Dual-task Learning 

**Authors**: Xuan Lin, Qingrui Liu, Hongxin Xiang, Daojian Zeng, Xiangxiang Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2505.02639)  

**Abstract**: Chemical reaction and retrosynthesis prediction are fundamental tasks in drug discovery. Recently, large language models (LLMs) have shown potential in many domains. However, directly applying LLMs to these tasks faces two major challenges: (i) lacking a large-scale chemical synthesis-related instruction dataset; (ii) ignoring the close correlation between reaction and retrosynthesis prediction for the existing fine-tuning strategies. To address these challenges, we propose ChemDual, a novel LLM framework for accurate chemical synthesis. Specifically, considering the high cost of data acquisition for reaction and retrosynthesis, ChemDual regards the reaction-and-retrosynthesis of molecules as a related recombination-and-fragmentation process and constructs a large-scale of 4.4 million instruction dataset. Furthermore, ChemDual introduces an enhanced LLaMA, equipped with a multi-scale tokenizer and dual-task learning strategy, to jointly optimize the process of recombination and fragmentation as well as the tasks between reaction and retrosynthesis prediction. Extensive experiments on Mol-Instruction and USPTO-50K datasets demonstrate that ChemDual achieves state-of-the-art performance in both predictions of reaction and retrosynthesis, outperforming the existing conventional single-task approaches and the general open-source LLMs. Through molecular docking analysis, ChemDual generates compounds with diverse and strong protein binding affinity, further highlighting its strong potential in drug design. 

---
# Bielik v3 Small: Technical Report 

**Authors**: Krzysztof Ociepa, Łukasz Flis, Remigiusz Kinas, Krzysztof Wróbel, Adrian Gwoździej  

**Link**: [PDF](https://arxiv.org/pdf/2505.02550)  

**Abstract**: We introduce Bielik v3, a series of parameter-efficient generative text models (1.5B and 4.5B) optimized for Polish language processing. These models demonstrate that smaller, well-optimized architectures can achieve performance comparable to much larger counterparts while requiring substantially fewer computational resources. Our approach incorporates several key innovations: a custom Polish tokenizer (APT4) that significantly improves token efficiency, Weighted Instruction Cross-Entropy Loss to balance learning across instruction types, and Adaptive Learning Rate that dynamically adjusts based on training progress. Trained on a meticulously curated corpus of 292 billion tokens spanning 303 million documents, these models excel across multiple benchmarks, including the Open PL LLM Leaderboard, Complex Polish Text Understanding Benchmark, Polish EQ-Bench, and Polish Medical Leaderboard. The 4.5B parameter model achieves results competitive with models 2-3 times its size, while the 1.5B model delivers strong performance despite its extremely compact profile. These advances establish new benchmarks for parameter-efficient language modeling in less-represented languages, making high-quality Polish language AI more accessible for resource-constrained applications. 

---
# Incentivizing Inclusive Contributions in Model Sharing Markets 

**Authors**: Enpei Zhang, Jingyi Chai, Rui Ye, Yanfeng Wang, Siheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.02462)  

**Abstract**: While data plays a crucial role in training contemporary AI models, it is acknowledged that valuable public data will be exhausted in a few years, directing the world's attention towards the massive decentralized private data. However, the privacy-sensitive nature of raw data and lack of incentive mechanism prevent these valuable data from being fully exploited. Addressing these challenges, this paper proposes inclusive and incentivized personalized federated learning (iPFL), which incentivizes data holders with diverse purposes to collaboratively train personalized models without revealing raw data. iPFL constructs a model-sharing market by solving a graph-based training optimization and incorporates an incentive mechanism based on game theory principles. Theoretical analysis shows that iPFL adheres to two key incentive properties: individual rationality and truthfulness. Empirical studies on eleven AI tasks (e.g., large language models' instruction-following tasks) demonstrate that iPFL consistently achieves the highest economic utility, and better or comparable model performance compared to baseline methods. We anticipate that our iPFL can serve as a valuable technique for boosting future AI models on decentralized private data while making everyone satisfied. 

---
# Optimizing Chain-of-Thought Reasoners via Gradient Variance Minimization in Rejection Sampling and RL 

**Authors**: Jiarui Yao, Yifan Hao, Hanning Zhang, Hanze Dong, Wei Xiong, Nan Jiang, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.02391)  

**Abstract**: Chain-of-thought (CoT) reasoning in large language models (LLMs) can be formalized as a latent variable problem, where the model needs to generate intermediate reasoning steps. While prior approaches such as iterative reward-ranked fine-tuning (RAFT) have relied on such formulations, they typically apply uniform inference budgets across prompts, which fails to account for variability in difficulty and convergence behavior. This work identifies the main bottleneck in CoT training as inefficient stochastic gradient estimation due to static sampling strategies. We propose GVM-RAFT, a prompt-specific Dynamic Sample Allocation Strategy designed to minimize stochastic gradient variance under a computational budget constraint. The method dynamically allocates computational resources by monitoring prompt acceptance rates and stochastic gradient norms, ensuring that the resulting gradient variance is minimized. Our theoretical analysis shows that the proposed dynamic sampling strategy leads to accelerated convergence guarantees under suitable conditions. Experiments on mathematical reasoning show that GVM-RAFT achieves a 2-4x speedup and considerable accuracy improvements over vanilla RAFT. The proposed dynamic sampling strategy is general and can be incorporated into other reinforcement learning algorithms, such as GRPO, leading to similar improvements in convergence and test accuracy. Our code is available at this https URL. 

---
# Optimizing LLMs for Resource-Constrained Environments: A Survey of Model Compression Techniques 

**Authors**: Sanjay Surendranath Girija, Shashank Kapoor, Lakshit Arora, Dipen Pradhan, Aman Raj, Ankit Shetgaonkar  

**Link**: [PDF](https://arxiv.org/pdf/2505.02309)  

**Abstract**: Large Language Models (LLMs) have revolutionized many areas of artificial intelligence (AI), but their substantial resource requirements limit their deployment on mobile and edge devices. This survey paper provides a comprehensive overview of techniques for compressing LLMs to enable efficient inference in resource-constrained environments. We examine three primary approaches: Knowledge Distillation, Model Quantization, and Model Pruning. For each technique, we discuss the underlying principles, present different variants, and provide examples of successful applications. We also briefly discuss complementary techniques such as mixture-of-experts and early-exit strategies. Finally, we highlight promising future directions, aiming to provide a valuable resource for both researchers and practitioners seeking to optimize LLMs for edge deployment. 

---
# Interpretable Emergent Language Using Inter-Agent Transformers 

**Authors**: Mannan Bhardwaj  

**Link**: [PDF](https://arxiv.org/pdf/2505.02215)  

**Abstract**: This paper explores the emergence of language in multi-agent reinforcement learning (MARL) using transformers. Existing methods such as RIAL, DIAL, and CommNet enable agent communication but lack interpretability. We propose Differentiable Inter-Agent Transformers (DIAT), which leverage self-attention to learn symbolic, human-understandable communication protocols. Through experiments, DIAT demonstrates the ability to encode observations into interpretable vocabularies and meaningful embeddings, effectively solving cooperative tasks. These results highlight the potential of DIAT for interpretable communication in complex multi-agent environments. 

---
# DNAZEN: Enhanced Gene Sequence Representations via Mixed Granularities of Coding Units 

**Authors**: Lei Mao, Yuanhe Tian, Yan Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.02206)  

**Abstract**: Genome modeling conventionally treats gene sequence as a language, reflecting its structured motifs and long-range dependencies analogous to linguistic units and organization principles such as words and syntax. Recent studies utilize advanced neural networks, ranging from convolutional and recurrent models to Transformer-based models, to capture contextual information of gene sequence, with the primary goal of obtaining effective gene sequence representations and thus enhance the models' understanding of various running gene samples. However, these approaches often directly apply language modeling techniques to gene sequences and do not fully consider the intrinsic information organization in them, where they do not consider how units at different granularities contribute to representation. In this paper, we propose DNAZEN, an enhanced genomic representation framework designed to learn from various granularities in gene sequences, including small polymers and G-grams that are combinations of several contiguous polymers. Specifically, we extract the G-grams from large-scale genomic corpora through an unsupervised approach to construct the G-gram vocabulary, which is used to provide G-grams in the learning process of DNA sequences through dynamically matching from running gene samples. A Transformer-based G-gram encoder is also proposed and the matched G-grams are fed into it to compute their representations and integrated into the encoder for basic unit (E4BU), which is responsible for encoding small units and maintaining the learning and inference process. To further enhance the learning process, we propose whole G-gram masking to train DNAZEN, where the model largely favors the selection of each entire G-gram to mask rather than an ordinary masking mechanism performed on basic units. Experiments on benchmark datasets demonstrate the effectiveness of DNAZEN on various downstream tasks. 

---
# Exploring new Approaches for Information Retrieval through Natural Language Processing 

**Authors**: Manak Raj, Nidhi Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2505.02199)  

**Abstract**: This review paper explores recent advancements and emerging approaches in Information Retrieval (IR) applied to Natural Language Processing (NLP). We examine traditional IR models such as Boolean, vector space, probabilistic, and inference network models, and highlight modern techniques including deep learning, reinforcement learning, and pretrained transformer models like BERT. We discuss key tools and libraries - Lucene, Anserini, and Pyserini - for efficient text indexing and search. A comparative analysis of sparse, dense, and hybrid retrieval methods is presented, along with applications in web search engines, cross-language IR, argument mining, private information retrieval, and hate speech detection. Finally, we identify open challenges and future research directions to enhance retrieval accuracy, scalability, and ethical considerations. 

---
# Attention Mechanisms Perspective: Exploring LLM Processing of Graph-Structured Data 

**Authors**: Zhong Guan, Likang Wu, Hongke Zhao, Ming He, Jianpin Fan  

**Link**: [PDF](https://arxiv.org/pdf/2505.02130)  

**Abstract**: Attention mechanisms are critical to the success of large language models (LLMs), driving significant advancements in multiple fields. However, for graph-structured data, which requires emphasis on topological connections, they fall short compared to message-passing mechanisms on fixed links, such as those employed by Graph Neural Networks (GNNs). This raises a question: ``Does attention fail for graphs in natural language settings?'' Motivated by these observations, we embarked on an empirical study from the perspective of attention mechanisms to explore how LLMs process graph-structured data. The goal is to gain deeper insights into the attention behavior of LLMs over graph structures. We uncovered unique phenomena regarding how LLMs apply attention to graph-structured data and analyzed these findings to improve the modeling of such data by LLMs. The primary findings of our research are: 1) While LLMs can recognize graph data and capture text-node interactions, they struggle to model inter-node relationships within graph structures due to inherent architectural constraints. 2) The attention distribution of LLMs across graph nodes does not align with ideal structural patterns, indicating a failure to adapt to graph topology nuances. 3) Neither fully connected attention nor fixed connectivity is optimal; each has specific limitations in its application scenarios. Instead, intermediate-state attention windows improve LLM training performance and seamlessly transition to fully connected windows during inference. Source code: \href{this https URL}{LLM4Exploration} 

---
# A Comprehensive Analysis for Visual Object Hallucination in Large Vision-Language Models 

**Authors**: Liqiang Jing, Guiming Hardy Chen, Ehsan Aghazadeh, Xin Eric Wang, Xinya Du  

**Link**: [PDF](https://arxiv.org/pdf/2505.01958)  

**Abstract**: Large Vision-Language Models (LVLMs) demonstrate remarkable capabilities in multimodal tasks, but visual object hallucination remains a persistent issue. It refers to scenarios where models generate inaccurate visual object-related information based on the query input, potentially leading to misinformation and concerns about safety and reliability. Previous works focus on the evaluation and mitigation of visual hallucinations, but the underlying causes have not been comprehensively investigated. In this paper, we analyze each component of LLaVA-like LVLMs -- the large language model, the vision backbone, and the projector -- to identify potential sources of error and their impact. Based on our observations, we propose methods to mitigate hallucination for each problematic component. Additionally, we developed two hallucination benchmarks: QA-VisualGenome, which emphasizes attribute and relation hallucinations, and QA-FB15k, which focuses on cognition-based hallucinations. 

---
# Explainability by design: an experimental analysis of the legal coding process 

**Authors**: Matteo Cristani, Guido Governatori, Francesco Olivieri, Monica Palmirani, Gabriele Buriola  

**Link**: [PDF](https://arxiv.org/pdf/2505.01944)  

**Abstract**: Behind a set of rules in Deontic Defeasible Logic, there is a mapping process of normative background fragments. This process goes from text to rules and implicitly encompasses an explanation of the coded fragments.
In this paper we deliver a methodology for \textit{legal coding} that starts with a fragment and goes onto a set of Deontic Defeasible Logic rules, involving a set of \textit{scenarios} to test the correctness of the coded fragments. The methodology is illustrated by the coding process of an example text. We then show the results of a series of experiments conducted with humans encoding a variety of normative backgrounds and corresponding cases in which we have measured the efforts made in the coding process, as related to some measurable features. To process these examples, a recently developed technology, Houdini, that allows reasoning in Deontic Defeasible Logic, has been employed.
Finally we provide a technique to forecast time required in coding, that depends on factors such as knowledge of the legal domain, knowledge of the coding processes, length of the text, and a measure of \textit{depth} that refers to the length of the paths of legal references. 

---
# Enhancing the Learning Experience: Using Vision-Language Models to Generate Questions for Educational Videos 

**Authors**: Markos Stamatakis, Joshua Berger, Christian Wartena, Ralph Ewerth, Anett Hoppe  

**Link**: [PDF](https://arxiv.org/pdf/2505.01790)  

**Abstract**: Web-based educational videos offer flexible learning opportunities and are becoming increasingly popular. However, improving user engagement and knowledge retention remains a challenge. Automatically generated questions can activate learners and support their knowledge acquisition. Further, they can help teachers and learners assess their understanding. While large language and vision-language models have been employed in various tasks, their application to question generation for educational videos remains underexplored. In this paper, we investigate the capabilities of current vision-language models for generating learning-oriented questions for educational video content. We assess (1) out-of-the-box models' performance; (2) fine-tuning effects on content-specific question generation; (3) the impact of different video modalities on question quality; and (4) in a qualitative study, question relevance, answerability, and difficulty levels of generated questions. Our findings delineate the capabilities of current vision-language models, highlighting the need for fine-tuning and addressing challenges in question diversity and relevance. We identify requirements for future multimodal datasets and outline promising research directions. 

---
# Unraveling Media Perspectives: A Comprehensive Methodology Combining Large Language Models, Topic Modeling, Sentiment Analysis, and Ontology Learning to Analyse Media Bias 

**Authors**: Orlando Jähde, Thorsten Weber, Rüdiger Buchkremer  

**Link**: [PDF](https://arxiv.org/pdf/2505.01754)  

**Abstract**: Biased news reporting poses a significant threat to informed decision-making and the functioning of democracies. This study introduces a novel methodology for scalable, minimally biased analysis of media bias in political news. The proposed approach examines event selection, labeling, word choice, and commission and omission biases across news sources by leveraging natural language processing techniques, including hierarchical topic modeling, sentiment analysis, and ontology learning with large language models. Through three case studies related to current political events, we demonstrate the methodology's effectiveness in identifying biases across news sources at various levels of granularity. This work represents a significant step towards scalable, minimally biased media bias analysis, laying the groundwork for tools to help news consumers navigate an increasingly complex media landscape. 

---
# Inducing Robustness in a 2 Dimensional Direct Preference Optimization Paradigm 

**Authors**: Sarvesh Shashidhar, Ritik, Nachiketa Patil, Suraj Racha, Ganesh Ramakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2505.01706)  

**Abstract**: Direct Preference Optimisation (DPO) has emerged as a powerful method for aligning Large Language Models (LLMs) with human preferences, offering a stable and efficient alternative to approaches that use Reinforcement learning via Human Feedback. In this work, we investigate the performance of DPO using open-source preference datasets. One of the major drawbacks of DPO is that it doesn't induce granular scoring and treats all the segments of the responses with equal propensity. However, this is not practically true for human preferences since even "good" responses have segments that may not be preferred by the annotator. To resolve this, a 2-dimensional scoring for DPO alignment called 2D-DPO was proposed. We explore the 2D-DPO alignment paradigm and the advantages it provides over the standard DPO by comparing their win rates. It is observed that these methods, even though effective, are not robust to label/score noise. To counter this, we propose an approach of incorporating segment-level score noise robustness to the 2D-DPO algorithm. Along with theoretical backing, we also provide empirical verification in favour of the algorithm and introduce other noise models that can be present. 

---
# Structured Prompting and Feedback-Guided Reasoning with LLMs for Data Interpretation 

**Authors**: Amit Rath  

**Link**: [PDF](https://arxiv.org/pdf/2505.01636)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities in natural language understanding and task generalization. However, their application to structured data analysis remains fragile due to inconsistencies in schema interpretation, misalignment between user intent and model output, and limited mechanisms for self-correction when failures occur. This paper introduces the STROT Framework (Structured Task Reasoning and Output Transformation), a method for structured prompting and feedback-driven transformation logic generation aimed at improving the reliability and semantic alignment of LLM-based analytical workflows. STROT begins with lightweight schema introspection and sample-based field classification, enabling dynamic context construction that captures both the structure and statistical profile of the input data. This contextual information is embedded in structured prompts that guide the model toward generating task-specific, interpretable outputs. To address common failure modes in complex queries, STROT incorporates a refinement mechanism in which the model iteratively revises its outputs based on execution feedback and validation signals. Unlike conventional approaches that rely on static prompts or single-shot inference, STROT treats the LLM as a reasoning agent embedded within a controlled analysis loop -- capable of adjusting its output trajectory through planning and correction. The result is a robust and reproducible framework for reasoning over structured data with LLMs, applicable to diverse data exploration and analysis tasks where interpretability, stability, and correctness are essential. 

---
# CHORUS: Zero-shot Hierarchical Retrieval and Orchestration for Generating Linear Programming Code 

**Authors**: Tasnim Ahmed, Salimur Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2505.01485)  

**Abstract**: Linear Programming (LP) problems aim to find the optimal solution to an objective under constraints. These problems typically require domain knowledge, mathematical skills, and programming ability, presenting significant challenges for non-experts. This study explores the efficiency of Large Language Models (LLMs) in generating solver-specific LP code. We propose CHORUS, a retrieval-augmented generation (RAG) framework for synthesizing Gurobi-based LP code from natural language problem statements. CHORUS incorporates a hierarchical tree-like chunking strategy for theoretical contents and generates additional metadata based on code examples from documentation to facilitate self-contained, semantically coherent retrieval. Two-stage retrieval approach of CHORUS followed by cross-encoder reranking further ensures contextual relevance. Finally, expertly crafted prompt and structured parser with reasoning steps improve code generation performance significantly. Experiments on the NL4Opt-Code benchmark show that CHORUS improves the performance of open-source LLMs such as Llama3.1 (8B), Llama3.3 (70B), Phi4 (14B), Deepseek-r1 (32B), and Qwen2.5-coder (32B) by a significant margin compared to baseline and conventional RAG. It also allows these open-source LLMs to outperform or match the performance of much stronger baselines-GPT3.5 and GPT4 while requiring far fewer computational resources. Ablation studies further demonstrate the importance of expert prompting, hierarchical chunking, and structured reasoning. 

---
# AdaParse: An Adaptive Parallel PDF Parsing and Resource Scaling Engine 

**Authors**: Carlo Siebenschuh, Kyle Hippe, Ozan Gokdemir, Alexander Brace, Arham Khan, Khalid Hossain, Yadu Babuji, Nicholas Chia, Venkatram Vishwanath, Rick Stevens, Arvind Ramanathan, Ian Foster, Robert Underwood  

**Link**: [PDF](https://arxiv.org/pdf/2505.01435)  

**Abstract**: Language models for scientific tasks are trained on text from scientific publications, most distributed as PDFs that require parsing. PDF parsing approaches range from inexpensive heuristics (for simple documents) to computationally intensive ML-driven systems (for complex or degraded ones). The choice of the "best" parser for a particular document depends on its computational cost and the accuracy of its output. To address these issues, we introduce an Adaptive Parallel PDF Parsing and Resource Scaling Engine (AdaParse), a data-driven strategy for assigning an appropriate parser to each document. We enlist scientists to select preferred parser outputs and incorporate this information through direct preference optimization (DPO) into AdaParse, thereby aligning its selection process with human judgment. AdaParse then incorporates hardware requirements and predicted accuracy of each parser to orchestrate computational resources efficiently for large-scale parsing campaigns. We demonstrate that AdaParse, when compared to state-of-the-art parsers, improves throughput by $17\times$ while still achieving comparable accuracy (0.2 percent better) on a benchmark set of 1000 scientific documents. AdaParse's combination of high accuracy and parallel scalability makes it feasible to parse large-scale scientific document corpora to support the development of high-quality, trillion-token-scale text datasets. The implementation is available at this https URL 

---
# Enhancing TCR-Peptide Interaction Prediction with Pretrained Language Models and Molecular Representations 

**Authors**: Cong Qi, Hanzhang Fang, Siqi jiang, Tianxing Hu, Wei Zhi  

**Link**: [PDF](https://arxiv.org/pdf/2505.01433)  

**Abstract**: Understanding the binding specificity between T-cell receptors (TCRs) and peptide-major histocompatibility complexes (pMHCs) is central to immunotherapy and vaccine development. However, current predictive models struggle with generalization, especially in data-scarce settings and when faced with novel epitopes. We present LANTERN (Large lAnguage model-powered TCR-Enhanced Recognition Network), a deep learning framework that combines large-scale protein language models with chemical representations of peptides. By encoding TCR \b{eta}-chain sequences using ESM-1b and transforming peptide sequences into SMILES strings processed by MolFormer, LANTERN captures rich biological and chemical features critical for TCR-peptide recognition. Through extensive benchmarking against existing models such as ChemBERTa, TITAN, and NetTCR, LANTERN demonstrates superior performance, particularly in zero-shot and few-shot learning scenarios. Our model also benefits from a robust negative sampling strategy and shows significant clustering improvements via embedding analysis. These results highlight the potential of LANTERN to advance TCR-pMHC binding prediction and support the development of personalized immunotherapies. 

---
